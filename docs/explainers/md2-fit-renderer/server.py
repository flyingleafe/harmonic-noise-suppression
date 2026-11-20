#!/usr/bin/env python
"""Live local backend of the MD2 Fig. 6(a) fit renderer (``docs/explainers/md2-fit-renderer.qmd``).

It renders REAL waveforms of the MD2 FLY125 standby/cruise fit pair, exactly as
``scripts/gen_model_figures.py`` ``fig_noise_fits`` draws panel (a): the clip
``supports.flight_michaels("FLY125", MICHAELS_RAMP_STARTS[0], 8.0)``,
microphone 3, ``render_noise_regimes(fits, rps, sr=16000, n_mics=4, seed=7)``
(no augmentation, no normalisation, the 45-65 rev/s smoothstep blend), and the
periodic-Hann NFFT 2048 / hop 512 periodogram ``|rfft(x w)|^2 / sum w^2`` in dB.
The real clip and the baseline render of the source payloads
(``experiments.noise_model.sectors_v4.FITS["md2"][0]``) are computed once at
start-up and never change; an edit renders a third, EDITED waveform.

EDITABLE PARAMETERS, per regime: the payload's ``params`` with
``shaft = {sigma_ref_rad_s, a_sigma, lam_ref_per_s, b_lam, f_ref_rev_s}``
(``eps = sigma_ref / (2 pi f_ref)``; the legacy ``sigma_nu_rad_s`` follows
``sigma_ref``; ``f_ref_rev_s`` is read-only, the source's 80 rev/s reference
that the speed law shares) and ``pedestal = {sigma2_amp, gamma_amp_hz, sigma2_phase,
width_phase_hz}`` (``gamma_phase_hz = W (1 - e^-v) / v``, ``W`` at ``v = 0``,
:func:`experiments.noise_model.prior_v4.phase_gamma_hz`). Derived values and
metadata the renderer does not read (:data:`NOT_EDITABLE`) are not editable;
the exported payload carries them from the source fit (``floor.ctrl_db``
recomputed when the floor changes). An unchanged value keeps the source fit's
exact derived number, so rendering the unedited parameters reproduces the
baseline sample for sample. No fitting cap is enforced.

HTTP (``python docs/explainers/md2-fit-renderer/server.py --port 8766``, bound
to 127.0.0.1 only; the Host header must name that address, POSTs must be
same-origin JSON of at most :data:`MAX_BODY_BYTES`):

* ``GET /`` redirects to ``/md2-fit-renderer.html``; other GETs serve generated
  web assets under ``docs/explainers`` (no sources, no dotfiles);
  ``/app.js`` and ``/style.css`` are the frontend's files, ``/plotly.min.js``
  the installed plotly package's bundle.
* ``GET /api/state``: ``{example, parameters, real, baseline, audio,
  playback_gain}``.
* ``POST /api/render`` ``{shaft_law_evaluation: "local_work_grid",
  parameters: {standby, cruise}}``: ``{edited, audio, playback_gain, parameters,
  elapsed_s, export_fits}``. One render at a time
  (409 while busy); nothing but the immutable baseline is kept between calls.

A signal is ``{spectrogram, waveform_sha256, rms}``: the SHA-256 of the raw
little-endian float64 samples before any playback gain, their RMS in the
render's absolute units, and ``spectrogram = {shape: [n_bins, n_frames],
db_f32_base64, times_s, freqs_hz}`` (frequency-major little-endian float32 dB,
frame centres, all bins to 8 kHz). The three audio data URLs are 16-bit WAVs
sharing ONE playback gain (the loudest of the three peaks at -1 dBFS). Errors
are ``{"error": message}`` with a 4xx/5xx status.
"""

from __future__ import annotations

import argparse
import base64
import copy
import functools
import hashlib
import io
import json
import math
import sys
import threading
import time
import traceback
import wave
from contextlib import suppress
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import numpy as np

HERE = Path(__file__).resolve().parent
EXPLAINERS = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data_processing.noise_model import v4 as V4  # noqa: E402
from data_processing.noise_model.render import render_noise_regimes  # noqa: E402
from experiments.noise_model import supports as S  # noqa: E402
from experiments.noise_model.prior_v4 import phase_gamma_hz  # noqa: E402
from experiments.noise_model.sectors_v4 import FITS  # noqa: E402

# ── the figure's example (gen_model_figures.fig_noise_fits, MD2 row) ─────────
RECORDING = "FLY125"
START_S = float(S.MICHAELS_RAMP_STARTS[0])
DURATION_S = 8.0
MIC = 3
#: The figure helper renders ``NOISESAMP_MIC + 1`` channels; the channel count
#: changes the draws, so it is part of the example's identity.
N_MICS_RENDER = MIC + 1
SR = 16000
SEED = 7
NFFT = 2048
HOP = 512
VMAX_FMAX_HZ = 3000.0
VMAX_PERCENTILE = 99.5
DYNAMIC_RANGE_DB = 50.0
REGIMES = ("standby", "cruise")
SHAFT_LAW_EVALUATION = "local_work_grid"
FREQS_HZ = np.linspace(0.0, SR / 2, NFFT // 2 + 1)

# ── editing ─────────────────────────────────────────────────────────────────
SHAFT_KEYS = ("sigma_ref_rad_s", "a_sigma", "lam_ref_per_s", "b_lam", "f_ref_rev_s")
PEDESTAL_KEYS = ("sigma2_amp", "gamma_amp_hz", "sigma2_phase", "width_phase_hz")
#: ``params`` entries left out of the edit state: values derived from the
#: canonical ones (``eps``, ``sigma_nu_rad_s``, ``gamma_phase_hz``,
#: ``ctrl_db``) and metadata the renderer does not read. Exports carry them.
NOT_EDITABLE: dict[tuple[str, ...], tuple[str, ...]] = {
    ("shaft",): ("eps", "sigma_nu_rad_s", "pinned", "sigma_nu_note"),
    ("pedestal",): ("gamma_phase_hz", "held", "source"),
    ("floor",): ("ctrl_db", "basis", "sr_work", "b_f_lo_hz", "b_f_ref_hz"),
    ("mic_response",): ("geometry_rule",),
    ("mic_response", "array_response"): ("source", "rig", "rule"),
}

# ── HTTP ────────────────────────────────────────────────────────────────────
HOST = "127.0.0.1"
PAGE = "/md2-fit-renderer.html"
MAX_BODY_BYTES = 2 * 1024 * 1024
#: Playback peak of the loudest of real / baseline / edited: -1 dBFS.
PEAK_TARGET = 10.0 ** (-1.0 / 20.0)
EXPLAINERS_ROOT = EXPLAINERS.resolve()
FRONTEND_FILES = {"/app.js": HERE / "app.js", "/style.css": HERE / "style.css"}
STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".map": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".otf": "font/otf",
    ".eot": "application/vnd.ms-fontobject",
}


class ApiError(Exception):
    """A request the server answers with ``status`` and ``{"error": message}``."""

    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


# ── signals ─────────────────────────────────────────────────────────────────


def spec_db(audio: np.ndarray) -> np.ndarray:
    """``(frames, bins)`` periodogram in dB, ``gen_model_figures._spec_db`` verbatim."""
    w = np.hanning(NFFT + 1)[:NFFT]
    n = (len(audio) - NFFT) // HOP + 1
    frames = np.stack([audio[i * HOP : i * HOP + NFFT] * w for i in range(n)])
    p = np.abs(np.fft.rfft(frames, axis=-1)) ** 2 / (w**2).sum()
    return 10.0 * np.log10(np.maximum(p, 1e-30))


def signal_summary(audio: np.ndarray, db: np.ndarray) -> dict[str, Any]:
    """The API's signal: spectrogram, raw-waveform hash and RMS (absolute units)."""
    n_frames, n_bins = db.shape
    times = (np.arange(n_frames) * HOP + NFFT / 2) / SR
    matrix = np.ascontiguousarray(db.T, dtype="<f4")
    return {
        "spectrogram": {
            "shape": [n_bins, n_frames],
            "db_f32_base64": base64.b64encode(matrix.tobytes()).decode("ascii"),
            "times_s": times.tolist(),
            "freqs_hz": FREQS_HZ.tolist(),
        },
        "waveform_sha256": hashlib.sha256(
            np.ascontiguousarray(audio, dtype="<f8").tobytes()
        ).hexdigest(),
        "rms": float(np.sqrt(np.mean(audio**2))),
    }


def wav_data_url(audio: np.ndarray, gain: float) -> str:
    """``audio * gain`` as a mono 16-bit WAV data URL."""
    pcm = np.clip(np.rint(audio * gain * 32767.0), -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SR)
        out.writeframes(pcm.tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def audio_block(
    real: np.ndarray, baseline: np.ndarray, edited: np.ndarray
) -> tuple[dict[str, str], float]:
    """The three WAVs at ONE shared gain that puts the loudest peak at -1 dBFS."""
    peak = max(float(np.max(np.abs(a))) for a in (real, baseline, edited))
    if not (peak > 0.0 and math.isfinite(peak)):
        raise ApiError(
            HTTPStatus.UNPROCESSABLE_ENTITY, f"cannot set a playback gain for peak {peak}"
        )
    gain = PEAK_TARGET / peak
    audio = {
        "real": wav_data_url(real, gain),
        "baseline": wav_data_url(baseline, gain),
        "edited": wav_data_url(edited, gain),
    }
    return audio, gain


def render_mic(fits: dict[str, dict[str, Any]], rps: np.ndarray) -> np.ndarray:
    """Microphone :data:`MIC` of the figure helper's render of ``fits``."""
    audio = render_noise_regimes(fits, rps, sr=SR, n_mics=N_MICS_RENDER, seed=SEED)
    return np.ascontiguousarray(np.asarray(audio, dtype=np.float64)[MIC])


def dumps(obj: Any) -> bytes:
    return json.dumps(obj, allow_nan=False, separators=(",", ":")).encode("utf-8")


# ── parameters ──────────────────────────────────────────────────────────────


def canonical_params(fit: dict[str, Any]) -> dict[str, Any]:
    """The edit state of one payload (module docstring), a fresh copy."""
    p = copy.deepcopy(fit["params"])
    shaft = p["shaft"]
    f_ref = float(shaft.get("f_ref_rev_s", fit.get("f_ref_rev_s", 80.0)))
    sigma_ref = 2.0 * math.pi * f_ref * float(shaft["eps"])
    for path, keys in NOT_EDITABLE.items():
        node: Any = p
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, dict):
            for key in keys:
                node.pop(key, None)
    p["shaft"] = {
        "sigma_ref_rad_s": sigma_ref,
        "a_sigma": float(shaft["a_sigma"]),
        "lam_ref_per_s": float(shaft["lam_ref_per_s"]),
        "b_lam": float(shaft["b_lam"]),
        "f_ref_rev_s": f_ref,
    }
    p["pedestal"] = {k: p["pedestal"][k] for k in PEDESTAL_KEYS}
    return p


def conform(value: Any, template: Any, path: str) -> Any:
    """``value`` checked against the source fit's edit state ``template``: the
    same keys, the same array shapes, finite numbers (integers where the source
    has integers), strings and nulls where it has them. Returns a fresh copy."""
    if isinstance(template, dict):
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object")
        missing, extra = sorted(set(template) - set(value)), sorted(set(value) - set(template))
        if missing or extra:
            raise ValueError(
                f"{path}: missing keys {missing}, unknown keys {extra} "
                "(edit the structure /api/state returns)"
            )
        return {k: conform(value[k], t, f"{path}.{k}") for k, t in template.items()}
    if isinstance(template, list):
        if not isinstance(value, list) or len(value) != len(template):
            raise ValueError(
                f"{path} must keep the source shape {list(np.shape(template))} "
                f"({len(template)} entries here)"
            )
        return [conform(v, t, f"{path}[{i}]") for i, (v, t) in enumerate(zip(value, template))]
    if template is None:
        if value is not None:
            raise ValueError(f"{path} must stay null (the source fit has none)")
        return None
    if isinstance(template, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{path} must be true or false, got {value!r}")
        return value
    if isinstance(template, (int, float)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path} must be a number, got {value!r}")
        try:
            x = float(value)
        except OverflowError:
            raise ValueError(f"{path} is out of floating-point range") from None
        if not math.isfinite(x):
            raise ValueError(f"{path} must be finite, got {value!r}")
        if isinstance(template, int):
            if x != math.floor(x):
                raise ValueError(f"{path} must be an integer, got {value!r}")
            return int(x)
        return x
    if isinstance(template, str):
        if not isinstance(value, str):
            raise ValueError(f"{path} must be a string, got {value!r}")
        return value
    raise TypeError(f"{path}: unsupported template type {type(template).__name__}")


def check_domain(c: dict[str, Any], base: dict[str, Any], regime: str) -> None:
    """The model's domain (no fitting caps): non-negative variances, rates,
    widths and scales, positive shaft rate; the shaft's reference speed is the
    source fit's (read-only)."""
    shaft = c["shaft"]
    f_ref = base["shaft"]["f_ref_rev_s"]
    if shaft["f_ref_rev_s"] != f_ref:
        raise ValueError(
            f"{regime}.shaft.f_ref_rev_s is read-only ({f_ref:g} rev/s): it is the coordinate "
            "convention of sigma_ref_rad_s, not a fitted parameter; the speed law uses the "
            "payload's own reference, so changing it alone would cancel through eps"
        )
    if shaft["sigma_ref_rad_s"] < 0.0:
        raise ValueError(f"{regime}.shaft.sigma_ref_rad_s must be >= 0")
    if shaft["lam_ref_per_s"] <= 0.0:
        raise ValueError(f"{regime}.shaft.lam_ref_per_s must be > 0, got {shaft['lam_ref_per_s']}")
    for key in PEDESTAL_KEYS:
        bad = np.flatnonzero(np.asarray(c["pedestal"][key], dtype=np.float64) < 0.0)
        if bad.size:
            raise ValueError(
                f"{regime}.pedestal.{key} must be >= 0; negative at order(s) k = "
                f"{(bad[:10] + 1).tolist()}{' ...' if bad.size > 10 else ''}"
            )
    for key in ("shape_sd_db", "static_rel"):
        if c["floor"][key] < 0.0:
            raise ValueError(f"{regime}.floor.{key} must be >= 0, got {c['floor'][key]}")


def changed_paths(a: Any, b: Any, path: str) -> list[str]:
    """Dotted paths whose value differs between edit states ``a`` and ``b``."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [p for k in a for p in changed_paths(a[k], b[k], f"{path}.{k}" if path else k)]
    return [] if a == b else [path]


def merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            merge(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)


# ── the renderer ────────────────────────────────────────────────────────────


class Md2Renderer:
    """The immutable real/baseline state and the edited renders."""

    def __init__(self) -> None:
        paths = dict(FITS["md2"][0])
        if set(paths) != set(REGIMES):
            raise RuntimeError(f"FITS['md2'][0] must hold {REGIMES}, has {sorted(paths)}")
        self.paths = {r: paths[r] for r in REGIMES}
        raw = {r: (ROOT / self.paths[r]).read_bytes() for r in REGIMES}
        self.sha256 = {r: hashlib.sha256(raw[r]).hexdigest() for r in REGIMES}
        self.original = {r: json.loads(raw[r]) for r in REGIMES}
        self.base_params = {r: canonical_params(self.original[r]) for r in REGIMES}

        clip = S.flight_clip(S.flight_michaels(RECORDING, START_S, DURATION_S))
        rps = np.asarray(clip.rps, dtype=np.float64)
        n_out = int(round(DURATION_S * SR))
        if int(clip.sr) != SR or rps.shape != (4, n_out) or clip.audio.shape[1] != n_out:
            raise RuntimeError(
                f"clip is sr {clip.sr}, rps {rps.shape}, audio {clip.audio.shape}; "
                f"expected sr {SR}, rps (4, {n_out}) and {n_out} samples"
            )
        rps.setflags(write=False)
        self.rps = rps
        self.real = np.ascontiguousarray(np.asarray(clip.audio[MIC], dtype=np.float64))
        self.baseline = render_mic(self.original, rps)
        if not np.all(np.isfinite(self.baseline)):
            raise RuntimeError("the source fits render non-finite samples")
        for a in (self.real, self.baseline):
            a.setflags(write=False)

        real_db, base_db = spec_db(self.real), spec_db(self.baseline)
        keep = FREQS_HZ <= VMAX_FMAX_HZ
        self.vmax_db = float(
            np.percentile(
                np.concatenate([d[:, keep].ravel() for d in (real_db, base_db)]), VMAX_PERCENTILE
            )
        )
        self.real_signal = signal_summary(self.real, real_db)
        self.baseline_signal = signal_summary(self.baseline, base_db)
        audio, gain = audio_block(self.real, self.baseline, self.baseline)
        self.state_body = dumps(
            {
                "example": {
                    "recording": RECORDING,
                    "start_s": START_S,
                    "duration_s": DURATION_S,
                    "mic": MIC,
                    "sr": SR,
                    "seed": SEED,
                    "n_fft": NFFT,
                    "hop": HOP,
                    "n_mics_render": N_MICS_RENDER,
                    "shaft_law_evaluation": SHAFT_LAW_EVALUATION,
                    "source_fits": {
                        r: {"path": self.paths[r], "sha256": self.sha256[r]} for r in REGIMES
                    },
                    "vmax_db": self.vmax_db,
                    "dynamic_range_db": DYNAMIC_RANGE_DB,
                },
                "parameters": self.base_params,
                "real": self.real_signal,
                "baseline": self.baseline_signal,
                "audio": audio,
                "playback_gain": gain,
            }
        )
        self._lock = threading.Lock()

    def payload(self, regime: str, params: dict[str, Any]) -> dict[str, Any]:
        """The source payload of ``regime`` carrying the edit state ``params``,
        labelled as an interactive edit."""
        original, base = self.original[regime], self.base_params[regime]
        fit = copy.deepcopy(original)
        p = fit["params"]
        merge(p, params)

        shaft, src_shaft = p["shaft"], original["params"]["shaft"]
        sigma_ref = shaft.pop("sigma_ref_rad_s")
        unchanged = sigma_ref == base["shaft"]["sigma_ref_rad_s"]
        shaft["eps"] = (
            src_shaft["eps"] if unchanged else sigma_ref / (2.0 * math.pi * shaft["f_ref_rev_s"])
        )
        if "sigma_nu_rad_s" in src_shaft:
            shaft["sigma_nu_rad_s"] = src_shaft["sigma_nu_rad_s"] if unchanged else sigma_ref

        ped, src_ped = p["pedestal"], original["params"]["pedestal"]
        s2 = np.asarray(ped["sigma2_phase"], dtype=np.float64)
        width = np.asarray(ped["width_phase_hz"], dtype=np.float64)
        same = (s2 == np.asarray(src_ped["sigma2_phase"], dtype=np.float64)) & (
            width == np.asarray(src_ped["width_phase_hz"], dtype=np.float64)
        )
        gamma = np.where(
            same, np.asarray(src_ped["gamma_phase_hz"], dtype=np.float64), phase_gamma_hz(s2, width)
        )
        ped["gamma_phase_hz"] = gamma.tolist()

        if "ctrl_db" in p["floor"] and params["floor"] != base["floor"]:
            floor = V4.FloorLaw.from_mapping(p["floor"], float(fit["band_hz"][0]))
            p["floor"]["ctrl_db"] = (floor.mean_db + floor.ctrl_db()).tolist()

        fit["interactive_edit"] = {
            "status": "interactive edit of a fitted payload; NOT a fitted result",
            "tool": "docs/explainers/md2-fit-renderer/server.py",
            "regime": regime,
            "source_fit": {"path": self.paths[regime], "sha256": self.sha256[regime]},
            "changed_parameters": changed_paths(params, base, ""),
            "derived": (
                "shaft.eps = sigma_ref_rad_s / (2 pi f_ref_rev_s), shaft.sigma_nu_rad_s = "
                "sigma_ref_rad_s; pedestal.gamma_phase_hz = width_phase_hz (1 - e^-v) / v, "
                "v = sigma2_phase (width_phase_hz at v = 0); floor.ctrl_db = mean_db + the "
                "spline of shape_z; unchanged values keep the source fit's numbers"
            ),
            "note": (
                "hand-edited parameters; the objective, optimiser, convergence, priors, measured, "
                "pool, hierarchy and pedestal held/source blocks describe the SOURCE fit, not "
                "these parameters, and no fitting cap was enforced"
            ),
            "example": {
                "recording": RECORDING,
                "start_s": START_S,
                "duration_s": DURATION_S,
                "mic": MIC,
                "sr": SR,
                "seed": SEED,
                "n_mics_render": N_MICS_RENDER,
                "shaft_law_evaluation": SHAFT_LAW_EVALUATION,
            },
        }
        return fit

    def _validated(self, body: Any) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """``(edit states, payloads)`` of a render request, or a 400."""
        if not isinstance(body, dict) or body.get("shaft_law_evaluation") != SHAFT_LAW_EVALUATION:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "The shaft renderer has changed. Reload the page to regenerate its baseline "
                "before rendering edits.",
            )
        if set(body) != {"parameters", "shaft_law_evaluation"}:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "body must contain shaft_law_evaluation and parameters with standby/cruise",
            )
        given = body["parameters"]
        if not isinstance(given, dict) or set(given) != set(REGIMES):
            raise ApiError(
                HTTPStatus.BAD_REQUEST, f"parameters must hold exactly the regimes {list(REGIMES)}"
            )
        params: dict[str, Any] = {}
        fits: dict[str, dict[str, Any]] = {}
        for regime in REGIMES:
            try:
                params[regime] = conform(given[regime], self.base_params[regime], regime)
                check_domain(params[regime], self.base_params[regime], regime)
                fits[regime] = self.payload(regime, params[regime])
                par = V4.V4Params.from_payload(fits[regime])
                par.mic.db(np.array([100.0, 1000.0]))  # the array response's own checks
            except (ValueError, KeyError, TypeError) as exc:
                msg = str(exc) if isinstance(exc, ValueError) else f"{type(exc).__name__}: {exc}"
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    msg if msg.startswith(regime) else f"{regime}: {msg}",
                ) from None
        return params, fits

    def render(self, body: Any) -> bytes:
        if not self._lock.acquire(blocking=False):
            raise ApiError(
                HTTPStatus.CONFLICT, "a render is already running; wait for it to finish"
            )
        try:
            t0 = time.perf_counter()
            params, fits = self._validated(body)
            edited = render_mic(fits, self.rps)
            if not np.all(np.isfinite(edited)):
                raise ApiError(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    "the edited parameters render non-finite samples (a law overflows: "
                    "check exponents, levels and rates)",
                )
            edited_signal = signal_summary(edited, spec_db(edited))
            audio, gain = audio_block(self.real, self.baseline, edited)
            return dumps(
                {
                    "edited": edited_signal,
                    "audio": audio,
                    "playback_gain": gain,
                    "parameters": params,
                    "elapsed_s": time.perf_counter() - t0,
                    "export_fits": fits,
                }
            )
        finally:
            self._lock.release()


# ── HTTP ────────────────────────────────────────────────────────────────────


@functools.cache
def plotly_js() -> bytes:
    from plotly.offline import get_plotlyjs

    return get_plotlyjs().encode("utf-8")


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON number {token} is not accepted")


class RendererServer(ThreadingHTTPServer):
    renderer: Md2Renderer


class Handler(BaseHTTPRequestHandler):
    server: RendererServer
    server_version = "md2-fit-renderer/1"

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        try:
            origin = self._check_host()
            path = urlsplit(self.path).path
            if method == "POST":
                if path != "/api/render":
                    raise ApiError(HTTPStatus.NOT_FOUND, f"no POST route {path}")
                self._check_same_origin(origin)
                self._send(HTTPStatus.OK, self.server.renderer.render(self._read_json()))
            elif path == "/api/render":
                raise ApiError(HTTPStatus.METHOD_NOT_ALLOWED, "/api/render takes POST")
            elif path == "/api/state":
                self._send(HTTPStatus.OK, self.server.renderer.state_body)
            elif path == "/":
                self.send_response(HTTPStatus.FOUND)
                self.send_header("Location", PAGE)
                self.send_header("Content-Length", "0")
                self.end_headers()
            elif path == "/plotly.min.js":
                self._send(HTTPStatus.OK, plotly_js(), STATIC_TYPES[".js"])
            else:
                file = FRONTEND_FILES.get(path) or self._static_file(path)
                try:
                    data = file.read_bytes()
                except FileNotFoundError:
                    raise ApiError(HTTPStatus.NOT_FOUND, f"{path} does not exist") from None
                self._send(HTTPStatus.OK, data, STATIC_TYPES[file.suffix.lower()])
        except (BrokenPipeError, ConnectionResetError):
            return
        except ApiError as exc:
            self._send_error(exc.status, exc.message)
        except Exception as exc:  # an unexpected failure: reported, never hidden
            traceback.print_exc()
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def _check_host(self) -> str:
        """The page origin the Host header names; refuses any other host
        (DNS rebinding) since the server answers on the loopback only."""
        port = self.server.server_address[1]
        host = (self.headers.get("Host") or "").lower()
        if host not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            raise ApiError(
                HTTPStatus.FORBIDDEN, f"Host {host!r} refused: this server is 127.0.0.1:{port} only"
            )
        return f"http://{host}"

    def _check_same_origin(self, origin: str) -> None:
        sent = self.headers.get("Origin")
        if sent is not None and sent.lower() != origin:
            raise ApiError(HTTPStatus.FORBIDDEN, f"cross-origin request from {sent!r} refused")
        site = self.headers.get("Sec-Fetch-Site")
        if site is not None and site not in ("same-origin", "none"):
            raise ApiError(HTTPStatus.FORBIDDEN, f"Sec-Fetch-Site {site!r} refused")
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            raise ApiError(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "POST bodies must be application/json"
            )

    def _read_json(self) -> Any:
        if self.headers.get("Transfer-Encoding"):
            raise ApiError(HTTPStatus.LENGTH_REQUIRED, "send a Content-Length body, not chunks")
        length = self.headers.get("Content-Length")
        if length is None:
            raise ApiError(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required")
        try:
            n = int(length)
        except ValueError:
            n = -1
        if n < 0:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"bad Content-Length {length!r}")
        if n > MAX_BODY_BYTES:
            raise ApiError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"body of {n} bytes exceeds the {MAX_BODY_BYTES}-byte limit",
            )
        raw = self.rfile.read(n)
        if len(raw) != n:
            raise ApiError(HTTPStatus.BAD_REQUEST, "body shorter than its Content-Length")
        try:
            return json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
        except (UnicodeDecodeError, ValueError) as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"invalid JSON body: {exc}") from None

    def _static_file(self, path: str) -> Path:
        """A generated web asset under ``docs/explainers``: no dotfiles, no
        traversal, only the :data:`STATIC_TYPES` suffixes (no sources)."""
        rel = unquote(path).lstrip("/")
        parts = Path(rel).parts
        if not rel or "\x00" in rel or "\\" in rel or any(p.startswith(".") for p in parts):
            raise ApiError(HTTPStatus.NOT_FOUND, f"{path} is not served")
        target = (EXPLAINERS_ROOT / rel).resolve()
        if (
            not target.is_relative_to(EXPLAINERS_ROOT)
            or target.suffix.lower() not in STATIC_TYPES
            or not target.is_file()
        ):
            raise ApiError(HTTPStatus.NOT_FOUND, f"{path} is not served")
        return target

    def _send(self, status: HTTPStatus, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        with suppress(BrokenPipeError, ConnectionResetError):
            self._send(status, dumps({"error": message}))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Live backend of the MD2 Fig. 6(a) fit renderer.")
    ap.add_argument("--port", type=int, default=8766, help="port on 127.0.0.1 (default 8766)")
    args = ap.parse_args(argv)
    if not 0 <= args.port <= 65535:
        ap.error(f"--port must be in 0..65535, got {args.port}")
    server = RendererServer((HOST, args.port), Handler)
    port = server.server_address[1]
    print(
        f"bound {HOST}:{port}; loading {RECORDING} @ {START_S} s and rendering the baseline ...",
        flush=True,
    )
    try:
        t0 = time.perf_counter()
        server.renderer = Md2Renderer()
        r = server.renderer
        print(
            f"ready in {time.perf_counter() - t0:.1f} s: http://{HOST}:{port}/  "
            f"(baseline sha256 {r.baseline_signal['waveform_sha256']}, "
            f"colour range {r.vmax_db - DYNAMIC_RANGE_DB:.1f}..{r.vmax_db:.1f} dB)",
            flush=True,
        )
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
