"""Coefficient-space GP control adapted from Lee, Ko, Seshadri & Rauleder (2026).

Lee et al., "Bayesian machine learning framework for time-domain prediction of
multirotor vehicle noise", JASA 159(4):3418-3435, DOI 10.1121/10.0043469. The
original trains a joint waveform sparse-variational GP (BPF-informed Fourier
kernel times separable Matern-5/2 spatial/velocity kernels, diagonal Fourier
prior variances, 1000 inducing points, predictive log likelihood) on
CONA-simulated flyovers of a 583.85 kg NASA 1-Pax quadrotor, splits tonal and
broadband parts with a four-level db4 DWT, phase-aligns every training signal
to the first fore-rotor BPF and models broadband as diagonal Gaussian noise.

This module is an ADAPTED CONTROL for real onboard recordings (method ids
``lee2026_gp`` with white broadband, ``lee2026_gp_colored`` with residual-
shaped broadband; both share ONE fitted checkpoint). It is neither the
authors' code nor a paper-exact reproduction, and no published Lee et al.
number is reproduced. It reuses the repository's factorised implementation:
the batched exact ARD Matern-5/2 GP factory
:func:`data_processing.egonoise_gp._make_gp` and the coefficient phase
convention of :func:`experiments.gp_rotor_noise.jasa_gp.align_coeffs`.

Fit (calibration windows only, :func:`fit_lee`):

1. Every full, non-overlapping ``frame_s`` frame of the native-rate (44.1 kHz)
   calibration audio is listed; at most ``max_frames`` are selected at evenly
   spaced manifest indices (deterministic, calibration-only).
2. Per frame, the supplied per-rotor RPS is interpolated onto the native grid
   and integrated to a blade-pass phase. One design column for DC plus a
   ``(sin, cos)`` pair per rotor and BPF order ``k = 1..n_harmonics`` is solved
   jointly for all microphones by Hann-weighted least squares with a RELATIVE
   singular-value cutoff ``lsq_rcond`` (the legacy rank-truncation intent,
   without its absolute ``1e-3`` floor).
3. The weighted residual (computed BEFORE any phase alignment) gives the
   per-microphone residual standard deviation and Welch PSD of the frame.
4. Each rotor block is de-rotated by ONE time shift per (frame, rotor) that
   zeroes the array-mean fundamental phase, in the ``align_coeffs`` convention
   (order ``k`` rotated by ``k * phi1``; never per-order). The shaft phases are
   unobserved, so this per-rotor convention removes the nuisance phase while
   keeping the measured inter-microphone phase.
5. One independent ARD Matern-5/2 GP per coefficient regresses the aligned
   coefficients on (microphone xyz from ``sources.geometry('michaels')``, the
   four frame-mean rotor RPS), inputs and targets standardised on the
   calibration frames only; Adam, float64, CPU. Tasks are trained in chunks of
   ``gp_task_chunk``; tasks share no parameters and Adam is elementwise, so
   chunking reproduces the all-task optimisation exactly.
6. Broadband: a per-microphone residual-SD table over the frame-mean RPS
   (sorted, exact duplicates merged by RMS, linearly interpolated, clamped at
   its ends) and a global per-microphone mean residual PSD shape.

Render (:func:`render_lee`): GP posterior-MEAN coefficients at the per-rotor
clip-mean RPS (as the repository ego-noise pool queries at the chunk mean);
the supplied instantaneous RPS drives continuous FM of every rotor comb with a
uniformly random initial BPF phase per rotor (the alignment convention leaves
it undetermined). The comb is synthesised at a work rate that is a multiple of
the native rate, at least ``min_work_sr``, and high enough that no harmonic
can alias into the output band; broadband is drawn at the native rate (white,
or shaped by the residual PSD and renormalised to the SAME expected
calibration variance). Both pass the stated anti-alias low-pass
(:func:`data_processing.noise_model.resample.antialias`) and the shared
decimator :func:`data_processing.noise_model.resample.decimate_audio`.
Padding on both sides absorbs filter edges; exactly ``T`` samples return. No
evaluation RMS/gain matching and no waveform reuse.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import time
import warnings
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from scipy import signal

from data_processing.egonoise_gp import _make_gp
from data_processing.noise_model.resample import AA_PASS_HZ, AA_STOP_HZ, antialias, decimate_audio
from data_processing.sources import geometry
from experiments.generator_paper.cases import FlightClip
from experiments.generator_paper.metrics import _jsonable
from experiments.gp_rotor_noise import jasa_gp as J

SCHEMA = "generator-paper-lee2026-gp/2"
METHOD_IDS = ("lee2026_gp", "lee2026_gp_colored")
DOI = "10.1121/10.0043469"
BIBLIOGRAPHY_UUID = "4d9d6491-24da-5c4a-977e-0ab228e7fce8"
REFERENCE = (
    "Lee, Ko, Seshadri & Rauleder (2026), Bayesian machine learning framework for "
    "time-domain prediction of multirotor vehicle noise, JASA 159(4):3418-3435"
)
#: Registry geometry per supported rig (DREGON geometry needs its raw tree).
GEOMETRY_BY_RIG = {"md2": "michaels"}
INPUT_NAMES = (
    "mic_x_m",
    "mic_y_m",
    "mic_z_m",
    "rotor0_rps",
    "rotor1_rps",
    "rotor2_rps",
    "rotor3_rps",
)
_REPO_ROOT = Path(__file__).resolve().parents[3]
#: Repository-relative sources this comparator executes (hash inventory).
GP_SOURCE_FILES = (
    "src/experiments/generator_paper/gp_baseline.py",
    "src/data_processing/egonoise_gp.py",
    "src/experiments/gp_rotor_noise/jasa_gp.py",
    "src/data_processing/noise_model/resample.py",
    "src/data_processing/sources/__init__.py",
    "src/data_processing/sources/michaels.py",
    "src/experiments/generator_paper/cases.py",
    "src/experiments/generator_paper/metrics.py",
)
_SOURCE_ROLES = {
    "src/experiments/generator_paper/gp_baseline.py": "this adapter",
    "src/data_processing/egonoise_gp.py": "_make_gp batched exact ARD Matern-5/2 GP factory",
    "src/experiments/gp_rotor_noise/jasa_gp.py": "align_coeffs coefficient phase convention",
    "src/data_processing/noise_model/resample.py": "antialias low-pass and decimate_audio",
    "src/data_processing/sources/__init__.py": "geometry registry",
    "src/data_processing/sources/michaels.py": "md2 microphone geometry",
    "src/experiments/generator_paper/cases.py": "FlightClip calibration windows",
    "src/experiments/generator_paper/metrics.py": "_jsonable report serialisation",
}
DEVIATIONS = (
    "Data: real md2 FLY125 onboard 8-microphone calibration recordings at native 44.1 kHz, not "
    "CONA-simulated NASA 1-Pax flyovers; source-fixed microphones, so no de-Dopplerisation.",
    "Model: closed-form per-frame coefficient projection followed by independent batched exact "
    "ARD Matern-5/2 GPs per coefficient (data_processing.egonoise_gp._make_gp), not the joint "
    "waveform SVGP with product Fourier x Matern kernel, 1000 inducing points and predictive "
    "log likelihood; no covariance between coefficients and no jointly learned Fourier prior.",
    "Conditioning: microphone xyz plus the four measured frame-mean rotor RPS (7 ARD inputs) "
    "instead of observer position and flight velocity.",
    "Basis: one BPF comb per rotor (orders 1..n_harmonics of blades x RPS_r) with phase integrated "
    "from the measured instantaneous rotor RPS, plus one shared DC term, instead of one "
    "vehicle BPF-informed comb.",
    "Phase: per-rotor alignment in the jasa_gp.align_coeffs convention (order k rotated by "
    "k*phi1), one rotation per (frame, rotor) from the array-mean fundamental shared by all "
    "microphones, because the four shaft phases are unobserved; the paper aligns each signal to "
    "the first fore-rotor BPF.",
    "Tonal/broadband split: residual of the weighted rank-truncated projection (before "
    "alignment), not a four-level db4 DWT.",
    "Broadband: per-microphone residual SD interpolated over mean RPS; white Gaussian as in the "
    "published family, plus a colored variant using the global per-microphone residual PSD shape "
    "at the same expected variance (an extension, not in the paper).",
    "Rendering: posterior-mean coefficients at clip-mean per-rotor RPS with continuous FM from "
    "the supplied RPS and a random initial BPF phase per rotor; no posterior samples or "
    "uncertainty quantification.",
    "Training design: a fixed calibration subset (at most 64 evenly spaced 1-s native frames) "
    "replaces the paper's spatial active learning of 12 observer points and its V=6/8/10 training "
    "speeds; the paper's original data and metadata are not used.",
    "Evaluation: the repository generator-paper protocol (spectral, order, spatial and nominal "
    "psychoacoustic diagnostics) replaces the paper's phase-aligned DTW, ISO 532-1/SQAT loudness "
    "and Zwicker-Fastl annoyance; results are not comparable to published Lee et al. scores.",
    "Scope: the render microphones are the eight training microphones; no spatial "
    "generalisation, uncertainty, or reproduction of published Lee et al. results is claimed.",
)


@dataclass(frozen=True)
class LeeSettings:
    """Fit/render settings. Defaults are the fixed production design.

    Smaller values exist only so a tiny synthetic smoke can exercise the code;
    every value is recorded in the report together with whether it equals the
    production default.
    """

    n_harmonics: int = 60
    n_blades: int = 2
    frame_s: float = 1.0
    max_frames: int = 64
    lsq_rcond: float = 1e-3
    gp_iterations: int = 200
    gp_lr: float = 0.05
    gp_task_chunk: int = 128
    welch_nperseg: int = 4096
    min_work_sr: int = 32000
    edge_pad_s: float = 0.1

    def __post_init__(self) -> None:
        ints = (
            "n_harmonics",
            "n_blades",
            "max_frames",
            "gp_iterations",
            "gp_task_chunk",
            "welch_nperseg",
            "min_work_sr",
        )
        for name in ints:
            if int(getattr(self, name)) != getattr(self, name) or getattr(self, name) < 1:
                raise ValueError(f"LeeSettings.{name} must be a positive integer")
        for name in ("frame_s", "lsq_rcond", "gp_lr", "edge_pad_s"):
            v = float(getattr(self, name))
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"LeeSettings.{name} must be positive and finite")
        if self.lsq_rcond >= 1.0:
            raise ValueError("LeeSettings.lsq_rcond is a relative cutoff and must be < 1")


# ── shared numerics ────────────────────────────────────────────────────────


def _integrate_phase(rps: np.ndarray, rate: float, blades: int, phi0: np.ndarray) -> np.ndarray:
    """Blade-pass phase ``(R, n)`` [rad]: trapezoidal integral of ``blades * rps``."""
    inc = (rps[:, 1:] + rps[:, :-1]) * (math.pi * blades / rate)
    out = np.empty_like(rps)
    out[:, 0] = 0.0
    np.cumsum(inc, axis=1, out=out[:, 1:])
    return out + np.asarray(phi0, dtype=np.float64)[:, None]


def _design(phase: np.ndarray, n_harm: int) -> np.ndarray:
    """``(n, 1 + 2RH)``: DC, then per rotor ``sin(k phi), cos(k phi)`` for ``k = 1..H``.

    The per-rotor column order is the ``jasa_gp.fourier_design`` /
    ``align_coeffs`` layout ``[A_1, B_1, A_2, B_2, ...]``.
    """
    n_rot, n = phase.shape
    a = np.empty((n, 1 + 2 * n_rot * n_harm))
    a[:, 0] = 1.0
    k = np.arange(1, n_harm + 1, dtype=np.float64)
    for r in range(n_rot):
        arg = np.multiply.outer(phase[r], k)
        base = 1 + 2 * n_harm * r
        a[:, base : base + 2 * n_harm : 2] = np.sin(arg)
        a[:, base + 1 : base + 2 * n_harm : 2] = np.cos(arg)
    return a


def _rotate(block: np.ndarray, phi1: float, n_harm: int) -> np.ndarray:
    """``align_coeffs`` rotation of ``(M, 2H)`` rotor blocks by an external ``phi1``."""
    k = np.arange(1, n_harm + 1, dtype=np.float64)
    c, s = np.cos(k * phi1), np.sin(k * phi1)
    a, b = block[:, 0::2], block[:, 1::2]
    out = np.empty_like(block)
    out[:, 0::2] = a * c + b * s
    out[:, 1::2] = b * c - a * s
    return out


def _align_rotors(
    coef: np.ndarray, n_rot: int, n_harm: int
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """One per-rotor time shift per frame, shared by all microphones.

    ``phi1`` is the fundamental phase of the array-mean rotor block. Its
    rotation is checked against :func:`jasa_gp.align_coeffs` applied to that
    array-mean vector, so the convention (sign included) is the source's.
    """
    out = coef.copy()
    diag = []
    for r in range(n_rot):
        sl = slice(1 + 2 * n_harm * r, 1 + 2 * n_harm * (r + 1))
        block = coef[:, sl]
        ref = block.mean(axis=0)
        phi1 = float(np.arctan2(ref[1], ref[0]))
        rotated = _rotate(block, phi1, n_harm)
        expected = J.align_coeffs(np.concatenate(([0.0], ref)), n_harm)[1:]
        scale = max(float(np.abs(block).max()), np.finfo(np.float64).tiny)
        if not np.allclose(rotated.mean(axis=0), expected, rtol=1e-9, atol=1e-9 * scale):
            raise RuntimeError("per-rotor alignment diverged from jasa_gp.align_coeffs")
        out[:, sl] = rotated
        amp = float(np.hypot(ref[0], ref[1]))
        per_mic = float(np.hypot(block[:, 0], block[:, 1]).mean())
        diag.append(
            {
                "rotor": r,
                "reference_phase_rad": phi1,
                "array_mean_fundamental_amplitude": amp,
                "array_fundamental_coherence": amp / per_mic if per_mic > 0 else None,
            }
        )
    return out, diag


def _predict_standardized(
    models: list[Any], chunks: list[tuple[int, int]], xs: torch.Tensor
) -> np.ndarray:
    """Posterior mean ``(Q, P)`` in standardised target units."""
    import gpytorch

    q, d = xs.shape
    out = np.empty((q, chunks[-1][1]))
    with torch.no_grad(), gpytorch.settings.skip_posterior_variances(True):
        for (lo, hi), model in zip(chunks, models, strict=True):
            xb = xs.unsqueeze(0).expand(hi - lo, q, d).contiguous()
            out[:, lo:hi] = model(xb).mean.numpy().T
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantiles(x: np.ndarray) -> dict[str, Any]:
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "min": float(v.min()),
        "median": float(np.median(v)),
        "max": float(v.max()),
        "mean": float(v.mean()),
    }


# ── fitted model ───────────────────────────────────────────────────────────


class LeeFit:
    """Loaded GP state. Build with :func:`fit_lee` or :meth:`LeeFit.load`."""

    def __init__(
        self, payload: dict[str, Any], report: dict[str, Any], checkpoint_sha256: str
    ) -> None:
        self.settings = LeeSettings(**payload["settings"])
        self.native_sr = int(payload["native_sr"])
        self.n_rotors = int(payload["n_rotors"])
        self.mic_positions = payload["mic_positions"].numpy()
        self.n_mics = int(self.mic_positions.shape[0])
        self.x_mean = payload["x_mean"].numpy()
        self.x_std = payload["x_std"].numpy()
        self.y_mean = payload["y_mean"].numpy()
        self.y_std = payload["y_std"].numpy()
        self.task_chunks = [(int(lo), int(hi)) for lo, hi in payload["task_chunks"]]
        self.bb_rps_grid = payload["bb_rps_grid"].numpy()
        self.bb_sigma = payload["bb_sigma"].numpy()
        self.psd_freqs = payload["psd_freqs"].numpy()
        self.psd_shape = payload["psd_shape"].numpy()
        train_x, train_y = payload["train_x"], payload["train_y"]
        self._models = []
        for (lo, hi), state in zip(self.task_chunks, payload["gp_states"], strict=True):
            model, likelihood, _, _ = _make_gp(train_x, train_y[:, lo:hi])
            model.to(dtype=torch.float64)
            model.load_state_dict(state)
            model.eval()
            likelihood.eval()
            self._models.append(model)
        self.report = report
        self.checkpoint_sha256 = checkpoint_sha256

    @classmethod
    def load(cls, path: Path) -> LeeFit:
        """Load a :func:`fit_lee` checkpoint (``torch.load(weights_only=True)``).

        The stored fit-time probe prediction is recomputed and must agree, so a
        checkpoint that does not reproduce its GP is refused.
        """
        path = Path(path)
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        if payload.get("schema") != SCHEMA:
            raise ValueError(f"{path.name}: schema {payload.get('schema')!r} is not {SCHEMA!r}")
        report = json.loads(payload["report_json"])
        fit = cls(payload, report, sha)
        probe_x = payload["probe_x"].numpy()
        probe = payload["probe_coeffs"].numpy()
        got = fit._predict(probe_x)
        err = float(np.abs(got - probe).max())
        tol = 1e-4 * max(float(np.abs(probe).max()), np.finfo(np.float64).tiny)
        if not err <= tol:
            raise ValueError(
                f"{path.name}: reloaded GP does not reproduce its fit probe ({err:g} > {tol:g})"
            )
        fit.report = {
            **report,
            "checkpoint_file": path.name,
            "checkpoint_sha256": sha,
            "load_probe_max_abs_error": err,
        }
        return fit

    def _predict(self, x: np.ndarray) -> np.ndarray:
        xs = torch.as_tensor(
            (np.asarray(x, dtype=np.float64) - self.x_mean) / self.x_std, dtype=torch.float64
        )
        return _predict_standardized(self._models, self.task_chunks, xs) * self.y_std + self.y_mean

    def predict_coeffs(self, rotor_rps: np.ndarray) -> np.ndarray:
        """Posterior-mean aligned coefficients ``(M, 1 + 2RH)`` at per-rotor RPS ``(R,)``."""
        q = np.asarray(rotor_rps, dtype=np.float64).reshape(-1)
        if q.size != self.n_rotors or not np.isfinite(q).all():
            raise ValueError(f"expected {self.n_rotors} finite rotor speeds, got {q}")
        x = np.concatenate([self.mic_positions, np.broadcast_to(q, (self.n_mics, q.size))], axis=1)
        return self._predict(x)

    def broadband_sigma(self, mean_rps: float) -> np.ndarray:
        """Per-microphone residual SD ``(M,)`` at native rate; clamped table interpolation."""
        return np.array(
            [np.interp(float(mean_rps), self.bb_rps_grid, row) for row in self.bb_sigma]
        )


# ── fit ────────────────────────────────────────────────────────────────────


def _check_clips(clips: list[FlightClip]) -> tuple[str, int, int, int, int]:
    if not clips:
        raise ValueError("fit_lee requires real calibration windows")
    for c in clips:
        if c.provenance.get("split") != "calibration":
            raise ValueError(
                f"fit_lee accepts calibration windows only; {c.key} is {c.provenance.get('split')!r}"
            )

    def one(name: str, values: set[Any]) -> Any:
        if len(values) != 1:
            raise ValueError(f"calibration windows disagree on {name}: {sorted(values)}")
        return values.pop()

    rig = one("rig", {c.rig for c in clips})
    if rig not in GEOMETRY_BY_RIG:
        raise ValueError(f"fit_lee supports rigs {sorted(GEOMETRY_BY_RIG)}, got {rig!r}")
    if len({c.key for c in clips}) != len(clips):
        raise ValueError("calibration window keys must be unique")
    return (
        rig,
        int(one("native_sr", {c.native_sr for c in clips})),
        int(one("sr", {c.sr for c in clips})),
        int(one("microphone count", {c.audio.shape[0] for c in clips})),
        int(one("rotor count", {c.rps.shape[0] for c in clips})),
    )


def _frame_rps(clip: FlightClip, start: int, n: int) -> np.ndarray:
    """Supplied RPS ``(R, n)`` linearly interpolated onto native frame samples."""
    t = (start + np.arange(n)) / clip.native_sr
    grid = np.arange(clip.rps.shape[-1]) / clip.sr
    return np.stack([np.interp(t, grid, np.asarray(row, dtype=np.float64)) for row in clip.rps])


def fit_lee(
    clips: list[FlightClip],
    *,
    seed: int,
    checkpoint: Path,
    settings: LeeSettings | None = None,
) -> LeeFit:
    """Fit the coefficient-GP control on calibration windows and save ``checkpoint``.

    Returns :meth:`LeeFit.load` of the written file, so the caller holds exactly
    the state every evaluation worker loads. The fit has no stochastic step
    (deterministic frame selection, initialisation and Adam); ``seed`` seeds
    torch and is recorded.
    """
    import gpytorch
    from gpytorch.utils.warnings import GPInputWarning

    cfg = settings or LeeSettings()
    clips = list(clips)
    started = time.monotonic()
    rig, native_sr, sr, n_mics, n_rot = _check_clips(clips)
    if n_rot != len(INPUT_NAMES) - 3:
        raise ValueError(f"the conditioning inputs are defined for 4 rotors, got {n_rot}")
    mic_pos = np.asarray(geometry(GEOMETRY_BY_RIG[rig])[0], dtype=np.float64)
    if mic_pos.shape != (n_mics, 3):
        raise ValueError(f"geometry has {mic_pos.shape} microphones, windows carry {n_mics}")
    n_frame = int(round(cfg.frame_s * native_sr))
    if abs(n_frame - cfg.frame_s * native_sr) > 1e-6:
        raise ValueError(f"frame_s={cfg.frame_s} is not a whole number of {native_sr} Hz samples")
    if cfg.welch_nperseg > n_frame:
        raise ValueError("welch_nperseg exceeds the frame length")
    torch.manual_seed(int(seed))
    H, B = cfg.n_harmonics, cfg.n_blades
    n_tasks = 1 + 2 * n_rot * H

    # Full manifest of eligible frames over the whole calibration set.
    frames = []
    for ci, clip in enumerate(clips):
        for j in range(clip.native_audio.shape[-1] // n_frame):
            frames.append(
                {
                    "frame": len(frames),
                    "clip_index": ci,
                    "clip_key": clip.key,
                    "recording_id": clip.recording_id,
                    "regime": clip.regime,
                    "clip_frame": j,
                    "native_start_sample_in_clip": j * n_frame,
                    "native_n_samples": n_frame,
                    "relative_start_s": float(clip.start_s + j * cfg.frame_s),
                    "duration_s": cfg.frame_s,
                    "mean_rps": _frame_rps(clip, j * n_frame, n_frame).mean(axis=1),
                }
            )
    if not frames:
        raise ValueError(f"no full {cfg.frame_s:g} s frame in the calibration windows")
    n_sel = min(cfg.max_frames, len(frames))
    selected = np.unique(np.round(np.linspace(0, len(frames) - 1, n_sel)).astype(np.int64))
    sel_set = set(selected.tolist())
    for f in frames:
        f["selected"] = f["frame"] in sel_set

    # Frame-at-a-time projection.
    weight = np.asarray(signal.get_window("hann", n_frame), dtype=np.float64)
    w2_sum = float(weight @ weight)
    xs, ws, bb_level, bb_sd, extraction = [], [], [], [], []
    psd_sum = None
    t_extract = time.monotonic()
    for idx in selected:
        f = frames[int(idx)]
        clip = clips[f["clip_index"]]
        a0 = f["native_start_sample_in_clip"]
        rps = _frame_rps(clip, a0, n_frame)
        if not (np.isfinite(rps).all() and (rps >= 0).all()):
            raise ValueError(f"{clip.key}: non-finite or negative RPS in frame {f['clip_frame']}")
        top = H * B * float(rps.max())
        if top >= native_sr / 2:
            raise ValueError(f"{clip.key}: order {H} reaches {top:.0f} Hz >= native Nyquist")
        y = np.asarray(clip.native_audio[:, a0 : a0 + n_frame], dtype=np.float64).T
        design = _design(_integrate_phase(rps, native_sr, B, np.zeros(n_rot)), H)
        design *= weight[:, None]
        yw = y * weight[:, None]
        coef, _, rank, sv = np.linalg.lstsq(design, yw, rcond=cfg.lsq_rcond)
        resid = yw - design @ coef  # weighted residual, before alignment
        del design
        sd = np.sqrt(np.sum(resid**2, axis=0) / w2_sum)
        frac = np.sum(resid**2, axis=0) / np.sum(yw**2, axis=0)
        freqs, psd = signal.welch(
            resid.T,
            fs=native_sr,
            window="hann",
            nperseg=cfg.welch_nperseg,
            noverlap=cfg.welch_nperseg // 2,
            detrend="constant",
            axis=-1,
        )
        psd_sum = psd if psd_sum is None else psd_sum + psd
        aligned, rot_diag = _align_rotors(coef.T, n_rot, H)
        frame_rps = rps.mean(axis=1)
        xs.append(np.concatenate([mic_pos, np.broadcast_to(frame_rps, (n_mics, n_rot))], axis=1))
        ws.append(aligned)
        bb_level.append(float(frame_rps.mean()))
        bb_sd.append(sd)
        kept = sv[sv > cfg.lsq_rcond * sv[0]]
        extraction.append(
            {
                "frame": int(idx),
                "clip_key": clip.key,
                "frame_mean_rps": frame_rps,
                "max_harmonic_hz": top,
                "design_columns": n_tasks,
                "rank": int(rank),
                "kept_singular_ratio_min": float(kept[-1] / sv[0]),
                "residual_sd": sd,
                "residual_energy_fraction": frac,
                "rotor_alignment": rot_diag,
            }
        )
    extract_seconds = time.monotonic() - t_extract
    X, W = np.concatenate(xs, axis=0), np.concatenate(ws, axis=0)
    assert psd_sum is not None
    psd_shape = psd_sum / len(selected)

    # Broadband table: sorted operating points, exact duplicates merged by RMS.
    levels = np.asarray(bb_level)
    sds = np.stack(bb_sd, axis=1)  # (M, n_frames)
    grid, inverse = np.unique(levels, return_inverse=True)
    table = np.stack(
        [np.sqrt(np.bincount(inverse, sds[m] ** 2) / np.bincount(inverse)) for m in range(n_mics)]
    )

    # Standardisation from calibration frames only; constant columns keep unit scale.
    x_mean, x_sd = X.mean(axis=0), X.std(axis=0)
    y_mean, y_sd = W.mean(axis=0), W.std(axis=0)
    x_std = np.where(x_sd > 0, x_sd, 1.0)
    y_std = np.where(y_sd > 0, y_sd, 1.0)
    # Closely spaced calibration states make the GP covariance ill-conditioned;
    # keep its distances, kernel parameters and factorisation in double precision.
    train_x = torch.tensor((X - x_mean) / x_std, dtype=torch.float64)
    train_y = torch.tensor((W - y_mean) / y_std, dtype=torch.float64)

    t_gp = time.monotonic()
    chunks = [
        (lo, min(n_tasks, lo + cfg.gp_task_chunk)) for lo in range(0, n_tasks, cfg.gp_task_chunk)
    ]
    models, states, traces = [], [], []
    ls, outscale, noise, constant, r2 = [], [], [], [], []
    for lo, hi in chunks:
        model, likelihood, tx, ty = cast(Any, _make_gp(train_x, train_y[:, lo:hi]))
        model.to(dtype=torch.float64)
        model.train()
        likelihood.train()
        opt = torch.optim.Adam(model.parameters(), lr=cfg.gp_lr)
        mll: Any = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        trace = []
        for _ in range(cfg.gp_iterations):
            opt.zero_grad(set_to_none=True)
            loss = -mll(model(tx), ty).sum()
            loss.backward()
            opt.step()
            trace.append(float(loss.item()))
        model.eval()
        likelihood.eval()
        with (
            torch.no_grad(),
            gpytorch.settings.skip_posterior_variances(True),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", GPInputWarning)
            mean_fit = model(tx).mean
        var = ty.var(-1, unbiased=False)
        r2_chunk = 1.0 - ((mean_fit - ty) ** 2).mean(-1) / var.clamp_min(1e-12)
        r2.append(torch.where(var > 0, r2_chunk, torch.full_like(r2_chunk, float("nan"))).numpy())
        ls.append(model.covar_module.base_kernel.lengthscale.detach().reshape(hi - lo, -1).numpy())
        outscale.append(model.covar_module.outputscale.detach().reshape(-1).numpy())
        noise.append(likelihood.noise.detach().reshape(-1).numpy())
        constant.append(model.mean_module.constant.detach().reshape(-1).numpy())
        models.append(model)
        states.append(model.state_dict())
        traces.append(trace)
    gp_seconds = time.monotonic() - t_gp
    lengthscale = np.concatenate(ls, axis=0)

    probe_x = X[:n_mics].copy()
    probe_std = _predict_standardized(
        models, chunks, torch.as_tensor((probe_x - x_mean) / x_std, dtype=torch.float64)
    )
    probe_coeffs = probe_std * y_std + y_mean
    analysis_nyquist = sr / 2
    below = psd_shape[:, freqs <= analysis_nyquist].sum(axis=1) / psd_shape.sum(axis=1)

    reuse = [
        {"path": rel, "role": _SOURCE_ROLES[rel], "sha256": _sha256(_REPO_ROOT / rel)}
        for rel in GP_SOURCE_FILES
    ]
    report = {
        "schema": SCHEMA,
        "method": "Lee et al. 2026 coefficient-GP adaptation (control)",
        "method_ids": {
            "lee2026_gp": "white Gaussian broadband",
            "lee2026_gp_colored": "residual-PSD-shaped broadband, same variance",
        },
        "status": (
            "adapted coefficient-space GP control on real calibration audio; not the authors' "
            "code, not a paper-exact reproduction, no published numbers reproduced"
        ),
        "doi": DOI,
        "bibliography_uuid": BIBLIOGRAPHY_UUID,
        "reference": REFERENCE,
        "deviations": list(DEVIATIONS),
        "adapter_choices": {
            "gp_factory": "data_processing.egonoise_gp._make_gp (independent batched ExactGP, "
            "ConstantMean, ScaleKernel(ARD Matern nu=2.5), GaussianLikelihood per task)",
            "phase_convention": "jasa_gp.align_coeffs: phi1=atan2(B1,A1), order k rotated by k*phi1, "
            "fundamental becomes +sin; one rotation per (frame, rotor) from the "
            "array-mean fundamental, shared by all microphones; checked against "
            "align_coeffs every frame",
            "legacy_not_used": "GPRotorNoiseModel and GPRotorNoisePool (alignment sign, generate "
            "standardisation/OLA gain, simulated training, target-RMS normalisation)",
            "basis": "DC + per rotor (sin, cos) of k*phi_r, k=1..n_harmonics, phi_r = 2*pi*blades*"
            "integral(rps_r) (trapezoid) on the native grid, phase 0 at frame start",
            "rps_on_native_grid": "linear interpolation of FlightClip.rps (16 kHz grid) at native sample times",
            "least_squares": "numpy.linalg.lstsq on Hann-weighted design and audio (sqrt-weight = "
            "scipy hann, periodic), joint for all microphones, rcond relative cutoff",
            "residual": "weighted residual hann*(y - A c) before alignment; SD = sqrt(sum r_w^2 / sum hann^2)",
            "gp_inputs": list(INPUT_NAMES),
            "gp_targets": f"{n_tasks} aligned coefficients [DC, rotor0 A1,B1..A{H},B{H}, ..., rotor3 ...]",
            "standardisation": "mean/SD over calibration (frame, microphone) points; zero-SD columns use SD 1",
            "gp_optimiser": "Adam on -sum_tasks ExactMarginalLogLikelihood, float64, CPU",
            "gp_task_chunking": "exactly equivalent to all-task training (no shared parameters, elementwise Adam)",
            "broadband_table": "x = frame mean over rotors of frame-mean RPS; per-microphone residual SD; "
            "sorted; exact duplicates merged by RMS; np.interp (clamped ends)",
            "colored_shape": "per-microphone mean Welch PSD of the weighted residual over selected "
            "frames, linearly interpolated onto the render rfft grid and normalised to "
            "unit expected variance",
            "render_coefficients": "posterior mean at per-rotor clip-mean RPS",
            "render_fm": "supplied instantaneous per-rotor RPS (linear interpolation, edge hold in padding)",
            "render_initial_phase": "uniform [0, 2*pi) BPF phase per rotor from the supplied rng",
            "render_random_stream": "rng.uniform(0, 2*pi, R) then rng.standard_normal((M, n_native))",
            "render_rates": "comb at work rate = smallest native multiple >= min_work_sr whose images "
            "of every harmonic stay above sr/2; broadband at native rate",
            "render_observation": "antialias(pass sr/2-100 Hz, stop sr/2, 100 dB) then decimate_audio "
            "to sr; real clips are decimate_audio of the native recording",
            "render_length": "edge_pad_s padding on both sides on a grid common to sr, work and native "
            "rates, then exactly T samples",
            "no_test_calibration": "no evaluation data, RMS/gain matching or recording reuse",
        },
        "settings": asdict(cfg),
        "production_settings": cfg == LeeSettings(),
        "seed": int(seed),
        "rig": rig,
        "geometry_source": f"data_processing.sources.geometry({GEOMETRY_BY_RIG[rig]!r})",
        "mic_positions_m": mic_pos,
        "native_sr": native_sr,
        "analysis_sr": sr,
        "n_mics": n_mics,
        "n_rotors": n_rot,
        "n_tasks": n_tasks,
        "n_points": int(X.shape[0]),
        "calibration_clip_keys": [c.key for c in clips],
        "calibration_recordings": sorted({c.recording_id for c in clips}),
        "calibration_splits": sorted({str(c.provenance.get("split")) for c in clips}),
        "frame_manifest": {
            "eligible_frames": len(frames),
            "selected_frames": int(selected.size),
            "selection": "np.round(np.linspace(0, eligible-1, min(max_frames, eligible))) over the "
            "manifest in clip order, then time; deterministic",
            "frames": [{k: v for k, v in f.items() if k != "clip_index"} for f in frames],
        },
        "extraction": {
            "frames": extraction,
            "rank_min": int(min(e["rank"] for e in extraction)),
            "rank_deficient_frames": int(sum(e["rank"] < n_tasks for e in extraction)),
            "residual_sd": _quantiles(sds),
            "residual_energy_fraction": _quantiles(
                np.stack([e["residual_energy_fraction"] for e in extraction])
            ),
            "max_harmonic_hz": max(e["max_harmonic_hz"] for e in extraction),
            "wall_seconds": extract_seconds,
        },
        "broadband": {
            "rps_grid": grid,
            "sigma_per_mic": table,
            "n_operating_points": int(grid.size),
            "merged_duplicates": int(levels.size - grid.size),
            "psd_nperseg": cfg.welch_nperseg,
            "psd_resolution_hz": float(native_sr / cfg.welch_nperseg),
            "residual_power_fraction_below_analysis_nyquist": below,
        },
        "standardisation": {
            "x_mean": x_mean,
            "x_std": x_std,
            "constant_inputs": [INPUT_NAMES[i] for i in np.flatnonzero(x_sd == 0)],
            "constant_targets": int(np.sum(y_sd == 0)),
            "gp_input_range": {"min": X.min(axis=0), "max": X.max(axis=0)},
        },
        "gp": {
            "task_chunks": chunks,
            "iterations": cfg.gp_iterations,
            "lr": cfg.gp_lr,
            "loss_traces": traces,
            "final_loss": [t[-1] if t else None for t in traces],
            "lengthscale_by_input": {
                name: _quantiles(lengthscale[:, i]) for i, name in enumerate(INPUT_NAMES)
            },
            "outputscale": _quantiles(np.concatenate(outscale)),
            "noise": _quantiles(np.concatenate(noise)),
            "mean_constant": _quantiles(np.concatenate(constant)),
            "in_sample_r2_standardised": _quantiles(np.concatenate(r2)),
            "in_sample_note": "posterior mean at training inputs; not a validation score",
            "wall_seconds": gp_seconds,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": version("scipy"),
            "torch": torch.__version__,
            "gpytorch": gpytorch.__version__,
        },
        "reuse_sources": reuse,
        "wall_seconds": {
            "extraction": extract_seconds,
            "gp": gp_seconds,
            "total_before_save": time.monotonic() - started,
        },
    }
    payload = {
        "schema": SCHEMA,
        "settings": asdict(cfg),
        "native_sr": native_sr,
        "n_rotors": n_rot,
        "mic_positions": torch.from_numpy(mic_pos),
        "x_mean": torch.from_numpy(x_mean),
        "x_std": torch.from_numpy(x_std),
        "y_mean": torch.from_numpy(y_mean),
        "y_std": torch.from_numpy(y_std),
        "train_x": train_x,
        "train_y": train_y,
        "task_chunks": [[lo, hi] for lo, hi in chunks],
        "gp_states": states,
        "bb_rps_grid": torch.from_numpy(grid.astype(np.float64)),
        "bb_sigma": torch.from_numpy(table),
        "psd_freqs": torch.from_numpy(np.asarray(freqs, dtype=np.float64)),
        "psd_shape": torch.from_numpy(np.ascontiguousarray(psd_shape, dtype=np.float64)),
        "probe_x": torch.from_numpy(probe_x),
        "probe_coeffs": torch.from_numpy(probe_coeffs),
        "report_json": json.dumps(_jsonable(report), allow_nan=False, sort_keys=True),
    }
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint.with_name(checkpoint.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, checkpoint)
    return LeeFit.load(checkpoint)


# ── render ─────────────────────────────────────────────────────────────────


def _shape_noise(white: np.ndarray, freqs: np.ndarray, psd: np.ndarray, rate: int) -> np.ndarray:
    """Filter unit white noise by the PSD shape at unit EXPECTED variance per row."""
    n = white.shape[-1]
    f = np.fft.rfftfreq(n, 1.0 / rate)
    bins = np.full(f.size, 2.0)
    bins[0] = 1.0
    if n % 2 == 0:
        bins[-1] = 1.0
    spec = np.fft.rfft(white, axis=-1)
    for m in range(white.shape[0]):
        gain = np.sqrt(np.interp(f, freqs, psd[m]))
        norm = math.sqrt(float(bins @ gain**2) / n)
        if not norm > 0:
            raise ValueError("residual PSD shape is identically zero")
        spec[m] *= gain / norm
    return np.fft.irfft(spec, n, axis=-1)


def render_lee(
    fit: LeeFit,
    rps: np.ndarray,
    *,
    sr: int,
    rng: np.random.Generator,
    colored: bool = False,
) -> np.ndarray:
    """Render ``(M, T)`` float32 audio at ``sr`` for supplied per-rotor RPS ``(R, T)``."""
    cfg = fit.settings
    track = np.asarray(rps, dtype=np.float64)
    if track.ndim != 2 or track.shape[0] != fit.n_rotors or track.shape[1] < 1:
        raise ValueError(f"rps must be ({fit.n_rotors}, T>0), got {track.shape}")
    if not (np.isfinite(track).all() and (track >= 0).all()):
        raise ValueError("rps must be finite and non-negative")
    sr = int(sr)
    if sr <= 0:
        raise ValueError(f"sr must be positive, got {sr}")
    H, B, native = cfg.n_harmonics, cfg.n_blades, fit.native_sr
    n_t = track.shape[1]
    mean_rps = track.mean(axis=1)
    coeffs = fit.predict_coeffs(mean_rps)
    sigma = fit.broadband_sigma(float(mean_rps.mean()))
    phi0 = rng.uniform(0.0, 2.0 * math.pi, size=fit.n_rotors)

    stop_hz = sr / 2.0
    pass_hz = stop_hz - (AA_STOP_HZ - AA_PASS_HZ)
    top = H * B * float(track.max())
    mult = 1
    while native * mult < cfg.min_work_sr or top >= native * mult - stop_hz:
        mult += 1
    work_sr = native * mult
    step = math.lcm(sr // math.gcd(sr, work_sr), sr // math.gcd(sr, native))
    pad = step * math.ceil(cfg.edge_pad_s * sr / step)
    n_out = n_t + 2 * pad

    n_work = math.ceil(n_out * work_sr / sr)
    t_work = np.arange(n_work) / work_sr - pad / sr
    t_track = np.arange(n_t) / sr
    rps_work = np.stack([np.interp(t_work, t_track, row) for row in track])
    phase = _integrate_phase(rps_work, work_sr, B, phi0)
    del rps_work
    tonal = np.empty((fit.n_mics, n_work))
    block = 8192
    for i in range(0, n_work, block):
        tonal[:, i : i + block] = coeffs @ _design(phase[:, i : i + block], H).T
    del phase

    n_native = math.ceil(n_out * native / sr)
    noise = rng.standard_normal((fit.n_mics, n_native))
    if colored:
        noise = _shape_noise(noise, fit.psd_freqs, fit.psd_shape, native)
    noise *= sigma[:, None]

    def observe(x: np.ndarray, rate: int) -> np.ndarray:
        y = antialias(x, rate, pass_hz=pass_hz, stop_hz=stop_hz)
        return np.asarray(decimate_audio(y, rate, sr), dtype=np.float64)[:, :n_out]

    if work_sr == native:
        out = observe(tonal + noise, native)
    else:
        out = observe(tonal, work_sr) + observe(noise, native)
    return np.ascontiguousarray(out[:, pad : pad + n_t], dtype=np.float32)


__all__ = [
    "DEVIATIONS",
    "DOI",
    "GP_SOURCE_FILES",
    "LeeFit",
    "LeeSettings",
    "METHOD_IDS",
    "fit_lee",
    "render_lee",
]
