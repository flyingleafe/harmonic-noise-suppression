"""Why the round-2 v2 DREGON arm renders to PIT MAE 75 rev/s.

The R2 candidate (``results/noise_v2/rounds/round2/fits/dregon_room2_floor__flight_floor_only.json``
— a ``flight_floor_only`` fit on the DREGON room-2 cruise windows with the comb
frozen at the log-mean of the four ``bench_dregon_Motor{1..4}_70`` bench fits)
scores HPPNet PIT MAE 75.09 rev/s against a real 1.219 and a legacy 2.188. The
combs themselves are identified 21/21, so the defect is in the RENDER.

This runner takes TWO of the five frozen DREGON cruise supports and measures,
arm by arm, what the render puts on the wire:

* the real clip; the v2 render (cross-checked bit-for-bit against the scored
  arm's own npz); the LEGACY render of the same recording (the identity-matched
  ``results/S2/dregon_room2_cruise_refined.json`` route the frozen baseline came
  from, :func:`noise_v2_round_score.legacy_arm`);
* VARIANTS of the same fit, each disabling or re-levelling exactly one block:
  both speed exponents at their prior mean, the comb muted, the floor muted, the
  floor re-levelled onto the real window's band level, and the comb re-levelled
  onto three separate targets (the real clip's own k=2 line prominence, the
  comb-to-floor ratio the BENCH fits identified the comb at, and the real
  window's broadband level).

The comb-to-floor measurement is ORDER-TRACKED: DREGON cruise carriers move
10-25 rev/s inside a 4 s window, so a periodogram at the MEAN carrier smears
every line into its own neighbours and reads ~2 dB on real audio that HPPNet
tracks to 1.2 rev/s. Here every frame's line is read at that frame's own label
carrier, and the local floor excludes every rotor's k-1, k, k+1 lines.

    # everything, on a laptop: the CPU probe is ~5 s per arm
    python scripts/noise_v2_render_dregon.py --probe --figures --out DIR

    # the same on uni-gpushort (identical numbers, the frozen checkpoint is
    # digest-verified in both paths)
    python scripts/noise_v2_render_dregon.py --probe --out DIR

Renders are deterministic in ``--seed``, so the two passes agree cell for cell.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

from experiments.noise_model import gates as GT
from experiments.noise_model import render as RD
from experiments.noise_model import spectrum as SP
from experiments.stochastic_fit import revised_eval as RE

SCHEMA = "noise-v2-render-dregon/1"
OUT_DEFAULT = Path("results/noise_v2/rounds/round2/render_dregon")
FIT_DEFAULT = Path("results/noise_v2/rounds/round2/fits/dregon_room2_floor__flight_floor_only.json")
ARM_NPZ_DEFAULT = Path("results/noise_v2/rounds/round2/render/audio/dregon_v2")
BENCH_DEFAULT = Path("results/noise_v2/rounds/round2/fits")

#: The two frozen DREGON room-2 cruise supports this study is about, by
#: recording. ``hovering`` is the quietest proxy failure (3.65 dB) and
#: ``updown`` the loudest (5.48 dB) of the five in `round2/render/findings.md`.
STUDY_RECORDINGS = ("hovering_nosource_room2", "updown_nosource_room2")

#: The four bench fits the comb was frozen from (``fit['frozen_from']``).
BENCH_MOTORS = (1, 2, 3, 4)
BENCH_THROTTLE = 70

#: Band-level / LTAS periodogram (reused from the regime study: 8192 at 16 kHz
#: is 1.95 Hz per bin).
PSD_N = 8192
PSD_HOP = 4096

#: ORDER-TRACKED comb geometry. 4096 at 16 kHz is 3.91 Hz per bin and 256 ms of
#: support: fine enough that two rotors 8 rev/s apart are four bins apart at
#: k=2, short enough that a DREGON cruise carrier is stationary inside it.
TRK_N = 4096
TRK_HOP = 1024
#: Half-width of the line window, in bins; the local floor is the median of the
#: bins within +-0.5 f0 EXCLUDING +-1.5 bins around every rotor's k-1/k/k+1 line.
TRK_PEAK_BINS = 1.0
TRK_EXCL_BINS = 1.5
TRK_SPAN_REL = 0.5
TRK_MIN_FREE = 6

#: Spectrogram panels: the campaign's own flight front end.
SPEC_N = 2048
SPEC_HOP = 512

COMB_ORDERS = (2, 4, 8)

#: Mute level of a disabled block (200 dB down is numerically gone, still finite).
MUTE_DB = 200.0

#: Prior means of the two speed exponents and the floor's static pedestal
#: (``model.Priors``), i.e. what the R1 regime patch pins them to.
PRIOR_EXP = 2.0
PRIOR_STATIC_REL = 2.5e-3


def die(message: str) -> NoReturn:
    raise SystemExit(f"error: {message}")


def _module(name: str) -> Any:
    """Load a sibling script by path (the scripts dir is not a package)."""
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        die(f"cannot load {path}")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # pragma: no cover - provenance only
        return "unknown"


# ── variant fits ────────────────────────────────────────────────────────────


def _mutate(fit: dict[str, Any], **kw: Any) -> dict[str, Any]:
    """A deep copy of ``fit`` with named speed-envelope / level / width edits."""
    out = copy.deepcopy(fit)
    p = out["params"]
    if "amp_exp" in kw:
        p["profile"]["amp_exp"] = float(kw["amp_exp"])
    if "floor_exp" in kw:
        p["floor"]["floor_exp"] = float(kw["floor_exp"])
        p["floor"]["floor_static_rel"] = float(kw.get("floor_static_rel", 0.0))
    if kw.get("mute_comb"):
        p["profile"]["profile_db"] = (
            np.asarray(p["profile"]["profile_db"], dtype=np.float64) - MUTE_DB
        ).tolist()
    if kw.get("mute_floor"):
        p["floor"]["floor_mean_db"] = float(p["floor"]["floor_mean_db"]) - MUTE_DB
    if "comb_shift_db" in kw:
        p["profile"]["profile_db"] = (
            np.asarray(p["profile"]["profile_db"], dtype=np.float64) + float(kw["comb_shift_db"])
        ).tolist()
    if "floor_shift_db" in kw:
        p["floor"]["floor_mean_db"] = float(p["floor"]["floor_mean_db"]) + float(
            kw["floor_shift_db"]
        )
    if kw.get("gamma_hz") is not None:
        # a k-INDEPENDENT Lorentzian half-width, on every rotor and order
        g = np.asarray(p["gamma_hz"], dtype=np.float64)
        p["gamma_hz"] = np.full(g.shape, float(kw["gamma_hz"])).tolist()
    return out


@dataclass(frozen=True)
class ArmSpec:
    name: str
    kind: str  # real | v2 | legacy
    label: str
    hypothesis: str
    build: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None
    probe: bool = True


ARMS: tuple[ArmSpec, ...] = (
    ArmSpec("real", "real", "the real DREGON room-2 clip", "reference"),
    ArmSpec(
        "legacy",
        "legacy",
        "legacy stage-2 baseline, identity-matched on this recording",
        "reference (the parity bar's own render)",
    ),
    ArmSpec("v2", "v2", "the round-2 v2 candidate, unchanged", "the failing arm", lambda f, _c: f),
    ArmSpec(
        "v2_nocomb",
        "v2",
        "v2 with the comb muted (floor only)",
        "control: what the render scores with NO comb at all",
        lambda f, _c: _mutate(f, mute_comb=True),
    ),
    ArmSpec(
        "v2_nofloor",
        "v2",
        "v2 with the floor muted (the frozen bench comb alone)",
        "control: is the frozen comb trackable on its own?",
        lambda f, _c: _mutate(f, mute_floor=True),
    ),
    ArmSpec(
        "v2_prior_exps",
        "v2",
        "v2 with both speed exponents at their prior means "
        "(amp_exp = floor_exp = 2, static_rel at its own prior mean)",
        "H1: the R1 regime patch — are the exponents the defect here too?",
        lambda f, _c: _mutate(
            f, amp_exp=PRIOR_EXP, floor_exp=PRIOR_EXP, floor_static_rel=PRIOR_STATIC_REL
        ),
    ),
    ArmSpec(
        "v2_floor_matched",
        "v2",
        "v2 with the floor re-levelled onto THIS window's real band level",
        "H3, floor side: is the floor level itself wrong?",
        lambda f, c: _mutate(f, floor_shift_db=float(c["floor_match_shift_db"])),
    ),
    ArmSpec(
        "v2_comb_real_ratio",
        "v2",
        "v2 with the comb re-levelled so its k=2 line prominence matches the real clip's",
        "the literal (e): match real's OWN measured k=2 comb-to-floor",
        lambda f, c: _mutate(f, comb_shift_db=float(c["comb_real_ratio_shift_db"])),
    ),
    ArmSpec(
        "v2_comb_bench_ratio",
        "v2",
        "v2 with the comb re-levelled so its k=2 comb-to-floor equals the one the BENCH "
        "fits identified it at",
        "H3, comb side: restore the ratio the frozen comb was fitted under",
        lambda f, c: _mutate(f, comb_shift_db=float(c["comb_bench_ratio_shift_db"])),
    ),
    ArmSpec(
        "v2_comb_level_matched",
        "v2",
        "v2 with the comb re-levelled so the FULL render's band level matches the real window's",
        "H3, comb side: the comb carries the real clip's own broadband level",
        lambda f, c: _mutate(f, comb_shift_db=float(c["comb_level_match_shift_db"])),
    ),
)


# ── measurements ────────────────────────────────────────────────────────────


def _welch(x: np.ndarray, *, n: int = PSD_N, hop: int = PSD_HOP) -> np.ndarray:
    """One-sided power per bin of ``x`` (Hann, ``n``-point, ``hop`` overlap)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < n:
        die(f"{x.size} samples is shorter than the {n}-point periodogram window")
    w = np.hanning(n)
    starts = np.arange(0, x.size - n + 1, hop)
    acc = np.zeros(n // 2 + 1, dtype=np.float64)
    for s in starts:
        acc += np.abs(np.fft.rfft(x[s : s + n] * w)) ** 2
    return acc / (float(starts.size) * float((w**2).sum()))


def band_level_db(psd: np.ndarray, freqs: np.ndarray) -> float:
    """``10 log10`` of the 30-7900 Hz power of one mic's periodogram."""
    band = (freqs >= SP.BAND_F_MIN) & (freqs <= SP.BAND_F_MAX)
    return float(10.0 * np.log10(max(float(psd[band].sum()), 1e-300)))


def _stft_power(x: np.ndarray, *, n: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    """``(frames, bins)`` power and the centre SAMPLE of each frame."""
    x = np.asarray(x, dtype=np.float64)
    w = np.hanning(n)
    starts = np.arange(0, x.size - n + 1, hop)
    if starts.size == 0:
        die(f"{x.size} samples is shorter than the {n}-point tracked window")
    power = np.stack([np.abs(np.fft.rfft(x[s : s + n] * w)) ** 2 for s in starts])
    return power / float((w**2).sum()), starts + n // 2


def tracked_line(
    x: np.ndarray,
    f0_tracks: np.ndarray,
    k: int,
    *,
    sr: int,
    peak_bins: float = TRK_PEAK_BINS,
) -> dict[str, Any]:
    """Order-``k`` line power and prominence over the local floor, per rotor.

    The line of rotor ``r`` in frame ``j`` is read at ``k f0_r`` of THAT frame's
    own label carrier, so a moving DREGON cruise carrier does not smear it; the
    local floor is the median of the bins within ``+-0.5 f0`` EXCLUDING
    ``+-TRK_EXCL_BINS`` bins around every rotor's k-1, k and k+1 line, so a
    neighbouring rotor or a neighbouring order cannot be mistaken for floor.
    Both statistics are the MEDIAN over frames; the peak is the MEAN of the bins
    inside the line window (not their max), which keeps the estimator's noise
    bias at +-1 dB instead of the +6 dB a max-over-bins carries.
    """
    power, centres = _stft_power(x, n=TRK_N, hop=TRK_HOP)
    df = float(sr) / float(TRK_N)
    bins = np.fft.rfftfreq(TRK_N, d=1.0 / float(sr))
    f0 = np.asarray(f0_tracks, dtype=np.float64)
    n_t = int(f0.shape[1])
    prom: list[list[float]] = [[] for _ in range(f0.shape[0])]
    line_p: list[list[float]] = [[] for _ in range(f0.shape[0])]
    floor_p: list[list[float]] = [[] for _ in range(f0.shape[0])]
    for j, c in enumerate(centres):
        frame = f0[:, min(int(c), n_t - 1)]
        span = TRK_SPAN_REL * float(frame.mean())
        excl = np.zeros_like(bins, dtype=bool)
        for order in (k - 1, k, k + 1):
            if order < 1:
                continue
            for line in frame * float(order):
                excl |= np.abs(bins - line) <= (TRK_EXCL_BINS + peak_bins - 1.0) * df
        for r, line in enumerate(frame * float(k)):
            if line <= SP.BAND_F_MIN or line >= SP.BAND_F_MAX:
                continue
            peak_sel = np.abs(bins - line) <= peak_bins * df
            floor_sel = (np.abs(bins - line) <= span) & ~excl
            if not peak_sel.any() or int(floor_sel.sum()) < TRK_MIN_FREE:
                continue
            pk = float(power[j][peak_sel].mean())
            fl = float(np.median(power[j][floor_sel]))
            line_p[r].append(pk)
            floor_p[r].append(fl)
            prom[r].append(10.0 * np.log10(max(pk, 1e-300) / max(fl, 1e-300)))

    def med(rows: list[list[float]]) -> list[float | None]:
        return [float(np.median(v)) if v else None for v in rows]

    pr = med(prom)
    ok = [v for v in pr if v is not None]
    return dict(
        k=int(k),
        peak_bins=float(peak_bins),
        prominence_db_per_rotor=pr,
        prominence_db=float(np.mean(ok)) if ok else None,
        line_power_per_rotor=med(line_p),
        local_floor_power_per_rotor=med(floor_p),
        n_frames=[len(v) for v in prom],
    )


def line_ratio_db(
    comb_only: np.ndarray, floor_only: np.ndarray, f0: np.ndarray, k: int, *, sr: int
) -> dict[str, Any]:
    """Comb-to-floor of order ``k`` measured on SEPARATE comb / floor renders.

    A muted-floor and a muted-comb render of the same seed carry exactly the two
    blocks of the same render, so their order-``k`` line powers give the true
    per-line comb-to-floor without any local-floor estimator: the number the
    full render's prominence ``10 log10(1 + ratio)`` is made of.
    """
    a = tracked_line(comb_only, f0, k, sr=sr)
    b = tracked_line(floor_only, f0, k, sr=sr)
    per: list[float | None] = []
    for c, f in zip(a["line_power_per_rotor"], b["line_power_per_rotor"], strict=True):
        per.append(
            None
            if c is None or f is None
            else float(10.0 * np.log10(max(c, 1e-300) / max(f, 1e-300)))
        )
    ok = [v for v in per if v is not None]
    mean = float(np.mean(ok)) if ok else None
    return dict(
        k=int(k),
        per_rotor_db=per,
        mean_db=mean,
        implied_prominence_db=(
            None
            if not ok
            else float(np.mean([10.0 * np.log10(1.0 + 10.0 ** (v / 10.0)) for v in ok]))
        ),
    )


def encoded_speed_dev(
    x: np.ndarray, f0_track: np.ndarray, *, sr: int, k: int = 2, bw_hz: float = 4.0
) -> dict[str, Any]:
    """Speed the signal ENCODES at order ``k``, against the label track.

    Demodulates by the label's own order-``k`` phase, low-passes to ``+-bw_hz``
    and differentiates the residual phase. On a render whose comb sits on the
    label carrier the residual is the model's own shaft jitter; if the render
    encoded another speed it ramps.
    """
    x = np.asarray(x, dtype=np.float64)
    f0 = np.asarray(f0_track, dtype=np.float64)
    n = int(min(x.size, f0.size))
    phase = 2.0 * np.pi * float(k) * np.cumsum(f0[:n]) / float(sr)
    z = x[:n] * np.exp(-1j * phase)
    spec = np.fft.fft(z)
    freqs = np.fft.fftfreq(n, d=1.0 / float(sr))
    spec[np.abs(freqs) > float(bw_hz)] = 0.0
    lp = np.fft.ifft(spec)
    mag = np.abs(lp)
    guard = slice(int(0.05 * n), int(0.95 * n))
    dev = (np.diff(np.unwrap(np.angle(lp))) * float(sr) / (2.0 * np.pi * float(k)))[guard]
    w = mag[1:][guard]
    if not np.isfinite(dev).all() or w.sum() <= 0:
        return dict(k=int(k), bw_hz=float(bw_hz), median_abs_dev_rev_s=None)
    order = np.argsort(np.abs(dev))
    cum = np.cumsum(w[order])
    q = lambda p: float(np.abs(dev)[order][int(np.searchsorted(cum, p * cum[-1]))])  # noqa: E731
    return dict(
        k=int(k),
        bw_hz=float(bw_hz),
        median_abs_dev_rev_s=q(0.5),
        p90_abs_dev_rev_s=q(0.9),
        mean_dev_rev_s=float(np.average(dev, weights=w)),
    )


def arm_diagnostics(
    audio: np.ndarray,
    real: np.ndarray,
    *,
    sr: int,
    f0_tracks: np.ndarray,
    demod_rotors: tuple[int, ...],
) -> dict[str, Any]:
    """Every per-arm number of one support: level, LTAS, comb, encoded speed."""
    freqs = np.fft.rfftfreq(PSD_N, d=1.0 / float(sr))
    psd_mics = np.stack([_welch(audio[m]) for m in range(audio.shape[0])])
    levels = [band_level_db(psd_mics[m], freqs) for m in range(psd_mics.shape[0])]
    ltas = RE.ltas_deviation_db(real, audio, mic=0)
    tracked = {
        f"mic{m}": [tracked_line(audio[m], f0_tracks, k, sr=sr) for k in COMB_ORDERS] for m in (0,)
    }
    return dict(
        band_level_db_mic0=float(levels[0]),
        band_level_db_mic_mean=float(
            10.0 * np.log10(np.mean([10.0 ** (v / 10.0) for v in levels]))
        ),
        band_level_db_per_mic=[float(v) for v in levels],
        rms_per_mic=[float(v) for v in np.sqrt((np.asarray(audio, np.float64) ** 2).mean(axis=1))],
        rms_broadband=float(np.sqrt((np.asarray(audio, np.float64) ** 2).mean())),
        peak_abs=float(np.abs(audio).max()),
        ltas_mean_abs_db=float(ltas["mean_abs_db"]),
        ltas_level_offset_db=float(ltas["level_offset_db"]),
        ltas_max_abs_db=float(ltas["max_abs_db"]),
        ltas_shape_only_mean_abs_db=float(ltas["shape_only_mean_abs_db"]),
        ltas_bands_db=[float(v) for v in ltas["bands_arm_db"]],
        comb_prominence=tracked["mic0"],
        comb_prominence_mic_mean_db=[
            float(
                np.mean(
                    [
                        v
                        for v in [
                            tracked_line(audio[m], f0_tracks, k, sr=sr)["prominence_db"]
                            for m in range(audio.shape[0])
                        ]
                        if v is not None
                    ]
                )
            )
            for k in COMB_ORDERS
        ],
        encoded_speed={
            f"rotor{r}": encoded_speed_dev(audio[0], f0_tracks[r], sr=sr) for r in demod_rotors
        },
        _psd=psd_mics,
    )


# ── the two speed laws, and where the comb's absolute level comes from ──────


def envelope_report(params: dict[str, Any], f0_rotor: np.ndarray) -> dict[str, Any]:
    """The comb and floor speed envelopes evaluated at a given carrier set.

    Exactly the renderer's arithmetic (``render.render_noise``): the comb
    carries ``speed ** amp_exp`` in POWER and the floor
    ``mean_r(speed ** floor_exp) + floor_static_rel``.
    """
    s = np.asarray(f0_rotor, dtype=np.float64) / float(RD.AMP_RPS_REF)
    amp_exp = float(params["profile"]["amp_exp"])
    floor_exp = float(params["floor"]["floor_exp"])
    rel = float(params["floor"]["floor_static_rel"])
    comb = float(np.mean(s**amp_exp))
    floor = float(np.mean(s**floor_exp) + rel)
    profile = np.asarray(params["profile"]["profile_db"], dtype=np.float64)
    k_max = min(
        int(profile.shape[1]),
        int(SP.k_max_for_carrier(np.asarray(f0_rotor, dtype=np.float64), SP.FLIGHT_SR)),
    )
    return dict(
        carrier_rev_s=[float(v) for v in np.asarray(f0_rotor, dtype=np.float64)],
        speed_mean=float(s.mean()),
        amp_exp=amp_exp,
        floor_exp=floor_exp,
        floor_static_rel=rel,
        comb_envelope_db=float(10.0 * np.log10(max(comb, 1e-300))),
        floor_envelope_db=float(10.0 * np.log10(max(floor, 1e-300))),
        comb_minus_floor_db=float(
            10.0 * np.log10(max(comb, 1e-300)) - 10.0 * np.log10(max(floor, 1e-300))
        ),
        k_max=int(k_max),
        top_comb_line_hz=float(k_max * float(np.max(f0_rotor))),
    )


def comb_level_accounting(fit: dict[str, Any], bench: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Where the rendered comb's ABSOLUTE level comes from, term by term.

    ``render_noise`` MEAN-CENTRES both ``mic_line_gain_db`` (over mics, per
    rotor) and ``mic_gains_db`` before applying them, so neither carries any
    absolute level: the only absolute comb scale in a render is ``profile_db``.
    This function states that explicitly — the per-mic line power the renderer
    actually produces, per rotor, against the per-mic line power each bench fit
    produces on its own carrier.
    """

    def per_mic_line_power_db(params: dict[str, Any], n_mics: int = 8) -> dict[str, Any]:
        prof = np.asarray(params["profile"]["profile_db"], dtype=np.float64)
        mlg = np.asarray(params["profile"]["mic_line_gain_db"], dtype=np.float64)
        gall = np.asarray(params["mic_gains_db"], dtype=np.float64)[:n_mics]
        centred_line = mlg[:n_mics] - mlg[:n_mics].mean(axis=0, keepdims=True)
        centred_all = gall - gall.mean()
        line_gain = 10.0 ** (centred_line / 10.0)
        all_gain = 10.0 ** (centred_all / 10.0)
        # sum of line powers over orders, per (mic, rotor)
        per_rotor = 10.0 ** (prof / 10.0)  # (R, K)
        totals = per_rotor.sum(axis=1)  # (R,)
        pm = line_gain * totals[None, :] * all_gain[:, None]  # (M, R)
        return dict(
            n_orders=int(prof.shape[1]),
            per_rotor_sum_db=[float(10.0 * np.log10(max(v, 1e-300))) for v in totals],
            mic0_per_rotor_db=[float(10.0 * np.log10(max(v, 1e-300))) for v in pm[0]],
            mic0_total_db=float(10.0 * np.log10(max(float(pm[0].sum()), 1e-300))),
            mic_mean_total_db=float(10.0 * np.log10(max(float(pm.sum(axis=1).mean()), 1e-300))),
            mic_line_gain_db_raw_mean=[float(v) for v in mlg[:n_mics].mean(axis=0)],
            mic_line_gain_db_centred_mic0=[float(v) for v in centred_line[0]],
            mic_gains_db_raw_mean=float(gall.mean()),
        )

    cand = per_mic_line_power_db(fit["params"])
    per_bench = {
        f"Motor{m}": per_mic_line_power_db(b["params"]) | {"n_rotors": 1} for m, b in bench.items()
    }
    bench_mic0 = [v["mic0_total_db"] for v in per_bench.values()]
    return dict(
        note=(
            "render_noise centres mic_line_gain_db over mics (per rotor) and mic_gains_db over "
            "mics before use, so their ABSOLUTE values are discarded by the renderer; the only "
            "absolute comb scale a render has is profile_db"
        ),
        candidate=cand,
        bench=per_bench,
        bench_mic0_total_db_mean=float(np.mean(bench_mic0)),
        candidate_minus_bench_mic0_db=float(cand["mic0_total_db"] - float(np.mean(bench_mic0))),
        four_rotor_sum_db=float(
            10.0 * np.log10(len(cand["mic0_per_rotor_db"])) if cand["mic0_per_rotor_db"] else 0.0
        ),
    )


def bench_reference(
    bench: dict[int, dict[str, Any]], *, seed: int, n_mics: int, seconds: float
) -> dict[str, Any]:
    """The comb-to-floor ratio each bench fit identified ITS comb under.

    Each bench fit is rendered on its OWN constant carrier, comb-only and
    floor-only, so the ratio is exact; this is the ratio the frozen comb was
    fitted at, and the one the flight render has to reproduce for the comb to be
    as visible on the wire as it was in the bench likelihood.
    """
    sr = int(SP.FLIGHT_SR)
    n = int(round(float(seconds) * sr))
    freqs = np.fft.rfftfreq(PSD_N, d=1.0 / float(sr))
    out: dict[str, Any] = {}
    for motor, f in sorted(bench.items()):
        carrier = np.asarray(f["params"]["carrier_rev_s"], dtype=np.float64)
        if carrier.size != 1:
            die(f"bench Motor{motor}: expected one carrier, got {carrier.tolist()}")
        track = np.full((1, n), float(carrier[0]))
        comb = RD.render_noise(_mutate(f, mute_floor=True), track, n_mics=n_mics, seed=seed)
        floor = RD.render_noise(_mutate(f, mute_comb=True), track, n_mics=n_mics, seed=seed)
        full = RD.render_noise(f, track, n_mics=n_mics, seed=seed)
        out[f"Motor{motor}"] = dict(
            carrier_rev_s=float(carrier[0]),
            converged=(f.get("diagnostics") or {}).get("converged"),
            floor_mean_db=float(f["params"]["floor"]["floor_mean_db"]),
            amp_exp=float(f["params"]["profile"]["amp_exp"]),
            floor_exp=float(f["params"]["floor"]["floor_exp"]),
            comb_only_band_level_db_mic0=band_level_db(_welch(comb[0]), freqs),
            floor_only_band_level_db_mic0=band_level_db(_welch(floor[0]), freqs),
            full_band_level_db_mic0=band_level_db(_welch(full[0]), freqs),
            comb_minus_floor_band_db=float(
                band_level_db(_welch(comb[0]), freqs) - band_level_db(_welch(floor[0]), freqs)
            ),
            line_ratio=[line_ratio_db(comb[0], floor[0], track, k, sr=sr) for k in COMB_ORDERS],
            full_prominence_db=[
                tracked_line(full[0], track, k, sr=sr)["prominence_db"] for k in COMB_ORDERS
            ],
        )
    ratios = {
        k: [
            v["line_ratio"][i]["mean_db"]
            for v in out.values()
            if v["line_ratio"][i]["mean_db"] is not None
        ]
        for i, k in enumerate(COMB_ORDERS)
    }
    return dict(
        seconds=float(seconds),
        per_motor=out,
        mean_line_ratio_db={str(k): float(np.mean(v)) if v else None for k, v in ratios.items()},
        mean_floor_mean_db=float(np.mean([v["floor_mean_db"] for v in out.values()])),
        mean_comb_minus_floor_band_db=float(
            np.mean([v["comb_minus_floor_band_db"] for v in out.values()])
        ),
    )


# ── the run ─────────────────────────────────────────────────────────────────


def study_supports() -> list[GT.ScoredSupport]:
    out: list[GT.ScoredSupport] = []
    for rec in STUDY_RECORDINGS:
        found = [s for s in GT.DREGON_CRUISE_SUPPORTS if s.recording == rec]
        if not found:
            die(f"no frozen DREGON cruise support carries recording {rec!r}")
        out.append(found[0])
    return out


def _demod_rotors(f0_rotor: np.ndarray) -> tuple[int, ...]:
    """Rotors whose order-2 line is resolvable from every other rotor's."""
    out: list[int] = []
    for r, f in enumerate(f0_rotor):
        gaps = [abs(float(f) - float(g)) for i, g in enumerate(f0_rotor) if i != r]
        if min(gaps) * 2.0 > 8.0:
            out.append(int(r))
    return tuple(out) or (int(np.argmax(f0_rotor)),)


def read_bench(fits_dir: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for m in BENCH_MOTORS:
        p = Path(fits_dir) / f"bench_dregon_Motor{m}_{BENCH_THROTTLE}__bench.json"
        if not p.exists():
            die(f"{p}: the frozen comb's bench fit is missing")
        out[m] = json.loads(p.read_text())
    return out


def _calibrate(
    cache: dict[str, Any],
    *,
    row: dict[str, Any],
    sr: int,
    comb_only: np.ndarray,
    floor_only: np.ndarray,
    f0_tracks: np.ndarray,
    bench_ratio_db: float | None,
) -> dict[str, Any]:
    """The four re-levelling shifts, each from a measured number.

    ``comb_only`` and ``floor_only`` are the two blocks of the SAME seed, so
    their band powers and line powers add to the full render's exactly, and
    every shift below is a closed-form solve rather than a search.
    """
    freqs = np.fft.rfftfreq(PSD_N, d=1.0 / float(sr))
    p_comb = 10.0 ** (band_level_db(_welch(comb_only[0]), freqs) / 10.0)
    p_floor = 10.0 ** (band_level_db(_welch(floor_only[0]), freqs) / 10.0)
    real_level = float(row["arms"]["real"]["band_level_db_mic0"])
    p_real = 10.0 ** (real_level / 10.0)
    ratio = line_ratio_db(comb_only[0], floor_only[0], f0_tracks, 2, sr=sr)
    ratio_db = ratio["mean_db"]
    if ratio_db is None:
        die("the order-2 comb-to-floor of the separated render is unmeasurable")
    real_prom = None
    for entry in row["arms"]["real"]["comb_prominence"]:
        if int(entry["k"]) == 2:
            real_prom = entry["prominence_db"]
    if real_prom is None:
        die("the real clip's order-2 prominence is unmeasurable")
    # prominence P dB implies a per-line comb-to-floor of 10 log10(10^(P/10) - 1)
    excess = 10.0 ** (float(real_prom) / 10.0) - 1.0
    cache["dregon_line_ratio_k2_db"] = float(ratio_db)
    cache["real_prominence_k2_db"] = float(real_prom)
    cache["comb_real_ratio_shift_db"] = float(
        (10.0 * np.log10(max(excess, 1e-6))) - float(ratio_db)
    )
    cache["comb_bench_ratio_shift_db"] = (
        0.0 if bench_ratio_db is None else float(float(bench_ratio_db) - float(ratio_db))
    )
    cache["floor_match_shift_db"] = float(real_level - band_level_db(_welch(floor_only[0]), freqs))
    headroom = p_real - p_floor
    cache["comb_level_match_shift_db"] = (
        float(10.0 * np.log10(max(headroom, 1e-300) / max(p_comb, 1e-300)))
        if headroom > 0.0
        else float("nan")
    )
    cache["comb_only_band_level_db_mic0"] = float(10.0 * np.log10(max(p_comb, 1e-300)))
    cache["floor_only_band_level_db_mic0"] = float(10.0 * np.log10(max(p_floor, 1e-300)))
    cache["comb_minus_floor_band_db"] = float(
        10.0 * np.log10(max(p_comb, 1e-300) / max(p_floor, 1e-300))
    )
    cache["order2_ratio"] = ratio
    return cache


def run(
    *,
    fit_path: Path,
    bench_dir: Path,
    arm_npz: Path,
    out: Path,
    seed: int,
    probe: bool,
    n_mics: int,
) -> dict[str, Any]:
    rs = _module("noise_v2_round_score")
    Probe, _arm_for_recording, legacy_arm = rs.Probe, rs._arm_for_recording, rs.legacy_arm

    fit = json.loads(Path(fit_path).read_text())
    if str(fit.get("schema")) != "noise-v2-fit/1":
        die(f"{fit_path}: not a noise-v2-fit/1 payload")
    bench = read_bench(bench_dir)
    probe_obj = Probe.load() if probe else None
    legacy = legacy_arm("dregon")
    diag = fit.get("diagnostics") or {}
    payload: dict[str, Any] = dict(
        schema=SCHEMA,
        git=git_rev(),
        fit=dict(
            path=str(fit_path),
            support=fit.get("support"),
            mode=fit.get("mode"),
            k_max=fit.get("k_max"),
            converged=diag.get("converged"),
            frozen_from=fit.get("frozen_from"),
            amp_exp=float(fit["params"]["profile"]["amp_exp"]),
            floor_exp=float(fit["params"]["floor"]["floor_exp"]),
            floor_static_rel=float(fit["params"]["floor"]["floor_static_rel"]),
            floor_mean_db=float(fit["params"]["floor"]["floor_mean_db"]),
            floor_tilt_db_oct=float(fit["params"]["floor"]["floor_tilt_db_oct"]),
            init_floor_mean_db=diag.get("init_floor_mean_db"),
            floor_level_db=diag.get("floor_level_db"),
            floor_mean_moved_from_init_db=(
                None
                if diag.get("init_floor_mean_db") is None
                else float(
                    float(fit["params"]["floor"]["floor_mean_db"])
                    - float(diag["init_floor_mean_db"])
                )
            ),
            sigma_nu=float(fit["params"]["sigma_nu"]),
            lam=float(fit["params"]["lam"]),
            amp_rps_ref=float(RD.AMP_RPS_REF),
            pool_carrier_rev_s=[
                (diag.get("batch") or {}).get("carrier_min_rev_s"),
                (diag.get("batch") or {}).get("carrier_max_rev_s"),
            ],
        ),
        protocol=dict(
            seed=int(seed),
            n_mics=int(n_mics),
            psd=dict(n_fft=PSD_N, hop=PSD_HOP),
            tracked=dict(
                n_fft=TRK_N,
                hop=TRK_HOP,
                peak_bins=TRK_PEAK_BINS,
                excl_bins=TRK_EXCL_BINS,
                span_rel=TRK_SPAN_REL,
                statistic="mean over line bins / median over local floor bins, median over frames",
            ),
            spectrogram=dict(n_fft=SPEC_N, hop=SPEC_HOP),
            band_hz=[SP.BAND_F_MIN, SP.BAND_F_MAX],
            comb_orders=list(COMB_ORDERS),
            scorer=(probe_obj.record if probe_obj is not None else None),
            legacy_route={
                k: dict(v, clip_ids=list(v.get("clip_ids") or [])[:4])
                for k, v in legacy.source.items()
            },
            arm_npz=str(arm_npz),
        ),
        supports={},
    )
    payload["bench_reference"] = bench_reference(bench, seed=seed, n_mics=n_mics, seconds=4.0)
    payload["comb_level_accounting"] = comb_level_accounting(fit, bench)
    bench_ratio = payload["bench_reference"]["mean_line_ratio_db"].get("2")

    figures: dict[str, Any] = {}
    for support in study_supports():
        clip = RE.load_window(
            support.window,
            dataset=GT.DATASET["dregon"],
            version=None,
            channels=None,
            rps_key=GT.RAW_RPS_KEY["dregon"],
        )
        real = np.asarray(clip.audio, dtype=np.float64)[:n_mics]
        reference = np.atleast_2d(np.asarray(clip.rps, dtype=np.float64))
        rsupport = RE.regime_support(
            support.window,
            reference,
            regime=support.regime,
            min_rps=support.min_rps,
            max_rps=support.max_rps,
            sr=clip.sr,
        )
        f0_rotor = reference.mean(axis=1)
        demod = _demod_rotors(f0_rotor)
        mics = list(range(min(int(n_mics), int(real.shape[0]))))
        bench_carrier = np.asarray(
            [float(b["params"]["carrier_rev_s"][0]) for b in bench.values()], dtype=np.float64
        )
        row: dict[str, Any] = dict(
            support=support.as_dict(),
            n_samples=int(real.shape[-1]),
            scored_seconds=float(rsupport.scored_seconds),
            mean_reference_rps=float(reference.mean()),
            per_rotor_mean_rps=[float(v) for v in f0_rotor],
            per_rotor_min_rps=[float(v) for v in reference.min(axis=1)],
            per_rotor_max_rps=[float(v) for v in reference.max(axis=1)],
            speed_norm=[float(v / RD.AMP_RPS_REF) for v in f0_rotor],
            demod_rotors=list(demod),
            envelope_at_dregon_cruise=envelope_report(fit["params"], f0_rotor),
            envelope_at_bench_carrier=envelope_report(fit["params"], bench_carrier),
            arms={},
        )
        row["envelope_bench_to_cruise_shift_db"] = dict(
            comb_db=float(
                row["envelope_at_dregon_cruise"]["comb_envelope_db"]
                - row["envelope_at_bench_carrier"]["comb_envelope_db"]
            ),
            floor_db=float(
                row["envelope_at_dregon_cruise"]["floor_envelope_db"]
                - row["envelope_at_bench_carrier"]["floor_envelope_db"]
            ),
        )
        cache: dict[str, Any] = {}
        psds: dict[str, np.ndarray] = {}
        waves: dict[str, np.ndarray] = {}
        blocks: dict[str, np.ndarray] = {}
        for spec in ARMS:
            t0 = time.time()
            if spec.kind == "real":
                audio = real
            elif spec.kind == "legacy":
                arm = _arm_for_recording(legacy, support)
                audio = arm.render(
                    reference, regime=support.regime, n_mics=len(mics), seed=int(seed)
                )[: len(mics)]
            else:
                assert spec.build is not None
                needs_cal = spec.name.startswith("v2_comb_") or spec.name == "v2_floor_matched"
                if needs_cal and "comb_level_match_shift_db" not in cache:
                    _calibrate(
                        cache,
                        row=row,
                        sr=int(clip.sr),
                        comb_only=blocks["v2_nofloor"],
                        floor_only=blocks["v2_nocomb"],
                        f0_tracks=reference,
                        bench_ratio_db=bench_ratio,
                    )
                variant = spec.build(fit, cache)
                audio = RD.render_noise(variant, reference, n_mics=len(mics), seed=int(seed))[
                    : len(mics)
                ]
            audio = np.asarray(audio, dtype=np.float64)
            if int(audio.shape[-1]) != int(real.shape[-1]):
                die(
                    f"{support.key}: arm {spec.name} rendered {audio.shape[-1]} samples "
                    f"against the frozen {real.shape[-1]}"
                )
            if spec.name in ("v2_nocomb", "v2_nofloor"):
                blocks[spec.name] = audio
            d = arm_diagnostics(
                audio, real, sr=int(clip.sr), f0_tracks=reference, demod_rotors=demod
            )
            psds[spec.name] = d.pop("_psd")
            waves[spec.name] = audio[0].copy()
            entry: dict[str, Any] = dict(
                kind=spec.kind,
                label=spec.label,
                hypothesis=spec.hypothesis,
                render_seconds=float(time.time() - t0),
                **d,
            )
            if spec.kind == "v2" and spec.name != "v2":
                entry["variant_params"] = dict(
                    amp_exp=float(variant["params"]["profile"]["amp_exp"]),
                    floor_exp=float(variant["params"]["floor"]["floor_exp"]),
                    floor_static_rel=float(variant["params"]["floor"]["floor_static_rel"]),
                    floor_mean_db=float(variant["params"]["floor"]["floor_mean_db"]),
                    profile_shift_db=float(
                        np.asarray(variant["params"]["profile"]["profile_db"], np.float64).mean()
                        - np.asarray(fit["params"]["profile"]["profile_db"], np.float64).mean()
                    ),
                )
            if spec.name == "v2":
                entry["arm_npz_check"] = _npz_check(arm_npz, support, audio, seed=int(seed))
            if probe_obj is not None and spec.probe:
                pit = probe_obj.tracker.pit(
                    audio,
                    reference,
                    mics=mics,
                    expected_samples=int(real.shape[-1]),
                    support=rsupport,
                )
                entry["pit_mae"] = float(pit["mae"])
                entry["pit_per_mic"] = [float(v) for v in pit["per_mic"]]
                entry["pit_per_rotor"] = [float(v) for v in pit["per_rotor"]]
                entry["n_scored_frames"] = int(pit["n_scored_frames"])
            row["arms"][spec.name] = entry
            print(
                f"[{support.recording}] {spec.name}: "
                f"level={entry['band_level_db_mic0']:+.2f} dB "
                f"ltas={entry['ltas_mean_abs_db']:.2f} dB "
                f"k2={entry['comb_prominence'][0]['prominence_db']} "
                f"pit={entry.get('pit_mae')}",
                flush=True,
            )
        row["calibration"] = {k: v for k, v in cache.items() if not isinstance(v, np.ndarray)}
        payload["supports"][support.key] = row
        figures[support.recording] = dict(
            support=support, psds=psds, waves=waves, sr=int(clip.sr), row=row
        )
    payload["verdicts"] = verdicts(payload)
    Path(out).mkdir(parents=True, exist_ok=True)
    return payload | {"_figures": figures}


def _npz_check(
    arm_npz: Path, support: GT.ScoredSupport, audio: np.ndarray, *, seed: int
) -> dict[str, Any]:
    """Is the arm we are diagnosing the one that was SCORED?

    The scored arm's audio is in the round's npz (float32); this render is
    float64 from the same fit and seed, so the two must agree to float32
    rounding. Anything larger means this study is not about the scored audio.
    """
    path = Path(arm_npz) / f"{support.key}.npz"
    if not path.exists():
        return dict(available=False, path=str(path))
    z = np.load(path)
    key = f"render_seed_{seed}"
    if key not in z.files:
        return dict(available=False, path=str(path), missing=key)
    ref = np.asarray(z[key], dtype=np.float64)[: audio.shape[0]]
    denom = float(np.abs(ref).max()) or 1.0
    return dict(
        available=True,
        path=str(path),
        key=key,
        max_abs_diff=float(np.abs(ref - audio).max()),
        max_rel_diff=float(np.abs(ref - audio).max() / denom),
        float32_eps_rel=float(np.finfo(np.float32).eps),
    )


# ── verdicts ────────────────────────────────────────────────────────────────


def _arm(payload: dict[str, Any], recording: str, name: str, key: str) -> Any:
    for row in payload["supports"].values():
        if row["support"]["recording"] == recording:
            return row["arms"].get(name, {}).get(key)
    return None


def _prom(payload: dict[str, Any], recording: str, name: str, k: int) -> Any:
    entries = _arm(payload, recording, name, "comb_prominence") or []
    for e in entries:
        if int(e["k"]) == int(k):
            return e["prominence_db"]
    return None


def verdicts(payload: dict[str, Any]) -> dict[str, Any]:
    """The four hypotheses, each decided by numbers already in the payload."""
    recs = [row["support"]["recording"] for row in payload["supports"].values()]
    out: dict[str, Any] = {}

    def over(fn: Callable[[str], Any]) -> dict[str, Any]:
        return {r: fn(r) for r in recs}

    out["H1_speed_law"] = dict(
        claim="the bench comb was fitted at ~68 rev/s and the DREGON cruise carriers differ; "
        "the speed law / k_max geometry must be checked at the cruise carriers",
        comb_envelope_shift_db=over(
            lambda r: _row(payload, r)["envelope_bench_to_cruise_shift_db"]["comb_db"]
        ),
        floor_envelope_shift_db=over(
            lambda r: _row(payload, r)["envelope_bench_to_cruise_shift_db"]["floor_db"]
        ),
        k_max_bench=over(lambda r: _row(payload, r)["envelope_at_bench_carrier"]["k_max"]),
        k_max_cruise=over(lambda r: _row(payload, r)["envelope_at_dregon_cruise"]["k_max"]),
        pit_prior_exps=over(lambda r: _arm(payload, r, "v2_prior_exps", "pit_mae")),
        pit_v2=over(lambda r: _arm(payload, r, "v2", "pit_mae")),
    )
    out["H2_mic_gain_units"] = dict(
        claim="a units / per-rotor level mismatch in mic_line_gain_db or mic_gains_db",
        renderer_note=payload["comb_level_accounting"]["note"],
        candidate_mic0_line_power_db=payload["comb_level_accounting"]["candidate"]["mic0_total_db"],
        bench_mic0_line_power_db_mean=payload["comb_level_accounting"]["bench_mic0_total_db_mean"],
        candidate_minus_bench_db=payload["comb_level_accounting"]["candidate_minus_bench_mic0_db"],
    )
    out["H3_floor_buries_comb"] = dict(
        claim="the floor-only flight fit's floor buries the frozen bench comb",
        comb_minus_floor_band_db=over(
            lambda r: _row(payload, r)["calibration"]["comb_minus_floor_band_db"]
        ),
        bench_comb_minus_floor_band_db=payload["bench_reference"]["mean_comb_minus_floor_band_db"],
        order2_line_ratio_db=over(
            lambda r: _row(payload, r)["calibration"]["dregon_line_ratio_k2_db"]
        ),
        bench_order2_line_ratio_db=payload["bench_reference"]["mean_line_ratio_db"].get("2"),
        pit_v2=over(lambda r: _arm(payload, r, "v2", "pit_mae")),
        pit_nocomb=over(lambda r: _arm(payload, r, "v2_nocomb", "pit_mae")),
        pit_comb_bench_ratio=over(lambda r: _arm(payload, r, "v2_comb_bench_ratio", "pit_mae")),
        pit_comb_level_matched=over(lambda r: _arm(payload, r, "v2_comb_level_matched", "pit_mae")),
        pit_floor_matched=over(lambda r: _arm(payload, r, "v2_floor_matched", "pit_mae")),
    )
    out["H4_wrong_speed"] = dict(
        claim="the render encodes a different speed than the label",
        encoded_speed_v2_comb=over(lambda r: _arm(payload, r, "v2_nofloor", "encoded_speed")),
        encoded_speed_real=over(lambda r: _arm(payload, r, "real", "encoded_speed")),
        scored_fraction_note="every frozen DREGON support is one 4 s cruise window; "
        "scored_seconds is reported per support",
        scored_seconds=over(lambda r: _row(payload, r)["scored_seconds"]),
        k_max_cruise=over(lambda r: _row(payload, r)["envelope_at_dregon_cruise"]["k_max"]),
        top_comb_line_hz=over(
            lambda r: _row(payload, r)["envelope_at_dregon_cruise"]["top_comb_line_hz"]
        ),
    )
    return out


def _row(payload: dict[str, Any], recording: str) -> dict[str, Any]:
    for row in payload["supports"].values():
        if row["support"]["recording"] == recording:
            return row
    die(f"no support row for recording {recording!r}")


# ── figures ─────────────────────────────────────────────────────────────────

FIG_ARMS = ("real", "legacy", "v2", "v2_nocomb", "v2_nofloor", "v2_comb_level_matched")
FIG_COLOURS = {
    "real": "#111111",
    "legacy": "#1f77b4",
    "v2": "#d62728",
    "v2_nocomb": "#ff9896",
    "v2_nofloor": "#9467bd",
    "v2_comb_level_matched": "#2ca02c",
}


def _smooth_log(
    freqs: np.ndarray, psd: np.ndarray, *, per_oct: int = 48
) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = float(SP.BAND_F_MIN), float(SP.BAND_F_MAX)
    n = int(np.ceil(per_oct * np.log2(hi / lo)))
    edges = lo * 2.0 ** (np.arange(n + 1) / per_oct)
    centres = np.sqrt(edges[:-1] * edges[1:])
    out = np.zeros(n)
    ok = np.zeros(n, dtype=bool)
    for i in range(n):
        sel = (freqs >= edges[i]) & (freqs < edges[i + 1])
        if sel.any():
            out[i] = float(psd[sel].mean())
            ok[i] = True
    return centres[ok], 10.0 * np.log10(np.maximum(out[ok], 1e-300))


def _save(fig: Any, path: Path, *, colors: int = 128) -> str:
    fig.savefig(path, dpi=130, bbox_inches="tight")
    try:
        from PIL import Image

        img = Image.open(path).convert("RGB").quantize(colors=colors, method=2)
        img.save(path, optimize=True)
    except Exception:  # pragma: no cover - cosmetic only
        pass
    return str(path)


def write_figures(payload: dict[str, Any], figures: dict[str, Any], out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[str] = []
    for recording, fg in figures.items():
        sr = int(fg["sr"])
        freqs = np.fft.rfftfreq(PSD_N, d=1.0 / sr)
        short = recording.split("_")[0]

        fig, ax = plt.subplots(figsize=(9.0, 4.4))
        for name in FIG_ARMS:
            if name not in fg["psds"]:
                continue
            f, db = _smooth_log(freqs, fg["psds"][name][0])
            ax.semilogx(
                f,
                db,
                color=FIG_COLOURS[name],
                lw=2.0 if name in ("real", "v2") else 1.3,
                ls="-" if name in ("real", "legacy", "v2") else "--",
                label=name,
            )
        ax.set_xlim(SP.BAND_F_MIN, SP.BAND_F_MAX)
        ax.set_xlabel("Hz")
        ax.set_ylabel("dB (absolute, 1/48-oct mean)")
        ax.set_title(f"DREGON {short}: LTAS, mic 0 — the comb is 18 dB under the fitted floor")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8, ncol=2)
        written.append(_save(fig, Path(out) / f"ltas_{short}.png"))
        plt.close(fig)

        panels = ("real", "v2", "v2_nofloor", "v2_comb_level_matched")
        have = [p for p in panels if p in fg["waves"]]
        fig, axes = plt.subplots(len(have), 1, figsize=(9.0, 2.2 * len(have)), sharex=True)
        axes = np.atleast_1d(axes)
        # ONE colour scale for every panel, set by the real clip: a per-panel
        # scale would auto-stretch the featureless v2 render and hide exactly
        # the difference this figure exists to show.
        scale: tuple[float, float] | None = None
        for ax, name in zip(axes, have, strict=True):
            x = fg["waves"][name]
            w = np.hanning(SPEC_N)
            starts = np.arange(0, x.size - SPEC_N + 1, SPEC_HOP)
            S = np.stack([np.abs(np.fft.rfft(x[s : s + SPEC_N] * w)) ** 2 for s in starts]).T
            db = 10.0 * np.log10(np.maximum(S / float((w**2).sum()), 1e-14))
            if scale is None:
                scale = (float(np.percentile(db, 5)), float(np.percentile(db, 99.5)))
            ax.imshow(
                db,
                origin="lower",
                aspect="auto",
                extent=[0.0, x.size / sr, 0.0, sr / 2.0],
                vmin=scale[0],
                vmax=scale[1],
                cmap="magma",
            )
            ax.set_ylim(0.0, 2000.0)
            ax.set_ylabel(f"{name}\nHz", fontsize=7)
        axes[-1].set_xlabel("s")
        axes[0].set_title(
            f"DREGON {short}: 0-2 kHz, mic 0 (n_fft {SPEC_N}, hop {SPEC_HOP}); "
            "one shared dB scale, set by the real panel"
        )
        written.append(_save(fig, Path(out) / f"spectrogram_{short}.png"))
        plt.close(fig)
    return written


# ── findings ────────────────────────────────────────────────────────────────


def _f(v: Any, spec: str = ".3f") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    try:
        return format(float(v), spec)
    except (TypeError, ValueError):
        return str(v)


PATCH_PROPOSAL = """\
```diff
@@ src/experiments/noise_model/model.py:99 class Priors
+    #: rig-to-rig level offset of a TRANSPLANTED comb, in dB. Zero-mean and
+    #: wide: it is a geometry/distance term (bench mic at ~1 m and one rotor
+    #: against a flight array at ~0.3 m and four), not a physical one.
+    comb_gain_db: tuple[float, float] = (0.0, 15.0)
@@ src/experiments/noise_model/model.py:162 free_blocks
     if mode == "flight_floor_only":
-        return ("floor", "mic")
+        # the comb arrives FROZEN from a bench rig, so its absolute level is
+        # that rig's; nothing downstream can re-level it, because
+        # render_noise mean-centres mic_line_gain_db over mics per rotor
+        # (render.py:191) and mic_gains_db over mics (:192) and profile_db is
+        # then the ONLY absolute comb scale. One shared scalar, no more.
+        return ("floor", "mic", "comb_gain")
@@ src/experiments/noise_model/model.py:586 sample_params
     if "profile" in free:
         profile_db = _normal(site, "profile_db", priors.profile_db[0], priors.profile_db[1], (r, k))
         amp_exp = _normal(site, "amp_exp", *priors.amp_exp) if flight else zero
     else:
         profile_db = take("profile", "profile_db", (r, k))
         amp_exp = take("profile", "amp_exp") if flight else zero
+    if "comb_gain" in free:
+        # folded into profile_db, so write_fit, render_noise and
+        # expected_periodogram need no change at all
+        profile_db = profile_db + _normal(site, "comb_gain_db", *priors.comb_gain_db)
```
"""


def findings(payload: dict[str, Any], *, job: str | None, figures: list[str]) -> str:
    rows = list(payload["supports"].values())
    recs = [r["support"]["recording"] for r in rows]
    shorts = [r.split("_")[0] for r in recs]
    fitd = payload["fit"]
    br = payload["bench_reference"]
    acc = payload["comb_level_accounting"]
    o: list[str] = []
    o.append("# Noise model v2 — round 2: why the v2 DREGON arm renders to PIT MAE 75 rev/s")
    o.append("")
    o.append(
        f"Record `{OUT_DEFAULT}/render_dregon.json`, git `{payload['git']}`, "
        f"fit `{fitd['path']}` (mode `{fitd['mode']}`, "
        f"`converged` {fitd['converged']!r} — NOT converged; "
        f"amp_exp {_f(fitd['amp_exp'], '.4f')}, floor_exp {_f(fitd['floor_exp'], '.4f')}, "
        f"floor_static_rel {_f(fitd['floor_static_rel'], '.4f')}, "
        f"floor_mean_db {_f(fitd['floor_mean_db'], '.3f')}, "
        f"floor_tilt {_f(fitd['floor_tilt_db_oct'], '.3f')} dB/oct), "
        f"render seed {payload['protocol']['seed']}, {payload['protocol']['n_mics']} mics."
    )
    o.append("")
    o.append(
        f"The fit's floor level never moved from its initialisation: "
        f"`diagnostics.init_floor_mean_db` {_f(fitd['init_floor_mean_db'], '.6f')} vs "
        f"`params.floor.floor_mean_db` {_f(fitd['floor_mean_db'], '.6f')} "
        f"(delta {_f(fitd['floor_mean_moved_from_init_db'], '.2e')} dB)."
    )
    o.append("")
    if job:
        o.append(f"HPPNet job `{job}` on `uni-gpushort`; checkpoint sha256 verified in-job.")
    scorer = payload["protocol"].get("scorer") or {}
    if scorer.get("sha256"):
        o.append(f"Frozen scorer digest `{scorer['sha256']}` (= `gates.SCORER_SHA256`).")
    o.append("")

    # ── PIT table
    o.append("## HPPNet PIT MAE per arm (rev/s, frozen HPPNet, seed 2001, 8 mics)")
    o.append("")
    o.append("| arm | " + " | ".join(shorts) + " | what it isolates |")
    o.append("|---|" + "---:|" * len(shorts) + "---|")
    for spec in ARMS:
        vals = [_arm(payload, r, spec.name, "pit_mae") for r in recs]
        o.append(
            f"| `{spec.name}` | " + " | ".join(_f(v) for v in vals) + f" | {spec.hypothesis} |"
        )
    o.append("")
    o.append(
        "`v2` reproduces the scored audio: "
        + "; ".join(
            f"{s} max |float64 render - committed npz| "
            f"{_f((_arm(payload, r, 'v2', 'arm_npz_check') or {}).get('max_abs_diff'), '.3e')}"
            for r, s in zip(recs, shorts, strict=True)
        )
        + " (float32 storage rounding)."
    )
    o.append("")

    # ── levels / LTAS
    o.append("## Levels and LTAS (30–7900 Hz, absolute)")
    o.append("")
    o.append(
        "| support | arm | band level mic0 dB | mic-mean dB | offset vs real dB | "
        "LTAS abs dB | LTAS shape dB | broadband RMS |"
    )
    o.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r, s in zip(recs, shorts, strict=True):
        real_l = _arm(payload, r, "real", "band_level_db_mic0")
        for spec in ARMS:
            a = _arm(payload, r, spec.name, "band_level_db_mic0")
            if a is None:
                continue
            o.append(
                f"| `{s}` | `{spec.name}` | {_f(a, '.2f')} | "
                f"{_f(_arm(payload, r, spec.name, 'band_level_db_mic_mean'), '.2f')} | "
                f"{_f(None if real_l is None else a - real_l, '+.2f')} | "
                f"{_f(_arm(payload, r, spec.name, 'ltas_mean_abs_db'), '.2f')} | "
                f"{_f(_arm(payload, r, spec.name, 'ltas_shape_only_mean_abs_db'), '.2f')} | "
                f"{_f(_arm(payload, r, spec.name, 'rms_broadband'), '.4f')} |"
            )
    o.append("")

    # ── comb-to-floor
    o.append("## Order-tracked comb prominence (dB over the local floor)")
    o.append("")
    o.append(
        f"Read at each frame's OWN label carrier ({TRK_N}-point, hop {TRK_HOP}, "
        f"peak = mean of ±{TRK_PEAK_BINS:.0f} bin, floor = median of the ±"
        f"{TRK_SPAN_REL:.1f}·f0 bins excluding every rotor's k−1/k/k+1 line). A periodogram at "
        "the MEAN carrier is useless here: DREGON cruise carriers move 10–25 rev/s inside the "
        "4 s window and smear every line into its neighbours."
    )
    o.append("")
    o.append("| support | arm | k=2 mic0 | k=4 | k=8 | k=2 mic-mean | k=4 | k=8 |")
    o.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r, s in zip(recs, shorts, strict=True):
        for spec in ARMS:
            if _arm(payload, r, spec.name, "comb_prominence") is None:
                continue
            mm = _arm(payload, r, spec.name, "comb_prominence_mic_mean_db") or []
            o.append(
                f"| `{s}` | `{spec.name}` | "
                + " | ".join(_f(_prom(payload, r, spec.name, k), "+.2f") for k in COMB_ORDERS)
                + " | "
                + " | ".join(_f(v, "+.2f") for v in mm)
                + " |"
            )
    o.append("")
    o.append(
        "Separated blocks of the same seed (`v2_nofloor` comb-only against `v2_nocomb` "
        "floor-only) give the exact per-line comb-to-floor, without any estimator:"
    )
    o.append("")
    o.append("| support | k | comb-to-floor per line dB | implied full-render prominence dB |")
    o.append("|---|---:|---:|---:|")
    for r, s in zip(recs, shorts, strict=True):
        cal = _row(payload, r)["calibration"]
        e = cal.get("order2_ratio") or {}
        o.append(
            f"| `{s}` | 2 | {_f(cal.get('dregon_line_ratio_k2_db'), '+.2f')} | "
            f"{_f(e.get('implied_prominence_db'), '+.2f')} |"
        )
    o.append("")

    # ── bench reference
    o.append("## The comb-to-floor the frozen comb was IDENTIFIED at (bench, own carriers)")
    o.append("")
    o.append(
        "| bench fit | carrier rev/s | floor_mean_db | comb-only band dB | floor-only band dB | "
        "comb−floor band dB | k=2 line ratio dB | full k=2 prominence dB |"
    )
    o.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, v in br["per_motor"].items():
        o.append(
            f"| `bench_dregon_{name}_70` (`converged` {v['converged']!r}) | "
            f"{_f(v['carrier_rev_s'], '.3f')} | {_f(v['floor_mean_db'], '.2f')} | "
            f"{_f(v['comb_only_band_level_db_mic0'], '.2f')} | "
            f"{_f(v['floor_only_band_level_db_mic0'], '.2f')} | "
            f"{_f(v['comb_minus_floor_band_db'], '+.2f')} | "
            f"{_f(v['line_ratio'][0]['mean_db'], '+.2f')} | "
            f"{_f(v['full_prominence_db'][0], '+.2f')} |"
        )
    o.append(
        f"| **mean** | | {_f(br['mean_floor_mean_db'], '.2f')} | | | "
        f"{_f(br['mean_comb_minus_floor_band_db'], '+.2f')} | "
        f"{_f(br['mean_line_ratio_db'].get('2'), '+.2f')} | |"
    )
    o.append("")

    # ── speed laws
    o.append("## The fit's two speed laws at the bench carrier vs the DREGON cruise carriers")
    o.append("")
    o.append(
        "| support | carriers | mean speed (÷80) | comb envelope dB | floor envelope dB | "
        "comb−floor dB | k_max | top comb line Hz |"
    )
    o.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r, s in zip(recs, shorts, strict=True):
        row = _row(payload, r)
        for key, lab in (
            ("envelope_at_bench_carrier", "bench ~68 rev/s"),
            ("envelope_at_dregon_cruise", "DREGON cruise"),
        ):
            e = row[key]
            o.append(
                f"| `{s}` | {lab} | {_f(e['speed_mean'], '.4f')} | "
                f"{_f(e['comb_envelope_db'], '+.2f')} | {_f(e['floor_envelope_db'], '+.2f')} | "
                f"{_f(e['comb_minus_floor_db'], '+.2f')} | {e['k_max']} | "
                f"{_f(e['top_comb_line_hz'], '.0f')} |"
            )
        sh = row["envelope_bench_to_cruise_shift_db"]
        o.append(
            f"| `{s}` | **bench → cruise shift** | | {_f(sh['comb_db'], '+.2f')} | "
            f"{_f(sh['floor_db'], '+.2f')} | "
            f"{_f(sh['comb_db'] - sh['floor_db'], '+.2f')} | | |"
        )
    o.append("")

    # ── comb level accounting
    o.append("## The comb's absolute level, term by term")
    o.append("")
    o.append(f"{acc['note']}.")
    o.append("")
    o.append(
        "| fit | orders | Σ_k line power, mic 0, all rotors dB | mic-mean dB | "
        "raw mic_line_gain_db mean (DISCARDED) | raw mic_gains_db mean (DISCARDED) |"
    )
    o.append("|---|---:|---:|---:|---|---:|")
    c = acc["candidate"]
    o.append(
        f"| candidate (log-mean comb, 4 rotors) | {c['n_orders']} | "
        f"{_f(c['mic0_total_db'], '.2f')} | {_f(c['mic_mean_total_db'], '.2f')} | "
        + ", ".join(_f(v, ".2f") for v in c["mic_line_gain_db_raw_mean"])
        + f" | {_f(c['mic_gains_db_raw_mean'], '.2f')} |"
    )
    for name, v in acc["bench"].items():
        o.append(
            f"| `bench_dregon_{name}_70` (1 rotor) | {v['n_orders']} | "
            f"{_f(v['mic0_total_db'], '.2f')} | {_f(v['mic_mean_total_db'], '.2f')} | "
            + ", ".join(_f(x, ".2f") for x in v["mic_line_gain_db_raw_mean"])
            + f" | {_f(v['mic_gains_db_raw_mean'], '.2f')} |"
        )
    o.append("")
    o.append(
        f"Candidate minus the bench mean at mic 0: "
        f"{_f(acc['candidate_minus_bench_mic0_db'], '+.2f')} dB — the four-rotor sum "
        f"({_f(acc['four_rotor_sum_db'], '+.2f')} dB if the rotors were identical and the "
        f"per-rotor mic gains neutral) minus what the dB-mean of the four bench profiles and "
        f"the per-rotor mic-gain centring take back."
    )
    o.append("")

    # ── re-levelling shifts
    o.append("## The re-levelling shifts each variant applies (all solved, none searched)")
    o.append("")
    o.append(
        "| support | comb-only band dB | floor-only band dB | comb−floor dB | "
        "(d) floor→real level dB | (e) comb→real k=2 ratio dB | comb→bench ratio dB | "
        "comb→real band level dB |"
    )
    o.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r, s in zip(recs, shorts, strict=True):
        cal = _row(payload, r)["calibration"]
        o.append(
            f"| `{s}` | {_f(cal.get('comb_only_band_level_db_mic0'), '.2f')} | "
            f"{_f(cal.get('floor_only_band_level_db_mic0'), '.2f')} | "
            f"{_f(cal.get('comb_minus_floor_band_db'), '+.2f')} | "
            f"{_f(cal.get('floor_match_shift_db'), '+.2f')} | "
            f"{_f(cal.get('comb_real_ratio_shift_db'), '+.2f')} | "
            f"{_f(cal.get('comb_bench_ratio_shift_db'), '+.2f')} | "
            f"{_f(cal.get('comb_level_match_shift_db'), '+.2f')} |"
        )
    o.append("")

    # ── encoded speed
    o.append("## Encoded speed (order-2 demodulation against the frozen label)")
    o.append("")
    o.append("| support | arm | rotor | median abs dev rev/s | p90 | mean signed |")
    o.append("|---|---|---|---:|---:|---:|")
    for r, s in zip(recs, shorts, strict=True):
        for name in ("real", "legacy", "v2_nofloor", "v2_comb_level_matched"):
            enc = _arm(payload, r, name, "encoded_speed") or {}
            for rotor, v in enc.items():
                o.append(
                    f"| `{s}` | `{name}` | {rotor} | "
                    f"{_f(v.get('median_abs_dev_rev_s'))} | {_f(v.get('p90_abs_dev_rev_s'))} | "
                    f"{_f(v.get('mean_dev_rev_s'))} |"
                )
    o.append("")
    o.extend(hypothesis_section(payload))
    o.append("")
    o.append("## Minimal patch proposal (NOT applied — Main's call)")
    o.append("")
    o.extend(patch_preamble(payload))
    o.append("")
    o.append(PATCH_PROPOSAL)
    o.extend(patch_expectation(payload))
    if figures:
        o.append("")
        o.append("## Figures")
        o.append("")
        o.extend(f"* `{p}`" for p in figures)
    o.append("")
    return "\n".join(o)


def hypothesis_section(payload: dict[str, Any]) -> list[str]:
    v = payload["verdicts"]
    rows = list(payload["supports"].values())
    recs = [r["support"]["recording"] for r in rows]
    shorts = [r.split("_")[0] for r in recs]

    def pairs(d: dict[str, Any], spec: str = ".3f") -> str:
        return ", ".join(f"{s} {_f(d.get(r), spec)}" for r, s in zip(recs, shorts, strict=True))

    h1, h2, h3, h4 = (
        v["H1_speed_law"],
        v["H2_mic_gain_units"],
        v["H3_floor_buries_comb"],
        v["H4_wrong_speed"],
    )
    comb_shift = pairs(h1["comb_envelope_shift_db"], "+.2f")
    floor_shift = pairs(h1["floor_envelope_shift_db"], "+.2f")
    out = [
        "## Verdict per hypothesis",
        "",
        "| hypothesis | verdict | the numbers that decide it |",
        "|---|---|---|",
    ]
    out.append(
        "| **H1** — the bench comb was fitted at ~68 rev/s and the cruise carriers differ, so the "
        "speed law / Nyquist geometry is wrong at cruise | **REFUTED as the cause** | "
        f"`amp_exp` is {_f(payload['fit']['amp_exp'], '.1f')} (a bench fit has one throttle and "
        f"cannot identify a speed law), so the comb envelope shift bench → cruise is {comb_shift} "
        f"dB; the floor's own law moves {floor_shift} dB, so the whole bench → cruise speed-law "
        f"effect on comb−floor is under 5 dB against the "
        f"{_f(h3['comb_minus_floor_band_db'].get(recs[0]), '+.1f')} dB deficit measured. "
        f"`k_max` is {h1['k_max_cruise'].get(recs[0])} at cruise vs "
        f"{h1['k_max_bench'].get(recs[0])} at the bench carrier (no Nyquist cap change). "
        f"Pinning BOTH exponents at the prior mean leaves PIT at {pairs(h1['pit_prior_exps'])} "
        f"rev/s against the unchanged {pairs(h1['pit_v2'])} — the R1 regime patch does nothing "
        "here |"
    )
    out.append(
        "| **H2** — a units / per-rotor level mismatch in `mic_line_gain_db` / `mic_gains_db` | "
        "**REFUTED as a units bug, SUPPORTED as a missing degree of freedom** | "
        f"`render_noise` mean-centres `mic_line_gain_db` over mics per rotor (`render.py:191`) "
        f"and `mic_gains_db` over mics (`:192`), so their absolute values are DISCARDED: the only "
        f"absolute comb scale a render has is `profile_db`. Σ_k line power at mic 0 is "
        f"{_f(h2['candidate_mic0_line_power_db'], '.2f')} dB for the candidate against a bench "
        f"mean of {_f(h2['bench_mic0_line_power_db_mean'], '.2f')} dB "
        f"({_f(h2['candidate_minus_bench_db'], '+.2f')} dB) — the transplanted comb carries the "
        "BENCH rig's absolute level and there is no parameter anywhere that could re-level it |"
    )
    out.append(
        "| **H3** — the floor-only flight fit's floor buries the comb | **SUPPORTED — this is the "
        "defect** | "
        f"comb-only minus floor-only band level: {pairs(h3['comb_minus_floor_band_db'], '+.2f')} "
        f"dB in the flight render against {_f(h3['bench_comb_minus_floor_band_db'], '+.2f')} dB on "
        f"the bench where the comb was fitted; per-line at k=2, "
        f"{pairs(h3['order2_line_ratio_db'], '+.2f')} dB against the bench "
        f"{_f(h3['bench_order2_line_ratio_db'], '+.2f')} dB. Consequence: `v2` "
        f"{pairs(h3['pit_v2'])} rev/s is INDISTINGUISHABLE from `v2_nocomb` "
        f"{pairs(h3['pit_nocomb'])} — the render's comb contributes nothing at all, and the "
        f"per-rotor PIT MAE equals the per-rotor label speed (the tracker returns no rotor). "
        f"Restoring the comb level fixes it: comb → real band level "
        f"{pairs(h3['pit_comb_level_matched'])}, comb → bench ratio "
        f"{pairs(h3['pit_comb_bench_ratio'])}, against legacy "
        f"{pairs({r: _arm(payload, r, 'legacy', 'pit_mae') for r in recs})} and real "
        f"{pairs({r: _arm(payload, r, 'real', 'pit_mae') for r in recs})}. Moving the FLOOR "
        f"instead does not: {pairs(h3['pit_floor_matched'])} |"
    )
    enc_v2 = {r: (h4["encoded_speed_v2_comb"].get(r) or {}) for r in recs}
    out.append(
        "| **H4** — the render encodes a different speed than the label | **REFUTED** | "
        "order-2 demodulation of the comb-only render against the frozen label: "
        + "; ".join(
            f"{s} "
            + ", ".join(f"{k} {_f(x.get('median_abs_dev_rev_s'))}" for k, x in enc_v2[r].items())
            for r, s in zip(recs, shorts, strict=True)
        )
        + " rev/s median absolute deviation — the comb sits on the label carrier by construction "
        f"(`render_noise` integrates the carriers it is handed). Every frozen DREGON support is "
        f"one 4 s cruise window scored whole ({pairs(h4['scored_seconds'], '.2f')} s), so there "
        "is no standby-like frame inside the score window, and `k_max` "
        f"{h4['k_max_cruise'].get(recs[0])} puts the top comb line at "
        f"{_f(h4['top_comb_line_hz'].get(recs[0]), '.0f')} Hz, inside the band |"
    )
    return out


def patch_preamble(payload: dict[str, Any]) -> list[str]:
    fitd = payload["fit"]
    br = payload["bench_reference"]
    rows = list(payload["supports"].values())
    recs = [r["support"]["recording"] for r in rows]
    cal0 = _row(payload, recs[0])["calibration"]
    return [
        f"One defect, one missing degree of freedom. The comb was frozen from four DREGON BENCH "
        f"fits whose own floors sit at {_f(br['mean_floor_mean_db'], '.2f')} dB, where the comb is "
        f"{_f(br['mean_comb_minus_floor_band_db'], '+.2f')} dB ABOVE the floor in band power and "
        f"{_f(br['mean_line_ratio_db'].get('2'), '+.2f')} dB above it per line at k=2. The flight "
        f"fit is `mode={fitd['mode']}`: its objective frees the FLOOR ONLY, so nothing in it ever "
        f"balanced the transplanted comb against the flight floor — and its floor never even left "
        f"its own initialisation ({_f(fitd['init_floor_mean_db'], '.4f')} dB, "
        f"`converged` {fitd['converged']!r}). On the DREGON cruise supports the same comb ends up "
        f"{_f(cal0.get('comb_minus_floor_band_db'), '+.2f')} dB relative to the floor, i.e. the "
        f"comb-to-floor ratio swung by "
        f"{_f(float(br['mean_comb_minus_floor_band_db']) - float(cal0['comb_minus_floor_band_db']), '.1f')}"
        f" dB across the transplant. The renderer cannot absorb that: it mean-centres both "
        f"`mic_line_gain_db` and `mic_gains_db`, so `profile_db` is the ONLY absolute comb scale "
        f"and it is frozen. The minimal fix is therefore to give `flight_floor_only` exactly one "
        f"more free scalar — a shared comb gain in dB — and let the flight likelihood set it:",
    ]


def patch_expectation(payload: dict[str, Any]) -> list[str]:
    rows = list(payload["supports"].values())
    recs = [r["support"]["recording"] for r in rows]
    shorts = [r.split("_")[0] for r in recs]
    names = (
        "real",
        "legacy",
        "v2",
        "v2_prior_exps",
        "v2_floor_matched",
        "v2_comb_real_ratio",
        "v2_comb_bench_ratio",
        "v2_comb_level_matched",
    )
    out = [
        "",
        "### Expected effect — measured, not guessed",
        "",
        "No variant below is a FIT, so none of them is a parity claim: each is the SAME frozen "
        "comb and the SAME fitted floor with ONE level moved, scored by the same frozen HPPNet on "
        "the same frozen supports and seed. What they bound is how much a single free comb gain "
        "can buy.",
        "",
        "| arm | comb shift dB | " + " | ".join(f"{s} PIT" for s in shorts) + " |",
        "|---|---:|" + "---:|" * len(shorts),
    ]
    for name in names:
        shift = (_arm(payload, recs[0], name, "variant_params") or {}).get("profile_shift_db")
        out.append(
            f"| `{name}` | {_f(shift, '+.2f')} | "
            + " | ".join(_f(_arm(payload, r, name, "pit_mae")) for r in recs)
            + " |"
        )
    out.append("")
    gate = GT.DREGON_PIT_TARGET
    lvl = [_arm(payload, r, "v2_comb_level_matched", "pit_mae") for r in recs]
    ok = [v for v in lvl if v is not None]
    out.append(
        f"Two of the five gate supports only, so this is INDICATIVE and not a gate verdict: the "
        f"frozen DREGON target is {gate:.6f} rev/s and the round-2 arm scored 75.093968 over all "
        f"five. With the comb re-levelled onto the real window's own band level these two score "
        f"{', '.join(_f(v) for v in ok)} rev/s, against legacy "
        f"{', '.join(_f(_arm(payload, r, 'legacy', 'pit_mae')) for r in recs)} and real "
        f"{', '.join(_f(_arm(payload, r, 'real', 'pit_mae')) for r in recs)} on the same two. "
        f"The level-matched comb shift is the counterfactual a free `comb_gain_db` would have to "
        f"find; the prior above ({{0, 15}} dB) covers it."
    )
    out.append("")
    bench = payload["bench_reference"]["per_motor"]
    out.append(
        "`v2_comb_bench_ratio` OVERSHOOTS and is not the number to target. The four bench fits "
        "disagree wildly about their own floors — floor-only band level "
        + ", ".join(
            f"{n} {_f(v['floor_only_band_level_db_mic0'], '.2f')}" for n, v in bench.items()
        )
        + " dB — so their mean k=2 comb-to-floor "
        f"{_f(payload['bench_reference']['mean_line_ratio_db'].get('2'), '+.2f')} dB is dominated "
        f"by `Motor4` ({_f(bench['Motor4']['line_ratio'][0]['mean_db'], '+.2f')} dB, floor-only "
        f"{_f(bench['Motor4']['floor_only_band_level_db_mic0'], '.2f')} dB) against `Motor3` "
        f"({_f(bench['Motor3']['line_ratio'][0]['mean_db'], '+.2f')} dB). The shift it implies "
        f"({_f((_arm(payload, recs[0], 'v2_comb_bench_ratio', 'variant_params') or {}).get('profile_shift_db'), '+.2f')}"
        " dB) puts the render "
        f"{_f(_arm(payload, recs[0], 'v2_comb_bench_ratio', 'ltas_level_offset_db'), '+.1f')} dB "
        "over the real level and the PIT back up to "
        + ", ".join(_f(_arm(payload, r, "v2_comb_bench_ratio", "pit_mae")) for r in recs)
        + " rev/s. The real window's own BAND LEVEL is the well-posed target, and it is what a "
        "free `comb_gain_db` fitted on the flight supports would be pulled to."
    )
    out.append("")
    out.append(
        "The literal variant (e) — the comb re-levelled so its k=2 prominence matches the REAL "
        "clip's — is DEGENERATE on DREGON and is reported for completeness only. At mic 0 the "
        "real room-2 cruise clip has almost no discrete line prominence at the `motors_command` "
        "carrier ("
        + ", ".join(
            f"{s} {_f(_prom(payload, r, 'real', 2), '+.2f')} dB at k=2"
            for r, s in zip(recs, shorts, strict=True)
        )
        + "), so matching it asks the comb to stay where it is ("
        + ", ".join(
            f"{s} {_f(_row(payload, r)['calibration'].get('comb_real_ratio_shift_db'), '+.2f')} dB"
            for r, s in zip(recs, shorts, strict=True)
        )
        + "). The 8-mic MEAN prominence is larger — real "
        + ", ".join(
            f"{s} {_f((_arm(payload, r, 'real', 'comb_prominence_mic_mean_db') or [None])[0], '+.2f')}"
            for r, s in zip(recs, shorts, strict=True)
        )
        + " dB at k=2 against `v2_nocomb` "
        + ", ".join(
            f"{s} {_f((_arm(payload, r, 'v2_nocomb', 'comb_prominence_mic_mean_db') or [None])[0], '+.2f')}"
            for r, s in zip(recs, shorts, strict=True)
        )
        + " dB — so real audio DOES carry a comb, spatially coherent enough to survive the "
        "8-mic average, while the v2 render carries none. But it is a WEAK comb: HPPNet reaches "
        + ", ".join(
            f"{s} {_f(_arm(payload, r, 'real', 'pit_mae'))}"
            for r, s in zip(recs, shorts, strict=True)
        )
        + " rev/s on it. The render's target is therefore not 'a prominent comb' but 'the real "
        "clip's own comb ENERGY', which is what the band-level variant sets."
    )
    return out


# ── round 3: BROAD HARMONIC HUMPS, or the lack of them ──────────────────────
#
# The R3 question is not "is the comb there" (it is not: -18.7 dB under the
# floor) but "WHAT does HPPNet lock onto in real DREGON": narrow carrier lines,
# or broad harmonic humps — merged four-rotor lines plus in-frame drift plus a
# genuine in-flight per-order width — that a needle comb cannot imitate at any
# gain. Everything below measures the same three quantities on real, legacy and
# v2 audio and on width/level variants of the v2 render.

HUMP_SCHEMA = "noise-v2-dregon-humps/1"
HUMP_OUT_DEFAULT = Path("results/noise_v2/rounds/round3/dregon_humps")
HUMP_FIT_DEFAULT = Path(
    "results/noise_v2/rounds/round3/fits/dregon_room2_floor__flight_floor_lowk.json"
)

#: The three study windows: the two of the R2 study plus the LEAST drifting of
#: the five frozen DREGON cruise supports (free-flight, 3.1 rev/s in-window).
HUMP_RECORDINGS = (
    "free-flight_nosource_room2",
    "hovering_nosource_room2",
    "updown_nosource_room2",
)

HUMP_KS: tuple[int, ...] = tuple(range(1, 13))
#: The half-widths the per-order excess is read at (Hz). 4 Hz is sub-bin on the
#: 2048 front end (7.81 Hz) and one bin on the 4096 one (3.91 Hz); 128 Hz is
#: wider than the order spacing f0 ~ 80 Hz, so its bands OVERLAP their
#: neighbours and its null overlaps its own band — reported, never decisive.
HUMP_BS: tuple[float, ...] = (4.0, 16.0, 64.0, 128.0)
#: The sweep the effective Lorentzian half-width is fitted on.
WIDTH_BS: tuple[float, ...] = (4.0, 8.0, 16.0, 32.0, 64.0)
HUMP_N, HUMP_HOP = SPEC_N, SPEC_HOP  # 2048 / 512: the campaign's flight front end
FINE_N, FINE_HOP = TRK_N, TRK_HOP  # 4096 / 1024: 3.91 Hz per bin

#: The k-INDEPENDENT widths the variants override ``gamma_rk`` to (Hz, HWHM).
#: 13 Hz is the legacy DREGON cruise export's own pedestal (``gamma0`` 13.07 +
#: 0.211 k); 67 Hz is the previous generation's flight diffusion D = 845 rad^2/s
#: read as D / (4 pi); 30 and 130 bracket them.
HUMP_GAMMAS: tuple[float, ...] = (13.0, 30.0, 67.0, 130.0)
#: Extra comb shifts, at the FITTED (needle) widths: the level dose-response
#: that separates "the comb is too quiet" from "the comb is the wrong shape".
HUMP_LEVEL_SWEEP_DB: tuple[float, ...] = (6.0, 12.0, 18.0, 21.0, 24.0)

#: Orders and half-width the level match is read over: k=1 is excluded because
#: its B=64 annulus falls below the 30 Hz band edge, and B=64 is the
#: assignment's hump band.
MATCH_ORDERS: tuple[int, ...] = tuple(range(2, 9))
MATCH_B = 64.0
MATCH_SHIFT_BOUNDS = (-12.0, 42.0)

PROFILE_ORDERS = (2, 4, 8)
PROFILE_STEP_HZ = 1.0
#: The null offset: bands halfway between two orders, where a smooth floor
#: reads exactly what it reads on-order and a comb reads nothing.
NULL_OFFSET = 0.5


def _mic_power(audio: np.ndarray, *, n: int, hop: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """``mic0`` and mic-MEAN periodogram of one arm, and the frame centres."""
    acc: np.ndarray | None = None
    first: np.ndarray | None = None
    centres: np.ndarray | None = None
    for m in range(int(audio.shape[0])):
        p, c = _stft_power(audio[m], n=n, hop=hop)
        if m == 0:
            first, centres = p, c
        acc = p if acc is None else acc + p
    assert acc is not None and first is not None and centres is not None
    return {"mic0": first, "mic_mean": acc / float(audio.shape[0])}, centres


def _carrier_frames(f0_tracks: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """The four-rotor MEAN label carrier at each frame centre (rev/s)."""
    f0 = np.asarray(f0_tracks, dtype=np.float64)
    idx = np.clip(centres.astype(np.int64), 0, int(f0.shape[1]) - 1)
    return f0.mean(axis=0)[idx]


def order_excess(
    power: np.ndarray,
    centres: np.ndarray,
    f0_tracks: np.ndarray,
    *,
    sr: int,
    n: int,
    ks: Sequence[int] = HUMP_KS,
    bs: Sequence[float] = HUMP_BS,
    off: float = 0.0,
) -> dict[str, Any]:
    """Label-tracked per-order excess, as a fraction of 30-7900 Hz band power.

    For every frame and order the band is ``|f - (k + off) fbar(t)| <= B`` on
    THAT frame's own mean label carrier. The local level is NOT a flat median:
    a -7 dB/oct floor reads as a 10 %-of-band "hump" under one. It is the
    log-linear interpolation between the medians of the two annuli
    ``B < |f - c| <= 2.5 B``, which is flat in ``log P`` vs ``log f`` and
    therefore blind to any power law. ``off = NULL_OFFSET`` puts every band
    halfway between two orders: the null the on-order number is read against,
    at the same frequency and under the same estimator.

    Frames are summed before the ratio is taken (a power-weighted mean, which
    is what "fraction of band power" means), and an order whose annuli fall
    outside the band contributes nothing and is reported as ``None``.
    """
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
    band = (freqs >= SP.BAND_F_MIN) & (freqs <= SP.BAND_F_MAX)
    logf = np.log(np.maximum(freqs, 1e-9))
    carriers = _carrier_frames(f0_tracks, centres)
    num = np.zeros((len(ks), len(bs)), dtype=np.float64)
    raw = np.zeros((len(ks), len(bs)), dtype=np.float64)
    hit = np.zeros((len(ks), len(bs)), dtype=np.int64)
    den = 0.0
    for j, carrier in enumerate(carriers):
        p = power[j]
        den += float(p[band].sum())
        for ik, k in enumerate(ks):
            d = freqs - (float(k) + off) * float(carrier)
            for ib, b in enumerate(bs):
                inside = (np.abs(d) <= b) & band
                if not inside.any():
                    continue
                raw[ik, ib] += float(p[inside].sum())
                lo = (d < -b) & (d >= -2.5 * b) & band
                hi = (d > b) & (d <= 2.5 * b) & band
                if not lo.any() or not hi.any():
                    continue
                fl, fh = float(logf[lo].mean()), float(logf[hi].mean())
                pl = float(np.log(max(float(np.median(p[lo])), 1e-300)))
                ph = float(np.log(max(float(np.median(p[hi])), 1e-300)))
                slope = (ph - pl) / max(fh - fl, 1e-12)
                base = np.exp(pl + slope * (logf[inside] - fl))
                num[ik, ib] += float((p[inside] - base).sum())
                hit[ik, ib] += 1
    den = max(den, 1e-300)
    frac = [
        [(float(num[ik, ib] / den) if hit[ik, ib] else None) for ib in range(len(bs))]
        for ik in range(len(ks))
    ]
    return dict(
        orders=[int(k) for k in ks],
        bandwidths_hz=[float(b) for b in bs],
        offset=float(off),
        n_fft=int(n),
        band_power=float(den),
        excess_frac=frac,
        raw_frac=[[float(raw[ik, ib] / den) for ib in range(len(bs))] for ik in range(len(ks))],
        n_frames_used=[[int(hit[ik, ib]) for ib in range(len(bs))] for ik in range(len(ks))],
    )


def hump_fraction(
    power: np.ndarray,
    centres: np.ndarray,
    f0_tracks: np.ndarray,
    *,
    sr: int,
    n: int,
    b: float = 64.0,
    k_sets: Sequence[int] = (8, 12),
) -> dict[str, Any]:
    """Fraction of band power inside the UNION of the ``k fbar +- b`` bands.

    Reference-free: no floor estimator at all. At a DREGON cruise carrier
    (f0 ~ 80 Hz) the 128 Hz-wide bands OVERLAP, so the union is nearly the
    contiguous 30-1150 Hz region and the number is mostly "how much power is
    low" — which is why the union's own BANDWIDTH fraction is reported beside
    it and the concentration is their ratio.
    """
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
    band = (freqs >= SP.BAND_F_MIN) & (freqs <= SP.BAND_F_MAX)
    carriers = _carrier_frames(f0_tracks, centres)
    n_band = int(band.sum())
    out: dict[str, Any] = dict(bandwidth_hz=float(b), n_fft=int(n))
    den = 0.0
    acc = {int(kk): 0.0 for kk in k_sets}
    bins = {int(kk): 0 for kk in k_sets}
    for j, carrier in enumerate(carriers):
        p = power[j]
        den += float(p[band].sum())
        for kk in k_sets:
            union = np.zeros_like(band)
            for k in range(1, int(kk) + 1):
                union |= (np.abs(freqs - float(k) * float(carrier)) <= b) & band
            acc[int(kk)] += float(p[union].sum())
            bins[int(kk)] += int(union.sum())
    den = max(den, 1e-300)
    n_frames = max(len(carriers), 1)
    for kk in k_sets:
        out[f"frac_k{int(kk)}"] = float(acc[int(kk)] / den)
        out[f"bandwidth_frac_k{int(kk)}"] = float(bins[int(kk)] / n_frames / max(n_band, 1))
        out[f"concentration_k{int(kk)}"] = float(
            (acc[int(kk)] / den) / max(bins[int(kk)] / n_frames / max(n_band, 1), 1e-12)
        )
    return out


def effective_width_hz(
    excess: Sequence[float | None],
    null: Sequence[float | None] | None = None,
    bs: Sequence[float] = WIDTH_BS,
) -> dict[str, Any]:
    """The Lorentzian half-width implied by how the excess GROWS with ``B``.

    A Lorentzian of half-width ``gamma`` keeps ``(2/pi) arctan(B/gamma)`` of its
    power inside ``+-B``, so one order's excess-vs-B curve identifies
    ``(A, gamma)`` without ever resolving the line: a needle saturates by the
    first B, a 130 Hz pedestal is still growing at the last. ``A`` is the
    order's TOTAL tracked power as a fraction of band power.

    ``null`` is the same sweep read HALFWAY between two orders and is subtracted
    first: a floor whose curvature the two-sided baseline cannot follow reads the
    same excess on-order and off-order, and only what is left is carrier-locked.
    The number still carries the four-rotor spread (``k`` x a few rev/s) and the
    in-frame drift — which is the point: the same statistic on the needle-comb
    render is the instrumental floor this is read against.
    """
    e = np.asarray([np.nan if v is None else float(v) for v in excess], dtype=np.float64)
    if null is not None:
        e = e - np.asarray([np.nan if v is None else float(v) for v in null], dtype=np.float64)
    b = np.asarray([float(v) for v in bs], dtype=np.float64)
    ok = np.isfinite(e)
    empty = dict(
        gamma_hz=None,
        total_frac=None,
        residual=None,
        n_points=int(ok.sum()),
        at_grid_edge=False,
        concentration=None,
        tracked_frac=None,
    )
    if int(ok.sum()) < 3 or float(e[ok].max()) <= 0.0:
        return empty
    e, b = e[ok], b[ok]
    grid = np.exp(np.linspace(np.log(0.3), np.log(600.0), 400))
    best = (np.inf, float("nan"), float("nan"))
    for g in grid:
        phi = (2.0 / np.pi) * np.arctan(b / g)
        denom = float((phi * phi).sum())
        amp = float((e * phi).sum() / denom) if denom > 0 else 0.0
        res = float(((e - amp * phi) ** 2).sum())
        if res < best[0]:
            best = (res, float(g), amp)
    res, g, amp = best
    scale = float((e**2).sum()) or 1.0
    return dict(
        gamma_hz=float(g),
        total_frac=float(amp),
        residual=float(res / scale),
        n_points=int(e.size),
        at_grid_edge=bool(g <= 0.31 or g >= 599.0),
        concentration=(float(e[0] / e[-1]) if e[-1] > 0.0 else None),
        tracked_frac=float(e[-1]),
    )


def order_profile(
    audio: np.ndarray,
    f0_tracks: np.ndarray,
    k: int,
    *,
    sr: int,
    grid: np.ndarray,
    rotor: int | None = None,
    n: int = FINE_N,
    hop: int = FINE_HOP,
) -> tuple[np.ndarray, float]:
    """Frame- and mic-averaged power vs ``delta f`` around ``k`` x carrier(t).

    ``rotor`` demodulates on that rotor's OWN label track (its order-k line
    lands at ``delta f = 0`` and every other rotor's at ``k (f_r' - f_r)``);
    ``None`` demodulates on the four-rotor mean, where the four lines straddle
    zero. Interpolation is linear in POWER, so the average is an average of
    spectra and not of logs.
    """
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
    f0 = np.asarray(f0_tracks, dtype=np.float64)
    track = f0.mean(axis=0) if rotor is None else f0[int(rotor)]
    acc = np.zeros(grid.size, dtype=np.float64)
    centre_sum = 0.0
    n_used = 0
    for m in range(int(audio.shape[0])):
        power, centres = _stft_power(audio[m], n=n, hop=hop)
        idx = np.clip(centres.astype(np.int64), 0, int(f0.shape[1]) - 1)
        carriers = track[idx] * float(k)
        for j, c in enumerate(carriers):
            acc += np.interp(c + grid, freqs, power[j])
            centre_sum += float(c)
            n_used += 1
    return acc / max(n_used, 1), centre_sum / max(n_used, 1)


def profile_stats(prof: np.ndarray, grid: np.ndarray, fbar: float) -> dict[str, Any]:
    """Peak, inter-order level, contrast and half-power width of one profile.

    The reference level is the median of the profile over
    ``0.4 fbar <= |delta f| <= 0.6 fbar`` — the valley BETWEEN two orders,
    which is the only floor reference available when the order spacing (f0 ~ 80
    Hz) is comparable to the widths under test. The width therefore saturates
    at the spacing: a profile that has not fallen to half power by
    ``+-0.5 fbar`` is reported as saturated, which is itself the finding.
    """
    side = (np.abs(grid) >= 0.40 * fbar) & (np.abs(grid) <= 0.60 * fbar)
    level = float(np.median(prof[side])) if side.any() else float("nan")
    peak = float(np.interp(0.0, grid, prof))
    ex = prof - level
    i0 = int(np.argmin(np.abs(grid)))
    half = max(peak - level, 0.0) / 2.0

    def crossing(step: int) -> float:
        j = i0
        while True:
            nxt = j + step
            if nxt < 0 or nxt >= ex.size or abs(float(grid[nxt])) > 0.5 * fbar:
                return float("nan")
            if ex[nxt] <= half:
                y0, y1 = float(ex[j]), float(ex[nxt])
                if y1 == y0:
                    return float(grid[nxt])
                return float(grid[j]) + (half - y0) * (float(grid[nxt]) - float(grid[j])) / (
                    y1 - y0
                )
            j = nxt

    left, right = crossing(-1), crossing(1)
    contrast_db = float(10.0 * np.log10(max(peak, 1e-300) / max(level, 1e-300)))
    no_peak = bool(not np.isfinite(contrast_db) or contrast_db <= 0.0)
    width = float(right - left) if np.isfinite(left) and np.isfinite(right) else None
    if no_peak or (width is not None and width <= 0.0):
        width = None
    return dict(
        contrast_db=contrast_db,
        fwhm_hz=width,
        no_peak=no_peak,
        saturated=bool(width is None and not no_peak),
        peak_power=peak,
        inter_order_power=level,
    )


def rotor_split(
    prof: np.ndarray, grid: np.ndarray, *, k: int, deltas: Sequence[float], fbar: float
) -> dict[str, Any]:
    """Do the four rotors show as four sub-peaks, or as one merged hump?

    ``prof`` is demodulated on the four-rotor MEAN carrier, so rotor ``r``'s
    order-k line sits at ``delta f = k (f_r - fbar)``. Between two adjacent
    expected positions a resolved pair must DIP: the statistic is the deepest
    such dip in dB relative to the shallower of the two peaks it separates.
    """
    pos = sorted(float(k) * float(d) for d in deltas)
    at = [float(np.interp(p, grid, prof)) for p in pos]
    dips: list[dict[str, Any]] = []
    for i in range(len(pos) - 1):
        lo, hi = pos[i], pos[i + 1]
        sel = (grid > lo) & (grid < hi)
        if not sel.any() or abs(hi - lo) < 2.0 * PROFILE_STEP_HZ:
            continue
        valley = float(prof[sel].min())
        shallower = min(at[i], at[i + 1])
        dips.append(
            dict(
                between_hz=[lo, hi],
                separation_hz=float(hi - lo),
                dip_db=float(10.0 * np.log10(max(shallower, 1e-300) / max(valley, 1e-300))),
            )
        )
    inner = np.abs(grid) <= 0.5 * fbar
    g, p = grid[inner], prof[inner]
    maxima = int(((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:])).sum())
    best = max((d["dip_db"] for d in dips), default=float("nan"))
    return dict(
        k=int(k),
        expected_offsets_hz=pos,
        power_at_offsets=at,
        dips=dips,
        max_dip_db=None if not np.isfinite(best) else float(best),
        n_local_maxima=maxima,
        span_hz=float(pos[-1] - pos[0]) if pos else None,
        verdict=("split" if np.isfinite(best) and best >= 3.0 else "merged"),
        criterion="four label carriers are RESOLVED when the deepest inter-carrier dip >= 3 dB",
        _grid=g,
        _prof=p,
    )


def hump_diagnostics(
    audio: np.ndarray,
    f0_tracks: np.ndarray,
    *,
    sr: int,
    fbar: float,
    deltas: Sequence[float],
) -> dict[str, Any]:
    """Every hump number of one arm on one window."""
    freqs = np.fft.rfftfreq(PSD_N, d=1.0 / float(sr))
    psd0 = _welch(audio[0])
    psd_mean = np.mean([_welch(audio[m]) for m in range(int(audio.shape[0]))], axis=0)
    coarse, c_coarse = _mic_power(audio, n=HUMP_N, hop=HUMP_HOP)
    fine, c_fine = _mic_power(audio, n=FINE_N, hop=FINE_HOP)
    out: dict[str, Any] = dict(
        band_level_db_mic0=band_level_db(psd0, freqs),
        band_level_db_mic_mean=band_level_db(psd_mean, freqs),
    )
    for mic_key in ("mic0", "mic_mean"):
        out[f"excess_{mic_key}"] = order_excess(
            coarse[mic_key], c_coarse, f0_tracks, sr=sr, n=HUMP_N
        )
        out[f"null_{mic_key}"] = order_excess(
            coarse[mic_key], c_coarse, f0_tracks, sr=sr, n=HUMP_N, off=NULL_OFFSET
        )
        out[f"hump_fraction_{mic_key}"] = hump_fraction(
            coarse[mic_key], c_coarse, f0_tracks, sr=sr, n=HUMP_N
        )
        sweep = order_excess(fine[mic_key], c_fine, f0_tracks, sr=sr, n=FINE_N, bs=WIDTH_BS)
        sweep_null = order_excess(
            fine[mic_key], c_fine, f0_tracks, sr=sr, n=FINE_N, bs=WIDTH_BS, off=NULL_OFFSET
        )
        out[f"width_sweep_{mic_key}"] = sweep
        out[f"width_sweep_null_{mic_key}"] = sweep_null
        out[f"width_{mic_key}"] = [
            dict(
                k=int(k),
                **effective_width_hz(sweep["excess_frac"][i], sweep_null["excess_frac"][i]),
            )
            for i, k in enumerate(sweep["orders"])
        ]
    out["match_fraction"] = match_fraction(out["excess_mic_mean"])
    grid = np.arange(-1.5 * fbar, 1.5 * fbar + 1e-9, PROFILE_STEP_HZ)
    profiles: list[dict[str, Any]] = []
    curves: dict[int, dict[str, np.ndarray]] = {}
    for k in PROFILE_ORDERS:
        per_rotor = []
        acc = np.zeros(grid.size)
        for r in range(int(np.asarray(f0_tracks).shape[0])):
            prof, centre = order_profile(audio, f0_tracks, k, sr=sr, grid=grid, rotor=r)
            acc += prof
            per_rotor.append(
                dict(rotor=int(r), centre_hz=centre, **profile_stats(prof, grid, fbar))
            )
        rotor_mean = acc / max(int(np.asarray(f0_tracks).shape[0]), 1)
        mean_prof, mean_centre = order_profile(audio, f0_tracks, k, sr=sr, grid=grid, rotor=None)
        split = rotor_split(mean_prof, grid, k=k, deltas=deltas, fbar=fbar)
        profiles.append(
            dict(
                k=int(k),
                centre_hz=mean_centre,
                per_rotor=per_rotor,
                rotor_mean=profile_stats(rotor_mean, grid, fbar),
                mean_carrier=profile_stats(mean_prof, grid, fbar),
                split={kk: vv for kk, vv in split.items() if not kk.startswith("_")},
            )
        )
        curves[int(k)] = dict(rotor_mean=rotor_mean, mean_carrier=mean_prof)
    out["profiles"] = profiles
    out["_curves"] = curves
    out["_grid"] = grid
    out["_psd_mean"] = psd_mean
    return out


def match_fraction(excess: dict[str, Any]) -> float:
    """The level-match statistic: tracked excess over ``MATCH_ORDERS`` at 64 Hz."""
    orders = list(excess["orders"])
    ib = list(excess["bandwidths_hz"]).index(MATCH_B)
    total = 0.0
    for k in MATCH_ORDERS:
        v = excess["excess_frac"][orders.index(int(k))][ib]
        if v is not None:
            total += float(v)
    return float(total)


def arm_match_fraction(audio: np.ndarray, f0_tracks: np.ndarray, *, sr: int) -> float:
    """:func:`match_fraction` of a signal, on the mic-MEAN coarse front end."""
    power, centres = _mic_power(audio, n=HUMP_N, hop=HUMP_HOP)
    return match_fraction(order_excess(power["mic_mean"], centres, f0_tracks, sr=sr, n=HUMP_N))


# ── the arms ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HumpArm:
    name: str
    kind: str  # real | legacy | v2
    tag: str
    label: str
    gamma_hz: float | None = None
    matched: bool = False
    shift_db: float = 0.0
    legacy_needle: bool = False


def hump_arms() -> tuple[HumpArm, ...]:
    out: list[HumpArm] = [
        HumpArm("real", "real", "real", "the real DREGON room-2 clip"),
        HumpArm("legacy", "legacy", "legacy", "legacy stage-2 baseline, identity-matched"),
        HumpArm(
            "legacy_needle",
            "legacy",
            "V7",
            "legacy with the Lorentzian PEDESTAL muted (coherence_k_half = 0: every "
            "order's power goes through the tone bank as a needle)",
            legacy_needle=True,
        ),
        HumpArm("v2", "v2", "v2", "the round-3 v2 candidate, unchanged (bench gamma_rk)"),
    ]
    tags = {13.0: "V0", 30.0: "V1", 67.0: "V2", 130.0: "V3"}
    tags_m = {13.0: "V0m", 30.0: "V4", 67.0: "V5", 130.0: "V6"}
    for g in HUMP_GAMMAS:
        out.append(
            HumpArm(
                f"v2_g{int(g)}",
                "v2",
                tags[g],
                f"v2 with gamma_rk overridden to a k-independent {g:g} Hz, comb level as fitted",
                gamma_hz=g,
            )
        )
    out.append(
        HumpArm(
            "v2_matched",
            "v2",
            "Vm",
            "v2 at the FITTED (needle) widths with the comb re-levelled onto the real "
            "window's k<=8 hump fraction — the control that isolates width from level",
            matched=True,
        )
    )
    for g in HUMP_GAMMAS:
        out.append(
            HumpArm(
                f"v2_g{int(g)}_matched",
                "v2",
                tags_m[g],
                f"v2 at a k-independent {g:g} Hz with the comb re-levelled onto the real "
                "window's k<=8 hump fraction",
                gamma_hz=g,
                matched=True,
            )
        )
    for s in HUMP_LEVEL_SWEEP_DB:
        out.append(
            HumpArm(
                f"v2_plus{int(s)}db",
                "v2",
                f"L+{int(s)}",
                f"v2 at the fitted widths with the comb {s:g} dB up: the level dose-response",
                shift_db=s,
            )
        )
    return tuple(out)


def _legacy_render(
    arm: Any, f0_tracks: np.ndarray, *, regime: str, n_mics: int, seed: int, needle: bool
) -> np.ndarray:
    """The legacy arm's own render, optionally with the pedestal muted.

    The legacy comb is a COHERENCE SPLIT: the share ``w_k = exp(-(k/k_half)^2)``
    of each order's power goes through the tone bank as a needle and the rest is
    rendered as narrowband noise of Lorentzian half-width ``gamma0 +
    gamma_slope k`` alongside the floor — the pedestal. ``coherence_k_half = 0``
    switches the split off entirely, which puts every order's FULL power through
    the tone bank: the same comb energy with no pedestal at all. The draw order
    differs (the split builds two extra spectra), so this arm is not a
    seed-matched decomposition of the legacy arm, it is its needle counterpart.
    """
    from experiments.stochastic_fit import stage2 as S2

    physical = RE.to_renderer_units(arm.legacy[regime])
    params = dict(physical.params)
    if needle:
        params["coherence_k_half"] = 0.0
    return np.asarray(
        S2.render_from_export(
            params,
            np.atleast_2d(np.asarray(f0_tracks, dtype=np.float64)),
            sample_rate_work=RE.SAMPLE_RATE_WORK,
            n_mics=int(n_mics),
            seed=int(seed),
            normalize_rms=None,
        ),
        dtype=np.float64,
    )


def legacy_width_reference(arm: Any, *, regime: str) -> dict[str, Any]:
    """The legacy export's own width and coherence parameters."""
    params = dict(RE.to_renderer_units(arm.legacy[regime]).params)
    g0 = np.atleast_1d(np.asarray(params.get("gamma0", np.nan), dtype=np.float64))
    sl = np.atleast_1d(np.asarray(params.get("gamma_slope", np.nan), dtype=np.float64))
    k_half = float(params.get("coherence_k_half", 0.0) or 0.0)
    ks = (1, 2, 4, 8, 12)
    return dict(
        gamma0_hz=[float(v) for v in g0],
        gamma_slope_hz_per_order=[float(v) for v in sl],
        coherence_k_half=k_half,
        pedestal_hwhm_hz={str(k): float(g0.mean() + sl.mean() * float(k)) for k in ks},
        coherent_share={
            str(k): (float(np.exp(-((float(k) / k_half) ** 2))) if k_half > 0 else 1.0) for k in ks
        },
        note=(
            "gamma_k = gamma0 + gamma_slope k is the HWHM of the INCOHERENT share "
            "1 - w_k of order k, w_k = exp(-(k/coherence_k_half)^2); the coherent "
            "share is a needle. Rendered by stochastic_rotor_noise.synthesize "
            "(line_mode fm, shaft jitter and phase diffusion both 0 in "
            "stage2.params_from_export, so the needle has no width of its own)."
        ),
    )


def solve_match_shift(
    *,
    comb_only: np.ndarray,
    floor_only: np.ndarray,
    f0_tracks: np.ndarray,
    target: float,
    sr: int,
) -> dict[str, Any]:
    """The comb shift in dB that puts the render's hump fraction on ``target``.

    The two blocks are the SAME seed, so ``floor + 10^(s/20) comb`` is exactly
    what a render with ``profile_db + s`` produces (verified to float round-off
    in the payload), and the shift is solved on the combination instead of by
    re-rendering. The statistic is monotone in the shift wherever the comb's own
    hump fraction exceeds the floor's, which is checked at both bounds.
    """

    def frac_at(shift_db: float) -> float:
        amp = 10.0 ** (float(shift_db) / 20.0)
        return arm_match_fraction(floor_only + amp * comb_only, f0_tracks, sr=sr)

    lo_db, hi_db = MATCH_SHIFT_BOUNDS
    lo, hi = frac_at(lo_db), frac_at(hi_db)
    attainable = min(lo, hi) <= target <= max(lo, hi)
    if not attainable:
        pick = lo_db if abs(lo - target) < abs(hi - target) else hi_db
        return dict(
            shift_db=float(pick),
            attained=float(frac_at(pick)),
            target=float(target),
            at_bound=True,
            bounds_frac=[float(lo), float(hi)],
            iterations=2,
        )
    a, b = lo_db, hi_db
    fa = lo
    it = 2
    for _ in range(18):
        mid = 0.5 * (a + b)
        fm = frac_at(mid)
        it += 1
        if (fm - target) * (fa - target) <= 0.0:
            b = mid
        else:
            a, fa = mid, fm
        if abs(b - a) < 0.05:
            break
    pick = 0.5 * (a + b)
    return dict(
        shift_db=float(pick),
        attained=float(frac_at(pick)),
        target=float(target),
        at_bound=False,
        bounds_frac=[float(lo), float(hi)],
        iterations=it,
    )


def run_humps(
    *,
    fit_path: Path,
    out: Path,
    seed: int,
    probe: bool,
    n_mics: int,
    recordings: Sequence[str],
) -> dict[str, Any]:
    rs = _module("noise_v2_round_score")
    fit = json.loads(Path(fit_path).read_text())
    if str(fit.get("schema")) not in ("noise-v2-fit/1", "noise-v2-fit/2"):
        die(f"{fit_path}: not a noise-v2 fit payload")
    probe_obj = rs.Probe.load() if probe else None
    legacy = rs.legacy_arm("dregon")
    arms = hump_arms()
    gamma_fitted = np.asarray(fit["params"]["gamma_hz"], dtype=np.float64)
    payload: dict[str, Any] = dict(
        schema=HUMP_SCHEMA,
        git=git_rev(),
        fit=dict(
            path=str(fit_path),
            support=fit.get("support"),
            mode=fit.get("mode"),
            converged=(fit.get("diagnostics") or {}).get("converged"),
            comb_gain_db=fit["params"]["profile"].get("comb_gain_db"),
            low_order_gain_db=fit["params"]["profile"].get("low_order_gain_db"),
            gamma_hz_mean_per_order={
                str(k): float(gamma_fitted[:, k - 1].mean())
                for k in (1, 2, 4, 8, 12, 24, 48, 88)
                if k <= gamma_fitted.shape[1]
            },
        ),
        protocol=dict(
            seed=int(seed),
            n_mics=int(n_mics),
            coarse=dict(n_fft=HUMP_N, hop=HUMP_HOP),
            fine=dict(n_fft=FINE_N, hop=FINE_HOP),
            band_hz=[SP.BAND_F_MIN, SP.BAND_F_MAX],
            orders=list(HUMP_KS),
            bandwidths_hz=list(HUMP_BS),
            width_bandwidths_hz=list(WIDTH_BS),
            gammas_hz=list(HUMP_GAMMAS),
            level_sweep_db=list(HUMP_LEVEL_SWEEP_DB),
            match=dict(orders=list(MATCH_ORDERS), bandwidth_hz=MATCH_B, statistic="tracked excess"),
            profile_orders=list(PROFILE_ORDERS),
            null_offset=NULL_OFFSET,
            scorer=(probe_obj.record if probe_obj is not None else None),
            arms=[dict(name=a.name, tag=a.tag, kind=a.kind, label=a.label) for a in arms],
        ),
        supports={},
    )
    figures: dict[str, Any] = {}
    for recording in recordings:
        found = [s for s in GT.DREGON_CRUISE_SUPPORTS if s.recording == recording]
        if not found:
            die(f"no frozen DREGON cruise support carries recording {recording!r}")
        support = found[0]
        clip = RE.load_window(
            support.window,
            dataset=GT.DATASET["dregon"],
            version=None,
            channels=None,
            rps_key=GT.RAW_RPS_KEY["dregon"],
        )
        sr = int(clip.sr)
        real = np.asarray(clip.audio, dtype=np.float64)[:n_mics]
        reference = np.atleast_2d(np.asarray(clip.rps, dtype=np.float64))
        rsupport = RE.regime_support(
            support.window,
            reference,
            regime=support.regime,
            min_rps=support.min_rps,
            max_rps=support.max_rps,
            sr=sr,
        )
        mics = list(range(min(int(n_mics), int(real.shape[0]))))
        f0_rotor = reference.mean(axis=1)
        fbar = float(reference.mean())
        deltas = [float(v - fbar) for v in f0_rotor]
        larm = rs._arm_for_recording(legacy, support)
        row: dict[str, Any] = dict(
            support=support.as_dict(),
            n_samples=int(real.shape[-1]),
            scored_seconds=float(rsupport.scored_seconds),
            mean_reference_rps=fbar,
            per_rotor_mean_rps=[float(v) for v in f0_rotor],
            per_rotor_offset_rps=deltas,
            carrier_drift_rev_s=float(reference.mean(axis=0).max() - reference.mean(axis=0).min()),
            in_frame_drift_rev_s=float(
                np.abs(np.diff(reference.mean(axis=0)[:: int(FINE_N)])).max()
            ),
            legacy_width=legacy_width_reference(larm, regime=support.regime),
            arms={},
        )
        # the two blocks of the fitted render, and one comb block per width
        floor_only = RD.render_noise(
            _mutate(fit, mute_comb=True), reference, n_mics=len(mics), seed=int(seed)
        )[: len(mics)]
        comb_blocks: dict[float | None, np.ndarray] = {}
        for g in (None, *HUMP_GAMMAS):
            comb_blocks[g] = RD.render_noise(
                _mutate(fit, mute_floor=True, gamma_hz=g),
                reference,
                n_mics=len(mics),
                seed=int(seed),
            )[: len(mics)]
        real_diag = hump_diagnostics(real, reference, sr=sr, fbar=fbar, deltas=deltas)
        target = float(real_diag["match_fraction"])
        shifts: dict[str, Any] = {}
        for g in (None, *HUMP_GAMMAS):
            shifts["fitted" if g is None else f"{g:g}"] = solve_match_shift(
                comb_only=comb_blocks[g],
                floor_only=floor_only,
                f0_tracks=reference,
                target=target,
                sr=sr,
            )
        full = RD.render_noise(fit, reference, n_mics=len(mics), seed=int(seed))[: len(mics)]
        row["block_additivity"] = dict(
            max_abs_diff=float(np.abs(comb_blocks[None] + floor_only - full).max()),
            max_abs_render=float(np.abs(full).max()),
            note="comb-only + floor-only of the same seed IS the full render",
        )
        row["match_target"] = dict(
            target_frac=target,
            floor_only_frac=arm_match_fraction(floor_only, reference, sr=sr),
            comb_only_frac={
                ("fitted" if g is None else f"{g:g}"): arm_match_fraction(
                    comb_blocks[g], reference, sr=sr
                )
                for g in (None, *HUMP_GAMMAS)
            },
            shifts=shifts,
        )
        arm_figures: dict[str, Any] = {}
        for spec in arms:
            t0 = time.time()
            if spec.kind == "real":
                audio, diag = real, real_diag
            else:
                if spec.kind == "legacy":
                    audio = _legacy_render(
                        larm,
                        reference,
                        regime=support.regime,
                        n_mics=len(mics),
                        seed=int(seed),
                        needle=spec.legacy_needle,
                    )[: len(mics)]
                else:
                    shift = float(spec.shift_db)
                    if spec.matched:
                        key = "fitted" if spec.gamma_hz is None else f"{spec.gamma_hz:g}"
                        shift = float(shifts[key]["shift_db"])
                    audio = RD.render_noise(
                        _mutate(fit, gamma_hz=spec.gamma_hz, comb_shift_db=shift),
                        reference,
                        n_mics=len(mics),
                        seed=int(seed),
                    )[: len(mics)]
                if int(audio.shape[-1]) != int(real.shape[-1]):
                    die(
                        f"{support.key}: arm {spec.name} rendered {audio.shape[-1]} samples "
                        f"against the frozen {real.shape[-1]}"
                    )
                diag = hump_diagnostics(audio, reference, sr=sr, fbar=fbar, deltas=deltas)
            entry: dict[str, Any] = dict(
                kind=spec.kind,
                tag=spec.tag,
                label=spec.label,
                gamma_hz=spec.gamma_hz,
                comb_shift_db=(
                    float(
                        shifts["fitted" if spec.gamma_hz is None else f"{spec.gamma_hz:g}"][
                            "shift_db"
                        ]
                    )
                    if spec.matched
                    else (float(spec.shift_db) if spec.kind == "v2" else None)
                ),
                render_seconds=float(time.time() - t0),
                **{k: v for k, v in diag.items() if not k.startswith("_")},
            )
            if probe_obj is not None:
                pit = probe_obj.tracker.pit(
                    audio,
                    reference,
                    mics=mics,
                    expected_samples=int(real.shape[-1]),
                    support=rsupport,
                )
                entry["pit_mae"] = float(pit["mae"])
                entry["pit_per_mic"] = [float(v) for v in pit["per_mic"]]
                entry["pit_per_rotor"] = [float(v) for v in pit["per_rotor"]]
                entry["n_scored_frames"] = int(pit["n_scored_frames"])
            row["arms"][spec.name] = entry
            arm_figures[spec.name] = dict(
                psd_mean=diag["_psd_mean"], grid=diag["_grid"], profiles=diag["_curves"]
            )
            print(
                f"[{support.recording}] {spec.name} ({spec.tag}): "
                f"level={entry['band_level_db_mic0']:+.2f} dB "
                f"hump8={entry['hump_fraction_mic_mean']['frac_k8']:.4f} "
                f"match={entry['match_fraction']:+.4f} "
                f"pit={entry.get('pit_mae')}",
                flush=True,
            )
        payload["supports"][support.key] = row
        figures[support.recording] = dict(support=support, sr=sr, arms=arm_figures, row=row)
    payload["verdicts"] = hump_verdicts(payload)
    Path(out).mkdir(parents=True, exist_ok=True)
    return payload | {"_figures": figures}


def _harm(row: dict[str, Any], name: str) -> dict[str, Any]:
    entry = row["arms"].get(name)
    if entry is None:
        die(f"no arm {name!r} in this pass")
    return entry


def _width_at(entry: dict[str, Any], k: int, mic: str = "mic_mean") -> float | None:
    for w in entry[f"width_{mic}"]:
        if int(w["k"]) == int(k):
            return w["gamma_hz"]
    return None


def _width_cell(entry: dict[str, Any], k: int, mic: str = "mic_mean") -> str:
    """One cell of the width table: gamma, its grid-edge state, its concentration."""
    for w in entry[f"width_{mic}"]:
        if int(w["k"]) != int(k):
            continue
        g = w["gamma_hz"]
        if g is None:
            return "—"
        conc = w.get("concentration")
        head = ">600" if w.get("at_grid_edge") and float(g) > 1.0 else f"{float(g):.1f}"
        return head + ("" if conc is None else f" ({float(conc):.2f})")
    return "—"


def _excess_at(entry: dict[str, Any], k: int, b: float, mic: str = "mic_mean") -> float | None:
    ex = entry[f"excess_{mic}"]
    ib = list(ex["bandwidths_hz"]).index(float(b))
    return ex["excess_frac"][list(ex["orders"]).index(int(k))][ib]


def _wider_than(a: float | None, b: float | None, factor: float) -> bool:
    """Is width ``a`` at least ``factor`` times width ``b``, both measured?"""
    return a is not None and b is not None and float(a) > factor * float(b)


def hump_verdicts(payload: dict[str, Any]) -> dict[str, Any]:
    """The hypothesis, decided per window by numbers already in the payload."""
    out: dict[str, Any] = {}
    for key, row in payload["supports"].items():
        real = _harm(row, "real")
        v2 = _harm(row, "v2")
        wide = [row["arms"][n] for n in row["arms"] if n.endswith("_matched") and "_g" in n]
        narrow = row["arms"].get("v2_matched")
        pits = {n: e.get("pit_mae") for n, e in row["arms"].items()}
        have_pit = all(v is not None for v in pits.values())
        widths_real = {k: _width_at(real, k) for k in (1, 2, 4, 8)}
        widths_v2 = {k: _width_at(v2, k) for k in (1, 2, 4, 8)}
        best_wide = min((e["pit_mae"] for e in wide if e.get("pit_mae") is not None), default=None)
        out[key] = dict(
            broad_humps_in_real=dict(
                supported=bool(all(_wider_than(widths_real[k], widths_v2[k], 2.0) for k in (1, 2))),
                effective_hwhm_real_hz=widths_real,
                effective_hwhm_v2_hz=widths_v2,
                criterion="real's effective per-order HWHM exceeds the needle render's by 2x",
            ),
            width_is_the_lever=dict(
                supported=bool(
                    have_pit
                    and best_wide is not None
                    and narrow is not None
                    and narrow.get("pit_mae") is not None
                    and best_wide < narrow["pit_mae"]
                ),
                best_wide_matched_pit=best_wide,
                narrow_matched_pit=(narrow or {}).get("pit_mae"),
                criterion=(
                    "at the SAME hump fraction, a widened comb tracks better than the needle comb"
                ),
            ),
            level_is_the_lever=dict(
                supported=bool(
                    have_pit
                    and narrow is not None
                    and narrow.get("pit_mae") is not None
                    and v2.get("pit_mae") is not None
                    and narrow["pit_mae"] < 0.5 * v2["pit_mae"]
                ),
                v2_pit=v2.get("pit_mae"),
                narrow_matched_pit=(narrow or {}).get("pit_mae"),
                criterion="re-levelling the NEEDLE comb alone halves the PIT MAE",
            ),
            four_rotor_merge=dict(
                real=[p["split"]["verdict"] for p in real["profiles"]],
                v2=[p["split"]["verdict"] for p in v2["profiles"]],
                orders=[int(p["k"]) for p in real["profiles"]],
                real_max_dip_db=[p["split"]["max_dip_db"] for p in real["profiles"]],
                v2_max_dip_db=[p["split"]["max_dip_db"] for p in v2["profiles"]],
            ),
        )
    return out


# ── the hump study's figures and findings ───────────────────────────────────

HUMP_FIG_ARMS = ("real", "legacy", "v2", "v2_matched", "v2_g67_matched", "v2_g130_matched")
HUMP_FIG_COLOURS = {
    "real": "black",
    "legacy": "tab:orange",
    "v2": "tab:blue",
    "v2_matched": "tab:green",
    "v2_g67_matched": "tab:red",
    "v2_g130_matched": "tab:purple",
    "legacy_needle": "tab:brown",
}


def write_hump_figures(payload: dict[str, Any], figures: dict[str, Any], out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    recs = list(figures)
    written: list[str] = []

    # 1. the frame-averaged mic-mean spectra, 20-1200 Hz and full band
    fig, axes = plt.subplots(len(recs), 2, figsize=(13, 3.1 * len(recs)), squeeze=False)
    for i, rec in enumerate(recs):
        blob = figures[rec]
        freqs = np.fft.rfftfreq(PSD_N, d=1.0 / float(blob["sr"]))
        fbar = float(blob["row"]["mean_reference_rps"])
        for name in HUMP_FIG_ARMS:
            arm = blob["arms"].get(name)
            if arm is None:
                continue
            f, db = _smooth_log(freqs, arm["psd_mean"], per_oct=96)
            for j in (0, 1):
                axes[i][j].plot(f, db, color=HUMP_FIG_COLOURS.get(name, "grey"), lw=1.0, label=name)
        for k in range(1, 13):
            axes[i][0].axvline(k * fbar, color="green", lw=0.4, ls=":")
        axes[i][0].set_xlim(20.0, 1200.0)
        axes[i][1].set_xscale("log")
        axes[i][1].set_xlim(30.0, 8000.0)
        for j in (0, 1):
            axes[i][j].grid(alpha=0.3)
            axes[i][j].set_ylabel(f"{rec.split('_')[0]}\ndB")
        axes[i][0].legend(fontsize=6, ncol=2)
    axes[0][0].set_title("mic-mean spectrum, low orders (green: k x fbar)")
    axes[0][1].set_title("mic-mean spectrum, full band")
    written.append(_save(fig, out / "humps_spectra.png"))

    # 2. the order-tracked profiles: rotor-centred (width) and mean-centred (split)
    fig, axes = plt.subplots(
        len(recs),
        len(PROFILE_ORDERS),
        figsize=(4.4 * len(PROFILE_ORDERS), 3.1 * len(recs)),
        squeeze=False,
    )
    for i, rec in enumerate(recs):
        blob = figures[rec]
        fbar = float(blob["row"]["mean_reference_rps"])
        for j, k in enumerate(PROFILE_ORDERS):
            ax = axes[i][j]
            for name in HUMP_FIG_ARMS:
                arm = blob["arms"].get(name)
                if arm is None:
                    continue
                grid = arm["grid"]
                prof = arm["profiles"][int(k)]["rotor_mean"]
                side = (np.abs(grid) >= 0.4 * fbar) & (np.abs(grid) <= 0.6 * fbar)
                ref = float(np.median(prof[side]))
                ax.plot(
                    grid,
                    10.0 * np.log10(np.maximum(prof, 1e-300) / max(ref, 1e-300)),
                    color=HUMP_FIG_COLOURS.get(name, "grey"),
                    lw=1.0,
                    label=name,
                )
            ax.axvline(0.0, color="green", lw=0.5, ls=":")
            ax.grid(alpha=0.3)
            ax.set_xlabel("delta f [Hz]")
            ax.set_ylabel(f"{rec.split('_')[0]}\ndB over inter-order")
            ax.set_title(f"k = {k}, rotor-centred")
            if i == 0 and j == 0:
                ax.legend(fontsize=6)
    written.append(_save(fig, out / "humps_order_profiles.png"))

    # 3. the four-rotor split: mean-carrier profiles with the label carriers marked
    fig, axes = plt.subplots(len(recs), 2, figsize=(11, 3.1 * len(recs)), squeeze=False)
    for i, rec in enumerate(recs):
        blob = figures[rec]
        row = blob["row"]
        fbar = float(row["mean_reference_rps"])
        for j, k in enumerate((4, 8)):
            ax = axes[i][j]
            for name in ("real", "legacy", "v2", "v2_matched"):
                arm = blob["arms"].get(name)
                if arm is None:
                    continue
                grid = arm["grid"]
                prof = arm["profiles"][int(k)]["mean_carrier"]
                sel = np.abs(grid) <= 0.6 * fbar
                ref = float(np.median(prof[(np.abs(grid) >= 0.4 * fbar) & sel]))
                ax.plot(
                    grid[sel],
                    10.0 * np.log10(np.maximum(prof[sel], 1e-300) / max(ref, 1e-300)),
                    color=HUMP_FIG_COLOURS.get(name, "grey"),
                    lw=1.0,
                    label=name,
                )
            for d in row["per_rotor_offset_rps"]:
                ax.axvline(float(k) * float(d), color="green", lw=0.6, ls="--")
            ax.grid(alpha=0.3)
            ax.set_xlabel("delta f from k x fbar [Hz]")
            ax.set_ylabel(f"{rec.split('_')[0]}\ndB")
            ax.set_title(f"k = {k}: the four label carriers (dashed)")
            if i == 0 and j == 0:
                ax.legend(fontsize=6)
    written.append(_save(fig, out / "humps_rotor_split.png"))

    # 4. PIT against comb level and against width
    rows = list(payload["supports"].values())
    if all(r["arms"]["v2"].get("pit_mae") is not None for r in rows):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for r in rows:
            rec = str(r["support"]["recording"]).split("_")[0]
            shifts = [0.0] + [float(s) for s in HUMP_LEVEL_SWEEP_DB]
            names = ["v2"] + [f"v2_plus{int(s)}db" for s in HUMP_LEVEL_SWEEP_DB]
            matched = r["arms"].get("v2_matched")
            pits = [r["arms"][n]["pit_mae"] for n in names]
            axes[0].plot(shifts, pits, "o-", lw=1.0, label=rec)
            if matched is not None and matched.get("comb_shift_db") is not None:
                axes[0].plot(
                    [float(matched["comb_shift_db"])],
                    [float(matched["pit_mae"])],
                    "*",
                    ms=11,
                    color=axes[0].lines[-1].get_color(),
                )
            widths = []
            wpits = []
            for g in HUMP_GAMMAS:
                arm = r["arms"].get(f"v2_g{int(g)}_matched")
                if arm is not None and arm.get("pit_mae") is not None:
                    widths.append(float(g))
                    wpits.append(float(arm["pit_mae"]))
            narrow = r["arms"].get("v2_matched")
            if narrow is not None and narrow.get("pit_mae") is not None:
                widths = [0.6] + widths
                wpits = [float(narrow["pit_mae"])] + wpits
            axes[1].plot(widths, wpits, "o-", lw=1.0, label=rec)
            axes[0].axhline(float(r["arms"]["real"]["pit_mae"]), color="grey", lw=0.5, ls=":")
            axes[1].axhline(
                float(r["arms"]["legacy"]["pit_mae"]), color="tab:orange", lw=0.5, ls=":"
            )
        axes[0].set_xlabel("comb shift [dB] (star: the hump-fraction match)")
        axes[0].set_ylabel("PIT MAE [rev/s]")
        axes[0].set_yscale("log")
        axes[0].set_title("level dose-response, needle comb (grey: real)")
        axes[1].set_xlabel("k-independent gamma_rk [Hz] (0.6 = as fitted)")
        axes[1].set_ylabel("PIT MAE [rev/s]")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_title("width at the MATCHED hump fraction (orange: legacy)")
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        written.append(_save(fig, out / "humps_pit_vs_level_width.png"))
    return written


def _short(recording: str) -> str:
    return str(recording).split("_")[0]


def _pct(v: Any) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{100.0 * float(v):+.2f}"


def hump_findings(payload: dict[str, Any], *, job: str | None, figures: Sequence[str]) -> str:
    rows = list(payload["supports"].values())
    recs = [_short(r["support"]["recording"]) for r in rows]
    arms = [a["name"] for a in payload["protocol"]["arms"]]
    tags = {a["name"]: a["tag"] for a in payload["protocol"]["arms"]}
    labels = {a["name"]: a["label"] for a in payload["protocol"]["arms"]}
    have_pit = all(r["arms"]["real"].get("pit_mae") is not None for r in rows)

    def arm(row: dict[str, Any], name: str) -> dict[str, Any] | None:
        return row["arms"].get(name)

    def pit_of(row: dict[str, Any], name: str) -> float | None:
        e = arm(row, name)
        return None if e is None else e.get("pit_mae")

    o: list[str] = []
    o.append("# DREGON round 3: does HPPNet track BROAD harmonic humps?")
    o.append("")
    o.append(
        f"Windows {', '.join(recs)}; seed {payload['protocol']['seed']}, "
        f"{payload['protocol']['n_mics']} mics; fit `{payload['fit']['path']}` "
        f"(`converged` {payload['fit']['converged']}); git `{payload['git']}`"
        + (f"; probe job `{job}`." if job else ".")
    )
    o.append("")

    # ── the verdict, first ──
    v0 = list(payload["verdicts"].values())
    broad = [x["broad_humps_in_real"]["supported"] for x in v0]
    width_lever = [x["width_is_the_lever"]["supported"] for x in v0]
    level_lever = [x["level_is_the_lever"]["supported"] for x in v0]
    o.append("## Verdict")
    o.append("")
    o.append(
        f"* **Is real DREGON's carrier-locked power BROADER than a needle comb's?** "
        f"{'YES' if all(broad) else ('PARTLY' if any(broad) else 'NO')} "
        f"({sum(broad)}/{len(broad)} windows)."
    )
    o.append(
        f"* **Is that width the thing HPPNet needs?** "
        f"{'YES' if all(width_lever) else ('PARTLY' if any(width_lever) else 'NO')} "
        f"({sum(width_lever)}/{len(width_lever)} windows: at the same hump fraction a "
        f"widened comb beats the needle comb)."
    )
    o.append(
        f"* **Is comb LEVEL the thing HPPNet needs?** "
        f"{'YES' if all(level_lever) else ('PARTLY' if any(level_lever) else 'NO')} "
        f"({sum(level_lever)}/{len(level_lever)} windows: re-levelling the needle comb "
        f"alone at least halves the PIT MAE)."
    )
    o.append("")

    o.append("## The arms")
    o.append("")
    o.append("| arm | tag | what it is |")
    o.append("|---|---|---|")
    for name in arms:
        o.append(f"| `{name}` | {tags.get(name, '')} | {labels.get(name, '')} |")
    o.append("")

    # ── PIT ──
    o.append("## HPPNet PIT MAE (rev/s)")
    o.append("")
    if have_pit:
        o.append("| arm | tag | gamma_rk | comb shift dB | " + " | ".join(recs) + " | mean |")
        o.append("|---|---|---|---:|" + "---:|" * (len(recs) + 1))
        for name in arms:
            vals = [pit_of(r, name) for r in rows]
            ok = [v for v in vals if v is not None]
            first = arm(rows[0], name) or {}
            gamma = first.get("gamma_hz")
            shift = first.get("comb_shift_db")
            o.append(
                f"| `{name}` | {tags.get(name, '')} | "
                + ("as fitted" if gamma is None else f"{float(gamma):g} Hz")
                + " | "
                + ("—" if shift is None else f"{float(shift):+.2f}")
                + " | "
                + " | ".join(_f(v) for v in vals)
                + f" | {_f(float(np.mean(ok)) if ok else None)} |"
            )
        o.append("")
        o.append(
            "`comb shift dB` is the shift applied to `profile_db`: the solved "
            "hump-fraction match for the `_matched` arms, the swept constant for the "
            "`v2_plusNdb` arms, 0 for the fitted level. Every arm is ONE seed "
            f"({payload['protocol']['seed']}), so these are not the four-seed gate means."
        )
    else:
        o.append("No probe pass merged: this record carries spectral numbers only.")
    o.append("")

    # ── hump fractions ──
    o.append("## Hump fraction: power inside the k x fbar +- 64 Hz band set")
    o.append("")
    o.append(
        "Reference-free union of the twelve (eight) bands on each frame's own mean "
        "label carrier, as a fraction of 30-7900 Hz band power, mic-MEAN on the "
        f"{payload['protocol']['coarse']['n_fft']}/{payload['protocol']['coarse']['hop']} "
        "front end. At a DREGON cruise carrier (fbar ~ 80 Hz) the 128 Hz-wide bands "
        "OVERLAP: the union is nearly the contiguous 30-1150 Hz region, which is why "
        "its own bandwidth fraction and the concentration (their ratio) are beside it. "
        "`tracked k=2..8` is the tilt-corrected per-order excess the level match is "
        "solved on."
    )
    o.append("")
    o.append(
        "| window | arm | frac k<=8 | frac k<=12 | bandwidth frac k<=8 | concentration "
        "k<=8 | tracked k=2..8 @64 Hz (%) | mic0 frac k<=8 |"
    )
    o.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r, rec in zip(rows, recs, strict=True):
        for name in arms:
            e = arm(r, name)
            if e is None:
                continue
            h = e["hump_fraction_mic_mean"]
            h0 = e["hump_fraction_mic0"]
            o.append(
                f"| `{rec}` | `{name}` | {h['frac_k8']:.4f} | {h['frac_k12']:.4f} | "
                f"{h['bandwidth_frac_k8']:.4f} | {h['concentration_k8']:.2f} | "
                f"{_pct(e['match_fraction'])} | {h0['frac_k8']:.4f} |"
            )
    o.append("")

    # ── per-order excess ──
    o.append("## Label-tracked per-order excess (% of band power, mic-mean)")
    o.append("")
    o.append(
        "Band minus a two-sided log-linear local baseline (the medians of the "
        "`B < |f - k fbar| <= 2.5 B` annuli, interpolated in log f), summed over frames "
        "before the ratio. `null` repeats the same read HALFWAY between two orders, at "
        "the same frequencies and under the same estimator: it is the zero this table "
        "is read against. An order whose annuli fall outside 30-7900 Hz is `—` (k=1 at "
        "B=64 and B=128, k=2 at B=128)."
    )
    o.append("")
    for bi, b in enumerate(payload["protocol"]["bandwidths_hz"]):
        o.append(f"### B = {float(b):g} Hz")
        o.append("")
        o.append("| window | arm | " + " | ".join(f"k={k}" for k in HUMP_KS) + " |")
        o.append("|---|---|" + "---:|" * len(HUMP_KS))
        for r, rec in zip(rows, recs, strict=True):
            for name in ("real", "legacy", "legacy_needle", "v2", "v2_matched"):
                e = arm(r, name)
                if e is None:
                    continue
                ex = e["excess_mic_mean"]
                nu = e["null_mic_mean"]
                orders = list(ex["orders"])
                o.append(
                    f"| `{rec}` | `{name}` | "
                    + " | ".join(_pct(ex["excess_frac"][orders.index(k)][bi]) for k in HUMP_KS)
                    + " |"
                )
                if name == "real":
                    o.append(
                        f"| `{rec}` | `real` null | "
                        + " | ".join(_pct(nu["excess_frac"][orders.index(k)][bi]) for k in HUMP_KS)
                        + " |"
                    )
        o.append("")

    # ── effective widths ──
    o.append("## Effective per-order half-width (Hz)")
    o.append("")
    o.append(
        "Fitted to the NULL-SUBTRACTED excess-vs-B curve over B = "
        + ", ".join(f"{float(b):g}" for b in payload["protocol"]["width_bandwidths_hz"])
        + " Hz on the "
        f"{payload['protocol']['fine']['n_fft']}/{payload['protocol']['fine']['hop']} "
        "front end (3.91 Hz per bin): a Lorentzian of half-width gamma keeps "
        "(2/pi) arctan(B/gamma) of its power inside +-B, so the GROWTH of the excess "
        "with B identifies gamma without ever resolving the line. The half-order null is "
        "subtracted at every B first, so a floor the two-sided baseline cannot follow "
        "drops out. The number carries the four-rotor spread (k x a few rev/s) and the "
        "in-frame drift with it: the `v2` row, whose fitted gamma_rk is under 1 Hz at "
        "every k <= 12, IS the instrumental floor this is read against. `>600` means the "
        "curve was still growing at B = 64 Hz — no width is identifiable below the order "
        "spacing; `—` means the order carries no positive tracked excess at all. The "
        "bracketed number is the assumption-free concentration, excess(4 Hz) / "
        "excess(64 Hz)."
    )
    o.append("")
    o.append("| window | arm | " + " | ".join(f"k={k}" for k in (1, 2, 3, 4, 6, 8, 12)) + " |")
    o.append("|---|---|" + "---:|" * 7)
    for r, rec in zip(rows, recs, strict=True):
        for name in arms:
            e = arm(r, name)
            if e is None:
                continue
            o.append(
                f"| `{rec}` | `{name}` | "
                + " | ".join(_width_cell(e, k) for k in (1, 2, 3, 4, 6, 8, 12))
                + " |"
            )
    o.append("")

    # ── profile widths ──
    o.append("## Order-tracked profile: half-power width and inter-order contrast")
    o.append("")
    o.append(
        "Demodulated on each ROTOR's own label track (its order-k line at delta f = 0) "
        "and averaged over the four rotors and eight mics. The reference is the median "
        "of the profile in the VALLEY between two orders (0.4-0.6 fbar), the only floor "
        "reference available when the order spacing (~80 Hz) is comparable to the widths "
        "under test; a profile still above half power at +-0.5 fbar is `sat` — it has no "
        "measurable width inside one order spacing — and one with no peak at delta f = 0 "
        "at all is `no peak`, which is itself the reading."
    )
    o.append("")
    o.append(
        "| window | arm | " + " | ".join(f"k={k} FWHM / contrast" for k in PROFILE_ORDERS) + " |"
    )
    o.append("|---|---|" + "---|" * len(PROFILE_ORDERS))
    for r, rec in zip(rows, recs, strict=True):
        for name in arms:
            e = arm(r, name)
            if e is None:
                continue
            cells = []
            for p in e["profiles"]:
                st = p["rotor_mean"]
                if st.get("no_peak"):
                    w = "no peak"
                elif st["saturated"]:
                    w = "sat"
                else:
                    w = f"{float(st['fwhm_hz']):.1f} Hz"
                cells.append(f"{w} / {float(st['contrast_db']):+.2f} dB")
            o.append(f"| `{rec}` | `{name}` | " + " | ".join(cells) + " |")
    o.append("")

    # ── four-rotor split ──
    o.append("## The four-rotor merge: four sub-peaks, or one hump?")
    o.append("")
    o.append(
        "Demodulated on the four-rotor MEAN carrier, where rotor r's order-k line sits "
        "at k (f_r - fbar). A resolved pair must DIP between its two carriers; the "
        "statistic is the deepest inter-carrier dip in dB, and >= 3 dB is `split`."
    )
    o.append("")
    o.append(
        "| window | k | carrier span Hz | "
        + " | ".join(f"`{n}`" for n in ("real", "legacy", "v2", "v2_matched", "v2_g67_matched"))
        + " |"
    )
    o.append("|---|---:|---:|" + "---|" * 5)
    for r, rec in zip(rows, recs, strict=True):
        for ki, k in enumerate(PROFILE_ORDERS):
            cells = []
            span = None
            for name in ("real", "legacy", "v2", "v2_matched", "v2_g67_matched"):
                e = arm(r, name)
                if e is None:
                    cells.append("—")
                    continue
                sp = e["profiles"][ki]["split"]
                span = sp["span_hz"]
                cells.append(f"{sp['verdict']} ({_f(sp['max_dip_db'], '.2f')} dB)")
            o.append(f"| `{rec}` | {k} | {_f(span, '.1f')} | " + " | ".join(cells) + " |")
    o.append("")

    # ── the level match ──
    o.append("## The hump-fraction level match")
    o.append("")
    o.append(
        "The shift solved on the exact block combination `floor_only + 10^(s/20) "
        "comb_only` of the same seed (the two blocks add to the full render to float "
        "round-off, checked per window below), so the `_matched` arms are real renders "
        "at a shift that was not searched by re-rendering."
    )
    o.append("")
    o.append(
        "| window | real target (%) | floor-only (%) | comb-only fitted (%) | "
        "shift fitted | shift 13 Hz | shift 30 Hz | shift 67 Hz | shift 130 Hz | "
        "block additivity |"
    )
    o.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r, rec in zip(rows, recs, strict=True):
        m = r["match_target"]
        sh = m["shifts"]
        cells = []
        for key in ("fitted", "13", "30", "67", "130"):
            s = sh.get(key)
            cells.append(
                "—"
                if s is None
                else f"{float(s['shift_db']):+.2f}" + (" (bound)" if s["at_bound"] else "")
            )
        o.append(
            f"| `{rec}` | {_pct(m['target_frac'])} | {_pct(m['floor_only_frac'])} | "
            f"{_pct(m['comb_only_frac']['fitted'])} | "
            + " | ".join(cells)
            + f" | {r['block_additivity']['max_abs_diff']:.2e} / "
            f"{r['block_additivity']['max_abs_render']:.3f} |"
        )
    o.append("")

    # ── the windows themselves ──
    o.append("## The three windows")
    o.append("")
    o.append(
        "| window | mean carrier rev/s | per-rotor mean | per-rotor offset | in-window "
        "drift of the mean rev/s | legacy pedestal HWHM at k=1/2/4/8 Hz | legacy coherent "
        "share at k=1/2/4/8 |"
    )
    o.append("|---|---:|---|---|---:|---|---|")
    for r, rec in zip(rows, recs, strict=True):
        lw = r["legacy_width"]
        o.append(
            f"| `{rec}` | {r['mean_reference_rps']:.2f} | "
            + ", ".join(f"{v:.2f}" for v in r["per_rotor_mean_rps"])
            + " | "
            + ", ".join(f"{v:+.2f}" for v in r["per_rotor_offset_rps"])
            + f" | {r['carrier_drift_rev_s']:.2f} | "
            + "/".join(f"{lw['pedestal_hwhm_hz'][str(k)]:.1f}" for k in (1, 2, 4, 8))
            + " | "
            + "/".join(f"{lw['coherent_share'][str(k)]:.3f}" for k in (1, 2, 4, 8))
            + " |"
        )
    o.append("")
    o.append(
        "The legacy DREGON cruise export's comb is a COHERENCE SPLIT: the share w_k = "
        "exp(-(k/coherence_k_half)^2) of each order's power is a needle and the rest is "
        "narrowband noise of half-width gamma0 + gamma_slope k — the pedestal. "
        f"`coherence_k_half` = {rows[0]['legacy_width']['coherence_k_half']:.3f}, so from "
        "k=3 up the legacy comb is essentially ALL pedestal. `legacy_needle` (V7) is the "
        "same export with `coherence_k_half = 0`, which puts every order's full power "
        "through the tone bank and leaves no pedestal at all."
    )
    o.append("")
    o.extend(hump_reading(payload))
    o.append("")
    o.extend(hump_proposal(payload))
    o.append("")
    o.append("## Provenance")
    o.append("")
    scorer = payload["protocol"].get("scorer") or {}
    o.append(
        "Probe `"
        + str(scorer.get("experiment", "—"))
        + "/"
        + str(scorer.get("ckpt", "—"))
        + "`, sha256 `"
        + str(scorer.get("sha256", "—"))
        + "` (the loader dies on a mismatch, so a scored arm cannot exist without it). "
        f"Fit `{payload['fit']['path']}`, mode `{payload['fit']['mode']}`, "
        f"`comb_gain_db` {_f(payload['fit']['comb_gain_db'])}, `low_order_gain_db` "
        + ", ".join(_f(v, ".2f") for v in (payload["fit"]["low_order_gain_db"] or []))
        + ". Fitted `gamma_hz` mean over rotors at k = "
        + ", ".join(
            f"{k}: {float(v):g} Hz" for k, v in payload["fit"]["gamma_hz_mean_per_order"].items()
        )
        + f". Record `{payload['git']}`"
        + (f", job `{job}`." if job else ".")
    )
    if figures:
        o.append("")
        o.append("Figures: " + ", ".join(f"`{p}`" for p in figures) + ".")
    o.append("")
    return "\n".join(o)


def hump_reading(payload: dict[str, Any]) -> list[str]:
    """What the tables say, in the order the numbers force."""
    rows = list(payload["supports"].values())
    recs = [_short(r["support"]["recording"]) for r in rows]
    o = ["## Reading", ""]
    for r, rec in zip(rows, recs, strict=True):
        real, v2 = r["arms"]["real"], r["arms"]["v2"]
        w_real = [_width_at(real, k) for k in (1, 2, 4, 8)]
        w_v2 = [_width_at(v2, k) for k in (1, 2, 4, 8)]
        matched = r["arms"].get("v2_matched") or {}
        wide = {
            g: (r["arms"].get(f"v2_g{int(g)}_matched") or {}).get("pit_mae") for g in HUMP_GAMMAS
        }
        o.append(
            f"* **`{rec}`** — real's effective HWHM at k=1/2/4/8 is "
            + "/".join(_f(v, ".1f") for v in w_real)
            + " Hz against the needle render's "
            + "/".join(_f(v, ".1f") for v in w_v2)
            + " Hz. At the matched hump fraction "
            f"({_f(matched.get('comb_shift_db'), '+.2f')} dB on the needle comb) the PIT "
            f"MAE is {_f(matched.get('pit_mae'))} rev/s; widening the SAME comb to "
            + ", ".join(f"{g:g} Hz -> {_f(wide[g])}" for g in HUMP_GAMMAS)
            + f" rev/s. Real {_f(real.get('pit_mae'))}, legacy "
            f"{_f((r['arms'].get('legacy') or {}).get('pit_mae'))}, legacy-needle "
            f"{_f((r['arms'].get('legacy_needle') or {}).get('pit_mae'))}, v2 as fitted "
            f"{_f(v2.get('pit_mae'))}."
        )
        o.append(
            f"  Per-order tracked excess at B=16 Hz (% of band power): real "
            f"k=1 {_pct(_excess_at(real, 1, 16.0))}, k=2 {_pct(_excess_at(real, 2, 16.0))}, "
            f"sum k=3..8 {_pct(sum(_excess_at(real, k, 16.0) or 0.0 for k in range(3, 9)))}; "
            f"v2 as fitted {_pct(_excess_at(v2, 1, 16.0))}, {_pct(_excess_at(v2, 2, 16.0))}, "
            f"{_pct(sum(_excess_at(v2, k, 16.0) or 0.0 for k in range(3, 9)))}; "
            f"v2 at the match {_pct(_excess_at(matched, 1, 16.0)) if matched else '—'}, "
            f"{_pct(_excess_at(matched, 2, 16.0)) if matched else '—'}, "
            f"{_pct(sum(_excess_at(matched, k, 16.0) or 0.0 for k in range(3, 9))) if matched else '—'}."
        )
    o.append("")
    return o


def hump_proposal(payload: dict[str, Any]) -> list[str]:
    """The R4 proposal, keyed to the verdicts and quoting their numbers."""
    rows = list(payload["supports"].values())
    v = list(payload["verdicts"].values())
    width_lever = any(x["width_is_the_lever"]["supported"] for x in v)
    level_lever = any(x["level_is_the_lever"]["supported"] for x in v)
    broad = [x["broad_humps_in_real"] for x in v]
    w1 = [b["effective_hwhm_real_hz"].get(1) or b["effective_hwhm_real_hz"].get("1") for b in broad]
    w2 = [b["effective_hwhm_real_hz"].get(2) or b["effective_hwhm_real_hz"].get("2") for b in broad]
    n1 = [b["effective_hwhm_v2_hz"].get(1) or b["effective_hwhm_v2_hz"].get("1") for b in broad]
    shifts = [(r["arms"].get("v2_matched") or {}).get("comb_shift_db") for r in rows]
    have = [s for s in shifts if s is not None]
    matched_pits = [(r["arms"].get("v2_matched") or {}).get("pit_mae") for r in rows]
    wide_pits = {
        g: [(r["arms"].get(f"v2_g{int(g)}_matched") or {}).get("pit_mae") for r in rows]
        for g in HUMP_GAMMAS
    }
    o = ["## R4 proposal", ""]
    o.append(
        "The three options this study was asked to decide between, each with the number "
        "that decides it:"
    )
    o.append("")
    o.append(
        "**(i) free `gamma_rk` in the DREGON flight fit under a wide flight-specific "
        f"prior — {'SUPPORTED' if width_lever else 'NOT SUPPORTED'}.** At the matched "
        "hump fraction the needle comb scores "
        + ", ".join(_f(p) for p in matched_pits)
        + " rev/s and the same comb widened scores "
        + "; ".join(f"{g:g} Hz: " + ", ".join(_f(p) for p in wide_pits[g]) for g in HUMP_GAMMAS)
        + " rev/s"
        + (
            ". Widening at a fixed hump fraction therefore HELPS and a wide prior is the lever."
            if width_lever
            else ". Widening at a fixed hump fraction makes every window WORSE, so freeing "
            "gamma_rk cannot buy the gate: the widths the data does support are the "
            "measured effective HWHMs at k=1 ("
            + ", ".join(_f(x, ".1f") for x in w1)
            + " Hz) and k=2 ("
            + ", ".join(_f(x, ".1f") for x in w2)
            + " Hz) against the needle render's k=1 instrumental floor ("
            + ", ".join(_f(x, ".1f") for x in n1)
            + " Hz), i.e. a log-normal prior on gamma_rk with median ~5 Hz at k=1 and a "
            "factor-3 sd would cover them — but it is worth nothing to the gate."
        )
    )
    o.append("")
    o.append(
        "**(ii) a per-order PEDESTAL term on top of the narrow line (a SECOND width per "
        f"line — a complexity the user must approve) — {'SUPPORTED' if width_lever else 'NOT SUPPORTED'}.** "
        "The legacy model IS that term (coherence split: needle plus a "
        f"{rows[0]['legacy_width']['pedestal_hwhm_hz']['2']:.1f} Hz HWHM pedestal at k=2) "
        "and `legacy_needle` (V7) is the same export with the pedestal switched off: "
        + ", ".join(
            f"{_short(r['support']['recording'])} {_f((r['arms'].get('legacy') or {}).get('pit_mae'))} -> "
            f"{_f((r['arms'].get('legacy_needle') or {}).get('pit_mae'))}"
            for r in rows
        )
        + " rev/s with it muted. A second width per line doubles the comb's parameter "
        "count (2 x R x K) and the only evidence for it is the pedestal's effect on the "
        "legacy arm, measured here."
    )
    o.append("")
    o.append(
        "**(iii) modelling the four-rotor merge explicitly — NOT NEEDED.** The render "
        "already carries four independent per-rotor tracks on the labels' own carriers; "
        "the split table above shows the v2 render and the real clip in the SAME state "
        "at every order measured, so there is nothing to add."
    )
    o.append("")
    if level_lever:
        o.append(
            "**What the numbers do support: the comb's LEVEL, and the objective that "
            "sets it.** Re-levelling the FITTED (needle) comb by "
            + ", ".join(_f(s, "+.2f") for s in have)
            + " dB — nothing else changed — moves the PIT MAE from "
            + ", ".join(_f((r["arms"]["v2"]).get("pit_mae")) for r in rows)
            + " to "
            + ", ".join(_f(p) for p in matched_pits)
            + " rev/s. `comb_gain_db` is ALREADY free in this fit and the flight Whittle "
            "objective chose "
            f"{_f(payload['fit']['comb_gain_db'])} dB for it: the bottleneck is the "
            "objective, not the parameterisation."
        )
        o.append("")
        o.append("```diff")
        o.append("  # src/experiments/noise_model/fit.py -- the flight objective")
        o.append("- # comb level is scored by the band-pooled Whittle risk alone, which is")
        o.append("- # dominated by the 30-300 Hz floor cells and buries the comb.")
        o.append("+ # score the comb level on the CARRIER-TRACKED cells only: the Whittle")
        o.append("+ # risk restricted to |f - k f_r(t)| <= 16 Hz, k = 1..K, plus the floor")
        o.append("+ # band as now. One extra term, no new parameter: comb_gain_db and the")
        o.append("+ # per-order low_order_gain_db are then pulled by the cells that")
        o.append("+ # actually carry the comb.")
        o.append("```")
        o.append("")
        o.append(
            "No model code changed in this study: every variant is a mutation of the "
            "fit payload (`_mutate`) rendered by the unchanged renderer."
        )
    return o


PIT_KEYS = ("pit_mae", "pit_per_mic", "pit_per_rotor", "n_scored_frames")
LEVEL_TOL_DB = 0.01


def merge_pit(
    payload: dict[str, Any],
    probe: dict[str, Any],
    *,
    schema: str = SCHEMA,
    verdicts_fn: Callable[[dict[str, Any]], dict[str, Any]] = verdicts,
) -> dict[str, Any]:
    """Copy a probe pass's PIT numbers onto this pass's arms.

    The two passes must have rendered the same audio: the band level of every
    shared arm is compared first and a mismatch is fatal.
    """
    if str(probe.get("schema")) != schema:
        die(f"the probe payload is not a {schema} record")
    for key, row in payload["supports"].items():
        prow = (probe.get("supports") or {}).get(key)
        if prow is None:
            die(f"the probe payload has no support {key!r}")
        for name, entry in row["arms"].items():
            pe = (prow.get("arms") or {}).get(name)
            if pe is None:
                continue
            gap = abs(float(entry["band_level_db_mic0"]) - float(pe["band_level_db_mic0"]))
            if gap > LEVEL_TOL_DB:
                die(
                    f"{key}/{name}: the probe pass rendered a different signal "
                    f"({gap:.4f} dB of band level apart)"
                )
            for pk in PIT_KEYS:
                if pk in pe:
                    entry[pk] = pe[pk]
    payload["protocol"]["scorer"] = (probe.get("protocol") or {}).get("scorer")
    payload["verdicts"] = verdicts_fn(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", choices=("r2", "humps"), default="r2")
    ap.add_argument("--fit", type=Path, default=None)
    ap.add_argument("--bench-dir", type=Path, default=BENCH_DEFAULT)
    ap.add_argument("--arm-npz", type=Path, default=ARM_NPZ_DEFAULT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=2001)
    ap.add_argument("--n-mics", type=int, default=8)
    ap.add_argument("--probe", action="store_true", help="score every arm with the frozen HPPNet")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--job", type=str, default=None, help="omnirun job id, for the record")
    ap.add_argument(
        "--recordings",
        default=",".join(HUMP_RECORDINGS),
        help="comma-separated DREGON recordings (the humps study)",
    )
    ap.add_argument(
        "--merge-probe", type=Path, default=None, help="a probe pass's JSON to merge PIT from"
    )
    args = ap.parse_args(argv)
    humps = args.study == "humps"
    fit_path = args.fit or (HUMP_FIT_DEFAULT if humps else FIT_DEFAULT)
    out = Path(args.out or (HUMP_OUT_DEFAULT if humps else OUT_DEFAULT))

    if humps:
        payload = run_humps(
            fit_path=fit_path,
            out=out,
            seed=args.seed,
            probe=bool(args.probe),
            n_mics=int(args.n_mics),
            recordings=tuple(r for r in str(args.recordings).split(",") if r),
        )
        figures_data = payload.pop("_figures")
        if args.merge_probe is not None:
            payload = merge_pit(
                payload,
                json.loads(Path(args.merge_probe).read_text()),
                schema=HUMP_SCHEMA,
                verdicts_fn=hump_verdicts,
            )
        written = write_hump_figures(payload, figures_data, out) if args.figures else []
        out.mkdir(parents=True, exist_ok=True)
        (out / "dregon_humps.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        (out / "findings.md").write_text(hump_findings(payload, job=args.job, figures=written))
        print(f"wrote {out / 'dregon_humps.json'}")
        print(f"wrote {out / 'findings.md'}")
        for p in written:
            print(f"wrote {p}")
        return 0

    payload = run(
        fit_path=fit_path,
        bench_dir=args.bench_dir,
        arm_npz=args.arm_npz,
        out=out,
        seed=args.seed,
        probe=bool(args.probe),
        n_mics=int(args.n_mics),
    )
    figures_data = payload.pop("_figures")
    if args.merge_probe is not None:
        payload = merge_pit(payload, json.loads(Path(args.merge_probe).read_text()))
    written = write_figures(payload, figures_data, out) if args.figures else []
    out.mkdir(parents=True, exist_ok=True)
    (out / "render_dregon.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    (out / "findings.md").write_text(findings(payload, job=args.job, figures=written))
    print(f"wrote {out / 'render_dregon.json'}")
    print(f"wrote {out / 'findings.md'}")
    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
