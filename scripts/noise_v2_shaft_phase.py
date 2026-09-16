"""What prior does the shaft-phase noise of the revised model deserve?

The revised rotor-noise model (``docs/explainers/bench-fit-revised-phase-model.qmd``
§3) says the shaft-angle error of rotor ``r`` is the integral of a stationary
OU speed error,

    dnu_r = -lam_r nu_r dt + sqrt(2 lam_r sigma_r^2) dW,
    theta_r(t) = int nu_r,

so that ``V_theta(tau) = 2 sigma^2 [tau/lam - (1 - exp(-lam tau))/lam^2]``,
and harmonic ``k`` of that rotor additionally carries an independent Wiener
phase of diffusion ``D_rk``, so that

    V_k(tau) = k^2 V_theta(tau) + 2 D_k tau.                            (*)

C2/C3 fix ``lam_ref = 6 1/s`` and fit one ``sigma_g`` and one ``D_g`` per rig.
This script measures, from data, whether ``theta`` really is integrated-OU
(speed error correlated over ``1/lam``) or plain Wiener (white speed error),
and at what ``(sigma_nu, lam, D)``, with two INDEPENDENT instruments.

Instrument 1 - TELEMETRY at the NATIVE rate (never the 100 Hz campaign grid):
    ``NeuroBEM/Blackbird/VID/NanoBench/PITCN-frames`` ``rps``, DREGON room1
    ``motors_measured`` (~929 Hz, a ~45 Hz sample-and-hold) and Michael's
    FLY124/FLY125 ``rps`` (~29 Hz). Per rig and rotor it reports
      * the speed-error series ``nu = 2 pi (rps - slow trend)``, the slow trend
        removed by a ZERO-PHASE 4th-order Butterworth high-pass (``sosfiltfilt``,
        so the power response is ``|H|^4``) at 0.5 Hz, and again at 2 Hz;
      * the phase structure function ``S(tau) = Var[theta(t+tau) - theta(t)]``
        at 40 log-spaced lags from ``1/fs`` to 10 s, its local log-log slope,
        and the lag where that slope crosses 1.5 (the integrated-OU slope-1.5
        point sits at ``lam tau = 2.1489``, so that lag fixes ``1/lam``);
      * the speed-error Welch PSD with a Lorentzian (OU) and a flat (white)
        model fitted on 0.5 Hz - 0.4 fs THROUGH the known high-pass response,
        scored by the Gamma (Whittle) likelihood, AIC and BIC;
      * the ESC quantisation floor: the step of the telemetry ladder (declared
        by the source where known, else read off the histogram of successive
        differences), the rate at which the value actually CHANGES, and the
        white-speed-noise-equivalent diffusion ``D_q`` it implies - so a
        Wiener-looking tail can be attributed to quantisation, or not.

Instrument 2 - ACOUSTICS on stationary supports: DREGON's single-motor bench
    recordings ``motor_Motor{1-4}_{70,80,90}`` (constant throttle, no label to
    be wrong) and Michael's FLY125 cruise windows (telemetry carrier, 8 mics).
    Orders ``k = 1..8`` are demodulated with the project's own heterodyne
    (``experiments.stochastic_fit.phase_stats.stft_at``, exact-centre DTFT with
    a frame-start phase reference). The lag statistic is the baseband
    coherence, which is what the model itself predicts
    (``R_k(tau) = exp(-V_k(tau)/2)``, §3.3) and which cannot be wrap-censored,
    plus an unwrapped-increment cross-check. ``(*)`` is then fitted JOINTLY
    over ``k`` - the rank-one-plus-diagonal decomposition of §5.1 - against
    four rivals (white shaft; ``D`` constant / prop k / prop k^2).

Every number and figure in ``results/noise_v2/shaft/`` comes from this file.

Usage:
    python scripts/noise_v2_shaft_phase.py                       # everything
    python scripts/noise_v2_shaft_phase.py --rigs michaels --limit 1   # smoke
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.signal import butter, sosfiltfilt, sosfreqz, welch  # noqa: E402

# ─── frozen analysis constants ────────────────────────────────────────────────

#: High-pass corners for the slow-trend removal. The FIRST is the headline.
HP_HZ = (0.5, 2.0)
#: Trend-removal variants actually run: the two mandated high-passes plus
#: ``None`` = per-segment LINEAR DETREND only. The third is not cosmetic. A
#: zero-phase high-pass at ``f_hp`` annihilates the spectrum below ``f_hp``
#: (the power response is ``|H|^4``, i.e. ``(f/f_hp)^16`` at low ``f``), so a
#: Lorentzian corner below 0.5 Hz - which is where a quadrotor's controller
#: actually lives - can NEVER be located from the high-passed series, no
#: matter how the model is corrected. The detrend-only variant is the only
#: instrument in this script that can see it, and its band starts at
#: :data:`PSD_F_LO_WIDE_HZ`.
HP_VARIANTS: tuple[float | None, ...] = (0.5, 2.0, None)
#: Key a variant carries in the JSON.
DETREND_KEY = "detrend"
#: Butterworth order of that high-pass, applied zero-phase with ``sosfiltfilt``
#: (so the POWER response is ``|H|^4``, which every model here multiplies by).
HP_ORDER = 4
#: Structure-function lag grid: 40 log-spaced lags, ``1/fs`` .. ``S_LAG_MAX``.
S_N_LAGS = 40
S_LAG_MAX_S = 10.0
#: PSD fit band: ``[PSD_F_LO, PSD_F_HI_FRAC * fs]``.
PSD_F_LO_HZ = 0.5
PSD_F_HI_FRAC = 0.4
#: Low edge of the detrend-only variant's fit band, Hz.
PSD_F_LO_WIDE_HZ = 0.05
#: Welch segment length in seconds and overlap fraction for the speed PSD.
PSD_SEG_S = 20.0
PSD_OVERLAP = 0.5
#: Effective-independence penalty of 50 %-overlapped Hann segments: the
#: normalised window autocorrelation at half-window lag is 1/6, so the
#: variance of the average is inflated by ``1 + 2 (1/6)^2`` over 1/n_seg.
WELCH_OVERLAP_INFLATION = 1.0 + 2.0 * (1.0 / 6.0) ** 2
#: ``lam * tau`` at which the integrated-OU structure function has local
#: log-log slope 1.5 (solved in :func:`ou_slope_1p5_u`).
SLOPE_TARGET = 1.5
#: Minimum airborne/stationary segment length kept, seconds.
MIN_SEG_S = 5.0

#: Acoustic demodulation. The analysis window must resolve neighbouring orders
#: (Hann main-lobe half width ``2/T_w`` must be below half the order spacing),
#: so its length is tied to the shaft period: ``DEMOD_REVS`` revolutions.
DEMOD_REVS = 6.0
#: Frames per window (8 -> 87.5 % overlap), i.e. hop = window / DEMOD_OVERSAMP.
DEMOD_OVERSAMP = 8
#: Largest phase-increment variance at which the unwrapped estimator is still
#: unbiased, rad^2. Measured on the planted control: inside 0.15 rad^2 it
#: returns ``V`` to within 10 % and ``D_k`` and its ``k`` exponent correctly;
#: at 5 rad^2 it returns 0.4x the truth, because the phase of a windowed sum
#: of phasors stops tracking the window-weighted mean phase.
ACOUSTIC_V_LINEAR_MAX = 0.15
#: Relative systematic error of the windowed forward model, measured on the
#: planted control. It is the floor on every acoustic cell's error bar.
ACOUSTIC_V_REL_SYSTEMATIC = 0.10
#: Orders demodulated.
K_MAX_ACOUSTIC = 8
#: In-band detection bar, dB, measured in the DEMODULATION band itself
#: (order power over the power at the half-order offset). The preregistered
#: profile bar of ``experiments.stochastic_fit.bench`` is the same 6 dB.
MARGIN_MIN_DB = 6.0
#: Coherence-based ``V`` is only trusted while the squared coherence stands
#: clear of its own sampling floor by this factor.
COH_FLOOR_FACTOR = 3.0
#: Acoustic lag grid, seconds (the script clips it to what a support supports).
ACOUSTIC_LAG_LO_S = 1e-3
ACOUSTIC_LAG_HI_S = 2.0
ACOUSTIC_N_LAGS = 22
#: Analysis window for a MULTI-ROTOR flight support, seconds. Four rotors put
#: their combs 1-30 Hz apart, so the window is set by that spread (Hann
#: main-lobe half width ``2/T_w`` = 4 Hz here, so an order must stand 8 Hz
#: clear of every other line) and not by the shaft period. The twin rotors of
#: a quad are 0.6 Hz apart at order 1 and are never resolvable inside a 16 s
#: window; the isolation gate drops them, as it should. This is the binding
#: compromise of the acoustic instrument on a flight rig: the window must be
#: LONG to resolve one rotor from another, and the usable lag grid starts at
#: one window, so a long window also destroys the short-lag knee that
#: identifies ``lam``.
ACOUSTIC_FLIGHT_WINDOW_S = 0.5

#: DREGON's single-motor bench throttle law, rev/s from the setpoint.
DREGON_THROTTLE_A = 0.975
DREGON_THROTTLE_B = 0.37
DREGON_BENCH_SETPOINTS = (70, 80, 90)
#: Bench analysis length, seconds (the shortest of these recordings is 25 s).
BENCH_SECONDS = 24.0

#: Michael's acoustic support: FLY125 cruise, the accepted flight fit's
#: geometry (8 windows of 16 s, every rotor above 65 rev/s).
MICHAELS_REC = "FLY125"
MICHAELS_DATASET = "michaels-frames"
MICHAELS_CLIP_S = 16.0
MICHAELS_MAX_CLIPS = 8
MICHAELS_MIN_RPS = 65.0
#: RAW calibrated ESC track, NOT ``rps_refined``: the refined label has already
#: absorbed part of the shaft phase error into the carrier, which is the very
#: quantity being measured here.
MICHAELS_RPS_KEY = "rps"

#: The fitted C3 values this study is compared against (frozen facts).
C3_REFERENCE = {
    "dregon": {"sigma_rad_s": 3.101, "D_rad2_s": 844.69, "lam_ref": 6.0},
    "michaels": {"sigma_rad_s": 4.207, "D_rad2_s": 1.631, "lam_ref": 6.0},
}

#: Telemetry rigs: JSON key -> (frames dataset, rps entry names, rig filter).
TELEMETRY_RIGS: dict[str, dict[str, Any]] = {
    "neurobem_quad": {"dataset": "NeuroBEM-frames", "keys": ("rps",), "rig": "neurobem_quad"},
    "blackbird_quad": {"dataset": "Blackbird-frames", "keys": ("rps",), "rig": "blackbird_quad"},
    "vid_m100": {"dataset": "VID-frames", "keys": ("rps",), "rig": "vid_m100"},
    "nanobench_cf21b": {"dataset": "NanoBench-frames", "keys": ("rps",), "rig": "nanobench_cf21b"},
    "pitcn_quad": {"dataset": "PITCN-frames", "keys": ("rps",), "rig": "pitcn_quad"},
    "dregon_room1": {
        "dataset": "DREGON-frames",
        "keys": ("motors_measured",),
        "rig": None,
        "select": "room1",
    },
    "dregon_room1_command": {
        "dataset": "DREGON-frames",
        "keys": ("motors_command",),
        "rig": None,
        "select": "room1",
        "auxiliary": True,
    },
    "michaels": {"dataset": "michaels-frames", "keys": ("rps",), "rig": None},
}

#: Declared telemetry ladder step (rev/s) where the source module states it.
#: Anything not listed is measured from the data alone.
DECLARED_STEP_RPS: dict[str, dict[str, Any]] = {
    "nanobench_cf21b": {
        "step_rps": 100.0 / (6.0 * 60.0),
        "why": "Crazyflie log field is eRPM/100; eRPM / 6 pole pairs / 60 (sources/nanobench.py)",
    },
    "vid_m100": {
        "step_rps": 1.0 / 60.0,
        "why": "C620 CAN int16 rpm field, rev/s = rpm/60 (sources/vid.py)",
    },
    "blackbird_quad": {
        "step_rps": 1.0 / 60.0,
        "why": "tachometer RPM / 60 (sources/blackbird.py)",
    },
    "michaels": {
        "step_rps": 1.00706 / 60.0,
        "why": "DatCon Motor:Speed integer RPM / 60 times MICHAELS_RPS_SCALE 1.00706",
    },
}

#: Acoustic supports.
ACOUSTIC_SUPPORTS = ("dregon_bench", "michaels_fly125")

#: Rig-name aliases accepted by ``--rigs``.
ALIASES: dict[str, tuple[str, ...]] = {
    "all": tuple(TELEMETRY_RIGS) + ACOUSTIC_SUPPORTS,
    "telemetry": tuple(TELEMETRY_RIGS),
    "acoustics": ACOUSTIC_SUPPORTS,
    "michaels": ("michaels", "michaels_fly125"),
    "dregon": ("dregon_room1", "dregon_room1_command", "dregon_bench"),
}


# ═════════════════════════════════════════════════════════════════════════════
# The model's own algebra
# ═════════════════════════════════════════════════════════════════════════════


def ou_structure(tau: np.ndarray, sigma: float, lam: float) -> np.ndarray:
    """``V_theta(tau) = 2 sigma^2 [tau/lam - (1 - exp(-lam tau))/lam^2]``.

    Evaluated through ``expm1`` because at ``lam tau << 1`` the two bracket
    terms cancel to every significant digit of a naive difference (§3.3).
    """
    tau = np.abs(np.asarray(tau, dtype=np.float64))
    u = lam * tau
    # tau/lam - (1 - e^-u)/lam^2 = (u - 1 + e^-u)/lam^2, and
    # u - 1 + e^-u = u + expm1(-u) is stable: both terms are O(u) and the
    # cancellation leaves u^2/2 which expm1 delivers to full precision.
    return 2.0 * sigma**2 * (u + np.expm1(-u)) / lam**2


def ou_slope(u: np.ndarray) -> np.ndarray:
    """Local log-log slope of :func:`ou_structure` at ``u = lam tau``."""
    u = np.asarray(u, dtype=np.float64)
    num = u * -np.expm1(-u)
    den = u + np.expm1(-u)
    return np.where(den > 0, num / np.maximum(den, 1e-300), 2.0)


def ou_slope_1p5_u() -> float:
    """``lam tau`` where the integrated-OU log-log slope equals 1.5."""
    lo, hi = 1e-6, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if ou_slope(np.array([mid]))[0] > SLOPE_TARGET:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


U_SLOPE_1P5 = ou_slope_1p5_u()


def ou_psd(f: np.ndarray, sigma: float, lam: float) -> np.ndarray:
    """One-sided PSD of the OU speed error, ``(rad/s)^2/Hz``.

    ``int_0^inf S df = sigma^2`` for ``S(f) = 4 sigma^2 lam / (lam^2 + 4 pi^2 f^2)``.
    """
    f = np.asarray(f, dtype=np.float64)
    return 4.0 * sigma**2 * lam / (lam**2 + (2.0 * np.pi * f) ** 2)


def d_from_white_psd(s0: float) -> float:
    """Wiener diffusion of ``theta`` implied by a FLAT speed PSD ``s0``.

    White ``nu`` with one-sided density ``s0`` integrates to
    ``Var[theta(t+tau) - theta(t)] = s0 tau / 2 = 2 D tau``, so ``D = s0/4``.
    """
    return 0.25 * float(s0)


def hp_sos(fs: float, f_hp: float) -> np.ndarray:
    return butter(HP_ORDER, f_hp / (0.5 * fs), btype="highpass", output="sos")


def hp_power_response(sos: np.ndarray, f: np.ndarray, fs: float) -> np.ndarray:
    """Power response of ``sosfiltfilt``: forward+backward gives ``|H|^2`` on the
    amplitude, hence ``|H|^4`` on the power."""
    w = 2.0 * np.pi * np.asarray(f, dtype=np.float64) / fs
    _, h = sosfreqz(sos, worN=w)
    return np.abs(h) ** 4


def filtered_theta_structure(
    tau: np.ndarray, sigma: float, lam: float, resp: np.ndarray, f_grid: np.ndarray
) -> np.ndarray:
    """``Var[theta(t+tau) - theta(t)]`` when ``nu`` has been high-passed.

    After the high-pass ``nu`` carries no DC, so ``theta = int nu`` is itself
    stationary with PSD ``S_nu(f) resp(f) / (2 pi f)^2`` and
    ``S(tau) = 2 int S_theta(f) [1 - cos(2 pi f tau)] df``. This is what makes
    the measured ``S(tau)`` saturate instead of growing forever, and it is the
    only honest curve to draw next to the data.
    """
    f = np.asarray(f_grid, dtype=np.float64)
    s_theta = ou_psd(f, sigma, lam) * resp / (2.0 * np.pi * f) ** 2
    tau = np.atleast_1d(np.asarray(tau, dtype=np.float64))
    kern = 1.0 - np.cos(2.0 * np.pi * f[None, :] * tau[:, None])
    return 2.0 * np.trapezoid(s_theta[None, :] * kern, f, axis=1)


# ═════════════════════════════════════════════════════════════════════════════
# Telemetry loading at the NATIVE rate
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class NativeTrack:
    rig: str
    flight: str
    key: str
    fs: float
    rps: np.ndarray  # (R, T) rev/s on a uniform fs grid, NaN where missing
    declared_rate_hz: float | None
    raw_rps: np.ndarray  # (R, N) as logged, before any regridding
    raw_dt: np.ndarray  # (N-1,) logged sample spacing, seconds


def _meta_dict(frame: Any) -> dict[str, Any]:
    from data_processing.frames import meta_dict

    return meta_dict(frame)


def _system_group(meta: dict[str, Any]) -> dict[str, Any]:
    group = meta.get("system")
    if group is None:
        return {}
    try:
        return {k: group[k] for k in group}
    except TypeError:
        return {}


def load_native(rig: str, spec: dict[str, Any], limit: int | None) -> list[NativeTrack]:
    """Every flight of one telemetry rig, at its OWN logging rate.

    Non-uniformly stamped tracks (VID, Blackbird, DREGON, Michael's) are put on
    a uniform grid at their own median rate by linear interpolation - the
    minimum needed for a PSD and a structure function - and NEVER decimated to
    the campaign's 100 Hz grid.
    """
    from data_processing.streams import iter_published_frames

    out: list[NativeTrack] = []
    select = spec.get("select")
    for frame in iter_published_frames(spec["dataset"]):
        meta = _meta_dict(frame)
        rid = str(meta.get("recording_id"))
        if select is not None and select not in rid:
            continue
        sysd = _system_group(meta)
        want = spec.get("rig")
        if want is not None and sysd.get("rig") not in (None, want):
            continue
        key = next((k for k in spec["keys"] if k in frame), None)
        if key is None:
            continue
        series = frame[key]
        values = np.asarray(series.data, dtype=np.float64)
        values = values[None, :] if values.ndim == 1 else values
        if values.shape[-1] < 64 or values.shape[0] != 4:
            continue
        idx = series.indexes["time"]
        sr = getattr(idx, "sr", None)
        if sr:
            fs = float(sr)
            grid = values
            dt = np.full(values.shape[-1] - 1, 1.0 / fs)
        else:
            t = np.asarray(idx.timestamps, dtype=np.float64)
            dt = np.diff(t)
            fs = 1.0 / float(np.median(dt))
            n = int(np.floor((t[-1] - t[0]) * fs)) + 1
            tg = t[0] + np.arange(n) / fs
            grid = np.stack([np.interp(tg, t, row) for row in values])
        declared = sysd.get("native_rate_hz")
        out.append(
            NativeTrack(
                rig=rig,
                flight=rid,
                key=key,
                fs=fs,
                rps=grid,
                declared_rate_hz=float(declared) if declared else None,
                raw_rps=values,
                raw_dt=dt,
            )
        )
        if limit is not None and len(out) >= limit:
            break
    return sorted(out, key=lambda t: t.flight)


def _interp_nans(y: np.ndarray) -> tuple[np.ndarray, float]:
    bad = ~np.isfinite(y)
    if not bad.any():
        return y, 0.0
    if bad.all():
        return y, 1.0
    idx = np.arange(y.size)
    filled = y.copy()
    filled[bad] = np.interp(idx[bad], idx[~bad], y[~bad])
    return filled, float(bad.mean())


def stationary_segments(rps: np.ndarray, fs: float) -> list[slice]:
    """Airborne segments by the campaign's frozen rule, at the NATIVE rate."""
    from experiments.rps_traj.data import airborne_segments

    segs = airborne_segments(rps, fs)
    return [s for s in segs if (s.stop - s.start) / fs >= MIN_SEG_S]


# ═════════════════════════════════════════════════════════════════════════════
# Quantisation
# ═════════════════════════════════════════════════════════════════════════════


def quantisation(track: NativeTrack) -> dict[str, Any]:
    """The telemetry ladder: its step, its update rate and the ``D_q`` it implies.

    The step is read off the histogram of successive RAW differences: the
    smallest difference that actually occurs, cross-checked by the fraction of
    all non-zero differences that are near-integer multiples of it. ``D_q``
    converts that step into the diffusion of a Wiener ``theta``: a step
    ``delta`` (rev/s) is ``2 pi delta`` rad/s, its rounding error has variance
    ``(2 pi delta)^2/12``, and if that error is renewed at the rate at which
    the value CHANGES (``f_update``, not the logging rate) it is white with
    one-sided density ``2 (2 pi delta)^2 / (12 f_update)`` and hence
    ``D_q = (2 pi delta)^2 / (24 f_update)``.
    """
    diffs = np.abs(np.diff(track.raw_rps, axis=1))
    diffs = diffs[np.isfinite(diffs)]
    nz = diffs[diffs > 0]
    n_total = int(diffs.size)
    held = float((diffs == 0).mean()) if n_total else float("nan")
    out: dict[str, Any] = {
        "logging_rate_hz": track.fs,
        "held_fraction": held,
        "n_diffs": n_total,
    }
    if nz.size < 16:
        out["step_rps_measured"] = None
        return out
    # The ladder step: the 0.5th percentile of non-zero jumps is robust to a
    # stray sub-step from the regridding while still landing on one rung.
    step = float(np.percentile(nz, 0.5))
    ratios = nz / step
    out["step_rps_measured"] = step
    out["near_integer_fraction"] = float((np.abs(ratios - np.round(ratios)) < 0.05).mean())
    out["distinct_values"] = int(np.unique(np.round(track.raw_rps / step)).size)
    declared = DECLARED_STEP_RPS.get(track.rig)
    if declared is not None:
        out["step_rps_declared"] = float(declared["step_rps"])
        out["step_declared_why"] = declared["why"]
    used = float(out.get("step_rps_declared", step))
    out["step_rps_used"] = used
    # How often the value actually moves - a sample-and-hold renews its
    # rounding error at THIS rate, not at the logging rate.
    n_change = int((np.diff(track.raw_rps, axis=1) != 0).sum())
    span_s = float(track.raw_rps.shape[1] - 1) * float(np.median(track.raw_dt))
    f_update = n_change / max(span_s * track.raw_rps.shape[0], 1e-9)
    out["update_rate_hz"] = f_update
    dw = 2.0 * np.pi * used
    out["D_q_at_logging_rate"] = dw**2 / (24.0 * track.fs)
    out["D_q_at_update_rate"] = dw**2 / (24.0 * max(f_update, 1e-9))
    out["sigma_q_rad_s"] = dw / math.sqrt(12.0)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Structure function and PSD of the telemetry speed error
# ═════════════════════════════════════════════════════════════════════════════


def lag_grid(fs: float, hi_s: float = S_LAG_MAX_S, n: int = S_N_LAGS) -> np.ndarray:
    lo = 1.0 / fs
    hi = max(hi_s, 2.0 * lo)
    return np.unique(np.round(np.geomspace(lo, hi, n) * fs).astype(np.int64))


def structure_function(
    theta_parts: list[np.ndarray], lags: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(S, n)`` for ``S(tau) = Var[theta(t+tau) - theta(t)]`` pooled over parts."""
    num = np.zeros(lags.size)
    cnt = np.zeros(lags.size)
    for th in theta_parts:
        for i, lag in enumerate(lags):
            if th.size <= lag:
                continue
            d = th[lag:] - th[:-lag]
            d = d - d.mean()
            num[i] += float(np.dot(d, d))
            cnt[i] += d.size
    s = np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan)
    return s, cnt


def local_slope(tau: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Central-difference local log-log slope, NaN where undefined."""
    lt, ls = np.log(tau), np.log(np.where(s > 0, s, np.nan))
    out = np.full(s.size, np.nan)
    ok = np.isfinite(ls)
    idx = np.flatnonzero(ok)
    for j in range(idx.size):
        i = idx[j]
        lo = idx[max(j - 1, 0)]
        hi = idx[min(j + 1, idx.size - 1)]
        if hi > lo:
            out[i] = (ls[hi] - ls[lo]) / (lt[hi] - lt[lo])
    return out


def slope_crossing(
    tau: np.ndarray, slope: np.ndarray, target: float = SLOPE_TARGET
) -> float | None:
    """First lag where the local slope drops through ``target`` (log-interpolated)."""
    ok = np.isfinite(slope)
    t, s = tau[ok], slope[ok]
    for i in range(t.size - 1):
        if s[i] > target >= s[i + 1]:
            w = (s[i] - target) / (s[i] - s[i + 1])
            return float(np.exp(np.log(t[i]) + w * (np.log(t[i + 1]) - np.log(t[i]))))
    return None


def welch_psd(parts: list[np.ndarray], fs: float) -> tuple[np.ndarray, np.ndarray, float]:
    """``(f, P, n_seg)`` pooled Welch PSD of the speed error, ``(rad/s)^2/Hz``."""
    nper = int(min(PSD_SEG_S * fs, max(len(p) for p in parts)))
    nper = max(int(2 ** math.floor(math.log2(max(nper, 8)))), 8)
    nov = int(PSD_OVERLAP * nper)
    acc = None
    freqs = None
    n_seg = 0.0
    for p in parts:
        if p.size < nper:
            continue
        f, pxx = welch(p, fs=fs, window="hann", nperseg=nper, noverlap=nov, detrend=False)
        k = 1 + (p.size - nper) // (nper - nov)
        acc = pxx * k if acc is None else acc + pxx * k
        freqs = f
        n_seg += k
    if acc is None or freqs is None:
        return np.zeros(0), np.zeros(0), 0.0
    return freqs, acc / n_seg, n_seg


def gamma_nll(power: np.ndarray, model: np.ndarray, shape: float) -> float:
    """``-2 log L`` of an averaged periodogram under the Whittle/Gamma model.

    An average of ``shape`` independent exponentials with mean ``S`` is
    ``Gamma(shape, S/shape)``, so up to model-independent constants
    ``-2 log L = 2 shape sum_j [log S_j + P_j / S_j]``.
    """
    m = np.maximum(model, 1e-300)
    return float(2.0 * shape * np.sum(np.log(m) + power / m))


def fit_speed_psd(
    f: np.ndarray,
    p: np.ndarray,
    fs: float,
    n_seg: float,
    sos: np.ndarray | None,
    f_lo: float = PSD_F_LO_HZ,
) -> dict[str, Any]:
    """Lorentzian (OU) versus flat (white) speed PSD on ``f_lo .. 0.4 fs``.

    With a high-pass in the chain both models are multiplied by its KNOWN
    power response, so the corner bin is not read as a real roll-off; with
    ``sos=None`` (the detrend-only variant) the response is unity.
    """
    band = (f >= f_lo) & (f <= PSD_F_HI_FRAC * fs) & (p > 0)
    fb, pb = f[band], p[band]
    if fb.size < 8:
        return {"identified": False, "why": "fewer than 8 PSD bins in the fit band"}
    resp = np.ones(fb.size) if sos is None else hp_power_response(sos, fb, fs)
    shape = max(n_seg / WELCH_OVERLAP_INFLATION, 1.0)

    def nll_ou(theta: np.ndarray) -> float:
        sigma, lam = np.exp(theta)
        return gamma_nll(pb, ou_psd(fb, sigma, lam) * resp, shape)

    def nll_white(theta: np.ndarray) -> float:
        return gamma_nll(pb, np.exp(theta[0]) * resp, shape)

    # A flat start plus two decades of lam either side of the band, so the
    # optimiser is not launched inside a plateau.
    best_ou: tuple[float, np.ndarray] | None = None
    for lam0 in (0.3, 1.0, 6.0, 30.0, 200.0):
        s0 = math.sqrt(max(float(np.trapezoid(pb, fb)), 1e-30))
        x0 = np.log([max(s0, 1e-6), lam0])
        res = least_squares(
            lambda th: np.sqrt(
                np.maximum(
                    2.0
                    * shape
                    * (
                        np.log(np.maximum(ou_psd(fb, *np.exp(th)) * resp, 1e-300))
                        + pb / np.maximum(ou_psd(fb, *np.exp(th)) * resp, 1e-300)
                    ),
                    0.0,
                )
            ),
            x0,
            method="lm",
            max_nfev=4000,
        )
        val = nll_ou(res.x)
        if best_ou is None or val < best_ou[0]:
            best_ou = (val, res.x)
    assert best_ou is not None
    nll_o, x_ou = best_ou
    sigma_hat, lam_hat = (float(v) for v in np.exp(x_ou))
    # one-parameter model: the exact ML mean of a Gamma sample is its mean
    s0_hat = float(np.mean(pb / resp))
    nll_w = nll_white(np.array([math.log(max(s0_hat, 1e-300))]))
    n = int(fb.size)
    out = {
        "identified": True,
        "band_hz": [float(fb[0]), float(fb[-1])],
        "n_bins": n,
        "gamma_shape": shape,
        "ou": {
            "sigma_nu_rad_s": sigma_hat,
            "lam_1_s": lam_hat,
            "corner_hz": lam_hat / (2.0 * np.pi),
            "D_theta_rad2_s": sigma_hat**2 / lam_hat,
            "nll": nll_o,
            "aic": nll_o + 2 * 2,
            "bic": nll_o + 2 * math.log(n),
        },
        "white": {
            "s0_rad2_s2_per_hz": s0_hat,
            "D_theta_rad2_s": d_from_white_psd(s0_hat),
            "nll": nll_w,
            "aic": nll_w + 2 * 1,
            "bic": nll_w + 1 * math.log(n),
        },
    }
    out["delta_aic_ou_minus_white"] = out["ou"]["aic"] - out["white"]["aic"]
    out["delta_bic_ou_minus_white"] = out["ou"]["bic"] - out["white"]["bic"]
    out["preferred"] = "ou" if out["delta_aic_ou_minus_white"] < 0 else "white"
    lo, hi = fb[0], fb[-1]
    out["corner_in_band"] = bool(lo <= lam_hat / (2 * np.pi) <= hi)
    return out


def analyse_telemetry_rig(
    rig: str, spec: dict[str, Any], limit: int | None
) -> dict[str, Any] | None:
    tracks = load_native(rig, spec, limit)
    if not tracks:
        return None
    fs = float(np.median([t.fs for t in tracks]))
    rep = dict(
        rig=rig,
        dataset=spec["dataset"],
        rps_key=tracks[0].key,
        auxiliary=bool(spec.get("auxiliary")),
        fs_hz=fs,
        declared_rate_hz=tracks[0].declared_rate_hz,
        n_flights=len(tracks),
        flights=[t.flight for t in tracks],
    )
    rep["quantisation"] = quantisation(tracks[0])
    per_flight_q = [quantisation(t) for t in tracks]
    steps = [q.get("step_rps_measured") for q in per_flight_q if q.get("step_rps_measured")]
    rep["quantisation"]["step_rps_measured_median_over_flights"] = (
        float(np.median(steps)) if steps else None
    )
    rep["quantisation"]["held_fraction_median_over_flights"] = float(
        np.median([q["held_fraction"] for q in per_flight_q])
    )
    rep["quantisation"]["update_rate_hz_median_over_flights"] = float(
        np.median([q.get("update_rate_hz", np.nan) for q in per_flight_q])
    )

    lags = lag_grid(fs)
    rep["lags_s"] = (lags / fs).tolist()
    rep["highpass"] = {}
    total_s = 0.0
    n_rotors = tracks[0].rps.shape[0]
    for f_hp in HP_VARIANTS:
        sos = None if f_hp is None else hp_sos(fs, f_hp)
        key = DETREND_KEY if f_hp is None else f"{f_hp:g}"
        f_lo = PSD_F_LO_WIDE_HZ if f_hp is None else PSD_F_LO_HZ
        min_len = int(10.0 * fs) if f_hp is None else int(4 * fs / f_hp)
        per_rotor: list[dict[str, Any]] = []
        nu_all: list[list[np.ndarray]] = [[] for _ in range(n_rotors)]
        th_all: list[list[np.ndarray]] = [[] for _ in range(n_rotors)]
        for track in tracks:
            if abs(track.fs - fs) / fs > 0.05:
                continue  # a differently-clocked flight is its own population
            segs = stationary_segments(track.rps, track.fs)
            for sl in segs:
                for r in range(n_rotors):
                    y, frac_bad = _interp_nans(track.rps[r, sl])
                    if frac_bad > 0.05 or y.size < min_len:
                        continue
                    if sos is None:
                        # linear detrend only: the least aggressive trend
                        # removal that still leaves theta = int nu bounded
                        t = np.arange(y.size) / fs
                        a, b = np.polyfit(t, y, 1)
                        resid = y - (a * t + b)
                    else:
                        resid = sosfiltfilt(sos, y - y.mean())
                    nu = 2.0 * np.pi * resid
                    nu_all[r].append(nu)
                    th_all[r].append(np.cumsum(nu) / fs)
                if f_hp == HP_VARIANTS[0]:
                    total_s += (sl.stop - sl.start) / track.fs
        if sos is not None:
            f_grid = np.geomspace(max(1e-3, 0.02 * float(f_hp)), 0.5 * fs, 600)
            resp_grid = hp_power_response(sos, f_grid, fs)
        for r in range(n_rotors):
            if not nu_all[r]:
                per_rotor.append({"rotor": r, "n_segments": 0})
                continue
            s, cnt = structure_function(th_all[r], lags)
            tau = lags / fs
            sl_local = local_slope(tau, s)
            cross = slope_crossing(tau, sl_local)
            f_psd, p_psd, n_seg = welch_psd(nu_all[r], fs)
            fit = fit_speed_psd(f_psd, p_psd, fs, n_seg, sos, f_lo)
            row: dict[str, Any] = {
                "rotor": r,
                "n_segments": len(nu_all[r]),
                "n_samples": int(sum(x.size for x in nu_all[r])),
                "nu_rms_rad_s": float(math.sqrt(np.mean(np.concatenate(nu_all[r]) ** 2))),
                "S_rad2": [None if not np.isfinite(v) else float(v) for v in s],
                "S_n_pairs": cnt.tolist(),
                "local_slope": [None if not np.isfinite(v) else float(v) for v in sl_local],
                "tau_slope1p5_s": cross,
                "lam_from_slope1p5_1_s": (U_SLOPE_1P5 / cross) if cross else None,
                "psd_fit": fit,
            }
            if fit.get("identified"):
                sigma, lam = fit["ou"]["sigma_nu_rad_s"], fit["ou"]["lam_1_s"]
                pred = (
                    ou_structure(tau, sigma, lam)
                    if sos is None
                    else filtered_theta_structure(tau, sigma, lam, resp_grid, f_grid)
                )
                row["S_model_ou_rad2"] = np.asarray(pred).tolist()
            per_rotor.append(row)
        pooled = pool_rotors(per_rotor)
        rep["highpass"][key] = {
            "f_hp_hz": f_hp,
            "hp_order": None if f_hp is None else HP_ORDER,
            "zero_phase": f_hp is not None,
            "trend_removal": "linear detrend per segment" if f_hp is None else "butterworth",
            "psd_band_lo_hz": f_lo,
            "per_rotor": per_rotor,
            "pooled": pooled,
        }
        if f_hp in (HP_VARIANTS[0], None):
            # PSD of the pooled rotors, for the figure
            f_psd, p_psd, n_seg = welch_psd([x for r in range(n_rotors) for x in nu_all[r]], fs)
            fit = fit_speed_psd(f_psd, p_psd, fs, n_seg, sos, f_lo)
            blob = {
                "f_hz": f_psd.tolist(),
                "P_rad2_s2_per_hz": p_psd.tolist(),
                "n_seg": n_seg,
                "fit": fit,
            }
            rep["psd" if f_hp is not None else "psd_detrend"] = blob
    rep["analysed_seconds"] = total_s
    return rep


def pool_rotors(per_rotor: list[dict[str, Any]]) -> dict[str, Any]:
    """Median over rotors of the identified per-rotor quantities."""

    def med(path: list[str]) -> float | None:
        vals = []
        for row in per_rotor:
            cur: Any = row
            for p in path:
                if not isinstance(cur, dict) or p not in cur:
                    cur = None
                    break
                cur = cur[p]
            if isinstance(cur, (int, float)) and np.isfinite(cur):
                vals.append(float(cur))
        return float(np.median(vals)) if vals else None

    prefs = [
        r["psd_fit"].get("preferred") for r in per_rotor if r.get("psd_fit", {}).get("identified")
    ]
    return {
        "n_rotors_identified": int(len(prefs)),
        "sigma_nu_rad_s": med(["psd_fit", "ou", "sigma_nu_rad_s"]),
        "lam_1_s": med(["psd_fit", "ou", "lam_1_s"]),
        "corner_hz": med(["psd_fit", "ou", "corner_hz"]),
        "D_theta_ou_rad2_s": med(["psd_fit", "ou", "D_theta_rad2_s"]),
        "D_theta_white_rad2_s": med(["psd_fit", "white", "D_theta_rad2_s"]),
        "s0_rad2_s2_per_hz": med(["psd_fit", "white", "s0_rad2_s2_per_hz"]),
        "delta_aic_ou_minus_white": med(["psd_fit", "delta_aic_ou_minus_white"]),
        "delta_bic_ou_minus_white": med(["psd_fit", "delta_bic_ou_minus_white"]),
        "tau_slope1p5_s": med(["tau_slope1p5_s"]),
        "lam_from_slope1p5_1_s": med(["lam_from_slope1p5_1_s"]),
        "nu_rms_rad_s": med(["nu_rms_rad_s"]),
        "preferred_by_aic": (max(set(prefs), key=prefs.count) if prefs else None),
        "n_rotors_preferring_ou": int(sum(p == "ou" for p in prefs)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Acoustics
# ═════════════════════════════════════════════════════════════════════════════


def _wrap(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


def demod_orders(
    audio: np.ndarray,
    fs: float,
    centres: np.ndarray,
    window: int,
    hop: int,
    k_max: int,
) -> dict[str, Any]:
    """Baseband complex amplitude of orders ``1..k_max`` and the off-order floor.

    ``centres`` is ``(N,)`` the per-frame SHAFT rate in Hz (constant on the
    bench, telemetry-driven in flight). ``z[k]`` is ``(M, N)`` with the NOMINAL
    phase ``k * 2 pi int rate dt`` removed, so what is left is
    ``psi_k = k theta + eps_k + alpha_mk`` plus measurement noise. The
    half-order offsets ``(k + 1/2)`` are demodulated with the same window to
    measure the in-band floor, which is the only SNR that matters for a phase
    read at this resolution.
    """
    from experiments.stochastic_fit.phase_stats import stft_at

    n_frames = int(1 + (audio.shape[1] - window) // hop)
    frames = np.arange(n_frames)
    starts = frames * hop
    ts = starts / fs
    # The centre of each frame is the WINDOW MEAN rate, and the nominal phase
    # is the exact cumulative telemetry phase at the frame START - the same
    # convention as ``revised_phase.estimate_shaft_dynamics``.
    phi1 = 2.0 * np.pi * np.cumsum(centres) / fs
    phi_at = phi1[np.minimum(starts, centres.size - 1)]
    if centres.size > 1:
        csum = np.concatenate([[0.0], np.cumsum(centres)])
        hi = np.minimum(starts + window, centres.size)
        rate_frame = (csum[hi] - csum[starts]) / np.maximum(hi - starts, 1)
    else:
        rate_frame = np.full(n_frames, float(centres[0]))
    z: dict[float, np.ndarray] = {}
    for k in list(range(1, k_max + 1)) + [k + 0.5 for k in range(1, k_max + 1)]:
        raw = stft_at(
            audio, frames, np.asarray(k * rate_frame, dtype=np.float64), window, hop, int(fs)
        )
        z[k] = raw * np.exp(-1j * k * phi_at)[None, :]
    return {"z": z, "ts": ts, "n_frames": n_frames, "frame_rate_hz": fs / hop}


def band_margin_db(z: dict[float, np.ndarray], k: int) -> np.ndarray:
    """``(M,)`` in-band order-to-floor ratio, dB, from the half-order offsets."""
    p = np.mean(np.abs(z[k]) ** 2, axis=1)
    fl = 0.5 * (
        np.mean(np.abs(z[k + 0.5]) ** 2, axis=1)
        + np.mean(np.abs(z[max(k - 1, 1) + 0.5]) ** 2, axis=1)
    )
    return 10.0 * np.log10(np.maximum(p, 1e-300) / np.maximum(fl, 1e-300))


#: A CONSTANT rate error is invisible to the coherence estimator, and this is
#: worth stating because the obvious "refine the rate first" step is not only
#: unnecessary but impossible. A residual rate error ``d`` multiplies the
#: baseband by ``exp(-i 2 pi k d t)``, so the lagged product picks up
#: ``exp(-i 2 pi k d tau)`` - a factor that is CONSTANT over ``t`` and
#: therefore leaves ``|sum_t z(t+tau) z*(t)|`` exactly unchanged. Trying to
#: maximise coherence over ``d`` returns whichever grid point float noise
#: favours (a planted control returned -0.28 rev/s on an exactly known rate,
#: which then wrecked every ``V_k``). Only a DRIFTING rate error matters, and
#: a drifting rate error IS the shaft process being measured.
RATE_ERROR_INVARIANT = (
    "|gamma(tau)| is invariant to a constant rate error: the lagged product "
    "gains only the t-independent factor exp(-i 2 pi k d tau)"
)


def _raw_g2(zd: np.ndarray, lag: int) -> np.ndarray:
    """``|gamma|^2`` per microphone, each summed coherently on its own."""
    a, b = zd[:, lag:], zd[:, :-lag]
    num = np.abs((a * np.conj(b)).sum(axis=1))
    den = np.sqrt((np.abs(a) ** 2).sum(axis=1) * (np.abs(b) ** 2).sum(axis=1))
    return (num / np.maximum(den, 1e-300)) ** 2


def coherence_g2(
    zd: np.ndarray, surrogate: np.ndarray, lag: int, oversamp: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(debiased g^2, floor)`` per microphone at one frame lag.

    The model itself says ``R_k(tau) = exp(-V_k(tau)/2)`` (§3.3), so
    ``V = -log g^2``. Each microphone is summed COHERENTLY on its own (every
    mic has its own fixed ``alpha_mk``, so a cross-mic complex sum would
    cancel), and the logarithm is taken only after averaging over microphones
    and supports - the linear ``g^2`` domain is the only one in which the
    cells are additive. ``E|gamma|^2 = g^2 (1 - F) + F`` with floor ``F``.

    The floor has TWO parts and a planted control needs both.

    * The noise part is MEASURED, not assumed: ``surrogate`` is ``|gamma|^2``
      of a half-order flank band - same window, frame grid, overlap and noise,
      no line - so its coherence IS the additive-noise floor. (The project's
      own ``phase_stats.lag_coherence`` uses a white-noise surrogate for
      exactly this reason.)
    * The increment part is the finite number of independent phase
      increments: ``psi(t+tau) - psi(t)`` shares almost all of its path with
      the increment a moment later, so the record holds only
      ``(n_pairs)/max(oversamp, lag)`` independent ones, and ``|sum|`` of that
      many unit phasors has the usual ``1/n`` floor.

    A theoretical floor alone was wrong by ~1.4x at short lags; the measured
    noise floor alone was wrong by ~2x at long ones.
    """
    g2 = _raw_g2(zd, lag)
    n_increments = max((zd.shape[1] - lag) / max(oversamp, float(lag)), 2.0)
    floor = np.clip(np.asarray(surrogate, dtype=np.float64) + 1.0 / n_increments, 0.0, 0.95)
    return (g2 - floor) / (1.0 - floor), floor


def g2_stats(g2d: np.ndarray, floor: np.ndarray) -> tuple[float, float, int, float]:
    """``(mean, error scale, n cells, mean floor)`` of debiased ``g^2`` cells.

    The error scale is the larger of the empirical spread over microphones and
    the FLOOR ITSELF. Not the floor over ``sqrt(n)``: the floor is a
    subtraction whose own scale is only known to a factor, so requiring
    ``g^2 > COH_FLOOR_FACTOR`` error scales is the statement "the correction
    is at most a quarter of the raw value", which is the regime in which a
    mis-specified floor moves ``V`` by under ten percent. It also gives the
    fit an inverse-variance weight that falls off near the ceiling, exactly
    where it should.
    """
    a = np.asarray(g2d, dtype=np.float64)
    f = np.asarray(floor, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(f)
    a, f = a[ok], f[ok]
    if a.size == 0:
        return float("nan"), float("nan"), 0, float("nan")
    m = float(a.mean())
    fbar = float(f.mean())
    se_emp = float(a.std(ddof=1) / math.sqrt(a.size)) if a.size > 1 else abs(m)
    return m, max(se_emp, fbar), int(a.size), fbar


def g2_to_V(mean: float, se: float, n: int) -> tuple[float | None, float | None]:
    """``(V, sd of V)``, or ``(None, None)`` where ``V`` is not resolved.

    Two gates, both needed. ``g^2`` must stand clear of ZERO by
    ``COH_FLOOR_FACTOR`` standard errors, or the lag is past the estimator's
    ceiling. And ``g^2`` must stand clear of ONE, or ``V`` is smaller than
    the estimator's own resolution: such a cell reads ``V ~ 0``, and an
    inverse-variance fit would then be dragged to ``sigma = lam = D = 0``.
    """
    if n < 2 or not np.isfinite(mean) or not np.isfinite(se) or se <= 0.0:
        return None, None
    if mean <= COH_FLOOR_FACTOR * se:
        return None, None
    if mean >= 1.0 - COH_FLOOR_FACTOR * se:
        return None, None
    return float(-math.log(mean)), float(se / mean)


def cross_g2(
    zk: np.ndarray, zl: np.ndarray, sur: np.ndarray, lag: int, oversamp: float
) -> tuple[np.ndarray, np.ndarray]:
    """Debiased ``g^2`` of the ORDER-PAIR product, per microphone.

    ``E exp[i(dpsi_k - dpsi_l)] = exp(-Var[dpsi_k - dpsi_l]/2)`` for Gaussian
    increments, and the product ``(z_k(t+tau) z_k*(t)) (z_l(t+tau) z_l*(t))*``
    has exactly that expectation - wrap-immune, and the route to
    ``Cov = (V_k + V_l - Var_diff)/2`` of §5.1.
    """
    pk = zk[:, lag:] * np.conj(zk[:, :-lag])
    pl = zl[:, lag:] * np.conj(zl[:, :-lag])
    q = pk * np.conj(pl)
    num = np.abs(q.sum(axis=1))
    den = np.sqrt((np.abs(pk) ** 2).sum(axis=1) * (np.abs(pl) ** 2).sum(axis=1))
    g2 = (num / np.maximum(den, 1e-300)) ** 2
    n_increments = max((zk.shape[1] - lag) / max(oversamp, float(lag)), 2.0)
    floor = np.clip(np.asarray(sur, dtype=np.float64) + 1.0 / n_increments, 0.0, 0.95)
    return (g2 - floor) / (1.0 - floor), floor


def unwrapped_V(zd: np.ndarray, lag: int, oversamp: float) -> tuple[np.ndarray, float]:
    """``(Var of the unwrapped phase increment per mic, independent-sample count)``.

    This is the PRIMARY acoustic estimator. It has no ratio bias and no
    sampling floor to subtract, and a planted control recovers ``D_k`` and its
    ``k`` exponent from it accurately - but only while the phase excursion is
    small: beyond ``V ~ ACOUSTIC_V_LINEAR_MAX`` the phase of a windowed sum of
    phasors stops tracking the window-weighted mean phase and the estimate
    saturates (on the planted control it fell to 0.4x the truth at ``V ~ 5``).
    The fit therefore keeps only cells whose FITTED value is inside that
    regime, and :func:`coherence_g2` is the wrap-immune fallback for a rig
    that has no such cells.

    The mean increment is removed with the variance, so a constant rate error
    cannot leak in; the cost is a ``1 - tau/T`` shrinkage that is under 10 %
    over the lag range used.
    """
    ph = np.unwrap(_wrap(np.angle(zd)), axis=1)
    d = ph[:, lag:] - ph[:, :-lag]
    n_ind = max((zd.shape[1] - lag) / max(oversamp, float(lag)), 2.0)
    return np.var(d, axis=1), n_ind


ACOUSTIC_MODELS = ("ou_free_D", "ou_D_const", "ou_D_lin", "ou_D_quad", "white_shaft")


def window_autocorr(window: int, fs: float, n_points: int = 193) -> tuple[np.ndarray, np.ndarray]:
    """``(s, w)``: the Hann analysis window's autocorrelation as a lag kernel.

    The demodulated coefficient is a WINDOWED integral of the harmonic, so the
    coherence it reports is not ``exp(-V(tau)/2)`` but

        ``gamma(tau) = int W2(s) exp(-V(tau+s)/2) ds / int W2(s) exp(-V(s)/2) ds``,

    with ``W2(s) = int w(u) w(u-s) du`` (§4.4: "the STFT window is part of the
    observation model"). This is EXACT for Gaussian phase at any magnitude,
    unlike a small-excursion expansion, and it is what makes lags shorter than
    the window interpretable at all.
    """
    w = np.hanning(window + 1)[:window]
    ac = np.correlate(w, w, mode="full")
    s = (np.arange(ac.size) - (window - 1)) / fs
    step = max(ac.size // n_points, 1)
    s, ac = s[::step], ac[::step]
    return s, ac / ac.sum()


def fit_Vk(
    ks: np.ndarray,
    taus: np.ndarray,
    V: np.ndarray,
    weights: np.ndarray,
    model: str,
    s_grid: np.ndarray,
    s_w: np.ndarray,
    c_meas: np.ndarray,
    estimator: str = "unwrapped",
    v_cut: float | None = None,
) -> dict[str, Any]:
    """Fit ``V_k(tau) = k^2 V_theta(tau) + 2 D_k tau`` and its rivals.

    What is OBSERVED is a WINDOWED functional of the phase, so the forward
    model carries the window (:func:`window_autocorr`) explicitly:

    * ``estimator="unwrapped"`` - the variance of the difference of two
      window-weighted mean phases,
      ``int W2(s) [V(|tau+s|) + V(|tau-s|) - 2 V(|s|)]/2 ds``;
    * ``estimator="coherence"`` - ``-2 log[C(tau)/C(0)]`` with
      ``C(tau) = int W2(s) exp(-V(tau+s)/2) ds``.

    The lag-independent offset a finite-SNR demodulation carries is NOT free.
    Past one window additive noise contributes no lagged correlation, so its
    whole cost is a constant, and that constant is PREDICTED by the measured
    in-band SNR: ``c_k = 1/SNR_k`` for the unwrapped estimator (two
    independent endpoints, each with phase variance ``1/(2 SNR)``) and
    ``2 log(1 + 1/SNR_k)`` for the coherence one. Only a single global gain
    ``g_c`` on that vector is fitted. Leaving one free offset per order looks
    safer and is not: over a single decade of lag a constant and ``2 D_k tau``
    are near-degenerate, and on a planted control the free offsets ate the
    ``D_k prop k`` signal and returned ``D_k`` flat.

    Parameters are fitted in logs, so positivity is structural.

    ``v_cut`` refits on the cells whose FITTED ``V`` is below it. The
    unwrapped estimator is only unbiased in the small-excursion regime, and a
    cut cannot be applied to the measured value (a saturated cell reads
    small), so it is applied to the model after a first pass.
    """
    uk = np.unique(ks)
    nk = uk.size
    shaft_free = model != "white_shaft"

    def unpack(th: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
        if shaft_free:
            sigma, lam = math.exp(th[0]), math.exp(th[1])
            i = 2
        else:
            sigma, lam = math.exp(th[0]), float("inf")
            i = 1
        if model == "ou_free_D":
            dk = np.exp(th[i : i + nk])
            i += nk
        else:
            d0 = math.exp(th[i])
            i += 1
            pw = {"ou_D_const": 0.0, "ou_D_lin": 1.0, "ou_D_quad": 2.0, "white_shaft": 0.0}[model]
            dk = d0 * uk**pw
        return sigma, lam, dk, math.exp(th[i]) * c_meas

    def v_theta(t: np.ndarray, sigma: float, lam: float) -> np.ndarray:
        if shaft_free:
            return ou_structure(t, sigma, lam)
        # white speed error: theta is Wiener, D_theta = sigma^2 - a
        # reparameterisation, so this model is the lam -> inf limit of the OU
        return 2.0 * sigma**2 * np.abs(t)

    def make_predict(k_sel: np.ndarray, t_sel: np.ndarray):
        idx = np.searchsorted(uk, k_sel)
        a = np.abs(t_sel[:, None] + s_grid[None, :])
        b = np.abs(t_sel[:, None] - s_grid[None, :])
        c0 = np.abs(s_grid)

        def predict(th: np.ndarray) -> np.ndarray:
            sigma, lam, dk, ck = unpack(th)
            out = np.empty(k_sel.size)
            for j, k in enumerate(uk):
                sel = idx == j
                if not sel.any():
                    continue

                def vk(t: np.ndarray, k=k, j=j) -> np.ndarray:
                    return k**2 * v_theta(t, sigma, lam) + 2.0 * dk[j] * np.abs(t)

                if estimator == "unwrapped":
                    out[sel] = (0.5 * (vk(a[sel]) + vk(b[sel]) - 2.0 * vk(c0)[None, :])) @ s_w + ck[
                        j
                    ]
                else:
                    cc = np.exp(-0.5 * np.clip(vk(a[sel]), 0.0, 700.0)) @ s_w
                    c00 = float(np.exp(-0.5 * np.clip(vk(c0), 0.0, 700.0)) @ s_w)
                    out[sel] = -2.0 * np.log(np.maximum(cc / max(c00, 1e-300), 1e-300)) + ck[j]
            return out

        return predict

    n_p = (2 if shaft_free else 1) + (nk if model == "ou_free_D" else 1) + 1

    def solve(k_sel: np.ndarray, t_sel: np.ndarray, v_sel: np.ndarray, w_sel: np.ndarray):
        predict = make_predict(k_sel, t_sel)

        def resid(th: np.ndarray) -> np.ndarray:
            return np.sqrt(w_sel) * (predict(th) - v_sel)

        best: tuple[float, np.ndarray] | None = None
        for lam0 in (1.0, 6.0, 30.0):
            for sig0 in (0.1, 1.0):
                th0 = np.zeros(n_p)
                th0[0] = math.log(sig0)
                if shaft_free:
                    th0[1] = math.log(lam0)
                th0[(2 if shaft_free else 1) :] = math.log(1e-2)
                th0[-1] = 0.0
                try:
                    res = least_squares(resid, th0, method="lm", max_nfev=8000)
                except Exception:
                    continue
                cost = float(np.sum(resid(res.x) ** 2))
                if best is None or cost < best[0]:
                    best = (cost, res.x)
        return best, predict

    best, predict = solve(ks, taus, V, weights)
    if best is None:
        return {"model": model, "identified": False}
    used = np.ones(ks.size, dtype=bool)
    if v_cut is not None:
        keep = predict(best[1]) <= v_cut
        if keep.sum() >= max(12, 2 * n_p) and np.unique(ks[keep]).size >= 2:
            uk2 = np.unique(ks[keep])
            if uk2.size == nk:
                b2, p2 = solve(ks[keep], taus[keep], V[keep], weights[keep])
                if b2 is not None:
                    best, predict, used = b2, p2, keep
    rss, th = best
    n = int(used.sum())
    sigma, lam, dk, ck = unpack(th)
    tau_u = np.unique(taus)
    out: dict[str, Any] = {
        "model": model,
        "estimator": estimator,
        "identified": True,
        "n_cells": n,
        "n_cells_offered": int(ks.size),
        "n_params": n_p,
        "rss_weighted": rss,
        "aic": n * math.log(max(rss, 1e-300) / n) + 2 * n_p,
        "bic": n * math.log(max(rss, 1e-300) / n) + n_p * math.log(n),
        "orders": uk.tolist(),
        "D_k_rad2_s": dk.tolist(),
        "c_k_rad2": ck.tolist(),
        "V_pred": predict(th).tolist(),
        "cells_used": used.tolist(),
        "V_theta_true_rad2": v_theta(tau_u, sigma, lam).tolist(),
        "taus_unique_s": tau_u.tolist(),
    }
    if shaft_free:
        d_theta = sigma**2 / lam
        out.update(
            sigma_nu_rad_s=float(sigma),
            lam_1_s=float(lam),
            D_theta_rad2_s=float(d_theta),
            tau_slope1p5_s=float(U_SLOPE_1P5 / lam),
        )
    else:
        d_theta = sigma**2
        out.update(sigma_nu_rad_s=None, lam_1_s=None, D_theta_rad2_s=float(d_theta))
    # the one robust scale: the lag at which the SHAFT term reaches 1 rad^2
    grid = np.geomspace(1e-4, 1e3, 4000)
    vth = v_theta(grid, sigma, lam) if shaft_free else 2.0 * sigma**2 * grid
    hit = np.flatnonzero(vth >= 1.0)
    out["tau_Vtheta_1rad2_s"] = float(grid[hit[0]]) if hit.size else None
    return out


def analyse_acoustic_support(
    name: str,
    audio: np.ndarray,
    fs: float,
    centres: np.ndarray,
    rate_hz: float,
    k_max: int = K_MAX_ACOUSTIC,
    isolation: list[float] | None = None,
    window_s: float | None = None,
    oversamp: int = DEMOD_OVERSAMP,
) -> dict[str, Any]:
    """One stationary support: demodulate, gate, and fit ``(*)`` jointly over k.

    ``window_s`` overrides the revolution-tied analysis window. A single-rotor
    bench support only has to resolve its OWN comb, so ``DEMOD_REVS``
    revolutions is enough; a four-rotor flight support has to resolve one
    rotor's order from every other rotor's nearby order, which needs a
    frequency resolution set by the rotor-speed spread, not by the shaft
    period (:data:`ACOUSTIC_FLIGHT_WINDOW_S`). ``oversamp`` sets the frame
    hop, hence the SHORTEST lag the support can report: a rig whose phase
    error is large needs a fine hop or its high orders never clear the
    coherence floor at any lag.
    """
    want = DEMOD_REVS * fs / rate_hz if window_s is None else window_s * fs
    window = int(2 ** round(math.log2(max(want, 256))))
    hop = max(window // int(oversamp), 1)
    if audio.shape[1] < window + 8 * hop:
        return {"support": name, "identified": False, "why": "support shorter than one window"}
    res_bw_hz = 2.0 * fs / window  # Hann main-lobe half width
    if isolation is not None and max(isolation) < 2.0 * res_bw_hz:
        return {
            "support": name,
            "identified": False,
            "why": "no order is resolvable from its neighbours at this resolution",
            "resolution_hz": res_bw_hz,
            "isolation_hz": list(isolation),
        }
    dm = demod_orders(audio, fs, centres, window, hop, k_max)
    z = dm["z"]
    margins = {k: band_margin_db(z, k) for k in range(1, k_max + 1)}
    keep = {k: np.flatnonzero(margins[k] >= MARGIN_MIN_DB) for k in range(1, k_max + 1)}
    if isolation is not None:
        for k in range(1, k_max + 1):
            if isolation[k - 1] < 2.0 * res_bw_hz:
                keep[k] = np.zeros(0, dtype=int)
    orders = [k for k in range(1, k_max + 1) if keep[k].size >= 2]
    if not orders:
        return {
            "support": name,
            "identified": False,
            "why": "no order clears the in-band margin bar",
            "margin_db_median": {k: float(np.median(v)) for k, v in margins.items()},
        }
    # No rate refinement: see RATE_ERROR_INVARIANT. The full microphone axis
    # is kept, because two orders generally pass the margin bar on DIFFERENT
    # microphone subsets and a cross-order statistic needs their intersection.
    zd = {k: z[k] for k in orders}
    # The floor surrogate for order k: the QUIETER of its two half-order
    # flanks, so that a neighbour line leaking into one flank cannot inflate
    # the floor and over-correct the estimate.
    sur: dict[int, np.ndarray] = {}
    for k in orders:
        cands = [z[k + 0.5]] + ([z[k - 0.5]] if k >= 2 else [])
        sur[k] = min(cands, key=lambda c: float(np.mean(np.abs(c) ** 2)))

    oversamp = float(window / hop)
    lag_hi = min(ACOUSTIC_LAG_HI_S, (dm["n_frames"] - 2) / dm["frame_rate_hz"] / 3.0)
    # The grid starts at ONE WINDOW, not at the hop. Below the window the
    # analysis frames overlap, so even pure noise is strongly coherent
    # (the Hann window autocorrelation at a lag of one hop is 0.96) and the
    # additive-noise contribution to the lagged product is not a constant
    # offset but a lag-dependent one. At tau >= T_w the window
    # autocorrelation is exactly zero, additive noise contributes NO lagged
    # correlation, and its whole effect is the constant c_k the fit carries.
    lag_lo = max(ACOUSTIC_LAG_LO_S, float(window) / fs)
    lags = np.unique(
        np.round(
            np.geomspace(lag_lo, max(lag_hi, 1.05 * lag_lo), ACOUSTIC_N_LAGS) * dm["frame_rate_hz"]
        ).astype(int)
    )
    lags = lags[(lags >= int(round(oversamp))) & (lags < dm["n_frames"] - 2)]
    if lags.size < 3:
        return {
            "support": name,
            "identified": False,
            "why": "support too short for three lags at or above one window",
            "window_s": window / fs,
        }
    taus = lags / dm["frame_rate_hz"]

    G2: dict[int, list[float]] = {}
    SE: dict[int, list[float]] = {}
    FL: dict[int, list[float]] = {}
    Nk: dict[int, list[int]] = {}
    Vk: dict[int, list[float | None]] = {}
    Vs: dict[int, list[float | None]] = {}
    Vu: dict[int, list[float | None]] = {}
    Us: dict[int, list[float | None]] = {}
    for k in orders:
        gs, ses, fls, ns, vs, vse, vu, use = [], [], [], [], [], [], [], []
        mics = keep[k]
        for lag in lags:
            f_raw = _raw_g2(sur[k][mics], int(lag))
            arr, floor = coherence_g2(zd[k][mics], f_raw, int(lag), oversamp)
            m, se, n, fbar = g2_stats(arr, floor)
            v, v_se = g2_to_V(m, se, n)
            gs.append(m)
            ses.append(se)
            fls.append(fbar)
            ns.append(n)
            vs.append(v)
            vse.append(v_se)
            vu_mic, n_ind = unwrapped_V(zd[k][mics], int(lag), oversamp)
            v_un = float(np.mean(vu_mic))
            vu.append(v_un)
            # Var of a variance from n_ind independent samples is 2 V^2 / n_ind.
            # The error is sampling OR systematic, whichever is larger. The
            # windowed forward model is good to about ten percent on a planted
            # control, so a well-sampled cell is systematics-limited; without
            # this floor the weights fall off as 1/lag and the fit is dominated
            # by the shortest lags, where 2 D_k tau is smallest and D_k is
            # therefore least identifiable (a planted D_k prop k came back flat).
            rel = max(math.sqrt(2.0 / (n_ind * max(mics.size, 1))), ACOUSTIC_V_REL_SYSTEMATIC)
            use.append(float(v_un * rel))
        G2[k], SE[k], FL[k], Nk[k] = gs, ses, fls, ns
        Vk[k], Vs[k], Vu[k], Us[k] = vs, vse, vu, use

    # the §5.1 cross-order estimate of the SAME shaft structure function
    cross: list[dict[str, Any]] = []
    for i, k in enumerate(orders):
        for m2 in orders[i + 1 :]:
            common = np.intersect1d(keep[k], keep[m2])
            if common.size < 2:
                continue
            vals = []
            for j, lag in enumerate(lags):
                pk = sur[k][common][:, int(lag) :] * np.conj(sur[k][common][:, : -int(lag)])
                pl = sur[m2][common][:, int(lag) :] * np.conj(sur[m2][common][:, : -int(lag)])
                q = pk * np.conj(pl)
                num = np.abs(q.sum(axis=1))
                den = np.sqrt((np.abs(pk) ** 2).sum(axis=1) * (np.abs(pl) ** 2).sum(axis=1))
                f_raw = (num / np.maximum(den, 1e-300)) ** 2
                arr, floor = cross_g2(zd[k][common], zd[m2][common], f_raw, int(lag), oversamp)
                m, se, n, _ = g2_stats(arr, floor)
                vd, _unused = g2_to_V(m, se, n)
                vk, vl = Vk[k][j], Vk[m2][j]
                if vd is None or vk is None or vl is None:
                    vals.append(None)
                    continue
                vals.append(0.5 * (vk + vl - vd) / (k * m2))
            cross.append({"k": k, "l": m2, "n_mics": int(common.size), "V_theta_rad2": vals})

    marg = {int(k): float(np.median(margins[k])) for k in orders}
    fits = fit_cells(orders, taus, Vu, Us, window, fs, marg, estimator="unwrapped")
    fits_coh = fit_cells(orders, taus, Vk, Vs, window, fs, marg, estimator="coherence")
    primary = "unwrapped" if fits.get("best_by_aic") else "coherence"
    return {
        "support": name,
        "identified": bool(fits),
        "fs_hz": fs,
        "rate_rps": rate_hz,
        "rate_error_invariant": RATE_ERROR_INVARIANT,
        "window_samples": window,
        "window_s": window / fs,
        "hop_samples": hop,
        "frame_rate_hz": dm["frame_rate_hz"],
        "resolution_hz": res_bw_hz,
        "n_frames": dm["n_frames"],
        "oversampling": oversamp,
        "orders_used": orders,
        "n_orders_used": len(orders),
        "margin_db_median": {str(k): float(np.median(v)) for k, v in margins.items()},
        "n_mics_kept": {str(k): int(keep[k].size) for k in range(1, k_max + 1)},
        "lags_s": taus.tolist(),
        "g2_mean": {str(k): G2[k] for k in orders},
        "g2_se": {str(k): SE[k] for k in orders},
        "g2_floor": {str(k): FL[k] for k in orders},
        "n_mics_finite": {str(k): Nk[k] for k in orders},
        "V_k_rad2": {str(k): Vk[k] for k in orders},
        "V_k_se_rad2": {str(k): Vs[k] for k in orders},
        "V_k_unwrapped_rad2": {str(k): Vu[k] for k in orders},
        "V_k_unwrapped_se_rad2": {str(k): Us[k] for k in orders},
        "cross_order_V_theta": cross,
        "primary_estimator": primary,
        "fits": fits if primary == "unwrapped" else fits_coh,
        "fits_unwrapped": fits,
        "fits_coherence": fits_coh,
    }


def fit_cells(
    orders: list[int],
    taus: np.ndarray,
    Vk: dict[int, list[float | None]],
    Vse: dict[int, list[float | None]],
    window: int,
    fs: float,
    margins: dict[int, float],
    *,
    estimator: str = "unwrapped",
) -> dict[str, Any]:
    """Every RESOLVED ``(k, tau)`` cell through the five rival models of ``(*)``.

    Weights are inverse variance from the estimator's own propagated error,
    never a bare ``1/V^2``: a cell the estimator could not resolve reads
    ``V ~ 0`` and ``1/V^2`` would give it infinite say (it dragged a planted
    control to ``sigma = lam = D = 0``).
    """
    ks_l, tau_l, v_l, w_l = [], [], [], []
    for k in orders:
        for j, tau in enumerate(taus):
            v, se = Vk[k][j], Vse[k][j]
            if v is None or se is None or not np.isfinite(v) or not np.isfinite(se) or se <= 0:
                continue
            if v <= 0.0:
                continue
            ks_l.append(float(k))
            tau_l.append(float(tau))
            v_l.append(float(v))
            w_l.append(1.0 / se**2)
    if len(set(ks_l)) < 2 or len(v_l) < 12:
        return {}
    ks_a, tau_a = np.asarray(ks_l), np.asarray(tau_l)
    v_a, w_a = np.asarray(v_l), np.asarray(w_l)
    s_grid, s_w = window_autocorr(int(window), float(fs))
    cut = ACOUSTIC_V_LINEAR_MAX if estimator == "unwrapped" else None
    # The predicted additive-noise offset per order, from the MEASURED in-band
    # margin: margin = (S+N)/N, so SNR = 10^(m/10) - 1.
    uk_a = np.unique(ks_a)
    snr = np.array([max(10.0 ** (margins.get(int(k), 6.0) / 10.0) - 1.0, 1e-3) for k in uk_a])
    c_meas = 1.0 / snr if estimator == "unwrapped" else 2.0 * np.log1p(1.0 / snr)
    fits: dict[str, Any] = {
        m: fit_Vk(ks_a, tau_a, v_a, w_a, m, s_grid, s_w, c_meas, estimator=estimator, v_cut=cut)
        for m in ACOUSTIC_MODELS
    }
    fits["best_by_aic"] = min(
        (m for m in ACOUSTIC_MODELS if fits[m].get("identified")),
        key=lambda m: fits[m]["aic"],
        default=None,
    )
    fits["window_samples"] = int(window)
    fits["estimator"] = estimator
    return fits


def pool_acoustic(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One fit over SEVERAL supports that share the lag grid.

    A 16 s flight window only leaves one or two orders standing per rotor, so
    a per-window joint fit has no cross-order leverage at all. Pooling every
    window of the SAME rotor restores it without mixing rotors, which would
    defeat the point. The pooling happens in the LINEAR ``g^2`` domain - the
    only domain in which the microphone-and-window cells are unbiased and
    additive - and the logarithm is taken once, at the end, on the pooled
    mean. That is what extends the usable lag range: the sampling floor of
    the pooled mean is smaller than any single cell's by ``sqrt(n_cells)``.
    """
    usable = [r for r in rows if r.get("lags_s") and r.get("g2_mean")]
    if not usable:
        return {"support": name, "identified": False, "why": "no window produced cells"}
    taus = np.asarray(usable[0]["lags_s"], dtype=float)
    usable = [r for r in usable if len(r["lags_s"]) == taus.size]
    Vk: dict[int, list[float | None]] = {}
    Vs: dict[int, list[float | None]] = {}
    Nk: dict[int, list[int]] = {}
    FL: dict[int, list[float]] = {}
    G2: dict[int, list[float]] = {}
    SE: dict[int, list[float]] = {}
    for k in range(1, K_MAX_ACOUSTIC + 1):
        vs: list[float | None] = []
        vse: list[float | None] = []
        ns: list[int] = []
        fls: list[float] = []
        gs: list[float] = []
        ses: list[float] = []
        for j in range(taus.size):
            num = fnum = var = 0.0
            den = 0
            for r in usable:
                g = (r["g2_mean"].get(str(k)) or [None] * taus.size)[j]
                se = (r["g2_se"].get(str(k)) or [None] * taus.size)[j]
                n = (r["n_mics_finite"].get(str(k)) or [0] * taus.size)[j]
                fl = (r["g2_floor"].get(str(k)) or [None] * taus.size)[j]
                if g is None or se is None or fl is None or n < 2 or not np.isfinite(g):
                    continue
                num += float(g) * n
                fnum += float(fl) * n
                var += (float(se) * n) ** 2
                den += n
            if den:
                mean = num / den
                fbar = fnum / den
                se_pooled = max(math.sqrt(var) / den, fbar / math.sqrt(den))
                v, v_se = g2_to_V(mean, se_pooled, den)
                gs.append(mean)
                ses.append(se_pooled)
                fls.append(fbar)
                ns.append(den)
                vs.append(v)
                vse.append(v_se)
            else:
                gs.append(float("nan"))
                ses.append(float("nan"))
                fls.append(float("nan"))
                ns.append(0)
                vs.append(None)
                vse.append(None)
        if any(v is not None for v in vs):
            Vk[k], Vs[k], Nk[k], FL[k], G2[k], SE[k] = vs, vse, ns, fls, gs, ses
    # the unwrapped estimator pools as a plain weighted mean of variances
    Vu: dict[int, list[float | None]] = {}
    Us: dict[int, list[float | None]] = {}
    for k in range(1, K_MAX_ACOUSTIC + 1):
        vu: list[float | None] = []
        us: list[float | None] = []
        for j in range(taus.size):
            num = var = 0.0
            den = 0
            for r in usable:
                v = (r.get("V_k_unwrapped_rad2", {}).get(str(k)) or [None] * taus.size)[j]
                se = (r.get("V_k_unwrapped_se_rad2", {}).get(str(k)) or [None] * taus.size)[j]
                if v is None or se is None or not np.isfinite(v) or not np.isfinite(se):
                    continue
                num += float(v)
                var += float(se) ** 2
                den += 1
            if den:
                vu.append(num / den)
                us.append(math.sqrt(var) / den)
            else:
                vu.append(None)
                us.append(None)
        if any(v is not None for v in vu):
            Vu[k], Us[k] = vu, us
    orders = [k for k in sorted(Vk) if sum(v is not None for v in Vk[k]) >= 4]
    orders_un = [k for k in sorted(Vu) if sum(v is not None for v in Vu[k]) >= 4]
    win = int(round(float(usable[0]["window_s"]) * float(usable[0]["fs_hz"])))
    fs_p = float(usable[0]["fs_hz"])
    marg: dict[int, float] = {}
    for k in range(1, K_MAX_ACOUSTIC + 1):
        vals = [
            float(r["margin_db_median"][str(k)])
            for r in usable
            if str(k) in (r.get("margin_db_median") or {})
        ]
        if vals:
            marg[k] = float(np.median(vals))
    fits_un = (
        fit_cells(orders_un, taus, Vu, Us, win, fs_p, marg, estimator="unwrapped")
        if len(orders_un) >= 2
        else {}
    )
    fits_coh = (
        fit_cells(orders, taus, Vk, Vs, win, fs_p, marg, estimator="coherence")
        if len(orders) >= 2
        else {}
    )
    primary = "unwrapped" if fits_un.get("best_by_aic") else "coherence"
    fits = fits_un if primary == "unwrapped" else fits_coh
    return {
        "support": name,
        "identified": bool(fits),
        "n_windows_pooled": len(usable),
        "windows": [r["support"] for r in usable],
        "orders_used": orders_un if primary == "unwrapped" else orders,
        "n_orders_used": len(orders_un if primary == "unwrapped" else orders),
        "orders_coherence": orders,
        "window_s": usable[0].get("window_s"),
        "fs_hz": usable[0].get("fs_hz"),
        "frame_rate_hz": usable[0].get("frame_rate_hz"),
        "resolution_hz": usable[0].get("resolution_hz"),
        "rate_rps": float(np.mean([r.get("rate_rps", np.nan) for r in usable])),
        "lags_s": taus.tolist(),
        "g2_mean": {str(k): G2[k] for k in orders},
        "g2_se": {str(k): SE[k] for k in orders},
        "g2_floor": {str(k): FL[k] for k in orders},
        "V_k_rad2": {str(k): Vk[k] for k in orders},
        "V_k_se_rad2": {str(k): Vs[k] for k in orders},
        "V_k_unwrapped_rad2": {str(k): Vu[k] for k in orders_un},
        "V_k_unwrapped_se_rad2": {str(k): Us[k] for k in orders_un},
        "n_mics_finite": {str(k): Nk[k] for k in orders},
        "primary_estimator": primary,
        "fits": fits,
        "fits_unwrapped": fits_un,
        "fits_coherence": fits_coh,
    }


def load_dregon_bench(limit: int | None) -> list[tuple[str, np.ndarray]]:
    from data_processing.streams import iter_published_frames

    out: list[tuple[str, np.ndarray]] = []
    for frame in iter_published_frames("DREGON-frames"):
        meta = _meta_dict(frame)
        if meta.get("split") != "motor":
            continue
        m = re.match(r"motor_(Motor\d)_(\d+)$", str(meta.get("recording_id")))
        if not m or int(m.group(2)) not in DREGON_BENCH_SETPOINTS:
            continue
        out.append((str(meta.get("recording_id")), np.asarray(frame["audio"].data, np.float64)))
        if limit is not None and len(out) >= limit:
            break
    return sorted(out)


def bench_rate(audio: np.ndarray, fs: float, setpoint: int) -> dict[str, Any]:
    """Shaft rate of one bench recording: the throttle law and the comb evidence."""
    from experiments.stochastic_fit.bench import _welch, estimate_rate

    n_an = 1 << 19
    n = min(audio.shape[1], 2 * n_an)
    seg = audio[:, :n] - audio[:, :n].mean(axis=1, keepdims=True)
    psd = np.stack([_welch(seg[m], n_an) for m in range(seg.shape[0])])
    df = fs / n_an
    est = float(estimate_rate(psd.mean(axis=0), df))
    law = DREGON_THROTTLE_A * setpoint + DREGON_THROTTLE_B
    return {"rate_comb_rps": est, "rate_law_rps": float(law), "df_hz": df}


def planted_control(
    *,
    fs: float = 44100.0,
    seconds: float = BENCH_SECONDS,
    rate_rps: float = 68.455,
    sigma: float = 0.4,
    lam: float = 6.0,
    d0: float = 0.02,
    d_exponent: float = 1.0,
    n_mic: int = 8,
    k_max: int = K_MAX_ACOUSTIC,
    snr_db: float = 22.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Synthesise a rotor with KNOWN ``(sigma, lam, D_k)`` and read it back.

    The acoustic instrument is the load-bearing part of this study, so it is
    run once on a signal whose answer is known: an exact integrated-OU shaft
    angle, independent Wiener phases with ``D_k = d0 k^d_exponent``,
    two-blade-like amplitudes (odd orders 10 dB down), independent per-mic
    fixed phases and white noise at ``snr_db``. What comes back is compared
    with what went in, and the report states the ratio.
    """
    from scipy.signal import lfilter

    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    dt = 1.0 / fs
    a = math.exp(-lam * dt)
    nu = lfilter([1.0], [1.0, -a], rng.normal(0.0, sigma * math.sqrt(1.0 - a * a), n))
    theta = np.cumsum(nu) * dt
    t = np.arange(n) * dt
    dk = {k: d0 * k**d_exponent for k in range(1, k_max + 1)}
    audio = np.zeros((n_mic, n))
    for k in range(1, k_max + 1):
        eps = np.cumsum(rng.normal(0.0, math.sqrt(2.0 * dk[k] * dt), n))
        base = 2.0 * np.pi * k * rate_rps * t + k * theta + eps
        amp = 1.0 if k % 2 == 0 else 0.35
        for m in range(n_mic):
            audio[m] += amp * np.cos(base + rng.uniform(0.0, 2.0 * np.pi))
    audio += rng.normal(0.0, math.sqrt(float(np.mean(audio**2)) / 10 ** (snr_db / 10)), audio.shape)
    row = analyse_acoustic_support(
        "planted", audio, fs, np.full(n, rate_rps), rate_rps, k_max=k_max
    )
    truth = {
        "sigma_nu_rad_s": sigma,
        "lam_1_s": lam,
        "D_theta_rad2_s": sigma**2 / lam,
        "D_k_rad2_s": {str(k): dk[k] for k in dk},
        "d0": d0,
        "d_exponent": d_exponent,
        "snr_db": snr_db,
        "seconds": seconds,
        "rate_rps": rate_rps,
    }
    f = (row.get("fits") or {}).get("ou_free_D") or {}
    recovered: dict[str, Any] = {"identified": bool(f.get("identified"))}
    if f.get("identified"):
        ks = np.asarray(f["orders"], dtype=float)
        dks = np.asarray(f["D_k_rad2_s"], dtype=float)
        ok = np.isfinite(dks) & (dks > 0)
        recovered.update(
            sigma_nu_rad_s=f["sigma_nu_rad_s"],
            lam_1_s=f["lam_1_s"],
            D_theta_rad2_s=f["D_theta_rad2_s"],
            D_k_rad2_s=f["D_k_rad2_s"],
            orders=f["orders"],
            sigma_ratio=f["sigma_nu_rad_s"] / sigma,
            lam_ratio=f["lam_1_s"] / lam,
            D_theta_ratio=f["D_theta_rad2_s"] / (sigma**2 / lam),
            D_k_loglog_slope=(
                float(np.polyfit(np.log(ks[ok]), np.log(dks[ok]), 1)[0]) if ok.sum() >= 3 else None
            ),
            best_model_by_aic=row["fits"].get("best_by_aic"),
        )
    return {"truth": truth, "recovered": recovered, "support": row}


def run_dregon_bench(limit: int | None) -> dict[str, Any]:
    fs = 44100.0
    recs = load_dregon_bench(limit)
    out: dict[str, Any] = {"support": "dregon_bench", "recordings": {}}
    for rid, audio in recs:
        setpoint = int(rid.rsplit("_", 1)[1])
        n = int(min(audio.shape[1], BENCH_SECONDS * fs))
        seg = audio[:, :n] - audio[:, :n].mean(axis=1, keepdims=True)
        rate = bench_rate(audio, fs, setpoint)
        centres = np.full(n, rate["rate_comb_rps"])
        row = analyse_acoustic_support(rid, seg, fs, centres, rate["rate_comb_rps"])
        row.update(rate)
        row["setpoint"] = setpoint
        row["seconds"] = n / fs
        out["recordings"][rid] = row
    out["n_recordings"] = len(recs)
    return out


def run_michaels_acoustic(limit: int | None) -> dict[str, Any]:
    from experiments.stochastic_fit import clips as C

    rec = C.load_recording(MICHAELS_DATASET, MICHAELS_REC, None, MICHAELS_RPS_KEY)
    wins = C.windows(
        rec,
        seconds=MICHAELS_CLIP_S,
        max_clips=MICHAELS_MAX_CLIPS if limit is None else min(MICHAELS_MAX_CLIPS, limit),
        min_rps=MICHAELS_MIN_RPS,
    )
    out: dict[str, Any] = {"support": "michaels_fly125", "clips": {}, "n_windows": len(wins)}
    fs = float(rec.sr)
    for start_s, dur in wins:
        clip = rec.cut(start_s, dur)
        audio = np.asarray(clip.audio, dtype=np.float64)
        audio = audio - audio.mean(axis=1, keepdims=True)
        rps = np.asarray(clip.rps, dtype=np.float64)  # (R, T) at audio rate
        for r in range(rps.shape[0]):
            centres = rps[r]
            rate = float(np.mean(centres))
            # isolation: distance in Hz from order k of THIS rotor to the
            # nearest line of any other rotor or order
            others = np.array(
                [
                    float(np.mean(rps[q])) * kk
                    for q in range(rps.shape[0])
                    for kk in range(1, K_MAX_ACOUSTIC + 3)
                    if q != r
                ]
            )
            iso = []
            for k in range(1, K_MAX_ACOUSTIC + 1):
                fc = k * rate
                d = np.abs(others - fc)
                same = np.abs(np.arange(1, K_MAX_ACOUSTIC + 3) * rate - fc)
                same[k - 1] = np.inf
                iso.append(float(min(d.min() if d.size else np.inf, same.min())))
            name = f"{MICHAELS_REC}@{start_s:.3f}+{dur:g}#r{r}"
            row = analyse_acoustic_support(
                name,
                audio,
                fs,
                centres,
                rate,
                isolation=iso,
                window_s=ACOUSTIC_FLIGHT_WINDOW_S,
            )
            row["rotor"] = r
            row["start_s"] = float(start_s)
            row["duration_s"] = float(dur)
            row["isolation_hz"] = iso
            out["clips"][name] = row
    by_rotor: dict[int, list[dict[str, Any]]] = {}
    for row in out["clips"].values():
        by_rotor.setdefault(int(row["rotor"]), []).append(row)
    out["supports"] = {}
    for r, rows in sorted(by_rotor.items()):
        pooled = pool_acoustic(f"{MICHAELS_REC}_cruise_rotor{r}", rows)
        pooled["rotor"] = r
        out["supports"][pooled["support"]] = pooled
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Verdicts and comparison
# ═════════════════════════════════════════════════════════════════════════════


def telemetry_verdict(rep: dict[str, Any]) -> dict[str, Any]:
    """The per-rig read, from the mandated 0.5 Hz high-pass plus the wide variant.

    The headline numbers are the 0.5 Hz ones. When their corner lands below
    the 0.5 Hz fit band the high-passed series CANNOT locate it, so the
    detrend-only variant - whose band starts at 0.05 Hz - supplies
    ``lam``/``sigma`` and the verdict says which instrument spoke.
    """
    pooled = rep["highpass"][f"{HP_HZ[0]:g}"]["pooled"]
    wide = (rep["highpass"].get(DETREND_KEY) or {}).get("pooled") or {}
    q = rep["quantisation"]
    fs = rep["fs_hz"]
    band_hi = PSD_F_HI_FRAC * fs
    lam = pooled.get("lam_1_s")
    corner = pooled.get("corner_hz")
    d_ou = pooled.get("D_theta_ou_rad2_s")
    d_q = q.get("D_q_at_update_rate")
    pref = pooled.get("preferred_by_aic")
    source = "highpass_0.5"
    if corner is None:
        label = "unidentified"
    elif pref == "white" or corner > 0.8 * band_hi:
        label = "wiener"
    elif corner < 1.2 * PSD_F_LO_HZ:
        w_corner = wide.get("corner_hz")
        if (
            w_corner is not None
            and PSD_F_LO_WIDE_HZ * 1.2 < w_corner < PSD_F_HI_FRAC * fs
            and wide.get("preferred_by_aic") == "ou"
        ):
            label = "integrated-OU (corner under the 0.5 Hz high-pass; located detrend-only)"
            source = DETREND_KEY
            lam = wide.get("lam_1_s")
            corner = w_corner
        else:
            label = "quasi-static (corner below every fit band; only sigma^2/lam identified)"
    else:
        label = "integrated-OU"
    sigma = (wide if source == DETREND_KEY else pooled).get("sigma_nu_rad_s")
    d_out = (wide if source == DETREND_KEY else pooled).get("D_theta_ou_rad2_s") or d_ou
    quant_share = float(d_q / d_out) if (d_out and d_q) else None
    return {
        "verdict": label,
        "estimator": source,
        "sigma_nu_rad_s": sigma,
        "lam_1_s": lam,
        "corner_hz": corner,
        "D_theta_rad2_s": d_out,
        "sigma_nu_rad_s_hp05": pooled.get("sigma_nu_rad_s"),
        "lam_1_s_hp05": pooled.get("lam_1_s"),
        "D_theta_rad2_s_hp05": d_ou,
        "sigma_nu_rad_s_hp2": (rep["highpass"].get("2") or {})
        .get("pooled", {})
        .get("sigma_nu_rad_s"),
        "lam_1_s_hp2": (rep["highpass"].get("2") or {}).get("pooled", {}).get("lam_1_s"),
        "sigma_nu_rad_s_detrend": wide.get("sigma_nu_rad_s"),
        "lam_1_s_detrend": wide.get("lam_1_s"),
        "D_theta_rad2_s_detrend": wide.get("D_theta_ou_rad2_s"),
        "nu_rms_rad_s": pooled.get("nu_rms_rad_s"),
        "tau_slope1p5_s": pooled.get("tau_slope1p5_s"),
        "lam_from_slope1p5_1_s": pooled.get("lam_from_slope1p5_1_s"),
        "tau_slope1p5_s_detrend": wide.get("tau_slope1p5_s"),
        "lam_from_slope1p5_1_s_detrend": wide.get("lam_from_slope1p5_1_s"),
        "delta_aic_ou_minus_white": pooled.get("delta_aic_ou_minus_white"),
        "delta_bic_ou_minus_white": pooled.get("delta_bic_ou_minus_white"),
        "delta_aic_ou_minus_white_detrend": wide.get("delta_aic_ou_minus_white"),
        "D_q_at_update_rate": d_q,
        "D_q_over_D_theta": quant_share,
        "quantisation_caveat": (
            "quantisation alone can explain the diffusive tail"
            if quant_share is not None and quant_share > 0.5
            else "quantisation floor is below the measured diffusion"
        ),
        "held_fraction": q.get("held_fraction"),
        "update_rate_hz": q.get("update_rate_hz"),
        "band_hz": [PSD_F_LO_HZ, band_hi],
        "band_hz_detrend": [PSD_F_LO_WIDE_HZ, band_hi],
    }


def _saturated(row: dict[str, Any]) -> bool:
    """True when the phase excursion already exceeds the linear regime at the
    SHORTEST usable lag of the LOWEST order, so no cell of this support can be
    read: the harmonic is blurred rather than a line, and the fit that comes
    back from such a support is meaningless."""
    src = (
        row.get("V_k_unwrapped_rad2")
        if row.get("primary_estimator") == "unwrapped"
        else row.get("V_k_rad2")
    ) or {}
    if not src:
        return True
    k0 = min(int(k) for k in src)
    series = [v for v in (src[str(k0)] or []) if v is not None and np.isfinite(v)]
    if not series:
        return True
    return bool(series[0] / k0**2 > ACOUSTIC_V_LINEAR_MAX)


def acoustic_verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sat = [r for r in rows if r.get("identified") and _saturated(r)]
    ok = [
        r
        for r in rows
        if r.get("identified") and r["fits"].get("best_by_aic") and not _saturated(r)
    ]
    if not ok:
        return {
            "verdict": (
                "unidentified (saturated: V_theta exceeds "
                f"{ACOUSTIC_V_LINEAR_MAX} rad^2 at the shortest usable lag, so the "
                "harmonic is blurred rather than a line)"
                if sat
                else "unidentified"
            ),
            "n_supports": len(rows),
            "n_saturated": len(sat),
            "n_identified": 0,
        }
    best_names = [r["fits"]["best_by_aic"] for r in ok]
    sig, lam, dth, dk_slope = [], [], [], []
    for r in ok:
        f = r["fits"]["ou_free_D"]
        if not f.get("identified"):
            continue
        if f.get("sigma_nu_rad_s"):
            sig.append(f["sigma_nu_rad_s"])
            lam.append(f["lam_1_s"])
            dth.append(f["D_theta_rad2_s"])
        ks = np.asarray(f["orders"], dtype=float)
        dks = np.asarray(f["D_k_rad2_s"], dtype=float)
        m = np.isfinite(dks) & (dks > 0) & (ks > 0)
        if m.sum() >= 3:
            dk_slope.append(float(np.polyfit(np.log(ks[m]), np.log(dks[m]), 1)[0]))
    lam_med = float(np.median(lam)) if lam else None
    verdict = "unidentified"
    if lam_med is not None:
        verdict = "integrated-OU" if lam_med < 200.0 else "wiener"
    return {
        "verdict": verdict,
        "n_supports": len(rows),
        "n_saturated": len(sat),
        "n_identified": len(ok),
        "best_model_votes": {m: best_names.count(m) for m in set(best_names)},
        "sigma_nu_rad_s_median": float(np.median(sig)) if sig else None,
        "sigma_nu_rad_s_range": [float(np.min(sig)), float(np.max(sig))] if sig else None,
        "lam_1_s_median": lam_med,
        "lam_1_s_range": [float(np.min(lam)), float(np.max(lam))] if lam else None,
        "D_theta_rad2_s_median": float(np.median(dth)) if dth else None,
        "D_theta_rad2_s_range": [float(np.min(dth)), float(np.max(dth))] if dth else None,
        "D_k_loglog_slope_median": float(np.median(dk_slope)) if dk_slope else None,
        "D_k_loglog_slope_range": (
            [float(np.min(dk_slope)), float(np.max(dk_slope))] if dk_slope else None
        ),
    }


def build_comparison(res: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"c3_reference": C3_REFERENCE, "rows": []}
    tel = res.get("telemetry", {})
    ac = res.get("acoustics", {})

    def tv(rig: str) -> dict[str, Any] | None:
        return (tel.get(rig) or {}).get("verdict")

    def av(name: str) -> dict[str, Any] | None:
        return (ac.get(name) or {}).get("verdict")

    pairs = [
        ("DREGON airframe", "dregon_room1", "dregon_bench", "dregon"),
        ("Michael's M100", "michaels", "michaels_fly125", "michaels"),
    ]
    for label, tkey, akey, cref in pairs:
        t, a = tv(tkey), av(akey)
        ref = C3_REFERENCE[cref]
        row: dict[str, Any] = {"rig": label, "c3": ref}
        if t:
            row["telemetry"] = {
                k: t.get(k) for k in ("verdict", "sigma_nu_rad_s", "lam_1_s", "D_theta_rad2_s")
            }
        if a:
            row["acoustics"] = {
                k: a.get(k)
                for k in (
                    "verdict",
                    "sigma_nu_rad_s_median",
                    "lam_1_s_median",
                    "D_theta_rad2_s_median",
                    "D_k_loglog_slope_median",
                )
            }
        # C3 comparison at the fixed reference lambda: sigma that reproduces
        # the measured long-lag diffusion, sigma^2 = D_theta * lam_ref
        for src, key in (("telemetry", "D_theta_rad2_s"), ("acoustics", "D_theta_rad2_s_median")):
            d = (row.get(src) or {}).get(key)
            if d:
                sig_eq = math.sqrt(d * ref["lam_ref"])
                row.setdefault("sigma_at_lam_ref", {})[src] = sig_eq
                row.setdefault("ratio_c3_over_measured", {})[src] = ref["sigma_rad_s"] / sig_eq
        out["rows"].append(row)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Figures
# ═════════════════════════════════════════════════════════════════════════════


def _panel_grid(n: int) -> tuple[int, int]:
    cols = min(3, max(1, n))
    rows = int(math.ceil(n / cols))
    return rows, cols


def fig_structure_functions(res: dict[str, Any], paths: list[Path]) -> None:
    tel = res.get("telemetry", {})
    rigs = [r for r in tel if tel[r].get("highpass")]
    if not rigs:
        return
    rows, cols = _panel_grid(len(rigs))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.7 * rows), squeeze=False)
    for ax, rig in zip(axes.ravel(), rigs, strict=False):
        rep = tel[rig]
        tau = np.asarray(rep["lags_s"], dtype=float)
        hp = rep["highpass"][f"{HP_HZ[0]:g}"]
        for row in hp["per_rotor"]:
            if not row.get("S_rad2"):
                continue
            s = np.array([np.nan if v is None else v for v in row["S_rad2"]])
            ax.loglog(tau, s, lw=1.1, alpha=0.85, label=f"rotor {row['rotor']}")
            if row.get("S_model_ou_rad2"):
                ax.loglog(
                    tau,
                    np.asarray(row["S_model_ou_rad2"], dtype=float),
                    ls=":",
                    lw=0.9,
                    color="k",
                    alpha=0.5,
                )
        hp2 = rep["highpass"][f"{HP_HZ[1]:g}"]
        for row in hp2["per_rotor"]:
            if not row.get("S_rad2"):
                continue
            s = np.array([np.nan if v is None else v for v in row["S_rad2"]])
            ax.loglog(tau, s, lw=0.8, alpha=0.45, color="grey")
        hpd = rep["highpass"].get(DETREND_KEY) or {"per_rotor": []}
        for row in hpd["per_rotor"]:
            if not row.get("S_rad2"):
                continue
            s = np.array([np.nan if v is None else v for v in row["S_rad2"]])
            ax.loglog(tau, s, lw=0.9, alpha=0.7, color="darkorange", ls="--")
        # guides anchored at the middle of the grid
        finite = [
            v
            for row in hp["per_rotor"]
            for v in (row.get("S_rad2") or [])
            if v is not None and np.isfinite(v)
        ]
        if finite:
            anchor_i = len(tau) // 2
            base = np.nanmedian(
                [
                    (row.get("S_rad2") or [None])[anchor_i]
                    for row in hp["per_rotor"]
                    if row.get("S_rad2") and row["S_rad2"][anchor_i] is not None
                ]
            )
            t0 = tau[anchor_i]
            if np.isfinite(base) and base > 0:
                ax.loglog(tau, base * (tau / t0), "k--", lw=0.8, alpha=0.7)
                ax.loglog(tau, base * (tau / t0) ** 2, "k-.", lw=0.8, alpha=0.7)
        dq = rep["quantisation"].get("D_q_at_update_rate")
        if dq:
            ax.loglog(tau, 2.0 * dq * tau, color="crimson", lw=1.2, alpha=0.8)
        v = rep["verdict"]
        ax.set_title(
            f"{rig}  fs={rep['fs_hz']:.0f} Hz\n{v['verdict']}"
            + (f", 1/lam={1.0 / v['lam_1_s']:.3f} s" if v.get("lam_1_s") else ""),
            fontsize=8.5,
        )
        ax.set_xlabel("lag tau [s]")
        ax.set_ylabel("S(tau) = Var[theta(t+tau)-theta(t)]  [rad^2]")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=6, loc="upper left", ncols=2)
    for ax in axes.ravel()[len(rigs) :]:
        ax.axis("off")
    fig.suptitle(
        "Shaft-angle-error structure function, native-rate telemetry. "
        f"Colour: {HP_HZ[0]:g} Hz high-pass per rotor; grey: {HP_HZ[1]:g} Hz; "
        "orange dashed: linear detrend only (the only variant whose long lags "
        "are not annihilated by the high-pass); dotted black: fitted OU through "
        "the same trend removal; dashed/dash-dot black: slope 1 / slope 2; "
        "red: quantisation floor 2 D_q tau",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, paths)


def fig_speed_psd(res: dict[str, Any], paths: list[Path]) -> None:
    tel = res.get("telemetry", {})
    rigs = [r for r in tel if tel[r].get("psd", {}).get("f_hz")]
    if not rigs:
        return
    rows, cols = _panel_grid(len(rigs))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.6 * rows), squeeze=False)
    for ax, rig in zip(axes.ravel(), rigs, strict=False):
        rep = tel[rig]
        f = np.asarray(rep["psd"]["f_hz"], dtype=float)
        p = np.asarray(rep["psd"]["P_rad2_s2_per_hz"], dtype=float)
        m = f > 0
        ax.loglog(f[m], p[m], lw=0.9, color="C0", label="Welch, 0.5 Hz high-pass")
        fit = rep["psd"]["fit"]
        if fit.get("identified"):
            sos = hp_sos(rep["fs_hz"], HP_HZ[0])
            resp = hp_power_response(sos, f[m], rep["fs_hz"])
            ou = ou_psd(f[m], fit["ou"]["sigma_nu_rad_s"], fit["ou"]["lam_1_s"]) * resp
            ax.loglog(f[m], ou, "C3", lw=1.4, label="OU (Lorentzian)")
            ax.loglog(
                f[m],
                np.full(m.sum(), fit["white"]["s0_rad2_s2_per_hz"]) * resp,
                "C2--",
                lw=1.2,
                label="white",
            )
            ax.axvline(fit["ou"]["corner_hz"], color="C3", ls=":", lw=1.0)
            ax.axvspan(*fit["band_hz"], color="grey", alpha=0.10)
            ax.set_title(
                f"{rig}\nsigma={fit['ou']['sigma_nu_rad_s']:.3g} rad/s, "
                f"lam={fit['ou']['lam_1_s']:.3g} 1/s, "
                f"dAIC(OU-white)={fit['delta_aic_ou_minus_white']:+.0f}",
                fontsize=8.5,
            )
        q = rep["quantisation"]
        if q.get("step_rps_used") and q.get("update_rate_hz"):
            s_q = 2.0 * (2 * np.pi * q["step_rps_used"]) ** 2 / (12.0 * q["update_rate_hz"])
            ax.axhline(s_q, color="crimson", lw=1.0, alpha=0.8)
        det = rep.get("psd_detrend")
        if det and det.get("f_hz"):
            fd = np.asarray(det["f_hz"], dtype=float)
            pd_ = np.asarray(det["P_rad2_s2_per_hz"], dtype=float)
            md = fd > 0
            ax.loglog(fd[md], pd_[md], lw=0.9, color="darkorange", label="Welch, detrend only")
            fitd = det["fit"]
            if fitd.get("identified"):
                ax.loglog(
                    fd[md],
                    ou_psd(fd[md], fitd["ou"]["sigma_nu_rad_s"], fitd["ou"]["lam_1_s"]),
                    color="saddlebrown",
                    lw=1.2,
                    ls="-.",
                    label="OU, detrend only",
                )
        ax.set_xlabel("f [Hz]")
        ax.set_ylabel("S_nu(f)  [(rad/s)^2/Hz]")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=6.0)
    for ax in axes.ravel()[len(rigs) :]:
        ax.axis("off")
    fig.suptitle(
        "Speed-error PSD per rig (rotors pooled). Blue: 0.5 Hz high-pass, with the OU "
        "and white fits THROUGH its |H|^4 power response. Orange: linear detrend only, "
        "with its own OU fit - the only variant that can place a corner below 0.5 Hz. "
        "Grey band: the 0.5 Hz - 0.4 fs fit band; red line: the quantisation-noise "
        "floor at the observed update rate",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, paths)


def fit_rows(blob: dict[str, Any]) -> list[dict[str, Any]]:
    """The rows of one acoustic group that carry the fits.

    ``supports`` (pooled) wins when a group has it, because a per-window fit
    of a flight support has no cross-order leverage; the per-window rows stay
    in the JSON as the diagnostic they are.
    """
    for key in ("supports", "recordings", "clips"):
        rows = list((blob.get(key) or {}).values())
        if rows:
            return rows
    return []


def _acoustic_rows(res: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, blob in (res.get("acoustics") or {}).items():
        for row in fit_rows(blob):
            if row.get("identified"):
                row = dict(row)
                row["support_group"] = name
                rows.append(row)
    return rows


def fig_acoustic_Vk(res: dict[str, Any], paths: list[Path]) -> None:
    rows = _acoustic_rows(res)
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["support"])[:12]
    nrow, ncol = _panel_grid(len(rows))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.5 * nrow), squeeze=False)
    for ax, row in zip(axes.ravel(), rows, strict=False):
        tau = np.asarray(row["lags_s"], dtype=float)
        cmap = plt.get_cmap("viridis")
        src = (
            row.get("V_k_unwrapped_rad2")
            if row.get("primary_estimator") == "unwrapped"
            else row.get("V_k_rad2")
        ) or {}
        for key, series in sorted(src.items(), key=lambda kv: int(kv[0])):
            k = int(key)
            v = np.array([np.nan if x is None else x for x in series], dtype=float)
            if v.size != tau.size:
                continue
            ax.loglog(
                tau,
                v / k**2,
                color=cmap((k - 1) / max(K_MAX_ACOUSTIC - 1, 1)),
                marker="o",
                ms=2.2,
                lw=1.0,
                label=f"k={k}",
            )
        f = row["fits"].get("ou_free_D", {})
        if f.get("identified") and f.get("sigma_nu_rad_s"):
            ax.loglog(
                tau,
                ou_structure(tau, f["sigma_nu_rad_s"], f["lam_1_s"]),
                "k--",
                lw=1.3,
                label="fitted V_theta",
            )
            ax.set_title(
                f"{row['support']}\nsigma={f['sigma_nu_rad_s']:.3g} rad/s, "
                f"lam={f['lam_1_s']:.3g} 1/s, best={row['fits'].get('best_by_aic')}",
                fontsize=7.5,
            )
        else:
            ax.set_title(row["support"], fontsize=7.5)
        ax.set_xlabel("lag tau [s]")
        ax.set_ylabel("V_k(tau)/k^2  [rad^2]")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=5.5, ncols=2)
    for ax in axes.ravel()[len(rows) :]:
        ax.axis("off")
    fig.suptitle(
        "Order-normalised phase increment variance. Under the model every curve "
        "collapses onto V_theta(tau) plus its own 2 D_k tau / k^2 and SNR offset",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, paths)


def fig_Dk_vs_k(res: dict[str, Any], paths: list[Path]) -> None:
    rows = _acoustic_rows(res)
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["support_group"], []).append(row)
    for ax, (grp, rs) in zip(axes, sorted(groups.items()), strict=False):
        for row in rs:
            f = row["fits"].get("ou_free_D", {})
            if not f.get("identified"):
                continue
            ks = np.asarray(f["orders"], dtype=float)
            dk = np.asarray(f["D_k_rad2_s"], dtype=float)
            ax.loglog(ks, dk, marker="o", ms=3, lw=0.9, alpha=0.8, label=row["support"][-18:])
        ax.set_title(f"{grp}: per-order diffusion", fontsize=9)
        ax.set_xlabel("order k")
        ax.set_ylabel("D_k [rad^2/s]")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=5.5, ncols=2)
        lim = ax.get_ylim()
        kk = np.array([1.0, float(K_MAX_ACOUSTIC)])
        for pw, ls in ((0.0, "--"), (1.0, "-."), (2.0, ":")):
            ax.plot(
                kk, np.sqrt(lim[0] * lim[1]) * (kk / kk[0]) ** pw, "k", ls=ls, lw=0.8, alpha=0.6
            )
        ax.set_ylim(lim)
    for ax in axes[len(groups) :]:
        ax.axis("off")
    fig.suptitle(
        "D_k against k, per support. Guides: D constant (dashed), D prop k (dash-dot), "
        "D prop k^2 (dotted)",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, paths)


def fig_summary_table(res: dict[str, Any], paths: list[Path]) -> None:
    lines = summary_rows(res)
    fig, ax = plt.subplots(figsize=(13.5, 0.42 * (len(lines) + 2) + 0.8))
    ax.axis("off")
    tab = ax.table(
        cellText=[r[1:] for r in lines[1:]],
        colLabels=lines[0][1:],
        rowLabels=[r[0] for r in lines[1:]],
        loc="center",
        cellLoc="center",
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(6.8)
    tab.scale(1.0, 1.25)
    ax.set_title(
        "Shaft-phase prior, measured. Verdict / sigma_nu [rad/s] / lam [1/s] / "
        "D_theta [rad^2/s] / quantisation share",
        fontsize=9,
    )
    fig.tight_layout()
    _save(fig, paths)


def _save(fig: Any, paths: list[Path]) -> None:
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# Report
# ═════════════════════════════════════════════════════════════════════════════


def _f(x: Any, fmt: str = "{:.3g}") -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    if isinstance(x, (int, float)):
        return fmt.format(x)
    return str(x)


def summary_rows(res: dict[str, Any]) -> list[list[str]]:
    head = [
        "rig / support",
        "instrument",
        "verdict",
        "sigma_nu",
        "lam",
        "1/lam [s]",
        "D_theta",
        "tau(slope 1.5)",
        "dAIC OU-white",
        "D_q/D_theta",
    ]
    out = [head]
    for rig, rep in (res.get("telemetry") or {}).items():
        v = rep.get("verdict") or {}
        out.append(
            [
                rig + (" (aux)" if rep.get("auxiliary") else ""),
                f"telemetry {rep['fs_hz']:.0f} Hz",
                _f(v.get("verdict")),
                _f(v.get("sigma_nu_rad_s")),
                _f(v.get("lam_1_s")),
                _f(1.0 / v["lam_1_s"] if v.get("lam_1_s") else None),
                _f(v.get("D_theta_rad2_s")),
                _f(v.get("tau_slope1p5_s")),
                _f(v.get("delta_aic_ou_minus_white"), "{:+.0f}"),
                _f(v.get("D_q_over_D_theta"), "{:.2f}"),
            ]
        )
    for name, blob in (res.get("acoustics") or {}).items():
        v = blob.get("verdict") or {}
        out.append(
            [
                name,
                f"acoustics ({v.get('n_identified', 0)}/{v.get('n_supports', 0)} supports)",
                _f(v.get("verdict")),
                _f(v.get("sigma_nu_rad_s_median")),
                _f(v.get("lam_1_s_median")),
                _f(1.0 / v["lam_1_s_median"] if v.get("lam_1_s_median") else None),
                _f(v.get("D_theta_rad2_s_median")),
                _f(U_SLOPE_1P5 / v["lam_1_s_median"] if v.get("lam_1_s_median") else None),
                "n/a",
                "n/a",
            ]
        )
    return out


def markdown_table(rows: list[list[str]]) -> str:
    head, body = rows[0], rows[1:]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for r in body:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def write_findings(res: dict[str, Any], out_dir: Path) -> Path:
    rows = summary_rows(res)
    tel = res.get("telemetry") or {}
    ac = res.get("acoustics") or {}
    cmp_ = res.get("comparison") or {}
    L: list[str] = []
    A = L.append
    A("# Shaft-phase noise, measured — findings")
    A("")
    A(
        "Source: `scripts/noise_v2_shaft_phase.py`. Numbers: "
        "`results/noise_v2/shaft/shaft.json`. Figures: "
        "`docs/explainers/noise-model-v2-plan/shaft_*.png`."
    )
    A("")
    A(
        "The model under test is "
        "`V_k(tau) = k^2 V_theta(tau) + 2 D_k tau` with "
        "`V_theta(tau) = 2 sigma_nu^2 [tau/lam - (1-exp(-lam tau))/lam^2]`. "
        "Integrated-OU means the speed error `nu` is correlated over `1/lam`; "
        "Wiener means `nu` is white, so `V_theta` is linear in `tau` at every lag. "
        f"The integrated-OU curve has local log-log slope 1.5 at `lam tau = {U_SLOPE_1P5:.4f}`, "
        "which is how a measured lag is turned into `lam`."
    )
    A("")
    A("## Verdict per rig")
    A("")
    A(markdown_table(rows))
    A("")
    A("Figure: `shaft_summary_table.png`.")
    A("")
    A("## What the telemetry says")
    A("")
    for rig, rep in tel.items():
        v = rep.get("verdict") or {}
        q = rep.get("quantisation") or {}
        aux = (
            " This rig is AUXILIARY: it is a command track, not a speed measurement."
            if rep.get("auxiliary")
            else ""
        )
        A(
            f"- **{rig}** ({rep['rps_key']}, {rep['fs_hz']:.0f} Hz, {rep['n_flights']} flights, "
            f"{rep.get('analysed_seconds', 0):.0f} s analysed). Verdict **{v.get('verdict')}**: "
            f"the Lorentzian fit gives `sigma_nu = {_f(v.get('sigma_nu_rad_s'))} rad/s`, "
            f"`lam = {_f(v.get('lam_1_s'))} 1/s` (corner {_f(v.get('corner_hz'))} Hz, "
            f"fit band {_f(v.get('band_hz', [None, None])[0])}-{_f(v.get('band_hz', [None, None])[1])} Hz), "
            f"`D_theta = sigma^2/lam = {_f(v.get('D_theta_rad2_s'))} rad^2/s`, and "
            f"`AIC(OU) - AIC(white) = {_f(v.get('delta_aic_ou_minus_white'), '{:+.0f}')}` "
            f"(BIC {_f(v.get('delta_bic_ou_minus_white'), '{:+.0f}')}). "
            f"The structure function crosses slope 1.5 at "
            f"`tau = {_f(v.get('tau_slope1p5_s'))} s`, i.e. "
            f"`lam = {_f(v.get('lam_from_slope1p5_1_s'))} 1/s`. "
            f"The ladder step is {_f(q.get('step_rps_used'))} rev/s, the value is held for "
            f"{_f(q.get('held_fraction'), '{:.1%}')} of samples and changes at "
            f"{_f(q.get('update_rate_hz'))} Hz, so `D_q = {_f(v.get('D_q_at_update_rate'))} rad^2/s` "
            f"= {_f(v.get('D_q_over_D_theta'), '{:.2f}')} of the measured `D_theta`; "
            f"{v.get('quantisation_caveat')}.{aux}"
        )
    A("")
    A("Figures: `shaft_structure_functions.png`, `shaft_speed_psd.png`.")
    A("")
    A("## What the acoustics say")
    A("")
    for name, blob in ac.items():
        v = blob.get("verdict") or {}
        A(
            f"- **{name}**: {v.get('n_identified')} of {v.get('n_supports')} supports identified. "
            f"Verdict **{v.get('verdict')}**: "
            f"`sigma_nu = {_f(v.get('sigma_nu_rad_s_median'))} rad/s` "
            f"(range {_f((v.get('sigma_nu_rad_s_range') or [None, None])[0])} - "
            f"{_f((v.get('sigma_nu_rad_s_range') or [None, None])[1])}), "
            f"`lam = {_f(v.get('lam_1_s_median'))} 1/s` "
            f"(range {_f((v.get('lam_1_s_range') or [None, None])[0])} - "
            f"{_f((v.get('lam_1_s_range') or [None, None])[1])}), "
            f"`D_theta = {_f(v.get('D_theta_rad2_s_median'))} rad^2/s`. "
            f"The per-order diffusion scales as `D_k ~ k^{_f(v.get('D_k_loglog_slope_median'), '{:.2f}')}` "
            f"(range {_f((v.get('D_k_loglog_slope_range') or [None, None])[0], '{:.2f}')} - "
            f"{_f((v.get('D_k_loglog_slope_range') or [None, None])[1], '{:.2f}')}). "
            f"Model votes by AIC: {v.get('best_model_votes')}."
        )
    A("")
    A("Figures: `shaft_acoustic_Vk.png`, `shaft_Dk_vs_k.png`.")
    A("")
    A("## Against the C3 fitted values")
    A("")
    A(
        "| rig | C3 sigma [rad/s] | C3 D [rad^2/s] | measured D_theta (telemetry) | "
        "measured D_theta (acoustics) | sigma at lam_ref=6 (telemetry) | "
        "sigma at lam_ref=6 (acoustics) | C3 sigma / measured |"
    )
    A("|---|---|---|---|---|---|---|---|")
    for row in cmp_.get("rows", []):
        sig_eq = row.get("sigma_at_lam_ref") or {}
        ratio = row.get("ratio_c3_over_measured") or {}
        A(
            f"| {row['rig']} | {_f(row['c3']['sigma_rad_s'])} | {_f(row['c3']['D_rad2_s'])} | "
            f"{_f((row.get('telemetry') or {}).get('D_theta_rad2_s'))} | "
            f"{_f((row.get('acoustics') or {}).get('D_theta_rad2_s_median'))} | "
            f"{_f(sig_eq.get('telemetry'))} | {_f(sig_eq.get('acoustics'))} | "
            f"tel {_f(ratio.get('telemetry'), '{:.1f}')}x / ac "
            f"{_f(ratio.get('acoustics'), '{:.1f}')}x |"
        )
    A("")
    A("## Caveats")
    A("")
    A(
        "- **Sample-and-hold.** DREGON `motors_measured` holds its value for "
        f"{_f((tel.get('dregon_room1') or {}).get('quantisation', {}).get('held_fraction'), '{:.1%}')} "
        "of native samples and changes at "
        f"{_f((tel.get('dregon_room1') or {}).get('quantisation', {}).get('update_rate_hz'))} Hz, "
        "so its spectrum above that rate is its own staircase, not rotor dynamics. "
        "The same check is reported for every rig in `shaft.json` "
        "(`quantisation.held_fraction`, `quantisation.update_rate_hz`)."
    )
    A(
        "- **Quantisation.** `D_q` is the diffusion a white rounding error of the measured "
        "ladder step would produce if it were renewed at the observed update rate: "
        "`D_q = (2 pi step)^2 / (24 f_update)`. Where `D_q/D_theta` approaches 1 the "
        "diffusive tail of `S(tau)` is an artefact of the telemetry channel, not shaft physics. "
        "Both `D_q` variants (logging rate and update rate) are in `shaft.json`."
    )
    A(
        f"- **High-pass choice.** The slow trend is removed with a zero-phase order-{HP_ORDER} "
        f"Butterworth high-pass at {HP_HZ[0]:g} Hz (headline) and {HP_HZ[1]:g} Hz (reported "
        "alongside). A high-pass at `f_hp` makes `theta` stationary, so `S(tau)` SATURATES "
        "above `tau ~ 1/(2 pi f_hp)` instead of growing; the fitted OU curve drawn on the "
        "same figure passes through the same filter, which is why data and model can be "
        "compared at all. The PSD fit multiplies both models by the known `|H|^4` power "
        "response, so the corner bin is not read as a real roll-off."
    )
    A(
        "- **Acoustic bandwidth.** One harmonic cannot be separated from its neighbours "
        "faster than one shaft revolution, so the acoustic lag grid starts at the frame "
        "rate of a window of "
        f"{DEMOD_REVS:g} revolutions, not at 1 ms; the requested 1 ms lag is unreachable "
        "with any harmonic-isolating demodulation and the script reports the window and "
        "frame rate it actually used per support (`window_s`, `frame_rate_hz`)."
    )
    A(
        "- **No acoustic trend removal, and none possible.** A constant rate error is "
        "INVISIBLE to this estimator: it multiplies the baseband by "
        "`exp(-i 2 pi k d t)`, so the lagged product gains only the t-independent factor "
        "`exp(-i 2 pi k d tau)` and `|sum_t z(t+tau) z*(t)|` does not move. So no rate "
        "refinement is applied, and the acoustic estimate carries NO analogue of the "
        "telemetry high-pass: a slow drift of the mean speed is measured, not removed. "
        "The planted control in `shaft.json` (`planted_control`) confirms the invariance: "
        "a coherence-maximising rate search returned -0.28 rev/s on an exactly known rate "
        "and corrupted every `V_k`, which is why it was removed."
    )
    A(
        "- **Estimator ceiling.** The coherence estimator can only resolve `V` while the "
        "debiased squared coherence stands clear of its own MEASURED floor, which is read "
        "off the half-order flank band (same window, frames and noise, no line). Cells "
        "past the ceiling are reported as `null`, never as a saturated value; `g2_floor` "
        "and `g2_se` in `shaft.json` give the floor and the error per cell."
    )
    A(
        "- **Finite SNR.** Additive noise costs the baseband coherence a lag-INDEPENDENT "
        "`2 log(1 + 1/SNR)`, so every joint fit carries a free non-negative offset `c_k` "
        "per order; without it a weak order inflates `D_k`."
    )
    path = out_dir / "findings.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════


def resolve_rigs(spec: list[str]) -> list[str]:
    out: list[str] = []
    for item in spec:
        for piece in str(item).split(","):
            piece = piece.strip()
            if not piece:
                continue
            out.extend(ALIASES.get(piece, (piece,)))
    seen: dict[str, None] = {}
    for r in out:
        seen.setdefault(r, None)
    return list(seen)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rigs", nargs="+", default=["all"], help="rig/support names or aliases")
    ap.add_argument("--limit", type=int, default=None, help="max flights/recordings per rig")
    ap.add_argument("--out", type=Path, default=ROOT / "results/noise_v2/shaft")
    ap.add_argument("--figdir", type=Path, default=ROOT / "docs/explainers/noise-model-v2-plan")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-control", action="store_true", help="skip the planted acoustic control")
    args = ap.parse_args(argv)

    wanted = resolve_rigs(args.rigs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.figdir)

    t0 = time.time()
    res: dict[str, Any] = {
        "config": {
            "hp_hz": list(HP_HZ),
            "hp_order": HP_ORDER,
            "zero_phase_high_pass": True,
            "structure_lags": {"n": S_N_LAGS, "lo": "1/fs", "hi_s": S_LAG_MAX_S},
            "psd_band": {"lo_hz": PSD_F_LO_HZ, "hi_frac_fs": PSD_F_HI_FRAC},
            "psd_seg_s": PSD_SEG_S,
            "u_slope_1p5": U_SLOPE_1P5,
            "acoustic": {
                "k_max": K_MAX_ACOUSTIC,
                "demod_revolutions": DEMOD_REVS,
                "oversampling": DEMOD_OVERSAMP,
                "margin_min_db": MARGIN_MIN_DB,
                "lag_range_s": [ACOUSTIC_LAG_LO_S, ACOUSTIC_LAG_HI_S],
                "models": list(ACOUSTIC_MODELS),
            },
            "rigs_requested": wanted,
            "limit": args.limit,
        },
        "telemetry": {},
        "acoustics": {},
    }

    for rig, spec in TELEMETRY_RIGS.items():
        if rig not in wanted:
            continue
        print(f"[telemetry] {rig}", flush=True)
        rep = analyse_telemetry_rig(rig, spec, args.limit)
        if rep is None:
            print(f"[telemetry] {rig}: no usable flights", flush=True)
            continue
        rep["verdict"] = telemetry_verdict(rep)
        res["telemetry"][rig] = rep
        print(
            f"[telemetry] {rig}: {rep['verdict']['verdict']}, "
            f"sigma={_f(rep['verdict']['sigma_nu_rad_s'])}, lam={_f(rep['verdict']['lam_1_s'])}",
            flush=True,
        )

    if "dregon_bench" in wanted:
        print("[acoustics] dregon_bench", flush=True)
        blob = run_dregon_bench(args.limit)
        blob["verdict"] = acoustic_verdict(fit_rows(blob))
        res["acoustics"]["dregon_bench"] = blob
        print(f"[acoustics] dregon_bench: {blob['verdict']}", flush=True)
    if "michaels_fly125" in wanted:
        print("[acoustics] michaels_fly125", flush=True)
        blob = run_michaels_acoustic(args.limit)
        blob["verdict"] = acoustic_verdict(fit_rows(blob))
        res["acoustics"]["michaels_fly125"] = blob
        print(f"[acoustics] michaels_fly125: {blob['verdict']}", flush=True)

    if not args.no_control and any(r in wanted for r in ACOUSTIC_SUPPORTS):
        print("[control] planted acoustic rotor", flush=True)
        pc = planted_control()
        res["planted_control"] = pc
        print(f"[control] {pc['recovered']}", flush=True)
    res["comparison"] = build_comparison(res)
    res["runtime_s"] = time.time() - t0

    (out_dir / "shaft.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(f"wrote {out_dir / 'shaft.json'}", flush=True)

    if not args.no_figures:
        for name, fn in (
            ("shaft_structure_functions.png", fig_structure_functions),
            ("shaft_speed_psd.png", fig_speed_psd),
            ("shaft_acoustic_Vk.png", fig_acoustic_Vk),
            ("shaft_Dk_vs_k.png", fig_Dk_vs_k),
            ("shaft_summary_table.png", fig_summary_table),
        ):
            fn(res, [out_dir / name, fig_dir / name])
            print(f"wrote {fig_dir / name}", flush=True)

    print(f"wrote {write_findings(res, out_dir)}", flush=True)
    print(f"done in {res['runtime_s']:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
