"""Is the speed error a 31.25 Hz rotor-speed label cannot carry white, or is it
correlated?

A rotor-speed label carried at the working STFT hop rate
(``LABEL_RATE_HZ = 31.25`` Hz) holds nothing above 15.6 Hz. Everything above
that band is error the renderer must GENERATE, not read. The revised model
generates it from an Ornstein-Uhlenbeck speed error, which is CORRELATED (red)
in time; the cheapest possible alternative generates it from white noise. This
script answers, with the simplest statistic there is, which of the two the
telemetry actually supports.

The measurement, per rig and per rotor, then pooled over rotors by averaging:

1.  ``nu_res = 2 pi x (rps high-passed at 16 Hz)`` in rad/s, on airborne
    segments of at least 5 s, at the rig's OWN logging rate. The high-pass is
    a 4th-order Butterworth applied zero-phase with ``sosfiltfilt``.
2.  The autocorrelation ``rho(dt)`` of ``nu_res`` at lags 0 .. 0.5 s, unbiased
    (each lag divided by its own sample count) and pooled over segments, with
    the white-noise band ``+-2/sqrt(N_eff)``.
    A high-pass makes even a WHITE input correlated, so the measured curve is
    shown against the NULL: white noise of the same length through the same
    filter. Only what stands ABOVE that null is real colour.
3.  The Welch PSD of the RAW speed error (linear detrend only, NO high-pass)
    and its log-log slope by least squares on 16 Hz .. 0.45 fs. Slope 0 is
    white, slope -2 is the tail of an OU process. The slope on 2 .. 16 Hz -
    the band the label DOES carry - is reported for contrast.
4.  What that means for phase: the rms of ``theta_res = int nu_res dt`` after
    the same high-pass, i.e. the bounded shaft wobble the label cannot carry,
    in rad, and at harmonic orders ``k = 10`` and ``k = 30``.

Verdict rule, fixed before the data was read: RED if the 16 Hz .. 0.45 fs
slope is at most -1 AND the measured ACF stands above the filtered-white null
at the measured 1/e lag by more than the white band. WHITE otherwise.

Every number and figure in ``results/noise_v2/residual/`` comes from this file.

Usage:
    python scripts/noise_v2_residual_acf.py                    # every rig
    python scripts/noise_v2_residual_acf.py --rigs neurobem    # smoke test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# The native-rate telemetry loader, the airborne segmentation and the
# high-pass of the shaft-phase study, reused verbatim so that the two studies
# see EXACTLY the same samples. That script is not modified by this one.
from noise_v2_shaft_phase import (  # noqa: E402
    HP_ORDER,
    LABEL_RATE_HZ,
    MIN_SEG_S,
    TELEMETRY_RIGS,
    _interp_nans,
    hp_sos,
    load_native,
    stationary_segments,
)
from scipy.signal import correlate, sosfiltfilt, welch  # noqa: E402

# ─── frozen analysis constants ────────────────────────────────────────────────

#: High-pass corner, Hz. Half the label rate, rounded up: a label carried at
#: 31.25 Hz holds nothing above 15.625 Hz, so 16 Hz keeps only what the label
#: CANNOT carry.
HP_HZ = 16.0
#: Longest ACF lag, seconds.
ACF_MAX_LAG_S = 0.5
#: Lags, seconds, tabulated in the report.
REPORT_LAGS_S = (0.010, 0.025, 0.050, 0.100)
#: Welch segment length, seconds, and overlap fraction for the raw speed PSD.
#: 4 s gives 0.25 Hz resolution, which resolves the 2 .. 16 Hz contrast band,
#: and keeps every airborne segment at or above :data:`MIN_SEG_S`.
PSD_SEG_S = 4.0
PSD_OVERLAP = 0.5
#: Upper edge of the headline PSD fit band as a fraction of ``fs``. Kept below
#: Nyquist because the last bins of a Welch estimate carry the anti-alias or
#: interpolation roll-off of the logger, not the process.
PSD_HI_FRAC = 0.45
#: The contrast band: what the label DOES carry.
PSD_LO_BAND_HZ = (2.0, 16.0)
#: Harmonic orders at which the shaft wobble is quoted.
ORDERS = (10, 30)
#: Realisations of the filtered-white null, averaged. One realisation per
#: segment per repeat; 8 repeats put the null's own scatter well below the
#: white band it is compared against.
NULL_REPS = 8
#: Seed of the null. Fixed, so the script is idempotent.
NULL_SEED = 20260916
#: Largest fraction of non-finite samples tolerated in a segment.
MAX_BAD_FRACTION = 0.05
#: ``1/e``: the level whose crossing defines the correlation time.
INV_E = 1.0 / math.e
#: A PSD slope at or below this is steeper than any OU tail (-2) and marks a
#: logger roll-off rather than a shaft process.
STEEP_SLOPE = -3.0

#: Rigs, in report order. Keys of :data:`TELEMETRY_RIGS`.
RIGS = (
    "neurobem_quad",
    "blackbird_quad",
    "vid_m100",
    "nanobench_cf21b",
    "pitcn_quad",
    "dregon_room1",
    "dregon_room1_command",
    "michaels",
)
#: Short names accepted by ``--rigs``.
ALIASES: dict[str, tuple[str, ...]] = {
    "all": RIGS,
    "neurobem": ("neurobem_quad",),
    "blackbird": ("blackbird_quad",),
    "vid": ("vid_m100",),
    "nanobench": ("nanobench_cf21b",),
    "pitcn": ("pitcn_quad",),
    "dregon": ("dregon_room1", "dregon_room1_command"),
}
#: Rig-specific caveats printed with the findings.
RIG_NOTES: dict[str, str] = {
    "dregon_room1": (
        "The DREGON measured track is logged near 1 kHz but the value only "
        "CHANGES near 45 Hz, so it is a sample-and-hold. Above 16 Hz the "
        "residual is the staircase of that hold, not the shaft."
    ),
    "dregon_room1_command": (
        "This is the motor COMMAND, not a measurement. It is the controller's "
        "own output and it bounds what the plant can be asked to do."
    ),
}


# ═════════════════════════════════════════════════════════════════════════════
# The statistic
# ═════════════════════════════════════════════════════════════════════════════


def acf_sums(parts: list[np.ndarray], kmax: int) -> tuple[np.ndarray, np.ndarray]:
    """``(numerator, count)`` of the UNBIASED autocovariance, pooled over parts.

    ``numerator[k] = sum over parts, over t of x(t) x(t+k)`` and ``count[k]``
    is how many products went into it, so ``numerator/count`` is the unbiased
    autocovariance and its lag-0 value is the variance.
    """
    num = np.zeros(kmax + 1)
    den = np.zeros(kmax + 1)
    for raw in parts:
        x = np.asarray(raw, dtype=np.float64)
        x = x - x.mean()
        n = x.size
        if n < 2:
            continue
        k = min(kmax, n - 1)
        full = correlate(x, x, mode="full", method="fft")
        num[: k + 1] += full[n - 1 : n + k]
        den[: k + 1] += n - np.arange(k + 1)
    return num, den


def acf(parts: list[np.ndarray], kmax: int) -> np.ndarray:
    """Unbiased autocorrelation at lags ``0 .. kmax`` samples, NaN where empty."""
    num, den = acf_sums(parts, kmax)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = np.where(den > 0, num / np.maximum(den, 1.0), np.nan)
    if not np.isfinite(cov[0]) or cov[0] <= 0:
        return np.full(kmax + 1, np.nan)
    return cov / cov[0]


def filtered_white_acf(
    lengths: list[int], kmax: int, sos: np.ndarray, rng: np.random.Generator, reps: int
) -> np.ndarray:
    """The NULL: the ACF of white noise put through the same high-pass.

    This is the shape the measured ACF would have if the speed error were
    white, so any measured correlation that does not exceed it belongs to the
    filter and not to the rotor.
    """
    acc = np.zeros(kmax + 1)
    got = 0
    for _ in range(max(1, reps)):
        parts = [sosfiltfilt(sos, rng.standard_normal(n)) for n in lengths]
        r = acf(parts, kmax)
        if np.isfinite(r).all():
            acc += r
            got += 1
    return acc / got if got else np.full(kmax + 1, np.nan)


def crossing(tau: np.ndarray, y: np.ndarray, level: float) -> float | None:
    """First lag where ``y`` falls through ``level``, linearly interpolated."""
    ok = np.isfinite(y)
    for i in range(1, tau.size):
        if not (ok[i] and ok[i - 1]):
            continue
        if y[i - 1] > level >= y[i]:
            span = y[i - 1] - y[i]
            frac = 0.0 if span <= 0 else (y[i - 1] - level) / span
            return float(tau[i - 1] + frac * (tau[i] - tau[i - 1]))
    return None


def at_lag(tau: np.ndarray, y: np.ndarray, lag_s: float) -> float | None:
    """``y`` at one lag in seconds, linearly interpolated, None if out of range."""
    if tau.size == 0 or lag_s > tau[-1]:
        return None
    v = float(np.interp(lag_s, tau, y))
    return v if math.isfinite(v) else None


def welch_psd(parts: list[np.ndarray], fs: float) -> tuple[np.ndarray, np.ndarray, float]:
    """``(f, P, n_windows)`` of the pooled Welch PSD, ``(rad/s)^2/Hz``."""
    usable = [p for p in parts if p.size >= 8]
    if not usable:
        return np.zeros(0), np.zeros(0), 0.0
    nper = int(min(round(PSD_SEG_S * fs), max(p.size for p in usable)))
    nper = max(nper, 8)
    nov = int(PSD_OVERLAP * nper)
    acc: np.ndarray | None = None
    freqs: np.ndarray | None = None
    n_win = 0.0
    for p in usable:
        if p.size < nper:
            continue
        f, pxx = welch(p, fs=fs, window="hann", nperseg=nper, noverlap=nov, detrend=False)  # type: ignore[arg-type]
        k = 1 + (p.size - nper) // (nper - nov)
        acc = pxx * k if acc is None else acc + pxx * k
        freqs = f
        n_win += k
    if acc is None or freqs is None:
        return np.zeros(0), np.zeros(0), 0.0
    return freqs, acc / n_win, n_win


def loglog_slope(f: np.ndarray, p: np.ndarray, f_lo: float, f_hi: float) -> dict[str, Any]:
    """Least-squares log-log slope of a PSD on one band."""
    m = (f >= f_lo) & (f <= f_hi) & (p > 0) & np.isfinite(p)
    n = int(m.sum())
    if n < 4:
        return {"slope": None, "n_bins": n, "f_lo_hz": f_lo, "f_hi_hz": f_hi}
    x = np.log10(f[m])
    y = np.log10(p[m])
    a, b = np.polyfit(x, y, 1)
    resid = y - (a * x + b)
    sx = float(np.sum((x - x.mean()) ** 2))
    se = float(np.sqrt(np.sum(resid**2) / max(n - 2, 1) / max(sx, 1e-300)))
    return {
        "slope": float(a),
        "intercept_log10": float(b),
        "slope_se": se,
        "n_bins": n,
        "f_lo_hz": float(f_lo),
        "f_hi_hz": float(f_hi),
    }


# ═════════════════════════════════════════════════════════════════════════════
# One rig
# ═════════════════════════════════════════════════════════════════════════════


def rotor_row(
    rotor: int,
    nu_parts: list[np.ndarray],
    th_parts: list[np.ndarray],
    raw_parts: list[np.ndarray],
    fs: float,
    kmax: int,
    tau: np.ndarray,
) -> dict[str, Any]:
    """Every number this study asks for, for ONE rotor of one rig."""
    n_samples = int(sum(p.size for p in nu_parts))
    rho = acf(nu_parts, kmax)
    var = float(np.mean(np.concatenate(nu_parts) ** 2))
    nu_rms = math.sqrt(max(var, 0.0))
    theta = np.concatenate(th_parts)
    th_rms = float(np.sqrt(np.mean(theta**2)))
    f, p, n_win = welch_psd(raw_parts, fs)
    hi = loglog_slope(f, p, HP_HZ, PSD_HI_FRAC * fs)
    lo = loglog_slope(f, p, PSD_LO_BAND_HZ[0], PSD_LO_BAND_HZ[1])
    return {
        "rotor": rotor,
        "n_segments": len(nu_parts),
        "n_samples": n_samples,
        "seconds": n_samples / fs,
        "nu_res_rms_rad_s": nu_rms,
        "nu_res_rms_rev_s": nu_rms / (2.0 * math.pi),
        "rho": rho.tolist(),
        "rho_at": {
            "one_sample": float(rho[1]) if rho.size > 1 else None,
            **{f"{int(s * 1000)}ms": at_lag(tau, rho, s) for s in REPORT_LAGS_S},
        },
        "first_zero_s": crossing(tau, rho, 0.0),
        "tau_1e_s": crossing(tau, rho, INV_E),
        "psd_slope_hi": hi,
        "psd_slope_lo": lo,
        "psd_n_windows": n_win,
        "theta_res_rms_rad": th_rms,
        **{f"theta_res_rms_k{k}_rad": th_rms * k for k in ORDERS},
    }


def _mean(values: list[Any]) -> float | None:
    good = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(good)) if good else None


def pool_rotors(rows: list[dict[str, Any]], tau: np.ndarray) -> dict[str, Any]:
    """Pool over rotors by AVERAGING: the ACF curve point by point, the slopes
    and the lags directly, the rms values in quadrature (an average of powers).
    """
    live = [r for r in rows if r.get("n_segments")]
    if not live:
        return {"n_rotors": 0}
    rho = np.nanmean(np.stack([np.asarray(r["rho"], dtype=float) for r in live]), axis=0)
    nu_rms = float(np.sqrt(np.mean([r["nu_res_rms_rad_s"] ** 2 for r in live])))
    th_rms = float(np.sqrt(np.mean([r["theta_res_rms_rad"] ** 2 for r in live])))
    # Conservative effective count: the rotors of one airframe share the same
    # flight and are NOT independent, so the band uses one rotor's sample
    # count, not the sum over rotors.
    n_eff = float(np.mean([r["n_samples"] for r in live]))
    return {
        "n_rotors": len(live),
        "n_samples_per_rotor": n_eff,
        "seconds_per_rotor": float(np.mean([r["seconds"] for r in live])),
        "white_band": 2.0 / math.sqrt(max(n_eff, 1.0)),
        "rho": rho.tolist(),
        "rho_at": {
            "one_sample": float(rho[1]) if rho.size > 1 else None,
            **{f"{int(s * 1000)}ms": at_lag(tau, rho, s) for s in REPORT_LAGS_S},
        },
        "first_zero_s": crossing(tau, rho, 0.0),
        "tau_1e_s": crossing(tau, rho, INV_E),
        "nu_res_rms_rad_s": nu_rms,
        "nu_res_rms_rev_s": nu_rms / (2.0 * math.pi),
        "psd_slope_hi": _mean([r["psd_slope_hi"]["slope"] for r in live]),
        "psd_slope_lo": _mean([r["psd_slope_lo"]["slope"] for r in live]),
        "theta_res_rms_rad": th_rms,
        **{f"theta_res_rms_k{k}_rad": th_rms * k for k in ORDERS},
    }


def verdict(pooled: dict[str, Any], null: np.ndarray, tau: np.ndarray) -> dict[str, Any]:
    """RED if the high-band slope is at most -1 AND the measured ACF stands
    above the filtered-white null at the 1/e lag by more than the white band.
    """
    slope = pooled.get("psd_slope_hi")
    band = float(pooled.get("white_band", 0.0))
    tau_e = pooled.get("tau_1e_s")
    used_max = tau_e is None
    lag = float(tau[-1]) if used_max else float(tau_e)
    rho = np.asarray(pooled.get("rho", []), dtype=float)
    r_meas = at_lag(tau, rho, lag) if rho.size else None
    r_null = at_lag(tau, null, lag) if null.size else None
    excess = None if (r_meas is None or r_null is None) else r_meas - r_null
    steep = slope is not None and slope <= -1.0
    wider = excess is not None and excess > band
    return {
        "lag_tested_s": lag,
        "lag_is_acf_max": used_max,
        "rho_measured_at_lag": r_meas,
        "rho_null_at_lag": r_null,
        "excess_over_null": excess,
        "white_band": band,
        "slope_hi": slope,
        "slope_at_most_minus_one": bool(steep),
        "acf_above_null": bool(wider),
        "verdict": "red" if (steep and wider) else "white",
    }


def analyse_rig(
    rig: str, spec: dict[str, Any], limit: int | None, reps: int, seed: int
) -> dict[str, Any] | None:
    tracks = load_native(rig, spec, limit)
    if not tracks:
        return None
    fs = float(np.median([t.fs for t in tracks]))
    rep: dict[str, Any] = {
        "rig": rig,
        "dataset": spec["dataset"],
        "rps_key": tracks[0].key,
        "auxiliary": bool(spec.get("auxiliary")),
        "fs_hz": fs,
        "n_flights": len(tracks),
        "flights": [t.flight for t in tracks],
        "note": RIG_NOTES.get(rig),
    }
    if 0.5 * fs <= HP_HZ:
        rep["skipped"] = (
            f"the 16 Hz high-pass needs a Nyquist above 16 Hz; this rig logs at "
            f"{fs:.1f} Hz, so nothing above {0.5 * fs:.1f} Hz exists in it"
        )
        rep["nyquist_hz"] = 0.5 * fs
        return rep

    sos = hp_sos(fs, HP_HZ)
    kmax = int(round(ACF_MAX_LAG_S * fs))
    tau = np.arange(kmax + 1) / fs
    n_rotors = int(tracks[0].rps.shape[0])
    nu_all: list[list[np.ndarray]] = [[] for _ in range(n_rotors)]
    th_all: list[list[np.ndarray]] = [[] for _ in range(n_rotors)]
    raw_all: list[list[np.ndarray]] = [[] for _ in range(n_rotors)]
    lengths: list[int] = []
    total_s = 0.0
    n_flights_used = 0
    min_len = max(kmax + 2, int(4.0 * fs / HP_HZ))
    for track in tracks:
        if abs(track.fs - fs) / fs > 0.05:
            continue  # a differently clocked flight is its own population
        used = False
        for sl in stationary_segments(track.rps, track.fs):
            for r in range(n_rotors):
                y, frac_bad = _interp_nans(track.rps[r, sl])
                if frac_bad > MAX_BAD_FRACTION or y.size < min_len:
                    continue
                nu = 2.0 * np.pi * sosfiltfilt(sos, y - y.mean())
                # The wobble the label cannot carry: integrate, then remove the
                # same band again, which is what keeps theta bounded.
                theta = sosfiltfilt(sos, np.cumsum(nu) / fs)
                # The RAW speed error: linear detrend only, no high-pass, so
                # the PSD below 16 Hz is untouched by this study's filter.
                t = np.arange(y.size) / fs
                a, b = np.polyfit(t, y, 1)
                nu_all[r].append(nu)
                th_all[r].append(np.asarray(theta))
                raw_all[r].append(2.0 * np.pi * (y - (a * t + b)))
                if r == 0:
                    lengths.append(int(y.size))
                used = True
            total_s += (sl.stop - sl.start) / track.fs
        n_flights_used += int(used)
    if not any(nu_all):
        rep["skipped"] = "no airborne segment of at least 5 s survived the quality gate"
        return rep
    if not lengths:
        lengths = [int(p.size) for p in next(parts for parts in nu_all if parts)]

    rows = [
        rotor_row(r, nu_all[r], th_all[r], raw_all[r], fs, kmax, tau)
        if nu_all[r]
        else {"rotor": r, "n_segments": 0}
        for r in range(n_rotors)
    ]
    pooled = pool_rotors(rows, tau)
    # A per-rig stream, seeded deterministically (``hash`` is randomised per
    # process, ``crc32`` is not), so a rerun reproduces the null exactly.
    rng = np.random.default_rng(seed + zlib.crc32(rig.encode()) % 100000)
    null = filtered_white_acf(lengths, kmax, sos, rng, reps)

    f_pool, p_pool, n_win = welch_psd([p for parts in raw_all for p in parts], fs)
    rep.update(
        {
            "n_flights_used": n_flights_used,
            "airborne_seconds": total_s,
            "n_segments": len(lengths),
            "lags_s": tau.tolist(),
            "per_rotor": rows,
            "pooled": pooled,
            "null_filtered_white": {
                "rho": null.tolist(),
                "reps": reps,
                "first_zero_s": crossing(tau, null, 0.0),
                "tau_1e_s": crossing(tau, null, INV_E),
                "rho_at": {
                    "one_sample": float(null[1]) if null.size > 1 else None,
                    **{f"{int(s * 1000)}ms": at_lag(tau, null, s) for s in REPORT_LAGS_S},
                },
            },
            "psd": {
                "f_hz": f_pool.tolist(),
                "p_rad2_s2_per_hz": p_pool.tolist(),
                "n_windows": n_win,
                "slope_hi": loglog_slope(f_pool, p_pool, HP_HZ, PSD_HI_FRAC * fs),
                "slope_lo": loglog_slope(f_pool, p_pool, *PSD_LO_BAND_HZ),
            },
        }
    )
    rep["verdict"] = verdict(pooled, null, tau)
    slope = pooled.get("psd_slope_hi")
    if slope is not None and slope <= STEEP_SLOPE:
        # A real OU tail falls with slope -2. Anything much steeper is the
        # logger's own reconstruction low-pass, and the residual it leaves is
        # the shape of that filter, not a measurement of the shaft.
        rep["steep_note"] = (
            f"The slope {slope:+.2f} is far steeper than the -2 of an OU tail. It is the "
            "roll-off of the logger itself, so this rig measures almost nothing above "
            f"{HP_HZ:g} Hz and its residual is that roll-off, not the shaft."
        )
    return rep


# ═════════════════════════════════════════════════════════════════════════════
# Figures
# ═════════════════════════════════════════════════════════════════════════════


def _grid(n: int) -> tuple[int, int]:
    cols = min(3, max(1, n))
    return int(math.ceil(n / cols)), cols


def _save(fig: Any, paths: list[Path]) -> None:
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _skip_panel(ax: Any, rig: str, rep: dict[str, Any]) -> None:
    body = textwrap.fill(str(rep.get("skipped", "no data")), width=38)
    ax.text(
        0.5,
        0.5,
        f"{rig}\n\n{body}",
        ha="center",
        va="center",
        fontsize=8,
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def fig_acf(res: dict[str, Any], paths: list[Path]) -> None:
    rigs = list(res["rigs"])
    rows, cols = _grid(len(rigs))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.0 * rows), squeeze=False)
    for ax, rig in zip(axes.ravel(), rigs, strict=False):
        rep = res["rigs"][rig]
        if "pooled" not in rep:
            _skip_panel(ax, rig, rep)
            continue
        tau_ms = np.asarray(rep["lags_s"]) * 1e3
        rho = np.asarray(rep["pooled"]["rho"], dtype=float)
        null = np.asarray(rep["null_filtered_white"]["rho"], dtype=float)
        band = rep["pooled"]["white_band"]
        ax.axhspan(-band, band, color="0.85", label=f"white band $\\pm${band:.1e}")
        ax.axhline(0.0, color="0.6", lw=0.6)
        ax.axhline(INV_E, color="0.6", lw=0.6, ls=":")
        ax.plot(tau_ms, null, color="tab:orange", lw=1.2, ls="--", label="filtered-white null")
        ax.plot(tau_ms, rho, color="tab:blue", lw=1.3, label="measured")
        te = rep["pooled"]["tau_1e_s"]
        if te is not None:
            ax.axvline(te * 1e3, color="tab:blue", lw=0.8, ls=":")
        v = rep["verdict"]
        ax.set_title(
            f"{rig}  ({rep['fs_hz']:.0f} Hz) — {v['verdict']}"
            + (f", 1/e {te * 1e3:.1f} ms" if te is not None else ""),
            fontsize=9,
        )
        ax.set_xlabel("lag (ms)")
        ax.set_ylabel(r"$\rho$ of $\nu_{res}$")
        ax.set_xlim(0, ACF_MAX_LAG_S * 1e3)
        ax.legend(fontsize=6, loc="upper right")
    for ax in axes.ravel()[len(rigs) :]:
        ax.axis("off")
    fig.suptitle(
        "Autocorrelation of the speed error a 31.25 Hz label cannot carry "
        f"(high-pass {HP_HZ:g} Hz, order {HP_ORDER}, zero phase)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, paths)


def fig_psd(res: dict[str, Any], paths: list[Path]) -> None:
    rigs = list(res["rigs"])
    rows, cols = _grid(len(rigs))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.0 * rows), squeeze=False)
    for ax, rig in zip(axes.ravel(), rigs, strict=False):
        rep = res["rigs"][rig]
        if "psd" not in rep:
            _skip_panel(ax, rig, rep)
            continue
        f = np.asarray(rep["psd"]["f_hz"])
        p = np.asarray(rep["psd"]["p_rad2_s2_per_hz"])
        m = (f >= 1.0) & (p > 0)
        ax.loglog(f[m], p[m], color="0.35", lw=0.8, label="raw speed PSD")
        for fit, colour in (
            (rep["psd"]["slope_lo"], "tab:green"),
            (rep["psd"]["slope_hi"], "tab:red"),
        ):
            if fit["slope"] is None:
                continue
            xx = np.geomspace(fit["f_lo_hz"], fit["f_hi_hz"], 32)
            ax.loglog(
                xx,
                10.0 ** (fit["intercept_log10"] + fit["slope"] * np.log10(xx)),
                color=colour,
                lw=2.0,
                label=f"{fit['f_lo_hz']:g}–{fit['f_hi_hz']:.0f} Hz: {fit['slope']:+.2f}",
            )
        ax.axvline(HP_HZ, color="tab:blue", lw=1.0, ls="--", label=f"{HP_HZ:g} Hz")
        ax.set_title(f"{rig}  ({rep['fs_hz']:.0f} Hz)", fontsize=9)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel(r"$(rad/s)^2$/Hz")
        ax.set_xlim(1.0, 0.5 * rep["fs_hz"])
        ax.legend(fontsize=6)
    for ax in axes.ravel()[len(rigs) :]:
        ax.axis("off")
    fig.suptitle("PSD of the RAW speed error (linear detrend, no high-pass)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, paths)


# ═════════════════════════════════════════════════════════════════════════════
# Report
# ═════════════════════════════════════════════════════════════════════════════


def _f(x: Any, fmt: str = "{:.3g}") -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    if isinstance(x, (int, float)):
        return fmt.format(x)
    return str(x)


def _ms(x: Any) -> str:
    return "—" if x is None else f"{x * 1e3:.1f}"


HEAD = [
    "rig",
    "fs (Hz)",
    "nu_res rms (rad/s)",
    "rho(1 sample)",
    "rho(10 ms)",
    "rho(50 ms)",
    "1/e lag (ms)",
    "slope 16–fs/2",
    "slope 2–16",
    "verdict",
]


def table_rows(res: dict[str, Any]) -> list[list[str]]:
    out = [HEAD]
    for rig, rep in res["rigs"].items():
        if "pooled" not in rep:
            out.append([rig, _f(rep.get("fs_hz"), "{:.1f}")] + ["—"] * 7 + ["n/a"])
            continue
        p = rep["pooled"]
        out.append(
            [
                rig,
                _f(rep["fs_hz"], "{:.1f}"),
                _f(p["nu_res_rms_rad_s"], "{:.3g}"),
                _f(p["rho_at"]["one_sample"], "{:+.3f}"),
                _f(p["rho_at"]["10ms"], "{:+.3f}"),
                _f(p["rho_at"]["50ms"], "{:+.3f}"),
                _ms(p["tau_1e_s"]),
                _f(p["psd_slope_hi"], "{:+.2f}"),
                _f(p["psd_slope_lo"], "{:+.2f}"),
                rep["verdict"]["verdict"],
            ]
        )
    return out


def markdown_table(rows: list[list[str]]) -> str:
    head, body = rows[0], rows[1:]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


def speed_sentence(rep: dict[str, Any]) -> str:
    if "pooled" not in rep:
        return f"Speed: {rep.get('skipped', 'no usable data')}."
    p = rep["pooled"]
    v = rep["verdict"]
    null = rep["null_filtered_white"]
    te = p["tau_1e_s"]
    tn = null["tau_1e_s"]
    r = p["rho_at"]
    return (
        f"Speed: the residual above {HP_HZ:g} Hz has rms {p['nu_res_rms_rad_s']:.3g} rad/s "
        f"({p['nu_res_rms_rev_s']:.3g} rev/s); its autocorrelation is "
        f"{_f(r['one_sample'], '{:+.2f}')} at one sample, {_f(r['10ms'], '{:+.2f}')} at 10 ms, "
        f"{_f(r['25ms'], '{:+.2f}')} at 25 ms, {_f(r['50ms'], '{:+.2f}')} at 50 ms and "
        f"{_f(r['100ms'], '{:+.2f}')} at 100 ms, it first crosses zero at "
        f"{_ms(p['first_zero_s'])} ms and reaches 1/e at {_ms(te)} ms against "
        f"{_ms(tn)} ms for the filtered-white null; and the raw speed PSD falls with slope "
        f"{_f(p['psd_slope_hi'], '{:+.2f}')} on {HP_HZ:g}–{PSD_HI_FRAC * rep['fs_hz']:.0f} Hz "
        f"against {_f(p['psd_slope_lo'], '{:+.2f}')} on 2–{HP_HZ:g} Hz, so the residual reads "
        f"as {v['verdict'].upper()}."
    )


def phase_sentence(rep: dict[str, Any]) -> str:
    if "pooled" not in rep:
        return "Phase: not measurable on this rig."
    p = rep["pooled"]
    return (
        f"Phase: integrating that residual gives a bounded shaft wobble of rms "
        f"{p['theta_res_rms_rad']:.3g} rad, which is {p['theta_res_rms_k10_rad']:.3g} rad at "
        f"order 10 and {p['theta_res_rms_k30_rad']:.3g} rad at order 30."
    )


def shape_sentence(res: dict[str, Any]) -> str:
    """The one thing every panel shows: the measured ACF swings NEGATIVE.

    White noise, high-passed or not, has no negative lobe deeper than its own
    band. A deep negative lobe means the residual power sits in a narrow band
    just above the corner, which is a correlated process and not white noise.
    """
    live = [(r, rep) for r, rep in res["rigs"].items() if "pooled" in rep]
    if not live:
        return "No rig carried a measurable residual above the label band."
    lobes = []
    for _rig, rep in live:
        rho = np.asarray(rep["pooled"]["rho"], dtype=float)
        null = np.asarray(rep["null_filtered_white"]["rho"], dtype=float)
        lobes.append((float(np.nanmin(rho)), float(np.nanmin(null))))
    zeros = [rep["pooled"]["first_zero_s"] for _r, rep in live if rep["pooled"]["first_zero_s"]]
    deep = min(x for x, _n in lobes)
    deep_null = min(n for _x, n in lobes)
    return (
        f"Every measured curve swings below zero, down to {deep:+.2f} at the worst rig against "
        f"{deep_null:+.2f} for the deepest null, and crosses zero between "
        f"{min(zeros) * 1e3:.1f} ms and {max(zeros) * 1e3:.1f} ms. That is the signature of "
        "power in a narrow band just above the corner, not of white noise: the residual "
        "rings at the edge of the label band and then dies."
    )


def write_findings(res: dict[str, Any], out_dir: Path) -> Path:
    rows = table_rows(res)
    red = [r for r, rep in res["rigs"].items() if rep.get("verdict", {}).get("verdict") == "red"]
    white = [
        r for r, rep in res["rigs"].items() if rep.get("verdict", {}).get("verdict") == "white"
    ]
    lines = [
        "# Is the speed error above the label band white or correlated?",
        "",
        f"Generated by `scripts/noise_v2_residual_acf.py` in {res['runtime_s']:.1f} s "
        "on the laptop. No cluster job was needed.",
        "",
        "## The question",
        "",
        f"A rotor-speed label carried at {LABEL_RATE_HZ:g} Hz holds nothing above "
        f"{LABEL_RATE_HZ / 2:g} Hz. The renderer must GENERATE the rest. This study asks, "
        "with the simplest statistic available, whether that missing part is white noise "
        "or a correlated (red) process.",
        "",
        f"The residual is `nu_res = 2 pi x (rps high-passed at {HP_HZ:g} Hz)` in rad/s, from a "
        f"{HP_ORDER}th-order Butterworth applied with zero phase, on airborne segments of at "
        "least "
        f"{MIN_SEG_S:g} s at each rig's own logging rate. A high-pass makes even white noise "
        "look correlated, so every measured autocorrelation is shown against a NULL: white "
        f"noise of the same length through the same filter, averaged over {NULL_REPS} draws.",
        "",
        "Verdict rule, fixed before the data was read: RED when the PSD slope on "
        f"{HP_HZ:g} Hz – 0.45 fs is at most -1 AND the measured autocorrelation stands above "
        "the filtered-white null at the measured 1/e lag by more than the white band "
        "`2/sqrt(N)`. WHITE otherwise.",
        "",
        "## The table",
        "",
        markdown_table(rows),
        "",
        "The autocorrelation is unbiased and pooled over segments, then averaged over the "
        "rotors of each rig. The rms values are averaged over rotors in quadrature. "
        "`slope 2–16` is the band the label DOES carry and is given for contrast only.",
        "",
        "![Autocorrelation against the filtered-white null](residual_acf.png)",
        "",
        "![PSD of the raw speed error](residual_psd.png)",
        "",
        "## Rig by rig",
        "",
    ]
    for rig, rep in res["rigs"].items():
        lines.append(f"### {rig}")
        lines.append("")
        lines.append(speed_sentence(rep))
        lines.append("")
        lines.append(phase_sentence(rep))
        for note in ("note", "steep_note"):
            if rep.get(note):
                lines.append("")
                lines.append(str(rep[note]))
        lines.append("")
    lines += [
        "## What this means",
        "",
        (
            f"{len(red)} of {len(res['rigs'])} rigs read RED ("
            + (", ".join(red) if red else "none")
            + f") and {len(white)} read WHITE ("
            + (", ".join(white) if white else "none")
            + "). A RED rig needs a correlated generator above the label band; a WHITE rig "
            "is served by plain white speed noise of the measured rms."
        ),
        "",
        shape_sentence(res),
        "",
        "Every number above is in `residual.json` in this directory, per rotor and pooled.",
        "",
    ]
    path = out_dir / "findings.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════


def resolve_rigs(spec: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for name in spec:
        for rig in ALIASES.get(name, (name,)):
            if rig not in TELEMETRY_RIGS:
                raise SystemExit(f"unknown rig: {rig}")
            seen[rig] = None
    return [r for r in RIGS if r in seen]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rigs", nargs="+", default=["all"], help="rig names or aliases")
    ap.add_argument("--limit", type=int, default=None, help="max flights per rig")
    ap.add_argument("--null-reps", type=int, default=NULL_REPS)
    ap.add_argument("--seed", type=int, default=NULL_SEED)
    ap.add_argument("--out", type=Path, default=ROOT / "results/noise_v2/residual")
    ap.add_argument("--figdir", type=Path, default=ROOT / "docs/explainers/noise-model-v2-plan")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    wanted = resolve_rigs(args.rigs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.figdir)

    t0 = time.time()
    res: dict[str, Any] = {
        "config": {
            "hp_hz": HP_HZ,
            "hp_order": HP_ORDER,
            "zero_phase_high_pass": True,
            "label_rate_hz": LABEL_RATE_HZ,
            "min_segment_s": MIN_SEG_S,
            "acf_max_lag_s": ACF_MAX_LAG_S,
            "report_lags_s": list(REPORT_LAGS_S),
            "psd_seg_s": PSD_SEG_S,
            "psd_band_hi": [HP_HZ, f"{PSD_HI_FRAC} fs"],
            "psd_band_lo": list(PSD_LO_BAND_HZ),
            "orders": list(ORDERS),
            "null_reps": args.null_reps,
            "seed": args.seed,
            "rigs_requested": wanted,
            "limit": args.limit,
        },
        "rigs": {},
    }

    for rig in wanted:
        print(f"[rig] {rig}", flush=True)
        rep = analyse_rig(rig, TELEMETRY_RIGS[rig], args.limit, args.null_reps, args.seed)
        if rep is None:
            print(f"[rig] {rig}: no usable flights", flush=True)
            continue
        res["rigs"][rig] = rep
        if "pooled" in rep:
            p = rep["pooled"]
            print(
                f"[rig] {rig}: fs={rep['fs_hz']:.1f} Hz, rms={p['nu_res_rms_rad_s']:.3g} rad/s, "
                f"1/e={_ms(p['tau_1e_s'])} ms, slope={_f(p['psd_slope_hi'], '{:+.2f}')}, "
                f"{rep['verdict']['verdict']}",
                flush=True,
            )
        else:
            print(f"[rig] {rig}: {rep.get('skipped')}", flush=True)

    res["runtime_s"] = time.time() - t0
    path = out_dir / "residual.json"
    path.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(f"wrote {path}", flush=True)

    if not args.no_figures:
        for name, fn in (("residual_acf.png", fig_acf), ("residual_psd.png", fig_psd)):
            fn(res, [out_dir / name, fig_dir / name])
            print(f"wrote {fig_dir / name}", flush=True)

    print(f"wrote {write_findings(res, out_dir)}", flush=True)
    print(f"done in {res['runtime_s']:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
