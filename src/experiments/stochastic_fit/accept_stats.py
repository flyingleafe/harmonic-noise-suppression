"""The acceptance statistics of the distributional-fit objective, in one place.

Both the gate (``scripts/_stage_accept.py``) and the fit's own level
calibration read these, so a mismatch between what is fitted and what is
scored is impossible by construction. The gate applies them to HELD-OUT data;
the fit applies them to training data only.

Two statistics:

* :func:`ltas_bands` — criterion 1. Seven-band long-term average spectrum, each
  clip referenced to its own 200-400 Hz mean, so only spectral SHAPE enters.
* :func:`comb_excess` — criterion 2. Per-order band-integrated power over a
  local floor, band = +-3x the measured width law, floor debiased from median
  to mean (an averaged periodogram bin is Gamma-distributed, and a band
  integral removes the MEAN, not the median). ``offset=0.5`` reads the
  half-integer comb: the null where no rotor line can exist.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaincinv

from experiments.stochastic_fit import bench

SR = 16000
BANDS = (
    (100, 300),
    (300, 700),
    (700, 1500),
    (1500, 3000),
    (3000, 5000),
    (5000, 7000),
    (7000, 7900),
)
REF_LO, REF_HI = 200.0, 400.0
ORDER_BANDS = ((2, 15), (15, 30), (30, 100))
#: Preregistered thresholds — never relaxed (see the objective).
LTAS_MEAN_MAX_DB = 1.5
LTAS_MAX_MAX_DB = 3.0
ORDER_TOL_DB = 1.5
SPEED_MATCH_REV_S = 5.0
#: Analysis geometry: 65536 points at 16 kHz is a 0.244 Hz bin, which resolves
#: the narrowest bench line (0.28 Hz measured natively).
N_ANALYSIS = 1 << 16
#: Peak margin over the local floor at which an order counts as detected.
MARGIN_MIN_DB = 6.0


def ltas_bands(x: np.ndarray, sr: int = SR, n: int = 8192) -> np.ndarray:
    """Band levels in dB, referenced to the clip's own 200-400 Hz mean."""
    xx = np.asarray(x, dtype=np.float64)
    w = np.hanning(n + 1)[:n]
    frames = np.stack([xx[s : s + n] * w for s in range(0, xx.size - n, n // 2)])
    power = (np.abs(np.fft.rfft(frames, axis=-1)) ** 2).mean(0)
    f = np.fft.rfftfreq(n, 1 / sr)
    db = 10.0 * np.log10(power + 1e-300)
    ref = db[(f >= REF_LO) & (f <= REF_HI)].mean()
    return np.array([db[(f >= lo) & (f < hi)].mean() - ref for lo, hi in BANDS])


def welch_averages(n_samples: int, n_fft: int = N_ANALYSIS) -> int:
    """Number of 50 %-overlapped segments :func:`bench._welch` averages."""
    return len(range(0, max(n_samples - n_fft, 0) + 1, n_fft // 2))


def median_to_mean(n_avg: int) -> float:
    """``mean / median`` of an ``n_avg``-averaged periodogram bin."""
    return float(n_avg / gammaincinv(n_avg, 0.5))


def comb_excess(
    psd: np.ndarray,
    df: float,
    rate: float,
    k_hi: int,
    *,
    gamma0: float,
    gamma_slope: float,
    offset: float = 0.0,
    n_avg: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """``(excess_db, margin_db)`` per order at centres ``(k + offset) * rate``.

    The floor window is always a quarter of the TRUE line spacing either side,
    so ``offset=0.5`` lands between real lines. Reading the null by scaling the
    rate instead (1.5x) puts every even order back on a true line and the null
    then reads the comb — measured at +29 dB before this was fixed.
    """
    excess = np.full(k_hi, np.nan)
    margin = np.full(k_hi, np.nan)
    half = rate / df / 2.0
    debias = median_to_mean(n_avg)
    for i, k in enumerate(range(1, k_hi + 1)):
        c0 = (k + offset) * rate / df
        span = max(4, int(round(3.0 * (gamma0 + gamma_slope * k) / df)))
        lo, hi = int(round(c0)) - span, int(round(c0)) + span + 1
        if lo < 1 or hi >= psd.size - 2:
            break
        fl_med = bench._local_floor(psd, c0, half)
        if not np.isfinite(fl_med) or fl_med <= 0:
            continue
        fl = fl_med * debias
        band = float((psd[lo:hi] - fl).sum())
        if band > 0:
            excess[i] = 10.0 * np.log10(band / fl)
        peak = float(psd[int(round(c0)) - 2 : int(round(c0)) + 3].max())
        margin[i] = 10.0 * np.log10(max(peak / fl, 1e-30))
    return excess, margin


def order_profile(
    x: np.ndarray,
    rate: float,
    *,
    gamma0: float,
    gamma_slope: float,
    sr: int = SR,
    offset: float = 0.0,
    k_hi: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(excess_db, margin_db)`` of one clip, from audio."""
    xx = np.asarray(x, dtype=np.float64)
    xx = xx - xx.mean()
    psd = bench._welch(xx, N_ANALYSIS)
    return comb_excess(
        psd,
        sr / N_ANALYSIS,
        rate,
        k_hi or ORDER_BANDS[-1][1],
        gamma0=gamma0,
        gamma_slope=gamma_slope,
        offset=offset,
        n_avg=welch_averages(xx.size),
    )


def pooled_excess(
    x: np.ndarray,
    rate: float,
    *,
    gamma0: float,
    gamma_slope: float,
    sr: int = SR,
    half: bool = False,
) -> dict[str, float]:
    """Median per-order excess over each order band, plus the detected count."""
    excess, margin = order_profile(
        x,
        rate,
        gamma0=gamma0,
        gamma_slope=gamma_slope,
        sr=sr,
        offset=0.5 if half else 0.0,
    )
    out: dict[str, float] = {}
    for lo, hi in ORDER_BANDS:
        seg = excess[lo - 1 : hi]
        vals = seg[np.isfinite(seg)]
        out[f"k{lo}_{hi}"] = float(np.median(vals)) if vals.size else float("nan")
    good = margin[np.isfinite(margin)]
    out["n_detected"] = float(np.count_nonzero(good >= MARGIN_MIN_DB))
    return out
