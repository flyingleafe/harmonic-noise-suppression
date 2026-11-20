"""Established forecast and sample-fidelity scores of the flight-regime campaign.

Every arm of the campaign (a :class:`~experiments.rps_traj.cruise_api.CruiseModel`)
is scored on held-out BLOCKS by the same pure functions here, all numpy/scipy:

**Probabilistic forecasts** — proper scoring rules (Gneiting and Raftery 2007,
"Strictly proper scoring rules, prediction, and estimation", JASA 102):

* :func:`energy_score` — the multivariate energy score of an ensemble, in its
  unbiased ("fair") form;
* :func:`crps` — the continuous ranked probability score of a scalar ensemble,
  the energy score in one dimension, same fair form (Ferro 2014, "Fair scores
  for ensemble forecasts", QJRMS 140), O(m log m) by sorting;
* :func:`forecast_scores` — both at several lead times over forecast origins
  in every block, next to the block's own climatology as the reference.

**Sample fidelity** of generated blocks against real ones (:func:`sample_scores`),
every statistic POOLED over blocks:

* increment-law distances: 1-Wasserstein (Villani 2009, as computed by
  :func:`scipy.stats.wasserstein_distance`) and excess kurtosis of the pooled
  increments at fixed lags (the stylised-facts checks of Cont 2001,
  "Empirical properties of asset returns: stylized facts and statistical
  issues", Quantitative Finance 1);
* second-order structure: pooled autocorrelation (:func:`pooled_acf`),
  zero-lag inter-rotor correlation and per-rotor variance;
* log-spectral distance (Gray and Markel 1976, "Distance measures for speech
  processing", IEEE TASSP 24) between Welch (1967) PSDs of the four mixer
  modes (:func:`welch_modes`);
* the unbiased squared maximum mean discrepancy with a Gaussian kernel and
  the median-heuristic bandwidth (Gretton et al. 2012, "A kernel two-sample
  test", JMLR 13) between fixed-length windows (:func:`mmd2_rbf`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import signal, stats
from scipy.spatial.distance import pdist

from experiments.rps_traj.cruise_api import NUM_ROTORS, CruiseModel, burn_samples
from tracking.rotors import MIXER

#: Longest Welch segment of :func:`sample_scores` (s).
WELCH_SEG_S = 8.0

#: Default upper edge of the PSD band as a fraction of the sample rate.
BAND_HI_FRAC = 0.4


# --------------------------------------------------------------------------
# Proper scoring rules
# --------------------------------------------------------------------------


def _energy_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fair energy scores of ``k`` ensembles at once: ``x`` ``(m, k, d)``, ``y`` ``(k, d)``."""
    m = x.shape[0]
    if m < 2:
        raise ValueError("the energy score needs at least two draws")
    term1 = np.sqrt(((x - y) ** 2).sum(axis=-1)).mean(axis=0)
    # pdist sums i < j, i.e. half of the i != j sum.
    pair = np.array([pdist(x[:, j]).sum() for j in range(x.shape[1])])
    return term1 - pair / (m * (m - 1))


def _crps_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fair CRPS of ``k`` scalar ensembles at once: ``x`` ``(m, k)``, ``y`` ``(k,)``."""
    m = x.shape[0]
    if m < 2:
        raise ValueError("the CRPS needs at least two draws")
    term1 = np.abs(x - y).mean(axis=0)
    weights = 2.0 * np.arange(1, m + 1) - m - 1.0
    return term1 - (weights @ np.sort(x, axis=0)) / (m * (m - 1))


def energy_score(draws: np.ndarray, obs: np.ndarray) -> float:
    """Fair (unbiased) energy score of an ensemble (Gneiting and Raftery 2007).

    ``ES = mean_i ||x_i - y|| - 1 / (2 m (m - 1)) sum_{i != j} ||x_i - x_j||``
    for ``draws`` ``(m, d)`` (``m >= 2``) and ``obs`` ``(d,)``. Its expectation
    over ensembles equals the energy score of the law the draws come from.
    """
    x = np.asarray(draws, dtype=np.float64)
    y = np.asarray(obs, dtype=np.float64)
    if x.ndim != 2 or y.shape != (x.shape[1],):
        raise ValueError(f"draws must be (m, d) and obs (d,), got {x.shape} and {y.shape}")
    return float(_energy_scores(x[:, None, :], y[None, :])[0])


def crps(draws: np.ndarray, obs: float) -> float:
    """Fair (unbiased) ensemble CRPS of a scalar (Gneiting and Raftery 2007; Ferro 2014).

    The energy score in one dimension; ``sum_{i<j} |x_i - x_j|`` is
    ``sum_k (2k - m - 1) x_(k)`` over the sorted draws, so the cost is
    O(m log m).
    """
    x = np.asarray(draws, dtype=np.float64).reshape(-1, 1)
    return float(_crps_scores(x, np.array([float(obs)]))[0])


def _lead_index(h_s: float, rate_hz: float) -> int:
    idx = int(round(h_s * rate_hz)) - 1
    if idx < 0:
        raise ValueError(f"horizon {h_s} s is shorter than one sample at {rate_hz} Hz")
    return idx


def forecast_scores(
    model: CruiseModel,
    blocks: Sequence[np.ndarray],
    *,
    horizons_s: Sequence[float] = (0.2, 1.0, 2.0),
    stride_s: float = 2.0,
    n_paths: int = 100,
    rng: np.random.Generator,
    max_origins: int | None = None,
) -> dict[str, Any]:
    """Energy score and CRPS of ``model``'s forecasts over all blocks.

    Origins in a block of ``n`` samples are ``o = burn, burn + stride, ...``
    (``burn`` = :func:`burn_samples` of ``model.rate_hz``) while
    ``o + H <= n``, ``H`` the longest lead in samples; each origin calls
    ``model.forecast(block[:, :o], H, n_paths, rng)`` once. The lead of a
    horizon ``h`` is index ``round(h * rate) - 1``, i.e. ``block[:, o + idx]``.
    Each lead is scored with :func:`energy_score` on the 4-vector and with
    :func:`crps` per rotor (mean over rotors). The climatology reference uses,
    per origin, ``n_paths`` samples drawn without replacement (with
    replacement if there are fewer) from the scored part ``block[:, burn:]``
    of the same block, shared by all leads.

    ``max_origins`` keeps that many origins, evenly spaced over the list of
    all origins of all blocks.

    Returns ``{"n_origins": int, "horizons": {f"{h:g}": {"energy", "crps",
    "energy_clim", "crps_clim"}}}`` with every score a mean over origins.
    """
    rate = float(model.rate_hz)
    burn = burn_samples(rate)
    leads = [_lead_index(h, rate) for h in horizons_s]
    horizon = max(leads) + 1
    stride = max(1, int(round(stride_s * rate)))

    origins = [
        (b, o)
        for b, block in enumerate(blocks)
        for o in range(burn, np.shape(block)[1] - horizon + 1, stride)
    ]
    if max_origins is not None and len(origins) > max_origins:
        keep = np.unique(np.round(np.linspace(0, len(origins) - 1, max_origins)).astype(int))
        origins = [origins[i] for i in keep]
    if not origins:
        raise ValueError("no forecast origin fits in any block")

    es = np.zeros(len(leads))
    cr = np.zeros(len(leads))
    es_clim = np.zeros(len(leads))
    cr_clim = np.zeros(len(leads))
    for b, o in origins:
        block = np.asarray(blocks[b], dtype=np.float64)
        paths = np.asarray(model.forecast(block[:, :o], horizon, n_paths, rng), dtype=np.float64)
        if paths.shape != (n_paths, NUM_ROTORS, horizon):
            raise ValueError(
                f"forecast returned {paths.shape}, expected {(n_paths, NUM_ROTORS, horizon)}"
            )
        pool = block[:, burn:]
        cols = rng.choice(pool.shape[1], size=n_paths, replace=pool.shape[1] < n_paths)
        y = block[:, [o + idx for idx in leads]].T  # (n_h, 4)
        draws = paths[:, :, leads].transpose(0, 2, 1)  # (n_paths, n_h, 4)
        clim = np.broadcast_to(pool[:, cols].T[:, None, :], draws.shape)
        es += _energy_scores(draws, y)
        es_clim += _energy_scores(clim, y)
        cr += _crps_scores(draws.reshape(n_paths, -1), y.ravel()).reshape(y.shape).mean(axis=1)
        cr_clim += _crps_scores(clim.reshape(n_paths, -1), y.ravel()).reshape(y.shape).mean(axis=1)

    n_o = len(origins)
    return {
        "n_origins": n_o,
        "horizons": {
            f"{h:g}": {
                "energy": float(es[k] / n_o),
                "crps": float(cr[k] / n_o),
                "energy_clim": float(es_clim[k] / n_o),
                "crps_clim": float(cr_clim[k] / n_o),
            }
            for k, h in enumerate(horizons_s)
        },
    }


# --------------------------------------------------------------------------
# Pooled sample statistics
# --------------------------------------------------------------------------


def _as_blocks(blocks: Sequence[np.ndarray]) -> list[np.ndarray]:
    out = [np.asarray(b, dtype=np.float64) for b in blocks]
    for b in out:
        if b.ndim != 2 or b.shape[0] != NUM_ROTORS:
            raise ValueError(f"blocks must be (4, n), got {b.shape}")
    if not out:
        raise ValueError("need at least one block")
    return out


def pooled_increments(blocks: Sequence[np.ndarray], lag: int) -> np.ndarray:
    """All increments ``y_{t+lag} - y_t`` of all rotors of all blocks, as one 1-D array.

    A block of at most ``lag`` samples contributes nothing.
    """
    if lag < 1:
        raise ValueError("lag must be at least one sample")
    parts = [(b[:, lag:] - b[:, :-lag]).ravel() for b in _as_blocks(blocks) if b.shape[1] > lag]
    return np.concatenate(parts) if parts else np.empty(0)


def pooled_acf(blocks: Sequence[np.ndarray], max_lag: int) -> np.ndarray:
    """``(4, max_lag + 1)`` pooled autocorrelation of demeaned blocks, per rotor.

    At lag ``k`` the products ``y_t y_{t+k}`` are summed over every block that
    has them and divided by their count (the pooled autocovariance); the
    result is divided by the pooled lag-0 value. A block of ``n <= k`` samples
    simply does not count at lag ``k``. Blocks are taken as already demeaned.
    """
    bl = _as_blocks(blocks)
    num = np.zeros((NUM_ROTORS, max_lag + 1))
    cnt = np.zeros(max_lag + 1)
    for b in bl:
        n = b.shape[1]
        nfft = 1 << int(np.ceil(np.log2(2 * n)))
        spec = np.fft.rfft(b, nfft, axis=1)
        prods = np.fft.irfft(spec * np.conj(spec), nfft, axis=1)
        k = min(max_lag + 1, n)
        num[:, :k] += prods[:, :k]
        cnt[:k] += n - np.arange(k)
    if np.any(cnt == 0):
        raise ValueError(f"max_lag {max_lag} exceeds every block")
    cov = num / cnt
    return cov / cov[:, :1]


def _pooled_cov(blocks: Sequence[np.ndarray]) -> np.ndarray:
    bl = _as_blocks(blocks)
    return np.sum([b @ b.T for b in bl], axis=0) / sum(b.shape[1] for b in bl)


def welch_modes(
    blocks: Sequence[np.ndarray], rate_hz: float, nperseg: int
) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD of the four mixer modes, averaged over blocks.

    Modes are ``MIXER.T @ y / 4`` (common, roll, pitch, yaw). Each block gets a
    Welch estimate (Hann, 50 % overlap, ``nperseg`` samples); the block
    estimates are averaged with weight = block length. Blocks shorter than
    ``nperseg`` are skipped. Returns ``(freqs, psd (4, n_freqs))``.
    """
    inv = MIXER.T / NUM_ROTORS
    acc: np.ndarray | None = None
    total = 0
    freqs = np.empty(0)
    for b in _as_blocks(blocks):
        n = b.shape[1]
        if n < nperseg:
            continue
        freqs, p = signal.welch(inv @ b, fs=rate_hz, nperseg=nperseg, axis=1)
        acc = n * p if acc is None else acc + n * p
        total += n
    if acc is None:
        raise ValueError(f"no block has nperseg = {nperseg} samples")
    return freqs, acc / total


def _windows(blocks: Sequence[np.ndarray], width: int) -> np.ndarray:
    rows = []
    for b in _as_blocks(blocks):
        k = b.shape[1] // width
        if k:
            # (4, k, width) -> (k, 4 * width), rotor-major within each window.
            w = b[:, : k * width].reshape(NUM_ROTORS, k, width).transpose(1, 0, 2)
            rows.append(w.reshape(k, NUM_ROTORS * width))
    if not rows:
        raise ValueError(f"no block holds a window of {width} samples")
    return np.concatenate(rows)


def _sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xx = np.einsum("ij,ij->i", x, x)
    yy = np.einsum("ij,ij->i", y, y)
    return np.maximum(xx[:, None] + yy[None, :] - 2.0 * (x @ y.T), 0.0)


def median_bandwidth(x: np.ndarray) -> float:
    """Median-heuristic Gaussian bandwidth: median pairwise Euclidean distance of ``x``."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.median(pdist(x)))


def mmd2_rbf(x: np.ndarray, y: np.ndarray, bandwidth: float | None = None) -> float:
    """Unbiased squared MMD with ``k(a, b) = exp(-||a - b||^2 / (2 bandwidth^2))``.

    Gretton et al. 2012, eq. (3): within-sample kernel means exclude the
    diagonal. ``x`` ``(m, d)`` and ``y`` ``(n, d)``, ``m, n >= 2``.
    ``bandwidth=None`` uses the median heuristic on ``x`` (the reference
    sample). The estimate can be negative when the laws are equal.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m, n = x.shape[0], y.shape[0]
    if m < 2 or n < 2 or x.shape[1:] != y.shape[1:]:
        raise ValueError(f"need (m, d) and (n, d) with m, n >= 2, got {x.shape}, {y.shape}")
    bw = median_bandwidth(x) if bandwidth is None else float(bandwidth)
    if not bw > 0.0:
        raise ValueError(f"bandwidth must be positive, got {bw}")
    g = -0.5 / (bw * bw)
    kxx = np.exp(g * _sq_dists(x, x))
    kyy = np.exp(g * _sq_dists(y, y))
    kxy = np.exp(g * _sq_dists(x, y))
    sxx = (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
    syy = (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
    return float(sxx + syy - 2.0 * kxy.mean())


def sample_scores(
    real_blocks: Sequence[np.ndarray],
    gen_blocks: Sequence[np.ndarray],
    rate_hz: float,
    *,
    lags_s: Sequence[float] = (0.2, 1.0),
    acf_max_s: float = 5.0,
    band_hz: tuple[float, float | None] = (0.1, None),
    window_s: float = 2.0,
    max_windows: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Distances between real and generated demeaned blocks ``(4, n)``, pooled over blocks.

    Keys (``<lag>`` is ``f"{lag_s:g}"``; null value in brackets):

    * ``inc_w1_<lag>`` [0] — 1-Wasserstein distance between the pooled
      increments (:func:`pooled_increments`, lag ``round(lag_s * rate)``),
      divided by the std of the real increments;
    * ``inc_kurt_real_<lag>``, ``inc_kurt_gen_<lag>``, ``inc_kurt_diff_<lag>``
      [diff 0] — their excess kurtosis and its absolute difference;
    * ``acf_rmse`` [0] — RMSE over rotors and lags ``1 .. round(acf_max_s * rate)``
      between the :func:`pooled_acf` of both sets;
    * ``xcorr_mae`` [0] — mean absolute error of the six zero-lag inter-rotor
      correlations of the pooled covariance;
    * ``var_log_ratio`` [0] — ``mean_r |log(var_gen_r / var_real_r)|`` of the
      pooled per-rotor variance;
    * ``psd_lsd_db`` [0] — log-spectral distance, ``sqrt(mean (10 log10(P_gen /
      P_real))^2)`` over the four mixer modes and the Welch bins inside
      ``band_hz`` (upper ``None`` = 0.4 * rate), Welch segment
      ``min(8 s, shortest block of either set)``;
    * ``mmd2`` [0] — :func:`mmd2_rbf` between non-overlapping ``window_s``
      windows flattened to ``4 * window`` numbers and divided by the pooled
      real std, at most ``max_windows`` per side (subsampled with ``rng``;
      ``None`` = seed 0), bandwidth by the median heuristic on the real windows.
    """
    real = _as_blocks(real_blocks)
    gen = _as_blocks(gen_blocks)
    rng = np.random.default_rng(0) if rng is None else rng
    out: dict[str, float] = {}

    for lag_s in lags_s:
        lag = max(1, int(round(lag_s * rate_hz)))
        key = f"{lag_s:g}"
        inc_r = pooled_increments(real, lag)
        inc_g = pooled_increments(gen, lag)
        if inc_r.size < 2 or inc_g.size < 2:
            raise ValueError(f"lag {lag_s} s exceeds every block of one set")
        out[f"inc_w1_{key}"] = float(stats.wasserstein_distance(inc_r, inc_g) / inc_r.std())
        k_r = float(stats.kurtosis(inc_r))
        k_g = float(stats.kurtosis(inc_g))
        out[f"inc_kurt_real_{key}"] = k_r
        out[f"inc_kurt_gen_{key}"] = k_g
        out[f"inc_kurt_diff_{key}"] = abs(k_g - k_r)

    max_lag = max(1, int(round(acf_max_s * rate_hz)))
    acf_r = pooled_acf(real, max_lag)[:, 1:]
    acf_g = pooled_acf(gen, max_lag)[:, 1:]
    out["acf_rmse"] = float(np.sqrt(np.mean((acf_g - acf_r) ** 2)))

    cov_r = _pooled_cov(real)
    cov_g = _pooled_cov(gen)
    sd_r = np.sqrt(np.diag(cov_r))
    sd_g = np.sqrt(np.diag(cov_g))
    iu = np.triu_indices(NUM_ROTORS, k=1)
    corr_r = (cov_r / np.outer(sd_r, sd_r))[iu]
    corr_g = (cov_g / np.outer(sd_g, sd_g))[iu]
    out["xcorr_mae"] = float(np.mean(np.abs(corr_g - corr_r)))
    out["var_log_ratio"] = float(np.mean(np.abs(np.log(np.diag(cov_g) / np.diag(cov_r)))))

    shortest = min(b.shape[1] for b in (*real, *gen))
    nperseg = min(int(round(WELCH_SEG_S * rate_hz)), shortest)
    freqs, psd_r = welch_modes(real, rate_hz, nperseg)
    _, psd_g = welch_modes(gen, rate_hz, nperseg)
    lo, hi = band_hz
    hi = BAND_HI_FRAC * rate_hz if hi is None else hi
    band = (freqs >= lo) & (freqs <= hi)
    if not band.any():
        raise ValueError(f"no Welch bin in band {band_hz} at segment {nperseg}")
    lsd = 10.0 * np.log10(psd_g[:, band] / psd_r[:, band])
    out["psd_lsd_db"] = float(np.sqrt(np.mean(lsd**2)))

    width = max(1, int(round(window_s * rate_hz)))
    scale = float(np.sqrt(np.mean(np.diag(cov_r))))
    win_r = _windows(real, width) / scale
    win_g = _windows(gen, width) / scale
    if win_r.shape[0] > max_windows:
        win_r = win_r[rng.choice(win_r.shape[0], size=max_windows, replace=False)]
    if win_g.shape[0] > max_windows:
        win_g = win_g[rng.choice(win_g.shape[0], size=max_windows, replace=False)]
    out["mmd2"] = mmd2_rbf(win_r, win_g, bandwidth=median_bandwidth(win_r))
    return out


__all__ = [
    "BAND_HI_FRAC",
    "WELCH_SEG_S",
    "crps",
    "energy_score",
    "forecast_scores",
    "median_bandwidth",
    "mmd2_rbf",
    "pooled_acf",
    "pooled_increments",
    "sample_scores",
    "welch_modes",
]
