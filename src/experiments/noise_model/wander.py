"""Slow amplitude wander of the rotor lines and of the floor, MEASURED on real windows.

Noise model v3 (``docs/explainers/noise-model-v3-wander.qmd``) gives every
block of ``block_s`` seconds its own amplitude latents in dB -- ``d_i(b)`` per
rotor, ``v_ik(b)`` per line, ``u(b)`` for the floor level, ``u_j(b)`` per floor
control point -- each an Ornstein-Uhlenbeck process across blocks with a
stationary sd ``sigma`` (dB) and a correlation time ``tau`` (s). Section 3.3a
of the explainer fixes those hyperparameters BEFORE the fit, by the method of
moments on the real windows. This module is that measurement; it evaluates no
model and fits no likelihood.

The block measurement
---------------------

The front end is the flight one the fit reads (2048-point periodic Hann, hop
512, 16 kHz; :func:`experiments.stochastic_fit.data.periodogram`), not the
8192/1024 grid of the R4 line-width estimator it is ported from
(``scripts/noise_v2_widen_dregon.py::line_width_db3``): a 0.5 s block holds
~15 frames of 128 ms and not one of 512 ms.

* LINE (:func:`measure_lines`). Per frame, per rotor, per order ``k``, the
  line's cells are the ``2 * LINE_HALF_BINS + 1`` bins nearest the frame's
  OWN label carrier ``k * f_r(n)`` -- R4's carrier-aligned accumulation, on
  bins instead of an interpolated grid. Three Hann bins hold 98-100 % of a
  line's power at any sub-bin offset (:func:`hann_capture`), so a carrier
  crawling across bins does not masquerade as amplitude. The block's line
  level is the block mean of those cells minus the LOCAL floor (the
  ``LOCAL_FLOOR_QUANTILE`` quantile of the cells ``LOCAL_FLOOR_GAP_BINS`` ..
  ``LOCAL_FLOOR_HALF_BINS`` bins either side over the block's frames, over
  the exponential law's ``-ln(1 - q)``), in dB: ``y[mic, rotor, k, block]``.
  With four rotors a few rev/s apart, other rotors' orders fall into a line's
  cells all the time; :attr:`LineBlocks.own_share` is the fraction of the
  cells' line power that is the line's own (window-mean line powers times
  the Hann capture of every neighbour, per frame).
* FLOOR (:func:`measure_floor`). Per frame, the cells within
  ``COMB_MASK_HALF_BINS`` of every order of every rotor are masked (the rank
  test's mask, ``scripts/_mic_gain_rank.py``), widened to
  ``STRONG_MASK_HALF_BINS`` around lines ``STRONG_LINE_DB`` over their floor
  (their Hann skirts would otherwise leak into the floor cells); the block
  band level is the mean of the surviving cells, in dB, on 1/3-octave bands
  and on bands around the floor spline's control points.

Block noise
-----------

* ``explainer`` -- ``s^2 = (10/ln10)^2 psi_1(n_eff)``, exact for the dB mean
  of ``n_eff`` INDEPENDENT exponential cells, with ``n_eff = frames x bins x
  lineFrac^2`` (explainer sections 1.6 and 3.3a), per microphone. Two of its
  assumptions fail on this front end: 75 %-overlap frames and adjacent Hann
  bins are correlated (a 15 x 3 rectangle of Gaussian-noise cells carries
  the information of 45 / 3.1 independent ones, :func:`overlap_inflation`),
  and a narrow constant-envelope line -- the model's line -- has no
  exponential scatter of its own, only the floor's cross terms.
* ``measured`` (the one the estimates use) -- for a LINE, the variance of the
  block mean measured from the frames inside the block
  (:func:`within_block_mean_var`: lag-0..3 autocovariance of the per-frame
  deviations, the overlap's reach, corrected exactly for the in-block
  centring), which assumes no cell law and no mic-to-mic correlation; for
  the FLOOR, whose cells ARE Gaussian noise, the exponential law at the
  exact equivalent count of the surviving cell pattern
  (:func:`correlated_cell_count`) times the measured mic-coherence count
  :attr:`FloorBlocks.m_eff` for the mic-mean track.

The statistics
--------------

:class:`LagMoments` pools the lag-``j`` products of CENTRED block tracks. A
track lives on a ``(W, B)`` layout -- one window, or every window of a
regime side by side -- and is centred on the mean of all its observed
blocks; products never cross a window. Centring on the window alone is
blind to anything slower than the window; centring on the rig mean is what
the v3 fit does implicitly (``p_ik`` shared by the pool). Either way the
centring biases every product low when ``tau`` is not short against what is
centred, and the expectation of every pooled product under an OU of
``(sigma^2, rho)`` plus independent block noise is linear in ``sigma^2``
with a known coefficient (:func:`_pair_geometry`), so :func:`fit_ou` solves
the moments exactly instead of reading them raw. For a PAIR of tracks the
lag-0 product also carries the block noise the two share (a broadband event
lifts every line of a block); ``fit_ou(..., skip_lag0=True)`` fits from lags
1-4 alone and returns that excess as the nugget. The explainer's literal
estimate -- ``Var_b(y) - s^2`` and ``Cov(y_b, y_b+1) / sigma^2`` -- is kept
beside it (:meth:`LagMoments.naive`).

Every valid track
-----------------

The moments above need a line level per block, so they run on lines that
stand clear of their floor -- a selection on low scatter. The per-line sd of
the v3 contract is read instead off EVERY classified track
(:func:`classify_tracks`: valid in >= ``MIN_VALID_FRAC`` of its window's
blocks) through its block PROMINENCE, with the forward model of the
intermittency diagnostic (``scripts/_dregon_intermittency.py`` test (b)):
:class:`ProminenceModel` -- a Gaussian line level in dB plus block noise, the
speed law and the track's measured local floor deviation, plus the
line-free cell-over-floor ratio. :func:`prominence_moments` tabulates the
weighted, per-track-centred lag sums of the data and of the model on a
``(sigma, tau)`` grid, summed per cell (window x order group x class), and
:func:`fit_prominence_ou` solves any set of cells -- one order group, a
window-bootstrap draw -- like :func:`fit_ou` does its linear moments.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.special import polygamma

from data_processing.noise_model.spectrum import (
    FLIGHT_HOP,
    FLIGHT_N_FFT,
    FLIGHT_SR,
    floor_ctrl_hz,
    k_max_for_carrier,
)
from experiments.stochastic_fit.data import Clip, periodogram

#: dB per neper of POWER: ``10 log10(x) = DB_PER_NEPER * ln(x)``.
DB_PER_NEPER = 10.0 / math.log(10.0)

#: The model's observation band (``supports.BAND_LEVEL_F_MIN/MAX``).
F_LO_HZ = 30.0
F_HI_HZ = 7900.0

#: Line cells: the nearest bin +- this many.
LINE_HALF_BINS = 1
#: The local floor of a line: cells this many bins or more from it ...
LOCAL_FLOOR_GAP_BINS = 3
#: ... and at most this many (12 bins = 94 Hz, about one order spacing).
LOCAL_FLOOR_HALF_BINS = 12
#: Quantile of the local cells read as the floor: a low quantile is not
#: lifted by the other rotors' lines that fall into the neighbourhood.
LOCAL_FLOOR_QUANTILE = 0.25
#: The floor mask around every order (the rank test's +-2 Hz = one bin) ...
COMB_MASK_HALF_BINS = 1
#: ... and around a line this far over its floor (window mean, mic mean): a
#: Hann skirt 1.5 bins out is -15 dB, so a 10 dB line leaks under the floor.
STRONG_LINE_DB = 10.0
STRONG_MASK_HALF_BINS = 3
#: A floor band keeps a block with at least this many surviving cells per frame.
MIN_FLOOR_CELLS_PER_FRAME = 2.0
#: Neighbouring lines further than this many bins are not counted in the
#: own-share (a Hann line 3.5 bins out puts 2e-4 of its power in the cells).
NEIGHBOUR_REACH_BINS = 4.0
#: A line is ROTOR-DOMINANT when at least this share of its cells' line power
#: is its own: a neighbour's wander then enters its track at <= 20 % weight.
OWN_SHARE_MIN = 0.8
#: The explainer's "resolvable" rule: block-level prominence >= 6 dB ...
PROMINENCE_DB = 6.0
#: ... in at least 80 % of the window's blocks.
PROMINENT_BLOCK_FRAC = 0.8
#: Inside a resolvable track, a block under this prominence is a missing
#: value, not a measurement: at 3 dB lineFrac is 0.5 and the explainer count
#: already quarters; below it the level is the floor estimate's noise.
MIN_BLOCK_PROMINENCE_DB = 3.0
#: A block needs at least this fraction of its nominal frame count.
BLOCK_MIN_FRAC = 0.5
#: The frame overlap of the front end reaches ``N / hop - 1`` hops.
WITHIN_BLOCK_MAX_LAG = FLIGHT_N_FFT // FLIGHT_HOP - 1
#: Across-block lags pooled by the statistics (the explainer's "lags 1-4").
MAX_LAG = 4
#: The correlation-time grid the OU moment fit searches, in seconds.
TAU_GRID_S = np.geomspace(0.02, 500.0, 4000)


def noise_var_db(n_eff: Any) -> np.ndarray:
    """``(10/ln10)^2 psi_1(n)``: the variance in dB^2 of ``10 log10`` of the mean
    of ``n`` independent unit exponentials (a ``Gamma(n, 1/n)`` variable).

    Exact for any real ``n > 0``; ``~ 18.86 / n`` for large ``n``. NaN where
    ``n <= 0``.
    """
    n = np.asarray(n_eff, dtype=np.float64)
    out = np.full(n.shape, np.nan)
    ok = np.isfinite(n) & (n > 0)
    out[ok] = DB_PER_NEPER**2 * polygamma(1, n[ok])
    return out


def rel_var_to_db2(rel_var: Any) -> np.ndarray:
    """A block mean's RELATIVE variance ``v`` in dB^2, through the Gamma law:
    :func:`noise_var_db` at the equivalent count ``1/v`` (``~ 18.86 v`` for
    small ``v``). Zero stays zero; NaN stays NaN."""
    v = np.asarray(rel_var, dtype=np.float64)
    out = np.where(v == 0.0, 0.0, np.nan)
    ok = np.isfinite(v) & (v > 0)
    out[ok] = noise_var_db(1.0 / v[ok])
    return out


def hann_periodic(n_fft: int = FLIGHT_N_FFT) -> np.ndarray:
    """The front end's window (:func:`experiments.stochastic_fit.data.periodogram`)."""
    return np.hanning(n_fft + 1)[:n_fft]


def hann_capture(delta_bins: Any) -> np.ndarray:
    """Share of a sinusoid's periodogram power that lands in ONE bin
    ``delta_bins`` away from it, for the periodic Hann (large-``N`` kernel
    ``[sinc(d) / (1 - d^2)]^2``; its integer-shift sum is 1.5, divided out)."""
    d = np.asarray(delta_bins, dtype=np.float64)
    den = 1.0 - d * d
    near = np.abs(den) < 1e-9
    k = np.where(near, 0.5, np.sinc(d) / np.where(near, 1.0, den))
    return k * k / 1.5


@lru_cache(maxsize=8)
def cell_power_correlation(
    n_fft: int = FLIGHT_N_FFT, hop: int = FLIGHT_HOP, max_bins: int = 4
) -> np.ndarray:
    """``rho2[dn, dk]``: power correlation of two Gaussian-noise periodogram cells
    ``dn`` frames and ``dk`` bins apart (``|cov(X, X')|^2 / (E|X|^2)^2`` for
    circular complex Gaussian coefficients; white noise, periodic Hann).
    """
    w = hann_periodic(n_fft)
    e = float(np.sum(w**2))
    n_lag = int(math.ceil(n_fft / hop))
    out = np.zeros((n_lag, max_bins + 1))
    idx = np.arange(n_fft, dtype=np.float64)
    for dn in range(n_lag):
        d = dn * hop
        prod = w[d:] * w[: n_fft - d]  # frame n+dn's window against frame n's
        for dk in range(max_bins + 1):
            c = np.sum(prod * np.exp(-2j * np.pi * dk * idx[d:] / n_fft))
            out[dn, dk] = float(np.abs(c) ** 2) / e**2
    return out


def correlated_cell_count(keep: np.ndarray) -> float:
    """Equivalent INDEPENDENT count of the cells ``keep`` ``(frames, bins)``
    (a boolean pattern, e.g. a band's floor cells in one block): ``n^2 /
    sum_{c, c'} rho2(c, c')`` -- the moment-matched Gamma shape of their mean
    for Gaussian noise of a flat spectrum."""
    k = np.asarray(keep, dtype=bool)
    n = int(k.sum())
    if n == 0:
        return 0.0
    rho2 = cell_power_correlation()
    t_, f_ = k.shape
    kf = k.astype(np.float64)
    tot = 0.0
    for dn in range(-(rho2.shape[0] - 1), rho2.shape[0]):
        a = abs(dn)
        if a >= t_:
            continue
        for dk in range(-(rho2.shape[1] - 1), rho2.shape[1]):
            r2 = rho2[a, abs(dk)]
            if r2 < 1e-6 or abs(dk) >= f_:
                continue
            x = kf[max(0, -dn) : t_ - max(0, dn), max(0, -dk) : f_ - max(0, dk)]
            y = kf[max(0, dn) : t_ - max(0, -dn), max(0, dk) : f_ - max(0, -dk)]
            tot += r2 * float(np.sum(x * y))
    return n * n / tot


def overlap_inflation(n_frames: int, n_bins: int) -> float:
    """``n_frames * n_bins / correlated_cell_count(rectangle)``: how much the
    front end's overlap inflates the variance of a rectangle's mean over the
    same number of independent cells (~1.9 in time, ~1.6 over three bins)."""
    return n_frames * n_bins / correlated_cell_count(np.ones((n_frames, n_bins), dtype=bool))


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------


def frame_blocks(
    frame_centres_s: np.ndarray,
    block_s: float,
    *,
    hop_s: float = FLIGHT_HOP / FLIGHT_SR,
    min_frac: float = BLOCK_MIN_FRAC,
) -> list[np.ndarray]:
    """Consecutive frame indices of each block: frame ``n`` belongs to block
    ``floor(centre_n / block_s)``; a block with fewer than ``min_frac`` of its
    nominal ``block_s / hop_s`` frames (a ragged window end) is dropped. Only
    the ends can be dropped, so consecutive list entries are consecutive
    blocks (a gap raises)."""
    t = np.asarray(frame_centres_s, dtype=np.float64)
    idx = np.floor(t / float(block_s) + 1e-9).astype(np.int64)
    need = max(2.0, float(min_frac) * float(block_s) / float(hop_s))
    kept = [
        (b, np.nonzero(idx == b)[0]) for b in range(int(idx.max()) + 1) if (idx == b).sum() >= need
    ]
    ids = [b for b, _ in kept]
    if ids and ids != list(range(ids[0], ids[0] + len(ids))):
        raise ValueError(f"block grid has a gap: kept blocks {ids}")
    return [f for _, f in kept]


@lru_cache(maxsize=256)
def _centred_geometry(n: int, max_lag: int) -> np.ndarray:
    """``G[j, d] = sum_i [P E_d P]_{i, i+j}`` for the ``n x n`` centring ``P``.

    ``E_d`` has ones where ``|i - i'| = d``: the expected sum of centred
    lag-``j`` products of a stationary sequence with autocovariance
    ``gamma_d`` (zero beyond ``max_lag``) is ``sum_d gamma_d G[j, d]``.
    """
    p = np.eye(n) - 1.0 / n
    g = np.zeros((max_lag + 1, max_lag + 1))
    for d in range(max_lag + 1):
        e = np.eye(n, k=d) + (np.eye(n, k=-d) if d else 0.0)
        m = p @ e @ p
        for j in range(min(max_lag, n - 1) + 1):
            g[j, d] = float(np.trace(m, offset=j))
    return g


def within_block_mean_var(
    e: np.ndarray,
    blocks: Sequence[np.ndarray],
    valid: np.ndarray,
    *,
    max_lag: int = WITHIN_BLOCK_MAX_LAG,
) -> np.ndarray:
    """Variance of each block's MEAN, from the frames inside the blocks.

    ``e`` is ``(..., T)`` per-frame values, each block already centred on its
    own mean (e.g. the relative deviation of a frame's line power from the
    block level); ``valid`` is ``(..., B)``. Per track (the leading axes) the
    centred lag-0..``max_lag`` products are pooled over the valid blocks, the
    autocovariances ``gamma_0..gamma_L`` solved from their exact expectation
    (:func:`_centred_geometry`), and the variance of the mean of block ``b``'s
    ``n_b`` frames is ``sum_{i, i'} gamma_|i - i'| / n_b^2``. Frames more than
    ``max_lag`` hops apart share no samples and are taken as uncorrelated.
    Returns ``(..., B)``, clipped at zero, NaN where the block is invalid.
    """
    lead = e.shape[:-1]
    c = np.zeros(lead + (max_lag + 1,))
    g = np.zeros(lead + (max_lag + 1, max_lag + 1))
    val = np.asarray(valid, dtype=bool)
    for b, fidx in enumerate(blocks):
        n = int(fidx.size)
        z = np.where(val[..., b, None], np.nan_to_num(e[..., fidx]), 0.0)
        for j in range(min(max_lag, n - 1) + 1):
            c[..., j] += np.sum(z[..., : n - j] * z[..., j:], axis=-1)
        g += val[..., b, None, None] * _centred_geometry(n, max_lag)
    out = np.full(lead + (len(blocks),), np.nan)
    has = val.any(axis=-1)
    if not has.any():
        return out
    gamma = np.full(lead + (max_lag + 1,), np.nan)
    gamma[has] = np.linalg.solve(g[has], c[has][..., None])[..., 0]
    for b, fidx in enumerate(blocks):
        n = int(fidx.size)
        wts = np.array([n] + [2 * max(n - d, 0) for d in range(1, max_lag + 1)], float) / n**2
        out[..., b] = np.where(val[..., b], np.maximum(gamma @ wts, 0.0), np.nan)
    return out


# ---------------------------------------------------------------------------
# lines
# ---------------------------------------------------------------------------


@dataclass
class LineBlocks:
    """Block line levels of one window. Axes: ``m`` mic, ``r`` rotor, ``k``
    order (``orders[k]``), ``b`` block."""

    orders: np.ndarray  # (K,)
    block_s: float
    n_frames: np.ndarray  # (B,) frames per block
    carrier_rev_s: np.ndarray  # (R, B) block-mean label carrier
    line_db: np.ndarray  # (M, R, K, B) 10 log10(line power) per mic, NaN if <= floor
    line_frac: np.ndarray  # (M, R, K, B) line power / cell power
    prominence_db: np.ndarray  # (R, K, B) mic-summed cells over mic-summed floor
    floor_db: np.ndarray  # (R, K, B) the mic-summed local floor (the prominence's denominator), dB
    window_prominence_db: np.ndarray  # (R, K) the same over the whole window
    n_cells: np.ndarray  # (R, K, B) frames x bins (the explainer's literal count)
    s2_explainer: np.ndarray  # (M, R, K, B) (10/ln10)^2 psi1(n_cells lineFrac^2)
    s2_emp: np.ndarray  # (M, R, K, B) measured variance of the block mean, dB^2
    s2_emp_micmean: np.ndarray  # (R, K, B) the same for the mic-mean track
    valid: np.ndarray  # (R, K, B) the line stays inside the band and the grid all block
    own_share: np.ndarray  # (R, K) the line's own share of its cells' line power

    @property
    def n_blocks(self) -> int:
        return int(self.n_frames.size)

    def resolvable(self) -> np.ndarray:
        """``(R, K)``: prominence >= PROMINENCE_DB in >= PROMINENT_BLOCK_FRAC of
        the window's blocks (a block the line leaves the grid in counts as
        not prominent)."""
        prom = np.where(self.valid, self.prominence_db, -np.inf)
        return (prom >= PROMINENCE_DB).mean(axis=-1) >= PROMINENT_BLOCK_FRAC

    def dominant(self) -> np.ndarray:
        """``(R, K)``: own share >= OWN_SHARE_MIN (the per-rotor attribution holds)."""
        return self.own_share >= OWN_SHARE_MIN

    def measured(self) -> np.ndarray:
        """``(R, K, B)``: blocks that carry a line level (valid and prominent enough)."""
        return self.valid & (
            np.nan_to_num(self.prominence_db, nan=-np.inf) >= MIN_BLOCK_PROMINENCE_DB
        )

    def micmean_db(self) -> np.ndarray:
        """``(R, K, B)`` mean of ``line_db`` over the mics that carry the line
        (NaN when fewer than half do, or the block is not :meth:`measured`)."""
        m = self.line_db.shape[0]
        fin = np.isfinite(self.line_db)
        cnt = fin.sum(axis=0)
        s = np.where(fin, self.line_db, 0.0).sum(axis=0)
        ok = (cnt >= (m + 1) // 2) & self.measured()
        return np.where(ok, s / np.maximum(cnt, 1), np.nan)

    def mic_db(self) -> np.ndarray:
        """``(M, R, K, B)`` per-mic levels, NaN where the block is not measured."""
        return np.where(self.measured()[None], self.line_db, np.nan)

    def s2_explainer_micmean(self) -> np.ndarray:
        """``(R, K, B)``: the explainer's count for the mic-mean track, at the
        mic-pooled line fraction ``1 - 10^(-prominence/10)`` and NO reduction
        for the mic average (a Gaussian line's scatter is common to the mics)."""
        frac = 1.0 - 10.0 ** (-np.nan_to_num(self.prominence_db, nan=-np.inf) / 10.0)
        frac = np.clip(frac, 0.0, 1.0)
        return np.where(self.measured(), noise_var_db(self.n_cells * frac**2), np.nan)

    def take_orders(self, idx: np.ndarray) -> LineBlocks:
        """The same window restricted to the order positions ``idx``."""
        i = np.asarray(idx, dtype=np.int64)
        return LineBlocks(
            orders=self.orders[i],
            block_s=self.block_s,
            n_frames=self.n_frames,
            carrier_rev_s=self.carrier_rev_s,
            line_db=self.line_db[:, :, i],
            line_frac=self.line_frac[:, :, i],
            prominence_db=self.prominence_db[:, i],
            floor_db=self.floor_db[:, i],
            window_prominence_db=self.window_prominence_db[:, i],
            n_cells=self.n_cells[:, i],
            s2_explainer=self.s2_explainer[:, :, i],
            s2_emp=self.s2_emp[:, :, i],
            s2_emp_micmean=self.s2_emp_micmean[:, i],
            valid=self.valid[:, i],
            own_share=self.own_share[:, i],
        )


def _gather(power: np.ndarray, bins: np.ndarray, offs: np.ndarray) -> np.ndarray:
    """``power[m, n, bins[k, n] + offs]`` -> ``(M, K, T, len(offs))``."""
    t = np.arange(power.shape[1])[None, :, None]
    return power[:, t, bins[:, :, None] + offs[None, None, :]]


def _own_share(
    car: np.ndarray, ks: np.ndarray, level: np.ndarray, in_grid: np.ndarray, df: float
) -> np.ndarray:
    """``(R, K)`` frame-mean share of a line's cells' line power that is its own.

    ``level`` ``(R, K)`` is each line's window-mean power (linear, >= 0). Per
    frame, every other rotor's orders within ``NEIGHBOUR_REACH_BINS`` of the
    line put ``level * sum_cells hann_capture`` into the line's cells.
    """
    r_, t_ = car.shape
    k_ = ks.size
    offs = np.arange(-LINE_HALF_BINS, LINE_HALF_BINS + 1, dtype=np.float64)
    out = np.zeros((r_, k_))
    for r in range(r_):
        x = ks[:, None] * car[r][None, :] / df  # (K, T) fractional bin
        b0 = np.rint(x)
        own = level[r][:, None] * hann_capture(b0[..., None] + offs - x[..., None]).sum(-1)
        other = np.zeros_like(own)
        for r2 in range(r_):
            if r2 == r:
                continue
            f2 = np.maximum(car[r2][None, :], 1e-9) / df  # (1, T) carrier in bins
            k_near = np.rint(x / f2)
            for dk in (-1.0, 0.0, 1.0):
                k2 = k_near + dk
                x2 = k2 * f2
                near = (k2 >= 1) & (k2 <= ks[-1]) & (np.abs(x2 - x) <= NEIGHBOUR_REACH_BINS)
                lvl = level[r2][np.clip(k2.astype(np.int64) - int(ks[0]), 0, k_ - 1)]
                cap = hann_capture(b0[..., None] + offs - x2[..., None]).sum(-1)
                other += np.where(near, lvl * cap, 0.0)
        tot = own + other
        share = np.where(tot > 0, own / np.where(tot > 0, tot, 1.0), 0.0)
        w = in_grid[r].astype(np.float64)
        out[r] = (share * w).sum(axis=1) / np.maximum(w.sum(axis=1), 1.0)
    return out


def measure_lines(
    power: np.ndarray,
    freqs_hz: np.ndarray,
    carriers_rev_s: np.ndarray,
    blocks: Sequence[np.ndarray],
    *,
    block_s: float,
    orders: Sequence[int] | None = None,
) -> LineBlocks:
    """Order-tracked block line levels of one window.

    ``power`` ``(M, T, F)`` is the front end's periodogram (``|X|^2 / sum w^2``),
    ``carriers_rev_s`` ``(R, T)`` the frame-mean label. ``orders`` (consecutive,
    from 1 for :attr:`LineBlocks.own_share` to see every neighbour) default to
    ``1..k_max_for_carrier`` of the window's slowest rotor mean; a line past
    ``F_HI_HZ`` in a block is invalid there, not dropped.
    """
    pw = np.asarray(power, dtype=np.float64)
    freqs = np.asarray(freqs_hz, dtype=np.float64)
    car = np.atleast_2d(np.asarray(carriers_rev_s, dtype=np.float64))
    m_, t_, f_ = pw.shape
    r_ = car.shape[0]
    df = float(freqs[1] - freqs[0])
    if orders is None:
        k_top = k_max_for_carrier(float(np.min(car.mean(axis=1))), FLIGHT_SR)
        orders = list(range(1, k_top + 1))
    ks = np.asarray(list(orders), dtype=np.int64)
    k_ = ks.size
    b_ = len(blocks)
    w_hi = LOCAL_FLOOR_HALF_BINS
    offs = np.arange(-LINE_HALF_BINS, LINE_HALF_BINS + 1)
    side = np.arange(LOCAL_FLOOR_GAP_BINS, LOCAL_FLOOR_HALF_BINS + 1)
    foffs = np.concatenate([-side[::-1], side])
    q_scale = -math.log1p(-LOCAL_FLOOR_QUANTILE)
    n_line = int(offs.size)

    shape4 = (m_, r_, k_, b_)
    line_db = np.full(shape4, np.nan)
    line_frac = np.full(shape4, np.nan)
    s2_expl = np.full(shape4, np.nan)
    s2_emp = np.full(shape4, np.nan)
    s2_emp_mm = np.full((r_, k_, b_), np.nan)
    prom = np.full((r_, k_, b_), np.nan)
    floor_db = np.full((r_, k_, b_), np.nan)
    n_cells = np.zeros((r_, k_, b_), dtype=np.int64)
    valid = np.zeros((r_, k_, b_), dtype=bool)
    cell_sum = np.zeros((r_, k_))
    floor_sum = np.zeros((r_, k_))
    level = np.zeros((r_, k_))
    in_grid_all = np.zeros((r_, k_, t_), dtype=bool)
    nfr = np.array([f.size for f in blocks], dtype=np.int64)
    car_b = np.stack([car[:, f].mean(axis=1) for f in blocks], axis=1)  # (R, B)
    need = (m_ + 1) // 2

    for r in range(r_):
        fr = ks[:, None].astype(np.float64) * car[r][None, :]  # (K, T) Hz
        b0 = np.rint(fr / df).astype(np.int64)
        in_grid = (fr >= F_LO_HZ) & (fr <= F_HI_HZ) & (b0 - w_hi >= 0) & (b0 + w_hi < f_)
        in_grid_all[r] = in_grid
        b0c = np.clip(b0, w_hi, f_ - 1 - w_hi)
        x = _gather(pw, b0c, offs).mean(axis=-1)  # (M, K, T) per-frame line cells
        nb = _gather(pw, b0c, foffs)  # (M, K, T, S) local floor cells
        e_frame = np.zeros(x.shape)
        lin_pos = np.zeros((m_, k_))
        n_pos = np.zeros(k_)
        for b, fidx in enumerate(blocks):
            okb = in_grid[:, fidx].all(axis=1)  # (K,)
            valid[r, :, b] = okb
            pbar = x[:, :, fidx].mean(axis=-1)  # (M, K)
            q = np.quantile(nb[:, :, fidx, :].reshape(m_, k_, -1), LOCAL_FLOOR_QUANTILE, axis=-1)
            fhat = q / q_scale
            lin = pbar - fhat
            pos = lin > 0
            safe = np.where(pos, lin, 1.0)
            line_db[:, r, :, b] = np.where(pos, 10.0 * np.log10(safe), np.nan)
            frac = np.clip(lin / pbar, 0.0, 1.0)
            line_frac[:, r, :, b] = frac
            n = int(fidx.size) * n_line
            n_cells[r, :, b] = n
            s2_expl[:, r, :, b] = np.where(pos, noise_var_db(n * frac**2), np.nan)
            ps, fs = pbar.sum(axis=0), fhat.sum(axis=0)
            prom[r, :, b] = 10.0 * np.log10(np.maximum(ps, 1e-300) / np.maximum(fs, 1e-300))
            floor_db[r, :, b] = 10.0 * np.log10(np.maximum(fs, 1e-300))
            cell_sum[r] += np.where(okb, ps, 0.0)
            floor_sum[r] += np.where(okb, fs, 0.0)
            lin_pos += np.where(okb[None], np.maximum(lin, 0.0), 0.0)
            n_pos += okb
            # frame deviations relative to the block's LINE level
            dev = (x[:, :, fidx] - pbar[..., None]) / safe[..., None]
            e_frame[:, :, fidx] = np.where(pos[..., None], dev, 0.0)
        level[r] = lin_pos.mean(axis=0) / np.maximum(n_pos, 1)
        ok_mb = np.isfinite(line_db[:, r]) & valid[r][None]  # (M, K, B)
        s2_emp[:, r] = rel_var_to_db2(within_block_mean_var(e_frame, blocks, ok_mb))
        # the mic-mean track: each frame's deviations averaged over the mics
        # that carry the line in that block (the dB mean's delta-method twin)
        e_mm = np.zeros((k_, t_))
        ok_mm = np.zeros((k_, b_), dtype=bool)
        for b, fidx in enumerate(blocks):
            okm = ok_mb[:, :, b]  # (M, K)
            cnt = okm.sum(axis=0)
            e_sum = np.where(okm[..., None], e_frame[:, :, fidx], 0.0).sum(axis=0)
            e_mm[:, fidx] = e_sum / np.maximum(cnt, 1)[:, None]
            ok_mm[:, b] = (cnt >= need) & valid[r, :, b]
        s2_emp_mm[r] = rel_var_to_db2(within_block_mean_var(e_mm, blocks, ok_mm))

    win_prom = 10.0 * np.log10(np.maximum(cell_sum, 1e-300) / np.maximum(floor_sum, 1e-300))
    return LineBlocks(
        orders=ks,
        block_s=float(block_s),
        n_frames=nfr,
        carrier_rev_s=car_b,
        line_db=line_db,
        line_frac=line_frac,
        prominence_db=prom,
        floor_db=floor_db,
        window_prominence_db=np.where(floor_sum > 0, win_prom, np.nan),
        n_cells=n_cells,
        s2_explainer=s2_expl,
        s2_emp=s2_emp,
        s2_emp_micmean=s2_emp_mm,
        valid=valid,
        own_share=_own_share(car, ks.astype(np.float64), level, in_grid_all, df),
    )


def block_line_power_db(
    audio: np.ndarray,
    f0_tracks: np.ndarray,
    k: int | Sequence[int],
    sr: int = FLIGHT_SR,
    block_s: float = 0.5,
) -> LineBlocks:
    """The R4 order-tracked line power, per BLOCK, from audio.

    ``audio`` ``(M, T)`` at the flight rate, ``f0_tracks`` ``(R, T)`` rotor
    speed in rev/s on the audio grid, ``k`` one order or several. Runs the
    frozen front end (:func:`experiments.stochastic_fit.data.periodogram`:
    periodic Hann 2048/512, frame-mean carriers) and :func:`measure_lines`.
    """
    if int(sr) != FLIGHT_SR:
        raise ValueError(f"the flight front end is defined at {FLIGHT_SR} Hz, got {sr}")
    x = np.atleast_2d(np.asarray(audio, dtype=np.float32))
    f0 = np.atleast_2d(np.asarray(f0_tracks, dtype=np.float64))
    if f0.shape[-1] != x.shape[-1]:
        raise ValueError(f"{x.shape[-1]} audio samples against {f0.shape[-1]} label samples")
    pg = periodogram(Clip("wander", "wander", x, f0, int(sr), f0, {}), FLIGHT_N_FFT, FLIGHT_HOP)
    blocks = frame_blocks(pg.times, block_s)
    orders = [int(v) for v in np.atleast_1d(np.asarray(k, dtype=np.int64))]
    return measure_lines(pg.power, pg.freqs, pg.rps, blocks, block_s=block_s, orders=orders)


# ---------------------------------------------------------------------------
# floor
# ---------------------------------------------------------------------------


def third_octave_edges(f_lo: float = F_LO_HZ, f_hi: float = F_HI_HZ) -> np.ndarray:
    """The rank test's 1/3-octave edges (``scripts/_mic_gain_rank.band_edges``)."""
    return 2.0 ** np.arange(np.log2(f_lo), np.log2(f_hi) + 1e-9, 1.0 / 3.0)


def control_band_edges(sr: int = FLIGHT_SR) -> np.ndarray:
    """Bands around the floor spline's control points (``floor_ctrl_hz``): edges
    at the geometric midpoints, closed by ``F_LO_HZ``/``F_HI_HZ``, so band
    ``j`` is where control value ``c_j`` carries the most weight."""
    c = floor_ctrl_hz(sr)
    mid = np.sqrt(c[:-1] * c[1:])
    return np.concatenate([[F_LO_HZ], mid, [F_HI_HZ]])


def comb_mask(
    freqs_hz: np.ndarray,
    carriers_rev_s: np.ndarray,
    half_bins: int = COMB_MASK_HALF_BINS,
    *,
    wide: np.ndarray | None = None,
    wide_half_bins: int = STRONG_MASK_HALF_BINS,
) -> np.ndarray:
    """``(T, F)``: True within ``half_bins`` of any order of any rotor, per
    frame, and within ``wide_half_bins`` of the orders ``wide[r, k - 1]``."""
    freqs = np.asarray(freqs_hz, dtype=np.float64)
    car = np.atleast_2d(np.asarray(carriers_rev_s, dtype=np.float64))
    df = float(freqs[1] - freqs[0])
    f_ = freqs.size
    t_ = car.shape[1]
    mask = np.zeros((t_, f_), dtype=bool)
    reach = max(int(half_bins), int(wide_half_bins) if wide is not None else 0)
    offs = np.arange(-reach, reach + 1)
    rows = np.arange(t_)
    for r in range(car.shape[0]):
        f0 = car[r]
        pos = f0[f0 > 0]
        if pos.size == 0:
            continue
        k_top = int(np.floor(float(freqs[-1]) / float(pos.min())))
        if k_top < 1:
            continue
        ks = np.arange(1, k_top + 1)
        half = np.full(k_top, int(half_bins))
        if wide is not None:
            w = np.asarray(wide[r], dtype=bool)[:k_top]
            half[: w.size][w] = int(wide_half_bins)
        b = np.rint(ks[:, None] * f0[None, :] / df).astype(np.int64)  # (K, T)
        cols = b[:, :, None] + offs[None, None, :]
        ok = (
            (cols >= 0)
            & (cols < f_)
            & (f0[None, :, None] > 0)
            & (np.abs(offs)[None, None, :] <= half[:, None, None])
        )
        rr = np.broadcast_to(rows[None, :, None], cols.shape)
        mask[rr[ok], cols[ok]] = True
    return mask


@dataclass
class FloorBlocks:
    """Block floor band levels of one window. Axes: ``m`` mic, ``j`` band, ``b`` block."""

    edges_hz: np.ndarray  # (J + 1,)
    centres_hz: np.ndarray  # (J,)
    level_db: np.ndarray  # (M, J, B) mean of the surviving cells, per mic
    n_cells: np.ndarray  # (J, B) surviving (frame, bin) cells
    n_equiv: np.ndarray  # (J, B) their equivalent independent count
    m_eff: np.ndarray  # (J,) mic-coherence count: mic-mean noise = single-mic / m_eff
    s2_explainer: np.ndarray  # (M, J, B) (10/ln10)^2 psi1(n_cells)
    s2_mic: np.ndarray  # (M, J, B) (10/ln10)^2 psi1(n_equiv)
    s2_micmean: np.ndarray  # (J, B) (10/ln10)^2 psi1(n_equiv m_eff)
    valid: np.ndarray  # (J, B)
    masked_frac: float  # of all (frame, bin) cells in 30-7900 Hz

    def micmean_db(self) -> np.ndarray:
        return np.where(self.valid, self.level_db.mean(axis=0), np.nan)

    def mic_db(self) -> np.ndarray:
        return np.where(self.valid[None], self.level_db, np.nan)


def measure_floor(
    power: np.ndarray,
    freqs_hz: np.ndarray,
    carriers_rev_s: np.ndarray,
    blocks: Sequence[np.ndarray],
    edges_hz: np.ndarray,
    *,
    strong: np.ndarray | None = None,
) -> FloorBlocks:
    """Comb-masked band levels per block (``strong`` ``(R, K)`` over orders
    ``1..K`` widens the mask, :func:`comb_mask`). A band keeps a block when at
    least ``MIN_FLOOR_CELLS_PER_FRAME`` cells per frame survive; a band the
    comb covers is NaN rather than fabricated."""
    pw = np.asarray(power, dtype=np.float64)
    freqs = np.asarray(freqs_hz, dtype=np.float64)
    edges = np.asarray(edges_hz, dtype=np.float64)
    m_, _, _ = pw.shape
    keep = ~comb_mask(freqs, carriers_rev_s, COMB_MASK_HALF_BINS, wide=strong)  # (T, F)
    band_bins = [
        np.nonzero((freqs >= lo) & (freqs < hi))[0] for lo, hi in zip(edges[:-1], edges[1:])
    ]
    j_ = len(band_bins)
    b_ = len(blocks)
    level = np.full((m_, j_, b_), np.nan)
    ncell = np.zeros((j_, b_), dtype=np.int64)
    neq = np.zeros((j_, b_))
    valid = np.zeros((j_, b_), dtype=bool)
    m_eff = np.full(j_, np.nan)
    for j, idx in enumerate(band_bins):
        if idx.size == 0:
            continue
        cov = np.zeros((m_, m_))
        for b, fidx in enumerate(blocks):
            sub = keep[fidx][:, idx]  # (F_b, n_j)
            n = int(sub.sum())
            ncell[j, b] = n
            if n < MIN_FLOOR_CELLS_PER_FRAME * fidx.size:
                continue
            cells = pw[:, fidx][:, :, idx][:, sub]  # (M, n)
            mean = cells.mean(axis=1)
            level[:, j, b] = 10.0 * np.log10(np.maximum(mean, 1e-300))
            neq[j, b] = correlated_cell_count(sub)
            valid[j, b] = True
            dev = cells / np.maximum(mean, 1e-300)[:, None] - 1.0
            cov += dev @ dev.T
        if valid[j].any():
            sd = np.sqrt(np.maximum(np.diag(cov), 1e-300))
            corr = cov / np.outer(sd, sd)
            m_eff[j] = float(np.clip(m_**2 / np.sum(corr), 1.0, m_))
    s2_expl = np.where(valid[None], noise_var_db(np.broadcast_to(ncell[None], level.shape)), np.nan)
    s2_mic = np.where(valid[None], noise_var_db(np.broadcast_to(neq[None], level.shape)), np.nan)
    s2_mm = np.where(valid, noise_var_db(neq * np.nan_to_num(m_eff, nan=1.0)[:, None]), np.nan)
    band = (freqs >= F_LO_HZ) & (freqs <= F_HI_HZ)
    lo, hi = edges[:-1], edges[1:]
    return FloorBlocks(
        edges_hz=edges,
        centres_hz=np.sqrt(lo * hi),
        level_db=level,
        n_cells=ncell,
        n_equiv=neq,
        m_eff=m_eff,
        s2_explainer=s2_expl,
        s2_mic=s2_mic,
        s2_micmean=s2_mm,
        valid=valid,
        masked_frac=float(1.0 - keep[:, band].mean()),
    )


# ---------------------------------------------------------------------------
# pooled lag moments and the OU moment fit
# ---------------------------------------------------------------------------


def _near_counts(o: np.ndarray, n_d: int) -> np.ndarray:
    """``(D, W, B)``: ``(E_d o)[w, t]``, the observed slots at distance ``d``
    from block ``t`` inside window ``w`` (``E_0 o = o``)."""
    w_, b_ = o.shape
    out = np.zeros((n_d, w_, b_))
    out[0] = o
    for d in range(1, min(n_d, b_)):
        out[d, :, d:] += o[:, :-d]
        out[d, :, :-d] += o[:, d:]
    return out


@lru_cache(maxsize=4096)
def _pair_geometry(
    ma: bytes, mb: bytes, shape: tuple[int, int], max_lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(g, n)`` for two observation masks on a ``(W, B)`` layout (W windows
    of B blocks, NaN-padded), each track centred on the mean of ALL its
    observed slots, products taken inside windows only.

    With ``P = D_o - o o^T / N`` and ``E_d`` the within-window distance-``d``
    indicator, ``g[j, d] = sum_{lag-j pairs} [Pa E_d Pb^T]_{i, i+j}``: an OU
    of unit variance and block correlation ``rho`` shared by the two tracks
    adds ``sum_d g[j, d] rho^d`` to the expected sum of their lag-``j``
    products (closed form: the four terms of ``Pa E Pb^T``). ``n[j]`` counts
    the products.
    """
    b_ = shape[1]
    fa = np.frombuffer(ma, dtype=bool).reshape(shape).astype(np.float64)
    fb = np.frombuffer(mb, dtype=bool).reshape(shape).astype(np.float64)
    na, nb = float(fa.sum()), float(fb.sum())
    ea, eb = _near_counts(fa, b_), _near_counts(fb, b_)
    s_ab = np.einsum("wt,dwt->d", fa, eb)  # o_a^T E_d o_b
    g = np.zeros((max_lag + 1, b_))
    n = np.zeros(max_lag + 1)
    for j in range(min(max_lag, b_ - 1) + 1):
        a0, b1 = fa[:, : b_ - j], fb[:, j:]
        nj = float(np.sum(a0 * b1))
        n[j] = nj
        t2 = np.einsum("wt,dwt,wt->d", a0, eb[:, :, : b_ - j], b1)  # o_a[i] (E o_b)[i] o_b[i+j]
        t3 = np.einsum("wt,dwt,wt->d", a0, ea[:, :, j:], b1)  # o_a[i] (E o_a)[i+j] o_b[i+j]
        g[j] = -t2 / nb - t3 / na + s_ab * nj / (na * nb)
        g[j, j] += nj
    return g, n


def _noise_expectation(o: np.ndarray, s: np.ndarray, max_lag: int) -> np.ndarray:
    """``(L+1,)``: expected lag-``j`` product sums of a track's OWN block noise
    (variance ``s``, independent across blocks) after the centring of
    :func:`_pair_geometry`, on a ``(W, B)`` layout."""
    b_ = o.shape[1]
    f = o.astype(np.float64)
    fs = f * s
    n_obs = float(f.sum())
    tot = float(fs.sum())
    out = np.zeros(max_lag + 1)
    for j in range(min(max_lag, b_ - 1) + 1):
        a0, b1 = f[:, : b_ - j], f[:, j:]
        nj = float(np.sum(a0 * b1))
        cross = float(np.sum(fs[:, : b_ - j] * b1) + np.sum(a0 * fs[:, j:]))
        out[j] = (tot if j == 0 else 0.0) - cross / n_obs + tot * nj / n_obs**2
    return out


@dataclass
class LagMoments:
    """Pooled centred lag products of a set of track pairs, with the exact
    expectation coefficients of each (:func:`_pair_geometry`).

    A track is a vector over a ``(W, B)`` layout -- ``W`` windows of ``B``
    blocks, NaN where not measured -- centred on the mean of all its observed
    blocks; lag products never cross a window boundary. ``W = 1`` centres each
    window on its own mean; one call with every window of a rig centres on
    the rig mean.
    """

    max_lag: int = MAX_LAG
    max_len: int = 64
    c: np.ndarray = field(init=False)  # (L+1,) sum of lag-j products
    n: np.ndarray = field(init=False)  # (L+1,) number of lag-j products
    g: np.ndarray = field(init=False)  # (L+1, D) OU expectation coefficients
    noise: np.ndarray = field(init=False)  # (L+1,) expected block-noise contribution
    s2_sum: float = field(init=False, default=0.0)  # sum of s^2 over observed blocks (auto)
    n_obs: int = field(init=False, default=0)
    dof: float = field(init=False, default=0.0)  # sum over pairs of (common blocks - 1)
    n_pairs: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.c = np.zeros(self.max_lag + 1)
        self.n = np.zeros(self.max_lag + 1)
        self.g = np.zeros((self.max_lag + 1, self.max_len))
        self.noise = np.zeros(self.max_lag + 1)

    def add(
        self,
        ya: np.ndarray,
        yb: np.ndarray | None = None,
        s2: np.ndarray | None = None,
        *,
        n_blocks: int | None = None,
        min_blocks: int = 3,
    ) -> bool:
        """Add the pair ``(ya, yb)``; ``yb=None`` is the track with itself, and
        its block noise ``s2`` enters the expectation. ``n_blocks`` is the
        layout's ``B`` (default: one window). NaN marks a missing block.
        Returns False (adding nothing) when either track has fewer than
        ``min_blocks`` observed blocks."""
        a = np.asarray(ya, dtype=np.float64).ravel()
        auto = yb is None
        b = a if auto else np.asarray(yb, dtype=np.float64).ravel()
        oa, ob = np.isfinite(a), np.isfinite(b)
        s = np.zeros(a.size) if s2 is None else np.asarray(s2, dtype=np.float64).ravel()
        if auto and s2 is not None:
            oa = oa & np.isfinite(s)
            ob = oa
        if oa.sum() < min_blocks or ob.sum() < min_blocks:
            return False
        b_ = int(n_blocks or a.size)
        if b_ > self.max_len or a.size % b_:
            raise ValueError(f"layout of {a.size} slots in windows of {b_} (max {self.max_len})")
        shape = (a.size // b_, b_)
        za = np.where(oa, a - a[oa].mean(), 0.0).reshape(shape)
        zb = np.where(ob, b - b[ob].mean(), 0.0).reshape(shape)
        g, n = _pair_geometry(oa.tobytes(), ob.tobytes(), shape, self.max_lag)
        for j in range(min(self.max_lag, b_ - 1) + 1):
            self.c[j] += float(np.sum(za[:, : b_ - j] * zb[:, j:]))
        self.n += n
        self.g[:, :b_] += g
        if auto:
            sv = np.where(oa, s, 0.0)
            self.noise += _noise_expectation(oa.reshape(shape), sv.reshape(shape), self.max_lag)
            self.s2_sum += float(sv.sum())
        self.n_obs += int(oa.sum())
        self.dof += float((oa & ob).sum()) - 1.0
        self.n_pairs += 1
        return True

    def merge(self, other: LagMoments) -> LagMoments:
        self.c += other.c
        self.n += other.n
        self.g += other.g
        self.noise += other.noise
        self.s2_sum += other.s2_sum
        self.n_obs += other.n_obs
        self.dof += other.dof
        self.n_pairs += other.n_pairs
        return self

    def expected_unit(self, rho: np.ndarray) -> np.ndarray:
        """``(len(rho), L+1)``: expected lag sums of a shared UNIT-variance OU."""
        r = np.atleast_1d(np.asarray(rho, dtype=np.float64))
        d = np.arange(self.g.shape[1])
        return (r[:, None] ** d[None, :]) @ self.g.T

    def mean_products(self) -> np.ndarray:
        return np.where(self.n > 0, self.c / np.maximum(self.n, 1), np.nan)

    def naive(self, block_s: float) -> dict[str, Any]:
        """The explainer's literal estimate, no correction for the centring:
        ``Var_b(y) - mean s^2`` (dof-corrected as for white blocks), ``rho_1 =
        Cov_1 / sigma^2`` and a log-linear fit of the mean products at lags
        1-4."""
        out: dict[str, Any] = dict(n_pairs=self.n_pairs, n_obs=self.n_obs)
        nan = float("nan")
        if self.n_pairs == 0 or self.dof <= 0:
            return out | dict(sigma2=nan, clipped=False, tau_lag1_s=nan, tau_loglin_s=nan)
        var = self.c[0] / self.dof
        s2 = self.s2_sum / self.n_obs if self.n_obs else 0.0
        sig2 = var - s2
        mean_c = self.mean_products()
        rho1 = mean_c[1] / sig2 if sig2 > 0 else nan
        out |= dict(var_db2=float(var), s2_mean_db2=float(s2), sigma2=float(max(sig2, 0.0)))
        out["clipped"] = bool(sig2 <= 0)
        out["rho1"] = float(rho1)
        out["tau_lag1_s"] = float(-block_s / math.log(rho1)) if 0 < rho1 < 1 else nan
        lags = np.arange(1, self.max_lag + 1)
        pos = np.isfinite(mean_c[1:]) & (mean_c[1:] > 0)
        if pos.sum() >= 2:
            slope = float(np.polyfit(lags[pos], np.log(mean_c[1:][pos]), 1)[0])
            out["tau_loglin_s"] = float(-block_s / slope) if slope < 0 else float("inf")
        else:
            out["tau_loglin_s"] = nan
        return out


@dataclass(frozen=True)
class OUFit:
    """``sigma2`` (dB^2) and ``tau_s`` of a moment-fitted OU (``rho = e^{-T/tau}``)."""

    sigma2: float
    tau_s: float
    rho: float
    clipped: bool  # no positive OU variance left: sigma2 set to 0
    edge: str  # "" / "tau_min" / "tau_max" (on the grid edge) / "few_lags" (not fitted)
    lags: tuple[int, ...]
    resid: float
    #: lag-0 excess per product (dB^2) the OU does not explain -- for a fit
    #: that skips lag 0, the block noise the data show (correlated noise, for
    #: a pair of tracks); zero by construction when lag 0 is solved exactly
    nugget: float = 0.0

    @property
    def sigma_db(self) -> float:
        return float(math.sqrt(max(self.sigma2, 0.0)))

    def as_dict(self) -> dict[str, Any]:
        return dict(
            sigma_db=self.sigma_db,
            sigma2_db2=self.sigma2,
            tau_s=self.tau_s,
            rho=self.rho,
            clipped=self.clipped,
            edge=self.edge,
            lags=list(self.lags),
            nugget_db2=self.nugget,
        )


def fit_ou(
    mom: LagMoments,
    block_s: float,
    *,
    lags: Sequence[int] = (1,),
    offset: np.ndarray | None = None,
    use_noise: bool = True,
    skip_lag0: bool = False,
) -> OUFit:
    """Exact method of moments for an OU of ``(sigma^2, tau)`` behind ``mom``.

    For every ``tau`` on :data:`TAU_GRID_S`, ``sigma^2`` solves the lag-0
    moment (products minus the block-noise and ``offset`` expectations, over
    the unit-OU expectation); ``tau`` then minimises the count-weighted
    squared residual of the products at ``lags``. ``lags=(1,)`` is the
    explainer's lag-1 estimate with the centring put back; ``(1, 2, 3, 4)``
    its "lags 1-4" twin. ``offset`` ``(L+1,)`` is a known expected
    contribution (the rotor-common part when the per-line residual is fitted).

    ``skip_lag0``: lag 0 is not used at all -- ``sigma^2`` is the weighted
    least-squares scale over ``lags`` (at least two) for every ``tau`` -- so
    block noise of ANY kind, including noise two tracks share within a block,
    cannot enter; the lag-0 excess comes back as :attr:`OUFit.nugget`.
    """
    off = np.zeros(mom.max_lag + 1) if offset is None else np.asarray(offset, dtype=np.float64)
    noise = mom.noise if use_noise else np.zeros_like(mom.noise)
    lag_t = tuple(int(j) for j in lags if mom.n[int(j)] > 0)
    nan = float("nan")
    target = mom.c - noise - off
    if mom.n_pairs == 0 or not np.all(np.isfinite(target)):
        return OUFit(0.0, nan, nan, False, "", lag_t, nan, nan)
    rho = np.exp(-float(block_s) / TAU_GRID_S)
    unit = mom.expected_unit(rho)  # (G, L+1)
    if skip_lag0:
        if len(lag_t) < 2:  # e.g. 1 s blocks of 4 s windows: nothing past lag 3
            return OUFit(nan, nan, nan, False, "few_lags", lag_t, nan, nan)
        w = np.array([1.0 / mom.n[j] for j in lag_t])
        u = unit[:, list(lag_t)]
        t = target[list(lag_t)]
        sig2 = np.maximum(
            (u * (w * t)).sum(axis=1) / np.maximum((u * u * w).sum(axis=1), 1e-300), 0.0
        )
        if not np.any(sig2 > 0):
            return OUFit(0.0, nan, nan, True, "", lag_t, nan, float(target[0] / mom.n[0]))
    else:
        if target[0] <= 0:
            return OUFit(0.0, nan, nan, True, "", lag_t, nan, float(target[0] / max(mom.n[0], 1)))
        sig2 = target[0] / unit[:, 0]
    loss = np.zeros(rho.size)
    for j in lag_t:
        loss += (target[j] - sig2 * unit[:, j]) ** 2 / mom.n[j]
    i = int(np.argmin(loss))
    edge = "tau_min" if i == 0 else ("tau_max" if i == rho.size - 1 else "")
    nugget = float((target[0] - sig2[i] * unit[i, 0]) / max(mom.n[0], 1))
    return OUFit(
        float(sig2[i]),
        float(TAU_GRID_S[i]),
        float(rho[i]),
        bool(sig2[i] <= 0),
        edge,
        lag_t,
        float(loss[i]),
        nugget,
    )


def ou_expected_products(
    mom: LagMoments, fit: OUFit, offset: np.ndarray | None = None
) -> np.ndarray:
    """``(L+1,)`` mean lag products the fitted OU (plus block noise, the
    fit's nugget at lag 0 and ``offset``) predicts for ``mom``'s tracks --
    the curve the figures overlay."""
    off = np.zeros(mom.max_lag + 1) if offset is None else np.asarray(offset, dtype=np.float64)
    unit = mom.expected_unit(np.array([fit.rho if np.isfinite(fit.rho) else 0.0]))[0]
    tot = fit.sigma2 * unit + mom.noise + off
    tot[0] += (fit.nugget if np.isfinite(fit.nugget) else 0.0) * mom.n[0]
    return np.where(mom.n > 0, tot / np.maximum(mom.n, 1), np.nan)


def ou_filter(z: np.ndarray, rho: float) -> np.ndarray:
    """A UNIT stationary OU along the last axis from standard normals ``z``:
    ``x_0 = z_0``, ``x_b = rho x_{b-1} + sqrt(1 - rho^2) z_b``."""
    x = np.empty(z.shape, dtype=np.float64)
    x[..., 0] = z[..., 0]
    c = math.sqrt(max(1.0 - rho * rho, 0.0))
    for b in range(1, z.shape[-1]):
        x[..., b] = rho * x[..., b - 1] + c * z[..., b]
    return x


def simulate_ou_blocks(
    rng: np.random.Generator,
    n_tracks: int,
    n_blocks: int,
    sigma_db: float,
    tau_s: float,
    block_s: float,
) -> np.ndarray:
    """``(n_tracks, n_blocks)`` stationary OU draws at the block rate, in dB --
    the explainer's recursion ``d_{b+1} = rho d_b + sqrt(1 - rho^2) sigma eps``."""
    z = np.empty((n_tracks, n_blocks))
    for b in range(n_blocks):
        z[:, b] = rng.standard_normal(n_tracks)
    return sigma_db * ou_filter(z, math.exp(-float(block_s) / float(tau_s)))


# ---------------------------------------------------------------------------
# every valid track: the line level read through its block prominence
# ---------------------------------------------------------------------------

#: A (window, rotor, order) track is CLASSIFIED -- and enters the all-track
#: estimate -- when it is valid (in band, on the grid) in at least this share
#: of the window's blocks: a line leaving the band is not wander.
MIN_VALID_FRAC = 0.8
#: A classified track is ALWAYS-UNDER when prominent (>= ``PROMINENCE_DB``) in
#: under this share of its valid blocks, INTERMITTENT from here up to
#: ``PROMINENT_BLOCK_FRAC``, RESOLVABLE above.
ALWAYS_UNDER_FRAC = 0.2
TRACK_CLASSES = ("resolvable", "intermittent", "under", "off_grid")
RESOLVABLE, INTERMITTENT, UNDER, OFF_GRID = range(4)
#: The line-free cell-over-floor ratio: every block of the always-under tracks
#: whose median prominence is under this (dB) -- floor cells with no line.
NULL_MEDIAN_DB = 1.0
#: Lower edges of the order groups the per-line sd is measured on:
#: ``[1, 3), [3, 9), [9, 25), [25, 61), [61, K]``.
ORDER_GROUP_EDGES = (1, 3, 9, 25, 61)
#: The prominence model's median map: draws, and the line-level grid (dB).
PROM_MAP_DRAWS = 20_000
PROM_MAP_MU_DB = np.arange(-30.0, 40.0001, 0.1)
#: The all-track moment table: sigma (dB, 0 first; interpolated in sigma^2)
#: and tau (s; interpolated in log tau onto ``PROM_TAU_FINE_S``).
PROM_SIGMA_GRID_DB = np.concatenate([[0.0], np.geomspace(0.25, 16.0, 32)])
PROM_TAU_GRID_S = np.geomspace(0.05, 50.0, 31)
PROM_TAU_FINE_S = np.geomspace(0.05, 50.0, 600)


def classify_tracks(prominence_db: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """``(R, K)`` index into :data:`TRACK_CLASSES` of every track of a window
    (``prominence_db``, ``valid`` ``(R, K, B)``): OFF_GRID when valid in under
    ``MIN_VALID_FRAC`` of the blocks, else by the share of its VALID blocks
    at >= ``PROMINENCE_DB``."""
    nval = valid.sum(axis=-1)
    prom = np.where(valid, prominence_db, -np.inf)
    pres = (np.nan_to_num(prom, nan=-np.inf) >= PROMINENCE_DB).sum(axis=-1)
    frac = pres / np.maximum(nval, 1)
    out = np.full(nval.shape, OFF_GRID)
    ok = nval >= MIN_VALID_FRAC * valid.shape[-1]
    out[ok & (frac >= PROMINENT_BLOCK_FRAC)] = RESOLVABLE
    out[ok & (frac >= ALWAYS_UNDER_FRAC) & (frac < PROMINENT_BLOCK_FRAC)] = INTERMITTENT
    out[ok & (frac < ALWAYS_UNDER_FRAC)] = UNDER
    return out


def order_group(orders: Any) -> np.ndarray:
    """Index into :data:`ORDER_GROUP_EDGES` of every order (``>= 1``)."""
    edges = np.asarray(ORDER_GROUP_EDGES)
    return np.searchsorted(edges, np.asarray(orders), side="right") - 1


def line_fraction(prominence_db: Any) -> np.ndarray:
    """``max(1 - 10^(-P/10), 0)``: the line's share of its cells at block
    prominence ``P`` -- the slope ``dP/dx`` of the prominence in the line level."""
    p = np.asarray(prominence_db, dtype=np.float64)
    return np.clip(1.0 - 10.0 ** (-p / 10.0), 0.0, 1.0)


@dataclass(frozen=True)
class ProminenceModel:
    """A Gaussian line level in dB, read as block prominence.

    A track's line over its local floor is ``mu + x_b + e_b + o_b`` dB: ``x``
    an OU of ``(sigma_db, tau_s)`` -- plus, when ``sigma_d_db > 0``, an
    independent OU of ``(sigma_d_db, tau_d_s)``, the rotor-common part -- ``e``
    white block noise of variance ``s2_db2``, ``o`` a known offset (the speed
    law minus the track's measured local floor deviation: the line held in
    absolute level, the floor as it was). The floor cells add a line-free
    cell-over-floor ratio drawn from ``eta`` (linear), and the block
    prominence is ``10 log10(10^((mu + x + e + o) / 10) + eta)``.
    """

    sigma_db: float
    tau_s: float
    s2_db2: float
    eta: np.ndarray
    block_s: float
    sigma_d_db: float = 0.0
    tau_d_s: float = 1.0


def line_free_eta(tracks: Sequence[np.ndarray]) -> np.ndarray:
    """The linear cell-over-floor ratios of every block of the ``tracks`` (block
    prominences, dB, of ALWAYS-UNDER tracks) whose median is under
    ``NULL_MEDIAN_DB``: :attr:`ProminenceModel.eta`."""
    quiet = [np.asarray(t, dtype=np.float64) for t in tracks if np.median(t) < NULL_MEDIAN_DB]
    if not quiet:
        raise ValueError(f"no always-under track with median prominence < {NULL_MEDIAN_DB} dB")
    return 10.0 ** (np.concatenate(quiet) / 10.0)


def prominence_median_map(
    model: ProminenceModel, rng: np.random.Generator, extra_var_db2: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """``(mu, median P)``: the model's median block prominence at line level
    ``mu`` (dB over the local floor) on :data:`PROM_MAP_MU_DB`, the Gaussian's
    variance ``sigma^2 + sigma_d^2 + s^2 + extra_var_db2``; monotone by
    construction."""
    var = model.sigma_db**2 + model.sigma_d_db**2 + model.s2_db2 + extra_var_db2
    z = rng.standard_normal(PROM_MAP_DRAWS) * math.sqrt(var)
    e = rng.choice(model.eta, PROM_MAP_DRAWS)
    mu = PROM_MAP_MU_DB
    med = np.empty(mu.size)
    for i in range(0, mu.size, 64):
        m = mu[i : i + 64, None]
        med[i : i + 64] = np.median(10.0 * np.log10(10.0 ** ((m + z) / 10.0) + e), axis=1)
    return mu, np.maximum.accumulate(med)


def prominence_line_levels(
    model: ProminenceModel,
    medians: np.ndarray,
    rng: np.random.Generator,
    extra_var_db2: float = 0.0,
) -> np.ndarray:
    """Each track's line level ``mu`` whose model median prominence is the
    track's observed median ``medians``."""
    mu_grid, med_grid = prominence_median_map(model, rng, extra_var_db2)
    return np.interp(np.asarray(medians, dtype=np.float64), med_grid, mu_grid)


def simulate_prominence(
    model: ProminenceModel,
    mus: np.ndarray,
    n_blocks: int,
    rng: np.random.Generator,
    offset: np.ndarray | None = None,
) -> np.ndarray:
    """``(len(mus), n_blocks)``: one forward-model draw of every track's block
    prominence (dB); ``offset`` ``(len(mus), n_blocks)`` is the known ``o``."""
    x = simulate_ou_blocks(rng, mus.size, n_blocks, model.sigma_db, model.tau_s, model.block_s)
    if model.sigma_d_db > 0.0:
        x = x + simulate_ou_blocks(
            rng, mus.size, n_blocks, model.sigma_d_db, model.tau_d_s, model.block_s
        )
    x = x + math.sqrt(model.s2_db2) * rng.standard_normal(x.shape)
    if offset is not None:
        x = x + offset
    return 10.0 * np.log10(10.0 ** ((mus[:, None] + x) / 10.0) + rng.choice(model.eta, x.shape))


def centred_lag_sums(x: np.ndarray, max_lag: int = MAX_LAG) -> tuple[np.ndarray, np.ndarray]:
    """``(c, n)`` ``(..., max_lag + 1)``: sums and counts of the lag-``j``
    products of every row of ``x`` (blocks on the last axis, NaN = missing),
    each row centred on the mean of its observed blocks."""
    ok = np.isfinite(x)
    cnt = ok.sum(axis=-1)
    mean = np.where(ok, x, 0.0).sum(axis=-1) / np.maximum(cnt, 1)
    z = np.where(ok, x - mean[..., None], 0.0)
    okf = ok.astype(np.float64)
    b_ = x.shape[-1]
    c = np.zeros(x.shape[:-1] + (max_lag + 1,))
    n = np.zeros_like(c)
    for j in range(min(max_lag, b_ - 1) + 1):
        c[..., j] = (z[..., : b_ - j] * z[..., j:]).sum(axis=-1)
        n[..., j] = (okf[..., : b_ - j] * okf[..., j:]).sum(axis=-1)
    return c, n


@dataclass
class ProminenceTracks:
    """Classified tracks (valid in >= ``MIN_VALID_FRAC`` of their window's
    blocks) on one ``(N, B)`` layout, rows padded with NaN to the longest
    window. Axes: ``n`` track, ``b`` block."""

    p_db: np.ndarray  # (N, B) block prominence, NaN where not valid
    offset_db: np.ndarray  # (N, B) the model's known o, centred per track; 0 where p is NaN
    window: np.ndarray  # (N,) window index
    rotor: np.ndarray  # (N,)
    order: np.ndarray  # (N,)
    cls: np.ndarray  # (N,) index into TRACK_CLASSES

    @classmethod
    def from_blocks(cls, lb: LineBlocks, speed_db: np.ndarray) -> ProminenceTracks:
        """Every classified track of one window; ``speed_db`` ``(R, B)`` is the
        line speed law per block (dB) -- ``o`` is it minus the local floor,
        centred on the track's valid blocks."""
        c = classify_tracks(lb.prominence_db, lb.valid)
        rr, kk = np.nonzero(c != OFF_GRID)
        valid = lb.valid[rr, kk]
        p = np.where(valid, lb.prominence_db[rr, kk], np.nan)
        o = np.where(valid, np.asarray(speed_db)[rr] - lb.floor_db[rr, kk], 0.0)
        o = o - o.sum(axis=1, keepdims=True) / np.maximum(valid.sum(axis=1, keepdims=True), 1)
        return cls(
            p_db=p,
            offset_db=np.where(valid, o, 0.0),
            window=np.zeros(rr.size, dtype=np.int64),
            rotor=rr.astype(np.int64),
            order=lb.orders[kk].astype(np.int64),
            cls=c[rr, kk].astype(np.int64),
        )

    @classmethod
    def stack(cls, parts: Sequence[ProminenceTracks]) -> ProminenceTracks:
        """One layout of several windows' tracks; ``window`` = position in ``parts``."""
        b_ = max(p.p_db.shape[1] for p in parts)

        def pad(a: np.ndarray, fill: float) -> np.ndarray:
            out = np.full((a.shape[0], b_), fill)
            out[:, : a.shape[1]] = a
            return out

        return cls(
            p_db=np.concatenate([pad(p.p_db, np.nan) for p in parts]),
            offset_db=np.concatenate([pad(p.offset_db, 0.0) for p in parts]),
            window=np.concatenate(
                [np.full(p.rotor.size, i, dtype=np.int64) for i, p in enumerate(parts)]
            ),
            rotor=np.concatenate([p.rotor for p in parts]),
            order=np.concatenate([p.order for p in parts]),
            cls=np.concatenate([p.cls for p in parts]),
        )

    def take(self, m: np.ndarray) -> ProminenceTracks:
        return ProminenceTracks(
            self.p_db[m],
            self.offset_db[m],
            self.window[m],
            self.rotor[m],
            self.order[m],
            self.cls[m],
        )

    def medians(self) -> np.ndarray:
        """``(N,)`` median block prominence over the valid blocks."""
        return np.nanmedian(self.p_db, axis=1)

    def weights(self) -> np.ndarray:
        """``(N,)`` the square of the line fraction at the track's median
        prominence: how much of a line-level change reaches the prominence (a
        track at 0 dB carries no line and weighs 0)."""
        return line_fraction(self.medians()) ** 2


@dataclass
class ProminenceMoments:
    """Weighted, per-track-centred lag sums of the observed block prominence
    and of the :class:`ProminenceModel` on a ``(sigma, tau)`` grid, summed per
    CELL (any grouping of the tracks, e.g. window x order group). The model's
    draws are centred per track exactly as the data, so ``sigma`` is the OU's
    STATIONARY sd; a set of cells is fitted by :func:`fit_prominence_ou`."""

    sigma_grid: np.ndarray  # (S,)
    tau_grid: np.ndarray  # (T,)
    block_s: float
    obs: np.ndarray  # (C, L+1) weighted lag-product sums
    n: np.ndarray  # (C, L+1) weighted product counts
    sim: np.ndarray  # (C, S, T, L+1) the model's expectation of obs
    weight: np.ndarray  # (C,) sum of the track weights
    n_tracks: np.ndarray  # (C,) tracks with weight > 0


def prominence_moments(
    tracks: ProminenceTracks,
    model: ProminenceModel,
    cells: np.ndarray,
    n_cells: int,
    rng: np.random.Generator,
    *,
    n_draw: int = 8,
    sigma_grid: np.ndarray = PROM_SIGMA_GRID_DB,
    tau_grid: np.ndarray = PROM_TAU_GRID_S,
    max_lag: int = MAX_LAG,
) -> ProminenceMoments:
    """The table behind :func:`fit_prominence_ou`. Each track is weighted by
    :meth:`ProminenceTracks.weights`; its line level ``mu`` is matched to its
    median at every ``sigma`` (:func:`prominence_line_levels`, the offsets'
    pooled variance added to the Gaussian); ``model.sigma_db``/``tau_s`` are
    the grid's, the rest of ``model`` is held. ``n_draw`` draws of common
    random numbers serve every grid point, so the table is smooth in both."""
    w_all = tracks.weights()
    use = w_all > 0
    p = tracks.p_db[use]
    off = tracks.offset_db[use]
    w = w_all[use]
    cid = np.asarray(cells)[use]
    med = tracks.medians()[use]
    ok = np.isfinite(p)
    n_, b_ = p.shape
    l1 = max_lag + 1

    def per_cell(v: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.bincount(cid, weights=v[:, j], minlength=n_cells) for j in range(v.shape[1])],
            axis=1,
        )

    c_obs, n_obs = centred_lag_sums(p, max_lag)
    obs, cnt = per_cell(w[:, None] * c_obs), per_cell(w[:, None] * n_obs)
    extra = float(np.mean(off[ok] ** 2)) if ok.any() else 0.0
    mus = [
        prominence_line_levels(replace(model, sigma_db=float(s)), med, rng, extra)
        for s in sigma_grid
    ]
    shape = (n_draw, n_, b_)
    zv = rng.standard_normal(shape)
    base = math.sqrt(model.s2_db2) * rng.standard_normal(shape) + off[None]
    if model.sigma_d_db > 0.0:
        rho_d = math.exp(-model.block_s / model.tau_d_s)
        base = base + model.sigma_d_db * ou_filter(rng.standard_normal(shape), rho_d)
    eta = rng.choice(model.eta, shape)
    sim = np.zeros((n_cells, sigma_grid.size, tau_grid.size, l1))
    for ti, tau in enumerate(tau_grid):
        uv = ou_filter(zv, math.exp(-model.block_s / float(tau)))
        for si, s in enumerate(sigma_grid):
            x = mus[si][None, :, None] + float(s) * uv + base
            pr = np.where(ok[None], 10.0 * np.log10(10.0 ** (x / 10.0) + eta), np.nan)
            c_sim, _ = centred_lag_sums(pr, max_lag)
            sim[:, si, ti] = per_cell(w[:, None] * c_sim.mean(axis=0))
    return ProminenceMoments(
        sigma_grid=np.asarray(sigma_grid, dtype=np.float64),
        tau_grid=np.asarray(tau_grid, dtype=np.float64),
        block_s=float(model.block_s),
        obs=obs,
        n=cnt,
        sim=sim,
        weight=np.bincount(cid, weights=w, minlength=n_cells),
        n_tracks=np.bincount(cid, minlength=n_cells),
    )


def _table_at_tau(sim: np.ndarray, tau_grid: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """``(S, len(taus), L+1)``: ``sim`` ``(S, T, L+1)`` linear in log tau."""
    lt, lf = np.log(tau_grid), np.log(np.clip(taus, tau_grid[0], tau_grid[-1]))
    i = np.clip(np.searchsorted(lt, lf) - 1, 0, lt.size - 2)
    fr = ((lf - lt[i]) / (lt[i + 1] - lt[i]))[None, :, None]
    return sim[:, i] * (1.0 - fr) + sim[:, i + 1] * fr


def _solve_sigma(
    table: np.ndarray, sigma_grid: np.ndarray, target0: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per column of ``table`` ``(S, F, L+1)``: the ``sigma^2`` whose lag-0
    expectation is ``target0`` (linear in ``sigma^2`` between grid points),
    the expected products there ``(F, L+1)``, and the clip / top-edge flags."""
    c0 = np.maximum.accumulate(table[..., 0], axis=0)  # (S, F)
    s_ = sigma_grid.size
    k = (c0 < target0).sum(axis=0)
    lo = np.clip(k - 1, 0, s_ - 2)
    cols = np.arange(c0.shape[1])
    a, b = c0[lo, cols], c0[lo + 1, cols]
    fr = np.clip((target0 - a) / np.where(b > a, b - a, 1.0), 0.0, 1.0)
    g2 = sigma_grid**2
    sig2 = g2[lo] + fr * (g2[lo + 1] - g2[lo])
    prod = table[lo, cols] * (1.0 - fr[:, None]) + table[lo + 1, cols] * fr[:, None]
    return sig2, prod, k == 0, k == s_


def _cell_sums(mom: ProminenceMoments, cell_weight: np.ndarray | None) -> tuple[np.ndarray, ...]:
    m = np.ones(mom.obs.shape[0]) if cell_weight is None else np.asarray(cell_weight, float)
    return m @ mom.obs, m @ mom.n, np.tensordot(m, mom.sim, axes=1)


def fit_prominence_ou(
    mom: ProminenceMoments,
    cell_weight: np.ndarray | None = None,
    *,
    lags: Sequence[int] = (1, 2, 3, 4),
    tau_s: float | None = None,
) -> OUFit:
    """Simulated method of moments on the cells ``cell_weight`` ``(C,)`` of
    ``mom`` (multiplicities: a window bootstrap repeats a window's cells):
    for every ``tau`` (``PROM_TAU_FINE_S``, or the given ``tau_s``) ``sigma``
    solves the weighted lag-0 sum, and ``tau`` minimises the count-weighted
    squared residual of the lag sums at ``lags`` -- :func:`fit_ou` with the
    model's expectation in place of the linear one."""
    obs, cnt, sim = _cell_sums(mom, cell_weight)
    lag_t = tuple(int(j) for j in lags if cnt[int(j)] > 0)
    nan = float("nan")
    if cnt[0] <= 0 or obs[0] <= 0:
        return OUFit(0.0, nan, nan, cnt[0] > 0, "", lag_t, nan)
    taus = PROM_TAU_FINE_S if tau_s is None else np.array([float(tau_s)])
    sig2, prod, clip, top = _solve_sigma(
        _table_at_tau(sim, mom.tau_grid, taus), mom.sigma_grid, float(obs[0])
    )
    loss = np.zeros(taus.size)
    for j in lag_t:
        loss += (obs[j] - prod[:, j]) ** 2 / cnt[j]
    i = int(np.argmin(loss))
    edge = ""
    if tau_s is None and lag_t and i in (0, taus.size - 1):
        edge = "tau_min" if i == 0 else "tau_max"
    if top[i]:
        edge = "sigma_max"
    tau = float(taus[i])
    return OUFit(
        0.0 if clip[i] else float(sig2[i]),
        tau,
        math.exp(-mom.block_s / tau),
        bool(clip[i]),
        edge,
        lag_t,
        float(loss[i]),
    )


def prominence_products(
    mom: ProminenceMoments, fit: OUFit, cell_weight: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``(observed, model)`` ``(L+1,)`` weighted mean lag products of the
    cells at ``fit`` (the model interpolated at its ``sigma``, ``tau``)."""
    obs, cnt, sim = _cell_sums(mom, cell_weight)
    tab = _table_at_tau(sim, mom.tau_grid, np.array([fit.tau_s]))[:, 0]  # (S, L+1)
    g2 = mom.sigma_grid**2
    model = np.array([np.interp(fit.sigma2, g2, tab[:, j]) for j in range(tab.shape[1])])
    den = np.where(cnt > 0, cnt, np.nan)
    return obs / den, model / den


__all__ = [
    "DB_PER_NEPER",
    "FloorBlocks",
    "LagMoments",
    "LineBlocks",
    "OUFit",
    "ProminenceModel",
    "ProminenceMoments",
    "ProminenceTracks",
    "block_line_power_db",
    "cell_power_correlation",
    "centred_lag_sums",
    "classify_tracks",
    "comb_mask",
    "control_band_edges",
    "correlated_cell_count",
    "fit_ou",
    "fit_prominence_ou",
    "frame_blocks",
    "hann_capture",
    "line_fraction",
    "line_free_eta",
    "measure_floor",
    "measure_lines",
    "noise_var_db",
    "order_group",
    "ou_expected_products",
    "ou_filter",
    "overlap_inflation",
    "prominence_line_levels",
    "prominence_median_map",
    "prominence_moments",
    "prominence_products",
    "rel_var_to_db2",
    "simulate_ou_blocks",
    "simulate_prominence",
    "third_octave_edges",
    "within_block_mean_var",
]
