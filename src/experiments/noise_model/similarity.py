"""Real-vs-synthetic similarity measures for multichannel rotor noise.

The DREGON comparison score so far, ``revised_eval.ltas_deviation_db``, reads
mic 0 only, on an 8192-point Welch, as seven band means between 100 and 7900
Hz. It cannot see below 100 Hz, cannot tell lines from floor, ignores time
structure (gusts, bursts) and the other microphones. Each function below
measures one of those blind spots. Every function takes the REFERENCE
``real (M, T)`` and the candidate ``arm (M, T)`` at ``sr`` (16 kHz) and returns
a dict: a ``summary`` of named scalars plus the per-band / per-mic arrays it
is made of. Signed differences are always ``arm - real``.

1. :func:`line_floor_lsd` — log-spectral distance on LINE bins and FLOOR cells
   separately, per octave band (fine 4096-point Welch).
2. :func:`order_level_error` — order-tracked line level (absolute peak and
   prominence over its local floor) per rotor and integer order.
3. :func:`level_distribution` — distributions of 128 ms band levels
   (Wasserstein-1, spread) and the modulation spectrum of the band envelopes.
4. :func:`spatial` — per-band level spread over mics and mean pairwise
   magnitude-squared coherence.
5. :func:`itakura_saito` — Itakura-Saito divergence of the long-term fine
   PSDs, over linear bins and octave-balanced.
6. :func:`spectro_temporal` — Fréchet distance of the frame third-octave
   level vectors, cross-band co-modulation and per-band level sd.

The rotor label ``labels (R, T)`` (rev/s on the audio grid) places the lines.
The arm is rendered on the real label, so one label serves both sides and the
two inputs have the same length.

Pure numpy/scipy plus :func:`experiments.stochastic_fit.diagnostics.floor_cells`
(the floor-cell rule of the flow-noise diagnostics).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import sqrtm
from scipy.signal import welch
from scipy.stats import wasserstein_distance

from experiments.stochastic_fit.diagnostics import floor_cells

__all__ = [
    "BAND_NAMES",
    "OCTAVE_EDGES",
    "ORDER_GROUPS",
    "SR",
    "THIRD_OCTAVE_EDGES",
    "battery",
    "itakura_saito",
    "level_distribution",
    "line_floor_lsd",
    "order_level_error",
    "spatial",
    "spectro_temporal",
]

SR = 16000
#: Octave bands 31.25-62.5 Hz ... 4-8 kHz.
OCTAVE_EDGES = 31.25 * 2.0 ** np.arange(9)
BAND_NAMES = tuple(
    f"{lo:g}-{hi:g}" for lo, hi in zip(OCTAVE_EDGES[:-1], OCTAVE_EDGES[1:], strict=True)
)
#: The fine analysis: periodic Hann, 4096 points (3.9 Hz bins), 75 % overlap.
FINE_NFFT, FINE_HOP = 4096, 1024
#: A bin is a LINE bin within this many orders of an integer order of a rotor.
LINE_HALF_ORDER = 0.05
#: A bin enters the floor comparison when it is a floor cell in at least this
#: fraction of the frames (the lines move; the cells follow them).
FLOOR_MIN_FRACTION = 0.25
#: Rotors slower than this carry no lines (``floor_cells``' own threshold).
MIN_RPS = 5.0
#: Order tracking: peak within +-0.08 order, local floor 0.3-0.5 orders away.
ORDER_PEAK_HALF = 0.08
ORDER_SIDE = (0.3, 0.5)
ORDER_K_MAX = 100
ORDER_F_MAX = 7900.0
ORDER_GROUPS: tuple[tuple[int, int], ...] = ((1, 2), (3, 10), (11, 30), (31, 100))
#: Frame band levels: 128 ms periodic-Hann frames, 16 ms hop.
LEVEL_NFFT, LEVEL_HOP = 2048, 256
#: Modulation spectrum: octave bins 0.5-32 Hz (the top bin ends at the frame
#: rate's Nyquist, 31.25 Hz); Welch segments of at most 128 frames (2 s), zero
#: padded to 256 so every octave bin holds at least two frequencies.
MOD_EDGES = 0.5 * 2.0 ** np.arange(7)
MOD_NPERSEG, MOD_NFFT = 128, 256
#: Itakura-Saito band.
IS_BAND = (31.25, 7900.0)
#: Third-octave bands 31.5 Hz - 7.9 kHz (24 bands; the last edge capped at 7.9 kHz).
THIRD_OCTAVE_EDGES = np.minimum(31.5 * 2.0 ** (np.arange(25) / 3.0), 7900.0)
#: Ridge (dB^2) on the frame-level covariances of the Fréchet distance.
FRECHET_RIDGE = 1e-6
_TINY = 1e-30


# ── shared front end ───────────────────────────────────────────────────────


def _db(x: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(x, _TINY))


def _stft(x: np.ndarray, nfft: int, hop: int) -> np.ndarray:
    """``(M, N, F)`` complex STFT, periodic Hann, no padding (complete frames only)."""
    xx = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if xx.shape[-1] < nfft:
        raise ValueError(f"{xx.shape[-1]} samples is shorter than one {nfft}-point frame")
    w = np.hanning(nfft + 1)[:nfft]
    frames = np.lib.stride_tricks.sliding_window_view(xx, nfft, axis=-1)[:, ::hop]
    return np.fft.rfft(frames * w, axis=-1)


def _frame_labels(labels: np.ndarray, n_frames: int, nfft: int, hop: int) -> np.ndarray:
    """``(R, N)`` the label's mean over each frame's samples."""
    lab = np.atleast_2d(np.asarray(labels, dtype=np.float64))
    c = np.concatenate([np.zeros((lab.shape[0], 1)), np.cumsum(lab, axis=1)], axis=1)
    s = np.arange(n_frames) * hop
    return (c[:, s + nfft] - c[:, s]) / float(nfft)


def _band_index(freqs: np.ndarray) -> np.ndarray:
    """``(F,)`` octave band of each bin, -1 outside 31.25 Hz - 8 kHz."""
    idx = np.searchsorted(OCTAVE_EDGES, freqs, side="right") - 1
    idx[(freqs < OCTAVE_EDGES[0]) | (freqs >= OCTAVE_EDGES[-1])] = -1
    return idx


def _band_mean(values: np.ndarray, band: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``(..., B)`` mean of ``values (..., F)`` over ``mask`` bins of each band (NaN if empty)."""
    out = np.full((*values.shape[:-1], len(BAND_NAMES)), np.nan)
    for b in range(len(BAND_NAMES)):
        sel = mask & (band == b)
        if sel.any():
            out[..., b] = values[..., sel].mean(axis=-1)
    return out


def _nanmean(x: Any) -> float:
    a = np.asarray(x, dtype=np.float64)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def _line_mask(freqs: np.ndarray, rps: np.ndarray) -> np.ndarray:
    """``(F,)`` bins within :data:`LINE_HALF_ORDER` of ``k * rps_r`` (``k >= 1``) for any rotor."""
    mask = np.zeros(freqs.shape, dtype=bool)
    for fr in np.atleast_1d(rps):
        if fr < MIN_RPS:
            continue
        kk = freqs / fr
        k = np.round(kk)
        mask |= (k >= 1) & (np.abs(kk - k) <= LINE_HALF_ORDER)
    return mask


def _smear_fraction(freqs: np.ndarray, fbar: np.ndarray, flab: np.ndarray) -> np.ndarray:
    """``(F,)`` per line bin: the fraction of frames whose label puts the bin's
    order more than :data:`LINE_HALF_ORDER` away from where the window-mean
    label puts it (the largest over the rotors whose line covers the bin)."""
    out = np.zeros(freqs.shape)
    for r, fr in enumerate(fbar):
        if fr < MIN_RPS:
            continue
        kk = freqs / fr
        k = np.round(kk)
        cover = (k >= 1) & (np.abs(kk - k) <= LINE_HALF_ORDER)
        moved = np.abs(freqs[None, :] / np.maximum(flab[r][:, None], MIN_RPS) - kk[None, :])
        frac = (moved > LINE_HALF_ORDER).mean(axis=0)
        out = np.where(cover, np.maximum(out, frac), out)
    return out


def _floor_psd(power: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """``(M, F)`` mean power over the frames where each bin is a floor cell
    (``cells (N, F)``)."""
    acc = np.einsum("mnf,nf->mf", power, cells.astype(np.float64))
    return acc / np.maximum(cells.sum(axis=0), 1)


# ── 1. line / floor log-spectral distance ──────────────────────────────────


def line_floor_lsd(
    real: np.ndarray, arm: np.ndarray, labels: np.ndarray, *, sr: int = SR
) -> dict[str, Any]:
    """Per mic and octave band, the mean ``|dB(arm) - dB(real)|`` of the fine
    Welch PSD over LINE bins (window-mean label, +-0.05 order) and, separately,
    over FLOOR cells (frame-wise :func:`floor_cells`, 5 Hz guard; a bin's floor
    PSD is its mean over the frames where it is a floor cell, kept when that
    holds on at least :data:`FLOOR_MIN_FRACTION` of the frames).
    ``smear_fraction`` flags line bins whose label moves more than 0.05 order
    during the window."""
    pr = np.abs(_stft(real, FINE_NFFT, FINE_HOP)) ** 2
    pa = np.abs(_stft(arm, FINE_NFFT, FINE_HOP)) ** 2
    freqs = np.fft.rfftfreq(FINE_NFFT, 1.0 / sr)
    band = _band_index(freqs)
    in_band = band >= 0
    flab = _frame_labels(labels, pr.shape[1], FINE_NFFT, FINE_HOP)
    fbar = np.atleast_2d(np.asarray(labels, dtype=np.float64)).mean(axis=1)
    lines = _line_mask(freqs, fbar) & in_band
    d_line = _db(pa.mean(axis=1)) - _db(pr.mean(axis=1))  # (M, F)
    cells = np.stack(
        [
            floor_cells(freqs, flab[:, i], OCTAVE_EDGES[0], OCTAVE_EDGES[-1])
            for i in range(flab.shape[1])
        ]
    )  # (N, F)
    floor = (cells.mean(axis=0) >= FLOOR_MIN_FRACTION) & in_band
    d_floor = _db(_floor_psd(pa, cells)) - _db(_floor_psd(pr, cells))
    smear = _smear_fraction(freqs, fbar, flab)
    line_abs = _band_mean(np.abs(d_line), band, lines)
    floor_abs = _band_mean(np.abs(d_floor), band, floor)
    line_signed = _band_mean(d_line, band, lines)
    floor_signed = _band_mean(d_floor, band, floor)
    return dict(
        bands=list(BAND_NAMES),
        line_abs_db=line_abs,
        line_signed_db=line_signed,
        floor_abs_db=floor_abs,
        floor_signed_db=floor_signed,
        line_bins=np.array([int((lines & (band == b)).sum()) for b in range(len(BAND_NAMES))]),
        floor_bins=np.array([int((floor & (band == b)).sum()) for b in range(len(BAND_NAMES))]),
        smear_fraction=_band_mean(smear, band, lines),
        summary=dict(
            line_abs_db=_nanmean(line_abs),
            floor_abs_db=_nanmean(floor_abs),
            line_signed_db=_nanmean(line_signed),
            floor_signed_db=_nanmean(floor_signed),
            smear_fraction=float(smear[lines].mean()) if lines.any() else float("nan"),
        ),
    )


# ── 2. order-tracked line levels ───────────────────────────────────────────


def _order_levels(
    power: np.ndarray, freqs: np.ndarray, flab: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(M, R, K)`` frame-mean peak level (dB) within +-0.08 order of ``k f_r``
    and its prominence over the median power 0.3-0.5 orders away. An order is
    NaN unless its side window stays below :data:`ORDER_F_MAX` in every frame."""
    m_, n_, _ = power.shape
    r_ = flab.shape[0]
    ks = np.arange(1, ORDER_K_MAX + 1, dtype=np.float64)
    peak = np.zeros((m_, r_, ORDER_K_MAX))
    prom = np.zeros((m_, r_, ORDER_K_MAX))
    valid = np.ones((r_, ORDER_K_MAX), dtype=bool)
    lo_s, hi_s = ORDER_SIDE
    for r in range(r_):
        for n in range(n_):
            fr = flab[r, n]
            if fr < MIN_RPS:
                valid[r] = False
                continue
            valid[r] &= (ks + hi_s) * fr < ORDER_F_MAX
            kmax = int(valid[r].sum())
            if kmax == 0:
                continue
            k = ks[:kmax]
            ss = np.searchsorted
            p_lo = ss(freqs, (k - ORDER_PEAK_HALF) * fr, side="left")
            p_hi = ss(freqs, (k + ORDER_PEAK_HALF) * fr, side="right")
            centre = np.rint(k * fr / (freqs[1] - freqs[0])).astype(int)
            s1 = (ss(freqs, (k - hi_s) * fr), ss(freqs, (k - lo_s) * fr, side="right"))
            s2 = (ss(freqs, (k + lo_s) * fr), ss(freqs, (k + hi_s) * fr, side="right"))
            col = power[:, n, :]
            for i in range(kmax):
                a, b = (p_lo[i], p_hi[i]) if p_hi[i] > p_lo[i] else (centre[i], centre[i] + 1)
                pk = col[:, a:b].max(axis=1)
                side = np.concatenate(
                    [col[:, s1[0][i] : s1[1][i]], col[:, s2[0][i] : s2[1][i]]], axis=1
                )
                pk_db = _db(pk)
                peak[:, r, i] += pk_db
                prom[:, r, i] += pk_db - _db(np.median(side, axis=1))
    peak /= n_
    prom /= n_
    peak[:, ~valid] = np.nan
    prom[:, ~valid] = np.nan
    return peak, prom


def _group_errors(delta: np.ndarray) -> dict[str, Any]:
    """``delta (M, R, K)`` → per order group ``|Δ|`` (mean over mics and orders)
    per rotor ``(R, G)`` and overall ``(G,)``, the signed overall ``(G,)`` and the grand mean."""
    r_ = delta.shape[1]
    per_rotor = np.full((r_, len(ORDER_GROUPS)), np.nan)
    overall = np.full(len(ORDER_GROUPS), np.nan)
    signed = np.full(len(ORDER_GROUPS), np.nan)
    for g, (lo, hi) in enumerate(ORDER_GROUPS):
        seg = delta[:, :, lo - 1 : hi]
        for r in range(r_):
            per_rotor[r, g] = _nanmean(np.abs(seg[:, r]))
        overall[g] = _nanmean(np.abs(seg))
        signed[g] = _nanmean(seg)
    return dict(
        per_rotor_abs_db=per_rotor,
        abs_db=overall,
        signed_db=signed,
        mean_abs_db=_nanmean(np.abs(delta)),
    )


def order_level_error(
    real: np.ndarray, arm: np.ndarray, labels: np.ndarray, *, sr: int = SR
) -> dict[str, Any]:
    """Order-tracked line level per mic, rotor and integer order ``k`` (1-100,
    below 7.9 kHz): per fine frame and rotor the peak power within +-0.08 order
    of ``k f_r`` (the frame-mean label) — its absolute level and its
    prominence over the median 0.3-0.5 orders away — averaged over frames in
    dB. ``|Δ|`` in order groups 1-2, 3-10, 11-30, 31-100, per rotor and
    overall. At high orders the four rotors' combs interleave, so the peak
    window can pick a neighbour rotor's line; that is the same on both sides."""
    freqs = np.fft.rfftfreq(FINE_NFFT, 1.0 / sr)
    pr = np.abs(_stft(real, FINE_NFFT, FINE_HOP)) ** 2
    pa = np.abs(_stft(arm, FINE_NFFT, FINE_HOP)) ** 2
    flab = _frame_labels(labels, pr.shape[1], FINE_NFFT, FINE_HOP)
    peak_r, prom_r = _order_levels(pr, freqs, flab)
    peak_a, prom_a = _order_levels(pa, freqs, flab)
    peak = _group_errors(peak_a - peak_r)
    prom = _group_errors(prom_a - prom_r)
    groups = [f"k{lo}-{hi}" for lo, hi in ORDER_GROUPS]
    summary = dict(peak_abs_db=peak["mean_abs_db"], prominence_abs_db=prom["mean_abs_db"])
    for g, name in enumerate(groups):
        summary[f"peak_abs_db_{name}"] = float(peak["abs_db"][g])
        summary[f"prominence_abs_db_{name}"] = float(prom["abs_db"][g])
    return dict(
        groups=groups,
        peak=peak,
        prominence=prom,
        real_peak_db=peak_r.mean(axis=0),
        arm_peak_db=peak_a.mean(axis=0),
        real_prominence_db=prom_r.mean(axis=0),
        arm_prominence_db=prom_a.mean(axis=0),
        summary=summary,
    )


# ── 3. level distributions and modulation spectrum ─────────────────────────


def _frame_band_power(x: np.ndarray, sr: int) -> np.ndarray:
    """``(M, N, B)`` linear band power of 128 ms frames at a 16 ms hop."""
    p = np.abs(_stft(x, LEVEL_NFFT, LEVEL_HOP)) ** 2
    band = _band_index(np.fft.rfftfreq(LEVEL_NFFT, 1.0 / sr))
    return np.stack([p[..., band == b].sum(axis=-1) for b in range(len(BAND_NAMES))], axis=-1)


def _mod_spectrum(env: np.ndarray, frame_rate: float) -> np.ndarray:
    """``(M, B, J)`` mean Welch PSD of ``env (M, N, B)`` (already centred) per modulation octave."""
    n = env.shape[1]
    nper = min(MOD_NPERSEG, n)
    f, p = welch(
        env,
        fs=frame_rate,
        window="hann",
        nperseg=nper,
        noverlap=nper // 2,
        nfft=max(MOD_NFFT, nper),
        detrend=False,
        axis=1,
    )  # p: (M, Fm, B)
    out = np.full((env.shape[0], env.shape[2], len(MOD_EDGES) - 1), np.nan)
    for j, (lo, hi) in enumerate(zip(MOD_EDGES[:-1], MOD_EDGES[1:], strict=True)):
        sel = (f >= lo) & (f < hi)
        if sel.any():
            out[:, :, j] = p[:, sel, :].mean(axis=1)
    return out


def level_distribution(real: np.ndarray, arm: np.ndarray, *, sr: int = SR) -> dict[str, Any]:
    """Per mic and octave band, the 128 ms / 16 ms-hop frame band levels (dB):
    the 1-Wasserstein distance between real and arm (absolute levels, and
    ``w1_centered`` with each side's mean removed — shape only), and the
    differences in sd and in p90 - p10. Plus the modulation spectrum: per
    band the PSD of the frame envelope on 0.5-32 Hz in octave bins, ``|ΔdB|``,
    for the envelope in linear power with its mean removed (``mod_abs``; a
    level offset of x dB shows as 2x dB) and for the envelope divided by its
    mean (``mod_norm``; the modulation depth, level-free)."""
    er = _frame_band_power(real, sr)
    ea = _frame_band_power(arm, sr)
    lr, la = _db(er), _db(ea)
    m_, _, b_ = lr.shape
    w1 = np.zeros((m_, b_))
    w1c = np.zeros((m_, b_))
    for m in range(m_):
        for b in range(b_):
            r, a = lr[m, :, b], la[m, :, b]
            w1[m, b] = wasserstein_distance(r, a)
            w1c[m, b] = wasserstein_distance(r - r.mean(), a - a.mean())
    d_sd = la.std(axis=1) - lr.std(axis=1)

    def spread(x: np.ndarray) -> np.ndarray:
        p90, p10 = np.percentile(x, [90.0, 10.0], axis=1)
        return p90 - p10

    d_spread = spread(la) - spread(lr)
    rate = float(sr) / LEVEL_HOP
    mr, ma = er.mean(axis=1, keepdims=True), ea.mean(axis=1, keepdims=True)
    mod_abs = _db(_mod_spectrum(ea - ma, rate)) - _db(_mod_spectrum(er - mr, rate))
    mod_norm = _db(_mod_spectrum(ea / ma - 1.0, rate)) - _db(_mod_spectrum(er / mr - 1.0, rate))
    return dict(
        bands=list(BAND_NAMES),
        mod_bands_hz=[
            [float(lo), float(min(hi, rate / 2))]
            for lo, hi in zip(MOD_EDGES[:-1], MOD_EDGES[1:], strict=True)
        ],
        w1_db=w1,
        w1_centered_db=w1c,
        sd_diff_db=d_sd,
        spread_diff_db=d_spread,
        real_sd_db=lr.std(axis=1),
        real_spread_db=spread(lr),
        mod_abs_diff_db=mod_abs,
        mod_norm_diff_db=mod_norm,
        summary=dict(
            w1_db=float(w1.mean()),
            w1_centered_db=float(w1c.mean()),
            sd_abs_diff_db=float(np.abs(d_sd).mean()),
            sd_signed_diff_db=float(d_sd.mean()),
            spread_abs_diff_db=float(np.abs(d_spread).mean()),
            spread_signed_diff_db=float(d_spread.mean()),
            mod_abs_abs_db=_nanmean(np.abs(mod_abs)),
            mod_norm_abs_db=_nanmean(np.abs(mod_norm)),
            mod_norm_signed_db=_nanmean(mod_norm),
        ),
    )


# ── 4. spatial structure ───────────────────────────────────────────────────


def _spatial_stats(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """``(B,)`` sd over mics of the band level (dB) and ``(B,)`` mean MSC over mic pairs."""
    z = _stft(x, FINE_NFFT, FINE_HOP)  # (M, N, F)
    freqs = np.fft.rfftfreq(FINE_NFFT, 1.0 / sr)
    band = _band_index(freqs)
    in_band = band >= 0
    psd = (np.abs(z) ** 2).mean(axis=1)  # (M, F)
    level = _db(_band_mean(psd, band, in_band))  # (M, B), power mean within a band
    csd = np.einsum("inf,jnf->ijf", z, np.conj(z)) / z.shape[1]
    iu, ju = np.triu_indices(z.shape[0], k=1)
    msc = np.abs(csd[iu, ju]) ** 2 / np.maximum(psd[iu] * psd[ju], _TINY)  # (P, F)
    return level.std(axis=0), _band_mean(msc, band, in_band).mean(axis=0)


def spatial(real: np.ndarray, arm: np.ndarray, *, sr: int = SR) -> dict[str, Any]:
    """Per octave band: the spread of the band level over mics (sd, dB) and the
    mean magnitude-squared coherence over all mic pairs — Welch's coherence
    estimator (``scipy.signal.coherence``'s) on the fine 4096-point periodic
    Hann frames at 75 % overlap. MSC is biased up by about ``1/n_frames``; the
    two sides of a comparison should have equal length."""
    sd_r, msc_r = _spatial_stats(real, sr)
    sd_a, msc_a = _spatial_stats(arm, sr)
    d_sd, d_msc = sd_a - sd_r, msc_a - msc_r
    return dict(
        bands=list(BAND_NAMES),
        real_level_sd_db=sd_r,
        arm_level_sd_db=sd_a,
        level_sd_diff_db=d_sd,
        real_msc=msc_r,
        arm_msc=msc_a,
        msc_diff=d_msc,
        summary=dict(
            level_sd_abs_diff_db=float(np.abs(d_sd).mean()),
            level_sd_signed_diff_db=float(d_sd.mean()),
            msc_abs_diff=float(np.abs(d_msc).mean()),
            msc_signed_diff=float(d_msc.mean()),
        ),
    )


# ── 5. Itakura-Saito ───────────────────────────────────────────────────────


def itakura_saito(real: np.ndarray, arm: np.ndarray, *, sr: int = SR) -> dict[str, Any]:
    """Per mic, ``P/Q - log(P/Q) - 1`` per fine-PSD bin in 31.25 Hz - 7.9 kHz
    with ``P`` real and ``Q`` arm (long-term, whole-window PSDs: time-blind),
    and its symmetric version ``(IS(P|Q) + IS(Q|P)) / 2``, aggregated two ways:
    the mean over LINEAR bins (``is_div``; dominated by 2-8 kHz, which hold
    most bins) and the OCTAVE-BALANCED mean of the eight per-octave means
    (``is_octave``). Summaries are the means over mics."""
    freqs = np.fft.rfftfreq(FINE_NFFT, 1.0 / sr)
    sel = (freqs >= IS_BAND[0]) & (freqs <= IS_BAND[1])
    band = _band_index(freqs[sel])
    p = np.maximum((np.abs(_stft(real, FINE_NFFT, FINE_HOP)) ** 2).mean(axis=1)[:, sel], _TINY)
    q = np.maximum((np.abs(_stft(arm, FINE_NFFT, FINE_HOP)) ** 2).mean(axis=1)[:, sel], _TINY)
    ratio = p / q
    c_pq = ratio - np.log(ratio) - 1.0
    c_qp = 1.0 / ratio + np.log(ratio) - 1.0
    d_pq, d_qp = c_pq.mean(axis=1), c_qp.mean(axis=1)
    sym = 0.5 * (d_pq + d_qp)
    every = np.ones(band.shape, dtype=bool)
    oct_pq = _band_mean(c_pq, band, every)  # (M, B)
    oct_sym = _band_mean(0.5 * (c_pq + c_qp), band, every)
    return dict(
        bands=list(BAND_NAMES),
        is_real_arm=d_pq,
        is_arm_real=d_qp,
        is_symmetric=sym,
        is_octave_real_arm=oct_pq,
        is_octave_symmetric=oct_sym,
        summary=dict(
            is_div=float(d_pq.mean()),
            is_symmetric=float(sym.mean()),
            is_octave=float(oct_pq.mean()),
            is_octave_symmetric=float(oct_sym.mean()),
        ),
    )


# ── 6. spectro-temporal: joint distribution of frame band levels ───────────


def _third_octave_levels(x: np.ndarray, sr: int) -> np.ndarray:
    """``(M, N, B3)`` frame log-power (dB) in the :data:`THIRD_OCTAVE_EDGES`
    bands, 128 ms periodic-Hann frames at a 16 ms hop."""
    p = np.abs(_stft(x, LEVEL_NFFT, LEVEL_HOP)) ** 2
    f = np.fft.rfftfreq(LEVEL_NFFT, 1.0 / sr)
    cols = []
    for lo, hi in zip(THIRD_OCTAVE_EDGES[:-1], THIRD_OCTAVE_EDGES[1:], strict=True):
        sel = (f >= lo) & (f < hi)
        if not sel.any():
            raise ValueError(
                f"third-octave band {lo:.1f}-{hi:.1f} Hz holds no {LEVEL_NFFT}-point bin"
            )
        cols.append(p[..., sel].sum(axis=-1))
    return _db(np.stack(cols, axis=-1))


def _frechet(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """``(mean term, covariance term)`` of the Fréchet distance between Gaussians
    fitted to the rows of ``a`` and ``b`` (``(N, D)``), ridge
    :data:`FRECHET_RIDGE` on both covariances."""
    ridge = FRECHET_RIDGE * np.eye(a.shape[1])
    ca = np.cov(a, rowvar=False) + ridge
    cb = np.cov(b, rowvar=False) + ridge
    sa = np.real(sqrtm(ca))
    cross = np.real(sqrtm(sa @ cb @ sa))
    mean_term = float(np.sum((a.mean(axis=0) - b.mean(axis=0)) ** 2))
    return mean_term, float(np.trace(ca + cb - 2.0 * cross))


def _offdiag_mean(r: np.ndarray) -> float:
    return float(r[~np.eye(r.shape[0], dtype=bool)].mean())


def spectro_temporal(real: np.ndarray, arm: np.ndarray, *, sr: int = SR) -> dict[str, Any]:
    """Time-aware spectral similarity, per mic, on 128 ms / 16 ms-hop frame
    log-power levels in ~24 third-octave bands (31.5 Hz - 7.9 kHz):

    - the Fréchet distance (dB²) between Gaussians fitted to the real and arm
      frame level VECTORS, split into its mean term ``|mu_r - mu_a|^2`` (the
      long-term spectrum) and its covariance term (how the band levels vary
      and co-vary over time);
    - cross-band co-modulation: the band-by-band correlation matrix of the
      frame levels; ``||R_a - R_r||_F / n_bands`` and each side's mean
      off-diagonal correlation (broadband bursts raise it);
    - per band the sd of the frame levels, ``arm - real``.

    Summaries are means over mics."""
    lr = _third_octave_levels(real, sr)
    la = _third_octave_levels(arm, sr)
    m_, _, b_ = lr.shape
    mean_t = np.zeros(m_)
    cov_t = np.zeros(m_)
    corr_d = np.zeros(m_)
    off_r = np.zeros(m_)
    off_a = np.zeros(m_)
    for m in range(m_):
        mean_t[m], cov_t[m] = _frechet(lr[m], la[m])
        rr = np.corrcoef(lr[m], rowvar=False)
        ra = np.corrcoef(la[m], rowvar=False)
        corr_d[m] = np.linalg.norm(ra - rr) / b_
        off_r[m], off_a[m] = _offdiag_mean(rr), _offdiag_mean(ra)
    d_sd = la.std(axis=1) - lr.std(axis=1)  # (M, B3)
    return dict(
        band_edges_hz=THIRD_OCTAVE_EDGES,
        frechet_mean_term_db2=mean_t,
        frechet_cov_term_db2=cov_t,
        comod_frobenius=corr_d,
        real_offdiag_corr=off_r,
        arm_offdiag_corr=off_a,
        sd_diff_db=d_sd,
        summary=dict(
            frechet_db2=float((mean_t + cov_t).mean()),
            frechet_mean_term_db2=float(mean_t.mean()),
            frechet_cov_term_db2=float(cov_t.mean()),
            comod_frobenius=float(corr_d.mean()),
            real_offdiag_corr=float(off_r.mean()),
            arm_offdiag_corr=float(off_a.mean()),
            sd_signed_diff_db=float(d_sd.mean()),
            sd_abs_diff_db=float(np.abs(d_sd).mean()),
        ),
    )


def battery(
    real: np.ndarray, arm: np.ndarray, labels: np.ndarray, *, sr: int = SR
) -> dict[str, dict[str, Any]]:
    """All six measures of one (real, arm) pair, keyed by function name."""
    return dict(
        line_floor_lsd=line_floor_lsd(real, arm, labels, sr=sr),
        order_level_error=order_level_error(real, arm, labels, sr=sr),
        level_distribution=level_distribution(real, arm, sr=sr),
        spatial=spatial(real, arm, sr=sr),
        itakura_saito=itakura_saito(real, arm, sr=sr),
        spectro_temporal=spectro_temporal(real, arm, sr=sr),
    )
