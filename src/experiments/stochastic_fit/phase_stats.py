"""Second-order statistics of the STFT along fitted lines: what the Whittle
fit cannot see.

The periodogram fit reads the *mean* power of every cell. Whether a line is a
coherent tone or narrowband noise, and whether its amplitude flickers with the
rotor speed or with its sub-bin position, lives in the complex STFT and in the
cell-to-cell structure. Two readouts:

``lag_coherence``
    the normalized cross-correlation ``|sum z_t z*_{t-L} e^{-i dphi}| /
    sqrt(sum |z_t|^2 sum |z_{t-L}|^2)`` at each line's bin, the expected phase
    advance of the reference carrier removed, for lags ``L`` in hops. White
    noise gives the analysis window's own overlap correlation (Hann,
    quarter-window hop: 0.659 / 0.167 / 0.008 / 0 at lags 1..4); a tone with
    Wiener phase (a Lorentzian line of half width ``gamma``) gives the
    Lorentzian autocorrelation smoothed by the window
    (:func:`lorentzian_lag_prediction`, which the window makes much slower
    than ``exp(-2 pi gamma tau)`` at short lags); a coherent tone plus noise
    sits on a plateau equal to the tone's share of the bin power. The null
    level (white noise through the same window, frames and demodulation) is
    returned beside the statistic because with few pairs it is not zero.

``centre_regressions``
    slope of ``log R`` at the line centre against the rotor's speed
    deviation (per cent) and against the centre's sub-bin offset, per order
    band — the two amplitude-flicker mechanisms that a fitted mean model
    could be missing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import HOP, N_FFT

HANN_OVERLAP = (1.0, 0.6592, 0.1667, 0.0075, 0.0)  # lags 0..4 at a quarter-window hop


def stft(audio: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """``(M, N, F)`` complex Hann STFT on the periodogram's frame grid."""
    audio = np.asarray(audio, dtype=np.float64)
    n = audio.shape[1]
    window = np.hanning(n_fft + 1)[:n_fft]
    starts = np.arange(1 + (n - n_fft) // hop) * hop
    frames = np.stack([audio[:, s : s + n_fft] for s in starts], axis=1) * window
    return np.fft.rfft(frames, axis=-1)


@dataclass
class LagCoherence:
    lags: np.ndarray
    coherence: np.ndarray  # (L,)
    null: np.ndarray  # (L,) white-noise surrogate through the same window/runs
    n_pairs: np.ndarray  # (L,)


def _runs(frames: np.ndarray) -> list[np.ndarray]:
    """Indices of maximal runs of consecutive frames."""
    if frames.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(frames) != 1)
    return np.split(np.arange(frames.size), breaks + 1)


def stft_at(
    audio: np.ndarray,
    frames: np.ndarray,
    centres_hz: np.ndarray,
    n_fft: int = N_FFT,
    hop: int = HOP,
    sr: int = 16000,
) -> np.ndarray:
    """``(M, T)`` Hann-windowed DTFT of each frame at its own centre frequency,
    frame-start phase reference — the STFT coefficient a line *would* have at
    exactly its centre. Reading the nearest FFT bin instead flips the phase
    reference by pi whenever the bin index changes between frames (the
    symmetric window's linear phase), which a moving line does constantly."""
    audio = np.asarray(audio, dtype=np.float64)
    window = np.hanning(n_fft + 1)[:n_fft]
    n = np.arange(n_fft)
    out = np.empty((audio.shape[0], frames.size), dtype=np.complex128)
    for i, (t, fc) in enumerate(zip(frames, centres_hz, strict=True)):
        seg = audio[:, t * hop : t * hop + n_fft] * window
        out[:, i] = seg @ np.exp(-2j * np.pi * fc * n / sr)
    return out


def lag_coherence(
    audio: np.ndarray,
    tracks: list[tuple[np.ndarray, np.ndarray]],
    lags: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16),
    *,
    hop: int = HOP,
    sr: int = 16000,
    min_run: int = 8,
    seed: int = 0,
) -> LagCoherence:
    """Pooled lag coherence over line tracks.

    ``audio`` is ``(M, T)``; each track is ``(frames, centres_hz)`` for one
    (rotor, order) — consecutive frames only are used (runs shorter than
    ``min_run`` are dropped). Coefficients are evaluated at the exact centre
    (:func:`stft_at`); the carrier's phase advance between consecutive
    frames is ``2 pi * mean(centre) * hop / sr``.
    """
    rng = np.random.default_rng(seed)
    lags_a = np.asarray(lags)
    # per run: |sum inc| / sqrt(sum|a|^2 sum|b|^2), then a pair-count-weighted
    # mean over runs. Runs cannot be pooled as one complex sum: every harmonic
    # has its own initial phase, so cross-run terms would cancel. The per-run
    # magnitude carries a positive finite-sample bias (~1/sqrt(n_pairs)); the
    # phase-scrambled null reproduces it with the same weights.
    w_sum = np.zeros(lags_a.size)
    coh_sum = np.zeros(lags_a.size)
    null_sum = np.zeros(lags_a.size)
    count = np.zeros(lags_a.size, dtype=int)
    for frames, centres in tracks:
        frames, centres = np.asarray(frames), np.asarray(centres, dtype=np.float64)
        for run in _runs(frames):
            if run.size < min_run:
                continue
            f_i, c_i = frames[run], centres[run]
            z = stft_at(audio, f_i, c_i, hop=hop, sr=sr)  # (M, T)
            advance = 2 * np.pi * 0.5 * (c_i[:-1] + c_i[1:]) * hop / sr
            phase = np.concatenate([[0.0], np.cumsum(advance)])
            zd = z * np.exp(-1j * phase)[None, :]
            # the null: white noise of the same length through the SAME window,
            # frames and demodulation. Successive lagged products of a Hann STFT
            # are correlated through the 75 % frame overlap, which inflates a
            # short run's |sum| beyond the iid finite-sample value; a per-frame
            # phase scramble destroys that correlation and under-reads the null
            # by ~30 % on 20-frame runs, a white-noise surrogate reproduces it.
            white = rng.standard_normal(audio.shape)
            zn = stft_at(white, f_i, c_i, hop=hop, sr=sr) * np.exp(-1j * phase)[None, :]
            for i, lag in enumerate(lags_a):
                if zd.shape[1] <= lag:
                    continue
                a, b = zd[:, lag:], zd[:, :-lag]
                den = float(np.sqrt(np.sum(np.abs(a) ** 2) * np.sum(np.abs(b) ** 2)))
                if den <= 0:
                    continue
                an, bn = zn[:, lag:], zn[:, :-lag]
                den_n = float(np.sqrt(np.sum(np.abs(an) ** 2) * np.sum(np.abs(bn) ** 2)))
                w = a.size
                coh_sum[i] += w * float(np.abs((a * np.conj(b)).sum())) / den
                null_sum[i] += w * float(np.abs((an * np.conj(bn)).sum())) / den_n
                w_sum[i] += w
                count[i] += w
    ok = w_sum > 0
    coh = np.where(ok, coh_sum / np.maximum(w_sum, 1e-30), np.nan)
    null = np.where(ok, null_sum / np.maximum(w_sum, 1e-30), np.nan)
    return LagCoherence(lags_a, coh, null, count)


def lorentzian_lag_prediction(
    gamma_hz: float, lags: np.ndarray, hop: int = HOP, sr: int = 16000, n_fft: int = N_FFT
) -> np.ndarray:
    """Lag coherence of a zero-mean Lorentzian line of half width ``gamma``
    seen through the Hann window at hop ``hop``.

    The STFT coefficient is the windowed sum of the process, so its lag-L
    covariance is the process autocorrelation ``exp(-2 pi gamma |t|)``
    convolved with the window's autocorrelation, evaluated at ``L hop``:
    ``sum_s R_w(s) exp(-2 pi gamma |L hop + s| / sr)``, normalized by the same
    sum at lag 0 (the diffusion also shrinks the coefficient's variance). For
    ``gamma -> inf`` this is the window overlap alone."""
    window = np.hanning(n_fft + 1)[:n_fft]
    r_w = np.correlate(window, window, mode="full")  # s = -(n_fft-1) .. n_fft-1
    s = np.arange(-(n_fft - 1), n_fft)
    var0 = float(np.sum(r_w * np.exp(-2 * np.pi * gamma_hz * np.abs(s) / sr)))
    out = []
    for lag in np.asarray(lags):
        cov = float(np.sum(r_w * np.exp(-2 * np.pi * gamma_hz * np.abs(lag * hop + s) / sr)))
        out.append(cov / var0)
    return np.asarray(out)


def centre_regressions(
    log_r: np.ndarray, speed_dev_pct: np.ndarray, sub_bin: np.ndarray
) -> dict[str, float]:
    """Slopes and correlations of ``log R`` at line centres against the two
    flicker covariates. ``sub_bin`` is the centre's distance to the nearest
    bin in bins (0 on a bin, 0.5 between)."""
    if log_r.size < 30:
        return dict(
            slope_speed=np.nan,
            slope_subbin=np.nan,
            corr_speed=np.nan,
            corr_subbin=np.nan,
            n=int(log_r.size),
        )
    a_s = float(np.polyfit(speed_dev_pct, log_r, 1)[0])
    a_b = float(np.polyfit(sub_bin, log_r, 1)[0])
    c_s = float(np.corrcoef(speed_dev_pct, log_r)[0, 1])
    c_b = float(np.corrcoef(sub_bin, log_r)[0, 1])
    return dict(
        slope_speed=a_s, slope_subbin=a_b, corr_speed=c_s, corr_subbin=c_b, n=int(log_r.size)
    )
