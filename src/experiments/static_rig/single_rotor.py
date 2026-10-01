"""Single-rotor harmonic profile by a linear fit of Lorentzian lines.

What a stationary single-rotor recording looks like (DREGON singles,
``docs/experiments/static-rig-profiles.md``): the shaft speed is constant to
the record's resolution, with a *fast* frequency jitter (memory ≪ 0.1 s) that
makes the k-th harmonic a Lorentzian of half-width ``γ_k = π k² D``, ``D`` the
phase diffusion in rev²/s. Everything the profile needs is therefore in one
Hann periodogram per microphone: the k-th line reads ``a_k · (L_{γ_k} ⊛ K)``
with ``K`` the window's power kernel, so its squared amplitude ``a_k`` (0 dB
= unit amplitude, the ``plots.spectrum_viewer`` convention) is the measured
line height divided by the known shape — ``a_k`` itself below the resolution,
``peak · π γ_k T / 1.5`` once the line is resolved. No fitting.

Pipeline (the shaft rate is known to a few percent going in):
:func:`strict_span` (rotor strictly on) → :func:`core_speed` (s̄ from the unresolved cores of orders 2–7) →
:func:`phase_wander` (``D`` from the BPF phase) → :func:`line_powers` →
:func:`synthesise` (the check: same speed law, same lines).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "LineFit",
    "RotorModel",
    "core_speed",
    "line_powers",
    "phase_wander",
    "strict_span",
    "synthesise",
    "analyse",
]


# ─── span and speed ───────────────────────────────────────────────────────────


def _block_spectra(x_ct: np.ndarray, fs: int, block_s: float = 0.25, pad: int = 4):
    nb = int(block_s * fs)
    n_blk = x_ct.shape[1] // nb
    w = np.hanning(nb)
    nfft = 1 << int(math.ceil(math.log2(pad * nb)))
    P = np.zeros((n_blk, nfft // 2 + 1))
    for i in range(n_blk):
        X = np.fft.rfft(x_ct[:, i * nb : (i + 1) * nb] * w, nfft, axis=1)
        P[i] = (np.abs(X) ** 2).sum(axis=0)
    return P, fs / nfft, nb


def strict_span(x_ct: np.ndarray, fs: int, s0: float, tol: float = 0.01, trim_s: float = 1.0):
    """``(start, stop)`` samples of the longest run of 0.25 s blocks whose
    blade-pass line (3-block median) sits within ``tol`` of ``s0``, trimmed
    by ``trim_s``."""
    P, df, nb = _block_spectra(x_ct, fs)
    lo, hi = int(0.8 * 2 * s0 / df), int(1.2 * 2 * s0 / df)
    sb = (lo + np.argmax(P[:, lo:hi], axis=1)) * df / 2
    sb = np.concatenate([sb[:1], np.median(np.stack([sb[:-2], sb[1:-1], sb[2:]]), axis=0), sb[-1:]])
    ok = np.abs(sb - s0) < tol * s0
    best, run, start, bs = 0, 0, 0, 0
    for i, v in enumerate(ok):
        run = run + 1 if v else 0
        if v and run == 1:
            start = i
        if run > best:
            best, bs = run, start
    t = int(round(trim_s / 0.25))
    return (bs + t) * nb, (bs + best - t) * nb


def _periodogram(x_ct: np.ndarray, fs: int, pad: int = 2):
    """Amplitude-normalised Hann periodogram ``(freqs, P (C, F))``: a
    sinusoid of amplitude ``A`` on a bin reads ``A²``."""
    n = x_ct.shape[1]
    w = np.hanning(n)
    nfft = 1 << int(math.ceil(math.log2(pad * n)))
    X = np.fft.rfft(x_ct * w, nfft, axis=1)
    P = (np.abs(X) * (2.0 / w.sum())) ** 2
    return np.fft.rfftfreq(nfft, 1.0 / fs), P, w


def core_speed(
    x_ct: np.ndarray, fs: int, s0: float, orders: tuple[int, ...] = (2, 3, 4, 5, 6, 7)
) -> tuple[float, np.ndarray]:
    """Mean shaft rate from the (resolution-limited) core peaks of ``orders``:
    ``(median, per-order-per-mic estimates (len(orders), C))``."""
    f, P, _ = _periodogram(x_ct, fs, pad=4)
    df = f[1]
    est = np.full((len(orders), P.shape[0]), np.nan)
    for j, k in enumerate(orders):
        c0 = int(round(k * s0 / df))
        h = max(3, int(round(0.01 * k * s0 / df)))
        seg = P[:, c0 - h : c0 + h + 1]
        i = np.argmax(seg, axis=1)
        for c in range(P.shape[0]):
            if 0 < i[c] < seg.shape[1] - 1:
                y0, y1, y2 = np.log(seg[c, i[c] - 1 : i[c] + 2] + 1e-30)
                d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
                est[j, c] = (c0 - h + i[c] + d) * df / k
    return float(np.nanmedian(est)), est


def phase_wander(
    x_ct: np.ndarray,
    fs: int,
    s: float,
    k: int = 2,
    lags_s: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0),
    half_bw: float = 40.0,
    rate: int = 400,
) -> np.ndarray:
    """RMS drift of the shaft phase from ``s·t`` after each lag, in
    revolutions (median over mics): demodulate order ``k`` within
    ``±half_bw`` Hz, unwrap, detrend. ``r(1 s)²`` is the phase diffusion ``D``."""
    nb = fs // rate
    n_blk = x_ct.shape[1] // nb
    t = np.arange(n_blk * nb) / fs
    y = (x_ct[:, : n_blk * nb] * np.exp(-2j * np.pi * k * s * t)).reshape(x_ct.shape[0], n_blk, nb)
    y = y.mean(axis=2)
    L = int(round(0.75 / half_bw * rate)) | 1
    w = np.hanning(L + 2)[1:-1]
    w /= w.sum()
    env = np.stack([np.convolve(y[c], w, mode="same") for c in range(y.shape[0])])[:, L:-L]
    ph = np.unwrap(np.angle(env), axis=1)
    tt = np.arange(ph.shape[1]) / rate
    out = []
    for lag in lags_s:
        m = int(lag * rate)
        if m >= ph.shape[1] // 2:
            out.append(np.nan)
            continue
        r = []
        for p in ph:
            rev = (p - np.polyval(np.polyfit(tt, p, 1), tt)) / (2 * np.pi * k)
            d = rev[m:] - rev[:-m]
            r.append(math.sqrt(float(np.mean(d**2))))
        out.append(float(np.median(r)))
    return np.array(out)


# ─── line powers ──────────────────────────────────────────────────────────────


@dataclass
class LineFit:
    """Per-mic harmonic powers of one recording, read off the periodogram."""

    s: float  #: shaft rate (rev/s)
    D: float  #: phase diffusion (rev²/s); HWHM of order k is ``π k² D`` Hz
    orders: np.ndarray  #: (K,) orders 1..K
    amp2: np.ndarray  #: (C, K) squared amplitude of each harmonic (0 dB = 1)
    T: float  #: span (s)

    @property
    def gamma(self) -> np.ndarray:
        """(K,) Lorentzian half-width at half-maximum of every order (Hz)."""
        return math.pi * self.orders.astype(float) ** 2 * self.D


def line_powers(
    x_ct: np.ndarray, fs: int, s: float, D: float, *, k_max: int = 150, f_max: float | None = None
) -> LineFit:
    """Squared amplitude of harmonics 1..``k_max`` (below ``f_max``, default
    0.95 Nyquist) from one Hann
    periodogram per mic: the periodogram is averaged over ``±max(γ_k, 1/T)``
    around ``k s`` and divided by the same average of the known line shape —
    the Lorentzian of HWHM ``γ_k = π k² D`` through the window kernel — so the
    reading is the line's total power whether the line is resolved
    (``γ_k ≫ 1/T``, peak ∝ 1/γ_k) or not. Everything under the line is
    attributed to it: no floor."""
    f, P, w = _periodogram(x_ct, fs, pad=2)
    df = float(f[1])
    T = x_ct.shape[1] / fs
    C = P.shape[0]
    nfft = 2 * (f.size - 1)
    Wk = np.abs(np.fft.rfft(w, nfft)) ** 2 / w.sum() ** 2  # K(j): K(0) = 1
    kern_h = int(round(4.0 / T / df)) + 1
    kern = np.concatenate([Wk[kern_h:0:-1], Wk[: kern_h + 1]])
    f_top = 0.95 * fs / 2 if f_max is None else f_max
    K = min(k_max, int(f_top // s))
    orders = np.arange(1, K + 1)
    amp2 = np.zeros((C, K))
    for j, k in enumerate(orders):
        gamma = math.pi * k * k * D
        c0 = int(round(k * s / df))
        half = int(round(max(gamma, 1.0 / T) / df)) + 1
        half = min(half, int(round(s / 3 / df)))  # never reach the neighbours
        sup = half + kern_h + int(round(20 * gamma / df))
        ff = np.arange(-sup, sup + 1) * df
        if gamma < df:
            lor = np.zeros_like(ff)
            lor[sup] = 1.0
        else:
            lor = (gamma / math.pi) / (ff**2 + gamma**2) * df
        shape = np.convolve(lor, kern, mode="same")
        shape_mean = shape[sup - half : sup + half + 1].mean()
        amp2[:, j] = P[:, c0 - half : c0 + half + 1].mean(axis=1) / shape_mean
    return LineFit(s=s, D=D, orders=orders, amp2=amp2, T=T)


# ─── synthesis ────────────────────────────────────────────────────────────────


def synthesise(fit: LineFit, fs: int, n: int, seed: int = 0) -> np.ndarray:
    """``(C, n)`` audio from the profile: one shaft (rate ``s``, phase
    diffusion ``D``) shared by all mics, every harmonic at its measured
    amplitude with a random phase. Nothing else."""
    rng = np.random.default_rng(seed)
    C = fit.amp2.shape[0]
    dtheta = fit.s / fs + math.sqrt(fit.D / fs) * rng.standard_normal(n)
    theta = np.cumsum(dtheta)
    out = np.zeros((C, n))
    for j, k in enumerate(fit.orders):
        base = 2 * np.pi * k * theta
        for c in range(C):
            out[c] += math.sqrt(fit.amp2[c, j]) * np.cos(base + rng.uniform(0, 2 * np.pi))
    return out


@dataclass
class RotorModel:
    """One recording: its strict span and the harmonic profile read with the
    KNOWN shaft rate and phase diffusion."""

    span: tuple[int, int]  #: samples of the strict rotor-on span in the input
    fit: LineFit


def analyse(
    x_ct: np.ndarray, fs: int, s: float, D: float, mics: list[int] | None = None, **fit_kw
) -> RotorModel:
    """Strict span (blade-pass line of ``mics`` within 1 % of ``s``) → every
    harmonic's power per mic, with the measured ``s`` (rev/s) and ``D``
    (rev²/s) taken as given — nothing is re-estimated."""
    sel = list(range(x_ct.shape[0])) if mics is None else list(mics)
    a, b = strict_span(x_ct[sel], fs, s)
    x = np.asarray(x_ct[:, a:b], dtype=np.float64)
    return RotorModel(span=(a, b), fit=line_powers(x, fs, s, D, **fit_kw))
