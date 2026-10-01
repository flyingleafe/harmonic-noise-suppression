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
from typing import Any

import numpy as np

__all__ = [
    "LineFit",
    "WhittleFit",
    "whittle_fit",
    "AmplitudeJitter",
    "amplitude_jitter",
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


# ─── Whittle fit: D, AM pedestal, amplitudes ──────────────────────────────────


@dataclass
class WhittleFit:
    """Profile-likelihood fit of the two-Lorentzian line model to one
    periodogram per mic: ``S_k(f) = a_k [L_{γ_k} + σ_m² L_{γ_k+γ_m}] ⊛ K + F_k``
    with ``γ_k = π k² D`` (shaft) and ``γ_m = λ_m / 2π`` (amplitude jitter)."""

    s: float
    D: float  #: rev²/s
    gamma_m: float  #: Hz, HWHM of the amplitude-jitter pedestal
    sigma_m2: float  #: pedestal power relative to the carrier
    orders: np.ndarray  #: (K,)
    amp2: np.ndarray  #: (C, K) carrier squared amplitude per mic (0 dB = 1)
    loglik: float  #: Whittle log-likelihood at the optimum (fit mics)
    grid: dict  #: profile log-likelihood slices through the optimum
    T: float
    mics: list[int]  #: mics the shape parameters were fitted on

    @property
    def gamma(self) -> np.ndarray:
        return math.pi * self.orders.astype(float) ** 2 * self.D


def _line_windows(f, P, s, K, T, half_hz=8.0, floor_hz=60.0):
    """Per order: the periodogram within ``±half_hz`` of ``k s`` (same width
    for all orders) ``(C, K, B)`` and the local floor level ``(C, K)`` (20th
    percentile within ±floor_hz, scaled to the mean of exponential noise)."""
    df = float(f[1])
    half = int(round(min(half_hz, s / 4) / df))
    B = 2 * half + 1
    C = P.shape[0]
    win = np.zeros((C, K, B))
    floor = np.zeros((C, K))
    hf = int(round(floor_hz / df))
    for j in range(K):
        c0 = int(round((j + 1) * s / df))
        win[:, j] = P[:, c0 - half : c0 + half + 1]
        floor[:, j] = np.percentile(P[:, max(0, c0 - hf) : c0 + hf], 20, axis=1) / 0.2231
    off = (np.arange(B) - half) * df
    return win, floor, off


def _shapes(off, gam_k, gamma_m, sigma_m2, kern, df, T):
    """Unit-carrier line shapes ``(K, B)``: ``L_{γ_k} + σ_m² L_{γ_k+γ_m}``
    through the window kernel; an unresolved carrier is the kernel itself."""
    from scipy.signal import fftconvolve

    g1 = gam_k[:, None]
    lor = (g1 / math.pi) / (off[None, :] ** 2 + g1**2) * df
    unres = gam_k < df
    if unres.any():
        delta = np.zeros_like(off)
        delta[off.size // 2] = 1.0
        lor[unres] = delta
    g2 = (gam_k + gamma_m)[:, None]
    ped = sigma_m2 * (g2 / math.pi) / (off[None, :] ** 2 + g2**2) * df
    sh = fftconvolve(lor + ped, kern[None, :], mode="same", axes=1)
    return np.maximum(sh, 1e-300)


def _profile_amplitudes(win, floor, shape, n_iter=15):
    """Whittle ML of ``a`` per (mic, order) for ``S = a·shape + F``: damped
    Newton in ``ln a`` from a peak-based start; returns ``(a (C,K), loglik)``."""
    g = shape[None]
    F = floor[:, :, None]
    B = win.shape[2]
    c = B // 2
    a = np.maximum(
        np.max(np.maximum(win[:, :, c - 2 : c + 3] - F, 0) / g[:, :, c - 2 : c + 3], axis=2), 1e-30
    )
    for _ in range(n_iter):
        S = a[:, :, None] * g + F
        r = win / S
        grad = np.sum(g / S * (r - 1), axis=2)  # dl/da
        hess = np.sum((g / S) ** 2 * (1 - 2 * r), axis=2)  # d2l/da2
        d1 = a * grad
        d2 = a * a * hess + d1
        step = np.where(d2 < 0, -d1 / np.where(d2 < 0, d2, -1.0), np.sign(d1) * 0.5)
        a = a * np.exp(np.clip(step, -1.0, 1.0))
    S = a[:, :, None] * g + F
    ll = -np.sum(np.log(S) + win / S)
    return a, float(ll)


def whittle_fit(
    x_ct: np.ndarray,
    fs: int,
    s: float,
    *,
    mics: list[int] | None = None,
    k_max: int = 150,
    f_max: float | None = None,
    D_grid=np.geomspace(1e-5, 1e-2, 10),
    gm_grid=np.geomspace(0.05, 3.0, 8),
    sm_grid=np.geomspace(0.003, 0.5, 6),
    refine: int = 2,
) -> WhittleFit:
    """Profile Whittle likelihood over ``(D, γ_m, σ_m²)`` with the per-line
    amplitudes solved in closed form at every point; the floor level of each
    line is a known constant (local 20th percentile), never a parameter.
    Shape parameters from ``mics`` (the non-windy ones), amplitudes for all."""
    sel = list(range(x_ct.shape[0])) if mics is None else list(mics)
    f, P, w = _periodogram(x_ct, fs, pad=1)
    df = float(f[1])
    T = x_ct.shape[1] / fs
    nfft = 2 * (f.size - 1)
    Wk = np.abs(np.fft.rfft(w, nfft)) ** 2 / w.sum() ** 2
    kh = 4
    kern = np.concatenate([Wk[kh:0:-1], Wk[: kh + 1]])
    f_top = 0.95 * fs / 2 if f_max is None else f_max
    K = min(k_max, int(f_top // s))
    orders = np.arange(1, K + 1)
    win, floor, off = _line_windows(f, P, s, K, T)
    kk = orders.astype(float)

    def ll_at(D, gm, sm):
        sh = _shapes(off, math.pi * kk**2 * D, gm, sm, kern, df, T)
        _, ll = _profile_amplitudes(win[sel], floor[sel], sh)
        return ll

    best = (-np.inf, None)
    grid = {
        "D": list(map(float, D_grid)),
        "gamma_m": list(map(float, gm_grid)),
        "sigma_m2": list(map(float, sm_grid)),
    }
    LL = np.full((len(D_grid), len(gm_grid), len(sm_grid)), -np.inf)
    for i, D in enumerate(D_grid):
        for j, gm in enumerate(gm_grid):
            for l_, sm in enumerate(sm_grid):
                LL[i, j, l_] = ll_at(D, gm, sm)
    i, j, l_ = np.unravel_index(int(np.argmax(LL)), LL.shape)
    D, gm, sm = float(D_grid[i]), float(gm_grid[j]), float(sm_grid[l_])
    best = (float(LL[i, j, l_]), (D, gm, sm))
    # local refinement: shrink the log-step around the optimum
    steps = [
        math.log(D_grid[1] / D_grid[0]),
        math.log(gm_grid[1] / gm_grid[0]),
        math.log(sm_grid[1] / sm_grid[0]),
    ]
    for _ in range(refine):
        steps = [st / 3 for st in steps]
        cand = [
            (D * math.exp(a * steps[0]), gm * math.exp(b * steps[1]), sm * math.exp(c * steps[2]))
            for a in (-1, 0, 1)
            for b in (-1, 0, 1)
            for c in (-1, 0, 1)
        ]
        vals = [ll_at(*cnd) for cnd in cand]
        m = int(np.argmax(vals))
        if vals[m] > best[0]:
            best = (float(vals[m]), cand[m])
            D, gm, sm = cand[m]
    grid["loglik_D"] = [float(v) for v in LL[:, j, l_]]
    grid["loglik_gamma_m"] = [float(v) for v in LL[i, :, l_]]
    grid["loglik_sigma_m2"] = [float(v) for v in LL[i, j, :]]
    sh = _shapes(off, math.pi * kk**2 * D, gm, sm, kern, df, T)
    amp2, _ = _profile_amplitudes(win, floor, sh)
    return WhittleFit(
        s=s,
        D=D,
        gamma_m=gm,
        sigma_m2=sm,
        orders=orders,
        amp2=amp2,
        loglik=best[0],
        grid=grid,
        T=T,
        mics=sel,
    )


# ─── amplitude jitter, measured directly per order ────────────────────────────


@dataclass
class AmplitudeJitter:
    """Per-order amplitude jitter from short-frame amplitude tracks."""

    orders: np.ndarray  #: (K,)
    sigma_m2: np.ndarray  #: (C, K) variance of ln A_k(t), noise-corrected (nan: line too weak)
    gamma_m: np.ndarray  #: (C, K) Hz, 1/(2π · integral correlation time of ln A_k)
    snr_db: np.ndarray  #: (C, K) mean line/noise of the frame amplitudes
    coherence: np.ndarray  #: (C,) median pairwise correlation of ln A_k(t) across usable orders
    frame_s: float


def amplitude_jitter(
    x_ct: np.ndarray,
    fs: int,
    s: float,
    *,
    k_max: int = 150,
    frame_s: float = 0.25,
    snr_min_db: float = 10.0,
) -> AmplitudeJitter:
    """Track every harmonic's amplitude with a ``frame_s`` Hann STFT (hop
    half a frame, peak of ±1 bin around ``k s`` so the shaft jitter stays
    inside the bin), then per order and mic: variance of ``ln A_k(t)`` minus
    the additive-noise part (``N/(2 A²)``, noise read midway between
    harmonics), the integral correlation time of ``ln A_k`` → ``γ_m``, and
    the median correlation of ``ln A_k`` across usable orders."""
    nb = int(frame_s * fs)
    hop = nb // 2
    n_fr = (x_ct.shape[1] - nb) // hop + 1
    w = np.hanning(nb)
    nfft = 1 << int(math.ceil(math.log2(2 * nb)))
    df = fs / nfft
    K = min(k_max, int(0.95 * fs / 2 // s))
    orders = np.arange(1, K + 1)
    C = x_ct.shape[0]
    A2 = np.zeros((C, K, n_fr))
    N2 = np.zeros((C, K, n_fr))
    idx = np.rint(orders * s / df).astype(int)
    idn = np.rint((orders + 0.5) * s / df).astype(int)
    h = max(1, int(round(0.5 * s / 4 / df)))  # ±1 native bin (4/frame_s Hz)
    for i in range(n_fr):
        X = np.fft.rfft(x_ct[:, i * hop : i * hop + nb] * w, nfft, axis=1)
        P = (np.abs(X) * (2.0 / w.sum())) ** 2
        for j in range(K):
            A2[:, j, i] = P[:, idx[j] - h : idx[j] + h + 1].max(axis=1)
            N2[:, j, i] = P[:, idn[j] - h : idn[j] + h + 1].max(axis=1)
    snr = 10 * np.log10(np.maximum(A2.mean(axis=2) / np.maximum(N2.mean(axis=2), 1e-300), 1e-3))
    lnA = 0.5 * np.log(np.maximum(A2, 1e-300))
    t = np.arange(n_fr) * hop / fs
    sig = np.full((C, K), np.nan)
    gam = np.full((C, K), np.nan)
    coh = np.full(C, np.nan)
    for c in range(C):
        tracks = []
        for j in range(K):
            if snr[c, j] < snr_min_db:
                continue
            y = lnA[c, j] - np.polyval(np.polyfit(t, lnA[c, j], 1), t)
            v = float(np.var(y)) - 0.5 * float(N2[c, j].mean() / A2[c, j].mean())
            sig[c, j] = max(v, 0.0)
            ac = np.correlate(y, y, "full")[y.size - 1 :] / (
                np.arange(y.size, 0, -1) * max(np.var(y), 1e-300)
            )
            pos = ac[: max(2, n_fr // 4)]
            first_neg = int(np.argmax(pos <= 0)) if (pos <= 0).any() else pos.size
            tau_c = float(np.sum(pos[:first_neg]) * hop / fs) - 0.5 * hop / fs
            gam[c, j] = 1.0 / (2 * math.pi * max(tau_c, hop / fs))
            tracks.append(y)
        if len(tracks) >= 3:
            R = np.corrcoef(np.stack(tracks))
            coh[c] = float(np.median(R[np.triu_indices(len(tracks), 1)]))
    return AmplitudeJitter(
        orders=orders, sigma_m2=sig, gamma_m=gam, snr_db=snr, coherence=coh, frame_s=frame_s
    )


# ─── synthesis ────────────────────────────────────────────────────────────────


def synthesise(
    fit: LineFit | WhittleFit,
    fs: int,
    n: int,
    seed: int = 0,
    am: tuple[Any, Any] | None = None,
    env_rate: int = 200,
) -> np.ndarray:
    """``(C, n)`` audio from the profile: one shaft (rate ``s``, phase
    diffusion ``D``) shared by all mics, every harmonic at its measured
    amplitude with a random phase; with ``am = (σ_m², γ_m)`` — scalars or
    per-order arrays — each order also carries a log-amplitude OU jitter
    ``exp(g_k(t))`` (variance ``σ_m²``, memory ``1/(2π γ_m)``), independent
    across orders and shared by the mics. Nothing else."""
    rng = np.random.default_rng(seed)
    C, K = fit.amp2.shape
    hop = fs // env_rate  # envelope sample spacing; blocks of one hop
    dt = hop / fs
    n_blk = -(-n // hop)
    n_pad = n_blk * hop
    theta = np.cumsum(fit.s / fs + math.sqrt(fit.D / fs) * rng.standard_normal(n_pad))
    frac = np.mod(theta, 1.0).reshape(n_blk, 1, hop)  # shaft angle in turns
    orders = np.asarray(fit.orders, dtype=np.float64)[None, :, None]
    phase = rng.uniform(0, 2 * np.pi, (C, K))
    cr = (np.sqrt(fit.amp2) * np.cos(phase)).astype(np.float32)
    ci = (np.sqrt(fit.amp2) * np.sin(phase)).astype(np.float32)
    env = None
    if am is not None:  # one log-amplitude OU envelope per order, shared by the mics
        sm = np.broadcast_to(np.asarray(am[0], float), (K,))
        gm = np.broadcast_to(np.asarray(am[1], float), (K,))
        rho = np.exp(-2 * np.pi * gm * dt)[:, None]
        sd = np.sqrt(sm)[:, None]
        noise = rng.standard_normal((K, n_blk + 1)) * sd * np.sqrt(1 - rho**2)
        noise[:, :1] = rng.standard_normal((K, 1)) * sd
        g = np.empty_like(noise)
        g[:, 0] = noise[:, 0]
        for i in range(1, n_blk + 1):
            g[:, i] = rho[:, 0] * g[:, i - 1] + noise[:, i]
        env = np.exp(g - 0.5 * sm[:, None]).astype(np.float32)  # (K, n_blk + 1)
    w = (np.arange(hop, dtype=np.float32) / hop)[None, None, :]
    out = np.empty((C, n_pad), dtype=np.float32)
    B = max(1, (1 << 22) // (K * hop))  # blocks per chunk (~16 MB per bank)
    for b0 in range(0, n_blk, B):
        b1 = min(n_blk, b0 + B)
        ang = (2 * np.pi * (orders * frac[b0:b1])).astype(np.float32)  # (b, K, hop)
        zr, zi = np.cos(ang), np.sin(ang)
        if env is not None:  # linear blend between the hop endpoints
            e = env.T[b0:b1, :, None] * (1 - w) + env.T[b0 + 1 : b1 + 1, :, None] * w
            zr *= e
            zi *= e
        zr = zr.transpose(1, 0, 2).reshape(K, -1)
        zi = zi.transpose(1, 0, 2).reshape(K, -1)
        out[:, b0 * hop : b1 * hop] = cr @ zr - ci @ zi  # Re(coef · e^{i2πkθ})
    return out[:, :n]


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
