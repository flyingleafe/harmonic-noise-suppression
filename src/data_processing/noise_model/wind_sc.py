"""Single-channel wind noise from a wind-speed profile — a numpy port of the
core of the SC-Wind-Noise-Generator (Mirabilii, Lodermeyer, Czwielong,
Becker, Habets, "Simulating wind noise with airflow speed-dependent
characteristics", IWAENC 2022; https://github.com/audiolabs/SC-Wind-Noise-Generator,
MIT). Kept: the Weibull wind-speed profile with smoothed fluctuations, the
long-term gain regression (dB vs m/s), the GARCH short-term variance (the
gusty turbulent-pressure envelope) and the order-5 LPC whose line-spectral
frequencies are regressed on the wind speed (2–18 m/s), applied in 2048-
sample overlap-add frames at 48 kHz. Dropped: playback, saving, plotting and
the ``spectrum`` dependency (``lsf2poly`` is inlined). The generator is fed
an explicit ``Generator`` instead of the global numpy seed, and the output is
NOT peak-normalised: callers scale it (:func:`wind_noise` returns unit-RMS).
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.signal import lfilter, resample_poly
from scipy.signal import resample as _resample

FS_NATIVE = 48000
_LT_REGRESSION = np.array([8.00071114414022, -220.332082908370])  # dB vs m/s
_GP_ALPHA = np.array(
    [
        -2.73244444508231e-05,
        0.00141129711949206,
        -0.0274652794467908,
        0.257613241095714,
        -0.139824587447063,
    ]
)
_GP_BETA = np.array(
    [-9.75160902595897e-05, 0.00464300106846736, -0.0871968755558256, 0.651013973757802]
)
_GP_OMEGA = np.array([9.69585296574741e-05, -0.00231853830578967, 0.0124681159197788])
# LSF-vs-speed regression: column n = n-th line spectral frequency, polynomial in speed
_LSF_REGRESSION = np.array(
    [
        [
            -2.63412497797108e-06,
            5.93162248595821e-05,
            0.000215613938043173,
            -0.000149723789407121,
            -0.000213703084399375,
        ],
        [
            9.50240139044154e-05,
            -0.00271741166649528,
            -0.0103783584000284,
            0.00483963669507075,
            0.00931864887930701,
        ],
        [
            -0.000699199223507821,
            0.0428714179385289,
            0.177250839818556,
            -0.0329542145779793,
            -0.129910107562929,
        ],
        [
            0.0106849674771013,
            -0.234688122194936,
            -1.21337646113093,
            -0.168053225019258,
            0.568371362156217,
        ],
        [
            -0.000966851130291645,
            0.541693139684727,
            3.24796925730457,
            2.54984352038733,
            1.86097523205089,
        ],
    ]
)
SPEED_MIN, SPEED_MAX = 2.0, 18.0


def _rs(x: np.ndarray, n: int) -> np.ndarray:
    return np.asarray(_resample(x, n), dtype=np.float64)


def lsf2poly(lsf: np.ndarray) -> np.ndarray:
    """LPC polynomial ``a`` (leading 1) from line spectral frequencies (rad),
    the textbook construction (the ``spectrum`` package's ``lsf2poly``)."""
    w = np.sort(np.asarray(lsf, dtype=np.float64))
    z = np.exp(1j * w)
    r_q = np.concatenate([z[0::2], np.conj(z[0::2])])
    r_p = np.concatenate([z[1::2], np.conj(z[1::2])])
    q, p = np.poly(r_q), np.poly(r_p)
    if w.size % 2:
        p1, q1 = np.convolve(p, [1.0, 0.0, -1.0]), q
    else:
        p1, q1 = np.convolve(p, [1.0, -1.0]), np.convolve(q, [1.0, 1.0])
    a = 0.5 * (p1 + q1)
    return np.real(a[:-1])


_LPC_STEP = 0.02  # m/s: the speed grid the LPC polynomials are cached on
_LPC_CACHE: dict[int, np.ndarray] = {}


def _lpc_for_speed(speed: float) -> np.ndarray:
    key = int(round(float(speed) / _LPC_STEP))
    a = _LPC_CACHE.get(key)
    if a is None:
        sp = key * _LPC_STEP
        a = lsf2poly(np.array([np.polyval(_LSF_REGRESSION[:, i], sp) for i in range(5)]))
        _LPC_CACHE[key] = a
    return a


def wind_speed_profile(rng: np.random.Generator, n: int, fs: int, gustiness: int = 3) -> np.ndarray:
    """Wind speed (m/s) per sample: ``gustiness`` Weibull(2) draws (scale 2)
    resampled over the clip, plus 100 ms-smoothed Gaussian fluctuations."""
    pts = 2.0 * rng.weibull(2.0, max(1, int(gustiness)))
    prof = _rs(pts, n) if pts.size > 1 else np.full(n, float(pts[0]))
    return prof + _smooth(rng, n, fs)


def _smooth(rng: np.random.Generator, n: int, fs: int) -> np.ndarray:
    """SC's speed fluctuations: Gaussian noise (sd 10) through a causal
    100 ms Hann FIR of unit sum (``lfilter(h, [1], x)``, by FFT convolution)."""
    m = int(fs * 0.1)
    h = np.hanning(m)
    x = 10.0 * rng.standard_normal(n)
    # hann(u) = 0.5 (1 - cos(2 pi u / (m - 1))): a boxcar and a modulated boxcar, both
    # running sums (the FIR is causal: y[t] = sum_u h[u] x[t - u])
    w = 2.0 * np.pi / (m - 1)
    box = np.concatenate([[0.0], np.cumsum(x)])
    y = 0.5 * (box[1 : n + 1] - box[np.maximum(np.arange(n) + 1 - m, 0)])
    t = np.arange(n)
    rot = np.exp(1j * w * t)
    box_c = np.concatenate([[0.0], np.cumsum(x * np.conj(rot))])
    y -= 0.5 * np.real(rot * (box_c[1 : n + 1] - box_c[np.maximum(t + 1 - m, 0)]))
    return y / h.sum()


def wind_noise(
    rng: np.random.Generator,
    fs: int,
    duration_s: float,
    *,
    speed_profile: np.ndarray | None = None,
    gustiness: int = 3,
    short_term_var: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """``(x, speed)``: unit-RMS wind noise of ``duration_s`` seconds at ``fs``
    and the wind-speed profile (m/s) it was generated from (at ``fs``)."""
    n48 = int(round(duration_s * FS_NATIVE))
    if speed_profile is None:
        prof = wind_speed_profile(rng, n48, FS_NATIVE, gustiness)
    else:
        prof = _rs(np.asarray(speed_profile, dtype=np.float64), n48) + _smooth(rng, n48, FS_NATIVE)
    exc = _excitation(rng, prof, short_term_var)
    x = _lpc_filter(exc, prof)
    if fs != FS_NATIVE:
        g = np.gcd(int(fs), FS_NATIVE)
        x = np.asarray(resample_poly(x, int(fs) // g, FS_NATIVE // g), dtype=np.float64)
        prof = _rs(prof, x.size)
    x = x[: int(round(duration_s * fs))]
    prof = prof[: x.size]
    return x / max(float(np.sqrt(np.mean(x**2))), 1e-12), prof


def _excitation(rng: np.random.Generator, prof: np.ndarray, short_term_var: bool) -> np.ndarray:
    """White noise under SC's envelope: the long-term gain of the speed per
    sample, times the GARCH short-term standard deviation per 64-sample hop,
    overlap-added in 128-sample Hann frames (the frame sum is the Hann
    convolution of the per-hop gains, one FFT convolution)."""
    win, hop = 128, 64
    n = prof.size
    wgn = rng.standard_normal(n)
    lt = np.sqrt(10.0 ** (np.polyval(_LT_REGRESSION, prof) / 10.0))
    pr = np.concatenate([2.0 * np.ones(win), prof, 2.0 * np.ones(win)])
    n_fr = (pr.size - win) // hop + 1
    cs = np.concatenate([[0.0], np.cumsum(pr)])
    starts = np.arange(n_fr) * hop
    sp = np.clip((cs[starts + win] - cs[starts]) / win, SPEED_MIN, SPEED_MAX)
    alpha, beta, omega = (np.polyval(c, sp) for c in (_GP_ALPHA, _GP_BETA, _GP_OMEGA))
    beta = np.where(alpha + beta > 1.0, 0.0, beta)
    st = _garch(alpha, beta, omega, rng.standard_normal(n_fr))
    st /= max(float(np.abs(st).max()), 1e-12)
    gain = np.sqrt(np.abs(st)) if short_term_var else np.ones(n_fr)
    gain[-1] = 0.0  # SC's loop stops one frame short
    # sum_i gain_i hann(t - i hop): with hop = win / 2 the sample t = hop j + u lies in
    # frame j (offset u) and frame j - 1 (offset u + hop)
    hann = np.hanning(win)
    env = (
        gain[:, None] * hann[None, :hop]
        + np.concatenate([[0.0], gain[:-1]])[:, None] * hann[None, hop:]
    )
    return (wgn * lt) * env.ravel()[win : win + n]


@njit(cache=True)
def _garch(alpha: np.ndarray, beta: np.ndarray, omega: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """SC's GARCH(1, 1) short-term variance, one step per hop (``st[-1]`` and
    ``cv[-1]`` read the last entry at ``i = 0``, as SC's numpy loop does)."""
    n = noise.size
    st = np.zeros(n)
    cv = np.zeros(n)
    for i in range(n):
        prev = n - 1 if i == 0 else i - 1
        cv[i] = omega[i] + alpha[i] * st[prev] ** 2 + beta[i] * cv[prev]
        st[i] = np.sqrt(abs(cv[i])) * noise[i]
    return st


def _lpc_filter(exc: np.ndarray, prof: np.ndarray) -> np.ndarray:
    win, hop = 2048, 1024
    hann = np.hanning(win)
    pr = np.concatenate([2.0 * np.ones(win), prof, 2.0 * np.ones(win)])
    ex = np.concatenate([np.zeros(win), exc, np.zeros(win)])
    out = np.zeros(ex.size)
    n_fr = (ex.size - win) // hop + 1
    for i in range(n_fr):
        idx = slice(i * hop, i * hop + win)
        sp = float(np.clip(pr[idx].mean(), SPEED_MIN, SPEED_MAX))
        out[idx] += lfilter([1.0], _lpc_for_speed(sp), ex[idx] * hann)
    return out[win:-win]
