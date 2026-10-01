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


def _lpc_for_speed(speed: float) -> np.ndarray:
    lsf = np.array([np.polyval(_LSF_REGRESSION[:, i], speed) for i in range(5)])
    return lsf2poly(lsf)


def wind_speed_profile(rng: np.random.Generator, n: int, fs: int, gustiness: int = 3) -> np.ndarray:
    """Wind speed (m/s) per sample: ``gustiness`` Weibull(2) draws (scale 2)
    resampled over the clip, plus 100 ms-smoothed Gaussian fluctuations."""
    pts = 2.0 * rng.weibull(2.0, max(1, int(gustiness)))
    prof = _rs(pts, n) if pts.size > 1 else np.full(n, float(pts[0]))
    fl = 10.0 * rng.standard_normal(n)
    h = np.hanning(int(fs * 0.1))
    h /= h.sum()
    return prof + lfilter(h, [1.0], fl)


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
        prof = _rs(np.asarray(speed_profile, dtype=np.float64), n48)
        h = np.hanning(int(FS_NATIVE * 0.1))
        h /= h.sum()
        prof = prof + lfilter(h, [1.0], 10.0 * rng.standard_normal(n48))
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
    win, hop = 128, 64
    n = prof.size
    wgn = np.concatenate([np.zeros(win), rng.standard_normal(n), np.zeros(win)])
    lt = np.sqrt(10.0 ** (np.polyval(_LT_REGRESSION, prof) / 10.0))
    lt = np.concatenate([np.zeros(win), lt, np.zeros(win)])
    hann = np.hanning(win)
    pr = np.concatenate([2.0 * np.ones(win), prof, 2.0 * np.ones(win)])
    n_fr = (pr.size - win) // hop + 1
    # GARCH short-term variance per 64-sample hop
    st = np.zeros(n_fr)
    cv = np.zeros(n_fr)
    for i in range(n_fr):
        sp = float(np.clip(pr[i * hop : i * hop + win].mean(), SPEED_MIN, SPEED_MAX))
        alpha, beta, omega = (np.polyval(c, sp) for c in (_GP_ALPHA, _GP_BETA, _GP_OMEGA))
        if alpha + beta > 1:
            beta = 0.0
        cv[i] = omega + alpha * st[i - 1] ** 2 + beta * cv[i - 1]
        st[i] = np.sqrt(abs(cv[i])) * rng.standard_normal()
    st /= max(float(np.abs(st).max()), 1e-12)
    cond = np.abs(st)
    exc = np.zeros(wgn.size)
    for i in range(n_fr - 1):
        idx = slice(i * hop, i * hop + win)
        g = lt[idx] * (np.sqrt(cond[i]) if short_term_var else 1.0)
        exc[idx] += g * wgn[idx] * hann
    return exc[win:-win]


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
