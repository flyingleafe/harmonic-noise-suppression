"""Step A0 of noise model v4: the phase-pedestal width ``W`` per order on the
DREGON single-rotor bench (``docs/explainers/noise-model-v4.qmd`` § Procedure A).

The instrument is the per-order phase coherence of
``scripts/static_rig_scratch/one_rotor_coherence.py``
(``docs/explainers/order-decoherence.qmd``), changed where its short lags
failed:

1. **Carrier-tracked demodulation.** The shaft angle ``theta`` is estimated
   from the cores of the low orders (:data:`SHAFT_ORDERS`, all calm
   microphones, three disjoint order groups) on a constant carrier. Every
   order ``k`` is then demodulated from ``x e^{-i k theta}``: the shaft is
   removed BEFORE the band filter, so the shaft smear of the high orders
   (wider than the band above k ~ 60 on this bench) is not cut by it. Orders
   inside the shaft set use the two groups they are not in.
2. **The band filter is modelled, not simulated.** One FFT per order, the band
   bins weighted by the zero-phase Butterworth response of
   :func:`utils.demod.demodulate` (``1 / (1 + (f/B)^12)``), inverse-transformed
   straight onto the 400.9 Hz phase grid. The complex autocorrelation of the
   baseband is then
   ``C(tau) = P (kappa * R)(tau) e^{-k^2 B(tau)/2} + N0 kappa(tau) + nu(tau)``:
   ``kappa`` the inverse transform of the window's power response, ``R`` the
   line's lag law (spec eq-lag without the shaft factor), ``N0`` the white
   floor, ``nu`` the pedestal tails of orders ``k +- 1`` inside the band
   through the same window, ``B(tau)`` the increment variance of the shaft
   estimate's own error (three-cornered hat of the group estimates).
3. **The floor is read between the lines.** ``N0`` comes from the median of a
   Welch spectrum at the half-integer orders (±10 % of the spacing), not from
   the flank of the line's own band. The flank of a band ±0.4 f0 wide sits
   inside the pedestal, so the old instrument counted the pedestal as noise
   (6-12 dB of in-band SNR at k = 18-26 on Motor1_70); a pedestal is more
   correlated at short lags than in-band white noise, so subtracting a
   white-noise floor of that level drives the net variance NEGATIVE at lags
   inside the window's correlation time and turns the pedestal's own
   decorrelation into a step between ~20 and 50 ms. The gap level itself
   holds the tails of the two adjacent pedestals: :data:`FLOOR_PASSES` passes
   fit every order, remove the fitted tails from the gaps and put them in the
   neighbouring bands as ``nu``.

The lag law ``2 s2 (1 - e^{-2 pi g tau})`` of the phase OU is fitted per order
through that forward model, jointly over the calm microphones (``s2, g``
shared, the line power of each microphone fixed by ``C(0)`` minus the rest),
to the magnitude of the normalised autocorrelation with block-jackknife
errors. No free offset is needed: the noise and the window enter through
``N0 kappa + nu``. The log-amplitude OU (``s2A, gA``) is fitted separately to
the log-amplitude increment variance at the resolved lags, with one constant
per microphone (the additive-noise part), and enters the phase fit as a known
factor (capped at the spec's amplitude cap; the prior median when no
microphone passes the gate).

**Reachable lag.** Two samples of the baseband closer than the window's
correlation time share their noise and their pedestal. The band cannot exceed
half the harmonic spacing without admitting the neighbouring order, so the
time resolution is at best ``1/(2B) = 1/f0``, one revolution: the first zero
of ``kappa``. The Butterworth window rings (``kappa/kappa(0)`` = -0.17 near
1.4 revolutions, +0.08 near 2.7), so a lag is *resolved* only when
``|kappa(tau')| / kappa(0) <= KAPPA_TOL`` for every ``tau' >= tau``
(:func:`reachable_lag`): about 3 revolutions at ``B = f0/2`` (33-61 ms over
the bench speeds; 3.7 revolutions at the old ±0.4 f0). A lag-domain floor
subtraction is biased anywhere below that. The forward model uses every lag
on the phase grid (the window is in the model); a second fit on the resolved
lags only (``W_long``) shows how much of ``W`` rests on the shorter ones.
``W`` of a pedestal whose rate exceeds ~2B is a lower bound (flag
``g_unresolved``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import fft as sfft
from scipy import optimize, signal

from experiments.noise_model.v4_priors import (
    AMP_GROUPS,
    amp_cap,
    amp_log_median,
    gamma_amp_log_median,
    pedestal_width,
)

__all__ = [
    "LAGS_MS",
    "RECORDINGS",
    "Bench",
    "Window",
    "analyse_recording",
    "findings_markdown",
    "load_bench",
    "reachable_lag",
    "reduce_rows",
]

DATASET = "DREGON-frames"
RPS_KEY = "motors_command"
SINGLES_PATH = Path("results/static_rig/single_rotor/dregon_singles.json")
AMP_JITTER_PATH = Path("results/static_rig/single_rotor/amplitude_jitter.json")
RECORDINGS = tuple(f"motor_Motor{m}_{t}" for m in (1, 2, 3, 4) for t in (50, 60, 70, 80, 90))
BLADES = 2

#: Lags of spec step A0 (ms).
LAGS_MS = (2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0)
#: 44100 / 110 = 400.9 Hz phase grid: one sample = 2.49 ms, the shortest lag.
DECIM = 110
#: One-sided band as a fraction of the harmonic spacing: half the spacing is
#: the widest band that does not admit the neighbouring order (the window's
#: power response at the neighbour is -72 dB).
BAND_FRAC = 0.5
F_MAX_HZ = 7200.0
#: Shaft angle from the cores of these orders (the old instrument's set: an
#: order's own phase wander enters the shaft estimate divided by k, so orders
#: 7-10 lower the error B(tau) that leaks into order k as k^2 B; measured on
#: Motor1_70, B(2 s) = 3.7e-3 rad^2 against 5.9e-3 with orders 1-6).
SHAFT_ORDERS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
#: Band (Hz) of the shaft estimate. The OU shaft phase has no measurable
#: content above it (integrated OU with lambda <= 30 1/s: < 3e-6 rad^2 above
#: 15 Hz); a narrower band admits less noise into the estimate.
SHAFT_BAND_HZ = 15.0
GATE_DB = 10.0  # peak / median of the in-band baseband spectrum
#: A microphone enters the fits of an order when its in-band line power is
#: at least the in-band floor (then a 1 dB floor error moves P by <= 26 %).
SNR_MIN_DB = 0.0
EDGE_S = 0.5
N_BLOCKS = 6
KAPPA_TOL = 0.05
GAP_FRAC = 0.1  # floor read within ±GAP_FRAC f0 of each half-integer order
WELCH_SEG = 1 << 15
SLIP_SIGMA = 6.0
#: Model-error allowance added in quadrature to the jackknife error of the
#: normalised autocorrelation.
RHO_SIGMA_FLOOR = 0.02
#: Fit flag ``detected``: the pedestal variance reaches 2 % of the line power.
DETECT_S2 = 0.02
#: Pooled-W rules (:func:`reduce_rows`): an order enters when its pedestal
#: variance reaches USE_S2_MIN, its rate is resolved and not at a bound, at
#: least USE_MIN_MICS microphones carry it, and a ±1 dB floor error keeps W
#: within a factor STABLE_FACTOR (max/min of the three W under STABLE_FACTOR^2).
#: On a planted rotor (s2 = 0.01 on k < 8) the 0.02 detection flag fired on
#: 4 of 7 orders with W from 0.3 to 105 Hz; these rules keep them out.
USE_S2_MIN = 0.05
USE_MIN_MICS = 2
STABLE_FACTOR = 2.0
S2_BOUNDS = (1e-3, 30.0)
G_LO = 0.02  # Hz
G_HI_BANDS = 3.0  # upper bound on g in units of B
G_UNRESOLVED_BANDS = 2.0
N0_SENS_DB = 1.0
#: Floor passes: each removes from the gap levels the pedestal tails of the
#: two adjacent lines as fitted in the previous pass; a gap keeps at least
#: FLOOR_MIN_FRAC of its level.
FLOOR_PASSES = 2
FLOOR_MIN_FRAC = 0.1
#: Motor orders k = 3 p n (DREGON: p = 7 pole pairs): no pedestal in the
#: model (spec § The motor line); flagged and left out of the pooled W.
MOTOR_STEP = 21
#: Half-normal 97.5th percentile in units of its scale (spec: 2.24 c_W).
HALF_NORMAL_Q975 = 2.24


# ─── data ─────────────────────────────────────────────────────────────────────


@dataclass
class Bench:
    """One recording's strict rotor-on span at the native rate."""

    key: str
    x: np.ndarray  # (M, T) float64, mean removed
    fs: float
    rate: float  # rev/s, core peaks of orders 2-7 (dregon_singles.json)
    calm: tuple[int, ...]
    span_s: tuple[float, float]


def load_bench(key: str) -> Bench:
    """Native-rate audio of ``key``, cut to the strict span of
    ``dregon_singles.json``, with its carrier and calm microphones."""
    from experiments.stochastic_fit import clips as C

    singles = json.loads(SINGLES_PATH.read_text())["recordings"][key]
    rec = C.load_recording(DATASET, key, None, RPS_KEY)
    fs = float(rec.sr)
    a, b = (float(v) for v in singles["span_s"])
    x = np.asarray(rec.audio[:, int(round(a * fs)) : int(round(b * fs))], dtype=np.float64)
    wind = {int(m) for m in singles["wind_mics"]}
    return Bench(
        key=key,
        x=x - x.mean(axis=1, keepdims=True),
        fs=fs,
        rate=float(singles["s_bar_rev_s"]),
        calm=tuple(m for m in range(x.shape[0]) if m not in wind),
        span_s=(a, b),
    )


# ─── the window ───────────────────────────────────────────────────────────────


def band_amplitude(f: np.ndarray, band: float) -> np.ndarray:
    """Amplitude response of :func:`utils.demod.demodulate` (zero-phase
    Butterworth of order 6 run forward and back)."""
    return 1.0 / (1.0 + (np.asarray(f, dtype=np.float64) / band) ** 12)


def window_acf(tau: np.ndarray | float, band: float) -> np.ndarray:
    """``kappa(tau) = int |H(f)|^2 e^{i 2 pi f tau} df`` (real, even; Hz)."""
    f = np.linspace(0.0, 4.0 * band, 4001)
    w = band_amplitude(f, band) ** 2 * (f[1] - f[0])
    w[[0, -1]] *= 0.5
    tau = np.atleast_1d(np.asarray(tau, dtype=np.float64))
    return 2.0 * np.cos(2.0 * np.pi * tau[:, None] * f[None, :]) @ w


def reachable_lag(band: float, tol: float = KAPPA_TOL) -> float:
    """Shortest lag (s) beyond which the window's noise correlation
    ``|kappa| / kappa(0)`` stays under ``tol``."""
    u = np.arange(0.0, 0.4, 2.5e-4)
    k = np.abs(window_acf(u, band))
    env = np.maximum.accumulate((k / k[0])[::-1])[::-1]
    return float(u[int(np.argmax(env <= tol))])


@dataclass
class Window:
    """The band window on a quadrature grid for ``(kappa * R)(tau)``."""

    band: float
    u: np.ndarray  # (U,) s
    weight: np.ndarray  # (U,) kappa(u) du, sums to 1
    kappa0: float  # kappa(0), Hz: the noise bandwidth

    @classmethod
    def make(cls, band: float, du: float = 5e-4) -> Window:
        u = np.arange(-0.4, 0.4 + du / 2, du)
        k = window_acf(u, band)
        keep = np.abs(k) > 1e-4 * k.max()
        lo, hi = int(np.argmax(keep)), int(keep.size - np.argmax(keep[::-1]))
        u, k = u[lo:hi], k[lo:hi]
        return cls(band=band, u=u, weight=k / k.sum(), kappa0=float(window_acf(0.0, band)[0]))

    def kappa(self, tau: np.ndarray) -> np.ndarray:
        return window_acf(tau, self.band)


def line_lag(tau: np.ndarray, s2a: float, ga: float, s2p: np.ndarray, gp: np.ndarray) -> np.ndarray:
    """``R(tau)`` of spec eq-lag without the shaft factor; ``s2p, gp`` broadcast
    against ``tau`` from the left (``(G, 1, ...)`` for a parameter grid)."""
    a = np.abs(tau)
    return np.exp(s2a * np.expm1(-2.0 * np.pi * ga * a) + s2p * np.expm1(-2.0 * np.pi * gp * a))


def smoothed_lag(
    win: Window, taus: np.ndarray, s2a: float, ga: float, s2p: np.ndarray, gp: np.ndarray
) -> np.ndarray:
    """``(kappa * R)(tau)`` (``kappa`` normalised to unit integral) at ``taus``
    for parameter arrays ``s2p, gp`` of shape ``(G,)``: ``(G, len(taus))``.
    Lags past twice the kernel support read ``R`` itself: the cusp of ``R``
    at 0 is outside the kernel there, and the Butterworth power response is
    maximally flat, so the kernel's moments up to order 11 vanish."""
    s2p = np.asarray(s2p, dtype=np.float64)[:, None]
    gp = np.asarray(gp, dtype=np.float64)[:, None]
    near = taus < 2.0 * win.u[-1]
    out = np.empty((s2p.shape[0], taus.size))
    out[:, ~near] = line_lag(taus[None, ~near], s2a, ga, s2p, gp)
    r_inf = np.exp(-s2a - s2p)  # (G, 1)
    arg = taus[near][:, None] - win.u[None, :]  # (Ln, U)
    rc = line_lag(arg[None], s2a, ga, s2p[:, :, None], gp[:, :, None]) - r_inf[:, :, None]
    out[:, near] = r_inf + rc @ win.weight
    return out


# ─── demodulation ─────────────────────────────────────────────────────────────


def _fast_len_below(n: int) -> int:
    m = n
    while sfft.next_fast_len(m) != m:
        m -= 1
    return m


def baseband(
    spec: np.ndarray, n: int, fs: float, fc: float, band: float, onesided: bool
) -> np.ndarray:
    """``(M, n / DECIM)`` baseband of order frequency ``fc`` from the FFT
    ``spec`` (``(M, n//2+1)`` rfft or ``(M, n)`` fft) of an ``n``-sample
    signal: band bins weighted by :func:`band_amplitude`, inverse-transformed
    onto the decimated grid. Equal to ``demodulate(...)[..., ::DECIM]`` up to
    the circular edges."""
    n_p = n // DECIM
    q = np.rint(np.fft.fftfreq(n_p, 1.0 / n_p)).astype(np.int64)
    b_c = int(round(fc * n / fs))
    b = b_c + q
    if onesided:
        ok = (b >= 0) & (b <= n // 2)
        y = np.where(ok, spec[:, np.clip(b, 0, n // 2)], 0.0)
    else:
        y = spec[:, b % n]
    y = y * band_amplitude(b * fs / n - fc, band)
    z = np.fft.ifft(y, axis=-1) * (n_p / n)
    t = np.arange(n_p) * DECIM / fs
    return z * np.exp(-2j * np.pi * (fc - b_c * fs / n) * t)


def unwrap_censored(z: np.ndarray) -> np.ndarray:
    """Unwrapped phase from the wrapped one-sample increments, slips beyond
    :data:`SLIP_SIGMA` robust sigmas replaced by the median increment
    (``noise_v2_order_decoherence.unwrap_censored``)."""
    d = (np.diff(np.angle(z), axis=-1) + np.pi) % (2.0 * np.pi) - np.pi
    med = np.median(d, axis=-1, keepdims=True)
    scale = 1.4826 * np.median(np.abs(d - med), axis=-1, keepdims=True)
    bad = np.abs(d - med) > SLIP_SIGMA * np.maximum(scale, 1e-12)
    d = np.where(bad, np.broadcast_to(med, d.shape), d)
    head = np.zeros(d.shape[:-1] + (1,))
    return np.concatenate([head, np.cumsum(d, axis=-1)], axis=-1)


def gate_db(z: np.ndarray, fs_p: float, band: float) -> np.ndarray:
    """Peak / median (dB) of the two-sided baseband Welch spectrum in band."""
    nper = min(1024, z.shape[-1])
    f, p = signal.welch(
        z,
        fs=fs_p,
        nperseg=nper,
        return_onesided=False,
        detrend=False,  # pyright: ignore[reportArgumentType]
        axis=-1,
    )
    sb = p[..., np.abs(f) <= band]
    return 10.0 * np.log10(sb.max(axis=-1) / np.maximum(np.median(sb, axis=-1), 1e-300))


# ─── shaft ────────────────────────────────────────────────────────────────────


@dataclass
class Shaft:
    groups: np.ndarray  # (3, n_p) group angle estimates, rad at order 1
    b_group: np.ndarray  # (3, L) increment variance of each group's error
    weights: np.ndarray  # (3,) inverse-variance weights at 100 ms


def shaft_groups(
    z_low: np.ndarray, gates: np.ndarray, edge: int, lags: np.ndarray, i_ref: int
) -> Shaft:
    """Three disjoint-order shaft estimates from ``z_low`` ``(Ks, M, n_p)``
    (SNR-weighted LS of unwrapped phase over k), the three-cornered hat of
    their increments at every lag, and inverse-variance weights at lag
    ``lags[i_ref]``."""
    ks = np.asarray(SHAFT_ORDERS, dtype=np.float64)
    phi = unwrap_censored(z_low)
    w = np.where(gates >= GATE_DB, 10.0 ** (gates / 10.0), 0.0)  # (Ks, M)
    grp = np.arange(ks.size) % 3
    groups = []
    for g in range(3):
        wg = w * (grp == g)[:, None]
        den = float(np.einsum("km,k->", wg, ks**2))
        if den <= 0:
            raise RuntimeError(f"shaft group {g} has no gated order")
        groups.append(np.einsum("km,k,kmt->t", wg, ks, phi) / den)
    th = np.stack(groups)
    core = th[:, edge:-edge]
    b = np.zeros((3, lags.size))
    for i, lag in enumerate(lags):
        d = core[:, lag:] - core[:, :-lag]
        p01, p02, p12 = (float(np.var(d[a] - d[c])) for a, c in ((0, 1), (0, 2), (1, 2)))
        b[:, i] = 0.5 * np.array([p01 + p02 - p12, p01 + p12 - p02, p02 + p12 - p01])
    b = np.maximum(b, 1e-12)
    inv = 1.0 / b[:, i_ref]
    return Shaft(groups=th, b_group=b, weights=inv / inv.sum())


def shaft_combo(sh: Shaft, use: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """``(theta_hat (n_p,), B(tau) (L,))`` of the inverse-variance combination
    of the groups ``use``."""
    w = sh.weights[list(use)]
    w = w / w.sum()
    return w @ sh.groups[list(use)], (w**2) @ sh.b_group[list(use)]


# ─── statistics of one order ──────────────────────────────────────────────────


def lag_stats(y: np.ndarray, lags: np.ndarray) -> dict[str, np.ndarray]:
    """Per microphone of ``y`` ``(M, T)``: ``C0``, complex ``C(lag)``, the
    jackknife error of ``|C|/C0`` over :data:`N_BLOCKS` time blocks, the
    amplitude-weighted circular variance (the old estimator) and the robust
    log-amplitude increment variance."""
    m, t = y.shape
    p = np.abs(y) ** 2
    c0 = p.mean(axis=1)
    edges = np.linspace(0, t, N_BLOCKS + 1).astype(int)
    c = np.zeros((m, lags.size), dtype=np.complex128)
    sig = np.zeros((m, lags.size))
    vcirc = np.zeros((m, lags.size))
    vamp = np.zeros((m, lags.size))
    la = np.log(np.abs(y) + 1e-300)
    for i, lag in enumerate(lags):
        prod = y[:, lag:] * np.conj(y[:, :-lag])
        c[:, i] = prod.mean(axis=1)
        vcirc[:, i] = -2.0 * np.log(
            np.clip(np.abs(prod.sum(axis=1)) / np.abs(prod).sum(axis=1), 1e-8, 1.0)
        )
        rb = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            h = min(hi, t - lag)
            if h - lo < 16:
                continue
            rb.append(prod[:, lo:h].mean(axis=1) / p[:, lo:hi].mean(axis=1))
        rb_a = np.stack(rb)
        nb = rb_a.shape[0]
        sig[:, i] = np.sqrt(
            np.sum(np.abs(rb_a - rb_a.mean(axis=0)) ** 2, axis=0) / max(nb * (nb - 1), 1)
        )
        d = la[:, lag:] - la[:, :-lag]
        mad = np.median(np.abs(d - np.median(d, axis=1, keepdims=True)), axis=1)
        vamp[:, i] = (1.4826 * mad) ** 2
    return {"c0": c0, "c": c, "sig": sig, "v_circ": vcirc, "v_amp": vamp}


def fit_amplitude(taus: np.ndarray, v: np.ndarray, band: float) -> dict[str, float] | None:
    """``v_m(tau) = 2 s2A (1 - e^{-2 pi gA tau}) + c_m`` jointly over the rows
    of ``v`` ``(M, L)`` (resolved lags only): grid in ``gA``, NNLS for
    ``(s2A, c_m)``."""
    m, n_l = v.shape
    if m == 0 or n_l < 4:
        return None
    rhs = v.reshape(-1)
    eye = np.kron(np.eye(m), np.ones((n_l, 1)))
    best: tuple[float, float, float] | None = None
    for ga in np.geomspace(0.02, 2.0 * band, 80):
        col = np.tile(2.0 * -np.expm1(-2.0 * np.pi * ga * taus), m)[:, None]
        sol, rn = optimize.nnls(np.hstack([col, eye]), rhs)
        if best is None or rn < best[0]:
            best = (float(rn), float(ga), float(sol[0]))
    assert best is not None
    rn, ga, s2a = best
    return {"s2_amp": s2a, "g_amp_hz": ga, "rms": rn / math.sqrt(rhs.size), "n": int(rhs.size)}


@dataclass
class OrderData:
    """What the phase fit needs for one order: the gated microphones' rows."""

    k: int
    taus: np.ndarray  # (L,) s, nonzero lags
    rho: np.ndarray  # (M', L) |C|/C0
    sig_c: np.ndarray  # (M', L) jackknife error of the complex ratio
    c0: np.ndarray  # (M',)
    n0: np.ndarray  # (M',) two-sided white floor density at the order
    kappa: np.ndarray  # (L,)
    leak: np.ndarray  # (L,) k^2 B(tau)
    s2a: float
    ga: float
    nu0: np.ndarray  # (M',) in-band power of the neighbouring lines' pedestal tails
    nu: np.ndarray  # (M', L) complex: their autocorrelation through the window


def _model_rho(
    win: Window, od: OrderData, s2p: np.ndarray, gp: np.ndarray, n0: np.ndarray
) -> np.ndarray:
    """Expected ``|C(tau)|/C0`` (Rician-biased by the estimator noise) per
    parameter pair: ``(G, M', L)``."""
    taus0 = np.concatenate([[0.0], od.taus])
    rt = smoothed_lag(win, taus0, od.s2a, od.ga, s2p, gp)  # (G, L+1)
    q = od.c0 - n0 * win.kappa0 - od.nu0  # (M',) in-band line power
    p = q[None, :] / rt[:, :1]  # (G, M')
    line = p[:, :, None] * rt[:, None, 1:] * np.exp(-0.5 * od.leak)[None, None, :]
    rest = n0[:, None] * od.kappa[None, :] + od.nu  # (M', L)
    rho = np.abs(line + rest[None]) / od.c0[None, :, None]
    return np.sqrt(rho**2 + od.sig_c[None] ** 2)


def fit_phase(
    win: Window,
    od: OrderData,
    lag_mask: np.ndarray | None = None,
    n0_scale: float = 1.0,
    start: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Bounded least squares of ``(log s2, log g)`` of the phase OU, from the
    best point of an 18 x 18 grid or from ``start = (s2, g)``."""
    n0 = od.n0 * n0_scale
    ok_m = od.c0 - n0 * win.kappa0 - od.nu0 > 0
    mask = np.ones(od.taus.size, bool) if lag_mask is None else lag_mask
    rho = od.rho[ok_m][:, mask]
    sig = np.sqrt(od.sig_c[ok_m][:, mask] ** 2 + RHO_SIGMA_FLOOR**2)
    if rho.size < 3:
        return {"ok": False, "reason": "no microphone above the floor"}
    sub = OrderData(
        k=od.k,
        taus=od.taus[mask],
        rho=rho,
        sig_c=od.sig_c[ok_m][:, mask],
        c0=od.c0[ok_m],
        n0=n0[ok_m],
        kappa=od.kappa[mask],
        leak=od.leak[mask],
        s2a=od.s2a,
        ga=od.ga,
        nu0=od.nu0[ok_m],
        nu=od.nu[ok_m][:, mask],
    )
    g_hi = G_HI_BANDS * win.band
    lo = np.log([S2_BOUNDS[0], G_LO])
    hi = np.log([S2_BOUNDS[1], g_hi])

    def resid(theta: np.ndarray) -> np.ndarray:
        r = _model_rho(win, sub, np.exp(theta[:1]), np.exp(theta[1:2]), sub.n0)[0]
        return ((rho - r) / sig).reshape(-1)

    if start is None:
        gs2, gg = np.meshgrid(np.linspace(lo[0], hi[0], 18), np.linspace(lo[1], hi[1], 18))
        gs2, gg = gs2.reshape(-1), gg.reshape(-1)
        chi = np.empty(gs2.size)
        for a in range(0, gs2.size, 64):
            r = _model_rho(win, sub, np.exp(gs2[a : a + 64]), np.exp(gg[a : a + 64]), sub.n0)
            chi[a : a + 64] = np.sum(((rho[None] - r) / sig[None]) ** 2, axis=(1, 2))
        x0 = np.array([gs2[int(np.argmin(chi))], gg[int(np.argmin(chi))]])
    else:
        x0 = np.log(np.asarray(start, dtype=np.float64))
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
    sol = optimize.least_squares(resid, x0, bounds=(lo, hi), x_scale=1.0)
    s2, g = float(math.exp(sol.x[0])), float(math.exp(sol.x[1]))
    n_pts, dof = int(rho.size), max(int(rho.size) - 2, 1)
    chi2 = float(np.sum(sol.fun**2))
    se_log_w = float("nan")
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * max(chi2 / dof, 1.0)
        grad = np.array([1.0 - s2 * math.exp(-s2) / -math.expm1(-s2), 1.0])
        se_log_w = float(math.sqrt(max(grad @ cov @ grad, 0.0)))
    except np.linalg.LinAlgError:
        pass
    return {
        "ok": bool(sol.success),
        "s2_phase": s2,
        "g_phase_hz": g,
        "W_hz": float(pedestal_width(g, s2)),
        "se_log_W": se_log_w,
        "chi2_red": chi2 / dof,
        "n_points": n_pts,
        "n_mics": int(ok_m.sum()),
        "detected": s2 >= DETECT_S2,
        "g_unresolved": g >= G_UNRESOLVED_BANDS * win.band,
        "at_bound": bool(np.any(np.abs(sol.x - lo) < 1e-3) or np.any(np.abs(sol.x - hi) < 1e-3)),
    }


# ─── one recording ────────────────────────────────────────────────────────────


def gap_levels(f_w: np.ndarray, p_w: np.ndarray, rate: float, k_top: int) -> np.ndarray:
    """``(M, k_top + 1)`` two-sided density at the half-integer orders
    ``j + 1/2``, ``j = 0..k_top``: the median of the Welch spectrum within
    ``±GAP_FRAC f0`` (onesided density halved)."""
    out = np.empty((p_w.shape[0], k_top + 1))
    for j in range(k_top + 1):
        sel = np.abs(f_w - (j + 0.5) * rate) <= GAP_FRAC * rate
        out[:, j] = 0.5 * np.median(p_w[:, sel], axis=1)
    return out


def pedestal_psd(f: np.ndarray, s2a: float, ga: float, s2p: float, gp: float) -> np.ndarray:
    """Two-sided density (1/Hz) at offsets ``f`` of the pedestal of a
    unit-power line with lag law ``R`` (the core excluded): the Lorentzian
    expansion of the two OU factors, half-widths ``m gA + n gP`` with Poisson
    weights."""
    from scipy.stats import poisson

    def terms(s2: float) -> np.ndarray:
        return np.arange(int(s2 + 8.0 * math.sqrt(s2) + 10.0))

    m, nn = terms(s2a), terms(s2p)
    w = (poisson.pmf(m, s2a)[:, None] * poisson.pmf(nn, s2p)[None, :]).reshape(-1)
    gam = (m[:, None] * ga + nn[None, :] * gp).reshape(-1)
    keep = (w > 1e-12) & (gam > 0)
    w, gam = w[keep], gam[keep]
    f2 = np.asarray(f, dtype=np.float64)[..., None] ** 2
    return np.sum(w * gam / (math.pi * (gam**2 + f2)), axis=-1)


def _round(a: Any, nd: int = 5) -> Any:
    arr = np.asarray(a, dtype=np.float64)
    return np.where(np.isfinite(arr), np.round(arr, nd), np.nan).tolist()


@dataclass
class _Order:
    """Per-order statistics kept between the floor passes."""

    k: int
    st: dict[str, np.ndarray]
    gates: np.ndarray
    leak: np.ndarray


def _fit_order(
    o: _Order,
    n0: np.ndarray,
    nu_full: np.ndarray,
    win: Window,
    taus: np.ndarray,
    resolved: np.ndarray,
    final: bool,
) -> dict[str, Any]:
    """Gate, amplitude fit and phase fit of one order at white floor ``n0``
    ``(M,)`` with the neighbouring lines' tails ``nu_full`` ``(M, L+1)``
    (autocorrelation through the window at lag 0 and ``taus``); ``final``
    adds the sensitivity fits and the net variance."""
    st, k = o.st, o.k
    kap = win.kappa(taus)
    nu0, nu = nu_full[:, 0].real, nu_full[:, 1:]
    rest0 = n0 * win.kappa0 + nu0
    q_all = st["c0"] - rest0
    snr_db = 10.0 * np.log10(np.maximum(q_all, 1e-30) / rest0)
    gated = (o.gates >= GATE_DB) & (snr_db >= SNR_MIN_DB)
    out: dict[str, Any] = {"snr_db": snr_db, "gated": gated}
    amp = fit_amplitude(taus[resolved], st["v_amp"][gated][:, resolved], win.band)
    cap = float(amp_cap(np.array([k]))[0])
    if amp is not None:
        s2a, ga, src = min(amp["s2_amp"], cap), amp["g_amp_hz"], "measured"
    else:
        s2a = float(np.exp(amp_log_median(np.array([k]))[0]))
        ga = float(np.exp(gamma_amp_log_median(np.array([k]))[0]))
        src = "prior"
    out["amp"] = amp
    out["amp_used"] = {"s2": s2a, "g_hz": ga, "source": src}
    if not gated.any():
        return out
    od = OrderData(
        k=k,
        taus=taus,
        rho=np.abs(st["c"][gated]) / st["c0"][gated][:, None],
        sig_c=st["sig"][gated],
        c0=st["c0"][gated],
        n0=n0[gated],
        kappa=kap,
        leak=o.leak,
        s2a=s2a,
        ga=ga,
        nu0=nu0[gated],
        nu=nu[gated],
    )
    fit = fit_phase(win, od)
    out["phase"] = fit
    if not fit.get("ok"):
        return out
    s2p, gp = fit["s2_phase"], fit["g_phase_hz"]
    r0 = float(smoothed_lag(win, np.zeros(1), s2a, ga, np.array([s2p]), np.array([gp]))[0, 0])
    out["line_power"] = np.where(q_all > 0, q_all / r0, 0.0)  # (M,) total baseband power
    out["lag_params"] = (s2a, ga, s2p, gp)
    if not final:
        return out
    sens = 10.0 ** (N0_SENS_DB / 10.0)
    best = (s2p, gp)
    lo = fit_phase(win, od, n0_scale=1.0 / sens, start=best)
    hi = fit_phase(win, od, n0_scale=sens, start=best)
    lng = fit_phase(win, od, lag_mask=resolved, start=best)
    fit["W_floor_minus_1db"] = lo.get("W_hz")
    fit["W_floor_plus_1db"] = hi.get("W_hz")
    fit["W_long"] = lng.get("W_hz")
    fit["s2_long"] = lng.get("s2_phase")
    fit["g_long_hz"] = lng.get("g_phase_hz")
    # net phase variance: the line part of C with the window's noise, the
    # neighbours' tails, the amplitude law and the shaft leak taken out
    q = od.c0 - od.n0 * win.kappa0 - od.nu0
    rest = od.n0[:, None] * kap[None] + od.nu
    line = np.maximum(np.abs(st["c"][gated] - rest), 1e-12)
    va = 2.0 * s2a * -np.expm1(-2.0 * np.pi * ga * taus)
    with np.errstate(invalid="ignore", divide="ignore"):
        vnet = -2.0 * np.log(line / q[:, None]) - va[None] - o.leak[None]
    vnet[q <= 0] = np.nan
    fit["v_net_mics"] = _round(vnet, 4)
    fit["v_net_median"] = _round(np.nanmedian(vnet, axis=0), 4)
    return out


def neighbour_tails(
    fits: list[dict[str, Any]], win: Window, taus0: np.ndarray, rate: float, m_mics: int
) -> tuple[np.ndarray, np.ndarray]:
    """From the fitted orders ``1..K`` (``fits[k-1]``): the pedestal-tail
    density of the two adjacent lines at each half-integer order
    ``(M, K+1)``, and per order the autocorrelation at ``taus0`` of the tails
    of orders ``k +- 1`` inside its band, through the window
    ``(K, M, len(taus0))``."""
    n_k = len(fits)
    tails = np.zeros((m_mics, n_k + 1))
    f = np.linspace(-1.6 * win.band, 1.6 * win.band, 641)
    kw = band_amplitude(f, win.band) ** 2 * (f[1] - f[0])
    ph = np.exp(2j * np.pi * f[None, :] * taus0[:, None])  # (L+1, F)
    dens = np.zeros((n_k, m_mics, f.size))
    for i, ft in enumerate(fits):
        if "line_power" not in ft:
            continue
        pw, prm = ft["line_power"], ft["lag_params"]
        half = float(pedestal_psd(np.array([0.5 * rate]), *prm)[0])
        tails[:, i] += pw * half
        tails[:, i + 1] += pw * half
        if i >= 1:  # order k = i + 1 is the upper neighbour of order k - 1
            dens[i - 1] += pw[:, None] * pedestal_psd(f - rate, *prm)[None, :]
        if i + 1 < n_k:  # and the lower neighbour of order k + 1
            dens[i + 1] += pw[:, None] * pedestal_psd(f + rate, *prm)[None, :]
    return tails, (dens * kw) @ ph.T


def analyse_recording(
    bench: Bench,
    *,
    lags_ms: tuple[float, ...] = LAGS_MS,
    band_frac: float = BAND_FRAC,
    k_max: int | None = None,
    log: Any = None,
) -> dict[str, Any]:
    """The A0 measurement on one recording (unit payload of the CLI)."""
    import time

    t0 = time.time()
    fs, rate = bench.fs, bench.rate
    x = bench.x[list(bench.calm)]
    n_p = _fast_len_below(x.shape[1] // DECIM)
    n = n_p * DECIM
    x = x[:, :n]
    fs_p = fs / DECIM
    band = band_frac * rate
    win = Window.make(band)
    tau_min = reachable_lag(band)
    lag_n = np.unique(np.maximum(1, np.rint(np.asarray(lags_ms) * 1e-3 * fs_p).astype(int)))
    edge = int(EDGE_S * fs_p)
    lag_n = lag_n[lag_n < (n_p - 2 * edge) // 3]
    taus = lag_n / fs_p
    resolved = taus >= tau_min - 1e-9
    k_top = int(F_MAX_HZ // rate) if k_max is None else min(int(k_max), int(F_MAX_HZ // rate))

    # shaft from the low-order cores on a constant carrier
    spec1 = np.fft.rfft(x, axis=-1)
    z_low = np.stack([baseband(spec1, n, fs, k * rate, SHAFT_BAND_HZ, True) for k in SHAFT_ORDERS])
    del spec1
    g_low = np.stack([gate_db(z[:, edge:-edge], fs_p, SHAFT_BAND_HZ) for z in z_low])
    sh = shaft_groups(z_low, g_low, edge, lag_n, int(np.argmin(np.abs(taus - 0.1))))
    combos: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {
        u: shaft_combo(sh, u) for u in ((0, 1, 2), (1, 2), (0, 2), (0, 1))
    }
    t_full = np.arange(n)
    t_dec = np.arange(n_p) * DECIM
    th_all = combos[(0, 1, 2)][0]
    drift = np.polyfit(np.arange(n_p - 2 * edge) / fs_p, th_all[edge:-edge], 1)[0]
    f_w, p_w = signal.welch(x, fs=fs, nperseg=WELCH_SEG, axis=-1)
    gaps = gap_levels(f_w, p_w, rate, k_top)

    # carrier-tracked demodulation and lag statistics of every order
    data: list[_Order] = []
    for k in range(1, k_top + 1):
        use = (0, 1, 2)
        if k in SHAFT_ORDERS:
            gk = SHAFT_ORDERS.index(k) % 3
            use = tuple(g for g in range(3) if g != gk)
        th, b_tau = combos[use]
        rot = np.exp(-1j * float(k) * np.interp(t_full, t_dec, th))
        spec = np.fft.fft(x * rot[None, :], axis=-1)
        y = baseband(spec, n, fs, k * rate, band, False)[:, edge:-edge]
        del spec, rot
        data.append(
            _Order(k=k, st=lag_stats(y, lag_n), gates=gate_db(y, fs_p, band), leak=k**2 * b_tau)
        )
        if log is not None and k % 20 == 0:
            log(f"{bench.key} demod k={k} {time.time() - t0:.0f}s")

    # floor passes: the gap level holds the pedestal tails of the two
    # adjacent lines, and those tails also enter the band of each order; each
    # pass removes from the gaps and models in the bands the tails of the
    # previous pass's fits
    floor = gaps.copy()
    m_mics = x.shape[0]
    taus0 = np.concatenate([[0.0], taus])
    nus = np.zeros((k_top, m_mics, taus0.size), dtype=np.complex128)
    fits: list[dict[str, Any]] = []
    tail_share = np.zeros_like(gaps)
    for p in range(FLOOR_PASSES + 1):
        final = p == FLOOR_PASSES
        n0 = np.sqrt(floor[:, :-1] * floor[:, 1:])  # (M, k_top) at the orders
        fits = [
            _fit_order(o, n0[:, o.k - 1], nus[o.k - 1], win, taus, resolved, final) for o in data
        ]
        if final:
            break
        tails, nus = neighbour_tails(fits, win, taus0, rate, m_mics)
        tail_share = np.minimum(tails / gaps, 1.0 - FLOOR_MIN_FRAC)
        floor = gaps * (1.0 - tail_share)
        if log is not None:
            log(f"{bench.key} floor pass {p}: median tail share {np.median(tail_share):.2f}")

    rows: list[dict[str, Any]] = []
    for o, ft in zip(data, fits):
        st = o.st
        row: dict[str, Any] = {
            "k": o.k,
            "even": o.k % BLADES == 0,
            "motor": o.k % MOTOR_STEP == 0,
            "gate_db": _round(o.gates, 2),
            "snr_db": _round(ft["snr_db"], 2),
            "floor_tail_share": _round(0.5 * (tail_share[:, o.k - 1] + tail_share[:, o.k]), 3),
            "neighbour_share": _round(nus[o.k - 1][:, 0].real / st["c0"], 4),
            "n_gated": int(ft["gated"].sum()),
            "used": ft["gated"].tolist(),
            "rho": _round(np.abs(st["c"]) / st["c0"][:, None], 5),
            "v_circ": _round(st["v_circ"], 4),
            "v_amp": _round(st["v_amp"], 4),
            "leak": _round(o.leak, 5),
            "amp": ft["amp"],
            "amp_used": ft["amp_used"],
        }
        if "phase" in ft:
            row["phase"] = ft["phase"]
        rows.append(row)

    return {
        "key": bench.key,
        "motor": bench.key.split("_")[1],
        "throttle": int(bench.key.split("_")[2]),
        "rate_rev_s": rate,
        "rate_drift_rev_s": float(drift / (2.0 * np.pi)),
        "span_s": list(bench.span_s),
        "mics": list(bench.calm),
        "fs_phase": fs_p,
        "band_hz": band,
        "band_frac": band_frac,
        "lags_ms": _round(taus * 1e3, 3),
        "lags_resolved": resolved.tolist(),
        "tau_min_ms": tau_min * 1e3,
        "kappa_rel": _round(win.kappa(taus) / win.kappa0, 4),
        "shaft_orders": list(SHAFT_ORDERS),
        "shaft_band_hz": SHAFT_BAND_HZ,
        "shaft_b_group": _round(sh.b_group, 8),
        "shaft_weights": _round(sh.weights, 4),
        "shaft_b_all": _round(combos[(0, 1, 2)][1], 8),
        "shaft_gate_db": _round(g_low, 2),
        "floor_passes": FLOOR_PASSES,
        "orders": rows,
        "seconds": time.time() - t0,
    }


# ─── reduction ────────────────────────────────────────────────────────────────


def _q(v: np.ndarray) -> dict[str, float]:
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "median": float(np.median(v)),
        "p90": float(np.percentile(v, 90)),
        "p97_5": float(np.percentile(v, 97.5)),
        "max": float(v.max()),
    }


def _w_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per fitted order of every recording, with the reason it is
    left out of the pooled W (``None`` when usable)."""
    out = []
    for r in rows:
        for o in r["orders"]:
            ph = o.get("phase")
            if not ph or not ph.get("ok"):
                continue
            w = float(ph["W_hz"])
            w_lo, w_hi = ph.get("W_floor_minus_1db"), ph.get("W_floor_plus_1db")
            reason = None
            if o.get("motor"):
                reason = "motor_order"
            elif ph["s2_phase"] < USE_S2_MIN:
                reason = "no_pedestal"
            elif ph["g_unresolved"]:
                reason = "g_unresolved"
            elif ph["at_bound"]:
                reason = "at_bound"
            elif ph["n_mics"] < USE_MIN_MICS:
                reason = "few_mics"
            elif (
                w_lo is None
                or w_hi is None
                or max(w_lo, w_hi, w) / max(min(w_lo, w_hi, w), 1e-9) > STABLE_FACTOR**2
            ):
                reason = "floor_unstable"
            out.append(
                {
                    "key": r["key"],
                    "k": o["k"],
                    "even": o["even"],
                    "W": w,
                    "W_long": ph.get("W_long"),
                    "W_floor_minus_1db": w_lo,
                    "W_floor_plus_1db": w_hi,
                    "s2": float(ph["s2_phase"]),
                    "g": float(ph["g_phase_hz"]),
                    "excluded": reason,
                }
            )
    return out


def _scale(stats: dict[str, Any]) -> float | None:
    return stats["p97_5"] / HALF_NORMAL_Q975 if stats.get("n") else None


def reduce_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pooled ``W`` per parity over the usable orders of every recording and
    the implied half-normal scales ``c_W = p97.5 / 2.24``."""
    tab = _w_table(rows)
    reasons: dict[str, int] = {}
    for t in tab:
        reasons[str(t["excluded"])] = reasons.get(str(t["excluded"]), 0) + 1
    out: dict[str, Any] = {
        "n_recordings": len(rows),
        "recordings": sorted(r["key"] for r in rows),
        "n_orders_fitted": len(tab),
        "exclusions": reasons,
        "rules": {
            "s2_min": USE_S2_MIN,
            "min_mics": USE_MIN_MICS,
            "floor_stable_factor": STABLE_FACTOR,
            "motor_step": MOTOR_STEP,
            "g_unresolved_bands": G_UNRESOLVED_BANDS,
        },
        "half_normal_q975": HALF_NORMAL_Q975,
        "band_frac": rows[0]["band_frac"] if rows else None,
        "tau_min_ms": {r["key"]: round(float(r["tau_min_ms"]), 2) for r in rows},
        "parity": {},
    }
    for name, even in (("even", True), ("odd", False)):
        use = [t for t in tab if t["even"] == even and t["excluded"] is None]
        stats: dict[str, Any] = _q(np.array([t["W"] for t in use]))
        stats["c_W"] = _scale(stats)
        for alt in ("W_long", "W_floor_minus_1db", "W_floor_plus_1db"):
            qa: dict[str, Any] = _q(np.array([t[alt] for t in use if t[alt] is not None]))
            qa["c_W"] = _scale(qa)
            stats[alt] = qa
        loose = [
            t
            for t in tab
            if t["even"] == even and t["excluded"] in (None, "floor_unstable", "at_bound")
        ]
        ql: dict[str, Any] = _q(np.array([t["W"] for t in loose]))
        ql["c_W"] = _scale(ql)
        stats["loose"] = ql
        groups = {}
        for lo, hi, *_ in AMP_GROUPS:
            groups[f"{lo}-{min(hi, 999)}"] = _q(
                np.array([t["W"] for t in use if lo <= t["k"] <= hi])
            )
        stats["by_order_group"] = groups
        stats["by_motor"] = {
            mot: _q(np.array([t["W"] for t in use if f"_{mot}_" in t["key"]]))
            for mot in ("Motor1", "Motor2", "Motor3", "Motor4")
        }
        stats["by_throttle"] = {
            str(th): _q(np.array([t["W"] for t in use if t["key"].endswith(f"_{th}")]))
            for th in (50, 60, 70, 80, 90)
        }
        ks = sorted({t["k"] for t in use})
        stats["by_order"] = {str(k): _q(np.array([t["W"] for t in use if t["k"] == k])) for k in ks}
        out["parity"][name] = stats
    out["amplitude_crosscheck"] = _amp_crosscheck(rows)
    return out


def _amp_crosscheck(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Median ``s2A`` per amplitude order group: this fit against
    ``amplitude_jitter.json`` (0.25 s frames) on the same recordings."""
    ref = json.loads(AMP_JITTER_PATH.read_text()) if AMP_JITTER_PATH.exists() else {}
    out = {}
    for lo, hi, med, cap in AMP_GROUPS:
        mine, theirs = [], []
        for r in rows:
            for o in r["orders"]:
                if lo <= o["k"] <= hi and o.get("amp") and o["n_gated"] > 0:
                    mine.append(o["amp"]["s2_amp"])
            j = ref.get(r["key"])
            if j:
                ks = np.asarray(j["orders"])
                sm = np.array(j["sigma_m2"], dtype=float)[j["mics_good"]]
                sel = (ks >= lo) & (ks <= hi)
                v = sm[:, sel]
                theirs.extend(v[np.isfinite(v)].tolist())
        out[f"{lo}-{min(hi, 999)}"] = {
            "s2_amp_median_this": float(np.median(mine)) if mine else None,
            "s2_amp_median_jitter_json": float(np.median(theirs)) if theirs else None,
            "prior_median": med,
            "prior_cap": cap,
            "n_this": len(mine),
        }
    return out


def findings_markdown(summary: dict[str, Any]) -> str:
    """Short numbers-first report of :func:`reduce_rows`."""
    par = summary["parity"]
    lines = [
        "# Noise model v4, step A0: phase-pedestal width W",
        "",
        "Generated by `scripts/noise_v4_phase_width.py reduce` "
        "(`experiments.static_rig.phase_width`).",
        "",
        f"Recordings: {summary['n_recordings']}; orders fitted {summary['n_orders_fitted']}; "
        f"left out by reason: {summary['exclusions']} (rules: {summary['rules']}).",
        f"Band ±{summary['band_frac']} f0; resolved minimum lag per recording "
        f"{min(summary['tau_min_ms'].values(), default=float('nan')):.1f}-"
        f"{max(summary['tau_min_ms'].values(), default=float('nan')):.1f} ms.",
        "",
        "| parity | n | median W (Hz) | p90 | p97.5 | c_W = p97.5/2.24 | c_W (resolved lags only) | c_W (floor -1 / +1 dB) | c_W (loose set, n) |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]

    def f(v: Any) -> str:
        return "-" if v is None else f"{v:.2f}"

    for name in ("even", "odd"):
        s = par[name]
        lines.append(
            f"| {name} | {s['n']} | {f(s.get('median'))} | {f(s.get('p90'))} | {f(s.get('p97_5'))} "
            f"| **{f(s.get('c_W'))}** | {f(s['W_long'].get('c_W'))} "
            f"| {f(s['W_floor_minus_1db'].get('c_W'))} / {f(s['W_floor_plus_1db'].get('c_W'))} "
            f"| {f(s['loose'].get('c_W'))} ({s['loose']['n']}) |"
        )
    lines += ["", "W by order group (median / p97.5, n):", ""]
    lines += ["| k | even | odd |", "|---|---|---|"]
    for grp in par["even"]["by_order_group"]:
        e, o = par["even"]["by_order_group"][grp], par["odd"]["by_order_group"][grp]
        lines.append(
            f"| {grp} | {f(e.get('median'))} / {f(e.get('p97_5'))} ({e['n']}) "
            f"| {f(o.get('median'))} / {f(o.get('p97_5'))} ({o['n']}) |"
        )
    lines += ["", "Amplitude cross-check (median s2_amp, neper^2):", ""]
    lines += ["| k | this fit | amplitude_jitter.json | prior median / cap |", "|---|--:|--:|--:|"]
    for grp, a in summary["amplitude_crosscheck"].items():
        lines.append(
            f"| {grp} | {f(a['s2_amp_median_this'])} | {f(a['s2_amp_median_jitter_json'])} "
            f"| {a['prior_median']} / {a['prior_cap']} |"
        )
    lines += [
        "",
        "Method: carrier-tracked demodulation (shaft removed before the band filter), "
        "forward model of the band window and the between-line floor on the "
        "normalised complex autocorrelation, joint fit over the calm microphones; "
        "see the module docstring of `experiments.static_rig.phase_width`.",
        "",
    ]
    return "\n".join(lines)
