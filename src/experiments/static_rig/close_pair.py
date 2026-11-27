"""Local acoustic test of one modulated shaft against two coherent shafts.

All rates are shaft rev/s; phase is 2*pi*k*(rate*t + drift*t**2/2), with
`t=0` at the input midpoint. There is no physical rotor assignment. The
first 48% of the span fixes the nonlinear parameters; the next 26% fixes
complex gains, and neither sees the final 26%. Channels with even zero-based
indices and orders k % 4 in (2, 3) are used in the nonlinear fit. The primary
predictive score holds out time AND either microphone or order. Each time
block is downconverted independently; in particular, a zero-phase filter
cannot transport test audio into the fit. A mono input holds out orders/time.

Limits: 2–8 s (long inputs are center-cropped), first eight simultaneous mics,
shaft orders 1–12, up to 5 rev/s center separation. One seed searches on both
sides (±2.8 rev/s); two seeds set a midpoint and the search half-width is half
their gap plus 1.05 rev/s (at least 1.7). Both rates can move, regardless of
whether the seeds coincide. Fits allow up to 1.8 rev/s total linear change
per shaft per span. Omit bands unable to contain that support, modulation
(a bounded slow trend, an absolute-Hz sinusoid, or an order-scaled sinusoid)
and roll-off without overlap. Cost is three real FFTs and reduced-rate
complex regressions, with five coarse drift nodes, nine off-grid pair starts
and 21 separation nodes plus local bracketing, not a sample-rate 4-D sweep.
The release joint tracker remains a baseline: its per-order peak exclusion
assumes resolved peaks, so it is not the likelihood used below.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any, cast

import numpy as np
from scipy import fft, optimize, signal
from scipy.signal.windows import tukey

__all__ = ["fit_pair", "predict_pair"]

_MAX_SPAN = 8.0
_MIN_SPAN = 2.0
_MAX_CHANNELS = 8
_MAX_ORDER = 12
_MAX_SPLIT = 5.0
_DRIFT_CHANGE = 1.8
_AM_DEPTH = 0.82
_TRAIN_FRACTION = 0.48
_CAL_END = 0.74


@dataclass(frozen=True)
class _Band:
    order: int
    flat: float
    stop: float


def _train_order(k: int) -> bool:
    return k % 4 in (2, 3)


def _bands(
    center: float,
    radius: float,
    margin: float,
    am_max: float,
    fs: float,
    others: np.ndarray,
    train_s: float,
) -> list[_Band]:
    bands = []
    for k in range(1, _MAX_ORDER + 1):
        fc = k * center
        flat = k * (radius + margin) + am_max + 0.6 / train_s
        stop = flat + max(2.0, 0.052 * center)
        if stop >= 0.49 * center or fc + stop >= 0.48 * fs:
            continue  # a neighboring shaft order or the audio Nyquist
        lattice = np.maximum(1, np.rint(fc / others)) if others.size else others
        if np.any(np.abs(lattice * others - fc) <= stop + 0.45 * lattice + 1.0 / train_s):
            continue
        bands.append(_Band(k, flat, stop))
    return bands


@dataclass
class _View:
    data: np.ndarray  # order, time, microphone
    t: np.ndarray
    orders: np.ndarray
    effective: np.ndarray  # independent complex samples / baseband samples
    span: float

    def __post_init__(self) -> None:
        self.w = (2.0 * np.pi * self.orders[:, None]) * self.t
        self.v = self.w * (self.t / (2.0 * self.span))
        self.scale = max(float(np.vdot(self.data, self.data).real), 1e-60)

    def wave(self, r: float, d: float) -> np.ndarray:
        return np.exp(1j * (r * self.w + d * self.v))


@dataclass
class _Block:
    t: np.ndarray
    z: np.ndarray
    sigma: np.ndarray
    effective: np.ndarray
    power: np.ndarray
    df: float

    def view(
        self,
        bands: list[_Band],
        span: float,
        order_ix: np.ndarray,
        chan_ix: np.ndarray,
        sigma: np.ndarray | None = None,
    ) -> _View:
        sel = np.ix_(order_ix, chan_ix)
        floor = (self.sigma if sigma is None else sigma)[sel]
        weight = np.sqrt(self.effective[order_ix])[:, None] / floor
        return _View(
            self.z[sel].transpose(0, 2, 1) * weight[:, None, :],
            self.t,
            np.array([bands[i].order for i in order_ix]),
            self.effective[order_ix],
            span,
        )


def _block(x: np.ndarray, fs: float, t0: float, center: float, bands: list[_Band]) -> _Block:
    n = x.shape[-1]
    duration = n / fs
    df = 1.0 / duration
    spec = cast(np.ndarray, fft.rfft(x * tukey(n, alpha=0.12).astype(x.dtype), axis=-1))
    power = np.abs(spec).astype(np.float64) ** 2
    n_env = cast(int, fft.next_fast_len(int(np.ceil(2.3 * bands[-1].stop * duration)) + 3))
    elapsed = np.arange(n_env) * (duration / n_env)
    edge = max(0.076 * duration, 0.024)
    use = (elapsed >= edge) & (elapsed < duration - edge)
    t = t0 + elapsed[use]
    z = np.empty((len(bands), x.shape[0], t.size), dtype=np.complex128)
    sigma = np.empty(z.shape[:2])
    effective = np.empty(len(bands))
    q = np.rint(cast(np.ndarray, fft.fftfreq(n_env)) * n_env).astype(int)
    for i, band in enumerate(bands):
        fc = band.order * center
        b0 = int(round(fc / df))
        bins = b0 + q
        offs = np.abs(bins * df - fc)
        keep = offs < band.stop
        taper = 0.5 + 0.5 * np.cos(
            np.pi * np.clip((offs[keep] - band.flat) / (band.stop - band.flat), 0.0, 1.0)
        )
        coeff = np.zeros((x.shape[0], n_env), dtype=np.complex128)
        coeff[:, keep] = spec[:, bins[keep]] * taper
        # The carrier has a common phase origin even though each FFT restarts.
        rot = np.exp(2j * np.pi * ((b0 * df - fc) * elapsed[use] - fc * t0))
        z[i] = (n_env / n) * cast(np.ndarray, fft.ifft(coeff, axis=-1))[:, use] * rot
        bw2 = np.sum(taper**2)
        effective[i] = min(1.0, bw2**2 / (n_env * np.sum(taper**4)))
        # Estimate a diagonal spectral noise model, with the central comb
        # masked away from the outer local floor whenever enough bins remain.
        lo = max(1, int(np.ceil((fc - 0.47 * center) / df)))
        hi = min(power.shape[-1], int(np.floor((fc + 0.47 * center) / df)) + 1)
        side = np.arange(lo, hi)
        outer = side[np.abs(side * df - fc) >= min(band.flat, 0.31 * center)]
        side = outer if outer.size >= 10 else side
        p = np.median(power[:, side], axis=-1) / np.log(2.0)
        # Avoid division by zero (e.g. dead mics), with a relative spectral floor.
        p = np.maximum(p, np.maximum(np.max(power[:, lo:hi], axis=-1) * 1e-10, 1e-48))
        sigma[i] = np.sqrt(p * bw2) / n
    return _Block(t, z, sigma, effective, power, df)


def _cross(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.einsum("kt,ktc->kc", a.conj(), y, optimize=False)


def _loss(view: _View, residue: np.ndarray) -> float:
    return float(np.vdot(residue, residue).real / view.scale)


def _derivatives(view: _View, residue: np.ndarray, term: np.ndarray) -> tuple[float, float]:
    # Variable projection: the part through the fitted gains vanishes.
    p = 2.0 * np.einsum("ktc,ktc->kt", residue.conj(), term, optimize=False).imag / view.scale
    return float(np.sum(p * view.w)), float(np.sum(p * view.v))


def _pair(view: _View, x: np.ndarray, gradient: bool = False) -> tuple:
    r1, r2, d1, d2 = x
    b1, b2 = view.wave(r1, d1), view.wave(r2, d2)
    # Thin two-column complex QR (modified Gram-Schmidt), simultaneous RHS
    # for all whitened microphones. No nearly singular normal equations.
    n = view.t.size
    alpha = np.mean(b1.conj() * b2, axis=1)
    orth = b2 - alpha[:, None] * b1
    norm2 = np.sum(np.abs(orth) ** 2, axis=1)
    full_rank = norm2 > 1e-10 * n
    beta = _cross(orth, view.data)
    g2 = np.divide(beta, norm2[:, None], out=np.zeros_like(beta), where=full_rank[:, None])
    g1 = _cross(b1, view.data) / n - alpha[:, None] * g2
    term1 = b1[:, :, None] * g1[:, None, :]
    term2 = b2[:, :, None] * g2[:, None, :]
    residue = view.data - term1 - term2
    loss = _loss(view, residue)
    if gradient:
        dr1, dd1 = _derivatives(view, residue, term1)
        dr2, dd2 = _derivatives(view, residue, term2)
        return loss, np.array([dr1, dr2, dd1, dd2])
    return loss, g1, g2, residue, norm2


def _features(view: _View, f: float, scaled: bool) -> np.ndarray:
    if f == 0.0:
        return np.broadcast_to(
            (view.t / view.span)[None, :, None], (view.orders.size, view.t.size, 1)
        )
    angle = (2.0 * np.pi * f) * view.t
    if scaled:
        angle = view.orders[:, None] * angle
    else:
        angle = np.broadcast_to(angle, (view.orders.size, angle.size))
    return np.stack((np.cos(angle), np.sin(angle)), axis=-1)


def _bounded_envelope(features: np.ndarray, y: np.ndarray, depth: float) -> np.ndarray:
    """Complex QR least squares with ||side gains|| <= depth * |carrier|.

    Given QR, this is Euclidean projection with one indefinite quadratic
    constraint. Its Lagrange multiplier lies before the single negative
    eigenvalue's pole. This bounds the envelope on *unseen* samples without
    shrinking valid low-frequency modulation merely because calibration
    covers less than one modulation cycle.
    """
    design = np.concatenate((np.ones((*features.shape[:2], 1)), features), axis=-1)
    q, R = np.linalg.qr(design)
    ri = np.linalg.inv(R)
    rhs = np.einsum("ktp,ktc->kpc", q, y, optimize=False)
    gain = np.einsum("kpq,kqc->kpc", ri, rhs, optimize=False)
    active = np.sum(np.abs(gain[:, 1:]) ** 2, axis=1) > (depth * np.abs(gain[:, 0])) ** 2
    order, channel = np.nonzero(active)
    if not order.size:
        return gain
    use, back = np.unique(order, return_inverse=True)
    ru = ri[use]
    G = ru.swapaxes(1, 2) @ (ru * np.r_[-(depth**2), np.ones(ru.shape[-1] - 1)][None, :, None])
    value, basis = np.linalg.eigh(G)
    e = (value / (-value[:, :1]))[back]
    vectors = basis[back]
    a = np.einsum("nip,ni->np", vectors, rhs[order, :, channel], optimize=False)
    a2 = np.abs(a) ** 2
    lo = np.zeros(order.size)
    hi = np.full(order.size, 1.0 - 1e-12)
    stay = np.sum(e * a2 / (1.0 + hi[:, None] * e) ** 2, axis=1) > 0.0
    for _ in range(32):
        x = 0.5 * (lo + hi)
        above = np.sum(e * a2 / (1.0 + x[:, None] * e) ** 2, axis=1) > 0.0
        lo = np.where(above, x, lo)
        hi = np.where(above, hi, x)
    a_fit = a / (1.0 + hi[:, None] * e)
    if np.any(stay):
        # If the datum is orthogonal to the negative eigendirection, the
        # optimum can sit at the pole. Complete that coordinate to the
        # boundary; preserve its phase when identifiable.
        aa = a_fit[stay]
        aa[:, 1:] = a[stay, 1:] / (1.0 + e[stay, 1:])
        aa[:, 0] = np.sqrt(np.sum(e[stay, 1:] * np.abs(aa[:, 1:]) ** 2, axis=1)) * np.exp(
            1j * np.angle(a[stay, 0])
        )
        a_fit[stay] = aa
    gain[order, :, channel] = np.einsum("npi,ni->np", ri[order] @ vectors, a_fit, optimize=False)
    return gain


def _am(
    view: _View,
    r: float,
    d: float,
    f: float,
    scaled: bool = False,
    gradient: bool = False,
    trend_extent: float = 0.5,
) -> tuple:
    b = view.wave(r, d)
    y = b.conj()[:, :, None] * view.data
    features = _features(view, f, scaled)
    # A complex envelope permits gradual AM and PM, but cannot arbitrarily
    # cancel the carrier. ||sin,cos|| = 1; for a linear trend use its actual
    # time-domain extent. The bound is relative to the stationary carrier.
    depth = _AM_DEPTH / trend_extent if f == 0.0 else _AM_DEPTH
    gain = _bounded_envelope(features, y, depth)
    carrier, am_gain = gain[:, 0], gain[:, 1:]
    term = b[:, :, None] * (
        carrier[:, None, :] + np.einsum("ktp,kpc->ktc", features, am_gain, optimize=False)
    )
    residue = view.data - term
    loss = _loss(view, residue)
    if gradient:
        return loss, np.array(_derivatives(view, residue, term))
    return loss, carrier, am_gain


def _candidates(view: _View, radius: float) -> list[tuple[float, float]]:
    """Coarse matched-filter proposals at five drift values, then off-grid fits."""
    mid = float(view.t.mean())
    kmax = int(view.orders.max())
    width = float(np.ptp(view.t))
    step = max(0.024, 0.43 / (kmax * width))
    n = min(351, 1 + int(np.ceil(2.0 * radius / step)))
    rates = np.linspace(-radius, radius, n)
    aggregate: list[tuple[float, float, float]] = []
    local: list[tuple[float, float, float]] = []
    for d in np.linspace(-_DRIFT_CHANGE, _DRIFT_CHANGE, 5):
        rr = rates - d * mid / view.span
        within = np.abs(rr) <= radius
        rr = rr[within]
        power = np.empty((view.orders.size, rr.size))
        for k in range(view.orders.size):
            b = np.exp(-1j * (rr[:, None] * view.w[k] + d * view.v[k]))
            z = np.einsum("nt,tc->nc", b, view.data[k], optimize=False)
            power[k] = np.sum(np.abs(z) ** 2, axis=1) / view.t.size
        for line, target in [(power.sum(axis=0), aggregate), *[(p, local) for p in power]]:
            peaks = np.r_[0, signal.find_peaks(line)[0], line.size - 1]
            ix = peaks[np.argsort(-line[peaks], kind="stable")[: 5 if target is aggregate else 2]]
            target.extend((float(line[j]), float(rr[j]), float(d)) for j in ix)
    out: list[tuple[float, float]] = []
    for group, count in ((aggregate, 14), (local, 24)):
        for _, r, d in sorted(group, reverse=True):
            if all(abs(r - a) > 0.055 or abs(d - b) > 0.30 for a, b in out):
                out.append((r, d))
            if len(out) >= count:
                break
    return out


def _bounds_coordinates(x: np.ndarray, radius: float) -> np.ndarray:
    rlo, rhi, dlo, dhi = x if x[0] <= x[1] else x[[1, 0, 3, 2]]
    sep = float(np.clip(rhi - rlo, 0.0, min(_MAX_SPLIT, 2.0 * radius - 1e-5)))
    return np.array([np.clip((rlo + rhi) / (2.0 * radius - sep), -1.0, 1.0), sep, dlo, dhi])


def _physical(p: np.ndarray, radius: float) -> np.ndarray:
    mid, sep, d1, d2 = p
    m = mid * (radius - 0.5 * sep)
    return np.array([m - 0.5 * sep, m + 0.5 * sep, d1, d2])


def _pair_objective(p: np.ndarray, view: _View, radius: float) -> tuple[float, np.ndarray]:
    loss, j = _pair(view, _physical(p, radius), gradient=True)
    mid, sep = p[:2]
    return loss, np.array(
        [
            (j[0] + j[1]) * (radius - 0.5 * sep),
            -0.5 * (1.0 + mid) * j[0] + 0.5 * (1.0 - mid) * j[1],
            j[2],
            j[3],
        ]
    )


def _fit_two(
    view: _View, peaks: list[tuple[float, float]], seed: np.ndarray, radius: float
) -> tuple[np.ndarray, list[np.ndarray], float]:
    starts = [
        np.array([r, s, d, e])
        for (r, d), (s, e) in combinations(peaks, 2)
        if abs(r - s) <= _MAX_SPLIT
    ]
    starts += [
        np.array([r - sep / 2, r + sep / 2, d, d])
        for r, d in peaks[:8] + [(float(np.mean(seed)), 0.0)]
        for sep in (0.045, 0.12, 0.29, 0.60, 1.25, 2.5, 4.2)
        if abs(r) + sep / 2 <= radius
    ]
    if seed.size == 2:
        starts.append(np.array([seed[0], seed[1], 0.0, 0.0]))
    trials = [_bounds_coordinates(x, radius) for x in starts]
    trials.sort(key=lambda p: _pair(view, _physical(p, radius))[0])
    chosen: list[np.ndarray] = []
    solutions: list[tuple[float, np.ndarray]] = []
    bound = [
        (-1.0, 1.0),
        (0.0, min(_MAX_SPLIT, 2.0 * radius - 1e-5)),
        (-_DRIFT_CHANGE, _DRIFT_CHANGE),
        (-_DRIFT_CHANGE, _DRIFT_CHANGE),
    ]
    for p in trials:
        if any(np.linalg.norm((p - q) * [3.5, 1.0, 0.38, 0.38]) < 0.16 for q in chosen):
            continue
        chosen.append(p)
        opt = optimize.minimize(
            _pair_objective,
            p,
            args=(view, radius),
            jac=True,
            bounds=bound,
            method="L-BFGS-B",
            options={"maxiter": 52, "ftol": 2e-9, "gtol": 2e-6},
        )
        solutions.append((float(opt.fun), opt.x))
        if len(chosen) >= 9:
            break
    solutions.sort(key=lambda v: v[0])
    best, p = solutions[0]
    return p, [x for _, x in solutions] + trials[:14], best


def _fit_null(
    view: _View,
    peaks: list[tuple[float, float]],
    pair: np.ndarray,
    radius: float,
    am_max: float,
    order_max: int,
) -> tuple[float, float, float, float, bool]:
    rd_bound = [(-radius, radius), (-_DRIFT_CHANGE, _DRIFT_CHANGE)]
    rates = peaks[:4] + [(pair[0], pair[2]), (pair[1], pair[3])]
    trend = [
        optimize.minimize(
            lambda x: _am(view, x[0], x[1], 0.0, gradient=True),
            rd,
            jac=True,
            bounds=rd_bound,
            method="L-BFGS-B",
            options={"maxiter": 35, "ftol": 2e-9},
        )
        for rd in rates
    ]
    trend.sort(key=lambda opt: opt.fun)
    rate0, drift0 = map(float, trend[0].x)
    best = (float(trend[0].fun), rate0, drift0, 0.0, False)
    rd = [tuple(opt.x) for opt in trend[:2]] + rates[-2:]
    for scaled in (False, True):
        # The frequency is common in Hz or rev/s, never free per order. The
        # order-scaled alternative makes a genuine one-sided AM/PM sideband
        # a serious competitor, even where it is indistinguishable from two
        # fixed shafts with unconstrained complex mixing.
        lever = float(view.orders.max()) if scaled else 1.0
        lower, upper = (0.018, min(0.66, am_max / order_max)) if scaled else (0.25, am_max)
        grid = np.linspace(
            lower, upper, max(5, 1 + int((upper - lower) * lever * np.ptp(view.t) * 2.4))
        )
        initials = [
            (float(_am(view, r, d, f, scaled)[0]), float(r), float(d), float(f))
            for r, d in rd
            for f in grid
        ]
        if scaled:
            # A sideband on one fitted line can sit exactly on the other.
            sep = float(pair[1] - pair[0])
            if lower <= sep <= upper:
                initials.extend(
                    (float(_am(view, r, d, sep, True)[0]), r, d, sep) for r, d in rates[-2:]
                )
        initials.sort()
        chosen: list[tuple[float, float, float]] = []
        for _, r, d, f in initials:
            if any(
                abs(r - a) + 0.25 * abs(d - b) + 0.45 * lever * abs(f - c) < 0.19
                for a, b, c in chosen
            ):
                continue
            chosen.append((r, d, f))

            def objective(
                x: np.ndarray, scaled: bool = scaled, lever: float = lever
            ) -> tuple[float, np.ndarray]:
                a, b, c = x
                value, jac = _am(view, a, b, c, scaled, gradient=True)
                h = 1e-4 / lever
                jf = (_am(view, a, b, c + h, scaled)[0] - _am(view, a, b, c - h, scaled)[0]) / (
                    2.0 * h
                )
                return value, np.r_[jac, jf]

            opt = optimize.minimize(
                objective,
                (r, d, f),
                jac=True,
                method="L-BFGS-B",
                bounds=[*rd_bound, (lower, upper)],
                options={"maxiter": 43, "ftol": 2e-9, "gtol": 3e-6},
            )
            rate, drift, freq = map(float, opt.x)
            best = min(best, (float(opt.fun), rate, drift, freq, scaled))
            if len(chosen) >= 4:
                break
    return best


def _profile(
    view: _View, p: np.ndarray, modes: list[np.ndarray], radius: float, best: float
) -> tuple[np.ndarray, list[float], list[list[float]], float]:
    """Profile separation with mid-rate and both drifts free, including zero."""
    gap_max = min(_MAX_SPLIT, 2.0 * radius - 1e-5)
    support: dict[float, tuple[float, np.ndarray]] = {float(p[1]): (best, p)}
    objective_bounds = [
        (-1.0, 1.0),
        (-_DRIFT_CHANGE, _DRIFT_CHANGE),
        (-_DRIFT_CHANGE, _DRIFT_CHANGE),
    ]

    def at(gap: float) -> tuple[float, np.ndarray]:
        if gap in support:
            return support[gap]
        warm = min(support.items(), key=lambda kv: abs(kv[0] - gap))[1][1]
        # Screening fixed-gap projections at multiple mid-rates/drift basins
        # prevents a single continuation branch masquerading as the profile.
        starts = [q[[0, 2, 3]] for q in [warm, p, *modes]]
        starts.extend(np.array([m, 0.0, 0.0]) for m in np.linspace(-1.0, 1.0, 13))
        starts.sort(key=lambda z: _pair(view, _physical(np.insert(z, 1, gap), radius))[0])

        def objective(z: np.ndarray) -> tuple[float, np.ndarray]:
            val, jac = _pair_objective(np.insert(z, 1, gap), view, radius)
            return val, jac[[0, 2, 3]]

        runs = []
        kept = []
        for z in starts:
            if any(np.linalg.norm((z - q) * [4.0, 0.4, 0.4]) < 0.12 for q in kept):
                continue
            kept.append(z)
            opt = optimize.minimize(
                objective,
                z,
                jac=True,
                method="L-BFGS-B",
                bounds=objective_bounds,
                options={"maxiter": 30, "ftol": 3e-8, "gtol": 4e-6},
            )
            runs.append((float(opt.fun), np.insert(opt.x, 1, gap)))
            if len(kept) == 2:
                break
        support[gap] = min(runs, key=lambda v: v[0])
        return support[gap]

    for gap in np.linspace(0.0, gap_max, 21):
        at(float(gap))
    best, p = min(support.values(), key=lambda v: v[0])
    # A basin recovered by profiling need not have seeded the unconstrained fit.
    opt = optimize.minimize(
        _pair_objective,
        p,
        args=(view, radius),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-1.0, 1.0), (0.0, gap_max), *objective_bounds[1:]],
        options={"maxiter": 52, "ftol": 1e-10, "gtol": 8e-7},
    )
    if opt.fun < best:
        best, p = float(opt.fun), opt.x
    support[float(p[1])] = (best, p)
    n_eff = max(1.0, view.t.size * view.data.shape[-1] * float(view.effective.sum()))
    # A nominal profile level, not a calibrated confidence interval or an LRT.
    # Residual excess above the local noise floor inflates the tolerance.
    # Bisection may reveal a lower mode missed by the initial grid. If so,
    # repeat with the improved optimum rather than quoting negative excess
    # loss or an interval centered on the wrong mode.
    for _ in range(2):
        cutoff = best + 1.92073 * max(1.0 / view.scale, best / n_eff)
        dense = sorted(support)
        for a, b in zip(dense[:-1], dense[1:]):
            fa, fb = support[a][0] <= cutoff, support[b][0] <= cutoff
            if fa == fb:
                continue
            for _ in range(9):
                m = 0.5 * (a + b)
                if (at(m)[0] <= cutoff) == fa:
                    a = m
                else:
                    b = m
        lowest, mode = min(support.values(), key=lambda v: v[0])
        if lowest >= best:
            break
        best, p = lowest, mode
    cutoff = best + 1.92073 * max(1.0 / view.scale, best / n_eff)
    inside = [g for g, (loss, _) in support.items() if loss <= cutoff]
    profile = [
        [float(g), float((loss - best) * view.scale)] for g, (loss, _) in sorted(support.items())
    ]
    return p, [float(min(inside)), float(max(inside))], profile, float((cutoff - best) * view.scale)


def _partition(channels: int, bands: list[_Band]) -> tuple[np.ndarray, ...]:
    ti = np.array([i for i, band in enumerate(bands) if _train_order(band.order)], dtype=int)
    vi = np.array([i for i, band in enumerate(bands) if not _train_order(band.order)], dtype=int)
    return ti, vi, np.arange(0, channels, 2), np.arange(1, channels, 2)


def _predict(
    cal: _Block,
    test: _Block,
    bands: list[_Band],
    channels: int,
    span: float,
    pair: np.ndarray,
    null: tuple[float, float, float, bool],
    trend_extent: float = 0.5,
) -> dict[str, Any]:
    ti, vi, tc, vc = _partition(channels, bands)
    groups = {
        "seen_both": (ti, tc),
        "unseen_orders": (vi, tc),
        "unseen_channels": (ti, vc),
        "unseen_both": (vi, vc),
    }
    losses: dict[str, dict[str, float | int]] = {}
    total_pair = total_null = 0.0
    for name, (oi, ci) in groups.items():
        if not oi.size or not ci.size:
            continue
        fit = cal.view(bands, span, oi, ci)
        pred = test.view(bands, span, oi, ci, cal.sigma)
        # Both views share calibration-only noise scales. Their effective
        # time-grid densities can differ slightly with integer block lengths.
        ratio = np.sqrt(pred.effective / fit.effective)[:, None]
        _, g1, g2, _, _ = _pair(fit, pair)
        b1, b2 = pred.wave(pair[0], pair[2]), pred.wave(pair[1], pair[3])
        r, d, f, scaled = null
        _, gn, am = _am(fit, r, d, f, scaled, trend_extent=trend_extent)
        env = gn[:, None, :] + np.einsum(
            "ktp,kpc->ktc", _features(pred, f, scaled), am, optimize=False
        )
        ep = (
            pred.data
            - (b1[:, :, None] * g1[:, None, :] + b2[:, :, None] * g2[:, None, :])
            * ratio[:, :, None]
        )
        en = pred.data - pred.wave(r, d)[:, :, None] * env * ratio[:, :, None]
        lp = float(np.vdot(ep, ep).real)
        ln = float(np.vdot(en, en).real)
        losses[name] = {"pair_sse": lp, "null_sse": ln, "n": int(ep.size)}
        if name != "seen_both":
            total_pair += lp
            total_null += ln
    score = float(1.0 - total_pair / total_null) if total_null > 1e-30 else None
    return {"score": score, "heldout": losses}


def _support(view: _View, pair: np.ndarray) -> tuple[list[int], list[int]]:
    _, g1, g2, residual, norm2 = _pair(view, pair)
    channels = view.data.shape[-1]
    noise = np.mean(np.abs(residual) ** 2, axis=(1, 2)) * channels
    # Conditional line energy after projecting out the other shaft. A single
    # huge line can't supply support at multiple independent orders.
    floor = np.maximum(11.0 * channels, 9.0 * noise)
    order1 = view.orders[(np.sum(np.abs(g1) ** 2, axis=1) * norm2) > floor]
    order2 = view.orders[(np.sum(np.abs(g2) ** 2, axis=1) * norm2) > floor]
    return order1.tolist(), order2.tolist()


def _aliases(
    train: _Block,
    pair: np.ndarray,
    center: float,
    span: float,
    support: tuple[list[int], list[int]],
    others: np.ndarray,
    blades: int,
) -> list[dict[str, Any]]:
    """Integer aliases on a common physical band, not a wider high-rate comb.

    Higher aliases cannot explain non-divisible orders. Lower aliases always
    explain the fitted orders by relabeling: only extra fractional teeth can
    favor them acoustically. Otherwise we report the primitive observed comb;
    exclusion of all lower aliases with missing harmonics is not identifiable.
    """

    def extra_teeth(i: int, q: int) -> list[int]:
        rate = center + pair[i]
        drift = pair[i + 2] / span
        mid = float(train.t.mean())
        extent = 0.5 * float(np.ptp(train.t))
        observed = []
        other_rates = np.r_[others, center + pair[1 - i] + pair[3 - i] * mid / span]
        power = train.power
        df = train.df
        for k in range(1, min(_MAX_ORDER, q * (blades + 2)) + 1):
            if k % q == 0:
                continue
            fc = (rate + drift * mid) * k / q
            width = 1.4 * df + abs(drift) * extent * k / q
            tooth = np.maximum(1, np.rint(fc / other_rates))
            if np.any(abs(fc - tooth * other_rates) <= width + 1.4 * df + 0.3 * tooth):
                continue
            lo, hi = max(1, int(np.ceil((fc - width) / df))), int((fc + width) / df) + 1
            if hi <= lo or hi >= power.shape[-1]:
                continue
            h = max(8, int(0.24 * rate / q / df))
            a = max(1, (lo + hi) // 2 - h)
            b = min(power.shape[-1], (lo + hi) // 2 + h + 1)
            side = np.concatenate(
                (power[:, a : max(a, lo - 2)], power[:, min(b, hi + 2) : b]), axis=1
            )
            if side.shape[1] < 7:
                continue
            local = np.maximum(np.median(side, axis=1) / np.log(2.0), 1e-48)
            if float(np.max(np.max(power[:, lo:hi], axis=1) / local)) >= 19.0:
                observed.append(k)
        return observed

    out = []
    for i, orders in enumerate(support):
        tested = []
        for q in sorted({2, 3, max(2, blades)}):
            nonmultiple = [k for k in orders if k % q != 0]
            extra = extra_teeth(i, q)
            tested.extend(
                [
                    {
                        "ratio": float(q),
                        "exclusive_orders": nonmultiple,
                        "ambiguous": not nonmultiple,
                    },
                    {"ratio": 1.0 / q, "extra_orders": extra, "ambiguous": len(extra) >= 2},
                ]
            )
        out.append({"rate_rps": float(center + pair[i]), "alternatives": tested})
    return out


def _failed(reason: str, **diagnostics: Any) -> dict[str, Any]:
    return {
        "rates": [],
        "score": None,
        "eligible": False,
        "reason": reason,
        "separation_rps": None,
        "separation_interval_rps": None,
        "drift_rps_per_s": [],
        "null_rate_rps": None,
        "diagnostics": diagnostics,
    }


def _prepare(audio_ct: np.ndarray, fs: float) -> np.ndarray:
    x = np.asarray(audio_ct)
    if x.ndim != 2 or x.shape[0] == 0 or not np.isrealobj(x):
        raise ValueError("audio_ct must be a nonempty real (channels, samples) array")
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError("fs must be the positive native sample rate")
    return x


def fit_pair(
    audio_ct: np.ndarray,
    fs: float,
    seed_rps: float | Sequence[float],
    *,
    other_rates: Sequence[float] = (),
    blades: int = 2,
) -> dict[str, Any]:
    """Return candidate rates and an untouched predictive score, not a decision.

    `seed_rps` is one or two starting shaft rates; `other_rates` are independently
    known nearby interfering shafts, not additional fit targets. Seeds and any
    data-dependent choice of other rates must come from the first 48% of the
    *analyzed* crop or independent data. Raw crop and t0 are reported. Callers
    accept only when `eligible` and the score exceeds a calibrated threshold;
    keep structurally rejected trials in the false-split denominator. The
    profile interval is a nominal noise-inflated support envelope, not a claim
    of coverage: correlated/cyclostationary residuals and the zero gap limit
    require calibration. `rates` may be present even when ineligible.
    """
    raw = _prepare(audio_ct, fs)
    if int(blades) != blades or not 1 <= blades <= 12:
        raise ValueError("blades must be an integer in [1, 12]")
    blades = int(blades)
    seed = np.sort(np.atleast_1d(np.asarray(seed_rps, dtype=float)))
    others = np.asarray(other_rates, dtype=float)
    if seed.ndim != 1 or seed.size not in (1, 2) or not np.all(np.isfinite(seed) & (seed > 0)):
        raise ValueError("seed_rps must contain one or two positive finite shaft rates")
    if others.ndim != 1 or not np.all(np.isfinite(others) & (others > 0)):
        raise ValueError("other_rates must contain only positive finite shaft rates")
    n_in = raw.shape[-1]
    n = min(n_in, int(_MAX_SPAN * fs))
    span = n / fs
    if span < _MIN_SPAN:
        return _failed("short_span", span_s=span)
    start = (n_in - n) // 2
    x = raw[:_MAX_CHANNELS, start : start + n].astype(np.float32, copy=False)
    if not np.all(np.isfinite(x)):
        return _failed("nonfinite_audio", span_s=span)
    center = float(seed.mean())
    gap = float(np.ptp(seed))
    if gap > _MAX_SPLIT:
        return _failed("seeds_too_far_apart", seed_rps=seed.tolist(), span_s=span)
    if center < 16.0 or center > 0.12 * fs:
        return _failed("unsupported_rate", seed_rps=seed.tolist(), span_s=span)
    radius = 2.8 if seed.size == 1 or gap == 0 else max(1.7, 0.5 * gap + 1.05)
    am_max = min(7.0, 0.095 * center)
    a, b = int(_TRAIN_FRACTION * n), int(_CAL_END * n)
    bands = _bands(center, radius, _DRIFT_CHANGE / 2, am_max, fs, others, a / fs)
    ti, vi, tc, vc = _partition(x.shape[0], bands)
    diag: dict[str, Any] = {
        "span_s": span,
        "sample_range": [start, start + n],
        "block_samples": [0, a, b, n],
        "fs": float(fs),
        "seed_rps": seed.tolist(),
        "other_rates": others.tolist(),
        "train_orders": [bands[i].order for i in ti],
        "heldout_orders": [bands[i].order for i in vi],
        "train_channels": tc.tolist(),
        "heldout_channels": vc.tolist(),
        "max_drift_rps_per_s": _DRIFT_CHANGE / span,
        "search_half_width_rps": radius,
        "am_max_hz": am_max,
    }
    if not ti.size or not vi.size:
        return _failed("no_clean_train_or_heldout_orders", **diag)
    t0 = -0.5 * (n - 1) / fs
    train = _block(x[tc, :a], fs, t0, center, bands)
    view = train.view(bands, span, ti, np.arange(tc.size))
    if float(np.max(np.abs(x[tc, :a]))) == 0.0:
        return _failed("no_training_energy", **diag)
    peaks = _candidates(view, radius)
    p, modes, best = _fit_two(view, peaks, seed - center, radius)
    p, interval, profile, cutoff = _profile(view, p, modes, radius, best)
    pair = _physical(p, radius)
    pair_loss = _pair(view, pair)[0]
    null_loss, nr, nd, nf, scaled = _fit_null(view, peaks, pair, radius, am_max, bands[-1].order)
    cal = _block(x[:, a:b], fs, t0 + a / fs, center, bands)
    test = _block(x[:, b:], fs, t0 + b / fs, center, bands)
    evidence = _predict(cal, test, bands, x.shape[0], span, pair, (nr, nd, nf, scaled))
    train_support = _support(view, pair)
    cal_support = _support(cal.view(bands, span, vi, np.arange(x.shape[0])), pair)
    support = (
        sorted(set(train_support[0]) | set(cal_support[0])),
        sorted(set(train_support[1]) | set(cal_support[1])),
    )
    aliases = _aliases(train, pair, center, span, support, others, blades)
    signed = p[1] + (pair[3] - pair[2]) * np.array([-0.5, 0.5])
    crossing = bool(signed[0] * signed[1] <= 0.0)
    minimum = 0.0 if crossing else float(np.min(np.abs(signed)))
    if evidence["score"] is None:
        reason = "no_heldout_energy"
    elif min(len(s) for s in train_support) < 1 or min(len(s) for s in cal_support) < 1:
        reason = "insufficient_cross_order_support"
    elif any(alt["ambiguous"] for mode in aliases for alt in mode["alternatives"]):
        reason = "shaft_alias_ambiguity"
    elif interval[0] <= 0.0:
        reason = "zero_in_separation_profile"
    elif (
        np.max(np.abs(pair[:2])) >= radius - 0.012
        or p[1] >= min(_MAX_SPLIT, 2.0 * radius) - 0.012
        or np.max(np.abs(pair[2:])) >= _DRIFT_CHANGE - 0.012
    ):
        reason = "search_boundary"
    else:
        reason = "ok"
    diag.update(
        {
            "order_support": [list(s) for s in support],
            "train_order_support": [list(s) for s in train_support],
            "calibration_heldout_order_support": [list(s) for s in cal_support],
            "alias_checks": aliases,
            "nominal_profile_delta_sse": cutoff,
            "baseband_samples": [train.t.size, cal.t.size, test.t.size],
            "train_pair_fraction_residual": pair_loss,
            "null_modulation": {
                "type": "trend" if nf == 0.0 else "per_order" if scaled else "hz",
                "frequency": nf,
            },
            "scaled_null_within_train_profile_tolerance": bool(
                scaled and null_loss <= pair_loss + cutoff / view.scale
            ),
            "heldout": evidence["heldout"],
            "crosses_in_span": crossing,
            "min_separation_in_span_rps": minimum,
            "signed_end_separations_rps": signed.tolist(),
            "prediction_model": {
                "center_rps": center,
                "bands": [[b.order, b.flat, b.stop] for b in bands],
                "channels": x.shape[0],
                "pair": pair.tolist(),
                "null": [nr, nd, nf, scaled],
            },
        }
    )
    rates = (center + pair[:2]).tolist() if p[1] > 0.0 else []
    return {
        "rates": rates,
        "score": evidence["score"],
        "eligible": reason == "ok",
        "reason": reason,
        "separation_rps": float(p[1]),
        "separation_interval_rps": interval,
        "separation_profile": profile,
        "drift_rps_per_s": (pair[2:] / span).tolist(),
        "null_rate_rps": center + nr,
        "null_drift_rps_per_s": nd / span,
        "null_am_hz": None if scaled else nf,
        "null_am_rps": nf if scaled else None,
        "diagnostics": diag,
    }


def predict_pair(
    audio_ct: np.ndarray, fs: float, fit: dict[str, Any], *, center_offset_s: float
) -> dict[str, Any]:
    """Score a disjoint local block using a frozen fit, calibrating gains only.

    `center_offset_s` is the new block midpoint minus the original analyzed
    crop midpoint, measured on the SAME native clock. No rates, slopes, bands,
    modulation frequency or channels are reselected. A positive score is
    predictive evidence, not a calibrated acceptance decision.
    """
    x = _prepare(audio_ct, fs)
    diag = fit.get("diagnostics", {})
    model = diag.get("prediction_model")
    if model is None:
        return {"score": None, "reason": "no_frozen_model"}
    if not np.isfinite(center_offset_s):
        raise ValueError("center_offset_s must be finite")
    n = x.shape[-1]
    if n / fs < 1.0 or n / fs > _MAX_SPAN:
        return {"score": None, "reason": "prediction_span_out_of_range"}
    channels = model["channels"]
    if fs != diag["fs"] or x.shape[0] < channels:
        return {"score": None, "reason": "prediction_clock_or_channels_changed"}
    x = x[:channels].astype(np.float32, copy=False)
    if not np.all(np.isfinite(x)):
        return {"score": None, "reason": "nonfinite_audio"}
    original = diag["span_s"]
    if abs(center_offset_s) + 1e-8 / fs < 0.5 * (original + n / fs):
        return {"score": None, "reason": "prediction_block_overlaps_fit"}
    center = model["center_rps"]
    pair = np.asarray(model["pair"])
    nr, nd, nf, scaled = model["null"]
    bands = [_Band(*band) for band in model["bands"]]
    t0 = float(center_offset_s) - 0.5 * (n - 1) / fs
    edges = np.array([t0, t0 + (n - 1) / fs])
    shifts = np.r_[
        np.max(np.abs(pair[:2, None] + pair[2:, None] * edges / original), axis=1),
        np.max(np.abs(nr + nd * edges / original)),
    ]
    if any(
        max(
            band.order * float(np.max(shifts[:2])),
            band.order * shifts[2] + nf * (band.order if scaled else 1),
        )
        > band.flat
        for band in bands
    ):
        return {"score": None, "reason": "prediction_outside_frozen_bands"}
    # The bounded trend is linear on the local fit; for disjoint forecast its
    # gain bound covers that new block too, using the same trend basis and
    # reference. Expressing it on a recentered t is just a phase/gain change;
    # the rate/slopes remain the same physical predictions.
    if nf == 0.0:
        pair = pair.copy()
        pair[:2] += pair[2:] * center_offset_s / original
        nr += nd * center_offset_s / original
        t0 -= center_offset_s
    a = n // 2
    cal = _block(x[:, :a], fs, t0, center, bands)
    test = _block(x[:, a:], fs, t0 + a / fs, center, bands)
    out = _predict(
        cal,
        test,
        bands,
        channels,
        original,
        pair,
        (nr, nd, nf, scaled),
        trend_extent=max(0.5, n / (2.0 * fs * original)),
    )
    return {**out, "reason": "ok" if out["score"] is not None else "no_heldout_energy"}
