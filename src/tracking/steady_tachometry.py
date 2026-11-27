"""Anonymous shaft rates from stationary and slowly drifting acoustic lines.

The AVQ static-rig method uses a long Hann observation, a per-band noise
floor, harmonic families competing for the same spectral evidence, and
shorter-window high-order readings to handle drift. This implementation uses
sparse peak votes rather than enumerating fixed-cardinality comb tuples. It
never requires a missing rotor to take any speed. It accepts synchronized
mics of one role, not mixtures of arbitrary clocks or of microphone roles.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks, peak_widths, resample_poly

from tracking.dsp import hann_line_power, log_peak_offset, running_median_db
from tracking.tach_corroboration import EVIDENCE_RECIPE, compare_families

RECIPE = "steady-tachometry/2"
ACCEPTANCE_RECIPE = "steady-tachometry-corroborate/2"
_MIN_S = 1.5
_LONG_S = 8.0
_NOMINAL_FRACTION = 0.15
_MIN_RATE = 15.0
_MAX_RATE = 600.0
_BASE_ORDER = 64
_MAX_ORDER = 128
_BAND_HZ = 7000.0
_HIGH_MOTOR_BAND_HZ = 16_000.0
_ORDERS = np.arange(1, _MAX_ORDER + 1)


@dataclass(frozen=True)
class _Physics:
    blades: int | None
    motor: tuple[int, ...]
    expected: int | None
    nominal: tuple[float, ...] = ()
    nominal_source: str | None = None

    @property
    def anchored(self) -> bool:
        return self.blades is not None or bool(self.motor)

    @property
    def orders(self) -> np.ndarray:
        cap = max(_BASE_ORDER, max(self.motor, default=0), min(_MAX_ORDER, 3 * (self.blades or 1)))
        return _ORDERS[:cap]

    @property
    def motor_spacing(self) -> int:
        # Two documented commutation lines can establish a common lattice:
        # k84 and k98, for example, are harmonics 6 and 7 of 14× shaft rate.
        return math.gcd(*self.motor) if self.blades is None and len(self.motor) > 1 else 1


@dataclass(frozen=True)
class _Peaks:
    duration: float
    f_max: float
    hz: np.ndarray
    prom: np.ndarray
    power: np.ndarray
    width: np.ndarray
    quiet: bool

    @property
    def tol(self) -> np.ndarray:
        # Do not treat zero-padding as an improvement in resolution. Broader
        # lines may seed a candidate, but cannot certify a narrow uncertainty.
        return np.maximum(0.65 / self.duration, np.minimum(0.46 * self.width, 0.001 * self.hz))


@dataclass(frozen=True)
class _Comb:
    rate: float
    uncertainty: float
    orders: np.ndarray
    peaks: np.ndarray
    compact: bool


def _positive(value) -> int | None:
    return int(value) if isinstance(value, (int, np.integer)) and int(value) > 0 else None


def _physics(facts: Mapping) -> _Physics:
    b = str(facts.get("blade_count", ""))
    blades = int(b) if b.isdecimal() and 0 < int(b) <= _MAX_ORDER else None
    if facts.get("propulsion") == "motor_only":
        blades = None
    motor = tuple(
        sorted(
            {
                k
                for k in (facts.get("shaft_orders") or ())
                if isinstance(k, int) and 1 <= k <= _MAX_ORDER
            }
        )
    )
    activity = facts.get("rotor_activity")
    if activity == "single":
        expected = 1
    else:
        expected = _positive(facts.get("n_active"))
        if expected is None and activity == "all":
            expected = _positive(facts.get("n_rotors"))
    nominal = (
        facts.get("nominal_rps")
        if facts.get("speed_regime") == "constant"
        and facts.get("speed_supplied") == "yes"
        and facts.get("supplied_kind") in ("nominal", "run_average", "command")
        else None
    )
    candidates = nominal if isinstance(nominal, (list, tuple)) else (nominal,)
    prior = tuple(
        float(s) for s in candidates if isinstance(s, (int, float)) and _MIN_RATE <= s <= _MAX_RATE
    )
    source = (
        ((facts.get("evidence") or {}).get("speed_supplied") or facts.get("supplied_kind"))
        if prior
        else None
    )
    return _Physics(blades, motor, expected, prior, source)


def _peaks(x: np.ndarray, fs: float, bandwidth: float = _BAND_HZ) -> _Peaks:
    t = x.shape[-1] / fs
    hi = min(bandwidth, 0.46 * fs)
    # A quiet channel need not be removed from the role. The pooled power has
    # no votes from zero channels and is invariant to microphone gain.
    if max(float(np.max(x)), -float(np.min(x))) <= 1e-7:
        empty = np.empty(0)
        return _Peaks(t, hi, empty, empty, empty, empty, True)
    power, df = hann_line_power(x, fs, hi)
    db = 10 * np.log10(np.maximum(power, 1e-30))
    floor = running_median_db(db[None], max(2, round(18 / df)))[0]
    prom = db - floor
    pk, _ = find_peaks(db, prominence=5.0, distance=max(1, round(0.9 / (t * df))))
    # Reject Hann sidelobes of much louder nearby lines before any comb vote.
    local_max = maximum_filter1d(db, 2 * round(7 / (t * df)) + 1)
    wide_max = maximum_filter1d(db, 2 * round(24 / (t * df)) + 1)
    pk = pk[
        (pk * df >= _MIN_RATE)
        & (prom[pk] >= 8.5)
        & (db[pk] >= local_max[pk] - 28)
        & (db[pk] >= wide_max[pk] - 48)
    ]
    # Bound candidate cost even in recordings with broad tonal interference;
    # keep coverage across the band, not the globally loudest fundamentals.
    groups = np.split(pk, np.flatnonzero(np.diff(pk * df // 500)) + 1)
    pk = np.concatenate(
        [np.sort(g[np.argsort(prom[g])[-80:]]) if len(g) > 80 else g for g in groups]
    )
    off = log_peak_offset(db[pk - 1], db[pk], db[pk + 1])
    width = peak_widths(power, pk, rel_height=0.5)[0] * df
    return _Peaks(t, hi, (pk + off) * df, prom[pk], power[pk], width, False)


def _seeds(p: _Peaks, phys: _Physics, prior: np.ndarray) -> np.ndarray:
    orders = phys.orders[phys.motor_spacing - 1 :: phys.motor_spacing]
    votes = p.hz[:, None] / orders
    good = (votes >= _MIN_RATE) & (votes <= _MAX_RATE)
    step = max(0.014, 0.09 / p.duration)
    bins = np.rint(votes[good] / step).astype(int)
    weights = np.minimum(3.0, 0.75 + (p.prom - 8.5) / 11)
    # High-order clutter alone produces many subharmonic votes. The low-order
    # comb carries the initial scale/profile decision; high-order teeth refine it.
    weight = weights[:, None] / np.maximum(1.0, np.sqrt(orders / 10))
    hist = np.bincount(bins, weights=np.broadcast_to(weight, good.shape)[good], minlength=2)
    peaks = find_peaks(hist, height=2.8, distance=2)[0]
    best = peaks[np.argsort(hist[peaks])[-160:]] * step
    anchors = tuple(sorted({phys.motor_spacing, phys.blades, *phys.motor} - {None}))
    strong = p.hz[np.argsort(p.prom)[-96:]]
    best = np.concatenate((best, (strong[:, None] / np.array(anchors)).ravel(), prior))
    keep = (best >= _MIN_RATE) & (best <= _MAX_RATE)
    best = np.unique(np.rint(best[keep] / (step / 4))) * (step / 4)
    return best


def _nearest(p: _Peaks, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    j = np.clip(np.searchsorted(p.hz, f), 1, len(p.hz) - 1)
    prev = np.abs(p.hz[j - 1] - f) < np.abs(p.hz[j] - f)
    j -= prev
    return j, np.abs(p.hz[j] - f)


def _measure(
    p: _Peaks, trial: np.ndarray, orders: np.ndarray, enabled: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rate and matching line evidence, growing from broad seeds to tight fits."""
    tol = p.tol
    for expansion in (1.75, 1.05, 1.0):
        f = trial[:, None] * orders
        j, d = _nearest(p, f)
        margin = tol[j] * expansion
        weight = np.clip(1 + (p.prom[j] - 10.5) / 12, 0.3, 3.0) * np.maximum(
            0, 1 - (d / margin) ** 2
        )
        weight *= (f <= p.f_max) & enabled
        if expansion == 1.0:
            break
        precision = weight * (np.minimum(orders, 30) / margin) ** 1.5
        total = precision.sum(axis=1)
        refit = (precision * (p.hz[j] / orders)).sum(axis=1)
        trial = np.divide(refit, total, out=trial.copy(), where=total > 0)
    return trial, j, weight


def _scales(
    p: _Peaks, phys: _Physics, j: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    present = w > 0.28
    matched_power = np.where(
        present[:, : max(12, 3 * (phys.blades or 1))],
        p.power[j[:, : max(12, 3 * (phys.blades or 1))]],
        0.0,
    )
    low_max = np.max(matched_power, axis=1)
    strength_prior = np.ones(w.shape[0])
    anchored = np.zeros(w.shape[0], dtype=bool)
    rich_bpf = np.zeros(w.shape[0], dtype=bool)
    if phys.blades is not None:
        b = phys.blades
        # Two physically different cases: the blade-pass comb may dominate
        # (AVQ), or the shaft line and its harmonics may dominate with a weak
        # BPF (AGH/DREGON). A lone strong line cannot anchor a BPF family:
        # require its overtone or separately visible shaft evidence.
        bpf = matched_power[:, b - 1]
        relative = bpf / np.maximum(1e-30, low_max)
        shaft = present[:, 0] & (p.prom[j[:, 0]] >= 10.5) & (matched_power[:, 0] >= 0.16 * low_max)
        if 3 * b <= w.shape[1]:
            blade_powers = matched_power[:, b - 1 :: b]
            overtones = np.count_nonzero(present[:, b - 1 :: b], axis=1)
            rich_bpf = (
                (relative >= 0.24)
                & (blade_powers[:, 1] >= 0.22 * bpf)
                & (blade_powers[:, 2] >= 0.07 * bpf)
            )
            first_three = present[:, 2 * b - 1] & present[:, 3 * b - 1]
            blade_supported = (relative >= 0.24) & (shaft | first_three | (overtones >= 4))
        else:
            blade_supported = (relative >= 0.24) & shaft
        # With a single source the dominant rotation harmonic can set the
        # scale (AGH). In a crowded mixture weak BPF is admissible only when
        # two independent odd-order lines support the same two-bladed shaft.
        odd_support = np.sum(present[:, 4:31:2] & (p.prom[j[:, 4:31:2]] >= 12.5), axis=1)
        promoted = shaft & (b == 2) & (relative >= 0.075) & (odd_support >= 2)
        shaft_dominant = shaft & (phys.expected == 1) & (p.prom[j[:, 0]] >= 15.0)
        shaft_dominant &= np.sum(present & (p.prom[j] >= 10.5), axis=1) >= 3
        anchored |= blade_supported | promoted | shaft_dominant
        strength_prior = np.where(
            ~blade_supported & shaft_dominant, 0.96, 0.48 + 0.52 * np.sqrt(relative)
        )
        strength_prior += np.where(promoted & (relative < 0.5), 0.10, 0.0)
    if phys.motor:
        idx = np.array(phys.motor) - 1
        n_motor = np.sum(present[:, idx] & (p.prom[j[:, idx]] >= 11.0), axis=1)
        verified = n_motor >= min(2, len(phys.motor))
        anchored |= verified
        strength_prior[verified] = 1.0 + 0.1 * n_motor[verified]
    if not phys.anchored:
        # Use primitive apparent periodicities for diagnostics, not as speeds.
        anchored = present[:, 0] & (np.sum(present[:, :10], axis=1) >= 3)
    return anchored, strength_prior, rich_bpf


def _intermod(rates: list[float]) -> np.ndarray:
    terms = []
    k = np.arange(1, 3)
    for i, s in enumerate(rates):
        a = s * k[:, None]
        for r in rates[:i]:
            b = r * k[None]
            terms.extend(((a + b).ravel(), abs(a - b).ravel()))
    return np.concatenate(terms) if terms else np.empty(0)


def _select(p: _Peaks, phys: _Physics, prior: np.ndarray) -> tuple[list[_Comb], bool]:
    if len(p.hz) < 3:
        return [], False
    orders_axis = phys.orders
    spacing = phys.motor_spacing
    trial, j, w = _measure(p, _seeds(p, phys, prior), orders_axis, orders_axis % spacing == 0)
    scale_ok, scale_prior, rich_bpf = _scales(p, phys, j, w)
    if phys.blades is not None and np.any(scale_ok):
        b = phys.blades
        # Close pairs can share one unresolved BPF line. Borrow its scale,
        # never its pitch evidence: both need independently split harmonics.
        anchors = np.sort(trial[scale_ok])
        x = np.searchsorted(anchors, trial)
        delta = np.minimum(
            abs(anchors[np.minimum(x, len(anchors) - 1)] - trial),
            abs(anchors[np.maximum(x - 1, 0)] - trial),
        )
        harmonics = w[:, 3 * b - 1 :: b] > 0.28
        borrowed = (~scale_ok) & (b * delta < 3 / p.duration) & (harmonics.sum(axis=1) >= 4)
        scale_ok |= borrowed
        scale_prior[borrowed] = 0.84
    keep = scale_ok & (np.sum(w > 0.28, axis=1) >= 3)
    if np.any(rich_bpf):
        # A rich lower-frequency BPF train already explains an exact
        # integer-multiple alternative. Its upper rival cannot count as an
        # independent rotor unless other evidence rules out the lower scale.
        rich_rates = np.sort(trial[rich_bpf])
        for m in (2, 3, 4):
            nearest = np.searchsorted(rich_rates, trial / m)
            distances = np.minimum(
                abs(rich_rates[np.minimum(nearest, len(rich_rates) - 1)] - trial / m),
                abs(rich_rates[np.maximum(nearest - 1, 0)] - trial / m),
            )
            keep &= distances > np.maximum(0.10, 0.0025 * trial / m)
    if spacing > 1:
        order_idx = np.array(phys.motor) - 1
        prim = spacing - 1
        keep &= (w[:, prim] > 0.28) & (p.prom[j[:, prim]] >= 13.0)
        # The two designated anchors must be resolved above their local
        # background, not just two arbitrary teeth in a dense motor spectrum.
        keep &= np.min(p.prom[j[:, order_idx]], axis=1) >= 13.0
        # Half-rate ghosts read intermediate commutation orders of a real
        # shaft. A primitive motor line and a high anchor at the doubled rate
        # expose such an ambiguity even if that larger family is incomplete.
        larger_idx = np.flatnonzero(keep)
        if len(larger_idx):
            doubled = _measure(p, 2 * trial[larger_idx], orders_axis, orders_axis % spacing == 0)
            _, jj, ww = doubled
            alongside = np.sum(
                (ww[:, order_idx] > 0.28) & (p.prom[jj[:, order_idx]] >= 12.0), axis=1
            )
            support = np.sum((ww > 0.28) & (p.prom[jj] >= 12.0), axis=1)
            higher = (alongside >= 1) & (support >= 4)
            keep[larger_idx[higher]] = False
    trial, j, w, scale_prior = trial[keep], j[keep], w[keep], scale_prior[keep]
    if len(trial) == 0:
        return [], False
    base = w * np.where(orders_axis <= max(12, 2 * (phys.blades or 1)), 1.2, 1.0)
    # Sparse analogue of the joint AVQ objective: commit only independent
    # orders and do not pay twice for a shared spectral line or an intermod.
    combs: list[_Comb] = []
    selected: list[float] = []
    blocked = np.zeros(len(p.hz), dtype=bool)
    distinct = np.ones(len(trial), dtype=bool)
    interference = False
    while len(combs) < (phys.expected or 12) + 1:
        unique = (~blocked[j]) & (w > 0.28)
        freq = p.hz[j]
        for s in selected:
            step = s * spacing
            other = np.rint(freq / step) * step
            unique &= np.abs(freq - other) >= np.maximum(1.85 / p.duration, 1.15 * p.width[j])
            distinct &= np.abs(trial - s) > 1.6 / (p.duration * len(orders_axis))
        im = _intermod(selected)
        if len(im):
            for f_im in im:
                unique &= np.abs(freq - f_im) > 1.15 / p.duration
        n = unique.sum(axis=1)
        low = np.sum(unique[:, : max(14, 3 * (phys.blades or 1))], axis=1)
        near = np.zeros(len(trial), dtype=bool)
        alias = np.zeros(len(trial), dtype=bool)
        for s in selected:
            near |= abs(trial - s) < max(0.30, 0.006 * s)
            ratio = np.maximum(trial / s, s / trial)
            alias |= (
                (np.rint(ratio) >= 2)
                & (np.rint(ratio) <= 4)
                & (abs(ratio - np.rint(ratio)) < 0.015)
            )
        # A second close rotor must have independently resolved high-order
        # teeth. Merely pulling a noisy or chirped line to each side is not a split.
        eligible = distinct & (n >= 3) & ((low >= 2) | (n >= 5) | bool(phys.motor))
        eligible &= ~near | ((n >= 3) & (np.sum(unique[:, 5:], axis=1) >= 3))
        eligible &= ~alias | (np.sum(unique[:, :8], axis=1) >= 3)
        eligible &= np.sum(unique & (p.prom[j] >= 10.5), axis=1) >= 3
        if spacing > 1:
            keys = np.array(phys.motor) - 1
            eligible &= np.sum(unique[:, keys], axis=1) >= min(2, len(keys))
            eligible &= (n >= 4) & (np.sum(unique[:, : 5 * spacing], axis=1) >= 2)
            eligible &= np.sum(unique & (p.prom[j] >= 13.0), axis=1) >= 4
        # Discount alias candidates supported mostly by already-explained
        # lines. Use support in independent low orders as well as total score.
        score = (
            np.sum(np.where(unique, base, 0.0), axis=1) + np.minimum(low, 4) * 0.7
        ) * scale_prior
        eligible &= (score >= 4.9) & (np.max(np.where(unique, p.prom[j], 0), axis=1) >= 12)
        idx = int(np.argmax(np.where(eligible, score, -np.inf)))
        if not eligible[idx]:
            break
        if phys.expected is not None and len(combs) == phys.expected:
            interference = True
            break
        chosen = unique[idx]
        peaks = j[idx, chosen]
        orders = orders_axis[chosen]
        levels = w[idx, chosen]
        rates = p.hz[peaks] / orders
        resolution = np.maximum(0.75 / p.duration, 0.50 * p.width[peaks]) / orders
        precision = levels / np.maximum(resolution, 0.003) ** 1.5
        rate = float(np.average(rates, weights=precision))
        # Lower bound dominated by independent, well-resolved high orders;
        # residual and broadened linewidth keep the reported precision honest.
        error = max(
            0.003,
            float(np.quantile(resolution, 0.35)),
            float(np.quantile(np.abs(rates - rate), 0.75)),
        )
        compact = bool(np.median(p.width[peaks] * p.duration) <= 1.9)
        combs.append(_Comb(rate, error, orders, peaks, compact))
        selected.append(rate)
        blocked[j[idx, w[idx] > 0.28]] = True
        distinct[idx] = False
        # Suppress almost coincident alternatives without erasing independently
        # resolved twins. Their winning lines have already been consumed.
        close = np.abs(trial - rate) < 1.5 * error
        if np.any(close):
            distinct[close & (n < 5)] = False
    combs.sort(key=lambda c: c.rate)
    return combs, interference


def _matched(a: list[_Comb], b: list[_Comb]) -> dict[int, _Comb]:
    if not a or not b:
        return {}
    delta = abs(np.array([c.rate for c in a])[:, None] - np.array([c.rate for c in b]))
    rows, cols = linear_sum_assignment(delta)
    return {int(i): b[j] for i, j in zip(rows, cols, strict=True)}


def _target_half(c: _Comb, others: list[_Comb], p: _Peaks) -> _Comb | None:
    """Refine a coarse family in an independent half using its own teeth.

    This is the AVQ block readout generalized to every usable isolated order:
    look locally around the coarse shaft line, combine independently visible
    high orders, and refuse a partial when only one order can pull the fit.
    """
    if len(p.hz) < 3:
        return None
    k = c.orders
    radius = max(0.12, 0.004 * c.rate * p.duration + 0.05)
    lo = np.searchsorted(p.hz, k * (c.rate - radius))
    hi = np.searchsorted(p.hz, k * (c.rate + radius))
    votes = np.concatenate([p.hz[a:b] / o for a, b, o in zip(lo, hi, k, strict=True)])
    if len(votes) < 3:
        return None
    trial = np.unique(np.concatenate(([c.rate], np.round(votes, 4))))
    f = trial[:, None] * k
    j, gap = _nearest(p, f)
    good = gap <= p.tol[j]
    for other in others:
        if other is c:
            continue
        hz = p.hz[j]
        crossing = abs(hz - np.rint(hz / other.rate) * other.rate)
        good &= crossing > max(1.6 / p.duration, abs(c.rate - other.rate) * 0.12)
    weight = np.where(good, np.minimum(3.0, 0.65 + (p.prom[j] - 8.5) / 12), 0.0)
    fit = np.sum(weight, axis=1) * (1.0 - 0.3 * abs(trial - c.rate) / radius)
    fit[good.sum(axis=1) < 3] = -np.inf
    ix = int(np.argmax(fit))
    if not np.isfinite(fit[ix]):
        return None
    orders = k[good[ix]]
    peaks = j[ix, good[ix]]
    sigma = np.maximum(0.75 / p.duration, 0.50 * p.width[peaks]) / orders
    value = p.hz[peaks] / orders
    rate = float(np.average(value, weights=weight[ix, good[ix]] / sigma**1.5))
    error = max(0.003, float(np.quantile(sigma, 0.35)), float(np.quantile(abs(value - rate), 0.75)))
    return _Comb(rate, error, orders, peaks, bool(np.median(p.width[peaks] * p.duration) <= 1.9))


def _confirm(
    full: list[_Comb], halves: tuple[list[_Comb], list[_Comb]], p: tuple[_Peaks, _Peaks], t: float
) -> tuple[list[tuple[_Comb, float]], bool, bool, float]:
    """Independent time/order checks; identify drift that warrants subdivision."""
    out = []
    moving = False
    unstable = False
    max_half_delta = 0.0
    left, right = [_matched(full, cs) for cs in halves]
    for i, c in enumerate(full):
        a, b = left.get(i), right.get(i)
        if a is None or abs(c.rate - a.rate) > max(0.13, 2 * c.uncertainty):
            a = _target_half(c, full, p[0]) or a
        if b is None or abs(c.rate - b.rate) > max(0.13, 2 * c.uncertainty):
            b = _target_half(c, full, p[1]) or b
        if a is None or b is None:
            continue
        drift = abs(a.rate - b.rate)
        midpoint = 0.5 * (a.rate + b.rate)
        agreement = max(0.065, min(0.18, 2 * max(a.uncertainty, b.uncertainty)))
        if max(abs(c.rate - a.rate), abs(c.rate - b.rate)) > max(2.0, 0.05 * c.rate):
            continue  # absence or unrelated family, not measured slew
        # A genuine, slow drift is a resolved time-dependent plateau, not a
        # constant speed. Reject fast changes and abrupt within-window jumps.
        if drift > agreement + max(0.25, 0.006 * c.rate) * t / 2 or abs(c.rate - midpoint) > max(
            agreement, 0.40 * drift
        ):
            unstable = True
            continue
        moving |= drift > agreement or not c.compact
        max_half_delta = max(max_half_delta, drift)
        uncertainty = max(
            c.uncertainty, 0.5 * (a.uncertainty + b.uncertainty), 2 * abs(c.rate - midpoint), drift
        )
        out.append((c, uncertainty))
    return out, moving, unstable, max_half_delta


def _merged_confirmation(c: _Comb, full: list[_Comb], halves: tuple[_Peaks, _Peaks]) -> bool:
    """Long-window twin exception: short FFTs may merge a genuinely static pair.

    The long result must have several compact, independently resolved teeth;
    both disjoint halves must still contain power in those same line cells.
    This exception is not available to a single wandering line.
    """
    if (
        not c.compact
        or len(c.orders) < 4
        or not any(d is not c and d.compact and abs(c.rate - d.rate) < 0.45 for d in full)
    ):
        return False
    for p in halves:
        if len(p.hz) < 3:
            return False
        _, d = _nearest(p, c.orders * c.rate)
        if int(np.sum(d <= 1.2 / p.duration)) < 3:
            return False
    return True


def _window(a: float, b: float, confirmed: list[tuple[_Comb, float]], quality: dict) -> dict:
    confirmed.sort(key=lambda item: item[0].rate)
    return {
        "start_s": a,
        "end_s": b,
        "center_s": (a + b) / 2,
        "accepted": bool(confirmed),
        "rates_rev_s": [c.rate for c, _ in confirmed],
        "uncertainty_rev_s": [sigma for _, sigma in confirmed],
        "order_support": [[int(k) for k in c.orders] for c, _ in confirmed],
        "quality": quality,
    }


def _rejected(a: float, b: float, reason: str) -> dict:
    return _window(a, b, [], {"reason": reason})


def _checked_modes(
    segment: np.ndarray,
    sr: float,
    phys: _Physics,
    p: _Peaks,
    halves: tuple[_Peaks, _Peaks],
    confirmed: list[tuple[_Comb, float]],
) -> list[dict]:
    """Freeze acoustic evidence and any independently observed alternatives."""
    confirmed.sort(key=lambda pair: pair[0].rate)
    rates = np.array([c.rate for c, _ in confirmed])
    bounds = np.array([e for _, e in confirmed])
    checks = compare_families(
        segment, sr, p.f_max, rates, bounds, phys.blades, phys.motor, phys.motor_spacing
    )
    for i, check in enumerate(checks):
        others = [c for j, (c, _) in enumerate(confirmed) if i != j]
        alternatives = []
        for mode in check.pop("modes"):
            if mode["ratio"] == 1:
                continue
            k = np.array(mode["orders"])
            frequency = k * mode["rate_rev_s"]
            j, d = _nearest(p, frequency)
            matched = d <= p.tol[j]
            if np.count_nonzero(matched) < 3:
                continue
            # The corroborator may recover a different scale, but before
            # looking at any labels it must also be fitted by the primary
            # peak method and observed on both sides of the window.
            k, j = k[matched], j[matched]
            semiwidth = np.maximum(0.75 / p.duration, 0.5 * p.width[j]) / k
            interp = p.hz[j] / k
            rate = float(np.average(interp, weights=1 / semiwidth**1.5))
            tol = max(float(np.median(semiwidth)), float(np.max(abs(interp - rate))))
            peak_comb = _Comb(rate, tol, k, j, bool(np.median(p.width[j] * p.duration) <= 1.9))
            a, b = [_target_half(peak_comb, others, h) for h in halves]
            if a is None or b is None:
                continue
            width = abs(a.rate - b.rate)
            if max(abs(rate - a.rate), abs(rate - b.rate)) > max(0.18, 2 * tol):
                continue
            alternatives.append(
                {
                    "rate_rev_s": rate,
                    "order_support": k.tolist(),
                    "uncertainty_rev_s": max(
                        tol, width, 2 * abs(rate - (a.rate + b.rate) / 2), 0.003
                    ),
                    "score": mode["score"],
                }
            )
        check["fitted_alternatives"] = alternatives
    return checks


class _Analysis:
    def __init__(self, x: np.ndarray, sr: float, absolute: float, phys: _Physics):
        self.x = x
        self.sr = sr
        self.absolute = absolute
        self.phys = phys
        self.windows: list[dict] = []
        self.quiet_s = 0.0
        self.transition_s = 0.0
        self.interference = False
        self.bandwidth = (
            _HIGH_MOTOR_BAND_HZ if max(phys.motor, default=0) > _BASE_ORDER else _BAND_HZ
        )

    def bands(self, i: int, j: int) -> _Peaks:
        return _peaks(self.x[:, i:j], self.sr, self.bandwidth)

    def read(self, i: int, j: int, p: _Peaks, cs: list[_Comb], crowded: bool) -> None:
        t = (j - i) / self.sr
        a, b = self.absolute + i / self.sr, self.absolute + j / self.sr
        if p.quiet:
            self.quiet_s += t
            self.windows.append(_rejected(a, b, "silent"))
            return
        if t < _MIN_S:
            self.windows.append(_rejected(a, b, "insufficient_contiguous_audio"))
            return
        m = (i + j) // 2
        pl, pr = self.bands(i, m), self.bands(m, j)
        proposals = np.array([c.rate for c in cs])
        left, cl = _select(pl, self.phys, proposals)
        right, cr = _select(pr, self.phys, proposals)
        confirmed, drift, unstable, half_delta = _confirm(cs, (left, right), (pl, pr), t)
        independent = len(confirmed)
        verified = {id(c) for c, _ in confirmed}
        misses = [c for c in cs if id(c) not in verified]
        twins = [c for c in misses if _merged_confirmation(c, cs, (pl, pr))]
        if len(twins) == len(misses) and twins and len(cs) <= (self.phys.expected or len(cs)):
            # A narrow, independently split long observation is preferable
            # to false short-window drift caused by swapping unresolved twins.
            confirmed.extend((c, c.uncertainty) for c in twins)
            if all(c.compact for c in cs):
                drift = unstable = False
        n = len(confirmed)
        persistent_extra = any(
            abs(first.rate - last.rate) < max(0.20, 2 * max(first.uncertainty, last.uncertainty))
            and all(abs(first.rate - c.rate) > 0.35 for c, _ in confirmed)
            for first in left
            for last in right
        )
        if t >= 2 * _MIN_S and (n == 0 or n < len(cs) or persistent_extra or drift or unstable):
            self.read(i, m, pl, left, cl)
            self.read(m, j, pr, right, cr)
            return
        self.interference |= crowded or (cl and cr)
        if unstable and n == 0:
            self.transition_s += t
        if n:
            checks = (
                _checked_modes(
                    self.x[:, i:j],
                    self.sr,
                    self.phys,
                    p,
                    (pl, pr),
                    confirmed,
                )
                if self.phys.anchored
                else []
            )
            self.windows.append(
                _window(
                    a,
                    b,
                    confirmed,
                    {
                        "drift_detected": bool(drift),
                        "half_window_delta_rev_s": half_delta,
                        "independent_half_support": independent == n,
                        "interference_possible": bool(crowded),
                        "analysis_band_hz": p.f_max,
                        "_band_checks": checks,
                        "acoustic_evidence_recipe": EVIDENCE_RECIPE,
                    },
                )
            )
        else:
            reason = (
                "nonstationary_or_inconsistent_halves"
                if unstable
                else "insufficient_independent_harmonic_support"
            )
            self.windows.append(_rejected(a, b, reason))


def _gap_regime(
    windows: list[dict],
    analyzed: float,
    quiet: float,
    transition: float,
    expected: int | None,
) -> tuple[str, str, float]:
    complete = [w for w in windows if expected is not None and len(w["rates_rev_s"]) == expected]
    coverage = sum(w["end_s"] - w["start_s"] for w in complete)
    if quiet >= 0.99 * analyzed > 0:
        return "off", "quiet_pcm", coverage
    if not complete:
        if transition >= 0.6 * analyzed > 0:
            return "variable", "half_window_rate_disagreement_through_most_of_audio", coverage
        return "unknown", "insufficient_full_rig_coverage", coverage
    if quiet + transition >= 0.15 * analyzed:
        return "mixed", "motor_on_plateaus_and_quiet_or_nonstationary_intervals", coverage
    for a, b in zip(complete, complete[1:]):
        x = np.array(a["rates_rev_s"])
        y = np.array(b["rates_rev_s"])
        if np.any(abs(x - y) > np.maximum(0.8, 0.027 * x)):
            return "mixed", "distinct_supported_plateau_rates", coverage
    if len(complete) > 1:
        # Comparing rate-sorted vectors is permutation invariant, not rotor
        # tracking. It detects substantial slow ramps missed by local jumps.
        q10, median, q90 = np.quantile(
            [w["rates_rev_s"] for w in complete], (0.1, 0.5, 0.9), axis=0
        )
        if np.any(q90 - q10 > np.maximum(1.25, 0.05 * median)):
            return "variable", "substantial_rate_range_across_supported_intervals", coverage
    if coverage >= 0.7 * analyzed:
        return "constant", "limited_rate_range_with_full_rig_coverage", coverage
    return "unknown", "insufficient_full_rig_coverage", coverage


def _result(
    windows: list[dict],
    phys: _Physics,
    *,
    analyzed: float = 0.0,
    quiet: float = 0.0,
    transition: float = 0.0,
    interference: bool = False,
    doppler: bool = False,
    separated: bool = False,
    status: str | None = None,
    reason: str | None = None,
) -> dict:
    good = [w for w in windows if w["accepted"]]
    accepted = sum(w["end_s"] - w["start_s"] for w in good)
    regime, regime_reason, complete = _gap_regime(good, analyzed, quiet, transition, phys.expected)
    if status is None:
        if not phys.anchored:
            status, reason = "unresolved", "unknown_shaft_order"
        elif doppler:
            status, reason = "unresolved", "uncorrected_doppler"
        elif not good:
            status = "unresolved"
            reason = "silent_windows" if regime == "off" else "no_repeatable_rotor_comb_or_plateau"
        elif phys.expected is None:
            status, reason = "partial", "audible_shafts_only_expected_count_unknown"
        elif phys.expected is not None and any(
            len(w["rates_rev_s"]) != phys.expected for w in good
        ):
            status, reason = "partial", "expected_rotors_unresolved_due_to_overlap_or_weak_support"
        elif accepted + quiet < analyzed - 0.01 or separated:
            status, reason = "partial", "shaft_rates_resolved_only_in_supported_intervals"
        else:
            status, reason = "resolved", "independent_harmonic_lines_with_half_window_support"
    return {
        "status": status,
        "reason": reason,
        "identity": "anonymous",
        "n_expected": phys.expected,
        "windows": windows,
        "shaft_order_anchored": phys.anchored,
        "possible_interference": interference,
        "possible_doppler": doppler,
        "uncertainty_kind": "acoustic_repeatability_tolerance",
        "nominal_scale_prior_rev_s": list(phys.nominal),
        "nominal_scale_prior_fraction": _NOMINAL_FRACTION if phys.nominal else None,
        "nominal_scale_prior_source": phys.nominal_source,
        "regime_evidence": {
            "suggested": regime,
            "reason": regime_reason,
            "analyzed_s": analyzed,
            "accepted_s": accepted,
            "full_rig_accepted_s": complete,
            "transition_s": transition,
            "quiet_s": quiet,
            "max_within_window_drift_rev_s": max(
                (w["quality"]["half_window_delta_rev_s"] for w in windows if w["rates_rev_s"]),
                default=0.0,
            ),
        },
    }


def _active_spans(x: np.ndarray, sr: float) -> list[tuple[int, int]]:
    """Exclude exact silence/dropouts without assuming low energy means off."""
    block = max(1, round(sr / 8))
    n = x.shape[-1]
    start = None
    spans = []
    for a in range(0, n, block):
        b = min(a + block, n)
        seg = x[:, a:b]
        loud = max(float(np.max(seg)), -float(np.min(seg))) > 1e-7
        if loud:
            if start is None:
                start = a
        elif start is not None:
            spans.append((start, a))
            start = None
    if start is not None:
        spans.append((start, n))
    return spans


def estimate(audio_ct, sr, facts, *, t_start=0.0, candidate_only=False) -> dict:
    """Estimate speeds of distinguishable lines on a role's actual time grid.

    ``facts`` supplies source-level blade count and/or documented shaft orders,
    optional active rotor count, and relative motion. Published nominal rates
    may disambiguate the scale but are never point estimates. Pass contiguous
    valid audio (not storage padding) with ``t_start`` on the Frame clock.
    Returned rates are unordered with
    respect to physical identities. Uncertainty is a conservative acoustic
    repeatability scale, not a calibrated statistical confidence interval.

    ``candidate_only`` returns private FFT proposals plus independent acoustic
    evidence. The publishable default applies the same acceptance gate as
    :func:`corroborate`, without native sensor values.
    """
    # Native labels/nominal values do not enter either acoustic fit.
    facts_phys = _physics(facts)
    phys = _Physics(facts_phys.blades, facts_phys.motor, facts_phys.expected)
    if (
        facts.get("rotor_activity") == "off"
        or facts.get("speed_regime") == "off"
        or facts.get("propulsion") == "none"
        or facts.get("sound_content") == "no_drone"
    ):
        return _result([], phys, status="not_applicable", reason="source_documents_no_active_rotor")
    x = np.asarray(audio_ct)
    if x.ndim == 1:
        x = x[None]
    if (
        x.ndim != 2
        or not x.shape[0]
        or not math.isfinite(sr)
        or sr <= 0
        or not math.isfinite(t_start)
    ):
        raise ValueError("expected (channel,time) audio and a finite sample rate/time origin")
    duration = x.shape[-1] / sr
    if duration < _MIN_S:
        return _result(
            [],
            phys,
            analyzed=duration,
            status="too_short",
            reason="less_than_1.5_s_of_actual_audio",
        )
    if sr < 2_000:
        return _result(
            [],
            phys,
            analyzed=duration,
            status="unresolved",
            reason="sample_rate_too_low_for_multiharmonic_reading",
        )
    # Integer anti-alias decimation preserves the known rate/coordinate without
    # a poorly conditioned rational-rate approximation. Copy one block at a time.
    min_rate_hz = 36_000 if max(phys.motor, default=0) > _BASE_ORDER else 16_000
    q = max(1, int(sr // min_rate_hz))
    sr_work = sr / q
    n_outer = max(1, math.ceil(duration / _LONG_S))
    windows: list[dict] = []
    quiet = transition = 0.0
    interference = facts.get("sound_content") in ("co_recorded", "digital_mix", "both")
    for k in range(n_outer):
        i, j = (k * x.shape[-1]) // n_outer, ((k + 1) * x.shape[-1]) // n_outer
        chunk = np.asarray(x[:, i:j], dtype=np.float32)
        if not np.all(np.isfinite(chunk)):
            windows.append(_rejected(t_start + i / sr, t_start + j / sr, "invalid_pcm"))
            continue  # caller should provide valid spans; don't fit across dropouts
        spans = _active_spans(chunk, sr)
        quiet += (j - i - sum(b - a for a, b in spans)) / sr
        prev = 0
        for a, b in spans:
            if prev < a:
                windows.append(
                    _rejected(t_start + (i + prev) / sr, t_start + (i + a) / sr, "silent")
                )
            prev = b
            if b - a < _MIN_S * sr:
                windows.append(
                    _rejected(
                        t_start + (i + a) / sr,
                        t_start + (i + b) / sr,
                        "insufficient_contiguous_audio",
                    )
                )
                continue
            seg = chunk[:, a:b]
            if q > 1:
                seg = resample_poly(seg, 1, q, axis=-1)
            part = _Analysis(seg, sr_work, t_start + (i + a) / sr, phys)
            p = part.bands(0, seg.shape[-1])
            combs, crowded = _select(p, phys, np.empty(0))
            part.read(0, seg.shape[-1], p, combs, crowded)
            final_end = t_start + (i + b) / sr
            if part.windows and part.windows[-1]["end_s"] > final_end:
                part.windows[-1]["end_s"] = final_end
                part.windows[-1]["center_s"] = (final_end + part.windows[-1]["start_s"]) / 2
            windows.extend(part.windows)
            quiet += min(part.quiet_s, (b - a) / sr)
            transition += min(part.transition_s, (b - a) / sr)
            interference |= part.interference
        if prev < j - i:
            windows.append(_rejected(t_start + (i + prev) / sr, t_start + j / sr, "silent"))
    windows = _prune_isolated_aliases(windows)
    doppler = facts.get("relative_motion") in ("moving", "mixed")
    result = _result(
        windows,
        phys,
        analyzed=duration,
        quiet=quiet,
        transition=transition,
        interference=interference,
        doppler=doppler,
    )
    if not phys.anchored or doppler:
        for w in windows:
            if not w["accepted"]:
                continue
            w["quality"][
                "apparent_periodicities_hz" if not phys.anchored else "apparent_rates_rev_s"
            ] = w["rates_rev_s"]
            w["quality"]["apparent_uncertainty"] = w["uncertainty_rev_s"]
            w["rates_rev_s"], w["uncertainty_rev_s"], w["order_support"] = [], [], []
            w["accepted"] = False
            w["quality"]["reason"] = (
                "unknown_shaft_order" if not phys.anchored else "uncorrected_doppler"
            )
        result["regime_evidence"].update(
            suggested="unknown",
            reason=result["reason"],
            accepted_s=0.0,
            full_rig_accepted_s=0.0,
            max_within_window_drift_rev_s=0.0,
        )
    elif (
        facts.get("relative_motion") != "fixed"
        or facts.get("relative_motion_basis") == "nominal_hover"
    ):
        for w in windows:
            w["quality"]["uncertainty_excludes_unknown_doppler"] = True
    result["publication_state"] = "private_candidates"
    return result if candidate_only else corroborate(result, facts)


def _prune_isolated_aliases(windows: list[dict]) -> list[dict]:
    """Keep genuine plateaus; reject isolated octave flips across a time series."""
    populated = [i for i, w in enumerate(windows) if w["rates_rev_s"]]
    if len(populated) < 3:
        return windows
    out = list(windows)
    for at, i in enumerate(populated):
        w = windows[i]
        nearby = [
            windows[j]
            for j in populated[max(0, at - 2) : at + 3]
            if j != i and abs(windows[j]["center_s"] - w["center_s"]) <= 12
        ]
        if len(nearby) < 2:
            continue
        keep = []
        for j, rate in enumerate(w["rates_rev_s"]):
            continuations, aliases = 0, 0
            for v in nearby:
                r = np.array(v["rates_rev_s"])
                continuations += int(np.any(abs(r - rate) <= max(0.18, 0.018 * rate)))
                q = np.maximum(rate / r, r / rate)
                aliases += int(
                    np.any((np.rint(q) >= 2) & (np.rint(q) <= 4) & (abs(q - np.rint(q)) < 0.024))
                )
            if continuations or aliases < 2:
                keep.append(j)
        if len(keep) == len(w["rates_rev_s"]):
            continue
        revised = dict(w)
        for key in ("rates_rev_s", "uncertainty_rev_s", "order_support"):
            revised[key] = [w[key][j] for j in keep]
        revised["accepted"] = bool(keep)
        revised["quality"] = {**w["quality"], "isolated_order_alias_rejected": True}
        if checks := w["quality"].get("_band_checks"):
            revised["quality"]["_band_checks"] = [checks[j] for j in keep]
        revised["quality"]["apparent_rates_rev_s"] = [
            s for j, s in enumerate(w["rates_rev_s"]) if j not in keep
        ]
        if not keep:
            revised["quality"]["reason"] = "order_alias_unresolved"
        out[i] = revised
    return out


def merge_estimates(parts: Iterable[dict]) -> dict:
    """Combine disjoint, same-role bounded views without averaging tracks."""
    rows = list(parts)
    if not rows:
        raise ValueError("expected at least one estimate")
    if len(rows) == 1:
        return rows[0]
    head = rows[0]
    windows = sorted((w for r in rows for w in r["windows"]), key=lambda w: w["start_s"])
    windows = _prune_isolated_aliases(windows)
    if any(a["end_s"] > b["start_s"] + 1e-8 for a, b in zip(windows, windows[1:])):
        raise ValueError("the tachometry views must not overlap")
    states = {r.get("publication_state") for r in rows if r["windows"]}
    if len(states) > 1:
        raise ValueError("cannot merge accepted measurements and private candidates")
    if any(
        r["n_expected"] != head["n_expected"]
        or r["shaft_order_anchored"] != head["shaft_order_anchored"]
        or r["nominal_scale_prior_rev_s"] != head["nominal_scale_prior_rev_s"]
        for r in rows
    ):
        raise ValueError("the tachometry views must share rotor and shaft-order facts")
    phys = _Physics(
        1 if head["shaft_order_anchored"] else None,
        (),
        head["n_expected"],
        tuple(head["nominal_scale_prior_rev_s"]),
        head["nominal_scale_prior_source"],
    )
    metrics = {
        key: sum(r["regime_evidence"][key] for r in rows)
        for key in ("analyzed_s", "quiet_s", "transition_s")
    }
    all_statuses = {r["status"] for r in rows}
    if not windows and len(all_statuses) == 1:
        status = head["status"]
        reason = (
            head["reason"]
            if all(r["reason"] == head["reason"] for r in rows)
            else "no_repeatable_rotor_comb_or_plateau"
        )
    else:
        status = reason = None
    out = _result(
        windows,
        phys,
        analyzed=metrics["analyzed_s"],
        quiet=metrics["quiet_s"],
        transition=metrics["transition_s"],
        interference=any(r["possible_interference"] for r in rows),
        doppler=any(r["possible_doppler"] for r in rows),
        separated=any(r["status"] == "partial" for r in rows),
        status=status,
        reason=reason,
    )
    if not phys.anchored or out["possible_doppler"]:
        out["regime_evidence"].update(suggested="unknown", reason=out["reason"])
    if states:
        out["publication_state"] = next(iter(states))
        if out["publication_state"] == "corroborated":
            counts: dict[str, int] = {}
            for w in windows:
                basis = w["quality"].get("acceptance_basis")
                if basis and w["accepted"]:
                    counts[basis] = counts.get(basis, 0) + len(w["rates_rev_s"])
            out["acceptance_basis_counts"] = counts
    return out


_MEASURED_BASES = frozenset({"native_shaft_feedback", "native_feedback_acoustically_aligned"})


def _reference_gaps(rates: np.ndarray, ref: Mapping) -> tuple[np.ndarray, np.ndarray] | None:
    if ref.get("basis") not in _MEASURED_BASES or float(ref.get("coverage", 0)) < 0.75:
        return None
    shaft = np.asarray(ref["rates_rev_s"], dtype=float)
    span = np.asarray(ref["ranges_rev_s"], dtype=float)
    if shaft.ndim != 1 or shaft.size == 0 or span.shape != (len(shaft), 2):
        return None
    if not np.all(np.isfinite(shaft)) or not np.all(np.isfinite(span)):
        return None
    # Rate clock and ESC conversion can differ at the percent scale (the
    # published MD2 raw values are uncorrected). The telemetry range widens
    # matching very slightly; it cannot accept transitions or octave gaps.
    budget = np.maximum(0.4, 0.012 * shaft) + np.minimum(0.25, 0.25 * np.ptp(span, axis=1))
    distance = abs(rates[:, None] - shaft[None])
    permitted = (distance <= budget[None]) & (shaft[None] >= _MIN_RATE)
    return distance, permitted


def _fit_options(w: Mapping, checks: list[dict]) -> list[dict]:
    candidates = []
    for i, (s, e, k) in enumerate(
        zip(w["rates_rev_s"], w["uncertainty_rev_s"], w["order_support"], strict=True)
    ):
        candidates.append(
            {
                "rate_rev_s": s,
                "uncertainty_rev_s": e,
                "order_support": k,
                "source_index": i,
                "primary": True,
            }
        )
        for alt in checks[i].get("fitted_alternatives", ()):
            candidates.append({**alt, "source_index": i, "primary": False})
    return candidates


def _sensor_options(candidates: list[dict], ref: Mapping) -> list[dict] | None:
    gaps = _reference_gaps(np.array([c["rate_rev_s"] for c in candidates]), ref)
    if gaps is None:
        return None
    distance, allowed = gaps
    # Match independent primary families, not a whole stack of aliases of
    # one resolved line to different ESC tracks. Hungarian includes one
    # dummy per family so an impossible association never steals a match.
    origin = np.array([c["source_index"] for c in candidates])
    n_proposals = int(np.max(origin)) + 1
    n_targets = distance.shape[1]
    cost = np.full((n_proposals, n_targets + n_proposals), 1e6)
    cost[:, n_targets:] = 1000
    option = np.full((n_proposals, n_targets), -1)
    for i, src in enumerate(origin):
        take = allowed[i] & (distance[i] < cost[src, :n_targets])
        cost[src, :n_targets][take] = distance[i, take]
        option[src, take] = i
    rows, cols = linear_sum_assignment(cost)
    return [
        candidates[option[r, c]]
        for r, c in zip(rows, cols, strict=True)
        if c < n_targets and option[r, c] >= 0
    ]


def _nominal_options(candidates: list[dict], nominal: tuple[float, ...]) -> list[dict]:
    good = [(c, min(abs(c["rate_rev_s"] / s - 1) for s in nominal)) for c in candidates]
    by_source: dict[int, dict] = {}
    for c, gap in sorted(good, key=lambda row: (not row[0]["primary"], row[1])):
        if gap <= _NOMINAL_FRACTION:
            by_source.setdefault(c["source_index"], c)
    return list(by_source.values())


def _same_scale(c: Mapping, check: Mapping, phys: _Physics) -> bool:
    mode = check.get("primary")
    if not mode:
        return False
    if abs(mode["rate_rev_s"] - c["rate_rev_s"]) > max(0.10, min(0.55, c["uncertainty_rev_s"])):
        return False
    orders = set(mode["orders"])
    shared = orders & set(c["order_support"])
    if len(shared) < 3:
        return False
    alternative = check.get("alternative")
    spacing = phys.motor_spacing
    if spacing > 1 and phys.expected != 1:
        # Dense periodic motor spectra admit accidental matches to 84/98-like
        # pairs. For several shafts, insist on the primitive carrier and its
        # first two overtones as well as the documented high-frequency lines.
        if not {spacing, 2 * spacing, 3 * spacing, *phys.motor} <= shared:
            return False
        return alternative is None or check["margin"] >= max(4.0, 0.18 * mode["score"])
    if alternative is None:
        return True
    margin = check["margin"]
    blades = phys.blades
    if blades is not None and alternative["ratio"] > 1:
        low = {k for k in shared if k <= max(8, 4 * blades)}
        if len(low) < 3:
            # Isolated even teeth at k=2 and 4 plus high matches can be a
            # shaft's fundamental and first overtone at half the true rate.
            return False
        if {blades, 2 * blades, 3 * blades} <= low:
            # If the two fits explain the same band, the larger order model
            # is not allowed to win by also scoring high frequencies outside
            # the smaller fit's cap. Require decisive blade-profile evidence
            # and no more than one orthogonal line for the upper rival.
            rr = alternative["ratio"]
            comparable = {k * rr for k in alternative["orders"] if k * rr <= max(orders)}
            if (
                mode["scale_score"] - alternative["scale_score"] >= 0.65
                and len(comparable - orders) <= 1
                and len(shared) >= 6
                and (margin >= 0 or (rr == 2 and margin >= -min(3.1, 0.10 * mode["score"])))
            ):
                return True
    return margin >= max(1.0, 0.07 * mode["score"])


def _close_twin(
    candidate: Mapping, check: Mapping, anchors: list[dict], blades: int, duration: float
) -> bool:
    """A certified neighbor can supply scale, but not any high-order teeth."""
    mode, rival = check.get("primary"), check.get("alternative")
    if not mode or not rival or rival["ratio"] != 2 or check["margin"] < 0:
        return False
    s, e = candidate["rate_rev_s"], candidate["uncertainty_rev_s"]
    if abs(mode["rate_rev_s"] - s) > max(0.06, min(0.15, e)):
        return False
    if mode["scale_score"] - rival["scale_score"] < 0.65:
        return False
    for anchor in anchors:
        delta = abs(anchor["rate_rev_s"] - s)
        if delta <= 3 * max(e, anchor["uncertainty_rev_s"]):
            continue
        if delta * blades > 2.4 / duration:
            continue
        # Both real spectra already masked shared cells. In addition demand
        # five mutually supported lines with actual spacing beyond one lobe.
        shared = set(mode["orders"]) & set(candidate["order_support"])
        if sum(delta * k >= 2.0 / duration for k in shared if k > 2 * blades) >= 5:
            return True
    return False


def _distinct_options(entries: list[dict]) -> list[dict]:
    """Do not duplicate lines when two private candidates map to one scale."""
    out: list[dict] = []
    for c in sorted(entries, key=lambda q: (not q["primary"], q["uncertainty_rev_s"])):
        if all(
            abs(c["rate_rev_s"] - other["rate_rev_s"])
            > max(0.006, min(c["uncertainty_rev_s"], other["uncertainty_rev_s"], 0.025))
            for other in out
        ):
            out.append(c)
    return sorted(out, key=lambda c: c["rate_rev_s"])


def corroborate(
    result: dict,
    facts: Mapping,
    *,
    references: Iterable[Mapping | None] | None = None,
) -> dict:
    """Apply the PCM-free publication gate to *private* fitted proposals.

    Optional references are independent native shaft-speed summaries in exact
    window order (same frame clock); they are a veto/ordinal selector only.
    A nominal is a broad scale check, not a trajectory. Without either, the
    independent band-power comb must agree at the proposed shaft scale.
    """
    if result.get("publication_state") == "corroborated":
        return result
    windows = result["windows"]
    refs = [None] * len(windows) if references is None else list(references)
    if len(windows) != len(refs):
        raise ValueError("reference summaries must match the estimate's window order")
    phys = _physics(facts)
    out_windows = []
    used: dict[str, int] = {}
    saw_candidate = False
    for w, ref in zip(windows, refs, strict=True):
        quality = dict(w.get("quality") or {})
        checks = quality.pop("_band_checks", [])
        row = {**w, "quality": quality}
        out_windows.append(row)
        if not w["rates_rev_s"]:
            continue
        saw_candidate = True
        basis = "independent_acoustic_comb"
        accepted: list[dict] = []
        if result.get("publication_state") == "private_candidates" and len(checks) == len(
            w["rates_rev_s"]
        ):
            fits = _fit_options(w, checks)
            sensor = _sensor_options(fits, ref) if ref is not None else None
            if sensor is not None:
                accepted = sensor
                basis = "sensor_corroborated"
                assert ref is not None
                quality["reference"] = {
                    k: ref[k] for k in ("source", "basis", "entry", "coverage") if k in ref
                }
            elif phys.nominal:
                accepted = _nominal_options(fits, phys.nominal)
                basis = "nominal_scale_checked"
            else:
                accepted = [
                    c
                    for c in fits
                    if c["primary"] and _same_scale(c, checks[c["source_index"]], phys)
                ]
                if phys.blades is not None and accepted:
                    anchors = list(accepted)
                    accepted.extend(
                        c
                        for c in fits
                        if c["primary"]
                        and c not in anchors
                        and _close_twin(
                            c,
                            checks[c["source_index"]],
                            anchors,
                            phys.blades,
                            w["end_s"] - w["start_s"],
                        )
                    )
        accepted = _distinct_options(accepted)
        surviving = {c["source_index"] for c in accepted if c["primary"]}
        rejected = [s for i, s in enumerate(w["rates_rev_s"]) if i not in surviving]
        if rejected:
            quality["apparent_rates_rev_s"] = rejected
        if not accepted:
            quality["reason"] = {
                "sensor_corroborated": "native_shaft_reference_disagrees",
                "nominal_scale_checked": "published_nominal_scale_disagrees",
                "independent_acoustic_comb": "independent_acoustic_order_unresolved",
            }[basis]
        else:
            quality["acceptance_basis"] = basis
            used[basis] = used.get(basis, 0) + len(accepted)
        if checks:
            quality.setdefault("acoustic_evidence_recipe", "avq-band-corroboration/1")
            quality["acoustic_crosscheck"] = [
                {k: c[k] for k in ("primary", "alternative", "margin")} for c in checks
            ]
        row["accepted"] = bool(accepted)
        for key in ("rates_rev_s", "uncertainty_rev_s", "order_support"):
            field = "rate_rev_s" if key == "rates_rev_s" else key
            row[key] = [c[field] for c in accepted]
    metrics = result["regime_evidence"]
    special = not windows or (not saw_candidate and not result.get("shaft_order_anchored", True))
    out = _result(
        out_windows,
        phys,
        analyzed=metrics["analyzed_s"],
        quiet=metrics["quiet_s"],
        transition=metrics["transition_s"],
        interference=result["possible_interference"],
        doppler=result["possible_doppler"],
        status=result["status"] if special else None,
        reason=result["reason"] if special else None,
    )
    if saw_candidate and not used:
        out["reason"] = "shaft_order_not_corroborated"
    out["acceptance_basis_counts"] = used
    out["acceptance_recipe"] = ACCEPTANCE_RECIPE
    out["publication_state"] = "corroborated"
    return out
