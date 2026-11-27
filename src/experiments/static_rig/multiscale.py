"""Full-record-first, anonymous acoustic shaft tracks on one native source clock.

The historical AVQ search used multichannel line evidence, dyadic time support,
non-overlapping comb evidence and a path cost. Here these are sparse peak
fits (not dense comb grids), with missing states rather than a fixed-cardinality
comb. Broad windows propose scale and distinguish closely spaced steady combs;
short windows must supply independent teeth before inheriting an estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
from scipy import fft, ndimage, signal, sparse
from scipy.signal.windows import hann
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

_ANALYSIS_FS = 12_000
_MAX_SAMPLES = 2**21
_MAX_RATE = 600.0
_MIN_RATE = 15.0
_FREQ_MAX = 5_300.0
_ORDERS = np.arange(1, 49)
_VOTES = _ORDERS[:24]
_MAX_LINK_S = 5.0
_MAX_DRIFT = 1.3  # rev/s per second; faster or ambiguous motion breaks identity


@dataclass(frozen=True)
class _Lines:
    duration: float
    limit: float
    hz: np.ndarray
    db: np.ndarray
    widths: np.ndarray
    power_db: np.ndarray | None = None

    @property
    def tolerance(self) -> np.ndarray:
        return np.maximum(0.74 / self.duration, np.minimum(0.40 * self.widths, 0.002 * self.hz))


@dataclass(frozen=True)
class _Point:
    rate: float
    error: float
    score: float
    orders: tuple[int, ...]
    cells: tuple[int, ...]
    provenance: str = "measured"


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(mask, prepend=False, append=False))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def _activity(x: np.ndarray, fs: float) -> tuple[list[tuple[int, int]], np.ndarray, int]:
    """Find explicit dropouts/quiet troughs, not nominal per-shaft activity."""
    n = x.shape[1]
    block = max(1, round(fs / 4))
    energy = np.empty(math.ceil(n / block))
    for i, start in enumerate(range(0, n, block)):
        part = x[:, start : start + block]
        energy[i] = np.einsum("ct,ct->", part, part, dtype=np.float64) / part.size
    loud = energy > 1e-14
    if np.any(loud):
        quiet = energy < np.percentile(energy[loud], 85) * 10 ** (-26 / 10)
        for a, b in _runs(quiet):
            if b - a >= 3:
                loud[a:b] = False
    spans = [(a * block, min(b * block, n)) for a, b in _runs(loud)]
    return spans, energy, block


def _missing_pcm(x: np.ndarray) -> list[tuple[int, int]]:
    if not np.issubdtype(x.dtype, np.inexact):
        return []
    missing = None
    for row in x:
        bad = ~np.isfinite(row)
        if bad.any():
            if missing is None:
                missing = bad
            else:
                missing |= bad
    return _runs(missing) if missing is not None else []


def _resample(
    x: np.ndarray, fs: float, duration: float, missing: list[tuple[int, int]]
) -> tuple[np.ndarray, int]:
    # Integer decimation handles arbitrary native sample rates without enormous
    # rational polyphase kernels. The original timeline is never redefined.
    target = min(_ANALYSIS_FS, max(2_400, (_MAX_SAMPLES - 1) / duration))
    decimation = max(1, math.ceil(fs / target))
    if decimation == 1:
        out = x.astype(np.float32, copy=bool(missing))
        for a, b in missing:
            out[:, a:b] = 0
        return out, decimation
    out = np.empty((len(x), math.ceil(x.shape[1] / decimation)), dtype=np.float32)
    for c, row in enumerate(x):
        row = row.astype(np.float32, copy=bool(missing))
        for a, b in missing:
            row[a:b] = 0
        out[c] = signal.resample_poly(row, 1, decimation)
    return out, decimation


def _lines(x: np.ndarray, fs: float) -> _Lines:
    """Multichannel, noncoherent whitened Hann spectrum, one FFT in memory."""
    n = x.shape[1]
    duration = n / fs
    limit = min(_FREQ_MAX, 0.44 * fs)
    if n < 32 or duration < 0.40:
        return _Lines(duration, limit, *(np.empty(0) for _ in range(3)))
    nfft = cast(int, fft.next_fast_len(min(2 * n, 2 * _MAX_SAMPLES), real=True))
    df = fs / nfft
    nb = min(nfft // 2 + 1, int(limit / df) + 2)
    # Estimating a spectral floor in small physical-frequency tiles is O(F),
    # unlike a sliding median whose window grows with recording length.
    tile = max(8, round(19 / df))
    starts = np.arange(0, nb, tile)
    centers = np.minimum(starts + tile / 2, (starts + nb) / 2)
    bins = np.arange(nb, dtype=np.float32)
    win = hann(n, sym=False).astype(np.float32)
    mean = np.zeros(nb, dtype=np.float32)
    best = np.zeros(nb, dtype=np.float32)
    power = np.zeros(nb, dtype=np.float32)
    nch = 0
    for row in x:
        if np.max(np.abs(row)) <= 1e-7:
            continue
        z = cast(np.ndarray, fft.rfft(win * row, nfft, workers=1))
        p = np.abs(z[:nb])
        del z
        np.square(p, out=p)
        ref = max(float(np.max(p[min(nb - 1, int(_MIN_RATE / df)) :])), 1e-30)
        power += p / ref
        m = nb // tile
        floors = np.empty(len(starts), dtype=np.float32)
        floors[:m] = np.median(p[: m * tile].reshape(m, tile), axis=1)
        if m < len(starts):
            floors[m] = np.median(p[m * tile :])
        np.maximum(floors, max(float(np.max(p)) * 1e-12, 1e-30), out=floors)
        p /= np.interp(bins, centers, floors).astype(np.float32)
        mean += p
        np.maximum(best, p, out=best)
        nch += 1
    if not nch:
        return _Lines(duration, limit, *(np.empty(0) for _ in range(3)))
    # Reward evidence reproduced across microphones while retaining a line
    # present on only a favourable mic. Never assume inter-mic phase coherence.
    best *= 0.5
    best += mean * (0.5 / nch)
    db = 10 * np.log10(np.maximum(best, 1e-30))
    pk = signal.find_peaks(
        db, height=8.3, prominence=5, distance=max(1, round(0.85 / (duration * df)))
    )[0]
    guard = ndimage.maximum_filter1d(db, 2 * round(7 / (duration * df)) + 1)
    pk = pk[(pk * df >= _MIN_RATE) & (pk * df < limit) & (db[pk] >= guard[pk] - 29)]
    if len(pk) > 64:
        groups = np.split(pk, np.flatnonzero(np.diff(np.floor(pk * df / 225))) + 1)
        pk = np.concatenate(
            [np.sort(g[np.argpartition(db[g], -24)[-24:]]) if len(g) > 24 else g for g in groups]
        )
    left, mid, right = db[pk - 1], db[pk], db[pk + 1]
    offset = np.clip(0.5 * (left - right) / np.minimum(left - 2 * mid + right, -1e-6), -0.5, 0.5)
    width = signal.peak_widths(best, pk, rel_height=0.5)[0] * df
    return _Lines(
        duration,
        limit,
        (pk + offset) * df,
        mid,
        width,
        10 * np.log10(np.maximum(power[pk] / nch, 1e-30)),
    )


def _seeds(p: _Lines, blades: int, hints: list[float]) -> np.ndarray:
    votes = p.hz[:, None] / _VOTES
    keep = (votes >= _MIN_RATE) & (votes <= _MAX_RATE)
    step = max(0.012, 0.11 / p.duration)
    bins = np.rint(votes[keep] / step).astype(np.intp)
    strength = np.clip((p.db - 7.0) / 6, 0.25, 4.0)
    weights = strength[:, None] / np.maximum(1.0, np.sqrt(_VOTES / 8))
    hist = np.bincount(bins, np.broadcast_to(weights, votes.shape)[keep], minlength=3)
    hist = ndimage.convolve1d(hist, [0.45, 1, 0.45], mode="constant")
    peaks = signal.find_peaks(hist, height=2.3, distance=2)[0]
    peaks = peaks[np.argsort(hist[peaks])[-64:]] * step
    strong = p.hz[np.argsort(p.db)[-18:]]
    direct = (strong[:, None] / np.unique([1, blades, 2 * blades])).ravel()
    # Explicit scale alternatives are searched even when their low-order
    # line did not land in the most popular vote bin.
    extra = np.asarray(hints)
    seed = np.concatenate((peaks, direct, extra))
    return seed[(seed >= _MIN_RATE) & (seed <= _MAX_RATE)]


def _fit(p: _Lines, trials: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trials = trials.copy()
    tol = p.tolerance
    levels = np.clip((p.db - 7.0) / 6, 0.25, 4.0)
    for widen in (1.6, 1.0, 0.92):
        freq = trials[:, None] * _ORDERS
        j = np.clip(np.searchsorted(p.hz, freq), 1, len(p.hz) - 1)
        j -= np.abs(p.hz[j - 1] - freq) < np.abs(p.hz[j] - freq)
        err = np.abs(p.hz[j] - freq)
        margin = tol[j] * widen
        w = levels[j] * np.maximum(0.0, 1 - (err / margin) ** 2)
        w *= freq <= p.limit
        if widen == 0.92:
            return trials, j, w
        precision = w * (np.minimum(_ORDERS, 24) / margin) ** 1.4
        mass = precision.sum(axis=1)
        trials = np.divide(
            (precision * (p.hz[j] / _ORDERS)).sum(axis=1), mass, out=trials, where=mass > 0
        )
    raise AssertionError("unreachable")


def _point(
    p: _Lines, cells: np.ndarray, orders: np.ndarray, score: float, provenance="measured"
) -> _Point:
    rates = p.hz[cells] / orders
    res = np.maximum(0.72 / p.duration, 0.5 * p.widths[cells]) / orders
    weight = np.clip(p.db[cells] - 7, 1, 20) / np.maximum(res, 0.005) ** 1.3
    rate = float(np.average(rates, weights=weight))
    # A linewidth/residual envelope, not a calibrated statistical interval.
    error = max(0.004, float(np.median(res)), float(np.max(np.abs(rates - rate))))
    return _Point(
        rate, error, float(score), tuple(orders.tolist()), tuple(cells.tolist()), provenance
    )


def _simple_ratio(ratio):
    close = np.zeros_like(ratio, dtype=bool)
    for denominator in (1, 2, 3, 4):
        product = ratio * denominator
        numerator = np.rint(product)
        close |= (
            (numerator > denominator)
            & (numerator <= 12)
            & (np.abs(product - numerator) <= 0.003 * product)
        )
    return close


def _harmonic_mask(hz, width, rate):
    order = np.rint(hz / rate)
    return (order > 0) & (np.abs(hz - order * rate) < width)


def _owned_cells(p: _Lines, point: _Point) -> np.ndarray:
    """Peaks already explained by one carrier, including coherent AM offsets."""
    width = np.maximum(1.18 / p.duration, 0.95 * p.widths)
    owned = _harmonic_mask(p.hz, width, point.rate)
    owned[list(point.cells)] = True
    order = np.rint(p.hz / point.rate).astype(int)
    offsets = p.hz - order * point.rate
    amplitude = p.power_db if p.power_db is not None else p.db
    # A fixed modulation offset is frequency-additive at different shaft
    # orders. A second close shaft instead produces an offset proportional
    # to order, normally on one side of the first carrier throughout.
    eligible = np.flatnonzero(
        ~owned & (order > 0) & (np.abs(offsets) <= min(12.0, 0.18 * point.rate))
    )
    if len(eligible) < 4:
        return owned
    carrier_db = np.full(int(order.max()) + 1, -np.inf)
    np.maximum.at(carrier_db, order[owned], amplitude[owned])
    idx = eligible[amplitude[eligible] <= carrier_db[order[eligible]] - 3.0]
    if len(idx) < 4:
        return owned
    magnitude = np.abs(offsets[idx])
    idx = idx[np.argsort(magnitude)]
    magnitude = np.abs(offsets[idx])
    tolerance = 0.65 / p.duration
    starts = np.searchsorted(magnitude, magnitude - tolerance)
    stops = np.searchsorted(magnitude, magnitude + tolerance, side="right")
    for first, last in zip(starts, stops, strict=True):
        block = idx[first:last]
        if len(block) < 4:
            continue
        upper = order[block[offsets[block] > 0]]
        lower = order[block[offsets[block] < 0]]
        if np.unique(order[block]).size >= 3 and np.intersect1d(upper, lower).size:
            owned[block] = True
    return owned


def _joint(p: _Lines, slots: int, blades: int, hints: list[float], historical=()) -> list[_Point]:
    if len(p.hz) < 3:
        return []
    seed = _seeds(p, blades, hints + list(historical))
    if not len(seed):
        return []
    if blades > 1:
        # Both interpretations must be tested, including the one that did
        # not win the sparse vote. Most octave failures are not missing ridges.
        aliases = np.concatenate((seed / blades, seed * blades))
        seed = np.unique(np.r_[seed, aliases[(aliases >= _MIN_RATE) & (aliases <= _MAX_RATE)]])
    trial, j, w = _fit(p, seed)
    level = p.db[j]
    support = w > 0.35
    # Prevent random narrow-band maxima (or speech formants) from acting as
    # a full rotor. Rich-but-weak evidence can substitute for a bright tooth.
    usable = support & (level >= 9.4)
    early = _ORDERS <= 15
    blade = _ORDERS % blades == 0
    bpf = blades - 1
    physical = p.power_db[j] if p.power_db is not None else level
    strongest = np.max(np.where(usable[:, early], physical[:, early], -300.0), axis=1)
    bpf_rel = physical[:, bpf] - strongest
    low_level = np.max(np.where(usable[:, early], level[:, early], 0.0), axis=1)
    blade_ok = (
        usable[:, bpf]
        & (bpf_rel >= 10.0 * math.log10(0.30))
        & ((usable[:, blade].sum(axis=1) >= 3) | usable[:, 0] | (usable[:, early].sum(axis=1) >= 4))
    )
    shaft_ok = (
        usable[:, 0]
        & (level[:, 0] >= np.maximum(16.0, low_level - 4.0))
        & ((usable & ~blade).sum(axis=1) >= 2)
    )
    valid = (blade_ok | shaft_ok) & (usable.sum(axis=1) >= 3)
    compact = p.widths[j] <= np.maximum(3.3 / p.duration, 0.0025 * p.hz[j])
    valid &= np.sum(usable & compact, axis=1) >= 2
    # Prefer the scale locating its strong lines at the early blade orders,
    # unless the data actually support a dominant, primitive shaft comb.
    bpf_profile = blade_ok * (0.7 + 0.9 * np.clip((bpf_rel + 5.23) / 5.23, 0, 1))
    bpf_profile += (
        1.1
        * blade_ok
        * usable[:, 2 * blades - 1]
        * (physical[:, 2 * blades - 1] >= physical[:, bpf] - 9)
    )
    shaft_profile = shaft_ok * np.where(~blade_ok, 0.5, 0.18)
    # Bounded analogue of the historical g-prior on shaft order: dozens
    # of accidental high teeth must not outweigh a clean set of early ones.
    # The separate scale contest still compares the same physical peaks.
    g = 100.0 * (np.where(blade, blades, 1) / _ORDERS) ** 2
    base = w * (0.60 + 0.40 * g / (1.0 + g)) * np.where(blade, 1.12, 1.0)
    if historical:
        d = np.min(np.abs(trial[:, None] - np.asarray(historical)), axis=1)
        base *= (1 + 0.045 * (d < 0.10))[:, None]
    options = np.flatnonzero(valid)
    if not len(options):
        return []
    trial, j, base = trial[options], j[options], base[options]
    usable, level, physical, compact = (
        usable[options],
        level[options],
        physical[options],
        compact[options],
    )
    profile = (bpf_profile + shaft_profile)[options]
    blade_scale = blade_ok[options] & usable[:, 2 * blades - 1] & usable[:, 3 * blades - 1]
    blade_scale &= physical[:, 2 * blades - 1] >= physical[:, bpf] - 15.0
    blade_primary, shaft_primary = blade_ok[options], shaft_ok[options]
    blade_centres = blades * np.arange(1, 4) - 1
    left, right = np.maximum(0, blade_centres - 1), blade_centres + 1
    flank = 0.5 * (physical[:, left] + physical[:, right])
    lifted = usable[:, blade_centres] & usable[:, left] & usable[:, right]
    lifted &= physical[:, blade_centres] > flank + 2.0
    blade_contrast = np.count_nonzero(lifted, axis=1) >= 2
    if hints:
        dist = np.abs(trial[:, None] - np.asarray(hints))
        # Coarser positions only break weak score ties; no window provides
        # evidence by dictating an expected rotor count or rate band.
        profile += 0.8 * np.any(dist <= np.maximum(0.14, 0.002 * trial[:, None]), axis=1)
    own = np.ones(trial.shape, dtype=bool)
    freq = p.hz[j]
    collision_width = np.maximum(1.18 / p.duration, 0.95 * p.widths[j])
    points: list[_Point] = []
    for _ in range(slots):
        n = usable.sum(axis=1)
        low = usable[:, early].sum(axis=1)
        best_db = np.max(np.where(usable, level, 0), axis=1)
        bright = np.sum(usable & (level >= 10.5), axis=1)
        enough = (
            own
            & (n >= 3)
            & ((low >= 2) | (n >= 5))
            & (((best_db >= 14.8) & (bright >= 3)) | ((bright >= 5) & (best_db >= 12.5)))
            & (np.sum(usable & compact, axis=1) >= 2)
        )
        for point in points:
            delta = np.abs(trial - point.rate)
            ratio = np.maximum(trial / point.rate, point.rate / trial)
            alias = _simple_ratio(ratio)
            enough &= ~alias | (usable[:, : 4 * blades].sum(axis=1) >= 3)
            nearby = delta < max(0.35, 0.005 * point.rate)
            enough &= ~nearby | (usable[:, 2 * blades :].sum(axis=1) >= 3)
            enough &= delta > max(0.022, 0.8 * point.error)
        if not np.any(enough):
            break
        # Joint marginal evidence: a cell claimed by an existing comb cannot
        # pay twice. Discount mere changes of scale, not shared information.
        score = (base * usable).sum(axis=1) + profile
        # Compare aliases on their *overlapping observed peaks*, not raw
        # harmonic counts (which favour a small spacing in dense spectra).
        # The historical blade-power rule distinguishes a genuine blade train
        # from the visible shaft fundamental of a rotor at twice that rate.
        reject = np.zeros(len(trial), dtype=bool)
        tested = np.flatnonzero(enough)
        for lo in tested:
            ratio = trial[tested] / trial[lo]
            upper = tested[(ratio > 1.2) & _simple_ratio(ratio)]
            if not len(upper):
                continue
            lower_at = freq[lo, usable[lo]]
            lower_width = collision_width[lo, usable[lo]]
            lower_order = _ORDERS[usable[lo]]
            lower_cells = j[lo, usable[lo]]
            for hi in upper:
                high_owned = _harmonic_mask(freq[hi], collision_width[hi], trial[lo])
                shared = usable[hi] & (high_owned | np.isin(j[hi], lower_cells))
                if np.count_nonzero(shared) < 3:
                    continue
                lower_unique = ~_harmonic_mask(lower_at, lower_width, trial[hi])
                high_unique = usable[hi] & ~shared
                low_order_limit = 4 * blades
                lo_private = np.sum(lower_unique & (lower_order <= low_order_limit))
                hi_private = np.sum(high_unique & (low_order_limit >= _ORDERS))
                # A low rational alternative cannot divide a fundamental and
                # its odd orders between two motors with no exclusive upper
                # evidence. This applies equally to 3:2, 4:3, and octaves.
                if lo_private >= 3 and hi_private < 2 and score[lo] >= 0.70 * score[hi]:
                    reject[hi] = True
                    continue
                if hi_private >= 3 and lo_private < 2 and score[hi] >= 0.70 * score[lo]:
                    reject[lo] = True
                    continue
                if abs(trial[hi] / trial[lo] - blades) > 0.003 * blades or not usable[lo, bpf]:
                    continue
                bpf_advantage = physical[hi, bpf] - physical[lo, bpf]
                primitive_teeth = shared & ~blade
                gapped_primitive = (
                    shaft_primary[hi]
                    and not blade_scale[lo]
                    and np.count_nonzero(primitive_teeth) >= 2
                )
                stronger_bpf = (
                    blade_primary[hi]
                    and usable[hi, bpf]
                    and (bpf_advantage > 5.23 or (bpf_advantage >= -4.0 and blade_contrast[hi]))
                )
                if gapped_primitive or stronger_bpf:
                    reject[lo] = True
                    continue
                unique_high = usable[hi] & ~shared & (level[hi] >= 10.5)
                if (
                    blade_scale[lo]
                    and np.count_nonzero(unique_high) < 2
                    and score[lo] >= 0.70 * score[hi]
                ):
                    reject[hi] = True
        enough &= ~reject
        if not np.any(enough):
            break
        i = int(np.argmax(np.where(enough, score, -np.inf)))
        mask = usable[i]
        point = _point(p, j[i, mask], _ORDERS[mask], score[i])
        points.append(point)
        own[i] = False
        # The fitted-order limit caps the search, not the claims of an
        # accepted comb: its unscored but overlapping teeth still count once.
        usable &= ~_owned_cells(p, point)[j]
        # Nonlinear low-order sum/difference products of accepted families
        # must not be counted as evidence for a new shaft.
        for other in points[:-1]:
            a = point.rate * np.arange(1, 3)[:, None]
            b = other.rate * np.arange(1, 3)
            for f in np.concatenate(((a + b).ravel(), np.abs(a - b).ravel())):
                usable &= np.abs(freq - f) >= 1.12 / p.duration
    return sorted(points, key=lambda point: point.rate)


def _match(point: _Point, family: list[_Point], floor: float = 0.07) -> _Point | None:
    close = [
        p
        for p in family
        if abs(p.rate - point.rate) <= min(0.17, max(floor, p.error + point.error))
    ]
    return close[0] if len(close) == 1 else None


def _tree(x, fs, lo, hi, slots, blades, root):
    """Cached dyadic proposals; prune an independently stable complete node."""
    if (hi - lo) / fs < 6.2:
        return [(lo, hi, root)], []
    mid = (lo + hi) // 2
    halves = []
    for a, b in ((lo, mid), (mid, hi)):
        halves.append(_joint(_lines(x[:, a:b], fs), slots, blades, [p.rate for p in root]))
    steady = [
        p
        for p in root
        if all(_match(p, half) is not None for half in halves)
        and all(
            len([q for q in root if _match(q, half) == _match(p, half)]) == 1 for half in halves
        )
    ]
    if len(steady) == len(root) == slots:
        return [(lo, hi, root)], steady
    left, _ = _tree(x, fs, lo, mid, slots, blades, halves[0])
    right, _ = _tree(x, fs, mid, hi, slots, blades, halves[1])
    return [(lo, hi, root), *left, *right], steady


def _weak(p: _Lines, prior: float, occupied: list[_Point], slack: float) -> _Point | None:
    """Read actual local teeth at a scale established outside this window."""
    if len(p.hz) < 2:
        return None
    rates, j, w = _fit(p, np.array([prior]))
    if abs(rates[0] - prior) > slack:
        return None
    cells = j[0]
    ok = (w[0] > 0.35) & (p.db[cells] >= 9.4)
    for point in occupied:
        ok &= ~_owned_cells(p, point)[cells]
    if np.sum(ok) < 2 or np.max(p.db[cells[ok]]) < 12:
        return None
    point = _point(p, cells[ok], _ORDERS[ok], w[0, ok].sum(), "track_supported")
    return point if abs(point.rate - prior) <= slack else None


def _paths(times: np.ndarray, frames: list[list[_Point]]) -> list[list[tuple[int, _Point]]]:
    """Global minimum-cost sparse path cover with a no-link option at each node."""
    flat = [(i, p) for i, frame in enumerate(frames) for p in frame]
    n = len(flat)
    if not n:
        return []
    offsets = np.cumsum([0, *(len(frame) for frame in frames)])
    rows, cols, costs = list(range(n)), list(range(n, 2 * n)), [7.0] * n
    for k, (i, p) in enumerate(flat):
        for end in range(i + 1, len(frames)):
            dt = times[end] - times[i]
            if dt > _MAX_LINK_S:
                break
            if not frames[end]:
                continue
            for j, q in enumerate(frames[end]):
                sigma = 0.12 + 0.27 * dt + min(0.25, p.error + q.error)
                delta = abs(p.rate - q.rate)
                if delta > min(_MAX_DRIFT * dt, 3 * sigma):
                    continue
                # No identity through a competing near-equidistant track, or
                # through a measured crossing hidden inside a skipped frame.
                margin = 0.12 + min(0.15, p.error + q.error) + 0.10 * (dt - 1)
                if any(abs(r.rate - q.rate) < delta + margin for r in frames[i] if r is not p):
                    continue
                if any(abs(r.rate - p.rate) < delta + margin for r in frames[end] if r is not q):
                    continue
                if any(
                    abs(r.rate - (p.rate + (q.rate - p.rate) * (times[m] - times[i]) / dt))
                    <= margin + min(0.15, r.error)
                    for m in range(i + 1, end)
                    for r in frames[m]
                ):
                    continue
                cost = 0.05 + (delta / sigma) ** 2 + 1.15 * (dt - 1)
                if cost < 7.0:
                    rows.append(k)
                    cols.append(int(offsets[end] + j))
                    costs.append(cost)
    _, match = min_weight_full_bipartite_matching(
        sparse.csr_matrix((costs, (rows, cols)), shape=(n, 2 * n))
    )
    has_prev = set(match[match < n])
    out = []
    for start in range(n):
        if start in has_prev:
            continue
        path = []
        k = start
        while k < n:
            path.append(flat[k])
            k = match[k]
        out.append(path)
    return out


def _fill(times, spectra, frames, chain, levels, unavailable=None):
    """Interpolate only missing interior frames with a unique, bounded bridge."""
    if unavailable is None:
        unavailable = np.zeros(len(times), dtype=bool)
    for (i, p), (j, q) in zip(chain, chain[1:]):
        if j - i <= 1:
            continue
        dt = times[j] - times[i]
        slope = abs(q.rate - p.rate) / dt
        if dt > _MAX_LINK_S or slope > 0.42 or min(levels[i], levels[j]) <= 0:
            continue
        # Silence in *available* PCM is not a recorder blackout.
        if any(
            not unavailable[k] and levels[k] < min(levels[i], levels[j]) * 0.20
            for k in range(i + 1, j)
        ):
            continue
        bridge = []
        for k in range(i + 1, j):
            f = (times[k] - times[i]) / dt
            rate = (1 - f) * p.rate + f * q.rate
            # Conditional envelope: if |drift| <= 0.42, departure from the
            # connecting line is bounded by (0.42 + |slope|) from either end.
            uncertainty = (1 - f) * p.error + f * q.error + (0.42 + slope) * min(f, 1 - f) * dt
            if any(
                abs(rate - other.rate) <= max(0.24, uncertainty + other.error)
                for other in frames[k]
            ):
                break
            rivals = [
                other.rate
                for other in frames[k]
                if _simple_ratio(max(other.rate / rate, rate / other.rate))
            ]
            if any(
                not any(r is not p and abs(r.rate - rival) <= 0.25 for r in frames[i])
                or not any(r is not q and abs(r.rate - rival) <= 0.25 for r in frames[j])
                for rival in rivals
            ):
                # A trace alternating between two interpretations cannot
                # be promoted to two shafts by overlapping interpolation.
                break
            weak = None
            if not unavailable[k]:
                weak = _weak(spectra[k], rate, frames[k], min(0.20, max(0.10, uncertainty)))
            # With observed PCM but neither teeth nor another rotor, an
            # acoustic continuation and motor-off are indistinguishable.
            if weak is None and not unavailable[k] and not frames[k]:
                break
            bridge.append((k, weak or _Point(rate, uncertainty, 0.0, (), (), "interpolated")))
        else:
            yield from bridge


def track(x_ct, fs, *, expected, blades, anchors=None) -> dict:
    """Track distinguishable shaft rates; slots do not denote physical motors.

    ``times_s`` and all intervals are relative to the supplied PCM, even after
    antialiasing. ``anchors`` are optional accepted points from this exact
    source clock, not independent truth; they only seed/tip acoustic fits.
    Nonfinite samples explicitly mark unavailable PCM (e.g. recorder dropout).
    A bounded interior gap may then be interpolated; finite zero/silent PCM
    is not an unavailable interval and is never bridged.
    Unknown blade count cannot turn an apparent periodicity into a shaft rate.
    Null cells encode unknown or inactive, never zero rev/s.
    """
    x_ct = np.asarray(x_ct)
    fs = float(fs)
    if x_ct.ndim != 2 or not all(x_ct.shape) or not math.isfinite(fs) or fs <= 0:
        raise ValueError("expected nonempty channels-by-samples audio and positive sample rate")
    if expected not in (1, 2, 3, 4) or (blades is not None and blades not in range(1, 9)):
        raise ValueError("expected 1–4 track slots and a known 1–8 blade count or None")
    duration = x_ct.shape[1] / fs
    times = np.arange(0.5, duration, 1.0)
    if not len(times):
        times = np.array([duration / 2])
    missing = _missing_pcm(x_ct)
    x, decimation = _resample(x_ct, fs, duration, missing)
    sr = fs / decimation
    span_samples, energy, block = _activity(x, sr)
    link_spans = span_samples
    unavailable = np.zeros(len(times), dtype=bool)
    if missing:
        available = np.zeros(x.shape[1], dtype=bool)
        for a, b in span_samples:
            available[a:b] = True
        masked = [(math.ceil(a / decimation), math.ceil(b / decimation)) for a, b in missing]
        for a, b in masked:
            available[a:b] = False
        span_samples = _runs(available)
        for a, b in masked:
            available[a:b] = True
        link_spans = _runs(available)
        for a, b in missing:
            first, last = np.searchsorted(times, [a / fs, b / fs])
            unavailable[first:last] = True
    spans = [(a / sr, min(b / sr, duration)) for a, b in span_samples]
    active = np.zeros(len(times), dtype=bool)
    frame_rows: list[list[tuple[int, _Point]]] = [[] for _ in times]
    local_points: list[list[_Point]] = [[] for _ in times]
    local_spectra: list[_Lines | None] = [None] * len(times)
    historical = sorted(
        (
            float(anchor["time_s"]),
            tuple(float(s) for s in anchor["rates_rps"] if _MIN_RATE <= s <= _MAX_RATE),
        )
        for anchor in (anchors or ())
        if 0 <= anchor["time_s"] < duration
    )
    anchor_supports = [{"time_s": t, "rates_rps": list(aa), "matches": []} for t, aa in historical]
    root_tiles = [
        (a, min(a + _MAX_SAMPLES, x.shape[1])) for a in range(0, x.shape[1], _MAX_SAMPLES)
    ]
    cache = {}

    def root_fit(a, b):
        if (a, b) not in cache:
            cache[a, b] = _joint(_lines(x[:, a:b], sr), expected, blades, []) if blades else []
        return cache[a, b]

    def summary(a, b, points):
        return {
            "start_s": a / sr,
            "stop_s": min(b / sr, duration),
            "resolution_hz": sr / (b - a),
            "rates_rps": [p.rate for p in points],
            "order_support": [list(p.orders) for p in points],
        }

    full = [summary(a, b, root_fit(a, b)) for a, b in root_tiles]
    steady_summaries = []
    for (a, b), (start, stop) in zip(span_samples, spans, strict=True):
        idx = np.flatnonzero((times >= start) & (times < stop) & ~unavailable)
        active[idx] = True
        if not blades or not len(idx):
            continue
        probes, stable = [], []
        for left in range(a, b, _MAX_SAMPLES):
            right = min(b, left + _MAX_SAMPLES)
            nodes, settled = _tree(x, sr, left, right, expected, blades, root_fit(left, right))
            probes.extend(nodes)
            stable.append((left, right, settled))
        sample_centers = times[idx] * sr
        hints_at: list[list[float]] = [[] for _ in idx]
        for left, right, family in probes:
            first, last = np.searchsorted(sample_centers, [left, right])
            hint = [p.rate for p in family]
            for k in range(first, last):
                hints_at[k].extend(hint)
        seen, spectra = [], []
        refs_at = [[] for _ in idx]
        for h, (at, aa) in enumerate(historical):
            if not start <= at < stop:
                continue
            k = int(np.argmin(np.abs(times[idx] - at)))
            if abs(times[idx[k]] - at) <= 0.65:
                refs_at[k].append((h, aa))
        for k, i in enumerate(idx):
            c = sample_centers[k]
            radius = min(0.85 * sr, c - a, b - c)
            lo, hi = max(a, round(c - radius)), min(b, round(c + radius))
            spectrum = _lines(x[:, lo:hi], sr)
            spectra.append(spectrum)
            external = tuple(rate for _, rr in refs_at[k] for rate in rr)
            candidates = _joint(spectrum, expected, blades, hints_at[k], external)
            seen.append(candidates)
            for h, aa in refs_at[k]:
                for rate in aa:
                    near = [p for p in candidates if abs(p.rate - rate) <= 0.14]
                    if len(near) == 1:
                        anchor_supports[h]["matches"].append(
                            {"time_s": float(times[i]), "rate_rps": near[0].rate}
                        )
        for left, right, models in stable:
            ii = np.flatnonzero((times[idx] * sr >= left) & (times[idx] * sr < right))
            rescued = 0
            for p in models:
                matches = [(_match(p, seen[k]), int(k)) for k in ii]
                # Halves have already confirmed the root at independent times.
                # Short spectra can contradict it or corroborate it with local
                # teeth; they do not have to rediscover its shaft scale.
                deviation = [abs(q.rate - p.rate) for q, _ in matches if q is not None]
                if len(deviation) >= 3 and np.quantile(deviation, 0.8) > 0.055:
                    continue
                for q, k in matches:
                    if q is not None:
                        # For a close pair, don't snap one blended local peak
                        # to two globally resolved frequencies.
                        if _match(q, models) is not p:
                            continue
                        loc = next(j for j, r in enumerate(seen[k]) if r is q)
                        seen[k][loc] = replace(
                            q, rate=p.rate, error=max(p.error, abs(q.rate - p.rate))
                        )
                    elif len(seen[k]) < expected:
                        weak = _weak(spectra[k], p.rate, seen[k], 0.12)
                        if weak is not None:
                            seen[k].append(
                                replace(
                                    weak, rate=p.rate, error=max(p.error, abs(weak.rate - p.rate))
                                )
                            )
                            rescued += 1
            if models:
                steady_summaries.append({**summary(left, right, models), "local_rescues": rescued})
        for k, i in enumerate(idx):
            local_points[i], local_spectra[i] = seen[k], spectra[k]
    next_id = 0
    for a, b in link_spans:
        idx = np.flatnonzero((times >= a / sr) & (times < b / sr) & (active | unavailable))
        local_times = times[idx]
        seen = [local_points[i] for i in idx]
        spectra = [local_spectra[i] for i in idx]
        levels = energy[np.minimum((local_times * sr / block).astype(int), len(energy) - 1)]
        pending: list[list[tuple[int, _Point]]] = [[] for _ in idx]
        for path in _paths(local_times, seen):
            for k, point in path:
                frame_rows[idx[k]].append((next_id, point))
            for k, point in _fill(local_times, spectra, seen, path, levels, unavailable[idx]):
                pending[k].append((next_id, point))
            next_id += 1
        for k, i in enumerate(idx):
            row = frame_rows[i]
            # Reject all overlapping guesses, not only whichever hypothesis
            # happened to be considered second.
            for tid, p in pending[k]:
                if any(
                    oid != tid and abs(p.rate - q.rate) <= max(0.24, p.error + q.error)
                    for oid, q in pending[k]
                ):
                    continue
                if len(row) >= expected:
                    break
                if any(abs(p.rate - q.rate) <= max(0.24, p.error + q.error) for _, q in row):
                    continue
                row.append((tid, p))
    rates, provenance, errors, orders, identity = [], [], [], [], []
    last_slot: dict[int, int] = {}
    for i, row in enumerate(frame_rows):
        r = [None] * expected
        v = ["unresolved" if active[i] or unavailable[i] else "inactive"] * expected
        e = [None] * expected
        o = [[] for _ in range(expected)]
        ids = [None] * expected
        slots = set(range(expected))
        pairs = []
        for tid, point in sorted(row, key=lambda pair: pair[1].rate):
            slot = last_slot.get(tid)
            if slot in slots:
                pairs.append((slot, tid, point))
                slots.remove(slot)
        assigned = {tid for _, tid, _ in pairs}
        unassigned = sorted((p.rate, tid, p) for tid, p in row if tid not in assigned)
        pairs.extend(
            (slot, tid, p) for slot, (_, tid, p) in zip(sorted(slots), unassigned, strict=False)
        )
        for slot, tid, p in pairs:
            r[slot], v[slot], e[slot], o[slot], ids[slot] = (
                p.rate,
                p.provenance,
                p.error,
                list(p.orders),
                tid,
            )
            last_slot[tid] = slot
        rates.append(r)
        provenance.append(v)
        errors.append(e)
        orders.append(o)
        identity.append(ids)
    return {
        "times_s": times.tolist(),
        "rates_rps": rates,
        "provenance": provenance,
        "uncertainty_rps": errors,
        "order_support": orders,
        "track_id": identity,
        "diagnostics": {
            "duration_s": duration,
            "analysis_fs_hz": sr,
            "active_spans": [list(s) for s in spans],
            "pcm_unavailable_spans": [[a / fs, b / fs] for a, b in missing],
            "full_span_fft": {"tiled": len(full) > 1, "spectra": full},
            "steady_spans": steady_summaries,
            "anchor_supports": anchor_supports,
            "uncertainty_kind": "linewidth_residual_and_bridge_envelope_not_calibrated",
        },
    }
