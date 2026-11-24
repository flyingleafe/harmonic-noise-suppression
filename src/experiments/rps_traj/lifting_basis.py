"""Mean-preserving linear-prediction lifting basis on a prescribed coarse schedule.

Each contiguous run of one coarse code (0 off, 1 armed, 2 airborne, 3 landed)
gets its own lifting tree; no coefficient mixes samples of two runs. At every
iteration consecutive nodes of a run are paired from the left. A pair with
sample counts wL, wR becomes its weighted arithmetic average and the detail
``(R - L) - slope * (cR - cL)``, where ``slope`` is the centred slope of the
neighbouring parent averages (one-sided at a run's finite ends, zero when the
run has a single parent; never wrapping). An odd trailing node passes
untouched. Iterating leaves one DC per run, which is exactly that run's
arithmetic mean. The pair map has determinant ``wL/W + wR/W = 1`` and the
prediction is a triangular shear, so the transform has unit determinant on
the time axis. Linear signals give zero details below each run's top split,
and a constant offset changes only the DCs.

The time-domain mean depends only on the coarse schedule. With no repeated
arming or landing among the training schedules (airborne-only rigs, the single
instantaneous Blackbird arming, NeuroBEM) it is one level coefficient per mode,
an exogenous step response from rest at an actual motors-on command (none when
the record starts powered) and a rundown after an actual landing: 4 mean
coefficients, one noise group, no learned timing scalars. With repeated events
it reuses the regulated bridge's event-response hat bases
(``population_bridge.MEAN_GRIDS``), keeping only events repeated in training
and, per event, the contiguous hat range that training schedules reach. It also
uses the idle/hover/shelf gates that training supports. There is no extra
intercept: every column is zero while motors are off before arming. The arming
support (``population_bridge`` rule, median training hold) is the only
schedule-derived scalar. Noise groups reuse the regulated model's dyadic event
ages (arming impulse vs locked plateau, takeoff, touchdown vs rundown) for
events with repeated training support. A bin unseen in training maps to the
nearest seen bin of the same event. Group 0 is steady airborne flight. Codes
are only the scheduler's exogenous partition. Nothing here reads speeds.

The basis acts on the ``bridge_program`` encoding: floored inverse softplus,
then the mixer. Stopped motors are floored to ``bp.FLOOR`` (z = 0), not given a
censored likelihood; that approximation belongs to the encoding, not this basis.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np

from experiments.rps_traj import bridge_program as bp
from experiments.rps_traj import population_bridge as pb

FS = bp.FS
BLOCK = bp.BLOCK  # 32 samples, 1.28 s: the regulated model's cruise block
DRIVE_LEVEL = int(np.log2(BLOCK))
# The regulated model's dyadic arming ages (samples); the last bin is the plateau.
ON_EDGES = (1, 2, 4, 8, 16, 32, 64)
ARMED = pb.ARMED_BIN
TO0 = ARMED + 1
LA0 = TO0 + len(pb.TO_EDGES) + 1
EVENTS = ("on", "to", "la")
GATES = ("idle", "hover", "shelf")
# Event family of each regulated raw bin: cruise, off, arming, takeoff, landing.
_FAMILY = np.repeat(np.arange(5), (1, 1, TO0 - 2, LA0 - TO0, pb.BINS - LA0))
_WIDTH = tuple(tuple(len(grid) - 2 for grid in grids) for grids in pb.MEAN_GRIDS)
_ABSENT = -1.0e6  # seconds; beyond every ramp and hat support


@dataclass(frozen=True)
class _Step:
    left: np.ndarray
    right: np.ndarray
    wl: np.ndarray
    wr: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    slope: np.ndarray


class Tree:
    """Lifting on the time axis (-2) of ``(..., time, channels)`` arrays.

    Coefficients are stored in place: the DC of a run at its first sample and
    each detail at the first sample of its right child. ``level`` is
    ``ceil(log2(support))`` for details, 0 for DCs; ``support`` counts the original
    samples summarised by each row.
    """

    def __init__(self, n, starts):
        starts = np.asarray(starts, dtype=np.intp)
        if (
            n < 1
            or starts.ndim != 1
            or starts[0] != 0
            or (np.diff(starts) <= 0).any()
            or starts[-1] >= n
        ):
            raise ValueError("expected increasing run starts beginning at 0")
        self.n = int(n)
        self.starts = starts
        seg = np.zeros(n, np.intp)
        seg[starts[1:]] = 1
        seg = np.cumsum(seg)
        rep = np.arange(n)
        cnt = np.ones(n, np.intp)
        level = np.zeros(n, np.intp)
        support = np.ones(n, np.intp)
        steps = []
        while True:
            m = len(rep)
            same = seg[1:] == seg[:-1]
            first = np.r_[True, ~same]
            pos = np.arange(m) - np.maximum.accumulate(np.where(first, np.arange(m), 0))
            left = (pos % 2 == 0) & np.r_[same, False]
            if not left.any():
                break
            li = np.flatnonzero(left)
            ri = li + 1
            keep = np.ones(m, bool)
            keep[ri] = False
            merged = cnt.copy()
            merged[li] += cnt[ri]
            centre = rep + 0.5 * (cnt - 1)
            new_rep, new_cnt, new_seg = rep[keep], merged[keep], seg[keep]
            new_centre = new_rep + 0.5 * (new_cnt - 1)
            parent = (np.cumsum(keep) - 1)[li]
            nsame = new_seg[1:] == new_seg[:-1]
            lo = np.where(np.r_[False, nsame][parent], parent - 1, parent)
            hi = np.where(np.r_[nsame, False][parent], parent + 1, parent)
            slope = np.divide(
                centre[ri] - centre[li],
                new_centre[hi] - new_centre[lo],
                out=np.zeros(len(li)),
                where=hi != lo,
            )
            total = cnt[li] + cnt[ri]
            steps.append(
                _Step(
                    rep[li],
                    rep[ri],
                    cnt[li] / total,
                    cnt[ri] / total,
                    new_rep[lo],
                    new_rep[hi],
                    slope,
                )
            )
            level[rep[ri]] = len(steps)
            support[rep[ri]] = total
            rep, cnt, seg = new_rep, new_cnt, new_seg
        dc = np.zeros(n, bool)
        dc[rep] = True
        support[rep] = cnt
        for a in (level, support, dc):
            a.setflags(write=False)
        self.level, self.support, self.dc = level, support, dc
        self._steps = tuple(steps)
        self._anchored = None

    def condition_edge_on_mean(self, response):
        """Condition a known startup/rundown response on the realized phase DC.

        Subtracting a multiple of the root from detail rows is a determinant-one
        shear. The population level then controls only DC; a single fast arming
        edge cannot re-estimate an entire flight's operating speed.
        """
        z = self.analyse(response)[:, 0]
        row = np.flatnonzero((~self.dc) & (np.abs(z) > 1e-12))
        if row.size:
            root = self.starts[np.searchsorted(self.starts, row, side="right") - 1]
            nonzero = np.abs(z[root]) > 1e-12
            row, root = row[nonzero], root[nonzero]
            self._anchored = (row, root, (z[row] / z[root])[:, None])

    def _copy(self, x):
        y = np.array(x, dtype=np.float64)
        if y.ndim < 2 or y.shape[-2] != self.n:
            raise ValueError("expected (..., time, channels) on this schedule")
        return y

    def analyse(self, x):
        y = self._copy(x)
        for s in self._steps:
            a = y[..., s.left, :]
            b = y[..., s.right, :]
            y[..., s.left, :] = s.wl[:, None] * a + s.wr[:, None] * b
            y[..., s.right, :] = (b - a) - s.slope[:, None] * (y[..., s.hi, :] - y[..., s.lo, :])
        if self._anchored is not None:
            row, root, scale = self._anchored
            y[..., row, :] -= scale * y[..., root, :]
        return y

    def synth(self, z):
        y = self._copy(z)
        if self._anchored is not None:
            row, root, scale = self._anchored
            y[..., row, :] += scale * y[..., root, :]
        for s in reversed(self._steps):
            d = y[..., s.right, :] + s.slope[:, None] * (y[..., s.hi, :] - y[..., s.lo, :])
            p = y[..., s.left, :]
            y[..., s.left, :] = p - s.wr[:, None] * d
            y[..., s.right, :] = p + s.wl[:, None] * d
        return y


@dataclass(frozen=True, eq=False)
class Design:
    """Row-aligned with ``tree.analyse`` output; blocks are chronological."""

    tree: Tree
    mean: tuple  # four transformed (n, p_m) matrices: collective, roll, pitch, yaw
    level: np.ndarray
    dc: np.ndarray
    support: np.ndarray
    group: np.ndarray
    block: np.ndarray
    cruise: np.ndarray
    drive: np.ndarray
    code: np.ndarray


class _Clock(NamedTuple):
    """Sample ages since actual commands; None when the schedule has none."""

    start: np.ndarray | None  # motors on (first code >= 1 after an off start)
    on: np.ndarray | None  # armed hold (first code 1 after an off start)
    to: np.ndarray | None  # takeoff (first code 2 after a ground start)
    la: np.ndarray | None  # landing (first code 3 after a powered start)


def _codes(codes):
    c = np.asarray(codes, dtype=int)
    if c.ndim != 1 or len(c) < 1 or c.min() < 0 or c.max() > 3 or (np.diff(c) < 0).any():
        raise ValueError("expected ordered coarse phase codes")
    return c


def _clock(c):
    ix = np.arange(len(c))

    def age(ok, mask):
        return ix - int(np.argmax(mask)) if ok and mask.any() else None

    return _Clock(
        age(c[0] == 0, c >= 1), age(c[0] == 0, c == 1), age(c[0] < 2, c == 2), age(c[0] < 3, c == 3)
    )


def _sec(age, n):
    return np.full(n, _ABSENT) if age is None else age / FS


def _down(c, ck):
    return bp._ramp(ck.la / FS + 0.16, 2.6) if ck.la is not None else (c == 3).astype(float)


def _level_column(c, ck):
    # Unit step response from rest at the actual motors-on command; a record
    # that starts powered has no command and no fabricated takeoff.
    rise = bp._ramp(_sec(ck.start, len(c)) + 0.04, 0.24) if c[0] == 0 else (c >= 1).astype(float)
    return rise * (1.0 - _down(c, ck))


def _stretch(t, support):
    if support is None:
        return t
    return np.minimum(t, 0.32) + np.maximum(t - 0.32, 0.0) * (5.2 - 0.32) / (support - 0.32)


def _event_parts(c, ck, support):
    n = len(c)
    t0 = _stretch(_sec(ck.on, n), support)
    t1, t2 = _sec(ck.to, n), _sec(ck.la, n)
    arm = bp._ramp(t0 + 0.04, 0.24) if ck.on is not None else (c >= 1).astype(float)
    lift = bp._ramp(t1 + 0.08, 1.6) if ck.to is not None else (c >= 2).astype(float)
    down = _down(c, ck)
    gates = {
        "idle": arm * (1.0 - lift) * (1.0 - down),
        "hover": lift * (1.0 - down),
        "shelf": bp._ramp(t2 + 0.12, 0.48) * (1.0 - bp._ramp(t2 - 1.36, 2.32)),
    }
    # A plateau level must affect only the arithmetic phase mean. If the
    # same coefficient also controls the sharp rise at a phase boundary,
    # hundreds of details can move the level away from every observed DC.
    # Keep the former smooth motion as a separate, fitted transient column.
    for name, code in (("idle", 1), ("hover", 2)):
        level = (c == code).astype(float)
        gates[name + "_response"] = np.where(c > 0, gates[name] - level, 0.0)
        gates[name] = level
    hats = tuple(
        {
            "on": bp._hats(t0, grids[0]) * (1.0 - lift)[:, None],
            "to": bp._hats(t1, grids[1]),
            "la": bp._hats(t2, grids[2]),
        }
        for grids in pb.MEAN_GRIDS
    )
    return hats, gates, t0


def _row_bins(raw, starts):
    # A DC summarises its whole run: use the bin at the run's centre sample.
    rows = raw.copy()
    ends = np.r_[starts[1:], len(raw)]
    rows[starts] = raw[(starts + ends - 1) // 2]
    return rows


@dataclass(frozen=True)
class Structure:
    """Schedule-only topology chosen from training schedules; no speeds."""

    kind: str = "compact"
    events: tuple[str, ...] = ()
    gates: tuple[str, ...] = ()
    hats: tuple = ((), ())  # per grid set (collective, contrast): ((event, lo, hi), ...)
    support: float | None = None
    bins: tuple[int, ...] = (0,)

    def __post_init__(self):
        ok = self.kind in ("compact", "events")
        if self.kind == "compact":
            ok &= not self.events and not self.gates and self.hats == ((), ())
            ok &= self.support is None and self.bins == (0,)
        else:
            ok &= set(self.events) <= set(EVENTS) and set(self.gates) <= set(GATES)
            ok &= (self.support is not None) == ("on" in self.events)
            ok &= self.support is None or 0.32 < self.support <= 5.2
            ok &= len(self.hats) == 2 and all(
                e in self.events and 0 <= lo < hi <= _WIDTH[g][EVENTS.index(e)]
                for g, h in enumerate(self.hats)
                for e, lo, hi in h
            )
            b = np.asarray(self.bins)
            ok &= b.size >= 1 and b[0] == 0 and (np.diff(b) > 0).all() and b[-1] < pb.BINS
        if not ok:
            raise ValueError("invalid lifting structure")

    @classmethod
    def from_training(cls, schedules):
        cs = [_codes(s) for s in schedules]
        if not cs:
            raise ValueError("no training schedules")
        cks = [_clock(c) for c in cs]

        def repeated(name):
            return sum(getattr(k, name) is not None for k in cks) >= 2

        if not (repeated("start") or repeated("la")):
            return cls()
        events = tuple(e for e in EVENTS if repeated(e))
        support = None
        if "on" in events:
            holds = [np.count_nonzero(c == 1) / FS for c in cs if (c == 1).any()]
            support = float(np.clip(np.median(holds) - 0.24, 0.48, 5.2))
        seen = (
            any((c == 1).any() for c in cs),
            any((c == 2).any() for c in cs),
            "la" in events,
        )
        gates = tuple(g for g, ok in zip(GATES, seen, strict=True) if ok)
        parts = [_event_parts(c, k, support) for c, k in zip(cs, cks, strict=True)]
        hats = []
        for g in range(2):
            kept = []
            for e in events:
                reach = [(p[0][g][e] != 0).any(axis=0) for p in parts]
                used = np.flatnonzero(np.any(reach, axis=0))
                if used.size:
                    kept.append((e, int(used[0]), int(used[-1]) + 1))
            hats.append(tuple(kept))
        base = cls("events", events, gates, tuple(hats), support, tuple(range(pb.BINS)))
        present = {0}
        for c, k, p in zip(cs, cks, parts, strict=True):
            starts = np.r_[0, np.flatnonzero(np.diff(c)) + 1]
            present.update(np.unique(_row_bins(base._raw(c, k, p[2]), starts)).tolist())
        return replace(base, bins=tuple(sorted(int(b) for b in present)))

    @property
    def n_params(self):
        """Learned timing scalars: only the schedule-derived arming support."""
        return int(self.support is not None)

    @property
    def n_groups(self):
        return len(self.bins)

    def layout(self, mode):
        """``(name, width)`` column blocks of ``design(...).mean[mode]``."""
        if self.kind == "compact":
            return (("level", 1),)
        g = int(mode != 0)
        return (
            tuple((e, hi - lo) for e, lo, hi in self.hats[g])
            + tuple((k, 1) for k in self.gates)
            + tuple((k + "_response", 1) for k in self.gates if k in ("idle", "hover"))
        )

    @property
    def n_mean(self):
        return sum(w for m in range(4) for _, w in self.layout(m))

    def penalty(self, mode):
        """The regulated model's fixed roughness penalty for this mode's columns."""
        lay = self.layout(mode)
        p = sum(w for _, w in lay)
        R = np.eye(p) * 1e-5
        start = 0
        for name, w in lay:
            if name in EVENTS:
                D = np.diff(np.pad(np.eye(w), ((1, 1), (0, 0))), n=2, axis=0)
                R[start : start + w, start : start + w] += D.T @ D / 25.0**2
            start += w
        return R

    def _raw(self, c, ck, t0):
        # The regulated model's sample bins, restricted to repeated events.
        raw = np.zeros(len(c), np.intp)
        raw[(c == 0) | (c == 3)] = 1
        one = c == 1
        if "on" in self.events and ck.on is not None:
            b = 2 + np.searchsorted(ON_EDGES, ck.on[one], side="right")
            b[t0[one] >= 5.2] = ARMED
            raw[one] = b
        else:
            raw[one] = ARMED
        if "to" in self.events and ck.to is not None:
            take = (ck.to >= -3) & (ck.to < 96) & (c >= 1)
            raw[take] = TO0 + np.searchsorted(pb.TO_EDGES, ck.to[take], side="right")
        if "la" in self.events and ck.la is not None:
            touch = (ck.la >= -16) & (ck.la < 200) & (c >= 2)
            raw[touch] = LA0 + np.searchsorted(pb.LA_EDGES, ck.la[touch], side="right")
        return raw

    def _table(self):
        present = np.asarray(self.bins)
        fam = _FAMILY[present]
        out = np.zeros(pb.BINS, np.intp)
        for raw in range(pb.BINS):
            same = np.flatnonzero(fam == _FAMILY[raw])
            if same.size:
                dist = np.abs(present[same] - raw)
                out[raw] = same[np.lexsort((present[same], dist))[0]]
        return out

    def design(self, codes):
        c = _codes(codes)
        n = len(c)
        ck = _clock(c)
        starts = np.r_[0, np.flatnonzero(np.diff(c)) + 1]
        tree = Tree(n, starts)
        if self.kind == "compact":
            x = _level_column(c, ck)[:, None]
            if ck.start is not None or ck.la is not None:
                tree.condition_edge_on_mean(x)
            cols = (x, x)
            raw = np.zeros(n, np.intp)
            steady = c == 2
        else:
            hats, gates, t0 = _event_parts(c, ck, self.support)
            cols = tuple(
                np.column_stack(
                    [hats[g][e][:, lo:hi] for e, lo, hi in self.hats[g]]
                    + [gates[k] for k in self.gates]
                    + [gates[k + "_response"] for k in self.gates if k in ("idle", "hover")]
                )
                for g in range(2)
            )
            raw = self._raw(c, ck, t0)
            steady = (c == 2) & (raw == 0)
        collective, contrast = (tree.analyse(x) for x in cols)
        for a in (collective, contrast):
            a.setflags(write=False)
        group = self._table()[_row_bins(raw, starts)]
        ends = np.r_[starts[1:], n]
        n_blocks = -(-(ends - starts) // BLOCK)
        seg = np.repeat(np.arange(len(starts)), ends - starts)
        block = np.r_[0, np.cumsum(n_blocks)][seg] + (np.arange(n) - starts[seg]) // BLOCK
        cruise = np.bincount(block, weights=(~steady).astype(float), minlength=n_blocks.sum()) == 0
        drive = (tree.level == DRIVE_LEVEL) & cruise[block]
        return Design(
            tree,
            (collective, contrast, contrast, contrast),
            tree.level,
            tree.dc,
            tree.support,
            group,
            block,
            cruise,
            drive,
            c,
        )

    def to_dict(self):
        return {
            "family": "lifting_basis",
            "kind": self.kind,
            "events": list(self.events),
            "gates": list(self.gates),
            "hats": [[[e, lo, hi] for e, lo, hi in h] for h in self.hats],
            "support": self.support,
            "bins": list(self.bins),
            "n_params": self.n_params,
        }

    @classmethod
    def from_dict(cls, data):
        if data.get("family") != "lifting_basis":
            raise ValueError("not a lifting_basis structure")
        return cls(
            str(data["kind"]),
            tuple(str(e) for e in data["events"]),
            tuple(str(g) for g in data["gates"]),
            tuple(tuple((str(e), int(lo), int(hi)) for e, lo, hi in h) for h in data["hats"]),
            None if data["support"] is None else float(data["support"]),
            tuple(int(b) for b in data["bins"]),
        )
