"""Duration-profiled population bridge with different random laws at command boundaries.

The motor arming response cannot last the whole standby: its support is a
rig-level parameter. Touchdown and rundown have a second time dilation. Choose
both by the same complete-record penalized likelihood as the remaining fit.
A bridge cannot use a noisy airborne endpoint as a parent of a settled-idle
sample. Its innovations restart at boundaries, about a continuous command
mean (the command itself may also jump at arming). Only physical tachometry
is modeled: auxiliary command-channel records are excluded *before* fitting.
See bridge_program for the two-level latent programme and midpoint likelihood.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve

from experiments.rps_traj import bridge_program as bp

# Physical time supports, not chosen on the scored phase windows. Samples at
# 25 Hz. The grid avoids a non-identifiable time warp within each fine knot.
SETTLING_SECONDS = (0.12, 0.24, 0.48, 0.96, 1.6, 2.4, 3.2, 4.8, 6.4)
TOUCHDOWN_DILATIONS = (0.125, 0.25, 0.5, 1.0, 1.5, 2.0)


def design(schedule, settling, dilation):
    c = np.asarray(schedule)
    if c.ndim != 1 or not len(c) or np.any((c < 0) | (c > 3)) or np.any(np.diff(c.astype(int)) < 0):
        raise ValueError("expected an ordered nonempty coarse schedule")
    n = len(c)
    ix = np.arange(n)

    def age(code):
        at = np.flatnonzero(c == code)
        return (ix - at[0]) / bp.FS if at.size else np.full(n, -1.0e6)

    on, to, land = (age(g) for g in (1, 2, 3))
    u, v = 4 * on / settling, land / dilation
    arm = bp._ramp(u + 0.04, 0.48) if (c == 1).any() else (c >= 2).astype(float)
    lift = bp._ramp(to + 0.08, 1.4) if (c == 2).any() else (c == 3).astype(float)
    if c[0] == 2:
        lift[:] = 1.0
    down = bp._ramp(v + 0.12, 4.5)
    fan = bp._ramp(v + 0.4, 0.8) * (1 - bp._ramp(v - 1.35, 1.6))
    gates = np.stack((arm * (1 - lift) * (1 - down), lift * (1 - down), fan), axis=1)
    bases = []
    for gs in (bp.KNOTS, bp.DIFF_KNOTS):
        # Include the arming sample itself, and let the drive respond immediately
        # before the scheduled takeoff. Never extrapolate the arming transient.
        start_grid = gs[0].copy()
        start_grid[0] = -0.12
        start = bp._hats(u, start_grid) * (c == 1)[:, None]
        take = bp._hats(to, gs[1]) * (c >= 1)[:, None]
        end = bp._hats(v, gs[2])
        bases.append(np.column_stack((start, take, end)))
    # Landing has a sudden contact impulse and a quieter rundown. Startup
    # impulses are distinguished from subsequent run-up by the fitted dilation.
    phase = np.select(
        (
            c == 0,
            (c == 1) & (on < min(0.16, settling / 4)),
            (c == 1) & (on < settling),
            c == 1,
            (c == 2) & (to < 3.0),
            (v >= -0.24) & (v < 0.48),
            (v >= 0.48) & (v < 4.8),
            c == 3,
        ),
        (0, 6, 1, 2, 3, 7, 5, 0),
        default=4,
    ).astype(int)
    # Draw the wavelet tree of each stationary interval separately. The
    # conditional mean and flight-level style are still shared across all of them.
    left, right, level, block = (np.zeros(n, dtype=int) for _ in range(4))
    frac = np.zeros(n)
    order, starts, bphase = [], [], []
    bounds = np.r_[0, np.flatnonzero(np.diff(phase)) + 1, n]
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        left[lo] = right[lo] = lo
        starts.append(len(order))
        order.append(lo)
        block[lo] = len(bphase)
        bphase.append(phase[lo])
        a = lo
        while a < hi - 1:
            b = min(a + bp.BLOCK, hi - 1)
            bi = len(bphase)
            starts.append(len(order))
            order.append(b)
            bphase.append(phase[b])
            left[b] = right[b] = a
            block[b] = bi
            queue = [(a, b)]
            for i, j in queue:
                if j - i <= 1:
                    continue
                m = (i + j) // 2
                left[m], right[m], frac[m], block[m] = i, j, (m - i) / (j - i), bi
                level[m] = min(5, 1 + int(np.floor(np.log2(bp.BLOCK / (j - i)))))
                order.append(m)
                queue.extend(((i, m), (m, j)))
            a = b
    return bp.Design(
        tuple(bases),
        gates,
        phase,
        left,
        right,
        frac,
        level,
        block,
        np.asarray(bphase),
        np.asarray(order),
        np.asarray(starts),
    )


class SettledBridge(bp.BridgeModel):
    def __init__(self, p, allowed, history, settling, dilation, profile):
        super().__init__(p, allowed, history)
        self.settling, self.dilation, self.profile = settling, dilation, profile

    @property
    def n_params(self):
        return super().n_params + 2

    def sample(self, schedule, n, rng):
        return self._draw(design(schedule, self.settling, self.dilation), n, rng)

    def to_dict(self):
        return super().to_dict() | dict(
            family="settled_bridge",
            settling=self.settling,
            dilation=self.dilation,
            profile=self.profile,
        )

    @classmethod
    def from_dict(cls, d):
        model = bp.BridgeModel.from_dict(d)
        return cls(
            model.p, model.allowed, model.history, d["settling"], d["dilation"], d["profile"]
        )


def _start(tracks, allowed, settling, dilation):
    k = len(allowed)
    source = np.ones(len(tracks), bool)
    tr = [bp._make_track(x, c, k, design(c, settling, dilation)) for x, c in tracks]
    p = bp._initial(tr, source, allowed)
    p["activity"][0] = 1.0
    p["phasevar"] = np.vstack((p["phasevar"], p["phasevar"][[1, 5]]))
    pen = bp._penalties(p, tr[0][0])
    # Initialize every mixture component. Choosing a label again after each
    # coordinate's fit can silently vacate one before the first E step.
    order = np.argsort(np.asarray([np.std(t[1][t[0].phase == 4, -1]) for t in tr]))
    labels = np.zeros(len(tr), int)
    for j, ids in enumerate(np.array_split(order, k)):
        labels[ids] = j
    for h in range(4):
        H = pen[h].copy()
        u = np.zeros(len(H))
        for t, j in zip(tr, labels, strict=True):
            X = t[3][j][h]
            H += X.T @ X
            u += X.T @ t[1][:, h]
        p["beta"][h] = solve(H, u, assume_a="pos", check_finite=False)
    ll, stats = bp._estep(tr, p, source, allowed)
    return tr, pen, p, stats, [float(bp._obj(ll, p, pen))]


def _opt(tr, pen, p, stats, history, allowed, n):
    source = np.ones(len(tr), bool)
    for _ in range(n):
        cand = bp._joint_mstep(tr, p, stats, allowed, pen)
        ll, est = bp._estep(tr, cand, source, allowed)
        obj = float(bp._obj(ll, cand, pen))
        if not np.isfinite(obj):
            raise FloatingPointError("nonfinite adaptive bridge objective")
        if obj < history[-1]:
            break
        delta = obj - history[-1]
        history.append(obj)
        p, stats = cand, est
        if delta < 1e-6 * sum(len(t[1]) for t in tr):
            break
    return tr, pen, p, stats, history


def fit(tracks, *, measured=None, max_iter=80):
    if measured is None:
        measured = [True] * len(tracks)
    tracks = [t for t, physical in zip(tracks, measured, strict=True) if physical]
    if not tracks:
        raise ValueError("no measured flight")
    allowed = np.ones(min(3, len(tracks)), bool)
    profile = []
    best = None
    settle = 4.8
    dilation = 1.0
    trial_iters = min(max_iter, 4)
    # A coordinate profile of the full marginal likelihood, not the phase
    # metric. Within each clock proposal refit both programme and response.
    for axis, grid in enumerate((SETTLING_SECONDS, TOUCHDOWN_DILATIONS)):
        best_obj = -np.inf if best is None else best[-1][-1]
        before = (settle, dilation)
        for value in grid:
            times = (value, before[1]) if axis == 0 else (before[0], value)
            if axis and times == before:
                continue
            trial = _opt(*_start(tracks, allowed, *times), allowed, trial_iters)
            obj = trial[-1][-1]
            profile.append([*times, obj])
            if obj > best_obj:
                best_obj, best, settle, dilation = obj, trial, *times
    assert best is not None
    _, _, p, _, history = _opt(*best, allowed, max(0, max_iter - trial_iters))
    return SettledBridge(p, allowed, history, settle, dilation, profile)
