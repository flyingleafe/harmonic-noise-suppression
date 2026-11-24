"""Population feedback bridge with a nonstationary disturbance envelope.

One whole-flight likelihood over an exchangeable flight style and signed
semi-Markov maneuvers. Residuals are propagated in *standard deviation units*:
a disturbance exiting a high-variance stage cannot persist at that amplitude
in a quiet stage. The prescribed command changes the predictor's timescale:
sample-rate feedback during actuation, endpoint/midpoint control in sustained
flight. Both use the same standardized Gaussian-innovation law. The envelope
is fitted jointly at fixed dyadic event ages, with separate collective and
contrast scales. No target-observed stage boundaries, reset, or fitted chunk.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize

from experiments.rps_traj import bridge_program as bp

MEAN_GRIDS = (
    (
        np.r_[-0.04, 0.04, 0.12, np.arange(0.24, 2.89, 0.16), 3.3, 4.0, 5.2],
        np.array([-0.20, -0.08, -0.04, 0.0, 0.16, 0.40, 0.72, 1.12, 1.6, 2.2, 3.0, 4.0]),
        np.array(
            [
                -2.0,
                -1.0,
                -0.48,
                -0.24,
                -0.12,
                -0.04,
                0.0,
                0.04,
                0.08,
                0.16,
                0.32,
                0.56,
                0.88,
                1.2,
                1.52,
                1.84,
                2.16,
                2.48,
                2.88,
                3.4,
                4.0,
                4.8,
                5.6,
                6.4,
                8.0,
            ]
        ),
    ),
    (
        np.array([-0.04, 0.48, 1.3, 1.9, 3.0, 5.2]),
        np.array([-0.20, -0.04, 0.16, 0.56, 1.2, 2.0, 4.0]),
        np.array([-2.0, -0.5, -0.08, 0.08, 0.32, 0.8, 1.44, 2.0, 2.8, 4.0, 8.0]),
    ),
)
# The left and right tails share the same off-sensor law. Ages in samples
# avoid roundoff exactly at a bin boundary. Bin zero is the cruise reference.
ON_EDGES = (1, 3, 8, 24, 48, 80, 128)
TO_EDGES = (0, 2, 8, 24, 48)
LA_EDGES = (-4, 0, 2, 5, 12, 32, 64, 128)
BINS = 2 + (len(ON_EDGES) + 1) + (len(TO_EDGES) + 1) + (len(LA_EDGES) + 1)
GROUP = np.array([0, 1, 1, 1])
ARMED_BIN = 2 + len(ON_EDGES)


def design(schedule, support=None, regulated=False):
    d = bp.design(schedule)
    c = np.asarray(schedule)
    ix = np.arange(len(c))

    def age(code):
        loc = np.flatnonzero(c == code)
        return ix - loc[0] if len(loc) else np.full(len(ix), -(10**7))

    on, to, la = (age(g) for g in (1, 2, 3))
    if c[0] == 2:
        # An airborne recording starts mid-flight. Its first sample is not a
        # takeoff command (no drive basis, elevated transient noise or reset).
        to = np.full_like(to, -(10**7))
    t0, t1, t2 = on / bp.FS, to / bp.FS, la / bp.FS
    if support is not None:
        # Keep fast arming dynamics in seconds, and ensure plateau coefficients
        # are identifiable when all train schedules start flying early.
        t0 = np.minimum(t0, 0.32) + np.maximum(t0 - 0.32, 0.0) * (5.2 - 0.32) / (support - 0.32)
    arm = bp._ramp(t0 + 0.04, 0.24) if (c == 1).any() else (c >= 2).astype(float)
    lift = bp._ramp(t1 + 0.08, 1.6) if (c == 2).any() else (c == 3).astype(float)
    if c[0] == 2:
        lift[:] = 1.0
    down = bp._ramp(t2 + 0.16, 2.6)
    shelf = bp._ramp(t2 + 0.12, 0.48) * (1 - bp._ramp(t2 - 1.36, 2.32))
    d.gates = np.column_stack((arm * (1 - lift) * (1 - down), lift * (1 - down), shelf))
    d.collective_shelf = True
    d.base = tuple(
        np.column_stack(
            (bp._hats(t0, gs[0]) * (1 - lift[:, None]), bp._hats(t1, gs[1]), bp._hats(t2, gs[2]))
        )
        for gs in MEAN_GRIDS
    )
    bins = np.zeros(len(c), int)
    bins[(c == 0) | (c == 3)] = 1
    on_edges = (1, 2, 4, 8, 16, 32, 64) if regulated else ON_EDGES
    bins[c == 1] = 2 + np.searchsorted(on_edges, on[c == 1], side="right")
    if support is not None:
        bins[(c == 1) & (t0 >= 5.2)] = ARMED_BIN
    ib = 3 + len(ON_EDGES)
    take = (to >= -3) & (to < 96) & (c >= 1)
    bins[take] = ib + np.searchsorted(TO_EDGES, to[take], side="right")
    ib += len(TO_EDGES) + 1
    touch = (la >= -16) & (la < 200) & (c >= 2)
    bins[touch] = ib + np.searchsorted(LA_EDGES, la[touch], side="right")
    d.phase = np.where(bins == 0, 4, 1)
    # Keep the preceding residual at every command change. Only the predictor
    # topology changes; the stochastic process and the style never restart.
    n = len(c)
    d.prev = np.maximum(0, ix - 1)
    d.right = d.prev.copy()
    d.frac = np.zeros(n)
    d.level = np.full(n, 6)
    d.block = np.zeros(n, int)
    order = []
    starts = []
    phases = []
    bounds = np.r_[0, np.flatnonzero(np.diff(d.phase)) + 1, n]
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        p = d.phase[a]
        starts.append(len(order))
        d.block[a:b] = len(phases)
        phases.append(p)
        if p != 4:
            order.extend(range(a, b))
            continue
        order.append(a)
        d.level[a] = 0
        while a < b - 1:
            last = min(a + bp.BLOCK, b - 1)
            idx = len(phases)
            starts.append(len(order))
            phases.append(p)
            order.append(last)
            d.block[a + 1 : last + 1] = idx
            d.prev[last] = d.right[last] = a
            d.level[last] = 0
            queue = [(a, last)]
            for i, j in queue:
                if j - i <= 1:
                    continue
                m = (i + j) // 2
                d.prev[m] = i
                d.right[m] = j
                d.frac[m] = (m - i) / (j - i)
                d.level[m] = min(5, 1 + int(np.floor(np.log2(bp.BLOCK / (j - i)))))
                order.append(m)
                queue.extend(((i, m), (m, j)))
            a = last
    d.block_phase = np.asarray(phases)
    d.order = np.asarray(order)
    d.block_starts = np.asarray(starts)
    if regulated:
        d.var_level = d.level.copy()
        d.level[c == 1] = 7
    return d, bins


def _transport(d, scale):
    return tuple(np.minimum(1.0, scale / scale[parent]) for parent in (d.prev, d.right))


def _normalised(tr, bins, q, bounded=False):
    from dataclasses import replace

    out = []
    jac = 0.0
    for (d, y, _, x, _), idx in zip(tr, bins, strict=True):
        scale = q[idx[:, None], GROUP]
        z = y * scale
        xx = [[a * scale[:, h, None] for h, a in enumerate(row)] for row in x]
        if bounded:
            left, right = _transport(d, scale)
            nd = replace(d, transport=(left, right))
            weights = ((1.0 - d.frac[:, None]) * left, d.frac[:, None] * right)

            def project(v, h=None, d=d, weights=weights):
                wl, wr = weights if h is None else tuple(w[:, h, None] for w in weights)
                result = wl * v[d.prev] + wr * v[d.right]
                result[0] = 0.0
                return result

            out.append(
                (nd, z, project(z), xx, [[project(a, h) for h, a in enumerate(row)] for row in xx])
            )
        else:
            out.append(
                (d, z, bp._neighbor(z, d), xx, [[bp._neighbor(a, d) for a in xxj] for xxj in xx])
            )
        jac += np.log(scale).sum()
    return out, jac


def _penalty(p):
    out = []
    for h, b in enumerate(p["beta"]):
        R = np.eye(len(b)) * 1e-5
        start = 0
        for grid in MEAN_GRIDS[int(h != 0)]:
            width = len(grid) - 2
            D = np.diff(np.pad(np.eye(width), ((1, 1), (0, 0))), n=2, axis=0)
            R[start : start + width, start : start + width] += D.T @ D / 25.0**2
            start += width
        # Flight equilibria have weak exchangeable shrinkage, not a zero target.
        K = len(p["mix"])
        S = np.kron(np.eye(K) - np.ones((K, K)) / K, np.eye(3))
        R[start:, start:] += S / 20.0**2
        out.append(R)
    return out


def _envelope(tr, bins, p, stats, old, bounded=False):
    # E[-log p] is a convex quadratic minus sum(n_b log q_b) in the inverse
    # scales q. Accumulate the tiny normal equations including parent terms;
    # treating parent scales as constants would no longer maximize likelihood.
    H = np.zeros((2, BINS, BINS))
    u = np.zeros((2, BINS))
    count = np.zeros((2, BINS))
    for (d, y, _, x, _), idx, (resp, gamma, _) in zip(tr, bins, stats, strict=True):
        source_bins = (idx, idx[d.prev], idx[d.right])
        root = (d.level == 0) & (d.phase == 4)
        mask = d.prev != np.arange(len(y))
        for h, group in enumerate(GROUP):
            parent = (
                tuple(np.where(old[idx, group] < old[b, group], idx, b) for b in source_bins)
                if bounded
                else source_bins
            )
            count[group] += np.bincount(idx, minlength=BINS)
            rho = p["rho"][d.level, h] * mask
            for j, w in enumerate(resp):
                if w < 1e-12:
                    continue
                e = y[:, h] - x[j][h] @ p["beta"][h]
                V = bp._var(p, d, j)[:, h]
                post = gamma[j][d.block]
                act = post[:, 1:].sum(axis=1)
                boost = np.where(d.phase == 4, p["boost"][h], 1.0)
                prec = w * (1 - act + act / boost) / V
                rhs = w * (post @ bp.SIGN) * root * p["pulse"][h] / (V * boost)
                v = (e, -rho * (1 - d.frac) * e[d.prev], -rho * d.frac * e[d.right])
                for a, ia in enumerate(parent):
                    u[group] += np.bincount(ia, weights=rhs * v[a], minlength=BINS)
                    for b, ib in enumerate(parent):
                        H[group] += np.bincount(
                            ia * BINS + ib, weights=prec * v[a] * v[b], minlength=BINS**2
                        ).reshape(BINS, BINS)
    new = old.copy()
    for g in range(2):
        free = np.ones(BINS, bool)
        free[[0, ARMED_BIN]] = False
        free &= count[g] > 0
        if not free.any():
            continue

        def loss(z, g=g, free=free):
            q = new[:, g].copy()
            q[free] = z
            grad = H[g] @ q - u[g]
            f = 0.5 * q @ H[g] @ q - u[g] @ q - count[g] @ np.log(q)
            return f, (grad - count[g] / q)[free]

        opt = minimize(
            loss,
            old[free, g],
            jac=True,
            bounds=[(0.005, 300.0)] * free.sum(),
            method="L-BFGS-B",
            options={"ftol": 1e-11, "maxiter": 160},
        )
        new[free, g] = opt.x
    return new


class PopulationBridge(bp.BridgeModel):
    regulated = False
    encode = None
    decode = None

    def __init__(self, p, allowed, history, q, support=None):
        super().__init__(p, allowed, history)
        self.q = q
        self.support = support

    def sample(self, schedule, n, rng):
        d, idx = design(schedule, self.support, self.regulated)
        q = self.q[idx[:, None], GROUP]
        if self.support is not None:
            d.transport = _transport(d, q)
        return self._draw(d, n, rng, 1 / q, decode=self.decode)

    @property
    def n_params(self):
        # Constant, unit coarse phase variances have no fitted entries.
        return (
            super().n_params
            - (self.p["phasevar"].size - 4)
            + 2 * (BINS - 2)
            + int(self.support is not None)
            - 2 * int(self.regulated)
        )

    def to_dict(self):
        return super().to_dict() | dict(
            family="bounded_bridge" if self.support is not None else "population_bridge",
            inverse_scales=self.q.tolist(),
            arming_support=self.support,
        )

    @classmethod
    def from_dict(cls, d):
        m = bp.BridgeModel.from_dict(d)
        return cls(
            m.p, m.allowed, m.history, np.asarray(d["inverse_scales"]), d.get("arming_support")
        )


def fit(tracks, *, measured=None, max_iter=80, bounded=False, model_type=PopulationBridge):
    if measured is not None:
        tracks = [t for t, b in zip(tracks, measured, strict=True) if b]
    if not tracks:
        raise ValueError("no physical tachometry")
    K = min(len(tracks), 3)
    allowed = np.ones(K, bool)
    src = np.ones(len(tracks), bool)
    support = None
    if bounded:
        holds = [np.count_nonzero(c == 1) / bp.FS for _, c in tracks if np.any(c == 1)]
        support = float(np.clip(np.median(holds) - 0.24, 0.48, 5.2)) if holds else 5.2
    des = [design(c, support, model_type.regulated) for _, c in tracks]
    bins = [idx for _, idx in des]
    tr = [
        bp._make_track(x, c, K, d, model_type.encode)
        for (x, c), (d, _) in zip(tracks, des, strict=True)
    ]
    p = bp._initial(tr, src, allowed)
    p["rho"] = np.vstack((p["rho"], [0.5] * 4))
    p["basevar"] = np.vstack((p["basevar"], np.ones(4)))
    if model_type.regulated:
        p["rho"] = np.vstack((p["rho"], np.zeros(4)))
    p["phasevar"][:] = 1.0
    p["activity"][:] = 1.0
    R = _penalty(p)
    labels = np.zeros(len(tr), int)
    for j, subset in enumerate(
        np.array_split(np.argsort(np.asarray([np.std(t[1][t[0].phase == 4, -1]) for t in tr])), K)
    ):
        labels[subset] = j
    for h in range(4):
        H = R[h].copy()
        u = np.zeros(len(H))
        for t, j in zip(tr, labels, strict=True):
            X = t[3][j][h]
            H += X.T @ X
            u += X.T @ t[1][:, h]
        p["beta"][h] = solve(H, u, assume_a="pos", check_finite=False)
    q = np.ones((BINS, 2))
    norm, jac = _normalised(tr, bins, q, bounded)
    ll, stats = bp._estep(norm, p, src, allowed)
    hist = [float(bp._obj(ll + jac, p, R))]
    for _ in range(max_iter):
        cand = bp._joint_mstep(norm, p, stats, allowed, R)
        cand["phasevar"][:] = 1.0
        # A unit pulse in a closed-loop motor controller is a decaying response,
        # not an integrated speed error. Enforce that on adjacent-time poles.
        cand["rho"][6] = np.clip(cand["rho"][6], -0.95, 0.95)
        q1 = _envelope(tr, bins, cand, stats, q, bounded)
        norm1, jac1 = _normalised(tr, bins, q1, bounded)
        ll, stats1 = bp._estep(norm1, cand, src, allowed)
        obj = float(bp._obj(ll + jac1, cand, R))
        if bounded and obj < hist[-1]:
            # The active face of min(q_parent,q_current) may change. Backtrack
            # the convex-block proposal against the actual marginal likelihood.
            for step in (0.5, 0.25, 0.125, 0.0):
                qt = q + step * (q1 - q)
                norm1, jac1 = _normalised(tr, bins, qt, bounded)
                ll, stats1 = bp._estep(norm1, cand, src, allowed)
                obj = float(bp._obj(ll + jac1, cand, R))
                if obj >= hist[-1]:
                    q1 = qt
                    break
        if not np.isfinite(obj):
            raise FloatingPointError("nonfinite population likelihood")
        if obj < hist[-1]:
            break
        gain = obj - hist[-1]
        p, q, norm, stats = cand, q1, norm1, stats1
        hist.append(obj)
        if gain < 1e-6 * sum(len(t[1]) for t in tr):
            break
    return model_type(p, allowed, hist, q, support)
