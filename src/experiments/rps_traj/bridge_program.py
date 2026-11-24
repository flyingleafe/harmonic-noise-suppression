"""Compact hierarchical motor-speed bridge on a prescribed schedule.

A flight draws one latent style (idle, hover, fan-out and activity); on each
coarse cruise block a semi-Markov child is quiet, a positive drive or a
negative drive. On exiting a drive it can reverse or return to quiet. Each
drive is a two-stage Erlang-like duration. Thus the flight parent controls a
reusable ordered action sequence. Conditional on that state, first draw the
block endpoint then its midpoints recursively. The AR is on residuals about a
compact event-clock mean. This is a triangular change of variables with unit
Jacobian in mode space: the likelihood of every time sample enters the same
objective. No stored reference trajectory or independently fitted curves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.linalg import solve
from scipy.special import logsumexp

from tracking.rotors import MIXER

FS = 25
BLOCK = 32
LEVELS = 6  # endpoint then dyadic widths up to 32,16,8,4,2 samples
FLOOR = 0.02  # dequantisation floor rev/s
LINK = 0.20
SIGMA = 0.035
# Knot locations in seconds, relative to supplied motors-on / takeoff / landing.
# These are fixed multiresolution bases, not learned sample-level traces.
KNOTS = (
    np.r_[0.0, 0.04, 0.12, np.arange(0.24, 2.89, 0.16), 3.2, 3.6, 4.0],
    np.r_[-0.16, 0.0, np.arange(0.16, 1.93, 0.16), 2.2, 2.6, 3.2, 4.0],
    np.array(
        [
            -2.5,
            -1.6,
            -1.0,
            -0.6,
            -0.36,
            -0.2,
            -0.08,
            -0.04,
            0.0,
            0.04,
            0.12,
            0.24,
            0.40,
            0.64,
            0.9,
            1.2,
            1.5,
            1.8,
            2.1,
            2.4,
            2.7,
            3.0,
            3.4,
            4.0,
            4.8,
        ]
    ),
)
DIFF_KNOTS = (
    np.array([0.0, 0.4, 1.0, 1.4, 1.75, 2.15, 2.6, 3.2, 4.0]),
    np.array([-0.16, 0.0, 0.32, 0.72, 1.12, 1.52, 2.0, 2.8, 4.0]),
    np.array([-2.5, -1.0, -0.48, -0.12, 0.0, 0.08, 0.32, 0.8, 1.4, 2.0, 2.6, 3.2, 4.0, 4.8]),
)
STATES = 5  # quiet, positive stages 1/2, negative stages 1/2
SIGN = np.array([0.0, 1.0, 1.0, -1.0, -1.0])
ACTIVE = SIGN != 0


def _inverse(w):
    y = np.maximum(w, FLOOR) / LINK
    return LINK * (y + np.log(-np.expm1(-y)))


def _ramp(t, width):
    v = np.clip(t / width, 0.0, 1.0)
    return v * v * (3.0 - 2.0 * v)


def _hats(t, grid):
    # Interior triangular functions: zero beyond the support and at both edges.
    return np.maximum(
        0.0,
        np.minimum(
            (t[:, None] - grid[:-2]) / np.diff(grid[:-1]),
            (grid[2:] - t[:, None]) / np.diff(grid[1:]),
        ),
    )


@dataclass
class Design:
    base: tuple
    gates: np.ndarray
    phase: np.ndarray
    prev: np.ndarray
    right: np.ndarray
    frac: np.ndarray
    level: np.ndarray
    block: np.ndarray
    block_phase: np.ndarray
    order: np.ndarray
    block_starts: np.ndarray
    collective_shelf: bool = False
    transport: tuple[np.ndarray, np.ndarray] | None = None
    var_level: np.ndarray | None = None


def design(codes):
    c = np.asarray(codes, dtype=int)
    if c.ndim != 1 or len(c) < 1 or c.min() < 0 or c.max() > 3 or (np.diff(c) < 0).any():
        raise ValueError("expected ordered coarse phase codes")
    n = len(c)
    t = np.arange(n) / FS

    def age(code):
        idx = np.flatnonzero(c == code)
        return t - idx[0] / FS if idx.size else np.full(n, -1.0e6)

    on, to, land = (age(i) for i in (1, 2, 3))
    # Smooth anchor gates. Clock coefficients learn departures from the motion
    # between these levels; they are not fitted start/end speed samples.
    a = _ramp(on, 0.16) if np.any(c == 1) else (c >= 2).astype(float)
    b = _ramp(to, 1.36) if np.any(c == 2) else (c == 3).astype(float)
    if c[0] == 2:
        b[:] = 1.0
    descent = _ramp(land + 0.12, 2.8)
    idle = a * (1.0 - b) * (1.0 - descent)
    hover = b * (1.0 - descent)
    fan = np.sin(np.pi * _ramp(land + 0.4, 3.5))
    phase = np.select(
        [
            c == 0,
            (c == 1) & (on < 3.0),
            c == 1,
            (c == 2) & (to < 3.0),
            (c >= 2) & (land >= -0.6) & (land < 4.8),
            c == 3,
        ],
        [0, 1, 2, 3, 5, 0],
        default=4,
    ).astype(int)
    base = []
    for knots in (KNOTS, DIFF_KNOTS):
        hv = [_hats(age_, g) for age_, g in zip((on, to, land), knots, strict=True)]
        hv[0] *= (c == 1)[:, None]
        hv[1] *= (c >= 2)[:, None]
        base.append(np.column_stack(hv))
    prev = np.zeros(n, int)
    right = prev.copy()
    frac = np.zeros(n)
    lev = np.zeros(n, int)
    blk = np.zeros(n, int)
    bphase = [int(phase[0])]
    order = [0]
    starts = [0]
    boundaries = np.unique(np.r_[0, np.flatnonzero(np.diff(phase)) + 1, n - 1])
    begin = 0
    for end in boundaries[1:]:
        while begin < end:
            last = min(begin + BLOCK, end)
            bi = len(bphase)
            bphase.append(int(phase[last]))
            starts.append(len(order))
            prev[last] = begin
            right[last] = begin
            blk[last] = bi
            order.append(last)
            queue = [(begin, last)]
            for i, j in queue:
                if j - i <= 1:
                    continue
                m = (i + j) // 2
                prev[m] = i
                right[m] = j
                frac[m] = (m - i) / (j - i)
                lev[m] = min(5, 1 + int(np.floor(np.log2(BLOCK / (j - i)))))
                blk[m] = bi
                order.append(m)
                queue.extend(((i, m), (m, j)))
            begin = last
    # Exogenous gate defines the state of an interval by its interior rather
    # than the far endpoint when two phases meet.
    for i, start in enumerate(starts[1:], 1):
        j = order[start]
        bphase[i] = int(phase[(prev[j] + j) // 2])
    return Design(
        tuple(base),
        np.stack((idle, hover, fan), axis=1),
        phase,
        prev,
        right,
        frac,
        lev,
        blk,
        np.array(bphase),
        np.array(order),
        np.array(starts),
    )


def _neighbor(v, d):
    a = d.frac[:, None]
    y = (1 - a) * v[d.prev] + a * v[d.right]
    y[d.prev == np.arange(len(v))] = 0.0
    return y


def _matrix(d, dim, style, k):
    # Flight style contributes levels for all four coordinates, plus the
    # amplitude of the low-frequency differential landing excursion.
    gates = d.gates[:, :2] if dim == 0 and not d.collective_shelf else d.gates
    m = gates.shape[1]
    out = np.zeros((len(d.phase), d.base[int(dim != 0)].shape[1] + m * k))
    p = d.base[int(dim != 0)].shape[1]
    out[:, :p] = d.base[int(dim != 0)]
    out[:, p + style * m : p + (style + 1) * m] = gates
    return out


def _transition(params):
    q, h, r = params
    a = np.zeros((STATES, STATES))
    a[0] = [1 - q, q / 2, 0, q / 2, 0]
    for s, other in ((1, 3), (3, 1)):
        a[s, s] = h
        a[s, s + 1] = 1 - h
        a[s + 1, s + 1] = h
        a[s + 1, other] = (1 - h) * r
        a[s + 1, 0] = (1 - h) * (1 - r)
    return a


def _fb(em, cruise, params):
    from experiments.rps_traj.unified_grammar import _backward, _forward

    A = _transition(params)
    row, col = np.nonzero(A)
    ptr = np.r_[0, np.bincount(row, minlength=STATES).cumsum()]
    lp = np.log(A[row, col])
    ph = np.asarray(cruise, dtype=np.int64)
    # The generic kernel propagates across repeated noncruise blocks; only
    # state 0 is allowed there. Remove p(quiet->quiet) from their likelihood.
    init = np.full((2, STATES), -np.inf)
    init[:, 0] = 0.0
    mask = np.ones((2, STATES), bool)
    mask[0, 1:] = False
    ids = np.arange(STATES)
    la = np.empty_like(em)
    norm = np.empty(len(em))
    scaled = em.copy()
    repeat = np.flatnonzero((ph[1:] == 0) & (ph[:-1] == 0)) + 1
    scaled[repeat, 0] -= np.log(A[0, 0])
    ll = _forward(scaled, ph, ids, ptr, col, lp, init, mask, la, norm, True)
    occ, edges = np.zeros_like(em), np.zeros(len(col))
    _backward(scaled, ph, ids, ptr, col, lp, mask, la, norm, occ, edges, np.zeros(STATES))
    counts = np.zeros_like(A)
    counts[row, col] = edges
    counts[0, 0] -= len(repeat)
    return float(ll), occ, np.maximum(counts, 0.0)


def _var(p, d, j):
    return np.maximum(
        1e-5,
        p["basevar"][d.level if d.var_level is None else d.var_level]
        * p["phasevar"][d.phase]
        * np.where((d.phase == 4)[:, None], p["activity"][j], 1.0),
    )


def _make_track(w, c, k, d=None, encode=None):
    if d is None:
        d = design(c)
    z = ((_inverse(w) - _inverse(FLOOR)).T @ MIXER) / 4 if encode is None else encode(w)
    x = [[_matrix(d, h, j, k) for h in range(4)] for j in range(k)]
    neighbors = [[_neighbor(v, d) for v in xx] for xx in x]
    return d, z, _neighbor(z, d), x, neighbors


def _residual(track, p, j):
    d, y, ny, x, nx = track
    mean = np.column_stack([x[j][h] @ p["beta"][h] for h in range(4)])
    if d.transport is None:
        neigh = ny - _neighbor(mean, d)
    else:
        neigh = ny - np.column_stack([nx[j][h] @ p["beta"][h] for h in range(4)])
    return y - mean - p["rho"][d.level] * neigh, mean, neigh


def _estep(tracks, p, source, allowed):
    es = []
    ll = 0.0
    k = len(p["mix"])
    for tr, src in zip(tracks, source, strict=True):
        d = tr[0]
        lg = []
        gam = []
        cts = []
        for j in range(k):
            if src != allowed[j]:
                lg.append(-np.inf)
                gam.append(None)
                cts.append(None)
                continue
            err, _, _ = _residual(tr, p, j)
            v = _var(p, d, j)
            root = ((d.level == 0) & (d.phase == 4))[:, None]
            logem = []
            for sign in (0, 1, -1):
                v1 = v if not sign else v * np.where((d.phase == 4)[:, None], p["boost"], 1.0)
                e = err if not sign else err - sign * root * p["pulse"]
                step_loglik = -0.5 * np.sum(np.log(2 * np.pi * v1) + e * e / v1, axis=1)
                logem.append(np.add.reduceat(step_loglik[d.order], d.block_starts))
            tab = np.column_stack([logem[s] for s in (0, 1, 1, 2, 2)])
            vj, g, c = _fb(tab, d.block_phase == 4, p["trans"][j])
            lg.append(np.log(p["mix"][j]) + vj)
            gam.append(g)
            cts.append(c)
        norm = float(cast(float, logsumexp(lg)))
        ll += norm
        es.append((np.exp(np.asarray(lg) - norm), gam, cts))
    return float(ll), es


def _initial(tracks, source, allowed):
    j = len(allowed)
    d = tracks[0][0]
    p = {
        "mix": np.zeros(j),
        "rho": np.tile([0.35, 0.75, 0.85, 0.94, 0.98, 0.99], (4, 1)).T.copy(),
        "basevar": np.tile([3.0, 2.0, 1.7, 1.3, 1.0, 0.6], (4, 1)).T ** 2,
        "phasevar": np.tile([0.03, 1.0, 0.08, 1.8, 1.0, 1.8], (4, 1)).T ** 2,
        "activity": np.ones((j, 4)),
        "boost": np.array([2.0, 2.0, 2.0, 4.0]),
        "pulse": np.array([0.2, 0.0, 0.0, 1.0]),
        "trans": np.tile([0.15, 0.4, 0.5], (j, 1)),
        "beta": [np.zeros(tracks[0][3][0][h].shape[1]) for h in range(4)],
    }
    # Data-informed warm start only. Subsequent ECM updates include every level and all source records.
    for src in np.unique(source):
        idx = np.flatnonzero(source == src)
        choices = np.flatnonzero(allowed == src)
        p["mix"][choices] = 1 / len(choices)
        order = sorted(
            idx, key=lambda i: np.log(np.std(tracks[i][1][tracks[i][0].phase == 4, -1]) + 0.01)
        )
        for a, sub in zip(choices, np.array_split(order, len(choices)), strict=True):
            for h in range(4):
                groups = []
                for i in sub:
                    d, y = tracks[i][:2]
                    g = []
                    for phase in (1, 4):
                        mask = (
                            (d.gates[:, 0] > 0.95) & (d.phase < 3) if phase == 1 else d.phase == 4
                        )
                        g.append(np.median(y[mask, h]) if mask.any() else np.median(y[:, h]))
                    groups.append(g)
                cols = 2 if h == 0 else 3
                loc = tracks[0][0].base[int(h != 0)].shape[1] + cols * a
                p["beta"][h][loc : loc + 2] = np.mean(groups, axis=0)
            mad = np.mean(
                [np.var(tracks[i][1][tracks[i][0].phase == 4], axis=0) for i in sub], axis=0
            )
            p["activity"][a] = np.clip(mad / 5.0, 0.05, 100.0)
    return p


def _penalties(p, d):
    result = []
    for h, beta in enumerate(p["beta"]):
        penalty = np.eye(len(beta)) / 200.0**2
        start = 0
        for grid in KNOTS if h == 0 else DIFF_KNOTS:
            width = len(grid) - 2
            eye = np.eye(width)
            diff = np.diff(np.vstack((np.zeros(width), eye, np.zeros(width))), n=2, axis=0)
            penalty[start : start + width, start : start + width] += (diff.T @ diff) / 15.0**2
            start += width
        result.append(penalty)
    return result


def _obj(ll, p, penalty):
    return ll - sum(0.5 * b @ R @ b for b, R in zip(p["beta"], penalty, strict=True))


def _joint_mstep(tracks, p, stats, allowed, penalty):
    import copy

    p = copy.deepcopy(p)
    k = len(allowed)
    wt = []
    H = [r.copy() for r in penalty]
    u = [np.zeros(len(b)) for b in p["beta"]]
    visits = np.zeros(k)
    counts = np.zeros((k, STATES, STATES))
    for tr, (resp, gamma, xi) in zip(tracks, stats, strict=True):
        d, y, ny, x, nx = tr
        local = []
        visits += resp
        for j, w in enumerate(resp):
            if w < 1e-14:
                continue
            counts[j] += w * xi[j]
            post = gamma[j][d.block]
            active = (post[:, 1:]).sum(axis=1)
            signed = post @ SIGN
            inv = 1 / _var(p, d, j)
            inv_on = inv / np.where((d.phase == 4)[:, None], p["boost"], 1.0)
            rgate = ((d.level == 0) & (d.phase == 4))[:, None]
            prec = w * ((1 - active[:, None]) * inv + active[:, None] * inv_on)
            shift = w * signed[:, None] * inv_on * rgate
            local.append((j, w, post, prec, shift, inv_on, rgate))
            for h in range(4):
                rho = p["rho"][d.level, h]
                A = x[j][h] - rho[:, None] * nx[j][h]
                wy = prec[:, h] * (y[:, h] - rho * ny[:, h]) - shift[:, h] * p["pulse"][h]
                H[h] += A.T @ (prec[:, h, None] * A)
                u[h] += A.T @ wy
        wt.append(local)
    for h in range(4):
        p["beta"][h] = solve(H[h], u[h], assume_a="pos", check_finite=False)
    for src in np.unique(allowed):
        ix = np.flatnonzero(allowed == src)
        p["mix"][ix] = np.maximum(visits[ix], 1e-8)
        p["mix"][ix] /= p["mix"][ix].sum()

    def prop(num, den, default):
        return np.clip(num / den, 0.002, 0.998) if den > 1e-8 else default

    for j, c in enumerate(counts):
        q, h, r = p["trans"][j]
        cdrive = c[1:, 1:].trace()
        p["trans"][j] = [
            prop(c[0, 1] + c[0, 3], c[0].sum(), q),
            prop(cdrive, c[1:].sum(), h),
            prop(c[2, 3] + c[4, 1], c[2, 3] + c[4, 1] + c[2, 0] + c[4, 0], r),
        ]

    nlevels = len(p["rho"])
    xx = np.zeros((nlevels, 4))
    xy = xx.copy()
    for tr, local in zip(tracks, wt, strict=True):
        d, y, _, _, _ = tr
        for j, _, _, prec, shift, _, _ in local:
            _, mean, ns = _residual(tr, p, j)
            z = y - mean
            for h in range(4):
                xx[:, h] += np.bincount(
                    d.level, weights=prec[:, h] * ns[:, h] ** 2, minlength=nlevels
                )
                xy[:, h] += np.bincount(
                    d.level,
                    weights=(prec[:, h] * z[:, h] - shift[:, h] * p["pulse"][h]) * ns[:, h],
                    minlength=nlevels,
                )
    p["rho"] = np.clip(np.divide(xy, xx, out=p["rho"].copy(), where=xx > 1e-8), -0.95, 1.25)
    p["rho"][0] = np.clip(p["rho"][0], -0.98, 0.98)
    nvar = len(p["basevar"])
    if nlevels > nvar:
        # Ground feedback has one collective and one torque-contrast pole.
        # Its innovation envelope is still learned jointly with all stages.
        denom = xx[-1, 1:].sum()
        if denom > 1e-8:
            p["rho"][-1, 1:] = xy[-1, 1:].sum() / denom
        p["rho"][-1] = np.clip(p["rho"][-1], -0.95, 0.95)
    num = np.zeros(4)
    den = np.zeros(4)
    for tr, local in zip(tracks, wt, strict=True):
        for j, w, gamma, _, shift, inv_on, rgate in local:
            err, _, _ = _residual(tr, p, j)
            num += (shift * err).sum(axis=0)
            den += (w * gamma[:, 1:].sum(axis=1)[:, None] * inv_on * rgate).sum(axis=0)
    p["pulse"] = np.divide(num, den, out=p["pulse"].copy(), where=den > 1e-8)
    # Variance ECM in the four-way cell table, not one parameter per cell.
    np_ = len(p["phasevar"])
    sse = np.zeros((nvar, np_, k, 2, 4))
    mass = np.zeros((nvar, np_, k, 2, 1))
    for tr, local in zip(tracks, wt, strict=True):
        d = tr[0]
        idx = (d.level if d.var_level is None else d.var_level) * np_ + d.phase
        for j, w, gamma, _, _, _, rgate in local:
            err, _, _ = _residual(tr, p, j)
            for sign, states in ((0, [0]), (1, [1, 2]), (-1, [3, 4])):
                g = w * gamma[:, states].sum(axis=1)
                e = (err - sign * rgate * p["pulse"]) ** 2
                a = int(sign != 0)
                mass[:, :, j, a, 0] += np.bincount(idx, weights=g, minlength=nvar * np_).reshape(
                    nvar, np_
                )
                for h in range(4):
                    sse[:, :, j, a, h] += np.bincount(
                        idx, weights=g * e[:, h], minlength=nvar * np_
                    ).reshape(nvar, np_)

    def variances():
        arr = (
            p["basevar"][:, None, None, None, :]
            * p["phasevar"][None, :, None, None, :]
            * np.ones((1, 1, k, 2, 1))
        )
        arr[:, 4, :, :, :] *= p["activity"][None, :, None, :]
        arr[:, 4, :, 1, :] *= p["boost"]
        return arr

    def ratio(S, N):
        return np.divide(S, N, out=np.ones_like(S), where=N > 1e-9)

    p["basevar"] = np.clip(
        p["basevar"] * ratio((sse / variances()).sum(axis=(1, 2, 3)), mass.sum(axis=(1, 2, 3))),
        0.001,
        10000.0,
    )
    factor = ratio((sse / variances()).sum(axis=(0, 2, 3)), mass.sum(axis=(0, 2, 3)))
    p["phasevar"] = np.clip(p["phasevar"] * factor, 0.0001, 10000.0)
    p["phasevar"][4] = 1.0
    factor = ratio((sse / variances())[:, 4].sum(axis=(0, 2)), mass[:, 4].sum(axis=(0, 2)))
    p["activity"] = np.clip(p["activity"] * factor, 0.01, 1000.0)
    p["activity"][0] = 1.0
    factor = ratio(
        (sse / variances())[:, 4, :, 1].sum(axis=(0, 1)), mass[:, 4, :, 1].sum(axis=(0, 1))
    )
    p["boost"] = np.clip(p["boost"] * factor, 0.1, 100.0)
    return p


class BridgeModel:
    def __init__(self, p, allowed, history):
        self.p, self.allowed, self.history = p, np.asarray(allowed, dtype=bool), list(history)

    @property
    def n_params(self):
        p = self.p
        return int(
            sum(len(b) for b in p["beta"])
            + p["rho"].size
            + p["basevar"].size
            + p["phasevar"].size
            - 4
            + p["activity"].size
            - 4
            + 8
            + p["trans"].size
            + len(p["mix"])
            - len(np.unique(self.allowed))
        )

    def sample(self, schedule, n, rng):
        return self._draw(design(schedule), n, rng)

    def _draw(self, d, n, rng, readout_scale=1.0, decode=None):
        p = self.p
        pi = p["mix"] * self.allowed
        styles = rng.choice(len(pi), size=n, p=pi / pi.sum())
        out = np.empty((n, len(d.phase), 4))
        for j in np.unique(styles):
            ids = np.flatnonzero(styles == j)
            mean = np.column_stack([_matrix(d, h, j, len(pi)) @ p["beta"][h] for h in range(4)])
            s = np.zeros((len(ids), len(d.block_phase)), int)
            A = _transition(p["trans"][j])
            for b in range(1, len(d.block_phase)):
                if d.block_phase[b] == 4 and d.block_phase[b - 1] == 4:
                    s[:, b] = (
                        (rng.random((len(ids), 1)) > A[s[:, b - 1]].cumsum(axis=1))
                        .sum(axis=1)
                        .clip(0, STATES - 1)
                    )
            states = s[:, d.block]
            root = (d.level == 0) & (d.phase == 4)
            var = _var(p, d, j)[None, :, :] * np.where(
                (ACTIVE[states] & (d.phase == 4))[:, :, None], p["boost"], 1.0
            )
            e = (
                rng.standard_normal(var.shape) * np.sqrt(var)
                + SIGN[states][:, :, None] * root[None, :, None] * p["pulse"]
            )
            for t in d.order[d.prev[d.order] != d.order]:
                u = d.frac[t]
                if d.transport is None:
                    parent = (1 - u) * e[:, d.prev[t]] + u * e[:, d.right[t]]
                else:
                    left, right = d.transport
                    parent = ((1 - u) * left[t]) * e[:, d.prev[t]] + (u * right[t]) * e[
                        :, d.right[t]
                    ]
                e[:, t] += p["rho"][d.level[t]] * parent
            out[ids] = e * readout_scale + mean
        if decode is not None:
            return decode(out)
        speed = LINK * np.logaddexp(0.0, (out @ MIXER.T + _inverse(FLOOR)) / LINK)
        return np.where(speed <= FLOOR, 0.0, speed).transpose(0, 2, 1)

    def to_dict(self):
        params = {
            k: ([a.tolist() for a in v] if k == "beta" else v.tolist()) for k, v in self.p.items()
        }
        return {
            "family": "bridge_program",
            "params": params,
            "allowed_measured": self.allowed.tolist(),
            "n_params": self.n_params,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, d):
        p = {
            k: ([np.asarray(a) for a in v] if k == "beta" else np.asarray(v))
            for k, v in d["params"].items()
        }
        return cls(p, d["allowed_measured"], d["history"])


def fit(tracks, *, measured=None, max_iter=80):
    source = np.ones(len(tracks), bool) if measured is None else np.asarray(measured, bool)
    n_aux = int(np.any(~source))
    n_meas = min(3 - n_aux, int(source.sum()))
    if n_meas == 0 or len(source) != len(tracks):
        raise ValueError("need measured training records and one source tag per track")
    allowed = np.array([True] * n_meas + [False] * n_aux)
    tr = [_make_track(x, c, len(allowed)) for x, c in tracks]
    p = _initial(tr, source, allowed)
    p["activity"][0] = 1.0
    penalty = _penalties(p, tr[0][0])
    # Initialize the shared event coefficients jointly across all training
    # records before fitting stochastic bridge gains and hidden paths.
    for h in range(4):
        H = penalty[h].copy()
        u = np.zeros(len(p["beta"][h]))
        for i, t in enumerate(tr):
            j = min(
                np.flatnonzero(allowed == source[i]),
                key=lambda a: np.mean((_residual(t, p, a)[1] - t[1]) ** 2),
            )
            A = t[3][j][h]
            H += A.T @ A
            u += A.T @ t[1][:, h]
        p["beta"][h] = solve(H, u, assume_a="pos", check_finite=False)
    history = []
    ll, stats = _estep(tr, p, source, allowed)
    history.append(_obj(ll, p, penalty))
    for _ in range(max_iter):
        cand = _joint_mstep(tr, p, stats, allowed, penalty)
        ll, next_stats = _estep(tr, cand, source, allowed)
        obj = _obj(ll, cand, penalty)
        if not np.isfinite(obj):
            raise FloatingPointError("nonfinite bridge likelihood")
        if obj < history[-1]:
            break
        gain = obj - history[-1]
        p, stats = cand, next_stats
        history.append(obj)
        if gain < 1e-6 * sum(len(t[0].phase) for t in tr):
            break
    return BridgeModel(p, allowed, history)
