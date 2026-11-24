"""Lifting-coefficient control law: one exact density of every motor-speed sample.

Speeds are moved to the old bridge's floored inverse-softplus rotor scale and
mixed to the four 25 Hz control modes (collective, roll, pitch, yaw), exactly
as :func:`experiments.rps_traj.bridge_program._make_track`.  The shared
mean-preserving lifting of :mod:`experiments.rps_traj.lifting_basis` maps the
mode trajectory of one flight to as many coefficient rows (unit Jacobian on
the time axis): one exact arithmetic phase mean (DC) per contiguous coarse
phase plus details.  The density of the flight is the product over rows of
four-variate Gaussians, so every sample enters ONE likelihood:

* mean: ``design.mean[m] @ beta[m]`` (transformed fixed designs);
* detail SD per mode: log-linear in dyadic level between four knots at log2
  support 1, 3, 5, 7; past 128 samples it declines as ``sqrt(128/support)``
  (stationary long-scale differences);
* one shared six-parameter correlation across modes (unit-norm Cholesky rows);
* DC: covariance of a finite mean of the stationary law, ``Var(detail at
  2S) / 2`` for a phase of ``S`` samples, plus diagonal equilibrium scatter
  (collective and differential scales, fitted only with several flights);
* one per-flight activity: two equiprobable classes scaling all fluctuation
  SDs by ``exp(+a)`` and ``exp(-a)``;
* cruise blocks: the 5-state quiet / positive / negative two-stage path of
  :mod:`bridge_program` (q, h, reversal), summed by exact forward-backward;
  driven states scale detail SDs by one active factor and shift the yaw
  coefficient of the predeclared ``design.drive`` rows by a signed amplitude;
* every noncruise noise group present in training: one collective and one
  contrast log SD multiplier relative to the cruise law.

Latent versus raw.  The latent shaft speed of a commanded-off (code 0) run is
exactly 0: its lifting rows (each tree covers one contiguous code run) carry
no fitted density and are generated as exact zeros.  The raw off telemetry is
still observed, through a FIXED sensor law: each reported rotor rev/s is
Normal about 0 with SD ``phases.MOVING_RPS`` (1 rev/s, the existing moving
threshold), independent of every fitted value and not counted as learned.
So the reported whole-record objective is: raw off readings under that law
(rev/s scale) plus the exact density of all running rows in transformed mode
coordinates (inverse-softplus scale; the constant encoding Jacobian is
omitted).  Floor convention: running speeds at or below
``bridge_program.FLOOR`` are encoded as the exact rotor value 0 (a point
value, not a censored tail); a generated rotor value at or below 0 reads out
as a stopped rotor, so every generated speed is nonnegative.

Fit: regularized MAP by generalized EM, summing over activity classes and
hidden paths.  Mean coefficients carry the Structure's fixed roughness penalty
(weak level ridge); scales carry weak Gaussian priors and box bounds.  Each
sweep exactly maximizes the expected complete objective over the mean and
drive amplitude (dense 4x4 precisions), improves it over the
scale/correlation vector with bounded L-BFGS, and updates the hazards in
closed form; a candidate is accepted only if the marginal objective does not
fall.  Learned scalars (``n_params``) = Structure timing scalars + mean
coefficients + 16 SD knots + 6 correlations + (2 scatter, 1 activity with
several flights) + (3 hazards, 1 active scale with repeated cruise blocks) +
(1 signed amplitude with drive rows) + 2 per supported noncruise group.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm as gaussian

from experiments.rps_traj import bridge_program as bp
from experiments.rps_traj.lifting_basis import Structure
from experiments.rps_traj.phases import MOVING_RPS
from tracking.rotors import MIXER

FAMILY = "lifting_control"
KNOTS = np.array([1.0, 3.0, 5.0, 7.0])  # log2 support of the four SD knots
LONG = 128.0  # beyond this support the SD falls as sqrt(LONG / support)
YAW = 3
CLASS = np.array([0, 1, 1, 1])  # collective / contrast share of a 2-vector
TRIL = np.tril_indices(4, -1)
LOG2PI = np.log(2.0 * np.pi)
RIDGE = 1e-6  # prior precision of the signed drive amplitude
ZERO = bp._inverse(bp.FLOOR)


def encode(w):
    """``(4, n)`` speeds to ``(n, 4)`` floored inverse-softplus modes."""
    return ((bp._inverse(np.asarray(w, dtype=float)) - ZERO).T @ MIXER) / 4


def readout(z):
    """``(..., n, 4)`` modes to ``(..., 4, n)`` speeds; rotor values <= 0 stop."""
    u = z @ MIXER.T
    speed = bp.LINK * np.logaddexp(0.0, (u + ZERO) / bp.LINK)
    return np.swapaxes(np.where(u <= 0.0, 0.0, speed), -1, -2)


def _logsd(knots, level, support):
    """Log detail SD ``(n, 4)`` at log2 support ``level`` and ``support`` samples."""
    x = np.clip(np.asarray(level, dtype=float), KNOTS[0], KNOTS[-1])
    out = np.stack([np.interp(x, KNOTS, knots[:, m]) for m in range(4)], axis=1)
    return out + 0.5 * np.log(np.minimum(1.0, LONG / np.maximum(support, 1.0)))[:, None]


def _dc_logsd(knots, support):
    """Log SD of a finite mean of ``support`` samples: Var(detail at 2S) / 2."""
    s2 = 2.0 * np.asarray(support, dtype=float)
    return _logsd(knots, np.log2(s2), s2) - 0.5 * np.log(2.0)


@dataclass(frozen=True)
class _Rows:
    """Data-independent row metadata of one schedule's lifting."""

    X: tuple
    stopped: np.ndarray
    det: np.ndarray
    dc: np.ndarray
    level: np.ndarray
    support: np.ndarray
    group: np.ndarray
    block: np.ndarray
    drive: np.ndarray
    dc_support: np.ndarray
    dc_group: np.ndarray
    dc_code: np.ndarray
    cruise: np.ndarray


def _rows(d):
    """Rows of a commanded-off (code 0) run hold the latent stopped shaft (exact
    0): each lifting tree covers one contiguous coarse-code run, so they carry
    no fitted density (raw off readings use the fixed sensor law) and are drawn
    as zeros."""
    dc = np.asarray(d.dc, bool)
    drive = np.asarray(d.drive, bool)
    if (drive & dc).any():
        raise ValueError("a phase mean cannot carry the signed drive")
    cruise = np.asarray(d.cruise, bool)
    block = np.asarray(d.block, int)
    code = np.asarray(d.code, int)
    stopped = code == 0
    det = np.flatnonzero(~stopped & ~dc)
    dcs = np.flatnonzero(~stopped & dc)
    support = np.asarray(d.support, float)
    group = np.asarray(d.group, int)
    return _Rows(
        X=tuple(np.asarray(x, float).reshape(len(dc), -1) for x in d.mean),
        stopped=stopped,
        det=det,
        dc=dcs,
        level=np.asarray(d.level, float)[det],
        support=support[det],
        group=group[det],
        block=block[det],
        drive=(drive & cruise[block])[det].astype(float),
        dc_support=support[dcs],
        dc_group=group[dcs],
        dc_code=code[dcs],
        cruise=cruise,
    )


@dataclass(frozen=True)
class _Law:
    C: np.ndarray
    Cinv: np.ndarray
    logdetC: float
    L: np.ndarray
    gm: np.ndarray  # (n_groups, 4) log SD multipliers
    tau2: np.ndarray  # (4,) DC scatter variances
    shift: np.ndarray  # (J,) activity log multipliers
    act: float  # active-state log multiplier


def _law(p, flags):
    L = np.eye(4)
    L[TRIL] = p["corr"]
    L /= np.linalg.norm(L, axis=1, keepdims=True)
    Linv = np.linalg.inv(L)
    tau2 = np.exp(2.0 * p["scatter"][CLASS]) if flags["scatter"] else np.zeros(4)
    a = float(p["style"])
    return _Law(
        C=L @ L.T,
        Cinv=Linv.T @ Linv,
        logdetC=2.0 * float(np.log(np.diag(L)).sum()),
        L=L,
        gm=p["group"][:, CLASS],
        tau2=tau2,
        shift=np.array([a, -a]) if flags["style"] else np.zeros(1),
        act=float(p["active"]) if flags["cruise"] else 0.0,
    )


def _gauss(r, logsd, law):
    u = r * np.exp(-logsd)
    quad = np.einsum("...a,ab,...b->...", u, law.Cinv, u)
    return -0.5 * (4 * LOG2PI + law.logdetC + 2.0 * logsd.sum(-1) + quad)


def _dc_cov(logsd, law):
    sd = np.exp(logsd)
    return sd[..., :, None] * law.C * sd[..., None, :] + np.diag(law.tau2)


def _dc_gauss(r, logsd, law):
    S = _dc_cov(logsd, law)
    _, ld = np.linalg.slogdet(S)
    quad = np.einsum("...a,...a->...", r, np.linalg.solve(S, r[..., None])[..., 0])
    return -0.5 * (4 * LOG2PI + ld + quad)


def _mean(X, beta):
    return np.column_stack([x @ b for x, b in zip(X, beta, strict=True)])


def _layout(flags):
    lay = [("knots", 16, -9.0, 7.0), ("corr", 6, -20.0, 20.0)]
    if flags["scatter"]:
        lay.append(("scatter", 2, -9.0, 6.0))
    if flags["style"]:
        lay.append(("style", 1, 0.0, 3.0))
    if flags["cruise"]:
        lay.append(("active", 1, -3.0, 4.0))
    n = int(np.sum(flags["supported"][1:]))
    if n:
        lay.append(("group", 2 * n, -7.0, 7.0))
    return lay


def _pack(p, flags):
    out = []
    for name, _, _, _ in _layout(flags):
        v = p[name][np.asarray(flags["supported"])][1:] if name == "group" else p[name]
        out.append(np.ravel(v))
    return np.concatenate(out)


def _unpack(vec, p, flags):
    q = dict(p)
    pos = 0
    for name, size, _, _ in _layout(flags):
        v = np.asarray(vec[pos : pos + size], dtype=float)
        pos += size
        if name == "group":
            g = np.zeros_like(p["group"])
            g[np.flatnonzero(flags["supported"])[1:]] = v.reshape(-1, 2)
            q[name] = g
        elif name in ("style", "active"):
            q[name] = float(v[0])
        else:
            q[name] = v.reshape(np.shape(p[name]))
    return q


def _theta_prior(p, flags):
    # Weak MAP terms that only keep weakly identified scales finite.
    lp = -0.5 * float(np.sum((p["corr"] / 3.0) ** 2) + np.sum((p["group"] / 3.0) ** 2))
    if flags["style"]:
        lp -= 0.5 * p["style"] ** 2
    if flags["cruise"]:
        lp -= 0.5 * (p["active"] / 3.0) ** 2
    return lp


def _logprior(p, flags):
    # Mean coefficients: the Structure's fixed roughness penalty (weak ridge on
    # levels and gates, so scarce phase means are not pulled toward zero).
    rough = sum(float(b @ R @ b) for b, R in zip(p["beta"], flags["penalty"], strict=True))
    return _theta_prior(p, flags) - 0.5 * (rough + RIDGE * p["amp"] ** 2)


@dataclass
class _Track:
    c: np.ndarray
    rows: _Rows
    key: np.ndarray = field(init=False)  # global detail-cell index of each modelled detail row
    off_ll: float  # fixed sensor log density of the raw commanded-off readings


def _residual(t, p):
    return t.c - _mean(t.rows.X, p["beta"])


def _onehot(nb):
    o = np.zeros((nb, bp.STATES))
    o[:, 0] = 1.0
    return o


def _estep(p, flags, data):
    """Marginal MAP objective and per-flight posteriors (classes, paths)."""
    law = _law(p, flags)
    J = len(law.shift)
    total = 0.0
    posts = []
    for t in data:
        rw = t.rows
        r = _residual(t, p)
        rd, rc = r[rw.det], r[rw.dc]
        base = _logsd(p["knots"], rw.level, rw.support) + law.gm[rw.group]
        dcb = _dc_logsd(p["knots"], rw.dc_support) + law.gm[rw.dc_group]
        shift = np.zeros_like(rd)
        shift[:, YAW] = p["amp"] * rw.drive
        nb = len(rw.cruise)
        lj, occ, cnt = np.empty(J), [], []
        for j, s in enumerate(law.shift):
            em = np.zeros((nb, bp.STATES))
            em[:, 0] = np.bincount(rw.block, _gauss(rd, base + s, law), minlength=nb)
            if flags["cruise"]:
                for col, sign in ((1, 1.0), (3, -1.0)):
                    e = _gauss(rd - sign * shift, base + s + law.act, law)
                    em[:, col] = em[:, col + 1] = np.bincount(rw.block, e, minlength=nb)
                ll, o, c = bp._fb(em, rw.cruise, p["trans"])
            else:
                ll, o, c = float(em[:, 0].sum()), _onehot(nb), np.zeros((bp.STATES,) * 2)
            lj[j] = ll + float(_dc_gauss(rc, dcb + s, law).sum()) - np.log(J)
            occ.append(o)
            cnt.append(c)
        norm = float(cast(float, logsumexp(lj)))
        total += norm + t.off_ll
        posts.append((np.exp(lj - norm), occ, cnt))
    return total + _logprior(p, flags), posts


def _row_weights(rw, resp, occ, law):
    """Expected precision factors of detail rows: all, signed and driven."""
    omega = np.zeros(len(rw.det))
    psi = np.zeros(len(rw.det))
    chi = np.zeros(len(rw.det))
    e2a = np.exp(-2.0 * law.act)
    for j, s in enumerate(law.shift):
        k = resp[j] * np.exp(-2.0 * s)
        o = occ[j][rw.block]
        gp, gn = o[:, 1] + o[:, 2], o[:, 3] + o[:, 4]
        omega += k * (o[:, 0] + e2a * (gp + gn))
        psi += k * e2a * (gp - gn)
        chi += k * e2a * (gp + gn)
    return omega, psi * rw.drive, chi * rw.drive


def _solve_mean(p, flags, data, posts):
    """Exact conditional maximizer of the expected objective in (beta, amp)."""
    law = _law(p, flags)
    sizes = [len(b) for b in p["beta"]]
    off = np.r_[0, np.cumsum(sizes)]
    n_par = int(off[-1]) + int(flags["drive"])
    H = RIDGE * np.eye(n_par)
    for m, R in enumerate(flags["penalty"]):
        H[off[m] : off[m + 1], off[m] : off[m + 1]] = R
    g = np.zeros(n_par)
    for t, (resp, occ, _) in zip(data, posts, strict=True):
        rw = t.rows
        inv = np.exp(-(_logsd(p["knots"], rw.level, rw.support) + law.gm[rw.group]))
        Pd = inv[:, :, None] * law.Cinv * inv[:, None, :]
        omega, psi, chi = _row_weights(rw, resp, occ, law)
        W = np.zeros((len(t.c), 4, 4))
        W[rw.det] = omega[:, None, None] * Pd
        dcb = _dc_logsd(p["knots"], rw.dc_support) + law.gm[rw.dc_group]
        for j, s in enumerate(law.shift):
            W[rw.dc] += resp[j] * np.linalg.inv(_dc_cov(dcb + s, law))
        Wc = np.einsum("nab,nb->na", W, t.c)
        for m in range(4):
            sm = slice(off[m], off[m + 1])
            for m2 in range(m, 4):
                H[sm, off[m2] : off[m2 + 1]] += rw.X[m].T @ (W[:, m, m2, None] * rw.X[m2])
            g[sm] += rw.X[m].T @ Wc[:, m]
        if flags["drive"]:
            pa = np.zeros((len(t.c), 4))
            pa[rw.det] = psi[:, None] * Pd[:, YAW, :]
            for m in range(4):
                H[off[m] : off[m + 1], -1] += rw.X[m].T @ pa[:, m]
            H[-1, -1] += float(np.sum(chi * Pd[:, YAW, YAW]))
            g[-1] += float(np.sum(pa * t.c))
    H = np.triu(H) + np.triu(H, 1).T
    sol = solve(H, g, assume_a="pos", check_finite=False)
    q = dict(p)
    q["beta"] = [sol[off[m] : off[m + 1]].copy() for m in range(4)]
    if flags["drive"]:
        q["amp"] = float(sol[-1])
    return q


@dataclass(frozen=True)
class _Stats:
    """Expected sufficient statistics of the complete-data scale objective."""

    level: np.ndarray
    support: np.ndarray
    group: np.ndarray
    N: np.ndarray  # (K, J, 2) quiet / driven weights per detail cell
    M: np.ndarray  # (K, J, 2, 4, 4) weighted residual outer products
    dc_r: np.ndarray
    dc_w: np.ndarray
    dc_support: np.ndarray
    dc_group: np.ndarray


def _stats(p, flags, data, posts, cells: tuple[np.ndarray, np.ndarray, np.ndarray]):
    law = _law(p, flags)
    J = len(law.shift)
    K = len(cells[0])
    N = np.zeros((K, J, 2))
    M = np.zeros((K, J, 2, 4, 4))
    dc_r, dc_w, dc_s, dc_g = [], [], [], []
    for t, (resp, occ, _) in zip(data, posts, strict=True):
        rw = t.rows
        r = _residual(t, p)
        rd = r[rw.det]
        shift = np.zeros_like(rd)
        shift[:, YAW] = p["amp"] * rw.drive
        rp, rn = rd - shift, rd + shift
        outer = rd[:, :, None] * rd[:, None, :]
        op = rp[:, :, None] * rp[:, None, :]
        on = rn[:, :, None] * rn[:, None, :]
        for j in range(J):
            o = occ[j][rw.block] * resp[j]
            wq, wp, wn = o[:, 0], o[:, 1] + o[:, 2], o[:, 3] + o[:, 4]
            N[:, j, 0] += np.bincount(t.key, wq, minlength=K)
            N[:, j, 1] += np.bincount(t.key, wp + wn, minlength=K)
            for a in range(4):
                for b in range(a, 4):
                    M[:, j, 0, a, b] += np.bincount(t.key, wq * outer[:, a, b], minlength=K)
                    v = wp * op[:, a, b] + wn * on[:, a, b]
                    M[:, j, 1, a, b] += np.bincount(t.key, v, minlength=K)
        dc_r.append(r[rw.dc])
        dc_w.append(np.tile(resp, (len(rw.dc), 1)))
        dc_s.append(rw.dc_support)
        dc_g.append(rw.dc_group)
    iu = np.triu_indices(4, 1)
    M[..., iu[1], iu[0]] = M[..., iu[0], iu[1]]
    return _Stats(
        *cells,
        N,
        M,
        np.concatenate(dc_r).reshape(-1, 4),
        np.concatenate(dc_w).reshape(-1, J),
        np.concatenate(dc_s),
        np.concatenate(dc_g).astype(int),
    )


def _q_scale(vec, p, flags, st):
    """Negative expected complete objective in the scale/correlation vector."""
    q = _unpack(vec, p, flags)
    law = _law(q, flags)
    base = _logsd(q["knots"], st.level, st.support) + law.gm[st.group]
    out = 0.0
    for j, s in enumerate(law.shift):
        for x, extra in ((0, 0.0), (1, law.act)):
            n = st.N[:, j, x]
            if not n.any():
                continue
            ls = base + s + extra
            inv = np.exp(-ls)
            scaled = st.M[:, j, x] * inv[:, :, None] * inv[:, None, :]
            quad = np.sum(scaled * law.Cinv, axis=(1, 2))
            out -= 0.5 * np.sum(n * (4 * LOG2PI + law.logdetC + 2.0 * ls.sum(1)) + quad)
    dcb = _dc_logsd(q["knots"], st.dc_support) + law.gm[st.dc_group]
    for j, s in enumerate(law.shift):
        out += float(np.sum(st.dc_w[:, j] * _dc_gauss(st.dc_r, dcb + s, law)))
    return -(out + _theta_prior(q, flags))


def _hazards(p, posts):
    counts = cast(
        np.ndarray,
        sum(sum(w * c for w, c in zip(resp, cnt, strict=True)) for resp, _, cnt in posts),
    )
    q, h, r = p["trans"]

    def prop(num, den, default):
        return float(np.clip(num / den, 0.002, 0.998)) if den > 1e-8 else default

    c = counts
    return np.array(
        [
            prop(c[0, 1] + c[0, 3], c[0].sum(), q),
            prop(c[1:, 1:].trace(), c[1:].sum(), h),
            prop(c[2, 3] + c[4, 1], c[2, 3] + c[4, 1] + c[2, 0] + c[4, 0], r),
        ]
    )


def _mstep(p, flags, data, posts, cells):
    q = _solve_mean(p, flags, data, posts)
    st = _stats(q, flags, data, posts, cells)
    lay = _layout(flags)
    x0 = _pack(q, flags)
    bounds = [(lo, hi) for _, size, lo, hi in lay for _ in range(size)]
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
    f0 = _q_scale(x0, q, flags, st)
    res = minimize(
        _q_scale,
        x0,
        args=(q, flags, st),
        method="L-BFGS-B",
        bounds=bounds,
        options=dict(maxiter=60),
    )
    if np.isfinite(res.fun) and res.fun <= f0:
        q = _unpack(res.x, q, flags)
    if flags["cruise"]:
        q["trans"] = _hazards(q, posts)
    return q


def _blend(p, cand, t):
    out = {}
    for k, v in p.items():
        if k == "beta":
            out[k] = [(1 - t) * a + t * b for a, b in zip(v, cand[k], strict=True)]
        else:
            out[k] = (1 - t) * np.asarray(v) + t * np.asarray(cand[k])
            if np.ndim(v) == 0:
                out[k] = float(out[k])
    return out


def _robust_sd(x, axis=0):
    med = np.median(x, axis=axis, keepdims=True)
    mad = 1.4826 * np.median(np.abs(x - med), axis=axis)
    rms = np.sqrt(np.mean(x * x, axis=axis))
    return np.maximum(np.where(mad > 0, mad, rms), 1e-3)


def _knots_from(r, level, support):
    """Log SD knots from detail residuals binned at their nearest knot."""
    adj = r / np.sqrt(np.minimum(1.0, LONG / np.maximum(support, 1.0)))[:, None]
    nearest = np.abs(np.clip(level, 1, 7)[:, None] - KNOTS).argmin(axis=1)
    out = np.full((4, 4), np.nan)
    for k in range(4):
        sel = nearest == k
        if sel.sum() >= 4:
            out[k] = np.log(_robust_sd(adj[sel]))
    have = np.flatnonzero(np.isfinite(out[:, 0]))
    if not have.size:
        return np.zeros((4, 4))
    for m in range(4):
        out[:, m] = np.interp(KNOTS, KNOTS[have], out[have, m])
    return out


def _initial(data, flags, n_groups, sizes):
    """Data-informed start: arithmetic phase means, then robust detail scales."""
    p: dict[str, Any] = dict(
        beta=[np.zeros(s) for s in sizes],
        amp=0.0,
        knots=np.zeros((4, 4)),
        corr=np.zeros(6),
        scatter=np.zeros(2),
        style=0.0,
        active=0.0,
        trans=np.array([0.15, 0.4, 0.5]),
        group=np.zeros((n_groups, 2)),
    )
    J = 1 + int(flags["style"])
    quiet = [(np.full(J, 1.0 / J), [_onehot(len(t.rows.cruise))] * J, None) for t in data]

    def cruise_details(p):
        r = np.concatenate([_residual(t, p)[t.rows.det][t.rows.group == 0] for t in data])
        lv = np.concatenate([t.rows.level[t.rows.group == 0] for t in data])
        sp = np.concatenate([t.rows.support[t.rows.group == 0] for t in data])
        return r.reshape(-1, 4), lv, sp

    p["knots"] = _knots_from(*cruise_details(p))
    if flags["scatter"]:
        # Start weights: flight-to-flight spread of same-command phase means.
        raw = np.concatenate([t.c[t.rows.dc] for t in data]).reshape(-1, 4)
        code = np.concatenate([t.rows.dc_code for t in data])
        v = np.mean([np.var(raw[code == k], axis=0) for k in np.unique(code)], axis=0)
        p["scatter"] = 0.5 * np.log(np.maximum([v[0], v[1:].mean()], 1e-4))
    # Levels from exact phase means (DC rows dominate the diagonal-precision fit).
    p = _solve_mean(p, flags, data, quiet)
    r, lv, sp = cruise_details(p)
    p["knots"] = _knots_from(r, lv, sp)
    u = r / np.exp(_logsd(p["knots"], lv, sp))
    C = 0.9 * np.corrcoef(u.T) + 0.1 * np.eye(4) if len(u) > 8 else np.eye(4)
    if np.isfinite(C).all():
        L = np.linalg.cholesky(C)
        p["corr"] = (L / np.diag(L)[:, None])[TRIL]
    if flags["scatter"]:
        rc = np.concatenate([_residual(t, p)[t.rows.dc] for t in data]).reshape(-1, 4)
        ms = np.mean(rc * rc, axis=0)
        p["scatter"] = 0.5 * np.log(np.maximum([ms[0], ms[1:].mean()], 1e-4))
    if flags["style"]:
        fine = []
        for t in data:
            sel = (t.rows.group == 0) & (t.rows.level <= 3)
            if sel.sum() >= 8:
                fine.append(np.log(_robust_sd(_residual(t, p)[t.rows.det][sel])).mean())
        # Nonzero start breaks the symmetric stationary point at a = 0.
        p["style"] = float(np.clip(np.std(fine) if len(fine) > 1 else 0.1, 0.05, 1.0))
    if flags["cruise"]:
        p["active"] = float(np.log(1.5))
    if flags["drive"]:
        yaw = np.concatenate([_residual(t, p)[t.rows.det][t.rows.drive > 0, YAW] for t in data])
        p["amp"] = float(_robust_sd(yaw)) if len(yaw) else 0.1
    for g in np.flatnonzero(flags["supported"])[1:]:
        r, lv, sp = [], [], []
        for t in data:
            sel = t.rows.group == g
            r.append(_residual(t, p)[t.rows.det][sel])
            lv.append(t.rows.level[sel])
            sp.append(t.rows.support[sel])
        r = np.concatenate(r).reshape(-1, 4)
        if len(r):
            z2 = (r / np.exp(_logsd(p["knots"], np.concatenate(lv), np.concatenate(sp)))) ** 2
            ratio = [z2[:, 0].mean(), z2[:, 1:].mean()]
            p["group"][g] = np.clip(0.5 * np.log(np.maximum(ratio, 1e-6)), -6.0, 6.0)
    return p


class LiftingControl:
    """Portable fitted law: Structure topology plus every learned scalar."""

    def __init__(self, structure, p: dict[str, Any], flags: dict[str, Any], history):
        self.structure = structure
        self.p = p
        self.flags = flags
        self.history = [float(h) for h in history]

    @property
    def n_params(self):
        f = self.flags
        return int(
            self.structure.n_params
            + sum(len(b) for b in self.p["beta"])
            + 16
            + 6
            + 2 * f["scatter"]
            + f["style"]
            + 4 * f["cruise"]
            + f["drive"]
            + 2 * int(np.sum(f["supported"][1:]))
        )

    def sample(self, codes, n, rng):
        """``(n, 4, len(codes))`` nonnegative speeds drawn from the fitted law."""
        codes = np.asarray(codes)
        d = self.structure.design(codes)
        rw = _rows(d)
        p, f = self.p, self.flags
        law = _law(p, f)
        J = len(law.shift)
        style = rng.integers(J, size=n)
        nb = len(rw.cruise)
        states = np.zeros((n, nb), int)
        if f["cruise"]:
            A = bp._transition(p["trans"])
            for b in range(1, nb):
                if rw.cruise[b] and rw.cruise[b - 1]:
                    states[:, b] = (
                        (rng.random((n, 1)) > A[states[:, b - 1]].cumsum(axis=1))
                        .sum(axis=1)
                        .clip(0, bp.STATES - 1)
                    )
        sign = bp.SIGN[states][:, rw.block]
        coef = np.broadcast_to(_mean(rw.X, p["beta"]), (n, len(codes), 4)).copy()
        base = _logsd(p["knots"], rw.level, rw.support) + law.gm[rw.group]
        scale = np.exp(
            base[None] + law.shift[style][:, None, None] + law.act * (sign != 0)[:, :, None]
        )
        eps = rng.standard_normal((n, len(rw.det), 4)) @ law.L.T
        coef[:, rw.det] += eps * scale
        coef[:, rw.det, YAW] += sign * p["amp"] * rw.drive
        dcb = _dc_logsd(p["knots"], rw.dc_support) + law.gm[rw.dc_group]
        xi = rng.standard_normal((n, len(rw.dc), 4))
        for j, s in enumerate(law.shift):
            ids = np.flatnonzero(style == j)
            if len(rw.dc) and len(ids):
                chol = np.linalg.cholesky(_dc_cov(dcb + s, law))
                coef[np.ix_(ids, rw.dc)] += np.einsum("rab,nrb->nra", chol, xi[ids])
        coef[:, rw.stopped] = 0.0
        return readout(np.asarray(d.tree.synth(coef)))

    def to_dict(self):
        p, f = self.p, self.flags
        return {
            "family": FAMILY,
            "structure": self.structure.to_dict(),
            "flags": {
                "scatter": bool(f["scatter"]),
                "style": bool(f["style"]),
                "cruise": bool(f["cruise"]),
                "drive": bool(f["drive"]),
                "supported": [bool(s) for s in f["supported"]],
            },
            "params": {
                "beta": [b.tolist() for b in p["beta"]],
                "amp": float(p["amp"]),
                "knots": p["knots"].tolist(),
                "corr": p["corr"].tolist(),
                "scatter": p["scatter"].tolist(),
                "style": float(p["style"]),
                "active": float(p["active"]),
                "trans": p["trans"].tolist(),
                "group": p["group"].tolist(),
            },
            "n_params": self.n_params,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data):
        if data.get("family") != FAMILY:
            raise ValueError(f"not a {FAMILY} law")
        q = data["params"]
        p = {
            "beta": [np.asarray(b, float) for b in q["beta"]],
            "amp": float(q["amp"]),
            "knots": np.asarray(q["knots"], float),
            "corr": np.asarray(q["corr"], float),
            "scatter": np.asarray(q["scatter"], float),
            "style": float(q["style"]),
            "active": float(q["active"]),
            "trans": np.asarray(q["trans"], float),
            "group": np.asarray(q["group"], float).reshape(-1, 2),
        }
        f = dict(data["flags"])
        f["supported"] = np.asarray(f["supported"], bool)
        model = cls(Structure.from_dict(data["structure"]), p, f, data.get("history", []))
        if model.n_params != data["n_params"]:
            raise ValueError("stored parameter count does not match the law")
        return model


def fit(tracks, *, measured=None, max_iter=80):
    """MAP fit of ``[(speeds (4, n), codes (n,)), ...]`` physical tachometry."""
    if measured is not None:
        if len(measured) != len(tracks):
            raise ValueError("need one source tag per track")
        tracks = [t for t, m in zip(tracks, measured, strict=True) if m]
    if not tracks:
        raise ValueError("no measured training flight")
    codes = [np.asarray(c) for _, c in tracks]
    structure = Structure.from_training(codes)
    data = []
    for w, c in tracks:
        d = structure.design(np.asarray(c))
        off = np.asarray(w, float)[:, np.asarray(c) == 0]
        off_ll = float(np.sum(gaussian.logpdf(off, scale=MOVING_RPS)))
        data.append(_Track(np.asarray(d.tree.analyse(encode(w)), float), _rows(d), off_ll))
    n_groups = int(structure.n_groups)
    # Detail cells share one covariance: (level, support, group) is all it uses.
    lv = np.concatenate([t.rows.level for t in data])
    sp = np.concatenate([t.rows.support for t in data])
    gr = np.concatenate([t.rows.group for t in data])
    table, inverse = np.unique(np.column_stack((lv, sp, gr)), axis=0, return_inverse=True)
    inverse = np.ravel(inverse)
    cells = (table[:, 0], table[:, 1], table[:, 2].astype(int))
    start = 0
    for t in data:
        t.key = inverse[start : start + len(t.rows.det)]
        start += len(t.rows.det)
    supported = np.zeros(n_groups, bool)
    supported[0] = True
    for t in data:
        supported[np.r_[t.rows.group, t.rows.dc_group].astype(int)] = True
    repeated = any((t.rows.cruise[1:] & t.rows.cruise[:-1]).any() for t in data)
    flags = dict(
        scatter=len(data) > 1,
        style=len(data) > 1,
        cruise=bool(repeated),
        drive=bool(repeated and any(t.rows.drive.any() for t in data)),
        supported=supported,
        penalty=[np.asarray(structure.penalty(m), float) for m in range(4)],
    )
    sizes = [x.shape[1] for x in data[0].rows.X]
    p = _initial(data, flags, n_groups, sizes)
    obj, posts = _estep(p, flags, data)
    if not np.isfinite(obj):
        raise FloatingPointError("nonfinite initial lifting objective")
    history = [obj]
    tol = 1e-6 * sum(len(t.c) for t in data)
    for _ in range(max_iter):
        full = _mstep(p, flags, data, posts, cells)
        cand = full
        new, new_posts = _estep(cand, flags, data)
        step = 1.0
        while not (np.isfinite(new) and new >= history[-1]) and step > 0.1:
            # Guard on the marginal objective itself: shrink toward the incumbent.
            step *= 0.5
            cand = _blend(p, full, step)
            new, new_posts = _estep(cand, flags, data)
        if not (np.isfinite(new) and new >= history[-1]):
            break
        gain = new - history[-1]
        p, posts = cand, new_posts
        history.append(new)
        if gain < tol:
            break
    return LiftingControl(structure, p, flags, history)
