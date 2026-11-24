"""Whole-flight likelihood generator of the four rotor speeds on a coarse schedule.

One model, one fitting process, for a whole recording: a shared stable
second-order response in mixer coordinates driven by the targets of a latent
action graph (:mod:`experiments.rps_traj.unified_grammar`), with persistent
per-flight idle and hover anchors.  The only schedule input is the coarse
phase code per 25 Hz sample (0 before motors-on, 1 motors-on to take-off,
2 take-off to landing, 3 landing to end); no level, arrival, motors-off time
or target speed of the flight to generate is used.

Model
-----
Rotor speeds ``w_t`` (rev/s, MIXER rotor order) map to the mixer modes
``x_t = MIXER.T w_t / 4`` (common, roll, pitch, yaw).  With velocity
``v_t = x_t - x_{t-1}``, for every sample ``t``::

    v_t = F v_{t-1} + K (m(s_t, j) - x_{t-1}) + e_t,   e_t ~ N(0, diag sigma(s_t)^2)

* ``F``, ``K`` diagonal (one value per mode), shared by every action, phase
  and flight, and constrained to the stability region of the second-order
  recursion with a held target (``|F| < 1``, ``0 < K < 2 (1 + F)``, minus a
  margin :data:`STABILITY_MARGIN`): any held target is approached without
  divergence.
* ``s_t`` is the emission label of the action graph (13 labels, fixed by the
  grammar); ``sigma(s)`` is a label-specific (hence phase/action-specific)
  diagonal disturbance.
* ``j`` in ``{0, .., J-1}`` (``J`` = :data:`N_ANCHORS` = 2) is ONE latent
  anchor pair per flight, drawn with probability ``pi_j`` and kept for the
  whole recording: idle anchor ``idle_j`` and hover anchor ``hover_j``
  (4-vectors, all modes).  The target of label ``s`` is
  ``m(s, j) = ref_s(j) + offset_s`` where ``ref_s`` is ``0``, ``idle_j`` or
  ``hover_j`` (the label's ``ref``) and ``offset_s`` a learned
  action-relative 4-vector; the reference labels ``idle`` and ``hold`` and
  the ``zero`` labels have no offset, which makes anchors and offsets
  identified.  Startup, standby, cruise and rundown share the anchors of
  their flight; nothing is a plateau mean or a template.
* The discrete graph resets at phase changes; the AR state never does.

Likelihood (the documented observed-speed AR likelihood)
-------------------------------------------------------
There is no separate observation noise: the measured speed IS the AR state,
so the likelihood is exact and needs no Kalman filtering.  Per flight::

    p(x | phases) = sum_j pi_j sum_{s_{0:T-1}} p(s | phases; graph) prod_t p(x_t | x_{t-1}, x_{t-2}, s_t, j)

conditioned on the first two samples ``x_0, x_1`` (they carry no
likelihood factor; every ``t >= 2`` does); a factor whose window
``(x_{t-2}, x_{t-1}, x_t)`` touches a non-finite sample is dropped (the
chain re-conditions after a gap: a conditional likelihood, not a
marginalisation over the missing speeds).  The sum over label paths is the
grammar's exact forward recursion on the dense ``(T, 13)`` log-emissions;
the sum over ``j`` is exact (a two-component mixture over WHOLE flights).

Fitting (generalized EM on the penalized observed likelihood)
-------------------------------------------------------------
E-step: for every flight and anchor pair, forward-backward over the graph
gives the flight log-likelihood ``L_nj`` and label occupancies; the
whole-flight responsibility is ``r_nj ~ pi_j exp(L_nj)``.  Every quantity the
response needs is a responsibility*occupancy weighted moment of
``(1, v_t, v_{t-1}, x_{t-1})`` up to second order, per (anchor, label, mode)
(:func:`_moments`), so the M-step never revisits samples.  M-steps, each an
exact conditional maximiser of the expected complete-data log-likelihood
plus the log-prior: graph (grammar's MAP on its expected counts), ``pi``
(Dirichlet MAP), ``sigma`` (inverse-gamma MAP with a floor
:data:`SIGMA_FLOOR` so a motionless label cannot collapse), ``(F, K)`` per
mode (2-D convex QP on the stability polygon, solved exactly), anchors and
offsets jointly per mode (linear system with weak Gaussian priors that keep
an empty anchor pair finite).  The penalized observed objective is
evaluated after each E-step; it is non-decreasing by construction, and a
decrease beyond floating-point tolerance reverts to the previous parameters
and stops (``stop_reason='non_monotone'``).  Stop when the gain per valid
sample falls below ``tol`` or after ``max_iter`` M-steps.

Initialisation (training tracks only): anchors from the two halves of the
training flights ordered by their phase-2 median common speed (phase-1 tail
and phase-2 median), ``(F, K)`` from a per-flight-demeaned phase-2
regression projected into the stability polygon, free offsets from the
grammar's sign hint times a quarter of the idle-to-hover gap, wide noise.

Sampling: anchor pair ``j ~ pi``, label path from the graph on the given
phases, then EXACTLY the recursion of the likelihood (no clipping fed back,
no upper cap, no learned bound).  First-two-sample convention of the
generator: ``x_0`` is the target of the first sampled label (rest, zero,
when the schedule starts before motors-on) and ``x_1`` follows the
recursion with ``v_0 = 0``; the likelihood instead conditions on the
observed ``x_0, x_1``.  :meth:`UnifiedResponseModel.sample` returns the
signed rotor speeds ``MIXER x`` of this fitted law by default; negative
values are model failures and are kept and reported: ``last_clip_fraction``
is the share of generated sample times with any negative rotor.  An
explicitly separate shaft readout ``max(MIXER x, 0)`` is available only on
request (``sample(..., shaft_only=True)``) and never enters the recursion
or the likelihood.  Observed speeds, including exact zeros at rest, are
scored as values of the AR state, not as censored at zero; no separate
observation/logger noise is identified (the data have no second sensor).

Limitations: the anchor law is a two-point mixture (a continuous per-flight
anchor would need integrating the anchor out of a non-linear-in-labels
likelihood; the mixture is exact and cheap); the response ``(F, K)`` is the
same for powered and unpowered motion (rundown differences go to the
label's offset and disturbance); gap handling is conditional, as above; the
Gaussian law puts mass below zero near rest and rundown, which shows up as
negative signed draws (reported, not hidden) rather than a censored law.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numba import njit
from scipy.special import logsumexp

from experiments.rps_traj import unified_grammar as ug
from tracking.rotors import MIXER, NUM_ROTORS

N_MODES = 4
N_ANCHORS = 2
#: Disturbance floor per mode (rev/s): motionless labels stop here.
SIGMA_FLOOR = 0.01
#: Inverse-gamma prior on every sigma^2: NU pseudo-samples at SD SIGMA_PRIOR_SD.
SIGMA_PRIOR_NU = 2.0
SIGMA_PRIOR_SD = 0.5
#: Gaussian priors (rev/s, mode units): spread of anchors around their mean,
#: absolute scale of anchors, action-relative offsets.  All weak: they only
#: keep an unvisited anchor pair or label finite.
ANCHOR_SPREAD_SD = 50.0
ANCHOR_SD = 1000.0
OFFSET_SD = 100.0
#: Dirichlet concentration of the anchor-pair weights.
PI_CONCENTRATION = 2.0
STABILITY_MARGIN = 1e-3
#: Relative tolerance of the monotonicity guard.
MONOTONE_RTOL = 1e-9

_REF_CODE = {"zero": 0, "idle": 1, "hover": 2}
# Moment layout of _features: [1, y, p, q, yy, yp, yq, pp, pq, qq]
# with y = v_t, p = v_{t-1}, q = x_{t-1}.
_N_MOM = 10
_I0, _IY, _IP, _IQ, _IYY, _IYP, _IYQ, _IPP, _IPQ, _IQQ = range(_N_MOM)


# ─── labels ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _LabelMeta:
    names: tuple[str, ...]
    ref: np.ndarray  # (E,) 0 zero, 1 idle, 2 hover
    free: np.ndarray  # (E,) bool: learned offset
    sign: np.ndarray  # (E,) init hint
    phases: tuple[tuple[int, ...], ...]

    @property
    def n(self) -> int:
        return len(self.names)


def _label_meta() -> _LabelMeta:
    specs = ug.LABELS
    if len(specs) != ug.N_LABELS:
        raise ValueError("grammar label table and N_LABELS disagree")
    ref = np.array([_REF_CODE[s.ref] for s in specs], dtype=np.int64)
    free = np.array([bool(s.free_offset) for s in specs])
    if np.any(free & (ref == 0)):
        raise ValueError("a zero-target label cannot carry an offset")
    for code in (1, 2):
        if not np.any((ref == code) & ~free):
            raise ValueError("each anchor needs one fixed reference label")
    return _LabelMeta(
        names=tuple(s.name for s in specs),
        ref=ref,
        free=free,
        sign=np.array([int(s.sign) for s in specs], dtype=np.int64),
        phases=tuple(tuple(int(p) for p in s.phases) for s in specs),
    )


# ─── parameters ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResponseParams:
    """Response, anchors, offsets, disturbances and anchor weights."""

    F: np.ndarray  # (4,) shared, or (2,4) cruise and transient response
    K: np.ndarray  # same shape as F
    idle: np.ndarray  # (J, 4)
    hover: np.ndarray  # (J, 4)
    offset: np.ndarray  # (E, 4), zero on labels without a free offset
    sigma: np.ndarray  # (E, 4)
    log_pi: np.ndarray  # (J,)

    def targets(self, meta: _LabelMeta) -> np.ndarray:
        """``(E, J, 4)`` target of every label under every anchor pair."""
        base = np.stack((np.zeros_like(self.idle), self.idle, self.hover))  # (3, J, 4)
        return base[meta.ref] + self.offset[:, None, :]


def _response_index(params: ResponseParams) -> np.ndarray:
    if params.F.ndim == 1:
        return np.zeros(ug.N_LABELS, dtype=np.intp)
    return np.array(
        [s.name not in ("hold", "drive_up", "drive_down") for s in ug.LABELS], dtype=np.intp
    )


def _stable_vertices() -> np.ndarray:
    eps = STABILITY_MARGIN
    f_lo, f_hi = -1.0 + eps, 1.0 - eps
    return np.array(
        [
            (f_lo, eps),
            (f_hi, eps),
            (f_hi, 2.0 * (1.0 + f_hi) * (1.0 - eps)),
            (f_lo, 2.0 * (1.0 + f_lo) * (1.0 - eps)),
        ]
    )


def _in_polygon(theta: np.ndarray, vert: np.ndarray) -> bool:
    # Convex, counter-clockwise: theta inside iff left of (or on) every edge.
    for i in range(len(vert)):
        a, b = vert[i], vert[(i + 1) % len(vert)]
        cross = (b[0] - a[0]) * (theta[1] - a[1]) - (b[1] - a[1]) * (theta[0] - a[0])
        if cross < 0.0:
            return False
    return True


def _stable_qp(A: np.ndarray, b: np.ndarray, current: np.ndarray) -> np.ndarray:
    """``argmin theta' A theta - 2 b' theta`` over the stability polygon of
    ``(F, K)``; exact for positive-definite ``A`` (interior optimum, else the
    best point of the four edges).  Keeps ``current`` without curvature."""
    vert = _stable_vertices()
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b))):
        return current
    if np.linalg.eigvalsh(A)[0] <= 1e-12 * max(1.0, float(np.trace(A))):
        return current
    theta = np.linalg.solve(A, b)
    if _in_polygon(theta, vert):
        return theta

    def f(t: np.ndarray) -> float:
        return float(t @ A @ t - 2.0 * b @ t)

    best, best_f = current, f(current) if _in_polygon(current, vert) else np.inf
    for i in range(len(vert)):
        a0, d = vert[i], vert[(i + 1) % len(vert)] - vert[i]
        curv = float(d @ A @ d)
        lam = 0.0 if curv <= 0 else float(np.clip(d @ (b - A @ a0) / curv, 0.0, 1.0))
        cand = a0 + lam * d
        fc = f(cand)
        if fc < best_f:
            best, best_f = cand, fc
    return best


# ─── data ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Flight:
    y: np.ndarray  # (T, 4) v_t, zero where invalid
    p: np.ndarray  # (T, 4) v_{t-1}
    q: np.ndarray  # (T, 4) x_{t-1}
    x: np.ndarray  # (T, 4) modes, NaN kept (initialisation only)
    valid: np.ndarray  # (T,) bool
    phases: np.ndarray  # (T,) int64

    @property
    def n_valid(self) -> int:
        return int(self.valid.sum())


def _check_phases(phases: np.ndarray, T: int | None = None) -> np.ndarray:
    ph = np.asarray(phases)
    if ph.ndim != 1 or (T is not None and len(ph) != T):
        raise ValueError(f"phases must be a ({T},) array, got {ph.shape}")
    if not np.issubdtype(ph.dtype, np.integer) and not np.all(np.equal(np.mod(ph, 1), 0)):
        raise ValueError("phase codes must be integers")
    ph = ph.astype(np.int64)
    if ph.size and (ph.min() < 0 or ph.max() > 3):
        raise ValueError("phase codes must lie in 0..3")
    return ph


def _prepare(speed: np.ndarray, phases: np.ndarray) -> _Flight:
    w = np.asarray(speed, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != NUM_ROTORS:
        raise ValueError(f"speed must be (4, T), got {w.shape}")
    T = w.shape[1]
    ph = _check_phases(phases, T)
    x = ((MIXER.T @ w) / NUM_ROTORS).T  # (T, 4); NaN propagates per sample
    ok = cast(np.ndarray, np.isfinite(x).all(axis=1))
    valid = np.zeros(T, dtype=bool)
    if T > 2:
        valid[2:] = ok[2:] & ok[1:-1] & ok[:-2]
    y, p, q = (np.zeros((T, N_MODES)) for _ in range(3))
    idx = np.flatnonzero(valid)
    y[idx] = x[idx] - x[idx - 1]
    p[idx] = x[idx - 1] - x[idx - 2]
    q[idx] = x[idx - 1]
    return _Flight(y=y, p=p, q=q, x=x, valid=valid, phases=ph)


def _features(f: _Flight) -> np.ndarray:
    """``(T, 4, 10)`` per-sample monomials; zero on invalid samples."""
    one = np.broadcast_to(f.valid[:, None].astype(np.float64), f.y.shape)
    y, p, q = f.y, f.p, f.q
    return np.stack((one, y, p, q, y * y, y * p, y * q, p * p, p * q, q * q), axis=-1)


# ─── likelihood ───────────────────────────────────────────────────────────────


def _drive(f: _Flight, params: ResponseParams) -> np.ndarray:
    """``a_t = v_t - F v_{t-1} + K x_{t-1}``: the residual is ``a_t - K m``."""
    return f.y - params.F * f.p + params.K * f.q


def _log_emissions(
    a: np.ndarray, valid: np.ndarray, targets_j: np.ndarray, K: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """``(T, E)`` ``log N(a_t; K m_e, diag sigma_e^2)``, exactly 0 on invalid rows.

    Expanded into three ``(T, 4) @ (4, E)`` products: O(T E) memory."""
    prec = 1.0 / sigma**2  # (E, 4)
    km = K * targets_j  # (E, 4)
    const = -0.5 * np.sum(np.log(2.0 * np.pi * sigma**2) + km * km * prec, axis=1)
    out = -0.5 * ((a * a) @ prec.T) + a @ (km * prec).T + const
    out[~valid] = 0.0
    return out


def _emission_tables(f: _Flight, params: ResponseParams, targets: np.ndarray):
    """One emission table per anchor. Two tied response groups need just two residual series."""
    if params.F.ndim == 1:
        a = _drive(f, params)
        for target in targets.transpose(1, 0, 2):
            yield _log_emissions(a, f.valid, target, params.K, params.sigma)
        return
    ids = _response_index(params)
    groups = [
        (np.flatnonzero(ids == g), f.y - F * f.p + K * f.q, K)
        for g, (F, K) in enumerate(zip(params.F, params.K, strict=True))
    ]
    for target in targets.transpose(1, 0, 2):
        table = np.empty((len(f.y), len(ids)))
        for sel, a, K in groups:
            table[:, sel] = _log_emissions(a, f.valid, target[sel], K, params.sigma[sel])
        yield table


def _moments(W: np.ndarray, feats: np.ndarray) -> np.ndarray:
    """``(E, 4, 10)`` weighted moments ``sum_t W[t, e] feats[t, d, k]``."""
    return np.einsum("te,tdk->edk", W, feats, optimize=True)


def _sse(S: np.ndarray, F: np.ndarray, K: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Weighted ``sum (y - F p + K q - K m)^2`` from moments ``S[..., 4, 10]``
    with targets ``m[..., 4]`` (broadcast over leading axes)."""
    g = lambda i: S[..., i]  # noqa: E731
    out = (
        g(_IYY)
        + F**2 * g(_IPP)
        + K**2 * g(_IQQ)
        + K**2 * m**2 * g(_I0)
        - 2 * F * g(_IYP)
        + 2 * K * g(_IYQ)
        - 2 * K * m * g(_IY)
        - 2 * F * K * g(_IPQ)
        + 2 * F * K * m * g(_IP)
        - 2 * K**2 * m * g(_IQ)
    )
    return np.maximum(out, 0.0)


# ─── priors and M-steps ───────────────────────────────────────────────────────


def _sigma_prior_ab() -> tuple[float, float]:
    alpha = 0.5 * SIGMA_PRIOR_NU
    return alpha, alpha * SIGMA_PRIOR_SD**2


def _response_log_prior(params: ResponseParams, meta: _LabelMeta) -> float:
    """Log-prior (up to constants) that the response M-steps maximise with."""
    alpha, beta = _sigma_prior_ab()
    var = params.sigma**2
    lp = float(np.sum(-(alpha + 1.0) * np.log(var) - beta / var))
    lp += float((PI_CONCENTRATION - 1.0) * np.sum(params.log_pi))
    for anchor in (params.idle, params.hover):
        lp -= 0.5 * float(np.sum((anchor - anchor.mean(axis=0)) ** 2)) / ANCHOR_SPREAD_SD**2
        lp -= 0.5 * float(np.sum(anchor**2)) / ANCHOR_SD**2
    lp -= 0.5 * float(np.sum(params.offset[meta.free] ** 2)) / OFFSET_SD**2
    return lp


def _m_step(
    params: ResponseParams, meta: _LabelMeta, mom: np.ndarray, resp_total: np.ndarray
) -> ResponseParams:
    """Sequential exact conditional maximisers: pi, sigma, (F, K), targets.

    ``mom``: ``(J, E, 4, 10)`` responsibility*occupancy weighted moments;
    ``resp_total``: ``(J,)`` summed whole-flight responsibilities."""
    J, E = mom.shape[:2]
    # pi: Dirichlet MAP.
    a = resp_total + PI_CONCENTRATION - 1.0
    log_pi = np.log(np.maximum(a, 1e-300) / a.sum())

    # sigma: inverse-gamma MAP with floor, under the current F, K, targets.
    m = params.targets(meta).transpose(1, 0, 2)  # (J, E, 4)
    ids = _response_index(params)
    F, K = np.atleast_2d(params.F).copy(), np.atleast_2d(params.K).copy()
    sse = _sse(mom, F[ids], K[ids], m).sum(axis=0)
    n = mom[..., _I0].sum(axis=0)
    alpha, beta = _sigma_prior_ab()
    var = (sse + 2.0 * beta) / (n + 2.0 * alpha + 2.0)
    sigma = np.sqrt(np.maximum(var, SIGMA_FLOOR**2))
    prec = 1.0 / sigma**2  # (E, 4)

    # (F, K) per mode: weighted LS of y on (p, u = m - q) on the stable polygon.
    S = mom * prec[None, :, :, None]  # (J, E, 4, 10)
    for group in range(len(F)):
        sel = np.flatnonzero(ids == group)
        sg, mg = S[:, sel], m[:, sel]
        for d in range(N_MODES):
            s, md = sg[:, :, d, :], mg[:, :, d]
            A = np.array(
                [
                    [s[..., _IPP].sum(), (md * s[..., _IP] - s[..., _IPQ]).sum()],
                    [0.0, (md**2 * s[..., _I0] - 2 * md * s[..., _IQ] + s[..., _IQQ]).sum()],
                ]
            )
            A[1, 0] = A[0, 1]
            b = np.array([s[..., _IYP].sum(), (md * s[..., _IY] - s[..., _IYQ]).sum()])
            F[group, d], K[group, d] = _stable_qp(A, b, np.array([F[group, d], K[group, d]]))

    # Anchors and offsets per mode: one linear system (weak Gaussian priors).
    free_idx = np.cumsum(meta.free) - 1
    n_free = int(meta.free.sum())
    nb = 2 * J + n_free
    idle, hover = params.idle.copy(), params.hover.copy()
    offset = params.offset.copy()
    spread = (np.eye(J) - 1.0 / J) / ANCHOR_SPREAD_SD**2 + np.eye(J) / ANCHOR_SD**2
    for d in range(N_MODES):
        H = np.zeros((nb, nb))
        g = np.zeros(nb)
        H[:J, :J] += spread
        H[J : 2 * J, J : 2 * J] += spread
        H[2 * J :, 2 * J :] += np.eye(n_free) / OFFSET_SD**2
        s = S[:, :, d, :]
        kd, fd = K[ids, d], F[ids, d]
        weight = kd**2 * s[..., _I0]
        drive = kd * (s[..., _IY] - fd * s[..., _IP] + kd * s[..., _IQ])
        for j in range(J):
            for e in range(E):
                idx = []
                if meta.ref[e] == 1:
                    idx.append(j)
                elif meta.ref[e] == 2:
                    idx.append(J + j)
                if meta.free[e]:
                    idx.append(2 * J + free_idx[e])
                if not idx:
                    continue
                ii = np.array(idx)
                H[np.ix_(ii, ii)] += weight[j, e]
                g[ii] += drive[j, e]
        beta_d = np.linalg.solve(H, g)
        idle[:, d] = beta_d[:J]
        hover[:, d] = beta_d[J : 2 * J]
        offset[meta.free, d] = beta_d[2 * J :]
    return ResponseParams(
        F=F.reshape(params.F.shape),
        K=K.reshape(params.K.shape),
        idle=idle,
        hover=hover,
        offset=offset,
        sigma=sigma,
        log_pi=log_pi,
    )


# ─── E-step ───────────────────────────────────────────────────────────────────


@dataclass
class _EStep:
    data_loglik: float
    resp: np.ndarray  # (N, J)
    flight_loglik: np.ndarray  # (N, J)
    mom: np.ndarray  # (J, E, 4, 10)
    stats: Any  # grammar sufficient statistics
    occupancy: np.ndarray  # (E,) responsibility-weighted label occupancy


def _e_step(
    flights: list[_Flight], grammar: ug.Grammar, params: ResponseParams, meta: _LabelMeta
) -> _EStep:
    J, E = len(params.log_pi), meta.n
    targets = params.targets(meta)
    mom = np.zeros((J, E, N_MODES, _N_MOM))
    stats = grammar.zero_stats()
    occ = np.zeros(E)
    resp = np.zeros((len(flights), J))
    lls = np.zeros((len(flights), J))
    total = 0.0
    for n, f in enumerate(flights):
        fbs = []
        for j, le in enumerate(_emission_tables(f, params, targets)):
            fb = grammar.forward_backward(le, f.phases)
            lls[n, j] = fb.loglik
            fbs.append(fb)
        lw = params.log_pi + lls[n]
        lse = float(cast(float, logsumexp(lw)))
        total += lse
        resp[n] = np.exp(lw - lse)
        feats = _features(f)
        for j in range(J):
            if resp[n, j] <= 0.0:
                continue
            stats.add(fbs[j], float(resp[n, j]))
            W = resp[n, j] * fbs[j].occupancy
            mom[j] += _moments(W, feats)
            occ += W.sum(axis=0)
    return _EStep(total, resp, lls, mom, stats, occ)


# ─── initialisation ───────────────────────────────────────────────────────────


def _finite_median(rows: np.ndarray) -> np.ndarray:
    rows = rows[np.isfinite(rows).all(axis=1)]
    return np.median(rows, axis=0) if len(rows) else np.full(N_MODES, np.nan)


def _init_params(flights: list[_Flight], meta: _LabelMeta, J: int) -> ResponseParams:
    idle_n, hover_n = [], []
    for f in flights:
        i1 = np.flatnonzero(f.phases == 1)
        i2 = np.flatnonzero(f.phases == 2)
        idle_n.append(_finite_median(f.x[i1[len(i1) // 2 :]]))
        hover_n.append(
            _finite_median(f.x[i2[len(i2) // 4 : max(3 * len(i2) // 4, len(i2) // 4 + 1)]])
        )
    idle_n, hover_n = np.array(idle_n), np.array(hover_n)
    has_h = np.isfinite(hover_n).all(axis=1)
    if not has_h.any():
        raise ValueError("no training flight has finite phase-2 (airborne) samples")
    has_i = cast(np.ndarray, np.isfinite(idle_n).all(axis=1))
    pooled_h = hover_n[has_h].mean(axis=0)
    pooled_i = idle_n[has_i].mean(axis=0) if has_i.any() else 0.35 * pooled_h
    order = np.argsort(np.where(has_h, hover_n[:, 0], np.inf))[: has_h.sum()]
    groups = np.array_split(order, J)
    idle, hover = np.empty((J, N_MODES)), np.empty((J, N_MODES))
    for j, g in enumerate(groups):
        hover[j] = hover_n[g].mean(axis=0) if len(g) else pooled_h
        gi = [i for i in g if has_i[i]]
        idle[j] = idle_n[gi].mean(axis=0) if gi else pooled_i
    # Break exact ties (e.g. one flight) without favouring a component.
    spread = np.linspace(-0.02, 0.02, J)[:, None] if J > 1 else np.zeros((1, 1))
    if np.allclose(hover, hover[0]):
        hover = hover + spread * np.abs(hover[:, :1])
    if np.allclose(idle, idle[0]):
        idle = idle + spread * np.abs(idle[:, :1])

    # (F, K): per-flight-demeaned regression of v_t on (v_{t-1}, x_{t-1}) in phase 2.
    A = np.zeros((N_MODES, 2, 2))
    b = np.zeros((N_MODES, 2))
    for f in flights:
        rows = np.flatnonzero(f.valid & (f.phases == 2))
        if len(rows) < 3:
            continue
        y, p, q = (z[rows] - z[rows].mean(axis=0) for z in (f.y, f.p, f.q))
        A[:, 0, 0] += (p * p).sum(0)
        A[:, 0, 1] -= (p * q).sum(0)
        A[:, 1, 1] += (q * q).sum(0)
        b[:, 0] += (y * p).sum(0)
        b[:, 1] -= (y * q).sum(0)
    F, K = np.empty(N_MODES), np.empty(N_MODES)
    default = np.array([0.5, 0.1])
    for d in range(N_MODES):
        A[d, 1, 0] = A[d, 0, 1]
        theta0 = default
        if np.linalg.det(A[d]) > 1e-12 * max(1.0, np.trace(A[d])) ** 2:
            theta0 = np.linalg.solve(A[d], b[d])
        F[d], K[d] = _stable_qp(np.eye(2), theta0, default)  # projection

    gap = float(np.mean(hover[:, 0] - idle[:, 0]))
    offset = np.zeros((meta.n, N_MODES))
    offset[meta.free, 0] = 0.25 * abs(gap) * meta.sign[meta.free]

    # Wide disturbances: SD of the AR(2) residual without target term per phase.
    sigma = np.empty((meta.n, N_MODES))
    for e in range(meta.n):
        res = [(f.y - F * f.p)[f.valid & np.isin(f.phases, meta.phases[e])] for f in flights]
        res = np.concatenate(res) if res else np.zeros((0, N_MODES))
        sd = res.std(axis=0) if len(res) > 1 else np.full(N_MODES, SIGMA_PRIOR_SD)
        sigma[e] = np.maximum(sd, SIGMA_FLOOR)
    return ResponseParams(
        F=F, K=K, idle=idle, hover=hover, offset=offset, sigma=sigma, log_pi=np.full(J, -np.log(J))
    )


# ─── sampling kernel ──────────────────────────────────────────────────────────


@njit(cache=True)
def _simulate(labels, comp, targets, sigma, F, K, eps, out):  # pragma: no cover - numba
    """Draw ``i``: anchor ``comp[i]``, labels ``labels[i]``; writes the
    latent modes ``x_t`` into ``out[i]`` (the likelihood's recursion)."""
    n, T = labels.shape
    D = out.shape[2]
    for i in range(n):
        j = comp[i]
        l0 = labels[i, 0]
        for d in range(D):
            out[i, 0, d] = targets[l0, j, d]
        for t in range(1, T):
            lab = labels[i, t]
            for d in range(D):
                x = out[i, t - 1, d]
                v = 0.0 if t == 1 else x - out[i, t - 2, d]
                out[i, t, d] = (
                    x
                    + F[lab, d] * v
                    + K[lab, d] * (targets[lab, j, d] - x)
                    + sigma[lab, d] * eps[i, t, d]
                )


@dataclass(frozen=True)
class Draws:
    """Generated flights: signed rotor speeds of the fitted law and latent choices."""

    raw: np.ndarray  # (n, 4, T) signed rotor speeds MIXER x, rev/s
    anchor: np.ndarray  # (n,) anchor pair per flight
    labels: np.ndarray  # (n, T) emission labels

    @property
    def negative_fraction(self) -> float:
        """Share of sample times (over draws) with any negative rotor speed."""
        return float(np.mean((self.raw < 0.0).any(axis=1))) if self.raw.size else 0.0

    @property
    def shaft(self) -> np.ndarray:
        """Output-only shaft readout ``max(raw, 0)``; not the fitted law."""
        return np.maximum(self.raw, 0.0)


# ─── model ────────────────────────────────────────────────────────────────────


def _arr(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


class UnifiedResponseModel:
    """Fitted whole-flight generator; see the module docstring."""

    def __init__(
        self,
        grammar: ug.Grammar,
        params: ResponseParams,
        *,
        hierarchical: bool,
        history: list[float] | None = None,
        fit_info: dict[str, Any] | None = None,
    ) -> None:
        self.grammar = grammar
        self.params = params
        self.hierarchical = bool(hierarchical)
        self.history = list(history or [])
        self.fit_info = dict(fit_info or {})
        self.meta = _label_meta()
        self.last_clip_fraction = float("nan")

    # parameter count ---------------------------------------------------------
    def n_params_breakdown(self) -> dict[str, int]:
        J, E = len(self.params.log_pi), self.meta.n
        return {
            "response_F_K": self.params.F.size + self.params.K.size,
            "anchors": 2 * J * N_MODES,
            "offsets": int(self.meta.free.sum()) * N_MODES,
            "disturbance_sd": E * N_MODES,
            "anchor_weights": J - 1,
            "grammar": int(self.grammar.n_params),
        }

    @property
    def n_params(self) -> int:
        """Free parameters; fixed offsets, zero targets and tied emissions excluded."""
        return int(sum(self.n_params_breakdown().values()))

    # likelihood --------------------------------------------------------------
    def loglik(self, speed: np.ndarray, phases: np.ndarray) -> float:
        """Marginal log-likelihood of one recording (anchors and labels summed out)."""
        f = _prepare(speed, phases)
        targets = self.params.targets(self.meta)
        lls = [self.grammar.loglik(le, phases) for le in _emission_tables(f, self.params, targets)]
        return float(cast(float, logsumexp(self.params.log_pi + np.array(lls))))

    # sampling ----------------------------------------------------------------
    def draw(self, phases: np.ndarray, n: int, rng: np.random.Generator) -> Draws:
        """``n`` flights on the coarse schedule, with latent speeds and choices."""
        ph = _check_phases(phases)
        if len(ph) == 0:
            raise ValueError("empty schedule")
        T = len(ph)
        J = len(self.params.log_pi)
        pi = np.exp(self.params.log_pi - logsumexp(self.params.log_pi))
        comp = rng.choice(J, size=n, p=pi).astype(np.int64)
        labels = np.asarray(self.grammar.sample(ph, rng, n=n), dtype=np.int64).reshape(n, T)
        eps = rng.standard_normal((n, T, N_MODES))
        x = np.empty((n, T, N_MODES))
        ix = _response_index(self.params)
        _simulate(
            labels,
            comp,
            self.params.targets(self.meta),
            self.params.sigma,
            np.atleast_2d(self.params.F)[ix],
            np.atleast_2d(self.params.K)[ix],
            eps,
            x,
        )
        raw = np.einsum("rd,ntd->nrt", MIXER, x)
        out = Draws(raw=raw, anchor=comp, labels=labels)
        self.last_clip_fraction = out.negative_fraction
        return out

    def sample(
        self, phases: np.ndarray, n: int, rng: np.random.Generator, *, shaft_only: bool = False
    ) -> np.ndarray:
        """``(n, 4, T)`` signed rotor rev/s of the fitted law on the coarse
        phase schedule (``shaft_only=True``: the separate ``max(., 0)``
        readout).  Sets ``last_clip_fraction`` (times with a negative rotor)."""
        d = self.draw(phases, n, rng)
        return d.shaft if shaft_only else d.raw

    # serialisation -----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        p = self.params
        d = {
            "model": "unified_response",
            "hierarchical": self.hierarchical,
            "labels": [
                {
                    "name": nm,
                    "phases": list(ph),
                    "ref": ("zero", "idle", "hover")[r],
                    "free_offset": bool(fr),
                }
                for nm, ph, r, fr in zip(
                    self.meta.names, self.meta.phases, self.meta.ref, self.meta.free
                )
            ],
            "params": {
                "F": p.F.tolist(),
                "K": p.K.tolist(),
                "idle": p.idle.tolist(),
                "hover": p.hover.tolist(),
                "offset": p.offset.tolist(),
                "sigma": p.sigma.tolist(),
                "log_pi": p.log_pi.tolist(),
            },
            "output": "signed MIXER x of the fitted recursion; optional shaft readout "
            "max(., 0) is output-only and not in the likelihood",
            "grammar": self.grammar.to_dict(),
            "n_params": self.n_params,
            "n_params_breakdown": self.n_params_breakdown(),
            "constants": {
                "sigma_floor": SIGMA_FLOOR,
                "sigma_prior_nu": SIGMA_PRIOR_NU,
                "sigma_prior_sd": SIGMA_PRIOR_SD,
                "anchor_spread_sd": ANCHOR_SPREAD_SD,
                "anchor_sd": ANCHOR_SD,
                "offset_sd": OFFSET_SD,
                "pi_concentration": PI_CONCENTRATION,
                "stability_margin": STABILITY_MARGIN,
            },
            "history": [float(h) for h in self.history],
            "fit": self.fit_info,
        }
        json.dumps(d)  # fail here, not in the caller, if anything is not JSON
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> UnifiedResponseModel:
        p = d["params"]
        params = ResponseParams(
            F=_arr(p["F"]),
            K=_arr(p["K"]),
            idle=_arr(p["idle"]),
            hover=_arr(p["hover"]),
            offset=_arr(p["offset"]),
            sigma=_arr(p["sigma"]),
            log_pi=_arr(p["log_pi"]),
        )
        model = cls(
            ug.Grammar.from_dict(d["grammar"]),
            params,
            hierarchical=d["hierarchical"],
            history=d.get("history"),
            fit_info=d.get("fit"),
        )
        names = [lab["name"] for lab in d["labels"]]
        if tuple(names) != model.meta.names:
            raise ValueError("saved label table differs from the grammar's")
        return model


# ─── fitting ──────────────────────────────────────────────────────────────────


def fit(
    tracks: list[tuple[np.ndarray, np.ndarray]],
    *,
    hierarchical: bool,
    max_iter: int = 30,
    tol: float = 1e-5,
    n_anchors: int = N_ANCHORS,
    initial: UnifiedResponseModel | None = None,
    split_response: bool = False,
) -> UnifiedResponseModel:
    """Fit on whole training recordings ``(speed (4, T) rev/s, phase codes (T,))``.

    ``hierarchical`` selects the parent-memory action graph (True) or the flat
    comparator with the same emission law (False).  Generalized EM on the
    penalized observed likelihood; stops when the gain per valid sample is
    below ``tol``, after ``max_iter`` M-steps, or (reverting) on a decrease."""
    if not tracks:
        raise ValueError("no training tracks")
    if max_iter < 0 or n_anchors < 1:
        raise ValueError("max_iter >= 0 and n_anchors >= 1 required")
    meta = _label_meta()
    flights = [_prepare(w, ph) for w, ph in tracks]
    n_valid = sum(f.n_valid for f in flights)
    if n_valid == 0:
        raise ValueError("no valid (finite) AR(2) windows in the training tracks")
    finite = np.concatenate([np.asarray(w, dtype=np.float64).ravel() for w, _ in tracks])
    finite = finite[np.isfinite(finite)]

    if initial is None:
        grammar = ug.build_grammar(hierarchical)
        params = _init_params(flights, meta, n_anchors)
        if split_response:
            # A faster transient attractor is an initialization, not a fixed
            # step response. All eight extra coefficients enter the same M-step.
            params = ResponseParams(
                F=np.stack((params.F, np.full(N_MODES, 0.4))),
                K=np.stack((params.K, np.full(N_MODES, 0.25))),
                idle=params.idle,
                hover=params.hover,
                offset=params.offset,
                sigma=params.sigma,
                log_pi=params.log_pi,
            )
        history: list[float] = []
        trace: list[dict[str, float]] = []
    else:
        if (
            initial.hierarchical != hierarchical
            or len(initial.params.log_pi) != n_anchors
            or (initial.params.F.ndim == 2) != split_response
            or not initial.history
            or len(initial.history) - 1 > max_iter
        ):
            raise ValueError("restart has an incompatible grammar, target law or iteration budget")
        grammar, params = initial.grammar, initial.params
        # Recompute the final E-step under the current data and priors rather
        # than trusting serialized sufficient statistics.
        history = list(initial.history[:-1])
        trace = list(initial.fit_info["trace"][:-1])
    first_iteration = len(history)
    prev: tuple[ug.Grammar, ResponseParams, _EStep] | None = None
    stop_reason = "max_iter"
    estep = None
    for it in range(first_iteration, max_iter + 1):
        estep = _e_step(flights, grammar, params, meta)
        obj = estep.data_loglik + _response_log_prior(params, meta) + float(grammar.log_prior())
        if not np.isfinite(obj):
            raise FloatingPointError(f"non-finite objective at iteration {it}")
        if (
            initial is not None
            and it == first_iteration
            and not np.isclose(obj, initial.history[-1], rtol=MONOTONE_RTOL, atol=1e-8)
        ):
            raise ValueError("restart likelihood does not match the checkpoint")
        if history and obj < history[-1] - MONOTONE_RTOL * max(1.0, abs(history[-1])):
            grammar, params, estep = prev  # type: ignore[misc]
            stop_reason = "non_monotone"
            break
        history.append(obj)
        ent = -np.sum(estep.resp * np.log(np.clip(estep.resp, 1e-300, None)), axis=1)
        trace.append(
            {
                "objective": obj,
                "data_loglik": estep.data_loglik,
                "per_sample": estep.data_loglik / n_valid,
                "mean_anchor_entropy": float(ent.mean()),
            }
        )
        if len(history) > 1 and (history[-1] - history[-2]) < tol * n_valid:
            stop_reason = "converged"
            break
        if it == max_iter:
            break
        prev = (grammar, params, estep)
        grammar = grammar.m_step(estep.stats)
        params = _m_step(params, meta, estep.mom, estep.resp.sum(axis=0))

    assert estep is not None
    p = params
    fit_info = {
        "n_flights": len(flights),
        "n_samples": int(sum(len(f.phases) for f in flights)),
        "n_valid": int(n_valid),
        "n_anchors": int(n_anchors),
        "n_iter": len(history) - 1,
        "max_iter": int(max_iter),
        "tol_per_sample": float(tol),
        "converged": stop_reason == "converged",
        "stop_reason": stop_reason,
        "trace": trace,
        "responsibilities": estep.resp.tolist(),
        "flight_loglik": estep.flight_loglik.tolist(),
        "label_occupancy": dict(zip(meta.names, estep.occupancy.tolist())),
        "sigma_at_floor": int(np.sum(p.sigma <= SIGMA_FLOOR * (1 + 1e-9))),
        "train_nonpositive_fraction": float(np.mean(finite <= 0.0)) if finite.size else 0.0,
        "spectral_radius": [
            float(np.max(np.abs(np.roots([1.0, -(1.0 + F - K), F]))))
            for F, K in zip(p.F.flat, p.K.flat)
        ],
    }
    return UnifiedResponseModel(
        grammar, p, hierarchical=hierarchical, history=history, fit_info=fit_info
    )


__all__ = ["Draws", "ResponseParams", "UnifiedResponseModel", "fit"]
