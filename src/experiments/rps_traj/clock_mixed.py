"""Clock-locked functional mixed-effects AR law of the four rotor speeds (whole flight).

A probabilistic motion-primitive model: every recording is ONE draw of a
Gaussian functional mixed model in mixer coordinates whose fixed and random
curves live on signed time from the three commanded events, with a stable
coloured AR(2) residual.  The whole-record marginal likelihood (random
effects and the per-flight activity class integrated out) is exact and is
what the fit maximizes.  The only schedule input is the coarse code per 25 Hz
sample (0 before motors-on, 1 motors-on to take-off, 2 take-off to landing or
end, 3 landing to end); event times are the first sample of each code.  No
level, motors-off time, target speed or waveform of a generated flight is
used, and no training waveform is stored.

Observation transform (positive shaft speeds)
---------------------------------------------
Rotor speeds ``w`` (rev/s, MIXER order) are floored at the fixed numerical
floor :data:`W_FLOOR` and mapped by the inverse softplus
``y = g(w) = s log(expm1(w / s))`` (``s`` = :data:`SOFT_SCALE`): ``g(w) ~ w``
at flight speeds, ``g(W_FLOOR) = -1.26`` at rest, no singular value.  Modes
are ``z = MIXER.T y / 4`` (common, roll, pitch, yaw).  The density of the
(floored) speeds is the density of ``z`` times the Jacobian
``prod_r g'(w_r) / 16`` per sample, included in every reported likelihood.
Samples at the floor are scored as values at the floor (a fixed-floor
density, not a censored mass).  Generation is the exact inverse:
``w = s log(1 + exp(y / s))``, read out as 0 where ``w <= W_FLOOR`` (the
same floor), so paths are finite and nonnegative by construction.

Temporal hierarchy (top to bottom)
----------------------------------
1. Flight level, drawn once per recording:
   * activity class ``k`` in ``{0..K-1}`` (``K`` = :data:`N_CLASSES`) with
     weights ``pi``; it sets the cruise innovation variance of EACH mode
     separately (contrast-specific mixture of scales, e.g. yaw-active vs
     roll-active flights);
   * eleven level effects ``b_lev``: common idle and hover, and per rotor
     difference (roll, pitch, yaw) idle trim, hover trim and landing
     fan-out amplitude.  They share a low-rank factor covariance
     ``b_lev = nu + Gamma f + u``, ``f ~ N(0, I_r)`` (``r`` =
     :data:`N_FACTORS`), ``u ~ N(0, diag D)``: landing fan-out, trims and
     levels of all modes co-vary through ``Gamma``; ``D > 0`` keeps
     independent per-effect uncertainty for a new flight even with three
     training flights (boundary-avoiding penalty, below);
   * two unit shape amplitudes of the collective (take-off and landing
     transient), ``N(0, 1)``, independent of the levels (factorized).
2. Event level, deterministic given the schedule: fixed C1 activations and
   cubic B-spline bases on signed time ``tau_e`` from motors-on, take-off and
   landing (:data:`BASES`).  The collective has high-resolution bases
   (0.2 s knots, 0.08 s at the landing mark: the landing mark is the 75 %
   drop, so its window starts 4 s BEFORE the mark and covers DREGON's
   1-2-sample cut as well as the ~3 s MD2 shelf and rundown, to +12 s);
   rotor differences get a low-resolution take-off basis only.  Mean of mode
   ``m``::

       mu_m(t) = X_m(t) beta_m + sum_j b_j c_j(t),
       c_j(t)  = a_j(t) + sum_l B_l(t) lambda_l          (column of effect j)

   ``a_j`` is the fixed plateau activation of effect ``j`` (idle: on from
   motors-on +0.5..4 s until take-off +0..1 s; hover: take-off +0..3 s until
   landing -0.6..+0.4 s; none for the shape amplitudes and fan-out) and
   ``lambda_l`` learned low-resolution loadings: how the random idle level
   enters start-up and transfers to the near-stop shelf after landing
   (``idle_on``, ``idle_la``), how the hover level enters/leaves
   (``hover_to``, ``hover_la``), the idle-trim transfer to near stop
   (``trim_la``, shared by roll/pitch/yaw), the landing fan-out profile
   (``fanout``, shared by roll/pitch/yaw: low rank) and the two collective
   shape profiles.  Trim transfers idle -> cruise through ``Gamma``.
3. Sample level: per mode ``e_t = phi1 e_{t-1} + phi2 e_{t-2} + eps_t``
   started at rest (``e_{-1} = e_{-2} = 0``), ``(phi1, phi2)`` inside the
   AR(2) stability triangle (margin :data:`STABILITY_MARGIN`),
   ``eps_t ~ N(0, sigma^2_{g(t), m})`` with clock-defined groups
   (:data:`GROUP_NAMES`: rest, start-up < 6 s, standby, take-off -0.5..4 s,
   landing -2..+2 s, rundown, cruise[k]); DREGON's command-only recordings
   (``measured=False``) get their own start-up/standby groups so a
   synthetic command idle is not equated with a measured waveform; every
   other parameter is shared with the measured flights.  A noise floor
   :data:`SIGMA_FLOOR` lets quiet standby/rest be (almost) noiseless.

An ordered within-flight manoeuvre process (Markov switching of the cruise
scale over windows) is NOT included: with continuous flight-level random
effects the exact marginal would need a Gaussian sum over all switching
paths.  The flight-level class keeps the likelihood exact; within-flight
cruise heterogeneity is carried by the AR residual only.

Likelihood
----------
With ``U = [X, activations, loading bases, z_m]`` and ``A`` the unit
lower-triangular AR filter (``det A = 1``), ``A z_m`` given ``b`` and ``k`` is
Gaussian with diagonal variances, so per flight and class::

    log p(z | k) = -1/2 [ sum_t log 2 pi sigma_t^2 + log|Psi| + log|P|
                          + a' G a - s' P^-1 s ]
    P = Psi^-1 + sum_m M_m' G_m M_m,   G_m = sum_g H_g(phi_m) / sigma^2_{g,m,k}

(Woodbury over all 15 random coordinates jointly), with ``H_g(phi)`` built
from per-group lagged Gram matrices of ``U`` computed ONCE from the data.
Flight log-likelihood ``log sum_k pi_k p(z | k)`` + Jacobian; the fitted
objective adds the log-priors (penalties) below; both are reported.

Fitting: ECM on the exact penalized marginal likelihood
-------------------------------------------------------
E-step: joint Gaussian posterior of ``(b, f)`` per flight and class and
class responsibilities.  CM-steps, each the exact maximizer of the expected
complete-data penalized log-likelihood over its block with the others
fixed: (1) ``beta`` and every loading ``lambda`` jointly (one linear system,
second-difference + ridge penalties); (2) ``nu``, ``Gamma`` per effect
(ridge) then ``D`` (penalty ``-phi0 / (2 D)``, boundary-avoiding: ``D``
cannot collapse; MAP ``(resid + phi0) / N``); (3) noise variances
(inverse-gamma priors, floor), cruise class variances shrunk toward a
learned per-mode scale (``IG(a_c, a_c c_m)``, so an empty class stays near
the pooled level) and that scale; (4) ``(phi1, phi2)`` per mode, a convex
QP on the stability triangle solved exactly; (5) ``pi`` (Dirichlet MAP).
The objective is therefore non-decreasing; it is evaluated after every
E-step, a decrease beyond :data:`MONOTONE_RTOL` reverts the step and stops
(``stop_reason='non_monotone'``), otherwise stop when the gain per sample is
below ``tol`` or after ``max_iter`` iterations.

Initialization (deterministic, training tracks only): plateau medians of
each flight give point levels; ``beta`` by penalized least squares with the
points fixed; collective shape loadings and the fan-out loading from the
leading principal direction of the per-flight residual projections on the
low-resolution bases; ``(phi1, phi2)`` from the point residuals; flights
ranked by cruise residual energy split into the initial classes;
``nu, Gamma, D`` by principal factors of the point levels.

Parameters (counted in :attr:`ClockMixedModel.n_params`; see
:meth:`ClockMixedModel.param_breakdown`): collective mean 62, rotor
difference mean 3 x 4, loadings 59, level means 11, factor loadings
``11 r``, unique variances 11, AR 8, noise ``4 x`` observed groups (<= 8),
cruise classes ``4 K`` + 4 scales, class weights ``K - 1``: 230 for
``K = 2, r = 2`` with command-only flights, 222 without.

Limitations: activations have fixed relative widths (loadings correct the
average and random shape only within the low-resolution windows); the
floor is a density at the floor, not censoring; within-flight activity
switching is absent (above); event times are the code changes, so an event
before the record start is treated as long past and a missing landing as
never reached.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
from scipy import linalg
from scipy.interpolate import BSpline
from scipy.signal import lfilter
from scipy.special import logsumexp

from tracking.rotors import MIXER, NUM_ROTORS

FAMILY = "clock_mixed"
VERSION = 1
FS = 25.0
N_MODES = 4
MODE_NAMES = ("common", "roll", "pitch", "yaw")
EVENTS = ("motors_on", "takeoff", "landing")

# ── observation transform ──
SOFT_SCALE = 1.0  #: rev/s, softplus scale
W_FLOOR = 0.25  #: rev/s, fixed numerical floor of observed and generated speeds
LOG_DET_MODES = -np.log(16.0)  #: log|det(MIXER.T / 4)| per sample

# ── clocks ──
#: Cubic B-spline breakpoints (s, signed time from the event); the two
#: boundary functions at each end are dropped, so every basis function
#: vanishes with its slope at the window edges.
BASES: dict[str, tuple[str, tuple[float, ...]]] = {
    "on_hr": (
        "motors_on",
        (
            -2.0,
            -1.0,
            -0.5,
            -0.2,
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
            1.2,
            1.4,
            1.6,
            1.8,
            2.0,
            2.2,
            2.4,
            2.7,
            3.0,
            3.5,
            4.0,
            5.0,
            6.0,
        ),
    ),
    "to_hr": (
        "takeoff",
        (
            -1.0,
            -0.5,
            -0.2,
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
            1.2,
            1.4,
            1.6,
            1.8,
            2.0,
            2.5,
            3.0,
            4.0,
            5.0,
        ),
    ),
    "la_hr": (
        "landing",
        (
            -4.0,
            -3.0,
            -2.0,
            -1.5,
            -1.0,
            -0.6,
            -0.3,
            -0.16,
            -0.08,
            0.0,
            0.08,
            0.16,
            0.3,
            0.6,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            5.0,
            6.0,
            8.0,
            10.0,
            12.0,
        ),
    ),
    "on_lr": ("motors_on", (-1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0)),
    "to_lr": ("takeoff", (-1.0, 0.0, 1.0, 2.0, 3.0, 5.0)),
    "la_lr": ("landing", (-4.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0, 5.0, 7.5, 10.0, 12.0)),
}
#: Fixed C1 activations ``(event, start, end)`` (s): ramps 0 -> 1 over ``[start, end]``.
REST_OFF = ("motors_on", -0.2, 0.2)
REST_ON = ("landing", 10.0, 12.0)
IDLE_ON = ("motors_on", 0.5, 4.0)
IDLE_OFF = ("takeoff", 0.0, 1.0)
HOVER_ON = ("takeoff", 0.0, 3.0)
HOVER_OFF = ("landing", -0.6, 0.4)
#: Noise-group clock limits (s).
STARTUP_S = 6.0
TAKEOFF_S = (-0.5, 4.0)
LANDING_S = (-2.0, 2.0)

REST, STARTUP, STANDBY, TAKEOFF, LANDING, RUNDOWN, STARTUP_CMD, STANDBY_CMD, CRUISE = range(9)
N_BASE = 9
GROUP_NAMES = (
    "rest",
    "startup",
    "standby",
    "takeoff",
    "landing",
    "rundown",
    "startup_cmd",
    "standby_cmd",
)

# ── structure ──
N_CLASSES = 2
N_FACTORS = 2
#: Loadings: name -> low-resolution basis block.
LOADINGS: dict[str, str] = {
    "idle_on": "on_lr",
    "idle_la": "la_lr",
    "hover_to": "to_lr",
    "hover_la": "la_lr",
    "takeoff_shape": "to_lr",
    "landing_shape": "la_lr",
    "trim_la": "la_lr",
    "fanout": "la_lr",
}


@dataclass(frozen=True)
class _Effect:
    name: str
    act: str | None  # fixed activation column
    loads: tuple[str, ...]  # learned loadings
    level: bool  # learned mean, factor covariance; else unit N(0, 1)


_COMMON = (
    _Effect("idle", "idle", ("idle_on", "idle_la"), True),
    _Effect("hover", "hover", ("hover_to", "hover_la"), True),
    _Effect("takeoff_shape", None, ("takeoff_shape",), False),
    _Effect("landing_shape", None, ("landing_shape",), False),
)
_DIFF = (
    _Effect("idle", "idle", ("trim_la",), True),
    _Effect("hover", "hover", (), True),
    _Effect("fanout", None, ("fanout",), True),
)
MODE_X = (("rest", "on_hr", "to_hr", "la_hr"), ("to_lr",), ("to_lr",), ("to_lr",))
MODE_EFFECTS = (_COMMON, _DIFF, _DIFF, _DIFF)

# ── priors / penalties (all in transformed rev/s) ──
SIGMA_FLOOR = 0.02
SIG_A, SIG_B = 1.0, 0.05**2  #: inverse-gamma prior of every non-cruise innovation variance
CRUISE_A = 3.0  #: IG shape of the class cruise variances around the learned per-mode scale
SPLINE_D2, SPLINE_RIDGE = 1e-2, 1e-3
LOAD_D2, LOAD_RIDGE = 1e-2, 1e-2
GAMMA_RIDGE = 1e-3
D_PRIOR = 0.5**2  #: boundary-avoiding penalty -D_PRIOR / (2 D) on unique level variances
PI_CONC = 2.0
STABILITY_MARGIN = 1e-3
MONOTONE_RTOL = 1e-9
_LAG_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


# ─── transform ────────────────────────────────────────────────────────────────


def to_obs(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Floored speeds -> ``(y, log g'(w))``, both shaped like ``w``."""
    x = np.maximum(np.asarray(w, dtype=np.float64), W_FLOOR) / SOFT_SCALE
    log1m = np.log(-np.expm1(-x))  # log(1 - e^-x)
    return SOFT_SCALE * (x + log1m), -log1m


def from_obs(y: np.ndarray) -> np.ndarray:
    """Exact inverse with the floor read out as 0."""
    w = SOFT_SCALE * np.logaddexp(0.0, np.asarray(y, dtype=np.float64) / SOFT_SCALE)
    return np.where(w > W_FLOOR, w, 0.0)


# ─── clocks and design ────────────────────────────────────────────────────────


def _check_codes(codes: Any, T: int | None = None) -> np.ndarray:
    c = np.asarray(codes)
    if c.ndim != 1 or len(c) == 0:
        raise ValueError("schedule codes must be a non-empty 1-D array")
    if not np.isin(c, (0, 1, 2, 3)).all():
        raise ValueError("schedule codes must be in {0, 1, 2, 3}")
    if (np.diff(c.astype(np.int64)) < 0).any():
        raise ValueError("schedule codes must be non-decreasing")
    if T is not None and len(c) != T:
        raise ValueError(f"schedule has {len(c)} samples, speeds {T}")
    return c.astype(np.int8)


def event_clocks(codes: Any) -> np.ndarray:
    """``(3, T)`` signed seconds from motors-on, take-off, landing; +inf if the
    event precedes the record, -inf if it never occurs."""
    c = _check_codes(codes)
    t = np.arange(len(c), dtype=np.float64)
    out = np.empty((3, len(c)))
    for e, level in enumerate((1, 2, 3)):
        hit = np.flatnonzero(c >= level)
        out[e] = np.inf if c[0] >= level else ((t - hit[0]) / FS if len(hit) else -np.inf)
    return out


def _bspline(tau: np.ndarray, breaks: tuple[float, ...]) -> np.ndarray:
    b = np.asarray(breaks, dtype=np.float64)
    knots = np.r_[[b[0]] * 3, b, [b[-1]] * 3]
    n = len(knots) - 4
    out = np.zeros((len(tau), n))
    inside = np.isfinite(tau) & (tau >= b[0]) & (tau <= b[-1])
    if inside.any():
        out[inside] = BSpline(knots, np.eye(n), 3, extrapolate=False)(tau[inside])
    return np.nan_to_num(out)[:, 2:-2]


def _ramp(tau: np.ndarray, spec: tuple[str, float, float]) -> np.ndarray:
    _, a, b = spec
    u = np.clip((tau - a) / (b - a), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _blocks(
    codes: Any, measured: bool = True
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Design blocks ``name -> (T, n)``, base noise group per sample, clocks."""
    c = _check_codes(codes)
    tau = event_clocks(c)
    ev: dict[str, np.ndarray] = dict(zip(EVENTS, tau))

    def ramp(spec):
        return _ramp(ev[spec[0]], spec)

    blocks = {
        "rest": (1.0 - ramp(REST_OFF) + ramp(REST_ON))[:, None],
        "idle": (ramp(IDLE_ON) * (1.0 - ramp(IDLE_OFF)))[:, None],
        "hover": (ramp(HOVER_ON) * (1.0 - ramp(HOVER_OFF)))[:, None],
    }
    for name, (event, breaks) in BASES.items():
        blocks[name] = _bspline(ev[event], breaks)
    on, to, la = tau
    g = np.full(len(c), CRUISE, dtype=np.int64)
    ground = c == 1
    g[ground] = np.where(on[ground] < STARTUP_S, STARTUP, STANDBY)
    if not measured:
        g[g == STARTUP], g[g == STANDBY] = STARTUP_CMD, STANDBY_CMD
    g[(to >= TAKEOFF_S[0]) & (to < TAKEOFF_S[1])] = TAKEOFF
    near = la >= LANDING_S[0]
    g[near] = np.where(la[near] < LANDING_S[1], LANDING, RUNDOWN)
    g[c == 0] = REST
    return blocks, g, tau


@dataclass(frozen=True)
class _ModeLayout:
    order: tuple[str, ...]  # U column blocks (then the data column)
    cols: dict[str, np.ndarray]
    p: int
    x: np.ndarray  # U columns of X
    beta: slice  # theta slice of beta_m
    y: int
    eff: np.ndarray  # global random-effect indices of this mode
    f: np.ndarray  # U column of each effect's activation or -1
    loads: tuple[tuple[int, slice, np.ndarray], ...]  # (local effect, theta slice, U columns)


@dataclass(frozen=True)
class _Layout:
    modes: tuple[_ModeLayout, ...]
    sizes: dict[str, int]
    load: dict[str, slice]
    n_theta: int
    level: np.ndarray  # global indices of level effects (factor covariance)
    shape: np.ndarray  # global indices of unit shape effects
    factor: np.ndarray  # global indices of the latent factors f
    q: int
    names: tuple[str, ...]


def _build_layout(n_factors: int = N_FACTORS) -> _Layout:
    sizes = {"rest": 1, "idle": 1, "hover": 1, **{k: len(br) - 2 for k, (_, br) in BASES.items()}}
    off, betas = 0, []
    for m in range(N_MODES):
        n = sum(sizes[b] for b in MODE_X[m])
        betas.append(slice(off, off + n))
        off += n
    load = {}
    for name, blk in LOADINGS.items():
        load[name] = slice(off, off + sizes[blk])
        off += sizes[blk]
    modes, level, shape, names, g = [], [], [], [], 0
    for m in range(N_MODES):
        effects = MODE_EFFECTS[m]
        acts = [e.act for e in effects if e.act]
        bls = list(dict.fromkeys(LOADINGS[k] for e in effects for k in e.loads))
        order = (*MODE_X[m], *acts, *bls)
        if len(set(order)) != len(order):
            raise AssertionError("a block appears twice in one mode")
        cols, c = {}, 0
        for b in order:
            cols[b] = np.arange(c, c + sizes[b])
            c += sizes[b]
        eff = np.arange(g, g + len(effects))
        for e, gi in zip(effects, eff):
            (level if e.level else shape).append(gi)
            names.append(f"{MODE_NAMES[m]}.{e.name}")
        g += len(effects)
        modes.append(
            _ModeLayout(
                order=order,
                cols=cols,
                p=c + 1,
                x=np.concatenate([cols[b] for b in MODE_X[m]]),
                beta=betas[m],
                y=c,
                eff=eff,
                f=np.array([cols[e.act][0] if e.act else -1 for e in effects]),
                loads=tuple(
                    (j, load[k], cols[LOADINGS[k]]) for j, e in enumerate(effects) for k in e.loads
                ),
            )
        )
    factor = np.arange(g, g + n_factors)
    names += [f"factor{r}" for r in range(n_factors)]
    return _Layout(
        tuple(modes),
        sizes,
        load,
        off,
        np.array(level),
        np.array(shape),
        factor,
        g + n_factors,
        tuple(names),
    )


def _u_matrix(L: _ModeLayout, blocks: dict[str, np.ndarray], zm: np.ndarray) -> np.ndarray:
    return np.hstack([blocks[b] for b in L.order] + [zm[:, None]])


def _columns(
    L: _ModeLayout, blocks: dict[str, np.ndarray], theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``X`` and the random-effect columns ``c_j`` (T, q_m) of one mode."""
    U = np.hstack([blocks[b] for b in L.order])
    C = np.zeros((len(U), len(L.f)))
    for j, fc in enumerate(L.f):
        if fc >= 0:
            C[:, j] = U[:, fc]
    for j, ts, us in L.loads:
        C[:, j] += U[:, us] @ theta[ts]
    return U[:, L.x], C


def _penalty(lay: _Layout) -> np.ndarray:
    def pen(s, d2, ridge):
        D = np.diff(np.eye(s), 2, axis=0)
        return d2 * D.T @ D + ridge * np.eye(s)

    R = np.zeros((lay.n_theta, lay.n_theta))
    for m, L in enumerate(lay.modes):
        o = L.beta.start
        for b in MODE_X[m]:
            s = lay.sizes[b]
            R[o : o + s, o : o + s] = pen(s, 0.0 if b == "rest" else SPLINE_D2, SPLINE_RIDGE)
            o += s
    for sl in lay.load.values():
        s = sl.stop - sl.start
        R[sl, sl] = pen(s, LOAD_D2, LOAD_RIDGE)
    return R


# ─── data ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Flight:
    H: tuple[np.ndarray, ...]  # per mode (6, N_BASE, p, p) lagged Gram matrices of U per group
    n: np.ndarray  # (N_BASE,) samples per group
    T: int
    logjac: float
    measured: bool


def _lag_grams(U: np.ndarray, groups: np.ndarray) -> np.ndarray:
    T, p = U.shape
    lags = [U, np.vstack([np.zeros((1, p)), U[:-1]]), np.vstack([np.zeros((2, p)), U[:-2]])[:T]]
    H = np.zeros((len(_LAG_PAIRS), N_BASE, p, p))
    for g in range(N_BASE):
        idx = groups == g
        if idx.any():
            sub = [lag[idx] for lag in lags]
            for q, (a, b) in enumerate(_LAG_PAIRS):
                H[q, g] = sub[a].T @ sub[b]
    return H


def _prepare(w: Any, codes: Any, measured: bool, lay: _Layout):
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != NUM_ROTORS:
        raise ValueError(f"speeds must be (4, T), got {w.shape}")
    if not np.isfinite(w).all():
        raise ValueError("speeds must be finite (bridge gaps before fitting)")
    c = _check_codes(codes, w.shape[1])
    y, lj = to_obs(w)
    z = MIXER.T @ y / NUM_ROTORS
    blocks, groups, tau = _blocks(c, measured)
    H = tuple(_lag_grams(_u_matrix(L, blocks, z[m]), groups) for m, L in enumerate(lay.modes))
    f = _Flight(
        H,
        np.bincount(groups, minlength=N_BASE),
        w.shape[1],
        float(lj.sum() + w.shape[1] * LOG_DET_MODES),
        bool(measured),
    )
    return f, z, blocks, groups, tau, c


def _whiten(H: np.ndarray, phi: np.ndarray) -> np.ndarray:
    fb = np.array([1.0, -phi[0], -phi[1]])
    G = np.zeros(H.shape[1:])
    for q, (a, b) in enumerate(_LAG_PAIRS):
        G += fb[a] * fb[b] * (H[q] if a == b else H[q] + H[q].transpose(0, 2, 1))
    return G


# ─── parameters ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Params:
    theta: np.ndarray  # beta of every mode, then loadings (see _Layout)
    nu: np.ndarray  # (11,) level means
    gamma: np.ndarray  # (11, r) factor loadings of the levels
    D: np.ndarray  # (11,) unique level variances
    phi: np.ndarray  # (4, 2) AR(2)
    sig2: np.ndarray  # (8 + K, 4) innovation variances: GROUP_NAMES then cruise classes
    cruise_scale: np.ndarray  # (4,) per-mode centre of the class cruise variances
    log_pi: np.ndarray  # (K,)


def _rows(k: int) -> np.ndarray:
    return np.r_[np.arange(CRUISE), CRUISE + k]


def _psi(p: Params, lay: _Layout) -> tuple[np.ndarray, np.ndarray]:
    """Prior mean and covariance of the joint random vector ``(b, f)``."""
    mean, Psi = np.zeros(lay.q), np.zeros((lay.q, lay.q))
    L, F = lay.level, lay.factor
    mean[L] = p.nu
    Psi[np.ix_(L, L)] = p.gamma @ p.gamma.T + np.diag(p.D)
    Psi[np.ix_(L, F)] = p.gamma
    Psi[np.ix_(F, L)] = p.gamma.T
    Psi[F, F] = 1.0
    Psi[lay.shape, lay.shape] = 1.0
    return mean, Psi


def _aM(L: _ModeLayout, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Residual ``e = U a(b)``, ``a(b) = a_c + M b`` over the mode's effects."""
    a = np.zeros(L.p)
    a[L.x] = -theta[L.beta]
    a[L.y] = 1.0
    M = np.zeros((L.p, len(L.f)))
    for j, fc in enumerate(L.f):
        if fc >= 0:
            M[fc, j] = -1.0
    for j, ts, us in L.loads:
        M[us, j] -= theta[ts]
    return a, M


def _G(Gw: np.ndarray, sig2: np.ndarray, k: int, m: int) -> np.ndarray:
    return np.tensordot(1.0 / sig2[_rows(k), m], Gw, axes=1)


# ─── E-step ───────────────────────────────────────────────────────────────────


@dataclass
class _Post:
    loglik: np.ndarray  # (n,) flight log-likelihood (classes and effects integrated)
    ll: np.ndarray  # (n, K) per class
    resp: np.ndarray  # (n, K)
    mean: np.ndarray  # (n, K, q) posterior mean of (b, f)
    cov: np.ndarray  # (n, K, q, q)


def _e_step(flights: list[_Flight], Gw: list[list[np.ndarray]], p: Params, lay: _Layout) -> _Post:
    n, K, q = len(flights), len(p.log_pi), lay.q
    prior_mean, Psi = _psi(p, lay)
    cPsi = linalg.cho_factor(Psi, lower=True)
    Psi_inv = linalg.cho_solve(cPsi, np.eye(q))
    logdet_psi = 2.0 * np.log(np.diag(cPsi[0])).sum()
    aM = [_aM(L, p.theta) for L in lay.modes]
    ll = np.zeros((n, K))
    mean, cov = np.zeros((n, K, q)), np.zeros((n, K, q, q))
    for i, f in enumerate(flights):
        for k in range(K):
            P, s, quad, noise = Psi_inv.copy(), np.zeros(q), 0.0, 0.0
            for m, L in enumerate(lay.modes):
                G = _G(Gw[i][m], p.sig2, k, m)
                a_c, M = aM[m]
                a_nu = a_c + M @ prior_mean[L.eff]
                Ga = G @ a_nu
                P[np.ix_(L.eff, L.eff)] += M.T @ G @ M
                s[L.eff] -= M.T @ Ga
                quad += a_nu @ Ga
                noise += f.n @ np.log(2.0 * np.pi * p.sig2[_rows(k), m])
            cP = linalg.cho_factor(P, lower=True)
            delta = linalg.cho_solve(cP, s)
            logdet_p = 2.0 * np.log(np.diag(cP[0])).sum()
            ll[i, k] = -0.5 * (noise + quad - s @ delta + logdet_psi + logdet_p) + f.logjac
            mean[i, k] = prior_mean + delta
            cov[i, k] = linalg.cho_solve(cP, np.eye(q))
    joint = ll + p.log_pi
    tot = cast(np.ndarray, logsumexp(joint, axis=1))
    return _Post(tot, ll, np.exp(joint - tot[:, None]), mean, cov)


# ─── priors ───────────────────────────────────────────────────────────────────


def _log_prior(p: Params, R: np.ndarray) -> float:
    lp = -0.5 * p.theta @ R @ p.theta
    lp += -0.5 * GAMMA_RIDGE * np.sum(p.gamma**2) - 0.5 * np.sum(D_PRIOR / p.D)
    s = p.sig2[:CRUISE]
    lp += np.sum(-(SIG_A + 1.0) * np.log(s) - SIG_B / s)
    c, sk = p.cruise_scale, p.sig2[CRUISE:]
    lp += np.sum(
        CRUISE_A * np.log(CRUISE_A * c) - (CRUISE_A + 1.0) * np.log(sk) - CRUISE_A * c / sk
    )
    lp += (PI_CONC - 1.0) * np.sum(p.log_pi)
    return float(lp)


# ─── CM-steps ─────────────────────────────────────────────────────────────────


def _solve_spd(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return linalg.cho_solve(linalg.cho_factor(A, lower=True), b)
    except linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def _cm_theta(
    flights, Gw, p: Params, post: _Post, lay: _Layout, R: np.ndarray, free: np.ndarray
) -> np.ndarray:
    """Exact joint maximizer over ``beta`` and the loadings (entries ``free``)."""
    A, c = np.zeros((lay.n_theta,) * 2), np.zeros(lay.n_theta)
    K = len(p.log_pi)
    for i in range(len(flights)):
        for k in range(K):
            w = post.resp[i, k]
            if w < 1e-14:
                continue
            for m, L in enumerate(lay.modes):
                G = _G(Gw[i][m], p.sig2, k, m)
                mu = post.mean[i, k, L.eff]
                S = post.cov[i, k][np.ix_(L.eff, L.eff)] + np.outer(mu, mu)
                feff = np.flatnonzero(L.f >= 0)
                fcol = L.f[feff]
                bx = L.beta
                A[bx, bx] += w * G[np.ix_(L.x, L.x)]
                c[bx] += w * (G[np.ix_(L.x, fcol)] @ mu[feff] - G[L.x, L.y])
                for j, ts, us in L.loads:
                    blk = w * mu[j] * G[np.ix_(L.x, us)]
                    A[bx, ts] += blk
                    A[ts, bx] += blk.T
                    for j2, ts2, us2 in L.loads:
                        A[ts, ts2] += w * S[j, j2] * G[np.ix_(us, us2)]
                    c[ts] += w * (G[np.ix_(us, fcol)] @ S[j, feff] - G[us, L.y] * mu[j])
    H = A + R
    theta = p.theta.copy()
    fx = ~free
    rhs = -c[free] - H[np.ix_(free, fx)] @ theta[fx]
    theta[free] = _solve_spd(H[np.ix_(free, free)], rhs)
    return theta


def _saa(lay: _Layout, theta: np.ndarray, post: _Post) -> list[list[list[np.ndarray]]]:
    """``E[a a']`` per flight, mode and class at ``theta``."""
    aM = [_aM(L, theta) for L in lay.modes]
    out = []
    for i in range(post.mean.shape[0]):
        row = []
        for m, L in enumerate(lay.modes):
            a_c, M = aM[m]
            per_k = []
            for k in range(post.mean.shape[1]):
                a = a_c + M @ post.mean[i, k, L.eff]
                per_k.append(np.outer(a, a) + M @ post.cov[i, k][np.ix_(L.eff, L.eff)] @ M.T)
            row.append(per_k)
        out.append(row)
    return out


def _cm_sigma(flights, Gw, p: Params, post: _Post, saa) -> tuple[np.ndarray, np.ndarray]:
    K = len(p.log_pi)
    ess, cnt = np.zeros((CRUISE + K, N_MODES)), np.zeros((CRUISE + K, N_MODES))
    for i, f in enumerate(flights):
        for k in range(K):
            w = post.resp[i, k]
            rows = _rows(k)
            for m in range(N_MODES):
                for g in np.flatnonzero(f.n):
                    ess[rows[g], m] += w * np.vdot(Gw[i][m][g], saa[i][m][k])
                    cnt[rows[g], m] += w * f.n[g]
    sig2 = np.empty_like(ess)
    sig2[:CRUISE] = (ess[:CRUISE] + 2.0 * SIG_B) / (cnt[:CRUISE] + 2.0 * SIG_A + 2.0)
    sig2[CRUISE:] = (ess[CRUISE:] + 2.0 * CRUISE_A * p.cruise_scale) / (
        cnt[CRUISE:] + 2.0 * CRUISE_A + 2.0
    )
    sig2 = np.maximum(sig2, SIGMA_FLOOR**2)
    return sig2, K / np.sum(1.0 / sig2[CRUISE:], axis=0)


def _stable_vertices() -> np.ndarray:
    v = np.array([[0.0, 1.0], [-2.0, -1.0], [2.0, -1.0]])  # counter-clockwise
    centre = v.mean(axis=0)
    return centre + (1.0 - STABILITY_MARGIN) * (v - centre)


def _inside(x: np.ndarray, V: np.ndarray) -> bool:
    for a, b in zip(V, np.roll(V, -1, axis=0)):
        if (b[0] - a[0]) * (x[1] - a[1]) - (b[1] - a[1]) * (x[0] - a[0]) < -1e-14:
            return False
    return True


def _ar2_qp(Kmat: np.ndarray) -> np.ndarray:
    """``argmin (1, -phi)' K (1, -phi)`` over the (shrunk) AR(2) stability triangle."""
    Q, g = Kmat[1:, 1:], Kmat[0, 1:]
    if np.trace(Q) <= 0.0:
        return np.zeros(2)

    def obj(x):
        return -2.0 * g @ x + x @ Q @ x

    V = _stable_vertices()
    try:
        x = np.linalg.solve(Q, g)
        if np.isfinite(x).all() and _inside(x, V):
            return x
    except np.linalg.LinAlgError:
        pass
    best, best_f = V[0], np.inf
    for a, b in zip(V, np.roll(V, -1, axis=0)):
        d = b - a
        dQd = d @ Q @ d
        s = np.clip((g @ d - d @ Q @ a) / dQd, 0.0, 1.0) if dQd > 0 else 0.0
        x = a + s * d
        for cand in (x, a, b):
            fx = obj(cand)
            if fx < best_f:
                best, best_f = cand, fx
    return np.array(best, dtype=np.float64)


def _cm_phi(flights, p: Params, sig2: np.ndarray, post: _Post, saa) -> np.ndarray:
    phi = np.zeros((N_MODES, 2))
    for m in range(N_MODES):
        Km = np.zeros((3, 3))
        for i, f in enumerate(flights):
            for k in range(len(p.log_pi)):
                w = post.resp[i, k]
                wg = 1.0 / sig2[_rows(k), m]
                for q, (a, b) in enumerate(_LAG_PAIRS):
                    t = w * np.vdot(np.tensordot(wg, f.H[m][q], axes=1), saa[i][m][k])
                    Km[a, b] += t
                    if a != b:
                        Km[b, a] += t
        phi[m] = _ar2_qp(Km)
    return phi


def _cm_levels(p: Params, post: _Post, lay: _Layout) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(nu, Gamma)`` per level by ridge regression on ``(1, f)``, then ``D``."""
    Lv, F = lay.level, lay.factor
    r = len(F)
    Ehh = np.zeros((1 + r, 1 + r))
    Ehb = np.zeros((1 + r, len(Lv)))
    Ebb = np.zeros(len(Lv))
    for i in range(post.mean.shape[0]):
        for k in range(post.mean.shape[1]):
            w = post.resp[i, k]
            mu = post.mean[i, k]
            S = post.cov[i, k] + np.outer(mu, mu)
            Ehh[0, 0] += w
            Ehh[0, 1:] += w * mu[F]
            Ehh[1:, 0] += w * mu[F]
            Ehh[1:, 1:] += w * S[np.ix_(F, F)]
            Ehb[0] += w * mu[Lv]
            Ehb[1:] += w * S[np.ix_(F, Lv)]
            Ebb += w * np.diag(S)[Lv]
    N = Ehh[0, 0]
    nu, gamma, D = np.empty(len(Lv)), np.empty((len(Lv), r)), np.empty(len(Lv))
    ridge = np.diag(np.r_[0.0, np.ones(r)])
    for j in range(len(Lv)):
        coef = _solve_spd(Ehh + GAMMA_RIDGE * p.D[j] * ridge, Ehb[:, j])
        nu[j], gamma[j] = coef[0], coef[1:]
        resid = max(Ebb[j] - 2.0 * coef @ Ehb[:, j] + coef @ Ehh @ coef, 0.0)
        D[j] = (resid + D_PRIOR) / N
    return nu, gamma, D


def _m_step(flights, Gw, p: Params, post: _Post, lay: _Layout, R: np.ndarray) -> Params:
    theta = _cm_theta(flights, Gw, p, post, lay, R, np.ones(lay.n_theta, dtype=bool))
    nu, gamma, D = _cm_levels(p, post, lay)
    saa = _saa(lay, theta, post)
    sig2, cs = _cm_sigma(flights, Gw, replace(p, theta=theta), post, saa)
    phi = _cm_phi(flights, p, sig2, post, saa)
    K = len(p.log_pi)
    pi = (post.resp.sum(axis=0) + PI_CONC - 1.0) / (len(flights) + K * (PI_CONC - 1.0))
    return Params(theta, nu, gamma, D, phi, sig2, cs, np.log(pi))


# ─── initialization ───────────────────────────────────────────────────────────


def _point_post(b: np.ndarray, resp: np.ndarray) -> _Post:
    n, K = resp.shape
    mean = np.repeat(b[:, None, :], K, axis=1)
    return _Post(
        np.zeros(n), np.zeros((n, K)), resp, mean, np.zeros((n, K, b.shape[1], b.shape[1]))
    )


def _project(f: _Flight, m: int, L: _ModeLayout, theta: np.ndarray, b: np.ndarray, block: str):
    """Least-squares coefficients of the point residual on a basis block, or None."""
    UU = f.H[m][0].sum(axis=0)
    a_c, M = _aM(L, theta)
    a = a_c + M @ b[L.eff]
    us = L.cols[block]
    BB = UU[np.ix_(us, us)]
    if np.trace(BB) < 1.0:
        return None
    return _solve_spd(BB + 1e-6 * np.eye(len(us)), UU[us] @ a)


def _leading(Gm: np.ndarray, centre: bool) -> tuple[np.ndarray, np.ndarray]:
    """Leading direction (scaled to the score RMS) and unit-RMS scores of rows ``Gm``."""
    X = Gm - Gm.mean(axis=0) if centre and len(Gm) > 1 else Gm
    _, sv, vt = np.linalg.svd(X, full_matrices=False)
    v = vt[0]
    s = X @ v
    scale = np.sqrt(np.mean(s**2)) if np.mean(s**2) > 1e-12 else 1.0
    return v * scale, s / scale


def _initialize(
    flights, zs, blockss, groupss, taus, codess, lay: _Layout, R: np.ndarray, K: int
) -> Params:
    n, q = len(flights), lay.q
    # 1. point levels: plateau medians per flight and mode
    lev = np.full((n, N_MODES, 2), np.nan)
    for i in range(n):
        on, to, la = taus[i]
        c = codess[i]
        idle = (c == 1) & (on > 0.5) & (to < -0.2)
        hover = (c == 2) & (to > 3.0) & (la < LANDING_S[0])
        if idle.sum() >= 3:
            lev[i, :, 0] = np.median(zs[i][:, idle], axis=1)
        if hover.sum() >= 3:
            lev[i, :, 1] = np.median(zs[i][:, hover], axis=1)
    powered = np.concatenate([z[:, c >= 1] for z, c in zip(zs, codess)], axis=1)
    fallback = np.median(powered, axis=1) if powered.size else np.zeros(N_MODES)
    for m in range(N_MODES):
        for e in range(2):
            col = lev[:, m, e]
            col[np.isnan(col)] = (
                np.nanmean(col) if np.isfinite(col).any() else (fallback[m] if m == 0 else 0.0)
            )
    b = np.zeros((n, q))
    for m, L in enumerate(lay.modes):
        b[:, L.eff[0]], b[:, L.eff[1]] = lev[:, m, 0], lev[:, m, 1]
    p = Params(
        theta=np.zeros(lay.n_theta),
        nu=np.zeros(len(lay.level)),
        gamma=np.zeros((len(lay.level), len(lay.factor))),
        D=np.ones(len(lay.level)),
        phi=np.zeros((N_MODES, 2)),
        sig2=np.ones((CRUISE + K, N_MODES)),
        cruise_scale=np.ones(N_MODES),
        log_pi=np.full(K, -np.log(K)),
    )
    beta_only = np.zeros(lay.n_theta, dtype=bool)
    for L in lay.modes:
        beta_only[L.beta] = True
    uniform = np.full((n, K), 1.0 / K)
    Gw = [[_whiten(H, p.phi[m]) for m, H in enumerate(f.H)] for f in flights]
    theta = _cm_theta(flights, Gw, p, _point_post(b, uniform), lay, R, beta_only)

    # 2. shape loadings, fan-out loading and their point scores
    common = lay.modes[0]
    for name, block, j in (("takeoff_shape", "to_lr", 2), ("landing_shape", "la_lr", 3)):
        rows = [
            (i, g)
            for i in range(n)
            if (g := _project(flights[i], 0, common, theta, b[i], block)) is not None
        ]
        if len(rows) >= 2:
            lam, s = _leading(np.array([g for _, g in rows]), centre=True)
            theta[lay.load[name]] = lam
            for (i, _), si in zip(rows, s):
                b[i, common.eff[j]] = si
        elif rows:
            theta[lay.load[name]] = 0.5 * rows[0][1]
    rows = [
        (i, m, g)
        for m in range(1, N_MODES)
        for i in range(n)
        if (g := _project(flights[i], m, lay.modes[m], theta, b[i], "la_lr")) is not None
    ]
    if rows:
        lam, s = _leading(np.array([g for *_, g in rows]), centre=False)
        theta[lay.load["fanout"]] = lam
        for (i, m, _), si in zip(rows, s):
            b[i, lay.modes[m].eff[2]] = si
    theta = _cm_theta(
        flights, Gw, replace(p, theta=theta), _point_post(b, uniform), lay, R, beta_only
    )
    p = replace(p, theta=theta)

    # 3. AR from the point residuals, then noise with a ranked class split
    point = _point_post(b, uniform)
    saa = _saa(lay, theta, point)
    p = replace(p, phi=_cm_phi(flights, p, p.sig2, point, saa))
    Gw = [[_whiten(H, p.phi[m]) for m, H in enumerate(f.H)] for f in flights]
    energy = np.zeros((n, N_MODES))
    for i, f in enumerate(flights):
        for m in range(N_MODES):
            energy[i, m] = np.vdot(Gw[i][m][CRUISE], saa[i][m][0]) / max(f.n[CRUISE], 1)
    has = np.array([f.n[CRUISE] > 0 for f in flights])
    pooled = energy[has].mean(axis=0) if has.any() else np.ones(N_MODES)
    score = np.where(
        has, np.log(np.maximum(energy, 1e-12) / np.maximum(pooled, 1e-12)).mean(axis=1), 0.0
    )
    rank = np.argsort(np.argsort(score, kind="stable"), kind="stable")
    hard = np.zeros((n, K))
    hard[np.arange(n), np.minimum(rank * K // n, K - 1)] = 1.0
    p = replace(p, cruise_scale=np.maximum(pooled, SIGMA_FLOOR**2))
    sig2, cs = _cm_sigma(
        flights, Gw, p, _point_post(b, hard), _saa(lay, theta, _point_post(b, hard))
    )
    pi = (hard.sum(axis=0) + PI_CONC - 1.0) / (n + K * (PI_CONC - 1.0))
    p = replace(p, sig2=sig2, cruise_scale=cs, log_pi=np.log(pi))

    # 4. principal factors of the point levels
    Bl = b[:, lay.level]
    nu = Bl.mean(axis=0)
    X = Bl - nu
    r = len(lay.factor)
    cov = X.T @ X / n
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1][:r]
    gamma = evecs[:, order] * np.sqrt(np.maximum(evals[order], 0.0))
    D = (np.maximum(np.diag(cov) - np.sum(gamma**2, axis=1), 0.0) * n + D_PRIOR) / n
    return replace(p, nu=nu, gamma=gamma, D=D)


# ─── model ────────────────────────────────────────────────────────────────────


def _config(K: int, r: int) -> dict[str, Any]:
    return {
        "family": FAMILY,
        "version": VERSION,
        "fs": FS,
        "soft_scale": SOFT_SCALE,
        "w_floor": W_FLOOR,
        "sigma_floor": SIGMA_FLOOR,
        "n_classes": K,
        "n_factors": r,
        "bases": {k: [e, list(b)] for k, (e, b) in BASES.items()},
        "activations": {
            "rest_off": list(REST_OFF),
            "rest_on": list(REST_ON),
            "idle_on": list(IDLE_ON),
            "idle_off": list(IDLE_OFF),
            "hover_on": list(HOVER_ON),
            "hover_off": list(HOVER_OFF),
        },
        "groups": {
            "startup_s": STARTUP_S,
            "takeoff_s": list(TAKEOFF_S),
            "landing_s": list(LANDING_S),
        },
        "loadings": dict(LOADINGS),
    }


def _tracks(tracks, measured, lay):
    if len(tracks) == 0:
        raise ValueError("need at least one training track")
    measured = [True] * len(tracks) if measured is None else [bool(v) for v in measured]
    if len(measured) != len(tracks):
        raise ValueError("measured must have one flag per track")
    return [_prepare(w, c, ms, lay) for (w, c), ms in zip(tracks, measured)]


class ClockMixedModel:
    """Fitted clock-locked functional mixed-effects AR law; see the module docstring."""

    def __init__(
        self,
        params: Params,
        observed: np.ndarray,
        *,
        history: list[float] | None = None,
        fit_info: dict[str, Any] | None = None,
    ):
        self.params = params
        self.observed = np.asarray(observed, dtype=bool)  # (8,) non-cruise groups seen in training
        self.history = list(history or [])
        self.fit_info = dict(fit_info or {})
        self.layout = _build_layout(params.gamma.shape[1])
        self.last_clip_fraction = 0.0

    @property
    def n_classes(self) -> int:
        return len(self.params.log_pi)

    def param_breakdown(self) -> dict[str, int]:
        lay, p = self.layout, self.params
        out = {
            "mean_common": lay.modes[0].beta.stop - lay.modes[0].beta.start,
            "mean_rotor_diff": sum(L.beta.stop - L.beta.start for L in lay.modes[1:]),
        }
        out.update({f"loading_{k}": sl.stop - sl.start for k, sl in lay.load.items()})
        out.update(
            {
                "level_means": p.nu.size,
                "level_factor_loadings": p.gamma.size,
                "level_unique_var": p.D.size,
                "ar": p.phi.size,
                "noise_groups": int(self.observed.sum()) * N_MODES,
                "cruise_class_var": self.n_classes * N_MODES,
                "cruise_scale": N_MODES,
                "class_weights": self.n_classes - 1,
            }
        )
        return out

    @property
    def n_params(self) -> int:
        return int(sum(self.param_breakdown().values()))

    # ── likelihood ──
    def loglik(
        self, tracks: list[tuple[np.ndarray, np.ndarray]], measured: list[bool] | None = None
    ) -> np.ndarray:
        """Exact marginal log-density of each whole track (floored speeds, Jacobian included)."""
        prep = _tracks(tracks, measured, self.layout)
        flights = [x[0] for x in prep]
        Gw = [[_whiten(H, self.params.phi[m]) for m, H in enumerate(f.H)] for f in flights]
        return _e_step(flights, Gw, self.params, self.layout).loglik

    # ── generation ──
    def sample(self, schedule: Any, n: int, rng: np.random.Generator) -> np.ndarray:
        """``(n, 4, T)`` nonnegative rotor speeds (rev/s) on the schedule's clock."""
        p, lay = self.params, self.layout
        codes = _check_codes(schedule)
        T = len(codes)
        blocks, groups, _ = _blocks(codes, measured=True)
        k = rng.choice(self.n_classes, size=n, p=np.exp(p.log_pi - logsumexp(p.log_pi)))
        mean, Psi = _psi(p, lay)
        chol = np.linalg.cholesky(Psi)
        b = mean + rng.standard_normal((n, lay.q)) @ chol.T
        eps = rng.standard_normal((n, N_MODES, T))
        Z = np.empty((n, N_MODES, T))
        for m, L in enumerate(lay.modes):
            X, C = _columns(L, blocks, p.theta)
            mu = X @ p.theta[L.beta] + b[:, L.eff] @ C.T
            sd = np.sqrt(np.stack([p.sig2[_rows(kk), m][groups] for kk in range(self.n_classes)]))[
                k
            ]
            e = lfilter([1.0], [1.0, -p.phi[m, 0], -p.phi[m, 1]], eps[:, m] * sd, axis=1)
            Z[:, m] = mu + e
        W = from_obs(np.einsum("rm,nmt->nrt", MIXER, Z))
        powered = np.isin(codes, (1, 2))
        self.last_clip_fraction = (
            float((W[:, :, powered] == 0.0).any(axis=1).mean()) if powered.any() else 0.0
        )
        return W

    # ── diagnostics ──
    def summary(self) -> dict[str, Any]:
        """Training-free readout of the fitted law (AR poles, noise, level spread, classes)."""
        p, lay = self.params, self.layout
        ar = {}
        for m, name in enumerate(MODE_NAMES):
            roots = np.roots([1.0, -p.phi[m, 0], -p.phi[m, 1]])
            ar[name] = {
                "phi": p.phi[m].tolist(),
                "pole_radius": np.abs(roots).tolist(),
                "pole_hz": (np.abs(np.angle(roots)) * FS / (2 * np.pi)).tolist(),
            }
        _, Psi = _psi(p, lay)
        Lv = lay.level
        sd = np.sqrt(np.diag(Psi)[Lv])
        corr = Psi[np.ix_(Lv, Lv)] / np.outer(sd, sd)
        names = [lay.names[i] for i in Lv]
        return {
            "ar": ar,
            "noise_sd": {
                g: np.sqrt(p.sig2[i]).tolist()
                for i, g in enumerate(GROUP_NAMES)
                if self.observed[i]
            },
            "cruise_class_sd": np.sqrt(p.sig2[CRUISE:]).tolist(),
            "class_weights": np.exp(p.log_pi).tolist(),
            "levels": {
                nm: {"mean": float(mu), "sd": float(s)} for nm, mu, s in zip(names, p.nu, sd)
            },
            "level_corr": {"names": names, "matrix": corr.tolist()},
            "factor_share": (np.sum(p.gamma**2, axis=1) / sd**2).tolist(),
            "param_breakdown": self.param_breakdown(),
        }

    # ── serialization ──
    def to_dict(self) -> dict[str, Any]:
        p = self.params
        return {
            "config": _config(self.n_classes, p.gamma.shape[1]),
            "params": {
                "theta": p.theta.tolist(),
                "nu": p.nu.tolist(),
                "gamma": p.gamma.tolist(),
                "D": p.D.tolist(),
                "phi": p.phi.tolist(),
                "sig2": p.sig2.tolist(),
                "cruise_scale": p.cruise_scale.tolist(),
                "log_pi": p.log_pi.tolist(),
            },
            "observed_groups": self.observed.tolist(),
            "n_params": self.n_params,
            "param_breakdown": self.param_breakdown(),
            "history": [float(v) for v in self.history],
            "fit_info": self.fit_info,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ClockMixedModel:
        cfg = d["config"]
        if cfg != _config(int(cfg["n_classes"]), int(cfg["n_factors"])):
            raise ValueError(
                "serialized clock_mixed model has a different configuration than this module"
            )
        q = d["params"]
        arr = {k: np.asarray(v, dtype=np.float64) for k, v in q.items()}
        arr["gamma"] = arr["gamma"].reshape(len(arr["nu"]), int(cfg["n_factors"]))
        return cls(
            Params(**arr),
            np.asarray(d["observed_groups"], dtype=bool),
            history=d.get("history"),
            fit_info=d.get("fit_info"),
        )


# ─── fitting ──────────────────────────────────────────────────────────────────


def fit(
    tracks: list[tuple[np.ndarray, np.ndarray]],
    *,
    measured: list[bool] | None = None,
    max_iter: int = 80,
    tol: float = 1e-7,
    n_classes: int = N_CLASSES,
    n_factors: int = N_FACTORS,
) -> ClockMixedModel:
    """ECM on the exact penalized whole-record marginal likelihood.

    ``tracks``: ``(speeds (4, T) rev/s MIXER order, codes (T,))`` at 25 Hz;
    ``measured``: per-track source flag (False = command-only).
    """
    lay = _build_layout(n_factors)
    R = _penalty(lay)
    prep = _tracks(tracks, measured, lay)
    flights = [x[0] for x in prep]
    zs, blockss, groupss, taus, codess = ([x[j] for x in prep] for j in range(1, 6))
    p = _initialize(flights, zs, blockss, groupss, taus, codess, lay, R, n_classes)
    observed = np.sum([f.n[:CRUISE] for f in flights], axis=0) > 0
    n_valid = int(sum(f.T for f in flights))

    def evaluate(params):
        Gw = [[_whiten(H, params.phi[m]) for m, H in enumerate(f.H)] for f in flights]
        post = _e_step(flights, Gw, params, lay)
        data = float(post.loglik.sum())
        return Gw, post, data, data + _log_prior(params, R)

    Gw, post, data, obj = evaluate(p)
    history = [obj]
    trace = [
        {
            "iter": 0,
            "objective": obj,
            "data_loglik": data,
            "log_prior": obj - data,
            "gain_per_sample": None,
        }
    ]
    stop, rejected = "max_iter", None
    for it in range(1, max_iter + 1):
        cand = _m_step(flights, Gw, p, post, lay, R)
        try:
            Gw_c, post_c, data_c, obj_c = evaluate(cand)
        except linalg.LinAlgError:
            obj_c = -np.inf
        if not np.isfinite(obj_c) or obj_c < obj - MONOTONE_RTOL * max(1.0, abs(obj)):
            stop = "non_monotone"
            rejected = {"iter": it, "objective": float(obj_c) if np.isfinite(obj_c) else None}
            break
        gain = (obj_c - obj) / n_valid
        p, Gw, post, data, obj = cand, Gw_c, post_c, data_c, obj_c
        history.append(obj)
        trace.append(
            {
                "iter": it,
                "objective": obj,
                "data_loglik": data,
                "log_prior": obj - data,
                "gain_per_sample": gain,
            }
        )
        if gain < tol:
            stop = "converged"
            break
    flights_info = []
    for i, f in enumerate(flights):
        mu = post.resp[i] @ post.mean[i]
        flights_info.append(
            {
                "measured": f.measured,
                "samples": f.T,
                "loglik": float(post.loglik[i]),
                "class_resp": post.resp[i].tolist(),
                "level_posterior": {lay.names[j]: float(mu[j]) for j in lay.level},
            }
        )
    fit_info = {
        "trace": trace,
        "stop_reason": stop,
        "n_iter": len(history) - 1,
        "n_valid": n_valid,
        "n_flights": len(flights),
        "objective": obj,
        "data_loglik": data,
        "monotone": bool(np.all(np.diff(history) >= -MONOTONE_RTOL * max(1.0, abs(obj)))),
        "rejected": rejected,
        "flights": flights_info,
    }
    return ClockMixedModel(p, observed, history=history, fit_info=fit_info)


__all__ = ["ClockMixedModel", "Params", "event_clocks", "fit", "from_obs", "to_obs"]
