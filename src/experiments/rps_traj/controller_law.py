"""Hierarchical feedback-controller law of the four rotor speeds (whole flight).

Every recording is ONE draw of a controlled process in mixer coordinates: the
rotors relax toward a commanded setpoint with phase-dependent dynamics, the
setpoint is built from event-clock transient shapes and a random per-flight
equilibrium, and settled cruise is driven by an explicit-duration manoeuvre
grammar under a per-flight programme.  The only schedule input is the coarse
code per 25 Hz sample (0 before motors-on, 1 motors-on to take-off, 2
take-off to landing or end, 3 landing to end); event times are the first
sample of each code.  No level, target speed or waveform of a generated
flight is used and no training waveform is stored.  Observation transform,
mixer modes, Jacobian and readout are those of
:mod:`experiments.rps_traj.clock_mixed` (inverse softplus with the fixed
floor ``W_FLOOR``; ``z = MIXER.T y / 4``; nonnegative readout by
construction).

Controller (sample level)
-------------------------
Per mode ``m`` (common, roll, pitch, yaw) and sample ``t``::

    z_t = phi1 z_{t-1} + phi2 z_{t-2} + (1 - phi1 - phi2) r_t + sigma eps_t

with ``(phi1, phi2, sigma)`` of the sample's dynamics group: unit DC gain,
so ``r_t`` is the equilibrium the rotors settle to and ``phi`` its
relaxation (pole radius = settling timescale).  Lags before the record are
the rest state (all rotors at the floor).  The state is carried across a
group change as it is (never rescaled by a new noise SD).  ``phi`` lies in
the AR(2) stability triangle with a Dirichlet(1 + :data:`STAB_PRIOR`) prior
on its barycentric coordinates (edges: a root at +1, i.e. zero relaxation
gain; a root at -1; complex roots on the unit circle), and is learned
separately for arming, steady standby, take-off, landing approach, stop
(touchdown + rundown) and every cruise action; rest and the command-only
ground law have ``phi = 0``.  Innovation SDs are learned per noise group
(rest, arming, standby, take-off, approach, touchdown = the landing-mark
sample alone, rundown, the two command-only ground groups, each cruise
action) with inverse-gamma priors.

Arming timescale: the arming -> standby gate, the idle activation
``ramp(tau_on; 0, tau_arm)`` and the fade-out of the arming shape (over
:data:`ARM_FADE` s after ``tau_arm``) share ONE learned arming time
``tau_arm`` (:data:`TAU_GRID`), chosen by a generalized CM-step that
re-maximizes the arming/standby relaxation and noise for every candidate.
The steady-idle law starts once the motors have settled (DREGON: fractions
of a second; MD2: seconds); the exogenous hold until take-off is pure
standby around the idle equilibrium, however long.

Setpoint (event level)
----------------------
``r_t = X_t beta + sum_j b_j c_j(t)`` with fixed C1 activations and cubic
B-splines on signed time from motors-on, take-off and landing
(:data:`BASES`).  Common mode: rest, arming spline (faded after
``tau_arm``, gated off by take-off), take-off and landing splines.  Rotor
differences: low-resolution take-off spline.  Random effects ``b`` (14 per
flight): common idle and hover equilibrium; roll/pitch/yaw idle trim, hover
trim, landing fan-out amplitude; unit arming (spin-up jitter), take-off and
landing shape amplitudes of the collective.  Shared low-dimensional
loadings (learned): the hover level's exit through landing (``hover_la``)
and ONE weight mixing its smooth exit with a cut at the landing mark
(``hover_cut``, all modes), the collective arming/take-off/landing shape
profiles, and ONE fan-out profile shared by roll/pitch/yaw (low rank).
Level effects are ``N(nu + delta [command-only], D)`` (diagonal), ``D``
with the boundary-avoiding penalty ``-LEV_B / D``; the shapes are
``N(0, 1)``.  Population effects only: nothing per training flight is kept.

Source law (command-only DREGON records, ``measured=False``)
-------------------------------------------------------------
Their ground samples are scored by a command sensor law -- white noise
(``arming_cmd``, ``standby_cmd`` SDs, ``phi = 0``) around a step from rest
to the record's idle level -- so they inform neither the arming spline, nor
the arming time, nor any measured ground dynamics.  Their idle and hover
equilibria get source intercepts ``delta`` (Gaussian prior, SD
:data:`SRC_SD`), so a different command trim cannot move the measured
population means.  In cruise they share the action dynamics with a
per-mode innovation-variance scale ``kappa``.  Generation is always for a
measured record.

Programme and action grammar (cruise)
-------------------------------------
Cruise (code 2, >= :data:`TAKEOFF_END` s after take-off, before the landing
approach) is cut into :data:`BLOCK`-sample blocks.  A flight-level
programme ``k`` (weights ``pi``) selects the initial-action law and the
action-to-action transition matrix (no self-transitions); each action
``a`` lasts an explicit number of blocks, ``d - 1 ~ NegBin(mean, shape)``
truncated at :data:`DMAX`, shared by the programmes; the last segment is
right-censored at the end of cruise.  Actions set ``(phi, sigma)`` of every
mode jointly: an ordered sequence of manoeuvre regimes with their own
spectral shape and intensity.

Likelihood and fit (one whole-record objective)
-----------------------------------------------
Given the lags the residual is conditionally Gaussian, so a group's
contribution is a quadratic form in Gram matrices of
``V_t = [z_t, z_{t-1}, z_{t-2}, U_t]`` computed once from the data.  The
random equilibria couple every cruise block, so the exact marginal over
``b`` and all action paths is a Gaussian sum with exponentially many terms.
The fit maximizes instead the structured variational lower bound (ELBO)
``log p(z) >= E_q[log p(z, b, k, path)] + H[q]`` with
``q = q(b) q(k, path)``: ``q(b)`` is a full Gaussian (exact given the path
posterior) and ``q(k, path)`` the exact programme/semi-Markov posterior
given ``q(b)`` (segment forward-backward).  When the actions have identical
dynamics the bound equals the exact marginal likelihood; otherwise the gap
is ``KL(q || p(b, k, path | z))`` (small: ``b`` is pinned by thousands of
samples).  Every reported data value is this bound plus the log-Jacobian of
the transform, never called an exact likelihood.  Variational ECM:
E-step = coordinate ascent on ``q`` from the previous ``q``; CM-steps =
maximizers of the expected complete-data penalized log-likelihood:
``beta`` and loadings jointly (one linear system, second-difference + ridge
penalties), ``phi`` per group (strictly convex: damped Newton inside the
triangle), noise variances and ``kappa`` (closed form), ``nu, delta, D``,
programme weights, initial and transition laws (Dirichlet MAP), duration
laws (bounded numeric, kept only if better), ``tau_arm`` (grid; generalized
step that never lowers the objective).  The objective (ELBO + log-prior) is
non-decreasing; a decrease beyond :data:`MONOTONE_RTOL` reverts and stops
(``non_monotone``).

Initialization (deterministic, training tracks only): settling time of the
measured arming transients snapped to the grid; plateau medians as point
equilibria; ``beta`` by penalized least squares; shape and fan-out loadings
from the leading direction of per-flight residual projections; cruise
blocks split by residual-energy terciles into the initial actions (runs ->
durations and transitions; flights ranked into programmes).

Parameters (:meth:`ControllerLawModel.param_breakdown`): collective mean 45,
rotor-difference mean 12, loadings 38, level means 11 and variances 11,
source intercepts 8, relaxation AR 2 x 4 per observed dynamics group (<= 64),
noise variances 4 per observed noise group (<= 48), command cruise scale 4,
programme weights 1, initial laws 4, transitions 6, durations 6, arming
time 1: 259 with command-only records, 239 without.

Limitations: the bound is not the exact marginal; actions change dynamics,
not setpoints; the activations' relative widths are fixed except the
arming ramp; the floor is a density at the floor, not censoring; events are
the code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, NamedTuple, cast

import numpy as np
from scipy import linalg, optimize
from scipy.special import gammaln, logsumexp

from experiments.rps_traj.clock_mixed import (
    FS,
    LOG_DET_MODES,
    MODE_NAMES,
    N_MODES,
    W_FLOOR,
    _bspline,
    _check_codes,
    _leading,
    _ramp,
    _solve_spd,
    event_clocks,
    from_obs,
    to_obs,
)
from tracking.rotors import MIXER, NUM_ROTORS

FAMILY = "controller_law"
VERSION = 1
EVENTS = ("motors_on", "takeoff", "landing")
Z_REST = (
    MIXER.T @ to_obs(np.full(NUM_ROTORS, W_FLOOR))[0] / NUM_ROTORS
)  #: modes of the floored rest state

# ── clocks ──
BASES: dict[str, tuple[str, tuple[float, ...]]] = {
    "on_c": (
        "motors_on",
        (-0.4, 0.0, 0.12, 0.28, 0.5, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 3.6, 4.4, 5.4, 6.6, 8.0),
    ),
    "on_j": ("motors_on", (-0.2, 0.0, 0.08, 0.16, 0.28, 0.44, 0.64, 1.0)),
    "to_c": (
        "takeoff",
        (-1.0, -0.4, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0),
    ),
    "la_c": (
        "landing",
        (
            -4.0,
            -2.5,
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
            2.0,
            3.5,
            6.0,
            9.0,
            12.0,
        ),
    ),
    "to_lr": ("takeoff", (-1.0, 0.0, 1.0, 2.0, 3.0, 5.0)),
    "la_lr": ("landing", (-4.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0, 5.0, 7.5, 10.0, 12.0)),
}
IDLE_OFF = ("takeoff", 0.0, 1.0)
HOVER_ON = ("takeoff", 0.0, 3.0)
HOVER_OFF = ("landing", -0.6, 0.4)
REST_ON = ("landing", 10.0, 12.0)
#: Candidate arming times (s).
TAU_GRID = (0.2, 0.32, 0.48, 0.64, 0.96, 1.28, 1.6, 2.0, 2.4, 3.2, 4.0, 4.8, 6.4)
ARM_FADE = 0.4  #: s over which the arming shape fades out after the arming time
CMD_ARM_S = 0.2
TAKEOFF_END = 4.0
APPROACH_S = -2.0
TOUCHDOWN_S = (
    0.0,
    0.5 / FS,
)  #: the landing-mark sample alone (DREGON: the block mean of a motor cut)
RUNDOWN_END = 12.0

# ── groups ──
REST, ARMING, STANDBY, TAKEOFF, APPROACH, TOUCHDOWN, RUNDOWN, ARMING_CMD, STANDBY_CMD = range(9)
N_SG = 9
SG_NAMES = (
    "rest",
    "arming",
    "standby",
    "takeoff",
    "approach",
    "touchdown",
    "rundown",
    "arming_cmd",
    "standby_cmd",
)
PG_NAMES = ("arming", "standby", "takeoff", "approach", "stop")
N_PG = 5
#: noise group -> relaxation group (-1: phi = 0)
PHI_ROW = (-1, 0, 1, 2, 3, 4, 4, -1, -1)
#: used to fill a group no training sample visited (generation and the tau step only)
SIG_FALLBACK = {
    ARMING: STANDBY,
    STANDBY: ARMING,
    APPROACH: TOUCHDOWN,
    TOUCHDOWN: APPROACH,
    RUNDOWN: APPROACH,
}
PHI_FALLBACK = {0: 1, 1: 0, 3: 4, 4: 3}

# ── grammar ──
N_ACTIONS = 3
N_PROGRAMMES = 2
BLOCK = 25  #: samples per action block (1 s)
DMAX = 240  #: longest action segment (blocks)

# ── structure ──
#: Loadings: name -> basis block.  ``hover_cut`` mixes the smooth hover exit with a step at the landing mark
#: (one weight shared by all modes: 1 = motors cut at the mark, 0 = smooth descent).
LOADINGS: dict[str, str] = {
    "hover_la": "la_lr",
    "hover_cut": "cut",
    "takeoff_shape": "to_lr",
    "landing_shape": "la_lr",
    "fanout": "la_lr",
    "arming_shape": "on_j",
}


@dataclass(frozen=True)
class _Effect:
    name: str
    act: str | None
    loads: tuple[str, ...]
    level: bool  # N(nu (+ delta), D); else N(0, 1)
    source: bool  # has a command-only intercept


_COMMON = (
    _Effect("idle", "idle", (), True, True),
    _Effect("hover", "hover", ("hover_la", "hover_cut"), True, True),
    _Effect("takeoff_shape", None, ("takeoff_shape",), False, False),
    _Effect("landing_shape", None, ("landing_shape",), False, False),
    _Effect("arming_shape", None, ("arming_shape",), False, False),
)
_DIFF = (
    _Effect("idle", "idle", (), True, True),
    _Effect("hover", "hover", ("hover_cut",), True, True),
    _Effect("fanout", None, ("fanout",), True, False),
)
MODE_X = (("rest", "on_c", "to_c", "la_c"), ("to_lr",), ("to_lr",), ("to_lr",))
MODE_EFFECTS = (_COMMON, _DIFF, _DIFF, _DIFF)

# ── priors / penalties (transformed rev/s) ──
SIGMA_FLOOR = 0.02
SIG_A, SIG_B = 1.0, 0.05**2
KAPPA_A, KAPPA_B = 2.0, 3.0  #: IG prior of the command cruise variance scale (mode 1)
#: Level variances: boundary-avoiding penalty ``-LEV_B / D`` (IG with shape LEV_A = -1, improper), so
#: with a handful of flights ``D = (SS + 2 LEV_B) / n`` is not shrunk by a proper prior's extra counts.
LEV_A, LEV_B = -1.0, 0.02
SRC_SD = 5.0
SPLINE_D2, SPLINE_RIDGE = 1e-2, 1e-3
LOAD_D2, LOAD_RIDGE = 1e-2, 1e-2
CONC = 2.0  #: Dirichlet concentration of programme weights, initial and transition laws
DUR_BOUNDS = ((np.log(0.05), np.log(DMAX)), (np.log(0.2), np.log(100.0)))
N_INNER = 2
#: Stability prior: Dirichlet(1 + STAB_PRIOR) on the barycentric coordinates of the AR(2) stability triangle,
#: whose edges are a root at +1 (zero relaxation gain), at -1, and complex roots on the unit circle.
STAB_PRIOR = 1.0
MONOTONE_RTOL = 1e-9
TAU_EVERY = 3


# ─── design ───────────────────────────────────────────────────────────────────


def _design(clk: np.ndarray, measured: bool, tau_arm: float) -> dict[str, np.ndarray]:
    """Setpoint columns ``name -> (n, k)`` on clock rows ``clk`` (3, n)."""
    on, to, la = clk
    ground = 1.0 - _ramp(to, IDLE_OFF)
    if measured:
        arm = _ramp(on, ("motors_on", 0.0, tau_arm))
        # the arming shape lives inside arming: it fades out over ARM_FADE s after tau_arm, so a
        # longer exogenous hold is pure standby around the idle equilibrium
        fade = 1.0 - _ramp(on, ("motors_on", tau_arm, tau_arm + ARM_FADE))
        on_c = _bspline(on, BASES["on_c"][1]) * (ground * fade)[:, None]
        on_j = _bspline(on, BASES["on_j"][1]) * (ground * fade)[:, None]
    else:  # command sensor law: a step from rest to idle, no shared arming shape
        arm = (on >= 0.0).astype(np.float64)
        on_c = np.zeros((len(on), len(BASES["on_c"][1]) - 2))
        on_j = np.zeros((len(on), len(BASES["on_j"][1]) - 2))
    hover_on = _ramp(to, HOVER_ON)
    out = {
        "rest": ((1.0 - arm) * ground + _ramp(la, REST_ON))[:, None],
        "idle": (arm * ground)[:, None],
        "hover": (hover_on * (1.0 - _ramp(la, HOVER_OFF)))[:, None],
        "cut": (hover_on * (_ramp(la, HOVER_OFF) - (la >= 0.0)))[:, None],
        "on_c": on_c,
        "on_j": on_j,
    }
    for name in ("to_c", "la_c", "to_lr", "la_lr"):
        event, breaks = BASES[name]
        out[name] = _bspline(clk[EVENTS.index(event)], breaks)
    return out


def _sgroups(codes: np.ndarray, clk: np.ndarray, measured: bool, tau_arm: float) -> np.ndarray:
    """Noise group per row; -1 = cruise (action blocks)."""
    on, to, la = clk
    g = np.full(len(codes), -1, dtype=np.int64)
    ground = codes == 1
    if measured:
        g[ground] = np.where(on[ground] < tau_arm, ARMING, STANDBY)
    else:
        g[ground] = np.where(on[ground] < CMD_ARM_S, ARMING_CMD, STANDBY_CMD)
    air = codes >= 2
    g[air & (to < TAKEOFF_END)] = TAKEOFF
    g[air & (la >= APPROACH_S)] = APPROACH
    g[air & (la >= TOUCHDOWN_S[0])] = TOUCHDOWN
    g[air & (la >= TOUCHDOWN_S[1])] = RUNDOWN
    g[(codes == 0) | (la >= RUNDOWN_END)] = REST
    return g


@dataclass(frozen=True)
class _Mode:
    order: tuple[str, ...]
    cols: dict[str, np.ndarray]  # U columns of each block
    p: int
    x: np.ndarray
    beta: slice
    eff: np.ndarray  # global effect indices
    act: np.ndarray  # U column of each effect's activation or -1
    loads: tuple[tuple[int, slice, np.ndarray], ...]
    parts: tuple[
        tuple[slice, np.ndarray, int], ...
    ]  # theta slice, U columns, multiplying effect (-1: none)


@dataclass(frozen=True)
class _Layout:
    modes: tuple[_Mode, ...]
    load: dict[str, slice]
    n_theta: int
    q: int
    names: tuple[str, ...]
    level: np.ndarray  # bool (q,)
    source: np.ndarray  # bool (q,)


def _build_layout() -> _Layout:
    sizes = {
        "rest": 1,
        "idle": 1,
        "hover": 1,
        "cut": 1,
        **{k: len(b) - 2 for k, (_, b) in BASES.items()},
    }
    off, betas = 0, []
    for m in range(N_MODES):
        n = sum(sizes[b] for b in MODE_X[m])
        betas.append(slice(off, off + n))
        off += n
    load = {}
    for name, blk in LOADINGS.items():
        load[name] = slice(off, off + sizes[blk])
        off += sizes[blk]
    modes, names, level, source, g = [], [], [], [], 0
    for m in range(N_MODES):
        effects = MODE_EFFECTS[m]
        acts = [e.act for e in effects if e.act]
        order = tuple(
            dict.fromkeys((*MODE_X[m], *acts, *(LOADINGS[k] for e in effects for k in e.loads)))
        )
        cols, c = {}, 0
        for b in order:
            cols[b] = np.arange(c, c + sizes[b])
            c += sizes[b]
        x = np.concatenate([cols[b] for b in MODE_X[m]])
        loads = tuple(
            (j, load[k], cols[LOADINGS[k]]) for j, e in enumerate(effects) for k in e.loads
        )
        parts = ((betas[m], x, -1), *((ts, us, j) for j, ts, us in loads))
        modes.append(
            _Mode(
                order,
                cols,
                c,
                x,
                betas[m],
                np.arange(g, g + len(effects)),
                np.array([cols[e.act][0] if e.act else -1 for e in effects]),
                loads,
                parts,
            )
        )
        for e in effects:
            names.append(f"{MODE_NAMES[m]}.{e.name}")
            level.append(e.level)
            source.append(e.source)
        g += len(effects)
    return _Layout(tuple(modes), load, off, g, tuple(names), np.array(level), np.array(source))


def _penalty(lay: _Layout) -> np.ndarray:
    def pen(s, d2, ridge):
        D = np.diff(np.eye(s), 2, axis=0)
        return d2 * D.T @ D + ridge * np.eye(s)

    R = np.zeros((lay.n_theta, lay.n_theta))
    for m, L in enumerate(lay.modes):
        o = L.beta.start
        for b in MODE_X[m]:
            s = len(L.cols[b])
            R[o : o + s, o : o + s] = pen(s, 0.0 if b == "rest" else SPLINE_D2, SPLINE_RIDGE)
            o += s
    for sl in lay.load.values():
        s = sl.stop - sl.start
        R[sl, sl] = pen(s, LOAD_D2, LOAD_RIDGE)
    return R


def _umat(L: _Mode, blocks: dict[str, np.ndarray]) -> np.ndarray:
    return np.hstack([blocks[b] for b in L.order])


def _wc_M(L: _Mode, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Setpoint coefficients over U: ``w(b) = w_c + M b``."""
    wc = np.zeros(L.p)
    wc[L.x] = theta[L.beta]
    M = np.zeros((L.p, len(L.act)))
    for j, col in enumerate(L.act):
        if col >= 0:
            M[col, j] = 1.0
    for j, ts, us in L.loads:
        M[us, j] += theta[ts]
    return wc, M


# ─── data ─────────────────────────────────────────────────────────────────────


@dataclass
class _Flight:
    T: int
    logjac: float
    measured: bool
    fixed: list[
        dict[int, tuple[np.ndarray, float]]
    ]  # per mode: noise group -> (Gram, n), tau-free rows
    cur: list[dict[int, tuple[np.ndarray, float]]]  # per mode at the current tau
    cb: list[np.ndarray]  # per mode (nb, P, P) cruise block Grams
    nblk: np.ndarray  # (nb,)
    w_z: np.ndarray  # (4, 3, nw) lags of the tau-dependent rows
    w_clk: np.ndarray  # (3, nw)
    w_codes: np.ndarray
    ground_on: np.ndarray  # on-clock of measured ground rows (initialization only)
    ground_z: np.ndarray  # (4, n) their modes
    hover_z: np.ndarray  # (4, n) cruise (or late take-off) modes, initialization only

    @property
    def nb(self) -> int:
        return len(self.nblk)


def _grams(V: np.ndarray, groups: np.ndarray) -> dict[int, tuple[np.ndarray, float]]:
    return {
        int(g): (V[groups == g].T @ V[groups == g], float(np.sum(groups == g)))
        for g in np.unique(groups)
    }


def _window_grams(
    f: _Flight, lay: _Layout, tau: float
) -> list[dict[int, tuple[np.ndarray, float]]]:
    if f.w_clk.shape[1] == 0:
        return [{} for _ in lay.modes]
    blocks = _design(f.w_clk, True, tau)
    g = _sgroups(f.w_codes, f.w_clk, True, tau)
    return [
        _grams(np.column_stack([f.w_z[m].T, _umat(L, blocks)]), g) for m, L in enumerate(lay.modes)
    ]


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for g, (G, n) in b.items():
        out[g] = (out[g][0] + G, out[g][1] + n) if g in out else (G, n)
    return out


def _set_tau(f: _Flight, lay: _Layout, tau: float) -> None:
    f.cur = [_merge(fx, wg) for fx, wg in zip(f.fixed, _window_grams(f, lay, tau))]


def _prepare(w: Any, codes: Any, measured: bool, lay: _Layout, tau: float) -> _Flight:
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != NUM_ROTORS:
        raise ValueError(f"speeds must be (4, T), got {w.shape}")
    if not np.isfinite(w).all():
        raise ValueError("speeds must be finite (bridge gaps before fitting)")
    T = w.shape[1]
    c = _check_codes(codes, T)
    y, lj = to_obs(w)
    z = MIXER.T @ y / NUM_ROTORS
    clk = event_clocks(c)
    rest = np.repeat(Z_REST[:, None], 2, axis=1)
    zl = np.concatenate([rest, z], axis=1)
    Z3 = np.stack([z, zl[:, 1 : T + 1], zl[:, :T]], axis=1)  # (4, 3, T): z_t, z_{t-1}, z_{t-2}
    groups = _sgroups(c, clk, measured, tau)
    win = np.zeros(T, dtype=bool)
    if measured:
        win = ((c == 1) | ((c >= 2) & (clk[1] >= 0.0) & (clk[1] < IDLE_OFF[2]))) & (groups >= 0)
    fixed_rows = (groups >= 0) & ~win
    cruise = np.flatnonzero(groups < 0)
    chunks = [cruise[i : i + BLOCK] for i in range(0, len(cruise), BLOCK)]
    blocks = _design(clk, measured, tau)
    fixed, cb = [], []
    for m, L in enumerate(lay.modes):
        V = np.column_stack([Z3[m].T, _umat(L, blocks)])
        fixed.append(_grams(V[fixed_rows], groups[fixed_rows]))
        cb.append(
            np.stack([V[ix].T @ V[ix] for ix in chunks])
            if chunks
            else np.zeros((0, L.p + 3, L.p + 3))
        )
    gmask = (c == 1) & measured
    hmask = groups < 0
    if not hmask.any():
        hmask = (c == 2) & (clk[1] > 3.0) & (clk[2] < APPROACH_S)
    idx = np.flatnonzero(win)
    f = _Flight(
        T,
        float(lj.sum() + T * LOG_DET_MODES),
        bool(measured),
        fixed,
        [],
        cb,
        np.array([len(ix) for ix in chunks], dtype=np.float64),
        Z3[:, :, idx],
        clk[:, idx],
        c[idx],
        clk[0, gmask],
        z[:, gmask],
        z[:, hmask],
    )
    _set_tau(f, lay, tau)
    return f


# ─── parameters ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Params:
    theta: np.ndarray  # beta of every mode, then the loadings
    nu: np.ndarray  # (q,) prior means (shapes: 0)
    delta: np.ndarray  # (q,) command-only intercepts (non-source effects: 0)
    D: np.ndarray  # (q,) prior variances (shapes: 1)
    phi: np.ndarray  # (N_PG + A, 4, 2)
    sig2: np.ndarray  # (N_SG + A, 4)
    kappa: np.ndarray  # (4,) command-only cruise variance scale
    log_pi: np.ndarray  # (K,)
    log_init: np.ndarray  # (K, A)
    log_trans: np.ndarray  # (K, A, A), -inf on the diagonal
    dur: np.ndarray  # (A, 2) log(mean - 1), log(shape) of the duration law
    tau_arm: float


def _dur_tables(dur: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Log pmf and log survival ``log P(D >= d)`` over ``d = 1..DMAX`` per action."""
    k = np.arange(DMAX)
    out_p, out_s = np.empty((len(dur), DMAX)), np.empty((len(dur), DMAX))
    for a, (lm, lr) in enumerate(dur):
        mean, r = np.exp(lm), np.exp(lr)
        pr = r / (r + mean)
        lp = gammaln(k + r) - gammaln(r) - gammaln(k + 1.0) + r * np.log(pr) + k * np.log1p(-pr)
        lp -= logsumexp(lp)
        out_p[a] = lp
        out_s[a] = np.logaddexp.accumulate(lp[::-1])[::-1]
    return out_p, out_s


def _phi(p: Params, row: int, m: int) -> np.ndarray:
    return np.zeros(2) if row < 0 else p.phi[row, m]


def _filled(p: Params, obs_s: np.ndarray, obs_p: np.ndarray) -> Params:
    """Groups never visited take their fallback's values (no data term changes)."""
    sig2, phi = p.sig2.copy(), p.phi.copy()
    for g, src in SIG_FALLBACK.items():
        if not obs_s[g] and obs_s[src]:
            sig2[g] = sig2[src]
    for g, src in PHI_FALLBACK.items():
        if not obs_p[g] and obs_p[src]:
            phi[g] = phi[src]
    return replace(p, sig2=sig2, phi=phi)


def _observed(flights: list[_Flight]) -> tuple[np.ndarray, np.ndarray]:
    obs_s = np.zeros(N_SG + N_ACTIONS, dtype=bool)
    for f in flights:
        for g in f.cur[0]:
            obs_s[g] = True
        if f.nb:
            obs_s[N_SG:] = True
    obs_p = np.zeros(N_PG + N_ACTIONS, dtype=bool)
    for g in np.flatnonzero(obs_s[:N_SG]):
        if PHI_ROW[g] >= 0:
            obs_p[PHI_ROW[g]] = True
    obs_p[N_PG:] = obs_s[N_SG:]
    return obs_s, obs_p


# ─── cells and expectations ───────────────────────────────────────────────────


class _Cell(NamedTuple):
    G: np.ndarray
    n: float
    s2: float
    phi: np.ndarray
    srow: int
    prow: int
    cmd_cruise: bool


def _cells(f: _Flight, p: Params, gamma: np.ndarray) -> list[list[_Cell]]:
    out = []
    for m in range(N_MODES):
        lst = [
            _Cell(G, n, p.sig2[g, m], _phi(p, PHI_ROW[g], m), g, PHI_ROW[g], False)
            for g, (G, n) in f.cur[m].items()
        ]
        if f.nb:
            scale = 1.0 if f.measured else p.kappa[m]
            for a in range(N_ACTIONS):
                lst.append(
                    _Cell(
                        np.tensordot(gamma[:, a], f.cb[m], axes=1),
                        float(gamma[:, a] @ f.nblk),
                        p.sig2[N_SG + a, m] * scale,
                        p.phi[N_PG + a, m],
                        N_SG + a,
                        N_PG + a,
                        not f.measured,
                    )
                )
        out.append(lst)
    return out


def _esq(
    G: np.ndarray, phi: np.ndarray, wc: np.ndarray, M: np.ndarray, mu: np.ndarray, S: np.ndarray
) -> float:
    """``E_q sum eps_t^2`` over a cell."""
    g = 1.0 - phi[0] - phi[1]
    c = np.r_[1.0, -phi[0], -phi[1], -g * (wc + M @ mu)]
    return float(c @ G @ c + g * g * np.vdot(G[3:, 3:], M @ S @ M.T))


def _ell(cell: _Cell, wcM: tuple[np.ndarray, np.ndarray], mu, S) -> float:
    return -0.5 * (
        cell.n * np.log(2.0 * np.pi * cell.s2) + _esq(cell.G, cell.phi, *wcM, mu, S) / cell.s2
    )


def _emissions(
    f: _Flight, p: Params, lay: _Layout, wcM, mu: np.ndarray, S: np.ndarray
) -> np.ndarray:
    """``E_q(b) log p(block j | action a)`` (nb, A)."""
    e = np.zeros((f.nb, N_ACTIONS))
    for m, L in enumerate(lay.modes):
        wc, M = wcM[m]
        mm, Sm = mu[L.eff], S[np.ix_(L.eff, L.eff)]
        MSM = M @ Sm @ M.T
        tr = np.einsum("jpq,pq->j", f.cb[m][:, 3:, 3:], MSM)
        scale = 1.0 if f.measured else p.kappa[m]
        for a in range(N_ACTIONS):
            ph = p.phi[N_PG + a, m]
            g = 1.0 - ph[0] - ph[1]
            c = np.r_[1.0, -ph[0], -ph[1], -g * (wc + M @ mm)]
            s2 = p.sig2[N_SG + a, m] * scale
            e[:, a] -= 0.5 * (
                f.nblk * np.log(2.0 * np.pi * s2) + ((f.cb[m] @ c) @ c + g * g * tr) / s2
            )
    return e


# ─── programme / action grammar ───────────────────────────────────────────────


@dataclass
class _Path:
    logz: float
    resp: np.ndarray  # (K,)
    gamma: np.ndarray  # (nb, A)
    init: np.ndarray  # (K, A) expected initial actions
    trans: np.ndarray  # (K, A, A) expected transitions
    dur: np.ndarray  # (A, DMAX) expected completed segments by length
    cens: np.ndarray  # (A, DMAX) expected censored (last) segments by length


def _hsmm(
    e: np.ndarray, log_init: np.ndarray, log_trans: np.ndarray, lp: np.ndarray, ls: np.ndarray
):
    """Exact segment forward-backward of an explicit-duration chain whose last
    segment is right-censored at the end of the sequence."""
    nb, A = e.shape
    dm = min(nb, lp.shape[1])
    C = np.vstack([np.zeros((1, A)), np.cumsum(e, axis=0)])
    Fs = np.full((nb, A), -np.inf)  # a segment of a starts at s
    Fe = np.full((nb + 1, A), -np.inf)  # a complete segment of a ends at t
    Fs[0] = log_init
    ends = None
    for t in range(1, nb + 1):
        d = np.arange(1, min(t, dm) + 1)
        seg = Fs[t - d] + C[t] - C[t - d]
        if t < nb:
            Fe[t] = logsumexp(seg + lp[:, d - 1].T, axis=0)
            Fs[t] = logsumexp(Fe[t][:, None] + log_trans, axis=0)
        else:
            ends = logsumexp(seg + ls[:, d - 1].T, axis=0)
    logz = float(cast(float, logsumexp(ends)))
    Bs = np.full((nb, A), -np.inf)
    Be = np.zeros((nb + 1, A))
    gd = np.zeros((nb + 1, A))
    dur, cens = np.zeros((A, lp.shape[1])), np.zeros((A, lp.shape[1]))
    for s in range(nb - 1, -1, -1):
        d = np.arange(1, min(nb - s, dm) + 1)
        t = s + d
        done = t < nb
        term = C[t] - C[s] + np.where(done[:, None], lp[:, d - 1].T + Be[t], ls[:, d - 1].T)
        Bs[s] = logsumexp(term, axis=0)
        mass = np.exp(Fs[s] + term - logz)
        dur[:, d[done] - 1] += mass[done].T
        cens[:, d[~done] - 1] += mass[~done].T
        gd[s] += mass.sum(axis=0)
        gd[t] -= mass
        if s >= 1:
            Be[s] = logsumexp(log_trans + Bs[s][None, :], axis=1)
    gamma = np.cumsum(gd, axis=0)[:nb]
    init = np.exp(log_init + Bs[0] - logz)
    trans = np.zeros((A, A))
    if nb > 1:
        lt = cast(
            np.ndarray,
            logsumexp(Fe[1:nb, :, None] + log_trans[None] + Bs[1:nb, None, :], axis=0),
        )
        trans = np.exp(lt - logz)
    return logz, gamma, init, trans, dur, cens


def _path(e: np.ndarray, p: Params, tables) -> _Path:
    K, A = p.log_init.shape
    if len(e) == 0:
        return _Path(
            0.0,
            np.exp(p.log_pi),
            e.copy(),
            np.zeros((K, A)),
            np.zeros((K, A, A)),
            np.zeros((A, DMAX)),
            np.zeros((A, DMAX)),
        )
    res = [_hsmm(e, p.log_init[k], p.log_trans[k], *tables) for k in range(K)]
    lz = p.log_pi + np.array([r[0] for r in res])
    tot = float(cast(float, logsumexp(lz)))
    resp = np.exp(lz - tot)
    return _Path(
        tot,
        resp,
        cast(np.ndarray, sum(w * r[1] for w, r in zip(resp, res))),
        np.stack([w * r[2] for w, r in zip(resp, res)]),
        np.stack([w * r[3] for w, r in zip(resp, res)]),
        cast(np.ndarray, sum(w * r[4] for w, r in zip(resp, res))),
        cast(np.ndarray, sum(w * r[5] for w, r in zip(resp, res))),
    )


def _sample_actions(
    nb: int, k: int, p: Params, pmf: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    out = np.empty(nb, dtype=np.int64)
    init, trans = np.exp(p.log_init[k]), np.exp(p.log_trans[k])
    a, pos = rng.choice(N_ACTIONS, p=init / init.sum()), 0
    while pos < nb:
        d = 1 + rng.choice(DMAX, p=pmf[a])
        out[pos : pos + d] = a
        pos += d
        a = rng.choice(N_ACTIONS, p=trans[a] / trans[a].sum())
    return out


# ─── E-step ───────────────────────────────────────────────────────────────────


@dataclass
class _Q:
    mu: np.ndarray  # (n, q)
    S: np.ndarray  # (n, q, q)
    paths: list[_Path]
    elbo: np.ndarray  # (n,) lower bound of each record's log-density (Jacobian included)


def _prior(p: Params, f: _Flight) -> tuple[np.ndarray, np.ndarray]:
    return p.nu + (0.0 if f.measured else p.delta), p.D


def _b_step(f: _Flight, cells, wcM, p: Params, lay: _Layout) -> tuple[np.ndarray, np.ndarray]:
    m0, Psi = _prior(p, f)
    P = np.diag(1.0 / Psi)
    h = m0 / Psi
    for m, L in enumerate(lay.modes):
        wc, M = wcM[m]
        ix = np.ix_(L.eff, L.eff)
        for c in cells[m]:
            g = 1.0 - c.phi[0] - c.phi[1]
            c0 = np.r_[1.0, -c.phi[0], -c.phi[1], -g * wc]
            P[ix] += (g * g / c.s2) * M.T @ c.G[3:, 3:] @ M
            h[L.eff] += (g / c.s2) * M.T @ (c.G[3:] @ c0)
    cP = linalg.cho_factor(P, lower=True)
    return linalg.cho_solve(cP, h), linalg.cho_solve(cP, np.eye(len(h)))


def _kl(mu: np.ndarray, S: np.ndarray, m0: np.ndarray, Psi: np.ndarray) -> float:
    _, logdet = np.linalg.slogdet(S)
    return 0.5 * float(
        np.sum((np.diag(S) + (mu - m0) ** 2) / Psi) - len(mu) + np.sum(np.log(Psi)) - logdet
    )


def _flight_q(
    f: _Flight, p: Params, lay: _Layout, wcM, tables, mu, S, n_inner: int, tol: float = 0.0
):
    path = _path(_emissions(f, p, lay, wcM, mu, S), p, tables)
    val = -np.inf
    for _ in range(n_inner):
        mu, S = _b_step(f, _cells(f, p, path.gamma), wcM, p, lay)
        path = _path(_emissions(f, p, lay, wcM, mu, S), p, tables)
        if tol:
            prev, val = val, _elbo_flight(f, p, lay, wcM, mu, S, path)
            if val - prev < tol:
                break
    return mu, S, path, _elbo_flight(f, p, lay, wcM, mu, S, path)


def _elbo_flight(f: _Flight, p: Params, lay: _Layout, wcM, mu, S, path: _Path) -> float:
    val = f.logjac + path.logz - _kl(mu, S, *_prior(p, f))
    for m, L in enumerate(lay.modes):
        mm, Sm = mu[L.eff], S[np.ix_(L.eff, L.eff)]
        for g, (G, n) in f.cur[m].items():
            val += _ell(
                _Cell(G, n, p.sig2[g, m], _phi(p, PHI_ROW[g], m), g, PHI_ROW[g], False),
                wcM[m],
                mm,
                Sm,
            )
    return float(val)


def _e_step(flights, p: Params, lay: _Layout, q: _Q, n_inner: int = N_INNER) -> _Q:
    wcM = [_wc_M(L, p.theta) for L in lay.modes]
    tables = _dur_tables(p.dur)
    mus, Ss, paths, elbo = [], [], [], []
    for i, f in enumerate(flights):
        mu, S, path, val = _flight_q(f, p, lay, wcM, tables, q.mu[i], q.S[i], n_inner)
        mus.append(mu)
        Ss.append(S)
        paths.append(path)
        elbo.append(val)
    return _Q(np.array(mus), np.array(Ss), paths, np.array(elbo))


# ─── priors ───────────────────────────────────────────────────────────────────


def _log_prior(p: Params, R: np.ndarray, lay: _Layout) -> float:
    lp = -0.5 * p.theta @ R @ p.theta
    lp += np.sum(-(SIG_A + 1.0) * np.log(p.sig2) - SIG_B / p.sig2)
    lp += np.sum(-(KAPPA_A + 1.0) * np.log(p.kappa) - KAPPA_B / p.kappa)
    D = p.D[lay.level]
    lp += np.sum(-(LEV_A + 1.0) * np.log(D) - LEV_B / D)
    lp += -0.5 * np.sum(p.delta[lay.source] ** 2) / SRC_SD**2
    off = ~np.eye(N_ACTIONS, dtype=bool)
    lp += (CONC - 1.0) * (np.sum(p.log_pi) + np.sum(p.log_init) + np.sum(p.log_trans[:, off]))
    lp += STAB_PRIOR * np.sum(np.log(_bary(p.phi)))
    return float(lp)


# ─── CM-steps ─────────────────────────────────────────────────────────────────


def _moment(mu: np.ndarray, Ebb: np.ndarray, j: int, k: int) -> float:
    if j < 0 and k < 0:
        return 1.0
    if j < 0:
        return float(mu[k])
    if k < 0:
        return float(mu[j])
    return float(Ebb[j, k])


#: Barycentric coordinates of ``phi`` in the stability triangle: ``A_BARY @ (1, phi1, phi2)``.
A_BARY = np.array([[0.25, -0.25, -0.25], [0.25, 0.25, -0.25], [0.5, 0.0, 0.5]])


def _bary(phi: np.ndarray) -> np.ndarray:
    return A_BARY[:, 0] + phi @ A_BARY[:, 1:].T


def _relax_step(Kmat: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """``argmin 0.5 (1, -phi)' K (1, -phi) - STAB_PRIOR sum log bary(phi)`` (strictly convex,
    interior of the stability triangle) by damped Newton; ``x0`` is kept unless improved."""
    Q, k, C = Kmat[1:, 1:], Kmat[0, 1:], A_BARY[:, 1:]

    def f(x):
        lam = _bary(x)
        return (
            np.inf
            if (lam <= 0.0).any()
            else 0.5 * (Kmat[0, 0] - 2.0 * k @ x + x @ Q @ x) - STAB_PRIOR * np.log(lam).sum()
        )

    x = np.array([0.0, -1.0 / 3.0])
    for _ in range(100):
        lam = _bary(x)
        grad = Q @ x - k - STAB_PRIOR * C.T @ (1.0 / lam)
        step = np.linalg.solve(Q + STAB_PRIOR * (C.T / lam**2) @ C, grad)
        t, fx = 1.0, f(x)
        while t > 1e-12 and not f(x - t * step) <= fx:
            t *= 0.5
        if t <= 1e-12:
            break
        x = x - t * step
        if np.abs(t * step).max() < 1e-12:
            break
    return min((np.asarray(x0, dtype=np.float64), x), key=f)


def _cm_theta(flights, cellss, q: _Q, lay: _Layout, R: np.ndarray) -> np.ndarray:
    A, c = R.copy(), np.zeros(lay.n_theta)
    for i, cells in enumerate(cellss):
        for m, L in enumerate(lay.modes):
            mu = q.mu[i, L.eff]
            Ebb = q.S[i][np.ix_(L.eff, L.eff)] + np.outer(mu, mu)
            acts = [(j, col) for j, col in enumerate(L.act) if col >= 0]
            for cell in cells[m]:
                g = 1.0 - cell.phi[0] - cell.phi[1]
                GUU = cell.G[3:, 3:]
                GU0 = cell.G[3:, :3] @ np.r_[1.0, -cell.phi[0], -cell.phi[1]]
                a1, a2 = g / cell.s2, g * g / cell.s2
                for sl1, u1, j1 in L.parts:
                    c[sl1] += a1 * _moment(mu, Ebb, j1, -1) * GU0[u1]
                    for j, col in acts:
                        c[sl1] -= a2 * _moment(mu, Ebb, j1, j) * GUU[u1, col]
                    for sl2, u2, j2 in L.parts:
                        A[sl1, sl2] += a2 * _moment(mu, Ebb, j1, j2) * GUU[np.ix_(u1, u2)]
    return _solve_spd(A, c)


def _phi_k(cellss, q: _Q, lay: _Layout, theta: np.ndarray, rows: int) -> np.ndarray:
    """``K`` with ``E_q sum eps^2 / sigma^2 = (1, -phi)' K (1, -phi)`` per relaxation group and mode."""
    Kmat = np.zeros((rows, N_MODES, 3, 3))
    for i, cells in enumerate(cellss):
        for m, L in enumerate(lay.modes):
            wc, M = _wc_M(L, theta)
            mu, Sm = q.mu[i, L.eff], q.S[i][np.ix_(L.eff, L.eff)]
            w = wc + M @ mu
            MSM = M @ Sm @ M.T
            for cell in cells[m]:
                if cell.prow < 0:
                    continue
                Gw = cell.G[:, 3:] @ w
                wGw = w @ cell.G[3:, 3:] @ w + np.vdot(cell.G[3:, 3:], MSM)
                Kmat[cell.prow, m] += (cell.G[:3, :3] - Gw[:3, None] - Gw[None, :3] + wGw) / cell.s2
    return Kmat


def _cm_phi(
    cellss, q: _Q, lay: _Layout, theta: np.ndarray, phi: np.ndarray, obs_p: np.ndarray
) -> np.ndarray:
    Kmat = _phi_k(cellss, q, lay, theta, len(phi))
    out = phi.copy()
    for r in np.flatnonzero(obs_p):
        for m in range(N_MODES):
            out[r, m] = _relax_step(Kmat[r, m], phi[r, m])
    return out


def _cm_sigma(flights, cellss, q: _Q, lay: _Layout, p: Params) -> tuple[np.ndarray, np.ndarray]:
    ess, cnt = np.zeros_like(p.sig2), np.zeros_like(p.sig2)
    cess, ccnt = np.zeros((N_ACTIONS, N_MODES)), np.zeros((N_ACTIONS, N_MODES))
    for i, cells in enumerate(cellss):
        for m, L in enumerate(lay.modes):
            wcM = _wc_M(L, p.theta)
            mu, Sm = q.mu[i, L.eff], q.S[i][np.ix_(L.eff, L.eff)]
            for cell in cells[m]:
                e = _esq(cell.G, cell.phi, *wcM, mu, Sm)
                if cell.cmd_cruise:
                    cess[cell.srow - N_SG, m] += e
                    ccnt[cell.srow - N_SG, m] += cell.n
                else:
                    ess[cell.srow, m] += e
                    cnt[cell.srow, m] += cell.n
    kappa = p.kappa
    sig2 = np.empty_like(ess)
    sig2[:N_SG] = (ess[:N_SG] + 2.0 * SIG_B) / (cnt[:N_SG] + 2.0 * SIG_A + 2.0)
    sig2[N_SG:] = (ess[N_SG:] + cess / kappa + 2.0 * SIG_B) / (
        cnt[N_SG:] + ccnt + 2.0 * SIG_A + 2.0
    )
    sig2 = np.maximum(sig2, SIGMA_FLOOR**2)
    if ccnt.sum() > 0:
        kappa = (np.sum(cess / sig2[N_SG:], axis=0) + 2.0 * KAPPA_B) / (
            ccnt.sum(axis=0) + 2.0 * KAPPA_A + 2.0
        )
    return sig2, kappa


def _cm_levels(
    flights, q: _Q, p: Params, lay: _Layout
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cmd = np.array([0.0 if f.measured else 1.0 for f in flights])
    n, nc = len(flights), cmd.sum()
    nu, delta, D = p.nu.copy(), p.delta.copy(), p.D.copy()
    var = np.diagonal(q.S, axis1=1, axis2=2)
    for j in np.flatnonzero(lay.level):
        mu = q.mu[:, j]
        if lay.source[j] and nc > 0:
            A = np.array([[n, nc], [nc, nc + D[j] / SRC_SD**2]])
            nu[j], delta[j] = np.linalg.solve(A, [mu.sum(), cmd @ mu])
        else:
            nu[j] = mu.mean()
        r = np.sum(var[:, j] + (mu - nu[j] - delta[j] * cmd) ** 2)
        D[j] = (r + 2.0 * LEV_B) / (n + 2.0 * LEV_A + 2.0)
    return nu, delta, D


def _dur_objective(x: np.ndarray, dc: np.ndarray, cc: np.ndarray) -> float:
    lp, ls = _dur_tables(x[None])
    return -float(dc @ lp[0] + cc @ ls[0])


def _cm_grammar(q: _Q, p: Params) -> dict[str, np.ndarray]:
    K, A = p.log_init.shape
    resp = np.sum([pa.resp for pa in q.paths], axis=0)
    init = np.sum([pa.init for pa in q.paths], axis=0)
    trans = np.sum([pa.trans for pa in q.paths], axis=0)
    dc = np.sum([pa.dur for pa in q.paths], axis=0)
    cc = np.sum([pa.cens for pa in q.paths], axis=0)
    log_pi = np.log((resp + CONC - 1.0) / (resp.sum() + K * (CONC - 1.0)))
    log_init = np.log((init + CONC - 1.0) / (init.sum(axis=1, keepdims=True) + A * (CONC - 1.0)))
    off = ~np.eye(A, dtype=bool)
    tr = np.where(off, trans + CONC - 1.0, 0.0)
    with np.errstate(divide="ignore"):
        log_trans = np.log(tr / tr.sum(axis=2, keepdims=True))
    dur = p.dur.copy()
    for a in range(A):
        if dc[a].sum() + cc[a].sum() <= 1e-12:
            continue
        old = _dur_objective(dur[a], dc[a], cc[a])
        res = optimize.minimize(
            _dur_objective, dur[a], args=(dc[a], cc[a]), method="L-BFGS-B", bounds=DUR_BOUNDS
        )
        if np.isfinite(res.fun) and res.fun < old:
            dur[a] = res.x
    return {"log_pi": log_pi, "log_init": log_init, "log_trans": log_trans, "dur": dur}


def _tau_value(
    flights, q: _Q, p: Params, lay: _Layout, R: np.ndarray, tau: float
) -> tuple[float, Params]:
    """Objective part of every tau-dependent row (+ log-prior) at arming time ``tau``, after
    re-maximizing the arming and standby relaxation and noise from ``p`` (alternating exact steps)."""
    grams = [_window_grams(f, lay, tau) if f.measured else [{} for _ in lay.modes] for f in flights]
    cellss = [
        [
            [_Cell(G, n, 1.0, np.zeros(2), g, PHI_ROW[g], False) for g, (G, n) in wg[m].items()]
            for m in range(N_MODES)
        ]
        for wg in grams
    ]
    Kmat = _phi_k(cellss, q, lay, p.theta, N_PG)  # unit sigma: raw expected squares
    phi, sig2 = p.phi.copy(), p.sig2.copy()
    for g in (ARMING, STANDBY):
        n = sum(wg[0][g][1] for wg in grams if g in wg[0])
        if n == 0:
            continue
        r = PHI_ROW[g]
        for m in range(N_MODES):
            for _ in range(3):
                phi[r, m] = _relax_step(Kmat[r, m] / sig2[g, m], phi[r, m])
                c = np.r_[1.0, -phi[r, m]]
                sig2[g, m] = max(
                    (c @ Kmat[r, m] @ c + 2.0 * SIG_B) / (n + 2.0 * SIG_A + 2.0), SIGMA_FLOOR**2
                )
    cand = replace(p, phi=phi, sig2=sig2, tau_arm=float(tau))
    wcM = [_wc_M(L, cand.theta) for L in lay.modes]
    val = _log_prior(cand, R, lay)
    for i, wg in enumerate(grams):
        for m, L in enumerate(lay.modes):
            mm, Sm = q.mu[i, L.eff], q.S[i][np.ix_(L.eff, L.eff)]
            for g, (G, n) in wg[m].items():
                cell = _Cell(G, n, cand.sig2[g, m], _phi(cand, PHI_ROW[g], m), g, PHI_ROW[g], False)
                val += _ell(cell, wcM[m], mm, Sm)
    return val, cand


def _cm_tau(flights, q: _Q, p: Params, lay: _Layout, R: np.ndarray) -> Params:
    """Generalized CM-step over the arming time: the current time starts from ``p`` itself,
    so the chosen candidate never lowers the objective."""
    obs_s, obs_p = _observed(flights)
    best_val, best = _tau_value(flights, q, p, lay, R, p.tau_arm)
    base = _filled(p, obs_s, obs_p)
    for tau in TAU_GRID:
        if tau != p.tau_arm:
            val, cand = _tau_value(flights, q, base, lay, R, tau)
            if val > best_val:
                best, best_val = cand, val
    if best.tau_arm != p.tau_arm:
        for f in flights:
            _set_tau(f, lay, best.tau_arm)
    return best


def _m_step(flights, q: _Q, p: Params, lay: _Layout, R: np.ndarray, do_tau: bool) -> Params:
    _, obs_p = _observed(flights)
    cellss = [_cells(f, p, pa.gamma) for f, pa in zip(flights, q.paths)]
    theta = _cm_theta(flights, cellss, q, lay, R)
    p = replace(p, theta=theta)
    p = replace(p, phi=_cm_phi(cellss, q, lay, theta, p.phi, obs_p))
    cellss = [_cells(f, p, pa.gamma) for f, pa in zip(flights, q.paths)]
    sig2, kappa = _cm_sigma(flights, cellss, q, lay, p)
    nu, delta, D = _cm_levels(flights, q, p, lay)
    p = replace(p, sig2=sig2, kappa=kappa, nu=nu, delta=delta, D=D, **_cm_grammar(q, p))
    return _cm_tau(flights, q, p, lay, R) if do_tau else p


# ─── initialization ───────────────────────────────────────────────────────────


def _tau_guess(flights_raw) -> float:
    """Median settling time of the measured arming transients, snapped to the grid."""
    times = []
    for on, zc in flights_raw:
        keep = np.isfinite(on)
        on, zc = on[keep], zc[keep]
        if len(on) < 5:
            continue
        final = np.median(zc[len(zc) // 2 :])
        smooth = np.median(
            np.lib.stride_tricks.sliding_window_view(np.pad(zc, 2, mode="edge"), 5), axis=1
        )
        bad = np.flatnonzero(np.abs(smooth - final) > max(0.5, 0.1 * abs(final - Z_REST[0])))
        times.append(0.0 if len(bad) == 0 else on[min(bad[-1] + 1, len(on) - 1)])
    if not times:
        return 1.28
    t = max(float(np.median(times)), TAU_GRID[0])
    return float(min(TAU_GRID, key=lambda g: abs(np.log(g / t))))


def _point_q(mu: np.ndarray, gammas: list[np.ndarray], resp: np.ndarray) -> _Q:
    n, q = mu.shape
    paths = [
        _Path(
            0.0,
            r,
            g,
            np.zeros((N_PROGRAMMES, N_ACTIONS)),
            np.zeros((N_PROGRAMMES, N_ACTIONS, N_ACTIONS)),
            np.zeros((N_ACTIONS, DMAX)),
            np.zeros((N_ACTIONS, DMAX)),
        )
        for g, r in zip(gammas, resp)
    ]
    return _Q(mu.copy(), np.zeros((n, q, q)), paths, np.zeros(n))


def _total_gram(f: _Flight, m: int) -> np.ndarray:
    G = cast(np.ndarray, sum(G for G, _ in f.cur[m].values()))
    return G + f.cb[m].sum(axis=0) if f.nb else G


def _project(f: _Flight, m: int, L: _Mode, theta: np.ndarray, b: np.ndarray, block: str):
    """Least-squares coefficients of the static point residual on a basis block."""
    G = _total_gram(f, m)
    wc, M = _wc_M(L, theta)
    c = np.r_[1.0, 0.0, 0.0, -(wc + M @ b[L.eff])]
    us = L.cols[block] + 3
    BB = G[np.ix_(us, us)]
    if np.trace(BB) < 1.0:
        return None
    return _solve_spd(BB + 1e-6 * np.eye(len(us)), G[us] @ c)


def _initialize(
    flights: list[_Flight], lay: _Layout, R: np.ndarray, tau: float
) -> tuple[Params, _Q]:
    n, q = len(flights), lay.q
    K, A = N_PROGRAMMES, N_ACTIONS
    # 1. point equilibria
    lev = np.full((n, N_MODES, 2), np.nan)
    for i, f in enumerate(flights):
        g = f.ground_on >= tau
        if g.sum() >= 3:
            lev[i, :, 0] = np.median(f.ground_z[:, g], axis=1)
        elif f.ground_z.shape[1] >= 3:
            lev[i, :, 0] = np.median(f.ground_z[:, -max(3, f.ground_z.shape[1] // 4) :], axis=1)
        if f.hover_z.shape[1] >= 3:
            lev[i, :, 1] = np.median(f.hover_z, axis=1)
    for m in range(N_MODES):
        for e, default in ((0, 15.0 if m == 0 else 0.0), (1, 80.0 if m == 0 else 0.0)):
            col = lev[:, m, e]
            col[np.isnan(col)] = np.nanmean(col) if np.isfinite(col).any() else default
    b = np.zeros((n, q))
    for m, L in enumerate(lay.modes):
        b[:, L.eff[0]], b[:, L.eff[1]] = lev[:, m, 0], lev[:, m, 1]
    off = ~np.eye(A, dtype=bool)
    log_trans = np.where(off, np.log(1.0 / (A - 1)), -np.inf)
    p = Params(
        theta=np.zeros(lay.n_theta),
        nu=np.zeros(q),
        delta=np.zeros(q),
        D=np.ones(q),
        phi=np.zeros((N_PG + A, N_MODES, 2)),
        sig2=np.full((N_SG + A, N_MODES), 0.25),
        kappa=np.ones(N_MODES),
        log_pi=np.full(K, -np.log(K)),
        log_init=np.full((K, A), -np.log(A)),
        log_trans=np.repeat(log_trans[None], K, axis=0),
        dur=np.tile([np.log(9.0), 0.0], (A, 1)),
        tau_arm=float(tau),
    )
    uniform = [np.full((f.nb, A), 1.0 / A) for f in flights]
    flat = np.full((n, K), 1.0 / K)
    _, obs_p = _observed(flights)

    def cells(pp, qq):
        return [_cells(f, pp, pa.gamma) for f, pa in zip(flights, qq.paths)]

    # 2. static mean, then shape and fan-out loadings from residual projections
    qp = _point_q(b, uniform, flat)
    theta = _cm_theta(flights, cells(p, qp), qp, lay, R)
    common = lay.modes[0]
    for name, block, j in (
        ("takeoff_shape", "to_lr", 2),
        ("landing_shape", "la_lr", 3),
        ("arming_shape", "on_j", 4),
    ):
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
    qp = _point_q(b, uniform, flat)
    p = replace(p, theta=_cm_theta(flights, cells(replace(p, theta=theta), qp), qp, lay, R))
    # 3. relaxation, then mean and noise again
    for _ in range(2):
        p = replace(p, phi=_cm_phi(cells(p, qp), qp, lay, p.theta, p.phi, obs_p))
        p = replace(p, theta=_cm_theta(flights, cells(p, qp), qp, lay, R))
    sig2, _ = _cm_sigma(flights, cells(p, qp), qp, lay, p)
    p = replace(p, sig2=sig2)
    # 4. actions from cruise residual-energy terciles; programmes by activity rank
    wcM = [_wc_M(L, p.theta) for L in lay.modes]
    energy = []
    for i, f in enumerate(flights):
        e = np.zeros(f.nb)
        for m, L in enumerate(lay.modes):
            ph = p.phi[N_PG, m]
            g = 1.0 - ph.sum()
            c = np.r_[1.0, -ph[0], -ph[1], -g * (wcM[m][0] + wcM[m][1] @ b[i, L.eff])]
            e += (f.cb[m] @ c) @ c / np.maximum(f.nblk, 1.0) / p.sig2[N_SG, m]
        energy.append(np.log(np.maximum(e, 1e-12)))
    allE = np.concatenate(energy) if energy else np.zeros(0)
    cuts = np.quantile(allE, np.arange(1, A) / A) if len(allE) else np.zeros(A - 1)
    labels = [np.searchsorted(cuts, e) for e in energy]
    activity = np.array([lab.mean() if len(lab) else np.nan for lab in labels])
    activity[np.isnan(activity)] = np.nanmean(activity) if np.isfinite(activity).any() else 0.0
    rank = np.argsort(np.argsort(activity, kind="stable"), kind="stable")
    resp = np.zeros((n, K))
    resp[np.arange(n), np.minimum(rank * K // n, K - 1)] = 1.0
    paths = []
    for i, (f, lab) in enumerate(zip(flights, labels)):
        k = int(np.argmax(resp[i]))
        gam = np.zeros((f.nb, A))
        gam[np.arange(f.nb), lab] = 1.0
        pa = _Path(
            0.0,
            resp[i],
            gam,
            np.zeros((K, A)),
            np.zeros((K, A, A)),
            np.zeros((A, DMAX)),
            np.zeros((A, DMAX)),
        )
        if f.nb:
            starts = np.r_[0, np.flatnonzero(np.diff(lab)) + 1]
            lens = np.diff(np.r_[starts, f.nb])
            pa.init[k, lab[0]] = 1.0
            for s0, s1 in zip(starts[:-1], starts[1:]):
                pa.trans[k, lab[s0], lab[s1]] += 1.0
            for s0, d in zip(starts[:-1], lens[:-1]):
                pa.dur[lab[s0], min(d, DMAX) - 1] += 1.0
            pa.cens[lab[starts[-1]], min(lens[-1], DMAX) - 1] += 1.0
        paths.append(pa)
    qh = _Q(b.copy(), np.zeros((n, q, q)), paths, np.zeros(n))
    p = replace(p, phi=_cm_phi(cells(p, qh), qh, lay, p.theta, p.phi, obs_p))
    sig2, kappa = _cm_sigma(flights, cells(p, qh), qh, lay, p)
    nu, delta, D = _cm_levels(flights, qh, p, lay)
    p = replace(p, sig2=sig2, kappa=kappa, nu=nu, delta=delta, D=D, **_cm_grammar(qh, p))
    return p, qh


# ─── model ────────────────────────────────────────────────────────────────────


def _config() -> dict[str, Any]:
    return {
        "family": FAMILY,
        "version": VERSION,
        "fs": FS,
        "w_floor": W_FLOOR,
        "sigma_floor": SIGMA_FLOOR,
        "bases": {k: [e, list(b)] for k, (e, b) in BASES.items()},
        "tau_grid": list(TAU_GRID),
        "n_actions": N_ACTIONS,
        "n_programmes": N_PROGRAMMES,
        "block": BLOCK,
        "dmax": DMAX,
        "groups": list(SG_NAMES),
        "relaxation_groups": list(PG_NAMES),
        "loadings": dict(LOADINGS),
    }


def _tracks(tracks, measured, lay: _Layout, tau: float) -> list[_Flight]:
    if len(tracks) == 0:
        raise ValueError("need at least one training track")
    measured = [True] * len(tracks) if measured is None else [bool(v) for v in measured]
    if len(measured) != len(tracks):
        raise ValueError("measured must have one flag per track")
    return [_prepare(w, c, ms, lay, tau) for (w, c), ms in zip(tracks, measured)]


_ARRAYS = (
    "theta",
    "nu",
    "delta",
    "D",
    "phi",
    "sig2",
    "kappa",
    "log_pi",
    "log_init",
    "log_trans",
    "dur",
)


class ControllerLawModel:
    """Fitted hierarchical feedback-controller law; see the module docstring."""

    def __init__(
        self,
        params: Params,
        observed_sigma: np.ndarray,
        observed_phi: np.ndarray,
        *,
        has_cmd: bool = False,
        has_cmd_cruise: bool = False,
        history: list[float] | None = None,
        fit_info: dict[str, Any] | None = None,
    ):
        self.params = params
        self.observed_sigma = np.asarray(observed_sigma, dtype=bool)
        self.observed_phi = np.asarray(observed_phi, dtype=bool)
        self.has_cmd = bool(has_cmd)
        self.has_cmd_cruise = bool(has_cmd_cruise)
        self.history = list(history or [])
        self.fit_info = dict(fit_info or {})
        self.layout = _build_layout()
        self.last_clip_fraction = 0.0

    def param_breakdown(self) -> dict[str, int]:
        lay, A, K = self.layout, N_ACTIONS, N_PROGRAMMES
        out = {
            "mean_common": lay.modes[0].beta.stop - lay.modes[0].beta.start,
            "mean_rotor_diff": sum(L.beta.stop - L.beta.start for L in lay.modes[1:]),
        }
        out.update({f"loading_{k}": sl.stop - sl.start for k, sl in lay.load.items()})
        out.update(
            {
                "level_means": int(lay.level.sum()),
                "level_var": int(lay.level.sum()),
                "source_intercepts": int(lay.source.sum()) if self.has_cmd else 0,
                "relaxation_ar": 2 * N_MODES * int(self.observed_phi.sum()),
                "noise_var": N_MODES * int(self.observed_sigma.sum()),
                "cmd_cruise_scale": N_MODES if self.has_cmd_cruise else 0,
                "programme_weights": K - 1,
                "action_init": K * (A - 1),
                "action_trans": K * A * (A - 2),
                "durations": 2 * A,
                "arming_time": 1,
            }
        )
        return out

    @property
    def n_params(self) -> int:
        return int(sum(self.param_breakdown().values()))

    def elbo(
        self,
        tracks: list[tuple[np.ndarray, np.ndarray]],
        measured: list[bool] | None = None,
        max_inner: int = 50,
    ) -> np.ndarray:
        """Variational lower bound of each whole track's log-density (floored
        speeds, Jacobian included), coordinate ascent run to convergence."""
        p, lay = self.params, self.layout
        flights = _tracks(tracks, measured, lay, p.tau_arm)
        wcM = [_wc_M(L, p.theta) for L in lay.modes]
        tables = _dur_tables(p.dur)
        out = []
        for f in flights:
            m0, Psi = _prior(p, f)
            out.append(_flight_q(f, p, lay, wcM, tables, m0, np.diag(Psi), max_inner, tol=1e-10)[3])
        return np.array(out)

    def sample(self, schedule: Any, n: int, rng: np.random.Generator) -> np.ndarray:
        """``(n, 4, T)`` nonnegative rotor speeds (rev/s) of measured records on the schedule's clock."""
        p = _filled(self.params, self.observed_sigma, self.observed_phi)
        lay = self.layout
        codes = _check_codes(schedule)
        T = len(codes)
        clk = event_clocks(codes)
        blocks = _design(clk, True, p.tau_arm)
        groups = _sgroups(codes, clk, True, p.tau_arm)
        cruise = np.flatnonzero(groups < 0)
        nb = -(-len(cruise) // BLOCK)
        k = rng.choice(N_PROGRAMMES, size=n, p=np.exp(p.log_pi - logsumexp(p.log_pi)))
        b = p.nu + np.sqrt(p.D) * rng.standard_normal((n, lay.q))
        pmf = np.exp(_dur_tables(p.dur)[0])
        sg = np.repeat(groups[None], n, axis=0)
        if nb:
            acts = np.stack([_sample_actions(nb, int(ki), p, pmf, rng) for ki in k])
            sg[:, cruise] = N_SG + acts[:, np.arange(len(cruise)) // BLOCK]
        prow = np.where(sg < N_SG, np.asarray(PHI_ROW)[np.minimum(sg, N_SG - 1)], sg - N_SG + N_PG)
        phi_tab = np.concatenate([p.phi, np.zeros((1, N_MODES, 2))])  # row -1 -> phi = 0
        ph = phi_tab[prow]  # (n, T, 4, 2)
        p1, p2 = ph[..., 0].transpose(0, 2, 1), ph[..., 1].transpose(0, 2, 1)
        gain = 1.0 - p1 - p2
        sd = np.sqrt(p.sig2[sg]).transpose(0, 2, 1)
        r = np.empty((n, N_MODES, T))
        for m, L in enumerate(lay.modes):
            U = _umat(L, blocks)
            wc, M = _wc_M(L, p.theta)
            r[:, m] = U @ wc + b[:, L.eff] @ (U @ M).T
        drive = gain * r + sd * rng.standard_normal((n, N_MODES, T))
        Z = np.empty((n, N_MODES, T))
        z1 = z2 = np.broadcast_to(Z_REST, (n, N_MODES))
        for t in range(T):
            zt = p1[:, :, t] * z1 + p2[:, :, t] * z2 + drive[:, :, t]
            Z[:, :, t] = zt
            z1, z2 = zt, z1
        W = from_obs(np.einsum("rm,nmt->nrt", MIXER, Z))
        powered = np.isin(codes, (1, 2))
        self.last_clip_fraction = (
            float((W[:, :, powered] == 0.0).any(axis=1).mean()) if powered.any() else 0.0
        )
        return W

    def summary(self) -> dict[str, Any]:
        """Training-free readout: arming time, relaxation poles, noise, levels, grammar."""
        p, lay = self.params, self.layout
        relax = {}
        for r, name in enumerate((*PG_NAMES, *(f"cruise{a}" for a in range(N_ACTIONS)))):
            if not self.observed_phi[r]:
                continue
            relax[name] = {}
            for m, mode in enumerate(MODE_NAMES):
                roots = np.roots([1.0, -p.phi[r, m, 0], -p.phi[r, m, 1]])
                relax[name][mode] = {
                    "phi": p.phi[r, m].tolist(),
                    "pole_radius": np.abs(roots).tolist(),
                    "pole_hz": (np.abs(np.angle(roots)) * FS / (2 * np.pi)).tolist(),
                }
        names = (*SG_NAMES, *(f"cruise{a}" for a in range(N_ACTIONS)))
        lv = np.flatnonzero(lay.level)
        return {
            "tau_arm_s": p.tau_arm,
            "relaxation": relax,
            "noise_sd": {
                names[g]: np.sqrt(p.sig2[g]).tolist() for g in np.flatnonzero(self.observed_sigma)
            },
            "cmd_cruise_scale": p.kappa.tolist() if self.has_cmd_cruise else None,
            "levels": {
                lay.names[j]: {
                    "mean": float(p.nu[j]),
                    "sd": float(np.sqrt(p.D[j])),
                    "cmd_intercept": float(p.delta[j]) if self.has_cmd else None,
                }
                for j in lv
            },
            "programme_weights": np.exp(p.log_pi).tolist(),
            "action_init": np.exp(p.log_init).tolist(),
            "action_trans": np.exp(p.log_trans).tolist(),
            "action_mean_s": ((1.0 + np.exp(p.dur[:, 0])) * BLOCK / FS).tolist(),
            "param_breakdown": self.param_breakdown(),
        }

    def to_dict(self) -> dict[str, Any]:
        p = self.params
        params = {
            k: np.where(np.isfinite(getattr(p, k)), getattr(p, k), -1e300).tolist() for k in _ARRAYS
        }
        params["tau_arm"] = p.tau_arm
        return {
            "config": _config(),
            "params": params,
            "observed_sigma": self.observed_sigma.tolist(),
            "observed_phi": self.observed_phi.tolist(),
            "has_cmd": self.has_cmd,
            "has_cmd_cruise": self.has_cmd_cruise,
            "n_params": self.n_params,
            "param_breakdown": self.param_breakdown(),
            "history": [float(v) for v in self.history],
            "fit_info": self.fit_info,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ControllerLawModel:
        if d["config"] != _config():
            raise ValueError(
                "serialized controller_law model has a different configuration than this module"
            )
        q = d["params"]
        arr = {k: np.asarray(q[k], dtype=np.float64) for k in _ARRAYS}
        arr["log_trans"] = np.where(arr["log_trans"] <= -1e300, -np.inf, arr["log_trans"])
        return cls(
            Params(**arr, tau_arm=float(q["tau_arm"])),
            np.asarray(d["observed_sigma"], dtype=bool),
            np.asarray(d["observed_phi"], dtype=bool),
            has_cmd=d["has_cmd"],
            has_cmd_cruise=d["has_cmd_cruise"],
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
) -> ControllerLawModel:
    """Variational ECM on the penalized whole-record lower bound.

    ``tracks``: ``(speeds (4, T) rev/s MIXER order, codes (T,))`` at 25 Hz;
    ``measured``: per-track source flag (False = command-only).
    """
    lay = _build_layout()
    R = _penalty(lay)
    probe = _tracks(tracks, measured, lay, TAU_GRID[0])
    tau = _tau_guess([(f.ground_on, f.ground_z[0]) for f in probe])
    flights = probe
    for f in flights:
        _set_tau(f, lay, tau)
    p, q = _initialize(flights, lay, R, tau)
    n_valid = int(sum(f.T for f in flights))
    q = _e_step(flights, p, lay, q)
    obj = float(q.elbo.sum()) + _log_prior(p, R, lay)
    history = [obj]
    trace = [
        {
            "iter": 0,
            "objective": obj,
            "elbo": float(q.elbo.sum()),
            "tau_arm": p.tau_arm,
            "gain_per_sample": None,
        }
    ]
    stop, rejected = "max_iter", None
    for it in range(1, max_iter + 1):
        old_tau = p.tau_arm
        try:
            cand = _m_step(flights, q, p, lay, R, do_tau=(it - 1) % TAU_EVERY == 0)
            q_c = _e_step(flights, cand, lay, q)
            obj_c = float(q_c.elbo.sum()) + _log_prior(cand, R, lay)
        except linalg.LinAlgError:
            cand, obj_c = p, -np.inf
        if not np.isfinite(obj_c) or obj_c < obj - MONOTONE_RTOL * max(1.0, abs(obj)):
            if cand.tau_arm != old_tau:
                for f in flights:
                    _set_tau(f, lay, old_tau)
            stop = "non_monotone"
            rejected = {"iter": it, "objective": float(obj_c) if np.isfinite(obj_c) else None}
            break
        gain = (obj_c - obj) / n_valid
        p, q, obj = cand, q_c, obj_c
        history.append(obj)
        trace.append(
            {
                "iter": it,
                "objective": obj,
                "elbo": float(q.elbo.sum()),
                "tau_arm": p.tau_arm,
                "gain_per_sample": gain,
            }
        )
        if gain < tol and it > TAU_EVERY:
            stop = "converged"
            break
    obs_s, obs_p = _observed(flights)
    flights_info = [
        {
            "measured": f.measured,
            "samples": f.T,
            "cruise_blocks": f.nb,
            "elbo": float(q.elbo[i]),
            "programme_resp": q.paths[i].resp.tolist(),
            "action_occupancy": q.paths[i].gamma.mean(axis=0).tolist() if f.nb else None,
            "level_posterior": {lay.names[j]: float(q.mu[i, j]) for j in np.flatnonzero(lay.level)},
        }
        for i, f in enumerate(flights)
    ]
    fit_info = {
        "trace": trace,
        "stop_reason": stop,
        "n_iter": len(history) - 1,
        "n_valid": n_valid,
        "n_flights": len(flights),
        "objective": obj,
        "elbo": float(q.elbo.sum()),
        "objective_kind": "variational lower bound (ELBO) + log-prior",
        "monotone": bool(np.all(np.diff(history) >= -MONOTONE_RTOL * max(1.0, abs(obj)))),
        "rejected": rejected,
        "tau_arm": p.tau_arm,
        "flights": flights_info,
    }
    return ControllerLawModel(
        p,
        obs_s,
        obs_p,
        has_cmd=any(not f.measured for f in flights),
        has_cmd_cruise=any(not f.measured and f.nb for f in flights),
        history=history,
        fit_info=fit_info,
    )


__all__ = ["ControllerLawModel", "Params", "event_clocks", "fit", "from_obs", "to_obs"]
