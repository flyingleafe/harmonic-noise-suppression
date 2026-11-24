"""Program-response generator: event-clock reference, action HSMM and AR residual.

A whole-flight likelihood model of the four rotor speeds given only the coarse
commanded schedule (one phase code per 25 Hz sample: 0 before motors-on,
1 motors-on to take-off, 2 take-off to landing, 3 landing to the end) and the
source modality of the record (measured shaft speed, or a command-only
reconstruction).  One generative process, one likelihood, one generalized-EM
fit for the whole recording; nothing is fitted per phase or per flight.

Hierarchy (outermost first)
---------------------------
1. **Flight** (persistent, drawn once, integrated over every sample):
   an anchor node ``q = (q_I, q_H)`` and an activity class ``a``.
   The idle and hover anchors (4-vectors in mixer modes) are
   ``I = Ibar + lam_I xi_{q_I}``, ``H = Hbar + lam_H xi_{q_H}``, where
   ``xi`` runs over the :data:`N_NODES`-point Gauss-Hermite law of a standard
   normal (fixed nodes and weights).  The anchor law IS this
   :data:`N_GRID`-point product law (a discretised two-factor Gaussian); it is
   summed exactly in the likelihood and drawn exactly by the sampler, so no
   quadrature error separates the two.  ``a in {0, 1}`` with learned weights:
   class 1 scales every cruise innovation SD by a per-mode factor ``kappa`` and
   uses its own manoeuvre-drive vector (calm vs. active flights).
2. **Phase** (supervised): the schedule codes.  The event clocks
   ``tau_on, tau_to, tau_land`` (seconds from motors-on, take-off and the
   landing mark) are known for the whole record.
3. **Action** (latent semi-Markov chain inside each phase, reset at every
   code change): standby ``idle <-> pulse``; airborne ``rise`` then three
   persistent parent contexts -- quiet hold, isolated events (hold -> one
   signed drive -> hold) and alternating drive (+ <-> - with brief holds); the
   parent only changes from a hold, so it shapes what follows a drive.
   Landing ``descend -> (shelf) -> stop -> rest``.  Dwell of each macro state
   is negative binomial (Erlang-like clock stages sharing one stay
   probability) -- :class:`experiments.rps_traj.unified_grammar.Grammar`
   supplies the graph, its exact sparse log-domain forward-backward kernels,
   MAP update and sampler.
4. **Clock stage** inside an action: the emission class is resolved per
   stage for the manoeuvre drives (``up0..2``/``down0..2``: a within-action
   drive profile ``rho_k u_a``) and for ``stop0..2`` (stage-specific
   spin-down drives that produce the post-landing fan-out at the latent stop
   time, not at a fixed clock time).

Observation link and modes
--------------------------
Rotor speeds ``w >= 0`` (rev/s) are mapped per rotor through the inverse
softplus ``z = s log(expm1(max(w, eps) / s))`` (``s`` = :data:`LINK_SCALE`,
``eps`` = :data:`FLOOR_RPS`), then to mixer modes ``x = MIXER^T z / 4``
(common, roll, pitch, yaw).  The generator maps back with
``w = s softplus(z / s) > 0`` -- the physical readout is positive by
construction, and stable AR overshoot in ``z`` below rest compresses into
``(0, eps)`` instead of producing negative speeds.  **Zero handling
(disclosed approximation):** a logged speed ``<= eps`` (0.05 rev/s; the logs
hold exact zeros and small negative interpolation residue at rest) is scored
as the density at ``eps``, not as the censored mass ``P(w <= eps)``;
symmetrically, the readout of a draw reports ``0`` where ``w <= eps``.
Above ``eps`` the readout is exactly the fitted law.  The reported
likelihood is in rev/s units (link and mixer Jacobians included).

Reference (the supervised, event-locked part)
---------------------------------------------
``mu_t = R (1 - F_on) + I (F_on - F_to + F_land) + H (F_to - F_land) + fan_t``
with ``R`` the rest point (all rotors at ``eps``), ``F_on(tau_on)``,
``F_to(tau_to)``, ``F_land(tau_land)`` piecewise-linear fractions on fixed
knots (seconds) with learned knot values (fixed endpoints: 0 before motors-on,
1 after the startup/take-off windows; the landing tail is learned, which
places the post-landing shelf between hover and idle), and ``fan`` a
piecewise-linear event-locked bump on the three differential modes around
the landing mark.  The landing knots are dense around the mark (one sample
apart at -0.08..+0.08 s) so a descent that finishes within ~0.08 s (DREGON)
and a slow shelf (MD2, M100) are both representable with learned
coefficients.  The same fractions carry the trims (differential modes) from
idle to hover and back -- a low-dimensional mode handover.  The schedule
fixes the event clocks; nothing is a per-flight template or plug-in level.

Residual (the action-conditioned dynamics)
------------------------------------------
For emission class ``k`` (from the action/stage), reference ``ref^k`` (``mu``
or, for ``rest``/``stop*``, the rest point ``R``), per mode ``d``::

    x_t - ref^k_t = phi1 (x_{t-1} - ref^k_{t-1}) + phi2 (x_{t-2} - ref^k_{t-2})
                    + u_k(a) + s_k(a) eps_t

with ``(phi1, phi2)`` per AR stream (rest/stop, standby, transition, cruise)
and mode, constrained to the AR(2) stability triangle (any frequency/damping
per mode, e.g. a 1-4 Hz yaw resonance); ``u_k`` the action drive; ``s_k`` the
innovation SD of the class's SD group and mode, times ``kappa_d`` for active
flights in cruise classes, times the command-source scale ``c_d`` on
command-only records.  The pre-sample residual state is zero
(``x_{-1} - ref = x_{-2} - ref = 0``) in BOTH the likelihood and the sampler,
so every sample carries a factor and the sampler draws exactly the scored law.
The conditional density is valid for any action path (the lags are
re-referenced to the current class), so switching ``descend -> stop`` is a
proper Markov switch, not a re-initialisation.

Likelihood::

    p(x | schedule) = sum_q w_q sum_a pi_a sum_paths p(path) prod_t N(e_t; 0, diag s^2)

exact: the path sum is the grammar's forward recursion on per-sample class
log-emissions, the node and activity sums are explicit.  Factors whose window
touches a non-finite sample are dropped (conditional restart; the packed
tracks have none).  **Source model:** a command-only record (``measured`` =
False, DREGON ``motors_command``) has its pre-take-off samples (standby
recovered from audio, not a measurement) treated as unobserved -- no factor,
but they serve as the lags of the first airborne factors -- and its airborne
innovations carry the learned per-mode source scale ``c``.  Draws are
measured-modality by default (``measured=True``).

Fitting (generalized EM on the penalized observed likelihood)
------------------------------------------------------------
E-step: per training record, the forward log-likelihood of every
``(node, activity)`` pair, the whole-record responsibilities and a
forward-backward pass for the pairs above :data:`RESP_FLOOR` (pairs below
carry negligible mass; the objective itself is always exact).  M-step:
sequential exact conditional maximisers of the expected complete-data
log-likelihood plus log-prior: graph (closed-form MAP), activity weights
(Dirichlet MAP), anchor law (WLS for ``Ibar, lam_I, Hbar, lam_H`` per mode),
reference shapes and fan (one joint WLS, shared across modes, with a
second-difference smoothness prior), AR coefficients (2-D QP on the
stability triangle), drives, stage profile, innovation SDs (inverse-gamma
MAP with floor), activity and source scales (inverse-gamma MAP).  The
objective is evaluated after every E-step; a decrease beyond rounding
reverts and stops.  Initialisation is deterministic and training-only
(pooled phase medians and pooled fraction regressions); every quantity is
then refitted by EM.

Parameter count (all learned numbers; knot positions, node grid, priors and
graph topology are structure): see :meth:`ProgramResponseModel.n_params_breakdown`
(~220, below the 277 budget).

Limitations: diagonal innovation covariance in mixer modes; a two-point
activity law; the anchor law is a 25-point discretised Gaussian; the
manoeuvre drive direction ``u_a`` is shared by all modes of one action (the
modes move together within an action, with per-mode AR and noise).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
from numba import njit
from scipy.special import logsumexp

from experiments.rps_traj import unified_grammar as ug
from experiments.rps_traj.unified_grammar import _backward, _forward, _sample
from tracking.rotors import MIXER, NUM_ROTORS

RATE_HZ = 25.0
N_MODES = 4
#: Numerical floor of the observation link (rev/s) and its softness.
FLOOR_RPS = 0.05
LINK_SCALE = 1.0
#: Innovation SD floor (link units) and its inverse-gamma prior.
SIGMA_FLOOR = 0.01
SIGMA_PRIOR_ALPHA = 1.0
SIGMA_PRIOR_BETA = 0.25
#: Inverse-gamma prior on kappa^2 and c^2 (mode 1).
SCALE_PRIOR_ALPHA = 2.0
SCALE_PRIOR_BETA = 3.0
#: Weak Gaussian priors.
DRIVE_SD = 30.0
RHO_SD = 3.0
ANCHOR_SD = 1000.0
LOAD_SD = 30.0
SHAPE_SMOOTH_SD = 0.25
SHAPE_RIDGE_SD = 5.0
FAN_SD = 20.0
PI_CONCENTRATION = 2.0
STABILITY_MARGIN = 1e-3
RESP_FLOOR = 1e-10
MONOTONE_RTOL = 1e-9

N_NODES = 5
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(N_NODES)
GH_NODES = _GH_X.astype(np.float64)
GH_WEIGHTS = (_GH_W / _GH_W.sum()).astype(np.float64)
#: Product grid (q_I, q_H) flattened as q = q_I * N_NODES + q_H.
NODE_XI = np.stack(np.meshgrid(GH_NODES, GH_NODES, indexing="ij"), axis=-1).reshape(-1, 2)
NODE_LOGW = (np.log(GH_WEIGHTS)[:, None] + np.log(GH_WEIGHTS)[None, :]).ravel()
N_GRID = N_NODES * N_NODES

Z_REST = float(LINK_SCALE * np.log(np.expm1(FLOOR_RPS / LINK_SCALE)))
REST = (MIXER.T @ np.full(NUM_ROTORS, Z_REST)) / NUM_ROTORS
LOG_DET_MODES = float(np.log(abs(np.linalg.det(MIXER.T / NUM_ROTORS))))

# ─── emission classes ─────────────────────────────────────────────────────────

CLASSES = (
    "rest",
    "stop0",
    "stop1",
    "stop2",
    "idle",
    "pulse",
    "rise",
    "quiet",
    "hold",
    "up0",
    "up1",
    "up2",
    "down0",
    "down1",
    "down2",
    "descend",
    "shelf",
)
N_CLASSES = len(CLASSES)
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
STREAMS = ("rest", "standby", "transition", "cruise")
SIGMA_GROUPS = ("rest", "stop", "standby", "rise", "quiet", "hold", "drive", "landing")
DRIVE_ROWS = (
    "pulse",
    "rise",
    "descend",
    "shelf",
    "stop0",
    "stop1",
    "stop2",
    "manoeuvre_a0",
    "manoeuvre_a1",
)
CLS_STREAM = np.array([0, 0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2], dtype=np.int64)
CLS_SIG = np.array([0, 1, 1, 1, 2, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 7, 7], dtype=np.int64)
CLS_ACT = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
CLS_DRIVE = np.array([-1, 4, 5, 6, -1, 0, 1, -1, -1, 7, 7, 7, 7, 7, 7, 2, 3], dtype=np.int64)
CLS_SIGN = np.array([0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, -1, -1, -1, 1, 1], dtype=np.float64)
CLS_STAGE = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 0, 1, 2, -1, -1], dtype=np.int64)
CLS_REST = CLS_STREAM == 0
STREAM_CLASSES = tuple(np.flatnonzero(s == CLS_STREAM) for s in range(len(STREAMS)))

# (name, phase, label, stages, successors, mean dwell s, branch law, entry weight, class per stage)
_MACROS: tuple[tuple, ...] = (
    ("off.rest", 0, "rest", 1, (), 0.0, (), 1.0, ("rest",)),
    ("start.idle", 1, "idle", 2, ("start.pulse",), 8.0, (1.0,), 0.9, ("idle",) * 2),
    ("start.pulse", 1, "pulse", 3, ("start.idle",), 1.0, (1.0,), 0.1, ("pulse",) * 3),
    (
        "air.rise",
        2,
        "rise",
        3,
        ("quiet.hold", "event.hold", "alt.hold"),
        2.0,
        (0.4, 0.4, 0.2),
        0.93,
        ("rise",) * 3,
    ),
    (
        "quiet.hold",
        2,
        "hold",
        2,
        ("event.hold", "alt.hold"),
        10.0,
        (0.6, 0.4),
        0.01,
        ("quiet",) * 2,
    ),
    (
        "event.hold",
        2,
        "hold",
        2,
        ("event.up", "event.down", "quiet.hold", "alt.hold"),
        4.0,
        (0.35, 0.35, 0.15, 0.15),
        0.01,
        ("hold",) * 2,
    ),
    ("event.up", 2, "drive_up", 3, ("event.hold",), 1.2, (1.0,), 0.01, ("up0", "up1", "up2")),
    (
        "event.down",
        2,
        "drive_down",
        3,
        ("event.hold",),
        1.2,
        (1.0,),
        0.01,
        ("down0", "down1", "down2"),
    ),
    (
        "alt.hold",
        2,
        "hold",
        2,
        ("alt.up", "alt.down", "quiet.hold", "event.hold"),
        0.6,
        (0.4, 0.4, 0.1, 0.1),
        0.01,
        ("hold",) * 2,
    ),
    (
        "alt.up",
        2,
        "drive_up",
        3,
        ("alt.down", "alt.hold"),
        0.5,
        (0.8, 0.2),
        0.01,
        ("up0", "up1", "up2"),
    ),
    (
        "alt.down",
        2,
        "drive_down",
        3,
        ("alt.up", "alt.hold"),
        0.5,
        (0.8, 0.2),
        0.01,
        ("down0", "down1", "down2"),
    ),
    (
        "land.descend",
        3,
        "descend",
        3,
        ("land.shelf", "land.stop"),
        1.0,
        (0.6, 0.4),
        0.8,
        ("descend",) * 3,
    ),
    ("land.shelf", 3, "shelf", 3, ("land.stop",), 1.5, (1.0,), 0.05, ("shelf",) * 3),
    ("land.stop", 3, "stop", 3, ("land.rest",), 1.5, (1.0,), 0.1, ("stop0", "stop1", "stop2")),
    ("land.rest", 3, "rest", 1, (), 0.0, (), 0.05, ("rest",)),
)
_STAGE_CLASSES = {m[0]: m[8] for m in _MACROS}


def default_grammar() -> ug.Grammar:
    """The action graph at its data-free defaults (every probability nonzero)."""
    macros, stay, branch, entry = [], [], [], []
    for name, phase, label, k, succ, dwell, law, w, _ in _MACROS:
        macros.append(ug.MacroSpec(name, phase, ug.LABEL_ID[label], k, succ))
        stay.append(1.0 if not succ else 1.0 - k / max(dwell * RATE_HZ, k + 1.0))
        branch.append(np.array(law, dtype=np.float64))
        entry.append(w)
    entry_a = np.array(entry)
    phases = np.array([m.phase for m in macros])
    for p in range(ug.N_PHASES):
        entry_a[phases == p] /= entry_a[phases == p].sum()
    return ug.Grammar(
        macros=tuple(macros),
        stay=np.array(stay),
        branch=tuple(branch),
        entry=entry_a,
        hierarchical=True,
        rate_hz=RATE_HZ,
    )


class _Graph:
    """The grammar with stage-resolved emission classes; exact kernels of
    :mod:`unified_grammar` on a class-valued state labelling."""

    def __init__(self, grammar: ug.Grammar) -> None:
        names = [m.name for m in grammar.macros]
        if names != [m[0] for m in _MACROS]:
            raise ValueError("grammar macro states differ from the program-response graph")
        c = grammar.compiled
        stage = np.arange(grammar.n_states) - c.first_state[c.state_macro]
        self.lab = np.array(
            [CLASS_ID[_STAGE_CLASSES[names[m]][st]] for m, st in zip(c.state_macro, stage)],
            dtype=np.int64,
        )
        self.mask = np.zeros((ug.N_PHASES, N_CLASSES), dtype=np.bool_)
        self.mask[c.state_phase, self.lab] = True
        self.grammar = grammar

    def loglik(self, log_e: np.ndarray, phases: np.ndarray) -> float:
        c = self.grammar.compiled
        return float(
            _forward(
                log_e,
                phases,
                self.lab,
                c.row_ptr,
                c.col,
                c.log_prob,
                c.log_init,
                self.mask,
                np.empty((0, 0)),
                np.empty(0),
                False,
            )
        )

    def forward_backward(self, log_e: np.ndarray, phases: np.ndarray):
        c = self.grammar.compiled
        T, S = log_e.shape[0], self.grammar.n_states
        la, norm = np.empty((T, S)), np.empty(T)
        ll = float(
            _forward(
                log_e,
                phases,
                self.lab,
                c.row_ptr,
                c.col,
                c.log_prob,
                c.log_init,
                self.mask,
                la,
                norm,
                True,
            )
        )
        if not np.isfinite(ll):
            raise FloatingPointError("observations impossible under the action graph")
        occ = np.zeros((T, N_CLASSES))
        edges = np.zeros(c.col.shape[0])
        entries = np.zeros(S)
        _backward(
            log_e,
            phases,
            self.lab,
            c.row_ptr,
            c.col,
            c.log_prob,
            self.mask,
            la,
            norm,
            occ,
            edges,
            entries,
        )
        return ll, occ, edges, entries

    def sample(self, phases: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
        c = self.grammar.compiled
        u = rng.random((n, phases.shape[0]))
        out = np.empty((n, phases.shape[0]), dtype=np.int64)
        _sample(phases, u, self.lab, c.row_ptr, c.col, c.prob, c.init, out)
        return out


# ─── event-clock shapes ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Shape:
    """Piecewise-linear curve on knots (s): learned values at ``free``,
    fixed values at ``fixed``; constant beyond the outer knots."""

    name: str
    free: tuple[float, ...]
    fixed: tuple[tuple[float, float], ...]

    @property
    def n(self) -> int:
        return len(self.free)

    def nodes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = sorted(
            [(k, i, 0.0) for i, k in enumerate(self.free)] + [(k, -1, v) for k, v in self.fixed]
        )
        knots = np.array([r[0] for r in rows])
        if np.any(np.diff(knots) <= 0):
            raise ValueError(f"{self.name}: knots must be distinct")
        return knots, np.array([r[1] for r in rows], dtype=np.int64), np.array([r[2] for r in rows])


SHAPE_ON = _Shape(
    "startup",
    (0.0, 0.04, 0.08, 0.12, 0.2, 0.32, 0.48, 0.64, 0.8, 1.0, 1.2, 1.5, 2.0, 2.6, 3.4, 4.4),
    ((-0.04, 0.0), (6.0, 1.0)),
)
SHAPE_TO = _Shape(
    "takeoff",
    (-1.0, -0.5, -0.2, 0.08, 0.3, 0.6, 0.9, 1.3, 1.8, 2.5, 3.5),
    ((-1.6, 0.0), (5.0, 1.0)),
)
SHAPE_LAND = _Shape(
    "landing",
    (
        -3.0,
        -2.0,
        -1.4,
        -1.0,
        -0.65,
        -0.4,
        -0.24,
        -0.12,
        -0.04,
        0.0,
        0.04,
        0.08,
        0.16,
        0.36,
        0.7,
        1.2,
        2.0,
    ),
    ((-4.0, 0.0),),
)
SHAPE_FAN = _Shape("landing_fan", (-0.2, 0.4, 1.0, 1.7), ((-0.8, 0.0), (2.8, 0.0)))
SHAPES = (SHAPE_ON, SHAPE_TO, SHAPE_LAND, SHAPE_FAN)
#: Offsets of the shared shape vector theta = [on, to, land, fan_roll, fan_pitch, fan_yaw].
_N_ON, _N_TO, _N_LAND, _N_FAN = SHAPE_ON.n, SHAPE_TO.n, SHAPE_LAND.n, SHAPE_FAN.n
_SL_ON = slice(0, _N_ON)
_SL_TO = slice(_N_ON, _N_ON + _N_TO)
_SL_LAND = slice(_N_ON + _N_TO, _N_ON + _N_TO + _N_LAND)
_FAN0 = _N_ON + _N_TO + _N_LAND
_N_THETA = _FAN0 + 3 * _N_FAN


def _basis(shape: _Shape, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(B (T, K), b0 (T,))`` with ``curve(tau) = B @ values + b0``."""
    knots, fidx, fval = shape.nodes()
    T = tau.shape[0]
    i = np.clip(np.searchsorted(knots, tau, side="right") - 1, 0, len(knots) - 2)
    with np.errstate(invalid="ignore"):
        frac = (tau - knots[i]) / (knots[i + 1] - knots[i])
    frac = np.clip(np.nan_to_num(frac, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    B = np.zeros((T, shape.n))
    b0 = np.zeros(T)
    rows = np.arange(T)
    for node, wgt in ((i, 1.0 - frac), (i + 1, frac)):
        fi = fidx[node]
        free = fi >= 0
        np.add.at(B, (rows[free], fi[free]), wgt[free])
        b0[~free] += wgt[~free] * fval[node[~free]]
    return B, b0


def _smooth_prior(shape: _Shape) -> tuple[np.ndarray, np.ndarray]:
    """``(Lambda, h)``: log prior ``-0.5 a' Lambda a + h' a`` (second differences
    over the extended knot sequence incl. fixed values, plus a weak ridge)."""
    _, fidx, fval = shape.nodes()
    n = len(fidx)
    E = np.zeros((n, shape.n))
    e0 = np.where(fidx < 0, fval, 0.0)
    E[np.flatnonzero(fidx >= 0), fidx[fidx >= 0]] = 1.0
    D2 = np.zeros((max(n - 2, 0), n))
    for r in range(n - 2):
        D2[r, r : r + 3] = (1.0, -2.0, 1.0)
    DE = D2 @ E
    lam = DE.T @ DE / SHAPE_SMOOTH_SD**2 + np.eye(shape.n) / SHAPE_RIDGE_SD**2
    return lam, -(DE.T @ (D2 @ e0)) / SHAPE_SMOOTH_SD**2


def _theta_prior() -> tuple[np.ndarray, np.ndarray]:
    lam = np.zeros((_N_THETA, _N_THETA))
    h = np.zeros(_N_THETA)
    for sh, sl in ((SHAPE_ON, _SL_ON), (SHAPE_TO, _SL_TO), (SHAPE_LAND, _SL_LAND)):
        lam[sl, sl], h[sl] = _smooth_prior(sh)
    lam[_FAN0:, _FAN0:] = np.eye(3 * _N_FAN) / FAN_SD**2
    return lam, h


_THETA_LAM, _THETA_H = _theta_prior()


def _event_times(ph: np.ndarray) -> np.ndarray:
    """Motors-on, take-off and landing sample indices; ``-inf`` for an event
    before the record (it starts in a later code), ``+inf`` for one after it."""
    if np.any(np.diff(ph) < 0):
        raise ValueError("phase codes must be non-decreasing (one motors-on, take-off and landing)")
    start = int(ph[0])
    out = np.empty(3)
    for j, code in enumerate((1, 2, 3)):
        idx = np.flatnonzero(ph == code)
        if start > code or (start == code and code >= 2):
            out[j] = -np.inf  # the record starts after this event
        elif len(idx):
            out[j] = float(idx[0])
        else:
            out[j] = np.inf
    return out


@dataclass(frozen=True)
class _Clock:
    """Event-clock bases of one schedule (no data)."""

    phases: np.ndarray  # (T,) int64
    tau: np.ndarray  # (3, T) seconds from motors-on, take-off, landing
    B_on: np.ndarray
    b_on: np.ndarray
    B_to: np.ndarray
    b_to: np.ndarray
    B_land: np.ndarray
    b_land: np.ndarray
    B_fan: np.ndarray
    support: np.ndarray  # rows where a shape value can enter a factor


def _clock(ph: np.ndarray) -> _Clock:
    ev = _event_times(ph)
    t = np.arange(ph.shape[0], dtype=np.float64)
    tau = (t[None, :] - ev[:, None]) / RATE_HZ
    B_on, b_on = _basis(SHAPE_ON, tau[0])
    B_to, b_to = _basis(SHAPE_TO, tau[1])
    B_land, b_land = _basis(SHAPE_LAND, tau[2])
    B_fan, _ = _basis(SHAPE_FAN, tau[2])
    nz = (B_on != 0).any(1) | (B_to != 0).any(1) | (B_land != 0).any(1) | (B_fan != 0).any(1)
    sup = nz.copy()
    sup[1:] |= nz[:-1]
    sup[2:] |= nz[:-2]
    return _Clock(ph, tau, B_on, b_on, B_to, b_to, B_land, b_land, B_fan, np.flatnonzero(sup))


# ─── link and data ────────────────────────────────────────────────────────────


def _latent(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rev/s -> link units, with ``log dz/dw``; values <= FLOOR_RPS map to the
    floor (disclosed approximation); NaN propagates."""
    wf = np.maximum(np.asarray(w, dtype=np.float64), FLOOR_RPS)
    s = wf / LINK_SCALE
    lj = -np.log(-np.expm1(-s))
    return wf + LINK_SCALE * np.log(-np.expm1(-s)), lj


def _readout(z: np.ndarray) -> np.ndarray:
    """Link units -> physical speed ``s softplus(z / s) > 0``; values at or
    below FLOOR_RPS are reported as 0 (the logs' representation of rest)."""
    w = LINK_SCALE * np.logaddexp(0.0, z / LINK_SCALE)
    return np.where(w > FLOOR_RPS, w, 0.0)


def _check_phases(phases: Any, T: int | None = None) -> np.ndarray:
    ph = np.asarray(phases)
    if ph.ndim != 1 or ph.size == 0 or (T is not None and len(ph) != T):
        raise ValueError(f"phases must be a non-empty ({T},) array, got {ph.shape}")
    if not np.issubdtype(ph.dtype, np.integer) and not np.all(np.equal(np.mod(ph, 1), 0)):
        raise ValueError("phase codes must be integers")
    ph = np.ascontiguousarray(ph, dtype=np.int64)
    if ph.min() < 0 or ph.max() > 3:
        raise ValueError("phase codes must lie in 0..3")
    return ph


@dataclass(frozen=True)
class _Flight:
    x: np.ndarray  # (T, 4) link-unit modes, 0 where non-finite
    obs: np.ndarray  # (T,) bool: sample carries a likelihood factor
    cmd: bool  # command-only source
    jac: float  # log Jacobian rev/s -> modes over the observed factors
    clock: _Clock

    @property
    def phases(self) -> np.ndarray:
        return self.clock.phases


def _prepare(speed: Any, phases: Any, measured: bool = True) -> _Flight:
    w = np.asarray(speed, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != NUM_ROTORS or w.shape[1] == 0:
        raise ValueError(f"speed must be (4, T) with T > 0, got {w.shape}")
    ph = _check_phases(phases, w.shape[1])
    z, lj = _latent(w.T)
    x = z @ MIXER / NUM_ROTORS
    fin = cast(np.ndarray, np.isfinite(x).all(axis=1))
    obs = fin.copy()
    obs[1:] &= fin[:-1]
    obs[2:] &= fin[:-2]
    if not measured:
        obs &= ph >= 2
    jac = float(lj[obs].sum() + LOG_DET_MODES * obs.sum())
    xf = np.where(fin[:, None], x, 0.0)
    return _Flight(xf, obs, not measured, jac, _clock(ph))


# ─── parameters ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProgramParams:
    a_on: np.ndarray  # (K_on,) startup fraction knot values
    a_to: np.ndarray  # (K_to,) take-off fraction knot values
    a_land: np.ndarray  # (K_land,) landing fraction knot values (tail free)
    fan: np.ndarray  # (3, K_fan) landing fan of roll, pitch, yaw
    anchor: np.ndarray  # (2, 4) idle, hover anchor means (link units, modes)
    load: np.ndarray  # (2, 4) idle, hover loadings on their factor
    phi: np.ndarray  # (4 streams, 4 modes, 2)
    sigma: np.ndarray  # (8 SD groups, 4)
    drive: np.ndarray  # (9 drive rows, 4)
    rho: np.ndarray  # (3,) stage profile of the manoeuvre drive, rho[1] = 1
    kappa_act: np.ndarray  # (4,) innovation scale of activity class 1 in cruise
    log_pi_act: np.ndarray  # (2,)
    kappa_src: np.ndarray  # (4,) innovation scale of command-only records

    FIELDS = (
        "a_on",
        "a_to",
        "a_land",
        "fan",
        "anchor",
        "load",
        "phi",
        "sigma",
        "drive",
        "rho",
        "kappa_act",
        "log_pi_act",
        "kappa_src",
    )

    def theta(self) -> np.ndarray:
        return np.concatenate((self.a_on, self.a_to, self.a_land, self.fan.ravel()))

    def with_theta(self, th: np.ndarray) -> ProgramParams:
        return replace(
            self,
            a_on=th[_SL_ON].copy(),
            a_to=th[_SL_TO].copy(),
            a_land=th[_SL_LAND].copy(),
            fan=th[_FAN0:].reshape(3, _N_FAN).copy(),
        )


def _anchors(P: ProgramParams, q: int) -> tuple[np.ndarray, np.ndarray]:
    xi = NODE_XI[q]
    return P.anchor[0] + P.load[0] * xi[0], P.anchor[1] + P.load[1] * xi[1]


def _curves(c: _Clock, P: ProgramParams):
    F_on = c.B_on @ P.a_on + c.b_on
    F_to = c.B_to @ P.a_to + c.b_to
    F_land = c.B_land @ P.a_land + c.b_land
    fan = np.zeros((len(F_on), N_MODES))
    fan[:, 1:] = c.B_fan @ P.fan.T
    return 1.0 - F_on, F_on - F_to + F_land, F_to - F_land, fan


def _mu(cur, P: ProgramParams, q: int) -> np.ndarray:
    cR, cI, cH, fan = cur
    idle, hover = _anchors(P, q)
    return REST * cR[:, None] + idle * cI[:, None] + hover * cH[:, None] + fan


def _lags(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r1 = np.zeros_like(r)
    r2 = np.zeros_like(r)
    r1[1:] = r[:-1]
    r2[2:] = r[:-2]
    return r1, r2


def _filt(z: np.ndarray, p1: Any, p2: Any) -> np.ndarray:
    """``z_t - p1 z_{t-1} - p2 z_{t-2}`` with zero pre-sample values."""
    out = np.array(z, dtype=np.float64, copy=True)
    out[1:] -= p1 * z[:-1]
    out[2:] -= p2 * z[:-2]
    return out


def _stream_residuals(x: np.ndarray, mu: np.ndarray, P: ProgramParams) -> np.ndarray:
    """``(4, T, 4)`` AR-filtered residuals of each stream (no drive)."""
    y = np.empty((len(STREAMS), *x.shape))
    for s in range(len(STREAMS)):
        y[s] = _filt(x - (REST if s == 0 else mu), P.phi[s, :, 0], P.phi[s, :, 1])
    return y


def _drive_table(P: ProgramParams, a: int) -> np.ndarray:
    """``(C, 4)`` drive ``u_k(a)`` of every class."""
    D = np.zeros((N_CLASSES, N_MODES))
    for k in range(N_CLASSES):
        j = CLS_DRIVE[k]
        if j < 0:
            continue
        if CLS_STAGE[k] >= 0:
            D[k] = CLS_SIGN[k] * P.rho[CLS_STAGE[k]] * P.drive[j + a]
        else:
            D[k] = CLS_SIGN[k] * P.drive[j]
    return D


def _act_scale(P: ProgramParams, a: int) -> np.ndarray:
    return np.where(CLS_ACT[:, None] & (a == 1), P.kappa_act[None, :], 1.0)


def _src_scale(P: ProgramParams, cmd: bool) -> np.ndarray:
    return P.kappa_src if cmd else np.ones(N_MODES)


def _sd_table(P: ProgramParams, a: int, cmd: bool) -> np.ndarray:
    """``(C, 4)`` innovation SD of every class."""
    return P.sigma[CLS_SIG] * _act_scale(P, a) * _src_scale(P, cmd)[None, :]


def _log_table(Eb: np.ndarray, D: np.ndarray, S: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """``(T, C)`` class log-emissions; exactly 0 on unobserved samples."""
    e = (Eb - D[:, None, :]) / S[:, None, :]
    lt = (
        -0.5 * np.einsum("ktd,ktd->kt", e, e)
        - (np.log(S).sum(axis=1) + 0.5 * N_MODES * np.log(2.0 * np.pi))[:, None]
    )
    out = np.ascontiguousarray(lt.T)
    out[~obs] = 0.0
    return out


# ─── E-step ───────────────────────────────────────────────────────────────────


@dataclass
class _EStep:
    loglik: float  # sum over records, link-unit modes
    combos: list  # per record: list of (q, a, W (T, C)) with W = resp * occupancy * obs
    stats: ug.Stats
    act_mass: np.ndarray  # (2,)
    node_mass: np.ndarray  # (N_GRID,)
    occupancy: np.ndarray  # (C,)


def _record_logliks(f: _Flight, graph: _Graph, P: ProgramParams) -> np.ndarray:
    """``(N_GRID, 2)`` ``log w_q + log pi_a + log p(x | q, a)``."""
    cur = _curves(f.clock, P)
    lpa = P.log_pi_act - logsumexp(P.log_pi_act)
    D = [_drive_table(P, a) for a in (0, 1)]
    S = [_sd_table(P, a, f.cmd) for a in (0, 1)]
    lw = np.empty((N_GRID, 2))
    for q in range(N_GRID):
        Eb = _stream_residuals(f.x, _mu(cur, P, q), P)[CLS_STREAM]
        for a in (0, 1):
            lw[q, a] = (
                NODE_LOGW[q] + lpa[a] + graph.loglik(_log_table(Eb, D[a], S[a], f.obs), f.phases)
            )
    return lw


def _e_step(flights: list[_Flight], graph: _Graph, P: ProgramParams) -> _EStep:
    total = 0.0
    stats = graph.grammar.zero_stats()
    act_mass, node_mass, occ_tot = np.zeros(2), np.zeros(N_GRID), np.zeros(N_CLASSES)
    combos = []
    for f in flights:
        lw = _record_logliks(f, graph, P)
        lse = float(cast(float, logsumexp(lw)))
        if not np.isfinite(lse):
            raise FloatingPointError("non-finite record likelihood")
        total += lse
        resp = np.exp(lw - lse)
        act_mass += resp.sum(axis=0)
        node_mass += resp.sum(axis=1)
        cur = _curves(f.clock, P)
        kept = []
        for q in np.flatnonzero(resp.max(axis=1) > RESP_FLOOR):
            Eb = _stream_residuals(f.x, _mu(cur, P, int(q)), P)[CLS_STREAM]
            for a in (0, 1):
                r = float(resp[q, a])
                if r <= RESP_FLOOR:
                    continue
                table = _log_table(Eb, _drive_table(P, a), _sd_table(P, a, f.cmd), f.obs)
                _, occ, edges, entries = graph.forward_backward(table, f.phases)
                stats.edges += r * edges
                stats.entries += r * entries
                W = r * occ * f.obs[:, None]
                occ_tot += W.sum(axis=0)
                kept.append((int(q), a, W))
        combos.append(kept)
    return _EStep(total, combos, stats, act_mass, node_mass, occ_tot)


# ─── M-step blocks (each an exact conditional maximiser of Q + log prior) ─────


def _triangle() -> np.ndarray:
    m = STABILITY_MARGIN
    return np.array([(-2.0 + 2.0 * m, -1.0 + m), (2.0 - 2.0 * m, -1.0 + m), (0.0, 1.0 - m)])


def _inside(theta: np.ndarray, vert: np.ndarray) -> bool:
    for i in range(len(vert)):
        a, b = vert[i], vert[(i + 1) % len(vert)]
        if (b[0] - a[0]) * (theta[1] - a[1]) - (b[1] - a[1]) * (theta[0] - a[0]) < -1e-12:
            return False
    return True


def _stable_qp(A: np.ndarray, b: np.ndarray, current: np.ndarray) -> np.ndarray:
    """``argmin t'At - 2b't`` over the AR(2) stability triangle (exact for
    positive-definite ``A``: interior optimum or the best edge point); keeps
    ``current`` without curvature."""
    vert = _triangle()
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b))):
        return current
    if np.linalg.eigvalsh(A)[0] <= 1e-12 * max(1.0, float(np.trace(A))):
        return current
    theta = np.linalg.solve(A, b)
    if _inside(theta, vert):
        return theta

    def f(t: np.ndarray) -> float:
        return float(t @ A @ t - 2.0 * b @ t)

    best, best_f = current, (f(current) if _inside(current, vert) else np.inf)
    for i in range(len(vert)):
        a0, d = vert[i], vert[(i + 1) % len(vert)] - vert[i]
        curv = float(d @ A @ d)
        lam = 0.0 if curv <= 0 else float(np.clip(d @ (b - A @ a0) / curv, 0.0, 1.0))
        cand = a0 + lam * d
        fc = f(cand)
        if fc < best_f:
            best, best_f = cand, fc
    return np.asarray(best, dtype=np.float64)


def _weights(
    W: np.ndarray, prec: np.ndarray, D: np.ndarray, s: int
) -> tuple[np.ndarray, np.ndarray]:
    """``Omega = sum_{k in s} W_k / s_k^2`` and ``Omega_u = sum_k W_k u_k / s_k^2``, ``(T, 4)``."""
    ks = STREAM_CLASSES[s]
    return W[:, ks] @ prec[ks], W[:, ks] @ (prec[ks] * D[ks])


def _per_q(kept: list) -> dict[int, list]:
    out: dict[int, list] = {}
    for q, a, W in kept:
        out.setdefault(q, []).append((a, W))
    return out


def _update_anchors(flights: list[_Flight], P: ProgramParams, combos: list) -> ProgramParams:
    A = np.zeros((N_MODES, 4, 4))
    b = np.zeros((N_MODES, 4))
    for f, kept in zip(flights, combos):
        if not kept:
            continue
        cR, cI, cH, fan = _curves(f.clock, P)
        base = f.x - REST * cR[:, None] - fan
        filt = {}
        for s in (1, 2, 3):
            p1, p2 = P.phi[s, :, 0], P.phi[s, :, 1]
            filt[s] = (
                _filt(np.repeat(cI[:, None], N_MODES, 1), p1, p2),
                _filt(np.repeat(cH[:, None], N_MODES, 1), p1, p2),
                _filt(base, p1, p2),
            )
        for q, a, W in kept:
            prec = _sd_table(P, a, f.cmd) ** -2.0
            D = _drive_table(P, a)
            x1, x2 = NODE_XI[q]
            g = np.array([1.0, x1, 0.0, 0.0])
            h = np.array([0.0, 0.0, 1.0, x2])
            for s in (1, 2, 3):
                LI, LH, Ly = filt[s]
                Om, OmD = _weights(W, prec, D, s)
                z = Om * Ly - OmD
                sII, sIH, sHH = (Om * LI * LI).sum(0), (Om * LI * LH).sum(0), (Om * LH * LH).sum(0)
                tI, tH = (LI * z).sum(0), (LH * z).sum(0)
                for d in range(N_MODES):
                    A[d] += (
                        sII[d] * np.outer(g, g)
                        + sIH[d] * (np.outer(g, h) + np.outer(h, g))
                        + sHH[d] * np.outer(h, h)
                    )
                    b[d] += tI[d] * g + tH[d] * h
    prior = np.diag([ANCHOR_SD**-2, LOAD_SD**-2, ANCHOR_SD**-2, LOAD_SD**-2])
    anchor, load = P.anchor.copy(), P.load.copy()
    for d in range(N_MODES):
        th = np.linalg.solve(A[d] + prior, b[d])
        anchor[0, d], load[0, d], anchor[1, d], load[1, d] = th
    return replace(P, anchor=anchor, load=load)


def _update_shapes(flights: list[_Flight], P: ProgramParams, combos: list) -> ProgramParams:
    G = np.zeros((_N_THETA, _N_THETA))
    rhs = np.zeros(_N_THETA)
    fan_sl = {d: slice(_FAN0 + _N_FAN * (d - 1), _FAN0 + _N_FAN * d) for d in (1, 2, 3)}
    for f, kept in zip(flights, combos):
        rows = f.clock.support
        if not kept or rows.size == 0:
            continue
        c = f.clock
        nr = rows.size
        Mt = np.zeros((3, N_MODES, nr, 4, 4))
        Rt = np.zeros((3, N_MODES, nr, 4))
        xr = f.x - REST
        b_tl = c.b_to - c.b_land
        Y0, Pon, Pto = {}, {}, {}
        for s in (1, 2, 3):
            p1, p2 = P.phi[s, :, 0], P.phi[s, :, 1]
            Y0[s] = _filt(xr, p1, p2)[rows]
            Pon[s] = _filt(np.repeat(c.b_on[:, None], N_MODES, 1), p1, p2)[rows]
            Pto[s] = _filt(np.repeat(b_tl[:, None], N_MODES, 1), p1, p2)[rows]
        for q, a, W in kept:
            prec = _sd_table(P, a, f.cmd) ** -2.0
            D = _drive_table(P, a)
            Wr = W[rows]
            idle, hover = _anchors(P, q)
            a_on, a_to = idle - REST, hover - idle
            sc = np.stack(
                (a_on, a_to, -a_to, (np.arange(N_MODES) > 0).astype(np.float64)), axis=1
            )  # (4 modes, 4 blocks)
            outer = sc[:, :, None] * sc[:, None, :]
            for s in (1, 2, 3):
                Om, OmD = _weights(Wr, prec, D, s)
                yt = Y0[s] - a_on * Pon[s] - a_to * Pto[s]
                z = Om * yt - OmD
                for d in range(N_MODES):
                    Mt[s - 1, d] += Om[:, d, None, None] * outer[d][None]
                    Rt[s - 1, d] += z[:, d, None] * sc[d][None]
        for s in (1, 2, 3):
            p1, p2 = P.phi[s, :, 0], P.phi[s, :, 1]
            for d in range(N_MODES):
                X = [
                    _filt(c.B_on, p1[d], p2[d])[rows],
                    _filt(c.B_to, p1[d], p2[d])[rows],
                    _filt(c.B_land, p1[d], p2[d])[rows],
                ]
                sls = [_SL_ON, _SL_TO, _SL_LAND]
                if d > 0:
                    X.append(_filt(c.B_fan, p1[d], p2[d])[rows])
                    sls.append(fan_sl[d])
                for i1, (X1, s1) in enumerate(zip(X, sls)):
                    rhs[s1] += X1.T @ Rt[s - 1, d, :, i1]
                    for i2, (X2, s2) in enumerate(zip(X, sls)):
                        G[s1, s2] += X1.T @ (Mt[s - 1, d, :, i1, i2, None] * X2)
    th = np.linalg.solve(G + _THETA_LAM, rhs + _THETA_H)
    return P.with_theta(th)


def _update_phi(flights: list[_Flight], P: ProgramParams, combos: list) -> ProgramParams:
    A = np.zeros((len(STREAMS), N_MODES, 2, 2))
    b = np.zeros((len(STREAMS), N_MODES, 2))
    for f, kept in zip(flights, combos):
        if not kept:
            continue
        cur = _curves(f.clock, P)
        r_rest = f.x - REST
        rest_lags = _lags(r_rest)
        for q, items in _per_q(kept).items():
            r_mu = f.x - _mu(cur, P, q)
            mu_lags = _lags(r_mu)
            for a, W in items:
                prec = _sd_table(P, a, f.cmd) ** -2.0
                D = _drive_table(P, a)
                for s in range(len(STREAMS)):
                    r, (r1, r2) = (r_rest, rest_lags) if s == 0 else (r_mu, mu_lags)
                    Om, OmD = _weights(W, prec, D, s)
                    z = Om * r - OmD
                    A[s, :, 0, 0] += (Om * r1 * r1).sum(0)
                    A[s, :, 0, 1] += (Om * r1 * r2).sum(0)
                    A[s, :, 1, 1] += (Om * r2 * r2).sum(0)
                    b[s, :, 0] += (z * r1).sum(0)
                    b[s, :, 1] += (z * r2).sum(0)
    A[..., 1, 0] = A[..., 0, 1]
    phi = P.phi.copy()
    for s in range(len(STREAMS)):
        for d in range(N_MODES):
            phi[s, d] = _stable_qp(A[s, d], b[s, d], P.phi[s, d])
    return replace(P, phi=phi)


def _class_moments(flights: list[_Flight], P: ProgramParams, combos: list):
    """Moments of the drive-free class residual per (activity, source):
    ``M0 (2,2,C)``, ``M1, M2 (2,2,C,4)``."""
    M0 = np.zeros((2, 2, N_CLASSES))
    M1 = np.zeros((2, 2, N_CLASSES, N_MODES))
    M2 = np.zeros((2, 2, N_CLASSES, N_MODES))
    for f, kept in zip(flights, combos):
        if not kept:
            continue
        cur = _curves(f.clock, P)
        cmd = int(f.cmd)
        for q, items in _per_q(kept).items():
            Eb = _stream_residuals(f.x, _mu(cur, P, q), P)[CLS_STREAM]
            for a, W in items:
                M0[a, cmd] += W.sum(axis=0)
                M1[a, cmd] += np.einsum("tk,ktd->kd", W, Eb)
                M2[a, cmd] += np.einsum("tk,ktd->kd", W, Eb * Eb)
    return M0, M1, M2


def _sse(mom, a: int, cmd: int, D: np.ndarray) -> np.ndarray:
    M0, M1, M2 = mom
    return np.maximum(M2[a, cmd] - 2.0 * D * M1[a, cmd] + D * D * M0[a, cmd][:, None], 0.0)


def _update_drives(P: ProgramParams, mom) -> ProgramParams:
    M0, M1, _ = mom
    num = np.zeros_like(P.drive)
    den = np.zeros_like(P.drive)
    for a in (0, 1):
        for cmd in (0, 1):
            prec = _sd_table(P, a, bool(cmd)) ** -2.0
            for k in range(N_CLASSES):
                j = CLS_DRIVE[k]
                if j < 0:
                    continue
                if CLS_STAGE[k] >= 0:
                    j, c = j + a, CLS_SIGN[k] * P.rho[CLS_STAGE[k]]
                else:
                    c = CLS_SIGN[k]
                num[j] += c * prec[k] * M1[a, cmd, k]
                den[j] += c * c * prec[k] * M0[a, cmd, k]
    return replace(P, drive=num / (den + DRIVE_SD**-2))


def _update_rho(P: ProgramParams, mom) -> ProgramParams:
    M0, M1, _ = mom
    rho = P.rho.copy()
    for st in (0, 2):
        num = den = 0.0
        for a in (0, 1):
            for cmd in (0, 1):
                prec = _sd_table(P, a, bool(cmd)) ** -2.0
                for k in np.flatnonzero(st == CLS_STAGE):
                    u = P.drive[CLS_DRIVE[k] + a]
                    num += CLS_SIGN[k] * float(np.sum(prec[k] * u * M1[a, cmd, k]))
                    den += float(np.sum(prec[k] * u * u)) * M0[a, cmd, k]
        rho[st] = (num + RHO_SD**-2) / (den + RHO_SD**-2)
    return replace(P, rho=rho)


def _update_sigma(P: ProgramParams, mom) -> ProgramParams:
    M0 = mom[0]
    S = np.zeros_like(P.sigma)
    n = np.zeros(len(SIGMA_GROUPS))
    for a in (0, 1):
        for cmd in (0, 1):
            scale2 = (_act_scale(P, a) * _src_scale(P, bool(cmd))[None, :]) ** 2
            sse = _sse(mom, a, cmd, _drive_table(P, a)) / scale2
            np.add.at(S, CLS_SIG, sse)
            np.add.at(n, CLS_SIG, M0[a, cmd])
    var = (S + 2.0 * SIGMA_PRIOR_BETA) / (n[:, None] + 2.0 * SIGMA_PRIOR_ALPHA + 2.0)
    return replace(P, sigma=np.sqrt(np.maximum(var, SIGMA_FLOOR**2)))


def _update_kappa_act(P: ProgramParams, mom) -> ProgramParams:
    M0 = mom[0]
    S, n = np.zeros(N_MODES), 0.0
    for cmd in (0, 1):
        base2 = (P.sigma[CLS_SIG] * _src_scale(P, bool(cmd))[None, :]) ** 2
        sse = _sse(mom, 1, cmd, _drive_table(P, 1)) / base2
        S += sse[CLS_ACT].sum(axis=0)
        n += float(M0[1, cmd][CLS_ACT].sum())
    v = (S + 2.0 * SCALE_PRIOR_BETA) / (n + 2.0 * SCALE_PRIOR_ALPHA + 2.0)
    return replace(P, kappa_act=np.sqrt(v))


def _update_kappa_src(P: ProgramParams, mom) -> ProgramParams:
    M0 = mom[0]
    S, n = np.zeros(N_MODES), 0.0
    for a in (0, 1):
        base2 = (P.sigma[CLS_SIG] * _act_scale(P, a)) ** 2
        S += (_sse(mom, a, 1, _drive_table(P, a)) / base2).sum(axis=0)
        n += float(M0[a, 1].sum())
    v = (S + 2.0 * SCALE_PRIOR_BETA) / (n + 2.0 * SCALE_PRIOR_ALPHA + 2.0)
    return replace(P, kappa_src=np.sqrt(v))


def _log_prior(P: ProgramParams) -> float:
    var = P.sigma**2
    lp = float(np.sum(-(SIGMA_PRIOR_ALPHA + 1.0) * np.log(var) - SIGMA_PRIOR_BETA / var))
    for k in (P.kappa_act, P.kappa_src):
        v = k**2
        lp += float(np.sum(-(SCALE_PRIOR_ALPHA + 1.0) * np.log(v) - SCALE_PRIOR_BETA / v))
    lp += (PI_CONCENTRATION - 1.0) * float(np.sum(P.log_pi_act - logsumexp(P.log_pi_act)))
    lp -= 0.5 * float(np.sum(P.drive**2)) / DRIVE_SD**2
    lp -= 0.5 * float((P.rho[0] - 1.0) ** 2 + (P.rho[2] - 1.0) ** 2) / RHO_SD**2
    lp -= (
        0.5 * float(np.sum(P.anchor**2)) / ANCHOR_SD**2
        + 0.5 * float(np.sum(P.load**2)) / LOAD_SD**2
    )
    th = P.theta()
    lp += float(-0.5 * th @ _THETA_LAM @ th + _THETA_H @ th)
    return lp


def _m_step(flights: list[_Flight], grammar: ug.Grammar, P: ProgramParams, est: _EStep):
    grammar = grammar.m_step(est.stats)
    a = est.act_mass + PI_CONCENTRATION - 1.0
    P = replace(P, log_pi_act=np.log(a / a.sum()))
    P = _update_anchors(flights, P, est.combos)
    P = _update_shapes(flights, P, est.combos)
    P = _update_phi(flights, P, est.combos)
    mom = _class_moments(flights, P, est.combos)
    P = _update_drives(P, mom)
    P = _update_rho(P, mom)
    P = _update_sigma(P, mom)
    P = _update_kappa_act(P, mom)
    P = _update_kappa_src(P, mom)
    return grammar, P


# ─── initialisation (training tracks only, deterministic) ─────────────────────


def default_params(idle_rps: float = 30.0, hover_rps: float = 80.0) -> ProgramParams:
    """Data-free generic parameters (fallback of the initialisation; tests)."""

    def k(sh: _Shape) -> np.ndarray:
        return np.array(sh.free)

    z_idle, z_hover = _latent(np.array([idle_rps, hover_rps]))[0]
    anchor = np.zeros((2, N_MODES))
    anchor[:, 0] = (z_idle, z_hover)
    load = np.array([[1.0, 0.3, 0.3, 0.3], [3.0, 0.5, 0.5, 0.5]])
    phi = np.zeros((len(STREAMS), N_MODES, 2))
    phi[0] = (0.5, 0.0)
    phi[1] = (0.9, 0.0)
    phi[2] = (1.2, -0.35)
    phi[3] = (1.2, -0.3)
    sigma = np.tile(np.array([0.05, 2.0, 0.3, 1.5, 0.3, 0.45, 0.6, 1.5])[:, None], (1, N_MODES))
    drive = np.zeros((len(DRIVE_ROWS), N_MODES))
    drive[0, 0] = 0.6
    drive[7] = 0.15
    drive[8] = 0.45
    return ProgramParams(
        a_on=np.clip(k(SHAPE_ON) / 0.4, 0.0, 1.0),
        a_to=np.clip((k(SHAPE_TO) + 0.2) / 2.2, 0.0, 1.0),
        a_land=np.clip((k(SHAPE_LAND) + 0.8) / 0.9, 0.0, 1.0),
        fan=np.zeros((3, _N_FAN)),
        anchor=anchor,
        load=load,
        phi=phi,
        sigma=sigma,
        drive=drive,
        rho=np.ones(3),
        kappa_act=np.full(N_MODES, 2.0),
        log_pi_act=np.log(np.full(2, 0.5)),
        kappa_src=np.ones(N_MODES),
    )


def _ar2_init(segments: list[np.ndarray], default: tuple[float, float]):
    """Per-mode AR(2) LS on demeaned segments, projected into the triangle,
    with the residual SD; ``default`` without data."""
    A = np.zeros((N_MODES, 2, 2))
    b = np.zeros((N_MODES, 2))
    res: list[np.ndarray] = []
    for seg in segments:
        if len(seg) < 8:
            continue
        z = seg - seg.mean(axis=0)
        y, p1, p2 = z[2:], z[1:-1], z[:-2]
        A[:, 0, 0] += (p1 * p1).sum(0)
        A[:, 0, 1] += (p1 * p2).sum(0)
        A[:, 1, 1] += (p2 * p2).sum(0)
        b[:, 0] += (y * p1).sum(0)
        b[:, 1] += (y * p2).sum(0)
    A[:, 1, 0] = A[:, 0, 1]
    phi = np.tile(np.array(default, dtype=np.float64), (N_MODES, 1))
    for d in range(N_MODES):
        if np.linalg.det(A[d]) > 1e-9 * max(1.0, np.trace(A[d])) ** 2:
            phi[d] = _stable_qp(np.eye(2), np.linalg.solve(A[d], b[d]), phi[d])
    for seg in segments:
        if len(seg) < 8:
            continue
        z = seg - seg.mean(axis=0)
        res.append(z[2:] - phi[:, 0] * z[1:-1] - phi[:, 1] * z[:-2])
    sd = np.concatenate(res).std(axis=0) if res else None
    return phi, sd


def _fraction_fit(shape: _Shape, Bs: list, ys: list, b0s: list, default: np.ndarray) -> np.ndarray:
    if not Bs or sum(len(y) for y in ys) < 4:
        return default
    B, y, b0 = np.concatenate(Bs), np.concatenate(ys), np.concatenate(b0s)
    lam, h = _smooth_prior(shape)
    return np.linalg.solve(B.T @ B + lam, B.T @ (y - b0) + h)


def _init_params(flights: list[_Flight]) -> ProgramParams:
    P = default_params()
    meas = [f for f in flights if not f.cmd]
    idle_rows, hover_rows, idle_med, hover_med = [], [], [], []
    for f in flights:
        tau, ph = f.clock.tau, f.phases
        sel_i = f.obs & (ph == 1) & (tau[0] >= 2.0) & (tau[1] <= -1.6)
        sel_h = f.obs & (ph == 2) & (tau[1] >= 6.0) & (tau[2] <= -4.0)
        if not f.cmd and sel_i.sum() >= 5:
            idle_rows.append(f.x[sel_i])
            idle_med.append(np.median(f.x[sel_i], axis=0))
        if sel_h.sum() >= 5:
            hover_rows.append(f.x[sel_h])
            hover_med.append(np.median(f.x[sel_h], axis=0))
    anchor, load = P.anchor.copy(), P.load.copy()
    if hover_rows:
        anchor[1] = np.median(np.concatenate(hover_rows), axis=0)
        if len(hover_med) > 1:
            load[1] = np.maximum(np.std(hover_med, axis=0), load[1] * 0.5)
    if idle_rows:
        anchor[0] = np.median(np.concatenate(idle_rows), axis=0)
        if len(idle_med) > 1:
            load[0] = np.maximum(np.std(idle_med, axis=0), load[0] * 0.5)
    else:
        anchor[0] = REST + 0.35 * (anchor[1] - REST)
    R0, I0, H0 = REST[0], anchor[0, 0], anchor[1, 0]
    P = replace(P, anchor=anchor, load=load)

    # Pooled fraction regressions on the common mode (initial values only).
    a_on = P.a_on
    if I0 - R0 > 1.0:
        Bs, ys, bs = [], [], []
        for f in meas:
            c = f.clock
            sel = f.obs & (f.phases >= 1) & (c.tau[0] <= 6.0) & (c.tau[1] < -1.6)
            Bs.append(c.B_on[sel])
            ys.append((f.x[sel, 0] - R0) / (I0 - R0))
            bs.append(c.b_on[sel])
        a_on = _fraction_fit(SHAPE_ON, Bs, ys, bs, P.a_on)
    a_to, a_land = P.a_to, P.a_land
    if H0 - I0 > 1.0:
        Bs, ys, bs, Bl, yl, bl = [], [], [], [], [], []
        for f in flights:
            c = f.clock
            F_on = c.B_on @ a_on + c.b_on
            sel = f.obs & (c.tau[1] >= -1.6) & (c.tau[1] <= 5.0) & (c.tau[2] < -4.0)
            Bs.append(c.B_to[sel])
            bs.append(c.b_to[sel])
            ys.append((f.x[sel, 0] - R0 - (I0 - R0) * F_on[sel]) / (H0 - I0))
            x0 = f.x[:, 0]
            sel = (
                f.obs
                & (c.tau[2] >= -4.0)
                & (c.tau[2] <= 2.5)
                & ((c.tau[2] <= 0.0) | (x0 > R0 + 0.5 * (I0 - R0)))
            )
            Bl.append(c.B_land[sel])
            bl.append(c.b_land[sel])
            yl.append((H0 - x0[sel]) / (H0 - I0))
        a_to = _fraction_fit(SHAPE_TO, Bs, ys, bs, P.a_to)
        a_land = _fraction_fit(SHAPE_LAND, Bl, yl, bl, P.a_land)
    P = replace(P, a_on=a_on, a_to=a_to, a_land=a_land)

    # AR coefficients and noise scales from demeaned standby / cruise segments.
    stand = [
        f.x[f.obs & (f.phases == 1) & (f.clock.tau[0] >= 2.0) & (f.clock.tau[1] <= -1.6)]
        for f in meas
    ]
    cruise = [
        f.x[f.obs & (f.phases == 2) & (f.clock.tau[1] >= 6.0) & (f.clock.tau[2] <= -4.0)]
        for f in flights
    ]
    phi = P.phi.copy()
    phi[1], sd_s = _ar2_init(stand, tuple(P.phi[1, 0]))
    phi[3], sd_c = _ar2_init(cruise, tuple(P.phi[3, 0]))
    sigma = P.sigma.copy()
    sd_c = np.maximum(sd_c, 0.05) if sd_c is not None else sigma[4]
    sd_s = np.maximum(sd_s, SIGMA_FLOOR) if sd_s is not None else sigma[2]
    sigma[2] = 1.5 * sd_s
    sigma[3] = 3.0 * sd_c
    sigma[4] = sd_c
    sigma[5] = 1.5 * sd_c
    sigma[6] = 2.0 * sd_c
    sigma[7] = 3.0 * sd_c
    drive = P.drive.copy()
    drive[0] = np.array([2.0 * sd_s[0], 0.0, 0.0, 0.0])
    drive[7] = 0.5 * sd_c
    drive[8] = 1.5 * sd_c
    return replace(P, phi=phi, sigma=np.maximum(sigma, SIGMA_FLOOR), drive=drive)


# ─── sampling kernel ──────────────────────────────────────────────────────────


@njit(cache=True)
def _simulate(cls, node, act, mus, rest, is_rest, phi, D, S, eps, out):  # pragma: no cover - numba
    """Exactly the likelihood recursion: class ``cls[i, t]``, anchor node
    ``node[i]``, activity ``act[i]``; writes link-unit modes into ``out``."""
    n, T = cls.shape
    M = out.shape[2]
    for i in range(n):
        q = node[i]
        a = act[i]
        for t in range(T):
            k = cls[i, t]
            for d in range(M):
                if is_rest[k]:
                    ref0 = rest[d]
                    ref1 = rest[d]
                    ref2 = rest[d]
                else:
                    ref0 = mus[q, t, d]
                    ref1 = mus[q, t - 1, d] if t >= 1 else 0.0
                    ref2 = mus[q, t - 2, d] if t >= 2 else 0.0
                r1 = out[i, t - 1, d] - ref1 if t >= 1 else 0.0
                r2 = out[i, t - 2, d] - ref2 if t >= 2 else 0.0
                out[i, t, d] = (
                    ref0
                    + phi[k, d, 0] * r1
                    + phi[k, d, 1] * r2
                    + D[a, k, d]
                    + S[a, k, d] * eps[i, t, d]
                )


@dataclass(frozen=True)
class Draws:
    """Generated records: physical readout, latent link-unit modes and choices."""

    speed: np.ndarray  # (n, 4, T) rev/s, >= 0 (0 where the readout is at/below FLOOR_RPS)
    latent: np.ndarray  # (n, T, 4) link-unit mixer modes of the fitted law
    node: np.ndarray  # (n,) anchor node
    activity: np.ndarray  # (n,) activity class
    classes: np.ndarray  # (n, T) emission classes

    @property
    def floor_fraction(self) -> float:
        """Share of (draw, sample) with any rotor read out as 0."""
        return float(np.mean((self.speed <= 0.0).any(axis=1))) if self.speed.size else 0.0


# ─── model ────────────────────────────────────────────────────────────────────


def _arr(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


class ProgramResponseModel:
    """Fitted program-response generator; see the module docstring."""

    def __init__(
        self,
        grammar: ug.Grammar,
        params: ProgramParams,
        *,
        history: list[float] | None = None,
        fit_info: dict[str, Any] | None = None,
    ) -> None:
        self.grammar = grammar
        self.params = params
        self.graph = _Graph(grammar)
        self.history = [float(h) for h in (history or [])]
        self.fit_info = dict(fit_info or {})
        self.last_clip_fraction = float("nan")

    def n_params_breakdown(self) -> dict[str, int]:
        P = self.params
        return {
            "shape_startup": P.a_on.size,
            "shape_takeoff": P.a_to.size,
            "shape_landing": P.a_land.size,
            "landing_fan": P.fan.size,
            "anchor_law": P.anchor.size + P.load.size,
            "ar_coefficients": P.phi.size,
            "innovation_sd": P.sigma.size,
            "action_drives": P.drive.size,
            "drive_stage_profile": 2,
            "activity_scale": P.kappa_act.size,
            "activity_weights": P.log_pi_act.size - 1,
            "command_source_scale": P.kappa_src.size,
            "grammar": int(self.grammar.n_params),
        }

    @property
    def n_params(self) -> int:
        return int(sum(self.n_params_breakdown().values()))

    def loglik(self, speed: np.ndarray, phases: np.ndarray, *, measured: bool = True) -> float:
        """Log density (rev/s units) of one record; nodes, activity and action
        paths summed out."""
        f = _prepare(speed, phases, measured)
        return float(cast(float, logsumexp(_record_logliks(f, self.graph, self.params)))) + f.jac

    def draw(
        self, phases: np.ndarray, n: int, rng: np.random.Generator, *, measured: bool = True
    ) -> Draws:
        ph = _check_phases(phases)
        n = int(n)
        if n < 1:
            raise ValueError("n must be >= 1")
        P = self.params
        clock = _clock(ph)
        cur = _curves(clock, P)
        mus = np.stack([_mu(cur, P, q) for q in range(N_GRID)])
        node = (
            rng.choice(N_NODES, size=n, p=GH_WEIGHTS) * N_NODES
            + rng.choice(N_NODES, size=n, p=GH_WEIGHTS)
        ).astype(np.int64)
        pi = np.exp(P.log_pi_act - logsumexp(P.log_pi_act))
        act = rng.choice(2, size=n, p=pi / pi.sum()).astype(np.int64)
        cls = self.graph.sample(ph, rng, n)
        eps = rng.standard_normal((n, ph.shape[0], N_MODES))
        D = np.stack([_drive_table(P, a) for a in (0, 1)])
        S = np.stack([_sd_table(P, a, not measured) for a in (0, 1)])
        x = np.empty((n, ph.shape[0], N_MODES))
        _simulate(
            cls,
            node,
            act,
            mus,
            REST.astype(np.float64),
            CLS_REST,
            np.ascontiguousarray(P.phi[CLS_STREAM]),
            D,
            S,
            eps,
            x,
        )
        speed = _readout(x @ MIXER.T).transpose(0, 2, 1)
        out = Draws(
            speed=np.ascontiguousarray(speed), latent=x, node=node, activity=act, classes=cls
        )
        self.last_clip_fraction = out.floor_fraction
        return out

    def sample(
        self, phases: np.ndarray, n: int, rng: np.random.Generator, *, measured: bool = True
    ) -> np.ndarray:
        """``(n, 4, T)`` nonnegative finite rotor speeds (rev/s, MIXER rotor
        order) on the coarse schedule; ``last_clip_fraction`` is the share of
        sample times with a rotor read out at 0 (at/below FLOOR_RPS)."""
        return self.draw(phases, n, rng, measured=measured).speed

    def to_dict(self) -> dict[str, Any]:
        P = self.params
        d = {
            "model": "program_response",
            "classes": list(CLASSES),
            "shapes": {
                s.name: {"free": list(s.free), "fixed": [list(v) for v in s.fixed]} for s in SHAPES
            },
            "params": {k: getattr(P, k).tolist() for k in ProgramParams.FIELDS},
            "grammar": self.grammar.to_dict(),
            "n_params": self.n_params,
            "n_params_breakdown": self.n_params_breakdown(),
            "constants": {
                "rate_hz": RATE_HZ,
                "floor_rps": FLOOR_RPS,
                "link_scale": LINK_SCALE,
                "sigma_floor": SIGMA_FLOOR,
                "n_nodes": N_NODES,
                "resp_floor": RESP_FLOOR,
                "stability_margin": STABILITY_MARGIN,
            },
            "history": list(self.history),
            "fit": self.fit_info,
        }
        json.dumps(d)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProgramResponseModel:
        if d.get("model") != "program_response" or list(d["classes"]) != list(CLASSES):
            raise ValueError("not a program_response model of this class vocabulary")
        for s in SHAPES:
            saved = d["shapes"][s.name]
            if tuple(saved["free"]) != s.free or tuple(tuple(v) for v in saved["fixed"]) != s.fixed:
                raise ValueError(f"saved knots of {s.name} differ from this module")
        if (
            int(d["constants"]["n_nodes"]) != N_NODES
            or float(d["constants"]["floor_rps"]) != FLOOR_RPS
        ):
            raise ValueError("saved anchor grid or link floor differs from this module")
        P = ProgramParams(**{k: _arr(v) for k, v in d["params"].items()})
        return cls(
            ug.Grammar.from_dict(d["grammar"]), P, history=d.get("history"), fit_info=d.get("fit")
        )


# ─── fitting ──────────────────────────────────────────────────────────────────


def fit(
    tracks: list[tuple[np.ndarray, np.ndarray]],
    *,
    measured: list[bool] | None = None,
    max_iter: int = 80,
    tol: float = 1e-6,
) -> ProgramResponseModel:
    """Fit on whole training records ``(speed (4, T) rev/s, phase codes (T,))``.

    ``measured[i]`` False marks a command-only record (source model in the
    module docstring).  Generalized EM on the penalized observed likelihood;
    stops when the gain per observed factor is below ``tol``, after
    ``max_iter`` M-steps, or (reverting) on a decrease."""
    if not tracks:
        raise ValueError("no training tracks")
    if max_iter < 0:
        raise ValueError("max_iter must be >= 0")
    meas = [True] * len(tracks) if measured is None else [bool(m) for m in measured]
    if len(meas) != len(tracks):
        raise ValueError("measured must have one flag per track")
    flights = [_prepare(w, ph, m) for (w, ph), m in zip(tracks, meas)]
    n_obs = int(sum(f.obs.sum() for f in flights))
    if n_obs == 0:
        raise ValueError("no observed samples in the training tracks")
    jac = float(sum(f.jac for f in flights))

    grammar = default_grammar()
    P = _init_params(flights)
    history: list[float] = []
    trace: list[dict[str, float]] = []
    prev = None
    est = None
    stop = "max_iter"
    for it in range(max_iter + 1):
        graph = _Graph(grammar)
        est = _e_step(flights, graph, P)
        obj = est.loglik + jac + _log_prior(P) + float(grammar.log_prior())
        if not np.isfinite(obj):
            raise FloatingPointError(f"non-finite objective at iteration {it}")
        if history and obj < history[-1] - MONOTONE_RTOL * max(1.0, abs(history[-1])):
            grammar, P, est = prev  # type: ignore[misc]
            stop = "non_monotone"
            break
        history.append(float(obj))
        trace.append(
            {
                "objective": float(obj),
                "data_loglik": float(est.loglik + jac),
                "per_factor": float((est.loglik + jac) / n_obs),
            }
        )
        if len(history) > 1 and history[-1] - history[-2] < tol * n_obs:
            stop = "converged"
            break
        if it == max_iter:
            break
        prev = (grammar, P, est)
        grammar, P = _m_step(flights, grammar, P, est)

    assert est is not None
    radius = [
        [
            float(np.max(np.abs(np.roots([1.0, -P.phi[s, d, 0], -P.phi[s, d, 1]]))))
            for d in range(N_MODES)
        ]
        for s in range(len(STREAMS))
    ]
    fit_info = {
        "n_records": len(flights),
        "n_measured": int(sum(meas)),
        "n_command_only": int(len(meas) - sum(meas)),
        "n_samples": int(sum(len(f.phases) for f in flights)),
        "n_factors": n_obs,
        "n_iter": len(history) - 1,
        "max_iter": int(max_iter),
        "tol_per_factor": float(tol),
        "converged": stop == "converged",
        "stop_reason": stop,
        "trace": trace,
        "class_occupancy": dict(zip(CLASSES, est.occupancy.tolist())),
        "activity_mass": est.act_mass.tolist(),
        "node_mass": est.node_mass.tolist(),
        "sigma_at_floor": int(np.sum(P.sigma <= SIGMA_FLOOR * (1 + 1e-9))),
        "ar_spectral_radius": dict(zip(STREAMS, radius)),
        "mean_dwell_s": dict(
            zip(
                [m.name for m in grammar.macros],
                [float(v) if np.isfinite(v) else None for v in grammar.mean_dwell_s()],
            )
        ),
    }
    return ProgramResponseModel(grammar, P, history=history, fit_info=fit_info)


__all__ = [
    "CLASSES",
    "Draws",
    "ProgramParams",
    "ProgramResponseModel",
    "default_grammar",
    "default_params",
    "fit",
]
