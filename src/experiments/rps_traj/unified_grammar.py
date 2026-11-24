"""The action grammar of the unified whole-flight motor-speed model.

A flight is told only its coarse commanded schedule, one phase code per 25 Hz
sample (:data:`PHASES`): 0 before motors-on, 1 motors-on to take-off (start-up
AND standby), 2 take-off to landing (take-off AND cruise), 3 landing to the
end (spin-down and rest).  Inside each phase a latent semi-Markov chain of
ACTIONS decides what the rotors are driven to; the response model
(:mod:`experiments.rps_traj.unified_response`) turns an action into a target
for its autoregression and owns everything continuous.  This module owns the
discrete part: the graph, its exact likelihood/posterior and its sampler.

**Emission labels.**  :data:`LABELS` is the transferable action vocabulary:
13 labels, each a target relative to the flight's zero, idle or hover anchor
(:class:`LabelSpec`).  The graph has more states than labels; every state
emits exactly one label, and states that mean the same rotor command share it
(the signed cruise drives and the hold are ONE emission each, whichever
parent context visits them).  The response model only ever sees labels.

**Graph.**  A graph is a list of macro states (:class:`MacroSpec`), each
belonging to one phase and emitting one label.  A macro state's dwell is
negative binomial: ``stages`` clock stages in series sharing one per-stage
stay probability (an Erlang-like dwell of at least ``stages`` samples with
mean ``stages / (1 - stay)``), not a geometric one and not a template clock.
On leaving its last stage it branches to one of its ``successors``; a macro
state without successors is absorbing (one stage, stays forever).  At the
first sample and at EVERY change of the phase code the chain is reset to the
new phase's entry law; nothing carries across a phase change in the graph
(the response model's state does carry).  There is no terminal constraint:
a phase may be cut anywhere.

* :func:`build_grammar` ``(hierarchical=True)`` — the ordered grammar.
  Phase 1: pre-drive -> common pulse -> release, which can pulse again
  or settle into idle until take-off. Phase 2: rise -> (overshoot) -> one of three
  persistent PARENT contexts — quiet hold, isolated events (hold -> one signed
  drive -> back to hold) and alternating drive (+ <-> - with brief holds);
  the parent persists over many child actions and only changes from a hold,
  so it changes what follows a drive.  Phase 3: descend -> (shelf/fan) ->
  stop -> rest.
* :func:`build_grammar` ``(hierarchical=False)`` — the flat comparator: the
  same labels, the same stage count per label and the same phases 0, 1, 3,
  but phase 2 has one macro state per label, so what follows an action
  depends on that action only, never on a parent.

**Inference.**  :meth:`Grammar.forward_backward` is the exact forward-backward
over the expanded (stage-level) chain on a sparse CSR transition list
(numba), with every message kept in the log domain (per-destination
log-sum-exp), so extreme but finite log-emissions of long low-noise tracks
never underflow a reachable state nor overflow an unreachable one.  Given
per-sample log-emissions of every label it
returns the log-likelihood, the per-sample label posteriors and the expected
transition and entry counts, from which :meth:`Grammar.m_step` refits every
stay probability, branch law and entry law in closed form (MAP under the
Beta/Dirichlet pseudo-counts of :meth:`Grammar.log_prior`, so EM on
``loglik + log_prior`` is monotone).  :meth:`Grammar.sample` draws label
paths from the same graph with one uniform per sample.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple, cast

import numpy as np
from numba import njit

#: Sample rate (Hz) the dwell defaults are expressed at.
RATE_HZ = 25.0

#: Coarse commanded phase codes.
PHASES = ("off", "startup", "airborne", "landing")
N_PHASES = len(PHASES)

#: Pseudo-counts of the MAP updates: Beta(1+a, 1+a) on every stay probability
#: and Dirichlet(1+a) on every branch law (``PSEUDO``), Dirichlet(1+a) on the
#: phase entry laws (``ENTRY_PSEUDO``, smaller: a flight enters each phase
#: once).  They keep every count finite and every probability inside (0, 1).
PSEUDO = 0.5
ENTRY_PSEUDO = 0.1


@dataclass(frozen=True)
class LabelSpec:
    """One emission label: the phases it may occur in, the anchor its target
    is relative to (``"zero"``, ``"idle"`` or ``"hover"``), whether the
    response model learns a free offset from that anchor (exactly one fixed
    label per anchor, for identifiability) and its drive sign."""

    name: str
    phases: tuple[int, ...]
    ref: str
    free_offset: bool
    sign: int = 0


LABELS: tuple[LabelSpec, ...] = (
    LabelSpec("rest", (0, 3), "zero", False),
    LabelSpec("predrive", (1,), "idle", True),
    LabelSpec("pulse", (1,), "idle", True),
    LabelSpec("release", (1,), "idle", True),
    LabelSpec("idle", (1,), "idle", False),
    LabelSpec("rise", (2,), "hover", True),
    LabelSpec("overshoot", (2,), "hover", True),
    LabelSpec("hold", (2,), "hover", False),
    LabelSpec("drive_up", (2,), "hover", True, +1),
    LabelSpec("drive_down", (2,), "hover", True, -1),
    LabelSpec("descend", (3,), "hover", True),
    LabelSpec("shelf", (3,), "idle", True),
    LabelSpec("stop", (3,), "zero", False),
)
N_LABELS = len(LABELS)
LABEL_ID: dict[str, int] = {s.name: i for i, s in enumerate(LABELS)}
PHASE_LABELS: tuple[tuple[int, ...], ...] = tuple(
    tuple(i for i, s in enumerate(LABELS) if p in s.phases) for p in range(N_PHASES)
)

#: Clock stages per label: the shape of every dwell emitting that label, the
#: same in the ordered and the flat grammar.  Absorbing states have one.
STAGES: dict[str, int] = {
    "rest": 1,
    "predrive": 2,
    "pulse": 3,
    "release": 3,
    "idle": 1,
    "rise": 3,
    "overshoot": 2,
    "hold": 2,
    "drive_up": 3,
    "drive_down": 3,
    "descend": 3,
    "shelf": 3,
    "stop": 3,
}

# Edge kinds of the expanded chain.
_STAY, _ADVANCE, _EXIT, _FIXED = 0, 1, 2, 3


@dataclass(frozen=True)
class MacroSpec:
    """A macro state: ``stages`` clock stages emitting ``label`` in ``phase``,
    branching to ``successors`` (macro names of the same phase) on exit;
    absorbing when it has none."""

    name: str
    phase: int
    label: int
    stages: int
    successors: tuple[str, ...] = ()

    @property
    def absorbing(self) -> bool:
        return not self.successors


class Compiled(NamedTuple):
    """The expanded chain as flat arrays (CSR by source state)."""

    state_macro: np.ndarray  # (S,) int64
    state_label: np.ndarray  # (S,) int64
    state_phase: np.ndarray  # (S,) int64
    first_state: np.ndarray  # (M,) int64, stage 0 of each macro state
    row_ptr: np.ndarray  # (S+1,) int64
    col: np.ndarray  # (E,) int64 destination state
    prob: np.ndarray  # (E,) float64
    log_prob: np.ndarray  # (E,) float64, -inf for zero-probability edges
    kind: np.ndarray  # (E,) int64, _STAY/_ADVANCE/_EXIT/_FIXED
    edge_macro: np.ndarray  # (E,) int64, macro state of the source
    edge_succ: np.ndarray  # (E,) int64, successor index for _EXIT, else -1
    init: np.ndarray  # (N_PHASES, S) entry law per phase
    log_init: np.ndarray  # (N_PHASES, S)
    mask: np.ndarray  # (N_PHASES, N_LABELS) bool, labels used in each phase


@dataclass(frozen=True)
class FB:
    """One forward-backward pass: ``loglik``, label posteriors ``occupancy``
    ``(T, N_LABELS)`` (rows sum to one) and the expected counts of every
    expanded-chain edge ``edges (E,)`` and of every state entered at a phase
    reset ``entries (S,)``."""

    loglik: float
    occupancy: np.ndarray
    edges: np.ndarray
    entries: np.ndarray


@dataclass
class Stats:
    """Weighted expected counts accumulated over flights (and anchor
    components) for :meth:`Grammar.m_step`."""

    edges: np.ndarray
    entries: np.ndarray

    def add(self, fb: FB, weight: float = 1.0) -> None:
        if fb.edges.shape != self.edges.shape or fb.entries.shape != self.entries.shape:
            raise ValueError("counts of a different graph")
        w = float(weight)
        if not np.isfinite(w) or w < 0.0:
            raise ValueError(f"weight must be finite and >= 0, got {weight}")
        self.edges += w * fb.edges
        self.entries += w * fb.entries


# ─── kernels ──────────────────────────────────────────────────────────────────


def _log0(x: np.ndarray) -> np.ndarray:
    """``log x`` with ``log 0 = -inf`` and no warning."""
    with np.errstate(divide="ignore"):
        return np.log(x)


@njit(cache=True, nogil=True)
def _forward(
    log_e, phases, lab, row_ptr, col, log_prob, log_init, mask, la_out, norm_out, store
):  # pragma: no cover - jitted
    """Log-domain forward pass.  Returns the log-likelihood (``-inf`` when the
    observations are impossible); with ``store`` fills the log filtered
    posteriors ``la_out (T, S)`` and the per-sample log normalisers
    ``norm_out (T,)`` (``log p(x_t | x_<t)``)."""
    n_s = lab.shape[0]
    n_t = log_e.shape[0]
    la = np.full(n_s, -np.inf)
    lp = np.empty(n_s)
    acc = np.empty(n_s)
    lw = np.empty(n_s)
    ll = 0.0
    for t in range(n_t):
        p = phases[t]
        if t == 0 or phases[t - 1] != p:
            for s in range(n_s):
                lp[s] = log_init[p, s]
        else:
            for s in range(n_s):
                lp[s] = -np.inf
                acc[s] = 0.0
            for i in range(n_s):
                a = la[i]
                if a == -np.inf:
                    continue
                for e in range(row_ptr[i], row_ptr[i + 1]):
                    v = a + log_prob[e]
                    if v > lp[col[e]]:
                        lp[col[e]] = v
            for i in range(n_s):
                a = la[i]
                if a == -np.inf:
                    continue
                for e in range(row_ptr[i], row_ptr[i + 1]):
                    j = col[e]
                    v = a + log_prob[e]
                    if v > -np.inf:
                        acc[j] += np.exp(v - lp[j])
            for s in range(n_s):
                if acc[s] > 0.0:
                    lp[s] += np.log(acc[s])
        m = -np.inf
        for s in range(n_s):
            label = lab[s]
            v = lp[s] + log_e[t, label] if (mask[p, label] and lp[s] > -np.inf) else -np.inf
            lw[s] = v
            if v > m:
                m = v
        if m == -np.inf:
            return -np.inf
        c = 0.0
        for s in range(n_s):
            if lw[s] > -np.inf:
                c += np.exp(lw[s] - m)
        z = m + np.log(c)
        for s in range(n_s):
            la[s] = lw[s] - z
        ll += z
        if store:
            for s in range(n_s):
                la_out[t, s] = la[s]
            norm_out[t] = z
    return ll


@njit(cache=True, nogil=True)
def _backward(
    log_e, phases, lab, row_ptr, col, log_prob, mask, la, norm, occ, edge_cnt, entry_cnt
):  # pragma: no cover - jitted
    """Log-domain backward pass accumulating label posteriors ``occ (T, L)``,
    expected edge counts and expected phase-entry counts in place.  Every
    accumulated term is ``exp`` of a log posterior (<= 0 up to rounding), so
    an unreachable state (``la = -inf``) contributes exactly zero whatever
    its backward message."""
    n_s = lab.shape[0]
    n_t = log_e.shape[0]
    lb = np.zeros(n_s)
    nb = np.empty(n_s)
    w = np.empty(n_s)
    for t in range(n_t - 1, -1, -1):
        for s in range(n_s):
            a = la[t, s]
            if a > -np.inf and lb[s] > -np.inf:
                occ[t, lab[s]] += np.exp(a + lb[s])
        p = phases[t]
        if t == 0 or phases[t - 1] != p:
            for s in range(n_s):
                a = la[t, s]
                if a > -np.inf and lb[s] > -np.inf:
                    entry_cnt[s] += np.exp(a + lb[s])
                lb[s] = 0.0
            continue
        z = norm[t]
        for j in range(n_s):
            label = lab[j]
            w[j] = log_e[t, label] + lb[j] - z if (mask[p, label] and lb[j] > -np.inf) else -np.inf
        for i in range(n_s):
            mx = -np.inf
            for e in range(row_ptr[i], row_ptr[i + 1]):
                v = log_prob[e] + w[col[e]]
                if v > mx:
                    mx = v
            if mx == -np.inf:
                nb[i] = -np.inf
                continue
            a = la[t - 1, i]
            acc = 0.0
            for e in range(row_ptr[i], row_ptr[i + 1]):
                v = log_prob[e] + w[col[e]]
                if v > -np.inf:
                    acc += np.exp(v - mx)
                    if a > -np.inf:
                        edge_cnt[e] += np.exp(a + v)
            nb[i] = mx + np.log(acc)
        for s in range(n_s):
            lb[s] = nb[s]


@njit(cache=True, nogil=True)
def _sample(phases, u, lab, row_ptr, col, prob, init, out):  # pragma: no cover - jitted
    """Label paths ``out (n, T)`` from the uniforms ``u (n, T)``."""
    n, n_t = u.shape
    n_s = lab.shape[0]
    for k in range(n):
        s = -1
        for t in range(n_t):
            p = phases[t]
            x = u[k, t]
            if t == 0 or phases[t - 1] != p:
                acc = 0.0
                last = -1
                nxt = -1
                for j in range(n_s):
                    q = init[p, j]
                    if q > 0.0:
                        last = j
                        acc += q
                        if x < acc:
                            nxt = j
                            break
                s = nxt if nxt >= 0 else last
            else:
                acc = 0.0
                last = -1
                nxt = -1
                for e in range(row_ptr[s], row_ptr[s + 1]):
                    q = prob[e]
                    if q > 0.0:
                        last = col[e]
                        acc += q
                        if x < acc:
                            nxt = col[e]
                            break
                s = nxt if nxt >= 0 else last
            out[k, t] = lab[s]


# ─── the grammar ──────────────────────────────────────────────────────────────


def _normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / v.sum()


@dataclass(frozen=True, eq=False)
class Grammar:
    """A semi-Markov action graph with its parameters.

    ``stay[m]`` is the per-stage stay probability of macro state ``m`` (1 for
    absorbing ones), ``branch[m]`` its law over ``macros[m].successors``,
    ``entry[m]`` the probability that a reset into ``macros[m].phase`` enters
    ``m`` (summing to one within each phase)."""

    macros: tuple[MacroSpec, ...]
    stay: np.ndarray
    branch: tuple[np.ndarray, ...]
    entry: np.ndarray
    hierarchical: bool = True
    rate_hz: float = RATE_HZ
    compiled: Compiled = field(init=False, repr=False)

    def __post_init__(self) -> None:
        macros = tuple(self.macros)
        index = {m.name: i for i, m in enumerate(macros)}
        if len(index) != len(macros):
            raise ValueError("macro state names must be unique")
        stay = np.asarray(self.stay, dtype=np.float64).copy()
        entry = np.asarray(self.entry, dtype=np.float64).copy()
        branch = tuple(np.asarray(b, dtype=np.float64).copy() for b in self.branch)
        if (
            stay.shape != (len(macros),)
            or entry.shape != (len(macros),)
            or len(branch) != len(macros)
        ):
            raise ValueError("one stay, branch and entry per macro state")
        for i, m in enumerate(macros):
            if not 0 <= m.phase < N_PHASES or not 0 <= m.label < N_LABELS:
                raise ValueError(f"{m.name}: phase/label out of range")
            if m.phase not in LABELS[m.label].phases:
                raise ValueError(
                    f"{m.name}: label {LABELS[m.label].name} not allowed in phase {m.phase}"
                )
            if m.stages < 1:
                raise ValueError(f"{m.name}: needs at least one stage")
            if len(set(m.successors)) != len(m.successors):
                raise ValueError(f"{m.name}: repeated successor")
            for s in m.successors:
                if s not in index or macros[index[s]].phase != m.phase or s == m.name:
                    raise ValueError(f"{m.name}: bad successor {s!r}")
            if m.absorbing:
                if m.stages != 1:
                    raise ValueError(f"{m.name}: an absorbing state has one stage")
                stay[i] = 1.0
                branch_i = np.zeros(0)
            else:
                if not 0.0 <= stay[i] < 1.0:
                    raise ValueError(f"{m.name}: stay must lie in [0, 1)")
                b = branch[i]
                if b.shape != (len(m.successors),) or (b < 0).any() or not np.isclose(b.sum(), 1.0):
                    raise ValueError(f"{m.name}: branch must be a law over its successors")
                branch_i = b / b.sum()
            branch = branch[:i] + (branch_i,) + branch[i + 1 :]
        if (entry < 0).any():
            raise ValueError("negative entry probability")
        for p in range(N_PHASES):
            sel = np.array([m.phase == p for m in macros])
            if sel.any():
                if not np.isclose(entry[sel].sum(), 1.0):
                    raise ValueError(f"entry law of phase {p} must sum to one")
                entry[sel] /= entry[sel].sum()
        for a in (stay, entry, *branch):
            a.setflags(write=False)
        object.__setattr__(self, "macros", macros)
        object.__setattr__(self, "stay", stay)
        object.__setattr__(self, "entry", entry)
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "compiled", self._compile(index))

    # ── structure ──

    def _compile(self, index: Mapping[str, int]) -> Compiled:
        macros = self.macros
        first = np.cumsum([0] + [m.stages for m in macros])[:-1].astype(np.int64)
        n_s = int(sum(m.stages for m in macros))
        state_macro = np.repeat(np.arange(len(macros)), [m.stages for m in macros]).astype(np.int64)
        state_label = np.array([macros[i].label for i in state_macro], dtype=np.int64)
        state_phase = np.array([macros[i].phase for i in state_macro], dtype=np.int64)
        # (source, destination, probability, kind, macro, successor index)
        rows: list[tuple[int, int, float, int, int, int]] = []
        for i, m in enumerate(macros):
            r = float(self.stay[i])
            for st in range(m.stages):
                s = int(first[i]) + st
                if m.absorbing:
                    rows.append((s, s, 1.0, _FIXED, i, -1))
                    continue
                rows.append((s, s, r, _STAY, i, -1))
                if st < m.stages - 1:
                    rows.append((s, s + 1, 1.0 - r, _ADVANCE, i, -1))
                else:
                    for j, name in enumerate(m.successors):
                        d = int(first[index[name]])
                        rows.append((s, d, (1.0 - r) * float(self.branch[i][j]), _EXIT, i, j))
        src, dst, prob, kind, emac, esucc = zip(*rows)
        row_ptr = np.zeros(n_s + 1, dtype=np.int64)
        np.add.at(row_ptr, np.asarray(src, dtype=np.int64) + 1, 1)
        row_ptr = np.cumsum(row_ptr)
        init = np.zeros((N_PHASES, n_s))
        mask = np.zeros((N_PHASES, N_LABELS), dtype=np.bool_)
        for i, m in enumerate(macros):
            init[m.phase, first[i]] = self.entry[i]
            mask[m.phase, m.label] = True
        arrays = Compiled(
            state_macro=state_macro,
            state_label=state_label,
            state_phase=state_phase,
            first_state=first,
            row_ptr=row_ptr,
            col=np.asarray(dst, dtype=np.int64),
            prob=np.asarray(prob, dtype=np.float64),
            log_prob=_log0(np.asarray(prob, dtype=np.float64)),
            kind=np.asarray(kind, dtype=np.int64),
            edge_macro=np.asarray(emac, dtype=np.int64),
            edge_succ=np.asarray(esucc, dtype=np.int64),
            init=init,
            log_init=_log0(init),
            mask=mask,
        )
        for a in arrays:
            a.setflags(write=False)
        return arrays

    @property
    def n_states(self) -> int:
        return int(self.compiled.state_macro.shape[0])

    @property
    def macro_index(self) -> dict[str, int]:
        return {m.name: i for i, m in enumerate(self.macros)}

    @property
    def n_params(self) -> int:
        """Free graph parameters: one stay probability and ``K - 1`` branch
        probabilities per non-absorbing macro state, ``n - 1`` entry
        probabilities per phase with ``n`` macro states.  Stage stay
        probabilities are tied and stage counts are structure, not counted."""
        n = sum(1 + len(m.successors) - 1 for m in self.macros if not m.absorbing)
        for p in range(N_PHASES):
            k = sum(m.phase == p for m in self.macros)
            n += max(k - 1, 0)
        return int(n)

    def mean_dwell_s(self) -> np.ndarray:
        """Mean dwell (s) of every macro state (``inf`` when absorbing)."""
        k = np.array([m.stages for m in self.macros], dtype=np.float64)
        with np.errstate(divide="ignore"):
            return np.where(self.stay < 1.0, k / (1.0 - self.stay), np.inf) / self.rate_hz

    # ── inference ──

    def _check(self, log_e: np.ndarray, phases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        log_e = np.ascontiguousarray(log_e, dtype=np.float64)
        phases = np.ascontiguousarray(phases, dtype=np.int64)
        if log_e.ndim != 2 or log_e.shape[1] != N_LABELS:
            raise ValueError(f"log_e must be (T, {N_LABELS}), got {log_e.shape}")
        if phases.shape != (log_e.shape[0],):
            raise ValueError("phases must be (T,) matching log_e")
        if log_e.shape[0] == 0:
            raise ValueError("empty sequence")
        self._check_phases(phases)
        if np.isnan(log_e).any() or np.isposinf(log_e).any():
            raise ValueError("log_e must not hold NaN or +inf")
        return log_e, phases

    def _check_phases(self, phases: np.ndarray) -> None:
        if phases.min() < 0 or phases.max() >= N_PHASES:
            raise ValueError(f"phase codes must lie in 0..{N_PHASES - 1}")
        present = cast(np.ndarray, self.compiled.mask.any(axis=1))
        missing = [int(p) for p in np.unique(phases) if not present[p]]
        if missing:
            raise ValueError(f"grammar has no states in phases {missing}")

    def loglik(self, log_e: np.ndarray, phases: np.ndarray) -> float:
        """Exact log-likelihood of one sequence (forward pass only)."""
        log_e, phases = self._check(log_e, phases)
        c = self.compiled
        none2, none1 = np.empty((0, 0)), np.empty(0)
        return float(
            _forward(
                log_e,
                phases,
                c.state_label,
                c.row_ptr,
                c.col,
                c.log_prob,
                c.log_init,
                c.mask,
                none2,
                none1,
                False,
            )
        )

    def forward_backward(self, log_e: np.ndarray, phases: np.ndarray) -> FB:
        """Exact posterior of one sequence: ``log_e (T, N_LABELS)`` holds
        ``log p(x_t | past, label)`` (entries of labels the phase does not use
        are ignored), ``phases (T,)`` the phase codes."""
        log_e, phases = self._check(log_e, phases)
        c = self.compiled
        n_t, n_s = log_e.shape[0], self.n_states
        la = np.empty((n_t, n_s))
        norm = np.empty(n_t)
        ll = float(
            _forward(
                log_e,
                phases,
                c.state_label,
                c.row_ptr,
                c.col,
                c.log_prob,
                c.log_init,
                c.mask,
                la,
                norm,
                True,
            )
        )
        if not np.isfinite(ll):
            raise ValueError("observations impossible under the grammar")
        occ = np.zeros((n_t, N_LABELS))
        edges = np.zeros(c.col.shape[0])
        entries = np.zeros(n_s)
        _backward(
            log_e,
            phases,
            c.state_label,
            c.row_ptr,
            c.col,
            c.log_prob,
            c.mask,
            la,
            norm,
            occ,
            edges,
            entries,
        )
        return FB(loglik=ll, occupancy=occ, edges=edges, entries=entries)

    # ── fitting ──

    def zero_stats(self) -> Stats:
        return Stats(edges=np.zeros(self.compiled.col.shape[0]), entries=np.zeros(self.n_states))

    def m_step(self, stats: Stats) -> Grammar:
        """The MAP graph given expected counts (maximises the expected
        complete-data log-likelihood plus :meth:`log_prior`)."""
        c = self.compiled
        if stats.edges.shape != c.col.shape or stats.entries.shape != (self.n_states,):
            raise ValueError("counts of a different graph")
        if not (np.isfinite(stats.edges).all() and np.isfinite(stats.entries).all()):
            raise ValueError("non-finite counts")
        n_m = len(self.macros)
        stays = np.bincount(c.edge_macro, weights=stats.edges * (c.kind == _STAY), minlength=n_m)
        leaves = np.bincount(
            c.edge_macro,
            weights=stats.edges * ((c.kind == _ADVANCE) | (c.kind == _EXIT)),
            minlength=n_m,
        )
        stay = np.ones(n_m)
        branch: list[np.ndarray] = []
        for i, m in enumerate(self.macros):
            if m.absorbing:
                branch.append(np.zeros(0))
                continue
            stay[i] = (stays[i] + PSEUDO) / (stays[i] + leaves[i] + 2.0 * PSEUDO)
            sel = (c.kind == _EXIT) & (c.edge_macro == i)
            ex = np.zeros(len(m.successors))
            np.add.at(ex, c.edge_succ[sel], stats.edges[sel])
            branch.append(_normalise(ex + PSEUDO))
        ent = stats.entries[c.first_state] + ENTRY_PSEUDO
        entry = np.zeros(n_m)
        for p in range(N_PHASES):
            sel = np.array([m.phase == p for m in self.macros])
            if sel.any():
                entry[sel] = _normalise(ent[sel])
        return replace(self, stay=stay, branch=tuple(branch), entry=entry)

    def log_prior(self) -> float:
        """Log density (up to a constant) of the pseudo-count prior that
        :meth:`m_step` maximises against."""
        lp = 0.0
        for i, m in enumerate(self.macros):
            if m.absorbing:
                continue
            r = self.stay[i]
            lp += PSEUDO * (np.log(r) + np.log1p(-r))
            if len(m.successors) > 1:
                lp += PSEUDO * float(np.log(self.branch[i]).sum())
        for p in range(N_PHASES):
            sel = np.array([m.phase == p for m in self.macros])
            if sel.sum() > 1:
                lp += ENTRY_PSEUDO * float(np.log(self.entry[sel]).sum())
        return float(lp)

    # ── generation ──

    def sample(
        self, phases: np.ndarray, rng: np.random.Generator, n: int | None = None
    ) -> np.ndarray:
        """Label paths along ``phases``: ``(T,)`` int64, or ``(n, T)`` when
        ``n`` is given."""
        phases = np.ascontiguousarray(phases, dtype=np.int64)
        if phases.ndim != 1 or phases.shape[0] == 0:
            raise ValueError("phases must be a non-empty (T,) array")
        self._check_phases(phases)
        k = 1 if n is None else int(n)
        u = rng.random((k, phases.shape[0]))
        out = np.empty((k, phases.shape[0]), dtype=np.int64)
        c = self.compiled
        _sample(phases, u, c.state_label, c.row_ptr, c.col, c.prob, c.init, out)
        return out[0] if n is None else out

    # ── serialisation ──

    def to_dict(self) -> dict[str, Any]:
        dwell = self.mean_dwell_s()
        return {
            "hierarchical": bool(self.hierarchical),
            "rate_hz": float(self.rate_hz),
            "n_params": self.n_params,
            "n_states": self.n_states,
            "labels": [s.name for s in LABELS],
            "macros": [
                {
                    "name": m.name,
                    "phase": m.phase,
                    "label": LABELS[m.label].name,
                    "stages": m.stages,
                    "successors": list(m.successors),
                    "stay": float(self.stay[i]),
                    "branch": [float(b) for b in self.branch[i]],
                    "entry": float(self.entry[i]),
                    "mean_dwell_s": float(dwell[i]) if np.isfinite(dwell[i]) else None,
                }
                for i, m in enumerate(self.macros)
            ],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Grammar:
        if list(d["labels"]) != [s.name for s in LABELS]:
            raise ValueError("grammar was saved with a different label vocabulary")
        ms = d["macros"]
        return cls(
            macros=tuple(
                MacroSpec(
                    m["name"],
                    int(m["phase"]),
                    LABEL_ID[m["label"]],
                    int(m["stages"]),
                    tuple(m["successors"]),
                )
                for m in ms
            ),
            stay=np.array([m["stay"] for m in ms]),
            branch=tuple(np.array(m["branch"], dtype=np.float64) for m in ms),
            entry=np.array([m["entry"] for m in ms]),
            hierarchical=bool(d["hierarchical"]),
            rate_hz=float(d["rate_hz"]),
        )


# ─── the two grammars ─────────────────────────────────────────────────────────

# (name, label, successors, default mean dwell s, default branch law, default entry weight)
_Def = tuple[str, str, tuple[str, ...], float, tuple[float, ...], float]

_OFF: tuple[_Def, ...] = (("off.rest", "rest", (), 0.0, (), 1.0),)
_STARTUP: tuple[_Def, ...] = (
    ("start.predrive", "predrive", ("start.pulse", "start.idle"), 1.0, (0.8, 0.2), 0.8),
    ("start.pulse", "pulse", ("start.release",), 1.0, (1.0,), 0.1),
    ("start.release", "release", ("start.idle", "start.pulse"), 1.5, (0.8, 0.2), 0.05),
    ("start.idle", "idle", (), 0.0, (), 0.05),
)
_LANDING: tuple[_Def, ...] = (
    ("land.descend", "descend", ("land.shelf", "land.stop"), 3.0, (0.7, 0.3), 0.85),
    ("land.shelf", "shelf", ("land.stop",), 2.0, (1.0,), 0.05),
    ("land.stop", "stop", ("land.rest",), 2.0, (1.0,), 0.05),
    ("land.rest", "rest", (), 0.0, (), 0.05),
)
_PARENTS = ("quiet.hold", "event.hold", "alt.hold")
_AIR_ORDERED: tuple[_Def, ...] = (
    ("air.rise", "rise", ("air.overshoot", *_PARENTS), 2.0, (0.5, 0.2, 0.2, 0.1), 0.9),
    ("air.overshoot", "overshoot", _PARENTS, 1.5, (0.4, 0.4, 0.2), 0.02),
    ("quiet.hold", "hold", ("event.hold", "alt.hold"), 15.0, (0.6, 0.4), 0.02),
    (
        "event.hold",
        "hold",
        ("event.up", "event.down", "quiet.hold", "alt.hold"),
        6.0,
        (0.35, 0.35, 0.15, 0.15),
        0.02,
    ),
    ("event.up", "drive_up", ("event.hold",), 1.5, (1.0,), 0.01),
    ("event.down", "drive_down", ("event.hold",), 1.5, (1.0,), 0.01),
    (
        "alt.hold",
        "hold",
        ("alt.up", "alt.down", "quiet.hold", "event.hold"),
        1.0,
        (0.4, 0.4, 0.1, 0.1),
        0.01,
    ),
    ("alt.up", "drive_up", ("alt.down", "alt.hold"), 1.0, (0.8, 0.2), 0.005),
    ("alt.down", "drive_down", ("alt.up", "alt.hold"), 1.0, (0.8, 0.2), 0.005),
)
_AIR_FLAT: tuple[_Def, ...] = (
    ("air.rise", "rise", ("air.overshoot", "cruise.hold"), 2.0, (0.5, 0.5), 0.9),
    ("air.overshoot", "overshoot", ("cruise.hold",), 1.5, (1.0,), 0.04),
    ("cruise.hold", "hold", ("cruise.up", "cruise.down"), 6.0, (0.5, 0.5), 0.04),
    ("cruise.up", "drive_up", ("cruise.down", "cruise.hold"), 1.2, (0.5, 0.5), 0.01),
    ("cruise.down", "drive_down", ("cruise.up", "cruise.hold"), 1.2, (0.5, 0.5), 0.01),
)


def build_grammar(hierarchical: bool, rate_hz: float = RATE_HZ) -> Grammar:
    """The ordered (``hierarchical=True``) or the flat comparator grammar at
    its data-free default parameters (dwell means and branch laws stating the
    intended roles; every entry, branch and stay probability nonzero)."""
    defs = (*_OFF, *_STARTUP, *(_AIR_ORDERED if hierarchical else _AIR_FLAT), *_LANDING)
    phase_of = {
        d[0]: p
        for p, ds in enumerate((_OFF, _STARTUP, _AIR_ORDERED + _AIR_FLAT, _LANDING))
        for d in ds
    }
    macros, stay, branch, entry = [], [], [], []
    for name, label, succ, dwell_s, law, w in defs:
        k = STAGES[label]
        macros.append(MacroSpec(name, phase_of[name], LABEL_ID[label], k, succ))
        stay.append(1.0 if not succ else 1.0 - k / max(dwell_s * rate_hz, k + 1.0))
        branch.append(np.array(law, dtype=np.float64))
        entry.append(w)
    entry_a = np.array(entry)
    for p in range(N_PHASES):
        sel = np.array([m.phase == p for m in macros])
        entry_a[sel] /= entry_a[sel].sum()
    return Grammar(
        macros=tuple(macros),
        stay=np.array(stay),
        branch=tuple(branch),
        entry=entry_a,
        hierarchical=hierarchical,
        rate_hz=rate_hz,
    )


__all__ = [
    "ENTRY_PSEUDO",
    "FB",
    "LABELS",
    "LABEL_ID",
    "N_LABELS",
    "N_PHASES",
    "PHASES",
    "PHASE_LABELS",
    "PSEUDO",
    "RATE_HZ",
    "STAGES",
    "Compiled",
    "Grammar",
    "LabelSpec",
    "MacroSpec",
    "Stats",
    "build_grammar",
]
