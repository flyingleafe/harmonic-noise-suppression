"""The action grammar must be exact where it claims to be.

The forward-backward kernel is checked against explicit enumeration of every
expanded-state path of short sequences (likelihood, label posteriors,
expected edge and entry counts), the sampler against the phase schedule, the
graph's edges and the negative-binomial dwell it claims, and the ordered
grammar against its flat comparator on data with parent memory: fitted by
EM on the same evidence, the ordered grammar must reproduce the dependence
of an action's successor on what preceded it, the flat one must not.
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np
import pytest
from scipy.special import comb, logsumexp

from experiments.rps_traj import unified_grammar as ug

UP, DOWN, HOLD = ug.LABEL_ID["drive_up"], ug.LABEL_ID["drive_down"], ug.LABEL_ID["hold"]


# ─── helpers ──────────────────────────────────────────────────────────────────


def _randomised(g: ug.Grammar, rng: np.random.Generator) -> ug.Grammar:
    n = len(g.macros)
    stay = np.where([m.absorbing for m in g.macros], 1.0, rng.uniform(0.1, 0.9, n))
    branch = tuple(
        rng.dirichlet(np.ones(len(m.successors))) if m.successors else np.zeros(0) for m in g.macros
    )
    entry = rng.uniform(0.1, 1.0, n)
    phase = np.array([m.phase for m in g.macros])
    for p in np.unique(phase):
        entry[phase == p] /= entry[phase == p].sum()
    return replace(g, stay=stay, branch=branch, entry=entry)


def _brute(g: ug.Grammar, log_e: np.ndarray, phases: np.ndarray):
    """Likelihood, label posteriors, edge and entry counts by enumerating
    every expanded-state path."""
    c = g.compiled
    n_s, n_t = g.n_states, len(phases)
    trans = np.zeros((n_s, n_s))
    eidx = -np.ones((n_s, n_s), dtype=np.int64)
    for i in range(n_s):
        for e in range(c.row_ptr[i], c.row_ptr[i + 1]):
            assert eidx[i, c.col[e]] < 0, "duplicate edge"
            trans[i, c.col[e]] = c.prob[e]
            eidx[i, c.col[e]] = e
    allowed = [np.flatnonzero(c.state_phase == p) for p in phases]
    paths = np.stack(np.meshgrid(*allowed, indexing="ij"), axis=-1).reshape(-1, n_t)
    lab = c.state_label[paths]
    with np.errstate(divide="ignore"):
        lw = np.log(c.init[phases[0], paths[:, 0]]) + log_e[0, lab[:, 0]]
        for t in range(1, n_t):
            if phases[t] != phases[t - 1]:
                lw += np.log(c.init[phases[t], paths[:, t]])
            else:
                lw += np.log(trans[paths[:, t - 1], paths[:, t]])
            lw += log_e[t, lab[:, t]]
    ll = float(cast(float, logsumexp(lw)))
    w = np.exp(lw - ll)
    occ = np.zeros((n_t, ug.N_LABELS))
    edges = np.zeros(c.col.shape[0])
    entries = np.zeros(n_s)
    for t in range(n_t):
        np.add.at(occ[t], lab[:, t], w)
        if t == 0 or phases[t] != phases[t - 1]:
            np.add.at(entries, paths[:, t], w)
        else:
            e = eidx[paths[:, t - 1], paths[:, t]]
            ok = w > 0
            assert (e[ok] >= 0).all()
            np.add.at(edges, e[ok], w[ok])
    return ll, occ, edges, entries


def _tiny(entry: tuple[float, ...] = (0.5, 0.2, 0.3, 0.6, 0.4)) -> ug.Grammar:
    L = ug.LABEL_ID
    macros = (
        ug.MacroSpec("a", 1, L["predrive"], 2, ("b",)),
        ug.MacroSpec("b", 1, L["pulse"], 1, ("a", "c")),
        ug.MacroSpec("c", 1, L["idle"], 1, ()),
        ug.MacroSpec("h", 2, L["hold"], 2, ("u",)),
        ug.MacroSpec("u", 2, L["drive_up"], 1, ("h",)),
    )
    return ug.Grammar(
        macros=macros,
        stay=np.array([0.6, 0.3, 1.0, 0.5, 0.4]),
        branch=(
            np.array([1.0]),
            np.array([0.7, 0.3]),
            np.zeros(0),
            np.array([1.0]),
            np.array([1.0]),
        ),
        entry=np.array(entry),
    )


def _dense(g: ug.Grammar, log_e: np.ndarray, phases: np.ndarray) -> tuple[float, np.ndarray]:
    """Log-domain dense forward-backward (scipy ``logsumexp`` over the full
    transition matrix): the reference for sequences too long to enumerate."""
    c = g.compiled
    n_s, n_t = g.n_states, len(phases)
    log_trans = np.full((n_s, n_s), -np.inf)
    for i in range(n_s):
        for e in range(c.row_ptr[i], c.row_ptr[i + 1]):
            log_trans[i, c.col[e]] = c.log_prob[e]
    em = np.where(c.mask[phases][:, c.state_label], log_e[:, c.state_label], -np.inf)
    reset = np.r_[True, phases[1:] != phases[:-1]]
    la = np.empty((n_t, n_s))
    z = np.empty(n_t)
    with np.errstate(divide="ignore", invalid="ignore"):
        for t in range(n_t):
            pred = (
                c.log_init[phases[t]]
                if reset[t]
                else logsumexp(la[t - 1][:, None] + log_trans, axis=0)
            )
            w = pred + em[t]
            z[t] = logsumexp(w)
            la[t] = w - z[t]
        occ = np.zeros((n_t, ug.N_LABELS))
        lb = np.zeros(n_s)
        for t in range(n_t - 1, -1, -1):
            np.add.at(occ[t], c.state_label, np.exp(la[t] + lb))
            lb = (
                np.zeros(n_s)
                if reset[t]
                else logsumexp(log_trans + (em[t] + lb)[None, :], axis=1) - z[t]
            )
    return float(z.sum()), occ


def _runs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(values, lengths)`` of the runs of equal consecutive entries."""
    cut = np.flatnonzero(np.diff(x) != 0) + 1
    starts = np.concatenate([[0], cut])
    return x[starts], np.diff(np.concatenate([starts, [len(x)]]))


def _schedule(n0: int, n1: int, n2: int, n3: int) -> np.ndarray:
    return np.repeat(np.arange(4), [n0, n1, n2, n3])


# ─── structure ────────────────────────────────────────────────────────────────


def test_vocabulary_and_parameter_count():
    """Both grammars use every label of every phase with the same stage count
    per label; free parameters are counted once (tied stages excluded)."""
    ordered, flat = ug.build_grammar(True), ug.build_grammar(False)
    for g in (ordered, flat):
        for p in range(ug.N_PHASES):
            used = {m.label for m in g.macros if m.phase == p}
            assert used == set(ug.PHASE_LABELS[p])
        for m in g.macros:
            assert m.stages == ug.STAGES[ug.LABELS[m.label].name]
    # one fixed-offset label per anchor
    for ref in ("idle", "hover"):
        assert sum(s.ref == ref and not s.free_offset for s in ug.LABELS) == 1
    # the drives and the hold are one emission each, visited by several parents
    assert sum(m.label == UP for m in ordered.macros) == 2
    assert sum(m.label == HOLD for m in ordered.macros) == 3
    assert sum(m.label == UP for m in flat.macros) == 1
    assert ordered.n_params == 46
    assert flat.n_params == 28


def test_serialisation_roundtrip():
    g = _randomised(ug.build_grammar(True), np.random.default_rng(0))
    h = ug.Grammar.from_dict(g.to_dict())
    rng = np.random.default_rng(1)
    phases = _schedule(3, 20, 40, 20)
    log_e = rng.normal(0.0, 2.0, (len(phases), ug.N_LABELS))
    assert h.loglik(log_e, phases) == pytest.approx(g.loglik(log_e, phases), rel=1e-12)
    assert h.n_params == g.n_params


# ─── exact inference ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("scale", [1.0, 500.0])
@pytest.mark.parametrize(
    ("which", "phases"),
    [
        ("tiny", [1, 1, 1, 2, 2, 1, 1]),
        ("tiny0", [1, 1, 1, 1, 2, 2, 1]),
        ("ordered", [0, 1, 2, 2, 3, 3]),
        ("ordered", [2, 2, 2, 2]),
        ("flat", [2, 2, 2, 2, 2]),
    ],
)
def test_forward_backward_matches_enumeration(which, phases, scale):
    """Exact against enumeration, also at log-emission gaps of thousands of
    nats (``scale``) and when the best-scoring allowed label is unreachable
    (``tiny0`` enters phase 1 only through ``a``, so ``idle`` cannot be
    reached before the fourth sample)."""
    rng = np.random.default_rng(7)
    if which == "tiny":
        g = _tiny()
    elif which == "tiny0":
        g = _tiny((1.0, 0.0, 0.0, 1.0, 0.0))
    else:
        g = _randomised(ug.build_grammar(which == "ordered"), rng)
    phases = np.asarray(phases)
    log_e = scale * rng.normal(0.0, 2.0, (len(phases), ug.N_LABELS))
    if which == "tiny0":
        log_e[:3, ug.LABEL_ID["idle"]] = 0.0
        log_e[:3, ug.LABEL_ID["predrive"]] = -3.0 * scale
    # labels a phase does not use must be ignored, however large
    used = g.compiled.mask[phases]
    log_e[~used] = 100.0 * scale
    ll, occ, edges, entries = _brute(g, log_e, phases)
    fb = g.forward_backward(log_e, phases)
    assert fb.loglik == pytest.approx(ll, rel=1e-11, abs=1e-9)
    assert g.loglik(log_e, phases) == pytest.approx(ll, rel=1e-11, abs=1e-9)
    np.testing.assert_allclose(fb.occupancy, occ, atol=1e-8)
    np.testing.assert_allclose(fb.edges, edges, atol=1e-8)
    np.testing.assert_allclose(fb.entries, entries, atol=1e-8)
    np.testing.assert_allclose(fb.occupancy.sum(axis=1), 1.0, atol=1e-9)
    assert fb.edges.sum() + fb.entries.sum() == pytest.approx(len(phases))


def test_long_low_noise_track_stays_exact():
    """A long track whose evidence contradicts the graph at every sample by
    10^4 nats (iid labels against multi-sample dwells): a linear-domain
    recursion underflows here; the log-domain one must stay finite and match
    the dense reference."""
    g = _randomised(ug.build_grammar(True), np.random.default_rng(4))
    phases = _schedule(20, 300, 2000, 300)
    rng = np.random.default_rng(3)
    log_e = rng.normal(0.0, 1.0, (len(phases), ug.N_LABELS))
    for t, p in enumerate(phases):
        wrong = np.setdiff1d(ug.PHASE_LABELS[p], rng.choice(ug.PHASE_LABELS[p]))
        log_e[t, wrong] -= 1e4
    fb = g.forward_backward(log_e, phases)
    ll, occ = _dense(g, log_e, phases)
    assert np.isfinite(fb.loglik) and fb.loglik < -1e5
    assert fb.loglik == pytest.approx(ll, rel=1e-10)
    assert g.loglik(log_e, phases) == pytest.approx(fb.loglik, rel=1e-12)
    np.testing.assert_allclose(fb.occupancy, occ, atol=1e-7)
    for a in (fb.occupancy, fb.edges, fb.entries):
        assert np.isfinite(a).all() and (a >= 0).all()
    np.testing.assert_allclose(fb.occupancy.sum(axis=1), 1.0, atol=1e-9)
    assert fb.edges.sum() + fb.entries.sum() == pytest.approx(len(phases))
    log_e[10, 0] = np.nan
    with pytest.raises(ValueError):
        g.loglik(log_e, phases)


# ─── sampling ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("hierarchical", [True, False])
def test_sampler_follows_schedule_and_graph(hierarchical):
    g = _randomised(ug.build_grammar(hierarchical), np.random.default_rng(11))
    phases = _schedule(10, 150, 600, 150)
    paths = g.sample(phases, np.random.default_rng(5), n=200)
    assert paths.shape == (200, len(phases))
    np.testing.assert_array_equal(paths, g.sample(phases, np.random.default_rng(5), n=200))
    assert g.sample(phases, np.random.default_rng(5)).shape == (len(phases),)
    index = g.macro_index
    pairs = {
        (m.label, g.macros[index[s]].label)
        for m in g.macros
        for s in m.successors
        if g.macros[index[s]].label != m.label
    }
    min_len = {
        (p, lab): min(m.stages for m in g.macros if m.phase == p and m.label == lab)
        for p in range(ug.N_PHASES)
        for lab in ug.PHASE_LABELS[p]
    }
    bounds = np.flatnonzero(np.diff(phases) != 0) + 1
    segments = np.split(np.arange(len(phases)), bounds)
    for path in paths:
        for seg in segments:
            p = int(phases[seg[0]])
            vals, lens = _runs(path[seg])
            assert set(vals) <= set(ug.PHASE_LABELS[p])
            assert all((a, b) in pairs for a, b in zip(vals[:-1], vals[1:]))
            # every completed dwell lasts at least its stage count
            assert all(n >= min_len[(p, v)] for v, n in zip(vals[:-1], lens[:-1]))


def test_sampler_dwell_is_negative_binomial():
    L = ug.LABEL_ID
    k, r = 3, 0.7
    g = ug.Grammar(
        macros=(
            ug.MacroSpec("a", 1, L["predrive"], k, ("b",)),
            ug.MacroSpec("b", 1, L["idle"], 1, ()),
        ),
        stay=np.array([r, 1.0]),
        branch=(np.array([1.0]), np.zeros(0)),
        entry=np.array([1.0, 0.0]),
    )
    paths = g.sample(np.ones(120, dtype=np.int64), np.random.default_rng(2), n=20_000)
    dwell = (paths == L["predrive"]).sum(axis=1)
    d = np.arange(1, 60)
    pmf = np.where(d >= k, comb(d - 1, k - 1) * (1 - r) ** k * r ** (d - k), 0.0)
    emp = np.bincount(dwell, minlength=120)[1:60] / len(dwell)
    assert dwell.min() >= k
    np.testing.assert_allclose(emp, pmf, atol=0.01)
    assert dwell.mean() == pytest.approx(k / (1 - r), rel=0.03)
    assert g.mean_dwell_s()[0] == pytest.approx(k / (1 - r) / ug.RATE_HZ)


# ─── fitting and the ablation ─────────────────────────────────────────────────


def _truth() -> ug.Grammar:
    """An ordered grammar with clear parent memory: isolated events after long
    holds, long alternating bursts that rarely rest."""
    g = ug.build_grammar(True)
    dwell = {
        "air.rise": 2.0,
        "air.overshoot": 1.0,
        "quiet.hold": 8.0,
        "event.hold": 3.0,
        "event.up": 0.8,
        "event.down": 0.8,
        "alt.hold": 0.6,
        "alt.up": 0.5,
        "alt.down": 0.5,
    }
    laws = {
        "quiet.hold": [0.5, 0.5],
        "event.hold": [0.4, 0.4, 0.1, 0.1],
        "alt.hold": [0.3, 0.3, 0.2, 0.2],
        "alt.up": [0.95, 0.05],
        "alt.down": [0.95, 0.05],
    }
    idx = g.macro_index
    stay = g.stay.copy()
    branch = list(g.branch)
    for name, s in dwell.items():
        k = g.macros[idx[name]].stages
        stay[idx[name]] = 1.0 - k / (s * ug.RATE_HZ)
    for name, law in laws.items():
        branch[idx[name]] = np.array(law)
    return replace(g, stay=stay, branch=tuple(branch))


def _evidence(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Noisy per-label log-evidence favouring the true label."""
    log_e = rng.normal(-3.0, 0.5, (len(labels), ug.N_LABELS))
    log_e[np.arange(len(labels)), labels] = 0.0
    return log_e


def _fit(g: ug.Grammar, data, n_iter: int) -> tuple[ug.Grammar, list[float]]:
    history = []
    for _ in range(n_iter):
        stats = g.zero_stats()
        total = 0.0
        for log_e, phases in data:
            fb = g.forward_backward(log_e, phases)
            stats.add(fb)
            total += fb.loglik
        history.append(total + g.log_prior())
        g = g.m_step(stats)
    return g, history


def _memory(g: ug.Grammar, phases: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """``P(down | up, previous down)`` and ``P(down | up, previous hold)`` over
    the label runs of sampled airborne phases."""
    counts = np.zeros((2, 2))
    for path in g.sample(phases, rng, n=40):
        vals, _ = _runs(path[phases == 2])
        for prev, cur, nxt in zip(vals[:-2], vals[1:-1], vals[2:]):
            if cur == UP and prev in (DOWN, HOLD):
                counts[int(prev == HOLD), int(nxt == DOWN)] += 1
    assert counts.sum(axis=1).min() > 100
    p = counts[:, 1] / counts.sum(axis=1)
    return float(p[0]), float(p[1])


def test_em_is_monotone_and_ordered_grammar_keeps_parent_memory():
    rng = np.random.default_rng(42)
    truth = _truth()
    phases = _schedule(25, 250, 3000, 250)
    data = [(_evidence(truth.sample(phases, rng), rng), phases) for _ in range(8)]
    ordered, h_ord = _fit(ug.build_grammar(True), data, 25)
    flat, h_flat = _fit(ug.build_grammar(False), data, 25)
    for h in (h_ord, h_flat):
        assert np.all(np.diff(h) >= -1e-6 * np.abs(h[:-1]))
    # memory is worth likelihood on the same evidence
    assert h_ord[-1] > h_flat[-1] + 50.0
    # what follows an up-drive depends on what preceded it only under the
    # ordered grammar
    long = _schedule(25, 250, 5000, 250)
    t_dd, t_hd = _memory(truth, long, np.random.default_rng(1))
    o_dd, o_hd = _memory(ordered, long, np.random.default_rng(1))
    f_dd, f_hd = _memory(flat, long, np.random.default_rng(1))
    assert t_dd - t_hd > 0.4
    assert o_dd - o_hd > 0.3
    assert abs(f_dd - f_hd) < 0.12
    # the fitted alternating parent alternates
    idx = ordered.macro_index
    assert ordered.branch[idx["alt.up"]][0] > 0.8
