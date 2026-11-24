"""Hierarchical feedback-controller law: exact semi-Markov forward-backward
against enumeration, the variational bound against dense Gaussians (equal when
the actions coincide, below otherwise), monotone fit and parameter accounting,
learned arming time, source isolation of command-only records, grammar
recovery, and the sampling/serialization contract."""

from __future__ import annotations

import itertools
import json
from dataclasses import replace
from typing import cast

import numpy as np
import pytest
from scipy.signal import lfilter
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from experiments.rps_traj import controller_law as cl
from tracking.rotors import MIXER

ARM_S = 1.0  # true spin-up time of the synthetic measured flights


def _ramp(t, a, b):
    u = np.clip((t - a) / (b - a), 0.0, 1.0)
    return u * u * (3 - 2 * u)


def _flight(rng, idle, hover, parts=(25, 150, 900, 200), trim=(0.5, -0.3, 2.0), cmd=False):
    """Measured: smooth spin-up over ARM_S, noisy idle; command: a step to idle, noiseless ground.
    Cruise alternates calm (sd 0.15) and active (sd 1.05) one-second blocks in runs of 4-9."""
    codes = np.repeat(np.arange(4), parts).astype(np.int8)
    T = len(codes)
    t = np.arange(T) / cl.FS
    on, to, la = np.cumsum(parts)[:3] / cl.FS
    armed = (t >= on).astype(float) if cmd else _ramp(t, on, on + ARM_S)
    landed = _ramp(t, la - 0.3, la + 0.5)
    modes = np.zeros((4, T))
    modes[0] = idle * armed + (hover - idle) * _ramp(t, to, to + 1.5) - hover * landed
    for m in range(1, 4):
        modes[m] = trim[m - 1] * armed * (1 - landed)
    blk = np.floor(t - to - cl.TAKEOFF_END).astype(int)
    runs = np.cumsum(rng.integers(4, 10, size=400))
    active = (np.searchsorted(runs, np.maximum(blk, 0), side="right") % 2).astype(float)
    cruise = (t >= to + cl.TAKEOFF_END) & (t < la + cl.APPROACH_S)
    sd = np.where(codes == 1, 0.0 if cmd else 0.05, 0.0) + np.where(
        codes == 2, 0.15 + 0.9 * active * cruise, 0.0
    )
    sd += np.where(codes == 3, 0.1, 0.0)
    for m in range(4):
        modes[m] += lfilter([1.0], [1.0, -0.7], rng.standard_normal(T) * sd)
    truth = active[cruise][:: cl.BLOCK]
    return np.maximum(MIXER @ modes, 0.0), codes, truth


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    return [
        _flight(rng, 30.0, 75.0),
        _flight(rng, 31.0, 80.0, trim=(1.0, 0.2, 2.5)),
        _flight(rng, 30.5, 78.0, parts=(20, 120, 700, 150)),
        _flight(rng, 34.0, 77.0, trim=(-1.5, 0.0, 0.5), cmd=True),
    ]


MEASURED = [True, True, True, False]


@pytest.fixture(scope="module")
def model(data):
    return cl.fit([(w, c) for w, c, _ in data], measured=MEASURED, max_iter=8)


# ─── semi-Markov grammar ──────────────────────────────────────────────────────


def _random_grammar(rng, A, dm):
    li = np.log(rng.dirichlet(np.ones(A)))
    lt = np.full((A, A), -np.inf)
    for a in range(A):
        lt[a, [b for b in range(A) if b != a]] = np.log(rng.dirichlet(np.ones(A - 1)))
    lp = np.log(rng.dirichlet(np.ones(dm), size=A))
    ls = np.logaddexp.accumulate(lp[:, ::-1], axis=1)[:, ::-1]
    return li, lt, lp, ls


def _paths(nb, A, dm):
    """Every (durations, labels) with no self-transition; the last segment censored."""

    def segs(n):
        if n == 0:
            yield []
            return
        for d in range(1, min(n, dm) + 1):
            for rest in segs(n - d):
                yield [d, *rest]

    for s in segs(nb):
        for labs in itertools.product(range(A), repeat=len(s)):
            if all(a != b for a, b in zip(labs, labs[1:])):
                yield s, labs


def _path_logp(s, labs, li, lt, lp, ls):
    v = li[labs[0]]
    for i, (d, a) in enumerate(zip(s, labs)):
        v += ls[a, d - 1] if i == len(s) - 1 else lp[a, d - 1]
        if i:
            v += lt[labs[i - 1], a]
    return v


def test_hsmm_forward_backward_matches_enumeration():
    rng = np.random.default_rng(0)
    A, nb, dm = 3, 6, 4  # nb > dm: the duration cap binds
    e = 2.0 * rng.standard_normal((nb, A))
    li, lt, lp, ls = _random_grammar(rng, A, dm)
    logz, gamma, init, trans, dur, cens = cl._hsmm(e, li, lt, lp, ls)
    recs = []
    for s, labs in _paths(nb, A, dm):
        lab_t = np.repeat(labs, s)
        recs.append((_path_logp(s, labs, li, lt, lp, ls) + e[np.arange(nb), lab_t].sum(), s, labs))
    Z = logsumexp([r[0] for r in recs])
    G, Dc, Cc, Tr, In = (
        np.zeros((nb, A)),
        np.zeros((A, dm)),
        np.zeros((A, dm)),
        np.zeros((A, A)),
        np.zeros(A),
    )
    for v, s, labs in recs:
        w = np.exp(v - Z)
        In[labs[0]] += w
        G[np.arange(nb), np.repeat(labs, s)] += w
        for i, (d, a) in enumerate(zip(s, labs)):
            (Cc if i == len(s) - 1 else Dc)[a, d - 1] += w
            if i:
                Tr[labs[i - 1], a] += w
    assert logz == pytest.approx(Z, abs=1e-10)
    for got, want in ((gamma, G), (init, In), (trans, Tr), (dur, Dc), (cens, Cc)):
        np.testing.assert_allclose(got, want, atol=1e-12)


# ─── likelihood: the bound against dense Gaussians ───────────────────────────


def _short(data, k=0):
    """A short measured record that still crosses every phase and holds two cruise blocks."""
    w, codes, _ = data[k]
    on, to, la = (int(np.argmax(codes >= c)) for c in (1, 2, 3))
    cut = np.r_[
        np.arange(on - 5, on + 60),
        np.arange(to, to + 100 + 2 * cl.BLOCK),
        np.arange(la - 50, la + 30),
    ]
    c = np.repeat([0, 1, 2, 3], [5, 60, 100 + 2 * cl.BLOCK + 50, 30]).astype(np.int8)
    return w[:, cut], c


def _dense_logpdf(model, w, codes, path_groups):
    """Exact log-density of the whole record given the cruise action of every sample
    (random effects integrated by a dense Gaussian per mode; modes are a priori independent)."""
    p, lay = model.params, model.layout
    y, lj = cl.to_obs(w)
    z = MIXER.T @ y / 4
    T = len(codes)
    clk = cl.event_clocks(codes)
    blocks = cl._design(clk, True, p.tau_arm)
    sg = path_groups
    prow = np.array([cl.PHI_ROW[g] if g < cl.N_SG else g - cl.N_SG + cl.N_PG for g in sg])
    total = lj.sum() + T * cl.LOG_DET_MODES
    for m, L in enumerate(lay.modes):
        ph = np.array([np.zeros(2) if r < 0 else p.phi[r, m] for r in prow])
        g = 1.0 - ph.sum(axis=1)
        Lm = np.eye(T) - np.diag(ph[1:, 0], -1) - np.diag(ph[2:, 1], -2)
        l0 = np.zeros(T)
        l0[0] = (ph[0, 0] + ph[0, 1]) * cl.Z_REST[m]
        l0[1] = ph[1, 1] * cl.Z_REST[m]
        U = cl._umat(L, blocks)
        wc, M = cl._wc_M(L, p.theta)
        Li = np.linalg.inv(Lm)
        A = Li @ (g[:, None] * (U @ M))
        mean = Li @ (g * (U @ wc) + l0) + A @ p.nu[L.eff]
        cov = Li @ np.diag(p.sig2[sg, m]) @ Li.T + A @ np.diag(p.D[L.eff]) @ A.T
        total += multivariate_normal(mean, cov, allow_singular=False).logpdf(z[m])
    return total


def test_bound_is_exact_when_actions_coincide(model, data):
    w, codes = _short(data)
    p = model.params
    same = replace(p, phi=p.phi.copy(), sig2=p.sig2.copy())
    same.phi[cl.N_PG :] = p.phi[cl.N_PG]
    same.sig2[cl.N_SG :] = p.sig2[cl.N_SG]
    m2 = cl.ControllerLawModel(same, model.observed_sigma, model.observed_phi)
    sg = cl._sgroups(codes, cl.event_clocks(codes), True, p.tau_arm)
    assert (sg < 0).sum() == 2 * cl.BLOCK
    sg = np.where(sg < 0, cl.N_SG, sg)
    exact = _dense_logpdf(m2, w, codes, sg)
    assert m2.elbo([(w, codes)])[0] == pytest.approx(exact, rel=1e-9, abs=1e-6)


def test_bound_is_below_the_exact_mixture_and_tight(model, data):
    w, codes = _short(data)
    p = model.params
    clk = cl.event_clocks(codes)
    base = cl._sgroups(codes, clk, True, p.tau_arm)
    cruise = np.flatnonzero(base < 0)
    lp, ls = cl._dur_tables(p.dur)
    terms = []
    for k in range(cl.N_PROGRAMMES):
        for s, labs in _paths(2, cl.N_ACTIONS, cl.DMAX):
            sg = base.copy()
            sg[cruise] = cl.N_SG + np.repeat(np.repeat(labs, s), cl.BLOCK)
            lpath = _path_logp(s, labs, p.log_init[k], p.log_trans[k], lp, ls)
            terms.append(p.log_pi[k] + lpath + _dense_logpdf(model, w, codes, sg))
    exact = cast(float, logsumexp(terms))
    bound = model.elbo([(w, codes)])[0]
    assert bound <= exact + 1e-6
    assert exact - bound < 1.0  # the flight effects are pinned: the mean-field gap is small


# ─── fit ──────────────────────────────────────────────────────────────────────


def test_fit_is_monotone_and_counts_parameters(model):
    h = np.asarray(model.history)
    assert len(h) == model.fit_info["n_iter"] + 1 >= 2 and np.isfinite(h).all()
    assert (np.diff(h) >= -cl.MONOTONE_RTOL * np.abs(h[1:])).all()
    assert model.fit_info["stop_reason"] in ("converged", "max_iter")
    br = model.param_breakdown()
    assert model.n_params == sum(br.values()) < 277
    assert (
        br["source_intercepts"] == 8 and br["cmd_cruise_scale"] == 4
    )  # command-only record present
    # every free scalar is counted exactly once: theta, observed relaxation and noise groups
    assert (
        br["mean_common"]
        + br["mean_rotor_diff"]
        + sum(v for k, v in br.items() if k.startswith("loading_"))
        == model.layout.n_theta
    )
    assert (
        br["noise_var"] == 4 * model.observed_sigma.sum()
        and br["relaxation_ar"] == 8 * model.observed_phi.sum()
    )
    p = model.params
    assert (cl._bary(p.phi) > 0).all() and (p.sig2 >= cl.SIGMA_FLOOR**2).all() and (p.D > 0).all()
    assert p.tau_arm in cl.TAU_GRID
    s = model.summary()
    assert {"tau_arm_s", "relaxation", "noise_sd", "levels", "action_trans"} <= set(s)


def test_arming_time_tracks_the_spin_up(model):
    # synthetic measured flights settle ARM_S after motors-on
    assert 0.6 * ARM_S <= model.params.tau_arm <= 1.4 * ARM_S


def test_command_records_cannot_move_measured_levels(data):
    measured_only = cl.fit([(w, c) for w, c, _ in data[:3]], max_iter=8)
    both = cl.fit([(w, c) for w, c, _ in data], measured=MEASURED, max_iter=8)
    lay = both.layout
    for name in ("common.idle", "common.hover", "roll.idle", "yaw.hover"):
        j = lay.names.index(name)
        assert both.params.nu[j] == pytest.approx(measured_only.params.nu[j], abs=0.15)
    idle, roll = lay.names.index("common.idle"), lay.names.index("roll.idle")
    assert both.params.delta[idle] == pytest.approx(34.0 - 30.5, abs=0.6)  # the command idle offset
    assert both.params.delta[roll] < -1.5
    assert measured_only.param_breakdown()["source_intercepts"] == 0
    assert measured_only.params.tau_arm == both.params.tau_arm


def test_programme_grammar_recovers_cruise_regimes(model, data):
    p, lay = model.params, model.layout
    sd_common = np.sqrt(p.sig2[cl.N_SG :, 0])
    wcM = [cl._wc_M(L, p.theta) for L in lay.modes]
    tables = cl._dur_tables(p.dur)
    hits = []
    for (w, c, truth), ms in zip(data, MEASURED):
        if not ms:
            continue
        f = cl._prepare(w, c, True, lay, p.tau_arm)
        m0, Psi = cl._prior(p, f)
        _, _, path, _ = cl._flight_q(f, p, lay, wcM, tables, m0, np.diag(Psi), 10)
        pred = path.gamma @ sd_common > 0.4
        n = min(len(truth), len(pred))
        hits.append(np.mean(pred[:n] == truth[:n].astype(bool)))
    assert np.mean(hits) > 0.85


def test_relaxation_step_is_the_constrained_minimum():
    rng = np.random.default_rng(1)
    g = np.linspace(-2, 2, 161)
    grid = np.array([(a, b) for a in g for b in np.linspace(-1, 1, 81)])
    grid = grid[(cl._bary(grid) > 1e-9).all(axis=1)]
    for _ in range(4):
        X = rng.standard_normal((40, 3))
        K = X.T @ X
        x = cl._relax_step(K, np.zeros(2))

        def f(v, K=K):
            return (
                0.5 * np.r_[1.0, -v] @ K @ np.r_[1.0, -v]
                - cl.STAB_PRIOR * np.log(cl._bary(v)).sum()
            )

        assert (cl._bary(x) > 0).all()
        assert f(x) <= min(f(v) for v in grid) + 1e-9


# ─── generation ───────────────────────────────────────────────────────────────


def test_sample_contract_and_serialization(model):
    codes = np.repeat(np.arange(4), [30, 100, 600, 120]).astype(np.int8)
    W = model.sample(codes, 5, np.random.default_rng(0))
    assert W.shape == (5, 4, len(codes)) and np.isfinite(W).all() and (W >= 0).all()
    assert np.median(W[:, :, 400:500].mean(axis=1)) > 60.0  # cruise is airborne
    assert np.all(W[:, :, :20] < 2.0)  # at rest before motors-on
    assert 0.0 <= model.last_clip_fraction <= 1.0
    blob = json.loads(json.dumps(model.to_dict()))
    again = cl.ControllerLawModel.from_dict(blob)
    assert again.n_params == model.n_params
    np.testing.assert_array_equal(again.sample(codes, 5, np.random.default_rng(0)), W)
    assert model.sample(np.full(200, 2, dtype=np.int8), 2, np.random.default_rng(1)).shape == (
        2,
        4,
        200,
    )
    no_landing = np.repeat([0, 1, 2], [10, 60, 300]).astype(np.int8)
    assert np.isfinite(model.sample(no_landing, 2, np.random.default_rng(2))).all()


def test_long_hold_stays_at_the_idle_equilibrium(model):
    # a 15 s hold (training: <= 6 s) is exogenous: standby relaxes around the idle level, no drift
    codes = np.repeat(np.arange(4), [25, 375, 300, 100]).astype(np.int8)
    W = model.sample(codes, 32, np.random.default_rng(4))
    idle = cl.from_obs(np.full(4, model.params.nu[model.layout.names.index("common.idle")]))[0]
    late = W[:, :, 25 + 325 : 25 + 375].mean(axis=1)  # last 2 s before take-off
    assert abs(np.median(late) - idle) < 1.0
    assert np.quantile(np.abs(late - idle), 0.9) < 2.0
