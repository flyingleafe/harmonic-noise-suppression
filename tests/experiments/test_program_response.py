"""Program-response generator: link, event-clock bases, sampler/likelihood
parity, exact mixture likelihood, source model, bounds and a tiny fit."""

from __future__ import annotations

import json
from typing import cast

import numpy as np
import pytest
from scipy.special import logsumexp

from experiments.rps_traj import program_response as pr
from tracking.rotors import MIXER


def _schedule(n0: int = 15, n1: int = 110, n2: int = 260, n3: int = 110) -> np.ndarray:
    return np.repeat(np.arange(4), [n0, n1, n2, n3]).astype(np.int64)


def _model(**kw) -> pr.ProgramResponseModel:
    return pr.ProgramResponseModel(pr.default_grammar(), pr.default_params(**kw))


def test_link_roundtrip_floor_and_jacobian():
    w = np.array([-0.2, 0.0, 0.03, 0.3, 3.0, 80.0])
    z, lj = pr._latent(w)
    back = pr._readout(z)
    above = w > pr.FLOOR_RPS
    np.testing.assert_allclose(back[above], w[above], rtol=1e-10)
    assert np.all(back[~above] == 0.0)  # floor values read out as exact zeros
    np.testing.assert_allclose(z[~above], pr.Z_REST, rtol=1e-12)
    for v in (0.3, 3.0, 80.0):
        h = 1e-6 * v
        num = (pr._latent(np.array([v + h]))[0] - pr._latent(np.array([v - h]))[0]) / (2 * h)
        assert np.exp(pr._latent(np.array([v]))[1][0]) == pytest.approx(num[0], rel=1e-6)
    assert np.all(pr._readout(np.array([-50.0, -5.0, 0.0, 5.0])) >= 0.0)


@pytest.mark.parametrize("shape", pr.SHAPES)
def test_basis_is_linear_interpolation_with_clamped_ends(shape):
    rng = np.random.default_rng(0)
    vals = rng.normal(size=shape.n)
    knots, fidx, fval = shape.nodes()
    ext = np.where(fidx >= 0, vals[np.maximum(fidx, 0)], fval)
    tau = np.concatenate([np.linspace(knots[0] - 2, knots[-1] + 2, 401), knots])
    B, b0 = pr._basis(shape, tau)
    np.testing.assert_allclose(B @ vals + b0, np.interp(tau, knots, ext), atol=1e-12)
    Bi, bi = pr._basis(shape, np.array([-np.inf, np.inf]))
    np.testing.assert_allclose(Bi @ vals + bi, [ext[0], ext[-1]], atol=1e-12)


def test_event_times_from_codes():
    np.testing.assert_array_equal(pr._event_times(_schedule()), [15, 125, 385])
    ev = pr._event_times(np.full(50, 2))  # starts airborne, no landing
    assert ev[0] == -np.inf and ev[1] == -np.inf and ev[2] == np.inf
    ev = pr._event_times(np.repeat([1, 2], [10, 20]))  # motors-on at the first sample
    np.testing.assert_array_equal(ev[:2], [0, 10])
    assert ev[2] == np.inf
    with pytest.raises(ValueError):
        pr._event_times(np.array([0, 2, 1, 3]))


def test_sampler_draws_exactly_the_likelihood_recursion():
    model = _model()
    P = model.params
    ph = _schedule()
    rng = np.random.default_rng(1)
    cur = pr._curves(pr._clock(ph), P)
    for q, a, measured in ((12, 0, True), (3, 1, True), (21, 1, False)):
        mus = np.stack([pr._mu(cur, P, j) for j in range(pr.N_GRID)])
        cls = model.graph.sample(ph, rng, 1)
        eps = rng.standard_normal((1, len(ph), 4))
        D = np.stack([pr._drive_table(P, b) for b in (0, 1)])
        S = np.stack([pr._sd_table(P, b, not measured) for b in (0, 1)])
        x = np.empty((1, len(ph), 4))
        pr._simulate(
            cls,
            np.array([q]),
            np.array([a]),
            mus,
            pr.REST,
            pr.CLS_REST,
            np.ascontiguousarray(P.phi[pr.CLS_STREAM]),
            D,
            S,
            eps,
            x,
        )
        Eb = pr._stream_residuals(x[0], mus[q], P)[pr.CLS_STREAM]
        t = np.arange(len(ph))
        e = (Eb[cls[0], t] - D[a][cls[0]]) / S[a][cls[0]]
        np.testing.assert_allclose(e, eps[0], rtol=1e-7, atol=1e-7)
        # the class log-emission of the drawn class is the innovation density
        table = pr._log_table(Eb, D[a], S[a], np.ones(len(ph), dtype=bool))
        want = (-0.5 * eps[0] ** 2 - np.log(S[a][cls[0]]) - 0.5 * np.log(2 * np.pi)).sum(axis=1)
        np.testing.assert_allclose(table[t, cls[0]], want, rtol=1e-9, atol=1e-8)


def test_loglik_is_exact_sum_over_nodes_activity_and_paths():
    model = _model()
    ph = _schedule()
    w = model.sample(ph, 1, np.random.default_rng(2))[0]
    f = pr._prepare(w, ph)
    P = model.params
    lpa = P.log_pi_act - logsumexp(P.log_pi_act)
    cur = pr._curves(f.clock, P)
    terms = []
    for q in range(pr.N_GRID):
        Eb = pr._stream_residuals(f.x, pr._mu(cur, P, q), P)[pr.CLS_STREAM]
        for a in (0, 1):
            tab = pr._log_table(Eb, pr._drive_table(P, a), pr._sd_table(P, a, False), f.obs)
            ll = model.graph.loglik(tab, f.phases)
            assert model.graph.forward_backward(tab, f.phases)[0] == pytest.approx(ll, rel=1e-12)
            terms.append(pr.NODE_LOGW[q] + lpa[a] + ll)
    exact = float(cast(float, logsumexp(terms))) + f.jac
    assert model.loglik(w, ph) == pytest.approx(exact, rel=1e-12)
    assert np.isfinite(f.jac)


def test_command_only_prefix_carries_no_factor():
    model = _model()
    ph = _schedule()
    w = model.sample(ph, 1, np.random.default_rng(3))[0]
    t_to = int(np.flatnonzero(ph == 2)[0])
    w2 = w.copy()
    w2[:, : t_to - 2] += 5.0  # reconstructed standby, never a lag of an airborne factor
    assert model.loglik(w2, ph, measured=False) == pytest.approx(
        model.loglik(w, ph, measured=False), rel=1e-12
    )
    assert model.loglik(w2, ph) != pytest.approx(model.loglik(w, ph), rel=1e-6)
    assert pr._prepare(w, ph, measured=False).obs.sum() == len(ph) - t_to


@pytest.mark.parametrize(
    "ph",
    [
        _schedule(),
        np.full(200, 2),
        np.repeat([0, 1, 2], [10, 60, 150]),
        np.repeat([2, 3], [120, 80]),
    ],
)
def test_samples_are_finite_nonnegative_and_the_readout_of_the_latent_law(ph):
    model = _model()
    for measured in (True, False):
        d = model.draw(ph, 4, np.random.default_rng(4), measured=measured)
        assert d.speed.shape == (4, 4, len(ph))
        assert np.isfinite(d.speed).all() and (d.speed >= 0).all()
        np.testing.assert_array_equal(d.speed, pr._readout(d.latent @ MIXER.T).transpose(0, 2, 1))
        assert 0.0 <= model.last_clip_fraction <= 1.0
    if ph[0] == 0:
        assert np.all(
            d.speed[:, :, : np.flatnonzero(ph > 0)[0]] < 1.0
        )  # quiet rest before motors-on


def test_parameter_budget_and_roundtrip():
    model = _model()
    bd = model.n_params_breakdown()
    assert model.n_params == sum(bd.values()) < 277
    assert bd["grammar"] == model.grammar.n_params
    d = json.loads(json.dumps(model.to_dict()))
    back = pr.ProgramResponseModel.from_dict(d)
    ph = _schedule()
    np.testing.assert_array_equal(
        back.sample(ph, 3, np.random.default_rng(5)), model.sample(ph, 3, np.random.default_rng(5))
    )
    assert back.n_params == model.n_params


def test_stable_qp_matches_grid_minimum_on_the_ar2_triangle():
    rng = np.random.default_rng(6)
    vert = pr._triangle()
    g = np.stack(np.meshgrid(np.linspace(-2, 2, 801), np.linspace(-1, 1, 401)), -1).reshape(-1, 2)
    inside = np.array([pr._inside(p, vert) for p in g])
    g = g[inside]
    for _ in range(5):
        M = rng.normal(size=(2, 2))
        A = M @ M.T + 0.1 * np.eye(2)
        b = rng.normal(size=2) * 3
        th = pr._stable_qp(A, b, np.zeros(2))
        assert pr._inside(th, vert)
        obj = lambda t, A=A, b=b: np.einsum("ni,ij,nj->n", t, A, t) - 2 * t @ b  # noqa: E731
        assert obj(th[None])[0] <= obj(g).min() + 1e-9


def test_small_fit_is_monotone_bounded_and_fits_the_hierarchy():
    truth = _model(idle_rps=30.0, hover_rps=80.0)
    rng = np.random.default_rng(7)
    tracks, measured = [], []
    for n1, n2, n3 in ((100, 240, 90), (140, 200, 110), (90, 280, 80)):
        ph = _schedule(10, n1, n2, n3)
        tracks.append((truth.sample(ph, 1, rng)[0], ph))
        measured.append(True)
    ph = np.repeat([0, 1, 2], [10, 60, 200])  # command-only record without a landing
    tracks.append((truth.sample(ph, 1, rng)[0], ph))
    measured.append(False)

    model = pr.fit(tracks, measured=measured, max_iter=3)
    h = np.array(model.history)
    assert len(h) >= 2 and np.isfinite(h).all()
    assert np.all(np.diff(h) >= -pr.MONOTONE_RTOL * np.abs(h[:-1]))
    P = model.params
    for s in range(len(pr.STREAMS)):
        for dd in range(4):
            assert pr._inside(P.phi[s, dd], pr._triangle())
    assert (P.sigma >= pr.SIGMA_FLOOR).all() and (P.kappa_act > 0).all() and (P.kappa_src > 0).all()
    assert P.rho[1] == 1.0
    assert model.n_params < 277
    info = model.fit_info
    assert info["n_command_only"] == 1 and info["n_records"] == 4
    assert "responsibilities" not in info and "flight_loglik" not in info
    json.dumps(model.to_dict())
    # fitted grammar stays a valid hierarchy: entry laws normalised per phase
    phases = np.array([m.phase for m in model.grammar.macros])
    for p in range(4):
        assert model.grammar.entry[phases == p].sum() == pytest.approx(1.0)
    # the hover anchor (common mode, link units ~ rev/s) is near the truth
    assert abs(P.anchor[1, 0] - truth.params.anchor[1, 0]) < 10.0
    out = model.sample(_schedule(), 2, np.random.default_rng(8))
    assert np.isfinite(out).all() and (out >= 0).all()
