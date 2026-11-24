"""Unified whole-flight response likelihood: parity on small known inputs,
anchor persistence of the generator, and a tiny end-to-end fit."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import norm

from experiments.rps_traj import unified_grammar as ug
from experiments.rps_traj import unified_response as ur
from tracking.rotors import MIXER

META = ur._label_meta()
E = META.n


def _schedule(n0: int = 25, n1: int = 150, n2: int = 300, n3: int = 150) -> np.ndarray:
    return np.repeat(np.arange(4), [n0, n1, n2, n3]).astype(np.int64)


def _params(*, sigma: float = 0.3, offsets: bool = True) -> ur.ResponseParams:
    offset = np.zeros((E, 4))
    if offsets:
        common = {
            "predrive": -8.0,
            "pulse": 15.0,
            "release": -4.0,
            "rise": 10.0,
            "overshoot": 5.0,
            "drive_up": 6.0,
            "drive_down": -6.0,
            "descend": -15.0,
            "shelf": 5.0,
        }
        for name, value in common.items():
            offset[ug.LABEL_ID[name], 0] = value
        offset[META.free, 1] = 0.5
    return ur.ResponseParams(
        F=np.array([0.4, 0.3, 0.3, 0.2]),
        K=np.array([0.25, 0.3, 0.3, 0.2]),
        idle=np.array([[20.0, 0.0, 0.5, 0.0], [40.0, 1.0, 0.0, 0.0]]),
        hover=np.array([[80.0, 0.0, 1.0, 0.0], [120.0, 0.0, 0.0, 1.0]]),
        offset=offset,
        sigma=np.full((E, 4), sigma),
        log_pi=np.log([0.5, 0.5]),
    )


def _model(params: ur.ResponseParams, hierarchical: bool = True) -> ur.UnifiedResponseModel:
    return ur.UnifiedResponseModel(
        ug.build_grammar(hierarchical), params, hierarchical=hierarchical
    )


def test_stable_qp_matches_grid_minimum_and_stays_stable():
    rng = np.random.default_rng(0)
    vert = ur._stable_vertices()
    fs = np.linspace(vert[0, 0], vert[1, 0], 301)
    grid = np.array(
        [
            (f, k)
            for f in fs
            for k in np.linspace(ur.STABILITY_MARGIN, 2 * (1 + f) * (1 - ur.STABILITY_MARGIN), 151)
        ]
    )
    for _ in range(6):
        L = rng.normal(size=(2, 2))
        A = L @ L.T + 0.1 * np.eye(2)
        b = rng.normal(scale=3.0, size=2)
        theta = ur._stable_qp(A, b, np.array([0.5, 0.1]))
        F, K = theta
        assert abs(F) < 1 and 0 < K < 2 * (1 + F)
        assert np.max(np.abs(np.roots([1.0, -(1 + F - K), F]))) < 1

        def obj(t: np.ndarray, A: np.ndarray = A, b: np.ndarray = b) -> np.ndarray:
            return np.einsum("ni,ij,nj->n", t, A, t) - 2 * t @ b

        assert obj(theta[None])[0] <= obj(grid).min() + 1e-9


def test_log_emissions_match_bruteforce_gaussian_ar2():
    rng = np.random.default_rng(1)
    T = 9
    w = 50 + rng.normal(scale=5.0, size=(4, T))
    w[2, 5] = np.nan  # removes the factors of t = 5, 6, 7
    phases = np.array([0, 1, 1, 1, 2, 2, 2, 3, 3])
    f = ur._prepare(w, phases)
    p = _params()
    targets = p.targets(META)
    x = ((MIXER.T @ w) / 4).T
    for j in range(2):
        got = ur._log_emissions(ur._drive(f, p), f.valid, targets[:, j], p.K, p.sigma)
        want = np.zeros((T, E))
        for t in range(2, T):
            if not np.isfinite(x[t - 2 : t + 1]).all():
                continue
            mean = x[t - 1] + p.F * (x[t - 1] - x[t - 2]) + p.K * (targets[:, j] - x[t - 1])
            want[t] = norm.logpdf(x[t], mean, p.sigma).sum(axis=1)
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-8)
    assert list(np.flatnonzero(~f.valid)) == [0, 1, 5, 6, 7]


def test_moment_sse_matches_direct_residuals():
    rng = np.random.default_rng(2)
    T = 40
    w = 60 + np.cumsum(rng.normal(size=(4, T)), axis=1)
    f = ur._prepare(w, np.full(T, 2))
    p = _params()
    W = rng.random((T, E))
    m = p.targets(META)[:, 1]  # (E, 4)
    got = ur._sse(ur._moments(W, ur._features(f)), p.F, p.K, m)
    a = ur._drive(f, p)
    want = np.einsum("te,ted->ed", W * f.valid[:, None], (a[:, None, :] - p.K * m[None]) ** 2)
    np.testing.assert_allclose(got, want, rtol=1e-8)


@pytest.mark.parametrize("split", [False, True])
def test_flight_loglik_is_exact_anchor_mixture_of_graph_likelihoods(split):
    rng = np.random.default_rng(3)
    p = _params()
    if split:
        p = replace(p, F=np.stack((p.F, 0.5 * p.F)), K=np.stack((p.K, 1.2 * p.K)))
    selectors = [
        int(split and name not in ("hold", "drive_up", "drive_down")) for name in META.names
    ]
    F = np.atleast_2d(p.F)[selectors]
    K = np.atleast_2d(p.K)[selectors]
    model = _model(p)
    phases = _schedule(10, 40, 60, 30)
    w = model.draw(phases, 1, rng).raw[0]
    x = ((MIXER.T @ w) / 4).T
    targets = p.targets(META)
    lls = []
    for j in range(2):
        le = np.zeros((len(phases), E))
        for t in range(2, len(phases)):
            mean = x[t - 1] + F * (x[t - 1] - x[t - 2]) + K * (targets[:, j] - x[t - 1])
            le[t] = norm.logpdf(x[t], mean, p.sigma).sum(axis=1)
        lls.append(model.grammar.loglik(le, phases))
    assert model.loglik(w, phases) == pytest.approx(logsumexp(p.log_pi + np.array(lls)), rel=1e-12)


def test_generator_keeps_one_anchor_pair_for_the_whole_flight():
    # Zero offsets: every startup label targets idle_j, every airborne label
    # hover_j, so the settled levels reveal the flight's anchor pair.
    p = _params(sigma=0.01, offsets=False)
    model = _model(p)
    phases = _schedule()
    end1 = np.flatnonzero(phases == 1)[-1]
    end2 = np.flatnonzero(phases == 2)[-1]
    d = model.draw(phases, 24, np.random.default_rng(4))
    w, comp, labels = d.raw, d.anchor, d.labels
    assert set(comp.tolist()) == {0, 1}
    common = w.mean(axis=1)
    for i, j in enumerate(comp):
        assert common[i, end1] == pytest.approx(p.idle[j, 0], abs=0.5)
        assert common[i, end2] == pytest.approx(p.hover[j, 0], abs=0.5)
    assert np.all(np.isin(labels[:, phases == 0], ug.PHASE_LABELS[0]))
    assert np.abs(w[:, :, phases == 0]).max() < 0.2  # latent rest stays at zero


def test_draws_follow_the_likelihood_recursion_without_feedback_clipping():
    # Generativity parity: standardised one-step residuals of the signed draws,
    # under the drawn anchor and labels, are N(0, 1) even where they go negative.
    p = _params(sigma=0.5)
    model = _model(p)
    phases = _schedule(40, 100, 150, 100)
    d = model.draw(phases, 8, np.random.default_rng(8))
    targets = p.targets(META)
    z = []
    for i in range(len(d.anchor)):
        x = ((MIXER.T @ d.raw[i]) / 4).T
        m = targets[d.labels[i], d.anchor[i]]  # (T, 4)
        v = np.diff(x, axis=0)
        r = v[1:] - p.F * v[:-1] - p.K * (m[2:] - x[1:-1])
        z.append(r / p.sigma[d.labels[i, 2:]])
    z = np.concatenate(z)
    assert abs(z.mean()) < 0.03 and abs(z.std() - 1.0) < 0.03
    assert d.negative_fraction > 0  # rest/rundown dips below zero and is kept ...
    assert model.last_clip_fraction == d.negative_fraction
    neg_times = (d.raw < 0).any(axis=1)
    assert d.negative_fraction == pytest.approx(neg_times.mean())
    np.testing.assert_array_equal(d.shaft, np.maximum(d.raw, 0.0))  # ... the shaft view is separate


def test_sample_shape_bounds_and_serialisation_roundtrip():
    model = _model(_params())
    phases = _schedule(5, 50, 80, 40)
    a = model.sample(phases, 3, np.random.default_rng(5))
    assert a.shape == (3, 4, len(phases)) and np.isfinite(a).all()
    shaft = model.sample(phases, 3, np.random.default_rng(5), shaft_only=True)
    np.testing.assert_array_equal(shaft, np.maximum(a, 0.0))  # same draws, readout only
    d = json.loads(json.dumps(model.to_dict()))
    back = ur.UnifiedResponseModel.from_dict(d)
    np.testing.assert_array_equal(back.sample(phases, 3, np.random.default_rng(5)), a)
    assert back.n_params == model.n_params
    bd = model.n_params_breakdown()
    assert bd["offsets"] == 4 * 9 and bd["disturbance_sd"] == 4 * 13 and bd["anchors"] == 16


def test_small_fit_is_monotone_finite_and_recovers_anchor_partition():
    truth = _params(sigma=0.4)
    gen = _model(truth)
    # A plateau anchor is identifiable only when its fixed-offset action is
    # actually visited. Include a long idle rather than fitting only spin-ups.
    phases = _schedule(15, 400, 300, 100)
    rng = np.random.default_rng(6)
    d = gen.draw(phases, 6, rng)
    w, comp = d.raw, d.anchor
    assert len(set(comp.tolist())) == 2
    assert all(np.count_nonzero(d.labels[i] == ug.LABEL_ID["idle"]) >= 15 for i in range(len(comp)))
    tracks = [(w[i].copy(), phases) for i in range(6)]
    tracks[0][0][:, 200:203] = np.nan  # a gap is conditioned over, not fatal
    model = ur.fit(tracks, hierarchical=True, max_iter=12)
    h = np.array(model.history)
    assert np.isfinite(h).all() and len(h) >= 2
    assert np.all(np.diff(h) >= -ur.MONOTONE_RTOL * np.abs(h[:-1]))
    info = model.fit_info
    assert info["stop_reason"] in {"converged", "max_iter"}
    assert max(info["spectral_radius"]) < 1
    assert np.all(model.params.sigma >= ur.SIGMA_FLOOR)
    resp = np.array(info["responsibilities"])
    hard = resp.argmax(axis=1)
    assert (hard == comp).all() or (hard == 1 - comp).all()
    order = np.argsort(model.params.hover[:, 0])
    np.testing.assert_allclose(model.params.hover[order, 0], truth.hover[:, 0], atol=3.0)
    np.testing.assert_allclose(model.params.idle[order, 0], truth.idle[:, 0], atol=3.0)
    assert isinstance(model.n_params, int)
    assert model.n_params == 8 + 16 + 36 + 52 + 1 + model.grammar.n_params
    json.dumps(model.to_dict())
    flat = ur.fit(tracks, hierarchical=False, max_iter=2)
    assert flat.sample(phases, 2, np.random.default_rng(7)).shape == (2, 4, len(phases))
