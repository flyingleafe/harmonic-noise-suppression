"""Clock-locked functional mixed-effects AR law: exact marginal likelihood
against a dense Gaussian, monotone ECM on three short synthetic flights,
sampling/serialization contract, and the stability QP."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.signal import lfilter
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from experiments.rps_traj import clock_mixed as cm
from tracking.rotors import MIXER


def _ramp(t, a, b):
    u = np.clip((t - a) / (b - a), 0.0, 1.0)
    return u * u * (3 - 2 * u)


def _flight(rng, idle, hover, parts=(25, 150, 450, 200), trim=(0.5, -0.3, 2.0), activity=1.0):
    codes = np.repeat(np.arange(4), parts).astype(np.int8)
    T = len(codes)
    t = np.arange(T) / cm.FS
    on, to, la = np.cumsum(parts)[:3] / cm.FS
    common = (
        idle * _ramp(t, on, on + 1.0)
        + (hover - idle) * _ramp(t, to, to + 1.5)
        - hover * _ramp(t, la - 0.3, la + 2.0)
    )
    powered = _ramp(t, on, on + 1.0) * (1 - _ramp(t, la - 0.3, la + 2.0))
    flying = (codes == 2).astype(float)
    modes = np.zeros((4, T))
    modes[0] = common
    for m in range(1, 4):
        modes[m] = trim[m - 1] * powered
    for m in range(4):
        eps = rng.standard_normal(T) * (0.05 + activity * (0.8 if m else 1.0) * flying)
        modes[m] += lfilter([1.0], [1.0, -0.6, 0.2], eps)
    return np.maximum(MIXER @ modes, 0.0), codes


@pytest.fixture(scope="module")
def tracks():
    rng = np.random.default_rng(3)
    return [
        _flight(rng, 30.0, 75.0, activity=0.6),
        _flight(rng, 32.0, 80.0, trim=(1.0, 0.2, 2.5), activity=1.5),
        _flight(rng, 31.0, 78.0, parts=(20, 120, 400, 150), trim=(0.2, -0.6, 1.5)),
    ]


@pytest.fixture(scope="module")
def model(tracks):
    return cm.fit(tracks, measured=[True, True, False], max_iter=12)


def test_transform_floor_roundtrip_and_jacobian():
    w = np.array([0.0, 0.1, cm.W_FLOOR, 0.3, 2.0, 80.0])
    y, lj = cm.to_obs(w)
    assert np.isfinite(y).all() and np.isfinite(lj).all()
    assert y[0] == y[1] == y[2] and y[2] > -2.0  # floor maps to one finite value
    back = cm.from_obs(y)
    np.testing.assert_allclose(back[3:], w[3:], rtol=1e-12)
    assert (back[:3] == 0.0).all()
    h = 1e-6
    num = (cm.to_obs(w[3:] + h)[0] - cm.to_obs(w[3:] - h)[0]) / (2 * h)
    np.testing.assert_allclose(num, np.exp(lj[3:]), rtol=1e-6)


def test_marginal_loglik_matches_dense_gaussian(model, tracks):
    w, codes = tracks[2]
    w, codes = (
        w[:, :260],
        codes[:260],
    )  # short enough for a dense covariance, still crosses take-off
    p, lay = model.params, model.layout
    y, lj = cm.to_obs(w)
    z = MIXER.T @ y / 4
    blocks, groups, _ = cm._blocks(codes, measured=False)
    mean_b, Psi = cm._psi(p, lay)
    T = len(codes)
    per_k = []
    for k in range(model.n_classes):
        # joint over modes: random effects are correlated across modes through the factors
        Xb, Cs = [], []
        for L in lay.modes:
            X, C = cm._columns(L, blocks, p.theta)
            Xb.append(X @ p.theta[L.beta])
            Cs.append(C)
        q = lay.q
        Cfull = np.zeros((4 * T, q))
        noise = np.zeros((4 * T, 4 * T))
        mean = np.zeros(4 * T)
        for m, L in enumerate(lay.modes):
            Cfull[m * T : (m + 1) * T, L.eff] = Cs[m]
            mean[m * T : (m + 1) * T] = Xb[m]
            A = np.eye(T) - p.phi[m, 0] * np.eye(T, k=-1) - p.phi[m, 1] * np.eye(T, k=-2)
            Ai = np.linalg.inv(A)
            s2 = p.sig2[cm._rows(k), m][groups]
            noise[m * T : (m + 1) * T, m * T : (m + 1) * T] = Ai @ np.diag(s2) @ Ai.T
        cov = noise + Cfull @ Psi @ Cfull.T
        per_k.append(
            # scipy's frozen-normal __call__ is unannotated: pyright types cov by its default 1.
            multivariate_normal(mean + Cfull @ mean_b, cov).logpdf(z.reshape(-1))  # pyright: ignore[reportArgumentType]
            + p.log_pi[k]
        )
    dense = logsumexp(per_k) + lj.sum() + T * cm.LOG_DET_MODES
    fast = model.loglik([(w, codes)], measured=[False])[0]
    assert np.isclose(fast, dense, rtol=1e-9, atol=1e-5)


def test_fit_is_monotone_and_counts_parameters(model):
    h = np.asarray(model.history)
    assert len(h) >= 2 and np.isfinite(h).all()
    assert (np.diff(h) >= -cm.MONOTONE_RTOL * np.abs(h[1:])).all()
    assert model.fit_info["stop_reason"] in ("converged", "max_iter")
    assert model.n_params == sum(model.param_breakdown().values()) < 277
    p = model.params
    assert (p.D > 0).all() and (p.sig2 >= cm.SIGMA_FLOOR**2).all()
    V = cm._stable_vertices()
    assert all(cm._inside(phi, V) for phi in p.phi)
    s = model.summary()
    assert set(s["ar"]) == set(cm.MODE_NAMES) and "startup_cmd" in s["noise_sd"]


def test_sample_contract_and_serialization(model):
    codes = np.repeat(np.arange(4), [30, 100, 300, 120]).astype(np.int8)
    W = model.sample(codes, 5, np.random.default_rng(0))
    assert W.shape == (5, 4, len(codes)) and np.isfinite(W).all() and (W >= 0).all()
    assert np.median(W[:, :, 250:300].mean(axis=1)) > 40.0  # cruise is airborne, not idle or rest
    assert 0.0 <= model.last_clip_fraction <= 1.0
    blob = json.loads(json.dumps(model.to_dict()))
    again = cm.ClockMixedModel.from_dict(blob)
    assert again.n_params == model.n_params
    np.testing.assert_array_equal(again.sample(codes, 5, np.random.default_rng(0)), W)
    airborne = np.full(200, 2, dtype=np.int8)  # starts airborne, never lands
    assert model.sample(airborne, 2, np.random.default_rng(1)).shape == (2, 4, 200)


def test_ar2_qp_is_constrained_minimum():
    rng = np.random.default_rng(1)
    V = cm._stable_vertices()
    grid = np.array([(a, b) for a in np.linspace(-2, 2, 161) for b in np.linspace(-1, 1, 81)])
    grid = grid[[cm._inside(x, V) for x in grid]]
    for _ in range(6):
        L = rng.normal(size=(3, 3))
        K = L @ L.T
        phi = cm._ar2_qp(K)
        assert cm._inside(phi, V)
        f = lambda x, K=K: np.r_[1.0, -x] @ K @ np.r_[1.0, -x]  # noqa: E731
        assert f(phi) <= min(f(x) for x in grid) + 1e-9
