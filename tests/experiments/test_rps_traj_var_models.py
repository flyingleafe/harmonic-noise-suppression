"""VAR, Markov-switching VAR and VAR(p)-HSMM cruise baselines (``experiments.rps_traj.var_models``).

The data these tests fit are simulated by independent loops written here, not
by the models' own samplers, so a bug shared by a sampler and a fit cannot
cancel out.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from experiments.rps_traj.cruise_api import CruiseModel
from experiments.rps_traj.var_models import (
    SwitchingVarModel,
    VarModel,
    fit_switching_var,
    fit_var,
    lasso_wls,
    msfe,
    select_regularization,
    shrink_covariance,
)

RATE = 25.0

A1 = np.array(
    [
        [0.50, 0.10, 0.00, 0.00],
        [0.00, 0.40, 0.10, 0.00],
        [0.05, 0.00, 0.30, 0.10],
        [0.00, 0.00, 0.00, 0.60],
    ]
)
A2 = np.array(
    [
        [-0.20, 0.00, 0.00, 0.05],
        [0.00, -0.10, 0.00, 0.00],
        [0.00, 0.10, 0.20, 0.00],
        [0.00, 0.00, 0.05, -0.30],
    ]
)
SIGMA = np.array(
    [
        [1.0, 0.3, 0.1, 0.0],
        [0.3, 2.0, 0.2, 0.1],
        [0.1, 0.2, 1.5, 0.3],
        [0.0, 0.1, 0.3, 0.8],
    ]
)


def _simulate(rng, states, mus, coefs, sigmas) -> list[np.ndarray]:
    """One ``(4, n)`` switching-VAR block per row of ``states (B, n)``, zero initial lags.

    Vectorised over blocks (a time loop only), independent of the module's sampler.
    """
    order = coefs.shape[1]
    n_blocks, n = states.shape
    chols = np.linalg.cholesky(sigmas)
    y = np.zeros((n_blocks, n + order, 4))
    eps = rng.standard_normal((n_blocks, n, 4))
    for t in range(n):
        s = states[:, t]
        acc = mus[s] + np.einsum("bij,bj->bi", chols[s], eps[:, t])
        for k in range(order):
            acc += np.einsum("bij,bj->bi", coefs[s, k], y[:, order + t - 1 - k])
        y[:, order + t] = acc
    return [np.ascontiguousarray(b[order:].T) for b in y]


def _draw(u, prob: np.ndarray):
    """Inverse-CDF draws: ``u`` scalar or ``(B,)`` with ``prob`` ``(K,)`` or ``(B, K)``."""
    cum = np.cumsum(prob, axis=-1)
    return np.minimum((np.asarray(u)[..., None] >= cum).sum(axis=-1), prob.shape[-1] - 1)


def _geometric_states(rng, n_blocks, n, delta, trans) -> np.ndarray:
    u = rng.random((n_blocks, n))
    s = np.empty((n_blocks, n), dtype=np.int64)
    s[:, 0] = _draw(u[:, 0], delta)
    for t in range(1, n):
        s[:, t] = _draw(u[:, t], trans[s[:, t - 1]])
    return s


def _explicit_states(rng, n_blocks, n, delta, q, pmf) -> np.ndarray:
    rows = []
    for _ in range(n_blocks):
        out: list[int] = []
        j = int(_draw(rng.random(), delta))
        while len(out) < n:
            out += [j] * (int(_draw(rng.random(), pmf[j])) + 1)
            j = int(_draw(rng.random(), q[j]))
        rows.append(out[:n])
    return np.array(rows, dtype=np.int64)


def _roundtrip(model):
    payload = json.loads(json.dumps(model.to_dict()))
    return type(model).from_dict(payload)


# ─── plain VAR ───────────────────────────────────────────────────────────────


def test_var_recovers_var2_bic_picks_two_and_round_trips():
    rng = np.random.default_rng(0)
    coefs = np.stack([A1, A2])[None]
    blocks = _simulate(
        rng, np.zeros((40, 1500), dtype=np.int64), np.zeros((1, 4)), coefs, SIGMA[None]
    )
    model = fit_var(blocks, RATE, order=None, p_max=6)
    assert isinstance(model, CruiseModel)
    assert model.order == 2
    assert model.n_params == 16 * 2 + 10
    assert not model.shrunk
    # 60k samples: coefficient standard errors are ~0.005
    np.testing.assert_allclose(model.coefs[0], A1, atol=0.025)
    np.testing.assert_allclose(model.coefs[1], A2, atol=0.025)
    np.testing.assert_allclose(model.sigma, SIGMA, atol=0.05)
    back = _roundtrip(model)
    np.testing.assert_array_equal(back.logpdf(blocks[0], 25), model.logpdf(blocks[0], 25))


def test_var_logpdf_and_msfe_equal_the_direct_formula():
    rng = np.random.default_rng(1)
    c = np.array([0.5, -1.0, 0.2, 0.0])
    model = VarModel(
        name="v",
        rate_hz=RATE,
        coefs=np.stack([A1, A2]),
        intercept=c,
        sigma=SIGMA,
        has_intercept=True,
    )
    assert model.n_params == 16 * 2 + 10 + 4
    y = rng.standard_normal((4, 7)) * 2.0
    burn = 3
    want, sq = [], []
    for t in range(burn, 7):
        mean = c + A1 @ y[:, t - 1] + A2 @ y[:, t - 2]
        want.append(multivariate_normal.logpdf(y[:, t], mean=mean, cov=SIGMA.tolist()))
        sq.append(((y[:, t] - mean) ** 2).sum())
    np.testing.assert_allclose(model.logpdf(y, burn), want, rtol=0, atol=1e-10)
    assert msfe(model, [y], burn) == pytest.approx(np.mean(sq), rel=1e-12)
    with pytest.raises(ValueError):
        model.logpdf(y, 1)


def test_var_sample_is_stationary_from_its_first_sample():
    model = VarModel(
        name="v",
        rate_hz=RATE,
        coefs=np.stack([A1, A2]),
        intercept=np.array([1.0, 0.0, -2.0, 0.5]),
        sigma=SIGMA,
        has_intercept=True,
    )
    cov = model.stationary_cov
    # Lyapunov: Gamma0 = A1 G0 A1' + A1 G1 A2' + ... ; check via the companion identity
    big = model._companion_cov()
    f = model.companion()
    q = np.zeros_like(big)
    q[:4, :4] = SIGMA
    np.testing.assert_allclose(f @ big @ f.T + q, big, atol=1e-10)
    rng = np.random.default_rng(2)
    y = model.sample(200_000, rng)
    np.testing.assert_allclose(y.mean(axis=1), model.stationary_mean, atol=0.05)
    scale = np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    assert np.abs(np.cov(y) - cov).max() / scale.max() < 0.03
    # the very first sample already has the stationary law (no burn-in transient)
    first = np.stack([model.sample(1, np.random.default_rng(100 + i))[:, 0] for i in range(4000)])
    np.testing.assert_allclose(first.mean(axis=0), model.stationary_mean, atol=0.12)
    assert np.abs(np.cov(first.T) - cov).max() / scale.max() < 0.12


def test_var_stability_guard_pulls_an_explosive_fit_under_the_bound():
    rng = np.random.default_rng(3)
    blocks = []
    for _ in range(10):
        y = np.zeros((4, 300))
        for t in range(1, 300):
            y[:, t] = 1.01 * y[:, t - 1] + rng.standard_normal(4)
        blocks.append(y)
    model = fit_var(blocks, RATE, order=1)
    assert model.shrunk
    assert 0.998 <= model.spectral_radius < 0.999


def test_var_forecast_lead_one_and_two_means_are_the_conditional_means():
    c = np.array([0.5, -1.0, 0.2, 0.0])
    model = VarModel(
        name="v",
        rate_hz=RATE,
        coefs=np.stack([A1, A2]),
        intercept=c,
        sigma=SIGMA,
        has_intercept=True,
    )
    rng = np.random.default_rng(4)
    hist = model.sample(50, rng)
    paths = model.forecast(hist, 2, 40_000, rng)
    m1 = c + A1 @ hist[:, -1] + A2 @ hist[:, -2]
    m2 = c + A1 @ m1 + A2 @ hist[:, -1]
    se = np.sqrt(np.diag(SIGMA) / 40_000)
    assert np.all(np.abs(paths[:, :, 0].mean(axis=0) - m1) < 4.5 * se)
    assert np.all(np.abs(paths[:, :, 1].mean(axis=0) - m2) < 4.5 * se * 2.0)
    np.testing.assert_allclose(np.cov(paths[:, :, 0].T), SIGMA, atol=0.05)


# ─── regularised estimators ──────────────────────────────────────────────────


def test_lasso_wls_zero_penalty_kkt_and_full_sparsity():
    rng = np.random.default_rng(5)
    n = 400
    x = np.column_stack([np.ones(n), rng.standard_normal((n, 6))])
    x[:, 2] += 0.8 * x[:, 1]  # correlated regressors
    beta = rng.standard_normal((7, 4))
    y = x @ beta + rng.standard_normal((n, 4))
    w = rng.uniform(0.0, 1.0, n)
    sw = np.sqrt(w)[:, None]
    wls = np.linalg.lstsq(x * sw, y * sw, rcond=None)[0]
    np.testing.assert_allclose(lasso_wls(x, y, w, 0.0, n_unpenalized=1), wls, atol=1e-10)
    lam = 40.0
    b = lasso_wls(x, y, w, lam, n_unpenalized=1)
    grad = 2.0 * ((x * w[:, None]).T @ (x @ b - y))  # d(weighted SSE)/db
    np.testing.assert_allclose(grad[0], 0.0, atol=1e-6)  # unpenalised intercept
    on = b[1:] != 0.0
    assert on.any() and (~on).any()
    np.testing.assert_allclose(grad[1:][on], -lam * np.sign(b[1:][on]), atol=1e-6)
    assert np.all(np.abs(grad[1:][~on]) <= lam + 1e-6)
    big = lasso_wls(x, y, w, 1e9, n_unpenalized=1)
    assert np.all(big[1:] == 0.0)
    np.testing.assert_allclose(big[0], (w[:, None] * y).sum(0) / w.sum(), atol=1e-10)


def test_shrink_covariance_keeps_trace_and_shrinks_eigen_spread():
    rng = np.random.default_rng(6)
    g = rng.standard_normal((4, 4))
    s = g @ g.T + 0.1 * np.eye(4)
    np.testing.assert_array_equal(shrink_covariance(s, 0.0), s)
    out = shrink_covariance(s, 2.0)
    assert np.trace(out) == pytest.approx(np.trace(s))
    ev, ev_out = np.linalg.eigvalsh(s), np.linalg.eigvalsh(out)
    np.testing.assert_allclose(ev_out, (ev + 2.0 * np.trace(s) / 4) / 3.0, rtol=1e-12)
    assert ev_out[-1] / ev_out[0] < ev[-1] / ev[0]
    np.testing.assert_allclose(shrink_covariance(s, 2.0, 0.5), out + 0.5 * np.eye(4))


# ─── switching models: likelihood mechanics ──────────────────────────────────


def _random_states(rng, m, order):
    mus = rng.standard_normal((m, 4))
    coefs = 0.3 * rng.standard_normal((m, order, 4, 4)) / np.sqrt(4 * order)
    g = rng.standard_normal((m, 4, 4))
    sigmas = g @ g.transpose(0, 2, 1) + 0.5 * np.eye(4)
    return mus, coefs, sigmas


def test_one_state_unpenalised_em_is_the_var_ols():
    rng = np.random.default_rng(7)
    coefs = np.stack([A1, A2])[None]
    mu = np.array([[1.0, -0.5, 0.0, 2.0]])
    blocks = _simulate(rng, np.zeros((8, 600), dtype=np.int64), mu, coefs, SIGMA[None])
    var = fit_var(blocks, RATE, order=2, intercept=True)
    sw = fit_switching_var(blocks, RATE, n_states=1, order=2, n_init=1)
    assert sw.converged_
    np.testing.assert_allclose(sw.coefs[0], var.coefs, atol=1e-9)
    np.testing.assert_allclose(sw.mu[0], var.intercept, atol=1e-9)
    np.testing.assert_allclose(sw.sigma[0], var.sigma, atol=1e-9)
    want = np.concatenate([var.logpdf(b, 2) for b in blocks]).sum()
    assert sw.loglik_ == pytest.approx(want, rel=1e-10)
    np.testing.assert_allclose(sw.logpdf(blocks[0], 5), var.logpdf(blocks[0], 5), atol=1e-9)


def _explicit_from_geometric(geo: SwitchingVarModel, d_max: int) -> SwitchingVarModel:
    stay = np.diag(geo.trans)
    q = geo.trans * (1.0 - np.eye(geo.n_states)) / (1.0 - stay)[:, None]
    n = np.arange(1, d_max + 1)
    pmf = (1.0 - stay)[:, None] * stay[:, None] ** (n - 1)
    pmf[:, -1] = stay ** (d_max - 1)
    return SwitchingVarModel(
        name="e",
        rate_hz=RATE,
        durations="explicit",
        mu=geo.mu,
        coefs=geo.coefs,
        sigma=geo.sigma,
        delta=geo.delta,
        trans=q,
        dur_pmf=pmf,
    )


@pytest.mark.parametrize("m", [2, 3])
def test_explicit_durations_reduce_to_the_geometric_model(m):
    rng = np.random.default_rng(8 + m)
    mus, coefs, sigmas = _random_states(rng, m, 2)
    delta = rng.dirichlet(np.ones(m))
    y = 2.0 * rng.standard_normal((4, 60))
    # (a) geometric dwell pmf with D past the block end: identical predictive law
    stay = rng.uniform(0.6, 0.95, m)
    off = rng.dirichlet(np.ones(m - 1), size=m)
    trans = np.diag(stay)
    for i in range(m):
        trans[i, np.arange(m) != i] = (1.0 - stay[i]) * off[i]
    geo = SwitchingVarModel("g", RATE, "geometric", mus, coefs, sigmas, delta, trans)
    exp = _explicit_from_geometric(geo, 60)
    np.testing.assert_allclose(exp.logpdf(y, 2), geo.logpdf(y, 2), rtol=0, atol=1e-10)
    np.testing.assert_allclose(exp.posterior(y), geo.posterior(y), atol=1e-10)
    # (b) D = 1: every visit lasts one sample, i.e. a chain with an empty diagonal
    q = trans * (1.0 - np.eye(m))
    q /= q.sum(axis=1, keepdims=True)
    geo0 = SwitchingVarModel("g", RATE, "geometric", mus, coefs, sigmas, delta, q)
    exp1 = SwitchingVarModel(
        "e", RATE, "explicit", mus, coefs, sigmas, delta, q, dur_pmf=np.ones((m, 1))
    )
    np.testing.assert_allclose(exp1.logpdf(y, 3), geo0.logpdf(y, 3), rtol=0, atol=1e-10)
    assert not np.allclose(exp1.logpdf(y, 3), geo.logpdf(y, 3))


def _brute_force(y, model: SwitchingVarModel):
    """Total likelihood and smoothed state probabilities over every segmentation."""
    p = model.order
    n_t = y.shape[1] - p
    m, d_max = model.n_states, model.max_dur
    dens = np.array(
        [
            [
                multivariate_normal(
                    model.mu[j] + sum(model.coefs[j, k] @ y[:, p + t - 1 - k] for k in range(p)),
                    model.sigma[j],
                ).pdf(y[:, p + t])
                for j in range(m)
            ]
            for t in range(n_t)
        ]
    )
    assert model.dur_pmf is not None
    pmf = model.dur_pmf
    surv = np.cumsum(pmf[:, ::-1], axis=1)[:, ::-1]
    total = 0.0
    post = np.zeros((m, n_t))

    def visits(t, j, prob, labels):
        nonlocal total
        left = n_t - t
        if left <= d_max:  # last visit, right-censored
            pr = prob * surv[j, left - 1] * np.prod(dens[t:, j])
            total += pr
            lab = labels + [j] * left
            post[lab, np.arange(n_t)] += pr
        for d in range(1, min(d_max, left - 1) + 1):
            pr = prob * pmf[j, d - 1] * np.prod(dens[t : t + d, j])
            for i in range(m):
                if model.trans[j, i] > 0:
                    visits(t + d, i, pr * model.trans[j, i], labels + [j] * d)

    for j in range(m):
        visits(0, j, model.delta[j], [])
    return total, post / total


@pytest.mark.parametrize("m", [2, 3])
def test_expanded_state_likelihood_and_posterior_equal_brute_force(m):
    rng = np.random.default_rng(20 + m)
    mus, coefs, sigmas = _random_states(rng, m, 1)
    q = rng.uniform(0.2, 1.0, (m, m)) * (1.0 - np.eye(m))
    q /= q.sum(axis=1, keepdims=True)
    model = SwitchingVarModel(
        "e",
        RATE,
        "explicit",
        mus,
        coefs,
        sigmas,
        rng.dirichlet(np.ones(m)),
        q,
        dur_pmf=rng.dirichlet(np.ones(3), size=m),
    )
    assert model.n_params == (m - 1) + m * 2 + m * (m - 2) + 4 * m + 10 * m + 16 * m
    y = rng.standard_normal((4, 8))
    total, post = _brute_force(y, model)
    assert model.logpdf(y, 1).sum() == pytest.approx(np.log(total), abs=1e-10)
    np.testing.assert_allclose(model.posterior(y), post, atol=1e-10)


# ─── switching models: EM on simulated regimes ───────────────────────────────

SCALES = np.array([1.0, 3.0])


def _regime_params():
    coefs = np.stack([0.5 * np.eye(4), 0.3 * np.eye(4)])[:, None]
    sigmas = np.stack([s**2 * SIGMA for s in SCALES])
    return np.zeros((2, 4)), coefs, sigmas


def _assert_monotone(trace):
    tr = np.asarray(trace)
    assert len(tr) > 3
    assert np.all(np.diff(tr) >= -1e-9 * np.abs(tr[:-1]))


def test_geometric_em_recovers_scales_dwells_and_beats_a_single_var():
    rng = np.random.default_rng(30)
    mus, coefs, sigmas = _regime_params()
    trans = np.array([[0.96, 0.04], [0.05, 0.95]])  # dwells 1.0 s and 0.8 s at 25 Hz
    delta = np.array([0.5, 0.5])

    def make(n_blocks):
        return _simulate(
            rng, _geometric_states(rng, n_blocks, 750, delta, trans), mus, coefs, sigmas
        )

    train, test = make(40), make(10)
    model = fit_switching_var(train, RATE, n_states=2, order=1, n_init=2, max_iter=150)
    assert isinstance(model, CruiseModel)
    assert model.n_params == 1 + 2 + 8 + 20 + 32
    _assert_monotone(model.loglik_trace_)
    lo, hi = np.argsort(np.trace(model.sigma, axis1=1, axis2=2))
    ratio = np.sqrt(np.trace(model.sigma[hi]) / np.trace(model.sigma[lo]))
    assert ratio == pytest.approx(3.0, rel=0.08)
    np.testing.assert_allclose(model.mean_dwell_s[[lo, hi]], [1.0, 0.8], rtol=0.2)
    occ = 0.05 / 0.09
    np.testing.assert_allclose(model.stationary_occupancy[[lo, hi]], [occ, 1 - occ], atol=0.05)
    var = fit_var(train, RATE, order=1)
    lp_sw = np.concatenate([model.logpdf(b, 25) for b in test]).mean()
    lp_var = np.concatenate([var.logpdf(b, 25) for b in test]).mean()
    assert lp_sw > lp_var + 0.1
    back = _roundtrip(model)
    np.testing.assert_array_equal(back.logpdf(test[0], 25), model.logpdf(test[0], 25))
    assert back.loglik_trace_ == model.loglik_trace_


def test_explicit_em_finds_the_mode_of_a_peaked_dwell_law():
    rng = np.random.default_rng(31)
    mus, coefs, sigmas = _regime_params()
    pmf = np.zeros((2, 40))
    pmf[:, [17, 18, 19, 20, 21]] = [0.05, 0.2, 0.5, 0.2, 0.05]  # mode at 20 samples
    q = np.array([[0.0, 1.0], [1.0, 0.0]])
    delta = np.array([0.5, 0.5])
    blocks = _simulate(rng, _explicit_states(rng, 30, 750, delta, q, pmf), mus, coefs, sigmas)
    model = fit_switching_var(
        blocks, RATE, durations="explicit", max_dur_s=1.6, n_init=1, max_iter=100
    )
    assert model.max_dur == 40
    assert model.n_params == 1 + 2 * 39 + 0 + 8 + 20 + 32
    _assert_monotone(model.loglik_trace_)
    assert model.dur_pmf is not None
    np.testing.assert_array_equal(model.dur_pmf.argmax(axis=1) + 1, [20, 20])
    np.testing.assert_allclose(model.mean_dwell_s, 20 / RATE, rtol=0.05)
    back = _roundtrip(model)
    np.testing.assert_array_equal(back.logpdf(blocks[0], 25), model.logpdf(blocks[0], 25))


def test_switching_sample_follows_the_dwell_law_and_occupancy():
    pmf = np.zeros((2, 40))
    pmf[0, 9] = 1.0  # state 0 lasts exactly 10 samples
    pmf[1, [19, 39]] = 0.5  # state 1 lasts 20 or 40 (= D) samples
    model = SwitchingVarModel(
        "e",
        RATE,
        "explicit",
        mu=np.zeros((2, 4)),
        coefs=np.zeros((2, 1, 4, 4)),
        sigma=np.stack([1e-12 * np.eye(4), np.eye(4)]),
        delta=np.array([1.0, 0.0]),
        trans=np.array([[0.0, 1.0], [1.0, 0.0]]),
        dur_pmf=pmf,
    )
    np.testing.assert_allclose(model.stationary_occupancy, [10 / 40, 30 / 40])
    y = model.sample(60_000, np.random.default_rng(32))
    quiet = np.abs(y).max(axis=0) < 1e-3
    edges = np.flatnonzero(np.diff(quiet.astype(int))) + 1
    runs = np.diff(edges)  # complete runs only
    kinds = quiet[edges[:-1]]
    assert set(runs[kinds]) == {10}
    loud = runs[~kinds]
    assert set(loud) == {20, 40}
    assert np.mean(loud == 40) == pytest.approx(0.5, abs=0.05)
    assert quiet.mean() == pytest.approx(0.25, abs=0.01)
    start = model.sample(15, np.random.default_rng(33), start="initial")
    assert np.all(np.abs(start[:, :10]) < 1e-3) and np.all(np.abs(start[:, 10:]).max(0) > 1e-3)


def test_switching_sample_clip_holds_an_explosive_state_and_lets_it_come_back():
    # state 0 doubles every step (a "ramp" state), state 1 decays slowly; they alternate every 10
    pmf = np.zeros((2, 10))
    pmf[:, 9] = 1.0
    model = SwitchingVarModel(
        "ramp",
        RATE,
        "explicit",
        mu=np.array([[1.0] * 4, [0.0] * 4]),
        coefs=np.stack([2.0 * np.eye(4), 0.9 * np.eye(4)])[:, None],
        sigma=np.stack([1e-6 * np.eye(4)] * 2),
        delta=np.array([1.0, 0.0]),
        trans=np.array([[0.0, 1.0], [1.0, 0.0]]),
        dur_pmf=pmf,
    )
    free = model.sample(400, np.random.default_rng(36), start="initial")
    assert free.max() > 1e6  # x 2^10 up, x 0.9^10 down, every cycle: it leaves any range
    held = model.sample(400, np.random.default_rng(36), start="initial", clip=(0.0, 50.0))
    assert held.min() >= 0.0 and held.max() == pytest.approx(50.0)
    # the cap is fed back: ten decaying steps start from 50, not from the free value
    np.testing.assert_allclose(held[:, 19], 50.0 * 0.9**10, rtol=1e-2)


def test_stable_fit_keeps_a_ramp_state_under_the_unit_circle():
    rng = np.random.default_rng(40)
    blocks, labels = [], []
    for _ in range(30):  # quiet noise, then every rotor growing 8 % a step from its own start
        quiet = 0.05 * rng.standard_normal((4, 40))
        ramp = rng.uniform(0.5, 1.5, (4, 1)) * 1.08 ** np.arange(40)[None, :]
        blocks.append(np.hstack([quiet, ramp + 0.01 * rng.standard_normal((4, 40))]))
        labels.append(np.r_[np.zeros(40, dtype=np.int64), np.ones(40, dtype=np.int64)])

    def fit(stable: bool) -> SwitchingVarModel:
        return fit_switching_var(
            blocks,
            RATE,
            n_states=2,
            order=1,
            init_labels=labels,
            n_init=1,
            max_iter=20,
            stable=stable,
        )

    def radius(model: SwitchingVarModel) -> float:
        return max(float(np.abs(np.linalg.eigvals(a[0])).max()) for a in model.coefs)

    free, held = fit(False), fit(True)
    assert radius(free) == pytest.approx(1.08, abs=0.02)
    assert radius(held) < 1.0
    # the shrunk state still climbs (its intercept was re-estimated), within reach of the data
    y = held.sample(4000, np.random.default_rng(41))
    top = max(float(b.max()) for b in blocks)
    assert 0.2 * top < y.max() < 30.0 * top


@pytest.mark.parametrize("durations", ["geometric", "explicit"])
def test_switching_forecast_lead_one_mean_is_the_filtered_predictive_mean(durations):
    rng = np.random.default_rng(34)
    mus = np.array([[5.0, 5.0, 5.0, 5.0], [-5.0, -5.0, -5.0, -5.0]])
    coefs = np.stack([0.2 * np.eye(4), -0.1 * np.eye(4)])[:, None]
    sigmas = np.stack([np.eye(4), np.eye(4)])
    if durations == "geometric":
        model = SwitchingVarModel(
            "g",
            RATE,
            "geometric",
            mus,
            coefs,
            sigmas,
            np.array([0.5, 0.5]),
            np.array([[0.7, 0.3], [0.4, 0.6]]),
        )
    else:
        model = SwitchingVarModel(
            "e",
            RATE,
            "explicit",
            mus,
            coefs,
            sigmas,
            np.array([0.5, 0.5]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            dur_pmf=np.array([[0.0, 0.2, 0.8], [0.6, 0.0, 0.4]]),
        )
    hist = model.sample(40, rng, start="initial")
    nxt = model.sample(41, np.random.default_rng(35), start="initial")[:, -1:]
    want = model.predict_mean(np.hstack([hist, nxt]), 40)[:, 0]
    paths = model.forecast(hist, 1, 50_000, rng)[:, :, 0]
    se = paths.std(axis=0) / np.sqrt(50_000)
    assert np.all(np.abs(paths.mean(axis=0) - want) < 4.5 * se)
    # the predictive mean differs from the filtered-state (no transition) mean
    if durations == "geometric":
        filt = model.posterior(hist)[:, -1]
        no_step = filt @ (mus + np.einsum("jab,b->ja", coefs[:, 0], hist[:, -1]))
        assert np.abs(no_step - want).max() > 0.5


def test_lasso_zeroes_ar_and_selection_minimises_validation_msfe():
    rng = np.random.default_rng(36)
    mus, coefs, sigmas = _regime_params()
    trans = np.array([[0.95, 0.05], [0.05, 0.95]])
    delta = np.array([0.5, 0.5])
    blocks = _simulate(rng, _geometric_states(rng, 16, 300, delta, trans), mus, coefs, sigmas)
    sparse = fit_switching_var(blocks[:12], RATE, lam_a=1e9, n_init=1, max_iter=30)
    assert np.all(sparse.coefs == 0.0)
    best, records = select_regularization(
        blocks[:12],
        blocks[12:],
        RATE,
        grid_a=[0.0, 1e9],
        grid_sigma=[0.0, 1.0],
        n_init=1,
        max_iter=30,
    )
    assert len(records) == 4
    top = min(records, key=lambda r: r["msfe"])
    assert (best.lam_a, best.lam_sigma) == (top["lam_a"], top["lam_sigma"])
    assert top["lam_a"] == 0.0  # the AR dynamics are real: zeroing them must cost MSFE
    assert msfe(best, blocks[12:], 25) == pytest.approx(top["msfe"])
    spreads = []
    for ls in (0.0, 1.0):
        fit = fit_switching_var(blocks[:12], RATE, lam_sigma=ls, n_init=1, max_iter=30)
        ev = np.linalg.eigvalsh(fit.sigma)
        spreads.append((ev[:, -1] / ev[:, 0]).max())
    assert spreads[1] < spreads[0]


def test_switching_burn_only_drops_scores_and_is_bounded_by_the_order():
    rng = np.random.default_rng(37)
    mus, coefs, sigmas = _random_states(rng, 2, 3)
    model = SwitchingVarModel(
        "g",
        RATE,
        "geometric",
        mus,
        coefs,
        sigmas,
        np.array([0.3, 0.7]),
        np.array([[0.9, 0.1], [0.2, 0.8]]),
    )
    y = rng.standard_normal((4, 30))
    full = model.logpdf(y, 3)
    np.testing.assert_array_equal(model.logpdf(y, 11), full[8:])
    np.testing.assert_allclose(model.predict_mean(y, 11), model.predict_mean(y, 3)[:, 8:])
    # the first scored density is the delta-weighted mixture at t = order
    first = [
        multivariate_normal(
            mus[j] + sum(coefs[j, k] @ y[:, 2 - k] for k in range(3)), sigmas[j]
        ).pdf(y[:, 3])
        for j in range(2)
    ]
    assert full[0] == pytest.approx(np.log(0.3 * first[0] + 0.7 * first[1]), abs=1e-12)
    with pytest.raises(ValueError):
        model.logpdf(y, 2)
    with pytest.raises(ValueError):
        fit_switching_var([y], RATE, n_states=1, durations="explicit")
