"""The new rotor-speed model must mean what it says.

The model is defined by the exact discrete state space of its two components,
and both the likelihood and the sampler are derived from that one object, so
these tests pin the places where the two could silently disagree: the spectrum
against the discrete stationary variance, the sampled path against the state
recursion it claims to realise, and the fit against parameters it generated
itself.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import torch
from scipy.linalg import solve_discrete_lyapunov

from experiments.rps_traj.data import RATE_HZ, Flight
from experiments.rps_traj.model import (
    F0_MAX_HZ,
    F0_MIN_HZ,
    TAU_SLOW_MAX_S,
    ZETA_MAX,
    ZETA_MIN,
    NewFit,
    Params,
    _Batch,
    _batch_nll,
    _car2_path,
    _fit_vector,
    _Predictor,
    _psd_matrix_sqrt,
    _state_space_torch,
    _unpack,
    car2_psd_grid,
    car2_state_space,
    corner_tau_s,
    cross_periodogram,
    fit_rate_hz,
    fit_rig,
    likelihood_batches,
    ou_psd_grid,
    ou_state_space,
    state_space_psd,
    steady_state_gain,
    tau_slow_from_u,
)
from experiments.rps_traj.posterior import (
    Posterior,
    fit_posterior,
    params_from_rig_vector,
    rig_vector,
)


def _params(**over: object) -> Params:
    base: dict[str, object] = {
        "mu": [90.0, 75.0, 82.0, 76.0],
        "theta": 0.3,
        "tau_slow": [1.0, 0.7, 0.7, 1.2],
        "sigma_slow": [3.0, 1.0, 1.0, 2.0],
        "f0": [1.0, 3.0, 2.0, 1.5],
        "zeta": [1.0, 0.4, 1.5, 0.8],
        "sigma_osc": [2.0, 0.5, 0.5, 1.5],
        "tau_e": 0.05,
        "sigma_e": 0.3,
        "s_c": 0.0,
        "s_r": [0.0, 0.0, 0.0, 0.0],
    }
    base.update(over)
    return Params(**base)  # type: ignore[arg-type]


# ─── (a) the spectrum is the state space's spectrum ───────────────────────────


@pytest.mark.parametrize(
    ("sigma", "f0", "zeta"),
    [(2.0, 1.0, 1.0), (1.5, 3.0, 0.3), (3.0, 0.2, 2.5), (1.0, 8.0, 0.7)],
)
def test_car2_psd_integrates_to_the_discrete_stationary_variance(
    sigma: float, f0: float, zeta: float
) -> None:
    """``int_0^{fs/2} S(f) df`` equals the variance the DISCRETE Lyapunov
    equation gives for the sampler's own ``(Phi, Q)`` — the one identity that
    keeps the likelihood and the sampler the same object.  Also checks the fast
    real form against the literal matrix expression."""
    phi, q, _ = car2_state_space(sigma, f0, zeta, RATE_HZ)
    lyapunov = solve_discrete_lyapunov(phi, q)

    f = np.linspace(0.0, 0.5 * RATE_HZ, 200_001)
    psd = car2_psd_grid(f, sigma, f0, zeta, RATE_HZ)
    assert np.trapezoid(psd, f) == pytest.approx(lyapunov[0, 0], rel=1e-5)
    assert lyapunov[0, 0] == pytest.approx(sigma**2, rel=1e-9)

    coarse = f[::4000]
    assert np.allclose(
        car2_psd_grid(coarse, sigma, f0, zeta, RATE_HZ),
        state_space_psd(coarse, phi, q, RATE_HZ),
        rtol=1e-9,
    )


@pytest.mark.parametrize(("sigma", "tau"), [(2.0, 1.3), (0.5, 0.05), (3.0, 9.9)])
def test_ou_psd_integrates_to_the_discrete_stationary_variance(sigma: float, tau: float) -> None:
    phi, q, _ = ou_state_space(sigma, tau, RATE_HZ)
    lyapunov = solve_discrete_lyapunov(phi, q)
    f = np.linspace(0.0, 0.5 * RATE_HZ, 200_001)
    assert np.trapezoid(ou_psd_grid(f, sigma, tau, RATE_HZ), f) == pytest.approx(
        lyapunov[0, 0], rel=1e-5
    )
    assert np.allclose(
        ou_psd_grid(f[::4000], sigma, tau, RATE_HZ),
        state_space_psd(f[::4000], phi, q, RATE_HZ),
        rtol=1e-9,
    )


def test_car2_is_steeper_than_any_ou_and_can_resonate() -> None:
    """The reason the second OU was replaced: a CAR2 falls at f^-4 above its
    corner, where a Lorentzian is capped at f^-2, and with ``zeta < 1/sqrt(2)``
    it has a real peak."""
    f = np.array([4.0, 40.0])
    psd = car2_psd_grid(f, 1.0, 1.0, 1.0, 1000.0)  # high fs: no aliasing
    decade_drop = np.log10(psd[0] / psd[1])
    assert decade_drop > 3.5  # ~4 decades = f^-4, vs 2 for an OU

    fine = np.linspace(0.05, 10.0, 4000)
    sharp = car2_psd_grid(fine, 1.0, 3.0, 0.25, 1000.0)
    assert fine[int(np.argmax(sharp))] == pytest.approx(3.0, abs=0.3)
    assert np.max(sharp) > 1.5 * sharp[0]
    flat = car2_psd_grid(fine, 1.0, 3.0, 2.5, 1000.0)
    assert np.max(flat) == pytest.approx(flat[0], rel=1e-6)  # overdamped: no peak


def test_periodogram_of_the_sampler_matches_the_model_spectrum() -> None:
    """The spectrum the fit evaluates is the spectrum of what the sampler draws."""
    p = _params(sigma_e=1e-4)
    rng = np.random.default_rng(3)
    n = 2000
    acc = np.zeros((n // 2 + 1, 4, 4))
    n_blocks = 200
    for _ in range(n_blocks):
        f, spec = cross_periodogram(p.sample_airborne(n, rng), RATE_HZ)
        acc += spec
    emp = acc / n_blocks
    model = p.spectral_matrix(f, RATE_HZ)
    band = (f >= 1.0) & (f <= 45.0)  # above the block's mean-removal bias
    for rotor in range(4):
        ratio = emp[band, rotor, rotor] / model[band, rotor, rotor]
        assert np.mean(ratio) == pytest.approx(1.0, abs=0.05)


# ─── (a2) the exact likelihood ────────────────────────────────────────────────


def _kalman_pieces(p: Params, fit_rate: float):
    vec = _fit_vector(p)
    phi, q, p_stat, h = _state_space_torch(_unpack(vec), 1.0 / fit_rate)
    k, s = steady_state_gain(phi, q, p_stat, h)
    return phi, q, p_stat, h, k, s


def test_state_space_is_stationary_and_matches_the_analytic_variance() -> None:
    """``P`` really is the stationary covariance of ``(Phi, Q)``, and the
    observation of it is the model's own per-rotor variance — the identity that
    ties the 16-state assembly to :meth:`Params.rotor_var`."""
    p = _params(sigma_e=0.4)
    phi, q, p_stat, h, _k, _s = _kalman_pieces(p, RATE_HZ)

    assert torch.allclose(p_stat, phi @ p_stat @ phi.T + q, atol=1e-10)
    assert np.allclose(
        np.diag((h @ p_stat @ h.T).numpy()), p.rotor_var(include_offset=False), rtol=1e-9
    )


def test_steady_state_nll_equals_the_time_varying_filter() -> None:
    """The steady-state filter's NLL equals the EXACT time-varying Kalman
    filter's on the same block after the burn-in.

    The naive filter is written out here (Riccati step per sample, gain per
    sample) precisely so that the fast path has an independent reference: the
    steady-state gain is only legitimate once the filter has forgotten its
    initial condition, which is what BURN_S buys.
    """
    p = _params(sigma_e=0.4, s_c=0.0, s_r=[0.0] * 4)
    phi, q, p_stat, h, k, s = _kalman_pieces(p, RATE_HZ)

    n_t, burn = 1500, 500
    y = p.sample_airborne(n_t, np.random.default_rng(0))
    y = y - y.mean(axis=1, keepdims=True)
    y_t = torch.as_tensor(y[None], dtype=torch.float64).transpose(1, 2)
    mask = torch.zeros(1, n_t, dtype=torch.float64)
    mask[0, burn:] = 1.0
    fast = float(_batch_nll(_Batch(y=y_t, mask=mask, burn=burn), phi, k, h, s))

    cov = p_stat.clone()
    x = torch.zeros(16, dtype=torch.float64)
    slow = 0.0
    for step in range(n_t):
        s_t = h @ cov @ h.T
        e = y_t[0, step] - h @ x
        if step >= burn:
            chol = torch.linalg.cholesky(s_t)
            slow += 0.5 * float(
                2.0 * torch.log(torch.diagonal(chol)).sum() + e @ torch.linalg.solve(s_t, e)
            )
        g = cov @ h.T
        gain = phi @ g @ torch.linalg.inv(s_t)
        cov = phi @ (cov - g @ torch.linalg.solve(s_t, g.T)) @ phi.T + q
        cov = 0.5 * (cov + cov.T)
        x = phi @ x + gain @ e

    assert abs(fast - slow) <= 0.01 * abs(slow)


def test_custom_adjoint_matches_autograd() -> None:
    """The recursion's hand-written backward must be autograd's backward.

    It exists only for speed — autograd's own backward through the loop was
    94 % of the gradient's cost on neurobem (18.2 s of 19.4 s, ~160 us of graph
    overhead per time step), and replacing it with one ``addmm`` per step made
    the gradient 19x cheaper — so the one thing that matters is that it
    computes the same thing.
    """
    torch.manual_seed(0)
    n_t, n_b = 60, 3
    drive = torch.randn(n_t, n_b, 16, dtype=torch.float64)
    f = 0.05 * torch.randn(16, 16, dtype=torch.float64) + 0.9 * torch.eye(16, dtype=torch.float64)
    h = torch.randn(4, 16, dtype=torch.float64)
    weight = torch.randn(n_t, n_b, 4, dtype=torch.float64)

    def loss(pred: torch.Tensor) -> torch.Tensor:
        return (pred * weight).sum() + (pred**2).sum()

    def naive(d: torch.Tensor, f_: torch.Tensor, h_: torch.Tensor) -> torch.Tensor:
        x = torch.zeros(n_b, 16, dtype=torch.float64)
        preds = []
        for t in range(n_t):
            preds.append(x @ h_.T)
            x = x @ f_.T + d[t]
        return torch.stack(preds)

    grads = []
    values = []
    for build in (lambda *a: cast(torch.Tensor, _Predictor.apply(*a)), naive):
        args = [x.clone().requires_grad_(True) for x in (drive, f, h)]
        value = loss(build(*args))
        grads.append(torch.autograd.grad(value, args))
        values.append(float(value))

    assert values[0] == pytest.approx(values[1], rel=1e-12)
    for mine, theirs in zip(grads[0], grads[1], strict=True):
        rel = float(torch.linalg.norm(mine - theirs)) / float(torch.linalg.norm(theirs))
        assert rel < 1e-8


def test_padding_and_masking_do_not_change_the_likelihood() -> None:
    """Blocks are zero-padded to batch them; the mask must make that exact."""
    p = _params(sigma_e=0.4, s_c=0.0, s_r=[0.0] * 4)
    phi, _q, _p_stat, h, k, s = _kalman_pieces(p, RATE_HZ)
    n_t, burn = 900, 200
    y = p.sample_airborne(n_t, np.random.default_rng(1))
    y = y - y.mean(axis=1, keepdims=True)
    y_t = torch.as_tensor(y[None], dtype=torch.float64).transpose(1, 2)
    mask = torch.zeros(1, n_t, dtype=torch.float64)
    mask[0, burn:] = 1.0
    plain = float(_batch_nll(_Batch(y=y_t, mask=mask, burn=burn), phi, k, h, s))

    pad = 400
    y_pad = torch.cat([y_t, torch.zeros(1, pad, 4, dtype=torch.float64)], dim=1)
    mask_pad = torch.cat([mask, torch.zeros(1, pad, dtype=torch.float64)], dim=1)
    padded = float(_batch_nll(_Batch(y=y_pad, mask=mask_pad, burn=burn), phi, k, h, s))

    assert padded == pytest.approx(plain, rel=1e-12)


def test_likelihood_blocks_cover_the_airborne_data_once() -> None:
    """Blocks tile each segment with a burn-in overlap, and only the burn-in is
    double-counted: the scored samples add up to the segment length."""
    p = _params()
    flights = [
        Flight(
            rig="sim",
            flight="f0",
            fs=RATE_HZ,
            t0=0.0,
            rps=p.sample_airborne(100 * int(RATE_HZ), np.random.default_rng(2)),
            source="sim",
        )
    ]
    batches = likelihood_batches(flights, RATE_HZ, RATE_HZ)
    scored = sum(b.n_scored for b in batches)
    # The airborne rule erodes 1 s at each end of the 100 s sample.
    assert scored == pytest.approx(98 * int(RATE_HZ), rel=0.02)
    assert all(b.burn in (0, int(round(5.0 * RATE_HZ))) for b in batches)


# ─── (b) the sampler ──────────────────────────────────────────────────────────


def test_car2_path_realises_its_state_recursion() -> None:
    """The oscillator path is generated as an ARMA(2,1) through one ``lfilter``
    pass (Cayley-Hamilton on the 2-state recursion).  It must be bit-for-bit
    the path the state recursion itself produces from the same randomness."""
    phi, q, p = car2_state_space(1.7, 2.5, 0.4, RATE_HZ)
    arma = _car2_path(phi, q, p, 50, np.random.default_rng(0))

    rng = np.random.default_rng(0)
    x = _psd_matrix_sqrt(p) @ rng.standard_normal(2)
    eta = _psd_matrix_sqrt(q) @ rng.standard_normal((2, 50))
    brute = []
    for k in range(50):
        brute.append(x[0])
        x = phi @ x + eta[:, k]

    assert np.allclose(arma, brute, atol=1e-12)


def test_sampler_realises_the_analytic_rotor_variance() -> None:
    """Pooled over independent 600 s flights the sampler's per-rotor variance
    is the analytic one to 5 %, and a single 600 s flight is within the
    estimator's own sampling error."""
    p = _params(sigma_e=0.2, s_c=0.0, s_r=[0.0, 0.0, 0.0, 0.0])
    analytic = p.rotor_var(include_offset=False)
    sampler = p.sampler()
    rng = np.random.default_rng(5)

    n = 600 * int(RATE_HZ)
    single = sampler(n, rng)
    assert single.shape == (4, n)
    assert np.all(np.abs(single.var(axis=1) / analytic - 1.0) < 0.15)

    # 20 independent draws: the mean of the estimator, not one realisation.
    pooled = np.mean([sampler(n, rng).var(axis=1) for _ in range(20)], axis=0)
    assert np.all(np.abs(pooled / analytic - 1.0) < 0.05)

    # The diagonal pairs are the correlated ones — the yaw signature.
    corr = np.corrcoef(single)
    assert corr[0, 2] > corr[0, 1]
    assert corr[1, 3] > corr[0, 1]


def test_per_rotor_offset_lifts_one_rotors_pooled_variance() -> None:
    """The offset's PER-ROTOR part is the only way the model can give one rotor
    more pooled variance than its neighbours — michaels' rotor 0 carries
    52.2 (rev/s)^2 against ~21 for the other three, purely from its own trim."""
    p = _params(sigma_e=0.1, s_c=0.0, s_r=[6.0, 0.2, 0.2, 0.5])
    analytic = p.rotor_var()
    assert analytic[0] > 1.5 * analytic[1]

    # 300 flights: the offset variance is estimated from ONE draw per flight, so
    # with 120 it carried a 13 % sampling error of its own.
    rng = np.random.default_rng(11)
    sampler = p.sampler()
    pooled = np.concatenate([sampler(60 * int(RATE_HZ), rng) for _ in range(300)], axis=1)
    assert np.all(np.abs(pooled.var(axis=1) / analytic - 1.0) < 0.15)


def test_antithetic_offsets_cancel_exactly() -> None:
    """Paired offsets cancel EXACTLY over an even number of flights, so the
    pooled model mean is mu — while each flight still carries a full-size
    offset, so the pooled variance is still s^2."""
    # No process and no measurement noise: each flight is exactly mu + delta,
    # which makes the cancellation exact rather than statistical.
    p = _params(
        sigma_slow=[0.0] * 4, sigma_osc=[0.0] * 4, sigma_e=0.0, s_c=0.0, s_r=[6.0, 1.0, 1.0, 2.0]
    )
    # 200 flights = 100 independent offset draws, each used twice with opposite
    # sign, so the variance estimate has ~14 % sampling error of its own.
    n, n_flights = 100, 200

    def flight_means(sampler) -> np.ndarray:
        rng = np.random.default_rng(3)
        return np.stack([sampler(n, rng).mean(axis=1) for _ in range(n_flights)], axis=0)

    anti = flight_means(p.sampler(antithetic_offsets=True))
    plain = flight_means(p.sampler())

    assert np.allclose(anti.mean(axis=0), p.mu, atol=1e-12)
    assert np.max(np.abs(plain.mean(axis=0) - p.mu)) > 0.05
    assert np.allclose(anti[0::2] - p.mu, -(anti[1::2] - p.mu))
    assert np.all(np.abs(anti.var(axis=0) / p.s_r**2 - 1.0) < 0.3)


def test_full_flight_starts_and_ends_on_the_ground() -> None:
    """The whole-flight wrapper reaches the ground at both ends, holds the idle
    plateau and never goes negative."""
    fit = NewFit(rig="sim", params=_params(), idle_rps=np.full(4, 30.0))
    w = fit.full_flight(90.0, RATE_HZ, np.random.default_rng(0))

    assert w.shape == (4, 90 * int(RATE_HZ))
    assert np.all(w >= 0.0)
    assert np.all(w[:, 0] == 0.0)
    assert np.all(w[:, -1] == 0.0)
    rotor_mean = w.mean(axis=0)
    assert np.any(np.abs(rotor_mean - 30.0) < 1.0)
    assert np.any(np.abs(rotor_mean - np.mean(fit.params.mu)) < 2.0)


# ─── (c) recovery ─────────────────────────────────────────────────────────────


def test_fit_recovers_the_parameters_it_generated() -> None:
    """Four 120 s flights of the model, fitted back.

    Tolerances are the MEASURED precision of 480 s of data, not a wish: over
    three independent data seeds the optimiser converged to an identical NLL
    with 6 and with 14 restarts (so what is left is sampling error, not a local
    optimum) and the per-mode errors were theta < 0.01 rad, f0 <= 12 %,
    tau_slow <= 22 %, the sigmas <= 20 %, and zeta 1-35 % — zeta and tau_slow
    both shape the same knee, so they trade off.  A real defect (wrong Q, wrong
    spectrum, wrong discretisation) misses by far more than this.
    """
    true = _params(sigma_e=1e-4)
    rng = np.random.default_rng(1)
    flights = [
        Flight(
            rig="sim",
            flight=f"f{i}",
            fs=RATE_HZ,
            t0=0.0,
            rps=true.sample_airborne(120 * int(RATE_HZ), rng),
            source="sim",
        )
        for i in range(4)
    ]

    fit = fit_rig(flights, "sim", seed=0, n_restarts=4, fit_rate=RATE_HZ)
    got = fit.params

    assert abs(got.theta - true.theta) < 0.15 * true.theta + 0.05
    assert np.all(np.abs(got.f0 / true.f0 - 1.0) < 0.15)
    zeta_err = np.abs(got.zeta / true.zeta - 1.0)
    assert np.all(zeta_err < 0.35)
    assert zeta_err.mean() < 0.2
    assert np.all(np.abs(got.sigma_slow / true.sigma_slow - 1.0) < 0.2)
    assert np.all(np.abs(got.sigma_osc / true.sigma_osc - 1.0) < 0.2)
    # tau_slow is identifiable only while its corner 1/(2 pi tau) stays inside
    # the fitted bins, which the MIN_BIN_RAYLEIGH cut starts at 2/T = 0.1 Hz for
    # a 20 s block; the test's 0.7-1.2 s corners (0.13-0.23 Hz) are inside.
    # Above ~1.6 s only the combination sigma_slow^2 / tau_slow is identified.
    assert np.all(np.abs(got.tau_slow / true.tau_slow - 1.0) < 0.3)
    # mu is the pooled airborne MEAN, so it carries the process' own sampling
    # error, which is ~0.5 rev/s for 480 s of a tau ~ 2 s, sigma ~ 5 rev/s
    # process.
    assert np.all(np.abs(got.mu - true.mu) < 2.0)


def test_fit_rate_is_a_per_rig_constant() -> None:
    """michaels is fitted at 25 Hz because its telemetry is logged at 29.41 Hz
    and its 100 Hz grid is interpolation above ~12 Hz; every other rig is
    genuine ESC feedback at >= 100 Hz native."""
    assert fit_rate_hz("michaels") == 25.0
    assert fit_rate_hz("michaels_test") == 25.0
    for rig in ("dregon", "neurobem_quad", "blackbird_quad", "who_knows"):
        assert fit_rate_hz(rig) == 100.0


def test_parameter_ranges_hold_for_any_coordinate() -> None:
    """Every bounded parameter is bounded by its parametrisation, so no draw or
    optimiser step can leave the identifiable region — in particular the
    oscillator is always the FASTER component."""
    # sigmoid saturates in float64, so the open bounds are closed at the
    # numerical limit: tau == tau_c is still a valid, ordered model.
    extreme = np.array([-500.0, -3.0, 3.0, 500.0])
    f0 = np.array([0.06, 1.0, 5.0, 19.0])
    tau = tau_slow_from_u(extreme, f0)
    assert np.all(tau >= corner_tau_s(f0))
    assert np.all(tau <= TAU_SLOW_MAX_S + 1e-12)

    rng = np.random.default_rng(0)
    for _ in range(200):
        p = Params.from_vector(rng.normal(0.0, 6.0, 32))
        assert np.all(p.f0 >= F0_MIN_HZ) and np.all(p.f0 <= F0_MAX_HZ)
        assert np.all(p.zeta >= ZETA_MIN) and np.all(p.zeta <= ZETA_MAX)
        assert np.all(p.tau_slow >= p.corner_tau)
        assert np.all(p.tau_slow <= TAU_SLOW_MAX_S)


# ─── (d) the posterior ────────────────────────────────────────────────────────


def test_rig_vector_round_trips() -> None:
    """Scale-free coordinates lose nothing but the ``s`` floor."""
    p = _params(sigma_e=0.2, s_c=2.0, s_r=[1.0, 0.2, 0.2, 0.5])
    back = params_from_rig_vector(rig_vector(p))

    assert np.allclose(back.mu, p.mu)
    assert back.theta == pytest.approx(p.theta)
    assert np.allclose(back.tau_slow, p.tau_slow)
    assert np.allclose(back.sigma_slow, p.sigma_slow)
    assert np.allclose(back.f0, p.f0)
    assert np.allclose(back.zeta, p.zeta)
    assert np.allclose(back.sigma_osc, p.sigma_osc)
    assert back.tau_e == pytest.approx(p.tau_e)
    assert back.sigma_e == pytest.approx(p.sigma_e)
    assert back.s_c == pytest.approx(p.s_c)
    assert np.allclose(back.s_r, p.s_r)


def test_posterior_draws_are_positive_drones_of_a_plausible_size() -> None:
    """Fitted on rigs three times apart in size, a draw is still a drone: every
    rotor mean positive, every bounded parameter in range, and a sample finite."""
    small = _params(mu=[78.0, 74.0, 80.0, 76.0], sigma_e=0.2)
    large = _params(
        mu=[280.0, 270.0, 285.0, 275.0],
        sigma_slow=[14.0, 4.0, 4.0, 10.0],
        f0=[0.5, 6.0, 5.0, 2.0],
        sigma_e=0.7,
    )
    post = fit_posterior({"small": small, "large": large})

    rng = np.random.default_rng(0)
    for _ in range(64):
        draw = post.sample(rng)
        assert np.all(draw.mu > 0.0)
        assert np.all(draw.tau_slow >= draw.corner_tau)
        assert np.all(draw.f0 >= F0_MIN_HZ) and np.all(draw.f0 <= F0_MAX_HZ)
        assert np.all(draw.zeta > ZETA_MIN) and np.all(draw.zeta < ZETA_MAX)
        w = draw.sample_airborne(10 * int(RATE_HZ), rng)
        assert np.isfinite(w).all()

    back = Posterior.from_json(post.to_json())
    assert np.allclose(back.mean, post.mean)
    assert np.allclose(back.std, post.std)
    assert back.rigs == post.rigs
