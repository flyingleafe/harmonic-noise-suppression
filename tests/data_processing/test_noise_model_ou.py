"""Exact covariance of the local-coefficient shaft process, without Monte Carlo."""

import numpy as np
import pytest

from data_processing.noise_model.ou import ou_transition, simulate_state_path


@pytest.mark.parametrize("vary_rate", [False, True])
def test_local_coefficients_follow_the_integrated_ou_covariance(vary_rate: bool) -> None:
    n, dt = 18, 0.005
    scales = np.repeat([0.2, 0.9, 0.0], 6)
    rates = np.repeat([12.0, 60.0, 4.0], 6) if vary_rate else np.full(n, 12.0)
    # Propagate the Gaussian state covariance independently of the path simulator.
    expected = np.empty((n, 2, 2))
    expected[0] = np.diag([0.0, scales[0] ** 2])
    for j in range(1, n):
        transition, noise = ou_transition(float(rates[j]), float(scales[j]), dt)
        expected[j] = transition @ expected[j - 1] @ transition.T + noise

    # The simulator is linear in its white coordinates: J J^T is its exact
    # covariance. Basis paths also exercise zero scale without random tolerances.
    columns = []
    for coordinate in range(2 * n):
        innovations = np.zeros((n, 2))
        innovations.flat[coordinate] = 1.0
        theta, nu = simulate_state_path(innovations, lam=rates, sigma=scales, dt=dt)
        columns.append(np.stack([theta, nu], axis=-1))
    jacobian = np.stack(columns)
    actual = np.einsum("cni,cnj->nij", jacobian, jacobian)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-16)
