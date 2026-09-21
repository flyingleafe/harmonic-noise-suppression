"""The integrated Ornstein-Uhlenbeck shaft process: transition, structure
function and exact path simulator.

MOVED verbatim from :mod:`experiments.stochastic_fit.revised_phase`, which
imports them back. They live here because the v2 renderer
(:func:`data_processing.noise_model.render.render_noise`) draws its shaft paths
with :func:`simulate_state` and its line shapes with the structure function
:func:`integrated_ou_increment_var`, and ``data_processing`` may not import
``experiments``. One definition, read by the fit and by the renderer.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "integrated_ou_increment_var",
    "ou_cholesky",
    "ou_transition",
    "simulate_state",
]


def _xp(*arrays: Any) -> Any:
    """``torch`` if any argument is a tensor, else ``numpy`` (single dispatch)."""
    for a in arrays:
        if isinstance(a, Tensor):
            return torch
    return np


def ou_transition(lam: float, sigma: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """EXACT discrete transition ``F`` and covariance ``Q`` of the integrated OU
    on a grid of step ``dt``, for the state ``z = (theta, nu)``::

        e = exp(-lam dt)
        F = [[1, (1 - e) / lam], [0, e]]
        Var[nu]        = sigma^2 (1 - e^2)
        Cov[theta, nu] = (sigma^2 / lam) (1 - e)^2
        Var[theta]     = (2 sigma^2 / lam^2) [x - 2(1 - e) + (1 - e^2)/2],  x = lam dt

    Every entry is evaluated through ``expm1``, and ``Var[theta]`` — whose
    bracket cancels to ``x^3/3 - x^4/4 + 7 x^5/60`` — switches to that series
    below ``x = 1e-2`` instead of subtracting two nearly equal numbers. The
    small-step limits are ``Var[nu] -> 2 sigma^2 lam dt``,
    ``Cov -> sigma^2 lam dt^2`` and ``Var[theta] -> (2/3) sigma^2 lam dt^3``.
    """
    lam, sigma, dt = float(lam), float(sigma), float(dt)
    if lam <= 0.0:
        raise ValueError(f"lam must be positive, got {lam!r}")
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma!r}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt!r}")
    x = lam * dt
    e = math.exp(-x)
    one_m_e = -math.expm1(-x)
    one_m_e2 = -math.expm1(-2.0 * x)
    f_mat = np.array([[1.0, one_m_e / lam], [0.0, e]], dtype=np.float64)
    var_nu = sigma**2 * one_m_e2
    cov = (sigma**2 / lam) * one_m_e**2
    bracket = (
        x**3 / 3.0 - x**4 / 4.0 + 7.0 * x**5 / 60.0
        if x < 1e-2
        else x - 2.0 * one_m_e + 0.5 * one_m_e2
    )
    var_theta = 2.0 * sigma**2 / lam**2 * bracket
    q_mat = np.array([[var_theta, cov], [cov, var_nu]], dtype=np.float64)
    return f_mat, q_mat


def ou_cholesky(lam: float, sigma: float, dt: float) -> tuple[float, float, float]:
    """Lower Cholesky ``[[a, 0], [b, c]]`` of :func:`ou_transition`'s ``Q``."""
    _, q = ou_transition(lam, sigma, dt)
    a = math.sqrt(max(float(q[0, 0]), 0.0))
    if a == 0.0:
        return 0.0, 0.0, math.sqrt(max(float(q[1, 1]), 0.0))
    b = float(q[0, 1]) / a
    c = math.sqrt(max(float(q[1, 1]) - b * b, 0.0))
    return a, b, c


def integrated_ou_increment_var(tau_s: Any, *, lam: float, sigma: float) -> Any:
    """``Var[theta(t + tau) - theta(t)]`` of a STATIONARILY initialized
    integrated OU: ``2 sigma^2 [ |tau|/lam - (1 - exp(-lam|tau|))/lam^2 ]``.

    In ``x = lam |tau|`` the bracket is ``(x - 1 + exp(-x)) / lam^2``, which is
    ``x^2/2 - x^3/6 + x^4/24`` for small ``x`` (so the increment variance is
    ``sigma^2 tau^2`` in the frozen-rate limit — a rate held at its stationary
    scale). Below ``x = 1e-3`` the series is used instead of the cancelling
    difference. NOTE this is a different bracket from the one-step TRANSITION
    ``Var[theta]`` of :func:`ou_transition`, which starts from ``nu = 0`` and is
    ``O(lam dt^3)``; the two must not be interchanged.
    """
    xp = _xp(tau_s)
    tau = xp.abs(tau_s)
    x = lam * tau
    series = x**2 / 2.0 - x**3 / 6.0 + x**4 / 24.0
    direct = x + xp.expm1(-x)
    bracket = xp.where(x < 1e-3, series, direct)
    return 2.0 * sigma**2 / lam**2 * bracket


def _ar1_causal(drive: Any, e: float) -> Any:
    """``y_j = e y_{j-1} + drive_j`` along the last axis, vectorized.

    ``lam`` is frozen and ``dt`` uniform, so the recursion is linear
    time-invariant: it is a causal convolution with ``e^n``. numpy uses
    ``scipy.signal.lfilter``; torch uses an FFT convolution (differentiable,
    ``O(S log S)``, batched over (clip, rotor), GPU-friendly) — a per-state
    autograd loop over the 8000-16000 states of a 16 s clip is not acceptable.
    The kernel DECAYS, so it is truncated where ``e^n`` falls below 1e-12
    (float32 atoms cannot see anything smaller); when ``e`` is within 1e-12 of 1
    no truncation is possible and the full length is used.
    """
    s = int(drive.shape[-1])
    klen = s if e >= 1.0 - 1e-12 else min(s, int(math.ceil(math.log(1e-12) / math.log(e))) + 1)
    if not isinstance(drive, Tensor):
        from scipy.signal import lfilter

        return lfilter(np.array([1.0]), np.array([1.0, -e]), np.asarray(drive), axis=-1)
    n = 1 << int(s + klen - 1).bit_length()
    kern = e ** torch.arange(klen, dtype=drive.dtype, device=drive.device)
    y = torch.fft.irfft(torch.fft.rfft(drive, n=n) * torch.fft.rfft(kern, n=n), n=n)
    return y[..., :s]


def simulate_state(innov: Any, *, lam: float, sigma: float, dt: float) -> tuple[Any, Any]:
    """``(theta, nu)`` on the state grid from WHITENED innovations.

    ``innov`` is ``(..., S, 2)`` standard-normal coordinates::

        nu_0    = sigma * innov[..., 0, 0]        (stationary initial law)
        theta_0 = 0                               (gauge)
        nu_j    = e nu_{j-1} + b innov[..., j, 0] + c innov[..., j, 1]
        theta_j = theta_{j-1} + ((1 - e)/lam) nu_{j-1} + a innov[..., j, 0]

    with ``e = exp(-lam dt)`` and ``[[a, 0], [b, c]]`` the Cholesky of the exact
    ``Q(dt)``. In these coordinates the state log-prior is EXACTLY
    ``0.5 sum e^2`` — one exact Gaussian integrated-OU prior, isotropic and well
    conditioned — and the MAP is identical to the MAP in ``z`` coordinates
    because the map is linear with constant Jacobian.

    ``innov[..., 0, 1]`` is unused (the ``theta_0 = 0`` gauge has no innovation);
    the prior must exclude it, see :meth:`_RevisedModel.prior`.
    """
    xp = _xp(innov)
    a, b, c = ou_cholesky(lam, sigma, dt)
    e = math.exp(-lam * dt)
    f01 = -math.expm1(-lam * dt) / lam
    e1 = innov[..., 0]
    e2 = innov[..., 1]
    drive = b * e1 + c * e2
    if xp is torch:
        drive = torch.cat((sigma * e1[..., :1], drive[..., 1:]), dim=-1)
    else:
        drive = np.concatenate((sigma * e1[..., :1], drive[..., 1:]), axis=-1)
    nu = _ar1_causal(drive, e)
    inc = f01 * nu[..., :-1] + a * e1[..., 1:]
    # the theta_0 = 0 gauge is seeded from a length-one slice of the STATE
    # grid, not of the increments: with S == 1 there is no increment at all,
    # and slicing the empty increment array returned a length-0 theta against
    # a length-1 nu, breaking the same-grid contract.
    if xp is torch:
        theta = torch.cat((torch.zeros_like(nu[..., :1]), torch.cumsum(inc, dim=-1)), dim=-1)
    else:
        theta = np.concatenate((np.zeros_like(nu[..., :1]), np.cumsum(inc, axis=-1)), axis=-1)
    return theta, nu
