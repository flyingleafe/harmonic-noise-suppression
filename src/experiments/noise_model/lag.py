"""The v2 residual-phase autocorrelation ``R_rk(tau)`` — Model R3.

ONE function, :func:`r_tau`, is the whole difference between v2 and C4 at the
level of the forward model: :mod:`.spectrum` hands it to the same finite-window
kernel C4 uses, and :mod:`.render` draws the processes it describes. Keeping it
here rather than inline in both means the fit and the renderer cannot disagree
about the law.

THE LAW (``docs/explainers/noise-model-v2-plan.qmd``, "Model R3")::

    R_rk(tau) = exp[ -k^2 V_theta(tau) / 2 - 2 pi gamma_rk |tau| ]

with ``V_theta`` the structure function of the STATIONARILY initialised
integrated OU speed error — :func:`revised_phase.integrated_ou_increment_var`,
imported, not re-derived — and the second term the increment variance of an
independent WIENER phase per line,

    d psi_rk = sqrt(4 pi gamma_rk) dW_rk,
    Var[psi_rk(t + tau) - psi_rk(t)] = 4 pi gamma_rk |tau|,

so ``E exp(i d psi) = exp(-2 pi gamma_rk |tau|)`` and the line is a LORENTZIAN
of half-width at half-maximum ``gamma_rk`` Hz. There is no law across ``k``:
``gamma_rk`` is a free non-negative number per rotor and order, replacing R1's
per-order OU pair ``(sigma_eps, lam_eps)`` and its fixed exponent ``p``, of
which the 128 ms flight window identified only the product and the bench found
no ``k^p`` law at all (``results/noise_v2/decoherence/findings.md``,
``rounds/round1/basin/findings.md``).

SPECIAL CASES, both exercised by the tests:

* ``gamma_hz = 0`` reduces the law to C4's :func:`revised_phase.prior_r_tau`
  at ``d = 0`` — the shaft term alone, the "no per-order noise" case;
* a line at the resolution limit of a window of length ``T`` has
  ``gamma_rk ~ 1 / (2 T)``, the floor the low-order check of :mod:`.fit`
  measures against; the PRIOR itself is the physical bench decoherence law
  (``0.01 k`` Hz, :class:`.model.Priors`), not the resolution.

SHAPES. ``tau_s``, ``k`` and ``gamma_hz`` broadcast against each other by the
ordinary rules, with the LAG the last axis: a ``(K,)`` or ``(R, K)``
``gamma_hz`` must therefore carry an explicit trailing lag axis
(``gamma[:, :, None]``) exactly as ``k`` does, and passing one that does not is
an error rather than a silent outer product.

BACKENDS. numpy in, numpy out; torch anywhere in, torch out and
differentiable. The dispatch reads EVERY argument, because during the fit the
lags are a fixed buffer while ``sigma_nu``, ``lam`` and ``gamma_hz`` are the
tensors carrying the gradient.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor

from experiments.stochastic_fit.revised_phase import integrated_ou_increment_var

__all__ = ["order_lag_support_s", "r_tau"]

TWO_PI = 2.0 * math.pi


def _ref_tensor(*values: Any) -> Tensor | None:
    """The first argument that already IS a tensor; it owns device and dtype."""
    for v in values:
        if isinstance(v, Tensor):
            return v
    return None


def _cast(value: Any, ref: Tensor | None) -> Any:
    if ref is None:
        return np.asarray(value, dtype=np.float64)
    if isinstance(value, Tensor):
        return value.to(dtype=ref.dtype, device=ref.device)
    return torch.as_tensor(value, dtype=ref.dtype, device=ref.device)


def _check_lag_axis(gamma_shape: tuple[int, ...], tau_shape: tuple[int, ...]) -> None:
    """``gamma_hz`` must broadcast along the LAG axis, not against it.

    A ``(K,)`` width vector passed beside a ``(K, 1)`` order column would
    broadcast to ``(K, K)`` and give every order every other order's width —
    numerically silent and physically nonsense. The caller adds the lag axis.
    """
    if not gamma_shape or not tau_shape:
        return
    n_tau = int(tau_shape[-1])
    if n_tau > 1 and int(gamma_shape[-1]) not in (1, n_tau):
        raise ValueError(
            f"gamma_hz of shape {gamma_shape} has no lag axis against {n_tau} lags; "
            "pass gamma[..., None] so the width broadcasts ALONG the lags"
        )


def r_tau(tau_s: Any, k: Any, *, sigma_nu: Any, lam: Any, gamma_hz: Any) -> Any:
    """``R_rk(tau)`` of the module docstring.

    Parameters
    ----------
    tau_s
        Lags in seconds; only ``|tau|`` is read, and its LAST axis is the lag
        axis. Broadcast against ``k`` exactly as
        :func:`revised_phase.prior_r_tau` expects it, e.g.
        ``r_tau(tau[None, :], k[:, None], ...)`` for a ``(K, T)`` block.
    k
        Harmonic order(s).
    sigma_nu, lam
        The shaft OU speed-error scale (rad/s) and rate (1/s), shared by every
        order of a rotor.
    gamma_hz
        The Lorentzian half-width at half-maximum (Hz) of the line, per rotor
        and order: a scalar, or an array broadcasting against ``k`` with the
        lag axis made explicit — ``(K, 1)`` beside a ``(K, T)`` block,
        ``(R, K, 1)`` beside an ``(R, K, T)`` one.
    """
    ref = _ref_tensor(tau_s, k, sigma_nu, lam, gamma_hz)
    if ref is not None and not ref.is_floating_point():
        ref = torch.zeros((), dtype=torch.float64, device=ref.device)
    if ref is None:
        tau = np.abs(np.asarray(tau_s, dtype=np.float64))
        k_v = np.asarray(k, dtype=np.float64)
        g = np.asarray(gamma_hz, dtype=np.float64)
        _check_lag_axis(g.shape, tau.shape)
        shaft = 0.5 * k_v**2 * integrated_ou_increment_var(tau, lam=lam, sigma=sigma_nu)
        return np.exp(-shaft - TWO_PI * g * tau)
    tau = torch.abs(_cast(tau_s, ref))
    k_t = _cast(k, ref)
    g_t = _cast(gamma_hz, ref)
    _check_lag_axis(tuple(g_t.shape), tuple(tau.shape))
    shaft = (
        0.5
        * k_t**2
        * integrated_ou_increment_var(tau, lam=_cast(lam, ref), sigma=_cast(sigma_nu, ref))
    )
    return torch.exp(-shaft - TWO_PI * g_t * tau)


def order_lag_support_s(
    k: Any,
    *,
    sigma_nu: float,
    lam: float,
    gamma_hz: Any = 0.0,
    threshold_nats: float = 40.0,
    tau_cap_s: float = math.inf,
) -> Any:
    """Lag(s) beyond which order ``k``'s factor is below ``e^-threshold``.

    Both terms of the law decay without bound in R3, so the support is set by
    their sum ``k^2 V_theta(tau) / 2 + 2 pi gamma_k tau``. :mod:`.spectrum`'s
    bench model uses it to stop summing an order's autocovariance where it has
    died, which is what makes a 30 s whole-segment periodogram affordable.

    ``gamma_hz`` is a scalar, a ``(K,)`` row or an ``(R, K)`` block; a block is
    reduced by its MINIMUM over rotors, so the grid one order gets is the one
    the SLOWEST-decaying rotor needs and no rotor is truncated early. The same
    reasoning makes the caller pass the prior CENTRE rather than a fitted
    width: a larger fitted ``gamma`` only decays faster than the grid assumes.

    Solved by bisection on the EXACT exponent, VECTORISED over ``k`` (it is
    called once per objective evaluation for every order), on detached
    floats — the closed-form inverses of the two asymptotic regimes bracket the
    root from the wrong side, so neither is usable as a bound. ``tau_cap_s`` is
    returned where the factor never reaches the threshold inside the cap.
    Scalar in, float out; array in, array out.
    """
    if not math.isfinite(tau_cap_s):
        raise ValueError("order_lag_support_s needs a finite tau_cap_s (the segment length)")
    k_arr = np.atleast_1d(np.asarray(k, dtype=np.float64))
    g = np.asarray(gamma_hz, dtype=np.float64)
    while g.ndim > 1:
        g = g.min(axis=0)
    g_arr = np.broadcast_to(np.atleast_1d(g), k_arr.shape) if g.ndim else np.full_like(k_arr, g)

    def total(t: Any) -> Any:
        shaft = 0.5 * k_arr**2 * integrated_ou_increment_var(t, lam=lam, sigma=sigma_nu)
        return shaft + TWO_PI * g_arr * t

    lo = np.zeros_like(k_arr)
    hi = np.full_like(k_arr, float(tau_cap_s))
    reaches = total(hi) > threshold_nats
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        below = total(mid) < threshold_nats
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    out = np.where(reaches, hi, float(tau_cap_s))
    return float(out[0]) if np.ndim(k) == 0 else out
