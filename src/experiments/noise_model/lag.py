"""The v2 residual-phase autocorrelation ``R_rk(tau)``.

ONE function, :func:`r_tau`, is the whole difference between v2 and C4 at the
level of the forward model: :mod:`.spectrum` hands it to the same finite-window
kernel C4 uses, and :mod:`.render` draws the processes it describes. Keeping it
here rather than inline in both means the fit and the renderer cannot disagree
about the law.

THE LAW (``docs/explainers/noise-model-v2-plan.qmd``, "Lag equations")::

    R_rk(tau) = exp[ -k^2 V_theta(tau) / 2
                     - sigma_eps^2 k^p (1 - exp(-lam_eps |tau|)) ]

with ``V_theta`` the structure function of the STATIONARILY initialised
integrated OU speed error — :func:`revised_phase.integrated_ou_increment_var`,
imported, not re-derived — and the second term the increment variance of an
independent per-order OU **on the phase**,

    d eps_rk = -lam_eps eps_rk dt + sqrt(2 lam_eps) sigma_eps k^{p/2} dW_rk,

whose stationary variance is ``sigma_eps^2 k^p`` and whose autocorrelation is
``exp(-lam_eps |tau|)``, so ``Var[eps(t+tau) - eps(t)] = 2 sigma_eps^2 k^p
(1 - exp(-lam_eps|tau|))`` and the characteristic function contributes half of
it in the exponent. ``p = 1`` is FIXED for the campaign (bench estimate 1.08,
CI 0.64-1.44) and is a keyword only so a later round can free it without
touching the callers.

PARITY. The bench measures even orders still diffusive at 500 ms and odd orders
saturating, so ``sigma_eps`` and ``lam_eps`` each come as a pair
``(even, odd)`` selected by the parity of ``k`` (:func:`parity_select`). A
scalar is accepted and means "the same value for both parities", which is what
the ``sigma_eps = 0`` degenerate case and the C4 differential test use.

SPECIAL CASES, both exercised by the tests:

* ``sigma_eps = 0`` reduces the law to C4's :func:`revised_phase.prior_r_tau`
  at ``d = 0`` — the shaft term alone;
* ``lam_eps -> inf`` freezes the per-order term at its saturated value
  ``exp(-sigma_eps^2 k^p)``, a flat per-order coherent fraction.

BACKENDS. numpy in, numpy out; torch anywhere in, torch out and
differentiable. The dispatch reads EVERY argument, because during the fit the
lags are a fixed buffer while ``sigma_nu``, ``lam``, ``sigma_eps`` and
``lam_eps`` are the tensors carrying the gradient.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor

from experiments.stochastic_fit.revised_phase import integrated_ou_increment_var

__all__ = ["P_ORDER_EXPONENT", "parity_select", "r_tau", "saturated_coherence"]

#: The order exponent of the per-order phase drive, FIXED at 1 by the approved
#: decision of 2026-09-17 (bench estimate 1.08, CI 0.64-1.44).
P_ORDER_EXPONENT = 1.0


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


def parity_select(pair: Any, k: Any) -> Any:
    """``pair[0]`` where ``k`` is EVEN, ``pair[1]`` where it is odd.

    ``pair`` is either a scalar (both parities share it) or a length-2
    sequence/tensor ``(even, odd)``. The result broadcasts against ``k``, and
    the selection is differentiable in both entries of the pair — a
    ``torch.where`` on an integer test, never a boolean index that would drop
    one branch's gradient when a parity is momentarily absent from ``k``.
    """
    ref = _ref_tensor(pair, k)
    if ref is None:
        p = np.asarray(pair, dtype=np.float64)
        ko = np.asarray(k, dtype=np.float64)
        if p.ndim == 0:
            return np.broadcast_to(p, np.shape(ko)) + 0.0 * ko
        if p.shape[-1] != 2:
            raise ValueError(f"parity pair must have 2 entries (even, odd), got shape {p.shape}")
        return np.where(np.remainder(ko, 2.0) == 0.0, p[..., 0], p[..., 1])
    p_t = _cast(pair, ref)
    k_t = _cast(k, ref)
    if p_t.ndim == 0:
        return p_t.expand(k_t.shape) if k_t.ndim else p_t
    if int(p_t.shape[-1]) != 2:
        raise ValueError(
            f"parity pair must have 2 entries (even, odd), got shape {tuple(p_t.shape)}"
        )
    even = torch.remainder(k_t, 2.0) == 0.0
    return torch.where(even, p_t[..., 0], p_t[..., 1])


def r_tau(
    tau_s: Any,
    k: Any,
    *,
    sigma_nu: Any,
    lam: Any,
    sigma_eps: Any,
    lam_eps: Any,
    p: float = P_ORDER_EXPONENT,
) -> Any:
    """``R_rk(tau)`` of the module docstring.

    Parameters
    ----------
    tau_s
        Lags in seconds; only ``|tau|`` is read. Broadcast against ``k`` by the
        caller exactly as :func:`revised_phase.prior_r_tau` expects it, e.g.
        ``r_tau(tau[None, :], k[:, None], ...)`` for a ``(K, T)`` block.
    k
        Harmonic order(s). Its PARITY selects the per-order pair.
    sigma_nu, lam
        The shaft OU speed-error scale (rad/s) and rate (1/s).
    sigma_eps, lam_eps
        Scalars, or ``(even, odd)`` pairs, of the per-order phase OU: the
        stationary phase deviation at ``k = 1`` (rad) and the rate (1/s).
    p
        Order exponent of the per-order drive; 1 for the campaign.
    """
    ref = _ref_tensor(tau_s, k, sigma_nu, lam, sigma_eps, lam_eps)
    if ref is not None and not ref.is_floating_point():
        ref = torch.zeros((), dtype=torch.float64, device=ref.device)
    if ref is None:
        tau = np.abs(np.asarray(tau_s, dtype=np.float64))
        k_v = np.asarray(k, dtype=np.float64)
        s_e = parity_select(sigma_eps, k_v)
        l_e = parity_select(lam_eps, k_v)
        shaft = 0.5 * k_v**2 * integrated_ou_increment_var(tau, lam=lam, sigma=sigma_nu)
        order = s_e**2 * k_v**p * -np.expm1(-l_e * tau)
        return np.exp(-shaft - order)
    tau = torch.abs(_cast(tau_s, ref))
    k_t = _cast(k, ref)
    s_e = parity_select(_cast(sigma_eps, ref), k_t)
    l_e = parity_select(_cast(lam_eps, ref), k_t)
    shaft = (
        0.5
        * k_t**2
        * integrated_ou_increment_var(tau, lam=_cast(lam, ref), sigma=_cast(sigma_nu, ref))
    )
    order = s_e**2 * k_t**p * -torch.expm1(-l_e * tau)
    return torch.exp(-shaft - order)


def saturated_coherence(k: Any, *, sigma_eps: Any, p: float = P_ORDER_EXPONENT) -> Any:
    """``exp(-sigma_eps^2 k^p)``: the peak-to-pedestal power ratio of order ``k``.

    The limit of :func:`r_tau`'s per-order factor at lags long against
    ``1/lam_eps``, i.e. the coherent fraction the campaign's two-component line
    fits used to carry as a free number per order. Reported in the fit
    findings, never fitted.
    """
    ref = _ref_tensor(k, sigma_eps)
    if ref is None:
        k_v = np.asarray(k, dtype=np.float64)
        return np.exp(-(parity_select(sigma_eps, k_v) ** 2) * k_v**p)
    k_t = _cast(k, ref)
    return torch.exp(-(parity_select(_cast(sigma_eps, ref), k_t) ** 2) * k_t**p)


def order_lag_support_s(
    k: Any,
    *,
    sigma_nu: float,
    lam: float,
    threshold_nats: float = 40.0,
    tau_cap_s: float = math.inf,
) -> Any:
    """Lag(s) beyond which order ``k``'s SHAFT factor is below ``e^-threshold``.

    Only the shaft term decays without bound (the per-order term saturates), so
    this alone bounds the lag support of a line. :mod:`.spectrum`'s bench model
    uses it to stop summing an order's autocovariance where it has died, which
    is what makes a 30 s whole-segment periodogram affordable: order 110 of a
    70 rev/s rotor is gone by 216 ms at the prior centre.

    Solved by bisection on the EXACT ``k^2 V_theta(tau)/2``, VECTORISED over
    ``k`` (it is called once per objective evaluation for every order), on
    detached floats — the closed-form inverses of the two asymptotic regimes
    bracket the root from the wrong side, so neither is usable as a bound.
    ``tau_cap_s`` is returned where the factor never reaches the threshold
    inside the cap. Scalar in, float out; array in, array out.
    """
    if not math.isfinite(tau_cap_s):
        raise ValueError("order_lag_support_s needs a finite tau_cap_s (the segment length)")
    k_arr = np.atleast_1d(np.asarray(k, dtype=np.float64))

    def half(t: Any) -> Any:
        return 0.5 * k_arr**2 * integrated_ou_increment_var(t, lam=lam, sigma=sigma_nu)

    lo = np.zeros_like(k_arr)
    hi = np.full_like(k_arr, float(tau_cap_s))
    reaches = half(hi) > threshold_nats
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        below = half(mid) < threshold_nats
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    out = np.where(reaches, hi, float(tau_cap_s))
    return float(out[0]) if np.ndim(k) == 0 else out
