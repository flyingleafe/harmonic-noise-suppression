"""Reading a ``noise-v2-fit`` payload: the schema gate and the line widths.

MOVED from :mod:`experiments.noise_model.model` (:func:`gamma_from_params`) and
:mod:`experiments.noise_model.render` (the schema check), both of which import
them back. This is everything a RENDERER needs out of a fit JSON; the torch
:func:`experiments.noise_model.model.params_from_dict`, which builds the
forward model's parameter object, stays upstairs because only the forward model
reads it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from data_processing.noise_model import READABLE_FIT_SCHEMAS

__all__ = ["check_schema", "gamma_from_params"]


def check_schema(fit: dict[str, Any]) -> dict[str, Any]:
    """The ``params`` block of ``fit``, or a raise naming the schemas read."""
    schema = fit.get("schema")
    if schema not in READABLE_FIT_SCHEMAS:
        raise ValueError(f"expected schema in {READABLE_FIT_SCHEMAS}, got {schema!r}")
    return fit["params"]


def gamma_from_params(d: dict[str, Any]) -> np.ndarray:
    """``(R, K)`` widths of a ``/2`` payload, or a ``/1`` payload MAPPED.

    A ``noise-v2-fit/1`` fit carried a per-order OU instead: phase variance
    ``sigma_eps^2 k^p`` relaxing at ``lam_eps``, both by the PARITY of ``k``.
    At lags short against ``1 / lam_eps`` — the regime every fitted rate of R1
    and R2 sat in on its own window — its exponent is
    ``sigma_eps^2 k^p lam_eps |tau|``, which is R3's ``2 pi gamma_rk |tau|``
    with

        gamma_rk = sigma_eps(parity of k)^2 k^p lam_eps(parity of k) / (2 pi).

    That is the equivalence used here, so an old fit renders as the same line
    shape it was fitted with wherever its per-order term was diffusive; where
    it had saturated, the mapped Lorentzian is WIDER than the old pedestal was
    (the saturated case is the one R3 removed for having no evidence).
    """
    if "gamma_hz" in d:
        return np.atleast_2d(np.asarray(d["gamma_hz"], dtype=np.float64))
    prof = np.atleast_2d(np.asarray(d["profile"]["profile_db"], dtype=np.float64))
    k = np.arange(1, prof.shape[1] + 1, dtype=np.float64)
    even = np.remainder(k, 2.0) == 0.0
    se = np.where(even, float(d["sigma_eps_even"]), float(d["sigma_eps_odd"]))
    le = np.where(even, float(d["lam_eps_even"]), float(d["lam_eps_odd"]))
    p = float(d.get("p", 1.0))
    gamma = se**2 * k**p * le / (2.0 * math.pi)
    return np.broadcast_to(gamma[None, :], prof.shape).copy()
