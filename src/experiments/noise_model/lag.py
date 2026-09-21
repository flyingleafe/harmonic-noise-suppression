"""The v2 residual-phase autocorrelation ``R_rk(tau)`` — Model R3.

MOVED to :mod:`data_processing.noise_model.lag`: the law is what the renderer
draws, the renderer now lives under ``data_processing`` (import-linter:
"nothing imports experiments"), and one definition is the whole point of the
module — :mod:`.spectrum` hands it to the same finite-window kernel C4 uses and
the renderer draws the processes it describes, so the fit and the renderer
cannot disagree about the law. Read that module's docstring for the law itself,
its special cases, its shape rules and its backend dispatch.

Every public name is re-exported here unchanged, so ``from .lag import r_tau``
and ``from experiments.noise_model.lag import order_lag_support_s`` keep
working.
"""

from __future__ import annotations

from data_processing.noise_model.lag import order_lag_support_s, r_tau

__all__ = ["order_lag_support_s", "r_tau"]
