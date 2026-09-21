"""The fitted generative model of multi-rotor rotor-speed trajectories.

The SAMPLING half of the campaign model documented in
``docs/experiments/rps-trajectory-model.md``: the 32 parameters and their exact
discrete state spaces (:mod:`~data_processing.trajectory_model.params`), the
fitted-rig record and its ESC clamp
(:mod:`~data_processing.trajectory_model.sampler`), the whole-flight envelope
(:mod:`~data_processing.trajectory_model.flight`), the rig posterior — the
global fit — and its draw (:mod:`~data_processing.trajectory_model.posterior`),
and the training-stream source behind ``rps.kind: fitted_traj``
(:mod:`~data_processing.trajectory_model.source`).

The FIT itself (torch Kalman likelihood, priors, optimiser, and the fit of the
posterior to the seven rigs) stays in :mod:`experiments.rps_traj`, which imports
from here so the math has exactly one home.
"""

from __future__ import annotations

from data_processing.trajectory_model.flight import (
    MIN_AIRBORNE_FRAC,
    AirborneSampler,
    wrap_airborne,
)
from data_processing.trajectory_model.params import (
    F0_MAX_HZ,
    F0_MIN_HZ,
    N_PARAMS,
    N_STATES,
    RATE_HZ,
    S_FLOOR,
    SIGMA_E_FLOOR,
    TAU_E_MAX_S,
    TAU_E_MIN_S,
    TAU_SLOW_MAX_S,
    ZETA_MAX,
    ZETA_MIN,
    Params,
    Sampler,
    car2_psd_grid,
    car2_state_space,
    corner_tau_s,
    f0_from_v,
    ou_psd_grid,
    ou_state_space,
    rotation,
    state_space_psd,
    tau_e_from_v,
    tau_slow_from_u,
    u_from_tau_slow,
    v_from_f0,
    v_from_tau_e,
    v_from_zeta,
    zeta_from_v,
)
from data_processing.trajectory_model.posterior import (
    RIG_DIM,
    Posterior,
    params_from_rig_vector,
    rig_vector,
)
from data_processing.trajectory_model.sampler import NewFit
from data_processing.trajectory_model.source import (
    FITTED_KIND,
    FLIGHT_KINDS,
    POSTERIOR_RIG,
    FitBundle,
    FittedTrajectorySource,
    RigDraw,
    build_from_config,
    load_bundle,
)
from data_processing.trajectory_model.window import (
    FlightCache,
    make_flight_cache,
    window_flight,
)

__all__ = [
    "F0_MAX_HZ",
    "F0_MIN_HZ",
    "FITTED_KIND",
    "FLIGHT_KINDS",
    "MIN_AIRBORNE_FRAC",
    "N_PARAMS",
    "N_STATES",
    "POSTERIOR_RIG",
    "RATE_HZ",
    "RIG_DIM",
    "SIGMA_E_FLOOR",
    "S_FLOOR",
    "TAU_E_MAX_S",
    "TAU_E_MIN_S",
    "TAU_SLOW_MAX_S",
    "ZETA_MAX",
    "ZETA_MIN",
    "AirborneSampler",
    "FitBundle",
    "FittedTrajectorySource",
    "FlightCache",
    "NewFit",
    "Params",
    "Posterior",
    "RigDraw",
    "Sampler",
    "build_from_config",
    "car2_psd_grid",
    "car2_state_space",
    "corner_tau_s",
    "f0_from_v",
    "load_bundle",
    "make_flight_cache",
    "ou_psd_grid",
    "ou_state_space",
    "params_from_rig_vector",
    "rig_vector",
    "rotation",
    "state_space_psd",
    "tau_e_from_v",
    "tau_slow_from_u",
    "u_from_tau_slow",
    "v_from_f0",
    "v_from_tau_e",
    "v_from_zeta",
    "window_flight",
    "wrap_airborne",
    "zeta_from_v",
]
