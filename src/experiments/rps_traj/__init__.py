"""Per-rig probabilistic model of multi-rotor rotor-speed (RPS) trajectories.

Two frozen pieces live here, and nothing else in the campaign may redefine
them:

- :mod:`experiments.rps_traj.data` — REAL trajectories on one common analysis
  grid (:data:`~experiments.rps_traj.data.RATE_HZ` = 100 Hz), plus the frozen
  airborne-segment rule that decides which samples any statistic is allowed
  to see;
- :mod:`experiments.rps_traj.stats` — the frozen summary statistics
  (:class:`~experiments.rps_traj.stats.TrajStats`), the five-family
  :func:`~experiments.rps_traj.stats.discrepancy` against real statistics and
  the :func:`~experiments.rps_traj.stats.passes` acceptance rule every
  candidate model is judged by.

The CLIs are ``scripts/rps_traj_real_stats.py`` (real statistics per rig) and
``scripts/rps_traj_compare.py`` (base vs new model verdict).
"""

from __future__ import annotations

from experiments.rps_traj.data import (
    MICHAELS_TEST_IS_SAME_RIG,
    RATE_HZ,
    Flight,
    airborne_segments,
    load_frames_dataset,
    load_rig,
    to_common_grid,
)
from experiments.rps_traj.stats import (
    FAMILIES,
    LAG_SAMPLES,
    LAGS_S,
    TrajStats,
    compute_stats,
    discrepancy,
    passes,
    stats_from_samples,
)

__all__ = [
    "FAMILIES",
    "LAGS_S",
    "LAG_SAMPLES",
    "MICHAELS_TEST_IS_SAME_RIG",
    "RATE_HZ",
    "Flight",
    "TrajStats",
    "airborne_segments",
    "compute_stats",
    "discrepancy",
    "load_frames_dataset",
    "load_rig",
    "passes",
    "stats_from_samples",
    "to_common_grid",
]
