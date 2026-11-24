"""Population bridge with nonamplifying physical carry at every noise-scale change."""

from experiments.rps_traj import population_bridge

MODEL_CLASS = population_bridge.PopulationBridge


def fit(tracks, *, measured=None, max_iter=80):
    return population_bridge.fit(tracks, measured=measured, max_iter=max_iter, bounded=True)
