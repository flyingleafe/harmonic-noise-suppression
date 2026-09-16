"""FIT the rig posterior — the campaign's GLOBAL fit over its seven rigs.

Seven rigs is not much to learn a 30-dimensional distribution from, so the
posterior is deliberately the simplest thing that can be sampled: a DIAGONAL
Gaussian over a reparametrised vector, fitted to the seven fitted rigs.  What
makes that defensible is the reparametrisation, not the density.

The rigs differ in absolute size by a factor of 3.6 — michaels hovers at
~78 rev/s and the small-frame rigs at ~280 rev/s — so a Gaussian over raw
parameters would mostly encode "how big is the drone", and a draw combining
one rig's hover level with another's absolute jitter would be nonsense.
:func:`~data_processing.trajectory_model.rig_vector` therefore makes every
coordinate dimensionless:

* ``log mean(mu)`` — the one scale coordinate;
* three RELATIVE trims ``(M^T mu / 4)[1:] / mean(mu)`` (roll/pitch/yaw trim as
  a fraction of the hover level; michaels' ~27 % relative spread is the same at
  idle and at cruise, so a ratio is the right invariant);
* ``theta``, ``u_slow``, ``log f0``, ``logit zeta`` and ``log tau_e`` — already
  dimensionless (a time constant does not scale with the drone's size);
* every ``sigma``, including the measurement process' ``sigma_e`` and both
  parts of the per-flight offset (``s_c`` common, ``s_r`` per rotor), as
  ``log(sigma / mean(mu))``, i.e. as a relative fluctuation.

The coordinates, the Gaussian and its draw live in
:mod:`data_processing.trajectory_model.posterior`, because the training streams
draw from it (``rps.kind: fitted_traj``, rig name ``posterior``); this module is
the campaign half — which coordinates a rig actually measures
(:func:`informative_mask`) and the width floors a seven-rig population needs
(:func:`fit_posterior`).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from data_processing.trajectory_model import NewFit, Params, Posterior, rig_vector
from data_processing.trajectory_model.params import S_FLOOR
from data_processing.trajectory_model.posterior import (
    IDX_LOG_S_C,
    IDX_LOG_S_R,
    IDX_LOG_SCALE,
    IDX_LOG_SIGMA_E,
    IDX_LOG_TAU_E,
    RIG_DIM,
)

#: Coordinates that are logs of a positive quantity: their prior width is
#: floored relatively as well as absolutely (see :func:`fit_posterior`).
LOG_DIMS = np.array(
    [IDX_LOG_SCALE, IDX_LOG_TAU_E, IDX_LOG_SIGMA_E, IDX_LOG_S_C]
    + list(range(9, 17))
    + list(range(21, 25))
    + list(range(28, 32))
)

#: Floors on a fitted coordinate's std: absolute, and relative to |mean| for
#: the log coordinates.
STD_FLOOR = 0.05
STD_FLOOR_REL = 0.1


def informative_mask(params: Params) -> np.ndarray:
    """``(32,)`` True where this rig's coordinate carries information.

    Only the per-flight offsets can be uninformative: a rig with no
    between-flight level difference has ``s = 0``, which
    :func:`~data_processing.trajectory_model.rig_vector` must write as
    ``log(S_FLOOR / scale)`` — a placeholder, not a measurement.
    :func:`fit_posterior` leaves those out of the moments, and anything
    reporting how far a rig sits from the posterior has to leave them out too,
    or a placeholder reads as a 14-sigma outlier.
    """
    mask = np.ones(RIG_DIM, dtype=bool)
    mask[IDX_LOG_S_C] = params.s_c > S_FLOOR
    mask[IDX_LOG_S_R] = np.asarray(params.s_r) > S_FLOOR
    return mask


def fit_posterior(fits: Mapping[str, NewFit | Params]) -> Posterior:
    """Diagonal Gaussian fitted to the rigs' scale-free vectors.

    The per-coordinate std is the population std over rigs, floored at
    :data:`STD_FLOOR` and — for the log coordinates, where a std is a relative
    width — additionally at :data:`STD_FLOOR_REL` x ``|mean|``.  With seven rigs
    the raw std of a coordinate they happen to agree on is far too tight to
    sample from; the floors say "we have seven drones, not a population".

    THE OFFSET COORDINATES ARE FITTED ONLY WHERE THEY EXIST.  A rig with no
    between-flight level difference — blackbird has ONE flight, dregon's flights
    share a level — has ``s = 0``, which
    :func:`~data_processing.trajectory_model.rig_vector` has to write as
    ``log(S_FLOOR / scale)``, i.e. -12.1 and -11.3 against an informative
    rig's -1.6 to -4.2.  Averaging those in dragged the mean to -5.7 and
    inflated the std to 3.9, so a +2 sigma draw asked for a per-flight offset
    EIGHT TIMES the hover level (the shipped round-5 draw 11 had
    ``s_c`` = 3267 rev/s against ``mu`` = 130).  A floored zero is a different
    fact from a small value, not a small value, so it is excluded from that
    coordinate's moments; if no rig is informative the coordinate keeps the
    floor and the default width.
    """
    if not fits:
        raise ValueError("no fits to build a posterior from")
    rigs = tuple(sorted(fits))
    params: list[Params] = []
    for rig in rigs:
        entry = fits[rig]
        params.append(entry.params if isinstance(entry, NewFit) else entry)
    vectors = np.stack([rig_vector(p) for p in params], axis=0)

    informative = np.stack([informative_mask(p) for p in params], axis=0)

    mean = np.empty(RIG_DIM)
    std = np.empty(RIG_DIM)
    for dim in range(RIG_DIM):
        keep = vectors[informative[:, dim], dim]
        column = keep if keep.size else vectors[:, dim]
        mean[dim] = column.mean()
        std[dim] = column.std()

    floor = np.full(RIG_DIM, STD_FLOOR)
    floor[LOG_DIMS] = np.maximum(floor[LOG_DIMS], STD_FLOOR_REL * np.abs(mean[LOG_DIMS]))
    return Posterior(mean=mean, std=np.maximum(std, floor), rigs=rigs)


__all__ = [
    "LOG_DIMS",
    "STD_FLOOR",
    "STD_FLOOR_REL",
    "fit_posterior",
    "informative_mask",
]
