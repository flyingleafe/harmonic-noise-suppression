"""A rig posterior over the new model's parameters, in scale-free coordinates.

Seven rigs is not much to learn a 30-dimensional distribution from, so the
posterior here is deliberately the simplest thing that can be sampled: a
DIAGONAL Gaussian over a reparametrised vector, fitted to the seven fitted
rigs.  What makes that defensible is the reparametrisation, not the density.

The rigs differ in absolute size by a factor of 3.6 — michaels hovers at
~78 rev/s and the small-frame rigs at ~280 rev/s — so a Gaussian over raw
parameters would mostly encode "how big is the drone", and a draw combining
one rig's hover level with another's absolute jitter would be nonsense.
:func:`rig_vector` therefore makes every coordinate dimensionless:

* ``log mean(mu)`` — the one scale coordinate;
* three RELATIVE trims ``(M^T mu / 4)[1:] / mean(mu)`` (roll/pitch/yaw trim as
  a fraction of the hover level; michaels' ~27 % relative spread is the same at
  idle and at cruise, so a ratio is the right invariant);
* ``theta``, ``u_slow``, ``log f0``, ``logit zeta`` and ``log tau_e`` — already
  dimensionless (a time constant does not scale with the drone's size);
* every ``sigma``, including the measurement process' ``sigma_e`` and both
  parts of the per-flight offset (``s_c`` common, ``s_r`` per rotor), as
  ``log(sigma / mean(mu))``, i.e. as a relative fluctuation.

Sampling maps back through the same transform, so a draw is "a drone this big,
fluctuating this much RELATIVE to its size".  Four invariants survive the round
trip by construction: ``mu > 0`` (a positive scale times trims clipped to
+/-30 %, whose worst case is 0.1 x the hover level), ``f0`` and ``tau_e`` inside
their ranges (clipped, because those coordinates are plain logs),
``zeta`` inside ``(ZETA_MIN, ZETA_MAX)`` and ``tau_slow`` inside
``(1/(2 pi f0), TAU_SLOW_MAX_S)`` (both by their sigmoid parametrisations, so
no clipping is needed however wide the Gaussian gets).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments.rps_traj.model import (
    F0_MAX_HZ,
    F0_MIN_HZ,
    N_PARAMS,
    S_FLOOR,
    SIGMA_E_FLOOR,
    TAU_E_MAX_S,
    TAU_E_MIN_S,
    NewFit,
    Params,
    tau_slow_from_u,
    u_from_tau_slow,
    v_from_zeta,
    zeta_from_v,
)
from tracking.rotors import MIXER, NUM_ROTORS

#: Dimension of a rig vector (= the model's parameter count).
RIG_DIM = N_PARAMS

#: Slices of the rig vector, in order.
IDX_LOG_SCALE = 0
IDX_TRIM = slice(1, 4)
IDX_THETA = 4
IDX_U_SLOW = slice(5, 9)
IDX_LOG_SIGMA_SLOW = slice(9, 13)
IDX_LOG_F0 = slice(13, 17)
IDX_LOGIT_ZETA = slice(17, 21)
IDX_LOG_SIGMA_OSC = slice(21, 25)
IDX_LOG_TAU_E = 25
IDX_LOG_SIGMA_E = 26
IDX_LOG_S_C = 27
IDX_LOG_S_R = slice(28, 32)

#: Coordinates that are logs of a positive quantity: their prior width is
#: floored relatively as well as absolutely (see :func:`fit_posterior`).
LOG_DIMS = np.array(
    [IDX_LOG_SCALE, IDX_LOG_TAU_E, IDX_LOG_SIGMA_E, IDX_LOG_S_C]
    + list(range(9, 17))
    + list(range(21, 25))
    + list(range(28, 32))
)

#: Relative trims are clipped to this range when mapping a draw back, which is
#: what keeps every rotor mean positive.
TRIM_CLIP = 0.3

#: Largest per-flight offset std a DRAW may have, as a fraction of its own
#: hover level.  The widest real one is neurobem's 0.19 (its gentle and
#: aggressive flights differ by 42 of 217 rev/s), so half the hover level is
#: already beyond anything measured; without the cap a wide Gaussian tail asks
#: for an offset several times the operating point, which is not a drone — the
#: first round-5 draw set had ``s_c`` = 3267 rev/s against ``mu`` = 130.
OFFSET_REL_MAX = 0.5

#: Floors on a fitted coordinate's std: absolute, and relative to |mean| for
#: the log coordinates.
STD_FLOOR = 0.05
STD_FLOOR_REL = 0.1


def rig_vector(params: Params) -> np.ndarray:
    """``(32,)`` scale-free coordinates of one rig's parameters."""
    scale = float(np.mean(params.mu))
    if scale <= 0.0:
        raise ValueError(f"mean(mu) must be positive, got {scale}")
    modes = (MIXER.T @ params.mu) / NUM_ROTORS
    return np.concatenate(
        [
            [np.log(scale)],
            modes[1:] / scale,
            [params.theta],
            u_from_tau_slow(params.tau_slow, params.f0),
            np.log(params.sigma_slow / scale),
            np.log(params.f0),
            v_from_zeta(params.zeta),
            np.log(params.sigma_osc / scale),
            [
                np.log(params.tau_e),
                np.log(max(params.sigma_e, SIGMA_E_FLOOR) / scale),
                np.log(max(params.s_c, S_FLOOR) / scale),
            ],
            np.log(np.maximum(params.s_r, S_FLOOR) / scale),
        ]
    )


def params_from_rig_vector(v: np.ndarray) -> Params:
    """Inverse of :func:`rig_vector`.

    Trims, ``f0`` and both per-flight offsets are CLIPPED to their ranges
    (their coordinates are plain logs or ratios, so a wide Gaussian tail can
    leave the plausible region); ``tau_slow``, ``tau_e`` and ``zeta`` need no
    clipping because their coordinates are logits of bounded transforms.
    """
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if v.size != RIG_DIM:
        raise ValueError(f"expected a {RIG_DIM}-dim rig vector, got {v.size}")
    scale = float(np.exp(v[IDX_LOG_SCALE]))
    modes = np.concatenate([[1.0], np.clip(v[IDX_TRIM], -TRIM_CLIP, TRIM_CLIP)]) * scale
    f0 = np.clip(np.exp(v[IDX_LOG_F0]), F0_MIN_HZ, F0_MAX_HZ)
    return Params(
        mu=MIXER @ modes,
        theta=float(v[IDX_THETA]),
        tau_slow=tau_slow_from_u(v[IDX_U_SLOW], f0),
        sigma_slow=scale * np.exp(v[IDX_LOG_SIGMA_SLOW]),
        f0=f0,
        zeta=zeta_from_v(v[IDX_LOGIT_ZETA]),
        sigma_osc=scale * np.exp(v[IDX_LOG_SIGMA_OSC]),
        tau_e=float(np.clip(np.exp(v[IDX_LOG_TAU_E]), TAU_E_MIN_S, TAU_E_MAX_S)),
        sigma_e=scale * float(np.exp(v[IDX_LOG_SIGMA_E])),
        s_c=scale * min(float(np.exp(v[IDX_LOG_S_C])), OFFSET_REL_MAX),
        s_r=scale * np.minimum(np.exp(v[IDX_LOG_S_R]), OFFSET_REL_MAX),
    )


@dataclass
class Posterior:
    """Diagonal Gaussian over :func:`rig_vector` coordinates."""

    mean: np.ndarray
    std: np.ndarray
    rigs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float64).reshape(RIG_DIM)
        self.std = np.asarray(self.std, dtype=np.float64).reshape(RIG_DIM)
        self.rigs = tuple(self.rigs)

    def sample(self, rng: np.random.Generator) -> Params:
        """One drone drawn from the posterior."""
        return params_from_rig_vector(self.mean + self.std * rng.standard_normal(RIG_DIM))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "rig_posterior",
            "dim": RIG_DIM,
            "rigs": list(self.rigs),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, allow_nan=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Posterior:
        return cls(mean=payload["mean"], std=payload["std"], rigs=tuple(payload.get("rigs", ())))

    @classmethod
    def from_json(cls, text: str | bytes) -> Posterior:
        return cls.from_dict(json.loads(text))


def informative_mask(params: Params) -> np.ndarray:
    """``(32,)`` True where this rig's coordinate carries information.

    Only the per-flight offsets can be uninformative: a rig with no
    between-flight level difference has ``s = 0``, which :func:`rig_vector` must
    write as ``log(S_FLOOR / scale)`` — a placeholder, not a measurement.
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
    share a level — has ``s = 0``, which :func:`rig_vector` has to write as
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
    "RIG_DIM",
    "STD_FLOOR",
    "STD_FLOOR_REL",
    "TRIM_CLIP",
    "Posterior",
    "fit_posterior",
    "informative_mask",
    "params_from_rig_vector",
    "rig_vector",
]
