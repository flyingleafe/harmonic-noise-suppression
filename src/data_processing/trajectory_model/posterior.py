"""The rig posterior: a drone drawn in scale-free coordinates.

A rig's parameters are mapped to 32 SCALE-FREE coordinates — a log hover level,
three relative trims, and every time constant / std expressed as a log ratio to
that level — so a Gaussian over them is a statement about drones rather than
about one airframe's size.  :func:`params_from_rig_vector` maps a draw back,
clipping the coordinates whose plain log has an implausible tail (trims, ``f0``
and the per-flight offsets), which is what keeps every rotor mean positive and
every offset smaller than half the hover level.

This module holds the coordinates and the draw; the FIT of the Gaussian to the
campaign's seven rigs (which coordinates are informative, the std floors) lives
in :mod:`experiments.rps_traj.posterior`, and the shipped posterior is
``results/rps_traj/posterior.json``, published in the ``rps-traj-fits`` dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from data_processing.trajectory_model.params import (
    F0_MAX_HZ,
    F0_MIN_HZ,
    N_PARAMS,
    S_FLOOR,
    SIGMA_E_FLOOR,
    TAU_E_MAX_S,
    TAU_E_MIN_S,
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


__all__ = [
    "IDX_LOGIT_ZETA",
    "IDX_LOG_F0",
    "IDX_LOG_S_C",
    "IDX_LOG_S_R",
    "IDX_LOG_SCALE",
    "IDX_LOG_SIGMA_E",
    "IDX_LOG_SIGMA_OSC",
    "IDX_LOG_SIGMA_SLOW",
    "IDX_LOG_TAU_E",
    "IDX_THETA",
    "IDX_TRIM",
    "IDX_U_SLOW",
    "OFFSET_REL_MAX",
    "RIG_DIM",
    "TRIM_CLIP",
    "Posterior",
    "params_from_rig_vector",
    "rig_vector",
]
