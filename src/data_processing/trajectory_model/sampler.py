"""A fitted rig: its parameters, its ESC limits, and how to sample from it.

:class:`NewFit` is the on-disk record one exact-likelihood MAP fit produces
(``results/rps_traj/fits/new/<rig>.json``, published as the ``rps-traj-fits``
dataset).  It carries the 32 model parameters, the rig's warm-up idle level,
the ESC floor and ceiling the sampler clamps to, and the fit's provenance, so
the JSON round-trips byte-for-byte between the fitting campaign
(:mod:`experiments.rps_traj.model`, which writes it) and the training streams
(:mod:`data_processing.trajectory_model.source`, which reads it).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from data_processing.trajectory_model.flight import FlightPhaseRanges, wrap_airborne
from data_processing.trajectory_model.params import RATE_HZ, Params, Sampler
from tracking.rotors import NUM_ROTORS


@dataclass
class NewFit:
    """A fitted rig: parameters plus the fit's provenance.

    ``rps_min``/``rps_max`` are the rig's real airborne extremes over all its
    flights, and the sampler clamps to them: a real ESC has a floor and a
    ceiling, and a Gaussian model does not know that.
    """

    rig: str
    params: Params
    idle_rps: np.ndarray
    rps_min: float = 0.0
    rps_max: float = float("inf")
    fs: float = RATE_HZ
    fit_rate_hz: float = RATE_HZ
    nll: float = float("nan")
    n_iter: int = 0
    wall_s: float = 0.0
    n_blocks: int = 0
    n_scored: int = 0
    n_flights: int = 0
    n_restarts: int = 0

    def __post_init__(self) -> None:
        self.idle_rps = np.asarray(self.idle_rps, dtype=np.float64).reshape(NUM_ROTORS)

    @property
    def clip(self) -> tuple[float, float]:
        return (float(self.rps_min), float(self.rps_max))

    def sampler(self, fs: float = RATE_HZ, *, antithetic_offsets: bool = False) -> Sampler:
        """The frozen sampler protocol (contract 3), clamped to the rig's ESC
        floor and ceiling."""
        return self.params.sampler(fs=fs, antithetic_offsets=antithetic_offsets, clip=self.clip)

    def full_flight(
        self,
        duration_s: float,
        fs: float = RATE_HZ,
        rng: np.random.Generator | int | None = None,
        phases: FlightPhaseRanges | None = None,
    ) -> np.ndarray:
        """``(4, n)`` whole flight: ground, spin-up, idle, take-off, the
        stationary airborne process, landing, spin-down, ground."""
        generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

        def airborne(n: int, rng_: np.random.Generator, fs: float = fs) -> np.ndarray:
            return self.params.sample_airborne(n, rng_, fs=fs, clip=self.clip)

        return wrap_airborne(airborne, duration_s, fs, generator, idle=self.idle_rps, phases=phases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "new",
            "rig": self.rig,
            "params": self.params.to_dict(),
            "idle_rps": self.idle_rps.tolist(),
            "rps_min": self.rps_min,
            "rps_max": self.rps_max,
            "fs": self.fs,
            "fit_rate_hz": self.fit_rate_hz,
            "nll": self.nll,
            "n_iter": self.n_iter,
            "wall_s": self.wall_s,
            "n_blocks": self.n_blocks,
            "n_scored": self.n_scored,
            "n_flights": self.n_flights,
            "n_restarts": self.n_restarts,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, allow_nan=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NewFit:
        return cls(
            rig=payload["rig"],
            params=Params.from_dict(payload["params"]),
            idle_rps=payload["idle_rps"],
            rps_min=float(payload.get("rps_min", 0.0)),
            rps_max=float(payload.get("rps_max", float("inf"))),
            fs=float(payload.get("fs", RATE_HZ)),
            fit_rate_hz=float(payload.get("fit_rate_hz", RATE_HZ)),
            nll=float(payload.get("nll", float("nan"))),
            n_iter=int(payload.get("n_iter", 0)),
            wall_s=float(payload.get("wall_s", 0.0)),
            n_blocks=int(payload.get("n_blocks", 0)),
            n_scored=int(payload.get("n_scored", 0)),
            n_flights=int(payload.get("n_flights", 0)),
            n_restarts=int(payload.get("n_restarts", 0)),
        )

    @classmethod
    def from_json(cls, text: str | bytes) -> NewFit:
        return cls.from_dict(json.loads(text))


__all__ = ["NewFit"]
