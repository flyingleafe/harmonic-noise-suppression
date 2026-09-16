"""Wrap a stationary airborne rotor-speed process into a WHOLE flight.

The trajectory model is a model of the *airborne* regime — that is what the
campaign's frozen statistics score, because the airborne rule throws everything
else away.  A training stream, however, needs a plausible whole flight: ground
silence, spin-up, a warm-up idle, a take-off ramp, the airborne process, a
landing ramp, spin-down, ground silence again.

:func:`wrap_airborne` is that wrapper, shared by every fit class so the fitted
model and the incumbent synthesiser produce the same kind of envelope and differ
only in the airborne part.  Phase durations come from
:class:`data_processing.rps_synthesis.FlightPhaseRanges` (the incumbent
scaffold, loosely calibrated to the DREGON/Michael's recordings); the ramps are
raised cosines, which is all the smoothness a stream needs and keeps the whole
thing a deterministic function of ``rng``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from data_processing.rps_synthesis import FlightPhaseRanges
from data_processing.trajectory_model.params import RATE_HZ

#: Fraction of the total a wrapped flight keeps for the airborne phase: if the
#: sampled fixed phases would leave less, they are all scaled down together.
MIN_AIRBORNE_FRAC = 0.3

#: An airborne sampler: ``(n_samples, rng, fs) -> (n_rotors, n_samples)``.
AirborneSampler = Callable[..., np.ndarray]


def _cosine_ramp(n: int, start: np.ndarray, stop: np.ndarray) -> np.ndarray:
    """``(n_rotors, n)`` raised-cosine ramp from ``start`` to ``stop``."""
    if n <= 0:
        return np.zeros((np.size(start), 0))
    u = 0.5 * (1.0 - np.cos(np.pi * (np.arange(n) + 0.5) / n))
    return start[:, None] + (stop - start)[:, None] * u[None, :]


def wrap_airborne(
    airborne: AirborneSampler,
    duration_s: float,
    fs: float = RATE_HZ,
    rng: np.random.Generator | int | None = None,
    *,
    idle: np.ndarray | float,
    phases: FlightPhaseRanges | None = None,
) -> np.ndarray:
    """``(n_rotors, n)`` whole flight of ``duration_s`` seconds at ``fs``.

    Args:
        airborne: the model's stationary sampler, called as
            ``airborne(n, rng, fs=fs)`` (the frozen sampler protocol plus the
            rate) and expected to return ``(n_rotors, n)`` rev/s.
        duration_s: total flight length, ground to ground.
        fs: sample rate (Hz).
        rng: generator or seed; all randomness, including the phase durations,
            comes from it.
        idle: per-rotor (or scalar) warm-up/standby level in rev/s.
        phases: duration ranges; defaults to :class:`FlightPhaseRanges`.

    Phase order is ``ground -> spin-up -> idle -> take-off -> airborne ->
    landing -> spin-down -> ground``; the take-off and landing ramps join the
    idle level to the airborne process' own first and last sample, so there is
    no step at either seam.  Output is clipped at 0 (a rotor cannot spin
    backwards) and the endpoints are exactly 0.
    """
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    ranges = phases or FlightPhaseRanges()
    total = float(duration_s)
    if total <= 0.0:
        raise ValueError("duration_s must be positive")

    names = (
        "pre_ground_s",
        "spinup_s",
        "warmup_s",
        "takeoff_s",
        "landing_s",
        "spindown_s",
        "post_ground_s",
    )
    fixed = np.array([generator.uniform(*getattr(ranges, name)) for name in names])
    budget = (1.0 - MIN_AIRBORNE_FRAC) * total
    if fixed.sum() > budget:
        fixed *= budget / fixed.sum()
    pre_g, spinup, warmup, takeoff, landing, spindown, post_g = fixed

    n = int(round(total * fs))
    counts = [int(round(d * fs)) for d in (pre_g, spinup, warmup, takeoff)]
    tail = [int(round(d * fs)) for d in (landing, spindown, post_g)]
    n_air = n - sum(counts) - sum(tail)
    if n_air <= 0:  # pragma: no cover - guarded by the budget rescale above
        raise ValueError(f"duration {total:.1f}s leaves no airborne samples")

    air = np.asarray(airborne(n_air, generator, fs=fs), dtype=np.float64)
    n_rotors = air.shape[0]
    idle_level = np.broadcast_to(np.asarray(idle, dtype=np.float64), (n_rotors,)).copy()
    zero = np.zeros(n_rotors)

    pieces = [
        np.zeros((n_rotors, counts[0])),
        _cosine_ramp(counts[1], zero, idle_level),
        np.repeat(idle_level[:, None], counts[2], axis=1),
        _cosine_ramp(counts[3], idle_level, air[:, 0]),
        air,
        _cosine_ramp(tail[0], air[:, -1], idle_level),
        _cosine_ramp(tail[1], idle_level, zero),
        np.zeros((n_rotors, tail[2])),
    ]
    return np.clip(np.concatenate(pieces, axis=1), 0.0, None)


__all__ = ["MIN_AIRBORNE_FRAC", "AirborneSampler", "FlightPhaseRanges", "wrap_airborne"]
