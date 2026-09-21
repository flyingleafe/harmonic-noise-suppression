"""Windowing a cached whole flight at the audio rate.

A synthetic rotor-noise pool does not draw a fresh trajectory per training
window: a whole flight is generated once at a low rate (``flight_fs``, a few
hundred hertz — a trajectory has no content above it), held for
``flight_reuse`` windows, and each window is a uniformly placed slice of it
interpolated onto the audio grid. Successive windows therefore visit the
ground, warm-up, takeoff, cruise and landing phases in proportion to their
durations, and the expensive part — generating or sampling the flight — is paid
once per ``flight_reuse`` windows instead of once per sample.

This is the machinery, shared by every pool that renders from a flight:
:class:`~data_processing.stochastic_rotor_noise.StochasticNoisePool` (``kind:
stochastic``) and :class:`~data_processing.noise_v2_pool.NoiseV2Pool` (``kind:
noise_v2``). The pool owns the POLICY — which generator makes the flight, what
else is redrawn when a new one is made — and calls
:func:`make_flight_cache` / :func:`window_flight` for the mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["FlightCache", "make_flight_cache", "window_flight"]


@dataclass
class FlightCache:
    """One whole flight at ``flight_fs``, with the windows drawn from it."""

    rps: np.ndarray
    t_low: np.ndarray
    uses: int = 0
    #: The flight's own hover speed (its 90th percentile). The amplitude law is
    #: written against THIS, not against a fixed 80 rev/s, so the level says
    #: "this aircraft is at such a fraction of its own hover" instead of "the
    #: speed is such a number". Without it a wide speed range would make the
    #: level a giveaway for the absolute speed — a shortcut, and a false one,
    #: since a small fast drone is not quieter than a big slow one.
    hover: float = 80.0
    #: The reference level this whole flight is recorded at, drawn once when the
    #: flight is made and held for every window of it — only when
    #: ``level_per_flight`` is on. A real recording has ONE gain: within it,
    #: loudness tracks rotor speed closely (Michael's two recordings couple
    #: level to speed at Spearman +0.73 / +0.48 with only 1.2 / 3.2 dB of
    #: scatter), and different recordings sit at very different absolute levels.
    #: Redrawing the level per WINDOW destroys that: it keeps the across-flight
    #: spread but throws the same spread INSIDE each flight, which is what left
    #: the synthetic streams at 7.4 to 12.8 dB of scatter.
    level: float | None = None


def make_flight_cache(
    flight: np.ndarray, flight_fs: float, *, level: float | None = None
) -> FlightCache:
    """Cache ``(R, N)`` low-rate rotor speeds with their time base and hover."""
    return FlightCache(
        rps=flight,
        t_low=np.arange(flight.shape[1]) / flight_fs,
        hover=float(np.percentile(flight, 90.0)),
        level=level,
    )


def window_flight(
    cache: FlightCache, rng: np.random.Generator, duration_s: float, sample_rate: int
) -> np.ndarray:
    """``(R, T)`` audio-rate rotor speeds: a uniform slice of the cached flight.

    The start is uniform over the flight's usable span, so the window's phase
    is not a function of how many windows were drawn before it, and the low-rate
    track is interpolated per rotor onto the audio grid.
    """
    n_samples = int(round(duration_s * sample_rate))
    flight, t_low = cache.rps, cache.t_low
    max_start = max(0.0, float(t_low[-1]) - duration_s)
    start_s = float(rng.uniform(0.0, max_start)) if max_start > 0 else 0.0
    t_win = start_s + np.arange(n_samples) / sample_rate
    return np.stack([np.interp(t_win, t_low, flight[r]) for r in range(flight.shape[0])])
