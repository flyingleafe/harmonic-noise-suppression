"""The fitted baseline must be reproducible, improvable and on-protocol."""

from __future__ import annotations

import json

import numpy as np

from data_processing.rps_synthesis import (
    DroneProfile,
    ManeuverModeParams,
    generate_intermittent,
)
from experiments.rps_traj.baseline import (
    N_REP_FINAL,
    BaselineFit,
    fit_baseline,
    objective_from_discrepancy,
    starting_profile,
)
from experiments.rps_traj.data import RATE_HZ, Flight
from experiments.rps_traj.stats import compute_stats, discrepancy, stats_from_samples

#: A "truth" airframe that neither stock profile matches: hovers well above
#: both, maneuvers three times as often for a third as long, and is markedly
#: more sluggish. Moment-matching the trims cannot fix any of that, so a fit
#: that only moved the trims would show no improvement.
TRUTH = DroneProfile(
    common=ManeuverModeParams(
        trim=105.0, cruise_std=1.4, maneuver_std=9.0, rate_hz=0.45, mean_maneuver_s=0.25
    ),
    roll=ManeuverModeParams(
        trim=-2.5, cruise_std=0.5, maneuver_std=2.0, rate_hz=0.40, mean_maneuver_s=0.25
    ),
    pitch=ManeuverModeParams(
        trim=3.0, cruise_std=0.5, maneuver_std=2.5, rate_hz=0.40, mean_maneuver_s=0.25
    ),
    yaw=ManeuverModeParams(
        trim=7.0, cruise_std=0.8, maneuver_std=3.0, rate_hz=0.35, mean_maneuver_s=0.35
    ),
    motor_tau=0.7,
    cruise_tau=1.5,
    rps_min=0.0,
    rps_max=200.0,
)


def _truth_flights(n_flights: int = 3, duration_s: float = 60.0) -> list[Flight]:
    return [
        Flight(
            rig="truth",
            flight=f"f{i}",
            fs=RATE_HZ,
            t0=0.0,
            rps=generate_intermittent(
                duration_s, RATE_HZ, profile=TRUTH, rng=np.random.default_rng(100 + i)
            ),
            source="synthetic",
        )
        for i in range(n_flights)
    ]


def _fit_for_tests() -> BaselineFit:
    return BaselineFit(
        rig="truth",
        profile=TRUTH,
        aggressiveness=1.0,
        idle_rps=42.0,
        objective=0.5,
        n_evals=7,
    )


def test_json_round_trip_reproduces_identical_samples() -> None:
    fit = _fit_for_tests()
    text = fit.to_json()
    assert isinstance(text, str)
    restored = BaselineFit.from_json(text)
    assert BaselineFit.from_dict(json.loads(text)) == restored

    assert restored.rig == fit.rig
    assert restored.profile == fit.profile
    assert (restored.idle_rps, restored.aggressiveness, restored.n_evals) == (42.0, 1.0, 7)

    a = fit.sampler()(1500, np.random.default_rng(3))
    b = restored.sampler()(1500, np.random.default_rng(3))
    assert np.array_equal(a, b)

    fa = fit.full_flight(90.0, RATE_HZ, np.random.default_rng(11))
    fb = restored.full_flight(90.0, RATE_HZ, np.random.default_rng(11))
    assert np.array_equal(fa, fb)


def test_sampler_honours_the_frozen_protocol() -> None:
    fit = _fit_for_tests()
    for fs, n in ((RATE_HZ, 1234), (50.0, 777)):
        sample = fit.sampler(fs=fs)
        out = sample(n, np.random.default_rng(0))
        assert out.shape == (4, n)
        assert np.all(np.isfinite(out))
        assert out.min() >= TRUTH.rps_min and out.max() <= TRUTH.rps_max
        # Right rate, not just the right length: the ACF at a fixed TIME lag
        # must be scale-free, and the level must be the hover trim.
        assert abs(float(out.mean()) - TRUTH.common.trim) < 5.0

    # All randomness comes from the rng: same generator state, same samples.
    once = fit.sampler()(600, np.random.default_rng(5))
    twice = fit.sampler()(600, np.random.default_rng(5))
    assert np.array_equal(once, twice)


def test_fit_baseline_improves_on_its_starting_profile() -> None:
    flights = _truth_flights()
    real = compute_stats(flights)
    durations = [f.duration_s for f in flights]

    start = BaselineFit(
        rig="truth",
        profile=starting_profile(flights),
        aggressiveness=1.0,
        idle_rps=0.0,
        objective=float("nan"),
        n_evals=0,
    )
    start_objective = objective_from_discrepancy(
        discrepancy(stats_from_samples(start.sampler(), durations, n_rep=N_REP_FINAL, seed=0), real)
    )

    fit = fit_baseline(flights, "truth", seed=0, n_evals=80)

    assert fit.rig == "truth"
    assert fit.aggressiveness == 1.0
    assert fit.n_evals <= 80
    assert fit.objective < start_objective
    # The returned objective is the fit's own, re-measured on the protocol.
    remeasured = objective_from_discrepancy(
        discrepancy(stats_from_samples(fit.sampler(), durations, n_rep=N_REP_FINAL, seed=0), real)
    )
    assert remeasured == fit.objective


def test_full_flight_spans_ground_to_hover_at_the_fitted_idle() -> None:
    fit = _fit_for_tests()  # idle_rps = 42, hover trim = 105
    flight = fit.full_flight(120.0, RATE_HZ, np.random.default_rng(2))

    assert flight.shape == (4, 12000)
    level = flight.mean(axis=0)
    # Ground at both ends; the tail only decays with motor_tau, so it is
    # "off" relative to idle rather than exactly zero.
    assert level[0] == 0.0
    assert level[-1] < 0.2 * fit.idle_rps
    assert level.max() > 0.9 * TRUTH.common.trim  # reaches hover

    # The warm-up plateau sits at idle_rps, not at the generator's random
    # default fraction of hover: take the levels strictly between ground and
    # hover that are held for a while.
    plateau = level[(level > 0.2 * fit.idle_rps) & (level < 0.8 * TRUTH.common.trim)]
    assert abs(float(np.median(plateau)) - fit.idle_rps) < 0.15 * fit.idle_rps

    # A request too short for the default phase ranges still produces a flight.
    short = fit.full_flight(25.0, RATE_HZ, np.random.default_rng(4))
    assert short.shape == (4, 2500)
    assert short.mean(axis=0).max() > 0.9 * TRUTH.common.trim
