import numpy as np
import pytest

from experiments.rps_traj.regulated_bridge import RegulatedBridge, fit


@pytest.mark.parametrize("n_flights", [1, 3])
def test_compact_law_replays_flight_with_altered_dwell(n_flights):
    rng = np.random.default_rng(61)
    schedule = np.repeat(np.arange(4), [6, 95, 152, 45])
    mu = 15.0 * (schedule == 1) + 83.0 * (schedule == 2)
    records = [
        (
            np.maximum(
                0,
                mu + i * (schedule == 2) + rng.normal(size=(4, len(mu))) * (0.04 + (schedule == 2)),
            ),
            schedule,
        )
        for i in range(n_flights)
    ]
    model = fit(records, max_iter=3)
    law = model.coefficients()
    reloaded = RegulatedBridge.from_coefficients(law)
    longer = np.repeat(np.arange(4), [8, 175, 190, 80])
    speeds = reloaded.sample(longer, 12, np.random.default_rng(78))
    np.testing.assert_array_equal(speeds, model.sample(longer, 12, np.random.default_rng(78)))
    assert law.size == model.n_params < 277
    assert speeds.min() >= 0 and abs(speeds[:, :, 125:155].mean() - 15) < 0.5
    assert speeds[:, :, 125:155].std() < speeds[:, :, 310:340].std()


def test_airborne_record_does_not_replay_takeoff_drive():
    from experiments.rps_traj.population_bridge import MEAN_GRIDS

    rng = np.random.default_rng(3)
    cruise = np.full(160, 2, dtype=np.int8)
    speed = 70 + rng.normal(0, 0.4, (4, len(cruise)))
    model = fit([(speed, cruise)], max_iter=1)
    changed = RegulatedBridge.from_coefficients(model.coefficients().copy())
    n_on = len(MEAN_GRIDS[0][0]) - 2
    n_to = len(MEAN_GRIDS[0][1]) - 2
    changed.p["beta"][0][n_on : n_on + n_to] += 25.0

    def draw(m, schedule):
        return m.sample(schedule, 1, np.random.default_rng(17))

    np.testing.assert_array_equal(draw(model, cruise), draw(changed, cruise))
    real_takeoff = np.r_[np.ones(30, dtype=np.int8), cruise]
    assert np.max(np.abs(draw(model, real_takeoff) - draw(changed, real_takeoff))) > 10.0
