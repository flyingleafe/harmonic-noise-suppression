"""Per-drone comparator schedule and physical initialization checks."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.rps_traj import per_drone_baselines as pdb
from experiments.rps_traj.phases import PHASES, phase_labels


def test_align_pads_rest_only_where_allowed() -> None:
    x = np.ones((4, 5))
    y, front, tail, missing = pdb._align(x, -2, 9, rest_tail=True)
    assert y[0].tolist() == [0, 0, 1, 1, 1, 1, 1, 0, 0]
    assert (front, tail, missing) == (2, 2, 0)
    y, front, tail, missing = pdb._align(x, 1, 6, rest_tail=False)
    assert np.isnan(y[:, 4:]).all() and (y[:, :4] == 1).all()
    assert (front, tail, missing) == (0, 0, 2)


def test_marks_are_index_only_with_training_hover() -> None:
    payload = {
        "fs": 25.0,
        "n": 100,
        "starts_ground": True,
        "ends_ground": False,
        "motors_on": 5,
        "standby": [10, 20],
        "takeoff": 20,
        "arrival": 30,
        "landing": None,
        "motors_off": None,
    }
    marks, unmarked = pdb._marks(payload, 100, {}, 50.0)
    assert not unmarked and marks.hover == 50.0 and marks.standby == (10, 20)
    assert marks.standby_level is None and marks.hover_in is None
    lab = phase_labels(marks)
    assert lab[-1] == PHASES.index("cruise") and lab[12] == PHASES.index("standby")
    with pytest.raises(ValueError, match="speed levels"):
        pdb._marks({**payload, "hover_in": [1, 2, 3, 4]}, 100, {}, 50.0)


def test_unmarked_target_is_airborne_only_without_events() -> None:
    marks, unmarked = pdb._marks(None, 40, {"motors_on": None}, 60.0)
    assert unmarked and not marks.starts_ground and not marks.ends_ground
    assert (phase_labels(marks) == PHASES.index("cruise")).all()
    with pytest.raises(ValueError):
        pdb._marks(None, 40, {"landing": 30}, 60.0)


def test_cropped_airborne_var_starts_in_stationary_flight():
    from experiments.rps_traj.data import Flight

    rng = np.random.default_rng(7)
    x = np.ones((4, 2500)) * 80
    noise = rng.normal(0, 0.15, x.shape)
    for i in range(1, x.shape[1]):
        x[:, i] = 80 + 0.98 * (x[:, i - 1] - 80) + noise[:, i]
    _, generate = pdb.fit_phasevar(
        [Flight("toy", "train", 100.0, 0.0, x, "measured")], "toy", "phasevar2"
    )
    marks, unmarked = pdb._marks(None, 80, {}, 80.0)
    target = pdb.Target(
        0,
        "validation",
        80,
        dict.fromkeys(("motors_on", "takeoff", "landing")),
        marks,
        [19],
        unmarked,
    )
    draws, _ = generate(target, 80.0, np.random.default_rng(19))
    assert np.all(np.abs(np.mean(draws[:, :, 0], axis=0) - 80) < 3)
