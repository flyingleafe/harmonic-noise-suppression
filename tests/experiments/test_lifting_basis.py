import json

import numpy as np
import pytest

from experiments.rps_traj import lifting_basis as lb


def _codes(*lengths):
    return np.repeat(np.arange(4), lengths)


def _runs(starts, n):
    ends = np.r_[starts[1:], n]
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


# Codes-only fixtures shaped like repeated arming/takeoff/landing rigs.
TRAIN = [_codes(30, 45, 400, 11), _codes(29, 65, 300, 40), _codes(28, 105, 350, 260)]
SEGMENTS = [
    (1, [0]),
    (2, [0, 1]),
    (7, [0]),
    (37, [0, 1, 4, 5, 12, 13, 30]),
    (300, [0, 3, 67, 68, 131, 200, 299]),
    (257, [0]),
]


@pytest.mark.parametrize(("n", "starts"), SEGMENTS)
def test_tree_is_exactly_invertible_on_odd_singleton_and_mixed_runs(n, starts):
    tree = lb.Tree(n, starts)
    x = np.random.default_rng(n).normal(size=(3, n, 4)) * 50
    z = tree.analyse(x)
    assert z.shape == x.shape
    np.testing.assert_allclose(tree.synth(z), x, rtol=0, atol=1e-9)
    np.testing.assert_allclose(tree.analyse(tree.synth(x)), x, rtol=0, atol=1e-9)
    assert tree.dc.sum() == len(starts) and tree.dc[starts].all()


@pytest.mark.parametrize(
    "schedule",
    [_codes(0, 7, 33, 1), _codes(0, 0, 1, 0), _codes(0, 0, 0, 5), _codes(3, 0, 9, 0), TRAIN[1]],
)
def test_cropped_schedule_designs_invert(schedule):
    for structure in (lb.Structure(), lb.Structure.from_training(TRAIN)):
        d = structure.design(schedule)
        x = np.random.default_rng(1).normal(size=(len(schedule), 4))
        np.testing.assert_allclose(d.tree.synth(d.tree.analyse(x)), x, rtol=0, atol=1e-9)
        assert all(len(m) == len(schedule) for m in d.mean)


def test_unit_determinant_on_time_axis():
    n, starts = 37, [0, 1, 4, 5, 12, 13, 30]
    A = lb.Tree(n, starts).analyse(np.eye(n))
    assert abs(abs(np.linalg.det(A)) - 1.0) < 1e-9


def test_dc_is_the_arithmetic_mean_of_each_phase():
    schedule = _codes(5, 7, 20, 3)
    d = lb.Structure().design(schedule)
    x = np.random.default_rng(2).normal(size=(len(schedule), 4)) + 9.0
    z = d.tree.analyse(x)
    starts = np.flatnonzero(np.r_[True, np.diff(schedule) != 0])
    assert np.array_equal(np.flatnonzero(d.dc), starts)
    for a, b in _runs(starts, len(schedule)):
        np.testing.assert_allclose(z[a], x[a:b].mean(axis=0), rtol=1e-12)
        assert d.support[a] == b - a and d.level[a] == 0
    detail = ~d.dc
    assert (d.support[detail] >= 2).all()
    np.testing.assert_array_equal(d.level[detail], np.ceil(np.log2(d.support[detail])).astype(int))


@pytest.mark.parametrize(("n", "starts"), SEGMENTS)
def test_phase_constants_change_only_dc(n, starts):
    tree = lb.Tree(n, starts)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(n, 2))
    shift = np.zeros((n, 2))
    offsets = rng.normal(size=(len(starts), 2)) * 30
    for (a, b), off in zip(_runs(np.asarray(starts), n), offsets, strict=True):
        shift[a:b] = off
    z0, z1 = tree.analyse(x), tree.analyse(x + shift)
    np.testing.assert_allclose(z1[~tree.dc], z0[~tree.dc], rtol=0, atol=1e-9)
    np.testing.assert_allclose(z1[tree.dc] - z0[tree.dc], offsets, rtol=0, atol=1e-9)


@pytest.mark.parametrize(("n", "starts"), SEGMENTS)
def test_linear_phases_use_only_dc_and_top_split(n, starts):
    tree = lb.Tree(n, starts)
    rng = np.random.default_rng(4)
    runs = _runs(np.asarray(starts), n)
    length = np.zeros(n, int)
    x = np.zeros((n, 1))
    for a, b in runs:
        length[a:b] = b - a
        x[a:b, 0] = rng.normal() * 5 + rng.normal() * np.arange(a, b)
    top = ~tree.dc & (tree.support == length)
    z = tree.analyse(x)
    np.testing.assert_allclose(z[~(tree.dc | top)], 0.0, rtol=0, atol=1e-8)
    # Interpolation: a DC plus one top split per phase synthesizes a straight line.
    z = np.zeros((n, 1))
    z[tree.dc | top] = rng.normal(size=(np.count_nonzero(tree.dc | top), 1))
    y = tree.synth(z)[:, 0]
    for a, b in runs:
        np.testing.assert_allclose(np.diff(y[a:b], 2), 0.0, rtol=0, atol=1e-9)


@pytest.mark.parametrize(
    "train",
    [
        [_codes(0, 0, 500, 0), _codes(0, 0, 321, 0)],
        [_codes(152, 0, 900, 0)],
        [_codes(0, 16, 182, 0), _codes(0, 0, 296, 81), _codes(0, 0, 400, 0)],
    ],
)
def test_without_repeated_arming_or_landing_the_structure_is_compact(train):
    s = lb.Structure.from_training(train)
    assert s.kind == "compact" and s.n_params == 0 and s.n_groups == 1 and s.n_mean == 4
    for schedule in train:
        d = s.design(schedule)
        assert [m.shape[1] for m in d.mean] == [1, 1, 1, 1]
        assert (d.group == 0).all()


def test_airborne_start_has_no_takeoff_and_no_false_drives():
    for s in (lb.Structure(), lb.Structure.from_training(TRAIN)):
        d = s.design(_codes(0, 0, 257, 0))
        assert (d.group == 0).all() and d.cruise.all()
        np.testing.assert_array_equal(d.drive, d.level == lb.DRIVE_LEVEL)
        assert not (d.drive & d.dc).any()
        names = [name for name, w in s.layout(0) for _ in range(w)]
        hover = names.index("hover") if "hover" in names else 0
        # Only the constant level/hover column, entirely in the airborne DC.
        expected = np.zeros_like(d.mean[0])
        expected[0, hover] = 1.0
        np.testing.assert_allclose(d.mean[0], expected, rtol=0, atol=1e-12)


def test_single_instantaneous_arming_steps_from_rest():
    schedule = _codes(152, 0, 900, 0)
    s = lb.Structure.from_training([schedule])
    d = s.design(schedule)
    assert d.mean[0][0, 0] == 0.0 and 0.99 < d.mean[0][152, 0] < 1.0
    np.testing.assert_allclose(d.tree.synth(d.mean[0])[:152], 0.0, rtol=0, atol=1e-12)
    assert not d.cruise[d.block[:152]].any() and not d.drive[:152].any()
    # Even a radically different arming speed profile must not move the
    # operating-level coefficient. Given the observed phase mean the details
    # of the unit response have no further dependence on that parameter.
    np.testing.assert_allclose(d.mean[0][~d.dc], 0.0, atol=1e-12)
    rng = np.random.default_rng(71)
    arbitrary = rng.normal(size=(len(schedule), 4)) + np.array([140.0, 149.0, 152.0, 145.0])
    np.testing.assert_allclose(d.tree.synth(d.tree.analyse(arbitrary)), arbitrary, atol=1e-10)


def test_repeated_events_resolve_impulse_plateau_touchdown_and_rundown():
    s = lb.Structure.from_training(TRAIN)
    assert s.kind == "events" and set(s.events) == {"on", "to", "la"}
    assert s.gates == ("idle", "hover", "shelf") and s.n_params == 1
    holds = [45 / 25, 65 / 25, 105 / 25]
    assert s.support == pytest.approx(np.clip(np.median(holds) - 0.24, 0.48, 5.2))
    schedule = TRAIN[2]
    on, take, land = 28, 133, 483
    d = s.design(schedule)
    assert d.group.max() < s.n_groups == len(s.bins)
    impulse, plateau = d.group[on + 1], d.group[on + 100]
    touchdown, rundown = d.group[land + 1], d.group[land + 150]
    assert len({0, impulse, plateau, touchdown, rundown}) == 5
    assert d.group[take + 1] not in (0, impulse, plateau, touchdown, rundown)
    assert d.group[300] == 0
    assert d.drive.any() and not (d.drive & d.dc).any()
    assert not d.drive[: take + 96].any() and not d.drive[land - 16 :].any()
    widths = [sum(w for _, w in s.layout(m)) for m in range(4)]
    assert [m.shape[1] for m in d.mean] == widths and s.n_mean == sum(widths)
    for m in range(4):
        assert s.penalty(m).shape == (widths[m], widths[m])
        # No intercept: every column vanishes while the motors are off.
        np.testing.assert_allclose(d.tree.synth(d.mean[m])[:on], 0.0, rtol=0, atol=1e-12)


def test_structure_round_trips_through_json():
    for s in (lb.Structure(), lb.Structure.from_training(TRAIN)):
        data = json.loads(json.dumps(s.to_dict()))
        assert data["n_params"] == s.n_params
        r = lb.Structure.from_dict(data)
        assert r == s
        a, b = s.design(TRAIN[0]), r.design(TRAIN[0])
        for m in range(4):
            np.testing.assert_array_equal(a.mean[m], b.mean[m])
        np.testing.assert_array_equal(a.group, b.group)
        np.testing.assert_array_equal(a.drive, b.drive)
