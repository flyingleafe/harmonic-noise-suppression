"""Bridge conditional law: marginalization, dependency ordering and rollout."""

from itertools import product

import numpy as np

from experiments.rps_traj import bridge_program as bp


def test_actions_integrate_all_paths_and_reset_at_cruise_entry():
    params = [0.4, 0.3, 0.8]
    A = bp._transition(params)
    em = np.random.default_rng(17).normal(size=(6, 5))
    cruise = np.array([False, False, True, True, True, True])
    ll, occ, edges = bp._fb(em, cruise, params)
    mass = 0.0
    expected = np.zeros_like(em)
    ec = np.zeros_like(A)
    for tail in product(range(5), repeat=3):
        path = np.array([0, 0, 0, *tail])
        w = A[path[2:-1], path[3:]].prod() * np.exp(em[np.arange(6), path].sum())
        mass += w
        expected[np.arange(6), path] += w
        np.add.at(ec, (path[2:-1], path[3:]), w)
    np.testing.assert_allclose(ll, np.log(mass), atol=1e-12)
    np.testing.assert_allclose(occ, expected / mass, atol=1e-12)
    np.testing.assert_allclose(edges, ec / mass, atol=1e-12)


def test_every_sample_drawn_after_its_bridge_parents():
    d = bp.design(np.repeat(np.arange(4), [7, 90, 121, 18]))
    inverse_order = np.argsort(d.order)
    assert np.array_equal(np.sort(d.order), np.arange(len(d.order)))
    for parent in (d.prev, d.right):
        assert np.all(inverse_order[1:] > inverse_order[parent[1:]])


def test_joint_fit_monotone_and_reproducible_with_source_specific_levels():
    rng = np.random.default_rng(842)
    schedule = np.repeat(np.arange(4), [5, 125, 250, 90])
    tracks = [
        (
            np.maximum(
                0,
                (schedule == 1) * (13 + 8 * j)
                + (schedule == 2) * (31 + 8 * j)
                + rng.normal(0, 0.25 + 2 * j, (4, len(schedule))),
            ),
            schedule,
        )
        for j in range(3)
    ]
    m = bp.fit(tracks, measured=[True, True, False], max_iter=3)
    diffs = np.diff(m.history)
    assert len(diffs) and np.all(diffs >= 0) and m.n_params < 277
    stored = bp.BridgeModel.from_dict(m.to_dict())
    samples = m.sample(schedule, 16, np.random.default_rng(19))
    np.testing.assert_array_equal(samples, stored.sample(schedule, 16, np.random.default_rng(19)))
    assert np.isfinite(samples).all() and samples.min() >= 0
    idle = np.median(samples[:, :, 94:126], axis=(1, 2))
    assert idle.max() < 26 and idle.min() > 10 and np.ptp(idle) > 4
