import numpy as np

from experiments.rps_traj import bridge_program as bp
from experiments.rps_traj import population_bridge as pb


def test_crossing_scale_boundary_preserves_standardized_error():
    c = np.repeat(np.arange(4), [2, 32, 150, 5])
    d, bins = pb.design(c)
    # Outside cruise the preceding disturbance must be transported even if
    # the prescribed drive's observation scale changes by orders of magnitude.
    boundaries = np.flatnonzero((np.diff(bins) != 0) & (d.phase[1:] != 4)) + 1
    assert len(boundaries) > 4
    assert np.array_equal(d.prev[boundaries], boundaries - 1)
    w = np.linspace(10.0, 20.0, len(c))[None, :] + np.arange(4)[:, None]
    tr = bp._make_track(w, c, 1, d)
    q = np.ones((pb.BINS, 2))
    q[bins[2]] = [0.01, 0.1]
    [(nd, obs, parents, _, _)], jac = pb._normalised([tr], [bins], q)
    weights = q[bins[:, None], pb.GROUP]
    np.testing.assert_allclose(obs, tr[1] * weights)
    np.testing.assert_allclose(parents[boundaries], obs[boundaries - 1])
    np.testing.assert_allclose(jac, np.log(weights).sum())


def test_floor_is_physical_and_arming_noise_does_not_persist():
    c = np.repeat(np.arange(4), [10, 180, 220, 135])
    rng = np.random.default_rng(19)
    t = np.arange(len(c)) - 10
    mu = (c == 1) * 15 * (-np.expm1(-np.maximum(0, t) / 4.0)) + (c == 2) * 82.0
    records = [
        (
            np.maximum(
                0, mu + (c == 2) * i + (0.04 + 2 * (c == 2)) * rng.standard_normal((4, len(c)))
            ),
            c,
        )
        for i in range(3)
    ]
    model = pb.fit(records, max_iter=6)
    restored = pb.PopulationBridge.from_dict(model.to_dict())
    x = model.sample(c, 16, np.random.default_rng(221))
    np.testing.assert_array_equal(x, restored.sample(c, 16, np.random.default_rng(221)))
    assert x.min() >= 0 and model.n_params < 277 and min(np.diff(model.history)) >= 0
    assert abs(x[:, :, 150:175].mean() - 15) < 0.35
    assert x[:, :, 150:175].std() < 0.5 * x[:, :, 280:330].std()
