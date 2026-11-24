"""Lifting control law: exact density pieces, MAP fit, calibration and replay."""

import json

import numpy as np
from scipy.stats import multivariate_normal, norm

from experiments.rps_traj import bridge_program as bp
from experiments.rps_traj import lifting_control as lc
from tracking.rotors import MIXER

FS = 25


def _flight(rng, codes, hover, *, idle=60.0, sd=(1.5, 0.6, 0.6, 0.4), rho=0.9, corr=None):
    """Mode-space AR(1) fluctuations about commanded levels; stopped rotors are 0."""
    codes = np.asarray(codes)
    n = len(codes)
    L = np.linalg.cholesky(np.eye(4) if corr is None else corr)
    e = (rng.standard_normal((n, 4)) @ L.T) * np.asarray(sd)
    x = np.zeros((n, 4))
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    land = np.flatnonzero(codes == 3)
    age = (np.arange(n) - (land[0] if land.size else n)) / FS
    level = np.select(
        [codes == 1, codes == 2, codes == 3],
        [np.full(n, idle), np.full(n, hover), idle * np.exp(-np.maximum(age, 0) / 0.6)],
    )
    quiet = np.where(codes == 2, 1.0, 0.3)[:, None]
    m = x * quiet
    m[:, 0] += level
    w = np.maximum(MIXER @ m.T, 0.0)
    stopped = (codes == 0) | ((codes == 3) & (age > 2.0))
    w[:, stopped] = 0.0
    return w, codes


def test_encoding_floor_and_long_scale_law():
    rng = np.random.default_rng(3)
    w = rng.uniform(0, 5, (4, 60))
    w[:, :7] = rng.uniform(0, bp.FLOOR, (4, 7))
    back = lc.readout(lc.encode(w)[None])[0]
    np.testing.assert_allclose(back, np.where(w <= bp.FLOOR, 0.0, w), rtol=1e-9, atol=1e-12)
    knots = np.zeros((4, 4))
    knots[:, 2] = np.log([1.0, 2.0, 3.0, 4.0])
    sd = np.exp(lc._logsd(knots, np.array([1, 2, 7, 9]), np.array([2, 4, 128, 512])))
    np.testing.assert_allclose(sd[:, 2], [1.0, np.sqrt(2.0), 4.0, 2.0])
    # A phase mean of S samples: half the variance of a detail spanning 2S.
    np.testing.assert_allclose(
        np.exp(lc._dc_logsd(knots, np.array([256.0])))[0, 2], 4.0 * np.sqrt(128 / 512 / 2)
    )


def test_phase_mean_density_matches_dense_gaussian():
    rng = np.random.default_rng(4)
    p = dict(
        corr=rng.normal(size=6),
        scatter=np.array([0.3, -0.5]),
        style=0.2,
        active=0.0,
        group=np.zeros((1, 2)),
    )
    law = lc._law(p, dict(scatter=True, style=True, cruise=False, supported=np.ones(1, bool)))
    np.testing.assert_allclose(np.diag(law.C), 1.0)
    logsd = rng.normal(size=(5, 4))
    r = rng.normal(size=(5, 4))
    got = lc._dc_gauss(r, logsd, law)
    for i in range(5):
        sd = np.exp(logsd[i])
        cov = np.outer(sd, sd) * law.C + np.diag(law.tau2)
        # scipy's frozen-normal __call__ is unannotated, so pyright types cov by its default 1.
        mvn = multivariate_normal(np.zeros(4), cov)  # pyright: ignore[reportArgumentType]
        assert np.isclose(got[i], mvn.logpdf(r[i]))


def test_cruise_only_fit_monotone_with_counted_scalars():
    rng = np.random.default_rng(11)
    tracks = [
        _flight(rng, np.full(n, 2), h) for n, h in ((700, 150.0), (900, 156.0), (1100, 161.0))
    ]
    m = lc.fit(tracks, max_iter=4)
    assert len(m.history) >= 2 and np.all(np.diff(m.history) >= 0)
    f = m.flags
    assert f["scatter"] and f["style"] and f["cruise"]
    expected = m.structure.n_params + sum(len(b) for b in m.p["beta"]) + 16 + 6 + 2 + 1 + 4
    assert m.n_params == expected + int(f["drive"])
    assert m.n_params <= 34


def test_long_sustained_level_is_calibrated_and_stopped_prefix_exact():
    # Blackbird-like: one instantaneous arming, a long sustained high-speed flight.
    rng = np.random.default_rng(12)
    codes = np.r_[np.zeros(150, int), np.full(4000, 2)]
    w, _ = _flight(rng, codes, 184.0, rho=0.98)
    m = lc.fit([(w, codes)], max_iter=3)
    assert np.all(np.diff(m.history) >= 0) and m.n_params <= 34
    assert not m.flags["scatter"] and not m.flags["style"]
    truth = np.median(w[:, 150 + 1000 : 150 + 3000])
    draws = m.sample(codes, 6, np.random.default_rng(5))
    assert draws.shape == (6, 4, len(codes))
    assert np.all(draws[:, :, :150] == 0.0)
    central = np.median(draws[:, :, 150 + 1000 : 150 + 3000], axis=(1, 2))
    np.testing.assert_allclose(central, truth, rtol=0.03)
    longer = np.r_[np.zeros(150, int), np.full(9000, 2)]
    late = np.median(m.sample(longer, 4, np.random.default_rng(6))[:, :, 4000:8000], axis=(1, 2))
    np.testing.assert_allclose(late, truth, rtol=0.03)


def test_correlated_mode_fluctuations_are_fitted_and_generated():
    rng = np.random.default_rng(13)
    corr = np.eye(4)
    corr[1, 2] = corr[2, 1] = 0.7
    tracks = [
        _flight(rng, np.full(1500, 2), 150.0, sd=(1.0, 2.0, 2.0, 0.8), rho=0.0, corr=corr)
        for _ in range(2)
    ]
    m = lc.fit(tracks, max_iter=3)
    law = lc._law(m.p, m.flags)
    assert law.C[1, 2] > 0.4 and abs(law.C[0, 3]) < 0.3
    draws = m.sample(np.full(1500, 2), 8, np.random.default_rng(8))
    dz = np.concatenate([np.diff(lc.encode(x), axis=0) for x in draws])
    assert np.corrcoef(dz[:, 1], dz[:, 2])[0, 1] > 0.4


def test_full_schedule_replay_is_portable_and_physical():
    rng = np.random.default_rng(14)
    codes = np.repeat([0, 1, 2, 3], [20, 100, 600, 150])
    tracks = [_flight(rng, codes, h) for h in (140.0, 150.0, 146.0)]
    m = lc.fit(tracks, max_iter=2)
    assert np.all(np.diff(m.history) >= 0) and m.n_params < 277
    stored = json.loads(json.dumps(m.to_dict()))
    assert set(stored) == {"family", "structure", "flags", "params", "n_params", "history"}
    replay = lc.LiftingControl.from_dict(stored)
    assert replay.n_params == m.n_params
    target = np.repeat([0, 1, 2, 3], [31, 87, 455, 120])
    a = m.sample(target, 5, np.random.default_rng(21))
    b = replay.sample(target, 5, np.random.default_rng(21))
    np.testing.assert_array_equal(a, b)
    assert a.shape == (5, 4, len(target)) and np.isfinite(a).all() and a.min() >= 0
    assert np.all(a[:, :, :31] == 0.0)
    assert np.median(a[:, :, 200:500]) > 100


def test_raw_off_readings_enter_only_through_fixed_sensor_law():
    rng = np.random.default_rng(15)
    codes = np.r_[np.zeros(40, int), np.full(600, 2)]
    w, _ = _flight(rng, codes, 150.0)
    noisy = w.copy()
    noisy[:, :40] = np.abs(rng.normal(0.0, 0.5, (4, 40)))
    clean = lc.fit([(w, codes)], max_iter=0)
    seen = lc.fit([(noisy, codes)], max_iter=0)
    sensor = norm.logpdf(noisy[:, :40], scale=1.0).sum() - norm.logpdf(w[:, :40], scale=1.0).sum()
    assert np.isclose(seen.history[0] - clean.history[0], sensor)
    assert seen.n_params == clean.n_params
