"""The fitted trajectory model, as a training-stream source.

``rps.kind: fitted_traj`` turns published fits into training trajectories, and
these tests pin the places where that can go quietly wrong: a fit that does not
survive its own serialisation (the dataset ships JSON, and a stream reading a
lossy copy is no longer reading the fit), a mixture that ignores its weights, a
mean shift that moves the level but leaves the ESC clamp behind (which would
pin every shifted flight at the unshifted ceiling), a policy error that reaches
a DataLoader worker instead of the pool constructor, and the pool's own
windowing of a fitted flight.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from omegaconf import OmegaConf

from data_processing import trajectory_model as tm
from data_processing.stochastic_rotor_noise import StochasticNoisePool

POLICY = Path("conf/online_mix/traj_fitted_5050.yaml")

#: Two plausible rigs under the names the shipped policy mixes, so the policy's
#: own ``rigs:`` block resolves against the tree built here. The VALUES are
#: invented (a small airframe and a large one); the campaign's real ones live in
#: the ``rps-traj-fits`` dataset, which is not in the repository.
RIGS: dict[str, dict[str, Any]] = {
    "michaels": {
        "mu": [90.0, 75.0, 82.0, 76.0],
        "theta": 0.3,
        "tau_slow": [1.0, 0.7, 0.7, 1.2],
        "sigma_slow": [3.0, 1.0, 1.0, 2.0],
        "f0": [1.0, 3.0, 2.0, 1.5],
        "zeta": [1.0, 0.4, 1.5, 0.8],
        "sigma_osc": [2.0, 0.5, 0.5, 1.5],
        "tau_e": 0.05,
        "sigma_e": 0.3,
        "s_c": 1.5,
        "s_r": [0.5, 0.5, 0.5, 0.5],
    },
    "dregon": {
        "mu": [220.0, 205.0, 215.0, 210.0],
        "theta": -0.1,
        "tau_slow": [2.0, 1.5, 1.5, 2.5],
        "sigma_slow": [6.0, 3.0, 3.0, 4.0],
        "f0": [0.8, 2.0, 2.5, 1.2],
        "zeta": [1.2, 0.6, 1.1, 0.9],
        "sigma_osc": [4.0, 1.5, 1.5, 3.0],
        "tau_e": 0.02,
        "sigma_e": 0.8,
        "s_c": 4.0,
        "s_r": [1.0, 1.0, 1.0, 1.0],
    },
}


def _fit(rig: str) -> tm.NewFit:
    params = tm.Params(**RIGS[rig])
    hover = float(np.mean(params.mu))
    return tm.NewFit(
        rig=rig,
        params=params,
        idle_rps=0.4 * params.mu,
        rps_min=0.65 * hover,
        rps_max=1.30 * hover,
        fs=tm.RATE_HZ,
        fit_rate_hz=tm.RATE_HZ,
        nll=-1234.5,  # a fit record carries a finite NLL; NaN is not JSON
        n_flights=5,
    )


@pytest.fixture
def fits_tree(tmp_path: Path) -> Path:
    """A published-layout fits tree: ``fits/<rig>.json`` + ``posterior.json``."""
    root = tmp_path / "rps-traj-fits"
    (root / "fits").mkdir(parents=True)
    fits = {rig: _fit(rig) for rig in RIGS}
    for rig, fit in fits.items():
        (root / "fits" / f"{rig}.json").write_text(fit.to_json(), encoding="utf-8")
    vectors = np.stack([tm.rig_vector(fit.params) for fit in fits.values()])
    posterior = tm.Posterior(
        mean=vectors.mean(axis=0),
        std=np.maximum(0.1 * np.abs(vectors.mean(axis=0)), 0.05),
        rigs=tuple(sorted(fits)),
    )
    (root / "posterior.json").write_text(posterior.to_json(), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"rigs": sorted(fits)}), encoding="utf-8")
    return root


def _source(fits_tree: Path, **over: Any) -> tm.FittedTrajectorySource:
    cfg: dict[str, Any] = {"fits": str(fits_tree), "rigs": {"michaels": 1.0}}
    cfg.update(over)
    return tm.build_from_config(cfg)


def _airborne_mean(source: tm.FittedTrajectorySource, seed: int, n: int = 6000) -> float:
    """Mean of one stationary airborne stretch from one drawn drone."""
    rng = np.random.default_rng(seed)
    draw = source.draw(rng)
    return float(np.mean(draw.params.sample_airborne(n, rng, tm.RATE_HZ, clip=draw.clip)))


# ── the fits survive the trip to disk ────────────────────────────────────────


def test_a_fit_json_round_trips_and_samples_identically(fits_tree: Path):
    """A fit read back from the published tree draws the same trajectory.

    The dataset carries the campaign's fit as JSON and the stream reads it, so
    a lossy field (or a reordered random stream) would show up as a different
    trajectory from the same seed.
    """
    path = fits_tree / "fits" / "michaels.json"
    direct = tm.NewFit.from_json(path.read_text(encoding="utf-8"))
    reloaded = tm.NewFit.from_json(direct.to_json())
    from_tree = tm.load_bundle(fits_tree).fits["michaels"]

    for other in (reloaded, from_tree):
        assert other.params.to_dict() == direct.params.to_dict()
        assert other.clip == direct.clip
        np.testing.assert_array_equal(other.idle_rps, direct.idle_rps)

    samples = [
        fit.sampler(fs=tm.RATE_HZ)(2048, np.random.default_rng(7))
        for fit in (direct, reloaded, from_tree)
    ]
    np.testing.assert_array_equal(samples[0], samples[1])
    np.testing.assert_array_equal(samples[0], samples[2])
    # And the unconstrained vector is a true round trip of the same parameters.
    revector = tm.Params.from_vector(direct.params.to_vector())
    np.testing.assert_allclose(revector.mu, direct.params.mu, rtol=1e-12)
    np.testing.assert_allclose(revector.f0, direct.params.f0, rtol=1e-9)
    np.testing.assert_allclose(revector.tau_slow, direct.params.tau_slow, rtol=1e-9)


# ── the mixture ──────────────────────────────────────────────────────────────


def test_b_mixture_honours_its_weights(fits_tree: Path):
    """Weights are relative frequencies per flight, and they need not sum to 1."""
    weights = {"michaels": 1.0, "dregon": 1.0, tm.POSTERIOR_RIG: 2.0}
    source = _source(fits_tree, rigs=weights)
    rng = np.random.default_rng(0)
    n = 2000
    counts = Counter(source.draw(rng).rig for _ in range(n))

    assert set(counts) == set(weights)
    total = sum(weights.values())
    chi2 = 0.0
    for rig, weight in weights.items():
        expected = n * weight / total
        chi2 += (counts[rig] - expected) ** 2 / expected
    # 2 degrees of freedom: 13.8 is the 0.999 quantile, so a correct sampler
    # fails this once in a thousand runs and a swapped weight always does.
    assert chi2 < 13.8, (counts, chi2)


def test_b_posterior_draws_are_new_drones(fits_tree: Path):
    """``posterior`` is the GLOBAL fit: every draw is a fresh rig, not a stored one."""
    source = _source(fits_tree, rigs={tm.POSTERIOR_RIG: 1.0})
    rng = np.random.default_rng(3)
    stored = [tm.Params(**p).mu for p in RIGS.values()]
    drawn = [source.draw(rng).params.mu for _ in range(16)]

    for mu in drawn:
        assert all(not np.allclose(mu, other) for other in stored)
    # Two draws are two different drones, not one repeated.
    assert not np.allclose(drawn[0], drawn[1])
    # Every draw is still a drone: four positive rotor means.
    assert all(np.all(mu > 0.0) for mu in drawn)


# ── the mean shift ───────────────────────────────────────────────────────────


def test_c_mean_shift_moves_the_level_and_the_clamp(fits_tree: Path):
    """A shift moves the hover level AND the ESC clamp.

    The shift is larger than the rig's own clamp margin, so a clamp left
    behind would pin the whole trajectory at the unshifted ceiling
    (1.30 x hover) instead of following the level up.
    """
    hover = float(np.mean(RIGS["michaels"]["mu"]))
    shift = 40.0
    base = _airborne_mean(_source(fits_tree), seed=11)
    moved = _airborne_mean(_source(fits_tree, mean_shift=[shift, shift]), seed=11)

    assert base == pytest.approx(hover, abs=4.0)
    assert moved == pytest.approx(hover + shift, abs=4.0)
    assert moved > 1.30 * hover  # impossible with the unshifted ceiling

    draw = _source(fits_tree, mean_shift=[shift, shift]).draw(np.random.default_rng(11))
    fit = _fit("michaels")
    assert draw.clip == pytest.approx((fit.rps_min + shift, fit.rps_max + shift))
    # The idle level is a fraction of the hover level, so it travels with it.
    assert float(np.mean(draw.idle_rps)) > float(np.mean(fit.idle_rps))


def test_c_mean_scale_multiplies_the_level_and_the_clamp(fits_tree: Path):
    hover = float(np.mean(RIGS["michaels"]["mu"]))
    scaled = _airborne_mean(_source(fits_tree, mean_scale=[2.0, 2.0]), seed=5)
    assert scaled == pytest.approx(2.0 * hover, abs=8.0)

    draw = _source(fits_tree, mean_scale=[2.0, 2.0]).draw(np.random.default_rng(5))
    fit = _fit("michaels")
    assert draw.clip == pytest.approx((2.0 * fit.rps_min, 2.0 * fit.rps_max))


def test_c_identity_defaults_leave_the_rig_alone(fits_tree: Path):
    draw = _source(fits_tree).draw(np.random.default_rng(1))
    fit = _fit("michaels")
    np.testing.assert_allclose(draw.params.mu, fit.params.mu)
    assert draw.clip == fit.clip


# ── policy errors, at parse time ─────────────────────────────────────────────


@pytest.mark.parametrize(
    ("over", "message"),
    [
        ({"rigs": {"no_such_rig": 1.0}}, "unknown rig"),
        ({"rigs": {"michaels": -1.0}}, "non-negative"),
        ({"rigs": {}}, "empty"),
        ({"fits": None}, "requires rps.fits"),
    ],
)
def test_d_invalid_configs_are_refused(fits_tree: Path, over: dict[str, Any], message: str):
    with pytest.raises(ValueError, match=message):
        _source(fits_tree, **over)


def test_d_measurement_noise_can_be_dropped(fits_tree: Path):
    """``measurement_noise: false`` removes the per-rotor measurement OU only.

    Its own variance leaves the rotor variance; the shaft modes stay, so the
    rotors' mutual correlation goes UP rather than the trajectory going quiet.
    """
    rng_a, rng_b = np.random.default_rng(2), np.random.default_rng(2)
    with_noise = _source(fits_tree).draw(rng_a)
    without = _source(fits_tree, measurement_noise=False).draw(rng_b)

    assert without.params.sigma_e == 0.0
    a = with_noise.params.sample_airborne(60_000, np.random.default_rng(4), tm.RATE_HZ)
    b = without.params.sample_airborne(60_000, np.random.default_rng(4), tm.RATE_HZ)
    sigma_e = RIGS["michaels"]["sigma_e"]
    # The shaft is untouched (same random stream), so what the flag removes is
    # exactly the measurement process: one OU of std ``sigma_e`` per rotor.
    assert float(np.mean(np.std(a - b, axis=1))) == pytest.approx(sigma_e, rel=0.1)
    assert np.all(np.var(b, axis=1) < np.var(a, axis=1))


# ── the pool ─────────────────────────────────────────────────────────────────


def test_e_pool_from_the_policy_windows_a_fitted_flight(fits_tree: Path):
    """The shipped policy builds a pool whose windows are usable rps tracks."""
    cfg = OmegaConf.to_container(OmegaConf.load(POLICY), resolve=True)
    assert isinstance(cfg, dict)
    sources = [s for s in cfg["sources"]["noise"] if s["kind"] == "stochastic"]
    assert len(sources) == 2
    assert all(s["rps"]["kind"] == "fitted_traj" for s in sources)
    assert all(s["rps"]["fits"] == "dload:rps-traj-fits" for s in sources)

    spec = dict(sources[0])
    # The published fits and the preset bank are gitignored artefacts; the
    # trajectory under test is the point, so read the fits from the tree built
    # here and let the pool sample its own timbre.
    spec["rps"] = {**spec["rps"], "fits": str(fits_tree)}
    spec.pop("preset_bank", None)
    pool = StochasticNoisePool.from_config(spec, duration_s=2.0, sample_rate=16000)
    assert pool.rps_kind == "fitted_traj"

    traj = pool._traj  # noqa: SLF001 - the pool's own fitted-trajectory source
    assert traj is not None
    rng = np.random.default_rng(0)
    scale_hi = pool.rps_scale_range[1]
    reached_cruise = False
    for _ in range(8):
        rps = pool.sample_rps(rng, 2.0)
        assert rps.shape == (4, 32_000)
        assert np.isfinite(rps).all()
        assert rps.min() >= 0.0
        draw = traj.last_draw  # the window's own provenance
        assert draw is not None
        assert rps.max() <= scale_hi * draw.clip[1] + 1e-6
        reached_cruise |= bool(np.median(rps) >= pool.rps_scale_range[0] * draw.clip[0])
    assert reached_cruise, "no window reached the airborne regime"


def test_e_pool_reuses_one_flight_for_flight_reuse_windows(fits_tree: Path):
    """``flight_reuse`` windows come from ONE flight, as for ``full_flight``."""
    spec = {
        "kind": "stochastic",
        "n_mics": 1,
        "rps": {
            "kind": "fitted_traj",
            "fits": str(fits_tree),
            "rigs": {"michaels": 1.0, "dregon": 1.0},
            "flight_fs": 200,
            "flight_reuse": 3,
        },
    }
    pool = StochasticNoisePool.from_config(spec, duration_s=1.0, sample_rate=16000)
    traj = pool._traj  # noqa: SLF001 - the pool's own fitted-trajectory source
    assert traj is not None
    rng = np.random.default_rng(0)
    rigs = []
    for _ in range(9):
        pool.sample_rps(rng, 1.0)
        rigs.append(traj.last_rig)
    # Three windows per flight: the rig can only change every third window.
    assert rigs[0] == rigs[1] == rigs[2]
    assert rigs[3] == rigs[4] == rigs[5]
    assert rigs[6] == rigs[7] == rigs[8]
