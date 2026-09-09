"""Population-profile invariants for the stochastic rig model."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from data_processing import stochastic_rotor_noise as srn
from experiments.stochastic_fit.controls import planted_population_control
from experiments.stochastic_fit.data import Clip, Periodogram
from experiments.stochastic_fit.model import Spec
from experiments.stochastic_fit.population import (
    PopulationFitSpec,
    _init_heldout_profiles,
)
from experiments.stochastic_fit.predictive import (
    ProfileDraws,
    latent_topology_diagnostic,
)
from experiments.stochastic_fit.raw_predictive import (
    line_floor_margins,
    population_ranges,
)
from experiments.stochastic_fit.rig import ClipInRig, RigClip, RigParams, RigSpec
from experiments.stochastic_fit.topology_calibration import (
    apply_topology_calibration,
    fit_topology_calibration,
)


def _spec(*, n_harm: int = 5) -> Spec:
    return Spec(
        freqs=np.linspace(0.0, 500.0, 65),
        times=np.linspace(0.0, 0.4, 5),
        rps=np.full((2, 5), 80.0),
        n_mics=1,
        n_harm=n_harm,
        gp_std_db=0.0,
        floor_gp_std_db=0.0,
        floor_tilt_gp_std=0.0,
        window_kernel=False,
    )


def test_profile_factor_is_correlated_across_orders_and_separate_from_level() -> None:
    spec = _spec()
    rig = RigParams(spec, RigSpec(profile_rank=1), n_rotors=2, device="cpu")
    clip = ClipInRig(spec, rig, device="cpu")

    raw_mode = torch.tensor([0.0, 2.0, 4.0, 2.0, 0.0])
    with torch.no_grad():
        assert rig.profile_basis_db is not None
        assert clip.profile_z is not None
        rig.profile_db.zero_()
        rig.profile_basis_db[0].copy_(raw_mode)
        assert rig.profile_mode_std_raw is not None
        centred_mode = raw_mode - raw_mode.mean()
        mode_std = centred_mode.square().mean().sqrt()
        rig.profile_mode_std_raw[0].copy_(torch.log(torch.expm1(mode_std)))
        clip.level_db.copy_(torch.tensor([3.0, -2.0]))
        clip.profile_z.copy_(torch.tensor([[1.5], [-0.5]]))

    line_db = 10.0 * torch.log10(clip.line_power()[:, :, 0])
    centred_mode = raw_mode - raw_mode.mean()
    expected = torch.stack((3.0 + 1.5 * centred_mode, -2.0 - 0.5 * centred_mode))

    torch.testing.assert_close(line_db, expected)
    torch.testing.assert_close(line_db.mean(dim=1), torch.tensor([3.0, -2.0]))


def test_rank_zero_preserves_the_original_state_dict_contract() -> None:
    spec = _spec()
    rig = RigParams(spec, RigSpec(), n_rotors=2, device="cpu")
    clip = ClipInRig(spec, rig, device="cpu")

    assert "profile_basis_db" not in rig.state_dict()
    assert "profile_mode_std_raw" not in rig.state_dict()
    assert "profile_z" not in clip.state_dict()


def test_heldout_near_zero_profile_mode_has_a_bounded_initialization() -> None:
    spec = _spec()
    rig = RigParams(spec, RigSpec(profile_rank=1), n_rotors=2, device="cpu")
    clip = ClipInRig(spec, rig, device="cpu")
    with torch.no_grad():
        assert rig.profile_basis_db is not None
        rig.profile_basis_db.fill_(1e-12)
        clip.profile_db.copy_(
            torch.tensor([[80.0, -40.0, 20.0, -30.0, 10.0], [-70.0, 30.0, -20.0, 40.0, -10.0]])
        )
    pg = Periodogram(
        np.ones((1, 5, 65), dtype=np.float32),
        spec.freqs,
        spec.times,
        spec.rps,
        128,
        32,
    )
    rc = RigClip("heldout", "control", pg, clip, torch.ones(1, 5, 65), 1.0)

    _init_heldout_profiles(rig, [rc])

    assert clip.profile_z is not None
    assert torch.isfinite(clip.profile_z).all()
    assert float(clip.profile_z.abs().max()) < 1e-6


def test_planted_population_recovers_predictive_profile_distribution() -> None:
    freqs = np.linspace(0.0, 512.0, 129)
    times = np.linspace(0.0, 1.6, 17)
    rps = np.stack((np.full(17, 30.0), np.full(17, 47.0)))
    periodograms = [
        Periodogram(
            np.ones((1, 17, 129), dtype=np.float32),
            freqs,
            times,
            rps,
            256,
            64,
        )
        for _ in range(12)
    ]
    spec = Spec(
        freqs=freqs,
        times=times,
        rps=rps,
        n_mics=1,
        n_harm=8,
        f_min=20.0,
        f_max=500.0,
        knot_dt_s=0.4,
        gp_std_db=1.0,
        gp_tau_s=0.5,
        floor_gp_std_db=1.0,
        floor_gp_tau_s=1.0,
        floor_tilt_gp_std=0.0,
        window_kernel=True,
        line_shape="gauss",
    )
    result = planted_population_control(
        periodograms,
        spec,
        RigSpec(rotor_delta=True, profile_rank=1, level_prior_db=5.0),
        n_train=8,
        fit_spec=PopulationFitSpec(
            mc_samples=2,
            iw_samples=4,
            init_posterior_std=0.1,
        ),
        seed=11,
        map_iters=(10, 10, 20),
        vi_iters=75,
        log=lambda _: None,
    )

    planted = result["planted_population"]
    fitted = result["fitted_population"]
    assert abs(fitted["line_floor_mean_db"] - planted["line_floor_mean_db"]) < 3.0
    assert 0.3 < fitted["line_floor_std_db"] / planted["line_floor_std_db"] < 1.7
    assert 0.5 < fitted["rotor_contrast_std_db"] / planted["rotor_contrast_std_db"] < 2.0
    assert result["profile_covariance_cosine"] > 0.9
    assert abs(result["heldout_excess"]) < 0.08
    assert result["heldout_iw_ess"] > 1.0


def _profile_draws(seed: int, n_clips: int, *, high_order_shift: float = 0.0) -> ProfileDraws:
    rng = np.random.default_rng(seed)
    n_rotors, n_harmonics = 4, 32
    orders = np.arange(1, n_harmonics + 1, dtype=np.float64)
    mean = -10.0 * np.log10(orders)
    mode = np.log2(orders)
    mode -= mode.mean()
    mode /= np.sqrt(np.mean(mode**2))
    factors = rng.normal(0.0, 2.0, (n_clips, n_rotors, 1))
    contrast = rng.normal(0.0, 2.0, (n_clips, n_rotors))
    contrast -= contrast.mean(axis=1, keepdims=True)
    profile = (
        mean[None, None, :]
        + factors * mode[None, None, :]
        + contrast[:, :, None]
        + rng.normal(0.0, 0.5, (n_clips, n_rotors, n_harmonics))
    )
    profile[:, :, 11:] += high_order_shift
    floor = np.full((n_clips, n_harmonics), -18.0)
    gamma = np.broadcast_to(
        1.0 + 0.1 * orders,
        (n_clips, n_rotors, n_harmonics),
    ).copy()
    return ProfileDraws(profile, floor, gamma)


def test_latent_diagnostic_accepts_same_population_and_rejects_shifted_orders() -> None:
    real_train = _profile_draws(1, 160)
    real_test = _profile_draws(2, 160)
    matched = latent_topology_diagnostic(
        real_train,
        real_test,
        _profile_draws(3, 160),
        _profile_draws(4, 160),
        seed=5,
        n_bootstrap=300,
    )
    shifted = latent_topology_diagnostic(
        real_train,
        real_test,
        _profile_draws(6, 160, high_order_shift=8.0),
        _profile_draws(7, 160, high_order_shift=8.0),
        seed=8,
        n_bootstrap=300,
    )

    assert matched["realism_gate"] is False
    assert matched["latent_consistent"]
    assert matched["classifier_auc_95"][1] < 0.70
    assert not shifted["latent_consistent"]
    assert shifted["classifier_auc"] > 0.9


def test_raw_topology_reads_waveform_lines_not_fitted_latents() -> None:
    sample_rate = 16000
    rps = np.full((1, 2 * sample_rate), 70.0)
    ranges = srn.StochasticRanges(
        rolloff_p=(0.7, 0.7),
        harm_jitter_db=(0.0, 0.0),
        blade_counts=(1,),
        rotor_similarity=(1.0, 1.0),
        gamma0_hz=(0.5, 0.5),
        gamma_slope_hz=(0.1, 0.1),
        shaft_jitter_rps=(0.0, 0.0),
        phase_diffusion_hz_per_order=(0.0, 0.0),
        floor_shape_std_db=(0.0, 0.0),
        floor_tilt_db_oct=(0.0, 0.0),
        floor_rel_db=(-24.0, -24.0),
        harm_gp_std_db=(0.0, 0.0),
        floor_gp_std_db=(0.0, 0.0),
        floor_tilt_gp_std=(0.0, 0.0),
    )
    rng = np.random.default_rng(50)
    params = srn.sample_params(
        rng,
        ranges,
        n_rotors=1,
        n_harmonics=32,
        sample_rate=sample_rate,
    )
    audio, _ = srn.synthesize(
        params,
        rps,
        rng=rng,
        n_mics=1,
        mic_gain_db=(0.0, 0.0),
        line_mode="fm",
    )
    correct = Clip("correct", "control", audio, rps, sample_rate)
    wrong = Clip("wrong", "control", audio, 1.27 * rps, sample_rate)

    correct_margin, correct_count = line_floor_margins(correct, k_max=24)
    wrong_margin, _ = line_floor_margins(wrong, k_max=24)

    assert correct_count[:, :16].sum() > 50
    assert np.nanmedian(correct_margin[:, :16]) > np.nanmedian(wrong_margin[:, :16]) + 6.0


def test_population_renderer_preserves_fitted_local_floor_reference() -> None:
    control_hz = np.geomspace(30.0, 8000.0, 14)
    floor_shape = np.linspace(8.0, -12.0, 14)
    summary = {
        "rig": {
            "profile_db": [0.0, 6.0, -2.0, -5.0, -8.0],
            "profile_basis_db": None,
            "delta_db": [[0.0] * 5],
            "gamma0": 2.0,
            "gamma_slope": 0.5,
            "width_scale": [1.2],
            "mic_gain_db": [[3.0]],
            "mic_floor_db": [-2.0],
            "gain_all_db": [1.0],
        },
        "population": {
            "line_floor_mean_db": 22.0,
            "line_floor_std_db": 0.0,
            "rotor_contrast_std_db": 0.0,
        },
        "train": {
            "clip": {
                "params": {
                    "floor_shape_db": floor_shape.tolist(),
                    "floor_ctrl_hz": control_hz.tolist(),
                    "floor_tilt_db_oct": -4.0,
                }
            }
        },
    }
    ranges = srn.StochasticRanges.from_dict(population_ranges(summary, {}))
    params = srn.sample_params(
        np.random.default_rng(9),
        ranges,
        n_rotors=1,
        n_harmonics=5,
        sample_rate=16000,
    )
    np.testing.assert_allclose(params.gamma0, [2.4])
    np.testing.assert_allclose(params.gamma_slope, [0.6])
    np.testing.assert_allclose(
        params.shaft_jitter_rps,
        [0.6 / np.sqrt(2.0 * np.log(2.0))],
    )
    assert params.fixed_mic_gain_db is not None
    assert params.fixed_mic_floor_db is not None
    assert params.fixed_mic_gain_all_db is not None
    np.testing.assert_allclose(params.fixed_mic_gain_db, [[3.0]])
    np.testing.assert_allclose(params.fixed_mic_floor_db, [-2.0])
    np.testing.assert_allclose(params.fixed_mic_gain_all_db, [1.0])
    reference_hz = 160.0
    fitted_local_margin = (
        22.0
        - np.interp(np.log2(reference_hz), np.log2(control_hz), floor_shape)
        - (-4.0) * np.log2(reference_hz / 500.0)
    )
    rendered_local_margin = (
        params.profile_db[0, 1]
        - params.floor_mean_db
        - srn.floor_shape_db(params, np.array([reference_hz]))[0]
    )

    assert rendered_local_margin == pytest.approx(fitted_local_margin, abs=0.1)


def test_synthetic_likelihood_recovers_training_only_profile_correction() -> None:
    rng = np.random.default_rng(44)
    n_clips, n_orders = 120, 32
    orders = np.arange(1, n_orders + 1, dtype=np.float64)
    synthetic = rng.normal(0.0, 1.0, (n_clips, 1, n_orders))
    planted = 4.0 + 2.0 * np.sin(np.log(orders)) + 1.5 * ((orders % 2) == 0)
    real = synthetic + planted[None, None, :] + rng.normal(0.0, 0.5, synthetic.shape)
    calibration = fit_topology_calibration(real, synthetic, reference_order=2)

    assert calibration.line_floor_shift_db == pytest.approx(planted[1], abs=0.3)
    expected_shape = planted - planted[1]
    assert np.sqrt(np.mean((calibration.profile_correction_db - expected_shape) ** 2)) < 0.5
    assert calibration.standardized_rmse_after < 0.2 * calibration.standardized_rmse_before

    summary = {
        "rig": {"profile_db": [0.0] * n_orders},
        "population": {"line_floor_mean_db": 20.0},
    }
    corrected = apply_topology_calibration(summary, calibration)
    np.testing.assert_allclose(corrected["rig"]["profile_db"], calibration.profile_correction_db)
    assert corrected["population"]["line_floor_mean_db"] == pytest.approx(
        20.0 + calibration.line_floor_shift_db
    )


def test_synthetic_likelihood_recovers_missing_profile_covariance() -> None:
    rng = np.random.default_rng(91)
    n_clips, n_orders = 320, 16
    order = np.arange(n_orders, dtype=np.float64)
    mode = np.sin(2.0 * np.pi * order / n_orders)
    mode = 4.0 * mode / np.sqrt(np.mean(mode**2))
    synthetic = rng.normal(0.0, 1.0, (n_clips, 1, n_orders))
    real = (
        rng.normal(0.0, 1.0, (n_clips, 1, n_orders))
        + rng.normal(size=(n_clips, 1, 1)) * mode[None, None, :]
    )
    calibration = fit_topology_calibration(
        real,
        synthetic,
        reference_order=2,
        max_rank=3,
    )

    assert calibration.selected_rank >= 1
    recovered = calibration.profile_basis_db.T @ calibration.profile_basis_db
    planted = np.outer(mode, mode)
    cosine = np.sum(recovered * planted) / (np.linalg.norm(recovered) * np.linalg.norm(planted))
    assert cosine > 0.9
