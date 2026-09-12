"""Population-profile invariants for the stochastic rig model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

import experiments.stochastic_fit.rig as rig_module
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
    RawTopologyDraws,
    line_floor_margins,
    population_ranges,
    repeated_waveform_gate_control,
)
from experiments.stochastic_fit.rig import ClipInRig, RigClip, RigParams, RigSpec
from experiments.stochastic_fit.topology_calibration import (
    TopologyCalibration,
    apply_topology_calibration,
    fit_topology_calibration,
)
from experiments.stochastic_fit.visibility import (
    apply_visibility_model,
    fit_visibility_model,
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


def test_adam_retries_a_nonfinite_final_step_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor([0.0]))
    rig = RigParams(_spec(), RigSpec(), n_rotors=2, device="cpu")
    calls = 0

    def objective(_rig, _clips):
        nonlocal calls
        calls += 1
        if calls == 1:
            parameter.grad = torch.full_like(parameter, float("nan"))
            return torch.zeros(()), 0.0
        loss = (parameter - 1.0).square().sum()
        loss.backward()
        return loss, float(loss)

    monkeypatch.setattr(rig_module, "_objective", objective)
    with pytest.warns(RuntimeWarning, match="gradient became non-finite"):
        result = rig_module._adam(rig, [], [parameter], iters=1, lr=0.1, retries=1)

    assert calls == 2
    assert np.isfinite(result)
    assert torch.isfinite(parameter).all()
    assert float(parameter) > 0.0


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


def test_zero_topology_correction_preserves_the_continuous_basis() -> None:
    old_basis = [[1.0, -1.0, 0.5], [-0.5, 0.0, 0.5]]
    summary = {
        "rig": {"profile_db": [0.0, 0.0, 0.0], "profile_basis_db": old_basis},
        "population": {"line_floor_mean_db": 20.0},
    }
    calibration = TopologyCalibration(
        reference_order=2,
        line_floor_shift_db=0.0,
        profile_correction_db=np.zeros(3),
        profile_basis_db=np.zeros((0, 3)),
        selected_rank=0,
        rank_cv_nll_per_order=[0.0],
        rank_cv_se_per_order=[0.0],
        smoothing_lambda=1.0,
        lambda_cv_score=[0.0],
        lambda_cv_se=[0.0],
        observed_orders=3,
        standardized_rmse_before=0.0,
        standardized_rmse_after=0.0,
    )

    corrected = apply_topology_calibration(summary, calibration)

    np.testing.assert_allclose(corrected["rig"]["profile_basis_db"], old_basis)


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


def test_exact_one_vs_four_gate_has_a_low_planted_false_rejection_rate() -> None:
    def pool(seed: int, carriers: int) -> RawTopologyDraws:
        rng = np.random.default_rng(seed)
        draws, orders = 8, 32
        carrier_effect = rng.normal(0.0, 2.0, (carriers, 1, orders))
        margin = carrier_effect[:, None] + rng.normal(0.0, 1.0, (carriers, draws, 1, orders))
        margin = margin.reshape(carriers * draws, 1, orders)
        carrier_id = np.repeat(np.arange(carriers), draws)
        return RawTopologyDraws(
            margin,
            np.ones_like(margin),
            carrier_id,
        )

    control = repeated_waveform_gate_control(
        pool(10, 10),
        pool(11, 14),
        synthetic_draws_per_carrier=4,
        repeats=50,
        n_bootstrap=100,
        seed=12,
    )

    assert control["false_rejection_rate"] <= 0.1


def test_visibility_model_recovers_probability_and_finite_off_component() -> None:
    rng = np.random.default_rng(72)
    n_clips, n_orders = 240, 32
    order = np.arange(1, n_orders + 1, dtype=np.float64)
    logit = 2.5 - 0.12 * order + 1.2 * (order % 2 == 0) + 2.0 * (order == 2)
    planted_probability = 1.0 / (1.0 + np.exp(-logit))
    synthetic = 12.0 - 0.1 * order[None, None, :] + rng.normal(0.0, 1.0, (n_clips, 1, n_orders))
    visible = rng.random((n_clips, 1, n_orders)) < planted_probability
    real = synthetic + rng.normal(0.0, 0.5, synthetic.shape)
    real = np.where(visible, real, real - 12.0)

    model = fit_visibility_model(real, synthetic, threshold_db=6.0)

    assert np.corrcoef(model.visible_probability, planted_probability)[0, 1] > 0.85
    assert np.median(model.attenuation_mean_db) > 8.0
    assert 1.0 <= model.attenuation_std_db <= 6.0
    assert np.isfinite(model.cv_nll_per_observation).all()
    summary = {"rig": {"profile_db": [0.0] * 40}, "population": {}}
    applied = apply_visibility_model(summary, model)
    assert applied["visibility_model"]["threshold_db"] == 6.0
    assert len(applied["visibility_model"]["visible_probability"]) == 40


class _FakeSeries:
    """The two attributes the envelope fitters read off a ``td.Series``."""

    def __init__(self, data: np.ndarray, rate: float | None = None) -> None:
        self.data = data
        self.tindex = type("TIndex", (), {"rate": rate})()


def _ou(rng: np.random.Generator, n: int, length: int, tau: float, std: float, rate: float):
    decay = np.exp(-1.0 / (rate * tau))
    out = np.empty((n, length))
    out[:, 0] = rng.normal(0.0, std, n)
    step = rng.normal(0.0, std * np.sqrt(1.0 - decay * decay), (n, length))
    for index in range(1, length):
        out[:, index] = decay * out[:, index - 1] + step[:, index]
    return out


def _planted_decomp_frame(seed: int = 11) -> tuple[dict, list[tuple[float, float, float]], float]:
    """Envelopes with a known speed law and a known two-component amplitude process."""
    rng = np.random.default_rng(seed)
    n_rotors, n_orders, n_frames, env_rate = 4, 64, 18000, 100.0
    truth = [(2.6, 4.5, 0.30), (3.9, 0.15, 0.05)]
    residual = np.zeros((n_rotors, n_orders, n_frames))
    for std, tau, coherence in truth:
        common = _ou(rng, n_rotors, n_frames, tau, std, env_rate)
        private = _ou(rng, n_rotors * n_orders, n_frames, tau, std, env_rate).reshape(
            n_rotors, n_orders, n_frames
        )
        residual += np.sqrt(coherence) * common[:, None, :] + np.sqrt(1.0 - coherence) * private
    residual += rng.normal(0.0, 0.8, residual.shape)
    speed = (
        60.0
        + 15.0 * np.sin(np.linspace(0.0, 9.0, n_frames))[None, :]
        + rng.normal(0.0, 0.5, (n_rotors, n_frames))
    )
    exponent = 4.1
    level = (
        -60.0
        + exponent * 10.0 * np.log10(speed)[:, None, :]
        + np.linspace(20.0, -20.0, n_orders)[None, :, None]
        + residual
    )
    amplitude = np.sqrt(10.0 ** (level / 10.0))[None].repeat(8, axis=0)
    frame = {
        "amp": _FakeSeries(amplitude),
        "amp_valid": _FakeSeries(np.ones((n_rotors, n_orders, n_frames), dtype=bool)),
        "rps": _FakeSeries(np.repeat(speed, 160, axis=1), rate=16000.0),
    }
    return frame, truth, exponent


def test_planted_amplitude_process_is_recovered_from_envelopes() -> None:
    """The dynamics fitter must find the planted mixture, not the noise floor.

    A sign or normalisation error in the OU spectral density, or in the
    coherence projection, changes these numbers by far more than the tolerances
    below; the mixture's TOTAL variance is what the renderer consumes, so it is
    checked tightest.
    """
    from experiments.stochastic_fit.decomp_dynamics import fit_decomp_dynamics

    frame, truth, exponent = _planted_decomp_frame()
    fitted = fit_decomp_dynamics(frame, recording_id="PLANTED", floor_law=False)

    assert fitted.selected_components == 2
    planted_variance = sum(std**2 for std, _, _ in truth)
    fitted_variance = sum(component["std_db"] ** 2 for component in fitted.components)
    assert fitted_variance == pytest.approx(planted_variance, rel=0.25)
    slow, fast = fitted.components
    assert slow["tau_s"] > 3.0 > fast["tau_s"]
    assert fast["tau_s"] == pytest.approx(truth[1][1], abs=0.1)
    # the slow component is the coherent one, by a wide margin
    assert slow["coherence"] > 3.0 * fast["coherence"]
    assert slow["coherence"] == pytest.approx(truth[0][2], abs=0.1)
    # few independent speed excursions in 180 s, so the exponent is loose
    assert fitted.speed_exponent == pytest.approx(exponent, abs=0.5)


def test_one_standard_error_rule_prefers_the_simpler_kernel() -> None:
    """A single-component truth must not buy a second component."""
    from experiments.stochastic_fit.decomp_dynamics import fit_decomp_dynamics

    rng = np.random.default_rng(5)
    n_rotors, n_orders, n_frames, env_rate = 4, 64, 18000, 100.0
    residual = _ou(rng, n_rotors * n_orders, n_frames, 0.8, 2.9, env_rate).reshape(
        n_rotors, n_orders, n_frames
    )
    speed = np.full((n_rotors, n_frames), 70.0) + rng.normal(0.0, 0.3, (n_rotors, n_frames))
    level = -60.0 + np.linspace(10.0, -10.0, n_orders)[None, :, None] + residual
    frame = {
        "amp": _FakeSeries(np.sqrt(10.0 ** (level / 10.0))[None].repeat(8, axis=0)),
        "amp_valid": _FakeSeries(np.ones((n_rotors, n_orders, n_frames), dtype=bool)),
        "rps": _FakeSeries(np.repeat(speed, 160, axis=1), rate=16000.0),
    }
    fitted = fit_decomp_dynamics(frame, recording_id="PLANTED", floor_law=False, min_rps=30.0)
    assert fitted.selected_components == 1
    assert fitted.components[0]["tau_s"] == pytest.approx(0.8, abs=0.3)


def test_width_population_separates_flight_spread_from_estimation_noise(tmp_path) -> None:
    """Only the common mode across a clip's rotors counts as a population.

    Independent per-rotor scatter is the fitters' own noise; a bug that counts
    it as flight-to-flight spread would report roughly the total, which is more
    than twice the planted value here.
    """
    import json

    from experiments.stochastic_fit.carrier_error import fit_width_population

    rng = np.random.default_rng(3)
    n_clips, n_rotors = 40, 4
    base = np.asarray([0.9, 0.5, 0.45, 0.6])
    planted_log_std, noise_log_std = 0.5, 0.7
    root = tmp_path / "gauss"
    root.mkdir()
    for clip in range(n_clips):
        common = rng.normal(0.0, planted_log_std)
        noise = rng.normal(0.0, noise_log_std, n_rotors)
        slope = base * np.exp(common + noise)
        np.savez(
            root / f"CLIP_{clip:02d}.npz",
            params=json.dumps({"gamma_slope": slope.tolist()}),
        )

    fitted = fit_width_population(tmp_path, clip_prefixes=("CLIP",), variant="gauss")

    assert fitted.common_log_std == pytest.approx(planted_log_std, abs=0.15)
    total = np.sqrt(planted_log_std**2 + noise_log_std**2)
    assert fitted.common_log_std < 0.75 * total
    assert fitted.rotor_median_slope_hz == pytest.approx(list(base), rel=0.3)


# -- the bench instrument ------------------------------------------------------


def _planted_bench_recording(
    rate: float,
    profile_db: np.ndarray,
    *,
    gamma_slope: float,
    sr: float,
    seconds: float,
    seed: int,
    n_mic: int = 2,
) -> np.ndarray:
    """A stationary single-rotor comb with a KNOWN profile and width law.

    Each order is a tone at ``k * rate`` whose phase diffuses so that its
    Lorentzian half width is ``gamma_slope * k`` Hz (the quasi-static reading
    of §2.2), on a flat noise floor. Nothing else varies, so the bench
    instrument's three outputs have known truths.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    out = rng.standard_normal((n_mic, n)) * 1e-3
    for i, level in enumerate(profile_db):
        k = i + 1
        if k * rate > 0.45 * sr:
            break
        amp = 10.0 ** (level / 20.0)
        # phase random walk giving HWHM = gamma_slope * k Hz
        d = 4.0 * np.pi * gamma_slope * k
        walk = np.cumsum(rng.standard_normal(n) * np.sqrt(d / sr))
        for m in range(n_mic):
            out[m] += amp * np.cos(2 * np.pi * k * rate * t + walk + m * 0.7)
    return out


def test_planted_bench_recovers_the_rate_profile_and_width_law() -> None:
    from experiments.stochastic_fit import bench

    sr, rate, slope = 8000.0, 40.0, 0.15
    # 40 orders reach 1.6 kHz: a comb that spans the band, so a half-rate
    # hypothesis has to pay for 40 empty orders instead of covering real ones.
    k = np.arange(1, 41)
    profile = -0.25 * k + np.where(k % 2, -8.0, 0.0)
    recs = {
        (motor, 50): _planted_bench_recording(
            rate, profile, gamma_slope=slope, sr=sr, seconds=8.0, seed=11 + j
        )
        for j, motor in enumerate(bench.MOTORS)
    }
    spectra = bench.bench_spectra(recs, sr=sr, n_analysis=1 << 14, k_max=20)
    fit = bench.fit_profile(spectra)
    width = bench.fit_width_law_joint(spectra, k_hi=16)

    assert len(spectra) == 4
    for s in spectra:
        assert abs(s.rate_rps - rate) < 0.05, f"{s.motor}: {s.rate_rps}"
    # the profile is relative to order 2, which is the planted reference
    got = fit.profile_db[:8]
    want = profile[:8] - profile[1]
    assert np.all(np.isfinite(got)), got
    assert np.max(np.abs(got - want)) < 3.0, np.round(got - want, 2)
    # and the planted width law is linear in k with the planted slope
    assert width, "no width law fitted"
    assert 0.6 * slope < width["gamma_slope_hz_per_order"] < 1.6 * slope, width


def test_planted_speed_law_is_recovered_from_setpoints() -> None:
    from experiments.stochastic_fit import bench

    sr, slope, q = 8000.0, 0.05, 3.0
    kk = np.arange(1, 31)
    base = -0.25 * kk + np.where(kk % 2, -8.0, 0.0)
    rates = {50: 30.0, 60: 36.0, 70: 42.0, 80: 48.0, 90: 54.0}
    recs = {}
    for j, motor in enumerate(bench.MOTORS[:2]):
        for setpoint, rate in rates.items():
            # planted speed law: every line grows as q dB per dB of rev/s
            level = base + q * 10.0 * np.log10(rate / 42.0)
            recs[(motor, setpoint)] = _planted_bench_recording(
                rate, level, gamma_slope=slope, sr=sr, seconds=6.0, seed=101 + 7 * j + setpoint
            )
    spectra = bench.bench_spectra(recs, sr=sr, n_analysis=1 << 14, k_max=24)
    law = bench.fit_speed_law(spectra, k_hi=8)
    assert law["line_cells"] >= 20, law
    assert abs(law["line_exponent"] - q) < 0.8, law


# ── per-order identifiability ────────────────────────────────────────────────


def _planted_identifiability_clip(
    *,
    speeds: tuple[float, ...],
    gamma_slope: float,
    n_harm: int = 24,
    n_mics: int = 2,
    seconds: float = 4.0,
    sr: int = 16000,
    n_fft: int = 2048,
    level_db: float = 20.0,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """A flat-floor, constant-speed clip whose lines are placed by hand.

    Only the geometry matters here: the Fisher matrix is built from the fitted
    parameters, never from data, so a planted parameter set is a complete test
    case.
    """
    df = sr / n_fft
    freqs = np.arange(n_fft // 2 + 1) * df
    starts = np.arange(0, int(seconds * sr) - n_fft, n_fft // 4)
    times = starts / sr + n_fft / 2 / sr
    rotors = len(speeds)
    rps = np.tile(np.asarray(speeds, dtype=float)[:, None], (1, times.size))
    k = np.arange(1, n_harm + 1)
    params = dict(
        profile_db=np.tile(level_db - 0.2 * k, (rotors, 1)).astype(float),
        gamma=np.tile(0.3 + gamma_slope * k, (rotors, 1)).astype(float),
        knots_s=np.linspace(0.0, float(times[-1] - times[0]), 8),
        floor_mean_db=0.0,
        floor_shape_db=np.zeros(14),
        floor_ctrl_hz=np.geomspace(30.0, float(freqs[-1]), 14),
        floor_tilt_db_oct=0.0,
        floor_level_db=np.zeros(8),
        floor_tilt_gp=np.zeros(8),
        mic_floor_db=np.zeros(n_mics),
        mic_gain_db=np.zeros((n_mics, rotors)),
        gain_all_db=np.zeros(n_mics),
        amp_exp=0.0,
        floor_exp=0.0,
        floor_static_rel=1.0,
    )
    return params, freqs, times, rps


def test_marginal_information_charges_for_a_nearly_degenerate_rotor_pair():
    """Two rotors 0.4 rev/s apart cannot both own their low orders.

    Order k of one rotor sits ``k * delta`` from the same order of the other,
    so a small speed difference leaves the FUNDAMENTALS on top of each other
    while the high orders separate cleanly. The isolated width cannot see this
    at all — it is the same in both layouts — and the marginal width is what
    decides whether a per-rotor profile value means anything.
    """
    from experiments.stochastic_fit.identify import marginal_information

    def fit(**over) -> Any:
        params, freqs, times, rps = _planted_identifiability_clip(gamma_slope=0.02, **over)
        return marginal_information(params, freqs=freqs, times=times, rps=rps, n_mics=2)

    apart = fit(speeds=(80.0, 61.0))
    close = fit(speeds=(80.0, 80.4))
    # The per-line noise is the same in both, so the isolated widths agree.
    assert close.isolated_std_db[0, 0] == pytest.approx(apart.isolated_std_db[0, 0], rel=0.5)
    # Separated rotors pay almost nothing for sharing the spectrum.
    assert apart.std_db[0, 0] < 1.5
    # A degenerate pair pays an order of magnitude at the fundamental ...
    assert close.std_db[0, 0] > 4.0
    assert close.std_db[0, 0] > 5.0 * close.isolated_std_db[0, 0]
    # ... and recovers as k * delta grows past the linewidth.
    assert close.std_db[0, 23] < 0.4 * close.std_db[0, 0]


def test_marginal_information_returns_the_prior_for_an_out_of_band_order():
    """An order past Nyquist carries no likelihood, so its width IS the prior.

    This is the mechanism behind the flat profile hold at the top of both
    fitted rigs: the exported value there is the prior's, and the instrument
    has to say so rather than reporting a number that looks measured.
    """
    from experiments.stochastic_fit.identify import marginal_information

    params, freqs, times, rps = _planted_identifiability_clip(
        speeds=(80.0, 61.0), gamma_slope=0.02, n_harm=140
    )
    info = marginal_information(
        params, freqs=freqs, times=times, rps=rps, n_mics=2, prior_std_db=10.0
    )
    in_band = int(np.floor(freqs[-1] / 80.0))
    assert info.std_db[0, 8] < 1.0  # a live order is measured
    assert info.std_db[0, in_band + 8] == pytest.approx(10.0, rel=1e-3)  # a dead one is the prior
    assert info.identifiable(3.0)[0, in_band + 8] == np.False_


def test_chirp_width_adds_the_sweep_the_labels_imply():
    """The known part of a line's width comes in without a free parameter.

    Order k of a rotor accelerating at ``ds/dt`` sweeps ``k ds`` hertz across an
    analysis window of ``1/df`` seconds, and a linear sweep of total width ``W``
    has half width ``W/2``. A stationary rotor must get nothing.
    """
    import numpy as np
    import torch

    from experiments.stochastic_fit.model import BASE_VARIANT, CombSpectrum, Spec

    sr, n_fft = 16000.0, 2048
    freqs = np.arange(n_fft // 2 + 1) * (sr / n_fft)
    times = np.arange(24) * (n_fft / 4) / sr
    accel = 30.0  # rev/s per second
    ramp = 80.0 + accel * (times - times[0])
    spec_kw = dict(BASE_VARIANT)
    spec_kw.pop("rps_offset")  # the carrier must be the labels for this check
    flat = Spec(
        freqs=freqs,
        times=times,
        rps=np.tile(np.full_like(times, 80.0), (2, 1)),
        n_mics=1,
        n_harm=8,
        **spec_kw,
    )
    ramped = Spec(
        freqs=freqs, times=times, rps=np.tile(ramp, (2, 1)), n_mics=1, n_harm=8, **spec_kw
    )
    with torch.no_grad():
        still = CombSpectrum(flat).gamma_frames().numpy()
        moving = CombSpectrum(ramped).gamma_frames().numpy()
        intrinsic = CombSpectrum(flat).gamma.numpy()

    # A stationary rotor's width is its intrinsic width, on every frame.
    assert np.allclose(still, intrinsic[:, :, None], atol=1e-5)
    # A ramping one gains 0.5 * k * |ds/dt| / df, and nothing else.
    df = float(freqs[1] - freqs[0])
    for k in (1, 4, 8):
        expected = intrinsic[0, k - 1] + 0.5 * k * accel / df
        assert moving[0, k - 1, len(times) // 2] == pytest.approx(expected, rel=1e-3)
    # The sweep grows with order, so it cannot be mistaken for a constant
    # offset. Differencing against the stationary rotor removes the intrinsic
    # width, which grows with order too.
    added = (moving - still)[0, :, 12]
    assert added[7] - added[0] == pytest.approx(0.5 * 7 * accel / df, rel=1e-3)


def test_corrected_forward_model_can_represent_a_sub_bin_line():
    """The width floor no longer hides a line narrower than the window.

    The old floor of 0.6 bins was 4.69 Hz at ``n_fft`` 2048, while the bench
    measures 0.14 Hz at order 4 — so the model could not express a real line at
    all, and the 4.69 Hz it kept returning was its own floor rather than a
    measurement. ``window_kernel`` already supplies the window's broadening.
    """
    import numpy as np
    import torch

    from experiments.stochastic_fit.model import BASE_VARIANT, CombSpectrum, Spec

    sr, n_fft = 16000.0, 2048
    freqs = np.arange(n_fft // 2 + 1) * (sr / n_fft)
    times = np.arange(8) * (n_fft / 4) / sr
    spec = Spec(
        freqs=freqs,
        times=times,
        rps=np.full((1, times.size), 80.0),
        n_mics=1,
        n_harm=4,
        free_gamma=True,
        **dict(BASE_VARIANT),
    )
    model = CombSpectrum(spec)
    with torch.no_grad():
        model.log_gamma_free.fill_(float(np.log(0.2)))
        narrow = float(model.gamma.max())
        spectrum = model.forward().numpy()
    # 0.2 Hz survives; under the old 0.6-bin floor it came back as 4.69 Hz,
    # and under a 0.05-bin floor as 0.39 -- still over the bench's 0.14 at k=4.
    assert narrow == pytest.approx(0.2, rel=1e-3)
    assert 0.6 * float(freqs[1] - freqs[0]) > 4.0  # the floor it would have hit
    # and the sub-bin line is still a finite, positive, power-carrying spectrum
    assert np.isfinite(spectrum).all() and (spectrum > 0).all()
