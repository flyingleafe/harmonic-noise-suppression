"""Population-profile invariants for the stochastic rig model."""

from __future__ import annotations

import numpy as np
import torch

from experiments.stochastic_fit.controls import planted_population_control
from experiments.stochastic_fit.data import Periodogram
from experiments.stochastic_fit.model import Spec
from experiments.stochastic_fit.population import PopulationFitSpec
from experiments.stochastic_fit.rig import ClipInRig, RigParams, RigSpec


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
    assert "profile_z" not in clip.state_dict()


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
