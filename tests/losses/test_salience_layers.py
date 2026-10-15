"""`losses.LayerPITSalienceBCELoss` on its grid spec: the grid check, the focus
resolution, the optimum at the target on a log grid with its off state, and
bin 0 as an off class (log grid) versus the 0 rev/s rate bin (linear grid)."""

from __future__ import annotations

import pytest
import tdseries as td
import torch

from losses.salience_layers import LayerPITSalienceBCELoss, layer_pit_bce
from models.salience_crf import gaussian_layer_target, layer_focus

RP = {"kind": "linear", "fmin": 0.0, "fmax": 159.1796875, "bins": 164}
LOG48 = {"kind": "log_off", "fmin": 27.5, "bins_per_octave": 48, "n_bins": 352}


def _frames(logits: torch.Tensor, rps: torch.Tensor) -> tuple[td.Frame, td.Frame]:
    sal = td.uniform(logits.numpy(), 100, dims=("batch", "freq", "time"), t_start=0.0)
    lab = td.uniform(rps.numpy(), 100, dims=("batch", "rotor", "time"), t_start=0.0)
    return td.Frame({"salience": sal}), td.Frame({"rps": lab})


def test_a_model_on_another_grid_is_refused():
    """An L1 model (353 bins per map) under the L2 loss (164) must not train."""
    loss = LayerPITSalienceBCELoss(grid=RP)
    pred, target = _frames(torch.randn(1, 4 * 353, 10), torch.full((1, 4, 10), 60.0))
    with pytest.raises(ValueError, match="rate bins"):
        loss(pred, target)


def test_focus_auto_is_the_grid_s_own_and_other_strings_are_refused():
    loss = LayerPITSalienceBCELoss(grid=LOG48, focus="auto")
    assert loss.has_off is True
    assert loss.focus == pytest.approx(layer_focus(loss._freqs, 1.0, has_off=True))
    assert loss.focus > LayerPITSalienceBCELoss(grid=RP, focus="auto").focus
    with pytest.raises(ValueError, match="auto"):
        LayerPITSalienceBCELoss(grid=RP, focus="half")


def test_the_optimum_is_the_target_on_a_log_grid_with_stopped_rotors():
    loss = LayerPITSalienceBCELoss(grid=LOG48, focus="auto")
    rps = torch.zeros(2, 4, 12)
    rps[:, 0] = 40.0
    rps[:, 1, 6:] = 90.0
    rps[:, 2] = 130.0
    tgt = gaussian_layer_target(rps, loss._freqs, loss.sigma_rev_s, has_off=True)
    ideal = torch.logit(tgt.clamp(1e-6, 1 - 1e-6))
    at_opt = float(layer_pit_bce(ideal, tgt, focus=loss.focus))
    torch.manual_seed(0)
    for _ in range(5):
        noisy = ideal + torch.randn_like(ideal) * 0.3
        worse = float(layer_pit_bce(noisy, tgt, focus=loss.focus))
        assert worse > at_opt
    pred, target = _frames(ideal.reshape(2, -1, 12), rps)
    assert float(loss(pred, target)) == pytest.approx(at_opt, rel=1e-5)


@pytest.mark.parametrize("name", ["log48", "linear"])
def test_bin_0_is_not_focus_weighted_on_any_grid(name):
    """Focus balances a sparse Gaussian peak against its grid; bin 0 is a class
    present on every stopped frame (the off state on a log grid, the 0 rev/s
    bin on the linear grid, which a log trunk reaches only through its pooled
    off logit), and weighting it the same way made the head say "0"
    everywhere (HPPNet-L2 pilots 1, 2 and 4, 2026-10-15). With bin 0 at
    weight 1, saying "stopped" on running frames must cost as much as the same
    logit error on a far rate bin, not 1/66 of it; and missing a stopped rotor
    costs the same again, not (1 + focus) times it."""
    grid, has_off = (LOG48, True) if name == "log48" else (RP, False)
    loss = LayerPITSalienceBCELoss(grid=grid, focus="auto")
    assert loss.has_off is has_off and loss.focus > 10  # the weight that tilted the head
    rps = torch.full((1, 4, 8), 60.0)  # every rotor running
    tgt = gaussian_layer_target(rps, loss._freqs, loss.sigma_rev_s, has_off=has_off)
    ideal = torch.logit(tgt.clamp(1e-6, 1 - 1e-6))
    base = float(layer_pit_bce(ideal, tgt, focus=loss.focus))
    said_stopped = ideal.clone()
    said_stopped[:, :, 0] = 3.0  # p(stopped) ~ 0.95 above a correct map
    stop_cost = float(layer_pit_bce(said_stopped, tgt, focus=loss.focus)) - base
    wrong_rate = ideal.clone()
    wrong_rate[:, :, ideal.shape[2] // 2] = 3.0  # the same error on a far rate bin
    rate_cost = float(layer_pit_bce(wrong_rate, tgt, focus=loss.focus)) - base
    assert stop_cost == pytest.approx(rate_cost, rel=1e-4)
    # A stopped rotor's bin 0 (target 1) carries weight 1, through the loss itself.
    rps_off = torch.zeros((1, 4, 8))
    tgt_off = gaussian_layer_target(rps_off, loss._freqs, loss.sigma_rev_s, has_off=has_off)
    assert float(tgt_off[0, 0, 0, 0]) == 1.0
    ideal_off = torch.logit(tgt_off.clamp(1e-6, 1 - 1e-6))
    said_running = ideal_off.clone()
    said_running[:, :, 0] = -3.0
    base_off = float(loss(*_frames(ideal_off.reshape(1, -1, 8), rps_off)))
    miss = float(loss(*_frames(said_running.reshape(1, -1, 8), rps_off))) - base_off
    assert miss == pytest.approx(rate_cost, rel=0.05)
