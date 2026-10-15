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
    at_opt = float(layer_pit_bce(ideal, tgt, focus=loss.focus, off_bin=True))
    torch.manual_seed(0)
    for _ in range(5):
        noisy = ideal + torch.randn_like(ideal) * 0.3
        worse = float(layer_pit_bce(noisy, tgt, focus=loss.focus, off_bin=True))
        assert worse > at_opt
    pred, target = _frames(ideal.reshape(2, -1, 12), rps)
    assert float(loss(pred, target)) == pytest.approx(at_opt, rel=1e-5)


def test_the_off_bin_is_not_focus_weighted():
    """Focus balances a sparse Gaussian peak against its grid; the off bin is a
    class present on every stopped frame, and weighting it the same way made
    an off head that says off everywhere (HPPNet-L2 pilot, 2026-10-15). With
    the off bin unweighted, saying off on ON frames must cost as much as
    saying on-at-the-wrong-rate, not 1/66 of it."""
    loss = LayerPITSalienceBCELoss(grid=LOG48, focus="auto")
    assert loss.focus > 10  # the weight that tilted the off head
    rps = torch.full((1, 4, 8), 60.0)  # every rotor ON
    tgt = gaussian_layer_target(rps, loss._freqs, loss.sigma_rev_s, has_off=True)
    ideal = torch.logit(tgt.clamp(1e-6, 1 - 1e-6))
    off_everywhere = ideal.clone()
    off_everywhere[:, :, 0] = 3.0  # p(off) ~ 0.95 above a correct map
    base = float(layer_pit_bce(ideal, tgt, focus=loss.focus, off_bin=True))
    g = ideal.shape[2]
    off_cost = float(layer_pit_bce(off_everywhere, tgt, focus=loss.focus, off_bin=True)) - base
    # The same logit error on a far-off rate bin (target 0, weight 1).
    wrong_rate = ideal.clone()
    wrong_rate[:, :, g // 2] = 3.0
    rate_cost = float(layer_pit_bce(wrong_rate, tgt, focus=loss.focus, off_bin=True)) - base
    assert off_cost == pytest.approx(rate_cost, rel=1e-4)
    # and the stopped frame's off bin still carries its (unweighted) BCE
    rps_off = torch.zeros((1, 4, 8))
    tgt_off = gaussian_layer_target(rps_off, loss._freqs, loss.sigma_rev_s, has_off=True)
    ideal_off = torch.logit(tgt_off.clamp(1e-6, 1 - 1e-6))
    said_on = ideal_off.clone()
    said_on[:, :, 0] = -3.0
    base_off = float(layer_pit_bce(ideal_off, tgt_off, focus=loss.focus, off_bin=True))
    miss = float(layer_pit_bce(said_on, tgt_off, focus=loss.focus, off_bin=True)) - base_off
    assert miss == pytest.approx(rate_cost, rel=0.05)


def test_on_the_linear_grid_a_stopped_rotor_is_a_focus_weighted_peak():
    """On the linear grid bin 0 is the 0 rev/s rate bin, not an off class: a
    stopped rotor is a Gaussian peak there and carries the focus weight of any
    peak. Saying 'stopped' on running frames costs the same as the same logit
    error on a far rate bin; missing a stopped rotor costs 1 + focus times it."""
    loss = LayerPITSalienceBCELoss(grid=RP, focus="auto")
    assert loss.has_off is False and loss.focus > 1
    rps = torch.full((1, 4, 8), 60.0)  # every rotor running
    tgt = gaussian_layer_target(rps, loss._freqs, loss.sigma_rev_s, has_off=False)
    ideal = torch.logit(tgt.clamp(1e-6, 1 - 1e-6))
    base = float(layer_pit_bce(ideal, tgt, focus=loss.focus))
    said_stopped = ideal.clone()
    said_stopped[:, :, 0] = 3.0
    stop_cost = float(layer_pit_bce(said_stopped, tgt, focus=loss.focus)) - base
    wrong_rate = ideal.clone()
    wrong_rate[:, :, ideal.shape[2] // 2] = 3.0
    rate_cost = float(layer_pit_bce(wrong_rate, tgt, focus=loss.focus)) - base
    assert stop_cost == pytest.approx(rate_cost, rel=1e-4)
    # A stopped rotor's bin-0 peak is weighted 1 + focus, through the loss itself.
    rps_off = torch.zeros((1, 4, 8))
    tgt_off = gaussian_layer_target(rps_off, loss._freqs, loss.sigma_rev_s, has_off=False)
    assert float(tgt_off[0, 0, 0, 0]) == 1.0 and float(tgt_off[0, 0, 1, 0]) > 0.0
    ideal_off = torch.logit(tgt_off.clamp(1e-6, 1 - 1e-6))
    said_running = ideal_off.clone()
    said_running[:, :, 0] = -3.0
    base_off = float(loss(*_frames(ideal_off.reshape(1, -1, 8), rps_off)))
    miss = float(loss(*_frames(said_running.reshape(1, -1, 8), rps_off))) - base_off
    assert miss == pytest.approx((1.0 + loss.focus) * rate_cost, rel=1e-3)
