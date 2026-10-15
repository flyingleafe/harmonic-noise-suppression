"""The adaptation ladder of the four multi-pitch baselines and L2's translation head.

L0 is the published shared map (``tests/models/test_harmonic_orig.py`` and
``test_salience_rps.py`` cover it); here: L1 = four maps on the model's own log
grid plus an off bin, L2 = L1 re-gridded onto the 164-bin protocol grid by a
fixed matrix and one identity-initialised depthwise convolution
(docs/experiments/paper-real-protocol.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from losses.salience_layers import LayerPITSalienceBCELoss, layer_pit_bce
from models.harmonic_ports import HarmoF0Orig, HPPNetOrig
from models.registry import build_model, count_parameters
from models.salience_crf import RP_LINEAR_GRID, gaussian_layer_target, grid_from_spec, log_off_grid
from models.salience_rps import (
    UNREACHABLE_LOGIT,
    BasicPitchSalience,
    LateDeepSalience,
    SalienceRPSPredictor,
    TranslationHead,
    translation_matrix,
)

CONF = Path(__file__).resolve().parents[2] / "conf"
SR = 16000
STEP = float(RP_LINEAR_GRID[1])
RP_SPEC = {"kind": "linear", "fmin": 0.0, "fmax": 163 * STEP, "bins": 164}

# name -> (class, extra kwargs, own log grid (fmin, bins/octave, n_bins), hop)
BASELINES = {
    "hppnet": (HPPNetOrig, {}, (27.5, 48, 352), 512),
    "harmof0": (HarmoF0Orig, {}, (27.5, 48, 352), 512),
    "basicpitch": (BasicPitchSalience, {}, (27.5, 36, 264), 256),
    "latedeep": (LateDeepSalience, {"fmin": 32.7}, (32.7, 60, 360), 256),
}
LOG_GRIDS = {name: log_off_grid(*spec[2]) for name, spec in BASELINES.items()}

# model config -> the loss / metrics config it trains under (same grid spec).
PAIRS = {
    "hppnet_l1": "salience_layers_log48",
    "harmof0_l1": "salience_layers_log48",
    "basicpitch_l1": "salience_layers_log36_h256",
    "latedeep_l1": "salience_layers_log60_h256",
    "hppnet_l2": "salience_layers_rp",
    "harmof0_l2": "salience_layers_rp",
    "basicpitch_l2": "salience_layers_rp_h256",
    "latedeep_l2": "salience_layers_rp_h256",
    "hppnet_l2_convlstm": "salience_layers_rp",
    "hppnet_l2k84_convlstm": "salience_layers_rp",
    "hppnet_pyramid_convlstm": "salience_layers_rp",
    "hppnet_pyramid_convlstm_dense": "salience_layers_rp",
}


def _build(name: str, **kw):
    cls, extra, _, _ = BASELINES[name]
    return cls(n_fft=2048, hop_length=512, num_rotors=4, **extra, **kw)


# ─── the translation matrix ────────────────────────────────────────────────


@pytest.mark.parametrize("name", list(LOG_GRIDS))
def test_translation_rows_sum_to_one_or_are_unreachable_zeros(name):
    """The L1 off logit feeds linear bin 0 and nothing else; on the linear grid
    the loss reads that bin as the 0 rev/s rate bin (a stopped rotor's peak)."""
    w, unreachable = translation_matrix(LOG_GRIDS[name], RP_LINEAR_GRID)
    sums = w.sum(axis=1)
    assert np.allclose(sums[~unreachable], 1.0, atol=1e-6)
    assert np.all(w[unreachable] == 0.0)
    # The off logit lands on output bin 0 (0 rev/s) and nowhere else.
    assert w[0, 0] == 1.0 and w[0, 1:].sum() == 0.0 and w[1:, 0].sum() == 0.0


@pytest.mark.parametrize("name", list(LOG_GRIDS))
def test_unreachable_rows_are_exactly_the_rates_below_the_trunk_grid(name):
    """Zero rows hold no log bin in their ±half-bin window: all below the grid's fmin."""
    fmin = LOG_GRIDS[name][1]
    _, unreachable = translation_matrix(LOG_GRIDS[name], RP_LINEAR_GRID)
    below = (RP_LINEAR_GRID > 0) & (fmin >= RP_LINEAR_GRID + STEP / 2)
    assert np.array_equal(unreachable, below)
    assert int(np.flatnonzero(~unreachable[1:])[0]) + 1 == int(np.ceil(fmin / STEP - 0.5))


@pytest.mark.parametrize("name", list(LOG_GRIDS))
def test_no_log_bin_inside_the_output_range_is_skipped(name):
    grid = LOG_GRIDS[name]
    w, _ = translation_matrix(grid, RP_LINEAR_GRID)
    in_range = grid[1:] <= RP_LINEAR_GRID[-1]
    assert np.all((w[:, 1:][:, in_range] > 0).any(axis=0))


def test_the_identity_initialised_head_is_pure_regridding():
    """At init the head is the fixed matrix, so the L1 off logit passes
    straight through to linear bin 0, which L2's loss and CRF treat as the
    0 rev/s rate bin."""
    grid = LOG_GRIDS["hppnet"]
    head = TranslationHead(grid, RP_LINEAR_GRID, n_maps=4)
    assert count_parameters(head) == 4 * (9 + 1)
    x = torch.randn(2, 4, len(grid), 7)
    y = head(x)
    w = torch.as_tensor(translation_matrix(grid, RP_LINEAR_GRID)[0])
    want = torch.einsum("oi,bmit->bmot", w, x)
    reach = ~head.unreachable  # type: ignore[operator]
    torch.testing.assert_close(y[:, :, reach], want[:, :, reach])
    assert torch.equal(y[:, :, 0], x[:, :, 0])
    assert torch.all(y[:, :, ~reach] == UNREACHABLE_LOGIT)


# ─── the four baselines at L1 and L2 ───────────────────────────────────────


@pytest.mark.parametrize("name", list(BASELINES))
@pytest.mark.parametrize("level", ["l1", "l2"])
def test_l1_and_l2_shapes_and_grids_on_one_second(name, level):
    m = _build(name, n_maps=4, translation_head=level == "l2").eval()
    hop = BASELINES[name][3]
    with torch.no_grad():
        y = m(torch.randn(2, SR))
    want_grid = LOG_GRIDS[name] if level == "l1" else RP_LINEAR_GRID
    assert y.shape == (2, 4 * len(want_grid), SR // hop + 1)
    assert np.allclose(m.output_freqs(), want_grid, rtol=1e-12, atol=0)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("name", list(BASELINES))
def test_l2_adds_exactly_the_minimal_head(name):
    l1 = count_parameters(_build(name, n_maps=4))
    l2 = count_parameters(_build(name, n_maps=4, translation_head=True))
    assert l2 - l1 == 4 * 10


def test_the_translation_head_needs_per_rotor_maps():
    with pytest.raises(ValueError, match="n_maps > 1"):
        _build("basicpitch", n_maps=1, translation_head=True)


@pytest.mark.parametrize("name", list(BASELINES))
@pytest.mark.parametrize("level", ["l1", "l2"])
def test_perfect_layers_decode_with_stops_and_hard_starts(name, level):
    """A perfect model's own output through its own CRF readout, on its own grid.

    L1 (log grid, ``out_has_off``) takes the 0 -> 60 start for free through the
    off state and a stopped rotor reads exactly 0. L2 (linear grid, no off
    state) refines a stopped rotor on the 0 rev/s bin like any rate, and pays
    the hinge for the 60 rev/s jump: the best path shaves it through ONE
    intermediate rate next to the step (see test_salience_crf.py's hard-start
    test; real telemetry never jumps that far in a frame)."""
    m = _build(name, n_maps=4, translation_head=level == "l2").double().eval()
    assert m.out_has_off is (level == "l1")
    t = 24
    rps = torch.zeros(1, 4, t, dtype=torch.float64)
    rps[0, 0] = 70.0 + torch.linspace(0, 6, t, dtype=torch.float64)
    rps[0, 1, 8:] = 60.0  # 0 -> 60 in one frame
    rps[0, 2] = 95.3
    # rotor 3 stays stopped
    tgt = gaussian_layer_target(rps, m.output_freqs(), sigma_rev_s=1.0, has_off=m.out_has_off)
    logits = torch.logit(tgt.clamp(1e-12, 1 - 1e-12)).reshape(1, -1, t)
    rec = m.decode_salience(logits)
    err = (rec - rps).abs()
    if m.out_has_off:
        assert torch.equal(rec[0, 3], torch.zeros(t, dtype=torch.float64))
        assert float(err.max()) < 1e-4
        return
    assert float(rec[0, 3].abs().max()) < 1e-9
    step = err[0, 1, 7:9] > 1e-4  # the frames either side of the step
    assert int(step.sum()) == 1
    assert 0.0 < float(rec[0, 1, 7:9][step]) < 60.0  # an intermediate rate
    rest = err.clone()
    rest[0, 1, 7:9] = 0.0
    assert float(rest.max()) < 1e-4


@pytest.mark.parametrize("name", list(BASELINES))
def test_l2_reads_zero_or_at_least_the_trunk_s_lowest_rate(name):
    """A log trunk re-gridded to linear can say 0 (the pooled off logit on bin
    0) or any rate from its own fmin up; bins 1..fmin are `UNREACHABLE_LOGIT`,
    so whatever the net emits, the CRF never decodes a rate strictly between
    0 and fmin (Dmitrii, 2026-10-15: silence or >= ~27.5 rev/s)."""
    m = _build(name, n_maps=4, translation_head=True).eval()
    assert m.translation is not None
    fmin = float(LOG_GRIDS[name][0])
    unreachable = m.translation.unreachable
    assert bool(unreachable[0]) is False and bool(unreachable[1:].any())
    torch.manual_seed(1)
    with torch.no_grad():
        y = m(torch.randn(3, SR))
    rec = m.decode_salience(y)
    low = rec[(rec > 1e-9) & (rec < fmin - 1.0)]
    assert low.numel() == 0, low[:5]


@pytest.mark.parametrize("name", ["harmof0", "basicpitch"])
def test_one_step_trains_the_off_logit_and_the_head(name):
    torch.manual_seed(0)
    m = _build(name, n_maps=4, translation_head=True).train()
    loss_fn = LayerPITSalienceBCELoss(grid=RP_SPEC, focus="auto")
    audio = torch.randn(2, SR)
    rps = torch.zeros(2, 4, SR // 512 + 1)
    rps[:, :3] = torch.tensor([60.0, 75.0, 90.0]).view(1, 3, 1)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)

    def loss() -> torch.Tensor:
        layers, tgt = loss_fn.layers_and_target(m(audio), rps)
        return layer_pit_bce(layers, tgt, focus=loss_fn.focus)

    before = loss()
    before.backward()
    assert m.off_head is not None and m.translation is not None
    for p in (m.off_head.proj.weight, m.translation.conv.weight):
        assert p.grad is not None and torch.any(p.grad != 0)
    opt.step()
    opt.zero_grad()
    assert float(loss()) < float(before)


# ─── configs: every per-rotor model trains on its own grid ─────────────────


def _yaml(kind: str, name: str) -> dict[str, Any]:
    return cast(
        dict[str, Any], OmegaConf.to_container(OmegaConf.load(CONF / kind / f"{name}.yaml"))
    )


@pytest.mark.parametrize("model_cfg", list(PAIRS))
def test_each_per_rotor_config_pairs_with_a_loss_on_its_own_grid(model_cfg):
    """The loss checks only the bin COUNT at run time; a wrong fmin or grid kind
    (bin 0 as off state vs 0 rev/s) would not raise."""
    params = dict(_yaml("model", model_cfg)["params"])
    m = cast(SalienceRPSPredictor, build_model(params.pop("name"), **params))
    for kind in ("loss", "metrics"):
        for term in _yaml(kind, PAIRS[model_cfg])["terms"]:
            grid, has_off = grid_from_spec(term["params"]["grid"])
            assert np.allclose(grid, m.output_freqs(), rtol=1e-12, atol=0), (kind, term["name"])
            assert has_off is m.out_has_off, (kind, term["name"])
