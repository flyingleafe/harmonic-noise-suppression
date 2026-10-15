"""The encode/decode pair must lose nothing on real-shaped trajectories, on the
linear protocol grid (bin 0 = the 0 rev/s rate bin) and on a log grid with its
off state at bin 0 (`models.salience_crf`, docstring item 3)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.salience_crf import (
    MAX_STEP_REV_S,
    RP_LINEAR_GRID,
    band_for_rev_s,
    crf_decode_layers,
    gaussian_layer_target,
    grid_from_spec,
    layer_focus,
    refine_peak,
    transition_penalty,
)

LOG48, LOG48_OFF = grid_from_spec(
    {"kind": "log_off", "fmin": 27.5, "bins_per_octave": 48, "n_bins": 352}
)
RP, RP_OFF = grid_from_spec(
    {"kind": "linear", "fmin": 0.0, "fmax": float(RP_LINEAR_GRID[-1]), "bins": 164}
)
GRIDS = {"linear": (RP, RP_OFF), "log48": (LOG48, LOG48_OFF)}


def _tracks(seed=0, n=3, r=4, t=120, lo=30.0, hi=155.0, hard_starts=True):
    """OU-like trajectories inside ``[lo, hi]`` with stopped stretches.

    With ``hard_starts`` each rotor is stopped (exactly 0) for its first
    ``5 + 3k`` frames and starts at full speed in one frame.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((n, r, t))
    for b in range(n):
        base = np.cumsum(rng.standard_normal(t)) * 0.6 + 80.0
        for k in range(r):
            track = base + rng.standard_normal(t).cumsum() * 0.2 + k * 7.0
            out[b, k] = np.clip(track, lo, hi)
            if hard_starts:
                out[b, k, : 5 + 3 * k] = 0.0
    return torch.tensor(out)


def test_the_spec_kind_decides_the_off_state():
    assert RP_OFF is False and LOG48_OFF is True
    assert RP.shape == (164,) and RP[0] == 0.0  # bin 0 is a rate on the linear grid
    assert LOG48.shape == (353,) and LOG48[0] == 0.0 and LOG48[1] == 27.5


@pytest.mark.parametrize("name", list(GRIDS))
def test_round_trip_is_exact_on_running_rotors(name):
    grid, has_off = GRIDS[name]
    rps = _tracks(hard_starts=False)
    sal = gaussian_layer_target(rps, grid, sigma_rev_s=1.0, has_off=has_off)
    rec = crf_decode_layers(sal, grid, has_off=has_off)
    assert float((rec - rps).abs().max()) < 1e-6


@pytest.mark.parametrize("name", list(GRIDS))
def test_a_stopped_rotor_decodes_to_exactly_zero(name):
    """Zero is a value on both grids: the off state on log, the 0 rev/s bin on linear."""
    grid, has_off = GRIDS[name]
    rps = torch.zeros(1, 4, 40, dtype=torch.float64)
    rps[0, 1] = 61.0
    rec = crf_decode_layers(
        gaussian_layer_target(rps, grid, 1.0, has_off=has_off), grid, has_off=has_off
    )
    assert torch.equal(rec[0, 0], torch.zeros(40, dtype=torch.float64))
    assert abs(float(rec[0, 1].mean()) - 61.0) < 1e-6


def test_a_start_from_0_to_60_in_one_frame_is_free_only_through_the_off_state():
    """60 rev/s in one frame is 2.4x the free slew. The log grid's off state
    makes it free and the round trip is exact; on the linear grid it is an
    ordinary 60 rev/s jump paid for by the hinge, and the best path shaves it
    through an intermediate bin (one frame ~3 rev/s off), the pre-2026-10
    behaviour of every linear-grid run. Real telemetry never moves more than
    23.8 rev/s in a frame (`MAX_STEP_REV_S`), so this start is synthetic."""
    rps = torch.zeros(1, 4, 20, dtype=torch.float64)
    rps[0, :, 10:] = 60.0
    rec = crf_decode_layers(
        gaussian_layer_target(rps, LOG48, 1.0, has_off=True), LOG48, has_off=True
    )
    assert float((rec - rps).abs().max()) < 1e-6
    pen_log = transition_penalty(LOG48, MAX_STEP_REV_S, has_off=True, dtype=torch.float64)
    pen_lin = transition_penalty(RP, MAX_STEP_REV_S, has_off=False, dtype=torch.float64)
    i60_log = int(np.argmin(np.abs(LOG48 - 60.0)))
    i60_lin = int(np.argmin(np.abs(RP - 60.0)))
    assert float(pen_log[0, i60_log]) == 0.0 and float(pen_log[i60_log, 0]) == 0.0
    assert float(pen_lin[0, i60_lin]) > 0.0 and float(pen_lin[0, i60_lin]) < 1e20
    rec = crf_decode_layers(gaussian_layer_target(rps, RP, 1.0, has_off=False), RP, has_off=False)
    err = (rec - rps).abs()
    shaved = (err > 1e-6).any(dim=(0, 1))  # (T,)
    assert shaved.nonzero().flatten().tolist() == [9, 10]  # the two frames at the jump
    assert 0.0 < float(err.max()) < 4.0


@pytest.mark.parametrize("name", list(GRIDS))
def test_rate_to_rate_obeys_the_rev_s_hinge_on_any_grid(name):
    grid, has_off = GRIDS[name]
    pen = transition_penalty(grid, MAX_STEP_REV_S, has_off=has_off, dtype=torch.float64)
    i40, i60, i100 = (int(np.argmin(np.abs(grid - r))) for r in (40.0, 60.0, 100.0))
    assert float(pen[i40, i60]) == 0.0  # 20 rev/s: inside the free slew
    assert float(pen[i40, i100]) > 0.0  # 60 rev/s: past it
    d = torch.as_tensor(np.abs(grid[:, None] - grid[None, :]))[1:, 1:]
    assert torch.all(pen[1:, 1:][d <= MAX_STEP_REV_S] == 0.0)
    assert torch.all(pen[1:, 1:][d > 4 * MAX_STEP_REV_S] > 1e20)


def test_the_parabola_never_straddles_the_off_state():
    """A path at the first rate bin reads within the rate bins, however loud bin 0 is."""
    grid = torch.as_tensor(LOG48)
    s = torch.full((1, len(grid), 1), -50.0, dtype=torch.float64)
    s[0, 0, 0], s[0, 1, 0], s[0, 2, 0] = 0.0, -0.1, -3.0
    rate = refine_peak(s, torch.tensor([[1]]), grid, has_off=True)
    assert float(grid[1]) <= float(rate) <= float(grid[3])
    assert float(refine_peak(s, torch.tensor([[0]]), grid, has_off=True)) == 0.0


def test_on_the_linear_grid_bin_0_is_refined_like_any_rate():
    """A Gaussian at 0.4 rev/s peaks at bin 0; the parabola through bins 0-2
    recovers 0.4, where an off-state readout would have returned 0.0."""
    grid = torch.as_tensor(RP)
    rps = torch.full((1, 1, 1), 0.4, dtype=torch.float64)
    s = torch.log(gaussian_layer_target(rps, RP, 1.0, has_off=False)[0])  # (1, G, 1)
    idx = s[0].argmax(dim=0)
    assert int(idx) == 0
    assert float(refine_peak(s, idx[None], grid, has_off=False)) == pytest.approx(0.4, abs=1e-9)


def test_the_target_puts_a_stopped_rotor_on_bin_0_only_on_the_log_grid():
    rps = torch.tensor([[[0.0, 0.3, 45.0]]], dtype=torch.float64)
    tgt = gaussian_layer_target(rps, LOG48, 1.0, has_off=True)[0, 0]  # (G, T)
    assert torch.equal(tgt[:, 0], tgt[:, 1])
    assert float(tgt[0, 0]) == 1.0 and float(tgt[1:, 0].abs().sum()) == 0.0
    assert float(tgt[0, 2]) == 0.0 and float(tgt[1:, 2].max()) > 0.9


def test_the_target_is_a_plain_gaussian_on_the_linear_grid():
    """A stopped rotor is a peak at 0 rev/s with the usual tail into bins 1-2."""
    rps = torch.tensor([[[0.0, 0.3, 45.0]]], dtype=torch.float64)
    tgt = gaussian_layer_target(rps, RP, 1.0, has_off=False)[0, 0]  # (G, T)
    assert float(tgt[0, 0]) == 1.0
    assert float(tgt[1, 0]) == pytest.approx(np.exp(-0.5 * float(RP[1]) ** 2))
    assert float(tgt[0, 1]) == pytest.approx(np.exp(-0.5 * 0.3**2))  # not snapped to 1
    assert int(tgt[:, 2].argmax()) == int(np.argmin(np.abs(RP - 45.0)))


def test_auto_focus_is_g_over_the_gaussian_mass():
    step = float(RP[1])
    want = len(RP) / (np.sqrt(2 * np.pi) * 1.0 / step)
    assert layer_focus(RP, 1.0, has_off=False) == pytest.approx(want, rel=0.01)
    # On the log grid the same formula averages over the represented rates.
    assert layer_focus(LOG48, 1.0, has_off=True) > layer_focus(RP, 1.0, has_off=False)


def test_grid_specs_reject_typos_and_log_grids_without_an_off_state():
    with pytest.raises(ValueError, match="takes keys"):
        grid_from_spec({"kind": "linear", "fmin": 0.0, "fmax": 150.0, "n_bins": 300})
    grid, has_off = grid_from_spec({"kind": "linear", "fmin": 20.0, "fmax": 150.0, "bins": 300})
    assert has_off is False and grid[0] == 20.0  # a linear grid may start anywhere
    with pytest.raises(ValueError, match="off state"):
        transition_penalty(grid, MAX_STEP_REV_S, has_off=True)


def test_the_band_admits_what_real_rotors_do():
    """15.3 rev/s in one frame is 31 bins at 0.5 rev/s; 3 bins cannot hold it."""
    span, _ = band_for_rev_s(15.3, 0.5)
    assert span >= 31


def test_flat_float32_maps_keep_the_discrete_path():
    """Sub-bin fitting cannot invent a vertex on an uninformative plateau."""
    layers = torch.zeros(1, 4, len(RP), 5, dtype=torch.float32)
    for logits in (False, True):
        discrete = crf_decode_layers(layers, RP, has_off=False, logits=logits, subgrid=False)
        refined = crf_decode_layers(layers, RP, has_off=False, logits=logits)
        torch.testing.assert_close(refined, discrete, rtol=0, atol=0)
