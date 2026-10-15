"""The encode/decode pair must lose nothing on real-shaped trajectories, on the
linear protocol grid and on a log grid with its off state."""

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

LOG48 = grid_from_spec({"kind": "log_off", "fmin": 27.5, "bins_per_octave": 48, "n_bins": 352})
GRIDS = {"linear": RP_LINEAR_GRID, "log48": LOG48}


def _tracks(seed=0, n=3, r=4, t=120, lo=30.0, hi=155.0):
    """OU-like trajectories inside ``[lo, hi]`` with stopped stretches and hard starts.

    Each rotor is stopped (exactly 0) for its first ``5 + 3k`` frames and
    starts at full speed in one frame — the transition only the off state can
    make free.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((n, r, t))
    for b in range(n):
        base = np.cumsum(rng.standard_normal(t)) * 0.6 + 80.0
        for k in range(r):
            track = base + rng.standard_normal(t).cumsum() * 0.2 + k * 7.0
            out[b, k] = np.clip(track, lo, hi)
            out[b, k, : 5 + 3 * k] = 0.0
    return torch.tensor(out)


@pytest.mark.parametrize("name", list(GRIDS))
def test_round_trip_is_exact(name):
    grid = GRIDS[name]
    rps = _tracks()
    sal = gaussian_layer_target(rps, grid, sigma_rev_s=1.0)
    rec = crf_decode_layers(sal, grid)
    assert float((rec - rps).abs().max()) < 1e-6


@pytest.mark.parametrize("name", list(GRIDS))
def test_a_stopped_rotor_decodes_to_exactly_zero(name):
    """Zero is the off state at bin 0, not an absence needing a threshold."""
    grid = GRIDS[name]
    rps = torch.zeros(1, 4, 40, dtype=torch.float64)
    rps[0, 1] = 61.0
    rec = crf_decode_layers(gaussian_layer_target(rps, grid, 1.0), grid)
    assert torch.equal(rec[0, 0], torch.zeros(40, dtype=torch.float64))
    assert abs(float(rec[0, 1].mean()) - 61.0) < 1e-6


@pytest.mark.parametrize("name", list(GRIDS))
def test_a_start_from_0_to_60_in_one_frame_is_decodable(name):
    """60 rev/s in one frame is 2.4x the free slew, yet off -> 60 costs nothing."""
    grid = GRIDS[name]
    rps = torch.zeros(1, 4, 20, dtype=torch.float64)
    rps[0, :, 10:] = 60.0
    rec = crf_decode_layers(gaussian_layer_target(rps, grid, 1.0), grid)
    assert float((rec - rps).abs().max()) < 1e-6


@pytest.mark.parametrize("name", list(GRIDS))
def test_only_the_off_state_jumps_for_free(name):
    """Rate -> rate obeys the rev/s hinge on any grid; off <-> anything is free."""
    grid = GRIDS[name]
    pen = transition_penalty(grid, MAX_STEP_REV_S, dtype=torch.float64)
    i40, i60, i100 = (int(np.argmin(np.abs(grid - r))) for r in (40.0, 60.0, 100.0))
    assert float(pen[0, i60]) == 0.0 and float(pen[i60, 0]) == 0.0
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
    rate = refine_peak(s, torch.tensor([[1]]), grid)
    assert float(grid[1]) <= float(rate) <= float(grid[3])
    assert float(refine_peak(s, torch.tensor([[0]]), grid)) == 0.0


def test_the_target_puts_a_stopped_rotor_on_bin_0_only():
    rps = torch.tensor([[[0.0, 0.3, 45.0]]], dtype=torch.float64)
    tgt = gaussian_layer_target(rps, LOG48, 1.0)[0, 0]  # (G, T)
    assert torch.equal(tgt[:, 0], tgt[:, 1])
    assert float(tgt[0, 0]) == 1.0 and float(tgt[1:, 0].abs().sum()) == 0.0
    assert float(tgt[0, 2]) == 0.0 and float(tgt[1:, 2].max()) > 0.9


def test_auto_focus_on_a_linear_grid_is_g_over_the_gaussian_mass():
    step = float(RP_LINEAR_GRID[1])
    want = len(RP_LINEAR_GRID) / (np.sqrt(2 * np.pi) * 1.0 / step)
    assert layer_focus(RP_LINEAR_GRID, 1.0) == pytest.approx(want, rel=0.01)


def test_grid_specs_reject_typos_and_grids_without_an_off_state():
    with pytest.raises(ValueError, match="takes keys"):
        grid_from_spec({"kind": "linear", "fmin": 0.0, "fmax": 150.0, "n_bins": 300})
    with pytest.raises(ValueError, match="off state"):
        grid_from_spec({"kind": "linear", "fmin": 20.0, "fmax": 150.0, "bins": 300})
    assert LOG48.shape == (353,) and LOG48[0] == 0.0 and LOG48[1] == 27.5


def test_the_band_admits_what_real_rotors_do():
    """15.3 rev/s in one frame is 31 bins at 0.5 rev/s; 3 bins cannot hold it."""
    span, _ = band_for_rev_s(15.3, 0.5)
    assert span >= 31


def test_flat_float32_maps_keep_the_discrete_path():
    """Sub-bin fitting cannot invent a vertex on an uninformative plateau."""
    layers = torch.zeros(1, 4, len(RP_LINEAR_GRID), 5, dtype=torch.float32)
    for logits in (False, True):
        discrete = crf_decode_layers(layers, RP_LINEAR_GRID, logits=logits, subgrid=False)
        refined = crf_decode_layers(layers, RP_LINEAR_GRID, logits=logits)
        torch.testing.assert_close(refined, discrete, rtol=0, atol=0)
