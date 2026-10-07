"""Contract tests for `models.harmonic_ports.hppnet_pyramid`.

Small windows (2048/1024/512 at 16 kHz) keep the suite fast; the production
pyramid differs only in its numbers.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.harmonic_ports.hppnet_pyramid import (
    HPPNetPyramid,
    RateConvLSTM,
    build_tap_table,
    upsample2_along_freq,
)

N_FFTS: tuple[int, ...] = (2048, 1024, 512)
LEVEL0_FMAX = 600.0
OUT_BINS = 40
K_MAX = 8
K_SUB = 3


@pytest.fixture(scope="module")
def model() -> HPPNetPyramid:
    torch.manual_seed(0)
    return HPPNetPyramid(
        n_ffts=N_FFTS, level0_fmax=LEVEL0_FMAX, out_bins=OUT_BINS, k_max=K_MAX, k_sub=K_SUB
    ).eval()


def test_output_grid_is_level0_grid(model: HPPNetPyramid) -> None:
    df0 = 16000 / N_FFTS[0]
    freqs = model.output_freqs()
    assert freqs[0] == 0.0 and len(freqs) == OUT_BINS
    assert np.allclose(np.diff(freqs), df0)
    # every level covers the same number of bins (range and step double together)
    assert model.frontend.n_bins[0] == model.frontend.n_bins[1] == model.frontend.n_bins[2]


def test_forward_shape_and_frame_grid(model: HPPNetPyramid) -> None:
    n = 16000  # 1 s
    y = model(torch.randn(2, n))
    assert y.shape == (2, model.n_maps * OUT_BINS, n // 512 + 1)
    assert torch.isfinite(y).all()


def test_taps_read_the_finest_covering_level(model: HPPNetPyramid) -> None:
    fe = model.frontend
    offsets = np.cumsum([0, *fe.n_bins[:-1]])
    mults = np.concatenate([np.arange(1, K_MAX + 1), 1.0 / np.arange(2, K_SUB + 1)])
    f = mults[:, None] * model.output_freqs()[None]
    lo = model.get_buffer("harm_lo").numpy()
    frac = model.get_buffer("harm_frac").numpy()
    valid = model.get_buffer("harm_valid").numpy()
    for j in range(len(mults)):
        for g in range(OUT_BINS):
            if not valid[j, g]:
                continue
            lvl = next(lv for lv in range(len(fe.n_bins)) if f[j, g] < fe.ceiling(lv))
            local = int(lo[j, g] - offsets[lvl])
            assert local >= 0 and local + 1 < fe.n_bins[lvl]
            assert (local + frac[j, g]) * fe.df[lvl] == pytest.approx(f[j, g], abs=1e-6)
    assert valid[-1, 1] == 0.0  # r/3 at r = one bin: below the DC-lobe floor, masked
    assert valid[0, 1] == 1.0  # the fundamental at r = one bin is read


def test_build_tap_table_masks_above_every_ceiling() -> None:
    lo, frac, valid = build_tap_table(
        np.array([1.0, 4.0]), np.array([10.0, 100.0]), [1.0, 2.0], [50.0, 150.0], [0, 60], 0.0, 1e9
    )
    assert valid.tolist() == [[1.0, 1.0], [1.0, 0.0]]  # 4*100 = 400 is above both ceilings
    assert lo[0, 1].item() == 60 + 50 and frac[0, 1].item() == 0.0  # 100 Hz read on level 1


def test_upsample2_is_exact_linear_interpolation() -> None:
    d = 0.5
    coarse = torch.tensor(3.0 * (np.arange(50) * 2 * d) + 1.0)
    for n_fine in (97, 98):  # odd and even
        up = upsample2_along_freq(coarse, n_fine)
        expect = torch.tensor(3.0 * (np.arange(n_fine) * d) + 1.0)
        assert torch.allclose(up, expect)


def test_backward_is_finite(model: HPPNetPyramid) -> None:
    model.train()
    try:
        y = model(torch.randn(1, 8000))
        y.square().mean().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) == sum(1 for _ in model.parameters())
        assert all(torch.isfinite(g).all() for g in grads)
    finally:
        model.zero_grad(set_to_none=True)
        model.eval()


def test_predict_rps_decodes_on_the_rate_grid(model: HPPNetPyramid) -> None:
    rps = model.predict_rps(torch.randn(1, 8000))
    assert rps.shape == (1, 4, 8000 // 512 + 1)
    assert float(rps.min()) >= 0.0 and float(rps.max()) <= model.output_freqs()[-1] + 1e-6


def test_rate_convlstm_state_reach_is_half_width_per_frame() -> None:
    """Output at (t, g) depends on input at (t - s, g ± s * half) and nothing wider."""
    torch.manual_seed(0)
    lstm = RateConvLSTM(3, 2, kernel_size=9, bidirectional=False).double()
    x = torch.randn(1, 3, 4, 60, dtype=torch.double, requires_grad=True)
    for step in (1, 2, 3):
        g = torch.autograd.grad(lstm(x)[0, :, step, 30].sum(), x)[0][0, :, 0]
        reach = (g.abs().sum(0) > 0).nonzero().flatten()
        assert (int(reach.min()), int(reach.max())) == (30 - 4 * step, 30 + 4 * step)


def test_rate_convlstm_directions() -> None:
    torch.manual_seed(0)
    lstm = RateConvLSTM(3, 2, kernel_size=3).double()
    x = torch.randn(1, 3, 5, 10, dtype=torch.double, requires_grad=True)
    fwd = torch.autograd.grad(lstm(x)[0, :2, 2].sum(), x)[0][0].abs().sum((0, 2))
    bwd = torch.autograd.grad(lstm(x)[0, 2:, 2].sum(), x)[0][0].abs().sum((0, 2))
    assert (fwd > 0).tolist() == [True, True, True, False, False]
    assert (bwd > 0).tolist() == [False, False, True, True, True]


def test_convlstm_head_kernel_from_max_slew() -> None:
    torch.manual_seed(0)
    m = HPPNetPyramid(
        n_ffts=N_FFTS,
        level0_fmax=LEVEL0_FMAX,
        out_bins=OUT_BINS,
        k_max=K_MAX,
        k_sub=K_SUB,
        head="convlstm",
        lstm_size=16,
        max_slew=15.0,
    ).eval()
    dr = 16000 / N_FFTS[0]  # 7.8125 rev/s -> half 2 -> kernel 5
    assert isinstance(m.temporal, RateConvLSTM)
    assert m.temporal.kernel_size == 2 * int(np.ceil(15.0 / dr)) + 1 == 5
    y = m(torch.randn(1, 8000))
    assert y.shape == (1, 4 * OUT_BINS, 8000 // 512 + 1)
    with pytest.raises(ValueError):
        HPPNetPyramid(n_ffts=N_FFTS, level0_fmax=LEVEL0_FMAX, out_bins=OUT_BINS, head="lstm")
