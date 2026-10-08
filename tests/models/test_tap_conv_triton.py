"""The fused Triton `tap_conv` equals the torch `_TapConvFn` (forward, dx, dW, db).

CUDA-only (skipped elsewhere). Tables with duplicate source rows (sub-harmonic
taps), invalid entries, the top-bin ``lo = F - 2, frac = 1`` read, and
non-power-of-two channel counts are all exercised, plus the two production
tables (pyramid ``k_max`` 84 on 154 bins; CQT shifts k = 2..84 on 385 bins).
"""

from __future__ import annotations

import math

import pytest
import torch

from models.harmonic_ports.hppnet_pyramid import HPPNetPyramid
from models.harmonic_ports.tap_conv import _TapConvFn, shift_tap_table
from models.harmonic_ports.tap_conv_triton import HAS_TRITON, tap_conv_triton

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_TRITON), reason="CUDA + triton required"
)
DEV = "cuda"


def _torch_path(x, w, b, lo, frac, valid) -> torch.Tensor:
    fr, va = frac.to(x.dtype), valid.to(x.dtype)
    out = _TapConvFn.apply(x, w, b, lo, (1 - fr) * va, fr * va, 8)
    assert isinstance(out, torch.Tensor)
    return out


def _compare(x, w, b, lo, frac, valid, *, rtol, atol):
    lo = lo.to(DEV)
    frac, valid = frac.to(DEV), valid.to(DEV)
    x_t = x.detach().clone().requires_grad_(True)
    w_t = w.detach().clone().requires_grad_(True)
    b_t = b.detach().clone().requires_grad_(True)
    fr, va = frac.to(x.dtype), valid.to(x.dtype)
    y_tri = tap_conv_triton(x_t, w_t, b_t, lo, (1 - fr) * va, fr * va)
    y_ref = _torch_path(x, w, b, lo, frac, valid)
    assert y_tri.shape == y_ref.shape
    torch.testing.assert_close(y_tri.float(), y_ref.float(), rtol=rtol, atol=atol)
    gy = torch.randn_like(y_ref)
    g_tri = torch.autograd.grad(y_tri, (x_t, w_t, b_t), gy)
    g_ref = torch.autograd.grad(y_ref, (x, w, b), gy)
    for name, a, r in zip(("dx", "dW", "db"), g_tri, g_ref, strict=True):
        torch.testing.assert_close(a.float(), r.float(), rtol=rtol, atol=atol, msg=name)


def _random_case(seed, b, c, t, f, g, j, dtype):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(b, c, t, f, generator=gen).to(DEV, dtype).requires_grad_(True)
    w = (torch.randn(j, 128, c, generator=gen) / math.sqrt(c * j)).to(DEV).requires_grad_(True)
    bias = torch.randn(128, generator=gen).to(DEV).requires_grad_(True)
    # Random positions incl. duplicates and the top bin; ~15 % invalid.
    pos = torch.rand(j, g, generator=gen) * (f - 1)
    lo = pos.floor().long().clamp(max=f - 2)
    frac = pos - lo
    valid = (torch.rand(j, g, generator=gen) > 0.15).double()
    return x, w, bias, lo, frac.double(), valid


@pytest.mark.parametrize("c,g,f,j", [(16, 37, 200, 11), (24, 70, 123, 5), (128, 33, 90, 3)])
def test_random_tables_fp32(c, g, f, j):
    x, w, b, lo, frac, valid = _random_case(0, 2, c, 5, f, g, j, torch.float32)
    _compare(x, w, b, lo, frac, valid, rtol=2e-3, atol=2e-3)  # tf32 dots


def test_random_table_fp16():
    x, w, b, lo, frac, valid = _random_case(1, 3, 16, 4, 300, 64, 9, torch.float16)
    _compare(x, w, b, lo, frac, valid, rtol=2e-2, atol=2e-2)


def test_pyramid_table_k84():
    """The real k ≤ 84 pyramid table on the 154-bin grid (duplicates from 1/k taps)."""
    m = HPPNetPyramid(out_bins=154, k_max=84)
    lo, frac, valid = (
        torch.as_tensor(getattr(m, k)) for k in ("harm_lo", "harm_frac", "harm_valid")
    )
    gen = torch.Generator(device="cpu").manual_seed(2)
    x = torch.randn(2, 16, 7, m.cat_width, generator=gen).to(DEV, torch.float16)
    x.requires_grad_(True)
    w = (torch.randn(lo.shape[0], 128, 16, generator=gen) / 40).to(DEV).requires_grad_(True)
    b = torch.zeros(128, device=DEV, requires_grad=True)
    _compare(x, w, b, lo, frac, valid, rtol=2e-2, atol=2e-2)


def test_cqt_shift_table_k84():
    dil = torch.tensor([round(math.log2(k) * 48) for k in range(2, 85)])
    offsets = torch.cat([-dil, torch.zeros(1, dtype=dil.dtype), dil])
    lo, frac, valid = shift_tap_table(offsets, 385)
    gen = torch.Generator(device="cpu").manual_seed(3)
    x = torch.randn(2, 16, 6, 385, generator=gen).to(DEV, torch.float16).requires_grad_(True)
    w = (torch.randn(len(offsets), 128, 16, generator=gen) / 50).to(DEV).requires_grad_(True)
    b = torch.randn(128, generator=gen).to(DEV).requires_grad_(True)
    _compare(x, w, b, lo, frac, valid, rtol=2e-2, atol=2e-2)
