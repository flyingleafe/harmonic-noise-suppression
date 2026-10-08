"""The fused "gather taps, one GEMM" operator shared by the HPPNet families.

``y[b, o, t, g] = bias[o] + Σ_j Σ_c W[j, o, c] · x[b, c, t, pos_j(g)]`` with a
linear-interpolated read at ``pos_j(g)`` described by an index table
``(lo, frac, valid)``: ``x[lo] (1 - frac) + x[lo + 1] frac``, zeroed where
``valid`` is 0. `HPPNetPyramid` reads proportional positions (``k * r``) with
it; `HarmonicDilatedConv` reads constant shifts (``±d_k``) with it — the
published sum of dilated ``(1, 3)`` convolutions is this operator with
integer taps, so both models run the same kernel.
"""

from __future__ import annotations

from typing import cast

import torch

__all__ = ["shift_tap_table", "tap_conv"]


class _TapConvFn(torch.autograd.Function):
    """Fused gather-lerp-GEMM over taps, gathered axis OUTERMOST, taps chunked.

    Profiled on a T4 (B=32, 2 s), the layer in ``(B, C, T, F)`` layout spent
    62 % of the whole training step in the backward's `index_add_` on the
    innermost axis (inner size 1: one uncoalesced atomicAdd per element) and
    another ~15 % in the permute copies around `einsum`. Here ``x`` is
    transposed ONCE to ``(F, B*T, C)``, so every `index_select` /
    `index_add_` moves whole contiguous rows of ``B*T*C`` elements. Taps are
    then taken ``chunk`` at a time: one gather builds ``(G*B*T, chunk*C)`` and
    one GEMM against the stacked ``(chunk*C, O)`` weights accumulates into
    the output, so the sum over taps lives in the GEMM's K dimension instead
    of ``J`` read-modify-writes of the ``(G*B*T, O)`` output. The backward
    keeps one fp32 input gradient and recomputes the gathers for the weight
    gradient, so only ``x`` is stored.
    """

    @staticmethod
    def _gather(
        xt: torch.Tensor, lo: torch.Tensor, wa: torch.Tensor, wc: torch.Tensor
    ) -> torch.Tensor:
        """``(G*N, J_c*C)`` lerp read of ``xt (F, N, C)`` for a ``(J_c, G)`` tap block."""
        jc, g = lo.shape
        n, c = xt.shape[1:]
        rows = lo.t().reshape(-1)  # (G*J_c): g-major so the reshape below is a view
        wa_t, wc_t = wa.t().reshape(-1, 1, 1), wc.t().reshape(-1, 1, 1)
        h = torch.index_select(xt, 0, rows) * wa_t
        h.addcmul_(torch.index_select(xt, 0, rows + 1), wc_t)  # (G*J_c, N, C)
        return h.view(g, jc, n, c).transpose(1, 2).reshape(g * n, jc * c)

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        lo: torch.Tensor,
        wa: torch.Tensor,
        wc: torch.Tensor,
        chunk: int,
    ) -> torch.Tensor:
        b, c, t, f = x.shape
        n_taps, c_out, _ = weight.shape
        g = lo.shape[1]
        xt = x.permute(3, 0, 2, 1).reshape(f, b * t, c)
        w = weight.to(x.dtype)
        y = bias.to(x.dtype).expand(g * b * t, c_out).clone()
        for j0 in range(0, n_taps, chunk):
            j1 = min(n_taps, j0 + chunk)
            h = _TapConvFn._gather(xt, lo[j0:j1], wa[j0:j1], wc[j0:j1])
            y.addmm_(h, w[j0:j1].transpose(1, 2).reshape((j1 - j0) * c, c_out))
        ctx.save_for_backward(xt, weight, lo, wa, wc)
        ctx.dims = (b, c, t, f, chunk)
        return y.reshape(g, b, t, c_out).permute(1, 3, 2, 0)

    @staticmethod
    def backward(ctx, gy: torch.Tensor):  # type: ignore[override]
        xt, weight, lo, wa, wc = ctx.saved_tensors
        b, c, t, f, chunk = ctx.dims
        n_taps, c_out, _ = weight.shape
        g = lo.shape[1]
        n = b * t
        w = weight.to(xt.dtype)
        acc = torch.float64 if xt.dtype == torch.float64 else torch.float32
        gyt = gy.to(xt.dtype).permute(3, 0, 2, 1).reshape(g * n, c_out)  # (G*N, O)
        gxt = torch.zeros((f, n, c), dtype=acc, device=xt.device)
        gw = torch.empty_like(weight, dtype=acc)
        for j0 in range(0, n_taps, chunk):
            j1 = min(n_taps, j0 + chunk)
            jc = j1 - j0
            h = _TapConvFn._gather(xt, lo[j0:j1], wa[j0:j1], wc[j0:j1])  # (G*N, J_c*C)
            ws = w[j0:j1].transpose(1, 2).reshape(jc * c, c_out)  # rows (j, c_in), as forward
            gw[j0:j1] = (gyt.t() @ h).reshape(c_out, jc, c).transpose(0, 1).to(acc)
            gh = (gyt @ ws.t()).view(g, n, jc, c).transpose(1, 2).reshape(g * jc, n, c)
            rows = lo[j0:j1].t().reshape(-1)
            wa_t, wc_t = wa[j0:j1].t().reshape(-1, 1, 1), wc[j0:j1].t().reshape(-1, 1, 1)
            gxt.index_add_(0, rows, (gh * wa_t).to(acc))
            gxt.index_add_(0, rows + 1, (gh * wc_t).to(acc))
        gb = gyt.to(acc).sum(dim=0)
        gx = gxt.to(xt.dtype).reshape(f, b, t, c).permute(1, 3, 2, 0)
        return gx, gw.to(weight.dtype), gb.to(weight.dtype), None, None, None, None


def tap_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    lo: torch.Tensor,
    frac: torch.Tensor,
    valid: torch.Tensor,
    chunk_elems: int = 160_000_000,
) -> torch.Tensor:
    """``(B, C_in, T, F) -> (B, C_out, T, G)`` for ``weight (J, C_out, C_in)`` and a
    ``(J, G)`` table. ``chunk_elems`` bounds the gathered block (about three such
    blocks are alive at once)."""
    fr, va = frac.to(x.dtype), valid.to(x.dtype)
    per_tap = lo.shape[1] * x.shape[0] * x.shape[2] * x.shape[1]
    chunk = max(1, min(int(weight.shape[0]), chunk_elems // max(1, per_tap)))
    return cast(torch.Tensor, _TapConvFn.apply(x, weight, bias, lo, (1 - fr) * va, fr * va, chunk))


def shift_tap_table(
    offsets: torch.Tensor, n_bins: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Table for constant integer shifts: tap ``j`` at bin ``g`` reads ``g + offsets[j]``.

    Out-of-range reads are invalid (zero), which is exactly `Conv2d`'s
    ``padding='same'`` zero padding. ``lo`` is kept in ``[0, n_bins - 2]`` so the
    kernel's ``lo + 1`` read stays in range; the top bin is reached as
    ``lo = n_bins - 2, frac = 1``.
    """
    g = torch.arange(int(n_bins), dtype=torch.int64)
    pos = g[None, :] + offsets.to(torch.int64)[:, None]  # (J, G)
    valid = ((pos >= 0) & (pos < int(n_bins))).to(torch.float64)
    lo = pos.clamp(0, int(n_bins) - 2)
    frac = (pos - lo).to(torch.float64) * valid
    lo = lo * valid.to(torch.int64)
    return lo, frac, valid
