"""Fused Triton kernels for `models.harmonic_ports.tap_conv`.

The torch path gathers every tap into a ``(G·B·T, J_c·C)`` block, lerps,
transposes and runs a GEMM — ~15 memory passes per tap, which measured
~6 ms per tap per 700k rows on a T4 and left the tap layers at 58 % of a
pyramid step. Here one program owns a tile of ``BG`` rate bins of one
``(b, t)`` frame and, per tap, reads the two interpolation rows straight from
``x``, lerps in registers and feeds the tensor cores; the ``(BG, O)``
accumulator is written once. Backward is two kernels: ``dx`` (per tile,
``gh = gy · W_jᵀ`` scattered with fp32 atomics onto the two source rows) and
``dw`` (per tap and row split, ``gyᵀ · a_j`` accumulated in registers,
partials summed in torch).

``x`` is read as ``(N, F, C)`` (``N = B·T``), i.e. the model's ``(B, C, T, F)``
transposed once; ``y`` is written directly in the model's ``(B, O, T, G)``.
Importable without triton (``HAS_TRITON``); `tap_conv` falls back to the
torch path.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised only on triton-less installs
    HAS_TRITON = False

__all__ = ["HAS_TRITON", "tap_conv_triton"]


if HAS_TRITON:

    @triton.jit
    def _fwd_kernel(
        XT, W, BIAS, LO, WA, WC, Y,
        T, G, F, J,
        C: tl.constexpr, CO: tl.constexpr,
        BC: tl.constexpr, BO: tl.constexpr, BG: tl.constexpr,
    ):  # fmt: skip
        n = tl.program_id(0)
        pg = tl.program_id(1)
        b = n // T
        t = n % T
        offs_g = pg * BG + tl.arange(0, BG)
        offs_c = tl.arange(0, BC)
        offs_o = tl.arange(0, BO)
        gmask = offs_g < G
        cmask = offs_c < C
        omask = offs_o < CO

        bias = tl.load(BIAS + offs_o, mask=omask, other=0.0).to(tl.float32)
        acc = tl.zeros((BG, BO), dtype=tl.float32) + bias[None, :]
        xrow = XT + n.to(tl.int64) * F * C
        for j in range(0, J):
            lo = tl.load(LO + j * G + offs_g, mask=gmask, other=0)
            wa = tl.load(WA + j * G + offs_g, mask=gmask, other=0.0).to(tl.float32)
            wc = tl.load(WC + j * G + offs_g, mask=gmask, other=0.0).to(tl.float32)
            pa = xrow + lo[:, None] * C + offs_c[None, :]
            xa = tl.load(pa, mask=cmask[None, :], other=0.0).to(tl.float32)
            xc = tl.load(pa + C, mask=cmask[None, :], other=0.0).to(tl.float32)
            a = xa * wa[:, None] + xc * wc[:, None]
            w = tl.load(
                W + j * CO * C + offs_o[None, :] * C + offs_c[:, None],
                mask=cmask[:, None] & omask[None, :],
                other=0.0,
            )  # (BC, BO)
            acc += tl.dot(a.to(w.dtype), w, input_precision="ieee")
        py = Y + ((b.to(tl.int64) * CO + offs_o[None, :]) * T + t) * G + offs_g[:, None]
        tl.store(py, acc.to(Y.dtype.element_ty), mask=gmask[:, None] & omask[None, :])

    @triton.jit
    def _dx_kernel(
        XT_GRAD, W, LO, WA, WC, GY,
        T, G, F, J,
        C: tl.constexpr, CO: tl.constexpr,
        BC: tl.constexpr, BO: tl.constexpr, BG: tl.constexpr,
    ):  # fmt: skip
        n = tl.program_id(0)
        pg = tl.program_id(1)
        b = n // T
        t = n % T
        offs_g = pg * BG + tl.arange(0, BG)
        offs_c = tl.arange(0, BC)
        offs_o = tl.arange(0, BO)
        gmask = offs_g < G
        cmask = offs_c < C
        omask = offs_o < CO

        pgy = GY + ((b.to(tl.int64) * CO + offs_o[None, :]) * T + t) * G + offs_g[:, None]
        gy = tl.load(pgy, mask=gmask[:, None] & omask[None, :], other=0.0)  # (BG, BO)
        grow = XT_GRAD + n.to(tl.int64) * F * C
        for j in range(0, J):
            lo = tl.load(LO + j * G + offs_g, mask=gmask, other=0)
            wa = tl.load(WA + j * G + offs_g, mask=gmask, other=0.0).to(tl.float32)
            wc = tl.load(WC + j * G + offs_g, mask=gmask, other=0.0).to(tl.float32)
            wt = tl.load(
                W + j * CO * C + offs_o[:, None] * C + offs_c[None, :],
                mask=omask[:, None] & cmask[None, :],
                other=0.0,
            )  # (BO, BC)
            gh = tl.dot(gy.to(wt.dtype), wt, input_precision="ieee")  # (BG, BC) fp32
            pa = grow + lo[:, None] * C + offs_c[None, :]
            m = gmask[:, None] & cmask[None, :]
            tl.atomic_add(pa, gh * wa[:, None], mask=m)
            tl.atomic_add(pa + C, gh * wc[:, None], mask=m)

    @triton.jit
    def _dw_kernel(
        XT, LO, WA, WC, GY, GW_PART,
        T, G, F, R, RPS, SPLITS,
        C: tl.constexpr, CO: tl.constexpr,
        BC: tl.constexpr, BO: tl.constexpr, BR: tl.constexpr,
    ):  # fmt: skip
        j = tl.program_id(0)
        s = tl.program_id(1)
        offs_c = tl.arange(0, BC)
        offs_o = tl.arange(0, BO)
        cmask = offs_c < C
        omask = offs_o < CO
        acc = tl.zeros((BO, BC), dtype=tl.float32)
        r0 = s * RPS
        r1 = tl.minimum(r0 + RPS, R)
        for rb in range(r0, r1, BR):
            offs_r = rb + tl.arange(0, BR)
            rmask = offs_r < r1
            n = offs_r // G
            g = offs_r % G
            b = n // T
            t = n % T
            lo = tl.load(LO + j * G + g, mask=rmask, other=0)
            wa = tl.load(WA + j * G + g, mask=rmask, other=0.0).to(tl.float32)
            wc = tl.load(WC + j * G + g, mask=rmask, other=0.0).to(tl.float32)
            pa = XT + (n[:, None].to(tl.int64) * F + lo[:, None]) * C + offs_c[None, :]
            m = rmask[:, None] & cmask[None, :]
            xa = tl.load(pa, mask=m, other=0.0).to(tl.float32)
            xc = tl.load(pa + C, mask=m, other=0.0).to(tl.float32)
            a = xa * wa[:, None] + xc * wc[:, None]  # (BR, BC)
            pgy = (
                GY
                + ((b[None, :].to(tl.int64) * CO + offs_o[:, None]) * T + t[None, :]) * G
                + g[None, :]
            )
            gy = tl.load(pgy, mask=omask[:, None] & rmask[None, :], other=0.0)  # (BO, BR)
            acc += tl.dot(gy, a.to(gy.dtype), input_precision="ieee")
        pw = GW_PART + (j * SPLITS + s) * CO * C + offs_o[:, None] * C + offs_c[None, :]
        tl.store(pw, acc, mask=omask[:, None] & cmask[None, :])

    def _pow2(n: int, lo: int = 16) -> int:
        return max(lo, 1 << (int(n) - 1).bit_length())

    class _TapConvTritonFn(torch.autograd.Function):
        SPLITS = 32  # row splits per tap (dw); partials summed in torch

        @staticmethod
        def _rows(c: int) -> int:
            """Rows per program: the ``(rows, BC)`` operand tile plus the
            ``(BC, BO)`` weight tile must fit a 64 KB shared memory; 64 rows at
            C = 16, 16 rows at C = 128."""
            return 64 if c <= 32 else 16

        @staticmethod
        def forward(  # type: ignore[override]
            ctx,
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
            lo: torch.Tensor,
            wa: torch.Tensor,
            wc: torch.Tensor,
        ) -> torch.Tensor:
            b, c, t, f = x.shape
            j, o, _ = weight.shape
            g = lo.shape[1]
            n = b * t
            xt = x.permute(0, 2, 3, 1).reshape(n, f, c).contiguous()  # (N, F, C)
            w = weight.to(x.dtype).contiguous()
            y = torch.empty((b, o, t, g), dtype=x.dtype, device=x.device)
            bg = _TapConvTritonFn._rows(c)
            _fwd_kernel[(n, triton.cdiv(g, bg))](  # pyright: ignore[reportArgumentType]
                xt, w, bias.to(torch.float32).contiguous(), lo, wa, wc, y,
                t, g, f, j,
                C=c, CO=o, BC=_pow2(c), BO=_pow2(o), BG=bg,  # pyright: ignore[reportArgumentType]
                num_warps=4, num_stages=2,  # pyright: ignore[reportCallIssue]
            )  # fmt: skip
            ctx.save_for_backward(xt, weight, lo, wa, wc)
            ctx.dims = (b, c, t, f, g)
            return y

        @staticmethod
        def backward(ctx, gy: torch.Tensor):  # type: ignore[override]
            xt, weight, lo, wa, wc = ctx.saved_tensors
            b, c, t, f, g = ctx.dims
            j, o, _ = weight.shape
            n = b * t
            gy = gy.to(xt.dtype).contiguous()
            w = weight.to(xt.dtype).contiguous()
            bg = _TapConvTritonFn._rows(c)
            gxt = torch.zeros((n, f, c), dtype=torch.float32, device=xt.device)
            _dx_kernel[(n, triton.cdiv(g, bg))](  # pyright: ignore[reportArgumentType]
                gxt, w, lo, wa, wc, gy,
                t, g, f, j,
                C=c, CO=o, BC=_pow2(c), BO=_pow2(o), BG=bg,  # pyright: ignore[reportArgumentType]
                num_warps=4, num_stages=2,  # pyright: ignore[reportCallIssue]
            )  # fmt: skip
            r = n * g
            br = _TapConvTritonFn._rows(c)
            rps = triton.cdiv(triton.cdiv(r, _TapConvTritonFn.SPLITS), br) * br
            splits = triton.cdiv(r, rps)
            gw_part = torch.empty((j, splits, o, c), dtype=torch.float32, device=xt.device)
            _dw_kernel[(j, splits)](  # pyright: ignore[reportArgumentType]
                xt, lo, wa, wc, gy, gw_part,
                t, g, f, r, rps, splits,
                C=c, CO=o, BC=_pow2(c), BO=_pow2(o), BR=br,  # pyright: ignore[reportArgumentType]
                num_warps=4, num_stages=2,  # pyright: ignore[reportCallIssue]
            )  # fmt: skip
            gw = gw_part.sum(dim=1)
            gb = gy.float().sum(dim=(0, 2, 3))
            gx = gxt.to(xt.dtype).reshape(b, t, f, c).permute(0, 3, 1, 2)
            return gx, gw.to(weight.dtype), gb.to(weight.dtype), None, None, None


def tap_conv_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    lo: torch.Tensor,
    wa: torch.Tensor,
    wc: torch.Tensor,
) -> torch.Tensor:
    """Fused `tap_conv` on CUDA: ``(B, C, T, F) -> (B, O, T, G)``.

    ``lo (J, G)`` int32, ``wa``/``wc (J, G)`` the two interpolation weights
    (``(1 - frac) · valid`` and ``frac · valid``), fp32. Requires triton.
    """
    if not HAS_TRITON:  # pragma: no cover - exercised only on triton-less installs
        raise RuntimeError("tap_conv_triton requires triton (HAS_TRITON is False)")
    out = _TapConvTritonFn.apply(
        x.contiguous(),
        weight,
        bias,
        lo.to(torch.int32).contiguous(),
        wa.to(torch.float32).contiguous(),
        wc.to(torch.float32).contiguous(),
    )
    return out  # type: ignore[no-any-return]
