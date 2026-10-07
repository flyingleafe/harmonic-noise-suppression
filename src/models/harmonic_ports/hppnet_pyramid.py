"""HPPNet on a nested multi-resolution STFT pyramid with a harmonic gather.

Design record: ``docs/pyramid-harmonic-frontend-design.md``. This module is the
second attempt at "HPPNet with its harmonic organ on a linear axis"; the first
(`hppnet_rps`, retired, kept loadable) moved the gather in front of the
convolutions and lost to the published front end. Here the published block
order is kept and only the coordinate system changes:

    audio -> L STFTs (window halving per level, one hop) -> per-level dB maps
          -> block_1 (shared)              on each level's NATIVE grid
          -> FPN top-down fusion           (coarse levels injected into finer)
          -> block_2, block_2_5 (shared)   per level
          -> HARMONIC TAPS                 gather at k*r and r/k from the finest
                                           level covering each frequency
          -> block_4, block_5              3-tap gathers at r*{1/2,1,2}, r*{2^-1/4,1,2^1/4}
          -> block_6..8                    temporal [5,1] convolutions
          -> head                          1x1 (variant C) or RateConvLSTM + 1x1 (variant A)
          -> LayerCRFReadout

WHY A PYRAMID. The CQT's advantage over a fixed STFT is its window length per
frequency (Q = 68.8 cycles: 1.7 s at 40 Hz, 30 ms at 2.3 kHz), which resolves
the BPF lines of the slow airframes. Its cost is the same rule at the top,
where lines are 30 Hz wide at 2 kHz. The pyramid keeps a 1 s window up to
``level0_fmax`` (1200 Hz) and halves it per level above, so the rate-unit
resolution ``Δf/k`` improves with harmonic order through the low harmonics and
is 4.5-9x finer than the CQT's ``0.0145 r`` everywhere.

WHY NESTED (every level starts at 0 Hz). Range and bin width double together,
so every level has the same bin count and a line is ~1.4 bins wide on every
level: one shared kernel bank serves all levels, as HPPNet's serves all CQT
octaves. Nesting is also what makes the top-down fusion possible: a coarser
level COVERS the finer one, so its first half can be upsampled x2 and added.

WHY THE TAPS ARE GATHERS. HPPNet's `HarmonicDilatedConv` reads, at pitch bin
``p``, the bins ``p - d_k, p, p + d_k`` (sub-harmonic, fundamental, harmonic)
with a per-harmonic kernel shared across pitch. On a linear axis the harmonic
is at ``k*r``: a position proportional to the hypothesis, which no fixed
dilation reaches and an index table does. Same weights for every rate, same
taps as the paper (``k = 1..K`` and ``1/k`` for ``k = 2..K_sub``), read with
linear interpolation from the level whose window suits the frequency.

TEMPORAL HEAD. `FreqGroupLSTM` runs one recurrence per rate bin and cannot move
state across bins; a rotor moves ~0.5 bins per frame on average and holds a bin
for ~2 frames outside cruise. ``head="conv1x1"`` (variant C) has temporal
convolutions and the CRF decoder as its only temporal model, which measures what
the LSTM was worth. ``head="convlstm"`` (variant A) is `RateConvLSTM`: a
bidirectional ConvLSTM over time whose gate convolutions run along the rate
axis with a half-width of ``ceil(max_slew / Δr)`` bins, ``max_slew`` being the
largest sustained speed change per frame in the raw telemetry (≈ 15 rev/s per
32 ms frame on DREGON `updown`, ≤ 8 elsewhere; design note § 7), so the state
can follow a rotor through any physical ramp. The other alternatives are § 7 of
the design note.

GRID. The output axis inherits level 0's step (``Δr = sr / n_fft[0]``, 0.977
rev/s) over ``out_bins`` bins from 0. Loss and metrics must build the same grid
(``conf/loss/salience_layers_pyr.yaml``): ``out_fmin 0``, ``out_fmax =
(out_bins - 1) * Δr``, ``out_bins``.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.harmonic_ports.hppnet_orig import _conv2d_block
from models.harmonic_ports.layer_readout import LayerCRFReadout
from models.multif0.utils import linear_freq_grid
from models.salience_rps import SalienceRPSPredictor

__all__ = [
    "HPPNetPyramid",
    "PyramidSTFT",
    "ProportionalTapConv",
    "RateConvLSTM",
    "build_tap_table",
    "upsample2_along_freq",
]


# ── front end ───────────────────────────────────────────────────────────────


class PyramidSTFT(nn.Module):
    """``L`` STFTs on one hop, each kept from DC to its own ceiling, in dB.

    Level ``l`` has window ``n_ffts[l]`` (halving per level by convention),
    bin width ``sr / n_ffts[l]`` and keeps ``n_bins[l]`` bins: enough to cover
    ``level0_fmax * 2**l`` plus one bin so that a linear read at the ceiling
    still has its right neighbour, capped at Nyquist. ``torch.stft(center=True)``
    emits ``n // hop + 1`` frames for every window, so the levels share the
    frame grid exactly.

    Output: list of ``(B, 1, T, n_bins[l])`` tensors in dB (power relative to
    the window's coherent gain, floored at ``eps``). Computed in float32 under
    autocast: STFT power of a unit-scaled clip at n_fft 16384 reaches ~1e7.
    """

    def __init__(
        self,
        sr: int = 16000,
        hop_length: int = 512,
        n_ffts: tuple[int, ...] = (16384, 8192, 4096, 2048),
        level0_fmax: float = 1200.0,
        eps: float = 1e-10,
    ):
        super().__init__()
        self.sr, self.hop = int(sr), int(hop_length)
        self.n_ffts = tuple(int(n) for n in n_ffts)
        self.eps = float(eps)
        self.df = [float(sr) / n for n in self.n_ffts]
        self.n_bins: list[int] = []
        for lvl, n in enumerate(self.n_ffts):
            n_valid = int(math.floor(float(level0_fmax) * 2**lvl / self.df[lvl])) + 1
            self.n_bins.append(min(n_valid + 1, n // 2 + 1))
        for lvl, n in enumerate(self.n_ffts):
            self.register_buffer(f"window_{lvl}", torch.hann_window(n), persistent=False)

    def ceiling(self, lvl: int) -> float:
        """Highest frequency a linear read may address on level ``lvl``."""
        return (self.n_bins[lvl] - 2) * self.df[lvl]

    def forward(self, audio: torch.Tensor) -> list[torch.Tensor]:
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        out: list[torch.Tensor] = []
        with torch.autocast(device_type=audio.device.type, enabled=False):
            x = audio.float()
            for lvl, n in enumerate(self.n_ffts):
                w = cast(torch.Tensor, getattr(self, f"window_{lvl}"))
                spec = torch.stft(
                    x, n_fft=n, hop_length=self.hop, window=w, center=True, return_complex=True
                )[:, : self.n_bins[lvl]]  # (B, F_l, T)
                power = (spec.real.square() + spec.imag.square()) / (w.sum().square())
                db = 10.0 * torch.log10(power + self.eps)
                out.append(db.transpose(1, 2).unsqueeze(1))  # (B, 1, T, F_l)
        return out


# ── fusion ──────────────────────────────────────────────────────────────────


def upsample2_along_freq(coarse: torch.Tensor, n_fine: int) -> torch.Tensor:
    """``(..., F_c)`` on a grid of step ``2Δ`` -> ``(..., n_fine)`` on step ``Δ``.

    Both grids start at DC, so fine bin ``i`` is coarse position ``i / 2``:
    even fine bins copy coarse bins, odd fine bins average their two coarse
    neighbours. Exact linear interpolation with the endpoints aligned; a
    nearest-neighbour ``i // 2`` would inject a 2-bin staircase at the spatial
    frequency the line detectors respond to. Needs ``F_c >= n_fine // 2 + 1``.
    """
    h = (n_fine + 1) // 2
    even = coarse[..., :h]
    odd = 0.5 * (coarse[..., : n_fine // 2] + coarse[..., 1 : n_fine // 2 + 1])
    if odd.shape[-1] < h:  # n_fine odd: one more even than odd sample
        odd = F.pad(odd, (0, 1))
    up = torch.stack([even, odd], dim=-1).reshape(*coarse.shape[:-1], 2 * h)
    return up[..., :n_fine]


# ── proportional taps (the gather) ──────────────────────────────────────────


def build_tap_table(
    mults: np.ndarray,
    rates_hz: np.ndarray,
    level_df: list[float],
    level_ceiling: list[float],
    level_offset: list[int],
    f_min: float,
    f_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Index table for reading ``mult * rate`` off concatenated level maps.

    For each (tap, rate) pair the frequency ``f = mult * rate`` is read from the
    FINEST level whose ceiling exceeds ``f`` (levels are ordered fine to
    coarse). Returns ``lo (J, G) long`` (index into the concatenated axis),
    ``frac (J, G)`` and ``valid (J, G)`` (0 where ``f`` is below ``f_min``, at or
    above ``f_max``, or above every ceiling). Invalid entries read bin 0 with
    frac 0 and are zeroed by ``valid``.
    """
    f = np.asarray(mults, dtype=np.float64)[:, None] * np.asarray(rates_hz, dtype=np.float64)[None]
    j, g = f.shape
    lo = np.zeros((j, g), dtype=np.int64)
    frac = np.zeros((j, g), dtype=np.float64)
    valid = np.zeros((j, g), dtype=np.float64)
    assigned = np.zeros((j, g), dtype=bool)
    for df, ceil, off in zip(level_df, level_ceiling, level_offset, strict=True):
        here = (~assigned) & (f < ceil)
        pos = f[here] / df
        lo_l = np.floor(pos)
        lo[here] = off + lo_l.astype(np.int64)
        frac[here] = pos - lo_l
        assigned |= here
    valid[assigned & (f >= f_min) & (f < f_max)] = 1.0
    lo[valid == 0] = 0
    frac[valid == 0] = 0.0
    return torch.from_numpy(lo), torch.from_numpy(frac), torch.from_numpy(valid)


class _TapConvFn(torch.autograd.Function):
    """Fused gather-lerp-GEMM over taps, with the gathered axis OUTERMOST.

    Profiled on a T4 (B=32, 2 s), the layer in ``(B, C, T, F)`` layout spent
    62 % of the whole training step in the backward's `index_add_` on the
    innermost axis (inner size 1: one uncoalesced atomicAdd per element) and
    another ~15 % in the permute copies around `einsum`. Here ``x`` is
    transposed ONCE to ``(F, B*T, C)``: every `index_select` / `index_add_`
    moves whole contiguous rows of ``B*T*C`` elements, and the per-tap
    contraction is a plain ``(G*B*T, C) @ (C, O)`` GEMM. The backward keeps
    one fp32 input gradient and recomputes the cheap gathers for the weight
    gradient, so only ``x`` is stored.
    """

    @staticmethod
    def _gather(
        xt: torch.Tensor, lo: torch.Tensor, wa: torch.Tensor, wc: torch.Tensor
    ) -> torch.Tensor:
        """``(G, N, C)`` lerp read of ``xt (F, N, C)`` at ``lo`` / ``lo + 1``."""
        a = torch.index_select(xt, 0, lo) * wa[:, None, None]
        return a.addcmul_(torch.index_select(xt, 0, lo + 1), wc[:, None, None])

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
        n_taps, c_out, _ = weight.shape
        g = lo.shape[1]
        xt = x.permute(3, 0, 2, 1).reshape(f, b * t, c)
        w = weight.to(x.dtype)
        y = bias.to(x.dtype).expand(g * b * t, c_out).clone()
        for j in range(n_taps):
            h = _TapConvFn._gather(xt, lo[j], wa[j], wc[j]).reshape(g * b * t, c)
            y.addmm_(h, w[j].t())
        ctx.save_for_backward(xt, weight, lo, wa, wc)
        ctx.dims = (b, c, t, f)
        return y.reshape(g, b, t, c_out).permute(1, 3, 2, 0)

    @staticmethod
    def backward(ctx, gy: torch.Tensor):  # type: ignore[override]
        xt, weight, lo, wa, wc = ctx.saved_tensors
        b, c, t, f = ctx.dims
        n_taps, c_out, _ = weight.shape
        g = lo.shape[1]
        w = weight.to(xt.dtype)
        acc = torch.float64 if xt.dtype == torch.float64 else torch.float32
        gyt = gy.to(xt.dtype).permute(3, 0, 2, 1).reshape(g * b * t, c_out)  # (G*N, O)
        gxt = torch.zeros((f, b * t, c), dtype=acc, device=xt.device)
        gw = torch.empty_like(weight, dtype=acc)
        for j in range(n_taps):
            h = _TapConvFn._gather(xt, lo[j], wa[j], wc[j]).reshape(g * b * t, c)
            gw[j] = (gyt.t() @ h).to(acc)
            gh = (gyt @ w[j]).reshape(g, b * t, c)  # (G, N, C)
            gxt.index_add_(0, lo[j], (gh * wa[j][:, None, None]).to(acc))
            gxt.index_add_(0, lo[j] + 1, (gh * wc[j][:, None, None]).to(acc))
        gb = gyt.to(acc).sum(dim=0)
        gx = gxt.to(xt.dtype).reshape(f, b, t, c).permute(1, 3, 2, 0)
        return gx, gw.to(weight.dtype), gb.to(weight.dtype), None, None, None


class ProportionalTapConv(nn.Module):
    """``y = Σ_j W_j · x[m_j · r]``: per-tap ``(C_out, C_in)`` weights shared across rate.

    The linear-axis form of a harmonic dilated convolution: the tap offsets are
    PROPORTIONAL to the hypothesis (``k*r``, ``r/k``, ``2r``, ...), so one weight
    per tap serves every candidate rate. Reads are linear interpolations off a
    concatenated feature axis, described by a table from :func:`build_tap_table`.

    Input ``(B, C_in, T, F_cat)``; output ``(B, C_out, T, G)``. Forward and
    backward are `_TapConvFn`: a loop over taps that keeps one
    ``(B, C_in, T, G)`` slice alive at a time and one dense input gradient.
    """

    def __init__(self, n_taps: int, c_in: int, c_out: int):
        super().__init__()
        self.n_taps, self.c_in, self.c_out = int(n_taps), int(c_in), int(c_out)
        bound = 1.0 / math.sqrt(self.c_in * self.n_taps)
        self.weight = nn.Parameter(
            torch.empty(self.n_taps, self.c_out, self.c_in).uniform_(-bound, bound)
        )
        self.bias = nn.Parameter(torch.zeros(self.c_out))

    def forward(
        self, x: torch.Tensor, lo: torch.Tensor, frac: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        fr, va = frac.to(x.dtype), valid.to(x.dtype)
        return cast(
            torch.Tensor, _TapConvFn.apply(x, self.weight, self.bias, lo, (1 - fr) * va, fr * va)
        )


# ── the model ───────────────────────────────────────────────────────────────


class RateConvLSTM(nn.Module):
    """ConvLSTM over time with gate convolutions along the rate axis.

    ``(B, C_in, T, G) -> (B, D * hidden, T, G)``, ``D = 2`` if bidirectional.
    The input contribution to the gates is a 1x1 convolution (the trunk has
    already supplied the rate context); the recurrent contribution is a
    ``k``-tap 1-D convolution of the previous hidden map, so state at rate bin
    ``g`` is updated from bins ``g ± k // 2`` of the previous frame and that is
    the state's whole reach per frame. The forget-gate bias starts at 1.
    """

    def __init__(
        self, c_in: int, hidden: int, kernel_size: int, bidirectional: bool = True
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.hidden, self.kernel_size = int(hidden), int(kernel_size)
        self.n_dir = 2 if bidirectional else 1
        pad = self.kernel_size // 2
        self.x_conv = nn.Conv2d(int(c_in), self.n_dir * 4 * self.hidden, kernel_size=1)
        self.h_conv = nn.ModuleList(
            [
                nn.Conv1d(self.hidden, 4 * self.hidden, self.kernel_size, padding=pad, bias=False)
                for _ in range(self.n_dir)
            ]
        )
        with torch.no_grad():
            bias = cast(torch.Tensor, self.x_conv.bias).view(self.n_dir, 4, self.hidden)
            bias[:, 1].fill_(1.0)

    def _run(self, gx: torch.Tensor, h_conv: nn.Conv1d, reverse: bool) -> torch.Tensor:
        b, _, t, g = gx.shape
        h = gx.new_zeros(b, self.hidden, g)
        c = gx.new_zeros(b, self.hidden, g)
        outs: list[torch.Tensor] = [h] * t
        steps = range(t - 1, -1, -1) if reverse else range(t)
        for s in steps:
            gi, gf, go, gu = (gx[:, :, s] + h_conv(h)).chunk(4, dim=1)
            c = torch.sigmoid(gf) * c + torch.sigmoid(gi) * torch.tanh(gu)
            h = torch.sigmoid(go) * torch.tanh(c)
            outs[s] = h
        return torch.stack(outs, dim=2)  # (B, hidden, T, G)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = self.x_conv(x).chunk(self.n_dir, dim=1)
        outs = [
            self._run(gx, cast(nn.Conv1d, conv), reverse=bool(d))
            for d, (gx, conv) in enumerate(zip(gates, self.h_conv, strict=True))
        ]
        return torch.cat(outs, dim=1)


class HPPNetPyramid(LayerCRFReadout, SalienceRPSPredictor):
    """Audio -> ``(B, n_maps * out_bins, T)`` rotor-rate salience logits.

    Args:
        sr, hop_length: audio rate and the frame hop (the salience time grid).
        n_ffts: pyramid windows, fine to coarse; level 0 sets the output step.
        level0_fmax: ceiling of level 0 in Hz; level ``l`` covers ``2**l`` times it.
        out_bins: output rate bins, ``r_g = g * sr / n_ffts[0]``.
        k_max: harmonics ``1..k_max``; ``k_sub``: sub-harmonics ``1/2..1/k_sub``.
        f_min, f_max: reads below/at-or-above these are masked (DC lobe; band top).
            ``f_min`` defaults to one level-0 bin.
        c_har, embedding: HPPNet's trunk widths.
        octave_taps: multipliers of ``block_4``; ``quarter_taps``: of ``block_5``.
        n_maps: per-rotor salience layers (``LayerCRFReadout``).
        head: ``"conv1x1"`` (variant C) or ``"convlstm"`` (variant A, `RateConvLSTM`
            of width ``lstm_size`` split over two directions, then 1x1).
        max_slew: rev/s per frame the ConvLSTM state must be able to follow; the
            gate kernel half-width is ``ceil(max_slew / Δr)`` bins.
    """

    def __init__(
        self,
        sr: int = 16000,
        hop_length: int = 512,
        num_rotors: int = 4,
        n_ffts: tuple[int, ...] = (16384, 8192, 4096, 2048),
        level0_fmax: float = 1200.0,
        out_bins: int = 307,
        k_max: int = 32,
        k_sub: int = 8,
        f_min: float | None = None,
        f_max: float = 7500.0,
        c_har: int = 16,
        embedding: int = 128,
        octave_taps: tuple[float, ...] = (0.5, 1.0, 2.0),
        quarter_taps: tuple[float, ...] = (2**-0.25, 1.0, 2**0.25),
        n_maps: int = 4,
        head: str = "conv1x1",
        lstm_size: int = 128,
        max_slew: float = 15.0,
    ):
        super().__init__(int(n_ffts[0]), hop_length, num_rotors)
        self.sr, self.n_maps = int(sr), int(n_maps)
        self.spec_sr, self.spec_hop = int(sr), int(hop_length)

        self.frontend = PyramidSTFT(sr, hop_length, n_ffts, level0_fmax)
        self.n_levels = len(self.frontend.n_ffts)
        dr = self.frontend.df[0]
        self.n_bins = int(out_bins)
        grid = linear_freq_grid(0.0, (self.n_bins - 1) * dr, self.n_bins)
        self.out_freqs = grid

        # Shared across levels: a line is ~1.4 bins wide on every level.
        self.block_1 = _conv2d_block(1, c_har, kernel_size=7)
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c_har, c_har, kernel_size=1) for _ in range(self.n_levels - 1)]
        )
        self.block_2 = _conv2d_block(c_har, c_har, kernel_size=7)
        self.block_2_5 = _conv2d_block(c_har, c_har, kernel_size=7)

        # Harmonic taps off the concatenated level maps.
        offsets = np.cumsum([0, *self.frontend.n_bins[:-1]]).tolist()
        self.cat_width = int(sum(self.frontend.n_bins))
        mults = np.concatenate(
            [np.arange(1, int(k_max) + 1, dtype=np.float64), 1.0 / np.arange(2, int(k_sub) + 1)]
        )
        fmin = dr if f_min is None else float(f_min)
        lo, frac, valid = build_tap_table(
            mults,
            grid,
            self.frontend.df,
            [self.frontend.ceiling(lvl) for lvl in range(self.n_levels)],
            offsets,
            fmin,
            float(f_max),
        )
        self.register_buffer("harm_lo", lo, persistent=False)
        self.register_buffer("harm_frac", frac, persistent=False)
        self.register_buffer("harm_valid", valid, persistent=False)
        self.conv_3 = ProportionalTapConv(len(mults), c_har, embedding)

        # Rate-axis context: HPPNet's octave / quarter-octave dilations as gathers
        # on the uniform output grid (one "level": step dr, ceiling (G-2)*dr).
        for name, taps in (("oct", octave_taps), ("quart", quarter_taps)):
            taps_np = np.asarray(taps, dtype=np.float64)
            lo, frac, valid = build_tap_table(
                taps_np, grid, [dr], [(self.n_bins - 2) * dr], [0], 0.0, float("inf")
            )
            # The identity tap (mult 1) must stay valid at the top bin too.
            for j in np.flatnonzero(np.isclose(taps_np, 1.0)):
                idx = torch.arange(self.n_bins)
                lo[j] = idx.clamp(max=self.n_bins - 2)
                frac[j] = (idx - lo[j]).to(frac.dtype)
                valid[j] = 1.0
            self.register_buffer(f"{name}_lo", lo, persistent=False)
            self.register_buffer(f"{name}_frac", frac, persistent=False)
            self.register_buffer(f"{name}_valid", valid, persistent=False)
        self.block_4 = ProportionalTapConv(len(octave_taps), embedding, embedding)
        self.norm_4 = nn.InstanceNorm2d(embedding)
        self.block_5 = ProportionalTapConv(len(quarter_taps), embedding, embedding)
        self.norm_5 = nn.InstanceNorm2d(embedding)
        self.block_6 = _conv2d_block(embedding, embedding, (5, 1))
        self.block_7 = _conv2d_block(embedding, embedding, (5, 1))
        self.block_8 = _conv2d_block(embedding, embedding, (5, 1))
        if head == "conv1x1":
            self.temporal: nn.Module = nn.Identity()
            head_in = embedding
        elif head == "convlstm":
            half = math.ceil(float(max_slew) / dr)
            self.temporal = RateConvLSTM(embedding, int(lstm_size) // 2, 2 * half + 1)
            head_in = 2 * (int(lstm_size) // 2)
        else:
            raise ValueError(f"head must be 'conv1x1' or 'convlstm', got {head!r}")
        self.head = nn.Conv2d(head_in, self.n_maps, kernel_size=1)

    # ── grid ────────────────────────────────────────────────────────────────

    def grid_params(self) -> dict:
        raise NotImplementedError(
            "HPPNetPyramid has no log-spaced grid; its salience axis is the linear "
            "candidate-rate grid exposed as `out_freqs`."
        )

    def output_freqs(self) -> np.ndarray:
        return np.asarray(self.out_freqs, dtype=np.float64)

    def num_grid_frames(self, n_samples: int) -> int:
        return int(n_samples) // self.spec_hop + 1

    # ── forward ─────────────────────────────────────────────────────────────

    def fuse(self, levels: list[torch.Tensor]) -> list[torch.Tensor]:
        """Top-down FPN pass over ``block_1`` outputs (fine to coarse order)."""
        fused = list(levels)
        for lvl in range(self.n_levels - 2, -1, -1):
            n_fine = fused[lvl].shape[-1]
            up = upsample2_along_freq(fused[lvl + 1], n_fine)
            fused[lvl] = fused[lvl] + self.lateral[lvl](up)
        return fused

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        maps = [self.block_1(m) for m in self.frontend(audio)]  # (B, c_har, T, F_l)
        maps = self.fuse(maps)
        maps = [self.block_2_5(self.block_2(m)) for m in maps]
        x = torch.cat(maps, dim=-1)  # (B, c_har, T, sum F_l)
        x = torch.relu(self.conv_3(x, self.harm_lo, self.harm_frac, self.harm_valid))
        x = self.norm_4(torch.relu(self.block_4(x, self.oct_lo, self.oct_frac, self.oct_valid)))
        x = self.norm_5(
            torch.relu(self.block_5(x, self.quart_lo, self.quart_frac, self.quart_valid))
        )
        x = self.block_8(self.block_7(self.block_6(x)))  # (B, emb, T, G)
        y = self.head(self.temporal(x)).transpose(2, 3)  # (B, n_maps, G, T)
        b, m, g, t = y.shape
        return y.reshape(b, m * g, t)
