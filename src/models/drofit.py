"""DroFiT — band-fused frequency-attention speech enhancer for UAV ego-noise.

Paper (implemented version = the FINAL published one):
    J. Lee, C. Jeon, H. Seo, K. Shim, T. Kang, "DroFiT: A Lightweight
    Band-Fused Frequency Attention Toward Real-Time UAV Speech Enhancement",
    Proc. Interspeech 2026, pp. 6820-6824, doi:10.21437/Interspeech.2026-1620
    (ISCA archive ``lee26n_interspeech.pdf``). The arXiv preprint
    (arXiv:2509.16945 v1 2025-09-21 and v2 2026-03-07, identical text) is an
    earlier, different model (no Pre-TCN, per-token TCN back-end, plain FFN,
    "Combine Block", 0.105 M params, 1.86 GMACs); it is used here only where it
    draws a sub-module the final paper leaves undrawn (the TCN layer).
Upstream code: none published (only the demo page https://ml-sp.github.io/DroFiT/).
This is a from-the-paper re-implementation; no code is copied. The full-/sub-band
encoder-decoder follows FSPEN (Yang et al., ICASSP 2024, the paper's ref. [24]),
which the final paper cites for the sub-band design and whose layer sizes
(kernels 6-8-6, strides 2-2-2, channels 2-4-16-32, 32 sub-band tokens) DroFiT
reuses; FSPEN's "1x1 conv for global features" is DroFiT's GConv.

Parameters: 167 884 with the defaults below (paper, Table 3: 0.168 M, -0.07 %);
Base and Lite share the same weights (Table 3 lists 0.168 M for both).
MACs (Conv/Linear weight MACs + the two attention matmuls; norms, activations
and STFT not counted): 0.328 G per 1 s of audio, 1.61 G per 5 s utterance
(Lite: 0.259 G / 1 s, 1.27 G / 5 s). Paper, Table 3: 2.14 G base, 1.21 G Lite,
time basis not stated. Every utterance in the paper is 5 s long, and the Lite
count per 5 s (1.27 G vs 1.21 G) and the near-zero MAC cost of the pooled
Post-TCN (paper: w/o Post-TCN 2.14 G, unchanged; port: -0.001 G) agree with that
basis; the base gap (base - Lite: paper 0.93 G, port 0.34 G) therefore lies in
the softmax-attention term, whose counting convention the paper does not state.

Pipeline (mono 16 kHz, STFT 1024/512 -> F = 513 bins, 31.25 frames/s)::

    x -> STFT X (513 bins)
      |- full-band encoder: (Re, Im) -> 3 x CNA (Conv1d over freq) -> GConv   -> 64 tokens x 32 ch
      |- sub-band encoder:  |X| split 32-32-64-128-257 bins -> per-group Conv1d+ELU
      |                     -> 8-8-6-5-5 tokens (32 tokens x 32 ch)
      -> Pre-TCN (L=1): TCN on full tokens, C-Linear on sub tokens, F-Linear + Norm over 96 tokens
      -> + learnable frequency positional encoding
      -> N=4 frequency-wise Transformer blocks over the 96 concatenated tokens
         (MHSA -> Dropout -> Add&Norm; per-path Conv-FFN -> Dropout -> Add&Norm)
      |- full tokens -> full-band decoder: 3 x CTNA with learnable (Conv1d+ELU) skips -> GConv -> (Re, Im)
      |- sub tokens  -> sub-band decoder: per-group concat(token, PWC+ELU skip) -> FC+ReLU -> |S| (513 bins)
      |- full tokens -> Post-TCN (M=2): avg-pool over freq -> TCN -> C-Linear -> sigmoid -> gain(t)
      -> Mask Gen: [|S|, Re, Im] -> Conv2D-ELU-Conv2D-ELU-Conv2D -> tanh -> x gain(t)
      -> complex mask M;  S_hat = M (*) X  -> iSTFT

Trace of every architectural statement of the final paper to this code:

* "input STFT ... 1024-point FFT and a 512-sample hop", 16 kHz (Sec. 3.1)
  -> ``n_fft=1024, hop_length=512``; 513 bins (Sec. 2.2, Table 2).
* Full-band encoder "composed of Conv1D-based C(T)NA (Conv(T)-Norm-Activation)
  blocks, which operate independently along the time axis while compressing
  features along the frequency axis" (Sec. 2.1) -> :class:`_CNA`, applied per
  frame to a ``(B*T, C, F)`` tensor; Table 2 "Frequency Dimension 513 -> 256 ->
  128 -> 64", "Encoder Channels 2 -> 4 -> 16 -> 32" -> ``channels=(2, 4, 16, 32)``;
  kernel (6, 8, 6) / stride (2, 2, 2) from the arXiv Table 2 (identical to FSPEN),
  padding ``(k - s) // 2`` reproduces exactly 513 -> 256 -> 128 -> 64.
* "A global convolution layer (GConv) is appended" (Sec. 2.1; Fig. 2a after the
  3rd CNA and after the 3rd CTNA) -> ``enc_gconv`` / ``dec_gconv`` (1x1 Conv1d,
  FSPEN's "1x1 convolution ... to obtain global features").
* Decoder "integrating both encoder representations and the latent module
  through learnable skip connections" (Sec. 2.1); Fig. 2a draws a "Conv1D ELU"
  on each of the three full-band skips -> ``skip_proj`` (1x1 Conv1d + ELU), the
  projected skip is concatenated with the decoder input (FSPEN fusion) and fed
  to the CTNA (:class:`_CTNA`, ConvTranspose1d-Norm-Activation) of the mirrored
  stage.
* Sub-band encoder "divides the input spectrogram into five groups along the
  frequency axis and transforms them into magnitude representations ... 513
  frequency bins are partitioned into groups of 32-32-64-128-257 bins" (Sec. 2.2)
  -> ``band_splits``; "a lightweight 1D convolutional layer is applied for
  group-wise compression" + Fig. 2c "Conv ELU" -> :class:`_SubBandEncoder`;
  Table 2 "Decimation 8, 8, 6, 5, 5" -> ``band_tokens`` (8 + 8 + 6 + 5 + 5 = 32 =
  the arXiv's ``F_S`` and FSPEN's 32 sub-band tokens).
* Sub-band decoder "follows a structure similar to the full-band decoder, with
  each group connected via learnable skip connections. Restoration is performed
  independently for each group using a fully connected (FC) layer to recover the
  original feature size" (Sec. 2.2); Fig. 2a "PWC ELU, 5 Groups x"; Fig. 2c
  "FC ReLU" -> :class:`_SubBandDecoder`.
* Pre-TCN (Sec. 2.3, Fig. 2b): "In the Full-band path, a TCN refines temporal
  features"; "the Sub-band branch ... applies a lightweight C-Linear
  (channel-wise projection)"; "an F-Linear (linear transformation) layer
  performs inter-band integration"; Fig. 2b: F-Linear followed by Norm, L = 1
  (Table 2) -> :class:`_PreTCN`.
* TCN layer: Table 2 "All TCN Kernel size: 3"; arXiv Fig. 1b draws one layer as
  Conv1D-Norm-Activation | Dropout-Conv1D-Norm with a residual around it ->
  :class:`_TCNLayer`. Table 2's receptive fields (288 / 480 / 224 / 96 ms) and
  look-aheads (128 / 224 / 96 / 32 ms) for (L, M) = (1, 2) / (0, 3) / (0, 2) /
  (1, 0) are reproduced exactly, for all four rows, only by ONE non-causal k = 3
  temporal conv per layer with dilation 1 in the Pre-TCN and 1, 2, 4, ... in the
  Post-TCN; hence the second Conv1D is pointwise (k = 1).
* Frequency-wise Transformer (Sec. 2.4, Fig. 2b): MHSA "exclusively along the
  frequency axis"; "learnable frequency positional encoding" ("P.E", added once
  before the N blocks) -> ``pos_emb``; "a single self-attention over
  concatenated full-band and sub-band tokens" -> :class:`_MHSA` over all 96
  tokens; block = MHA -> Dropout -> Norm with residual, Conv-FFN -> Dropout ->
  Norm with residual (Fig. 2b) -> :class:`_FreqTransformerBlock`; "the
  conventional FFN is replaced with a Conv-FFN composed of pointwise (PW) and
  depthwise (DW) convolutions, operating independently within each path";
  Fig. 2b draws PW-DW-PW-ReLU once for the F_F and once for the F_S tokens ->
  two :class:`_ConvFFN` per block (the DW conv runs along the token axis, so a
  path never convolves across the full/sub boundary). N = 4 (Table 2).
* DroFiT-Lite "adopts Linear Attention [27]" (Katharopoulos et al.) with the
  same parameter count (Table 3) -> ``attention="linear"`` (elu + 1 feature
  map). The base model is ``attention="softmax"``.
* Post-TCN (Sec. 2.5, Fig. 2b): "the Full-band energy is compressed via average
  pooling before being fed into the TCN" (B F_F, C, T -> B, C, T), TCN x M,
  C-Linear to (B, 1, T), sigmoid -> :class:`_PostTCN`; M = 2 (Table 2); its input
  is the Transformer's full-band output (blue path in Fig. 2a).
* Mask Generation Block (Sec. 2.6, Fig. 2b): full-band and sub-band outputs
  "concatenated and processed via a Conv2D layer, reducing the channel dimension
  to two corresponding to the real and imaginary components. A Tanh activation
  is then applied, and the result is multiplied with the Post-TCN output to
  produce the final complex mask"; Fig. 2b: Conv2D-ELU-Conv2D-ELU-Conv2D-Tanh,
  (F, T, 3) input; the arXiv names the three channels (magnitude, real,
  imaginary) -> :class:`_MaskGen`. "S_hat(f, t) = M(f, t) (*) X(f, t), followed by
  inverse STFT" -> complex multiply + ``torch.istft`` in :meth:`DroFiT.forward`.

Settings the paper leaves unstated, with the values chosen:

* STFT window: periodic Hann, ``win_length = n_fft``, ``center=True`` (Hann at
  50 % overlap satisfies COLA, so identity mask -> exact reconstruction).
* No input normalisation or power-law compression (none described).
* CNA/CTNA norm: ``BatchNorm1d`` (FSPEN's choice; foldable into the conv for the
  paper's INT8 deployment); activation ELU (FSPEN; DroFiT's skips and mask
  generator use ELU). The last CTNA keeps its Norm + ELU (the paper names every
  decoder block CTNA) and the decoder GConv provides the linear output.
* Sub-band encoder conv: kernel = stride = ceil(width / tokens) = 4, 4, 11, 26, 52
  with zero padding at the top of each group (a "decimation" conv; FSPEN uses
  overlapping kernels whose layout does not fit 8-8-6-5-5). Sub-band input is the
  linear magnitude ``|X|``. The sub-band decoder crops the same padding.
* Sub-band decoder fusion: concat(token, ELU(PWC(encoder token))) -> FC(2C -> bins
  per token) -> ReLU, as FSPEN's sub-band decoder; output is a magnitude.
* Full-band skip fusion: concatenation (FSPEN), so each CTNA takes 2 x channels.
* Pre-TCN Norm: ``LayerNorm`` over (tokens, channels) jointly (FSPEN's
  ``LayerNorm([freq, channels])``). Transformer norms: ``LayerNorm(C)``.
* TCN: ``BatchNorm1d``, ReLU (the "normalized ReLU" of Lea et al., ref. [22]),
  non-causal symmetric padding (Table 2 lists look-ahead = half the receptive
  field extension).
* Attention heads: 4 (head dim 8). Dropout: 0.1 everywhere.
* Conv-FFN: hidden width 160 (= 5 x d), DW kernel 3. The paper gives neither;
  the hidden width is the one free size chosen to match the published 0.168 M
  parameters once every other size is fixed by the paper or by FSPEN
  (H = 96 / 128 / 160 give 132 556 / 150 220 / 167 884 parameters).
* Mask generator: hidden channels 16, kernel (3, 1) = 3 frequency bins x 1 frame.
  A temporal extent > 1 would contradict Table 2's receptive fields.
* Positional encoding: learnable ``(96, 32)`` table, truncated-normal(0.02) init.
* Weight init: PyTorch defaults.

Consistency with the Table 3 ablations (parameters, port vs paper). The port
reproduces w/o Pre-TCN 1 (L = 0, M = 3): 151 372 vs 0.151 M, and nearly
w/o Pre-TCN 2 (L = 0, M = 2): 147 084 vs 0.145 M. It does not reproduce
w/o Post-TCN (159 275 vs 0.148 M) nor w/o Sub-Band (99 419 vs 0.115 M); the
paper's differences imply a TCN layer of ~6 k parameters (port: 4 288) and a
Post-TCN of ~20 k (port: 8 609), and a sub-band-only share of ~53 k (port:
68 465, i.e. H ~ 104 if the gap were all in the sub-band Conv-FFNs). Variants
tried: two k = 3 convs per TCN layer (6 336 per layer, matches both w/o Pre-TCN
rows, but contradicts all four receptive fields of Table 2, so rejected);
H = 104 (the sub-band difference then matches, 53 009, but the total drops to
136 972). No single reading of the paper fits every ablation row; the defaults
favour the explicitly stated receptive fields and the headline 0.168 M.

Not implemented: the arXiv preprint's local attention windows (w_F = w_S = 8;
the final paper states a single global self-attention) and DroFiT-Quant
(weight-only INT8 post-training quantisation of Lite; a deployment step, not an
architecture).

Deviations: none intended in the architecture. Training-side settings of the
paper are outside the model and not reproduced here: Adam, lr 1e-4, loss
(1 - beta) L_Mag + beta L_Complex + alpha L_time with alpha = 0.2, beta = 0.8
(L_Mag = L1 of log magnitudes, L_Complex = squared complex error, L_time =
negative SI-SDR), float64 training, float32 inference, 5 s utterances, epochs
chosen on validation (batch size not stated). The model is native 16 kHz, so
no resampling. Output is ``(B, 1, T)`` with the input length restored by
``istft(length=T)``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DroFiT", "build_drofit"]


class _CNA(nn.Module):
    """Conv1d over frequency -> BatchNorm1d -> ELU on ``(B*T, C, F)``."""

    def __init__(self, c_in: int, c_out: int, kernel: int, stride: int):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel, stride, padding=(kernel - stride) // 2)
        self.norm = nn.BatchNorm1d(c_out)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class _CTNA(nn.Module):
    """ConvTranspose1d over frequency -> BatchNorm1d -> ELU on ``(B*T, C, F)``."""

    def __init__(self, c_in: int, c_out: int, kernel: int, stride: int, output_padding: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            c_in,
            c_out,
            kernel,
            stride,
            padding=(kernel - stride) // 2,
            output_padding=output_padding,
        )
        self.norm = nn.BatchNorm1d(c_out)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class _TCNLayer(nn.Module):
    """Residual TCN layer on ``(N, C, T)``: dilated k-tap conv -> Norm -> ReLU ->
    Dropout -> 1x1 conv -> Norm, plus the input (arXiv Fig. 1b). Non-causal,
    symmetric padding: receptive field ``(k - 1) * dilation`` frames, half of it
    look-ahead."""

    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        if kernel % 2 != 1:
            raise ValueError(f"TCN kernel must be odd for symmetric padding, got {kernel}")
        self.conv1 = nn.Conv1d(
            channels, channels, kernel, padding=dilation * (kernel - 1) // 2, dilation=dilation
        )
        self.norm1 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, 1)
        self.norm2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop(self.act(self.norm1(self.conv1(x))))
        return x + self.norm2(self.conv2(y))


class _PreTCN(nn.Module):
    """Pre-TCN (Sec. 2.3): TCN x L on the full-band tokens, C-Linear on the
    sub-band tokens, then F-Linear across all tokens and a LayerNorm over
    (tokens, channels)."""

    def __init__(
        self,
        channels: int,
        n_full: int,
        n_sub: int,
        n_layers: int,
        kernel: int,
        dropout: float,
    ):
        super().__init__()
        self.tcn = nn.ModuleList([_TCNLayer(channels, kernel, 1, dropout) for _ in range(n_layers)])
        self.c_linear = nn.Linear(channels, channels)
        n_tok = n_full + n_sub
        self.f_linear = nn.Linear(n_tok, n_tok)
        self.norm = nn.LayerNorm([n_tok, channels])

    def forward(self, full: torch.Tensor, sub: torch.Tensor) -> torch.Tensor:
        """``full``: ``(B, T, F_F, C)``, ``sub``: ``(B, T, F_S, C)`` -> ``(B, T, F_F + F_S, C)``."""
        b, t, f_full, c = full.shape
        if self.tcn:
            y = full.permute(0, 2, 3, 1).reshape(b * f_full, c, t)  # (B F_F, C, T)
            for layer in self.tcn:
                y = layer(y)
            full = y.reshape(b, f_full, c, t).permute(0, 3, 1, 2)
        sub = self.c_linear(sub)
        z = torch.cat([full, sub], dim=2)  # (B, T, F, C)
        z = self.f_linear(z.transpose(2, 3)).transpose(2, 3)  # mix along tokens
        return self.norm(z)


class _MHSA(nn.Module):
    """Multi-head self-attention over the token (frequency) axis of ``(N, L, d)``.

    ``mode="softmax"`` is scaled dot-product attention (DroFiT base);
    ``mode="linear"`` is the linear attention of Katharopoulos et al. 2020 with
    the ``elu(x) + 1`` feature map (DroFiT-Lite). Both share the same weights."""

    def __init__(self, dim: int, n_heads: int, mode: str):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")
        if mode not in ("softmax", "linear"):
            raise ValueError(f"attention must be 'softmax' or 'linear', got {mode!r}")
        self.n_heads = n_heads
        self.mode = mode
        self.in_proj = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, length, dim = x.shape
        q, k, v = (
            self.in_proj(x)
            .reshape(n, length, 3, self.n_heads, dim // self.n_heads)
            .permute(2, 0, 3, 1, 4)
        )  # each (N, H, L, Dh)
        if self.mode == "softmax":
            y = F.scaled_dot_product_attention(q, k, v)
        else:
            q = F.elu(q) + 1.0
            k = F.elu(k) + 1.0
            kv = torch.einsum("nhld,nhle->nhde", k, v)
            denom = torch.einsum("nhld,nhd->nhl", q, k.sum(dim=2)).unsqueeze(-1)
            y = torch.einsum("nhld,nhde->nhle", q, kv) / (denom + 1e-6)
        y = y.transpose(1, 2).reshape(n, length, dim)
        return self.out_proj(y)


class _ConvFFN(nn.Module):
    """Conv-FFN of one path: PW -> DW (along tokens) -> PW -> ReLU on ``(N, L, d)``."""

    def __init__(self, dim: int, hidden: int, kernel: int):
        super().__init__()
        self.pw1 = nn.Conv1d(dim, hidden, 1)
        self.dw = nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2, groups=hidden)
        self.pw2 = nn.Conv1d(hidden, dim, 1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw2(self.dw(self.pw1(x.transpose(1, 2))))
        return self.act(y).transpose(1, 2)


class _FreqTransformerBlock(nn.Module):
    """Post-norm Transformer block over the frequency tokens (Fig. 2b)."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_full: int,
        ffn_hidden: int,
        ffn_kernel: int,
        dropout: float,
        attention: str,
    ):
        super().__init__()
        self.n_full = n_full
        self.attn = _MHSA(dim, n_heads, attention)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn_full = _ConvFFN(dim, ffn_hidden, ffn_kernel)
        self.ffn_sub = _ConvFFN(dim, ffn_hidden, ffn_kernel)
        self.drop2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.drop1(self.attn(x)))
        y = torch.cat(
            [self.ffn_full(x[:, : self.n_full]), self.ffn_sub(x[:, self.n_full :])], dim=1
        )
        return self.norm2(x + self.drop2(y))


class _PostTCN(nn.Module):
    """Post-TCN (Sec. 2.5): mean over the full-band tokens -> TCN x M (dilations
    1, 2, 4, ...) -> C-Linear -> sigmoid; ``(B, T, F_F, C)`` -> ``(B, T)``."""

    def __init__(self, channels: int, n_layers: int, kernel: int, dropout: float):
        super().__init__()
        self.tcn = nn.ModuleList(
            [_TCNLayer(channels, kernel, 2**i, dropout) for i in range(n_layers)]
        )
        self.c_linear = nn.Linear(channels, 1)

    def forward(self, full: torch.Tensor) -> torch.Tensor:
        y = full.mean(dim=2).transpose(1, 2)  # (B, C, T)
        for layer in self.tcn:
            y = layer(y)
        return torch.sigmoid(self.c_linear(y.transpose(1, 2))).squeeze(-1)


class _SubBandEncoder(nn.Module):
    """Per-group decimating Conv1d(1 -> C, k = s) + ELU over ``|X|`` (Fig. 2c)."""

    def __init__(self, splits: Sequence[int], tokens: Sequence[int], channels: int):
        super().__init__()
        self.splits = list(splits)
        self.kernels = [math.ceil(w / n) for w, n in zip(splits, tokens, strict=True)]
        self.pads = [k * n - w for k, n, w in zip(self.kernels, tokens, splits, strict=True)]
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(1, channels, k, stride=k), nn.ELU()) for k in self.kernels]
        )

    def forward(self, mag: torch.Tensor) -> list[torch.Tensor]:
        """``mag``: ``(N, 1, F)`` -> per-group ``(N, C, tokens_g)``."""
        out = []
        for band, pad, conv in zip(
            mag.split(self.splits, dim=-1), self.pads, self.convs, strict=True
        ):
            out.append(conv(F.pad(band, (0, pad))))
        return out


class _SubBandDecoder(nn.Module):
    """Per-group concat(token, PWC+ELU skip) -> FC(2C -> k_g) + ReLU, cropped
    back to the group width; returns the magnitude ``(N, F)``."""

    def __init__(
        self,
        splits: Sequence[int],
        tokens: Sequence[int],
        kernels: Sequence[int],
        channels: int,
    ):
        super().__init__()
        self.splits = list(splits)
        self.tokens = list(tokens)
        self.skip = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(channels, channels, 1), nn.ELU()) for _ in splits]
        )
        self.fc = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * channels, k), nn.ReLU()) for k in kernels]
        )

    def forward(self, x: torch.Tensor, enc: list[torch.Tensor]) -> torch.Tensor:
        """``x``: ``(N, C, F_S)`` decoder tokens, ``enc``: encoder tokens per group."""
        n = x.shape[0]
        out = []
        for xg, eg, width, skip, fc in zip(
            x.split(self.tokens, dim=-1), enc, self.splits, self.skip, self.fc, strict=True
        ):
            y = torch.cat([xg, skip(eg)], dim=1).transpose(1, 2)  # (N, tokens, 2C)
            out.append(fc(y).reshape(n, -1)[:, :width])
        return torch.cat(out, dim=-1)


class _MaskGen(nn.Module):
    """Mask Generation Block (Sec. 2.6): Conv2D-ELU-Conv2D-ELU-Conv2D -> tanh on
    ``(B, 3, F, T)`` -> ``(B, 2, F, T)`` (gain applied by the caller)."""

    def __init__(self, hidden: int, kernel_freq: int):
        super().__init__()
        k = (kernel_freq, 1)
        p = (kernel_freq // 2, 0)
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, k, padding=p),
            nn.ELU(),
            nn.Conv2d(hidden, hidden, k, padding=p),
            nn.ELU(),
            nn.Conv2d(hidden, 2, k, padding=p),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DroFiT(nn.Module):
    """DroFiT (Interspeech 2026 final version). See the module docstring."""

    window: torch.Tensor

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 512,
        enc_channels: Sequence[int] = (2, 4, 16, 32),
        enc_kernels: Sequence[int] = (6, 8, 6),
        enc_strides: Sequence[int] = (2, 2, 2),
        band_splits: Sequence[int] = (32, 32, 64, 128, 257),
        band_tokens: Sequence[int] = (8, 8, 6, 5, 5),
        n_pre_tcn: int = 1,
        n_transformer: int = 4,
        n_post_tcn: int = 2,
        tcn_kernel: int = 3,
        n_heads: int = 4,
        ffn_hidden: int = 160,
        ffn_kernel: int = 3,
        mask_hidden: int = 16,
        mask_kernel: int = 3,
        dropout: float = 0.1,
        attention: str = "softmax",
    ):
        super().__init__()
        n_freq = n_fft // 2 + 1
        if sum(band_splits) != n_freq:
            raise ValueError(f"band_splits sum to {sum(band_splits)}, expected {n_freq} bins")
        if len(enc_channels) != len(enc_kernels) + 1 or len(enc_kernels) != len(enc_strides):
            raise ValueError(
                "need len(enc_channels) == len(enc_kernels) + 1 == len(enc_strides) + 1"
            )
        if enc_channels[0] != 2:
            raise ValueError("the full-band encoder input is (Re, Im): enc_channels[0] must be 2")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        chans = list(enc_channels)
        dim = chans[-1]

        # Full-band encoder: frequency sizes per stage (513 -> 256 -> 128 -> 64).
        sizes = [n_freq]
        for k, s in zip(enc_kernels, enc_strides, strict=True):
            sizes.append((sizes[-1] + 2 * ((k - s) // 2) - k) // s + 1)
        self.encoder = nn.ModuleList(
            [
                _CNA(chans[i], chans[i + 1], enc_kernels[i], enc_strides[i])
                for i in range(len(enc_kernels))
            ]
        )
        self.enc_gconv = nn.Conv1d(dim, dim, 1)
        n_full = sizes[-1]

        self.sub_encoder = _SubBandEncoder(band_splits, band_tokens, dim)
        n_sub = sum(band_tokens)
        self.n_full = n_full

        self.pre_tcn = _PreTCN(dim, n_full, n_sub, n_pre_tcn, tcn_kernel, dropout)
        self.pos_emb = nn.Parameter(torch.zeros(n_full + n_sub, dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.transformer = nn.ModuleList(
            [
                _FreqTransformerBlock(
                    dim, n_heads, n_full, ffn_hidden, ffn_kernel, dropout, attention
                )
                for _ in range(n_transformer)
            ]
        )
        self.post_tcn = _PostTCN(dim, n_post_tcn, tcn_kernel, dropout)

        # Full-band decoder, mirrored stage by stage with learnable skips.
        self.skip_proj = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in reversed(range(len(enc_kernels))):
            k, s = enc_kernels[i], enc_strides[i]
            c_skip = chans[i + 1]
            self.skip_proj.append(nn.Sequential(nn.Conv1d(c_skip, c_skip, 1), nn.ELU()))
            produced = (sizes[i + 1] - 1) * s - 2 * ((k - s) // 2) + k
            self.decoder.append(_CTNA(2 * c_skip, chans[i], k, s, sizes[i] - produced))
        self.dec_gconv = nn.Conv1d(chans[0], 2, 1)

        self.sub_decoder = _SubBandDecoder(band_splits, band_tokens, self.sub_encoder.kernels, dim)
        self.mask_gen = _MaskGen(mask_hidden, mask_kernel)

    def _stft(self, wav: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            wav,
            self.n_fft,
            self.hop_length,
            window=self.window,
            center=True,
            return_complex=True,
        )

    def _istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            spec, self.n_fft, self.hop_length, window=self.window, center=True, length=length
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: mono mixture ``(B, 1, T)`` or ``(B, T)`` -> enhanced ``(B, 1, T)``."""
        if x.ndim == 3:
            if x.shape[1] != 1:
                raise ValueError(f"DroFiT is single-microphone, got {x.shape[1]} channels")
            x = x[:, 0]
        n_samples = x.shape[-1]
        # STFT / complex mask / iSTFT stay fp32 under autocast (no ComplexHalf kernels).
        with torch.autocast(device_type=x.device.type, enabled=False):
            spec = self._stft(x.float())  # (B, F, T) complex
        b, n_freq, t = spec.shape

        ri = torch.stack([spec.real, spec.imag], dim=1)  # (B, 2, F, T)
        h = ri.permute(0, 3, 1, 2).reshape(b * t, 2, n_freq)  # per-frame (BT, 2, F)
        skips = []
        for cna in self.encoder:
            h = cna(h)
            skips.append(h)
        h = self.enc_gconv(h)  # (BT, C, F_F)
        dim = h.shape[1]

        mag = spec.abs().transpose(1, 2).reshape(b * t, 1, n_freq)
        sub_enc = self.sub_encoder(mag)  # per group (BT, C, tokens)
        sub = torch.cat(sub_enc, dim=-1)  # (BT, C, F_S)

        full_tok = h.transpose(1, 2).reshape(b, t, self.n_full, dim)
        sub_tok = sub.transpose(1, 2).reshape(b, t, -1, dim)
        z = self.pre_tcn(full_tok, sub_tok) + self.pos_emb  # (B, T, F_F + F_S, C)
        n_tok = z.shape[2]
        z = z.reshape(b * t, n_tok, dim)
        for block in self.transformer:
            z = block(z)
        z_full = z[:, : self.n_full]  # (BT, F_F, C)
        z_sub = z[:, self.n_full :]  # (BT, F_S, C)

        gain = self.post_tcn(z_full.reshape(b, t, self.n_full, dim))  # (B, T)

        d = z_full.transpose(1, 2)  # (BT, C, F_F)
        for proj, ctna, skip in zip(self.skip_proj, self.decoder, reversed(skips), strict=True):
            d = ctna(torch.cat([d, proj(skip)], dim=1))
        d = self.dec_gconv(d)  # (BT, 2, F)
        sub_mag = self.sub_decoder(z_sub.transpose(1, 2), sub_enc)  # (BT, F)

        feats = torch.cat([sub_mag.unsqueeze(1), d], dim=1)  # (BT, 3, F): |S|, Re, Im
        feats = feats.reshape(b, t, 3, n_freq).permute(0, 2, 3, 1)  # (B, 3, F, T)
        mask = self.mask_gen(feats) * gain[:, None, None, :]  # (B, 2, F, T)

        with torch.autocast(device_type=x.device.type, enabled=False):
            mask = mask.float()
            est = torch.complex(mask[:, 0], mask[:, 1]) * spec
            out = self._istft(est, n_samples)
        return out.unsqueeze(1)


def build_drofit(**params) -> nn.Module:
    """Factory for the Hydra ``_target_`` path — plain kwargs, see conf/model/drofit.yaml."""
    return DroFiT(**params)
