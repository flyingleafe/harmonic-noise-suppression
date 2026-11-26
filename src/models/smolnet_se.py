"""SMoLnet and SMoLnet-T: frequency-dilated CNN speech enhancers for drone audition.

Two related single-microphone models by the same group, both doing **direct
complex spectral mapping** (noisy real/imag STFT in, clean real/imag STFT out,
no mask) on a long 2048-sample STFT window, so that the narrow-band rotor
harmonics are resolved.

* **SMoLnet** — Z.-W. Tan, A. H. T. Nguyen, A. W. H. Khong, "An Efficient
  Dilated Convolutional Neural Network for UAV Noise Reduction at Low Input
  SNR", APSIPA ASC 2019, pp. 1885-1892, doi:10.1109/APSIPAASC47483.2019.9023324.
* **SMoLnet-T** — Z.-W. Tan, A. W. H. Khong, "SMoLnet-T: An Efficient
  Complex-spectral Mapping Speech Enhancement Approach with Frame-wise CNN and
  Spectral Combination Transformer for Drone Audition", APSIPA ASC 2024,
  doi:10.1109/APSIPAASC63619.2025.10848922.

No upstream code exists for either paper; this is a re-implementation from the
paper text only. There is therefore no numerical parity reference: the parity
gate is the parameter count (below) plus the paper-to-code trace.

Shared wrapper (:class:`_ComplexSpectralMapper`)
------------------------------------------------
``forward(x: (B, 1, N)) -> (B, 1, N)`` at 16 kHz:
per-utterance RMS normalisation -> STFT (``n_fft = 2048``, hop 1024 = 50 %
overlap, ``center=True`` with zero padding) -> features ``(B, 2, T, F)``
holding (Re, Im) as two channels, F = 1025 -> :meth:`map` -> (Re, Im) of the
estimate -> iSTFT (same window, ``length=N``) -> undo the RMS scale.

SMoLnet (2019) — paper statement -> code
----------------------------------------
* Eq. (8) TCS input/target: real and imaginary parts stacked "along the filter
  dimension" -> the 2 input channels of ``body[0]``; Eq. (9) reconstruction
  X = X_re + j X_im -> ``torch.complex(out[:, 0], out[:, 1])`` in ``forward``.
* Table I layers 1-10: kernel (3, 1) along frequency, 64 filters, dilation
  (2^(l-1), 1) for l = 1..10, output (1025, 9, 64) i.e. "same" zero padding ->
  ``_FreqConvBlock(kernel_t=1, dilation_f=2**l)``.
* Table I layers 11-13: kernel (3, 3), 64 filters, no dilation, same size ->
  ``_FreqConvBlock(kernel_t=3, dilation_f=1)``.
* Eq. (3) "followed by batch normalization and ReLU" -> conv -> BatchNorm2d ->
  ReLU in every one of layers 1-13 (``norm="batch"``).
* Table II / Sec. II-B2, layer 14: one 1x1 convolution, no dilation, 2 filters
  (TCS), linear activation -> ``SMoLnet.out``.
* Sec. II-B: 16 kHz, 0.64 s segments = 10 240 samples = 9 frames of a
  2048-sample window, F = 1025; Sec. II-C: sine window, 50 % overlap ->
  ``window="sine"``, ``n_fft=2048``, ``hop_length=1024``, ``context_frames=9``.
* Table V: 224 k parameters. This port: **224 194** (convolutions 222 530 +
  BatchNorm affine 13 x 2 x 64 = 1 664).

The 2024 paper's Table I lists SMoLnet at **249 k**. That number is SMoLnet
with *layer normalisation over the 1025 frequency bins* in place of batch
normalisation (the same normalisation SMoLnet-T's Eq. (24) uses):
222 530 + 13 x 2 x 1025 = **249 180**. ``norm="layer"`` builds that variant.

SMoLnet-T (2024) — paper statement -> code
------------------------------------------
* Eq. (23) Z^[0]_t = [y_re,t, y_im,t]^T -> the 2 input channels of ``cnn[0]``.
* Eqs. (22), (24) frame-wise CNN f_CNN = f_relu o f_ln o f_di-conv o f_pad,
  kernel k = 3, dilation D(l) = 2^(l-1) along frequency, C = 64, zero padding
  -> ``_FreqConvBlock(kernel_t=1, dilation_f=2**l, norm="layer")``; the time
  kernel is 1, so each frame is processed on its own ("frame-wise").
  f_ln is ``nn.LayerNorm(F)`` over the frequency axis of each (frame, channel),
  the reading that reproduces the 249 k SMoLnet count of the same table.
  Number of layers: "the first ten layers of the SMoLnet" -> ``n_cnn_layers=10``
  (see *unstated settings* for the L = log2(F) + 1 formula).
* Eq. (25) M consecutive transformers -> ``SMoLnetT.transformers``
  (``n_transformers`` = M in {1, 2, 3}).
* D_f in R^{T x C}: one sequence over TIME per frequency bin f -> the
  ``(B, C, T, F) -> (B * F, T, C)`` reshape in :meth:`SMoLnetT.map`.
* Eq. (26) f_SCT = f_ln2 o f_ffn2 o f_ffn1 o f_ln1 o f_mha o f_pe: positional
  encoding is added at the input of *every* transformer ->
  :meth:`SpectralCombinationTransformer.forward`.
* Eqs. (27)-(28) D'_f = D_f + E_{T x C}, fixed sinusoidal encoding ->
  :func:`sinusoidal_positional_encoding`.
* Eqs. (29)-(33) multi-head self-attention over time -> ``nn.MultiheadAttention``
  (H heads, Q/K/V/O projections, softmax over the T axis).
* Eq. (34) Ddot = ln1(A + D')       -> ``self.ln1(a + d)`` (residual uses D').
* Eq. (35) ffn1 = relu(Ddot W1 + b1), W1 in R^{C x 4C}  -> ``relu(self.ffn1(.))``.
* Eq. (36) ffn2 = relu(D'' W2 + b2), W2 in R^{4C x C}   -> ``relu(self.ffn2(.))``
  (the paper applies ReLU after the second FFN layer too; kept).
* Eq. (37) D''' = ln2(Ddot + ffn2)  -> ``self.ln2(d1 + d3)`` (post-norm).
* Eqs. (38)-(39) [x_re, x_im] = D^[M]_f W_last + b_last, W_last in R^{C x 2}
  -> ``SMoLnetT.out = nn.Linear(C, 2)``; X = x_re + j x_im.
* Sec. IV: C = 64, Hamming window of length 2048 with 50 % overlap, F = 1025
  -> ``window="hamming"``, ``n_fft=2048``, ``hop_length=1024``, ``channels=64``.
* Table I sizes: 192 k / 254 k / 317 k for M = 1 / 2 / 3. This port:
  **182 230 / 232 214 / 282 198** (-5.1 % / -8.6 % / -11.0 %); frame-wise CNN
  111 616 conv + 10 x 2 050 LayerNorm(F) = 132 116, output layer 130, and per
  transformer 49 984 (MHA 4C^2 + 4C = 16 640, FFN 8C^2 + 5C = 33 088, two
  LayerNorm(C) 256). See *parameter-count gap* below.

Parameter-count gap (SMoLnet-T)
-------------------------------
The published sizes grow by 62 k and 63 k per extra transformer, so a model
of the form base + M x block has block in about [62 k, 63 k] and base in about
[128.5 k, 130.5 k]. The transformer the equations describe (C = 64, FFN 4C,
two LayerNorms) has 49 984 parameters (49 728 if the Q/K/V/O projections of
Eqs. (29)-(33) were bias-free, as literally written); no head count changes
that. Interpretations tried:
1. Equations as written, biased attention projections (**implemented; the
   default of conf/model/smolnet_t.yaml**): 182 230 / 232 214 / 282 198
   (-5.1 / -8.6 / -11.0 %).
2. Equations as written, bias-free Q/K/V/O: 181 974 / 231 702 / 281 430
   (-5.2 / -8.8 / -11.2 %).
3. CNN of L = log2(F) + 1 = 11 layers instead of 10 (dilation up to 1024,
   ``n_cnn_layers=11``): 196 632 / 246 616 / 296 600 (+2.4 / -2.9 / -6.4 %).
4. Three additional C x C (+bias) projections per transformer, e.g. separate
   Q/K/V ``nn.Linear`` layers feeding an ``nn.MultiheadAttention`` that has its
   own input projection (block 62 464): 194 710 / 257 174 / 319 638
   (+1.4 / +1.2 / +0.8 %). Nothing in the paper describes it, and two stacked
   linear maps add no expressiveness; it is NOT implemented.
5. FFN width 5.5C = 352 instead of 4C (block 62 368): 194 614 / 256 982 /
   319 350 (+1.4 / +1.2 / +0.7 %); equally unsupported, NOT implemented.
All counts above were measured on built models. Only 4 and 5 are within 2 %
of all three numbers, and neither also explains the base: the published sizes
want a base (M = 0) of about 129.3 k, the implemented CNN + output layer has
132 246. The published numbers most likely come from code that differs from
the text in more than one place.

Long inputs and receptive field
-------------------------------
* **SMoLnet** was trained on and evaluated with 9-frame (0.64 s) segments: the
  2019 paper "split[s] the spectrogram to segments that have the same duration
  of 0.64 s" for evaluation, and the 2024 paper calls it "the nine frames
  context in SMoLnet". ``context_frames=9`` (default) reproduces that: the
  utterance STFT is cut into consecutive non-overlapping 9-frame blocks (the
  last one zero-padded), each block runs through the network independently
  (zero time padding at block edges, as in training), and the blocks are
  concatenated before one iSTFT over the whole utterance (overlap-add across
  block borders is the normal STFT overlap). The time receptive field is
  +-3 frames (three 3x3 layers) truncated at block edges, i.e. at most 7 frames
  = 6 hops + 1 window = 8 192 samples = 0.51 s; no information crosses a block
  border. ``context_frames=None`` runs the network fully convolutionally over
  all frames instead (the network is fully convolutional, so this is the same
  function wherever a frame is at least 3 frames from a block edge): every
  frame then gets the full +-3-frame context. On the waveform, each output
  sample is the overlap-add of two adjacent frames, so it depends on at most
  8 input frames = 7 hops + 1 window = 9 216 samples = 0.58 s (fewer next to
  a block border). The block grid is anchored at frame 0 (frames
  [9k, 9k + 9)), so with blocks the output is not invariant to shifting the
  input by a hop. A long input costs memory linear in its length (one batch
  entry per block). Along frequency the ten dilated layers give Eq. (5)
  R = 2^10 x 2 - 3 + 2 = 2 047 bins (> 1 025), and layers 11-13 widen it by
  one bin per side each.
* **SMoLnet-T** attends over all T frames of the input (full-utterance
  self-attention, any length; the paper trains with T = 123 frames, which at
  hop 1024 is about 7.9 s, the length of a WSJ0 utterance; its Table I caption
  says "64 K samples" per example, which matches T = 63, not 123 — the paper
  does not resolve this). Receptive field: the whole utterance in time,
  2 047 bins in frequency. Lengths far beyond the trained T extrapolate the
  sinusoidal encoding, and the attention cost grows as F x T^2 (computed as
  B x 1025 sequences of length T).
* Both models were run on 4 000, 16 001 and 160 000 samples (0.25 s to 10 s,
  odd length included): output length equals input length, all finite; the
  STFT/iSTFT wrapper alone (identity ``map``) reconstructs the input to
  3e-6.

Unstated settings and the values chosen
---------------------------------------
* Input/output scaling (both papers): not stated. ``normalize=True`` divides
  the mixture by its RMS and multiplies the estimate back, which makes the
  model exactly scale-equivariant. Without it a direct mapper's output level
  is tied to the absolute training level, and SMoLnet-T's LayerNorm(F) after
  layer 1 removes the input level almost completely.
* STFT framing: SMoLnet's 9 frames from 10 240 samples imply frames fully
  inside the segment (``center=False``). Here the whole utterance is framed
  with ``center=True`` and zero padding so every sample is covered by two
  frames and the iSTFT is exact; the window envelope is normalised by
  ``torch.istft``. Sine window: w[n] = sin(pi (n + 1/2) / N) (MLT window,
  sine^2 is COLA at 50 %). Hamming: periodic (``torch.hamming_window``).
* SMoLnet-T number of layers: the text gives L = log2(F) + 1, which is 11 for
  F = 1025 (and for F = 1024), but also "the first ten layers of the SMoLnet";
  ten is used (it is what SMoLnet's Table I and Eq. (5) need to cover 1025
  bins). Eq. (24)'s padded width F + D(l) + 1 does not keep F bins for D > 1;
  read as F + 2 D(l) ("same" padding), which the LayerNorm(F) requires anyway.
* SMoLnet-T attention heads H: not stated; 8, as in SepFormer [21] whose
  transformer (fixed sinusoidal encoding, FFN 4x) the paper builds on. Head
  count does not change the parameter count. Attention scale: Eq. (30) writes
  sqrt(C) but the same paper writes sqrt(D) for DPTNet's standard
  ``nn.MultiheadAttention`` (Eq. 8), so the standard sqrt(C / H) is used.
  Biases in Q/K/V/O: not written in Eqs. (29)-(33) (nor in Eq. 7 for DPTNet,
  whose code has them); kept, since the reported sizes exceed even the biased
  count. Dropout: not mentioned; none.
* Positional encoding Eq. (28) writes sin/cos(t / 10000^(2c/C)) by channel
  parity with t from 1; this is the Vaswani et al. [19] encoding it cites, so
  [19]'s convention is used: positions 0..T-1, channel 2i -> sin, channel
  2i + 1 -> cos, both with frequency 10000^(-2i/C).
* SMoLnet-T sampling rate and loss: not stated; SMoLnet's 16 kHz and MSE on
  the (Re, Im) spectrum are assumed.
* Initialisation, BatchNorm momentum/eps, LayerNorm eps: not stated; PyTorch
  defaults.
* ``conf/model/smolnet.yaml`` follows the 2019 paper (sine window,
  BatchNorm). The 2024 paper re-trains SMoLnet as a baseline with "a Hamming
  window of length 2048 with 50% overlap" and, judging by its 249 k size,
  LayerNorm(F); that baseline is ``window="hamming", norm="layer"``.

Deviations
----------
* Only SMoLnet's best target, TCS (complex spectral mapping), is implemented;
  the TMS (magnitude, softplus) and cIRM (mask) heads of Table II are not.
* The 2019 model was trained in double precision "due to the low SNR"; this
  module is precision-agnostic and the project trains in fp32.
* The training loss (MSE between estimated and clean (Re, Im) spectra) is not
  part of the model; the project's loss configuration applies.

Training recipes as published (facts only): SMoLnet — MSE on the complex
spectrum, Adam, batch 16, learning rate tried in {1e-3, 5e-4, 1e-4} and halved
after 3 epochs of loss plateau, best model by validation ESTOI, 0.64 s
segments, training SNR -10 to -20 dB, float64. SMoLnet-T — Adam, learning rate
1e-3, batch 2, best validation model over 100 epochs, T = 123 frames, training
SNR {-30, -27, -24, -21, -18} dB; loss not stated.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "SMoLnet",
    "SMoLnetT",
    "SpectralCombinationTransformer",
    "build_smolnet",
    "build_smolnet_t",
    "sinusoidal_positional_encoding",
]

_RMS_EPS = 1e-8


def _analysis_window(kind: str, n_fft: int) -> torch.Tensor:
    if kind == "sine":
        n = torch.arange(n_fft, dtype=torch.float64)
        return torch.sin(math.pi * (n + 0.5) / n_fft).float()
    if kind == "hamming":
        return torch.hamming_window(n_fft, periodic=True)
    raise ValueError(f"window must be 'sine' or 'hamming', got {kind!r}")


def sinusoidal_positional_encoding(
    length: int, channels: int, device: torch.device | None = None, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Fixed sinusoidal encoding E in R^{length x channels} (Vaswani et al. 2017)."""
    pos = torch.arange(length, dtype=torch.float64, device=device).unsqueeze(1)
    inv_freq = torch.exp(
        torch.arange(0, channels, 2, dtype=torch.float64, device=device)
        * (-math.log(10000.0) / channels)
    )
    pe = torch.zeros(length, channels, dtype=torch.float64, device=device)
    pe[:, 0::2] = torch.sin(pos * inv_freq)
    pe[:, 1::2] = torch.cos(pos * inv_freq[: channels // 2])
    return pe.to(dtype or torch.get_default_dtype())


class _FreqConvBlock(nn.Module):
    """Zero-padded conv -> norm -> ReLU on ``(B, C, T, F)``.

    Kernel ``(kernel_t, 3)``: 3 taps along frequency with dilation
    ``dilation_f``, ``kernel_t`` taps along time (1 = frame-wise).
    ``norm="batch"``: BatchNorm2d over channels; ``norm="layer"``: LayerNorm
    over the ``n_freqs`` frequency bins of each (frame, channel).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_t: int,
        dilation_f: int,
        norm: str,
        n_freqs: int,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_t, 3),
            dilation=(1, dilation_f),
            padding=(kernel_t // 2, dilation_f),
        )
        if norm == "batch":
            self.norm: nn.Module = nn.BatchNorm2d(out_channels)
        elif norm == "layer":
            self.norm = nn.LayerNorm(n_freqs)
        else:
            raise ValueError(f"norm must be 'batch' or 'layer', got {norm!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.norm(self.conv(x)))


class _ComplexSpectralMapper(nn.Module):
    """Waveform -> (Re, Im) STFT features -> :meth:`map` -> waveform."""

    def __init__(self, *, n_fft: int, hop_length: int, window: str, normalize: bool):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_freqs = n_fft // 2 + 1
        self.normalize = normalize
        self.window: torch.Tensor
        self.register_buffer("window", _analysis_window(window, n_fft), persistent=False)

    def map(self, feats: torch.Tensor) -> torch.Tensor:
        """``(B, 2, T, F)`` noisy (Re, Im) -> ``(B, 2, T, F)`` estimated (Re, Im)."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            if x.shape[1] != 1:
                raise ValueError(f"expected mono input (B, 1, N), got {tuple(x.shape)}")
            x = x[:, 0]
        elif x.dim() != 2:
            raise ValueError(f"expected (B, 1, N) or (B, N), got {tuple(x.shape)}")
        n = x.shape[-1]
        scale = None
        if self.normalize:
            scale = x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(_RMS_EPS)
            x = x / scale
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )  # (B, F, T)
        feats = torch.view_as_real(spec).permute(0, 3, 2, 1)  # (B, 2, T, F)
        out = self.map(feats)
        est = torch.complex(out[:, 0], out[:, 1]).transpose(1, 2)  # (B, F, T)
        y = torch.istft(
            est,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            length=n,
        )
        if scale is not None:
            y = y * scale
        return y.unsqueeze(1)


class SMoLnet(_ComplexSpectralMapper):
    """SMoLnet-TCS (Tan, Nguyen, Khong 2019): 13 conv layers + 1x1 output layer."""

    def __init__(
        self,
        *,
        n_fft: int = 2048,
        hop_length: int = 1024,
        window: str = "sine",
        channels: int = 64,
        n_dilated_layers: int = 10,
        n_joint_layers: int = 3,
        norm: str = "batch",
        context_frames: int | None = 9,
        normalize: bool = True,
    ):
        super().__init__(n_fft=n_fft, hop_length=hop_length, window=window, normalize=normalize)
        if context_frames is not None and context_frames < 1:
            raise ValueError(f"context_frames must be positive or None, got {context_frames}")
        self.context_frames = context_frames
        layers = [
            _FreqConvBlock(
                2 if i == 0 else channels,
                channels,
                kernel_t=1,
                dilation_f=2**i,
                norm=norm,
                n_freqs=self.n_freqs,
            )
            for i in range(n_dilated_layers)
        ]
        layers += [
            _FreqConvBlock(
                channels, channels, kernel_t=3, dilation_f=1, norm=norm, n_freqs=self.n_freqs
            )
            for _ in range(n_joint_layers)
        ]
        self.body = nn.Sequential(*layers)
        self.out = nn.Conv2d(channels, 2, kernel_size=1)

    def map(self, feats: torch.Tensor) -> torch.Tensor:
        k = self.context_frames
        if k is None:
            return self.out(self.body(feats))
        b, c, t, f = feats.shape
        n_blocks = -(-t // k)
        x = F.pad(feats, (0, 0, 0, n_blocks * k - t))
        x = x.reshape(b, c, n_blocks, k, f).transpose(1, 2).reshape(b * n_blocks, c, k, f)
        y = self.out(self.body(x))
        y = y.reshape(b, n_blocks, 2, k, f).transpose(1, 2).reshape(b, 2, n_blocks * k, f)
        return y[:, :, :t]


class SpectralCombinationTransformer(nn.Module):
    """One spectral combination transformer, Eqs. (26)-(37) of SMoLnet-T.

    Input/output ``(N, T, C)``: N = batch x frequency sequences over time.
    """

    def __init__(self, channels: int, n_heads: int, ffn_mult: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(channels)
        self.ffn1 = nn.Linear(channels, ffn_mult * channels)
        self.ffn2 = nn.Linear(ffn_mult * channels, channels)
        self.ln2 = nn.LayerNorm(channels)

    def forward(self, d: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        d = d + pe  # Eq. (27)
        a = self.mha(d, d, d, need_weights=False)[0]  # Eqs. (29)-(33)
        d1 = self.ln1(a + d)  # Eq. (34)
        d3 = F.relu(self.ffn2(F.relu(self.ffn1(d1))))  # Eqs. (35)-(36)
        return self.ln2(d1 + d3)  # Eq. (37)


class SMoLnetT(_ComplexSpectralMapper):
    """SMoLnet-T (Tan & Khong 2024): frame-wise dilated CNN + M time transformers."""

    def __init__(
        self,
        *,
        n_fft: int = 2048,
        hop_length: int = 1024,
        window: str = "hamming",
        channels: int = 64,
        n_cnn_layers: int = 10,
        n_transformers: int = 3,
        n_heads: int = 8,
        ffn_mult: int = 4,
        normalize: bool = True,
    ):
        super().__init__(n_fft=n_fft, hop_length=hop_length, window=window, normalize=normalize)
        self.cnn = nn.Sequential(
            *[
                _FreqConvBlock(
                    2 if i == 0 else channels,
                    channels,
                    kernel_t=1,
                    dilation_f=2**i,
                    norm="layer",
                    n_freqs=self.n_freqs,
                )
                for i in range(n_cnn_layers)
            ]
        )
        self.transformers = nn.ModuleList(
            SpectralCombinationTransformer(channels, n_heads, ffn_mult)
            for _ in range(n_transformers)
        )
        self.out = nn.Linear(channels, 2)

    def map(self, feats: torch.Tensor) -> torch.Tensor:
        z = self.cnn(feats)  # (B, C, T, F), Eq. (22)
        b, c, t, f = z.shape
        d = z.permute(0, 3, 2, 1).reshape(b * f, t, c)  # D_f in R^{T x C} per bin
        pe = sinusoidal_positional_encoding(t, c, device=d.device, dtype=d.dtype)
        for block in self.transformers:  # Eq. (25)
            d = block(d, pe)
        out = self.out(d)  # Eq. (38), (B * F, T, 2)
        return out.reshape(b, f, t, 2).permute(0, 3, 2, 1)


def build_smolnet(**params) -> SMoLnet:
    """Factory for ``conf/model/smolnet.yaml``."""
    return SMoLnet(**params)


def build_smolnet_t(**params) -> SMoLnetT:
    """Factory for ``conf/model/smolnet_t.yaml``."""
    return SMoLnetT(**params)
