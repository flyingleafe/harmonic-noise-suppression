"""MP-SENet generator — port of the CURRENT official release (2.26 M parameters).

Paper: Y.-X. Lu, Y. Ai, Z.-H. Ling, "MP-SENet: A Speech Enhancement Model with
Parallel Denoising of Magnitude and Phase Spectra", Interspeech 2023
(arXiv:2305.13686); long version "Explicit Estimation of Magnitude and Phase
Spectra in Parallel for High-Quality Speech Enhancement", Neural Networks 189
(2025) 107562 (arXiv:2308.08926). The current release implements the long
version (Table I: 2.26 M parameters).

Upstream: https://github.com/yxlu-0102/MP-SENet, commit
``89932cfe90d1dacb8e170e4a331d762462c21792`` (2025-05-19), MIT licence,
Copyright (c) 2023 Yexin Lu. Adapted with attribution from ``models/model.py``,
``models/transformer.py``, ``utils.py`` and ``dataset.py``. The state-dict keys
of :class:`MPNet` are identical to upstream's ``MPNet``, so the released
checkpoints ``best_ckpt/g_best_vb`` / ``g_best_dns`` (``{"generator": sd}``)
load with ``strict=True``. ``models/conformer.py`` is not used by the current
release (it belongs to the 2.05 M Interspeech version) and is not ported.

Pipeline (``forward``, waveform in, waveform out, all inside the model):

    x (B, 1, T) -> per-utterance RMS normalisation (upstream inference.py and
    dataset.py: ``norm = sqrt(T / sum(x**2))``) -> STFT (n_fft 400, hop 100,
    win 400, periodic Hann, center, reflect) -> magnitude ** 0.3, phase ->
    MPNet -> magnitude ** (1 / 0.3) -> iSTFT -> divide by ``norm`` -> (B, 1, T).

Architecture traced to the paper (long version, Sec. III-A):
  * encoder = Conv(1x1)+IN+PReLU, dilated DenseNet (4 layers, dilations 1/2/4/8
    along time, kernel (2, 3)), Conv(1x3, stride 2 on frequency)+IN+PReLU;
    C = 64, F' = 101 -> :class:`DenseEncoder`, :class:`DenseBlock`.
  * N = 4 TF-Transformer blocks, each a "time" then a "frequency" GRU-based
    transformer (MHSA with M = 4 heads, Bi-GRU FFN with hidden 2C, linear;
    LayerNorm + residual; no positional encoding) -> :class:`TSTransformerBlock`,
    :class:`TransformerBlock`, :class:`FFN`.
  * magnitude mask decoder = DenseNet + sub-pixel deconv block (SPConv, r=2)
    + Conv(1x2) + learnable sigmoid with beta = 2 and a per-frequency slope
    -> :class:`MaskDecoder`, :class:`LearnableSigmoid2d`.
  * phase decoder = DenseNet + sub-pixel deconv block + two parallel Conv(1x2)
    producing pseudo real / imaginary parts, then ``atan2`` ->
    :class:`PhaseDecoder`.

Where code and paper disagree the code (which produced the published numbers
and checkpoints) is followed:
  * the FFN activation is ``leaky_relu`` (paper text: ReLU);
  * the paper's Fig. 2 reshapes to ``BF' x T x C`` for the Time-Transformer and
    ``BT x F' x C`` for the Frequency-Transformer, but upstream feeds those
    tensors to ``nn.MultiheadAttention`` and ``nn.GRU`` built with the default
    ``batch_first=False``. Both layers therefore treat dim 0 as the sequence:
    the block named ``time_transformer`` attends and recurs along the
    flattened ``(batch, frequency)`` axis, and ``freq_transformer`` along
    ``(batch, time)``. The README acknowledges this "small bug". With one
    utterance per call (upstream inference) this is a clean
    frequency-then-time model; with B > 1 (upstream training, 2 utterances per
    GPU) the sequence runs across utterance boundaries, so an utterance's output
    depends on its batch neighbours.

``transformer_layout`` selects how the batch axis is handled:
  * ``"per_utterance"`` (default): exactly the computation upstream performs on
    a single utterance (sequence = frequency for ``time_transformer``,
    sequence = time for ``freq_transformer``), applied to each utterance of the
    batch independently. Identical to upstream at B = 1 (the setting of the
    published checkpoints at inference) and batch-independent at any B. This
    is the one deviation from the released code: it removes the cross-utterance
    coupling that upstream has during B > 1 training.
  * ``"upstream"``: upstream's tensor layout verbatim, including the coupling
    of the utterances of a batch. Use it only to reproduce upstream numerics
    at B > 1.
  The README's suggestion of setting ``batch_first=True`` on the attention
  (attention over the named axis, GRU unchanged) is a different model from the
  released one and is not offered.

Other deviations / unstated settings:
  * ``forward`` calls ``istft(..., length=T)``; upstream omits ``length`` and
    returns ``hop * floor(T / hop)`` samples. The samples upstream returns are
    identical; the up-to-99 tail samples are the same overlap-add continued to
    ``T`` (needed for the same-length contract).
  * The RMS normalisation divides by ``sum(x**2)`` clamped below at ``1e-12``
    so an all-zero input does not produce ``inf``; upstream has no clamp. Any
    non-silent input is unaffected.
  * STFT / iSTFT and the magnitude/phase maths run with autocast disabled in
    fp32; upstream trains in fp32 throughout (no AMP in ``train.py``).
  * ``LearnableSigmoid*`` drop upstream's no-op ``self.slope.requiresGrad = True``
    (a misspelt attribute; it has no effect).

Parity with upstream (scratch clone, same weights and inputs, fp32 CPU, torch
2.7.0; ``tests/models/test_mpsenet_official.py`` slow tests repeat it):
  * parameters: 2,263,372, identical to upstream's ``MPNet`` (paper: 2.26 M);
  * random init, 1 s input: B = 1 max |diff| 5.6e-7 (outputs ~0.48, 1.2e-6
    relative); ``"upstream"`` layout at B = 2 vs upstream at B = 2: 1.1e-6;
    ``"per_utterance"`` at B = 2 vs upstream run per utterance: 6.2e-7
    (upstream's own B = 2 coupling moves an utterance by 3.9e-2);
  * released ``g_best_vb`` / ``g_best_dns`` on 2 s of VoiceBank p232_032:
    1.8e-7 / 1.6e-7 (upstream WB-PESQ on that excerpt: noisy 1.11, vb 2.64,
    dns 2.37).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "MPNet",
    "MPSENetOfficial",
    "build_mpsenet_official",
    "mag_pha_istft",
    "mag_pha_stft",
    "rms_norm_factor",
]

TRANSFORMER_LAYOUTS = ("per_utterance", "upstream")


# --------------------------------------------------------------------------- #
# Spectral front end (upstream dataset.py)                                    #
# --------------------------------------------------------------------------- #


def mag_pha_stft(
    y: torch.Tensor,
    n_fft: int,
    hop_size: int,
    win_size: int,
    compress_factor: float = 1.0,
    center: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(B, L)`` waveform -> compressed magnitude, phase ``(B, F, T)``, complex ``(B, F, T, 2)``."""
    hann_window = torch.hann_window(win_size, device=y.device, dtype=y.dtype)
    stft_spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        return_complex=True,
    )
    stft_spec = torch.view_as_real(stft_spec)
    mag = torch.sqrt(stft_spec.pow(2).sum(-1) + (1e-9))
    pha = torch.atan2(stft_spec[:, :, :, 1] + (1e-10), stft_spec[:, :, :, 0] + (1e-5))
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
    return mag, pha, com


def mag_pha_istft(
    mag: torch.Tensor,
    pha: torch.Tensor,
    n_fft: int,
    hop_size: int,
    win_size: int,
    compress_factor: float = 1.0,
    center: bool = True,
    length: int | None = None,
) -> torch.Tensor:
    """Compressed magnitude + phase ``(B, F, T)`` -> waveform ``(B, L)``.

    ``length=None`` reproduces upstream (``hop * (T - 1)`` samples).
    """
    mag = torch.pow(mag, (1.0 / compress_factor))
    com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
    hann_window = torch.hann_window(win_size, device=com.device, dtype=mag.dtype)
    return torch.istft(
        com,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        length=length,
    )


def rms_norm_factor(x: torch.Tensor) -> torch.Tensor:
    """Upstream ``sqrt(L / sum(x**2))`` per utterance; ``x`` is ``(B, L)``, returns ``(B, 1)``."""
    energy = x.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.sqrt(x.shape[-1] / energy)


# --------------------------------------------------------------------------- #
# Network (upstream models/model.py, models/transformer.py, utils.py)         #
# --------------------------------------------------------------------------- #


class LearnableSigmoid2d(nn.Module):
    """``beta * sigmoid(slope * x)`` with a per-frequency slope ``(F, 1)``."""

    def __init__(self, in_features: int, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.beta * torch.sigmoid(self.slope * x)


class SPConvTranspose2d(nn.Module):
    """Sub-pixel convolution upsampling the last (frequency) axis by ``r``."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size, r: int = 1) -> None:
        super().__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, h, w = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, h, w))
        out = out.permute(0, 2, 3, 4, 1)
        return out.contiguous().view((batch_size, nchannels // self.r, h, -1))


class DenseBlock(nn.Module):
    """Dilated DenseNet: ``depth`` convs, time dilation ``2**i``, causal time padding."""

    def __init__(self, dense_channel: int, kernel_size=(2, 3), depth: int = 4) -> None:
        super().__init__()
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dilation = 2**i
            self.dense_block.append(
                nn.Sequential(
                    nn.ConstantPad2d((1, 1, dilation, 0), value=0.0),
                    nn.Conv2d(
                        dense_channel * (i + 1),
                        dense_channel,
                        kernel_size,
                        dilation=(dilation, 1),
                    ),
                    nn.InstanceNorm2d(dense_channel, affine=True),
                    nn.PReLU(dense_channel),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, dense_channel: int, in_channel: int) -> None:
        super().__init__()
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, dense_channel, (1, 1)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel),
        )
        self.dense_block = DenseBlock(dense_channel, depth=4)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(dense_channel, dense_channel, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_conv_1(x)  # [B, C, T, F]
        x = self.dense_block(x)  # [B, C, T, F]
        return self.dense_conv_2(x)  # [B, C, T, F // 2 + 1]


class MaskDecoder(nn.Module):
    def __init__(self, dense_channel: int, n_fft: int, beta: float, out_channel: int = 1) -> None:
        super().__init__()
        self.dense_block = DenseBlock(dense_channel, depth=4)
        self.mask_conv = nn.Sequential(
            SPConvTranspose2d(dense_channel, dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel),
            nn.Conv2d(dense_channel, out_channel, (1, 2)),
        )
        self.lsigmoid = LearnableSigmoid2d(n_fft // 2 + 1, beta=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)  # [B, F, T]
        return self.lsigmoid(x)


class PhaseDecoder(nn.Module):
    def __init__(self, dense_channel: int, out_channel: int = 1) -> None:
        super().__init__()
        self.dense_block = DenseBlock(dense_channel, depth=4)
        self.phase_conv = nn.Sequential(
            SPConvTranspose2d(dense_channel, dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel),
        )
        self.phase_conv_r = nn.Conv2d(dense_channel, out_channel, (1, 2))
        self.phase_conv_i = nn.Conv2d(dense_channel, out_channel, (1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x = torch.atan2(self.phase_conv_i(x), self.phase_conv_r(x))
        return x.permute(0, 3, 2, 1).squeeze(-1)  # [B, F, T]


class FFN(nn.Module):
    """Bi-GRU (hidden ``2 * d_model``) -> leaky ReLU -> dropout -> linear. Sequence-first input."""

    def __init__(self, d_model: int, bidirectional: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.gru = nn.GRU(d_model, d_model * 2, 1, bidirectional=bidirectional)
        self.linear = nn.Linear(d_model * 2 * (2 if bidirectional else 1), d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        return self.linear(x)


class TransformerBlock(nn.Module):
    """GRU-based transformer layer; input ``(seq, batch, d_model)`` (``batch_first=False``)."""

    def __init__(
        self, d_model: int, n_heads: int, bidirectional: bool = True, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, bidirectional=bidirectional)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = self.norm1(x)
        xt, _ = self.attention(xt, xt, xt, need_weights=False)
        x = x + self.dropout1(xt)
        xt = self.norm2(x)
        xt = self.ffn(xt)
        x = x + self.dropout2(xt)
        return self.norm3(x)


class TSTransformerBlock(nn.Module):
    """One TF-Transformer block; see the module docstring for ``layout``."""

    def __init__(self, dense_channel: int, n_heads: int, layout: str) -> None:
        super().__init__()
        if layout not in TRANSFORMER_LAYOUTS:
            raise ValueError(
                f"transformer_layout must be one of {TRANSFORMER_LAYOUTS}, got {layout!r}"
            )
        self.layout = layout
        self.time_transformer = TransformerBlock(d_model=dense_channel, n_heads=n_heads)
        self.freq_transformer = TransformerBlock(d_model=dense_channel, n_heads=n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, f = x.size()
        if self.layout == "upstream":
            x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
            x = self.time_transformer(x) + x
            x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
            x = self.freq_transformer(x) + x
            return x.view(b, t, f, c).permute(0, 3, 1, 2)
        # per_utterance: sequence = frequency (batch = B*T), then sequence = time (batch = B*F)
        x = x.permute(3, 0, 2, 1).reshape(f, b * t, c)
        x = self.time_transformer(x) + x
        x = x.view(f, b, t, c).permute(2, 1, 0, 3).reshape(t, b * f, c)
        x = self.freq_transformer(x) + x
        return x.view(t, b, f, c).permute(1, 3, 0, 2)


class MPNet(nn.Module):
    """Upstream ``MPNet``: compressed magnitude + phase ``(B, F, T)`` in, enhanced spectra out."""

    def __init__(
        self,
        dense_channel: int = 64,
        n_fft: int = 400,
        beta: float = 2.0,
        num_tsblocks: int = 4,
        n_heads: int = 4,
        transformer_layout: str = "per_utterance",
    ) -> None:
        super().__init__()
        self.num_tscblocks = num_tsblocks
        self.dense_encoder = DenseEncoder(dense_channel, in_channel=2)
        self.TSTransformer = nn.ModuleList(
            [
                TSTransformerBlock(dense_channel, n_heads, transformer_layout)
                for _ in range(num_tsblocks)
            ]
        )
        self.mask_decoder = MaskDecoder(dense_channel, n_fft, beta, out_channel=1)
        self.phase_decoder = PhaseDecoder(dense_channel, out_channel=1)

    def forward(
        self, noisy_amp: torch.Tensor, noisy_pha: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.stack((noisy_amp, noisy_pha), dim=-1).permute(0, 3, 2, 1)  # [B, 2, T, F]
        x = self.dense_encoder(x)
        for block in self.TSTransformer:
            x = block(x)
        denoised_amp = noisy_amp * self.mask_decoder(x)
        denoised_pha = self.phase_decoder(x)
        denoised_com = torch.stack(
            (denoised_amp * torch.cos(denoised_pha), denoised_amp * torch.sin(denoised_pha)),
            dim=-1,
        )
        return denoised_amp, denoised_pha, denoised_com


# --------------------------------------------------------------------------- #
# Waveform wrapper (project SE contract)                                      #
# --------------------------------------------------------------------------- #


class MPSENetOfficial(nn.Module):
    """Waveform-in / waveform-out MP-SENet generator; ``forward((B, 1, T)) -> (B, 1, T)``.

    The upstream network is ``self.generator`` (state-dict prefix ``generator.``);
    :meth:`spectra` exposes the spectral-domain pass the losses need.
    """

    def __init__(
        self,
        n_fft: int = 400,
        hop_size: int = 100,
        win_size: int = 400,
        compress_factor: float = 0.3,
        dense_channel: int = 64,
        num_tsconformers: int = 4,
        n_heads: int = 4,
        beta: float = 2.0,
        sampling_rate: int = 16000,
        transformer_layout: str = "per_utterance",
        checkpoint: str | None = None,
    ) -> None:
        super().__init__()
        if sampling_rate != 16000:
            raise ValueError("MP-SENet official release is 16 kHz only")
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.compress_factor = compress_factor
        self.sampling_rate = sampling_rate
        self.generator = MPNet(
            dense_channel=dense_channel,
            n_fft=n_fft,
            beta=beta,
            num_tsblocks=num_tsconformers,
            n_heads=n_heads,
            transformer_layout=transformer_layout,
        )
        if checkpoint is not None:
            self.load_upstream_checkpoint(checkpoint)

    def load_upstream_checkpoint(self, checkpoint: str) -> None:
        """Load an upstream generator file (``{"generator": state_dict}``), strict."""
        from utils.checkpoints import resolve_checkpoint_uri

        path = resolve_checkpoint_uri(checkpoint)
        pkg = torch.load(path, map_location="cpu", weights_only=True)
        self.generator.load_state_dict(pkg["generator"], strict=True)

    def stft(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return mag_pha_stft(wav, self.n_fft, self.hop_size, self.win_size, self.compress_factor)

    def istft(self, mag: torch.Tensor, pha: torch.Tensor, length: int | None) -> torch.Tensor:
        return mag_pha_istft(
            mag, pha, self.n_fft, self.hop_size, self.win_size, self.compress_factor, length=length
        )

    def spectra(
        self, noisy_wav: torch.Tensor, length: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Already-normalised ``(B, L)`` noisy waveform -> ``(mag_g, pha_g, com_g, audio_g)``."""
        with torch.autocast(device_type=noisy_wav.device.type, enabled=False):
            noisy_mag, noisy_pha, _ = self.stft(noisy_wav.float())
            mag_g, pha_g, com_g = self.generator(noisy_mag, noisy_pha)
            audio_g = self.istft(mag_g, pha_g, length)
        return mag_g, pha_g, com_g, audio_g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 1, T)`` or ``(B, T)`` noisy 16 kHz waveform -> ``(B, 1, T)`` enhanced."""
        if x.dim() == 3:
            if x.shape[1] != 1:
                raise ValueError(f"MP-SENet is single-channel, got {x.shape[1]} channels")
            x = x.squeeze(1)
        n = x.shape[-1]
        x = x.float()
        norm = rms_norm_factor(x)
        _, _, _, audio_g = self.spectra(x * norm, length=n)
        return (audio_g / norm).unsqueeze(1)


def build_mpsenet_official(**params: Any) -> nn.Module:
    """Factory for the Hydra ``_target_`` path: ``build_mpsenet_official(**params)``."""
    return MPSENetOfficial(**params)
