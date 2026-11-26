"""DroneAudioNet — AudioSep's text-queried ResUNet re-framed as a drone-noise estimator.

Paper
    C. Gupta, S. Ramesh, Y. Luo, S. Nanayakkara, "DRONEAUDIONET: Noise Suppression
    for Drone Audition-based Search and Rescue", arXiv:2608.00875 (2026).
Upstream
    https://github.com/DroneAudioNet/DroneAudioNet-Code @ b2b2fc6f8bd0bc74d882a6e66efd834bdfb4a8fa
    — **no licence file**: nothing of the DroneAudioNet-specific code is copied; the
    residual head below is re-implemented from Sec. 3.2 / 4.3 of the paper and the
    upstream module was used only as a numerical reference in a scratch script.
Base model
    AudioSep (Liu et al., "Separate Anything You Describe", TASLP 2024),
    https://github.com/Audio-AGI/AudioSep @ 944583f18b84589dc965de3ad77525c945334252,
    **MIT licence** — the ResUNet30 trunk + FiLM conditioning below are adapted from its
    ``models/resunet.py`` (Copyright (c) Audio-AGI, MIT), restructured but layer-for-layer
    identical so AudioSep / DroneAudioNet checkpoints load with their own key names.

Architecture (paper statement -> code)
    * "fixed drone-based query ... to estimate the drone noise d̂ from the noisy mixture x"
      (Sec. 3.1): the CLAP text embedding of the query is a fixed (512,) buffer loaded from
      ``query_embedding.pt``; it drives the FiLM layers (``film``) exactly as AudioSep's
      query does. CLAP itself is not part of this module (it is frozen in the paper).
    * "ŝ = x − d̂ in the time domain" (Sec. 3.1): :meth:`DroneAudioNet.forward` returns the
      mixture minus the drone estimate.
    * "frequency-domain ResUNet separator ... magnitude mask |M| and phase residual ∠M,
      D̂ = |M| ⊙ |X| e^{j(∠X+∠M)}" (Sec. 3.2): :class:`ResUNet30` — STFT (n_fft 2048,
      hop 320, periodic Hann, centre/reflect, 32 kHz), BatchNorm over frequency, 6
      residual encoder blocks + centre block + 6 decoder blocks (widths 32-32-64-128-256-
      384-384), FiLM additive biases from the query embedding, ``after_conv`` 1x1 head.
    * "|M| ∈ (0,1) via a sigmoid" and the phase from two tanh channels: channels 0..2 of
      ``after_conv`` (sigmoid magnitude, tanh real / tanh imag normalised to a unit phasor).
    * "learnable scalar α (initialised to 1) multiplied with the magnitude mask,
      D̂_M = α(M̂ ⊙ X)" (Sec. 3.2): ``base.alpha``.
    * "additive residual with magnitude β|R| and phase ∠R; |R| sigmoid-constrained,
      β initialised to 1; D̂ = α(M̂ ⊙ X) + β|R|e^{j∠R}" (Sec. 3.2): ``base.beta`` and
      channels 3..5 of ``after_conv`` (sigmoid |R|, tanh real / tanh imag -> unit phasor).
    * "ResUNet outputs six channels ... three additional channels corresponding to residual
      magnitude, real and imaginary components. All the additional weights/biases ... are
      randomly assigned using Xavier initialization" (Sec. 4.3): warm start from the
      3-channel AudioSep head in :meth:`DroneAudioNet.load_upstream_state_dict`.
    * "we fine-tune the publicly available AudioSep model" / "The entire ResUNet model
      belongs under the trainable parameters" (Sec. 4.3, App. A.2): every parameter of
      this module is trainable; the query embedding is a non-trainable buffer.

Parameter count (App. A.2: 238M total, 212M non-trainable, 26.4M trainable)
    This module (full size): 26,448,042 parameters, all trainable — identical to the
    trainable set of upstream's ResUNet30 (measured on the upstream module). Upstream's
    non-trainable 212,151,641 are CLAP (199,564,633 parameters, frozen) + torchlibrosa's
    frozen DFT kernels inside the ResUNet (12,587,008); neither exists here (CLAP is
    replaced by the stored embedding, the DFT kernels by ``torch.stft``/``torch.istft``).
    Upstream total 238,599,683.

Query embedding (``query_embedding.pt``, float32 (512,), unit norm)
    Computed once on 2026-11-25 in a throw-away environment (torch 2.5.1 CPU,
    transformers 4.28.1) with the upstream ``CLAP_Encoder.get_query_embed(modality="text")``
    of DroneAudioNet-Code @ b2b2fc6 (AudioSep's HTSAT-base + RoBERTa CLAP, eval mode) for
    ``QUERY_TEXT`` = "Drone motor and propeller sounds", with the frozen
    ``query_encoder.model.*`` weights of ``droneaudionet.ckpt`` (bit-identical to those in
    ``audiosep_base_4M_steps.ckpt``). sha256 of the 2048 float32 bytes:
    dd2ad0e6ba70c004e1afbf314181dbd0eb181e67f1aafdb51877f37730696129. It is registered as
    a non-persistent buffer: a fixed input, not a weight, so it stays out of
    ``state_dict()``. This package has no CLAP / transformers / torchlibrosa dependency.

Parity (scratch script, upstream ``models/resunet.py::ResUNet30`` vs
:meth:`DroneAudioNet.estimate_drone_native`, same weights, same 32 kHz input, fp32 CPU)
    max |diff| / max |upstream| — ``droneaudionet.ckpt``: 5.6e-6 (1.5 s, B=2, eval),
    7.8e-7 (0.53 s odd length), 2.2e-6 (train-mode BatchNorm); ``audiosep_base_4M_steps.ckpt``
    warm start copied into both: 1.1e-5, 3.7e-6, 2.9e-5 (train-mode BN); random init
    copied through ``state_dict``: 1.6e-6, 2.7e-6, 4.3e-5 (train-mode BN). The same
    AudioSep-base comparison in float64 (upstream DFT kernels built in float64) agrees to
    2.8e-11, so the fp32 gaps are rounding (torchlibrosa's fp32 conv-DFT vs ``torch.stft``).

Settings the paper leaves unstated (values from the upstream code / checkpoints)
    * Query text: "Drone motor and propeller sounds" — the caption of every entry of
      upstream ``datafiles/train_file.json`` and of ``inference.py`` (the paper's Sec. 3.1
      spells "propellor"; the code and the figure use "propeller").
    * |R| and α|M||X| are passed through a ReLU (so a negative α or β cannot flip the
      phase), and the unit phasors use ``x / clamp(|x|, 1e-10)``; the mixture magnitude is
      ``sqrt(clamp(re²+im², 1e-10))`` — as in the upstream code.
    * The new 3 head channels: Xavier-uniform on the (3, 32, 1, 1) weight slice, bias 0
      (upstream ``modify_audiosep_checkpoints/with_residual_mag_and_phase.py``). That
      script writes α = 2.0; we follow the paper (α = β = 1) — the published checkpoint's
      α = 1.0196 is consistent with an initial value of 1, not 2.
    * Subtraction rate: upstream inference resamples the 32 kHz drone estimate to 16 kHz
      and subtracts it from the original 16 kHz mixture; we do the same (``forward``).

Deviations
    * Sample rate: the model is native to 32 kHz. ``forward`` resamples 16 kHz -> 32 kHz
      with ``julius.ResampleFrac`` (differentiable sinc), runs the ResUNet, and resamples
      the drone estimate back (upstream: librosa/soxr at load time, torchaudio in training).
    * STFT/iSTFT use ``torch.stft``/``torch.istft`` (identical maths to torchlibrosa's
      conv-DFT up to fp32 rounding; no frozen DFT-kernel parameters).
    * FiLM biases are computed once from the (1, 512) embedding and broadcast over the
      batch (upstream computes the identical vector per batch element).
    * Upstream inference optionally runs AudioSep's chunked inference (5 s windows, 1 s
      overlap at each side); this port always processes the whole input at once.
    * Unused-but-present tensors are kept for checkpoint compatibility and faithful
      counts: every decoder block's ``bn2`` and FiLM ``decoder_block*->beta2`` layers are
      never used in the forward pass (as in AudioSep) and receive no gradient.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from julius.resample import ResampleFrac
from torch import nn

__all__ = ["DroneAudioNet", "ResUNet30", "build_droneaudionet", "QUERY_TEXT"]

QUERY_TEXT = "Drone motor and propeller sounds"
_QUERY_EMBEDDING_FILE = Path(__file__).with_name("query_embedding.pt")

# AudioSep ResUNet30 widths: pre_conv, encoder_block1..6 (centre block keeps the last).
AUDIOSEP_CHANNELS = (32, 32, 64, 128, 256, 384, 384)
# Encoder pooling factors (time, freq); the decoder mirrors them in reverse.
_DOWNSAMPLE = ((2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (1, 2))
_TIME_DOWNSAMPLE = 2**5
_UNUSED_UPSTREAM_PREFIXES = ("base.stft.", "base.istft.")


def _init_layer(layer: nn.Conv2d | nn.ConvTranspose2d | nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        layer.bias.data.fill_(0.0)


def _init_bn(bn: nn.BatchNorm2d) -> None:
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def _unit_phasor(re: torch.Tensor, im: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mag = (re**2 + im**2) ** 0.5
    mag = torch.clamp(mag, 1e-10, math.inf)
    return re / mag, im / mag


class _ConvBlockRes(nn.Module):
    """BN -> (+FiLM bias) -> LeakyReLU -> 3x3 conv, twice, with a residual shortcut."""

    def __init__(self, in_channels: int, out_channels: int, momentum: float) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), padding=(1, 1), bias=False)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, (1, 1)) if in_channels != out_channels else None
        )
        _init_bn(self.bn1)
        _init_bn(self.bn2)
        _init_layer(self.conv1)
        _init_layer(self.conv2)
        if self.shortcut is not None:
            _init_layer(self.shortcut)

    def film_sizes(self) -> dict[str, int]:
        return {"beta1": self.bn1.num_features, "beta2": self.bn2.num_features}

    def forward(self, x: torch.Tensor, film: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        h = self.conv1(F.leaky_relu(self.bn1(x) + film[f"{prefix}->beta1"], negative_slope=0.01))
        h = self.conv2(F.leaky_relu(self.bn2(h) + film[f"{prefix}->beta2"], negative_slope=0.01))
        return (self.shortcut(x) if self.shortcut is not None else x) + h


class _EncoderBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, downsample: tuple[int, int], momentum: float
    ) -> None:
        super().__init__()
        self.conv_block1 = _ConvBlockRes(in_channels, out_channels, momentum)
        self.downsample = downsample

    def film_sizes(self) -> dict[str, int]:
        return {f"conv_block1->{k}": v for k, v in self.conv_block1.film_sizes().items()}

    def forward(
        self, x: torch.Tensor, film: dict[str, torch.Tensor], prefix: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.conv_block1(x, film, f"{prefix}->conv_block1")
        return F.avg_pool2d(enc, kernel_size=self.downsample), enc


class _DecoderBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, upsample: tuple[int, int], momentum: float
    ) -> None:
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=upsample, stride=upsample, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.conv_block2 = _ConvBlockRes(out_channels * 2, out_channels, momentum)
        # Present in AudioSep (checkpoint key + FiLM "beta2" layer) but never used.
        self.bn2 = nn.BatchNorm2d(in_channels, momentum=momentum)
        _init_bn(self.bn1)
        _init_layer(self.conv1)

    def film_sizes(self) -> dict[str, int]:
        sizes = {"beta1": self.bn1.num_features, "beta2": self.bn2.num_features}
        sizes.update({f"conv_block2->{k}": v for k, v in self.conv_block2.film_sizes().items()})
        return sizes

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, film: dict[str, torch.Tensor], prefix: str
    ) -> torch.Tensor:
        h = self.conv1(F.leaky_relu(self.bn1(x) + film[f"{prefix}->beta1"], negative_slope=0.01))
        h = torch.cat((h, skip), dim=1)
        return self.conv_block2(h, film, f"{prefix}->conv_block2")


class ResUNet30(nn.Module):
    """AudioSep ResUNet30 trunk with the DroneAudioNet 6-channel head (α-scaled mask + residual).

    ``forward(wav (B, T) @ 32 kHz, film) -> drone estimate (B, T)``.
    """

    window: torch.Tensor

    def __init__(
        self,
        channels: Sequence[int] = AUDIOSEP_CHANNELS,
        n_fft: int = 2048,
        hop_length: int = 320,
        momentum: float = 0.01,
    ) -> None:
        super().__init__()
        if len(channels) != 7:
            raise ValueError(
                f"channels needs 7 widths (pre_conv + 6 encoder blocks), got {channels}"
            )
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.k = 6
        self.register_buffer(
            "window", torch.hann_window(self.n_fft, periodic=True), persistent=False
        )

        c = list(channels)
        self.bn0 = nn.BatchNorm2d(self.n_fft // 2 + 1, momentum=momentum)
        self.pre_conv = nn.Conv2d(1, c[0], kernel_size=(1, 1))
        for i, ds in enumerate(_DOWNSAMPLE, start=1):
            self.add_module(f"encoder_block{i}", _EncoderBlock(c[i - 1], c[i], ds, momentum))
        self.conv_block7a = _EncoderBlock(c[6], c[6], (1, 1), momentum)
        # decoder_block j upsamples by the factor of encoder block 7-j and joins its skip.
        for j in range(1, 7):
            self.add_module(
                f"decoder_block{j}",
                _DecoderBlock(c[6] if j == 1 else c[8 - j], c[7 - j], _DOWNSAMPLE[6 - j], momentum),
            )
        self.after_conv = nn.Conv2d(c[1], self.k, kernel_size=(1, 1))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        _init_bn(self.bn0)
        _init_layer(self.pre_conv)
        _init_layer(self.after_conv)

    def film_sizes(self) -> dict[str, int]:
        """FiLM layer name -> width, in AudioSep's pre-order (checkpoint key names)."""
        sizes: dict[str, int] = {}
        for name, block in self.named_children():
            if isinstance(block, (_EncoderBlock, _DecoderBlock)):
                sizes.update({f"{name}->{k}": v for k, v in block.film_sizes().items()})
        return sizes

    def forward(self, wav: torch.Tensor, film: dict[str, torch.Tensor]) -> torch.Tensor:
        n = wav.shape[-1]
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        ).transpose(1, 2)  # (B, T, F)
        re, im = spec.real, spec.imag
        mag = torch.clamp(re**2 + im**2, 1e-10, math.inf) ** 0.5
        cos_in, sin_in = re / mag, im / mag
        mag = mag.unsqueeze(1)  # (B, 1, T, F)

        x = self.bn0(mag.transpose(1, 3)).transpose(1, 3)
        n_frames = x.shape[2]
        pad = math.ceil(n_frames / _TIME_DOWNSAMPLE) * _TIME_DOWNSAMPLE - n_frames
        x = F.pad(x, (0, 0, 0, pad))
        x = x[..., : x.shape[-1] - 1]  # 1025 -> 1024 bins

        x = self.pre_conv(x)
        skips = []
        for i in range(1, 7):
            x, enc = getattr(self, f"encoder_block{i}")(x, film, f"encoder_block{i}")
            skips.append(enc)
        x, _ = self.conv_block7a(x, film, "conv_block7a")
        for j in range(1, 7):
            x = getattr(self, f"decoder_block{j}")(x, skips[6 - j], film, f"decoder_block{j}")
        x = self.after_conv(x)
        x = F.pad(x, (0, 1))[:, :, :n_frames, :]  # (B, 6, T, F)

        mask_mag = torch.sigmoid(x[:, 0])
        mask_cos, mask_sin = _unit_phasor(torch.tanh(x[:, 1]), torch.tanh(x[:, 2]))
        res_mag = torch.sigmoid(x[:, 3])
        res_cos, res_sin = _unit_phasor(torch.tanh(x[:, 4]), torch.tanh(x[:, 5]))
        mag = mag[:, 0]
        # ∠X + ∠M via the angle-sum identity; |D_M| = α|M||X|; residual β|R|e^{j∠R}.
        out_cos = cos_in * mask_cos - sin_in * mask_sin
        out_sin = sin_in * mask_cos + cos_in * mask_sin
        out_mag = F.relu(self.alpha * mag * mask_mag)
        res_mag = F.relu(self.beta * res_mag)
        real = out_mag * out_cos + res_mag * res_cos
        imag = out_mag * out_sin + res_mag * res_sin
        return torch.istft(
            torch.complex(real, imag).transpose(1, 2),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            length=n,
        )


class DroneAudioNet(nn.Module):
    """16 kHz SE wrapper: ``forward((B, 1, T) mixture) -> (B, 1, T)`` = mixture − drone estimate."""

    def __init__(
        self,
        checkpoint: str | None = None,
        sample_rate: int = 16000,
        native_sample_rate: int = 32000,
        channels: Sequence[int] = AUDIOSEP_CHANNELS,
        n_fft: int = 2048,
        hop_length: int = 320,
        momentum: float = 0.01,
        condition_size: int = 512,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.native_sample_rate = int(native_sample_rate)
        self.base = ResUNet30(
            channels=channels, n_fft=n_fft, hop_length=hop_length, momentum=momentum
        )
        self.film = nn.ModuleDict()
        for name, width in self.base.film_sizes().items():
            layer = nn.Linear(condition_size, width)
            _init_layer(layer)
            self.film[name] = layer

        emb = torch.load(_QUERY_EMBEDDING_FILE, map_location="cpu", weights_only=True)
        if emb.shape != (condition_size,):
            raise ValueError(
                f"query embedding has shape {tuple(emb.shape)}, expected ({condition_size},)"
            )
        self.register_buffer("query_embedding", emb.reshape(1, -1).float(), persistent=False)

        g = math.gcd(self.sample_rate, self.native_sample_rate)
        up, down = self.sample_rate // g, self.native_sample_rate // g
        self.upsample = ResampleFrac(up, down) if up != down else None
        self.downsample = ResampleFrac(down, up) if up != down else None

        if checkpoint is not None:
            from utils.checkpoints import resolve_checkpoint_uri

            path = resolve_checkpoint_uri(checkpoint)
            ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            self.load_upstream_state_dict(ckpt.get("state_dict", ckpt))

    def load_upstream_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load an AudioSep or DroneAudioNet Lightning ``state_dict`` (``ss_model.*`` keys).

        A 3-channel AudioSep head is warm-started as in the paper: channels 0..2 copied,
        channels 3..5 Xavier-uniform with zero bias, α = β = 1. A 6-channel DroneAudioNet
        checkpoint (with α, β) loads verbatim. CLAP (``query_encoder.*``) and the frozen
        torchlibrosa DFT kernels are dropped; everything else must match exactly.
        """
        prefix = "ss_model." if any(k.startswith("ss_model.") for k in state) else ""
        sd = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        sd = {
            k: v
            for k, v in sd.items()
            if not k.startswith(("query_encoder.", *_UNUSED_UPSTREAM_PREFIXES))
        }
        w, b = sd["base.after_conv.weight"], sd["base.after_conv.bias"]
        if w.shape[0] == 3:  # AudioSep base -> DroneAudioNet head
            new_w = torch.zeros(self.base.k, *w.shape[1:], dtype=w.dtype)
            new_w[:3] = w
            nn.init.xavier_uniform_(new_w[3:])
            new_b = torch.zeros(self.base.k, dtype=b.dtype)
            new_b[:3] = b
            sd["base.after_conv.weight"], sd["base.after_conv.bias"] = new_w, new_b
            sd.setdefault("base.alpha", torch.tensor(1.0))
            sd.setdefault("base.beta", torch.tensor(1.0))
        elif w.shape[0] != self.base.k:
            raise ValueError(f"unexpected after_conv width {w.shape[0]} (want 3 or {self.base.k})")
        leftover = sorted(k for k in sd if not k.startswith(("base.", "film.")))
        if leftover:
            raise ValueError(f"unexpected keys in upstream state_dict: {leftover[:5]}")
        # Per submodule: the resampler kernels are ours and absent from upstream checkpoints.
        for name in ("base", "film"):
            sub = {k[len(name) + 1 :]: v for k, v in sd.items() if k.startswith(f"{name}.")}
            getattr(self, name).load_state_dict(sub, strict=True)

    def film_biases(self) -> dict[str, torch.Tensor]:
        """FiLM additive biases from the fixed query embedding, shaped (1, C, 1, 1)."""
        return {
            name: layer(self.query_embedding)[:, :, None, None] for name, layer in self.film.items()
        }

    def estimate_drone_native(self, wav: torch.Tensor) -> torch.Tensor:
        """Drone-noise estimate at the native rate: (B, T) or (B, 1, T) -> (B, 1, T)."""
        if wav.dim() == 3:
            wav = wav[:, 0]
        return self.base(wav, self.film_biases()).unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, T) or (B, T) mixture @ ``sample_rate`` -> (B, 1, T) enhanced (x − d̂)."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.shape[1] != 1:
            raise ValueError(
                f"DroneAudioNet is single-channel, got input of shape {tuple(x.shape)}"
            )
        n = x.shape[-1]
        xn = self.upsample(x) if self.upsample is not None else x
        drone = self.estimate_drone_native(xn)
        if self.downsample is not None:
            drone = self.downsample(drone)
        if drone.shape[-1] < n:
            drone = F.pad(drone, (0, n - drone.shape[-1]))
        return x - drone[..., :n]


def build_droneaudionet(checkpoint: str | None = None, **params: Any) -> nn.Module:
    """Factory for the ``_target_`` config path.

    ``checkpoint``: local path or ``r2://`` URI of an AudioSep Lightning checkpoint
    (``audiosep_base_4M_steps.ckpt``: fine-tuning start, head warm-started) or of the
    published DroneAudioNet checkpoint (``droneaudionet.ckpt``: loaded verbatim);
    ``None`` builds with random weights.
    """
    return DroneAudioNet(checkpoint=checkpoint, **params)
