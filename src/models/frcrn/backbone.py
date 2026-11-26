"""FRCRN backbone — the released ModelScope ``speech_frcrn_ans_cirm_16k`` network.

Adapted (Apache-2.0) from ModelScope ``modelscope/models/audio/ans/`` —
``frcrn.py``, ``unet.py``, ``complex_nn.py``, ``conv_stft.py``,
``se_module_complex.py``, ``layers/uni_deep_fsmn.py`` — at
https://github.com/modelscope/modelscope commit
``987d79f0029f6438197a8093a406848467336dfd`` (Copyright (c) Alibaba, Inc. and
its affiliates, Apache License 2.0). ModelScope's complex layers are in turn
"modified based on" https://github.com/sweetcocoa/DeepComplexUNetPyTorch.
Paper: Zhao, Ma, Watcharasupat, Gan, "FRCRN: Boosting Feature Representation
Using Frequency Recurrence for Monaural Speech Enhancement", ICASSP 2022,
arXiv:2206.07293.

Module names, registration order and the duplicated registrations
(``encoder{i}`` *and* ``encoders.{i}``, ``fsmn_dec6`` and ``se_layer_dec5``
that the forward never uses) are kept verbatim so the published
``pytorch_model.bin`` (1692 keys) loads with ``strict=True``.

Parity (scratch script against the ModelScope source run with stubbed
``modelscope`` imports, CPU fp32, torch 2.7.0): with the published checkpoint
loaded ``strict=True`` into both, a 1 s random input (batch 2) gives max
``|diff|`` = 0.0 on both outputs ``wav_l1``/``wav_l2``; the same holds for
random upstream weights copied through ``state_dict``. Parameter count
13,963,604 in both (13,141,844 learnable + 821,760 fixed STFT/iSTFT kernels;
``named_parameters`` counts the double-registered modules once).

Changes against upstream, none of which alters the numerics:

* ModelScope plumbing removed (``TorchModel``/registry decorator, Kaldi
  import/export on ``UniDeepFsmn``, the loss methods — the project supplies
  its own loss).
* ``UniDeepFsmn.forward`` ends in ``input + out1.squeeze()`` upstream; here
  ``squeeze(1)`` (the only axis that is meant to go), which is identical
  whenever upstream's broadcast does not silently misbehave (``B*T == 1``).
* ``UNet`` keeps only the depth-14 size table (the one the published config
  uses) and the complex path (the real path never ran upstream: its
  ``nn.Conv2d`` would receive the 5-D complex tensor). ``log_amp`` and
  ``model_complexity`` were inert for depth 14 and are dropped.
* ``UNet.forward`` takes an optional per-encoder ``enc_adapters`` sequence
  (``None`` = upstream behaviour); see :mod:`models.frcrn.model`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FRCRN", "UNet"]


# --------------------------------------------------------------------------- STFT


def _init_kernels(
    win_len: int, fft_len: int, win_type: str | None, invers: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Conv1d (i)STFT kernels, float64 numpy then float32 — upstream ``init_kernels``."""
    from scipy.signal import get_window  # lazy: keep ``import models.frcrn`` cheap

    if win_type == "None" or win_type is None:
        window = np.ones(win_len)
    else:
        window = get_window(win_type, win_len, fftbins=True) ** 0.5  # sqrt-Hann
    fourier_basis = np.fft.rfft(np.eye(fft_len))[:win_len]
    kernel = np.concatenate([np.real(fourier_basis), np.imag(fourier_basis)], 1).T
    if invers:
        kernel = np.linalg.pinv(kernel).T
    kernel = (kernel * window)[:, None, :]
    return (
        torch.from_numpy(kernel.astype(np.float32)),
        torch.from_numpy(window[None, :, None].astype(np.float32)),
    )


class ConvSTFT(nn.Module):
    """Un-centred, un-padded STFT as a strided Conv1d: (B, T) -> (B, 2*(N/2+1), frames)."""

    def __init__(self, win_len: int, win_inc: int, fft_len: int, win_type: str) -> None:
        super().__init__()
        kernel, _ = _init_kernels(win_len, fft_len, win_type)
        self.weight = nn.Parameter(kernel, requires_grad=False)  # upstream fix=True
        self.stride = win_inc

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)
        return F.conv1d(inputs, self.weight, stride=self.stride)


class ConviSTFT(nn.Module):
    """Pseudo-inverse synthesis + squared-window overlap normalisation (torch-stft)."""

    window: torch.Tensor
    enframe: torch.Tensor

    def __init__(self, win_len: int, win_inc: int, fft_len: int, win_type: str) -> None:
        super().__init__()
        kernel, window = _init_kernels(win_len, fft_len, win_type, invers=True)
        self.weight = nn.Parameter(kernel, requires_grad=False)
        self.stride = win_inc
        self.register_buffer("window", window)
        self.register_buffer("enframe", torch.eye(win_len)[:, None, :])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.stride)
        t = self.window.repeat(1, 1, inputs.size(-1)) ** 2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)
        return outputs / (coff + 1e-8)


# --------------------------------------------------------------------------- complex layers


class UniDeepFsmn(nn.Module):
    """Unidirectional deep-FSMN cell: ReLU-linear, projection, causal depthwise memory.

    ``(N, L, D) -> (N, L, D)``; the memory runs along ``L`` with ``lorder`` taps.
    """

    def __init__(self, input_dim: int, output_dim: int, lorder: int, hidden_size: int) -> None:
        super().__init__()
        self.lorder = lorder
        self.linear = nn.Linear(input_dim, hidden_size)
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        self.conv1 = nn.Conv2d(
            output_dim, output_dim, (lorder, 1), (1, 1), groups=output_dim, bias=False
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        f1 = F.relu(self.linear(input))
        p1 = self.project(f1)
        x_per = p1.unsqueeze(1).permute(0, 3, 2, 1)  # (N, D, L, 1)
        y = F.pad(x_per, [0, 0, self.lorder - 1, 0])
        out = x_per + self.conv1(y)
        return input + out.permute(0, 3, 2, 1).squeeze(1)


class ComplexUniDeepFsmn(nn.Module):
    """Two stacked complex FSMN layers over TIME (the bottleneck recurrent module).

    ``(B, C, H, T, 2)`` with ``C*H == nIn``; memory along ``T``.
    """

    def __init__(self, nIn: int, nHidden: int = 128, nOut: int = 128) -> None:
        super().__init__()
        self.fsmn_re_L1 = UniDeepFsmn(nIn, nHidden, 20, nHidden)
        self.fsmn_im_L1 = UniDeepFsmn(nIn, nHidden, 20, nHidden)
        self.fsmn_re_L2 = UniDeepFsmn(nHidden, nOut, 20, nHidden)
        self.fsmn_im_L2 = UniDeepFsmn(nHidden, nOut, 20, nHidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, T, d = x.size()
        x = torch.reshape(x, (b, c * h, T, d)).transpose(1, 2)  # (b, T, c*h, 2)
        real_L1 = self.fsmn_re_L1(x[..., 0]) - self.fsmn_im_L1(x[..., 1])
        imaginary_L1 = self.fsmn_re_L1(x[..., 1]) + self.fsmn_im_L1(x[..., 0])
        real = self.fsmn_re_L2(real_L1) - self.fsmn_im_L2(imaginary_L1)
        imaginary = self.fsmn_re_L2(imaginary_L1) + self.fsmn_im_L2(real_L1)
        output = torch.stack((real, imaginary), dim=-1).transpose(1, 2)
        return torch.reshape(output, (b, c, h, T, d))


class ComplexUniDeepFsmn_L1(nn.Module):
    """One complex FSMN layer over FREQUENCY (the "frequency recurrence" of FRCRN).

    ``(B, C, H, T, 2)``: features = the ``C`` channels, memory along ``H`` (freq).
    """

    def __init__(self, nIn: int, nHidden: int = 128, nOut: int = 128) -> None:
        super().__init__()
        self.fsmn_re_L1 = UniDeepFsmn(nIn, nHidden, 20, nHidden)
        self.fsmn_im_L1 = UniDeepFsmn(nIn, nHidden, 20, nHidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, T, d = x.size()
        x = torch.reshape(torch.transpose(x, 1, 3), (b * T, h, c, d))
        real = self.fsmn_re_L1(x[..., 0]) - self.fsmn_im_L1(x[..., 1])
        imaginary = self.fsmn_re_L1(x[..., 1]) + self.fsmn_im_L1(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return torch.transpose(torch.reshape(output, (b, T, h, c, d)), 1, 3)


class ComplexConv2d(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size, **kwargs) -> None:
        super().__init__()
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, **kwargs)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        return torch.stack((real, imaginary), dim=-1)


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size, **kwargs) -> None:
        super().__init__()
        self.tconv_re = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, **kwargs)
        self.tconv_im = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real = self.tconv_re(x[..., 0]) - self.tconv_im(x[..., 1])
        imaginary = self.tconv_re(x[..., 1]) + self.tconv_im(x[..., 0])
        return torch.stack((real, imaginary), dim=-1)


class ComplexBatchNorm2d(nn.Module):
    """Independent BatchNorm2d on the real and imaginary parts (upstream's "complex BN")."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.bn_re = nn.BatchNorm2d(num_features)
        self.bn_im = nn.BatchNorm2d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack((self.bn_re(x[..., 0]), self.bn_im(x[..., 1])), dim=-1)


class SELayer(nn.Module):
    """Complex squeeze-and-excitation (the "attention block" on the skip path).

    Global average over (freq, time) — the reason the released model is non-causal.
    """

    def __init__(self, channel: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_r = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )
        self.fc_i = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        x_r = self.avg_pool(x[..., 0]).view(b, c)
        x_i = self.avg_pool(x[..., 1]).view(b, c)
        y_r = self.fc_r(x_r).view(b, c, 1, 1, 1) - self.fc_i(x_i).view(b, c, 1, 1, 1)
        y_i = self.fc_r(x_i).view(b, c, 1, 1, 1) + self.fc_i(x_r).view(b, c, 1, 1, 1)
        return x * torch.cat([y_r, y_i], 4)


# --------------------------------------------------------------------------- U-Net


class Encoder(nn.Module):
    """Complex Conv2d -> complex BN -> LeakyReLU (the conv half of a CR block)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, padding_mode):
        super().__init__()
        self.conv = ComplexConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
        )
        self.bn = ComplexBatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.transconv = ComplexConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bn = ComplexBatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.transconv(x)))


# Depth-14 size table of upstream ``UNet.set_size`` (axes: freq, time).
_ENC_CHANNELS = [1, 128, 128, 128, 128, 128, 128, 128]
_ENC_KERNELS = [(5, 2)] * 6 + [(2, 2)]
_ENC_STRIDES = [(2, 1)] * 7
_ENC_PADDINGS = [(0, 1)] * 7
_DEC_CHANNELS = [64, 128, 128, 128, 128, 128, 128, 1]
_DEC_KERNELS = [(2, 2), (5, 2), (5, 2), (5, 2), (6, 2), (5, 2), (5, 2)]
_DEC_STRIDES = [(2, 1)] * 7
_DEC_PADDINGS = [(0, 1)] * 7


def encoder_freq_sizes(n_freq: int) -> list[int]:
    """Frequency size of every encoder output (321 bins -> [159, 78, 37, 17, 7, 2, 1])."""
    sizes = []
    for (k, _), (s, _), (p, _) in zip(_ENC_KERNELS, _ENC_STRIDES, _ENC_PADDINGS, strict=True):
        n_freq = (n_freq + 2 * p - k) // s + 1
        sizes.append(n_freq)
    return sizes


class UNet(nn.Module):
    """Complex CRED U-Net, depth 14 (7 encoder / 7 decoder CR blocks), 128 channels."""

    def __init__(self, padding_mode: str = "zeros") -> None:
        super().__init__()
        self.model_length = 7
        self.fsmn = ComplexUniDeepFsmn(128, 128, 128)
        encoders, se_enc, fsmn_enc = [], [], []
        for i in range(self.model_length):
            m = ComplexUniDeepFsmn_L1(128, 128, 128)
            self.add_module(f"fsmn_enc{i}", m)
            fsmn_enc.append(m)
            m = Encoder(
                _ENC_CHANNELS[i],
                _ENC_CHANNELS[i + 1],
                _ENC_KERNELS[i],
                _ENC_STRIDES[i],
                _ENC_PADDINGS[i],
                padding_mode,
            )
            self.add_module(f"encoder{i}", m)
            encoders.append(m)
            m = SELayer(_ENC_CHANNELS[i + 1], 8)
            self.add_module(f"se_layer_enc{i}", m)
            se_enc.append(m)
        decoders, fsmn_dec, se_dec = [], [], []
        for i in range(self.model_length):
            m = ComplexUniDeepFsmn_L1(128, 128, 128)
            self.add_module(f"fsmn_dec{i}", m)
            fsmn_dec.append(m)
            m = Decoder(
                _DEC_CHANNELS[i] * 2,
                _DEC_CHANNELS[i + 1],
                _DEC_KERNELS[i],
                _DEC_STRIDES[i],
                _DEC_PADDINGS[i],
            )
            self.add_module(f"decoder{i}", m)
            decoders.append(m)
            if i < self.model_length - 1:
                m = SELayer(_DEC_CHANNELS[i + 1], 8)
                self.add_module(f"se_layer_dec{i}", m)
                se_dec.append(m)
        self.linear = ComplexConv2d(_DEC_CHANNELS[-1], 1, 1)
        self.decoders = nn.ModuleList(decoders)
        self.encoders = nn.ModuleList(encoders)
        self.se_layers_enc = nn.ModuleList(se_enc)
        self.se_layers_dec = nn.ModuleList(se_dec)
        self.fsmn_enc = nn.ModuleList(fsmn_enc)
        self.fsmn_dec = nn.ModuleList(fsmn_dec)

    def forward(
        self, inputs: torch.Tensor, enc_adapters: Sequence[nn.Module] | None = None
    ) -> torch.Tensor:
        """(B, 1, F, T, 2) -> (B, 1, F, T, 2) unbounded complex mask logits.

        ``enc_adapters[i]`` (if given) maps encoder ``i``'s output before it
        feeds both the skip-path SE layer and the next encoder block.
        """
        x = inputs
        xs_se = [x]
        for i, encoder in enumerate(self.encoders):
            if i > 0:
                x = self.fsmn_enc[i](x)
            x = encoder(x)
            if enc_adapters is not None:
                x = enc_adapters[i](x)
            xs_se.append(self.se_layers_enc[i](x))
        x = self.fsmn(x)
        p = x
        for i, decoder in enumerate(self.decoders):
            p = decoder(p)
            if i < self.model_length - 1:
                p = self.fsmn_dec[i](p)
            if i == self.model_length - 1:
                break
            if i < self.model_length - 2:
                p = self.se_layers_dec[i](p)
            p = torch.cat([p, xs_se[self.model_length - 1 - i]], dim=1)
        return self.linear(p)


# --------------------------------------------------------------------------- FRCRN


class FRCRN(nn.Module):
    """Two cascaded U-Nets predicting a cIRM; ``unet2`` refines ``unet1``'s mask residually.

    ``forward(wav (B, T))`` returns ``(est_wav_l1, est_wav_l2)``; ``est_wav_l2``
    (mask ``tanh(u2) + tanh(u1)``) is the model output, as ``wav_l2`` upstream.
    Output length is ``(frames - 1) * win_inc + win_len``, with
    ``frames = (T - win_len) // win_inc + 1`` (no centring, no padding).
    """

    def __init__(
        self,
        win_len: int = 640,
        win_inc: int = 320,
        fft_len: int = 640,
        win_type: str = "hann",
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.feat_dim = fft_len // 2 + 1
        self.win_len = win_len
        self.win_inc = win_inc
        self.stft = ConvSTFT(win_len, win_inc, fft_len, win_type)
        self.istft = ConviSTFT(win_len, win_inc, fft_len, win_type)
        self.unet = UNet(padding_mode)
        self.unet2 = UNet(padding_mode)

    def forward(
        self,
        inputs: torch.Tensor,
        enc_adapters: Sequence[Sequence[nn.Module] | None] = (None, None),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cmp_spec = self.stft(inputs).unsqueeze(1)  # (B, 1, 2F, T)
        cmp_spec = torch.cat(
            [cmp_spec[:, :, : self.feat_dim, :], cmp_spec[:, :, self.feat_dim :, :]], 1
        )  # (B, 2, F, T)
        cmp_spec = torch.transpose(cmp_spec.unsqueeze(4), 1, 4)  # (B, 1, F, T, 2)
        unet1_out = self.unet(cmp_spec, enc_adapters[0])
        cmp_mask1 = torch.tanh(unet1_out)
        unet2_out = self.unet2(unet1_out, enc_adapters[1])
        cmp_mask2 = torch.tanh(unet2_out) + cmp_mask1
        return self._apply_mask(cmp_spec, cmp_mask1), self._apply_mask(cmp_spec, cmp_mask2)

    def _apply_mask(self, cmp_spec: torch.Tensor, cmp_mask: torch.Tensor) -> torch.Tensor:
        est_spec = torch.cat(
            [
                cmp_spec[..., 0] * cmp_mask[..., 0] - cmp_spec[..., 1] * cmp_mask[..., 1],
                cmp_spec[..., 0] * cmp_mask[..., 1] + cmp_spec[..., 1] * cmp_mask[..., 0],
            ],
            1,
        )  # (B, 2, F, T)
        est_spec = torch.cat([est_spec[:, 0], est_spec[:, 1]], 1)  # (B, 2F, T)
        return self.istft(est_spec).squeeze(1)
