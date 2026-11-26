# pyright: reportCallIssue=false, reportArgumentType=false
"""NCSN++M — the StoRM variant of the NCSN++ U-Net (predictor AND score network).

Port of ``sgmse/backbones/ncsnpp.py::NCSNpp`` from https://github.com/sp-uhh/storm
(commit 257e9636a7251ca40aa200753d5c0fe918e31879, MIT; the network itself is
Apache-2.0 code from https://github.com/yang-song/score_sde). The building blocks
(``ResnetBlockBigGANpp``, ``Combine``, FIR up/down-sampling, Gaussian Fourier
time embedding, ...) are imported from :mod:`models.sgmse.ncsnpp_utils`, which is
numerically identical to the sgmse/storm ``ncsnpp_utils`` (StoRM's copy differs
only in ``ddpm_conv*`` skipping ``zeros_(bias)`` for ``bias=False`` convs, a case
this network never builds).

Why not :class:`models.sgmse.ncsnpp.NCSNpp`: that port hard-codes 2 complex input
channels (x_t, y) and the noise-conditional path. StoRM needs 1 complex input
channel for the predictor (``discriminative=True``) and 3 for the score network
(x_t, y, D(y)), which cannot be configured there.

Differences from the SGMSE+ NCSN++ (all upstream StoRM behaviour, kept verbatim):
  * ``input_channels`` real channels in, ``2`` real (= 1 complex) out through a
    final 1x1 ``output_layer`` applied after the ``scale_by_sigma`` division;
  * ``centered=False`` default, i.e. the stacked real/imag input is mapped through
    ``2 x - 1`` before the first conv;
  * ``discriminative=True`` (the predictor D_theta): no noise conditioning
    (``conditional=False``: the two time-embedding ``Linear`` layers are not
    built), no ``scale_by_sigma``, ``input_channels=2`` and ``time_cond=None``.
    As upstream, the residual blocks still own their (unused) ``Dense_0``
    time projections and the Fourier embedding still owns its ``W`` — they are
    part of the published parameter count (27,342,350) and of the checkpoint.

State-dict keys (``output_layer.*``, ``all_modules.<i>.*``) match upstream, so a
StoRM Lightning checkpoint's ``denoiser_net.*`` / ``score_net.*`` entries load
unchanged. Mono only (upstream ``spatial_channels=1``, the default it was trained
with); the multi-microphone branch is dropped.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from models.sgmse import ncsnpp_utils as layerspp

get_act = layerspp.get_act
default_initializer = layerspp.default_init
conv3x3 = layerspp.conv3x3
ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine


class NCSNppM(nn.Module):
    """StoRM's NCSN++ (``--backbone ncsnpp``). Defaults are the NCSN++M sizes:
    ``nf=128``, ``ch_mult=(1, 2, 2, 2)``, one ResNet block per level, attention in
    the bottleneck only (``attn_resolutions=(0,)`` never matches a feature size).

    ``forward(x, time_cond)``: ``x`` complex ``(B, input_channels // 2, F, T)``,
    ``time_cond`` the diffusion time ``(B,)`` (``None`` for the predictor);
    returns complex ``(B, 1, F, T)``. ``F`` must be divisible by
    ``2 ** (len(ch_mult) - 1)``, ``T`` too (callers pad ``T`` to 64).
    """

    def __init__(
        self,
        scale_by_sigma: bool = True,
        nonlinearity: str = "swish",
        nf: int = 128,
        ch_mult: Sequence[int] = (1, 2, 2, 2),
        num_res_blocks: int = 1,
        attn_resolutions: Sequence[int] = (0,),
        resamp_with_conv: bool = True,
        conditional: bool = True,
        fir: bool = True,
        fir_kernel: Sequence[int] = (1, 3, 3, 1),
        skip_rescale: bool = True,
        resblock_type: str = "biggan",
        progressive: str = "output_skip",
        progressive_input: str = "input_skip",
        progressive_combine: str = "sum",
        init_scale: float = 0.0,
        fourier_scale: float = 16,
        image_size: int = 256,
        embedding_type: str = "fourier",
        input_channels: int = 4,
        dropout: float = 0.0,
        centered: bool = False,
        discriminative: bool = False,
    ) -> None:
        super().__init__()
        self.act = act = get_act(nonlinearity)
        self.nf = nf
        ch_mult = tuple(int(c) for c in ch_mult)
        fir_kernel = [int(k) for k in fir_kernel]
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions = tuple(int(a) for a in attn_resolutions)
        self.num_resolutions = num_resolutions = len(ch_mult)
        all_resolutions = [image_size // (2**i) for i in range(num_resolutions)]

        self.discriminative = discriminative
        if discriminative:
            # upstream overrides these for the predictive network
            conditional = False
            scale_by_sigma = False
            input_channels = 2  # y.real, y.imag

        self.conditional = conditional
        self.centered = centered
        self.scale_by_sigma = scale_by_sigma
        self.skip_rescale = skip_rescale
        self.resblock_type = resblock_type = resblock_type.lower()
        self.progressive = progressive = progressive.lower()
        self.progressive_input = progressive_input = progressive_input.lower()
        self.embedding_type = embedding_type = embedding_type.lower()
        assert progressive in ["none", "output_skip", "residual"]
        assert progressive_input in ["none", "input_skip", "residual"]
        if embedding_type != "fourier":
            raise ValueError(
                f"embedding type {embedding_type!r} unsupported (StoRM uses 'fourier')"
            )
        assert input_channels % 2 == 0, "input_channels counts real channels (2 per complex input)"
        combine_method = progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)
        self.input_channels = total_channels = input_channels

        self.output_layer = nn.Conv2d(total_channels, 2, 1)

        modules: list[nn.Module] = []
        AttnBlock = functools.partial(
            layerspp.AttnBlockpp, init_scale=init_scale, skip_rescale=skip_rescale
        )
        Upsample = functools.partial(
            layerspp.Upsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel
        )
        if progressive == "output_skip":
            self.pyramid_upsample = layerspp.Upsample(
                fir=fir, fir_kernel=fir_kernel, with_conv=False
            )
        elif progressive == "residual":
            pyramid_upsample = functools.partial(
                layerspp.Upsample, fir=fir, fir_kernel=fir_kernel, with_conv=True
            )
        Downsample = functools.partial(
            layerspp.Downsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel
        )
        if progressive_input == "input_skip":
            self.pyramid_downsample = layerspp.Downsample(
                fir=fir, fir_kernel=fir_kernel, with_conv=False
            )
        elif progressive_input == "residual":
            pyramid_downsample = functools.partial(
                layerspp.Downsample, fir=fir, fir_kernel=fir_kernel, with_conv=True
            )

        if resblock_type == "ddpm":
            ResnetBlock = functools.partial(
                ResnetBlockDDPM,
                act=act,
                dropout=dropout,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
            )
        elif resblock_type == "biggan":
            ResnetBlock = functools.partial(
                ResnetBlockBigGAN,
                act=act,
                dropout=dropout,
                fir=fir,
                fir_kernel=fir_kernel,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
            )
        else:
            raise ValueError(f"resblock type {resblock_type} unrecognized.")

        # Time embedding (built even for the predictor, as upstream)
        modules.append(layerspp.GaussianFourierProjection(embedding_size=nf, scale=fourier_scale))
        embed_dim = 2 * nf
        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        # Downsampling block
        input_pyramid_ch = total_channels
        modules.append(conv3x3(total_channels, nf))
        hs_c = [nf]
        in_ch = nf
        for i_level in range(num_resolutions):
            for _ in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == "ddpm":
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))
                if progressive_input == "input_skip":
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == "cat":
                        in_ch *= 2
                elif progressive_input == "residual":
                    modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
                    input_pyramid_ch = in_ch
                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for _ in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if progressive != "none":
                if i_level == num_resolutions - 1:
                    if progressive == "output_skip":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
                            )
                        )
                        modules.append(conv3x3(in_ch, total_channels, init_scale=init_scale))
                        pyramid_ch = total_channels
                    else:  # residual
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
                            )
                        )
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                else:
                    if progressive == "output_skip":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
                            )
                        )
                        modules.append(
                            conv3x3(in_ch, total_channels, bias=True, init_scale=init_scale)
                        )
                        pyramid_ch = total_channels
                    else:  # residual
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch

            if i_level != 0:
                if resblock_type == "ddpm":
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != "output_skip":
            modules.append(
                nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
            )
            modules.append(conv3x3(in_ch, total_channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor, time_cond: torch.Tensor | None = None) -> torch.Tensor:
        modules = self.all_modules
        m_idx = 0

        # complex (B, C, F, T) -> real (B, 2C, F, T) as [c0.re, c0.im, c1.re, c1.im, ...]
        x = torch.cat(
            [
                torch.cat([x[:, [c], :, :].real, x[:, [c], :, :].imag], dim=1)
                for c in range(self.input_channels // 2)
            ],
            dim=1,
        )

        # Gaussian Fourier embedding of log(t); upstream feeds the diffusion time t
        # (not sigma(t)) here and divides by it below when scale_by_sigma.
        used_sigmas = time_cond
        temb = modules[m_idx](torch.log(used_sigmas)) if used_sigmas is not None else None
        m_idx += 1

        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if not self.centered:
            x = 2 * x - 1.0

        input_pyramid = x if self.progressive_input != "none" else None

        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            for _ in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-2] in self.attn_resolutions:  # frequency axis, as upstream
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](hs[-1])
                else:
                    h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if self.progressive_input == "input_skip":
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1
                elif self.progressive_input == "residual":
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.0)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid
                hs.append(h)

        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        pyramid = None
        for i_level in reversed(range(self.num_resolutions)):
            for _ in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            if h.shape[-2] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if self.progressive != "none":
                if i_level == self.num_resolutions - 1:
                    pyramid = self.act(modules[m_idx](h))
                    m_idx += 1
                    pyramid = modules[m_idx](pyramid)
                    m_idx += 1
                elif self.progressive == "output_skip":
                    pyramid = self.pyramid_upsample(pyramid)
                    pyramid_h = self.act(modules[m_idx](h))
                    m_idx += 1
                    pyramid_h = modules[m_idx](pyramid_h)
                    m_idx += 1
                    pyramid = pyramid + pyramid_h
                else:  # residual
                    pyramid = modules[m_idx](pyramid)
                    m_idx += 1
                    pyramid = (pyramid + h) / np.sqrt(2.0) if self.skip_rescale else pyramid + h
                    h = pyramid

            if i_level != 0:
                h = modules[m_idx](h) if self.resblock_type == "ddpm" else modules[m_idx](h, temb)
                m_idx += 1

        assert not hs

        if self.progressive == "output_skip":
            assert pyramid is not None
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        assert m_idx == len(modules), "Implementation error"
        if self.scale_by_sigma:
            assert used_sigmas is not None
            h = h / used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))

        h = self.output_layer(h)  # (B, 2, F, T)
        h = torch.permute(h, (0, 2, 3, 1)).contiguous()  # (B, F, T, 2)
        return torch.view_as_complex(h)[:, None, :, :]  # (B, 1, F, T)
