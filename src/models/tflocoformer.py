"""TF-Locoformer speech-enhancement model — faithful pure-PyTorch port.

Paper: K. Saijo, G. Wichern, F. G. Germain, Z. Pan, J. Le Roux,
"TF-Locoformer: Transformer with Local Modeling by Convolution for Speech
Separation and Enhancement", IWAENC 2024, arXiv:2408.03440.

Upstream: https://github.com/merlresearch/tf-locoformer at commit
``7a615460d347ff7334a13dbb831d16280da72cdc`` (Apache-2.0, Copyright (C) 2024
Mitsubishi Electric Research Laboratories). The separator below is adapted
from ``standalone/tflocoformer_separator.py`` (identical to the ESPnet
version ``espnet2/enh/separator/tflocoformer_separator.py`` up to ESPnet
plumbing). The STFT front-end, the variance normalisation and the weight
initialisation reproduce the ESPnet-SE pipeline the paper trained with
(ESPnet commit ``90eed8e53498e7af682bc6ff39d9067ae440d6a4``:
``espnet2/layers/stft.py``, ``espnet2/enh/espnet_model.py``,
``espnet2/torch_utils/initialize.py``), with the recipe
``egs2/dns_ins20/enh1/conf/tuning/train_enh_tflocoformer.yaml``.

Pipeline (mono, one output source) and where each paper statement lives:

* "Each input mixture is normalized by dividing it by its standard
  deviation" (Sec. 3.1) -> ``TFLocoformer.forward``: ``x / x.std()``
  (unbiased, per utterance, no epsilon, exactly ESPnet ``normalize_variance``);
  the estimate is multiplied back by the same std.
* "STFT window and hop sizes to 16 ms and 8 ms" (Sec. 3.1) at 16 kHz (DNS)
  -> ``n_fft = win = 256``, ``hop = 128``, periodic Hann, ``center=True``
  reflect padding, one-sided (F = 129), un-normalised ``torch.stft`` /
  ``torch.istft`` with ``length=T`` (ESPnet ``Stft`` defaults).
* "Z = gLN(Conv2D(X))", Eq. (1) -> ``TFLocoformerSeparator.conv``:
  Conv2d(2 -> D, 3x3, pad 1) + GroupNorm(1, D).
* Frequency then temporal modelling (``tf_order="ft"``), B blocks ->
  ``TFLocoformerBlock.freq_path`` / ``.frame_path``.
* Eqs. (2)-(4) Macaron block: ConvSwiGLU, MHSA, ConvSwiGLU, each pre-normed
  with a residual -> ``LocoformerBlock``.
* MHSA with H heads of D/H dims and rotary positional encoding along the
  sequence (freq bins on the frequency path, frames on the time path), one
  RoPE table shared by all frequency paths and one by all time paths ->
  ``MultiHeadSelfAttention`` + ``RotaryEmbedding``.
* Eqs. (6)-(8) ConvSwiGLU: Norm, ``Swish(Conv1D(Z)) * Conv1D(Z)`` (one Conv1d
  of 2C channels split into value and gate halves), Deconv1D (kernel K,
  stride S) -> ``SwiGLUConvDeconv1d``.
* RMSGroupNorm with G groups and a D-dim gain (upstream has no bias) ->
  ``RMSGroupNorm``.
* Eq. (5) ``S = DeConv2D(Z)`` -> ``TFLocoformerSeparator.deconv``:
  ConvTranspose2d(D -> 2, 3x3, pad 1), the RI of the enhanced speech
  (direct complex spectral mapping, no mask).

Medium configuration (Table 1, "M"): D = 128, B = 6, C = 384 (both FFNs),
K = 4, S = 1, H = 4, G = 4, attention dim 128, dropout 0, eps 1e-5. Paper:
15.0M parameters. This port: 14,973,346 ``parameters()`` elements, of which
32 are the two non-trainable RoPE frequency tables (``requires_grad=False``,
kept as parameters because upstream's ``rotary_embedding_torch`` stores
them so and the published checkpoints carry them) -> 14,973,314 trainable.
Identical to the upstream module on the same configuration.

Deviations from upstream, none of them numerical in fp32:

* **The 1/2 Macaron factor is absent**, as in upstream's code and published
  checkpoints (upstream comment: the paper's experiments used it, the code
  dropped it "by mistake" and the released models were trained without it).
* **RoPE is re-implemented** (~20 lines) instead of importing
  ``rotary_embedding_torch``: upstream pins 0.6.1, the project pins 0.8.6,
  and both compute the rotation positions in the query dtype (bf16 position
  rounding above 256 frames under autocast) and keep a frequency cache that
  is not refreshed by ``load_state_dict``. This version computes the angles in
  fp32 from the ``freqs`` parameter at every call; in fp32 it is the same
  arithmetic (outer product, interleaved pair repeat, ``rotate_half``) and
  the state-dict key (``...attn.rope.freqs``) is unchanged.
* **No ``torch.backends.cuda.sdp_kernel`` context.** Upstream (with
  ``flash_attention=False``) disables only the flash kernel; flash is never
  eligible in fp32 anyway, so PyTorch's default kernel choice is used.
* **Autocast guards use the input's device type** (upstream guards only
  ``"cuda"``) around the encoder conv, the decoder deconv, RoPE and
  RMSGroupNorm; the STFT/iSTFT run in fp32 as in ESPnet's encoder/decoder.
* **ESPnet's ``init: xavier_uniform`` is applied in the constructor**
  (``espnet_xavier_init=True``): every parameter with ``dim > 1`` gets
  ``xavier_uniform_``, every 1-D ``*.bias`` is zeroed, GroupNorm is reset.
  Upstream's module itself keeps PyTorch defaults; the trainer applied this.
* Ablation switches of upstream (``norm_type="layernorm"``, ``pos_enc="nope"``,
  ``tf_order="tf"``, ``conv1d`` FFN, ``num_spk > 1``, ``flash_attention``) are
  not ported; the published medium enhancement configuration needs none.

Unstated in the paper, taken from the upstream DNS recipe: Hann window
(ESPnet default), ``center=True``, attention dim = D = 128, RoPE base 10000
over the full head dim 32 (``rotary_embedding_torch`` defaults), xavier
initialisation, ``eps = 1e-5`` in GroupNorm and RMSGroupNorm.

The model is native 16 kHz; no resampling. Pretrained weights for exactly
this configuration (DNS-Interspeech2020, 16 kHz, 1 speaker) ship with the
upstream repository at
``egs2/dns_ins20/enh1/exp/enh_train_enh_tflocoformer_raw/valid.loss.ave_5best.pth``;
their ESPnet keys (``separator.*``) load into :class:`TFLocoformer` with
``strict=True`` unchanged (``build_tflocoformer(checkpoint=...)``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TFLocoformer", "TFLocoformerSeparator", "build_tflocoformer"]


class RotaryEmbedding(nn.Module):
    """Rotary positional encoding over the second-to-last axis (interleaved pairs).

    Same arithmetic as ``rotary_embedding_torch.RotaryEmbedding(dim)`` with its
    defaults (``freqs_for="lang"``, ``theta=10000``, rotation over the full
    ``dim``), always in fp32, without a cache.
    """

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
        self.freqs = nn.Parameter(freqs, requires_grad=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Rotate ``t`` of shape ``(..., seq, dim)``."""
        with torch.autocast(t.device.type, enabled=False):
            pos = torch.arange(t.shape[-2], device=t.device, dtype=torch.float32)
            angles = torch.einsum("n,f->nf", pos, self.freqs.float())
            angles = angles.repeat_interleave(2, dim=-1)  # (seq, dim): (f r), r = 2
            x = t.float()
            x1, x2 = x.unflatten(-1, (-1, 2)).unbind(dim=-1)
            rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
            out = x * angles.cos() + rotated * angles.sin()
        return out.to(t.dtype)


class RMSGroupNorm(nn.Module):
    """RMS normalisation of each of ``num_groups`` sub-vectors of the last axis, D-dim gain."""

    def __init__(self, num_groups: int, dim: int, eps: float = 1e-5):
        super().__init__()
        if dim % num_groups != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_groups ({num_groups})")
        self.num_groups = num_groups
        self.dim_per_group = dim // num_groups
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(x.device.type, enabled=False):
            others = x.shape[:-1]
            x = x.view(others + (self.num_groups, self.dim_per_group))
            rms = x.norm(2, dim=-1, keepdim=True) * self.dim_per_group ** (-1.0 / 2)
            out = (x / (rms + self.eps)).view(others + (-1,))
            return out * self.gamma


class SwiGLUConvDeconv1d(nn.Module):
    """ConvSwiGLU FFN, Eqs. (7)-(8): Conv1d -> SwiGLU gate -> ConvTranspose1d along seq2."""

    def __init__(
        self, dim: int, dim_inner: int, conv1d_kernel: int, conv1d_shift: int, dropout: float = 0.0
    ):
        super().__init__()
        self.conv1d = nn.Conv1d(dim, dim_inner * 2, conv1d_kernel, stride=conv1d_shift)
        self.swish = nn.SiLU()
        self.deconv1d = nn.ConvTranspose1d(dim_inner, dim, conv1d_kernel, stride=conv1d_shift)
        self.dropout = nn.Dropout(dropout)
        self.dim_inner = dim_inner
        self.diff_ks = conv1d_kernel - conv1d_shift
        self.conv1d_kernel = conv1d_kernel
        self.conv1d_shift = conv1d_shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, seq1, seq2, C)``; the convolution runs along ``seq2``."""
        b, s1, s2, h = x.shape
        x = x.contiguous().view(b * s1, s2, h).transpose(-1, -2)

        seq_len = (
            math.ceil((s2 + 2 * self.diff_ks - self.conv1d_kernel) / self.conv1d_shift)
            * self.conv1d_shift
            + self.conv1d_kernel
        )
        x = F.pad(x, (self.diff_ks, seq_len - s2 - self.diff_ks))

        x = self.conv1d(x)
        gate = self.swish(x[..., self.dim_inner :, :])
        x = x[..., : self.dim_inner, :] * gate
        x = self.dropout(x)
        x = self.deconv1d(x).transpose(-1, -2)

        x = x[..., self.diff_ks : self.diff_ks + s2, :]
        return self.dropout(x).view(b, s1, s2, h)


class MultiHeadSelfAttention(nn.Module):
    """Bias-free QKV projection, RoPE on Q/K, scaled dot-product attention, output projection."""

    def __init__(
        self, emb_dim: int, attention_dim: int, n_heads: int, dropout: float, rope: RotaryEmbedding
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.rope = rope
        self.qkv = nn.Linear(emb_dim, attention_dim * 3, bias=False)
        self.aggregate_heads = nn.Sequential(
            nn.Linear(attention_dim, emb_dim, bias=False), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(N, seq, C)``."""
        n_batch, seq_len = x.shape[:2]
        qkv = (
            self.qkv(x).reshape(n_batch, seq_len, 3, self.n_heads, -1).movedim(-2, 1)
        )  # (N, H, seq, 3, d)
        query, key, value = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]
        query = self.rope(query)
        key = self.rope(key)
        out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=self.dropout if self.training else 0.0
        )  # (N, H, seq, d)
        out = out.transpose(1, 2)
        out = out.reshape(out.shape[:2] + (-1,))
        return self.aggregate_heads(out)


class LocoformerBlock(nn.Module):
    """Eqs. (2)-(4) along ``seq2`` of a ``(B, seq1, seq2, C)`` tensor.

    ``ffn_hidden_dim`` of length 2 gives the Macaron form (FFN, MHSA, FFN);
    length 1 gives MHSA, FFN. ``ffn[0]`` is the FFN after attention and
    ``ffn[-1]`` the one before it (upstream's ordering; it fixes the keys).
    """

    def __init__(
        self,
        rope: RotaryEmbedding,
        emb_dim: int,
        num_groups: int,
        n_heads: int,
        attention_dim: int,
        ffn_hidden_dim: Sequence[int],
        conv1d_kernel: int,
        conv1d_shift: int,
        dropout: float,
        eps: float,
    ):
        super().__init__()
        if len(ffn_hidden_dim) not in (1, 2):
            raise ValueError(f"ffn_hidden_dim needs 1 or 2 entries, got {list(ffn_hidden_dim)}")
        self.macaron_style = len(ffn_hidden_dim) == 2

        self.ffn_norm = nn.ModuleList()
        self.ffn = nn.ModuleList()
        for f_dim in list(ffn_hidden_dim)[::-1]:
            self.ffn_norm.append(RMSGroupNorm(num_groups, emb_dim, eps=eps))
            self.ffn.append(
                SwiGLUConvDeconv1d(emb_dim, f_dim, conv1d_kernel, conv1d_shift, dropout=dropout)
            )

        self.attn_norm = RMSGroupNorm(num_groups, emb_dim, eps=eps)
        self.attn = MultiHeadSelfAttention(emb_dim, attention_dim, n_heads, dropout, rope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s1, s2, c = x.shape

        if self.macaron_style:
            # No 1/2 factor: upstream code and checkpoints (see module docstring).
            x = self.ffn[-1](self.ffn_norm[-1](x)) + x

        out = self.attn_norm(x).reshape([b * s1, s2, c])
        x = self.attn(out).reshape([b, s1, s2, c]) + x

        return self.ffn[0](self.ffn_norm[0](x)) + x


class TFLocoformerBlock(nn.Module):
    """One dual-path block: frequency modelling, then temporal modelling (``tf_order="ft"``)."""

    def __init__(self, rope_freq: RotaryEmbedding, rope_time: RotaryEmbedding, **kwargs):
        super().__init__()
        self.freq_path = LocoformerBlock(rope_freq, **kwargs)
        self.frame_path = LocoformerBlock(rope_time, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, C, T, F)`` -> ``(B, C, T, F)``."""
        out = x.movedim(1, -1)  # (B, T, F, C): attention over F
        out = self.freq_path(out)
        out = out.transpose(1, 2)  # (B, F, T, C): attention over T
        out = self.frame_path(out)
        return out.transpose(-1, 1)


class TFLocoformerSeparator(nn.Module):
    """TF-Locoformer on a complex STFT ``(B, T, F)`` -> one complex estimate ``(B, T, F)``.

    Module tree and parameter names match upstream's ``TFLocoformerSeparator``
    with ``num_spk=1``, so upstream state dicts load unchanged.
    """

    def __init__(
        self,
        n_layers: int = 6,
        emb_dim: int = 128,
        num_groups: int = 4,
        n_heads: int = 4,
        attention_dim: int = 128,
        ffn_hidden_dim: Sequence[int] = (384, 384),
        conv1d_kernel: int = 4,
        conv1d_shift: int = 1,
        dropout: float = 0.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        if attention_dim % n_heads != 0:
            raise ValueError(
                f"attention_dim ({attention_dim}) must be divisible by n_heads ({n_heads})"
            )

        ks, padding = (3, 3), (1, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),  # gLN
        )

        rope_freq = RotaryEmbedding(attention_dim // n_heads)
        rope_time = RotaryEmbedding(attention_dim // n_heads)
        self.blocks = nn.ModuleList(
            TFLocoformerBlock(
                rope_freq,
                rope_time,
                emb_dim=emb_dim,
                num_groups=num_groups,
                n_heads=n_heads,
                attention_dim=attention_dim,
                ffn_hidden_dim=tuple(ffn_hidden_dim),
                conv1d_kernel=conv1d_kernel,
                conv1d_shift=conv1d_shift,
                dropout=dropout,
                eps=eps,
            )
            for _ in range(n_layers)
        )

        self.deconv = nn.ConvTranspose2d(emb_dim, 2, ks, padding=padding)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        x = torch.stack((spec.real, spec.imag), dim=1)  # (B, 2, T, F)
        with torch.autocast(x.device.type, enabled=False):
            x = self.conv(x)
        for block in self.blocks:
            x = block(x)
        with torch.autocast(x.device.type, enabled=False):
            x = self.deconv(x)  # (B, 2, T, F)
        return torch.complex(x[:, 0], x[:, 1])


class TFLocoformer(nn.Module):
    """Single-source TF-Locoformer speech enhancer, waveform in, waveform out.

    ``forward(x: (B, 1, T) or (B, T)) -> (B, 1, T)`` at 16 kHz: std
    normalisation, STFT, :class:`TFLocoformerSeparator`, iSTFT, std restore.
    """

    window: torch.Tensor

    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 128,
        normalize_variance: bool = True,
        n_layers: int = 6,
        emb_dim: int = 128,
        num_groups: int = 4,
        n_heads: int = 4,
        attention_dim: int = 128,
        ffn_hidden_dim: Sequence[int] = (384, 384),
        conv1d_kernel: int = 4,
        conv1d_shift: int = 1,
        dropout: float = 0.0,
        eps: float = 1e-5,
        espnet_xavier_init: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.normalize_variance = normalize_variance
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.separator = TFLocoformerSeparator(
            n_layers=n_layers,
            emb_dim=emb_dim,
            num_groups=num_groups,
            n_heads=n_heads,
            attention_dim=attention_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            conv1d_kernel=conv1d_kernel,
            conv1d_shift=conv1d_shift,
            dropout=dropout,
            eps=eps,
        )
        if espnet_xavier_init:
            self._espnet_xavier_init()

    def _espnet_xavier_init(self) -> None:
        """``espnet2.torch_utils.initialize.initialize(model, "xavier_uniform")``."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p.data)
        for name, p in self.named_parameters():
            if ".bias" in name and p.dim() == 1:
                p.data.zero_()
        for m in self.modules():
            if isinstance(m, (nn.Embedding, nn.LayerNorm, nn.GroupNorm)):
                m.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            if x.shape[1] != 1:
                raise ValueError(f"TFLocoformer is mono; got {x.shape[1]} channels")
            x = x[:, 0]
        n_samples = x.shape[-1]

        with torch.autocast(x.device.type, enabled=False):
            x = x.float()
            if self.normalize_variance:
                std = x.std(dim=-1, keepdim=True)
                x = x / std
            spec = torch.stft(
                x,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window,
                center=True,
                normalized=False,
                onesided=True,
                return_complex=True,
            ).transpose(1, 2)  # (B, T, F)

        est = self.separator(spec)

        with torch.autocast(x.device.type, enabled=False):
            wav = torch.istft(
                est.transpose(1, 2),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window,
                center=True,
                normalized=False,
                onesided=True,
                length=n_samples,
            )
            if self.normalize_variance:
                wav = wav * std
        return wav.unsqueeze(1)


def build_tflocoformer(checkpoint: str | None = None, **params) -> TFLocoformer:
    """Factory for the Hydra ``_target_`` path — plain kwargs, see ``conf/model/tflocoformer.yaml``.

    ``checkpoint``: optional path to an upstream ESPnet state dict (keys
    ``separator.*``, e.g. the DNS ``valid.loss.ave_5best.pth``), loaded with
    ``strict=True``.
    """
    model = TFLocoformer(**params)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    return model
