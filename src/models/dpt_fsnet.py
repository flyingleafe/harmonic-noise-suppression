"""DPT-FSNet speech-enhancement model, re-implemented from the paper.

Reference
---------
Feng Dang, Hangting Chen, Pengyuan Zhang, "DPT-FSNet: Dual-path Transformer
Based Full-band and Sub-band Fusion Network for Speech Enhancement",
ICASSP 2022, arXiv:2104.13002 (v2, 25 Jan 2022).

Upstream code: https://github.com/dangf15/DPT-FSNet @ ``10cbda3bf65b292ad0878a9c59fa8a4f1845f907``
(``model.py`` + ``conv_stft.py``). The repository has **no licence**, so no
code is copied here: this module is written from the paper, and the upstream
code served only as a numerical reference from a scratch clone (the weight
key mapping and the parity check live in ``tests/models/test_dpt_fsnet.py``,
``slow``, skipped when the clone is absent). No pretrained weights are
published.

Native rate is 16 kHz (paper §4.1: "All the utterances are downsampled to
16kHz"), so no resampling wrapper is needed.

Architecture (paper statement → code)
-------------------------------------
* §3.1 / §4.2 "STFT ... window length and frame shift ... 25ms and 6.25ms,
  FFT length 512" → :class:`_STFT` with ``win_length=400``,
  ``hop_length=100``, ``n_fft=512``. Encoder input "complex spectrum
  X ∈ R^{2×T×F}" → real/imag stacked as 2 channels.
* §3.1 "The encoder consists of a 1x1 convolutional layer and a dilated-dense
  block" + §4.2 "All convolutional layers in the encoder and the decoder are
  followed by layer normalization and parametric ReLU" → :class:`_Encoder`
  (``Conv2d(2, C, 1)`` → LayerNorm → PReLU(C) → :class:`_DenseBlock`).
* §4.2 "C ... set to 64" → ``channels=64``.
* §4.2 "dense block consists of four dilated convolutional layers ... input
  channels ... C, 2C, 3C, 4C, and the output after each convolution has C
  channels"; Fig. 1(b) "2×3 Conv", "Concat" → :class:`_DenseBlock`: layer
  ``i`` sees the concatenation of all previous outputs and the block input
  (``C·(i+1)`` channels), kernel (time 2, freq 3), LayerNorm, PReLU(C).
* §3.2 "Before the DPTs, we use a 1x1 convolutional layer to halve the
  channel dimension" + "Convolutional layers in the DPTPM are followed by
  parametric ReLU" → :class:`_DPTPM` ``in_conv`` (C → C/2) + ``in_act``.
* §3.2 "B dual-path transformers", §4.2 "B = 4 and h = 4" → ``n_blocks=4``,
  ``n_heads=4``; transformer width d = C/2 = 32 (= d_model of Eq. 12).
* §3.2 Eq. 7 intra-transformer "acts on the second dimension of D", i.e. a
  sequence over the T time steps of each sub-band → ``_DualPathBlock.intra``
  over time, batch over ``B·F``. Eq. 8 inter-transformer over the F bins at
  each time step (full band) → ``_DualPathBlock.inter``. "handles time and
  frequency paths alternatively" → intra then inter, sequentially.
  Fig. 1(c): each path = transformer → Group Norm → residual add.
* §2.1 Eq. 1–4 multi-head self-attention, ``Mid = LayerNorm(Z + MultiHead)``
  → :class:`_ImprovedTransformer` ``attn`` + ``attn_norm`` (post-LN, no
  positional encoding, §2).
* §2.2 Eq. 5–6 ``FFN(Mid) = ReLU(GRU(Mid)) W1 + b1``, ``W1 ∈ R^{dff×d}``,
  ``dff = 4d``, ``Output = LayerNorm(Mid + FFN)`` → ``gru`` (output width
  ``4d``) → ReLU → ``ffn_out`` Linear(4d, d) → residual → ``ffn_norm``.
* §3.2 "passed through a 1x1 convolutional layer to double the channel
  dimension ... and then through a gated convolutional layer" → ``out_act``
  + ``out_conv`` (C/2 → C), then ``tanh(gate_tanh(·)) * sigmoid(gate_sigmoid(·))``.
* §3.3 "The decoder consists of a 1x1 convolutional layer and a dilated-dense
  block, where the dilated-dense block is the same as in the encoder ... to
  obtain the estimated complex ratio mask"; Fig. 1(a) orders it dense block
  → 1x1 conv → :class:`_Decoder` (``_DenseBlock`` → ``Conv2d(C, 2, 1)``).
  "element-wise multiplication between encoder's input and the mask" →
  complex multiplication in :meth:`DPTFSNet.forward`, then ISTFT.

Settings the paper leaves unstated (value taken from the upstream code)
-----------------------------------------------------------------------
* **Bidirectional GRU, hidden 2d.** The paper fixes only the FFN width
  ``dff = 4d``; a bidirectional GRU with hidden size ``2d`` yields exactly
  that width (upstream: ``GRU(d, 2d, bidirectional=True)``).
* **Dense-block dilation.** The paper says "dilated rate d = 2"; upstream
  dilates the time axis by ``2**i`` (1, 2, 4, 8) and leaves frequency
  undilated. Padding is causal in time (``dilation`` zeros in front) and
  ``1`` on each side in frequency, so T and F are preserved.
* **DC bin dropped.** The network sees bins 1..256 (``F = n_fft // 2``); the
  DC bin of the enhanced spectrum is zero. The LayerNorms normalise over this
  F = 256 axis (``nn.LayerNorm(F)`` on ``[B, C, T, F]``), so ``n_fft`` fixes
  their size.
* **Group Norm** in the dual path: one group, ``eps = 1e-8``.
* **PReLU placement in the DPTPM.** Upstream puts the doubling conv *after*
  a single-parameter PReLU (``PReLU → Conv``) and the halving conv before one
  (``Conv → PReLU``); both PReLUs share one slope across channels. Kept.
* **Mask conv is linear.** The final ``Conv2d(C, 2, 1)`` has no norm and no
  activation (an unbounded complex ratio mask), despite §4.2's "all
  convolutional layers ... are followed by layer normalization and PReLU".
* **STFT framing.** Periodic Hann analysis window of ``win_length`` samples
  placed at the start of each ``n_fft`` frame (zero-padded at the end); the
  signal is zero-padded by ``win_length - hop_length`` on both sides. The
  inverse is the least-squares (pseudo-inverse) synthesis of that truncated
  real DFT, windowed, overlap-added and divided by the overlap-added squared
  window (+1e-8), then cropped by ``win_length - hop_length`` on both sides.
  This reproduces upstream's ``ConvSTFT``/``ConviSTFT`` numerically.
* **No input normalisation** (upstream has none). The mask multiplies the
  input spectrum, so the output scales with the input.
* **Dropout 0** everywhere (upstream default).

Deviations
----------
* **Upstream defect not reproduced: magnitude/phase fed as real/imag.** The
  published ``model.py`` builds its STFT with ``feature_type='real'``, which
  returns ``(magnitude, phase)``, and then treats those as ``(real, imag)``
  for both the network input and the complex mask product before a complex
  ISTFT. The paper (§3.1, §3.3) specifies the complex spectrum and a complex
  ratio mask; this port follows the paper. The parity check patches upstream
  to emit real/imag accordingly.
* **STFT settings follow the paper (400/100/512), not the upstream default**
  ``Net(L=512)`` (window 512, hop 256, FFT 512). Both are configurable; the
  parity check covers both.
* **Arbitrary length.** Upstream returns ``T - (T mod hop)`` samples. Here
  the input is zero-padded on the right to a multiple of ``hop_length`` and
  the output is cropped back to ``T``; for ``T`` a multiple of the hop the
  computation is identical.
* ``forward`` accepts ``(B, T)`` or ``(B, 1, T)`` and returns ``(B, 1, T)``.

Size
----
879,460 parameters with the default config (paper Table 1: 0.88 M).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DPTFSNet", "build_dpt_fsnet"]


class _STFT(nn.Module):
    """Start-aligned windowed real DFT and its least-squares inverse.

    Analysis: pad ``win - hop`` zeros on both sides, cut frames of ``win``
    samples every ``hop``, multiply by a periodic Hann window, real DFT of
    length ``n_fft`` (frame zero-padded at the end).

    Synthesis: per frame, the least-squares ``win``-sample signal whose
    truncated real DFT best matches the given ``[re; im]`` vector (pseudo-
    inverse of the ``(2F, win)`` analysis basis), windowed, overlap-added,
    divided by the overlap-added squared window, cropped by ``win - hop``.
    """

    window: torch.Tensor
    synth: torch.Tensor

    def __init__(self, n_fft: int, hop_length: int, win_length: int) -> None:
        super().__init__()
        if win_length > n_fft:
            raise ValueError(f"win_length {win_length} > n_fft {n_fft}")
        if win_length < 2 * hop_length:
            raise ValueError("win_length must be at least twice hop_length")
        self.n_fft = n_fft
        self.hop = hop_length
        self.win = win_length
        window = torch.hann_window(win_length, periodic=True, dtype=torch.float64)
        n = torch.arange(win_length, dtype=torch.float64)
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)
        phase = 2.0 * torch.pi * k[:, None] * n[None, :] / n_fft
        basis = torch.cat([torch.cos(phase), -torch.sin(phase)], dim=0)  # (2F, win)
        synth = torch.linalg.pinv(basis) * window[:, None]  # (win, 2F)
        self.register_buffer("window", window.float(), persistent=False)
        self.register_buffer("synth", synth.float(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T)`` → complex ``(B, frames, n_fft // 2 + 1)``."""
        pad = self.win - self.hop
        frames = F.pad(x, (pad, pad)).unfold(-1, self.win, self.hop)
        return torch.fft.rfft(frames * self.window, n=self.n_fft)

    def inverse(self, spec: torch.Tensor) -> torch.Tensor:
        """Complex ``(B, frames, n_fft // 2 + 1)`` → ``(B, (frames + 1) * hop - win)``."""
        n_frames = spec.shape[1]
        coeffs = torch.cat([spec.real, spec.imag], dim=-1)  # (B, frames, 2F)
        frames = coeffs @ self.synth.T  # (B, frames, win)
        out_len = (n_frames - 1) * self.hop + self.win
        fold = dict(output_size=(1, out_len), kernel_size=(1, self.win), stride=(1, self.hop))
        signal = F.fold(frames.transpose(1, 2), **fold)[:, 0, 0]
        wsq = (self.window**2)[None, :, None].expand(1, self.win, n_frames)
        norm = F.fold(wsq, **fold)[:, 0, 0]
        signal = signal / (norm + 1e-8)
        pad = self.win - self.hop
        return signal[:, pad : out_len - pad]


class _DenseLayer(nn.Module):
    """Causal-in-time dilated 2x3 conv → LayerNorm over F → PReLU(C)."""

    def __init__(self, in_channels: int, channels: int, n_freqs: int, dilation: int) -> None:
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv2d(in_channels, channels, kernel_size=(2, 3), dilation=(dilation, 1))
        self.norm = nn.LayerNorm(n_freqs)
        self.act = nn.PReLU(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (1, 1, self.dilation, 0))
        return self.act(self.norm(self.conv(x)))


class _DenseBlock(nn.Module):
    """Dilated dense block: layer ``i`` sees ``[out_{i-1}, ..., out_0, input]``
    (``C·(i+1)`` channels), dilation ``2**i`` in time; returns the last output."""

    def __init__(self, channels: int, n_freqs: int, depth: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            _DenseLayer(channels * (i + 1), channels, n_freqs, 2**i) for i in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = x
        out = x
        for layer in self.layers:
            out = layer(feats)
            feats = torch.cat([out, feats], dim=1)
        return out


class _Encoder(nn.Module):
    def __init__(self, channels: int, n_freqs: int, depth: int) -> None:
        super().__init__()
        self.in_conv = nn.Conv2d(2, channels, kernel_size=1)
        self.in_norm = nn.LayerNorm(n_freqs)
        self.in_act = nn.PReLU(channels)
        self.dense = _DenseBlock(channels, n_freqs, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(self.in_act(self.in_norm(self.in_conv(x))))


class _ImprovedTransformer(nn.Module):
    """Post-LN self-attention + GRU feed-forward (paper §2, Eq. 1–6)."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.gru = nn.GRU(d_model, 2 * d_model, batch_first=True, bidirectional=True)
        self.ffn_out = nn.Linear(4 * d_model, d_model)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``z``: ``(N, L, d)`` → ``(N, L, d)``."""
        mid = self.attn_norm(z + self.attn(z, z, z, need_weights=False)[0])
        hidden, _ = self.gru(mid)
        return self.ffn_norm(mid + self.ffn_out(F.relu(hidden)))


class _DualPathBlock(nn.Module):
    """Intra (sub-band: over time per bin) then inter (full-band: over bins
    per frame), each followed by GroupNorm and a residual add."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.intra = _ImprovedTransformer(d_model, n_heads)
        self.intra_norm = nn.GroupNorm(1, d_model, eps=1e-8)
        self.inter = _ImprovedTransformer(d_model, n_heads)
        self.inter_norm = nn.GroupNorm(1, d_model, eps=1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, f = x.shape
        seq = x.permute(0, 3, 2, 1).reshape(b * f, t, c)
        y = self.intra(seq).reshape(b, f, t, c).permute(0, 3, 2, 1)
        x = x + self.intra_norm(y)
        seq = x.permute(0, 2, 3, 1).reshape(b * t, f, c)
        y = self.inter(seq).reshape(b, t, f, c).permute(0, 3, 1, 2)
        return x + self.inter_norm(y)


class _DPTPM(nn.Module):
    """1x1 conv (C → C/2) + PReLU, ``n_blocks`` dual-path blocks, PReLU + 1x1
    conv (C/2 → C), gated 1x1 conv ``tanh(·) * sigmoid(·)``."""

    def __init__(self, channels: int, n_blocks: int, n_heads: int) -> None:
        super().__init__()
        half = channels // 2
        self.in_conv = nn.Conv2d(channels, half, kernel_size=1)
        self.in_act = nn.PReLU()
        self.blocks = nn.ModuleList(_DualPathBlock(half, n_heads) for _ in range(n_blocks))
        self.out_act = nn.PReLU()
        self.out_conv = nn.Conv2d(half, channels, kernel_size=1)
        self.gate_tanh = nn.Conv2d(channels, channels, kernel_size=1)
        self.gate_sigmoid = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_act(self.in_conv(x))
        for block in self.blocks:
            x = block(x)
        x = self.out_conv(self.out_act(x))
        return torch.tanh(self.gate_tanh(x)) * torch.sigmoid(self.gate_sigmoid(x))


class _Decoder(nn.Module):
    def __init__(self, channels: int, n_freqs: int, depth: int) -> None:
        super().__init__()
        self.dense = _DenseBlock(channels, n_freqs, depth)
        self.mask_conv = nn.Conv2d(channels, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mask_conv(self.dense(x))


class DPTFSNet(nn.Module):
    """DPT-FSNet: ``(B, 1, T)`` or ``(B, T)`` noisy 16 kHz waveform →
    ``(B, 1, T)`` enhanced waveform via a complex ratio mask."""

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 100,
        win_length: int = 400,
        channels: int = 64,
        dense_depth: int = 4,
        n_blocks: int = 4,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        if channels % 2 or (channels // 2) % n_heads:
            raise ValueError(f"channels={channels} must be even and C/2 divisible by n_heads")
        self.stft = _STFT(n_fft, hop_length, win_length)
        n_freqs = n_fft // 2  # DC bin dropped
        self.encoder = _Encoder(channels, n_freqs, dense_depth)
        self.dptpm = _DPTPM(channels, n_blocks, n_heads)
        self.decoder = _Decoder(channels, n_freqs, dense_depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            if x.shape[1] != 1:
                raise ValueError(f"DPT-FSNet is mono; got {x.shape[1]} channels")
            x = x[:, 0]
        n_samples = x.shape[-1]
        hop = self.stft.hop
        x = F.pad(x, (0, -n_samples % hop))

        spec = self.stft(x)[..., 1:]  # (B, T', F) complex, DC dropped
        feats = torch.stack([spec.real, spec.imag], dim=1)  # (B, 2, T', F)
        mask = self.decoder(self.dptpm(self.encoder(feats)))
        mask = torch.complex(mask[:, 0], mask[:, 1])
        est = F.pad(spec * mask, (1, 0))  # zero DC bin back in
        return self.stft.inverse(est)[:, None, :n_samples]


def build_dpt_fsnet(**params: Any) -> DPTFSNet:
    """Factory for ``conf/model/dpt_fsnet.yaml`` (``_target_`` path)."""
    return DPTFSNet(**params)
