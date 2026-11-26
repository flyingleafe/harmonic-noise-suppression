"""CRNN and ULCNet of the cMPDR + DNN paper, ported from Keras to torch.

Source: https://github.com/narrietal/cMPDR_DNN at
753c04cbe986c84315ad4aef3f251ad69b2f606f, ``src/networks/CRNN.py`` and
``src/networks/ULCNet.py`` with ``src/networks/network_config.yml`` (MIT,
Copyright (c) 2026 Nicolas Arrieta Larraza, Giovanni Bologni). The complex
ratio mask is ``CRM_tensorflow`` 0.1.6 (https://github.com/narrietal/CRM_tensorflow,
MIT) mode ``'E'``; the window is ``libsegmenter`` 1.0.8 ``WindowSelector("hann75",
"wola", 512)`` (MIT, Niels de Koeijer, Martin Bo Moller). Paper: Bologni, Arrieta
Larraza, Heusdens, Hendriks, EUSIPCO 2026, arXiv:2602.12986; ULCNet itself is
Shetu et al., "Ultra low complexity deep learning based noise suppression",
ICASSP 2024 (arXiv:2312.08132), as re-implemented by the cMPDR_DNN authors.

Shared signal path (both networks, paper Sec. III-C: 32 ms window, 8 ms hop,
512-point FFT): ``tf.signal.frame(x, 512, 128)`` (no padding) times the
analysis window ``sqrt(hann_periodic(512) / 2)`` (float32, Hann normalised by
its COLA constant, square root for WOLA), ``rfft`` -> 257 bins; synthesis is
``irfft`` times the same window, ``tf.signal.overlap_and_add(., 128)``.
No input normalisation anywhere (upstream has none).

CRNN (``CRNNModel.build_model``): |X| (B, L, 257, 1) -> Conv2D(8, 3x3, same,
no bias) - BN - ReLU -> Conv2D(4, 3x3) - BN - ReLU - MaxPool(1, 2) ->
Conv2D(4, 3x3) - BN - ReLU - MaxPool(1, 2) -> reshape (B, L, 64 * 4) ->
GRU(128) -> Dropout(0.25) -> Dense(257, ReLU) = real mask; Y = mask * X;
irfft + overlap-add. 181,945 parameters (counted on the Keras model, BN
moving statistics included, as Keras reports them).

ULCNet (``ULCNet.build_model``): power-law compression (0.3) of Re and Im;
compressed magnitude and phase; channel-wise feature reorientation (bands of
48 bins, hop ceil(48 * 0.67) = 33, 8 bands, the last zero-padded); four
SeparableConv2D (1x3) layers 32/64/96/128 with ReLU, max-pool (1, 2) after the
2nd, 3rd and 4th (48 -> 6 frequency positions); bidirectional GRU(64) along
frequency; 1x1 Conv2D(64) + ReLU; reshape to 384 and split into two
192-wide sub-bands, each through GRU(128) -> GRU(128); concat (256) ->
Dense(257, ReLU) -> Dense(257, ReLU) = magnitude mask; intermediate features
mask * (cos, sin)(compressed phase); Conv2D(32, 1x3) + ReLU, Conv2D(32, 1x3) +
ReLU, Conv2D(2, 1x1) + ReLU = complex mask; CRM 'E' (``tanh(|M|) |Xc|
exp(j(angle Xc + angle M))``, ``M_re + 1e-8``) on the compressed spectrum;
decompression ``sign(.) |.|^(1/0.3)``. 685,147 parameters.

Keras conventions kept: BatchNorm eps 1e-3 / momentum 0.99 (torch 0.01);
GRU with ``reset_after=True`` (gate math identical to ``torch.nn.GRU``; the
Keras kernels map to torch by gate reordering z, r, h -> r, z, n); frequency
is flattened major, channel minor, as ``tf.reshape`` of channels-last tensors.

Deviations
----------
* Length: ``tf.signal.frame`` drops the tail that does not fill a frame and
  ``overlap_and_add`` returns ``(L - 1) * 128 + 512`` samples. Here the input
  is zero-padded at the end to whole frames and the output is cropped to T,
  so ``forward`` keeps the length. Identical to upstream whenever
  ``(T - 512) % 128 == 0``. The first and last 384 samples keep upstream's
  incomplete WOLA overlap (taper), as in upstream.
* CRNN rebuilds ``|X| * mask * exp(j angle X)`` as ``mask * X`` (same value up
  to rounding, defined gradient at X = 0).
* ULCNet: upstream returns the estimated complex spectrogram (its training loss
  is in the time-frequency domain); ``forward`` adds the WOLA synthesis
  (irfft, window, overlap-add) that the CRNN uses, to return a waveform. The
  spectrogram is available from :meth:`ULCNet.spectrum`.
* ULCNet compression: ``|x|^0.3`` has an infinite derivative at 0; no gradient
  flows into the input features (they come from the non-learned front end),
  so nothing changes there.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def libsegmenter_hann75_wola(block_len: int) -> np.ndarray:
    """``libsegmenter.WindowSelector("hann75", "wola", block_len).analysis_window``.

    Same float32 arithmetic as libsegmenter 1.0.8 (``_hann``, ``check_cola``,
    ``_adapt_window``).
    """
    if block_len % 4:
        raise ValueError(f"block_len must be a multiple of 4, got {block_len}")
    hop = block_len // 4
    idx = np.arange(block_len, dtype=np.float32)
    w = 0.5 * (1.0 - np.cos(2.0 * np.pi * idx / block_len, dtype=np.float32))
    dc_value = float(np.sum(w, dtype=np.float32)) / hop
    upper = lower = dc_value
    for k in range(1, hop):
        csin = np.exp(1j * 2.0 * np.pi * (k / hop) * np.arange(w.size))
        c = np.sum(w * np.conjugate(csin))
        upper += np.abs(c) / hop
        lower -= np.abs(c) / hop
    norm = (upper + lower) / 2.0
    return np.sqrt(np.divide(w, np.float32(norm))).astype(np.float32)


class WolaSTFT(nn.Module):
    """Upstream ``STFTLayer`` / ``OverlapAddLayer`` pair, length-preserving."""

    window: Tensor

    def __init__(self, block_len: int = 512, block_shift: int = 128) -> None:
        super().__init__()
        self.block_len = int(block_len)
        self.block_shift = int(block_shift)
        win = torch.from_numpy(libsegmenter_hann75_wola(self.block_len))
        self.register_buffer("window", win, persistent=False)

    def stft(self, x: Tensor) -> Tensor:
        """``(B, T)`` -> ``(B, L, block_len // 2 + 1)`` complex."""
        n = x.shape[-1]
        n_frames = max(1, math.ceil((n - self.block_len) / self.block_shift) + 1)
        pad = (n_frames - 1) * self.block_shift + self.block_len - n
        frames = F.pad(x, (0, pad)).unfold(-1, self.block_len, self.block_shift)
        return torch.fft.rfft(frames * self.window, dim=-1)

    def istft(self, spec: Tensor, length: int) -> Tensor:
        """``(B, L, F)`` complex -> ``(B, length)`` (irfft, window, overlap-add)."""
        b, n_frames, _ = spec.shape
        frames = torch.fft.irfft(spec, n=self.block_len, dim=-1) * self.window
        total = (n_frames - 1) * self.block_shift + self.block_len
        out = F.fold(
            frames.transpose(1, 2),
            output_size=(1, total),
            kernel_size=(1, self.block_len),
            stride=(1, self.block_shift),
        ).reshape(b, total)
        return out[:, :length]


def _keras_bn(channels: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)


class CRNN(nn.Module):
    """Magnitude-mask CRNN; ``forward((B, T)) -> (B, T)``."""

    def __init__(
        self,
        block_len: int = 512,
        block_shift: int = 128,
        rnn_units: int = 128,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.stft = WolaSTFT(block_len, block_shift)
        n_freq = block_len // 2 + 1
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1, bias=False)
        self.bn1 = _keras_bn(8)
        self.conv2 = nn.Conv2d(8, 4, 3, padding=1, bias=False)
        self.bn2 = _keras_bn(4)
        self.conv3 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.bn3 = _keras_bn(4)
        self.gru = nn.GRU((n_freq // 2 // 2) * 4, rnn_units, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(rnn_units, n_freq)

    def mask(self, mag: Tensor) -> Tensor:
        """``(B, L, F)`` magnitude -> ``(B, L, F)`` real mask."""
        b, n_frames, _ = mag.shape
        h = mag[:, None]  # (B, 1, L, F): time = H, freq = W
        h = F.relu(self.bn1(self.conv1(h)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.max_pool2d(h, (1, 2))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.max_pool2d(h, (1, 2))
        h = h.permute(0, 2, 3, 1).reshape(b, n_frames, -1)  # freq major, channel minor
        h, _ = self.gru(h)
        return F.relu(self.dense(self.dropout(h)))

    def forward(self, x: Tensor) -> Tensor:
        spec = self.stft.stft(x)
        est = spec * self.mask(spec.abs())
        return self.stft.istft(est, x.shape[-1])


class SeparableConv2d(nn.Module):
    """Keras ``SeparableConv2D`` (depth multiplier 1, no bias, 'same')."""

    def __init__(self, in_ch: int, out_ch: int, kernel: tuple[int, int]) -> None:
        super().__init__()
        pad = (kernel[0] // 2, kernel[1] // 2)
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel, padding=pad, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class ULCNet(nn.Module):
    """ULCNet as re-implemented in cMPDR_DNN; ``forward((B, T)) -> (B, T)``."""

    def __init__(
        self,
        block_len: int = 512,
        block_shift: int = 128,
        bidirectional_frnn_units: int = 64,
        sub_band_rnn_units: int = 128,
        compression_factor: float = 0.3,
        crm_eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.stft = WolaSTFT(block_len, block_shift)
        self.n_freq = block_len // 2 + 1
        self.compression_factor = float(compression_factor)
        self.crm_eps = float(crm_eps)
        # ChannelWiseFeatureReorientation
        self.band_width = 48
        self.band_hop = math.ceil(self.band_width * (1 - 0.33))
        self.n_bands = math.ceil((self.n_freq - self.band_width) / self.band_hop + 1)
        self.sep1 = SeparableConv2d(self.n_bands, 32, (1, 3))
        self.sep2 = SeparableConv2d(32, 64, (1, 3))
        self.sep3 = SeparableConv2d(64, 96, (1, 3))
        self.sep4 = SeparableConv2d(96, 128, (1, 3))
        self.freq_gru = nn.GRU(128, bidirectional_frnn_units, batch_first=True, bidirectional=True)
        self.pointwise = nn.Conv2d(2 * bidirectional_frnn_units, 64, 1, bias=False)
        n_pos = self.band_width // 2 // 2 // 2
        sub_in = n_pos * 64 // 2
        self.sub1_gru1 = nn.GRU(sub_in, sub_band_rnn_units, batch_first=True)
        self.sub1_gru2 = nn.GRU(sub_band_rnn_units, sub_band_rnn_units, batch_first=True)
        self.sub2_gru1 = nn.GRU(sub_in, sub_band_rnn_units, batch_first=True)
        self.sub2_gru2 = nn.GRU(sub_band_rnn_units, sub_band_rnn_units, batch_first=True)
        self.fc1 = nn.Linear(2 * sub_band_rnn_units, self.n_freq)
        self.fc2 = nn.Linear(self.n_freq, self.n_freq)
        self.cnn1 = nn.Conv2d(2, 32, (1, 3), padding=(0, 1), bias=False)
        self.cnn2 = nn.Conv2d(32, 32, (1, 3), padding=(0, 1), bias=False)
        self.mask_conv = nn.Conv2d(32, 2, 1, bias=False)

    def _reorient(self, mag: Tensor) -> Tensor:
        """``(B, L, F)`` -> ``(B, n_bands, L, 48)`` (bands become channels)."""
        last = (self.n_bands - 1) * self.band_hop + self.band_width
        padded = F.pad(mag, (0, max(0, last - self.n_freq)))
        bands = padded.unfold(-1, self.band_width, self.band_hop)[:, :, : self.n_bands]
        return bands.permute(0, 2, 1, 3)

    def spectrum(self, x: Tensor) -> Tensor:
        """Upstream model output: estimated complex spectrogram ``(B, L, F)``."""
        spec = self.stft.stft(x)
        c = self.compression_factor
        cr = torch.sign(spec.real) * spec.real.abs() ** c
        ci = torch.sign(spec.imag) * spec.imag.abs() ** c
        mag = torch.sqrt(cr**2 + ci**2)
        phase = torch.atan2(ci, cr)
        b, n_frames, _ = mag.shape

        h = F.relu(self.sep1(self._reorient(mag)))
        h = F.relu(self.sep2(h))
        h = F.max_pool2d(h, (1, 2))
        h = F.relu(self.sep3(h))
        h = F.max_pool2d(h, (1, 2))
        h = F.relu(self.sep4(h))
        h = F.max_pool2d(h, (1, 2))  # (B, 128, L, 6)
        n_pos = h.shape[-1]
        h = h.permute(0, 2, 3, 1).reshape(b * n_frames, n_pos, -1)
        h, _ = self.freq_gru(h)
        h = h.reshape(b, n_frames, n_pos, -1).permute(0, 3, 1, 2)
        h = F.relu(self.pointwise(h))  # (B, 64, L, 6)
        h = h.permute(0, 2, 3, 1).reshape(b, n_frames, -1)  # (B, L, 384)
        s1, s2 = h.chunk(2, dim=-1)
        s1, _ = self.sub1_gru2(self.sub1_gru1(s1)[0])
        s2, _ = self.sub2_gru2(self.sub2_gru1(s2)[0])
        h = torch.cat([s1, s2], dim=-1)
        mask = F.relu(self.fc2(F.relu(self.fc1(h))))  # (B, L, F)

        inter = torch.stack([mask * torch.cos(phase), mask * torch.sin(phase)], dim=1)
        h = F.relu(self.cnn1(inter))
        h = F.relu(self.cnn2(h))
        cm = F.relu(self.mask_conv(h))  # (B, 2, L, F)
        m_re = cm[:, 0] + self.crm_eps
        m_im = cm[:, 1]
        m_mag = torch.tanh(torch.sqrt(m_re**2 + m_im**2))
        m_phase = torch.atan2(m_im, m_re)
        est_mag = m_mag * mag
        est_phase = phase + m_phase
        er = est_mag * torch.cos(est_phase)
        ei = est_mag * torch.sin(est_phase)
        dr = torch.sign(er) * er.abs() ** (1 / c)
        di = torch.sign(ei) * ei.abs() ** (1 / c)
        return torch.complex(dr, di)

    def forward(self, x: Tensor) -> Tensor:
        return self.stft.istft(self.spectrum(x), x.shape[-1])
