"""BSRNN — the URGENT 2026 discriminative speech-enhancement baseline, 16 kHz.

Reference model
    Band-split RNN for speech enhancement: J. Yu, H. Chen, Y. Luo, R. Gu,
    C. Weng, "High Fidelity Speech Enhancement with Band-split RNN",
    Interspeech 2023, arXiv:2212.00406; J. Yu and Y. Luo, "Efficient monaural
    speech enhancement with universal sample rate band-split RNN", ICASSP 2023.

Ported form: the BSRNN baseline of the URGENT 2026 challenge, track 1
    https://github.com/urgent-challenge/urgent2026_challenge_track1
    commit b1dc3ad1e86419ff0bd666f455bda7936bff0e9a, Apache-2.0,
    ``baseline_code/models/bsrnn.py::BSRNN_SE`` with
    ``conf/models/BSRNN_baseline.yaml`` (``num_channel: 196, num_layer: 6``).
    That class wraps ESPnet's ``STFTEncoder`` / ``STFTDecoder`` /
    ``BSRNNSeparator``; the network itself is ESPnet's
    ``espnet2/enh/layers/bsrnn.py`` (tag ``v.202412``, commit cccc2902,
    Apache-2.0, itself ported from github.com/sungwon23/BSRNN). The code
    below is adapted from those two Apache-2.0 sources with attribution.

Architecture (ESPnet ``BSRNN``, all traced to the classes below)
    * STFT (``BSRNNSE``): Hann window (``torch.hann_window``, periodic),
      ``center=True`` with reflect padding, not normalised, one-sided. The
      challenge model is defined at 48 kHz with ``n_fft = win = 960`` and
      ``hop = 480`` (20 ms / 10 ms) and rescales both to the input rate
      (``n_fft * fs // 48000``), so at 16 kHz: ``n_fft = win = 320``,
      ``hop = 160``, F = 161 bins of 50 Hz. iSTFT with the same window,
      ``length = T``. No input or output normalisation (upstream has none).
    * Band split (``BandSplit``): the 48 kHz table
      ``[5] + [4]*19 + [10]*6 + [40]*7 + [60]`` bins (20 x 200 Hz up to
      4 kHz, 6 x 500 Hz up to 7 kHz, 7 x 2 kHz, one 3 kHz). Per band:
      interleaved (re, im) features ``2*b`` -> ``GroupNorm(1, 2b, eps=1e-8)``
      -> ``Conv1d(2b, N, 1)``.
    * 6 residual layers (``BSRNN``): time path ``GroupNorm(1, N)`` ->
      BLSTM(N -> 2N per direction) over frames, per band -> ``Linear(4N, N)``
      -> residual add; band path ``GroupNorm(1, N)`` -> BLSTM(N -> 2N per
      direction) over bands, per frame -> ``Linear(4N, N)`` -> residual add.
      ``N = 196`` (``num_channel``).
    * Mask decoder (``MaskDecoder``): per band two MLPs (mask and residual),
      each ``GroupNorm(1, N, eps=1e-8)`` -> ``Conv1d(N, 4N, 1)`` -> Tanh ->
      ``Conv1d(4N, 4b, 1)`` -> GLU, giving a complex mask ``m`` and a complex
      residual ``r`` per bin; output spectrum ``m * X + r``.

What 16 kHz keeps and what was dropped
    ESPnet's band loop stops at the first band whose start bin reaches the
    input's F. At 16 kHz (F = 161) that is after 27 bands: the first 26 cover
    bins 0..139 exactly, and band 27 (40 bins, 7-9 kHz at 48 kHz) receives the
    21 real bins 140..160 followed by 19 zeros. The zeros enter that band's
    GroupNorm statistics; this port reproduces them by zero-padding the
    spectrum to ``sum(subbands) = 181`` bins, then discards the decoded bins
    above 160 exactly as upstream (``m[..., :F]``). Dropped: the 7 bands
    above 9 kHz (their 4,171,932 parameters are never touched at 16 kHz),
    the per-call ``fs`` reconfiguration of the STFT, the ``fs`` early-exit
    of the band loop, the ``num_spk`` axis in the SE wrapper (the backbone
    keeps it), the cLN/cfLN/BN norm choices (the baseline uses ``GN``), the
    ``ilens`` padding masks (batches here are equal length), and ESPnet's
    ``ComplexTensor`` / spec-transform plumbing (``spec_transform_type`` is
    ``None`` in the baseline, i.e. identity). Not part of the model and not
    reproduced: upstream ``inference.py`` peak-normalises every enhanced
    file to 0.9 (``enhanced / enhanced.abs().max() * 0.9``) before writing
    it; ``BSRNNSE`` returns the network output at its own scale.

Parameter count
    33,628,912 parameters at 16 kHz (37,800,844 in the full 48 kHz module
    and in the published checkpoint). The upstream config reports
    "Parameters: 32.0456657409668 M" at 16 kHz: that is 33,602,316 / 2**20,
    i.e. its profiler counts in units of 2**20 and omits the 26,596
    GroupNorm affine parameters (724 band-split + 21,168 decoder + 4,704
    layer norms); 33,602,316 + 26,596 = 33,628,912 exactly. The same rule
    reproduces its 48 kHz figure (37,800,844 - 33,284 = 36.01795 * 2**20).

Differences between the URGENT baseline and the Interspeech paper (the port
follows the baseline, not the paper)
    The paper's 16 kHz model uses N = 128, a 32 ms / 8 ms STFT, a 192-unit
    LSTM, uni-directional band modelling above 8 kHz and GAN fine-tuning
    (MGD + MRSD discriminators); the baseline uses N = 196, 2N = 392 LSTM
    units, bi-directional modelling over all bands, a 20 ms / 10 ms STFT and
    only a regression loss.

Upstream training recipe (facts, for reference; not implemented here)
    ``MultiResL1SpecLoss(window_sz=[256, 512, 768, 1024], eps=1e-6,
    normalize_variance=True, time_domain_weight=0.5)``; AdamW, lr 1e-3,
    eps 1e-8, weight decay 1e-6; StepLR step 1 epoch, gamma 0.85; gradient
    clip 0.5; max segment 96000 samples (length-bucketed, multi-rate data);
    no EMA; steps whose loss is NaN are skipped and NaN gradients are zeroed;
    fp32 (Lightning default precision) with
    ``torch.set_float32_matmul_precision("medium")``; DDP. The SI-SNR is
    logged only, not trained on. ``BSRNN_baseline.yaml`` says batch 4,
    80 epochs, 1 GPU; the published checkpoint's pickled config says batch 8
    on 2 GPUs, 150 epochs planned, dynamic mixing, seed 1996, stopped (best
    val) at epoch 55 / step 1,315,000; ``val_check_interval`` 5000 steps.

Pretrained weights
    ``https://huggingface.co/lichenda/icassp_2026_urgent_baseline/resolve/main/bsrnn.ckpt``
    (151 MB, sha256 5d6b24eb...31c8, model card licence MIT, trained on
    "700h-TBF", a mixed-rate subset of the URGENT data, see arXiv:2506.23859).
    ``build_bsrnn(checkpoint=...)`` loads it, keeping the 27 bands used at
    16 kHz; the port reproduces the upstream output on it bit-exactly.

Reuse
    ``BSRNN`` (the backbone) is independent of the waveform wrapper: it maps a
    complex spectrogram ``(B, T, F, 2)`` to ``(B, num_spk, T, F, 2)``
    (``m * X + r``), for any band table and any ``F`` the table covers.
    ``BSRNNSE`` adds only the 16 kHz STFT / iSTFT around it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "SUBBANDS_48K",
    "SUBBANDS_16K",
    "BandSplit",
    "MaskDecoder",
    "BSRNN",
    "BSRNNSE",
    "build_bsrnn",
    "convert_urgent_state_dict",
    "load_urgent_checkpoint",
]

# ESPnet's 48 kHz band table (n_fft = 960, 50 Hz bins at 48 kHz).
SUBBANDS_48K: tuple[int, ...] = tuple([5] + [4] * 19 + [10] * 6 + [40] * 7 + [60])
# The prefix ESPnet's band loop visits for a 161-bin (16 kHz) spectrum.
SUBBANDS_16K: tuple[int, ...] = SUBBANDS_48K[:27]

# Sample rate and STFT of the 16 kHz model (48 kHz 960/480 rescaled by 1/3).
SAMPLE_RATE = 16000
N_FFT = 320
HOP_LENGTH = 160


class BandSplit(nn.Module):
    """Per-band ``GroupNorm`` + 1x1 ``Conv1d`` of the interleaved (re, im) bins."""

    def __init__(self, subbands: Sequence[int], channels: int) -> None:
        super().__init__()
        self.subbands = [int(b) for b in subbands]
        self.norm = nn.ModuleList(nn.GroupNorm(1, 2 * b, eps=1e-8) for b in self.subbands)
        self.fc = nn.ModuleList(nn.Conv1d(2 * b, channels, 1) for b in self.subbands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, sum(subbands), 2) -> (B, N, T, K)."""
        B, T = x.shape[:2]
        outs = []
        for band, norm, fc in zip(torch.split(x, self.subbands, dim=2), self.norm, self.fc):
            feat = band.reshape(B, T, -1).transpose(1, 2)  # (B, 2b, T)
            outs.append(fc(norm(feat)))
        return torch.stack(outs, dim=-1)


class MaskDecoder(nn.Module):
    """Per-band MLPs predicting a complex mask and a complex residual."""

    def __init__(self, subbands: Sequence[int], channels: int, num_spk: int = 1) -> None:
        super().__init__()
        self.subbands = tuple(int(b) for b in subbands)
        self.num_spk = num_spk

        def mlp(b: int) -> nn.Sequential:
            return nn.Sequential(
                nn.GroupNorm(1, channels, eps=1e-8),
                nn.Conv1d(channels, 4 * channels, 1),
                nn.Tanh(),
                nn.Conv1d(4 * channels, 4 * b * num_spk, 1),
                nn.GLU(dim=1),
            )

        self.mlp_mask = nn.ModuleList(mlp(b) for b in self.subbands)
        self.mlp_residual = nn.ModuleList(mlp(b) for b in self.subbands)

    def _decode(self, x: torch.Tensor, mlps: nn.ModuleList) -> torch.Tensor:
        B, _, T, _ = x.shape
        outs = []
        for i, net in enumerate(mlps):
            out = net(x[:, :, :, i]).transpose(1, 2).contiguous()  # (B, T, 2 * b * num_spk)
            outs.append(out.reshape(B, T, self.num_spk, -1, 2))
        return torch.cat(outs, dim=3).moveaxis(1, 2)  # (B, num_spk, T, F, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, N, T, K) -> mask, residual, each (B, num_spk, T, sum(subbands), 2)."""
        return self._decode(x, self.mlp_mask), self._decode(x, self.mlp_residual)


class BSRNN(nn.Module):
    """Band-split RNN backbone on a complex spectrogram (ESPnet ``BSRNN``).

    ``forward(x)`` takes ``x`` of shape ``(B, T, F, 2)`` (frames, bins,
    re/im) with ``sum(subbands) - subbands[-1] < F <= sum(subbands)``; bins
    missing from the last band are zero-filled, as ESPnet's band loop does.
    Returns the enhanced spectrum ``m * X + r`` as ``(B, num_spk, T, F, 2)``.
    State-dict keys match ESPnet's ``BSRNN`` (minus the unused high bands).
    """

    def __init__(
        self,
        subbands: Sequence[int] = SUBBANDS_16K,
        num_channel: int = 196,
        num_layer: int = 6,
        causal: bool = False,
        num_spk: int = 1,
    ) -> None:
        super().__init__()
        self.subbands = tuple(int(b) for b in subbands)
        self.n_freq = sum(self.subbands)
        self.num_layer = num_layer
        self.num_spk = num_spk
        self.band_split = BandSplit(self.subbands, num_channel)

        hdim = 2 * num_channel
        self.norm_time = nn.ModuleList()
        self.rnn_time = nn.ModuleList()
        self.fc_time = nn.ModuleList()
        self.norm_freq = nn.ModuleList()
        self.rnn_freq = nn.ModuleList()
        self.fc_freq = nn.ModuleList()
        for _ in range(num_layer):
            self.norm_time.append(nn.GroupNorm(1, num_channel))
            self.rnn_time.append(
                nn.LSTM(num_channel, hdim, batch_first=True, bidirectional=not causal)
            )
            self.fc_time.append(nn.Linear(hdim if causal else 2 * hdim, num_channel))
            self.norm_freq.append(nn.GroupNorm(1, num_channel))
            self.rnn_freq.append(nn.LSTM(num_channel, hdim, batch_first=True, bidirectional=True))
            self.fc_freq.append(nn.Linear(2 * hdim, num_channel))

        self.mask_decoder = MaskDecoder(self.subbands, num_channel, num_spk=num_spk)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_bins = x.shape[2]
        if not self.n_freq - self.subbands[-1] < n_bins <= self.n_freq:
            raise ValueError(
                f"BSRNN with subbands summing to {self.n_freq} needs "
                f"{self.n_freq - self.subbands[-1]} < F <= {self.n_freq}, got F={n_bins}"
            )
        z = self.band_split(F.pad(x, (0, 0, 0, self.n_freq - n_bins)))
        B, N, T, K = z.shape
        skip = z
        for i in range(self.num_layer):
            out = self.norm_time[i](skip)
            out = out.transpose(1, 3).reshape(B * K, T, N)
            out, _ = self.rnn_time[i](out)
            out = self.fc_time[i](out)
            out = out.reshape(B, K, T, N).transpose(1, 3)
            skip = skip + out

            out = self.norm_freq[i](skip)
            out = out.permute(0, 2, 3, 1).contiguous().reshape(B * T, K, N)
            out, _ = self.rnn_freq[i](out)
            out = self.fc_freq[i](out)
            out = out.reshape(B, T, K, N).permute(0, 3, 1, 2).contiguous()
            skip = skip + out

        m, r = self.mask_decoder(skip)
        m = torch.view_as_complex(m)[..., :n_bins]
        r = torch.view_as_complex(r)[..., :n_bins]
        spec = torch.view_as_complex(x)
        return torch.view_as_real(m * spec.unsqueeze(1) + r)


class BSRNNSE(nn.Module):
    """Waveform-in / waveform-out 16 kHz wrapper: STFT -> ``BSRNN`` -> iSTFT."""

    window: torch.Tensor

    def __init__(self, num_channel: int = 196, num_layer: int = 6, causal: bool = False) -> None:
        super().__init__()
        self.n_fft = N_FFT
        self.hop_length = HOP_LENGTH
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)
        self.bsrnn = BSRNN(
            subbands=SUBBANDS_16K,
            num_channel=num_channel,
            num_layer=num_layer,
            causal=causal,
            num_spk=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, T) or (B, T) mono 16 kHz -> (B, 1, T) enhanced."""
        n = x.shape[-1]
        wav = x.reshape(-1, n)
        # ESPnet runs the STFT pair (not the network) in fp32, autocast off.
        with torch.autocast(device_type=wav.device.type, enabled=False):
            spec = torch.stft(
                wav.float(),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window,
                center=True,
                normalized=False,
                onesided=True,
                return_complex=True,
            )  # (B, F, frames)
        feat = torch.view_as_real(spec.transpose(1, 2))  # (B, frames, F, 2)
        est = self.bsrnn(feat)[:, 0]  # (B, frames, F, 2)
        with torch.autocast(device_type=wav.device.type, enabled=False):
            out = torch.istft(
                torch.view_as_complex(est.float().contiguous()).transpose(1, 2),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window,
                center=True,
                normalized=False,
                onesided=True,
                length=n,
            )
        return out.unsqueeze(1)


class _UrgentConfig:
    """Inert replacement for the checkpoint's unused training configuration."""

    __slots__ = ()

    def __setstate__(self, _state: object) -> None:
        # Ignore the metadata; only the model state dict is used.
        pass


_BAND_KEY = re.compile(r"^(band_split\.(?:norm|fc)|mask_decoder\.mlp_(?:mask|residual))\.(\d+)\.")


def convert_urgent_state_dict(
    state: dict[str, torch.Tensor],
    prefix: str = "se_model.bsrnn.bsrnn.",
    n_bands: int = len(SUBBANDS_16K),
) -> dict[str, torch.Tensor]:
    """Map an upstream state dict onto ``BSRNN`` keys.

    Strips ``prefix`` (``se_model.bsrnn.bsrnn.`` in the Lightning checkpoint,
    ``bsrnn.bsrnn.`` in a bare ``BSRNN_SE``) and drops the band-specific
    tensors of bands ``>= n_bands`` (unused at 16 kHz).
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not key.startswith(prefix):
            raise KeyError(f"unexpected key in URGENT BSRNN state dict: {key}")
        key = key[len(prefix) :]
        match = _BAND_KEY.match(key)
        if match is not None and int(match.group(2)) >= n_bands:
            continue
        out[key] = value
    return out


def load_urgent_checkpoint(path: str, n_bands: int = len(SUBBANDS_16K)) -> dict[str, torch.Tensor]:
    """Read the URGENT Lightning ``bsrnn.ckpt`` into a ``BSRNN`` state dict."""
    with torch.serialization.safe_globals([(_UrgentConfig, "baseline_code.config.Config")]):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return convert_urgent_state_dict(ckpt["state_dict"], n_bands=n_bands)


def build_bsrnn(checkpoint: str | None = None, **params: Any) -> BSRNNSE:
    """Build the 16 kHz URGENT BSRNN; optionally load the published weights."""
    model = BSRNNSE(**params)
    if checkpoint is not None:
        from utils.checkpoints import resolve_checkpoint_uri

        state = load_urgent_checkpoint(resolve_checkpoint_uri(checkpoint))
        model.bsrnn.load_state_dict(state, strict=True)
    return model
