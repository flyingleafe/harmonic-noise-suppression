"""FRCRN with frequency-axis bottleneck adapters for drone speech enhancement.

Paper: Xingyu Chen, Hanwen Bi, Wei-Ting Lai, Fei Ma, "Monaural Speech
Enhancement on Drone via Adapter Based Transfer Learning", IWAENC 2024,
arXiv:2405.10022 (CC0). The adapters have no public code; they are
re-implemented here from the paper text and its Fig. 2. The backbone is the
released ModelScope FRCRN (:mod:`models.frcrn.backbone`, Apache-2.0,
``damo/speech_frcrn_ans_cirm_16k``, trained on DNS-2020 at 16 kHz).

Paper statement -> code
-----------------------
* "pre-trained FRCRN ... parameters frozen" (Sec. 3.2, Fig. 2a snowflakes) ->
  ``freeze_backbone=True`` sets ``requires_grad=False`` on every backbone
  tensor; :meth:`FRCRNEnhancer.train` keeps the backbone in ``eval()`` so its
  complex BatchNorms use (and never update) the pretrained running stats.
* "the adapter embedded in the encoder module, positioned after each CR
  block" (Fig. 2a) -> one :class:`ComplexFreqAdapter` after every encoder
  block (7) of the first U-Net (see "Which U-Net(s)" below). Its output feeds
  the next encoder block and the skip-path attention block (Fig. 2a draws the
  skip taps at the CR-block/adapter pair).
* "the parameters of the adapter and the attention block are fine-tuned ...
  The attention block acts as the skip pathway" -> the attention block is the
  complex SE layer on the skip path (``se_layer_enc*``/``se_layer_dec*``;
  FRCRN paper Table 1 calls it "Attention Block"). ``train_attention=True``
  in the paper-reproduction config leaves every SE layer of both U-Nets trainable.
* "The input and output dimensions of the adapter remain consistent. The
  adapter has a skip connection" (Fig. 2b) -> ``x + adapter(x)``.
* Eq. (6): ``A_r_out = Adapter_r(A_r_in) - Adapter_i(A_i_in)``,
  ``A_i_out = Adapter_r(A_i_in) + Adapter_i(A_r_in)`` -> ``ComplexFreqAdapter``.
* Eq. (7): ``h = ReLU(W1 U_r + b1)``, ``U_r' = W2 h + b2``, ``W1`` projects the
  frequency axis F -> F/2, ``W2`` F/2 -> F, shared over channels and frames;
  "Complex cell shares the same structure" -> ``_AdapterCell`` (two
  ``nn.Linear`` on the frequency axis) for each of ``Adapter_r``/``Adapter_i``.
* "its parameters are initialized to zero, configuring the adapter as an
  approximate identity function" -> see ``adapter_init`` below.
* Sec. 3.3 "we use only SI-SNR as the loss function" -> a training-recipe
  fact; the loss lives in the project config, not here.

Settings the paper leaves unstated (value chosen, why)
------------------------------------------------------
* **Which U-Net(s).** The released FRCRN cascades two U-Nets; Fig. 2 draws
  one. We infer placement from Table 2's parameter counts. Counting every registered tensor
  (the release registers ``fsmn_enc0`` and ``se_layer_dec5`` that its forward
  never uses) and rounding as the table does elsewhere ("w/o pre-trained: 14"
  = 13,963,604 incl. the fixed STFT kernels; truncation would give 13):
  "Fine-tuning: 1.2 M" = ``fsmn_enc0..6`` + all SE layers of BOTH U-Nets =
  1,213,248 (encoder SE only: 1.11 M; first U-Net only: 0.61 M), so the
  attention blocks are all SE layers of both U-Nets (220,480). "Adapter
  tuning: 0.3 M" then fits adapters in the first U-Net only: 66,616 +
  220,480 = 287,096; adapters in both U-Nets give 353,712 (0.35 -> "0.4").
  Default ``adapt_unet2=False``; ``adapt_unet2=True`` adds the second set.
* **F/2 for odd or unit F.** Encoder outputs have F = 159, 78, 37, 17, 7, 2, 1
  bins (321-bin input); the hidden size is ``max(1, F // 2)``.
* **Initialisation.** A literally all-zero adapter is an exact identity but a
  dead fixed point: with ``W2 = 0`` and ``h = ReLU(b1) = 0`` no weight ever
  receives gradient (only ``b2`` moves). The paper says "approximate
  identity", so the default ``adapter_init="zero_up"`` zeroes the
  up-projection (``W2``, ``b2``) and keeps PyTorch's default init for the
  down-projection: every adapter adds exactly zero at step 0, with non-zero
  gradient on ``W2``. ``adapter_init="zeros"`` is the literal reading. The
  whole adapted model then matches the plain backbone to fp32 rounding, not
  bit-exactly (published checkpoint, 1 s input: max ``|diff|`` 1.26e-6 at
  max ``|y|`` 0.0287, i.e. 4.4e-5 relative; cause not traced).
* **Input level.** ModelScope's inference pipeline normalises every input
  with ``audio_norm`` (two-stage RMS normalisation to -25 dBFS) and returns
  the normalised level. ``input_norm=True`` applies the same normalisation
  (batched, per utterance, gain detached, ``eps`` against silence) and then
  divides the output by that gain, so the enhanced waveform is on the input's
  scale (needed by the project's scale-sensitive SDR metrics).
* **Length.** Upstream frames without centring or padding. ``forward``
  right-pads with zeros to the next ``640 + k*320`` samples and crops back to
  ``T``. The pipeline's extra padding to >= 1 s and its 120 s segmentation
  are not reproduced (inputs here are a few seconds).

The STFT is ModelScope's: 640-point sqrt-Hann window (40 ms), 320 hop
(20 ms), 640-point FFT -> 321 bins, un-centred Conv1d framing (not the
20 ms / 10 ms of the FRCRN paper's WSJ0 experiments).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import FRCRN, encoder_freq_sizes

__all__ = ["ComplexFreqAdapter", "FRCRNEnhancer", "build_adapter_frcrn", "build_frcrn"]

_ADAPTER_INITS = ("zero_up", "zeros")


class _AdapterCell(nn.Module):
    """Eq. (7): frequency-axis bottleneck ``W2 ReLU(W1 u + b1) + b2`` on ``(..., F)``."""

    def __init__(self, n_freq: int, n_hidden: int, init: str) -> None:
        super().__init__()
        self.down = nn.Linear(n_freq, n_hidden)
        self.up = nn.Linear(n_hidden, n_freq)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        if init == "zeros":
            nn.init.zeros_(self.down.weight)
            nn.init.zeros_(self.down.bias)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.up(F.relu(self.down(u)))


class ComplexFreqAdapter(nn.Module):
    """Complex bottleneck adapter over the frequency axis with a skip (Fig. 2b, Eq. 6).

    ``(B, C, F, T, 2) -> (B, C, F, T, 2)``.
    """

    def __init__(self, n_freq: int, init: str = "zero_up") -> None:
        super().__init__()
        if init not in _ADAPTER_INITS:
            raise ValueError(f"adapter init must be one of {_ADAPTER_INITS}, got {init!r}")
        n_hidden = max(1, n_freq // 2)
        self.cell_r = _AdapterCell(n_freq, n_hidden, init)
        self.cell_i = _AdapterCell(n_freq, n_hidden, init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.transpose(2, 3)  # (B, C, T, F, 2): frequency last for nn.Linear
        a_r, a_i = xf[..., 0], xf[..., 1]
        out_r = self.cell_r(a_r) - self.cell_i(a_i)
        out_i = self.cell_r(a_i) + self.cell_i(a_r)
        return x + torch.stack((out_r, out_i), dim=-1).transpose(2, 3)


def _audio_norm_gain(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """ModelScope ``audio_norm`` as a per-utterance gain: ``(B, T) -> (B, 1)``."""
    target = 10 ** (-25 / 20)
    scale1 = target / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(eps)
    pow_x = (x * scale1).pow(2)
    loud = pow_x > pow_x.mean(-1, keepdim=True)
    loud_mean = (pow_x * loud).sum(-1, keepdim=True) / loud.sum(-1, keepdim=True).clamp_min(1)
    scale2 = target / loud_mean.sqrt().clamp_min(eps)
    return scale1 * scale2


class FRCRNEnhancer(nn.Module):
    """``(B, 1, T)`` 16 kHz waveform -> ``(B, T)`` enhanced, FRCRN with optional adapters.

    ``adapters=False`` is the plain released FRCRN; ``adapters=True`` adds the
    frequency adapters of Chen et al. after the 7 encoder blocks of the first
    U-Net (``adapt_unet2=True``: of the second U-Net too). ``freeze_backbone=True``
    freezes every backbone tensor except, with ``train_attention=True``, the SE
    attention blocks, and pins the backbone in eval mode (see :meth:`train`).
    """

    def __init__(
        self,
        adapters: bool = True,
        checkpoint: str | None = None,
        freeze_backbone: bool = True,
        train_attention: bool = False,
        adapter_init: str = "zero_up",
        adapt_unet2: bool = False,
        input_norm: bool = True,
        win_len: int = 640,
        win_inc: int = 320,
        fft_len: int = 640,
        win_type: str = "hann",
    ) -> None:
        super().__init__()
        self.frcrn = FRCRN(win_len, win_inc, fft_len, win_type)
        if checkpoint is not None:
            from utils.checkpoints import resolve_checkpoint_uri

            path = resolve_checkpoint_uri(checkpoint)
            state = torch.load(path, map_location="cpu", weights_only=True)
            self.frcrn.load_state_dict(state, strict=True)
        self.adapters: nn.ModuleList | None = None
        if adapters:
            sizes = encoder_freq_sizes(fft_len // 2 + 1)
            self.adapters = nn.ModuleList(
                nn.ModuleList(ComplexFreqAdapter(f, adapter_init) for f in sizes)
                for _ in range(2 if adapt_unet2 else 1)
            )
        self.freeze_backbone = freeze_backbone
        self.input_norm = input_norm
        if freeze_backbone:
            for name, p in self.frcrn.named_parameters():
                p.requires_grad = train_attention and ".se_layer" in name
            self.frcrn.eval()

    def train(self, mode: bool = True) -> FRCRNEnhancer:
        """Frozen-backbone mode keeps ``self.frcrn`` (all its BatchNorms) in eval.

        The backbone has no dropout; its only mode-dependent layers are the
        complex BatchNorms, which must keep the pretrained running statistics.
        The trainable SE attention blocks have no mode-dependent layer.
        """
        super().train(mode)
        if self.freeze_backbone:
            self.frcrn.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)  # (B, 1, T) -> (B, T)
        n = x.shape[-1]
        win, hop = self.frcrn.win_len, self.frcrn.win_inc
        n_pad = win + max(0, -(-(n - win) // hop)) * hop - n
        if self.input_norm:
            gain = _audio_norm_gain(x.detach())
            x = x * gain
        x = F.pad(x, (0, n_pad))
        enc_adapters = (None, None) if self.adapters is None else (*self.adapters, None)[:2]
        _, wav = self.frcrn(x, enc_adapters)
        wav = wav[..., :n]
        if self.input_norm:
            wav = wav / gain
        return wav


def build_adapter_frcrn(
    checkpoint: str | None = None, freeze_backbone: bool = True, **params: Any
) -> nn.Module:
    """FRCRN + frequency adapters (Chen et al. 2024); ``checkpoint`` = path or ``r2://`` URI."""
    return FRCRNEnhancer(
        adapters=True, checkpoint=checkpoint, freeze_backbone=freeze_backbone, **params
    )


def build_frcrn(
    checkpoint: str | None = None, freeze_backbone: bool = False, **params: Any
) -> nn.Module:
    """The plain released FRCRN backbone (Zhao et al. 2022), fully trainable by default."""
    return FRCRNEnhancer(
        adapters=False, checkpoint=checkpoint, freeze_backbone=freeze_backbone, **params
    )
