"""StoRM — diffusion-based Stochastic Regeneration Model for speech enhancement.

Paper: J.-M. Lemercier, J. Richter, S. Welker, T. Gerkmann, "StoRM: A
Diffusion-based Stochastic Regeneration Model for Speech Enhancement and
Dereverberation", IEEE/ACM TASLP 31:2724-2737, 2023 (arXiv:2212.11851).
Upstream: https://github.com/sp-uhh/storm, commit
257e9636a7251ca40aa200753d5c0fe918e31879, MIT licence (Copyright (c) 2022 Signal
Processing (SP), Universitaet Hamburg); adapted with attribution.
``sgmse/model.py::StochasticRegenerationModel`` is the reference for everything
below; the NCSN++M networks are in :mod:`.ncsnpp`, the sampler in :mod:`.sampling`.

Model (paper Sec. III-C, Alg. 1/2; upstream class in brackets):
  * STFT n_fft=510, hop=128, periodic Hann, ``center=True`` (F=256 bins), then the
    amplitude compression ``spec_factor * |X|^0.5 e^{i angle X}`` [SpecsDataModule];
  * predictor D_theta = NCSN++M in discriminative mode, complex y -> complex D(y)
    [``denoiser_net``, 27,342,350 parameters];
  * score network s_phi = NCSN++M, input [x_tau, y, D(y)] (``condition="both"``),
    noise-conditioned on tau [``score_net``, 27,776,038 parameters];
  * OUVE SDE theta (paper gamma)=1.5, sigma_min=0.05, sigma_max=0.5, T=1,
    t_eps=0.03, whose steady-state mean is D(y) (``sde_input = y_denoised``);
  * spectrograms are zero-padded to a multiple of 64 frames (``pad_spec``).

``forward(mix, target)`` returns the upstream ``_step`` loss:
``w * 0.5 * sum|D(y) - x|^2 + (1 - w) * 0.5 * sum|s(x_tau, [y, D(y)], tau) sigma(tau) + z|^2``
with ``w = weighting_denoiser_to_score = 0.5``; the sums run over the whole batch
(upstream ``_reduce_op``; no batch mean). Gradients of the score term reach the
predictor through D(y) (``regen-joint-training``). ``forward(mix)`` runs the
predictor, then the PC sampler from ``D(y) + sigma(T) z`` with the published
setting N=50, one annealed-Langevin corrector step, snr r=0.5, and returns the
waveform ``(B, 1, T)``.

EMA: a ``torch_ema.ExponentialMovingAverage(decay=0.999, use_num_updates=True)``
over both networks, kept as buffers (``ema.*`` keys, plus ``ema.num_updates``) so
it is saved and restored by ``state_dict``/``load_state_dict``. In ``eval()``
mode every forward (sampling and validation loss) runs on the EMA weights via
``torch.func.functional_call``; the live parameters are never swapped, so a
``state_dict`` taken in either mode is the same. ``use_ema=False`` disables it.

Deviations from upstream:
  * EMA update timing: upstream updates after every ``optimizer_step``; this
    module cannot see the optimizer, so each grad-enabled training-mode
    ``forward(mix, target)`` updates the EMA first, with the parameters left by
    the previous optimizer step. After k steps the EMA equals upstream's after
    k-1 steps, the warm-up counter running one ahead
    (decay ``min(0.999, (1+n)/(10+n))``) — irrelevant once n > 9000.
  * Normalisation: upstream divides x and y by max|y| of the FULL noisy file
    before cropping the 256-frame training segment; here the normaliser is max|y|
    of the segment the model receives (identical at inference, where the whole
    utterance is given). The normaliser is clamped at 1e-8 (upstream divides by
    zero on an all-zero mixture).
  * Training on segments whose frame count is not a multiple of 64: upstream
    trains on exactly 256 frames (32640 samples), so its loss never sees
    padding. Here the spectrograms are padded like at inference and both loss
    terms are summed over the unpadded frames only — identical to upstream for
    256-frame (or any 64k-frame) segments.
  * Batched inference: upstream ``enhance`` processes one utterance; here each
    batch row is normalised by its own max, as if enhanced alone (the sampler
    noise then differs from a row-by-row run).
  * Dropped upstream branches: multi-microphone (``spatial_channels``), the
    ODE sampler, the Langevin (non-annealed) corrector, mae/sisdr losses,
    GaGNet/ConvTasNet predictors.

Settings the paper and code disagree on / leave open (value chosen, why):
  * window: paper says sqrt-Hann, the code default AND the published WSJ0+Chime3
    checkpoint's hyper-parameters say ``window=hann`` -> Hann.
  * spec_factor: unstated in the paper; checkpoint hparams ``spec_factor=0.33``
    (also the code's argparse default) -> 0.33 (SGMSE+ uses 0.15).
  * loss balance: paper Eq. 17 ``J_DSM + alpha J_sup`` with alpha=1; the code and
    checkpoint use ``0.5 J_sup + 0.5 J_DSM`` (the same objective up to a factor 2)
    -> code form.
  * sampler: the code's ``enhance`` default is N=30 without corrector; the paper
    (Sec. IV-B, all tables) and ``enhancement.py`` defaults use N=50 with one ALD
    step, r=0.5 -> N=50 + ALD.

Training ``mode`` (upstream ``--mode``): ``regen-joint-training`` (default; both
networks trained by the joint loss), ``regen-freeze-denoiser`` (predictor frozen,
run without grad), ``denoiser-only`` (stage-1 predictor pre-training as upstream
``DiscriminativeModel``: loss ``mean_b 0.5 sum|D(y) - x|^2``, inference returns
D(y)). Both networks exist in every mode so a stage-1 state_dict loads into the
joint model unchanged.

Parameter count: 27,342,350 (predictor) + 27,776,038 (score network) =
55,118,388, identical to the published WSJ0+Chime3 checkpoint and to upstream
built at this commit. Paper: "roughly 27.8M" per NCSN++M and "removes only 1.8%"
for the predictive variant; measured gap 1.56% (the predictor has no
time-embedding MLP and 2 instead of 6 real input channels).

Upstream training recipe (paper Sec. IV-B e; ``train.py``; checkpoint hparams):
Adam, lr 1e-4, no weight decay; Lightning defaults for clipping (none) and
precision (fp32), not overridden in ``train.py``; effective batch 16
(checkpoint: ``batch_size=8`` per process); EMA decay 0.999; 256-frame
segments (2.04 s) normalised by max|y| of the full utterance; at most 1000
epochs with early stopping on the validation loss (paper: patience 10; code:
``EarlyStopping(patience=50)``); default strategy: pre-train the predictor with
the MSE loss (``denoiser-only``), then train both jointly with Eq. 17. The
published WSJ0+Chime3 checkpoint stopped at epoch 148 (119,051 EMA updates).
Inference: N=50, reverse-diffusion predictor + one ALD corrector step, r=0.5.

Parity (scratch, upstream @257e9636 on CPU, same perturbed random weights, same
seeds): spectrogram, predictor output, score-network output, the joint training
loss (B=2, 64 frames) and the sampler output (N=10 + ALD on live weights; N=3 on
distinct EMA weights) agree bit-exactly (max abs difference 0). Gradients of the
joint loss (B=1) agree to 1.7e-6 relative per tensor (global 6.5e-7), except an
attention key bias whose true gradient is zero (|g| ~ 1e-11, rounding noise).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from models.sgmse.sde import OUVESDE

from .ncsnpp import NCSNppM
from .sampling import pc_sample

_MODES = ("regen-joint-training", "regen-freeze-denoiser", "denoiser-only")
_CONDITIONS = ("noisy", "post_denoiser", "both")
_PAD_FRAMES = 64  # upstream pad_spec


class ParamEMA(nn.Module):
    """``torch_ema.ExponentialMovingAverage`` (v0.3, ``use_num_updates=True``) whose
    shadow parameters are buffers, so they travel with ``state_dict`` and ``.to``.

    ``shadow <- shadow - (1 - d) (shadow - param)``, ``d = min(decay, (1+n)/(10+n))``.
    """

    def __init__(self, named_params: Iterable[tuple[str, nn.Parameter]], decay: float) -> None:
        super().__init__()
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        self.decay = float(decay)
        self.names: list[str] = []
        for name, p in named_params:
            self.register_buffer(self._key(name), p.detach().clone())
            self.names.append(name)
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    @staticmethod
    def _key(name: str) -> str:
        return name.replace(".", "__")

    def shadow(self, name: str) -> torch.Tensor:
        return getattr(self, self._key(name))

    def shadow_dict(self, prefix: str) -> dict[str, torch.Tensor]:
        """Shadow tensors of the parameters under ``prefix`` (names relative to it)."""
        return {n[len(prefix) :]: self.shadow(n) for n in self.names if n.startswith(prefix)}

    @torch.no_grad()
    def update(self, named_params: Iterable[tuple[str, nn.Parameter]]) -> None:
        self.num_updates += 1
        n = int(self.num_updates)
        one_minus_decay = 1.0 - min(self.decay, (1 + n) / (10 + n))
        for name, p in named_params:
            s = self.shadow(name)
            s.sub_((s - p.detach()).mul_(one_minus_decay))


class StoRMModel(nn.Module):
    def __init__(
        self,
        *,
        n_fft: int = 510,
        hop_length: int = 128,
        window: str = "hann",
        spec_factor: float = 0.33,
        spec_abs_exponent: float = 0.5,
        theta: float = 1.5,
        sigma_min: float = 0.05,
        sigma_max: float = 0.5,
        t_eps: float = 0.03,
        backbone: dict[str, Any] | None = None,
        condition: str = "both",
        mode: str = "regen-joint-training",
        weighting_denoiser_to_score: float = 0.5,
        ema_decay: float = 0.999,
        use_ema: bool = True,
        sampler_steps: int = 50,
        corrector: str = "ald",
        corrector_steps: int = 1,
        snr: float = 0.5,
    ) -> None:
        super().__init__()
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        if condition not in _CONDITIONS:
            raise ValueError(f"condition must be one of {_CONDITIONS}, got {condition!r}")
        if window not in ("hann", "sqrthann"):
            raise ValueError(f"window must be 'hann' or 'sqrthann', got {window!r}")
        if corrector not in ("ald", "none"):
            raise ValueError(f"corrector must be 'ald' or 'none', got {corrector!r}")
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.spec_factor = float(spec_factor)
        self.spec_abs_exponent = float(spec_abs_exponent)
        self.t_eps = float(t_eps)
        self.condition = condition
        self.mode = mode
        self.weighting_denoiser_to_score = float(weighting_denoiser_to_score)
        self.use_ema = bool(use_ema)
        self.sampler_steps = int(sampler_steps)
        self.corrector = corrector
        self.corrector_steps = int(corrector_steps)
        self.snr = float(snr)
        self.sde = OUVESDE(theta=theta, sigma_min=sigma_min, sigma_max=sigma_max, N=sampler_steps)

        win = torch.hann_window(self.n_fft, periodic=True)
        self.register_buffer(
            "window", win.sqrt() if window == "sqrthann" else win, persistent=False
        )

        bb = dict(backbone or {})
        for key in ("input_channels", "discriminative"):
            if key in bb:
                raise ValueError(f"backbone.{key} is set by StoRMModel, not the config")
        n_down = 2 ** (len(bb.get("ch_mult", (1, 2, 2, 2))) - 1)
        if (self.n_fft // 2 + 1) % n_down or _PAD_FRAMES % n_down:
            raise ValueError(f"F={self.n_fft // 2 + 1} and {_PAD_FRAMES} must divide by {n_down}")
        n_cond = 2 if condition == "both" else 1
        self.denoiser_net = NCSNppM(**bb, discriminative=True)
        self.score_net = NCSNppM(**bb, input_channels=2 * (1 + n_cond), discriminative=False)
        if mode == "regen-freeze-denoiser":
            self.denoiser_net.requires_grad_(False)

        self.ema = ParamEMA(self.named_parameters(), ema_decay)

    # ─── EMA ──────────────────────────────────────────────────────────────────

    def update_ema(self) -> None:
        self.ema.update(self.named_parameters())

    def _networks(self) -> tuple[Callable[..., torch.Tensor], Callable[..., torch.Tensor]]:
        """(predictor, score network) callables; EMA weights in eval mode."""
        if self.training or not self.use_ema:
            return self.denoiser_net, self.score_net
        den_w = self.ema.shadow_dict("denoiser_net.")
        score_w = self.ema.shadow_dict("score_net.")

        def denoiser(y: torch.Tensor) -> torch.Tensor:
            return functional_call(self.denoiser_net, den_w, (y,))

        def score_net(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return functional_call(self.score_net, score_w, (x, t))

        return denoiser, score_net

    # ─── STFT + amplitude compression ─────────────────────────────────────────

    def _spec(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, T) waveform -> compressed complex (B, 1, F, frames)."""
        window = self.window
        assert isinstance(window, torch.Tensor)
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
            return_complex=True,
        )
        if self.spec_abs_exponent != 1:
            e = self.spec_abs_exponent
            spec = spec.abs() ** e * torch.exp(1j * spec.angle())
        return (spec * self.spec_factor).unsqueeze(1)

    def _wave(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        """Compressed complex (B, F, frames) -> (B, length) waveform."""
        spec = spec / self.spec_factor
        if self.spec_abs_exponent != 1:
            e = self.spec_abs_exponent
            spec = spec.abs() ** (1 / e) * torch.exp(1j * spec.angle())
        window = self.window
        assert isinstance(window, torch.Tensor)
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
            length=length,
        )

    @staticmethod
    def _pad(spec: torch.Tensor) -> torch.Tensor:
        num_pad = (-spec.shape[-1]) % _PAD_FRAMES
        return F.pad(spec, (0, num_pad)) if num_pad else spec

    @staticmethod
    def _mono(audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 3:
            if audio.shape[1] != 1:
                raise ValueError(f"StoRM is single-channel; got {audio.shape[1]} channels")
            return audio[:, 0]
        if audio.ndim != 2:
            raise ValueError(f"expected (B, 1, T) or (B, T), got {tuple(audio.shape)}")
        return audio

    def _conditioning(self, y: torch.Tensor, y_den: torch.Tensor) -> list[torch.Tensor]:
        if self.condition == "noisy":
            return [y]
        if self.condition == "post_denoiser":
            return [y_den]
        return [y, y_den]

    # ─── forward ──────────────────────────────────────────────────────────────

    def forward(self, mix: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        if target is not None:
            return self._loss(mix, target)
        return self._enhance(mix)

    def _loss(self, mix: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.training and torch.is_grad_enabled() and self.use_ema:
            self.update_ema()
        y_wav = self._mono(mix)
        x_wav = self._mono(target)
        norm = y_wav.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        y = self._spec(y_wav / norm)
        x = self._spec(x_wav / norm)
        n_frames = y.shape[-1]
        y, x = self._pad(y), self._pad(x)
        denoiser, score_net = self._networks()

        with torch.set_grad_enabled(
            torch.is_grad_enabled() and self.mode != "regen-freeze-denoiser"
        ):
            y_den = denoiser(y)
        den_sq = torch.square(torch.abs(y_den - x))[..., :n_frames]
        if self.mode == "denoiser-only":
            # DiscriminativeModel._loss (mse): batch mean of the per-example half sum
            return torch.mean(0.5 * torch.sum(den_sq.reshape(den_sq.shape[0], -1), dim=-1))

        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        mean, std = self.sde.marginal_prob(x, y_den, t)
        z = torch.randn_like(x)  # complex normal, variance 0.5 per component
        sigma = std[:, None, None, None]
        x_t = mean + sigma * z
        score = -score_net(torch.cat([x_t, *self._conditioning(y, y_den)], dim=1), t)
        err = score * sigma + z
        loss_score = 0.5 * torch.sum(torch.square(torch.abs(err))[..., :n_frames])
        loss_denoiser = 0.5 * torch.sum(den_sq)
        w = self.weighting_denoiser_to_score
        return w * loss_denoiser + (1 - w) * loss_score

    @torch.no_grad()
    def _enhance(self, mix: torch.Tensor) -> torch.Tensor:
        y_wav = self._mono(mix)
        length = y_wav.shape[-1]
        norm = y_wav.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        y = self._pad(self._spec(y_wav / norm))
        denoiser, score_net = self._networks()
        y_den = denoiser(y)
        if self.mode == "denoiser-only":
            sample = y_den
        else:
            cond = self._conditioning(y, y_den)

            def score_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                return -score_net(torch.cat([x, *cond], dim=1), t)

            sample = pc_sample(
                self.sde,
                score_fn,
                y_den,
                N=self.sampler_steps,
                eps=self.t_eps,
                corrector=self.corrector,
                snr=self.snr,
                corrector_steps=self.corrector_steps,
            )
        # upstream to_audio: iSTFT of the padded spectrogram, cut to the input length
        wave = self._wave(sample[:, 0], length) * norm
        return wave.unsqueeze(1)


def build_storm(**params: Any) -> StoRMModel:
    """Factory for ``conf/model/storm.yaml``."""
    return StoRMModel(**params)


__all__ = ["ParamEMA", "StoRMModel", "build_storm"]
