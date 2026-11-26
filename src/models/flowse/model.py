"""FlowSE — conditional flow matching speech enhancement (Lee et al., ICASSP 2025).

Paper
    Seonggyu Lee, Sein Cheong, Sangwook Han, Jong Won Shin, "FlowSE: Flow
    Matching-based Speech Enhancement", ICASSP 2025,
    doi:10.1109/ICASSP49660.2025.10888274, arXiv:2508.06840. (Not the
    mel-spectrogram + vocoder "FlowSE" of Wang et al., Interspeech 2025.)

Official code
    https://github.com/seongq/flowmse @ f6b479d13fecc6cb6f12394f46dfc6799fb479b6
    (built on sp-uhh/sgmse). The repository has NO licence, so nothing of it is
    copied here: the flow-matching path, the loss and the Euler sampler below
    are re-implemented from the paper's equations, and the upstream code served
    only as a numerical reference in scratch scripts. The backbone is the
    in-repo NCSN++ port :class:`models.sgmse.ncsnpp.NCSNpp` (sp-uhh/sgmse, MIT),
    which is the network flowmse ships (``flowmse/backbones/ncsnpp.py`` differs
    from sgmse's only in argparse plumbing and in dividing the output by the
    time input unconditionally, which equals ``scale_by_sigma=True``).

Parity (scratch script ``/tmp/flowse_parity.py`` against the upstream clone,
CPU fp32, upstream random init copied into the port): parameter count
65,590,822 on both sides; vector field (2 x 256 x 64 bins, tau 0.7 / 0.05),
the CFM loss of three Adam(1e-4) + EMA training steps with identical seeds,
the raw parameters and the EMA shadows after them, and the Euler sampler
with the same seed (T = 16000 / 8064 / 12345 samples, N = 5 / 5 / 1, eval =
EMA weights; both on the port's own trained weights and on the upstream
checkpoint loaded through :meth:`FlowSEModel.load_flowmse_checkpoint`) all
agree BIT-EXACTLY (max |diff| 0). The prior noise is drawn with
``randn_like`` semantics (the STFT's frame-major memory layout), which is what
makes the seeded sample identical for unpadded frame counts too.

Time convention
    The paper runs the flow from noisy (t = 0) to clean (t = 1). The code, and
    hence every published checkpoint, feeds the network the REVERSED time
    tau = 1 - t (tau = 1 at the noisy end), because NCSN++ embeds log(tau) and
    divides its output by tau. This module uses tau everywhere; with t = 1 - tau
    every formula is the paper's:

    * Eq. (20)/(21): mu = tau * y + (1 - tau) * x1, sigma_tau = tau * sigma
      (``_path``), sigma = 0.487 (Sec. IV, chosen on validation WB-PESQ).
    * Eq. (16)/(19) with the OT-style path: the conditional vector field in
      paper time is x1 - y - sigma * z; in tau it is the negative,
      ``sigma * z + (y - x1)`` (the training target in ``loss``).
    * Eq. (17): CFM loss, the squared error summed over all TF bins and both
      real/imag parts, times 0.5, averaged over the batch (the sgmse
      reduction the upstream code keeps).
    * Sec. III-A: t ~ U[0, 1 - t_delta], t_delta = 0.03, i.e.
      tau = (1 - U[0, 1)) * (1 - t_delta) + t_delta.
    * Eq. (22) + Sec. IV: Euler from x ~ N(y, sigma^2 I) over the N time points
      tau = linspace(1, t_delta, N), step tau_i - tau_{i+1}, last step t_delta
      (so the path ends at tau = 0); N = 5 (NFE 5, Table I). N = 1 takes a
      single step of length 1.
    * v_theta(x, y, tau) = -NCSNpp(cat[x, y], tau): the network sees x and y as
      four real channels (Re/Im of each) and returns one complex channel; the
      leading minus sign is the upstream convention and is part of what a
      checkpoint encodes.

Signal path (Sec. IV: "All configurations of STFT and amplitude compression
followed those in [BBED, CRP]", i.e. the sgmse settings; native 16 kHz, no
resampling):
    peak-normalise the mixture (the clean target by the same factor) -> STFT
    (n_fft 510, hop 128, periodic Hann, centred, reflect padding; 256 bins)
    -> 0.15 * |X|^0.5 * exp(i angle X) -> pad frames up to a multiple of 64
    (the U-Net's six 2x down-samplings) -> flow / sampler -> inverse
    compression -> iSTFT of the padded spectrogram cut to the input length ->
    undo the peak normalisation.

Backbone (Sec. IV: "NCSN++ ... with the same configuration as BBED and CRP"):
nf 128, ch_mult (1, 1, 2, 2, 2, 2, 2), 2 BigGAN residual blocks per level,
attention at the 16-bin level, Gaussian Fourier time embedding (scale 16),
FIR (1, 3, 3, 1) resampling, output-skip / input-skip pyramids: 65,590,822
parameters, of which 65,590,694 are trainable and 128 are the frozen weights of
the Gaussian Fourier time projection (measured; the official code builds the
same count). The FlowSE paper gives no count; the SGMSE+ paper quotes 65.6M for
the same network, which matches.

Settings the paper leaves unstated (values taken from the official code)
    * sigma_min = 0 (Eq. (21) already implies it; the code's ``sigma_min``
      argument defaults to 0.0).
    * time input tau = 1 - t, log-tau Fourier embedding, output divided by tau,
      the minus sign of v_theta (see above).
    * prior noise: complex standard normal (variance 1/2 per real component)
      scaled by sigma, the code's ``torch.randn_like`` on a complex tensor.
    * training segments: 256 STFT frames = 32,640 samples (2.04 s) cut at a
      random offset; inference pads the frames to a multiple of 64 and runs the
      iSTFT over the padded frames (no crop in the spectral domain).
    * EMA: ``torch_ema.ExponentialMovingAverage(decay=0.999)`` with its default
      warm-up, decay_k = min(0.999, (1 + k) / (10 + k)) at update k, updated
      after every optimizer step; evaluation uses the EMA weights.

Deviations from the official code
    * EMA update timing. Upstream updates the EMA in the optimizer-step hook.
      This module owns no optimizer, so the update for step k runs lazily at the
      start of training-mode forward k + 1, or when the model is switched to
      eval, or when ``state_dict`` is taken -- whichever comes first. The
      sequence of EMA values is identical to upstream as long as every
      grad-enabled training forward is followed by exactly one optimizer step
      (no gradient accumulation).
    * Inputs whose frame count is not a multiple of 64 are zero-padded in
      training as well (upstream training always sees exactly 256 frames, so it
      never pads); the loss sums only over the original frames, so on 256-frame
      segments the loss is upstream's exactly.
    * A silent (all-zero) mixture is left unscaled instead of being divided by
      zero.
    * Each batch row is peak-normalised on its own (upstream normalises per file
      both in its dataset and in its evaluation script; the same thing).

Generative protocol (as :class:`models.sgmse.SGMSEModel`): ``forward(mix,
target)`` returns the scalar CFM loss; ``forward(mix)`` runs the Euler sampler
and returns ``(B, 1, T)``. The EMA copy of the backbone lives in ``self.ema``
as buffers, so ``state_dict()`` / ``load_state_dict()`` carry it; eval mode
runs the EMA weights, train mode the raw ones.

Upstream training recipe (paper Sec. IV + ``train.py`` / ``flowmse/model.py``)
    CFM loss of Eq. (17) as above; tau ~ U(0.03, 1]; sigma 0.487 constant
    (sigma_min 0); Adam, lr 1e-4, no schedule, no gradient clipping, fp32
    (Lightning default);
    batch 8 of 256-frame (2.04 s) segments, noisy-peak normalised; EMA 0.999
    with torch_ema warm-up; 250 epochs in the paper (``max_epochs=1000`` hard
    coded in ``train.py``), model selection by WB-PESQ on 10 validation files
    per epoch; inference: Euler, N = 5 (``evaluate.py`` default). Published
    checkpoints (Google Drive links in the README, no licence, 1.2 GB each per
    the Drive page, Lightning ``.ckpt``; not downloaded here):
    WSJ0_CHIME3_FLOWSE_ICASSP_2025.ckpt and VB_DMD_FLOWSE_ICASSP_2025.ckpt;
    ``load_flowmse_checkpoint`` maps their ``state_dict`` + ``ema`` entries.
"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from models.sgmse.ncsnpp import NCSNpp

# NCSN++ settings fixed by the FlowSE backbone; every other NCSNpp keyword
# (nf, ch_mult, ...) comes from the ``backbone`` config dict.
_BACKBONE_FIXED: dict[str, Any] = {"scale_by_sigma": True}
_BACKBONE_KEYS = frozenset(inspect.signature(NCSNpp.__init__).parameters) - {
    "self",
    "unused_kwargs",
    *_BACKBONE_FIXED,
}


class ParameterEMA(nn.Module):
    """Exponential moving average of a module's parameters, stored as buffers.

    ``self.module`` is a deep copy of the tracked network whose parameters are
    re-registered as buffers: it carries the averaged weights through
    ``state_dict`` under the tracked module's own key names, follows ``.to()``,
    stays out of ``parameters()``, and can be called directly to evaluate the
    network with the averaged weights. The update is torch_ema 0.3's
    (``ExponentialMovingAverage(use_num_updates=True)``) operation for
    operation: ``decay_k = min(decay, (1 + k) / (10 + k))``,
    ``shadow -= (1 - decay_k) * (shadow - param)``.
    """

    def __init__(self, module: nn.Module, decay: float) -> None:
        super().__init__()
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"EMA decay must lie in [0, 1], got {decay}")
        names = [n for n, _ in module.named_parameters()]
        if len(names) != len(list(module.named_parameters(remove_duplicate=False))):
            raise ValueError("ParameterEMA does not support shared parameters")
        self.decay = float(decay)
        self.num_updates = 0
        self._names = names
        shadow = copy.deepcopy(module)
        for sub in shadow.modules():
            for name, p in list(sub._parameters.items()):
                del sub._parameters[name]
                if p is not None:
                    sub.register_buffer(name, p.detach().clone())
        shadow.requires_grad_(False)
        self.module = shadow

    def shadow_tensors(self) -> list[torch.Tensor]:
        """The averaged tensors, in ``tracked.named_parameters()`` order."""
        return [self.module.get_buffer(n) for n in self._names]

    @torch.no_grad()
    def update(self, tracked: nn.Module) -> None:
        params = [p.detach() for p in tracked.parameters()]
        shadow = self.shadow_tensors()
        if len(params) != len(shadow):
            raise ValueError("tracked module does not match the EMA copy")
        self.num_updates += 1
        k = self.num_updates
        decay = min(self.decay, (1 + k) / (10 + k))
        tmp = torch._foreach_sub(shadow, params)
        torch._foreach_mul_(tmp, 1.0 - decay)
        torch._foreach_sub_(shadow, tmp)

    def get_extra_state(self) -> dict[str, int]:
        return {"num_updates": self.num_updates}

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        self.num_updates = int(state["num_updates"])


class FlowSEModel(nn.Module):
    """FlowSE: NCSN++ vector field, conditional flow matching, Euler sampler."""

    window: torch.Tensor

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        n_fft: int = 510,
        hop_length: int = 128,
        spec_factor: float = 0.15,
        spec_abs_exponent: float = 0.5,
        sigma: float = 0.487,
        t_eps: float = 0.03,
        t_rev: float = 1.0,
        num_steps: int = 5,
        ema_decay: float = 0.999,
        backbone: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if sample_rate != 16000:
            raise ValueError(f"FlowSE is a native 16 kHz model, got sample_rate={sample_rate}")
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        if not 0.0 < t_eps < t_rev <= 1.0:
            raise ValueError(f"need 0 < t_eps < t_rev <= 1, got t_eps={t_eps}, t_rev={t_rev}")
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.spec_factor = float(spec_factor)
        self.spec_abs_exponent = float(spec_abs_exponent)
        self.sigma = float(sigma)
        self.t_eps = float(t_eps)
        self.t_rev = float(t_rev)
        self.num_steps = int(num_steps)

        bb = dict(backbone or {})
        unknown = set(bb) - _BACKBONE_KEYS
        if unknown:
            raise ValueError(f"unknown NCSN++ backbone keys: {sorted(unknown)}")
        for key in ("ch_mult", "attn_resolutions"):
            if key in bb:
                bb[key] = tuple(int(v) for v in bb[key])
        self.dnn = NCSNpp(**bb, **_BACKBONE_FIXED)
        self.frame_multiple = 2 ** (self.dnn.num_resolutions - 1)
        n_bins = n_fft // 2 + 1
        if n_bins % self.frame_multiple:
            raise ValueError(
                f"n_fft={n_fft} gives {n_bins} bins, not a multiple of {self.frame_multiple}"
            )
        self.ema = ParameterEMA(self.dnn, ema_decay)
        self._ema_pending = False
        self.register_buffer("window", torch.hann_window(n_fft, periodic=True), persistent=False)

    # ── EMA bookkeeping ──────────────────────────────────────────────────────

    def _flush_ema(self) -> None:
        if self._ema_pending:
            self.ema.update(self.dnn)
            self._ema_pending = False

    def train(self, mode: bool = True) -> FlowSEModel:
        if not mode:
            self._flush_ema()
        return super().train(mode)

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        self._flush_ema()
        return super().state_dict(*args, **kwargs)

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        self._ema_pending = False
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def load_flowmse_checkpoint(
        self, state_dict: Mapping[str, torch.Tensor], ema_state: Mapping[str, Any]
    ) -> None:
        """Load an official flowmse Lightning checkpoint's weights.

        ``state_dict`` is ``ckpt["state_dict"]`` (keys ``dnn.*``) and
        ``ema_state`` is ``ckpt["ema"]`` (torch_ema: ``shadow_params`` in
        ``dnn.parameters()`` order, plus ``num_updates``). Reading the
        ``.ckpt`` file itself needs ``torch.load(..., weights_only=False)``
        with the ``flowmse`` package importable, because Lightning pickles the
        hyper-parameters (including the ``SpecsDataModule`` class).
        """
        dnn_state = {k[len("dnn.") :]: v for k, v in state_dict.items() if k.startswith("dnn.")}
        self.dnn.load_state_dict(dnn_state, strict=True)
        shadow = list(ema_state["shadow_params"])
        targets = self.ema.shadow_tensors()
        if len(shadow) != len(targets):
            raise ValueError(f"EMA holds {len(shadow)} tensors, expected {len(targets)}")
        with torch.no_grad():
            for dst, src in zip(targets, shadow, strict=True):
                if dst.shape != src.shape:
                    raise ValueError(f"EMA tensor shape {tuple(src.shape)} != {tuple(dst.shape)}")
                dst.copy_(src)
        self.ema.num_updates = int(ema_state.get("num_updates") or 0)
        self._ema_pending = False

    # ── spectral front end ───────────────────────────────────────────────────

    def _stft(self, wave: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            wave,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            return_complex=True,
        )

    def _istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            length=length,
        )

    def _compress(self, spec: torch.Tensor) -> torch.Tensor:
        e = self.spec_abs_exponent
        if e != 1:
            spec = spec.abs() ** e * torch.exp(1j * spec.angle())
        return spec * self.spec_factor

    def _decompress(self, spec: torch.Tensor) -> torch.Tensor:
        spec = spec / self.spec_factor
        e = self.spec_abs_exponent
        if e != 1:
            spec = spec.abs() ** (1 / e) * torch.exp(1j * spec.angle())
        return spec

    def _pad_frames(self, spec: torch.Tensor) -> torch.Tensor:
        pad = -spec.shape[-1] % self.frame_multiple
        return nn.functional.pad(spec, (0, pad)) if pad else spec

    def _to_spec(self, wave: torch.Tensor) -> torch.Tensor:
        """(B, T) normalised waveform -> compressed (B, 1, F, N) spectrogram."""
        return self._compress(self._stft(wave)).unsqueeze(1)

    @staticmethod
    def _mono(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            if x.shape[1] != 1:
                raise ValueError(f"FlowSE is single-channel, got {x.shape[1]} channels")
            return x[:, 0]
        if x.ndim != 2:
            raise ValueError(f"expected (B, T) or (B, 1, T), got shape {tuple(x.shape)}")
        return x

    @staticmethod
    def _peak(x: torch.Tensor) -> torch.Tensor:
        peak = x.abs().amax(dim=-1, keepdim=True)
        return torch.where(peak > 0, peak, torch.ones_like(peak))

    # ── flow ─────────────────────────────────────────────────────────────────

    def vector_field(
        self, x: torch.Tensor, tau: torch.Tensor, y: torch.Tensor, net: nn.Module | None = None
    ) -> torch.Tensor:
        """v_theta(x, y, tau) in network time tau = 1 - t; x, y are (B, 1, F, N) complex."""
        if net is None:
            net = self.dnn if self.training else self.ema.module
        return -net(torch.cat([x, y], dim=1), tau)

    def _path(
        self, x1: torch.Tensor, y: torch.Tensor, tau: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and std of p(x_tau | x1, y), Eqs. (20)-(21) with t = 1 - tau."""
        mean = (1 - tau)[:, None, None, None] * x1 + tau[:, None, None, None] * y
        return mean, tau * self.sigma

    def loss(self, mix: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Conditional flow-matching loss, Eq. (17)."""
        if self.training and torch.is_grad_enabled():
            self._flush_ema()
        y_wave = self._mono(mix)
        x_wave = self._mono(target)
        norm = self._peak(y_wave)
        x1 = self._to_spec(x_wave / norm)
        y = self._to_spec(y_wave / norm)
        n_frames = x1.shape[-1]
        x1 = self._pad_frames(x1)
        y = self._pad_frames(y)

        batch = x1.shape[0]
        u = torch.rand(batch, device=x1.device)
        tau = torch.clamp_max((1 - u) * (self.t_rev - self.t_eps) + self.t_eps, self.t_rev)
        mean, std = self._path(x1, y, tau)
        z = torch.randn_like(x1)
        x_tau = mean + std[:, None, None, None] * z
        target_vf = self.sigma * z + (y - x1)
        err = (self.vector_field(x_tau, tau, y) - target_vf)[..., :n_frames]
        loss = torch.mean(0.5 * torch.sum(torch.square(err.abs()).reshape(batch, -1), dim=-1))

        if self.training and torch.is_grad_enabled():
            self._ema_pending = True
        return loss

    @torch.no_grad()
    def enhance(
        self,
        mix: torch.Tensor,
        num_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Euler ODE sampler, Eq. (22): (B, T) or (B, 1, T) mixture -> (B, 1, T)."""
        n = self.num_steps if num_steps is None else int(num_steps)
        if n < 1:
            raise ValueError(f"num_steps must be >= 1, got {n}")
        y_wave = self._mono(mix)
        length = y_wave.shape[-1]
        norm = self._peak(y_wave)
        y = self._pad_frames(self._to_spec(y_wave / norm))
        net = self.dnn if self.training else self.ema.module

        # randn_like semantics (fills y's memory layout, which for an unpadded
        # STFT is frame-major), so a seed gives upstream's exact prior sample.
        z = torch.empty_like(y).normal_(generator=generator)
        x = y + z * self.sigma
        taus = torch.linspace(self.t_rev, self.t_eps, n, device=y.device)
        for i in range(n):
            tau = taus[i]
            step = tau - taus[i + 1] if i != n - 1 else taus[-1]
            vec_tau = torch.ones(y.shape[0], device=y.device) * tau
            x = x + self.vector_field(x, vec_tau, y, net) * -step
        wave = self._istft(self._decompress(x[:, 0]), length)
        return (wave * norm).unsqueeze(1)

    def forward(self, mix: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        if target is not None:
            return self.loss(mix, target)
        return self.enhance(mix)


def build_flowse(**params: Any) -> FlowSEModel:
    return FlowSEModel(**params)
