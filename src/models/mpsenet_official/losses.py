"""MP-SENet training losses — official release (``train.py``, ``models/model.py``,
``models/discriminator.py``, ``dataset.py`` of https://github.com/yxlu-0102/MP-SENet
at ``89932cfe90d1dacb8e170e4a331d762462c21792``, MIT licence, (c) 2023 Yexin Lu).

Every term of upstream's training step is an importable function, and
:func:`mpsenet_losses` computes one batch's generator and discriminator losses
from raw ``(noisy, clean)`` waveforms. No optimizer logic lives here.

Upstream step (``train.py``), reproduced by :func:`mpsenet_losses`:
  1. normalise both waveforms by the NOISY utterance's RMS factor
     ``sqrt(L / sum(noisy**2))`` (``dataset.py``);
  2. STFT (400/100/400, Hann, ``|.|**0.3``) of clean and noisy; generator pass;
     iSTFT -> ``audio_g``; re-STFT of ``audio_g`` -> ``mag_g_hat, com_g_hat``;
  3. PESQ target ``(PESQ_wb(clean, audio_g) - 1) / 3.5`` on the CPU, ``None``
     when any utterance of the batch fails (then the fake-pair D term is 0);
  4. ``L_D = MSE(D(clean_mag, clean_mag), 1) + MSE(D(clean_mag, mag_g_hat.detach()), Q_pesq)``;
  5. ``L_G = 0.9 L_mag + 0.3 (L_ip + L_gd + L_iaf) + 0.1 L_com + 0.1 L_con
     + 0.05 L_metric + 0.2 L_time``, where ``L_com``/``L_con`` are
     ``2 * MSE`` over the stacked (real, imag) tensor and
     ``L_metric = MSE(D(clean_mag, mag_g_hat), 1)``.

Ordering note: upstream steps the discriminator optimizer between 4 and 5, so
its ``L_metric`` is evaluated by the just-updated D. :func:`mpsenet_losses`
evaluates both with the D it is given (no optimizer inside). A loop that wants
upstream's exact order calls :func:`generator_forward`,
:func:`discriminator_loss`, steps D, then :func:`generator_loss`.
``L_G.backward()`` also writes gradients into D's parameters (upstream relies
on ``optim_d.zero_grad()`` at the start of the next step); keep D's and G's
gradients separate (zero D's grads before ``L_D.backward()``).

Deviation: upstream's iSTFT returns ``hop * floor(L / hop)`` samples and then
computes ``L1(clean, audio_g)``, which only type-checks because training
segments (32000) are a multiple of the hop. Here ``audio_g`` is reconstructed
with ``length=L`` so any length works; identical at upstream's segment length.
Upstream's unused helper ``discriminator.metric_loss`` and the validation-only
``model.pesq_score`` are not ported.

Parity with upstream (same weights, same 2 x 0.5 s VoiceBank excerpts, same
dropout seed, fp32 CPU): PESQ target 1.7e-7 absolute; L_D, L_G and every term
agree to <= 3.6e-6 relative (most bit-exact); the discriminator's train-mode
output and its spectral-norm ``u``/``v`` after one call are bit-exact.
Parameter gradients: max |diff| is 5.1e-5 (G) / 3.0e-4 (D) of the largest
gradient; worst single tensor 8.6e-4 relative for G (its conv biases before
InstanceNorm, whose true gradient is 0, excluded) and 4.5e-3 for D. For scale,
perturbing the input by 1e-7 of its peak moves the gradients of the same step
by up to 65 % (random init; max-pool argmax, atan2), so these gradient
differences are rounding.

Upstream recipe (``config.json``, ``train.py``; facts only): AdamW for both G
and D, lr 5e-4, betas (0.8, 0.99), torch-default weight decay 0.01;
``ExponentialLR(gamma=0.99)`` stepped once per epoch for both; batch size 4 in
total (``CUDA_VISIBLE_DEVICES=0,1``, i.e. 2 per GPU); 2 s segments (32000
samples, random crop, zero-padded if shorter); 400 epochs by default
(``--training_epochs``; the long paper says 500k steps); D step then G step
every iteration; loss weights as in 5; fp32 (no AMP), no EMA, no gradient
clipping; seed 1234; best checkpoint = highest validation PESQ, validated
every 5000 steps from epoch 40, on the file list ``test.txt`` by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from pesq import pesq

from models.mpsenet_official.discriminator import MetricDiscriminator
from models.mpsenet_official.generator import MPSENetOfficial, rms_norm_factor

__all__ = [
    "LossWeights",
    "MPSENetBatch",
    "MPSENetLosses",
    "anti_wrapping_function",
    "batch_pesq",
    "cal_pesq",
    "complex_loss",
    "consistency_loss",
    "discriminator_loss",
    "generator_forward",
    "generator_loss",
    "magnitude_loss",
    "metric_loss",
    "mpsenet_losses",
    "phase_losses",
    "time_loss",
]


# --------------------------------------------------------------------------- #
# Individual terms                                                            #
# --------------------------------------------------------------------------- #


def anti_wrapping_function(x: torch.Tensor) -> torch.Tensor:
    """``|x - 2 pi round(x / 2 pi)|``."""
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def phase_losses(
    phase_r: torch.Tensor, phase_g: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Instantaneous-phase, group-delay (diff over F) and IAF (diff over T) losses; ``(B, F, T)``."""
    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(
        anti_wrapping_function(torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1))
    )
    iaf_loss = torch.mean(
        anti_wrapping_function(torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2))
    )
    return ip_loss, gd_loss, iaf_loss


def magnitude_loss(clean_mag: torch.Tensor, mag_g: torch.Tensor) -> torch.Tensor:
    """MSE between compressed magnitudes."""
    return F.mse_loss(clean_mag, mag_g)


def complex_loss(clean_com: torch.Tensor, com_g: torch.Tensor) -> torch.Tensor:
    """``2 * MSE`` over the stacked (real, imag) compressed spectra."""
    return F.mse_loss(clean_com, com_g) * 2


def consistency_loss(com_g: torch.Tensor, com_g_hat: torch.Tensor) -> torch.Tensor:
    """``2 * MSE`` between the generated spectrum and STFT(iSTFT(generated))."""
    return F.mse_loss(com_g, com_g_hat) * 2


def time_loss(clean_audio: torch.Tensor, audio_g: torch.Tensor) -> torch.Tensor:
    """L1 between (normalised) waveforms."""
    return F.l1_loss(clean_audio, audio_g)


def metric_loss(metric_g: torch.Tensor) -> torch.Tensor:
    """Generator adversarial term ``MSE(D(clean, enhanced), 1)``."""
    flat = metric_g.flatten()
    return F.mse_loss(flat, torch.ones_like(flat))


def discriminator_loss(
    metric_r: torch.Tensor, metric_g: torch.Tensor, pesq_target: torch.Tensor | None
) -> torch.Tensor:
    """``MSE(D(clean, clean), 1) + MSE(D(clean, enhanced), Q_pesq)``; the second term is 0 if ``pesq_target is None``."""
    flat_r = metric_r.flatten()
    loss = F.mse_loss(torch.ones_like(flat_r), flat_r)
    if pesq_target is not None:
        loss = loss + F.mse_loss(pesq_target.to(metric_g), metric_g.flatten())
    return loss


def cal_pesq(clean: np.ndarray, enhanced: np.ndarray, sr: int = 16000) -> float:
    """Wide-band PESQ (``pesq`` package), ``-1`` on any error (e.g. a silent segment)."""
    try:
        return pesq(sr, clean, enhanced, "wb")
    except Exception:  # noqa: BLE001 — upstream swallows every PESQ error
        return -1.0


def batch_pesq(clean: torch.Tensor, enhanced: torch.Tensor, sr: int = 16000) -> torch.Tensor | None:
    """Discriminator target ``(PESQ - 1) / 3.5`` per utterance, or ``None`` if any utterance fails.

    ``clean``/``enhanced``: ``(B, L)``. Runs on the CPU, sequentially (upstream
    parallelises the same calls with joblib).
    """
    c = clean.detach().float().cpu().numpy()
    e = enhanced.detach().float().cpu().numpy()
    scores = np.array([cal_pesq(ci, ei, sr) for ci, ei in zip(c, e, strict=True)])
    if -1 in scores:
        return None
    return torch.FloatTensor((scores - 1) / 3.5)


# --------------------------------------------------------------------------- #
# Batch-level API                                                             #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LossWeights:
    """Upstream ``train.py`` generator-loss weights."""

    magnitude: float = 0.9
    phase: float = 0.3
    complex: float = 0.1
    consistency: float = 0.1
    metric: float = 0.05
    time: float = 0.2


@dataclass
class MPSENetBatch:
    """Every tensor of one upstream training step (normalised domain)."""

    clean_audio: torch.Tensor
    clean_mag: torch.Tensor
    clean_pha: torch.Tensor
    clean_com: torch.Tensor
    mag_g: torch.Tensor
    pha_g: torch.Tensor
    com_g: torch.Tensor
    audio_g: torch.Tensor
    mag_g_hat: torch.Tensor
    com_g_hat: torch.Tensor


@dataclass
class MPSENetLosses:
    generator: torch.Tensor
    discriminator: torch.Tensor
    terms: dict[str, torch.Tensor] = field(default_factory=dict)
    pesq_target: torch.Tensor | None = None
    batch: MPSENetBatch | None = None


def _as_bl(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        if x.shape[1] != 1:
            raise ValueError(f"expected mono (B, 1, L), got {tuple(x.shape)}")
        x = x.squeeze(1)
    if x.dim() != 2:
        raise ValueError(f"expected (B, L) or (B, 1, L), got {tuple(x.shape)}")
    return x.float()


def generator_forward(
    model: MPSENetOfficial, noisy: torch.Tensor, clean: torch.Tensor
) -> MPSENetBatch:
    """Normalise by the noisy RMS factor, run the generator, re-analyse its output."""
    noisy = _as_bl(noisy)
    clean = _as_bl(clean)
    if noisy.shape != clean.shape:
        raise ValueError(f"noisy {tuple(noisy.shape)} and clean {tuple(clean.shape)} differ")
    with torch.autocast(device_type=noisy.device.type, enabled=False):
        norm = rms_norm_factor(noisy)
        clean_audio = clean * norm
        clean_mag, clean_pha, clean_com = model.stft(clean_audio)
        mag_g, pha_g, com_g, audio_g = model.spectra(noisy * norm, length=noisy.shape[-1])
        mag_g_hat, _, com_g_hat = model.stft(audio_g)
    return MPSENetBatch(
        clean_audio=clean_audio,
        clean_mag=clean_mag,
        clean_pha=clean_pha,
        clean_com=clean_com,
        mag_g=mag_g,
        pha_g=pha_g,
        com_g=com_g,
        audio_g=audio_g,
        mag_g_hat=mag_g_hat,
        com_g_hat=com_g_hat,
    )


def generator_loss(
    discriminator: MetricDiscriminator,
    batch: MPSENetBatch,
    weights: LossWeights = LossWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Upstream ``loss_gen_all`` and its individual terms."""
    loss_mag = magnitude_loss(batch.clean_mag, batch.mag_g)
    loss_ip, loss_gd, loss_iaf = phase_losses(batch.clean_pha, batch.pha_g)
    loss_pha = loss_ip + loss_gd + loss_iaf
    loss_com = complex_loss(batch.clean_com, batch.com_g)
    loss_con = consistency_loss(batch.com_g, batch.com_g_hat)
    loss_time = time_loss(batch.clean_audio, batch.audio_g)
    loss_metric = metric_loss(discriminator(batch.clean_mag, batch.mag_g_hat))
    total = (
        loss_mag * weights.magnitude
        + loss_pha * weights.phase
        + loss_com * weights.complex
        + loss_con * weights.consistency
        + loss_metric * weights.metric
        + loss_time * weights.time
    )
    terms = {
        "magnitude": loss_mag,
        "phase_ip": loss_ip,
        "phase_gd": loss_gd,
        "phase_iaf": loss_iaf,
        "phase": loss_pha,
        "complex": loss_com,
        "consistency": loss_con,
        "time": loss_time,
        "metric": loss_metric,
    }
    return total, terms


def mpsenet_losses(
    generator: MPSENetOfficial,
    discriminator: MetricDiscriminator,
    noisy: torch.Tensor,
    clean: torch.Tensor,
    *,
    weights: LossWeights = LossWeights(),
    pesq_target: torch.Tensor | None | str = "compute",
    sr: int = 16000,
) -> MPSENetLosses:
    """Generator and discriminator losses of one batch of raw ``(noisy, clean)`` waveforms.

    ``pesq_target="compute"`` evaluates :func:`batch_pesq` on the CPU; pass a
    ``(B,)`` tensor to supply it, or ``None`` to drop the fake-pair D term (as
    upstream does when PESQ fails). Call order matches upstream (D on the real
    pair, D on the detached fake pair, then D inside the generator loss).
    """
    batch = generator_forward(generator, noisy, clean)
    if isinstance(pesq_target, str):
        if pesq_target != "compute":
            raise ValueError(
                f"pesq_target must be 'compute', a tensor or None, got {pesq_target!r}"
            )
        pesq_target = batch_pesq(batch.clean_audio, batch.audio_g, sr)
    metric_r = discriminator(batch.clean_mag, batch.clean_mag)
    metric_g = discriminator(batch.clean_mag, batch.mag_g_hat.detach())
    loss_d = discriminator_loss(metric_r, metric_g, pesq_target)
    loss_g, terms = generator_loss(discriminator, batch, weights)
    return MPSENetLosses(
        generator=loss_g,
        discriminator=loss_d,
        terms=terms,
        pesq_target=pesq_target,
        batch=batch,
    )
