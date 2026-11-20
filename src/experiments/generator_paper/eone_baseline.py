"""EONE-RPS curve bank adapted from Doerfler, Kuntz & Zimmer (2026).

Doerfler, Kuntz, Zimmer, "Gradient-Based Learning of Parametric Engine Sound
Representations for Real-Time Resynthesis and Tuning on Embedded Systems",
AES 2026 International Conference on Automotive Audio, paper 27,
arXiv:2606.21521 (https://arxiv.org/html/2606.21521v1). The paper's EONE
model has four stages (Sec. II-B, Fig. 1): a VQT/BiGRU timbre
encoder-decoder pretrained on a proprietary engine corpus, a Softplus curve
bank ``G_rpm, G_trq in R^{K x P}`` (``P = 40``), a soft-mask control
projection (Eq. 1) and a 16 kHz differentiable synthesizer: ``K_h = 64``
harmonics at multipliers ``0.5, 1.0, ..., 32`` of the firing frequency
``RPM/120`` with cumulative-sum phase plus ONE random ``phi0`` per chunk
(Eqs. 2-3), and ``K_b = 64`` ERB-spaced cosine-filtered random-phase IRFFT
noise bands that loop with the synthesis chunk (Eqs. 4-5); the output is
their sum (Eq. 6). For a target engine, only the curve bank is fine-tuned
(Sec. II-D2) against a multi-resolution STFT loss (FFT 32768 down to 32,
75 % overlap, Hann; spectral convergence, linear, log and energy terms with
equal weights and scale-invariant normalisation) plus a harmonic energy
loss (FFT 65536, window 16384, hop 256) along RPM-derived order tracks
(Sec. II-D3) and a curvature regulariser on second differences of the curve
points (Sec. II-D2). At deployment the projection collapses to a table
lookup: the exported positive curves ARE the synthesizer (Sec. II-C).

This module (method id ``doerfler2026_eone``, display "EONE-RPS curve bank")
is an ADAPTED, DIRECTLY FITTED SOURCE-STAGE CONTROL. It is not the authors'
code, has no pretrained encoder/decoder (none is published; no checkpoint,
encoder or warm-start proxy is fabricated), no torque input, and reproduces
no published number. What it implements completely is the curve bank, the
control projection, the synthesizer, the losses, the curvature regulariser,
the direct curve-bank optimisation and a CPU deployment renderer that runs
the SAME torch synthesis code with the exported positive curves.

Fixed engine-to-rotor adaptation (decided before any evaluation):

* Sources: every rotor ``r`` of the rig is an independent EONE source driven
  by its own supplied RPS track (``RPM_r = 60 * RPS_r``). Each of the ``M``
  microphones mixes all rotors with its OWN gain curves; a rotor's harmonic
  carriers and noise loops are shared by all microphones. No propagation
  delay or spatial claim.
* Curves: ``P = n_knots`` (40) knots uniformly spanning ``[0, rpm_max]`` =
  ``[0, 10000]`` RPM (the paper's ``RPM * 1e-4`` normalisation), curve value
  ``G = s_m * softplus(theta)`` per (microphone, rotor, component, knot). The
  per-microphone calibration scale ``s_m`` (calibration RMS) is folded into
  the exported curves. RPM outside the range holds the end knot. The paper's
  torque curve is unavailable and FIXED to the multiplicative identity
  ``A_trq = 1`` (not learned, not proxied).
* Harmonics: ``n_orders`` (128) integer SHAFT orders ``k`` per rotor,
  equivalently blade-pass (2 x RPS) multipliers ``0.5, 1.0, ..., 64`` - the
  paper's half-step multiplier set extended from 64 to 128 components so the
  rotor comb covers the benchmark band (the GP control uses 120 shaft
  orders). Engine ``RPM/120`` is never applied to drone RPS. Phase is
  sample-accurate from the supplied 16 kHz RPS, Eq. 2 with an inclusive sum:
  ``phi_rk[n] = (k * 2 pi * sum_{t<=n} RPS_r[t] / sr mod 2 pi) + phi0_r``,
  accumulated in float64; ONE uniform ``phi0_r`` per rotor and chunk is
  added to every order (not ``k * phi0``). Curves do not learn phases.
* Anti-alias gate (not in the paper): carrier ``k`` of rotor ``r`` is
  multiplied per sample by a raised-cosine gate of its instantaneous
  frequency ``f = k * RPS_r[n]``: 1 for ``f <= sr/2 - alias_transition_hz``
  (7900 Hz, the repository anti-alias pass edge
  ``resample.AA_PASS_HZ``), ``0.5 * (1 + cos(pi * x))`` across the 100 Hz
  transition, exactly 0 for ``f >= sr/2``. No component at or above Nyquist
  is ever synthesised, so nothing folds.
* Noise: ``n_bands`` (64) cosine lobes per rotor, Eq. 4 with ERB rate
  ``E(f) = 21.4 log10(1 + 0.00437 f)``: centres ``c_b = b * s``,
  ``b = 0..K_b-1``, ``s = E(sr/2) / (K_b - 1)``, ``Delta_b = 2 s``, lobe
  ``cos(pi (E - c_b) / Delta_b)`` for ``|E - c_b| < s`` and ZERO outside, so
  adjacent lobes overlap by half and are power complementary (the paper's
  lobe support, spacing and normalisation are not stated). Each loop is one
  ``noise_loop`` = 65536-sample period (the paper's 4.096 s chunk) of an
  independent uniform-random-phase spectrum through ``irfft``; DC and
  Nyquist bins are zero and the lobe magnitudes are scaled so every loop is
  EXACTLY zero-mean and unit-RMS over its period (Parseval) whatever its
  phases. This fixes the otherwise free amplitude unit; the curve gains are
  free and absorb it, and the level does not depend on the requested
  duration. A request of any length reads the periodic loop from one random
  offset per rotor (crop or circular tiling).
* Control frames: the paper's hop of 256 samples is 16 ms at 16 kHz (its
  parenthetical "64 ms" is inconsistent). Frame ``j`` holds the mean RPS of
  samples ``[j*hop, (j+1)*hop)`` (last frame: the samples present), placed at
  its centre ``j*hop + (hop-1)/2``. Its soft knot mask (two-neighbour linear
  interpolation, Eq. 1) gives per-frame curve activations, linearly
  interpolated to every sample between neighbouring frame centres and held
  before the first / after the last centre. Training and rendering share
  this torch code (:func:`_synthesize`).

Calibration (:func:`fit_eone`, calibration windows only): every full,
non-overlapping ``piece_s`` (4 s) piece of the 16 kHz calibration audio
(21 x 8 s FLY125 windows -> 42 pieces); paper pretraining used 4.096 s / 50 %
overlap, direct fine-tune segmentation is unstated. Flat initial curves:
half of each microphone's calibration power to the harmonics, half to the
noise, equally over rotors and components (nominal counts, before gating).
One step = one piece with all microphones jointly; ``epochs`` (50) passes in
seeded random order; AdamW (``weight_decay`` 0: decay of pre-Softplus
parameters would pull every gain towards the arbitrary ``s_m softplus(0)``;
smoothness is the curvature term's job) with torch ``OneCycleLR`` cosine
annealing to ``max_lr`` 0.01. Losses, exactly (``S`` STFT magnitudes,
``g`` generated, ``t`` target, per microphone, then averaged):

* MRSTFT: for every ``n`` in ``mrstft_max_fft, /2, ..., mrstft_min_fft``,
  periodic Hann length ``n``, hop ``n/4``, centred zero-padded frames:
  ``SC = ||S_g - S_t||_F / ||S_t||_F``, ``LIN = ||S_g - S_t||_1 / ||S_t||_1``,
  ``LOG = mean |ln(S_g + e) - ln(S_t + e)|`` with
  ``e = log_floor_rel * mean(S_t)`` (-60 dB re the mean target magnitude),
  ``EN = sum_tau |E_g - E_t| / sum_tau E_t`` with frame energy
  ``E = sum_f S^2``; ``L_mr = mean_n mean(SC, LIN, LOG, EN)``.
* Harmonic: STFT ``harmonic_n_fft`` / periodic Hann ``harmonic_win`` /
  ``harmonic_hop``. Track ``(r, k)`` in frame ``tau`` sits at
  ``k * RPS_r`` with ``RPS_r`` the Hann-weighted mean of the supplied RPS
  under that frame's window; its energy sums ``|X|^2`` over
  ``+-harmonic_halfwidth_bins`` (4 bins = +-0.98 Hz, the -6 dB half-width of
  the 16384 Hann main lobe on the 65536 grid) around the nearest bin. A track
  counts only if its whole mask lies in ``[1, n_fft/2 - 1]`` bins, its upper
  mask edge and ``k * max RPS`` under the window are at or below the gate
  pass edge. ``L_h = mean_valid |ln(E_g + e_m) - ln(E_t + e_m)|``,
  ``e_m = energy_floor_rel * mean_valid E_t`` (-60 dB); weight 1.0.
* Curvature: on the unscaled positive curves ``u = softplus(theta)``,
  ``L_c = mean_curves [mean_p (u_{p+1} - 2 u_p + u_{p-1})^2 / mean_p u_p^2]``
  (per-curve scale invariant); weight 0.001.

The best epoch minimises the calibration objective re-evaluated after every
epoch (and at initialisation, epoch 0) with FIXED per-piece random draws, so
epochs compare like with like; its curves are exported. Nothing from the
evaluation split is loaded, and :func:`render_eone` takes no target: no
RMS/gain matching.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.version import cuda as torch_cuda

from data_processing.noise_model.resample import AA_PASS_HZ, AA_STOP_HZ
from experiments.generator_paper.cases import FlightClip
from experiments.generator_paper.metrics import _jsonable

SCHEMA = "generator-paper-doerfler2026-eone/1"
EONE_METHOD_ID = "doerfler2026_eone"
EONE_DISPLAY_NAME = "EONE-RPS curve bank"
ARXIV_ID = "2606.21521"
BIBLIOGRAPHY_UUID = "3c5f134f-9419-5b72-9743-9a455063093b"
REFERENCE = (
    "Doerfler, Kuntz & Zimmer (2026), Gradient-Based Learning of Parametric Engine Sound "
    "Representations for Real-Time Resynthesis and Tuning on Embedded Systems, AES 2026 "
    "International Conference on Automotive Audio, paper 27; arXiv:2606.21521"
)
SUPPORTED_RIGS = ("md2",)
BLADES = 2
_REPO_ROOT = Path(__file__).resolve().parents[3]
#: Repository-relative sources this comparator executes (hash inventory).
EONE_SOURCE_FILES = (
    "src/experiments/generator_paper/eone_baseline.py",
    "src/experiments/generator_paper/cases.py",
    "src/experiments/generator_paper/metrics.py",
    "src/data_processing/noise_model/resample.py",
)
_SOURCE_ROLES = {
    "src/experiments/generator_paper/eone_baseline.py": "this adapter (curve bank, synthesis, losses, fit, render)",
    "src/experiments/generator_paper/cases.py": "FlightClip calibration windows",
    "src/experiments/generator_paper/metrics.py": "_jsonable report serialisation",
    "src/data_processing/noise_model/resample.py": "AA_PASS_HZ/AA_STOP_HZ anti-alias edges of the benchmark audio",
}
DEVIATIONS = (
    "Initialisation: no pretrained VQT/BiGRU timbre encoder-decoder exists publicly, so the curve "
    "bank starts flat from per-microphone calibration RMS (half harmonic, half noise power, equal "
    "over rotors and components) instead of the averaged decoder output over a warm-up set; this "
    "is the direct fine-tuning stage only. No surrogate encoder or checkpoint is used.",
    "Torque: no torque signal exists for the rotors; the torque gain factor is fixed to 1 "
    "(A_trq = 1), so each component has one RPM gain curve.",
    "Carrier: four independent rotor sources indexed by their own RPM = 60 x RPS, with 128 integer "
    "shaft orders (blade-pass multipliers 0.5..64) instead of 64 multipliers 0.5..32 of the engine "
    "firing frequency RPM/120.",
    "Multichannel: eight per-microphone gain-curve sets mix shared per-rotor excitations; the paper "
    "models one mono target per fit. No propagation delays or spatial extrapolation.",
    "Anti-alias gate: raised-cosine gate of each carrier's instantaneous frequency between the "
    "benchmark anti-alias pass edge (sr/2 - 100 Hz) and Nyquist; not described in the paper.",
    "Noise bank details the paper omits: centres uniformly spaced in ERB rate from 0 to E(sr/2), "
    "lobe width twice the spacing (power-complementary half-overlap), zero DC/Nyquist bins, exact "
    "unit-RMS loops of 65536 samples, one random read offset per rotor and call.",
    "Control frames: hop 256 = 16 ms at 16 kHz (paper's 64 ms parenthetical is inconsistent); frame "
    "mean RPS at frame centres, linear interpolation between centres, edge hold; RPM clamped to "
    "[0, 10000].",
    "Data: 42 non-overlapping 4-s pieces of the 21 FLY125 8-s calibration windows (8 microphones "
    "jointly per step) instead of 4.096-s chunks with 50 % overlap in batches of 16.",
    "Loss completion: stated formulas for SC/LIN/LOG/energy terms, -60 dB relative log floors, "
    "+-4-bin harmonic masks on Hann-weighted mean order tracks, per-curve normalised curvature; the "
    "paper does not give these details.",
    "Optimisation: 50 epochs, AdamW without weight decay, one-cycle cosine schedule to 0.01, best "
    "epoch by a fixed-draw calibration objective; the paper's fine-tuning epochs/learning rate are "
    "unstated (its pretraining uses 200 epochs and one-cycle cosine AdamW).",
    "Evaluation: the repository generator-paper protocol replaces the paper's LSD and listening "
    "test; no published EONE number is reproduced or comparable.",
)
#: Render/training synthesis processes this many 'hop'-sample blocks at a
#: time (bounds carrier memory; does not change the result).
_CHUNK_BLOCKS = 64
_TRAIN_DOMAIN, _EVAL_DOMAIN, _PROBE_DOMAIN = 1, 2, 3
_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class EoneSettings:
    """Fit/render settings. Defaults are the fixed production design.

    Smaller values exist only so a tiny real smoke can exercise the code on
    CPU; the report records every value and ``production_settings``.
    """

    sample_rate: int = 16000
    piece_s: float = 4.0
    n_orders: int = 128
    n_bands: int = 64
    n_knots: int = 40
    rpm_max: float = 10000.0
    hop: int = 256
    noise_loop: int = 65536
    alias_transition_hz: float = AA_STOP_HZ - AA_PASS_HZ
    epochs: int = 50
    max_lr: float = 0.01
    weight_decay: float = 0.0
    onecycle_pct_start: float = 0.3
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1e4
    mrstft_max_fft: int = 32768
    mrstft_min_fft: int = 32
    log_floor_rel: float = 1e-3
    harmonic_n_fft: int = 65536
    harmonic_win: int = 16384
    harmonic_hop: int = 256
    harmonic_halfwidth_bins: int = 4
    energy_floor_rel: float = 1e-6
    harmonic_weight: float = 1.0
    curvature_weight: float = 1e-3

    def __post_init__(self) -> None:
        ints = (
            "sample_rate",
            "n_orders",
            "n_bands",
            "n_knots",
            "hop",
            "noise_loop",
            "epochs",
            "mrstft_max_fft",
            "mrstft_min_fft",
            "harmonic_n_fft",
            "harmonic_win",
            "harmonic_hop",
        )
        for name in ints:
            v = getattr(self, name)
            if isinstance(v, bool) or int(v) != v or v < 1:
                raise ValueError(f"EoneSettings.{name} must be a positive integer")
        hw = self.harmonic_halfwidth_bins
        if isinstance(hw, bool) or int(hw) != hw or hw < 0:
            raise ValueError("EoneSettings.harmonic_halfwidth_bins must be a non-negative integer")
        for name in (
            "piece_s",
            "rpm_max",
            "max_lr",
            "onecycle_div_factor",
            "onecycle_final_div_factor",
            "log_floor_rel",
            "energy_floor_rel",
            "alias_transition_hz",
        ):
            v = float(getattr(self, name))
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"EoneSettings.{name} must be positive and finite")
        for name in ("weight_decay", "harmonic_weight", "curvature_weight"):
            v = float(getattr(self, name))
            if not (math.isfinite(v) and v >= 0.0):
                raise ValueError(f"EoneSettings.{name} must be non-negative and finite")
        if not 0.0 < self.onecycle_pct_start < 1.0:
            raise ValueError("EoneSettings.onecycle_pct_start must lie in (0, 1)")
        if self.n_bands < 2 or self.n_knots < 3:
            raise ValueError("EoneSettings needs n_bands >= 2 and n_knots >= 3 (curvature)")
        if self.noise_loop % 2 or self.harmonic_n_fft % 2:
            raise ValueError("noise_loop and harmonic_n_fft must be even")
        for name in ("mrstft_max_fft", "mrstft_min_fft"):
            v = getattr(self, name)
            if v < 4 or v & (v - 1):
                raise ValueError(f"EoneSettings.{name} must be a power of two >= 4")
        if self.mrstft_min_fft > self.mrstft_max_fft:
            raise ValueError("mrstft_min_fft exceeds mrstft_max_fft")
        if self.harmonic_win > self.harmonic_n_fft:
            raise ValueError("harmonic_win exceeds harmonic_n_fft")
        if self.alias_transition_hz >= self.sample_rate / 2:
            raise ValueError("alias_transition_hz must be below Nyquist")
        n_piece = self.piece_s * self.sample_rate
        if abs(n_piece - round(n_piece)) > 1e-6:
            raise ValueError(f"piece_s={self.piece_s} is not a whole number of samples")

    @property
    def piece_samples(self) -> int:
        return int(round(self.piece_s * self.sample_rate))

    @property
    def n_components(self) -> int:
        return self.n_orders + self.n_bands

    @property
    def pass_hz(self) -> float:
        return self.sample_rate / 2.0 - self.alias_transition_hz

    @property
    def mrstft_ffts(self) -> tuple[int, ...]:
        out, n = [], self.mrstft_max_fft
        while n >= self.mrstft_min_fft:
            out.append(n)
            n //= 2
        return tuple(out)


# ── representation: knots, ERB bank, statics ───────────────────────────────


def knot_rpm(settings: EoneSettings) -> np.ndarray:
    """``(P,)`` RPM of the curve knots, uniformly ``0..rpm_max`` inclusive."""
    return np.linspace(0.0, settings.rpm_max, settings.n_knots)


def _erb_rate(f: np.ndarray | float) -> np.ndarray:
    return 21.4 * np.log10(1.0 + 0.00437 * np.asarray(f, dtype=np.float64))


def _erb_rate_inv(e: np.ndarray) -> np.ndarray:
    return (10.0 ** (np.asarray(e, dtype=np.float64) / 21.4) - 1.0) / 0.00437


def noise_band_layout(settings: EoneSettings) -> dict[str, Any]:
    """ERB-rate centres and nonzero supports (Hz) of the noise-band lobes."""
    e_max = float(_erb_rate(settings.sample_rate / 2.0))
    s = e_max / (settings.n_bands - 1)
    centres = np.arange(settings.n_bands) * s
    return {
        "centre_erb": centres,
        "half_support_erb": s,
        "delta_erb": 2.0 * s,
        "centre_hz": _erb_rate_inv(centres),
        "lower_hz": _erb_rate_inv(np.clip(centres - s, 0.0, e_max)),
        "upper_hz": _erb_rate_inv(np.clip(centres + s, 0.0, e_max)),
    }


def _band_magnitudes(settings: EoneSettings) -> np.ndarray:
    """``(K_b, L/2+1)`` lobe magnitudes, each scaled to an exactly unit-RMS loop."""
    n = settings.noise_loop
    lay = noise_band_layout(settings)
    e = _erb_rate(np.arange(n // 2 + 1) * settings.sample_rate / n)
    x = (e[None, :] - lay["centre_erb"][:, None]) / lay["delta_erb"]
    mag = np.where(np.abs(x) < 0.5, np.cos(np.pi * x), 0.0)
    mag[:, 0] = 0.0
    mag[:, -1] = 0.0
    # irfft (1/L normalisation) of zero-DC/Nyquist spectra: rms^2 = 2 sum |X_k|^2 / L^2.
    power = 2.0 * np.sum(mag**2, axis=1) / float(n) ** 2
    if not np.all(power > 0):
        empty = np.flatnonzero(~(power > 0)).tolist()
        raise ValueError(f"noise bands {empty} have no bin on the {n}-sample loop grid")
    return mag / np.sqrt(power)[:, None]


@dataclass(frozen=True)
class _Statics:
    orders64: torch.Tensor
    orders32: torch.Tensor
    alpha: torch.Tensor
    band_mag: torch.Tensor
    pass_hz: float
    stop_hz: float
    mr_windows: tuple[tuple[int, torch.Tensor], ...]
    harm_window: torch.Tensor


def _statics(cfg: EoneSettings, device: torch.device) -> _Statics:
    orders = torch.arange(1, cfg.n_orders + 1, dtype=torch.float64, device=device)
    hop = cfg.hop
    alpha = (torch.arange(hop, dtype=torch.float64) + hop // 2 - (hop - 1) / 2.0) / hop
    return _Statics(
        orders64=orders,
        orders32=orders.to(torch.float32),
        alpha=alpha.to(device=device, dtype=torch.float32),
        band_mag=torch.as_tensor(_band_magnitudes(cfg), dtype=torch.float32, device=device),
        pass_hz=cfg.pass_hz,
        stop_hz=cfg.sample_rate / 2.0,
        mr_windows=tuple(
            (n, torch.hann_window(n, periodic=True, device=device)) for n in cfg.mrstft_ffts
        ),
        harm_window=torch.hann_window(cfg.harmonic_win, periodic=True, device=device),
    )


# ── control projection ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Controls:
    """Fixed per-track control data on the block grid.

    The virtual time axis starts ``lead = hop - hop//2`` samples before
    sample 0, so block ``b`` spans the samples between frame centres
    ``b - 1`` and ``b`` (frame ``-1`` and ``F`` mean edge hold).
    """

    n: int
    n_blocks: int
    frame_rps: np.ndarray  # (R, F)
    mask: torch.Tensor  # (R, F, P) float32 soft knot masks
    rps: torch.Tensor  # (R, n_blocks, hop) float32 per-sample RPS, 0 outside the track
    phase: torch.Tensor  # (R, n_blocks, hop) float64 shaft phase [rad], 0 outside
    index: torch.Tensor  # (n_blocks, hop) int64 real sample index


def _knot_mask(rpm: np.ndarray, cfg: EoneSettings) -> np.ndarray:
    """Eq. 1 soft masks ``(..., P)``: linear interpolation between the two nearest knots."""
    p = cfg.n_knots
    u = np.clip(rpm / cfg.rpm_max, 0.0, 1.0) * (p - 1)
    i0 = np.minimum(np.floor(u).astype(np.int64), p - 2)
    w = u - i0
    mask = np.zeros(rpm.shape + (p,))
    np.put_along_axis(mask, i0[..., None], (1.0 - w)[..., None], axis=-1)
    np.put_along_axis(mask, i0[..., None] + 1, w[..., None], axis=-1)
    return mask


def _controls(rps: np.ndarray, cfg: EoneSettings, device: torch.device) -> _Controls:
    track = np.asarray(rps, dtype=np.float64)
    n_rot, n = track.shape
    hop = cfg.hop
    lead = hop - hop // 2
    n_blocks = -(-(n + lead) // hop)
    n_frames = -(-n // hop)
    padded = np.zeros((n_rot, n_frames * hop))
    padded[:, :n] = track
    counts = np.minimum(hop, n - np.arange(n_frames) * hop)
    frame_rps = padded.reshape(n_rot, n_frames, hop).sum(axis=-1) / counts
    mask = _knot_mask(60.0 * frame_rps, cfg)
    virt = n_blocks * hop
    rps_v = np.zeros((n_rot, virt))
    rps_v[:, lead : lead + n] = track
    ph_v = np.zeros((n_rot, virt))
    ph_v[:, lead : lead + n] = np.cumsum(track, axis=1) * (_TWO_PI / cfg.sample_rate)
    index = (np.arange(virt) - lead).reshape(n_blocks, hop)
    return _Controls(
        n=n,
        n_blocks=n_blocks,
        frame_rps=frame_rps,
        mask=torch.as_tensor(mask, dtype=torch.float32, device=device),
        rps=torch.as_tensor(
            rps_v.reshape(n_rot, n_blocks, hop), dtype=torch.float32, device=device
        ),
        phase=torch.as_tensor(
            ph_v.reshape(n_rot, n_blocks, hop), dtype=torch.float64, device=device
        ),
        index=torch.as_tensor(index, dtype=torch.int64, device=device),
    )


# ── differentiable synthesis (shared by training and rendering) ────────────


def _alias_gate(f: torch.Tensor, pass_hz: float, stop_hz: float) -> torch.Tensor:
    x = ((f - pass_hz) / (stop_hz - pass_hz)).clamp(0.0, 1.0)
    g = 0.5 + 0.5 * torch.cos(math.pi * x)
    return torch.where(f >= stop_hz, torch.zeros_like(g), g)


def _harmonic_carriers(
    ctl: _Controls, b0: int, b1: int, phi0: torch.Tensor, st: _Statics
) -> torch.Tensor:
    """``(nb, R*K_h, hop)`` gated ``cos(k Phi_r + phi0_r)``."""
    ph = ctl.phase[:, b0:b1].transpose(0, 1).unsqueeze(2)  # (nb, R, 1, hop)
    arg = torch.remainder(ph * st.orders64.view(1, 1, -1, 1), _TWO_PI).to(torch.float32)
    car = torch.cos(arg + phi0.view(1, -1, 1, 1))
    f = ctl.rps[:, b0:b1].transpose(0, 1).unsqueeze(2) * st.orders32.view(1, 1, -1, 1)
    car = car * _alias_gate(f, st.pass_hz, st.stop_hz)
    nb, n_rot, n_h, hop = car.shape
    return car.reshape(nb, n_rot * n_h, hop)


def _noise_carriers(
    ctl: _Controls, b0: int, b1: int, loops: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """``(nb, R*K_b, hop)``: periodic loops read from one offset per rotor."""
    n_rot, n_b, n_loop = loops.shape
    hop = ctl.index.shape[1]
    idx = ctl.index[b0:b1].reshape(-1)
    car = torch.stack(
        [loops[r].index_select(1, torch.remainder(idx + offsets[r], n_loop)) for r in range(n_rot)]
    )
    return car.view(n_rot, n_b, b1 - b0, hop).permute(2, 0, 1, 3).reshape(b1 - b0, n_rot * n_b, hop)


def _noise_loops(band_mag: torch.Tensor, theta: torch.Tensor, n_loop: int) -> torch.Tensor:
    """``(R, K_b, L)`` unit-RMS periodic band loops from random phases ``(R, K_b, L/2+1)``."""
    spec = torch.complex(band_mag * torch.cos(theta), band_mag * torch.sin(theta))
    return torch.fft.irfft(spec, n=n_loop, dim=-1)


def _mix(amp: torch.Tensor, car: torch.Tensor, b0: int, alpha: torch.Tensor) -> torch.Tensor:
    """``(M, nb, hop)``: frame activations linearly interpolated between frame centres times carriers."""
    nb = car.shape[0]
    left = amp[:, :, b0 : b0 + nb].permute(2, 0, 1)
    right = amp[:, :, b0 + 1 : b0 + nb + 1].permute(2, 0, 1)
    y = torch.bmm(left, car) * (1.0 - alpha) + torch.bmm(right, car) * alpha
    return y.transpose(0, 1)


def _synthesize(
    curves: torch.Tensor,
    ctl: _Controls,
    phi0: torch.Tensor,
    loops: torch.Tensor,
    offsets: torch.Tensor,
    st: _Statics,
) -> torch.Tensor:
    """Eqs. 1-6 for all microphones: ``(M, n)`` from positive curves ``(M, R, K, P)``."""
    n_mic, n_rot, n_comp, _ = curves.shape
    n_h = st.orders64.numel()
    n_b = n_comp - n_h
    amp = torch.einsum("rfp,mrkp->mrkf", ctl.mask, curves)
    amp = torch.cat((amp[..., :1], amp, amp[..., -1:]), dim=-1)  # edge hold
    amp_h = amp[:, :, :n_h].reshape(n_mic, n_rot * n_h, -1)
    amp_n = amp[:, :, n_h:].reshape(n_mic, n_rot * n_b, -1)
    out = []
    for b0 in range(0, ctl.n_blocks, _CHUNK_BLOCKS):
        b1 = min(ctl.n_blocks, b0 + _CHUNK_BLOCKS)
        y = _mix(amp_h, _harmonic_carriers(ctl, b0, b1, phi0, st), b0, st.alpha)
        y = y + _mix(amp_n, _noise_carriers(ctl, b0, b1, loops, offsets), b0, st.alpha)
        out.append(y)
    hop = ctl.index.shape[1]
    lead = hop - hop // 2
    return torch.cat(out, dim=1).reshape(n_mic, -1)[:, lead : lead + ctl.n]


# ── losses ─────────────────────────────────────────────────────────────────


def _stft_mag(x: torch.Tensor, n: int, window: torch.Tensor) -> torch.Tensor:
    return torch.stft(
        x,
        n_fft=n,
        hop_length=n // 4,
        win_length=n,
        window=window,
        center=True,
        pad_mode="constant",
        return_complex=True,
    ).abs()


def _mrstft_terms(
    gen: torch.Tensor, tgt: torch.Tensor, st: _Statics, cfg: EoneSettings
) -> torch.Tensor:
    """``(4,)`` = (SC, LIN, LOG, EN), each averaged over microphones and scales."""
    acc = []
    for n, win in st.mr_windows:
        sg = _stft_mag(gen, n, win)
        with torch.no_grad():
            stg = _stft_mag(tgt, n, win)
        diff = sg - stg
        sc = diff.square().sum((1, 2)).sqrt() / stg.square().sum((1, 2)).sqrt()
        lin = diff.abs().sum((1, 2)) / stg.sum((1, 2))
        eps = cfg.log_floor_rel * stg.mean((1, 2), keepdim=True)
        log = (torch.log(sg + eps) - torch.log(stg + eps)).abs().mean((1, 2))
        e_g, e_t = sg.square().sum(1), stg.square().sum(1)
        en = (e_g - e_t).abs().sum(1) / e_t.sum(1)
        acc.append(torch.stack((sc, lin, log, en)).mean(1))
    return torch.stack(acc).mean(0)


@dataclass(frozen=True)
class _HarmTarget:
    bins: torch.Tensor  # (F_h, R*K_h) int64 centre bins, clamped in range
    valid: torch.Tensor  # (F_h, R*K_h) bool
    energy: torch.Tensor  # (M, F_h, R*K_h) float32 target track energies
    floor: torch.Tensor  # (M, 1, 1)
    n_valid: int


def _track_energy(
    x: torch.Tensor, bins: torch.Tensor, st: _Statics, cfg: EoneSettings
) -> torch.Tensor:
    spec = torch.stft(
        x,
        n_fft=cfg.harmonic_n_fft,
        hop_length=cfg.harmonic_hop,
        win_length=cfg.harmonic_win,
        window=st.harm_window,
        center=True,
        pad_mode="constant",
        return_complex=True,
    )
    power = (spec.real.square() + spec.imag.square()).transpose(1, 2)  # (M, F_h, bins)
    if power.shape[1] != bins.shape[0]:
        raise RuntimeError(f"harmonic STFT has {power.shape[1]} frames, tracks {bins.shape[0]}")
    hw = cfg.harmonic_halfwidth_bins
    offs = torch.arange(-hw, hw + 1, device=bins.device)
    idx = (bins.unsqueeze(-1) + offs).reshape(bins.shape[0], -1)
    got = power.gather(2, idx.unsqueeze(0).expand(power.shape[0], -1, -1))
    return got.view(power.shape[0], bins.shape[0], bins.shape[1], 2 * hw + 1).sum(-1)


def _harm_target(
    audio: torch.Tensor, rps: np.ndarray, st: _Statics, cfg: EoneSettings
) -> _HarmTarget:
    """Order tracks under each harmonic-STFT window and the target's masked energies."""
    n_rot, n = rps.shape
    nfft, win, hop, hw = (
        cfg.harmonic_n_fft,
        cfg.harmonic_win,
        cfg.harmonic_hop,
        cfg.harmonic_halfwidth_bins,
    )
    n_frames = 1 + n // hop
    n_bins = nfft // 2 + 1
    # torch.stft(center=True) frame tau covers samples [tau*hop - left, tau*hop - left + win).
    left = nfft // 2 - (nfft - win) // 2
    right = max(0, (n_frames - 1) * hop + win - left - n)
    w = torch.hann_window(win, periodic=True, dtype=torch.float64).view(1, 1, -1)
    x = torch.as_tensor(rps, dtype=torch.float64).unsqueeze(1)
    num = F.conv1d(F.pad(x, (left, right)), w, stride=hop)[..., :n_frames]
    den = F.conv1d(F.pad(torch.ones(1, 1, n, dtype=torch.float64), (left, right)), w, stride=hop)[
        ..., :n_frames
    ]
    mean_rps = (num / den).squeeze(1)  # (R, F_h)
    max_rps = F.max_pool1d(F.pad(x, (left, right), value=-math.inf), win, stride=hop)[
        ..., :n_frames
    ].squeeze(1)
    k = torch.arange(1, cfg.n_orders + 1, dtype=torch.float64).view(1, -1, 1)
    freq = mean_rps.unsqueeze(1) * k  # (R, K_h, F_h)
    bins = torch.round(freq * nfft / cfg.sample_rate).to(torch.int64)
    valid = (
        (bins - hw >= 1)
        & (bins + hw <= n_bins - 2)
        & ((bins + hw) * (cfg.sample_rate / nfft) <= cfg.pass_hz)
        & (max_rps.unsqueeze(1) * k <= cfg.pass_hz)
    )
    bins = bins.clamp(hw, n_bins - 1 - hw)
    bins = bins.permute(2, 0, 1).reshape(n_frames, n_rot * cfg.n_orders).to(audio.device)
    valid = valid.permute(2, 0, 1).reshape(n_frames, n_rot * cfg.n_orders).to(audio.device)
    with torch.no_grad():
        energy = _track_energy(audio, bins, st, cfg)
    n_valid = int(valid.sum())
    if n_valid:
        mean = (energy * valid).sum((1, 2), keepdim=True) / n_valid
        floor = (cfg.energy_floor_rel * mean).clamp_min(torch.finfo(torch.float32).tiny)
    else:
        floor = torch.full(
            (audio.shape[0], 1, 1), torch.finfo(torch.float32).tiny, device=audio.device
        )
    return _HarmTarget(bins=bins, valid=valid, energy=energy, floor=floor, n_valid=n_valid)


def _harmonic_loss(
    gen: torch.Tensor, ht: _HarmTarget, st: _Statics, cfg: EoneSettings
) -> torch.Tensor:
    if ht.n_valid == 0:
        return gen.new_zeros(())
    e_g = _track_energy(gen, ht.bins, st, cfg)
    d = (torch.log(e_g + ht.floor) - torch.log(ht.energy + ht.floor)).abs()
    return (d * ht.valid).sum() / (ht.n_valid * gen.shape[0])


def _curvature(u: torch.Tensor) -> torch.Tensor:
    d2 = u[..., 2:] - 2.0 * u[..., 1:-1] + u[..., :-2]
    return (d2.square().mean(-1) / u.square().mean(-1)).mean()


# ── fitted model ───────────────────────────────────────────────────────────


class EoneFit:
    """Exported EONE-RPS curve bank. Build with :func:`fit_eone` or :meth:`load`.

    ``curves`` ``(M, R, K_h + K_b, P)`` float32 are the positive RPM gain
    curves (waveform units of the calibration audio: harmonic entries are
    cosine peak amplitudes, noise entries the RMS of a unit-RMS band loop) at
    :func:`knot_rpm`; the torque factor is identically 1.
    """

    def __init__(
        self,
        *,
        settings: EoneSettings,
        curves: np.ndarray,
        report: dict[str, Any],
        checkpoint_sha256: str | None = None,
    ) -> None:
        c = np.array(curves, dtype=np.float32)
        if c.ndim != 4 or c.shape[2:] != (settings.n_components, settings.n_knots):
            raise ValueError(
                f"curves must be (M, R, {settings.n_components}, {settings.n_knots}), got {c.shape}"
            )
        if not (np.isfinite(c).all() and (c >= 0).all()):
            raise ValueError("curves must be finite and non-negative")
        c.flags.writeable = False
        self.settings = settings
        self.curves = c
        self.n_mics, self.n_rotors = int(c.shape[0]), int(c.shape[1])
        self.sample_rate = int(settings.sample_rate)
        self.report = report
        self.checkpoint_sha256 = checkpoint_sha256
        self._statics = _statics(settings, torch.device("cpu"))

    @property
    def knot_rpm(self) -> np.ndarray:
        return knot_rpm(self.settings)

    @property
    def harmonic_curves(self) -> np.ndarray:
        """``(M, R, K_h, P)``: shaft order ``k = 1..K_h`` (index ``k - 1``)."""
        return self.curves[:, :, : self.settings.n_orders]

    @property
    def noise_curves(self) -> np.ndarray:
        """``(M, R, K_b, P)`` in :func:`noise_band_layout` order."""
        return self.curves[:, :, self.settings.n_orders :]

    @classmethod
    def load(cls, path: Path) -> EoneFit:
        """Load a :func:`fit_eone` checkpoint (``torch.load(weights_only=True)``).

        The stored fit-time probe render is recomputed with the same seed and
        must agree, so a checkpoint that does not reproduce its renderer is
        refused.
        """
        path = Path(path)
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        if payload.get("schema") != SCHEMA:
            raise ValueError(f"{path.name}: schema {payload.get('schema')!r} is not {SCHEMA!r}")
        if float(payload["torque_factor"]) != 1.0:
            raise ValueError(f"{path.name}: torque factor must be the fixed identity 1")
        report = json.loads(payload["report_json"])
        settings = EoneSettings(**payload["settings"])
        training = report.get("training", {})
        if (
            training.get("completed_epochs") != settings.epochs
            or training.get("planned_epochs") != settings.epochs
        ):
            raise ValueError(
                f"{path.name}: not a completed fit ({training.get('completed_epochs')!r} of "
                f"{settings.epochs} epochs)"
            )
        fit = cls(
            settings=settings,
            curves=payload["curves_rpm"].numpy(),
            report=report,
            checkpoint_sha256=sha,
        )
        probe = payload["probe_audio"].numpy()
        got = render_eone(
            fit,
            payload["probe_rps"].numpy(),
            sr=fit.sample_rate,
            rng=np.random.default_rng(int(payload["probe_seed"])),
        )
        err = float(np.abs(got.astype(np.float64) - probe).max())
        tol = 1e-4 * max(float(np.abs(probe).max()), float(np.finfo(np.float32).tiny))
        if not err <= tol:
            raise ValueError(
                f"{path.name}: reloaded curves do not reproduce the fit probe ({err:g} > {tol:g})"
            )
        fit.report = {
            **report,
            "checkpoint_file": path.name,
            "checkpoint_sha256": sha,
            "load_probe_max_abs_error": err,
            "load_probe_tolerance": tol,
        }
        return fit


# ── render ─────────────────────────────────────────────────────────────────


def render_eone(fit: EoneFit, rps: np.ndarray, *, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Render ``(M, T)`` float32 audio for supplied per-rotor RPS ``(R, T)`` on CPU.

    The deployed table lookup: the exported curves drive the same synthesis
    code as training. Random stream, in order: ``phi0 = rng.uniform(0, 2 pi,
    R)``; loop phases ``2 pi * rng.random((R, K_b, L/2+1), float32)``; loop
    read offsets ``rng.integers(0, L, R)``. No target, gain or RMS matching.
    """
    cfg = fit.settings
    track = np.asarray(rps, dtype=np.float64)
    if track.ndim != 2 or track.shape[0] != fit.n_rotors or track.shape[1] < 1:
        raise ValueError(f"rps must be ({fit.n_rotors}, T>0), got {track.shape}")
    if not (np.isfinite(track).all() and (track >= 0).all()):
        raise ValueError("rps must be finite and non-negative")
    if int(sr) != fit.sample_rate:
        raise ValueError(f"EONE synthesises at its {fit.sample_rate} Hz model rate, got sr={sr}")
    n_loop = cfg.noise_loop
    phi0 = rng.uniform(0.0, _TWO_PI, fit.n_rotors)
    theta = rng.random((fit.n_rotors, cfg.n_bands, n_loop // 2 + 1), dtype=np.float32)
    theta *= np.float32(_TWO_PI)
    offsets = rng.integers(0, n_loop, fit.n_rotors)
    st = fit._statics
    cpu = torch.device("cpu")
    with torch.no_grad():
        ctl = _controls(track, cfg, cpu)
        loops = _noise_loops(st.band_mag, torch.from_numpy(theta), n_loop)
        del theta
        y = _synthesize(
            torch.tensor(fit.curves),
            ctl,
            torch.as_tensor(phi0, dtype=torch.float32),
            loops,
            torch.as_tensor(offsets, dtype=torch.int64),
            st,
        )
    return np.ascontiguousarray(y.numpy(), dtype=np.float32)


# ── fit ────────────────────────────────────────────────────────────────────


def _check_clips(clips: list[FlightClip], cfg: EoneSettings) -> tuple[str, int, int]:
    if not clips:
        raise ValueError("fit_eone requires real calibration windows")
    for c in clips:
        if c.provenance.get("split") != "calibration":
            raise ValueError(
                f"fit_eone accepts calibration windows only; {c.key} is {c.provenance.get('split')!r}"
            )

    def one(name: str, values: set[Any]) -> Any:
        if len(values) != 1:
            raise ValueError(f"calibration windows disagree on {name}: {sorted(values)}")
        return values.pop()

    rig = one("rig", {c.rig for c in clips})
    if rig not in SUPPORTED_RIGS:
        raise ValueError(f"fit_eone supports rigs {SUPPORTED_RIGS}, got {rig!r}")
    if len({c.key for c in clips}) != len(clips):
        raise ValueError("calibration window keys must be unique")
    sr = int(one("sr", {c.sr for c in clips}))
    if sr != cfg.sample_rate:
        raise ValueError(f"calibration audio is {sr} Hz, EONE runs at {cfg.sample_rate} Hz")
    for c in clips:
        r = np.asarray(c.rps)
        if not (np.isfinite(r).all() and (r >= 0).all()):
            raise ValueError(f"{c.key}: non-finite or negative RPS")
        if not np.isfinite(np.asarray(c.audio)).all():
            raise ValueError(f"{c.key}: non-finite audio")
    return (
        rig,
        int(one("microphone count", {c.audio.shape[0] for c in clips})),
        int(one("rotor count", {c.rps.shape[0] for c in clips})),
    )


def _seed_int(*words: int) -> int:
    return int(np.random.SeedSequence([int(w) for w in words]).generate_state(1)[0])


def _torch_draws(gen: torch.Generator, n_rot: int, cfg: EoneSettings, device: torch.device):
    phi0 = torch.rand(n_rot, generator=gen, device=device) * _TWO_PI
    theta = (
        torch.rand(n_rot, cfg.n_bands, cfg.noise_loop // 2 + 1, generator=gen, device=device)
        * _TWO_PI
    )
    offsets = torch.randint(0, cfg.noise_loop, (n_rot,), generator=gen, device=device)
    return phi0, theta, offsets


@dataclass(frozen=True)
class _Piece:
    audio: torch.Tensor
    ctl: _Controls
    harm: _HarmTarget


_PART_NAMES = (
    "total",
    "mrstft",
    "harmonic",
    "curvature",
    "mr_spectral_convergence",
    "mr_linear_magnitude",
    "mr_log_magnitude",
    "mr_energy",
)


def _objective(
    raw: torch.Tensor,
    scale: torch.Tensor,
    piece: _Piece,
    st: _Statics,
    cfg: EoneSettings,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    u = F.softplus(raw)
    curves = u * scale.view(-1, 1, 1, 1)
    phi0, theta, offsets = _torch_draws(gen, raw.shape[1], cfg, raw.device)
    loops = _noise_loops(st.band_mag, theta, cfg.noise_loop)
    del theta
    y = _synthesize(curves, piece.ctl, phi0, loops, offsets, st)
    terms = _mrstft_terms(y, piece.audio, st, cfg)
    mr = terms.mean()
    h = _harmonic_loss(y, piece.harm, st, cfg)
    c = _curvature(u)
    total = mr + cfg.harmonic_weight * h + cfg.curvature_weight * c
    return total, torch.stack((total, mr, h, c, *terms)).detach()


def _parts(v: torch.Tensor) -> dict[str, float]:
    return dict(zip(_PART_NAMES, (float(x) for x in v.tolist()), strict=True))


def _softplus_inv(y: float) -> float:
    return y + math.log(-math.expm1(-y))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantiles(x: np.ndarray) -> dict[str, Any]:
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "min": float(v.min()),
        "median": float(np.median(v)),
        "max": float(v.max()),
        "mean": float(v.mean()),
    }


def fit_eone(
    clips: list[FlightClip],
    *,
    seed: int,
    checkpoint: Path,
    settings: EoneSettings | None = None,
    device: str = "cuda",
) -> EoneFit:
    """Fit the EONE-RPS curve bank on calibration windows and save ``checkpoint``.

    Returns :meth:`EoneFit.load` of the written file, so the caller holds
    exactly the state every evaluation worker loads. ``device`` (explicit
    ``"cuda"``/``"cuda:N"`` or ``"cpu"``) is a runtime choice, not a scientific
    setting: the same settings, data, losses and epochs run on either. At
    production settings one step keeps a few GB live on the device (dominated
    by the 65536-point harmonic STFT of eight microphones); the measured peak
    is reported.
    """
    cfg = settings or EoneSettings()
    production = cfg == EoneSettings()
    dev = torch.device(device)
    if dev.type not in ("cuda", "cpu"):
        raise ValueError(f"fit_eone runs on 'cuda' or 'cpu', got {device!r}")
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
    clips = list(clips)
    started = time.monotonic()
    rig, n_mics, n_rot = _check_clips(clips, cfg)
    torch.manual_seed(int(seed))
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    n_piece, sr = cfg.piece_samples, cfg.sample_rate
    n_h, n_b, n_k = cfg.n_orders, cfg.n_bands, cfg.n_knots
    st = _statics(cfg, dev)

    # Fixed per-piece data, prepared once.
    t_prep = time.monotonic()
    manifest: list[dict[str, Any]] = []
    pieces: list[_Piece] = []
    piece_rps: list[np.ndarray] = []
    leftover = []
    power_sum = np.zeros(n_mics)
    n_total = 0
    for clip in clips:
        n_clip = clip.audio.shape[-1]
        n_here = n_clip // n_piece
        leftover.append(
            {"clip_key": clip.key, "unused_tail_samples": int(n_clip - n_here * n_piece)}
        )
        for j in range(n_here):
            a0 = j * n_piece
            audio = np.asarray(clip.audio[:, a0 : a0 + n_piece], dtype=np.float64)
            rps = np.asarray(clip.rps[:, a0 : a0 + n_piece], dtype=np.float64)
            power_sum += np.sum(audio**2, axis=1)
            n_total += n_piece
            audio_t = torch.as_tensor(audio, dtype=torch.float32, device=dev)
            ctl = _controls(rps, cfg, dev)
            harm = _harm_target(audio_t, rps, st, cfg)
            pieces.append(_Piece(audio=audio_t, ctl=ctl, harm=harm))
            piece_rps.append(rps)
            manifest.append(
                {
                    "piece": len(manifest),
                    "clip_key": clip.key,
                    "recording_id": clip.recording_id,
                    "regime": clip.regime,
                    "piece_in_clip": j,
                    "start_sample_in_clip": a0,
                    "n_samples": n_piece,
                    "relative_start_s": float(clip.start_s + a0 / sr),
                    "duration_s": cfg.piece_s,
                    "mean_rps": rps.mean(axis=1),
                    "min_rps": rps.min(axis=1),
                    "max_rps": rps.max(axis=1),
                    "harmonic_tracks_valid": harm.n_valid,
                    "harmonic_tracks_total": int(harm.valid.numel()),
                }
            )
    if not pieces:
        raise ValueError(f"no full {cfg.piece_s:g} s piece in the calibration windows")
    prep_seconds = time.monotonic() - t_prep
    rms = np.sqrt(power_sum / n_total)
    if not np.all(rms > 0):
        raise ValueError(f"silent calibration microphone(s): {np.flatnonzero(~(rms > 0)).tolist()}")
    # The nonempty sum returns an array of frame weights (R, P).
    coverage = cast(np.ndarray, sum(p.ctl.mask.sum(1).double().cpu().numpy() for p in pieces))
    all_rps = np.concatenate(piece_rps, axis=1)

    # Flat calibration-only initialisation.
    scale = torch.as_tensor(rms, dtype=torch.float32, device=dev)
    init_h = 1.0 / math.sqrt(n_rot * n_h)
    init_b = 1.0 / math.sqrt(2.0 * n_rot * n_b)
    raw0 = torch.empty(n_mics, n_rot, n_h + n_b, n_k, dtype=torch.float32)
    raw0[:, :, :n_h] = _softplus_inv(init_h)
    raw0[:, :, n_h:] = _softplus_inv(init_b)
    raw = torch.nn.Parameter(raw0.to(dev))

    n_steps = cfg.epochs * len(pieces)
    opt = torch.optim.AdamW([raw], lr=cfg.max_lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=cfg.max_lr,
        total_steps=n_steps,
        pct_start=cfg.onecycle_pct_start,
        anneal_strategy="cos",
        div_factor=cfg.onecycle_div_factor,
        final_div_factor=cfg.onecycle_final_div_factor,
    )
    train_gen = torch.Generator(device=dev)
    train_gen.manual_seed(_seed_int(seed, _TRAIN_DOMAIN))

    def evaluate() -> dict[str, float]:
        acc = torch.zeros(len(_PART_NAMES), dtype=torch.float64, device=dev)
        with torch.no_grad():
            for i, piece in enumerate(pieces):
                gen = torch.Generator(device=dev)
                gen.manual_seed(_seed_int(seed, _EVAL_DOMAIN, i))
                _, parts = _objective(raw, scale, piece, st, cfg, gen)
                acc += parts.double()
        out = _parts(acc / len(pieces))
        if not all(math.isfinite(v) for v in out.values()):
            raise RuntimeError(f"non-finite EONE calibration objective: {out}")
        return out

    t_train = time.monotonic()
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "lr_end": None,
            "train": None,
            "eval": evaluate(),
            "wall_seconds": time.monotonic() - t_train,
        }
    ]
    best_epoch, best_obj = 0, history[0]["eval"]["total"]
    best_raw = raw.detach().cpu().clone()
    step_total: list[float] = []
    step_lr: list[float] = []
    for epoch in range(1, cfg.epochs + 1):
        order = torch.randperm(len(pieces), generator=train_gen, device=dev).tolist()
        acc = torch.zeros(len(_PART_NAMES), dtype=torch.float64, device=dev)
        for i in order:
            step_lr.append(float(opt.param_groups[0]["lr"]))
            total, parts = _objective(raw, scale, pieces[i], st, cfg, train_gen)
            if not torch.isfinite(total):
                raise RuntimeError(f"non-finite EONE training loss at epoch {epoch}, piece {i}")
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()
            sched.step()
            acc += parts.double()
            step_total.append(float(parts[0]))
        ev = evaluate()
        history.append(
            {
                "epoch": epoch,
                "lr_end": float(opt.param_groups[0]["lr"]),
                "train": _parts(acc / len(pieces)),
                "eval": ev,
                "wall_seconds": time.monotonic() - t_train,
            }
        )
        if ev["total"] < best_obj:
            best_epoch, best_obj = epoch, ev["total"]
            best_raw = raw.detach().cpu().clone()
    train_seconds = time.monotonic() - t_train
    peak_bytes = int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else None
    peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    with torch.no_grad():
        u_best = F.softplus(best_raw)
        curves = (u_best * scale.cpu().view(-1, 1, 1, 1)).to(torch.float32)
    curves_np = curves.numpy()
    probe_n = min(n_piece, sr // 2)
    probe_rps = np.ascontiguousarray(piece_rps[0][:, :probe_n])
    probe_seed = _seed_int(seed, _PROBE_DOMAIN)
    probe_audio = render_eone(
        EoneFit(settings=cfg, curves=curves_np, report={}),
        probe_rps,
        sr=sr,
        rng=np.random.default_rng(probe_seed),
    )

    knots = knot_rpm(cfg)
    bands = noise_band_layout(cfg)
    max_rps = float(all_rps.max())
    reuse = [
        {"path": rel, "role": _SOURCE_ROLES[rel], "sha256": _sha256(_REPO_ROOT / rel)}
        for rel in EONE_SOURCE_FILES
    ]
    software = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch_cuda,
        "device": str(dev),
        "dtype": "float32 (phase float64)",
        "device_name": torch.cuda.get_device_name(dev)
        if dev.type == "cuda"
        else platform.processor(),
    }
    report = {
        "schema": SCHEMA,
        "method_id": EONE_METHOD_ID,
        "display_name": EONE_DISPLAY_NAME,
        "method": "Doerfler, Kuntz & Zimmer 2026 EONE curve bank, direct rotor-RPS adaptation",
        "status": (
            "adapted directly fitted EONE source-stage curve bank on real calibration audio; not the "
            "authors' code, no pretrained encoder-decoder, torque fixed, no published numbers reproduced"
        ),
        "reference": REFERENCE,
        "arxiv": ARXIV_ID,
        "bibliography_uuid": BIBLIOGRAPHY_UUID,
        "deviations": list(DEVIATIONS),
        "pretraining": {
            "used": False,
            "reason": "no public EONE encoder-decoder or checkpoint; none fabricated",
        },
        "torque": {
            "factor": 1.0,
            "learned": False,
            "reason": "no torque signal for rotors; multiplicative identity",
        },
        "no_test_data": (
            "fit consumed only the supplied calibration-split windows; no evaluation loader, "
            "evaluation audio, RMS or gain matching; render takes no target"
        ),
        "adapter_choices": {
            "sources": "independent rotor sources, per-microphone gain curves, excitations shared over microphones",
            "curve_parameterisation": "G[m,r,k,p] = s_m * softplus(theta[m,r,k,p]); s_m = calibration RMS of "
            "microphone m, folded into exported curves",
            "knots": "uniform 0..rpm_max inclusive; RPM = 60 * RPS; clamp outside (end-knot hold)",
            "control_frames": "frame j = mean supplied RPS over samples [j*hop, (j+1)*hop) (last: samples "
            "present), at centre j*hop + (hop-1)/2; Eq. 1 two-knot linear soft mask; "
            "activations linear between frame centres, held at the edges",
            "harmonic_orders": "shaft orders k = 1..n_orders (blade-pass multipliers k/2)",
            "phase": "phi[n] = (k * 2*pi * cumsum_{t<=n}(RPS[t]) / sr mod 2*pi) + phi0_r, float64 "
            "accumulation; one phi0_r ~ U(0, 2*pi) per rotor and chunk, same for all orders",
            "alias_gate": {
                "pass_hz": cfg.pass_hz,
                "stop_hz": sr / 2.0,
                "shape": "1 below pass, 0.5*(1+cos(pi*(f-pass)/(stop-pass))) between, 0 at/above "
                "stop; f = k * per-sample RPS",
                "always_open_max_order_at_calibration_max_rps": int(
                    math.floor(cfg.pass_hz / max_rps)
                )
                if max_rps > 0
                else None,
            },
            "noise_bank": {
                "erb_rate": "E(f) = 21.4*log10(1 + 0.00437 f)",
                "lobe": "cos(pi*(E - c_b)/Delta_b) for |E - c_b| < Delta_b/2, else 0; Delta_b = 2*spacing",
                "centres": "c_b = b * E(sr/2)/(K_b - 1), b = 0..K_b-1",
                "loop": "irfft of lobe x uniform random phase, DC/Nyquist zero, exact zero-mean "
                "unit-RMS period of noise_loop samples; one random read offset per rotor",
                "centre_hz": bands["centre_hz"],
                "lower_hz": bands["lower_hz"],
                "upper_hz": bands["upper_hz"],
            },
            "amplitude_units": "waveform units of FlightClip.audio; harmonic curve = cosine peak amplitude, "
            "noise curve = RMS of its band (loops are unit RMS)",
            "initialisation": {
                "harmonic_relative": init_h,
                "noise_relative": init_b,
                "rule": "half of each microphone's calibration power to harmonics (a^2/2 each), "
                "half to noise bands (unit RMS each), equal over nominal R*K_h and R*K_b "
                "components before gating",
                "per_mic_scale_rms": rms,
            },
            "mrstft": {
                "ffts": list(cfg.mrstft_ffts),
                "window": "periodic Hann, win = n_fft",
                "hop": "n_fft/4",
                "framing": "torch.stft center=True, zero padding",
                "terms": "SC=||Sg-St||_F/||St||_F; LIN=||Sg-St||_1/||St||_1; LOG=mean|ln(Sg+e)-ln(St+e)|, "
                "e=log_floor_rel*mean(St); EN=sum_tau|Eg-Et|/sum_tau Et, E=sum_f S^2",
                "combination": "per microphone, mean over microphones, mean of four terms, mean over scales",
            },
            "harmonic_loss": {
                "n_fft": cfg.harmonic_n_fft,
                "win": cfg.harmonic_win,
                "hop": cfg.harmonic_hop,
                "mask_halfwidth_bins": cfg.harmonic_halfwidth_bins,
                "mask_halfwidth_hz": cfg.harmonic_halfwidth_bins * sr / cfg.harmonic_n_fft,
                "track": "k * Hann-weighted mean RPS under each frame window, nearest bin",
                "valid": "mask within [1, n_fft/2-1] bins, mask top <= pass_hz, k * max RPS in "
                "window <= pass_hz",
                "loss": "mean over valid (mic, frame, rotor, order) of |ln(Eg+e_m)-ln(Et+e_m)|, "
                "e_m = energy_floor_rel * mean valid target energy",
                "weight": cfg.harmonic_weight,
            },
            "curvature": {
                "on": "unscaled positive curves u = softplus(theta)",
                "formula": "mean over curves of mean_p(second difference^2) / mean_p(u^2)",
                "weight": cfg.curvature_weight,
            },
            "optimizer": {
                "name": "torch.optim.AdamW",
                "max_lr": cfg.max_lr,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": cfg.weight_decay,
                "scheduler": "torch OneCycleLR anneal_strategy=cos, cycle_momentum=True "
                "(beta1 0.85..0.95)",
                "pct_start": cfg.onecycle_pct_start,
                "div_factor": cfg.onecycle_div_factor,
                "final_div_factor": cfg.onecycle_final_div_factor,
                "steps": n_steps,
                "steps_per_epoch": len(pieces),
                "batch": "one piece, all microphones jointly",
            },
            "random_streams": {
                "training": "one torch.Generator on the fit device seeded SeedSequence([seed, 1]); per epoch a "
                "piece permutation, per step phi0 (R), loop phases (R, K_b, L/2+1), offsets (R)",
                "evaluation": "per piece a fresh generator seeded SeedSequence([seed, 2, piece]), identical "
                "every epoch",
                "probe": "render_eone with numpy default_rng(SeedSequence([seed, 3]))",
            },
            "best_state": "epoch (0 = initialisation) with the lowest fixed-draw calibration objective",
            "render": "CPU torch float32 (float64 phase), the training synthesis code with exported curves; "
            "random stream phi0, loop phases, offsets from the caller's numpy rng",
        },
        "settings": asdict(cfg),
        "production_settings": production,
        "seed": int(seed),
        "rig": rig,
        "sample_rate": sr,
        "n_mics": n_mics,
        "n_rotors": n_rot,
        "n_orders": n_h,
        "n_bands": n_b,
        "n_knots": n_k,
        "knot_rpm": knots,
        "n_parameters": int(raw.numel()),
        "calibration_clip_keys": [c.key for c in clips],
        "calibration_recordings": sorted({c.recording_id for c in clips}),
        "calibration_splits": sorted({str(c.provenance.get("split")) for c in clips}),
        "pieces": {
            "n": len(pieces),
            "piece_s": cfg.piece_s,
            "manifest": manifest,
            "unused": leftover,
            "segmentation": "non-overlapping from each window start; tails shorter than piece_s unused",
        },
        "calibration_rps": {
            "min": float(all_rps.min()),
            "max": max_rps,
            "mean": float(all_rps.mean()),
        },
        "knot_coverage_frames": {
            "per_rotor_knot": coverage,
            "unvisited_knots_per_rotor": (coverage == 0).sum(axis=1),
        },
        "training": {
            "history": history,
            "step_total_loss": step_total,
            "step_lr": step_lr,
            "planned_epochs": cfg.epochs,
            "completed_epochs": len(history) - 1,
            "completed_steps": len(step_total),
            "planned_steps": n_steps,
            "best_epoch": best_epoch,
            "best_eval_objective": best_obj,
            "initial_eval_objective": history[0]["eval"]["total"],
            "final_eval_objective": history[-1]["eval"]["total"],
            "best_minus_initial_objective": best_obj - history[0]["eval"]["total"],
            "relative_objective_improvement": 1.0 - best_obj / history[0]["eval"]["total"],
            "improved_over_initialisation": bool(best_epoch > 0),
            "checkpoint_policy": "written once, atomically (tmp + os.replace), only after every "
            "planned epoch completed; no intermediate checkpoint exists",
        },
        "curves_summary": {
            "harmonic": _quantiles(curves_np[:, :, :n_h]),
            "noise": _quantiles(curves_np[:, :, n_h:]),
        },
        "runtime": {
            "device": str(dev),
            "peak_device_bytes": peak_bytes,
            "peak_process_rss_kib": peak_rss_kib,
            "torch_threads": torch.get_num_threads(),
            "prep_seconds": prep_seconds,
            "train_seconds": train_seconds,
            "chunk_blocks": _CHUNK_BLOCKS,
            "total_before_save_seconds": time.monotonic() - started,
        },
        "software": software,
        "reuse_sources": reuse,
        "probe": {"seed": probe_seed, "n_samples": probe_n, "source_piece": 0},
    }
    payload = {
        "schema": SCHEMA,
        "method_id": EONE_METHOD_ID,
        "settings": asdict(cfg),
        "curves_rpm": curves,
        "torque_factor": 1.0,
        "knot_rpm": torch.from_numpy(knots),
        "calibration_scale": scale.cpu(),
        "theta_best": best_raw,
        "theta_final": raw.detach().cpu().clone(),
        "theta_init": raw0,
        "best_epoch": best_epoch,
        "probe_rps": torch.from_numpy(probe_rps),
        "probe_seed": probe_seed,
        "probe_audio": torch.from_numpy(probe_audio),
        "report_json": json.dumps(_jsonable(report), allow_nan=False, sort_keys=True),
    }
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint.with_name(checkpoint.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, checkpoint)
    return EoneFit.load(checkpoint)


__all__ = [
    "DEVIATIONS",
    "EONE_DISPLAY_NAME",
    "EONE_METHOD_ID",
    "EONE_SOURCE_FILES",
    "EoneFit",
    "EoneSettings",
    "fit_eone",
    "knot_rpm",
    "noise_band_layout",
    "render_eone",
]
