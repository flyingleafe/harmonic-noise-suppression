"""Four-regime frame labels from a rotor-speed TARGET — the ONE implementation.

Rule (first match wins), applied to the ``(R, T)`` target of each clip:

    ramp     |d(rotor-mean)/dt| >= 20 rev/s^2   (a transition, any level)
    zero     max rotor < 1 rev/s                (every rotor stopped)
    standby  rotor mean < 45 rev/s              (idling / warm-up)
    cruise   otherwise                          (steady flight)

The derivative is a central difference over +-0.25 s of the rotor mean,
boxcar-smoothed over 0.288 s first (9 frames at the 31.25 Hz STFT grid; the
real labels are piecewise-held, so an unsmoothed difference is a spike train).
Indices are clamped at the clip edges (a one-sided, shorter difference there).
The thresholds are stated from label statistics, never fitted to a model's
error; every boundary is a function of the target only, so all models share
one labelling. Consumers: the training panel (``training.validation``) and
``scripts/_regime_decomp.py``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["REGIMES", "THRESHOLDS", "frame_regimes"]

#: Regime names; a label's int8 code is its index here.
REGIMES: tuple[str, ...] = ("zero", "standby", "ramp", "cruise")

#: Every number that defines the split.
THRESHOLDS: dict[str, float] = {
    "ramp_slew_rev_s2": 20.0,
    "zero_max_rev_s": 1.0,
    "standby_mean_rev_s": 45.0,
    "derivative_halfwidth_s": 0.25,
    "smoothing_width_s": 9 / 31.25,
}


def _smooth(mean: torch.Tensor, width: int) -> torch.Tensor:
    """Centred boxcar moving average along the last axis, edge-padded."""
    if width < 3:
        return mean
    pad = width // 2
    padded = F.pad(mean[:, None, :], (pad, pad), mode="replicate")
    kernel = torch.full((1, 1, width), 1.0 / width, dtype=mean.dtype, device=mean.device)
    return F.conv1d(padded, kernel)[:, 0, :]


def frame_regimes(target_rps: torch.Tensor, frame_rate: float) -> torch.Tensor:
    """``(B, T)`` int8 regime codes (indices into :data:`REGIMES`).

    ``target_rps`` is ``(B, R, T)`` rev/s on a grid of ``frame_rate`` frames/s.
    Computed in float64 on the input's device, so the split is identical on
    CPU and GPU and at the threshold boundaries.
    """
    if target_rps.ndim != 3:
        raise ValueError(f"frame_regimes needs (B, R, T), got {tuple(target_rps.shape)}")
    if frame_rate <= 0:
        raise ValueError(f"frame_rate must be positive, got {frame_rate}")
    target = target_rps.to(torch.float64)
    n_frames = int(target.shape[-1])
    mean = target.mean(dim=1)
    width = int(round(THRESHOLDS["smoothing_width_s"] * frame_rate))
    width += 1 - width % 2  # odd, so the window is centred
    smoothed = _smooth(mean, width)

    half = int(round(THRESHOLDS["derivative_halfwidth_s"] * frame_rate))
    index = torch.arange(n_frames, device=target.device)
    hi = (index + half).clamp(max=n_frames - 1)
    lo = (index - half).clamp(min=0)
    span = (hi - lo).to(torch.float64) / frame_rate
    delta = smoothed[:, hi] - smoothed[:, lo]
    slew = torch.where(span > 0, delta / span.clamp(min=1e-12), torch.zeros_like(delta)).abs()

    labels = torch.full(mean.shape, REGIMES.index("cruise"), dtype=torch.int8, device=target.device)
    labels[mean < THRESHOLDS["standby_mean_rev_s"]] = REGIMES.index("standby")
    labels[target.amax(dim=1) < THRESHOLDS["zero_max_rev_s"]] = REGIMES.index("zero")
    labels[slew >= THRESHOLDS["ramp_slew_rev_s2"]] = REGIMES.index("ramp")
    return labels
