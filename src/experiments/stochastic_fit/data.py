"""Clips and their periodograms — the tensors every fit consumes.

:class:`Clip` is real or synthetic audio ``(M, T)`` with the rotor-speed track
``(R, T)`` the fit holds FIXED, on the audio grid. :class:`Periodogram` is its
Hann periodogram on the models' front-end grid, which is what the Whittle
likelihood scores.

Real clips come from the published frames datasets — see
:mod:`experiments.stochastic_fit.clips`, the only loader. Synthetic control
clips come from the renderer itself with the same geometry, so a fit can be
checked against planted parameters before it is believed on real audio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

N_FFT = 2048
HOP = 512
SR = 16000


@dataclass
class Clip:
    clip_id: str
    group: str  # recording family: dregon_room2 / fly125 / dregon_room1 / fly124 / synthetic
    audio: np.ndarray  # (M, T) float32
    rps: np.ndarray  # (R, T) float64, the trajectory the fit holds fixed
    sr: int = SR
    rps_original: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.audio.shape[1] / self.sr


@dataclass
class Periodogram:
    power: np.ndarray  # (M, N, F) |STFT|^2 / sum(w^2): white noise of variance s^2 -> mean s^2
    freqs: np.ndarray  # (F,) Hz
    times: np.ndarray  # (N,) s, frame centres
    rps: np.ndarray  # (R, N) rev/s at frame centres
    n_fft: int = N_FFT
    hop: int = HOP

    @property
    def df(self) -> float:
        return float(self.freqs[1] - self.freqs[0])


def periodogram(clip: Clip, n_fft: int = N_FFT, hop: int = HOP) -> Periodogram:
    """Hann periodogram on the models' front-end grid, no padding at the edges
    (frames fully inside the clip), so every frame is a genuine observation."""
    audio = np.asarray(clip.audio, dtype=np.float64)
    n = audio.shape[1]
    window = np.hanning(n_fft + 1)[:n_fft]
    n_frames = 1 + (n - n_fft) // hop
    starts = np.arange(n_frames) * hop
    frames = np.stack([audio[:, s : s + n_fft] for s in starts], axis=1) * window
    spec = np.fft.rfft(frames, axis=-1)
    power = (spec.real**2 + spec.imag**2) / float(np.sum(window**2))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / clip.sr)
    times = (starts + n_fft / 2.0) / clip.sr
    # frame-mean speed, not the centre sample: a ramp's line drifts within the window
    rps = np.stack(
        [
            np.mean(np.stack([clip.rps[r, s : s + n_fft] for s in starts]), axis=1)
            for r in range(clip.rps.shape[0])
        ]
    )
    return Periodogram(power.astype(np.float32), freqs, times, rps, n_fft, hop)


def synthetic_clip(
    seed: int,
    rps: np.ndarray,
    *,
    n_mics: int = 8,
    duration_s: float | None = None,
    ranges: dict[str, Any] | None = None,
    mic_gain_db: tuple[float, float] = (-12.0, 0.0),
    line_mode: str = "stochastic",
) -> tuple[Clip, dict[str, Any]]:
    """A renderer clip with known parameters, driven by ``rps`` ``(R, T)``.

    Returns the clip and the diagnostics (``params``, GP draws, PSD pieces).
    """
    from data_processing.stochastic_rotor_noise import StochasticRanges, sample_params, synthesize

    rng = np.random.default_rng(seed)
    rps = np.asarray(rps, dtype=np.float64)
    if duration_s is not None:
        rps = rps[:, : int(duration_s * SR)]
    nyq = SR / 2.0
    slowest = max(float(np.min(rps[rps > 5.0])) if np.any(rps > 5.0) else 20.0, 5.0)
    n_harm = int(min(np.floor(nyq / slowest), 400))
    params = sample_params(
        rng,
        StochasticRanges.from_dict(ranges),
        n_rotors=rps.shape[0],
        n_harmonics=n_harm,
        sample_rate=SR,
    )
    audio, diag = synthesize(
        params, rps, rng=rng, n_mics=n_mics, mic_gain_db=mic_gain_db, line_mode=line_mode
    )
    diag["params"] = params
    clip = Clip(
        f"synthetic_{seed}",
        "synthetic",
        audio.astype(np.float32),
        rps,
        SR,
        rps.copy(),
        dict(seed=seed),
    )
    return clip, diag
