"""Clips and their periodograms.

Two sources of real audio with usable rotor-speed references:

* the 27 four-second training crops of the 2026-09-06 linewidth audit
  (17 DREGON room2, 10 FLY125), whose telemetry was refined against the
  audio's low-order harmonics (``pi_kalman_refine``) — ``*.npz`` files with
  ``audio (8, T)``, ``rps (4, T)`` refined, ``rps_original``;
* the frozen validation split ``DREGON-LM-V4-michaels-valid-full`` — the
  ``nosource`` eight-second clips of DREGON room1 (measured shaft rate) and
  Michael's FLY124, which also cover stopped rotors and ramps, i.e. the frames
  where synthetic-only models fail worst.

Synthetic control clips come from the renderer itself with the same geometry,
so the fit can be checked against planted parameters before it is believed on
real audio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

N_FFT = 2048
HOP = 512
SR = 16000

REFS_DIR = Path(
    "/home/flyingleafe/Research/PhD/projects/harmonic-noise-suppression/.worktrees/jhtr-refinement"
    "/results/jhtr/trajectory-linewidth/refs"
)
VALID_DIR = Path(
    "/home/flyingleafe/.cache/dload/materialized/DREGON-LM-V4-michaels-valid-full/9604f3ffc2c9"
)


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


def load_ref_clip(path: Path) -> Clip:
    with np.load(path, allow_pickle=True) as z:
        meta = json.loads(str(z["meta"]))
        audio = np.asarray(z["audio"], dtype=np.float32)
        rps = np.asarray(z["rps"], dtype=np.float64)
        original = np.asarray(z["rps_original"], dtype=np.float64)
    rec = meta.get("recording_id", path.stem)
    group = "fly125" if rec.startswith("FLY") else "dregon_room2"
    return Clip(path.stem, group, audio, rps, int(meta.get("sample_rate", SR)), original, meta)


def ref_clips() -> list[Path]:
    return sorted(p for p in REFS_DIR.glob("*.npz") if not p.stem.endswith("_matched"))


def valid_nosource_ids() -> list[tuple[str, str]]:
    meta = json.loads((VALID_DIR / "metadata.json").read_text())["valid"]
    return [(s["id"], s["recording_id"]) for s in meta if s["source_type"] == "nosource"]


def load_valid_clip(sample_id: str) -> Clip:
    meta = json.loads((VALID_DIR / "metadata.json").read_text())["valid"]
    entry = next(s for s in meta if s["id"] == sample_id)
    audio, sr = sf.read(VALID_DIR / sample_id / "mixture.wav", dtype="float32", always_2d=True)
    audio = np.ascontiguousarray(audio.T)
    rps_tel = np.load(VALID_DIR / sample_id / "rps.npy").astype(np.float64)
    t_tel = np.arange(rps_tel.shape[1]) / float(entry["motor_sample_rate"])
    t_audio = np.arange(audio.shape[1]) / sr
    rps = np.stack([np.interp(t_audio, t_tel, r) for r in rps_tel])
    rps = np.maximum(rps, 0.0)
    rec = entry["recording_id"]
    group = "fly124" if "FLY" in rec else "dregon_room1"
    return Clip(f"{rec}_{sample_id}", group, audio, rps, int(sr), rps.copy(), dict(entry))


def refine_rps(clip: Clip, **overrides: Any) -> Clip:
    """Refine the carrier against the audio's low orders (the audit's recipe).

    Rotors slower than the tracker's ``min_rate`` are left as telemetry; the
    refinement is a local correction (sub-rev/s), not a search.
    """
    from tracking.decompose import interp_rps
    from tracking.phase_increment_tracker import pi_kalman_refine

    settings: dict[str, Any] = dict(
        k_max=16, k_caps=(8, 16, 16), lowk_gate="none", pair_mode="joint", threads=1
    )
    settings.update(overrides)
    n = clip.audio.shape[1]
    t_audio = np.arange(n, dtype=np.float64) / clip.sr
    ft = np.arange(n // HOP + 1, dtype=np.float64) * HOP / clip.sr
    initial = interp_rps(clip.rps, t_audio, ft)
    r_grid, diag = pi_kalman_refine(clip.audio, initial, ft, sr=clip.sr, **settings)
    rps = interp_rps(r_grid, ft, t_audio)
    meta = dict(clip.meta, refinement=dict(settings=settings, n_iter=len(diag.get("rotors", []))))
    return Clip(clip.clip_id, clip.group, clip.audio, np.maximum(rps, 0.0), clip.sr, clip.rps, meta)


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
