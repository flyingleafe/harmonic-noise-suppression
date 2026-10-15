"""Tiny synthetic rps_prediction dataset + model for the training-framework
tests (collate/validate/loop). Referenced by dotted path
(``tests.training._fixtures.TinyRPSFrameDataset`` /
``tests.training._fixtures.TinyRPSModel``) from hand-built configs, so
:func:`training.config.build_dataset` / :func:`training.config.instantiate_model`
exercise the exact same ``_target_`` dispatch a real Hydra run would use.
"""

from __future__ import annotations

import numpy as np
import tdseries as td
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, IterableDataset

__all__ = [
    "TinyRPSFrameDataset",
    "TinyRPSIterableDataset",
    "TinyPanelFrameDataset",
    "TinyRPSModel",
    "make_tiny_frame",
    "TinyNoiseGenFrameDataset",
]


def make_tiny_frame(
    *,
    recording_id: str,
    input_snr: float,
    duration_s: float = 0.5,
    sample_rate: int = 16000,
    hop_length: int = 512,
    num_rotors: int = 4,
    rng: np.random.Generator | None = None,
) -> td.Frame:
    """One deterministic synthetic ``rps_prediction`` sample Frame."""
    rng = rng or np.random.default_rng(0)
    n_audio = int(round(duration_s * sample_rate))
    audio = rng.standard_normal(n_audio).astype(np.float32) * 0.1
    n_frames = n_audio // hop_length + 1
    rps = rng.uniform(20.0, 100.0, size=(num_rotors, n_frames)).astype(np.float32)

    mixture = td.uniform(audio, sample_rate, dims=("time",), t_start=0.0)
    rps_idx = td.GridIndex.create((sample_rate, hop_length), n_frames, t_start=0)
    rps_series = td.Series(rps, ("rotor", "time"), {"time": rps_idx})
    meta = td.Frame({"recording_id": recording_id, "input_snr": float(input_snr)})
    return td.Frame({"mixture": mixture, "rps": rps_series, "meta": meta})


class TinyRPSFrameDataset(Dataset):
    """Map-style dataset of ``n_samples`` deterministic synthetic Frames."""

    def __init__(
        self,
        *,
        n_samples: int = 8,
        duration_s: float = 0.5,
        sample_rate: int = 16000,
        hop_length: int = 512,
        num_rotors: int = 4,
        seed: int = 0,
    ) -> None:
        self.n_samples = int(n_samples)
        rng = np.random.default_rng(seed)
        self._frames = [
            make_tiny_frame(
                recording_id=f"tiny_{i}",
                input_snr=float(-30 + 5 * (i % 7)),
                duration_s=duration_s,
                sample_rate=sample_rate,
                hop_length=hop_length,
                num_rotors=num_rotors,
                rng=rng,
            )
            for i in range(self.n_samples)
        ]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> td.Frame:
        return self._frames[idx]


class TinyRPSIterableDataset(IterableDataset):
    """Infinite deterministic stream for optimizer-step cadence tests."""

    def __init__(self, **kwargs) -> None:
        self._finite = TinyRPSFrameDataset(**kwargs)

    def __iter__(self):
        while True:
            for index in range(len(self._finite)):
                yield self._finite[index]


def _profile_level(profile: str, n_frames: int) -> np.ndarray:
    """Rotor-mean speed track (rev/s) of one scripted panel clip."""
    if profile == "zero":
        return np.zeros(n_frames)
    if profile == "standby":
        return np.full(n_frames, 30.0)
    if profile == "cruise":
        return np.full(n_frames, 80.0)
    if profile == "ramp":
        return np.linspace(10.0, 110.0, n_frames)
    if profile == "zero_then_cruise":
        return np.where(np.arange(n_frames) < n_frames // 2, 0.0, 80.0)
    raise ValueError(f"unknown profile {profile!r}")


class TinyPanelFrameDataset(Dataset):
    """A stand-in for the flattened rp validation panel: one mono item per
    (clip, channel), clip-major, each carrying the panel meta (``dataset``,
    ``source_type``, ``clip_index``, ``channel``, ``recording_id``,
    ``snr_db``) and a scripted target. ``clips`` entries are
    ``{dataset, source_type, profile}`` with ``profile`` in zero / standby /
    cruise / ramp / zero_then_cruise; rotors sit 0/2/4/6 % above the level."""

    def __init__(
        self,
        *,
        clips: list[dict],
        n_channels: int = 2,
        duration_s: float = 2.0,
        sample_rate: int = 16000,
        hop_length: int = 512,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        n_audio = int(round(duration_s * sample_rate))
        n_frames = n_audio // hop_length + 1
        scale = 1.0 + 0.02 * np.arange(4)[:, None]
        self._frames: list[td.Frame] = []
        for clip_index, clip in enumerate(clips):
            rps = (scale * _profile_level(str(clip["profile"]), n_frames)).astype(np.float32)
            for channel in range(n_channels):
                audio = (rng.standard_normal(n_audio) * 0.1).astype(np.float32)
                rps_idx = td.GridIndex.create((sample_rate, hop_length), n_frames, t_start=0)
                meta = td.Frame(
                    {
                        "dataset": str(clip["dataset"]),
                        "source_type": str(clip["source_type"]),
                        "clip_index": clip_index,
                        "channel": channel,
                        "recording_id": f"{clip['dataset']}_rec",
                        "snr_db": None,
                    }
                )
                self._frames.append(
                    td.Frame(
                        {
                            "mixture": td.uniform(audio, sample_rate, dims=("time",), t_start=0.0),
                            "rps": td.Series(rps, ("rotor", "time"), {"time": rps_idx}),
                            "meta": meta,
                        }
                    )
                )

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> td.Frame:
        return self._frames[idx]


class TinyRPSModel(nn.Module):
    """2-layer Conv1d RPS predictor: mono ``(B, T)`` audio -> ``(B, num_rotors,
    T // hop_length + 1)`` — small enough for a CPU smoke test, no front-end
    dependency (unlike the real ``SimpleConv*`` family)."""

    def __init__(self, *, hop_length: int = 512, num_rotors: int = 4, hidden: int = 8) -> None:
        super().__init__()
        self.hop_length = hop_length
        self.conv1 = nn.Conv1d(
            1, hidden, kernel_size=hop_length * 2, stride=hop_length, padding=hop_length
        )
        self.conv2 = nn.Conv1d(hidden, num_rotors, kernel_size=1)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        n_frames = audio.shape[-1] // self.hop_length + 1
        h = F.relu(self.conv1(audio.unsqueeze(1)))
        out = self.conv2(h)
        if out.shape[-1] != n_frames:
            out = F.interpolate(out, size=n_frames, mode="linear", align_corners=False)
        return out


def make_tiny_noise_gen_frame(
    *,
    drone: str,
    duration_s: float = 0.5,
    sample_rate: int = 16000,
    num_rotors: int = 4,
    num_mics: int = 2,
    rng: np.random.Generator | None = None,
) -> td.Frame:
    """One deterministic synthetic ``noise_generation`` sample Frame — no
    real DREGON/Michael's data needed (unlike
    ``data_processing.frame_datasets.NoiseGenFrameDataset``, which wraps a
    real ``NoiseRPSDataset``): geometry/RPS/audio are all synthetic, just
    shaped to satisfy ``tasks.task.noise_generation``'s input spec plus a
    ``meta.drone`` name for
    :class:`tasks.codecs.NoiseGenerationCodec`'s ``conditioned`` path."""
    rng = rng or np.random.default_rng(0)
    n_audio = int(round(duration_s * sample_rate))
    rps = rng.uniform(20.0, 100.0, size=(num_rotors, n_audio)).astype(np.float32)
    audio = (rng.standard_normal((num_mics, n_audio)) * 0.1).astype(np.float32)
    mic_pos = (rng.standard_normal((num_mics, 3)) * 0.2).astype(np.float32)
    rotor_pos = (rng.standard_normal((num_rotors, 3)) * 0.2).astype(np.float32)

    rps_series = td.uniform(rps, sample_rate, dims=("rotor", "time"), t_start=0.0)
    audio_series = td.uniform(audio, sample_rate, dims=("mic", "time"), t_start=0.0)
    return td.Frame(
        {
            "rps": rps_series,
            "audio": audio_series,
            "mic_pos": td.wrap(mic_pos, dims=("mic", None)),
            "rotor_pos": td.wrap(rotor_pos, dims=("rotor", None)),
            "meta": td.Frame({"drone": drone}),
        }
    )


class TinyNoiseGenFrameDataset(Dataset):
    """Map-style dataset of ``n_samples`` deterministic synthetic
    ``noise_generation`` Frames, alternating ``drone in {"dregon",
    "michaels"}`` — for exercising the ``conditioned=True``
    :class:`~tasks.codecs.NoiseGenerationCodec` path end-to-end via
    ``training.validate.validate_config`` without needing real data."""

    def __init__(
        self,
        *,
        n_samples: int = 4,
        duration_s: float = 0.5,
        sample_rate: int = 16000,
        num_rotors: int = 4,
        num_mics: int = 2,
        seed: int = 0,
    ) -> None:
        self.n_samples = int(n_samples)
        rng = np.random.default_rng(seed)
        drones = ("dregon", "michaels")
        self._frames = [
            make_tiny_noise_gen_frame(
                drone=drones[i % len(drones)],
                duration_s=duration_s,
                sample_rate=sample_rate,
                num_rotors=num_rotors,
                num_mics=num_mics,
                rng=rng,
            )
            for i in range(self.n_samples)
        ]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> td.Frame:
        return self._frames[idx]
