"""Native-sample-rate clips, and the decimation the 16 kHz datasets should have used.

Every clip the fit has ever seen came from a 16 kHz derived dataset, and every
one of those carries a dead band above ~7.9 kHz: an anti-alias notch of the
publishing resampler, not something the drone does. It is a trivially learnable
domain cue in synthetic-vs-real, and it truncates the comb exactly where the
high-order question lives.

Both raw trees are 44.1 kHz, 8 channels, and both are already local:

* DREGON — ``DREGON_<rec>/DREGON_<rec>.wav`` plus ``_motors.mat`` (1 kHz
  telemetry) and ``_audiots.mat`` (the per-sample audio clock in absolute unix
  time, which is why the reference clips' ``start_s`` are timestamps);
* Michael's — ``recording_{1,2}/12{4,5}.wav`` plus the DJI flight log, aligned
  by :func:`data_processing.sources.michaels.load_raw_aligned` (per-recording
  time offset and the calibrated rev/s scale).

This module reads those through the canonical source loaders — never by
re-parsing a ``.mat`` or CSV — cuts the SAME windows the 16 kHz clips use, and
offers :func:`decimate` for a controlled comparison at any target rate. The
order Nyquist at 44.1 kHz is 22.05 kHz, so a 90 rev/s rotor reaches order 245
with no notch at all.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import scipy.signal as sps

from experiments.stochastic_fit.data import SR, Clip, ref_clips

NATIVE_SR = 44100


#: Raw trees, as materialized by dload. Both are inputs to the canonical
#: builders, so a change here is a change of data, not of formatting.
#:
#: Resolved through ``streams.ensure_local`` so the same code runs on a cluster
#: node, where no path under this user's home exists. The local cache hit is a
#: marker-file check, so the lookup is free once the tree is present.
def _raw_tree(dataset: str, fallback: str) -> Path:
    local = Path(fallback)
    if local.exists():
        return local
    from data_processing.streams import ensure_local

    return ensure_local(dataset)


DREGON_RAW = _raw_tree("DREGON", "/home/flyingleafe/.cache/dload/materialized/DREGON/db39bcf762d0")
MICHAELS_RAW = _raw_tree(
    "recording_with_motor_speed",
    "/home/flyingleafe/.cache/dload/materialized/recording_with_motor_speed/5b7eab554710",
)
BENCH_DIR = DREGON_RAW / "DREGON_individual_motors_recordings"

#: Where extracted native windows are cached (npz per clip). Native audio is
#: 2.8x the samples and 8 channels, so re-cutting on every call is wasteful.
CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache" / "native_clips"


def _dregon_sample(recording_id: str) -> dict:
    """The discovered raw-tree sample dict for one DREGON recording."""
    from data_processing.sources.dregon import discover_recordings

    for s in discover_recordings(DREGON_RAW):
        if s["recording_id"] == recording_id:
            return s
    raise KeyError(f"no DREGON recording {recording_id!r} under {DREGON_RAW}")


@lru_cache(maxsize=2)
def _dregon_recording(recording_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """``(audio, audio_t, rps, sr)`` of one DREGON recording at native rate.

    ``audio_t`` is the recording's own clock (absolute unix seconds, from
    ``*_audiots.mat``), which is the time base the reference clips' ``start_s``
    live on. ``rps`` is the measured telemetry interpolated onto the audio
    grid, the same convention as :class:`~experiments.stochastic_fit.data.Clip`.
    """
    from data_processing.sources.dregon import get_geometry, load_timeframe

    frame = load_timeframe(_dregon_sample(recording_id), geometry=get_geometry(DREGON_RAW))
    audio_s = frame["audio"]
    audio = np.asarray(audio_s.data, dtype=np.float32)
    sr = float(_series_rate(audio_s, audio.shape[-1]))
    audio_t = _series_time(audio_s, audio.shape[-1], sr)

    entries = set(frame.keys())
    key = next(k for k in ("motors_measured", "motors_command") if k in entries)
    tel = frame[key]
    tel_v = np.asarray(tel.data, dtype=np.float64)
    tel_t = _series_time(tel, tel_v.shape[-1], None)
    rps = np.stack([np.interp(audio_t, tel_t, r) for r in tel_v])
    return audio, audio_t, np.maximum(rps, 0.0), sr


@lru_cache(maxsize=2)
def _michaels_recording(recording_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """``(audio, audio_t, rps, sr)`` of FLY124/FLY125 at native rate.

    ``load_raw_aligned`` applies the recording's calibrated time offset and
    rev/s scale, so ``audio_t`` starts at 0 and matches the ``start_s`` of the
    16 kHz reference clips.
    """
    from data_processing.sources.michaels import MICHAELS_FILES, load_raw_aligned

    stem = recording_id.upper()
    for wav, csv, offset, dilation in MICHAELS_FILES:
        if Path(csv).stem.upper() == stem:
            audio, stamps, rps_log, sr = load_raw_aligned(
                MICHAELS_RAW / wav, MICHAELS_RAW / csv, offset, dilation, sr=None
            )
            audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
            if audio.ndim == 2 and audio.shape[0] > audio.shape[1]:
                audio = audio.T
            audio_t = np.arange(audio.shape[-1]) / float(sr)
            rps_log = np.asarray(rps_log, dtype=np.float64)
            if rps_log.ndim == 2 and rps_log.shape[0] != len(stamps):
                pass
            else:
                rps_log = rps_log.T
            rps = np.stack([np.interp(audio_t, np.asarray(stamps, float), r) for r in rps_log])
            return audio, audio_t, np.maximum(rps, 0.0), float(sr)
    raise KeyError(f"no Michael's recording {recording_id!r}")


def _time_index(series: Any) -> Any:
    """The ``time`` index of a ``td.Series`` — ``GridIndex`` or ``StampIndex``."""
    try:
        return series.indexes["time"]
    except (AttributeError, KeyError, TypeError) as exc:  # pragma: no cover - guard
        raise ValueError(f"series carries no time index: {series!r}") from exc


def _series_rate(series: Any, n: int) -> float:
    """Sample rate of a ``td.Series``.

    A ``GridIndex`` states it exactly (``sr_num / sr_den``); a stamped series
    only implies it, so it is taken as ``(n - 1) / span``.
    """
    idx = _time_index(series)
    sr = getattr(idx, "sr", None)
    if sr:
        return float(sr)
    stamps = np.asarray(idx.timestamps, dtype=np.float64)
    if stamps.size > 1:
        return float((stamps.size - 1) / (stamps[-1] - stamps[0]))
    raise ValueError(f"cannot determine the sample rate of a {type(idx).__name__} ({n} samples)")


def _series_time(series: Any, n: int, sr: float | None) -> np.ndarray:
    """Absolute time stamps of a ``td.Series``, on the recording's own clock.

    ``GridIndex.sample_times()`` already carries ``t_start``, which for DREGON
    is the ``audiots`` unix epoch the reference clips index into.
    """
    idx = _time_index(series)
    getter = getattr(idx, "sample_times", None)
    if callable(getter):
        return np.asarray(getter(), dtype=np.float64)
    stamps = np.asarray(idx.timestamps, dtype=np.float64)
    if stamps.size == n:
        return stamps
    rate = sr if sr is not None else _series_rate(series, n)
    return float(getattr(idx, "t_start", 0.0)) + np.arange(n) / float(rate)


def is_dregon(recording_id: str) -> bool:
    return not recording_id.upper().startswith("FLY")


def _load_recording(recording_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if is_dregon(recording_id):
        return _dregon_recording(recording_id)
    return _michaels_recording(recording_id)


def clip_specs() -> dict[str, dict[str, Any]]:
    """``clip_id -> {recording_id, start_s, duration_s}`` for the fit's clips.

    Read from the 16 kHz reference clips themselves, so a native clip is the
    same window of the same recording — that is what makes the two comparable.
    """
    out: dict[str, dict[str, Any]] = {}
    for path in ref_clips():
        with np.load(path, allow_pickle=True) as z:
            meta = json.loads(str(z["meta"]))
        out[path.stem] = {
            "recording_id": meta["recording_id"],
            "start_s": float(meta["start_s"]),
            "duration_s": float(meta.get("duration_s", 4.0)),
        }
    return out


def load_native_clip(
    recording_id: str,
    start_s: float,
    duration_s: float = 4.0,
    *,
    clip_id: str | None = None,
    cache: bool = True,
) -> Clip:
    """One native-rate :class:`Clip` cut from a raw recording.

    ``start_s`` is on the recording's own clock: absolute unix seconds for
    DREGON (its ``audiots`` stamps), seconds from the aligned start for
    Michael's. Both conventions come straight from the 16 kHz clips, so
    :func:`clip_specs` values can be passed through unchanged.
    """
    cid = clip_id or f"{recording_id}@{start_s:.3f}+{duration_s:g}"
    path = CACHE_DIR / f"{cid.replace('/', '_')}_native.npz"
    if cache and path.exists():
        try:
            z_ok = True
            with np.load(path, allow_pickle=True) as z:
                z_ok = "audio" in z
        except Exception:
            z_ok = False
        if not z_ok:
            path.unlink(missing_ok=True)
    if cache and path.exists():
        with np.load(path, allow_pickle=True) as z:
            return Clip(
                cid,
                str(z["group"]),
                np.asarray(z["audio"], dtype=np.float32),
                np.asarray(z["rps"], dtype=np.float64),
                int(z["sr"]),
                np.asarray(z["rps"], dtype=np.float64),
                json.loads(str(z["meta"])),
            )

    audio, audio_t, rps, sr = _load_recording(recording_id)
    i0 = int(np.searchsorted(audio_t, start_s))
    i1 = i0 + int(round(duration_s * sr))
    if i1 > audio.shape[-1]:
        raise ValueError(
            f"{recording_id}: window [{start_s}, +{duration_s}s) runs past the recording "
            f"({audio.shape[-1] / sr:.1f} s of audio)"
        )
    group = (
        ("dregon_room1" if "room1" in recording_id else "dregon_room2")
        if is_dregon(recording_id)
        else ("fly124" if "124" in recording_id else "fly125")
    )
    meta = {
        "recording_id": recording_id,
        "start_s": float(start_s),
        "duration_s": float(duration_s),
        "sample_rate": int(sr),
        "native": True,
        "clock_start_s": float(audio_t[i0]),
    }
    clip = Clip(
        cid,
        group,
        np.ascontiguousarray(audio[:, i0:i1]),
        np.ascontiguousarray(rps[:, i0:i1]),
        int(sr),
        np.ascontiguousarray(rps[:, i0:i1]),
        meta,
    )
    if cache:
        # Atomic publish. Two jobs sharing this cache raced: one wrote the npz
        # while the other read it, and the reader died on "File is not a zip
        # file". A unique temp file plus os.replace makes a reader see either
        # the old file or the complete new one.
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
        np.savez_compressed(
            tmp,
            audio=clip.audio,
            rps=clip.rps,
            sr=clip.sr,
            group=clip.group,
            meta=json.dumps(meta),
        )
        os.replace(tmp, path)
    return clip


def load_native(clip_id: str, **kwargs: Any) -> Clip:
    """The native-rate twin of one of the fit's 16 kHz clips."""
    spec = clip_specs()[clip_id]
    return load_native_clip(
        spec["recording_id"], spec["start_s"], spec["duration_s"], clip_id=clip_id, **kwargs
    )


def decimate(clip: Clip, target_sr: int = SR) -> Clip:
    """Resample a clip to ``target_sr`` with a polyphase anti-alias filter.

    ``scipy.signal.resample_poly`` uses a Kaiser-windowed FIR whose transition
    band sits just below the new Nyquist, so the result keeps content up to
    ~0.99 x Nyquist instead of the ~7.9 kHz the published 16 kHz datasets stop
    at. The rps track is decimated by interpolation, which is exact for a 1 kHz
    telemetry track already interpolated onto the audio grid.
    """
    if clip.sr == target_sr:
        return clip
    from math import gcd

    g = gcd(int(clip.sr), int(target_sr))
    up, down = int(target_sr) // g, int(clip.sr) // g
    audio = sps.resample_poly(clip.audio.astype(np.float64), up, down, axis=-1)
    n = audio.shape[-1]
    t_new = np.arange(n) / float(target_sr)
    t_old = np.arange(clip.audio.shape[-1]) / float(clip.sr)
    rps = np.stack([np.interp(t_new, t_old, r) for r in clip.rps])
    meta = dict(clip.meta)
    meta.update(sample_rate=int(target_sr), decimated_from=int(clip.sr))
    return replace(
        clip,
        audio=np.ascontiguousarray(audio.astype(np.float32)),
        rps=np.ascontiguousarray(rps),
        sr=int(target_sr),
        rps_original=np.ascontiguousarray(rps),
        meta=meta,
    )


def bench_clip(motor: int, speed: int, duration_s: float = 4.0, start_s: float = 2.0) -> Clip:
    """One native-rate window of a DREGON single-motor bench recording.

    ``Motor{1-4}_{50,60,70,80,90}`` spin one rotor at a fixed setpoint with
    nothing else moving — the cleanest comb the project has. The rps track is
    unknown here (the bench has no telemetry), so ``rps`` is left empty and the
    rate comes from :mod:`experiments.stochastic_fit.bench`'s comb evidence.
    """
    import soundfile as sf

    path = BENCH_DIR / f"Motor{motor}_{speed}.wav"
    info = sf.info(path)
    i0 = int(round(start_s * info.samplerate))
    audio, sr = sf.read(
        path,
        start=i0,
        frames=int(round(duration_s * info.samplerate)),
        dtype="float32",
        always_2d=True,
    )
    return Clip(
        f"Motor{motor}_{speed}",
        "dregon_bench",
        np.ascontiguousarray(audio.T),
        np.zeros((1, audio.shape[0])),
        int(sr),
        None,
        {
            "recording_id": f"Motor{motor}_{speed}",
            "setpoint": speed,
            "motor": motor,
            "sample_rate": int(sr),
            "native": True,
            "start_s": float(start_s),
        },
    )
