"""Real audio for the fit, read from the published ``*-frames`` datasets.

**One source of recordings.** `DREGON-frames` and `michaels-frames` publish
every recording at its NATIVE sample rate (44.1 kHz, 8 channels) together with
the rotor-speed tracks, so the frames datasets are a strict superset of what a
raw-tree loader could reach:

* audio is the untouched recording — no 16 kHz brick wall (the published
  training sets stop at 7.9 kHz with 88-90 dB of attenuation, which is a
  learnable domain cue, and was the reason this package used to cut its own
  windows out of the raw trees);
* ``rps_refined`` is the regime-gated refined rotor-speed label
  (`docs/experiments/refined-rps-labels.md`) — the label the fit should hold
  fixed — published beside the untouched telemetry (``rps`` for Michael's,
  ``motors_measured``/``motors_command`` for DREGON);
* the DREGON single-motor bench recordings are in `DREGON-frames` as
  ``split="motor"`` (`recording_id` ``motor_Motor1_80``); they carry no rotor
  track at all, so :func:`load_recording` returns ``rps=None`` for them and the
  bench fit supplies its own rate.

Everything is addressed by dataset name (optionally ``NAME@VERSION``),
recording id, window and channel selection. Nothing here reads a path under
this user's home, so the same call runs on a cluster node: dload streams the
shard it needs.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import scipy.signal as sps

from experiments.stochastic_fit.data import SR, Clip

#: Both rigs record at 44.1 kHz; the published frames keep that rate.
NATIVE_SR = 44100

#: The label the fit holds fixed, unless the caller asks for another track.
DEFAULT_RPS_KEY = "rps_refined"
#: Tried in order when ``rps_key="auto"``: the refined label first, then the
#: raw tracks in the published preference order (``frames.PUBLISHED_RPS_KEYS``).
RPS_KEYS = ("rps_refined", "rps", "motors_measured", "motors_command")

#: Where cut windows are cached (one npz per clip). A native-rate 16 s window
#: of 8 channels is 22 MB; decoding the whole recording to cut it again costs
#: several seconds and a few hundred MB of RAM.
CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache" / "frame_clips"


def split_dataset(name: str) -> tuple[str, str | None]:
    """``"michaels-frames@8e9d1495"`` -> ``("michaels-frames", "8e9d1495")``."""
    dataset, _, version = str(name).partition("@")
    return dataset, (version or None)


def _time_index(series: Any) -> Any:
    try:
        return series.indexes["time"]
    except (AttributeError, KeyError, TypeError) as exc:  # pragma: no cover - guard
        raise ValueError(f"series carries no time index: {series!r}") from exc


def _series_rate(series: Any) -> float:
    """Sample rate of a ``td.Series``: exact for a grid, implied by stamps."""
    idx = _time_index(series)
    rate = getattr(idx, "sr", None)
    if rate:
        return float(rate)
    stamps = np.asarray(idx.timestamps, dtype=np.float64)
    if stamps.size > 1:
        return float((stamps.size - 1) / (stamps[-1] - stamps[0]))
    raise ValueError(f"cannot determine the sample rate of a {type(idx).__name__}")


def _series_stamps(series: Any, n: int) -> np.ndarray:
    """Absolute time stamps of a ``td.Series`` on the recording's own clock."""
    idx = _time_index(series)
    getter = getattr(idx, "sample_times", None)
    if callable(getter):
        return np.asarray(getter(), dtype=np.float64)
    stamps = np.asarray(idx.timestamps, dtype=np.float64)
    if stamps.size == n:
        return stamps
    return float(getattr(idx, "t_start", 0.0)) + np.arange(n) / _series_rate(series)


def _group(recording_id: str, meta: dict[str, Any]) -> str:
    """The rig-tie group of a recording: what a rig fit may share across clips."""
    rid = str(recording_id)
    if str(meta.get("split")) == "motor":
        return "dregon_bench"
    if rid.upper().startswith("FLY"):
        return rid.lower()
    return "dregon_room1" if "room1" in rid else "dregon_room2"


@dataclass(frozen=True)
class Recording:
    """One published recording: native-rate audio plus one rotor-speed track.

    ``t_start`` is the recording's own clock at audio sample 0 — absolute unix
    seconds for DREGON (its ``audiots`` stamps), 0 for Michael's aligned
    recordings. Window starts are on that clock, so :func:`windows` output can
    be passed straight to :func:`load_clip`.
    """

    dataset: str
    version: str | None
    recording_id: str
    group: str
    audio: np.ndarray  # (M, T) float32, native rate
    sr: float
    t_start: float
    rps: np.ndarray | None  # (R, N) rev/s on ``rps_t``, None on the bench
    rps_t: np.ndarray | None  # (N,) s on the recording clock
    rps_key: str | None
    meta: dict[str, Any]

    @property
    def n_channels(self) -> int:
        return int(self.audio.shape[0])

    @property
    def duration_s(self) -> float:
        return float(self.audio.shape[-1] / self.sr)

    @property
    def coverage(self) -> tuple[float, float]:
        """``(t0, t1)`` where BOTH audio and the rotor label exist.

        DREGON's telemetry starts 2.5-6.5 s after its audio and stops before
        it (room2: +5.223 s in, 2.873 s early, its last sample still at
        69-76 rev/s). ``np.interp`` would hold those endpoint speeds flat over
        the uncovered audio, so an uncovered window looks like steady cruise
        and passes the regime test with a carrier nothing measured. Every
        window is therefore required to lie inside this span.
        """
        t0, t1 = self.t_start, self.t_start + self.duration_s
        if self.rps_t is None or self.rps_t.size == 0:
            return t0, t1
        return max(t0, float(self.rps_t[0])), min(t1, float(self.rps_t[-1]))

    def rps_at(self, times: np.ndarray) -> np.ndarray:
        """``(R, len(times))`` rotor speeds at ``times`` (recording clock).

        Outside the label's own span this would be ``np.interp``'s held
        endpoint, i.e. an invented carrier, so it is refused instead. One
        telemetry step of slack covers a window that ends on the last stamp.
        """
        if self.rps is None or self.rps_t is None:
            raise ValueError(f"{self.recording_id}: no rotor-speed track published")
        step = float(np.median(np.diff(self.rps_t))) if self.rps_t.size > 1 else 0.0
        lo, hi = float(self.rps_t[0]) - step, float(self.rps_t[-1]) + step
        if float(times[0]) < lo or float(times[-1]) > hi:
            raise ValueError(
                f"{self.recording_id}: [{float(times[0]):.3f}, {float(times[-1]):.3f}] "
                f"leaves the {self.rps_key} span [{lo:.3f}, {hi:.3f}]"
            )
        return np.maximum(
            np.stack([np.interp(times, self.rps_t, r) for r in self.rps]),
            0.0,
        )

    def cut(
        self,
        start_s: float,
        duration_s: float,
        *,
        channels: str | tuple[int, ...] | None = None,
        clip_id: str | None = None,
    ) -> Clip:
        """One window of this recording as a :class:`Clip`, at the native rate.

        ``start_s`` is on the recording clock. The rotor track is interpolated
        onto the window's audio grid (the :class:`Clip` contract); a recording
        with no track (the bench) gets zeros, and the caller supplies the rate.
        """
        chans = resolve_channels(channel_spec(channels), self.n_channels)
        i0 = int(round((float(start_s) - self.t_start) * self.sr))
        i1 = i0 + int(round(float(duration_s) * self.sr))
        if i0 < 0 or i1 > self.audio.shape[-1]:
            raise ValueError(
                f"{self.recording_id}: window [{start_s}, +{duration_s}s) outside the "
                f"recording ({self.duration_s:.1f} s from t_start {self.t_start:.3f})"
            )
        audio = np.ascontiguousarray(self.audio[list(chans), i0:i1])
        times = self.t_start + np.arange(i0, i1) / self.sr
        rps = (
            self.rps_at(times)
            if self.rps is not None
            else np.zeros((1, audio.shape[-1]), dtype=np.float64)
        )
        meta = {
            "recording_id": self.recording_id,
            "dataset": self.dataset,
            "dataset_version": self.version,
            "rps_key": self.rps_key,
            "channels": list(chans),
            "start_s": float(start_s),
            "duration_s": float(duration_s),
            "sample_rate": int(self.sr),
            "clock_start_s": float(times[0]),
            "native": True,
        }
        cid = clip_id or f"{self.recording_id}@{start_s:.3f}+{duration_s:g}"
        return Clip(cid, self.group, audio, rps, int(self.sr), np.ascontiguousarray(rps), meta)


def _recording_from_frame(
    dataset: str, version: str | None, frame: Any, meta: dict[str, Any], rps_key: str
) -> Recording:
    """Decode one frame's audio and the requested rotor track.

    ``rps_key="auto"`` takes the first of :data:`RPS_KEYS` the frame carries;
    an explicit key the frame does not carry is an error, because silently
    fitting raw telemetry when the refined label was asked for would make two
    campaigns incomparable. The bench (``split == "motor"``) publishes no rotor
    track at all, so there the absence is expected and gives ``rps=None``.
    """
    recording_id = str(meta.get("recording_id"))
    audio_s = frame["audio"]
    audio = np.asarray(audio_s.data, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[None, :]
    sr = _series_rate(audio_s)
    t_start = float(getattr(_time_index(audio_s), "t_start", 0.0))

    key: str | None = None
    if rps_key == "auto":
        key = next((k for k in RPS_KEYS if k in frame), None)
    elif rps_key:
        if rps_key in frame:
            key = rps_key
        elif str(meta.get("split")) == "motor":
            key = None
        else:
            available = [k for k in RPS_KEYS if k in frame]
            raise KeyError(
                f"{dataset}:{recording_id} publishes no {rps_key!r} track "
                f"(rotor tracks present: {available or 'none'})"
            )
    rps = rps_t = None
    if key is not None:
        series = frame[key]
        values = np.asarray(series.data, dtype=np.float64)
        values = values[None, :] if values.ndim == 1 else values
        rps = np.maximum(values, 0.0)
        rps_t = _series_stamps(series, values.shape[-1])
    return Recording(
        dataset=dataset,
        version=version,
        recording_id=recording_id,
        group=_group(recording_id, meta),
        audio=np.ascontiguousarray(audio),
        sr=float(sr),
        t_start=t_start,
        rps=rps,
        rps_t=rps_t,
        rps_key=key,
        meta=meta,
    )


def iter_recordings(
    dataset: str,
    ids: tuple[str, ...] | None = None,
    version: str | None = None,
    rps_key: str = DEFAULT_RPS_KEY,
    *,
    split: str | None = None,
) -> Iterator[Recording]:
    """ONE pass over a published frames dataset, one recording alive at a time.

    Only the wanted samples are decoded, and the pass stops as soon as every
    id in ``ids`` has been seen — which is what makes a 20-cell bench fit read
    the 272-sample DREGON dataset once instead of once per cell. ``split``
    filters on ``meta.split`` and needs every frame decoded.
    """
    from data_processing.frames import meta_dict
    from data_processing.streams import decode_tdframe, is_data_sample, open_repository

    dataset, pinned = split_dataset(dataset)
    version = version or pinned
    wanted = None if ids is None else dict.fromkeys(str(i) for i in ids)
    ds = open_repository().dataset(dataset, version)
    for sample in ds.samples():
        if not is_data_sample(sample):
            continue
        key = str(sample[0])
        if wanted is not None and key not in wanted:
            continue
        frame = decode_tdframe(sample)
        meta = meta_dict(frame)
        meta.setdefault("recording_id", key)
        if split is not None and str(meta.get("split")) != split:
            continue
        yield _recording_from_frame(dataset, version, frame, meta, rps_key)
        if wanted is not None:
            wanted.pop(key, None)
            if not wanted:
                return
    if wanted:
        raise KeyError(f"no recording {sorted(wanted)} in {dataset}@{version or 'pinned'}")


@lru_cache(maxsize=2)
def load_recording(
    dataset: str = "michaels-frames",
    recording_id: str = "FLY125",
    version: str | None = None,
    rps_key: str = DEFAULT_RPS_KEY,
) -> Recording:
    """One published recording, decoded once per process (LRU of 2)."""
    return next(iter(iter_recordings(dataset, (recording_id,), version, rps_key)))


def channel_spec(text: str | tuple[int, ...] | None) -> tuple[int, ...] | None:
    """The REQUESTED channels, resolved against nothing: ``None`` means all.

    ``"0,3-5"`` -> ``(0, 3, 4, 5)``; ``None``/``"all"`` -> ``None``. Keeping
    "all" distinct from an explicit list is what lets a cache key be built
    before the recording is decoded.
    """
    if text is None:
        return None
    if isinstance(text, tuple):
        return tuple(dict.fromkeys(int(c) for c in text))
    if str(text).strip().lower() in ("", "all"):
        return None
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            lo, _, hi = part.partition("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return tuple(dict.fromkeys(out))


def resolve_channels(spec: tuple[int, ...] | None, n_channels: int) -> tuple[int, ...]:
    """Bounds-check a :func:`channel_spec` against a recording's channel count."""
    if spec is None:
        return tuple(range(int(n_channels)))
    bad = [c for c in spec if not 0 <= c < int(n_channels)]
    if bad:
        raise ValueError(f"channels {bad} outside the recording's 0..{n_channels - 1}")
    return spec


def _cache_path(
    dataset: str,
    version: str | None,
    recording_id: str,
    start_s: float,
    duration_s: float,
    channels: tuple[int, ...] | None,
    rps_key: str,
) -> Path:
    # The key names the DATA, never a loop index: a cruise probe once read the
    # standby clips because its cache key was ``probe_00``.
    stamp = "all" if channels is None else "-".join(str(c) for c in channels)
    return CACHE_DIR / (
        f"{dataset}_{(version or 'pinned')[:12]}_{recording_id}"
        f"_{start_s:.3f}+{duration_s:g}_ch{stamp}_{rps_key}.npz"
    )


def load_clip(
    dataset: str,
    recording_id: str,
    start_s: float,
    duration_s: float,
    *,
    version: str | None = None,
    channels: str | tuple[int, ...] | None = None,
    rps_key: str = DEFAULT_RPS_KEY,
    clip_id: str | None = None,
    cache: bool = True,
) -> Clip:
    """One native-rate :class:`~experiments.stochastic_fit.data.Clip`.

    ``start_s`` is on the recording's own clock (see :class:`Recording`).
    ``channels`` defaults to every channel; the rotor track is interpolated onto
    the clip's audio grid, which is the :class:`Clip` contract.
    """
    dataset, pinned = split_dataset(dataset)
    version = version or pinned
    cid = clip_id or f"{recording_id}@{start_s:.3f}+{duration_s:g}"
    spec = channel_spec(channels)
    path = _cache_path(dataset, version, recording_id, start_s, duration_s, spec, rps_key)
    if cache and path.exists():
        cached = read_cached_clip(path, cid)
        if cached is not None:
            return cached

    rec = load_recording(dataset, recording_id, version, rps_key)
    clip = rec.cut(start_s, duration_s, channels=spec, clip_id=cid)
    if cache:
        write_cached_clip(path, clip)
    return clip


def read_cached_clip(path: Path, clip_id: str) -> Clip | None:
    try:
        with np.load(path, allow_pickle=True) as z:
            if "audio" not in z:
                raise ValueError("incomplete cache entry")
            meta = json.loads(str(z["meta"]))
            rps = np.asarray(z["rps"], dtype=np.float64)
            return Clip(
                clip_id,
                str(z["group"]),
                np.asarray(z["audio"], dtype=np.float32),
                rps,
                int(z["sr"]),
                rps,
                meta,
            )
    except Exception:
        path.unlink(missing_ok=True)
        return None


def write_cached_clip(path: Path, clip: Clip) -> None:
    # UNCOMPRESSED: a 16 s native window is 23 MB of incompressible audio, and
    # ``savez_compressed`` spent 40 s on it against 0.2 s here — more than the
    # decode it is meant to save. The rps track travels as float32 (its own
    # producer grid is 32 ms; the audio-rate copy is interpolation).
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
    # Atomic publish: two jobs sharing the cache raced, and the reader died on
    # "File is not a zip file". A unique temp file plus os.replace makes a
    # reader see either the old file or the complete new one.
    np.savez(
        tmp,
        audio=clip.audio,
        rps=clip.rps.astype(np.float32),
        sr=clip.sr,
        group=clip.group,
        meta=json.dumps(clip.meta),
    )
    os.replace(tmp, path)


def windows(
    rec: Recording,
    *,
    seconds: float,
    max_clips: int = 8,
    min_rps: float | None = None,
    max_rps: float | None = None,
    stride_s: float | None = None,
) -> list[tuple[float, float]]:
    """``[(start_s, duration_s)]`` windows where every rotor stays in one regime.

    The regime is read from the rotor track at telemetry resolution, so the
    test is the same one the label carries. Windows advance by ``stride_s``
    (default: no overlap), and a window is kept only if it lies inside
    :attr:`Recording.coverage` AND every rotor stays inside
    ``[min_rps, max_rps]`` for its whole span.

    The grid is anchored to the AUDIO start, not to the coverage start, so a
    recording's window offsets do not move when a label's span changes; an
    uncovered window is dropped, not shifted.
    """
    step = float(seconds if stride_s is None else stride_s)
    cov0, cov1 = rec.coverage
    out: list[tuple[float, float]] = []
    start = 0.0
    while start + seconds <= rec.duration_s and len(out) < max_clips:
        t0 = rec.t_start + start
        keep = cov0 - 1e-9 <= t0 and t0 + seconds <= cov1 + 1e-9
        if keep and (min_rps is not None or max_rps is not None):
            times = np.arange(t0, t0 + seconds, 1.0 / 200.0)
            block = rec.rps_at(times)
            if min_rps is not None:
                keep = bool(np.nanmin(block) >= float(min_rps))
            if keep and max_rps is not None:
                keep = bool(np.nanmax(block) <= float(max_rps))
        if keep:
            out.append((t0, float(seconds)))
        start += step
    return out


def decimate(clip: Clip, target_sr: int = SR) -> Clip:
    """Resample a clip to ``target_sr`` with a polyphase anti-alias filter.

    ``scipy.signal.resample_poly`` uses a Kaiser-windowed FIR whose transition
    band sits just below the new Nyquist, so the result keeps content up to
    ~0.99 x Nyquist instead of the ~7.9 kHz the published 16 kHz training sets
    stop at. The rps track is decimated by interpolation, which is exact for a
    track already interpolated onto the audio grid.
    """
    if int(clip.sr) == int(target_sr):
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


def recording_ids(
    dataset: str, version: str | None = None, *, split: str | None = None
) -> list[str]:
    """Every recording id in a published frames dataset (decodes each frame).

    ``split`` filters on ``meta.split`` (``"motor"`` for the DREGON bench).
    """
    from data_processing.frames import meta_dict
    from data_processing.streams import decode_tdframe, is_data_sample, open_repository

    dataset, pinned = split_dataset(dataset)
    ds = open_repository().dataset(dataset, version or pinned)
    out: list[str] = []
    for sample in ds.samples():
        if not is_data_sample(sample):
            continue
        if split is None:
            out.append(str(sample[0]))
            continue
        if str(meta_dict(decode_tdframe(sample)).get("split")) == split:
            out.append(str(sample[0]))
    return out


__all__ = [
    "CACHE_DIR",
    "DEFAULT_RPS_KEY",
    "NATIVE_SR",
    "RPS_KEYS",
    "Recording",
    "channel_spec",
    "read_cached_clip",
    "write_cached_clip",
    "decimate",
    "iter_recordings",
    "load_clip",
    "load_recording",
    "recording_ids",
    "resolve_channels",
    "split_dataset",
    "windows",
]
