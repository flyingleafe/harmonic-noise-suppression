"""The ``drone-recording-annotations`` release: packaging, discovery, annotated Frames.

One dload sample per annotated parent ``dataset@version`` (field ``sqlite.gz``),
committed one-sample-per-shard so a reader fetches exactly one parent's
database through the dload cache. Parent audio is read lazily, per selected
key, from the byte span recorded in the database: a ranged read of the parent
shard (or the whole shard through the dload cache when most of it is wanted).
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import Any

import dload
import msgpack
import numpy as np
import soundfile as sf
import tdseries as td
from dload.errors import NotFoundError
from dload.manifest import ShardInfo
from dload.pack import MAGIC, PackReader
from dload.remote import LocalRemote, S3Remote
from dload.repo import _shard_key

from data_processing import streams
from data_processing.frames import audio_series, with_meta
from data_processing.recording_annotations.index import (
    FACET_NAMES,
    Index,
    NotAnnotatedError,
    SampleLocation,
    _plain,
    location_from_entry,
)
from data_processing.recording_annotations.media import (
    ADTS_HEADER_BYTES,
    aac_runs,
    append_adts,
    parse_mp4_audio,
)

RELEASE_NAME = "drone-recording-annotations"
RELEASE_LAYOUT = "recording-annotations-v1"
DB_FIELD = "sqlite.gz"
#: Decompressed databases kept under ``<dload cache>/recording-annotations``.
DB_CACHE_BYTES = 1024**3
#: Parent indexes a :class:`Release` keeps open at once.
MAX_OPEN_INDEXES = 4
#: Above this fraction of a shard's bytes, a batch reads the whole shard
#: through the dload cache instead of ranged requests.
WHOLE_SHARD_FRACTION = 0.5
AUDIO_EXTS = frozenset({"wav", "flac", "ogg", "aif", "aiff", "mp3"})
#: Raw-tree layouts (dload CLI convention: field = file extension); ``None`` =
#: trees committed before manifests carried a layout.
RAW_LAYOUTS = (None, "raw-files", "raw-files-v1")
#: The raw AuDroK tree and its chamber-mic MDM XML containers (exactly 3,276
#: keys in AuDroK@ea0e533e6ce4); every other XML there is a document.
AUDROK_RAW = "AuDroK"
_AUDROK_CONTAINER = re.compile(r"^Measurements_\d+-\d+/\d{2}-[A-Z0-9]{2}-Mi[1-6]$")
#: What ``rps_acoustic_uncertainty`` holds unless the estimator says otherwise.
UNCERTAINTY_KIND = "estimator_reported_scale"
_FOOTER = struct.Struct("<Q")


@dataclass(frozen=True, slots=True)
class ParentInfo:
    dataset: str
    version: str
    key: str
    n_rows: int
    db_bytes: int
    db_sha256: str
    shard: int | None = None  # release shard (dload release only)
    n_tach: int = 0
    n_tach_windows: int = 0


# ─── Parent packs ──────────────────────────────────────────────────────────────


def pack_index(path: str | Path) -> tuple[list[str], list[dict[str, list[int]]]]:
    """``(keys, [{field: [offset, length]}])`` from a dload pack footer."""
    with open(path, "rb") as fh:
        size = fh.seek(0, 2)
        fh.seek(size - 16)
        footer = fh.read(16)
        if footer[8:] != MAGIC:
            raise ValueError(f"{path}: not a dload pack")
        (n,) = _FOOTER.unpack(footer[:8])
        fh.seek(size - 16 - n)
        index = msgpack.unpackb(fh.read(n), raw=False, strict_map_key=False)
    return list(index["keys"]), list(index["samples"])


def located_samples(
    dataset: str,
    version: str | None = None,
    *,
    repo: dload.Repository | None = None,
    shards: Iterable[int] | None = None,
) -> Iterator[tuple[SampleLocation, str, dict[str, bytes]]]:
    """``(location, key, fields)`` of every data sample, shard by shard (pulls
    whole shards through the dload cache — runner/test side, not for laptops)."""
    repo = repo or streams.open_repository()
    manifest = repo.manifest(str(dataset), version)
    wanted = None if shards is None else {int(s) for s in shards}
    for i, shard in enumerate(manifest.shards):
        if wanted is not None and i not in wanted:
            continue
        with repo.open_shard(shard) as path:
            keys, entries = pack_index(path)
            with PackReader(path) as reader:
                for j, key in enumerate(keys):
                    if key.startswith("_"):
                        continue
                    yield location_from_entry(i, entries[j]), key, reader.read(j)[1]


def _retrying(fn: Any) -> Any:
    for attempt in range(streams.SHARD_OPEN_ATTEMPTS):
        try:
            return fn()
        except streams._RETRY_ERRORS:
            if attempt == streams.SHARD_OPEN_ATTEMPTS - 1:
                raise
            time.sleep(min(2.0**attempt, 16.0))
    raise AssertionError("unreachable")  # pragma: no cover


def _read_file_spans(path: Path, spans: Sequence[tuple[int, int]]) -> Iterator[bytes]:
    with open(path, "rb") as fh:
        for off, n in spans:
            fh.seek(off)
            data = fh.read(n)
            if len(data) != n:
                raise OSError(f"{path}: short read at {off} ({len(data)} of {n} bytes)")
            yield data


def read_spans(
    repo: dload.Repository, shard: ShardInfo, spans: Sequence[tuple[int, int]]
) -> Iterator[bytes]:
    """Bytes of ``(offset, length)`` spans of one parent shard, lazily in order.

    A cached shard is read in place; otherwise each span is one ranged read
    (S3 ``Range`` GET / local-remote seek) unless the spans cover more than
    :data:`WHOLE_SHARD_FRACTION` of the shard, which then goes through the
    dload cache (budgeted, shared with training).
    """
    pinned = repo.cache.pin_existing(shard.digest)
    if pinned is None and sum(n for _, n in spans) > WHOLE_SHARD_FRACTION * shard.size:
        pinned = repo.open_shard(shard)
    if pinned is not None:
        with pinned as path:
            yield from _read_file_spans(Path(path), spans)
        return
    remote = repo.remote
    key = _shard_key(shard.digest)
    if isinstance(remote, LocalRemote):
        yield from _read_file_spans(remote.root / key, spans)
        return
    if isinstance(remote, S3Remote):
        for off, n in spans:
            if n == 0:
                yield b""
                continue

            def get(off: int = off, n: int = n) -> bytes:
                resp = remote._s3.get_object(
                    Bucket=remote.bucket,
                    Key=remote._full_key(key),
                    Range=f"bytes={off}-{off + n - 1}",
                )
                return resp["Body"].read()

            data = _retrying(get)
            if len(data) != n:
                raise OSError(
                    f"shard {shard.digest[:12]}: ranged read returned {len(data)} of {n} bytes"
                )
            yield data
        return
    with repo.open_shard(shard) as path:
        yield from _read_file_spans(Path(path), spans)


# ─── Decoding by parent layout ─────────────────────────────────────────────────


def _audio_ext(name: str) -> str | None:
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else name.lower()
    return ext if ext in AUDIO_EXTS else None


def _decode_audio(data: bytes) -> td.Series:
    raw, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return audio_series(np.ascontiguousarray(raw.T), int(sr))


def _is_sound_file(data: bytes) -> bool:
    try:
        sf.info(io.BytesIO(data))
    except (sf.LibsndfileError, RuntimeError, TypeError):
        return False
    return True


def _raw_audio_field(key: str, fields: Mapping[str, bytes], has_audio: bool | None) -> str | None:
    """The audio field of a raw-tree sample (dload CLI convention: field = file
    extension, ``data``/``""`` = extensionless file named by the key)."""
    named = [
        f
        for f in sorted(fields)
        if _audio_ext(f)
        or (f in streams._EXTENSIONLESS_FIELDS and _audio_ext(PurePosixPath(key).name))
    ]
    if len(named) > 1:
        raise ValueError(f"raw sample {key!r} has several audio fields {named}")
    if named:
        return named[0]
    if has_audio:
        sniffed = [f for f in sorted(fields) if _is_sound_file(fields[f])]
        if len(sniffed) != 1:
            raise ValueError(
                f"raw sample {key!r} is annotated has_audio but {len(sniffed)} of its fields "
                f"{sorted(fields)} decode as sound"
            )
        return sniffed[0]
    return None


def _sample_dir_meta(fields: Mapping[str, bytes], fmap: Mapping[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for name in ("meta", "metadata"):
        if name not in fields:
            continue
        if str(fmap.get(name, f"{name}.json")).endswith(".json"):
            decoded = json.loads(fields[name])
            if isinstance(decoded, dict):
                meta.update(decoded)
        else:
            meta[f"{name}_text"] = fields[name].decode("utf-8", errors="replace")
    return meta


def decode_sample(
    layout: str | None,
    key: str,
    fields: Mapping[str, bytes],
    *,
    dataset: str | None = None,
    parent_meta: Mapping[str, Any] | None = None,
    sample_rate: float | None = None,
    has_audio: bool | None = None,
) -> td.Frame:
    """Decode one parent sample with its layout's established convention.

    - ``tdframe-v1``: :func:`streams.decode_tdframe` (entries untouched);
    - raw trees (layout absent, ``raw-files``, ``raw-files-v1``): the audio
      field (by extension, by the key's suffix for extensionless files, else
      sniffed when the index says ``has_audio``) -> ``audio`` ``(time,)``/
      ``(mic, time)``; never an empty Frame for an audio sample. Raw
      ``AuDroK`` chamber-mic MDM XML containers (``Measurements_*/NN-PH-MiK``)
      decode through ``sources.audrok.decode_container`` (mono float32, its
      own rate, only the samples actually present); other XML stays opaque;
    - ``sample-dir-v1``: mixture convention (``decode_dregon_lm`` when ``rps``
      is present) plus every wav stem (``mixture``/``vocals``/``noise``) as its
      own entry; json meta merged, text meta kept as ``<name>_text``;
    - ``pcm16-mono-v1``: int16 ``pcm`` at the stored sample rate.
    """
    parent_meta = parent_meta or {}
    if layout == streams.TDFRAME_LAYOUT:
        return streams.decode_tdframe((key, dict(fields)))
    if layout in RAW_LAYOUTS:
        meta: dict[str, Any] = {"recording_id": key, "source_fields": sorted(fields)}
        entries: dict[str, Any] = {}
        if dataset == AUDROK_RAW and _AUDROK_CONTAINER.match(key) and set(fields) == {"xml"}:
            from data_processing.sources.audrok import decode_container

            mono, header, wav = decode_container(fields["xml"], name=f"{dataset}/{key}")
            entries["audio"] = td.uniform(
                np.array(mono, dtype=np.float32),
                int(wav["sample_rate"]),
                dims=("time",),
                t_start=0.0,
            )
            meta.update(container_header=dict(header), container_wav=_plain(wav))
        elif has_audio and (dataset, key) in MEDIA_AUDIO:
            raise ValueError(
                f"{dataset}/{key} is a media container: decode its audio in place with "
                "media_input(...) + decode_media_frame(...) (Release.frames does), not from bytes"
            )
        else:
            field = _raw_audio_field(key, fields, has_audio)
            if field is not None:
                entries["audio"] = _decode_audio(fields[field])
        entries["meta"] = td.Frame(meta)
        return td.Frame(entries)
    if layout == "sample-dir-v1":
        fmap = dict(parent_meta.get("fields") or {})
        stems = [f for f in sorted(fields) if _audio_ext(str(fmap.get(f, f)))]
        meta = _sample_dir_meta(fields, fmap)
        if "mixture" in fields and "rps" in fields:
            sr = int(sf.info(io.BytesIO(fields["mixture"])).samplerate)
            frame = streams.decode_dregon_lm(
                (key, {"mixture": fields["mixture"], "rps": fields["rps"]}), sample_rate=sr
            )
            meta = {**meta, "recording_id": key}
        else:
            frame = td.Frame({})
            meta.setdefault("recording_id", key)
        frame = frame.with_entry("meta", td.Frame(meta))
        for stem in stems:
            if stem not in frame:
                frame = frame.with_entry(stem, _decode_audio(fields[stem]))
        if has_audio and not stems:
            raise ValueError(
                f"sample-dir sample {key!r} is annotated has_audio but has no audio stem"
            )
        return frame
    if layout == "pcm16-mono-v1":
        sr = sample_rate or parent_meta.get("sample_rate")
        if not sr:
            raise ValueError(f"pcm16 sample {key!r}: sample rate unknown")
        pcm = np.frombuffer(fields["pcm"], dtype=np.int16).astype(np.float32) / 32767.0
        return td.Frame(
            {
                "audio": td.uniform(pcm, int(sr), dims=("time",), t_start=0.0),
                "meta": td.Frame({"recording_id": key}),
            }
        )
    raise ValueError(f"unsupported parent layout {layout!r}")


# ─── Audio tracks inside raw media containers ─────────────────────────────────

#: Raw media files whose audio track is genuine source audio, by exact
#: ``(dataset, key) -> (field, bytes)``: the seven AVQ-raw GoPro MP4s (AAC 48 kHz;
#: S1 stereo, S2 mono; measured on uni-cpu, see ``legacy._AVQ_GOPRO_AUDIO``).
#: Any other video stays opaque.
MEDIA_AUDIO: dict[tuple[str, str], tuple[str, int]] = {
    ("AVQ-raw", "S1__seq4__Video"): ("mp4", 331_984_292),
    ("AVQ-raw", "S2__seq3__video"): ("mp4", 28_055_089),
    ("AVQ-raw", "S2__seq4__video"): ("mp4", 27_783_331),
    ("AVQ-raw", "S2__seq5__video"): ("mp4", 42_778_765),
    ("AVQ-raw", "S2__seq6__video"): ("mp4", 44_836_957),
    ("AVQ-raw", "S2__seq7__video"): ("mp4", 40_556_438),
    ("AVQ-raw", "S2__seq8__video"): ("mp4", 42_953_072),
}
_MAX_AAC_BYTES = 32 << 20
_MAX_PCM_BYTES = 384 << 20
_AAC_READ_WORKERS = 8


def _media_tool(name: str) -> str:
    exe = shutil.which(name)
    if exe is None:
        raise FileNotFoundError(f"{name} not on PATH (uni: `module load ffmpeg`)")
    return exe


def _run_media(stage: str, args: list[str], payload: bytearray | None = None) -> bytes:
    """FFmpeg only sees an AAC pipe, never a URL or object-store credentials."""
    proc = subprocess.run(
        args, input=payload, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"media {stage} failed: {Path(args[0]).name} exited {proc.returncode}"
        ) from None
    return proc.stdout


def media_field(dataset: str, key: str, loc: SampleLocation | None) -> tuple[str, int, int] | None:
    """``(field, offset in shard, length)`` of a listed media audio container, else None."""
    spec = MEDIA_AUDIO.get((str(dataset), str(key)))
    if spec is None or loc is None:
        return None
    for name, rel, length in loc.fields:
        if name == spec[0] and length == spec[1]:
            return name, loc.offset + rel, length
    return None


@dataclass(frozen=True, slots=True)
class MediaInput:
    """The length and a half-open, field-relative range reader of an MP4."""

    length: int
    read: Callable[[int, int], bytes]


@contextlib.contextmanager
def media_input(
    repo: dload.Repository, shard: ShardInfo, offset: int, length: int
) -> Iterator[MediaInput]:
    """Read an MP4 field in place; neither its video nor a cold shard is cached."""
    if offset < 0 or length <= 0 or offset + length > shard.size:
        raise ValueError("media field falls outside its parent shard")

    def check(start: int, end: int) -> None:
        if not 0 <= start < end <= length:
            raise ValueError("media range falls outside the MP4 field")

    @contextlib.contextmanager
    def local(path: Path) -> Iterator[MediaInput]:
        with path.open("rb") as file:

            def read(start: int, end: int) -> bytes:
                check(start, end)
                data = os.pread(file.fileno(), end - start, offset + start)
                if len(data) != end - start:
                    raise OSError("short read of local MP4 field")
                return data

            yield MediaInput(length, read)

    pinned = repo.cache.pin_existing(shard.digest)
    if pinned is not None:
        with pinned as path, local(path) as source:
            yield source
        return

    remote = repo.remote
    key = _shard_key(shard.digest)
    if isinstance(remote, LocalRemote):
        with local(remote.root / key) as source:
            yield source
    elif isinstance(remote, S3Remote):
        client = remote._s3
        object_key = remote._full_key(key)

        def read(start: int, end: int) -> bytes:
            check(start, end)
            begin, last = offset + start, offset + end - 1

            def get() -> bytes:
                response = client.get_object(
                    Bucket=remote.bucket, Key=object_key, Range=f"bytes={begin}-{last}"
                )
                with contextlib.closing(response["Body"]) as body:
                    if response.get("ContentRange") != f"bytes {begin}-{last}/{shard.size}":
                        raise OSError("R2 returned an unexpected Content-Range for MP4 audio")
                    data = body.read()
                    if len(data) != end - start:
                        raise OSError("short R2 read of MP4 audio")
                    return data

            return _retrying(get)

        yield MediaInput(length, read)
    else:
        raise TypeError(f"no ranged media input for remote {type(remote).__name__}")


def decode_media_audio(source: MediaInput) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Decode only the first AAC track, continuously, on the MP4 presentation clock.

    Read the MP4 tables and contiguous AAC chunks by range and wrap the access
    units in ADTS for local ffmpeg. Trim priming and the final AAC padding
    according to the tables. The all-frame decode preserves inter-frame state.
    """
    track = parse_mp4_audio(source.read, source.length)
    n_frames = len(track.sizes)
    if int(track.sizes.sum()) + ADTS_HEADER_BYTES * n_frames > _MAX_AAC_BYTES:
        raise ValueError("MP4 AAC track exceeds the compressed audio read limit")
    if 4 * track.channels * n_frames * 1024 > _MAX_PCM_BYTES:
        raise ValueError("MP4 AAC track exceeds the decoded audio limit")
    ffmpeg = _media_tool("ffmpeg")

    def fetch(run: tuple[int, int]) -> bytes:
        a, b = run
        return source.read(int(track.offsets[a]), int(track.offsets[b - 1] + track.sizes[b - 1]))

    adts = bytearray()
    runs = iter(aac_runs(track, 0, n_frames))
    with ThreadPoolExecutor(max_workers=_AAC_READ_WORKERS) as pool:
        # Keep only one batch of completed ranges resident, in track order.
        while batch := list(islice(runs, _AAC_READ_WORKERS)):
            for (a, b), data in zip(batch, pool.map(fetch, batch), strict=True):
                append_adts(adts, track, a, b, data)
    raw = _run_media(
        "audio stream a:0 decode",
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-f",
            "aac",
            "-i",
            "pipe:0",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ],
        adts,
    )
    if not raw or len(raw) % (4 * track.channels):
        raise ValueError("empty or truncated AAC decoder output")
    pcm = np.frombuffer(raw, dtype="<f4").reshape(-1, track.channels)
    end = int(track.starts[-1])
    if not end <= len(pcm) < end + 1024:
        raise ValueError(f"AAC decoder returned {len(pcm)} samples; MP4 tables specify {end}")
    audio = pcm[track.skip : end].T
    info = {
        "codec": "aac",
        "channels": track.channels,
        "sample_rate": track.sr,
        "t_start_s": track.delay_s,
        "audio_duration_s": audio.shape[1] / track.sr,
        "decoded_samples": audio.shape[1],
        "decoder": _run_media("decoder version", [ffmpeg, "-version"]).decode().splitlines()[0],
    }
    return audio, track.sr, info


def decode_media_frame(source: MediaInput, key: str, fields: Sequence[str]) -> td.Frame:
    """The canonical Frame of a listed media container: its audio track on the
    container clock (``audio`` ``(time,)``/``(mic, time)``) plus ``meta.media``."""
    audio, sr, info = decode_media_audio(source)
    series = audio_series(audio, sr)
    if info["t_start_s"]:
        series = series.shift(float(info["t_start_s"]))
    meta = {"recording_id": key, "source_fields": sorted(fields), "media": _plain(info)}
    return td.Frame({"audio": series, "meta": td.Frame(meta)})


# ─── The annotated view ────────────────────────────────────────────────────────


def _has_time(frame: td.Frame) -> bool:
    for name in frame:
        entry = frame[name]
        if isinstance(entry, td.Series) and entry.has_time:
            return True
        if isinstance(entry, td.Frame) and _has_time(entry):
            return True
    return False


def acoustic_series(
    tach: Mapping[str, Any], t_start: float, t_end: float
) -> dict[str, td.Series] | None:
    """The ``rps_acoustic*`` entries from :meth:`Index.tach_arrays` output.

    One stamp per ATTEMPTED window at its absolute centre — rejected windows
    stay as all-NaN columns, so gaps (motor off, refusals) are explicit and no
    consumer should interpolate across them. Dim ``rotor`` only for
    ``identity == "physical"``; otherwise ``rank``: the ascending rate rank
    WITHIN each window, not a tracked rotor. ``None`` when no accepted window
    resolved a rate (then no acoustic speed is claimed at all).
    """
    rates, unc, accepted = tach["rates"], tach["uncertainty"], tach["accepted"]
    shown = np.where(accepted[:, None], rates, np.nan).astype(np.float32)
    if not np.isfinite(shown).any():
        return None
    t0 = float(tach.get("audio_t_start_s") or 0.0)
    times = tach["times"].astype(np.float64) + t0
    centers = times[:, 2]
    if centers[0] < t_start - 1e-6 or centers[-1] >= t_end + 1e-6:
        raise ValueError(
            f"tachometry centres [{centers[0]:.3f}, {centers[-1]:.3f}] s fall outside the Frame "
            f"domain [{t_start:.3f}, {t_end:.3f}) s — clock mismatch"
        )
    centers = np.clip(centers, t_start, np.nextafter(t_end, -np.inf))
    index = td.StampIndex.from_stamps(centers, t_start=float(t_start), t_end=float(t_end))
    dim = "rotor" if tach.get("identity") == "physical" else "rank"
    indexes: dict[str, Any] = {"time": index}
    rotor_ids = (tach.get("summary") or {}).get("rotor_ids")
    if dim == "rotor" and rotor_ids is not None:
        if len(rotor_ids) != shown.shape[1]:
            raise ValueError(
                f"rotor_ids {rotor_ids} do not name the {shown.shape[1]} physical slots"
            )
        indexes["rotor"] = td.LabelIndex(tuple(str(r) for r in rotor_ids))
    # Anonymous ranks never get labels: a rank is not a motor identity.
    shown_unc = np.where(np.isfinite(shown), unc, np.nan).astype(np.float32)
    return {
        "rps_acoustic": td.Series(np.ascontiguousarray(shown.T), (dim, "time"), indexes),
        "rps_acoustic_uncertainty": td.Series(
            np.ascontiguousarray(shown_unc.T), (dim, "time"), dict(indexes)
        ),
        "rps_acoustic_window": td.Series(
            np.ascontiguousarray(times[:, :2].T), ("bound", "time"), {"time": index}
        ),
        "rps_acoustic_accepted": td.Series(accepted.copy(), ("time",), {"time": index}),
    }


def annotate_frame(
    frame: td.Frame | None,
    record: Mapping[str, Any],
    tach: Mapping[str, Any] | None,
    release: Mapping[str, Any],
) -> td.Frame:
    """Join one stored record (``Index.record(key, windows=False)``) and its
    ``Index.tach_arrays(key)`` onto a decoded parent Frame (``None`` =
    metadata-only). Existing entries — native ``rps``/measured/command
    tracks — are never touched; acoustic estimates only add ``rps_acoustic*``."""
    conditions = dict(record)
    if conditions.get("tach") is not None:
        summary = conditions["tach"].get("summary") or {}
        conditions["tach"] = {
            **conditions["tach"],
            "uncertainty_kind": summary.get("uncertainty_kind", UNCERTAINTY_KIND),
        }
    conditions["annotation_release"] = dict(release)
    if frame is None:
        frame = td.Frame({"meta": td.Frame({"recording_id": record["key"]})})
    frame = with_meta(frame, recording_conditions=_plain(conditions))
    if tach is None or not tach["n_windows"]:
        return frame
    if _has_time(frame):
        t0, t1 = float(frame.t_start), float(frame.t_end)
    else:
        t0 = float(tach.get("audio_t_start_s") or 0.0)
        duration = (record.get("info") or {}).get("duration_s")
        last = t0 + float(tach["times"][-1, 2])
        t1 = max(t0 + float(duration), last + 1e-6) if duration else last + 1e-6
    series = acoustic_series(tach, t0, t1)
    for name, s in (series or {}).items():
        if name in frame:
            raise ValueError(f"parent Frame already has an entry named {name!r}")
        frame = frame.with_entry(name, s)
    return frame


# ─── Releases ──────────────────────────────────────────────────────────────────


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _db_parent_info(path: Path) -> tuple[ParentInfo, Index]:
    idx = Index(path)
    meta = idx.meta()
    info = ParentInfo(
        dataset=idx.dataset,
        version=idx.version,
        key=f"{idx.dataset}@{idx.version}",
        n_rows=idx.n_rows,
        db_bytes=path.stat().st_size,
        db_sha256=_file_sha256(path),
        n_tach=int(meta.get("n_tach", 0)),
        n_tach_windows=int(meta.get("n_tach_windows", 0)),
    )
    return info, idx


class Release:
    """A set of annotation databases: a dload release version or a staging dir.

    ``repo`` serves the release shards (dload source) and the parent audio.
    """

    def __init__(
        self,
        parents: Sequence[ParentInfo],
        *,
        repo: dload.Repository | None,
        name: str = RELEASE_NAME,
        version: str | None = None,
        manifest: dload.Manifest | None = None,
        root: Path | None = None,
    ) -> None:
        self.name = name
        self.version = version
        self.repo = repo
        self._manifest = manifest
        self._root = root
        self._parents = {(p.dataset, p.version): p for p in parents}
        self._indexes: OrderedDict[str, Index] = OrderedDict()
        self._lock = threading.RLock()

    def __repr__(self) -> str:
        where = f"@{self.version[:12]}" if self.version else f" (staging {self._root})"
        return f"Release({self.name}{where}, {len(self._parents)} parents)"

    @property
    def _repo(self) -> dload.Repository:
        return self.repo or streams.open_repository()

    def parents(self) -> list[ParentInfo]:
        return sorted(self._parents.values(), key=lambda p: p.key)

    def _resolve_parent(self, dataset: str, version: str | None) -> str:
        if version is not None and len(version) == 64:
            return version
        return self._repo.resolve(str(dataset), version)

    def _db_path(self, parent: ParentInfo) -> Path:
        """Local path of one parent's database. dload releases extract into the
        version-addressed (``db_sha256``) cache dir, bounded by
        :data:`DB_CACHE_BYTES`; the transport shard is evicted afterwards so
        nothing is stored twice."""
        if self._root is not None:
            return self._root / f"{parent.key}.sqlite"
        repo = self._repo
        cache = Path(repo.cache.root) / "recording-annotations"
        path = cache / f"{parent.db_sha256}.sqlite"
        if path.exists():
            os.utime(path)
            return path
        assert self._manifest is not None and parent.shard is not None
        shard = self._manifest.shards[parent.shard]
        cache.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{parent.db_sha256[:16]}.", suffix=".tmp", dir=cache
        )
        os.close(fd)
        tmp = Path(tmp_name)
        digest = hashlib.sha256()
        try:
            with repo.open_shard(shard) as shard_path, PackReader(shard_path) as reader:
                gz = reader.read_key(parent.key)[1][DB_FIELD]
            with gzip.GzipFile(fileobj=io.BytesIO(gz)) as src, open(tmp, "wb") as dst:
                while chunk := src.read(1 << 20):
                    digest.update(chunk)
                    dst.write(chunk)
            if digest.hexdigest() != parent.db_sha256:
                raise ValueError(
                    f"{self.name}: database of {parent.key} does not match its manifest sha256"
                )
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        repo.cache.evict(shard.digest)
        _prune(cache, keep=path)
        return path

    def index(self, dataset: str, version: str | None = None) -> Index:
        """The annotation index of the EXACT parent version (``None`` = the
        parent's dload.lock pin). Raises :class:`NotAnnotatedError` otherwise.
        Only this parent's database is fetched; at most
        :data:`MAX_OPEN_INDEXES` stay referenced by the release."""
        full = self._resolve_parent(dataset, version)
        parent = self._parents.get((str(dataset), full))
        if parent is None:
            have = sorted(v[:12] for d, v in self._parents if d == str(dataset))
            raise NotAnnotatedError(
                f"{dataset}@{full[:12]} has no annotations in {self!r} (annotated versions: {have or 'none'})"
            )
        with self._lock:  # one extraction per parent even under threaded callers
            idx = self._indexes.get(parent.key)
            if idx is None:
                idx = Index(self._db_path(parent))
                if (idx.dataset, idx.version) != (parent.dataset, parent.version):
                    raise ValueError(
                        f"{parent.key}: database holds {idx.dataset}@{idx.version[:12]}"
                    )
                self._indexes[parent.key] = idx
                while len(self._indexes) > MAX_OPEN_INDEXES:
                    self._indexes.popitem(last=False)  # dropped; closes once unreferenced
            self._indexes.move_to_end(parent.key)
            return idx

    def close(self) -> None:
        """Close every database this release opened."""
        with self._lock:
            for idx in self._indexes.values():
                idx.close()
            self._indexes.clear()

    def _release_meta(self, idx: Index) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "parent_dataset": idx.dataset,
            "parent_version": idx.version,
        }

    def frames(
        self, idx: Index, keys: Sequence[str], *, audio: bool = True
    ) -> Iterator[tuple[str, td.Frame]]:
        """Annotated Frames of ``keys`` in parent storage order (shard, offset)."""
        release = self._release_meta(idx)
        if not audio:
            for key in keys:
                record = idx.record(key, windows=False)
                if record is None:
                    raise KeyError(f"{idx.dataset}@{idx.version[:12]}: no annotated key {key!r}")
                yield key, annotate_frame(None, record, idx.tach_arrays(key), release)
            return
        located = idx.locations(keys)
        missing = [k for k, (loc, _) in located.items() if loc is None]
        if missing:
            raise ValueError(f"{idx.dataset}: no parent byte location stored for {missing[:3]}")
        shards = idx.parent_shards()
        parent_meta = idx.meta().get("parent_meta") or {}
        by_shard: dict[int, list[tuple[str, SampleLocation, dict[str, Any]]]] = {}
        for key, (loc, info) in located.items():
            assert loc is not None
            by_shard.setdefault(loc.shard, []).append((key, loc, info))
        for shard_i in sorted(by_shard):
            group = sorted(by_shard[shard_i], key=lambda item: item[1].offset)
            shard = ShardInfo(*shards[shard_i])
            # Listed media containers are decoded in place by ffmpeg (ranged), never
            # read into memory; everything else is one ranged read per sample.
            media = {
                key: media_field(idx.dataset, key, loc) if info.get("has_audio") else None
                for key, loc, info in group
            }
            plain = [(loc.offset, loc.length) for key, loc, _ in group if media[key] is None]
            payloads = read_spans(self._repo, shard, plain) if plain else iter(())
            for key, loc, info in group:
                spec = media[key]
                if spec is not None:
                    with media_input(self._repo, shard, spec[1], spec[2]) as source:
                        frame = decode_media_frame(source, key, [name for name, _, _ in loc.fields])
                else:
                    data = next(payloads)
                    fields = {name: data[off : off + length] for name, off, length in loc.fields}
                    frame = decode_sample(
                        idx.layout,
                        key,
                        fields,
                        dataset=idx.dataset,
                        parent_meta=parent_meta,
                        sample_rate=info.get("sample_rate"),
                        has_audio=info.get("has_audio"),
                    )
                record = idx.record(key, windows=False)
                assert record is not None
                yield key, annotate_frame(frame, record, idx.tach_arrays(key), release)


def _prune(cache: Path, *, keep: Path) -> None:
    files = sorted(cache.glob("*.sqlite"), key=lambda p: p.stat().st_mtime, reverse=True)
    total = 0
    for path in files:
        total += path.stat().st_size
        if total > DB_CACHE_BYTES and path != keep:
            path.unlink(missing_ok=True)


def _dir_release(root: Path, repo: dload.Repository | None) -> Release:
    parents = []
    for path in sorted(root.glob("*.sqlite")):
        info, idx = _db_parent_info(path)
        idx.close()
        if path.name != f"{info.key}.sqlite":
            raise ValueError(f"{path}: staged database must be named {info.key}.sqlite")
        parents.append(info)
    return Release(parents, repo=repo, root=root)


_RELEASES: dict[tuple[str | None, int], Release] = {}
_RELEASES_LOCK = threading.Lock()


def open_release(
    source: str | Path | None = None,
    *,
    version: str | None = None,
    repo: dload.Repository | None = None,
) -> Release:
    """The annotation release: ``source=None`` -> dload ``drone-recording-annotations``
    (``version`` or its dload.lock pin, else refs/latest; cached per process);
    a directory -> the staged ``<dataset>@<version>.sqlite`` files, rescanned
    on every call (staging dirs grow)."""
    if source is not None:
        return _dir_release(Path(source), repo)
    cache_key = (version, id(repo))
    with _RELEASES_LOCK:
        found = _RELEASES.get(cache_key)
        if found is not None:
            return found
        r = repo or streams.open_repository()
        manifest = r.manifest(RELEASE_NAME, version)
        meta = manifest.meta if isinstance(manifest.meta, dict) else {}
        if meta.get("layout") != RELEASE_LAYOUT:
            raise ValueError(
                f"{RELEASE_NAME}@{manifest.version[:12]}: layout {meta.get('layout')!r}"
            )
        parents = [
            ParentInfo(**{k: p[k] for k in ParentInfo.__slots__ if k in p}) for p in meta["parents"]
        ]
        release = Release(parents, repo=repo, version=manifest.version, manifest=manifest)
        _RELEASES[cache_key] = release
        return release


def publish(
    staging_dir: str | Path,
    *,
    repo: dload.Repository | None = None,
    carry_forward: bool = True,
    base_version: str | None = None,
    recipe: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dload.Manifest:
    """Commit a new ``drone-recording-annotations`` version from staged
    databases. With ``carry_forward`` every parent DB of the base release
    (``base_version`` or the current pin/latest) that is not restaged is kept
    byte-for-byte (same shard, nothing re-uploaded). Each staged DB must name
    an existing parent version whose shards match the ones it recorded.
    ``recipe`` (the reproducing command) and ``provenance`` (code snapshot,
    rule/estimator revisions) go to the manifest; each parent entry also
    carries its DB's own ``source_commit``/``provenance``. Pinning stays manual."""
    repo = repo or streams.open_repository()
    samples: dict[str, bytes] = {}
    parents: dict[str, dict[str, Any]] = {}
    facets = None
    for path in sorted(Path(staging_dir).glob("*.sqlite")):
        info, idx = _db_parent_info(path)
        facets = facets or idx.facets
        db_meta = idx.meta()
        recorded = idx.parent_shards()
        idx.close()
        if path.name != f"{info.key}.sqlite":
            raise ValueError(f"{path}: staged database must be named {info.key}.sqlite")
        actual = [
            (s.digest, s.size, s.num_samples)
            for s in repo.manifest(info.dataset, info.version).shards
        ]
        if recorded and recorded != actual:
            raise ValueError(f"{info.key}: recorded parent shards differ from the parent manifest")
        samples[info.key] = gzip.compress(path.read_bytes(), compresslevel=9, mtime=0)
        parents[info.key] = {
            **{k: getattr(info, k) for k in ParentInfo.__slots__ if k != "shard"},
            "source_commit": db_meta.get("source_commit"),
            "provenance": db_meta.get("provenance") or {},
            "created": db_meta.get("created"),
        }
    if carry_forward:
        try:
            base = repo.manifest(RELEASE_NAME, base_version)
        except NotFoundError:
            if base_version is not None:
                raise
            base = None
        if base is not None:
            for p in base.meta.get("parents", []):
                if p["key"] in samples:
                    continue
                with (
                    repo.open_shard(base.shards[p["shard"]]) as shard_path,
                    PackReader(shard_path) as reader,
                ):
                    gz = reader.read_key(p["key"])[1][DB_FIELD]
                if _sha256(gzip.decompress(gz)) != p["db_sha256"]:
                    raise ValueError(f"carried {p['key']}: database does not match base manifest")
                samples[p["key"]] = gz
                parents[p["key"]] = {k: v for k, v in p.items() if k != "shard"}
                facets = facets or base.meta.get("facets")
    if not samples:
        raise ValueError(f"nothing to publish from {staging_dir}")
    order = sorted(samples)
    meta = {
        "layout": RELEASE_LAYOUT,
        "schema_version": 1,
        "facets": facets or {},
        "facet_names": list(FACET_NAMES),
        "provenance": _plain(dict(provenance or {})),
        "parents": [{**parents[k], "shard": i} for i, k in enumerate(order)],
    }
    manifest = repo.commit(
        RELEASE_NAME,
        ((k, {DB_FIELD: samples[k]}) for k in order),
        meta=meta,
        recipe=recipe,
        target_shard_size=1,
    )
    if len(manifest.shards) != len(order) or any(s.num_samples != 1 for s in manifest.shards):
        raise RuntimeError(
            f"{RELEASE_NAME}@{manifest.version[:12]}: expected one parent DB per shard"
        )
    _RELEASES.clear()
    return manifest


# ─── Module-level conveniences ────────────────────────────────────────────────


def _release(release: Release | None) -> Release:
    return release if release is not None else open_release()


def select(
    dataset: str,
    filters: Mapping[str, Any] | None = None,
    *,
    version: str | None = None,
    release: Release | None = None,
    limit: int = 100,
    offset: int = 0,
    order: str = "key",
) -> list[dict[str, Any]]:
    """Rows of one parent matching ``filters`` (no audio touched)."""
    return (
        _release(release)
        .index(dataset, version)
        .select(filters, limit=limit, offset=offset, order=order)
    )


def select_all(
    filters: Mapping[str, Any] | None = None,
    *,
    datasets: Iterable[str] | None = None,
    release: Release | None = None,
    limit_per_parent: int = 100,
) -> dict[str, list[dict[str, Any]]]:
    """Per-parent limited query over every annotated parent at its pinned version
    (``datasets`` restricts; parents whose pin is not annotated are skipped)."""
    rel = _release(release)
    wanted = None if datasets is None else set(datasets)
    out: dict[str, list[dict[str, Any]]] = {}
    for dataset in sorted({p.dataset for p in rel.parents()}):
        if wanted is not None and dataset not in wanted:
            continue
        try:
            idx = rel.index(dataset)
        except NotAnnotatedError:
            continue
        out[dataset] = idx.select(filters, limit=limit_per_parent)
    return out


def load_frame(
    dataset: str,
    key: str,
    *,
    version: str | None = None,
    release: Release | None = None,
    audio: bool = True,
) -> td.Frame:
    """The canonical annotated Frame of one parent sample (see the package doc)."""
    rel = _release(release)
    idx = rel.index(dataset, version)
    ((_, frame),) = rel.frames(idx, [str(key)], audio=audio)
    return frame


def iter_frames(
    dataset: str,
    filters: Mapping[str, Any] | None = None,
    *,
    version: str | None = None,
    release: Release | None = None,
    limit: int | None = None,
    audio: bool = True,
) -> Iterator[tuple[str, td.Frame]]:
    """Annotated Frames of the keys matching ``filters`` (filtered in SQLite
    first; only matching samples' bytes are read), in parent storage order."""
    rel = _release(release)
    idx = rel.index(dataset, version)
    keys = list(islice(idx.keys(filters), limit))
    yield from rel.frames(idx, keys, audio=audio)
