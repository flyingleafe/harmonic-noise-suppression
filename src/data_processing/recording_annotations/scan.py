"""Bounded, read-only access to published dload samples for the annotation runner.

The annotation campaign must classify every recording from its metadata and
measure (tachometry / wind) only the selected ones, without pulling whole
training shards. A dload pack keeps its msgpack index in the footer, so one
ranged read gives every key and field span; the ``tdframe-v1`` ``_frame``
descriptor plus the ``.npy`` headers then give the full Series map (dims,
shape, dtype, rate, time offset) and the unwrapped ``meta`` without touching
PCM. Audio is read as time views (per-channel byte ranges for ``(mic, time)``
C-order arrays, one range for time-major ones; WAV/FLAC fields through a
seekable ranged file object), so a long ultrasonic take never sits in RAM.

Bytes come from a local copy of the shard when one exists under a given root
(``<root>/shards/<xx>/<digest>``: a ``LocalRemote`` stage or a dload cache)
and otherwise from S3 ``Range`` GETs on the dload remote. Nothing is written
anywhere.
"""

from __future__ import annotations

import ast
import io
import json
import os
import struct
import subprocess
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgpack
import numpy as np

#: Leaf-name tokens of audio-rate Series that are not waveforms.
_NOT_AUDIO_TOKENS = (
    "time",
    "stamp",
    "clock",
    "rps",
    "rpm",
    "index",
    "label",
    "mask",
    "vad",
    "count",
)
_MAX_AUDIO_CHANNELS = 512

__all__ = [
    "AUDIO_MIN_RATE_HZ",
    "RemoteRanges",
    "SampleScan",
    "SeriesRef",
    "ShardBytes",
    "ShardOpener",
    "finite_runs",
    "read_view",
    "scan_sample",
    "scan_shard",
]

_MAGIC = b"DLOADPK1"
_FOOTER = 16
_TICKS = 1_000_000_000
#: Uniform Series at or above this rate are audio; telemetry tracks are far below.
AUDIO_MIN_RATE_HZ = 2000.0
#: Coalesce ranged reads whose gap is below this many bytes.
_GAP = 1 << 20
#: Never coalesce one GET beyond this many bytes.
_MAX_RANGE = 64 << 20
#: Meta Series up to this size are unwrapped into lists; bigger ones are summarized.
_META_SERIES_MAX = 256 << 10
#: Stamp index arrays up to this size are read to get the track span/rate.
_STAMPS_MAX = 16 << 20
_SOUNDFILE_EXT = frozenset({"wav", "flac", "ogg", "aif", "aiff", "mp3"})
#: dload CLI field names of extensionless files (``streams._EXTENSIONLESS_FIELDS``).
_EXTENSIONLESS = frozenset({"data", ""})
#: Range block used when sniffing unknown fields for an audio header.
_SNIFF_BLOCK = 64 << 10
#: Largest XML field decoded as a base64-WAV measurement container (AuDroK MDM).
_XML_CONTAINER_MAX = 256 << 20
#: Video containers whose audio stream is read through ffmpeg (job-local transient copy).
_VIDEO_EXT = frozenset({"mp4", "mov", "m4a"})
_MATERIALIZED: dict[tuple[str, int], Path] = {}


def _materialize(shard: ShardBytes, off: int, length: int, ext: str) -> Path:
    """One transient local copy of a container field per process (ffmpeg needs seeks);
    under ``$TMPDIR``, removed at exit. Never uploaded anywhere."""
    import atexit
    import tempfile

    key = (shard.digest, int(off))
    path = _MATERIALIZED.get(key)
    if path is None or not path.exists():
        fd, name = tempfile.mkstemp(suffix=f".{ext}")
        with os.fdopen(fd, "wb") as fh:
            for start in range(0, length, 64 << 20):
                fh.write(shard.read(off + start, min(64 << 20, length - start)))
        path = Path(name)
        _MATERIALIZED[key] = path
        atexit.register(path.unlink, missing_ok=True)
    return path


def _ffmpeg_ref(
    shard: ShardBytes, key: str, name: str, off: int, length: int, ext: str
) -> SeriesRef:
    path = _materialize(shard, off, length, ext)
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,duration,start_time",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(probe.stdout).get("streams") or []
    if not streams:
        raise ValueError(f"{key}: container has no audio stream")
    st = streams[0]
    sr, ch, dur = float(st["sample_rate"]), int(st["channels"]), float(st["duration"])
    # container-clock start = best-effort pts of the first decoded audio frame, the shared
    # loader's rule (ra.decode_media_audio): priming packets are dropped by the decoder
    frames = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-read_intervals",
            "%+#8",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    first = [
        f.get("best_effort_timestamp_time") for f in json.loads(frames.stdout).get("frames") or []
    ]
    t0 = float(next((v for v in first if v not in (None, "N/A")), 0.0))
    n = int(round(dur * sr))
    return SeriesRef(
        path=name,
        field=name,
        container="ffmpeg",
        shape=(ch, n),
        dtype="<f4",
        dims=("mic", "time"),
        offset=off,
        length=length,
        time_axis=1,
        index="grid",
        rate_hz=sr,
        t_start_s=t0,
        t_end_s=t0 + n / sr,
        n_time=n,
    )


def _ffmpeg_window(
    shard: ShardBytes, ref: SeriesRef, i0: int, n: int, chans: Sequence[int]
) -> np.ndarray:
    """Float PCM ``(C, n)`` of the first audio stream from sample ``i0`` (decoded, not resampled)."""
    ext = ref.path.rsplit(".", 1)[-1] if "." in ref.path else "mp4"
    path = _materialize(shard, ref.offset, ref.length, ext if ext in _VIDEO_EXT else "mp4")
    rate = float(ref.rate_hz or 0.0)
    nch = ref.n_channels
    proc = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{float(ref.t_start_s or 0.0) + i0 / rate:.9f}",
            "-t",
            f"{(n + 1) / rate:.9f}",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-vn",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ar",
            str(int(rate)),
            "-ac",
            str(nch),
            "-",
        ],
        check=True,
        capture_output=True,
    )
    pcm = np.frombuffer(proc.stdout, dtype="<f4")
    pcm = pcm[: (pcm.size // nch) * nch].reshape(-1, nch)[:n]
    return np.ascontiguousarray(pcm[:, list(chans)].T, dtype=np.float32)


# ─── byte access ───────────────────────────────────────────────────────────


class RemoteRanges:
    """S3 ``Range`` GETs on the dload remote; one boto3 client per process."""

    def __init__(self, endpoint_url: str, bucket: str, prefix: str = "") -> None:
        self.endpoint_url = endpoint_url
        self.bucket = bucket
        self.prefix = f"{prefix.strip('/')}/" if prefix.strip("/") else ""
        self._client: Any = None
        self._pid = -1
        self.n_requests = 0
        self.n_bytes = 0

    @classmethod
    def from_repository(cls, repo: Any) -> RemoteRanges:
        remote = repo.remote
        return cls(remote.endpoint_url, remote.bucket, getattr(remote, "prefix", ""))

    def _s3(self) -> Any:
        if self._client is None or self._pid != os.getpid():
            import boto3

            self._client = boto3.Session().client("s3", endpoint_url=self.endpoint_url)
            self._pid = os.getpid()
        return self._client

    def get(self, key: str, offset: int, length: int) -> bytes:
        if length <= 0:
            return b""
        rng = f"bytes={offset}-{offset + length - 1}"
        for attempt in range(6):
            try:
                resp = self._s3().get_object(Bucket=self.bucket, Key=self.prefix + key, Range=rng)
                data = resp["Body"].read()
                if len(data) != length:
                    raise OSError(f"short ranged read {key} {rng}: {len(data)} B")
                self.n_requests += 1
                self.n_bytes += length
                return data
            except Exception:  # noqa: BLE001 — transient network errors retry, then raise
                if attempt == 5:
                    raise
                self._client = None
                time.sleep(min(30.0, 0.5 * 2**attempt))
        raise AssertionError("unreachable")


class ShardBytes:
    """Random access to one immutable shard (local file or remote ranges)."""

    def __init__(self, digest: str, size: int, *, path: Path | None, remote: RemoteRanges | None):
        if path is None and remote is None:
            raise ValueError(f"shard {digest[:12]} has neither a local copy nor a remote")
        self.digest = digest
        self.size = int(size)
        self.path = path
        self.remote = remote
        self._fh: Any = None
        self._buf: bytes | None = None
        self._index: tuple[list[str], list[dict[str, list[int]]]] | None = None

    @property
    def source(self) -> str:
        return "local" if self.path is not None else "remote"

    def prefetch(self) -> None:
        """Hold the whole (small, remote) shard in memory: one GET for dense clip packs."""
        if self.path is None and self._buf is None:
            assert self.remote is not None
            self._buf = self.remote.get(f"shards/{self.digest[:2]}/{self.digest}", 0, self.size)

    def read(self, offset: int, length: int) -> bytes:
        if offset < 0 or length < 0 or offset + length > self.size:
            raise ValueError(
                f"range {offset}+{length} outside shard {self.digest[:12]} ({self.size} B)"
            )
        if self._buf is not None:
            return self._buf[offset : offset + length]
        if self.path is not None:
            if self._fh is None:
                self._fh = open(self.path, "rb")  # noqa: SIM115 — closed by close()
            self._fh.seek(offset)
            data = self._fh.read(length)
            if len(data) != length:
                raise OSError(f"short read from {self.path}")
            return data
        assert self.remote is not None
        return self.remote.get(f"shards/{self.digest[:2]}/{self.digest}", offset, length)

    def read_many(self, ranges: Sequence[tuple[int, int]]) -> list[bytes]:
        """Read several ``(offset, length)`` spans, coalescing near neighbours."""
        order = sorted(range(len(ranges)), key=lambda i: ranges[i][0])
        out: list[bytes] = [b""] * len(ranges)
        group: list[int] = []
        g0 = g1 = 0

        def flush() -> None:
            if not group:
                return
            blob = self.read(g0, g1 - g0)
            for i in group:
                off, n = ranges[i]
                out[i] = blob[off - g0 : off - g0 + n]

        for i in order:
            off, n = ranges[i]
            if group and off - g1 <= _GAP and max(g1, off + n) - g0 <= _MAX_RANGE:
                g1 = max(g1, off + n)
                group.append(i)
                continue
            flush()
            group, g0, g1 = [i], off, off + n
        flush()
        return out

    def index(self) -> tuple[list[str], list[dict[str, list[int]]]]:
        """``(keys, [{field: [offset, length]}])`` from the pack footer."""
        if self._index is None:
            footer = self.read(self.size - _FOOTER, _FOOTER)
            if footer[8:] != _MAGIC:
                raise ValueError(f"shard {self.digest[:12]}: bad pack magic {footer[8:]!r}")
            (n,) = struct.unpack("<Q", footer[:8])
            raw = self.read(self.size - _FOOTER - n, n)
            idx = msgpack.unpackb(raw, raw=False, strict_map_key=False)
            self._index = (list(idx["keys"]), list(idx["samples"]))
        return self._index

    def close(self) -> None:
        self._buf = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class ShardOpener:
    """Resolve a shard to a local copy (size-checked) under ``local_roots`` or to the remote."""

    def __init__(self, local_roots: Iterable[str | Path] = (), remote: RemoteRanges | None = None):
        self.local_roots = [Path(r) for r in local_roots]
        self.remote = remote

    def locate(self, digest: str, size: int) -> Path | None:
        for root in self.local_roots:
            p = root / "shards" / digest[:2] / digest
            try:
                if p.stat().st_size == int(size):
                    return p
            except OSError:
                continue
        return None

    def open(self, digest: str, size: int) -> ShardBytes:
        return ShardBytes(digest, size, path=self.locate(digest, size), remote=self.remote)


class _RangeFile(io.RawIOBase):
    """Seekable read-only file over one field span (for soundfile/zipfile)."""

    _BLOCK = 4 << 20

    def __init__(
        self, shard: ShardBytes, offset: int, length: int, *, block: int | None = None
    ) -> None:
        super().__init__()
        self._shard, self._off, self._len, self._pos = shard, int(offset), int(length), 0
        if block:
            self._BLOCK = int(block)
        self._blocks: dict[int, bytes] = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, pos: int, whence: int = 0) -> int:
        base = {0: 0, 1: self._pos, 2: self._len}[whence]
        self._pos = max(0, base + int(pos))
        return self._pos

    def _block(self, b: int) -> bytes:
        if b not in self._blocks:
            if len(self._blocks) >= 8:
                self._blocks.pop(next(iter(self._blocks)))
            start = b * self._BLOCK
            self._blocks[b] = self._shard.read(
                self._off + start, min(self._BLOCK, self._len - start)
            )
        return self._blocks[b]

    def readinto(self, buf: Any) -> int:
        want = min(len(buf), self._len - self._pos)
        if want <= 0:
            return 0
        if want > 2 * self._BLOCK:  # big sequential read: one direct range, no block cache
            data = self._shard.read(self._off + self._pos, want)
            buf[:want] = data
            self._pos += want
            return want
        got = 0
        while got < want:
            b, inner = divmod(self._pos, self._BLOCK)
            chunk = self._block(b)[inner : inner + want - got]
            buf[got : got + len(chunk)] = chunk
            got += len(chunk)
            self._pos += len(chunk)
        return got


# ─── typed descriptors ─────────────────────────────────────────────────────


def _npy_header(head: bytes) -> tuple[np.dtype, tuple[int, ...], bool, int]:
    """``(dtype, shape, fortran_order, data_offset)`` of an ``.npy`` blob prefix."""
    if head[:6] != b"\x93NUMPY":
        raise ValueError("not an .npy field")
    major = head[6]
    if major == 1:
        (hlen,) = struct.unpack("<H", head[8:10])
        start = 10
    else:
        (hlen,) = struct.unpack("<I", head[8:12])
        start = 12
    if len(head) < start + hlen:
        raise _NeedMore(start + hlen)
    d = ast.literal_eval(head[start : start + hlen].decode("latin1"))
    return (
        np.dtype(d["descr"]),
        tuple(int(s) for s in d["shape"]),
        bool(d["fortran_order"]),
        start + hlen,
    )


class _NeedMore(Exception):
    def __init__(self, n: int) -> None:
        super().__init__(n)
        self.n = n


@dataclass
class SeriesRef:
    """Where and how one array (or audio file) lives inside a shard."""

    path: str
    field: str
    container: str  # "npy" | "soundfile"
    shape: tuple[int, ...]
    dtype: str
    dims: tuple[str | None, ...] = ()
    fortran: bool = False
    offset: int = 0  # absolute shard offset of the field
    length: int = 0
    data_offset: int = 0  # absolute shard offset of the array data (npy only)
    time_axis: int | None = None
    index: str | None = None  # grid | stamps | spans | None
    rate_hz: float | None = None
    t_start_s: float | None = None
    t_end_s: float | None = None
    n_time: int | None = None
    sniffed: bool = False  # audio container recognised by header, not by name
    index_start_ticks: int | None = None  # origin of stored (relative) stamp/span ticks

    @property
    def is_audio(self) -> bool:
        """A waveform: a decoded audio container, or a uniformly sampled PCM-typed
        array of at most (channel, time) at an audio rate whose name is not a
        clock/label/track (sample-clock counters and masks share the audio grid)."""
        if self.container in ("soundfile", "ffmpeg"):
            return True
        if self.index != "grid" or self.rate_hz is None or self.rate_hz < AUDIO_MIN_RATE_HZ:
            return False
        dt = np.dtype(self.dtype)
        if dt.kind not in "fiu" or (dt.kind in "iu" and dt.itemsize > 4) or len(self.shape) > 2:
            return False
        if len(self.shape) == 2 and self.n_channels > _MAX_AUDIO_CHANNELS:
            return False
        leaf = self.path.rsplit("/", 1)[-1].lower()
        return not any(t in leaf for t in _NOT_AUDIO_TOKENS)

    @property
    def n_channels(self) -> int:
        if self.time_axis is None:
            return 0
        n = 1
        for ax, s in enumerate(self.shape):
            if ax != self.time_axis:
                n *= s
        return n

    @property
    def duration_s(self) -> float | None:
        if self.t_start_s is None or self.t_end_s is None:
            return None
        return self.t_end_s - self.t_start_s

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "shape": list(self.shape),
            "dims": list(self.dims),
            "dtype": self.dtype,
            "index": self.index,
            "container": self.container,
        }
        for name in ("rate_hz", "t_start_s", "t_end_s", "n_time"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        if self.sniffed:
            out["sniffed"] = True
        if self.time_axis is not None:
            out["n_channels"] = self.n_channels
            out["duration_s"] = self.duration_s
            out["audio"] = self.is_audio
        return out


@dataclass
class SampleScan:
    """Metadata-only view of one published sample."""

    key: str
    shard: int
    offset: int  # span of the sample's fields inside the shard
    length: int
    fields: dict[str, list[int]]
    meta: dict[str, Any]
    series: dict[str, SeriesRef]
    info: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def audio_refs(self) -> dict[str, SeriesRef]:
        return {p: s for p, s in self.series.items() if s.is_audio}


# ─── layout readers ────────────────────────────────────────────────────────


def _npy_refs(shard: ShardBytes, wanted: Mapping[str, tuple[int, int]]) -> dict[str, tuple]:
    """Parse ``.npy`` headers of several fields with coalesced small reads."""
    names = list(wanted)
    heads = shard.read_many([(wanted[n][0], min(wanted[n][1], 512)) for n in names])
    out: dict[str, tuple] = {}
    for name, head in zip(names, heads, strict=True):
        try:
            out[name] = _npy_header(head)
        except _NeedMore as more:
            out[name] = _npy_header(shard.read(wanted[name][0], more.n))
    return out


def _walk_desc(
    desc: Mapping[str, Any], prefix: str, acc: list[tuple[str, Mapping[str, Any]]]
) -> None:
    for name, entry in desc["entries"].items():
        path = f"{prefix}{name}"
        if entry["kind"] == "series":
            acc.append((path, entry))
        elif entry["kind"] == "frame":
            _walk_desc(entry, f"{path}/", acc)


def _meta_values(desc: Mapping[str, Any], prefix: str, arrays: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, entry in desc["entries"].items():
        path = f"{prefix}{name}"
        kind = entry["kind"]
        if kind == "value":
            out[name] = entry["value"]
        elif kind == "frame":
            out[name] = _meta_values(entry, f"{path}/", arrays)
        else:
            out[name] = arrays.get(path, {"series_dims": entry["dims"], "omitted": "large"})
    return out


def _scan_tdframe(shard: ShardBytes, si: int, key: str, fields: dict[str, list[int]]) -> SampleScan:
    fo, fl = fields["_frame"]
    header = json.loads(shard.read(fo, fl))
    if header.get("format") != "tdframe-v1":
        raise ValueError(f"{key}: unsupported frame format {header.get('format')!r}")
    desc = header["frame"]
    entries: list[tuple[str, Mapping[str, Any]]] = []
    _walk_desc(desc, "", entries)
    present = {p: e for p, e in entries if p in fields}
    heads = _npy_refs(shard, {p: (fields[p][0], fields[p][1]) for p in present})

    # stamp arrays (time span of irregular tracks) and small meta arrays, one batched read
    extra: dict[str, tuple[int, int]] = {}
    for path, entry in present.items():
        for dim, idesc in entry.get("indexes", {}).items():
            sk = f"{path}#{dim}.stamps"
            if idesc["type"] == "stamps" and sk in fields and fields[sk][1] <= _STAMPS_MAX:
                extra[sk] = (fields[sk][0], fields[sk][1])
        if path.startswith("meta/") and fields[path][1] <= _META_SERIES_MAX:
            extra[path] = (fields[path][0], fields[path][1])
    blobs = dict(zip(extra, shard.read_many(list(extra.values())), strict=True))

    series: dict[str, SeriesRef] = {}
    meta_arrays: dict[str, Any] = {}
    for path, entry in present.items():
        dtype, shape, fortran, doff = heads[path]
        off, length = fields[path]
        dims = tuple(None if d is None else str(d) for d in entry["dims"])
        ref = SeriesRef(
            path=path,
            field=path,
            container="npy",
            shape=shape,
            dtype=dtype.str,
            dims=dims,
            fortran=fortran,
            offset=off,
            length=length,
            data_offset=off + doff,
        )
        for dim, idesc in entry.get("indexes", {}).items():
            if idesc["type"] not in ("grid", "stamps", "spans") or dim not in dims:
                continue
            ref.time_axis = dims.index(dim)
            ref.index = idesc["type"]
            ref.n_time = shape[ref.time_axis] if shape else 0
            t0 = int(idesc.get("t_start_ticks", 0))
            if idesc["type"] == "grid":
                ref.rate_hz = int(idesc["sr_num"]) / int(idesc["sr_den"])
                ref.t_start_s = t0 / _TICKS
                ref.t_end_s = ref.t_start_s + ref.n_time / ref.rate_hz
            elif idesc["type"] in ("stamps", "spans"):
                ref.index_start_ticks = t0  # stored stamps/starts are relative to this origin
                sk = f"{path}#{dim}.stamps"
                if sk in blobs:
                    stamps = np.load(io.BytesIO(blobs[sk]), allow_pickle=False)
                    if stamps.size:
                        ref.t_start_s = (t0 + int(stamps[0])) / _TICKS
                        ref.t_end_s = (t0 + int(stamps[-1])) / _TICKS
                        if stamps.size > 1 and stamps[-1] > stamps[0]:
                            ref.rate_hz = (stamps.size - 1) * _TICKS / float(stamps[-1] - stamps[0])
        if path.startswith("meta/"):
            if path in blobs:
                meta_arrays[path] = np.load(io.BytesIO(blobs[path]), allow_pickle=False).tolist()
            continue
        series[path] = ref

    meta_desc = desc["entries"].get("meta")
    meta = (
        _meta_values(meta_desc, "meta/", meta_arrays)
        if meta_desc and meta_desc["kind"] == "frame"
        else {}
    )
    if "recording_id" not in meta:
        meta["recording_id"] = key  # streams.decode_tdframe default
    span = [v for v in fields.values()]
    s0 = min(o for o, _ in span)
    s1 = max(o + n for o, n in span)
    return SampleScan(key, si, s0, s1 - s0, fields, meta, series)


def _soundfile_ref(
    shard: ShardBytes, name: str, off: int, length: int, *, block: int | None = None
) -> SeriesRef:
    import soundfile as sf

    with sf.SoundFile(_RangeFile(shard, off, length, block=block)) as f:
        n, ch, sr = int(f.frames), int(f.channels), float(f.samplerate)
    return SeriesRef(
        path=name,
        field=name,
        container="soundfile",
        shape=(ch, n),
        dtype="<f4",
        dims=("mic", "time"),
        offset=off,
        length=length,
        time_axis=1,
        index="grid",
        rate_hz=sr,
        t_start_s=0.0,
        t_end_s=n / sr,
        n_time=n,
    )


def _xml_container_ref(shard: ShardBytes, key: str, name: str, off: int, length: int) -> SeriesRef:
    """A base64-WAV measurement XML (AuDroK MDM) as a mono float32 Series, decoded with
    the source builder's own parser (present samples only, never padded)."""
    from data_processing.sources.audrok import decode_container

    pcm, _header, layout = decode_container(shard.read(off, length), name=key)
    sr = float(layout["sample_rate"])
    n = int(pcm.shape[-1])
    return SeriesRef(
        path=name,
        field=name,
        container="mdm-xml",
        shape=(1, n),
        dtype="<f4",
        dims=("mic", "time"),
        offset=off,
        length=length,
        time_axis=1,
        index="grid",
        rate_hz=sr,
        t_start_s=0.0,
        t_end_s=n / sr,
        n_time=n,
    )


def _field_ext(key: str, name: str, fields_map: Mapping[str, str]) -> str:
    """File extension of a field: the manifest map (sample-dir-v1), else the dload
    CLI convention (field = extension; ``data``/``""`` = the key's own suffix)."""
    mapped = str(fields_map.get(name, name))
    if name not in fields_map and name in _EXTENSIONLESS:
        mapped = key.rsplit("/", 1)[-1]
        if "." not in mapped:
            return ""
    return mapped.rsplit(".", 1)[-1].lower() if "." in mapped else mapped.lower()


def _scan_files(
    shard: ShardBytes,
    si: int,
    key: str,
    fields: dict[str, list[int]],
    fields_map: Mapping[str, str],
) -> SampleScan:
    """``sample-dir-v1`` / raw CLI / auraflow JSON+wav samples: files per field.

    Audio fields follow ``storage.decode_sample``: by extension, else (no named
    audio field) any other field whose header libsndfile parses (sniffed)."""
    series: dict[str, SeriesRef] = {}
    meta: dict[str, Any] = {}
    json_fields: dict[str, tuple[int, int]] = {}
    text_fields: dict[str, tuple[int, int]] = {}
    npy_fields: dict[str, tuple[int, int]] = {}
    other: list[str] = []
    errors: dict[str, str] = {}
    for name, (off, length) in fields.items():
        ext = _field_ext(key, name, fields_map)
        if ext in _SOUNDFILE_EXT:
            try:
                series[name] = _soundfile_ref(shard, name, off, length)
            except Exception as exc:  # noqa: BLE001 — undecodable container is reported, not fatal
                errors[name] = f"{type(exc).__name__}: {exc}"
        elif ext == "json" and length <= _META_SERIES_MAX:
            json_fields[name] = (off, length)
        elif ext == "npy":
            npy_fields[name] = (off, length)
        elif name in ("meta", "metadata") and length <= _META_SERIES_MAX:
            text_fields[name] = (off, length)
        elif ext in _VIDEO_EXT:
            try:  # e.g. AVQ GoPro mp4: its first audio stream (the file itself is never re-uploaded)
                series[name] = _ffmpeg_ref(shard, key, name, off, length, ext)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                errors[name] = f"no decodable audio stream: {exc}"
        elif ext == "xml" and length <= _XML_CONTAINER_MAX:
            try:  # AuDroK MDM container: base64 WAV body (a descriptor XML has none)
                series[name] = _xml_container_ref(shard, key, name, off, length)
            except ValueError as exc:
                errors[name] = f"not a measurement container: {exc}"
        else:
            other.append(name)
    if not series:  # no named audio field: sniff the rest by header, as the decoder does
        for name in other:
            off, length = fields[name]
            try:
                ref = _soundfile_ref(shard, name, off, length, block=_SNIFF_BLOCK)
            except Exception:  # noqa: BLE001, S112 — not a sound container
                continue
            ref.sniffed = True
            series[name] = ref
    for name, blob in zip(json_fields, shard.read_many(list(json_fields.values())), strict=True):
        try:
            value = json.loads(blob)
        except ValueError:
            continue
        if name == "meta" and isinstance(value, dict):
            meta.update(value)
        else:
            meta.setdefault("json_fields", {})[name] = value
    for name, blob in zip(text_fields, shard.read_many(list(text_fields.values())), strict=True):
        try:  # extensionless ``meta`` of the auraflow sims is JSON; sample-dir txt stays text
            value = json.loads(blob)
        except ValueError:
            value = None
        if isinstance(value, dict):
            meta.update(value)
        else:
            meta[f"{name}_text"] = blob.decode("utf-8", errors="replace")
    for name, (dtype, shape, fortran, doff) in _npy_refs(shard, npy_fields).items():
        off, length = npy_fields[name]
        series[name] = SeriesRef(
            path=name,
            field=name,
            container="npy",
            shape=shape,
            dtype=dtype.str,
            fortran=fortran,
            offset=off,
            length=length,
            data_offset=off + doff,
        )
    span = list(fields.values())
    s0 = min(o for o, _ in span) if span else 0
    s1 = max(o + n for o, n in span) if span else 0
    return SampleScan(key, si, s0, s1 - s0, fields, meta, series, errors=errors)


def _info(
    scan: SampleScan,
    *,
    dataset: str,
    version: str,
    layout: str,
    digest: str,
    dataset_meta: Mapping[str, Any] | None,
) -> dict[str, Any]:
    audio = scan.audio_refs()
    primary = audio.get("audio") or (
        max(audio.values(), key=lambda r: (r.duration_s or 0.0, r.n_channels)) if audio else None
    )
    info: dict[str, Any] = {
        "fields": sorted(scan.fields),
        "series": {p: r.summary() for p, r in sorted(scan.series.items())},
        "has_audio": bool(audio),
        "audio_entries": sorted(audio),
        "duration_s": primary.duration_s if primary else None,
        "sample_rate": primary.rate_hz if primary else None,
        "version": version,
        "layout": layout,
        "shard": scan.shard,
        "shard_digest": digest,
    }
    if dataset_meta is not None:
        info["dataset_meta"] = dataset_meta
    return info


def scan_sample(
    shard: ShardBytes,
    shard_index: int,
    key: str,
    fields: Mapping[str, Sequence[int]],
    *,
    dataset: str,
    version: str,
    layout: str,
    dataset_meta: Mapping[str, Any] | None = None,
    fields_map: Mapping[str, str] | None = None,
) -> SampleScan:
    """Metadata-only scan of one sample from its pack-footer field spans.

    ``fields_map`` is the manifest ``meta["fields"]`` stem -> filename map of
    file-per-field layouts (``sample-dir-v1``: ``mixture`` -> ``mixture.wav``);
    defaults to ``dataset_meta["fields"]``. ``dataset_meta`` (if given) is
    attached to ``info`` for the classifier.
    """
    fmap: Mapping[str, str]
    if fields_map is not None:
        fmap = fields_map
    else:
        meta = dataset_meta if isinstance(dataset_meta, Mapping) else {}
        fmap = meta.get("fields", {}) if isinstance(meta.get("fields"), Mapping) else {}
    spans = {str(k): [int(v[0]), int(v[1])] for k, v in fields.items()}
    if "_frame" in spans:
        scan = _scan_tdframe(shard, shard_index, key, spans)
        kind = "tdframe-v1"
    else:
        scan = _scan_files(shard, shard_index, key, spans, fmap)
        kind = layout or "raw-files"
    scan.info = _info(
        scan,
        dataset=dataset,
        version=version,
        layout=kind,
        digest=shard.digest,
        dataset_meta=dataset_meta,
    )
    if scan.errors:
        scan.info["scan_errors"] = dict(scan.errors)
    return scan


def scan_shard(
    shard: ShardBytes,
    shard_index: int,
    *,
    dataset: str,
    version: str,
    layout: str,
    dataset_meta: Mapping[str, Any] | None = None,
    fields_map: Mapping[str, str] | None = None,
    keys: Iterable[str] | None = None,
) -> Iterator[SampleScan]:
    """Metadata-only scans of the shard's data samples (keys not starting with ``_``),
    in pack order; ``keys`` restricts to those keys (missing ones raise ``KeyError``)."""
    all_keys, samples = shard.index()
    pos = {k: i for i, k in enumerate(all_keys)}
    order = range(len(all_keys)) if keys is None else [pos[k] for k in keys]
    for i in order:
        key = all_keys[i]
        if key.startswith("_"):
            continue
        yield scan_sample(
            shard,
            shard_index,
            key,
            samples[i],
            dataset=dataset,
            version=version,
            layout=layout,
            dataset_meta=dataset_meta,
            fields_map=fields_map,
        )


# ─── audio views ───────────────────────────────────────────────────────────


def _to_float(a: np.ndarray) -> np.ndarray:
    kind = a.dtype.kind
    if kind == "f":
        return a.astype(np.float32, copy=False)
    if kind == "i":
        return (a.astype(np.float32) / float(2 ** (8 * a.dtype.itemsize - 1))).astype(np.float32)
    if kind == "u":
        half = float(2 ** (8 * a.dtype.itemsize - 1))
        return ((a.astype(np.float32) - half) / half).astype(np.float32)
    raise TypeError(f"not PCM: dtype {a.dtype}")


def read_view(
    shard: ShardBytes,
    ref: SeriesRef,
    t0_s: float,
    t1_s: float,
    channels: Sequence[int] | None = None,
) -> tuple[np.ndarray, float]:
    """``(float32 (C, T) PCM, actual view start on the Frame clock)`` of ``[t0, t1)``.

    Integer PCM is scaled to [-1, 1). Only the requested byte ranges are read.
    """
    if ref.rate_hz is None or ref.t_start_s is None or ref.time_axis is None:
        raise ValueError(f"{ref.path}: not a uniformly sampled Series")
    rate = ref.rate_hz
    i0 = max(0, int(round((t0_s - ref.t_start_s) * rate)))
    i1 = min(int(ref.n_time or 0), int(round((t1_s - ref.t_start_s) * rate)))
    n = max(0, i1 - i0)
    t_view = ref.t_start_s + i0 / rate
    nch = ref.n_channels
    chans = list(range(nch)) if channels is None else [int(c) for c in channels]
    if any(c < 0 or c >= nch for c in chans):
        raise IndexError(f"{ref.path}: channels {chans} outside 0..{nch - 1}")
    if n == 0:
        return np.zeros((len(chans), 0), np.float32), t_view

    if ref.container == "ffmpeg":  # audio track of a video container: decode just [t0, t1)
        return _ffmpeg_window(shard, ref, i0, n, chans), t_view
    if ref.container == "mdm-xml":  # base64 body: decode the (few-MB) container, then slice
        from data_processing.sources.audrok import decode_container

        pcm, _h, _l = decode_container(shard.read(ref.offset, ref.length), name=ref.path)
        return np.ascontiguousarray(pcm[None, i0:i1], dtype=np.float32), t_view
    if ref.container == "soundfile":
        import soundfile as sf

        with sf.SoundFile(_RangeFile(shard, ref.offset, ref.length)) as f:
            f.seek(i0)
            data = f.read(n, dtype="float32", always_2d=True)  # (T, C)
        return np.ascontiguousarray(data[:, chans].T), t_view

    dtype = np.dtype(ref.dtype)
    isz = dtype.itemsize
    if len(ref.shape) == 1:
        raw = shard.read(ref.data_offset + i0 * isz, n * isz)
        return _to_float(np.frombuffer(raw, dtype))[None, :].copy(), t_view
    if len(ref.shape) != 2:
        raise ValueError(f"{ref.path}: audio with {len(ref.shape)} dims is not supported")
    tax = ref.time_axis
    n_time = ref.shape[tax]
    # Memory order: time is the fastest axis iff (C-order and time last) or (Fortran and time first).
    time_fastest = (tax == 1) != ref.fortran
    if time_fastest:
        ranges = [(ref.data_offset + (c * n_time + i0) * isz, n * isz) for c in chans]
        rows = shard.read_many(ranges)
        out = np.empty((len(chans), n), np.float32)
        for j, raw in enumerate(rows):
            out[j] = _to_float(np.frombuffer(raw, dtype))
        return out, t_view
    raw = shard.read(ref.data_offset + i0 * nch * isz, n * nch * isz)
    block = np.frombuffer(raw, dtype).reshape(n, nch)
    return np.ascontiguousarray(_to_float(block[:, chans]).T), t_view


def finite_runs(x: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """``[start, stop)`` sample runs where every channel is finite, at least ``min_len`` long."""
    ok = np.isfinite(x).all(axis=0) if x.ndim == 2 else np.isfinite(x)
    if ok.all():
        return [(0, ok.size)] if ok.size >= min_len else []
    edges = np.flatnonzero(np.diff(np.concatenate(([0], ok.view(np.int8), [0]))))
    return [
        (int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True) if b - a >= min_len
    ]
