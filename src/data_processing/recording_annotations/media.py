"""Byte-range access to the first AAC track in a non-fragmented MP4.

The reader asks for file-relative box headers and the bounded ``moov`` box;
callers choose which audio packet runs to fetch from the media payload. Neither
this module nor its ADTS framing helper reads or decodes video.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

# ISO/IEC 14496-3, Table 1.18: the sampling-frequency indexes used by ADTS.
AAC_RATES = (
    96000,
    88200,
    64000,
    48000,
    44100,
    32000,
    24000,
    22050,
    16000,
    12000,
    11025,
    8000,
    7350,
)
ADTS_HEADER_BYTES = 7
# A track with 2M AAC-LC frames covers more than five hours even at 96 kHz.
MAX_FRAMES = 2_000_000


@dataclass(frozen=True, slots=True)
class Mp4Audio:
    """AAC codec configuration, file-relative packet ranges and media clock."""

    aot: int
    sr_index: int
    channels: int
    sr: int
    offsets: np.ndarray  # (frames,) int64, byte offset from file start
    sizes: np.ndarray  # (frames,) int64, packet bytes
    starts: np.ndarray  # (frames + 1,) int64 media samples, including the end
    skip: int  # leading media samples omitted by the edit (encoder priming)
    delay_s: float  # presentation time of the first kept sample

    @property
    def total(self) -> int:
        return int(self.starts[-1]) - self.skip


def _boxes(buf: bytes, pos: int, end: int) -> Iterator[tuple[bytes, int, int]]:
    """Type and half-open body bounds of the boxes in ``buf[pos:end]``."""
    while pos + 8 <= end:
        size, kind = struct.unpack_from(">I4s", buf, pos)
        head = 8
        if size == 1:
            if pos + 16 > end:
                raise ValueError(f"truncated MP4 box header at byte {pos}")
            size, head = struct.unpack_from(">Q", buf, pos + 8)[0], 16
        elif size == 0:
            size = end - pos
        if size < head or pos + size > end:
            raise ValueError(f"MP4 box {kind!r} at byte {pos} overruns its parent")
        yield kind, pos + head, pos + size
        pos += size


def _child(buf: bytes, pos: int, end: int, *path: bytes) -> tuple[int, int] | None:
    for kind in path:
        hit = next(((a, b) for k, a, b in _boxes(buf, pos, end) if k == kind), None)
        if hit is None:
            return None
        pos, end = hit
    return pos, end


def _timescale(buf: bytes, pos: int, end: int) -> int:
    if pos >= end or buf[pos] not in (0, 1):
        raise ValueError("invalid MP4 timescale box")
    pos += 20 if buf[pos] == 1 else 12
    if pos + 4 > end:
        raise ValueError("truncated MP4 timescale box")
    return struct.unpack_from(">I", buf, pos)[0]


def _descriptor(buf: bytes, pos: int, end: int) -> tuple[int, int, int]:
    if pos < end:
        tag, pos, n = buf[pos], pos + 1, 0
        for _ in range(4):
            if pos >= end:
                break
            b = buf[pos]
            pos += 1
            n = (n << 7) | (b & 0x7F)
            if not b & 0x80 and pos + n <= end:
                return tag, pos, pos + n
    raise ValueError("invalid MPEG-4 audio descriptor")


def _audio_config(buf: bytes, stbl: tuple[int, int]) -> tuple[int, int, int]:
    stsd = _child(buf, *stbl, b"stsd")
    entry = next(_boxes(buf, stsd[0] + 8, stsd[1]), None) if stsd else None
    if entry is None or entry[0] != b"mp4a":
        raise ValueError("first MP4 sound track is not AAC (mp4a)")
    esds = _child(buf, entry[1] + 28, entry[2], b"esds")
    if esds is None:
        raise ValueError("mp4a entry without esds")
    tag, pos, end = _descriptor(buf, esds[0] + 4, esds[1])
    if tag != 3 or pos + 3 > end:
        raise ValueError("mp4a entry without ES descriptor")
    flags = buf[pos + 2]
    pos += 3 + (2 if flags & 0x80 else 0)
    if flags & 0x40 and pos < end:
        pos += 1 + buf[pos]
    if flags & 0x20:
        pos += 2
    tag, pos, end = _descriptor(buf, pos, end)
    if tag != 4:
        raise ValueError("mp4a entry without DecoderConfigDescriptor")
    tag, pos, end = _descriptor(buf, pos + 13, end)
    if tag != 5 or pos + 2 > end:
        raise ValueError("no AAC AudioSpecificConfig in esds")
    bits = struct.unpack_from(">H", buf, pos)[0]
    aot, rate, channels = bits >> 11, (bits >> 7) & 0xF, (bits >> 3) & 0xF
    if not 1 <= aot <= 4 or rate >= len(AAC_RATES) or channels not in (1, 2):
        raise ValueError(
            f"unsupported AAC configuration (type {aot}, rate index {rate}, channels {channels})"
        )
    return aot, rate, channels


def _edit(buf: bytes, root: tuple[int, int], trak: tuple[int, int]) -> tuple[int, float]:
    box = _child(buf, *trak, b"edts", b"elst")
    if box is None:
        return 0, 0.0
    pos, end = box
    if pos + 8 > end or buf[pos] not in (0, 1):
        raise ValueError("invalid MP4 edit list")
    fmt, step = (">Qqhh", 20) if buf[pos] == 1 else (">Iihh", 12)
    count = struct.unpack_from(">I", buf, pos + 4)[0]
    if pos + 8 + count * step > end:
        raise ValueError("truncated MP4 edit list")
    delay = 0
    for i in range(count):
        duration, media_time, rate, frac = struct.unpack_from(fmt, buf, pos + 8 + i * step)
        if media_time == -1:
            delay += duration
            continue
        if media_time < 0 or i != count - 1 or (rate, frac) != (1, 0):
            raise ValueError("only a single normal-speed MP4 media edit is supported")
        if not delay:
            return media_time, 0.0
        mvhd = _child(buf, *root, b"mvhd")
        ts = _timescale(buf, *mvhd) if mvhd else 0
        if not ts:
            raise ValueError("empty edit in the MP4 edit list without movie timescale")
        return media_time, delay / ts
    raise ValueError("MP4 edit list has no presented samples")


def _table(buf: bytes, stbl: tuple[int, int], name: bytes, fmt: str, width: int) -> np.ndarray:
    box = _child(buf, *stbl, name)
    if box is None:
        return np.empty(0, dtype=np.int64)
    pos, end = box
    if pos + 8 > end:
        raise ValueError(f"truncated MP4 sample table {name!r}")
    count = struct.unpack_from(">I", buf, pos + 4)[0]
    if pos + 8 + count * width > end:
        raise ValueError(f"truncated MP4 sample table {name!r}")
    return np.frombuffer(
        buf, dtype=fmt, offset=pos + 8, count=count * width // np.dtype(fmt).itemsize
    ).astype(np.int64)


def _packet_tables(
    buf: bytes, stbl: tuple[int, int], length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stts = _table(buf, stbl, b"stts", ">u4", 8).reshape(-1, 2)
    stsc = _table(buf, stbl, b"stsc", ">u4", 12).reshape(-1, 3)
    stsz = _child(buf, *stbl, b"stsz")
    if stsz is None or stsz[0] + 12 > stsz[1]:
        raise ValueError("missing or truncated stsz audio sample table")
    fixed, n_frames = struct.unpack_from(">II", buf, stsz[0] + 4)
    if not 0 < n_frames <= MAX_FRAMES or stts[:, 0].sum() != n_frames:
        raise ValueError("invalid or unsupported MP4 audio frame count")
    if fixed:
        sizes = np.full(n_frames, fixed, dtype=np.int64)
    else:
        if stsz[0] + 12 + 4 * n_frames > stsz[1]:
            raise ValueError("truncated stsz audio sample table")
        sizes = np.frombuffer(buf, dtype=">u4", offset=stsz[0] + 12, count=n_frames).astype(
            np.int64
        )
    if sizes.min() <= 0 or sizes.max() + ADTS_HEADER_BYTES >= 1 << 13:
        raise ValueError("AAC frame sizes are not representable in ADTS")
    chunks = _table(buf, stbl, b"stco", ">u4", 4)
    if not len(chunks):
        chunks = _table(buf, stbl, b"co64", ">u8", 8)
    if (
        not len(stsc)
        or stsc[0, 0] != 1
        or stsc[-1, 0] > len(chunks)
        or len(chunks) > n_frames
        or np.any(np.diff(stsc[:, 0]) <= 0)
    ):
        raise ValueError("invalid stsc audio sample table")
    runs = np.diff(stsc[:, 0], append=len(chunks) + 1)
    if np.any(stsc[:, 2] != 1) or np.any(stsc[:, 1] < 1) or runs @ stsc[:, 1] != n_frames:
        raise ValueError("inconsistent MP4 sample tables (stsc/stsz)")
    per_chunk = np.repeat(stsc[:, 1], runs)
    chunk_of = np.repeat(np.arange(len(chunks)), per_chunk)
    before = np.cumsum(sizes) - sizes
    chunk_first = np.cumsum(per_chunk) - per_chunk
    offsets = chunks[chunk_of] + before - before[chunk_first[chunk_of]]
    if (
        offsets[0] < 0
        or np.any(offsets[1:] < offsets[:-1] + sizes[:-1])
        or offsets[-1] > length - sizes[-1]
    ):
        raise ValueError("overlapping or out-of-bounds MP4 audio frames")
    durations = np.repeat(stts[:, 1], stts[:, 0])
    if np.any(durations <= 0):
        raise ValueError("non-positive MP4 audio frame duration")
    starts = np.empty(n_frames + 1, dtype=np.int64)
    starts[0] = 0
    np.cumsum(durations, out=starts[1:])
    return offsets, sizes, starts


def parse_mp4_audio(
    read: Callable[[int, int], bytes], length: int, *, max_metadata_bytes: int = 8 << 20
) -> Mp4Audio:
    """Read the non-fragmented MP4's first AAC track tables, not its media payload.

    ``read(start, end)`` takes file-relative half-open byte bounds. Unsupported
    MP4 formats raise ``ValueError``; failed and short reads raise ``OSError``.
    """

    def exact(start: int, end: int) -> bytes:
        data = read(start, end)
        if len(data) != end - start:
            raise OSError(f"short MP4 read at byte {start}: {len(data)} of {end - start} bytes")
        return data

    pos = 0
    while pos + 8 <= length:
        size, kind = struct.unpack(">I4s", exact(pos, pos + 8))
        header = 8
        if size == 1:
            if pos + 16 > length:
                raise ValueError("truncated extended MP4 box header")
            size, header = struct.unpack(">Q", exact(pos + 8, pos + 16))[0], 16
        elif size == 0:
            size = length - pos
        if size < header or pos + size > length:
            raise ValueError(f"malformed MP4 box {kind!r} at byte {pos}")
        if kind == b"moov":
            if size > max_metadata_bytes:
                raise ValueError("MP4 metadata exceeds the read limit")
            moov = exact(pos + header, pos + size)
            break
        pos += size
    else:
        raise ValueError("no moov box (not a seekable MP4)")
    root = (0, len(moov))
    if _child(moov, *root, b"mvex") is not None:
        raise ValueError("fragmented MP4 audio is not supported")
    for kind, a, b in _boxes(moov, *root):
        if kind != b"trak":
            continue
        hdlr = _child(moov, a, b, b"mdia", b"hdlr")
        if hdlr is None or hdlr[0] + 12 > hdlr[1] or moov[hdlr[0] + 8 : hdlr[0] + 12] != b"soun":
            continue
        mdhd = _child(moov, a, b, b"mdia", b"mdhd")
        stbl = _child(moov, a, b, b"mdia", b"minf", b"stbl")
        if mdhd is None or stbl is None:
            raise ValueError("MP4 sound track without mdhd/stbl")
        sr = _timescale(moov, *mdhd)
        aot, rate, channels = _audio_config(moov, stbl)
        if sr != AAC_RATES[rate]:
            raise ValueError(
                f"MP4 track timescale {sr} differs from the AAC rate {AAC_RATES[rate]}"
            )
        skip, delay = _edit(moov, root, (a, b))
        offsets, sizes, starts = _packet_tables(moov, stbl, length)
        if skip >= starts[-1]:
            raise ValueError("MP4 edit skips the whole audio track")
        return Mp4Audio(aot, rate, channels, sr, offsets, sizes, starts, skip, delay)
    raise ValueError("no MP4 sound track")


def aac_runs(track: Mp4Audio, first: int, last: int) -> Iterator[tuple[int, int]]:
    """Group consecutive audio frames into contiguous half-open packet runs."""
    for j in range(first + 1, last):
        if track.offsets[j] != track.offsets[j - 1] + track.sizes[j - 1]:
            yield first, j
            first = j
    if first < last:
        yield first, last


def append_adts(out: bytearray, track: Mp4Audio, first: int, last: int, payload: bytes) -> None:
    """Append a contiguous packet run (from ``track.offsets[first]``) as ADTS."""
    data = memoryview(payload)
    pos = 0
    for size in track.sizes[first:last]:
        n = int(size) + ADTS_HEADER_BYTES
        out.extend(
            (
                0xFF,
                0xF1,
                ((track.aot - 1) << 6) | (track.sr_index << 2) | (track.channels >> 2),
                ((track.channels & 3) << 6) | (n >> 11),
                (n >> 3) & 0xFF,
                ((n & 7) << 5) | 0x1F,
                0xFC,
            )
        )
        end = pos + int(size)
        out.extend(data[pos:end])
        pos = end
    if pos != len(data):
        raise ValueError(f"ADTS run expected {pos} bytes, received {len(data)}")
