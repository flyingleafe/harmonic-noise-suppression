"""DJI iOSD Mark II CAN-bus logs from the Chirocopter deposit.

The observed container starts with LE u32 words (4, 1, 5) and 116 zero
bytes. Its body normally has blocks of 1024 12-byte CAN records followed by
``55 55 55 aa aa aa 66 66 66 cc cc cc``. Each record is
``<u16 (can_id << 4 | dlc), u16 sequence, u8 data[8]>``. The first 256
records are a sparse prelude (sequence jumps and incomplete messages), not
part of the continuous capture. Only the first DLC bytes belong to the CAN
stream. The record counter wraps at 65536; it counts records, not time.

Each CAN ID has a separate byte stream of messages. The observed message
wrapper is ``55 | length u16 | crc8 | type u16 | payload | crc16`` with
little-endian values and total message length. Both CRCs cover the original
wire bytes, with DJI initial values 0x77 and 0x3692 respectively.

The 0x1002 and 0x1003 layouts and explicit OSD XOR mask (payload byte
117) are documented in Pawelsky's A2 CAN decoder, 20171104 beta
(https://www.rcgroups.com/forums/showpost.php?p=30774154&postcount=549).
The named field offsets and scales are corroborated by the Chirocopter
files. Date/time are the original decimal YYYYMMDD and HHMMSS words, not
the NAZA packed calendar. This reader does not assign a timezone, a UTC
epoch, an audio offset or a sample rate.

When an archived block is short, long or off the 12-byte grid, resume
from its next exact marker. Off-grid spans retain only the consecutive
record runs aligned to the surrounding markers. Payload CRCs are checked
even in recovered spans. CAN record indices count the retained slots from
the first record in the file (including the sparse prelude, excluding block
markers and unframed bytes).

OSD rows also have an unwrapped OSD packet ordinal, their native sample
clock. Duplicate/backward OSD counters are retained separately without a
sample coordinate. GPS and controller packets have no independent
subsecond timestamp. Undocumented controller bytes are not RPM or PWM.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import IO

import numpy as np
from numba import njit

_MAGIC = bytes.fromhex("040000000100000005000000") + bytes(116)
_SYNC = bytes.fromhex("555555aaaaaa666666cccccc")
_RECORD_BYTES = 12
_BLOCK_BYTES = 1024 * _RECORD_BYTES
_PRELUDE_RECORDS = 256
_MAX_MESSAGE_BYTES = 1023


def _field_names(dtype: np.dtype) -> tuple[str, ...]:
    """Field names of one of the structured layouts below."""
    names = dtype.names
    if names is None:
        raise TypeError(f"{dtype} is not a structured dtype")
    return names


# Confirmed payload layouts, including unnamed bytes between named fields.
_OSD_PAYLOAD = np.dtype(
    {
        "names": [
            "lon_rad",
            "lat_rad",
            "gps_height_m",
            "accel_g",
            "gyro_rad_s",
            "baro_height_m",
            "quaternion",
            "velocity_ned_m_s",
            "magnetometer_raw",
            "satellites",
            "seq",
        ],
        "formats": [
            "<f8",
            "<f8",
            "<f4",
            ("<f4", 3),
            ("<f4", 3),
            "<f4",
            ("<f4", 4),
            ("<f4", 3),
            ("<i2", 3),
            "u1",
            "<u2",
        ],
        "offsets": [0, 8, 16, 20, 32, 44, 48, 76, 100, 116, 118],
        "itemsize": 120,
    }
)
_GPS_PAYLOAD = np.dtype(
    {
        "names": [
            "date",
            "time",
            "lon_e7",
            "lat_e7",
            "height_mm",
            "velocity_ned_cm_s",
            "satellites",
            "seq",
        ],
        "formats": ["<u4", "<u4", "<i4", "<i4", "<i4", ("<f4", 3), "u1", "<u2"],
        "offsets": [0, 4, 8, 12, 16, 20, 64, 66],
        "itemsize": 68,
    }
)
# 0x4a44/0xa249 after the subtype. The six 94000-at-rest words start at
# byte 76 in these recordings, two bytes earlier than the beta reference's
# motor field. They vary independently in flight; sustained commands and
# their transitions coincide with rotor-band audio. Keep controller channel
# order (no physical rotor permutation or output-to-RPM mapping is known).
_CONTROL_PAYLOAD = np.dtype(
    {
        "names": ["seq", "motor_output_code", "voltage_mv"],
        "formats": ["<u4", ("<u4", 6), ("<u2", 3)],
        "offsets": [0, 76, 165],
        "itemsize": 171,
    }
)
_CONTROL_TRACK = np.dtype(
    [("record_index", "<u4"), ("payload", "u1", _CONTROL_PAYLOAD.itemsize)]
    + [(name, _CONTROL_PAYLOAD[name]) for name in _field_names(_CONTROL_PAYLOAD)]
)
_OSD_TRACK = np.dtype(
    [("record_index", "<u4"), ("sample", "<u4")]
    + [(name, _OSD_PAYLOAD[name]) for name in _field_names(_OSD_PAYLOAD)]
)
_GPS_TRACK = np.dtype(
    [("record_index", "<u4")] + [(name, _GPS_PAYLOAD[name]) for name in _field_names(_GPS_PAYLOAD)]
)


@dataclass(slots=True)
class IosdTelemetry:
    """Typed observations on the native logger sequence; not audio-aligned."""

    osd: np.ndarray
    gps: np.ndarray
    controller: np.ndarray
    osd_unplaced: np.ndarray
    irregular_blocks: int
    unframed_bytes: int


def _crc_table(poly: int, dtype: str) -> np.ndarray:
    out = np.empty(256, dtype=dtype)
    for i in range(256):
        value = i
        for _ in range(8):
            value = (value >> 1) ^ (poly if value & 1 else 0)
        out[i] = value
    return out


_CRC8 = _crc_table(0x8C, "u1")
_CRC16 = _crc_table(0x8408, "<u2")


@njit(cache=True)
def _scan_messages(data: np.ndarray, kind: int) -> tuple[np.ndarray, int]:
    """Locate complete, CRC-checked packets of one selected schema."""
    width = (128, 76, 181)[kind]
    message = (0x1002, 0x1003, 0x4A44)[kind]
    offsets = np.empty(data.size // width, dtype=np.int32)
    count = 0
    i = 0
    while i + 4 <= data.size:
        if data[i] != 0x55:
            i += 1
            continue
        crc8 = 0x77
        for j in range(i, i + 3):
            crc8 = _CRC8[crc8 ^ data[j]]
        length = int(data[i + 1]) | (int(data[i + 2]) << 8)
        if crc8 != data[i + 3] or length < 8 or length > _MAX_MESSAGE_BYTES:
            i += 1
            continue
        if i + length > data.size:
            break
        crc16 = 0x3692
        for j in range(i, i + length - 2):
            crc16 = (crc16 >> 8) ^ _CRC16[(crc16 ^ data[j]) & 0xFF]
        expected = int(data[i + length - 2]) | (int(data[i + length - 1]) << 8)
        if crc16 != expected:
            i += 1
            continue
        tag = int(data[i + 4]) | (int(data[i + 5]) << 8)
        if (
            length == width
            and tag == message
            and (kind != 2 or (data[i + 6] == 0x49 and data[i + 7] == 0xA2))
        ):
            offsets[count] = i
            count += 1
        i += length
    return offsets[:count], i


def _track(payload: np.ndarray, schema: np.dtype, dtype: np.dtype, idx: np.ndarray) -> np.ndarray:
    decoded = payload.view(schema).ravel()
    out = np.empty(len(decoded), dtype=dtype)
    out["record_index"] = idx
    for name in _field_names(schema):
        out[name] = decoded[name]
    return out


def _record_run(raw: memoryview, *, reverse: bool = False) -> int:
    words = np.frombuffer(raw, dtype="<u2").reshape(-1, 6)
    if reverse:
        words = words[::-1]
    if not len(words):
        return 0
    hdr, seq = words[:, 0], words[:, 1]
    valid = (hdr <= 0x7FF8) & ((hdr & 0xF) <= 8) & np.any(words != 0, axis=1)
    valid[1:] &= np.diff(seq) == (65535 if reverse else 1)
    invalid = np.flatnonzero(~valid)
    return int(invalid[0]) if len(invalid) else len(words)


def _record_batches(stream: IO[bytes], quality: list[int]) -> Iterator[np.ndarray]:
    """Remove exact sync words and preserve only verifiable off-grid records."""
    carry = b""
    while data := stream.read(256 * (_BLOCK_BYTES + len(_SYNC))):
        data = carry + data if carry else data
        raw = memoryview(data)
        clean = bytearray()
        start = 0
        while (end := data.find(_SYNC, start)) >= 0:
            width = end - start
            if width != _BLOCK_BYTES:
                quality[0] += 1
            remainder = width % _RECORD_BYTES
            if remainder:
                # The two known boundaries imply two grids. Only retain the
                # contiguous CAN-record runs anchored to those boundaries.
                left = _record_run(raw[start : end - remainder]) * _RECORD_BYTES
                right = _record_run(raw[start + remainder : end], reverse=True) * _RECORD_BYTES
                right = min(right, (width - left) // _RECORD_BYTES * _RECORD_BYTES)
                clean.extend(raw[start : start + left])
                clean.extend(raw[end - right : end])
                quality[1] += width - left - right
            else:
                clean.extend(raw[start:end])
            start = end + len(_SYNC)
        carry = data[start:]
        if len(carry) > 2 * _BLOCK_BYTES:
            if carry.count(0) != len(carry):
                raise ValueError("unframed data past the observed iOSD block bounds")
            carry = b""
        if clean:
            yield np.frombuffer(clean, dtype=np.uint8).reshape(-1, _RECORD_BYTES)
    # No closing marker at EOF. Trust only the continuous, unpadded prefix.
    tail = memoryview(carry)
    full = len(tail) // _RECORD_BYTES * _RECORD_BYTES
    keep = _record_run(tail[:full]) * _RECORD_BYTES
    quality[1] += len(tail[keep:]) - carry.count(0, keep)
    if keep:
        yield np.frombuffer(tail[:keep], dtype=np.uint8).reshape(-1, _RECORD_BYTES)


@njit(cache=True)
def _sample_ordinals(
    seq: np.ndarray, previous: int, start: int
) -> tuple[np.ndarray, np.ndarray, int, int]:
    keep = np.empty(seq.size, dtype=np.bool_)
    samples = np.empty(seq.size, dtype=np.uint32)
    for i in range(seq.size):
        value = int(seq[i])
        step = (value - previous) & 65535
        if previous < 0 or 0 < step <= 32767:
            if previous >= 0:
                start += step
            previous = value
            keep[i] = True
            samples[i] = start
        else:
            keep[i] = False
    return keep, samples, previous, start


def read_iosd_dat(stream: IO[bytes]) -> IosdTelemetry:
    """Read a Chirocopter ZIP member opened in binary mode, without extraction.

    Fixed-size working batches in addition to the returned track arrays. A
    final all-zero file allocation tail or incomplete message is ignored.
    The original member in the DOI-pinned archive remains the source for
    power-on or undocumented packets not represented by these tracks.
    """
    if stream.read(len(_MAGIC)) != _MAGIC:
        raise ValueError("not the Chirocopter iOSD CAN log header")
    result: list[list[np.ndarray]] = [[], [], []]
    tails: list[np.ndarray] = [np.empty(0, dtype=np.uint8) for _ in result]
    tail_indices: list[np.ndarray] = [np.empty(0, dtype=np.uint32) for _ in result]
    last_seq = -1
    last_sample = 0
    ordinal = 0
    quality = [0, 0]  # irregular block count, discarded unframed bytes
    unplaced: list[np.ndarray] = []
    unplaced_dtype = np.dtype([entry for entry in _OSD_TRACK.descr if entry[0] != "sample"])
    for slots in _record_batches(stream, quality):
        record_index = ordinal + np.arange(len(slots), dtype=np.uint32)
        hdr = slots.view("<u2")[:, 0]
        dlc = hdr & 0xF
        can_id = hdr >> 4
        for kind, cid in enumerate((0x092, 0x7F8, 0x388)):
            selected = (can_id == cid) & (record_index >= _PRELUDE_RECORDS) & (dlc > 0) & (dlc <= 8)
            idx = record_index[selected]
            width = dlc[selected]
            body = slots[selected, 4:][np.arange(8) < width[:, None]]
            old_tail = tails[kind]
            old_idx = tail_indices[kind]
            payload_bytes = np.concatenate((old_tail, body)) if old_tail.size else body
            offsets, end = _scan_messages(payload_bytes, kind)
            # Byte offsets that land in the previous batch's unfinished
            # packet inherit its original CAN record indices.
            stops = width.cumsum(dtype=np.int32)
            places = np.concatenate((offsets, np.arange(end, len(payload_bytes), dtype=np.int32)))
            starts_in_tail = places < old_tail.size
            indices = np.empty(len(places), dtype=np.uint32)
            indices[starts_in_tail] = old_idx[places[starts_in_tail]]
            indices[~starts_in_tail] = idx[
                np.searchsorted(stops, places[~starts_in_tail] - old_tail.size, side="right")
            ]
            tails[kind] = payload_bytes[end:].copy()
            tail_indices[kind] = indices[len(offsets) :].copy()
            if not offsets.size:
                continue
            if kind == 0:
                payload = payload_bytes[offsets[:, None] + np.arange(6, 126)]
                payload ^= payload[:, 117, None].copy()
                out = _track(payload, _OSD_PAYLOAD, _OSD_TRACK, indices[: len(offsets)])
                norms = np.sum(out["quaternion"] ** 2, axis=1)
                if not np.all(np.abs(norms - 1.0) < 0.02):
                    raise ValueError("invalid iOSD quaternion norm or unknown XOR layout")
                keep, samples, last_seq, last_sample = _sample_ordinals(
                    out["seq"], last_seq, last_sample
                )
                if not np.all(keep):
                    rejected = np.empty(np.count_nonzero(~keep), dtype=unplaced_dtype)
                    for name in _field_names(unplaced_dtype):
                        rejected[name] = out[name][~keep]
                    unplaced.append(rejected)
                    out = out[keep]
                out["sample"] = samples[keep]
            elif kind == 1:
                payload = payload_bytes[offsets[:, None] + np.arange(6, 74)]
                out = _track(payload, _GPS_PAYLOAD, _GPS_TRACK, indices[: len(offsets)])
            else:
                payload = payload_bytes[offsets[:, None] + np.arange(8, 179)]
                out = _track(payload, _CONTROL_PAYLOAD, _CONTROL_TRACK, indices[: len(offsets)])
                out["payload"] = payload
            result[kind].append(out)
        ordinal += len(slots)
    osd, gps, control = (
        np.concatenate(group) if group else np.empty(0, dtype=dtype)
        for group, dtype in zip(result, (_OSD_TRACK, _GPS_TRACK, _CONTROL_TRACK))
    )
    if not len(osd):
        raise ValueError("no complete OSD packets in iOSD CAN log")
    unordered = np.concatenate(unplaced) if unplaced else np.empty(0, dtype=unplaced_dtype)
    return IosdTelemetry(osd, gps, control, unordered, *quality)
