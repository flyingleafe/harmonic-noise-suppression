"""Native reader for DJI M100 flight-controller ``.DAT`` logs (``DJI_LOG_V3``).

Native groups are decoded independently, on their own packet clocks. The
optional sibling DatCon CSV is a separate export, never a native-data source.

Container format (as decoded by DatCon, BudWalkerJava/DatCon,
``src/V3/Files/DatFileV3.java`` + ``src/DatConRecs/Payload.java``): after the
``DJI_LOG_V3`` magic the file is a stream of records

    0x55 | len u8 | 0x00 | hdr u8 | type u16 | tick u32 | payload | crc u16

(little-endian, ``len`` counts the whole record). The CRC-16 (seed 0x3692,
table below) covers everything before the CRC; every payload byte is XORed
with ``tick & 0xFF``. ``tick`` is a free-running 32-bit counter, so it WRAPS:
DatCon assumes 85 MHz for the M100 (one wrap per 50.53 s), GPS regression on
these logs measures ~84.0025 MHz (one wrap per ~51.13 s).

Motor speeds live in record type 10090 with a 152-byte payload
(``src/DatConRecs/Created4V3/RecMotor152_10090.java``, 4-motor branch): one
19-byte block per rotor in the order RFront, LFront, LBack, RBack; speed is
the int16 at byte 3 of the block, in RPM (the CSV's ``Motor:Speed:*``).

DatCon assumes an 85 MHz clock and exports held values on a separate row
clock. FLY013's export also drops out-of-order motor packets and inserts
five spurious wraps. Reading native ticks retains the original timing,
including spin-down packets after the CSV ends.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from numpy.typing import ArrayLike, NDArray

from data_processing.sources._dat_records import BUILTIN, decode_builtin
from data_processing.sources._dat_records import ROTORS as ROTORS
from data_processing.sources._dat_schema import DEFINITION_RECORD_TYPE, parse_definitions

DAT_MAGIC = b"DJI_LOG_V3"
MOTOR_RECORD_TYPE = 10090
MOTOR_PAYLOAD_LEN = 152
_HEADER_LEN = 10
_CRC_LEN = 2
#: DatCon's assumed M100 tick rate (``DatHeader.getAcType``). NOMINAL, not a
#: measurement — GPS time on FLY018/FLY022 gives 84.0025 MHz.
DATCON_CLOCK_HZ = 85e6

# fmt: off
_CRC16 = np.array([
    0x0000, 0x1189, 0x2312, 0x329b, 0x4624, 0x57ad, 0x6536, 0x74bf, 0x8c48, 0x9dc1, 0xaf5a, 0xbed3, 0xca6c, 0xdbe5, 0xe97e, 0xf8f7,
    0x1081, 0x0108, 0x3393, 0x221a, 0x56a5, 0x472c, 0x75b7, 0x643e, 0x9cc9, 0x8d40, 0xbfdb, 0xae52, 0xdaed, 0xcb64, 0xf9ff, 0xe876,
    0x2102, 0x308b, 0x0210, 0x1399, 0x6726, 0x76af, 0x4434, 0x55bd, 0xad4a, 0xbcc3, 0x8e58, 0x9fd1, 0xeb6e, 0xfae7, 0xc87c, 0xd9f5,
    0x3183, 0x200a, 0x1291, 0x0318, 0x77a7, 0x662e, 0x54b5, 0x453c, 0xbdcb, 0xac42, 0x9ed9, 0x8f50, 0xfbef, 0xea66, 0xd8fd, 0xc974,
    0x4204, 0x538d, 0x6116, 0x709f, 0x0420, 0x15a9, 0x2732, 0x36bb, 0xce4c, 0xdfc5, 0xed5e, 0xfcd7, 0x8868, 0x99e1, 0xab7a, 0xbaf3,
    0x5285, 0x430c, 0x7197, 0x601e, 0x14a1, 0x0528, 0x37b3, 0x263a, 0xdecd, 0xcf44, 0xfddf, 0xec56, 0x98e9, 0x8960, 0xbbfb, 0xaa72,
    0x6306, 0x728f, 0x4014, 0x519d, 0x2522, 0x34ab, 0x0630, 0x17b9, 0xef4e, 0xfec7, 0xcc5c, 0xddd5, 0xa96a, 0xb8e3, 0x8a78, 0x9bf1,
    0x7387, 0x620e, 0x5095, 0x411c, 0x35a3, 0x242a, 0x16b1, 0x0738, 0xffcf, 0xee46, 0xdcdd, 0xcd54, 0xb9eb, 0xa862, 0x9af9, 0x8b70,
    0x8408, 0x9581, 0xa71a, 0xb693, 0xc22c, 0xd3a5, 0xe13e, 0xf0b7, 0x0840, 0x19c9, 0x2b52, 0x3adb, 0x4e64, 0x5fed, 0x6d76, 0x7cff,
    0x9489, 0x8500, 0xb79b, 0xa612, 0xd2ad, 0xc324, 0xf1bf, 0xe036, 0x18c1, 0x0948, 0x3bd3, 0x2a5a, 0x5ee5, 0x4f6c, 0x7df7, 0x6c7e,
    0xa50a, 0xb483, 0x8618, 0x9791, 0xe32e, 0xf2a7, 0xc03c, 0xd1b5, 0x2942, 0x38cb, 0x0a50, 0x1bd9, 0x6f66, 0x7eef, 0x4c74, 0x5dfd,
    0xb58b, 0xa402, 0x9699, 0x8710, 0xf3af, 0xe226, 0xd0bd, 0xc134, 0x39c3, 0x284a, 0x1ad1, 0x0b58, 0x7fe7, 0x6e6e, 0x5cf5, 0x4d7c,
    0xc60c, 0xd785, 0xe51e, 0xf497, 0x8028, 0x91a1, 0xa33a, 0xb2b3, 0x4a44, 0x5bcd, 0x6956, 0x78df, 0x0c60, 0x1de9, 0x2f72, 0x3efb,
    0xd68d, 0xc704, 0xf59f, 0xe416, 0x90a9, 0x8120, 0xb3bb, 0xa232, 0x5ac5, 0x4b4c, 0x79d7, 0x685e, 0x1ce1, 0x0d68, 0x3ff3, 0x2e7a,
    0xe70e, 0xf687, 0xc41c, 0xd595, 0xa12a, 0xb0a3, 0x8238, 0x93b1, 0x6b46, 0x7acf, 0x4854, 0x59dd, 0x2d62, 0x3ceb, 0x0e70, 0x1ff9,
    0xf78f, 0xe606, 0xd49d, 0xc514, 0xb1ab, 0xa022, 0x92b9, 0x8330, 0x7bc7, 0x6a4e, 0x58d5, 0x495c, 0x3de3, 0x2c6a, 0x1ef1, 0x0f78,
], dtype=np.uint32)
# fmt: on


@njit(cache=True)
def _packet_index(buf: np.ndarray, start: int):
    """Index validated packets by (type, width) in one pass, without decoding payloads."""
    blocks = {}
    counts = {}
    first_tick = -1
    n = buf.size
    previous = np.int64(-1)
    tick = np.int64(0)
    maximum = np.int64(-1)
    p = start
    while p + 12 <= n:
        if buf[p] != 0x55:
            p += 1
            continue
        length = np.int64(buf[p + 1])
        if length < 12 or buf[p + 2] != 0 or p + length > n:
            p += 1
            continue
        v = np.uint32(0x3692)
        for i in range(length - 2):
            v = (v >> np.uint32(8)) ^ _CRC16[(np.uint32(buf[p + i]) ^ v) & np.uint32(0xFF)]
        stored = np.uint32(buf[p + length - 2]) | (np.uint32(buf[p + length - 1]) << np.uint32(8))
        if v != stored:
            p += 1
            continue
        raw = (
            np.int64(buf[p + 6])
            | (np.int64(buf[p + 7]) << 8)
            | (np.int64(buf[p + 8]) << 16)
            | (np.int64(buf[p + 9]) << 24)
        )
        if previous < 0:
            tick = raw
            first_tick = int(raw)
        else:
            # Late packets can straddle a wrap in either direction. The
            # contiguous M100 logs have subsecond steps; reject gross jumps.
            step = ((raw - previous + 2**31) % 2**32) - 2**31
            if abs(step) >= 2**30:
                raise ValueError("discontinuous DAT packet clock")
            tick += step
        previous = raw
        in_order = tick > maximum
        maximum = max(maximum, tick)
        rtype = int(buf[p + 4]) | (int(buf[p + 5]) << 8)
        key = (rtype << 8) | (int(length) - _HEADER_LEN - _CRC_LEN)
        if key not in blocks:
            blocks[key] = np.empty((16, 2), np.int64)
            counts[key] = 0
        size = counts[key]
        block = blocks[key]
        if size == len(block):
            larger = np.empty((size * 2, 2), np.int64)
            larger[:size] = block
            blocks[key] = block = larger
        block[size, 0] = p if in_order else -p
        block[size, 1] = tick
        counts[key] = size + 1
        p += length
    return blocks, counts, first_tick


@njit(cache=True)
def _payloads(buf: np.ndarray, index: np.ndarray, width: int) -> np.ndarray:
    out = np.empty((len(index), width), np.uint8)
    for i in range(len(index)):
        start = abs(index[i, 0]) + _HEADER_LEN
        key = index[i, 1] & 0xFF
        for j in range(width):
            out[i, j] = buf[start + j] ^ key
    return out


def datcon_ticks(ticks: ArrayLike) -> NDArray[np.int64]:
    """Unwrap a full ordered DatCon Clock:Tick# column from its native low word.

    The first row is in epoch zero; successive CSV rows must be less than
    one wrap apart. Repair the clock *before* slicing (FLY013 has false high words).
    """
    low = np.asarray(ticks).astype(np.uint32, casting="unsafe", copy=False)
    if low.ndim != 1:
        raise ValueError("expected the one-dimensional Clock:Tick# column")
    result = np.empty(low.size, dtype=np.int64)
    if result.size:
        result[0] = low[0]
        np.cumsum(np.diff(low), dtype=np.int64, out=result[1:])
        result[1:] += result[0]
    return result


_TEXT_GROUPS = {
    0x8000: "EventLog",
    0xFF00: "SDLog",
    0xFFFC: "ModuleName",
    DEFINITION_RECORD_TYPE: "RecDefs",
    0xFFFE: "FirmwareVersion",
    0xFFFF: "ConfigLog",
}
_EXPORT = "DatCon"
_CLOCK_FIELDS = frozenset({"tick", "tick_raw", "offset", "in_order", "time_s", "audio_s"})
type _Key = tuple[int, int]


def _fields(payload: np.ndarray, dtype: np.dtype) -> Iterator[tuple[str, np.ndarray]]:
    records = payload.view(dtype).reshape(-1)
    for field in dtype.names or ():
        yield field, records[field]


class DatTelemetry(Mapping[str, pd.DataFrame]):
    """Lazy native tables, one row per packet; bind a table to keep it in memory.

    ``list(log)`` lists available groups without loading them. Each lookup
    decodes just that group. ``log["Motor"]`` includes the four RPM columns.
    ``log["DatCon"]`` (only when a sibling CSV exists) preserves *every*
    export column, including computed signals, on the corrected CSV row clock.
    Unspecified packet formats are exposed losslessly as bytes, not guessed.
    """

    def __init__(
        self,
        path: Path,
        clock_hz: float | None,
        audio_clock: Callable[[np.ndarray], np.ndarray] | None,
    ) -> None:
        self._buf = np.memmap(path, dtype=np.uint8, mode="r")
        header = self._buf[:4096].tobytes().find(DAT_MAGIC)
        if header < 0:
            raise ValueError(f"{path}: no {DAT_MAGIC.decode()} header")
        blocks, counts, self._origin = _packet_index(self._buf, header + len(DAT_MAGIC))
        if not counts:
            raise ValueError(f"{path}: no valid packets")
        self._index: dict[_Key, np.ndarray] = {
            (k >> 8, k & 255): block[: counts[k]] for k, block in blocks.items()
        }
        self._hz = clock_hz
        self._audio_clock = audio_clock
        defs = []
        for (rtype, width), idx in self._index.items():
            if rtype == DEFINITION_RECORD_TYPE:
                rows = _payloads(self._buf, idx, width)
                defs.extend(
                    (abs(pos), row.tobytes()) for pos, row in zip(idx[:, 0], rows, strict=True)
                )
        self._schemas = parse_definitions(msg for _, msg in sorted(defs))
        self._groups: dict[str, list[_Key]] = {}
        for key in self._index:
            rtype, _ = key
            spec = self._schemas.get(key)
            builtin = BUILTIN.get(key)
            layout = builtin if builtin is not None else spec
            name = layout[0] if layout is not None else _TEXT_GROUPS.get(rtype, f"record_{rtype}")
            if name in self._groups and self._groups[name][0][0] != rtype:
                name = f"{name}:{rtype}"
            self._groups.setdefault(name, []).append(key)
        self._csv = path.with_suffix(".csv")
        if self._csv.is_file():
            self._groups[_EXPORT] = []

    def __iter__(self) -> Iterator[str]:
        return iter(self._groups)

    def __len__(self) -> int:
        return len(self._groups)

    def __contains__(self, key: object) -> bool:
        return key in self._groups

    def __repr__(self) -> str:
        return f"DatTelemetry({', '.join(self._groups)})"

    def _native(self, key: _Key) -> pd.DataFrame:
        idx = self._index[key]
        rtype, width = key
        payload = _payloads(self._buf, idx, width)
        cols = {
            "tick": idx[:, 1],
            "tick_raw": idx[:, 1].astype(np.uint32),
            "offset": np.abs(idx[:, 0]),
            "in_order": idx[:, 0] > 0,
        }
        if key in BUILTIN:
            for field, values in _fields(payload, BUILTIN[key][1]):
                cols[f"packet:{field}" if field in _CLOCK_FIELDS else field] = values
            for field, values in decode_builtin(rtype, payload).items():
                if field in cols or field in _CLOCK_FIELDS:
                    raise ValueError(f"duplicate engineered column {field!r} in record {rtype}")
                cols[field] = values
        if key in self._schemas:
            schema = self._schemas[key][1]
            fields = dict(schema.fields or {})
            builtin_fields = dict(BUILTIN[key][1].fields or {}) if key in BUILTIN else {}
            for name, values in _fields(payload, schema):
                if name in builtin_fields and fields.get(name) == builtin_fields[name]:
                    continue  # same named field, same byte offset and representation
                if name in cols or name in _CLOCK_FIELDS:
                    name = f"firmware:{name}"
                cols[name] = values
        elif key not in BUILTIN:
            if rtype in _TEXT_GROUPS:
                cols["message"] = np.array(
                    [
                        row.tobytes().rstrip(b"\x00").decode("utf-8", "backslashreplace")
                        for row in payload
                    ],
                    dtype=object,
                )
            else:
                cols["payload"] = np.array([row.tobytes() for row in payload], dtype=object)
        return pd.DataFrame(cols)

    def __getitem__(self, group: str) -> pd.DataFrame:
        keys = self._groups[group]
        if group == _EXPORT:
            df = pd.read_csv(self._csv, low_memory=False)
            df.insert(0, "tick", datcon_ticks(df["Clock:Tick#"]))
        elif len(keys) == 1:
            df = self._native(keys[0])
        else:
            df = pd.concat([self._native(k) for k in keys], ignore_index=True)
            df.sort_values("offset", inplace=True, ignore_index=True)
        if self._hz is not None:
            df["time_s"] = (df["tick"] - self._origin) / self._hz
        if self._audio_clock is not None:
            df.insert(0, "audio_s", self._audio_clock(df["tick"].to_numpy()))
        return df


def read_motor_dat(
    path: str | Path,
    clock_hz: float | None = None,
    *,
    audio_clock: Callable[[np.ndarray], np.ndarray] | None = None,
) -> DatTelemetry:
    """Open all telemetry on its native packet clocks; no CSV or network required.

    If requested, ``time_s`` is relative to the first valid packet in the file
    at the given tick rate, shared by all groups. ``audio_clock`` maps the
    unwrapped tick array of each table to seconds from an external WAV start.
    """
    return DatTelemetry(Path(path), clock_hz, audio_clock)
