"""Native reader for DJI M100 flight-controller ``.DAT`` logs (``DJI_LOG_V3``).

Reads the per-motor speed telemetry straight from the original ``FLYnnn.DAT``
with each record's own flight-controller tick — no DatCon CSV in between.

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

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

DAT_MAGIC = b"DJI_LOG_V3"
MOTOR_RECORD_TYPE = 10090
MOTOR_PAYLOAD_LEN = 152
#: Column order of the motor blocks in the payload (= DatCon CSV suffixes).
ROTORS = ("RFront", "LFront", "LBack", "RBack")
_ROTOR_BLOCK = 19
_SPEED_OFFSET = 3
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
def _motor_records(buf: np.ndarray, start: int) -> Iterator[tuple[int, int, bool]]:
    """Unwrap time across all CRC-valid packets, yielding only motor records."""
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
        if (
            length == MOTOR_PAYLOAD_LEN + _HEADER_LEN + _CRC_LEN
            and (int(buf[p + 4]) | (int(buf[p + 5]) << 8)) == MOTOR_RECORD_TYPE
        ):
            yield int(p), int(tick), bool(in_order)
        p += length


def read_motor_dat(path: str | Path, clock_hz: float | None = None) -> pd.DataFrame:
    """All native motor-speed records of an M100 ``.DAT``, with native ticks.

    One row per CRC-valid motor record (type 10090, ~16.7 Hz), in file order:

    - ``tick`` — int64 flight-controller tick, unwrapped over every valid
      record of the file; ``tick & 0xFFFFFFFF`` is the stored value and the
      high word is 0 at the file's first record.
    - ``tick_raw`` — the stored uint32 tick; ``offset`` — record byte offset.
    - ``in_order`` — tick exceeds every earlier record's tick in the file;
      late but valid packets are retained regardless.
    - ``RFront``, ``LFront``, ``LBack``, ``RBack`` — int16 motor speed, RPM.
    - ``time_s`` (only if ``clock_hz`` is given) — ``(tick - tick[0]) /
      clock_hz``. The tick rate is a calibration choice: :data:`DATCON_CLOCK_HZ`
      is DatCon's nominal value, not a measurement.
    """
    buf = np.memmap(path, dtype=np.uint8, mode="r")
    magic = buf[:4096].tobytes().find(DAT_MAGIC)
    if magic < 0:
        raise ValueError(f"{path}: no {DAT_MAGIC.decode()} header")
    records = np.fromiter(
        _motor_records(buf, magic + len(DAT_MAGIC)),
        dtype=[("offset", "i8"), ("tick", "i8"), ("in_order", "?")],
    )
    if not records.size:
        raise ValueError(f"{path}: no valid four-rotor M100 motor records")
    o = records["offset"] + _HEADER_LEN
    tick = records["tick"]
    key = tick.astype(np.uint8)
    cols: dict[str, np.ndarray] = {
        "tick": tick,
        "tick_raw": tick.astype(np.uint32),
        "offset": records["offset"],
        "in_order": records["in_order"],
    }
    for i, name in enumerate(ROTORS):
        at = o + i * _ROTOR_BLOCK + _SPEED_OFFSET
        lo = (buf[at] ^ key).astype(np.uint16)
        hi = (buf[at + 1] ^ key).astype(np.uint16)
        cols[name] = (lo | (hi << 8)).view(np.int16)
    df = pd.DataFrame(cols)
    if clock_hz is not None:
        df["time_s"] = (df["tick"] - df["tick"].iloc[0]) / clock_hz
    return df
