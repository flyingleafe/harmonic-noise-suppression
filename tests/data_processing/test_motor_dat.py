"""Native DJI_LOG_V3 motor reader on a synthetic .DAT (record framing, CRC, XOR, wrap)."""

from __future__ import annotations

import struct

import numpy as np

from data_processing.sources import motor_dat


def _record(rtype: int, tick: int, payload: bytes, corrupt: bool = False) -> bytes:
    length = len(payload) + 12
    key = tick & 0xFF
    body = (
        bytes([0x55, length, 0x00, 0x00])
        + struct.pack("<HI", rtype, tick)
        + bytes(b ^ key for b in payload)
    )
    v = 0x3692
    for byte in body:
        v = (v >> 8) ^ int(motor_dat._CRC16[(byte ^ v) & 0xFF])
    return body + struct.pack("<H", v ^ (1 if corrupt else 0))


def _motor_payload(rpm: tuple[int, int, int, int]) -> bytes:
    p = bytearray(motor_dat.MOTOR_PAYLOAD_LEN)
    for i, r in enumerate(rpm):
        struct.pack_into("<h", p, i * 19 + 3, r)
    return bytes(p)


def test_read_motor_dat_unwraps_and_decodes(tmp_path):
    w = 2**32
    recs = [
        _record(2160, w - 90_000, bytes(24)),
        _record(motor_dat.MOTOR_RECORD_TYPE, w - 60_000, _motor_payload((5000, 5100, -3, 4900))),
        _record(motor_dat.MOTOR_RECORD_TYPE, 20_000, _motor_payload((1, 2, 3, 4)), corrupt=True),
        b"\x00" * 7,
        _record(2160, 10_000, bytes(24)),  # post-wrap record ahead of a late pre-wrap one
        _record(motor_dat.MOTOR_RECORD_TYPE, w - 5_000, _motor_payload((6000, 6001, 6002, 6003))),
        _record(motor_dat.MOTOR_RECORD_TYPE, 40_000, _motor_payload((7000, 7100, 7200, 7300))),
    ]
    path = tmp_path / "FLY000.DAT"
    path.write_bytes(b"\x05" + b"\x00" * 241 + motor_dat.DAT_MAGIC + b"\x12\x34" + b"".join(recs))

    df = motor_dat.read_motor_dat(path, clock_hz=84e6)

    assert df["tick"].tolist() == [w - 60_000, w - 5_000, w + 40_000]
    assert df["tick_raw"].tolist() == [w - 60_000, w - 5_000, 40_000]
    assert df["in_order"].tolist() == [True, False, True]
    assert df[list(motor_dat.ROTORS)].to_numpy().tolist() == [
        [5000, 5100, -3, 4900],
        [6000, 6001, 6002, 6003],
        [7000, 7100, 7200, 7300],
    ]
    np.testing.assert_allclose(df["time_s"], [0.0, 55_000 / 84e6, 100_000 / 84e6])
