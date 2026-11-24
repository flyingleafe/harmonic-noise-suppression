"""Native DJI_LOG_V3 motor reader on a synthetic .DAT (record framing, CRC, XOR, wrap)."""

from __future__ import annotations

import struct

import numpy as np

from data_processing.sources import motor_dat
from data_processing.sources.new_motor_data import audio_seconds, read_aligned_motor


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

    log = motor_dat.read_motor_dat(path, clock_hz=84e6)
    df = log["Motor"]

    assert df["tick"].tolist() == [w - 60_000, w - 5_000, w + 40_000]
    assert df["tick_raw"].tolist() == [w - 60_000, w - 5_000, 40_000]
    assert df["in_order"].tolist() == [True, False, True]
    assert df[list(motor_dat.ROTORS)].to_numpy().tolist() == [
        [5000, 5100, -3, 4900],
        [6000, 6001, 6002, 6003],
        [7000, 7100, 7200, 7300],
    ]
    np.testing.assert_allclose(df["time_s"], np.array([30_000, 85_000, 130_000]) / 84e6)


def test_independent_native_and_export_clocks(tmp_path):
    """The file has no motor packet; don't discard other sensors or date them by CSV rows."""
    wrap = 2**32
    raw = tmp_path / "FLY013.dat"
    records = [
        _record(65533, 1, b"Op.[config]\nname\ttest_pressure\ntype\t4321\0"),
        _record(65533, 2, b"Op.int16_t\tpressure 0\0"),
        _record(65533, 3, b"Op.[config]\nname\ttest_status\ntype\t4322\0"),
        _record(65533, 4, b"Op.uint8_t\tstatus 0\0"),
        _record(4321, 101, struct.pack("<h", -37)),
        _record(0x8000, 124, b"start\0"),
        _record(4567, 131, b"\xff\x00"),
        _record(0x8000, 149, b"status OK\0"),
        _record(4321, 175, struct.pack("<h", 108)),
        _record(0x8000, 182, b"done!\0"),
        _record(4567, 190, b"\0\x10\x00"),
    ]
    raw.write_bytes(motor_dat.DAT_MAGIC + b"".join(records))
    native = read_aligned_motor(raw)
    assert "DatCon" not in native
    pressure = native["test_pressure"]
    assert pressure["pressure"].tolist() == [-37, 108]
    assert pressure["tick"].tolist() == [101, 175]
    np.testing.assert_array_equal(pressure["audio_s"], audio_seconds("FLY013", [101, 175]))
    assert native["EventLog"]["message"].tolist() == ["start", "status OK", "done!"]
    assert native["record_4567"]["payload"].tolist() == [b"\xff\x00", b"\0\x10\x00"]

    # Standalone DAT values stay on their original clock, and the export keeps
    # all its columns (including computed and nonnumeric ones), not just RPM.
    raw.with_suffix(".csv").write_text(
        f"Clock:Tick#,pressure:C,Attribute|Value\n0,,Model|M100\n{wrap + 150},99,\n",
    )
    log = read_aligned_motor(raw)
    exported = log["DatCon"]
    assert exported["Clock:Tick#"].tolist() == [0, wrap + 150]
    assert exported["tick"].tolist() == [0, 150]
    assert exported["pressure:C"].iloc[1] == 99
    assert exported["Attribute|Value"].iloc[0] == "Model|M100"
    np.testing.assert_array_equal(exported["audio_s"], audio_seconds("FLY013", [0, 150]))
    assert log["test_pressure"]["tick"].tolist() == [101, 175]
