"""Lossless November frames on a synthetic FLY018 session (DAT + WAV + CSV + photo)."""

from __future__ import annotations

import hashlib
import json
import struct

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from data_processing.sources import motor_dat, november_frames
from data_processing.sources.new_motor_data import AUDIO_CLOCK, audio_seconds, read_aligned_motor
from data_processing.streams import frame_to_sample, sample_to_frame

FLIGHT = "FLY018"  # negative clock offset: early ticks fall before the WAV
SR = 8000
WRAP = 2**32
SNAN = 0x7FA00001  # signalling NaN: its payload bits must survive


def _record(rtype: int, tick: int, payload: bytes, corrupt: bool = False) -> bytes:
    key = tick & 0xFF
    body = (
        bytes([0x55, len(payload) + 12, 0x00, 0x00])
        + struct.pack("<HI", rtype, tick % WRAP)
        + bytes(b ^ key for b in payload)
    )
    v = 0x3692
    for byte in body:
        v = (v >> 8) ^ int(motor_dat._CRC16[(byte ^ v) & 0xFF])
    return body + struct.pack("<H", v ^ (1 if corrupt else 0))


def _defs(tick: int, lines: list[str]) -> list[bytes]:
    return [_record(65533, tick + i, line.encode() + b"\0") for i, line in enumerate(lines)]


def _motor(tick: int, rpm: tuple[int, int, int, int]) -> bytes:
    p = bytearray(motor_dat.MOTOR_PAYLOAD_LEN)
    for i, r in enumerate(rpm):
        struct.pack_into("<h", p, i * 19 + 3, r)
    return _record(motor_dat.MOTOR_RECORD_TYPE, tick, bytes(p))


def _gps(tick: int, date: int, hms: int) -> bytes:
    p = bytearray(68)
    struct.pack_into("<IIii", p, 0, date, hms, -1_234_567, 515_000_000)
    return _record(2096, tick, bytes(p))


def _mixed(tick: int, i8: int, u32: int, f32_bits: int, f64: float, i16: int) -> bytes:
    return _record(1006, tick, struct.pack("<bIIdh", i8, u32, f32_bits, f64, i16))


MIXED_DEFS = [
    "Op.[config]\nname\tmixed\ntype\t1006",
    "Op.int8_t\ti8 0",
    "Op.uint32_t\tu32 0",
    "Op.float\tf32 0",
    "Op.double\tf64 0",
    "Op.int16_t\ti16 0",
]
# FLY013's truncated fdi_info definition: (26, 22) from the file, (26, 26) restored.
FDI_DEFS = [
    "Op.[config]\nname\tfdi_info\ntype\t26",
    "Op.uint8_t\tgyro_bias_raw_flag_device 0",
    "Op.float\tgyrox_bias_raw_device 0",
    "Op.float\tgyroz_bias_raw_device 0",
    "Op.uint8_t\tgyro_static_flag 0",
    "Op.float\twx_stdvar 0",
    "Op.float\twy_stdvar 0",
    "Op.float\twz_stdvar 0",
]


@pytest.fixture
def session(tmp_path):
    offset_s, hz = AUDIO_CLOCK[FLIGHT]
    in_wav = int((0.1 - offset_s) * hz)  # 0.1 s into the WAV, past one tick wrap
    recs = [
        *_defs(10, MIXED_DEFS),
        *_defs(20, FDI_DEFS),
        _motor(1000, (5000, -3, 32767, -32768)),
        _mixed(1100, -128, 2**32 - 1, SNAN, float("-inf"), -2),
        _record(0x8000, 1150, b"start\0\0\0"),
        _motor(2000, (5100, 5101, 5102, 5103)),
        _motor(2000, (5200, 5201, 5202, 5203)),  # duplicate tick
        _motor(1500, (4900, 4901, 4902, 4903)),  # late packet
        b"\x55\x0f\x00garbage",  # invalid range
        _mixed(1600, 127, 0, 0x7FC00000, float("nan"), 32767),
        _record(0x8000, 1700, b"bad \xff\xfe \\x41 utf\0"),
        _record(4567, 1800, b"\xff\x00"),
        _record(4567, 1800, b"\x00\x10\x00"),  # other width, same tick
        _gps(1900, 20261114, 163751),
        _gps(1950, 0, 0),  # unset date -> NaT
        _record(26, 2100, bytes(range(22))),
        _record(26, 2200, bytes(range(26))),
        _motor(2500, (1, 2, 3, 4)),
        _record(4567, 2600, b"\x01\x02"),
        _record(1006, 2700, struct.pack("<bIIdh", 0, 7, 0x3F800000, 1.5, 0), corrupt=True),
        _motor(3000, (6000, 6001, 6002, 6003)),
    ]
    tick = 3000
    while in_wav - tick > 2**29:  # step through the wrap, packet gaps < 2**30
        tick += 2**29
        recs.append(_record(4568, tick, b"\x07"))
    recs.append(_motor(in_wav, (7000, 7001, 7002, 7003)))
    recs.append(_record(0x8000, in_wav + 5, b"done!"))
    recs.append(b"\x00" * 5)  # trailing padding

    d = tmp_path / "raw" / "14_Nov"
    d.mkdir(parents=True)
    dat = d / f"{FLIGHT}.dat"
    dat.write_bytes(b"\x05" + b"\x00" * 9 + motor_dat.DAT_MAGIC + b"\x12\x34" + b"".join(recs))
    pcm = np.random.default_rng(0).integers(-32768, 32768, size=(2000, 2), dtype=np.int16)
    pcm[0] = (-32768, 32767)
    sf.write(d / "ZOOM0003.wav", pcm, SR, subtype="PCM_16")
    (d / f"{FLIGHT}.csv").write_text("Clock:Tick#,Motor:Speed:RFront\n1000,5000\n")
    (d / "Image_1.jpg").write_bytes(b"\xff\xd8jpeg")
    return tmp_path / "raw", dat, pcm


def _round_trip(raw):
    [(key, frame)] = list(november_frames.build(raw))
    assert key == FLIGHT
    return sample_to_frame(frame_to_sample(frame))


def test_every_group_reproduces_the_reader_exactly(session):
    raw, dat, _ = session
    frame = _round_trip(raw)
    log = read_aligned_motor(dat)
    assert set(frame["dat"]) == set(log) - {"DatCon"}
    for name in frame["dat"]:
        pd.testing.assert_frame_equal(
            november_frames.group_to_table(frame, name), log[name], check_exact=True
        )

    mixed = november_frames.group_to_table(frame, "mixed")
    assert mixed["i8"].tolist() == [-128, 127] and mixed["i8"].dtype == np.int8
    assert mixed["u32"].tolist() == [2**32 - 1, 0] and mixed["u32"].dtype == np.uint32
    assert mixed["f32"].to_numpy().view(np.uint32).tolist() == [SNAN, 0x7FC00000]
    assert np.isneginf(mixed["f64"][0]) and np.isnan(mixed["f64"][1])
    utc = november_frames.group_to_table(frame, "gps_glns_00")["utc"]
    assert utc.dtype == "datetime64[s]" and utc.isna().tolist() == [False, True]
    assert str(utc[0]) == "2026-11-14 16:37:51"
    fdi = frame["meta"]["dat_layout"]["fdi_info"]
    assert [k["width"] for k in fdi["keys"]] == [22, 26] and set(frame["dat"]["fdi_info"]) == {
        "rows_22",
        "rows_26",
    }
    events = november_frames.group_to_table(frame, "EventLog")
    assert events["message"].tolist() == ["start", "bad \\xff\\xfe \\x41 utf", "done!"]
    payload = november_frames.group_to_table(frame, "record_4567")["payload"]
    assert payload.tolist() == [b"\xff\x00", b"\x00\x10\x00", b"\x01\x02"]


def test_rows_are_tick_ordered_events_on_the_audio_clock(session):
    raw, dat, _ = session
    frame = _round_trip(raw)
    motor = frame["dat"]["Motor"]["rows"]
    rows = motor.data
    assert rows["tick"].tolist() == sorted(rows["tick"].tolist())
    # file order 1000, 2000, 2000 (dup), 1500 (late), 2500, 3000, in-WAV
    assert rows["in_order"].tolist() == [True, False, True, False, True, True, True]
    ns = np.rint(audio_seconds(FLIGHT, rows["tick"]) * 1e9).astype(np.int64)
    np.testing.assert_array_equal(motor.tindex.abs_stamps_ticks, ns)
    # The variable-width stream keeps the NUL padding the reader strips.
    ev = frame["dat"]["EventLog"]
    first = ev["rows"].data[0]
    assert ev["payload"].data[first["payload_start"] :][: first["payload_length"]].tobytes() == (
        b"start\0\0\0"
    )


def test_audio_rps_and_dat_bytes_are_native(session, tmp_path):
    raw, dat, pcm = session
    frame = _round_trip(raw)
    audio = frame["audio"]
    assert audio.dims == ("mic", "time") and audio.tindex.sr == SR and audio.t_start == 0.0
    np.testing.assert_array_equal(audio.data, pcm.T.astype(np.float32) / 32768)

    rps = frame["rps"]
    motor = frame["dat"]["Motor"]["rows"].data
    assert rps.dims == ("rotor", "time")
    np.testing.assert_array_equal(
        rps.data, np.stack([motor[r] for r in motor_dat.ROTORS]).astype(np.float64) / 60
    )
    stamps = rps.tindex.abs_stamps
    assert stamps[0] < 0 < stamps[-1] < frame["audio"].t_end  # not cropped to the WAV
    assert rps.dim_size("time") == 7 and frame.t_start <= rps.t_start

    data = dat.read_bytes()
    assert frame["dat_bytes"].data.tobytes() == data
    files = frame["meta"]["files"]
    assert files["dat"]["sha256"] == hashlib.sha256(data).hexdigest()
    assert files["dat"]["relpath"] == f"14_Nov/{FLIGHT}.dat" and files["dat"]["size"] == len(data)
    assert files["csv"]["name"] == f"{FLIGHT}.csv" and files["other"][0]["name"] == "Image_1.jpg"
    assert november_frames.restore_dat(frame, tmp_path / "copy.dat").read_bytes() == data
    clock = frame["meta"]["clock"]
    assert (clock["offset_s"], clock["ticks_per_second"]) == AUDIO_CLOCK[FLIGHT]
    stats = frame["meta"]["dat"]
    assert stats["bytes_outside_packets"] > 0 and stats["n_bytes"] == len(data)
    meta = {k: frame["meta"][k] for k in frame["meta"] if k not in ("system", "observation")}
    json.dumps(meta, allow_nan=False)


def test_time_slice_keeps_payload_stream(session):
    raw, _, _ = session
    frame = _round_trip(raw)
    clip = frame.time[0.0:0.25]
    assert clip["rps"].dim_size("time") == 1
    ev = clip["dat"]["EventLog"]
    row = ev["rows"].data[0]
    stream = ev["payload"].data
    assert stream[row["payload_start"] :][: row["payload_length"]].tobytes() == b"done!"


def test_lossy_wav_encoding_is_refused(session):
    raw, dat, pcm = session
    sf.write(dat.parent / "ZOOM0003.wav", pcm / 32768.0, SR, subtype="FLOAT")
    with pytest.raises(ValueError, match="FLOAT"):
        list(november_frames.build(raw))
