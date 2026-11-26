"""UaVirBASE v1.0: one exact 8-channel Frame per take directory, streamed from the release ZIP."""

from __future__ import annotations

import hashlib
import json
import struct
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from data_processing.derivations import _acoustic_key
from data_processing.sources import uavirbase as uv
from data_processing.streams import frame_to_sample, sample_to_frame

SR = uv.SAMPLE_RATE
RAIL_POS = 2**31 - 256  # 0x7FFFFF00
AMBIENT, FRONT, LEFT = "20241115_093128", "20241115_093611", "20241115_093711"


def _wav(pcm: np.ndarray, *, channels: int = uv.N_MICS) -> bytes:
    """``(C, T)`` int32 -> canonical 44-byte-header PCM32 WAV."""
    data = np.ascontiguousarray(pcm.T).astype("<i4").tobytes()
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(data),
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        SR,
        SR * channels * 4,
        channels * 4,
        32,
        b"data",
        len(data),
    )
    return header + data


def _label(start: str, n: int, drone: dict | None, mics=uv.MICROPHONES) -> str:
    t0 = datetime.strptime(start, "%Y-%m-%d %H:%M:%S.%f")
    end = t0 + timedelta(seconds=n / SR + 0.098)
    blank = {k: None for k in ("type", "movement", "distance", "height", "azimuth", "rotation")}
    lab = {
        "audio_interface_name": "Behringer UMC1820",
        "start_recording_time": start,
        "end_recording_time": end.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "additional_info": "",
        "weather_data": {
            "timestamp": (t0 + timedelta(minutes=59)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "+00:00",
            "measurements": {
                "air temperature": "4.6 C",
                "air humidity": "90 % RH",
                "light intensity": "10792 Lux",
                "barometric pressure": "100810 Pa",
                "wind direction": "240 degrees",
                "wind speed": "1.7 m/s",
                "rainfall hourly": "0 mm/hour",
                "uv index": 1.3,
            },
        },
        "microphone_array": {**uv.ARRAY_CENTER, "microphones": [dict(m) for m in mics]},
        "drone": {"sound_source": "Ambient Noise", **blank, "additional_info": ""}
        if drone is None
        else {
            "sound_source": "Drone",
            "type": "DJI Mavic 3 cine",
            "movement": "Static",
            **drone,
            "additional_info": "",
        },
    }
    return json.dumps(lab, indent=4)


def _pcm24(rng: np.random.Generator, n: int) -> np.ndarray:
    """24-bit content left-aligned in int32, with channel c offset so order is visible."""
    x = rng.integers(-(2**23), 2**23, size=(uv.N_MICS, n), dtype=np.int64)
    x[:, 0] = np.arange(1, uv.N_MICS + 1)  # channel marker
    x[0, 5], x[1, 6] = -(2**23), 2**23 - 1  # both 24-bit rails survive untouched
    return (x << 8).astype(np.int32)


def _takes(rng: np.random.Generator) -> dict[str, tuple[str, np.ndarray]]:
    exact = _pcm24(rng, 300)
    inexact = _pcm24(rng, 300)
    inexact[2, 200] = 2**31 - 1  # 31 significant bits: float32 rounds it to 2**31
    inexact[3, 201] = 2**24 + 1  # 25 significant bits
    inexact[4, 202] = -(2**31) + 1
    return {
        AMBIENT: (_label("2024-11-15 09:31:28.676", 200, None), _pcm24(rng, 200)),
        FRONT: (
            _label(
                "2024-11-15 09:36:11.641",
                300,
                {"distance": "20", "height": "20", "azimuth": "0", "rotation": "Front"},
            ),
            exact,
        ),
        LEFT: (
            _label(
                "2024-11-15 09:37:11.095",
                300,
                {"distance": "20", "height": "10", "azimuth": "45", "rotation": "Left"},
            ),
            inexact,
        ),
    }


def _zip(
    path: Path, takes: dict[str, tuple[str, np.ndarray]], extra: dict[str, bytes] | None = None
) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{uv.ARCHIVE_ROOT}/", b"")
        for d, (label, pcm) in takes.items():
            zf.writestr(f"{uv.ARCHIVE_ROOT}/{d}/", b"")
            zf.writestr(f"{uv.ARCHIVE_ROOT}/{d}/{uv.LABEL_NAME}", label)
            zf.writestr(f"{uv.ARCHIVE_ROOT}/{d}/{uv.WAV_NAME}", _wav(pcm))
        for name, blob in (extra or {}).items():
            zf.writestr(name, blob)
    return path


def _build(path: Path, n_drone: int = 2, n_ambient: int = 1) -> dict:
    return dict(uv.build_from_zip(path, n_drone=n_drone, n_ambient=n_ambient, full_design=False))


@pytest.fixture(autouse=True)
def _small_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(uv, "BLOCK_FRAMES", 64)  # several blocks per fixture take


def test_takes_decode_exactly_in_channel_order(tmp_path: Path) -> None:
    takes = _takes(np.random.default_rng(0))
    frames = _build(_zip(tmp_path / uv.ZIP_NAME, takes))
    assert list(frames) == [AMBIENT, FRONT, LEFT]  # key = original directory, chronological

    front = frames[FRONT]
    audio = front["audio"]
    data = np.asarray(audio.data)
    assert audio.dims == ("mic", "time") and data.dtype == np.float32 and data.shape == (8, 300)
    np.testing.assert_array_equal(data.astype(np.float64), takes[FRONT][1] / 2.0**31)
    np.testing.assert_array_equal(
        data[:, 0] * 2.0**31 / 256, np.arange(1, 9)
    )  # WAV interleave order
    assert (
        data[0, 5] == -1.0 and data[1, 6] * 2.0**31 == RAIL_POS
    )  # rails kept, not normalized away
    mic_labels = tuple(audio.indexes["mic"].labels)
    assert mic_labels == uv.CHANNEL_NAMES == tuple(front["mic_pos"].indexes["mic"].labels)
    assert tuple(front["mic_pos"].indexes["enu"].labels) == ("east", "north", "up")

    meta = front["meta"]
    assert meta["recording_id"] == FRONT and meta["dataset"] == uv.FRAMES_DATASET
    assert (
        meta["operating"]["condition"] == "flight" and meta["operating"]["flight_mode"] == "hover"
    )
    lab = meta["label"]
    assert (lab["distance_m"], lab["height_m"], lab["azimuth_deg"], lab["side_facing_array"]) == (
        20.0,
        20.0,
        0.0,
        "front",
    )
    assert lab["subcategory"] == "mavic3cine_static_dist20m_height20m"
    assert meta["label_json"] == takes[FRONT][0]  # complete publisher JSON, byte for byte
    assert meta["weather"]["measurements"]["wind speed"] == {"value": 1.7, "unit": "m/s"}
    assert meta["weather"]["measurements_raw"]["air temperature"] == "4.6 C"
    clock = meta["clock"]
    assert clock["start_recording_time"] == "2024-11-15 09:36:11.641"
    assert clock["previous_take"] == AMBIENT and clock["next_take"] == LEFT
    assert clock["gap_from_previous_s"] > 0 and clock["weather_timestamp"].endswith("+00:00")
    assert [m["mic"] for m in meta["microphones"]] == list(uv.CHANNEL_NAMES)
    assert meta["microphones"][4]["raw"]["latitude"] == uv.MICROPHONES[4]["latitude"]
    assert meta["pcm_float32_exact"] and meta["channel_valid_range"] == [[0, 300]] * 8

    amb = frames[AMBIENT]["meta"]
    assert (
        amb["operating"]["condition"] == "rotors_off" and amb["operating"]["n_active_rotors"] == 0
    )
    assert amb["label"]["subcategory"] == "ambient_noise" and amb["system"]["category"] == "ambient"
    assert json.loads(amb["label_json"])["drone"]["type"] is None

    for key, frame in frames.items():
        assert _acoustic_key(key, frame).startswith("external/")
        back = sample_to_frame(frame_to_sample(frame))
        np.testing.assert_array_equal(
            np.asarray(back["audio"].data), np.asarray(frame["audio"].data)
        )
        assert np.asarray(back["audio"].data).dtype == np.asarray(frame["audio"].data).dtype
        assert tuple(back["mic_pos"].indexes["mic"].labels) == uv.CHANNEL_NAMES


def test_values_float32_cannot_hold_fall_back_to_exact_float64(tmp_path: Path) -> None:
    takes = _takes(np.random.default_rng(1))
    frames = _build(_zip(tmp_path / uv.ZIP_NAME, takes))
    pcm = takes[LEFT][1]
    data = np.asarray(frames[LEFT]["audio"].data)
    assert data.dtype == np.float64 and not frames[LEFT]["meta"]["pcm_float32_exact"]
    np.testing.assert_array_equal(data * 2.0**31, pcm.astype(np.float64))  # every int32 recovered
    assert data[2, 200] == (2**31 - 1) / 2**31
    assert np.float32(data[2, 200]) == 1.0 and np.float32(data[3, 201]) != data[3, 201]
    assert np.asarray(frames[FRONT]["audio"].data).dtype == np.float32  # decided per take


def test_geometry_is_the_published_enu_and_wgs84() -> None:
    for m, pos in zip(uv.MICROPHONES, uv.MIC_POS, strict=True):
        lat, lon = uv.recorder_wgs84(float(m["distance"]), float(m["azimuth"]))
        assert (repr(lat), repr(lon)) == (m["latitude"], m["longitude"])  # bit-exact
        assert np.hypot(pos[0], pos[1]) == pytest.approx(1.72)  # elevation is not a position
    np.testing.assert_allclose(
        uv.MIC_POS[:4],
        [[0, 1.72, 1.49], [1.72, 0, 1.49], [0, -1.72, 1.49], [-1.72, 0, 1.49]],
        atol=1e-12,
    )
    c = 1.72 / np.sqrt(2)
    np.testing.assert_allclose(
        uv.MIC_POS[4:], [[c, c, 2.38], [c, -c, 2.38], [-c, -c, 2.38], [-c, c, 2.38]], atol=1e-12
    )
    # ch2 (azimuth 90) is east of the centre: azimuth runs clockwise from north
    assert float(uv.MICROPHONES[1]["longitude"]) > float(uv.ARRAY_CENTER["center_longitude"])


def test_archive_identity_fails_closed(tmp_path: Path) -> None:
    path = _zip(tmp_path / uv.ZIP_NAME, _takes(np.random.default_rng(2)))
    blob = path.read_bytes()
    uv.verify_archive(path, len(blob), hashlib.md5(blob).hexdigest())
    with pytest.raises(ValueError, match="md5"):
        uv.verify_archive(path, len(blob), "0" * 32)
    with pytest.raises(ValueError, match="B, want"):
        uv.verify_archive(path, len(blob) + 1, hashlib.md5(blob).hexdigest())


def test_layout_labels_and_clocks_fail_closed(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    takes = _takes(rng)
    extra = _zip(tmp_path / "extra.zip", takes, {f"{uv.ARCHIVE_ROOT}/README.txt": b"x"})
    with pytest.raises(ValueError, match="unexpected ZIP member"):
        _build(extra)

    swapped = list(uv.MICROPHONES)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    bad_mics = dict(takes)
    bad_mics[FRONT] = (
        _label(
            "2024-11-15 09:36:11.641",
            300,
            {"distance": "20", "height": "20", "azimuth": "0", "rotation": "Front"},
            swapped,
        ),
        takes[FRONT][1],
    )
    with pytest.raises(ValueError, match="microphone_array"):
        _build(_zip(tmp_path / "mics.zip", bad_mics))

    overlap = dict(takes)
    overlap[LEFT] = (
        _label(
            "2024-11-15 09:36:11.900",
            300,
            {"distance": "20", "height": "10", "azimuth": "45", "rotation": "Left"},
        ),
        takes[LEFT][1],
    )
    with pytest.raises(ValueError, match="directory name|overlapping"):
        _build(_zip(tmp_path / "overlap.zip", overlap))

    with pytest.raises(ValueError, match="ambient takes"):
        _build(_zip(tmp_path / "count.zip", takes), n_drone=3, n_ambient=0)

    path = _zip(tmp_path / "seven.zip", takes)
    with zipfile.ZipFile(path) as zf:
        blobs = {i.filename: zf.read(i) for i in zf.infolist()}
    blobs[f"{uv.ARCHIVE_ROOT}/{FRONT}/{uv.WAV_NAME}"] = _wav(takes[FRONT][1][:7], channels=7)
    with zipfile.ZipFile(path, "w") as zf:
        for name, blob in blobs.items():
            zf.writestr(name, blob)
    with pytest.raises(ValueError, match="header"):
        _build(path)


def test_full_design_is_every_cell_once() -> None:
    cells = [
        (d, h, a, r)
        for d, h in uv.DESIGN["distance_height_m"]
        for a in uv.DESIGN["azimuth_deg"]
        for r in uv.DESIGN["rotation"]
    ]
    assert len(cells) == uv.N_DRONE_TAKES
    uv.check_design(cells)
    with pytest.raises(ValueError, match="design"):
        uv.check_design(cells[:-1] + cells[:1])
