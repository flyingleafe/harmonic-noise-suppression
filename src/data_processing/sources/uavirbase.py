"""UaVirBASE v1.0: a static DJI Mavic 3 Cine around an outdoor 8-microphone mast.

Zenodo record 15391924 (doi:10.5281/zenodo.15391924, concept 15391923, CC BY
4.0; Jekateryńczuk, Szadkowski, Piotrowski, Military University of Technology,
Warsaw). Paper: Appl. Sci. 15, 5378 (2025), doi:10.3390/app15105378.

The release is one file, ``Microphone_array.zip`` (13,783,664,678 B, MD5
``3c4081511630f8045fda0347cac45534``). It holds 132 directories
``Microphone_array/<YYYYMMDD_HHMMSS>/``, each with one ``output.wav`` and one
``label.json``. Each WAV is one recorder session: 8 interleaved channels, PCM
(format tag 1), 32 bits, 96 kHz. The release has 128 drone takes (40 s; one is
40.097 s) and 4 ambient-noise takes (60, 40, 256.398 and 60 s). Label clocks
put at least 8.178 s between consecutive takes, so each take is one Frame and
none are joined.

Channels stay in WAV order. Label ``microphones[c].channel == c + 1``, so the
``mic`` axis labels ``ch1..ch8`` match ``mic_pos`` and the label table. The
samples are normalized as ``int32 / 2**31``. A take is float32 only when every
int32 sample converts exactly; otherwise the take is float64. The paper's
interface records 24-bit audio; the decoder still checks every sample and does
not assume it.

The builder reads the stored ZIP members in place (no extraction, no
intermediate PCM). It decodes a whole take in blocks into one output array.
It checks the archive size and the publisher MD5, the member layout, every
RIFF header and CRC, the label schema and the channel list. It also checks the
microphone geometry against ``MICROPHONES`` and the label clocks.

Geometry (``GEOMETRY``): ``mic_pos`` is local ENU in metres. The origin is the
published array centre at ground level. ``(east, north, up) = (r sin az,
r cos az, height)`` comes from each channel's label ``distance``/``azimuth``/
``height``. The label's WGS84 coordinates match these values bit-exactly under
the recorder's spherical conversion. The label ``elevation`` is a microphone
angle and does not enter the position.

Not in the release: drone telemetry, GPS, trajectories or rotor speeds. The
drone label is a static position: distance, height, azimuth and the side
facing the array.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import IO, Any

import numpy as np
import tdseries as td

from data_processing.sources._common import meta_frame, validate_operating

DATASET = "UaVirBASE"
FRAMES_DATASET = "UaVirBASE-frames"
ZENODO_RECORD = 15391924
DOI = "10.5281/zenodo.15391924"
CONCEPT_DOI = "10.5281/zenodo.15391923"
PAPER_DOI = "10.3390/app15105378"
ZIP_NAME = "Microphone_array.zip"
ZIP_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{ZIP_NAME}/content"
ZIP_BYTES = 13_783_664_678
ZIP_MD5 = "3c4081511630f8045fda0347cac45534"
ARCHIVE_ROOT = "Microphone_array"
WAV_NAME = "output.wav"
LABEL_NAME = "label.json"

SAMPLE_RATE = 96_000
N_MICS = 8
CHANNEL_NAMES = tuple(f"ch{c}" for c in range(1, N_MICS + 1))
ENU_AXES = ("east", "north", "up")
N_DRONE_TAKES = 128
N_AMBIENT_TAKES = 4
#: Shortest label-clock gap between consecutive takes in v1.0
#: (20241115_124309 end -> 20241115_124357 start).
MIN_INTERTAKE_GAP_S = 8.178
#: Label wall-clock duration minus samples / 96 kHz: 0.082-0.115 s on 130
#: takes, +0.002 / -0.007 s on 20241115_111523 / 20241115_111723.
WALL_AUDIO_TOLERANCE_S = 0.5
#: Sphere radius of the recorder's WGS84 conversion (WGS84 equatorial radius).
RECORDER_EARTH_RADIUS_M = 6_378_137.0
#: Frames per decode block (2 MiB of PCM32 x 8).
BLOCK_FRAMES = 1 << 16

_PCM_SCALE = 2.0**-31
_HEADER_BYTES = 44

ARRAY_CENTER = {"center_latitude": "52.25532752777778", "center_longitude": "20.90053722222222"}

#: ``label.json`` ``microphone_array.microphones``: identical in all 132 takes.
MICROPHONES: tuple[dict[str, Any], ...] = tuple(
    {
        "channel": ch,
        "name": "NTG-2",
        "latitude": lat,
        "longitude": lon,
        "height": h,
        "elevation": el,
        "azimuth": az,
        "distance": "1.72",
    }
    for ch, lat, lon, h, el, az in (
        (1, "52.25534297880067", "20.90053722222222", "1.49", "10", "0.0"),
        (2, "52.25532752777778", "20.900562463062652", "1.49", "10", "90.0"),
        (3, "52.25531207675489", "20.90053722222222", "1.49", "10", "180.0"),
        (4, "52.25532752777778", "20.900511981381786", "1.49", "10", "270.0"),
        (5, "52.25533845330084", "20.90055507019165", "2.38", "20", "45.0"),
        (6, "52.25531660225472", "20.90055507019165", "2.38", "20", "135.0"),
        (7, "52.25531660225472", "20.900519374252788", "2.38", "20", "225.0"),
        (8, "52.25533845330084", "20.900519374252788", "2.38", "20", "315.0"),
    )
)


def mic_enu(mics: Sequence[Mapping[str, Any]] = MICROPHONES) -> np.ndarray:
    """``(mic, 3)`` local ENU metres from the label ``distance``/``azimuth``/``height``."""
    r = np.array([float(m["distance"]) for m in mics])
    az = np.deg2rad([float(m["azimuth"]) for m in mics])
    return np.stack(
        [r * np.sin(az), r * np.cos(az), np.array([float(m["height"]) for m in mics])], axis=1
    )


def recorder_wgs84(
    distance_m: float, azimuth_deg: float, center: Mapping[str, str] = ARRAY_CENTER
) -> tuple[float, float]:
    """The UaVirBASE Recorder's latitude/longitude for a microphone (spherical, R = 6378137 m)."""
    lat0, lon0 = float(center["center_latitude"]), float(center["center_longitude"])
    az = np.deg2rad(azimuth_deg)
    lat = lat0 + float(np.rad2deg(distance_m * np.cos(az) / RECORDER_EARTH_RADIUS_M))
    lon = lon0 + float(
        np.rad2deg(distance_m * np.sin(az) / (RECORDER_EARTH_RADIUS_M * np.cos(np.deg2rad(lat0))))
    )
    return lat, lon


MIC_POS = mic_enu()

GEOMETRY: dict[str, Any] = {
    "array": "'Acoustic Head': 8 Rode NTG-2 supercardioid shotgun microphones on a central "
    "mast, two independent levels of four arms (paper Sec. 2.1, Figs. 4-6). ch1-ch4: "
    "height 1.49 m, azimuth 0/90/180/270 deg (N/E/S/W); ch5-ch8: height 2.38 m, azimuth "
    "45/135/225/315 deg (NE/SE/SW/NW); horizontal radius 1.72 m for all",
    "mic_pos_frame": "local ENU, metres. Origin is the published array centre "
    "(label microphone_array.center_latitude/longitude, WGS84) at ground level. "
    "x is east, y is north, z is up (label height above ground). y is the array's "
    "azimuth-0 direction, set on site with a compass (paper Sec. 2.3)",
    "mic_pos_formula": "(east, north, up) = (distance sin(azimuth), distance cos(azimuth), "
    "height) from each channel's label fields; azimuth is clockwise from north "
    "(ch2 at 90 deg has the larger longitude)",
    "mic_pos_status": "hand-measured design positions as published (tape measure for distance "
    "and height, compass for azimuth; paper Sec. 2.3). Not a survey; arm and capsule "
    "offsets are not given",
    "wgs84": "the label's per-channel latitude/longitude are reproduced bit-exactly by "
    "lat0 + deg(d cos az / R), lon0 + deg(d sin az / (R cos lat0)), R = 6378137 m (the "
    "recorder's spherical conversion). On the WGS84 ellipsoid they lie within 3.6 mm "
    "of mic_pos horizontally. They inherit the compass north",
    "north": "compass north per the paper. The recorder's WGS84 coordinates treat it as "
    "geographic north; magnetic declination (~6-7 deg E at Warsaw in 2024 [INFERENCE]) is "
    "not corrected anywhere",
    "elevation": "label 'elevation' (ch1-ch4: 10 deg, ch5-ch8: 20 deg) is a microphone "
    "ANGLE, not a position. It does not enter the published coordinates (the horizontal "
    "distance stays 1.72 m), and the recorder sets it per level beside distance and "
    "height (paper Fig. 8). [INFERENCE: upward tilt of each shotgun above horizontal; "
    "Figs. 4 and 6 show upward-tilted microphones]",
    "pointing": "[INFERENCE] each supercardioid points radially outward along its arm "
    "(Fig. 4), i.e. pointing azimuth = position azimuth. Neither the paper nor the label "
    "states it",
    "channel_order": "WAV interleave order = label microphones[].channel 1..8 = mic axis "
    "labels ch1..ch8 on audio and mic_pos",
    "nominal_radius_m": 1.72,
    "drone_label": "distance = horizontal distance from the array (measuring wheel, checked "
    "with a Bosch GLM 150 C laser); height from the UAV's onboard sensors; azimuth with "
    "north = 0 deg (paper Secs. 2.1-2.2) [INFERENCE: clockwise, like the microphone "
    "azimuths]; rotation = the drone side facing the array (paper Sec. 3.1, 'On the left "
    "side, towards the acoustic array'). Static position labels set in the recorder, not "
    "telemetry",
    "evidence": "label.json microphone_array (identical in all 132 takes); Jekateryńczuk "
    "et al. 2025, doi:10.3390/app15105378, Secs. 2.1-2.3, 3.1, Figs. 4-10, Table 1",
}

CLOCK: dict[str, Any] = {
    "audio": "audio t = seconds from the first sample of the take's output.wav; "
    "the 8 channels share one interface sample clock (one interleaved WAV)",
    "label_clock": "start/end_recording_time are naive wall-clock strings (ms) written by "
    "the UaVirBASE Recorder. They bracket the session, not the first/last sample (wall "
    "duration exceeds samples/96 kHz by 0.08-0.12 s). [INFERENCE: CET, UTC+1: 106 of 132 "
    "output.wav ZIP mtimes fall within 3 s of end_recording_time - 1 h]",
    "directory": "directory name = session start to the second; 0.04-1.02 s before "
    "start_recording_time",
    "weather_clock": "weather_data.timestamp is the SenseCAP S2120 reading time with suffix "
    "+00:00. Readings come every ~302 s. Each stamp is 55.7-62.3 min ahead of the "
    "label start, so the suffix and the label clock disagree by ~1 h, and a reading can "
    "postdate the take. Kept as published; no offset applied",
}

PROVENANCE: dict[str, Any] = {
    "source_url": f"https://zenodo.org/records/{ZENODO_RECORD}",
    "doi": DOI,
    "concept_doi": CONCEPT_DOI,
    "version": "1.0 (Zenodo, published 2025-05-12)",
    "download_url": ZIP_URL,
    "license": "CC BY 4.0 (Zenodo record licence cc-by-4.0)",
    "attribution": "Jekateryńczuk, G.; Szadkowski, R.; Piotrowski, Z. (2025). UaVirBASE: A "
    "Public-Access Unmanned Aerial Vehicle Sound Source Localization Dataset (1.0). "
    "Zenodo. https://doi.org/10.5281/zenodo.15391924",
    "citation": "G. Jekateryńczuk, R. Szadkowski, Z. Piotrowski, 'UaVirBASE: A Public-Access "
    "Unmanned Aerial Vehicle Sound Source Localization Dataset', Appl. Sci. 15(10), 5378 "
    f"(2025), doi:{PAPER_DOI}",
    "paper_doi": PAPER_DOI,
    "code": "https://gitlab.com/g.jekaterynczuk/uavirbase_ssl (baseline SSL model; not used here)",
    "collection_method": "15 November 2024, one outdoor session at the array centre "
    "52.255328 N, 20.900537 E (Warsaw [INFERENCE: coordinates]). A DJI Mavic 3 Cine held "
    "static positions at horizontal distance 10/20 m, height 10/20 m and azimuth "
    "0-315 deg in 45 deg steps, with its front, back, left or right side towards the "
    "array. Each combination was recorded once for 40 s. Four ambient-noise takes "
    "contain no drone source. Weather from a co-located SenseCAP S2120 station",
    "equipment": "8 Rode NTG-2 supercardioid shotgun microphones on a self-built two-level "
    "mast; Behringer UMC1820 interface; UaVirBASE Recorder 1.0; SenseCAP S2120 8-in-1 "
    "weather sensor; Bosch GLM 150 C laser rangefinder and measuring wheel (paper Table 1). "
    "The paper's side recorder (Zoom H4essential) is not in this release",
    "observation_type": "external_mast_array",
    "sample_rate": SAMPLE_RATE,
    "channels": N_MICS,
    "archive": {
        "zip": ZIP_NAME,
        "bytes": ZIP_BYTES,
        "md5": ZIP_MD5,
        "identity": "Zenodo file checksum (md5) and size; the builder verifies both and every "
        "member CRC-32",
        "layout": f"{ARCHIVE_ROOT}/<YYYYMMDD_HHMMSS>/{{{WAV_NAME},{LABEL_NAME}}} x 132 + "
        "directory entries; members deflated",
        "wav_format": "RIFF WAVE, 44-byte header (fmt PCM tag 1, 8 ch, 96000 Hz, 32 bit), "
        "then data",
        "pcm_bytes": 17_008_112_960,
    },
    "frames": "132 Frames, one per take directory, keyed by the directory name: 128 drone "
    "takes (condition flight / hover) and 4 ambient takes (rotors_off). audio (mic, time) "
    "ch1..ch8 at 96 kHz with release lengths; float32 when every int32 converts exactly, "
    "else float64",
    "telemetry": "none: static labelled positions only; no GPS track, trajectory, IMU or "
    "rotor speed is published",
    "splits": "none published. The paper's baseline split each 40 s take in time (30 s "
    "train / 10 s test, Sec. 3.1). Split by take, never by microphone or excerpt",
    "limits": "uncalibrated full-scale audio (no microphone sensitivity, gain or SPL "
    "reference published); hand-measured nominal geometry, compass north; drone height "
    "from onboard sensors; one drone, one site, one day. Raw audit (uni-cpu job "
    "37575013): every PCM32 sample of all 132 WAVs has a zero low byte (24 effective "
    "bits), so float32 is exact everywhere; 1,056 distinct channel SHA-256s. "
    "15,336,313 samples sit exactly at a 24-bit rail (int32 -2**31 or 2**31 - 256) "
    "in 112 takes, including all 4 ambient takes. They are kept as recorded",
    "geometry": GEOMETRY,
    "clock": CLOCK,
    "description": "UaVirBASE v1.0: 128 static-hover DJI Mavic 3 Cine takes on a "
    "4 x 8 x 4 grid of distance/height, azimuth and facing side, plus 4 ambient takes. "
    "Recorded on an outdoor 8-channel NTG-2 shotgun mast array at 96 kHz PCM32, with "
    "weather data, for UAV sound-source localization.",
}

_DIR = re.compile(r"^\d{8}_\d{6}$")
_TIME = "%Y-%m-%d %H:%M:%S.%f"
_TOP_KEYS = {
    "audio_interface_name",
    "start_recording_time",
    "end_recording_time",
    "additional_info",
    "weather_data",
    "microphone_array",
    "drone",
}
_WEATHER_KEYS = {
    "air temperature",
    "air humidity",
    "light intensity",
    "barometric pressure",
    "wind direction",
    "wind speed",
    "rainfall hourly",
    "uv index",
}
_DRONE_KEYS = {
    "sound_source",
    "type",
    "movement",
    "distance",
    "height",
    "azimuth",
    "rotation",
    "additional_info",
}
_ROTATIONS = ("Front", "Back", "Left", "Right")
_DRONE_TYPE = "DJI Mavic 3 cine"
#: Full factorial of the v1.0 drone takes: (distance m, height m) x azimuth x rotation.
DESIGN = {
    "distance_height_m": ((10, 10), (10, 20), (20, 10), (20, 20)),
    "azimuth_deg": tuple(range(0, 360, 45)),
    "rotation": _ROTATIONS,
}


# ─── Archive ──────────────────────────────────────────────────────────────────


def verify_archive(path: Path, size: int = ZIP_BYTES, md5: str = ZIP_MD5) -> None:
    """Fail closed unless the release ZIP has the publisher's size and MD5."""
    actual = path.stat().st_size
    if actual != size:
        raise ValueError(f"{DATASET}: {path} is {actual} B, want {size} B")
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        while chunk := fh.read(16 << 20):
            digest.update(chunk)
    if digest.hexdigest() != md5:
        raise ValueError(f"{DATASET}: {path} md5 {digest.hexdigest()}, want {md5}")


def _take_members(zf: zipfile.ZipFile) -> dict[str, dict[str, zipfile.ZipInfo]]:
    """Directory name -> {"wav", "label"} ZipInfo; nothing else may be in the archive."""
    infos = zf.infolist()
    names = [i.filename for i in infos]
    if len(set(names)) != len(names):
        raise ValueError(f"{DATASET}: duplicate ZIP member names")
    takes: dict[str, dict[str, zipfile.ZipInfo]] = {}
    for info in infos:
        parts = info.filename.rstrip("/").split("/")
        if parts == [ARCHIVE_ROOT] and info.is_dir():
            continue
        if len(parts) == 2 and parts[0] == ARCHIVE_ROOT and _DIR.match(parts[1]) and info.is_dir():
            takes.setdefault(parts[1], {})
            continue
        if (
            len(parts) == 3
            and parts[0] == ARCHIVE_ROOT
            and _DIR.match(parts[1])
            and not info.is_dir()
        ):
            kind = {WAV_NAME: "wav", LABEL_NAME: "label"}.get(parts[2])
            if kind is not None:
                takes.setdefault(parts[1], {})[kind] = info
                continue
        raise ValueError(f"{DATASET}: unexpected ZIP member {info.filename!r}")
    bad = sorted(d for d, m in takes.items() if set(m) != {"wav", "label"})
    if bad:
        raise ValueError(
            f"{DATASET}: directories without exactly one {WAV_NAME} + {LABEL_NAME}: {bad[:5]}"
        )
    return takes


def wav_frames(header: bytes, member_bytes: int, where: str) -> int:
    """Frames of a canonical 44-byte 8-ch 96 kHz PCM32 header whose sizes match the member."""
    if len(header) != _HEADER_BYTES:
        raise ValueError(f"{DATASET}: {where} shorter than a WAV header")
    riff, riff_size, wave, fmt, fmt_size, tag, ch, sr, rate, align, bits, data, data_size = (
        struct.unpack("<4sI4s4sIHHIIHH4sI", header)
    )
    expected = (
        b"RIFF",
        member_bytes - 8,
        b"WAVE",
        b"fmt ",
        16,
        1,
        N_MICS,
        SAMPLE_RATE,
        SAMPLE_RATE * N_MICS * 4,
        N_MICS * 4,
        32,
        b"data",
        member_bytes - _HEADER_BYTES,
    )
    got = (riff, riff_size, wave, fmt, fmt_size, tag, ch, sr, rate, align, bits, data, data_size)
    if got != expected or data_size % (N_MICS * 4) or data_size == 0:
        raise ValueError(f"{DATASET}: {where} header {got}, want {expected}")
    return data_size // (N_MICS * 4)


def _read_exact(fh: IO[bytes], n: int, where: str) -> bytes:
    raw = fh.read(n)
    if len(raw) != n:
        raise ValueError(f"{DATASET}: {where} ended after {len(raw)} of {n} bytes")
    return raw


def decode_pcm32(
    fh: IO[bytes], n_frames: int, dtype: type[np.floating], where: str
) -> np.ndarray | None:
    """Interleaved little-endian int32 x 8 -> ``(mic, time)`` ``int32 / 2**31``.

    Decodes in ``BLOCK_FRAMES`` blocks into one preallocated array. With
    ``float32``, returns ``None`` at the first sample that float32 cannot hold
    exactly (more than 24 significant bits); the caller then decodes again as
    float64. Reading past the end checks that the stream is exhausted, which also
    makes ``zipfile`` verify the member CRC-32.
    """
    out = np.empty((N_MICS, n_frames), dtype=dtype)
    pos = 0
    while pos < n_frames:
        m = min(BLOCK_FRAMES, n_frames - pos)
        blk = np.frombuffer(_read_exact(fh, m * N_MICS * 4, where), dtype="<i4").reshape(m, N_MICS)
        if dtype == np.float32:
            f = blk.astype(np.float32)
            if not np.array_equal(f.astype(np.int64), blk):
                return None
            np.multiply(f.T, np.float32(_PCM_SCALE), out=out[:, pos : pos + m])
        else:
            np.multiply(blk.T, _PCM_SCALE, out=out[:, pos : pos + m])
        pos += m
    if fh.read(1):
        raise ValueError(f"{DATASET}: {where} has data past its header's length")
    return out


def decode_take(zf: zipfile.ZipFile, info: zipfile.ZipInfo, where: str) -> np.ndarray:
    """One ``output.wav`` member -> exact normalized audio (float32 if lossless, else float64)."""
    with zf.open(info) as fh:
        n = wav_frames(fh.read(_HEADER_BYTES), info.file_size, where)
        audio = decode_pcm32(fh, n, np.float32, where)
    if audio is None:
        with zf.open(info) as fh:
            fh.read(_HEADER_BYTES)
            audio = decode_pcm32(fh, n, np.float64, where)
        assert audio is not None
    return audio


# ─── Labels ───────────────────────────────────────────────────────────────────


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"{DATASET}: duplicate JSON key {k!r}")
        out[k] = v
    return out


def parse_label(raw: str, where: str) -> dict[str, Any]:
    """Strictly parse one ``label.json`` against the v1.0 schema (fails closed)."""
    lab = json.loads(raw, object_pairs_hook=_unique_keys)
    if set(lab) != _TOP_KEYS:
        raise ValueError(f"{DATASET}: {where} keys {sorted(lab)}")
    weather = lab["weather_data"]
    if (
        set(weather) != {"timestamp", "measurements"}
        or set(weather["measurements"]) != _WEATHER_KEYS
    ):
        raise ValueError(f"{DATASET}: {where} weather_data keys {sorted(weather)}")
    array = lab["microphone_array"]
    want = {**ARRAY_CENTER, "microphones": [dict(m) for m in MICROPHONES]}
    if array != want:
        raise ValueError(f"{DATASET}: {where} microphone_array differs from MICROPHONES")
    if [m["channel"] for m in array["microphones"]] != list(range(1, N_MICS + 1)):
        raise ValueError(f"{DATASET}: {where} microphone channels out of order")
    drone = lab["drone"]
    if set(drone) != _DRONE_KEYS:
        raise ValueError(f"{DATASET}: {where} drone keys {sorted(drone)}")
    if drone["sound_source"] == "Drone":
        ok = (
            drone["type"] == _DRONE_TYPE
            and drone["movement"] == "Static"
            and all(
                isinstance(drone[k], str) and drone[k].isdigit()
                for k in ("distance", "height", "azimuth")
            )
            and drone["rotation"] in _ROTATIONS
        )
    elif drone["sound_source"] == "Ambient Noise":
        ok = all(drone[k] is None for k in _DRONE_KEYS - {"sound_source", "additional_info"})
    else:
        ok = False
    if not ok:
        raise ValueError(f"{DATASET}: {where} unexpected drone label {drone}")
    return lab


def _times(lab: Mapping[str, Any]) -> tuple[datetime, datetime]:
    return (
        datetime.strptime(lab["start_recording_time"], _TIME),
        datetime.strptime(lab["end_recording_time"], _TIME),
    )


def check_design(positions: Sequence[tuple[int, int, int, str]]) -> None:
    """Drone takes must cover ``DESIGN`` exactly once each (distance, height, azimuth, rotation)."""
    want = sorted(
        (d, h, a, r)
        for d, h in DESIGN["distance_height_m"]
        for a in DESIGN["azimuth_deg"]
        for r in DESIGN["rotation"]
    )
    if sorted(positions) != want:
        raise ValueError(
            f"{DATASET}: drone takes do not cover the {len(want)}-cell design exactly once"
        )


def _measurement(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        number, _, unit = value.partition(" ")
        return {"value": float(number), "unit": unit}
    return {"value": value, "unit": "UV index (dimensionless)"}


# ─── Build ────────────────────────────────────────────────────────────────────


def build(root: Path) -> Iterator[tuple[str, td.Frame]]:
    """Raw dir with the verified ``Microphone_array.zip`` -> one Frame per take directory."""
    zip_path = Path(root) / ZIP_NAME
    verify_archive(zip_path)
    yield from build_from_zip(
        zip_path, n_drone=N_DRONE_TAKES, n_ambient=N_AMBIENT_TAKES, full_design=True
    )


def build_from_zip(
    zip_path: Path, *, n_drone: int, n_ambient: int, full_design: bool
) -> Iterator[tuple[str, td.Frame]]:
    """Validate every label and the take chronology first, then decode take by take."""
    with zipfile.ZipFile(zip_path) as zf:
        members = _take_members(zf)
        raws = {d: zf.read(m["label"]).decode("utf-8") for d, m in members.items()}
        labels = {d: parse_label(raw, f"{d}/{LABEL_NAME}") for d, raw in raws.items()}
        drone = [d for d, lab in labels.items() if lab["drone"]["sound_source"] == "Drone"]
        if (len(drone), len(labels) - len(drone)) != (n_drone, n_ambient):
            raise ValueError(
                f"{DATASET}: {len(drone)} drone + {len(labels) - len(drone)} ambient takes, "
                f"want {n_drone} + {n_ambient}"
            )
        if full_design:
            check_design(
                [
                    (
                        int(labels[d]["drone"]["distance"]),
                        int(labels[d]["drone"]["height"]),
                        int(labels[d]["drone"]["azimuth"]),
                        labels[d]["drone"]["rotation"],
                    )
                    for d in drone
                ]
            )
        order = sorted(labels)
        clocks = {d: _times(labels[d]) for d in order}
        for d in order:
            start, end = clocks[d]
            lead = (start - datetime.strptime(d, "%Y%m%d_%H%M%S")).total_seconds()
            if not (start < end and 0.0 <= lead < 2.0):
                raise ValueError(f"{DATASET}: {d} label clock {start}..{end} vs directory name")
        gaps = [(clocks[b][0] - clocks[a][1]).total_seconds() for a, b in zip(order, order[1:])]
        if any(g <= 0 for g in gaps):
            raise ValueError(
                f"{DATASET}: overlapping or unordered takes: {[(order[i], g) for i, g in enumerate(gaps) if g <= 0]}"
            )

        for i, d in enumerate(order):
            where = f"{d}/{WAV_NAME}"
            audio = decode_take(zf, members[d]["wav"], where)
            n = audio.shape[1]
            start, end = clocks[d]
            wall = (end - start).total_seconds()
            if abs(wall - n / SAMPLE_RATE) > WALL_AUDIO_TOLERANCE_S:
                raise ValueError(
                    f"{DATASET}: {d} label duration {wall} s vs {n / SAMPLE_RATE} s of audio"
                )
            clock = {
                "start_recording_time": labels[d]["start_recording_time"],
                "end_recording_time": labels[d]["end_recording_time"],
                "label_duration_s": wall,
                "label_minus_audio_s": wall - n / SAMPLE_RATE,
                "directory_lead_s": (start - datetime.strptime(d, "%Y%m%d_%H%M%S")).total_seconds(),
                "take_index": i,
                "previous_take": order[i - 1] if i else None,
                "gap_from_previous_s": gaps[i - 1] if i else None,
                "next_take": order[i + 1] if i + 1 < len(order) else None,
                "gap_to_next_s": gaps[i] if i < len(gaps) else None,
                "weather_timestamp": labels[d]["weather_data"]["timestamp"],
                "weather_minus_label_start_s": (
                    datetime.fromisoformat(labels[d]["weather_data"]["timestamp"]).replace(
                        tzinfo=None
                    )
                    - start
                ).total_seconds(),
                **CLOCK,
            }
            frame = _take_frame(d, audio, labels[d], raws[d], members[d], clock)
            del audio
            yield d, frame
            del frame


def _mic_table() -> list[dict[str, Any]]:
    rows = []
    for name, m, pos in zip(CHANNEL_NAMES, MICROPHONES, MIC_POS, strict=True):
        rows.append(
            {
                "mic": name,
                "channel": m["channel"],
                "model": "Rode NTG-2 supercardioid shotgun (label name 'NTG-2')",
                "level": "lower" if m["channel"] <= 4 else "upper",
                "raw": {
                    k: m[k]
                    for k in (
                        "name",
                        "latitude",
                        "longitude",
                        "height",
                        "elevation",
                        "azimuth",
                        "distance",
                    )
                },
                "latitude_deg": float(m["latitude"]),
                "longitude_deg": float(m["longitude"]),
                "height_above_ground_m": float(m["height"]),
                "position_azimuth_deg_cw_from_north": float(m["azimuth"]),
                "horizontal_radius_m": float(m["distance"]),
                "label_elevation_deg": float(m["elevation"]),
                "label_elevation_meaning": "microphone angle, not position [INFERENCE: upward tilt]",
                "pointing_azimuth_deg_inferred": float(m["azimuth"]),
                "enu_m": [float(v) for v in pos],
            }
        )
    return rows


def _take_frame(
    d: str,
    audio: np.ndarray,
    lab: dict[str, Any],
    raw_label: str,
    members: Mapping[str, zipfile.ZipInfo],
    clock: dict[str, Any],
) -> td.Frame:
    n = audio.shape[1]
    dr = lab["drone"]
    is_drone = dr["sound_source"] == "Drone"
    where = f"{DATASET} {d}"
    if is_drone:
        dist, height, az, rot = (
            int(dr["distance"]),
            int(dr["height"]),
            int(dr["azimuth"]),
            dr["rotation"],
        )
        system = {
            "category": "drone",
            "make": "DJI",
            "model": "Mavic 3 Cine",
            "n_rotors": 4,
            "label_type": dr["type"],
        }
        operating = {
            "condition": "flight",
            "flight_mode": "hover",
            "movement_label": dr["movement"],
            "n_active_rotors": 4,
            "contains_rotor_noise": True,
            "external_source": "none",
            "confidence": "high",
        }
        validate_operating(operating, 4, where)
        label = {
            "subcategory": f"mavic3cine_static_dist{dist}m_height{height}m",
            "content": "drone_static_hover_external_array_recording",
            "recording_kind": "source_recording",
            "sound_source": dr["sound_source"],
            "distance_m": float(dist),
            "height_m": float(height),
            "azimuth_deg": float(az),
            "rotation": rot,
            "side_facing_array": rot.lower(),
            "position_semantics": GEOMETRY["drone_label"],
        }
        source_motion = "hovering at a labelled static position"
    else:
        system = {
            "category": "ambient",
            "description": "no drone source (publisher label 'Ambient Noise')",
        }
        operating = {
            "condition": "rotors_off",
            "n_active_rotors": 0,
            "contains_rotor_noise": False,
            "external_source": "none",
            "confidence": "medium",
            "note": "publisher label 'Ambient Noise' with no drone fields; the drone's state "
            "during these takes is not otherwise documented",
        }
        validate_operating(operating, 0, where)
        label = {
            "subcategory": "ambient_noise",
            "content": "outdoor_ambient_external_array_recording",
            "recording_kind": "ambient_background",
            "sound_source": dr["sound_source"],
        }
        source_motion = "none"
    weather = lab["weather_data"]
    meta = meta_frame(
        d,
        FRAMES_DATASET,
        system=system,
        observation={
            "mic_placement": "external",
            "type": "ground_mast_array_static",
            "array": "8 Rode NTG-2 shotguns on a two-level mast, radius 1.72 m",
            "recorder": lab["audio_interface_name"],
            "source_motion": source_motion,
            "relative_trajectory": "none (static labelled position; no telemetry)",
            "environment": "outdoor, Warsaw, Poland [INFERENCE: array centre coordinates]",
            "date": "2024-11-15",
            "n_channels": N_MICS,
            "sample_rate": SAMPLE_RATE,
            "sample_format": f"{audio.dtype.name} = int32 / 2**31 (exact)",
        },
        operating=operating,
        label=label,
        extra={
            "source": {
                "archive": f"Zenodo {ZENODO_RECORD} v1.0 {ZIP_NAME} (md5 {ZIP_MD5})",
                "directory": f"{ARCHIVE_ROOT}/{d}/",
                "wav_member": members["wav"].filename,
                "wav_bytes": members["wav"].file_size,
                "wav_crc32": f"{members['wav'].CRC:08x}",
                "label_member": members["label"].filename,
                "label_bytes": members["label"].file_size,
                "label_crc32": f"{members['label'].CRC:08x}",
                "grouping": "one take = one directory = one 8-channel WAV; never joined",
            },
            "label_json": raw_label,
            "audio_interface_name": lab["audio_interface_name"],
            "additional_info": lab["additional_info"],
            "drone_label_raw": dict(dr),
            "weather": {
                "timestamp": weather["timestamp"],
                "measurements_raw": dict(weather["measurements"]),
                "measurements": {k: _measurement(v) for k, v in weather["measurements"].items()},
                "station": "SenseCAP S2120 8-in-1 weather sensor at the recording site",
                "clock": CLOCK["weather_clock"],
            },
            "clock": clock,
            "array_center_raw": dict(ARRAY_CENTER),
            "microphones": _mic_table(),
            "channel_names": list(CHANNEL_NAMES),
            "channel_valid_range": [[0, n]] * N_MICS,
            "native_format": "RIFF WAVE PCM (tag 1), 8 interleaved channels, 32-bit, 96000 Hz",
            "audio_dtype": audio.dtype.name,
            "pcm_float32_exact": audio.dtype == np.float32,
            "units": "normalized full scale (int32 / 2**31); no acoustic calibration published",
            "recording_n_samples": n,
            "recording_duration_s": n / SAMPLE_RATE,
            "geometry": GEOMETRY,
            "doi": DOI,
            "paper_doi": PAPER_DOI,
            "attribution": PROVENANCE["attribution"],
            "license": PROVENANCE["license"],
        },
    )
    mics = td.LabelIndex(CHANNEL_NAMES)
    return td.Frame(
        {
            "audio": td.Series(
                audio, ("mic", "time"), {"mic": mics, "time": td.GridIndex(SAMPLE_RATE, n)}
            ),
            "mic_pos": td.wrap(
                MIC_POS.copy(), ("mic", "enu"), {"mic": mics, "enu": td.LabelIndex(ENU_AXES)}
            ),
            "meta": meta,
        }
    )
