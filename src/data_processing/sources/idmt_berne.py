"""Fraunhofer IDMT Berne drone-flight array recordings (Zenodo 17525811).

Real outdoor flights of three drones (DJI Phantom 4, DJI Mavic, Align) on
2022-08-03 in Berne, Lower Saxony, recorded by one static 8-microphone 15 cm
cubic ground array, published only as 30 s segments
(``MicArrayData30s.zip``): per segment one 8-channel 48 kHz 16-bit WAV and a
``DOATruePos.csv`` (``time,X,Y,Z,ele,azi`` at 100 rows/s) of the drone
position relative to the array, from GPS-Logger 3 units on drone and array
(10 Hz, +-2.5 m), with Z = 0 (the publisher's equal-altitude assumption).

The publisher's 30 s segments are rejoined into maximal contiguous RUNS, one
Frame each, keyed ``<flight_id>/<part>_<first segment>-<part>_<last
segment>``. Two adjacent segments of a flight join only when every test
holds: same publisher part, consecutive segment index, ``clean_audio_seam``
passes on all 8 channels, and the GPS position across the boundary moves no
more than the trajectory itself moves row-to-row next to it
(:func:`gps_continuous`). The part boundaries fail (GPS jumps of 8-166 m), so
parts stay separate runs. A run holds ``audio`` ``(mic, time)`` (every sample
of every segment, back to back, on the run clock, ``t_start`` = 0), the CSV
rows of each segment as ``drone_pos`` ``(axis, time)`` (m) and ``drone_doa``
``(angle, time)`` (deg) on a ``StampIndex`` = segment start (its first audio
sample on the run clock) + the CSV ``time``, the array geometry ``mic_pos``
``(mic, 3)``, and ``meta.source.segments`` / ``meta.source.boundaries``: the
segment map and the evidence for every boundary. Every CSV value is kept;
nothing is resampled or shifted.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import tdseries as td

from data_processing.sources._common import meta_frame

DATASET = "IDMT-Berne-frames"
ZENODO_RECORD = "17525811"
ZIP_NAME = "MicArrayData30s.zip"
ZIP_SIZE = 1_515_112_211
ZIP_MD5 = "fd8cfda2d852e351b2d33e1ae9a5aba4"
SAMPLE_RATE = 48_000
N_MICS = 8
SEGMENT_S = 30
#: ``clean_audio_seam`` looks at <= 4096 samples either side of a boundary.
SEAM_WINDOW = 4096
#: GPS continuity: rows either side of a boundary whose own row-to-row steps
#: bound the step across it (> the longest publisher hold, 192 rows).
GPS_CONTEXT_ROWS = 200

#: Array-local microphone coordinates (m), mic 1..8, Yang et al. EUSIPCO 2025
#: Fig. 1(a) "Microphone array coordinates"; identical to
#: SOUNDS-RESEARCH/DroneSynthesis2DOA@da80f4a ``mic_array/Reusev301.yaml``.
MIC_POS = np.array(
    [
        [0.05625, 0.0375, 0.15],
        [0.0, 0.09, 0.15],
        [0.01875, 0.15, 0.1145],
        [0.05625, 0.15, 0.009],
        [0.15, 0.15, 0.075],
        [0.13125, 0.075, 0.009],
        [0.15, 0.01875, 0.15],
        [0.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

GEOMETRY = {
    "array": "Reusev301: 8 microphones on a 15 cm cubic frame, static on the ground site",
    "mic_pos_frame": "array-local Cartesian, metres, origin at mic 8 (a cube corner)",
    "mic_pos_source": "Yang, Naylor, Doclo, Bitzer, EUSIPCO 2025, Fig. 1(a) table; same "
    "values in github.com/SOUNDS-RESEARCH/DroneSynthesis2DOA@da80f4a22c35 "
    "mic_array/Reusev301.yaml",
    "mic_pos_status": "published nominal coordinates (as printed, 0.1 mm resolution); "
    "not a calibration measurement",
    "channel_map": "ASSUMED: WAV channel i = published mic i+1 (no channel map is published)",
    "local_to_world": "UNKNOWN. drone_pos/azimuth use x=north, z=up, y by the right-hand "
    "rule (Zenodo description); the array-local mic frame's orientation to north is not "
    "published. The upstream simulation config for this array sets centre [0,0,3] m and "
    "rotation_angles_xyz [0,0,157.5] deg; whether that reflects the field setup is unverified.",
}

TRAJECTORY = {
    "source": "DOATruePos.csv in each segment directory (publisher-computed)",
    "columns": ["time", "X", "Y", "Z", "ele", "azi"],
    "drone_pos": "X, Y, Z of the drone relative to the array, metres; x north, z up, "
    "y = right-hand rule (west)",
    "drone_doa": "ele, azi, degrees as published; ele is 90 throughout (horizontal), "
    "azi = atan2(Y, X) in [0, 360)",
    "measurement": "GPS-Logger 3 (SM Modellbau) on drone and array, 10 Hz, +-2.5 m; "
    "distance by haversine assuming equal altitude, so Z = 0 and elevation is not observed",
    "rate": "100 rows/s in the CSV. Over all 163 segments, 278,594 of 282,665 position "
    "changes come one row (10 ms) after the previous change: the rows are publisher-"
    "interpolated, not 100 Hz measurements; the rest are holds of 10-192 rows "
    "(mostly multiples of 10, i.e. whole 10 Hz fixes) where the position does not move",
    "clock": "CSV time column = seconds from the segment's first audio sample, as "
    "aligned by the publisher; the GPS-to-audio alignment method and accuracy are not "
    "published",
}

PROVENANCE: dict[str, Any] = {
    "source_url": f"https://zenodo.org/records/{ZENODO_RECORD}",
    "doi": "10.5281/zenodo.17525811",
    "concept_doi": "10.5281/zenodo.17525810",
    "license": "CC-BY-4.0",
    "license_scope": "Zenodo record licence of the dataset files themselves (open access)",
    "attribution": "Yang, Ximei; Bitzer, Joerg. Real-World Drone Audio Dataset Collected by "
    "Fraunhofer IDMT (August 2022). Zenodo, 2025-11-04. doi:10.5281/zenodo.17525811",
    "citation": "X. Yang, P. A. Naylor, S. Doclo, J. Bitzer, Neural Drone Localization "
    "Exploiting Signal Synthesis of Real-World Audio Data, EUSIPCO 2025",
    "paper_url": "https://eurasip.org/Proceedings/Eusipco/Eusipco2025/pdfs/0000256.pdf",
    "file": {"name": ZIP_NAME, "size": ZIP_SIZE, "md5": ZIP_MD5},
    "collection_method": "static outdoor ground array, drones flying static/linear/random "
    "patterns within ~200 m (max 430 m)",
    "equipment": "8-mic 15 cm cubic array; GPS-Logger 3 on drone and array",
    "sample_rate": SAMPLE_RATE,
    "channels": N_MICS,
    "description": "8 flights (Phantom 4 static x2 / linear x2 / random, Mavic static x2, "
    "Align random), 163 publisher 30 s segments (81.5 min). The publisher states ~106 min "
    "for the 8 selected flights; only the 30 s segments are distributed. The "
    "DroneSynthesis2DOA README calls this set 'preprocessed' without saying how. "
    "Rejoined into 12 contiguous runs (151 segment boundaries joined, 4 part "
    "boundaries kept apart); see the module docstring.",
    "geometry": GEOMETRY,
    "trajectory": TRAJECTORY,
    "archive": {
        "repository": "Zenodo",
        "record_id": int(ZENODO_RECORD),
        "concept_record_id": 17525810,
        "title": "Real-World Drone Audio Dataset Collected by Fraunhofer IDMT (August 2022)",
        "creators": ["Yang, Ximei", "Bitzer, Joerg"],
        "publisher_affiliation": "Fraunhofer IDMT (Oldenburg branch HSA)",
        "publication_date": "2025-11-04",
        "record_created": "2025-11-04T19:13:00.584018+00:00",
        "version": None,
        "versions_in_concept": 1,
        "access_right": "open",
        "keywords": [
            "drone audio dataset",
            "UAV audio",
            "microphone array",
            "flight recordings",
            "azimuth",
            "Haversine formula",
            "GPS tracking",
        ],
        "zip_entries": "1080 central-directory entries: 163 segment dirs each with "
        "<segment>.wav + DOATruePos.csv (326 files), 50 .DS_Store, and macOS __MACOSX/ "
        "resource forks (ignored); every member deflate-compressed",
        "wav_format": "RIFF WAVE PCM 16-bit, 8 ch, 48 kHz, 1,440,000 frames per segment "
        "(one segment: 1,440,001)",
        "csv_format": "header time,X,Y,Z,ele,azi; 3000 rows, time 0.00..29.99 s step 0.01",
        "fetched_from": f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{ZIP_NAME}/content",
    },
}


def _digest(path: Path) -> dict[str, Any]:
    """Size, sha256 and zip-comparable CRC-32 of one extracted member."""
    import hashlib
    import zlib

    sha, crc = hashlib.sha256(), 0
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            sha.update(block)
            crc = zlib.crc32(block, crc)
    return {"size": path.stat().st_size, "sha256": sha.hexdigest(), "crc32": f"{crc:08x}"}


_TAIL = re.compile(r"^(?P<flight>.+)_(?P<part>\d+)_(?P<segment>\d+)$")
_FLIGHT = re.compile(
    r"^(?P<start>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})_"
    r"(?:reusev301\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}_|_)?"
    r"Dronen_records_(?P<drone>Phantom4|Mavic|Align)[-_]"
    r"(?P<pattern>static|linear|random)[-_]flights?(?:[-_](?P<run>\d+))?$"
)
_DRONES = {
    "Phantom4": {"make": "DJI", "model": "Phantom 4", "n_rotors": 4},
    "Mavic": {"make": "DJI", "model": "Mavic (variant not stated)", "n_rotors": 4},
    "Align": {"make": "Align", "model": "not stated"},
}
_FLIGHT_MODE = {"static": "hover", "linear": "manoeuvre", "random": "manoeuvre"}


def parse_segment(name: str) -> dict[str, Any]:
    """Segment directory name -> flight/drone/pattern/part/segment tokens."""
    tail = _TAIL.match(name)
    flight = _FLIGHT.match(tail.group("flight")) if tail else None
    if tail is None or flight is None:
        raise ValueError(f"{DATASET}: unparsed segment name {name!r}")
    return {
        "flight_id": tail.group("flight"),
        "part": int(tail.group("part")),
        "segment": int(tail.group("segment")),
        "start_token": flight.group("start"),
        "drone_token": flight.group("drone"),
        "pattern": flight.group("pattern"),
        "run": None if flight.group("run") is None else int(flight.group("run")),
    }


def _read_doa(path: Path) -> np.ndarray:
    """``DOATruePos.csv`` -> ``(rows, 6)`` float64, header checked."""
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        if header != TRAJECTORY["columns"]:
            raise ValueError(f"{path}: unexpected columns {header}")
        rows = np.loadtxt(fh, delimiter=",", dtype=np.float64, ndmin=2)
    if rows.shape[1] != 6 or not np.all(np.diff(rows[:, 0]) > 0):
        raise ValueError(f"{path}: expected 6 columns with increasing time")
    return rows


def _segment_dirs(root: Path) -> list[Path]:
    dirs = sorted(p.parent for p in Path(root).rglob("DOATruePos.csv") if "__MACOSX" not in p.parts)
    names = [d.name for d in dirs]
    if not dirs:
        raise FileNotFoundError(f"{DATASET}: no DOATruePos.csv under {root}")
    if len(set(names)) != len(names):
        raise ValueError(f"{DATASET}: duplicate segment names under {root}")
    return dirs


def gps_continuous(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """GPS test for the boundary between two adjacent segments' CSV rows.

    The horizontal step from the left segment's last row to the right one's
    first row must not exceed the largest row-to-row step within
    :data:`GPS_CONTEXT_ROWS` rows on either side (longer than the longest
    publisher hold, 192 rows), i.e. the boundary may not move the drone more
    than its own published trajectory moves next to it. Where the track is
    interpolated the boundary step typically EQUALS the neighbouring step, so
    the comparison allows a relative 1e-9 for float rounding.
    """
    a = left[-GPS_CONTEXT_ROWS:, 1:3]
    b = right[:GPS_CONTEXT_ROWS, 1:3]
    steps = np.concatenate(
        (np.linalg.norm(np.diff(a, axis=0), axis=1), np.linalg.norm(np.diff(b, axis=0), axis=1))
    )
    bound = float(steps.max()) if steps.size else 0.0
    jump = float(np.linalg.norm(b[0] - a[-1]))
    return {
        "gps_continuous": jump <= bound * (1 + 1e-9),
        "gps_step_m": jump,
        "gps_local_max_step_m": bound,
    }


def _scan_segment(root: Path, seg_dir: Path) -> dict[str, Any]:
    """One segment's identity, sizes, CSV rows and boundary windows (no full read)."""
    import soundfile as sf

    name = seg_dir.name
    wav = seg_dir / f"{name}.wav"
    csv = seg_dir / "DOATruePos.csv"
    info = sf.info(str(wav))
    if (info.samplerate, info.channels, info.subtype) != (SAMPLE_RATE, N_MICS, "PCM_16"):
        raise ValueError(f"{wav}: {info.samplerate} Hz x {info.channels} ch {info.subtype}")
    n = int(info.frames)
    width = min(SEAM_WINDOW, n)
    head, _ = sf.read(str(wav), start=0, stop=width, dtype="float32", always_2d=True)
    tail, _ = sf.read(str(wav), start=n - width, stop=n, dtype="float32", always_2d=True)
    return {
        "name": name,
        "tokens": parse_segment(name),
        "wav": wav,
        "n_samples": n,
        "doa": _read_doa(csv),
        "head": np.ascontiguousarray(head.T),
        "tail": np.ascontiguousarray(tail.T),
        "files": {
            "wav": {"relpath": str(wav.relative_to(root)), **_digest(wav)},
            "csv": {"relpath": str(csv.relative_to(root)), **_digest(csv)},
        },
    }


def _boundary(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Every join test for two adjacent segments of one flight, with the verdict."""
    from data_processing.sources._common import clean_audio_seam

    lt, rt = left["tokens"], right["tokens"]
    same_part = lt["part"] == rt["part"]
    consecutive = same_part and rt["segment"] == lt["segment"] + 1
    seam = clean_audio_seam(left["tail"], right["head"], SAMPLE_RATE)
    gps = gps_continuous(left["doa"], right["doa"])
    failed = [
        test
        for test, ok in (
            ("same_part", same_part),
            ("consecutive_segment", consecutive),
            ("clean_audio_seam", seam),
            ("gps_continuous", gps["gps_continuous"]),
        )
        if not ok
    ]
    return {
        "left": left["name"],
        "right": right["name"],
        "same_part": same_part,
        "consecutive_segment": consecutive,
        "clean_audio_seam": seam,
        **gps,
        "joined": not failed,
        "failed_tests": failed,
    }


def _runs(
    root: Path,
) -> Iterator[tuple[list[dict[str, Any]], list[dict[str, Any]], dict | None, dict | None]]:
    """Per flight, in publisher order: ``(segments, inner boundaries, boundary
    before, boundary after)`` of every maximal contiguous run."""
    flights: dict[str, list[Path]] = {}
    for seg_dir in _segment_dirs(root):
        tokens = parse_segment(seg_dir.name)
        flights.setdefault(tokens["flight_id"], []).append(seg_dir)
    for flight_id in sorted(flights):
        dirs = sorted(
            flights[flight_id],
            key=lambda d: (parse_segment(d.name)["part"], parse_segment(d.name)["segment"]),
        )
        segs = [_scan_segment(root, d) for d in dirs]
        bounds = [_boundary(a, b) for a, b in zip(segs, segs[1:])]
        start = 0
        for i in range(len(segs)):
            if i == len(segs) - 1 or not bounds[i]["joined"]:
                yield (
                    segs[start : i + 1],
                    bounds[start:i],
                    bounds[start - 1] if start > 0 else None,
                    bounds[i] if i < len(bounds) else None,
                )
                start = i + 1


def _run_key(segs: list[dict[str, Any]]) -> str:
    first, last = segs[0]["tokens"], segs[-1]["tokens"]
    return (
        f"{first['flight_id']}/{first['part']}_{first['segment']}-{last['part']}_{last['segment']}"
    )


def build(root: Path) -> Iterator[tuple[str, td.Frame]]:
    """Yield one ``(<flight_id>/<part>_<seg>-<part>_<seg>, Frame)`` per contiguous run."""
    import soundfile as sf

    root = Path(root)
    for segs, inner, before, after in _runs(root):
        key = _run_key(segs)
        tokens = segs[0]["tokens"]
        n_total = sum(s["n_samples"] for s in segs)
        audio = np.empty((N_MICS, n_total), dtype=np.float32)
        stamps, rows, seg_map = [], [], []
        offset = 0
        for s in segs:
            n = s["n_samples"]
            with sf.SoundFile(str(s["wav"])) as fh:
                data = fh.read(n, dtype="float32", always_2d=True)
            if data.shape != (n, N_MICS):
                raise ValueError(f"{s['wav']}: read {data.shape}, header says {n} x {N_MICS}")
            audio[:, offset : offset + n] = data.T
            del data
            t0 = offset / SAMPLE_RATE
            stamps.append(t0 + s["doa"][:, 0])
            rows.append(s["doa"][:, 1:])
            seg_map.append(
                {
                    "segment": s["name"],
                    "part": s["tokens"]["part"],
                    "index": s["tokens"]["segment"],
                    "start_sample": offset,
                    "n_samples": n,
                    "start_s": t0,
                    "csv_rows": int(s["doa"].shape[0]),
                    "files": s["files"],
                }
            )
            offset += n
        t = np.concatenate(stamps)
        doa = np.concatenate(rows)
        if not np.all(np.diff(t) > 0) or t[-1] >= n_total / SAMPLE_RATE:
            raise ValueError(f"{DATASET} {key}: trajectory stamps not increasing inside the run")
        time = td.StampIndex.from_stamps(t, t_start=0.0, t_end=n_total / SAMPLE_RATE)
        drone = _DRONES[tokens["drone_token"]]
        meta = meta_frame(
            key,
            DATASET,
            system={
                "category": "drone",
                "make": drone["make"],
                "model": drone["model"],
                "n_rotors": drone.get("n_rotors"),
                "drone_token": tokens["drone_token"],
            },
            observation={
                "mic_placement": "external",
                "type": "ground_array_static",
                "source_motion": "static" if tokens["pattern"] == "static" else "moving",
                "relative_trajectory": "gps_horizontal_position",
                "environment": "outdoor field site, Berne (Lower Saxony, DE)",
                "n_channels": N_MICS,
                "sample_rate": SAMPLE_RATE,
                "sample_format": "PCM_16 (held as float32, exact)",
            },
            operating={
                "condition": "flight",
                "flight_mode": _FLIGHT_MODE[tokens["pattern"]],
                "flight_pattern": tokens["pattern"],
                "contains_rotor_noise": True,
                "external_source": "unknown",
            },
            label={
                "subcategory": f"{tokens['drone_token'].lower()}_{tokens['pattern']}",
                "content": "drone_flight_array_recording",
                "recording_kind": "source_recording",
            },
            extra={
                "source": {
                    "flight_id": tokens["flight_id"],
                    "flight_start_token": tokens["start_token"],
                    "flight_start_timezone": "not stated",
                    "flight_run_number": tokens["run"],
                    "token_meaning": "trailing <part>_<segment> of the publisher's segment "
                    "name; segment = 30 s index within the part (contiguous from 0); parts "
                    "are not explained by the publisher and are not contiguous with each other",
                    "segments": seg_map,
                    "boundaries": inner,
                    "boundary_before": before,
                    "boundary_after": after,
                    "join_rule": "same part, consecutive segment index, "
                    "_common.clean_audio_seam on all 8 channels, and gps_continuous "
                    f"(step across <= largest row step within {GPS_CONTEXT_ROWS} rows either "
                    "side); the segments carry no absolute timestamps, so a join rests on "
                    "publisher order plus these signal tests",
                    "archive": f"Zenodo {ZENODO_RECORD} {ZIP_NAME} (md5 {ZIP_MD5})",
                    "segmentation": "publisher 30 s segments of the flight recording, "
                    "rejoined into this contiguous run",
                },
                "clock": "audio t = seconds from the first sample of the run's first segment; "
                "segments sit back to back at their start_sample; trajectory t = segment "
                "start_s + the segment CSV time",
                "recording_n_samples": n_total,
                "recording_duration_s": n_total / SAMPLE_RATE,
                "geometry": GEOMETRY,
                "trajectory": {**TRAJECTORY, "n_rows": int(t.size)},
                "attribution": PROVENANCE["attribution"],
                "license": PROVENANCE["license"],
            },
        )
        frame = td.Frame(
            {
                "audio": td.uniform(audio, SAMPLE_RATE, dims=("mic", "time"), t_start=0.0),
                "mic_pos": td.wrap(MIC_POS.copy(), dims=("mic", None)),
                "drone_pos": td.Series(
                    np.ascontiguousarray(doa[:, 0:3].T),
                    ("axis", "time"),
                    {"time": time, "axis": td.LabelIndex(("X", "Y", "Z"))},
                ),
                "drone_doa": td.Series(
                    np.ascontiguousarray(doa[:, 3:5].T),
                    ("angle", "time"),
                    {"time": time, "angle": td.LabelIndex(("ele", "azi"))},
                ),
                "meta": meta,
            }
        )
        yield key, frame
        del audio, frame
