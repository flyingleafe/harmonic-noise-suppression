"""AIRA-UAS source: 8-channel onboard-array recordings of a DJI Matrice 100.

Publisher: C. Rascon / IIMAS-UNAM, part of the AIRA corpus
(https://calebrascon.info/AIRA-UAS/, https://aira.iimas.unam.mx/). The AIRA
landing page's schema.org ``Dataset`` record licenses the whole corpus and its
"UAS 8 (Unmanned Aerial System)" part as CC0-1.0.

The only distribution is one Dropbox ``UAS.tar.gz`` (pinned here by size and
sha256): ``UAS/Protocol{1,2,3}/Recording{1..21}/wav_ch_{1..8}.wav`` — one mono
48 kHz 32-bit integer PCM file per microphone, written by the 8SoundsUSB
interface on a Raspberry Pi 2 carried by the M100 with an unfinalised header
(RIFF/data size fields left 8/0; the samples run to end of file) — plus one
``RecordingInfo.xlsx`` per protocol (the per-recording settings). 21
recordings, 168 WAVs, no licence/README file, no telemetry, no flight log, no
published microphone coordinates.

What each recording IS lives in the xlsx; it is restated ONCE in
:data:`RECORDINGS` (with the project vocabulary: ``condition``/``flight_mode``,
``subcategory``) and the builder REFUSES a recording whose xlsx row disagrees
with that table, or a recording/row without its counterpart.

Each Frame is one whole recording, keyed ``Protocol<p>/Recording<n>``: the 8
per-microphone files of one recording, written by one 8-channel interface, are
combined sample-for-sample into ``audio`` ``(mic, time)`` at 48 kHz on the
recording's own clock (``t_start`` = 0, seconds from the first sample),
``mic_pos`` ``(mic, 3)`` and ``meta``. Row i of ``audio`` and
of ``mic_pos`` = ``wav_ch_{i+1}.wav``. No publication gives coordinates (the
ICUAS 2018 paper is closed access); ``mic_pos`` is an ESTIMATE from acoustic
self-calibration on the recordings themselves (see :data:`GEOMETRY`).
"""

from __future__ import annotations

import hashlib
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import tdseries as td

from data_processing.downloaders import http_fetch
from data_processing.sources._common import meta_frame

DATASET = "AIRA-UAS"
PAGE_URL = "https://calebrascon.info/AIRA-UAS/"
URL = "https://www.dropbox.com/s/rm2j1in47crs6sk/UAS.tar.gz?dl=1"
ARCHIVE = "UAS.tar.gz"
ARCHIVE_BYTES = 688_854_322
ARCHIVE_SHA256 = "67218177b3895ba0566b2f970c86cef2b22231140ca1a3e0323fde1523406f27"
EXTRACTED_MARKER = ".extracted"

N_CHANNELS = 8
SAMPLE_RATE = 48_000
#: Float64 preserves every published PCM_32 sample bit exactly after scaling
#: by 2**-31 (8 channels x 48 kHz x 8 bytes = 3.07 MB per second).

_M100 = {"make_model": "DJI Matrice 100", "n_rotors": 4}
_SECOND_DRONE = {
    "3DR": {"model": "3DR Solo", "n_rotors": 4, "propeller": "1045 (10 x 4.5 in, 2 blades)"},
    "Parrot Bebop 2": {
        "model": "Parrot Bebop 2",
        "n_rotors": 4,
        "propeller": "5040 (5 x 4 in, 3 blades)",
    },
}

#: Per-recording annotation, keyed by the xlsx "Recording #". ``xlsx`` = the
#: row's (Drone activated?, Displacement, horizontal m/s, vertical m/s, height
#: [m], Additional Drone) cells verbatim, checked against the file at build.
#: ``subcategory``: background (M100 motors off, nothing else flying) /
#: ground_motors_on / flight_hover / flight_vertical / flight_horizontal /
#: second_drone_only (M100 parked with motors off, the other drone flies) /
#: flight_hover_with_second_drone. Evidence: RecordingInfo.xlsx + Rascon,
#: Ruiz-Espitia, Martinez-Carranza, Sensors 19(18):3902 (2019) §2.2 / Table 1.
RECORDINGS: dict[int, dict[str, Any]] = {
    1: {
        "xlsx": ("NO", "NO", "0", "0", "0", None),
        "height_m": (0.0, 0.0),
        "subcategory": "background",
        "note": "system test (Sensors 2019: preliminary test, motors not activated)",
    },
    2: {
        "xlsx": ("NO", "NO", "0", "0", "0", None),
        "height_m": (0.0, 0.0),
        "subcategory": "background",
    },
    3: {
        "xlsx": ("NO", "NO", "0", "0", "0", None),
        "height_m": (0.0, 0.0),
        "subcategory": "background",
    },
    4: {
        "xlsx": ("YES", "NO", "0", "0", "0", None),
        "height_m": (0.0, 0.0),
        "subcategory": "ground_motors_on",
    },
    5: {
        "xlsx": ("YES", "YES", "0", "0.6", "0 - 3", None),
        "height_m": (0.0, 3.0),
        "subcategory": "flight_vertical",
    },
    6: {
        "xlsx": ("YES", "NO", "0", "0", "3", None),
        "height_m": (3.0, 3.0),
        "subcategory": "flight_hover",
    },
    7: {
        "xlsx": ("YES", "YES", "1.4", "0", "3", None),
        "height_m": (3.0, 3.0),
        "subcategory": "flight_horizontal",
    },
    8: {
        "xlsx": ("YES", "YES", "0", "0.4", "5", None),
        "height_m": (3.0, 5.0),
        "subcategory": "flight_vertical",
        "note": "xlsx height cell '5' but the description climbs 3 -> 5 m (Sensors 2019 Table 1: 3-5)",
    },
    9: {
        "xlsx": ("YES", "NO", "0", "0", "5", None),
        "height_m": (5.0, 5.0),
        "subcategory": "flight_hover",
        "note": "the AIRA-UAS web page says 'Drone activated: No'; xlsx and Sensors 2019 Table 1 say YES",
    },
    10: {
        "xlsx": ("YES", "YES", "1.4", "0", "5", None),
        "height_m": (5.0, 5.0),
        "subcategory": "flight_horizontal",
    },
    11: {
        "xlsx": ("YES", "YES", "0", "0.6", "5 - 10", None),
        "height_m": (5.0, 10.0),
        "subcategory": "flight_vertical",
    },
    12: {
        "xlsx": ("YES", "NO", "0", "0", "10", None),
        "height_m": (10.0, 10.0),
        "subcategory": "flight_hover",
    },
    13: {
        "xlsx": ("YES", "YES", "1.4", "0", "10", None),
        "height_m": (10.0, 10.0),
        "subcategory": "flight_horizontal",
        "note": "listed in the xlsx and Sensors 2019 Table 1; absent from the web page index",
    },
    14: {
        "xlsx": ("YES", "NO", "0", "0", "15", None),
        "height_m": (15.0, 15.0),
        "subcategory": "flight_hover",
    },
    15: {
        "xlsx": ("YES", "NO", "0", "0", "20", None),
        "height_m": (20.0, 20.0),
        "subcategory": "flight_hover",
    },
    16: {
        "xlsx": ("NO", "NO", "0", "0", "0", "3DR"),
        "height_m": (0.0, 0.0),
        "subcategory": "second_drone_only",
        "second_drone_motion": "hover",
        "second_drone_horizontal_distance_m": 2.0,
        "second_drone_height_m": 1.0,
    },
    17: {
        "xlsx": ("NO", "NO", "0", "0", "0", "3DR"),
        "height_m": (0.0, 0.0),
        "subcategory": "second_drone_only",
        "second_drone_motion": "circle",
        "second_drone_horizontal_distance_m": 2.0,
        "second_drone_height_m": 1.0,
        "note": "the web page labels this 'static'; xlsx and Sensors 2019 §2.2.2: a full 2 m-radius circle",
    },
    18: {
        "xlsx": ("YES", "NO", "0", "0", "5", "3DR"),
        "height_m": (5.0, 5.0),
        "subcategory": "flight_hover_with_second_drone",
        "second_drone_motion": "hover",
        "second_drone_horizontal_distance_m": 3.0,
        "second_drone_height_m": 5.0,
        "note": "xlsx: both hover at 5 m, 3 m apart; the web page says the 3DR flies a circle",
    },
    19: {
        "xlsx": ("NO", "NO", "0", "0", "0", "Parrot Bebop 2"),
        "height_m": (0.0, 0.0),
        "subcategory": "second_drone_only",
        "second_drone_motion": "hover",
        "second_drone_horizontal_distance_m": 2.0,
        "second_drone_height_m": 1.0,
    },
    20: {
        "xlsx": ("NO", "NO", "0", "0", "0", "Parrot Bebop 2"),
        "height_m": (0.0, 0.0),
        "subcategory": "second_drone_only",
        "second_drone_motion": "circle",
        "second_drone_horizontal_distance_m": 2.0,
        "second_drone_height_m": 1.0,
        "note": "the web page labels this 'static'; xlsx and Sensors 2019 §2.2.3: a full 2 m-radius circle",
    },
    21: {
        "xlsx": ("YES", "NO", "0", "0", "5", "Parrot Bebop 2"),
        "height_m": (5.0, 5.0),
        "subcategory": "flight_hover_with_second_drone",
        "second_drone_motion": "hover",
        "second_drone_horizontal_distance_m": 3.0,
        "second_drone_height_m": 5.0,
        "note": "xlsx: both hover at 5 m, 3 m apart; the web page says the Bebop flies a circle",
    },
}

#: Estimated microphone positions (metres), row i = ``wav_ch_{i+1}.wav``. Frame:
#: origin = ring centroid, z up (z = 0 is the ring plane: the 8 mics are coplanar
#: within the fit), x = "front" = bisector of ch1-4, y = left.
MIC_POS = np.array(
    [
        [+0.141, +0.289, 0.0],
        [+0.309, +0.131, 0.0],
        [+0.304, -0.129, 0.0],
        [+0.131, -0.269, 0.0],
        [-0.129, -0.270, 0.0],
        [-0.314, -0.126, 0.0],
        [-0.314, +0.101, 0.0],
        [-0.129, +0.273, 0.0],
    ]
)

GEOMETRY: dict[str, Any] = {
    "mic_pos_quality": "estimated (acoustic self-calibration)",
    "frame": "origin = centroid of the 8-mic ring; z up, z = 0 the ring plane; x = 'front' = "
    "bisector of ch1-4; y = left; metres",
    "method": "TDOA self-calibration: GCC-PHAT (0.2 s frames, 300-6000 Hz, all 28 pairs) on "
    "Recordings 17 (3DR Solo) and 20 (Bebop 2), a second drone circling the motors-off Matrice at "
    "r = 2 m, h = 1 m; frames with pair-closure rms < 40 us kept (129 / 175); near-field NLS with "
    "per-frame source azimuth and range (prior 2 m), circle-centre offset, source 0.5 m above the "
    "mics (0.3/0.7 m moves r by 0.01 m), c = 343 m/s; the two fits Procrustes-aligned and averaged "
    "(fit rms 6-7 mm; Rec17 vs Rec20 differ 0.018-0.056 m per channel, rms 0.033 m)",
    "result": "near-regular ring, radius 0.32 +- 0.03 m; wav_ch_1..8 run sequentially around it; "
    "pairs (1,2), (3,4), (5,6), (7,8) straddle the X-frame arm diagonals at about +-20 deg",
    "uncertainty": "+-0.03 m horizontal and +-5 deg azimuth per mic; relative z within +-0.08 m "
    "(weakly identifiable, set to 0)",
    "ambiguities": "(a) mirror chirality is acoustically unidentifiable: ch1 may be front-RIGHT "
    "(y -> -y) instead of front-left as published; (b) the front axis is INFERRED from Cabrera-Ponce, "
    "Martinez-Carranza, Rascon, IMAV 2019 p.2 ('microphones 1, 2, 3 and 4 mounted in the front and "
    "5-8 in the back' on 'the same 3D-printed structure used by' AIRA-UAS), assumed carried over; "
    "(c) ring height above the rotor plane unknown (Sensors 2019 Fig. 2c: booms rise above the arms)",
    "rotors": "DJI Matrice 100 X-frame, motors on the arm diagonals at r = 0.325 m (650 mm "
    "wheelbase), below the ring by an unknown height; rotor_pos not published",
    "placement": "onboard: 8 microphones on a 3D-printed structure on the DJI Matrice 100, two per "
    "boom at the ends of four booms on the arm diagonals (Sensors 2019 Fig. 2c; IMAV 2019 Fig. 2)",
    "channel_map": "audio/mic_pos row i = UAS/.../wav_ch_{i+1}.wav (file name)",
    "evidence": "self-calibration scripts on Hetzner ~/hns-ingest-20261126/tmp/airaku_geo/gcc_aira.py "
    "(+ local fit selfcal_aira.py); Sensors 19(18):3902 §2 and Fig. 2c; IMAV 2019 "
    "https://www.imavs.org/papers/2019/18.pdf p.2-3 and Fig. 2; ICUAS 2018 (closed) not inspected",
}

PROVENANCE: dict[str, Any] = {
    "source_url": PAGE_URL,
    "download_url": URL,
    "archive": ARCHIVE,
    "archive_bytes": ARCHIVE_BYTES,
    "archive_sha256": ARCHIVE_SHA256,
    "license": "CC0-1.0 (schema.org Dataset 'license' of https://aira.iimas.unam.mx/, "
    "corpus-wide and on its 'UAS 8 (Unmanned Aerial System)' part)",
    "license_evidence": (
        "https://aira.iimas.unam.mx/ JSON-LD: Dataset 'Acoustic Interactions for Robot Audition' "
        "license https://creativecommons.org/publicdomain/zero/1.0/ and hasPart 'UAS 8 (Unmanned "
        "Aerial System)' license https://creativecommons.org/publicdomain/zero/1.0/; page retrieved "
        "2026-11-26, 22054 B, sha256 dc628138ac2bf930e1ed165b411bc19b1d1ebeb9508ee72fbff88f91e68bb290, "
        "Last-Modified 2024-12-04, ETag 5626-628789c603040. The archive holds no licence/README file "
        "(168 wav_ch_N.wav + 3 RecordingInfo.xlsx only)."
    ),
    "attribution": "cite Ruiz-Espitia, Martinez-Carranza, Rascon, ICUAS 2018 and Rascon, "
    "Ruiz-Espitia, Martinez-Carranza, Sensors 2019 (requested by the publisher, not required by CC0)",
    "citation": (
        "O. Ruiz-Espitia, J. Martinez-Carranza, C. Rascon, 'AIRA-UAS: an Evaluation Corpus for "
        "Audio Processing in Unmanned Aerial System', ICUAS 2018, pp. 836-845, "
        "doi 10.1109/ICUAS.2018.8453466; C. Rascon, O. Ruiz-Espitia, J. Martinez-Carranza, "
        "'On the Use of the AIRA-UAS Corpus to Evaluate Audio Processing Algorithms in Unmanned "
        "Aerial Systems', Sensors 19(18):3902 (2019), doi 10.3390/s19183902."
    ),
    "collection_method": (
        "Outdoors, 2017-10-04 (archive member mtimes 18:54-19:45, timezone unrecorded): an "
        "8SoundsUSB interface + Raspberry Pi 2 with eight microphones on a DJI Matrice 100 "
        "(camera/gimbal removed; 650 mm wheelbase, DJI 1345 13x4.5 in 2-blade props). Protocol 1 "
        "(rec 1-15): background, motors on at ground, hover at 3/5/10/15/20 m, vertical climbs "
        "0.4-0.6 m/s, horizontal passes 1.4 m/s. Protocols 2/3 (rec 16-21): a 3DR Solo / Parrot "
        "Bebop 2 hovering or circling 2 m from the parked M100 (motors off) at 1 m, then both "
        "hovering 3 m apart at 5 m."
    ),
    "equipment": "DJI Matrice 100 + 8SoundsUSB (8 mic inputs) + Raspberry Pi 2 (Raspbian)",
    "observation_type": "onboard_array",
    "sample_rate": SAMPLE_RATE,
    "channels": N_CHANNELS,
    "recordings": len(RECORDINGS),
    "frames": "one whole recording per Frame (no chunking)",
    "recording_separation": "independent captures: archive mtimes (file close) minus "
    "durations put 14.5-658 s between consecutive recordings, and each has its own settings; "
    "none is joined to another",
    "leading_zeros": "Recordings 1, 10 and 16 start with exactly 1024 all-zero frames on all 8 "
    "channels (one interface buffer); kept, counted in meta.leading_digital_zero_frames",
    "telemetry": "none published (no flight log, no motor speeds)",
    "splits": "none published",
    "overlap": "AIRA-UAS archive paths, recorder and protocols differ from SPCUP19 bonus-task "
    "archives and AVQ sequences (AVQ native rate 44.1 kHz on tripod; AIRA-UAS 48 kHz outdoors).",
    "description": (
        "AIRA-UAS: 21 outdoor 8-channel 48 kHz recordings from microphones on a DJI Matrice 100 "
        "(ego-noise at ground/hover/climb/horizontal flight, background, and a second drone "
        "flying nearby), one Frame per whole recording on its own clock; per-recording settings "
        "from the archive's RecordingInfo.xlsx. No telemetry; mic coordinates estimated from TDOA."
    ),
}


# ─── Fetch ────────────────────────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_aira_uas(dest: Path) -> Path:
    """Fetch + verify ``UAS.tar.gz`` into *dest* and extract ``UAS/`` (idempotent).

    The archive is kept (it is the pinned identity); extraction admits only
    regular files and directories under ``UAS/`` (``tarfile`` ``data`` filter).
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / ARCHIVE
    marker = dest / EXTRACTED_MARKER
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == ARCHIVE_SHA256:
        return dest
    http_fetch(URL, archive, expected_size=ARCHIVE_BYTES)
    size, digest = archive.stat().st_size, _sha256(archive)
    if (size, digest) != (ARCHIVE_BYTES, ARCHIVE_SHA256):
        raise OSError(
            f"{URL}: got {size} B sha256 {digest}, want {ARCHIVE_BYTES} B {ARCHIVE_SHA256}"
        )
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for m in members:
            parts = Path(m.name).parts
            if not (m.isfile() or m.isdir()) or not parts or parts[0] != "UAS" or ".." in parts:
                raise OSError(f"{ARCHIVE}: unexpected member {m.name!r}")
        tar.extractall(dest, members=members, filter="data")
    marker.write_text(ARCHIVE_SHA256 + "\n", encoding="utf-8")
    return dest


# ─── RecordingInfo.xlsx ───────────────────────────────────────────────────────

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _read_xlsx_rows(path: Path) -> list[dict[str, str]]:
    """First worksheet of a small xlsx → ``[{column letter: cell text}]`` (stdlib only)."""
    with zipfile.ZipFile(path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            shared = [
                "".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t"))
                for si in root.findall("m:si", _NS)
            ]
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.iter(f"{{{_NS['m']}}}row"):
        cells: dict[str, str] = {}
        for c in row.findall("m:c", _NS):
            col = "".join(ch for ch in c.get("r", "") if ch.isalpha())
            v = c.find("m:v", _NS)
            if v is not None and v.text is not None:
                text = shared[int(v.text)] if c.get("t") == "s" else v.text
            else:
                text = "".join(t.text or "" for t in c.iter(f"{{{_NS['m']}}}t"))
            if text.strip():
                cells[col] = text.strip()
        if cells:
            rows.append(cells)
    return rows


def load_recording_info(uas_root: Path) -> dict[int, dict[str, Any]]:
    """``{recording #: native row}`` from every ``Protocol*/RecordingInfo.xlsx``."""
    info: dict[int, dict[str, Any]] = {}
    for xlsx in sorted(Path(uas_root).glob("Protocol*/RecordingInfo.xlsx")):
        protocol = int(xlsx.parent.name.removeprefix("Protocol"))
        for cells in _read_xlsx_rows(xlsx):
            if not cells.get("A", "").isdigit():
                continue  # header rows
            n = int(cells["A"])
            if n in info:
                raise ValueError(f"{xlsx}: recording {n} listed twice")
            info[n] = {
                "protocol": protocol,
                "n_channels": cells.get("B"),
                "drone_activated": cells.get("C"),
                "mounted_microphones": cells.get("D"),
                "displacement": cells.get("E"),
                "horizontal_velocity": cells.get("F"),
                "vertical_velocity": cells.get("G"),
                "height": cells.get("H"),
                "description": cells.get("I"),
                "additional_drone": cells.get("J"),
                "xlsx": f"UAS/{xlsx.parent.name}/{xlsx.name}",
            }
    return info


# ─── Build ────────────────────────────────────────────────────────────────────


def _recording_meta(n: int, row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    ann = RECORDINGS[n]
    native = (
        row["drone_activated"],
        row["displacement"],
        row["horizontal_velocity"],
        row["vertical_velocity"],
        row["height"],
        row["additional_drone"],
    )
    if (
        native != ann["xlsx"]
        or row["n_channels"] != str(N_CHANNELS)
        or row["mounted_microphones"] != "YES"
    ):
        raise ValueError(f"{DATASET} recording {n}: xlsx row {native} != annotation {ann['xlsx']}")
    ego = row["drone_activated"] == "YES"
    sub = ann["subcategory"]
    flying = sub.startswith("flight_")
    h_lo, h_hi = ann["height_m"]
    second = _SECOND_DRONE.get(row["additional_drone"]) if row["additional_drone"] else None
    condition = ("flight" if flying else "bench") if ego else "rotors_off"
    operating = {
        "condition": condition,
        "flight_mode": ("hover" if sub.endswith("hover") or "hover_with" in sub else "manoeuvre")
        if flying
        else None,
        "ego_rotors_active": ego,
        "n_active_rotors": 4 if ego else 0,
        "contains_rotor_noise": ego or second is not None,
        "displacement": row["displacement"] == "YES",
        "platform_motion": "moving"
        if row["displacement"] == "YES"
        else ("hovering" if flying else "on_ground"),
        "horizontal_velocity_mps": float(row["horizontal_velocity"]),
        "vertical_velocity_mps": float(row["vertical_velocity"]),
        "height_min_m": h_lo,
        "height_max_m": h_hi,
        "height_native": row["height"],
        "on_ground": sub in ("background", "ground_motors_on", "second_drone_only"),
    }
    label = {
        "subcategory": sub,
        "protocol": f"Protocol{row['protocol']}",
        "native_description": row["description"],
        "external_source": "other" if second else "none",
        "annotation_note": ann.get("note"),
    }
    if second:
        label.update(
            second_drone=second["model"],
            second_drone_n_rotors=second["n_rotors"],
            second_drone_propeller=second["propeller"],
            second_drone_motion=ann["second_drone_motion"],
            second_drone_horizontal_distance_m=ann["second_drone_horizontal_distance_m"],
            second_drone_height_m=ann["second_drone_height_m"],
        )
    return {"operating": operating, "label": label}


#: The 44-byte header every AIRA-UAS WAV carries: RIFF/WAVE, ``fmt `` PCM (1),
#: 1 channel, 48 kHz, 192000 B/s, block 4, 32 bits, then ``data``. The writer
#: never finalised it: the RIFF and data size fields are left 8 and 0, so the
#: data run to end of file and are read by offset, not through the header.
_HEADER = (
    b"RIFF\x08\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\xbb\x00\x00"
    b"\x00\xee\x02\x00\x04\x00\x20\x00data\x00\x00\x00\x00"
)
_BYTES_PER_FRAME = 4
#: int32 full scale (soundfile's PCM_32 -> float convention).
_FULL_SCALE = float(2**31)


def _open_pcm32(path: Path) -> np.memmap:
    """Samples of one unfinalised AIRA-UAS WAV as a read-only ``int32`` memmap."""
    with open(path, "rb") as fh:
        head = fh.read(len(_HEADER))
    n_bytes = path.stat().st_size - len(_HEADER)
    if (
        head[:4] != _HEADER[:4]
        or head[8:40] != _HEADER[8:40]
        or (int.from_bytes(head[4:8], "little"), int.from_bytes(head[40:44], "little"))
        not in ((8, 0), (n_bytes + 36, n_bytes))
        or n_bytes <= 0
        or n_bytes % _BYTES_PER_FRAME
    ):
        raise ValueError(f"{path}: unexpected WAV header {head!r} for {n_bytes} data bytes")
    return np.memmap(
        path, dtype="<i4", mode="r", offset=len(_HEADER), shape=(n_bytes // _BYTES_PER_FRAME,)
    )


def _leading_zero_frames(block: np.ndarray) -> int:
    """Frames at the start where every channel is exactly 0."""
    nonzero = np.flatnonzero(np.any(block != 0, axis=0))
    return int(nonzero[0]) if nonzero.size else int(block.shape[1])


def build(root: Path) -> Iterator[tuple[str, td.Frame]]:
    """Fetched raw dir (containing ``UAS/``) → one Frame per whole recording.

    A recording spans its longest channel. Recording 6's ch7 and ch8 end 11
    and 1024 frames early; they are set to NaN past their own end, valid
    lengths in ``meta.channel_frames``. GCC-PHAT lags of ch7/ch8 against ch1
    at 0.5-1 s and just before their own ends differ by 5 and 3 samples (other,
    complete channels wander up to ~100 samples), nothing near a 1024-frame
    buffer: the missing frames are a tail truncation, not a mid-file drop."""
    uas = Path(root) / "UAS"
    info = load_recording_info(uas)
    dirs = {
        int(p.name.removeprefix("Recording")): p
        for p in uas.glob("Protocol*/Recording*")
        if p.is_dir()
    }
    if set(dirs) != set(RECORDINGS) or set(info) != set(RECORDINGS):
        raise ValueError(
            f"{DATASET}: recordings on disk {sorted(dirs)} / in xlsx {sorted(info)} "
            f"!= annotated {sorted(RECORDINGS)}"
        )
    for n in sorted(RECORDINGS):
        rec_dir, row = dirs[n], info[n]
        if rec_dir.parent.name != f"Protocol{row['protocol']}":
            raise ValueError(
                f"{DATASET} recording {n}: directory {rec_dir} vs xlsx protocol {row['protocol']}"
            )
        paths = [rec_dir / f"wav_ch_{c}.wav" for c in range(1, N_CHANNELS + 1)]
        if sorted(p.name for p in rec_dir.iterdir()) != sorted(p.name for p in paths):
            raise ValueError(
                f"{DATASET}: {rec_dir} holds {sorted(p.name for p in rec_dir.iterdir())}"
            )
        ch_frames: list[int] = []
        channels = []
        for p in paths:
            samples = _open_pcm32(p)
            ch_frames.append(int(samples.shape[0]))
            channels.append(samples)
        n_frames = max(ch_frames)
        block = np.full((N_CHANNELS, n_frames), np.nan, dtype=np.float64)
        for c, samples in enumerate(channels):
            np.multiply(samples, 1 / _FULL_SCALE, out=block[c, : samples.size])
        del channels
        rec_id = f"UAS/{rec_dir.parent.name}/{rec_dir.name}"
        groups = _recording_meta(n, row)
        meta = meta_frame(
            rec_id,
            DATASET,
            system={"category": "drone", **_M100, "recorder": "8SoundsUSB + Raspberry Pi 2"},
            observation={
                "mic_placement": "onboard",
                "type": "onboard_array",
                "environment": "outdoor",
            },
            operating=groups["operating"],
            label=groups["label"],
            extra={
                "geometry": td.Frame(dict(GEOMETRY)),
                "recording_number": n,
                "source_kind": "original_recording",
                "raw_paths": ";".join(f"{rec_id}/{p.name}" for p in paths),
                "recording_info": row["xlsx"],
                "native_format": "WAV PCM_32; audio = int32 / 2**31 as lossless float64",
                "channel_frames": ";".join(str(f) for f in ch_frames),
                "channels_complete": len(set(ch_frames)) == 1,
                "missing_tail": "NaN past a channel's own end" if len(set(ch_frames)) > 1 else None,
                "recording_frames": n_frames,
                "recording_duration_s": n_frames / SAMPLE_RATE,
                "leading_digital_zero_frames": _leading_zero_frames(block),
            },
        )
        audio = td.uniform(block, SAMPLE_RATE, dims=("mic", "time"), t_start=0.0)
        frame = td.Frame(
            {"audio": audio, "mic_pos": td.wrap(MIC_POS, dims=("mic", None)), "meta": meta}
        )
        yield f"{rec_dir.parent.name}/{rec_dir.name}", frame
