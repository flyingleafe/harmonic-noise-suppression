"""Salford DroneNoise Database v3 (Edzell, Scotland, 17 August 2022).

University of Salford figshare article 22133411 **version 3**
(doi:10.17866/rd.salford.22133411.v3, CC BY 4.0; Ramos-Romero, Torija Martinez,
Green, Asensio; EPSRC DroneNoise EP/V031848/1). The deposit is one EXTERNAL
ground array: nine free-field microphones mounted inverted 7 mm above metal
ground plates in one straight line, spaced so that each is a further 15 deg off
vertical as seen from 10 m above the centre microphone M5 (-60..+60 deg).
Drones flew 10 m AGL either straight across the line over M5 (flyover) or held
a hover over M5 (Ramos-Romero et al., Forum Acusticum 2023:
doi:10.61782/fa.2023.0248 Sec. 2.1, Fig. 2; doi:10.61782/fa.2023.0249 Sec. 2,
Table 1).

Every event microphone is a separate mono WAV ``Ed_<sUAS>_<HAGL>_<op><speed>_
<payload>_<start>_<direction>_ev<k>_M<m>.wav``: IEEE float32 at 50 kHz, values
in pascal [INFERENCE: the nine ``Calib_Scotland_ch<k>.wav`` 94 dB calibrator
takes read 1.0 Pa RMS in their quiet seconds at the same scale]. The token
names come from the native workbook ``DNM_SQM_FLYO_AESCT.xlsx`` (sheet
``flyovers`` header: Operador, sUAS, HAGL, Operation, Speed, Payload,
Starting, Direction).

Grouping (checked 2026-11-26 on the verified archive; see ``SYNC_EVIDENCE``):
the nine files of one event have equal length (1,000,000 or 1,500,000 samples)
and no padding. Their GCC-PHAT lags (500 Hz-8 kHz, 1 s windows over the WHOLE
file) fit one physical source on one sample clock. In hovers the lags against
M5 are constant and symmetric and match the stated geometry (M1/M9 ~24-30 ms,
M2/M8 ~9-12.6 ms, M3/M7 ~2.9-4.8 ms, M4/M6 ~0.3-1.3 ms; 29.4/12.2/4.6/1.0 ms
predicted at c = 340 m/s) at the start, middle and end. In flyovers they fit a
source whose transverse distance falls from ~100 m to ~0 and rises again, with
a 0.07-0.1 ms median residual. Files of different events are incoherent (PHAT
peak <= 0.05, against 0.2-0.7 within an event). So each complete event is ONE
Frame, ``audio`` ``(mic, time)`` in natural order M1..M9, on the publisher's
common window clock (``t_start`` = 0 = first sample of the published window;
the absolute time and any trimming of the acquisition are unpublished).

Kept apart (one Frame each, never joined):

- ``Ed_3p_10_H00_N_C_nw`` ev2/ev3: only M5 was published, so each is a mono
  take of its own event (separate uploads 46212084/46212087).
- ``Calib_Scotland_ch1..9``: sequential calibrator takes of different lengths,
  one per DAQ channel. They are not simultaneous and not a drone event.
- Distinct events (``ev<k>``) and operating tags are never merged.

Duplicate file IDs: 46212081 and 46212195 are both
``Ed_3p_10_H00_N_C_nw_ev1_M5.wav`` with identical bytes (figshare prefixes the
ZIP member names with the ID). The audio is read once and both IDs are kept in
``meta.source``. Its lags against ev1 M4/M6 (0.86/0.99 ms) place it in ev1.

The builder reads the stored ZIP members in place (no extraction). The
downloaded figshare v3 JSON is the file table: its (id, size, md5, name) list
must hash to ``FILE_TABLE_SHA256``, and every ZIP member must match its own
record (size and md5). The full-archive hash is not used, because figshare
generates the ZIP on request.

Geometry limits: ``mic_pos`` is the NOMINAL design (x = 10 m * tan(15 deg *
(m - 5)) along the line, y = 0, z = 0.007 m), not a survey. The TDOAs confirm
that M1..M9 run in order along the line. Neither the mirror (which end M1 is
on) nor the compass bearing of the line is published. The source path is
nominal (over M5 at 10 m AGL); no GPS track is deposited.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import tdseries as td

from data_processing.frames import audio_series
from data_processing.sources._common import meta_frame, validate_operating

DATASET = "Salford-DroneNoise"
FRAMES_DATASET = "Salford-DroneNoise-frames"
ARTICLE_ID = 22133411
ARTICLE_VERSION = 3
DOI = "10.17866/rd.salford.22133411.v3"
ZIP_NAME = "22133411_v3.zip"
META_NAME = "figshare-article-22133411-v3.json"
ZIP_URL = f"https://ndownloader.figshare.com/articles/{ARTICLE_ID}/versions/{ARTICLE_VERSION}"
META_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}/versions/{ARTICLE_VERSION}"
WORKBOOK = "DNM_SQM_FLYO_AESCT.xlsx"

SAMPLE_RATE = 50_000
N_MICS = 9
HAGL_M = 10.0
MIC_HEIGHT_M = 0.007
MIC_NAMES = tuple(f"M{m}" for m in range(1, N_MICS + 1))
MIC_ANGLE_DEG = tuple(15.0 * (m - 5) for m in range(1, N_MICS + 1))
MIC_POS = np.array(
    [[HAGL_M * np.tan(np.deg2rad(a)), 0.0, MIC_HEIGHT_M] for a in MIC_ANGLE_DEG],
    dtype=np.float64,
)

#: SHA-256 of the v3 file list as ``{id}\t{size}\t{computed_md5}\t{name}\n``
#: lines sorted by figshare id: pins the identity of the downloaded figshare
#: JSON (keeps both duplicate IDs, excludes volatile counters).
FILE_TABLE_SHA256 = "df64f9a7dc2536164d09a36cb09e92cd69ae6b342ffe2b1bee5ee5dd5435b56d"

FileEntry = tuple[int, str, int, str]  # (figshare id, name, bytes, md5)

#: sUAS codes. ``3p``/``M3``/``Yn``: Ramos-Romero et al. 2023 (fa.2023.0249)
#: Table 1. ``Fp`` is in neither paper; its 795 g and 4 rotors come from the
#: workbook (``flyovers`` rows 5-6).
SUAS: dict[str, dict[str, Any]] = {
    "3p": {
        "make": "DJI",
        "model": "Mini 3 Pro",
        "n_rotors": 4,
        "weight_g": 249,
        "evidence": "doi:10.61782/fa.2023.0249 Table 1",
    },
    "M3": {
        "make": "DJI",
        "model": "Matrice 300 RTK",
        "n_rotors": 4,
        "weight_g": 6300,
        "payload_g": 930,
        "diagonal_wheelbase_mm": 895,
        "evidence": "doi:10.61782/fa.2023.0249 Table 1; fa.2023.0248 Table 1",
    },
    "Yn": {
        "make": "Yuneec",
        "model": "H520E",
        "n_rotors": 6,
        "weight_g": 1633,
        "payload_g": 350,
        "diagonal_wheelbase_mm": 520,
        "evidence": "doi:10.61782/fa.2023.0249 Table 1",
    },
    "Fp": {
        "n_rotors": 4,
        "weight_g": 795,
        "model_inferred": "DJI FPV [INFERENCE: code 'Fp' and the workbook gross weight "
        "795 g, the DJI FPV take-off weight; not named in the papers or the deposit]",
        "evidence": "workbook DNM_SQM_FLYO_AESCT.xlsx sheet flyovers rows 5-6",
    },
}

#: Per-event grouping evidence, measured 2026-11-26 on the verified v3 archive
#: (GCC-PHAT, 500 Hz-8 kHz, 1 s windows over the whole file, every mic vs M5;
#: least-squares source fit with HAGL fixed at 10 m and c = 340 m/s over the
#: windows whose PHAT peak exceeded the 0.05 incoherent-control maximum).
#: Hovers also keep the per-mic median lag vs M5 (ms, M1..M9).
SYNC_EVIDENCE: dict[str, dict[str, Any]] = {
    "Ed_3p_10_F15_N_W_dw/ev1": {
        "windows_fit": "16/20",
        "fit_residual_ms_median": 0.065,
        "min_transverse_m": 2,
    },
    "Ed_3p_10_F15_N_W_dw/ev2": {
        "windows_fit": "18/20",
        "fit_residual_ms_median": 0.105,
        "min_transverse_m": 0,
    },
    "Ed_3p_10_F15_N_W_dw/ev3": {
        "windows_fit": "20/20",
        "fit_residual_ms_median": 0.065,
        "min_transverse_m": 13,
    },
    "Ed_Fp_10_F15_N_W_dw/ev1": {
        "windows_fit": "16/20",
        "fit_residual_ms_median": 0.082,
        "min_transverse_m": 0,
    },
    "Ed_Fp_10_F15_N_W_dw/ev2": {
        "windows_fit": "18/20",
        "fit_residual_ms_median": 0.066,
        "min_transverse_m": 0,
    },
    "Ed_Fp_10_F15_N_W_dw/ev3": {
        "windows_fit": "17/20",
        "fit_residual_ms_median": 0.087,
        "min_transverse_m": 15,
    },
    "Ed_M3_10_F15_N_E_dw/ev1": {
        "windows_fit": "17/20",
        "fit_residual_ms_median": 0.074,
        "min_transverse_m": 29,
    },
    "Ed_M3_10_F15_N_E_dw/ev2": {
        "windows_fit": "18/20",
        "fit_residual_ms_median": 0.072,
        "min_transverse_m": 30,
    },
    "Ed_M3_10_F15_N_E_dw/ev3": {
        "windows_fit": "17/20",
        "fit_residual_ms_median": 0.075,
        "min_transverse_m": 29,
    },
    "Ed_M3_10_F15_N_E_dw/ev4": {
        "windows_fit": "17/20",
        "fit_residual_ms_median": 0.074,
        "min_transverse_m": 31,
    },
    "Ed_M3_10_F15_N_E_dw/ev5": {
        "windows_fit": "17/20",
        "fit_residual_ms_median": 0.072,
        "min_transverse_m": 27,
    },
    "Ed_Yn_10_F15_Y_W_dw/ev1": {
        "windows_fit": "26/30",
        "fit_residual_ms_median": 0.080,
        "min_transverse_m": 21,
    },
    "Ed_Yn_10_F15_Y_W_dw/ev2": {
        "windows_fit": "28/30",
        "fit_residual_ms_median": 0.077,
        "min_transverse_m": 5,
    },
    "Ed_Yn_10_F15_Y_W_dw/ev3": {
        "windows_fit": "29/30",
        "fit_residual_ms_median": 0.073,
        "min_transverse_m": 6,
    },
    "Ed_3p_10_H00_N_C_nw/ev1": {
        "windows_fit": "20/20",
        "fit_residual_ms_median": 0.067,
        "max_transverse_m": 4,
        "lag_vs_M5_ms": [28.284, 11.525, 4.123, 0.864, 0.0, 0.994, 4.547, 11.994, 29.137],
    },
    "Ed_M3_10_H00_N_C_nw/ev1": {
        "windows_fit": "20/20",
        "fit_residual_ms_median": 0.119,
        "max_transverse_m": 8,
        "lag_vs_M5_ms": [27.909, 11.692, 4.446, 1.195, 0.0, 0.321, 2.851, 8.958, 24.437],
    },
    "Ed_M3_10_H00_Y_C_nw/ev1": {
        "windows_fit": "20/20",
        "fit_residual_ms_median": 0.138,
        "max_transverse_m": 7,
        "lag_vs_M5_ms": [27.671, 11.578, 4.287, 1.276, 0.0, 0.357, 2.918, 9.305, 25.033],
    },
    "Ed_Yn_10_H00_Y_C_nw/ev1": {
        "windows_fit": "20/20",
        "fit_residual_ms_median": 0.234,
        "max_transverse_m": 4,
        "lag_vs_M5_ms": [29.786, 12.584, 4.791, 1.241, 0.0, 0.427, 3.471, 10.683, 27.569],
    },
}

SYNC_METHOD = (
    "GCC-PHAT 500 Hz-8 kHz in 1 s windows over the whole file, every microphone vs M5 and "
    "adjacent pairs vs their spacing/c bound; source fit with HAGL 10 m, c = 340 m/s; "
    "controls pairing files of different events peak <= 0.05 (within events 0.2-0.7); "
    "measured 2026-11-26 on the verified v3 archive"
)

GEOMETRY: dict[str, Any] = {
    "array": "9 free-field microphones, inverted 7 mm above metal ground plates, in one "
    "straight line; 15 deg steps seen from 10 m above the centre microphone M5",
    "mic_pos_frame": "metres; x along the line from M1 towards M9 with M5 at the origin, "
    "z up from the ground plane, y completes a right-handed frame (horizontal, across the line)",
    "mic_pos_status": "NOMINAL design positions x = 10 m * tan(15 deg * (m - 5)), "
    "z = 0.007 m; no surveyed coordinates are published",
    "mic_angle_deg": list(MIC_ANGLE_DEG),
    "channel_order": "M1..M9 as in the file names. Hover TDOAs are symmetric about M5 "
    "and grow outwards (M4/M6 < M3/M7 < M2/M8 < M1/M9), so the numbering runs "
    "monotonically along the line",
    "unknown": "which end of the line M1 is on (mirror); compass bearing of the line "
    "[INFERENCE: flyover start tokens W/E and the transverse path suggest a roughly "
    "north-south line]; exact plate positions",
    "source_path": "nominal: hover over M5 at 10 m AGL; flyover across the line over M5 "
    "at 10 m AGL. No GPS track is deposited (fa.2023.0248 Sec. 4 notes GPS would "
    "have improved the angle resolution)",
    "hagl_m": HAGL_M,
    "evidence": "Ramos-Romero et al., Forum Acusticum 2023, doi:10.61782/fa.2023.0248 "
    "Sec. 2.1 + Fig. 2 (flyover setup) and doi:10.61782/fa.2023.0249 Sec. 2 + Fig. 1 "
    "(hover setup, -60 <= theta <= 60 deg); grouping TDOAs in SYNC_EVIDENCE",
}

CLOCK = (
    "audio t = seconds from the first sample of the publisher's window; the nine "
    "microphones of one event share that window and one sample clock (SYNC_EVIDENCE). "
    "Absolute time, recorder and any trimming of the original acquisition are not published"
)

PROVENANCE: dict[str, Any] = {
    "source_url": f"https://salford.figshare.com/articles/dataset/DroneNoise_Database/{ARTICLE_ID}/{ARTICLE_VERSION}",
    "doi": DOI,
    "article_id": ARTICLE_ID,
    "version": f"{ARTICLE_VERSION} (published 2024-05-13, modified 2025-01-23; first online 2023-02-23)",
    "download_url": ZIP_URL,
    "metadata_url": META_URL,
    "license": "CC BY 4.0 (figshare article licence)",
    "attribution": "Ramos Romero, Carlos; Torija Martinez, Antonio Jose; Green, Nathan; "
    "Asensio, Cesar (2023). DroneNoise Database. University of Salford. Dataset. "
    "https://doi.org/10.17866/rd.salford.22133411.v3",
    "citation": "C. Ramos-Romero, N. Green, C. Asensio, A. J. Torija, 'On-field measurement "
    "for sUAS noise characterisation' and 'sUAS noise characterisation during hovering "
    "operations', Forum Acusticum 2023, doi:10.61782/fa.2023.0248 / fa.2023.0249",
    "funding": "EPSRC EP/V031848/1 (DroneNoise)",
    "collection_method": "field campaign at Edzell, Scotland, 17 August 2022: drones at "
    "10 m AGL fly across or hover over a 9-microphone ground-plate line array; "
    "temperature 13-16 C, wind 0-6 m/s, background LAeq ~35 dB (fa.2023.0248 Sec. 2.1)",
    "equipment": "9 free-field microphones on metal ground plates; 50 kHz float32 "
    "acquisition; acoustic calibrator (model not published)",
    "observation_type": "external_ground_array",
    "sample_rate": SAMPLE_RATE,
    "channels": N_MICS,
    "archive": {
        "zip": ZIP_NAME,
        "identity": "per-file figshare id, size, md5 and name from the v3 JSON, pinned "
        "by FILE_TABLE_SHA256; figshare builds the article ZIP on request, so the "
        "archive hash is not an identity",
        "files": "175: 165 event WAVs (one file ID duplicated with equal bytes), "
        "9 calibration WAVs, 1 workbook",
        "member_names": "figshare prefixes members whose names collide with "
        "'<file id>_' (46212081_/46212195_Ed_3p_10_H00_N_C_nw_ev1_M5.wav)",
        "wav_format": "RIFF WAVE IEEE float (tag 3), 32-bit, mono, 50000 Hz; chunks fmt, "
        "fact, PEAK, data",
    },
    "frames": "18 nine-microphone events (14 flyovers, 4 hovers) as one (mic, time) Frame "
    "each; 2 single-microphone hover takes and 9 calibrator takes as mono Frames",
    "workbook": "DNM_SQM_FLYO_AESCT.xlsx sheet 'flyovers': per flyover and microphone "
    "N5, S5, R5, F5, I5, T5 (5 % exceedance sound-quality metrics; the hover paper "
    "computed loudness/sharpness/fluctuation strength in HEAD ArtemiS SUITE 12.0). "
    "The other two sheets are formula views of it; one, BY_SQM_flyovers I5 M7, "
    "references the M8 column",
    "telemetry": "none deposited",
    "splits": "none published; split by event, never by microphone",
    "geometry": GEOMETRY,
    "description": "Salford DroneNoise v3: DJI Mini 3 Pro, DJI FPV [inferred], DJI "
    "Matrice 300 RTK and Yuneec H520E flyovers (15 m/s) and hovers at 10 m AGL over a "
    "9-microphone ground line array, 50 kHz float32, with per-microphone sound-quality "
    "metrics for the flyovers.",
}

_EVENT = re.compile(
    r"^(?P<base>(?P<operator>[A-Za-z]+)_(?P<suas>[A-Za-z0-9]+)_(?P<hagl>\d+)_(?P<operation>[FH])"
    r"(?P<speed>\d+)_(?P<payload>[NY])_(?P<start>[WEC])_(?P<direction>[a-z]{2}))"
    r"_ev(?P<event>\d+)_M(?P<mic>[1-9])\.wav$"
)
_CALIB = re.compile(r"^Calib_Scotland_ch(?P<ch>[1-9])\.wav$")
_PATTERN = {"F": "flyover", "H": "hover"}
_FLIGHT_MODE = {"F": "manoeuvre", "H": "hover"}
_START = {"W": "west", "E": "east", "C": "centre (over M5)"}
_SQM_UNITS = {
    "N5": "loudness, sone",
    "S5": "sharpness, acum",
    "R5": "roughness, asper [INFERENCE]",
    "F5": "fluctuation strength, vacil",
    "I5": "impulsiveness, iu [INFERENCE]",
    "T5": "tonality, tu [INFERENCE]",
}

#: How far sheet order (k-th row of an ID = ev<k>) is supported, from Spearman
#: rho between workbook N5 and per-mic A-weighted Leq over all row
#: permutations (2026-11-26). The workbook has no event column.
WORKBOOK_ROW_ORDER = {
    "Ed_3p_10_F15_N_W": "supported: sheet order is the best of 6 permutations (rho 0.993, next 0.883)",
    "Ed_Fp_10_F15_N_W": "weakly supported: sheet order is best (rho 0.993) but another permutation reaches 0.980",
    "Ed_Yn_10_F15_Y_W": "supported: sheet order is the best of 6 permutations (rho 0.984, next 0.955)",
    "Ed_M3_10_F15_N_E": "unresolved: event levels are nearly equal; sheet order ties another of "
    "120 permutations (rho 0.984)",
}


# ─── Archive ──────────────────────────────────────────────────────────────────


def file_table(
    meta: Mapping[str, Any], expected_sha256: str = FILE_TABLE_SHA256
) -> tuple[FileEntry, ...]:
    """Pinned (id, name, bytes, md5) entries of the figshare article JSON, sorted by id.

    Fails closed unless the JSON is article 22133411 v3, every supplied md5
    equals figshare's computed md5, and the canonical file listing hashes to
    ``expected_sha256``.
    """
    if (meta.get("id"), meta.get("version")) != (ARTICLE_ID, ARTICLE_VERSION):
        raise ValueError(f"{DATASET}: metadata is article {meta.get('id')} v{meta.get('version')}")
    records = sorted(meta["files"], key=lambda f: f["id"])
    bad = [f["name"] for f in records if f.get("supplied_md5") != f["computed_md5"]]
    if bad:
        raise ValueError(f"{DATASET}: supplied md5 differs from computed md5 for {bad[:5]}")
    listing = "".join("{id}\t{size}\t{computed_md5}\t{name}\n".format(**f) for f in records)
    digest = hashlib.sha256(listing.encode("utf-8")).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"{DATASET}: file table sha256 {digest}, want {expected_sha256}")
    return tuple(
        (int(f["id"]), str(f["name"]), int(f["size"]), str(f["computed_md5"])) for f in records
    )


def member_names(files: tuple[FileEntry, ...]) -> dict[int, str]:
    """figshare id -> ZIP member name (``<id>_<name>`` where names collide)."""
    counts: dict[str, int] = {}
    for _, name, _, _ in files:
        counts[name] = counts.get(name, 0) + 1
    return {fid: (name if counts[name] == 1 else f"{fid}_{name}") for fid, name, _, _ in files}


def _read_member(zf: zipfile.ZipFile, member: str, size: int, md5: str) -> bytes:
    raw = zf.read(member)
    if len(raw) != size or hashlib.md5(raw).hexdigest() != md5:
        raise ValueError(
            f"{DATASET}: {member} is {len(raw)} B md5 {hashlib.md5(raw).hexdigest()}, want {size} B {md5}"
        )
    return raw


def _decode_wav(raw: bytes, where: str) -> np.ndarray:
    """Native mono IEEE float32 WAV -> exact float32 samples (fails closed on anything else)."""
    import soundfile as sf

    with sf.SoundFile(io.BytesIO(raw)) as fh:
        if (fh.samplerate, fh.channels, fh.subtype) != (SAMPLE_RATE, 1, "FLOAT"):
            raise ValueError(
                f"{DATASET}: {where} is {fh.samplerate} Hz x {fh.channels} {fh.subtype}"
            )
        x = fh.read(dtype="float32")
    if not np.isfinite(x).all():
        raise ValueError(f"{DATASET}: {where} holds non-finite samples")
    return np.ascontiguousarray(x)


# ─── Workbook ─────────────────────────────────────────────────────────────────


def _cell_number(value: object, where: str) -> int | float | str:
    """A workbook cell value that ``float``/``int`` can convert, else ``ValueError``."""
    if not isinstance(value, (int, float, str)):
        raise ValueError(f"{DATASET}: {WORKBOOK} {where} is {value!r}, want a number")
    return value


def read_workbook(raw: bytes) -> dict[str, list[dict[str, Any]]]:
    """Sheet ``flyovers`` -> ``{<operator..start prefix>: [row, ...]}`` in sheet order.

    The first six ID tokens (operator, sUAS, HAGL, operation+speed, payload,
    start) key the rows. The direction token is kept per row because the
    workbook and the file names disagree on it for 3p and M3 (``uw`` vs ``dw``).
    """
    import openpyxl

    ws = openpyxl.load_workbook(io.BytesIO(raw), data_only=False)["flyovers"]
    header = [ws.cell(3, c).value for c in range(1, 13)]
    if header[:2] != ["N", "ID"] or header[10:12] != ["Gross weight", "Rotors"]:
        raise ValueError(f"{DATASET}: unexpected {WORKBOOK} flyovers header {header}")
    blocks: list[tuple[str, int]] = []
    metrics: list[str] | None = None
    for m, name in enumerate(MIC_NAMES):
        col = 13 + 6 * m
        if ws.cell(1, col).value != name:
            raise ValueError(
                f"{DATASET}: {WORKBOOK} column {col} is {ws.cell(1, col).value!r}, want {name}"
            )
        names = [str(ws.cell(2, col + j).value) for j in range(6)]
        if metrics is not None and names != metrics:
            raise ValueError(f"{DATASET}: {WORKBOOK} metric columns differ between microphones")
        metrics = names
        blocks.append((name, col))
    assert metrics is not None
    rows: dict[str, list[dict[str, Any]]] = {}
    for r in range(4, ws.max_row + 1):
        rid = ws.cell(r, 2).value
        if not rid:
            continue
        tokens = str(rid).split("_")
        if len(tokens) != 7:
            raise ValueError(f"{DATASET}: {WORKBOOK} row {r} ID {rid!r}")
        sqm = {
            metric: [
                float(_cell_number(ws.cell(r, col + j).value, f"row {r} column {col + j}"))
                for _, col in blocks
            ]
            for j, metric in enumerate(metrics)
        }
        rows.setdefault("_".join(tokens[:6]), []).append(
            {
                "sheet_row": r,
                "N": int(_cell_number(ws.cell(r, 1).value, f"row {r} column 1")),
                "ID": str(rid),
                "direction_token": tokens[6],
                "gross_weight_g": ws.cell(r, 11).value,
                "rotors": ws.cell(r, 12).value,
                "sqm": sqm,
            }
        )
    return rows


# ─── Build ────────────────────────────────────────────────────────────────────


def build(root: Path) -> Iterator[tuple[str, td.Frame]]:
    """Raw dir with ``22133411_v3.zip`` + its figshare JSON -> event / take Frames."""
    root = Path(root)
    files = file_table(json.loads((root / META_NAME).read_text(encoding="utf-8")))
    yield from build_from_zip(root / ZIP_NAME, files, SYNC_EVIDENCE)


def build_from_zip(
    zip_path: Path, files: tuple[FileEntry, ...], sync_evidence: Mapping[str, Mapping[str, Any]]
) -> Iterator[tuple[str, td.Frame]]:
    """Group, verify and decode the archive against the ``files`` table.

    A complete M1..M9 event becomes one multichannel Frame only when
    ``sync_evidence`` holds a measured entry for it (fail closed otherwise);
    an incomplete event yields one mono Frame per published microphone.
    """
    members = member_names(files)
    with zipfile.ZipFile(zip_path) as zf:
        present = {i.filename for i in zf.infolist() if not i.is_dir()}
        if present != set(members.values()):
            raise ValueError(
                f"{DATASET}: ZIP members differ from the file table: missing "
                f"{sorted(set(members.values()) - present)[:5]}, extra {sorted(present - set(members.values()))[:5]}"
            )
        events: dict[tuple[str, int], dict[int, list[FileEntry]]] = {}
        tokens: dict[str, dict[str, str]] = {}
        calibs: list[tuple[int, FileEntry]] = []
        workbook = None
        for fid, name, size, md5 in files:
            m = _EVENT.match(name)
            if m:
                events.setdefault((m["base"], int(m["event"])), {}).setdefault(
                    int(m["mic"]), []
                ).append((fid, name, size, md5))
                tokens[m["base"]] = m.groupdict()
            elif c := _CALIB.match(name):
                calibs.append((int(c["ch"]), (fid, name, size, md5)))
            elif name == WORKBOOK:
                workbook = read_workbook(_read_member(zf, members[fid], size, md5))
            else:
                raise ValueError(f"{DATASET}: unrecognised file {name!r} (id {fid})")

        wb_tables = _workbook_tables(events, tokens, workbook)
        complete = {
            f"{b}/ev{e}"
            for (b, e), mics in events.items()
            if set(mics) == set(range(1, N_MICS + 1))
        }
        if complete != set(sync_evidence):
            raise ValueError(
                f"{DATASET}: complete events without measured sync evidence "
                f"{sorted(complete - set(sync_evidence))}; evidence for absent/incomplete events "
                f"{sorted(set(sync_evidence) - complete)}"
            )
        for base, ev in sorted(events):
            mics = events[(base, ev)]
            tok = tokens[base]
            wb = None
            if base in wb_tables:
                position = sorted(e for b, e in events if b == base).index(ev)
                wb = _workbook_meta(wb_tables[base], position, tok)
            if f"{base}/ev{ev}" in complete:
                key = f"{base}/ev{ev}"
                yield (
                    key,
                    _event_frame(zf, members, key, tok, ev, mics, dict(sync_evidence[key]), wb),
                )
            else:
                for mic in sorted(mics):
                    key = f"{base}/ev{ev}_M{mic}"
                    yield (
                        key,
                        _single_frame(zf, members, key, tok, ev, mic, mics[mic], sorted(mics), wb),
                    )
        for ch, entry in sorted(calibs):
            key = f"calibration/{Path(entry[1]).stem}"
            yield key, _calibration_frame(zf, members, key, ch, entry)


def _workbook_tables(
    events: Mapping[tuple[str, int], Any],
    tokens: Mapping[str, Mapping[str, str]],
    workbook: Mapping[str, list[dict[str, Any]]] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Flyover ID -> all workbook rows of its operator..start prefix (one per event, unassigned)."""
    out: dict[str, list[dict[str, Any]]] = {}
    flyover_bases = sorted({b for b, _ in events if tokens[b]["operation"] == "F"})
    prefixes = ["_".join(b.split("_")[:6]) for b in flyover_bases]
    if len(set(prefixes)) != len(prefixes):
        raise ValueError(f"{DATASET}: flyover IDs differing only in direction: {flyover_bases}")
    for base, prefix in zip(flyover_bases, prefixes, strict=True):
        rows = (workbook or {}).get(prefix)
        evs = sorted(e for b, e in events if b == base)
        if rows is None or len(rows) != len(evs):
            raise ValueError(
                f"{DATASET}: {base} has events {evs} but {WORKBOOK} has "
                f"{0 if rows is None else len(rows)} rows for {prefix}"
            )
        out[base] = rows
    unused = set(workbook or {}) - set(prefixes)
    if unused:
        raise ValueError(f"{DATASET}: {WORKBOOK} rows without published events: {sorted(unused)}")
    return out


def _unique_source(entries: list[FileEntry], where: str) -> tuple[FileEntry, list[int]]:
    """One readable copy of a microphone file + every figshare id carrying it."""
    if len({md5 for _, _, _, md5 in entries}) != 1:
        raise ValueError(f"{DATASET}: {where} has differing duplicate files {entries}")
    entries = sorted(entries)
    return entries[0], [fid for fid, _, _, _ in entries]


def _operating(tok: Mapping[str, str], where: str) -> dict[str, Any]:
    suas = SUAS[tok["suas"]]
    op = tok["operation"]
    rec = {
        "condition": "flight",
        "flight_mode": _FLIGHT_MODE[op],
        "flight_pattern": _PATTERN[op],
        "nominal_speed_m_s": float(tok["speed"]),
        "hagl_m": float(tok["hagl"]),
        "payload_attached": tok["payload"] == "Y",
        "start_side": _START[tok["start"]],
        "direction_token": tok["direction"],
        "n_active_rotors": suas["n_rotors"],
        "contains_rotor_noise": True,
        "external_source": "none",
        "confidence": "high",
    }
    validate_operating(rec, suas["n_rotors"], where)
    return rec


def _system(tok: Mapping[str, str]) -> dict[str, Any]:
    suas = dict(SUAS[tok["suas"]])
    payload = suas.pop("payload_g", None)
    return {
        "category": "drone",
        "suas_code": tok["suas"],
        **suas,
        "payload_g": payload if tok["payload"] == "Y" else 0,
    }


def _tokens_meta(tok: Mapping[str, str]) -> dict[str, Any]:
    return {
        "id": tok["base"],
        "operator": tok["operator"],
        "suas": tok["suas"],
        "hagl_m": int(tok["hagl"]),
        "operation": tok["operation"],
        "speed_m_s": int(tok["speed"]),
        "payload": tok["payload"],
        "starting": tok["start"],
        "direction": tok["direction"],
        "meaning": "workbook flyovers header: Operador, sUAS, HAGL, Operation (F flyover, "
        "H hover), Speed, Payload (N/Y), Starting (W/E/C), Direction (dw/uw/nw; "
        "[INFERENCE] downwind/upwind/no wind, not defined by the publisher)",
    }


def _workbook_meta(
    rows: list[dict[str, Any]], position: int, tok: Mapping[str, str]
) -> dict[str, Any]:
    """Every workbook row of this flyover ID, keyed by sheet row; the event link stays a candidate."""
    prefix = "_".join(tok["base"].split("_")[:6])
    table = {}
    for row in rows:
        rec = {
            "N": row["N"],
            "ID": row["ID"],
            "gross_weight_g": row["gross_weight_g"],
            "rotors": row["rotors"],
            "sqm_per_mic": row["sqm"],
        }
        if row["direction_token"] != tok["direction"]:
            rec["direction_conflict"] = (
                f"workbook ID says {row['direction_token']!r}, file names say {tok['direction']!r}"
            )
        table[str(row["sheet_row"])] = rec
    return {
        "sheet": "flyovers",
        "rows_by_sheet_row": table,
        "event_row": "not published: the workbook has no event column, so per-event "
        "values are not assigned",
        "sheet_order_candidate_row": rows[position]["sheet_row"],
        "sheet_order_status": WORKBOOK_ROW_ORDER.get(prefix, "unverified"),
        "sqm_meaning": _SQM_UNITS,
        "known_defects": "the first row of each sUAS group carries the previous group's "
        "gross weight/rotors (Fp N 4, sheet row 7: 249 g; M3 N 7, sheet row 10: 795 g; "
        "Yn N 12, sheet row 15: 6300 g, 4 rotors)",
    }


def _file_record(fid_entry: FileEntry, ids: list[int], member: str) -> dict[str, Any]:
    fid, name, size, md5 = fid_entry
    rec = {"figshare_id": fid, "name": name, "zip_member": member, "bytes": size, "md5": md5}
    if len(ids) > 1:
        rec["duplicate_figshare_ids"] = ids
        rec["duplicate_note"] = "byte-identical copies of one file; audio read once"
    return rec


def _event_frame(
    zf: zipfile.ZipFile,
    members: Mapping[int, str],
    key: str,
    tok: Mapping[str, str],
    ev: int,
    mics: Mapping[int, list[FileEntry]],
    sync: dict[str, Any],
    workbook: dict[str, Any] | None,
) -> td.Frame:
    channels, records = [], []
    for mic in range(1, N_MICS + 1):
        entry, ids = _unique_source(mics[mic], f"{key} M{mic}")
        member = members[entry[0]]
        channels.append(_decode_wav(_read_member(zf, member, entry[2], entry[3]), member))
        records.append({"mic": MIC_NAMES[mic - 1], **_file_record(entry, ids, member)})
    lengths = {x.size for x in channels}
    if len(lengths) != 1:
        raise ValueError(f"{DATASET}: {key} microphones are not coextensive: {sorted(lengths)}")
    n = lengths.pop()
    audio = np.stack(channels)
    del channels
    where = f"{DATASET} {key}"
    extra: dict[str, Any] = {
        "source": {
            "files": records,
            "event": ev,
            "grouping": "same ID tokens and event number, all nine microphones, equal "
            "length; sample-synchronous per SYNC_EVIDENCE",
            "sync_evidence": sync,
            "sync_method": SYNC_METHOD,
            "archive": f"figshare {ARTICLE_ID} v{ARTICLE_VERSION} {ZIP_NAME}",
        },
        "tokens": _tokens_meta(tok),
        "channel_names": list(MIC_NAMES),
        "channel_angle_deg": list(MIC_ANGLE_DEG),
        "channel_valid_range": [[0, n]] * N_MICS,
        "units": "Pa [INFERENCE: calibrator takes at the same scale read ~1.0 Pa RMS]",
        "native_format": "IEEE float32 mono WAV per microphone, 50 kHz; samples kept exactly",
        "clock": CLOCK,
        "recording_n_samples": n,
        "recording_duration_s": n / SAMPLE_RATE,
        "geometry": GEOMETRY,
        "attribution": PROVENANCE["attribution"],
        "license": PROVENANCE["license"],
    }
    if workbook is not None:
        extra["workbook"] = workbook
    meta = meta_frame(
        key,
        FRAMES_DATASET,
        system=_system(tok),
        observation=_observation(tok, N_MICS),
        operating=_operating(tok, where),
        label=_label(tok),
        extra=extra,
    )
    return td.Frame(
        {
            "audio": td.uniform(audio, SAMPLE_RATE, dims=("mic", "time"), t_start=0.0),
            "mic_pos": td.wrap(MIC_POS.copy(), dims=("mic", None)),
            "meta": meta,
        }
    )


def _observation(tok: Mapping[str, str], n_channels: int) -> dict[str, Any]:
    return {
        "mic_placement": "external",
        "type": "ground_array_static",
        "source_motion": "moving" if tok["operation"] == "F" else "hovering",
        "relative_trajectory": "nominal only (no GPS deposited)",
        "environment": "outdoor field site, Edzell, Scotland (UK)",
        "date": "2022-08-17",
        "n_channels": n_channels,
        "sample_rate": SAMPLE_RATE,
        "sample_format": "IEEE float32 (native, exact)",
    }


def _label(tok: Mapping[str, str]) -> dict[str, Any]:
    pattern = _PATTERN[tok["operation"]]
    return {
        "subcategory": f"{tok['suas'].lower()}_{pattern}",
        "content": f"drone_{pattern}_ground_array_recording",
        "recording_kind": "source_recording",
    }


def _single_frame(
    zf: zipfile.ZipFile,
    members: Mapping[int, str],
    key: str,
    tok: Mapping[str, str],
    ev: int,
    mic: int,
    entries: list[FileEntry],
    published_mics: list[int],
    workbook: dict[str, Any] | None,
) -> td.Frame:
    entry, ids = _unique_source(entries, key)
    member = members[entry[0]]
    x = _decode_wav(_read_member(zf, member, entry[2], entry[3]), member)
    extra: dict[str, Any] = {
        "source": {
            "files": [{"mic": MIC_NAMES[mic - 1], **_file_record(entry, ids, member)}],
            "event": ev,
            "grouping": "not grouped: the deposit publishes only microphones "
            f"{[MIC_NAMES[m - 1] for m in published_mics]} for this event",
            "archive": f"figshare {ARTICLE_ID} v{ARTICLE_VERSION} {ZIP_NAME}",
        },
        "tokens": _tokens_meta(tok),
        "channel_names": [MIC_NAMES[mic - 1]],
        "channel_angle_deg": [MIC_ANGLE_DEG[mic - 1]],
        "channel_valid_range": [[0, int(x.size)]],
        "units": "Pa [INFERENCE: calibrator takes at the same scale read ~1.0 Pa RMS]",
        "native_format": "IEEE float32 mono WAV, 50 kHz; samples kept exactly",
        "clock": "audio t = seconds from the first sample of the published file",
        "recording_n_samples": int(x.size),
        "recording_duration_s": x.size / SAMPLE_RATE,
        "geometry": GEOMETRY,
        "attribution": PROVENANCE["attribution"],
        "license": PROVENANCE["license"],
    }
    if workbook is not None:
        extra["workbook"] = workbook
    meta = meta_frame(
        key,
        FRAMES_DATASET,
        system=_system(tok),
        observation=_observation(tok, 1),
        operating=_operating(tok, f"{DATASET} {key}"),
        label=_label(tok),
        extra=extra,
    )
    return td.Frame(
        {
            "audio": audio_series(x[None, :], SAMPLE_RATE),
            "mic_pos": td.wrap(MIC_POS[mic - 1 : mic].copy(), dims=("mic", None)),
            "meta": meta,
        }
    )


def _calibration_frame(
    zf: zipfile.ZipFile, members: Mapping[int, str], key: str, ch: int, entry: FileEntry
) -> td.Frame:
    member = members[entry[0]]
    x = _decode_wav(_read_member(zf, member, entry[2], entry[3]), member)
    operating = {
        "condition": "rotors_off",
        "n_active_rotors": 0,
        "contains_rotor_noise": False,
        "external_source": "other",
        "confidence": "high",
    }
    validate_operating(operating, 0, f"{DATASET} {key}")
    meta = meta_frame(
        key,
        FRAMES_DATASET,
        system={"category": "calibration", "device": "acoustic calibrator (model not published)"},
        observation={
            "mic_placement": "external",
            "type": "calibrator_coupler",
            "environment": "field calibration, Edzell, Scotland (UK)",
            "n_channels": 1,
            "sample_rate": SAMPLE_RATE,
            "sample_format": "IEEE float32 (native, exact)",
        },
        operating=operating,
        label={
            "subcategory": "calibrator_tone",
            "content": "microphone_calibration",
            "recording_kind": "calibration",
        },
        extra={
            "source": {
                "files": [_file_record(entry, [entry[0]], member)],
                "grouping": "not grouped: one calibrator take per DAQ channel, recorded "
                "one after another with different lengths; not simultaneous",
                "archive": f"figshare {ARTICLE_ID} v{ARTICLE_VERSION} {ZIP_NAME}",
            },
            "daq_channel": ch,
            "presumed_mic": f"M{ch} [INFERENCE: matching numbering; not stated]",
            "observed": "a steady tone reading 1038 Hz at the 50 kHz header rate, ~1.0 Pa RMS "
            "(94 dB) in quiet seconds, with handling transients (calibrator nominal "
            "level/frequency not published)",
            "channel_valid_range": [[0, int(x.size)]],
            "clock": "audio t = seconds from the first sample of the published file",
            "recording_n_samples": int(x.size),
            "recording_duration_s": x.size / SAMPLE_RATE,
            "attribution": PROVENANCE["attribution"],
            "license": PROVENANCE["license"],
        },
    )
    return td.Frame({"audio": audio_series(x[None, :], SAMPLE_RATE), "meta": meta})
