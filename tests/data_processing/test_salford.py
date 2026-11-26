"""Salford DroneNoise v3: one Frame per synchronous nine-microphone event, nothing else joined."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from data_processing.derivations import _acoustic_key
from data_processing.sources import salford
from data_processing.streams import frame_to_sample, sample_to_frame

SR = salford.SAMPLE_RATE
N = 400  # samples per fixture file
FLY = "Ed_M3_10_F15_N_E_dw"
HOVER = "Ed_3p_10_H00_N_C_nw"
METRICS = ["N5", "S5", "R5", "F5", "I5", "T5"]


def _wav(x: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x.astype(np.float32), SR, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def _workbook() -> bytes:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None  # a new Workbook always has one active sheet
    ws.title = "flyovers"
    header = [
        "N",
        "ID",
        "Operador",
        "sUAS",
        "HAGL",
        "Operation",
        "Speed",
        "Payload",
        "Starting",
        "Direction",
        "Gross weight",
        "Rotors",
    ]
    for c, h in enumerate(header, start=1):
        ws.cell(3, c, h)
    ws.cell(4, 1, 1)
    ws.cell(4, 2, "Ed_M3_10_F15_N_E_uw")  # the real workbook says uw where file names say dw
    ws.cell(4, 11, 6300)
    ws.cell(4, 12, 4)
    for m in range(9):
        col = 13 + 6 * m
        ws.cell(1, col, f"M{m + 1}")
        for j, metric in enumerate(METRICS):
            ws.cell(2, col + j, metric)
            ws.cell(4, col + j, 10.0 * (j + 1) + m)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _fixture(tmp_path: Path) -> tuple[Path, tuple, dict[str, np.ndarray]]:
    rng = np.random.default_rng(0)
    payloads: list[tuple[int, str, bytes]] = []
    signals: dict[str, np.ndarray] = {}
    fid = 1000
    for base in (FLY, HOVER):
        for m in range(1, 10):
            x = rng.standard_normal(N).astype(np.float32) * 0.01
            signals[f"{base}_ev1_M{m}"] = x
            payloads.append((fid, f"{base}_ev1_M{m}.wav", _wav(x)))
            fid += 1
    # the hover M5 is published twice under two ids with identical bytes
    payloads.append((fid, f"{HOVER}_ev1_M5.wav", _wav(signals[f"{HOVER}_ev1_M5"])))
    fid += 1
    lone = rng.standard_normal(N).astype(np.float32)
    signals[f"{HOVER}_ev2_M5"] = lone
    payloads.append((fid, f"{HOVER}_ev2_M5.wav", _wav(lone)))
    fid += 1
    tone = np.sin(2 * np.pi * 1000 * np.arange(2 * N) / SR).astype(np.float32)
    signals["calib1"] = tone
    payloads.append((fid, "Calib_Scotland_ch1.wav", _wav(tone)))
    fid += 1
    payloads.append((fid, salford.WORKBOOK, _workbook()))

    files = tuple(sorted((i, n, len(b), hashlib.md5(b).hexdigest()) for i, n, b in payloads))
    members = salford.member_names(files)
    zpath = tmp_path / salford.ZIP_NAME
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as zf:
        for i, _, b in payloads:
            zf.writestr(members[i], b)
    return zpath, files, signals


def _evidence() -> dict[str, dict]:
    return {f"{FLY}/ev1": {"windows_fit": "1/1"}, f"{HOVER}/ev1": {"windows_fit": "1/1"}}


def test_events_group_once_and_everything_else_stays_separate(tmp_path: Path) -> None:
    zpath, files, signals = _fixture(tmp_path)
    frames = dict(salford.build_from_zip(zpath, files, _evidence()))
    assert set(frames) == {
        f"{FLY}/ev1",
        f"{HOVER}/ev1",
        f"{HOVER}/ev2_M5",
        "calibration/Calib_Scotland_ch1",
    }

    fly = frames[f"{FLY}/ev1"]
    audio = np.asarray(fly["audio"].data)
    assert fly["audio"].dims == ("mic", "time") and audio.dtype == np.float32
    np.testing.assert_array_equal(
        audio, np.stack([signals[f"{FLY}_ev1_M{m}"] for m in range(1, 10)])
    )
    np.testing.assert_allclose(np.asarray(fly["mic_pos"].data)[:, 0], salford.MIC_POS[:, 0])
    meta = fly["meta"]
    assert (
        meta["operating"]["flight_mode"] == "manoeuvre"
        and meta["operating"]["n_active_rotors"] == 4
    )
    assert meta["channel_names"] == list(salford.MIC_NAMES)
    assert meta["channel_valid_range"] == [[0, N]] * 9
    wb = meta["workbook"]
    (row,) = wb["rows_by_sheet_row"].values()  # kept by sheet row, not asserted as the event's
    assert row["sqm_per_mic"]["S5"] == [20.0 + m for m in range(9)]
    assert "uw" in row["direction_conflict"] and "dw" in row["direction_conflict"]
    assert wb["sheet_order_candidate_row"] == 4
    assert wb["sheet_order_status"].startswith("unresolved")

    hover = frames[f"{HOVER}/ev1"]
    (m5,) = [r for r in hover["meta"]["source"]["files"] if r["mic"] == "M5"]
    assert len(m5["duplicate_figshare_ids"]) == 2
    assert "workbook" not in hover["meta"]

    lone = frames[f"{HOVER}/ev2_M5"]
    assert lone["audio"].dims == ("time",)
    np.testing.assert_array_equal(np.asarray(lone["audio"].data), signals[f"{HOVER}_ev2_M5"])
    assert "M5" in lone["meta"]["source"]["grouping"]
    np.testing.assert_array_equal(np.asarray(lone["mic_pos"].data), salford.MIC_POS[4:5])

    cal = frames["calibration/Calib_Scotland_ch1"]
    assert cal["meta"]["operating"]["condition"] == "rotors_off"
    assert "mic_pos" not in cal  # calibrator location is unknown
    assert cal["meta"]["label"]["recording_kind"] == "calibration"

    for key, frame in frames.items():
        _acoustic_key(key, frame)  # placement/condition/subcategory labels are valid
        back = sample_to_frame(frame_to_sample(frame))
        np.testing.assert_array_equal(
            np.asarray(back["audio"].data), np.asarray(frame["audio"].data)
        )


def test_grouping_and_identity_fail_closed(tmp_path: Path) -> None:
    zpath, files, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="sync evidence"):
        list(salford.build_from_zip(zpath, files, {f"{FLY}/ev1": {}}))
    with pytest.raises(ValueError, match="sync evidence"):
        list(salford.build_from_zip(zpath, files, {**_evidence(), f"{HOVER}/ev2": {}}))
    tampered = tuple(
        (i, n, s, "0" * 32 if n.endswith("_ev1_M3.wav") else h) for i, n, s, h in files
    )
    with pytest.raises(ValueError, match="md5"):
        list(salford.build_from_zip(zpath, tampered, _evidence()))


def test_unequal_microphone_lengths_are_not_grouped(tmp_path: Path) -> None:
    zpath, files, _ = _fixture(tmp_path)
    members = salford.member_names(files)
    short = _wav(np.zeros(N - 1, np.float32))
    (fid,) = [i for i, n, _, _ in files if n == f"{FLY}_ev1_M9.wav"]
    files = tuple(
        (i, n, len(short), hashlib.md5(short).hexdigest()) if i == fid else (i, n, s, h)
        for i, n, s, h in files
    )
    with zipfile.ZipFile(zpath) as zf:
        blobs = {i.filename: zf.read(i.filename) for i in zf.infolist()}
    blobs[members[fid]] = short
    with zipfile.ZipFile(zpath, "w") as zf:
        for name, b in blobs.items():
            zf.writestr(name, b)
    with pytest.raises(ValueError, match="coextensive"):
        list(salford.build_from_zip(zpath, files, _evidence()))


def test_file_table_pins_the_versioned_json() -> None:
    records = [
        {"id": 2, "name": "b.wav", "size": 10, "computed_md5": "x", "supplied_md5": "x"},
        {"id": 1, "name": "a.wav", "size": 5, "computed_md5": "y", "supplied_md5": "y"},
    ]
    listing = "1\t5\ty\ta.wav\n2\t10\tx\tb.wav\n"
    digest = hashlib.sha256(listing.encode()).hexdigest()
    meta = {"id": salford.ARTICLE_ID, "version": salford.ARTICLE_VERSION, "files": records}
    assert salford.file_table(meta, digest) == ((1, "a.wav", 5, "y"), (2, "b.wav", 10, "x"))
    with pytest.raises(ValueError, match="sha256"):
        salford.file_table(meta)
    with pytest.raises(ValueError, match="article"):
        salford.file_table({**meta, "version": 2}, digest)


def test_nominal_geometry_follows_the_paper() -> None:
    x = salford.MIC_POS[:, 0]
    np.testing.assert_allclose(x, -x[::-1], atol=1e-12)
    np.testing.assert_allclose(x[[4, 5, 6, 7, 8]], [0.0, 2.6795, 5.7735, 10.0, 17.3205], atol=1e-4)
    assert set(salford.SYNC_EVIDENCE) and all(k.count("/ev") == 1 for k in salford.SYNC_EVIDENCE)
