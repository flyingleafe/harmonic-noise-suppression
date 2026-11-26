"""IDMT-Berne: publisher segments rejoin into contiguous runs only when every test holds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from data_processing.sources import idmt_berne

FLIGHT = "2022-08-03T14-36-48__Dronen_records_Mavic_static_flights_01"
SR = idmt_berne.SAMPLE_RATE
SEG = 4800  # 0.1 s segments keep the fixture small
ROWS = 10  # CSV rows at 0.01 s


def _write_segment(root: Path, part: int, seg: int, audio: np.ndarray, xy0: np.ndarray) -> None:
    name = f"{FLIGHT}_{part}_{seg}"
    d = root / "MicArrayData30s" / name
    d.mkdir(parents=True)
    sf.write(d / f"{name}.wav", audio.T, SR, subtype="PCM_16")
    t = np.arange(ROWS) * 0.01
    x = xy0[0] + 0.01 * np.arange(ROWS)
    y = np.full(ROWS, xy0[1])
    azi = np.degrees(np.arctan2(y, x)) % 360
    rows = np.column_stack([t, x, y, np.zeros(ROWS), np.full(ROWS, 90.0), azi])
    with open(d / "DOATruePos.csv", "w", encoding="utf-8") as fh:
        fh.write("time,X,Y,Z,ele,azi\n")
        np.savetxt(fh, rows, delimiter=",", fmt="%.6f")


def test_runs_join_contiguous_segments_and_split_parts(tmp_path: Path) -> None:
    n = np.arange(3 * SEG)
    phase = np.arange(idmt_berne.N_MICS)[:, None]
    signal = 0.25 * np.sin(2 * np.pi * 200 * n / SR + phase)
    signal = np.round(signal * 32768) / 32768
    # part 0: two segments cut from one continuous signal, GPS continuous
    _write_segment(tmp_path, 0, 0, signal[:, :SEG], np.array([10.0, 5.0]))
    _write_segment(tmp_path, 0, 1, signal[:, SEG : 2 * SEG], np.array([10.095, 5.0]))
    # part 1: a later piece; the drone is 30 m away from where part 0 ended
    _write_segment(tmp_path, 1, 0, signal[:, 2 * SEG :], np.array([40.0, 5.0]))

    frames = dict(idmt_berne.build(tmp_path))
    assert sorted(frames) == [f"{FLIGHT}/0_0-0_1", f"{FLIGHT}/1_0-1_0"]

    run = frames[f"{FLIGHT}/0_0-0_1"]
    audio = np.asarray(run["audio"].data)
    np.testing.assert_array_equal(audio, signal[:, : 2 * SEG].astype(np.float32))
    source = run["meta"]["source"]
    assert [s["start_sample"] for s in source["segments"]] == [0, SEG]
    (inner,) = source["boundaries"]
    assert inner["joined"] and inner["clean_audio_seam"] and inner["gps_continuous"]
    after = source["boundary_after"]
    assert not after["joined"] and set(after["failed_tests"]) >= {"same_part", "gps_continuous"}

    # trajectory rows keep their values; the second segment's stamps sit at its audio start
    t = np.asarray(run["drone_pos"].tindex.timestamps)
    x = np.asarray(run["drone_pos"].data)[0]
    assert x.shape == t.shape == (2 * ROWS,)
    np.testing.assert_allclose(x[ROWS], 10.095)
    np.testing.assert_allclose(t[ROWS], SEG / SR)
