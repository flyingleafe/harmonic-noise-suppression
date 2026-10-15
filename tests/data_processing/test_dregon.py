"""Tests for `data_processing.sources.dregon` — DREGON dataset loader."""

from __future__ import annotations

import numpy as np
import pytest

# ── _parse_mic_positions_txt ────────────────────────────────────────────


def test_parse_mic_positions_txt_valid(tmp_path):
    from data_processing.sources.dregon import _parse_mic_positions_txt

    content = """\
micPositions = [
1.0 2.0 3.0;
4.0 5.0 6.0;
7.0 8.0 9.0;
10.0 11.0 12.0;
13.0 14.0 15.0;
16.0 17.0 18.0;
19.0 20.0 21.0;
22.0 23.0 24.0
];
"""
    p = tmp_path / "micPos.txt"
    p.write_text(content)
    arr = _parse_mic_positions_txt(p)
    assert arr.shape == (8, 3)


def test_parse_mic_positions_txt_no_matrix_raises(tmp_path):
    from data_processing.sources.dregon import _parse_mic_positions_txt

    p = tmp_path / "micPos.txt"
    p.write_text("garbage content without matrix")
    with pytest.raises(ValueError, match="no matrix"):
        _parse_mic_positions_txt(p)


# ── clean_command_spikes ────────────────────────────────────────────────


def test_clean_command_spikes_removes_spikes():
    from data_processing.sources.dregon import clean_command_spikes

    signal = np.ones((4, 100), dtype=np.float64)
    signal[:, 50] = 100.0  # spike across all rotors
    cleaned = clean_command_spikes(signal, kernel=5)
    # Spike should be reduced by median filter
    assert cleaned[0, 50] < 50.0


def test_clean_command_spikes_noop_on_clean():
    from data_processing.sources.dregon import clean_command_spikes

    signal = np.ones((4, 100), dtype=np.float64)
    cleaned = clean_command_spikes(signal, kernel=5)
    np.testing.assert_allclose(cleaned, signal)


# ── _download_file (mocked) ──────────────────────────────────────────────


def test_download_file_skips_existing(tmp_path):
    from data_processing.sources.dregon import _download_file

    dest = tmp_path / "test.wav"
    dest.write_text("dummy")
    result = _download_file("http://example.com/file.wav", dest)
    assert result == dest


# ── discover_recordings ──────────────────────────────────────────────────


def test_discover_recordings_finds_dirs(tmp_path):
    from data_processing.sources.dregon import discover_recordings

    # Create a minimal DREGON mirror
    rec_dir = tmp_path / "free-flight_speech-high_room1"
    rec_dir.mkdir()
    (rec_dir / "micPos.txt").write_text("micPositions = [1 2 3;];")
    (rec_dir / "rotorsPos.mat").write_bytes(b"dummy")  # just to exist

    recordings = discover_recordings(tmp_path)
    assert isinstance(recordings, list)


def test_test_split_recordings_are_discovered_as_test_and_nowhere_else(tmp_path):
    """A held-out recording must surface ONLY under split "test", so no
    split-based selection (in_flight_source, in_flight_noise, ...) reaches it."""
    from data_processing.sources.dregon import (
        SPLIT_RECORDINGS,
        TEST_RECORDING_IDS,
        discover_recordings,
    )

    for rid in [*TEST_RECORDING_IDS, "free-flight_speech-low_room1"]:
        rec_dir = tmp_path / f"DREGON_{rid}"
        rec_dir.mkdir()
        (rec_dir / f"DREGON_{rid}.wav").write_bytes(b"")
    split_of = {s["recording_id"]: s["split"] for s in discover_recordings(tmp_path)}
    assert {rid: split_of[rid] for rid in TEST_RECORDING_IDS} == dict.fromkeys(
        TEST_RECORDING_IDS, "test"
    )
    assert split_of["free-flight_speech-low_room1"] == "in_flight_source"
    others = [r for name, ids in SPLIT_RECORDINGS.items() if name != "test" for r in ids]
    assert not set(others) & set(TEST_RECORDING_IDS)
