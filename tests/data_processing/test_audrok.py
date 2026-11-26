"""AuDroK chamber take assembly: clock blocks and per-row valid spans."""

from __future__ import annotations

import numpy as np
import pytest

from data_processing.sources.audrok import CHAMBER_CHANNELS, CHAMBER_UNEQUAL, chamber_blocks

SR = 44100


def _rows(take: str) -> dict[str, np.ndarray]:
    lengths = CHAMBER_UNEQUAL[take]["samples"]
    return {c: np.full(lengths[c], k + 1, dtype=np.float32) for k, c in enumerate(CHAMBER_CHANNELS)}


def test_04_ld_keeps_unplaced_block_on_its_own_grid() -> None:
    entries, meta = chamber_blocks("04-LD", _rows("04-LD"), SR)

    assert set(entries) == {"audio", "mic_pos", "additional_audio", "additional_mic_pos"}
    audio, extra = entries["audio"], entries["additional_audio"]
    assert audio.dims == ("mic", "time") and extra.dims == ("additional_mic", "time")
    assert entries["additional_mic_pos"].dims == ("additional_mic", None)
    assert audio.data.shape == (3, 687872) and extra.data.shape == (3, 688128)
    assert meta["clock_blocks"] == ["audio", "additional_audio"]
    assert meta["channel_labels"] == ["Mi1", "Mi2", "Mi3"]
    assert meta["additional_audio_labels"] == ["Mi4", "Mi5", "Mi6"]
    assert meta["channel_start_samples"] == [0, 0, 0]
    assert meta["channel_valid_samples"] == [687616, 687616, 687872]
    assert meta["additional_audio_start_samples"] == [0, 0, 0]
    assert meta["additional_audio_valid_samples"] == [688128] * 3
    assert "unverified" in meta["channel_alignment_confidence"]
    # Mi1/Mi2 are NaN only over Mi3's extra tail; nothing else is padded or cropped.
    assert np.isnan(audio.data[:2, 687616:]).all() and not np.isnan(audio.data[:2, :687616]).any()
    assert (audio.data[2] == 3).all() and not np.isnan(extra.data).any()
    assert (extra.data == np.array([[4], [5], [6]], dtype=np.float32)).all()


def test_common_end_places_short_rows_late() -> None:
    entries, meta = chamber_blocks("40-H2", _rows("40-H2"), SR)

    assert set(entries) == {"audio", "mic_pos"}
    data = entries["audio"].data
    assert data.shape == (6, 344064)
    assert meta["channel_start_samples"] == [256, 256, 256, 256, 0, 0]
    assert meta["channel_valid_samples"] == [344064] * 6
    assert np.isnan(data[:4, :256]).all() and not np.isnan(data[:4, 256:]).any()


def test_unaudited_unequal_take_raises() -> None:
    rows = {
        c: np.zeros(1024 + (256 if c == "Mi6" else 0), dtype=np.float32) for c in CHAMBER_CHANNELS
    }
    with pytest.raises(ValueError, match="unaudited"):
        chamber_blocks("01-TO", rows, SR)
