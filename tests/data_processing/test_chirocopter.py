"""Regression for partitioning iOSD rows across audio and independent logs."""

import numpy as np
import tdseries as td

from data_processing.sources.aerial_bioacoustics import (
    _chiro_log_frame,
    _iosd_tracks,
    _iosd_window,
)
from data_processing.sources.iosd_dat import _CONTROL_TRACK, _GPS_TRACK, _OSD_TRACK, IosdTelemetry
from data_processing.streams import frame_to_sample, sample_to_frame


def test_iosd_rows_partition_across_audio_and_native_clock() -> None:
    osd = np.zeros(6, dtype=_OSD_TRACK)
    osd["record_index"] = [100, 200, 300, 400, 500, 600]
    osd["sample"] = np.arange(6)
    osd["seq"] = [10, 11, 12, 13, 14, 15]
    osd["baro_height_m"] = [0.0, 1.5, 5.0, 10.5, 8.0, 2.0]
    gps = np.zeros(3, dtype=_GPS_TRACK)
    gps["record_index"] = [50, 250, 650]  # before, inside, after the OSD packets
    unplaced = np.zeros(1, dtype=np.dtype([f for f in _OSD_TRACK.descr if f[0] != "sample"]))
    unplaced["record_index"], unplaced["seq"], unplaced["baro_height_m"] = 450, 12, 9.5
    log = IosdTelemetry(osd, gps, np.empty(0, _CONTROL_TRACK), unplaced, 0, 0)
    tracks = _iosd_tracks(log, seconds_per_sample=0.5)

    pcm = np.array([0.125, -0.5, 0.25], dtype=np.float32)
    audio = td.Frame({"audio": td.uniform(pcm, 2, dims=("time",), t_start=12.0)})
    start, end = audio.t_start_ticks, audio.t_end_ticks
    offset = round(11.5 * td.TICKS_PER_SECOND)
    window = _iosd_window(tracks, start, end, offset)
    merged = audio.with_entry("iosd", td.Frame(window), expand=False)
    decoded = sample_to_frame(frame_to_sample(merged))
    assert (decoded.t_start_ticks, decoded.t_end_ticks) == (start, end)
    assert decoded["audio"].tindex == audio["audio"].tindex
    np.testing.assert_array_equal(decoded["audio"].data, pcm)
    overlap = decoded["iosd"]["osd"]
    assert isinstance(overlap.tindex, td.StampIndex)
    np.testing.assert_array_equal(overlap.tindex.timestamps, [12.0, 12.5, 13.0])
    np.testing.assert_array_equal(overlap.data, osd[1:4])

    log_frame = _chiro_log_frame(
        tracks,
        {"recording": "survey.wav"},
        {},
        (start - offset, end - offset),
        "survey/IOSD/FLY095.DAT",
        0,
        "survey.zip",
        "survey",
        ("survey.wav",),
        {},
    )
    assert log_frame is not None
    key, frame = log_frame
    assert key == "survey__FLY095"
    tail = sample_to_frame(frame_to_sample(frame))
    assert "audio" not in tail
    sections = tail["iosd"]
    assert set(sections) == {"before_verified_overlap", "after_verified_overlap", "unstamped"}
    assert list(tail["meta"]["iosd_sections"]) == list(sections)
    before = sections["before_verified_overlap"]["osd"]
    after = sections["after_verified_overlap"]["osd"]
    assert isinstance(before.tindex, td.StampIndex) and isinstance(after.tindex, td.StampIndex)
    np.testing.assert_array_equal(before.tindex.timestamps, [0.0])
    np.testing.assert_array_equal(after.tindex.timestamps, [2.0, 2.5])
    # The row at exactly the audio end is native-only. Every OSD row appears once.
    np.testing.assert_array_equal(np.concatenate((before.data, overlap.data, after.data)), osd)
    unstamped = sections["unstamped"]
    assert set(unstamped) == {"gps_before_first_osd", "gps_after_last_osd", "osd_unplaced"}
    assert not unstamped["osd_unplaced"].has_time
    # GPS rows outside the bracketing OSD packets are unstamped; the one between
    # them is interpolated into the audio window. Each row appears once.
    assert "gps" not in sections["before_verified_overlap"]
    np.testing.assert_array_equal(decoded["iosd"]["gps"].data, gps[1:2])
    np.testing.assert_array_equal(unstamped["gps_before_first_osd"].data, gps[:1])
    np.testing.assert_array_equal(unstamped["gps_after_last_osd"].data, gps[2:])
