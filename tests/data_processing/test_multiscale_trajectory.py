import numpy as np
import pytest
import tdseries as td

from data_processing import streams
from data_processing.recording_annotations.trajectories import TRACK_STATES, attach_track


def test_source_clock_and_track_identity_survive_codec():
    conditions = {
        "key": "flight",
        "n_active": 2,
        "annotation_release": {"parent_dataset": "X", "parent_version": "v"},
    }
    audio = td.uniform(np.arange(40, dtype="f4"), 10, dims=("time",), t_start=5.0)
    frame = td.Frame({"audio": audio, "meta": td.Frame({"recording_conditions": conditions})})
    row = {
        "dataset": "X",
        "parent_version": "v",
        "key": "flight",
        "role": "audio",
        "expected": 2,
        "sample_rate_hz": 10,
        "audio_t_start_s": 5.0,
        "duration_s": 4.0,
        "estimator_sha256": "model",
        "review_status": "experimental",
        "track": {
            "times_s": [0.5, 1.5, 2.5, 3.5],
            "rates_rps": [[60.0, 70.0], [60.1, 69.9], [None, None], [None, 72.0]],
            "provenance": [
                ["measured", "track_supported"],
                ["interpolated", "measured"],
                ["inactive", "inactive"],
                ["unresolved", "measured"],
            ],
            "uncertainty_rps": [[0.1, 0.2], [0.4, 0.1], [None, None], [None, 0.1]],
            "track_id": [[11, 12], [11, 12], [None, None], [None, 13]],
        },
    }
    attached = attach_track(frame, row)
    got = streams.sample_to_frame(streams.frame_to_sample(attached))
    np.testing.assert_allclose(got["rps_multiscale"].tindex.abs_stamps, [5.5, 6.5, 7.5, 8.5])
    np.testing.assert_array_equal(got["audio"].data, audio.data)
    assert got["rps_multiscale_provenance"].data[0, 1] == TRACK_STATES["interpolated"]
    assert np.isnan(got["rps_multiscale"].data[0, 3])
    np.testing.assert_array_equal(got["rps_multiscale_track_id"].data[1], [1, 1, -1, 2])
    assert got["meta"]["multiscale_track"]["track_id_labels"] == [11, 12, 13]
    with pytest.raises(ValueError, match="not match the parent"):
        attach_track(frame, dict(row, audio_t_start_s=4.5))
    with pytest.raises(ValueError, match="multiscale track is for"):
        attach_track(frame, dict(row, parent_version="another"))
    with pytest.raises(ValueError, match="support mask disagree"):
        attach_track(
            frame,
            dict(
                row,
                track={
                    **row["track"],
                    "rates_rps": [[60.0, 70.0], [60.1, 69.9], [66.0, None], [None, 72.0]],
                },
            ),
        )
