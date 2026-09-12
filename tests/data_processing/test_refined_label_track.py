"""The published ``rps_refined`` track: defined everywhere, anchored on audio."""

from __future__ import annotations

import numpy as np
import pytest
import tdseries as td

from data_processing.refined_label_track import (
    REFINED_KEY,
    attach_refined,
    refined_series,
)


def _sidecar(tmp_path, rid="REC", *, n=50, dt=0.032, t0=0.25, value=70.0):
    ft = np.arange(n) * dt + t0
    np.savez(
        tmp_path / f"{rid}.npz",
        ft=ft,
        t0_offset_s=np.float64(t0),
        r_telemetry=np.full((4, n), value, dtype=np.float64),
        r_refined=np.full((4, n), value - 0.5, dtype=np.float64),
    )
    return ft


def _frame(audio_t_start: float, tel_t_start: float, key: str = "motors_measured"):
    audio = td.uniform(
        np.zeros((8, 16000), np.float32), 16000, dims=("mic", "time"), t_start=audio_t_start
    )
    tel = td.events(
        np.arange(200) * 0.01 + tel_t_start,
        np.full((4, 200), 70.0, np.float32),
        dims=("rotor", "time"),
        t_start=tel_t_start,
    )
    return td.Frame({"audio": audio, key: tel})


def test_anchored_on_audio_not_telemetry(tmp_path):
    """The sidecar clock is the AUDIO clock; anchoring on telemetry double-shifts."""
    ft = _sidecar(tmp_path, "REC", t0=0.25)
    # telemetry starts 0.25 s after the audio, the same as the sidecar's offset
    frame = _frame(audio_t_start=100.0, tel_t_start=100.25)
    out = attach_refined(frame, "REC", label_dir=tmp_path)
    stamps = np.asarray(out[REFINED_KEY].tindex.abs_stamps, dtype=np.float64)
    assert stamps[0] == pytest.approx(100.0 + ft[0])
    # anchoring on the telemetry track would have put it at 100.50
    assert stamps[0] == pytest.approx(100.25)


def test_missing_sidecar_publishes_the_reference_track(tmp_path):
    frame = _frame(audio_t_start=0.0, tel_t_start=0.0, key="motors_command")
    out = attach_refined(frame, "NO_SUCH_RECORDING", label_dir=tmp_path)
    assert REFINED_KEY in out
    assert np.array_equal(
        np.asarray(out[REFINED_KEY].data), np.asarray(frame["motors_command"].data)
    )


def test_frame_without_any_label_is_untouched(tmp_path):
    bench = td.Frame(
        {"audio": td.uniform(np.zeros((1, 100), np.float32), 16000, dims=("mic", "time"))}
    )
    assert REFINED_KEY not in attach_refined(bench, "motor_Motor1_50", label_dir=tmp_path)


def test_label_without_audio_is_an_error(tmp_path):
    _sidecar(tmp_path, "REC")
    tel = td.events(
        np.arange(10) * 0.01, np.full((4, 10), 70.0, np.float32), dims=("rotor", "time")
    )
    with pytest.raises(ValueError, match="anchor"):
        attach_refined(td.Frame({"motors_measured": tel}), "REC", label_dir=tmp_path)


def test_shape_mismatch_is_rejected(tmp_path):
    ft = np.arange(10) * 0.032
    np.savez(
        tmp_path / "BAD.npz",
        ft=ft,
        t0_offset_s=np.float64(0.0),
        r_telemetry=np.full((4, 10), 70.0),
        r_refined=np.full((4, 7), 70.0),
    )
    ref = td.events(ft, np.full((4, 10), 70.0, np.float32), dims=("rotor", "time"))
    with pytest.raises(ValueError, match="does not match"):
        refined_series("BAD", ref, audio_t_start=0.0, label_dir=tmp_path)
