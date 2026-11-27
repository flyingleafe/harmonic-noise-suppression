"""Regime inference uses only documented, in-audio shaft telemetry."""

import io

import numpy as np
import pytest

from data_processing.recording_annotations.measured_regime import (
    assess,
    from_scan,
    reference_windows,
)
from data_processing.recording_annotations.scan import SampleScan, SeriesRef

_PROVENANCE = {
    "source": "test shaft logger",
    "basis": "measured",
    "expected_rotors": 4,
    "authoritative": True,
}


def _track():
    t = np.arange(-100, 600) / 50
    x = np.broadcast_to(np.array([51.0, 52.0, 53.0, 55.0])[:, None], (4, t.size)).copy()
    return t, x


def _assess(x, t):
    return assess(x, t, audio_start_s=0, audio_end_s=10, role="audio", provenance=_PROVENANCE)


def test_steady_drift_and_transitions_with_out_of_audio_rows():
    t, x = _track()
    x += 0.07 * t
    x[:, (t < 0) | (t >= 10)] = 0
    got = _assess(x, t)
    assert got["suggested"] == "constant"
    assert got["strong"]
    assert got["spans_s"][0][0] == 0
    assert got["spans_s"][-1][1] < 10
    x[1, (t > 4) & (t < 9)] += 12
    got = _assess(x, t)
    assert got["suggested"] == "mixed"
    assert got["shafts"][1]["variable_s"] > 0
    assert got["shafts"][0]["variable_s"] == 0


def test_variable_and_proven_zero_are_distinct():
    t, x = _track()
    assert _assess(x + 2 * t, t)["suggested"] == "variable"
    assert _assess(np.zeros_like(x), t)["suggested"] == "off"
    x[:] = np.nan
    assert _assess(x, t)["suggested"] == "unknown"


def test_refuses_incomplete_reversed_and_gapped_tracks():
    t, x = _track()
    assert _assess(x[:1], t)["reason"] == "incomplete_rotor_set"
    assert _assess(x, t[::-1])["reason"] == "unordered_timestamps"
    keep = (t < 2) | (t > 8)
    got = _assess(x[:, keep], t[keep])
    assert got["reason"] == "sparse_joint_coverage"
    assert got["coverage"] < 0.4
    assert not got["strong"]


def test_duplicate_stamps_never_add_coverage_or_interpolate_conflicts():
    t, x = _track()
    timestamps = np.insert(t, 201, t[200])
    values = np.insert(x, 201, x[:, 200], axis=1)
    same = _assess(values, timestamps)
    assert same["suggested"] == _assess(x, t)["suggested"]
    assert same["coverage"] == _assess(x, t)["coverage"]
    assert same["duplicate_rows"] == 1 and same["disputed_timestamps"] == 0
    values[0, 201] = 100
    disputed = _assess(values, timestamps)
    assert disputed["disputed_timestamps"] == 1
    assert disputed["coverage"] < same["coverage"]
    assert disputed["shafts"][1]["coverage"] == same["shafts"][1]["coverage"]


def test_bursty_high_rate_logger_still_supports_plateau():
    # 90 ms bursts at 1 kHz with an 11 ms interval between bursts. A gate
    # based only on 4 × the median interval rejects every 1.5 s window.
    t = np.arange(10_000) / 1000
    t = t[np.arange(t.size) % 100 < 90]
    x = np.broadcast_to(np.array([58.0, 59.0, 60.0, 62.0])[:, None], (4, t.size))
    got = _assess(x, t)
    assert got["suggested"] == "constant" and got["strong"]


class _Ranges:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def read(self, offset, length):
        self.calls.append((offset, length))
        return self.payload[offset : offset + length]


@pytest.mark.parametrize("fortran,stamped", [(False, False), (True, True)])
def test_exact_scan_accepts_only_native_frames_and_in_audio_values(fortran, stamped):
    t, x = _track()
    x = (x * 2).astype("<f8")
    start, stop = np.searchsorted(t, [0, 10])
    raw = x.tobytes(order="F" if fortran else "C")
    offset = 21
    ticks = io.BytesIO()
    if stamped:
        np.save(ticks, np.rint((t - t[0]) * 1e9).astype(np.int64), allow_pickle=False)
    shard = _Ranges(b"\0" * offset + raw + ticks.getvalue())
    path = "telemetry/bpf_label"
    shaft_ref = SeriesRef(
        path,
        path,
        "npy",
        x.shape,
        x.dtype.str,
        dims=("rotor", "time"),
        time_axis=1,
        fortran=fortran,
        offset=0,
        data_offset=offset,
        length=offset + len(raw),
        n_time=t.size,
        index="stamps" if stamped else "grid",
        rate_hz=50,
        t_start_s=t[0],
        index_start_ticks=round(t[0] * 1e9) if stamped else None,
    )
    # Only the audio descriptor exists; any audio byte read would fail the range checks.
    audio_ref = SeriesRef(
        "audio",
        "audio",
        "npy",
        (480_000,),
        "<f4",
        time_axis=0,
        index="grid",
        rate_hz=48_000,
        t_start_s=0.0,
        t_end_s=10.0,
    )
    track_meta = {"on_audio_clock": False, "unit": "Hz"}
    sc = SampleScan(
        "key",
        0,
        0,
        len(shard.payload),
        {f"{path}#time.stamps": [offset + len(raw), len(ticks.getvalue())]} if stamped else {},
        {"telemetry_tracks": {"bpf_label": track_meta}},
        {path: shaft_ref, "audio": audio_ref},
    )
    assert from_scan(shard, sc, "D3SET-frames") is None
    assert from_scan(shard, sc, "DREGON-LM-train") is None
    assert shard.calls == []
    track_meta["on_audio_clock"] = True
    got = from_scan(shard, sc, "D3SET-frames")
    assert got is not None
    assert got["suggested"] == "constant"
    assert got["basis"] == "acoustically_calibrated_bpf"
    assert not got["strong"]
    assert [r["median_rev_s"] for r in got["shafts"]] == [51, 52, 53, 55]
    if fortran:
        assert shard.calls == [
            (offset + len(raw), len(ticks.getvalue())),
            (offset + start * 4 * 8, (stop - start) * 4 * 8),
        ]
    else:
        assert shard.calls == [
            (offset + (c * t.size + start) * 8, (stop - start) * 8) for c in range(4)
        ]


def _feedback(t, x, path, meta):
    raw = np.ascontiguousarray(x.T, dtype="<f8").tobytes()
    origin = round(t[0] * 1e9)
    stamp = io.BytesIO()
    np.save(stamp, np.rint(t * 1e9).astype(np.int64) - origin, allow_pickle=False)
    ticks = stamp.getvalue()
    shard = _Ranges(raw + ticks)
    ref = SeriesRef(
        path,
        path,
        "npy",
        (t.size, x.shape[0]),
        "<f8",
        dims=("time", "channel"),
        time_axis=0,
        length=len(raw),
        index="stamps",
        n_time=t.size,
        index_start_ticks=origin,
    )
    audio = SeriesRef(
        "audio",
        "audio",
        "npy",
        (480_000,),
        "<f4",
        time_axis=0,
        index="grid",
        rate_hz=48_000,
        t_start_s=0.0,
        t_end_s=10.0,
    )
    sc = SampleScan(
        "key",
        0,
        0,
        len(shard.payload),
        {f"{path}#time.stamps": [len(raw), len(ticks)]},
        meta,
        {path: ref, "audio": audio},
    )
    return shard, sc


def test_reference_reads_only_requested_raw_rpm_windows_in_native_channel_order():
    t, x = _track()
    x = x[::-1]
    shard, sc = _feedback(
        t,
        x * 60,
        "motor_speed",
        {
            "provenance": {
                "builder": "data_processing.sources.michaels.build_frame",
                "alignment": "audio-aligned fixture",
            },
            "rps_scale": 1.2,
        },
    )
    windows = [
        {"start_s": 5.0, "end_s": 8.0},
        {"start_s": -1.0, "end_s": 1.0},
        {"start_s": 1.0, "end_s": 4.0},
        {"start_s": 8.0, "end_s": 11.0},
        {"start_s": 8.0, "end_s": 10.0},
    ]
    assert reference_windows(shard, sc, "D3SET-frames", windows) == [None] * len(windows)
    assert not shard.calls
    refs = reference_windows(shard, sc, "michaels-test-frames", windows)
    assert refs[1] is None and refs[3] is None
    for ref in (refs[0], refs[2], refs[4]):
        assert ref is not None
        assert ref["entry"] == "motor_speed"
        assert ref["basis"] == "native_feedback_acoustically_aligned"
        assert ref["coverage"] > 0.98
        np.testing.assert_allclose(ref["rates_rev_s"], x[:, 0])
        np.testing.assert_allclose(ref["ranges_rev_s"], np.repeat(x[:, :1], 2, axis=1))
    n = t.size * 32
    assert shard.calls == [
        (n, len(shard.payload) - n),
        (150 * 32, 150 * 32),
        (350 * 32, 250 * 32),
    ]


@pytest.mark.parametrize("defect", ["gap", "nan", "conflicting_stamp", "incomplete_rotors"])
def test_reference_refuses_unsafe_window_support(defect):
    t, x = _track()
    if defect == "gap":
        keep = (t <= 2.5) | (t >= 2.7)
        t, x = t[keep], x[:, keep]
    elif defect == "nan":
        x[2, 225] = np.nan
    elif defect == "conflicting_stamp":
        t[226] = t[225]
        x[2, 226] += 1
    else:
        x = x[:1]
    shard, sc = _feedback(t, x, "motors_measured", {})
    assert reference_windows(shard, sc, "DREGON-frames", [{"start_s": 1.0, "end_s": 6.0}]) == [None]
