"""D3SET builder: whole-recording Frames, array co-registration, excerpts, logger axes."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from data_processing.sources import d3set
from data_processing.streams import frame_to_sample, sample_to_frame

SR = 16_000


def _recording(root: Path, rel: str, csvs: list[str], start_ms: float | None) -> d3set.Recording:
    group, sub, split, role = d3set._group(rel + "/")
    folder = root / rel
    return d3set.Recording(
        rel=rel,
        group=group,
        subcategory=sub,
        split=split,
        role=role,
        start_ms=start_ms,
        additional={},
        wavs={w: sf.info(folder / w) for w in d3set.STREAMS if (folder / w).exists()},
        csvs=csvs,
        files=[],
    )


def _flight(
    root: Path, rel: str, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    folder = root / rel
    folder.mkdir(parents=True)
    drone = rng.integers(-3000, 3000, n, dtype=np.int16)
    array = rng.integers(-3000, 3000, (n, 3), dtype=np.int16)
    array[:400] = 0  # publisher padding
    sf.write(folder / "drone.wav", drone, SR, subtype="PCM_16")
    sf.write(folder / "matrix.wav", array, SR, subtype="PCM_16")
    (folder / "actuator.csv").write_text(
        "MOTOR_1;MOTOR_2;MOTOR_3;MOTOR_4;TIMESTAMP\n"
        "0;0;0;0;0\n100;101;102;103;8\n110;111;112;113;16\n",
        encoding="utf-8",
    )
    (folder / "acceleration.csv").write_text(
        "ACCEL_FRW;ACCEL_RIGHT;ACCEL_DOWN;TIMESTAMP\n"
        "1;2;3;1700000000000\n"
        '"N/A";"N/A";"N/A";"N/A"\n'
        "4;5;6;1700000000900\n7;8;9;1700000003000\n",
        encoding="utf-8",
    )
    return drone, array


def test_coregistered_flight_is_one_whole_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rel = "Flights/Training/clap_ok"
    n = 2 * SR + 123  # not a whole number of seconds: nothing is chunked or trimmed
    drone, array = _flight(tmp_path, rel, n, np.random.default_rng(0))
    monkeypatch.setitem(d3set.ARRAY_SYNC_EVIDENCE, rel, {"evidence": "synthetic"})
    rec = _recording(tmp_path, rel, ["actuator.csv", "acceleration.csv"], 1700000000100.0)

    frames = list(d3set._frames_of(tmp_path, rec, None, []))
    assert [k for k, _ in frames] == [rel]
    frame = sample_to_frame(frame_to_sample(frames[0][1]))

    np.testing.assert_array_equal(frame["audio"].data, drone / 32768)
    assert frame["audio"].data.dtype == np.float32
    assert frame["array_audio"].dims == (d3set.ARRAY_DIM, "time")
    np.testing.assert_array_equal(frame["array_audio"].data, array.T / 32768)
    meta = frame["meta"]
    assert meta["observation"]["mic_placement"] == "onboard_and_external"
    assert meta["audio_streams"]["array_audio"]["leading_digital_zero_samples"] == 400
    assert meta["array_sync"]["verdict"] == "published alignment consistent"

    tel = frame["telemetry"]
    acc = tel["acceleration_frd"]
    np.testing.assert_allclose(acc.indexes["time"].timestamps, [-0.1, 0.8, 2.9])
    np.testing.assert_array_equal(acc.data, [[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    bpf = tel["bpf_label"]
    np.testing.assert_allclose(bpf.indexes["time"].timestamps, [0.0, 0.008, 0.016])
    untimed = frame["telemetry_unaligned"]["acceleration_frd_untimed"]
    np.testing.assert_array_equal(untimed["source_row"].data, [1])
    assert np.isnan(untimed["values"].data).all()
    assert meta["telemetry_tracks"]["acceleration_frd"]["rows_without_timestamp"] == 1


def test_refuted_array_alignment_stays_a_separate_unshifted_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rel = "Flights/Training/clap_off_by_one"
    drone, array = _flight(tmp_path, rel, SR, np.random.default_rng(1))
    monkeypatch.setitem(d3set.ARRAY_MISALIGNED, rel, {"evidence": "synthetic"})
    rec = _recording(tmp_path, rel, ["acceleration.csv"], 1700000000100.0)

    frames = dict(d3set._frames_of(tmp_path, rec, None, []))
    assert sorted(frames) == [rel, f"{rel}/matrix.wav"]
    main, apart = frames[rel], frames[f"{rel}/matrix.wav"]
    assert "array_audio" not in main and "telemetry" in main
    np.testing.assert_array_equal(main["audio"].data, drone / 32768)
    assert set(apart) == {"array_audio", "meta"}
    assert apart["array_audio"].t_start == 0.0
    np.testing.assert_array_equal(apart["array_audio"].data, array.T / 32768)
    assert apart["meta"]["array_sync"]["verdict"] == "published alignment refuted"


def test_requantised_excerpt_is_located_and_kept_as_its_own_derivative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(2)
    parent_rel, ex_rel = "Other/early", "Delivery/early_1"
    (tmp_path / parent_rel).mkdir(parents=True)
    (tmp_path / ex_rel).mkdir(parents=True)
    parent_pcm = rng.integers(-3000, 3000, 6 * SR, dtype=np.int16)
    off = 2 * SR
    excerpt = parent_pcm[off : off + SR] + rng.integers(-1, 2, SR).astype(np.int16)
    sf.write(tmp_path / parent_rel / "drone.wav", parent_pcm, SR, subtype="PCM_16")
    sf.write(tmp_path / ex_rel / "drone.wav", excerpt, SR, subtype="PCM_16")
    (tmp_path / ex_rel / "pwm.csv").write_text(
        "ACTUATOR;TIMESTAMP\n1900;1700000001600\n1300;1700000002000\n", encoding="utf-8"
    )
    parent = _recording(tmp_path, parent_rel, [], 1700000000000.0)
    # The excerpt StartTimestamp sits 0.5 s before its measured audio offset.
    ex = _recording(tmp_path, ex_rel, ["pwm.csv"], 1700000001500.0)

    m = d3set.match_excerpt(tmp_path, ex, parent)
    assert m.offset_samples == off
    stats = m.streams["drone.wav"]
    assert stats["max_abs_diff_lsb"] == 1 and 0 < stats["samples_differing"] < SR
    assert m.start_residual_ms == pytest.approx(-500.0)

    frames = dict(d3set._frames_of(tmp_path, ex, m, []))
    frame = frames[ex_rel]
    np.testing.assert_array_equal(frame["audio"].data, excerpt / 32768)
    meta = frame["meta"]
    assert meta["label"]["role"] == "derivative"
    assert meta["derivative_of"]["parent_id"] == parent_rel
    assert meta["derivative_of"]["parent_offset_samples"] == off
    assert "telemetry" not in frame  # the publisher's start disagrees: no audio clock
    pwm = frame["telemetry_unaligned"]["servo_pwm"]
    np.testing.assert_allclose(pwm["logger_seconds"].data, [0.1, 0.5])

    bad = parent_pcm[off : off + SR].astype(np.int32) + 2
    sf.write(tmp_path / ex_rel / "drone.wav", bad.astype(np.int16), SR, subtype="PCM_16")
    assert d3set.match_excerpt(tmp_path, ex, parent).offset_samples is None


def test_unaligned_logger_tables_survive_decode_and_audio_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rel = "Other/clock_check"
    folder = tmp_path / rel
    folder.mkdir(parents=True)
    pcm = np.arange(SR, dtype=np.int16)
    sf.write(folder / "drone.wav", pcm, SR, subtype="PCM_16")
    (folder / "pwm.csv").write_text(
        "ACTUATOR;TIMESTAMP\n1900;1727115218094\n1300;1727115222094\n", encoding="utf-8"
    )
    (folder / "acceleration.csv").write_text(
        "ACCEL_FRW;ACCEL_RIGHT;ACCEL_DOWN;TIMESTAMP\n"
        "1;2;3;1727115223094\n4;5;6;1727115224094\n7;8;9;1727115225094\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(d3set.OTHER_CLOCK_EVIDENCE, rel, {"evidence": "synthetic"})
    rec = _recording(tmp_path, rel, ["pwm.csv", "acceleration.csv"], 1727115219094.0)

    [(key, frame)] = list(d3set._frames_of(tmp_path, rec, None, []))
    assert key == rel
    decoded = sample_to_frame(frame_to_sample(frame))
    assert (decoded.t_start, decoded.t_end) == (0.0, 1.0)
    clip = decoded.time[0.25:0.75]
    assert (clip.t_start, clip.t_end) == (0.25, 0.75)
    np.testing.assert_array_equal(clip["audio"].data, pcm[4000:12000] / 32768)
    assert decoded["meta"]["audio_start_evidence"]["evidence"] == "synthetic"

    for result in (decoded, clip):
        logs = result["telemetry_unaligned"]
        pwm, acc = logs["servo_pwm"], logs["acceleration_frd"]
        assert set(pwm["values"].dims).isdisjoint(acc["values"].dims)
        for track, columns, values, stamps in (
            (pwm, ("ACTUATOR",), [[1900, 1300]], [-1, 3]),
            (
                acc,
                ("ACCEL_FRW", "ACCEL_RIGHT", "ACCEL_DOWN"),
                [[1, 4, 7], [2, 5, 8], [3, 6, 9]],
                [4, 5, 6],
            ),
        ):
            data, logger_seconds = track["values"], track["logger_seconds"]
            col_dim, row_dim = data.dims
            assert data.indexes[col_dim].labels == columns
            assert logger_seconds.dims == (row_dim,) and "time" not in data.dims
            assert data.dim_index(row_dim) == logger_seconds.dim_index(row_dim)
            np.testing.assert_array_equal(data.data, values)
            np.testing.assert_array_equal(logger_seconds.data, stamps)
