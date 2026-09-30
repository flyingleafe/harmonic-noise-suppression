"""Tests for data_processing.sources.

Registry integrity is torch-free. The build round-trip writes a synthetic
MIMII-like tree, runs the real builder, and serializes each frame through the
actual ``tdframe-v1`` codec (``streams.frame_to_sample`` → ``sample_to_frame``)
— the plumbing proof that a published sample decodes back to the same Frame.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from data_processing import sources, streams
from data_processing.sources import (
    _common,
    avq,
    droneaudio,
    hustmotor,
    kaist,
    mimii,
    spcup19,
)

SR = 16000


def test_registry_integrity():
    for name, spec in sources.REGISTRY.items():
        assert spec.name == name
        assert spec.builder is None or callable(spec.builder)
        assert spec.download is None or spec.download.kind in {
            "zenodo",
            "mendeley",
            "hf",
            "gdrive",
            "http",
        }
        for key in ("description",):
            assert key in spec.provenance or key == "license", f"{name} missing provenance[{key!r}]"


def test_every_source_frames_spec_names_its_registry_entry():
    """A renamed ``frames_dataset`` must not silently orphan its derivation:
    `derive <spec>` publishes under the SPEC name but its meta/builder come
    from ``gen['source']``, so the two names have to agree — except for a
    historical adopt-only pin whose source now publishes a DERIVABLE rebuild
    under a new name (``SPCUP19-egonoise`` → ``SPCUP19-frames``)."""
    from data_processing import derivations

    for spec_name, entry in derivations.SPECS.items():
        if entry["generator"] != "source_frames":
            continue
        source = entry["gen"]["source"]
        assert source in sources.REGISTRY, f"{spec_name}: unknown source {source!r}"
        frames_name = sources.get(source).frames_name
        if entry["adopt_only"] and frames_name != spec_name:
            successor = derivations.SPECS.get(frames_name)
            assert successor is not None and not successor["adopt_only"], (
                f"{spec_name}: historical pin, but {source!r} publishes as {frames_name!r}, "
                "which is not a derivable spec"
            )
            continue
        assert frames_name == spec_name, (
            f"{spec_name}: source {source!r} publishes as {frames_name!r}"
        )


def test_telemetry_sources_are_buildable_and_fetchable():
    """The rps-campaign rigs: no audio, so ``modality`` must say so, and each
    has to be obtainable (fetcher/download/dload raw) AND buildable — a
    telemetry entry with no builder would publish an empty frames dataset."""
    telemetry = {n: s for n, s in sources.REGISTRY.items() if s.modality == "telemetry"}
    assert set(telemetry) == {"NeuroBEM", "Blackbird", "VID", "NanoBench", "PI-TCN"}
    for name, spec in telemetry.items():
        assert spec.builder is not None, f"{name} has no builder"
        assert spec.fetcher is not None or spec.download is not None or spec.raw_dataset, (
            f"{name} has no way to obtain its raw files"
        )


def test_dataset_meta_marks_layout():
    for name in sources.list_names():
        entry = sources.get(name)
        if entry.builder is None:
            continue  # raw-only entry has no frames dataset
        meta = sources.dataset_meta(name)
        assert meta[streams.LAYOUT_META_KEY] == "tdframe-v1"
        assert meta.get("description")
        assert "description" in meta


def test_safe_key_never_leading_underscore():
    assert _common.safe_key("_meta") != "_meta"
    assert _common.safe_key("a/b/c.wav").startswith("a__b__c")
    assert _common.safe_key("///") == "sample"


def _write_mimii_tree(root, snr, machine, unit, condition, stem, channels=8, n=1600):
    d = root / f"{snr}_dB_{machine}" / machine / unit / condition
    d.mkdir(parents=True, exist_ok=True)
    audio = (np.random.default_rng(0).standard_normal((n, channels)) * 0.1).astype(np.float32)
    sf.write(str(d / f"{stem}.wav"), audio, SR)


def test_build_mimii_and_roundtrip(tmp_path):
    _write_mimii_tree(tmp_path, -6, "fan", "id_00", "normal", "00000000")
    _write_mimii_tree(tmp_path, -6, "fan", "id_00", "abnormal", "00000001")

    frames = list(mimii.build_mimii(tmp_path))
    assert len(frames) == 2
    keys = {k for k, _ in frames}
    assert len(keys) == 2  # unique keys

    key, frame = frames[0]
    assert frame["audio"].dims == ("mic", "time")
    assert frame["audio"].shape == (8, 1600)
    assert frame["mic_pos"].shape == (8, 3)
    assert frame["source_pos"].shape == (1, 3)
    meta = frame["meta"]
    assert meta["dataset"] == "MIMII"
    assert meta["system"]["machine_type"] == "fan"
    assert meta["system"]["unit_id"] == "id_00"
    assert meta["operating"]["snr_db"] == -6
    assert meta["label"]["normal_vs_anomaly"] in ("normal", "abnormal")

    # Round-trip through the real codec.
    fields = streams.frame_to_sample(frame)
    frame2 = streams.sample_to_frame(fields)
    np.testing.assert_array_equal(np.asarray(frame["audio"].data), np.asarray(frame2["audio"].data))
    assert frame2["meta"]["system"]["machine_type"] == "fan"
    assert frame2["meta"]["operating"]["snr_db"] == -6
    np.testing.assert_array_equal(
        np.asarray(frame["mic_pos"].data), np.asarray(frame2["mic_pos"].data)
    )


def test_decode_drone_detection_row():
    """HF parquet row (audio bytes + int label) → mono Frame with class."""
    import io

    buf = io.BytesIO()
    audio = (np.random.default_rng(0).standard_normal((800, 1)) * 0.1).astype(np.float32)
    sf.write(buf, audio, SR, format="WAV")
    for label, expect in [(1, "drone"), (0, "no_drone")]:
        row = {"audio": {"bytes": buf.getvalue(), "path": "clip.wav"}, "label": label}
        key, frame = droneaudio._decode_drone_detection_row(3, row)
        assert frame["meta"]["label"]["class"] == expect
        assert frame["meta"]["label"]["raw_label"] == label
        assert frame["audio"].dims == ("time",)  # mono
        assert key.startswith("000003_clip")


def test_decode_droneaudioset_row_multichannel():
    """HF parquet row (decoded array + file_path) → multichannel Frame with
    subset/throttle/mic-distance parsed from the path."""
    arr = (np.random.default_rng(1).standard_normal((4, 2000)) * 0.1).tolist()  # (C, T)
    row = {
        "audio": {"array": arr, "sampling_rate": 48000, "path": "x.wav"},
        "file_path": "drone-only/drone2-only/mic-dist-50cm/throttle-low/mic0-x.wav",
        "data_type": "drone",
    }
    key, frame = droneaudio._decode_droneaudioset_row(7, row)
    assert frame["audio"].dims == ("mic", "time")
    assert frame["audio"].shape == (4, 2000)
    assert frame["meta"]["observation"]["mic_to_source_m"] == 0.5
    assert frame["meta"]["operating"]["throttle"] == "low"
    assert frame["meta"]["label"]["subset"] == "drone-only"
    assert frame["meta"]["system"]["drone_token"] == "drone2"


def test_no_datasets_are_streaming():
    # All datasets with a download spec use snapshot-download (zenodo/mendeley/hf/http/gdrive)
    # — no live hub streaming. SourceDataset has no `streaming` flag; the
    # guarantee is that the DownloadSpec kind is a local fetch.
    for name in sources.list_names():
        entry = sources.get(name)
        if entry.download is not None:
            assert entry.download.kind in {"zenodo", "mendeley", "hf", "http", "gdrive"}


def test_build_drone_detection_from_local_parquet(tmp_path):
    """The parquet builder reads snapshotted local shards (audio bytes + label)."""
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    buf = io.BytesIO()
    audio = (np.random.default_rng(0).standard_normal((800, 1)) * 0.1).astype(np.float32)
    sf.write(buf, audio, SR, format="WAV")
    rows = {
        "audio": [
            {"bytes": buf.getvalue(), "path": "a.wav"},
            {"bytes": buf.getvalue(), "path": "b.wav"},
        ],
        "label": [1, 0],
    }
    (tmp_path / "data").mkdir()
    pq.write_table(pa.table(rows), str(tmp_path / "data" / "train-00000.parquet"))

    frames = dict(droneaudio.build_drone_detection(tmp_path))
    assert len(frames) == 2
    assert {f["meta"]["label"]["class"] for f in frames.values()} == {"drone", "no_drone"}


def test_build_droneaudioset_arrow_path(tmp_path):
    """DroneAudioSet parquet (list<list<double>> audio) read via arrow buffers,
    not to_pylist — the fix for the huge-array hang. Multichannel + path meta."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    arr = np.random.default_rng(2).standard_normal((2, 500)) * 0.1  # (C, T)
    audio = {"array": arr.tolist(), "sampling_rate": 48000, "path": "x.wav"}
    table = pa.table(
        {
            "audio": [audio, audio],
            "file_path": [
                "drone-only/mic-dist-25cm/throttle-high/x.wav",
                "source-only/y.wav",
            ],
            "data_type": ["drone", "source"],
        }
    )
    (tmp_path / "drone-only").mkdir()
    pq.write_table(table, str(tmp_path / "drone-only" / "train_001.parquet"))

    frames = list(droneaudio.build_droneaudioset(tmp_path))
    assert len(frames) == 2
    _, frame = frames[0]
    assert frame["audio"].dims == ("mic", "time")
    assert frame["audio"].shape == (2, 500)
    assert int(frame["audio"].tindex.sr) == 48000
    assert frame["meta"]["observation"]["mic_to_source_m"] == 0.25
    assert frame["meta"]["operating"]["throttle"] == "high"
    assert frame["meta"]["label"]["subset"] == "drone-only"


def test_build_droneaudioset_samples_major(tmp_path):
    """Real DroneAudioSet is samples-major (outer list = time). The reshape
    path must not iterate the huge outer dim and must still yield (C, T)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    tc = np.random.default_rng(3).standard_normal((500, 2)) * 0.1  # (T, C) samples-major
    audio = {"array": tc.tolist(), "sampling_rate": 16000, "path": "z.wav"}
    table = pa.table({"audio": [audio], "file_path": ["drone-only/z.wav"], "data_type": ["drone"]})
    (tmp_path / "drone-only").mkdir()
    pq.write_table(table, str(tmp_path / "drone-only" / "s.parquet"))

    frames = list(droneaudio.build_droneaudioset(tmp_path))
    assert len(frames) == 1
    assert frames[0][1]["audio"].shape == (2, 500)  # transposed to (C, T)


def test_build_hustmotor_parses_header_and_channels(tmp_path):
    """HUST .txt: text header then tab-separated time,X,Y,Z,Sound → acoustic
    (Sound) as audio + 3-axis vibration track; sr from the time increment."""
    n, sr = 500, 25600
    t = np.arange(n) / sr
    cols = np.stack([t, t * 0 + 1, t * 0 + 2, t * 0 + 3, np.sin(2 * np.pi * 50 * t)], axis=1)
    header = "Title:\tBF_10HZ\nDAQ Settings:\nChannels:\nLegend\tX\tY\tZ\tSound\n"
    header += "Time (seconds) and Data Channels\n"
    lines = header + "\n".join("\t".join(f"{v:.8f}" for v in row) for row in cols)
    (tmp_path / "Raw data").mkdir()
    (tmp_path / "Raw data" / "BF_10HZ.txt").write_text(lines)

    frames = dict(hustmotor.build(tmp_path))
    assert len(frames) == 1
    frame = next(iter(frames.values()))
    assert frame["audio"].dims == ("time",)
    assert frame["audio"].shape == (n,)  # the Sound channel
    assert frame["vibration"].shape == (3, n)  # X, Y, Z
    assert frame["meta"]["label"]["health"] == "bearing_fault"
    # the "Sound" column is the 50 Hz sine, not a constant vibration axis
    assert float(np.std(np.asarray(frame["audio"].data))) > 0.1


def test_build_kaist_reads_signal_struct(tmp_path):
    """KAIST .mat: Signal struct with y_values.values + x_values.increment."""
    from scipy.io import savemat

    arr = np.sin(2 * np.pi * 100 * np.arange(2000) / 51200).astype(np.float64)
    aco = tmp_path / "acoustic"
    aco.mkdir()
    savemat(
        str(aco / "0Nm_BPFI_03.mat"),
        {"Signal": {"x_values": {"increment": 1.0 / 51200}, "y_values": {"values": arr}}},
    )
    frames = dict(kaist.build(tmp_path))
    assert len(frames) == 1
    frame = next(iter(frames.values()))
    assert frame["audio"].shape == (2000,)
    assert int(frame["audio"].tindex.sr) == 51200
    assert frame["meta"]["label"]["fault"] == "BPFI"
    assert frame["meta"]["label"]["severity"] == "03"


def test_spcup19_annotations_are_complete_and_pass_the_sanity_checks():
    """The committed per-team annotations load and validate; every recording has
    a condition; AGH's clean static set is rotors-off (its 'static corrupted'
    set is the same stand with the rotors running, per report + xlsx)."""
    docs = spcup19.load_annotations()
    recs = {r["key"]: r for d in docs.values() for r in d["recordings"]}
    assert all(r["condition"] in _common.CONDITIONS for r in recs.values())
    agh_clean = [r for k, r in recs.items() if k.startswith("AGH__static_clean__")]
    assert len(agh_clean) == 10
    assert all(r["condition"] == "rotors_off" and r["n_active_rotors"] == 0 for r in agh_clean)
    chums = [r for k, r in recs.items() if k.startswith("ChuMS__")]
    assert len(chums) == 9 and all("recorder_align" in r for r in chums)


_AVQ_SEQS = {"S1": ["seq1", "seq2", "seq3", "seq4"], "S2": [f"seq{i}" for i in range(1, 9)]}


def _fake_avq(root, drop=None):
    """Tiny AVQ tree: every sequence as 8 MONO-00i.wav files + misc/mic_pos.mat."""
    from scipy.io import savemat

    for session, seqs in _AVQ_SEQS.items():
        misc = root / session / "misc"
        misc.mkdir(parents=True)
        savemat(misc / "mic_pos.mat", {"mic_pos": np.arange(24, dtype=float).reshape(3, 8)})
        for seq in seqs:
            if f"{session}_{seq}" == drop:
                continue
            (root / session / seq).mkdir()
            for ch in range(8):
                sf.write(root / session / seq / f"MONO-00{ch}.wav", np.zeros(64), 44100)


def test_avq_annotations_follow_the_spec_table():
    """The drone never flies; 5 ego-noise-only, 3 motors-muted speech-only
    sequences and 4 mixtures (spec table); S1's talkers stand still, S2's move."""
    by_key = avq.load_annotations()["by_key"]
    assert sorted(by_key) == sorted(f"{s}_{q}" for s, qs in _AVQ_SEQS.items() for q in qs)
    content = {k: r["content"] for k, r in by_key.items()}
    assert sorted(k for k, c in content.items() if c == "ego_noise_only") == [
        "S1_seq1",
        "S1_seq2",
        "S1_seq3",
        "S2_seq1",
        "S2_seq2",
    ]
    assert sorted(k for k, r in by_key.items() if r["condition"] == "rotors_off") == [
        "S1_seq4",
        "S2_seq3",
        "S2_seq4",
    ]
    assert {r["condition"] for r in by_key.values()} == {
        "bench",
        "bench_varying_speed",
        "rotors_off",
    }
    varying = sorted(k for k, r in by_key.items() if r["condition"] == "bench_varying_speed")
    assert varying == ["S2_seq2", "S2_seq7", "S2_seq8"]
    assert by_key["S1_seq4"]["external_source_motion"] == "piecewise_static"
    assert all(by_key[f"S2_seq{i}"]["external_source_motion"] == "moving" for i in range(3, 9))


def test_build_avq_joins_the_annotation_and_refuses_gaps(tmp_path):
    _fake_avq(tmp_path)
    frames = dict(avq.build(tmp_path))
    assert len(frames) == 12
    meta = frames["S2_seq7"]["meta"]
    assert meta["operating"]["condition"] == "bench_varying_speed"
    assert meta["operating"]["n_active_rotors"] == 4
    assert meta["label"]["content"] == "mixture"
    assert meta["label"]["external_source_motion"] == "moving"
    muted = frames["S1_seq4"]["meta"]
    assert muted["operating"]["contains_rotor_noise"] is False
    assert muted["label"]["external_source"] == "speech"
    assert frames["S1_seq1"]["mic_pos"].shape == (8, 3)

    other = tmp_path / "missing"
    other.mkdir()
    _fake_avq(other, drop="S2_seq5")
    with pytest.raises(ValueError, match="without a recording"):
        dict(avq.build(other))


def _one_team_doc(keys):
    return {
        "team": "Idea_ssu",
        "report": "report.pdf",
        "drone": {"make_model": "test quad", "n_rotors": 4},
        "frame": {"description": "body frame, metres"},
        "geometry": {
            "pair": {
                "mic_pos": [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
                "rotor_pos": [
                    [0.2, 0.2, 0.1],
                    [0.2, -0.2, 0.1],
                    [-0.2, 0.2, 0.1],
                    [-0.2, -0.2, 0.1],
                ],
                "mic_pos_quality": "reported",
                "rotor_pos_quality": "estimated",
            }
        },
        "recordings": [
            {
                "key": k,
                "raw": "x.wav",
                "geometry": "pair",
                "condition": "bench",
                "flight_mode": None,
                "contains_rotor_noise": True,
                "n_active_rotors": 4,
                "external_source": "none",
                "description": "test",
                "source": "test",
                "confidence": "high",
            }
            for k in keys
        ],
    }


def test_build_spcup19_joins_every_recording_to_its_annotation(tmp_path, monkeypatch):
    """A decoded recording gets its annotation's tags and DREGON-style
    geometry (``mic_pos`` (mic, 3), row i = channel i; ``rotor_pos`` (rotor, 3))
    through the publish codec; an unannotated file or an annotation without its
    file fails the build instead of publishing untagged data."""
    team = tmp_path / "Idea_ssu"
    team.mkdir()
    sf.write(str(team / "hover_1.wav"), np.zeros((4410, 2), dtype=np.float32), 44100)
    monkeypatch.setattr(spcup19, "_DOWNLOAD_IDS", {"Idea_ssu": 393})
    monkeypatch.setattr(
        spcup19, "load_annotations", lambda: {"Idea_ssu": _one_team_doc(["Idea_ssu__hover_1"])}
    )
    ((key, frame),) = list(spcup19.build(tmp_path))
    frame = streams.sample_to_frame(streams.frame_to_sample(frame))
    assert key == "Idea_ssu__hover_1"
    assert frame["mic_pos"].dims == ("mic", None) and np.asarray(frame["mic_pos"].data).shape == (
        2,
        3,
    )
    assert np.asarray(frame["rotor_pos"].data).shape == (4, 3)
    meta = frame["meta"]
    assert meta["operating"]["condition"] == "bench"
    assert meta["operating"]["n_active_rotors"] == 4
    assert meta["geometry"]["config"] == "pair"

    sf.write(str(team / "extra.wav"), np.zeros((4410, 2), dtype=np.float32), 44100)
    with pytest.raises(ValueError, match="no annotation"):
        list(spcup19.build(tmp_path))
    (team / "extra.wav").unlink()
    monkeypatch.setattr(
        spcup19,
        "load_annotations",
        lambda: {"Idea_ssu": _one_team_doc(["Idea_ssu__hover_1", "Idea_ssu__gone"])},
    )
    with pytest.raises(ValueError, match="annotated but not found"):
        list(spcup19.build(tmp_path))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("condition", "hovering"),
        ("n_active_rotors", 0),
        ("flight_mode", "hover"),
        ("geometry", "nope"),
    ],
)
def test_spcup19_annotation_validation_rejects_inconsistent_tags(field, value):
    """Vocabulary and cross-field rules: a bench recording cannot have 0 active
    rotors or a flight mode, and geometry must name a defined config."""
    doc = _one_team_doc(["Idea_ssu__a"])
    doc["recordings"][0][field] = value
    with pytest.raises(ValueError):
        spcup19.validate_team(doc)


@pytest.mark.parametrize("k", [5, -7])
def test_spcup19_align_recorders_undoes_the_two_recorder_offset(k: int) -> None:
    """``B[n+K] ≈ A[n]``: after alignment every channel carries the same samples."""
    base = np.random.default_rng(0).standard_normal(size=200).astype(np.float32)
    lead = np.random.default_rng(1).standard_normal(size=abs(k)).astype(np.float32)
    a = base if k > 0 else np.concatenate([lead, base])
    b = np.concatenate([lead, base]) if k > 0 else base
    out = spcup19._align_recorders(
        [a, b, a, b], {"group_a": [0, 2], "group_b": [1, 3], "k_samples": k}
    )
    np.testing.assert_array_equal(out, np.stack([base] * 4))


def test_spcup19_chums_rejects_unlabelled_runs(tmp_path):
    """A ``Details`` string that does not name the propeller count/repeat is a
    layout change: fail instead of publishing unlabelled arrays."""
    from scipy.io import savemat

    mic = np.zeros((1, 8), dtype=[(f, object) for f in ("Fs", "OASPL", "RawTruncatedCalibrated")])
    for j in range(8):
        mic[0, j] = (16000.0, 90.0, np.zeros(100))
    tests = np.zeros((1, 1), dtype=[("Details", object), ("Data", object)])
    tests[0, 0] = ("calibration tone", mic)
    path = tmp_path / "UAV_rotor_recordings.mat"
    savemat(str(path), {"TestResults": {"Test": tests, "MicPositions": np.zeros((8, 2))}})
    with pytest.raises(ValueError, match="Details"):
        list(spcup19._chums_runs(path))


def test_spcup19_registered_with_http_download():
    spec = sources.get("SPCUP19-egonoise")
    assert spec.download is not None and spec.download.kind == "http"
    assert len(spec.download.params["urls"]) == 10  # 10 teams
