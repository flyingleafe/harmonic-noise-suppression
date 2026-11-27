"""drone-recording-annotations: SQLite index, release packaging, annotated Frames.

Everything runs against synthetic parents committed to a local file-backed
dload repository (no R2, no real audio)."""

from __future__ import annotations

import gzip
import io
import json
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf
import tdseries as td
from dload.cache import ShardCache
from dload.remote import LocalRemote
from dload.repo import Repository

import data_processing.recording_annotations as ra
from data_processing import streams
from data_processing.recording_annotations import schema
from data_processing.recording_annotations import storage as ra_storage

SR = 8000
PARENT = "PARENT-FRAMES-TEST"
RAW = "PARENT-RAW-TEST"


def _repo(root, cache="cache") -> Repository:
    return Repository(
        LocalRemote(root / "remote"), ShardCache(root / cache, None), lock_path=root / "dload.lock"
    )


def _recording(seed: int, *, t0: float, channels: int, audio: bool = True) -> td.Frame:
    rng = np.random.default_rng(seed)
    entries: dict = {
        "meta": td.Frame(
            {
                "recording_id": f"rec{seed}",
                "note": seed,
                "geometry": {"mics": [[0.0, 0.1, 0.0]], "unit": "m"},
            }
        )
    }
    if audio:
        data = (0.1 * rng.standard_normal((channels, 2 * SR))).astype(np.float32)
        entries["audio"] = td.uniform(data, SR, dims=("mic", "time"), t_start=t0)
    stamps = t0 + np.linspace(0.0, 1.9, 20)
    measured = np.tile(np.array([[50.0], [51.0], [52.0], [53.0]], np.float32), (1, 20))
    entries["motors_measured"] = td.Series(
        measured,
        ("rotor", "time"),
        {"time": td.StampIndex.from_stamps(stamps, t_start=t0, t_end=t0 + 2.0)},
    )
    # Native supplied speed + a nested log with an unaligned (non-time) clock and gaps.
    entries["rps"] = td.Series(
        measured / 1.0,
        ("rotor", "time"),
        {"time": td.StampIndex.from_stamps(stamps, t_start=t0, t_end=t0 + 2.0)},
    )
    log_stamps = np.array([10.0, np.nan, 12.5])
    entries["telemetry"] = td.Frame(
        {
            "logger_s": td.wrap(log_stamps),
            "imu": td.Series(np.array([[0.1, np.nan, 0.3]]), ("axis", "row")),
            "source": "flight-controller",
        }
    )
    return td.Frame(entries)


def _record(**overrides) -> dict:
    base = {
        "rig_mount": "fixed_stand",
        "speed_regime": "constant",
        "propulsion": "propellers",
        "rotor_activity": "all",
        "relative_motion": "fixed",
        "speed_supplied": "yes",
        "sound_content": "drone_only",
        "environment": "indoors",
        "blade_count": "2",
        "n_rotors": 4,
        "n_active": 4,
        "provenance_kind": "recording",
        "supplied_kind": "measured",
        "role_motion": {"onboard": "fixed"},
        "source_detail": {"doc": "synthetic test"},
        "evidence": {
            "rig_mount": "test stand",
            "speed_regime": "test plateau",
            "propulsion": "test props",
            "rotor_activity": "all four",
            "relative_motion": "onboard array",
            "speed_supplied": "tachometer",
            "sound_content": "no source",
            "environment": "lab",
            "blade_count": "photo",
        },
    }
    base.update(overrides)
    return schema.normalize(base, has_audio=True)


TACH = {
    "status": "partial",
    "reason": "one coincident line",
    "identity": "anonymous",
    "n_expected": 4,
    "uncertainty_kind": "acoustic_repeatability_tolerance",
    "windows": [
        # t_start of rec1 is 0.5: window times are on the Frame clock.
        {
            "start_s": 0.5,
            "end_s": 1.5,
            "center_s": 1.0,
            "rates_rev_s": [50.1, 51.0, 52.2],
            "uncertainty_rev_s": [0.02, 0.03, 0.05],
            "order_support": [6, 6, 4],
        },
        {
            "start_s": 1.0,
            "end_s": 2.0,
            "center_s": 1.5,
            "rates_rev_s": [],
            "uncertainty_rev_s": [],
            "accepted": False,
            "reason": "octave ambiguity",
        },
        {
            "start_s": 1.5,
            "end_s": 2.5,
            "center_s": 2.0,
            "rates_rev_s": [50.3, 51.1, 52.4, 53.0],
            "uncertainty_rev_s": [0.02, 0.03, 0.04, 0.06],
            "order_support": [6, 6, 5, 3],
        },
    ],
}


@pytest.fixture
def world(tmp_path):
    repo = _repo(tmp_path)
    frames = {
        "rec0": _recording(0, t0=0.0, channels=2),
        "rec1": _recording(1, t0=0.5, channels=3),
        "rec2": _recording(2, t0=0.0, channels=1, audio=False),
    }
    v1 = repo.commit(
        PARENT,
        ((k, streams.frame_to_sample(f)) for k, f in frames.items()),
        meta={"layout": streams.TDFRAME_LAYOUT},
        target_shard_size=64 * 1024,
    )
    assert len(v1.shards) >= 2
    stage = tmp_path / "stage"
    stage.mkdir()
    records = {
        "rec0": _record(),
        "rec1": _record(
            rig_mount="free_flight",
            speed_regime="mixed",
            n_active=None,
            role_motion={"onboard": "fixed", "ground": "moving"},
            relative_motion="mixed",
            evidence={
                **_record()["evidence"],
                "rig_mount": "flight",
                "relative_motion": "roles",
                "speed_regime": "steady parts",
            },
        ),
        "rec2": schema.normalize({"provenance_kind": "log_only"}, has_audio=False),
    }
    path = stage / f"{PARENT}@{v1.version}.sqlite"
    with ra.IndexWriter(
        path,
        dataset=PARENT,
        version=v1.version,
        layout=v1.meta["layout"],
        parent_shards=v1.shards,
        source_commit="test",
    ) as w:
        for loc, key, _ in ra.located_samples(PARENT, v1.version, repo=repo):
            info = {
                "has_audio": key != "rec2",
                "duration_s": 2.0,
                "sample_rate": SR,
                "series": {"audio": {"shape": list(frames[key]["audio"].data.shape)}}
                if key != "rec2"
                else {},
            }
            w.add(
                key,
                records[key],
                info=info,
                location=loc,
                tach=TACH if key == "rec1" else None,
                audio_t_start_s=0.5 if key == "rec1" else 0.0,
                measure={"channels": [0]} if key == "rec1" else None,
            )
        w.close(expected_keys=frames)
    return {"repo": repo, "v1": v1, "stage": stage, "frames": frames, "db": path}


def _record_of(idx: ra.Index, key: str, *, windows: bool = True) -> dict:
    record = idx.record(key, windows=windows)
    assert record is not None, key
    return record


def _location(idx: ra.Index, key: str) -> ra.SampleLocation:
    loc, _ = idx.locate(key)
    assert loc is not None, key
    return loc


def test_filters_and_counts(world):
    idx = ra.open_index(world["db"])
    assert (idx.dataset, idx.version, idx.n_rows) == (PARENT, world["v1"].version, 3)
    assert idx.count() == 3
    assert [r["key"] for r in idx.select({"rig_mount": ["fixed_stand", "free_flight"]})] == [
        "rec0",
        "rec1",
    ]
    assert (
        idx.count({"rig_mount": "free_flight", "speed_regime": "constant"}) == 0
    )  # AND across facets
    assert [r["key"] for r in idx.select({"rig_mount": ra.KNOWN})] == ["rec0", "rec1"]
    assert [r["key"] for r in idx.select({"rig_mount": "not_applicable"})] == ["rec2"]
    assert [r["key"] for r in idx.select({"wind": "unknown"})] == ["rec0", "rec1"]
    assert [r["key"] for r in idx.select({"n_active": None})] == ["rec1", "rec2"]
    assert [r["key"] for r in idx.select({"role_motion": {"ground": "moving"}})] == ["rec1"]
    assert [r["key"] for r in idx.select({"tach_status": "none"})] == ["rec0", "rec2"]
    assert [r["key"] for r in idx.select({"tach_status": "partial", "has_audio": True})] == ["rec1"]
    assert [r["key"] for r in idx.select({"supplied_kind": [ra.NULL_LABEL]})] == ["rec2"]
    assert [r["key"] for r in idx.select({"provenance_kind": "log_only"})] == ["rec2"]
    assert list(idx.keys({"key_prefix": "rec"})) == ["rec0", "rec1", "rec2"]
    assert [r["key"] for r in idx.select(limit=1, offset=1)] == ["rec1"]
    counts = idx.facet_counts({"rig_mount": "fixed_stand", "tach_status": "none"})
    assert counts["rig_mount"] == {"fixed_stand": 1, "not_applicable": 1}  # own filter excluded
    assert counts["speed_regime"] == {"constant": 1}
    assert counts["tach_status"] == {"none": 1}
    assert counts["role_motion"] == {"onboard=fixed": 1}
    assert counts["supplied_kind"] == {"measured": 1}
    row = idx.select({"keys": ["rec1"]})[0]
    assert row["n_channels"] == 3 and row["tach_n_windows"] == 3 and row["has_audio"] is True
    with pytest.raises(ValueError, match="vocabulary"):
        idx.count({"rig_mount": "drone"})
    with pytest.raises(ValueError, match="unknown annotation filter"):
        idx.count({"colour": "red"})
    with pytest.raises(ValueError):
        idx.select(limit=ra.MAX_PAGE + 1)


def test_record_normalized_and_tach(world):
    idx = ra.open_index(world["db"])
    rec = _record_of(idx, "rec1")
    assert rec["source_detail"] == {"doc": "synthetic test"} and rec["measure"] == {"channels": [0]}
    assert rec["role_motion"] == {"onboard": "fixed", "ground": "moving"}
    tach = rec["tach"]
    assert (tach["status"], tach["n_windows"], tach["n_accepted"]) == ("partial", 3, 2)
    assert tach["summary"] == {"uncertainty_kind": "acoustic_repeatability_tolerance"}
    w0, w1, _ = tach["windows"]
    assert (w0["center_s"], w0["center_abs_s"], w0["slots"]) == (0.5, 1.0, [0, 1, 2])
    assert w1["accepted"] is False and w1["reason"] == "octave ambiguity"
    assert tach["rate_min_rev_s"] == 50.1 and tach["rate_max_rev_s"] == 53.0
    assert idx.record("missing") is None
    # Normalization: two identical records share one profile.
    con = idx._db
    assert con.execute("SELECT COUNT(*) FROM profile").fetchone()[0] == 3


def test_load_frame_roundtrip_and_version_isolation(world, monkeypatch):
    repo, v1, frames = world["repo"], world["v1"], world["frames"]
    rel = ra.open_release(world["stage"], repo=repo)
    assert [(p.dataset, p.version, p.n_rows) for p in rel.parents()] == [(PARENT, v1.version, 3)]
    # Fresh cache: parent bytes must come from ranged reads, not whole-shard copies
    # (each tiny test shard holds ~one sample, so lift the whole-shard threshold).
    monkeypatch.setattr(ra_storage, "WHOLE_SHARD_FRACTION", 1.01)
    cold = _repo(world["db"].parent.parent, cache="cold-cache")
    rel = ra.open_release(world["stage"], repo=cold)
    frame = ra.load_frame(PARENT, "rec1", version=v1.version, release=rel)
    assert isinstance(cold.remote, LocalRemote)
    assert cold.remote.op_counts.get("get_to_file", 0) == 0
    src = frames["rec1"]
    np.testing.assert_array_equal(frame["audio"].data, src["audio"].data)
    np.testing.assert_array_equal(frame["motors_measured"].data, src["motors_measured"].data)
    assert frame["audio"].t_start == 0.5
    cond = frame["meta"]["recording_conditions"]
    assert (
        cond["rig_mount"] == "free_flight"
        and cond["annotation_release"]["parent_version"] == v1.version
    )
    assert cond["tach"]["uncertainty_kind"] == "acoustic_repeatability_tolerance"
    assert "windows" not in cond["tach"] and cond["tach"]["n_windows"] == 3
    rps, unc = frame["rps_acoustic"], frame["rps_acoustic_uncertainty"]
    assert rps.dims == ("rank", "time") and "rank" not in rps.indexes  # ranks carry no motor labels
    # Every attempted window keeps its centre; the rejected one is all-NaN (an explicit gap).
    np.testing.assert_allclose(rps.tindex.abs_stamps, [1.0, 1.5, 2.0])
    assert np.isnan(rps.data[:, 1]).all() and not frame["rps_acoustic_accepted"].data[1]
    np.testing.assert_allclose(frame["rps_acoustic_window"].data[:, 0], [0.5, 1.5])
    assert np.isnan(rps.data[3, 0]) and np.isnan(unc.data[3, 0])  # unresolved, never filled
    np.testing.assert_allclose(rps.data[:, 2], [50.3, 51.1, 52.4, 53.0], rtol=1e-6)
    np.testing.assert_allclose(unc.data[:3, 0], [0.02, 0.03, 0.05], rtol=1e-6)
    assert frame["meta"]["note"] == 1
    # Everything the parent published (nested log, native rps, meta) is untouched.
    assert_overlay_preserves(frame, streams.decode_tdframe(("rec1", streams.frame_to_sample(src))))
    # Annotated view survives the tdframe codec.
    again = streams.sample_to_frame(streams.frame_to_sample(frame))
    np.testing.assert_array_equal(again["rps_acoustic"].data, rps.data)

    # Metadata-only rows: no audio fabricated.
    log_only = ra.load_frame(PARENT, "rec2", version=v1.version, release=rel)
    assert "audio" not in log_only and "rps_acoustic" not in log_only
    assert log_only["meta"]["recording_conditions"]["rig_mount"] == "not_applicable"
    meta_only = ra.load_frame(PARENT, "rec1", version=v1.version, release=rel, audio=False)
    assert "audio" not in meta_only and meta_only["rps_acoustic"].data.shape == (4, 3)

    # Filtered iteration reads only matching samples.
    got = [
        k
        for k, _ in ra.iter_frames(
            PARENT, {"rig_mount": "fixed_stand"}, version=v1.version, release=rel
        )
    ]
    assert got == ["rec0"]

    # A newer parent version is NOT annotated by v1's database.
    v2 = repo.commit(
        PARENT,
        [("rec0", streams.frame_to_sample(frames["rec0"]))],
        meta={"layout": streams.TDFRAME_LAYOUT},
    )
    with pytest.raises(ra.NotAnnotatedError, match=v2.version[:12]):
        rel.index(PARENT, v2.version)


def test_publish_carry_forward(world, tmp_path):
    repo, v1 = world["repo"], world["v1"]
    m1 = ra.publish(world["stage"], repo=repo)
    assert len(m1.shards) == 1 and m1.meta["parents"][0]["key"] == f"{PARENT}@{v1.version}"
    entry = m1.meta["parents"][0]
    assert entry["db_sha256"] and entry["n_rows"] == 3

    # Second parent, raw-files layout (mono wav + 2-channel flac), staged alone.
    stereo = np.random.default_rng(3).standard_normal((SR, 2)).astype(np.float32) * 0.1
    mono = np.random.default_rng(4).standard_normal(SR).astype(np.float32) * 0.1
    buf_w, buf_f = io.BytesIO(), io.BytesIO()
    sf.write(buf_w, mono, SR, format="WAV", subtype="FLOAT")
    sf.write(buf_f, stereo, SR, format="FLAC")
    raw = repo.commit(
        RAW,
        [("take/a", {"wav": buf_w.getvalue()}), ("take/b", {"flac": buf_f.getvalue()})],
        meta={"layout": "raw-files"},
    )
    stage2 = tmp_path / "stage2"
    with ra.IndexWriter(
        stage2 / f"{RAW}@{raw.version}.sqlite",
        dataset=RAW,
        version=raw.version,
        layout="raw-files",
        parent_shards=raw.shards,
    ) as w:
        rec = _record()
        for loc, key, _ in ra.located_samples(RAW, raw.version, repo=repo):
            w.add(key, rec, info={"has_audio": True, "sample_rate": SR}, location=loc)
        w.close(expected_keys=["take/a", "take/b"])
    m2 = ra.publish(stage2, repo=repo)
    keys = [p["key"] for p in m2.meta["parents"]]
    assert keys == sorted([f"{PARENT}@{v1.version}", f"{RAW}@{raw.version}"])
    assert m1.shards[0] in m2.shards  # carried byte-for-byte, same shard

    cold = _repo(tmp_path, cache="cold-cache-2")
    rel = ra.open_release(version=m2.version, repo=cold)
    assert rel.version == m2.version
    b = ra.load_frame(RAW, "take/b", version=raw.version, release=rel)
    assert b["audio"].dims == ("mic", "time")
    np.testing.assert_allclose(b["audio"].data, stereo.T, atol=1e-4)
    a = ra.load_frame(RAW, "take/a", version=raw.version, release=rel)
    np.testing.assert_array_equal(a["audio"].data, mono)
    assert a["meta"]["recording_conditions"]["blade_count"] == "2"
    # Release DB fetched once, cached decompressed and verified.
    stored = list((cold.cache.root / "recording-annotations").glob("*.sqlite"))
    assert len(stored) == 1
    with cold.open_shard(
        m2.shards[[p["key"] for p in m2.meta["parents"]].index(f"{RAW}@{raw.version}")]
    ) as p:
        from dload.pack import PackReader

        with PackReader(p) as r:
            assert gzip.decompress(r.read(0)[1][ra.DB_FIELD]) == stored[0].read_bytes()


def test_writer_rejects_bad_input(world, tmp_path):
    v1 = world["v1"]
    path = tmp_path / "bad.sqlite"
    w = ra.IndexWriter(path, dataset=PARENT, version=v1.version, layout="tdframe-v1")
    with pytest.raises(ValueError, match="invalid rig_mount"):
        w.add("k", {**_record(), "rig_mount": "drone"}, info={})
    with pytest.raises(ValueError, match="chronological"):
        bad = {**TACH, "windows": [TACH["windows"][2], TACH["windows"][0]]}
        w.add("k2", _record(), info={}, tach=bad)
    w.abort()
    w = ra.IndexWriter(path, dataset=PARENT, version=v1.version, layout="tdframe-v1")
    w.add("rec0", _record(), info={})
    with pytest.raises(ValueError, match="unannotated"):
        w.close(expected_keys=["rec0", "rec1"])
    assert not path.exists()
    with pytest.raises(ValueError, match="64-hex"):
        ra.IndexWriter(path, dataset=PARENT, version=v1.version[:12], layout="tdframe-v1")
    with pytest.raises(ValueError, match="slots"):
        w = ra.IndexWriter(path, dataset=PARENT, version=v1.version, layout="tdframe-v1")
        win = {**TACH["windows"][0], "slots": [0, 0, 1]}
        w.add("k", _record(), info={}, tach={**TACH, "windows": [win]})


def test_sample_dir_mixture_convention():
    mix = np.random.default_rng(5).standard_normal((2, 16000)).astype(np.float32) * 0.1
    buf = io.BytesIO()
    sf.write(buf, mix.T, 16000, format="WAV", subtype="FLOAT")
    rps = io.BytesIO()
    np.save(rps, np.full((4, 10), 60.0, np.float32))
    fields = {
        "mixture": buf.getvalue(),
        "vocals": buf.getvalue(),
        "rps": rps.getvalue(),
        "meta": json.dumps({"snr_db": -10}).encode(),
    }
    frame = ra.decode_sample(
        "sample-dir-v1",
        "sample_00001",
        fields,
        parent_meta={
            "fields": {
                "mixture": "mixture.wav",
                "vocals": "vocals.wav",
                "rps": "rps.npy",
                "meta": "meta.json",
            }
        },
    )
    assert set(frame) >= {"mixture", "vocals", "rps", "meta"}
    np.testing.assert_allclose(frame["mixture"].data, mix)
    assert frame["meta"]["snr_db"] == -10


def test_unresolved_track_claims_no_speed(world, tmp_path):
    v1 = world["v1"]
    refused = {
        "status": "unresolved",
        "reason": "unknown shaft order",
        "identity": "anonymous",
        "n_expected": 4,
        "windows": [
            {
                "start_s": 0.0,
                "end_s": 1.0,
                "center_s": 0.5,
                "rates_rev_s": [],
                "uncertainty_rev_s": [],
                "accepted": False,
                "quality": {"apparent_hz": [101.2]},
                "order_support": {"apparent_only": True},
            }
        ],
    }
    path = tmp_path / "u.sqlite"
    with ra.IndexWriter(path, dataset=PARENT, version=v1.version, layout="tdframe-v1") as w:
        w.add("k", _record(speed_supplied="no"), info={"duration_s": 2.0}, tach=refused)
        w.close()
    idx = ra.open_index(path)
    win = _record_of(idx, "k")["tach"]["windows"][0]
    assert win["quality"] == {"apparent_hz": [101.2]} and win["order_support"] == {
        "apparent_only": True
    }
    frame = ra.annotate_frame(
        None, _record_of(idx, "k", windows=False), idx.tach_arrays("k"), {"name": "t"}
    )
    assert "rps_acoustic" not in frame
    assert frame["meta"]["recording_conditions"]["speed_supplied"] == "no"
    assert frame["meta"]["recording_conditions"]["tach"]["status"] == "unresolved"

    # A named singleton hub (identity bound by the source): labelled by the source's motor name.
    hub = {
        "status": "resolved",
        "identity": "physical",
        "n_expected": 1,
        "rotor_ids": ["Motor4"],
        "windows": [
            {
                "start_s": 0.0,
                "end_s": 1.0,
                "center_s": 0.5,
                "rates_rev_s": [70.2],
                "uncertainty_rev_s": [0.01],
            }
        ],
    }
    path = tmp_path / "hub.sqlite"
    with ra.IndexWriter(path, dataset=PARENT, version=v1.version, layout="tdframe-v1") as w:
        w.add("k", _record(rotor_activity="single", n_active=1), info={"duration_s": 2.0}, tach=hub)
        with pytest.raises(ValueError, match="ranks are not motors"):
            w.add("k2", _record(), info={}, tach={**hub, "identity": "anonymous"})
        w.close()
    idx = ra.open_index(path)
    assert idx.count() == 1 and idx.record("k2") is None  # the rejected add left no partial rows
    frame = ra.annotate_frame(
        None, _record_of(idx, "k", windows=False), idx.tach_arrays("k"), {"name": "t"}
    )
    assert frame["rps_acoustic"].dim_index("rotor").labels == ("Motor4",)
    assert frame["rps_acoustic_uncertainty"].dim_index("rotor").labels == ("Motor4",)


def test_streams_conditions_overlay(world):
    repo, v1, frames = world["repo"], world["v1"], world["frames"]
    ra.publish(world["stage"], repo=repo)
    legacy = list(streams.iter_published_frames(PARENT, v1.version, repo=repo))
    assert len(legacy) == 3 and all("recording_conditions" not in f["meta"] for f in legacy)
    assert all("rps_acoustic" not in f for f in legacy)
    picked = list(
        streams.iter_published_frames(
            PARENT, v1.version, conditions={"rig_mount": "fixed_stand"}, repo=repo
        )
    )
    assert len(picked) == 1 and picked[0]["meta"]["recording_id"] == "rec0"
    assert picked[0]["meta"]["recording_conditions"]["rig_mount"] == "fixed_stand"
    assert_overlay_preserves(
        picked[0], streams.decode_tdframe(("rec0", streams.frame_to_sample(frames["rec0"])))
    )
    # rec1 has acoustic windows; its native `rps` track stays the published one.
    (rec1,) = streams.iter_published_frames(
        PARENT, v1.version, conditions={"tach_status": "partial"}, repo=repo
    )
    np.testing.assert_array_equal(rec1["rps"].data, frames["rec1"]["rps"].data)
    assert "rps_acoustic" in rec1


def _mdm_container(samples: np.ndarray, *, declared_extra: int = 0) -> bytes:
    import base64
    import struct

    data = samples.astype("<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, 48000, 48000 * 4, 4, 32)
    body = (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)
        + fmt
    )
    body += b"data" + struct.pack("<I", len(data) + declared_extra) + data
    return (
        b"<ns2:doc><ns2:Identifier>01-FB-Mi1</ns2:Identifier><ns2:Created>2021-01-01</ns2:Created>"
        b"<ns3:binary mime='audio/wav'>" + base64.b64encode(body) + b"</ns3:binary></ns2:doc>"
    )


def test_audrok_raw_containers():
    samples = np.linspace(-0.5, 0.5, 480, dtype=np.float32)
    xml = _mdm_container(samples, declared_extra=400)  # truncated: 100 declared samples absent
    frame = ra.decode_sample(
        "raw-files", "Measurements_1-19/01-FB-Mi1", {"xml": xml}, dataset="AuDroK", has_audio=True
    )
    assert frame["audio"].dims == ("time",) and frame["audio"].tindex.sr == 48000
    np.testing.assert_array_equal(frame["audio"].data, samples)  # present samples only, no padding
    assert frame["meta"]["container_header"]["Identifier"] == "01-FB-Mi1"
    assert frame["meta"]["container_wav"]["truncated"] is True
    # The offer description and any non-AuDroK dataset stay opaque XML.
    assert "audio" not in ra.decode_sample(
        "raw-files", "Angebotsbeschreibung", {"xml": xml}, dataset="AuDroK"
    )
    assert "audio" not in ra.decode_sample(
        "raw-files", "Measurements_1-19/01-FB-Mi1", {"xml": xml}, dataset="other"
    )


def test_raw_tree_conventions():
    mono = (0.1 * np.random.default_rng(6).standard_normal(SR)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, mono, SR, format="WAV", subtype="FLOAT")
    wav = buf.getvalue()
    # Pre-layout raw tree (manifest layout absent), extensionless file named by the key.
    frame = ra.decode_sample(None, "takes/clip.wav", {"data": wav})
    np.testing.assert_array_equal(frame["audio"].data, mono)
    # Upper-case extension field, a sidecar field alongside.
    frame = ra.decode_sample("raw-files", "takes/clip", {"WAV": wav, "xml": b"<x/>"})
    assert frame["meta"]["source_fields"] == ["WAV", "xml"]
    # Unknown extension but the index says has_audio: sniffed, never an empty Frame.
    frame = ra.decode_sample("raw-files", "takes/clip", {"bin": wav}, has_audio=True)
    assert "audio" in frame
    with pytest.raises(ValueError, match="has_audio"):
        ra.decode_sample("raw-files", "takes/clip", {"txt": b"hello"}, has_audio=True)
    assert "audio" not in ra.decode_sample("raw-files-v1", "README", {"md": b"# x"})


_OVERLAY = {
    "rps_acoustic",
    "rps_acoustic_uncertainty",
    "rps_acoustic_window",
    "rps_acoustic_accepted",
}


def _index_skeleton(series: td.Series) -> td.Series:
    return td.Series(np.zeros(np.shape(series.data), dtype=bool), series.dims, series.indexes)


def _assert_same_series_data(actual: np.ndarray, expected: np.ndarray, path: str) -> None:
    assert actual.dtype == expected.dtype and actual.shape == expected.shape, path
    if expected.dtype.names:
        # Record equality treats NaT/NaN leaves as unequal. Compare the exact
        # native bytes instead, including null encodings and NaN payloads.
        opaque = np.dtype(f"V{expected.dtype.itemsize}")
        actual, expected = actual.view(opaque), expected.view(opaque)
    np.testing.assert_array_equal(actual, expected, err_msg=path)


def test_chums_spectral_products_are_not_audio(tmp_path, monkeypatch):
    dataset = "SPCUP19-egonoise"
    stem = "ChuMS__UAV_rotor_recordings_TestResults__Test_0___Data_0___"
    products = {"Freq": [0.0, 30.0, 60.0], "SPL": [50.0, 25.0, np.nan]}
    native = {}
    for kind, values in {**products, "RawTruncatedCalibrated": [0.1, -0.2]}.items():
        native[stem + kind] = streams.frame_to_sample(
            td.Frame({"audio": td.uniform(np.array(values, np.float32), SR, dims=("time",))})
        )
    repo = _repo(tmp_path)
    manifest = repo.commit(dataset, list(native.items()), meta={"layout": "tdframe-v1"})
    # The misencoding is known only on the frozen parent version.
    example = stem + "Freq"
    assert "audio" in ra.decode_sample(
        "tdframe-v1", example, native[example], dataset=dataset, parent_version=manifest.version
    )
    monkeypatch.setattr(ra_storage, "_CHUMS_SPECTRAL_PARENT_VERSION", manifest.version)
    stage = tmp_path / "stage"
    path = stage / f"{dataset}@{manifest.version}.sqlite"
    with ra.IndexWriter(
        path,
        dataset=dataset,
        version=manifest.version,
        layout="tdframe-v1",
        parent_shards=manifest.shards,
    ) as w:
        for loc, key, _ in ra.located_samples(dataset, manifest.version, repo=repo):
            audio = key.endswith("___RawTruncatedCalibrated")
            facts = _record() if audio else schema.normalize({}, has_audio=False)
            w.add(
                key,
                facts,
                location=loc,
                info={"has_audio": audio, "series": {"audio": {"shape": [1, 3]}}},
            )
    release = ra.open_release(stage, repo=repo)
    idx = release.index(dataset, manifest.version)
    assert list(idx.keys({"has_audio": True})) == [stem + "RawTruncatedCalibrated"]
    for kind in products:
        key = stem + kind
        row = idx.select({"keys": [key]})[0]
        assert row["n_channels"] is None
        frame = ra.load_frame(dataset, key, version=manifest.version, release=release)
        assert "audio" not in frame and frame.duration == 0
        spectral = frame["spectral_values"]
        assert spectral.dims == ("frequency_bin",) and not spectral.has_time
        original = streams.sample_to_frame(native[key])["audio"]
        np.testing.assert_array_equal(spectral.data.view("u4"), original.data.view("u4"))
        assert frame["meta"]["spectral_source"] == {
            "product": kind,
            "representation": "frequency_axis" if kind == "Freq" else "level_spectrum",
            "stored_field": "audio",
            "stored_layout": "tdframe-v1",
            "stored_axis": "time",
            "is_waveform": False,
        }
    waveform = ra.load_frame(
        dataset, stem + "RawTruncatedCalibrated", version=manifest.version, release=release
    )
    assert "audio" in waveform and "spectral_values" not in waveform


def assert_overlay_preserves(got: td.Frame, want: td.Frame, path: str = "") -> None:
    """``got`` = ``want`` plus only the annotation overlay: nested Frames
    recursively, Series by dims/indexes (any time-index kind) and NaN-aware
    data, invariant values by ``==``; extra keys only ``rps_acoustic*`` at the
    top and ``meta.recording_conditions``."""
    extra = set(got) - set(want)
    allowed = _OVERLAY if path == "" else {"recording_conditions"} if path == "meta/" else set()
    assert extra <= allowed and set(want) <= set(got), (
        path,
        sorted(extra),
        sorted(set(want) - set(got)),
    )
    overlay = [got[name] for name in extra & _OVERLAY]
    assert got.t_start_ticks == min([want.t_start_ticks, *(s.t_start_ticks for s in overlay)]), path
    assert got.t_end_ticks == max([want.t_end_ticks, *(s.t_end_ticks for s in overlay)]), path
    for name in want:
        a, b = got[name], want[name]
        where = f"{path}{name}"
        if isinstance(b, td.Frame):
            assert isinstance(a, td.Frame), where
            assert_overlay_preserves(a, b, f"{where}/")
        elif isinstance(b, td.Series):
            assert isinstance(a, td.Series) and _index_skeleton(a).equal(_index_skeleton(b)), where
            _assert_same_series_data(np.asarray(a.data), np.asarray(b.data), where)
        else:
            assert type(a) is type(b) and (a == b or (a != a and b != b)), where  # NaN == NaN


@pytest.mark.parametrize("nested", [False, True])
def test_tach_bounds_use_analysed_audio_grid(nested):
    # Native GridIndex(sr, n) can declare zero duration despite having PCM.
    # A different, longer role (or the parent hull) must not mask bad centres.
    t0 = 4.0 if nested else 0.0
    grid = td.GridIndex(SR, 2 * SR, t_start_ticks=td.secs_to_ticks(t0))
    pcm = td.Series(np.zeros((17, 2 * SR), np.float32), ("mic", "time"), {"time": grid})
    role = "remote/pcm" if nested else "audio"
    if nested:
        dat = np.array(
            [(np.datetime64("NaT", "ns"), np.nan)], dtype=[("utc", "M8[ns]"), ("x", "f8")]
        )
        parent = td.Frame(
            {
                "audio": td.uniform(np.zeros(10 * SR), SR),
                "remote": td.Frame({"pcm": pcm}),
                "dat": td.Frame({"rows": td.Series(dat, ("row",))}),
            }
        )
    else:
        parent = td.Frame({"audio": pcm})
        assert parent.t_start == parent.t_end == 0.0
    parent = streams.decode_tdframe(("example", streams.frame_to_sample(parent)))
    record = {"dataset": PARENT, "key": "example", "measure": {"tach": {"role": role}}}
    tach = {
        "role": "audio",  # declared preference; the measurement's role takes precedence
        "n_windows": 1,
        "n_accepted": 1,
        "audio_t_start_s": t0,
        "times": np.array([[0.0, 1.0, 0.5]], np.float32),
        "accepted": np.array([True]),
        "rates": np.array([[50.0]], np.float32),
        "uncertainty": np.array([[0.1]], np.float32),
    }
    got = ra.annotate_frame(parent, record, tach, {})
    acoustic = got["rps_acoustic"]
    assert (acoustic.t_start, acoustic.t_end) == (t0, t0 + 2.0)
    np.testing.assert_allclose(acoustic.tindex.abs_stamps, [t0 + 0.5])
    assert_overlay_preserves(got, parent)
    tach["times"] = np.array([[2.5, 3.5, 3.0]], np.float32)
    with pytest.raises(ValueError, match=rf"example role '{role}'.*outside the audio domain"):
        ra.annotate_frame(parent, record, tach, {})


@pytest.mark.network
def test_live_release_roundtrip():
    """Post-publication proof against R2: run as a uni-cpu job, not on a laptop
    (reads one sample per annotated parent). Exact versions, stored parent
    shards, canonical Frame join, and no borrowing by other versions."""
    repo = streams.open_repository()
    rel = ra.open_release(repo=repo)
    assert rel.parents(), "release lists no parents"
    for parent in rel.parents():
        idx = rel.index(parent.dataset, parent.version)
        manifest = repo.manifest(parent.dataset, parent.version)
        assert idx.parent_shards() == [(s.digest, s.size, s.num_samples) for s in manifest.shards]
        assert idx.count() == idx.n_rows == parent.n_rows <= manifest.num_samples
        assert sum(idx.facet_counts()["rig_mount"].values()) == idx.n_rows
        # A small page ordered by duration bounds the locate() calls.
        rows = idx.select({"has_audio": True}, limit=20, order="duration_s")
        if rows:
            key = min(rows, key=lambda r: _location(idx, r["key"]).length)["key"]
            print(f"round-trip {parent.dataset}/{key}", flush=True)
            frame = ra.load_frame(parent.dataset, key, version=parent.version, release=rel)
            loc = _location(idx, key)
            spec = ra.media_field(parent.dataset, key, loc)
            shard = manifest.shards[loc.shard]
            if spec is not None:
                # Media must stay uncached even during the live verification.
                assert frame["meta"]["media"]["codec"] == "aac"
                assert frame["audio"].tindex.sr == 48_000
                assert frame["audio"].t_start == frame["meta"]["media"]["t_start_s"]
            else:
                with repo.open_shard(shard) as path:  # independent whole-shard read
                    from dload.pack import PackReader

                    with PackReader(path) as reader:
                        raw = ra.decode_sample(
                            idx.layout,
                            key,
                            reader.read_key(key)[1],
                            dataset=parent.dataset,
                            parent_meta=idx.meta()["parent_meta"],
                            has_audio=True,
                        )
                assert_overlay_preserves(frame, raw)
            cond = frame["meta"]["recording_conditions"]
            record = _record_of(idx, key, windows=False)
            assert all(cond[f] == record[f] for f in ra.FACET_NAMES)
            tach = idx.tach_arrays(key)
            claims = tach is not None and bool(
                np.isfinite(np.where(tach["accepted"][:, None], tach["rates"], np.nan)).any()
            )
            assert ("rps_acoustic" in frame) == claims
        else:
            print(f"index-only {parent.dataset}: no audio rows", flush=True)
        others = [m.version for m in repo.versions(parent.dataset)]
        unannotated = [
            v
            for v in others
            if (parent.dataset, v) not in {(p.dataset, p.version) for p in rel.parents()}
        ]
        if unannotated:
            with pytest.raises(ra.NotAnnotatedError):
                rel.index(parent.dataset, unannotated[0])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_media_container_audio_ranged(tmp_path, monkeypatch):
    """A listed MP4's AAC track is decoded by packet ranges on its own clock."""
    mp4 = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=32x32:rate=5:duration=1",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1", "-ac", "2",
         "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(mp4)],
        check=True,
    )  # fmt: skip
    data = mp4.read_bytes()
    repo = _repo(tmp_path)
    raw = repo.commit(
        "MEDIA-RAW-TEST",
        [("cam/notes", {"txt": b"hi"}), ("cam/a__Video", {"mp4": data})],
        meta={"layout": "raw-files"},
    )
    monkeypatch.setitem(
        ra_storage.MEDIA_AUDIO, ("MEDIA-RAW-TEST", "cam/a__Video"), ("mp4", len(data))
    )
    stage = tmp_path / "stage"
    with ra.IndexWriter(stage / f"MEDIA-RAW-TEST@{raw.version}.sqlite", dataset="MEDIA-RAW-TEST",
                        version=raw.version, layout="raw-files", parent_shards=raw.shards) as w:  # fmt: skip
        for loc, key, _ in ra.located_samples("MEDIA-RAW-TEST", raw.version, repo=repo):
            w.add(key, _record(), info={"has_audio": key == "cam/a__Video"}, location=loc)
        w.close()
    cold = _repo(tmp_path, cache="cold-media-cache")
    rel = ra.open_release(stage, repo=cold)
    frame = ra.load_frame("MEDIA-RAW-TEST", "cam/a__Video", version=raw.version, release=rel)
    assert isinstance(cold.remote, LocalRemote)
    assert cold.remote.op_counts.get("get_to_file", 0) == 0  # no shard copy
    assert not any((tmp_path / "cold-media-cache" / "shards").rglob("*"))
    audio = frame["audio"]
    assert audio.dim_size("time") == 48_000  # discard encoder priming and final padding
    assert audio.dims == ("mic", "time") and audio.tindex.sr == 48000
    media = frame["meta"]["media"]
    assert (
        media["codec"] == "aac" and media["channels"] == 2 and audio.t_start == media["t_start_s"]
    )
    # The encoder-priming packet (pts -21.3 ms) decodes to nothing; the first sample is pts 0.
    assert media["t_start_s"] == 0.0
    t = np.arange(audio.data.shape[1]) / 48000.0 + audio.t_start
    ref = np.sin(2 * np.pi * 440 * t)
    mid = slice(4800, 43200)
    assert np.corrcoef(audio.data[0, mid], ref[mid])[0, 1] > 0.99  # the source tone, on its clock
    meta_only = ra.load_frame(
        "MEDIA-RAW-TEST", "cam/a__Video", version=raw.version, release=rel, audio=False
    )
    assert "audio" not in meta_only
    with pytest.raises(ValueError, match="media container"):
        ra.decode_sample(
            "raw-files", "cam/a__Video", {"mp4": data}, dataset="MEDIA-RAW-TEST", has_audio=True
        )


def test_invalid_mp4_does_not_disclose_input(tmp_path, capfd):
    invalid = tmp_path / "invalid.mp4?X-Amz-Signature=private-test-token"
    invalid.write_bytes(b"not an MP4")
    with invalid.open("rb") as file:

        def read(start: int, end: int) -> bytes:
            file.seek(start)
            return file.read(end - start)

        source = ra_storage.MediaInput(invalid.stat().st_size, read)
        with pytest.raises(ValueError, match="malformed MP4 box") as exc:
            ra.decode_media_audio(source)
    output = capfd.readouterr()
    diagnostic = str(exc.getrepr(style="short")) + output.out + output.err
    assert "X-Amz-Signature" not in diagnostic
    assert "private-test-token" not in diagnostic
