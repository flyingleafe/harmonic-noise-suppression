"""Bounded metadata scans and audio views over synthetic dload packs (no network)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import tdseries as td
from dload.pack import PackWriter

from data_processing.frames import audio_series
from data_processing.recording_annotations.scan import (
    SeriesRef,
    ShardBytes,
    ShardOpener,
    finite_runs,
    read_view,
    scan_shard,
)
from data_processing.streams import frame_to_sample


def _pack(root: Path, samples: list[tuple[str, dict[str, bytes]]]) -> tuple[str, int]:
    tmp = root / "tmp.pack"
    w = PackWriter(tmp)
    for key, fields in samples:
        w.add(key, fields)
    digest, size, _ = w.finish()
    dest = root / "shards" / digest[:2] / digest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp.rename(dest)
    return digest, size


def _frame(audio: td.Series, **extra: object) -> td.Frame:
    stamps = np.array([0, 250_000_000, 500_000_000, 1_000_000_000], dtype=np.int64)
    track = td.Series(
        np.arange(8.0).reshape(2, 4),
        ("rotor", "time"),
        {"time": td.StampIndex(stamps, 0, 2_000_000_000)},
    )
    meta = td.Frame({"recording_id": "r1", "system": td.Frame({"n_rotors": 4}), **extra})
    return td.Frame({"audio": audio, "rps": track, "meta": meta})


def test_tdframe_scan_and_views(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    sr = 8000
    c_order = rng.standard_normal((3, 2 * sr)).astype(np.float32)
    fortran = np.asfortranarray(rng.standard_normal((2, 2 * sr)).astype(np.float32))
    mono = rng.standard_normal(2 * sr).astype(np.float32)
    shifted = td.uniform(mono, sr, dims=("time",), t_start=1.25)
    samples = [
        ("c", frame_to_sample(_frame(audio_series(c_order, sr)))),
        ("f", frame_to_sample(_frame(td.uniform(fortran, sr, dims=("mic", "time"), t_start=0.0)))),
        ("m", frame_to_sample(_frame(shifted))),
        ("_meta", {"metadata": b"{}"}),
    ]
    digest, size = _pack(tmp_path, samples)
    shard = ShardOpener([tmp_path]).open(digest, size)
    assert shard.source == "local"
    scans = {
        s.key: s for s in scan_shard(shard, 0, dataset="X", version="v" * 64, layout="tdframe-v1")
    }
    assert set(scans) == {"c", "f", "m"}

    c = scans["c"]
    assert c.meta["system"]["n_rotors"] == 4 and c.info["has_audio"] and c.info["duration_s"] == 2.0
    rps = c.info["series"]["rps"]
    assert rps["index"] == "stamps" and rps["t_end_s"] == 1.0 and not rps["audio"]
    x, t0 = read_view(shard, c.series["audio"], 0.5, 1.0, [2, 0])
    assert t0 == 0.5 and np.array_equal(x, c_order[[2, 0], sr // 2 : sr])

    f = scans["f"]
    assert f.series["audio"].fortran
    x, _ = read_view(shard, f.series["audio"], 0.25, 0.75, None)
    assert np.array_equal(x, np.asarray(fortran)[:, sr // 4 : 3 * sr // 4])

    m = scans["m"]
    assert m.info["series"]["audio"]["t_start_s"] == 1.25
    x, t0 = read_view(shard, m.series["audio"], 1.5, 2.0)
    assert t0 == 1.5 and np.array_equal(x[0], mono[sr // 4 : 3 * sr // 4])
    shard.close()


def test_wav_field_scan_and_int_pcm(tmp_path: Path) -> None:
    sr = 16000
    pcm = (np.arange(2 * sr * 2, dtype=np.int64) % 2000 - 1000).astype(np.int16).reshape(-1, 2)
    buf = io.BytesIO()
    sf.write(buf, pcm, sr, format="WAV", subtype="PCM_16")
    digest, size = _pack(tmp_path, [("dir/a", {"wav": buf.getvalue(), "json": b'{"k": 1}'})])
    shard = ShardOpener([tmp_path]).open(digest, size)
    (s,) = list(scan_shard(shard, 0, dataset="X", version="v" * 64, layout=""))
    assert s.info["series"]["wav"]["shape"] == [2, 2 * sr] and s.meta["json_fields"]["json"] == {
        "k": 1
    }
    x, _ = read_view(shard, s.series["wav"], 1.0, 1.5, [1])
    assert np.allclose(x[0], pcm[sr : sr + sr // 2, 1] / 32768.0)


def test_file_fields_follow_decoder_rules(tmp_path: Path) -> None:
    sr = 8000
    buf = io.BytesIO()
    sf.write(buf, np.zeros((sr, 1), np.float32), sr, format="WAV", subtype="FLOAT")
    wav = buf.getvalue()
    samples = [
        ("sample_00000", {"mixture": wav, "meta": b'{"split": "valid"}', "metadata": b"Source: x"}),
        ("dir/clip.wav", {"data": wav}),
        ("dir/hidden", {"bin": wav, "txt": b"notes"}),
        ("dir/doc", {"md": b"# readme"}),
    ]
    digest, size = _pack(tmp_path, samples)
    shard = ShardOpener([tmp_path]).open(digest, size)
    fmap = {"mixture": "mixture.wav", "meta": "meta.json"}
    scans = {
        s.key: s
        for s in scan_shard(shard, 0, dataset="X", version="v" * 64, layout="", fields_map=fmap)
    }
    mix = scans["sample_00000"]
    assert mix.info["audio_entries"] == ["mixture"] and mix.meta == {
        "split": "valid",
        "metadata_text": "Source: x",
    }
    assert scans["dir/clip.wav"].info["audio_entries"] == ["data"]
    hidden = scans["dir/hidden"].info["series"]["bin"]
    assert hidden["sniffed"] and hidden["audio"] and hidden["duration_s"] == 1.0
    assert not scans["dir/doc"].info["has_audio"]


def test_audio_rate_tracks_are_not_audio() -> None:
    def ref(path: str, dtype: str = "<f4") -> SeriesRef:
        return SeriesRef(
            path,
            path,
            "npy",
            (2, 100),
            dtype,
            ("mic", "time"),
            time_axis=1,
            index="grid",
            rate_hz=44100.0,
            t_start_s=0.0,
            t_end_s=1.0,
            n_time=100,
        )

    assert ref("array_audio").is_audio
    assert not ref("sample_time").is_audio and not ref("angle_vad").is_audio
    assert not ref("audio", "<i8").is_audio


def test_finite_runs() -> None:
    x = np.ones((2, 20), np.float32)
    x[1, 5] = np.nan
    x[0, 12:14] = np.inf
    assert finite_runs(x, 3) == [(0, 5), (6, 12), (14, 20)]
    assert finite_runs(x, 7) == []
    with pytest.raises(IndexError):
        ref = SeriesRef(
            "a",
            "a",
            "npy",
            (1, 4),
            "<f4",
            ("mic", "time"),
            time_axis=1,
            index="grid",
            rate_hz=1.0,
            t_start_s=0.0,
            t_end_s=4.0,
            n_time=4,
        )
        read_view(ShardBytes("0" * 64, 1, path=Path("/nonexistent"), remote=None), ref, 0, 1, [3])
