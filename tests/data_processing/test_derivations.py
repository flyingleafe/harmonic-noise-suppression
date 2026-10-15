"""Tests for data_processing.derivations (dload derived-dataset declarations).

The fast checks (specs integrity, fingerprint stability, wav encoding, meta
shape) are torch-free — importing ``derivations`` must not pull torch, so
offline fingerprinting/adoption works on any box. The final round-trip test
imports ``streams`` (→ torch) lazily inside the test body, so
``-k "not roundtrip"`` stays torch-free.
"""

from __future__ import annotations

import io
import itertools
import json
import subprocess
import sys
from pathlib import Path

import dload
import numpy as np
import pytest
import soundfile as sf

import data_processing.derivations as der

from .conftest import speech_dataset  # noqa: F401

SR = 16000


def test_import_is_torch_free():
    # Check a *fresh* interpreter: pytest shares sys.modules across the session,
    # so a sibling test file that imports torch would otherwise false-fail this.
    code = "import sys, data_processing.derivations as _; sys.exit('torch' in sys.modules)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"importing derivations pulled torch:\n{r.stderr}"


def test_wav_bytes_roundtrip_mono_and_multichannel():
    rng = np.random.default_rng(0)
    for shape in [(SR,), (SR, 8)]:
        arr = (rng.standard_normal(shape) * 0.1).astype(np.float32)
        raw, sr = sf.read(io.BytesIO(der.wav_bytes(arr, SR)), dtype="float32")
        assert sr == SR
        # Default WAV subtype (matches the disk-writing CLIs) is PCM_16, so
        # allow one 16-bit quantization step rather than exact equality.
        np.testing.assert_allclose(raw.reshape(arr.shape), arr, atol=2e-4)


def test_dataset_meta_shape():
    for name in der.SPECS:
        meta = der.dataset_meta(name)
        assert meta["layout"] == der.spec_layout(name)
        if meta["layout"] == "sample-dir-v1":
            assert meta["fields"] == der.SPECS[name]["fields"]
            assert meta["meta_sample"] == {
                "key": "_meta",
                "fields": {"metadata": "metadata.json"},
            }
            assert "mixture" in meta["fields"]


def test_specs_integrity():
    for name, entry in der.SPECS.items():
        assert entry["generator"] in der._GENERATORS, name
        assert isinstance(entry["adopt_only"], bool), name
        gen = entry["gen"]
        assert "recipe_version" in gen, f"{name} missing gen['recipe_version']"
        kind = entry["generator"]
        if kind in ("dregon_lm", "dn_lm"):
            assert entry["fields"], name
            for key in ("seed", "num_samples", "split", "parents", "params"):
                assert key in gen, f"{name} missing gen[{key!r}]"
            assert gen["split"] in ("train", "valid"), name
        elif kind == "se_valid":
            for key in ("seed", "categories", "category_noise", "per_snr", "snr_grid"):
                assert key in gen, f"{name} missing gen[{key!r}]"
        elif kind in ("source_frames", "raw_subset"):
            assert "source" in gen, f"{name} missing gen['source']"
        elif kind == "frame_subset":
            assert "parent" in gen and "include_keys" in gen, name
        elif kind == "beatvk_valid":
            assert "parents" in gen and "recordings" in gen, name
        elif kind == "real_valid_panel":
            assert entry["fields"]["meta"] == "meta.json", name
            for key in ("seed", "num_samples", "split", "params", "sources", "speech_mix"):
                assert key in gen, f"{name} missing gen[{key!r}]"
            for source in gen["sources"]:
                assert "@" in source["dataset"] and source["label_key"], f"{name}: {source}"
        # Parents are pinned dload URIs, never local data/ paths.
        for key in ("parents",):
            for uri in (gen.get(key) or {}).values():
                assert uri.startswith("dload:") and "@" in uri, f"{name}: {uri}"
        for key in ("parent",):
            if gen.get(key):
                assert gen[key].startswith("dload:") and "@" in gen[key], name
        for spec in gen.get("noise_sources") or []:
            assert "@" in spec["dataset"], f"{name}: unpinned noise source {spec}"


def test_fingerprint_stable_and_unique():
    fps = {name: der.fingerprint(name) for name in der.SPECS}
    # deterministic
    assert all(der.fingerprint(n) == fps[n] for n in der.SPECS)
    # every dataset gets a distinct derivation identity
    assert len(set(fps.values())) == len(fps)


def test_fingerprint_tracks_recipe_version(monkeypatch):
    """Bumping recipe_version must change the fingerprint (stale-snapshot guard)."""
    name = "DN-LM-train"
    before = der.fingerprint(name)
    bumped = dict(der.SPECS[name])
    bumped["gen"] = {**bumped["gen"], "recipe_version": 999}
    monkeypatch.setitem(der.SPECS, name, bumped)
    assert der.fingerprint(name) != before


def _lock_datasets() -> dict[str, str]:
    """``dload.lock``'s pinned ``{dataset: version}``, repo-root relative."""
    import tomllib

    lock_path = Path(der.__file__).resolve().parents[2] / "dload.lock"
    with lock_path.open("rb") as fh:
        return tomllib.load(fh)["datasets"]


def test_parents_match_dload_lock():
    """Pin-drift guard: PARENTS must carry the current dload.lock versions.
    Update deliberately — a change mints new derivation identities."""
    lock = _lock_datasets()
    for key, uri in der.PARENTS.items():
        name, _, sha = uri.removeprefix("dload:").partition("@")
        assert lock.get(name) == sha, (
            f"PARENTS[{key!r}] stale: {sha[:12]} != lock {lock.get(name, '')[:12]}"
        )


def test_lock_coverage():
    """Every dload.lock dataset is covered: a derivation spec, a raw source
    registry entry (or its frames dataset), a published artefact tree, or a
    declared historical pin."""
    from data_processing import sources

    lock = _lock_datasets()
    frames_names = {s.frames_name for s in sources.REGISTRY.values()}
    for name in lock:
        covered = (
            name in der.SPECS
            or name in sources.REGISTRY
            or name in frames_names
            or name in der.ARTEFACT_PINS
            or name in der.HISTORICAL_PINS
        )
        assert covered, f"{name}: no spec, source entry, artefact pin, or historical pin"


def test_build_pipeline_is_fingerprintable():
    # No ValueError from dload's fingerprint = the partial(module-fn, json-spec)
    # bridge is legal (no lambdas/locals/non-JSON params).
    for name in der.SPECS:
        assert isinstance(der.build_pipeline(name).fingerprint(), str)


def test_sample_convention_roundtrips_through_streams(tmp_path, monkeypatch):
    """The (key, {field: bytes}) convention derivations emits must decode and
    reconstruct through the real consumer (streams), using derivations'
    encoders + dataset_meta — the plumbing proof for a materialized dataset."""
    from dload.cache import ShardCache
    from dload.remote import LocalRemote
    from dload.repo import Repository

    import data_processing.streams as streams

    repo = Repository(
        LocalRemote(tmp_path / "remote"),
        ShardCache(tmp_path / "cache", None),
        lock_path=tmp_path / "dload.lock",
    )
    monkeypatch.setattr(streams, "_repository", repo)

    rng = np.random.default_rng(1)
    n, channels, hop = 3, 8, 512
    samples = []
    for i in range(n):
        mix = (rng.standard_normal((SR, channels)) * 0.1).astype(np.float32)
        rps = rng.uniform(20.0, 100.0, size=(4, 29)).astype(np.float32)
        samples.append(
            (
                f"sample_{i:05d}",
                {
                    "mixture": der.wav_bytes(mix, SR),
                    "vocals": der.wav_bytes(mix, SR),
                    "noise": der.wav_bytes(mix, SR),
                    "rps": dload.codecs.npy_bytes(rps),
                    "meta": dload.codecs.json_bytes(
                        {"id": f"sample_{i:05d}", "input_snr": -5.0 - i}
                    ),
                },
            )
        )
    samples.append(
        ("_meta", {"metadata": dload.codecs.json_bytes({"train": [{"id": "sample_00000"}]})})
    )

    name = "DREGON-LM-V4-train"
    manifest = repo.commit(name, samples, meta=der.dataset_meta(name))
    dataset = dload.Dataset(repo, manifest)

    # (a) streaming decode → training Frame
    from functools import partial

    frames = list(
        streams.to_frames(
            dataset.samples(), partial(streams.decode_dregon_lm, sample_rate=SR, hop_length=hop)
        )
    )
    assert len(frames) == n  # _meta dropped
    assert frames[0]["mixture"].dims == ("mic", "time")
    assert frames[0]["mixture"].shape == (channels, SR)
    assert frames[0]["rps"].dims == ("rotor", "time")

    # (b) ensure_local reconstructs the sample-dir tree + split metadata.json
    root = streams.ensure_local(name)
    assert (root / "sample_00000" / "mixture.wav").exists()
    assert (root / "sample_00000" / "rps.npy").exists()
    assert (root / "sample_00000" / "meta.json").exists()
    assert (root / "metadata.json").exists()  # from the _meta bookkeeping sample


def test_se_valid_category_grid_and_snr(patched_repo, speech_dataset):  # noqa: F811
    """``iter_se_valid_category`` yields per_snr clips per grid point, each at
    its grid SNR, with the clean target being the mixture's speech component.

    Guards the SE-valid rebuild path in ``data_processing.derivations``
    (``generate_se_valid`` / ``iter_se_valid_category``).
    """
    from .conftest import SPEECH_DATASET

    snr_grid = [-20.0, -5.0]
    clips = list(
        der.iter_se_valid_category(
            "probe",
            [{"kind": "audio_pool", "dataset": SPEECH_DATASET}],
            per_snr=2,
            snr_grid=snr_grid,
            duration_s=0.5,
            sample_rate=SR,
            seed=7,
            heldout_speakers=["19"],
            librispeech=SPEECH_DATASET,
        )
    )
    assert len(clips) == 2 * len(snr_grid)
    assert [k for k, _ in clips] == [
        "probe_snr-20_0000",
        "probe_snr-20_0001",
        "probe_snr-05_0002",
        "probe_snr-05_0003",
    ]
    for _, frame in clips:
        mixture = np.asarray(frame["mixture"].data, np.float32).reshape(-1)
        target = np.asarray(frame["target"].data, np.float32).reshape(-1)
        snr = float(frame["meta"]["input_snr"])
        assert snr in snr_grid
        noise = mixture - target
        assert 10.0 * np.log10(np.mean(target**2) / np.mean(noise**2)) == pytest.approx(
            snr, abs=1e-2
        )


def test_se_valid_speaker_holdout_is_respected(patched_repo, speech_dataset):  # noqa: F811
    """The valid speech pool draws ONLY the held-out speakers — the guarantee
    that no speaker is shared with a training policy's ``exclude`` list."""
    from data_processing.online_mixing import build_speech_stream

    from .conftest import SPEECH_DATASET

    pipe = build_speech_stream(
        {"dataset": SPEECH_DATASET, "subpath": "LibriSpeech/train-clean-100", "include": ["19"]},
        sample_rate=SR,
        window_s=0.5,
        lanes=1,
        seed=3,
    )
    assert len(list(itertools.islice(pipe, 6))) == 6  # cycles the 4 speaker-19 files


def test_se_valid_frame_crops_the_off_by_one_noise_window():
    """A noise window one sample longer than the speech lane (inclusive
    time-slice bounds) must be cropped, not crash the broadcast — the defect
    that aborted the first SE-valid v2 re-derivation."""
    import tdseries as td

    from data_processing.frames import audio_series

    target_len = SR // 2
    noise_tf = td.Frame({"audio": audio_series(np.ones((1, target_len + 1), np.float32), SR)})
    speech = [np.full(target_len, 0.1, np.float32)]
    frame = der._se_valid_frame("probe", SR, [-10.0], 1, 0, noise_tf, speech, 0.0)
    mixture = np.asarray(frame["mixture"].data)
    target = np.asarray(frame["target"].data)
    assert mixture.shape[-1] == target_len
    assert target.shape[-1] == target_len
    noise = mixture - target
    assert 10.0 * np.log10(np.mean(target**2) / np.mean(noise**2)) == pytest.approx(-10.0, abs=1e-2)


# ─── real_valid_panel (rp-real-valid-*) ───────────────────────────────────────


#: One-second clips per 4 s fixture recording (audio and telemetry both span 0-4 s).
_N = 4
_PANEL_DREGON = "PANEL-DREGON-TEST"


@pytest.fixture
def panel_dregon(patched_repo, dregon_command_values, dregon_measured_values):
    """Room-1-style recordings under their real ids (the source type is read
    off the id) + a room-2 flight that publishes no motors_measured."""
    import data_processing.streams as streams
    from data_processing.frames import make_recording_frame

    from .conftest import _audio_series, _motor_series

    def rec(rid: str, seed: int, measured: bool):
        tracks = {
            "audio": _audio_series(SR, seed),
            "motors_command": _motor_series(dregon_command_values),
        }
        if measured:
            tracks["motors_measured"] = _motor_series(dregon_measured_values)
        frame = make_recording_frame(tracks, meta={"recording_id": rid})
        return rid, streams.frame_to_sample(frame)

    patched_repo.commit(
        _PANEL_DREGON,
        [
            rec("free-flight_nosource_room1", 0, True),
            rec("free-flight_speech-low_room1", 1, True),
            rec("updown_nosource_room2", 2, False),
        ],
        meta={"layout": streams.TDFRAME_LAYOUT},
    )


def _panel_gen(**overrides):
    from .conftest import MICHAELS_DATASET, SPEECH_DATASET

    gen = {
        "recipe_version": 1,
        "seed": 43,
        "num_samples": 4 * _N,  # DREGON nosource + loudspeaker, MD2 raw, MD2 mixed
        "split": "valid",
        "params": {
            "sample_rate": SR,
            "sample_duration": 1.0,
            "min_motor_rps": 0.0,
            "max_non_overlapping": True,
        },
        "sources": [
            {
                "dataset": _PANEL_DREGON,
                "drone": "dregon",
                "label_key": "motors_measured",
                "recording_ids": ["free-flight_nosource_room1", "free-flight_speech-low_room1"],
            },
            {
                "dataset": MICHAELS_DATASET,
                "drone": "md2",
                "label_key": "rps",
                "recording_ids": ["FLY124"],
            },
        ],
        "parents": {"speech": f"dload:{SPEECH_DATASET}"},
        "speech_mix": {
            "drone": "md2",
            "subpath": "LibriSpeech/train-clean-100",
            "seed": 5,
            "snr_db": [-5.0, -5.0, -20.0, -20.0],
            # Above the recordings' 0.05 rms: the floor, not the noise, is the reference.
            "snr_ref_floor_rms": 0.1,
        },
    }
    gen.update(overrides)
    return gen


def _decode_panel(samples):
    out = []
    for key, fields in samples:
        if key.startswith("_"):
            continue
        mix, _ = sf.read(io.BytesIO(fields["mixture"]), dtype="float32", always_2d=True)
        out.append((mix.T, np.load(io.BytesIO(fields["rps"])), json.loads(fields["meta"])))
    return out


_PANEL_PARENTS = ("panel_dregon", "michaels_frames_dataset", "speech_dataset")


@pytest.mark.usefixtures(*_PANEL_PARENTS)
def test_real_valid_panel_order_meta_and_labels(dregon_measured_values):
    clips = _decode_panel(der.generate_real_valid_panel(_panel_gen()))
    metas = [m for _, _, m in clips]
    assert [m["clip_index"] for m in metas] == list(range(4 * _N))
    assert all(tuple(sorted(m)) == tuple(sorted(der.REAL_VALID_PANEL_META_KEYS)) for m in metas)
    assert [(m["dataset"], m["source_type"]) for m in metas] == (
        [("dregon", "nosource")] * _N
        + [("dregon", "loudspeaker")] * _N
        + [("md2", "nosource")] * _N
        + [("md2", "mixed")] * _N
    )
    assert [m["snr_db"] for m in metas] == [None] * (3 * _N) + [-5.0, -5.0, -20.0, -20.0]
    # The mixed copies follow the raw clips of their drone, in clip order.
    raw_md2, mixed_md2 = metas[2 * _N : 3 * _N], metas[3 * _N :]
    assert [m["t_start_s"] for m in mixed_md2] == [m["t_start_s"] for m in raw_md2]
    utts = [u for m in mixed_md2 for u in m["speech_utterances"]]
    assert len(utts) == _N * 2 and len(set(utts)) == len(utts)  # one distinct utterance per lane
    assert all(m["speech_utterances"] is None for m in metas[: 3 * _N])
    # DREGON scores the declared motors_measured track (55+r), not the command (60+r).
    rps0 = clips[0][1]
    np.testing.assert_allclose(np.median(rps0, axis=1), dregon_measured_values[:, 0])
    np.testing.assert_array_equal(clips[3 * _N][1], clips[2 * _N][1])


@pytest.mark.usefixtures(*_PANEL_PARENTS)
def test_real_valid_panel_mixes_at_snr_against_the_reference_floor():
    clips = _decode_panel(der.generate_real_valid_panel(_panel_gen()))
    for raw, mixed in zip(clips[2 * _N : 3 * _N], clips[3 * _N :], strict=True):
        speech = mixed[0].astype(np.float64) - raw[0].astype(np.float64)
        snr = 10.0 * np.log10(np.mean(speech**2) / 0.1**2)
        assert snr == pytest.approx(mixed[2]["snr_db"], abs=0.05)


@pytest.mark.usefixtures(*_PANEL_PARENTS)
def test_real_valid_panel_refuses_test_recordings_and_bad_specs():
    test_src = {
        "dataset": _PANEL_DREGON,
        "drone": "dregon",
        "label_key": "motors_measured",
        "recording_ids": ["free-flight_speech-high_room1"],
    }
    with pytest.raises(ValueError, match="test recordings"):
        list(der.generate_real_valid_panel(_panel_gen(sources=[test_src])))
    no_label = {**test_src, "recording_ids": ["updown_nosource_room2"]}  # command only
    with pytest.raises(KeyError, match="motors_measured"):
        list(der.generate_real_valid_panel(_panel_gen(sources=[no_label])))
    mix = {**_panel_gen()["speech_mix"], "snr_db": [-5.0, -20.0]}
    with pytest.raises(ValueError, match=f"2 SNRs for {_N} md2 clips"):
        list(der.generate_real_valid_panel(_panel_gen(speech_mix=mix)))
    with pytest.raises(ValueError, match=f"panel has {4 * _N} clips, spec says 99"):
        list(der.generate_real_valid_panel(_panel_gen(num_samples=99)))
