"""Derived / aggregate / simulated recording-condition rules (pure metadata, no I/O)."""

from __future__ import annotations

import json

import pytest

from data_processing.recording_annotations import derived, legacy, native, rules, schema


def _norm(dataset: str, key: str, meta: dict, info: dict) -> dict:
    info = {"has_audio": True, "fields": ["audio"], **info}
    return schema.normalize(derived.classify(dataset, key, meta, info), has_audio=info["has_audio"])


def test_scope_is_the_remaining_inventory():
    parents = frozenset(rules.PARENT_VERSIONS)
    assert derived.DATASETS.issubset(parents)
    assert not derived.DATASETS & (legacy.DATASETS | native.DATASETS)
    assert parents == derived.DATASETS | legacy.DATASETS | native.DATASETS
    assert len(derived.DATASETS) == 36


def test_summary_rows_pick_the_dataset_split():
    blob = json.dumps(
        {
            "train": [
                {
                    "id": "sample_00000",
                    "noise_source": "spinning_nosource_room2",
                    "noise_start_time": 1.0,
                }
            ],
            "valid": [
                {
                    "id": "sample_00000",
                    "recording_id": "free-flight_speech-low_room1",
                    "start_time": 8.0,
                }
            ],
        }
    ).encode()
    train = derived.summary_rows("DREGON-LM-V4-train", {"_meta": blob})
    valid = derived.summary_rows("DREGON-LM-V4-valid", {"_meta": blob})
    assert train["sample_00000"]["noise_source"] == "spinning_nosource_room2"
    assert valid["sample_00000"]["recording_id"] == "free-flight_speech-low_room1"
    assert derived.summary_rows("drone_audio", {}) == {}
    with pytest.raises(ValueError):
        derived.summary_rows("rp-real-test-v1", {"_meta": blob})


def test_eval_probe_rows_verify_their_metadata_txt():
    rows = derived.summary_rows(
        "DREGON-LM-rps_eval_long_samples",
        {"sample_000": b"Source: datasets/DREGON-LM/valid/sample_00114\nStart time: 0.031s\n"},
    )
    assert rows["sample_000"]["noise_source"] == "free-flight_nosource_room2"
    with pytest.raises(ValueError):
        derived.summary_rows(
            "DREGON-LM-rps_eval_long_samples",
            {"sample_000": b"Source: datasets/DREGON-LM/valid/sample_00001\n"},
        )


def test_real_valid_clip_is_not_a_digital_mix_and_binds_to_its_parent():
    meta = {
        "id": "sample_00000",
        "recording_id": "free-flight_speech-low_room1",
        "start_time": 8.0,
        "duration": 8.0,
        "source_type": "speech",
    }
    rec = _norm(
        "DREGON-LM-V4-valid",
        "sample_00000",
        meta,
        {"fields": ["mixture", "rps"], "duration_s": 8.0},
    )
    assert rec["sound_content"] == "co_recorded"
    assert rec["parent"]["binding"] == "crop" and rec["parent"]["offset_s"] == 8.0
    assert rec["wind_role"] == "mixture"
    fly = dict(meta, recording_id="michaels_FLY124", source_type="nosource")
    rec = _norm("DREGON-LM-V4-michaels-valid", "sample_00000", fly, {"fields": ["mixture", "rps"]})
    assert rec["parent"] == rec["parent"] | {
        "dataset": "michaels-frames",
        "key": "FLY124",
        "binding": "none",
    }


def test_synthesized_mix_and_motor_combo():
    meta = {
        "id": "sample_00000",
        "noise_source": "free-flight_nosource_room1",
        "noise_start_time": 3.5,
    }
    rec = _norm(
        "DREGON-LM-V4-train",
        "sample_00000",
        meta,
        {"fields": ["mixture", "noise", "rps", "vocals"], "duration_s": 1.0},
    )
    assert rec["sound_content"] == "digital_mix" and rec["parent"]["binding"] == "crop"
    assert rec["tach_candidate"] is False and rec["wind_role"] == "noise"
    combo = {
        "id": "sample_00001",
        "noise_source": "motor_combo_[(4, 50), (1, 60)]",
        "noise_start_time": 0.0,
    }
    rec = _norm(
        "DREGON-LM-V2-train",
        "sample_00001",
        combo,
        {"fields": ["mixture", "noise", "rps", "vocals"]},
    )
    assert rec["rotor_activity"] == "subset" and rec["n_active"] == 2
    assert rec["nominal_rps"] == [60.0, 0.0, 0.0, 50.0] and rec["parent"]["binding"] == "none"
    assert rec["wind_role"] == "noise" and rec["tach_role"] == "noise"
    assert "separate captures" in rec["source_detail"]["composition"]


def test_dn_lm_wind_reads_the_drone_noise_stem():
    meta = {
        "id": "sample_00000",
        "input_snr": -21.7,
        "noise_source": "B_S2_D1_112-bebop_003_.wav",
        "speech_distance": 5.4,
    }
    rec = _norm(
        "DN-LM-train",
        "sample_00000",
        meta,
        {"fields": ["meta", "mixture", "noise", "vocals"], "duration_s": 1.0},
    )
    assert rec["sound_content"] == "digital_mix" and rec["wind"] == "unknown"
    assert rec["wind_role"] == "noise" and rec["tach_candidate"] is False


def test_dads_kaist_block_follows_the_audited_ordering_rule():
    assert derived.dads_kaist_origin(1592)["fault_code"] == "MF1"
    assert derived.dads_kaist_origin(111918) == {
        "drone": "C",
        "direction": "B",
        "fault_code": "MF3",
        "split_group": "test_or_valid",
        "mic": "mic1",
    }
    assert derived.dads_kaist_origin(159917)["direction"] == "R"
    assert derived.dads_kaist_origin(109591)["split_group"] == "train"
    rec = _norm("drone-detection-samples", "123456_drone-2000", {}, {})
    assert (
        rec["rig_mount"] == "tethered" and rec["blade_count"] == "2" and not rec["tach_candidate"]
    )
    assert (
        _norm("drone-detection-samples", "000000_no-drone-1", {}, {})["sound_content"] == "no_drone"
    )
    salford = _norm("drone-detection-samples", "000001_drone-1400", {}, {})
    assert salford["relative_motion"] == "unknown" and salford["environment"] == "outdoors"
    assert salford["rig_mount"] == "unknown" and salford["propulsion"] == "unknown"
    audrok = _norm("drone-detection-samples", "000002_drone-1550", {}, {})
    assert audrok["rig_mount"] == "free_flight" and audrok["propulsion"] == "propellers"
    assert audrok["relative_motion"] == "unknown"


def test_droneaudioset_paths():
    rec = derived.droneaudioset_path_facts(
        "drone-only-recordings/drone1-only/mic-dist-25cm/throttle-high/mic2_8array-down-File2.wav"
    )
    assert rec["rig_mount"] == "fixed_stand" and rec["blade_count"] == "2"
    assert rec["speed_regime"] == "constant" and rec["speed_supplied"] == "no"
    assert rec["wind"] == "unknown" and "downwash_exposure" in rec["source_detail"]
    assert rec["source_detail"]["model"].startswith("DJI F450") and rec["tach_candidate"]
    src = _norm(
        "DroneAudioSet",
        "000001_source-only-recordings__speaker-loudness-60db__room1__speaker-dist-1m__"
        "mic-dist-25cm__mic1_soundskrit-silence.wav",
        {},
        {},
    )
    assert src["sound_content"] == "no_drone" and src["provenance_kind"] == "control"
    assert src["speed_regime"] == "not_applicable"
    with pytest.raises(ValueError):
        derived.droneaudioset_path_facts("drone-only-recordings/x.wav")


def test_se_valid_drone_category_uses_the_waveform_audit():
    assert (
        _norm("SE-valid-drone", "drone_snr-30_0000", {}, {"fields": ["mixture", "target"]})[
            "sound_content"
        ]
        == "no_drone"
    )
    rec = _norm("SE-valid-harmonic", "drone_snr-30_0002", {}, {"fields": ["mixture", "target"]})
    assert rec["sound_content"] == "digital_mix" and rec["speed_supplied"] == "no"
    assert rec["tach_role"] == rec["wind_role"] == "mixture"
    assert _norm("SE-valid-harmonic", "mimii_snr-10_0001", {}, {})["provenance_kind"] == "non_drone"


def test_kaist_half_second_windows_and_icsv_clips():
    key = "onboard/tethered/drone-A_motor_fault_artificial-mix/A/test/mic1/A_B_MF1_0_ConstructionSite_6_snr=10.9"
    rec = _norm("KAIST-drone-fault-frames", key, {"label": {"fault_code": "MF1"}}, {})
    assert rec["environment"] == "anechoic" and rec["sound_content"] == "digital_mix"
    assert rec["speed_regime"] == "unknown" and not rec["tach_candidate"]
    assert rec["tach_skip"]["status"] == "too_short" and rec["tach_skip"]["duration_s"] == 0.5
    negative = _norm("drone-detection-samples", "000000_no-drone-1", {}, {})
    assert "tach_skip" not in negative
    icsv = "onboard/tethered/drone-B_normal_artificial-mix/train/train_B_Front_normal_000001"
    assert _norm("ICSV31-drone-anomaly-frames", icsv, {}, {})["tach_candidate"]


@pytest.mark.parametrize(
    ("key", "expected"),
    [("n119", "constant"), ("n120", "variable"), ("n121", "constant"), ("n122", "variable")],
)
def test_survey_file_operating_regime(key, expected):
    split = "test" if key in {"n121", "n122"} else "train"
    rec = _norm("zenodo_drone_noises", f"raw/noises-{split}-drones/{key}", {}, {"has_audio": True})
    assert rec["speed_regime"] == expected
    assert rec["speed_supplied"] == "no"


def test_simulations():
    ego = {"prescribed_rps": 55.0, "n_blades": 2, "n_mics": 64, "rps_prescribed": True}
    rec = _norm("drone-egonoise", "dregon_R055p0_A10p0_D02p0_M064_s000", ego, {"fields": ["wav"]})
    assert rec["environment"] == "simulated" and rec["speed_regime"] == "constant"
    assert rec["nominal_rps"] == 55.0 and rec["provenance_kind"] == "simulation"
    jasa = _norm(
        "jasa-flyovers",
        "V01p0_A30p0_H000p0_Y+00p0_s000",
        {"bpf_hz": 33.55, "speed": 1.0},
        {"fields": ["wav"]},
    )
    assert jasa["blade_count"] == "3" and jasa["relative_motion"] == "moving"
    assert (
        jasa["speed_supplied"] == "no" and "nominal_rps" not in jasa and "supplied_kind" not in jasa
    )
    assert jasa["source_detail"]["hover_reference"]["bpf_hz"] == 33.55
