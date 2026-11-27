"""Legacy rig / AVQ / SPCUP19 condition rules: exact published keys, no audio, no network."""

from __future__ import annotations

import pytest

from data_processing.recording_annotations import legacy
from data_processing.recording_annotations.schema import FACETS, normalize

FRAMES = {"has_audio": True, "fields": ["_frame", "audio"]}


def _facts(
    dataset: str, key: str, *, meta: dict | None = None, has_audio: bool = True, series=()
) -> dict:
    info = {"has_audio": has_audio, "series": {s: {} for s in series}}
    return normalize(legacy.classify(dataset, key, meta or {}, info), has_audio=has_audio)


def test_scope():
    scope = set(legacy.DATASETS)
    assert scope == {
        "DREGON-frames",
        "DREGON",
        "michaels-frames",
        "recording_with_motor_speed",
        "michaels-test-frames",
        "new-drone-noises",
        "new_motor_data-frames",
        "new_motor_data",
        "AVQ",
        "AVQ-raw",
        "AVQ-egonoise",
        "AVQ-egonoise-vkrps",
        "SPCUP19-frames",
        "SPCUP19-egonoise",
        "noise-v2-bench-points",
    }


def test_dregon_flights_and_supplied_speed_kind():
    room1 = _facts(
        "DREGON-frames",
        "free-flight_speech-low_room1",
        series=("audio", "motors_measured", "motors_command"),
    )
    assert (
        room1["rig_mount"],
        room1["sound_content"],
        room1["speed_supplied"],
        room1["supplied_kind"],
    ) == ("free_flight", "co_recorded", "yes", "measured+command")
    room2 = _facts(
        "DREGON-frames",
        "hovering_nosource_room2",
        series=("audio", "motors_command", "rps_refined"),
    )
    assert (
        room2["supplied_kind"] == "command"
        and room2["speed_regime"] == "unknown"
        and room2["tach_candidate"]
    )
    assert room2["blade_count"] == "2" and room2["relative_motion"] == "fixed"
    silent = _facts("DREGON-frames", "silent-flight_whitenoise-low_room1")
    assert (
        silent["rig_mount"],
        silent["rotor_activity"],
        silent["sound_content"],
        silent["n_active"],
    ) == ("handheld", "off", "no_drone", 0)
    assert silent["held_out"] == "test" and not silent["tach_candidate"]


def test_dregon_motor_bench_is_tethered_nominal_constant():
    one = _facts("DREGON-frames", "motor_Motor3_80")
    assert (one["rig_mount"], one["speed_regime"], one["rotor_activity"], one["propulsion"]) == (
        "tethered",
        "constant",
        "single",
        "propellers",
    )
    assert (
        one["nominal_rps"] == 80.0
        and one["active_rotor_ids"] == ["Motor3"]
        and one["supplied_kind"] == "nominal"
    )
    assert one["wind"] == "unknown" and one["tach_candidate"]
    assert (
        one["wind_exposed_channels"] == {"audio": [1, 2]}
        and one["valid_spans_s"][0][0] < one["valid_spans_s"][0][1]
    )
    crop = _facts("noise-v2-bench-points", "DREGON-bench__motor_Motor3_80")
    assert crop["wind"] == "unknown" and "valid_spans_s" not in crop
    quad = _facts("DREGON-frames", "motor_allMotors_70")
    assert (
        quad["rotor_activity"] == "all"
        and quad["n_active"] == 4
        and quad["nominal_rps"] == [70.0] * 4
    )
    clean = _facts("DREGON-frames", "clean_speech_45_0_1.2__3")
    assert (
        clean["environment"],
        clean["sound_content"],
        clean["rig_mount"],
        clean["provenance_kind"],
    ) == ("semi_anechoic", "no_drone", "not_applicable", "control")


def test_dregon_raw_tree_links_to_frames():
    wav = _facts("DREGON", "nosource/DREGON_updown_nosource_room2")
    assert wav["provenance_kind"] == "alias" and wav["parent"]["key"] == "updown_nosource_room2"
    assert wav["parent"]["binding"] == "exact"
    log = _facts(
        "DREGON",
        "DREGON_free-flight_nosource_room1/DREGON_free-flight_nosource_room1_motors",
        has_audio=False,
    )
    assert (
        log["provenance_kind"] == "log_only"
        and log["speed_supplied"] == "yes"
        and not log["tach_candidate"]
    )
    assert log["sound_content"] == "not_applicable" and log["parent"]["binding"] == "none"
    motor = _facts("DREGON", "DREGON_individual_motors_recordings/Motor1_50")
    assert motor["parent"]["key"] == "motor_Motor1_50" and motor["rotor_activity"] == "single"
    cache = _facts("DREGON", ".cache/hovering_nosource_room2_16000hz_6b02ce1c2d18", has_audio=False)
    assert cache["provenance_kind"] == "derivative"
    assert _facts("DREGON", "micPos", has_audio=False)["rig_mount"] == "not_applicable"
    assert _facts("DREGON", "emitted_signals/2min_TIMIT")["provenance_kind"] == "non_drone"


def test_michaels_and_november():
    fly = _facts("michaels-frames", "FLY124")
    assert (
        fly["environment"],
        fly["relative_motion"],
        fly["supplied_kind"],
        fly["blade_count"],
    ) == ("outdoors", "fixed", "measured", "2")
    test = _facts("michaels-test-frames", "FLY108")
    assert test["relative_motion"] == "unknown" and test["held_out"] == "test"
    alias = _facts("new-drone-noises", "108_2")
    assert alias["parent"] == {
        "dataset": "michaels-test-frames",
        "binding": "exact",
        "key": "FLY108",
        "role": "audio",
    }
    bare = _facts("new_motor_data-frames", "FLY013")
    assert (bare["propulsion"], bare["blade_count"], bare["rig_mount"], bare["shaft_orders"]) == (
        "motor_only",
        "0",
        "grounded",
        [84, 98],
    )
    flight = _facts("new_motor_data-frames", "FLY018")
    assert (flight["rig_mount"], flight["environment"], flight["blade_count"]) == (
        "free_flight",
        "outdoors",
        "2",
    )
    assert _facts("new_motor_data", "14_Nov/ZOOM0003")["parent"]["key"] == "FLY018"
    assert (
        _facts("new_motor_data", "10_Nov/FLY013", has_audio=False)["provenance_kind"] == "log_only"
    )


def test_avq_sequences_and_views():
    steady = _facts("AVQ", "S1_seq2")
    assert (
        steady["rig_mount"],
        steady["speed_regime"],
        steady["speed_supplied"],
        steady["blade_count"],
    ) == ("fixed_stand", "constant", "no", "2")
    assert steady["throttle_percent"] == 100.0 and steady["tach_candidate"]
    muted = _facts("AVQ", "S2_seq3")
    assert (muted["rotor_activity"], muted["sound_content"], muted["relative_motion"]) == (
        "off",
        "no_drone",
        "fixed",
    )
    assert _facts("AVQ", "S2_seq7")["sound_content"] == "co_recorded"
    ego = _facts("AVQ-egonoise", "S1_seq1")
    assert ego["sound_content"] == "drone_only" and ego["parent"]["channels"] == [0]
    seg = _facts(
        "AVQ-egonoise-vkrps", "S2_seq2_seg01", meta={"segment_start_s": 160.0, "duration_s": 20.0}
    )
    assert seg["parent"]["binding"] == "crop" and seg["parent"]["offset_s"] == 160.0
    assert seg["speed_supplied"] == "no" and seg["speed_regime"] == "variable"
    gopro = _facts("AVQ-raw", "S2__seq7__video", has_audio=False)
    assert gopro["provenance_kind"] == "derivative" and gopro["sound_content"] == "co_recorded"
    assert gopro["parent"]["binding"] == "crop" and gopro["parent"]["offset_s"] == -1.0831
    assert (
        _facts("AVQ-raw", "S2__seq7__angle_vad", has_audio=False)["provenance_kind"] == "log_only"
    )
    assert all(
        k in legacy._AVQ_GOPRO_AUDIO
        for k in ("S1__seq4__Video", "S2__seq3__video", "S2__seq8__video")
    )


def test_spcup_team_rules():
    chums = _facts("SPCUP19-frames", "ChuMS__UAV_rotor_recordings__2prop_repeat1")
    assert (
        chums["rig_mount"],
        chums["blade_count"],
        chums["n_active"],
        chums["propellers_per_rotor"],
    ) == ("free_flight", "4", 4, 2)
    agh = _facts("SPCUP19-frames", "AGH__ego-noise__single_rotors__0")
    assert agh["active_rotor_ids"] == ["rotor 4"]
    assert _facts("SPCUP19-frames", "AGH__ego-noise__single_rotors__7")["active_rotor_ids"] == [
        "rotor 1"
    ]
    assert (agh["rotor_activity"], agh["speed_regime"], agh["environment"], agh["blade_count"]) == (
        "single",
        "constant",
        "anechoic",
        "3",
    )
    off = _facts("SPCUP19-frames", "AGH__static_clean__8")
    assert (
        off["rotor_activity"] == "off"
        and off["sound_content"] == "no_drone"
        and off["rig_mount"] == "fixed_stand"
    )
    assert (
        _facts("SPCUP19-frames", "KU_Leuven__SPCUP19_KU_Leuven_Team_1_recording")["rig_mount"]
        == "handheld"
    )
    assert _facts("SPCUP19-frames", "Maverick__4")["rig_mount"] == "grounded"
    assert _facts("SPCUP19-frames", "LEADS_UAV__data__3")["sound_content"] == "co_recorded"
    edited = _facts("SPCUP19-frames", "NSS_Chellamma__task_audio_edited")
    assert edited["provenance_kind"] == "derivative" and edited["parent"]["offset_s"] == 3.88
    # Test 6 = 1prop_repeat2, K = -315058 < 0: group_a (mics 0,1,3,4) is the late recorder group whose
    # first |K| samples SPCUP19-frames drops; Frame length n = 2551443 (= 2866501 - 315058).
    early = _facts(
        "SPCUP19-egonoise",
        "ChuMS__UAV_rotor_recordings_TestResults__Test_5___Data_6___RawTruncatedCalibrated",
    )
    assert early["parent"]["key"].endswith("1prop_repeat2") and early["blade_count"] == "2"
    assert early["parent"]["offset_s"] == 0.0 and early["parent"]["duration_s"] == pytest.approx(
        2551443 / 44100
    )
    late = _facts(
        "SPCUP19-egonoise",
        "ChuMS__UAV_rotor_recordings_TestResults__Test_5___Data_0___RawTruncatedCalibrated",
    )
    assert late["parent"]["offset_s"] == pytest.approx(-315058 / 44100)
    assert late["parent"]["duration_s"] == pytest.approx((315058 + 2551443) / 44100)
    spl = _facts(
        "SPCUP19-egonoise", "ChuMS__UAV_rotor_recordings_TestResults__Test_0___Data_0___SPL"
    )
    assert spl["parent"]["binding"] == "none" and not spl["tach_candidate"]
    assert _facts("SPCUP19-egonoise", "Idea_ssu__stationary_1")["provenance_kind"] == "alias"


def test_bench_points_inherit_parents():
    dregon = _facts("noise-v2-bench-points", "DREGON-bench__motor_Motor1_50")
    assert dregon["parent"]["dataset"] == "DREGON-frames" and dregon["speed_supplied"] == "yes"
    chums = _facts("noise-v2-bench-points", "SPCUP19-ChuMS-bench__3prop_repeat2")
    assert (
        chums["rig_mount"] == "free_flight"
        and chums["n_active"] == 4
        and chums["blade_count"] == "6"
    )
    bogus = _facts("noise-v2-bench-points", "SPCUP19-egonoise__AGH__static_clean__1")
    assert bogus["rotor_activity"] == "off" and not bogus["tach_candidate"]
    assert any("spurious" in q for q in bogus["quality"])


@pytest.mark.parametrize(
    "dataset,key",
    [
        ("DREGON-frames", "motor_Motor9_50"),
        ("SPCUP19-frames", "nope"),
        ("noise-v2-bench-points", "nope"),
    ],
)
def test_unknown_keys_raise(dataset, key):
    with pytest.raises(KeyError):
        legacy.classify(dataset, key, {}, {})


def test_parent_facts_are_fresh():
    a = legacy.parent_facts("DREGON-frames", "free-flight_nosource_room1")
    a["evidence"]["rig_mount"] = "mutated"
    b = legacy.parent_facts("DREGON-frames", "free-flight_nosource_room1")
    assert b["evidence"]["rig_mount"] != "mutated" and set(FACETS) <= set(b)
