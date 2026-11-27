"""Condition evidence for the project's own rigs, AVQ, SPCUP19 and their raw/derived views.

Covers the Frames rigs (``DREGON-frames``, ``michaels-frames``, ``michaels-test-frames``,
``new_motor_data-frames``), their raw parents (``DREGON``, ``recording_with_motor_speed``,
``new-drone-noises``, ``new_motor_data``), ``AVQ`` + ``AVQ-raw`` / ``AVQ-egonoise`` /
``AVQ-egonoise-vkrps``, ``SPCUP19-frames`` + ``SPCUP19-egonoise`` and
``noise-v2-bench-points``. Facts come from the publishers' papers/pages, the in-repo
per-recording annotation tables (``sources/avq_meta.yaml``, ``sources/spcup19_meta``)
and the setup photos shipped inside the raw parents; ``meta``/``info`` only refine
(which telemetry series a Frame actually carries). Every raw or processed copy is
labelled by its own physical take and points at it through ``parent``; acoustic
estimates (``rps_refined``, blind-VK pseudo-labels, survey comb readings) never make
``speed_supplied`` yes. No audio or network I/O.
"""

from __future__ import annotations

import copy
import json
import re
from functools import cache
from pathlib import Path
from typing import Any

_DREGON = (
    "Strauss et al., IROS 2018 (doi:10.1109/IROS.2018.8593581) + dregon.inria.fr/datasets/dregon"
)
_DREGON_FIG1 = "DREGON paper Fig. 1 photo: two-bladed propellers on the MK-Quadro"
_MICH = (
    "recording_with_motor_speed photos 20220811_180536_HDR.jpg / Microphone_Array_Configuration.jpeg "
    "+ docs/data-catalog.md § Michael's DJI Matrice 100 rig"
)
_NOV = (
    "new_motor_data setup photos (10_Nov, 14_Nov) + DatCon export columns + docs/data-catalog.md "
    "§ November 2026 motor recordings + docs/experiments/rps-refine-precision.md § November M100"
)
_AVQ = "sources/avq_meta.yaml (publisher spec table, setupA/B.jpg, platform.jpg); Wang et al., IROS 2019"
_AVQ_PLATFORM = "AVQ demo page platform.jpg (webspace.eecs.qmul.ac.uk/lin.wang/demo/avq_files/platform.jpg): each of the four hubs carries a two-blade propeller"
_SPCUP = "sources/spcup19_meta/{team}.yaml (team report + supplementary files)"
_BENCH = "src/data_processing/noise_v2_bench_points.json + results/noise_v2/survey/findings.md"
_SINGLES = (
    "results/static_rig/single_rotor/dregon_singles.json @ 3edac1b9 (sha256 ff14a2b0eeaa…), "
    "docs/experiments/static-rig-profiles.md § Round 3"
)

#: Embedded copy of ``_SINGLES`` (the packaged runner snapshot excludes ``results/``):
#: per single-motor file, the zero-based mics with rotor-downwash wind (Dmitrii's
#: reading) and the strict rotor-on span in seconds on the DREGON-frames clock.
_DREGON_SINGLES: dict[str, dict[str, Any]] = {
    rid: {"wind_mics": mics, "span_s": span}
    for rid, mics, span in (
        ("motor_Motor1_50", [5, 6], [6.8, 36.2]),
        ("motor_Motor1_60", [5, 6], [5.2, 41.2]),
        ("motor_Motor1_70", [5, 6], [5.2, 34.8]),
        ("motor_Motor1_80", [5, 6], [5.0, 14.8]),
        ("motor_Motor1_90", [5, 6], [5.2, 16.0]),
        ("motor_Motor2_50", [0], [5.0, 14.8]),
        ("motor_Motor2_60", [0], [4.2, 14.8]),
        ("motor_Motor2_70", [0], [4.8, 15.0]),
        ("motor_Motor2_80", [0], [5.2, 15.8]),
        ("motor_Motor2_90", [0], [6.2, 16.0]),
        ("motor_Motor3_50", [1, 2], [4.2, 13.8]),
        ("motor_Motor3_60", [1, 2], [4.8, 15.2]),
        ("motor_Motor3_70", [1, 2], [5.8, 15.8]),
        ("motor_Motor3_80", [1, 2], [5.0, 15.5]),
        ("motor_Motor3_90", [1, 2], [5.8, 15.8]),
        ("motor_Motor4_50", [4], [5.0, 14.5]),
        ("motor_Motor4_60", [4], [4.0, 14.0]),
        ("motor_Motor4_70", [4], [4.8, 14.8]),
        ("motor_Motor4_80", [4], [6.2, 16.0]),
        ("motor_Motor4_90", [4], [6.2, 16.0]),
    )
}

#: SPCUP19-frames@a81ee75821c6 ChuMS audio length in samples at 44.1 kHz (= the valid
#: overlap of the two recorder groups after ``spcup19._align_recorders``), read from the
#: published Frame descriptors; the per-mic ``RawTruncatedCalibrated`` arrays of
#: SPCUP19-egonoise@043e0a4e8264 have exactly that Frame's ``meta.mic_n_samples``.
_CHUMS_FRAME_SAMPLES = {
    "1prop_repeat1": 3219301,
    "2prop_repeat1": 2646001,
    "3prop_repeat1": 2513701,
    "2prop_repeat2": 3045301,
    "3prop_repeat2": 3002611,
    "1prop_repeat2": 2551443,
    "1prop_repeat3": 2574675,
    "2prop_repeat3": 3225609,
    "3prop_repeat3": 3426138,
}
_CHUMS_SR = 44100


N_DREGON_ROTORS = 4

#: DREGON rooms (paper §II-A/B): in-flight source recordings in the larger room, and the
#: hovering/up-down/rectangle/spinning flights in the smaller one.
_DREGON_ROOMS = {
    "room1": "larger carpeted indoor room (12 x 12 x 3.5 m), RT < 150 ms, Vicon",
    "room2": "smaller carpeted indoor room (10 x 10 x 2.5 m), RT < 150 ms, Vicon",
}
_DREGON_FLIGHT = re.compile(
    r"^(free-flight|hovering|updown|rectangle|spinning|silent-flight)_"
    r"(nosource|speech-high|speech-low|whitenoise-high|whitenoise-low)_(room[12])$"
)
_DREGON_MOTOR = re.compile(r"^motor_(?:Motor([1-4])|allMotors)_(\d+)$")
_DREGON_CLEAN = re.compile(
    r"^clean_(speech|whitenoise|chirps)_(-?\d+)_(-?\d+)_([\d.]+)(?:__(\d+))?$"
)
_DREGON_TEST = frozenset(
    {
        "free-flight_speech-high_room1",
        "free-flight_whitenoise-high_room1",
        "silent-flight_whitenoise-low_room1",
    }
)

#: Michael's M100 (airframe mcID 041DE70827): raw tree file -> recording id.
_MICH_RAW = {"recording_1": "FLY124", "recording_2": "FLY125"}
_MICH_TEST = {"103_2": "FLY103", "108_2": "FLY108"}
#: November M100 (airframe mcID 041DE21193): raw session directory -> recording id.
_NOV_SESSIONS = {"10_Nov": "FLY013", "14_Nov": "FLY018", "15_Nov": "FLY022"}

#: ChuMS ``TestResults.Test(k)`` (1-based, report Table 3 + ``Details``) -> SPCUP19-frames key suffix.
_CHUMS_TESTS = (
    "1prop_repeat1",
    "2prop_repeat1",
    "3prop_repeat1",
    "2prop_repeat2",
    "3prop_repeat2",
    "1prop_repeat2",
    "1prop_repeat3",
    "2prop_repeat3",
    "3prop_repeat3",
)
_CHUMS_EGONOISE = re.compile(
    r"ChuMS__UAV_rotor_recordings_TestResults__Test_([0-8])___Data_([0-7])___"
    r"(Freq|SPL|RawTruncatedCalibrated)"
)
_CHUMS_SPECTRAL_PRODUCTS = {
    "Freq": ("frequency_axis", "frequency-bin coordinates"),
    "SPL": ("level_spectrum", "sound-pressure levels by frequency bin"),
}


def is_non_waveform(dataset: str, key: str) -> bool:
    """Recognize ChuMS MATLAB spectral products stored under the old ``audio`` field."""
    if dataset != "SPCUP19-egonoise":
        return False
    match = _CHUMS_EGONOISE.fullmatch(key)
    return match is not None and match[3] in _CHUMS_SPECTRAL_PRODUCTS


# ─── record helpers ───────────────────────────────────────────────────────────


def _set(record: dict, ref: str, **facets: tuple[str, str]) -> None:
    for facet, (value, reason) in facets.items():
        record[facet] = value
        record["evidence"][facet] = f"{ref}: {reason}"


def _new(kind: str = "recording") -> dict:
    return {"evidence": {}, "provenance_kind": kind, "blade_count_basis": "unknown"}


def _motion(
    record: dict, ref: str, value: str, reason: str, roles: dict[str, str], basis: str
) -> None:
    _set(record, ref, relative_motion=(value, reason))
    record["relative_motion_basis"] = basis
    record["roles"] = dict(roles)
    record["role_motion"] = {role: value for role in roles}


def _rotors(record: dict, ref: str, n_rotors: int | None, n_active: int, reason: str) -> None:
    record.update(n_rotors=n_rotors, n_active=n_active)
    if n_active == 0:
        activity = "off"
    elif n_active == 1:
        activity = "single"
    elif n_rotors is not None and n_active == n_rotors:
        activity = "all"
    else:
        activity = "subset"
    _set(record, ref, rotor_activity=(activity, reason))


def _off(record: dict, ref: str, reason: str) -> None:
    """Rotors not running: keep the rig/room facts the caller set, note the control."""
    _set(
        record,
        ref,
        speed_regime=("off", reason),
        propulsion=("none", f"{reason}; no propulsion running"),
        speed_supplied=("not_applicable", f"{reason}; no rotor speed to supply"),
        blade_count=("not_applicable", f"{reason}; no propeller noise source in this take"),
    )
    record["provenance_kind"] = "control"
    record["tach_candidate"] = False


def _view(base: dict, kind: str, parent: dict, note: str) -> dict:
    """A raw copy, log companion, cache or excerpt of one physical take.

    The physical-take facts stay; the copy is never a second independent recording.
    Non-audio companions get ``not_applicable`` sound/wind (``schema.normalize``
    then disables tachometry on them via ``has_audio``).
    """
    out = copy.deepcopy(base)
    out["provenance_kind"] = kind
    out["parent"] = parent
    out["quality"] = [*out.get("quality", []), note]
    if parent.get("binding") != "exact":
        out.pop(
            "valid_spans_s", None
        )  # spans are on the parent clock; the runner projects crops itself
    if kind == "log_only":
        for facet in ("sound_content", "wind"):
            out[facet] = "not_applicable"
            out["evidence"][facet] = f"{note}: no audio in this file"
        out.pop("roles", None)
        out.pop("role_motion", None)
        out["tach_candidate"] = False
    return out


def _asset(note: str, parent: dict | None = None) -> dict:
    """Documentation/geometry/marker files: no recording facts at all (all not_applicable)."""
    out = {
        "evidence": {},
        "provenance_kind": "log_only",
        "quality": [note],
        "tach_candidate": False,
    }
    if parent:
        out["parent"] = parent
    return out


def _parent(dataset: str, key: str | None, binding: str, **extra: Any) -> dict:
    """Runner link to the canonical parent sample (ConditionsJobs projection convention).

    ``binding``: ``exact`` = same audio on the same time origin (t=0 = the parent audio's
    first sample; sample rate/channels may differ), ``crop`` = ``offset_s`` (alias t=0 on the
    parent audio clock, relative to its first sample) + ``duration_s``, ``none`` = no
    sample correspondence (logs, spectra, unknown crops): nothing is projected.
    """
    out: dict[str, Any] = {"dataset": dataset, "binding": binding}
    if key is not None:
        out.update(key=key, role="audio")
    out.update({k: v for k, v in extra.items() if v is not None})
    return out


# ─── DREGON ───────────────────────────────────────────────────────────────────


def _dregon(rid: str, info: dict | None = None) -> dict:
    info = info or {}
    series = info.get("series") or {}
    rec = _new()
    if m := _DREGON_FLIGHT.match(rid):
        flight, source, room = m.groups()
        env = ("indoors", f"{room} = {_DREGON_ROOMS[room]} (paper §II-A)")
        roles = {"audio": "8-mic cube array mounted under the UAV body (paper §II-A, Fig. 1)"}
        if flight == "silent-flight":
            _set(
                rec,
                _DREGON,
                rig_mount=(
                    "handheld",
                    "motors off; a human operator carried the UAV around the room to simulate a flight (paper §II-B)",
                ),
                sound_content=(
                    "no_drone",
                    "white-noise loudspeaker only, motors off, plus 10 s of room silence (paper §II-B)",
                ),
                environment=env,
                wind=("unknown", "no rotor downwash; room ambient not assessed by the source"),
            )
            _rotors(
                rec,
                _DREGON,
                N_DREGON_ROTORS,
                0,
                "motors off for the whole recording (paper §II-B, site 'Silent Flight')",
            )
            _off(rec, _DREGON, "silent flight: UAV motors off")
            _motion(
                rec,
                _DREGON,
                "fixed",
                "rotor hubs and the array are rigidly mounted on the same carried airframe; rotors are off (control)",
                roles,
                "rigid_mount",
            )
            rec["held_out"] = "test"
            rec["quality"] = ["DREGON held-out TEST recording (sources.dregon.TEST_RECORDING_IDS)"]
            return rec
        _set(
            rec,
            _DREGON,
            rig_mount=(
                "free_flight",
                "tele-operated flights (take-off, flight figures, landing); paper §II-B/C",
            ),
            propulsion=(
                "propellers",
                "MK-Quadro with its four propellers in flight (paper Fig. 1)",
            ),
            blade_count=("2", _DREGON_FIG1),
            environment=env,
            wind=(
                "unknown",
                "paper Fig. 3a shows rotor wind-noise bursts in a free flight; not quantified per recording",
            ),
        )
        rec["blade_count_basis"] = "observed"
        _rotors(
            rec,
            _DREGON,
            N_DREGON_ROTORS,
            4,
            "all four MK2832-35 motors fly the quadrotor (paper §II-A)",
        )
        _motion(
            rec,
            _DREGON,
            "fixed",
            "array rigidly mounted under the UAV; hubs do not move relative to the mics in flight",
            roles,
            "rigid_mount",
        )
        if flight == "hovering":
            _set(
                rec,
                _DREGON,
                speed_regime=(
                    "unknown",
                    "hover flight incl. take-off ramp (~7 s) and landing; hover constancy not established by the source",
                ),
            )
        else:
            what = {
                "free-flight": "free flight combining take-off, stabilisation, hovering, lines, circles, spinning, landing",
                "updown": "up-and-down flight",
                "rectangle": "rectangle flight path",
                "spinning": "spinning (yaw) flight",
            }[flight]
            _set(
                rec,
                _DREGON,
                speed_regime=(
                    "variable",
                    f"{what}; take-off/landing ramps in every flight (paper §II-B/C, catalog in-flight window table)",
                ),
            )
        measured = "motors_measured" in series if series else room == "room1"
        if measured:
            rec["supplied_kind"] = "measured+command"
            _set(
                rec,
                _DREGON,
                speed_supplied=(
                    "yes",
                    "motors.mat 'measured' (feedback) and 'command' rotor speeds in turns/s, ~929 Hz; audio timestamps aligned to motor onsets, sync error < 100 ms",
                ),
            )
        else:
            rec["supplied_kind"] = "command"
            _set(
                rec,
                _DREGON,
                speed_supplied=(
                    "yes",
                    "motors.mat has only 'command' rotor speeds (turns/s, ~929 Hz) for this room-2 flight; Frames publish it cleaned as motors_command",
                ),
            )
        if source == "nosource":
            _set(
                rec,
                _DREGON,
                sound_content=("drone_only", "noise-only flight, no target source (paper §II-C)"),
            )
        else:
            kind, level = source.split("-")
            _set(
                rec,
                _DREGON,
                sound_content=(
                    "co_recorded",
                    f"Genelec loudspeaker on a central table emitting {'TIMIT speech' if kind == 'speech' else 'white noise'} at the {level} level during the flight (paper §II-B)",
                ),
            )
        rec["tach_candidate"] = True
        rec["quality"] = [
            "telemetry-backed flight: acoustic estimates are validated against the supplied track, not used as labels"
        ]
        if "rps_refined" in series or not series:
            rec["quality"].append(
                "rps_refined is a project acoustic refinement (F_VK/L-BFGS), not a supplied speed"
            )
        if rid in _DREGON_TEST:
            rec["held_out"] = "test"
            rec["quality"].append(
                "DREGON held-out TEST recording (sources.dregon.TEST_RECORDING_IDS)"
            )
        return rec
    if m := _DREGON_MOTOR.match(rid):
        motor, speed = m.groups()
        s = float(speed)
        _set(
            rec,
            _DREGON,
            rig_mount=(
                "tethered",
                "airframe SUSPENDED on a nylon string hanging freely in the room and held still by a human operator (handheld restraint, not a stand); paper §II-C, site 'Individual Motor Recordings'",
            ),
            speed_regime=(
                "constant",
                f"one commanded speed of {speed} turns/s per file (paper Table I); file includes motor-off lead/tail (11-35 s of motor in 25-45 s, survey findings step 1)",
            ),
            propulsion=(
                "propellers",
                "paper §I: 'recordings of the individual propellers at various speed'; Fig. 3b shows propeller harmonics",
            ),
            speed_supplied=(
                "yes",
                f"nominal motor speed {speed} turn/s from the file name (site: 'YY is the motor speed in turn/second'; paper Table I 'Commanded speed [turns/s]'), not a percent; no measured trace for bench files; acoustic check: law 0.975*S + 0.37 rev/s, 20/20 single-motor files, mean |err| 0.56 rev/s ({_BENCH})",
            ),
            sound_content=("drone_only", "pure motor/propeller noise, no source (paper §II-C)"),
            environment=(
                "indoors",
                "UAV hung in a room of the indoor recording site (paper §II-C); room not named",
            ),
            blade_count=("2", _DREGON_FIG1),
            wind=(
                "unknown",
                "publisher: holding the UAV still gives only a 'limited amount of in-flight wind noise' — not a per-channel or per-file no-wind statement; left to the waveform wind assessment",
            ),
        )
        rec["blade_count_basis"] = "observed"
        rec["supplied_kind"] = "nominal"
        _motion(
            rec,
            _DREGON,
            "fixed",
            "array rigidly mounted under the suspended, stationary UAV",
            {"audio": "8-mic cube array under the UAV"},
            "rigid_mount",
        )
        if motor is None:
            _rotors(
                rec,
                _DREGON,
                N_DREGON_ROTORS,
                4,
                f"allMotors_{speed}: all four motors at {speed} turns/s (paper Table I)",
            )
            rec["nominal_rps"] = [s] * N_DREGON_ROTORS
            rec["quality"] = [
                "historical whole-record acoustic fit (docs/experiments/static-rig-profiles.md § Local smoke on DREGON / "
                "round 1): three strong anonymous rotors ~67.68 / 68.75 / 69.56 rev/s; a fourth rotor is not resolved "
                "(the noise-v2 survey's 64.65 is weak and unconfirmed: lines at 2 of 40 orders, 0.77 dB margin) — no "
                f"fourth numeric label; publisher nominal {speed} turn/s, single-motor law {0.975 * s + 0.37:.1f}"
            ]
        else:
            _rotors(
                rec,
                _DREGON,
                N_DREGON_ROTORS,
                1,
                f"Motor{motor}_{speed}: only motor {motor} runs (paper Table I, file naming convention)",
            )
            rec["nominal_rps"] = s
            rec["active_rotor_ids"] = [f"Motor{motor}"]
            rec["quality"] = [f"single physical rotor (DREGON Motor {motor}); identity known"]
            single = _DREGON_SINGLES.get(rid)
            if single:
                # Independent reading of this file: downwash-exposed mics and the strict rotor-on span.
                rec["wind_exposed_channels"] = {"audio": list(single["wind_mics"])}
                rec["valid_spans_s"] = [list(single["span_s"])]
                rec["evidence"]["wind"] += (
                    f"; {_SINGLES} documents rotor-downwash wind on zero-based mics "
                    f"{single['wind_mics']} of this file (other mics not established as wind-free)"
                )
                rec["quality"].append(f"strict rotor-on span {single['span_s']} s ({_SINGLES})")
        return rec
    if m := _DREGON_CLEAN.match(rid):
        kind, az, el, dist, _ = m.groups()
        _set(
            rec,
            _DREGON,
            rig_mount=(
                "not_applicable",
                "no UAV: the 8-mic array alone fixed on a tripod, loudspeaker on a separate tripod (site 'Noise-Free Recordings')",
            ),
            speed_regime=("off", "no UAV present"),
            propulsion=("none", "no UAV present"),
            speed_supplied=("not_applicable", "no rotors"),
            sound_content=(
                "no_drone",
                f"tripod loudspeaker emitting {kind} at azimuth {az}, elevation {el}, {dist} m",
            ),
            environment=(
                "semi_anechoic",
                "third, semi-anechoic and acoustically isolated room (site + paper §II-D)",
            ),
            blade_count=("not_applicable", "no propeller"),
            relative_motion=("not_applicable", "no rotor hubs"),
            wind=("unknown", "not assessed by the source"),
        )
        _rotors(rec, _DREGON, None, 0, "noise-free source recording without the UAV")
        rec.update(provenance_kind="control", tach_candidate=False)
        return rec
    raise KeyError(f"DREGON recording id {rid!r} is not in the published inventory")


_DREGON_LOGS = {
    "audiots": "per-sample audio timestamps",
    "imu": "IMU log",
    "motors": "motor speed log",
    "sourcepos": "Vicon source-position log",
}


def _dregon_raw(key: str, info: dict) -> dict:
    parent = lambda rid, binding="exact": _parent("DREGON-frames", rid, binding)  # noqa: E731
    parts = key.split("/")
    name = parts[-1]
    if parts[0] == ".cache" and (m := re.match(r"^(.+)_16000hz_[0-9a-f]+$", name)):
        return _view(
            _dregon(m.group(1)),
            "derivative",
            parent(m.group(1), "none"),
            "16 kHz resampled loader cache (.npz) left in the raw tree; not an independent recording",
        )
    if parts[0] == "DREGON_individual_motors_recordings" and (
        m := re.match(r"^(Motor\d|allMotors)_(\d+)$", name)
    ):
        rid = f"motor_{name}"
        return _view(
            _dregon(rid),
            "alias",
            parent(rid),
            "original publisher WAV of this DREGON-frames recording",
        )
    if parts[0] == "clean_sources" and len(parts) == 4:
        rid = f"clean_{parts[1]}_{name}"
        return _view(
            _dregon(rid),
            "alias",
            parent(rid),
            "original publisher WAV of this DREGON-frames recording",
        )
    if (
        len(parts) == 2
        and (parts[0].startswith("DREGON_") or parts[0] == "nosource")
        and name.startswith("DREGON_")
    ):
        stem = name[len("DREGON_") :]
        log = next((s for s in _DREGON_LOGS if stem.endswith(f"_{s}")), None)
        rid = stem[: -len(log) - 1] if log else stem
        dup = " (duplicate copy under nosource/)" if parts[0] == "nosource" else ""
        if log:
            return _view(
                _dregon(rid),
                "log_only",
                parent(rid, "none"),
                f"DREGON {_DREGON_LOGS[log]} of this flight{dup}",
            )
        return _view(
            _dregon(rid),
            "alias",
            parent(rid),
            f"original publisher WAV of this DREGON-frames recording{dup}",
        )
    if parts[0] == "emitted_signals":
        sig = "TIMIT speech" if "TIMIT" in name else "white noise"
        out = {
            "evidence": {},
            "provenance_kind": "non_drone",
            "tach_candidate": False,
            "quality": [
                f"digital source signal ({sig}) played by the loudspeaker in the in-flight source recordings; not a recording"
            ],
        }
        _set(
            out,
            _DREGON,
            rig_mount=("not_applicable", "digital source file"),
            speed_regime=("not_applicable", "digital source file"),
            propulsion=("not_applicable", "digital source file"),
            rotor_activity=("not_applicable", "digital source file"),
            relative_motion=("not_applicable", "digital source file"),
            speed_supplied=("not_applicable", "digital source file"),
            sound_content=(
                "no_drone",
                f"emitted {sig} signal file (site 'Clean Emitted Source Signals')",
            ),
            environment=("not_applicable", "never acoustically recorded"),
            blade_count=("not_applicable", "digital source file"),
            wind=("not_applicable", "digital source file"),
        )
        return out
    if name == ".unzipped":
        return _asset("archive-extraction marker file")
    return _asset(f"DREGON geometry/documentation/workspace file {key!r}; no recording")


# ─── Michael's M100 and the November M100 ─────────────────────────────────────


def _michaels(rid: str, info: dict | None = None) -> dict:
    if rid not in ("FLY124", "FLY125", "FLY103", "FLY108"):
        raise KeyError(f"Michael's M100 recording {rid!r} is not in the published inventory")
    test = rid in ("FLY103", "FLY108")
    rec = _new()
    _set(
        rec,
        _MICH,
        rig_mount=(
            "free_flight",
            "DJI Matrice 100 flights: DatCon log shows GPS fix and relative height; audio spans ground idle, warm-up and cruise",
        ),
        speed_regime=(
            "variable",
            "telemetry calls the audio span ground idle / warm-up / cruise (FLY103 1.4/12.7/86.0 %, FLY108 6.8/9.6/83.6 %); cruise speeds follow attitude control",
        ),
        propulsion=("propellers", "M100 flown with its four propellers (photos)"),
        speed_supplied=(
            "yes",
            "DJI 'Motor:Speed:*' ESC-reported RPM per rotor (~29 Hz) aligned to the audio; published 'rps' carries the per-recording global scale "
            "(MICHAELS_RPS_SCALE, fitted against the VK residual: a label-for-this-audio correction), the raw 'motor_speed' block is uncalibrated",
        ),
        sound_content=("drone_only", "ego-noise recording; no deliberate source documented"),
        blade_count=(
            "2",
            "photo 20220811_180536_HDR / Microphone_Array_Configuration of airframe 041DE70827 shows two-blade propellers"
            + (
                "; FLY103/108 (2022-04-14) are the same airframe (DatCon mcID), photographed 4 months later"
                if test
                else ""
            ),
        ),
        wind=("unknown", "outdoor flights; wind not documented"),
    )
    rec["blade_count_basis"] = "observed"
    rec["supplied_kind"] = "measured"
    _rotors(rec, _MICH, 4, 4, "all four M100 rotors fly the aircraft; four Motor:Speed channels")
    if test:
        _set(
            rec,
            _MICH,
            environment=(
                "outdoors",
                "same airframe as the outdoor FLY124/125 field session (DatCon mcID 041DE70827); DatCon logs 14-17 GPS satellites",
            ),
        )
        _motion(
            rec,
            _MICH,
            "unknown",
            "single microphone (not the 8-mic ring); its placement is not documented",
            {"audio": "mono microphone, placement undocumented"},
            "unknown",
        )
        rec["held_out"] = "test"
        rec["quality"] = [
            "held-out TEST recording: no training/validation derivation may root on it; anchored clock (1.19 % dilation)"
        ]
    else:
        _set(
            rec,
            _MICH,
            environment=(
                "outdoors",
                "photos: grass field, M100 on a landing mat; GPS-logged flight",
            ),
        )
        _motion(
            rec,
            _MICH,
            "fixed",
            "8-mic ring rigidly mounted on a boom 20 cm from / 33 cm above the body (Microphone_Array_Configuration.jpeg)",
            {"audio": "8-mic ring (16.5 cm) on a boom above the airframe"},
            "rigid_mount",
        )
        rec["quality"] = ["rps_refined is a project acoustic refinement, not supplied"]
    rec["tach_candidate"] = True
    rec["quality"].append(
        "telemetry-backed: raw Motor:Speed values are independent of the audio; the published rps scale and clock are acoustically fitted"
    )
    return rec


def _november(rid: str, info: dict | None = None) -> dict:
    if rid not in ("FLY013", "FLY018", "FLY022"):
        raise KeyError(f"November M100 recording {rid!r} is not in the published inventory")
    rec = _new()
    common = dict(
        speed_supplied=(
            "yes",
            "native DAT Motor packets (RPM/60 = rev/s, ~60 ms) of all four rotors; values unchanged, packet clock mapped to the WAV by an acoustically fitted affine map (A, f)",
        ),
        sound_content=("drone_only", "ego-noise of the M100 only; no deliberate source documented"),
        wind=("unknown", "not documented"),
    )
    _set(rec, _NOV, **common)
    rec["supplied_kind"] = "measured"
    _rotors(
        rec,
        _NOV,
        4,
        4,
        "DatCon Motor:Speed shows all four motors running together (FLY013 94.6 %, FLY018 96.2 %, FLY022 77.6 % of rows; a single-motor subset in <= 0.1 %)",
    )
    if rid == "FLY013":
        _set(
            rec,
            _NOV,
            rig_mount=(
                "grounded",
                "setup photo Image_20261110233345: airframe resting on the floor; DatCon relative height ~0 m, 0 GPS satellites",
            ),
            speed_regime=(
                "mixed",
                "bench runs at several speeds (DatCon 1282-8003 RPM 5-95 %) with stops; steady runs used as fit windows (drone-noise-generator-paper.md § November)",
            ),
            propulsion=("motor_only", "setup photo: bare motor bells, no propellers mounted"),
            environment=("indoors", "setup photo: carpeted floor; no GPS reception"),
            blade_count=("0", "setup photo: bare shafts (no propeller)"),
        )
        rec["blade_count_basis"] = "observed"
        rec["shaft_orders"] = [84, 98]
        rec["quality"] = [
            "bare-motor bench: shaft orders 84 and 98 (rps-refine-precision.md § November M100), not blade-pass lines"
        ]
        _motion(
            rec,
            _NOV,
            "fixed",
            "ZOOM H1n strapped to the top plate of the airframe (photo)",
            {"audio": "ZOOM H1n stereo recorder on the airframe top plate"},
            "rigid_mount",
        )
    else:
        height = "1.1-3.8 m, 15-19 satellites" if rid == "FLY018" else "1.0-3.4 m, 14-15 satellites"
        _set(
            rec,
            _NOV,
            rig_mount=(
                "free_flight",
                f"DatCon IMU relative height {height}; airborne cruise excerpts used in the generator fits",
            ),
            speed_regime=(
                "variable",
                "flight under attitude control incl. ground phases (DatCon: motors stopped in part of FLY022)",
            ),
            propulsion=(
                "propellers",
                "propeller recording (rps-refine-precision.md § November M100: even orders 2-14)",
            ),
            environment=("outdoors", f"GPS fix with {height.split(', ')[1]}; airborne"),
        )
        if rid == "FLY018":
            _set(
                rec,
                _NOV,
                blade_count=(
                    "2",
                    "setup photo Image_20261114163751: two-blade propellers on all four motors",
                ),
            )
            rec["blade_count_basis"] = "observed"
            _motion(
                rec,
                _NOV,
                "fixed",
                "ZOOM H1n strapped to the top plate (photo)",
                {"audio": "ZOOM H1n stereo recorder on the airframe top plate"},
                "rigid_mount",
            )
        else:
            _set(
                rec,
                _NOV,
                blade_count=(
                    "2",
                    "no FLY022 photo; same airframe (DatCon mcID 041DE21193) photographed with two-blade props the day before, and even shaft orders 2-14 carry the comb",
                ),
            )
            rec["blade_count_basis"] = "inferred_same_airframe"
            _motion(
                rec,
                _NOV,
                "unknown",
                "no setup photo for this session; recorder placement not documented",
                {"audio": "ZOOM H1n stereo recorder, placement undocumented"},
                "unknown",
            )
        rec["quality"] = []
    rec["tach_candidate"] = True
    rec["quality"].append(
        "telemetry-backed: measured RPM values are independent of the audio; only the clock map is acoustic"
    )
    return rec


def _raw_mich(key: str, info: dict) -> dict:
    d, _, name = key.rpartition("/")
    if (rid := _MICH_RAW.get(d)) and name in (rid[3:], rid):
        if name == rid:
            return _view(
                _michaels(rid),
                "log_only",
                _parent("michaels-frames", rid, "none"),
                "DatCon CSV flight log (uncalibrated Motor:Speed RPM)",
            )
        return _view(
            _michaels(rid),
            "alias",
            _parent("michaels-frames", rid, "none"),
            "original WAV; michaels-frames crops its head to the flight-log clock (crop length not recorded statically)",
        )
    note = {
        "Microphone_Array_Configuration": "rig photo with mic-array measurements",
        "md2_config": "downscaled rig photo with mic-array measurements",
        "20220811_180536_HDR": "rig photo (outdoor field)",
        "md2_image": "downscaled rig photo",
    }.get(key)
    return _asset(
        note or f"raw-tree file {key!r} (no recording)", _parent("michaels-frames", None, "none")
    )


def _raw_mich_test(key: str, info: dict) -> dict:
    if rid := _MICH_TEST.get(key):
        return _view(
            _michaels(rid),
            "alias",
            _parent("michaels-test-frames", rid, "exact"),
            "original mono WAV (held-out TEST); the anchored Frame keeps the audio uncropped",
        )
    if key in ("FLY103", "FLY108"):
        return _view(
            _michaels(key),
            "log_only",
            _parent("michaels-test-frames", key, "none"),
            "DatCon CSV flight log (held-out TEST)",
        )
    return _asset(
        f"raw-tree file {key!r} (CSV column readme)", _parent("michaels-test-frames", None, "none")
    )


def _raw_nov(key: str, info: dict) -> dict:
    session, _, name = key.partition("/")
    rid = _NOV_SESSIONS.get(session)
    if rid is None:
        raise KeyError(f"new_motor_data key {key!r} outside the three sessions")
    if name == rid:
        return _view(
            _november(rid),
            "log_only",
            _parent("new_motor_data-frames", rid, "none"),
            "native DAT flight log + DatCon CSV export (no audio)",
        )
    if name.startswith("ZOOM"):
        return _view(
            _november(rid),
            "alias",
            _parent("new_motor_data-frames", rid, "exact"),
            "original ZOOM WAV; new_motor_data-frames carries it unchanged",
        )
    return _asset(f"session setup photo {name}", _parent("new_motor_data-frames", rid, "none"))


# ─── AVQ ──────────────────────────────────────────────────────────────────────


@cache
def _avq_table() -> dict[str, dict]:
    from data_processing.sources.avq import load_annotations

    return load_annotations()["by_key"]


def _avq(seq: str, info: dict | None = None) -> dict:
    ann = _avq_table().get(seq)
    if ann is None:
        raise KeyError(f"AVQ sequence {seq!r} is not in avq_meta.yaml")
    rec = _new()
    roles = {"audio": "onboard 8-mic ring (~20 cm) ~15 cm above the rotor arms (platform.jpg)"}
    _set(
        rec,
        _AVQ,
        rig_mount=(
            "fixed_stand",
            "quadcopter on a camera tripod on grass in every sequence, never airborne (setupA/B.jpg)",
        ),
        environment=("outdoors", "paper abstract 'recorded outdoors'; setup photos on grass"),
        wind=("unknown", "outdoor recording; wind not documented"),
    )
    _motion(
        rec,
        _AVQ,
        "fixed",
        "array and rotors on the same tripod-mounted platform",
        roles,
        "rigid_mount",
    )
    spec = f"spec row {ann['spec_row']}"
    if ann["condition"] == "rotors_off":
        _rotors(rec, _AVQ, 4, 0, f"{spec}: Drone 'muted'")
        _set(
            rec,
            _AVQ,
            sound_content=(
                "no_drone",
                f"{spec}: speech only ({ann['external_source_motion']} talker/loudspeaker), motors muted",
            ),
        )
        _off(rec, _AVQ, f"{spec}: motors muted")
        rec["quality"] = ["same tripod platform and array as the ego-noise sequences (control)"]
        return rec
    _rotors(rec, _AVQ, 4, 4, f"{spec}: Drone constant/dynamic (all motors)")
    _set(
        rec,
        _AVQ,
        propulsion=("propellers", "platform.jpg shows four propellers"),
        blade_count=("2", _AVQ_PLATFORM),
        speed_supplied=(
            "no",
            f"{spec}: only a throttle setting ({ann['throttle_or_speed']}); no rotor speed",
        ),
    )
    rec["blade_count_basis"] = "observed"
    if ann["condition"] == "bench":
        _set(
            rec,
            _AVQ,
            speed_regime=(
                "constant",
                f"{spec}: Drone 'constant ({ann['throttle_or_speed'].split()[1]})'",
            ),
        )
        rec["throttle_percent"] = float(ann["throttle_or_speed"].split()[1])
    else:
        _set(
            rec,
            _AVQ,
            speed_regime=("variable", f"{spec}: Drone 'dynamic' (time-varying motor speed)"),
        )
    if ann["content"] == "ego_noise_only":
        _set(rec, _AVQ, sound_content=("drone_only", f"{spec}: Type EO (ego-noise only)"))
    else:
        _set(
            rec,
            _AVQ,
            sound_content=(
                "co_recorded",
                f"{spec}: Type MIX, {ann['external_source_motion']} loudspeaker speech in the {ann['external_source_area']} area",
            ),
        )
    if ann["throttle_or_speed"].startswith("constant 150"):
        rec["quality"] = ["throttle '150 %' is printed by the publisher; its meaning is not stated"]
    return rec


#: AVQ-raw GoPro mp4 audio tracks, measured on uni-cpu (jobs legacy-avq-mp4-probe-1f78f4: MP4 box
#: tree; legacy-avq-gopro-compare-28beed: ffmpeg 6.1.1 decode vs the AVQ@a37840cace1e Frame).
#: Every file has an AAC 48 kHz track (S1 stereo 128 kb/s, S2 mono ~69 kb/s) in a container
#: re-muxed by ffmpeg (encoder Lavf57.83.100, generic Video/SoundHandler, not GoPro-native).
#: Against the onboard array the track keeps a constant lag (change <= 1.5 ms over 134-706 s)
#: and correlates at |rho| 0.61-0.86 with the 8-channel array mean (inverted polarity). That is
#: compatible with a probable time-aligned derivative of the same event (e.g. a downmix of the
#: array), but it does not prove identical PCM or the same microphone: a separate mic on a shared
#: or resampled clock could correlate too. Exact channel weights/identity are unverified.
#: Value: (lag_s, duration_s, channels, lag_note); lag = camera time - array time of the same
#: sound (two windows agreeing unless noted).
_AVQ_GOPRO_AUDIO = {
    "S1__seq4__Video": (-0.0027, 796.160, 2, "two windows: -0.0034 / -0.0019 s"),
    "S2__seq3__video": (-0.4266, 215.211, 1, "two windows: -0.4266 / -0.4265 s"),
    "S2__seq4__video": (-0.4231, 217.216, 1, "two windows: -0.4231 / -0.4231 s"),
    "S2__seq5__video": (-0.4206, 303.403, 1, "WEAK: early window found no peak; late window only"),
    "S2__seq6__video": (-0.4211, 270.720, 1, "two windows: -0.4212 / -0.4210 s"),
    "S2__seq7__video": (1.0831, 256.064, 1, "two windows: +1.0830 / +1.0832 s"),
    "S2__seq8__video": (0.6521, 247.936, 1, "two windows: +0.6521 / +0.6521 s"),
}


def _avq_raw(key: str, info: dict) -> dict:
    parts = key.split("__")
    if len(parts) == 3 and parts[1].startswith("seq"):
        seq = f"{parts[0]}_{parts[1]}"
        if key in _AVQ_GOPRO_AUDIO:
            lag, dur, nch, lag_note = _AVQ_GOPRO_AUDIO[key]
            out = _view(
                _avq(seq),
                "derivative",
                _parent(
                    "AVQ",
                    seq,
                    "crop",
                    offset_s=-lag,
                    duration_s=dur,
                    field="mp4",
                    project_wind=False,
                    lag_note=lag_note,
                    # A single-window lag is not a verified correspondence: the runner measures directly.
                    verify="lag" if lag_note.startswith("WEAK") else None,
                ),
                f"GoPro video (DOA ground truth) with a {nch}-ch AAC 48 kHz audio track: compatible/probable "
                f"time-aligned derivative of the same event as the onboard array ({lag:+.4f} s, {lag_note}); "
                "exact channel weights/identity unverified (_AVQ_GOPRO_AUDIO)",
            )
            _set(
                out,
                _AVQ,
                wind=(
                    "unknown",
                    "own AAC mix of unverified channel weights; array wind is not transferred",
                ),
            )
            out["roles"] = {
                "mp4": "AAC audio track of the publisher's re-muxed GoPro video (probable derivative of the onboard array, unverified mix)"
            }
            out["role_motion"] = {"mp4": out["relative_motion"]}
            return out
        return _view(
            _avq(seq),
            "log_only",
            _parent("AVQ", seq, "none"),
            "DOA/VAD ground truth (angle_vad.mat)",
        )
    return _asset(
        f"AVQ documentation/calibration/geometry file {key!r}", _parent("AVQ", None, "none")
    )


def _avq_derived(dataset: str, key: str, meta: dict) -> dict:
    seq = re.sub(r"_seg\d+$", "", key)
    base = _avq(seq)
    built = "AVQ@50dd53d1a6c0 (recipe 1; same audio as recipe 2)"
    if dataset == "AVQ-egonoise":
        parent = _parent("AVQ", seq, "exact", channels=[0], built_from=built)
        note = "channel 0 only, 16 kHz mono copy of the whole sequence; its stale 'speech + rotor ego-noise' meta is ignored"
    else:
        start = meta.get("segment_start_s")
        if start is None:
            parent = _parent("AVQ", seq, "none", built_from=built)
        else:
            parent = _parent(
                "AVQ",
                seq,
                "crop",
                offset_s=float(start),
                duration_s=meta.get("duration_s"),
                channels=[0],
                built_from=built,
            )
        note = (
            "blind-VK accepted segment (16 kHz mono ch0); its 'rps' is a pseudo-label "
            "(scripts/vk_pseudolabel.py @ fa5053fc), not a supplied speed"
        )
    out = _view(base, "derivative", parent, note)
    out["roles"] = {"audio": "channel 0 of the onboard 8-mic ring"}
    out["role_motion"] = {"audio": out["relative_motion"]}
    return out


# ─── SPCUP19 ──────────────────────────────────────────────────────────────────


@cache
def _spcup_table() -> dict[str, tuple[dict, dict]]:
    from data_processing.sources.spcup19 import load_annotations

    return {r["key"]: (doc, r) for doc in load_annotations().values() for r in doc["recordings"]}


#: Team-level facts read from each team YAML (summary/drone/geometry sources).
_TEAM = {
    "AGH": dict(
        environment=("anechoic", "summary: anechoic-chamber recordings"),
        mount=("fixed_stand", "summary: small quadcopter fixed on a tall stand; nothing was flown"),
        blade=(
            "3",
            "observed",
            "Fig. 2 photo shows 3-blade propellers; single-rotor blade-pass = 3x shaft rate",
        ),
        motion=(
            "fixed",
            "array mounted below the drone on the stand, reference mic on its own stand; drone static",
            "rigid_mount",
        ),
    ),
    "ChuMS": dict(
        environment=("outdoors", "summary: hover tests in a park"),
        mount=(
            "free_flight",
            "summary/open question: flying UAV under altitude hold at ~2.5 m (not a propeller test rig)",
        ),
        blade=None,
        motion=("fixed", "boom + mast array rigidly attached to the chassis", "rigid_mount"),
    ),
    "Diagonal_Unloading": dict(
        environment=("outdoors", "summary: quiet field outdoors"),
        mount=("free_flight", "summary: hovering / yaw / rectangle flights"),
        blade=(
            "2",
            "spec",
            "DJI Phantom 4 PRO manufacturer spec: 9455S two-blade propellers (installation not photographed per take)",
        ),
        motion=(
            "fixed",
            "16-MEMS Octopus array fixed under the drone on a composite plate",
            "rigid_mount",
        ),
    ),
    "Idea_ssu": dict(
        environment=("indoors", "summary: large low-reflection hall, indoors"),
        mount=("free_flight", "summary: hovering, yaw spinning, up/down and free flight"),
        blade=("2", "spec", "DJI Phantom 4 manufacturer spec: 9450S two-blade propellers"),
        motion=(
            "fixed",
            "mic taped to the landing stand, hanging on its cable (small swing possible)",
            "rigid_mount",
        ),
    ),
    "KU_Leuven": dict(
        environment=("indoors", "summary: medium-sized room, indoors"),
        mount=(
            "handheld",
            "summary: drone static, 'held approximately half a metre above the floor'",
        ),
        blade=None,
        motion=("fixed", "8-mic cube hung 19 cm below the motor plane", "rigid_mount"),
    ),
    "KumamoTech": dict(
        environment=("outdoors", "summary: almost flat outdoor field"),
        mount=("free_flight", "summary: climb / hover / excursion and rectangular circuits"),
        blade=None,
        motion=("fixed", "16-mic octagon frame carried by the quadcopter", "rigid_mount"),
    ),
    "LEADS_UAV": dict(
        environment=("outdoors", "summary: open lawn outdoors"),
        mount=("free_flight", "summary: hovering DJI Phantom 3 Advanced"),
        blade=("2", "spec", "DJI Phantom 3 Advanced manufacturer spec: 9450 two-blade propellers"),
        motion=("fixed", "mic on a rigid rod 0.12 m below the landing bracket", "rigid_mount"),
    ),
    "Maverick": dict(
        environment=("anechoic", "summary: inside an anechoic chamber"),
        mount=None,
        blade=(
            "2",
            "observed",
            "Team Maverick acquisition report p.4 Fig. 2, top view: two folding blades on each of the four hubs",
        ),
        motion=("fixed", "three lavalier mics taped under the body", "rigid_mount"),
    ),
    "NSS_Chellamma": dict(
        environment=("outdoors", "summary: outdoor recording over barren land"),
        mount=("free_flight", "summary: flown while the pilot kept it stable"),
        blade=(
            "3",
            "spec",
            "report names Dalprop Cyclone T3056C; product specification 'DAL T3056C Cyclone Tri-Blade 3\" Props' (https://www.unmannedtechshop.co.uk/products/dal-cyclone-t-3056-c-propellers)",
        ),
        motion=(
            "fixed",
            "single microphone 'embedded on the UAV' (exact position unknown)",
            "rigid_mount",
        ),
    ),
    "Shout_COOEE": dict(
        environment=("indoors", "summary: curtained lab, indoors"),
        mount=(
            "free_flight",
            "summary: manual take-off, ~70 s hover, landing (armed on a cushion)",
        ),
        blade=None,
        motion=("fixed", "8-mic array on a 3D-printed holder under the belly", "rigid_mount"),
    ),
}

#: Per-recording speed regime: (value, reason). Keys not listed follow the rules in ``_spcup``.
_SPCUP_REGIME = {
    "AGH__static_corrupted__0": (
        "mixed",
        "tonal lines drift during the file and rise sharply in the last ~1.5 s",
    ),
    "AGH__static_corrupted__1": ("constant", "tonal lines ~stationary ('medium speed')"),
    "KU_Leuven__SPCUP19_KU_Leuven_Team_1_recording": (
        "mixed",
        "spin-up, steady plateau, spin-down",
    ),
    "KU_Leuven__SPCUP19_KU_Leuven_Team_2_recording": (
        "mixed",
        "spin-up, steady plateau, spin-down",
    ),
    "Maverick__4": (
        "constant",
        "all motors at constant speed on the ground (report Fig. 5; two slightly different rotor speeds)",
    ),
    "Shout_COOEE__SPCUP19_Shout_COOEE_StaticSubmission1": (
        "mixed",
        "steady ground idle (comb 33.9 Hz), take-off, ~70 s hover, landing",
    ),
    "Shout_COOEE__SPCUP19_Shout_COOEE_StaticSubmission2": (
        "mixed",
        "steady ground idle (comb 33.9 Hz), take-off, hover, landing, idle",
    ),
    "NSS_Chellamma__task_audio": (
        "constant",
        "report p.1: the acquisition aimed to 'keep the drone stable and angular speed fixed'; "
        "no throttle setting or rotor speed supplied, and observed tonal drift does not establish constant actual RPM",
    ),
    "NSS_Chellamma__task_audio_edited": (
        "constant",
        "Audacity edit of task_audio, inheriting its fixed angular-speed target (report p.1), "
        "not a measured RPM or throttle setting",
    ),
}
_SPCUP_MOUNT = {
    "Maverick__1": ("handheld", "running drone carried by hand around the chamber"),
    "Maverick__5": ("handheld", "running drone carried along a circular path by hand"),
    "Maverick__4": ("grounded", "motors on, drone stays on the ground (did not take off)"),
    "Maverick__2": ("free_flight", "tele-operated free flight"),
    "Maverick__3": ("free_flight", "tele-operated free flight"),
    "Maverick__6": ("free_flight", "tele-operated free flight"),
    "KumamoTech__data2_rotors_off": ("grounded", "after landing, rotors stopped"),
}


def _spcup(key: str, info: dict | None = None) -> dict:
    found = _spcup_table().get(key)
    if found is None:
        raise KeyError(f"SPCUP19 recording {key!r} is not in spcup19_meta")
    doc, r = found
    team = doc["team"]
    ref = _SPCUP.format(team=team)
    t = _TEAM[team]
    rec = _new()
    desc = " ".join(str(r.get("description", "")).split())
    mount = _SPCUP_MOUNT.get(key) or t["mount"]
    _set(
        rec,
        ref,
        rig_mount=mount,
        environment=t["environment"],
        wind=("unknown", "not assessed per recording by the team report"),
    )
    value, reason, basis = t["motion"]
    roles = {"audio": reason}
    if team == "AGH" and r.get("geometry") is None:
        roles = {"audio": "AKG C451 B reference mic on its own stand beside the fixed drone"}
    _motion(rec, ref, value, reason, roles, basis)
    n_rotors = int(doc["drone"]["n_rotors"])
    n_active = int(r["n_active_rotors"])
    ext = r["external_source"]
    _rotors(
        rec,
        ref,
        n_rotors,
        n_active,
        f"n_active_rotors={n_active} (team YAML; {str(r.get('source', ''))[:160]})",
    )
    if n_active == 1 and (named := re.search(r"'(rotor \d)' in the team's numbering", desc)):
        rec["active_rotor_ids"] = [named.group(1)]  # team label; physical location undocumented
    if team == "NSS_Chellamma":
        rec["quality"] = [
            "rotor count not stated by the team; 4 inferred in the YAML from the 218 mm frame"
        ]
    if r["condition"] == "rotors_off":
        _set(
            rec, ref, sound_content=("no_drone", f"rotors off; external source {ext}: {desc[:160]}")
        )
        _off(rec, ref, "rotors off (team YAML condition rotors_off)")
        return rec
    _set(
        rec,
        ref,
        propulsion=("propellers", "propeller-driven rotors"),
        speed_supplied=(
            "no",
            f"no supplied rotor speed; team value: {r.get('throttle_or_speed')!r} (setting, qualitative or acoustic reading)",
        ),
    )
    if team == "ChuMS":
        n_prop = int(key.rsplit("__", 1)[1][0])
        _set(
            rec,
            ref,
            blade_count=(
                str(2 * n_prop),
                f"{n_prop} two-blade T-Motor P15x5 propeller(s) stacked per motor = {2 * n_prop} blades per rotor assembly (report Sec. 3.2, Table 3); BPF = {2 * n_prop} x shaft",
            ),
        )
        rec["blade_count_basis"] = "observed"
        rec["propellers_per_rotor"] = n_prop
        rec["blades_per_propeller"] = 2
        rec.setdefault("quality", []).append(
            "corpus-survey 'propeller rig / n_rotors 3' reading of ChuMS is wrong: flying quad, all 4 rotors, stacked props"
        )
    elif t["blade"]:
        b, basis_b, why = t["blade"]
        _set(rec, ref, blade_count=(b, why))
        rec["blade_count_basis"] = basis_b
    else:
        _set(rec, ref, blade_count=("unknown", "propeller not documented by the team"))
    if key in _SPCUP_REGIME:
        _set(rec, ref, speed_regime=_SPCUP_REGIME[key])
    elif r["condition"] == "bench":
        _set(
            rec,
            ref,
            speed_regime=(
                "constant",
                f"fixed setting {r.get('throttle_or_speed')!r}; {desc[:120]}",
            ),
        )
    elif r["condition"] == "handheld":
        _set(rec, ref, speed_regime=("variable", f"carried by hand; {r.get('throttle_or_speed')}"))
    elif r.get("flight_mode") == "hover":
        _set(
            rec,
            ref,
            speed_regime=(
                "unknown",
                "hover; constancy not established (a hover is not a governed plateau)",
            ),
        )
        rec["tach_candidate"] = True
    elif r.get("flight_mode") == "unknown":
        _set(rec, ref, speed_regime=("unknown", f"unlabelled flight; {r.get('throttle_or_speed')}"))
        rec["tach_candidate"] = True
    else:
        _set(
            rec, ref, speed_regime=("variable", f"flight mode {r.get('flight_mode')}: {desc[:140]}")
        )
    if rec["speed_regime"] == "mixed" or key.startswith("NSS_Chellamma__task_audio"):
        rec["tach_candidate"] = True
    if ext == "none":
        _set(rec, ref, sound_content=("drone_only", "no external sound source (team YAML)"))
    else:
        _set(
            rec,
            ref,
            sound_content=(
                "co_recorded",
                f"external source {ext} played/spoken during the take: {desc[:160]}",
            ),
        )
    if key == "NSS_Chellamma__task_audio_edited":
        rec["provenance_kind"] = "derivative"
        rec["parent"] = _parent(
            "SPCUP19-frames", "NSS_Chellamma__task_audio", "crop", offset_s=3.88, duration_s=160.28
        )
        rec.setdefault("quality", []).append(
            "Audacity-processed, trimmed copy of task_audio (<100 Hz altered), not a separate take"
        )
    return rec


def _spcup_egonoise(key: str, info: dict) -> dict:
    if m := _CHUMS_EGONOISE.fullmatch(key):
        test, mic, what = int(m.group(1)), int(m.group(2)), m.group(3)
        rid = f"ChuMS__UAV_rotor_recordings__{_CHUMS_TESTS[test]}"
        if product := _CHUMS_SPECTRAL_PRODUCTS.get(what):
            representation, description = product
            note = (
                f"ChuMS Test {test + 1}, mic {mic + 1}, MATLAB {what}: {description}, "
                "stored under the old audio field but not a time-domain waveform"
            )
            out = _view(
                _spcup(rid),
                "derivative",
                _parent("SPCUP19-frames", rid, "none", channel=mic),
                note,
            )
            _set(
                out,
                _SPCUP.format(team="ChuMS"),
                speed_regime=("not_applicable", note),
                sound_content=("not_applicable", note),
                wind=("not_applicable", note),
            )
            out.pop("roles", None)
            out.pop("role_motion", None)
            out.update(representation=representation, tach_candidate=False)
            return out
        # The egonoise arrays are the untouched TestResults.Test(t).Data(m).RawTruncatedCalibrated
        # (lengths = the Frame's meta.mic_n_samples). SPCUP19-frames drops the first |K| samples of
        # the LATE recorder group (group_b if K > 0, else group_a) and cuts all mics to the
        # common length n (spcup19._align_recorders). So for a late-group mic, alias sample |K| is
        # Frame sample 0; for an early-group mic the two start together. Either way the Frame's
        # n samples are the only overlap; the alias tail beyond it has no Frame counterpart.
        align = _spcup_table()[rid][1]["recorder_align"]
        k = int(align["k_samples"])
        late = align["group_b"] if k > 0 else align["group_a"]
        lead = abs(k) if mic in late else 0
        n = _CHUMS_FRAME_SAMPLES[_CHUMS_TESTS[test]]
        group = "late" if lead else "early"
        return _view(
            _spcup(rid),
            "derivative",
            _parent(
                "SPCUP19-frames",
                rid,
                "crop",
                offset_s=-lead / _CHUMS_SR if lead else 0.0,
                duration_s=(lead + n) / _CHUMS_SR,
                channels=[mic],
                alias_span_s=[lead / _CHUMS_SR, (lead + n) / _CHUMS_SR],
            ),
            f"one channel (mic {mic + 1}, {group} recorder group) of ChuMS Test {test + 1}, "
            f"recorder offset K={k} not undone here; Frame overlap = alias "
            f"[{lead / _CHUMS_SR:.4f}, {(lead + n) / _CHUMS_SR:.4f}) s",
        )
    return _view(
        _spcup(key),
        "alias",
        _parent("SPCUP19-frames", key, "exact"),
        "older per-file pin of the same team audio (SPCUP19-frames is the annotated recipe)",
    )


# ─── noise-v2 bench points ───────────────────────────────────────────────────


@cache
def _bench_points() -> dict[str, dict]:
    path = Path(__file__).resolve().parents[1] / "noise_v2_bench_points.json"
    return {p["key"]: p for p in json.loads(path.read_text())["points"]}


def _bench(key: str, info: dict) -> dict:
    p = _bench_points().get(key)
    if p is None:
        raise KeyError(f"noise-v2-bench-points key {key!r} is not in the committed manifest")
    start, dur = p["publish_start_s"], p["publish_s"]
    corpus, sid = p["corpus"], p["source_id"]
    if corpus == "DREGON-bench":
        base, parent = (
            _dregon(sid),
            _parent("DREGON-frames", sid, "crop", offset_s=start, duration_s=dur),
        )
    elif corpus == "AVQ":
        base, parent = (
            _avq(sid),
            _parent(
                "AVQ",
                sid,
                "crop",
                offset_s=start,
                duration_s=dur,
                built_from="AVQ@50dd53d1a6c0 (recipe 1; same audio)",
            ),
        )
    elif corpus == "SPCUP19-egonoise":
        base, parent = (
            _spcup(sid),
            _parent(
                "SPCUP19-frames",
                sid,
                "crop",
                offset_s=start,
                duration_s=dur,
                built_from="SPCUP19-egonoise@043e0a4e8264 (same team audio)",
            ),
        )
    elif corpus == "SPCUP19-ChuMS-bench":
        rid = f"ChuMS__UAV_rotor_recordings__{sid}"
        base, parent = (
            _spcup(rid),
            _parent(
                "SPCUP19-frames",
                rid,
                "none",
                start_s=start,
                duration_s=dur,
                built_from="publisher ChuMS archive (recorder alignment not established)",
            ),
        )
    elif corpus == "DroneAudioSet":
        from . import derived

        base = derived.droneaudioset_path_facts(sid)
        parent = _parent(
            "DroneAudioSet",
            None,
            "none",
            file_path=sid,
            start_s=start,
            duration_s=dur,
            built_from="publisher drone-only parquet, not read from the dload pin",
        )
    else:
        raise KeyError(f"noise-v2-bench-points {key!r}: unknown corpus {corpus!r}")
    note = (
        f"<= {p['publish_s']:.0f} s stationary excerpt chosen by the survey; meta.speed_rev_s is a "
        "Welch harmonic-sum estimate (scripts/noise_v2_bench_speed.py), not a supplied speed"
    )
    out = _view(base, "derivative", parent, note)
    if "speed_supplied" in out["evidence"]:
        out["evidence"]["speed_supplied"] += (
            f" [excerpt: survey speed_rev_s is an acoustic estimate, {_BENCH}]"
        )
    if out.get("rotor_activity") == "off":
        out["quality"].append(
            f"source take has rotors OFF ({sid}); the survey's speed reading "
            f"{p['speed_rev_s']} is spurious and must not be used"
        )
        out["tach_candidate"] = False
    return out


# ─── dispatch ────────────────────────────────────────────────────────────────


def _frames(fn):
    return lambda key, meta, info: fn(key, info)


_CLASSIFIERS = {
    "DREGON-frames": _frames(_dregon),
    "DREGON": lambda key, meta, info: _dregon_raw(key, info),
    "michaels-frames": _frames(_michaels),
    "michaels-test-frames": _frames(_michaels),
    "recording_with_motor_speed": lambda key, meta, info: _raw_mich(key, info),
    "new-drone-noises": lambda key, meta, info: _raw_mich_test(key, info),
    "new_motor_data-frames": _frames(_november),
    "new_motor_data": lambda key, meta, info: _raw_nov(key, info),
    "AVQ": _frames(_avq),
    "AVQ-raw": lambda key, meta, info: _avq_raw(key, info),
    "AVQ-egonoise": lambda key, meta, info: _avq_derived("AVQ-egonoise", key, meta),
    "AVQ-egonoise-vkrps": lambda key, meta, info: _avq_derived("AVQ-egonoise-vkrps", key, meta),
    "SPCUP19-frames": _frames(_spcup),
    "SPCUP19-egonoise": lambda key, meta, info: _spcup_egonoise(key, info),
    "noise-v2-bench-points": lambda key, meta, info: _bench(key, info),
}

DATASETS = frozenset(_CLASSIFIERS)


def classify(dataset: str, key: str, meta: dict, info: dict) -> dict:
    """Classify one exact sample of a legacy rig / AVQ / SPCUP19 dataset (fresh dict)."""
    return _CLASSIFIERS[dataset](key, meta or {}, info or {})


def parent_facts(dataset: str, recording_id: str) -> dict:
    """Facts of a DREGON-frames / michaels(-test)-frames / new_motor_data-frames / AVQ /
    SPCUP19-frames recording for derived datasets that inherit them (fresh dict)."""
    return classify(dataset, recording_id, {}, {})
