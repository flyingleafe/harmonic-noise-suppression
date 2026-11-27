"""Condition evidence for the native acoustic Frames. No audio or network I/O."""

from __future__ import annotations

from urllib.parse import unquote

_D3SET = "arXiv:2602.09991 §III-A and pinned IntRoLab D3SET README"
_AIRA = "AIRA-UAS RecordingInfo.xlsx and doi:10.3390/s19183902 §§2.1.1–2.2/Table 1"
_ANU = "Zenodo 16196947 filenames and ANU DroneAcousticDataset Readme.pdf"
_AUDROK = "Mobilithek 605778370199691264 and doi:10.61782/fa.2023.0049 §3.1"
_QMUL_BIRD = "Zenodo 8403622 and doi:10.1111/2041-210X.14234 §4.2"
_QMUL_INSITU = "Zenodo 19925797 and doi:10.1111/2041-210X.70324"
_CHIRO = "Dryad cf472n3, doi:10.1111/2041-210X.12992 and doi:10.1038/s41598-018-26122-z Methods"
_SDU = "Zenodo 7329733 and doi:10.1002/ece3.9577 §§2.1–2.5"
_HEAR = "DataverseNL doi:10.34894/ARULTD v1 and Hear-and-avoid §2.1.2"
_KU = "KU Leuven RDR doi:10.48804/PZAVUC v1.1 and arXiv:2211.02690"
_IDMT = "Zenodo 17525811 description and native flight names"
_SALFORD = "Figshare doi:10.17866/rd.salford.22133411.v3 and doi:10.61782/fa.2023.0249 Table 1"
_UAVIR = "Zenodo 15391924 v1.0 label.json and doi:10.3390/app15105378"


def _set(record: dict, ref: str, **facets: tuple[str, str]) -> None:
    for facet, (value, reason) in facets.items():
        record[facet] = value
        record["evidence"][facet] = f"{ref}: {reason}"


def _record(ref: str, speed_note: str, **facets: tuple[str, str]) -> dict:
    result: dict = {"evidence": {}, "provenance_kind": "recording", "tach_candidate": False}
    _set(result, ref, speed_supplied=("no", speed_note), **facets)
    return result


def _off(record: dict, ref: str, reason: str) -> None:
    _set(
        record,
        ref,
        speed_regime=("off", reason),
        propulsion=("none", reason),
        rotor_activity=("off", reason),
        sound_content=("no_drone", reason),
        blade_count=("not_applicable", "no propeller noise source in this take"),
        relative_motion=("not_applicable", "no active rotor hubs to compare with the mic"),
    )
    record.update(n_active=0, tach_candidate=False, provenance_kind="control")


def _rotors(
    record: dict,
    ref: str,
    count: int | None,
    reason: str,
    *,
    blade: str = "unknown",
    blade_reason: str | None = None,
    blade_basis: str = "unknown",
) -> None:
    if count is not None:
        record.update(n_rotors=count, n_active=count)
        _set(record, ref, rotor_activity=("single" if count == 1 else "all", reason))
    if blade_reason:
        _set(record, ref, blade_count=(blade, blade_reason))
        record["blade_count_basis"] = blade_basis


def _motion(
    record: dict,
    ref: str,
    motion: str,
    reason: str,
    roles: dict[str, tuple[str, str]],
    *,
    basis: str | None = None,
) -> None:
    """Role keys are exact audio entry paths, not captions or channel groups."""
    _set(record, ref, relative_motion=(motion, reason))
    if basis is not None:
        record["relative_motion_basis"] = basis
    record["roles"] = {name: description for name, (description, _) in roles.items()}
    record["role_motion"] = {name: value for name, (_, value) in roles.items()}


def _sub(meta: dict, key: str) -> str:
    # The published browse key starts with three URL-escaped labels. The
    # underlying source builder key has neither those labels nor escape codes.
    if sub := meta.get("label", {}).get("subcategory"):
        return sub
    parts = key.split("/", 3)
    return unquote(parts[2]) if len(parts) == 4 else ""


def _d3set(key: str, meta: dict, info: dict) -> dict:
    op = meta.get("operating", {})
    sub = _sub(meta, key)
    rid = meta.get("recording_id", key)
    streams = meta.get("audio_streams", {})
    available = streams or info.get("series", {}) or info.get("fields", [])
    track = meta.get("telemetry_tracks", {})
    result = _record(
        _D3SET,
        "no per-motor BPF payload on this Frame",
        environment=("outdoors", "delivery/distance trials at an outdoor field site"),
    )
    roles = {}
    if "audio" in available:
        roles["audio"] = ("onboard drone.wav microphone mounted to the airframe", "fixed")
    if "array_audio" in available:
        array_name = "external ground array (matrix.wav; some released channels omitted)"
        array_motion = "unknown" if sub in ("delivery_excerpt", "snr_test_excerpt") else "mixed"
        roles["array_audio"] = (array_name, array_motion)
    if op.get("condition") == "rotors_off" or rid == "Flights/SNR test/background noise":
        _off(
            result,
            _D3SET,
            "array background excerpt; the original actuator values are disarmed (1000 us)",
        )
        if roles:
            result.update(
                roles={name: detail for name, (detail, _) in roles.items()},
                role_motion={name: "not_applicable" for name in roles},
            )
        if "array_audio" in roles:
            result["wind_role"] = "array_audio"
    else:
        _set(
            result,
            _D3SET,
            rig_mount=(
                "free_flight",
                "unrestrained X500 flights; their ground/air phases are not a stand test",
            ),
            propulsion=(
                "propellers",
                "flight with four propellers shown on the modified X500 V2 in Fig. 4",
            ),
        )
        _rotors(
            result,
            _D3SET,
            4,
            "four motors on the X500 V2, active during powered flight",
            blade="2",
            blade_reason="Fig. 4 shows two-blade props; the stock X500 V2 kit specifies 1045",
            blade_basis="observed",
        )
        if roles:
            mode = "mixed" if len(roles) == 2 else next(iter(roles.values()))[1]
            bases = {
                name: "rigid_mount"
                if name == "audio"
                else "mixed_flight_phases"
                if motion == "mixed"
                else "unknown"
                for name, (_, motion) in roles.items()
            }
            _motion(
                result,
                _D3SET,
                mode,
                "drone.wav moves with the airframe; the ground array sees idle/hover holds and translations "
                "across full sorties, including movements between sites on the distance trials. "
                "An excerpt does not establish its own position hold.",
                roles,
                basis="mixed_roles" if len(roles) == 2 else next(iter(bases.values())),
            )
            result["role_motion_basis"] = bases
            tach = "audio" if "audio" in roles else "array_audio"
            result.update(tach_role=tach, wind_role=tach)
            # Publisher alignment pads the external PCM at both ends. Do not
            # interpret those padded spans as quiet audio or zero-speed data.
            stream = streams.get(tach, {})
            if (sr := stream.get("sample_rate")) and (n := stream.get("num_samples")):
                first = stream.get("leading_digital_zero_samples", 0)
                last = n - stream.get("trailing_digital_zero_samples", 0)
                if 0 <= first < last <= n and (first or last < n):
                    result["valid_spans_s"] = [[first / sr, last / sr]]
        if sub == "distance_test_hover":
            _set(
                result,
                _D3SET,
                speed_regime=("unknown", "hover at long range; not a governed shaft-speed plateau"),
            )
        elif sub in ("delivery_excerpt", "snr_test_excerpt"):
            _set(
                result,
                _D3SET,
                speed_regime=("unknown", "excerpt does not isolate a measured shaft-speed plateau"),
            )
            if sub == "snr_test_excerpt" and rid == "Flights/SNR test/drone":
                _set(
                    result,
                    _D3SET,
                    sound_content=(
                        "drone_only",
                        "published drone source for SNR tests, not a digital mixture",
                    ),
                )
        elif op.get("flight_mode") == "mixed":
            _set(
                result,
                _D3SET,
                speed_regime=(
                    "mixed",
                    "whole sortie has powered flight, transitions and idle; some steady subspans may exist",
                ),
            )
            if "array_audio" in roles or meta.get("array_sync"):
                _set(
                    result,
                    _D3SET,
                    sound_content=(
                        "co_recorded",
                        "dual-recorder flights include synchronization claps; augmentations were not deposited in these files",
                    ),
                )
        result["tach_candidate"] = True
    if "bpf_label" in track:
        d = track["bpf_label"]
        where = (
            "telemetry.bpf_label" if d.get("on_audio_clock") else "telemetry_unaligned.bpf_label"
        )
        _set(
            result,
            _D3SET,
            speed_supplied=(
                "yes",
                f"{where}: BPF Hz / 2 gives shaft rev/s. Optical-tach PWM calibration averaged across four motors, then each flight manually adjusted to audible peaks; not independently measured aloft",
            ),
        )
        result["supplied_kind"] = "bpf"
        entry = where.replace(".", "/")
        if f"{entry}/values" in info.get("series", {}):
            entry += "/values"
        speed_source = {
            "entry": entry,
            "hz_per_rev_s": 2,
            "columns": d.get("columns"),
            "independent_tach_validation": False,
            "hub_assignment": "unpublished",
        }
    else:
        speed_note = (
            "actuator output PWM is not speed"
            if "motor_pwm" in track
            else "no BPF label in this Frame"
        )
        _set(
            result,
            _D3SET,
            speed_supplied=(
                "no",
                speed_note + "; CRNN bpf_estimate_* outputs are not supplied shaft values",
            ),
        )
        speed_source = None
    result["source_detail"] = {
        "rig": "Holybro X500 V2 with package-drop servo",
        "motor": "2216 KV920 (X500 V2 stock kit specification)",
        "propeller": "two-blade; stock X500 V2 kit 10 × 4.5 in (1045), installed SKU not independently logged",
        "flight_mode": op.get("flight_mode"),
        "group": meta.get("group"),
        "split": meta.get("label", {}).get("split"),
        "speed_reference": speed_source,
    }
    if derivative := meta.get("derivative_of"):
        result["parent"] = {
            "dataset": "D3SET-frames",
            "recording_id": derivative["parent_id"],
            "start_s": derivative.get("parent_start_s"),
            "end_s": derivative.get("parent_end_s"),
        }
        result["provenance_kind"] = "derivative"
    elif rid != key and key.endswith("/matrix.wav") and meta.get("array_sync"):
        result["source_detail"]["related_recording_id"] = rid
        result["source_detail"]["clock_warning"] = (
            "separate array Frame; not aligned to onboard audio or motor logs"
        )
    return result


def _aira(key: str, meta: dict, info: dict) -> dict:
    op, label = meta.get("operating", {}), meta.get("label", {})
    sub = _sub(meta, key)
    second_only = sub == "second_drone_only"
    both = sub == "flight_hover_with_second_drone"
    second = label.get("second_drone")
    second_blades = {"3DR Solo": "2", "Parrot Bebop 2": "3"}.get(second, "unknown")
    result = _record(
        _AIRA,
        "no shaft telemetry, only translational velocity/height protocol labels",
        environment=("outdoors", "outdoor recordings, onboard eight-channel array"),
    )
    if sub == "background":
        _off(
            result,
            _AIRA,
            "Recordings 1–3: parked Matrice 100, motors not activated and no second drone",
        )
        _set(
            result, _AIRA, rig_mount=("grounded", "the microphone carrier is parked on the ground")
        )
        result["n_rotors"] = 4
        _motion(
            result,
            _AIRA,
            "fixed",
            "onboard array has fixed geometry to the parked M100",
            {"audio": ("onboard array on the parked M100", "fixed")},
            basis="rigid_mount",
        )
        return result
    if sub == "ground_motors_on":
        mount = "grounded"
        mount_note = (
            "Recording 4: Matrice 100 on the ground with motors activated; no stand documented"
        )
    else:
        mount = "free_flight"
        mount_note = "the active source is airborne; for interference-only controls the M100 recorder is parked"
    _set(
        result,
        _AIRA,
        rig_mount=(mount, mount_note),
        propulsion=(
            "propellers",
            "the propeller-equipped aircraft are powered; no bare-motor procedure",
        ),
        speed_regime=(
            "unknown",
            "specified displacement, hover or grounded activation does not specify a shaft-speed profile",
        ),
        sound_content=(
            "co_recorded" if both else "drone_only",
            "M100 and another drone (interferer) fly together"
            if both
            else "one drone is running; no other experimental source, incidental ambient not ruled out",
        ),
    )
    if meta.get("recording_number") == 5:
        _set(
            result,
            "doi:10.3390/s19183902 §3.3/Fig. 7a",
            speed_regime=(
                "variable",
                "Recording 5 includes take-off and thrust change; acoustic tones change during the departure",
            ),
        )
    if second_only or both:
        second_mode = label.get("second_drone_motion")
        motion, basis, reason = {
            "hover": (
                "fixed",
                "nominal_hover",
                "second drone holds the position specified in the protocol; not a rigidly coupled source",
            ),
            "circle": (
                "moving",
                "flight_protocol",
                "second drone makes a full circle relative to the parked microphone carrier",
            ),
        }.get(
            second_mode,
            (
                "unknown",
                "unknown",
                "relative trajectory of the additional drone is not established",
            ),
        )
        if both:
            reason += "; M100's relative geometry to its array is rigid"
    else:
        motion, basis, reason = (
            "fixed",
            "rigid_mount",
            "M100 array is mounted to the sound source airframe",
        )
    _motion(
        result,
        _AIRA,
        motion,
        reason,
        {"audio": ("8-channel M100 onboard array", motion)},
        basis=basis,
    )
    if second_only or both:
        blade = second_blades
        if both and second_blades == "3":
            blade = "mixed"
        reason = f"actual second drone specified in protocol: {second}; 3DR 1045 is two-blade, Bebop 5040 is three-blade"
        if both:
            reason += "; M100 uses two-blade propellers"
        _rotors(
            result,
            _AIRA,
            8 if both else 4,
            "each of the active aircraft is a quadrotor",
            blade=blade,
            blade_reason=reason,
            blade_basis="reported" if blade != "unknown" else "unknown",
        )
    else:
        _rotors(
            result,
            _AIRA,
            4,
            "M100 is a quadrotor",
            blade="2",
            blade_reason="§2.1.1 reports DJI 1345 two-blade propellers for the recorded M100",
            blade_basis="reported",
        )
    result.update(
        tach_candidate=True,
        source_detail={
            "rig": "DJI Matrice 100",
            "motor": "DJI 3510 (DJI M100 support, Propulsion System)",
            "propeller": "DJI 1345, two blades (acquisition paper §2.1.1)",
            "mounted_rotors": 4,
            "mounted_active_rotors": op.get("n_active_rotors"),
            "second_drone": second,
            "second_propeller": label.get("second_drone_propeller"),
            "second_drone_n_rotors": label.get("second_drone_n_rotors"),
            "second_drone_motion": label.get("second_drone_motion"),
            "second_drone_height_m": label.get("second_drone_height_m"),
            "second_drone_horizontal_distance_m": label.get("second_drone_horizontal_distance_m"),
            "flight_mode": op.get("flight_mode"),
            "translation_m_s": [op.get("horizontal_velocity_mps"), op.get("vertical_velocity_mps")],
            "height_range_m": [op.get("height_min_m"), op.get("height_max_m")],
            "protocol": label.get("protocol"),
            "annotation_note": label.get("annotation_note"),
        },
    )
    if second_only:
        # Useful interference-only controls, not quiet/background negatives.
        result["provenance_kind"] = "control"
    if meta.get("channels_complete") is False and meta.get("channel_frames"):
        lengths = [int(n) for n in meta["channel_frames"].split(";")]
        longest = max(lengths)
        result["tach_channels"] = [i for i, n in enumerate(lengths) if n == longest]
    return result


def _anu(key: str, meta: dict, info: dict) -> dict:
    op = meta.get("operating", {})
    sub = _sub(meta, key)
    result = _record(
        _ANU,
        "11 declared motor current settings, not RPM and no shaft trace",
        environment=(
            "indoors",
            "bench acquisition; the exact room treatment for this deposit is not specified",
        ),
        rig_mount=("fixed_stand", "ANU stand with body-mounted array"),
    )
    result["n_rotors"] = 4
    if sub == "room-noise_control":
        _off(result, _ANU, "Room_noise.mat motor-off control")
        _motion(
            result,
            _ANU,
            "fixed",
            "onboard microphones on the stand-fixed body, motors off",
            {"audio": ("body-mounted array room-noise control", "fixed")},
            basis="rigid_mount",
        )
        return result
    current = op.get("motor_current_mA")
    _set(
        result,
        _ANU,
        speed_regime=(
            "unknown",
            f"constant-current bench run ({current} mA), not an observed shaft-speed plateau",
        ),
        propulsion=(
            "unknown",
            "current-specific deposit does not state the installed load; the earlier related 150 mm propeller protocol is not proof",
        ),
        rotor_activity=(
            "unknown",
            "powered rotor count not documented; the previous single-motor experiment is not this deposit",
        ),
        blade_count=(
            "unknown",
            "related 2020 rig had two blades; its identity with this session is not established",
        ),
        sound_content=("drone_only", "motor-current noise files, no intentional signal source"),
    )
    _motion(
        result,
        _ANU,
        "fixed",
        "onboard array and stand-fixed body",
        {"audio": ("32 input channels; 7 and 8 unused (author microphone table)", "fixed")},
        basis="rigid_mount",
    )
    result.update(
        tach_candidate=True,
        tach_channels=[*range(6), *range(8, 32)],
        source_detail={
            "rig": "ANU 30-mic onboard array, 32 DAQ inputs",
            "motor_current_mA": current,
            "health": meta.get("label", {}).get("health"),
            "earlier_rig_note": "APSIPA 2020 described brushed DC motors and 150 mm two-blade props; not confirmed for this deposit",
            "sample_rate_inferred": True,
        },
    )
    return result


def _audrok(key: str, meta: dict, info: dict) -> dict:
    sub, op = _sub(meta, key), meta.get("operating", {})
    chamber = meta.get("campaign") == "anechoic" or sub.startswith("anechoic_")
    hover = op.get("flight_mode") == "hover" or sub == "anechoic_hover"
    result = _record(
        _AUDROK,
        "flight phase and, at Dresden, translational speed token only; SQL labels and shaft speeds not public",
        environment=(
            "anechoic" if chamber else "outdoors",
            "Measurements phase sequence matches anechoic Fig. 4 (campaign inferred)"
            if chamber
            else "Fraunhofer IVI Dresden field campaign",
        ),
        rig_mount=("free_flight", "unrestrained chamber or outdoor flight, not static rotor bench"),
        propulsion=("propellers", "paper describes small rotorcraft and flight propeller variants"),
        blade_count=(
            "unknown",
            "no take-to-model mapping or as-installed propeller specification in the offer",
        ),
    )
    if chamber:
        _set(
            result,
            _AUDROK,
            sound_content=("drone_only", "single UAV free-field Class A protocol, no added source"),
        )
        speed = (
            (
                "unknown",
                "hover at an external microphone location does not specify motor steady state",
            )
            if hover
            else (
                ("variable", "take-off/landing change thrust from/to ground state")
                if sub in ("anechoic_takeoff", "anechoic_landing")
                else ("unknown", "transit between room positions; no shaft regime specified")
            )
        )
        motion = "fixed" if hover else "moving"
        basis = "nominal_hover" if hover else "flight_protocol"
        reason = (
            "H1/H2/HF are dedicated position holds at the chamber measurement sites; flight corrections can occur"
            if hover
            else "this chamber phase traverses the room or changes altitude near the launch platform"
        )
        roles = {"audio": ("external chamber microphones Mi1–Mi6 (see channel_labels)", motion)}
        if "additional_audio" in meta.get("clock_blocks", []):
            roles = {
                "audio": ("external chamber Mi1–Mi3", motion),
                "additional_audio": ("external chamber Mi4–Mi6, separate unaligned clock", motion),
            }
    else:
        speed = ("unknown", "S/F/var is translational; no documented shaft-speed schedule")
        motion, basis = "moving", "flight_protocol"
        reason = (
            "Dresden flights follow the field route shown in Fig. 5; no labelled stationary spans"
        )
        roles = {
            "audio": ("external Mic1", motion),
            "array_sum_audio": (
                "three publisher array sums (derived), not 72 mic channels",
                motion,
            ),
        }
    _set(result, _AUDROK, speed_regime=speed)
    _motion(result, _AUDROK, motion, reason, roles, basis=basis)
    result.update(
        tach_candidate=True,
        source_detail={
            "campaign": meta.get("campaign"),
            "measurement": meta.get("measurement"),
            "phase": op.get("phase"),
            "drone_index": meta.get("system", {}).get("drone_index"),
            "translation_speed_class": op.get("speed"),
            "clock_blocks": meta.get("clock_blocks"),
            "variant_note": "undocumented variant; paper notes a propeller swap and payload trials; no take mapping"
            if op.get("speed") == "var"
            else None,
        },
    )
    return result


def _audrok_raw(key: str, meta: dict, info: dict) -> dict:
    """The raw mic files are components of the canonical Frames, not extra takes."""
    from data_processing.sources.audrok import _DRESDEN, _MEAS, _SPEED, CHAMBER_UNEQUAL, PHASES

    folders, _, filename = key.rpartition("/")
    distribution = folders.rsplit("/", 1)[-1]
    chamber = distribution.startswith("Measurements_") and _MEAS.fullmatch(
        filename.removesuffix(".xml")
    )
    dresden = distribution.startswith("Dresden-II-S-") and _DRESDEN.fullmatch(
        filename.removesuffix(".wav")
    )
    if chamber:
        num, phase, mic = (chamber[k] for k in ("num", "phase", "mic"))
        meaning, mode, movement = PHASES[phase]
        sub, take = f"anechoic_{movement}", f"{num}-{phase}"
        take_meta = {
            "campaign": "anechoic",
            "measurement": int(num),
            "operating": {"flight_mode": mode, "phase": meaning},
        }
        suffix = f"Mi{mic}"
        entry = "additional_audio" if take == "04-LD" and int(mic) > 3 else "audio"
        channel_index = int(mic) - (4 if entry == "additional_audio" else 1)
        container = "BASt TrafficDataService XML wrapping one base64 WAV"
    elif dresden:
        code = _SPEED[dresden["speed"]]
        sub, take, suffix = f"outdoor_{code}", dresden["take"], dresden["chan"]
        take_meta = {
            "campaign": "dresden",
            "measurement": int(dresden["num"]),
            "system": {"drone_index": int(dresden["drone"])},
            "operating": {"flight_mode": "mixed", "speed": code},
        }
        entry = "audio" if suffix == "Mic1" else "array_sum_audio"
        channel_index = 0 if entry == "audio" else int(suffix.removeprefix("Arr")) - 1
        container = "mono WAV" if entry == "audio" else "mono publisher array-sum WAV"
    else:
        return {
            "provenance_kind": "alias",
            "tach_candidate": False,
            "source_detail": {"release_artifact": filename, "audio_member": False},
        }
    parent_key = f"external/flight/{sub}/{take}"
    take_meta["label"] = {"subcategory": sub}
    result = _audrok(parent_key, take_meta, {})
    raw_role = next(iter(info.get("audio_entries", [])), "xml" if chamber else "wav")
    # Only this channel of this clock block is the same observation as the raw file.
    # Raw t=0 can be later than the block origin; never project another block's estimates.
    starts = CHAMBER_UNEQUAL.get(take, {}).get("starts", {}) if chamber else {}
    rate = info.get("sample_rate") or 44_100  # native rate of every measurement file
    offset = starts.get(suffix, 0) / rate
    parent = {
        "dataset": "AuDroK-frames",
        "key": parent_key,
        "recording_id": take,
        "entry": entry,
        "channel_index": channel_index,
        "role": entry,
        "channels": [channel_index],
        "offset_s": offset,
        "binding": "crop" if take in CHAMBER_UNEQUAL else "exact",
        "binding_reason": f"{suffix} is the unchanged source channel of this take and clock block",
    }
    if info.get("duration_s") is not None:
        parent["duration_s"] = info["duration_s"]
    result.update(
        provenance_kind="alias",
        tach_candidate=True,
        tach_role=raw_role,
        wind_role=raw_role,
        parent=parent,
    )
    result["roles"] = {raw_role: f"{suffix}, {container}"}
    result["role_motion"] = {raw_role: result["relative_motion"]}
    result["source_detail"].update(distribution=distribution, channel=suffix, container=container)
    return result


def _qmul(key: str, meta: dict, info: dict, *, insitu: bool) -> dict:
    ref = _QMUL_INSITU if insitu else _QMUL_BIRD
    sub = _sub(meta, key)
    op = meta.get("operating", {})
    placement = meta.get("observation", {}).get("mic_placement", key.partition("/")[0])
    source_kind = meta.get("source_kind", "")
    result = _record(ref, "no rotor speed telemetry, only hover height labels")
    ambient = op.get("condition") == "rotors_off" or sub.startswith(
        ("ambient_", "playback_no_drone", "playback_source")
    )
    if insitu or sub != "playback_source_reference":
        _set(
            result,
            ref,
            environment=(
                "outdoors",
                "woodland field acquisition" if insitu else "outdoor playback experiment",
            ),
        )
    if ambient:
        _off(
            result,
            ref,
            "ambient-only ground recording"
            if insitu
            else "drone-off loudspeaker control or third-party playback reference, not rotor audio",
        )
        result["roles"] = {
            "audio": "ground recorder"
            if insitu
            else "original bird playback file"
            if sub == "playback_source_reference"
            else "microphone 1 m above the bird-playback speaker"
        }
    else:
        _set(
            result,
            ref,
            rig_mount=(
                "free_flight",
                "hovering Mavic 2 Pro; it is the microphone that hangs on a line, not the aircraft tethered to ground",
            ),
            propulsion=(
                "propellers",
                "Mavic 2 Pro operating with propellers in the field protocol",
            ),
            speed_regime=("unknown", "parked hover position, rotor speed / drift not measured"),
            sound_content=(
                "co_recorded",
                "wild bird calls plus drone and woodland ambience"
                if insitu
                else "bird calls emitted by a ground speaker and picked up live with rotor noise; not digitally added",
            ),
        )
        _rotors(
            result,
            ref,
            4,
            "Mavic 2 Pro is a quadrotor",
            blade="2",
            blade_reason="DJI specifies two-blade 8743 Mavic 2 props; Wang et al. used the identified Mavic 2 Pro",
            blade_basis="inferred_stock",
        )
        roles = {
            "audio": (
                "microphone hanging on a flexible 2 m line under the UAV; no measured sway",
                "unknown",
            )
        }
        bases = {"audio": "unmeasured_suspension"}
        if placement == "external":
            roles = {"audio": ("external ground recorder during nominal hover", "fixed")}
            bases["audio"] = "nominal_hover"
        if "ground_audio" in (meta.get("series", {}) or info.get("audio_entries", [])):
            roles["ground_audio"] = (
                "independent ground recorder; nominal hover, not phase-aligned",
                "fixed",
            )
            bases["ground_audio"] = "nominal_hover"
        motion = "mixed" if len(roles) > 1 else roles["audio"][1]
        _motion(
            result,
            ref,
            motion,
            "nominal position hold over the ground recorder during the drone-on hover spans; "
            "the line-suspended recorder may sway relative to the aircraft",
            roles,
            basis="mixed_roles" if len(roles) > 1 else bases["audio"],
        )
        result["role_motion_basis"] = bases
        result["tach_candidate"] = True
    if insitu and (source_kind not in ("", "original_recording") or sub.startswith("derivative")):
        result.update(
            provenance_kind="derivative",
            tach_candidate=False,
            parent={
                "dataset": "QMUL-insitu-aerial-frames",
                "recording_ids": meta.get("parent_frames", []),
                "derivation": meta.get("derivative_of"),
            },
        )
    result["source_detail"] = {
        "rig": None if ambient else "DJI Mavic 2 Pro, 2 m suspended mic/recorder",
        "propeller": None if ambient else "DJI 8743 (stock specification)",
        "hover_altitude_m": op.get("hover_altitude_m"),
        "processing": source_kind,
        "session": meta.get("label", {}).get("session"),
    }
    return result


def _qmul_insitu(key: str, meta: dict, info: dict) -> dict:
    return _qmul(key, meta, info, insitu=True)


def _qmul_bird(key: str, meta: dict, info: dict) -> dict:
    return _qmul(key, meta, info, insitu=False)


def _chiro(key: str, meta: dict, info: dict) -> dict:
    sub = _sub(meta, key)
    result = _record(_CHIRO, "iOSD motor outputs are command codes, not calibrated RPM")
    if sub == "iosd_native_log" or meta.get("source_kind") == "original_telemetry_log":
        result.update(
            provenance_kind="log_only",
            n_rotors=6,
            parent={"dataset": "Chirocopter-frames", "recording_id": meta.get("associated_audio")},
        )
        return result
    baffle_test = sub.startswith("noise_test_")
    result.update(
        n_rotors=6,
        tach_candidate=True,
        source_detail={
            "rig": "DJI Spreading Wings S900, A2 controller",
            "motor": "DJI 4114 (stock S900 manual)",
            "propeller": "DJI 1552 (stock S900 manual)",
            "baffle": meta.get("label", {}).get("baffle"),
            "pole_1m": meta.get("label", {}).get("pole_1m"),
            "flight_log": meta.get("associated_flight_log_id"),
        },
    )
    _motion(
        result,
        _CHIRO,
        "fixed",
        "onboard microphone mounted to the hexacopter (with or without a pole)",
        {"audio": ("Ultramic-250K on the S900; see per-take baffle configuration", "fixed")},
        basis="rigid_mount",
    )
    if baffle_test:
        _set(
            result,
            _CHIRO,
            rig_mount=(
                "unknown",
                "noise test filenames describe baffle/mic mounting; no aircraft mounting or flight condition",
            ),
            speed_regime=("unknown", "50 ms baffle-test WAVs do not establish shaft regime"),
            rotor_activity=(
                "unknown",
                "how many of the six motors were powered is not documented for these tests",
            ),
        )
    else:
        _set(
            result,
            _CHIRO,
            rig_mount=(
                "free_flight",
                "bat observations use an unrestrained hover ladder above the cave",
            ),
            speed_regime=(
                "mixed",
                "take-off/landing and changes of altitude, with one-minute nominal hover plateaux",
            ),
            propulsion=("propellers", "S900 is flying with propellers"),
            sound_content=(
                "co_recorded",
                "wild Tadarida brasiliensis echolocation recorded live with drone sound",
            ),
            environment=("outdoors", "above a natural cave entrance, New Mexico"),
            wind=(
                "unknown",
                "operations in weather below 10 km/h do not characterize mic wind or baffle airflow",
            ),
        )
        _rotors(
            result,
            _CHIRO,
            6,
            "S900 hexacopter uses six motors in flight",
            blade="2",
            blade_reason="Fu et al. Fig. 1 shows the rig; S900 stock 1552 is two-blade",
            blade_basis="inferred_stock",
        )
    return result


def _sdu(key: str, meta: dict, info: dict) -> dict:
    sub = _sub(meta, key)
    result = _record(
        _SDU,
        "published settings are mic suspension distances and sonar padding, not shaft telemetry",
    )
    if sub == "system_noise_floor":
        _off(result, _SDU, "recorder self-noise, microphones embedded in foam, no drone")
        _set(
            result,
            _SDU,
            wind=("not_applicable", "control isolates recorder and capsule self-noise with foam"),
        )
        result["roles"] = {"audio": "four-channel foam-isolated receiver-system floor"}
        return result
    bat_excerpt = sub == "field_bats_tether_30m"
    sonar = sub.endswith("_sonar_unshielded")
    mono = sub == "noise_test_tether_1m"
    content = (
        "bat calls recorded while drone hovers"
        if bat_excerpt
        else (
            "unshielded onboard 40 kHz sonar pinger audible in addition to rotor noise"
            if sonar
            else "open-field rotor-noise test; 40 kHz onboard sonar attenuated by foam; incidental ambient not ruled out"
        )
    )
    _set(
        result,
        _SDU,
        environment=("outdoors", "SDU open-field noise tests or bat field sites around Odense"),
        rig_mount=(
            "free_flight",
            "nylon tether between the free hovering drone and the mic array, not the ground",
        ),
        speed_regime=("unknown", "hovering position not evidence for fixed RPM"),
        propulsion=("propellers", "hovering Matrice 210 V2 rotorcraft"),
        sound_content=("co_recorded" if bat_excerpt or sonar else "drone_only", content),
        wind=(
            "unknown",
            "array high-pass and noise controls do not measure mic wind; 1 m mono chain differs",
        ),
    )
    _motion(
        result,
        _SDU,
        "unknown",
        "tether sag/sway not recorded; 1 m mono microphone not identified",
        {
            "audio": (
                "mono noise-test microphone (identity unknown)"
                if mono
                else "vertical four-mic suspended line array",
                "unknown",
            )
        },
        basis="unmeasured_suspension",
    )
    _rotors(
        result,
        _SDU,
        4,
        "Matrice 210 V2 quadrotor in hover",
        blade="2",
        blade_reason="M210 V2 is identified; DJI nominal 1760S is two-blade, installed SKU not recorded",
        blade_basis="inferred_stock",
    )
    op = meta.get("operating", {})
    result.update(
        tach_candidate=True,
        provenance_kind="derivative" if bat_excerpt else "recording",
        source_detail={
            "rig": "DJI Matrice 210 V2",
            "propeller_nominal": "DJI 1760S",
            "suspension_length_m": op.get("tether_m"),
            "sonar_foam_pad": op.get("sonar_foam_pad"),
            "species": meta.get("label", {}).get("species"),
            "usable_band": "actual per-file chain unpublished; archive probe finds low bands survive in 1 m mono and are strongly suppressed in bat excerpts",
        },
    )
    return result


def _hear(key: str, meta: dict, info: dict) -> dict:
    phone = _sub(meta, key) == "phone_audio"
    result = _record(
        _HEAR,
        "no turbomachinery/rotor speed files; flight/video clock cues are not tachometry",
        environment=("outdoors", "Lelystad general aviation airfield"),
        rig_mount=("not_applicable", "manned aircraft, no UAV rig in release"),
        speed_regime=("not_applicable", "no drone rotor regime"),
        propulsion=(
            "not_applicable",
            "full-size aircraft propellers and helicopters, not drone propulsion",
        ),
        rotor_activity=("not_applicable", "not drone recordings"),
        blade_count=("not_applicable", "not drone recordings"),
        relative_motion=("not_applicable", "no drone hubs to compare with the microphone"),
        sound_content=(
            "no_drone",
            "manned flyovers and airfield background; the paper's onboard Bebop recordings are not in the deposit",
        ),
    )
    result.update(
        provenance_kind="non_drone",
        roles={
            "audio": "ground phone track"
            if phone
            else "one rescaled channel of the ground acoustic camera"
        },
        source_detail={
            "control": "manned-aircraft negative",
            "take_kind": "phone_audio" if phone else "array_channel",
            "array_capture_derivative": not phone,
        },
    )
    return result


def _ku(key: str, meta: dict, info: dict) -> dict:
    op = meta.get("operating", {})
    rpm = op.get("rpm_mean")
    result = _record(
        _KU,
        "no shaft-speed payload in the take metadata",
        environment=("semi_anechoic", "Siemens DISW chamber, Leuven"),
        rig_mount=("fixed_stand", "vehicle is clamped in place while airframe simulates hover"),
        propulsion=("propellers", "installed propellers, one laser probe beneath one blade"),
        speed_regime=(
            "constant",
            "fixed joystick throttle on an immobilized stand; intended steady runs, not proven constant per shaft",
        ),
        sound_content=("co_recorded", "male VCTK utterance played live about 2 m from the stand"),
    )
    _rotors(
        result,
        _KU,
        4,
        "MK EASY Quadro V3 (+) with four powered rotors",
        blade="2",
        blade_reason="two blades per 274.8 mm propeller in the documented test rig",
        blade_basis="reported",
    )
    _motion(
        result,
        _KU,
        "fixed",
        "both drone-mounted mics and external reference have fixed geometry to the stand",
        {
            "audio": (
                "Ch01–12 main array, Ch13–16 under-rotor references, Ch17 fixed external speech reference",
                "fixed",
            )
        },
        basis="fixed_stand",
    )
    result.update(
        tach_candidate=True,
        tach_channels=list(range(16)),
        wind_channels=list(range(16)),
        source_detail={
            "rig": "MikroKopter MK EASY Quadro V3",
            "propeller_diameter_m": 0.2748,
            "health": meta.get("label", {}).get("health"),
            "rpm_mean": rpm,
            "channel_groups": {
                "onboard_main": [1, 12],
                "onboard_rotor_refs": [13, 16],
                "external_speech": [17, 17],
            },
            "reference_clock": "Ch17 not phase-verified; fixed geometrically, excluded from estimation",
        },
    )
    if rpm in (4000, 5000, 6000):
        _set(
            result,
            _KU,
            speed_supplied=(
                "yes",
                f"{rpm} rpm, published laser-probe run average under one propeller; no per-rotor trajectory or run dispersion",
            ),
        )
        result.update(supplied_kind="run_average", nominal_rps=rpm / 60)
    return result


def _idmt(key: str, meta: dict, info: dict) -> dict:
    op, system = meta.get("operating", {}), meta.get("system", {})
    pattern = op.get("flight_pattern")
    drone = system.get("drone_token")
    result = _record(
        _IDMT,
        "GPS and DOA are positions/directions, not motor speed",
        rig_mount=(
            "free_flight",
            "recorded field manoeuvres or hover; the array, not the aircraft, is fixed",
        ),
        environment=("outdoors", "field site Berne, Lower Saxony"),
        propulsion=("propellers", "airborne rotorcraft recorded by the array"),
        speed_regime=("unknown", f"{pattern} is a flight path, not a shaft-speed measurement"),
    )
    if pattern == "static":
        motion, basis = "fixed", "nominal_hover"
        reason = "publisher static flights are nominal position holds relative to the fixed array; GPS accuracy is limited"
    elif pattern in ("linear", "random"):
        motion, basis = "moving", "flight_protocol"
        reason = "publisher linear/random tracks change position relative to the fixed array; GPS assumes equal altitude"
    else:
        motion, basis, reason = "unknown", "unknown", "relative flight path not identified"
    _motion(
        result,
        _IDMT,
        motion,
        reason,
        {"audio": ("external 8-channel cubic ground array", motion)},
        basis=basis,
    )
    if drone == "Phantom4":
        _rotors(
            result,
            _IDMT,
            4,
            "DJI Phantom 4 quadrotor",
            blade="2",
            blade_reason="two-blade 9450S stock propellers on the Phantom 4 (DJI manual); installed SKU unpublished",
            blade_basis="inferred_stock",
        )
    elif drone == "Mavic":
        _rotors(
            result,
            _IDMT,
            4,
            "DJI Mavic quadrotor",
            blade_reason="the deposit does not identify the Mavic variant or its installed blades",
        )
    else:
        _set(result, _IDMT, blade_count=("unknown", "Align type and installed blades unspecified"))
    result.update(
        tach_candidate=True,
        source_detail={"rig": system.get("model"), "drone_token": drone, "flight_pattern": pattern},
    )
    return result


def _salford(key: str, meta: dict, info: dict) -> dict:
    sub = _sub(meta, key)
    result = _record(
        _SALFORD,
        "flight plan speed is translational m/s; no rotor telemetry",
        environment=("outdoors", "Edzell field campaign, including microphone calibration"),
    )
    if sub == "calibrator_tone":
        _off(result, _SALFORD, "independent DAQ-channel acoustic calibrator take; no active drone")
        _set(
            result,
            _SALFORD,
            wind=("not_applicable", "coupled calibrator, not an open-air microphone measurement"),
        )
        result["roles"] = {"audio": "external DAQ channel with acoustic calibrator coupler"}
        return result
    op, system = meta.get("operating", {}), meta.get("system", {})
    hover = op.get("flight_mode") == "hover" or sub.endswith("_hover")
    _set(
        result,
        _SALFORD,
        rig_mount=("free_flight", "airborne hover or flyover above the ground line array"),
        propulsion=("propellers", "depicted airborne propeller-equipped multicopters"),
        speed_regime=(
            "unknown",
            "nominal hover / flyover translation settings are not motor-speed settings",
        ),
        sound_content=(
            "drone_only",
            "target drone with no additional experimental source, incidental field ambience not ruled out",
        ),
        wind=(
            "unknown",
            "direction tokens and weather/windscreens do not establish recorded wind noise",
        ),
    )
    if sub == "m3_hover" and op.get("payload_attached"):
        _set(
            result,
            "doi:10.61782/fa.2023.0249 §3 Figs. 4–5",
            speed_regime=(
                "constant",
                "the M3-with-payload hover has persistent, stable rotor tones across the nominal 20 s hold; no independent per-shaft record",
            ),
        )
    motion = "fixed" if hover else "moving"
    _motion(
        result,
        _SALFORD,
        motion,
        "20 s position hold above M5; group TDOA is consistent with nominal stationarity, not rigid coupling"
        if hover
        else "aircraft traverses the ground array centre at nominal 15 m/s; no Doppler correction",
        {"audio": ("M1–M9 ground line array, or M5 alone for marked events", motion)},
        basis="nominal_hover" if hover else "flight_protocol",
    )
    count = system.get("n_rotors")
    _rotors(
        result, _SALFORD, count, "four (3p/M3/Fp) or six (Yn), paper Table 1 and flyover workbook"
    )
    if sub.startswith("fp_"):
        _set(
            result,
            _SALFORD,
            blade_count=(
                "unknown",
                "795 g and 4 rotors constrain Fp; DJI FPV is inferred and actual propellers unpublished",
            ),
        )
    else:
        _set(
            result,
            _SALFORD,
            blade_count=(
                "2",
                "identified factory aircraft platforms with nominal two-blade propellers; no replacement SKU identified",
            ),
        )
        result["blade_count_basis"] = "inferred_stock"
    result.update(
        tach_candidate=True,
        source_detail={
            "rig": system.get("model") or system.get("model_inferred"),
            "suas_code": system.get("suas_code"),
            "profile": "hover" if hover else "flyover",
            "nominal_speed_m_s": op.get("nominal_speed_m_s"),
            "payload_attached": op.get("payload_attached"),
            "payload_g": system.get("payload_g"),
            "height_agl_m": op.get("hagl_m"),
            "start_side": op.get("start_side"),
            "direction_token": op.get("direction_token"),
        },
    )
    return result


def _uavir(key: str, meta: dict, info: dict) -> dict:
    result = _record(
        _UAVIR,
        "source distance, height, azimuth and rotation describe static hover pose, not shaft speed",
        environment=("outdoors", "field site Warsaw, external mast; 2024-11-15"),
        wind=(
            "unknown",
            "per-take weather-station measurements are not mic wind noise; synchronization not established",
        ),
    )
    if _sub(meta, key) == "ambient_noise":
        _off(result, _UAVIR, "publisher Ambient Noise tag, no drone source")
        result["roles"] = {"audio": "8-channel external mast ambient recording"}
        return result
    _set(
        result,
        _UAVIR,
        rig_mount=(
            "free_flight",
            "Mavic 3 Cine hovering above the ground with four aspect orientations",
        ),
        propulsion=("propellers", "propeller-driven DJI Mavic 3 Cine airborne"),
        speed_regime=(
            "unknown",
            "static position is not a fixed RPM condition; no per-shaft measurements",
        ),
        sound_content=(
            "drone_only",
            "Drone-labelled samples, no deliberate other sound source; separate ambient controls",
        ),
    )
    _motion(
        result,
        _UAVIR,
        "fixed",
        "take is a nominal fixed-pose hold with respect to the mast; flight corrections are not tracked",
        {"audio": ("8 synchronized NTG-2 mics on a ground mast", "fixed")},
        basis="nominal_hover",
    )
    _rotors(
        result,
        _UAVIR,
        4,
        "DJI Mavic 3 Cine quadrotor",
        blade="2",
        blade_reason="release identifies Mavic 3 Cine; DJI's nominal 9453F is two-blade, installed SKU not recorded",
        blade_basis="inferred_stock",
    )
    label = meta.get("label", {})
    result.update(
        tach_candidate=True,
        source_detail={
            "rig": "DJI Mavic 3 Cine",
            "propeller": "DJI 9453F (stock specification)",
            **{
                k: label[k]
                for k in ("distance_m", "height_m", "azimuth_deg", "rotation")
                if k in label
            },
        },
    )
    return result


_CLASSIFIERS = {
    "D3SET-frames": _d3set,
    "AIRA-UAS-frames": _aira,
    "ANU-drone-motor-current-frames": _anu,
    "AuDroK": _audrok_raw,
    "AuDroK-frames": _audrok,
    "QMUL-insitu-aerial-frames": _qmul_insitu,
    "QMUL-bird-drone-frames": _qmul_bird,
    "Chirocopter-frames": _chiro,
    "SDU-DroneBatArray-frames": _sdu,
    "Hear-and-Avoid-frames": _hear,
    "KULeuven-UAV-egonoise-frames": _ku,
    "IDMT-Berne-frames": _idmt,
    "Salford-DroneNoise-frames": _salford,
    "UaVirBASE-frames": _uavir,
}
DATASETS = frozenset(_CLASSIFIERS)


def classify(dataset: str, key: str, meta: dict, info: dict) -> dict:
    """Classify a native production sample key and unwrapped Frame metadata."""
    result = _CLASSIFIERS[dataset](key, meta, info)
    result.setdefault("blade_count_basis", "unknown")
    return result
