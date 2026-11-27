"""Recording conditions of derived, mixed, aggregated and simulated drone-noise datasets.

Pure metadata classification: no waveform, network or file I/O. Rows that copy or
cut a native recording take the rig facts of that exact parent recording through
``legacy.parent_facts`` and then state what the derivation changed (digital
speech, resampling, cut window, channel). Where the parent window, rate and
channels are known on the parent's *current* time base the row carries an
``inherit`` block: the runner copies the parent's acoustic shaft-rate windows
(and, when the channels match, its wind result) instead of re-estimating a
duplicate. ``tach_candidate`` is reserved for audio that exists nowhere else in a
better form.

Sample-dir (``DREGON-LM*``, ``DN-LM*``, ``rp-real*``) provenance lives in the
dataset's ``_meta`` sample: the runner calls :func:`summary_rows` once per
dataset and merges ``rows[key]`` into ``meta`` before :func:`classify`.
"""

from __future__ import annotations

import json
import re
from typing import Any

# ── dataset scope ───────────────────────────────────────────────────────────

#: Speech (or white-noise) digitally mixed over cut drone noise: DREGON-LM V1-V4 + probes.
DREGON_LM_SYNTH = frozenset(
    {
        "DREGON-LM-train",
        "DREGON-LM-valid",
        "DREGON-LM-test-train",
        "DREGON-LM-test-valid",
        "DREGON-LM-V2-train",
        "DREGON-LM-V2-valid",
        "DREGON-LM-V3-train",
        "DREGON-LM-V3-valid",
        "DREGON-LM-V4-train",
        "DREGON-LM-V4-michaels-train",
        "DREGON-LM-rps_eval_long_samples",
        "DREGON-LM-rps_eval_specific_samples",
        "DREGON-LM-rps_train_specific_samples",
    }
)
#: ``real_valid`` clips: the raw recording, NOT a digital mixture.
DREGON_LM_REAL = frozenset(
    {
        "DREGON-LM-V4-valid",
        "DREGON-LM-V4-michaels-valid",
        "DREGON-LM-V4-michaels-valid-full",
        "DREGON-LM-V4-michaels-valid-full-nospeech",
    }
)
#: Paper real-data panels: raw clips plus (valid only) FLY124 + dev-clean mixtures.
RP_REAL = frozenset({"rp-real-valid-v1", "rp-real-test-v1"})
DN_LM = frozenset({"DN-LM-train", "DN-LM-valid"})
SE_VALID = frozenset(
    {"SE-valid-harmonic", "SE-valid-drone", "SE-valid-avq-survey", "SE-valid-avq-split"}
)
#: Whole-recording copies / analyses of project flights.
RECORDING_COPIES = frozenset({"beatvk-valid-raw", "decomp-frames-v1", "decomp-frames-v2"})
#: Publisher-made mixtures of unreleased tethered anechoic recordings.
KAIST_FAMILY = frozenset({"KAIST-drone-fault-frames", "ICSV31-drone-anomaly-frames"})
#: Third-party aggregates and raw public trees.
PUBLIC_TREES = frozenset(
    {"DroneAudioSet", "drone_audio", "drone-detection-samples", "zenodo_drone_noises"}
)
#: Auralisations (no recorded audio).
SIMULATIONS = frozenset({"drone-egonoise", "jasa-flyovers"})

DATASETS = frozenset(
    DREGON_LM_SYNTH
    | DREGON_LM_REAL
    | RP_REAL
    | DN_LM
    | SE_VALID
    | RECORDING_COPIES
    | KAIST_FAMILY
    | PUBLIC_TREES
    | SIMULATIONS
)

_TEN = (
    "rig_mount",
    "speed_regime",
    "propulsion",
    "rotor_activity",
    "relative_motion",
    "speed_supplied",
    "sound_content",
    "environment",
    "blade_count",
    "wind",
)
_RIG = (
    "rig_mount",
    "propulsion",
    "rotor_activity",
    "relative_motion",
    "environment",
    "blade_count",
)

#: DREGON held-out TEST recordings (``sources.dregon.SPLIT_RECORDINGS["test"]``).
_DREGON_TEST = frozenset(
    {
        "free-flight_speech-high_room1",
        "free-flight_whitenoise-high_room1",
        "silent-flight_whitenoise-low_room1",
    }
)
_MICHAELS_TEST = frozenset({"FLY103", "FLY108"})

# ── sources (evidence prefixes) ─────────────────────────────────────────────

_CAT = "docs/data-catalog.md"
_V4 = f"derivations.SPECS (dregon_lm, adopt-only) + {_CAT} § DREGON-LM V4"
_HIST = (
    f"derivations.HISTORICAL_PINS + published _meta metadata.json ({_CAT} § Superseded variants)"
)
_RP = f"derivations.SPECS real_valid_panel + per-clip meta.json ({_CAT} § rp-real-valid-v1)"
_DNLM = f"derivations.SPECS dn_lm + _meta metadata.json ({_CAT} § DN-LM)"
_SE = "derivations.SE_CATEGORY_NOISE + docs/explainers/harmonic-noise-datasets/audit.py waveform audit"
_BEAT = "derivations.SPECS beatvk-valid-raw + per-Frame meta (source_dataset/source_version/spans)"
_DECOMP = "derivations.SPECS decomp_frames + per-Frame meta (source_spec, label_variant)"
_KAIST = "Zenodo 7779574 + Yi, Choi & Lee ICSV29 2023 (arXiv:2304.11708 §2); sources/kaist_drone.py"
_ICSV = "ICSV31 AI Challenge archives (derived from Zenodo 7779574); sources/kaist_drone.py ICSV31_PROVENANCE"
_DAS = "DroneAudioSet arXiv:2510.15383 §3.1-3.5, Fig. 1(b), Tables 2/4, Fig. 8; HF file_path"
_ALEMADI = (
    "Al-Emadi et al., Sensors 21:4953 (2021) §3.3.1/Table 1; github saraalemadi/DroneAudioDataset"
)
_DADS = (
    f"DADS card (HF geronimobasso/drone-audio-detection-samples) + audited id blocks ({_CAT} "
    "§ Derivative overlap; KAIST block ordering rule from PCM matches)"
)
_SURVEY = "Mukhutdinov et al., IEEE Access 2023, doi:10.1109/ACCESS.2023.3253719 §V-VI"
_AS = "Wang & Cavallaro, IEEE Sensors J. 18(11) 2018, doi:10.1109/JSEN.2018.2825879 §III"
_EGO = "auraflow datasets/drone_egonoise.py + jasa.generate_flyover(prescribed_omega); sample meta JSON"
_JASA = "auraflow datasets/jasa.py + nasa_1pax.py (4 rotors x 3 blades, hover BPF 33.55 Hz); sample meta JSON"


# ── record helpers ──────────────────────────────────────────────────────────


def _new(kind: str) -> dict[str, Any]:
    return {"evidence": {}, "provenance_kind": kind, "tach_candidate": False}


def _too_short(rec: dict, info: dict, nominal_s: float, why: str) -> None:
    """Powered drone audio that would be a tachometry case except for its length: the runner
    reports ``tach_status='too_short'`` with this reason instead of an unexplained absence."""
    duration = info.get("duration_s")
    seconds = float(duration) if duration is not None else nominal_s
    rec["tach_candidate"] = False
    if seconds < 1.5:
        rec["tach_skip"] = {
            "status": "too_short",
            "reason": why,
            "duration_s": seconds,
            "min_s": 1.5,
        }


def _set(rec: dict, ref: str, **facets: tuple[str, str]) -> dict:
    for facet, (value, reason) in facets.items():
        rec[facet] = value
        rec["evidence"][facet] = f"{ref}: {reason}"
    return rec


def _no_drone(
    rec: dict, ref: str, reason: str, *, environment: tuple[str, str] | None = None
) -> dict:
    """A clip without any drone: rig/rotor facets do not apply."""
    na = "no drone in this audio"
    _set(
        rec,
        ref,
        sound_content=("no_drone", reason),
        rig_mount=("not_applicable", na),
        speed_regime=("not_applicable", na),
        propulsion=("not_applicable", na),
        rotor_activity=("not_applicable", na),
        relative_motion=("not_applicable", na),
        speed_supplied=("not_applicable", na),
        blade_count=("not_applicable", na),
    )
    if environment:
        _set(rec, ref, environment=environment)
    rec.update(n_active=0, tach_candidate=False)
    return rec


def _parent(dataset: str, rid: str) -> dict[str, Any]:
    from . import legacy

    return legacy.parent_facts(dataset, rid)


def _digital(value: str) -> str:
    """Sound content after a digital source was added to parent audio."""
    return {
        "drone_only": "digital_mix",
        "co_recorded": "both",
        "digital_mix": "digital_mix",
        "both": "both",
        "no_drone": "no_drone",
    }.get(value, "unknown")


def _from_parent(pds: str, rid: str, *, window: bool, mixed: bool, how: str) -> dict:
    """Parent rig facts for an audio row cut (``window``) or copied from ``pds/rid``."""
    p = _parent(pds, rid)
    pe = p.get("evidence", {})
    tag = f"parent {pds}/{rid} ({how})"
    rec = _new("derivative")
    for facet in _RIG:
        if facet in p:
            rec[facet] = p[facet]
            rec["evidence"][facet] = f"{tag}: {pe.get(facet, 'parent rule')}"
    for extra in ("n_rotors", "n_active", "shaft_orders"):
        if p.get(extra) is not None:
            rec[extra] = p[extra]
    regime = p.get("speed_regime", "unknown")
    if not window or regime in ("constant", "off"):
        rec["speed_regime"] = regime
        rec["evidence"]["speed_regime"] = f"{tag}: {pe.get('speed_regime', 'parent rule')}"
    else:
        _set(
            rec,
            tag,
            speed_regime=(
                "unknown",
                f"cut window of a parent whose whole-take regime is "
                f"{regime}; window regime follows from inherited estimates",
            ),
        )
    wind = p.get("wind", "unknown")
    if not window or wind in ("not_significant", "not_applicable"):
        rec["wind"] = wind
        rec["evidence"]["wind"] = f"{tag}: {pe.get('wind', 'parent rule')}"
    else:
        _set(
            rec,
            tag,
            wind=("unknown", f"parent whole-take wind {wind}; window-level value only via inherit"),
        )
    content = p.get("sound_content", "unknown")
    if mixed:
        _set(
            rec,
            tag,
            sound_content=(
                _digital(content),
                f"parent content {content} plus a digitally added speech/white-noise target",
            ),
        )
    else:
        rec["sound_content"] = content
        rec["evidence"]["sound_content"] = f"{tag}: {pe.get('sound_content', 'parent rule')}"
    rec["parent"] = {"dataset": pds, "key": rid}
    return rec


def _merge(records: list[dict], ref: str) -> dict:
    """Facts common to every candidate parent; disagreeing facets become unknown."""
    out = _new("derivative")
    first = records[0]
    for facet in _TEN:
        values = {r.get(facet, "unknown") for r in records}
        if len(values) == 1:
            out[facet] = values.pop()
            out["evidence"][facet] = (
                f"{ref}; all candidate parents agree: {first['evidence'].get(facet, '')}"
            )
        else:
            _set(
                out,
                ref,
                **{facet: ("unknown", f"candidate parents disagree ({', '.join(sorted(values))})")},
            )
    for extra in ("n_rotors", "n_active"):
        vals = {r.get(extra) for r in records}
        if len(vals) == 1 and None not in vals:
            out[extra] = vals.pop()
    out["parent"] = {"candidates": [r["parent"] for r in records]}
    return out


def _bind(
    rec: dict,
    offset_s: float | None,
    duration_s: float | None,
    *,
    channels: list[int] | None,
    safe: bool,
    reason: str,
    exact: bool = False,
    verify: str | None = None,
) -> None:
    """Runner projection seam: ``parent.binding`` exact/crop lets the campaign copy the
    parent's acoustic windows and wind (shifted by ``offset_s``); ``none`` borrows nothing."""
    parent = rec["parent"]
    binding = ("exact" if exact else "crop") if safe and offset_s is not None else "none"
    parent.update(role="audio", binding=binding, binding_reason=reason)
    if offset_s is not None:
        parent["offset_s"] = float(offset_s)
    if duration_s is not None:
        parent["duration_s"] = float(duration_s)
    if channels is not None:
        parent["channels"] = channels
    if verify:
        parent["verify"] = verify


def _supplied(rec: dict, ref: str, info: dict, kind: str | None, reason: str) -> None:
    if "rps" in info.get("fields", ()) or "rps_raw" in info.get("fields", ()):
        _set(rec, ref, speed_supplied=("yes", reason))
        if kind:
            rec["supplied_kind"] = kind
    else:
        _set(rec, ref, speed_supplied=("no", "no rotor-speed field in this sample"))


def _drone_stem(info: dict) -> str:
    """Audio entry holding the drone noise of a sample-dir row: the unmixed ``noise`` stem
    when published, else ``mixture`` (raw real_valid clips, digital mixes without a stem)."""
    return "noise" if "noise" in info.get("fields", ()) else "mixture"


# ── project-flight parents ──────────────────────────────────────────────────


def _flight_parent(name: str) -> tuple[str, str]:
    """Recording id as written by the mixers -> (parent dataset, key)."""
    name = re.sub(r"_ch\d$", "", name)
    if name.startswith("michaels_"):
        name = name.removeprefix("michaels_")
    if name in _MICHAELS_TEST:
        return "michaels-test-frames", name
    if re.fullmatch(r"FLY\d+", name):
        return "michaels-frames", name
    return "DREGON-frames", name


def _channel(name: str) -> list[int] | None:
    m = re.search(r"_ch(\d)$", name)
    return [int(m[1])] if m else None


def _held_out(pds: str, rid: str) -> bool:
    return (pds == "DREGON-frames" and rid in _DREGON_TEST) or pds == "michaels-test-frames"


def _split_of(dataset: str) -> str:
    if dataset == "rp-real-test-v1":
        return "test"
    if dataset.endswith("-train") or dataset == "DREGON-LM-rps_train_specific_samples":
        return "train"
    return "valid"


#: rps_eval probes: per-sample metadata.txt names a DREGON-LM (V1) valid sample; that
#: sample's row in the V1 metadata.json (read 2026-11-27 from the pinned bytes) gives
#: the noise recording/offset. ``crop`` = the probe's extra start inside the V1 sample.
_EVAL_PROBES: dict[str, dict[str, tuple[str, str, float, float]]] = {
    "DREGON-LM-rps_eval_long_samples": {
        "sample_000": (
            "sample_00114",
            "free-flight_nosource_room2",
            20.6132211028966,
            0.03125649598963226,
        ),
        "sample_001": (
            "sample_00025",
            "free-flight_nosource_room2",
            31.547670498207985,
            0.0229589194624177,
        ),
        "sample_002": (
            "sample_00281",
            "rectangle_nosource_room2",
            25.460069945015306,
            0.16590957480555954,
        ),
        "sample_003": (
            "sample_00250",
            "free-flight_nosource_room1",
            52.81854030611366,
            0.12216210355963046,
        ),
        "sample_004": (
            "sample_00228",
            "free-flight_nosource_room2",
            40.31474079974413,
            0.132270322788585,
        ),
    },
    "DREGON-LM-rps_eval_specific_samples": {
        "sample_00000": ("sample_00000", "hovering_nosource_room2", 17.891109671244894, 0.0),
        "sample_00149": ("sample_00149", "hovering_nosource_room2", 19.993936938126875, 0.0),
        "sample_00299": ("sample_00299", "updown_nosource_room2", 35.55014492202716, 0.0),
        "sample_00449": ("sample_00449", "free-flight_nosource_room2", 11.549842489576047, 0.0),
        "sample_00599": ("sample_00599", "spinning_nosource_room2", 10.564170800262662, 0.0),
    },
}

_SUMMARY_SETS = DREGON_LM_SYNTH | DREGON_LM_REAL | RP_REAL | DN_LM


def summary_rows(dataset: str, blobs: dict[str, bytes]) -> dict[str, dict[str, Any]]:
    """Per-key provenance rows of a sample-dir dataset from its published ``_meta``.

    ``blobs["_meta"]`` = bytes of the ``_meta`` sample's ``metadata`` field; for the
    rps_eval probe sets also ``blobs[key]`` = that sample's ``metadata`` (txt). The root
    metadata.json holds every split of the recipe with colliding ids, so the list is
    chosen by the dataset's own split. Raises on any inconsistency."""
    if dataset not in _SUMMARY_SETS:
        return {}
    if dataset in _EVAL_PROBES:
        rows = {}
        for key, (src, rec, start, crop) in _EVAL_PROBES[dataset].items():
            text = blobs.get(key)
            if text is not None:
                got = re.search(rb"Source: datasets/DREGON-LM/valid/(sample_\d+)", text)
                if got is None or got[1].decode() != src:
                    raise ValueError(f"{dataset}/{key}: metadata.txt does not name valid/{src}")
            rows[key] = {
                "id": key,
                "source_split": "valid",
                "source_id": src,
                "noise_source": rec,
                "noise_start_time": start + crop,
                "probe_crop_s": crop,
            }
        return rows
    blob = blobs.get("_meta")
    if blob is None:
        raise ValueError(f"{dataset}: _meta metadata missing")
    data = json.loads(blob)
    split = _split_of(dataset)
    if split not in data or not isinstance(data[split], list):
        raise ValueError(f"{dataset}: _meta has no {split!r} list ({sorted(data)})")
    rows = {}
    for row in data[split]:
        if not isinstance(row, dict) or "id" not in row:
            raise ValueError(f"{dataset}: malformed _meta row {row!r}")
        rows[row["id"]] = dict(row, source_split=split)
    return rows


# ── DREGON-LM ───────────────────────────────────────────────────────────────


def _dregon_lm(dataset: str, key: str, meta: dict, info: dict) -> dict:
    split = _split_of(dataset)
    if dataset in DREGON_LM_REAL:
        rec = _dregon_lm_real(dataset, key, meta, info)
    else:
        rec = _dregon_lm_synth(dataset, key, meta, info)
    parent = rec.get("parent", {})
    rec["source_detail"] = {
        "split": split,
        "parent_held_out_test": bool(
            parent.get("key") and _held_out(parent["dataset"], parent["key"])
        ),
        "recipe": "V4"
        if "V4" in dataset
        else "V2"
        if "V2" in dataset
        else "V3"
        if "V3" in dataset
        else "probe"
        if "rps_" in dataset
        else "V1",
        **rec.get("source_detail", {}),
        **{
            k: meta[k]
            for k in (
                "input_snr",
                "speech_source",
                "source_type",
                "noise_source",
                "noise_start_time",
                "start_time",
                "source_id",
                "probe_crop_s",
            )
            if k in meta
        },
    }
    rec["roles"] = {
        name: role
        for name, role in (
            ("mixture", "model input: drone noise (+ digital target unless real_valid)"),
            ("noise", "drone-noise stem as mixed"),
            ("vocals", "digital target stem (speech or white noise)"),
        )
        if name in info.get("fields", ())
    }
    rec["wind_role"] = _drone_stem(info)
    rec.setdefault("tach_role", rec["wind_role"])
    return rec


def _dregon_lm_real(dataset: str, key: str, meta: dict, info: dict) -> dict:
    rid_raw = meta.get("recording_id")
    if not rid_raw:
        raise ValueError(f"{dataset}/{key}: no _meta row merged (summary_rows)")
    pds, rid = _flight_parent(rid_raw)
    rec = _from_parent(
        pds, rid, window=True, mixed=False, how="real_valid raw 8-ch clip, no mixing"
    )
    ref = _V4
    if dataset.endswith("-nospeech"):
        ref = f"derivations.SPECS dregon_lm_subset + {_CAT} § noise-only twin"
        rec["source_detail"] = {
            "copy_of": {"dataset": "DREGON-LM-V4-michaels-valid-full", "key": meta.get("source_id")}
        }
    if pds == "DREGON-frames":
        kind, safe = "command", True
        why = (
            "DREGON audio unchanged since the V4 cut; start_time = seconds from the parent audio start; "
            "44.1 -> 16 kHz resample of the same 8 mics"
        )
    else:
        kind, safe = "measured", False
        why = (
            "cut on Michael's pre-2026-07-31 alignment: the frozen FLY124 clips sit 86 ms later than the "
            f"current time base ({_CAT} § rp-real-valid-v1); exact offset not established"
        )
    _bind(
        rec,
        meta.get("start_time"),
        meta.get("duration", info.get("duration_s")),
        channels=None,
        safe=safe,
        reason=why,
    )
    _supplied(
        rec,
        ref,
        info,
        kind,
        "rps.npy = DREGON motors_command cut with the clip (V4 parents were cut from motors_command)"
        if kind == "command"
        else "rps.npy = DJI motor speed cut with the clip, pre-2026-07-31 calibration",
    )
    rec["evidence"]["sound_content"] += " — real_valid: mixture IS the raw recording, nothing mixed"
    return rec


def _dregon_lm_synth(dataset: str, key: str, meta: dict, info: dict) -> dict:
    src = meta.get("noise_source")
    if not src:
        raise ValueError(f"{dataset}/{key}: no _meta row merged (summary_rows)")
    v4 = "V4" in dataset
    ref = _V4 if v4 else _HIST
    combo = re.fullmatch(r"motor_combo_(\[.*\])", src)
    if combo:
        return _motor_combo(combo[1])
    pds, rid = _flight_parent(src)
    chans = _channel(src)
    mono = not v4
    rec = _from_parent(
        pds,
        rid,
        window=True,
        mixed=True,
        how="cut noise window"
        + (f", channel {chans[0]}" if chans else ", mono" if mono else ", 8 ch"),
    )
    t0 = meta.get("noise_start_time")
    if v4 and pds == "DREGON-frames":
        safe, why = (
            True,
            (
                "V4 cut from the published DREGON frames; noise_start_time = seconds from the parent "
                "audio start; DREGON audio unchanged; 16 kHz resample of all 8 mics"
            ),
        )
    elif v4:
        safe, why = (
            False,
            (
                "Michael's flight cut on the pre-2026-07-31 alignment; offset on the current time "
                "base not established"
            ),
        )
    elif chans is None:
        safe, why = (
            False,
            (
                "historical mono recipe (deleted creation CLI, legacy loader): channel and time base "
                "not recorded"
            ),
        )
    else:
        safe, why = False, "historical per-channel recipe (legacy loader): time base not recorded"
    if t0 is None:
        why = "no noise offset recorded in this recipe's metadata"
    _bind(rec, t0, info.get("duration_s"), channels=chans, safe=safe, reason=why)
    kind = "command" if v4 and pds == "DREGON-frames" else "measured" if v4 else None
    _supplied(
        rec,
        ref,
        info,
        kind,
        "rps.npy cut from the parent telemetry with the noise window"
        + ("" if kind else " (which DREGON track, measured or command, is not recorded)"),
    )
    rec["evidence"]["sound_content"] += (
        "; target = LibriSpeech speech or white noise (source_white_noise_prob)"
    )
    return rec


def _motor_combo(spec: str) -> dict:
    """V2 motor combo: digital sum of separate DREGON single-motor bench runs."""
    pairs = [tuple(int(v) for v in p) for p in re.findall(r"\((\d),\s*(\d+)\)", spec)]
    parents = [
        _from_parent(
            "DREGON-frames",
            f"motor_Motor{m}_{s}",
            window=True,
            mixed=False,
            how="single-motor bench run, digitally summed",
        )
        for m, s in pairs
    ]
    ref = f"{_HIST}; V2 noise_source motor_combo_{spec}"
    rec = _merge(parents, ref)
    n = len(pairs)
    rotors = "all" if n == 4 else "single" if n == 1 else "subset"
    _set(
        rec,
        ref,
        rotor_activity=(
            rotors,
            f"{n} of 4 DREGON motors, each recorded alone on the bench, summed digitally "
            "(not a simultaneous take)",
        ),
        sound_content=(
            "digital_mix",
            "digital sum of separate single-motor recordings + digital speech target",
        ),
        speed_supplied=("yes", "per-motor speeds = DREGON MotorK_S publisher speed S in turn/s"),
    )
    rec.update(
        n_rotors=4,
        n_active=n,
        supplied_kind="nominal",
        nominal_rps=[next((float(s) for m, s in pairs if m == k), 0.0) for k in (1, 2, 3, 4)],
    )
    rec["parent"] = {
        "components": [p["parent"] for p in parents],
        "binding": "none",
        "binding_reason": "digital composition of several parents; no single parent window",
    }
    rec["source_detail"] = {
        "motor_combo": pairs,
        "composition": "each motor captured alone on the DREGON bench; the rates come from "
        "separate captures summed digitally, not simultaneous physical motors",
    }
    rec["tach_role"] = "noise"
    return rec


# ── rp-real panels ──────────────────────────────────────────────────────────


def _rp_real(dataset: str, key: str, meta: dict, info: dict) -> dict:
    rid_raw = meta.get("recording_id")
    if not rid_raw:
        raise ValueError(f"{dataset}/{key}: no meta.json recording_id")
    pds, rid = _flight_parent(rid_raw)
    stype = meta.get("source_type")
    mixed = stype == "mixed"
    rec = _from_parent(
        pds,
        rid,
        window=True,
        mixed=mixed,
        how=f"8 s {'+ dev-clean speech ' if mixed else ''}clip at {meta.get('t_start_s')} s, 16 kHz",
    )
    if mixed:
        rec["evidence"]["sound_content"] += (
            f" (LibriSpeech dev-clean at {meta.get('snr_db')} dB, one utterance per channel)"
        )
    _bind(
        rec,
        meta.get("t_start_s"),
        meta.get("duration_s"),
        channels=None,
        safe=True,
        reason="derived from today's pinned parent; t_start_s = seconds from the parent audio start; "
        "same mics resampled to 16 kHz",
    )
    label = meta.get("label_key")
    _supplied(
        rec,
        _RP,
        info,
        "measured",
        f"rps.npy = parent {label} track cut with the clip"
        + (" (calibrated DJI motor speed)" if label == "rps" else ""),
    )
    rec["source_detail"] = {
        "split": _split_of(dataset),
        "held_out_test": dataset == "rp-real-test-v1",
        "parent_held_out_test": _held_out(pds, rid),
        **{
            k: meta.get(k)
            for k in ("clip_index", "source_type", "snr_db", "t_start_s", "label_key", "dataset")
        },
    }
    rec["wind_role"] = rec["tach_role"] = _drone_stem(info)
    return rec


# ── DN-LM (Al-Emadi yes_drone + LibriSpeech) ────────────────────────────────


def _alemadi_kind(stem: str) -> str:
    if stem.startswith("mixed_"):
        return "augmented"
    if re.match(r"(B_S\d+_D\d+_\d+-bebop|Membo_\d+_\d+-membo|extra_membo_D\d+)", stem):
        return "original"
    return "unknown"


def _alemadi(stem: str) -> dict:
    """One Al-Emadi yes_drone clip (Parrot Bebop / Mambo, smartphone, indoors)."""
    rec = _new("recording")
    kind = _alemadi_kind(stem)
    _set(
        rec,
        _ALEMADI,
        rig_mount=("free_flight", "drones 'flying and hovering in a quiet indoor environment'"),
        propulsion=("propellers", "propeller sound of flying Parrot Bebop / Mambo"),
        rotor_activity=("all", "airborne Parrot Bebop / Mambo quadcopters"),
        relative_motion=(
            "unknown",
            "smartphone mic 1-20 m from a drone that was 'flying and hovering'; "
            "per-segment flight state not labelled",
        ),
        speed_supplied=("no", "no rotor speed published"),
        speed_regime=("unknown", "flying/hovering takes cut into 1 s segments; no speed data"),
        environment=("indoors", "quiet indoor environment, altitudes 1-4 m"),
        blade_count=("unknown", "propeller variant not stated by the release"),
    )
    rec["n_rotors"] = 4
    rec["n_active"] = 4
    if kind == "augmented":
        _set(
            rec,
            _ALEMADI,
            sound_content=(
                "digital_mix",
                "'mixed_' clip: ESC-50 / Speech Commands noise "
                "overlaid at -32..-3 dB (augmented half)",
            ),
        )
        rec["provenance_kind"] = "derivative"
    elif kind == "original":
        _set(
            rec,
            _ALEMADI,
            sound_content=(
                "drone_only",
                "original smartphone clip (bebop 331 / mambo 275 + 56 "
                "'extra_membo' = 331 originals per Table 1)",
            ),
        )
    else:
        _set(
            rec,
            _ALEMADI,
            sound_content=("unknown", "clip family not identified from the file name"),
        )
    rec["source_detail"] = {
        "drone": "Parrot Bebop"
        if "bebop" in stem.lower()
        else "Parrot Mambo"
        if "membo" in stem.lower()
        else None,
        "clip_kind": kind,
    }
    return rec


def _dn_lm(dataset: str, key: str, meta: dict, info: dict) -> dict:
    src = meta.get("noise_source")
    if not src:
        raise ValueError(f"{dataset}/{key}: no meta.json noise_source")
    stem = src.removesuffix(".wav")
    base = _alemadi(stem)
    rec = _new("derivative")
    for facet in _TEN:
        if facet in base:
            rec[facet] = base[facet]
            rec["evidence"][facet] = (
                f"parent drone_audio yes_drone/{stem}: {base['evidence'][facet]}"
            )
    rec.update(n_rotors=4, n_active=4)
    _set(
        rec,
        _DNLM,
        sound_content=(
            "digital_mix",
            f"LibriSpeech speech at {meta.get('input_snr')} dB with inverse-distance "
            f"attenuation over the drone clip ({base['sound_content']})",
        ),
        speed_supplied=("no", "DN-LM has no rps field"),
    )
    rec["parent"] = {"dataset": "drone_audio", "key": f"Binary_Drone_Audio/yes_drone/{stem}"}
    rec["source_detail"] = {
        "split": _split_of(dataset),
        "speech_distance_m": meta.get("speech_distance"),
        "input_snr": meta.get("input_snr"),
        "clip_kind": _alemadi_kind(stem),
    }
    rec["quality"] = (
        "1 s clip: below the 1.5 s tachometry minimum; no inheritance (parent is a 1 s clip too)"
    )
    rec["wind_role"] = rec["tach_role"] = _drone_stem(info)
    _too_short(rec, info, 1.0, "1 s Al-Emadi drone clip under digital speech")
    return rec


# ── SE-valid ────────────────────────────────────────────────────────────────

#: Noise source of each ``drone_snr*_NNNN`` clip (index = NNNN) from the waveform audit
#: (audit.json, 2026-11-25; identical for SE-valid-drone and the drone category of
#: SE-valid-harmonic): R real flight (DREGON room 1 nosource or FLY124, by
#: elimination), S DroneAudioSet source-only loudspeaker speech (no drone), U tiled
#: drone_audio ``unknown`` (no drone).
_SE_DRONE = (
    "SSRSRSRUSUSSSSRURSRURSRSRRUURSURRRUSURSUUURRRRRSURRURURRUSSURRRSSRUSRRRSURSSRRURRUSRRSRURSRRRRURSURRRSSR"
    "URUSSURURRRSRRRURRUURRSSURRRRSURRRRRRRSRUURURRRSURRRURRUURSRRRSRRURUSRRRRRSURRSRUSURRRRRUSSURRSURRRSRRU"
    "RSSRRURRUURUSUSSRRUURSSSRRRSUUURRSSSSSRSRUURSSURSSSURRUURUSUSSSRSRUSSRURSRUSRRURSRSUSUSSURRRRSSURRSRUUSS"
    "USRURRSSSUUSSRRURUSUSRSRSURRUURRRSSSUSR"
)
assert len(_SE_DRONE) == 350

_SE_NON_DRONE = {
    "mimii": "MIMII industrial machine (fan/pump/slider/valve) + LibriSpeech",
    "mimii_dg": "MIMII-DG machine sound + LibriSpeech",
    "aircraft": "AeroSonicDB aircraft flyover/ambient + LibriSpeech",
    "motors": "HUSTmotor / KAIST-rotating-acoustic bench motor + LibriSpeech",
    "horns": "HornBase car horn / traffic + LibriSpeech",
}
_AVQ_EGO = {
    "avq_ego": ("S1_seq1", "S1_seq2", "S1_seq3", "S2_seq1", "S2_seq2"),
    "avq_ego_s1": ("S1_seq1", "S1_seq2", "S1_seq3"),
    "avq_ego_s2": ("S2_seq1", "S2_seq2"),
}


def _se_valid(dataset: str, key: str, meta: dict, info: dict) -> dict:
    m = re.fullmatch(r"(\w+?)_snr([+-]\d\d)_(\d{4})", key)
    if m is None:
        raise ValueError(f"{dataset}/{key}: unexpected SE-valid key")
    cat, snr, idx = m[1], int(m[2]), int(m[3])
    ref = f"{_SE}; category {cat}, input SNR {snr} dB"
    rec = _new("derivative")
    if cat in _SE_NON_DRONE:
        _no_drone(
            rec,
            ref,
            f"digital mixture: {_SE_NON_DRONE[cat]} — no drone",
            environment=("unknown", "noise corpus mixes several acquisition settings"),
        )
        rec["provenance_kind"] = "non_drone"
    elif cat == "drone":
        code = _SE_DRONE[idx]
        if code == "R":
            rec = _merge(
                [
                    _from_parent(
                        "DREGON-frames",
                        "free-flight_nosource_room1",
                        window=True,
                        mixed=True,
                        how="2 s noise crop",
                    ),
                    _from_parent(
                        "michaels-frames", "FLY124", window=True, mixed=True, how="2 s noise crop"
                    ),
                ],
                f"{ref}; clip attributed to DREGON room1 or FLY124 by elimination (audit)",
            )
        elif code == "S":
            _no_drone(
                rec,
                ref,
                "noise = DroneAudioSet source-only loudspeaker speech (no drone), exact "
                "16 kHz sub-sequence match; + LibriSpeech target",
                environment=("indoors", "DroneAudioSet source-only take in a university room"),
            )
            rec["provenance_kind"] = "control"
        else:
            _no_drone(
                rec,
                ref,
                "noise = tiled drone_audio 'unknown' clip (ESC-50 / white noise / silence), "
                "exact repetition at the clip length; + LibriSpeech target",
                environment=("unknown", "ESC-50/Speech Commands/silence negatives"),
            )
            rec["provenance_kind"] = "control"
        rec.setdefault("source_detail", {})["audit_class"] = code
    elif cat in _AVQ_EGO:
        rec = _merge(
            [
                _from_parent("AVQ", s, window=True, mixed=True, how="2 s ch0 crop via AVQ-egonoise")
                for s in _AVQ_EGO[cat]
            ],
            f"{ref}; noise = AVQ-egonoise crop of {'/'.join(_AVQ_EGO[cat])}",
        )
    else:
        raise ValueError(f"{dataset}/{key}: unknown SE-valid category {cat!r}")
    if rec.get("sound_content") != "no_drone":
        _set(rec, ref, speed_supplied=("no", "SE-valid stores mixture/target only, no rps"))
    rec["wind_role"] = rec["tach_role"] = "mixture"
    rec.setdefault("source_detail", {}).update(category=cat, input_snr_db=snr, split="valid")
    rec["quality"] = (
        "2 s crops at unrecorded offsets: no inheritance; parents are estimated natively"
    )
    return rec


# ── whole-recording copies ──────────────────────────────────────────────────


def _beatvk(dataset: str, key: str, meta: dict, info: dict) -> dict:
    pds = meta.get("source_dataset") or (
        "michaels-frames" if key.startswith("FLY") else "DREGON-frames"
    )
    rec = _from_parent(pds, key, window=False, mixed=False, how="native-rate 8-ch full-audio copy")
    span = (meta.get("spans") or {}).get("audio") or [0.0, info.get("duration_s")]
    _bind(
        rec,
        span[0],
        (span[1] - span[0]) if span[1] is not None else None,
        channels=None,
        safe=True,
        exact=True,
        verify="pcm_equal",
        reason=f"same 44.1 kHz rate, 8 mics, duration and audio anchor as the current parent; source "
        f"version {str(meta.get('source_version'))[:12]} predates the pin, so the runner must "
        "confirm decoded PCM equality before projecting",
    )
    stale = pds == "michaels-frames"
    _supplied(
        rec,
        _BEAT,
        info,
        "measured",
        f"rps_raw = raw {meta.get('source_rps_entry')} telemetry"
        + (" from Michael's pre-2026-07-31 (uncalibrated, stale) build" if stale else ""),
    )
    rec["source_detail"] = {
        "split": "valid",
        "source_version": meta.get("source_version"),
        "windows_s": meta.get("window_s"),
        "stale_telemetry": stale,
    }
    rec["tach_role"] = rec["wind_role"] = "audio"
    return rec


def _decomp(dataset: str, key: str, meta: dict, info: dict) -> dict:
    spec = str(meta.get("source_spec", ""))
    pds = "michaels-frames" if "michaels" in spec else "DREGON-frames"
    rec = _from_parent(
        pds, key, window=False, mixed=False, how="VK decomposition (envelopes/residual)"
    )
    rec["provenance_kind"] = "derivative"
    variant = meta.get("label_variant")
    _supplied(
        rec,
        _DECOMP,
        info,
        "measured" if variant == "published" else "refined_telemetry",
        f"rps = {meta.get('source_rps_key')} ({variant}) of {spec.split('@')[0].removeprefix('frames:')}"
        f"@{spec.split('@')[-1][:12]}",
    )
    rec["quality"] = "no audio entry: residual/envelope analysis product of the parent flight"
    rec["source_detail"] = {"source_spec": spec, "label_variant": variant}
    return rec


# ── KAIST drone fault / ICSV31 ──────────────────────────────────────────────

_KAIST_DRONES = {"A": "Holy Stone HS720", "B": "MJX Bugs 12 EIS", "C": "ZLRC SG906 Pro2"}


def _kaist_rig(rec: dict, ref: str, *, bg: str) -> dict:
    _set(
        rec,
        ref,
        rig_mount=(
            "tethered",
            "elastic ropes 1.5 m to the ceiling and 1 m to the floor through rotatable rings",
        ),
        speed_regime=("unknown", "commanded maneuver directions under tether; no speed data"),
        propulsion=(
            "propellers",
            "propeller-driven toy quadcopters; PCk = one prop ~10 % cut, MFk = dented "
            "motor cap — still propeller-driven",
        ),
        rotor_activity=(
            "all",
            "quadcopters maneuvering under tether need all four rotors powered [inference]",
        ),
        relative_motion=("fixed", "both RØDE Wireless GO II transmitters on top of the drone body"),
        speed_supplied=("no", "no rotor speed released"),
        sound_content=(
            "digital_mix",
            f"publisher mixed every window with {bg}; clean drone audio never released",
        ),
        environment=(
            "anechoic",
            "recorded in the KAIST anechoic chamber; the added background is digital",
        ),
        blade_count=(
            "2",
            "observed: Yi et al. arXiv:2304.11708 §2.1 Fig. 1(a-c) top-view photographs show two "
            "blades on every one of the 12 hubs of drones A/B/C; PCk cuts ~10 % off one blade",
        ),
        wind=("unknown", "no wind evidence; downwash on the body-top mic not characterised"),
    )
    rec.update(n_rotors=4, n_active=4, blade_count_basis="observed")
    return rec


def _kaist(dataset: str, key: str, meta: dict, info: dict) -> dict:
    label = meta.get("label") or {}
    extra = meta
    if dataset == "KAIST-drone-fault-frames":
        m = re.search(r"(?:^|/)([ABC])/(train|valid|test)/(mic1=mic2|mic1|mic2)/([^/]+)$", key)
        if m is None:
            raise ValueError(f"{dataset}/{key}: unexpected key")
        drone, split, mics = m[1], m[2], m[3]
        rec = _kaist_rig(_new("derivative"), _KAIST, bg="campus background at SNR 10-15 dB")
        rec["quality"] = "0.5 s window: below the 1.5 s tachometry minimum" + (
            "; label_disputed (name contradicts waveform identity)"
            if label.get("label_disputed")
            else ""
        )
        _too_short(rec, info, 0.5, "0.5 s publisher window of a powered tethered flight")
        detail = {
            "split": split,
            "mics": mics,
            "parent_window": extra.get("parent_window"),
            "window_offset_s_inferred": extra.get("window_offset_s_inferred"),
            "snr_db": extra.get("snr_db"),
            "background": (meta.get("operating") or {}).get("external_source_detail"),
        }
    else:
        m = re.search(r"(?:^|/)(train|eval|test)/(?:train|eval|test)_([ABC])_", key)
        if m is None:
            raise ValueError(f"{dataset}/{key}: unexpected key")
        split, drone = m[1], m[2]
        rec = _kaist_rig(
            _new("derivative"),
            _ICSV,
            bg="campus pond/hill/gate or DEMAND park/square/traffic noise at SNR -5..5 dB",
        )
        rec.update(tach_candidate=True, tach_role="audio")
        rec["quality"] = (
            "2 s window of a KAIST recording (no exact published mapping); ~40 % of eval/test "
            "overlap a train clip in time"
        )
        detail = {
            "split": split,
            "anomaly_flag": label.get("anomaly_flag"),
            "challenge_rule": label.get("challenge_rule"),
            "parent_group_2s": extra.get("parent_group_2s"),
        }
    rec["parent"] = {"source": "Zenodo 7779574 clean per-mic anechoic recordings (never released)"}
    rec["source_detail"] = {
        "drone": drone,
        "platform": _KAIST_DRONES[drone],
        "health": label.get("health"),
        "fault_code": label.get("fault_code"),
        "fault_unit": label.get("fault_unit"),
        "maneuver": (meta.get("operating") or {}).get("maneuver_direction"),
        **detail,
    }
    rec["wind_role"] = "audio"
    return rec


# ── DADS aggregate ──────────────────────────────────────────────────────────

_KAIST_DIRS = ("B", "CC", "C", "F", "L", "R")  # byte order of '<d>_<dir>_' stems
_KAIST_FAULTS = ("MF1", "MF2", "MF3", "MF4", "N", "PC1", "PC2", "PC3", "PC4")


def dads_kaist_origin(n: int) -> dict[str, str]:
    """DADS ``drone-<n>`` (1592..163591) -> KAIST mic1 family by the audited ordering rule:
    rig blocks of 54,000, families of 1,000 in byte-sorted stem order, 400 test+valid then
    600 train per family (exact PCM matches at 69 spot checks, rule otherwise inferred)."""
    k = n - 1592
    if not 0 <= k < 162000:
        raise ValueError(f"drone-{n} is outside the KAIST block")
    drone = "ABC"[k // 54000]
    fam, pos = divmod(k % 54000, 1000)
    direction, fault = _KAIST_DIRS[fam // 9], _KAIST_FAULTS[fam % 9]
    return {
        "drone": drone,
        "direction": direction,
        "fault_code": fault,
        "split_group": "test_or_valid" if pos < 400 else "train",
        "mic": "mic1",
    }


def _dads(dataset: str, key: str, meta: dict, info: dict) -> dict:
    m = re.fullmatch(r"\d{6}_(drone|no-drone)-(\d+)", key)
    if m is None:
        raise ValueError(f"{dataset}/{key}: unexpected DADS key")
    cls, n = m[1], int(m[2])
    rec = _new("derivative")
    if cls == "no-drone":
        _no_drone(
            rec,
            _DADS,
            "label 0: UrbanSound8K / TUT 2017 / ESC-50 / DNC clips (per-id origin not audited)",
            environment=("unknown", "mixed non-drone corpora"),
        )
        rec["provenance_kind"] = "non_drone"
        return rec
    if n >= 1592:
        o = dads_kaist_origin(n)
        rec = _kaist_rig(
            rec, _DADS + "; KAIST rig facts per " + _KAIST, bg="campus background at SNR 10-15 dB"
        )
        rec["parent"] = {
            "dataset": "KAIST-drone-fault-frames",
            "family": f"{o['drone']}_{o['direction']}_{o['fault_code']}",
            "mic": "mic1",
            "split_group": o["split_group"],
            "mapping": "ordering rule; floor(mic1*32768)/32768 PCM equality on checked ids",
        }
        rec["quality"] = "0.5 s 16-bit copy of a KAIST mic1 mixture; split across KAIST splits"
        _too_short(rec, info, 0.5, "0.5 s KAIST mic1 window (DADS copy) of a powered flight")
        rec["source_detail"] = o
        return rec
    blocks = (
        (1332, "drone_audio", "Al-Emadi yes_drone"),
        (1365, "SPCUP19-frames", "SPCup19 ego-noise"),
        (1370, "DREGON-frames", "DREGON"),
        (1535, "Salford-DroneNoise-frames", "Salford originals"),
        (1591, "AuDroK-frames", "AuDroK Dresden Mic1"),
    )
    pds, label = next((d, lab) for hi, d, lab in blocks if n <= hi)
    rec["parent"] = {
        "dataset": pds,
        "block": label,
        "mapping": "id block audited against source lengths/"
        "waveforms; per-file identity not established",
    }
    _set(rec, _DADS, speed_supplied=("no", "DADS carries no speed; per-file parent unknown"))
    if pds == "drone_audio":
        base = _alemadi("")
        for facet in (
            "rig_mount",
            "propulsion",
            "rotor_activity",
            "relative_motion",
            "environment",
            "blade_count",
            "speed_regime",
        ):
            rec[facet] = base[facet]
            rec["evidence"][facet] = f"{_DADS}; Al-Emadi block: {base['evidence'][facet]}"
        rec.update(n_rotors=4, n_active=4)
        _set(
            rec,
            _DADS,
            sound_content=(
                "unknown",
                "block mixes 662 original and 670 augmented (digital_mix) "
                "clips; per-id kind not established",
            ),
        )
        _too_short(rec, info, 1.0, "1 s Al-Emadi drone clip (DADS copy)")
    elif pds == "Salford-DroneNoise-frames":
        _set(
            rec,
            _DADS,
            environment=(
                "outdoors",
                f"{label}: every Salford take is from the outdoor Edzell campaign",
            ),
        )
        rec["quality"] = (
            f"{label}: block may mix hovers, flyovers and calibration crops (not established); "
            "rig/propulsion/motion stay unknown per clip"
        )
    elif pds == "AuDroK-frames":
        _set(
            rec,
            _DADS,
            environment=("outdoors", f"{label}: Fraunhofer IVI Dresden field campaign"),
            rig_mount=(
                "free_flight",
                "the 56 ids equal the 56 AuDroK Dresden flight takes (Mic1); "
                "the Dresden campaign has flights only",
            ),
            propulsion=("propellers", "airborne rotorcraft of the Dresden flights"),
        )
        rec["quality"] = (
            f"{label}: flight phases (hover/translation) mixed within takes; motion stays unknown"
        )
    rec["quality"] = f"{label} copy; use the native release for estimation" + (
        "; " + rec["quality"] if "quality" in rec else ""
    )
    return rec


# ── Al-Emadi raw tree ───────────────────────────────────────────────────────

_SC_BACKGROUND = (
    "doing_the_dishes",
    "exercise_bike",
    "running_tap",
    "white_noise",
    "pink_noise",
    "dude_miaowing",
)


def _drone_audio(dataset: str, key: str, meta: dict, info: dict) -> dict:
    if not info.get("has_audio"):
        rec = _new("alias")
        rec["quality"] = "non-audio file of the release"
        return rec
    m = re.fullmatch(
        r"(Binary_Drone_Audio|Multiclass_Drone_Audio)/(yes_drone|bebop_1|membo_1|unknown)/(.+)", key
    )
    if m is None:
        raise ValueError(f"{dataset}/{key}: unexpected key")
    tree, cls, stem = m[1], m[2], m[3]
    if cls == "unknown":
        rec = _no_drone(
            _new("control"),
            _ALEMADI,
            "'unknown' class: "
            + (
                "silence clip"
                if stem.startswith("silence")
                else "Speech Commands background noise"
                if stem.startswith(_SC_BACKGROUND)
                else "ESC-50 clip segment"
                if re.fullmatch(r"\d+-\d+-[A-Z]-\d+", stem)
                else "non-drone negative"
            ),
            environment=("unknown", "third-party negatives (ESC-50, Speech Commands, silence)"),
        )
    else:
        rec = _alemadi(stem)
        rec["quality"] = "1 s clip: below the 1.5 s tachometry minimum"
        _too_short(rec, info, 1.0, "1 s segment of a flying/hovering Parrot drone")
    if tree == "Multiclass_Drone_Audio":
        rec["provenance_kind"] = "alias"
        twin = "yes_drone" if cls != "unknown" else "unknown"
        rec["parent"] = {
            "dataset": dataset,
            "key": f"Binary_Drone_Audio/{twin}/{stem}",
            "relation": "same file name and byte length in the Binary tree",
        }
    return rec


# ── DroneAudioSet ───────────────────────────────────────────────────────────

_DAS_DRONES = {
    "drone1": ("DJI F450 (D_large)", "9.4 x 5.0 in"),
    "drone2": ("DJI F330 (D_small)", "8 x 4.5 in"),
}
_DAS_MICS = {
    "mic1_soundskrit": "M_center single mic above the drone",
    "mic2_8array-down": "M_down 8-ch array below the drone",
    "mic3_8array-up": "M_up 8-ch array above the drone",
}


def droneaudioset_path_facts(path: str) -> dict:
    """Ten-facet record of one DroneAudioSet publisher file (HF ``file_path``)."""
    p = re.sub(r"^\d{6}_", "", path).replace("__", "/").split("/")
    rec = _new("recording")
    subset = p[0]
    if subset == "ground-truth-audio":
        _no_drone(
            rec,
            _DAS,
            "publisher source file played through the loudspeaker; not a capture",
            environment=("not_applicable", "digital source file"),
        )
        rec["provenance_kind"] = "control"
        _set(rec, _DAS, wind=("not_applicable", "digital source file"))
        return rec
    settings = {
        m[1]: m[2]
        for t in p[1:-1]
        if (m := re.fullmatch(r"(mic-dist|throttle|speaker-loudness|speaker-dist)-(.+)", t))
    }
    last = p[-1]
    fm = re.fullmatch(r"(mic\d_[\w-]+?)-(File\d|silence)\.wav", last)
    if fm is None or fm[1] not in _DAS_MICS:
        raise ValueError(f"unparseable DroneAudioSet path {path!r}")
    mic, segment = fm[1], fm[2]
    room = next((t for t in p if re.fullmatch(r"room\d", t)), None)
    env = ("indoors", f"{room or 'one of three university rooms'} (paper §3.4)")
    detail: dict[str, Any] = {
        "subset": subset,
        "mic": mic,
        "mic_role": _DAS_MICS[mic],
        "segment": segment,
        "room": room,
        **settings,
    }
    if subset == "source-only-recordings":
        reason = (
            "loudspeaker playback of source " + segment
            if segment != "silence"
            else "silent segment: room ambience only"
        )
        _no_drone(rec, _DAS, f"source-only take, no drone on the frame: {reason}", environment=env)
        rec["provenance_kind"] = "control"
        rec["source_detail"] = detail
        return rec
    drone = next((t for t in p if re.fullmatch(r"drone[12](-only)?", t)), "").removesuffix("-only")
    throttle = next((t.removeprefix("throttle-") for t in p if t.startswith("throttle-")), None)
    model, prop = _DAS_DRONES[drone]
    _set(
        rec,
        _DAS,
        rig_mount=(
            "fixed_stand",
            "drone 'firmly affixed' to a fixed aluminium frame 1.5 m above ground to emulate hover",
        ),
        speed_regime=(
            "unknown",
            f"commanded '{throttle}' throttle; no measured speed — steady commanded take",
        ),
        propulsion=("propellers", f"{model} with {prop} propellers (Table 4)"),
        rotor_activity=(
            "all",
            "quadcopter on the frame at a throttle setting (four motors with propellers)",
        ),
        relative_motion=(
            "fixed",
            "arrays and mic mounted on the same frame 25/50 cm from the drone",
        ),
        speed_supplied=("no", "only 'low'/'high' throttle labels (SPL in Table 5), no rpm"),
        environment=env,
        blade_count=("2", f"Fig. 1(b) photo shows two-blade props on both drones; {prop} nominal"),
    )
    rec.update(
        n_rotors=4, n_active=4, tach_candidate=True, tach_role="audio", blade_count_basis="observed"
    )
    _set(
        rec,
        _DAS,
        wind=("unknown", "left to the PCM assessor; M_down exposure is a group-level finding"),
    )
    if mic == "mic2_8array-down":
        detail["downwash_exposure"] = (
            "documented: M_down below the propellers shows low-frequency wind noise and "
            "the lowest median SNR across conditions (§3.5, Fig. 8); not a per-take result"
        )
    if subset == "drone-only-recordings":
        _set(rec, _DAS, sound_content=("drone_only", "drone-only subset (no loudspeaker source)"))
    elif segment == "silence":
        _set(
            rec,
            _DAS,
            sound_content=(
                "unknown",
                "drone-with-source take, 'silence' segment: speaker state not documented",
            ),
        )
    else:
        _set(
            rec,
            _DAS,
            sound_content=(
                "co_recorded",
                f"loudspeaker played source {segment} (human vocal / "
                "non-vocal / ambient sounds) during the take",
            ),
        )
    detail.update(
        drone=drone,
        model=model,
        throttle=throttle,
        drone_identity="drone1 = F450 / drone2 = F330: drone1 alone at 90 dB and room2, drone2 alone in "
        "room3, matching Table 2",
    )
    rec["source_detail"] = detail
    rec["wind_role"] = "audio"
    return rec


def _droneaudioset(dataset: str, key: str, meta: dict, info: dict) -> dict:
    return droneaudioset_path_facts(key)


# ── survey WAVs (zenodo_drone_noises) ───────────────────────────────────────

#: n116-n120 = the five AVQ ego-noise sequences, n121/n122 = AS (survey §VI). Matched to AVQ
#: keys by duration: n118 = 40 s = S1_seq3; n116/n117 = 120 s = S1_seq1/S1_seq2 (order not
#: established); n119/n120 = 215.2/218.7 s vs S2_seq1/S2_seq2 = 210.7/214.2 s (unresolved).
_SURVEY_AVQ = {
    "n116": ("S1_seq1", "S1_seq2"),
    "n117": ("S1_seq1", "S1_seq2"),
    "n118": ("S1_seq3",),
    "n119": ("S2_seq1", "S2_seq2"),
    "n120": ("S2_seq1", "S2_seq2"),
}


def _zenodo(dataset: str, key: str, meta: dict, info: dict) -> dict:
    m = re.fullmatch(r"raw/noises-(train|test)-drones/(n\d+)", key)
    if m is None or not info.get("has_audio"):
        rec = _new("alias")
        rec["quality"] = (
            "ZIP / Arrow / JSON copy of the seven survey WAVs (not separate recordings)"
        )
        rec["parent"] = {"dataset": dataset, "keys": "raw/noises-*-drones/n116..n122"}
        return rec
    split, n = m[1], m[2]
    if n in _SURVEY_AVQ:
        cands = _SURVEY_AVQ[n]
        rec = _merge(
            [
                _from_parent("AVQ", s, window=False, mixed=False, how="first channel, survey copy")
                for s in cands
            ],
            f"{_SURVEY}: n116-n120 are the five AVQ ego-noise sequences; "
            f"{n} duration-matches {'/'.join(cands)}",
        )
        rec["quality"] = (
            "survey states n116-n120 are the AVQ ego-noise sequences; per-file AVQ identity, channel "
            "offset and rate relation are not bound, so this copy is measured on its own"
        )
        rec.update(tach_candidate=True, tach_role="wav")
        _set(rec, _SURVEY, speed_supplied=("no", "survey WAV carries no speed track"))
    else:
        rec = _new("recording")
        _set(
            rec,
            f"{_SURVEY} + {_AS}",
            rig_mount=("fixed_stand", "3DR IRIS quadcopter fixed on a tripod 1.8 m high"),
            propulsion=("propellers", "3DR IRIS motors and propellers running"),
            rotor_activity=(
                "all",
                "quadcopter operated at a fraction of hover power on the tripod",
            ),
            relative_motion=("fixed", "8-mic circular array on top of the MAV body"),
            speed_supplied=("no", "only % of hovering power; no rpm"),
            speed_regime=(
                "unknown",
                "AS ego-noise segments: constant or randomly varying motor speed",
            ),
            sound_content=("drone_only", "survey uses n121/n122 as pure ego-noise test sequences"),
            environment=("unknown", "acquisition room not stated"),
            blade_count=("unknown", "IRIS propeller not stated in the paper"),
        )
        rec.update(n_rotors=4, n_active=4, tach_candidate=True, tach_role="wav")
        rec["parent"] = {"source": "AS ego-noise (Wang & Cavallaro 2018); no native copy in dload"}
    rec["source_detail"] = {"survey_split": split, "survey_id": n}
    rec["wind_role"] = "wav"
    return rec


# ── simulations ─────────────────────────────────────────────────────────────


def _egonoise(dataset: str, key: str, meta: dict, info: dict) -> dict:
    rec = _new("simulation")
    rps = meta.get("prescribed_rps", meta.get("rps"))
    blades = meta.get("n_blades")
    _set(
        rec,
        _EGO,
        rig_mount=(
            "fixed_stand",
            "prescribed_omega bypasses the flight sim: vehicle held static at the reference point",
        ),
        speed_regime=("constant", f"all rotors at exactly {rps} rev/s (rps_prescribed)"),
        propulsion=("propellers", "BEMT/FW-H rotor blades (DJI 9450 shape scaled)"),
        rotor_activity=("all", "all four rotors spin at the prescribed speed"),
        relative_motion=(
            "fixed",
            f"{meta.get('n_mics')} virtual mics on shells {meta.get('sphere_radii_m')} m "
            "around the static vehicle",
        ),
        speed_supplied=("yes", f"prescribed {rps} rev/s"),
        sound_content=("drone_only", "tonal + broadband ego-noise of one simulated vehicle"),
        environment=("simulated", "free-field auralisation, no ground reflection or absorption"),
        blade_count=(str(blades) if blades else "unknown", "n_blades in the case meta"),
        wind=(
            "not_significant",
            "gust_w20 = 0 and no microphone-flow source in the FW-H + BPM model",
        ),
    )
    rec.update(
        n_rotors=4,
        n_active=4,
        supplied_kind="simulated",
        nominal_rps=float(rps) if rps is not None else None,
        tach_candidate=True,
        tach_role="wav",
        wind_role="wav",
    )
    rec["source_detail"] = {
        k: meta.get(k) for k in ("drone", "label", "bpf_hz", "seed", "duration", "n_mics")
    }
    rec["quality"] = "synthetic validation case: prescribed speed is ground truth, not a recording"
    return rec


def _jasa(dataset: str, key: str, meta: dict, info: dict) -> dict:
    rec = _new("simulation")
    bpf = meta.get("bpf_hz")
    _set(
        rec,
        _JASA,
        rig_mount=(
            "free_flight",
            f"simulated level flyover at {meta.get('speed')} m/s, {meta.get('altitude')} m",
        ),
        speed_regime=(
            "unknown",
            "closed-loop 6-DOF sim trims each rotor speed; histories not published",
        ),
        propulsion=("propellers", "NASA 1-Pax rotors"),
        rotor_activity=("all", "four rotors"),
        relative_motion=("moving", "ground microphone grid under a passing vehicle"),
        speed_supplied=(
            "no",
            "case arrays are audio/tonal/broadband/mics/band_centers only: no time-resolved "
            "rotor speed; bpf_hz is the vehicle's hover reference datum, not this flight",
        ),
        sound_content=("drone_only", "one simulated vehicle"),
        environment=(
            "simulated",
            "free-field CONA auralisation, no ground reflection or absorption",
        ),
        blade_count=("3", "NASA 1-Pax published 4 rotors x 3 blades"),
        wind=("not_significant", f"gust_w20 = {meta.get('gust_w20')}; no microphone-flow source"),
    )
    rec.update(n_rotors=4, n_active=4)
    detail: dict[str, Any] = {
        k: meta.get(k)
        for k in ("speed", "altitude", "heading_deg", "lateral_offset", "seed", "duration")
    }
    detail["hover_reference"] = {
        "bpf_hz": bpf,
        "rpm": 671,
        "basis": "NASA 1-Pax published hover "
        "datum (nasa_1pax.py); the closed-loop sim trims each rotor",
    }
    rec["source_detail"] = detail
    rec["quality"] = "1 s Doppler-shifted flyover: not a tachometry case"
    return rec


# ── dispatch ────────────────────────────────────────────────────────────────


def classify(dataset: str, key: str, meta: dict, info: dict) -> dict:
    """Facts of one exact sample of a derived/aggregate/simulated dataset."""
    if dataset in DREGON_LM_SYNTH or dataset in DREGON_LM_REAL:
        rec = _dregon_lm(dataset, key, meta, info)
    elif dataset in RP_REAL:
        rec = _rp_real(dataset, key, meta, info)
    elif dataset in DN_LM:
        rec = _dn_lm(dataset, key, meta, info)
    elif dataset in SE_VALID:
        rec = _se_valid(dataset, key, meta, info)
    elif dataset == "beatvk-valid-raw":
        rec = _beatvk(dataset, key, meta, info)
    elif dataset.startswith("decomp-frames"):
        rec = _decomp(dataset, key, meta, info)
    elif dataset in KAIST_FAMILY:
        rec = _kaist(dataset, key, meta, info)
    else:
        rec = {
            "drone-detection-samples": _dads,
            "drone_audio": _drone_audio,
            "DroneAudioSet": _droneaudioset,
            "zenodo_drone_noises": _zenodo,
            "drone-egonoise": _egonoise,
            "jasa-flyovers": _jasa,
        }[dataset](dataset, key, meta, info)
    parent = rec.get("parent")
    if parent is not None and "binding" not in parent:
        parent["binding"] = "none"
    return rec
