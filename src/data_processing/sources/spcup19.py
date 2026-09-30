"""SPCUP19 ego-noise source: the IEEE SP Cup 2019 bonus-task packages (DREGON/INRIA).

Ten student teams recorded their own drone's ego-noise with their own
microphones. The packages are heterogeneous — loose .wav (8 teams), one .mat
(ChuMS: 9 hover tests, two unsynchronised recorders), ROS bags (KumamoTech:
16-ch HARK audio + pose) — and their meaning lives in each team's technical
report and supplementary material, not in the files.

That meaning is recorded ONCE, per team, in ``spcup19_meta/<Team>.yaml``
(read from the reports, the supplementary files and, where the report is
silent, measurements on the audio — every value carries its ``source``). The
builder joins every recording it decodes to its annotation by key and REFUSES
a recording without an annotation or an annotation without a recording, so
the published tags cover the dataset exactly.

Each recording Frame holds:

- ``audio`` — ``(mic, time)`` (``(time,)`` for one channel) at the native rate;
- ``mic_pos`` ``(mic, 3)`` / ``rotor_pos`` ``(rotor, 3)`` — metres, in the
  team's body frame (``meta.geometry.frame``), row i of ``mic_pos`` = audio
  channel i; present only where the annotation derives them (DREGON layout);
- KumamoTech only: ``position`` (East/North/up m from take-off) and
  ``attitude`` (quaternion x, y, z, w) tracks on the audio clock;
- ``meta`` — ``system`` (drone, team, n_rotors), ``operating`` (``condition``
  ∈ :data:`CONDITIONS`, ``flight_mode``, ``contains_rotor_noise``,
  ``n_active_rotors``, ``throttle_or_speed``), ``label`` (``external_source``),
  ``geometry`` (config, frame, qualities, channel-map evidence, sources),
  ``annotation`` (description, source, confidence) and format extras.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import tdseries as td

from data_processing.sources._common import audio_frame, meta_frame, read_audio_file, safe_key

#: Team package → dregon.inria.fr download id (dataset page, "Show data...").
_DOWNLOAD_IDS: dict[str, int] = {
    "Idea_ssu": 393,
    "Maverick": 394,
    "Diagonal_Unloading": 395,
    "ChuMS": 396,
    "LEADS_UAV": 397,
    "NSS_Chellamma": 398,
    "KumamoTech": 399,
    "AGH": 400,
    "KU_Leuven": 401,
    "Shout_COOEE": 405,
}
URLS = {
    f"{team}.zip": f"http://dregon.inria.fr/?smd_process_download=1&download_id={i}"
    for team, i in _DOWNLOAD_IDS.items()
}

#: Per-team annotations (one YAML per team; schema in the module docstring).
ANNOTATION_DIR = Path(__file__).with_name("spcup19_meta")

#: ``operating.condition`` vocabulary. ``bench`` = drone/rig fixed, rotors at
#: ~constant speed; ``bench_varying_speed`` = fixed, speed deliberately varied;
#: ``handheld`` = not airborne, rotors running, drone carried/moved by a person;
#: ``flight`` = airborne (``flight_mode`` says how); ``rotors_off`` = no rotor
#: noise at all (calibration, clean-source captures, motors off).
CONDITIONS = ("flight", "bench", "bench_varying_speed", "handheld", "rotors_off")
FLIGHT_MODES = ("hover", "manoeuvre", "mixed", "unknown")
EXTERNAL_SOURCES = (
    "none",
    "speech",
    "chirp",
    "white_noise",
    "music",
    "mixed",
    "other",
    "unknown",
)
CONFIDENCE = ("high", "medium", "low")

_CHUMS_DETAILS = re.compile(r"\s*(\d+)\s*propellers?\b.*?Repeat\s*:\s*(\d+)", re.IGNORECASE)


# ─── annotations ──────────────────────────────────────────────────────────────


def _positions(value: Any, what: str) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or not np.isfinite(arr).all():
        raise ValueError(f"{what} must be a finite (n, 3) list, got shape {arr.shape}")
    return arr


def validate_team(doc: dict[str, Any]) -> None:
    """Raise on any annotation that breaks the vocabulary or its own geometry."""
    team = doc["team"]
    configs = doc.get("geometry") or {}
    n_rotors = int(doc["drone"]["n_rotors"])
    for name, cfg in configs.items():
        _positions(cfg.get("mic_pos"), f"{team}.{name}.mic_pos")
        rotors = _positions(cfg.get("rotor_pos"), f"{team}.{name}.rotor_pos")
        if rotors is not None and rotors.shape[0] != n_rotors:
            raise ValueError(
                f"{team}.{name}: {rotors.shape[0]} rotor_pos rows, drone has {n_rotors}"
            )
    seen: set[str] = set()
    for rec in doc["recordings"]:
        key = rec["key"]
        where = f"{team}/{key}"
        if key in seen or not key.startswith(f"{team}__"):
            raise ValueError(f"{where}: duplicate key or key outside the team prefix")
        seen.add(key)
        cond, mode = rec["condition"], rec.get("flight_mode")
        if cond not in CONDITIONS:
            raise ValueError(f"{where}: condition {cond!r} not in {CONDITIONS}")
        if (cond == "flight") != (mode is not None) or (
            mode is not None and mode not in FLIGHT_MODES
        ):
            raise ValueError(f"{where}: flight_mode {mode!r} is set iff condition is flight")
        n_active, noisy = rec["n_active_rotors"], rec["contains_rotor_noise"]
        if n_active is not None and not (isinstance(n_active, int) and 0 <= n_active <= n_rotors):
            raise ValueError(f"{where}: n_active_rotors {n_active!r} not an int in 0..{n_rotors}")
        if (cond == "rotors_off") != (n_active == 0) or (cond == "rotors_off") == bool(noisy):
            raise ValueError(f"{where}: rotors_off <=> n_active_rotors 0 <=> no rotor noise")
        if rec["external_source"] not in EXTERNAL_SOURCES:
            raise ValueError(f"{where}: external_source {rec['external_source']!r}")
        if rec.get("confidence") not in CONFIDENCE:
            raise ValueError(f"{where}: confidence {rec.get('confidence')!r}")
        if rec.get("geometry") is not None and rec["geometry"] not in configs:
            raise ValueError(f"{where}: unknown geometry {rec['geometry']!r}")


@lru_cache(maxsize=1)
def load_annotations() -> dict[str, dict[str, Any]]:
    """``{team: parsed spcup19_meta/<team>.yaml}``, validated."""
    import yaml

    out: dict[str, dict[str, Any]] = {}
    for path in sorted(ANNOTATION_DIR.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text())
        if doc["team"] != path.stem or doc["team"] not in _DOWNLOAD_IDS:
            raise ValueError(f"{path.name}: team {doc['team']!r} does not name a package")
        validate_team(doc)
        out[doc["team"]] = doc
    if set(out) != set(_DOWNLOAD_IDS):
        raise ValueError(f"annotations missing for {sorted(set(_DOWNLOAD_IDS) - set(out))}")
    return out


def annotations_digest() -> str:
    """sha256 over the annotation files: part of the derivation identity."""
    h = hashlib.sha256()
    for path in sorted(ANNOTATION_DIR.glob("*.yaml")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


# ─── frame assembly ───────────────────────────────────────────────────────────


def _frame(
    doc: dict[str, Any],
    rec: dict[str, Any],
    audio_ct: np.ndarray,
    sr: int,
    *,
    relpath: str,
    tracks: dict[str, td.Series] | None = None,
    extra: dict[str, Any] | None = None,
) -> td.Frame:
    team, key = doc["team"], rec["key"]
    n_ch = int(audio_ct.shape[0])
    cfg_name = rec.get("geometry")
    cfg = (doc.get("geometry") or {}).get(cfg_name) if cfg_name else None
    mic_pos = _positions(cfg.get("mic_pos"), "mic_pos") if cfg else None
    rotor_pos = _positions(cfg.get("rotor_pos"), "rotor_pos") if cfg else None
    if mic_pos is not None and mic_pos.shape[0] != n_ch:
        raise ValueError(
            f"{key}: {n_ch} audio channels, geometry {cfg_name} has {mic_pos.shape[0]} mics"
        )
    geometry: dict[str, Any] = {"config": cfg_name, "frame": doc["frame"]["description"]}
    if cfg:
        for field in (
            "mic_pos_quality",
            "rotor_pos_quality",
            "channel_map_source",
            "source",
            "notes",
            "reported_mic_pos_source",
        ):
            geometry[field] = cfg.get(field)
        if cfg.get("reported_mic_pos") is not None:
            geometry["reported_mic_pos"] = td.wrap(
                np.asarray(cfg["reported_mic_pos"], dtype=np.float64), dims=("mic", None)
            )
    meta = meta_frame(
        key,
        "SPCUP19-egonoise",
        system={
            "category": "drone",
            "team": team,
            "make_model": doc["drone"]["make_model"],
            "n_rotors": int(doc["drone"]["n_rotors"]),
        },
        observation={"n_channels": n_ch, "channels_used": rec.get("channels_used")},
        operating={
            "condition": rec["condition"],
            "flight_mode": rec.get("flight_mode"),
            "contains_rotor_noise": bool(rec["contains_rotor_noise"]),
            "n_active_rotors": rec["n_active_rotors"],
            "throttle_or_speed": rec.get("throttle_or_speed"),
        },
        label={
            "team": team,
            "drone": doc["drone"]["make_model"],
            "external_source": rec["external_source"],
        },
        extra={
            "raw_relpath": relpath,
            "competition": "IEEE SP Cup 2019 ego-noise (bonus task)",
            "team_report": doc["report"],
            "description": " ".join(str(rec["description"]).split()),
            **(extra or {}),
        },
    )
    meta = td.Frame(
        {
            **dict(meta.items()),
            "geometry": td.Frame({k: v for k, v in geometry.items() if v is not None}),
            "annotation": td.Frame(
                {"source": " ".join(str(rec["source"]).split()), "confidence": rec["confidence"]}
            ),
        }
    )
    frame = audio_frame(audio_ct, int(sr), meta, mic_pos=mic_pos)
    if rotor_pos is not None:
        frame = frame.with_entry("rotor_pos", td.wrap(rotor_pos, dims=("rotor", None)))
    for name, series in (tracks or {}).items():
        frame = frame.with_entry(name, series)
    return frame


# ─── readers ──────────────────────────────────────────────────────────────────


def _chums_runs(mat_path: Path) -> Iterator[tuple[str, int, list[np.ndarray], int, dict[str, Any]]]:
    """``(key suffix, test index, per-mic signals, Fs, extras)`` per ``TestResults.Test``.

    The .mat holds ``MicPositions`` and ``Test(1..9)``, each with ``Details``
    ("N propellers. Repeat:R" — N two-blade propellers STACKED on every motor
    of the hovering quad) and ``Data(1..8)`` (``RawTruncatedCalibrated`` Pa at
    ``Fs`` + per-mic ``OASPL``; ``Freq``/``SPL`` are spectra, not audio).
    """
    from scipy.io import loadmat

    d = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    if "TestResults" not in d:
        raise ValueError(f"SPCUP19 ChuMS: {mat_path.name} has no TestResults struct")
    root = np.atleast_1d(d["TestResults"])[0]
    n_mics = np.asarray(root.MicPositions).shape[0]
    for run, test in enumerate(np.atleast_1d(root.Test), start=1):
        details = str(test.Details).strip()
        match = _CHUMS_DETAILS.match(details)
        if match is None:
            raise ValueError(
                f"SPCUP19 ChuMS: Test({run}) Details {details!r} is not 'N propellers. Repeat:R'"
            )
        mics = np.atleast_1d(test.Data)
        rates = {int(round(float(mic.Fs))) for mic in mics}
        if len(rates) != 1 or len(mics) != n_mics:
            raise ValueError(
                f"SPCUP19 ChuMS: Test({run}) has {len(mics)} mics at Fs {sorted(rates)}"
            )
        sigs = [np.asarray(m.RawTruncatedCalibrated, dtype=np.float32).reshape(-1) for m in mics]
        n_props, repeat = int(match.group(1)), int(match.group(2))
        extras = {
            "details": details,
            "n_stacked_propellers": n_props,
            "blades_per_rotor": 2 * n_props,
            "repeat": repeat,
            "audio_unit": "Pa",
            "oaspl_db": [float(m.OASPL) for m in mics],
            "mic_n_samples": [int(s.size) for s in sigs],
        }
        yield f"{n_props}prop_repeat{repeat}", run, sigs, rates.pop(), extras


def _align_recorders(sigs: list[np.ndarray], align: dict[str, Any]) -> np.ndarray:
    """Undo the two-recorder offset (``B[n+K] ≈ A[n]``), then cut to the common length."""
    a, b, k = set(align["group_a"]), set(align["group_b"]), int(align["k_samples"])
    if a & b or a | b != set(range(len(sigs))):
        raise ValueError(
            f"recorder groups {sorted(a)} / {sorted(b)} do not partition {len(sigs)} mics"
        )
    late = b if k > 0 else a
    shifted = [s[abs(k) :] if i in late else s for i, s in enumerate(sigs)]
    n = min(s.size for s in shifted)
    return np.stack([s[:n] for s in shifted])


def _time_cut(audio: np.ndarray, sr: int, rng: list | None) -> tuple[np.ndarray, float]:
    if rng is None:
        return audio, 0.0
    start = int(round(float(rng[0]) * sr))
    stop = audio.shape[1] if rng[1] is None else int(round(float(rng[1]) * sr))
    if not 0 <= start < stop <= audio.shape[1]:
        raise ValueError(f"time_range_s {rng} outside a {audio.shape[1] / sr:.2f} s recording")
    return audio[:, start:stop], start / sr


def _telemetry(info: dict[str, Any], t0: float, t1: float) -> dict[str, td.Series]:
    """KumamoTech pose tracks inside ``[t0, t1)`` s of the bag audio clock, re-zeroed at ``t0``."""
    out: dict[str, td.Series] = {}
    for name, field in (("position", "xyz"), ("attitude", "xyzw")):
        if name not in info:
            continue
        t = np.asarray(info[name]["t"], dtype=np.float64)
        keep = (t >= t0) & (t < t1)
        if keep.sum() >= 2:
            vals = np.asarray(info[name][field], dtype=np.float64)[keep]
            out[name] = td.events(t[keep] - t0, vals.T.astype(np.float32), dims=(None, "time"))
    return out


def _recordings(team: str, team_dir: Path, raw_dir: Path, recs: dict[str, dict]):
    """``(key, audio (C, T), sr, relpath, tracks, extras)`` for every recording of one package."""
    for wav in sorted(team_dir.rglob("*.wav")):
        audio, sr = read_audio_file(wav)
        rel = wav.relative_to(team_dir).with_suffix("")
        yield safe_key(f"{team}__{rel}"), audio, sr, str(wav.relative_to(raw_dir)), {}, {}
    for mat in sorted(team_dir.rglob("*.mat")):
        if team != "ChuMS":
            raise ValueError(f"SPCUP19 {team}: no parser for {mat.relative_to(raw_dir)}")
        stem = str(mat.relative_to(team_dir).with_suffix(""))
        for suffix, run, sigs, sr, extras in _chums_runs(mat):
            key = safe_key(f"{team}__{stem}/{suffix}")
            align = recs.get(key, {}).get("recorder_align")
            if align is None:
                raise ValueError(f"{key}: ChuMS runs need recorder_align in the annotation")
            audio = _align_recorders(sigs, align)
            rel = f"{mat.relative_to(raw_dir)}:TestResults/Test({run})"
            yield key, audio, sr, rel, {}, {**extras, "recorder_align": dict(align)}
    bags = sorted(team_dir.rglob("*.bag"))
    if team == "KumamoTech":
        from data_processing.sources.spcup19_bags import read_bag_audio

        for bag in bags:
            audio, sr, info = read_bag_audio(bag)
            rel = str(bag.relative_to(raw_dir))
            for key, rec in recs.items():
                if Path(str(rec["raw"])).name != bag.name:
                    continue
                cut, t0 = _time_cut(audio, sr, rec.get("time_range_s"))
                tracks = _telemetry(info, t0, t0 + cut.shape[1] / sr)
                extras = {
                    "time_range_s": [t0, t0 + cut.shape[1] / sr],
                    "sound_sources_en_m": {
                        k: list(v) for k, v in info.get("sound_sources", {}).items()
                    },
                    "hark_frames": {"length": info["frame_length"], "advance": info["advance"]},
                    "audio_unit": "int16 full scale (RASP-ZX level uncalibrated)",
                }
                yield key, cut, sr, rel, tracks, extras
    # Shout_COOEE's bag logs an unpublished dynamic flight (no audio): not a recording.


def build(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """Every annotated recording of the 10 packages under ``raw_dir/<team>/``.

    Keys come from the team-relative path (``AGH/static clean/1.wav`` →
    ``AGH__static_clean__1``), ChuMS's per-test labels, and the annotation's
    own keys for bag segments. A decoded recording without an annotation, or
    an annotation whose recording is absent, raises."""
    docs = load_annotations()
    for team in _DOWNLOAD_IDS:
        team_dir = Path(raw_dir) / team
        doc = docs[team]
        recs = {r["key"]: r for r in doc["recordings"]}
        if not team_dir.exists():
            raise FileNotFoundError(f"SPCUP19 package {team} missing under {raw_dir}")
        produced: set[str] = set()
        for key, audio, sr, rel, tracks, extras in _recordings(team, team_dir, Path(raw_dir), recs):
            if key not in recs:
                raise ValueError(f"SPCUP19 {team}: recording {key} ({rel}) has no annotation")
            if key in produced:
                raise ValueError(f"SPCUP19 {team}: duplicate recording key {key}")
            produced.add(key)
            yield key, _frame(doc, recs[key], audio, sr, relpath=rel, tracks=tracks, extra=extras)
        missing = set(recs) - produced
        if missing:
            raise ValueError(f"SPCUP19 {team}: annotated but not found: {sorted(missing)}")


PROVENANCE = {
    "source_url": "https://dregon.inria.fr/datasets/the-spcup19-egonoise-dataset/",
    "doi": "10.48550/arXiv.1907.04655",
    "license": "free for personal, educational and academic use only",
    "citation": "Deleforge et al., Audio-Based Search and Rescue With a Drone: IEEE SP Cup 2019 (IEEE SPM 36(5), 2019).",
    "collection_method": "10 student teams recorded their own drone's ego-noise (bonus task); per-team technical reports at dregon.inria.fr/SPCup2019_egonoise/",
    "equipment": "heterogeneous drones and arrays (1/3/8/16 ch); per recording: meta.system, meta.geometry, mic_pos/rotor_pos",
    "observation_type": "onboard arrays and rigs; per recording meta.operating.condition (flight / bench / bench_varying_speed / handheld / rotors_off)",
    "sample_rate": "per recording: 16 / 44.1 / 48 kHz",
    "channels": "per recording: 1 / 3 / 8 / 16",
    "description": "Drone ego-noise from 10 team rigs, every recording annotated from its team's technical report + supplementary material (src/data_processing/sources/spcup19_meta/): condition, rotor-noise presence, active rotors, external sources, mic/rotor geometry with the channel map. ChuMS: 9 hover tests with 1/2/3 stacked propellers per rotor, the two recorders re-aligned. KumamoTech: 16-ch HARK audio + pose decoded from the ROS bags.",
}
