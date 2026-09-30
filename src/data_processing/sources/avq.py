"""AVQ source (audio-visual quadrotor: onboard 8-mic array + speech + DOA/VAD).

Twelve sequences in two sessions, all recorded with the quadcopter on a tripod
(it never flies). What each sequence IS — ego-noise only / speech only with the
motors muted / mixture, constant or time-varying motor speed, fixed or walking
talker — lives in the publisher's spec table, not in the files; it is recorded
ONCE in ``avq_meta.yaml`` (every value sourced) and joined by key. The builder
refuses a sequence without an annotation and an annotation without a sequence.

Each recording Frame holds ``audio`` ``(mic, time)`` at 44.1 kHz, ``mic_pos``
``(mic, 3)`` from ``misc/mic_pos.mat`` (row i = channel i, ASSUMED — see
``meta.geometry.channel_map_source``), ``angle_vad`` where the sequence has DOA/
VAD ground truth, and ``meta`` with ``operating`` (``condition`` ∈
:data:`~data_processing.sources._common.CONDITIONS`, ``contains_rotor_noise``,
``n_active_rotors``, ``throttle_or_speed``), ``label`` (``content``,
``external_source``, ``external_source_motion``, ``external_source_area``),
``geometry`` and ``annotation`` (source, confidence).
"""

from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import tdseries as td

from data_processing.frames import audio_series
from data_processing.sources._common import (
    meta_frame,
    read_audio_file,
    safe_key,
    validate_operating,
)

URL = "https://webspace.eecs.qmul.ac.uk/lin.wang/download/avq.zip"

#: The per-sequence annotation (schema in its header comment).
ANNOTATIONS = Path(__file__).with_name("avq_meta.yaml")
#: ``label.content``: what is audible, per the spec table's Type column.
CONTENTS = ("ego_noise_only", "source_only", "mixture")
#: ``label.external_source_motion``: the talker/loudspeaker, not the drone.
EXTERNAL_SOURCE_MOTIONS = ("none", "piecewise_static", "moving")


@lru_cache(maxsize=1)
def load_annotations() -> dict[str, Any]:
    """Parsed ``avq_meta.yaml`` plus ``by_key``, validated."""
    import yaml

    doc = yaml.safe_load(ANNOTATIONS.read_text())
    n_rotors = int(doc["drone"]["n_rotors"])
    by_key: dict[str, dict[str, Any]] = {}
    for rec in doc["recordings"]:
        key = rec["key"]
        where = f"AVQ/{key}"
        if key in by_key:
            raise ValueError(f"{where}: duplicate key")
        validate_operating(rec, n_rotors, where)
        content, motion = rec["content"], rec["external_source_motion"]
        if content not in CONTENTS or motion not in EXTERNAL_SOURCE_MOTIONS:
            raise ValueError(f"{where}: content {content!r} / source motion {motion!r}")
        silent_source = rec["external_source"] == "none"
        if (content == "ego_noise_only") != silent_source or silent_source != (motion == "none"):
            raise ValueError(f"{where}: ego_noise_only <=> no external source <=> no source motion")
        if (content == "source_only") != (rec["condition"] == "rotors_off"):
            raise ValueError(f"{where}: source_only <=> rotors_off")
        by_key[key] = rec
    return {**doc, "by_key": by_key}


def annotations_digest() -> str:
    """sha256 of the annotation file: part of the derivation identity."""
    return hashlib.sha256(ANNOTATIONS.read_bytes()).hexdigest()


URL = "https://webspace.eecs.qmul.ac.uk/lin.wang/download/avq.zip"


def _mic_pos(session_dir: Path) -> np.ndarray | None:
    """``misc/mic_pos.mat`` -> ``(mic, 3)`` positions (source is ``(3, 8)``)."""
    from scipy.io import loadmat

    p = session_dir / "misc" / "mic_pos.mat"
    if not p.exists():
        return None
    mp = np.asarray(loadmat(str(p), squeeze_me=True)["mic_pos"], dtype=np.float64)
    if mp.ndim == 2 and mp.shape[0] == 3:  # (3, 8) -> (8, 3)
        mp = mp.T
    return np.ascontiguousarray(mp)


def _session_info(session_dir: Path) -> dict[str, Any]:
    """Session-level calibration (``angle_a = a1*angle_v + a2``) + readme text."""
    from scipy.io import loadmat

    out: dict[str, Any] = {}
    cal_p = session_dir / "misc" / "av_calibration.mat"
    if cal_p.exists():
        d = loadmat(str(cal_p), squeeze_me=True, struct_as_record=False)
        cal = d.get("Calibration")
        if cal is not None and hasattr(cal, "_fieldnames"):
            for f in cal._fieldnames:  # pyright: ignore[reportAttributeAccessIssue]
                with contextlib.suppress(TypeError, ValueError):
                    out[f"av_calib_{f}"] = float(getattr(cal, f))
        if "description" in d:
            out["av_calib_description"] = str(d["description"])
    readme_p = session_dir / "misc" / "readme.txt"
    if readme_p.exists():
        out["readme"] = readme_p.read_text(errors="ignore").strip()
    return out


def _angle_vad(seq_dir: Path) -> tuple[np.ndarray | None, str]:
    """``angle_vad.mat`` -> ``(step, col)`` DOA/VAD ground truth + its column
    description (schema differs per session — preserved verbatim, not unified)."""
    from scipy.io import loadmat

    p = seq_dir / "angle_vad.mat"
    if not p.exists():
        return None, ""
    d = loadmat(str(p), squeeze_me=True)
    angles = np.asarray(d["angles"], dtype=np.float64)
    if angles.ndim == 1:
        angles = angles[:, None]
    return np.ascontiguousarray(angles), str(d.get("description", ""))


def _channels(seq_dir: Path) -> tuple[np.ndarray, int] | None:
    """Stack the ``MONO-000..NNN`` mono files (case-insensitive) into ``(C, T)``,
    truncating to the shortest channel. Returns None if no channel files."""
    wavs = sorted(
        (
            p
            for p in seq_dir.iterdir()
            if p.suffix.lower() == ".wav" and p.stem.lower().startswith("mono")
        ),
        key=lambda p: p.name.lower(),
    )
    if not wavs:
        return None
    chans: list[np.ndarray] = []
    sr0: int | None = None
    for w in wavs:
        audio_ct, sr = read_audio_file(w)  # mono file -> (1, T)
        chans.append(audio_ct[0])
        sr0 = sr0 or sr
    length = min(c.shape[0] for c in chans)
    audio = np.stack([c[:length] for c in chans], axis=0)  # (C, T)
    return np.ascontiguousarray(audio), int(sr0 or 44100)


def _meta(
    doc: dict[str, Any], rec: dict[str, Any], rid: str, n_ch: int, extra: dict[str, Any]
) -> td.Frame:
    geo = doc["geometry"]["array8"]
    meta = meta_frame(
        rid,
        "AVQ",
        system={
            "category": "drone",
            "make_model": doc["drone"]["make_model"],
            "n_rotors": int(doc["drone"]["n_rotors"]),
            "mounting": doc["platform"]["mounting"],
            "mic_array": f"{n_ch}ch onboard",
        },
        observation={
            "type": "onboard_array",
            "n_channels": n_ch,
            "source_motion": rec["external_source_motion"],
            "platform_motion": "static (tripod-mounted, never airborne)",
            "video_ground_truth": extra["has_video"],
        },
        operating={
            "condition": rec["condition"],
            "contains_rotor_noise": bool(rec["contains_rotor_noise"]),
            "n_active_rotors": rec["n_active_rotors"],
            "throttle_or_speed": rec["throttle_or_speed"],
        },
        label={
            "content": rec["content"],
            "external_source": rec["external_source"],
            "external_source_motion": rec["external_source_motion"],
            "external_source_area": rec["external_source_area"],
            "has_angle_vad": extra["has_angle_vad"],
            "has_video": extra["has_video"],
        },
        extra={
            **extra,
            "spec_row": rec["spec_row"],
            "description": " ".join(str(rec["description"]).split()),
        },
    )
    geometry = {
        "config": "array8",
        "frame": " ".join(doc["frame"]["description"].split()),
        "mic_pos_quality": geo["mic_pos_quality"],
        "rotor_pos_quality": geo["rotor_pos_quality"],
        "channel_map_source": " ".join(geo["channel_map_source"].split()),
        "source": geo["source"],
        "notes": " ".join(geo["notes"].split()),
    }
    return td.Frame(
        {
            **dict(meta.items()),
            "geometry": td.Frame(geometry),
            "annotation": td.Frame(
                {"source": " ".join(str(rec["source"]).split()), "confidence": rec["confidence"]}
            ),
        }
    )


def build(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """One recording Frame per annotated sequence: 8-ch ``audio`` (native
    44.1 kHz) + ``mic_pos`` + ``angle_vad`` (labeled sequences) + the
    annotation and per-session meta (AV calibration, readme). The bulky/opaque
    blobs (video, cameraParams.mat, .docx) are kept byte-exact in the companion
    raw dataset ``AVQ-raw``."""
    doc = load_annotations()
    root = Path(raw_dir)
    seen: set[str] = set()
    # mic_pos.mat lives at <session>/misc/mic_pos.mat -> session = parent of misc/
    session_dirs = sorted({p.parent.parent for p in root.rglob("misc/mic_pos.mat")})
    for sess_dir in session_dirs:
        session = sess_dir.name  # "S1" / "S2"
        mic_pos = _mic_pos(sess_dir)
        info = _session_info(sess_dir)
        seq_dirs = sorted(
            p for p in sess_dir.iterdir() if p.is_dir() and p.name.lower().startswith("seq")
        )
        for seq_dir in seq_dirs:
            got = _channels(seq_dir)
            if got is None:
                continue
            audio, sr = got
            rid = f"{session}_{seq_dir.name}"
            rec = doc["by_key"].get(rid)
            if rec is None:
                raise ValueError(f"AVQ {rid}: no annotation in {ANNOTATIONS.name}")
            seen.add(rid)
            angles, ang_desc = _angle_vad(seq_dir)
            has_video = any(p.suffix.lower() == ".mp4" for p in seq_dir.iterdir())
            extra: dict[str, Any] = {
                "session": session,
                "sequence": seq_dir.name,
                "sample_rate": int(sr),
                "n_channels": int(audio.shape[0]),
                "duration_s": round(audio.shape[1] / sr, 3),
                "has_video": has_video,
                "has_angle_vad": angles is not None,
                **info,
            }
            if angles is not None:
                extra["angle_vad_columns"] = ang_desc
            entries: dict[str, Any] = {
                "audio": audio_series(audio, int(sr)),
                "meta": _meta(doc, rec, rid, int(audio.shape[0]), extra),
            }
            if mic_pos is not None:
                if mic_pos.shape[0] != audio.shape[0]:
                    raise ValueError(f"AVQ {rid}: {audio.shape[0]} channels, {len(mic_pos)} mics")
                entries["mic_pos"] = td.wrap(mic_pos, dims=("mic", None))
            if angles is not None:
                entries["angle_vad"] = td.wrap(angles, dims=("avq_step", None))
            yield safe_key(rid), td.Frame(entries)
    unused = sorted(set(doc["by_key"]) - seen)
    if unused:
        raise ValueError(f"AVQ annotations without a recording: {unused}")


PROVENANCE = {
    "source_url": "https://webspace.eecs.qmul.ac.uk/lin.wang/download/avq.zip",
    "project_url": "https://webspace.eecs.qmul.ac.uk/lin.wang/",
    "license": "free for academic/research use (courtesy of Lin Wang, QMUL)",
    "citation": "L. Wang, R. Sanchez-Matilla and A. Cavallaro, 'Audio-visual sensing from a quadcopter: dataset and baselines for source localization and sound enhancement', IROS 2019, doi 10.1109/IROS40897.2019.8968183.",
    "collection_method": "tripod-mounted quadcopter (never airborne) with an onboard 8-mic array: ego-noise-only sequences (constant or time-varying motor speed), speech-only sequences with the motors muted (two fixed talkers / a walking loudspeaker) and mixtures; synchronized GoPro video gives DOA + VAD ground truth. Per-sequence tags in sources/avq_meta.yaml from the publisher's spec table.",
    "equipment": "quadrotor + onboard 8-microphone array (positions in mic_pos.mat) + GoPro camera",
    "observation_type": "onboard_array",
    "sample_rate": 44100,
    "channels": 8,
    "description": "Audio-visual quadrotor: 2 sessions (S1/S2), 12 sequences of 8-ch onboard-array audio (44.1 kHz): 5 ego-noise only, 3 speech only (motors muted), 4 mixtures; the drone sits on a tripod throughout. Each sequence carries meta.operating (condition bench / bench_varying_speed / rotors_off, active rotors, throttle) and meta.label (content, external source, its motion). Labeled sequences carry angle_vad.mat (per-session DOA/VAD schema, preserved verbatim) + mic geometry + AV calibration. Bulky/opaque blobs (video, cameraParams.mat, .docx) live byte-exact in the companion raw dataset AVQ-raw.",
    "companion_raw_dataset": "AVQ-raw",
}
