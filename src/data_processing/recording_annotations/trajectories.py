"""Opt-in overlay of a multiscale corpus result on its exact annotated parent.

This is separate from the independently accepted ``rps_acoustic`` observations.
Slots can be recycled after a dropout. Use the accompanying anonymous track
IDs for continuity; neither a slot nor an ID denotes a physical motor.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import tdseries as td

from data_processing.frames import get_meta, with_meta
from data_processing.recording_annotations.storage import _audio_bounds, load_frame, open_release

# Codes are part of the serialized Frame contract. Track values and provenance
# MUST be consumed together: interpolation is not an independent measurement.
TRACK_STATES = {
    "unresolved": 0,
    "measured": 1,
    "track_supported": 2,
    "interpolated": 3,
    "inactive": 4,
}


def attach_track(frame: td.Frame, row: Mapping[str, Any]) -> td.Frame:
    """Attach one corpus JSONL row to ``ra.load_frame(..., audio=True|False)``.

    The native PCM and speed series stay untouched. Timestamps are relative to
    the analyzed role in the artifact; convert exactly once to the Frame clock.
    Do not use a row from a different microphone view, crop, or parent version.
    """
    cond = get_meta(frame, "recording_conditions")
    if cond is None:
        raise ValueError("attach_track requires an annotated parent Frame")
    binding = cond["annotation_release"]
    actual = binding["parent_dataset"], binding["parent_version"], cond["key"]
    wanted = row["dataset"], row["parent_version"], row["key"]
    if actual != wanted:
        raise ValueError(f"multiscale track is for {wanted!r}, not {actual!r}")
    names = (
        "rps_multiscale",
        "rps_multiscale_provenance",
        "rps_multiscale_uncertainty",
        "rps_multiscale_track_id",
    )
    if any(name in frame for name in names):
        raise ValueError("this Frame already carries a multiscale track")
    fs = float(row["sample_rate_hz"])
    t0, duration = float(row["audio_t_start_s"]), float(row["duration_s"])
    if not (
        np.isfinite(fs) and fs > 0 and np.isfinite(t0) and np.isfinite(duration) and duration > 0
    ):
        raise ValueError("invalid source clock")
    if int(row["expected"]) != cond["n_active"]:
        raise ValueError("track slot count is inconsistent with the source rotor configuration")
    if "audio" in frame:
        role = row["role"]
        entry = frame
        for key in role.split("/"):
            if not isinstance(entry, td.Frame) or key not in entry:
                # File-backed WAVs are decoded as `audio`; the raw field is named in metadata.
                if key != role or role not in get_meta(frame, "source_fields", []):
                    raise ValueError(f"missing analyzed role {role!r}")
                entry = frame["audio"]
                break
            entry = entry[key]
        if not isinstance(entry, td.Series) or not isinstance(entry.tindex, td.GridIndex):
            raise ValueError("analyzed role is not uniformly sampled PCM")
        lo, hi = _audio_bounds(entry)
        if (
            abs(entry.tindex.sr - fs) > 1e-6
            or max(abs(lo - t0), abs(hi - t0 - duration)) > 1.1 / fs
        ):
            raise ValueError("track's analyzed audio clock does not match the parent")
    track = row["track"]
    times = np.asarray(track["times_s"], dtype=np.float64)
    rates = np.asarray(track["rates_rps"], dtype=np.float32)
    tags = np.asarray(track["provenance"], dtype=object)
    if (
        times.ndim != 1
        or not times.size
        or not np.isfinite(times).all()
        or not np.all(np.diff(times) > 0)
    ):
        raise ValueError("timestamps must be finite and strictly increasing")
    if (
        times[0] < 0
        or times[-1] >= duration
        or rates.shape != (len(times), int(row["expected"]))
        or tags.shape != rates.shape
    ):
        raise ValueError("invalid relative time domain or track dimensions")
    states = np.full(tags.shape, 255, dtype=np.uint8)
    for tag, code in TRACK_STATES.items():
        states[tags == tag] = code
    supported = (states >= 1) & (states <= 3)
    if (
        (states == 255).any()
        or np.isinf(rates).any()
        or not np.array_equal(np.isfinite(rates), supported)
        or np.any(rates[supported] <= 0)
    ):
        raise ValueError("track rates and support mask disagree")
    stamps = td.StampIndex.from_stamps(times + t0, t_start=t0, t_end=t0 + duration)
    axes = ("track", "time")

    def series(a: np.ndarray) -> td.Series:
        return td.Series(np.ascontiguousarray(a.T), axes, {"time": stamps})

    out = frame.with_entry(names[0], series(rates)).with_entry(names[1], series(states))
    ids = np.asarray(track["track_id"], dtype=object)
    if ids.shape != rates.shape or not np.array_equal(ids != None, supported):  # noqa: E711
        raise ValueError("track IDs and support mask disagree")
    # IDs need not match the anonymous slot index and may not be numeric.
    # Dictionary-encode them for a lossless, codec-friendly integer Series.
    labels = list(dict.fromkeys(ids[supported].tolist()))
    codes = np.full(rates.shape, -1, dtype=np.int32)
    for i, label in enumerate(labels):
        codes[ids == label] = i
    out = out.with_entry(names[3], series(codes))
    if "uncertainty_rps" in track:
        unc = np.asarray(track["uncertainty_rps"], dtype=np.float32)
        if (
            unc.shape != rates.shape
            or np.isinf(unc).any()
            or np.isfinite(unc[~supported]).any()
            or np.any(unc[np.isfinite(unc)] < 0)
        ):
            raise ValueError("invalid multiscale uncertainty")
        out = out.with_entry(names[2], series(unc))
    return with_meta(
        out,
        multiscale_track={
            "estimator_sha256": row["estimator_sha256"],
            "scale_estimator_sha256": row.get("scale_estimator_sha256"),
            "bridge_sha256": row.get("bridge_sha256"),
            "reference_sha256": row.get("reference_sha256"),
            "scale_calibration": track.get("diagnostics", {}).get("scale_calibration"),
            "uncertainty_kind": "estimator tolerance, not a calibrated confidence interval",
            "role": row["role"],
            "review_status": row["review_status"],
            "provenance_codes": dict(TRACK_STATES),
            "identity": "anonymous",
            "track_id_labels": labels,
            "units": "rev/s",
        },
    )


def load_tracked_frame(
    root: str | Path, dataset: str, key: str, *, audio: bool = False
) -> td.Frame:
    """Load one calibrated result by exact source key, optionally with its native PCM.

    Only ``review/source_shards.json``, the selected ``tracks/`` shard and its
    ``receipts/`` JSON are needed from the R2 corpus artifact. The independently
    gated rate series remains available on the returned Frame.
    """
    root = Path(root)
    catalog = json.loads((root / "review" / "source_shards.json").read_text())
    part = catalog["sources"][dataset][key]
    if part != Path(part).name or not part.endswith(".jsonl.gz"):
        raise ValueError(f"invalid track shard name {part!r}")
    receipt = json.loads((root / "receipts" / part.replace(".jsonl.gz", ".json")).read_text())
    path = root / "tracks" / part
    with path.open("rb") as raw:
        if hashlib.file_digest(raw, "sha256").hexdigest() != receipt["sha256"]:
            raise ValueError(f"track shard does not match its receipt: {path}")
        raw.seek(0)
        with gzip.open(raw, "rt") as records:
            for line in records:
                row = json.loads(line)
                if (row["dataset"], row["key"]) != (dataset, key):
                    continue
                if row["estimator_sha256"] != catalog["estimator_sha256"]:
                    raise ValueError("source index and track shard contain different estimators")
                parent = load_frame(
                    dataset,
                    key,
                    version=row["parent_version"],
                    release=open_release(version=catalog["release"]),
                    audio=audio,
                )
                return attach_track(parent, row)
    raise KeyError(f"{dataset}/{key} not found in its recorded shard: {part}")
