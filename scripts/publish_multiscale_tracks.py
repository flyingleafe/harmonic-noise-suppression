"""Publish and cold-verify opt-in trajectory Frames, never source audio or the default RPS labels.

Uses the calibrated corpus artifact with per-source receipts. Known count
sets the number of slots; it does not certify multiplicity. Scientific review
and held-out comparisons travel with the data; a fully populated hypothesis is
not necessarily an accurate four-shaft annotation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import tdseries as td
from dload.pack import PackReader

from data_processing import streams
from data_processing.recording_annotations.trajectories import attach_track

NAME = "drone-rotor-track-candidates"


def samples(root: Path, meta: dict, source_index: dict):
    seen = set()
    groups = set()
    for part in sorted((root / "tracks").glob("part-*.jsonl.gz")):
        receipt = json.loads(
            (root / "receipts" / part.name.replace(".jsonl.gz", ".json")).read_text()
        )
        with part.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != receipt["sha256"]:
                raise ValueError(f"mismatched artifact receipt: {part}")
            stream.seek(0)
            with gzip.open(stream, "rt") as rows:
                for row in map(json.loads, rows):
                    dataset, key = row["dataset"], row["key"]
                    identity = f"{dataset}/{key}"
                    if (
                        identity in seen
                        or row["estimator_sha256"] != meta["estimator_sha256"]
                        or source_index[dataset][key] != part.name
                    ):
                        raise ValueError(f"duplicate or mismatched source identity: {identity}")
                    seen.add(identity)
                    groups.add(row["group_id"])
                    cond = {
                        "key": key,
                        "n_active": row["expected"],
                        "annotation_release": {
                            "parent_dataset": dataset,
                            "parent_version": row["parent_version"],
                            "version": meta["release"],
                        },
                    }
                    source = td.Frame(
                        {
                            "meta": td.Frame(
                                {
                                    "recording_id": identity,
                                    "recording_group": row["group_id"],
                                    "recording_conditions": cond,
                                    "source_audio_t_start_s": row["audio_t_start_s"],
                                    "source_duration_s": row["duration_s"],
                                    "source_role": row["role"],
                                    "source_channels": row["source_view"]["channels"],
                                    "blade_count": row.get("blades"),
                                    "blade_count_basis": row.get("blade_count_basis"),
                                    "heldout_prior_rates": sum(
                                        h["n_prior_rates"] for h in row["heldout"]
                                    ),
                                    "matched_heldout_prior_rates": sum(
                                        len(h["levels"]["filled"]["matched_errors_rps"])
                                        for h in row["heldout"]
                                    ),
                                    "complete_reference_times": sum(
                                        h["n_prior_rates"] == row["expected"]
                                        for h in row["heldout"]
                                    ),
                                    "matched_complete_reference_times": sum(
                                        len(h["levels"]["filled"]["matched_errors_rps"])
                                        == row["expected"]
                                        for h in row["heldout"]
                                    ),
                                    "scale_validation": "shadow fit; additional acoustic references only, not physical ground truth",
                                }
                            )
                        }
                    )
                    frame = attach_track(source, row)
                    yield identity, streams.frame_to_sample(frame)
    if (len(seen), len(groups)) != (meta["sources"], meta["groups"]):
        raise ValueError("incomplete original-source coverage")


def publish(root: Path, receipt_path: Path):
    m = json.loads((root / "summary.json").read_text())
    catalog = json.loads((root / "review/source_shards.json").read_text())
    if (m["sources"], m["groups"]) != (1728, 608) or (
        catalog["release"],
        catalog["estimator_sha256"],
    ) != (m["release"], m["estimator_sha256"]):
        raise ValueError("not a verified fixed-throttle corpus")
    expected = {f"{ds}/{key}" for ds, keys in catalog["sources"].items() for key in keys}
    if len(expected) != 1728:
        raise ValueError("source catalog is incomplete")
    repo = streams.open_repository()
    manifest = repo.commit(
        NAME,
        samples(root, m, catalog["sources"]),
        meta={
            "layout": streams.TDFRAME_LAYOUT,
            "status": "experimental estimates, explicitly not verified per-motor labels",
            "estimator_sha256": m["estimator_sha256"],
            "scale_estimator_sha256": m["scale_estimator_sha256"],
            "bridge_sha256": m["bridge_sha256"],
            "annotation_base": m["release"],
            "cohort_sha256": m["cohort_sha256"],
            "calibrated_artifact": root.name,
            "original_source_views": 1728,
            "original_recording_groups": 608,
            "units": "rev/s",
            "no_audio": True,
        },
        recipe="scripts/multiscale_corpus.py -> scripts/multiscale_calibrate.py --bridge -> scripts/publish_multiscale_tracks.py",
        target_shard_size=2 * 1024**2,
    )
    keys = set()
    states = {}
    size = 0
    checked = repo.manifest(NAME, manifest.version)
    for shard in checked.shards:
        repo.cache.evict(shard.digest)
        with repo.open_shard(shard) as p:
            with Path(p).open("rb") as f:
                if hashlib.file_digest(f, "sha256").hexdigest() != shard.digest:
                    raise ValueError("cold pack digest mismatch")
            with PackReader(p) as pack:
                for key, fields in pack:
                    frame = streams.sample_to_frame(fields)
                    if (
                        key in keys
                        or "audio" in frame
                        or "rps" in frame
                        or frame["meta"]["recording_id"] != key
                    ):
                        raise ValueError(f"invalid trajectory readback: {key}")
                    rates = frame["rps_multiscale"].data
                    prov = frame["rps_multiscale_provenance"].data
                    if rates.shape != prov.shape or not np.array_equal(
                        np.isfinite(rates), np.isin(prov, [1, 2, 3])
                    ):
                        raise ValueError(f"bad decoded support: {key}")
                    keys.add(key)
                    for v, n in zip(*np.unique(prov, return_counts=True), strict=True):
                        states[str(int(v))] = states.get(str(int(v)), 0) + int(n)
        size += shard.size
    if keys != expected:
        raise ValueError("cold readback differs from original source keys")
    receipt = {
        "name": NAME,
        "version": manifest.version,
        "source_views": len(keys),
        "groups": 608,
        "shards": len(checked.shards),
        "bytes": size,
        "provenance_counts": states,
        "baseline": m["release"],
        "producer_summary_sha256": hashlib.sha256((root / "summary.json").read_bytes()).hexdigest(),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--confirm", choices=[NAME], required=True)
    args = p.parse_args()
    publish(args.root, args.receipt)
