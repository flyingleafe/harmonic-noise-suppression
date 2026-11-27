"""List every original fixed-throttle recording by published acoustic coverage (no PCM).

Usage: python scripts/close_pair_inventory.py --release FULL_SHA --out results/close_pair_inventory
The parent roster and grouping are the original-source census in docs/data-catalog.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from data_processing import recording_annotations as ra

PARENTS = (
    "DroneAudioSet",
    "DREGON-frames",
    "SPCUP19-frames",
    "AVQ",
    "KULeuven-UAV-egonoise-frames",
    "zenodo_drone_noises",
    "Salford-DroneNoise-frames",
)
FILTER = {"speed_regime": "constant", "has_audio": True, "provenance_kind": "recording"}
DAS = re.compile(
    r"\d{6}_(.+)__(mic1_soundskrit|mic2_8array-down|mic3_8array-up)-(File\d|silence)\.wav"
)
FIELDS = (
    "dataset",
    "recording",
    "status",
    "expected_active_rotors",
    "max_simultaneous",
    "all_seen_in_one_window",
    "coverage_source_key",
    "assessed_s",
    "any_speed_s",
    "all_speeds_s",
    "source_keys_json",
)


def _seconds(times: np.ndarray) -> float:
    """Union length on one source clock; do not double-count overlapping observations."""
    total = 0.0
    previous_end = -np.inf
    for start, end in sorted(times[:, :2].tolist()):
        total += max(0.0, end - max(previous_end, start))
        previous_end = max(previous_end, end)
    return round(total, 6)


def build(release: str, out: Path) -> Counter:
    r = ra.open_release(version=release)
    rows = []
    for dataset in PARENTS:
        idx = r.index(dataset)
        groups = defaultdict(dict)
        microphones = defaultdict(set)
        for key in idx.keys(FILTER):
            recording = key
            if dataset == "DroneAudioSet":
                m = DAS.fullmatch(key)
                if not m:
                    raise ValueError(f"unexpected publisher key: {key}")
                recording = f"{m[1]}__{m[3]}"
                microphones[recording].add(m[2])
            a = idx.tach_arrays(key)
            if not a or not a["n_expected"]:
                raise ValueError(f"missing active-shaft assessment: {dataset}/{key}")
            n = np.isfinite(a["rates"]).sum(axis=1) * a["accepted"]
            groups[recording][key] = {
                "status": a["status"],
                "reason": a["reason"],
                "expected": a["n_expected"],
                "max_simultaneous": int(n.max(initial=0)),
                "assessed_s": _seconds(a["times"]),
                "any_speed_s": _seconds(a["times"][n > 0]),
                "all_speeds_s": _seconds(a["times"][n == a["n_expected"]]),
            }
        for recording, views in groups.items():
            if dataset == "DroneAudioSet" and (len(views) != 3 or len(microphones[recording]) != 3):
                raise ValueError(f"incomplete three-view group: {recording}")
            (expected,) = {v["expected"] for v in views.values()}
            best = max(v["max_simultaneous"] for v in views.values())
            # Choose ONE view, never sum seconds or shaft counts across unsynchronized clocks.
            key = max(
                sorted(views),
                key=lambda k: (
                    views[k]["status"] == "resolved",
                    views[k]["all_speeds_s"],
                    views[k]["any_speed_s"],
                    views[k]["max_simultaneous"],
                ),
            )
            view = views[key]
            status = (
                "resolved" if view["status"] == "resolved" else "partial" if best else "unresolved"
            )
            rows.append(
                {
                    "dataset": dataset,
                    "recording": recording,
                    "status": status,
                    "expected_active_rotors": expected,
                    "max_simultaneous": best,
                    "all_seen_in_one_window": int(best == expected),
                    "coverage_source_key": key,
                    "assessed_s": view["assessed_s"],
                    "any_speed_s": view["any_speed_s"],
                    "all_speeds_s": view["all_speeds_s"],
                    "source_keys_json": json.dumps(views, sort_keys=True, separators=(",", ":")),
                }
            )
    rows.sort(key=lambda row: (row["dataset"], row["recording"]))
    out.mkdir(parents=True, exist_ok=True)
    for tag in ("all", "partial", "resolved", "unresolved"):
        with (out / f"{tag}.tsv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(row for row in rows if tag == "all" or row["status"] == tag)
    counts = Counter(row["status"] for row in rows)
    by_dataset = {
        ds: dict(Counter(row["status"] for row in rows if row["dataset"] == ds)) for ds in PARENTS
    }
    summary = {
        "release": r.version,
        "recordings": len(rows),
        "counts": counts,
        "by_dataset": by_dataset,
        "simultaneous_full_at_least_once": sum(row["all_seen_in_one_window"] for row in rows),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return counts


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    print(build(args.release, args.out))
