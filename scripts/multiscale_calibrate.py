"""Resolve shaft-order ambiguity in an existing corpus artifact (no PCM reads).

Keep blind/fitted coverage and held-out evaluation separate from the final
all-anchor estimate. Arguments refer to a complete, hash-verified corpus or
one of its immutable per-part receipts. Never use another microphone clock.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from multiscale_corpus import (
    compare_heldout,
    coverage,
    group_view,
    read_rows,
    sha256,
    write_json,
    write_rows,
)
from multiscale_das_condition import from_condition, load_cells
from multiscale_quality import audit

from experiments.static_rig import bracket_tracks, scale_tracks


def split(row):
    points = row["baseline"]["accepted_points"]
    if row["anchors"].get("reason"):
        return [], [], [], "unbound_prior"
    held_times = {q["time_s"] for q in row["heldout"]}
    model_used = {q["time_s"] for q in row["anchors"]["points"]}
    if model_used:
        train = [p for p in points if p["time_s"] in model_used]
        test = [p for p in points if p["time_s"] in held_times]
        policy = "same_block_holdout_as_waveform_model"
    else:
        # No old speed was supplied to the parent algorithm in this case.
        hold = {p["time_s"] for p in points if int(p["time_s"] // 12) % 5 == 2}
        if not hold or len(hold) == len(points):
            hold = {p["time_s"] for p in points if int(p["time_s"] // 4) % 3 == 1}
        if not hold or len(hold) == len(points):
            hold = {p["time_s"] for p in points}
        train = [p for p in points if p["time_s"] not in hold]
        test = [p for p in points if p["time_s"] in hold]
        policy = "blind_waveform_blocked_scale_calibration"
    if {p["time_s"] for p in test} & (model_used | {p["time_s"] for p in train}):
        raise ValueError("held-out anchor leaked into waveform or order calibration")
    return points, train, test, policy


def apply_rows(rows, *, bridge, cells=None):
    for row in rows:
        points, train, test, policy = split(row)
        raw = row["track"]
        expected = row["expected"]
        duration = row["duration_s"]
        ident = row["estimator_sha256"]
        shadow = scale_tracks.calibrate(raw, train, written_for="omitted_reference_blocks")
        if bridge:
            shadow = bracket_tracks.fill_tracks(shadow)
        if cells and row["dataset"] == "DroneAudioSet":
            shadow = from_condition(shadow, cells, row["key"])
        held = compare_heldout(shadow, test, duration)
        base = compare_heldout(raw, test, duration)
        final = (
            scale_tracks.calibrate(raw, points, written_for="all_available_same_source_anchors")
            if len(points) != len(train)
            else shadow
        )
        if bridge and final is not shadow:
            final = bracket_tracks.fill_tracks(final)
        if final is not shadow and cells and row["dataset"] == "DroneAudioSet":
            final = from_condition(final, cells, row["key"])
        anchors = {
            **row["anchors"],
            "waveform_mode": row["anchors"]["mode"],
            "order_policy": policy,
            "order_cv_train_count": len(train),
            "order_cv_test_count": len(test),
            "order_final_anchor_count": len(points),
        }
        yield {
            **row,
            "estimator_sha256": ident,
            "scale_estimator_sha256": sha256(scale_tracks.__file__),
            "bridge_sha256": sha256(bracket_tracks.__file__) if bridge else None,
            "review_status": "experimental_scale_assisted",
            "anchors": anchors,
            "track": final,
            "coverage": coverage(final, expected, duration),
            "heldout": held,
            "waveform_only_heldout": base,
            "heldout_policy": "shadow predictions, not in-sample all-anchor predictions",
        }


def part(spec):
    src, dst, name, bridge, prior = spec
    p = src / "tracks" / name
    receipt = src / "receipts" / name.replace(".jsonl.gz", ".json")
    proof = json.loads(receipt.read_text())
    assert proof["sha256"] == sha256(p), name
    target = dst / "tracks" / name
    write_rows(
        target, apply_rows(read_rows(p), bridge=bridge, cells=load_cells(prior) if prior else None)
    )
    write_json(
        dst / "receipts" / receipt.name,
        {
            "input_sha256": proof["sha256"],
            "sha256": sha256(target),
            "part": name,
            "waveform_model": proof["estimator_sha256"],
            "calibrator_sha256": sha256(scale_tracks.__file__),
            "bridge_sha256": sha256(bracket_tracks.__file__) if bridge else None,
            "rig_prior_sha256": sha256(prior) if prior else None,
        },
    )
    return name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--bridge", action="store_true", help="fill bracketed, noncrowded missing ranks")
    p.add_argument(
        "--das-prior",
        type=Path,
        help="independent archived per-airframe/setting scale readings; leave matching scene out",
    )
    a = p.parse_args()
    summary = json.loads((a.source / "summary.json").read_text())
    base_groups = {g["group_id"]: g for g in read_rows(a.source / "groups.jsonl.gz")}
    grouped = {k: [] for k in base_groups}
    files = sorted((a.source / "tracks").glob("part-*.jsonl.gz"))
    if len(base_groups) != 608 or len(files) != len(summary["parts"]):
        raise ValueError("incomplete parent artifacts")
    with ProcessPoolExecutor(a.workers) as pool:
        for name in pool.map(
            part, [(a.source, a.out, k.name, a.bridge, a.das_prior) for k in files]
        ):
            print("calibrated", name, flush=True)
    counts = Counter()
    matched = {x: Counter() for x in ("waveform_only", "scale_shadow")}
    seen = set()
    for f in files:
        for r in read_rows(a.out / "tracks" / f.name):
            k = (r["dataset"], r["parent_version"], r["key"])
            g = base_groups[r["group_id"]]
            if (
                k in seen
                or k[2] not in g["keys"]
                or r["estimator_sha256"] != summary["estimator_sha256"]
            ):
                raise ValueError(k)
            seen.add(k)
            grouped[r["group_id"]].append(
                {key: r[key] for key in ("key", "coverage", "review_status")}
            )
            counts.update(
                q["basis"] for q in r["track"]["diagnostics"]["scale_calibration"]["tracks"]
            )
            for mode, field in [
                ("waveform_only", "waveform_only_heldout"),
                ("scale_shadow", "heldout"),
            ]:
                for h in r[field]:
                    z = h["levels"]["filled"]
                    m = len(z["matched_errors_rps"])
                    matched[mode].update(
                        {
                            "prior_rates": h["n_prior_rates"],
                            "output_rates": z["n_prediction"],
                            "matched_rates": m,
                        }
                    )
                    if h["n_prior_rates"] == r["expected"]:
                        matched[mode].update(
                            {"complete_ref_points": 1, "complete_matches": int(m == r["expected"])}
                        )
    assert len(seen) == 1728
    groups = []
    for name, rows in sorted(grouped.items()):
        g = base_groups[name]
        assert {r["key"] for r in rows} == set(g["keys"])
        w = group_view(rows)
        groups.append(
            {
                k: g[k]
                for k in (
                    "group_id",
                    "dataset",
                    "recording",
                    "expected",
                    "baseline_whole",
                    "baseline_ever",
                )
            }
            | {"coverage_source_key": w["key"], "coverage": w["coverage"]}
        )
    write_rows(a.out / "groups.jsonl.gz", groups)
    summary = {
        "release": summary["release"],
        "estimator_sha256": summary["estimator_sha256"],
        "scale_estimator_sha256": sha256(scale_tracks.__file__),
        "bridge_sha256": sha256(bracket_tracks.__file__) if a.bridge else None,
        "rig_prior_sha256": sha256(a.das_prior) if a.das_prior else None,
        "cohort_sha256": summary["cohort_sha256"],
        "sources": len(seen),
        "groups": len(groups),
        "source_view_matched": matched,
        "track_scale_sources": counts,
    }
    write_json(a.out / "summary.json", summary)
    result = audit(a.out, a.out / "review")
    print(json.dumps({**summary, "review": result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
