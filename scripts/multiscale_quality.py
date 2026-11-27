"""Separate completeness from evidence of accuracy for every original recording.

Read already-synced multiscale corpus shards, never parent audio. A four-slot
row matching only a subset of historical rates does not validate its fourth
slot. Full-vector comparisons below use only prior anchor times NOT supplied
to the estimator and never combine different source clocks.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

TOL = 0.65


def summary_for(row: dict) -> dict:
    expected = row["expected"]
    withheld = row["heldout"]
    full = [r for r in withheld if r["n_prior_rates"] == expected]
    same = [r for r in full if len(r["levels"]["filled"]["matched_errors_rps"]) == expected]
    with_all = [r for r in full if r["levels"]["filled"]["n_prediction"] == expected]
    # Apply one preregistered descriptive screen, not a guarantee for times
    # without references. Require 3 independent *centres*, separated in time,
    # and no wrong or missing assignment at any available complete reference.
    # They are acoustic references and may be correlated within one recording.
    enough = (
        len(full) >= 3 and (max(r["time_s"] for r in full) - min(r["time_s"] for r in full)) >= 4.0
    )
    status = (
        "consistent_at_complete_anchors"
        if enough and len(same) == len(full)
        else "incomplete_or_disagrees_at_complete_anchors"
        if len(same) < len(full)
        else "too_few_complete_anchors"
        if full
        else "no_complete_anchor"
    )
    return {
        "withheld_times": len(withheld),
        "withheld_reference_rates": sum(r["n_prior_rates"] for r in withheld),
        "matched_reference_rates": sum(
            len(r["levels"]["filled"]["matched_errors_rps"]) for r in withheld
        ),
        "complete_reference_times": len(full),
        "output_full_at_reference": len(with_all),
        "matched_full_at_reference": len(same),
        "matched_full_measured_at_reference": sum(
            len(r["levels"]["direct"]["matched_errors_rps"]) == expected for r in full
        ),
        "worst_error_at_matched_full_rps": max(
            (max(r["levels"]["filled"]["matched_errors_rps"]) for r in same), default=None
        ),
        "anchor_agreement": status,
    }


def _rows(path):
    with gzip.open(path, "rt") as stream:
        yield from (json.loads(line) for line in stream)


def audit(root: Path, output: Path) -> dict:
    meta = json.loads((root / "summary.json").read_text())
    groups = list(_rows(root / "groups.jsonl.gz"))
    choose = {(g["dataset"], g["coverage_source_key"]): g for g in groups}
    seen = set()
    report = []
    shard_for = {}
    for file in sorted((root / "tracks").glob("part-*.jsonl.gz")):
        for r in _rows(file):
            k = (r["dataset"], r["key"])
            identity = (*k, r["parent_version"])
            if identity in seen or r["estimator_sha256"] != meta["estimator_sha256"]:
                raise ValueError(f"duplicate or wrong model: {identity}")
            seen.add(identity)
            shard_for.setdefault(r["dataset"], {})[r["key"]] = file.name
            if k not in choose:
                continue
            g = choose[k]
            c = r["coverage"]
            expected = r["expected"]
            score = summary_for(r)
            report.append(
                {
                    "dataset": r["dataset"],
                    "recording": g["recording"],
                    "source_key": r["key"],
                    "source_parent_version": r["parent_version"],
                    "shard": file.name,
                    "active_rotors": expected,
                    "native_s": r["duration_s"],
                    "active_s": c["active_s"],
                    "any_estimate_s": c["filled"]["any_s"],
                    "coverage_status": (
                        "complete"
                        if c["filled"]["whole"]
                        else "partial"
                        if c["filled"]["any_s"] > 0
                        else "unresolved"
                        if c["active_s"] > 0
                        else "activity_unresolved"
                    ),
                    "max_simultaneous": c["filled"]["max_simultaneous"],
                    "all_measured_s": c["direct"]["all_s"],
                    "all_supported_s": c["supported"]["all_s"],
                    "all_including_interpolation_s": c["filled"]["all_s"],
                    "whole_including_interpolation": c["filled"]["whole"],
                    "ever_including_interpolation": c["filled"]["ever"],
                    **score,
                    "review_status": r["review_status"],
                }
            )
    if len(seen) != 1728 or len(report) != len(groups) or len(report) != 608:
        raise ValueError(f"incomplete source-view/group coverage: {len(seen)}, {len(report)}")
    report.sort(key=lambda x: (x["dataset"], x["recording"]))
    output.mkdir(parents=True, exist_ok=True)
    (output / "source_shards.json").write_text(
        json.dumps(
            {
                "release": meta["release"],
                "estimator_sha256": meta["estimator_sha256"],
                "sources": shard_for,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    for tag, rows in [
        ("all", report),
        ("incomplete", [r for r in report if not r["whole_including_interpolation"]]),
        *(
            (state, [r for r in report if r["coverage_status"] == state])
            for state in ("partial", "unresolved", "complete", "activity_unresolved")
        ),
        (
            "without_four_rate_validation",
            [
                r
                for r in report
                if r["active_rotors"] == 4
                and r["anchor_agreement"] != "consistent_at_complete_anchors"
            ],
        ),
    ]:
        with (output / f"{tag}.tsv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=tuple(report[0]), delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    sums = {}
    for name, rows in [
        ("four_rotor", [r for r in report if r["active_rotors"] == 4]),
        ("single_rotor", [r for r in report if r["active_rotors"] == 1]),
    ]:
        sums[name] = {
            "groups": len(rows),
            "coverage_categories": dict(Counter(r["coverage_status"] for r in rows)),
            "all_rates_for_at_least_90pct_active": sum(
                r["active_s"] > 0 and r["all_including_interpolation_s"] >= 0.90 * r["active_s"]
                for r in rows
            ),
            "agreement_categories": dict(Counter(r["anchor_agreement"] for r in rows)),
            "whole_and_agrees": sum(
                r["whole_including_interpolation"]
                and r["anchor_agreement"] == "consistent_at_complete_anchors"
                for r in rows
            ),
            "whole_without_validation": sum(
                r["whole_including_interpolation"]
                and r["anchor_agreement"] != "consistent_at_complete_anchors"
                for r in rows
            ),
            **{
                key: sum(r[key] for r in rows)
                for key in (
                    "complete_reference_times",
                    "output_full_at_reference",
                    "matched_full_at_reference",
                    "matched_full_measured_at_reference",
                    "matched_reference_rates",
                    "withheld_reference_rates",
                )
            },
        }
    result = {
        "model": meta["estimator_sha256"],
        "release": meta["release"],
        "tol_rps": TOL,
        "coverage_source": "single source selected on coverage, never on validation score",
        "interpretation": "agreement with withheld acoustic measurements; not independent ESC truth or confirmation elsewhere",
        "corpus_summary_sha256": hashlib.sha256((root / "summary.json").read_bytes()).hexdigest(),
        **sums,
    }
    (output / "review.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.output), indent=2))
