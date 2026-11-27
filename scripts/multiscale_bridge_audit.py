"""Audit linear interpolation between *measured complete rate sets*, without PCM.

This conservative baseline cannot invent an unobserved fourth family. All rates
are sorted ranks, not physical motors. Calibrates neither the old measurements
nor their rank identities: the reference is agreement with withheld same-clock
acoustic anchors. A discontinuity, an off window or a conflicting observed rate
blocks the bridge. No extrapolation, even to the first/last active second.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from close_pair_inventory import FILTER, PARENTS

from data_processing import recording_annotations as ra

# Fixed before examining holdout errors. No data fitting on the test positions.
MAX_GAP_S = 16.0
MAX_ENDPOINT_DELTA = 1.0  # rev/s, per sorted rank across the entire gap
TOL = 0.65  # rev/s: same tolerance as the historical AVQ reference check
BASE = "497ea2c5828a6cac19dc23ceabe3e1c4340725265ad99c06bc0218108797cd10"


def score_source(a: dict) -> tuple[list[tuple], list[dict]]:
    rates = a["rates"]
    times = a["times"].astype(float)
    t = times[:, 2]
    expected = int(a["n_expected"])
    observations = [
        np.sort(x[np.isfinite(x)]) if ok else np.array([])
        for x, ok in zip(rates, a["accepted"], strict=True)
    ]
    complete = [int(i) for i, s in enumerate(observations) if len(s) == expected]
    blocked = np.array(
        [
            str((note.get("quality") or {}).get("reason", note.get("reason")))
            in {"silent", "nonstationary_or_inconsistent_halves"}
            for note in a["notes"]
        ]
    )

    def bridge(left: int, right: int, hidden: range):
        if right <= left + 1 or t[right] - t[left] > MAX_GAP_S or blocked[left : right + 1].any():
            return None
        # The windows need to describe one continuous analysed span; no assumption
        # about gaps caused by a paused recorder or a motor being switched off.
        if np.any(times[left + 1 : right + 1, 0] > times[left:right, 1] + 1e-4):
            return None
        lo, hi = observations[left], observations[right]
        if np.max(np.abs(hi - lo)) > MAX_ENDPOINT_DELTA:
            return None
        out = {}
        for j in range(left + 1, right):
            pred = lo + (hi - lo) * (t[j] - t[left]) / (t[right] - t[left])
            obs = observations[j]
            # Minimum worst error over order-preserving one-to-one matches.
            if (
                j not in hidden
                and len(obs)
                and min(
                    max(abs(obs - pred[list(ix)])) for ix in combinations(range(expected), len(obs))
                )
                > TOL
            ):
                return None
            out[j] = pred
        return out

    production = []
    for left, right in zip(complete, complete[1:]):
        candidates = bridge(left, right, range(0))
        if candidates:
            for j, prediction in candidates.items():
                obs = observations[j]
                rr = prediction.copy()
                states = ["interpolated"] * expected
                if len(obs):
                    slots = min(
                        combinations(range(expected), len(obs)),
                        key=lambda ix: max(abs(obs - prediction[list(ix)])),
                    )
                    rr[list(slots)] = obs
                    for slot in slots:
                        states[slot] = "measured"
                production.append(
                    (
                        round(t[j], 6),
                        round(t[left], 6),
                        round(t[right], 6),
                        float(times[j, 1] - times[j, 0]),
                        rr.tolist(),
                        states,
                    )
                )
    heldout = []
    # Non-overlapping test brackets within a view. Withhold *all* interior
    # observations (including partials); an interior truth is never a seed.
    for i in range(0, len(complete) - 2, 3):
        left, j, right = complete[i : i + 3]
        out = bridge(left, right, range(left + 1, right))
        heldout.append(
            {
                "t_s": t[j],
                "gap_s": t[right] - t[left],
                "error_rps": None if out is None else float(np.max(abs(out[j] - observations[j]))),
            }
        )
    return production, heldout


def run(out: Path) -> None:
    rel = ra.open_release(version=BASE)
    reports = {}
    for ds in PARENTS:
        idx = rel.index(ds)
        records = {}
        for k in idx.keys(FILTER):
            a = idx.tach_arrays(k)
            assert a is not None
            prod, withheld = score_source(a)
            if prod or withheld:
                records[k] = {
                    "bridged_centers": prod,
                    "heldout": withheld,
                    "n_expected": a["n_expected"],
                }
        reports[ds] = records
    out.mkdir(parents=True, exist_ok=True)
    sums = defaultdict(dict)
    for ds, group in reports.items():
        cases = [test for v in group.values() for test in v["heldout"]]
        errors = [c["error_rps"] for c in cases if c["error_rps"] is not None]
        sums[ds] = {
            "proposed_source_view_s": sum(
                p[3] for v in group.values() for p in v["bridged_centers"]
            ),
            "proposed_source_view_cells": sum(len(v["bridged_centers"]) for v in group.values()),
            "holdout_total": len(cases),
            "holdout_covered": len(errors),
            "holdout_all_match": sum(e <= TOL for e in errors),
            "median_max_abs_rps": float(np.median(errors)) if errors else None,
        }
    payload = {
        "base": BASE,
        "counting_unit": "native source view; not independent recording",
        "maximum_bracket_s": MAX_GAP_S,
        "bridged_center_fields": [
            "t_s",
            "left_anchor_s",
            "right_anchor_s",
            "duration_s",
            "rates_rps",
            "provenance",
        ],
        "maximum_endpoint_delta_rps": MAX_ENDPOINT_DELTA,
        "match_tol_rps": TOL,
        "by_dataset": sums,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "source_views.json").write_text(json.dumps(reports, separators=(",", ":")) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    run(p.parse_args().out)
