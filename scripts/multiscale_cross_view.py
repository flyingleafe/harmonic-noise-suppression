"""Compare reconstructed upper and lower rates on *verified* native-clock pairs.

An independent recorder is not an independent tachometer: shared rotor tones
may yield the same octave error. No cross-view estimates are merged or used to
fill anything here. Score distinct rates simultaneously, not cumulative sightings.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

PAIR = "mic2_8array-down/mic3_8array-up"
TOL = 0.65


def pit(a, b):
    d = np.abs(a[:, None] - b[None, :])
    i, j = linear_sum_assignment(np.where(d <= TOL, d, 99.0))
    return d[i, j]


def run(root: Path, clock_json: Path, out: Path):
    clocks = json.loads(clock_json.read_text())
    sm = json.loads((root / "summary.json").read_text())
    if sm["release"] != clocks["release"]:
        raise ValueError("mismatched source release for cross-view checks")
    wanted = {v["scene"]: v for v in clocks["rows"] if v["pairs"][PAIR]["screen_pass"]}
    data = {k: {} for k in wanted}
    for f in sorted((root / "tracks").glob("part-*.jsonl.gz")):
        with gzip.open(f, "rt") as lines:
            for row in map(json.loads, lines):
                if row["dataset"] != "DroneAudioSet" or row["recording"] not in wanted:
                    continue
                view = row["source_view"]["view_label"]
                if view not in PAIR.split("/"):
                    continue
                tr = row["track"]
                data[row["recording"]][view] = (
                    np.array(tr["times_s"]),
                    np.array(tr["rates_rps"], float),
                    np.array(tr["provenance"]),
                    row["sample_rate_hz"],
                )
    rows = []
    for k, clocks_for_take in wanted.items():
        lo, up = (data[k][v] for v in PAIR.split("/"))
        assert lo[3] == up[3]
        t = np.asarray(clocks_for_take["times_s"])
        lags = np.array(clocks_for_take["pairs"][PAIR]["band_lag_samples_and_peak_ratio"])[
            :, :, 0
        ].mean(1)
        slope, intercept = np.polyfit(t[::2], lags[::2] / lo[3], 1)
        totals = Counter()
        overlaps = []
        for i, tl in enumerate(lo[0]):
            tu = tl + slope * tl + intercept
            j = int(np.argmin(abs(up[0] - tu)))
            if abs(up[0][j] - tu) > 0.55:
                continue
            if all(lo[2][i] == "inactive") or all(up[2][j] == "inactive"):
                continue
            a = lo[1][i][np.isfinite(lo[1][i])]
            b = up[1][j][np.isfinite(up[1][j])]
            e = pit(a, b)
            totals["common_centres"] += 1
            totals["matched_rates"] += int((e <= TOL).sum())
            if len(a) == len(b) == 4:
                totals["both_four"] += 1
                totals["all_four_matched"] += int(e.max() <= TOL)
                overlaps.extend(e.tolist())
                if all(np.isin(lo[2][i], ["measured", "track_supported"])) and all(
                    np.isin(up[2][j], ["measured", "track_supported"])
                ):
                    totals["both_four_no_interp"] += 1
                    totals["all_four_matched_no_interp"] += int(e.max() <= TOL)
        rows.append(
            dict(
                recording=k,
                **{
                    x: totals[x]
                    for x in (
                        "common_centres",
                        "both_four",
                        "all_four_matched",
                        "both_four_no_interp",
                        "all_four_matched_no_interp",
                        "matched_rates",
                    )
                },
                median_abs_on_comparable=np.median(overlaps) if overlaps else None,
            )
        )
    if len(rows) != 23:
        raise ValueError("expected the 23 clock-screened pairs from the frozen pilot")
    out.mkdir(exist_ok=True, parents=True)
    with (out / "cross_array.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tuple(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    report = {
        "clock_verified_groups": len(rows),
        "groups_ever_all_matching": sum(bool(r["all_four_matched"]) for r in rows),
        **{
            k: sum(r[k] for r in rows)
            for k in (
                "common_centres",
                "both_four",
                "all_four_matched",
                "both_four_no_interp",
                "all_four_matched_no_interp",
                "matched_rates",
            )
        },
        "tol_rps": TOL,
        "waveform_estimator": sm["estimator_sha256"],
        "scale_estimator": sm.get("scale_estimator_sha256"),
        "bridge_sha256": sm.get("bridge_sha256"),
    }
    (out / "cross_array.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("clocks", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    run(args.root, args.clocks, args.out)
