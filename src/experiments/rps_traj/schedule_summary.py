"""Reduce the fixed-schedule experiment to independent-flight paired comparisons.

Run after `rps_traj_regimes.py schedule`: `python -m experiments.rps_traj.schedule_summary
[--controls-only] [--out results/rps_traj/regimes/schedule_eval]`.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.rps_traj.phase_dynamics import EXAMPLE_ARMS

PHASES = ("startup", "standby", "takeoff", "cruise", "landing")
SEED = 2026
REPLICATES = 10_000


def _load(path: Path) -> dict:
    report = json.loads(path.read_text())
    scores = {}
    for row in report["rows"]:
        k = row["arm"], row["rig"], row["flight"], row["phase"]
        if k in scores or row["score"] is None or not np.isfinite(row["score"]):
            raise ValueError(f"duplicate or unscored endpoint {k}: {row['status']}")
        scores[k] = row["score"]
    return scores


def controls(scores: dict) -> list[dict]:
    groups: dict = defaultdict(list)
    for (arm, rig, flight, phase), score in scores.items():
        if arm != "real":
            clean = scores["real", rig, flight, phase]
            groups[arm, phase, rig].append((clean, score - clean))
    results = []
    for (arm, phase, rig), pairs in sorted(groups.items()):
        clean, delta = np.array(pairs).T
        mad = np.median(np.abs(clean - np.median(clean)))
        results.append(
            dict(
                arm=arm,
                phase=phase,
                rig=rig,
                n=len(delta),
                worse=int(np.count_nonzero(delta > 0)),
                clean_mad=float(mad),
                median_penalty=float(np.median(delta)),
                max_absolute=float(abs(delta).max()),
                passes_effect_gate=bool(np.median(delta) > mad and np.mean(delta > 0) > 0.5),
            )
        )
    return results


def compare(scores: dict, reference: dict) -> dict:
    arms = (*EXAMPLE_ARMS, "real")
    targets = sorted({(r, f) for a, r, f, p in reference if a == "real"})
    rigs = sorted({r for r, _ in targets})
    if len(targets) != 13 or len(rigs) != 3:
        raise ValueError(f"incomplete measured flight set: {targets}")
    expected = {(a, r, f, p) for a in EXAMPLE_ARMS for r, f in targets for p in PHASES}
    if set(scores) != expected:
        raise ValueError(
            f"non-comparable panels: missing {expected - set(scores)}, extra {set(scores) - expected}"
        )
    combined = scores | {k: v for k, v in reference.items() if k[0] == "real"}
    x = np.array([[[combined[a, r, f, p] for a in arms] for p in PHASES] for r, f in targets])
    # No weighting by number of samples, origins, ensemble size or number of flights on each rig.
    ids = [np.array([i for i, (r, _) in enumerate(targets) if r == rig]) for rig in rigs]
    by_rig = np.stack([x[ix].mean(axis=0) for ix in ids])
    macro = by_rig.mean(axis=0)
    flight = x.mean(axis=1)
    rng = np.random.default_rng(SEED)
    boot = np.stack(
        [flight[rng.choice(ix, size=(REPLICATES, len(ix)))].mean(axis=1) for ix in ids]
    ).mean(axis=0)
    paired = {}
    for k, arm in enumerate(arms[1:], 1):
        delta = flight[:, k] - flight[:, 0]
        ci = np.quantile(boot[:, k] - boot[:, 0], (0.025, 0.975))
        phase_delta = macro[:, k] - macro[:, 0]
        paired[arm] = dict(
            delta=float(macro[:, k].mean() - macro[:, 0].mean()),
            ci95=ci.tolist(),
            wins=int(np.sum(delta > 0)),
            ties=int(np.sum(delta == 0)),
            leave_phase_out={
                p: float(np.delete(phase_delta, j).mean()) for j, p in enumerate(PHASES)
            },
            rig_delta={r: float(delta[ix].mean()) for r, ix in zip(rigs, ids)},
        )
    return {
        "order": list(arms),
        "phases": list(PHASES),
        "rigs": rigs,
        "n": len(targets),
        "bootstrap": {
            "replicates": REPLICATES,
            "seed": SEED,
            "unit": "paired flight within rig; fixed fitted folds",
        },
        "overall": dict(zip(arms, macro.mean(axis=0).tolist())),
        "by_phase": {p: dict(zip(arms, row.tolist())) for p, row in zip(PHASES, macro)},
        "by_rig": {
            r: dict(zip(arms, table.mean(axis=0).tolist())) for r, table in zip(rigs, by_rig)
        },
        "paired_baseline_minus_ours": paired,
        "by_flight": [
            dict(rig=r, flight=f, scores=dict(zip(arms, values.tolist())))
            for (r, f), values in zip(targets, flight)
        ],
    }


def _paper_table(c: dict) -> str:
    """Static TeX rows: no fitting or score computation during the paper build."""
    labels = {
        "ours": "Proposed",
        "phasevar2": "Phase VAR(2)",
        "phasevar4": "Phase VAR(4)",
        "phasevar2+msvar3": "VAR(2) + MS-VAR",
        "phasevar2+hsmm3": "VAR(2) + VAR-HSMM",
        "real": "Other real flights",
    }
    rows = [
        r"\begin{tabular}{@{}lrrrrrrrc@{}}",
        r"\toprule",
        r"Method & Start-up & Standby & Take-off & Cruise & Landing & Balanced"
        r" & $\Delta$ [95\% interval] & Wins\\",
        r"\midrule",
    ]
    for arm in c["order"]:
        scores = [c["by_phase"][p][arm] for p in PHASES] + [c["overall"][arm]]
        cells = [labels[arm], *(f"{s:.2f}" for s in scores)]
        if arm in ("ours", "real"):
            cells.extend(["---", "---"])
        else:
            paired = c["paired_baseline_minus_ours"][arm]
            lo, hi = paired["ci95"]
            cells.extend(
                [f"{paired['delta']:.2f} [{lo:.2f}, {hi:.2f}]", f"{paired['wins']}/{c['n']}"]
            )
        rows.append(" & ".join(cells) + r" \\")
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/rps_traj/regimes/schedule_eval"))
    parser.add_argument("--controls-only", action="store_true")
    args = parser.parse_args()
    reference = _load(args.out / "schedule_controls.json")
    summary: dict = {"controls": controls(reference)}
    if not args.controls_only:
        summary["comparison"] = compare(_load(args.out / "schedule_models.json"), reference)
        (args.out / "schedule-table.tex").write_text(_paper_table(summary["comparison"]))
        import matplotlib.pyplot as plt

        from experiments.rps_traj.phase_dynamics import FS
        from plots.generator_paper import schedule_example

        with np.load(args.out / "example_michaels_FLY124.npz") as example:
            fig = schedule_example(dict(example), FS)
        for ext in ("pdf", "png"):
            fig.savefig(args.out / f"schedule-example.{ext}", dpi=200)
        plt.close(fig)
    path = args.out / ("controls_summary.json" if args.controls_only else "summary.json")
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
