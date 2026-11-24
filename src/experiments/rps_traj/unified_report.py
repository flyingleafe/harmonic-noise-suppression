"""Read held-out scores and generate a like-for-like whole-flight comparison.

python -m experiments.rps_traj.unified_report [--baseline-traces-only]
No fitting of candidate models here. The old illustration replay uses the
archived training fold, seed and draw count, never a best-of-ensemble draw.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.rps_traj.phase_dynamics import EXAMPLE_ARMS, fit_arm
from experiments.rps_traj.schedule_eval import FS, PHASES, _cut_generated, plan

NEW = ("unified_hierarchical", "unified_flat")
NAMES = {
    "ours": "Template + switching cruise",
    "phasevar2": "Phase VAR(2)",
    "phasevar4": "Phase VAR(4)",
    "phasevar2+msvar3": "VAR(2) + MS-VAR",
    "phasevar2+hsmm3": "VAR(2) + VAR-HSMM",
    "unified_hierarchical": "Unified hierarchical",
    "unified_flat": "Unified flat",
}
RIGS = ("michaels", "vid_m100", "dregon")


def reference_panels(out: Path, campaign: Path) -> None:
    record = json.loads((campaign / "schedule_eval/schedule_models.json").read_text())
    frozen = {(r["rig"], r["flight"], r["phase"], r["arm"]): r for r in record["rows"]}
    panels: dict[str, np.ndarray] = {}
    for rig in RIGS:
        splits, _ = plan(rig, campaign, require_caches=True)
        for fp in splits:
            for target in fp.targets:
                cells = [g for g in fp.groups if g.flight == target.name]
                for arm in EXAMPLE_ARMS:
                    row = frozen[rig, target.name, "cruise", arm]
                    generator = fit_arm(arm, rig, fp.fold, str(campaign), 0, same_clock=True)
                    paths = generator(
                        target.marks,
                        row["n_draws"],
                        np.random.default_rng(np.random.SeedSequence(row["seed"])),
                    )
                    if paths is None or len(paths) != row["n_draws"]:
                        raise ValueError(f"incomplete ensemble: {rig} {target.name} {arm}")
                    for g in cells:
                        draws, status, _ = _cut_generated(paths[:1], target, g)
                        if draws is None or status != "ok":
                            raise ValueError((rig, target.name, g.phase, arm, status))
                        origin = len(g.starts) // 2
                        k = f"{rig}__{fp.fold}__{target.name}__{g.phase}__"
                        panels[k + arm] = draws[origin][0].astype("float32")
                        if arm == EXAMPLE_ARMS[0]:
                            panels[k + "real"] = g.obs[origin].astype("float32")
                            panels[k + "start"] = np.array(g.starts[origin] / FS)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "baseline_draw0.npz", allow_pickle=False, **panels)


def compare(out: Path, campaign: Path, new_arms: tuple[str, ...] = NEW) -> tuple[dict, list[dict]]:
    old_path = campaign / "schedule_eval/schedule_models.json"
    new_path = out / "unified_models.json"
    old = json.loads(old_path.read_text())["rows"]
    new = json.loads(new_path.read_text())["rows"]
    arms = (*new_arms, *EXAMPLE_ARMS)
    keys = sorted({(r["rig"], r["fold"], r["flight"]) for r in old})
    if len(keys) != 13:
        raise ValueError(f"expected 13 complete measured targets, found {keys}")
    scores = {}
    for r in old + new:
        k = (r["rig"], r["fold"], r["flight"], r["arm"], r["phase"])
        if k in scores or r["status"] != "ok" or r["n_draws"] != 32 or not np.isfinite(r["score"]):
            raise ValueError(f"unusable/duplicate cell {k}: {r['status']}")
        scores[k] = r["score"]
    if len(scores) != len(keys) * len(arms) * len(PHASES):
        raise ValueError("incomplete comparison")
    x = np.array([[[scores[*k, arm, phase] for phase in PHASES] for arm in arms] for k in keys])
    indices = [np.flatnonzero([k[0] == rig for k in keys]) for rig in RIGS]
    byrig = np.stack([x[ix].mean(axis=0) for ix in indices])
    macro = byrig.mean(axis=0)
    rng = np.random.default_rng(2026)
    boot = np.mean([x[rng.choice(ix, (10000, len(ix)))].mean(axis=1) for ix in indices], axis=0)

    def endpoint(j, col):
        delta = col(boot[:, j]) - col(boot[:, 0])
        return {
            "baseline_minus_new": float(col(macro[j]) - col(macro[0])),
            "ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
            "wins_out_of_13": int(np.count_nonzero(col(x[:, 0]) < col(x[:, j]))),
        }

    components = {}
    for phase in PHASES:
        readouts = (
            ("level_es_rps", "spectral_es")
            if phase == "cruise"
            else ("path_es_rps", "summary_es_rps")
        )
        components[phase] = {
            metric: {
                arm: float(
                    np.mean(
                        np.asarray(
                            [
                                np.mean(
                                    [
                                        r["diagnostics"][metric]
                                        for r in old + new
                                        if (r["phase"], r["arm"], r["rig"]) == (phase, arm, rig)
                                    ]
                                )
                                for rig in RIGS
                            ]
                        )
                    )
                )
                for arm in arms
            }
            for metric in readouts
        }
    pairs = {
        arm: {
            **{p: endpoint(j, lambda v, i=i: v[..., i]) for i, p in enumerate(PHASES)},
            "balanced": endpoint(j, lambda v: v.mean(axis=-1)),
        }
        for j, arm in enumerate(arms)
        if j
    }
    result = {
        "arms": list(arms),
        "rigs": list(RIGS),
        "phases": list(PHASES),
        "score_by_phase": {p: dict(zip(arms, col.tolist())) for p, col in zip(PHASES, macro.T)},
        "component_scores_by_phase": components,
        "overall": dict(zip(arms, macro.mean(axis=1).tolist())),
        "by_rig": {
            rig: {p: dict(zip(arms, col.tolist())) for p, col in zip(PHASES, table.T)}
            for rig, table in zip(RIGS, byrig)
        },
        "paired": pairs,
        "beats_all_historical_by_phase": dict(
            zip(PHASES, (macro[0] < macro[len(new_arms) :].min(axis=0)).tolist())
        ),
        "source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (new_path, old_path)
        },
    }
    return result, [dict(rig=k[0], fold=k[1], flight=k[2]) for k in keys]


def parameter_counts(out: Path, campaign: Path, targets: list[dict]) -> dict:
    rows = {}
    manifest = json.loads((out / "pack/manifest.json").read_text())
    for t in targets:
        k = f"{t['rig']}__{t['fold']}"
        baseline = {
            arm: v["n_params"] for arm, v in manifest["folds"][k]["baseline_param_counts"].items()
        }
        for arm in NEW:
            unit = json.loads(
                (out / "fits" / f"{k}__{arm.removeprefix('unified_')}.json").read_text()
            )
            baseline[arm] = unit["n_params"]
        rows[k] = baseline
    return rows


def make_visuals(
    out: Path,
    targets: list[dict],
    new_arms: tuple[str, ...] = NEW,
    names: dict[str, str] | None = None,
) -> None:
    from plots.unified_flights import render

    with np.load(out / "baseline_draw0.npz") as base:
        cells = []
        for t in targets:
            rig, fold, flight = t["rig"], t["fold"], t["flight"]
            prefix = f"{rig}__{fold}__{flight}__"
            paths = {
                p: {arm: base[prefix + p + "__" + arm] for arm in ("real", *EXAMPLE_ARMS)}
                for p in PHASES
            }
            for arm in new_arms:
                k = f"{rig}__{fold}__{arm.removeprefix('unified_')}"
                with np.load(out / "draws" / f"{k}.npz") as draws:
                    for p in PHASES:
                        w = draws[f"{flight}__{p}__draws"]
                        paths[p][arm] = w[len(w) // 2, 0]
            for p in PHASES:
                cells.append((rig, flight, p, float(base[prefix + p + "__start"]), paths[p]))
    render(cells, NAMES if names is None else names, out, FS, primary=new_arms[0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results/rps_traj/unified"))
    p.add_argument("--campaign", type=Path, default=Path("results/rps_traj/regimes"))
    p.add_argument("--baseline-traces-only", action="store_true")
    args = p.parse_args()
    if args.baseline_traces_only:
        reference_panels(args.out, args.campaign)
        return
    from experiments.rps_traj.unified_diagnostics import diagnose

    summary, targets = compare(args.out, args.campaign)
    summary["parameter_counts"] = parameter_counts(args.out, args.campaign, targets)
    summary["diagnostics"] = diagnose(args.out, targets)
    make_visuals(args.out, targets)
    (args.out / "comparison.json").write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: summary[k] for k in ("score_by_phase", "overall", "beats_all_historical_by_phase")},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
