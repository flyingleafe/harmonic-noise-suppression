"""Compute paired evidence, not a winner by mixing arms across phases."""

import json
from pathlib import Path

import numpy as np

from experiments.rps_traj import unified_experiment as ue


def audit(out: Path, result: dict, families: list[str], targets: list[dict]) -> dict:
    phases = result["phases"]
    new = json.loads((out / "unified_models.json").read_text())["rows"]
    oldfile = Path(json.loads((out / "pack/manifest.json").read_text())["baseline"]["models"])
    old = json.loads(oldfile.read_text())["rows"]
    arms0 = sorted({r["arm"] for r in old})
    strongest = [min(arms0, key=result["score_by_phase"][p].__getitem__) for p in phases]
    keyed = {(r["rig"], r["flight"], r["phase"], r["arm"]): r for r in new + old}
    index = [np.flatnonzero([t["rig"] == rig for t in targets]) for rig in result["rigs"]]
    rng = np.random.default_rng(2026)
    boot_ids = [rng.choice(ix, (10000, len(ix)), replace=True) for ix in index]
    evidence = {}
    for family in families:
        arm = "unified_" + family
        gaps = np.array(
            [
                [
                    keyed[t["rig"], t["flight"], p, b]["score"]
                    - keyed[t["rig"], t["flight"], p, arm]["score"]
                    for p, b in zip(phases, strongest, strict=True)
                ]
                for t in targets
            ]
        )
        boot = np.mean([gaps[ix].mean(axis=1) for ix in boot_ids], axis=0)
        by_phase = {}
        for i, (p, b) in enumerate(zip(phases, strongest, strict=True)):
            gap = result["score_by_phase"][p][b] - result["score_by_phase"][p][arm]
            by_phase[p] = dict(
                strongest=b,
                baseline_minus_new=gap,
                ci95=np.quantile(boot[:, i], [0.025, 0.975]).tolist(),
                wins=int(np.count_nonzero(gaps[:, i] > 0)),
                rigs_won=sum(bool(gaps[ix, i].mean() > 0) for ix in index),
            )
        counts, extremes, iteration_counts = [], [], []
        for t in targets:
            k = t["rig"] + "__" + t["fold"]
            model = json.loads((out / "fits" / f"{k}__{family}.json").read_text())
            counts.append(model["n_params"])
            iteration_counts.append(len(model["history"]))
            for d in model["draws"]:
                if d["flight"] == t["flight"]:
                    extremes.append((d["minimum_rps"], d["maximum_rps"]))
        evidence[family] = dict(
            phases=by_phase,
            n_params_range=[min(counts), max(counts)],
            history_lengths=iteration_counts,
            minimum_rps=float(np.min(extremes, axis=0)[0]),
            maximum_rps=float(np.max(extremes, axis=0)[1]),
            draws_per_flight=ue.N_DRAWS,
            all_phase_means_pass=all(v["baseline_minus_new"] > 0 for v in by_phase.values()),
        )
    return evidence
