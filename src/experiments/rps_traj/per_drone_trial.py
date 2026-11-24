"""Compare one new joint likelihood against immutable per-drone baseline results.

Only the new arm's phase scores are recomputed. The optional full-path and
visual diagnostics always use the same sample support and first-draw rule.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.rps_traj import per_drone_data as data
from experiments.rps_traj import per_drone_paths as paths
from experiments.rps_traj.per_drone_run import ROOT
from experiments.rps_traj.unified_experiment import _sha256, _write_json
from plots.per_drone_flights import METHODS, RIG_NAMES, overview


def _levels(checks):
    per_record = defaultdict(list)
    for row in checks:
        per_record[row["recording"]].append([row["observed"], row["generated"]])
    if not per_record:
        return None
    pair = np.stack([np.mean(values, axis=0) for values in per_record.values()])
    obs, sim = pair.mean(axis=0)
    return dict(
        n_recordings=len(pair),
        observed=obs.tolist(),
        generated=sim.tolist(),
        collective_bias=float(np.mean(sim - obs)),
        motor_rms_bias=float(np.mean((sim - obs) ** 2) ** 0.5),
    )


def _physical_envelope(root: Path, rig: str, arm: str):
    from experiments.rps_traj.phases import block_mean, bridge_nans

    with np.load(root / "pack" / f"{rig}.train.npz") as training:
        x, offsets = training["x"], training["offsets"]
        ceiling = max(
            float(np.max(block_mean(bridge_nans(x[:, a:b]), 100.0, 25.0)))
            for a, b in zip(offsets[:-1], offsets[1:], strict=True)
        )
    high, total, observed_high, observed_n = 0, 0, 0, 0
    peak = 0.0
    with (
        np.load(root / "runs" / rig / f"{arm}.draws.npz") as draws,
        np.load(root / "pack" / f"{rig}.valid.npz") as measured,
    ):
        for key in measured.files:
            ensemble = draws[key]
            if not np.isfinite(ensemble).all() or ensemble.min() < 0:
                raise ValueError(f"{rig}/{arm}/{key}: nonphysical ensemble")
            maxima = ensemble.max(axis=(1, 2))
            peak = max(peak, float(maxima.max()))
            high += int((maxima > 1.25 * ceiling).sum())
            total += len(maxima)
            observed_high += int(np.nanmax(measured[key]) > 1.25 * ceiling)
            observed_n += 1
    return dict(
        train_max=ceiling,
        draw_max=peak,
        draws_over_125pct=high,
        draws=total,
        observations_over_125pct=observed_high,
        observations=observed_n,
    )


def _paired_intervals(rows, rig, phase, arm, baseline):
    """Across physical validation recordings; fixed fits, draws and comparator."""
    scores = {
        (r["recording"], r["arm"]): r["score"]
        for r in rows
        if r["rig"] == rig and r["phase"] == phase and r["status"] == "ok"
    }
    names = sorted(n for n, a in scores if a == arm and (n, baseline) in scores)
    gap = np.array([scores[n, baseline] - scores[n, arm] for n in names])
    ci = None
    if len(names) > 1:
        ids = np.random.default_rng(2026).integers(len(names), size=(8000, len(names)))
        ci = np.quantile(gap[ids].mean(axis=1), [0.025, 0.975]).tolist()
    return dict(winning_recordings=int((gap > 0).sum()), n_paired=len(names), gap_interval=ci)


def collect(root: Path, arm: str, label: str):
    out = root / "trials" / arm
    scorefile, _ = data.score(root, arms=(arm,), destination=out)
    old = json.loads((root / "scores.json").read_text())
    new = json.loads(scorefile.read_text())
    if (
        new["missing_runs"]
        or new["manifest_sha256"] != old["manifest_sha256"]
        or old["missing_runs"]
    ):
        raise ValueError("unpaired study or missing training units")
    arms = tuple(old["protocol"]["arms"])
    if arm in arms:
        raise ValueError("a new trial must not replace a locked comparison")
    joined = old["rows"] + new["rows"]
    merged = data.summarise(joined, {rig: (*arms, arm) for rig in RIG_NAMES})
    outcomes = {}
    table = []
    sizes = {}
    views = {}
    level_check = {}
    envelopes = {}
    cols = ("real", arm, "regulated_bridge", "template", "hsmm3")
    for rig, s in merged.items():
        byphase = {}
        for phase, p in s["phases"].items():
            if (
                p["failed"]
                or p["n_valid"] != old["summary"][rig]["phases"][phase]["n_valid"]
                or p["paired_recordings"]
                != old["summary"][rig]["phases"][phase]["paired_recordings"]
            ):
                raise ValueError(f"changed support: {rig}/{phase}: {p}")
            means = p["means"]
            strongest = min(arms, key=means.__getitem__)
            byphase[phase] = dict(
                n=p["n_valid"],
                score=means[arm],
                best_previous=strongest,
                best_previous_score=means[strongest],
                win=means[arm] < means[strongest],
                **_paired_intervals(joined, rig, phase, arm, strongest),
            )
            table.append(dict(rig=rig, phase=phase, n=p["n_valid"], **means))
        table.append(dict(rig=rig, phase="balanced", n=len(byphase), **s["balanced"]["means"]))
        outcomes[rig] = byphase
        fits = {
            a: json.loads((root / "runs" / rig / f"{a}.fit.json").read_text()) for a in (*arms, arm)
        }
        newfit = fits[arm]
        if _sha256(root / "runs" / rig / f"{arm}.draws.npz") != newfit["draws_sha256"]:
            raise ValueError(f"{rig}: stale draws")
        sizes[rig] = {a: int(f["n_params"]) for a, f in fits.items()}
        level_check[rig] = _levels(newfit["training_level_check"])
        envelopes[rig] = _physical_envelope(root, rig, arm)
        targets = json.loads((root / "pack" / f"{rig}.train.json").read_text())["targets"]
        candidates = sorted(
            enumerate(targets), key=lambda pair: (pair[1]["recording"], pair[1]["id"])
        )
        i, t = next(((i, t) for i, t in candidates if t["n"] >= 400), candidates[0])
        with np.load(root / "pack" / f"{rig}.valid.npz") as actual:
            traces = {"real": actual[f"t{i}"]}
        for a in cols[1:]:
            with np.load(root / "runs" / rig / f"{a}.draws.npz") as generated:
                traces[a] = generated[f"t{i}"][0]
        views[rig] = (t["id"], traces)
    with (out / "comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rig", "phase", "n", arm, *arms])
        w.writeheader()
        w.writerows(table)
    allcells = [v for g in outcomes.values() for v in g.values()]
    acceptance = dict(
        n_cells=len(allcells),
        wins=sum(v["win"] for v in allcells),
        smaller_than_all={r: p[arm] < min(p[a] for a in arms) for r, p in sizes.items()},
        all_phase_means_win=all(v["win"] for v in allcells),
    )
    _write_json(
        out / "acceptance.json",
        dict(
            arm=arm,
            phase_outcomes=outcomes,
            acceptance=acceptance,
            n_params=sizes,
            training_level_check=level_check,
            speed_envelopes=envelopes,
            baseline_sha256=_sha256(root / "scores.json"),
        ),
    )
    paths.score(root, arms=(*arms, arm), destination=out)
    overview(views, cols, METHODS | {arm: label}, out / "every-drone.png")
    print(json.dumps(acceptance, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--arm", default="lifting_control")
    p.add_argument("--label", default="Lifting control")
    args = p.parse_args()
    collect(args.root, args.arm, args.label)
