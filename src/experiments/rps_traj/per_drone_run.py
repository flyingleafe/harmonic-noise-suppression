"""Run one restartable unit per (drone, generator) on training-only packs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.rps_traj.data import RIGS
from experiments.rps_traj.unified_experiment import _sha256
from utils.gridrun import Unit, add_gridrun_args, gridrun_from_args

ARMS = ("regulated_bridge", "template", "phasevar2", "phasevar4", "msvar3", "hsmm3")
ROOT = Path("results/rps_traj/per_drone")


def worker(unit: Unit) -> dict:
    p = unit.params
    root, rig, arm = Path(str(p["root"])), str(p["rig"]), str(p["arm"])
    manifest = json.loads((root / "pack" / f"{rig}.train.json").read_text())
    if arm in ("regulated_bridge", "lifting_control"):
        from experiments.rps_traj.per_drone_bridge import fit

        fit(root, rig, arm=arm)
    else:
        from experiments.rps_traj.per_drone_baselines import fit

        fit(root, rig, arm)
    path = root / "runs" / rig / f"{arm}.fit.json"
    result = json.loads(path.read_text())
    if result["train_sha256"] != manifest["train_sha256"] or list(result["train_ids"]) != [
        t["id"] for t in manifest["train"]
    ]:
        raise ValueError(f"{rig}/{arm} did not use its declared training set")
    draws = path.with_name(arm + ".draws.npz")
    return dict(
        rig=rig,
        arm=arm,
        n_train=len(manifest["train"]),
        n_targets=len(manifest["targets"]),
        n_params=result["n_params"],
        train_sha256=result["train_sha256"],
        fit_sha256=_sha256(path),
        draws_sha256=_sha256(draws),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--rigs", nargs="+", choices=RIGS, default=list(RIGS))
    parser.add_argument("--arms", nargs="+", choices=(*ARMS, "lifting_control"), default=list(ARMS))
    add_gridrun_args(parser, jobs=4)
    args = parser.parse_args()
    # Incomplete reruns go to the same immutable split's artifact directory.
    units = [
        Unit(f"{rig}__{arm}", dict(rig=rig, arm=arm, root=str(args.root)))
        for rig in args.rigs
        for arm in args.arms
    ]
    result = gridrun_from_args(args, units, worker, args.root / "units")
    raise SystemExit(result.exit_code)


if __name__ == "__main__":
    main()
