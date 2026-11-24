"""Supplement the frozen phase scores with the joint ordered-path score on all durations.

Every method sees precisely the same time points. A phase outside the support
of a comparator is missing, not zero. This raw rev/s score does not substitute
for the phase-insensitive cruise spectral score or for its train-only scaling.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from experiments.rps_traj.per_drone_run import ARMS, ROOT
from experiments.rps_traj.scores import energy_score
from experiments.rps_traj.unified_experiment import _write_json


def score(root: Path, *, arms=ARMS, destination: Path | None = None) -> None:
    destination = root if destination is None else destination
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "pack/manifest.json").read_text())
    summary = []
    detail = []
    for rig in manifest["protocol"]["rigs"]:
        meta = json.loads((root / "pack" / f"{rig}.valid.json").read_text())
        by_flight = defaultdict(list)
        total = used = 0
        with ExitStack() as stack:
            real = stack.enter_context(np.load(root / "pack" / f"{rig}.valid.npz"))
            gens = {
                arm: stack.enter_context(np.load(root / "runs" / rig / f"{arm}.draws.npz"))
                for arm in arms
            }
            for i, t in enumerate(meta["targets"]):
                key = f"t{i}"
                obs = real[key]
                draws = {a: f[key] for a, f in gens.items()}
                common = np.isfinite(obs).all(axis=0)
                for x in draws.values():
                    if x.shape != (32, 4, obs.shape[-1]) or np.isinf(x).any():
                        raise ValueError(f"{rig}/{t['id']}: wrong shape or divergent path")
                    common &= np.isfinite(x).all(axis=(0, 1))
                n = int(common.sum())
                used += n
                total += len(common)
                row = dict(
                    rig=rig,
                    recording=t["recording"],
                    id=t["id"],
                    samples=len(common),
                    common_samples=n,
                )
                if n:
                    y = np.ascontiguousarray(obs[:, common]).ravel()
                    row["scores"] = {
                        a: energy_score(np.ascontiguousarray(x[:, :, common]).reshape(32, -1), y)
                        / np.sqrt(len(y))
                        for a, x in draws.items()
                    }
                    by_flight[t["recording"]].append(row["scores"])
                detail.append(row)
        if not by_flight:
            raise ValueError(f"{rig} has no mutually supported validation samples")
        across = [{a: np.mean([v[a] for v in rows]) for a in arms} for rows in by_flight.values()]
        summary.append(
            dict(
                rig=rig,
                n_valid=len(by_flight),
                common_samples=used,
                total_samples=total,
                **{a: float(np.mean(np.asarray([r[a] for r in across]))) for a in arms},
            )
        )
    _write_json(destination / "whole_path_details.json", detail)
    with (destination / "whole_path_scores.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    score(p.parse_args().root)
