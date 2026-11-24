"""Event-conditioned likelihood models on the fixed whole-flight benchmark.

Use `prepare` on the laptop, `fit` on uni-cpu, and `collect` after pulling the
fits/draws. Only the fold's training NPZ and schedule JSON are needed to fit.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np

from experiments.rps_traj import unified_experiment as ue

ROOT = Path("results/rps_traj/event_response")
PREVIOUS = Path("results/rps_traj/unified")
CAMPAIGN = Path("results/rps_traj/regimes")
FAMILIES = {
    "clock_mixed": "Mixed-effect clock AR",
    "program_response": "Program-response HSMM",
    "bridge_program": "Multiscale programme bridge",
    "settled_bridge": "Settling-time programme bridge",
    "population_bridge": "Population feedback bridge",
    "bounded_bridge": "Nonamplifying feedback bridge",
    "regulated_bridge": "Regulated feedback bridge",
    "allocation_bridge": "Collective and allocation bridge",
    "controller_law": "Hierarchical feedback controller",
}


def prepare(out: Path, previous: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in ("pack", "baseline_draw0.npz"):
        src = (previous / name).resolve(strict=True)
        dest = out / name
        if not dest.exists():
            dest.symlink_to(src, target_is_directory=src.is_dir())
        elif not dest.samefile(src):
            raise ValueError(f"{dest} does not point to the frozen {src}")
    manifest = json.loads((out / "pack/manifest.json").read_text())
    if manifest["verification"]["cells"] != 65 or manifest["verification"]["targets"] != 13:
        raise ValueError("incomplete donor pack")


def fit_unit(out: Path, rig: str, fold: str, family: str, max_iter: int) -> None:
    module = importlib.import_module(f"experiments.rps_traj.{family}")
    if module.__file__ is None:
        raise ValueError(f"{family} is not a file-backed model")
    paths = ue._paths(out, rig, fold, family)
    meta = json.loads(paths["meta"].read_text())
    if ue._sha256(paths["train"]) != meta["train_sha256"]:
        raise ValueError(f"{rig}/{fold}: stale training input")
    with np.load(paths["train"]) as z:
        x, ph, offsets = z["train_x"], z["train_phase"], z["train_offsets"]
    tracks = [(x[:, a:b], ph[a:b]) for a, b in zip(offsets[:-1], offsets[1:])]
    measured = [r["measured"] for r in meta["train"]]
    if len(tracks) != len(measured):
        raise ValueError("source/track mismatch")
    start = time.monotonic()
    model = module.fit(tracks, measured=measured, max_iter=max_iter)
    fit_seconds = time.monotonic() - start
    stored, draws = {}, []
    for t in meta["targets"]:
        flight = t["flight"]
        schedule = ue.schedule_codes(t["samples"], t["events"])
        paths_drawn = np.asarray(
            model.sample(
                schedule, ue.N_DRAWS, np.random.default_rng(np.random.SeedSequence(t["seed"]))
            )
        )
        if (
            paths_drawn.shape != (ue.N_DRAWS, 4, len(schedule))
            or not np.isfinite(paths_drawn).all()
        ):
            raise ValueError(f"{rig}/{fold}/{family}: invalid ensemble")
        stored[f"{flight}__paths"] = paths_drawn.astype(np.float32)
        stored[f"{flight}__phase"] = schedule
        for cell in t["cells"]:
            stored[f"{flight}__{cell['phase']}__draws"] = np.stack(
                [paths_drawn[:, :, a : a + cell["n"]] for a in cell["starts"]]
            )
        draws.append(
            dict(
                flight=flight,
                seed=t["seed"],
                status="ok",
                minimum_rps=float(paths_drawn.min()),
                maximum_rps=float(paths_drawn.max()),
                negative_fraction=float((paths_drawn < 0).mean()),
            )
        )
    ue._write_npz(paths["draws"], stored)
    ue._write_json(
        paths["fit"],
        dict(
            rig=rig,
            fold=fold,
            arm=family,
            n_params=int(model.n_params),
            params=model.to_dict(),
            history=np.asarray(model.history).tolist(),
            fit_seconds=fit_seconds,
            train_sha256=meta["train_sha256"],
            source_sha256=ue._sha256(Path(module.__file__)),
            draws=draws,
            draws_sha256=ue._sha256(paths["draws"]),
        ),
    )
    print(
        rig,
        fold,
        family,
        "params",
        model.n_params,
        "fit_seconds",
        round(fit_seconds, 2),
        flush=True,
    )


def collect(out: Path, families: list[str]) -> None:
    from experiments.rps_traj import unified_report as report
    from experiments.rps_traj.clock_audit import audit
    from plots.flight_review import review_plates

    manifest = json.loads((out / "pack/manifest.json").read_text())
    if (
        ue._sha256(CAMPAIGN / "schedule_eval/schedule_models.json")
        != manifest["baseline"]["models_sha256"]
    ):
        raise ValueError("baseline record no longer matches the pack")
    rows, sizes = [], {}
    for key in manifest["folds"]:
        rig, fold = key.split("__")
        sizes[key] = {}
        for family in families:
            row_record = json.loads(ue.score_unit(out, rig, fold, family).read_text())
            rows.extend(row_record["rows"])
            sizes[key][family] = row_record["rows"][0]["n_params"]
    if len(rows) != 65 * len(families) or not all(
        r["donor_scales_match"] and r["status"] == "ok" for r in rows
    ):
        raise ValueError("incomplete results or altered reference scales")
    ue._write_json(out / "unified_models.json", dict(rows=rows, parameter_counts=sizes))
    arms = tuple(f"unified_{f}" for f in families)
    result, targets = report.compare(out, CAMPAIGN, arms)
    result["parameter_counts"] = sizes
    historic_arms = list(report.EXAMPLE_ARMS)
    best = {p: min(s[a] for a in historic_arms) for p, s in result["score_by_phase"].items()}
    result["acceptance"] = {
        f: {p: result["score_by_phase"][p][a] < best[p] for p in best}
        for f, a in zip(families, arms, strict=True)
    }
    result["max_parameters"] = {f: max(s[f] for s in sizes.values()) for f in families}
    result["paired_acceptance"] = audit(out, result, families, targets)
    names = {k: report.NAMES[k] for k in historic_arms} | {
        a: FAMILIES[f] for f, a in zip(families, arms, strict=True)
    }
    report.make_visuals(out, targets, arms, names)
    review_plates(out, out, families, targets)
    ue._write_json(out / "comparison.json", result)
    print(
        json.dumps(
            {k: result[k] for k in ("score_by_phase", "overall", "acceptance", "max_parameters")},
            indent=1,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--previous", type=Path, default=PREVIOUS)
    fit = subs.add_parser("fit")
    fit.add_argument("--rig", required=True)
    fit.add_argument("--fold", required=True)
    fit.add_argument("--family", choices=FAMILIES, required=True)
    fit.add_argument("--max-iter", type=int, default=80)
    score = subs.add_parser("collect")
    score.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    for cmd in (prep, fit, score):
        cmd.add_argument("--out", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.out, args.previous)
    elif args.command == "fit":
        fit_unit(args.out, args.rig, args.fold, args.family, args.max_iter)
    else:
        collect(args.out, args.families)


if __name__ == "__main__":
    main()
