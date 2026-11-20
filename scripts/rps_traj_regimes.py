#!/usr/bin/env python3
"""Flight-regime campaign driver: fit, score, aggregate.

Thin CLI over :mod:`experiments.rps_traj.regime_campaign`; the write-up is
``docs/experiments/rps-flight-regimes.md``.  One unit = one JSON under
``<out>/raw`` (``utils.gridrun``: restartable, a failed unit leaves an
``.err`` and the rest go on).

    # held-out comparison of the cruise models, every rig
    python scripts/rps_traj_regimes.py run --stage cruise --jobs 4
    # whole flights on the ground-to-ground rigs (reads the cruise units)
    python scripts/rps_traj_regimes.py run --stage flight --jobs 4
    # the per-drone parameter sets: every flight, no held-out score
    python scripts/rps_traj_regimes.py run --stage cruise --final
    python scripts/rps_traj_regimes.py run --stage flight --final --arms ours
    # tables, figure and params/<rig>.json from whatever has run
    python scripts/rps_traj_regimes.py table
    # does the Gaussian fit run faster on a GPU?
    python scripts/rps_traj_regimes.py bench --rigs michaels pitcn_quad
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if (_HERE.parent / "src").is_dir() else Path.cwd().resolve()
sys.path.insert(0, str(_ROOT / "src"))

from experiments.rps_traj import regime_campaign as rc  # noqa: E402  (after the path pin)
from utils.gridrun import run_grid  # noqa: E402

DEFAULT_OUT = "results/rps_traj/regimes"


def _run(args: argparse.Namespace) -> int:
    rigs = args.rigs or list(rc.RIGS)
    if args.stage == "cruise":
        units = rc.cruise_units(
            rigs, args.families or rc.FAMILIES, final=args.final, threads=args.threads
        )
    else:
        units = rc.flight_units(
            rigs, args.arms or rc.FLIGHT_ARMS, args.out, final=args.final, threads=args.threads
        )
    if args.folds:
        keep = set(args.folds)
        units = [u for u in units if str(u.params["fold"]) in keep]
    result = run_grid(
        units,
        rc.run_unit,
        args.out,
        jobs=args.jobs,
        resume=not args.no_resume,
        blas_threads=args.threads,
        mp_context="spawn",
        summarize=lambda rows: {"n_units": len(rows)},
        summary_name="units.json",
    )
    return result.exit_code


def _figure(summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cruise = summary["cruise"]
    rigs = [r for r in rc.RIGS if r in cruise]
    keys = [
        "logpdf",
        "es_skill_0.2",
        "es_skill_1",
        "inc_w1_0.2",
        "inc_w1_1",
        "inc_kurt_diff_1",
        "acf_rmse",
        "psd_lsd_db",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.4))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, key in zip(axes.ravel(), keys):
        label, higher = rc.CRUISE_METRICS[key]
        for a, arm in enumerate(rc.CRUISE_ARMS):
            xs, ys, es = [], [], []
            for i, rig in enumerate(rigs):
                stat = cruise[rig].get(arm, {}).get("mean", {}).get(key)
                if not stat or not stat["n"]:
                    continue
                ref = cruise[rig]["gauss"]["mean"][key]["mean"] if key == "logpdf" else 0.0
                xs.append(i + (a - 3.5) * 0.09)
                ys.append(stat["mean"] - ref)
                es.append(stat["sd"] / max(np.sqrt(stat["n"]), 1.0))
            marker = "o" if a < 3 else "s"
            ax.errorbar(xs, ys, yerr=es, fmt=marker, ms=5, lw=0.8, color=colors[a], label=arm)
        title = label + (" (minus gauss)" if key == "logpdf" else "")
        ax.set_title(f"{title}\n{'higher' if higher else 'lower'} is better", fontsize=9)
        ax.set_xticks(range(len(rigs)))
        ax.set_xticklabels(
            [r.replace("_quad", "").replace("_cf21b", "") for r in rigs], rotation=25, fontsize=8
        )
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(
        "Held-out cruise scores: mean over folds, bar = standard error over folds", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _table(args: argparse.Namespace) -> int:
    out = Path(args.out)
    rows = [json.loads(p.read_text()) for p in sorted((out / "raw").glob("*.json"))]
    summary = rc.aggregate(rows)
    params = summary.pop("params")
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / "tables.md").write_text(rc.markdown(summary) + "\n")
    (out / "params").mkdir(exist_ok=True)
    for rig, payload in params.items():
        (out / "params" / f"{rig}.json").write_text(json.dumps(payload, indent=1) + "\n")
    if summary["cruise"]:
        _figure(summary, out / "cruise_scores.png")
    print(rc.markdown(summary))
    print(f"{len(rows)} units -> {out}/summary.json, tables.md, params/ ({len(params)} rigs)")
    return 0


def _bench(args: argparse.Namespace) -> int:
    import torch

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    rows = []
    for rig in args.rigs or ["michaels", "pitcn_quad"]:
        for device in devices:
            row = rc.bench_gaussian_fit(rig, device)
            row["threads"] = torch.get_num_threads()
            row["gpu"] = torch.cuda.get_device_name(0) if device == "cuda" else None
            rows.append(row)
            print(json.dumps(row), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench_device.json").write_text(json.dumps(rows, indent=1) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="fit and score units")
    run.add_argument("--stage", choices=("cruise", "flight"), required=True)
    run.add_argument("--rigs", nargs="*", default=None)
    run.add_argument("--families", nargs="*", default=None, help=f"cruise families {rc.FAMILIES}")
    run.add_argument("--arms", nargs="*", default=None, help=f"whole-flight arms {rc.FLIGHT_ARMS}")
    run.add_argument("--folds", nargs="*", default=None, help="only these folds (e.g. 0 1)")
    run.add_argument("--final", action="store_true", help="fit on every flight (fold 'all')")
    run.add_argument("--jobs", type=int, default=1, help="units in parallel")
    run.add_argument("--threads", type=int, default=1, help="threads inside one unit")
    run.add_argument("--no-resume", action="store_true")
    run.set_defaults(func=_run)
    table = sub.add_parser("table", help="aggregate the unit JSONs")
    table.set_defaults(func=_table)
    bench = sub.add_parser("bench", help="time the Gaussian fit's objective on CPU and GPU")
    bench.add_argument("--rigs", nargs="*", default=None)
    bench.set_defaults(func=_bench)
    for p in (run, table, bench):
        p.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
