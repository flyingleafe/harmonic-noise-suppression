#!/usr/bin/env python
"""Tabulate the validation rounds of rp_* runs (docs/experiments/paper-real-protocol.md).

Reads ``results/<experiment>/validation_history.jsonl`` (pulled with ``omnirun
pull``) and prints one row per round: optimizer step, LR, the control score
(``real``), its 5-round median, the per-dataset views and the stop verdict.
``--best`` prints only the best-median round per run, next to the stopping
step, which is what the result tables quote.

    python scripts/rp_history.py rp_scv2_all rp_hppnet_l2_all
    python scripts/rp_history.py --best 'rp_*'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VIEWS = (
    "real",
    "all",
    "dregon",
    "md2",
    "all_nosource",
    "all_source",
    "all_zero",
    "all_standby",
    "all_ramp",
    "all_cruise",
)


def rounds(experiment: str) -> list[dict]:
    path = REPO_ROOT / "results" / experiment / "validation_history.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wandb_rounds(run_id: str) -> tuple[str, list[dict]]:
    """Rounds of a (possibly still running) run straight from W&B."""
    import wandb

    run = wandb.Api().run(f"flyingleafe/harmonic-noise-suppression/{run_id}")
    keys = ["optimizer_step", "lr", "val/real_median", "validation/stop_reason"] + [
        f"val/{v}" for v in VIEWS
    ]
    return f"{run.name} [{run.state}]", list(run.scan_history(keys=keys))


def fmt(row: dict, key: str) -> str:
    value = row.get(key)
    return f"{value:6.2f}" if isinstance(value, (int, float)) else "     -"


def print_run(experiment: str, rows: list[dict], best_only: bool) -> None:
    if not rows:
        print(f"{experiment}: no validation_history.jsonl")
        return
    header = f"{'step':>7} {'lr':>8} " + " ".join(f"{v[:10]:>10}" for v in VIEWS) + "  real_med"
    print(f"== {experiment} ({len(rows)} rounds)")
    print(header)
    chosen = rows
    if best_only:
        scored = [r for r in rows if isinstance(r.get("val/real_median"), (int, float))]
        chosen = [min(scored, key=lambda r: r["val/real_median"])] if scored else []
    for r in chosen:
        views = " ".join(f"{fmt(r, f'val/{v}'):>10}" for v in VIEWS)
        mark = " <- best median" if best_only else ""
        print(
            f"{int(r['optimizer_step']):7d} {r['lr']:8.2e} {views}  {fmt(r, 'val/real_median')}{mark}"
        )
    last = rows[-1]
    stop = last.get("validation/stop_reason")
    print(
        f"   last round at step {int(last['optimizer_step'])}"
        + (f"; stopped: {stop}" if stop else "; running")
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("experiments", nargs="*", help="experiment names or globs over results/")
    p.add_argument("--best", action="store_true", help="only the best-median round per run")
    p.add_argument("--wandb", action="append", default=[], help="W&B run id (running jobs)")
    args = p.parse_args()
    if not args.experiments and not args.wandb:
        p.error("give experiment names or --wandb <run id>")
    names: list[str] = []
    for pattern in args.experiments:
        if any(ch in pattern for ch in "*?["):
            names += sorted(d.name for d in (REPO_ROOT / "results").glob(pattern) if d.is_dir())
        else:
            names.append(pattern)
    for name in names:
        print_run(name, rounds(name), args.best)
    for run_id in args.wandb:
        label, rows = wandb_rounds(run_id)
        print_run(label, rows, args.best)


if __name__ == "__main__":
    main()
