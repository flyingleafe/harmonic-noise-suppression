"""Score training arms from their live R2 validation history (v2 transfer protocol).

For each experiment name, fetches
``r2://ml-data/artifacts/<exp>/checkpoints/{validation_history.jsonl,best_checkpoints.json}``
(always refetched: a running arm appends a row every validation round) and prints
one row in the column order of ``docs/experiments/noise-v2-transfer.md``
§ "Stage-1 and curriculum results":

    rounds | sel | smoothed | raw @ sel | best raw | r1 | r2 | r3 | r1/r2 | wall h

* ``sel`` is the validation round ``best_real_overall.ckpt`` was written at
  (selection is on the SMOOTHED ``real_overall``); ``raw @ sel`` is that round's
  unsmoothed value and is the number to quote.
* ``best raw`` is the lowest raw ``val/real_overall`` anywhere in the run (no
  checkpoint corresponds to it). ``r1``/``r2``/``r3`` are the per-view scores at sel.
* ``wall h`` is first train start to last validation end, from the ``perf/*`` rows.

Checked against ``nv2_easy_scv2``: 102 rounds, sel 81, 7.94 / 7.94 / best raw 7.18,
r1 10.89, r2 9.67.

Usage (with ``.env`` sourced for the R2 credentials)::

    PYTHONPATH=src python scripts/_nv3_arm_history.py nv3_easy_scv2 nv3_hard_scv2 [--json out.json]
"""

import argparse
import json
import tempfile
from pathlib import Path

from utils.checkpoints import resolve_checkpoint_uri

KEY = "val/real_overall"


def fetch(exp: str, cache: Path) -> tuple[list[dict], dict]:
    root = f"r2://ml-data/artifacts/{exp}/checkpoints"
    dest = cache / exp
    dest.mkdir(parents=True, exist_ok=True)
    local = resolve_checkpoint_uri(f"{root}/validation_history.jsonl", dest)
    rows = [json.loads(line) for line in Path(local).read_text().splitlines() if line.strip()]
    try:
        best = json.loads(
            Path(resolve_checkpoint_uri(f"{root}/best_checkpoints.json", dest)).read_text()
        )
    except Exception as exc:  # noqa: BLE001 - arm may not have written a best yet
        print(f"{exp}: no best_checkpoints.json ({exc})")
        best = {}
    return rows, best


def score(rows: list[dict], best: dict) -> dict:
    raw = [r[KEY] for r in rows if r.get(KEY) is not None]
    sel = best.get("real_overall", {})
    scores = sel.get("scores", {})
    starts = [r["perf/train_started_unix"] for r in rows if "perf/train_started_unix" in r]
    ends = [
        r["perf/validation_finished_unix"] for r in rows if "perf/validation_finished_unix" in r
    ]
    r1, r2 = scores.get("real_r1"), scores.get("real_r2")
    return {
        "rounds": len(rows),
        "sel": sel.get("validation_round"),
        "sel_step": sel.get("optimizer_step"),
        "smoothed": sel.get("smoothed"),
        "raw_at_sel": sel.get("raw"),
        "best_raw": min(raw) if raw else None,
        "best_raw_round": raw.index(min(raw)) if raw else None,
        "last_raw": raw[-1] if raw else None,
        "r1": r1,
        "r2": r2,
        "r3": scores.get("real_r3"),
        "r1_over_r2": r1 / r2 if r1 and r2 else None,
        "lr": rows[-1].get("lr") if rows else None,
        "wall_h": (max(ends) - min(starts)) / 3600 if starts and ends else None,
    }


def fmt(v, spec=".2f"):
    return "—" if v is None else format(v, spec) if isinstance(v, float) else str(v)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score training arms from their R2 validation history."
    )
    ap.add_argument("exps", nargs="+")
    ap.add_argument("--json", type=Path, help="write the per-arm scores here")
    args = ap.parse_args()
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for exp in args.exps:
            try:
                rows, best = fetch(exp, Path(tmp))
            except Exception as exc:  # noqa: BLE001 - arm not started yet
                print(f"{exp}: no history ({exc})")
                continue
            out[exp] = score(rows, best)
    cols = [
        "rounds",
        "sel",
        "smoothed",
        "raw_at_sel",
        "best_raw",
        "r1",
        "r2",
        "r3",
        "r1_over_r2",
        "last_raw",
        "lr",
        "wall_h",
    ]
    print("| arm | " + " | ".join(cols) + " |")
    for exp, s in out.items():
        print(
            f"| {exp} | "
            + " | ".join(fmt(s[c], ".0e" if c == "lr" else ".2f") for c in cols)
            + " |"
        )
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
