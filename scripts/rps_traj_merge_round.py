#!/usr/bin/env python3
"""Merge one-job-per-rig round records into a single round snapshot.

The campaign's exact-likelihood round is one ``omnirun`` job per rig, so each
job writes a single-rig ``summary.json`` and would overwrite its siblings'.
``scripts/rps_traj_fit.py --round <name>`` therefore also writes one record per
rig under ``rounds/<name>/<rig>.json``; this script collects those into the
combined ``rounds/<name>.json`` and ``summary.json`` that the comparison,
diagnostic and explainer tooling expects.

The per-rig records already carry both discrepancy dicts and the provenance
(NLL, iterations, wall time, retention, parameters), so nothing is recomputed
and nothing is inferred: the merge is a concatenation in the campaign's rig
order plus the overall verdict.

Usage::

    python scripts/rps_traj_merge_round.py --round round4 --out results/rps_traj
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if (_HERE.parent / "src").is_dir() else Path.cwd().resolve()
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from rps_traj_fit import RIGS  # noqa: E402  (sibling script, after the path pin)

from experiments.rps_traj.data import RATE_HZ  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--round", required=True, help="round name, e.g. round4")
    ap.add_argument("--out", type=Path, default=Path("results/rps_traj"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-rep", type=int, default=40)
    args = ap.parse_args()

    parts_dir = Path(args.out) / "rounds" / args.round
    paths = sorted(parts_dir.glob("*.json"))
    if not paths:
        raise SystemExit(f"no per-rig records under {parts_dir}")
    records = {p.stem: json.loads(p.read_text()) for p in paths}

    order = [rig for rig in RIGS if rig in records]
    order += [rig for rig in sorted(records) if rig not in order]
    missing = [rig for rig in RIGS if rig not in records]
    if missing:
        print(f"WARNING: no record for {', '.join(missing)}")

    verdicts = {rig: records[rig].get("verdict") for rig in order}
    scored = {rig: v for rig, v in verdicts.items() if v is not None}
    overall = bool(scored) and all(scored.values())
    summary = json.dumps(
        {
            "rate_hz": RATE_HZ,
            "seed": args.seed,
            "n_rep": args.n_rep,
            "overall": overall if scored else None,
            "rigs": [records[rig] for rig in order],
        },
        indent=2,
        allow_nan=False,
    )
    for target in (
        Path(args.out) / "summary.json",
        Path(args.out) / "rounds" / f"{args.round}.json",
    ):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(summary)
        print(f"wrote {target}")

    print(
        f"merged {len(order)} rigs: "
        + ", ".join(
            f"{rig}={'PASS' if verdicts[rig] else 'FAIL' if verdicts[rig] is not None else 'n/a'}"
            for rig in order
        )
    )
    print(f"overall: {'PASS' if overall else 'FAIL'} ({sum(scored.values())}/{len(scored)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
