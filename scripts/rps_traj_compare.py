#!/usr/bin/env python3
"""Model-vs-model verdict on the frozen rotor-speed statistics.

Compares a NEW candidate's statistics to the BASE ones, both scored against
the REAL per-rig statistics by :func:`experiments.rps_traj.stats.discrepancy`,
and applies the frozen acceptance rule
(:func:`experiments.rps_traj.stats.passes`): a candidate passes when no
family regressed and at least three strictly improved.

Run (one rig)::

    python scripts/rps_traj_compare.py \
        --real results/rps_traj/stats/real/dregon.json \
        --base results/rps_traj/stats/base/dregon.json \
        --new  results/rps_traj/stats/new/dregon.json

Run (many rigs under one root, ``<root>/{real,base,new}/<rig>.json``)::

    python scripts/rps_traj_compare.py --rigs michaels,dregon --root results/rps_traj/stats

Exit code 0 on PASS (every rig passes), 1 on FAIL.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if (_HERE.parent / "src").is_dir() else Path.cwd().resolve()
sys.path.insert(0, str(_ROOT / "src"))

from experiments.rps_traj.stats import (  # noqa: E402  (after the sys.path pin above)
    FAMILIES,
    LAGS_S,
    STRICT_EPS,
    TrajStats,
    discrepancy,
    passes,
)

#: What each family's real value looks like, for the leftmost table column.
_REPORT_LAGS = (0.05, 0.5, 5.0)


def _load(path: Path) -> TrajStats:
    return TrajStats.from_json(Path(path).read_text())


def _nanmean(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def real_summary(real: TrajStats) -> dict[str, str]:
    """One-line description of the real value behind each family."""
    off = ~np.eye(real.n_rotors, dtype=bool)
    idx = [int(np.argmin(np.abs(LAGS_S - lag))) for lag in _REPORT_LAGS]
    acf_at = "/".join(f"{_nanmean(real.acf[:, j]):.3f}" for j in idx)
    return {
        "overall_mean": f"{real.overall_mean:.2f} rev/s",
        "rotor_mean": "[" + " ".join(f"{v:.1f}" for v in np.asarray(real.rotor_mean)) + "] rev/s",
        "rotor_var": "[" + " ".join(f"{v:.2f}" for v in np.asarray(real.rotor_var)) + "]",
        "acf": f"@{'/'.join(f'{lag:g}' for lag in _REPORT_LAGS)}s = {acf_at}",
        "xcorr": f"mean off-diag {_nanmean(np.asarray(real.xcorr)[off]):.3f}",
    }


def report_rig(rig: str, real: TrajStats, base: TrajStats, new: TrajStats) -> bool:
    base_d = discrepancy(base, real)
    new_d = discrepancy(new, real)
    verdict, ok = passes(new_d, base_d)
    summary = real_summary(real)

    print(f"\n=== {rig} ===")
    print(f"{'family':<13} {'real':<34} {'base':>10} {'new':>10}  verdict")
    for family in FAMILIES:
        strict = new_d[family] < base_d[family] - STRICT_EPS
        mark = ("✔" if ok[family] else "✘") + ("*" if strict else " ")
        print(
            f"{family:<13} {summary[family]:<34} "
            f"{base_d[family]:10.5f} {new_d[family]:10.5f}  {mark}"
        )
    print(
        f"{rig}: {'PASS' if verdict else 'FAIL'} "
        f"({sum(new_d[f] < base_d[f] - STRICT_EPS for f in FAMILIES)}/5 strictly better, "
        f"{sum(not ok[f] for f in FAMILIES)} regressed)"
    )
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--real", type=Path, help="real TrajStats JSON (single rig)")
    ap.add_argument("--base", type=Path, help="base model TrajStats JSON (single rig)")
    ap.add_argument("--new", type=Path, help="new model TrajStats JSON (single rig)")
    ap.add_argument("--rigs", help="comma-separated rigs, resolved under --root")
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("results/rps_traj/stats"),
        help="root holding real/, base/, new/ subdirectories (with --rigs)",
    )
    args = ap.parse_args()

    jobs: list[tuple[str, Path, Path, Path]] = []
    if args.rigs:
        root = Path(args.root)
        for rig in (r.strip() for r in str(args.rigs).split(",")):
            if not rig:
                continue
            jobs.append(
                (
                    rig,
                    root / "real" / f"{rig}.json",
                    root / "base" / f"{rig}.json",
                    root / "new" / f"{rig}.json",
                )
            )
    elif args.real and args.base and args.new:
        jobs.append((Path(args.real).stem, args.real, args.base, args.new))
    else:
        ap.error("give either --rigs (with --root) or all three of --real/--base/--new")

    missing = [str(p) for _, *paths in jobs for p in paths if not Path(p).is_file()]
    if missing:
        ap.error("missing stats files: " + ", ".join(missing))

    verdicts = {
        rig: report_rig(rig, _load(real), _load(base), _load(new)) for rig, real, base, new in jobs
    }
    overall = all(verdicts.values())
    print(
        f"\noverall: {'PASS' if overall else 'FAIL'} "
        f"({sum(verdicts.values())}/{len(verdicts)} rigs pass: "
        + ", ".join(f"{rig}={'PASS' if v else 'FAIL'}" for rig, v in verdicts.items())
        + ")"
    )
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
