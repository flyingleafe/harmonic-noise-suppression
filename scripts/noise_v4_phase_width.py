#!/usr/bin/env python3
"""Noise model v4, step A0: the phase-pedestal width W per order on the 20
DREGON single-rotor bench recordings, and the half-normal scales C_W_EVEN /
C_W_ODD it implies (``docs/explainers/noise-model-v4.qmd`` § Procedure A).

The measurement lives in :mod:`experiments.static_rig.phase_width`; this is
the restartable driver (one ``utils.gridrun`` unit per recording) and the
reduction::

    PYTHONPATH=src .venv/bin/python scripts/noise_v4_phase_width.py run --jobs 4
    PYTHONPATH=src .venv/bin/python scripts/noise_v4_phase_width.py reduce

``run`` writes ``<out>/raw/<recording>.json`` and reduces at the end;
``reduce`` re-reads the unit JSONs and writes ``<out>/summary.json`` and
``<out>/findings.md``. A smoke run on one recording, kept out of the
default output so a later full run does not skip it::

    PYTHONPATH=src .venv/bin/python scripts/noise_v4_phase_width.py run \\
        --keys motor_Motor1_70 --k-max 40 --out /tmp/phase_width_smoke --jobs 1

Cluster (one recording takes 2-5 min on one core; ~1.5 GB per worker)::

    omnirun submit --backend uni-cpu --gpus 0 --cpus 16 --mem 48 --time 1h \\
        --outputs 'results/noise_v4/phase_width/**' -- \\
        bash -c 'export PYTHONPATH="$PWD/src"; python scripts/noise_v4_phase_width.py run --jobs 10'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from utils.gridrun import Unit, add_gridrun_args, gridrun_from_args  # noqa: E402

OUT = Path("results/noise_v4/phase_width")


def worker(unit: Unit) -> dict[str, Any]:
    from experiments.static_rig import phase_width as PW

    p = unit.params
    bench = PW.load_bench(str(p["key"]))
    return PW.analyse_recording(
        bench,
        band_frac=float(p["band_frac"]),
        k_max=None if p["k_max"] is None else int(p["k_max"]),
        log=lambda s: print(s, flush=True),
    )


def write_findings(out: Path, summary: dict[str, Any]) -> None:
    from experiments.static_rig import phase_width as PW

    (out / "findings.md").write_text(PW.findings_markdown(summary))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from experiments.static_rig import phase_width as PW

    return PW.reduce_rows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="measure every recording (restartable)")
    run.add_argument("--keys", nargs="*", default=None, help="recordings (default: all 20)")
    run.add_argument("--k-max", type=int, default=None, help="highest order (default: 7.2 kHz)")
    run.add_argument("--band-frac", type=float, default=None, help="band / harmonic spacing")
    run.add_argument("--out", type=Path, default=OUT)
    add_gridrun_args(run, jobs=4)
    red = sub.add_parser("reduce", help="pool the unit JSONs into summary.json + findings.md")
    red.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    from experiments.static_rig import phase_width as PW

    if args.cmd == "run":
        keys = args.keys or list(PW.RECORDINGS)
        band_frac = PW.BAND_FRAC if args.band_frac is None else args.band_frac
        units = [
            Unit(uid=k, params={"key": k, "k_max": args.k_max, "band_frac": band_frac})
            for k in keys
        ]
        res = gridrun_from_args(args, units, worker, args.out, summarize=summarize)
        write_findings(args.out, res.summary)
        raise SystemExit(res.exit_code)

    rows = [json.loads(p.read_text()) for p in sorted((args.out / "raw").glob("*.json"))]
    summary = PW.reduce_rows(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    write_findings(args.out, summary)
    print(json.dumps(summary["parity"], indent=1))


if __name__ == "__main__":
    main()
