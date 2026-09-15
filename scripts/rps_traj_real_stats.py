#!/usr/bin/env python3
"""Real per-rig rotor-speed statistics — the reference every model is scored against.

Loads every real flight of each rig through
:func:`experiments.rps_traj.data.load_rig` (published ``*-frames`` datasets →
100 Hz common grid), applies the frozen airborne rule, and writes the frozen
:class:`~experiments.rps_traj.stats.TrajStats` per rig.

Run::

    python scripts/rps_traj_real_stats.py --rigs michaels,dregon \
        --out results/rps_traj/stats/real/

Outputs per rig ``<rig>.json`` (TrajStats) and ``<rig>.flights.json`` (per
flight: source track, duration, NaN fraction, airborne segment table) plus a
summary table on stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if (_HERE.parent / "src").is_dir() else Path.cwd().resolve()
sys.path.insert(0, str(_ROOT / "src"))

from experiments.rps_traj.data import (  # noqa: E402  (after the sys.path pin above)
    RATE_HZ,
    Flight,
    airborne_segments,
    load_rig,
)
from experiments.rps_traj.stats import (  # noqa: E402
    LAGS_S,
    TrajStats,
    compute_stats,
)

#: Lags reported on stdout (the JSON always carries all 24).
REPORT_LAGS_S = (0.05, 0.5, 5.0)


def flight_report(flight: Flight) -> dict[str, Any]:
    """Per-flight bookkeeping: what the statistics were actually allowed to see."""
    rps = np.asarray(flight.rps, dtype=np.float64)
    segments = airborne_segments(rps, flight.fs)
    airborne = sum(sl.stop - sl.start for sl in segments)
    return {
        "rig": flight.rig,
        "flight": flight.flight,
        "source": flight.source,
        "fs": float(flight.fs),
        "t0": float(flight.t0),
        "n_rotors": flight.n_rotors,
        "duration_s": round(flight.duration_s, 3),
        "nan_fraction": round(float(np.isnan(rps).mean()), 6),
        "airborne_s": round(airborne / float(flight.fs), 3),
        "segments": [
            {
                "start_s": round(sl.start / float(flight.fs), 3),
                "duration_s": round((sl.stop - sl.start) / float(flight.fs), 3),
            }
            for sl in segments
        ],
        "rotor_mean_rps": [
            None if not np.isfinite(v) else round(float(v), 4) for v in _nanmean_rows(rps)
        ],
    }


def _nanmean_rows(rps: np.ndarray) -> np.ndarray:
    finite = np.isfinite(rps)
    total = np.where(finite, rps, 0.0).sum(axis=1)
    count = finite.sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _lag_columns() -> list[tuple[float, int]]:
    """``(requested lag, index into LAGS_S)`` for the stdout summary."""
    return [(lag, int(np.argmin(np.abs(LAGS_S - lag)))) for lag in REPORT_LAGS_S]


def print_summary(rig: str, stats: TrajStats, flights: list[Flight]) -> None:
    total = sum(f.duration_s for f in flights)
    print(f"\n=== {rig}: {len(flights)} flights, {total:.1f} s recorded ===")
    print(f"overall_mean = {stats.overall_mean:.4f} rev/s")
    print(
        "rotor      mean[rev/s]      var[(rev/s)^2]   "
        + "  ".join(f"acf@{lag:g}s" for lag, _ in _lag_columns())
    )
    for r in range(stats.n_rotors):
        acfs = "  ".join(f"{stats.acf[r, j]:9.4f}" for _, j in _lag_columns())
        print(f"{r:<10d} {stats.rotor_mean[r]:12.4f} {stats.rotor_var[r]:16.4f}   {acfs}")
    print("xcorr:")
    for r in range(stats.n_rotors):
        print("  " + "  ".join(f"{v:7.4f}" for v in stats.xcorr[r]))


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--rigs", default="michaels,dregon", help="comma-separated rig ids")
    ap.add_argument("--out", type=Path, default=Path("results/rps_traj/stats/real"))
    args = ap.parse_args()

    rigs = [r.strip() for r in str(args.rigs).split(",") if r.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for rig in rigs:
        flights = load_rig(rig)
        if not flights:
            raise SystemExit(f"rig {rig!r} yielded no flights")
        # One rig may be fed by several datasets; keep the rigs they declare.
        by_rig: dict[str, list[Flight]] = {}
        for flight in flights:
            by_rig.setdefault(flight.rig, []).append(flight)
        for rig_id, rig_flights in sorted(by_rig.items()):
            stats = compute_stats(rig_flights)
            (out_dir / f"{rig_id}.json").write_text(stats.to_json())
            (out_dir / f"{rig_id}.flights.json").write_text(
                json.dumps(
                    {
                        "rig": rig_id,
                        "rate_hz": RATE_HZ,
                        "n_flights": len(rig_flights),
                        "flights": [flight_report(f) for f in rig_flights],
                    },
                    indent=2,
                )
            )
            print_summary(rig_id, stats, rig_flights)
            print(f"wrote {out_dir / f'{rig_id}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
