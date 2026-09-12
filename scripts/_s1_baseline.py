"""The real-vs-real baseline of the S1 criteria, on the clean steady spans.

The gate asks a clip to sit within a tolerance of the speed-matched median. That
question has an answer for REAL clips too, and it is the only honest reference
for a population model: a synthetic clip cannot be less distinguishable from the
real population than a real clip is. This script measures it for both criteria,
each cell against the speed-matched median of the OTHER motors, and writes
``results/S1/baseline.json``.

Campaign-temporary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.stochastic_fit import accept_stats as stats
from experiments.stochastic_fit import native, stage1


def main() -> None:
    fit = stage1.load("results/S1/fit.json")
    cells, ltas, orders, rates = [], {}, {}, {}
    for motor in stage1.MOTORS:
        for speed in stage1.SPEEDS:
            m = stage1.measure_cell(motor, speed)
            start_s, duration_s = stage1.bench_span(motor, speed)
            clip = native.decimate(
                native.bench_clip(motor, speed, duration_s=duration_s, start_s=start_s),
                stage1.SR,
            )
            x = clip.audio[stage1.CHANNEL].astype(np.float64)
            key = (motor, speed)
            cells.append(key)
            rates[key] = m.rate_rps
            ltas[key] = stats.ltas_bands(x)
            orders[key] = stats.pooled_excess(
                x, m.rate_rps, gamma0=fit.gamma0_hz, gamma_slope=fit.gamma_slope_hz
            )
            print(f"measured Motor{motor}_{speed} ({m.rate_rps:.1f} rev/s)", flush=True)

    keys = [f"k{lo}_{hi}" for lo, hi in stats.ORDER_BANDS]
    rows = []
    for key in cells:
        others = [
            o
            for o in cells
            if o[0] != key[0] and abs(rates[o] - rates[key]) <= stats.SPEED_MATCH_REV_S
        ]
        if not others:
            continue
        ref = np.median(np.array([ltas[o] for o in others]), axis=0)
        dev = ltas[key] - ref
        row = {
            "cell": f"Motor{key[0]}_{key[1]}",
            "motor": key[0],
            "rate": round(rates[key], 2),
            "ltas_mean_abs_db": float(np.abs(dev).mean()),
            "ltas_max_abs_db": float(np.abs(dev).max()),
            "ltas_dev_db": dev.round(2).tolist(),
            "held_out": key[0] in stage1.HELD_OUT_MOTORS,
        }
        for k in keys:
            ref_k = float(np.nanmedian([orders[o][k] for o in others]))
            row[f"order_delta_{k}"] = float(orders[key][k] - ref_k)
        rows.append(row)

    def med(field: str, only_held: bool = False) -> float:
        vals = [r[field] for r in rows if (r["held_out"] or not only_held)]
        return float(np.nanmedian(np.abs(np.asarray(vals, dtype=np.float64))))

    summary = {
        "n_cells": len(rows),
        "all_cells": {
            "ltas_mean_abs_db": med("ltas_mean_abs_db"),
            "ltas_max_abs_db": med("ltas_max_abs_db"),
            **{f"order_{k}": med(f"order_delta_{k}") for k in keys},
        },
        "held_out_cells": {
            "ltas_mean_abs_db": med("ltas_mean_abs_db", True),
            "ltas_max_abs_db": med("ltas_max_abs_db", True),
            **{f"order_{k}": med(f"order_delta_{k}", True) for k in keys},
        },
        "thresholds": {
            "ltas_mean_abs_db": stats.LTAS_MEAN_MAX_DB,
            "ltas_max_abs_db": stats.LTAS_MAX_MAX_DB,
            "order_tol_db": stats.ORDER_TOL_DB,
        },
        "rows": rows,
    }
    out = Path("results/S1/baseline.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))

    print("\n=== real-vs-real baseline, clean steady spans ===")
    for label, d in (
        ("all 20 cells", summary["all_cells"]),
        ("held-out Motor4", summary["held_out_cells"]),
    ):
        print(
            f"{label:18s} LTAS mean|dev| {d['ltas_mean_abs_db']:.2f} dB, "
            f"max|dev| {d['ltas_max_abs_db']:.2f} dB, orders "
            + ", ".join(f"{k} {d['order_' + k]:.2f}" for k in keys)
        )
    print(
        f"thresholds        LTAS mean {stats.LTAS_MEAN_MAX_DB}, max {stats.LTAS_MAX_MAX_DB}, "
        f"order tol {stats.ORDER_TOL_DB}"
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
