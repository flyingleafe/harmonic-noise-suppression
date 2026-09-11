"""Preregistered stage acceptance for the distributional-fit objective.

Criterion 1 — LTAS band deviation. Seven bands, each clip normalised to its own
200-400 Hz mean, so only spectral SHAPE is compared: per-clip
``mean |dev| <= 1.5 dB`` and ``max |dev| <= 3.0 dB``, median over synthetic
draws must pass, against speed-matched real clips (within 5 rev/s).

Criterion 2 — order-domain comb fidelity. The band-integrated per-order excess
over a local median floor (``bench.read_orders``: integration band is +-3x the
width law's own prediction, floor subtracted per bin without clipping) pooled
over orders 2-15, 15-30 and 30-100; synthetic must sit within 1.5 dB of real in
every band. The half-integer comb is scored as the null.

Criterion 3 (S2/S3 only) is the frozen-probe test and lives in
``scripts/_synthetic_probe.py``; this script reads its JSON when present.

Real audio always comes from the native 44.1 kHz source decimated by
``native.decimate``; the published 16 kHz datasets are never used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.stochastic_fit import accept_stats as stats
from experiments.stochastic_fit import native, stage1

SR = 16000
BANDS = stats.BANDS
#: Preregistered thresholds. Never relaxed.
LTAS_MEAN_MAX_DB = stats.LTAS_MEAN_MAX_DB
LTAS_MAX_MAX_DB = stats.LTAS_MAX_MAX_DB
ORDER_BANDS = stats.ORDER_BANDS
ORDER_TOL_DB = stats.ORDER_TOL_DB
SPEED_MATCH_REV_S = stats.SPEED_MATCH_REV_S


def ltas_bands(x, sr=SR, n=8192):
    """See :func:`accept_stats.ltas_bands` — one implementation, shared."""
    return stats.ltas_bands(x, sr, n)


def order_excess(x, rate, fit, sr=SR, half=False):
    """See :func:`accept_stats.pooled_excess` — one implementation, shared."""
    return stats.pooled_excess(
        x, rate, gamma0=fit.gamma0_hz, gamma_slope=fit.gamma_slope_hz, sr=sr, half=half
    )


def _pass(value: float, target: float, tol: float) -> bool:
    return bool(np.isfinite(value) and np.isfinite(target) and abs(value - target) <= tol)


def accept_s1(n_draws: int = 5) -> dict[str, Any]:
    """S1: synthetic single-rotor clips against the HELD-OUT bench motor."""
    fit = stage1.load("results/S1/fit.json")
    measurements = stage1.measure_all()
    held = [m for m in measurements if m.motor in stage1.HELD_OUT_MOTORS]

    real_rows, syn_rows = [], []
    for m in held:
        real_clip = native.decimate(
            native.bench_clip(m.motor, m.setpoint, duration_s=stage1.SECONDS, start_s=3.0), SR
        )
        xr = real_clip.audio[stage1.CHANNEL].astype(np.float64)
        real_rows.append(
            {
                "cell": f"Motor{m.motor}_{m.setpoint}",
                "rate": m.rate_rps,
                "ltas": ltas_bands(xr).tolist(),
                "orders": order_excess(xr, m.rate_rps, fit),
                "orders_half": order_excess(xr, m.rate_rps, fit, half=True),
            }
        )

    for m in held:
        for d in range(n_draws):
            xs = stage1.render(fit, m.rate_rps, seconds=stage1.SECONDS, seed=1000 + d)
            syn_rows.append(
                {
                    "cell": f"Motor{m.motor}_{m.setpoint}",
                    "rate": m.rate_rps,
                    "draw": d,
                    "ltas": ltas_bands(xs).tolist(),
                    "orders": order_excess(xs, m.rate_rps, fit),
                    "orders_half": order_excess(xs, m.rate_rps, fit, half=True),
                }
            )

    # criterion 1: every synthetic draw against speed-matched real clips
    per_draw = []
    for s in syn_rows:
        matched = [r for r in real_rows if abs(r["rate"] - s["rate"]) <= SPEED_MATCH_REV_S]
        ref = np.median(np.array([r["ltas"] for r in matched]), axis=0)
        dev = np.asarray(s["ltas"]) - ref
        per_draw.append(
            {
                "cell": s["cell"],
                "draw": s["draw"],
                "mean_abs_db": float(np.abs(dev).mean()),
                "max_abs_db": float(np.abs(dev).max()),
                "dev_db": dev.round(2).tolist(),
            }
        )
    med_mean = float(np.median([d["mean_abs_db"] for d in per_draw]))
    med_max = float(np.median([d["max_abs_db"] for d in per_draw]))
    c1 = {
        "median_mean_abs_db": med_mean,
        "median_max_abs_db": med_max,
        "threshold_mean_db": LTAS_MEAN_MAX_DB,
        "threshold_max_db": LTAS_MAX_MAX_DB,
        "passed": med_mean <= LTAS_MEAN_MAX_DB and med_max <= LTAS_MAX_MAX_DB,
        "per_draw": per_draw,
    }

    # criterion 2: pooled per-order excess, synthetic vs real
    c2_bands = {}
    for lo, hi in ORDER_BANDS:
        key = f"k{lo}_{hi}"
        rv = float(np.nanmedian([r["orders"][key] for r in real_rows]))
        sv = float(np.nanmedian([s["orders"][key] for s in syn_rows]))
        rn = float(np.nanmedian([r["orders_half"][key] for r in real_rows]))
        sn = float(np.nanmedian([s["orders_half"][key] for s in syn_rows]))
        c2_bands[key] = {
            "real_db": rv,
            "synthetic_db": sv,
            "delta_db": sv - rv,
            "real_half_null_db": rn,
            "synthetic_half_null_db": sn,
            "passed": _pass(sv, rv, ORDER_TOL_DB),
        }
    c2 = {
        "bands": c2_bands,
        "tolerance_db": ORDER_TOL_DB,
        "real_detected_orders": float(np.median([r["orders"]["n_detected"] for r in real_rows])),
        "synthetic_detected_orders": float(
            np.median([s["orders"]["n_detected"] for s in syn_rows])
        ),
        "passed": all(v["passed"] for v in c2_bands.values()),
    }

    return {
        "stage": "S1",
        "held_out_cells": [r["cell"] for r in real_rows],
        "fit_motors": list(stage1.FIT_MOTORS),
        "channel": stage1.CHANNEL,
        "n_draws_per_cell": n_draws,
        "criterion_1_ltas": c1,
        "criterion_2_orders": c2,
        "passed": bool(c1["passed"] and c2["passed"]),
        "real": real_rows,
        "synthetic": syn_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["S1", "S2", "S3"])
    ap.add_argument("--draws", type=int, default=5)
    args = ap.parse_args()
    if args.stage != "S1":
        raise SystemExit(f"{args.stage} acceptance is not implemented yet")
    result = accept_s1(args.draws)
    out = Path(f"results/{args.stage}/acceptance.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))

    c1, c2 = result["criterion_1_ltas"], result["criterion_2_orders"]
    print(f"=== {args.stage} acceptance ===")
    print(
        f"held-out cells: {', '.join(result['held_out_cells'])}  (fit motors "
        f"{result['fit_motors']}, channel {result['channel']})"
    )
    print(
        f"\ncriterion 1 LTAS: median mean|dev| {c1['median_mean_abs_db']:.2f} dB "
        f"(<= {LTAS_MEAN_MAX_DB}), median max|dev| {c1['median_max_abs_db']:.2f} dB "
        f"(<= {LTAS_MAX_MAX_DB})  ->  {'PASS' if c1['passed'] else 'FAIL'}"
    )
    print(
        "   per-band median deviation: "
        + ", ".join(
            f"{lo}-{hi}: {v:+.1f}"
            for (lo, hi), v in zip(
                BANDS,
                np.median(np.array([d["dev_db"] for d in c1["per_draw"]]), axis=0),
                strict=True,
            )
        )
    )
    print(f"\ncriterion 2 order-domain excess (tol {ORDER_TOL_DB} dB):")
    for key, v in c2["bands"].items():
        print(
            f"   {key:8s} real {v['real_db']:7.2f}  synth {v['synthetic_db']:7.2f}  "
            f"delta {v['delta_db']:+6.2f}  null real/synth "
            f"{v['real_half_null_db']:+5.2f}/{v['synthetic_half_null_db']:+5.2f}  "
            f"{'PASS' if v['passed'] else 'FAIL'}"
        )
    print(
        f"   detected orders: real {c2['real_detected_orders']:.0f}, "
        f"synthetic {c2['synthetic_detected_orders']:.0f}"
    )
    print(f"\n{args.stage}: {'PASS' if result['passed'] else 'FAIL'}  ->  {out}")


if __name__ == "__main__":
    main()
