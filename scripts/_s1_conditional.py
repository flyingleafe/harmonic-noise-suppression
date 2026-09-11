"""S1a/S1b — the CONDITIONAL S1 checks, simplest case first.

The population gate compares a synthetic clip against the median of the OTHER
motors, and real clips fail that test themselves (2.09 dB mean|dev| against a
1.5 dB threshold) because each motor carries its own acoustic transfer. The
conditional check asks the question that is actually answerable:

* **S1a** — fit ONE motor with no per-rotor terms, and compare each synthetic
  draw against THAT motor's own clip at the same setpoint. This is the
  simplest thing the family has to do. A leave-one-setpoint-out variant tests
  the speed law: fit four setpoints, score the fifth.
* **S1b** — fit motors 1-3 WITH per-rotor terms, score each fit motor
  conditionally (its own deviation applied), and score held-out Motor4 both
  conditionally (population mean shape — the honest prediction for an unseen
  rotor) and distributionally against the real-vs-real baseline.

Thresholds are the preregistered ones: LTAS mean |dev| <= 1.5 dB and
max |dev| <= 3.0 dB, order-band excess within 1.5 dB.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.stochastic_fit import accept_stats as stats
from experiments.stochastic_fit import native, stage1

ORDER_KEYS = [f"k{lo}_{hi}" for lo, hi in stats.ORDER_BANDS]


def real_clip(motor: int, speed: int) -> np.ndarray:
    start_s, duration_s = stage1.bench_span(motor, speed)
    clip = native.decimate(
        native.bench_clip(motor, speed, duration_s=duration_s, start_s=start_s), stage1.SR
    )
    return clip.audio[stage1.CHANNEL].astype(np.float64)


def score_cell(
    fit: stage1.Stage1Fit, m: stage1.CellMeasurement, *, n_draws: int, motor_term: int | None
) -> dict[str, Any]:
    """One cell: n_draws synthetic clips against that cell's own real clip."""
    xr = real_clip(m.motor, m.setpoint)
    ltas_r = stats.ltas_bands(xr)
    orders_r = stats.pooled_excess(
        xr, m.rate_rps, gamma0=fit.gamma0_hz, gamma_slope=fit.gamma_slope_hz
    )
    _, duration_s = stage1.bench_span(m.motor, m.setpoint)
    draws = []
    for d in range(n_draws):
        xs = stage1.render(fit, m.rate_rps, seconds=duration_s, seed=3000 + d, motor=motor_term)
        dev = stats.ltas_bands(xs) - ltas_r
        # PAIRED per-order comparison: same orders, both finite, real detected.
        paired = stats.paired_delta(
            xr, xs, m.rate_rps, gamma0=fit.gamma0_hz, gamma_slope=fit.gamma_slope_hz
        )
        draws.append(
            {
                "draw": d,
                "ltas_mean_abs_db": float(np.abs(dev).mean()),
                "ltas_max_abs_db": float(np.abs(dev).max()),
                "ltas_dev_db": dev.round(2).tolist(),
                **{f"order_delta_{k}": float(paired[k]) for k in ORDER_KEYS},
                "n_detected_synth": paired["n_detected_synth"],
            }
        )
    return {
        "cell": f"Motor{m.motor}_{m.setpoint}",
        "motor": m.motor,
        "setpoint": m.setpoint,
        "rate": round(m.rate_rps, 2),
        "n_detected_real": orders_r["n_detected"],
        "median_ltas_mean_abs_db": float(np.median([d["ltas_mean_abs_db"] for d in draws])),
        "median_ltas_max_abs_db": float(np.median([d["ltas_max_abs_db"] for d in draws])),
        **{
            f"median_order_delta_{k}": float(np.median([d[f"order_delta_{k}"] for d in draws]))
            for k in ORDER_KEYS
        },
        "draws": draws,
    }


def report(title: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    print(f"\n=== {title} ===")
    print(
        f"{'cell':14s}{'rate':>7}{'mean|dev|':>11}{'max|dev|':>10}"
        + "".join(f"{k:>10}" for k in ORDER_KEYS)
        + f"{'det r/s':>10}"
    )
    for r in rows:
        print(
            f"{r['cell']:14s}{r['rate']:7.1f}"
            f"{r['median_ltas_mean_abs_db']:11.2f}{r['median_ltas_max_abs_db']:10.2f}"
            + "".join(f"{r['median_order_delta_' + k]:+10.2f}" for k in ORDER_KEYS)
            + f"{r['n_detected_real']:5.0f}/{np.median([d['n_detected_synth'] for d in r['draws']]):<4.0f}"
        )
    med_mean = float(np.median([r["median_ltas_mean_abs_db"] for r in rows]))
    med_max = float(np.median([r["median_ltas_max_abs_db"] for r in rows]))
    orders = {
        k: float(np.median([abs(r[f"median_order_delta_{k}"]) for r in rows])) for k in ORDER_KEYS
    }
    c1 = med_mean <= stats.LTAS_MEAN_MAX_DB and med_max <= stats.LTAS_MAX_MAX_DB
    c2 = all(v <= stats.ORDER_TOL_DB for v in orders.values())
    print(
        f"\nmedian over cells: LTAS mean|dev| {med_mean:.2f} dB "
        f"(<= {stats.LTAS_MEAN_MAX_DB}), max|dev| {med_max:.2f} dB "
        f"(<= {stats.LTAS_MAX_MAX_DB})  ->  {'PASS' if c1 else 'FAIL'}"
    )
    print(
        "median |order delta|: "
        + ", ".join(f"{k} {v:.2f}" for k, v in orders.items())
        + f"  (tol {stats.ORDER_TOL_DB})  ->  {'PASS' if c2 else 'FAIL'}"
    )
    print(f"{title}: {'PASS' if (c1 and c2) else 'FAIL'}")
    return {
        "title": title,
        "median_ltas_mean_abs_db": med_mean,
        "median_ltas_max_abs_db": med_max,
        "median_order_abs_delta": orders,
        "criterion_1_passed": c1,
        "criterion_2_passed": c2,
        "passed": bool(c1 and c2),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["S1a", "S1b"], required=True)
    ap.add_argument("--motor", type=int, default=1, help="S1a: which single rotor to fit")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument(
        "--holdout-speed",
        type=int,
        default=70,
        help="S1a: setpoint left out of the fit to test the speed law",
    )
    args = ap.parse_args()

    ms = stage1.measure_all()
    out: dict[str, Any] = {
        "stage": args.stage,
        "channel": stage1.CHANNEL,
        "draws_per_cell": args.draws,
    }

    if args.stage == "S1a":
        motor = args.motor
        cells = [m for m in ms if m.motor == motor]
        fit = stage1.fit_population(ms, motors=(motor,))
        fit = stage1.calibrate_profile(fit, [m for m in ms if m.motor == motor], rounds=3)
        stage1.save(fit, f"results/S1/fit_motor{motor}.json")
        out["fit"] = {
            "motors": [motor],
            "width_slope_hz_per_order": fit.gamma_slope_hz,
            "amp_exp": fit.amp_exp,
            "floor_exp": fit.floor_exp,
            "calibration": fit.diagnostics.get("calibration", []),
        }
        out["in_fit"] = report(
            f"S1a conditional — Motor{motor} alone, all five setpoints in the fit",
            [score_cell(fit, m, n_draws=args.draws, motor_term=None) for m in cells],
        )
        kept = tuple(s for s in stage1.SPEEDS if s != args.holdout_speed)
        fit_lo = stage1.fit_population(ms, motors=(motor,), speeds=kept)
        fit_lo = stage1.calibrate_profile(
            fit_lo, [m for m in ms if m.motor == motor and m.setpoint in kept], rounds=3
        )
        held = [m for m in cells if m.setpoint == args.holdout_speed]
        out["leave_one_speed_out"] = report(
            f"S1a speed law — setpoint {args.holdout_speed} LEFT OUT of the fit",
            [score_cell(fit_lo, m, n_draws=args.draws, motor_term=None) for m in held],
        )
    else:
        fit = stage1.fit_population(ms, motors=stage1.FIT_MOTORS)
        fit = stage1.calibrate_profile(fit, ms, rounds=3)
        stage1.save(fit, "results/S1/fit.json")
        out["conditional_fit_motors"] = report(
            "S1b conditional — motors 1-3, each with its own fitted deviation",
            [
                score_cell(fit, m, n_draws=args.draws, motor_term=m.motor)
                for m in ms
                if m.motor in stage1.FIT_MOTORS
            ],
        )
        out["held_out_population"] = report(
            "S1b held-out Motor4 — population mean shape, no per-rotor term",
            [
                score_cell(fit, m, n_draws=args.draws, motor_term=None)
                for m in ms
                if m.motor in stage1.HELD_OUT_MOTORS
            ],
        )

    path = Path(f"results/S1/{args.stage}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
