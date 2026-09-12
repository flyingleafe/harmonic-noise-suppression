"""Gate the committed refined rotor-speed labels by flight regime.

The refiner corrects the whole recording, including standby, where the comb is
weak and dense and the correction is fitted to noise. This driver rewrites each
sidecar in ``src/data_processing/refined_labels`` so that

* standby frames carry the telemetry EXACTLY,
* settled cruise frames carry the refined trajectory EXACTLY,
* ramp frames carry a blended correction, so the label is continuous.

The untouched refiner output is kept in the sidecar as ``r_refined_raw``, so
nothing is lost and the gate can be re-derived or changed later.

    python scripts/fix_refined_labels.py --check      # report only
    python scripts/fix_refined_labels.py --write      # rewrite the sidecars
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data_processing.rps_gating import GatePolicy, gate_refinement, regime_summary

LABEL_DIR = Path("src/data_processing/refined_labels")


def check_invariants(
    tel: np.ndarray,
    raw: np.ndarray,
    gated: np.ndarray,
    w: np.ndarray,
) -> dict[str, float | bool]:
    """The three properties the gate must have, measured rather than asserted."""
    standby = w <= 0.0
    full = w >= 1.0
    d_standby = float(np.abs(gated[:, standby] - tel[:, standby]).max()) if standby.any() else 0.0
    d_full = float(np.abs(gated[:, full] - raw[:, full]).max()) if full.any() else 0.0
    # continuity: the largest frame-to-frame step the GATE introduces, i.e. the
    # step in the gated label minus the step the raw refined label already has
    step_gated = np.abs(np.diff(gated, axis=1)).max() if gated.shape[-1] > 1 else 0.0
    step_raw = np.abs(np.diff(raw, axis=1)).max() if raw.shape[-1] > 1 else 0.0
    return dict(
        standby_exact=bool(d_standby == 0.0),
        standby_max_dev=d_standby,
        cruise_exact=bool(d_full == 0.0),
        cruise_max_dev=d_full,
        max_step_gated=float(step_gated),
        max_step_raw=float(step_raw),
        introduced_step=bool(step_gated > step_raw + 1e-9),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-dir", type=Path, default=LABEL_DIR)
    ap.add_argument("--write", action="store_true", help="rewrite the sidecars in place")
    ap.add_argument("--standby-max", type=float, default=GatePolicy().standby_max_rps)
    ap.add_argument("--cruise-min", type=float, default=GatePolicy().cruise_min_rps)
    ap.add_argument("--settle", type=float, default=GatePolicy().settle_s)
    args = ap.parse_args()
    policy = GatePolicy(args.standby_max, args.cruise_min, args.settle)

    files = sorted(p for p in args.label_dir.glob("*.npz"))
    print(f"policy: {policy.as_dict()}\n{len(files)} sidecar(s) in {args.label_dir}\n")
    for path in files:
        with np.load(path, allow_pickle=True) as z:
            payload = {k: z[k] for k in z.files}
        ft = np.asarray(payload["ft"], dtype=np.float64)
        tel = np.asarray(payload["r_telemetry"], dtype=np.float64)
        raw = np.asarray(payload.get("r_refined_raw", payload["r_refined"]), dtype=np.float64)
        gated, w = gate_refinement(ft, tel, raw, policy)
        inv = check_invariants(tel, raw, gated, w)
        summ = regime_summary(ft, tel, policy)
        d_raw = np.abs(raw - tel)
        d_gate = np.abs(gated - tel)
        print(
            f"{path.stem:34} frames {int(summ['frames']):5d}  "
            f"standby {summ['standby_frac']:5.1%} ramp {summ['ramp_frac']:5.1%} "
            f"cruise {summ['cruise_frac']:5.1%}"
        )
        print(
            f"{'':34}  correction rms {d_raw.mean():.3f} -> {d_gate.mean():.3f} rev/s;"
            f" standby correction {d_raw[:, w <= 0].mean() if (w <= 0).any() else 0:.3f}"
            f" -> {d_gate[:, w <= 0].mean() if (w <= 0).any() else 0:.3f}"
        )
        flags = [k for k in ("standby_exact", "cruise_exact") if not inv[k]]
        if inv["introduced_step"]:
            flags.append("introduced_step")
        print(
            f"{'':34}  invariants {'OK' if not flags else 'FAILED: ' + ', '.join(flags)}"
            f"  (max step gated {inv['max_step_gated']:.3f} vs raw {inv['max_step_raw']:.3f})"
        )
        if not args.write:
            continue
        payload["r_refined_raw"] = raw
        payload["r_refined"] = gated
        payload["gate_weight"] = w
        payload["gate_policy"] = json.dumps(policy.as_dict())
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **payload)
        tmp.replace(path)
        rep_path = path.with_suffix("").with_suffix(".report.json")
        if rep_path.exists():
            rep = json.loads(rep_path.read_text())
            rep["gate"] = dict(policy=policy.as_dict(), invariants=inv, regimes=summ)
            rep_path.write_text(json.dumps(rep, indent=1))
    if args.write:
        print("\nrewritten; raw refiner output kept as r_refined_raw in each sidecar")


if __name__ == "__main__":
    main()
