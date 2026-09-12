"""Fit the DREGON bench RIG: all four motors, five setpoints each.

Stage 1a fitted one motor. This fits the rig - four motors, twenty cells - with
the width law, the coherence and the profile SHAPE tied across motors and a
per-motor profile offset, which is what a rig fit is for: the motors differ in
level and in acoustic position, not in physics.

The result is the parameter set that Stage 1c transfers to DREGON IN FLIGHT
without seeing a single flight recording.

    python scripts/_stage1b_fit.py --out results/S1/bayes_rig.json
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from experiments.stochastic_fit import stage1_bayes as SB


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motors", type=int, nargs="*", default=(1, 2, 3, 4))
    ap.add_argument("--speeds", type=int, nargs="*", default=list(SB.SPEEDS))
    ap.add_argument("--iters", type=int, nargs=3, default=(120, 120, 300))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-rotor-delta", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/S1/bayes_rig.json"))
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"torch threads {torch.get_num_threads()}", flush=True)
    t0 = time.time()
    summary = SB.fit(
        motors=tuple(args.motors),
        speeds=tuple(args.speeds),
        rotor_delta=not args.no_rotor_delta,
        iters=tuple(args.iters),
    )
    print(f"\nwrote {SB.save(summary, args.out)}  ({time.time() - t0:.0f}s)", flush=True)
    for cid, e in summary["clips"].items():
        p, s = e["params"], e["scores"]
        print(
            f"{cid}: nll {s['nll_fit']:.4f} excess {s['excess_over_loo']:+.3f} "
            f"k_half {p.get('coherence_k_half', 0.0):.2f} "
            f"gamma0 {float(np.atleast_1d(p['gamma0'])[0]):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
