"""Fit Michael's FLY125 cruise with the Stage-1 line model (coherent + stochastic).

One entry point for both local runs and remote jobs, so the same command can be
submitted to a CPU backend with omnirun:

    python scripts/_stage2_fit.py --clips 8 --seconds 16 --out results/S2/cruise.json
    omnirun submit --backend uni-cpu --gpus 0 --time 8h --yes -- \
        python scripts/_stage2_fit.py --clips 8 --seconds 16 \
        --out results/S2/cruise.json

The fitted model is the one Stage 1 selected: every order carries one power,
split by a coherent fraction ``w_k = exp(-(k/k_half)^2)`` between a coherent
needle with the window's own power response and a Rayleigh pedestal of the
fitted width. Nothing here is calibrated against an acceptance statistic.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.stochastic_fit import stage2 as S2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default=S2.FIT_RECORDING)
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=S2.CRUISE_SECONDS)
    ap.add_argument("--k-cap", type=int, default=S2.K_CAP)
    ap.add_argument("--mics", type=int, default=None, help="fit fewer channels (default: all)")
    ap.add_argument("--iters", type=int, nargs=3, default=(120, 120, 300))
    ap.add_argument("--ladder", type=int, nargs="*", default=(16, 48))
    ap.add_argument("--no-rotor-delta", action="store_true")
    ap.add_argument("--regime", default="cruise", choices=sorted(S2.REGIMES))
    ap.add_argument("--floor-dynamics", action="store_true", help="let the floor level/tilt drift")
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = leave default)")
    ap.add_argument("--out", type=Path, default=Path("results/S2/cruise.json"))
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"torch threads {torch.get_num_threads()}", flush=True)
    t0 = time.time()
    summary = S2.fit(
        args.recording,
        max_clips=args.clips,
        seconds=args.seconds,
        k_cap=args.k_cap,
        n_mics=args.mics,
        iters=tuple(args.iters),
        ladder=tuple(args.ladder),
        rotor_delta=not args.no_rotor_delta,
        regime=args.regime,
        floor_dynamics=args.floor_dynamics,
    )
    path = S2.save(summary, args.out)
    print(f"\nwrote {path}  ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for cid, entry in summary["clips"].items():
        s = entry["scores"]
        p = entry["params"]
        rows.append(
            dict(
                clip=cid,
                nll_fit=round(float(s["nll_fit"]), 4),
                excess=round(float(s["excess_over_loo"]), 3),
                k_half=round(float(p.get("coherence_k_half", 0.0)), 2),
                gamma0=round(float(np.atleast_1d(p["gamma0"])[0]), 3),
                gamma_slope=round(float(np.atleast_1d(p["gamma_slope"])[0]), 4),
            )
        )
        print(rows[-1], flush=True)
    Path(str(args.out) + ".summary.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
