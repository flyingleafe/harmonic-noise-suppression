"""Fit the rotor-noise model — the one fitting entry point.

Clips come from a dload frames dataset (`DREGON-frames`, `michaels-frames`) at
the recording's native 44.1 kHz, with the published refined rotor-speed label
held fixed. Nothing reads a raw tree or a path under a user's home, so the same
command runs on a cluster node.

The REGIME selects what is fitted (`experiments.stochastic_fit.campaign`):

    # Michael's FLY125 cruise: the accepted flight fit, all 8 channels
    python scripts/stochastic_fit.py --regime cruise --recording FLY125 \
        --clips 8 --seconds 16 --out results/S2/cruise_8clip.json

    # FLY125 standby: one window, a taller comb (226 orders under Nyquist)
    python scripts/stochastic_fit.py --regime standby --clips 1 \
        --out results/S2/standby.json

    # one DREGON free-flight recording, 4 channels, raw telemetry instead
    python scripts/stochastic_fit.py --regime cruise --dataset DREGON-frames \
        --recording free-flight_nosource_room2 --clips 1 --channels 0-3 \
        --rps-key motors_command --out results/S2/dregon_flight.json

    # the DREGON single-motor BENCH rig: static cells, the rate is fitted
    python scripts/stochastic_fit.py --regime bench --motors 1,2,3 \
        --speeds 50,60,70,80,90 --out results/S1/bayes_rig.json

    omnirun submit --backend uni-gpushort --gpus 1 --time 1h --yes -- \
        python scripts/stochastic_fit.py --regime cruise --out results/S2/cruise.json

The model is the one the bench selected: every order carries one power, split
by a coherent fraction ``w_k = exp(-(k/k_half)^2)`` between a coherent needle
with the analysis window's own power response and a Rayleigh pedestal of the
fitted width. Nothing here is calibrated against an acceptance statistic — the
gate is ``scripts/_stage2_probe.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.stochastic_fit import campaign
from experiments.stochastic_fit import clips as C
from experiments.stochastic_fit import stage1_bayes as SB
from experiments.stochastic_fit import stage2 as S2


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(p) for p in str(text).replace(" ", "").split(",") if p)


def _device(name: str) -> str:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda asked for, but torch.cuda.is_available() is False")
    label = torch.cuda.get_device_name(0) if name == "cuda" else "cpu"
    print(f"device {name} ({label})  torch threads {torch.get_num_threads()}", flush=True)
    return name


def _report(summary: dict[str, Any], out: Path) -> None:
    """Write the summary and print one row per clip."""
    path = campaign.save(summary, out)
    print(f"\nwrote {path}  ({summary.get('seconds', float('nan')):.0f}s)", flush=True)
    rows = []
    for cid, entry in summary["clips"].items():
        s, p = entry["scores"], entry["params"]
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
    Path(str(out) + ".summary.json").write_text(json.dumps(rows, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--regime", default="cruise", choices=sorted(campaign.REGIMES))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--dataset",
        default=None,
        help="frames dataset NAME[@VERSION] (default: the regime's — "
        f"{S2.FIT_DATASET} in flight, {SB.BENCH_DATASET} on the bench)",
    )
    ap.add_argument("--version", default=None, help="dataset version (overrides NAME@VERSION)")
    ap.add_argument(
        "--channels",
        default=None,
        help=f"'all' (flight default) or a list like '0,3-5' (bench default: {SB.CHANNEL})",
    )
    ap.add_argument(
        "--rps-key",
        default=C.DEFAULT_RPS_KEY,
        help="rotor track held fixed: rps_refined (default), rps, motors_measured, "
        "motors_command, or 'auto' for the first one published. The bench publishes "
        "none, so there the rate is fitted",
    )
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--iters", type=int, nargs=3, default=(120, 120, 300))
    ap.add_argument("--ladder", type=int, nargs="*", default=(16, 48))
    ap.add_argument("--k-cap", type=int, default=None, help="override the regime's ladder cap")
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = leave default)")
    ap.add_argument(
        "--rotor-delta",
        dest="rotor_delta",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="one profile offset per rotor/motor (default: the regime's)",
    )

    flight = ap.add_argument_group("flight regimes (cruise, standby)")
    flight.add_argument(
        "--recording",
        nargs="+",
        default=[S2.FIT_RECORDING],
        help="[dataset[@version]:]RECORDING_ID; several pool into ONE rig fit "
        "(e.g. the five DREGON room2 training flights), --clips is per recording",
    )
    flight.add_argument("--clips", type=int, default=8, help="how many windows to fit")
    flight.add_argument("--seconds", type=float, default=None, help="window length")
    flight.add_argument("--min-rps", type=float, default=None, help="override the regime's floor")
    flight.add_argument(
        "--floor-dynamics", action="store_true", help="let the floor level/tilt drift"
    )

    bench = ap.add_argument_group("bench regime")
    bench.add_argument("--motors", default=",".join(str(m) for m in SB.FIT_MOTORS))
    bench.add_argument("--speeds", default=",".join(str(s) for s in SB.SPEEDS))

    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    t0 = time.time()
    summary = campaign.fit(
        args.regime,
        recordings=tuple(args.recording),
        dataset=args.dataset,
        version=args.version,
        rps_key=args.rps_key,
        channels=args.channels,
        max_clips=args.clips,
        seconds=args.seconds,
        k_cap=args.k_cap,
        min_rps=args.min_rps,
        motors=_ints(args.motors),
        speeds=_ints(args.speeds),
        rotor_delta=args.rotor_delta,
        floor_dynamics=args.floor_dynamics,
        iters=tuple(args.iters),
        ladder=tuple(args.ladder),
        device=_device(args.device),
        log=lambda m: print(m, flush=True),
    )
    _report(summary, args.out)
    print(f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
