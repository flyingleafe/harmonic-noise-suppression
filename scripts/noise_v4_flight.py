#!/usr/bin/env python
"""Noise model v4, procedure B: fit MD2 (Michael's drone) in free flight.

Spec: ``docs/explainers/noise-model-v4.qmd`` § Procedure B; module
:mod:`experiments.noise_model.flight_v4`. One restart per call; the restarts
are reduced afterwards to the lowest objective.

    # one restart (seed 0 = prior medians; seeds 1-3 draw sigma_nu from its prior)
    python scripts/noise_v4_flight.py fit --seed 0 --device cuda --out results/noise_v4/flight
    # the four restarts reduced to one payload
    python scripts/noise_v4_flight.py reduce --out results/noise_v4/flight
    # a CPU smoke: one standby + one cruise window, mics 0,1, 24 orders
    python scripts/noise_v4_flight.py fit --max-windows 2 --mics 0,1 --k-cap 24 \\
        --adam-steps 20 --lbfgs-iters 10 --polish-iters 4 --out /tmp/nv4_smoke

The pool is FLY125 standby (``michaels-standby``, 3 x 4 s) and cruise
(``michaels-cruise`` FLY125 rows, 8 x 8 s), labels ``rps_refined``, the nv3r4
front end (frames thinned by 4, 32 kHz work grid, 16-frame chunks) with the
channels divided by the measured array response (``--channel-gains-band
transfer``). Supports come from the ``.npz`` cache or are built from the data
(``scripts/noise_v2_supports.py build`` fills the cache in a remote job).

Writes ``<out>/restarts/<name>__flight_v4__s<seed>.json`` (``noise-v4-fit/3``);
``reduce`` writes ``<out>/<name>__flight_v4.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

NAME = "md2_fly125"
SETS = ("michaels-standby", "michaels-cruise")
K_CAP = 130
CHANNEL_GAINS = "results/noise_v2/mic_gains/mic_gains.json"


def _pool_specs(sets: list[str], max_windows: int | None) -> list[str]:
    """FLY125 windows of ``sets``; ``max_windows`` takes them round-robin
    across the sets, so a two-window smoke still joins standby and cruise."""
    from experiments.noise_model import supports as SU

    per_set = []
    for name in sets:
        specs = [s for s in SU.support_set(name) if s.args.get("recording") == "FLY125"]
        per_set.append([s.text if hasattr(s, "text") else str(s) for s in specs])
    out: list[str] = []
    for i in range(max(len(s) for s in per_set)):
        out.extend(s[i] for s in per_set if i < len(s))
    return out if max_windows is None else out[: int(max_windows)]


def _build(args: argparse.Namespace) -> tuple[Any, Any]:
    import numpy as np
    import torch

    from experiments.noise_model import fit as FT
    from experiments.noise_model import flight_v4 as FV
    from experiments.noise_model import model as MD
    from experiments.noise_model import supports as SU

    torch.set_num_threads(max(1, int(args.threads)))
    mics = [int(v) for v in str(args.mics).split(",")] if args.mics else None
    specs = list(args.support) if args.support else _pool_specs(list(args.sets), args.max_windows)
    members = []
    first = None
    for spec in specs:
        s = SU.load_support(spec)
        first = first or s
        power = np.asarray(s.power, dtype=np.float64)
        members.append(
            (
                s.name,
                power if mics is None else power[np.asarray(mics, dtype=np.int64)],
                np.asarray(s.carrier_rev_s_audio, dtype=np.float64),
                np.asarray(s.frame_starts, dtype=np.int64),
            )
        )
    assert first is not None
    gains = FT.load_channel_gains(
        str(args.channel_gains), rig="michaels", mics=mics, band=str(args.channel_gains_band)
    )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    batch = MD.flight_batch(
        name=str(args.name),
        members=members,
        sr=int(first.sr),
        n_fft=int(first.n_fft),
        hop=int(first.hop),
        k_cap=int(args.k_cap),
        frame_stride=int(args.frame_stride),
        channel_gains=gains,
        device=device,
        chunk_frames=None if int(args.chunk_frames) <= 0 else int(args.chunk_frames),
        sr_work=int(args.work_rate),
    )
    data = FV.prepare(batch, k_cap=int(args.k_cap), mics=mics, wander_path=str(args.wander))
    return data, gains


def cmd_fit(args: argparse.Namespace) -> int:
    from experiments.noise_model import flight_v4 as FV

    t0 = time.time()
    data, gains = _build(args)
    b = data.batch
    print(
        f"pool {b.name}: {len(b.members)} windows, {int(b.power.shape[1])} frames, "
        f"{b.n_mics} mics, K {data.k_max}, motor orders {data.motor_orders.tolist()}, "
        f"device {b.power.device}, build {time.time() - t0:.1f}s",
        flush=True,
    )
    t1 = time.time()
    measured = FV.measure(data)
    print(
        f"measured: mu {measured.floor_mean_db:.2f} dB  sigma_B {measured.floor_shape_sd_db:.2f} dB  "
        f"({time.time() - t1:.1f}s)",
        flush=True,
    )
    if args.time_eval:
        import torch

        model = FV.FlightV4Model(data, measured, kappa=float(args.kappa))
        t2 = time.time()
        loss = float(model.objective(b, grad=True))
        FV._sync(b.power.device)
        dt = time.time() - t2
        grads = sum(float((p.grad**2).sum()) for p in model.params.values() if p.grad is not None)
        print(
            f"ONE objective call (forward + gradient, {int(b.power.shape[1])} frames, "
            f"{b.n_cells} cells): {dt:.2f}s  loss {loss:.6g}  |grad|^2 {grads:.3g}  "
            f"threads {torch.get_num_threads()}",
            flush=True,
        )
        return 0
    optim = FV.OptimV4(
        adam_steps=int(args.adam_steps),
        adam_lr=float(args.adam_lr),
        adam_batch=int(args.adam_batch),
        lbfgs_iters=int(args.lbfgs_iters),
        lbfgs_frames=int(args.lbfgs_frames),
        polish_iters=int(args.polish_iters),
        seed=int(args.seed),
    )
    model, record = FV.fit_flight(
        data, measured, kappa=float(args.kappa), optim=optim, progress=int(args.progress)
    )
    pay = FV.payload(model, record, channel_gains=gains)
    pay["pool"]["specs"] = (
        list(args.support) if args.support else _pool_specs(list(args.sets), args.max_windows)
    )
    out = Path(args.out) / "restarts" / f"{args.name}__{FV.FLIGHT_V4_MODE}__s{int(args.seed)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pay, indent=1))
    conv = record["optimiser"]["convergence"]
    print(
        f"wrote {out}: total {record['objective']['total_nats']:.6g} nats "
        f"(start {record['objective']['start_total_nats']:.6g}); converged {conv['converged']} "
        f"(gain/cell {conv['gain_per_cell']:.3g}, moved {conv['movement']}); "
        f"wall {time.time() - t0:.1f}s",
        flush=True,
    )
    return 0


def cmd_reduce(args: argparse.Namespace) -> int:
    from experiments.noise_model import flight_v4 as FV

    paths = sorted((Path(args.out) / "restarts").glob(f"{args.name}__{FV.FLIGHT_V4_MODE}__s*.json"))
    if not paths:
        raise SystemExit(f"no restarts under {Path(args.out) / 'restarts'}")
    payloads = [json.loads(p.read_text()) for p in paths]
    red = FV.reduce_payloads(payloads)
    red["restarts"]["paths"] = [str(p) for p in paths]
    out = Path(args.out) / f"{args.name}__{FV.FLIGHT_V4_MODE}.json"
    out.write_text(json.dumps(red, indent=1))
    r = red["restarts"]
    print(
        f"wrote {out}: seed {r['selected_seed']} of {r['seeds']}; totals {r['total_nats']}; "
        f"converged {r['converged']}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    from experiments.noise_model import flight_v4 as FV

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="one restart")
    f.add_argument("--out", required=True)
    f.add_argument("--name", default=NAME)
    f.add_argument("--sets", nargs="+", default=list(SETS))
    f.add_argument("--support", nargs="*", default=None, help="explicit support specs instead")
    f.add_argument("--max-windows", type=int, default=None)
    f.add_argument("--mics", default=None, metavar="I,J,...")
    f.add_argument("--k-cap", type=int, default=K_CAP)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--device", choices=("cuda", "cpu"), default=None)
    f.add_argument("--kappa", type=float, default=0.0, help="floor push (spec, calibrated in A)")
    f.add_argument("--wander", default=FV.DEFAULT_WANDER)
    f.add_argument("--channel-gains", default=CHANNEL_GAINS)
    f.add_argument(
        "--channel-gains-band", choices=("above_500", "full", "transfer"), default="transfer"
    )
    f.add_argument("--frame-stride", type=int, default=4)
    f.add_argument("--chunk-frames", type=int, default=16)
    f.add_argument("--work-rate", type=int, default=32000)
    f.add_argument("--threads", type=int, default=4)
    f.add_argument("--adam-steps", type=int, default=1500)
    f.add_argument("--adam-lr", type=float, default=0.02)
    f.add_argument("--adam-batch", type=int, default=8)
    f.add_argument("--lbfgs-iters", type=int, default=600)
    f.add_argument("--lbfgs-frames", type=int, default=64)
    f.add_argument("--polish-iters", type=int, default=150)
    f.add_argument("--progress", type=int, default=0)
    f.add_argument(
        "--time-eval", action="store_true", help="time ONE objective call on the pool and exit"
    )
    r = sub.add_parser("reduce", help="lowest objective over the restarts")
    r.add_argument("--out", required=True)
    r.add_argument("--name", default=NAME)
    args = ap.parse_args(argv)
    return cmd_fit(args) if args.cmd == "fit" else cmd_reduce(args)


if __name__ == "__main__":
    raise SystemExit(main())
