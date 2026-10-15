#!/usr/bin/env python
"""LR range test for one rp experiment (paper-real-protocol.md, "Max LR per model").

Composes the experiment as ``train.py`` does (model, loss, data stream), runs
``--steps`` optimizer steps with the LR growing geometrically from
``--lr-start`` to ``--lr-end`` (Smith 2017; fastai ``lr_find``), logging the
EMA of the training loss; no validation. Recommendation = a tenth of the LR
at which the smoothed loss leaves its floor for good (stays ``--rise``
relative above the minimum for ``--window`` steps), snapped to the
half-decade grid. Output:
``results/rp_lr_range/<exp>.json`` (the raw curve; ``--summary`` recomputes).

Run from the project root on a GPU: ``python scripts/lr_range_test.py --experiment rp_hppnet_l2_all``
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp.grad_scaler import GradScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from training.config import (  # noqa: E402
    build_dataset,
    build_losses,
    build_task_and_codec,
    instantiate_model,
    register_configs,
)
from training.loop import _forward, _make_loader, _to_device, get_optimizer  # noqa: E402

HALF_DECADES = [10 ** (k / 2) for k in range(-10, 1)]  # 1e-5 .. 1 in sqrt(10) steps
READ_BETA = 0.99  # EMA factor the recommendation is read from (~100-step memory)


def snap_half_decade(lr: float) -> float:
    """Nearest value of the half-decade grid in log space."""
    return min(HALF_DECADES, key=lambda g: abs(math.log10(g) - math.log10(lr)))


def recommend(lrs: np.ndarray, loss_ema: np.ndarray, rise: float, window: int) -> dict:
    """Max LR = a tenth of the rise LR, at most the floor LR, snapped to the half-decade grid.

    ``loss_ema`` is already smoothed. Floor = its global minimum. Rise = the
    first step after the floor from which the loss stays above
    ``floor * (1 + rise)`` for ``window`` consecutive steps. With AdamW and
    gradient clipping the loss does not blow up inside the swept range, it
    drifts up, so the rise is read at a small relative threshold rather than
    at a divergence; a steepest-slope reading is not used because the loss
    falls at any LR during the first few hundred steps. A curve that never
    rises (Basic Pitch) is capped at the LR of its floor instead of at a
    tenth of the sweep's end. The plateau's lower edge (first step inside
    the same band) is reported with it.
    """
    n = len(lrs)
    floor_step = int(np.argmin(loss_ema))
    band = loss_ema[floor_step] * (1.0 + rise)
    above = loss_ema > band
    rise_step = None
    for i in range(floor_step, n - window + 1):
        if above[i : i + window].all():
            rise_step = i
            break
    lr_rise = float(lrs[rise_step]) if rise_step is not None else float(lrs[-1])
    plateau_start = int(np.flatnonzero(loss_ema <= band)[0])
    return {
        "lr_plateau_start": float(lrs[plateau_start]),
        "lr_floor": float(lrs[floor_step]),
        "lr_rise": float(lrs[rise_step]) if rise_step is not None else None,
        "lr_recommended": float(snap_half_decade(min(lr_rise / 10.0, float(lrs[floor_step])))),
        "step_plateau_start": plateau_start,
        "step_floor": floor_step,
        "step_rise": rise_step,
        "loss_floor": float(loss_ema[floor_step]),
    }


def compose_experiment(name: str):
    from hydra import compose, initialize_config_dir

    register_configs()
    overrides = [f"experiment={name}", "~schema", "validation=disabled"]
    with initialize_config_dir(config_dir=str(REPO_ROOT / "conf"), version_base=None):
        return compose(config_name="config", overrides=overrides)


def run(args: argparse.Namespace) -> dict:
    cfg = compose_experiment(args.experiment)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(cfg.seed))
    _task, codec = build_task_and_codec(cfg.model)
    model = instantiate_model(cfg.model).to(device)
    loss_fn = build_losses(cfg.loss).to(device)
    batch_size = int(cfg.data.batch_size or cfg.batch_size)
    grad_accum = max(1, int(cfg.grad_accum_steps))
    workers = cfg.data.num_workers if cfg.data.num_workers is not None else cfg.num_workers
    loader = _make_loader(
        build_dataset(cfg.data.train), batch_size=batch_size, num_workers=int(workers), shuffle=True
    )
    optimizer = get_optimizer(
        model,
        name=cfg.optim.optimizer,
        lr=args.lr_start,
        weight_decay=float(cfg.optim.weight_decay),
        extra_params=dict(cfg.optim.optimizer_params or {}),
    )
    amp = bool(cfg.amp) and device.type == "cuda"
    scaler = GradScaler(device.type, enabled=amp)
    ratio = (args.lr_end / args.lr_start) ** (1.0 / max(args.steps - 1, 1))
    lrs: list[float] = []
    losses: list[float] = []
    ema: list[float] = []
    ema_val = 0.0
    model.train()
    it = iter(loader)
    t0 = time.time()
    for step in range(args.steps):
        lr = args.lr_start * ratio**step
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(grad_accum):
            batch = _to_device(next(it), device)
            pred = _forward(codec, model, batch, device=device, amp=amp, amp_dtype=torch.float16)
            loss = loss_fn(pred, batch)
            if not torch.isfinite(loss.detach()):
                total = float("nan")
                break
            scaler.scale(loss / grad_accum).backward()
            total += float(loss.detach()) / grad_accum
        if not math.isfinite(total):
            print(f"step {step}: non-finite loss at lr {lr:.2e}; stopping", flush=True)
            break
        if cfg.grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
        scaler.step(optimizer)
        scaler.update()
        ema_val = args.beta * ema_val + (1 - args.beta) * total
        lrs.append(lr)
        losses.append(total)
        ema.append(ema_val / (1 - args.beta ** (step + 1)))  # bias-corrected
        if step % 100 == 0:
            print(f"step {step:5d} lr {lr:.2e} loss {total:.4f} ema {ema[-1]:.4f}", flush=True)
        if ema[-1] > args.stop_factor * min(ema):
            print(
                f"step {step}: loss exploded ({ema[-1]:.3g} > {args.stop_factor}x min); stopping",
                flush=True,
            )
            break
    rec = recommend(np.asarray(lrs), ema_curve(losses, READ_BETA), args.rise, args.window)
    result = {
        "experiment": args.experiment,
        "model": str(cfg.model.params.get("name", cfg.model.get("_target_", ""))),
        "steps_run": len(lrs),
        "steps_requested": args.steps,
        "lr_start": args.lr_start,
        "lr_end": args.lr_end,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum,
        "ema_beta": args.beta,
        "rise": args.rise,
        "window": args.window,
        "seconds": time.time() - t0,
        "device": str(device),
        **rec,
        "lr": lrs,
        "loss": losses,
        "loss_ema": ema,
    }
    out = REPO_ROOT / "results" / "rp_lr_range" / f"{args.experiment}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(
        f"{args.experiment}: floor {rec['lr_floor']:.2e}, rise {rec['lr_rise']} "
        f"-> recommended max LR {rec['lr_recommended']:.0e} ({out})",
        flush=True,
    )
    return result


def ema_curve(losses: list[float], beta: float) -> np.ndarray:
    """Zero-seeded, bias-corrected EMA of the per-step loss."""
    out = np.empty(len(losses))
    val = 0.0
    for i, x in enumerate(losses):
        val = beta * val + (1 - beta) * x
        out[i] = val / (1 - beta ** (i + 1))
    return out


def summarize() -> None:
    """One line per committed range test, recomputed from the raw (lr, loss) curve.

    The recommendation fields stored at run time are not trusted: runs before
    44c1aa89 seeded the EMA with the first loss (its first window looked like
    the steepest descent) and runs before the rise rule stored a steepest-slope
    reading; ``rise`` falls back to the default for them.
    """
    for path in sorted((REPO_ROOT / "results" / "rp_lr_range").glob("*.json")):
        r = json.loads(path.read_text())
        ema = ema_curve(r["loss"], READ_BETA)
        r.update(recommend(np.asarray(r["lr"]), ema, r.get("rise", 0.05), r["window"]))
        rise = "none" if r["step_rise"] is None else f"{r['lr_rise']:.2e} @{r['step_rise']}"
        print(
            f"{r['experiment']:28s} steps {r['steps_run']:4d}/{r['steps_requested']} "
            f"batch {r['batch_size']}x{r['grad_accum_steps']} "
            f"plateau {r['lr_plateau_start']:.2e} @{r['step_plateau_start']} "
            f"floor {r['loss_floor']:.3f} @ {r['lr_floor']:.2e} "
            f"rise {rise} -> {r['lr_recommended']:.0e}"
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--experiment", help="omit with --summary")
    p.add_argument("--summary", action="store_true", help="tabulate results/rp_lr_range/*.json")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr-start", type=float, default=1e-5)
    p.add_argument("--lr-end", type=float, default=1e-1)
    p.add_argument("--beta", type=float, default=0.98, help="EMA factor of the loss")
    p.add_argument(
        "--rise",
        type=float,
        default=0.05,
        help="relative rise over the floor that ends the plateau",
    )
    p.add_argument("--window", type=int, default=100, help="steps the rise must persist")
    p.add_argument(
        "--stop-factor",
        type=float,
        default=4.0,
        help="stop when the EMA exceeds this x its minimum",
    )
    args = p.parse_args()
    if args.summary:
        summarize()
    elif args.experiment:
        run(args)
    else:
        p.error("--experiment or --summary is required")


if __name__ == "__main__":
    main()
