#!/usr/bin/env python
"""LR range test for one rp experiment (paper-real-protocol.md, "Max LR per model").

Composes the experiment as ``train.py`` does (model, loss, data stream), runs
``--steps`` optimizer steps with the LR growing geometrically from
``--lr-start`` to ``--lr-end`` (Smith 2017; fastai ``lr_find``), logging the
EMA of the training loss; no validation. Recommendation = LR of the steepest
sustained descent of the smoothed loss, capped at 1/3 of the LR where the
loss first rises ``--divergence`` (relative) above its running minimum, then
snapped to the half-decade grid. Output: ``results/rp_lr_range/<exp>.json``.

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


def snap_half_decade(lr: float) -> float:
    """Nearest value of the half-decade grid in log space."""
    return min(HALF_DECADES, key=lambda g: abs(math.log10(g) - math.log10(lr)))


def recommend(lrs: np.ndarray, loss_ema: np.ndarray, divergence: float, window: int) -> dict:
    """Steepest sustained descent, capped at 1/3 of the divergence LR.

    ``loss_ema`` is already smoothed; the slope is d(loss)/d(log lr) over
    ``window`` steps. Divergence = first step where the loss exceeds its
    running minimum by ``divergence`` (relative) after that minimum.
    """
    log_lr = np.log10(lrs)
    n = len(lrs)
    run_min = np.minimum.accumulate(loss_ema)
    diverged = np.flatnonzero(loss_ema > run_min * (1.0 + divergence))
    end = int(diverged[0]) if diverged.size else n
    cap = lrs[end - 1] / 3.0 if diverged.size else lrs[-1]
    slopes = np.full(n, np.nan)
    for i in range(window, end):
        d_lr = log_lr[i] - log_lr[i - window]
        if d_lr > 0:
            slopes[i] = (loss_ema[i] - loss_ema[i - window]) / d_lr
    valid = np.flatnonzero(np.isfinite(slopes) & (lrs <= cap))
    if valid.size == 0:
        raise RuntimeError("no descending segment below the divergence cap; widen the range")
    best = int(valid[np.argmin(slopes[valid])])
    raw = float(lrs[best])
    return {
        "lr_steepest": raw,
        "lr_divergence": float(lrs[end - 1]) if diverged.size else None,
        "lr_cap": float(cap),
        "lr_recommended": float(snap_half_decade(min(raw, cap))),
        "step_steepest": best,
        "step_divergence": int(end) if diverged.size else None,
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
    rec = recommend(np.asarray(lrs), np.asarray(ema), args.divergence, args.window)
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
        "divergence": args.divergence,
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
        f"{args.experiment}: steepest {rec['lr_steepest']:.2e}, cap {rec['lr_cap']:.2e} "
        f"-> recommended max LR {rec['lr_recommended']:.0e} ({out})",
        flush=True,
    )
    return result


def summarize() -> None:
    """One line per committed range test: the numbers the sibling .md files quote."""
    for path in sorted((REPO_ROOT / "results" / "rp_lr_range").glob("*.json")):
        r = json.loads(path.read_text())
        if r["step_divergence"] is None:
            diverge = "none"
        else:
            diverge = f"{r['lr_divergence']:.2e} @{r['step_divergence']}"
        print(
            f"{r['experiment']:28s} steps {r['steps_run']:4d}/{r['steps_requested']} "
            f"batch {r['batch_size']}x{r['grad_accum_steps']} "
            f"steepest {r['lr_steepest']:.2e} @{r['step_steepest']} "
            f"diverge {diverge} cap {r['lr_cap']:.2e} "
            f"-> {r['lr_recommended']:.0e}"
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
        "--divergence", type=float, default=0.5, help="relative rise over the running minimum"
    )
    p.add_argument("--window", type=int, default=100, help="slope window in steps")
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
