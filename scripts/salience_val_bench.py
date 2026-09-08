#!/usr/bin/env python
"""Time the unified full-panel validator on trained salience checkpoints.

Temporary campaign script (docs/experiments/paper-regime-matrix.md, salience
rerun gate): for each experiment, load its best checkpoint through ``zoo``,
run ``training.validation.validate_rps`` with the ``salience_rps`` readout
over the frozen panel of ``conf/validation/rps_unified.yaml``, and record
wall time (cold and warm), peak VRAM and every subset score. Also spot-checks
that the validator's real_r3 score equals the per-clip ``predict_rps`` +
``pit_mae`` route ``scripts/rps_dump.py`` uses, on the first real clips.

    python scripts/salience_val_bench.py --experiments hb_sal_hf0_orig,hf0_l2_r2_s0 \
        --batch-size 32 --out results/salience_val_bench
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import tdseries as td
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

import zoo
from data_processing.collate import frame_collate
from experiments.rps_bench import pit_mae
from losses._common import get_tensor
from metrics.rps import batched_pit_mae
from training.config import build_losses, register_configs
from training.validation import build_validation_plan, rps_readout_for, validate_rps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiments", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", default="results/salience_val_bench")
    ap.add_argument("--spot-clips", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    cuda = device.type == "cuda"
    register_configs()
    val_cfg = OmegaConf.load("conf/validation/rps_unified.yaml")
    plan = build_validation_plan(val_cfg)
    loader = DataLoader(
        plan.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=frame_collate,
    )
    readout = rps_readout_for("salience_rps")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    for name in args.experiments.split(","):
        fm = zoo.load(name, ckpt="best", device=device)
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=str(Path("conf").resolve()), version_base=None):
            cfg = compose(config_name="config", overrides=[f"experiment={name}"])
        loss_fn = build_losses(cfg.loss).to(device)
        model = fm.model.to(device).eval()
        row: dict = {"batch_size": args.batch_size}
        for label in ("cold", "warm"):
            if cuda:
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            scores, val_loss = validate_rps(
                model=model,
                codec=fm.codec,
                loss_fn=loss_fn,
                valid_loader=loader,
                plan=plan,
                device=device,
                amp=cuda,
                amp_dtype=torch.float16 if cuda else None,
                readout=readout,
            )
            if cuda:
                torch.cuda.synchronize(device)
            row[f"{label}_s"] = time.perf_counter() - t0
            if cuda:
                row[f"{label}_peak_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 2**30
        row["scores"] = scores
        row["val_loss"] = val_loss

        # Spot check against the per-clip deployed route on the first real clips.
        spot = []
        for i in range(args.spot_clips):
            frame = cast(td.Frame, plan.dataset[i])
            audio = get_tensor(frame, "mixture")[None].to(device)
            with torch.no_grad():
                pred = (
                    cast(Any, model).predict_rps(audio)[0].float().cpu().numpy().astype(np.float64)
                )
            gt = np.asarray(get_tensor(frame, "rps"), dtype=np.float64)
            spot.append(float(pit_mae(pred, gt)))
        sub = DataLoader(
            Subset(plan.dataset, list(range(args.spot_clips))),
            batch_size=args.spot_clips,
            collate_fn=frame_collate,
        )
        batch = next(iter(sub)).map_data(lambda v: v.to(device))
        inputs = fm.codec.to_inputs(batch)
        with torch.inference_mode(), torch.autocast(device.type, enabled=cuda, dtype=torch.float16):
            pred_frame = fm.codec.to_frame(fm.codec.call_model(model, inputs), batch)
        batched = batched_pit_mae(readout(model, pred_frame, inputs), get_tensor(batch, "rps"))
        row["spot_per_clip"] = spot
        row["spot_batched_fp16"] = [float(x) for x in batched.cpu()]
        summary[name] = row
        print(json.dumps({name: row}, indent=1), flush=True)
        del model, fm
        if cuda:
            torch.cuda.empty_cache()

    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
