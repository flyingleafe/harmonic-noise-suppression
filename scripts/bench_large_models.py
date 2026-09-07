#!/usr/bin/env python3
"""Temporary sequential 2-second batch/memory benchmark for large RPS models."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from bench_unified_regime import _gpu_sample, _phase_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/large_model_benchmark"))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs_root = args.out / "runs"
    runs_root.mkdir(exist_ok=True)

    cases: list[dict[str, Any]] = []
    for experiment in ("real_r4_scv2_unified", "real_r4_tm_unified", "real_r4_gru_unified"):
        short = experiment.removeprefix("real_r4_").removesuffix("_unified")
        for batch in (128, 256):
            cases.append(
                {
                    "name": f"reg_{short}_2s_b{batch}",
                    "experiment": experiment,
                    "batch": batch,
                    "unified": True,
                }
            )
    for experiment in (
        "g2_hcqt_transformer",
        "hb_sal_hppnet_orig",
        "hb_sal_hf0_orig",
        "hb_sal_multif0_l4",
    ):
        short = experiment.removeprefix("hb_sal_")
        for batch in (128, 256):
            cases.append(
                {
                    "name": f"large_{short}_2s_b{batch}",
                    "experiment": experiment,
                    "batch": batch,
                    "unified": False,
                }
            )

    telemetry: list[dict[str, Any]] = []
    phase = {"case": "setup"}
    stop = threading.Event()

    def sample_loop() -> None:
        while not stop.wait(0.5):
            try:
                telemetry.append(_gpu_sample(phase["case"]))
            except Exception as error:
                telemetry.append({"unix": time.time(), "case": phase["case"], "error": repr(error)})

    sampler = threading.Thread(target=sample_loop, daemon=True)
    sampler.start()
    summaries: list[dict[str, Any]] = []
    try:
        for case in cases:
            name = case["name"]
            batch = int(case["batch"])
            phase["case"] = name
            command = [
                sys.executable,
                "train.py",
                f"experiment={case['experiment']}",
                f"experiment_name=bench_{name}",
                f"results_root={runs_root}",
                "epochs=2",
                f"batch_size={batch}",
                f"num_workers={args.workers}",
                "artifacts.enabled=false",
                "logging.enabled=false",
                "+data.train.params.duration_s=2.0",
            ]
            if case["unified"]:
                command.append("validation.max_optimizer_steps=1000")
            else:
                command.extend(
                    [
                        f"samples_per_validation={batch * 50}",
                        "data.valid.params.flatten_channels=false",
                        "+data.valid.params.channel=0",
                        "data.valid_batch_size=16",
                    ]
                )
            started = time.time()
            log_path = args.out / f"{name}.log"
            with log_path.open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
            stopped = time.time()
            case_samples = [sample for sample in telemetry if sample.get("case") == name]
            history_path = runs_root / f"bench_{name}" / "validation_history.jsonl"
            history = (
                [json.loads(line) for line in history_path.read_text().splitlines()]
                if history_path.exists()
                else []
            )
            train_windows = [
                (row["perf/train_started_unix"], row["perf/train_finished_unix"])
                for row in history
                if "perf/train_started_unix" in row
            ]
            numeric = [
                value
                for row in history
                for key, value in row.items()
                if key.startswith(("train/", "val/")) and isinstance(value, (int, float))
            ]
            nonfinite = [value for value in numeric if not math.isfinite(value)]
            summaries.append(
                {
                    **case,
                    "returncode": result.returncode,
                    "wall_s": stopped - started,
                    "history_rounds": len(history),
                    "optimizer_steps": history[-1].get("optimizer_step") if history else None,
                    "nonfinite": bool(nonfinite),
                    "nonfinite_scores": history[-1].get("validation/nonfinite_scores", [])
                    if history
                    else [],
                    "peak_allocated_gb": max(
                        (row.get("perf/peak_allocated_gb", 0.0) for row in history),
                        default=0.0,
                    ),
                    "peak_reserved_gb": max(
                        (row.get("perf/peak_reserved_gb", 0.0) for row in history),
                        default=0.0,
                    ),
                    "nvidia_smi_peak_mib": max(
                        (sample["memory_mib"] for sample in case_samples if "memory_mib" in sample),
                        default=0.0,
                    ),
                    "training_gpu": _phase_stats(case_samples, train_windows),
                    "log_tail": log_path.read_text(errors="replace").splitlines()[-20:]
                    if result.returncode != 0 or not history
                    else [],
                }
            )
            (args.out / "summary.partial.json").write_text(json.dumps(summaries, indent=2) + "\n")
    finally:
        stop.set()
        sampler.join(timeout=10)
        (args.out / "gpu_telemetry.json").write_text(json.dumps(telemetry, indent=2) + "\n")

    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
