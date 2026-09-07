#!/usr/bin/env python3
"""Temporary R3/R4 unified-regime A100 benchmark driver."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


def _gpu_sample(case: str) -> dict[str, Any]:
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=5,
    ).strip()
    util, memory, power = (float(value.strip()) for value in raw.split(","))
    return {
        "unix": time.time(),
        "case": case,
        "gpu_util": util,
        "memory_mib": memory,
        "power_w": power,
    }


def _phase_stats(samples: list[dict[str, Any]], windows: list[tuple[float, float]]) -> dict:
    values = [
        sample["gpu_util"]
        for sample in samples
        if any(start <= sample["unix"] <= stop for start, stop in windows)
    ]
    if not values:
        return {"samples": 0}
    ordered = sorted(values)
    return {
        "samples": len(values),
        "mean_gpu_util": statistics.mean(values),
        "median_gpu_util": statistics.median(values),
        "p10_gpu_util": ordered[max(0, int(0.1 * len(ordered)) - 1)],
        "p90_gpu_util": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "max_gpu_util": max(values),
        "fraction_ge_90": sum(value >= 90 for value in values) / len(values),
        "fraction_zero": sum(value == 0 for value in values) / len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/unified_regime_benchmark"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--context-rounds", type=int, default=3)
    parser.add_argument("--other-rounds", type=int, default=2)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs_root = args.out / "runs"
    runs_root.mkdir(exist_ok=True)

    cases = [
        ("r4_scv2_1s_b128", "real_r4_scv2_unified", 1.0, 128, args.context_rounds),
        ("r4_scv2_2s_b64", "real_r4_scv2_unified", 2.0, 64, args.context_rounds),
        ("r4_tm_1s_b128", "real_r4_tm_unified", 1.0, 128, args.context_rounds),
        ("r4_tm_2s_b64", "real_r4_tm_unified", 2.0, 64, args.context_rounds),
        ("r4_gru_1s_b128", "real_r4_gru_unified", 1.0, 128, args.other_rounds),
        ("r4_sc_1s_b128", "real_r4_sc_unified", 1.0, 128, args.other_rounds),
    ]
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
        for name, experiment, duration, batch_size, rounds in cases:
            phase["case"] = name
            command = [
                sys.executable,
                "train.py",
                f"experiment={experiment}",
                f"experiment_name=bench_{name}",
                f"results_root={runs_root}",
                f"epochs={rounds}",
                f"validation.max_optimizer_steps={rounds * 500}",
                f"batch_size={batch_size}",
                f"num_workers={args.workers}",
                "artifacts.enabled=false",
                "logging.enabled=false",
            ]
            if duration != 1.0:
                command.append(f"data.train.params.duration_s={duration}")
            started = time.time()
            with (args.out / f"{name}.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
            stopped = time.time()
            if result.returncode != 0:
                raise RuntimeError(f"{name} failed; inspect {args.out / f'{name}.log'}")

            history_path = runs_root / f"bench_{name}" / "validation_history.jsonl"
            history = [json.loads(line) for line in history_path.read_text().splitlines()]
            if len(history) != rounds:
                raise RuntimeError(f"{name} produced {len(history)} rounds, expected {rounds}")
            numeric = [
                value
                for row in history
                for key, value in row.items()
                if key.startswith(("train/", "val/")) and isinstance(value, (int, float))
            ]
            if not all(math.isfinite(value) for value in numeric):
                raise RuntimeError(f"{name} produced a non-finite metric")
            train_windows = [
                (row["perf/train_started_unix"], row["perf/train_finished_unix"]) for row in history
            ]
            validation_windows = [
                (row["perf/validation_started_unix"], row["perf/validation_finished_unix"])
                for row in history
            ]
            case_samples = [sample for sample in telemetry if sample.get("case") == name]
            summaries.append(
                {
                    "case": name,
                    "experiment": experiment,
                    "duration_s": duration,
                    "batch_size": batch_size,
                    "optimizer_steps": history[-1]["optimizer_step"],
                    "wall_s": stopped - started,
                    "train_s": [row["perf/train_s"] for row in history],
                    "validation_s": [row["perf/validation_s"] for row in history],
                    "peak_allocated_gb": max(row["perf/peak_allocated_gb"] for row in history),
                    "peak_reserved_gb": max(row["perf/peak_reserved_gb"] for row in history),
                    "nvidia_smi_peak_mib": max(
                        (sample["memory_mib"] for sample in case_samples if "memory_mib" in sample),
                        default=0,
                    ),
                    "training_gpu": _phase_stats(case_samples, train_windows),
                    "validation_gpu": _phase_stats(case_samples, validation_windows),
                    "scores": [
                        {
                            key: value
                            for key, value in row.items()
                            if key.startswith("val/")
                            and not key.endswith(("_log", "_median", "_log_median"))
                        }
                        for row in history
                    ],
                }
            )
            (args.out / "summary.partial.json").write_text(json.dumps(summaries, indent=2) + "\n")
    finally:
        phase["case"] = "done"
        stop.set()
        sampler.join(timeout=10)
        (args.out / "gpu_telemetry.json").write_text(json.dumps(telemetry, indent=2) + "\n")

    metadata = {
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True
        ).strip(),
        "cpu_count": len(__import__("os").sched_getaffinity(0)),
        "cases": summaries,
    }
    (args.out / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
