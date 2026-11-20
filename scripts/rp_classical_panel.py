#!/usr/bin/env python3
"""Training-free estimators on the paper's real validation panel (``rp_real``).

Scores the five classical estimators of :mod:`experiments.classical_rps.predictors`
(pYIN, cepstrum, harmonic product spectrum, matched comb filters, NMF) and the
optimal-transport inverse harmonic clustering of :mod:`experiments.otmp_baseline`
(``adapted`` preset) on the 52 clips x 8 microphones of ``rp-real-valid-v1``,
with the training loop's own reduction: per-frame errors under one PIT rotor
assignment per clip (``metrics.rps.batched_pit_frame_errors``), frame regimes
from the target (``metrics.regimes``), views and the macro aggregate of
``conf/validation/rp_real.yaml`` (``training.validation._reduce_views``). The
numbers are therefore read exactly as the rows of the paper's real-data tables.

Each estimator sees one microphone (the panel's mono items). One prediction JSON
per (method, item) lands under ``<out>/units/raw`` through ``utils.gridrun``, so a
killed run resumes. Cost: about 1.5 CPU-s per classical unit and about 370 CPU-s
per OT unit (416 units each); the OT rows are a cluster job::

    PYTHONPATH=src:scripts python scripts/rp_classical_panel.py --jobs 32 \\
        --out results/rp_classical_panel
"""

from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

METHODS = ("pyin", "cepstral", "hps", "matched_filter", "nmf", "otmp")
N_ROTORS = 4
VIEWS = (
    "real_macro",
    "dregon",
    "dregon_zero",
    "dregon_standby",
    "dregon_ramp",
    "dregon_cruise",
    "md2",
    "md2_zero",
    "md2_standby",
    "md2_ramp",
    "md2_cruise",
)

_DATASET: Any = None


def _dataset(params: dict[str, Any]) -> Any:
    """The panel's dataset, built once per worker process."""
    global _DATASET
    if _DATASET is None:
        from data_processing.frame_datasets import DregonLMFrameDataset

        _DATASET = DregonLMFrameDataset(**params)
    return _DATASET


def worker(unit: Any) -> dict[str, Any]:
    """One (method, panel item) prediction in rev/s, on the method's own grid."""
    method = str(unit.params["method"])
    item = int(unit.params["item"])
    audio = np.asarray(
        _dataset(dict(unit.params["dataset"]))[item]["mixture"].data, dtype=np.float32
    ).reshape(-1)
    if method == "otmp":
        from experiments.otmp_baseline.estimate import adapted_drone_config, estimate_clip

        cfg = adapted_drone_config()
        _times, pitches = estimate_clip(audio.astype(np.float64), cfg.sample_rate, cfg)
        pred = np.asarray(pitches, dtype=np.float64)
    else:
        from experiments.classical_rps.predictors import CLASSICAL_TRACKERS

        pred = np.asarray(CLASSICAL_TRACKERS[method](audio), dtype=np.float64)
    return {"method": method, "item": item, "pred": pred.tolist()}


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--out", type=Path, default=Path("results/rp_classical_panel"))
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    import torch
    from omegaconf import OmegaConf
    from rp_panel_eval import _panel

    from data_processing.frames import meta_dict
    from experiments.rps_bench import resample_like_metric
    from losses._common import get_tensor
    from metrics.regimes import frame_regimes
    from metrics.rps import batched_pit_frame_errors
    from training.validation import _reduce_views, build_validation_plan
    from utils.gridrun import Unit, run_grid

    panel = _panel("rp_real", "rp_scv2_all")
    plan = build_validation_plan(panel)
    datasets = OmegaConf.to_container(panel.datasets, resolve=True)
    assert isinstance(datasets, dict)
    (spec,) = datasets.values()
    params = dict(spec["params"])
    ds = _dataset(params)
    if len(ds) != plan.size:
        raise SystemExit(f"panel has {plan.size} items, the dataset {len(ds)}")
    target = torch.stack([get_tensor(ds[i], "rps").float() for i in range(plan.size)])
    n_t = int(target.shape[-1])
    regime = frame_regimes(target, plan.frame_rate).numpy()

    units = [
        Unit(uid=f"{m}_item{i:03d}", params={"method": m, "item": i, "dataset": params})
        for m in args.methods
        for i in range(plan.size)
    ]
    result = run_grid(
        units, worker, args.out / "units", jobs=args.jobs, summarize=lambda rows: {"n": len(rows)}
    )
    if result.n_failed:
        raise SystemExit(f"{result.n_failed} unit(s) failed; see {args.out}/units/raw/*.err")

    raw = args.out / "units" / "raw"
    report: dict[str, Any] = {}
    for m in args.methods:
        pred = np.empty((plan.size, N_ROTORS, n_t), dtype=np.float64)
        resampled = False
        for i in range(plan.size):
            p = np.asarray(json.loads((raw / f"{m}_item{i:03d}.json").read_text())["pred"])
            if p.shape[-1] != n_t:
                p = resample_like_metric(p, n_t)
                resampled = True
            pred[i] = p
        err = batched_pit_frame_errors(torch.from_numpy(pred).float(), target).numpy()
        scores = _reduce_views(plan, err, regime)
        report[m] = dict(scores={v: scores[v] for v in VIEWS}, frames_resampled=resampled)
        print(m, " ".join(f"{v} {scores[v]:.2f}" for v in VIEWS), flush=True)

    metas = [meta_dict(ds[i]) for i in range(0, plan.size, 8)]
    (args.out / "report.json").write_text(
        json.dumps(
            dict(
                panel="conf/validation/rp_real.yaml",
                dataset=params["data_dir"],
                n_items=plan.size,
                frame_rate=float(plan.frame_rate),
                clips=metas,
                methods=report,
            ),
            indent=1,
            default=str,
        )
    )
    head = "| method | " + " | ".join(VIEWS) + " |"
    rows = [head, "|---|" + "---:|" * len(VIEWS)]
    rows += [
        f"| {m} | " + " | ".join(f"{report[m]['scores'][v]:.2f}" for v in VIEWS) + " |"
        for m in args.methods
    ]
    (args.out / "report.md").write_text(
        "# Training-free estimators on rp_real (PIT-MAE, Hz)\n\n" + "\n".join(rows) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
