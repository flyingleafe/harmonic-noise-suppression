"""Build a ``noise-prior-bank/1`` bank: gated draws from the generic drone prior.

    PYTHONPATH=src python scripts/prior_bank_build.py --n 16384 --seed 20261002 \\
        --workers 16 --out data/rig_banks/prior_v1_n16384.npz

Each worker draws rigs with its own stream (``SeedSequence([seed, worker])``),
gates every draw on ONE shared trajectory pool (``experiments.noise_model.
identifiability``: ``threshold_db`` over everything else for ``min_orders``
orders per rotor in ``pass_frac`` of its eligible frames; the pool is drawn
from the trajectory hyperprior with ``SeedSequence([seed, "pool"])``), and
keeps its quota of passing rigs. The bank is written by
:func:`data_processing.noise_model.prior_bank.write_prior_bank`; a report with
the draw statistics goes beside it (``<out>.report.json``) and under
``results/prior_rigs/``. ``--laws`` scores the first rigs under the laws
(``drone_prior.law_scores``) into the report.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

SEED = 20261002


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _worker(args: tuple[int, int, int, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    """Draw and gate until ``quota`` rigs pass; return them with statistics."""
    seed, worker, quota, gate_cfg, pool_cfg = args
    from experiments.noise_model import drone_prior as PR
    from experiments.noise_model.identifiability import GateConfig, frame_pool, gate

    rng_pool = np.random.default_rng(np.random.SeedSequence([seed, 7919]))
    wins = PR.trajectory_windows(rng_pool, int(pool_cfg["windows"]), float(pool_cfg["duration_s"]))
    pool = frame_pool(wins, n_frames=int(pool_cfg["frames"]), seed=seed)
    cfg = GateConfig(**gate_cfg)
    rng = np.random.default_rng(np.random.SeedSequence([seed, worker]))
    out: list[dict[str, Any]] = []
    tried = 0
    reasons: dict[str, int] = {}
    t0 = time.time()
    while len(out) < quota:
        tried += 1
        rig = PR.sample_prior(rng)
        res = gate(rig, pool, cfg)
        if res.passed:
            rig["_prior"]["gate"] = dict(
                frames_ok=res.frames_ok.tolist(),
                eligible=res.eligible.tolist(),
                threshold_db=cfg.threshold_db,
            )
            rig["_prior"]["seed"] = int(seed * 1000 + worker)
            out.append(rig)
        else:
            key = str(res.reason).split(" (")[0]
            reasons[key] = reasons.get(key, 0) + 1
    return dict(worker=worker, rigs=out, tried=tried, reasons=reasons, seconds=time.time() - t0)


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--threshold-db", type=float, default=3.0)
    ap.add_argument("--min-orders", type=int, default=2)
    ap.add_argument("--pass-frac", type=float, default=0.8)
    ap.add_argument("--pool-windows", type=int, default=8)
    ap.add_argument("--pool-frames", type=int, default=80)
    ap.add_argument("--pool-duration-s", type=float, default=8.0)
    ap.add_argument(
        "--laws", type=int, default=64, help="rigs scored under the laws for the report"
    )
    return ap.parse_args()


def _draw_all(a: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gate_cfg = dict(
        threshold_db=a.threshold_db,
        min_orders=a.min_orders,
        pass_frac=a.pass_frac,
        min_eligible_frames=5,
    )
    pool_cfg = dict(windows=a.pool_windows, frames=a.pool_frames, duration_s=a.pool_duration_s)
    quotas = [a.n // a.workers + (1 if w < a.n % a.workers else 0) for w in range(a.workers)]
    jobs = [(a.seed, w, q, gate_cfg, pool_cfg) for w, q in enumerate(quotas) if q > 0]
    t0 = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        parts = pool.map(_worker, jobs)
    rigs = [r for part in parts for r in part["rigs"]]
    reasons: dict[str, int] = {}
    for p in parts:
        for k, v in p["reasons"].items():
            reasons[k] = reasons.get(k, 0) + v
    stats = dict(
        tried=sum(p["tried"] for p in parts),
        rejected=reasons,
        seconds=time.time() - t0,
        gate=gate_cfg,
        pool=pool_cfg,
    )
    return rigs, stats


def main() -> None:
    a = _args()
    from data_processing.noise_model.prior_bank import write_prior_bank
    from experiments.noise_model.drone_prior import PriorSpec, law_scores

    out = a.out or Path(f"data/rig_banks/prior_v1_n{a.n}.npz")
    rigs, stats = _draw_all(a)
    print(f"{len(rigs)} rigs of {stats['tried']} draws in {stats['seconds']:.0f} s", flush=True)
    scores = [{law: [v, z] for law, v, z in law_scores(r)} for r in rigs[: max(0, a.laws)]]
    header = dict(
        builder="scripts/prior_bank_build.py",
        git=_git_sha(),
        host=platform.node(),
        built_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        seed=a.seed,
        workers=a.workers,
        n=len(rigs),
        spec=asdict(PriorSpec()),
        law_scores=scores,
        **stats,
    )
    path = write_prior_bank(out, rigs, header)
    report = Path("results/prior_rigs") / f"{path.stem}.report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(header, indent=1, default=float))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB) and {report}", flush=True)


if __name__ == "__main__":
    main()
