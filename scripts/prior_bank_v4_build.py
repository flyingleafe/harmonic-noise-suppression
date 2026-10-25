"""Build a ``noise-prior-bank-v4/2`` bank: gated draws from the wide prior in noise model v4.

    PYTHONPATH=src python scripts/prior_bank_v4_build.py --n 16384 --workers 32

Each worker draws with :func:`experiments.noise_model.prior_v4.sample_prior_v4`
(draw ``attempt`` of worker ``w`` from ``SeedSequence([seed, w, attempt])``: the
prior_v1 laws, the v4 rules, the phase pedestal; no wind, which is the
``noise_v2`` render option ``wind_sc``) and gates every draw on ONE shared
trajectory pool (``experiments.noise_model.identifiability.gate``, the rule
from the flags, by default the agreed one of ``GateConfig``; the pool is
``--pool-windows`` windows of ``--pool-duration-s`` from the trajectory
hyperprior, ``SeedSequence([seed, 7919])``, ``--pool-frames`` frames). A draw
that fails is rejected and redrawn; a worker stops at its quota. The bank is
written by :func:`data_processing.noise_model.prior_bank_v4.write_prior_bank_v4`
(default ``data/rig_banks/prior_v4_n<N>.npz``). The report — draws and
rejections, every v1 law's scalar over all draws and over the kept ones, the
pedestal by order group, motor lines, floor and level ranges, a float16 round
trip of a sample of rigs — goes to ``results/prior_rigs/<stem>.report.json``
and into the bank's ``meta.header``. Publishing to dload is a separate step.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

#: The trajectory pool's stream tag (``scripts/prior_bank_build.py``'s).
POOL_TAG = 7919


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _worker(job: tuple[int, int, int, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    """Draw and gate until ``quota`` draws pass; the kept payloads, each
    draw's v1 scalars and verdict, and the rejections by failing rotor."""
    seed, worker, quota, gate_cfg, pool_cfg = job
    from experiments.noise_model import drone_prior as PR
    from experiments.noise_model.identifiability import GateConfig, frame_pool, gate, pool_coverage
    from experiments.noise_model.prior_v4 import sample_prior_v4

    rng_pool = np.random.default_rng(np.random.SeedSequence([seed, POOL_TAG]))
    wins = PR.trajectory_windows(rng_pool, int(pool_cfg["windows"]), float(pool_cfg["duration_s"]))
    pool = frame_pool(wins, n_frames=int(pool_cfg["frames"]), seed=seed)
    cfg = GateConfig(**gate_cfg)
    kept: list[dict[str, Any]] = []
    drawn: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    t0 = time.time()
    attempt = 0
    while len(kept) < quota:
        pl = sample_prior_v4(seed, worker, attempt)
        attempt += 1
        res = gate(pl, pool, cfg)
        drawn.append(dict(pl.pop("_prior_v1_drawn"), kept=bool(res.passed)))
        if res.passed:
            kept.append(pl)
            if len(kept) % 100 == 0 or len(kept) == quota:
                print(
                    f"worker {worker}: {len(kept)}/{quota} kept of {attempt} draws "
                    f"in {time.time() - t0:.0f} s",
                    flush=True,
                )
        else:
            key = str(res.reason).split(" passed")[0]  # "rotor r" or the untestable reason
            reasons[key] = reasons.get(key, 0) + 1
    return dict(
        worker=worker,
        kept=kept,
        drawn=drawn,
        tried=attempt,
        rejected=reasons,
        eligible=pool_coverage(pool, cfg).eligible.tolist(),
        seconds=time.time() - t0,
    )


def _q(x: Any) -> dict[str, float]:
    a = np.asarray(x, dtype=np.float64).ravel()
    if a.size == 0:
        return {"n": 0}
    qs = np.quantile(a, [0.0, 0.05, 0.5, 0.95, 1.0])
    return dict(
        n=int(a.size),
        mean=float(a.mean()),
        sd=float(a.std()),
        min=float(qs[0]),
        p05=float(qs[1]),
        p50=float(qs[2]),
        p95=float(qs[3]),
        max=float(qs[4]),
    )


def _stats(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """The report's counts and distributions over the kept rigs."""
    from data_processing.noise_model.prior_bank_v4 import SILENT_LINE_DB
    from experiments.noise_model import prior_v4 as PV4
    from experiments.noise_model import v4_priors as VP

    P = [pl["params"] for pl in payloads]
    motor = [np.asarray(p["profile_motor_db"]) for p in P]
    live = np.asarray([bool(np.any(m > SILENT_LINE_DB)) for m in motor])
    n_orders = np.asarray([len(p["motor_orders"]) for p in P])
    alpha = np.asarray([p["alpha_motor"] for p in P])
    s2a = np.stack([p["pedestal"]["sigma2_amp"] for p in P])
    s2p = np.stack([p["pedestal"]["sigma2_phase"] for p in P])
    w = np.stack([p["pedestal"]["width_phase_hz"] for p in P])
    K = s2a.shape[1]
    k = np.arange(1, K + 1)
    groups = []
    for lo, hi, cap in PV4.AMP_CAPS:
        sel = (k >= lo) & (k <= min(hi, K))
        if not np.any(sel):
            continue
        a = s2a[:, sel]
        groups.append(
            dict(
                orders=[int(lo), int(min(hi, K))],
                amp_cap=float(cap),
                amp_at_cap_frac=float(np.mean(a >= cap * (1.0 - 1e-9))),
                sigma2_amp=_q(a),
                sigma2_phase=_q(s2p[:, sel]),
                width_phase_hz=_q(w[:, sel]),
            )
        )
    aero = np.stack([np.asarray(p["profile_aero_db"]) for p in P])
    live_motor = np.concatenate([m[m > SILENT_LINE_DB] for m in motor] or [np.zeros(0)])
    return dict(
        n=len(payloads),
        motor_lines=dict(
            rigs_with_motor_lines=int(live.sum()),
            rigs_with_motor_lines_frac=float(live.mean()),
            rigs_silent_motor=int((~live).sum()),
            orders_per_rig=_q(n_orders),
            live_lines=int(sum(int(np.sum(m > SILENT_LINE_DB)) for m in motor)),
            profile_motor_db=_q(live_motor),
        ),
        alpha_motor=dict(
            **_q(alpha),
            abs_max=float(np.abs(alpha).max()),
            frac_abs_gt_2=float(np.mean(np.abs(alpha) > 2.0)),
            prior=dict(normal=list(VP.ALPHA_MOTOR), bound=VP.ALPHA_MOTOR_MAX),
        ),
        alpha_aero=_q([p["alpha_aero"] for p in P]),
        pedestal=dict(
            sigma2_phase_at_max_frac=float(np.mean(s2p >= VP.PHASE_SIGMA2_MAX * (1 - 1e-9))),
            width_at_max_frac=float(np.mean(w >= VP.W_MAX_HZ * (1 - 1e-9))),
            caps=dict(sigma2_phase=VP.PHASE_SIGMA2_MAX, width_hz=VP.W_MAX_HZ),
            by_group=groups,
        ),
        shaft=dict(
            sigma_nu_rad_s=_q([2.0 * np.pi * 80.0 * p["shaft"]["eps"] for p in P]),
            lam_per_s=_q([p["shaft"]["lam_ref_per_s"] for p in P]),
        ),
        floor=dict(
            mean_db=_q([p["floor"]["mean_db"] for p in P]),
            shape_sd_db=_q([p["floor"]["shape_sd_db"] for p in P]),
            b0=_q([p["floor"]["b0"] for p in P]),
            static_rel=_q([p["floor"]["static_rel"] for p in P]),
        ),
        levels=dict(
            profile_aero_db=_q(aero),
            profile_aero_max_db_per_rig=_q(aero.max(axis=(1, 2))),
        ),
    )


def _laws(drawn: list[dict[str, Any]]) -> dict[str, Any]:
    """Every v1 law's scalar draw over all draws and over the kept ones: what
    the gate selects."""
    kept = np.asarray([d["kept"] for d in drawn])
    out: dict[str, Any] = {}
    for key in drawn[0]:
        if key == "kept":
            continue
        x = np.asarray([d[key] for d in drawn], dtype=np.float64)
        out[key] = dict(all=_q(x), kept=_q(x[kept]), rejected=_q(x[~kept]))
    return out


def _round_trip(path: Path, payloads: list[dict[str, Any]], n: int = 64) -> dict[str, Any]:
    """Largest float16 deviations of a sample of rigs read back from the bank."""
    from data_processing.noise_model.prior_bank_v4 import PriorBankV4

    bank = PriorBankV4(path)
    pick = np.linspace(0, len(payloads) - 1, min(n, len(payloads))).astype(int)
    worst: dict[str, float] = {}

    def upd(key: str, a: Any, b: Any, rel: bool = False) -> None:
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        d = np.abs(a - b) / (np.abs(b) + 1e-12) if rel else np.abs(a - b)
        worst[key] = max(worst.get(key, 0.0), float(d.max()) if d.size else 0.0)

    for i in pick:
        q, p = bank.payload(int(i))["params"], payloads[int(i)]["params"]
        upd("profile_aero_db", q["profile_aero_db"], p["profile_aero_db"])
        upd("profile_motor_db", q["profile_motor_db"], p["profile_motor_db"])
        for key in ("sigma2_amp", "sigma2_phase", "gamma_amp_hz"):
            upd(f"pedestal.{key}_rel", q["pedestal"][key], p["pedestal"][key], rel=True)
        if q["motor_orders"] != p["motor_orders"]:
            raise RuntimeError(f"rig {i}: motor orders differ after the round trip")
    return dict(rigs=int(pick.size), max_abs_dev=worst)


def _args() -> argparse.Namespace:
    from experiments.noise_model.prior_v4 import SEED

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--threshold-db", type=float, default=None)
    ap.add_argument("--min-orders", type=int, default=None)
    ap.add_argument("--pass-frac", type=float, default=None)
    ap.add_argument("--min-eligible-frames", type=int, default=None)
    ap.add_argument("--pool-windows", type=int, default=8)
    ap.add_argument("--pool-frames", type=int, default=80)
    ap.add_argument("--pool-duration-s", type=float, default=8.0)
    return ap.parse_args()


def main() -> None:
    a = _args()
    # one BLAS / OpenMP thread per worker: the workers are the parallelism
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    from data_processing.noise_model.prior_bank_v4 import write_prior_bank_v4
    from experiments.noise_model.identifiability import GateConfig
    from experiments.noise_model.prior_v4 import PRIOR, RULES

    t0 = time.time()
    out = a.out or Path(f"data/rig_banks/prior_v4_n{a.n}.npz")
    flags = dict(
        threshold_db=a.threshold_db,
        min_orders=a.min_orders,
        pass_frac=a.pass_frac,
        min_eligible_frames=a.min_eligible_frames,
    )
    gate_cfg = asdict(GateConfig(**{k: v for k, v in flags.items() if v is not None}))
    pool_cfg = dict(windows=a.pool_windows, frames=a.pool_frames, duration_s=a.pool_duration_s)
    quotas = [a.n // a.workers + (1 if w < a.n % a.workers else 0) for w in range(a.workers)]
    jobs = [(a.seed, w, q, gate_cfg, pool_cfg) for w, q in enumerate(quotas) if q > 0]
    t_draw = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        parts = pool.map(_worker, jobs)
    t_draw = time.time() - t_draw
    payloads = [pl for part in parts for pl in part["kept"]]
    drawn = [d for part in parts for d in part["drawn"]]
    tried = sum(p["tried"] for p in parts)
    rejected: dict[str, int] = {}
    for p in parts:
        for key, v in p["rejected"].items():
            rejected[key] = rejected.get(key, 0) + v
    print(
        f"{len(payloads)} rigs kept of {tried} draws ({len(payloads) / tried:.1%}) in "
        f"{t_draw:.0f} s ({len(jobs)} workers)",
        flush=True,
    )
    header: dict[str, Any] = dict(
        builder="scripts/prior_bank_v4_build.py",
        git=_git_sha(),
        host=platform.node(),
        built_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        prior=PRIOR,
        seed=a.seed,
        workers=len(jobs),
        n=len(payloads),
        rules=RULES,
        gate=dict(
            config=gate_cfg,
            pool=dict(pool_cfg, stream=[a.seed, POOL_TAG]),
            eligible_frames=parts[0]["eligible"],
            tried=tried,
            kept=len(payloads),
            kept_frac=len(payloads) / tried,
            rejected_by=rejected,
        ),
        laws=_laws(drawn),
        stats=_stats(payloads),
    )
    t_write = time.time()
    path = write_prior_bank_v4(out, payloads, header)
    t_write = time.time() - t_write
    header["round_trip"] = _round_trip(path, payloads)
    header["seconds"] = dict(
        draw=round(t_draw, 1),
        write=round(t_write, 1),
        total=round(time.time() - t0, 1),
    )
    header["bank"] = dict(path=str(path), mb=round(path.stat().st_size / 1e6, 2))
    report = Path("results/prior_rigs") / f"{path.stem}.report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(header, indent=1, default=float))
    print(
        f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB) and {report} "
        f"in {header['seconds']['total']:.0f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
