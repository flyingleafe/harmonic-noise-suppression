"""Build the synthetic validation sectors of the wide-prior (prior_v4) runs.

    PYTHONPATH=src python scripts/validation_sectors_v4_build.py --workers 32

Writes ``data/validation-sectors-v4/`` (``experiments.noise_model.sectors_v4``):

* ``uniform.npz``, ``weak_comb.npz``, ``strong_motor.npz``, ``wide_pedestal.npz``:
  ``noise-prior-bank-v4/2`` banks of ``--n`` rigs each. A pool of ``--axis-pool``
  draws of :func:`.prior_v4.sample_prior_v4` from the VALIDATION seed
  (``--seed``; the training bank's is ``prior_v4.SEED``) that pass the bank's
  gate on the bank's trajectory pool (``SeedSequence([--pool-seed, 7919])``,
  8 windows of 8 s, 80 frames, ``GateConfig()``). ``uniform``: ``--n`` of them
  at random; the axis sectors: the ``--n`` with the lowest v1
  ``k2_over_floor_db``, the highest v1 ``motor_boost_db`` and the highest
  :func:`.sectors_v4.pedestal_width_hz`.
* ``near_dregon.json``, ``near_md2.json``: ``noise-v2-bank/1`` banks of ``--n``
  gated :func:`.sectors_v4.perturb_fit` draws around each fit of
  ``sectors_v4.FITS``, every entry flown on its own drone's trajectory law
  (``traj_rig``). The strength is calibrated per fit: ``--calib`` ungated draws
  at each strength of ``--ladder``, the median :func:`.sectors_v4.mel_distance_db`
  to the fit interpolated (in log strength) to ``NEIGHBOURHOOD_FRAC`` of the
  fit-to-fit distance.
* ``report.json``: the calibration, the gate counts, and per sector the law
  coordinates and the mel distance of every rig to both fits.
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

POOL_TAG = 7919
#: SeedSequence tags of the neighbourhood draws: calibration, then the gated draws.
CALIB_TAG, DRAW_TAG = 1, 2
FIT_TAG = {"dregon": 11, "md2": 12}

_STATE: dict[str, Any] = {}


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _init(pool_seed: int) -> None:
    """Per process: the gate's frame pool and the fits."""
    from experiments.noise_model import drone_prior as PR
    from experiments.noise_model import sectors_v4 as S
    from experiments.noise_model.identifiability import frame_pool

    rng = np.random.default_rng(np.random.SeedSequence([pool_seed, POOL_TAG]))
    wins = PR.trajectory_windows(rng, 8, 8.0)
    _STATE["pool"] = frame_pool(wins, n_frames=80, seed=pool_seed)
    _STATE["fits"] = {name: S.fit_regimes(name) for name in S.FITS}
    _STATE["refs"] = {name: S.log_mel(_one(r)) for name, r in _STATE["fits"].items()}


def _one(regs: dict[str, Any]) -> Any:
    return regs.get("single", regs)


def _dists(lm: np.ndarray) -> dict[str, float]:
    from experiments.noise_model.sectors_v4 import mel_distance_db

    return {name: mel_distance_db(lm, ref) for name, ref in _STATE["refs"].items()}


def _prior_worker(job: tuple[int, int, int]) -> dict[str, Any]:
    """Gated prior draws of one worker until ``quota`` pass."""
    seed, worker, quota = job
    from experiments.noise_model import sectors_v4 as S
    from experiments.noise_model.identifiability import gate
    from experiments.noise_model.prior_v4 import sample_prior_v4

    kept, attempt, t0 = [], 0, time.time()
    while len(kept) < quota:
        pl = sample_prior_v4(seed, worker, attempt)
        attempt += 1
        drawn = pl.pop("_prior_v1_drawn")
        if gate(pl, _STATE["pool"]).passed:
            kept.append(dict(payload=pl, drawn=drawn, width_hz=S.pedestal_width_hz(pl)))
    print(
        f"prior worker {worker}: {quota} kept of {attempt} in {time.time() - t0:.0f} s", flush=True
    )
    return dict(kept=kept, tried=attempt)


def _calib(job: tuple[int, str, float, int]) -> tuple[str, float, float]:
    seed, name, s, i = job
    from experiments.noise_model import sectors_v4 as S

    rng = np.random.default_rng(np.random.SeedSequence([seed, CALIB_TAG, FIT_TAG[name], i]))
    q = S.perturb_fit(_STATE["fits"][name], rng, s)
    return name, s, S.mel_distance_db(S.log_mel(_one(q)), _STATE["refs"][name])


def _nbhd(job: tuple[int, str, float, int]) -> dict[str, Any]:
    seed, name, s, attempt = job
    from experiments.noise_model import sectors_v4 as S
    from experiments.noise_model.identifiability import gate

    rng = np.random.default_rng(np.random.SeedSequence([seed, DRAW_TAG, FIT_TAG[name], attempt]))
    q = S.perturb_fit(_STATE["fits"][name], rng, s)
    res = gate(_one(q), _STATE["pool"])
    return dict(
        name=name,
        attempt=attempt,
        passed=bool(res.passed),
        reason=str(res.reason),
        regimes=q if res.passed else None,
        dist=_dists(S.log_mel(_one(q))) if res.passed else None,
    )


def _sector_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def q(x: list[float]) -> dict[str, float]:
        a = np.asarray(x, dtype=np.float64)
        return {
            "min": float(a.min()),
            "p10": float(np.quantile(a, 0.1)),
            "median": float(np.median(a)),
            "p90": float(np.quantile(a, 0.9)),
            "max": float(a.max()),
        }

    out: dict[str, Any] = {"n": len(rows)}
    for key in ("k2_over_floor_db", "motor_boost_db", "slope_db_dec"):
        if all(key in r.get("drawn", {}) for r in rows):
            out[key] = q([r["drawn"][key] for r in rows])
    if all("width_hz" in r for r in rows):
        out["pedestal_width_hz"] = q([r["width_hz"] for r in rows])
    for fit in ("dregon", "md2"):
        out[f"mel_db_to_{fit}"] = q([r["dist"][fit] for r in rows])
    return out


def _args() -> argparse.Namespace:
    from experiments.noise_model.prior_v4 import SEED

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--pool-seed", type=int, default=SEED)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--axis-pool", type=int, default=1024)
    ap.add_argument("--calib", type=int, default=32)
    ap.add_argument("--ladder", type=float, nargs="+", default=[0.1, 0.2, 0.35, 0.5, 0.75, 1.0])
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    ap.add_argument("--out", type=Path, default=Path("data/validation-sectors-v4"))
    return ap.parse_args()


def main() -> None:
    a = _args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    from data_processing.noise_model.prior_bank_v4 import write_prior_bank_v4
    from data_processing.noise_v2_pool import PRESET_BANK_FORMAT
    from experiments.noise_model import sectors_v4 as S
    from experiments.noise_model.identifiability import GateConfig
    from experiments.noise_model.prior_v4 import PRIOR, RULES

    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init, initargs=(a.pool_seed,)) as pool:
        _init(a.pool_seed)  # the parent's own fits and references
        gap = S.mel_distance_db(_STATE["refs"]["dregon"], _STATE["refs"]["md2"])
        target = S.NEIGHBOURHOOD_FRAC * gap
        print(f"fit-to-fit {gap:.3f} dB; neighbourhood median target {target:.3f} dB", flush=True)

        # --- calibration: median distance per strength, per fit ---
        jobs = [(a.seed, n, s, i) for n in S.FITS for s in a.ladder for i in range(a.calib)]
        calib = pool.map(_calib, jobs, chunksize=2)
        strength: dict[str, float] = {}
        table: dict[str, dict[str, float]] = {}
        for name in S.FITS:
            med = [
                float(np.median([d for n, s, d in calib if n == name and s == st]))
                for st in a.ladder
            ]
            table[name] = {f"{st:g}": m for st, m in zip(a.ladder, med, strict=True)}
            if not med[0] <= target <= med[-1] or np.any(np.diff(med) <= 0.0):
                raise RuntimeError(
                    f"{name}: medians {med} do not bracket {target:.3f} monotonically"
                )
            strength[name] = float(np.exp(np.interp(target, med, np.log(a.ladder))))
            print(
                f"{name}: medians {np.round(med, 2).tolist()} -> strength {strength[name]:.3f}",
                flush=True,
            )
        t_calib = time.time() - t0

        # --- the prior pool (gated) and the gated neighbourhood draws, in parallel ---
        quotas = [
            a.axis_pool // a.workers + (w < a.axis_pool % a.workers) for w in range(a.workers)
        ]
        prior_async = pool.map_async(
            _prior_worker, [(a.seed, w, q) for w, q in enumerate(quotas) if q]
        )
        nbhd: dict[str, list[dict[str, Any]]] = {}
        tried: dict[str, int] = {}
        reasons: dict[str, dict[str, int]] = {}
        for name in S.FITS:
            done: list[dict[str, Any]] = []
            while sum(r["passed"] for r in done) < a.n:
                batch = max(2 * (a.n - sum(r["passed"] for r in done)), a.workers)
                jobs = [(a.seed, name, strength[name], len(done) + i) for i in range(batch)]
                done.extend(pool.map(_nbhd, jobs))
                print(
                    f"{name}: {sum(r['passed'] for r in done)}/{a.n} passed of {len(done)}",
                    flush=True,
                )
            done.sort(key=lambda r: r["attempt"])
            nbhd[name] = [r for r in done if r["passed"]][: a.n]
            # the attempts consumed: up to the last kept one, in attempt order
            tried[name] = int(nbhd[name][-1]["attempt"]) + 1
            reasons[name] = {}
            for r in done[: tried[name]]:
                if not r["passed"]:
                    key = r["reason"].split(" passed")[0]
                    reasons[name][key] = reasons[name].get(key, 0) + 1
        parts = prior_async.get()
        t_draw = time.time() - t0 - t_calib

        # --- the prior sectors: every pool rig's distance to the fits ---
        kept = [k for p in parts for k in p["kept"]]
        dist = pool.map(_dists_of_payload, [k["payload"] for k in kept], chunksize=4)
    for k, d in zip(kept, dist, strict=True):
        k["dist"] = d
    rng = np.random.default_rng(np.random.SeedSequence([a.seed, 99]))
    order = {
        "uniform": rng.permutation(len(kept))[: a.n],
        "weak_comb": np.argsort([k["drawn"]["k2_over_floor_db"] for k in kept])[: a.n],
        "strong_motor": np.argsort([-k["drawn"]["motor_boost_db"] for k in kept])[: a.n],
        "wide_pedestal": np.argsort([-k["width_hz"] for k in kept])[: a.n],
    }
    gate_cfg = dict(
        config=asdict(GateConfig()),
        pool=dict(windows=8, duration_s=8.0, frames=80, stream=[a.pool_seed, POOL_TAG]),
    )
    header_base = dict(
        builder="scripts/validation_sectors_v4_build.py",
        git=_git_sha(),
        host=platform.node(),
        built_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        seed=a.seed,
        gate=gate_cfg,
    )
    report: dict[str, Any] = dict(
        header_base,
        fit_to_fit_db=gap,
        target_db=target,
        calibration=dict(ladder=a.ladder, draws=a.calib, median_db=table, strength=strength),
        prior_pool=dict(
            kept=len(kept),
            tried=int(sum(p["tried"] for p in parts)),
            prior=PRIOR,
            rules=RULES,
            stats=_sector_stats(kept),
        ),
        sectors={},
    )
    for sector, idx in order.items():
        rows = [kept[int(i)] for i in idx]
        header = dict(header_base, sector=sector, prior=PRIOR, rules=RULES, n=len(rows))
        path = write_prior_bank_v4(a.out / f"{sector}.npz", [r["payload"] for r in rows], header)
        report["sectors"][sector] = dict(file=path.name, rule=_RULES[sector], **_sector_stats(rows))
    for name, rows in nbhd.items():
        _, traj_rig = S.FITS[name]
        entries = [
            {
                "name": f"near_{name}_{i:02d}",
                "cruise": r["regimes"].get("cruise", r["regimes"].get("single")),
                "standby": r["regimes"].get("standby"),
                "traj_rig": traj_rig,
                "provenance": {"fit": name, "attempt": r["attempt"], "strength": strength[name]},
            }
            for i, r in enumerate(rows)
        ]
        bank = {
            "format": PRESET_BANK_FORMAT,
            "entries": entries,
            "provenance": dict(
                header_base,
                sector=f"near_{name}",
                fits=S.FITS[name][0],
                strength=strength[name],
            ),
        }
        path = a.out / f"near_{name}.json"
        path.write_text(json.dumps(bank, default=float))
        stats = _sector_stats(rows)
        report["sectors"][f"near_{name}"] = dict(
            file=path.name,
            rule=f"gated sectors_v4.perturb_fit of {name} at strength {strength[name]:.3f}",
            tried=tried[name],
            kept_frac=len(rows) / tried[name],
            rejected_by=reasons[name],
            **stats,
        )
    report["seconds"] = dict(
        calibration=round(t_calib, 1), draws=round(t_draw, 1), total=round(time.time() - t0, 1)
    )
    (a.out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    print(
        json.dumps({k: v for k, v in report.items() if k != "prior_pool"}, indent=1, default=float)
    )


_RULES = {
    "uniform": "random gated prior_v4 draws",
    "weak_comb": "lowest v1 k2_over_floor_db of the gated pool",
    "strong_motor": "highest v1 motor_boost_db of the gated pool",
    "wide_pedestal": "highest sectors_v4.pedestal_width_hz of the gated pool",
}


def _dists_of_payload(pl: dict[str, Any]) -> dict[str, float]:
    from experiments.noise_model.sectors_v4 import log_mel

    return _dists(log_mel(pl))


if __name__ == "__main__":
    main()
