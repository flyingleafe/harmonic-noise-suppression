"""Build the EASY training bank of the priorv4 grid: drones drawn around the two fits.

    PYTHONPATH=src python scripts/easy_bank_v4_build.py --n-per-fit 1024

Decided by Dmitrii, 2026-10-26 (``docs/experiments/rp-synthetic.md`` § "Easy
setup"): every draw is ONE prior-shaped perturbation
(``experiments.noise_model.sectors_v4.perturb_fit``) of a base fit, at the
strength the validation neighbourhoods were calibrated to (median spectral
distance to the fit one third of the DREGON-MD2 gap; ``validation-sectors-v4``
report), UNGATED (the fits are real drones; the gate is for prior draws), and
flown on its base's own trajectory law (``traj_rig``):

* DREGON: the accepted DREGON flight payload, the converged bench fit (c7) with
  its fitted SC gust wind (``results/noise_v4/flight/benchwind/``); the
  perturbation keeps the wind;
* MD2: the standby/cruise pair of :data:`experiments.noise_model.sectors_v4.FITS`.

Draw ``i`` of a fit uses ``SeedSequence([seed, tag, i])``, so any draw can be
regenerated; the builder regenerates a few and checks the bank gives them back
exactly. Writes a ``noise-nbhd-bank-v4/1`` bank
(:mod:`data_processing.noise_model.nbhd_bank_v4`), equal counts per fit, so the
pool's uniform entry draw is a 50/50 mix.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_processing.noise_model.nbhd_bank_v4 import NbhdBankV4, write_nbhd_bank_v4  # noqa: E402
from experiments.noise_model.sectors_v4 import FITS, perturb_fit  # noqa: E402

SEED = 20261027
#: name -> (regime paths, trajectory rig, strength, seed tag). Strengths: the
#: validation-sectors-v4 calibration (report.json, near_dregon / near_md2 rules).
BASES: dict[str, tuple[dict[str, str], str, float, int]] = {
    "dregon": (
        {"single": "results/noise_v4/flight/benchwind/dregon__c7__benchwind_gust.json"},
        "dregon",
        0.375,
        21,
    ),
    "md2": (dict(FITS["md2"][0]), FITS["md2"][1], 0.374, 22),
}


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _regimes(paths: dict[str, str]) -> dict[str, dict[str, Any]]:
    return {k: json.loads((ROOT / p).read_text()) for k, p in paths.items()}


def _draw(name: str, regs: dict[str, Any], i: int, seed: int) -> dict[str, dict[str, Any]]:
    _, _, strength, tag = BASES[name]
    rng = np.random.default_rng(np.random.SeedSequence([seed, tag, i]))
    return perturb_fit(regs, rng, strength, keep_wind=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n-per-fit", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=None, help="default data/easy-v4-banks/easy_v4_n<N>.npz")
    a = ap.parse_args()
    n = int(a.n_per_fit)
    out = Path(a.out or f"data/easy-v4-banks/easy_v4_n{2 * n}.npz")
    t0 = time.time()
    regs = {name: _regimes(paths) for name, (paths, _, _, _) in BASES.items()}
    bases = [
        {"name": name, "traj_rig": traj, "regimes": regs[name]}
        for name, (_, traj, _, _) in BASES.items()
    ]

    def draws():
        for b, name in enumerate(BASES):
            for i in range(n):
                yield {
                    "base": b,
                    "name": f"easy_{name}_{i:04d}",
                    "regimes": _draw(name, regs[name], i, a.seed),
                }

    header = {
        "builder": "scripts/easy_bank_v4_build.py",
        "git": _git_sha(),
        "seed": a.seed,
        "n_per_fit": n,
        "rule": "sectors_v4.perturb_fit, keep_wind, ungated; one draw per entry (Dmitrii, 2026-10-26)",
        "bases": {
            name: {"regimes": paths, "traj_rig": traj, "strength": s, "seed_tag": tag}
            for name, (paths, traj, s, tag) in BASES.items()
        },
    }
    write_nbhd_bank_v4(out, bases, draws(), header)
    t_write = time.time() - t0
    bank = NbhdBankV4(out)
    rng = np.random.default_rng(0)
    for j in rng.choice(len(bank), 8, replace=False):
        name = "dregon" if int(j) < n else "md2"
        want = _draw(name, regs[name], int(j) % n, a.seed)
        got = bank.regimes(int(j))
        for regime, payload in want.items():
            if json.dumps(got[regime], sort_keys=True) != json.dumps(payload, sort_keys=True):
                raise RuntimeError(f"draw {j} ({regime}) does not read back as regenerated")
    print(
        f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB): {len(bank)} draws "
        f"({n} per fit) in {t_write:.0f} s; 8 regenerated draws read back exactly",
        flush=True,
    )


if __name__ == "__main__":
    main()
