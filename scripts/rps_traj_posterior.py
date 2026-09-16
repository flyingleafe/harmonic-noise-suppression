#!/usr/bin/env python3
"""Fit the rig posterior to the fitted rigs and draw sample drones from it.

Reads every ``fits/new/<rig>.json`` under ``--fits``, fits the diagonal
Gaussian over scale-free rig coordinates
(:func:`experiments.rps_traj.posterior.fit_posterior`), writes it to ``--out``
and writes ``--draws`` parameter draws next to it as ``draws/<k>.json``.

Every draw is checked before it is written: 60 s sampled at 100 Hz must be
finite and strictly positive, i.e. the drawn drone is a drone whose rotors
never stop.  A draw that fails is REJECTED and redrawn (up to
:data:`MAX_TRIES` times per slot) so that the written draws are all usable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if (_HERE.parent / "src").is_dir() else Path.cwd().resolve()
sys.path.insert(0, str(_ROOT / "src"))

from data_processing.trajectory_model import NewFit, Params, rig_vector  # noqa: E402
from experiments.rps_traj.data import RATE_HZ  # noqa: E402
from experiments.rps_traj.posterior import fit_posterior, informative_mask  # noqa: E402

#: Check length (s) and the per-slot redraw budget.
CHECK_S = 60.0
MAX_TRIES = 20


def check_draw(params: Params, seed: int) -> tuple[bool, dict[str, float]]:
    """Sample :data:`CHECK_S` seconds and report whether it is a usable drone."""
    rng = np.random.default_rng(seed)
    w = params.sample_airborne(int(round(CHECK_S * RATE_HZ)), rng)
    finite = bool(np.isfinite(w).all())
    lo = float(np.min(w)) if finite else float("nan")
    return finite and lo > 0.0, {
        "min_rps": lo,
        "max_rps": float(np.max(w)) if finite else float("nan"),
        "mean_rps": float(np.mean(w)) if finite else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--fits", type=Path, default=Path("results/rps_traj/fits/new"))
    ap.add_argument("--out", type=Path, default=Path("results/rps_traj/posterior.json"))
    ap.add_argument("--draws", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(p for p in Path(args.fits).glob("*.json"))
    if not paths:
        raise SystemExit(f"no fits under {args.fits}")
    fits = {p.stem: NewFit.from_json(p.read_text()) for p in paths}
    post = fit_posterior(fits)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(post.to_json())
    print(f"posterior over {len(fits)} rigs: {', '.join(post.rigs)}")
    print(f"  mean {np.array2string(post.mean, precision=2, floatmode='fixed')}")
    print(f"  std  {np.array2string(post.std, precision=2, floatmode='fixed')}")
    for rig, fit in fits.items():
        v = rig_vector(fit.params)
        # Only over the coordinates the posterior was fitted on: a rig with no
        # between-flight level difference stores a placeholder in its offset
        # coordinates, and that placeholder is not an outlier.
        keep = informative_mask(fit.params)
        z = (v[keep] - post.mean[keep]) / post.std[keep]
        print(f"  {rig:<18} |z|max {np.max(np.abs(z)):.2f}  scale {np.exp(v[0]):7.1f} rev/s")

    draws_dir = out.parent / "draws"
    draws_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    records: list[dict[str, Any]] = []
    for k in range(int(args.draws)):
        redraws = 0
        while True:
            params = post.sample(rng)
            ok, info = check_draw(params, seed=args.seed + k)
            if ok:
                break
            redraws += 1
            if redraws >= MAX_TRIES:
                raise SystemExit(
                    f"draw {k} failed {MAX_TRIES} times — the posterior samples "
                    "drones whose rotors stop; last sample: " + json.dumps(info)
                )
        (draws_dir / f"{k}.json").write_text(params.to_json())
        records.append({"draw": k, "redraws": redraws, **info})
        print(
            f"  draw {k:2d}: mean {info['mean_rps']:7.1f}  "
            f"min {info['min_rps']:7.1f}  max {info['max_rps']:7.1f} rev/s"
            + (f"  ({redraws} redraws)" if redraws else "")
        )

    (out.parent / "draws.json").write_text(
        json.dumps({"check_s": CHECK_S, "draws": records}, indent=2, allow_nan=False)
    )
    print(f"wrote {out} and {len(records)} draws under {draws_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
