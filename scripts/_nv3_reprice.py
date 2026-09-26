"""Throwaway: re-price noise-v3 fits' total objective under the MEASURED wander.

Totals of fits made under different wander priors (r1 schema-2, r2 mm1, r3b
with the u_j kernel) do not rank. This keeps each fit's recorded Whittle term
and rig -log prior and recomputes the OU -log prior of its fitted latents under
`results/noise_v3/wander/<rig>.json` (independent u_j tracks), the prior of
round 3a. Light JSON + torch on CPU; safe on the laptop.

Usage: `_nv3_reprice.py [--round TAG=DIR ...] [--out JSON]`; the default
rounds are r2 and r1, the default out `results/noise_v3/diag/reprice_measured.json`.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data_processing.noise_model.v3 import Wander
from experiments.noise_model import model as MD

POOLS = (
    ("dregon_room2_floor", "dregon"),
    ("michaels_fly125_cruise", "michaels"),
    ("michaels_fly125_standby", "michaels"),
)


def t(a):
    return torch.as_tensor(np.asarray(a, dtype=np.float64))


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument(
        "--round",
        action="append",
        default=None,
        help="TAG=DIR of <pool>__flight_v3.json fits (repeatable)",
    )
    ap.add_argument("--out", default="results/noise_v3/diag/reprice_measured.json")
    args = ap.parse_args()
    rounds = [
        r.split("=", 1)
        for r in (args.round or ["r2=results/noise_v3/fits_r2", "r1=results/noise_v3/fits"])
    ]
    out = {}
    for pool, rig in POOLS:
        meas = Wander.from_mapping(load(f"results/noise_v3/wander/{rig}.json"))
        for rnd, d in rounds:
            fit = load(f"{d}/{pool}__flight_v3.json")
            lat = {
                int(w["window"]): MD.WindowLatents(
                    d=t(w["d"]), v=t(w["v"]), u=t(w["u"]), uj=t(w["uj"])
                )
                for w in fit["latents"]["windows"]
            }
            own = Wander.from_mapping(fit["params"]["wander"])
            o = fit["objective"]
            ou_m, ou_own = MD.ou_prior_nats(lat, meas), MD.ou_prior_nats(lat, own)
            out[f"{rnd}/{pool}"] = dict(
                whittle=o["whittle_nats"],
                rig=o["rig_neg_log_prior_nats"],
                ou_own=ou_own,
                ou_recorded=o["ou_neg_log_prior_nats"],
                ou_measured=ou_m,
                total_own=o["total_nats"],
                total_measured=o["whittle_nats"] + o["rig_neg_log_prior_nats"] + ou_m,
                n_cells=o["n_cells"],
            )
            print(rnd, pool, {k: round(v, 1) for k, v in out[f"{rnd}/{pool}"].items()})
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
