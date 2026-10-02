"""How likely are the real fits (and the prior's own draws) under the prior's laws?

    PYTHONPATH=src .venv/bin/python scripts/static_rig_scratch/prior_law_scores.py

Prints, per payload, every law's measured value and its standard score
(:func:`experiments.noise_model.drone_prior.law_scores`), plus the comb's
Mahalanobis distance from the prior's centre (the seven comb laws) and the
chi-square tail probability of that distance.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from experiments.noise_model.drone_prior import PriorSpec, law_scores

COMB = (
    "k2_over_floor_db",
    "slope_db_dec",
    "tail_over_floor_db",
    "bpf_boost_db",
    "odd_a_db",
    "odd_b_db",
    "motor_boost_db",
)
FITS = {
    "michaels cruise": "results/noise_v3/fits_r4/michaels_fly125_cruise__flight_v3.json",
    "michaels standby": "results/noise_v3/fits_r4/michaels_fly125_standby__flight_v3.json",
    "dregon free flight": "results/noise_v3/fits_r4/dregon_room2_floor__flight_v3.json",
}


def main() -> None:
    spec = PriorSpec()
    payloads = {name: json.loads(Path(p).read_text()) for name, p in FITS.items()}
    for p in sorted(glob.glob("results/prior_rigs/prior_v1_seed0/rig_*.json"))[:3]:
        payloads[f"prior {Path(p).stem}"] = json.loads(Path(p).read_text())
    names = list(payloads)
    scores = {name: law_scores(payloads[name], spec) for name in names}
    laws = [row[0] for row in scores[names[0]]]
    head = f"{'law':20s}" + "".join(f"{n[:22]:>24s}" for n in names)
    print(head)
    for i, law in enumerate(laws):
        line = f"{law:20s}"
        for name in names:
            _, v, z = scores[name][i]
            line += f"{v:12.2f} (z{z:+5.1f})   "
        print(line)
    print()
    for name in names:
        z = np.array([z for law, _, z in scores[name] if law in COMB])
        d2 = float(np.sum(z**2))
        print(
            f"{name:22s}: comb Mahalanobis d^2 = {d2:6.1f} over {z.size} laws, "
            f"P(chi2_{z.size} >= d^2) = {chi2.sf(d2, z.size):.3g}"
        )


if __name__ == "__main__":
    main()
