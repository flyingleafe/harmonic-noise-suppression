"""Rotor-speed statistics of ACTUAL windows of a policy's noise pool, the
noise_v3_traj_stats definitions (2 s windows, 100 Hz grid), against the
legacy-hard and real reference rows in traj_stats.json."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import noise_v3_traj_stats as TS  # noqa: E402

policy, n = sys.argv[1], int(sys.argv[2])
import copy  # noqa: E402

sys.path.insert(0, "notebooks")
import noise_lab as nl  # noqa: E402

from data_processing.noise_v2_pool import NoiseV2Pool  # noqa: E402

cfg = copy.deepcopy(nl._policy_noise_source(policy))
cfg["rps"]["flight_reuse"] = 1
pool = NoiseV2Pool.from_config(cfg, duration_s=TS.DURATION_S, sample_rate=nl.SR)
rng = np.random.default_rng(20260926)
wins = []
for _ in range(n):
    entry = pool.entries[int(rng.integers(len(pool.entries)))]
    wins.append(TS._to_grid(pool.sample_rps(rng, TS.DURATION_S, entry), nl.SR))
rows = [TS.window_stats(w) for w in wins]
ref = json.loads(Path("docs/explainers/noise-model-v3-latent-runaway/traj_stats.json").read_text())[
    "summary"
]
keys = ("mean", "range", "sep_mean", "sep_max", "rate_med", "frac_low", "frac_high")
print(f"{'stat':10s} {'r4 hard (median [q1,q3])':28s} {'legacy hard':26s} {'real':26s}")
for k in keys:
    v = np.array([r[k] for r in rows])
    q = np.percentile(v, [25, 50, 75])
    lh, re = ref["legacy_hard"][k], ref["real"][k]
    print(
        f"{k:10s} {q[1]:6.2f} [{q[0]:6.2f},{q[2]:6.2f}]     "
        f"{lh['median']:6.2f} [{lh['q1']:6.2f},{lh['q3']:6.2f}]   "
        f"{re['median']:6.2f} [{re['q1']:6.2f},{re['q3']:6.2f}]"
    )

out = {
    "policy": policy,
    "n_windows": n,
    "seed": 20260926,
    "stats": {
        k: {
            "q1": float(np.percentile([r[k] for r in rows], 25)),
            "median": float(np.median([r[k] for r in rows])),
            "q3": float(np.percentile([r[k] for r in rows], 75)),
        }
        for k in keys
    },
    "legacy_hard": {k: ref["legacy_hard"][k] for k in keys},
    "real": {k: ref["real"][k] for k in keys},
}
Path("results/noise_v3r4/rig_sampler/hard_window_stats.json").write_text(json.dumps(out, indent=1))
