import sys
import time
import json
import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "notebooks")
import single_rotor_lab as L
from experiments.static_rig import single_rotor as SR
from pathlib import Path

out_dir = Path("results/static_rig/single_rotor/whittle")
out_dir.mkdir(parents=True, exist_ok=True)
for key in L.RECORDINGS:
    if (out_dir / f"{key}.json").exists():
        continue
    x, fs = L.load(key)
    s_bar, r1 = L.TABLE[key]
    wind = L.WIND[key.split("_")[1]]
    good = [c for c in range(x.shape[0]) if c not in wind]
    a, b = SR.strict_span(x[good], fs, s_bar)
    xs = np.asarray(x[:, a:b], dtype=np.float64)
    t = time.time()
    wf = SR.whittle_fit(xs, fs, s_bar, mics=good)
    dt = time.time() - t
    rec = {
        "key": key,
        "s": wf.s,
        "D": wf.D,
        "gamma_m": wf.gamma_m,
        "sigma_m2": wf.sigma_m2,
        "loglik": wf.loglik,
        "T": wf.T,
        "mics": wf.mics,
        "span": [a, b],
        "orders": wf.orders.tolist(),
        "amp2": wf.amp2.tolist(),
        "grid": wf.grid,
        "seconds": dt,
        "D_phase": r1**2,
    }
    json.dump(rec, open(out_dir / f"{key}.json", "w"))
    g = wf.grid
    i = int(np.argmax(g["loglik_D"]))
    print(
        f"{key}: {dt:.0f}s D {wf.D:.2e} (phase {r1**2:.2e}) gamma_m {wf.gamma_m:.3f} sigma_m2 {wf.sigma_m2:.3f} | D profile 2-nat range: "
        + ",".join(
            f"{d:.0e}" for d, v in zip(g["D"], g["loglik_D"]) if v > wf.loglik - 2 * 1.5 * 100
        ),
        flush=True,
    )
