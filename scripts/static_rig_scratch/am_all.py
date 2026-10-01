import sys
import json
import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "notebooks")
import single_rotor_lab as L
from experiments.static_rig import single_rotor as SR
from pathlib import Path

out = {}
print(
    "key            T    orders>10dB  sigma_m2 med (k<=8 | 9-30 | >30)   gamma_m med Hz (k<=8 | 9-30 | >30)   coherence   Whittle sigma_m2/gamma_m"
)
for key in L.RECORDINGS:
    x, fs = L.load(key)
    s_bar, r1 = L.TABLE[key]
    wind = L.WIND[key.split("_")[1]]
    good = [c for c in range(x.shape[0]) if c not in wind]
    a, b = SR.strict_span(x[good], fs, s_bar)
    xs = np.asarray(x[:, a:b], dtype=np.float64)
    aj = SR.amplitude_jitter(xs, fs, s_bar)
    k = aj.orders

    def med(arr, lo, hi):
        v = arr[good][:, (k >= lo) & (k <= hi)]
        return float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")

    wf = L.load_whittle(key)
    n_ok = int(np.isfinite(aj.sigma_m2[good]).sum(axis=1).mean())
    print(
        f"{key:14s} {xs.shape[1] / fs:4.1f}  {n_ok:3d}       {med(aj.sigma_m2, 1, 8):.3f} | {med(aj.sigma_m2, 9, 30):.3f} | {med(aj.sigma_m2, 31, 150):.3f}        "
        f"{med(aj.gamma_m, 1, 8):.3f} | {med(aj.gamma_m, 9, 30):.3f} | {med(aj.gamma_m, 31, 150):.3f}       {np.nanmedian(aj.coherence[good]):.2f}       "
        + (f"{wf.sigma_m2:.3f}/{wf.gamma_m:.3f}" if wf else "-")
    )
    out[key] = {
        "orders": k.tolist(),
        "sigma_m2": np.where(np.isfinite(aj.sigma_m2), aj.sigma_m2, None).tolist(),
        "gamma_m": np.where(np.isfinite(aj.gamma_m), aj.gamma_m, None).tolist(),
        "snr_db": aj.snr_db.tolist(),
        "coherence": np.where(np.isfinite(aj.coherence), aj.coherence, None).tolist(),
        "frame_s": aj.frame_s,
        "mics_good": good,
    }
Path("results/static_rig/single_rotor").mkdir(parents=True, exist_ok=True)
json.dump(out, open("results/static_rig/single_rotor/amplitude_jitter.json", "w"))
