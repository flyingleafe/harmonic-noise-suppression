"""Signature A on the lines: per (rotor, mic, duty, order) the line level
relative to the array mean, L = 10 log10(amp2[c,k] / mean_good amp2[:,k])
(source cancels). Across two duties, correlate L for pairs of orders whose
frequencies COINCIDE (|k s_a − k' s_b| < tol) against pairs that are close
but not coincident (10–40 Hz apart) and against same-order pairs."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "notebooks")
import single_rotor_lab as L  # noqa: E402

ROTORS = ["Motor1", "Motor2", "Motor3", "Motor4"]
DUTIES = ["50", "60", "70", "80", "90"]
K_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 60
TOL = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
OU = Path("results/static_rig/single_rotor/ou")

tot = {"coincident": [], "near": [], "same order": []}
for r in ROTORS:
    good = [c for c in range(8) if c not in L.WIND[r]]
    Lv, S = {}, {}
    for d in DUTIES:
        key = f"motor_{r}_{d}"
        a = np.array(json.loads((OU / f"{key}.json").read_text())["amp2"])[:, :K_MAX]
        Lv[d] = 10 * np.log10(np.maximum(a, 1e-30) / a[good].mean(axis=0, keepdims=True))
        S[d] = L.TABLE[key][0]
    res = {"coincident": [], "near": [], "same order": []}
    for ia, da in enumerate(DUTIES):
        for db in DUTIES[ia + 1 :]:
            ka, kb = np.arange(1, K_MAX + 1)[:, None], np.arange(1, K_MAX + 1)[None, :]
            dfreq = np.abs(ka * S[da] - kb * S[db])
            for name, sel in (
                ("coincident", (dfreq < TOL) & (ka != kb)),
                ("near", (dfreq > 10) & (dfreq < 40) & (ka != kb)),
                ("same order", ka == kb),
            ):
                ii, jj = np.nonzero(sel)
                for c in good:
                    res[name].append(np.column_stack([Lv[da][c, ii], Lv[db][c, jj]]))
    line = f"{r}:"
    for name, v in res.items():
        v = np.concatenate(v)
        rho = np.corrcoef(v[:, 0], v[:, 1])[0, 1]
        rms = np.sqrt(np.mean((v[:, 0] - v[:, 1]) ** 2))
        line += f"  {name}: corr {rho:+.2f}, rms diff {rms:.2f} dB (n={v.shape[0]})"
        tot[name].append(v)
    print(line)
print(
    "all rotors:",
    "  ".join(
        f"{n}: corr {np.corrcoef(np.concatenate(v)[:, 0], np.concatenate(v)[:, 1])[0, 1]:+.2f}"
        for n, v in tot.items()
    ),
)
print(f"(line level spread itself: sd {np.concatenate(tot['same order'])[:, 0].std():.2f} dB)")
