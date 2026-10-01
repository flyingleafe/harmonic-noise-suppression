import sys

import numpy as np

key = sys.argv[1]
CAL = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
z = np.load(f"/tmp/avq_refine_{key}.npz")
ks = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30, 36, 42, 48, 56, 63, 64, 84]
print(
    f"== {key}: line/noise (dB, noise = floor power in the 3 Hz ENBW) of the mic-median per 0.5 s block; p10/p50/p90 over blocks; '--' = order collides with another rotor (within 3 Hz) in >50% blocks"
)
print("   k    " + "".join(f"{k:>14d}" for k in ks))
for f in "ABCD":
    pw, fl = z[f"{f}_pw"], z[f"{f}_fl"]
    snr = 10 * np.log10(np.maximum(pw / fl / CAL - 1, 1e-3))
    # merged flag not stored; recompute from the other rotors' refined tracks
    s1 = z[f"{f}_s1"]
    row = []
    for k in ks:
        fk = k * s1
        coll = np.zeros(len(s1), bool)
        for g in "ABCD":
            if g == f:
                continue
            so = z[f"{g}_s1"]
            coll |= np.abs(fk - np.round(fk / so) * so) < 3.0
        v = np.median(snr[:, k - 1, :], axis=1)[~coll]
        if v.size < len(s1) / 2:
            row.append("            --")
        else:
            row.append(
                f" {np.percentile(v, 10):4.0f}/{np.percentile(v, 50):3.0f}/{np.percentile(v, 90):3.0f}"
            )
    print(f"   {f}    " + "".join(row))
