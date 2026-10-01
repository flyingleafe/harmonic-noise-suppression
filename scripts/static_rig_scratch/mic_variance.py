"""How much of the mic-to-mic deviation of a rotor's line levels is a fixed
per-(rotor, mic) EQ, how much a per-mic (rotor-independent) sensitivity, and
what remains.

y[r, d, c, k] = line level at mic c re the array mean (dB), from the OU
profiles. Held-out-duty evaluation: the EQ of a pair is a Gaussian-kernel
smooth (width W Hz) of the other four duties' points on the frequency axis;
residual = y − EQ(f). Per-mic sensitivity: the same, pooled over the other
rotors as well (rotor-independent). Usage: mic_variance.py [k_max]."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "notebooks")
import single_rotor_lab as L  # noqa: E402

ROTORS = ["Motor1", "Motor2", "Motor3", "Motor4"]
DUTIES = ["50", "60", "70", "80", "90"]
K_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 60
OU = Path("results/static_rig/single_rotor/ou")
WIDTHS = [50, 100, 200, 400, 800]

pts = {}  # (r, d, c) -> (f, y)
for r in ROTORS:
    good = [c for c in range(8) if c not in L.WIND[r]]
    for d in DUTIES:
        key = f"motor_{r}_{d}"
        a = np.array(json.loads((OU / f"{key}.json").read_text())["amp2"])[:, :K_MAX]
        s = L.TABLE[key][0]
        y = 10 * np.log10(np.maximum(a, 1e-30) / a[good].mean(axis=0, keepdims=True))
        f = np.arange(1, K_MAX + 1) * s
        for c in good:
            pts[(r, d, c)] = (f, y[c])


def smooth_eval(train, f_eval, W):
    f_tr = np.concatenate([t[0] for t in train])
    y_tr = np.concatenate([t[1] for t in train])
    w = np.exp(-0.5 * ((f_eval[:, None] - f_tr[None, :]) / W) ** 2)
    return (w * y_tr[None, :]).sum(1) / np.maximum(w.sum(1), 1e-9)


total = np.concatenate([v[1] for v in pts.values()])
print(
    f"orders ≤ {K_MAX}: mic-to-mic deviation of line levels, sd {total.std():.2f} dB  ({len(pts)} (rotor, duty, mic) curves)"
)
print(
    f"{'EQ width W':>12s} | {'pair RTF (r,c)':>16s} | {'mic only (c)':>14s} | {'pair, same-duty fit (in-sample)':>32s}"
)
for W in WIDTHS:
    res_pair, res_mic, res_in = [], [], []
    for (r, d, c), (f, y) in pts.items():
        train_pair = [pts[(r, d2, c)] for d2 in DUTIES if d2 != d]
        train_mic = [
            pts[(r2, d2, c)] for r2 in ROTORS for d2 in DUTIES if d2 != d and (r2, d2, c) in pts
        ]
        res_pair.append(y - smooth_eval(train_pair, f, W))
        res_mic.append(y - smooth_eval(train_mic, f, W))
        res_in.append(y - smooth_eval([(f, y)], f, W))
    rp, rm, ri = (np.concatenate(v) for v in (res_pair, res_mic, res_in))
    print(
        f"{W:>9d} Hz | sd {rp.std():.2f} dB ({1 - rp.var() / total.var():4.0%} explained) | "
        f"sd {rm.std():.2f} ({1 - rm.var() / total.var():4.0%}) | sd {ri.std():.2f} ({1 - ri.var() / total.var():4.0%})"
    )
# how much of the pair EQ is the mic-only part: compare the two held-out EQ curves
W = 200
d_eq = []
for (r, d, c), (f, y) in pts.items():
    eq_pair = smooth_eval([pts[(r, d2, c)] for d2 in DUTIES if d2 != d], f, W)
    eq_mic = smooth_eval(
        [pts[(r2, d2, c)] for r2 in ROTORS for d2 in DUTIES if d2 != d and (r2, d2, c) in pts], f, W
    )
    d_eq.append((eq_pair, eq_mic))
ep, em = (np.concatenate(v) for v in zip(*d_eq, strict=True))
print(
    f"W = {W} Hz: pair EQ sd {ep.std():.2f} dB, mic-only EQ sd {em.std():.2f} dB, their difference sd {(ep - em).std():.2f} dB, corr {np.corrcoef(ep, em)[0, 1]:+.2f}"
)
