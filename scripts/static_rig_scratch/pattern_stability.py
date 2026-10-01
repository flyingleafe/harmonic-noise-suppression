"""Per-line spatial pattern (line level at mic c re the array mean, dB):
(a) within-run reproducibility — first half vs second half of the strict
span, read with the OU shape; (b) dependence of the between-duty scatter on
the line's SNR (floor contamination) and on parity. Usage: pattern_stability.py [k_max]."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "notebooks")
import single_rotor_lab as L  # noqa: E402
from experiments.static_rig import single_rotor as SR  # noqa: E402

ROTORS = ["Motor1", "Motor2", "Motor3", "Motor4"]
DUTIES = ["50", "60", "70", "80", "90"]
K_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 60
OU = Path("results/static_rig/single_rotor/ou")

halves, full = {}, {}
for r in ROTORS:
    good = [c for c in range(8) if c not in L.WIND[r]]
    for d in DUTIES:
        key = f"motor_{r}_{d}"
        s, _ = L.TABLE[key]
        x, fs, model = L._analysed(key)
        a, b = model.span
        p = json.loads((OU / f"{key}.json").read_text())
        mid = (a + b) // 2
        lv = []
        for lo, hi in ((a, mid), (mid, b)):
            fit = SR.line_powers_ou(
                x[:, lo:hi].astype(np.float64), fs, s, p["sigma_nu"], p["lam"], k_max=K_MAX
            )
            lv.append(
                10 * np.log10(np.maximum(fit.amp2, 1e-30) / fit.amp2[good].mean(0, keepdims=True))
            )
        halves[(r, d)] = (lv[0][good], lv[1][good])
        amp = np.array(p["amp2"])[:, :K_MAX]
        full[(r, d)] = (
            10 * np.log10(np.maximum(amp, 1e-30) / amp[good].mean(0, keepdims=True))[good]
        )
        # SNR per line (array mean line power over the local floor), from the fit's periodogram floor
        print(
            f"{key}: half-vs-half rms diff {np.sqrt(np.mean((lv[0][good] - lv[1][good]) ** 2)):.2f} dB",
            flush=True,
        )

h1 = np.concatenate([v[0].ravel() for v in halves.values()])
h2 = np.concatenate([v[1].ravel() for v in halves.values()])
print(
    f"\nwithin-run (half vs half): rms diff {np.sqrt(np.mean((h1 - h2) ** 2)):.2f} dB, corr {np.corrcoef(h1, h2)[0, 1]:+.2f}  (pattern sd {h1.std():.2f})"
)
# between-run at the same order (duties d, d'), for reference
b1, b2 = [], []
for r in ROTORS:
    for i, d in enumerate(DUTIES):
        for d2 in DUTIES[i + 1 :]:
            b1.append(full[(r, d)].ravel())
            b2.append(full[(r, d2)].ravel())
b1, b2 = np.concatenate(b1), np.concatenate(b2)
print(
    f"between runs (same order, other duty): rms diff {np.sqrt(np.mean((b1 - b2) ** 2)):.2f} dB, corr {np.corrcoef(b1, b2)[0, 1]:+.2f}"
)
# half-vs-half by parity and by order band
k = np.arange(1, K_MAX + 1)
for name, sel in (
    ("even", k % 2 == 0),
    ("odd", k % 2 == 1),
    ("k≤15", k <= 15),
    ("16–30", (k > 15) & (k <= 30)),
    ("31–60", k > 30),
):
    d = np.concatenate([(v[0] - v[1])[:, sel].ravel() for v in halves.values()])
    print(f"    half-vs-half rms diff, {name}: {np.sqrt(np.mean(d**2)):.2f} dB")
