"""Per (rotor, mic) relative response R_rm(f) = P_rm(f) / mean_m' P_rm'(f)
(dB, Welch spectra of the strict span, non-windy mics as the reference),
at the five duties. Hypothesis (1), a fixed transfer function: the five
curves coincide on the FREQUENCY axis. Hypothesis (2), a per-harmonic
pattern: they coincide on the ORDER axis (f / s̄).

Prints, per rotor and mic, the mean correlation between duties of R on each
axis, and writes results/static_rig/single_rotor/figs/mic_eq_<rotor>.png."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import welch  # noqa: E402

sys.path.insert(0, "notebooks")
import single_rotor_lab as L  # noqa: E402

ROTORS = ["Motor1", "Motor2", "Motor3", "Motor4"]
DUTIES = ["50", "60", "70", "80", "90"]
F_LO, F_HI = 300.0, 6000.0
SMOOTH_HZ = 20.0
ORD_LO, ORD_HI = 5.0, 70.0  # order axis range (covers 300–6000 Hz at all duties)
OUT = Path("results/static_rig/single_rotor/figs")
OUT.mkdir(parents=True, exist_ok=True)

f_grid = np.arange(F_LO, F_HI, 5.0)
o_grid = np.arange(ORD_LO, ORD_HI, 0.1)
summary = {}
for r in ROTORS:
    windy = set(L.WIND[r])
    good = [c for c in range(8) if c not in windy]
    Rf, Ro = {}, {}
    for d in DUTIES:
        key = f"motor_{r}_{d}"
        s, _ = L.TABLE[key]
        x, fs, model = L._analysed(key)
        a, b = model.span
        f, P = welch(x[:, a:b].astype(np.float64), fs, nperseg=8192, axis=1)  # ~5.4 Hz bins
        w = int(round(SMOOTH_HZ / (f[1] - f[0])))
        Ps = np.stack([np.convolve(p, np.ones(w) / w, "same") for p in P])
        ref = Ps[good].mean(axis=0)
        R = 10 * np.log10(Ps / ref)  # (8, F) dB relative to the array mean
        Rf[d] = np.stack([np.interp(f_grid, f, R[c]) for c in range(8)])
        Ro[d] = np.stack([np.interp(o_grid * s, f, R[c]) for c in range(8)])
    # pairwise correlations between duties, per mic, on each axis
    rows = []
    fig, axs = plt.subplots(2, len(good), figsize=(4.2 * len(good), 7), sharey="row")
    for i, c in enumerate(good):
        cf, co = [], []
        for ia, da in enumerate(DUTIES):
            for db in DUTIES[ia + 1 :]:
                cf.append(np.corrcoef(Rf[da][c], Rf[db][c])[0, 1])
                co.append(np.corrcoef(Ro[da][c], Ro[db][c])[0, 1])
        sd = np.mean([Rf[d][c].std() for d in DUTIES])
        rows.append((c, float(np.mean(cf)), float(np.mean(co)), float(sd)))
        for d in DUTIES:
            axs[0, i].plot(f_grid, Rf[d][c], lw=0.8, label=d)
            axs[1, i].plot(o_grid, Ro[d][c], lw=0.8, label=d)
        axs[0, i].set_title(f"{r} ch{c}: f axis, corr {np.mean(cf):+.2f}")
        axs[1, i].set_title(f"order axis, corr {np.mean(co):+.2f}")
        axs[0, i].set_xlabel("Hz")
        axs[1, i].set_xlabel("order f/s̄")
    axs[0, 0].set_ylabel("R (dB re array mean)")
    axs[1, 0].set_ylabel("R (dB re array mean)")
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / f"mic_eq_{r}.png", dpi=70)
    plt.close(fig)
    summary[r] = rows
    print(f"{r} (ref = mics {good}); mean correlation of R between duties:")
    for c, cf_, co_, sd in rows:
        print(
            f"    ch{c}: frequency axis {cf_:+.2f}   order axis {co_:+.2f}   (R sd over f: {sd:.1f} dB)"
        )
    print(
        f"    rotor mean: frequency {np.mean([t[1] for t in rows]):+.2f}, order {np.mean([t[2] for t in rows]):+.2f}"
    )
