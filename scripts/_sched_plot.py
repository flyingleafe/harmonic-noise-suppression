"""Schedule figure for docs/experiments/paper-real-protocol.md: LR warm-up + cosine and the per-augmentation
severity ramps over the 100k-step budget, with the warm-up end, min_rounds and ramps-done marks.
Writes /tmp/sched.png. Run: python scripts/_sched_plot.py"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = 100_000
s = np.arange(0, S + 1, 100)
f = s / S


def cos_ramp(f, a, b, v0, v1):  # cosine from v0 at a to v1 at b (fractions)
    t = np.clip((f - a) / (b - a), 0, 1)
    return v0 + (v1 - v0) * (1 - np.cos(np.pi * t)) / 2


lr = np.where(
    f < 0.05, 1e-3 * f / 0.05, 1e-5 + (1e-3 - 1e-5) * (1 + np.cos(np.pi * (f - 0.05) / 0.95)) / 2
)
marks = {"warm-up end 5 %": 0.05, "min_rounds 40×500": 0.20, "ramps done 35 %": 0.35}
fig, ax = plt.subplots(5, 1, figsize=(9, 10.5), sharex=True)
ax[0].plot(s, lr, color="k")
ax[0].set_yscale("log")
ax[0].set_ylabel("LR")
ax[0].set_title("Learning rate: linear warm-up 0→1e-3 (5 %), cosine → 1e-5 at 100k", fontsize=10)
# speech SNR range
hi = cos_ramp(f, 0.10, 0.35, -30, 5)
lo = np.full_like(f, -30.0)
ax[1].fill_between(s, lo, hi, color="tab:red", alpha=0.25)
ax[1].plot(s, hi, color="tab:red")
ax[1].plot(s, lo, color="tab:red", ls="--")
ax[1].set_ylabel("speech SNR (dB)")
ax[1].set_title(
    "Speech mixing, p = 1: SNR ~ U(−30, upper); upper bound −30 → +5 dB over 10–35 % (strongly destroying)",
    fontsize=10,
)
ax[1].set_ylim(-35, 10)
# reverb DRR
dlo = cos_ramp(f, 0.05, 0.20, 15, 3)
dhi = np.full_like(f, 15.0)
ax[2].fill_between(s, dlo, dhi, color="tab:purple", alpha=0.25)
ax[2].plot(s, dlo, color="tab:purple")
ax[2].plot(s, dhi, color="tab:purple", ls="--")
ax[2].set_ylabel("reverb DRR (dB)")
ax[2].set_title(
    "random_reverb: DRR ~ U(lower, 15); lower bound 15 → 3 dB over 5–20 % (mildly destroying); RT60 U(0.1, 0.8) s throughout",
    fontsize=10,
)
ax[2].set_ylim(0, 17)
# info-preserving: gain, EQ
ax[3].fill_between(s, -6, 6, color="tab:green", alpha=0.25, label="random_gain U(−6, +6) dB, p 0.5")
ax[3].fill_between(s, -8, 8, color="tab:olive", alpha=0.15, label="spectral_recolor ±8 dB anchors")
ax[3].set_ylabel("dB")
ax[3].legend(loc="lower right", fontsize=8, frameon=False)
ax[3].set_title("Info-preserving: full strength from step 0", fontsize=10)
ax[3].set_ylim(-10, 10)
# rate warp
ax[4].fill_between(
    s,
    0.5,
    1.5,
    color="tab:blue",
    alpha=0.25,
    label="rate_warp α₀ ~ U(0.5, 1.5) (+ ±12 % wander), labelaug arms only",
)
ax[4].plot(s, np.ones_like(f), color="tab:blue", ls=":")
ax[4].set_ylabel("α")
ax[4].legend(loc="lower right", fontsize=8, frameon=False)
ax[4].set_title("Label-altering: full strength from step 0 (labels transformed)", fontsize=10)
ax[4].set_ylim(0.3, 1.7)
for a in ax:
    for x in marks.values():
        a.axvline(x * S, color="0.4", lw=0.8, ls="--")
for lab, x in marks.items():
    ax[0].text(x * S, 2e-3, lab, rotation=90, fontsize=8, va="bottom", ha="right", color="0.3")
ax[-1].set_xlabel(
    "optimizer steps (validation every 500; earliest stop = min_rounds 20k + final_patience 15k = 35k steps)"
)
ax[-1].set_xlim(0, S)
sec = ax[0].secondary_xaxis("top", functions=(lambda x: x / S * 100, lambda p: p / 100 * S))
sec.set_xlabel("% of budget")
fig.tight_layout()
fig.savefig("/tmp/sched.png", dpi=130)
