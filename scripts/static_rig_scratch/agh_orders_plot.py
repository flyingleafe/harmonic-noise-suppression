import importlib.util
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
_o = importlib.util.spec_from_file_location(
    "orc", "scripts/static_rig_scratch/one_rotor_coherence.py"
)
orc = importlib.util.module_from_spec(_o)
_o.loader.exec_module(orc)
FS = 44100.0
fig, ax = plt.subplots(5, 1, figsize=(16, 15), sharex=False)
for row, i in enumerate([4, 3, 2, 1, 0]):
    a, r = orc.agh_single_rotor(i)
    x = a[0] - a[0].mean()
    n = len(x)
    w = np.hanning(n)
    P = np.abs(np.fft.rfft(x * w)) ** 2 / (w**2).sum() * 2 / FS
    fr = np.fft.rfftfreq(n, 1 / FS)
    sel = (fr > 3 * r * 0.985) & (fr < 3 * r * 1.015)
    s = fr[sel][np.argmax(P[sel])] / 3
    # smooth to 1 Hz for display
    k1 = int(1 / fr[1])
    Ps = np.convolve(P, np.ones(k1) / k1, mode="same")
    m = fr < 25 * s
    ax[row].plot(fr[m] / s, 10 * np.log10(Ps[m]), lw=0.5, color="0.2")
    for k in range(1, 25):
        c = "tab:blue" if k % 3 == 0 else "tab:orange"
        ax[row].axvline(k, color=c, lw=0.8, alpha=0.6)
    ax[row].set(
        ylabel="dB",
        title=f"AGH rotor 4 idx {i}: {s:.1f} rev/s  (blue 3n = BPF, orange = asymmetry orders)",
        xlim=(0, 25),
    )
    ax[row].grid(alpha=0.3)
ax[-1].set_xlabel("order k = f / shaft rate")
fig.tight_layout()
fig.savefig("/tmp/agh_odd.png", dpi=90)
