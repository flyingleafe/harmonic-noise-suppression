import json
import sys

import numpy as np
from scipy.signal import welch

sys.path.insert(0, "src")
from experiments.static_rig import spectra as S
from experiments.static_rig.spectra import _floor

key = sys.argv[1]
u = {v["key"]: v for v in json.load(open("/tmp/sr_avq/units.json"))}[key]
x0 = np.load(u["audio"], mmap_mode="r")
fs = u["fs"]
a, b = S.motor_on_span(np.asarray(x0), fs, S.Params())
x = np.asarray(x0[:, a:b], dtype=np.float64)
d = np.load(f"/tmp/avq_amp_{key}.npz")
means = {f: float(d[f"spd_{f}"].mean()) for f in "ABCD"}
fr, P = welch(x, fs, nperseg=fs // 2, axis=1)  # 2 Hz
Pm = np.median(P, axis=0)  # mic-median PSD
fl = 10 ** (_floor(10 * np.log10(Pm[None] + 1e-30), 10)[0] / 10)  # +-20 Hz median
prom = 10 * np.log10(Pm / fl)
# peaks > 6 dB prominence above 2.5 kHz
from scipy.signal import find_peaks

pk, _ = find_peaks(prom, height=6, distance=3)
pk = pk[fr[pk] > 2500]
print(
    f"== {key} mean speeds {means}; peaks >6 dB over +-20 Hz median floor above 2.5 kHz: {pk.size}"
)
print("   f(Hz)  prom(dB)  nearest harmonic (rotor k, offset Hz) for each rotor")
for i in pk[np.argsort(prom[pk])[::-1]][:40]:
    f = fr[i]
    cand = []
    for r, s in means.items():
        k = round(f / s)
        cand.append(f"{r}{k:<3d}{f - k * s:+6.1f}")
    print(f"  {f:7.0f}  {prom[i]:5.1f}   " + "  ".join(cand))
# spectrum summary per kHz: top prominence and count of peaks
for lo in range(0, 22000, 1000):
    m = (fr >= lo) & (fr < lo + 1000)
    print(
        f"   {lo / 1000:4.0f}-{lo / 1000 + 1:.0f} kHz: floor {10 * np.log10(fl[m].mean()):6.1f} dB, max prom {prom[m].max():5.1f} dB at {fr[m][np.argmax(prom[m])]:.0f} Hz, peaks>6dB {int((prom[m] > 6).sum())}"
    )
