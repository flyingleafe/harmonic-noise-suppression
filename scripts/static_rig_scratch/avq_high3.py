import json
import sys

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, "src")
from experiments.static_rig import spectra as S

key = sys.argv[1]
flo, fhi = float(sys.argv[2]), float(sys.argv[3])
u = {v["key"]: v for v in json.load(open("/tmp/sr_avq/units.json"))}[key]
x0 = np.load(u["audio"], mmap_mode="r")
fs = u["fs"]
a, b = S.motor_on_span(np.asarray(x0), fs, S.Params())
r = json.load(open(f"/tmp/sr_joint/AVQ__{key}.json"))
d = np.load(f"/tmp/avq_amp_{key}.npz")
print(
    f"== {key} {flo:.0f}-{fhi:.0f} Hz: per window, peaks >= 8 dB over the 20th-percentile floor (+-60 Hz), mic-median periodogram smoothed to 0.5 Hz; assignment = rotor k if |off| < 0.15% "
)
from scipy.ndimage import percentile_filter

for wi, lf in enumerate(r["linked"]):
    sp = {f: float(d[f"spd_{f}"][wi]) for f in "ABCD"}
    seg = np.asarray(x0[:, a + int(lf["t0"] * fs) : a + int(lf["t1"] * fs)], dtype=np.float64)
    n = seg.shape[1]
    w = np.hanning(n)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    dfp = fs / nfft
    lo, hi = int(flo / dfp), int(fhi / dfp)
    P = np.median(np.abs(np.fft.rfft(seg * w, nfft, axis=1)[:, lo - 2000 : hi + 2000]) ** 2, axis=0)
    sm = int(round(0.5 / dfp))
    P = np.convolve(P, np.ones(sm) / sm, "same")
    fl = percentile_filter(P, 20, size=int(120 / dfp))
    prom = 10 * np.log10(P / fl)
    pk, _ = find_peaks(prom, height=8, distance=int(3 / dfp))
    pk = pk[(pk >= 2000) & (pk < hi - lo + 2000)]
    items = []
    for i in pk[np.argsort(prom[pk])[::-1]][:12]:
        f = (lo - 2000 + i) * dfp
        best = min(
            (
                (abs(f - round(f / s) * s) / f, g, round(f / s), f - round(f / s) * s)
                for g, s in sp.items()
            )
        )
        tag = f"{best[1]}{best[2]}({best[3]:+.1f})" if best[0] < 0.0015 else "?"
        items.append(f"{f:6.0f}:{prom[i]:4.1f}dB {tag}")
    print(f"  {lf['t0']:5.1f}-{lf['t1']:5.1f}  " + "  ".join(items))
