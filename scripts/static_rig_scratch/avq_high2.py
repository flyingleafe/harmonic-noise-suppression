import json
import sys

import numpy as np

sys.path.insert(0, "src")
from experiments.static_rig import spectra as S

key = sys.argv[1]
KS = [int(k) for k in sys.argv[2].split(",")]
u = {v["key"]: v for v in json.load(open("/tmp/sr_avq/units.json"))}[key]
x0 = np.load(u["audio"], mmap_mode="r")
fs = u["fs"]
a, b = S.motor_on_span(np.asarray(x0), fs, S.Params())
r = json.load(open(f"/tmp/sr_joint/AVQ__{key}.json"))
d = np.load(f"/tmp/avq_amp_{key}.npz")
print(
    f"== {key}: per-window D harmonics; prom = peak within +-0.6% of k*s vs +-20 Hz median floor (dB, mic-median); off = peak offset from k*s (Hz)"
)
print("   win(s)      " + "".join(f"   k{k:<4d}    " for k in KS))
for wi, lf in enumerate(r["linked"]):
    s = float(d["spd_D"][wi])
    seg = np.asarray(x0[:, a + int(lf["t0"] * fs) : a + int(lf["t1"] * fs)], dtype=np.float64)
    n = seg.shape[1]
    w = np.hanning(n)
    nfft = 1 << int(np.ceil(np.log2(4 * n)))
    dfp = fs / nfft
    P = np.abs(np.fft.rfft(seg * w, nfft, axis=1)) ** 2
    Pm = np.median(P, axis=0)
    out = []
    for k in KS:
        f = k * s
        c = int(round(f / dfp))
        h = int(round(0.006 * f / dfp))
        loc = Pm[c - h : c + h + 1]
        # smooth over native resolution (pad 4 -> 4 bins) before peak-picking
        locs = np.convolve(loc, np.ones(8) / 8, mode="same")
        i = int(np.argmax(locs))
        fl = np.median(Pm[c - int(20 / dfp) : c + int(20 / dfp)])
        out.append(f" {10 * np.log10(locs[i] / fl):5.1f} {(i - h) * dfp:+6.1f}")
    print(f"   {lf['t0']:5.1f}-{lf['t1']:5.1f} " + " ".join(out))
