import json
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "/tmp")
from dregon_simple import strict_span

from experiments.static_rig import spectra as S

key = sys.argv[1]
units = {v["key"]: v for v in json.load(open("/tmp/sr_dregon/units.json"))}
prior = {
    r["uid"].split("__")[-1]: r["full"]["whitened"]["speed"]
    for r in json.load(open("/tmp/bretthorst_dregon.json"))
}
u = units[key]
fs = u["fs"]
x0 = np.load(u["audio"], mmap_mode="r")
a, b = S.motor_on_span(np.asarray(x0), fs, S.Params())
x = np.asarray(x0[:, a:b], dtype=np.float64)
s0 = prior[key]
a2, b2 = strict_span(x, fs, s0)
x = x[:, a2:b2]
n = x.shape[1]
w = np.hanning(n)
nfft = 1 << int(np.ceil(np.log2(2 * n)))
df = fs / nfft
X = np.stack([np.abs(np.fft.rfft(x[c] * w, nfft)) ** 2 for c in range(x.shape[0])])
X = np.median(X, axis=0)
offs = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
print(
    f"{key} T={n / fs:.1f}s: line shape per order; pedestal level (dB re core peak) at |offset| (Hz), averaged over +-25% of the offset, both sides; 'floor' = 20th pct within +-150 Hz"
)
print("  k    core/floor  " + "".join(f"{o:>7.2f}" for o in offs))
for k in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40):
    f0 = k * s0
    c0 = int(round(f0 / df))
    h = int(150 / df)
    lo = max(0, c0 - h)
    P = X[lo : c0 + h]
    f = (np.arange(lo, c0 + h) - c0) * df
    floor = np.percentile(P, 20)
    i = np.argmax(P[np.abs(f) < 3]) + np.nonzero(np.abs(f) < 3)[0][0]
    fc = f[i]
    pk = P[i]
    row = []
    for o in offs:
        m = np.abs(np.abs(f - fc) - o) <= 0.25 * o
        row.append(10 * np.log10(max(np.mean(P[m]) - floor, 1e-30) / pk))
    print(f"  k{k:<3d}   {10 * np.log10(pk / floor):5.1f}    " + "".join(f"{v:7.1f}" for v in row))
