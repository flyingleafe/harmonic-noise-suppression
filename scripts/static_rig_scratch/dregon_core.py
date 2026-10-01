import json
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "/tmp")
from dregon_simple import strict_span

from experiments.static_rig import spectra as S

units = {v["key"]: v for v in json.load(open("/tmp/sr_dregon/units.json"))}
prior = {
    r["uid"].split("__")[-1]: r["full"]["whitened"]["speed"]
    for r in json.load(open("/tmp/bretthorst_dregon.json"))
}
WIND = {"Motor1": [6], "Motor2": [0], "Motor3": [1, 2], "Motor4": [4]}
print(
    "key              T(s)  1/T(Hz)  s_core(median k2-12, good ch)  sd over orders  sd over ch  per-order (k:dev*1000)   core HWHM(k2,k8)/resolution"
)
for key in sys.argv[1:] or [
    f"motor_Motor{m}_{t}" for m in (1, 2, 3, 4) for t in (50, 60, 70, 80, 90)
]:
    u = units[key]
    fs = u["fs"]
    x0 = np.load(u["audio"], mmap_mode="r")
    a, b = S.motor_on_span(np.asarray(x0), fs, S.Params())
    x = np.asarray(x0[:, a:b], dtype=np.float64)
    s0 = prior[key]
    a2, b2 = strict_span(x, fs, s0)
    x = x[:, a2:b2]
    n = x.shape[1]
    T = n / fs
    w = np.hanning(n)
    nfft = 1 << int(np.ceil(np.log2(4 * n)))
    df = fs / nfft
    good = [c for c in range(x.shape[0]) if c not in WIND[key.split("_")[1]]]
    ks = list(range(2, 13))
    pk = np.full((len(good), len(ks)), np.nan)
    hw = {}
    for ci, c in enumerate(good):
        X = np.abs(np.fft.rfft(x[c] * w, nfft)) ** 2
        for j, k in enumerate(ks):
            c0 = int(round(k * s0 / df))
            h = int(1.0 / df)
            seg = X[c0 - h : c0 + h + 1]
            i = int(np.argmax(seg))
            if i == 0 or i == seg.size - 1:
                continue
            y0, y1, y2 = np.log(seg[i - 1 : i + 2])
            d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
            pk[ci, j] = (c0 - h + i + d) * df / k
            if k in (2, 8) and ci == 0:
                half = seg >= seg[i] / 2
                hw[k] = (np.nonzero(half)[0].max() - np.nonzero(half)[0].min() + 1) * df / 2
    per_order = np.nanmedian(pk, axis=0)
    per_ch = np.nanmedian(pk, axis=1)
    s = np.nanmedian(pk)
    print(
        f"{key:16s} {T:4.1f}  {1 / T:.3f}   {s:9.4f}                 {np.nanstd(per_order):.4f}          {np.nanstd(per_ch):.4f}     "
        + " ".join(f"{k}:{(v - s) * 1000:+.0f}" for k, v in zip(ks, per_order))
        + f"   {hw.get(2, 0) / (1 / T):.1f} {hw.get(8, 0) / (1 / T):.1f}"
    )
