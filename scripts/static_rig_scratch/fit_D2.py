"""D from the measured HWHM of resolved lines: gamma_k = pi k^2 D."""

import math
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "notebooks")
import single_rotor_lab as L

from experiments.static_rig import single_rotor as SR

print(
    "key           s     D_phase   D_hwhm   D_hwhm/s   n_orders  (gamma_k/k^2/pi over k: p25/p50/p75)"
)
out = []
for key in L.RECORDINGS:
    x, fs = L.load(key)
    s_bar, r1 = L.TABLE[key]
    good = [c for c in range(x.shape[0]) if c not in L.WIND[key.split("_")[1]]]
    a, b = SR.strict_span(x[good], fs, s_bar)
    xs = np.asarray(x[good, a:b], dtype=np.float64)
    f, P, w = SR._periodogram(xs, fs, pad=4)
    df = float(f[1])
    T = xs.shape[1] / fs
    Pm = np.median(P, axis=0)
    Pm = np.convolve(Pm, np.ones(5) / 5, mode="same")
    Ds = []
    for k in range(8, 100):
        c0 = int(round(k * s_bar / df))
        half = int(round(s_bar / 4 / df))
        seg = Pm[c0 - half : c0 + half + 1]
        i = int(np.argmax(seg))
        pk = seg[i]
        floor = np.percentile(seg, 20)
        if pk < 10 * floor:
            continue  # need 10 dB over the local level
        # half-power points relative to (pk - floor)
        lvl = floor + (pk - floor) / 2
        lo = i
        while lo > 0 and seg[lo] > lvl:
            lo -= 1
        hi = i
        while hi < seg.size - 1 and seg[hi] > lvl:
            hi += 1
        hwhm = (hi - lo) / 2 * df
        if hwhm < 2.0 / T:
            continue  # unresolved (Hann main lobe)
        if hwhm > s_bar / 8:
            continue  # merged
        Ds.append(hwhm / (math.pi * k * k))
    Ds = np.array(Ds)
    D_h = float(np.median(Ds)) if Ds.size else float("nan")
    out.append((s_bar, r1**2, D_h))
    print(
        f"{key:14s} {s_bar:6.2f} {r1**2:.2e}  {D_h:.2e}  {D_h / s_bar:.2e}   {Ds.size:3d}    "
        + (
            f"{np.percentile(Ds, 25):.2e}/{np.percentile(Ds, 50):.2e}/{np.percentile(Ds, 75):.2e}"
            if Ds.size
            else ""
        )
    )
r = np.array(out)
print(
    f"\nD_hwhm/s median {np.nanmedian(r[:, 2] / r[:, 0]):.2e} (p10-p90 {np.nanpercentile(r[:, 2] / r[:, 0], 10):.2e}-{np.nanpercentile(r[:, 2] / r[:, 0], 90):.2e}); D_hwhm median {np.nanmedian(r[:, 2]):.2e} (p10-p90 {np.nanpercentile(r[:, 2], 10):.2e}-{np.nanpercentile(r[:, 2], 90):.2e})"
)
ok = np.isfinite(r[:, 2])
print(
    f"corr(log D_hwhm, log s) {np.corrcoef(np.log(r[ok, 2]), np.log(r[ok, 0]))[0, 1]:.2f}; corr(log D_hwhm, log D_phase) {np.corrcoef(np.log(r[ok, 2]), np.log(r[ok, 1]))[0, 1]:.2f}"
)
