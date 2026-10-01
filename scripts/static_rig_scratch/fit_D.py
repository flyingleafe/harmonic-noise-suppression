"""Fit the phase diffusion D per recording: the D whose Lorentzian-through-
kernel line shape best matches the measured lines (top 12 dB of each line,
orders 6..80, mic-median periodogram), in log-spectral least squares."""

import math
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "notebooks")
import single_rotor_lab as L  # noqa: E402

from experiments.static_rig import single_rotor as SR  # noqa: E402

D_GRID = np.geomspace(3e-5, 5e-3, 60)
ORDERS = range(6, 101)


def line_model(ff, gamma, kern, df):
    if gamma < df:
        lor = np.zeros_like(ff)
        lor[ff.size // 2] = 1.0
    else:
        lor = (gamma / math.pi) / (ff**2 + gamma**2) * df
    return np.convolve(lor, kern, mode="same")


print("key           s      D_phase   D_fit    D_fit/s   ratio  err_fit  err_phase (dB rms)")
rows = []
for key in L.RECORDINGS:
    x, fs = L.load(key)
    s_bar, r1 = L.TABLE[key]
    wind = L.WIND[key.split("_")[1]]
    good = [c for c in range(x.shape[0]) if c not in wind]
    a, b = SR.strict_span(x[good], fs, s_bar)
    xs = np.asarray(x[good, a:b], dtype=np.float64)
    f, P, w = SR._periodogram(xs, fs, pad=2)
    df = float(f[1])
    T = xs.shape[1] / fs
    Pm = np.median(P, axis=0)
    Pm = np.convolve(Pm, np.ones(3) / 3, mode="same")
    nfft = 2 * (f.size - 1)
    Wk = np.abs(np.fft.rfft(w, nfft)) ** 2 / w.sum() ** 2
    kh = int(round(4.0 / T / df)) + 1
    kern = np.concatenate([Wk[kh:0:-1], Wk[: kh + 1]])
    half = int(round(s_bar / 4 / df))
    ff = np.arange(-half, half + 1) * df
    segs = []
    for k in ORDERS:
        c0 = int(round(k * s_bar / df))
        seg = Pm[c0 - half : c0 + half + 1]
        pk = seg.max()
        if pk < 10**1.5 * np.percentile(seg, 20):  # need 15 dB over the local level
            continue
        top = seg >= pk * 10 ** (-1.0)
        if top.sum() < 3:
            continue
        segs.append((seg, top, k))
    errs = []
    for D in D_GRID:
        e = []
        for seg, top, k in segs:
            m = line_model(ff, math.pi * k * k * D, kern, df)
            # amplitude by the same rule as line_powers: mean over +-max(gamma,1/T)
            hw = max(1, int(round(max(math.pi * k * k * D, 1.0 / T) / df)))
            a_k = seg[half - hw : half + hw + 1].mean() / m[half - hw : half + hw + 1].mean()
            d = 10 * np.log10(seg[top] / np.maximum(a_k * m[top], 1e-30))
            e.append(np.mean(d**2))
        errs.append(math.sqrt(np.mean(e)))
    errs = np.array(errs)
    i = int(np.argmin(errs))
    D_fit = float(D_GRID[i])
    D_ph = r1**2
    e_ph = float(np.interp(math.log(D_ph), np.log(D_GRID), errs))
    print(
        f"{key:14s} {s_bar:6.2f}  {D_ph:.2e}  {D_fit:.2e}  {D_fit / s_bar:.2e}  {D_fit / D_ph:5.2f}  {errs[i]:5.2f}    {e_ph:5.2f}   n_orders {len(segs)}  err(D=4e-6*s) {float(np.interp(math.log(4e-6 * s_bar), np.log(D_GRID), errs)):5.2f}"
    )
    rows.append((key, s_bar, D_ph, D_fit))
r = np.array([[v[1], v[2], v[3]] for v in rows])
print(
    f"\nD_fit/s: median {np.median(r[:, 2] / r[:, 0]):.2e}, spread (p10-p90) {np.percentile(r[:, 2] / r[:, 0], 10):.2e}-{np.percentile(r[:, 2] / r[:, 0], 90):.2e};"
    f"  D_fit alone: median {np.median(r[:, 2]):.2e}, p10-p90 {np.percentile(r[:, 2], 10):.2e}-{np.percentile(r[:, 2], 90):.2e}"
)
print(
    f"corr(log D_fit, log s) = {np.corrcoef(np.log(r[:, 2]), np.log(r[:, 0]))[0, 1]:.2f};  corr(log D_fit, log D_phase) = {np.corrcoef(np.log(r[:, 2]), np.log(r[:, 1]))[0, 1]:.2f}"
)
