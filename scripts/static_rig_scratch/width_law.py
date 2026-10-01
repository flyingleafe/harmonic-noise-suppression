"""Per-order free Lorentzian half-width γ_k (1-D Whittle grid, amplitude in
closed form, local floor known) for every DREGON single, then the power law
γ_k ∝ k^p over orders with a clear line. Writes
results/static_rig/single_rotor/width_law.json."""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "notebooks")
import single_rotor_lab as L  # noqa: E402
from experiments.static_rig import single_rotor as SR  # noqa: E402

G = np.geomspace(0.02, 25.0, 48)
out = {}
for key in sys.argv[1:] or list(L.TABLE):
    s, _ = L.TABLE[key]
    x, fs, model = L._analysed(key)
    a, b = model.span
    x = x[:, a:b].astype(np.float64)
    good = [c for c in range(x.shape[0]) if c not in L.WIND[key.split("_")[1]]]
    f, P, w = SR._periodogram(x[good], fs, pad=1)
    df = float(f[1])
    T = x.shape[1] / fs
    nfft = 2 * (f.size - 1)
    Wk = np.abs(np.fft.rfft(w, nfft)) ** 2 / w.sum() ** 2
    kern = np.concatenate([Wk[4:0:-1], Wk[:5]])
    K = min(150, int(0.95 * fs / 2 // s))
    win, floor, off = SR._line_windows(f, P, s, K, T, half_hz=25.0, floor_hz=80.0)
    kk = np.arange(1, K + 1).astype(float)
    LL = np.zeros((G.size, K))
    for i, g in enumerate(G):
        sh = SR._shapes(off, np.full(K, g), 0.0, 0.0, kern, df, T)
        amp, _ = SR._profile_amplitudes(win, floor, sh)
        S = amp[:, :, None] * sh[None] + floor[:, :, None]
        LL[i] = -(np.log(S) + win / S).sum(axis=(0, 2))  # per order, summed over mics
    best = LL.argmax(axis=0)
    gam = G[best]
    # evidence that the width is resolved: ll gain of best over the narrowest grid point
    gain = LL[best, np.arange(K)] - LL[0]
    snr = 10 * np.log10(np.maximum(win.max(axis=2) / floor, 1e-3)).mean(axis=0)
    ok = (snr > 15) & (gain > 20) & (kk >= 3)
    p = np.polyfit(np.log(kk[ok]), np.log(gam[ok]), 1) if ok.sum() > 5 else [np.nan, np.nan]
    out[key] = {
        "gamma_k": gam.tolist(),
        "snr_db": snr.tolist(),
        "ok": ok.tolist(),
        "power": float(p[0]),
        "coef": float(math.exp(p[1])),
    }

    def sel(lo, hi):
        m = ok & (kk >= lo) & (kk <= hi)
        return np.median(gam[m]) if m.any() else np.nan

    print(
        f"{key}: γ_k median k 3–10 {sel(3, 10):.2f}, 11–30 {sel(11, 30):.2f}, 31–60 {sel(31, 60):.2f}, 61–100 {sel(61, 100):.2f} Hz; power law γ ∝ k^{p[0]:.2f} ({ok.sum()} orders); γ/k over ok: median {np.median(gam[ok] / kk[ok]):.3f} Hz/order",
        flush=True,
    )
Path("results/static_rig/single_rotor/width_law.json").write_text(json.dumps(out))
