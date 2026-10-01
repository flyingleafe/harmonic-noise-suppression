"""Correlation of the per-order amplitude tracks ln A_k(t) as a function of
order separation Δk (non-windy mics, orders with SNR ≥ 10 dB), to see
whether neighbouring harmonics breathe together even if far ones do not."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "notebooks")
import single_rotor_lab as L  # noqa: E402
from experiments.static_rig import single_rotor as SR  # noqa: E402

DK = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50]
out = {}
for key, (s, _) in L.TABLE.items():
    x, fs = L.load(key)
    a, b = SR.strict_span(x, fs, s)
    x = x[:, a:b]
    aj = SR.amplitude_jitter(x, fs, s)
    windy = L.WIND.get(key.split("_")[1], ())
    good = [c for c in range(x.shape[0]) if c not in windy]
    # rebuild the detrended tracks exactly as amplitude_jitter does
    nb = int(aj.frame_s * fs)
    hop = nb // 2
    n_fr = (x.shape[1] - nb) // hop + 1
    w = np.hanning(nb)
    nfft = 1 << int(np.ceil(np.log2(2 * nb)))
    df = fs / nfft
    K = aj.orders.size
    idx = np.rint(aj.orders * s / df).astype(int)
    h = max(1, int(round(0.5 * s / 4 / df)))
    A2 = np.zeros((x.shape[0], K, n_fr))
    for i in range(n_fr):
        X = np.fft.rfft(x[:, i * hop : i * hop + nb] * w, nfft, axis=1)
        P = (np.abs(X) * (2.0 / w.sum())) ** 2
        for j in range(K):
            A2[:, j, i] = P[:, idx[j] - h : idx[j] + h + 1].max(axis=1)
    lnA = 0.5 * np.log(np.maximum(A2, 1e-300))
    t = np.arange(n_fr) * hop / fs
    rows = {}
    for dk in DK:
        vals = []
        for c in good:
            for j in range(K - dk):
                if aj.snr_db[c, j] < 10 or aj.snr_db[c, j + dk] < 10:
                    continue
                y1 = lnA[c, j] - np.polyval(np.polyfit(t, lnA[c, j], 1), t)
                y2 = lnA[c, j + dk] - np.polyval(np.polyfit(t, lnA[c, j + dk], 1), t)
                vals.append(np.corrcoef(y1, y2)[0, 1])
        rows[dk] = (float(np.median(vals)), float(np.mean(vals)), len(vals)) if vals else None
    # neighbour correlation by order band
    band = {}
    for lo, hi in ((1, 8), (9, 30), (31, 150)):
        vals = []
        for c in good:
            for j in range(lo - 1, min(hi, K) - 1):
                if aj.snr_db[c, j] < 10 or aj.snr_db[c, j + 1] < 10:
                    continue
                y1 = lnA[c, j] - np.polyval(np.polyfit(t, lnA[c, j], 1), t)
                y2 = lnA[c, j + 1] - np.polyval(np.polyfit(t, lnA[c, j + 1], 1), t)
                vals.append(np.corrcoef(y1, y2)[0, 1])
        band[f"{lo}-{hi}"] = float(np.median(vals)) if vals else None
    out[key] = {"by_dk": rows, "dk1_by_band": band}
    print(
        key,
        " ".join(f"Δ{dk}:{rows[dk][0]:+.2f}" for dk in DK if rows[dk]),
        "| Δ1 by band",
        " ".join(f"{k}:{v:+.2f}" for k, v in band.items() if v is not None),
        flush=True,
    )
Path("results/static_rig/single_rotor/am_corr_lag.json").write_text(json.dumps(out, indent=1))
