"""Hypothesis: X_rm[k] = X_r[k] · H_m[k] + ε  — for every harmonic k the
(rotor × mic) amplitude matrix is rank one: a true rotor profile X_r times a
per-mic per-harmonic response H_m.

Fit in dB (multiplicative ε → additive), windy cells masked, alternating
means. Baselines: (0) rotor profile only, (1) rotor profile × scalar mic gain
(H_m flat in k). Usage: mic_factor.py <duty> [k_max]."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "notebooks")
import single_rotor_lab as L  # noqa: E402

duty = sys.argv[1] if len(sys.argv) > 1 else "70"
K_MAX = int(sys.argv[2]) if len(sys.argv) > 2 else 150
ROTORS = ["Motor1", "Motor2", "Motor3", "Motor4"]

Y, M = [], []
for r in ROTORS:
    d = json.loads(
        (Path("results/static_rig/single_rotor/ou") / f"motor_{r}_{duty}.json").read_text()
    )
    y = 10 * np.log10(np.maximum(np.array(d["amp2"])[:, :K_MAX], 1e-30))
    m = np.ones_like(y, bool)
    m[list(L.WIND[r])] = False
    Y.append(y)
    M.append(m)
Y, M = np.stack(Y), np.stack(M)  # (R, C, K) dB, mask
R, C, K = Y.shape
W = M.astype(float)


def wmean(v, w, axis):
    return np.where(w.sum(axis) > 0, (w * v).sum(axis) / np.maximum(w.sum(axis), 1e-9), 0.0)


def fit(h_mode, n_iter=300):
    p = wmean(Y, W, axis=1)  # (R, K)
    h = np.zeros((C, K))
    for _ in range(n_iter):
        p = wmean(Y - h[None], W, axis=1)
        if h_mode == "per_k":
            h = wmean(Y - p[:, None, :], W, axis=0)  # (C, K)
        elif h_mode == "flat":
            h = np.repeat(wmean(Y - p[:, None, :], W, axis=(0, 2))[:, None], K, axis=1)
        h -= wmean(h, np.ones_like(h), axis=0)[None]  # identifiability: mean over mics 0 per k
    res = Y - p[:, None, :] - h[None]
    return p, h, res


k = np.arange(1, K + 1)
bands = [(1, 8), (9, 30), (31, 60), (61, K)]
print(f"duty {duty}, orders 1–{K}, cells {M.sum()} (mics per rotor {M[:, :, 0].sum(1).tolist()})")
for name, mode in (
    ("0: X_r only (H_m ≡ 1)", "none"),
    ("1: X_r × scalar mic gain", "flat"),
    ("2: X_r[k] × H_m[k]  (hypothesis)", "per_k"),
):
    p, h, res = fit(mode)
    by_band = "  ".join(
        f"k{lo}–{hi}: {res[:, :, lo - 1 : hi][M[:, :, lo - 1 : hi]].std():.2f}" for lo, hi in bands
    )
    print(f"  {name:36s} ε sd {res[M].std():.2f} dB   by band: {by_band}")
    if mode == "per_k":
        np.save(f"/tmp/mic_h_{duty}.npy", h)
        np.save(f"/tmp/mic_p_{duty}.npy", p)
        np.save(f"/tmp/mic_res_{duty}.npy", np.where(M, res, np.nan))
        print(
            "  ε sd per rotor:",
            "  ".join(f"{r}: {res[i][M[i]].std():.2f}" for i, r in enumerate(ROTORS)),
        )
        print(
            "  ε sd per mic:  ",
            "  ".join(f"ch{c}: {res[:, c][M[:, c]].std():.2f}" for c in range(C)),
        )
        print(
            "  H_m spread (sd over k, dB):", "  ".join(f"ch{c}: {h[c].std():.2f}" for c in range(C))
        )
        # degrees of freedom check: with R rotors and C mics per k, rank-1 leaves (R-1)(C-1) residual dof per k
        dof = sum(
            max(0, (M[:, :, j].sum(0) > 0).sum() - 1) * max(0, (M[:, :, j].sum(1) > 0).sum() - 1)
            for j in range(K)
        )
        print(
            f"  residual dof ≈ {dof} of {M.sum()} cells ({dof / M.sum():.0%}); ε is estimated on that fraction"
        )
