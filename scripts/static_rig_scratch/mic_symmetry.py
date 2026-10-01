"""(b) Is the pair-specific residual of the rank-1-per-harmonic fit smooth in
frequency?  (c) C2 symmetry: Motor1 at mic c vs Motor3 at mic c+4 (and
Motor2/Motor4) — under identical rotors + symmetric geometry the difference
is a function of k only.  Usage: mic_symmetry.py <duty> [k_max]."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "notebooks")
import single_rotor_lab as L  # noqa: E402

duty = sys.argv[1] if len(sys.argv) > 1 else "70"
K_MAX = int(sys.argv[2]) if len(sys.argv) > 2 else 60
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
Y, M = np.stack(Y), np.stack(M)
R, C, K = Y.shape
W = M.astype(float)
k = np.arange(1, K + 1)


def wmean(v, w, axis):
    return np.where(w.sum(axis) > 0, (w * v).sum(axis) / np.maximum(w.sum(axis), 1e-9), 0.0)


p = wmean(Y, W, axis=1)
h = np.zeros((C, K))
for _ in range(300):
    p = wmean(Y - h[None], W, axis=1)
    h = wmean(Y - p[:, None, :], W, axis=0)
    h -= h.mean(axis=0, keepdims=True)
E = np.where(M, Y - p[:, None, :] - h[None], np.nan)  # (R, C, K) pair residual

print(f"duty {duty}, orders 1–{K}: ε sd {np.nanstd(E):.2f} dB")
print("(b) smoothness of ε along k, pooled over (rotor, mic):")
for lag in (1, 2, 3, 4, 6, 8):
    a = E[:, :, :-lag][M[:, :, :-lag] & M[:, :, lag:]]
    b = E[:, :, lag:][M[:, :, :-lag] & M[:, :, lag:]]
    print(f"    autocorrelation at Δk={lag}: {np.corrcoef(a, b)[0, 1]:+.2f}")
for w in (3, 5, 9):
    ker = np.ones(w) / w
    sm = np.stack(
        [[np.convolve(np.nan_to_num(E[r, c]), ker, "same") for c in range(C)] for r in range(R)]
    )
    v = (sm[M] ** 2).mean() / (np.nan_to_num(E)[M] ** 2).mean()
    print(f"    fraction of ε variance in a {w}-order moving average: {v:.2f}")
ev, od = E[:, :, 1::2][M[:, :, 1::2]], E[:, :, 0::2][M[:, :, 0::2]]
print(f"    ε sd on even orders {ev.std():.2f} dB, odd orders {od.std():.2f} dB")

print(
    "(c) C2 symmetry (mic c ↔ c+4): sd across mics of the profile difference per order, median over k"
)
for ra, rb in ((0, 2), (1, 3)):
    rows = []
    for name, perm in (("matched c↔c+4", (np.arange(C) + 4) % C), ("same mic index", np.arange(C))):
        D = Y[ra] - Y[rb][perm]
        Mm = M[ra] & M[rb][perm]
        sd_k = np.array(
            [np.std(D[Mm[:, j], j]) if Mm[:, j].sum() >= 3 else np.nan for j in range(K)]
        )
        rows.append((name, np.nanmedian(sd_k), np.nanmedian(sd_k[:30]), np.nanmedian(sd_k[30:])))
    rng = np.random.default_rng(0)
    null = []
    for _ in range(200):
        perm = rng.permutation(C)
        D = Y[ra] - Y[rb][perm]
        Mm = M[ra] & M[rb][perm]
        null.append(
            np.nanmedian(
                [np.std(D[Mm[:, j], j]) if Mm[:, j].sum() >= 3 else np.nan for j in range(K)]
            )
        )
    print(f"  {ROTORS[ra]} vs {ROTORS[rb]}:")
    for name, all_, lo, hi in rows:
        print(f"    {name:16s}: {all_:.2f} dB (k≤30 {lo:.2f}, k>30 {hi:.2f})")
    print(
        f"    random mic pairing: {np.median(null):.2f} dB (5–95 %: {np.percentile(null, 5):.2f}–{np.percentile(null, 95):.2f})"
    )
    print(
        f"    mic-to-mic spread of {ROTORS[ra]} alone: {np.nanmedian([np.std(Y[ra][M[ra][:, j], j]) for j in range(K)]):.2f} dB"
    )
