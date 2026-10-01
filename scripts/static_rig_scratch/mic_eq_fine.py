"""Fine-frequency version of mic_eq.py on the inter-harmonic gaps only.

Per (rotor, mic, duty): R(f) = P_m(f) / mean_m' P_m'(f) in dB at ~2.7 Hz
bins (Welch, nperseg 16384), harmonic bins masked (± max(3 γ_k, 4 Hz) around
every k·s̄ of that duty). Cross-duty correlation on the frequency axis at
smoothing 0 / 5 / 10 / 20 / 50 Hz (both duties' masks applied), the
autocorrelation width of R along f (room correlation bandwidth), and the
gap-floor level with the rotor on vs the pre-spin-up segment (rotor off)."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch

sys.path.insert(0, "notebooks")
import single_rotor_lab as L  # noqa: E402

ROTORS = ["Motor1", "Motor2", "Motor3", "Motor4"]
DUTIES = ["50", "60", "70", "80", "90"]
F_LO, F_HI = 400.0, 5000.0
NPERSEG = 16384
SMOOTHS = [0, 5, 10, 20, 50]
OU = Path("results/static_rig/single_rotor/ou")


def hwhm_k(key, K):
    d = json.loads((OU / f"{key}.json").read_text())
    from experiments.static_rig import single_rotor as SR

    return np.array([SR.ou_hwhm(k, d["sigma_nu"], d["lam"]) for k in range(1, K + 1)])


out = {}
for r in ROTORS:
    windy = set(L.WIND[r])
    good = [c for c in range(8) if c not in windy]
    Rd, Md, floor_on_off = {}, {}, {}
    f = None
    for d in DUTIES:
        key = f"motor_{r}_{d}"
        s, _ = L.TABLE[key]
        x, fs, model = L._analysed(key)
        a, b = model.span
        f, P = welch(x[:, a:b].astype(np.float64), fs, nperseg=NPERSEG, axis=1)
        sel = (f >= F_LO) & (f <= F_HI)
        f, P = f[sel], P[:, sel]
        K = int(F_HI // s) + 1
        hw = hwhm_k(key, K)
        mask = np.ones(f.size, bool)
        for k in range(1, K + 1):
            half = max(3 * hw[k - 1], 4.0)
            mask &= np.abs(f - k * s) > half
        ref = P[good].mean(axis=0)
        Rd[d] = 10 * np.log10(P / ref)
        Md[d] = mask
        # gap floor on vs off (pre-span, rotor off): median over gap bins of the array-mean spectrum
        pre = x[:, : max(a - int(0.5 * fs), int(fs))].astype(np.float64)
        f0, P0 = welch(pre, fs, nperseg=NPERSEG, axis=1)
        P0 = P0[:, (f0 >= F_LO) & (f0 <= F_HI)]
        on, off = P[good].mean(0)[mask], P0[good].mean(0)[mask]
        floor_on_off[d] = float(10 * np.log10(np.median(on / off)))
    assert f is not None
    df = float(f[1] - f[0])
    print(
        f"{r}: gap floor, rotor on − rotor off (median over gaps, dB): "
        + ", ".join(f"{d}: {v:+.1f}" for d, v in floor_on_off.items())
    )
    res = {}
    for sm in SMOOTHS:
        w = max(1, int(round(sm / df)))
        ker = np.ones(w) / w
        corr = []
        for c in good:
            for ia, da in enumerate(DUTIES):
                for db in DUTIES[ia + 1 :]:
                    m = Md[da] & Md[db]
                    xa, xb = Rd[da][c], Rd[db][c]
                    if w > 1:  # smooth over unmasked bins only (masked bins interpolated first)
                        xa = np.convolve(np.interp(f, f[Md[da]], xa[Md[da]]), ker, "same")
                        xb = np.convolve(np.interp(f, f[Md[db]], xb[Md[db]]), ker, "same")
                    corr.append(np.corrcoef(xa[m], xb[m])[0, 1])
        res[sm] = float(np.mean(corr))
    print(
        "    cross-duty correlation of R on the frequency axis vs smoothing: "
        + ", ".join(f"{sm} Hz: {v:+.2f}" for sm, v in res.items())
    )
    # autocorrelation of unsmoothed R along f (gaps, per mic & duty), pooled
    ac = {}
    for lag_hz in (2.7, 5.4, 8.1, 10.8, 16.2, 27, 54, 108):
        lag = int(round(lag_hz / df))
        vals = []
        for d in DUTIES:
            m = Md[d]
            for c in good:
                x_ = Rd[d][c] - np.convolve(
                    np.interp(f, f[m], Rd[d][c][m]), np.ones(int(200 / df)) / int(200 / df), "same"
                )  # remove the broad shape
                mm = m[:-lag] & m[lag:]
                vals.append(np.corrcoef(x_[:-lag][mm], x_[lag:][mm])[0, 1])
        ac[lag_hz] = float(np.mean(vals))
    print(
        "    autocorrelation of R along f (broad shape removed): "
        + ", ".join(f"{k:.0f} Hz: {v:+.2f}" for k, v in ac.items())
    )
    fine_sd = np.mean(
        [
            np.std(
                Rd[d][c][Md[d]]
                - np.convolve(
                    np.interp(f, f[Md[d]], Rd[d][c][Md[d]]),
                    np.ones(int(50 / df)) / int(50 / df),
                    "same",
                )[Md[d]]
            )
            for d in DUTIES
            for c in good
        ]
    )
    print(f"    sd of the fine (< 50 Hz) part of R in the gaps: {fine_sd:.2f} dB")
    out[r] = {
        "corr_vs_smoothing": res,
        "autocorr": ac,
        "floor_on_off": floor_on_off,
        "fine_sd": float(fine_sd),
    }
Path("results/static_rig/single_rotor/mic_eq_fine.json").write_text(json.dumps(out, indent=1))
