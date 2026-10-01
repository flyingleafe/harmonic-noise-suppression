"""Real vs OU-model (model 2) spectra for every DREGON single (full band + zooms at
chosen orders) and the directly measured line HWHM per order against the
OU HWHM law of the quick fit and a linear law.

Writes results/static_rig/single_rotor/figs/<key>.png and
results/static_rig/single_rotor/line_widths.json."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "notebooks")
import single_rotor_lab as L  # noqa: E402
from experiments.static_rig import single_rotor as SR  # noqa: E402

OUT = Path("results/static_rig/single_rotor/figs")
OUT.mkdir(parents=True, exist_ok=True)
ZOOM = [10, 42, 70, 84]


def hwhm(f, P, f0, half=12.0):
    """Half width at half maximum above the local floor of the line at f0
    (periodogram smoothed over 3 bins); nan if the line is < 6 dB over floor."""
    m = (f > f0 - half) & (f < f0 + half)
    ff, pp = f[m], np.convolve(P[m], np.ones(3) / 3, "same")
    floor = np.percentile(pp, 20) / 0.2231
    i = int(np.argmax(pp))
    pk = pp[i] - floor
    if pk < 3 * floor:
        return np.nan, floor, pp[i]
    lvl = floor + pk / 2
    lo = i
    while lo > 0 and pp[lo] > lvl:
        lo -= 1
    hi = i
    while hi < pp.size - 1 and pp[hi] > lvl:
        hi += 1
    return 0.5 * (ff[hi] - ff[lo]), floor, pp[i]


res = {}
keys = sys.argv[1:] or list(L.TABLE)
for key in keys:
    s, r1 = L.TABLE[key]
    x, fs, model = L._analysed(key)
    a, b = model.span
    x = x[:, a:b].astype(np.float64)
    wf = L.load_ou(key)
    am = L.am_per_order(key, wf.orders.size)
    y = SR.synthesise(wf, fs, b - a, am=am).astype(np.float64)
    c = 0 if "Motor4" in key else 3
    f, Pr, _ = SR._periodogram(x[c : c + 1], fs, pad=1)
    _, Ps, _ = SR._periodogram(y[c : c + 1], fs, pad=1)
    Pr, Ps = Pr[0], Ps[0]
    K = wf.orders.size
    w_real, w_synth = np.full(K, np.nan), np.full(K, np.nan)
    for j in range(K):
        w_real[j], _, _ = hwhm(f, Pr, (j + 1) * s)
        w_synth[j], _, _ = hwhm(f, Ps, (j + 1) * s)
    k = np.arange(1, K + 1)
    ok = np.isfinite(w_real) & (k >= 5)
    lin = np.polyfit(k[ok], w_real[ok], 1) if ok.sum() > 3 else [np.nan, np.nan]
    res[key] = {
        "hwhm_real": [None if not np.isfinite(v) else float(v) for v in w_real],
        "hwhm_synth": [None if not np.isfinite(v) else float(v) for v in w_synth],
        "linear_fit_hz_per_order": float(lin[0]),
        "D_whittle": wf.D,
    }
    fig, axs = plt.subplots(2, 3, figsize=(18, 9))
    ax = axs[0, 0]
    sm = np.ones(33) / 33
    ax.plot(f, 10 * np.log10(np.convolve(Pr, sm, "same") + 1e-30), lw=0.5, label="real")
    ax.plot(
        f, 10 * np.log10(np.convolve(Ps, sm, "same") + 1e-30), lw=0.5, alpha=0.7, label="model 2"
    )
    ax.set_xlim(0, 8000)
    ax.set_title(f"{key} ch{c}: full band (33-bin mean)")
    ax.legend()
    ax = axs[0, 1]
    ax.plot(k, w_real, ".", label="real HWHM")
    ax.plot(k, w_synth, "x", alpha=0.6, label="model 2 HWHM")
    ax.plot(k, wf.gamma, "-", label=f"OU HWHM σ_ν={wf.sigma_nu:.3f} λ={wf.lam:.1f}")
    ax.plot(k, np.polyval(lin, k), "--", label=f"linear {lin[0]:.3f} Hz/order")
    ax.set_ylim(0, 12)
    ax.set_xlabel("order k")
    ax.set_ylabel("HWHM (Hz)")
    ax.legend()
    ax.set_title("line half-width vs order")
    for ax, kz in zip(list(axs.flat)[2:], ZOOM, strict=False):
        f0 = kz * s
        m = (f > f0 - 20) & (f < f0 + 20)
        ax.plot(f[m] - f0, 10 * np.log10(Pr[m] + 1e-30), lw=0.7, label="real")
        ax.plot(f[m] - f0, 10 * np.log10(Ps[m] + 1e-30), lw=0.7, alpha=0.7, label="model 2")
        ax.set_title(
            f"k={kz} ({f0:.0f} Hz): OU HWHM = {SR.ou_hwhm(kz, wf.sigma_nu, wf.lam):.2f} Hz"
        )
        ax.set_xlabel("Hz from k·s̄")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"{key}.png", dpi=80)
    plt.close(fig)
    print(key, "done", flush=True)
Path("results/static_rig/single_rotor/line_widths.json").write_text(json.dumps(res))
