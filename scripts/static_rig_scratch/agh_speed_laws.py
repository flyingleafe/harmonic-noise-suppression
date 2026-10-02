import importlib.util
import json
import sys

import numpy as np
from scipy import signal

sys.path.insert(0, "src")
_s = importlib.util.spec_from_file_location("decoh", "scripts/noise_v2_order_decoherence.py")
decoh = importlib.util.module_from_spec(_s)
sys.modules["decoh"] = decoh
_s.loader.exec_module(decoh)
_o = importlib.util.spec_from_file_location(
    "orc", "scripts/static_rig_scratch/one_rotor_coherence.py"
)
orc = importlib.util.module_from_spec(_o)
_o.loader.exec_module(orc)
FS = 44100.0

# ---------- DREGON gain check: rotor-off ambient level per recording ----------
print(
    "=== DREGON rotor-off ambient level (dB, median PSD over quiet mics) per throttle: gain consistency ==="
)
singles = json.load(open("results/static_rig/single_rotor/dregon_singles.json"))["recordings"]
audio = dict(decoh.dregon_motor_audio(lambda r: r.startswith("motor_Motor")))
for m in (1, 2, 3, 4):
    row = []
    for t in (50, 60, 70, 80, 90):
        rid = f"motor_Motor{m}_{t}"
        a = audio[rid]
        mics = [i for i in range(8) if i not in set(singles[rid]["wind_mics"])]
        act0, _ = decoh.active_window(a)
        off = a[mics, : max(int(act0 - 0.5 * FS), int(FS))]
        f, pn = signal.welch(off - off.mean(1, keepdims=True), fs=FS, nperseg=4096, axis=-1)
        lo = 10 * np.log10(np.median(pn[:, (f > 100) & (f < 400)], axis=1)).mean()
        mid = 10 * np.log10(np.median(pn[:, (f > 1000) & (f < 4000)], axis=1)).mean()
        hi = 10 * np.log10(np.median(pn[:, (f > 8000) & (f < 16000)], axis=1)).mean()
        row.append(f"{t}%: {lo:6.1f}/{mid:6.1f}/{hi:6.1f}")
    print(f"M{m} (100-400 / 1-4k / 8-16k Hz): " + "  ".join(row))

# ---------- AGH rotor 4 ladder ----------
print("\n=== AGH rotor 4 (anechoic, mono, 3-blade), idx 0-4 ===")
recs = {}
for i in range(5):
    a, r = orc.agh_single_rotor(i)
    x = a[0] - a[0].mean()
    n = len(x)
    # refine the rate on the whole record from the k=3 (BPF) line of a fine periodogram
    w = np.hanning(n)
    X = np.fft.rfft(x * w)
    fr = np.fft.rfftfreq(n, 1 / FS)
    P = np.abs(X) ** 2 / (w**2).sum() * 2 / FS  # ~ PSD per Hz
    sel = (fr > 3 * r * 0.985) & (fr < 3 * r * 1.015)
    s = fr[sel][np.argmax(P[sel])] / 3
    # line power per order: sum of the periodogram within +-W of k s, minus the local floor; W = max(2 Hz, 0.3 % of f)
    df = fr[1]
    K = 150
    line = np.full(K, np.nan)
    floor_k = np.full(K, np.nan)
    for k in range(1, K + 1):
        c = k * s
        if c > 20000:
            break
        W = max(2.0, 0.003 * c)
        inw = np.abs(fr - c) <= W
        ring = (np.abs(fr - c) <= 60) & ~inw
        fl = np.percentile(P[ring], 20) / 0.223  # mean floor PSD per Hz
        line[k - 1] = max((P[inw] - fl).sum() * df, 1e-30)
        floor_k[k - 1] = fl
    fw, pw = signal.welch(x, fs=FS, nperseg=8192)
    recs[i] = dict(
        s=s,
        line_db=10 * np.log10(line),
        floor_k_db=10 * np.log10(floor_k),
        fw=fw,
        pw=pw,
        total_db=10 * np.log10((pw[(fw > 100) & (fw < 20000)]).sum() * fw[1]),
    )
    print(
        f"idx {i}: rate {s:7.2f} rev/s (survey {r}), total 100 Hz-20 kHz {recs[i]['total_db']:.1f} dB"
    )

xs = np.array([10 * np.log10(recs[i]["s"]) for i in range(5)])


def slope(y):
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return np.nan, np.nan, np.nan
    A = np.c_[xs[ok], np.ones(ok.sum())]
    c, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    r = y[ok] - A @ c
    se = np.sqrt((r**2).sum() / max(ok.sum() - 2, 1) / ((xs[ok] - xs[ok].mean()) ** 2).sum())
    return c[0], se, np.sqrt((r**2).mean())


print(f"total power exponent: {slope(np.array([recs[i]['total_db'] for i in range(5)]))[0]:.1f}")
print("\norder  family   alpha ± se (rms dB)   line-over-floor dB at idx4..0 (77->160 rev/s)")
for k in list(range(1, 25)) + [27, 30, 33, 36, 39, 42, 45, 48, 54, 60, 63, 72, 84, 90, 105, 126]:
    y = np.array([recs[i]["line_db"][k - 1] for i in range(5)])
    snr = np.array(
        [
            recs[i]["line_db"][k - 1]
            - recs[i]["floor_k_db"][k - 1]
            - 10 * np.log10(2 * max(2.0, 0.003 * k * recs[i]["s"]))
            for i in range(5)
        ]
    )
    fam = "BPF" if k % 3 == 0 else "asym"
    if k % 21 == 0:
        fam += "+motor"
    q, se, rms = slope(np.where(snr >= 6, y, np.nan))
    print(
        f"k={k:<4} {fam:<10} {q:5.1f} ± {se:3.1f} ({rms:3.1f})   snr "
        + " ".join(f"{v:4.0f}" for v in snr[::-1])
    )

# floor at fixed frequency and on the f/s axis
print("\nfloor exponent at fixed f (20th pct of Welch PSD) per band:")
for lo, hi in [
    (100, 200),
    (200, 400),
    (400, 800),
    (800, 1600),
    (1600, 3200),
    (3200, 6400),
    (6400, 12800),
    (12800, 20000),
]:
    y = np.array(
        [
            10
            * np.log10(
                np.percentile(recs[i]["pw"][(recs[i]["fw"] >= lo) & (recs[i]["fw"] < hi)], 20)
            )
            for i in range(5)
        ]
    )
    q, se, rms = slope(y)
    print(f"  {lo:>5}-{hi:<5} b = {q:4.1f} ± {se:3.1f} (rms {rms:3.1f} dB)")
print("\nfloor under harmonic k (fixed order) exponent beta:")
for k in [3, 6, 9, 12, 15, 21, 30, 42, 60, 84, 105]:
    y = np.array([recs[i]["floor_k_db"][k - 1] for i in range(5)])
    q, se, rms = slope(y)
    print(f"  k={k:<4} beta = {q:4.1f} ± {se:3.1f} (rms {rms:3.1f} dB)")

edges = 300 * 2 ** (np.arange(0, 40) / 6)
edges = edges[edges < 18000]
fc = np.sqrt(edges[:-1] * edges[1:])
S_REF = 110.0


def band20(fw, pw, lo, hi):
    sel = (fw >= lo) & (fw < hi)
    return 10 * np.log10(np.percentile(pw[sel], 20)) if sel.sum() >= 3 else np.nan


def axis_resid(curves, xsel, beta):
    Y = np.array([curves[i][xsel] - beta * 10 * np.log10(recs[i]["s"]) for i in range(5)])
    return float(np.sqrt(np.nanmean((Y - np.nanmean(Y, 0)) ** 2)))


cf = {
    i: np.array(
        [band20(recs[i]["fw"], recs[i]["pw"], edges[j], edges[j + 1]) for j in range(len(fc))]
    )
    for i in range(5)
}
cs = {
    i: np.array(
        [
            band20(
                recs[i]["fw"],
                recs[i]["pw"],
                edges[j] / S_REF * recs[i]["s"],
                edges[j + 1] / S_REF * recs[i]["s"],
            )
            for j in range(len(fc))
        ]
    )
    for i in range(5)
}
grid = np.arange(-2, 8.01, 0.1)
print("\naxis test (residual dB over 5 speeds): fixed f vs f/s")
for lo, hi in [(300, 16000), (300, 1500), (1500, 4000), (4000, 16000)]:
    xsel = (fc >= lo) & (fc < hi)
    bf = min(grid, key=lambda b: axis_resid(cf, xsel, b))
    bs = min(grid, key=lambda b: axis_resid(cs, xsel, b))
    print(
        f"  {lo:>5}-{hi:<5}: fixed-f b={bf:4.1f} resid {axis_resid(cf, xsel, bf):.2f} | f/s beta={bs:4.1f} resid {axis_resid(cs, xsel, bs):.2f}"
    )
json.dump(
    {
        str(i): dict(
            s=recs[i]["s"],
            line_db=recs[i]["line_db"].tolist(),
            floor_k_db=recs[i]["floor_k_db"].tolist(),
        )
        for i in recs
    },
    open("/tmp/agh_laws.json", "w"),
)

print(
    "\n=== per-recording band floors (dB) and line levels, to see whether idx 4 (77.7 rev/s) is an outlier ==="
)
bands = [(100, 200), (200, 400), (400, 800), (800, 1600), (1600, 3200), (3200, 6400), (6400, 12800)]
for i in range(4, -1, -1):
    fl = [
        10
        * np.log10(np.percentile(recs[i]["pw"][(recs[i]["fw"] >= lo) & (recs[i]["fw"] < hi)], 20))
        for lo, hi in bands
    ]
    ln = [recs[i]["line_db"][k - 1] for k in (1, 3, 6, 9, 12, 15, 18, 21)]
    print(
        f"idx {i} s={recs[i]['s']:6.1f}  floor "
        + " ".join(f"{v:6.1f}" for v in fl)
        + "   lines k1,3,6,9,12,15,18,21 "
        + " ".join(f"{v:6.1f}" for v in ln)
    )
print("\nexponents on idx 0-3 only (97-160 rev/s):")
xs4 = xs[:4]


def slope4(y):
    y = y[:4]
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return np.nan, np.nan
    A = np.c_[xs4[ok], np.ones(ok.sum())]
    c, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    r = y[ok] - A @ c
    return c[0], np.sqrt((r**2).mean())


for k in (1, 3, 6, 9, 12, 15, 18, 21):
    y = np.array([recs[i]["line_db"][k - 1] for i in range(5)])
    q, rms = slope4(y)
    print(f"  k={k:<3} alpha {q:5.1f} (rms {rms:3.1f})")
for lo, hi in bands:
    y = np.array(
        [
            10
            * np.log10(
                np.percentile(recs[i]["pw"][(recs[i]["fw"] >= lo) & (recs[i]["fw"] < hi)], 20)
            )
            for i in range(5)
        ]
    )
    q, rms = slope4(y)
    print(f"  floor {lo:>5}-{hi:<5} b {q:5.1f} (rms {rms:3.1f})")
