import json
import sys

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, "src")
from experiments.static_rig import spectra as S
from experiments.static_rig.spectra import _floor

key = sys.argv[1]
fam_lo, fam_hi = float(sys.argv[2]), float(sys.argv[3])
FS = 16000
u = {v["key"]: v for v in json.load(open("/tmp/sr_avq/units.json"))}[key]
x0 = np.load(u["audio"], mmap_mode="r")
fs0 = u["fs"]
a, b = S.motor_on_span(np.asarray(x0), fs0, S.Params())
r = json.load(open(f"/tmp/sr_joint/AVQ__{key}.json"))
rows = []
for lf in r["linked"][2:8]:
    sp = [s for s in lf["speeds"] if fam_lo <= s <= fam_hi]
    if not sp:
        continue
    s = sp[0]
    t0, t1 = lf["t0"], lf["t1"]
    seg = resample_poly(
        np.asarray(x0[:, a + int(t0 * fs0) : a + int(t1 * fs0)], dtype=np.float64), FS, fs0, axis=1
    )
    n = seg.shape[1]
    w = np.hanning(n)
    nfft = 1 << int(np.ceil(np.log2(8 * n)))
    dfp = FS / nfft
    P = np.abs(np.fft.rfft(seg * w, nfft, axis=1)) ** 2
    fl = 10 ** (_floor(10 * np.log10(P + 1e-30), max(2, round(5 / dfp))) / 10) / np.log(2)
    out = []
    for k in (1, 2, 4, 6, 8, 10, 12, 14):
        f = k * s
        # local peak within +-1% (track error), then energy vs half-width
        c = int(round(f / dfp))
        h0 = int(round(0.012 * f / dfp)) + 2
        i = c - h0 + int(np.argmax(P[:, c - h0 : c + h0 + 1].sum(0)))
        vals = []
        for hw in (0.25, 0.5, 1, 2, 3, 5, 8):
            h = max(1, int(round(hw / dfp)))
            line = (P[:, i - h : i + h + 1].sum(1) - fl[:, i - h : i + h + 1].sum(1)) / 8
            vals.append(10 * np.log10(max(np.median(line), 1e-30)))
        snr = 10 * np.log10(
            max(
                np.median(
                    (P[:, i - int(2 / dfp) : i + int(2 / dfp) + 1].sum(1) / 8)
                    / (fl[:, i - int(2 / dfp) : i + int(2 / dfp) + 1].sum(1) / 8)
                ),
                1e-3,
            )
        )
        out.append(
            f"k{k:<2d} off {(i - c) * dfp:+5.2f}Hz snr4Hz {snr:5.1f} | "
            + " ".join(f"{v - vals[-1]:5.1f}" for v in vals)
        )
    print(
        f"win {t0:.0f}-{t1:.0f}s s={s:.2f}  (dB rel. to +-8 Hz band; half-widths 0.25 0.5 1 2 3 5 8 Hz)"
    )
    for o in out:
        print("   ", o)
