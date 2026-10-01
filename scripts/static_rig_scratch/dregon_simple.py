"""DREGON single-rotor: strict rotor-on span, then per channel one Hann
periodogram; per order k: line centroid f_k and rms half-width w_k.
Mean speed = median of f_k/k; jitter sigma = slope of w_k vs k."""

import json
import sys

import numpy as np

sys.path.insert(0, "src")
from experiments.static_rig import spectra as S  # noqa: E402

ORDERS = range(1, 41)
SNR_MIN = 15.0  # dB line peak over local floor, per channel and order


def strict_span(x, fs, s0):
    nb = int(0.25 * fs)
    n_blk = x.shape[1] // nb
    w = np.hanning(nb)
    nfft = 1 << int(np.ceil(np.log2(4 * nb)))
    df = fs / nfft
    lo, hi = int(0.8 * 2 * s0 / df), int(1.2 * 2 * s0 / df)
    sb = np.array(
        [
            (
                lo
                + np.argmax(
                    (
                        np.abs(np.fft.rfft(x[:, i * nb : (i + 1) * nb] * w, nfft, axis=1)[:, lo:hi])
                        ** 2
                    ).sum(0)
                )
            )
            * df
            / 2
            for i in range(n_blk)
        ]
    )
    ok = np.abs(sb - np.median(sb)) < 0.01 * np.median(sb)
    best, run, start, bs = 0, 0, 0, 0
    for i, v in enumerate(ok):
        run = run + 1 if v else 0
        if v and run == 1:
            start = i
        if run > best:
            best, bs = run, start
    return (bs + 2) * nb, (bs + best - 2) * nb


def main(keys):
    units = {v["key"]: v for v in json.load(open("/tmp/sr_dregon/units.json"))}
    prior = {
        r["uid"].split("__")[-1]: r["full"]["whitened"]["speed"]
        for r in json.load(open("/tmp/bretthorst_dregon.json"))
    }
    print(
        "key            span(s)   T    s_mean   sd_ord  sd_ch   ch_used  k_used  sigma_jit  sd_ch   icpt   wind_ch(k2 snr dB)"
    )
    for key in keys:
        u = units[key]
        fs = u["fs"]
        x0 = np.load(u["audio"], mmap_mode="r")
        a, b = S.motor_on_span(np.asarray(x0), fs, S.Params())
        x = np.asarray(x0[:, a:b], dtype=np.float64)
        s0 = prior[key]
        a2, b2 = strict_span(x, fs, s0)
        x = x[:, a2:b2]
        n = x.shape[1]
        T = n / fs
        w = np.hanning(n)
        nfft = 1 << int(np.ceil(np.log2(2 * n)))
        df = fs / nfft
        C = x.shape[0]
        cent = np.full((C, len(ORDERS)), np.nan)
        width = np.full((C, len(ORDERS)), np.nan)
        snr2 = np.zeros(C)
        for c in range(C):
            X = np.abs(np.fft.rfft(x[c] * w, nfft)) ** 2
            for j, k in enumerate(ORDERS):
                f0 = k * s0
                if f0 > 8000:
                    break
                hw = max(3.0, 0.01 * f0)  # search/integration half-width (Hz)
                lo, hi = int((f0 - 4 * hw) / df), int((f0 + 4 * hw) / df)
                P = X[lo:hi]
                f = np.arange(lo, hi) * df
                floor = np.percentile(P, 20)
                m = np.abs(f - f0) <= hw
                snr = 10 * np.log10(P[m].max() / floor)
                if k == 2:
                    snr2[c] = snr
                if snr < SNR_MIN:
                    continue
                Pl = np.maximum(P - floor, 0)
                fc = np.sum(f[m] * Pl[m]) / np.sum(Pl[m])
                m2 = np.abs(f - fc) <= hw
                cent[c, j] = fc / k
                width[c, j] = np.sqrt(np.sum((f[m2] - fc) ** 2 * Pl[m2]) / np.sum(Pl[m2]))
        # wind: channels whose BPF line is > 10 dB below the channel median
        wind = snr2 < np.median(snr2) - 10
        cent[wind] = np.nan
        width[wind] = np.nan
        ks = np.array(list(ORDERS))
        per_order = np.nanmedian(cent, axis=0)
        per_ch = np.nanmedian(cent, axis=1)
        s_mean = float(np.nanmedian(cent))
        # jitter: w_k = sigma * k + c, fit per channel over orders with data (k >= 2)
        sig_ch = []
        for c in range(C):
            ok = np.isfinite(width[c]) & (ks >= 2)
            if ok.sum() >= 4:
                sig_ch.append(np.polyfit(ks[ok], width[c, ok], 1))
        sig_ch = np.array(sig_ch)
        n_k = int(np.isfinite(per_order).sum())
        print(
            f"{key:14s} {(a + a2) / fs:4.1f}-{(a + b2) / fs:4.1f} {T:4.1f} {s_mean:8.4f} {np.nanstd(per_order):7.4f} {np.nanstd(per_ch):7.4f}   {C - wind.sum()}/{C}     {n_k:2d}   "
            f"{np.median(sig_ch[:, 0]):7.3f}  {np.std(sig_ch[:, 0]):6.3f} {np.median(sig_ch[:, 1]):6.2f}   "
            + (
                "-"
                if not wind.any()
                else ",".join(f"ch{c}({snr2[c]:.0f})" for c in np.nonzero(wind)[0])
            )
            + f"   | k2 snr per ch: {np.round(snr2).astype(int).tolist()}"
        )


if __name__ == "__main__" and not (len(sys.argv) > 1 and sys.argv[1] == "detail"):
    keys = sys.argv[1:] or [
        f"motor_Motor{m}_{t}" for m in (1, 2, 3, 4) for t in (50, 60, 70, 80, 90)
    ]
    main(keys)


def detail(key):
    units = {v["key"]: v for v in json.load(open("/tmp/sr_dregon/units.json"))}
    prior = {
        r["uid"].split("__")[-1]: r["full"]["whitened"]["speed"]
        for r in json.load(open("/tmp/bretthorst_dregon.json"))
    }
    u = units[key]
    fs = u["fs"]
    x0 = np.load(u["audio"], mmap_mode="r")
    a, b = S.motor_on_span(np.asarray(x0), fs, S.Params())
    x = np.asarray(x0[:, a:b], dtype=np.float64)
    s0 = prior[key]
    a2, b2 = strict_span(x, fs, s0)
    x = x[:, a2:b2]
    n = x.shape[1]
    w = np.hanning(n)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    df = fs / nfft
    X = np.stack([np.abs(np.fft.rfft(x[c] * w, nfft)) ** 2 for c in range(x.shape[0])])
    print(
        f"{key}: T={n / fs:.1f}s df={df:.3f}Hz; per order: centroid/k - s0 (rev/s), rms half-width (Hz), HWHM (Hz), snr(dB), channel-median"
    )
    for k in (1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 30, 36, 40):
        f0 = k * s0
        hw = max(3.0, 0.01 * f0)
        lo, hi = int((f0 - 4 * hw) / df), int((f0 + 4 * hw) / df)
        f = np.arange(lo, hi) * df
        r = []
        for c in range(x.shape[0]):
            P = X[c, lo:hi]
            floor = np.percentile(P, 20)
            m = np.abs(f - f0) <= hw
            Pl = np.maximum(P - floor, 0)
            fc = np.sum(f[m] * Pl[m]) / np.sum(Pl[m])
            m2 = np.abs(f - fc) <= hw
            wr = np.sqrt(np.sum((f[m2] - fc) ** 2 * Pl[m2]) / np.sum(Pl[m2]))
            sm = np.convolve(P, np.ones(5) / 5, "same")
            pk = sm[m].max()
            half = (sm >= pk / 2) & (np.abs(f - fc) <= hw)
            hwhm = (f[half].max() - f[half].min()) / 2
            r.append((fc / k - s0, wr, hwhm, 10 * np.log10(pk / floor)))
        r = np.median(np.array(r), axis=0)
        print(
            f"  k{k:<3d} dc {r[0]:+7.4f}  rms {r[1]:5.2f}  hwhm {r[2]:5.2f}  (k*0.3 rev/s -> {0.3 * k:5.2f} Hz)  snr {r[3]:4.0f}"
        )


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "detail":
    detail(sys.argv[2])
