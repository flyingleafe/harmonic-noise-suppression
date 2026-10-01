"""Throwaway: refine each rotor's speed track at 0.5 s resolution from its
strongest high-order line, then measure every order up to 10 kHz along the
refined track (demod, ENBW 3 Hz) against a 20th-percentile floor."""

import json
import sys

import numpy as np

sys.path.insert(0, "src")
from experiments.static_rig import spectra as S  # noqa: E402

BLK = 0.5  # s
F_MAX = 10000.0
K_MAX = 130
SEARCH = 0.004  # +-0.4 % of k*s around the coarse track


def coarse(key):
    r = json.load(open(f"/tmp/sr_joint/AVQ__{key}.json"))
    d = np.load(f"/tmp/avq_amp_{key}.npz")
    tc = np.array([(lf["t0"] + lf["t1"]) / 2 for lf in r["linked"]])
    wins = [(lf["t0"], lf["t1"]) for lf in r["linked"]]
    return tc, {f: d[f"spd_{f}"] for f in "ABCD"}, wins


def block_spectra(x, fs):
    """Hann periodogram of every BLK block, padded x8: (n_blk, C, F)."""
    nb = int(BLK * fs)
    n_blk = x.shape[1] // nb
    w = np.hanning(nb)
    nfft = 1 << int(np.ceil(np.log2(8 * nb)))
    kp = int(F_MAX * 1.02 / (fs / nfft))
    out = np.zeros((n_blk, x.shape[0], kp), np.float32)
    for i in range(n_blk):
        out[i] = np.abs(np.fft.rfft(x[:, i * nb : (i + 1) * nb] * w, nfft, axis=1)[:, :kp]) ** 2
    return out, fs / nfft


def _peak(Pm, dfp, f, search):
    c = int(round(f / dfp))
    h = max(3, int(round(search * f / dfp)))
    loc = Pm[c - h : c + h + 1]
    j = int(np.argmax(loc))
    if j == 0 or j == loc.size - 1:
        return None
    y0, y1, y2 = np.log(loc[j - 1 : j + 2] + 1e-30)
    dj = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
    fl = np.percentile(Pm[max(0, c - int(60 / dfp)) : c + int(60 / dfp)], 20)
    return (c - h + j + dj) * dfp, 10 * np.log10(loc[j] / fl)


LOW = (2, 4, 6, 10)
HIGH = (42, 36, 30, 28, 24, 20, 16)
TOL_AGREE = 0.15  # rev/s
SEP_HZ = 12.0  # a reference line must be this far from every other rotor's harmonic


def low_pass(P, dfp, tc, s0, n_blk):
    tb = (np.arange(n_blk) + 0.5) * BLK
    sc = np.interp(tb, tc, s0)
    s1 = sc.copy()
    src = np.zeros(n_blk, int)
    for i in range(n_blk):
        Pm = P[i].sum(axis=0)
        low = None
        for k in LOW:
            r = _peak(Pm, dfp, k * sc[i], SEARCH)
            if r is None or r[1] < 8:
                continue
            if low is None or r[1] > low[1]:
                low = (r[0] / k, r[1], k)
        if low is not None:
            s1[i], src[i] = low[0], low[2]
    return tb, s1, src


def refine(P, dfp, f, lows, n_blk):
    """Replace the low-order speed of rotor ``f`` by the highest usable order's
    estimate: line >= 8 dB, >= SEP_HZ from every other rotor's harmonics,
    agreeing with the low-order speed within TOL_AGREE."""
    tb, s_low, src_low = lows[f]
    s1 = s_low.copy()
    src = src_low.copy()
    diff = np.full(n_blk, np.nan)
    others = [lows[g][1] for g in lows if g != f]
    for i in range(n_blk):
        if src_low[i] == 0:
            continue
        Pm = P[i].sum(axis=0)
        for k in HIGH:
            fq = k * s_low[i]
            if fq > F_MAX:
                continue
            if min(abs(fq - np.round(fq / so[i]) * so[i]) for so in others) < SEP_HZ:
                continue
            r = _peak(Pm, dfp, fq, 0.0025)
            if r is None or r[1] < 8:
                continue
            diff[i] = r[0] / k - s_low[i]
            if abs(diff[i]) <= TOL_AGREE:
                s1[i], src[i] = r[0] / k, k
            break
    s_med = s1.copy()
    for i in range(1, n_blk - 1):
        s_med[i] = np.median(s1[i - 1 : i + 2])
    ok = np.isfinite(diff)
    print(
        f"    high vs low-order speed: |diff| median {np.nanmedian(np.abs(diff)):.3f}, 90% {np.nanpercentile(np.abs(diff), 90):.3f} rev/s over {ok.sum()} blocks"
    )
    return tb, s_med, src, diff


def orders_along(x, fs, tb, s, n_blk):
    """Line ms power (n_blk, K, C) of every order, demod with Hann BLK blocks."""
    nb = int(BLK * fs)
    t = np.arange(n_blk * nb) / fs
    st = np.interp(t, tb, s)
    ph = 2 * np.pi * np.cumsum(st) / fs
    w = np.hanning(nb)
    w /= w.sum()
    xx = x[:, : n_blk * nb]
    out = np.zeros((n_blk, K_MAX, x.shape[0]))
    for k in range(1, K_MAX + 1):
        y = (xx * np.exp(-1j * k * ph)).reshape(x.shape[0], n_blk, nb)
        e = (y * w).sum(axis=2)  # (C, n_blk)
        out[:, k - 1, :] = (2 * np.abs(e) ** 2).T
    return out


def main(key):
    u = {v["key"]: v for v in json.load(open("/tmp/sr_avq/units.json"))}[key]
    fs = u["fs"]
    x0 = np.load(u["audio"], mmap_mode="r")
    a, b = S.motor_on_span(np.asarray(x0), fs, S.Params())
    x = np.asarray(x0[:, a:b], dtype=np.float64)
    tc, s0, wins = coarse(key)
    P, dfp = block_spectra(x, fs)
    n_blk = P.shape[0]
    enbw_bins = 1.5 / BLK / dfp  # Hann ENBW in padded bins
    res = {}
    print(f"== {key}: {n_blk} blocks of {BLK} s, fs {fs}")
    lows = {f: low_pass(P, dfp, tc, s0[f], n_blk) for f in "ABCD"}
    for f in "ABCD":
        tb, s1, src, snr_ref = refine(P, dfp, f, lows, n_blk)
        sc = np.interp(tb, tc, s0[f])
        ok = src > 0
        print(
            f"  rotor {f}: reference found in {ok.sum()}/{n_blk} blocks; orders used "
            + ", ".join(f"k{k}:{(src == k).sum()}" for k in LOW + HIGH if (src == k).any())
            + f"; refined-coarse: median {np.median(np.abs(s1[ok] - sc[ok])):.3f} max {np.abs(s1[ok] - sc[ok]).max():.3f} rev/s;"
            f" within-window sd of refined speed: {np.mean([np.std(s1[(tb >= t0) & (tb < t1) & ok]) for t0, t1 in wins if ((tb >= t0) & (tb < t1) & ok).sum() > 2]):.3f} rev/s"
        )
        pw = orders_along(x, fs, tb, s1, n_blk)  # (n_blk, K, C)
        # floor at each order per block: 20th percentile over +-60 Hz of the block periodogram, times ENBW
        kk = np.arange(1, K_MAX + 1)
        fl = np.full_like(pw, np.nan)
        for i in range(n_blk):
            fk = kk * s1[i]
            for ki, fq in enumerate(fk):
                if fq > F_MAX:
                    break
                c = int(round(fq / dfp))
                h = int(60 / dfp)
                q = np.percentile(P[i][:, max(0, c - h) : c + h], 20, axis=1)
                fl[i, ki] = (
                    8.0 * q / (0.2231 * int(BLK * fs) ** 2)
                )  # demod noise power: 20th pct -> mean, Hann-mean gain
        snr = 10 * np.log10(np.maximum(pw / fl - 1, 1e-3))  # line / noise-in-ENBW
        for i in range(n_blk):
            fk = kk * s1[i]
            for g in "ABCD":
                if g == f:
                    continue
                so = lows[g][1][i]
                snr[i, np.abs(fk - np.round(fk / so) * so) < 3.0, :] = np.nan
        med = np.nanmedian(snr, axis=(0, 2))  # per order, median over blocks and mics
        frac = np.nanmean(snr > 6, axis=(0, 2))
        strong = [k + 1 for k in range(K_MAX) if frac[k] >= 0.5]
        print(f"    orders with SNR>6 dB (ENBW 3 Hz) in >=50% of (block, mic): {strong}")
        top = np.argsort(med)[::-1][:12]
        print(
            "    top orders by median SNR: "
            + ", ".join(f"k{k + 1}:{med[k]:.1f}" for k in sorted(top))
        )
        res[f] = dict(tb=tb, s1=s1, src=src, pw=pw, fl=fl)
    np.savez(
        f"/tmp/avq_refine_{key}.npz",
        **{f"{f}_{k}": v for f, r in res.items() for k, v in r.items()},
    )


if __name__ == "__main__":
    for key in sys.argv[1:]:
        main(key)
