"""Throwaway: AVQ S1 seq1-3 amplitude profiles.

Two estimators of the mean-square power of every (rotor, order, mic) cell:
  band  — per joint-search window: Hann periodogram band (+-tol around k*s)
          minus running-median floor  (Bretthorst line power of a wander-
          broadened line; no track needed inside the window)
  demod — heterodyne along the piecewise-linear speed track, Hann-smoothed
          envelope (ENBW 1-4 Hz by order), power minus floor*ENBW
Then rebuild the signal from (track, envelopes) and from (track, constant
amplitude profile) and compare Welch spectra with the recording."""

import json
import sys

import numpy as np
from scipy.signal import resample_poly, welch

sys.path.insert(0, "src")
from experiments.static_rig import spectra as S  # noqa: E402
from experiments.static_rig.spectra import _floor  # noqa: E402

FS = 16000
K = 40
F_MAX = 3000.0
PAD = 4
DRIFT = 1e-3
FLOOR_HZ = 10.0
DEC = 100  # envelope sample rate (Hz)
FAM = {
    "S1_seq1": {"A": (51, 54.5), "B": (54.5, 56.5), "C": (56.5, 62), "D": (77, 80)},
    "S1_seq2": {"A": (79, 82.4), "B": (81.6, 85), "C": (85, 88.5), "D": (97, 98.5)},
    "S1_seq3": {"A": (88, 90), "B": (92.5, 94), "C": (95, 97), "D": (109.5, 112.5)},
}


def enbw(k):
    return max(1.0, min(4.0, 0.5 * k))


def tracks(key):
    r = json.load(open(f"/tmp/sr_joint/AVQ__{key}.json"))
    tc = np.array([(lf["t0"] + lf["t1"]) / 2 for lf in r["linked"]])
    out, found = {}, {}
    for n, (lo, hi) in FAM[key].items():
        ts, ss = [], []
        for lf, t in zip(r["linked"], tc, strict=True):
            c = [s for s in sorted(lf["speeds"]) if lo <= s <= hi]
            if c:
                ts.append(t)
                ss.append(c[0])
        ts, ss = np.array(ts), np.array(ss)
        if ts.size > 2:
            keep = np.ones(ts.size, bool)
            for i in range(1, ts.size - 1):
                pred = np.interp(ts[i], ts[[i - 1, i + 1]], ss[[i - 1, i + 1]])
                keep[i] = abs(ss[i] - pred) < 1.0
            ts, ss = ts[keep], ss[keep]
        out[n] = np.interp(tc, ts, ss)
        found[n] = np.isin(tc, ts)
    wins = [(lf["t0"], lf["t1"]) for lf in r["linked"]]
    return tc, out, found, wins


def band_estimates(x, fs, wins, spd):
    """spd: fam -> speeds per window. Returns ms power (W, R, K, C), floor ms
    in band (W, R, K, C), merged flag (W, R, K)."""
    fams = list(spd)
    W, R, C = len(wins), len(fams), x.shape[0]
    ms = np.full((W, R, K, C), np.nan)
    fl_ms = np.full((W, R, K, C), np.nan)
    merged = np.zeros((W, R, K), bool)
    for wi, (t0, t1) in enumerate(wins):
        seg = x[:, int(t0 * fs) : int(t1 * fs)]
        n = seg.shape[1]
        T = n / fs
        w = np.hanning(n)
        nfft = 1 << int(np.ceil(np.log2(PAD * n)))
        dfp = fs / nfft
        X = np.fft.rfft(seg * w, nfft, axis=1)[:, : int(F_MAX / dfp) + 2]
        P = np.abs(X) ** 2
        fl = 10 ** (_floor(10 * np.log10(P + 1e-30), max(2, round(0.5 * FLOOR_HZ / dfp))) / 10)
        fl = fl / np.log(2)  # median -> mean of the exponential floor
        norm = 2.0 / (n * np.sum(w**2))  # band sum / PAD -> mean-square power
        cz = np.concatenate([np.zeros((C, 1)), np.cumsum(P, axis=1)], axis=1)
        cf = np.concatenate([np.zeros((C, 1)), np.cumsum(fl, axis=1)], axis=1)
        allf = []
        for ri, f in enumerate(fams):
            s = spd[f][wi]
            k = np.arange(1, K + 1)
            fk = k * s
            tol = 0.5 * np.maximum(1.5 / T, fk * DRIFT * T)
            allf.append((fk, tol))
        # intermods (order <= 2)
        im = []
        kk = np.arange(1, 3)
        for i in range(R):
            for j in range(i + 1, R):
                fi, fj = kk * spd[fams[i]][wi], kk * spd[fams[j]][wi]
                im += list((fi[:, None] + fj[None, :]).ravel()) + list(
                    np.abs(fi[:, None] - fj[None, :]).ravel()
                )
        im = np.array(im)
        for ri, (fk, tol) in enumerate(allf):
            for rj, (fk2, tol2) in enumerate(allf):
                if rj == ri:
                    continue
                d = np.abs(fk[:, None] - fk2[None, :]) < tol[:, None] + tol2[None, :]
                merged[wi, ri] |= d.any(axis=1)
            merged[wi, ri] |= (
                np.abs(fk[:, None] - im[None, :]) < tol[:, None] + 0.5 * tol[:, None]
            ).any(axis=1)
            ok = fk < F_MAX - 20
            b = np.rint(fk[ok] / dfp).astype(int)
            h = np.maximum(1, np.rint(tol[ok] / dfp)).astype(int)
            sp = (cz[:, b + h + 1] - cz[:, b - h]) / PAD
            sf = (cf[:, b + h + 1] - cf[:, b - h]) / PAD
            ms[wi, ri, ok, :] = ((sp - sf) * norm).T
            fl_ms[wi, ri, ok, :] = (sf * norm).T
    return ms, fl_ms, merged


def demod_env(x, fs, tc, s_knots):
    """Complex envelopes (C, K, n_env) at DEC Hz along the track (Hann smoothed,
    ENBW by order) and the shaft angle at full rate."""
    n = x.shape[1]
    t = np.arange(n) / fs
    s = np.interp(t, tc, s_knots)
    ph = 2 * np.pi * np.cumsum(s) / fs
    nb = fs // DEC
    n_env = n // nb
    env = np.zeros((x.shape[0], K, n_env), complex)
    for k in range(1, K + 1):
        y = (
            (x[:, : n_env * nb] * np.exp(-1j * k * ph[: n_env * nb]))
            .reshape(x.shape[0], n_env, nb)
            .mean(axis=2)
        )
        L = int(round(1.5 / enbw(k) * DEC)) | 1
        hw = np.hanning(L + 2)[1:-1]
        hw /= hw.sum()
        for c in range(x.shape[0]):
            env[c, k - 1] = np.convolve(y[c], hw, mode="same")
    return env, ph


def remod(env, ph, fs):
    n = ph.size
    nb = fs // DEC
    n_env = env.shape[2]
    t_env = (np.arange(n_env) + 0.5) * nb / fs
    t = np.arange(n) / fs
    out = np.zeros((env.shape[0], n))
    for k in range(1, K + 1):
        car = np.exp(1j * k * ph)
        for c in range(env.shape[0]):
            e = np.interp(t, t_env, env[c, k - 1].real) + 1j * np.interp(
                t, t_env, env[c, k - 1].imag
            )
            out[c] += 2 * np.real(e * car)
    return out


def analyse(key):
    u = {v["key"]: v for v in json.load(open("/tmp/sr_avq/units.json"))}[key]
    fs0 = u["fs"]
    x0 = np.load(u["audio"], mmap_mode="r")
    a, b = S.motor_on_span(np.asarray(x0), fs0, S.Params())
    x = resample_poly(np.asarray(x0[:, a:b], dtype=np.float64), FS, fs0, axis=1)
    tc, spd, found, wins = tracks(key)
    fams = list(spd)
    ms_b, fl_b, merged = band_estimates(x, FS, wins, spd)
    # demod along the track; per-window power to compare cell by cell
    ms_d = np.full_like(ms_b, np.nan)
    synth = np.zeros_like(x)
    param = np.zeros_like(x)
    rng = np.random.default_rng(0)
    prof = {}
    for ri, f in enumerate(fams):
        env, ph = demod_env(x, FS, tc, spd[f])
        p_env = 2 * np.abs(env) ** 2  # ms power of the line (A^2/2 = 2|A/2 e^{i th}|^2)
        t_env = (np.arange(env.shape[2]) + 0.5) / DEC
        for wi, (t0, t1) in enumerate(wins):
            sel = (t_env >= t0) & (t_env < t1)
            fl = fl_b[wi, ri]  # floor ms in the band (K, C)
            # floor in the envelope's band: floor density * enbw
            T = t1 - t0
            kk = np.arange(1, K + 1)
            bw_band = 2 * 0.5 * np.maximum(1.5 / T, kk * spd[f][wi] * DRIFT * T)
            fl_env = fl * (np.array([enbw(k) for k in kk]) / bw_band)[:, None]
            ms_d[wi, ri] = p_env[:, :, sel].mean(axis=2).T - fl_env
        synth += remod(env, ph, FS)
        # parametric: constant amplitude from the mean (over found windows) band power
        amp = np.sqrt(
            np.nan_to_num(
                np.maximum(
                    np.nanmean(np.where(found[f][:, None, None], ms_b[:, ri], np.nan), axis=0), 0
                )
            )
        )  # (K, C)
        prof[f] = amp
        n = ph.size
        for k in range(1, K + 1):
            th = rng.uniform(0, 2 * np.pi, x.shape[0])
            param += np.sqrt(2) * amp[k - 1][:, None] * np.cos(k * ph[None, :] + th[:, None])
    return dict(
        x=x,
        synth=synth,
        param=param,
        ms_b=ms_b,
        fl_b=fl_b,
        ms_d=ms_d,
        merged=merged,
        found=found,
        spd=spd,
        wins=wins,
        fams=fams,
        prof=prof,
    )


def report(key, r):
    x, synth, param = r["x"], r["synth"], r["param"]
    ms_b, fl_b, ms_d, merged = r["ms_b"], r["fl_b"], r["ms_d"], r["merged"]
    fams = r["fams"]
    print(f"\n== {key}  windows {len(r['wins'])}, 8 mics, orders 1..{K}")
    # detectability: band excess > 5 sigma of the floor sum, per cell
    # sigma of floor sum ~ fl_ms / sqrt(n_nat); n_nat = bw*T
    T = np.array([t1 - t0 for t0, t1 in r["wins"]])
    kk = np.arange(1, K + 1)
    tab = {}
    for ri, f in enumerate(fams):
        fnd = r["found"][f]
        s = r["spd"][f]
        bw = 2 * 0.5 * np.maximum(1.5 / T[:, None], kk[None, :] * s[:, None] * DRIFT * T[:, None])
        n_nat = bw * T[:, None]
        sig = fl_b[:, ri] / np.sqrt(n_nat)[:, :, None]
        det = (ms_b[:, ri] > 5 * sig) & fnd[:, None, None] & ~merged[:, ri][:, :, None]
        snr_db = 10 * np.log10(np.maximum(ms_b[:, ri] / fl_b[:, ri], 1e-3))  # line vs floor-in-band
        det_any = (ms_b[:, ri] > 5 * sig) & fnd[:, None, None]
        frac_any = det_any.sum(axis=(0, 2)) / max(1, fnd.sum() * x.shape[0])
        # per order: fraction of (found window, mic) cells detected; median SNR
        frac = det.sum(axis=(0, 2)) / max(1, fnd.sum() * x.shape[0])
        med_snr = np.nanmedian(np.where(fnd[:, None, None], snr_db, np.nan), axis=(0, 2))
        # demod vs band agreement on detected cells, per order
        dd = 10 * np.log10(np.maximum(ms_d[:, ri], 1e-30)) - 10 * np.log10(
            np.maximum(ms_b[:, ri], 1e-30)
        )
        agree = np.array(
            [np.nanmedian(dd[:, k][det[:, k]]) if det[:, k].any() else np.nan for k in range(K)]
        )
        # window scatter of the line level (dB) per order, median over mics
        lv = 10 * np.log10(np.maximum(ms_b[:, ri], 1e-30))
        scat = np.array(
            [
                np.nanmedian(
                    [
                        np.std(lv[det[:, k, c], k, c]) if det[:, k, c].sum() > 2 else np.nan
                        for c in range(x.shape[0])
                    ]
                )
                for k in range(K)
            ]
        )
        last = max([k + 1 for k in range(K) if frac[k] >= 0.5], default=0)
        n_det = int((frac >= 0.5).sum())
        mg = merged[:, ri][fnd].mean(axis=0)
        print(
            f"  rotor {f}  mean {s.mean():6.2f} rev/s found {fnd.sum()}/{fnd.size} win | orders with >=50% cells detected: {n_det}, highest {last}; merged-cell frac per order mean {mg.mean():.2f}"
        )
        hdr = (
            "    k  : "
            + " ".join(f"{k:5d}" for k in range(1, 25))
            + "  ...  "
            + " ".join(f"{k:5d}" for k in (30, 35, 40))
        )
        sel = list(range(24)) + [29, 34, 39]
        print(hdr)
        print(
            "    det : "
            + " ".join(f"{frac[k]:5.2f}" for k in sel[:24])
            + "       "
            + " ".join(f"{frac[k]:5.2f}" for k in sel[24:])
        )
        print(
            "    any : "
            + " ".join(f"{frac_any[k]:5.2f}" for k in sel[:24])
            + "       "
            + " ".join(f"{frac_any[k]:5.2f}" for k in sel[24:])
        )
        print(
            "    snr : "
            + " ".join(f"{med_snr[k]:5.1f}" for k in sel[:24])
            + "       "
            + " ".join(f"{med_snr[k]:5.1f}" for k in sel[24:])
        )
        print(
            "    d-b : "
            + " ".join(f"{agree[k]:5.1f}" for k in sel[:24])
            + "       "
            + " ".join(f"{agree[k]:5.1f}" for k in sel[24:])
        )
        print(
            "    sd  : "
            + " ".join(f"{scat[k]:5.1f}" for k in sel[:24])
            + "       "
            + " ".join(f"{scat[k]:5.1f}" for k in sel[24:])
        )
        print(
            "    mrg : "
            + " ".join(f"{mg[k]:5.2f}" for k in sel[:24])
            + "       "
            + " ".join(f"{mg[k]:5.2f}" for k in sel[24:])
        )
        tab[f] = dict(frac=frac, snr=med_snr, agree=agree, scat=scat, merged=mg)
    # spectra
    nper = FS // 2  # 2 Hz resolution
    fr, Pxx = welch(x, FS, nperseg=nper, axis=1)
    _, Pss = welch(synth, FS, nperseg=nper, axis=1)
    _, Prr = welch(x - synth, FS, nperseg=nper, axis=1)
    _, Ppp = welch(param, FS, nperseg=nper, axis=1)
    band = fr <= F_MAX
    tot = Pxx[:, band].sum(axis=1)
    fl_w0 = 10 ** (_floor(10 * np.log10(Pxx + 1e-30), 5) / 10)
    print(
        f"  floor share of 0-{F_MAX:.0f} Hz power: {100 * (fl_w0[:, band].sum(1) / tot).mean():.1f}%"
    )
    print(
        f"  power 0-{F_MAX:.0f} Hz: envelope-rebuild explains {100 * (1 - Prr[:, band].sum(1) / tot).mean():.1f}% (mics {np.round(100 * (1 - Prr[:, band].sum(1) / tot)).astype(int)}),"
        f" synth/real {100 * (Pss[:, band].sum(1) / tot).mean():.1f}%, parametric/real {100 * (Ppp[:, band].sum(1) / tot).mean():.1f}%"
    )
    # per-line peaks: real vs parametric vs residual at k*mean speed (mic 0 and mic-median)
    fl_w = 10 ** (_floor(10 * np.log10(Pxx + 1e-30), 5) / 10)
    for ri, f in enumerate(fams):
        s = r["spd"][f].mean()
        rows = []
        for k in (1, 2, 4, 6, 8, 10, 14, 20, 30):
            fk = k * s
            if fk > F_MAX:
                continue
            h = max(
                2, int(np.ceil((0.5 * fk * DRIFT * 60) / 2))
            )  # half-bins around k*s (track spread over the record)
            b = int(round(fk / 2))
            sl = slice(b - h, b + h + 1)
            pr = Pxx[:, sl].max(1) / fl_w[:, b]
            pp = Ppp[:, sl].max(1) / fl_w[:, b]
            ps = Pss[:, sl].max(1) / fl_w[:, b]
            pres = Prr[:, sl].max(1) / fl_w[:, b]
            rows.append(
                f"k{k:<2d} real {10 * np.log10(np.median(pr)):5.1f} env {10 * np.log10(np.median(ps)):5.1f} par {10 * np.log10(np.median(pp)):5.1f} resid {10 * np.log10(np.median(pres)):5.1f}"
            )
        print(f"  rotor {f} peak/floor dB (median over mics; Welch 2 Hz):  " + " | ".join(rows))
    # figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(16, 11))
    c = 0
    for a_, (lo, hi) in zip(ax, [(0, 600), (600, 1500), (1500, 3000)], strict=True):
        m = (fr >= lo) & (fr <= hi)
        a_.plot(fr[m], 10 * np.log10(Pxx[c, m]), lw=0.7, label="recording")
        a_.plot(
            fr[m], 10 * np.log10(Pss[c, m] + 1e-30), lw=0.7, label="rebuild from track+envelopes"
        )
        a_.plot(
            fr[m],
            10 * np.log10(Ppp[c, m] + fl_w[c, m]),
            lw=0.7,
            label="parametric (const. amplitude profile) + floor",
        )
        a_.plot(
            fr[m],
            10 * np.log10(Prr[c, m]),
            lw=0.7,
            label="residual (recording - rebuild)",
            alpha=0.7,
        )
        for ri, f in enumerate(fams):
            s = r["spd"][f].mean()
            for k in range(1, K + 1):
                if lo <= k * s <= hi:
                    a_.axvline(k * s, color=f"C{ri + 4}", lw=0.4, alpha=0.4)
        a_.set_xlim(lo, hi)
        a_.set_ylabel("PSD dB (mic 1)")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title(f"AVQ {key}: Welch 2 Hz; vertical lines = harmonics of the four mean speeds")
    ax[2].set_xlabel("Hz")
    fig.tight_layout()
    fig.savefig(f"/tmp/avq_recon_{key}.png", dpi=110)
    return tab


if __name__ == "__main__":
    keys = sys.argv[1:] or list(FAM)
    out = {}
    for key in keys:
        r = analyse(key)
        out[key] = report(key, r)
        np.savez(
            f"/tmp/avq_amp_{key}.npz",
            ms_b=r["ms_b"],
            fl_b=r["fl_b"],
            ms_d=r["ms_d"],
            merged=r["merged"],
            **{f"spd_{f}": v for f, v in r["spd"].items()},
            **{f"found_{f}": v for f, v in r["found"].items()},
            **{f"prof_{f}": v for f, v in r["prof"].items()},
        )
