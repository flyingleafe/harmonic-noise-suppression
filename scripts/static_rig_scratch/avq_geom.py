"""Throwaway: AVQ S1 seq1-3 — fill speed tracks, demodulate harmonics along
them, inter-mic delays from the harmonic phases, TDOA localisation vs the rig
geometry, per-mic level signatures across recordings."""

import json
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import resample_poly

sys.path.insert(0, "src")
from experiments.static_rig import spectra as S  # noqa: E402

FS = 16000
K = 40
BLOCK = 0.25  # s boxcar envelope -> ENBW 4 Hz
C_SOUND = 343.0
FAM = {
    "S1_seq1": {"A": (51, 54.5), "B": (54.5, 56.5), "C": (56.5, 62), "D": (77, 80)},
    "S1_seq2": {"A": (79, 82.4), "B": (81.6, 85), "C": (85, 88.5), "D": (97, 98.5)},
    "S1_seq3": {"A": (88, 90), "B": (92.5, 94), "C": (95, 97), "D": (109.5, 112.5)},
}
G = json.load(open("results/static_rig/geometry/avq.json"))["configs"]["avq_S1"]
MIC = np.array(G["mic_pos"])
ROT = np.array(G["rotor_pos"])


def tracks(key):
    """Per family: (t_centre, speed) from the joint-search windows, missing
    windows filled by linear interpolation; returns dict fam -> (t, s)."""
    r = json.load(open(f"/tmp/sr_joint/AVQ__{key}.json"))
    a = r["span_s"][0]
    tc = np.array([(lf["t0"] + lf["t1"]) / 2 + a for lf in r["linked"]])
    out = {}
    for n, (lo, hi) in FAM[key].items():
        used = set()
        ts, ss = [], []
        for lf, t in zip(r["linked"], tc, strict=True):
            c = [s for s in sorted(lf["speeds"]) if lo <= s <= hi]
            if c:
                ts.append(t)
                ss.append(c[0])
        ts, ss = np.array(ts), np.array(ss)
        # drop outliers > 1 rev/s off the neighbours' line
        if ts.size > 2:
            keep = np.ones(ts.size, bool)
            for i in range(1, ts.size - 1):
                pred = np.interp(ts[i], ts[[i - 1, i + 1]], ss[[i - 1, i + 1]])
                keep[i] = abs(ss[i] - pred) < 1.0
            ts, ss = ts[keep], ss[keep]
        out[n] = (tc, np.interp(tc, ts, ss))
    return out, r["span_s"]


def demod(x, fs, t_knots, s_knots, k_max):
    """Boxcar envelopes (C, k_max, n_blocks) of harmonics 1..k_max along the
    trajectory, plus the same at control offsets +-2.5*ENBW."""
    n = x.shape[1]
    t = np.arange(n) / fs
    s = np.interp(t, t_knots, s_knots)
    ph = 2 * np.pi * np.cumsum(s) / fs  # shaft angle (rad)
    nb = int(BLOCK * fs)
    n_blk = n // nb
    env = np.zeros((x.shape[0], k_max, n_blk), complex)
    ctl = np.zeros((x.shape[0], k_max))
    enbw = 1.0 / BLOCK
    for k in range(1, k_max + 1):
        car = np.exp(-1j * k * ph)
        y = (x * car)[:, : n_blk * nb].reshape(x.shape[0], n_blk, nb).mean(axis=2)
        env[:, k - 1] = y
        for off in (-2.5 * enbw, 2.5 * enbw):
            c2 = car * np.exp(-2j * np.pi * off * t)
            yc = (x * c2)[:, : n_blk * nb].reshape(x.shape[0], n_blk, nb).mean(axis=2)
            ctl[:, k - 1] += 0.5 * np.mean(np.abs(yc) ** 2, axis=1)
    return env, ctl


def analyse(key):
    u = {v["key"]: v for v in json.load(open("/tmp/sr_avq/units.json"))}[key]
    fs0 = u["fs"]
    x = np.load(u["audio"], mmap_mode="r")
    a, b = S.motor_on_span(np.asarray(x), fs0, S.Params())
    x = resample_poly(np.asarray(x[:, a:b], dtype=np.float64), FS, fs0, axis=1)
    tr, span = tracks(key)
    res = {}
    for fam, (tk, sk) in tr.items():
        env, ctl = demod(x, FS, tk - a / fs0, sk, K)
        p_line = np.mean(np.abs(env) ** 2, axis=2)  # (C, K)
        snr = 10 * np.log10(np.maximum(p_line, 1e-30) / np.maximum(ctl, 1e-30))
        level = 10 * np.log10(np.maximum(p_line - ctl, 1e-30))
        # inter-mic cross terms vs the loudest mic per order
        ref = np.argmax(p_line, axis=0)
        phase = np.zeros((x.shape[0], K))
        coh = np.zeros((x.shape[0], K))
        for k in range(K):
            r = env[ref[k], k]
            cross = np.mean(env[:, k] * np.conj(r), axis=1)
            coh[:, k] = np.abs(cross) / np.sqrt(
                np.mean(np.abs(env[:, k]) ** 2, axis=1) * np.mean(np.abs(r) ** 2) + 1e-300
            )
            phase[:, k] = np.angle(cross)
        res[fam] = {
            "speed_mean": float(sk.mean()),
            "speed_range": [float(sk.min()), float(sk.max())],
            "level": level,
            "snr": snr,
            "phase": phase,
            "coh": coh,
            "ref": ref,
        }
    return res


def delays(r, snr_min=6.0, coh_min=0.6):
    """Per mic: delay (s) relative to mic 0 from the phase-vs-frequency slope
    over usable orders, unwrapped by continuity. Returns (tau (C,), n_orders, rms_rad)."""
    s = r["speed_mean"]
    C = r["level"].shape[0]
    # re-reference every order's phase to mic 0
    ph = r["phase"] - r["phase"][0][None, :]
    ok = (
        (r["snr"] >= snr_min)
        & (r["coh"] >= coh_min)
        & (r["snr"][0][None, :] >= snr_min)
        & (r["coh"][0][None, :] >= coh_min)
    )
    tau = np.full(C, np.nan)
    nord = np.zeros(C, int)
    rms = np.full(C, np.nan)
    for c in range(1, C):
        ks = np.nonzero(ok[c])[0] + 1
        if ks.size < 3:
            continue
        f = ks * s
        p = ph[c, ks - 1].copy()
        # unwrap by continuity in k using the running slope
        for i in range(1, ks.size):
            slope = (p[i - 1]) / f[i - 1] if i == 1 else np.polyfit(f[:i], p[:i], 1)[0]
            pred = slope * f[i]
            p[i] = pred + np.angle(np.exp(1j * (p[i] - pred)))
        slope, _ = np.polyfit(f, p, 1)
        tau[c] = -slope / (2 * np.pi)
        nord[c] = ks.size
        rms[c] = float(np.sqrt(np.mean(np.angle(np.exp(1j * (p - slope * f))) ** 2)))
    return tau, nord, rms


def localise(tau, w):
    """Source position from delays rel. mic 0 (least squares)."""
    ok = np.isfinite(tau) & (w > 0)
    ok[0] = False

    def resid(p):
        d = np.linalg.norm(MIC - p, axis=1)
        return w[ok] * ((d[ok] - d[0]) / C_SOUND - tau[ok])

    best = None
    for init in [ROT.mean(axis=0)] + list(ROT):
        sol = least_squares(resid, init)
        if best is None or sol.cost < best.cost:
            best = sol
    return best.x, float(np.sqrt(2 * best.cost / max(ok.sum(), 1)))


def main():
    out = {}
    for key in FAM:
        res = analyse(key)
        out[key] = res
        print(f"== {key}")
        for fam, r in res.items():
            tau, nord, rms = delays(r)
            pos, rres = localise(tau, nord.astype(float))
            dist = np.linalg.norm(ROT - pos, axis=1)
            pred_tau = (
                np.linalg.norm(MIC - ROT[np.argmin(dist)], axis=1)
                - np.linalg.norm(MIC[0] - ROT[np.argmin(dist)])
            ) / C_SOUND
            print(
                f"  {fam} s={r['speed_mean']:.2f} [{r['speed_range'][0]:.1f}-{r['speed_range'][1]:.1f}]"
                f" orders>=6dB(med mic): {int((np.median(r['snr'], axis=0) >= 6).sum())}"
                f" | tau(ms) "
                + " ".join("  nan" if not np.isfinite(t) else f"{1e3 * t:5.2f}" for t in tau[1:])
                + " | n_ord "
                + " ".join(f"{n:2d}" for n in nord[1:])
                + f" | fit rms {np.nanmean(rms):.2f} rad"
            )
            print(
                f"     localised ({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) resid {1e3 * rres:.3f} ms;"
                f" dist to rotors FL/FR/RL/RR: "
                + " ".join(f"{d:.3f}" for d in dist)
                + f" -> {G['rotor_labels'][int(np.argmin(dist))]}"
                + " | predicted tau(ms) "
                + " ".join(f"{1e3 * t:5.2f}" for t in pred_tau[1:])
            )
    np.save("/tmp/avq_geom_res.npy", out, allow_pickle=True)


if __name__ == "__main__":
    main()
