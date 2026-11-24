#!/usr/bin/env python3
"""Measured WAV <-> native-DAT-tick alignment of the November 2026 M100 sessions.

FLY013 (``10_Nov/ZOOM0002.wav``), FLY018 (``14_Nov/ZOOM0003.wav``) and FLY022
(``15_Nov/ZOOM0004.wav``) were recorded by a ZOOM recorder with no shared
clock. The telemetry is the DAT's own motor packets (``motor_dat.read_motor_dat``)
on the native flight-controller tick; nothing here reads file lengths,
timestamps or the DatCon row clock. The output map is

    t_wav [s] = A + B * tick / 84e6          (tick: unwrapped native int64)

**The measure** (the ``coarse_align.py`` comb, generalised). A streamed
16 kHz mono STFT, log-magnitude z-scored per frame over ``band``; for a
candidate telemetry lag the native RPM (interpolated only between packets
<= 0.2 s apart — holes are never bridged) predicts lines at
``order * rps * rps_scale`` and the score is the mean z at those bins over the
frames where the mean rotor speed exceeds ``thr``. Orders are acoustic, not
assumed: FLY018/FLY022 fly two-blade props (blade-pass orders 2..20); FLY013 is
a bench run of four bare motors (photo), whose strong family is the 7x
electrical order (7..56) — its even blade comb gives no usable peak.

**Passes.**

  1. ``global``  one lag over the whole feasible range (0.1 s step, every 4th
     frame, +-1 bin) — the unambiguity check.
  2. ``scale``   a joint (lag, rps_scale) grid around it — a nuisance
     parameter for the high orders, reported, NOT a calibration constant.
  3. ``windows`` the lag re-measured on consecutive 20 s windows tiling the
     overlap of audio and motor activity (2 ms step, parabolic peak).
     ``begin``/``end`` are the first/last windows. Three fits of
     ``t_wav = A + B * u`` (u = tick/84e6): through begin+end only (the
     interior windows are then an independent validation), through all
     accepted windows, and a constant offset (B = 1) for comparison.
  4. ``events``  broadband (100-4000 Hz) RMS max-slope at each motor start
     vs the first non-zero native packet — an order-free cross-check.

    python scripts/michaels_calib/new_motor_align.py               # all three
    python scripts/michaels_calib/new_motor_align.py --rid FLY013

Runs on one core in ~2-4 min per recording (``nice -n 19 taskset -c 0``).
Writes ``results/new_motor_data_alignment/{<rid>.json,<rid>_windows.csv,
<rid>.png,maps.json}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import scipy.signal as ss
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from data_processing.sources.motor_dat import ROTORS, datcon_ticks, read_motor_dat  # noqa: E402

#: the reporting basis of the map; NOT a claim about the true tick rate.
TICK_HZ = 84e6
#: DatCon's assumed rate behind ``Clock:offsetTime``.
DATCON_HZ = 85e6
SR, N_FFT, HOP = 16000, 2048, 512
HOLE_S = 0.2
WINDOW_S = 20.0

SESSIONS: dict[str, dict[str, Any]] = {
    "FLY013": {
        "dir": "10_Nov",
        "wav": "ZOOM0002.wav",
        "dat": "FLY013.dat",
        "csv": "FLY013.csv",
        "orders": [7 * k for k in range(1, 9)],
        "band": (100.0, 7900.0),
        "thr": 15.0,
        "rig": "bench, four bare motors (no props)",
    },
    "FLY018": {
        "dir": "14_Nov",
        "wav": "ZOOM0003.wav",
        "dat": "FLY018.dat",
        "csv": "FLY018.csv",
        "orders": [2 * k for k in range(1, 11)],
        "band": (100.0, 4000.0),
        "thr": 45.0,
        "rig": "flight, two-blade props",
    },
    "FLY022": {
        "dir": "15_Nov",
        "wav": "ZOOM0004.wav",
        "dat": "FLY022.dat",
        "csv": "FLY022.csv",
        "orders": [2 * k for k in range(1, 11)],
        "band": (100.0, 4000.0),
        "thr": 45.0,
        "rig": "flight, two-blade props",
    },
}


# ─────────────────────────────────────────────────────────────── inputs
def stream_spectrogram(
    wav: Path, band: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, float, int, float]:
    """``(z (T, F_band) float16, frame centres (T,), df, lo_bin, duration)``.

    Streamed: 48 kHz blocks are channel-averaged, decimated x3 with a 0.1 s
    overlap margin and framed from a carried buffer, so no whole-file array
    of the input or of the full spectrum ever exists.
    """
    info = sf.info(str(wav))
    down = info.samplerate // SR
    if down * SR != info.samplerate:
        raise ValueError(f"{wav}: {info.samplerate} Hz is not a multiple of {SR}")
    margin = int(0.1 * info.samplerate) // down * down
    block = 30 * info.samplerate // down * down
    df = SR / N_FFT
    lo, hi = int(round(band[0] / df)), int(round(band[1] / df))
    win = ss.get_window("hann", N_FFT).astype(np.float32)
    frames: list[np.ndarray] = []
    buf = np.zeros(0, np.float32)
    with sf.SoundFile(str(wav)) as f:
        n = f.frames
        for start in range(0, n, block):
            a, b = max(0, start - margin), min(n, start + block + margin)
            f.seek(a)
            y = f.read(b - a, dtype="float32", always_2d=True).mean(1)
            y = ss.resample_poly(y, 1, down).astype(np.float32)
            y = y[(start - a) // down : len(y) - (b - min(n, start + block)) // down]
            buf = np.concatenate([buf, y])
            m = (len(buf) - N_FFT) // HOP + 1
            if m <= 0:
                continue
            idx = np.arange(m)[:, None] * HOP + np.arange(N_FFT)[None, :]
            spec = np.log(np.abs(np.fft.rfft(buf[idx] * win, axis=1)[:, lo:hi]) + 1e-10)
            z = (spec - spec.mean(1, keepdims=True)) / (spec.std(1, keepdims=True) + 1e-9)
            frames.append(z.astype(np.float16))
            buf = buf[m * HOP :]
    z = np.concatenate(frames)
    t = np.arange(len(z)) * HOP / SR + N_FFT / 2 / SR
    return np.ascontiguousarray(z), t, df, lo, n / info.samplerate


def native_motors(dat: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(tick int64, u = tick/84e6, rps (4, M))`` sorted by tick, rps >= 0."""
    df = read_motor_dat(dat)["Motor"]
    df = df.sort_values("tick", kind="stable").drop_duplicates("tick")
    tick = df["tick"].to_numpy(np.int64)
    rpm = df[list(ROTORS)].to_numpy(np.float64).T
    return tick, tick / TICK_HZ, np.clip(rpm, 0.0, None) / 60.0


def broadband_db(wav: Path, hop_s: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """10 ms RMS (dB) of the 100-4000 Hz band, streamed."""
    info = sf.info(str(wav))
    hop = int(hop_s * info.samplerate)
    sos = cast(
        np.ndarray,
        ss.butter(4, (100.0, 4000.0), btype="band", fs=info.samplerate, output="sos"),
    )
    zi = np.zeros((sos.shape[0], 2))
    out = []
    for blk in sf.blocks(
        str(wav), blocksize=60 * info.samplerate // hop * hop, dtype="float32", always_2d=True
    ):
        y, zi = ss.sosfilt(sos, blk.mean(1), zi=zi)
        k = len(y) // hop * hop
        out.append(np.sqrt((y[:k].reshape(-1, hop) ** 2).mean(1)))
    e = np.concatenate(out)
    return np.arange(len(e)) * hop_s + hop_s / 2, 20 * np.log10(e + 1e-9)


# ─────────────────────────────────────────────────────────────── the measure
class Comb:
    def __init__(self, z, tf, df, lo, u, rps, orders, thr):
        self.zt, self.nb = z.reshape(-1), z.shape[1]
        self.tf, self.df, self.lo = tf, df, lo
        self.u, self.rps = u, rps
        self.gap = np.r_[np.diff(u), np.inf]
        self.orders = np.asarray(orders, float)
        self.thr = thr

    def rps_at(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Telemetry at log times ``q``; second value: inside support, no hole."""
        i = np.clip(np.searchsorted(self.u, q) - 1, 0, len(self.u) - 1)
        ok = (q >= self.u[0]) & (q <= self.u[-1]) & (self.gap[i] <= HOLE_S)
        r = np.stack([np.interp(q.ravel(), self.u, row).reshape(q.shape) for row in self.rps], -2)
        return r, ok

    def score(self, lags, frames=None, *, scale=1.0, tol=0, chunk=32):
        """Mean comb z for each lag (``u = lag + t_wav``); and active-frame counts."""
        fi = np.arange(len(self.tf)) if frames is None else np.asarray(frames)
        base = (fi * self.nb)[None, None, :]
        out = np.full(len(lags), np.nan)
        cnt = np.zeros(len(lags), int)
        for s in range(0, len(lags), chunk):
            x = np.asarray(lags[s : s + chunk], float)
            r, ok = self.rps_at(x[:, None] + self.tf[fi][None, :])  # (B, 4, T)
            r = r * scale
            act = (r.mean(1) > self.thr) & ok
            num, c = np.zeros(len(x)), np.zeros(len(x), int)
            for k in self.orders:
                b = np.rint(r * (k / self.df)).astype(np.int64) - self.lo
                val = (b >= tol) & (b < self.nb - tol) & act[:, None, :]
                b = np.clip(b, tol, self.nb - 1 - tol) + base
                v = self.zt[b].astype(np.float32)
                for sh in range(1, tol + 1):
                    v = np.maximum(v, np.maximum(self.zt[b - sh], self.zt[b + sh]))
                num += np.where(val, v, 0).sum((1, 2))
                c += val.sum((1, 2))
            out[s : s + len(x)] = np.where(c > 0, num / np.maximum(c, 1), np.nan)
            cnt[s : s + len(x)] = c
        return out, cnt


def parabolic(xs, ys):
    i = int(np.nanargmax(ys))
    if i == 0 or i == len(xs) - 1:
        return float(xs[i]), float(ys[i]), True
    y0, y1, y2 = ys[i - 1], ys[i], ys[i + 1]
    den = y0 - 2 * y1 + y2
    if not den < 0:
        return float(xs[i]), float(y1), False
    h = xs[i] - xs[i - 1]
    return float(xs[i] - 0.5 * h * (y2 - y0) / den), float(y1 - 0.125 * (y2 - y0) ** 2 / den), False


def verdict(xs, ys, exclude_s):
    x_hat, y_hat, edge = parabolic(xs, ys)
    med = float(np.nanmedian(ys))
    mad = float(np.nanmedian(np.abs(ys - med))) or 1e-9
    far = np.abs(xs - x_hat) >= exclude_s
    j = int(np.nanargmax(np.where(far, ys, np.nan)))
    return {
        "lag_s": x_hat,
        "peak": y_hat - med,
        "z_mad": (y_hat - med) / mad,
        "rival_lag_s": float(xs[j]),
        "rival_ratio": float((ys[j] - med) / (y_hat - med)),
        "edge": edge,
    }


def fit_line(u, a):
    """``a = A + B*u`` least squares; returns A, B, residuals (s)."""
    B, A = np.polyfit(u, a, 1)
    return float(A), float(B), a - (A + B * u)


# ─────────────────────────────────────────────────────────────── driver
def run_one(rid: str, root: Path, out: Path) -> dict[str, Any]:
    cfg = SESSIONS[rid]
    d = root / cfg["dir"]
    z, tf, df, lo, dur = stream_spectrogram(d / cfg["wav"], cfg["band"])
    tick, u, rps = native_motors(d / cfg["dat"])
    comb = Comb(z, tf, df, lo, u, rps, cfg["orders"], cfg["thr"])
    print(
        f"{rid}: wav {dur:.3f} s, {len(tf)} frames; native {len(u)} packets u {u[0]:.3f}..{u[-1]:.3f}"
    )

    # 1. global lag: the whole feasible range (>= 30 % of the best overlap)
    xs = np.round(np.arange(u[0] - dur + 10, u[-1] - 10, 0.1), 4)
    ys, cnt = comb.score(xs, np.arange(0, len(tf), 4), tol=1)
    ys = np.where(cnt >= 0.3 * cnt.max(), ys, np.nan)
    g = verdict(xs, ys, 2.0)
    print(
        f"  global lag {g['lag_s']:+.3f} s  z {g['z_mad']:.1f} MAD  rival {g['rival_ratio']:.2f} @ {g['rival_lag_s']:+.1f}"
    )

    # 2. nuisance rev/s scale at that lag
    fx = np.round(np.arange(g["lag_s"] - 0.15, g["lag_s"] + 0.15 + 1e-9, 0.01), 4)
    scales = np.round(np.arange(0.990, 1.0201, 0.001), 4)
    every4 = np.arange(0, len(tf), 4)
    grid = np.stack([comb.score(fx, every4, scale=s)[0] for s in scales])
    si, oi = np.unravel_index(int(np.nanargmax(grid)), grid.shape)
    scale = parabolic(scales, grid[:, oi])[0]
    print(f"  rps_scale {scale:.5f}")

    # 3. windows tiling the active overlap
    q = g["lag_s"] + tf
    r, ok = comb.rps_at(q)
    act_t = tf[ok & (r.mean(0) > comb.thr)]
    a0, a1 = float(act_t[0]), float(act_t[-1])
    n_win = max(2, int(round((a1 - a0) / WINDOW_S)))
    starts = np.linspace(a0, a1 - WINDOW_S, n_win)
    wins = []
    for i, s0 in enumerate(starts):
        fr = np.where((tf >= s0) & (tf < s0 + WINDOW_S))[0]
        wx = np.round(np.arange(g["lag_s"] - 0.5, g["lag_s"] + 0.5 + 1e-9, 0.002), 4)
        wy, wc = comb.score(wx, fr, scale=scale)
        v = verdict(wx, wy, 0.1)
        v.update(
            window=i,
            role="begin" if i == 0 else "end" if i == n_win - 1 else "interior",
            t_wav_start=float(s0),
            t_wav_centre=float(s0 + WINDOW_S / 2),
            active_frames=int(wc.max()),
        )
        v["accepted"] = bool(not v["edge"] and v["rival_ratio"] < 0.8 and v["z_mad"] >= 3.0)
        v["u_centre"] = v["t_wav_centre"] + v["lag_s"]
        wins.append(v)
        print(
            f"  win {i:2d} [{v['role'][:3]}] wav {v['t_wav_centre']:7.1f}  lag {v['lag_s']:+.4f}  "
            f"z {v['z_mad']:5.1f}  rival {v['rival_ratio']:.2f}  {'ok' if v['accepted'] else 'REJECT'}"
        )
    acc = [w for w in wins if w["accepted"]]
    ends = [w for w in (wins[0], wins[-1]) if w["accepted"]]
    if len(ends) < 2:
        raise SystemExit(f"{rid}: begin/end anchor window rejected — no anchor fit")
    uc = np.array([w["u_centre"] for w in acc])
    ac = np.array([w["t_wav_centre"] for w in acc])
    A2, B2, _ = fit_line(
        np.array([w["u_centre"] for w in ends]), np.array([w["t_wav_centre"] for w in ends])
    )
    interior = np.array([w["role"] == "interior" for w in acc])
    res_anchor = ac - (A2 + B2 * uc)
    A, B, res_all = fit_line(uc, ac)
    c0 = float(np.mean(ac - uc))
    res_const = ac - (c0 + uc)
    for w, ra, rl in zip(acc, res_anchor, res_all, strict=True):
        w["resid_anchor_fit_ms"] = 1e3 * float(ra)
        w["resid_all_fit_ms"] = 1e3 * float(rl)

    def stats(x):
        x = 1e3 * np.asarray(x)
        return {
            "n": int(x.size),
            "mean_ms": float(x.mean()),
            "rms_ms": float(np.sqrt((x**2).mean())),
            "max_abs_ms": float(np.abs(x).max()),
        }

    # 4. order-free event cross-check: motor starts
    te, eb = broadband_db(d / cfg["wav"])
    slope = np.gradient(np.convolve(eb, np.ones(5) / 5, "same"))
    on = rps.mean(0) > 0
    events = []
    for i in np.where(np.diff(on.astype(int)) == 1)[0]:
        ue = float(u[i + 1])
        aw = A + B * ue
        sel = np.where((te > aw - 1.5) & (te < aw + 1.5))[0]
        if len(sel) < 10:
            continue
        j = sel[np.argmax(slope[sel])]
        events.append(
            {
                "u_first_nonzero": ue,
                "t_wav_onset": float(te[j]),
                "onset_minus_map_ms": 1e3 * float(te[j] - aw),
                "slope_db_per_10ms": float(slope[j]),
            }
        )

    # domain: where the map can carry native RPM onto the WAV
    holes = [
        {
            "u_start": float(u[i]),
            "u_end": float(u[i + 1]),
            "t_wav_start": A + B * float(u[i]),
            "t_wav_end": A + B * float(u[i + 1]),
        }
        for i in np.where(np.diff(u) > HOLE_S)[0]
    ]
    support = [max(0.0, A + B * float(u[0])), min(dur, A + B * float(u[-1]))]
    csv_map = csv_basis(rid, d / cfg["csv"], A, B)
    result = {
        "rid": rid,
        "rig": cfg["rig"],
        "wav": f"{cfg['dir']}/{cfg['wav']}",
        "dat": f"{cfg['dir']}/{cfg['dat']}",
        "wav_duration_s": dur,
        "telemetry": "read_motor_dat native motor packets (type 10090), sorted by tick; RPM/60, negatives clipped to 0",
        "measure": {
            "sr": SR,
            "n_fft": N_FFT,
            "hop": HOP,
            "band_hz": list(cfg["band"]),
            "orders": cfg["orders"],
            "active_mean_rps_gt": cfg["thr"],
            "hole_s": HOLE_S,
            "window_s": WINDOW_S,
        },
        "global": {k: v for k, v in g.items()},
        "rps_scale_nuisance": scale,
        "windows": wins,
        "fits": {
            "anchor_begin_end": {
                "A_s": A2,
                "B": B2,
                "interior_validation": stats(res_anchor[interior]),
            },
            "all_windows": {"A_s": A, "B": B, "residuals": stats(res_all)},
            "constant_offset_B1": {"A_s": c0, "residuals": stats(res_const)},
        },
        "map": {
            "formula": "t_wav_s = A_s + B * tick / 84e6",
            "A_s": A,
            "B": B,
            "seconds_per_tick": B / TICK_HZ,
            "implied_tick_hz_on_wav_clock": TICK_HZ / B,
            "tick_basis": "read_motor_dat tick: int64, low word = stored u32, high word 0 at the DAT's first record; == new_motor_data.datcon_ticks(CSV Clock:Tick#)",
            "fit": "all accepted windows (all_windows)",
        },
        "csv": csv_map,
        "domain": {
            "native_u_span": [float(u[0]), float(u[-1])],
            "native_on_wav_s": [A + B * float(u[0]), A + B * float(u[-1])],
            "supported_wav_s": support,
            "active_overlap_wav_s": [a0, a1],
            "holes": holes,
            "unsupported": "t_wav outside supported_wav_s (no native packet) and inside any hole; never interpolate across",
        },
        "events_motor_start": events,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{rid}.json").write_text(json.dumps(result, indent=1))
    keys = [
        "window",
        "role",
        "t_wav_start",
        "t_wav_centre",
        "u_centre",
        "lag_s",
        "z_mad",
        "rival_ratio",
        "accepted",
        "resid_anchor_fit_ms",
        "resid_all_fit_ms",
    ]
    lines = [",".join(keys)] + [",".join(str(w.get(k, "")) for k in keys) for w in wins]
    (out / f"{rid}_windows.csv").write_text("\n".join(lines) + "\n")
    plot(result, out / f"{rid}.png")
    print(
        f"  map t_wav = {A:+.6f} + {B:.11f} * tick/84e6; all-window RMS {stats(res_all)['rms_ms']:.2f} ms; "
        f"anchor-only interior RMS {stats(res_anchor[interior])['rms_ms']:.2f} ms; const-offset RMS {stats(res_const)['rms_ms']:.2f} ms"
    )
    return result


def csv_basis(rid: str, csv: Path, A: float, B: float) -> dict[str, Any]:
    """The same map on the CSV clocks (only the two clock columns are read)."""
    import pandas as pd

    # pandas' SequenceNotStr.index lets start/stop be keywords; list.index is positional-only.
    c = pd.read_csv(csv, usecols=["Clock:Tick#", "Clock:offsetTime"])  # pyright: ignore[reportCallIssue, reportArgumentType]
    raw = c["Clock:Tick#"].to_numpy(np.int64)
    native = datcon_ticks(raw)
    ot = c["Clock:offsetTime"].to_numpy(np.float64)
    # offsetTime = raw/85e6 + c0 (ms-rounded): c0 per file
    c0 = float(np.median(ot[1:] - raw[1:] / DATCON_HZ))
    spurious = np.flatnonzero(np.diff(raw[1:] - native[1:]) != 0)
    k = DATCON_HZ / TICK_HZ
    out = {
        "offsetTime_equals": f"Clock:Tick#/85e6 + {c0:.7f} (ms rounding, |dev| <= {1e3 * float(np.abs(ot[1:] - raw[1:] / DATCON_HZ - c0).max()):.2f} ms)",
        "tick_from_csv": "new_motor_data.datcon_ticks(whole Clock:Tick# column)",
        "spurious_high_word_increments_offsetTime": [float(ot[1:][i]) for i in spurious],
        "row_note": "CSV Motor:Speed rows hold the latest native packet; its stamp is a median 29-34 ms late — time RPM by the DAT packet tick, other CSV signals by datcon_ticks",
    }
    if spurious.size == 0:
        out["offsetTime_map"] = {
            "formula": "t_wav_s = A_csv + B_csv * offsetTime",
            "A_csv_s": A - B * k * c0,
            "B_csv": B * k,
        }
    else:
        out["offsetTime_map"] = None
        out["offsetTime_note"] = (
            f"{spurious.size} false +2^32 increments: offsetTime is not a valid clock past {ot[1:][spurious[0]]:.3f} s; use datcon_ticks"
        )
    return out


def plot(res: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    wins = res["windows"]
    acc = [w for w in wins if w["accepted"]]
    rej = [w for w in wins if not w["accepted"]]
    A, B = res["map"]["A_s"], res["map"]["B"]
    ax[0].plot(
        [w["t_wav_centre"] for w in acc],
        [1e3 * (w["lag_s"] + A) for w in acc],
        "o",
        label="accepted",
    )
    ax[0].plot(
        [w["t_wav_centre"] for w in rej],
        [1e3 * (w["lag_s"] + A) for w in rej],
        "x",
        color="r",
        label="rejected",
    )
    tt = np.array(res["domain"]["active_overlap_wav_s"])
    # lag(t) implied by the map: u = (t - A)/B -> lag = u - t
    ax[0].plot(tt, 1e3 * ((tt - A) / B - tt + A), "-", color="k", lw=0.8, label="map")
    ax[0].set_ylabel("lag + A_s [ms]")
    ax[0].legend()
    ax[0].set_title(f"{res['rid']}: t_wav = {A:+.4f} + {B:.8f}·tick/84e6")
    ax[1].plot(
        [w["t_wav_centre"] for w in acc],
        [w["resid_all_fit_ms"] for w in acc],
        "o",
        label="all-window fit",
    )
    ax[1].plot(
        [w["t_wav_centre"] for w in acc],
        [w["resid_anchor_fit_ms"] for w in acc],
        "s",
        mfc="none",
        label="begin/end-only fit",
    )
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_ylabel("residual [ms]")
    ax[1].set_xlabel("WAV time [s]")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--rid", choices=sorted(SESSIONS), action="append")
    ap.add_argument(
        "--root", type=Path, default=None, help="raw tree (default: raw_root('new_motor_data'))"
    )
    ap.add_argument("--out", type=Path, default=REPO / "results" / "new_motor_data_alignment")
    args = ap.parse_args()
    if args.root is None:
        from data_processing.sources import raw_root

        args.root = raw_root("new_motor_data")
    rids = args.rid or sorted(SESSIONS)
    summary = {}
    for rid in rids:
        r = run_one(rid, args.root, args.out)
        summary[rid] = {
            "map": r["map"],
            "csv_offsetTime_map": r["csv"]["offsetTime_map"],
            "fits": r["fits"],
            "domain": r["domain"],
        }
    maps = args.out / "maps.json"
    old = json.loads(maps.read_text()) if maps.exists() else {}
    old.update(summary)
    maps.write_text(json.dumps(old, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
