"""Standby (idle) interval of the DREGON flights, recovered from the audio.

Between motor start and take-off the MikroKopter's ESCs hold every rotor at one
idle speed (15.27 rev/s on room1's ``motors_measured``). ``motors_command``
reads 0 there, and the rotors leave idle only when the command ramp crosses the
idle speed. Room2 recordings carry the command only, so the standby START is
read from the audio (20-60 Hz level, below the loudspeaker's output) and the
END from the command's idle crossing; both are validated on room1 against
``motors_measured``. Run ``PYTHONPATH=src python -m
experiments.rps_traj.standby_audio`` -> ``results/rps_traj/standby_audio/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, resample_poly, sosfiltfilt, stft, welch

from data_processing.streams import iter_published_frames
from experiments.rps_traj.data import _meta_scalar, _series_times

OUT = Path("results/rps_traj/standby_audio")
IDLE_RPS = 15.27  # room1 measured idle plateau (rev/s)
SR = 4410.0  # audio decimated 10x: everything used here lies below 600 Hz
SPAN_S = (-2.5, 40.0)  # audio kept around the first motor stamp
BAND_HZ = (20.0, 60.0)  # level band: idle shaft harmonics 2-3, under the loudspeaker
WIN_S, HOP_S = 0.1, 0.01  # level window / hop
CMD_MEDIAN = 51  # command samples (~50 ms) median filter against telemetry spikes
LEVEL_WIN_S = 0.5  # idle level L = median level over the last 0.5 s of standby
TROUGH_DB = 10.0  # standby starts after the last level sample below L - 10 dB
MIN_STANDBY_S = 0.3  # a shorter interval is reported as "no standby"
COMB_FMAX = 600.0  # comb salience uses shaft harmonics 2 .. 600 Hz


def comb_db(f, lp, f0, ks):
    """Mean dB of bins k*f0 over half-way bins (k+-0.5)*f0, k in ks; lp is (freq, time), f0 scalar or per frame."""
    f0 = np.broadcast_to(np.asarray(f0, dtype=float), (lp.shape[1],))

    def at(m):
        x = np.asarray(m, dtype=float)[:, None] * f0 / f[1]
        i = x.astype(int)
        return np.take_along_axis(lp, i, 0) * (i + 1 - x) + np.take_along_axis(lp, i + 1, 0) * (
            x - i
        )

    return (at(ks) - 0.5 * (at(ks - 0.5) + at(ks + 0.5))).mean(0)


def level_db(x, t0):
    """Mic-averaged 20-60 Hz power in dB on 0.1 s windows; returns (times, dB)."""
    y = sosfiltfilt(butter(4, BAND_HZ, "bandpass", fs=SR, output="sos"), x, axis=1)
    c = np.concatenate([[0.0], np.cumsum((y**2).mean(0))])
    w, h = int(WIN_S * SR), int(HOP_S * SR)
    s = np.arange(0, y.shape[1] - w, h)
    return t0 + (s + w / 2) / SR, 10 * np.log10((c[s + w] - c[s]) / w + 1e-20)


def spec(x, t0, a, b, win):
    i, j = max(0, int((a - t0) * SR)), int((b - t0) * SR)
    f, t, z = stft(x[:, i:j], fs=SR, nperseg=int(win * SR), noverlap=int(win * SR) - int(0.01 * SR))
    return f, t0 + i / SR + t, 10 * np.log10((np.abs(z) ** 2).mean(0) + 1e-20)


def clock_lag(x, t0, tt, speed, a, b):
    """Lag (s, + = audio later) maximising the comb salience at harmonics 1-6 of the 4 rotors' speeds."""
    f, tf, lp = spec(x, t0, a - 1, b + 1, 0.2)
    sel = (tf >= a) & (tf <= b)
    lags = np.round(np.arange(-0.6, 0.6001, 0.01), 2)
    score = [
        np.mean(
            [
                comb_db(f, lp[:, sel], np.interp(tf[sel] - lag, tt, s), np.arange(1, 7))
                for s in speed
            ]
        )
        for lag in lags
    ]
    return float(lags[int(np.argmax(score))])


def knee(t, e):
    """Rejected audio-only end: knee of a flat-then-linear fit to the 1.5 s before the level reaches P90 - 20 dB."""
    tu = t[np.argmax((e >= np.percentile(e[t >= 0], 90) - 20) & (t >= 0))]
    w = (t >= tu - 1.5) & (t <= tu)
    tw, ew = t[w], e[w]
    res = [
        np.linalg.lstsq(np.c_[np.ones_like(tw), np.maximum(tw - k, 0)], ew, rcond=None)[1]
        for k in tw[5:-5]
    ]
    return float(tw[5:-5][int(np.argmin([r[0] if r.size else np.inf for r in res]))])


def analyse(rid, frame):
    cmd_s = frame["motors_command"]
    cmd = np.asarray(cmd_s.data, dtype=float)
    stamps = _series_times(cmd_s, cmd.shape[1])
    tt = stamps - stamps[0]
    a = frame["audio"]
    sr0 = float(a.tindex.rate)
    off = stamps[0] - float(a.t_start)  # first motor stamp, s into the audio
    i0 = max(0, int((off + SPAN_S[0]) * sr0))
    x = resample_poly(
        np.asarray(a.data[:, i0 : int((off + SPAN_S[1]) * sr0)], dtype=float), 1, 10, axis=1
    )
    t0 = i0 / sr0 - off
    cmd_f = median_filter(cmd, (1, CMD_MEDIAN), mode="nearest")
    mean_cmd = cmd_f.mean(0)
    i_end = int(np.argmax(mean_cmd >= IDLE_RPS))
    end = float(tt[i_end])
    t, e = level_db(x, t0)
    lvl = float(np.median(e[(t >= end - LEVEL_WIN_S) & (t < end)]))
    below = np.flatnonzero((t < end) & (e < lvl - TROUGH_DB))
    start: float | None = float(t[below[-1] + 1]) if below.size else None
    if start is not None and end - start < MIN_STANDBY_S:
        start = None
    present = start is not None
    r: dict[str, Any] = dict(
        recording_id=rid,
        channel="measured" if "motors_measured" in frame else "command",
        standby_start_s=start,
        standby_end_s=end if present else None,
        standby_s=None if start is None else round(end - start, 3),
        idle_level_db=round(lvl, 1),
        command_ramp_onset_s=float(tt[i_end - int(np.argmax(mean_cmd[:i_end][::-1] <= 0.5))]),
        end_audio_knee_s=knee(t, e),
    )
    if (
        start is not None
    ):  # idle comb read-out on the standby audio, and the 15.27 Hz comb across the end
        seg = x[:, int((start + 0.3 - t0) * SR) : int((end - 0.1 - t0) * SR)]
        f, p = welch(seg, fs=SR, nperseg=min(int(SR), seg.shape[1]))
        f0 = np.round(np.arange(12.0, 30.0, 0.01), 2)
        lw = 10 * np.log10(p.mean(0) + 1e-20)[:, None]
        sal = np.array([comb_db(f, lw, g, np.arange(2, int(COMB_FMAX // g) + 1))[0] for g in f0])
        near = (f0 >= 14.0) & (f0 <= 16.5)
        r.update(
            idle_rps_audio=float(f0[near][np.argmax(sal[near])]),
            idle_comb_db=round(float(sal[near].max()), 2),
            idle_comb_is_global_peak_12_30=bool(sal[near].max() >= sal.max()),
        )
        fs, ts, lp = spec(x, t0, end - 1.5, end + 1.5, 0.25)
        c = comb_db(fs, lp, IDLE_RPS, np.arange(2, int(COMB_FMAX // IDLE_RPS) + 1))
        r.update(
            comb_idle_db_before_end=round(
                float(c[(ts >= end - 0.8) & (ts < end - 0.05)].mean()), 2
            ),
            comb_idle_db_after_end=round(float(c[(ts > end + 0.2) & (ts <= end + 1.0)].mean()), 2),
        )
    plot: dict[str, Any] = dict(t=t, e=e, tt=tt, cmd=mean_cmd, meas=None)
    lag_speed = cmd_f
    if "motors_measured" in frame:
        meas = np.asarray(frame["motors_measured"].data, dtype=float)
        i_s = int(np.argmax((meas >= IDLE_RPS - 1).all(0)))
        m = meas.mean(0)
        j = int(np.argmax(m > 30))
        i_e = j - 1 - int(np.argmax(m[:j][::-1] <= IDLE_RPS + 0.5))
        ts_, te_ = float(tt[i_s]), float(tt[i_e])
        r.update(
            truth_start_s=ts_,
            truth_end_s=te_,
            truth_standby_s=round(te_ - ts_, 3),
            idle_rps_measured=round(float(meas[:, i_s : i_e + 1].mean()), 3),
            command_max_before_ramp_onset=float(
                cmd[:, (tt >= ts_) & (tt < r["command_ramp_onset_s"] - 0.1)].max()
            ),
            err_start_s=None if start is None else round(start - ts_, 3),
            err_end_s=round(end - te_, 3) if present else None,
            err_end_audio_knee_s=round(r["end_audio_knee_s"] - te_, 3),
        )
        plot["meas"], plot["truth"], lag_speed = (
            m,
            (ts_, te_),
            median_filter(meas, (1, CMD_MEDIAN), mode="nearest"),
        )
    r["audio_lag_s"] = clock_lag(
        x, t0, tt, lag_speed, end + 3, min(end + 30, tt[-1] - 1, SPAN_S[1] - 1.5)
    )
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, plot


ROOM1 = [
    f"free-flight_{s}_room1"
    for s in ("nosource", "speech-high", "speech-low", "whitenoise-high", "whitenoise-low")
]
ROOM2 = [
    f"{s}_nosource_room2" for s in ("free-flight", "hovering", "rectangle", "spinning", "updown")
]
STEPS = [
    "Level: band-pass every microphone to 20-60 Hz (4th-order Butterworth, zero-phase; the loudspeaker radiates almost "
    "nothing there, the idle rotors' shaft harmonics 2-3 do), average the power over the 8 mics, 0.1 s windows, "
    "10 ms hop, in dB.",
    "End: standby ends when the rotor-mean command (50-sample median filter against telemetry spikes) first reaches "
    f"the idle speed {IDLE_RPS} rev/s; below it the ESCs hold the idle, so the rotors do not move before that.",
    f"Idle level L: median level over the last {LEVEL_WIN_S} s before the end.",
    f"Start: the first window after the last one, before the end, whose level is more than {TROUGH_DB} dB below L "
    "(the motors-off floor, or the trough between the ESC start-up burst and the idle plateau).",
    f"An interval shorter than {MIN_STANDBY_S} s (or no window {TROUGH_DB} dB below L) is reported as no standby.",
]


def main() -> None:
    res, plots = {}, {}
    for frame in iter_published_frames("DREGON-frames"):
        if "motors_command" not in frame:
            continue
        rid = str(_meta_scalar(frame, "recording_id"))
        if rid in ROOM1 + ROOM2:
            res[rid], plots[rid] = analyse(rid, frame)
    recs = [res[r] for r in ROOM1 + ROOM2]
    r1 = recs[:5]
    errs = [abs(r[k]) for r in r1 for k in ("err_start_s", "err_end_s") if r[k] is not None]
    idle = float(np.mean([r["idle_rps_measured"] for r in r1]))
    lead = [r["truth_end_s"] - r["command_ramp_onset_s"] for r in r1]
    step1 = (
        f"During the measured {idle:.2f} rev/s plateau the command reads 0 "
        f"(max {max(r['command_max_before_ramp_onset'] for r in r1):.2f} rev/s) until its ramp starts "
        f"{min(lead):.2f}-{max(lead):.2f} s before take-off; the rotors leave idle when that ramp crosses "
        f"{IDLE_RPS} rev/s (command crossing minus measured take-off: "
        + ", ".join(f"{r['err_end_s']:+.3f}" for r in r1)
        + " s), i.e. measured = max(idle, command)."
    )
    knee_r1 = [r["err_end_audio_knee_s"] for r in r1]
    knee_r2 = [
        round(r["end_audio_knee_s"] - r["standby_end_s"], 3) for r in recs[5:] if r["standby_end_s"]
    ]
    summary = dict(
        method=dict(
            steps=STEPS,
            thresholds=dict(
                idle_rps=IDLE_RPS,
                band_hz=list(BAND_HZ),
                window_s=WIN_S,
                hop_s=HOP_S,
                command_median_samples=CMD_MEDIAN,
                idle_level_window_s=LEVEL_WIN_S,
                start_below_idle_db=TROUGH_DB,
                min_standby_s=MIN_STANDBY_S,
            ),
            times="seconds from the first motor stamp of the recording",
            truth="room1 motors_measured: start = first stamp with all four rotors >= idle-1 rev/s; end = last stamp "
            "with rotor-mean <= idle+0.5 rev/s before it first exceeds 30 rev/s",
            rejected=[
                "Broadband RMS envelope (all mics): the loudspeaker swamps it (white noise holds a flat broadband "
                "floor through motor start, speech swings 30 dB), so no off->idle step exists on 4 of 5 room1 runs.",
                "Audio-only end (knee of a flat-then-linear fit of the 20-60 Hz level): late because the level rises "
                "slowly at the start of the ramp; signed error vs measured take-off "
                + ", ".join(f"{e:+.2f}" for e in knee_r1)
                + " s on room1, and "
                + ", ".join(f"{e:+.2f}" for e in knee_r2)
                + " s after the command idle crossing on room2, where the "
                "15.27 Hz comb in the audio vanishes at the crossing (comb_idle_db_before/after_end).",
                "Command ramp onset (last command sample <= 0.5 rev/s before the idle crossing) as the end: precedes "
                f"the measured take-off by {min(lead):.2f}-{max(lead):.2f} s.",
                "Idle-comb salience (15.27 Hz harmonics vs half-way bins) as the detector: clear only without "
                "loudspeaker; kept only as the idle-speed read-out and as a check of the end.",
            ],
            clock_check="audio_lag_s: lag maximising the audio comb salience at harmonics 1-6 of the four rotors' "
            "speeds (room1 measured, room2 command) over 27 s of flight; + = audio later",
        ),
        command_vs_measured=step1,
        idle_rps_measured_room1=round(idle, 3),
        validation=dict(
            worst_abs_error_s=max(errs),
            target_s=0.2,
            met=max(errs) <= 0.2,
            misses=[
                f"{r['recording_id']} {k} {r[k]:+.3f}"
                for r in r1
                for k in ("err_start_s", "err_end_s")
                if abs(r[k]) > 0.2
            ],
        ),
        recordings=recs,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axs = plt.subplots(10, 1, figsize=(13, 25), sharex=True)
    for ax, r in zip(axs, recs):
        p, sp = plots[r["recording_id"]], ax.twinx()
        ax.plot(p["t"], p["e"], lw=0.7, color="tab:blue", label="20-60 Hz level (dB)")
        ax.axhline(
            r["idle_level_db"] - TROUGH_DB, color="tab:blue", ls=":", lw=0.8, label="L - 10 dB"
        )
        if r["standby_s"] is not None:
            ax.axvspan(
                r["standby_start_s"],
                r["standby_end_s"],
                color="tab:green",
                alpha=0.25,
                label="detected standby",
            )
        for x in p.get("truth", ()):
            ax.axvline(x, color="tab:red", ls="--", lw=1.0)
        sp.plot(p["tt"], p["cmd"], color="k", lw=0.8, label="command (rotor mean)")
        if p["meas"] is not None:
            sp.plot(
                p["tt"],
                p["meas"],
                color="tab:red",
                lw=0.8,
                label="measured (rotor mean); -- true standby",
            )
        sp.axhline(IDLE_RPS, color="gray", lw=0.5)
        ax.set(ylabel="dB", xlim=(-1, 10))
        sp.set(ylabel="rev/s", ylim=(-2, 95))
        dur = f"{r['standby_s']:.2f} s" if r["standby_s"] else "none"
        ax.set_title(f"{r['recording_id']} - standby {dur}", fontsize=9, loc="left")
        if ax is axs[0]:
            ax.legend(loc="upper left", fontsize=7)
            sp.legend(loc="lower right", fontsize=7)
    axs[-1].set_xlabel("s from first motor stamp")
    fig.tight_layout()
    fig.savefig(OUT / "overview.png", dpi=90)


if __name__ == "__main__":
    main()
