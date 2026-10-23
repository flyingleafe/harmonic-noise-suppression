"""Build the assets of ``docs/explainers/synthetic-transfer-failures.qmd``:
where the synthetic-trained HPPNet-L2 arms fail on the rp_real panel.

Inputs: ``results/rp_panel_eval/synthetic.json`` (scores of the four
synthetic-trained checkpoints, ``scripts/rp_panel_eval.py``),
``results/rp_panel_eval/pred_synthetic/*.npz`` (per-frame predictions,
``rp_panel_eval.py --dump``), the rp_real panel audio and the DREGON-frames
telemetry the panel was cut from (to place the DREGON labels at their recorded
time stamps). Writes figures, WAVs (every clip at the same RMS) and
``index.json`` next to this file.

    PYTHONPATH=src .venv/bin/python docs/explainers/synthetic-transfer-failures/build.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import rp_panel_eval  # noqa: E402

from plots.timeframe.renderers import ROTOR_COLORS  # noqa: E402
from training.validation import build_validation_plan  # noqa: E402

DUMP = ROOT / "results/rp_panel_eval/pred_synthetic"
SCORES = ROOT / "results/rp_panel_eval/synthetic.json"
MODELS = {
    "prior_v1": "prior_v1_hppnet_l2__best_real_overall.npz",
    "nv3r4 legacytraj": "nv3r4_hard_legacytraj_hppnet_l2__best_real_overall.npz",
}
SR = 16000
TARGET_RMS = 0.05
REG = ("zero", "standby", "ramp", "cruise")
SHADE = {0: "#9e9e9e", 1: "#9ecae1", 2: "#fdae6b"}
MODEL_COLOR = {"prior_v1": "#d62728", "nv3r4 legacytraj": "#1f77b4"}
DREGON = {
    "free-flight_nosource_room1": "no loudspeaker",
    "free-flight_speech-low_room1": "loudspeaker speech",
    "free-flight_whitenoise-low_room1": "loudspeaker white noise",
}
WHINE_DB = 15.0  # motor-whine onset threshold (dB over the side bands), see whine_contrast

runs = {name: dict(np.load(DUMP / f)) for name, f in MODELS.items()}
ref = runs["prior_v1"]
META = json.loads(str(ref["meta"]))
TARGET, REGIME = ref["target"], ref["regime"]
RATE = float(ref["frame_rate"])
N, R, T = TARGET.shape
T_AX = np.arange(T) / RATE
DATASET = np.array([m["dataset"] for m in META])
RECORDING = np.array([m["recording_id"] for m in META])
PLAN = build_validation_plan(rp_panel_eval._panel("rp_real", "rp_scv2_all"))
assert PLAN.size == N


def audio(i: int) -> np.ndarray:
    return np.asarray(cast(Any, PLAN.dataset)[i]["mixture"].data, dtype=np.float64).reshape(-1)


def write_wav(x: np.ndarray, slug: str) -> str:
    y = x / (np.sqrt(np.mean(x**2)) + 1e-12) * TARGET_RMS
    y = y / max(1.0, float(np.max(np.abs(y))) / 0.99)
    sf.write(HERE / f"{slug}.wav", y.astype(np.float32), SR)
    return f"{slug}.wav"


def power_frames(
    x: np.ndarray, nfft: int, hop: int = 512
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(t (s, frame centres), f (Hz), power (F, n))`` of a Hann STFT."""
    win = np.hanning(nfft + 1)[:nfft]
    starts = np.arange(0, x.size - nfft + 1, hop)
    frames = np.stack([x[s : s + nfft] * win for s in starts])
    power = np.abs(np.fft.rfft(frames, axis=-1)) ** 2 / np.sum(win**2)
    return (starts + nfft / 2) / SR, np.fft.rfftfreq(nfft, 1.0 / SR), power.T


def cell(name: str, ds: str, code: int) -> tuple[np.ndarray, np.ndarray]:
    """``(label, prediction)`` of every rotor-frame of one dataset / regime."""
    sel = np.broadcast_to(
        (np.equal(DATASET, ds)[:, None] & np.equal(REGIME, code))[:, None, :], TARGET.shape
    )
    return TARGET[sel], runs[name]["pred"][sel]


def shade(ax, i: int) -> None:
    for code, colour in SHADE.items():
        edges = np.flatnonzero(np.diff(np.r_[0, (REGIME[i] == code).astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2], strict=True):
            ax.axvspan(T_AX[a], T_AX[min(b, T - 1)], color=colour, alpha=0.35, lw=0)


def fig_octave_map() -> None:
    cells = [("dregon", 1), ("dregon", 2), ("dregon", 3), ("md2", 1), ("md2", 2), ("md2", 3)]
    fig, axes = plt.subplots(2, 6, figsize=(22, 7.6), sharex=True, sharey=True)
    edges = np.linspace(0, 160, 161)
    for i, name in enumerate(MODELS):
        for j, (ds, code) in enumerate(cells):
            ax = axes[i, j]
            g, p = cell(name, ds, code)
            ax.hist2d(g, p, bins=[edges, edges], norm=LogNorm(vmin=1), cmap="magma")
            for k, ls in ((1.0, "-"), (2.0, "--"), (0.5, ":")):
                ax.plot(
                    [0, 160],
                    [0, 160 * k],
                    color="cyan",
                    lw=1.0,
                    ls=ls,
                    label=f"pred = {k:g} x label",
                )
            ax.text(
                0.03,
                0.97,
                f"MAE {np.mean(np.abs(p - g)):.1f}",
                transform=ax.transAxes,
                va="top",
                fontsize=11,
                color="white",
                bbox=dict(facecolor="black", alpha=0.5, lw=0),
            )
            ax.set_xlim(0, 160)
            ax.set_ylim(0, 160)
            if i == 0:
                ax.set_title(f"{ds} {REG[code]}", fontsize=13)
            if j == 0:
                ax.set_ylabel(f"{name}\npredicted rev/s", fontsize=12)
            if i == 1:
                ax.set_xlabel("label rev/s", fontsize=12)
    axes[0, 0].legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "octave_map.png", dpi=110)
    plt.close(fig)


def fig_ratio_hist() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 7.5), sharex=True)
    bins = np.linspace(-2.5, 2.5, 201)
    marks = ((0, "1"), (1, "2"), (-1, "1/2"), (np.log2(1.5), "3/2"), (np.log2(2 / 3), "2/3"))
    for i, ds in enumerate(("dregon", "md2")):
        for j, code in enumerate((1, 2, 3)):
            ax = axes[i, j]
            for name in MODELS:
                g, p = cell(name, ds, code)
                ok = (g > 5) & (p > 0.5)
                ax.hist(
                    np.log2(p[ok] / g[ok]),
                    bins=bins,
                    histtype="step",
                    lw=1.6,
                    color=MODEL_COLOR[name],
                    density=True,
                    label=name,
                )
            for v, lab in marks:
                ax.axvline(v, color="k", lw=0.7, ls=":")
                ax.text(
                    v,
                    1.0,
                    lab,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            ax.set_yscale("log")
            ax.set_title(f"{ds} {REG[code]}", fontsize=13, pad=14)
            if i == 1:
                ax.set_xlabel(
                    "log2(predicted / label); label > 5, prediction > 0.5 rev/s", fontsize=11
                )
    axes[0, 0].legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / "ratio_hist.png", dpi=110)
    plt.close(fig)


def breakdown() -> dict:
    """Outcome shares per model / dataset / regime and MAE per recording."""
    out: dict = {"cells": [], "recordings": []}
    for name in MODELS:
        for ds in ("dregon", "md2"):
            for code in range(4):
                g, p = cell(name, ds, code)
                row = dict(model=name, cell=f"{ds} {REG[code]}", mae=float(np.mean(np.abs(p - g))))
                if code == 0:
                    row["on"] = float(np.mean(p > 5))
                else:
                    keep = g > 5
                    g, p = g[keep], p[keep]
                    r = p / g
                    ok, off = np.abs(r - 1) < 0.1, p < 5
                    x2, xh = np.abs(r / 2 - 1) < 0.1, np.abs(r * 2 - 1) < 0.1
                    row.update(
                        ok=float(ok.mean()),
                        off=float(off.mean()),
                        x2=float(x2.mean()),
                        xh=float(xh.mean()),
                        other=float((~(ok | off | x2 | xh)).mean()),
                    )
                out["cells"].append(row)
        err = runs[name]["frame_err"]
        for rid in sorted(set(RECORDING)):
            row: dict[str, Any] = dict(model=name, recording=rid)
            for code in range(4):
                m = np.equal(RECORDING, rid)[:, None] & np.equal(REGIME, code)
                row[REG[code]] = float(err[m].mean()) if m.any() else None
            out["recordings"].append(row)
    return out


def stopped_spectrum(rid: str, nfft: int = 16384) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean periodogram of one recording's windows lying wholly in stopped frames."""
    acc, count, f = 0.0, 0, np.fft.rfftfreq(nfft, 1.0 / SR)
    for i in np.flatnonzero(rid == RECORDING):
        t, f, pw = power_frames(audio(i), nfft, hop=nfft // 4)
        lo = np.minimum(((t - nfft / 2 / SR) * RATE).astype(int), T - 1)
        hi = np.minimum(((t + nfft / 2 / SR) * RATE).astype(int), T - 1)
        keep = (REGIME[i][lo] == 0) & (REGIME[i][hi] == 0)
        acc = acc + pw[:, keep].sum(axis=1)
        count += int(keep.sum())
    return f, 10 * np.log10(np.asarray(acc) / max(count, 1) + 1e-20), count


def fig_stopped() -> float:
    """DREGON stopped frames per recording; returns the room tone (Hz)."""
    colour = dict(zip(DREGON, ("k", "#2ca02c", "#d62728"), strict=True))
    fig, axes = plt.subplots(1, 2, figsize=(19, 6.5), gridspec_kw=dict(width_ratios=[1, 1.4]))
    tone, levels = 0.0, []
    for rid, label in DREGON.items():
        sel = np.flatnonzero(rid == RECORDING)
        mask = np.broadcast_to((REGIME[sel] == 0)[:, None, :], TARGET[sel].shape)
        p = runs["prior_v1"]["pred"][sel][mask]
        pn = runs["nv3r4 legacytraj"]["pred"][sel][mask]
        axes[0].hist(
            p,
            bins=np.linspace(0, 160, 161),
            histtype="step",
            lw=1.8,
            color=colour[rid],
            label=f"{label}: prior_v1 mean {p.mean():.1f} (nv3r4 {pn.mean():.1f})",
        )
        f, spec, count = stopped_spectrum(rid)
        axes[1].plot(f, spec, color=colour[rid], lw=0.9, label=f"{label} ({count} windows)")
        levels.append(spec[(f >= 20) & (f <= 500)])
        if rid == "free-flight_nosource_room1":
            band = (f > 75) & (f < 95)
            tone = float(f[band][np.argmax(spec[band])])
    for k in range(1, 6):
        axes[1].axvline(
            k * tone,
            color="#1f77b4",
            lw=0.8,
            ls="--",
            label=f"multiples of {tone:.1f} Hz" if k == 1 else None,
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel(
        "predicted rev/s, DREGON panel 'zero' frames (all before the first telemetry stamp)",
        fontsize=12,
    )
    axes[0].set_ylabel("rotor-frames", fontsize=12)
    axes[0].set_title("What prior_v1 predicts on the DREGON 'zero' frames", fontsize=13)
    axes[0].legend(fontsize=10)
    axes[1].set_xlim(20, 500)
    lv = np.concatenate(levels)
    axes[1].set_ylim(np.percentile(lv, 1) - 3, lv.max() + 12)
    axes[1].set_xlabel("Hz", fontsize=12)
    axes[1].set_ylabel("dB, mean 16384-point periodogram", fontsize=12)
    axes[1].set_title(
        "DREGON room 1, panel 'zero' frames: the room's own harmonic series", fontsize=13
    )
    axes[1].legend(fontsize=10, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(HERE / "stopped.png", dpi=110)
    plt.close(fig)
    return tone


def spectro(ax, x: np.ndarray, nfft: int, fmin: float, fmax: float, log: bool) -> None:
    """dB spectrogram, colour scale p30-p99.9 of the panel (per clip, not absolute)."""
    t, f, pw = power_frames(x, nfft)
    keep = (f >= fmin) & (f <= fmax)
    db = 10 * np.log10(pw[keep] + 1e-20)
    ax.pcolormesh(
        t,
        f[keep],
        db,
        vmin=np.percentile(db, 30),
        vmax=np.percentile(db, 99.9),
        cmap="inferno",
        shading="auto",
    )
    if log:
        ax.set_yscale("log")
    ax.set_ylim(fmin, fmax)


def harmonics(ax, t: np.ndarray, rps: np.ndarray, ks, **style) -> None:
    """``k x rps`` lines where the rotors turn (``rps`` rotor mean, rev/s)."""
    for n, k in enumerate(ks):
        lab = style.pop("label", None) if n == 0 else None
        ax.plot(t, np.where(rps > 1, k * rps, np.nan), label=lab, **style)


def fig_example(i: int, title: str, slug: str, true: np.ndarray) -> None:
    """One panel item; ``true`` is its label at the telemetry's recorded stamps
    (drawn where it differs from the stretched panel label)."""
    x = audio(i)
    fig, axes = plt.subplots(
        4, 1, figsize=(16, 15), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 1.4, 1, 1])
    )
    spectro(axes[0], x, 4096, 20, 8000, log=True)
    axes[0].set_ylabel("Hz (log)", fontsize=12)
    axes[0].set_title(title, fontsize=13)
    spectro(axes[1], x, 8192, 0, 400, log=False)
    harmonics(
        axes[1],
        T_AX,
        TARGET[i].mean(0),
        range(1, 9),
        color="white",
        lw=0.8,
        ls=":",
        label="label rotor mean x k, k = 1..8",
    )
    harmonics(
        axes[1],
        T_AX,
        runs["prior_v1"]["pred"][i].mean(0),
        (1, 2),
        color="cyan",
        lw=1.3,
        ls="--",
        label="prior_v1 rotor mean x 1, x 2",
    )
    axes[1].set_ylabel("Hz (0-400, 8192-point)", fontsize=12)
    axes[1].legend(loc="upper left", fontsize=10, facecolor="0.25", labelcolor="white")
    moved = bool(np.isnan(true).any() or np.abs(true - TARGET[i]).max() > 0.5)
    for ax, name in zip(axes[2:], MODELS, strict=True):
        shade(ax, i)
        p = runs[name]["pred"][i]
        for r in range(R):
            ax.plot(T_AX, TARGET[i, r], color=ROTOR_COLORS[r], lw=2.2)
            ax.plot(T_AX, p[r], color=ROTOR_COLORS[r], lw=1.3, ls="--")
        if moved:
            gap = np.isnan(true).any(axis=0)
            if gap.any():
                ax.axvspan(
                    0,
                    T_AX[np.flatnonzero(gap)[-1]],
                    facecolor="none",
                    edgecolor="0.45",
                    hatch="//",
                    lw=0,
                    label="no telemetry yet",
                )
            ax.plot(
                T_AX,
                true.mean(axis=0),
                color="#e377c2",
                lw=3.0,
                ls=":",
                zorder=5,
                label="telemetry at its recorded stamps (rotor mean)",
            )
            ax.legend(loc="center left", fontsize=10)
        ax.set_ylim(-4, max(160.0, float(p.max()) + 5))
        ax.set_ylabel(f"{name}\nrev/s", fontsize=12)
        ax.text(
            0.01,
            0.96,
            f"solid label, dashed prediction, colour = rotor; clip MAE "
            f"{runs[name]['frame_err'][i].mean():.1f}",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.85, lw=0),
        )
    axes[-1].set_xlabel("s", fontsize=12)
    fig.tight_layout()
    fig.savefig(HERE / f"{slug}.png", dpi=110)
    plt.close(fig)


def recording(rid: str, until_s: float | None = None):
    """One panel recording stitched from its contiguous 8 s clips:
    ``(audio (mics, n), label time (L,), label (R, L), regime (L,))``."""
    starts = sorted({m["t_start_s"] for m in META if m["recording_id"] == rid})
    starts = [s for s in starts if until_s is None or s < until_s]
    assert np.allclose(np.diff(starts), 8.0), f"{rid}: clips are not contiguous: {starts}"
    chans = sorted({m["channel"] for m in META if m["recording_id"] == rid})
    where = {(m["recording_id"], m["t_start_s"], m["channel"]): i for i, m in enumerate(META)}
    x = np.stack([np.concatenate([audio(where[(rid, s, c)]) for s in starts]) for c in chans])
    first = [where[(rid, s, chans[0])] for s in starts]
    t = np.concatenate([s + T_AX for s in starts])
    label = np.concatenate([TARGET[i] for i in first], axis=1)
    regime = np.concatenate([REGIME[i] for i in first])
    return x, t, label, regime


def mic_mean_db(x: np.ndarray, nfft: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mic-mean power spectrogram in dB."""
    t, f, acc = power_frames(x[0], nfft)
    for ch in x[1:]:
        acc = acc + power_frames(ch, nfft)[2]
    return t, f, 10 * np.log10(acc / len(x) + 1e-20)


def true_time_labels() -> tuple[np.ndarray, np.ndarray, dict]:
    """The DREGON panel labels rebuilt on the telemetry's own time stamps.

    The panel stores each clip's telemetry samples as an untimed ``(R, M)``
    array, and the loader stretches them end to end over the 8 s clip
    (``stretch_rps_to_frames``). Here the same samples sit at their recorded
    stamps. Returns ``(label (N, R, T), lag (N, T), first stamp per
    recording)``: ``label`` is NaN where a clip has no telemetry; ``lag`` is
    the recorded time of the sample the stretched label shows at a frame,
    minus that frame's time (> 0: the panel shows a value early)."""
    from data_processing.derivations import PARENTS, _panel_recordings
    from data_processing.streams import stretch_rps_to_frames

    src = dict(
        dataset=PARENTS["DREGON-frames"], label_key="motors_measured", recording_ids=list(DREGON)
    )
    label, lag, first = np.array(TARGET, dtype=np.float64), np.zeros((N, T)), {}
    for fr in _panel_recordings(src, sample_rate=SR):
        rid = str(fr["meta"]["recording_id"])
        st = np.asarray(fr["rps"].tindex.abs_stamps, dtype=np.float64) - float(fr["audio"].t_start)
        val = np.asarray(fr["rps"].data, dtype=np.float64)
        first[rid] = float(st[0])
        for i in np.flatnonzero(rid == RECORDING):
            t0 = float(META[i]["t_start_s"])
            m = (st >= t0) & (st < t0 + 8.0)
            sc, vc, M = st[m], val[:, m], int(m.sum())
            err = np.abs(stretch_rps_to_frames(vc.astype(np.float32), T) - TARGET[i]).max()
            assert err < 1e-3, f"{rid} t0 {t0}: cannot reproduce the panel label ({err})"
            j = np.clip((np.arange(T) + 0.5) * M / T - 0.5, 0, M - 1)
            lag[i] = np.interp(j, np.arange(M), sc) - (t0 + T_AX)
            ft = t0 + T_AX
            out = (ft < sc[0] - 0.5 / RATE) | (ft > sc[-1] + 0.5 / RATE)
            for r in range(R):
                label[i, r] = np.where(out, np.nan, np.interp(ft, sc, vc[r]))
    return label, lag, first


def rescore(true_label: np.ndarray) -> list[dict]:
    """DREGON MAE per regime: panel labels, panel labels on the frames that
    have telemetry, and true-time labels on those frames (regimes re-derived)."""
    import torch

    from metrics.regimes import frame_regimes

    filled = np.nan_to_num(true_label, nan=0.0)
    true_reg = frame_regimes(torch.from_numpy(filled).float(), RATE).numpy()
    known = ~np.isnan(true_label).any(axis=1)
    dreg = (DATASET == "dregon")[:, None]
    variants = (
        ("panel", TARGET, REGIME, np.ones((N, T), bool)),
        ("panel, telemetry frames only", TARGET, REGIME, known),
        ("true time", filled, true_reg, known),
    )
    rows = []
    for name in MODELS:
        for which, lab, reg, ok in variants:
            err = np.abs(runs[name]["pred"] - lab).mean(axis=1)
            row: dict = dict(model=name, labels=which, all=float(err[dreg & ok].mean()))
            for code in range(4):
                m = dreg & ok & (reg == code)
                row[REG[code]], row[f"n_{REG[code]}"] = float(err[m].mean()), int(m.sum())
            rows.append(row)
    return rows


def lag_summary(lag: np.ndarray, true_label: np.ndarray, first: dict) -> list[dict]:
    out = []
    for rid in DREGON:
        idx = [i for i in np.flatnonzero(rid == RECORDING) if META[i]["channel"] == 0]
        c0 = [i for i in idx if META[i]["t_start_s"] == 0.0][0]
        rest = [i for i in idx if i != c0]
        out.append(
            dict(
                recording=rid,
                first_stamp_s=first[rid],
                clip0_frames_without_telemetry=int(np.isnan(true_label[c0]).any(axis=0).sum()),
                clip0_max_lag_s=float(np.abs(lag[c0]).max()),
                other_clips_max_lag_s=float(np.abs(lag[rest]).max()),
            )
        )
    return out


def fig_lag(lag: np.ndarray) -> None:
    """Stretched-label time error along each DREGON recording (mic 0 clips)."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    for rid, colour in zip(DREGON, ("k", "#2ca02c", "#d62728"), strict=True):
        idx = sorted(
            (i for i in np.flatnonzero(rid == RECORDING) if META[i]["channel"] == 0),
            key=lambda i: META[i]["t_start_s"],
        )
        t = np.concatenate([META[i]["t_start_s"] + T_AX for i in idx])
        y = np.concatenate([lag[i] for i in idx])
        for ax in axes:
            ax.plot(t, y, color=colour, lw=1.2, label=f"{rid} ({DREGON[rid]})")
    axes[0].set_ylabel("lag (s)", fontsize=12)
    axes[0].set_title(
        "Recorded time of the telemetry sample the panel label shows, minus the frame time",
        fontsize=13,
    )
    axes[0].legend(fontsize=10)
    axes[1].set_ylim(-0.1, 0.1)
    axes[1].set_ylabel("lag (s), zoom", fontsize=12)
    axes[1].set_xlabel("s from the start of the recording (8 s clips)", fontsize=12)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "label_lag.png", dpi=110)
    plt.close(fig)


def whine_contrast(t: np.ndarray, f: np.ndarray, db: np.ndarray) -> np.ndarray:
    """Peak dB in 5.9-6.4 kHz over the median of its 5.3-5.8 / 6.5-7.0 kHz
    neighbours, 0.5 s running median: the motor whine near 6.1 kHz."""
    line = (f >= 5900) & (f <= 6400)
    side = ((f >= 5300) & (f < 5800)) | ((f > 6500) & (f <= 7000))
    c = db[line].max(axis=0) - np.median(db[side], axis=0)
    w = int(round(0.5 / (t[1] - t[0])))
    return np.array([np.median(c[max(0, n - w // 2) : n + w // 2 + 1]) for n in range(c.size)])


def start_figure(
    x: np.ndarray, title: str, tracks: list, first_stamp: float, slug: str, band_onset: bool
) -> dict:
    """The first 16 s of one recording ``x`` (mics, n): spectrograms, band levels,
    the 6.1 kHz whine and the label ``tracks`` (``(t, label (R, L), ls,
    legend)``; the first one also draws its harmonics, the second dashed in
    cyan). ``band_onset``: also mark where the 2-8 kHz level first rises 10 dB
    over its first second (only meaningful without a loudspeaker). Writes
    ``slug``.png/.wav, returns the audio onsets (s) found before take-off."""
    fig, axes = plt.subplots(
        5, 1, figsize=(16, 19), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1.3, 0.7, 0.7, 1])
    )
    tw, fw, dbw = mic_mean_db(x, 1024)
    axes[0].pcolormesh(
        tw,
        fw / 1000,
        dbw,
        vmin=np.percentile(dbw, 30),
        vmax=np.percentile(dbw, 99.9),
        cmap="inferno",
        shading="auto",
    )
    axes[0].set_ylabel("kHz (mic mean)", fontsize=12)
    axes[0].set_title(title, fontsize=14)
    t, f, db = mic_mean_db(x, 4096)
    keep = f <= 300
    axes[1].pcolormesh(
        t,
        f[keep],
        db[keep],
        vmin=np.percentile(db[keep], 30),
        vmax=np.percentile(db[keep], 99.9),
        cmap="inferno",
        shading="auto",
    )
    for (lt, lab, _, name), (colour, ls) in zip(
        tracks, (("white", ":"), ("cyan", "--")), strict=False
    ):
        harmonics(
            axes[1],
            lt,
            np.nan_to_num(lab.mean(0)),
            range(1, 7),
            color=colour,
            lw=1.1,
            ls=ls,
            label=f"{name}: rotor mean x k, k = 1..6",
        )
    axes[1].set_ylim(0, 300)
    axes[1].set_ylabel("Hz (mic mean)", fontsize=12)
    axes[1].legend(loc="upper left", fontsize=10, facecolor="0.25", labelcolor="white")
    lin = 10 ** (dbw / 10)
    hi_band = 10 * np.log10(lin[(fw >= 2000) & (fw <= 8000)].sum(axis=0))
    axes[2].plot(
        tw,
        10 * np.log10(lin[(fw >= 20) & (fw <= 200)].sum(axis=0)),
        color="k",
        lw=1.0,
        label="20-200 Hz",
    )
    axes[2].plot(tw, hi_band, color="#9467bd", lw=1.0, label="2000-8000 Hz")
    axes[2].set_ylabel("band level\n(dB, mic mean)", fontsize=11)
    axes[2].legend(loc="upper left", fontsize=10)
    whine = whine_contrast(tw, fw, dbw)
    axes[3].plot(tw, whine, color="#9467bd", lw=1.2)
    axes[3].axhline(WHINE_DB, color="k", lw=0.8, ls=":")
    axes[3].set_ylabel("6.1 kHz whine\n(dB over sides)", fontsize=11)
    take_t, take_lab = tracks[-1][0], np.nan_to_num(tracks[-1][1].mean(0))
    takeoff = float(take_t[np.argmax(take_lab > 20)])
    pre = tw < takeoff
    on = np.flatnonzero((whine > WHINE_DB) & pre)  # masked by a loudspeaker: None
    onsets = {"whine_onset_s": float(tw[on[0]]) if on.size else None, "band_onset_s": None}
    if band_onset:
        on = np.flatnonzero((hi_band > np.median(hi_band[tw < 1.0]) + 10) & pre)
        onsets["band_onset_s"] = float(tw[on[0]]) if on.size else None
    for lt, lab, ls, name in tracks:
        for r in range(R):
            axes[4].plot(lt, lab[r], color=ROTOR_COLORS[r], lw=1.8 if ls == "-" else 1.4, ls=ls)
        axes[4].plot([], [], color="0.3", lw=1.6, ls=ls, label=name)
    if first_stamp > 0:
        axes[4].axvspan(0, first_stamp, color="0.85", hatch="//", lw=0, label="no telemetry yet")
    marks = [(first_stamp, "0.4", "first telemetry stamp")]
    marks += [
        (onsets["whine_onset_s"], "#2ca02c", f"audio: whine above {WHINE_DB:.0f} dB"),
        (onsets["band_onset_s"], "#ff7f0e", "audio: 2-8 kHz 10 dB over its first second"),
    ]
    for at, colour, what in marks:
        if at is None or at < 0:
            continue
        for ax in axes:
            ax.axvline(at, color=colour, lw=1.3, ls="-.", label=what if ax is axes[4] else None)
    axes[4].set_ylabel("rotor rev/s", fontsize=12)
    axes[4].legend(loc="upper left", fontsize=10)
    axes[4].set_xlabel("s from the start of the recording", fontsize=12)
    axes[0].set_xlim(0, 16)
    fig.tight_layout()
    fig.savefig(HERE / f"{slug}.png", dpi=110)
    plt.close(fig)
    return dict(
        slug=slug, wav=write_wav(x[0], slug), first_stamp_s=first_stamp, takeoff_s=takeoff, **onsets
    )


def label_onsets(lt: np.ndarray, lab: np.ndarray, thr: float = 1.0) -> list[float]:
    """Per rotor, the first label time above ``thr`` rev/s."""
    return [
        round(float(lt[np.argmax(np.nan_to_num(lab[r]) > thr)]), 2) for r in range(lab.shape[0])
    ]


def fig_start(rid: str, true_label: np.ndarray, first_stamp: float, slug: str) -> dict:
    """The first 16 s of one DREGON panel recording: the panel label (stretched)
    against the same telemetry at its recorded stamps."""
    x, lt, label, _ = recording(rid, until_s=16.0)
    idx = sorted(
        (
            i
            for i, m in enumerate(META)
            if m["recording_id"] == rid and m["channel"] == 0 and m["t_start_s"] < 16.0
        ),
        key=lambda i: META[i]["t_start_s"],
    )
    true = np.concatenate([true_label[i] for i in idx], axis=1)
    out = start_figure(
        x,
        f"{rid} ({DREGON[rid]}): the first 16 s",
        [
            (lt, label, "-", "panel label (stretched over the clip)"),
            (lt, true, "--", "the same telemetry at its recorded stamps"),
        ],
        first_stamp,
        slug,
        band_onset=rid == "free-flight_nosource_room1",
    )
    return dict(
        recording=rid,
        panel_idle_onset_s=label_onsets(lt, label),
        true_idle_onset_s=label_onsets(lt, true),
        **out,
    )


TRAIN_STREAM = ROOT / "conf/online_mix/rp_all.yaml"
IDLE_RPS = 15.0  # DREGON idle is 15.3 rev/s on motors_measured (room 1)


def training_starts() -> list[dict]:
    """The first 16 s of every real recording the rp training streams draw from,
    labelled exactly as the stream labels a chunk cut there."""
    from omegaconf import OmegaConf

    from data_processing import online_mixing as om
    from data_processing.frames import get_meta
    from data_processing.mixing import resolve_motor_tracks

    cfg = cast(dict[str, Any], OmegaConf.to_container(OmegaConf.load(TRAIN_STREAM), resolve=True))
    out = []
    for spec in (s for s in cfg["sources"]["noise"] if s["kind"] == "frames"):
        for rec in om._load_real_records(spec, sample_rate=SR, window_s=16.0):
            tf = rec["tf"]
            rid = str(get_meta(tf, "recording_id", "?"))
            a0 = float(tf["audio"].t_start)
            chunk = om._cut_record_window(rec, 16.0, 0.0)
            t0 = float(chunk["audio"].t_start) - a0
            assert abs(t0) < 0.02, (
                f"{rid}: the stream's span starts at {t0:.2f} s, not at the audio"
            )
            x = np.asarray(chunk["audio"].data, dtype=np.float64)
            n = x.shape[1] // 512 + 1
            lab = om.interpolate_rps_to_stft_grid(chunk, n_frames=n, hop_length=512)
            lt = np.arange(n) * 512 / SR
            _, key, _ = resolve_motor_tracks(tf)
            first = float(np.asarray(tf[key].tindex.abs_stamps, dtype=np.float64)[0]) - a0
            # Before the first stamp a 2 s training window gets 0 (wholly before it)
            # or the first sample clamped (straddling it): the label is undefined there.
            lab[:, lt < first] = np.nan
            track = "motors_command" if spec["dataset"] == "DREGON-frames" else "calibrated rps"
            fig = start_figure(
                x,
                f"{rid} (rp training stream): the first 16 s",
                [(lt, lab, "-", f"training label ({track})")],
                max(first, 0.0),
                f"train_{rid.replace('-', '_')}",
                band_onset=True,
            )
            idle = lt[np.argmax(lab.mean(0) >= IDLE_RPS)]
            out.append(
                {
                    **fig,
                    "recording": rid,
                    "track": track,
                    "first_stamp_s": first,
                    "label_onset_s": label_onsets(lt, lab),
                    "idle_reached_s": float(idle),
                    "span_s": float(rec["valid_end"]) - float(rec["valid_start"]),
                }
            )
            print("built", fig["slug"], flush=True)
    return out


def fig_command_vs_measured() -> list[dict]:
    """Room 1 carries both rotor tracks: the tachometer and the command. The
    rp training streams only have the command (room 2 publishes nothing else)."""
    from data_processing.derivations import PARENTS, _split_dload_uri
    from data_processing.frames import get_meta
    from data_processing.streams import iter_published_frames

    fig, axes = plt.subplots(len(DREGON), 1, figsize=(16, 11), sharex=True)
    rows = []
    for fr in iter_published_frames(*_split_dload_uri(PARENTS["DREGON-frames"])):
        rid = str(get_meta(fr, "recording_id", ""))
        if rid not in DREGON:
            continue
        ax, row = axes[list(DREGON).index(rid)], cast(dict[str, Any], dict(recording=rid))
        a0 = float(fr["audio"].t_start)
        for key, ls in (("motors_measured", "-"), ("motors_command", "--")):
            st = np.asarray(fr[key].tindex.abs_stamps, dtype=np.float64) - a0
            v = np.asarray(fr[key].data, dtype=np.float64)
            m = st < 16.0
            for r in range(R):
                ax.plot(st[m], v[r, m], color=ROTOR_COLORS[r], ls=ls, lw=1.8 if ls == "-" else 1.3)
            ax.plot([], [], color="0.3", ls=ls, label=key)
            row[f"{key}_above_1_s"] = float(st[np.argmax(v.mean(0) > 1)])
            row[f"{key}_reaches_idle_s"] = float(st[np.argmax(v.mean(0) >= IDLE_RPS)])
        ax.set_title(f"{rid} ({DREGON[rid]})", fontsize=13)
        ax.set_ylabel("rev/s", fontsize=12)
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(alpha=0.3)
        rows.append(row)
    axes[-1].set_xlabel("s from the start of the recording (recorded stamps)", fontsize=12)
    axes[-1].set_xlim(4, 16)
    fig.tight_layout()
    fig.savefig(HERE / "command_vs_measured.png", dpi=110)
    plt.close(fig)
    return sorted(rows, key=lambda r: list(DREGON).index(r["recording"]))


def worst(rid: str, code: int, exclude=()) -> int:
    """The panel item of ``rid`` with prior_v1's largest error in regime ``code``."""
    err = runs["prior_v1"]["frame_err"]
    cand = [
        i
        for i, m in enumerate(META)
        if m["recording_id"] == rid
        and (REGIME[i] == code).sum() >= 16
        and m["clip_index"] not in exclude
    ]
    return max(cand, key=lambda i: float(err[i][REGIME[i] == code].mean()))


def examples(true_label: np.ndarray) -> list[dict]:
    a = worst("free-flight_whitenoise-low_room1", 0)
    s = worst("FLY124", 1)
    picks = [
        (a, 0, "DREGON stopped, loudspeaker white noise", "ex_dregon_stopped_whitenoise"),
        (
            worst("free-flight_speech-low_room1", 0),
            0,
            "DREGON stopped, loudspeaker speech",
            "ex_dregon_stopped_speech",
        ),
        (
            worst("free-flight_nosource_room1", 1),
            1,
            "DREGON standby, no loudspeaker",
            "ex_dregon_standby_nosource",
        ),
        (
            worst("free-flight_whitenoise-low_room1", 1, exclude=(META[a]["clip_index"],)),
            1,
            "DREGON standby and take-off, loudspeaker white noise",
            "ex_dregon_standby_whitenoise",
        ),
        (s, 1, "MD2 standby", "ex_md2_standby"),
        (worst("FLY124", 2, exclude=(META[s]["clip_index"],)), 2, "MD2 ramp", "ex_md2_ramp"),
    ]
    out = []
    err = runs["prior_v1"]["frame_err"]
    for i, code, why, slug in picks:
        m = META[i]
        c = float(err[i][REGIME[i] == code].mean())
        title = (
            f"{why}: {m['recording_id']} t0 {m['t_start_s']:.0f} s, mic {m['channel']}; "
            f"prior_v1 {REG[code]} MAE {c:.1f}"
        )
        fig_example(i, title, slug, true_label[i])
        out.append(dict(slug=slug, title=title, wav=write_wav(audio(i), slug)))
        print("built", slug, flush=True)
    return out


def main() -> None:
    scores = json.loads(SCORES.read_text())
    fig_octave_map()
    fig_ratio_hist()
    tone = fig_stopped()
    true_label, lag, first = true_time_labels()
    index = dict(
        scores=scores, breakdown=breakdown(), room_tone_hz=tone, examples=examples(true_label)
    )
    index["lag"] = lag_summary(lag, true_label, first)
    fig_lag(lag)
    index["rescore"] = rescore(true_label)
    index["starts"] = []
    for rid in DREGON:
        slug = f"start_{rid.replace('-', '_')}"
        index["starts"].append(fig_start(rid, true_label, first[rid], slug))
        print("built", slug, flush=True)
    index["command_vs_measured"] = fig_command_vs_measured()
    index["training"] = training_starts()
    (HERE / "index.json").write_text(json.dumps(index, indent=1) + "\n")


if __name__ == "__main__":
    main()
