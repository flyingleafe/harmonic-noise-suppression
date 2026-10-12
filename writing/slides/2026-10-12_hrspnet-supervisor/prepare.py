#!/usr/bin/env python
"""Assets for the HRSPNet supervisor deck (2026-10-12).

Everything is computed from local data and committed numbers; nothing is
trained. Run from the project root environment:

    python writing/slides/2026-10-12_hrspnet-supervisor/prepare.py

Writes assets/*.png. Numbers in the bar charts are copied from the sources
named next to them (docs/handoff-pyramid-harmonic.md, the paper draft's
tables, docs/experiments/cross-drone-generalization-fly125.md).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from experiments import rps_bench as rb  # noqa: E402
from metrics._common import get_array  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PAPER = ROOT / "writing/papers/2026-08_wrapup"
ASSETS = HERE / "assets"
ROTOR_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
FS, HOP, FR = 16000, 512, 31.25
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def frame_arrays(i: int) -> tuple[np.ndarray, np.ndarray]:
    f = rb.part("real")[i]
    au = np.asarray(get_array(f, "mixture"))
    au = au[0] if au.ndim == 2 else au
    return au.astype(np.float32), np.asarray(get_array(f, "rps"), dtype=np.float64)


def stft_db(audio: np.ndarray, n_fft: int) -> np.ndarray:
    x = torch.as_tensor(audio)
    s = (
        torch.stft(
            x,
            n_fft,
            hop_length=HOP,
            win_length=n_fft,
            window=torch.hann_window(n_fft),
            center=True,
            return_complex=True,
        )
        .abs()
        .numpy()
    )
    return 20 * np.log10(s + 1e-6)


# ── 1. the signal: spectrogram with the telemetry comb overlaid ────────────────
def spectrogram_comb(i: int = 24) -> None:
    au, rps = frame_arrays(i)
    n_fft = 16384
    s = stft_db(au, n_fft)
    f = np.arange(s.shape[0]) * FS / n_fft
    keep = f <= 800
    t = np.arange(s.shape[1]) / FR
    tr = np.arange(rps.shape[1]) / FR
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1]})
    for ax in axes:
        ax.imshow(
            s[keep],
            origin="lower",
            aspect="auto",
            cmap="magma",
            extent=(t[0], t[-1], 0, 800),
            vmin=np.percentile(s[keep], 5),
            vmax=np.percentile(s[keep], 99.5),
        )
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("frequency (Hz)")
    axes[0].set_title("one microphone, DREGON cruise, 8 s")
    for r_i, c in zip(rps, ROTOR_COLORS):
        for k in range(1, 11):
            axes[1].plot(tr, k * r_i, color=c, lw=0.9, alpha=0.9)
    axes[1].set_ylim(0, 800)
    axes[1].set_title(r"harmonics $k\,r_i(t)$ from telemetry, one colour per rotor")
    axes[1].set_yticks([])
    fig.tight_layout()
    fig.savefig(ASSETS / "spectrogram_comb.png", dpi=170)
    plt.close(fig)


# ── 2. rotors are not notes: real tracks vs a piano-roll cartoon ──────────────
def tracks_vs_pianoroll() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), gridspec_kw={"width_ratios": [1, 1.2, 1.2]})
    ax = axes[0]
    notes = [
        (0.2, 1.1, 60),
        (0.6, 1.6, 64),
        (1.3, 2.4, 67),
        (1.9, 3.0, 72),
        (2.6, 3.4, 65),
        (3.1, 3.9, 71),
    ]
    for s0, s1, p in notes:
        ax.add_patch(plt.Rectangle((s0, p - 0.4), s1 - s0, 0.8, color="#444"))
    for p in range(58, 75):
        ax.axhline(p, color="0.85", lw=0.5, zorder=0)
    ax.set_xlim(0, 4)
    ax.set_ylim(57.5, 74.5)
    ax.set_yticks([60, 64, 67, 72])
    ax.set_yticklabels(["C4", "E4", "G4", "C5"])
    ax.set_xlabel("time (s)")
    ax.set_title("music: notes on a grid, on/off")
    for ax, i, title in (
        (axes[1], 8, "DREGON take-off and cruise"),
        (axes[2], 216, "MD2 cruise with manoeuvres"),
    ):
        _, rps = frame_arrays(i)
        t = np.arange(rps.shape[1]) / FR
        for r_i, c in zip(rps, ROTOR_COLORS):
            ax.plot(t, r_i, color=c, lw=1.4)
        ax.set_xlabel("time (s)")
        ax.set_title(title)
    axes[1].set_ylabel("rotor speed (rev/s)")
    fig.tight_layout()
    fig.savefig(ASSETS / "tracks_vs_pianoroll.png", dpi=170)
    plt.close(fig)


# ── 3. encode -> decode through a shared log-axis salience map ────────────────
def encode_decode(i: int = 24) -> None:
    _, rps = frame_arrays(i)
    bpo, fmin, nbins = 48, 27.5, 352
    grid = fmin * 2 ** (np.arange(nbins) / bpo)
    T = rps.shape[1]
    sal = np.zeros((nbins, T))
    for r_i in rps:
        pos = bpo * np.log2(np.maximum(r_i, fmin) / fmin)
        sal = np.maximum(sal, np.exp(-0.5 * (np.arange(nbins)[:, None] - pos[None]) ** 2))
    dec = np.full((4, T), np.nan)
    prev = np.full(4, np.nan)
    for tt in range(T):
        col = sal[:, tt]
        peaks = [
            b
            for b in range(1, nbins - 1)
            if col[b] > 0.5 and col[b] >= col[b - 1] and col[b] >= col[b + 1]
        ]
        vals = np.array([grid[b] for b in peaks], dtype=np.float64)
        cur = np.full(4, np.nan)
        if np.isfinite(prev).sum() >= 1 and len(vals):
            # identity by nearest previous track; a merged pair yields one peak, so one track is lost
            cost = np.abs(np.nan_to_num(prev, nan=1e6)[:, None] - vals[None, :])
            ri, ci = linear_sum_assignment(cost)
            cur[ri] = vals[ci]
        else:
            cur[: min(4, len(vals))] = np.sort(vals)[:4]
        dec[:, tt] = cur
        prev = np.where(np.isnan(cur), prev, cur)
    err = np.nanmean(np.abs(np.sort(dec, 0) - np.sort(rps, 0)))
    t = np.arange(T) / FR
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    lo, hi = 60, 100
    sel = (grid >= lo) & (grid <= hi)
    axes[0].imshow(
        sal[sel],
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=(t[0], t[-1], grid[sel][0], grid[sel][-1]),
    )
    axes[0].set_title("telemetry encoded as ONE log-axis map (48 bins/oct)", fontsize=10)
    axes[0].set_ylabel("rev/s")
    axes[0].set_xlabel("time (s)")
    for r_i, d_i, c in zip(rps, dec, ROTOR_COLORS):
        axes[1].plot(t, r_i, color="0.6", lw=1)
        axes[1].plot(t, d_i, color=c, lw=1.2)
    axes[1].set_ylim(lo, hi)
    axes[1].set_title("peak-pick + assign: a close pair becomes one rotor", fontsize=10)
    axes[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(ASSETS / "encode_decode.png", dpi=170)
    plt.close(fig)
    print(f"encode-decode floor on frame {i}: {err:.2f} rev/s")


# ── 4. how fast rotors move between bins ──────────────────────────────────────
def slew_hist() -> None:
    frames = rb.part("real")
    d = []
    for i in range(0, len(frames), 8):
        rps = np.asarray(get_array(frames[i], "rps"), dtype=np.float64)
        on = rps.mean(0) > 30
        diff = np.abs(np.diff(rps, axis=-1))[:, on[1:] & on[:-1]]
        d.append(diff.ravel())
    d = np.concatenate(d)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    bins = np.logspace(-2, np.log10(30), 50)
    ax.hist(d, bins=bins, color="#555")
    ax.set_xscale("log")
    for q, lab in ((50, "median"), (99, "99th pct")):
        v = np.percentile(d, q)
        ax.axvline(v, color="#d62728", lw=1)
        ax.text(
            v,
            ax.get_ylim()[1] * (0.9 if q == 99 else 0.25),
            f" {lab}\n {v:.2f}",
            color="#d62728",
            fontsize=9,
            va="top",
        )
    ax.axvline(0.977, color="#1f77b4", ls="--", lw=1)
    ax.text(0.977, ax.get_ylim()[1] * 0.5, " one bin\n (0.98 rev/s)", color="#1f77b4", fontsize=9)
    ax.set_xlabel("speed change per frame (rev/s), validation telemetry, motors on")
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(ASSETS / "slew_hist.png", dpi=170)
    plt.close(fig)


# ── 5. rate-synchronous spectrum of one rotor: where the motor lines sit ──────
def rate_sync_spectrum(i: int = 24, rotor: int = 0) -> None:
    au, rps = frame_arrays(i)
    n_fft = 16384
    s = stft_db(au, n_fft)
    f = np.arange(s.shape[0]) * FS / n_fft
    orders = np.arange(0.5, 90, 0.02)
    acc = np.zeros_like(orders)
    n = 0
    for tt in range(s.shape[1]):
        r = rps[rotor, min(tt, rps.shape[1] - 1)]
        if r < 40:
            continue
        acc += np.interp(orders * r, f, s[:, tt])
        n += 1
    acc /= max(n, 1)
    # floor: running median over 1.5 orders
    k = int(1.5 / 0.02)
    floor = np.array([np.median(acc[max(0, j - k) : j + k]) for j in range(len(acc))])
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(orders, acc - floor, color="#333", lw=0.8)
    for m in (14, 42, 70):
        ax.axvline(m, color="#d62728", lw=1, ls="--")
        ax.text(
            m, ax.get_ylim()[1] * 0.95, f" k={m}\n motor", color="#d62728", fontsize=9, va="top"
        )
    ax.axvspan(0.5, 9, color="#1f77b4", alpha=0.12)
    ax.text(
        1, ax.get_ylim()[1] * 0.95, " HPPNet reads k ≤ 9", color="#1f77b4", fontsize=9, va="top"
    )
    ax.set_xlim(0.5, 90)
    ax.set_xlabel(
        "harmonic order k (frequency / rotor speed), one DREGON rotor, 8 s cruise, 16384-point STFT"
    )
    ax.set_ylabel("level above local floor (dB)")
    fig.tight_layout()
    fig.savefig(ASSETS / "rate_sync_spectrum.png", dpi=170)
    plt.close(fig)


# ── 6. resolution in rate units vs order, per front end ───────────────────────
def resolution_curves(r: float = 80.0, slew: float = 3.0) -> None:
    k = np.arange(1, 85)
    f = k * r
    cqt = 0.0145 * r * np.ones_like(k, dtype=float)
    stft = (FS / 2048) / k
    lvl = np.clip(np.ceil(np.log2(np.maximum(f / 1200, 1))), 0, 3)
    pyr = (FS / 16384) * 2**lvl / k
    opt = 2 * np.sqrt(slew / k)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(k, cqt, label="CQT (Q = 68.8)", color="#2ca02c")
    ax.plot(k, stft, label="single STFT, 2048", color="#7f7f7f")
    ax.plot(k, pyr, label="STFT pyramid", color="#1f77b4", lw=2)
    ax.plot(
        k, opt, label=r"optimum $2\sqrt{\dot r/k}$ at $\dot r$ = 3 rev/s²", color="#d62728", ls="--"
    )
    ax.set_yscale("log")
    ax.set_xlabel("harmonic order k (rotor at 80 rev/s)")
    ax.set_ylabel("rate resolution (rev/s)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "resolution_curves.png", dpi=170)
    plt.close(fig)


# ── 7. bar charts from committed numbers ──────────────────────────────────────
def bars(
    path: str,
    groups: list[tuple[str, float, str]],
    title: str,
    ylabel: str = "matched MAE (rev/s)",
    note: str | None = None,
    figsize=(6.5, 3.4),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(groups))
    vals = [g[1] for g in groups]
    ax.bar(x, vals, color=[g[2] for g in groups], width=0.65)
    for xi, v in zip(x, vals):
        ax.text(xi, v + max(vals) * 0.015, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    if note:
        ax.text(
            0.99, 0.97, note, transform=ax.transAxes, ha="right", va="top", fontsize=8, color="0.35"
        )
    fig.tight_layout()
    fig.savefig(ASSETS / path, dpi=170)
    plt.close(fig)


def all_bars() -> None:
    g, b, o = "#7f7f7f", "#1f77b4", "#ff7f0e"
    # source: docs/handoff-pyramid-harmonic.md sec. 2 (unified panel, val/real_overall, one seed);
    # paper tab:real (regressors, HPPNet-L2 2.27 frozen split); nv2_mixed_hppnet_l2 2.25 on the panel
    bars(
        "bars_current.png",
        [
            ("Conv+BiGRU\n(regressor)", 2.99, g),
            ("HPPNet-L2\n(CQT, per-rotor out)", 2.27, g),
            ("CQT k≤9\n+ rate ConvLSTM", 2.16, b),
            ("pyramid k≤32\n+ rate ConvLSTM", 2.17, b),
        ],
        "real-data training only, all validation frames",
        note="no synthetic pretraining; one seed",
    )
    # source: paper tab:output / appendix sec:app-salience
    bars(
        "bars_output.png",
        [
            ("HPPNet\npublished", 7.77, g),
            ("HPPNet\nper-rotor", 2.27, b),
            ("HarmoF0\npublished", 10.8, g),
            ("HarmoF0\nper-rotor", 2.45, b),
            ("LateDeep\npublished", 12.6, g),
            ("LateDeep\nper-rotor", 3.83, b),
            ("Basic Pitch\npublished", 27.3, g),
            ("Basic Pitch\nper-rotor", 27.6, g),
        ],
        "output representation: published salience map vs four per-rotor layers",
        figsize=(9, 3.4),
    )
    # source: docs/handoff-pyramid-harmonic.md sec. 3 (tap ablation, no retraining)
    fig, axes = plt.subplots(3, 1, figsize=(5.6, 6.0), sharex=True)
    lad = ["none", "k ≥ 61", "k ≥ 31", "k ≥ 16"]
    pyr = {
        "all": [2.68, 2.22, 3.31, 4.65],
        "DREGON": [3.33, 2.59, 3.59, 5.65],
        "MD2": [1.73, 1.69, 2.90, 3.20],
    }
    cqt = {
        "all": [2.31, 2.31, 4.53, 11.1],
        "DREGON": [2.70, 2.58, 5.79, 17.3],
        "MD2": [1.74, 1.92, 2.68, 2.03],
    }
    x = np.arange(4)
    for ax, ds in zip(axes, ("all", "DREGON", "MD2")):
        ax.bar(x - 0.18, pyr[ds], width=0.34, color=b, label="pyramid, k ≤ 84")
        ax.bar(x + 0.18, cqt[ds], width=0.34, color=g, label="CQT, k ≤ 84")
        for xi, (vp, vc) in enumerate(zip(pyr[ds], cqt[ds])):
            ax.text(xi - 0.18, min(vp, 7.4) + 0.12, f"{vp:.2f}", ha="center", fontsize=8)
            ax.text(xi + 0.18, min(vc, 7.4) + 0.12, f"{vc:.2f}", ha="center", fontsize=8)
        ax.set_ylim(0, 8.3)
        ax.set_ylabel("MAE (rev/s)", fontsize=9)
        ax.text(0.01, 0.92, ds, transform=ax.transAxes, fontsize=10, va="top")
    axes[0].legend(fontsize=8, frameon=False, loc="upper center", ncol=2)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(["taps zeroed: " + lad[0]] + lad[1:], fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "bars_taps.png", dpi=170)
    plt.close(fig)
    # source: docs/experiments/cross-drone-generalization-fly125.md (PIT RMSE on FLY124 cruise frames)
    bars(
        "bars_transfer.png",
        [
            ("trained on DREGON only", 7.96, g),
            ("+ one MD2 flight (FLY125)", 1.63, b),
        ],
        "regressor scored on an unseen MD2 flight (FLY124), cruise",
        ylabel="PIT RMSE (rev/s)",
        note="SimpleConv, 2026-06",
        figsize=(5, 3.4),
    )
    # source: paper tab:nv2 (docs/experiments/noise-v2-transfer.md)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    arms = [
        "real only",
        "synthetic\neasy",
        "synthetic\nhard",
        "mixed",
        "easy →\nfine-tune",
        "hard →\nfine-tune",
    ]
    reg = [2.99, 7.94, 7.06, 2.53, 2.73, 2.23]
    hpp = [2.27, 6.46, 6.47, 2.25, 2.11, np.nan]
    for ax, name, v in (
        (axes[0], "Conv+BiGRU (regressor)", reg),
        (axes[1], "HPPNet-L2 (CQT, per-rotor out)", hpp),
    ):
        x = np.arange(6)
        cols = [g, o, o, b, b, b]
        ax.bar(x, np.nan_to_num(v), color=cols, width=0.65)
        for xi, vv in zip(x, v):
            ax.text(
                xi,
                (vv if np.isfinite(vv) else 0.2) + 0.1,
                f"{vv:.2f}" if np.isfinite(vv) else "unmeasured",
                ha="center",
                fontsize=8,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(arms, fontsize=8)
        ax.set_title(name, fontsize=11)
    axes[0].set_ylabel("matched MAE (rev/s)")
    fig.suptitle(
        "pretraining on the fitted generative model (previous models; HRSPNet arms pending)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "bars_pretrain.png", dpi=170)
    plt.close(fig)


# ── 8. TikZ assets compiled standalone ────────────────────────────────────────
STANDALONE = r"""\documentclass[border=3pt]{standalone}
\usepackage{tikz,graphicx,amsmath,amssymb}
\usetikzlibrary{positioning,arrows.meta,calc,fit,backgrounds}
\newcommand{\revps}{\ensuremath{\,\mathrm{rev/s}}}
\graphicspath{{%s/}}
\begin{document}
%s
\end{document}
"""


def compile_tikz(name: str, body: str, graphics_dir: Path) -> None:
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / f"{name}.tex"
        src.write_text(STANDALONE % (graphics_dir.as_posix(), body))
        subprocess.run(["tectonic", src.name], cwd=td, check=True, capture_output=True)
        subprocess.run(
            ["pdftoppm", "-r", "220", "-png", "-singlefile", f"{name}.pdf", name],
            cwd=td,
            check=True,
        )
        shutil.copy(Path(td) / f"{name}.png", ASSETS / f"{name}.png")


def tikz_from_paper(label: str) -> str:
    """The tikzpicture of the paper figure carrying ``label``, refs flattened."""
    tex = (PAPER / "src/main.tex").read_text()
    figs = re.findall(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", tex, flags=re.S)
    fig = next(f for f in figs if f"\\label{{{label}}}" in f)
    start = fig.index("\\begin{tikzpicture}")
    end = fig.rindex("\\end{tikzpicture}") + len("\\end{tikzpicture}")
    pic = fig[start:end]
    pic = re.sub(r"\s*\(Section~\\ref\{[^}]*\}\)", "", pic)
    pic = re.sub(r"Section~\\ref\{[^}]*\}", "the paper", pic)
    return pic


HPDC_FANS = r"""
\begin{tikzpicture}[font=\small,
  lvl/.style={draw, fill=gray!8, minimum height=6mm, inner sep=0, anchor=south west},
  tap/.style={-{Latex[length=1.2mm]}, thin, blue!70!black},
  tap2/.style={-{Latex[length=1.2mm]}, thin, red!70!black}]
\def\bw{7.0}\def\gval{200}
\foreach \l/\y/\rng/\dfr in {0/0/{0--1.2 kHz}/{1}, 1/0.9/{0--2.4 kHz}/{2}, 2/1.8/{0--4.8 kHz}/{4}, 3/2.7/{0--8 kHz}/{8}} {
  \node[lvl, minimum width=\bw cm] (L\l) at (3.2,\y) {};
  \node[anchor=west] at (L\l.east) {level \l: \rng, $\Delta f = \dfr$ Hz};
  \pgfmathsetmacro{\range}{1200*2^\l}
  \foreach \k in {1,...,40} {
    \pgfmathsetmacro{\xx}{\k*\gval/\range}
    \ifdim \xx pt < 0.98pt
      \draw[gray!60, very thin] ({3.2+\bw*\xx},\y+0.08) -- ({3.2+\bw*\xx},\y+0.52);
    \fi
  }
}
\draw[-{Latex[length=1.4mm]}] (1.0,-0.1) -- (1.0,3.4) node[anchor=south] {candidate rate $g$};
\coordinate (g1) at (1.0,0.5); \coordinate (g2) at (1.0,2.2);
\fill[blue!70!black] (g1) circle (1.5pt) node[anchor=east, blue!70!black] {$g$};
\fill[red!70!black] (g2) circle (1.5pt) node[anchor=east, red!70!black] {$2g$};
\foreach \k/\l in {1/0, 2/0, 3/0, 5/0, 8/1, 16/2, 20/2} {
  \pgfmathsetmacro{\range}{1200*2^\l}\pgfmathsetmacro{\xx}{\k*\gval/\range}\pgfmathsetmacro{\yy}{0.9*\l+0.3}
  \draw[tap] (g1) -- ({3.2+\bw*\xx},\yy);
}
\foreach \k/\l in {1/0, 2/0, 4/1, 5/1, 8/2, 10/2} {
  \pgfmathsetmacro{\range}{1200*2^\l}\pgfmathsetmacro{\xx}{2*\k*\gval/\range}\pgfmathsetmacro{\yy}{0.9*\l+0.3}
  \draw[tap2] (g2) -- ({3.2+\bw*\xx},\yy);
}
\node[anchor=north west, align=left, text width=9.5cm] at (1.0,-0.4)
  {tap $k$ of candidate $g$ reads the fused map at $k g$ Hz from the finest level which holds it;
   the dilation grows with $g$ (compare $g$ and $2g$), the weight $W_k$ is shared across $g$};
\end{tikzpicture}
"""

LSTM_CARTOON = r"""
\begin{tikzpicture}[font=\small, cell/.style={draw, minimum size=5mm, inner sep=0}]
% left: FreqGroupLSTM, one recurrence per bin
\foreach \t in {0,...,6} { \foreach \b in {0,...,4} { \node[cell, fill=gray!8] at (\t*0.55,\b*0.55) {}; } }
\foreach \b in {0,...,4} { \draw[-{Latex[length=1mm]}, gray!30!black] (0.3,\b*0.55) -- (3.0,\b*0.55); }
\draw[thick, red!70!black] (0,0.55) .. controls (1.2,0.6) and (2.0,1.9) .. (3.3,2.2);
\node[anchor=north, align=center, text width=5.2cm, font=\footnotesize] at (1.65,-0.45) {FreqGroupLSTM: the state stays in its bin; a moving rotor leaves it after $\sim$2 frames};
% right: rate ConvLSTM
\begin{scope}[xshift=6cm]
\foreach \t in {0,...,6} { \foreach \b in {0,...,4} { \node[cell, fill=gray!8] at (\t*0.55,\b*0.55) {}; } }
\foreach \t/\b in {0/0.1, 1/0.2, 2/0.5, 3/0.9, 4/1.4, 5/1.8, 6/2.1} {
  \draw[fill=blue!20, draw=blue!60!black] (\t*0.55-0.2,\b-0.5) rectangle (\t*0.55+0.2,\b+0.5);
}
\draw[thick, red!70!black] (0,0.55) .. controls (1.2,0.6) and (2.0,1.9) .. (3.3,2.2);
\draw[-{Latex[length=1.2mm]}, blue!60!black, thick] (0,0.1) -- (3.3,2.1);
\node[anchor=north, align=center, text width=5.2cm, font=\footnotesize] at (1.65,-0.45) {rate ConvLSTM: gates convolve along rate; the state reaches $\pm 16$ bins per frame and follows the rotor};
\end{scope}
\end{tikzpicture}
"""


# ── 9. CQT vs pyramid on an MD2 and a DREGON clip ─────────────────────────────
def cqt_vs_pyramid() -> None:
    import librosa

    rows = [(264, "MD2 (FLY124), no wind at the ring"), (24, "DREGON room 1, in the downwash")]
    fig, axes = plt.subplots(
        2, 5, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.3, 1, 1, 1, 1]}
    )
    for row, (i, title) in enumerate(rows):
        au, _ = frame_arrays(i)
        c = librosa.cqt(au, sr=FS, hop_length=HOP, fmin=27.5, n_bins=352, bins_per_octave=48)
        cdb = 20 * np.log10(np.abs(c) + 1e-6)
        ax = axes[row, 0]
        ax.imshow(
            cdb,
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=np.percentile(cdb, 5),
            vmax=np.percentile(cdb, 99.5),
        )
        yt = [0, 48, 96, 144, 192, 240, 288, 336]
        ax.set_yticks(yt)
        ax.set_yticklabels([f"{27.5 * 2 ** (b / 48):.0f}" for b in yt], fontsize=8)
        ax.set_xticks([])
        ax.set_ylabel("Hz (log)", fontsize=9)
        ax.set_title(f"CQT, 48 bins/oct\n{title}", fontsize=10)
        for lvl, (n_fft, f_hi) in enumerate(
            [(16384, 1200), (8192, 2400), (4096, 4800), (2048, 8000)]
        ):
            s = stft_db(au, n_fft)
            f = np.arange(s.shape[0]) * FS / n_fft
            keep = f <= f_hi
            ax = axes[row, lvl + 1]
            ax.imshow(
                s[keep],
                origin="lower",
                aspect="auto",
                cmap="magma",
                vmin=np.percentile(s[keep], 5),
                vmax=np.percentile(s[keep], 99.5),
                extent=(0, 8, 0, f_hi / 1000),
            )
            ax.set_xticks([])
            ax.tick_params(labelsize=8)
            if row == 0:
                ax.set_title(
                    f"pyramid level {lvl}: 0–{f_hi / 1000:g} kHz, Δf = {FS / n_fft:.3g} Hz",
                    fontsize=9,
                )
            if lvl == 0:
                ax.set_ylabel("kHz (linear)", fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "cqt_vs_pyramid.png", dpi=150)
    plt.close(fig)


ARCH_HORIZONTAL = r"""
\begin{tikzpicture}[font=\small,
  img/.style={inner sep=0, outer sep=0, anchor=south west, draw=black!50, line width=0.2pt},
  blk/.style={draw, rounded corners=1.5pt, align=center, inner sep=2.5pt, fill=white, font=\small, minimum height=9mm, text width=1.75cm},
  new/.style={blk, fill=blue!8, draw=blue!50!black},
  shp/.style={font=\footnotesize, text=gray!30!black, align=center},
  big/.style={-{Triangle[length=2mm,width=2.4mm]}, line width=1.4pt, magenta!55!black},
  tap/.style={-{Latex[length=0.9mm]}, very thin, blue!70!black}]
\def\dx{0.18}\def\dy{0.15}
% ================= row 1: input, per-level convolutions, HPDC =================
\foreach \l in {3,2,1,0} {
  \pgfmathsetmacro{\ox}{\l*\dx}\pgfmathsetmacro{\oy}{1.1+\l*\dy}
  \node[img] (L\l) at (\ox,\oy) {\includegraphics[width=1.9cm,height=1.2cm]{level\l.png}};
}
\node[shp, anchor=north] at (1.2,0.95) {STFT pyramid\\ 4 levels, hop 512\\ $T \times F_\ell$};
\node[blk] (b1) at (4.1,1.9) {conv $7{\times}7$\\ $1 \to 16$ ch};
\node[blk] (fu) at (6.25,1.9) {top-down fusion\\ $\times 2\uparrow$, $1{\times}1$, add};
\node[blk] (b2) at (8.4,1.9) {conv $7{\times}7$\\ $16 \to 16$};
\node[blk] (b25) at (10.55,1.9) {conv $7{\times}7$\\ $16 \to 16$};
\node[shp, anchor=north] at (7.3,1.2) {per level, weights shared across levels, $T \times F_\ell \times 16$};
\draw[big] (3.0,1.9) -- (b1.west);
\draw[big] (b1) -- (fu); \draw[big] (fu) -- (b2); \draw[big] (b2) -- (b25);
% HPDC: frequency strips -> rate strip
\begin{scope}[shift={(12.2,0.5)}]
\foreach \l/\x/\h in {0/0/2.8, 1/0.4/2.8, 2/0.8/2.8, 3/1.2/2.8} {
  \draw[fill=gray!10] (\x,0) rectangle (\x+0.3,\h);
  \foreach \k in {1,...,24} {
    \pgfmathsetmacro{\yy}{\k*200/(1200*2^\l)*\h}
    \ifdim \yy pt < 2.75pt \draw[gray!60, very thin] (\x+0.02,\yy) -- (\x+0.28,\yy); \fi
  }
}
\node[shp, anchor=north] at (0.75,-0.05) {fused levels,\\ frequency axis};
\draw[fill=blue!8, draw=blue!50!black] (2.6,0) rectangle (2.9,2.8);
\node[shp, anchor=north] at (2.9,-0.05) {rate axis,\\ 307 bins\\ of 0.977 rev/s};
\coordinate (g) at (2.6,0.45);
\fill[blue!70!black] (g) circle (1.2pt);
\foreach \k/\l in {1/0, 2/0, 3/0, 5/0, 8/1, 16/2, 20/2} {
  \pgfmathsetmacro{\x}{\l*0.4+0.3}
  \pgfmathsetmacro{\yy}{\k*200/(1200*2^\l)*2.8}
  \draw[tap] (g) -- (\x,\yy);
}
\node[new, anchor=south, text width=3.2cm] at (1.45,2.95) {\textbf{HPDC}: taps at $kg$, $k \le 84$\\ frequency grid $\to$ rate grid, $16 \to C$};
\end{scope}
\draw[big] (b25.east) -- (12.1,1.9);
% ================= row 2: rate-axis blocks, temporal convolutions, ConvLSTM =================
\def\ry{-2.6}
\node[blk] (b4) at (1.0,\ry) {3-tap HPDC\\ $g\cdot\{\tfrac12,1,2\}$};
\node[blk] (b5) at (3.15,\ry) {3-tap HPDC\\ $g\cdot\{2^{-\frac14},1,2^{\frac14}\}$};
\node[blk] (b6) at (5.3,\ry) {temporal conv\\ $5{\times}1$};
\node[blk] (b7) at (7.45,\ry) {temporal conv\\ $5{\times}1$};
\node[blk] (b8) at (9.6,\ry) {temporal conv\\ $5{\times}1$};
\node[new, text width=2.9cm] (lstm) at (12.6,\ry) {\textbf{rate ConvLSTM}\\ bidirectional, gate conv\\ 33 taps along rate};
\node[shp, anchor=north] at (2.1,\ry-0.75) {octave / quarter-octave priors on the rate axis};
\node[shp, anchor=north] at (7.45,\ry-0.75) {$T \times 307 \times C$ throughout};
\node[shp, anchor=east] at (14.3,\ry-0.95) {$T \times 307 \times 2H$};
\draw[big] (15.1,0.45) -- (15.1,-0.9) -- (1.0,-0.9) -- (b4.north);
\draw[big] (b4) -- (b5); \draw[big] (b5) -- (b6); \draw[big] (b6) -- (b7); \draw[big] (b7) -- (b8); \draw[big] (b8) -- (lstm);
% ================= row 3: head, salience layers, readout =================
\def\sy{-5.7}
\node[blk] (hd) at (1.0,\sy) {conv $1{\times}1$\\ $2H \to 4$};
\foreach \i in {3,2,1,0} {
  \pgfmathsetmacro{\ox}{2.6+\i*\dx}\pgfmathsetmacro{\oy}{\sy-0.75+\i*\dy}
  \node[img] (S\i) at (\ox,\oy) {\includegraphics[width=2.2cm,height=1.1cm]{layer\i.png}};
}
\node[shp, anchor=north] at (3.9,\sy-0.9) {4 per-rotor salience layers\\ $T \times 307 \times 4$, PIT cross-entropy};
\node[blk] (crf) at (7.0,\sy) {CRF readout\\ one path per layer};
\node[img, draw=none, anchor=west] (tr) at (8.4,\sy) {\includegraphics[width=3.4cm]{tracks.png}};
\node[shp, anchor=north] at (10.1,\sy-0.9) {$\hat{\mathbf r}[\tau]$, $T \times 4$, against telemetry};
\draw[big] (lstm.east) -- (14.6,\ry) -- (14.6,-4.1) -- (1.0,-4.1) -- (hd.north);
\draw[big] (hd.east) -- (2.5,\sy);
\draw[big] (5.5,\sy) -- (crf.west);
\draw[big] (crf.east) -- (tr.west);
\end{tikzpicture}
"""

HPPNET_VS_L2 = r"""
\begin{tikzpicture}[font=\footnotesize, node distance=3mm,
  blk/.style={draw, rounded corners=1.5pt, align=center, inner sep=2.5pt, fill=white, minimum height=7mm},
  old/.style={blk, fill=gray!10, draw=gray!50!black},
  chg/.style={blk, fill=blue!8, draw=blue!50!black},
  gone/.style={blk, fill=red!6, draw=red!50!black, dashed},
  arr/.style={-{Latex[length=1.2mm]}, thin}]
% row 1: HPPNet as published
\node[anchor=west, font=\small\bfseries] at (-0.2,1.2) {HPPNet (Wei et al. 2022), piano transcription};
\node[old] (a1) at (0.9,0) {CQT\\ 352 log bins};
\node[old, right=of a1] (a2) {conv $7{\times}7$\\ $\times 3$};
\node[old, right=of a2] (a3) {harmonic dilated conv\\ $k \le 9$, dilation $48\log_2 k$};
\node[old, right=of a3] (a4) {pool $\to$ 88 keys\\ dilated conv $\times 4$};
\node[old, right=of a4] (a5) {FreqGroupLSTM\\ one LSTM per key};
\node[old, right=of a5] (a6) {shared salience map\\ + onset / offset / velocity};
\node[old, right=of a6] (a7) {threshold,\\ note events};
\foreach \i/\j in {a1/a2, a2/a3, a3/a4, a4/a5, a5/a6, a6/a7} { \draw[arr] (\i) -- (\j); }
% row 2: HPPNet-L2
\node[anchor=west, font=\small\bfseries] at (-0.2,-1.7) {HPPNet-L2 (last week): the output made for rotors};
\node[old] (b1) at (0.9,-2.9) {CQT\\ 352 log bins};
\node[old, right=of b1] (b2) {conv $7{\times}7$\\ $\times 3$};
\node[old, right=of b2] (b3) {harmonic dilated conv\\ $k \le 9$, dilation $48\log_2 k$};
\node[old, right=of b3] (b4) {dilated conv $\times 4$\\ (no pool)};
\node[old, right=of b4] (b5) {FreqGroupLSTM\\ one LSTM per bin};
\node[chg, right=of b5] (b6) {log $\to$ linear grid\\ 0--150 rev/s, 300 bins};
\node[chg, right=of b6] (b7) {4 per-rotor layers\\ PIT cross-entropy};
\node[chg, right=of b7] (b8) {CRF readout\\ one path per layer};
\foreach \i/\j in {b1/b2, b2/b3, b3/b4, b4/b5, b5/b6, b6/b7, b7/b8} { \draw[arr] (\i) -- (\j); }
\node[gone, below=4mm of b5, text width=4.2cm] (gone) {dropped: onset / offset / velocity heads, note decoding (a rotor is always on)};
\node[anchor=north west, align=left, text width=10cm] at (b1.south west |- gone.south) {\vspace{1mm}\\ why: one shared map on a log axis cannot hold four close, crossing tracks (8.2 rev/s floor with no model); four layers on a linear grid are lossless. HPPNet 7.77 $\to$ 2.27 rev/s, HarmoF0 10.8 $\to$ 2.45, LateDeep 12.6 $\to$ 3.83.};
\end{tikzpicture}
"""

LSTM_BLOCKS = r"""
\begin{tikzpicture}[font=\footnotesize,
  blk/.style={draw, rounded corners=1.5pt, align=center, inner sep=3pt, fill=white, minimum height=7mm},
  old/.style={blk, fill=gray!10, draw=gray!50!black},
  chg/.style={blk, fill=blue!8, draw=blue!50!black},
  arr/.style={-{Latex[length=1.2mm]}, thin}]
% FreqGroupLSTM
\node[anchor=west, font=\small\bfseries] at (0,2.4) {FreqGroupLSTM (HPPNet)};
\node[old, text width=3.0cm] (f1) at (1.6,1.3) {feature map\\ $x[t, b, :]$, $T \times B \times C$};
\node[old, text width=4.6cm, right=4mm of f1] (f2) {for every bin $b$ separately:\\ $h_b[t] = \mathrm{LSTM}\bigl(x[t,b,:],\, h_b[t{-}1]\bigr)$\\ same weights for all $b$};
\node[old, text width=2.2cm, right=4mm of f2] (f3) {$h[t, b, :]$\\ $T \times B \times H$};
\draw[arr] (f1) -- (f2); \draw[arr] (f2) -- (f3);
\node[anchor=north west, text width=9.6cm, align=left] at (0,0.45) {no term couples $h_b$ and $h_{b\pm1}$: the state of a bin cannot move; a rotor which leaves the bin starts from nothing in the next one. Fine for a held piano key, not for a continuously moving pitch.};
% rate ConvLSTM
\begin{scope}[yshift=-3.2cm]
\node[anchor=west, font=\small\bfseries] at (0,2.4) {rate ConvLSTM (HRSPNet)};
\node[chg, text width=3.0cm] (c1) at (1.6,1.3) {feature map\\ $x[t, g, :]$, $T \times G \times C$};
\node[chg, text width=5.6cm, right=4mm of c1] (c2) {gates $i,f,o,\tilde c$ at every rate bin $g$:\\ $W_x\, x[t]$ ($1{\times}1$) $\;+\; W_h \circledast_g h[t{-}1]$ (33 taps along $g$)\\ forward and backward in time};
\node[chg, text width=2.2cm, right=4mm of c2] (c3) {$h[t, g, :]$\\ $T \times G \times 2H$};
\draw[arr] (c1) -- (c2); \draw[arr] (c2) -- (c3);
\node[anchor=north west, text width=9.6cm, align=left] at (0,0.45) {the hidden-to-gate map is a convolution along rate, so the state at $g$ reads the state at $g \pm 16$ bins of the previous frame: it follows a rotor moving up to 15 rev/s per frame (the largest sustained slew in the telemetry). 1.15 M parameters.};
\end{scope}
\end{tikzpicture}
"""


# ── 10. why level 3 hurts: the k ≈ 70 line belongs to a different rotor per room ──
def room_lines(k: int = 70, band: tuple[float, float] = (4.6, 7.0), win_s: float = 8.0) -> None:
    """Level-3 spectrogram of a cruise window in each DREGON room with the k-th
    harmonic of every rotor overlaid, rotors coloured by rank of mean speed.
    Room 1 uses the measured speeds (validation labels), room 2 the commanded
    ones (training labels)."""
    from data_processing.sources import dregon as D

    dd = ROOT / "data/DREGON"
    geo = D.get_geometry(dd)
    samples = {s["recording_id"]: s for s in D.discover_recordings(dd)}
    rooms = [
        ("free-flight_nosource_room1", "motors_measured", "room 1 (validation), measured speeds"),
        ("free-flight_nosource_room2", "motors_command", "room 2 (training), commanded speeds"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
    for ax, (rid, key, title) in zip(axes, rooms):
        tf = D.load_timeframe(samples[rid], geometry=geo, target_sr=FS)
        tel = tf[key].resample((16000, 512))
        r = np.asarray(tel.data, dtype=np.float64)  # (4, N) on the frame grid
        idx = np.flatnonzero(r.mean(0) > 70)
        run = max(np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1), key=len)
        n = int(win_s * FR)
        mid = run[len(run) // 2]
        f0, f1 = max(run[0], mid - n // 2), min(run[-1], mid + n // 2)
        t0 = float(tel.t_start) + f0 / FR
        au = np.asarray(tf["audio"].time[t0 : t0 + (f1 - f0) / FR].data)[0]
        s = stft_db(au.astype(np.float32), 2048)
        f = np.arange(s.shape[0]) * FS / 2048 / 1000
        keep = (f >= band[0]) & (f <= band[1])
        t = np.arange(s.shape[1]) / FR
        ax.imshow(
            s[keep],
            origin="lower",
            aspect="auto",
            cmap="magma",
            extent=(t[0], t[-1], band[0], band[1]),
            vmin=np.percentile(s[keep], 10),
            vmax=np.percentile(s[keep], 99.7),
        )
        rw = r[:, f0:f1]
        rank = np.argsort(np.argsort(-rw.mean(-1)))  # 0 = fastest rotor
        tr = np.arange(rw.shape[1]) / FR
        for i in range(4):
            ax.plot(tr, k * rw[i] / 1000, color=ROTOR_COLORS[rank[i]], lw=1.0, ls="--", alpha=0.9)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("kHz")
    axes[-1].set_xlabel("time (s); 2048-point STFT, 4.6–7 kHz")
    fig.suptitle(
        f"dashed: {k}·r_i(t) for the four rotors, coloured by speed rank (blue = fastest)",
        fontsize=10,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "room_lines.png", dpi=170)
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    spectrogram_comb()
    tracks_vs_pianoroll()
    room_lines()
    encode_decode()
    slew_hist()
    rate_sync_spectrum()
    resolution_curves()
    all_bars()
    fig1 = (PAPER / "src/main.tex").read_text()
    compile_tikz("fig_hrspnet", tikz_from_paper("fig:hrspnet"), PAPER / "figures/overview")
    compile_tikz("genmodel_overview", tikz_from_paper("fig:overview"), PAPER / "figures")
    compile_tikz("hpdc_fans", HPDC_FANS, PAPER / "figures")
    compile_tikz("lstm_cartoon", LSTM_CARTOON, PAPER / "figures")
    cqt_vs_pyramid()
    compile_tikz("arch_horizontal", ARCH_HORIZONTAL, PAPER / "figures/overview")
    compile_tikz("hppnet_vs_l2", HPPNET_VS_L2, PAPER / "figures")
    compile_tikz("lstm_blocks", LSTM_BLOCKS, PAPER / "figures")
    for name in ("gen_traj_real_vs_sampled", "gen_noise_real_vs_sampled"):
        shutil.copy(PAPER / "src/figures" / f"{name}.png", ASSETS / f"{name}.png")
    shutil.copy(PAPER / "figures/freq_probe_panels.png", ASSETS / "freq_probe_panels.png")
    del fig1
    print(f"assets in {ASSETS}")


if __name__ == "__main__":
    main()
