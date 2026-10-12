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
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    lad = ["none", "k ≥ 61", "k ≥ 31", "k ≥ 16"]
    pyr = {
        "all": [2.68, 2.22, 3.31, 4.65],
        "DREGON": [3.33, 2.59, 3.59, 5.65],
        "FLY124": [1.73, 1.69, 2.90, 3.20],
    }
    cqt = {
        "all": [2.31, 2.31, 4.53, 11.1],
        "DREGON": [2.70, 2.58, 5.79, 17.3],
        "FLY124": [1.74, 1.92, 2.68, 2.03],
    }
    for ax, name, d in ((axes[0], "pyramid, k ≤ 84", pyr), (axes[1], "CQT, k ≤ 84", cqt)):
        x = np.arange(4)
        for j, (ds, c) in enumerate((("all", "#333"), ("DREGON", b), ("FLY124", o))):
            ax.bar(x + (j - 1) * 0.26, d[ds], width=0.26, color=c, label=ds)
        ax.set_xticks(x)
        ax.set_xticklabels(["taps zeroed: " + lad[0]] + lad[1:], fontsize=9)
        ax.set_title(name, fontsize=11)
        ax.set_ylim(0, 8)
    axes[0].set_ylabel("matched MAE (rev/s)")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].text(3.0, 7.3, "11.1 / 17.3\n(clipped)", fontsize=8, color="0.35", ha="center")
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
\usepackage{tikz,graphicx,amsmath}
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


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    spectrogram_comb()
    tracks_vs_pianoroll()
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
    for name in ("gen_traj_real_vs_sampled", "gen_noise_real_vs_sampled"):
        shutil.copy(PAPER / "src/figures" / f"{name}.png", ASSETS / f"{name}.png")
    shutil.copy(PAPER / "figures/freq_probe_panels.png", ASSETS / "freq_probe_panels.png")
    del fig1
    print(f"assets in {ASSETS}")


if __name__ == "__main__":
    main()
