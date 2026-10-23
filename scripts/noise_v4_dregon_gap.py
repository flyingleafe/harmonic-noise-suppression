#!/usr/bin/env python3
"""Noise model v4 — why the DREGON score is bad: real vs nv3r4 vs a v4 bench fit
on the five frozen DREGON cruise scoring windows, plus the v4 bench fit against
its own bench recordings (``docs/explainers/noise-model-v4-dregon-gap.qmd``)::

    python scripts/noise_v4_dregon_gap.py flight --v4 <bench payload> --v3 <nv3r4 payload>
    python scripts/noise_v4_dregon_gap.py flight --rig michaels --v4 <MD2 payload> \\
        --v3 <nv3r4 cruise> --v3-standby <nv3r4 standby> [--v4-standby <v4 standby>]
    python scripts/noise_v4_dregon_gap.py insample --v4 <bench payload>
    python scripts/noise_v4_dregon_gap.py hppnet --v4 <DREGON flight payload> --v3 <nv3r4 payload>

``--rig michaels`` does the same on Michael's five FLY124 scoring windows
(standby, ramp, cruise); a per-regime pair renders through the scorer's
``render_noise_regimes``; its outputs carry the prefix ``md2_``
(``figdata_md2.json``, ``md2_fig_*.png``, ``audio/md2_<regime>_*.wav``).

``flight`` renders each window from both fits on the window's raw telemetry
(the scorer's carrier, render seed 2001, all eight mics), and writes Welch
spectra (mic 0 and the mic mean), the scorer's band LTAS, per-mic level offsets,
spectrograms of the first window, the v4 expected periodogram split into lines
and floor, and WAVs of mic 0. ``insample`` evaluates the bench payload's model
on its own recordings with the bench fit's forward model; run it with the
``src/`` of the commit that produced the payload first on ``PYTHONPATH`` (the
payload must be read by its own code). Outputs: ``--out`` (default
``docs/explainers/noise-model-v4-dregon-gap``) ``figdata_<mode>.json``,
``fig_*.png``, ``audio/*.wav``. ``hppnet`` puts one scoring window (default
hovering) through the frozen scorer for real, nv3r4, the v4 payload and the v4
payload with its lines raised ``--line-db`` (matched level): spectrograms with
the scorer's readout, WAVs on one scale per mic, the per-mic MAE
(``fig_hppnet_window.png``, ``audio/hppnet_*.wav``, ``figdata_hppnet.json``).
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

# appended, not prepended: ``insample`` puts the payload's own ``src/`` first
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

OUT = Path("docs/explainers/noise-model-v4-dregon-gap")
SEED = 2001
SR = 16000
NPERSEG = 4096
COLORS = {"real": "black", "nv3r4": "tab:blue", "v4": "tab:red"}


def _git() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


def _welch(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import welch

    f, p = welch(x, fs=SR, nperseg=NPERSEG, noverlap=3 * NPERSEG // 4, window="hann", axis=-1)
    return f, p


def _db(x: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(x, 1e-30))


def _octave_bands(lo: float = 31.25, hi: float = 7900.0, per_oct: int = 3) -> np.ndarray:
    n = int(np.ceil(np.log2(hi / lo) * per_oct))
    return lo * 2.0 ** (np.arange(n + 1) / per_oct)


def _band_db(f: np.ndarray, p: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mean PSD per band (dB), last axis frequency."""
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (f >= a) & (f < b)
        out.append(_db(p[..., sel].mean(axis=-1)))
    return np.stack(out, axis=-1)


#: Orders of the line-prominence diagnostic and its STFT (256 ms frames).
PROM_K_MAX = 60
PROM_NFFT = 4096
PROM_HOP = 1024


def _order_prominence(x: np.ndarray, rps: np.ndarray) -> np.ndarray:
    """``(PROM_K_MAX,)`` line prominence (dB) per integer order: per frame and
    rotor, the peak mic-mean power within 0.08 orders of ``k f_r`` over the median
    power 0.3-0.5 orders away, averaged over frames and rotors."""
    from scipy.signal import stft

    f, t, z = stft(
        x,
        fs=SR,
        nperseg=PROM_NFFT,
        noverlap=PROM_NFFT - PROM_HOP,
        window="hann",
        boundary=None,
        padded=False,
    )
    p = (np.abs(z) ** 2).mean(axis=0)  # (F, N)
    n_samp = x.shape[-1]
    t_rps = np.arange(rps.shape[1]) * (n_samp / rps.shape[1]) / SR
    out = np.zeros((rps.shape[0], PROM_K_MAX))
    for r in range(rps.shape[0]):
        fr = np.interp(t, t_rps, rps[r])
        acc = np.zeros(PROM_K_MAX)
        for n in range(p.shape[1]):
            kk = f / max(fr[n], 1.0)
            col = p[:, n]
            for k in range(1, PROM_K_MAX + 1):
                d = np.abs(kk - k)
                peak = col[d < 0.08]
                side = col[(d > 0.3) & (d < 0.5)]
                if peak.size and side.size:
                    acc[k - 1] += _db(peak.max()) - _db(np.median(side))
        out[r] = acc / p.shape[1]
    return out.mean(axis=0)


def _wav(path: Path, x: np.ndarray, scale: float) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(x * scale, -1.0, 1.0), SR, subtype="PCM_16")


def cmd_flight(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    from data_processing.noise_model import v4 as V4
    from data_processing.noise_model.render import render_noise, render_noise_regimes
    from experiments.noise_model import gates as GT
    from experiments.stochastic_fit import accept_stats
    from experiments.stochastic_fit import revised_eval as RE

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rig = str(args.rig)
    tag = "" if rig == "dregon" else "md2_"
    fits = {
        "nv3r4": json.loads(Path(args.v3).read_text()),
        "v4": json.loads(Path(args.v4).read_text()),
    }
    # a per-regime pair (standby below 45 rev/s, cruise above 65) renders through
    # the scorer's composition, ``render_noise_regimes``
    standby = {
        "nv3r4": json.loads(Path(args.v3_standby).read_text()) if args.v3_standby else None,
        "v4": json.loads(Path(args.v4_standby).read_text()) if args.v4_standby else None,
    }
    supports = GT.DREGON_CRUISE_SUPPORTS if rig == "dregon" else GT.MICHAELS_SUPPORTS
    edges = _octave_bands()
    centres = np.sqrt(edges[:-1] * edges[1:])
    ltas_bands = [list(b) for b in accept_stats.BANDS]
    rows: list[dict[str, Any]] = []
    spectra: dict[str, dict[str, np.ndarray]] = {}
    first: dict[str, np.ndarray] = {}
    decomp: dict[str, Any] = {}
    n_mics = 8
    heard: set[str] = set()
    for i, sup in enumerate(supports):
        clip = RE.load_window(
            sup.window,
            dataset=GT.DATASET[rig],
            version=None,
            channels=None,
            rps_key=GT.RAW_RPS_KEY[rig],
        )
        real = np.asarray(clip.audio, dtype=np.float64)[:8]
        n_mics = real.shape[0]
        rps = np.atleast_2d(np.asarray(clip.rps, dtype=np.float64))
        audio = {"real": real}
        for name, fit in fits.items():
            sb = standby[name]
            x = (
                render_noise_regimes({"standby": sb, "cruise": fit}, rps, n_mics=n_mics, seed=SEED)
                if sb is not None
                else render_noise(fit, rps, n_mics=n_mics, seed=SEED)
            )
            audio[name] = np.asarray(x, dtype=np.float64)[:, : real.shape[1]]
        f, _ = _welch(real[0])
        key = (
            sup.recording.replace("_nosource_room2", "")
            if rig == "dregon"
            else f"{sup.regime} @{sup.start_s:g} s"
        )
        spectra[key] = {}
        row: dict[str, Any] = dict(
            key=key,
            recording=sup.recording,
            start_s=sup.start_s,
            rps_mean=rps.mean(axis=1).round(2).tolist(),
            rps_min=float(rps.min()),
            rps_max=float(rps.max()),
        )
        for name, x in audio.items():
            _, p = _welch(x)  # (8, F)
            spectra[key][name] = p
            row[f"ltas_{name}"] = RE.absolute_ltas_bands(x[0]).tolist()
            row[f"band_mic0_{name}"] = _band_db(f, p[0], edges).tolist()
            row[f"band_mean_{name}"] = _band_db(f, p.mean(axis=0), edges).tolist()
            row[f"level_db_per_mic_{name}"] = _db(
                p[:, (f >= 31.25) & (f <= 7900)].mean(axis=1)
            ).tolist()
            row[f"prominence_{name}"] = _order_prominence(x, rps).tolist()
        for name in fits:
            dev = RE.ltas_deviation_db(real, audio[name], mic=0)
            row[f"scorer_{name}"] = {
                k: dev[k]
                for k in ("mean_abs_db", "max_abs_db", "level_offset_db", "shape_only_mean_abs_db")
            }
        # the v4 expected periodogram at the window's mean speeds, total and parts
        # (of the regime's own fit when v4 is a pair: standby below the ramp)
        rbar = rps.mean(axis=1)
        v4_fit = (
            standby["v4"] if standby["v4"] is not None and float(rbar.min()) < 45.0 else fits["v4"]
        )
        lines_only = copy.deepcopy(v4_fit)
        lines_only["params"]["floor"]["mean_db"] = -300.0
        floor_only = copy.deepcopy(v4_fit)
        pr = floor_only["params"]
        pr["profile_aero_db"] = (np.asarray(pr["profile_aero_db"]) - 300.0).tolist()
        pr["profile_motor_db"] = (np.asarray(pr["profile_motor_db"]) - 300.0).tolist()
        n_fft = NPERSEG
        fe = np.fft.rfftfreq(n_fft, 1.0 / SR)
        parts = {
            "total": V4.expected_periodogram(v4_fit, rbar, n_fft=n_fft, sr=SR),
            "lines": V4.expected_periodogram(lines_only, rbar, n_fft=n_fft, sr=SR),
            "floor": V4.expected_periodogram(floor_only, rbar, n_fft=n_fft, sr=SR),
        }
        # Welch "density" is periodogram / sr; the expected periodogram is in
        # periodogram units (white noise of variance v reads v): divide by sr/2
        # for the one-sided density
        decomp[key] = {k: (v / (SR / 2.0)) for k, v in parts.items()}
        decomp[key]["f"] = fe
        rows.append(row)
        if i == 0:
            first = {k: v[0] for k, v in audio.items()}
        if rig == "dregon" and i == 0 or rig != "dregon" and sup.regime not in heard:
            heard.add(sup.regime)
            scale = 0.5 / float(np.max(np.abs(real[0])))
            slug = key if rig == "dregon" else f"md2_{sup.regime}"
            for name, x in audio.items():
                _wav(out / "audio" / f"{slug}_{name}_mic0.wav", x[0], scale)
        print(
            f"{key}: " + ", ".join(f"{n} {row[f'scorer_{n}']['mean_abs_db']:.2f} dB" for n in fits),
            flush=True,
        )

    names = ["real", "nv3r4", "v4"]
    keys = [r["key"] for r in rows]

    # Fig 1: Welch PSD, mic 0, each window
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 2.6 * len(keys)), sharex=True)
    inband = (f >= 30.0) & (f <= 7900.0)
    for ax, key in zip(axes, keys):
        curves = [_db(spectra[key][n][0]) for n in names]
        for n, c in zip(names, curves):
            ax.semilogx(f, c, color=COLORS[n], lw=0.8, label=n)
        lo = min(float(np.min(c[inband])) for c in curves)
        hi = max(float(np.max(c[inband])) for c in curves)
        ax.set_ylim(lo - 3.0, hi + 3.0)
        ax.set_xlim(30, 8000)
        ax.set_ylabel("PSD (dB)")
        ax.set_title(f"{key}: mic 0", fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("frequency (Hz)")
    fig.tight_layout()
    fig.savefig(out / f"{tag}fig_psd_mic0.png", dpi=130)
    plt.close(fig)

    # Fig 2: third-octave model - real, mic 0 and mic mean, every window
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, which in zip(axes, ("mic0", "mean")):
        for key in keys:
            r = next(x for x in rows if x["key"] == key)
            for n, ls in (("nv3r4", "-"), ("v4", "-")):
                d = np.asarray(r[f"band_{which}_{n}"]) - np.asarray(r[f"band_{which}_real"])
                ax.semilogx(centres, d, color=COLORS[n], lw=1.0, alpha=0.7, ls=ls)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlim(30, 8000)
        ax.set_title("mic 0" if which == "mic0" else f"mean over the {n_mics} mics", fontsize=10)
        ax.set_xlabel("third-octave band centre (Hz)")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("model − real (dB)")
    axes[0].plot([], [], color=COLORS["nv3r4"], label=f"nv3r4 ({len(keys)} windows)")
    axes[0].plot([], [], color=COLORS["v4"], label=f"v4 ({len(keys)} windows)")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out / f"{tag}fig_band_error.png", dpi=130)
    plt.close(fig)

    # Fig 3: per-mic level offset (31-7900 Hz mean PSD, model - real)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for n, off in (("nv3r4", -0.12), ("v4", 0.12)):
        for j, key in enumerate(keys):
            r = next(x for x in rows if x["key"] == key)
            d = np.asarray(r[f"level_db_per_mic_{n}"]) - np.asarray(r["level_db_per_mic_real"])
            ax.plot(
                np.arange(n_mics) + off,
                d,
                "o",
                color=COLORS[n],
                ms=4,
                alpha=0.7,
                label=n if j == 0 else None,
            )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(n_mics))
    ax.set_xlabel("microphone")
    ax.set_ylabel("broadband level, model − real (dB)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"{tag}fig_mic_offsets.png", dpi=130)
    plt.close(fig)

    # Fig 4: spectrograms of the first window, mic 0
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    vmax = None
    for ax, n in zip(axes, names):
        fs, ts, S = spectrogram(first[n], fs=SR, nperseg=1024, noverlap=768, window="hann")
        Sd = _db(S)
        if vmax is None:
            vmax = float(np.percentile(Sd, 99.5))
        im = ax.pcolormesh(ts, fs, Sd, shading="auto", vmin=vmax - 70, vmax=vmax, cmap="magma")
        ax.set_ylim(0, 3000)
        ax.set_title(f"{n}", fontsize=10)
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("frequency (Hz)")
    fig.colorbar(im, ax=axes, shrink=0.9, label="dB")
    fig.savefig(out / f"{tag}fig_spectrogram.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Fig 5: v4 decomposition vs real, mic 0: the first window (DREGON), or the
    # first window of each regime (MD2)
    shown = (
        [keys[0]]
        if rig == "dregon"
        else [next(r["key"] for r in rows if r["key"].startswith(g)) for g in ("standby", "cruise")]
    )
    fig, axes = plt.subplots(len(shown), 2, figsize=(13, 4.2 * len(shown)), squeeze=False)
    for row_axes, key in zip(axes, shown):
        dk = decomp[key]
        for ax, xlim in zip(row_axes, ((30, 8000), (30, 600))):
            ax.plot(f, _db(spectra[key]["real"][0]), color="black", lw=0.8, label="real (Welch)")
            ax.plot(
                f,
                _db(spectra[key]["v4"][0]),
                color="tab:red",
                lw=0.8,
                alpha=0.6,
                label="v4 render (Welch)",
            )
            ax.plot(
                dk["f"],
                _db(dk["total"][0]),
                color="tab:red",
                lw=1.4,
                ls="--",
                label="v4 expected: total",
            )
            ax.plot(
                dk["f"], _db(dk["lines"][0]), color="tab:orange", lw=1.0, label="v4 expected: lines"
            )
            ax.plot(
                dk["f"], _db(dk["floor"][0]), color="tab:green", lw=1.0, label="v4 expected: floor"
            )
            ax.set_xscale("log")
            ax.set_xlim(*xlim)
            ax.set_xlabel("frequency (Hz)")
            ax.grid(alpha=0.3, which="both")
        row_axes[0].set_ylabel("PSD (dB)")
        row_axes[0].set_title(f"{key}: mic 0, 30 Hz – 8 kHz", fontsize=10)
        row_axes[1].set_title(f"{key}: 30 – 600 Hz", fontsize=10)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{tag}fig_v4_parts.png", dpi=130)
    plt.close(fig)

    # Fig 6: line prominence per order (peak over the 0.3-0.5-order sides), the
    # mean over the windows of each regime
    groups = ["cruise"] if rig == "dregon" else ["standby", "ramp", "cruise"]
    fig, axes = plt.subplots(1, len(groups), figsize=(5.2 * len(groups), 3.8), squeeze=False)
    korders = np.arange(1, PROM_K_MAX + 1)
    for ax, g in zip(axes[0], groups):
        sel = [r for r in rows if rig == "dregon" or r["key"].startswith(g)]
        for n in names:
            v = np.mean([r[f"prominence_{n}"] for r in sel], axis=0)
            ax.plot(korders, v, color=COLORS[n], lw=1.0, marker=".", ms=3, label=n)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(f"{g}: {len(sel)} window(s)", fontsize=10)
        ax.set_xlabel("rotor order k")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("line prominence (dB)")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{tag}fig_prominence.png", dpi=130)
    plt.close(fig)

    data = dict(
        git=_git(),
        rig=rig,
        v4=str(args.v4),
        v4_standby=args.v4_standby,
        v3=str(args.v3),
        v3_standby=args.v3_standby,
        seed=SEED,
        welch=dict(nperseg=NPERSEG, overlap=0.75, window="hann", sr=SR),
        third_octave_centres_hz=centres.round(2).tolist(),
        ltas_bands_hz=ltas_bands,
        windows=rows,
    )
    (out / ("figdata_flight.json" if rig == "dregon" else "figdata_md2.json")).write_text(
        json.dumps(data, indent=1)
    )
    print(f"wrote {out}", flush=True)
    return 0


def cmd_insample(args: argparse.Namespace) -> int:
    """The bench payload's own forward model (``bench_v4.RigModel``) on its own
    recordings: data (mic mean) vs model total, lines, floor and ambient."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from experiments.noise_model import bench_v4 as B

    torch.set_num_threads(int(args.threads))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pay = json.loads(Path(args.v4).read_text())
    keys = [r["key"] for r in pay["recordings"]]
    want = [k for k in keys if k.endswith("_80")] + ["motor_Motor1_50", "motor_Motor1_90"]
    # the payload's whole rig: per-recording parameters exist only for all of them
    settings = [B.load_setting(k.split("_")[1], int(k.split("_")[2])) for k in keys]
    motors = tuple(dict.fromkeys(r["motor"] for r in pay["recordings"]))
    rig = B.build_rig(motors, settings=settings)
    model = B.RigModel(rig, device="cpu", kappa=float(pay["priors"]["floor_push_kappa"]))
    model.load_payload(pay)
    edges = _octave_bands(per_oct=6)
    centres = np.sqrt(edges[:-1] * edges[1:])
    rows = []
    fig, axes = plt.subplots(len(want), 2, figsize=(13, 2.7 * len(want)))
    shown = [keys.index(k) for k in want]
    with torch.no_grad():
        for row_i, j in enumerate(shown):
            s, fz = rig.settings[j], model.frozen[j]
            f = np.asarray(fz.grid.freqs_hz)
            band = np.asarray(fz.band.cpu())
            comp = {k: v.cpu().numpy() for k, v in model.components(j).items()}
            amb = fz.ambient.cpu().numpy()
            data = fz.power.cpu().numpy()
            total = comp["lines"] + comp["floor"] + comp["wind"] + amb
            parts = {
                "data": data.mean(0),
                "model": total.mean(0),
                "lines": comp["lines"].mean(0),
                "floor": comp["floor"].mean(0),
                "ambient": amb.mean(0),
            }
            bands = {k: _band_db(f[band], v[band], edges) for k, v in parts.items()}
            rows.append(
                dict(
                    key=s.key,
                    carrier_rev_s=s.carrier,
                    bands_db={k: v.tolist() for k, v in bands.items()},
                    whittle_per_cell=float(
                        (np.log(total[:, band]) + data[:, band] / total[:, band]).mean()
                    ),
                )
            )
            ax = axes[row_i, 0]
            for k, c in (
                ("data", "black"),
                ("model", "tab:red"),
                ("lines", "tab:orange"),
                ("floor", "tab:green"),
                ("ambient", "tab:gray"),
            ):
                ax.semilogx(
                    centres, bands[k], color=c, lw=1.2 if k in ("data", "model") else 0.9, label=k
                )
            ax.set_xlim(30, 8000)
            ax.set_title(f"{s.key} ({s.carrier:.1f} rev/s): sixth-octave, mic mean", fontsize=9)
            ax.grid(alpha=0.3, which="both")
            ax.set_ylabel("dB")
            ax = axes[row_i, 1]
            sel = band & (f <= 600)
            ax.plot(f[sel], _db(parts["data"][sel]), color="black", lw=0.6, label="data")
            ax.plot(f[sel], _db(parts["model"][sel]), color="tab:red", lw=0.8, label="model")
            ax.plot(f[sel], _db(parts["floor"][sel]), color="tab:green", lw=0.8, label="floor")
            ax.set_title("30 – 600 Hz, every bin", fontsize=9)
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, ncol=5)
    axes[-1, 0].set_xlabel("frequency (Hz)")
    axes[-1, 1].set_xlabel("frequency (Hz)")
    fig.tight_layout()
    fig.savefig(out / "fig_bench_insample.png", dpi=120)
    plt.close(fig)
    (out / "figdata_insample.json").write_text(
        json.dumps(
            dict(
                v4=str(args.v4), sixth_octave_centres_hz=centres.round(2).tolist(), recordings=rows
            ),
            indent=1,
        )
    )
    print(f"wrote {out}", flush=True)
    return 0


#: ``hppnet``: the matched-level v4 arm raises both line-profile rows by this much
#: (the probe that brought every mic of the k0 stage above the scorer's level
#: threshold on the hovering and spinning windows)
MATCHED_LINE_DB = 3.0


def cmd_hppnet(args: argparse.Namespace) -> int:
    """Real vs nv3r4 vs a v4 flight payload (as fitted, and with its lines raised
    by ``--line-db``) on ONE DREGON scoring window, through the frozen scorer:
    spectrograms with the scorer's per-rotor readout over the label, WAVs on one
    scale per mic (levels stay comparable), the per-mic MAE and band levels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _synthetic_probe as SP
    import stochastic_fit_revised_eval as SRE

    from data_processing.noise_model.render import render_noise
    from experiments.noise_model import gates as GT
    from experiments.stochastic_fit import revised_eval as RE

    out = Path(args.out)
    sup = GT.DREGON_CRUISE_SUPPORTS[int(args.window)]
    clip = RE.load_window(
        sup.window, dataset="DREGON-frames", version=None, channels=None, rps_key="motors_command"
    )
    real = np.asarray(clip.audio, dtype=np.float64)[:8]
    rps = np.atleast_2d(np.asarray(clip.rps, dtype=np.float64))
    v4 = json.loads(Path(args.v4).read_text())
    raised = copy.deepcopy(v4)
    for key in ("profile_aero_db", "profile_motor_db"):
        raised["params"][key] = (np.asarray(raised["params"][key]) + args.line_db).tolist()
    labels = {
        "real": "real",
        "nv3r4": "nv3r4",
        "v4": "v4 as fitted",
        "v4_matched": f"v4, lines +{args.line_db:g} dB",
    }
    arms = {"real": real}
    for key, pay in (
        ("nv3r4", json.loads(Path(args.v3).read_text())),
        ("v4", v4),
        ("v4_matched", raised),
    ):
        arms[key] = np.asarray(render_noise(pay, rps, n_mics=8, seed=SEED))[:, : real.shape[1]]
    tr = SRE.Tracker(dict(experiment="hppnet_l2_r2_s0", ckpt="best"))
    mics = [int(m) for m in args.mics.split(",")]
    n = real.shape[1]
    scores: dict[str, Any] = {}
    reads: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for key, a in arms.items():
        per = []
        for m in range(8):
            pred, truth, mae = SP.score(tr.fm, tr.metric, a, rps, SR, m)
            per.append(float(mae))
            if m in mics:
                reads[(key, m)] = (pred, truth)
        scores[key] = dict(mae=float(np.mean(per)), per_mic=per)
        print(f"{key}: MAE {np.mean(per):.2f}  per mic {np.round(per, 1).tolist()}", flush=True)
    edges = np.array([30, 60, 120, 250, 500, 1000, 2000, 4000, 8000], dtype=float)
    bands: dict[str, Any] = {}
    for key, a in arms.items():
        f, p = _welch(a[mics])
        bands[key] = _band_db(f, p, edges).round(2).tolist()
    # spectrograms: rows arms, columns mics; one colour scale per mic (the real clip's)
    fig, axes = plt.subplots(
        len(arms),
        len(mics),
        figsize=(5.6 * len(mics), 2.5 * len(arms)),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    axes = np.atleast_2d(axes).reshape(len(arms), len(mics))
    nper = 4096  # 3.9 Hz bins: the four rotors' k = 1 lines are 2-8 Hz apart
    for c, m in enumerate(mics):
        f, t, s = spectrogram(real[m], fs=SR, nperseg=nper, noverlap=nper - 512, window="hann")
        hi = float(np.percentile(_db(s[f <= args.fmax]), 99.5))
        for r, (key, a) in enumerate(arms.items()):
            f, t, s = spectrogram(a[m], fs=SR, nperseg=nper, noverlap=nper - 512, window="hann")
            ax = axes[r, c]
            sel = f <= args.fmax
            mesh = ax.pcolormesh(
                t, f[sel], _db(s[sel]), vmin=hi - 50, vmax=hi, shading="auto", cmap="magma"
            )
            ax.set_title(f"{labels[key]}, mic {m}", fontsize=10)
            if c == 0:
                ax.set_ylabel("frequency (Hz)")
        fig.colorbar(mesh, ax=axes[:, c], shrink=0.6, label="dB (same scale in this column)")
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.savefig(out / "fig_hppnet_window.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    # the scorer's per-rotor readout against the label, one panel per arm and mic
    fig, axes = plt.subplots(
        len(arms), len(mics), figsize=(5.2 * len(mics), 1.9 * len(arms)), sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes).reshape(len(arms), len(mics))
    for c, m in enumerate(mics):
        for r, key in enumerate(arms):
            pred, truth = reads[(key, m)]
            tc = np.asarray(RE.output_frame_centres(int(pred.shape[-1]), n, sr=SR))
            ax = axes[r, c]
            for j in range(pred.shape[0]):
                ax.plot(tc, truth[j], color="0.6", lw=1.2)
                # a reading below the axis (the scorer's "off") is drawn on its floor
                ax.plot(tc, np.maximum(pred[j], 61.0), lw=0.0, marker=".", ms=2.5, color=f"C{j}")
            ax.set_ylim(60, 95)
            ax.set_title(
                f"{labels[key]}, mic {m}: MAE {scores[key]['per_mic'][m]:.1f} rev/s", fontsize=10
            )
            ax.grid(alpha=0.3)
            if c == 0:
                ax.set_ylabel("rev/s")
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    axes[0, 0].plot([], [], color="0.6", lw=1.2, label="label (motors_command)")
    axes[0, 0].plot([], [], lw=0, marker=".", color="C0", label="scorer, one colour per rotor")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.text(0.5, 0.0, "readings below 61 rev/s (the scorer's 'off') are drawn at 61", ha="center")
    fig.tight_layout()
    fig.savefig(out / "fig_hppnet_readout.png", dpi=110)
    plt.close(fig)
    for m in mics:
        scale = 0.5 / max(float(np.max(np.abs(a[m]))) for a in arms.values())
        for key, a in arms.items():
            _wav(out / "audio" / f"hppnet_{key}_mic{m}.wav", a[m], scale)
    (out / "figdata_hppnet.json").write_text(
        json.dumps(
            dict(
                window=sup.window.recording,
                window_start_s=sup.window.start_s,
                v4=str(args.v4),
                v3=str(args.v3),
                line_db=args.line_db,
                seed=SEED,
                scorer=tr.record,
                mics=mics,
                labels=labels,
                scores=scores,
                band_edges_hz=edges.tolist(),
                band_db=bands,
                wav_scale="one per mic: 0.5 / the largest |sample| over the four arms",
                git=_git(),
            ),
            indent=1,
        )
    )
    print(f"wrote {out}", flush=True)
    return 0


#: ``full``: the clip length, and the log-frequency axis of its spectrograms
FULL_SECONDS = 8.0
FULL_FMIN, FULL_FMAX = 30.0, 8000.0
FULL_NPERSEG, FULL_HOP = 2048, 256


def cmd_full(args: argparse.Namespace) -> int:
    """8 s spectrograms over the whole band (log frequency, 30 Hz - 8 kHz) on
    each window's real rotor-speed track: real, nv3r4, v4 (and v4 with its lines
    raised ``--line-db``). DREGON: the five recordings from the start of each
    4 s scoring window (scored 4 s, the 2 s guard, then the first 2 s of the fit
    segment). MD2: the scorer's five 8 s FLY124 windows; per-regime pairs render
    through ``render_noise_regimes``. One colour scale per mic (the real clip's);
    WAVs on one scale per mic and window."""
    import dataclasses

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    from data_processing.noise_model.render import render_noise, render_noise_regimes
    from experiments.noise_model import gates as GT
    from experiments.stochastic_fit import revised_eval as RE

    out = Path(args.out)
    rig = str(args.rig)
    tag = "dregon" if rig == "dregon" else "md2"
    mics = [int(m) for m in (args.mics or ("1,2" if rig == "dregon" else "0,4")).split(",")]
    v4 = json.loads(Path(args.v4).read_text())
    fits: dict[str, tuple[dict[str, Any], dict[str, Any] | None]] = {
        "nv3r4": (
            json.loads(Path(args.v3).read_text()),
            json.loads(Path(args.v3_standby).read_text()) if args.v3_standby else None,
        ),
        "v4": (v4, json.loads(Path(args.v4_standby).read_text()) if args.v4_standby else None),
    }
    labels = {"real": "real", "nv3r4": "nv3r4", "v4": "v4"}
    if args.line_db is not None:
        raised = copy.deepcopy(v4)
        for key in ("profile_aero_db", "profile_motor_db"):
            raised["params"][key] = (np.asarray(raised["params"][key]) + args.line_db).tolist()
        fits["v4_lines"] = (raised, None)
        labels["v4_lines"] = f"v4, lines +{args.line_db:g} dB"
    supports = GT.DREGON_CRUISE_SUPPORTS if rig == "dregon" else GT.MICHAELS_SUPPORTS
    rows = []
    for sup in supports:
        window = dataclasses.replace(sup.window, duration_s=float(args.seconds))
        clip = RE.load_window(
            window,
            dataset=GT.DATASET[rig],
            version=None,
            channels=None,
            rps_key=GT.RAW_RPS_KEY[rig],
        )
        real = np.asarray(clip.audio, dtype=np.float64)[:8]
        rps = np.atleast_2d(np.asarray(clip.rps, dtype=np.float64))
        audio = {"real": real}
        for name, (fit, sb) in fits.items():
            x = (
                render_noise_regimes({"standby": sb, "cruise": fit}, rps, n_mics=8, seed=SEED)
                if sb is not None
                else render_noise(fit, rps, n_mics=8, seed=SEED)
            )
            audio[name] = np.asarray(x, dtype=np.float64)[:, : real.shape[1]]
        key = (
            sup.recording.replace("_nosource_room2", "")
            if rig == "dregon"
            else f"{sup.regime}_{sup.start_s:g}s"
        )
        fig, axes = plt.subplots(
            len(audio),
            len(mics),
            figsize=(6.4 * len(mics), 2.3 * len(audio)),
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        axes = np.atleast_2d(axes).reshape(len(audio), len(mics))
        levels = {}
        for c, m in enumerate(mics):
            spec = {}
            for name, x in audio.items():
                f, t, s = spectrogram(
                    x[m],
                    fs=SR,
                    nperseg=FULL_NPERSEG,
                    noverlap=FULL_NPERSEG - FULL_HOP,
                    window="hann",
                )
                spec[name] = (f, t, _db(s))
            f, _, s_real = spec["real"]
            sel = (f >= FULL_FMIN) & (f <= FULL_FMAX)
            # the top of the scale: the real clip's loudest time-averaged bin + 6 dB
            # (a percentile over all bins is set by the many high-frequency bins and
            # saturates the low band on a log axis)
            hi = float(np.max(_db(np.mean(10.0 ** (s_real[sel] / 10.0), axis=1)))) + 6.0
            for r, (name, (f, t, s)) in enumerate(spec.items()):
                ax = axes[r, c]
                mesh = ax.pcolormesh(
                    t, f[sel], s[sel], vmin=hi - 70, vmax=hi, shading="auto", cmap="magma"
                )
                ax.set_yscale("log")
                ax.set_ylim(FULL_FMIN, FULL_FMAX)
                ax.set_title(f"{labels[name]}, mic {m}", fontsize=10)
                if c == 0:
                    ax.set_ylabel("Hz (log)")
            fig.colorbar(mesh, ax=axes[:, c], shrink=0.5, label="dB (real clip's scale)")
            for name, x in audio.items():
                levels.setdefault(name, []).append(float(_db(np.mean(x[m] ** 2))))
            scale = 0.5 / max(float(np.max(np.abs(x[m]))) for x in audio.values())
            for name, x in audio.items():
                _wav(out / "audio" / f"full_{tag}_{key}_{name}_mic{m}.wav", x[m], scale)
        for ax in axes[-1]:
            ax.set_xlabel("time (s)")
        fig.suptitle(
            f"{tag.upper()} {sup.recording} @ {sup.start_s:.2f} s, {args.seconds:g} s, "
            f"rotors {rps.min():.0f}-{rps.max():.0f} rev/s",
            fontsize=11,
        )
        fig.savefig(out / f"fig_full_{tag}_{key}.png", dpi=100)
        plt.close(fig)
        rows.append(
            dict(
                key=key,
                recording=sup.recording,
                start_s=sup.start_s,
                regime=sup.regime,
                rps_min=float(rps.min()),
                rps_max=float(rps.max()),
                rms_db_per_mic=levels,
                figure=f"fig_full_{tag}_{key}.png",
            )
        )
        print(f"{tag} {key}: rotors {rps.min():.1f}-{rps.max():.1f} rev/s", flush=True)
    (out / f"figdata_full_{tag}.json").write_text(
        json.dumps(
            dict(
                rig=rig,
                seconds=args.seconds,
                mics=mics,
                seed=SEED,
                labels=labels,
                v3=[args.v3, args.v3_standby],
                v4=[args.v4, args.v4_standby],
                line_db=args.line_db,
                windows=rows,
                git=_git(),
            ),
            indent=1,
        )
    )
    # the explainer's tabs (one per window: figure + audio), read by a Quarto
    # ``include`` from the .qmd; paths are relative to the .qmd's directory
    rel = f"{out.name}"
    md = ["::: {.panel-tabset}", ""]
    for row in rows:
        k = row["key"]
        title = (
            k if rig == "dregon" else f"{row['regime']} @ {row['start_s']:g} s"
        ) + f" ({row['rps_min']:.0f}–{row['rps_max']:.0f} rev/s)"
        fig_md = f"![]({rel}/{row['figure']}){{.column-page}}"
        md += [f"### {title}", "", fig_md, "", "```{=html}", "<table>"]
        md.append("<tr><th></th>" + "".join(f"<th>mic {m}</th>" for m in mics) + "</tr>")
        for name, lab in labels.items():
            cells = "".join(
                f'<td><audio controls preload="none" '
                f'src="{rel}/audio/full_{tag}_{k}_{name}_mic{m}.wav"></audio></td>'
                for m in mics
            )
            md.append(f"<tr><td>{lab}</td>{cells}</tr>")
        md += ["</table>", "```", ""]
    md += [":::", ""]
    (out / f"_full_{tag}.md").write_text("\n".join(md))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("flight")
    p.add_argument("--rig", choices=("dregon", "michaels"), default="dregon")
    p.add_argument("--v4", required=True, help="the v4 payload (MD2: joint, or the cruise half)")
    p.add_argument("--v3", required=True, help="the nv3r4 payload (MD2: the cruise half)")
    p.add_argument("--v4-standby", default=None, help="MD2: the v4 standby half of a pair")
    p.add_argument("--v3-standby", default=None, help="MD2: the nv3r4 standby half")
    p.add_argument("--out", default=str(OUT))
    p.set_defaults(func=cmd_flight)
    p = sub.add_parser("insample")
    p.add_argument("--v4", required=True)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--threads", type=int, default=4)
    p.set_defaults(func=cmd_insample)
    p = sub.add_parser("hppnet")
    p.add_argument("--v4", required=True, help="a DREGON v4 flight payload")
    p.add_argument("--v3", required=True, help="the nv3r4 DREGON payload")
    p.add_argument("--window", type=int, default=1, help="index into DREGON_CRUISE_SUPPORTS")
    p.add_argument("--mics", default="1,2")
    p.add_argument("--line-db", type=float, default=MATCHED_LINE_DB)
    p.add_argument("--fmax", type=float, default=1000.0)
    p.add_argument("--out", default=str(OUT))
    p.set_defaults(func=cmd_hppnet)
    p = sub.add_parser("full")
    p.add_argument("--rig", choices=("dregon", "michaels"), default="dregon")
    p.add_argument("--v4", required=True, help="the v4 payload (MD2: the cruise half)")
    p.add_argument("--v3", required=True, help="the nv3r4 payload (MD2: the cruise half)")
    p.add_argument("--v4-standby", default=None, help="MD2: the v4 standby half")
    p.add_argument("--v3-standby", default=None, help="MD2: the nv3r4 standby half")
    p.add_argument("--line-db", type=float, default=None, help="add a v4 arm with lines +dB")
    p.add_argument("--mics", default=None, help="default: DREGON 1,2; MD2 0,4")
    p.add_argument("--seconds", type=float, default=FULL_SECONDS)
    p.add_argument("--out", default=str(OUT))
    p.set_defaults(func=cmd_full)
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
