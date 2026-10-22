#!/usr/bin/env python3
"""Noise model v4 — why the DREGON score is bad: real vs nv3r4 vs a v4 bench fit
on the five frozen DREGON cruise scoring windows, plus the v4 bench fit against
its own bench recordings (``docs/explainers/noise-model-v4-dregon-gap.qmd``)::

    python scripts/noise_v4_dregon_gap.py flight --v4 <bench payload> --v3 <nv3r4 payload>
    python scripts/noise_v4_dregon_gap.py insample --v4 <bench payload>

``flight`` renders each window from both fits on the window's raw telemetry
(the scorer's carrier, render seed 2001, all eight mics), and writes Welch
spectra (mic 0 and the mic mean), the scorer's band LTAS, per-mic level offsets,
spectrograms of the first window, the v4 expected periodogram split into lines
and floor, and WAVs of mic 0. ``insample`` evaluates the bench payload's model
on its own recordings with the bench fit's forward model; run it with the
``src/`` of the commit that produced the payload first on ``PYTHONPATH`` (the
payload must be read by its own code). Outputs: ``--out`` (default
``docs/explainers/noise-model-v4-dregon-gap``) ``figdata_<mode>.json``,
``fig_*.png``, ``audio/*.wav``.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
    from data_processing.noise_model.render import render_noise
    from experiments.noise_model import gates as GT
    from experiments.stochastic_fit import accept_stats
    from experiments.stochastic_fit import revised_eval as RE

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fits = {
        "nv3r4": json.loads(Path(args.v3).read_text()),
        "v4": json.loads(Path(args.v4).read_text()),
    }
    edges = _octave_bands()
    centres = np.sqrt(edges[:-1] * edges[1:])
    ltas_bands = [list(b) for b in accept_stats.BANDS]
    rows: list[dict[str, Any]] = []
    spectra: dict[str, dict[str, np.ndarray]] = {}
    first: dict[str, np.ndarray] = {}
    decomp: dict[str, Any] = {}
    for i, sup in enumerate(GT.DREGON_CRUISE_SUPPORTS):
        clip = RE.load_window(
            sup.window,
            dataset=GT.DATASET["dregon"],
            version=None,
            channels=None,
            rps_key=GT.RAW_RPS_KEY["dregon"],
        )
        real = np.asarray(clip.audio, dtype=np.float64)[:8]
        rps = np.atleast_2d(np.asarray(clip.rps, dtype=np.float64))
        audio = {"real": real}
        for name, fit in fits.items():
            audio[name] = np.asarray(render_noise(fit, rps, n_mics=8, seed=SEED), dtype=np.float64)[
                :, : real.shape[1]
            ]
        f, _ = _welch(real[0])
        key = sup.recording.replace("_nosource_room2", "")
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
        for name in fits:
            dev = RE.ltas_deviation_db(real, audio[name], mic=0)
            row[f"scorer_{name}"] = {
                k: dev[k]
                for k in ("mean_abs_db", "max_abs_db", "level_offset_db", "shape_only_mean_abs_db")
            }
        # the v4 expected periodogram at the window's mean speeds, total and parts
        rbar = rps.mean(axis=1)
        lines_only = copy.deepcopy(fits["v4"])
        lines_only["params"]["floor"]["mean_db"] = -300.0
        floor_only = copy.deepcopy(fits["v4"])
        pr = floor_only["params"]
        pr["profile_aero_db"] = (np.asarray(pr["profile_aero_db"]) - 300.0).tolist()
        pr["profile_motor_db"] = (np.asarray(pr["profile_motor_db"]) - 300.0).tolist()
        n_fft = NPERSEG
        fe = np.fft.rfftfreq(n_fft, 1.0 / SR)
        parts = {
            "total": V4.expected_periodogram(fits["v4"], rbar, n_fft=n_fft, sr=SR),
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
            scale = 0.5 / float(np.max(np.abs(real[0])))
            for name, x in audio.items():
                _wav(out / "audio" / f"{key}_{name}_mic0.wav", x[0], scale)
        print(
            f"{key}: " + ", ".join(f"{n} {row[f'scorer_{n}']['mean_abs_db']:.2f} dB" for n in fits),
            flush=True,
        )

    names = ["real", "nv3r4", "v4"]
    keys = [r["key"] for r in rows]

    # Fig 1: Welch PSD, mic 0, each window
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 2.6 * len(keys)), sharex=True)
    for ax, key in zip(axes, keys):
        for n in names:
            ax.semilogx(f, _db(spectra[key][n][0]), color=COLORS[n], lw=0.8, label=n)
        ax.set_xlim(30, 8000)
        ax.set_ylabel("PSD (dB)")
        ax.set_title(f"{key}: mic 0", fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("frequency (Hz)")
    fig.tight_layout()
    fig.savefig(out / "fig_psd_mic0.png", dpi=130)
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
        ax.set_title("mic 0" if which == "mic0" else "mean over the 8 mics", fontsize=10)
        ax.set_xlabel("third-octave band centre (Hz)")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("model − real (dB)")
    axes[0].plot([], [], color=COLORS["nv3r4"], label="nv3r4 (5 windows)")
    axes[0].plot([], [], color=COLORS["v4"], label="v4 round 4 (5 windows)")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "fig_band_error.png", dpi=130)
    plt.close(fig)

    # Fig 3: per-mic level offset (31-7900 Hz mean PSD, model - real)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for n, off in (("nv3r4", -0.12), ("v4", 0.12)):
        for j, key in enumerate(keys):
            r = next(x for x in rows if x["key"] == key)
            d = np.asarray(r[f"level_db_per_mic_{n}"]) - np.asarray(r["level_db_per_mic_real"])
            ax.plot(
                np.arange(8) + off,
                d,
                "o",
                color=COLORS[n],
                ms=4,
                alpha=0.7,
                label=n if j == 0 else None,
            )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(8))
    ax.set_xlabel("microphone")
    ax.set_ylabel("broadband level, model − real (dB)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_mic_offsets.png", dpi=130)
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
    fig.savefig(out / "fig_spectrogram.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Fig 5: v4 decomposition vs real, first window, mic 0
    key = keys[0]
    dk = decomp[key]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, xlim in zip(axes, ((30, 8000), (30, 600))):
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
        ax.plot(dk["f"], _db(dk["floor"][0]), color="tab:green", lw=1.0, label="v4 expected: floor")
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.set_xlabel("frequency (Hz)")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("PSD (dB)")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"{key}: mic 0, 30 Hz – 8 kHz", fontsize=10)
    axes[1].set_title("30 – 600 Hz", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_v4_parts.png", dpi=130)
    plt.close(fig)

    data = dict(
        git=_git(),
        v4=str(args.v4),
        v3=str(args.v3),
        seed=SEED,
        welch=dict(nperseg=NPERSEG, overlap=0.75, window="hann", sr=SR),
        third_octave_centres_hz=centres.round(2).tolist(),
        ltas_bands_hz=ltas_bands,
        windows=rows,
    )
    (out / "figdata_flight.json").write_text(json.dumps(data, indent=1))
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
    settings = [B.load_setting(k.split("_")[1], int(k.split("_")[2])) for k in want]
    rig = B.build_rig(tuple(dict.fromkeys(s.motor for s in settings)), settings=settings)
    model = B.RigModel(rig, device="cpu", kappa=float(pay["priors"]["floor_push_kappa"]))
    model.load_payload(pay)
    edges = _octave_bands(per_oct=6)
    centres = np.sqrt(edges[:-1] * edges[1:])
    rows = []
    fig, axes = plt.subplots(len(want), 2, figsize=(13, 2.7 * len(want)))
    with torch.no_grad():
        for j, (s, fz) in enumerate(zip(rig.settings, model.frozen)):
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
            ax = axes[j, 0]
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
            ax = axes[j, 1]
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("flight")
    p.add_argument("--v4", required=True)
    p.add_argument("--v3", required=True)
    p.add_argument("--out", default=str(OUT))
    p.set_defaults(func=cmd_flight)
    p = sub.add_parser("insample")
    p.add_argument("--v4", required=True)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--threads", type=int, default=4)
    p.set_defaults(func=cmd_insample)
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
