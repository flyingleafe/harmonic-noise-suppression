"""S1 listening assets from the BAYESIAN fit: real | fitted | pre-fit.

The fitted object is the project's generative model — one harmonic comb plus a
smoothly coloured broadband floor — with every dynamic term pinned off, so the
fitted vector IS a ``StochasticParams``. "Fitted" below is that parameter set
rendered through ``stochastic_rotor_noise.synthesize``, the same renderer the
training streams use. "Pre-fit" is a draw from the untouched default
``StochasticRanges`` at the same rotor speed.

Run: ``PYTHONPATH=src python docs/explainers/stage1-bench/build_bayes.py``
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from data_processing import stochastic_rotor_noise as srn
from experiments.stochastic_fit import accept_stats as stats
from experiments.stochastic_fit import clips as C
from experiments.stochastic_fit import stage1_bayes as SB
from experiments.stochastic_fit.data import Clip

OUT = Path(__file__).resolve().parent
FIT_JSON = os.environ.get("FIT_JSON", "results/S1/bayes_motor1_needle.json")
#: The bench cells come from the published frames dataset; override to pin a
#: version (``DREGON-frames@<hash>``) when a page must be rebuilt exactly.
BENCH_DATASET = os.environ.get("BENCH_DATASET", SB.BENCH_DATASET)


def prefit(rate: float, seconds: float, seed: int) -> np.ndarray:
    """A draw from the default ranges at the same rotor speed."""
    rng = np.random.default_rng(seed)
    k_max = max(2, int(np.floor((C.NATIVE_SR / 2) / max(rate, 1.0))))
    params = srn.sample_params(
        rng,
        srn.StochasticRanges(),
        n_rotors=1,
        n_harmonics=min(k_max, 200),
        sample_rate=C.NATIVE_SR,
    )
    rps = np.full((1, int(round(seconds * C.NATIVE_SR))), float(rate))
    audio, _ = srn.synthesize(params, rps, rng=rng, n_mics=1, line_mode="fm", n_fft=SB.OLA_N_FFT)
    clip = Clip("prefit", "synthetic", np.asarray(audio, np.float32), rps, C.NATIVE_SR, None, {})
    return np.asarray(C.decimate(clip, SB.SR).audio[0], dtype=np.float64)


def spec_panel(ax, x: np.ndarray, title: str) -> None:
    n, hop = 2048, 512
    w = np.hanning(n + 1)[:n]
    frames = np.stack([x[s : s + n] * w for s in range(0, x.size - n, hop)])
    S = 20 * np.log10(np.abs(np.fft.rfft(frames, axis=-1)).T + 1e-10)
    v = np.percentile(S, 99.5)
    ax.imshow(
        S,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, x.size / SB.SR, 0, SB.SR / 2000],
        vmin=v - 75,
        vmax=v,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("s")


def main() -> None:
    fit = json.loads(Path(FIT_JSON).read_text())
    index = []
    for cell, entry in fit["clips"].items():
        p = entry["params"]
        rate = float(np.asarray(p["carrier"]).mean())
        motor, speed = int(cell[5]), int(cell.split("_")[1])
        real = SB.bench_clip(motor, speed, dataset=BENCH_DATASET)
        dur = float(real.meta["duration_s"])
        xr = np.asarray(real.audio[0], dtype=np.float64)
        xf = SB.render_from_export(p, rate, seconds=dur, seed=4242)
        xp = prefit(rate, dur, seed=4242)

        g0 = float(np.atleast_1d(p["gamma0"])[0])
        gs = float(np.atleast_1d(p["gamma_slope"])[0])
        row = {
            "cell": cell,
            "rate": round(rate, 2),
            "excess_over_loo": entry["scores"]["excess_over_loo"],
            "gamma0_hz": g0,
            "gamma_slope_hz": gs,
            "bands": {},
            "orders": {},
        }
        fig, axes = plt.subplots(
            1, 4, figsize=(20, 4.2), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]}
        )
        for ax, (x, name) in zip(
            axes[:3],
            ((xr, "real (channel 7)"), (xf, "fitted comb + floor"), (xp, "pre-fit default family")),
            strict=True,
        ):
            spec_panel(ax, x, f"{cell} — {name}")
        axes[0].set_ylabel("kHz")

        centres = [(lo + hi) / 2000 for lo, hi in stats.BANDS]
        for x, name, colour in (
            (xr, "real", "#111111"),
            (xf, "fitted", "#1f77b4"),
            (xp, "pre-fit", "#d62728"),
        ):
            b = stats.ltas_bands(x)
            axes[3].plot(centres, b, marker="o", ms=4, lw=1.4, color=colour, label=name)
            row["bands"][name] = np.round(b, 2).tolist()
        dev_f = np.asarray(row["bands"]["fitted"]) - np.asarray(row["bands"]["real"])
        dev_p = np.asarray(row["bands"]["pre-fit"]) - np.asarray(row["bands"]["real"])
        row["fitted_mean_abs_db"] = float(np.abs(dev_f).mean())
        row["fitted_max_abs_db"] = float(np.abs(dev_f).max())
        row["prefit_mean_abs_db"] = float(np.abs(dev_p).mean())
        row["orders"] = {
            k: float(v)
            for k, v in stats.paired_delta(xr, xf, rate, gamma0=g0, gamma_slope=gs).items()
        }
        axes[3].set_xscale("log")
        axes[3].set_xlabel("kHz")
        axes[3].set_ylabel("dB re 200-400 Hz")
        axes[3].grid(alpha=0.3)
        axes[3].legend(fontsize=8)
        axes[3].set_title(
            f"LTAS — fitted mean|dev| {row['fitted_mean_abs_db']:.2f} dB, "
            f"pre-fit {row['prefit_mean_abs_db']:.2f} dB",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(OUT / f"bayes_{cell}.png", dpi=100)
        plt.close(fig)

        for x, name in ((xr, "real"), (xf, "fitted"), (xp, "prefit")):
            y = x / max(float(np.abs(x).max()), 1e-9) * 0.7
            sf.write(OUT / f"bayes_{cell}_{name}.wav", y.astype(np.float32), SB.SR)
        index.append(row)
        print(
            f"{cell}: rate {rate:.1f}  fitted {row['fitted_mean_abs_db']:.2f} dB  "
            f"pre-fit {row['prefit_mean_abs_db']:.2f} dB  "
            f"orders {row['orders']['k2_15']:+.2f}/{row['orders']['k15_30']:+.2f}/"
            f"{row['orders']['k30_100']:+.2f}",
            flush=True,
        )

    (OUT / "index_bayes.json").write_text(json.dumps(index, indent=1))
    print(f"\nwrote {len(index)} panels, {3 * len(index)} wavs, index_bayes.json")


if __name__ == "__main__":
    main()
