"""Build the S1 listening assets: real | fitted | pre-fit, panels and WAVs.

Criterion 4 of the objective: statistics alone never close a stage, so every
stage ships spectrogram panels AND audio for at least four clips. ``pre-fit`` is
a draw from the 2025 default ``StochasticRanges`` at the same rotor speed — the
family as it stood before any bench measurement — so the page shows what the
fit actually bought.

Run: ``PYTHONPATH=src python docs/explainers/stage1-bench/build.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from data_processing import stochastic_rotor_noise as srn
from experiments.stochastic_fit import accept_stats as stats
from experiments.stochastic_fit import native, stage1

OUT = Path(__file__).resolve().parent
#: Two held-out cells and two fit cells, slow and fast of each.
CELLS = ((4, 50), (4, 80), (1, 60), (3, 90))
SECONDS = 6.0


def prefit_render(rate: float, seconds: float, seed: int) -> np.ndarray:
    """A draw from the untouched default ranges, at the same rotor speed."""
    rng = np.random.default_rng(seed)
    k_max = max(2, int(np.floor((native.NATIVE_SR / 2) / max(rate, 1.0))))
    params = srn.sample_params(
        rng,
        srn.StochasticRanges(),
        n_rotors=1,
        n_harmonics=min(k_max, 200),
        sample_rate=native.NATIVE_SR,
    )
    rps = np.full((1, int(round(seconds * native.NATIVE_SR))), float(rate))
    audio, _ = srn.synthesize(params, rps, rng=rng, n_mics=1, line_mode="fm")
    clip = stage1.Clip(
        "prefit", "synthetic", np.asarray(audio, dtype=np.float32), rps, native.NATIVE_SR, None, {}
    )
    return np.asarray(native.decimate(clip, stage1.SR).audio[0], dtype=np.float64)


def spectrogram(ax, x: np.ndarray, sr: int, title: str) -> None:
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
        extent=[0, x.size / sr, 0, sr / 2000],
        vmin=v - 75,
        vmax=v,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("s")


def main() -> None:
    fit = stage1.load("results/S1/fit.json")
    index = []
    for motor, speed in CELLS:
        m = stage1.measure_cell(motor, speed)
        rate = m.rate_rps
        real_clip = native.decimate(
            native.bench_clip(motor, speed, duration_s=SECONDS, start_s=3.0), stage1.SR
        )
        xr = real_clip.audio[stage1.CHANNEL].astype(np.float64)
        xf = stage1.render(fit, rate, seconds=SECONDS, seed=4242)
        xp = prefit_render(rate, SECONDS, seed=4242)

        slug = f"Motor{motor}_{speed}"
        row = {
            "slug": slug,
            "motor": motor,
            "setpoint": speed,
            "rate": round(rate, 2),
            "held_out": motor in stage1.HELD_OUT_MOTORS,
            "bands": {},
        }
        fig, axes = plt.subplots(
            1, 4, figsize=(20, 4.2), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]}
        )
        for ax, (x, name) in zip(
            axes[:3],
            ((xr, "real (channel 7)"), (xf, "fitted S1 preset"), (xp, "pre-fit default family")),
            strict=True,
        ):
            spectrogram(ax, x, stage1.SR, f"{slug} — {name}")
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
        axes[3].set_xscale("log")
        axes[3].set_xlabel("kHz")
        axes[3].set_ylabel("dB re 200-400 Hz")
        axes[3].grid(alpha=0.3)
        axes[3].legend(fontsize=8)
        axes[3].set_title(
            f"LTAS bands — fitted mean|dev| "
            f"{np.abs(np.array(row['bands']['fitted']) - np.array(row['bands']['real'])).mean():.2f} dB, "
            f"pre-fit "
            f"{np.abs(np.array(row['bands']['pre-fit']) - np.array(row['bands']['real'])).mean():.2f} dB",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(OUT / f"{slug}.png", dpi=100)
        plt.close(fig)

        for x, name in ((xr, "real"), (xf, "fitted"), (xp, "prefit")):
            y = x / max(float(np.abs(x).max()), 1e-9) * 0.7
            sf.write(OUT / f"{slug}_{name}.wav", y.astype(np.float32), stage1.SR)

        def pooled(x: np.ndarray, rate: float = rate) -> dict[str, float]:
            return {
                k: round(v, 2)
                for k, v in stats.pooled_excess(
                    x, rate, gamma0=fit.gamma0_hz, gamma_slope=fit.gamma_slope_hz
                ).items()
            }

        row["orders"] = {"real": pooled(xr), "fitted": pooled(xf), "pre-fit": pooled(xp)}
        index.append(row)
        print(
            f"{slug}: rate {rate:.1f} rev/s  fitted mean|dev| "
            f"{np.abs(np.array(row['bands']['fitted']) - np.array(row['bands']['real'])).mean():.2f} dB, "
            f"pre-fit {np.abs(np.array(row['bands']['pre-fit']) - np.array(row['bands']['real'])).mean():.2f} dB",
            flush=True,
        )

    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    print(f"\nwrote {len(index)} panels, {3 * len(index)} wavs, index.json")


if __name__ == "__main__":
    main()
