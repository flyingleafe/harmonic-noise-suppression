"""Visual and statistical comparison for the Stage-2 cruise fit.

For each fitted cruise window: real | fitted | pre-fit spectrograms of the same
microphone, the LTAS band profile of all three, and the per-microphone level
pattern, which is the part a four-rotor eight-microphone fit has to get right
and a single-channel fit cannot even represent.

    python scripts/_stage2_panels.py --fit results/S2/cruise_8clip.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from data_processing import stochastic_rotor_noise as srn
from experiments.stochastic_fit import accept_stats as stats
from experiments.stochastic_fit import native
from experiments.stochastic_fit import stage2 as S2

OUT = Path("docs/explainers/stage2-cruise")


def prefit(rps: np.ndarray, n_mics: int, seed: int) -> np.ndarray:
    """A draw from the untouched default family at the same rotor speeds."""
    rng = np.random.default_rng(seed)
    p = srn.sample_params(
        rng,
        srn.StochasticRanges(),
        n_rotors=rps.shape[0],
        n_harmonics=S2.K_CAP,
        sample_rate=native.NATIVE_SR,
    )
    n_native = int(round(rps.shape[-1] / S2.SR * native.NATIVE_SR))
    t_src = np.linspace(0.0, 1.0, rps.shape[-1])
    t_dst = np.linspace(0.0, 1.0, n_native)
    rps_native = np.stack([np.interp(t_dst, t_src, r) for r in rps])
    audio, _ = srn.synthesize(p, rps_native, rng=rng, n_mics=n_mics, line_mode="fm", n_fft=1 << 16)
    from experiments.stochastic_fit.data import Clip

    clip = Clip("prefit", "synthetic", np.asarray(audio, np.float32), rps_native, native.NATIVE_SR)
    return np.asarray(native.decimate(clip, S2.SR).audio, dtype=np.float64)


def spec_panel(ax, x: np.ndarray, title: str) -> None:
    n, hop = 2048, 512
    w = np.hanning(n + 1)[:n]
    fr = np.stack([x[s : s + n] * w for s in range(0, x.size - n, hop)])
    S = 20 * np.log10(np.abs(np.fft.rfft(fr, axis=-1)).T + 1e-9)
    v = np.percentile(S, 99.5)
    ax.imshow(
        S,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, x.size / S2.SR, 0, S2.SR / 2000],
        vmin=v - 70,
        vmax=v,
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=Path, default=Path("results/S2/cruise_8clip.json"))
    ap.add_argument("--mic", type=int, default=0)
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=8.0)
    args = ap.parse_args()

    summary = json.loads(args.fit.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for i, (cid, entry) in enumerate(list(summary["clips"].items())[: args.clips]):
        p = entry["params"]
        spec = S2.cruise_windows(max_clips=40, stride_s=args.seconds)
        start_s, dur = spec[i % len(spec)]
        real_clip = native.decimate(
            native.load_native_clip(
                S2.FIT_RECORDING, start_s, min(dur, args.seconds), clip_id=f"panel_{i:02d}"
            ),
            S2.SR,
        )
        rps = np.asarray(real_clip.rps, dtype=np.float64)
        xr_all = np.asarray(real_clip.audio, dtype=np.float64)
        xf_all = S2.render_from_export(p, rps, n_mics=xr_all.shape[0], seed=900 + i)
        xp_all = prefit(rps, xr_all.shape[0], seed=900 + i)
        n = min(xr_all.shape[-1], xf_all.shape[-1], xp_all.shape[-1])
        xr, xf, xp = xr_all[args.mic, :n], xf_all[args.mic, :n], xp_all[args.mic, :n]

        fig, axes = plt.subplots(
            1, 5, figsize=(24, 4.2), gridspec_kw={"width_ratios": [1, 1, 1, 1.1, 0.9]}
        )
        for ax, (x, name) in zip(
            axes[:3],
            (
                (xr, f"real (mic {args.mic})"),
                (xf, "fitted: coherent tones + stochastic floor"),
                (xp, "pre-fit default family"),
            ),
            strict=True,
        ):
            spec_panel(ax, x, f"{cid} — {name}")
        axes[0].set_ylabel("kHz")

        centres = [(lo + hi) / 2000 for lo, hi in stats.BANDS]
        bands = {}
        for x, name, colour in (
            (xr, "real", "#111111"),
            (xf, "fitted", "#1f77b4"),
            (xp, "pre-fit", "#d62728"),
        ):
            b = stats.ltas_bands(x)
            bands[name] = np.round(b, 2).tolist()
            axes[3].plot(centres, b, marker="o", ms=4, lw=1.4, color=colour, label=name)
        dev_f = np.abs(np.asarray(bands["fitted"]) - np.asarray(bands["real"]))
        dev_p = np.abs(np.asarray(bands["pre-fit"]) - np.asarray(bands["real"]))
        axes[3].set_xscale("log")
        axes[3].set_xlabel("kHz")
        axes[3].set_ylabel("dB re 200-400 Hz")
        axes[3].grid(alpha=0.3)
        axes[3].legend(fontsize=8)
        axes[3].set_title(
            f"LTAS — fitted mean|dev| {dev_f.mean():.2f} dB, pre-fit {dev_p.mean():.2f} dB",
            fontsize=9,
        )

        # the per-microphone level pattern: what a four-rotor rig fit owns
        lv_r = 10 * np.log10((xr_all[:, :n] ** 2).mean(axis=1) + 1e-20)
        lv_f = 10 * np.log10((xf_all[:, :n] ** 2).mean(axis=1) + 1e-20)
        axes[4].plot(lv_r - lv_r.mean(), "o-", color="#111111", label="real")
        axes[4].plot(lv_f - lv_f.mean(), "s--", color="#1f77b4", label="fitted")
        axes[4].set_xlabel("microphone")
        axes[4].set_ylabel("dB re channel mean")
        axes[4].grid(alpha=0.3)
        axes[4].legend(fontsize=8)
        axes[4].set_title(
            f"channel pattern — rms dev {np.abs((lv_r - lv_r.mean()) - (lv_f - lv_f.mean())).mean():.2f} dB",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(OUT / f"cruise_{i:02d}.png", dpi=100)
        plt.close(fig)

        for x, name in ((xr, "real"), (xf, "fitted"), (xp, "prefit")):
            y = x / max(float(np.abs(x).max()), 1e-9) * 0.7
            sf.write(OUT / f"cruise_{i:02d}_{name}.wav", y.astype(np.float32), S2.SR)

        row = dict(
            clip=cid,
            start_s=start_s,
            rotors=np.round(rps.mean(axis=1), 1).tolist(),
            k_half=round(float(p.get("coherence_k_half", 0.0)), 2),
            bands=bands,
            fitted_mean_abs_db=round(float(dev_f.mean()), 2),
            fitted_max_abs_db=round(float(dev_f.max()), 2),
            prefit_mean_abs_db=round(float(dev_p.mean()), 2),
            channel_rms_dev_db=round(
                float(np.abs((lv_r - lv_r.mean()) - (lv_f - lv_f.mean())).mean()), 2
            ),
            figure=f"cruise_{i:02d}.png",
        )
        index.append(row)
        print(
            f"{cid}: rotors {row['rotors']} k_half {row['k_half']}  "
            f"LTAS fitted {row['fitted_mean_abs_db']} dB (pre-fit {row['prefit_mean_abs_db']})  "
            f"channels {row['channel_rms_dev_db']} dB",
            flush=True,
        )

    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    print(f"\nwrote {len(index)} panels and {3 * len(index)} wavs to {OUT}")


if __name__ == "__main__":
    main()
