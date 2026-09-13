"""A/B renders: two fitted parameter vectors on the SAME real window.

Every arm of a window is written with ONE common gain, so the fitted level
difference survives into the audio.

The point is to hear what a fit's parameters sound like, with everything else
held equal — the same real clip, the same rotor trajectory, the same seed, the
same resampling path. Any difference is the parameter vector.

    python scripts/_ab_render.py --slug fly125 \
        --dataset michaels-frames --recording FLY125 --windows 0,1 \
        --fit results/S2/cruise_8clip_refined.json --label refined \
        --fit results/S2/cruise_8clip.json --label raw

Writes `docs/explainers/refit/ab_<slug>_<NN>_{real,<label>...}.wav`, one
comparison figure per window, and `ab_<slug>.json` with the LTAS deviations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from experiments.stochastic_fit import accept_stats as stats
from experiments.stochastic_fit import clips as C
from experiments.stochastic_fit import stage2 as S2

OUT = Path("docs/explainers/refit")
COLOURS = {"real": "#111111", "refined": "#1f77b4", "raw": "#d62728", "pre-fit": "#7f7f7f"}


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
        vmin=v - 75,
        vmax=v,
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("s")


def window_of(summary: dict[str, Any], index: int) -> tuple[str, float, float, str]:
    """``(recording, start_s, seconds, clip_id)`` of one fitted window."""
    data = summary.get("data") or {}
    cids = list(summary["clips"])
    cid = cids[index]
    if data.get("starts_s"):
        regime = str(data.get("regime", "cruise"))
        known = [str(r).rpartition(":")[2] for r in data.get("recordings", [])]
        rec = next((k for k in known if cid.startswith(f"{k.lower()}_{regime}_")), known[0])
        return rec, float(data["starts_s"][index]), float(data.get("seconds", 16.0)), cid
    raise SystemExit(f"{cid}: this summary records no window starts; pass --start explicitly")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--dataset", default=S2.FIT_DATASET)
    ap.add_argument("--version", default=None)
    ap.add_argument("--recording", default=None, help="default: the first fit's own recording")
    ap.add_argument(
        "--windows", default="0", help="comma-separated window indices of the FIRST fit"
    )
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--channels", default=None)
    ap.add_argument("--mic", type=int, default=0)
    ap.add_argument("--rps-key", default=C.DEFAULT_RPS_KEY)
    ap.add_argument("--fit", action="append", required=True, type=Path)
    ap.add_argument("--label", action="append", required=True)
    ap.add_argument(
        "--clip-index",
        action="append",
        type=int,
        default=None,
        help="which clip of each fit supplies the parameters (default: the window index, clamped)",
    )
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    fits = [json.loads(p.read_text()) for p in args.fit]
    labels = list(args.label)
    if len(fits) != len(labels):
        raise SystemExit("--fit and --label must come in pairs")

    index: list[dict[str, Any]] = []
    for w in (int(v) for v in str(args.windows).split(",")):
        rec, start_s, seconds, cid = window_of(fits[0], w)
        rec = args.recording or rec
        seconds = min(seconds, args.seconds)
        real_clip = C.decimate(
            C.load_clip(
                args.dataset,
                rec,
                start_s,
                seconds,
                version=args.version,
                channels=args.channels,
                rps_key=args.rps_key,
                clip_id=f"ab_{rec}_{start_s:.2f}_{seconds:g}",
            ),
            S2.SR,
        )
        rps = np.asarray(real_clip.rps, dtype=np.float64)
        xr_all = np.asarray(real_clip.audio, dtype=np.float64)
        waves: dict[str, np.ndarray] = {"real": xr_all}
        for j, (summary, label) in enumerate(zip(fits, labels, strict=True)):
            n_clips = len(summary["clips"])
            k = (
                args.clip_index[j]
                if args.clip_index and j < len(args.clip_index)
                else min(w, n_clips - 1)
            )
            export = list(summary["clips"].values())[k]["params"]
            # the SAME seed for every arm: the difference is the parameters
            waves[label] = S2.render_from_export(export, rps, n_mics=xr_all.shape[0], seed=4242 + w)
        n = min(v.shape[-1] for v in waves.values())
        waves = {k: v[:, :n] for k, v in waves.items()}

        fig, axes = plt.subplots(
            1,
            len(waves) + 1,
            figsize=(5.2 * (len(waves) + 1), 4.2),
            gridspec_kw={"width_ratios": [1] * len(waves) + [1.15]},
        )
        bands: dict[str, list[float]] = {}
        centres = [(lo + hi) / 2000 for lo, hi in stats.BANDS]
        for ax, (name, wav) in zip(axes[: len(waves)], waves.items(), strict=True):
            spec_panel(ax, wav[args.mic], f"{cid} — {name}")
            b = stats.ltas_bands(wav[args.mic])
            bands[name] = np.round(b, 2).tolist()
            axes[-1].plot(
                centres,
                b,
                marker="o",
                ms=4,
                lw=1.4,
                color=COLOURS.get(name, "#2ca02c"),
                label=name,
            )
        axes[0].set_ylabel("kHz")
        axes[-1].set_xscale("log")
        axes[-1].set_xlabel("kHz")
        axes[-1].set_ylabel("dB re 200-400 Hz")
        axes[-1].grid(alpha=0.3)
        axes[-1].legend(fontsize=8)
        devs = {
            name: float(np.abs(np.asarray(b) - np.asarray(bands["real"])).mean())
            for name, b in bands.items()
            if name != "real"
        }
        axes[-1].set_title(
            "LTAS mean|dev| vs real: " + ", ".join(f"{k} {v:.2f} dB" for k, v in devs.items()),
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(OUT / f"ab_{args.slug}_{w:02d}.png", dpi=100)
        plt.close(fig)

        # ONE gain for every arm of a window. Normalising each arm to its own
        # peak would erase exactly what the comparison is about: the fitted
        # levels. The gain is written into the index so it can be quoted.
        peak = max(float(np.abs(wav[args.mic]).max()) for wav in waves.values())
        gain = 0.7 / max(peak, 1e-9)
        for name, wav in waves.items():
            y = wav[args.mic] * gain
            sf.write(OUT / f"ab_{args.slug}_{w:02d}_{name}.wav", y.astype(np.float32), S2.SR)

        # the per-microphone level pattern, which is where the arms differ most
        pattern = {}
        for name, wav in waves.items():
            lv = 10 * np.log10((wav**2).mean(axis=1) + 1e-20)
            pattern[name] = np.round(lv - lv.mean(), 2).tolist()
        row = dict(
            slug=args.slug,
            window=w,
            clip=cid,
            recording=rec,
            start_s=start_s,
            seconds=seconds,
            rotors=np.round(rps.mean(axis=1), 1).tolist(),
            ltas_mean_abs_db=devs,
            channel_pattern_db=pattern,
            channel_rms_dev_db={
                name: round(float(np.abs(np.asarray(v) - np.asarray(pattern["real"])).mean()), 2)
                for name, v in pattern.items()
                if name != "real"
            },
            playback_gain=round(float(gain), 4),
            peak_dbfs={
                name: round(float(20 * np.log10(max(np.abs(wav[args.mic]).max(), 1e-12) * gain)), 2)
                for name, wav in waves.items()
            },
            figure=f"ab_{args.slug}_{w:02d}.png",
            arms=list(waves),
        )
        index.append(row)
        print(
            f"{cid}: "
            + "  ".join(f"{k} LTAS {v:.2f} dB" for k, v in devs.items())
            + "  channels "
            + str(row["channel_rms_dev_db"]),
            flush=True,
        )

    (OUT / f"ab_{args.slug}.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {OUT / f'ab_{args.slug}.json'}")


if __name__ == "__main__":
    main()
