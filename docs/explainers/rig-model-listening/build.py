"""Build the assets of ``docs/explainers/rig-model-listening.qmd``: for every
chosen real clip, the real audio (one microphone), a draw of the matching rig
preset rendered on the SAME rotor-speed trajectories, and — for contrast — a
draw of the old hand-ranged family (``conf/online_mix/salv2_stoch.yaml``
defaults, stochastic line mode). Audio is level-matched; spectrograms share
one colour scale per row.

Run from the repo root::

    PYTHONPATH=src python docs/explainers/rig-model-listening/build.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import yaml

from data_processing import stochastic_rotor_noise as srn
from experiments.stochastic_fit import diagnostics as D

HERE = Path(__file__).resolve().parent
POLICY = HERE.parents[2] / "conf/online_mix/rig_fm_5050.yaml"
SR = 16000
CLIPS = {
    "dregon": [
        ("hovering_nosource_room2_03", "hover, room 2"),
        ("free-flight_nosource_room2_04", "free flight, room 2"),
        ("free-flight_nosource_room1_sample_00015", "spin-up ramp 2 → 90 rev/s, room 1"),
        ("free-flight_nosource_room1_sample_00017", "cruise, room 1 (measured shaft rate)"),
    ],
    "michaels": [
        ("FLY125_00", "idle ≈ 35 rev/s, FLY125"),
        ("FLY125_04", "cruise ≈ 80 rev/s, FLY125"),
        ("michaels_FLY124_sample_00022", "take-off ramp 0 → 50 rev/s, FLY124"),
        ("michaels_FLY124_sample_00029", "cruise, FLY124 (8 s)"),
    ],
}
MIC = 0
TARGET_RMS = 0.05


def level(x: np.ndarray) -> np.ndarray:
    return (x / (np.sqrt(np.mean(x**2)) + 1e-9) * TARGET_RMS).astype(np.float32)


def spectrogram_db(x: np.ndarray, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    win = np.hanning(n_fft + 1)[:n_fft]
    starts = np.arange(0, x.size - n_fft + 1, hop)
    frames = np.stack([x[s : s + n_fft] * win for s in starts])
    return 10.0 * np.log10(np.abs(np.fft.rfft(frames, axis=-1)) ** 2 + 1e-12).T


def main() -> None:
    pol = yaml.safe_load(POLICY.read_text())
    presets = {"dregon": pol["sources"]["noise"][0], "michaels": pol["sources"]["noise"][1]}
    old_ranges = srn.StochasticRanges()
    rng = np.random.default_rng(20260909)
    index: list[dict] = []
    for rig, items in CLIPS.items():
        src = presets[rig]
        ranges = srn.StochasticRanges.from_dict(src["ranges"])
        for cid, label in items:
            clip = D.load_clip_cached(cid)
            rps = clip.rps.astype(np.float64)
            n_harm = int(
                np.clip(
                    np.ceil(
                        SR
                        / 2
                        / max(float(np.median(rps[rps > 5])) if (rps > 5).any() else 80.0, 1.0)
                    ),
                    40,
                    200,
                )
            )
            params = srn.sample_params(rng, ranges, n_rotors=4, n_harmonics=n_harm, sample_rate=SR)
            new, _ = srn.synthesize(
                params,
                rps,
                rng=rng,
                n_mics=8,
                mic_gain_db=tuple(src["mic_gain_db"]),
                line_mode="fm",
            )
            old_params = srn.sample_params(
                rng, old_ranges, n_rotors=4, n_harmonics=n_harm, sample_rate=SR
            )
            old, _ = srn.synthesize(
                old_params, rps, rng=rng, n_mics=8, mic_gain_db=(-12.0, 0.0), line_mode="stochastic"
            )
            trio = {"real": level(clip.audio[MIC]), "rig": level(new[MIC]), "old": level(old[MIC])}
            slug = f"{rig}_{cid}"
            for name, x in trio.items():
                sf.write(HERE / f"{slug}_{name}.wav", x, SR)
            specs = {k: spectrogram_db(v) for k, v in trio.items()}
            vmax = max(np.percentile(s, 99.5) for s in specs.values())
            vmin = vmax - 70
            fig, axes = plt.subplots(
                1, 4, figsize=(16, 3.6), gridspec_kw=dict(width_ratios=[1.6, 3, 3, 3])
            )
            t = np.arange(rps.shape[1]) / SR
            for r in range(rps.shape[0]):
                axes[0].plot(t, rps[r], lw=1)
            axes[0].set_title("rotor speeds (label)", fontsize=10)
            axes[0].set_xlabel("s")
            axes[0].set_ylabel("rev/s")
            for ax, (name, title) in zip(
                axes[1:],
                [
                    ("real", "real recording"),
                    ("rig", "rig preset (same RPS)"),
                    ("old", "old family (same RPS)"),
                ],
            ):
                s = specs[name]
                ax.imshow(
                    s,
                    origin="lower",
                    aspect="auto",
                    cmap="magma",
                    vmin=vmin,
                    vmax=vmax,
                    extent=[0, t[-1], 0, SR / 2 / 1000],
                )
                ax.set_title(title, fontsize=10)
                ax.set_xlabel("s")
            axes[1].set_ylabel("kHz")
            for ax in axes[2:]:
                ax.set_yticklabels([])
            fig.suptitle(f"{rig}: {cid} — {label}", fontsize=11)
            fig.tight_layout()
            fig.savefig(HERE / f"{slug}.png", dpi=110)
            plt.close(fig)
            index.append(
                dict(
                    rig=rig,
                    clip=cid,
                    label=label,
                    slug=slug,
                    dur=round(clip.audio.shape[1] / SR, 1),
                )
            )
            print("built", slug, flush=True)
    (HERE / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
