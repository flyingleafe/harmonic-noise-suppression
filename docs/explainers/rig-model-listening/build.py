"""Build the assets of ``docs/explainers/rig-model-listening.qmd``: for every
chosen real clip, the real audio (one microphone) and three synthetic renders
driven by the SAME rotor-speed trajectories —

* ``fit``  the FITTED presets of 2026-09-10 (`pop-michaels-final-w.json`,
  `pop-dregon-final-w.json`), pushed through
  ``raw_predictive.population_ranges`` — the very function the
  posterior-predictive gate renders with, so what you hear is what was gated;
* ``rig``  the earlier hand-ranged rig presets (``rig_fm_5050.yaml``), the
  starting point of the fitting campaign;
* ``old``  the pre-rig family (``StochasticRanges()`` defaults, filtered-noise
  lines) the first synthetic-only regressors were trained on.

Audio is level-matched; spectrograms share one colour scale per row.

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
from experiments.stochastic_fit.raw_predictive import population_ranges

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
POLICY = ROOT / "conf/online_mix/rig_fm_5050.yaml"
#: The fitted summaries the gate accepted, per rig. Derived artefacts, so they
#: live outside git; regenerate with the pipeline in
#: docs/experiments/stochastic-fit.md if they are missing.
#: THE fitted summaries — the rigs as refitted under the uncensored,
#: chirp-aware forward model, exported through the clip-centred population.
#: The previous pair (pop-*-final-w.json) is superseded: its width parameter was
#: the estimator's own floor and its exported profile centre was the one
#: quantity in the hierarchy the likelihood does not pin.
FITTED = {
    "dregon": ROOT / "omnirun-outputs/refit-dregon-aug.json",
    "michaels": ROOT / "omnirun-outputs/refit-michaels-aug.json",
}
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


def prominence_db(x: np.ndarray, lo: float, hi: float, n_fft: int = 2048, hop: int = 512) -> float:
    """Median over frames of (95th percentile - median) power in a band, in dB.

    High when a band holds sparse strong lines, low when it is a continuum, and
    insensitive to the band's absolute level — which is what distinguishes a
    comb that survived from one that merged.
    """
    spec = spectrogram_db(x, n_fft, hop)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / SR)
    band = spec[(freqs >= lo) & (freqs < hi)]
    return float(np.median(np.percentile(band, 95, axis=0) - np.median(band, axis=0)))


#: The bands the diagnosis quotes: the mid comb, the region where DREGON's comb
#: merges, and the top of the band where its real lines are strongest.
PROMINENCE_BANDS = ((1000.0, 3000.0), (4000.0, 6000.0), (6000.0, 8000.0))


def fitted_arm(
    rig: str, base_ranges: dict, source: dict[str, Path] | None = None
) -> tuple[srn.StochasticRanges, dict]:
    """The gated preset: ranges through the gate's own transfer, plus its rig scalars.

    ``population_ranges`` carries the profile, floor curve, widths, microphone
    structure, amplitude process, label error and width population; the three
    speed-law scalars live on the fitted rig and are applied to the drawn
    parameters exactly as ``render_matched_population`` does.
    """
    path = (source or FITTED)[rig]
    if not path.exists():
        raise FileNotFoundError(f"{path}: fitted summary missing — see the pipeline in the log")
    summary = json.loads(path.read_text())
    ranges = srn.StochasticRanges.from_dict(population_ranges(summary, base_ranges))
    scalars = dict(
        amp_rps_exponent=float(summary["rig"]["amp_exp"]),
        amp_rps_exponent_floor=float(summary["rig"]["floor_exp"]),
        floor_static_rel=float(summary["rig"]["floor_static_rel"]),
    )
    return ranges, scalars


def main() -> None:
    pol = yaml.safe_load(POLICY.read_text())
    presets = {"dregon": pol["sources"]["noise"][0], "michaels": pol["sources"]["noise"][1]}
    old_ranges = srn.StochasticRanges()
    rng = np.random.default_rng(20260909)
    index: list[dict] = []
    for rig, items in CLIPS.items():
        src = presets[rig]
        ranges = srn.StochasticRanges.from_dict(src["ranges"])
        fit_ranges, fit_scalars = fitted_arm(rig, src.get("ranges", {}))
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
            fitted_profile = fit_ranges.profile_mean_db or ()
            n_fit = min(n_harm, len(fitted_profile)) if fitted_profile else n_harm
            fit_params = srn.sample_params(
                rng, fit_ranges, n_rotors=4, n_harmonics=n_fit, sample_rate=SR
            ).with_(**fit_scalars)
            fit, _ = srn.synthesize(
                fit_params,
                rps,
                rng=rng,
                n_mics=8,
                mic_gain_db=tuple(src["mic_gain_db"]),
                line_mode="fm",
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
            trio = {
                "real": level(clip.audio[MIC]),
                "fit": level(fit[MIC]),
                "rig": level(new[MIC]),
                "old": level(old[MIC]),
            }
            slug = f"{rig}_{cid}"
            for name, x in trio.items():
                sf.write(HERE / f"{slug}_{name}.wav", x, SR)
            specs = {k: spectrogram_db(v) for k, v in trio.items()}
            vmax = max(np.percentile(s, 99.5) for s in specs.values())
            vmin = vmax - 70
            columns = [
                ("real", "real recording"),
                ("fit", "fitted preset (same RPS)"),
                ("rig", "pre-fit rig preset"),
                ("old", "old family"),
            ]
            fig, axes = plt.subplots(
                1,
                1 + len(columns),
                figsize=(4 + 3.2 * len(columns), 3.4),
                gridspec_kw=dict(width_ratios=[1.5] + [3] * len(columns)),
            )
            t = np.arange(rps.shape[1]) / SR
            for r in range(rps.shape[0]):
                axes[0].plot(t, rps[r], lw=1)
            axes[0].set_title("rotor speeds (label)", fontsize=10)
            axes[0].set_xlabel("s")
            axes[0].set_ylabel("rev/s")
            for ax, (name, title) in zip(axes[1:], columns, strict=True):
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
            prominence = {
                name: [prominence_db(x, lo, hi) for lo, hi in PROMINENCE_BANDS]
                for name, x in trio.items()
            }
            index.append(
                dict(
                    prominence=prominence,
                    prominence_bands=[list(b) for b in PROMINENCE_BANDS],
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
