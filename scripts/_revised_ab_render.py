"""Family A/B renders: REAL vs PREVIOUS (coherent+Lorentzian) vs REVISED (C3)
on the SAME real window, for `docs/explainers/rotor-noise-fit.qmd` §19.5.

Conventions are those of ``scripts/_ab_render.py`` (§16.5 of that page): the
same real clip, the same raw rotor trajectory, the same seed per window, ONE
common playback gain per window, and shape-only LTAS/channel statistics. The
differences from that script:

* the arms are model FAMILIES, not two label variants of one family;
* the revised arm goes through
  ``experiments.stochastic_fit.revised_phase.render_revised`` — the primary
  held-out predictive renderer, driven by the RAW telemetry plus the learned
  bias law, physical levels, render-side antialias — and is then scaled to the
  SAME whole-array RMS (0.1) that ``stochastic_rotor_noise.synthesize`` gives
  the previous-model arms by default, so all synthetic arms share one loudness
  convention and the ear compares spectrum and texture, not fitted gain;
* every synthetic render keeps its PRE-normalisation RMS, recorded in the
  index as ``rms_dbfs_pre``, so the level statements stay diagnostic and
  explicit instead of hidden.

Windows are fixed, not searched: the two FLY125 cruise windows are exactly
§16.5's A/B windows; the FLY125 standby window is the C3 export's own standby
training support; the two DREGON windows are the first two supports of the
five-recording cohort both models were fitted on.

    PYTHONPATH="$PWD/src:$PWD/scripts" python scripts/_revised_ab_render.py

Writes ``docs/explainers/revised-compare/revcmp_<slug>_<ww>_<arm>.wav``, one
figure per window, and ``revcmp_<slug>.json`` with the metrics and provenance.
"""

from __future__ import annotations

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
from experiments.stochastic_fit import revised_eval as RE
from experiments.stochastic_fit import revised_phase as RP
from experiments.stochastic_fit import stage2 as S2

OUT = Path("docs/explainers/revised-compare")
RMS_TARGET = 0.1  # srn.synthesize's default normalize_rms; the arms' shared convention
COLOURS = {"real": "#111111", "prev": "#d62728", "revised": "#2ca02c"}

#: The C3 fits (best existing revised fit; C4 is implemented but untrained).
C3_EXPORT = {
    "michaels": Path(
        ".worktrees/revised-phase-preflight/omnirun-outputs/"
        "revised-phase-c3-fits-ad2c57/results/revised_phase/c3/michaels.json"
    ),
    "dregon": Path(
        ".worktrees/revised-phase-preflight/omnirun-outputs/"
        "revised-phase-c3-fits-ad2c57/results/revised_phase/c3/dregon.json"
    ),
}
RAW_KEY = {"michaels": "rps", "dregon": "motors_command"}
DATASET = {"michaels": "michaels-frames", "dregon": "DREGON-frames"}

#: Previous-model exports: the raw-label 2026-09 FLY125 cruise fit and its
#: standby fit; the pooled refined-label DREGON rig fit (no raw pooled DREGON
#: fit exists — §16.5's warning applies to this pair too).
PREV_FIT = {
    "fly125_cruise": Path("results/S2/cruise_8clip.json"),
    "fly125_standby": Path("results/S2/standby.json"),
    "dregon": Path("results/S2/dregon_room2_cruise_refined.json"),
}


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


def rms(x: np.ndarray) -> float:
    """Whole-array RMS, exactly ``stochastic_rotor_noise.synthesize``'s definition."""
    return float(np.sqrt(np.mean(np.square(x))))


def dbfs(x: np.ndarray) -> float:
    return float(20.0 * np.log10(max(rms(x), 1e-12)))


def load_real(rig: str, recording: str, start_s: float, seconds: float, clip_id: str):
    clip = C.decimate(
        C.load_clip(
            DATASET[rig],
            recording,
            start_s,
            seconds,
            rps_key=RAW_KEY[rig],
            clip_id=clip_id,
        ),
        S2.SR,
    )
    return clip


def prev_export(fit_path: Path, selector: Any) -> dict[str, Any]:
    """One clip's fitted parameter vector of a previous-model fit."""
    summary = json.loads(fit_path.read_text())
    clips = list(summary["clips"].values())
    if isinstance(selector, int):
        return clips[selector]["params"]
    for cid, entry in summary["clips"].items():
        if cid.startswith(str(selector)):
            return entry["params"]
    raise SystemExit(f"{fit_path}: no clip starts with {selector!r}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # validation gate: read_candidate_export refuses legacy, C1 and malformed
    # exports, so a wrong file fails loudly before any render
    for p in C3_EXPORT.values():
        RE.read_candidate_export(p)
    c3_summary = {rig: json.loads(p.read_text()) for rig, p in C3_EXPORT.items()}

    # (slug, window key, rig, recording, start_s, seconds, regime, prev selector)
    windows = [
        ("fly125", "00", "michaels", "FLY125", 16.0, 8.0, "cruise", ("fly125_cruise", 0)),
        ("fly125", "01", "michaels", "FLY125", 32.0, 8.0, "cruise", ("fly125_cruise", 1)),
        ("fly125", "sb", "michaels", "FLY125", 2.0, 8.0, "standby", ("fly125_standby", 0)),
        (
            "dregon",
            "00",
            "dregon",
            "free-flight_nosource_room2",
            1512727417.2050455,
            8.0,
            "cruise",
            ("dregon", "free-flight_nosource_room2"),
        ),
        (
            "dregon",
            "01",
            "dregon",
            "hovering_nosource_room2",
            1511903913.3944898,
            8.0,
            "cruise",
            ("dregon", "hovering_nosource_room2"),
        ),
    ]

    indices: dict[str, list[dict[str, Any]]] = {}
    for slug, wk, rig, rec, start_s, seconds, regime, (fit_id, sel) in windows:
        real = load_real(rig, rec, start_s, seconds, f"revcmp_{rec}_{start_s:.3f}_{seconds:g}")
        rps = np.asarray(real.rps, dtype=np.float64)
        xr = np.asarray(real.audio, dtype=np.float64)
        mic = 0
        seed = 4242 + (int(wk, 16) if wk.isdigit() else 40)
        waves: dict[str, np.ndarray] = {"real": xr}

        pv = prev_export(PREV_FIT[fit_id], sel)
        waves["prev"] = S2.render_from_export(pv, rps, n_mics=xr.shape[0], seed=seed)

        rr = RP.render_revised(c3_summary[rig], rps, n_mics=xr.shape[0], seed=seed)
        rev = np.asarray(rr.audio, dtype=np.float64)
        match_gain = RMS_TARGET / max(rms(rev), 1e-12)
        waves["revised"] = rev * match_gain

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
            spec_panel(ax, wav[mic], f"{rec} {regime} — {name}")
            b = stats.ltas_bands(wav[mic])
            bands[name] = np.round(b, 2).tolist()
            axes[-1].plot(
                centres,
                b,
                marker="o",
                ms=4,
                lw=1.4,
                color=COLOURS[name],
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
        fig.savefig(OUT / f"revcmp_{slug}_{wk}.png", dpi=100)
        plt.close(fig)

        peak = max(float(np.abs(wav[mic]).max()) for wav in waves.values())
        gain = 0.7 / max(peak, 1e-9)
        for name, wav in waves.items():
            y = wav[mic] * gain
            sf.write(OUT / f"revcmp_{slug}_{wk}_{name}.wav", y.astype(np.float32), S2.SR)

        pattern: dict[str, list[float]] = {}
        for name, wav in waves.items():
            lv = 10 * np.log10((wav**2).mean(axis=1) + 1e-20)
            pattern[name] = np.round(lv - lv.mean(), 2).tolist()

        row = dict(
            slug=slug,
            window=wk,
            regime=regime,
            clip=f"{rec}@{start_s:.3f}+{seconds:g}",
            recording=rec,
            start_s=start_s,
            seconds=seconds,
            rotors=np.round(rps.mean(axis=1), 1).tolist(),
            arms=list(waves),
            ltas_mean_abs_db={k: round(v, 3) for k, v in devs.items()},
            channel_pattern_db=pattern,
            channel_rms_dev_db={
                name: round(float(np.abs(np.asarray(v) - np.asarray(pattern["real"])).mean()), 2)
                for name, v in pattern.items()
                if name != "real"
            },
            playback_gain=round(float(gain), 4),
            peak_dbfs={
                name: round(float(20 * np.log10(max(np.abs(wav[mic]).max(), 1e-12) * gain)), 2)
                for name, wav in waves.items()
            },
            rms_dbfs_pre={
                "real": round(dbfs(waves["real"]), 2),
                "prev": round(dbfs(waves["prev"]), 2),
                "revised": round(dbfs(rev), 2),
            },
            revised_rms_match_gain_db=round(float(20 * np.log10(match_gain)), 2),
            seed=seed,
            rps_key=RAW_KEY[rig],
            dataset=DATASET[rig],
            prev_fit=str(PREV_FIT[fit_id]),
            prev_arm_selector=str(sel),
            revised_export=str(C3_EXPORT[rig]),
            revised_export_sha256=RE.artifact_digest(C3_EXPORT[rig])["sha256"],
            revised_render_mean_abs_physical_minus_reference_rps=rr.diagnostics[
                "mean_abs_physical_minus_reference_rps"
            ],
            figure=f"revcmp_{slug}_{wk}.png",
        )
        indices.setdefault(slug, []).append(row)
        print(
            f"{row['clip']}: "
            + "  ".join(f"{k} LTAS {v:.2f} dB" for k, v in devs.items())
            + "  channels "
            + str(row["channel_rms_dev_db"])
            + f"  revised phys-vs-ref {rr.diagnostics['mean_abs_physical_minus_reference_rps']:.3f} rps",
            flush=True,
        )

    for slug, rows in indices.items():
        (OUT / f"revcmp_{slug}.json").write_text(json.dumps(rows, indent=1))
        print(f"wrote {OUT / f'revcmp_{slug}.json'}")


if __name__ == "__main__":
    main()
