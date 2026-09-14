"""Demo for :mod:`experiments.stochastic_fit.rig_sampler` — phase C.

Two rigs, two sampling modes, one fixed seed:

* CLOSE NEIGHBOURHOOD of one fit (``sample_rig``): 8 draws at strength 1.0 and
  4 at strength 2.0, per rig, rendered on that rig's own real telemetry.
* PATH CLOUD between the two fits (``sample_path``): 12 draws with ``t``
  uniform at spread 1.0, plus the ``spread = 0`` endpoint check that ``t = 0``
  reproduces anchor A and ``t = 1`` reproduces anchor B.
* One deliberately OVER-WIDE request (strength 6) per rig, to show which guards
  fire and that the sampler raises instead of emitting an invalid export.

Conventions, following ``scripts/_revised_ab_render.py``:

* the anchors and their windows are FIXED, not searched — the FLY125 cruise
  window is §16.5's A window and the DREGON window is the first support of the
  five-recording room-2 cohort;
* every render uses the raw rotor trajectory with that dataset's own rps key,
  ``normalize_rms=None`` so absolute level stays diagnostic, and then ONE
  common playback gain per rig so the ear compares spectrum and texture;
* each render's pre-gain dBFS is recorded in the index.

    PYTHONPATH="$PWD/src:$PWD/scripts" python scripts/_rig_sampler_demo.py

Writes into ``docs/explainers/rig-sampler/``: profile, LTAS, spectrogram and
path-cloud figures, WAVs for the anchor plus four samples per rig, four path
draws, and ``index.json`` with seeds, strengths, guard statistics, realised
spreads and provenance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from experiments.stochastic_fit import clips as C
from experiments.stochastic_fit import rig_sampler as RS
from experiments.stochastic_fit import stage2 as S2

OUT = Path("docs/explainers/rig-sampler")
RESULTS = Path("results/rig_sampler")
SECONDS = 4.0
SEED = 20260914
RMS_TARGET = 0.1  # srn.synthesize's default normalize_rms: the listening convention
N_S1, N_S2, N_PATH = 8, 4, 12
N_WAV = 4
#: the easy transfer arm has to COVER the real rigs, so its strength is chosen
#: by measured coverage inside the user's [2, 3] range, not assumed
SWEEP_STRENGTHS = (2.0, 2.5, 3.0)
#: probed only to REPORT what would be needed if nothing in range works
EXTENDED_STRENGTHS = (4.0, 5.0)
N_SWEEP = 32
BRACKET_TARGET = 0.90
#: largest amp_exp any of the six measured fits produced (dregon_flight);
#: reported so a drawn speed law can be seen against what was ever fitted
AMP_EXP_MEASURED_MAX = 14.111

RIGS: dict[str, dict[str, Any]] = {
    "michael_cruise": {
        "fit": Path("results/S2/cruise_8clip.json"),
        "clip_id": "fly125_cruise_00",
        "dataset": "michaels-frames",
        "recording": "FLY125",
        "start_s": 16.0,
        "rps_key": "rps",
        "colour": "#1f77b4",
    },
    "dregon_cruise": {
        "fit": Path("results/S2/dregon_room2_cruise_refined.json"),
        "clip_id": "free-flight_nosource_room2_cruise_00",
        "dataset": "DREGON-frames",
        "recording": "free-flight_nosource_room2",
        "start_s": 1512727417.2050455,
        "rps_key": "motors_command",
        "colour": "#d62728",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# anchors are loaded by rig_sampler.load_anchor, which folds each clip's own
# scores.power_scale into profile_db and floor_mean_db. The raw params dict
# that scripts/_revised_ab_render.prev_export returns is in the fit's SCALED
# unit and is not comparable in absolute level across rigs; the raw levels are
# still loaded here, with physical=False, only to report what folding changed.


def load_clip(spec: dict[str, Any]):
    return C.decimate(
        C.load_clip(
            spec["dataset"],
            spec["recording"],
            spec["start_s"],
            SECONDS,
            rps_key=spec["rps_key"],
            clip_id=spec["clip_id"],
        ),
        S2.SR,
    )


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def dbfs(x: np.ndarray) -> float:
    return float(20.0 * np.log10(max(rms(x), 1e-12)))


#: the module owns the band analysis so the guard envelope and these figures
#: are measured the same way
ltas_bands = RS.audio_bands


def spec_panel(ax, x: np.ndarray, title: str, vmax: float | None = None) -> float:
    """One spectrogram panel. ``vmax`` is shared across a grid on purpose: with
    per-panel autoscaling the level and line/floor balance differences between
    samples — the whole point of the grid — are normalised away."""
    n, hop = 2048, 512
    w = np.hanning(n + 1)[:n]
    fr = np.stack([x[s : s + n] * w for s in range(0, x.size - n, hop)])
    S = 20 * np.log10(np.abs(np.fft.rfft(fr, axis=-1)).T + 1e-9)
    v = float(np.percentile(S, 99.5)) if vmax is None else float(vmax)
    ax.imshow(
        S,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, x.size / S2.SR, 0, S2.SR / 2000],
        vmin=v - 75,
        vmax=v,
    )
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("s", fontsize=8)
    ax.tick_params(labelsize=7)
    return v


def plot_profile(ax, p: np.ndarray, *, colour: str, lw: float, alpha=1.0, label=None, z=2) -> None:
    """One rotor profile as TWO branches: even orders solid, odd orders dotted.

    Drawn as a single zig-zag the parity split is a picket fence and nothing
    about it is legible. Split into its two subsequences, the blade-passing
    (even) branch and the odd branch are two clean curves: the vertical gap
    between them IS the parity amplitude and the slope of each IS the power-law
    falloff, which is exactly what this figure has to show.
    """
    k = np.arange(1, p.size + 1)
    ev = k % 2 == 0
    ax.plot(k[ev], p[ev], color=colour, lw=lw, alpha=alpha, label=label, zorder=z)
    ax.plot(k[~ev], p[~ev], color=colour, lw=lw, alpha=alpha, ls=":", zorder=z)


def profile_figure(
    path: Path, title: str, layers: list[tuple[str, list[np.ndarray], dict]]
) -> None:
    """Per-rotor profile overlay, dB against log order, even/odd split."""
    n_rotors = max(len(v) for _, group, _ in layers for v in [group[0]])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for r, ax in enumerate(axes.ravel()[:n_rotors]):
        for label, group, style in layers:
            for i, prof in enumerate(group):
                plot_profile(ax, prof[r], label=label if i == 0 else None, **style)
        ax.set_xscale("log")
        ax.set_title(f"rotor {r}", fontsize=9)
        ax.grid(alpha=0.25, which="both")
        if r >= 2:
            ax.set_xlabel("harmonic order k")
        if r % 2 == 0:
            ax.set_ylabel("profile [dB]")
    axes.ravel()[0].legend(
        fontsize=7, title="solid = even (blade passing), dotted = odd", title_fontsize=6
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # path WAVs are named after their realised t, so a previous run's names do
    # not overlap this one's; clear them or the directory accumulates orphans
    for stale in OUT.glob("path_t*.wav"):
        stale.unlink()
    RESULTS.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {
        "seed": SEED,
        "seconds": SECONDS,
        "rms_target": RMS_TARGET,
        "widths_strength1": RS.WIDTHS.as_dict(),
        "structure_json": str(RS.STRUCTURE_PATH),
        "path_interpolation": RS.PATH_INTERP,
        "provenance": {},
        "rigs": {},
        "path_cloud": {},
    }
    anchors: dict[str, dict[str, Any]] = {}
    tele: dict[str, Any] = {}
    levels: dict[str, dict[str, float]] = {}

    print("=" * 96)
    print("PHASE C - rig sampler demo")
    print("=" * 96)

    for name, spec in RIGS.items():
        anchors[name] = RS.load_anchor(spec["fit"], spec["clip_id"])
        tele[name] = load_clip(spec)
        index["provenance"][name] = {
            "fit": str(spec["fit"]),
            "sha256": sha256(spec["fit"]),
            "clip_id": spec["clip_id"],
            "dataset": spec["dataset"],
            "recording": spec["recording"],
            "start_s": spec["start_s"],
            "rps_key": spec["rps_key"],
            "rates_rev_s": [float(v) for v in tele[name].rps.mean(axis=1)],
            "n_orders": int(np.shape(anchors[name]["profile_db"])[1]),
            "power_scale_folded_db": float(anchors[name]["power_scale_folded_db"]),
        }

    # ---- anchor levels on the PHYSICAL, BAND-RESTRICTED coordinate --------
    lo, hi = RS.LEVEL_BAND_HZ
    print(
        f"\n-- anchor levels: power_scale folded AND the declared rate term applied, "
        f"level measured in {lo:g}-{hi:g} Hz --"
    )
    for name, spec in RIGS.items():
        raw = RS.load_anchor(spec["fit"], spec["clip_id"], physical=False)
        band = RS.render_level_db(anchors[name], tele[name].rps, seed=SEED)
        raw_band = RS.render_level_db(raw, tele[name].rps, seed=SEED)
        real_band_level = RS.band_level_db(tele[name].audio)
        levels[name] = {
            "band_level_db": band,
            "band_level_db_unconverted": raw_band,
            "power_scale_folded_db": float(anchors[name]["power_scale_folded_db"]),
            "rate_factor_db": float(anchors[name]["rate_factor_db"]),
            "real_clip_band_level_db": real_band_level,
            "fit_level_error_db": band - real_band_level,
            "rms_dbfs": RS.render_level_db(
                anchors[name], tele[name].rps, seed=SEED, band=(0.0, 1e9)
            ),
            "real_clip_rms_dbfs": dbfs(tele[name].audio),
        }
        v = levels[name]
        print(
            f"  {name:16s} power_scale {v['power_scale_folded_db']:+8.3f} + rate "
            f"{v['rate_factor_db']:+6.3f} dB | band level {band:+8.3f} dB "
            f"(raw export {raw_band:+8.3f}) | REAL clip {real_band_level:+8.3f} dB | "
            f"fit level error {v['fit_level_error_db']:+7.3f} dB"
        )
    names = list(RIGS)
    a, b = levels[names[0]], levels[names[1]]
    off = b["band_level_db"] - a["band_level_db"]
    off_real = b["real_clip_band_level_db"] - a["real_clip_band_level_db"]
    off_rms = b["rms_dbfs"] - a["rms_dbfs"]
    print(
        f"  OFFSET {names[1]} - {names[0]} on the band coordinate: {off:+.3f} dB "
        f"(whole-array RMS on the same exports says {off_rms:+.3f} dB — that coordinate is "
        f"dominated by the k=1-3 excess and is NOT used)"
    )
    print(
        f"  decomposition of the {off:+.3f} dB: real recordings differ by {off_real:+.3f} dB, "
        f"{names[1]}'s fit is {b['fit_level_error_db']:+.3f} dB off its own clip and "
        f"{names[0]}'s is {a['fit_level_error_db']:+.3f} dB "
        f"= {off_real + b['fit_level_error_db'] - a['fit_level_error_db']:+.3f} dB"
    )
    print(
        "  the two REAL recordings are within "
        f"{abs(off_real):.3f} dB of each other in this band: whatever level difference the "
        "exports carry is fit error, not rigs."
    )
    index["anchor_levels"] = {
        "per_rig": levels,
        "level_band_hz": [lo, hi],
        "offset_band_db": off,
        "offset_real_clips_band_db": off_real,
        "offset_whole_array_rms_db": off_rms,
        "note": (
            "render_level_db = the anchor's own render level in LEVEL_BAND_HZ at "
            "normalize_rms=None on its own telemetry, with BOTH halves of the declared "
            "conversion applied (scores.power_scale folded per clip, plus "
            "to_renderer_units' 10 log10(work/analysis) = +4.4032 dB at the 44.1 kHz work "
            "grid). Band-restricted because 50-200 Hz carries a known k=1-3 render excess "
            "that dominates a whole-array RMS; see rig_sampler.LEVEL_BAND_HZ."
        ),
    }

    # ---- the LTAS envelope: real-drone-to-real-drone ----------------------
    print(
        f"\n-- LTAS plausibility envelope: {RS.LTAS_ENVELOPE_X:g} x the measured difference "
        f"between the two REAL drones --"
    )
    real_dev: dict[str, dict[str, float]] = {}
    for name in RIGS:
        d = RS.ltas_vs_real(anchors[name], tele[name].rps, tele[name].audio, seed=SEED)
        real_dev[name] = d
        print(
            f"  {name:16s} anchor-to-real: {lo:g}-{hi:g} Hz rms {d['rms_db']:5.2f} dB, worst "
            f"band {d['band_max_db']:5.2f} dB, level {d['level_db']:+5.2f} dB | full band rms "
            f"{d['rms_db_full_band']:5.2f}, worst {d['band_max_db_full_band']:5.2f} dB"
        )
    cen_pair, bm = ltas_bands(tele[names[0]].audio)
    _, bd = ltas_bands(tele[names[1]].audio)
    m_pair = (cen_pair >= lo) & (cen_pair <= hi)
    pair = (bd - bm)[m_pair]
    pair_full = bd - bm
    print(
        f"  REAL {names[0]} vs REAL {names[1]}: rms {np.sqrt((pair**2).mean()):5.2f} dB, "
        f"worst band {np.abs(pair).max():5.2f} dB (full band rms "
        f"{np.sqrt((pair_full**2).mean()):5.2f}, worst {np.abs(pair_full).max():5.2f})"
    )
    print(
        f"  -> guard: LTAS rms deviation from the reference <= {RS.LTAS_ENVELOPE_X:g} x "
        f"{RS.REAL_PAIR_RMS_DB:g} = {RS.LTAS_ENVELOPE_X * RS.REAL_PAIR_RMS_DB:.2f} dB, in "
        f"{lo:g}-{hi:g} Hz, worst band reported but NOT guarded"
    )
    index["ltas_envelope"] = {
        "anchor_to_real": real_dev,
        "real_pair": {
            "rms_db": float(np.sqrt((pair**2).mean())),
            "band_max_db": float(np.abs(pair).max()),
            "rms_db_full_band": float(np.sqrt((pair_full**2).mean())),
            "band_max_db_full_band": float(np.abs(pair_full).max()),
        },
        "multiple": RS.LTAS_ENVELOPE_X,
        "tolerance_rms_db": RS.LTAS_ENVELOPE_X * RS.REAL_PAIR_RMS_DB,
        "note": (
            "the envelope is twice the measured REAL-to-REAL difference between the two "
            "drones, in LEVEL_BAND_HZ. It is not tied to the anchor-to-real deviation, "
            "which on the corrected coordinate is only 0.6-2.1 dB rms (the fits already "
            "agree with real above 300 Hz) and would be tighter than the between-clip "
            "spread. Structural guards (amp_exp/floor_exp >= 0, parity sign, falling trend, "
            "gamma excursion, finite render) are not re-scoped."
        ),
    }

    # ======================================================================
    # mode 1: close neighbourhood of one fit
    # ======================================================================
    for name, _spec in RIGS.items():
        anchor, clip = anchors[name], tele[name]
        print(f"\n{'=' * 96}\n-- {name}: close neighbourhood --")
        batches: dict[float, list[dict]] = {}
        for strength, n in ((1.0, N_S1), (2.0, N_S2)):
            batches[strength] = RS.sample_batch(anchor, n, SEED, strength=strength)

        # guard / rejection statistics
        stats: dict[str, Any] = {}
        for strength, samples in batches.items():
            attempts = sum(s["_sampler"]["attempts"] for s in samples)
            fired: dict[str, int] = {}
            for s in samples:
                for failed in s["_sampler"]["rejected"]:
                    for g in failed:
                        fired[g] = fired.get(g, 0) + 1
            rejected = attempts - len(samples)
            stats[f"strength_{strength:g}"] = {
                "n_samples": len(samples),
                "attempts": attempts,
                "rejected": rejected,
                "rejection_rate": rejected / attempts,
                "guard_fired": fired,
            }
            print(
                f"  strength {strength:g}: {len(samples)} samples in {attempts} attempts, "
                f"rejection rate {rejected / attempts:.3f}, guards fired {fired or '{}'}"
            )

        # realised spreads, against the phase A targets
        print("  realised spread across draws (target = width x strength):")
        realised: dict[str, Any] = {}
        for strength, samples in batches.items():
            gains, slopes, gammas = [], [], []
            for s in samples:
                prof = RS.effective_profile(s)
                prof_a = RS.effective_profile(anchor)
                k = min(prof.shape[1], prof_a.shape[1])
                for r in range(prof.shape[0]):
                    pa = RS.decompose_profile(prof_a[r, :k])
                    ps = RS.decompose_profile(prof[r, :k])
                    gains.append(ps.gain - pa.gain)
                    slopes.append(ps.slope - pa.slope)
                gammas.append(
                    float(np.mean(s["gamma_slope"])) / float(np.mean(anchor["gamma_slope"]))
                )
            realised[f"strength_{strength:g}"] = {
                "rotor_gain_db_std": float(np.std(gains, ddof=1)),
                "rotor_gain_db_target": strength * RS.WIDTHS.rotor_gain_db,
                "slope_db_dec_std": float(np.std(slopes, ddof=1)),
                "slope_db_dec_target": strength * RS.WIDTHS.slope_db_dec,
                "gamma_ratio_ln_std": float(np.std(np.log(gammas), ddof=1)),
                "gamma_ratio_ln_target": strength * RS.WIDTHS.gamma_ln,
                "gamma_ratio_range": [float(min(gammas)), float(max(gammas))],
            }
            v = realised[f"strength_{strength:g}"]
            print(
                f"    s={strength:g}  rotor gain {v['rotor_gain_db_std']:5.2f} dB "
                f"(target {v['rotor_gain_db_target']:5.2f})   "
                f"slope {v['slope_db_dec_std']:5.2f} dB/dec "
                f"(target {v['slope_db_dec_target']:5.2f})   "
                f"gamma ln {v['gamma_ratio_ln_std']:6.4f} "
                f"(target {v['gamma_ratio_ln_target']:6.4f}), "
                f"ratio range {v['gamma_ratio_range'][0]:.3f}-{v['gamma_ratio_range'][1]:.3f}"
            )

        # the 8-draw statistic above is what the ticket asks for but it is a
        # noisy estimate of a width; this is the same measurement over enough
        # draws to actually verify the widths against phase A
        check = RS.sample_batch(anchor, 64, SEED + 1, strength=1.0)
        cg, cs, cp = [], [], []
        for s in check:
            prof, prof_a = RS.effective_profile(s), RS.effective_profile(anchor)
            k = min(prof.shape[1], prof_a.shape[1])
            for r in range(prof.shape[0]):
                pa = RS.decompose_profile(prof_a[r, :k])
                ps = RS.decompose_profile(prof[r, :k])
                cg.append(ps.gain - pa.gain)
                cs.append(ps.slope - pa.slope)
                cp.append(np.log(ps.parity_low_db / pa.parity_low_db))
        width_check = {
            "n_draws": len(check),
            "rotor_gain_db": [float(np.std(cg, ddof=1)), RS.WIDTHS.rotor_gain_db],
            "slope_db_dec": [float(np.std(cs, ddof=1)), RS.WIDTHS.slope_db_dec],
            "parity_ln": [float(np.std(cp, ddof=1)), RS.WIDTHS.parity_ln],
        }
        print(
            f"  width check over {len(check)} strength-1 draws (realised vs frozen): "
            f"rotor gain {width_check['rotor_gain_db'][0]:.2f}/"
            f"{width_check['rotor_gain_db'][1]:.2f} dB, slope "
            f"{width_check['slope_db_dec'][0]:.2f}/{width_check['slope_db_dec'][1]:.2f} dB/dec, "
            f"parity ln {width_check['parity_ln'][0]:.3f}/{width_check['parity_ln'][1]:.3f} "
            f"(parity is re-MEASURED after the residual redraw, which moves it too)"
        )

        # renders: anchor first, then every sample, one common gain
        audio = {"anchor": S2.render_from_export(anchor, clip.rps, seed=SEED, normalize_rms=None)}
        pre_dbfs = {"anchor": dbfs(audio["anchor"])}
        for strength, samples in batches.items():
            for i, s in enumerate(samples):
                tag = f"s{strength:g}_{i:02d}"
                audio[tag] = S2.render_from_export(s, clip.rps, seed=SEED, normalize_rms=None)
                pre_dbfs[tag] = dbfs(audio[tag])
        assert all(np.isfinite(a).all() for a in audio.values()), "a render was not finite"
        gain = RMS_TARGET / max(rms(audio["anchor"]), 1e-12)

        # LTAS deviation from the anchor, per sample
        cen, band_anchor = ltas_bands(audio["anchor"])
        _, band_real = ltas_bands(clip.audio)
        # power_scale is folded and normalize_rms=None, so the anchor render's
        # ABSOLUTE level is comparable with the real clip's: the real curve is
        # plotted as measured, not level-matched. The shape-only number (the
        # level-matched convention of scripts/_ab_render.py) is reported too, so
        # the tilt mismatch can be separated from the level error.
        band_real_matched = band_real + (dbfs(audio["anchor"]) - dbfs(clip.audio))
        dev: dict[str, Any] = {}
        print(
            "  per-sample LTAS deviation from the anchor [dB] — 'render' is measured on "
            "the 4 s renders, 'model' is what the guard sees:"
        )
        flat = {f"s{st:g}_{i:02d}": s for st, ss in batches.items() for i, s in enumerate(ss)}
        for tag, a in audio.items():
            if tag == "anchor":
                continue
            _, b = ltas_bands(a)
            d = b - band_anchor
            g = flat[tag]["_sampler"]["guards"]
            dev[tag] = {
                "rms_db": float(np.sqrt((d**2).mean())),
                "band_max_db": float(np.abs(d).max()),
                "level_db": float(d.mean()),
                "model_rms_db": g["ltas_rms_db"],
                "model_band_max_db": g["ltas_band_max_db"],
                "pre_gain_dbfs": pre_dbfs[tag],
            }
        for tag in sorted(dev):
            v = dev[tag]
            print(
                f"    {tag:10s} render rms {v['rms_db']:5.2f} worst {v['band_max_db']:5.2f} | "
                f"model rms {v['model_rms_db']:5.2f} worst {v['model_band_max_db']:5.2f} | "
                f"level {v['level_db']:+6.2f}  pre-gain {v['pre_gain_dbfs']:+7.2f} dBFS"
            )
        d_real = band_real - band_anchor
        d_real_shape = band_real_matched - band_anchor
        print(
            f"    REAL clip  ABSOLUTE rms {np.sqrt((d_real**2).mean()):5.2f} "
            f"worst {np.abs(d_real).max():5.2f} level {d_real.mean():+6.2f} | "
            f"level-matched rms {np.sqrt((d_real_shape**2).mean()):5.2f} "
            f"worst {np.abs(d_real_shape).max():5.2f}"
        )
        print(
            f"    guard tolerance (model LTAS, {lo:g}-{hi:g} Hz): "
            f"{RS.LTAS_ENVELOPE_X * RS.REAL_PAIR_RMS_DB:5.2f} dB rms, worst band unguarded"
        )

        # ---- figures -----------------------------------------------------
        profile_figure(
            OUT / f"profiles_{name}.png",
            f"{name}: anchor and sampled profiles, per rotor",
            [
                (
                    "samples, strength 1.0",
                    [RS.effective_profile(s) for s in batches[1.0]],
                    {"colour": "#1f77b4", "lw": 0.7, "alpha": 0.6},
                ),
                (
                    "samples, strength 2.0",
                    [RS.effective_profile(s) for s in batches[2.0]],
                    {"colour": "#ff7f0e", "lw": 0.7, "alpha": 0.7},
                ),
                (
                    "anchor",
                    [RS.effective_profile(anchor)],
                    {"colour": "#111111", "lw": 1.8, "z": 5},
                ),
            ],
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        for strength, colour in ((1.0, "#1f77b4"), (2.0, "#ff7f0e")):
            for i in range(len(batches[strength])):
                _, b = ltas_bands(audio[f"s{strength:g}_{i:02d}"])
                ax.plot(
                    cen,
                    b,
                    color=colour,
                    lw=0.8,
                    alpha=0.7,
                    label=f"samples, strength {strength:g}" if i == 0 else None,
                )
        ax.plot(cen, band_anchor, color="#111111", lw=2.0, label="anchor render")
        ax.plot(cen, band_real, color="#2ca02c", lw=2.2, label="REAL clip, ABSOLUTE")
        ax.axvline(lo, color="#888888", lw=1.0, ls="-.", label=f"{lo:g} Hz: level/guard band edge")
        ax.plot(
            cen,
            band_real_matched,
            color="#2ca02c",
            lw=1.4,
            ls="--",
            label="REAL clip, level-matched (shape only)",
        )
        ax.set_xscale("log")
        ax.set_xlabel("Hz")
        ax.set_ylabel("1/3-octave level [dB], physical scale")
        ax.set_title(
            f"{name}: sampled LTAS envelope against real data (power_scale folded)", fontsize=10
        )
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f"ltas_{name}.png", dpi=130)
        plt.close(fig)

        tags = ["anchor"] + [f"s1_{i:02d}" for i in range(N_WAV)]
        fig, axes = plt.subplots(1, len(tags), figsize=(3.1 * len(tags), 3.4), sharey=True)
        vmax = None
        for ax, tag in zip(axes, tags):
            vmax = spec_panel(ax, audio[tag][0] * gain, tag, vmax)
        axes[0].set_ylabel("kHz")
        fig.suptitle(f"{name}: anchor and four strength-1 samples, mic 0", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / f"spectrograms_{name}.png", dpi=130)
        plt.close(fig)

        wavs = {}
        for tag in tags:
            wav = OUT / f"{name}_{tag}.wav"
            sf.write(wav, (audio[tag][0] * gain).astype(np.float32), S2.SR)
            wavs[tag] = str(wav)

        # ---- the deliberately over-wide request --------------------------
        over: dict[str, Any] = {"strength": 6.0, "max_attempts": 16}
        try:
            bad = RS.sample_rig(anchor, np.random.default_rng([SEED, 999]), strength=6.0)
            over["raised"] = False
            over["attempts"] = bad["_sampler"]["attempts"]
            over["guards_fired"] = [f for a in bad["_sampler"]["rejected"] for f in a]
            print(
                f"  OVER-WIDE strength 6: accepted after {over['attempts']} attempts "
                f"(guards fired on the rejected ones: {over['guards_fired']})"
            )
        except RS.SampleRejected as exc:
            over["raised"] = True
            over["message"] = str(exc)
            print(f"  OVER-WIDE strength 6: SampleRejected raised -> {exc}")
        index["rigs"][name] = {
            "guards": stats,
            "realised_spread": realised,
            "width_check": width_check,
            "ltas_deviation": dev,
            "ltas_real_clip": {
                "absolute_rms_db": float(np.sqrt((d_real**2).mean())),
                "absolute_band_max_db": float(np.abs(d_real).max()),
                "absolute_level_db": float(d_real.mean()),
                "level_matched_rms_db": float(np.sqrt((d_real_shape**2).mean())),
                "level_matched_band_max_db": float(np.abs(d_real_shape).max()),
            },
            "common_gain": float(gain),
            "anchor_pre_gain_dbfs": pre_dbfs["anchor"],
            "wavs": wavs,
            "figures": [
                str(OUT / f"profiles_{name}.png"),
                str(OUT / f"ltas_{name}.png"),
                str(OUT / f"spectrograms_{name}.png"),
            ],
            "over_wide": over,
        }

    # ======================================================================
    # easy-arm strength: choose it by COVERAGE of the real clip
    # ======================================================================
    print(f"\n{'=' * 96}\n-- easy-arm strength choice: does the cloud BRACKET real? --")
    print(
        f"  bracketing fraction = share of 1/3-octave bands with cloud min <= real <= cloud "
        f"max, over {N_SWEEP} draws per setting, absolute levels on the physical path. "
        f"Reported over ALL bands and over {lo:g}-{hi:g} Hz, because below 300 Hz the "
        f"render carries the known k=1-3 excess and covering it is chasing that defect."
    )
    sweep: dict[str, dict[str, Any]] = {}
    for name in RIGS:
        anchor, clip = anchors[name], tele[name]
        cen_r, real_band = ltas_bands(clip.audio)
        in_band = (cen_r >= lo) & (cen_r <= hi)
        sweep[name] = {}
        for s in SWEEP_STRENGTHS:
            draws = RS.sample_batch(anchor, N_SWEEP, SEED + 11, strength=s, max_attempts=32)
            bands = np.stack(
                [
                    ltas_bands(S2.render_from_export(d, clip.rps, seed=SEED, normalize_rms=None))[1]
                    for d in draws
                ]
            )
            c_lo, c_hi = bands.min(axis=0), bands.max(axis=0)
            inside = (real_band >= c_lo) & (real_band <= c_hi)
            attempts = sum(d["_sampler"]["attempts"] for d in draws)
            fired: dict[str, int] = {}
            for d in draws:
                for failed in d["_sampler"]["rejected"]:
                    for g in failed:
                        fired[g] = fired.get(g, 0) + 1
            gains, slopes = [], []
            prof_a = RS.effective_profile(anchor)
            for d in draws:
                prof = RS.effective_profile(d)
                k = min(prof.shape[1], prof_a.shape[1])
                for r in range(prof.shape[0]):
                    pa = RS.decompose_profile(prof_a[r, :k])
                    ps = RS.decompose_profile(prof[r, :k])
                    gains.append(ps.gain - pa.gain)
                    slopes.append(ps.slope - pa.slope)
            amp = [float(d["amp_exp"]) for d in draws]
            sweep[name][f"{s:g}"] = {
                "n_draws": len(draws),
                "bracket_fraction": float(inside[in_band].mean()),
                "bracket_fraction_all_bands": float(inside.mean()),
                "bands_missed_hz": [float(v) for v in cen_r[~inside]],
                "attempts": attempts,
                "acceptance_rate": len(draws) / attempts,
                "guard_fired": fired,
                "rotor_gain_db_std": float(np.std(gains, ddof=1)),
                "slope_db_dec_std": float(np.std(slopes, ddof=1)),
                "amp_exp_mean": float(np.mean(amp)),
                "amp_exp_max": float(np.max(amp)),
                "amp_exp_over_measured_max": float(np.mean(np.asarray(amp) > AMP_EXP_MEASURED_MAX)),
            }
            v = sweep[name][f"{s:g}"]
            print(
                f"  {name:16s} s={s:g}  bracket {100 * v['bracket_fraction']:5.1f}% in band "
                f"({int(inside[in_band].sum())}/{int(in_band.sum())}), "
                f"{100 * v['bracket_fraction_all_bands']:5.1f}% all bands  acceptance "
                f"{v['acceptance_rate']:.3f}  gain sd {v['rotor_gain_db_std']:5.2f} dB  "
                f"slope sd {v['slope_db_dec_std']:4.2f}  amp_exp mean "
                f"{v['amp_exp_mean']:5.2f} max {v['amp_exp_max']:5.2f} "
                f"({100 * v['amp_exp_over_measured_max']:.0f}% over the fitted max "
                f"{AMP_EXP_MEASURED_MAX:g})  fired {v['guard_fired'] or '{}'}"
            )
    chosen = None
    for s in SWEEP_STRENGTHS:
        if min(sweep[n][f"{s:g}"]["bracket_fraction"] for n in RIGS) >= BRACKET_TARGET:
            chosen = s
            break
    extended: dict[str, Any] = {}
    if chosen is None:
        print(
            f"  NO strength in [{min(SWEEP_STRENGTHS):g}, {max(SWEEP_STRENGTHS):g}] brackets "
            f"real in >= {100 * BRACKET_TARGET:g}% of bands on BOTH rigs. Probing beyond the "
            f"range, for REPORTING only — the range is not silently exceeded:"
        )
        for s in EXTENDED_STRENGTHS:
            fr = {}
            for name in RIGS:
                anchor, clip = anchors[name], tele[name]
                _, real_band = ltas_bands(clip.audio)
                draws = RS.sample_batch(anchor, N_SWEEP, SEED + 11, strength=s, max_attempts=32)
                bands = np.stack(
                    [
                        ltas_bands(
                            S2.render_from_export(d, clip.rps, seed=SEED, normalize_rms=None)
                        )[1]
                        for d in draws
                    ]
                )
                cen_x, _ = ltas_bands(clip.audio)
                keep_x = (cen_x >= lo) & (cen_x <= hi)
                inside = (real_band >= bands.min(axis=0)) & (real_band <= bands.max(axis=0))
                fr[name] = {
                    "bracket_fraction": float(inside[keep_x].mean()),
                    "bracket_fraction_all_bands": float(inside.mean()),
                    "acceptance_rate": len(draws) / sum(d["_sampler"]["attempts"] for d in draws),
                }
            extended[f"{s:g}"] = fr
            print(
                f"    s={s:g}  "
                + "  ".join(
                    f"{n}: bracket {100 * fr[n]['bracket_fraction']:5.1f}% acc "
                    f"{fr[n]['acceptance_rate']:.3f}"
                    for n in RIGS
                )
            )
            if min(fr[n]["bracket_fraction"] for n in RIGS) >= BRACKET_TARGET:
                print(
                    f"    -> s={s:g} would reach the target; it is OUTSIDE the requested "
                    f"[2, 3] range and is reported, not adopted"
                )
                break
        easy = max(
            SWEEP_STRENGTHS, key=lambda s: min(sweep[n][f"{s:g}"]["bracket_fraction"] for n in RIGS)
        )
        print(
            f"  best in range: s={easy:g} with "
            + ", ".join(f"{n} {100 * sweep[n][f'{easy:g}']['bracket_fraction']:.1f}%" for n in RIGS)
        )
    else:
        easy = chosen
        print(
            f"  RECOMMENDED easy-arm strength: {easy:g} — smallest value in "
            f"[{min(SWEEP_STRENGTHS):g}, {max(SWEEP_STRENGTHS):g}] bracketing real in "
            f">= {100 * BRACKET_TARGET:g}% of bands on both rigs"
        )
    for name in RIGS:
        v = sweep[name][f"{easy:g}"]
        print(
            f"  at s={easy:g}, {name:16s} acceptance {v['acceptance_rate']:.3f} "
            f"(rejection {1 - v['acceptance_rate']:.3f})"
            + (
                "  <- MATERIAL rejection: the effective distribution is the TRUNCATED one, "
                "not the nominal one"
                if v["acceptance_rate"] < 0.9
                else ""
            )
        )
    index["easy_arm"] = {
        "sweep": sweep,
        "extended_probe": extended,
        "bracket_target": BRACKET_TARGET,
        "chosen_strength": easy,
        "chosen_in_requested_range": chosen is not None,
        "n_draws_per_setting": N_SWEEP,
        "note": (
            "bracketing fraction = share of 1/3-octave bands (50 Hz - 7.9 kHz) where the "
            "cloud's per-band min/max over the draws contains the real clip's band level, "
            "absolute levels, power_scale folded, normalize_rms=None"
        ),
    }

    # ======================================================================
    # mode 2: the path cloud between the two anchors
    # ======================================================================
    a_name, b_name = names
    A, B = anchors[a_name], anchors[b_name]
    lvl = (levels[a_name]["band_level_db"], levels[b_name]["band_level_db"])
    print(f"\n{'=' * 96}\n-- path cloud: {a_name} -> {b_name} --")
    print(f"  levels removed: {lvl[0]:+.3f} / {lvl[1]:+.3f} dBFS, offset {lvl[1] - lvl[0]:+.3f} dB")

    end: dict[str, Any] = {}
    for tag, t, target in (("t0", 0.0, A), ("t1", 1.0, B)):
        s = RS.sample_path(A, B, np.random.default_rng([SEED, 0]), t=t, spread=0.0, levels=lvl)
        d = RS.renderable_deviation(s, target)
        end[tag] = d
        worst = max(
            (v for k, v in d["field_max_abs"].items() if not k.endswith("_shape")), default=0.0
        )
        print(
            f"  endpoint t={t:g}, spread 0: worst renderable field |delta| {worst:.3e}, "
            f"floor level at anchor speed {d['floor_anchor_level_db']:+.3e} dB, "
            f"model LTAS worst band {d['ltas_band_max_db']:.3e} dB "
            f"(floor_exp {d['floor_exp'][0]:+.3f} vs {d['floor_exp'][1]:+.3f})"
        )
        for k, v in d["field_max_abs"].items():
            if k.endswith("_shape"):
                print(f"    note: {k[:-6]} {v} — shared order grid drops the extra orders")

    rng = np.random.default_rng([SEED, 1])
    # t is STRATIFIED for the display figures — one draw per 1/N_PATH stratum,
    # uniform inside it — so twelve draws are guaranteed to cover the path
    # instead of leaving a gap by luck. sample_path itself draws t uniformly
    # on [0, 1]; only this demo's choice of t is stratified, and the marginal
    # is still uniform.
    edges = np.linspace(0.0, 1.0, N_PATH + 1)
    t_grid = [float(rng.uniform(edges[i], edges[i + 1])) for i in range(N_PATH)]
    cloud = [RS.sample_path(A, B, rng, t=t, spread=1.0, levels=lvl) for t in t_grid]
    ts = [s["_sampler"]["path"]["t"] for s in cloud]
    attempts = sum(s["_sampler"]["attempts"] for s in cloud)
    fired: dict[str, int] = {}
    for s in cloud:
        for failed in s["_sampler"]["rejected"]:
            for g in failed:
                fired[g] = fired.get(g, 0) + 1
    print(
        f"  {N_PATH} draws, t in [{min(ts):.3f}, {max(ts):.3f}], {attempts} attempts, "
        f"rejection rate {(attempts - N_PATH) / attempts:.3f}, guards fired {fired or '{}'}"
    )
    slopes, gains, gammas = [], [], []
    for s in cloud:
        prof = RS.effective_profile(s)
        for r in range(prof.shape[0]):
            p = RS.decompose_profile(prof[r])
            slopes.append(p.slope)
            gains.append(p.gain)
        gammas.append(float(np.mean(s["gamma_slope"])))
    anchor_slopes = {}
    for nm, ex in ((a_name, A), (b_name, B)):
        prof = RS.effective_profile(ex)
        anchor_slopes[nm] = [
            float(RS.decompose_profile(prof[r]).slope) for r in range(prof.shape[0])
        ]
    print(
        f"  cloud trend slope {min(slopes):.2f} to {max(slopes):.2f} dB/dec; anchors "
        f"{a_name} {np.round(anchor_slopes[a_name], 1).tolist()}, "
        f"{b_name} {np.round(anchor_slopes[b_name], 1).tolist()}"
    )
    print(
        f"  cloud gamma_slope {min(gammas):.3f} to {max(gammas):.3f}; anchors "
        f"{float(np.mean(A['gamma_slope'])):.3f} and {float(np.mean(B['gamma_slope'])):.3f}"
    )
    print(f"  cloud rotor gain {min(gains):.2f} to {max(gains):.2f} dB")

    # cloud profile overlay, both anchors in a contrasting colour
    profile_figure(
        OUT / "path_profiles.png",
        f"path cloud {a_name} -> {b_name}: profiles ({N_PATH} draws, spread 1)",
        [
            (
                "path cloud",
                [RS.effective_profile(s) for s in cloud],
                {"colour": "#7f7f7f", "lw": 0.7, "alpha": 0.8},
            ),
            (
                f"anchor {a_name}",
                [RS.effective_profile(A)],
                {"colour": RIGS[a_name]["colour"], "lw": 2.0, "z": 5},
            ),
            (
                f"anchor {b_name}",
                [RS.effective_profile(B)],
                {"colour": RIGS[b_name]["colour"], "lw": 2.0, "z": 5},
            ),
        ],
    )

    # cloud LTAS from the MODEL spectrum: a hybrid rig has no telemetry of its
    # own, and the model LTAS is the render's expectation at its own rates
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, s in enumerate(cloud):
        level, _ = RS.model_ltas(s, RS.nominal_rates(s))
        cen, b = RS.third_octave(RS.LTAS_FREQS, level)
        ax.plot(
            cen,
            b,
            color="#7f7f7f",
            lw=0.8,
            alpha=0.8,
            label="path cloud (t uniform, spread 1)" if i == 0 else None,
        )
    for nm, ex in ((a_name, A), (b_name, B)):
        level, _ = RS.model_ltas(ex, RS.nominal_rates(ex))
        cen, b = RS.third_octave(RS.LTAS_FREQS, level)
        ax.plot(cen, b, color=RIGS[nm]["colour"], lw=2.0, label=f"anchor {nm}", zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("Hz")
    ax.set_ylabel("1/3-octave model level [dB]")
    ax.set_title("path cloud between the two rigs: model LTAS", fontsize=10)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "path_ltas.png", dpi=130)
    plt.close(fig)

    # four path draws spanning t, rendered on rig A's telemetry
    order = np.argsort(ts)
    picks = [int(order[i]) for i in np.linspace(0, len(order) - 1, N_WAV).round().astype(int)]
    path_wavs = {}
    for j in picks:
        s = cloud[j]
        a = S2.render_from_export(s, tele[a_name].rps, seed=SEED, normalize_rms=None)
        assert np.isfinite(a).all(), "a path render was not finite"
        tag = f"path_t{ts[j]:.2f}".replace(".", "p")
        wav = OUT / f"{tag}.wav"
        sf.write(
            wav,
            (a[0] * RMS_TARGET / max(rms(a), 1e-12)).astype(np.float32),
            S2.SR,
        )
        path_wavs[tag] = {"t": ts[j], "wav": str(wav), "pre_gain_dbfs": dbfs(a)}
    print(f"  rendered {len(picks)} cloud draws on {a_name} telemetry, all finite")

    # ---- does the WIDE cloud cover BOTH real rigs? ------------------------
    print(
        f"\n  path-cloud coverage of BOTH real clips ({N_SWEEP} draws per spread, each "
        f"rendered on each rig's own telemetry):"
    )
    path_cover: dict[str, dict[str, Any]] = {}
    for spread in (1.0, float(easy)):
        key = f"{spread:g}"
        if key in path_cover:
            continue
        rng_c = np.random.default_rng([SEED, 2, int(spread * 10)])
        edges_c = np.linspace(0.0, 1.0, N_SWEEP + 1)
        draws = [
            RS.sample_path(
                A,
                B,
                rng_c,
                t=float(rng_c.uniform(edges_c[i], edges_c[i + 1])),
                spread=spread,
                levels=lvl,
                max_attempts=32,
            )
            for i in range(N_SWEEP)
        ]
        att = sum(d["_sampler"]["attempts"] for d in draws)
        path_cover[key] = {
            "n_draws": len(draws),
            "acceptance_rate": len(draws) / att,
            "t": [float(d["_sampler"]["path"]["t"]) for d in draws],
        }
        for name in RIGS:
            _, real_band = ltas_bands(tele[name].audio)
            bands = np.stack(
                [
                    ltas_bands(
                        S2.render_from_export(d, tele[name].rps, seed=SEED, normalize_rms=None)
                    )[1]
                    for d in draws
                ]
            )
            cen_p, _ = ltas_bands(tele[name].audio)
            keep_p = (cen_p >= lo) & (cen_p <= hi)
            inside = (real_band >= bands.min(axis=0)) & (real_band <= bands.max(axis=0))
            path_cover[key][name] = float(inside[keep_p].mean())
            path_cover[key][name + "_all_bands"] = float(inside.mean())
        print(
            f"    spread {spread:g}: acceptance {path_cover[key]['acceptance_rate']:.3f}  "
            + "  ".join(
                f"{n} bracket {100 * path_cover[key][n]:5.1f}% in band / "
                f"{100 * path_cover[key][n + '_all_bands']:5.1f}% all"
                for n in RIGS
            )
        )

    index["path_cloud"] = {
        "anchor_a": a_name,
        "anchor_b": b_name,
        "levels_removed_dbfs": list(lvl),
        "n_draws": N_PATH,
        "t": [float(v) for v in ts],
        "attempts": attempts,
        "rejection_rate": (attempts - N_PATH) / attempts,
        "guard_fired": fired,
        "endpoint_check": end,
        "cloud_slope_db_dec": [float(min(slopes)), float(max(slopes))],
        "anchor_slope_db_dec": anchor_slopes,
        "cloud_gamma_slope": [float(min(gammas)), float(max(gammas))],
        "anchor_gamma_slope": [
            float(np.mean(A["gamma_slope"])),
            float(np.mean(B["gamma_slope"])),
        ],
        "cloud_rotor_gain_db": [float(min(gains)), float(max(gains))],
        "n_orders": cloud[0]["_sampler"]["path"]["n_orders"],
        "orders_dropped": cloud[0]["_sampler"]["path"]["orders_dropped"],
        "wavs": path_wavs,
        "figures": [str(OUT / "path_profiles.png"), str(OUT / "path_ltas.png")],
        "coverage_of_real": path_cover,
    }

    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    print(f"\nwrote {OUT / 'index.json'} and {len(list(OUT.glob('*')))} assets in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
