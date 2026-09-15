#!/usr/bin/env python
"""New figures for the noise-model-and-fit deck.

Everything this writes into ``assets/`` is derived from artifacts already in
the repository — the stage-2 MAP exports under ``results/S2/``, the measured
band tables under ``docs/explainers/flight-startup/``, the two rig banks under
``data/rig_banks/`` and the real recordings themselves. No number is typed in
by hand; the printed summary names the number each figure drew and where it
came from so a reader can check it against the source.

    PYTHONPATH="$PWD/src:$PWD/scripts" python prepare.py

Six figures:

1. ``model_anatomy.png``   the fitted comb and the fitted floor, drawn apart
                           and summed, with each fitted parameter labelled at
                           the feature it controls.
2. ``speed_law.png``       the three fitted speed exponents as comb level
                           against rotor speed.
3. ``band_residual.png``   model minus real per band and per rotor order, from
                           the measured flight tables.
4. ``cloud_space.png``     the two neighbourhood clouds and the path cloud in
                           the parameter coordinates that separate the rigs.
5. ``samples_hard_spectrograms.png``
                           six path-cloud draws spanning the mixing coordinate,
                           between the two real cruise clips.
6. ``real_noise.png``      the two real cruise clips alone, no model.

The path cloud's mixing coordinate ``t`` is NOT stored per entry in
``rig_hard_n2048.json`` (the bank keeps only the renderer's own fields), so it
is recovered by REPLAYING the builder's accept/reject loop from the bank's
recorded seed. The replay is checked against the bank two ways: the realised
``t`` distribution must equal the ``path_coordinate`` block the builder wrote,
and entry ``i``'s ``profile_db`` must equal replayed draw ``i``'s exactly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import experiments.stochastic_fit.clips as C
import experiments.stochastic_fit.rig_sampler as RS
import experiments.stochastic_fit.stage2 as S2
from data_processing import stochastic_rotor_noise as srn
from experiments.stochastic_fit import revised_eval as RE

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = HERE / "assets"
EXPLAIN = ROOT / "docs/explainers/flight-startup"
BANKS = ROOT / "data/rig_banks"

#: every render and every replay in this file is seeded from here
SEED = 20260915

#: the three fitted anchors. ``fit``/``clip`` locate the MAP export the
#: exponents and the profile are read from; ``speed_rps`` is the operating
#: speed the flight explainer recorded for that anchor.
ANCHORS: dict[str, dict[str, Any]] = {
    "michael_cruise": {
        "fit": "results/S2/cruise_8clip.json",
        "clip": "fly125_cruise_00",
        "label": "Michael cruise",
        "colour": "#1f77b4",
    },
    "michael_standby": {
        "fit": "results/S2/standby.json",
        "clip": "fly125_cruise_00",
        "label": "Michael standby",
        "colour": "#2ca02c",
    },
    "dregon_cruise": {
        "fit": "results/S2/dregon_room2_cruise_refined.json",
        "clip": "free-flight_nosource_room2_cruise_00",
        "label": "DREGON cruise",
        "colour": "#d62728",
    },
}

#: the two real cruise windows. FLY125 [32, 36) s is the support the low-order
#: diagnosis measured; the DREGON window is that fit's own first support.
REAL_CLIPS: dict[str, dict[str, Any]] = {
    "michael_cruise": {
        "dataset": "michaels-frames",
        "recording": "FLY125",
        "start_s": 32.0,
        "rps_key": "rps",
        "label": "Michael FLY125 cruise, 32-36 s",
    },
    "dregon_cruise": {
        "dataset": "DREGON-frames",
        "recording": "free-flight_nosource_room2",
        "start_s": 1512727417.2050455,
        "rps_key": "motors_command",
        "label": "DREGON room 2 free flight, cruise",
    },
}

CLIP_SECONDS = 4.0

#: the band the flight explainer calls agreement: above this the measured
#: residual is inside +-BAND_TOL_DB for both rigs
BAND_SPLIT_HZ = 300.0
BAND_TOL_DB = 1.8

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "text.color": "#111111",
        "axes.labelcolor": "#111111",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
    }
)


# ── shared loaders ──────────────────────────────────────────────────────────


def fit_entry(fit: str, clip: str) -> dict[str, Any]:
    """One clip's raw export entry out of a stage-2 fit summary."""
    clips = json.loads((ROOT / fit).read_text())["clips"]
    if clip in clips:
        return clips[clip]
    return next(e for cid, e in clips.items() if cid.startswith(clip))


def physical_params(fit: str, clip: str) -> dict[str, Any]:
    """One clip's export with its own ``scores.power_scale`` folded in."""
    return RE._to_physical(fit_entry(fit, clip))


def real_clip(key: str, seconds: float = CLIP_SECONDS):
    spec = REAL_CLIPS[key]
    return C.decimate(
        C.load_clip(
            spec["dataset"],
            spec["recording"],
            spec["start_s"],
            seconds,
            rps_key=spec["rps_key"],
            clip_id=key,
        ),
        S2.SR,
    )


def spectrogram_db(x: np.ndarray, n: int = 2048, hop: int = 512) -> np.ndarray:
    """``(F, N)`` magnitude spectrogram in dB of one channel."""
    x = np.asarray(x, dtype=np.float64).ravel()
    w = np.hanning(n + 1)[:n]
    frames = np.stack([x[s : s + n] * w for s in range(0, x.size - n, hop)])
    return 20.0 * np.log10(np.abs(np.fft.rfft(frames, axis=-1)).T + 1e-9)


def spec_panel(ax, x: np.ndarray, title: str, *, vmin: float, vmax: float, sr: int = S2.SR):
    spec = spectrogram_db(x)
    ax.imshow(
        spec,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0.0, np.asarray(x).size / sr, 0.0, sr / 2000.0],
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=11, pad=4)
    ax.grid(False)
    return spec


def rms_norm(x: np.ndarray, target: float = 0.1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x * (target / max(float(np.sqrt(np.mean(np.square(x)))), 1e-12))


def profile_coords(profile_db: np.ndarray) -> tuple[float, float]:
    """``(per-rotor mean gain in dB, mean trend slope in dB/decade)``."""
    prof = np.atleast_2d(np.asarray(profile_db, dtype=np.float64))
    parts = [RS.decompose_profile(prof[r]) for r in range(prof.shape[0])]
    return float(np.mean([p.gain for p in parts])), float(np.mean([p.slope for p in parts]))


#: chi-squared 95% quantile on 2 degrees of freedom — the Mahalanobis radius
#: that holds 95% of a bivariate normal, which is what the drawn island is
#: cut at. The full convex hull of 1024 draws is set by its few extreme
#: points and says nothing about where the cloud actually lives.
MAHALANOBIS_95 = 5.991


def cloud_hull(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """``(closed hull vertices, fraction of points inside)`` of a cloud's core."""
    from scipy.spatial import ConvexHull

    pts = np.column_stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)])
    centred = pts - pts.mean(axis=0)
    d2 = np.einsum("ij,jk,ik->i", centred, np.linalg.inv(np.cov(pts.T)), centred)
    core = pts[d2 <= MAHALANOBIS_95]
    hull = ConvexHull(core)
    return core[hull.vertices], float(np.mean(d2 <= MAHALANOBIS_95))


# ── 1. model anatomy ────────────────────────────────────────────────────────


def fig_model_anatomy() -> str:
    """The fitted comb, the fitted floor, and their sum, with the parameters
    labelled at the features they control.

    Computed, not drawn: ``build_psd`` is the renderer's own spectrum builder,
    so the comb trace and the floor trace are the model's two terms exactly as
    synthesis mixes them. The Gaussian-process widths are zero in a fitted
    export, so one frame is the whole static spectrum.

    ONE ROTOR of the four is drawn. The model is a per-rotor comb, and the four
    fitted rotors of this clip run at 92.4, 75.1, 80.3 and 75.1 rev/s, so all
    four combs together are four interleaved ladders and no single line can be
    pointed at. Rotor 0 alone is the model's own unit of structure.
    """
    spec = ANCHORS["michael_cruise"]
    export = physical_params(spec["fit"], spec["clip"])
    clip = real_clip("michael_cruise")
    rates = np.asarray(clip.rps, dtype=np.float64).mean(axis=1)
    params = S2.params_from_export(export, rates, sample_rate=S2.SR, n_mics=8)

    n_fft = 16384
    freqs = np.fft.rfftfreq(n_fft, 1.0 / S2.SR)
    rate = float(rates[0])
    psd = srn.build_psd(
        params,
        np.array([[rate]]),
        freqs,
        dt=n_fft / (4.0 * S2.SR),
        rng=np.random.default_rng(SEED),
    )
    gain0 = 10.0 ** (float(np.asarray(params.fixed_mic_gain_db, dtype=np.float64)[0, 0]) / 10.0)
    floor_gain = 10.0 ** (float(np.asarray(params.fixed_mic_floor_db, dtype=np.float64)[0]) / 10.0)
    comb = psd["lines"][0, 0, :] * gain0
    floor = psd["floor"][0] * floor_gain
    to_db = lambda p: 10.0 * np.log10(np.maximum(p, 1e-30))  # noqa: E731
    comb_db, floor_db, total_db = to_db(comb), to_db(floor), to_db(comb + floor)

    k_axis = np.arange(1, params.n_harmonics + 1, dtype=np.float64)
    share = np.exp(-((k_axis / float(params.coherence_k_half)) ** 2))
    gamma = params.gamma0[0] + params.gamma_slope[0] * k_axis
    line_hz = k_axis * rate
    in_band = line_hz < freqs[-1] - 20.0
    peak_db = np.full(k_axis.size, np.nan)
    peak_db[in_band] = [total_db[int(np.argmin(np.abs(freqs - f)))] for f in line_hz[in_band]]
    gain_db, slope_db_dec = profile_coords(params.profile_db)

    fig = plt.figure(figsize=(12, 5))
    grid = GridSpec(1, 3, figure=fig, width_ratios=[2.05, 0.02, 1.0], wspace=0.02)
    ax = fig.add_subplot(grid[0, 0])
    axr = fig.add_subplot(grid[0, 2])

    f_hi = 1250.0
    m = freqs <= f_hi
    ax.plot(freqs[m], comb_db[m], color="#1f77b4", lw=2.2, label="harmonic comb", zorder=3)
    ax.plot(freqs[m], floor_db[m], color="#e07b00", lw=2.6, label="broadband floor", zorder=2)
    ax.plot(
        freqs[m],
        total_db[m],
        color="#111111",
        lw=1.1,
        ls=(0, (4, 2)),
        label="their sum",
        zorder=5,
    )
    view = np.isfinite(peak_db) & (line_hz <= f_hi)
    lo = float(np.min(floor_db[m])) - 16.0
    hi = float(np.max(peak_db[view])) + 27.0
    ax.set_xlim(0, f_hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("model level (dB, periodogram units)")
    ax.set_title("linear frequency", pad=6)
    ax.legend(loc="upper left", ncol=3, framealpha=0.95, fontsize=13, borderaxespad=0.3)

    def note(axes, text, xy, xytext, *, ha="left", va="bottom"):
        axes.annotate(
            text,
            xy=xy,
            xycoords="data",
            xytext=xytext,
            textcoords="axes fraction",
            ha=ha,
            va=va,
            fontsize=11.5,
            color="#111111",
            arrowprops=dict(
                arrowstyle="-|>",
                color="#111111",
                lw=1.1,
                shrinkA=2,
                shrinkB=3,
                connectionstyle="arc3,rad=0.0",
            ),
            bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#777777", lw=0.7, alpha=0.97),
            zorder=10,
        )

    # Annotations name the VISUAL FEATURE in a few words. No symbols and no
    # formulas: the slide text carries those, the figure carries the picture.
    note(ax, "coherent low orders", (line_hz[0], peak_db[0]), (0.02, 0.90), va="top")
    k_p = 4
    note(
        ax,
        "line power per order",
        (line_hz[k_p - 1], peak_db[k_p - 1]),
        (0.355, 0.90),
        va="top",
    )
    k_w = 12
    note(
        ax,
        "width grows with order",
        (line_hz[k_w - 1], peak_db[k_w - 1] - 3.0),
        (0.995, 0.90),
        ha="right",
        va="top",
    )
    f_fl = 640.0
    note(
        ax,
        "floor level",
        (f_fl, float(np.interp(f_fl, freqs, floor_db))),
        (0.42, 0.02),
    )

    ok = np.isfinite(peak_db) & (line_hz < 7900.0)
    axr.plot(
        line_hz[ok],
        peak_db[ok],
        color="#1f77b4",
        lw=1.0,
        marker="o",
        ms=2.6,
        label="line peaks",
    )
    axr.plot(freqs, floor_db, color="#e07b00", lw=2.6, label="floor")
    # the floor's own decomposition: level + tilt, before the shape curve. This
    # is the term ``floor_tilt_db_oct`` controls, so the annotation has a line
    # to point at instead of a shaped section that happens to rise.
    tilt_only = (
        float(export["floor_mean_db"])
        + float(np.asarray(params.fixed_mic_floor_db, dtype=np.float64)[0])
        + params.floor_tilt_db_oct * np.log2(np.maximum(freqs, srn.FLOOR_SHAPE_F_MIN) / 500.0)
    )
    axr.plot(
        freqs,
        tilt_only,
        color="#a04b00",
        lw=1.6,
        ls=(0, (5, 3)),
        label="floor, no shape",
    )
    ctrl_hz = np.asarray(params.floor_ctrl_hz, dtype=np.float64)
    ctrl_db = np.interp(
        np.log(np.maximum(ctrl_hz, 30.0)), np.log(np.maximum(freqs, 30.0)), floor_db
    )
    axr.plot(
        ctrl_hz,
        ctrl_db,
        ls="none",
        marker="s",
        ms=5,
        mfc="white",
        mec="#a04b00",
        mew=1.4,
        label="control points",
    )
    axr.set_xscale("log")
    axr.set_xlim(40, 8000)
    axr.set_ylim(lo, hi)
    axr.set_xlabel("frequency (Hz), log")
    axr.set_title("log frequency, whole band", pad=6)
    axr.tick_params(labelleft=False)
    axr.set_xticks([50, 100, 500, 1000, 5000])
    axr.set_xticklabels(["50", "100", "500", "1k", "5k"])
    axr.legend(loc="upper left", ncol=2, framealpha=0.95, fontsize=11, borderaxespad=0.3)
    f_t = 6000.0
    note(
        axr,
        "floor tilt",
        (f_t, float(np.interp(f_t, freqs, tilt_only))),
        (0.985, 0.02),
        ha="right",
    )
    fig.suptitle(
        f"one model spectrum, two views: one rotor at {rate:.1f} rev/s, Michael cruise fit, mic 0",
        fontsize=14,
        y=0.995,
    )
    fig.savefig(ASSETS / "model_anatomy.png")
    plt.close(fig)
    return (
        "model_anatomy.png: results/S2/cruise_8clip.json:fly125_cruise_00 folded by "
        f"revised_eval._to_physical (power_scale_folded_db {export['power_scale_folded_db']:.3f}); "
        f"rotor 0 of 4 drawn at {rate:.2f} rev/s (all four "
        f"{np.array2string(rates, precision=2)} rev/s), k_use {params.n_harmonics}, "
        f"profile gain {gain_db:.2f} dB, trend {slope_db_dec:.2f} dB/decade, "
        f"gamma0 {params.gamma0[0]:.3f} Hz + {params.gamma_slope[0]:.4f} Hz/order "
        f"({gamma[k_w - 1]:.2f} Hz at k={k_w}), coherence_k_half {params.coherence_k_half:.4f} "
        f"(w_1 {share[0]:.3f}, w_2 {share[1]:.3f}, w_3 {share[2]:.3f}), "
        f"floor_tilt {params.floor_tilt_db_oct:.3f} dB/oct, "
        f"floor_mean {float(export['floor_mean_db']):.2f} dB, "
        f"k=1 peak {peak_db[0]:.1f} dB, "
        f"{peak_db[0] - float(np.interp(line_hz[0], freqs, floor_db)):.1f} dB over the floor; "
        f"y range {lo:.1f} to {hi:.1f} dB"
    )


# ── 2. speed law ────────────────────────────────────────────────────────────


def fig_speed_law() -> str:
    """Comb level against rotor speed for the three fitted exponents."""
    rows = []
    for key in ("michael_cruise", "michael_standby", "dregon_cruise"):
        spec = ANCHORS[key]
        export = physical_params(spec["fit"], spec["clip"])
        rows.append(
            dict(
                key=key,
                label=spec["label"],
                colour=spec["colour"],
                amp_exp=float(export["amp_exp"]),
                floor_exp=float(export.get("floor_exp", export["amp_exp"])),
                speed=float(np.mean(np.asarray(export["carrier"], dtype=np.float64))),
            )
        )
    # the flight explainer's own recorded operating speeds for the same anchors
    flight = {
        "michaels": json.loads((EXPLAIN / "flight_michaels.json").read_text()),
        "dregon": json.loads((EXPLAIN / "flight_dregon.json").read_text()),
    }
    recorded = {
        "michael_cruise": ("michaels", "cruise"),
        "michael_standby": ("michaels", "standby"),
        "dregon_cruise": ("dregon", "cruise"),
    }
    for row in rows:
        rig, label = recorded[row["key"]]
        anchor = next(a for a in flight[rig]["anchors"] if a["label"] == label)
        row["speed_rps"] = float(anchor["speed_rps"])
        row["amp_exp_flight"] = float(anchor["amp_exp_fitted"])
    step = flight["michaels"]["segments_step_db"]

    ref = float(srn.StochasticParams.amp_rps_ref)
    speed = np.linspace(10.0, 105.0, 400)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(10, 105)
    # the top band above +14 dB is empty by construction: it exists so the
    # legend has somewhere to sit that covers no curve
    ax.set_ylim(-95, 42)
    ax.axvline(ref, color="#555555", lw=1.2, ls="--", zorder=1)
    ax.axhline(0.0, color="#555555", lw=1.0, zorder=1)
    ax.annotate(
        f"reference {ref:.0f} rev/s",
        xy=(ref - 1.5, 33.0),
        ha="right",
        va="center",
        fontsize=11.5,
        color="#333333",
    )
    for row in rows:
        level = 10.0 * row["amp_exp"] * np.log10(speed / ref)
        ax.plot(
            speed,
            level,
            color=row["colour"],
            lw=2.8,
            label=f"{row['label']}: amp_exp {row['amp_exp']:.2f} dB/dB",
        )
        row["y_anchor"] = 10.0 * row["amp_exp"] * np.log10(row["speed_rps"] / ref)
        ax.plot(
            [row["speed_rps"]],
            [row["y_anchor"]],
            marker="o",
            ms=12,
            mfc=row["colour"],
            mec="white",
            mew=1.8,
            ls="none",
            zorder=6,
        )
        row["half_db"] = 10.0 * row["amp_exp"] * np.log10(0.5)
    # The two cruise anchors sit within 0.11 rev/s of each other AND at the same
    # 0 dB, so one label goes up-left and the other down-right of the crossing.
    # Every label is one line: the rig name is already in the legend.
    offsets = {
        "michael_cruise": (-11.0, 7.0, "right", "bottom"),
        "dregon_cruise": (5.5, -8.0, "left", "top"),
        "michael_standby": (-1.8, 5.0, "right", "bottom"),
    }
    for row in rows:
        dx, dy, ha, va = offsets[row["key"]]
        ax.annotate(
            f"anchor {row['speed_rps']:.1f} rev/s",
            xy=(row["speed_rps"], row["y_anchor"]),
            xytext=(row["speed_rps"] + dx, row["y_anchor"] + dy),
            ha=ha,
            va=va,
            fontsize=11.5,
            color=row["colour"],
            arrowprops=dict(arrowstyle="-", color=row["colour"], lw=1.0, shrinkA=1, shrinkB=7),
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.88),
            zorder=8,
        )

    cruise = next(r for r in rows if r["key"] == "michael_cruise")
    standby = next(r for r in rows if r["key"] == "michael_standby")
    y_c = 10.0 * cruise["amp_exp"] * np.log10(standby["speed_rps"] / ref)
    y_s = standby["y_anchor"]
    ax.annotate(
        "",
        xy=(standby["speed_rps"], y_s),
        xytext=(standby["speed_rps"], y_c),
        arrowprops=dict(arrowstyle="<|-|>", color="#111111", lw=1.6),
    )
    ax.annotate(
        f"{abs(y_s - y_c):.1f} dB",
        xy=(standby["speed_rps"], 0.5 * (y_s + y_c)),
        ha="center",
        va="center",
        fontsize=11.5,
        color="#111111",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.92),
        zorder=9,
    )
    ax.set_xlabel("rotor speed (rev/s)")
    ax.set_ylabel("comb level relative to 80 rev/s (dB)")
    ax.set_title("fitted speed law: comb power goes as (rev/s)^amp_exp", pad=6)
    ax.legend(loc="upper left", framealpha=0.97, borderaxespad=0.5)
    ax.annotate(
        "half the speed costs\n"
        + "\n".join(f"{r['half_db']:7.1f} dB   {r['label']}" for r in rows)
        + f"\n\nmeasured real standby-to-cruise step {step['real']:.1f} dB\n"
        + f"cruise law at {standby['speed_rps']:.0f} rev/s is {abs(y_s - y_c):.1f} dB "
        "under the standby fit",
        xy=(0.995, 0.02),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=11.5,
        color="#111111",
        linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#777777", lw=0.7, alpha=0.97),
    )
    fig.savefig(ASSETS / "speed_law.png")
    plt.close(fig)
    return (
        "speed_law.png: "
        + "; ".join(
            f"{r['label']} amp_exp {r['amp_exp']:.4f} (flight json {r['amp_exp_flight']:.4f}), "
            f"floor_exp {r['floor_exp']:.4f}, anchor {r['speed_rps']:.2f} rev/s "
            f"(export carrier {r['speed']:.2f}), half speed {r['half_db']:.2f} dB"
            for r in rows
        )
        + (
            f"; reference {ref:.0f} rev/s; measured real standby-to-cruise step "
            f"{step['real']:.2f} dB, model {step['model']:.2f} dB; cruise law extrapolated to "
            f"{standby['speed_rps']:.2f} rev/s reads {y_c:.2f} dB against the standby law's "
            f"{y_s:.2f} dB, a gap of {abs(y_s - y_c):.2f} dB"
        )
    )


# ── 3. band residual ────────────────────────────────────────────────────────


def fig_band_residual() -> str:
    """Model minus real, per frequency band and per rotor order.

    Route: both panels are READ, not recomputed. The band panel uses
    ``settled_absolute_bands`` of the two flight JSONs — absolute band levels
    of the whole settled cruise segment, model and real, on the physical level
    path. The order panel uses ``settled_low_orders``, which separates the level
    AT a line from the level between lines, and the low-order diagnosis JSON
    supplies the coherent share that explains the excess.
    """
    rigs = {
        "michaels": json.loads((EXPLAIN / "flight_michaels.json").read_text()),
        "dregon": json.loads((EXPLAIN / "flight_dregon.json").read_text()),
    }
    diag = json.loads((EXPLAIN / "low_order_diagnosis.json").read_text())
    style = {
        "michaels": ("Michael cruise", "#1f77b4"),
        "dregon": ("DREGON cruise", "#d62728"),
    }

    fig, (ax, axr) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(wspace=0.22))

    edges = np.asarray(rigs["michaels"]["settled_absolute_bands"]["edges_hz"], dtype=np.float64)
    assert np.allclose(edges, rigs["dregon"]["settled_absolute_bands"]["edges_hz"])
    n_band = edges.size - 1
    x = np.arange(n_band, dtype=np.float64)
    width = 0.38
    summary_bands: dict[str, Any] = {}
    for i, (rig, (label, colour)) in enumerate(style.items()):
        resid = np.asarray(
            rigs[rig]["settled_absolute_bands"]["model_minus_real_db"], dtype=np.float64
        )
        ax.bar(x + (i - 0.5) * width, resid, width=width, color=colour, label=label, zorder=3)
        above = edges[:-1] >= BAND_SPLIT_HZ
        summary_bands[rig] = dict(
            resid=resid,
            max_abs_above=float(np.abs(resid[above]).max()),
            low=float(resid[(edges[:-1] >= 50.0) & (edges[1:] <= 200.0)].max()),
        )
    i_split = int(np.argmin(np.abs(edges - BAND_SPLIT_HZ)))
    ax.axhline(0.0, color="#111111", lw=1.3, zorder=4)
    ax.axvspan(i_split - 0.5, n_band - 0.5, color="#dddddd", alpha=0.55, zorder=0)
    ax.fill_between(
        [i_split - 0.5, n_band - 0.5],
        -BAND_TOL_DB,
        BAND_TOL_DB,
        color="#4daf4a",
        alpha=0.30,
        zorder=1,
        label=f"+-{BAND_TOL_DB} dB above {BAND_SPLIT_HZ:.0f} Hz",
    )

    def band_name(lo: float, hi: float) -> str:
        fmt = lambda v: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"  # noqa: E731
        return f"{fmt(lo)}-{fmt(hi)}"

    ax.set_xticks(x)
    ax.set_xticklabels(
        [band_name(edges[i], edges[i + 1]) for i in range(n_band)],
        fontsize=11,
        rotation=40,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_xlim(-0.7, n_band - 0.3)
    ax.set_ylim(-7.0, 27.0)
    ax.set_xlabel("band (Hz)")
    ax.set_ylabel("model minus real (dB)")
    ax.set_title("whole settled cruise segment, all 8 mics", pad=6)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=11)
    ax.annotate(
        f"above {BAND_SPLIT_HZ:.0f} Hz the worst band is "
        f"{max(summary_bands[r]['max_abs_above'] for r in summary_bands):.1f} dB",
        xy=(0.98, 0.56),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=11.5,
        color="#111111",
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#777777", lw=0.7, alpha=0.97),
    )

    orders = np.asarray(rigs["michaels"]["settled_low_orders"]["orders"], dtype=np.float64)
    summary_orders: dict[str, Any] = {}
    for rig, (label, colour) in style.items():
        low = rigs[rig]["settled_low_orders"]
        assert list(low["orders"]) == list(orders)
        at_line = np.asarray(low["line_model_minus_real_db"], dtype=np.float64)
        between = np.asarray(low["null_model_minus_real_db"], dtype=np.float64)
        axr.plot(
            orders, at_line, color=colour, lw=2.6, marker="o", ms=7, label=f"{label}, at the line"
        )
        axr.plot(
            orders,
            between,
            color=colour,
            lw=1.8,
            ls=":",
            marker="s",
            ms=6,
            mfc="white",
            label=f"{label}, between lines",
        )
        summary_orders[rig] = dict(at_line=at_line, between=between)
    axr.axhline(0.0, color="#111111", lw=1.3, zorder=4)
    axr.fill_between([0.6, 12], -BAND_TOL_DB, BAND_TOL_DB, color="#4daf4a", alpha=0.30, zorder=1)
    axr.set_xscale("log")
    axr.set_xticks(orders)
    axr.set_xticklabels([f"{int(k)}" for k in orders])
    axr.set_xlim(0.85, 11.5)
    axr.set_ylim(-7.0, 34.0)
    axr.set_xlabel("rotor order k (line at k x 80.7 Hz)")
    axr.set_ylabel("model minus real (dB)")
    axr.set_title("the excess sits ON the first lines, not in the floor", pad=6)
    axr.legend(loc="upper right", framealpha=0.97, fontsize=10.5)
    share = [float(r["coherent_share"]) for r in diag["michaels"]["rows"][:3]]
    axr.annotate(
        "coherent share w_k\n" + ", ".join(f"{v:.2f}" for v in share) + " at k = 1, 2, 3",
        xy=(0.97, 0.55),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=11.5,
        color="#111111",
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#777777", lw=0.7, alpha=0.97),
    )
    fig.savefig(ASSETS / "band_residual.png")
    plt.close(fig)
    parts = [
        "band_residual.png: ROUTE = read from docs/explainers/flight-startup "
        "(settled_absolute_bands + settled_low_orders); nothing recomputed"
    ]
    for rig, (label, _) in style.items():
        b, o = summary_bands[rig], summary_orders[rig]
        parts.append(
            f"{label}: bands {np.array2string(b['resid'], precision=2)} dB over edges "
            f"{[int(v) for v in edges]}; max |resid| above {BAND_SPLIT_HZ:.0f} Hz "
            f"{b['max_abs_above']:.2f} dB; 50-200 Hz excess up to {b['low']:.2f} dB; "
            f"at-line residual {np.array2string(o['at_line'], precision=2)} dB vs between-line "
            f"{np.array2string(o['between'], precision=2)} dB at orders {[int(k) for k in orders]}"
        )
    parts.append(
        "michaels LTAS mean |model-real| "
        f"{rigs['michaels']['settled_ltas']['mean_abs_db']:.2f} dB, dregon "
        f"{rigs['dregon']['settled_ltas']['mean_abs_db']:.2f} dB"
    )
    return "; ".join(parts)


# ── the two banks, and the path cloud's mixing coordinate ───────────────────


def load_bank(name: str) -> dict[str, Any]:
    return json.loads((BANKS / f"rig_{name}_n2048.json").read_text())


def bank_coords(entries: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Per-entry coordinates read out of the bank's own renderer fields."""
    gain, slope, gamma, amp = [], [], [], []
    for e in entries:
        g, s = profile_coords(e["profile_db"])
        gain.append(g)
        slope.append(s)
        gamma.append(float(np.mean(np.asarray(e["gamma_slope"], dtype=np.float64))))
        amp.append(float(e["amp_rps_exponent"]))
    return dict(
        gain=np.asarray(gain),
        slope=np.asarray(slope),
        gamma_slope=np.asarray(gamma),
        amp_exp=np.asarray(amp),
    )


def replay_hard_path(bank: dict[str, Any]) -> tuple[list[dict[str, Any]], np.ndarray, str]:
    """``(exports, t per entry, what the replay proved)``.

    The hard bank stores only the renderer's fields, so the mixing coordinate
    is recovered by rerunning ``scripts/_build_rig_bank.build``'s accept/reject
    loop from the recorded seed. Both checks below must pass or the coordinate
    is not the bank's.
    """
    import _build_rig_bank as brb

    prov = bank["provenance"]
    anchors = [RS.load_anchor(ROOT / a["fit"], a["clip"]) for a in prov["anchors"]]
    levels = (RS.model_level_db(anchors[0]), RS.model_level_db(anchors[1]))
    exports: list[dict[str, Any]] = []
    sub = 0
    while len(exports) < int(prov["n"]):
        rng = np.random.default_rng([int(prov["seed"]), sub])
        sub += 1
        cand = RS.sample_path(
            anchors[0],
            anchors[1],
            rng,
            spread=float(prov["spread"]),
            max_attempts=int(prov["max_attempts"]),
            levels=levels,
        )
        t = float(cand["_sampler"]["path"]["t"])
        if not brb.silent_when_stopped(cand):
            continue
        cand, clipped = brb.clip_exponents(cand)
        if clipped:
            guards = RS.check_sample(cand, brb.path_reference(anchors, levels, t))
            if not guards["ok"]:
                continue
        exports.append(cand)
    t = np.asarray([float(e["_sampler"]["path"]["t"]) for e in exports], dtype=np.float64)

    recorded = prov["path_coordinate"]
    counts = np.histogram(t, bins=10, range=(0.0, 1.0))[0]
    mean_ok = abs(float(t.mean()) - float(recorded["mean"])) < 1e-9
    hist_ok = list(int(c) for c in counts) == list(recorded["counts"])
    worst = 0.0
    for i in range(0, len(exports), 97):
        pe = np.asarray(bank["entries"][i]["profile_db"], dtype=np.float64)
        px = np.atleast_2d(np.asarray(exports[i]["profile_db"], dtype=np.float64))
        worst = max(worst, float(np.abs(pe - px[: pe.shape[0], : pe.shape[1]]).max()))
    if not (mean_ok and hist_ok and worst == 0.0):
        raise RuntimeError(
            f"hard-bank replay does not reproduce the bank: mean_ok={mean_ok} "
            f"hist_ok={hist_ok} worst profile_db mismatch={worst}"
        )
    proof = (
        f"replayed {sub} substreams for {len(exports)} entries; t mean {t.mean():.6f} == recorded "
        f"{float(recorded['mean']):.6f}, ten-bin counts identical, entry-vs-replay profile_db "
        "max |diff| 0.0 over every 97th entry"
    )
    return exports, t, proof


# ── 4. cloud space ──────────────────────────────────────────────────────────


def fig_cloud_space(easy: dict[str, Any], hard: dict[str, Any], t: np.ndarray) -> str:
    """Easy neighbourhoods against the hard path cloud, in the coordinates that
    tell the two real rigs apart."""
    ce = bank_coords(easy["entries"])
    ch = bank_coords(hard["entries"])
    split = int(easy["provenance"]["anchors"][0]["entries"])
    rates = np.asarray(easy["provenance"]["order_ladder"]["min_rps_vector"], dtype=np.float64)

    anchors = {}
    for key, arm in (("michael_cruise", 0), ("dregon_cruise", 1)):
        meta = easy["provenance"]["anchors"][arm]
        export = RS.load_anchor(ROOT / meta["fit"], meta["clip"])
        params = S2.params_from_export(
            export, rates, sample_rate=int(easy["entries"][0]["sample_rate"]), n_mics=8
        )
        gain, slope = profile_coords(params.profile_db)
        anchors[key] = dict(
            gain=gain,
            slope=slope,
            gamma_slope=float(np.mean(params.gamma_slope)),
            label=ANCHORS[key]["label"],
            colour=ANCHORS[key]["colour"],
            marker="*" if arm == 0 else "P",
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw=dict(wspace=0.06))
    panels = (
        ("gain", "per-rotor mean line gain (dB)"),
        ("gamma_slope", "line width slope (Hz per order)"),
    )
    order = np.argsort(t)
    sc = None
    hulls: dict[str, dict[str, float]] = {}
    for ax, (xkey, xlabel) in zip(axes, panels):
        # Each easy arm is drawn as an ISLAND — a filled, thickly outlined hull
        # over its own draws — and the hard cloud as small t-coloured dots on
        # top. The two populations then differ in kind, not only in colour, so
        # "two islands, one bridge" is legible without reading a legend.
        for arm, (sl, colour, label) in enumerate(
            (
                (slice(0, split), "#1f77b4", "easy: Michael island"),
                (slice(split, None), "#d62728", "easy: DREGON island"),
            )
        ):
            ax.scatter(ce[xkey][sl], ce["slope"][sl], s=5, c=colour, alpha=0.18, lw=0, zorder=1)
            poly, inside = cloud_hull(ce[xkey][sl], ce["slope"][sl])
            ax.fill(
                poly[:, 0],
                poly[:, 1],
                facecolor=colour,
                alpha=0.30,
                edgecolor=colour,
                lw=3.0,
                zorder=2,
                label=f"{label} (95%)" if xkey == "gain" else None,
            )
            hulls[f"{xkey}:{arm}"] = dict(inside=inside)
        sc = ax.scatter(
            ch[xkey][order],
            ch["slope"][order],
            s=9,
            c=t[order],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            alpha=0.95,
            lw=0,
            zorder=4,
            label=None,
        )
        ax.plot(
            [anchors["michael_cruise"][xkey], anchors["dregon_cruise"][xkey]],
            [anchors["michael_cruise"]["slope"], anchors["dregon_cruise"]["slope"]],
            color="#111111",
            lw=2.0,
            ls="--",
            zorder=6,
        )
        for a in anchors.values():
            ax.plot(
                [a[xkey]],
                [a["slope"]],
                marker=a["marker"],
                ms=24 if a["marker"] == "*" else 16,
                mfc=a["colour"],
                mec="white",
                mew=2.2,
                ls="none",
                zorder=7,
                label=f"{a['label']} fit" if xkey == "gain" else None,
            )
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("line profile trend (dB per decade of order)")
    axes[1].tick_params(labelleft=False)
    axes[0].set_title("level coordinate: the islands overlap", pad=6)
    axes[1].set_title("linewidth coordinate: the islands do not touch", pad=6)
    # the bottom band below the lowest draw is empty by construction so the
    # legend has somewhere to sit that covers no island and no anchor
    lo = min(ce["slope"].min(), ch["slope"].min()) - 7.0
    hi = max(ce["slope"].max(), ch["slope"].max()) + 1.5
    for ax in axes:
        ax.set_ylim(lo, hi)
    axes[0].legend(loc="lower right", framealpha=0.96, fontsize=12, borderaxespad=0.4)
    cb = fig.colorbar(sc, ax=axes, orientation="horizontal", fraction=0.055, pad=0.11, aspect=45)
    cb.set_label(
        "hard cloud: mixing coordinate t   (0 = Michael rig,  1 = DREGON rig)", fontsize=13
    )
    cb.ax.tick_params(labelsize=12)
    fig.suptitle(
        "easy = two islands around two fits;  hard = one bridge between them",
        fontsize=15,
        y=0.97,
    )
    fig.savefig(ASSETS / "cloud_space.png")
    plt.close(fig)

    def desc(a: np.ndarray) -> str:
        return f"{a.mean():.2f}+-{a.std():.2f} [{a.min():.2f}, {a.max():.2f}]"

    corr = {k: float(np.corrcoef(t, ch[k])[0, 1]) for k in ("gain", "slope", "gamma_slope")}
    mc, dc = anchors["michael_cruise"], anchors["dregon_cruise"]
    inside = [hulls[k]["inside"] for k in sorted(hulls)]
    return (
        f"cloud_space.png: easy Michael arm n={split} gain {desc(ce['gain'][:split])} "
        f"slope {desc(ce['slope'][:split])} gamma_slope {desc(ce['gamma_slope'][:split])}; "
        f"easy DREGON arm n={len(easy['entries']) - split} gain {desc(ce['gain'][split:])} "
        f"slope {desc(ce['slope'][split:])} gamma_slope {desc(ce['gamma_slope'][split:])}; "
        f"hard n={len(hard['entries'])} gain {desc(ch['gain'])} slope {desc(ch['slope'])} "
        f"gamma_slope {desc(ch['gamma_slope'])}, t {desc(t)}; corr(t, .) "
        f"gain {corr['gain']:+.3f} slope {corr['slope']:+.3f} "
        f"gamma_slope {corr['gamma_slope']:+.3f}; anchors Michael (gain {mc['gain']:.2f} dB, "
        f"slope {mc['slope']:.2f} dB/dec, gamma_slope {mc['gamma_slope']:.4f} Hz/order) and "
        f"DREGON (gain {dc['gain']:.2f} dB, slope {dc['slope']:.2f} dB/dec, "
        f"gamma_slope {dc['gamma_slope']:.4f} Hz/order); drawn islands are convex hulls of the "
        f"95% Mahalanobis core, holding {min(inside):.3f}-{max(inside):.3f} of each arm's draws"
    )


# ── 5. hard-cloud spectrograms ──────────────────────────────────────────────


def fig_samples_hard_spectrograms(exports: list[dict[str, Any]], t: np.ndarray) -> str:
    """Six path-cloud draws spanning t, between the two real cruise clips.

    Each draw is rendered by the same path the bank declares
    (``stage2.render_from_export``, 44.1 kHz work grid, antialias, decimate to
    16 kHz) on the Michael cruise clip's own telemetry, so the only thing that
    changes across the six panels is the rig.
    """
    targets = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    picks = [int(np.argmin(np.abs(t - v))) for v in targets]
    michael = real_clip("michael_cruise")
    dregon = real_clip("dregon_cruise")

    renders = []
    for n, i in enumerate(picks):
        audio = S2.render_from_export(
            exports[i],
            michael.rps,
            seed=SEED + n,
            n_mics=1,
            normalize_rms=None,
        )
        renders.append(np.asarray(audio, dtype=np.float64)[0])

    gamma = [float(np.mean(np.asarray(exports[i]["gamma_slope"], dtype=np.float64))) for i in picks]
    signals = [rms_norm(np.asarray(michael.audio)[0])]
    titles = ["REAL Michael FLY125 cruise\n32-36 s, mic 0"]
    signals += [rms_norm(r) for r in renders]
    titles += [
        f"hard draw, t = {t[i]:.2f}\ngamma_slope {g:.2f} Hz/order" for i, g in zip(picks, gamma)
    ]
    signals.append(rms_norm(np.asarray(dregon.audio)[0]))
    titles.append("REAL DREGON room 2 cruise\nfree flight, mic 0")

    specs = [spectrogram_db(s) for s in signals]
    vmax = float(np.percentile(np.concatenate([s.ravel() for s in specs]), 99.8))
    vmin = vmax - 75.0

    fig, axes = plt.subplots(2, 4, figsize=(15, 5), gridspec_kw=dict(wspace=0.09, hspace=0.30))
    for ax, sig, title in zip(axes.ravel(), signals, titles):
        spec_panel(ax, sig, title, vmin=vmin, vmax=vmax)
    for i, ax in enumerate(axes.ravel()):
        if i % 4 == 0:
            ax.set_ylabel("kHz")
        else:
            ax.tick_params(labelleft=False)
        if i >= 4:
            ax.set_xlabel("time (s)")
        else:
            ax.tick_params(labelbottom=False)
    for ax in (axes.ravel()[0], axes.ravel()[7]):
        for side in ax.spines.values():
            side.set_color("#111111")
            side.set_linewidth(2.4)
    fig.savefig(ASSETS / "samples_hard_spectrograms.png")
    plt.close(fig)
    return (
        f"samples_hard_spectrograms.png: t targets {np.array2string(targets, precision=1)} -> "
        f"entries {picks} at t {np.array2string(t[picks], precision=3)}; every panel "
        f"RMS-normalised to 0.1 and drawn on ONE shared colour scale, {vmin:.1f} to {vmax:.1f} dB "
        "(99.8th percentile over all eight panels, 75 dB range); renders 4.0 s, 1 mic, "
        f"seeds {SEED}..{SEED + len(picks) - 1}, stage2.render_from_export on the Michael cruise "
        f"telemetry (mean {float(np.asarray(michael.rps).mean()):.2f} rev/s); gamma_slope per "
        f"panel {[round(g, 3) for g in gamma]} Hz/order; reference panels are the real clips, "
        "outlined"
    )


# ── 6. the real noise, alone ────────────────────────────────────────────────


def fig_real_noise() -> str:
    """The two real cruise clips as spectrograms. No model anywhere."""
    clips = {k: real_clip(k) for k in REAL_CLIPS}
    signals = {k: rms_norm(np.asarray(c.audio)[0]) for k, c in clips.items()}
    specs = {k: spectrogram_db(v) for k, v in signals.items()}
    vmax = float(np.percentile(np.concatenate([s.ravel() for s in specs.values()]), 99.8))
    vmin = vmax - 75.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw=dict(wspace=0.07))
    for ax, (key, sig) in zip(axes, signals.items()):
        spec_panel(ax, sig, REAL_CLIPS[key]["label"], vmin=vmin, vmax=vmax)
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("frequency (kHz)")
    axes[1].tick_params(labelleft=False)
    fig.savefig(ASSETS / "real_noise.png")
    plt.close(fig)
    return (
        f"real_noise.png: mic 0 of 8, 4.0 s, 0-8 kHz, shared colour scale {vmin:.1f} to "
        f"{vmax:.1f} dB, each panel RMS-normalised to 0.1; "
        f"{REAL_CLIPS['michael_cruise']['label']} "
        f"(mean {float(np.asarray(clips['michael_cruise'].rps).mean()):.2f} rev/s) and "
        f"{REAL_CLIPS['dregon_cruise']['label']} "
        f"(mean {float(np.asarray(clips['dregon_cruise'].rps).mean()):.2f} rev/s)"
    )


# ── main ────────────────────────────────────────────────────────────────────


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    t0 = time.time()
    lines.append(fig_model_anatomy())
    lines.append(fig_speed_law())
    lines.append(fig_band_residual())
    lines.append(fig_real_noise())
    easy, hard = load_bank("easy"), load_bank("hard")
    exports, t, proof = replay_hard_path(hard)
    lines.append("hard-bank t recovery: " + proof)
    lines.append(fig_cloud_space(easy, hard, t))
    lines.append(fig_samples_hard_spectrograms(exports, t))
    print("\n".join(f"- {line}" for line in lines))
    names = (
        "model_anatomy.png",
        "speed_law.png",
        "band_residual.png",
        "cloud_space.png",
        "samples_hard_spectrograms.png",
        "real_noise.png",
    )
    print(f"\nwrote {len(names)} figures to {ASSETS} in {time.time() - t0:.1f} s:")
    for name in names:
        path = ASSETS / name
        print(f"  {path.relative_to(ROOT)}  {path.stat().st_size / 1e3:.0f} kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
