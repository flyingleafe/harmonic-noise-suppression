"""Build the *parameter* figures of ``docs/explainers/rig-model-listening.qmd``.

One figure per group of the model's fitted quantities, both rigs on the same
axes wherever they are comparable. Everything is read from the two accepted
summaries (``pop-{dregon,michaels}-final-w.json``) and from the hand-ranged
policy they were exported against, so a figure cannot drift from the fit.

Separate from ``build.py`` because that one loads clips and synthesizes audio
(minutes); this one is pure parameter reading (seconds).

Run from the repo root::

    PYTHONPATH=src python docs/explainers/rig-model-listening/params.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
POLICY = ROOT / "conf/online_mix/rig_fm_5050.yaml"
FITTED = {
    "dregon": ROOT / "omnirun-outputs/pop-dregon-final-w.json",
    "michaels": ROOT / "omnirun-outputs/pop-michaels-final-w.json",
}
NAME = {"dregon": "DREGON (room 2, 17 crops)", "michaels": "Michael's (FLY125, 10 crops)"}
RIG_C = {"dregon": "#1f77b4", "michaels": "#d62728"}
ROTOR_C = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
DPI = 150

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


def load() -> dict[str, dict]:
    out = {}
    for rig, path in FITTED.items():
        if not path.exists():
            raise FileNotFoundError(f"{path}: fitted summary missing — see the pipeline in the log")
        out[rig] = json.loads(path.read_text())
    return out


def support(profile: np.ndarray, drop_db: float = 30.0) -> int:
    """``raw_predictive.measured_profile_support`` in miniature: the last order
    before the profile falls off a cliff (DREGON's does; Michael's does not)."""
    from experiments.stochastic_fit.raw_predictive import measured_profile_support

    return int(measured_profile_support(profile.tolist()))


# =============================================================================
# 1. the order profile: rig mean, per-rotor deviation, clip-to-clip population
# =============================================================================


def fig_profile(S: dict[str, dict]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.4))
    for ax, rig in zip(axes, ("michaels", "dregon"), strict=True):
        rig_p = S[rig]["rig"]
        mean = np.asarray(rig_p["profile_db"], float)
        delta = np.asarray(rig_p["delta_db"], float)
        basis = np.asarray(rig_p["profile_basis_db"], float)
        res = S[rig].get("profile_residual_std_db")
        k = np.arange(1, mean.size + 1)
        sup = support(mean)

        # clip-to-clip standard deviation: the basis is orthogonal in loading
        # space, so modes add in quadrature; the residual is per order.
        pop = np.sqrt((basis**2).sum(axis=0))
        if res is not None:
            pop = np.sqrt(pop**2 + np.asarray(res, float)[: mean.size] ** 2)

        # The informative range is the measured support; DREGON's profile falls
        # off a 130 dB cliff above it, which would squash everything else.
        inner = mean[:sup]
        lo = float(np.min(inner - pop[:sup])) - 3.0
        hi = float(np.max(inner + pop[:sup])) + 3.0

        ax.fill_between(
            k,
            mean - pop,
            mean + pop,
            color=RIG_C[rig],
            alpha=0.15,
            lw=0,
            label=r"clip-to-clip $\pm1\sigma$  ($B_{jk}$, $s_k$)",
        )
        ax.vlines(k, lo, mean, color=RIG_C[rig], lw=1.1, alpha=0.5)
        ax.plot(k, mean, "o", ms=3.8, color=RIG_C[rig], label=r"rig mean $A_k$", zorder=4, mew=0)
        for r in range(delta.shape[0]):
            ax.plot(
                k,
                mean + delta[r],
                marker="_",
                ls="none",
                ms=7,
                mew=1.5,
                color=ROTOR_C[r],
                alpha=0.9,
                label=rf"rotor {r + 1}:  $A_k+\Delta_{{{r + 1},k}}$",
                zorder=3,
            )
        if sup < mean.size:
            ax.axvspan(sup + 0.5, mean.size + 0.5, color="0.55", alpha=0.2, lw=0)
            ax.annotate(
                f"measured support ends at $k$={sup}: the fitted profile drops "
                f"{mean[:sup].min() - mean[sup:].min():.0f} dB below here,\n"
                "so the renderer truncates and holds the last measured order (then pads to 200)",
                xy=(sup - 1, lo + 0.5 * (hi - lo) * 0.08),
                ha="right",
                fontsize=8.5,
                color="0.2",
            )
        ax.set_xlim(0, mean.size + 1)
        ax.set_ylim(lo, hi)
        ax.set_title(
            f"{NAME[rig]} — order profile: "
            rf"$A_2$={mean[1]:+.1f} dB, spread at $k$=64 "
            rf"$\pm${pop[min(63, pop.size - 1)]:.1f} dB"
        )
        ax.set_ylabel("line power (dB, relative)")
        ax.legend(
            ncol=6,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.10),
            framealpha=0.0,
            handletextpad=0.4,
            columnspacing=1.1,
        )
    axes[-1].set_xlabel("harmonic order $k$ (order 2 = blade passing)")
    fig.suptitle(
        "Line amplitudes: the rig profile, its four rotor deviations, and the clip population",
        fontsize=12.5,
    )
    fig.tight_layout()
    fig.savefig(HERE / "params_profile.png", dpi=DPI)
    plt.close(fig)


def fig_profile_modes(S: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    for ax, rig in zip(axes, ("michaels", "dregon"), strict=True):
        basis = np.asarray(S[rig]["rig"]["profile_basis_db"], float)
        k = np.arange(1, basis.shape[1] + 1)
        stds = S[rig]["rig"].get("profile_mode_std_db")
        for j, row in enumerate(basis):
            if np.max(np.abs(row)) < 0.05:  # an unused mode of the padded basis
                continue
            lab = f"mode {j + 1}"
            if stds is not None and j < len(stds):
                lab += f" (std {stds[j]:.1f} dB)"
            ax.plot(k, row, lw=1.4, label=lab)
        ax.axhline(0, color="0.4", lw=0.8)
        sup = support(np.asarray(S[rig]["rig"]["profile_db"], float))
        if sup < basis.shape[1]:
            ax.axvspan(sup + 0.5, basis.shape[1] + 0.5, color="0.55", alpha=0.2, lw=0)
            ax.annotate(
                f"beyond the measured support ($k>{sup}$):\ntruncated away before rendering",
                xy=(sup - 2, ax.get_ylim()[1] * 0.55),
                ha="right",
                fontsize=8,
                color="0.25",
            )
        if rig == "michaels":
            ax.axvline(64, color="0.35", ls=":", lw=1.1)
            ax.annotate(
                "the calibration is fitted to $k\\leq64$ —\nmode 2's spike sits exactly on its edge (§9.2)",
                xy=(62, -12),
                ha="right",
                fontsize=8,
                color="0.25",
            )
        ax.set_title(f"{NAME[rig]} — clip-population modes $B_{{jk}}$")
        ax.set_xlabel("harmonic order $k$")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("dB per unit loading")
    fig.suptitle("What varies between two clips of the same rig", fontsize=12)
    fig.tight_layout()
    fig.savefig(HERE / "params_profile_modes.png", dpi=DPI)
    plt.close(fig)


# =============================================================================
# 2. the width law
# =============================================================================


def fig_width(S: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, rig in zip(axes, ("michaels", "dregon"), strict=True):
        rig_p = S[rig]["rig"]
        g0 = float(np.ravel(rig_p["gamma0"])[0])
        gs = float(np.ravel(rig_p["gamma_slope"])[0])
        w = np.asarray(rig_p["width_scale"], float)
        k = np.arange(1, 81)
        for r, wr in enumerate(w):
            ax.plot(
                k,
                wr * (g0 + gs * k),
                color=ROTOR_C[r],
                lw=1.6,
                label=rf"rotor {r + 1}: $w$={wr:.2f}, $\sigma$={wr * gs / np.sqrt(2 * np.log(2)):.2f} rev/s",
            )
        ax.plot(
            k,
            g0 + gs * k,
            "k--",
            lw=1.2,
            label=rf"rig law $\gamma_0+\gamma_s k$ ({g0:.2f}, {gs:.2f})",
        )
        for s_ref, style in ((80.0, "-"), (35.0, ":")):
            ax.axhline(
                s_ref / 2,
                color="0.35",
                ls=style,
                lw=1.0,
            )
            ax.annotate(
                f"lines merge above here at {s_ref:.0f} rev/s  ($\\gamma=s/2$)",
                xy=(2, s_ref / 2 * 1.06),
                fontsize=8,
                color="0.3",
            )
        ax.set_yscale("log")
        ax.set_xlabel("harmonic order $k$")
        ax.set_title(f"{NAME[rig]} — half width at half maximum")
        ax.legend(loc="lower right", ncol=1)
    axes[0].set_ylabel("HWHM (Hz)")
    fig.suptitle(
        "Line width: one rig law times a per-rotor scale, and where it eats the comb", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(HERE / "params_width.png", dpi=DPI)
    plt.close(fig)


# =============================================================================
# 3. the floor
# =============================================================================


def fig_floor(S: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[2.1, 1]))
    ax = axes[0]
    for rig in ("michaels", "dregon"):
        ex = next(iter(S[rig]["train"].values()))["params"]
        hz = np.asarray(ex["floor_ctrl_hz"], float)
        db = np.asarray(ex["floor_shape_db"], float)
        tilt = float(np.ravel(S[rig]["rig"]["floor_tilt_db_oct"])[0])
        ax.plot(
            hz,
            db,
            "o--",
            ms=4,
            color=RIG_C[rig],
            lw=1.1,
            alpha=0.5,
            label=f"{NAME[rig]}: shape $F$ alone, 14 control points",
        )
        ax.plot(
            hz,
            db + tilt * np.log2(hz / 500.0),
            "-",
            lw=2.0,
            color=RIG_C[rig],
            label=rf"      + tilt {tilt:+.2f} dB/oct about 500 Hz  = the rendered floor",
        )
    ax.axvline(500.0, color="0.5", ls=":", lw=1.0)
    ax.annotate("tilt reference\n500 Hz", xy=(520, -95), fontsize=8, color="0.35")
    ax.set_xscale("log")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("floor level (dB, relative)")
    ax.set_title("Floor shape $F(f)$ (dashed) and $F$ + tilt (solid) = the rendered floor")
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[1]
    rows = []
    for rig in ("michaels", "dregon"):
        pop = S[rig]["population"]
        rows.append(
            (
                NAME[rig].split(" ")[0],
                pop["line_floor_mean_db"],
                pop["line_floor_std_db"],
                pop["rotor_contrast_std_db"],
                RIG_C[rig],
            )
        )
    x = np.arange(len(rows))
    ax.bar(
        x - 0.18,
        [r[1] for r in rows],
        width=0.36,
        color=[r[4] for r in rows],
        label="line-to-floor mean (dB)",
    )
    ax.errorbar(
        x - 0.18,
        [r[1] for r in rows],
        yerr=[r[2] for r in rows],
        fmt="none",
        ecolor="0.15",
        capsize=4,
        label="clip-to-clip std",
    )
    ax.bar(
        x + 0.22,
        [r[3] for r in rows],
        width=0.36,
        color="0.6",
        label="rotor contrast std (dB)",
    )
    for xi, r in zip(x, rows, strict=True):
        ax.annotate(f"{r[1]:.1f}", (xi - 0.18, r[1] + r[2] + 1.4), ha="center", fontsize=9)
        ax.annotate(f"{r[3]:.2f}", (xi + 0.22, r[3] + 1.2), ha="center", fontsize=9)
    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_ylabel("dB")
    ax.set_title("Line/floor level population")
    ax.legend(loc="upper center", fontsize=8)
    fig.suptitle("The broadband floor and how loud the comb sits above it", fontsize=12)
    fig.tight_layout()
    fig.savefig(HERE / "params_floor.png", dpi=DPI)
    plt.close(fig)


# =============================================================================
# 4. the array
# =============================================================================


def fig_mics(S: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw=dict(width_ratios=[1, 1, 1.35]))
    ims = []
    for ax, rig in zip(axes[:2], ("michaels", "dregon"), strict=True):
        g = np.asarray(S[rig]["rig"]["mic_gain_db"], float)
        im = ax.imshow(g, cmap="RdBu_r", vmin=-7.5, vmax=7.5, aspect="auto")
        ims.append(im)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{g[i, j]:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=7.4,
                    color="0.1" if abs(g[i, j]) < 4.5 else "white",
                )
        ax.set_xticks(range(g.shape[1]), [f"R{j + 1}" for j in range(g.shape[1])])
        ax.set_yticks(range(g.shape[0]), [f"m{i + 1}" for i in range(g.shape[0])])
        ax.set_title(f"{NAME[rig].split(' ')[0]} — $g_{{m\\rho}}$ (dB)")
        ax.grid(False)
    fig.colorbar(ims[-1], ax=axes[1], fraction=0.046, label="dB")

    ax = axes[2]
    m = np.arange(1, 9)
    ax.plot(
        m,
        np.asarray(S["michaels"]["rig"]["gain_all_db"], float),
        "o-",
        color=RIG_C["michaels"],
        label="Michael's: broadband per-mic gain",
    )
    ax.plot(
        m,
        np.asarray(S["dregon"]["rig"]["mic_floor_db"], float),
        "s-",
        color=RIG_C["dregon"],
        label="DREGON: per-mic FLOOR offset",
    )
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xlabel("microphone")
    ax.set_ylabel("dB")
    ax.set_title("The one array term each rig needs")
    ax.legend(loc="best")
    fig.suptitle(
        "Microphones: a per-(mic, rotor) line gain on both rigs; broadband on Michael's, "
        "floor-only on DREGON",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(HERE / "params_mics.png", dpi=DPI)
    plt.close(fig)


# =============================================================================
# 5. dynamics and the speed laws (Michael's only — DREGON has no decomposition)
# =============================================================================


def fig_dynamics(S: dict[str, dict]) -> None:
    dyn = S["michaels"]["dynamics"]
    comp = dyn["components"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    ax = axes[0]
    f = np.logspace(-2.2, 0.9, 400)
    total = np.zeros_like(f)
    for c in comp:
        s = 2 * c["std_db"] ** 2 * c["tau_s"] / (1 + (2 * np.pi * f * c["tau_s"]) ** 2)
        total += s
        ax.plot(
            f,
            10 * np.log10(s),
            lw=1.9,
            color=RIG_C["michaels"],
            label=rf"OU: $\sigma$={c['std_db']:.2f} dB, $\tau$={c['tau_s']:.2f} s, $\varrho$={c['coherence']:.3f}",
        )
    w = dyn["white_floor_db2_per_hz"]
    ax.axhline(
        10 * np.log10(w),
        color="0.45",
        ls=":",
        lw=1.4,
        label=f"white term {w:.2f} dB²/Hz (estimation noise — NOT transferred)",
    )
    lo, hi = dyn["band_hz"]
    ax.axvspan(lo, hi, color="#55a868", alpha=0.12, lw=0)
    ax.annotate(f"fit band\n{lo}–{hi} Hz", xy=(0.06, 10 * np.log10(total.max()) - 12), fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlabel("modulation frequency (Hz)")
    ax.set_ylabel("envelope PSD (dB² / Hz, dB)")
    ax.set_title(r"Amplitude process $h_{\rho k}(t)$")
    ax.legend(loc="lower left", fontsize=7.6)

    ax = axes[1]
    r = np.asarray(dyn["residual_std_db"], float)
    ax.plot(np.arange(1, r.size + 1), r, lw=1.5, color=RIG_C["michaels"])
    ax.axhline(
        comp[0]["std_db"], color="0.3", ls="--", lw=1.2, label="transferred $\\sigma$ = 2.90 dB"
    )
    ax.axvline(dyn["k_min"], color="0.5", ls=":", lw=1.1, label=f"fit uses $k\\geq${dyn['k_min']}")
    ax.set_xlabel("harmonic order $k$")
    ax.set_ylabel("dB")
    ax.set_title("Per-order envelope residual std")
    ax.legend(loc="best")

    ax = axes[2]
    fl = dyn["floor"]
    bands = np.asarray(fl["band_slopes"], float)
    ax.plot(
        np.arange(1, bands.size + 1),
        bands,
        "o-",
        color="0.35",
        ms=5,
        label="per-band floor slope (7 octave-ish bands)",
    )
    ax.axhline(
        dyn["speed_exponent"],
        color=RIG_C["michaels"],
        lw=2,
        label=rf"line exponent $q$ = {dyn['speed_exponent']:.2f} $\pm$ {dyn['speed_exponent_se']:.2f}",
    )
    ax.axhspan(
        dyn["speed_exponent"] - dyn["speed_exponent_se"],
        dyn["speed_exponent"] + dyn["speed_exponent_se"],
        color=RIG_C["michaels"],
        alpha=0.15,
        lw=0,
    )
    ax.axhline(
        fl["floor_exponent"],
        color="#55a868",
        lw=2,
        label=rf"floor exponent $q_{{fl}}$ = {fl['floor_exponent']:.2f} (q05 {fl['floor_exponent_q05_q95'][0]:.2f})",
    )
    ax.axhline(
        2.5, color="0.15", ls="--", lw=1.3, label="the 2.5 that had been pinned (DREGON keeps it)"
    )
    ax.set_xlabel("band index (low → high)")
    ax.set_ylabel("dB per dB rev/s")
    ax.set_title("Speed laws")
    ax.legend(loc="lower right", fontsize=7.6)

    fig.suptitle(
        "Michael's rig only: everything measured from the Vold–Kalman envelopes "
        "(DREGON's decomposed recording is held out)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(HERE / "params_dynamics.png", dpi=DPI)
    plt.close(fig)


# =============================================================================
# 6. the shaft populations
# =============================================================================


def fig_shaft(S: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), gridspec_kw=dict(width_ratios=[2, 2, 1.2]))
    for ax, rig in zip(axes[:2], ("michaels", "dregon"), strict=True):
        ce = S[rig]["carrier_error"]
        off = np.asarray(ce["per_clip_rotor_offsets"], float)
        rob, std = ce["static_scale_rps"], ce["static_std_rps"]
        lo, hi = ce["static_scale_q05_q95"]
        for r in range(off.shape[1]):
            ax.plot(
                np.arange(off.shape[0]) + 1,
                off[:, r],
                "o",
                ms=5,
                color=ROTOR_C[r],
                alpha=0.85,
                label=f"rotor {r + 1}",
            )
        ax.axhline(0, color="0.35", lw=0.9)
        ax.axhspan(
            -rob,
            rob,
            color=RIG_C[rig],
            alpha=0.16,
            lw=0,
            label=f"robust scale $\\pm${rob:.2f} rev/s",
        )
        for s in (-std, std):
            ax.axhline(s, color="0.25", ls="--", lw=1.1)
        ax.annotate(
            f"plain std {std:.2f} (rejected: heavy tails)",
            xy=(0.6, std * 1.03),
            fontsize=8,
            color="0.25",
        )
        ax.set_xlabel("training clip")
        ax.set_title(
            f"{NAME[rig].split(' ')[0]} — fitted shaft $-$ label offset\n"
            f"robust {rob:.2f} rev/s (90% CI {lo:.2f}–{hi:.2f})"
        )
        ax.legend(ncol=3, loc="lower right", fontsize=7.6)
    axes[0].set_ylabel(r"$\hat\delta_\rho$ (rev/s)")

    ax = axes[2]
    x = np.arange(2)
    vals, los, his, cols = [], [], [], []
    for rig in ("michaels", "dregon"):
        wp = S[rig]["width_population"]
        vals.append(wp["common_log_std"])
        los.append(wp["common_log_std_q05_q95"][0])
        his.append(wp["common_log_std_q05_q95"][1])
        cols.append(RIG_C[rig])
    ax.bar(x, vals, color=cols, width=0.55)
    ax.errorbar(
        x,
        vals,
        yerr=[np.array(vals) - np.array(los), np.array(his) - np.array(vals)],
        fmt="none",
        ecolor="0.15",
        capsize=5,
    )
    for xi, v in zip(x, vals, strict=True):
        ax.annotate(f"{v:.3f}", (xi, v + 0.03), ha="center", fontsize=9)
    ax.set_xticks(x, ["Michael's", "DREGON"])
    ax.set_ylabel(r"$s_w$ (log width, per clip)")
    ax.set_title("Clip-common width spread\n(bootstrap CI reaches 0)")
    fig.suptitle(
        "The two shaft populations: how wrong the label is, and how much gustier one clip is than another",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(HERE / "params_shaft.png", dpi=DPI)
    plt.close(fig)


# =============================================================================
# 7. calibration and every one-standard-error decision
# =============================================================================


def fig_selection(S: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    ax = axes[0]
    for rig in ("michaels", "dregon"):
        tc = S[rig]["topology_calibration"]
        c = np.asarray(tc["profile_correction_db"], float)
        ax.plot(
            np.arange(1, c.size + 1),
            c,
            lw=1.7,
            color=RIG_C[rig],
            label=(
                f"{NAME[rig].split(' ')[0]}: rank {tc['selected_rank']}, "
                f"$\\lambda$={tc['smoothing_lambda']:g}, level {float(tc['line_floor_shift_db']):+.2f} dB"
            ),
        )
    ax.axhline(0, color="0.35", lw=0.9)
    ax.set_xlabel("harmonic order $k$")
    ax.set_ylabel("dB added to the profile")
    ax.set_title("Render calibration $\\hat c_k$ (training clips only)")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    dp = S["michaels"].get("decomp_profile_model")
    if dp:
        ll = np.asarray(dp["rank_valid_loglik_per_order"], float)
        se = np.asarray(dp["rank_valid_se_per_order"], float)
        r = np.arange(ll.size)
        ax.errorbar(r, ll, yerr=se, fmt="o-", color=RIG_C["michaels"], capsize=3, ms=4)
        best = int(np.argmax(ll))
        ax.axhline(ll[best] - se[best], color="0.25", ls="--", lw=1.2, label="best $-$ 1 SE")
        ax.axvline(
            dp["selected_rank"],
            color="#55a868",
            lw=2,
            label=f"selected rank {dp['selected_rank']} (argmax was {best})",
        )
        ax.set_xlabel("profile rank $J$")
        ax.set_ylabel("held-out log-lik / order")
        ax.set_title("Profile rank, Michael's envelopes")
        ax.legend(loc="lower right", fontsize=8)

    ax = axes[2]
    dyn = S["michaels"]["dynamics"]
    v = np.asarray(dyn["component_valid_whittle"], float)
    s = np.asarray(dyn["component_valid_whittle_se"], float)
    n = np.arange(1, v.size + 1)
    ax.errorbar(n, v, yerr=s, fmt="o-", color=RIG_C["michaels"], capsize=3, ms=5)
    ax.axhline(v.max() - s[int(np.argmax(v))], color="0.25", ls="--", lw=1.2, label="best $-$ 1 SE")
    ax.axvline(dyn["selected_components"], color="#55a868", lw=2, label="selected: 1 OU")
    ax.set_xticks(n)
    ax.set_xlabel("number of OU components")
    ax.set_ylabel("held-out Whittle / bin")
    ax.set_title("Amplitude kernel order (leave-one-rotor-out)")
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "The calibration, and the two places the one-standard-error rule chose the simpler model",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(HERE / "params_selection.png", dpi=DPI)
    plt.close(fig)


def main() -> None:
    S = load()
    pol = yaml.safe_load(POLICY.read_text())
    assert pol["sources"]["noise"][0]["kind"] == "stochastic"
    fig_profile(S)
    fig_profile_modes(S)
    fig_width(S)
    fig_floor(S)
    fig_mics(S)
    fig_dynamics(S)
    fig_shaft(S)
    fig_selection(S)
    print("built parameter figures in", HERE)


if __name__ == "__main__":
    main()
