"""Generate the generative-model figures for the supervisor deck and the paper.

Writes ``gen_*.pdf`` and ``gen_*.png`` into the paper submodule's ``src/figures/``
and ``gen_*.tex`` (booktabs) into ``src/tables/``; the PNGs are what the deck
(``writing/slides/2026-09-22_generative-model-supervisor/slides.qmd``) embeds.

Figures
-------
``gen_traj_real_vs_sampled``
    30 s of real airborne telemetry (4 rotors, 100 Hz grid) beside a 30 s draw
    from that rig's fitted trajectory model, dregon and michaels, shared y-axis
    per rig.
``gen_traj_hyperprior``
    Four flights drawn from the rig hyperprior (``traj_rig="posterior"``),
    30 s each, hover level in the title.
``gen_noise_fits``
    (a) Real DREGON room-2 free-flight and MD2 FLY125 standby-to-cruise windows
    against the noise-model-v4 fits of the easy training bank rendered on the
    SAME speed label, one colour range per pair; (b) the two fits with every
    rotor at 80 Hz: line cores, pedestals and broadband.
``gen_noise_prior``
    (a) One drone of each wide-prior validation sector on the same training
    trajectory, rotor noise at unit RMS, one colour range; (b) the metric MDS
    of the spectral rig distance: training bank, easy neighbourhoods, fits and
    the drones of (a).

The paper's figures (``fits``, ``noise``, ``hyper``, ``bank``, ``prior``) are
drawn at print size in :func:`plots.paper.paper_style`; the deck's (``traj``,
``traj_other``) keep the deck style.

Run: ``systemd-run --user --scope -p MemoryMax=10G python scripts/gen_model_figures.py [targets]``
"""

from __future__ import annotations

import functools
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from plots.paper import COLUMN_IN, TEXT_IN, paper_style  # noqa: E402

PAPER = ROOT / "writing/papers/2026-08_wrapup/src"
FIGS = PAPER / "figures"
TABLES = PAPER / "tables"

SR = 16000
NFFT = 2048
HOP = 512
ROTOR_LABELS = ("RFront", "LFront", "LBack", "RBack")
COLOURS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})


def save(fig, name: str, *, paper: bool = False) -> None:
    """Write ``name``.pdf/.png. A ``paper`` figure is drawn at its print size and
    saved at exactly that size (no tight bbox), so it is included unscaled."""
    FIGS.mkdir(parents=True, exist_ok=True)
    kw = {} if paper else {"bbox_inches": "tight"}
    fig.savefig(FIGS / f"{name}.pdf", **kw)
    fig.savefig(FIGS / f"{name}.png", dpi=300 if paper else 160, **kw)
    plt.close(fig)
    print(f"wrote {name}.pdf/.png", flush=True)


def _paper(fn: Callable[..., None]) -> Callable[..., None]:
    """Draw ``fn``'s figure in the paper's house style (:mod:`plots.paper`)."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs) -> None:
        with paper_style():
            fn(*args, **kwargs)

    return wrapped


# ── trajectories ────────────────────────────────────────────────────────────


def _real_window(rig: str, seconds: float = 30.0):
    from experiments.rps_traj.data import RATE_HZ, airborne_segments, load_rig

    n = int(seconds * RATE_HZ)
    best = None
    for fl in load_rig(rig):
        for sl in airborne_segments(fl.rps, RATE_HZ):
            seg = fl.rps[:, sl]
            if seg.shape[1] >= n and np.isfinite(seg[:, :n]).all():
                span = float(np.nanmax(seg[:, :n]) - np.nanmin(seg[:, :n]))
                if best is None or span > best[0]:
                    best = (span, seg[:, :n])
    if best is None:
        raise RuntimeError(f"no {seconds} s airborne window on {rig}")
    return best[1]


def _plot_tracks(ax, rps, fs, title, ylim=None):
    t = np.arange(rps.shape[1]) / fs
    for r in range(rps.shape[0]):
        ax.plot(t, rps[r], lw=0.7, color=COLOURS[r % 4], label=ROTOR_LABELS[r % 4])
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("rotor speed (Hz)")
    ax.set_xlim(0, t[-1])
    if ylim is not None:
        ax.set_ylim(*ylim)


def fig_traj_real_vs_sampled(
    rigs=("dregon", "michaels"), name: str = "gen_traj_real_vs_sampled", row_h: float = 3.1
) -> None:
    import noise_lab as NL

    from experiments.rps_traj.data import RATE_HZ

    fig, axes = plt.subplots(len(rigs), 2, figsize=(11, row_h * len(rigs)))
    axes = np.atleast_2d(axes)
    for row, rig in enumerate(rigs):
        real = _real_window(rig, 30.0)
        traj = NL.trajectory("fitted", rig=rig, seed=11 + row, duration_s=30.0)
        # ``trajectory`` already carries the plot track at RPS_PLOT_SR = RATE_HZ
        samp = np.asarray(traj["rps"].data)
        lo = min(np.nanmin(real), samp.min())
        hi = max(np.nanmax(real), samp.max())
        pad = 0.06 * (hi - lo)
        ylim = (lo - pad, hi + pad)
        _plot_tracks(axes[row, 0], real, RATE_HZ, f"{rig} — real telemetry, 30 s airborne", ylim)
        _plot_tracks(axes[row, 1], samp, RATE_HZ, f"{rig} — fitted model, 30 s sample", ylim)
    axes[0, 1].legend(ncol=4, fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, name)


@_paper
def fig_traj_hyperprior(n: int = 4) -> None:
    """Four 30 s flights drawn from the drone hyperprior (``traj_rig="posterior"``),
    one per panel, the drawn hover level in the title; column width."""
    import noise_lab as NL

    from experiments.rps_traj.data import RATE_HZ

    fig, axes = plt.subplots(2, 2, figsize=(COLUMN_IN, 2.6), sharex=True, layout="constrained")
    for i, ax in enumerate(axes.ravel()[:n]):
        traj = NL.trajectory("fitted", rig="posterior", seed=101 + i, duration_s=30.0)
        rps = np.asarray(traj["rps"].data)
        hover = float(traj["meta"]["hover_rev_s"])
        t = np.arange(rps.shape[1]) / RATE_HZ
        for r in range(rps.shape[0]):
            ax.plot(t, rps[r], lw=0.45, color=COLOURS[r % 4], label=f"rotor {r + 1}")
        ax.set_xlim(0, t[-1])
        ax.set_title(f"drone {i + 1}, hover {hover:.0f} Hz", pad=2)
    for ax in axes[:, 0]:
        ax.set_ylabel("rotor speed (Hz)")
    for ax in axes[1]:
        ax.set_xlabel("time (s)")
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=4,
        handlelength=1.5,
    )
    save(fig, "gen_traj_hyperprior", paper=True)


#: Paper names of the trajectory rigs (Table tab:gen-traj-data).
TRAJ_LABELS = {
    "michaels": "MD2",
    "dregon": "DREGON",
    "neurobem_quad": "NeuroBEM",
    "pitcn_quad": "PI-TCN",
    "nanobench_cf21b": "NanoBench",
    "vid_m100": "VID",
    "blackbird_quad": "Blackbird",
}
#: Label offsets (points, horizontal alignment) where the default collides.
_TRAJ_LABEL_POS = {
    "dregon": ((6, 1), "left"),
    "michaels": ((-6, -7), "right"),
    "vid_m100": ((-6, 3), "right"),
    "blackbird_quad": ((-6, 3), "right"),
    "pitcn_quad": ((5, -9), "left"),
}


def _traj_level_and_fluctuation(params) -> tuple[float, float]:
    """``(log hover, log relative std)`` of one drone: the hover level in Hz and
    the stationary std of a rotor speed within a flight (RMS over rotors, speed
    modes only: no measurement process, no per-flight offset) over the hover."""
    hover = float(np.mean(params.mu))
    var = (params.mixing**2) @ params.mode_var
    return float(np.log(hover)), float(np.log(np.sqrt(np.mean(var)) / hover))


@_paper
def fig_traj_fits(n_hyper: int = 3000, n_reduced: int = 1500, seed: int = 0) -> None:
    """The paper's two-pane trajectory figure, ``gen_traj_fits``, at its print size
    (full text width) in the paper style.

    (a) 30 s of real airborne telemetry of DREGON and MD2 beside a 30 s draw from
    that drone's fitted law, y-axis shared per drone.

    (b) The seven fitted drones, the hyperprior and the constrained hyperprior on
    two axes: hover level and speed std relative to it, both log, each scaled by
    its hyperprior sd so that a step of one sd has the same length on both.

    Also writes ``results/rps_traj/fit_map.json``: the numbers of (b) and how
    much of the fits' spread in the 25 drone coordinates of the hyperprior
    (scale, trims, theta, speed modes; whitened by its std) the map keeps,
    against the best linear map (the top two principal components).
    """
    import json

    import noise_lab as NL
    from matplotlib.lines import Line2D
    from omegaconf import OmegaConf

    from data_processing.trajectory_model.posterior import IDX_SHAFT, rig_vector
    from data_processing.trajectory_model.source import build_from_config, load_bundle
    from experiments.rps_traj.data import RATE_HZ

    bundle = load_bundle("dload:rps-traj-fits")
    post = bundle.posterior
    assert post is not None
    rng = np.random.default_rng(seed)
    hyper = np.array([_traj_level_and_fluctuation(post.sample(rng)) for _ in range(n_hyper)])
    rps = OmegaConf.load(ROOT / "conf/online_mix/rp_priorv4.yaml").sources.noise[0].rps
    source = build_from_config(OmegaConf.to_container(rps, resolve=True))
    reduced = []
    for _ in range(n_reduced):
        source.flight(
            rng, 200.0
        )  # the stream's own draw, its rotor_sep / rps_max rejection included
        assert source.last_draw is not None
        reduced.append(_traj_level_and_fluctuation(source.last_draw.params))
    reduced = np.array(reduced)
    rigs = [r for r in TRAJ_LABELS if r in bundle.fits]
    fits = np.array([_traj_level_and_fluctuation(bundle.fits[r].params) for r in rigs])
    centre, sd = hyper.mean(axis=0), hyper.std(axis=0)

    # How much of the fits' spread the two axes keep (pairwise distances).
    drone = slice(0, IDX_SHAFT.stop)
    z = np.array([(rig_vector(bundle.fits[r].params) - post.mean) / post.std for r in rigs])[
        :, drone
    ]
    iu = np.triu_indices(len(rigs), 1)

    def dists(x: np.ndarray) -> np.ndarray:
        return np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))[iu]

    full = dists(z)
    zc = z - z.mean(axis=0)
    pcs = zc @ np.linalg.svd(zc, full_matrices=False)[2][:2].T
    kept = {
        name: {
            "r": round(float(np.corrcoef(dists(x), full)[0, 1]), 3),
            "squared_share": round(float(np.sum(dists(x) ** 2) / np.sum(full**2)), 3),
        }
        for name, x in (("map", (fits - centre) / sd), ("pca2", pcs))
    }

    fig = plt.figure(figsize=(TEXT_IN, 3.0), layout="constrained")
    left, right = fig.subfigures(1, 2, width_ratios=[1.3, 1.0], wspace=0.01)
    tracks = left.subplots(2, 2, sharex=True)
    for row, rig in enumerate(("dregon", "michaels")):
        real = _real_window(rig, 30.0)
        draw = NL.trajectory("fitted", rig=rig, seed=11 + row, duration_s=30.0)
        samp = np.asarray(draw["rps"].data)  # already on the RATE_HZ plot grid
        lo, hi = min(np.nanmin(real), samp.min()), max(np.nanmax(real), samp.max())
        pad = 0.06 * (hi - lo)
        for col, (rps, what) in enumerate(((real, "real flight"), (samp, "fitted law"))):
            axt = tracks[row, col]
            t = np.arange(rps.shape[1]) / RATE_HZ
            for r in range(rps.shape[0]):
                axt.plot(t, rps[r], lw=0.45, color=COLOURS[r % 4], label=f"rotor {r + 1}")
            axt.set_xlim(0, t[-1])
            axt.set_ylim(lo - pad, hi + pad)
            axt.set_title(f"{TRAJ_LABELS[rig]}, {what}", pad=2)
            axt.tick_params(labelleft=col == 0)
        tracks[row, 0].set_ylabel("rotor speed (Hz)")
    for axt in tracks[1]:
        axt.set_xlabel("time (s)")
    left.legend(
        *tracks[0, 0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=4,
        handlelength=1.5,
    )
    left.suptitle("(a)", x=0.01, ha="left", fontweight="bold")
    right.suptitle("(b)", x=0.01, ha="left", fontweight="bold")
    ax = right.subplots()
    ax.scatter(
        np.exp(hyper[:, 0]),
        100 * np.exp(hyper[:, 1]),
        s=1.5,
        color="0.55",
        alpha=0.25,
        lw=0,
        rasterized=True,
    )
    t = np.linspace(0.0, 2.0 * np.pi, 200)
    for k in (1, 2):
        ax.plot(
            np.exp(centre[0] + k * sd[0] * np.cos(t)),
            100 * np.exp(centre[1] + k * sd[1] * np.sin(t)),
            color="0.35",
            lw=0.8,
            ls="-" if k == 1 else "--",
        )
    ax.scatter(
        np.exp(reduced[:, 0]),
        100 * np.exp(reduced[:, 1]),
        s=1.5,
        color="#e6550d",
        alpha=0.35,
        lw=0,
        rasterized=True,
    )
    for (lx, ly), rig in zip(fits, rigs):
        ours = rig in ("dregon", "michaels")
        ax.scatter(
            np.exp(lx),
            100 * np.exp(ly),
            s=90 if ours else 28,
            marker="*" if ours else "o",
            color="#08519c" if ours else "black",
            zorder=7,
            edgecolors="white",
            linewidths=0.6,
        )
        offset, ha = _TRAJ_LABEL_POS.get(rig, ((5, 3), "left"))
        ax.annotate(
            TRAJ_LABELS[rig],
            (np.exp(lx), 100 * np.exp(ly)),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            zorder=6,
            bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.7},
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(25, 500)
    ax.set_ylim(1.0, 150.0)
    xt, yt = [30, 50, 100, 200, 400], [1, 2, 5, 10, 20, 50, 100]
    ax.set_xticks(xt, [str(v) for v in xt])
    ax.set_yticks(yt, [str(v) for v in yt])
    ax.minorticks_off()
    # one hyperprior sd is as long on y as on x (log10 units of each axis)
    ax.set_aspect(sd[0] / sd[1], adjustable="box")
    ax.set_xlabel("hover level (Hz)")
    ax.set_ylabel("speed std (% of hover)")
    handles = [
        Line2D([], [], ls="", marker="o", ms=3, color="0.55", label="hyperprior"),
        Line2D([], [], color="0.35", lw=0.8, label=r"its 1$\sigma$, 2$\sigma$"),
        Line2D([], [], ls="", marker="o", ms=3, color="#e6550d", label="constrained"),
        Line2D([], [], ls="", marker="*", ms=8, color="#08519c", label="fit, DREGON, MD2"),
        Line2D([], [], ls="", marker="o", ms=4, color="black", label="fit, other drones"),
    ]
    right.legend(
        handles=handles,
        loc="outside lower center",
        ncol=3,
        handletextpad=0.3,
        columnspacing=1.0,
    )
    save(fig, "gen_traj_fits", paper=True)

    def pct(x: np.ndarray, scale: float = 1.0) -> list[float]:
        return [round(float(scale * np.exp(v)), 2) for v in np.percentile(x, [5, 50, 95])]

    summary = {
        "axes": "hover = mean(mu) Hz; std = sqrt(mean over rotors of (A*A) @ mode_var) / hover",
        "fits": {
            TRAJ_LABELS[r]: {
                "hover_hz": round(float(np.exp(f[0])), 1),
                "std_pct": round(float(100 * np.exp(f[1])), 2),
            }
            for r, f in zip(rigs, fits)
        },
        "hyperprior": {
            "n": n_hyper,
            "hover_hz_p5_50_95": pct(hyper[:, 0]),
            "std_pct_p5_50_95": pct(hyper[:, 1], 100.0),
            "sd_log": [round(float(v), 3) for v in sd],
        },
        "constrained": {
            "n": n_reduced,
            "hover_hz_p5_50_95": pct(reduced[:, 0]),
            "std_pct_p5_50_95": pct(reduced[:, 1], 100.0),
            "source_stats": dict(source.stats),
        },
        "distance_kept_vs_25d_whitened": kept,
        "seed": seed,
    }
    out = ROOT / "results/rps_traj/fit_map.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


# ── noise ───────────────────────────────────────────────────────────────────


def _spec_db(audio: np.ndarray) -> np.ndarray:
    """``(frames, bins)`` periodogram in dB on the model's flight front end (NFFT
    2048, hop 512, periodic Hann, ``|rfft(x w)|^2 / sum w^2``)."""
    w = np.hanning(NFFT + 1)[:NFFT]
    n = (len(audio) - NFFT) // HOP + 1
    frames = np.stack([audio[i * HOP : i * HOP + NFFT] * w for i in range(n)])
    p = np.abs(np.fft.rfft(frames, axis=-1)) ** 2 / (w**2).sum()
    return 10.0 * np.log10(np.maximum(p, 1e-30))


def _show_spec(ax, db, title, *, fmax=8000.0, dur=None, vmax=None, dyn_range=45.0):
    """``db`` (frames, bins) as an image, frequency in kHz up to ``fmax``, clipped
    to ``[vmax - dyn_range, vmax]`` (``vmax`` None: its own 99.5th percentile)."""
    freqs = np.linspace(0, SR / 2, db.shape[1])
    keep = freqs <= fmax
    d = db[:, keep]
    top = float(np.percentile(d, 99.5)) if vmax is None else float(vmax)
    dur = dur if dur is not None else d.shape[0] * HOP / SR
    im = ax.imshow(
        d.T,
        origin="lower",
        aspect="auto",
        extent=(0.0, dur, 0.0, freqs[keep][-1] / 1000.0),
        cmap="magma",
        vmin=top - dyn_range,
        vmax=top,
        interpolation="nearest",
        rasterized=True,
    )
    ax.set_title(title, pad=2)
    ax.grid(False)
    return im


#: ``gen_noise_fits`` (a): the microphone (capsule 3: the DREGON gust fit's
#: least windy capsule, as in the recording), the render seed, the band shown
#: and the colour range (dB below the pair's 99.5th percentile). (b): the
#: reference speed every rotor turns at.
NOISESAMP_MIC = 3
NOISESAMP_SEED = 7
NOISESAMP_FMAX_HZ = 3000.0
NOISESAMP_DYN_DB = 50.0
PROFILE_F0_HZ = 80.0
#: (b)'s colours: aerodynamic and motor lines, pedestals, broadband floor, wind.
PROFILE_COLOURS = {
    "aero": "#1f5fbf",
    "motor": "#e6550d",
    "ped": "#c51b1b",
    "bb": "0.45",
    "wind": "#2a9d8f",
}
#: Offsets (Hz) at which each line's pedestal is sampled for the 1 Hz bin
#: integral: geometric down to below the narrowest width the skirt law allows.
_PROFILE_OFFSETS = np.geomspace(1e-11, 50.0, 300)


def _easy_bank_bases() -> dict:
    """The easy training bank's base fits, ``scripts/easy_bank_v4_build.BASES``:
    name -> (regime payload paths, trajectory rig, strength, seed tag)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "easy_bank_v4_build", ROOT / "scripts" / "easy_bank_v4_build.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BASES


def _render_regimes(paths: dict[str, str], rps: np.ndarray, seed: int) -> np.ndarray:
    """``(n_mics, T)`` the fit(s) of ``paths`` on the carriers ``rps``, as
    ``NoiseV2Pool.render`` renders an entry: a standby/cruise pair through
    ``render_noise_regimes``, a single fit through ``render_noise`` (its own
    wind included), absolute level, no render-time augmentation."""
    from data_processing.noise_model.render import render_noise, render_noise_regimes

    fits = {k: _load_json(ROOT / p) for k, p in paths.items()}
    n_mics = NOISESAMP_MIC + 1
    if set(fits) == {"standby", "cruise"}:
        return np.asarray(render_noise_regimes(fits, rps, sr=SR, n_mics=n_mics, seed=seed))
    (fit,) = fits.values()
    return np.asarray(render_noise(fit, rps, sr=SR, n_mics=n_mics, seed=seed))


def _fit_at_speed(fit: dict, f0: float, fmax: float) -> dict[str, np.ndarray]:
    """A ``noise-v4-fit/3`` payload with every rotor at ``f0`` rev/s, as
    ``notebooks/prior_rigs_lab.py`` ``plot_rig_profiles`` draws it, simplified:
    POWER IN 1 Hz BINS (centred on the integers, so on every line at an integer
    ``f0``) in the render's units, microphone mean, lines and pedestals the
    MEAN over the rotors. A broadband density reads the same per 1 Hz; a
    needle reads its power; a pedestal its skirt integrated over the bin, so
    a pedestal much narrower than 1 Hz reads at most its own power instead of
    a peak density far above its line.

    ``f_aero, needle_aero``: each aerodynamic line's core (power
    ``e^-(s2a + s2p)`` of the line); ``f_motor, needle_motor``: the motor lines
    (no pedestal); ``grid, ped``: bin centres and the summed amplitude + phase
    pedestal skirts of all lines; ``bb``: the broadband floor (its speed law and
    static share, through the array response) without the wind, the whole
    drone's; ``wind``: the mean of the fitted wind alone (zero without one)."""
    from data_processing.noise_model import v4 as V4
    from experiments.noise_model.prior_v4 import pedestal_skirt_density

    par = V4.V4Params.from_payload(fit)
    s = par.speed(np.full(par.n_rotors, float(f0)))
    p_aero, p_motor = par.line_power(s)  # (R, K), (R, n_m)
    k = np.arange(1, par.k_max + 1, dtype=np.float64)
    km = np.asarray(par.motor_orders, dtype=np.float64)
    # E_rm at each line, mean over the mics, then the mean over the rotors
    aero = (10.0 ** (par.mic.db(k * f0) / 10.0)).mean(axis=1) * p_aero
    motor = (10.0 ** (par.mic.db(km * f0) / 10.0)).mean(axis=1) * p_motor
    aero, motor = aero.mean(axis=0), motor.mean(axis=0)
    ped = par.pedestal
    shape = (ped.sigma2_amp, ped.gamma_amp_hz, ped.sigma2_phase, ped.gamma_phase_hz)
    tot = ped.sigma2_amp + ped.sigma2_phase
    in_band = k * f0 < fmax
    # the skirt density (of a UNIT-power line: it integrates to 1 - e^-(s2a +
    # s2p) itself) on the bin edges plus a geometric grid around every line in
    # the band, times the line power, integrated (trapezoid), differenced
    edges = np.arange(0.5, fmax, 1.0)
    near = np.concatenate([-_PROFILE_OFFSETS[::-1], [0.0], _PROFILE_OFFSETS])
    x = np.unique(np.concatenate([edges, (k[in_band, None] * f0 + near).ravel()]))
    x = x[(x >= edges[0]) & (x <= edges[-1])]
    dens = np.zeros(x.size)
    for a in range(0, x.size, 4096):
        s2a, ga, s2p, gp = (v[None, :] for v in shape)
        d = pedestal_skirt_density(
            s2a, ga, s2p, gp, x[a : a + 4096, None] - k[None, :] * f0, tol=1e-6
        )
        dens[a : a + 4096] = d @ aero
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(x))])
    grid = edges[:-1] + 0.5
    bb = 2.0 * V4.broadband_psd(par, grid, s, wind=False).mean(axis=0) / par.sr
    wind = 2.0 * V4.broadband_psd(par, grid, s).mean(axis=0) / par.sr - bb
    keep_m = km * f0 < fmax
    return dict(
        f_aero=k[in_band] * f0,
        needle_aero=(aero * np.exp(-tot))[in_band],
        f_motor=km[keep_m] * f0,
        needle_motor=motor[keep_m],
        grid=grid,
        ped=np.diff(cum[np.searchsorted(x, edges)]),
        bb=bb,
        wind=np.maximum(wind, 0.0),
    )


def _db(x: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(x, 1e-30))


@_paper
def fig_noise_fits() -> None:
    """The paper's noise-fit figure, ``gen_noise_fits``, full text width.

    (a) Real DREGON and MD2 windows beside the noise-model-v4 fits of the easy
    training bank rendered on the SAME speed labels: DREGON the bench fit c7
    with its fitted gust wind, MD2 the FLY125 standby/cruise pair composed per
    regime. Same microphone, same STFT, one colour range per pair.

    (b) The same fits with every rotor at :data:`PROFILE_F0_HZ` (MD2: the
    cruise fit alone, the blend weight being one there), :func:`_fit_at_speed`.
    """
    from matplotlib.lines import Line2D

    from experiments.noise_model import supports as S

    bases = _easy_bank_bases()
    scored = dict(S.DREGON_SCORED)["free-flight_nosource_room2"]
    rows = (
        (
            "DREGON",
            # dregon-flight-wide row j = 1 of free-flight (the gust wind's pool)
            S.flight_dregon("free-flight_nosource_room2", S.dregon_floor_start(scored) + 8.0, 8.0),
            bases["dregon"][0],
            "fit + wind",
        ),
        (
            "MD2",
            # FLY125's ramp context window: standby, spool-up, cruise
            S.flight_michaels("FLY125", S.MICHAELS_RAMP_STARTS[0], 8.0),
            bases["md2"][0],
            "standby/cruise fits",
        ),
    )
    fig = plt.figure(figsize=(TEXT_IN, 3.0), layout="constrained")
    left, right = fig.subfigures(1, 2, width_ratios=[1.2, 1.0], wspace=0.01)
    spec_axes = left.subplots(2, 2, sharex=True, sharey=True)
    prof_axes = right.subplots(2, 1, sharex=True)
    for row, (drone, spec, paths, fit_tag) in enumerate(rows):
        clip = S.flight_clip(spec)
        rps = np.asarray(clip.rps, dtype=np.float64)
        real = np.asarray(clip.audio[NOISESAMP_MIC], dtype=np.float64)
        samp = _render_regimes(paths, rps, NOISESAMP_SEED)[NOISESAMP_MIC]
        dbs = (_spec_db(real), _spec_db(samp))
        keep = np.linspace(0, SR / 2, dbs[0].shape[1]) <= NOISESAMP_FMAX_HZ
        top = float(np.percentile(np.concatenate([d[:, keep].ravel() for d in dbs]), 99.5))
        dur = rps.shape[1] / SR
        for col, (db, what) in enumerate(zip(dbs, ("recording", fit_tag))):
            im = _show_spec(
                spec_axes[row, col],
                db,
                f"{drone}, {what}",
                fmax=NOISESAMP_FMAX_HZ,
                dur=dur,
                vmax=top,
                dyn_range=NOISESAMP_DYN_DB,
            )
        cb = left.colorbar(im, ax=spec_axes[row], pad=0.02, aspect=12, fraction=0.04)
        cb.set_label("dB", labelpad=1)
        cb.ax.tick_params(length=1.5, pad=1)
        spec_axes[row, 0].set_ylabel("frequency (kHz)")
        print(
            f"  {drone}: {spec.text}, mic {NOISESAMP_MIC}, seed {NOISESAMP_SEED}, "
            f"rotor label {rps.min():.1f}-{rps.max():.1f} Hz (mean {rps.mean():.1f}), "
            f"colour range {top - NOISESAMP_DYN_DB:.1f}..{top:.1f} dB, payloads {paths}",
            flush=True,
        )

        # (b): at 80 Hz the regime blend weight is one, so MD2 is its cruise fit
        path = paths.get("single") or paths["cruise"]
        prof = _fit_at_speed(_load_json(ROOT / path), PROFILE_F0_HZ, NOISESAMP_FMAX_HZ)
        ax = prof_axes[row]
        bb_db, ped_db, wind_db = _db(prof["bb"]), _db(prof["ped"]), _db(prof["wind"])
        tops = np.concatenate([_db(prof["needle_aero"]), _db(prof["needle_motor"])])
        hi = float(max(tops.max(), ped_db.max())) + 5.0
        lo = hi - 70.0  # DREGON's bench floor dips far below and leaves the axis
        grid_khz = prof["grid"] / 1000.0
        ax.plot(grid_khz, bb_db, color=PROFILE_COLOURS["bb"], lw=0.9, zorder=2)
        if prof["wind"].max() > 0.0:
            ax.plot(grid_khz, wind_db, color=PROFILE_COLOURS["wind"], lw=0.9, zorder=2)
        ax.plot(grid_khz, ped_db, color=PROFILE_COLOURS["ped"], lw=0.6, zorder=3)
        for fam in ("aero", "motor"):
            ax.vlines(
                prof[f"f_{fam}"] / 1000.0,
                lo,
                _db(prof[f"needle_{fam}"]),
                color=PROFILE_COLOURS[fam],
                lw=0.7,
                zorder=4,
            )
        ax.set_xlim(0.0, NOISESAMP_FMAX_HZ / 1000.0)
        ax.set_ylim(lo, hi)
        ax.set_ylabel("dB per Hz")
        ax.set_title(f"{drone}, fit, every rotor at {PROFILE_F0_HZ:.0f} Hz", pad=2)
        probe = (100.0, 500.0, 1000.0, 2000.0, 2999.0)

        def at(x: np.ndarray, fs=probe, g=prof["grid"]) -> list[float]:
            return [round(float(np.interp(f, g, x)), 1) for f in fs]

        print(
            f"  {drone} at {PROFILE_F0_HZ:.0f} Hz ({path}): aero needles k 1-6 "
            f"{_db(prof['needle_aero'][:6]).round(1)}, max {tops.max():.1f} at "
            f"{np.concatenate([prof['f_aero'], prof['f_motor']])[tops.argmax()]:.0f} Hz; "
            f"motor {_db(prof['needle_motor']).round(1)} at {prof['f_motor']} Hz; "
            f"broadband floor at {probe} Hz {at(bb_db)}, wind {at(wind_db)}; "
            f"pedestals in the line bins k 1-12 "
            f"{at(ped_db, fs=prof['f_aero'][:12])}, max {ped_db.max():.1f}; "
            f"y range {lo:.1f}..{hi:.1f}",
            flush=True,
        )
    for ax in spec_axes[1]:
        ax.set_xlabel("time (s)")
    prof_axes[1].set_xlabel("frequency (kHz)")
    prof_axes[0].legend(
        handles=[
            Line2D([], [], color=PROFILE_COLOURS["aero"], lw=0.9, label="aerodynamic lines"),
            Line2D([], [], color=PROFILE_COLOURS["motor"], lw=0.9, label="motor lines"),
            Line2D([], [], color=PROFILE_COLOURS["ped"], lw=0.9, label="line pedestals"),
            Line2D([], [], color=PROFILE_COLOURS["bb"], lw=0.9, label="broadband floor"),
            Line2D([], [], color=PROFILE_COLOURS["wind"], lw=0.9, label="wind (DREGON)"),
        ],
        loc="upper right",
        ncol=2,
        handlelength=1.2,
        columnspacing=0.8,
    )
    left.suptitle("(a)", x=0.01, ha="left", fontweight="bold")
    right.suptitle("(b)", x=0.01, ha="left", fontweight="bold")
    save(fig, "gen_noise_fits", paper=True)


#: ``gen_noise_prior``: the rig-distance map (``scripts/rig_distance_map.py``),
#: the validation sectors it maps, and per sector of (a) its label, the v1
#: draw (or the payload statistic) that defines it (``None``: the medoid), the
#: marker of its drone in (b) and its colour (the map's).
PRIOR_MAP = ROOT / "results/rig_distance_v4/distances.npz"
PRIOR_SECTORS_SRC = "dload:validation-sectors-v4"
PRIOR_SECTORS = (
    ("uniform", "uniform", None, "o", "#1f77b4"),
    ("weak_comb", "weak comb", "k2_over_floor_db", "s", "#2ca02c"),
    ("strong_motor", "strong motor", "motor_boost_db", "^", "#9467bd"),
    ("wide_pedestal", "wide pedestal", "pedestal_width_hz", "v", "#8c564b"),
)
#: The easy neighbourhoods and the fits: map group, legend label, colour.
PRIOR_NEAR = (
    ("near_dregon", "easy neighbourhood, DREGON", "#d62728"),
    ("near_md2", "easy neighbourhood, MD2", "#ff7f0e"),
)
#: The seed of the one training trajectory every drone of (a) flies.
PRIOR_TRAJ_SEED = 0
PRIOR_DUR_S = 8.0
#: Where (points) each drone's tag sits beside its marker in (b).
PRIOR_TAG_OFFSETS = ((-6, -6), (6, 6), (7, 0), (-6, 6))


def _sector_stat(job: tuple[int, int, int, str]) -> float:
    """The statistic ``key`` of the prior draw ``(seed, worker, attempt)``,
    redrawn as ``scripts/validation_sectors_v4_build.py`` drew it: a v1 law
    draw, or the payload's :func:`.sectors_v4.pedestal_width_hz`."""
    import torch

    from experiments.noise_model.prior_v4 import sample_prior_v4
    from experiments.noise_model.sectors_v4 import pedestal_width_hz

    torch.set_num_threads(2)
    seed, worker, attempt, key = job
    pl = sample_prior_v4(seed, worker, attempt)
    drawn = pl.pop("_prior_v1_drawn")
    return float(pedestal_width_hz(pl) if key == "pedestal_width_hz" else drawn[key])


def _prior_picks(z) -> list[tuple[int, int, float]]:
    """Per sector of :data:`PRIOR_SECTORS`, ``(member, map row, value)``: an
    axis sector's member at its median statistic (the lower middle member of
    the 32), the uniform sector's ``d_total`` medoid (value: its mean
    distance to the other members)."""
    import concurrent.futures as cf
    import json
    import multiprocessing as mp

    from data_processing.streams import resolve_source

    root = Path(resolve_source(PRIOR_SECTORS_SRC))
    seed = int(json.loads((root / "report.json").read_text())["seed"])
    names = list(z["names"])
    jobs: list[tuple[int, int, int, str]] = []
    for sec, _, key, _, _ in PRIOR_SECTORS:
        if key is not None:
            with np.load(root / f"{sec}.npz") as b:
                jobs += [
                    (seed, int(w), int(a), key)
                    for w, a in zip(b["draw_worker"], b["draw_attempt"], strict=True)
                ]
    with cf.ProcessPoolExecutor(6, mp_context=mp.get_context("spawn")) as ex:
        vals = iter(list(ex.map(_sector_stat, jobs)))
    out = []
    for sec, _, key, _, _ in PRIOR_SECTORS:
        rows = np.flatnonzero(z["groups"] == sec)
        if key is None:
            d = z["d_total"][np.ix_(rows, rows)].sum(axis=1) / (len(rows) - 1)
            i = int(np.argmin(d))
            value = float(d[i])
        else:
            x = np.asarray([next(vals) for _ in rows])
            i = int(np.argsort(x)[(len(x) - 1) // 2])
            value = float(x[i])
        out.append((i, names.index(f"{sec}_{i}"), value))
    return out


def _prior_rps() -> np.ndarray:
    """``(R, T)`` one :data:`PRIOR_DUR_S` window as the wide-prior training
    stream flies its bank (``conf/online_mix/rp_priorv4.yaml``'s ``rps``
    block, ``NoiseV2Pool.sample_rps``), drawn from :data:`PRIOR_TRAJ_SEED`."""
    import yaml

    from data_processing.noise_v2_pool import NoiseV2Pool

    cfg = yaml.safe_load((ROOT / "conf/online_mix/rp_priorv4.yaml").read_text())
    src = dict(cfg["sources"]["noise"][0], n_mics=NOISESAMP_MIC + 1)
    pool = NoiseV2Pool.from_config(src, duration_s=PRIOR_DUR_S, sample_rate=SR)
    rps = pool.sample_rps(np.random.default_rng(PRIOR_TRAJ_SEED), PRIOR_DUR_S)
    return np.asarray(rps, dtype=np.float64)


@_paper
def fig_noise_prior() -> None:
    """The paper's prior figure, ``gen_noise_prior``, full text width.

    (a) One drone per validation sector of the wide prior
    (:data:`PRIOR_SECTORS`, :func:`_prior_picks`), rotor noise only, all on the
    SAME training trajectory (:func:`_prior_rps`) with the same render seed,
    microphone and STFT as ``gen_noise_fits``; each render at unit RMS, one
    colour range for all four. (b) The 2-D metric MDS of the spectral rig
    distance ``d_total`` (``scripts/rig_distance_map.py``): the training bank,
    the easy neighbourhoods of the two fits, the fits, and the drones of (a).
    """
    from data_processing.noise_model.prior_bank_v4 import PriorBankV4
    from data_processing.noise_model.render import render_noise
    from data_processing.streams import resolve_source

    z = dict(np.load(PRIOR_MAP))
    picks = _prior_picks(z)
    rps = _prior_rps()
    root = Path(resolve_source(PRIOR_SECTORS_SRC))
    dbs = []
    for (sec, *_), (i, _, _) in zip(PRIOR_SECTORS, picks, strict=True):
        payload = PriorBankV4(root / f"{sec}.npz").payload(i)
        audio = np.asarray(
            render_noise(payload, rps, sr=SR, n_mics=NOISESAMP_MIC + 1, seed=NOISESAMP_SEED),
            dtype=np.float64,
        )[NOISESAMP_MIC]
        dbs.append(_spec_db(audio / np.sqrt(np.mean(audio**2))))
    keep = np.linspace(0, SR / 2, dbs[0].shape[1]) <= NOISESAMP_FMAX_HZ
    top = float(np.percentile(np.concatenate([d[:, keep].ravel() for d in dbs]), 99.5))

    fig = plt.figure(figsize=(TEXT_IN, 3.0), layout="constrained")
    left, right = fig.subfigures(1, 2, width_ratios=[1.2, 1.0], wspace=0.01)
    spec_axes = left.subplots(2, 2, sharex=True, sharey=True)
    for n, (ax, db, (_, label, *_)) in enumerate(
        zip(spec_axes.ravel(), dbs, PRIOR_SECTORS, strict=True), start=1
    ):
        im = _show_spec(
            ax,
            db,
            f"{n}: {label}",
            fmax=NOISESAMP_FMAX_HZ,
            dur=rps.shape[1] / SR,
            vmax=top,
            dyn_range=NOISESAMP_DYN_DB,
        )
    cb = left.colorbar(im, ax=spec_axes, pad=0.02, aspect=25, fraction=0.04)
    cb.set_label("dB re RMS", labelpad=1)
    cb.ax.tick_params(length=1.5, pad=1)
    for ax in spec_axes[:, 0]:
        ax.set_ylabel("frequency (kHz)")
    for ax in spec_axes[1]:
        ax.set_xlabel("time (s)")
    _prior_map(right.subplots(), z, picks)
    left.suptitle("(a)", x=0.01, ha="left", fontweight="bold")
    right.suptitle("(b)", x=0.01, ha="left", fontweight="bold")
    save(fig, "gen_noise_prior", paper=True)
    _prior_report(z, picks, rps, top)


def _prior_map(ax, z: dict, picks: list[tuple[int, int, float]]) -> None:
    """(b) of ``gen_noise_prior``: ``xy_total`` with equal aspect."""
    from scipy.spatial import ConvexHull

    xy, g = z["xy_total"], z["groups"]
    bank = xy[g == "bank"]
    ax.scatter(*bank.T, s=1.5, c="0.72", lw=0, rasterized=True, label="training bank")
    for grp, label, colour in PRIOR_NEAR:
        pts = xy[g == grp]
        loop = ConvexHull(pts).vertices
        loop = np.append(loop, loop[0])
        ax.fill(*pts[loop].T, color=colour, alpha=0.12, lw=0)
        ax.plot(*pts[loop].T, color=colour, lw=0.7)
        ax.scatter(*pts.T, s=4, c=colour, lw=0, label=label)
    for grp, label, marker, colour, size in (
        ("fit_dregon", "DREGON fit", "*", "#d62728", 90),
        ("fit_md2", "MD2 fit", "D", "#ff7f0e", 28),
    ):
        ax.scatter(
            *xy[g == grp][0],
            s=size,
            marker=marker,
            c=colour,
            edgecolors="k",
            lw=0.6,
            zorder=5,
            label=label,
        )
    for n, ((_, _, _, marker, colour), (_, row, _)) in enumerate(
        zip(PRIOR_SECTORS, picks, strict=True), start=1
    ):
        ax.scatter(*xy[row], s=24, marker=marker, c=colour, edgecolors="k", lw=0.6, zorder=6)
        ax.annotate(
            str(n),
            xy[row],
            xytext=PRIOR_TAG_OFFSETS[n - 1],
            textcoords="offset points",
            ha="center",
            va="center",
            fontweight="bold",
            zorder=7,
        )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("MDS 1 (dB)")
    ax.set_ylabel("MDS 2 (dB)")
    leg = ax.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.3)
    for h in leg.legend_handles[:3]:  # the bank's and the neighbourhoods' dots
        cast(Any, h).set_sizes([10.0])
    print(f"  map: {len(bank)} bank rigs drawn", flush=True)


def _prior_report(
    z: dict, picks: list[tuple[int, int, float]], rps: np.ndarray, top: float
) -> None:
    """Print what ``gen_noise_prior`` shows: the drones, the trajectory, the
    colour range, the map's stress and the FULL ``d_total`` to each fit."""
    g, d = z["groups"], z["d_total"]
    fits = {f: int(np.flatnonzero(g == f"fit_{f}")[0]) for f in ("dregon", "md2")}
    print(
        f"  trajectory: rp_priorv4.yaml rps block, seed {PRIOR_TRAJ_SEED}, {PRIOR_DUR_S:g} s, "
        f"rotors {rps.min():.1f}-{rps.max():.1f} Hz (mean {rps.mean():.1f}); render seed "
        f"{NOISESAMP_SEED}, mic {NOISESAMP_MIC}, unit RMS, colour range "
        f"{top - NOISESAMP_DYN_DB:.1f}..{top:.1f} dB; MDS stress-1 {float(z['stress'][0]):.3f}; "
        f"fit-to-fit d_total {d[fits['dregon'], fits['md2']]:.2f} dB",
        flush=True,
    )
    for n, ((sec, _, key, _, _), (i, row, value)) in enumerate(
        zip(PRIOR_SECTORS, picks, strict=True), start=1
    ):
        what = f"{key} {value:.3f}" if key else f"medoid, mean d_total {value:.2f} dB"
        to = ", ".join(f"{f} {d[row, j]:.2f}" for f, j in fits.items())
        print(
            f"  {n}: {sec}_{i} ({what}); map ({z['xy_total'][row][0]:.1f}, "
            f"{z['xy_total'][row][1]:.1f}); d_total to fits: {to} dB",
            flush=True,
        )
    for f, j in fits.items():
        for grp in ("bank", "near_dregon", "near_md2"):
            x = d[j, g == grp]
            print(
                f"  d_total {grp} -> {f}: min {x.min():.2f} median {np.median(x):.2f}", flush=True
            )


def _load_json(path: Path) -> dict:
    import json

    with open(path) as fh:
        return json.load(fh)


# ── tables ──────────────────────────────────────────────────────────────────


def _booktabs(
    colspec: str,
    header: list[str] | list[list[str] | str],
    rows: list[list[str] | str],
    caption: str,
    label: str,
    *,
    star: bool = False,
    size: str = "small",
    tabcolsep: str | None = None,
    source: str | None = None,
):
    # A row is either a list of already-escaped cells or a raw LaTeX line
    # (\midrule, \cmidrule, a hand-written header row) emitted verbatim. A header
    # holding any cell list is a sequence of such lines; a flat list of strings is
    # one header row.
    line = lambda r: "    " + (r if isinstance(r, str) else " & ".join(r) + " \\\\")  # noqa: E731
    head = header if any(not isinstance(h, str) for h in header) else [header]
    env = "table*" if star else "table"
    out = ["% " + ln for ln in source.splitlines()] if source else []
    out += [f"\\begin{{{env}}}[tb]", "  \\centering", f"  \\{size}"]
    if tabcolsep is not None:
        out.append(f"  \\setlength{{\\tabcolsep}}{{{tabcolsep}}}")
    out += [f"  \\begin{{tabular}}{{{colspec}}}", "    \\toprule"]
    out += [line(h) for h in head]
    out.append("    \\midrule")
    out += [line(r) for r in rows]
    out += [
        "    \\bottomrule",
        "  \\end{tabular}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"\\end{{{env}}}",
        "",
    ]
    return "\n".join(out)


def write_table(name: str, colspec, header, rows, caption, label, **kw) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / f"{name}.tex").write_text(_booktabs(colspec, header, rows, caption, label, **kw))
    print(f"wrote {name}.tex", flush=True)


def tables() -> None:
    write_table(
        "gen_traj_model_params",
        "llll",
        ["symbol", "meaning", "shape", "range / prior"],
        [
            ["$\\mu$", "per-rotor hover level (Hz)", "4", "estimated from airborne samples"],
            [
                "$\\theta$",
                "rotation of the (collective, yaw) plane (rad)",
                "1",
                "$(-\\pi/4,\\pi/4]$ canonical",
            ],
            [
                "$\\tau_{\\mathrm{slow}}$",
                "slow OU time constant per mode (s)",
                "4",
                "$(\\tau_c,10]$, LN(2.0\\,s, 1.5)",
            ],
            [
                "$\\sigma_{\\mathrm{slow}}$",
                "slow OU stationary sd per mode (Hz)",
                "4",
                "LN(1.0, 3.0)",
            ],
            ["$f_0$", "oscillator centre frequency per mode (Hz)", "4", "$(0.05,20)$"],
            ["$\\zeta$", "oscillator damping ratio per mode", "4", "$(0.2,3.0)$"],
            [
                "$\\sigma_{\\mathrm{osc}}$",
                "oscillator stationary sd per mode (Hz)",
                "4",
                "LN(1.0, 3.0)",
            ],
            [
                "$\\tau_e$",
                "measurement OU time constant (s)",
                "1",
                "$(10^{-3},1)$, LN(0.05\\,s, 2.0)",
            ],
            ["$\\sigma_e$", "measurement OU sd, per rotor (Hz)", "1", "LN(0.1, 3.0)"],
            [
                "$s_c$",
                "sd of the per-flight level common to the rotors (Hz)",
                "1",
                "moment estimate",
            ],
            ["$s_r$", "sd of the per-flight per-rotor level (Hz)", "4", "moment estimate"],
        ],
        "Parameters of the rotor-speed trajectory model (32 in total; 23 enter the "
        "likelihood, $\\mu$, $s_c$ and $s_r$ are estimated from the airborne samples). "
        "$\\tau_c = 1/(2\\pi f_0)$ orders the two shaft components without a label-switching gauge. "
        "LN$(m,s)$: log-normal with median $m$ and log-sd $s$.",
        "tab:gen-traj-model-params",
        star=True,
    )

    write_table(
        "gen_noise_model_params",
        "lllp{0.30\\linewidth}",
        ["symbol", "meaning", "shape", "prior"],
        [
            [
                "$\\sigma_\\nu$",
                "shaft OU stationary speed-error sd (rad/s)",
                "per rotor",
                "HN(0.6)",
            ],
            [
                "$\\lambda$",
                "shaft OU rate (s$^{-1}$)",
                "per rotor",
                "pinned in flight (0.5); LN(2, 1.0) on the bench",
            ],
            [
                "$\\gamma_{ik}$",
                "Lorentzian half-width of line $(i,k)$ (Hz)",
                "rotor $\\times$ order",
                "$\\gamma_{ik}/(\\gamma_0 k) \\sim$ HN(3), $\\gamma_0 = 0.01$\\,Hz",
            ],
            [
                "$p_{ik}$",
                "mean line power of order $k$ at $r_{\\mathrm{ref}}$ (dB), the comb profile",
                "rotor $\\times$ order",
                "N($\\hat p_{ik}$, 10) for every order, $\\hat p_{ik}$ the pooled level at $k r_i$",
            ],
            ["$a$", "comb speed exponent", "1", "N(2, 1); pinned at 2 on a short span"],
            [
                "$d_i(\\ell)$",
                "rotor-common line wander in block $\\ell$ (dB)",
                "rotor $\\times$ block",
                "OU($\\sigma_d$, $\\tau_d$), measured and fixed",
            ],
            [
                "$v_{ik}(\\ell)$",
                "per-line wander in block $\\ell$ (dB)",
                "line $\\times$ block",
                "OU($\\sigma_v$, $\\tau_v$), measured and fixed",
            ],
            ["$b$", "floor speed exponent", "1", "LN(2, 0.5); same span pin"],
            ["$s$", "static fraction of the floor", "1", "LN($2.5\\times10^{-3}$, 1.0)"],
            ["$\\mu$", "floor level (dB)", "1", "measured band median, fixed"],
            [
                "$z$",
                "GP control points of the floor shape",
                "14",
                "N(0, 1) each, scale $\\sigma_B$ measured and fixed",
            ],
            [
                "$u(\\ell)$",
                "floor-level drift in block $\\ell$ (dB)",
                "block",
                "OU($\\sigma_u$, $\\tau_u$), measured and fixed",
            ],
            [
                "$u_j(\\ell)$",
                "floor-colour drift at control point $j$ (dB)",
                "14 $\\times$ block",
                "OU($\\sigma_{u_j}$, $\\tau_{u_j}$), measured and fixed",
            ],
            [
                "$w_m$",
                "static wind level below 500\\,Hz (dB), DREGON only",
                "mic",
                "N(measured low-band excess, 6)",
            ],
            [
                "$r_i$",
                "carrier (Hz)",
                "per rotor",
                "given: window-refined or telemetry label",
            ],
        ],
        "Parameters of the rotor-noise model (v3) and their priors. HN$(\\sigma)$: half-normal "
        "of scale $\\sigma$ in the parameter's own units; LN$(m,s)$: log-normal with median $m$ "
        "and log-sd $s$; N$(\\mu,\\sigma)$: normal; OU$(\\sigma,\\tau)$: stationary "
        "Ornstein--Uhlenbeck across blocks of 0.5\\,s with sd $\\sigma$ and correlation time "
        "$\\tau$, the values of Table~\\ref{tab:wander}. There is no microphone parameter: "
        "the channels are normalised in the data.",
        "tab:gen-noise-model-params",
        star=True,
        source="source: docs/explainers/noise-model-v3-wander.qmd, section 2.6; "
        "src/experiments/noise_model/model.py (PriorsV3);\n"
        "results/noise_v3/wander/{dregon,michaels}.json (block_s 0.5)",
    )

    write_table(
        "gen_noise_data",
        "lllrl",
        ["recording set", "class", "rotors / mics", "duration (s)", "role in the fit"],
        [
            [
                "DREGON single-motor bench, 50--90\\,\\%",
                "bench",
                "1 / 8",
                "12--34 per window",
                "identifies $\\sigma_\\nu$, $\\lambda$, $\\gamma_{rk}$",
            ],
            [
                "DREGON \\texttt{motor\\_allMotors\\_70} bench",
                "bench",
                "4 / 8",
                "$\\sim$30",
                "four-rotor control",
            ],
            [
                "DREGON room 2 free flight / hover / rectangle / spinning",
                "flight",
                "4 / 8",
                "4 per window, pooled",
                "floor and flight profile (R5 fit)",
            ],
            [
                "Michael's FLY125 cruise",
                "flight",
                "4 / 1",
                "4 per window, pooled",
                "cruise comb and floor (R3 fit)",
            ],
            [
                "Michael's FLY125 standby",
                "flight",
                "4 / 1",
                "4 per window, pooled",
                "standby regime of the same rig",
            ],
        ],
        "Recordings behind the shipped noise fits. Flight supports are periodic-Hann frames, "
        "$N=2048$, hop 512 at 16\\,kHz, pooled over windows; bench supports are one stationary "
        "periodogram each. The fit band is $30\\,\\mathrm{Hz} \\le f \\le 7900\\,\\mathrm{Hz}$.",
        "tab:gen-noise-data",
        star=True,
        size="footnotesize",
        tabcolsep="4pt",
    )

    ref = "\\textit{real reference}"
    # rows: real_overall, overall, zero, standby, ramp, cruise, spread standby/ramp/cruise
    regime_rows: list[tuple[str, str, str, list[str]]] = [
        (
            "SCv2",
            f"\\texttt{{real\\_r4\\_scv2\\_unified}} ({ref})",
            "\\textbf{2.99}",
            ["3.11", "2.44", "3.20", "8.17", "2.95", "9.74", "6.58", "10.94"],
        ),
        (
            "HPPNet L2",
            f"\\texttt{{hppnet\\_l2\\_r2\\_s0}} ({ref})",
            "\\textbf{2.27}",
            ["2.22", "1.05", "3.11", "12.51", "1.74", "8.81", "6.76", "13.53"],
        ),
        (
            "SCv2",
            "\\texttt{nv2\\_easy\\_scv2} (synthetic only)",
            "7.94",
            ["7.99", "10.64", "11.05", "11.12", "6.87", "5.71", "6.42", "16.76"],
        ),
        (
            "SCv2",
            "\\texttt{nv2\\_hard\\_scv2} (synthetic only)",
            "7.06",
            ["7.05", "1.25", "6.16", "15.81", "7.76", "6.27", "8.14", "18.56"],
        ),
        (
            "SCv2",
            "\\texttt{nv2\\_mixed\\_scv2} (mixed)",
            "2.53",
            ["2.54", "1.78", "2.25", "7.00", "2.49", "9.76", "7.82", "12.77"],
        ),
        (
            "SCv2",
            "\\texttt{nv2\\_easy\\_ft\\_scv2} (easy $\\to$ real)",
            "2.73",
            ["2.73", "2.23", "2.36", "8.16", "2.60", "10.16", "7.37", "12.75"],
        ),
        (
            "SCv2",
            "\\texttt{nv2\\_hard\\_ft\\_scv2} (hard $\\to$ real)",
            "\\textbf{2.23}",
            ["2.23", "0.93", "1.44", "6.71", "2.35", "9.22", "6.63", "11.23"],
        ),
        (
            "HPPNet L2",
            "\\texttt{nv2\\_easy\\_hppnet\\_l2} (synthetic only)",
            "6.46",
            ["6.45", "3.58", "5.04", "26.50", "6.14", "7.54", "9.95", "15.46"],
        ),
        (
            "HPPNet L2",
            "\\texttt{nv2\\_hard\\_hppnet\\_l2} (synthetic only)",
            "6.47",
            ["6.47", "1.63", "4.25", "30.07", "6.44", "5.37", "6.11", "17.25"],
        ),
        (
            "HPPNet L2",
            "\\texttt{nv2\\_mixed\\_hppnet\\_l2} (mixed)",
            "2.25",
            ["2.25", "0.29", "2.28", "9.09", "2.23", "8.31", "7.83", "14.44"],
        ),
        (
            "HPPNet L2",
            "\\texttt{nv2\\_easy\\_ft\\_hppnet\\_l2} (easy $\\to$ real)",
            "\\textbf{2.11}",
            ["2.11", "1.50", "2.84", "8.25", "1.77", "9.08", "7.29", "13.06"],
        ),
        (
            # unmeasured: the saved checkpoint is round 4, the run's raw 1.81 at
            # round 1 was never written, so this row is not a result.
            "HPPNet L2",
            "\\texttt{nv2\\_hard\\_ft\\_hppnet\\_l2} (hard $\\to$ real)$^{*}$",
            "4.38",
            ["4.42", "20.91", "2.83", "9.50", "1.51", "10.57", "7.42", "13.84"],
        ),
    ]
    write_table(
        "gen_results_by_dataset_regime",
        "llrrrrrrrrr",
        [
            "& & & & \\multicolumn{4}{c}{per-frame PIT MAE} "
            "& \\multicolumn{3}{c}{output spread} \\\\",
            "\\cmidrule(lr){5-8}\\cmidrule(lr){9-11}",
            [
                "trunk",
                "arm",
                "real\\_overall",
                "overall",
                "zero",
                "standby",
                "ramp",
                "cruise",
                "standby",
                "ramp",
                "cruise",
            ],
        ],
        [
            *[[trunk, arm, overall, *cells] for trunk, arm, overall, cells in regime_rows[:2]],
            "\\midrule",
            *[[trunk, arm, overall, *cells] for trunk, arm, overall, cells in regime_rows[2:7]],
            "\\midrule",
            *[[trunk, arm, overall, *cells] for trunk, arm, overall, cells in regime_rows[7:]],
            "\\midrule",
            [
                "\\multicolumn{2}{l}{\\textit{the target labels of the same frames}}",
                "---",
                "---",
                "---",
                "---",
                "---",
                "---",
                "10.63",
                "7.85",
                "13.79",
            ],
        ],
        "Rotor-speed error by frame regime on the frozen real validation split (Hz, lower is "
        "better), at the checkpoint each arm selected on the real overall error. "
        "\\texttt{real\\_overall} is \\texttt{val/real\\_r3} at that checkpoint, i.e. the "
        "\\emph{all} column of Table~\\ref{tab:nv2}; \\emph{overall} is the same clip-level "
        "metric recomputed by \\texttt{scripts/\\_regime\\_decomp.py} on eight microphones, and "
        "the four regime columns are its per-frame decomposition. Thresholds are stated, not "
        "fitted (ramp first, at $|\\mathrm{d}\\bar r/\\mathrm{d}t|\\ge20$\\,Hz/s, then zero "
        "/ standby / cruise by level); frame shares are zero 12.7\\,\\%, standby 11.6\\,\\%, ramp "
        "3.8\\,\\%, cruise 72.0\\,\\%. \\emph{Output spread} is the mean over frames of the "
        "peak-to-peak spread of the four \\emph{predicted} speeds --- how far apart the model "
        "puts the outer rotors --- and the last row gives the same quantity for the labels "
        "themselves, which is the target the spread columns are read against (at zero the "
        "labels' spread is 0.01). The decomposition is pooled over DREGON and MD2: the committed "
        "decomposition carries no per-dataset split. $^{*}$\\texttt{nv2\\_hard\\_ft\\_hppnet\\_l2} "
        "is its saved round-4 checkpoint and not a result; its raw score of 1.81 at round 1 was "
        "never saved, so the arm is unmeasured, and its 20.91 at zero is a warm-started run one "
        "round in.",
        "tab:gen-results-by-regime",
        star=True,
        size="footnotesize",
        tabcolsep="4pt",
        source=(
            "source: docs/experiments/noise-v2-transfer.md,\n"
            '"Stage-1 and curriculum results" (real_overall) and\n'
            '"Four-regime decomposition" (overall, regime cells, output spread)'
        ),
    )


def main() -> None:
    which = sys.argv[1:] or [
        "tables",
        "traj",
        "traj_other",
        "hyper",
        "fits",
        "noise",
        "prior",
    ]
    if "tables" in which:
        tables()
    if "traj" in which:
        fig_traj_real_vs_sampled()
    if "traj_other" in which:
        fig_traj_real_vs_sampled(
            ("pitcn_quad", "nanobench_cf21b", "vid_m100"),
            name="gen_traj_real_vs_sampled_other",
            row_h=2.9,
        )
    if "hyper" in which:
        fig_traj_hyperprior()
    if "fits" in which:
        fig_traj_fits()
    if "noise" in which:
        fig_noise_fits()
    if "prior" in which:
        fig_noise_prior()


if __name__ == "__main__":
    main()
