#!/usr/bin/env python
"""Calibrate ``rps.shrink`` and compare the cruise dynamics it draws with the
full hyperprior, the rejection references and real DREGON / MD2 telemetry.

The calibrated quantity is a drone's RELATIVE shaft dynamics: the closed-form
:meth:`~data_processing.trajectory_model.params.Params.shaft_dynamics` on the
labels' 31.25 Hz grid (std inside a 2 s window, std of the 0.32 s increment),
each the max over rotors, divided by the hover level and then by the same
statistic of the livelier of DREGON and michaels; a drone's ``ratio`` is the
larger of the two. The shrink weight is the smallest one at which at most
``--exceed`` of the drones have ``ratio > --margin``, found by bisection on
common random numbers (the statistic is scale-free, so the size is fixed).

Every arm then draws flights as a training stream does (posterior drones at
``hover_range`` [35, 95], ``rps_max`` 150, ``rotor_sep`` [2, 15], measurement
process off) and is read on the stationary AIRBORNE part only
(``FittedTrajectorySource.last_airborne``), the part
:func:`experiments.rps_traj.data.airborne_segments` keeps of a real flight:
per-rotor std in 2 s windows (1 s hop), per-rotor speed of change
``|w(t + 0.32 s) - w(t)| / 0.32 s``, and the share of frames whose rotor-mean
slew reaches the panel's 20 rev/s^2 ramp threshold. ``cap xM`` arms are the
REJECTION references: the fitted posterior conditioned on ``ratio <= M``.

    python scripts/rps_traj_shaft_shrink.py [--margin 2 --exceed 0.05] [--weight W]
        [--margins 1 1.5 2] [--flights 200]

Writes ``results/rps_traj/shaft_shrink/{stats.json,cdf.png,draws.png}``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

from data_processing.trajectory_model import (  # noqa: E402
    Params,
    Posterior,
    build_from_config,
    load_bundle,
    rig_vector,
)
from experiments.rps_traj.data import RATE_HZ, airborne_segments, load_rig  # noqa: E402

OUT = REPO_ROOT / "results" / "rps_traj" / "shaft_shrink"
FITS = "dload:rps-traj-fits"
REF_RIGS = ("dregon", "michaels")
FS_LABEL = 31.25
WIN, HOP, LAG = 62, 31, 10  # 2 s windows, 1 s hop, 0.32 s increment at 31.25 Hz
BASE = {
    "fits": FITS,
    "rigs": {"posterior": 1.0},
    "hover_range": [35.0, 95.0],
    "rps_max": 150.0,
    "rotor_sep": [2.0, 15.0],
    "measurement_noise": False,
}
COLOURS = {"DREGON": "black", "MD2": "0.55", "full": "tab:red"}


def relative(params: Params) -> np.ndarray:
    """``[window std, increment std] / hover``, each the max over rotors."""
    window, increment = params.shaft_dynamics(
        window_s=WIN / FS_LABEL, lag_s=LAG / FS_LABEL, fs=FS_LABEL
    )
    return np.array([np.max(window), np.max(increment)]) / float(np.mean(params.mu))


def reference() -> np.ndarray:
    bundle = load_bundle(FITS)
    return np.stack([relative(bundle.fits[r].params) for r in REF_RIGS]).max(axis=0)


def ratio(params: Params, ref: np.ndarray) -> float:
    return float(np.max(relative(params) / ref))


@dataclass
class Capped(Posterior):
    """The rejection reference: the posterior conditioned on ``ratio <= margin``."""

    ref: np.ndarray | None = None
    margin: float = 2.0
    refused: int = 0

    def sample(self, rng: np.random.Generator, *, hover: float | None = None) -> Params:
        assert self.ref is not None
        while True:
            params = super().sample(rng, hover=hover)
            if ratio(params, self.ref) <= self.margin:
                return params
            self.refused += 1


def shrink_target() -> np.ndarray:
    bundle = load_bundle(FITS)
    return np.mean([rig_vector(bundle.fits[r].params) for r in REF_RIGS], axis=0)


def drone_ratios(posterior: Posterior, ref: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([ratio(posterior.sample(rng, hover=65.0), ref) for _ in range(n)])


def calibrate(margin: float, exceed: float, n: int, seed: int) -> float:
    """Smallest weight (to 0.01, rounded up) with share(ratio > margin) <= exceed."""
    posterior, target, ref = load_bundle(FITS).posterior, shrink_target(), reference()
    assert posterior is not None

    def share(w: float) -> float:
        r = drone_ratios(posterior.shrunk_toward(target, w), ref, n, seed)
        return float(np.mean(r > margin))

    lo, hi = 0.0, 1.0
    while hi - lo > 0.005:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if share(mid) <= exceed else (mid, hi)
        print(f"  calibrate: weight in [{lo:.3f}, {hi:.3f}]", flush=True)
    return math.ceil(hi * 100.0 - 1e-9) / 100.0


def to_label_grid(x: np.ndarray) -> np.ndarray:
    """100 Hz -> 31.25 Hz (x 5/16, polyphase with its anti-alias filter)."""
    return resample_poly(np.asarray(x, dtype=np.float64), 5, 16, axis=-1)


def samples(series: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Pooled per-rotor window stds, speeds of change and rotor-mean slews."""
    wstd, slew, mslew = [], [], []
    for x in series:
        y = to_label_grid(x)
        for s0 in range(0, y.shape[1] - WIN + 1, HOP):
            wstd.append(y[:, s0 : s0 + WIN].std(axis=1))
        slew.append((np.abs(y[:, LAG:] - y[:, :-LAG]) * FS_LABEL / LAG).ravel())
        m = np.convolve(np.pad(y.mean(0), 4, mode="edge"), np.ones(9) / 9, mode="valid")
        mslew.append(np.abs(m[16:] - m[:-16]) * FS_LABEL / 16)  # +-0.25 s, the panel's rule
    return {
        "window_std": np.concatenate(wstd),
        "slew": np.concatenate(slew),
        "mean_slew": np.concatenate(mslew),
    }


def summary(s: dict[str, np.ndarray]) -> dict[str, float]:
    out = {}
    for key in ("window_std", "slew"):
        for q in (50, 90, 99):
            out[f"{key}_p{q}"] = float(np.percentile(s[key], q))
    out["mean_slew_p99"] = float(np.percentile(s["mean_slew"], 99))
    out["ramp_share"] = float(np.mean(s["mean_slew"] >= 20.0))
    return out


def real_airborne(rig: str) -> list[np.ndarray]:
    return [
        f.rps[:, sl]
        for f in load_rig(rig)
        for sl in airborne_segments(f.rps, RATE_HZ)
        if np.isfinite(f.rps[:, sl]).all()
    ]


def synthetic(
    n: int, seed: int, ref: np.ndarray, *, weight: float | None = None, cap: float | None = None
) -> tuple[list[np.ndarray], dict]:
    """``n`` accepted flights' airborne parts, from the full posterior, the
    shrunk one (``weight``) or the rejection reference (``cap``)."""
    cfg = dict(BASE)
    if weight is not None:
        cfg["shrink"] = {"rigs": list(REF_RIGS), "weight": weight}
    src = build_from_config(cfg)
    if cap is not None:
        p = src.posterior
        assert p is not None
        src.posterior = Capped(mean=p.mean, std=p.std, rigs=p.rigs, ref=ref, margin=cap)
    rng = np.random.default_rng(seed)
    xs, hovers, ratios = [], [], []
    for _ in range(n):
        src.flight(rng, RATE_HZ)
        assert src.last_airborne is not None and src.last_draw is not None
        xs.append(src.last_airborne.copy())
        hovers.append(float(np.mean(src.last_draw.params.mu)))
        ratios.append(ratio(src.last_draw.params, ref))
    meta: dict = dict(src.stats)
    if isinstance(src.posterior, Capped):
        meta["drones_refused"] = src.posterior.refused
    meta["hover_p10_p50_p90"] = [float(v) for v in np.percentile(hovers, [10, 50, 90])]
    meta["ratio_p50_p90_p99"] = [float(v) for v in np.percentile(ratios, [50, 90, 99])]
    meta["ratio_above"] = {f"x{m:g}": float(np.mean(np.asarray(ratios) > m)) for m in (1, 1.5, 2)}
    return xs, meta


def fig_cdf(arms: dict[str, dict[str, np.ndarray]], colours: dict[str, str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = {
        "window_std": "rotor-speed std in a 2 s window (rev/s)",
        "slew": "speed of change over 0.32 s (rev/s per s)",
    }
    xlims = {"window_std": (0.2, 50.0), "slew": (0.5, 300.0)}
    for ax, key in zip(axes, labels, strict=True):
        for name, s in arms.items():
            x = np.sort(s[key])
            y = 1.0 - np.arange(x.size) / x.size
            real = name in ("DREGON", "MD2")
            shrunk = name.startswith("shrink")
            ax.plot(
                x,
                y,
                color=colours[name],
                lw=2.2 if real or shrunk else 1.4,
                ls="-" if real or shrunk else "--",
                label=name,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1.0)
        ax.set_xlim(*xlims[key])
        ax.set_xlabel(labels[key])
        ax.set_ylabel("share of samples above x")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Airborne cruise dynamics: real telemetry vs hyperprior draws", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "cdf.png", dpi=150)
    plt.close(fig)


def fig_draws(rows: dict[str, list[np.ndarray]], seconds: float = 30.0, span: float = 60.0) -> None:
    """30 s airborne windows, three per row, every panel on the same
    ``span`` rev/s y-range about its own mean so the amplitudes compare; a
    row whose excursions do not fit gets a wider span, named in its titles."""
    n = int(seconds * RATE_HZ)
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 2.3 * len(rows)), sharex=True)
    for r, (name, series) in enumerate(rows.items()):
        windows = [series[c][:, :n] for c in range(3)]
        reach = max(float(np.max(np.abs(x - np.mean(x)))) for x in windows)
        row_span = max(span, 20.0 * np.ceil(reach / 10.0))
        note = "" if row_span == span else f", y-span {row_span:.0f} rev/s"
        for c, x in enumerate(windows):
            ax = axes[r, c]
            t = np.arange(x.shape[1]) / RATE_HZ
            for k in range(4):
                ax.plot(t, x[k], lw=0.7)
            mid = float(np.mean(x))
            ax.set_ylim(mid - row_span / 2, mid + row_span / 2)
            ax.set_title(f"{name} — hover {mid:.0f} rev/s{note}", fontsize=9)
            if c == 0:
                ax.set_ylabel("rev/s")
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(OUT / "draws.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--margin", type=float, default=2.0, help="calibration multiple")
    ap.add_argument("--exceed", type=float, default=0.05, help="drone share allowed above it")
    ap.add_argument("--weight", type=float, default=None, help="skip calibration, use this")
    ap.add_argument("--calib-drones", type=int, default=2000)
    ap.add_argument("--margins", type=float, nargs="*", default=[1.0, 1.5, 2.0])
    ap.add_argument("--flights", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ref = reference()
    weight = args.weight
    if weight is None:
        weight = calibrate(args.margin, args.exceed, args.calib_drones, args.seed)
    print(f"shrink weight {weight:.2f} (margin x{args.margin:g}, exceed {args.exceed:g})")

    real = {"DREGON": real_airborne("dregon"), "MD2": real_airborne("michaels")}
    arms = {name: samples(xs) for name, xs in real.items()}
    meta: dict[str, dict] = {name: {"segments": len(xs)} for name, xs in real.items()}
    draws: dict[str, list[np.ndarray]] = {}
    synth: list[tuple[str, dict]] = [("full", {})]
    synth += [(f"cap x{m:g}", {"cap": m}) for m in args.margins]
    synth += [(f"shrink {weight:.2f}", {"weight": weight})]
    for name, kw in synth:
        xs, meta[name] = synthetic(args.flights, args.seed, ref, **kw)
        arms[name] = samples(xs)
        draws[name] = xs
        print(name, meta[name], flush=True)
    table = {name: summary(s) for name, s in arms.items()}
    payload = {
        "calibration": {"margin": args.margin, "exceed": args.exceed, "weight": weight},
        "reference_rel": ref.tolist(),
        "stats": table,
        "meta": meta,
    }
    (OUT / "stats.json").write_text(json.dumps(payload, indent=2) + "\n")
    for name, row in table.items():
        print(f"{name:12s} " + "  ".join(f"{k}={v:.2f}" for k, v in row.items()))

    caps = [n for n in arms if n.startswith("cap")]
    blues = plt.get_cmap("Blues")(np.linspace(0.5, 0.95, max(len(caps), 1)))
    colours = {**COLOURS, **{n: to_hex(c) for n, c in zip(caps, blues, strict=False)}}
    colours[f"shrink {weight:.2f}"] = "tab:green"
    fig_cdf(arms, colours)
    rows = {"DREGON (real)": real["DREGON"][:3], "MD2 (real)": real["MD2"][:3], **draws}
    fig_draws(rows)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
