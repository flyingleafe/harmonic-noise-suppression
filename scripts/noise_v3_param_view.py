#!/usr/bin/env python
"""The parameter view of a noise-model fit, v3 beside the v2 fit it replaces.

A port of ``notebooks/noise_lab.ipynb``'s inline ``plot_v2_params`` cell (the
"Parameter view — comb over floor, per rotor" section) to a script that takes
fit JSONs. Stems are the model's EXPECTED periodogram at ``k f`` (mic mean, one
frame at a constant speed ``--rps`` for every rotor), grey is the same
expectation with the comb switched off (``profile_db -> -300 dB``) and the red
cap is ``±gamma_rk``. One change from the notebook cell: each rotor's pane
switches the OTHER rotors' combs off, so a pane shows that rotor's own profile
and not the four-rotor sum at one shared speed.

A second figure puts the numbers side by side, because at 8 kHz span a
0.03 k Hz cap is invisible: ``gamma_rk / (0.01 k)`` against ``k`` (with the
half-normal prior's scale ``c_gamma`` and the ``5 gamma_0 k`` bar), ``profile_db``
against ``k`` per rotor, and the comb-off floor curve.

    PYTHONPATH=src python scripts/noise_v3_param_view.py \
        --fit v3=results/noise_v3/fits/dregon_room2_floor__flight_v3.json \
        --fit v2=results/noise_v2/rounds/round5/fits/dregon_room2_floor__flight_profile.json \
        --rps 80 --name dregon --out results/noise_v3/param_view

Writes ``<out>/<name>_comb.png`` and ``<out>/<name>_params.png``.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from data_processing.noise_model import params as MDP  # noqa: E402
from experiments.noise_model.rig_sampler import ModelProbe  # noqa: E402

COMB_OFF_DB = -300.0
GAMMA0_HZ = 0.01
WIDTH_BAR = 5.0


def _off(fit: dict[str, Any], keep: int | None) -> dict[str, Any]:
    """``fit`` with every rotor's comb switched off except ``keep`` (None: all off)."""
    out = copy.deepcopy(fit)
    prof = np.asarray(out["params"]["profile"]["profile_db"], dtype=np.float64)
    mask = np.ones(prof.shape[0], dtype=bool)
    if keep is not None:
        mask[keep] = False
    prof[mask] = COMB_OFF_DB
    out["params"]["profile"]["profile_db"] = prof.tolist()
    return out


def view(fit: dict[str, Any], probe: ModelProbe, rps: float) -> dict[str, Any]:
    """Everything one column of the figures draws, for one fit."""
    p = fit["params"]
    gamma = np.atleast_2d(np.asarray(MDP.gamma_from_params(p), dtype=np.float64))  # (R, K)
    prof = np.atleast_2d(np.asarray(p["profile"]["profile_db"], dtype=np.float64))
    f = probe.freqs_hz
    floor = 10.0 * np.log10(
        np.maximum(probe.periodogram(_off(fit, None), rps).mean(axis=0), 1e-300)
    )
    own = [
        10.0 * np.log10(np.maximum(probe.periodogram(_off(fit, r), rps).mean(axis=0), 1e-300))
        for r in range(prof.shape[0])
    ]
    return dict(f=f, floor=floor, own=own, gamma=gamma, profile=prof)


def figure_comb(views: dict[str, dict[str, Any]], rps: float, fmax: float, title: str) -> Any:
    names = list(views)
    n_r = max(v["gamma"].shape[0] for v in views.values())
    fig, axes = plt.subplots(
        n_r,
        len(names),
        figsize=(10 * len(names), 2.4 * n_r),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    hi = -np.inf
    # the anti-alias roll-off above ~7.6 kHz would stretch the axis by 60 dB
    band = {n: (v["f"] > 100) & (v["f"] < 0.95 * fmax) for n, v in views.items()}
    lo = min(float(v["floor"][band[n]].min()) for n, v in views.items())
    for c, name in enumerate(names):
        v = views[name]
        f, floor = v["f"], v["floor"]
        k_ = v["gamma"].shape[1]
        k = np.arange(1, k_ + 1)
        freq = k * rps
        keep = freq <= fmax
        for r in range(v["gamma"].shape[0]):
            ax = axes[r][c]
            lvl = np.interp(freq, f, v["own"][r])
            hi = max(hi, float(np.nanmax(lvl[keep])), float(floor[f < fmax].max()))
            ax.plot(f, floor, color="0.4", lw=1.2, label="floor (comb off)")
            ax.vlines(
                freq[keep],
                np.interp(freq[keep], f, floor),
                lvl[keep],
                color="C0",
                lw=1.0,
                label="expected periodogram at k f (this rotor's comb only)",
            )
            g = v["gamma"][r][:k_][keep]
            ax.hlines(
                lvl[keep], freq[keep] - g, freq[keep] + g, color="C3", lw=2.5, label="±gamma_rk"
            )
            ax.set_ylabel(f"rotor {r}\ndB")
            ax.grid(alpha=0.3)
        axes[0][c].set_title(f"{name}: {title} at {rps:g} rev/s")
    for row in axes:
        for ax in row:
            ax.set_ylim(lo - 15.0, hi + 5.0)
    axes[0][0].legend(loc="upper right", ncol=3, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Hz")
        ax.set_xlim(0, fmax)
    fig.tight_layout()
    return fig


def figure_params(views: dict[str, dict[str, Any]], title: str, fmax: float) -> Any:
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.2))
    styles = {0: "-", 1: "--", 2: ":", 3: "-."}
    colours = {name: f"C{i}" for i, name in enumerate(views)}
    for name, v in views.items():
        g = v["gamma"]
        k = np.arange(1, g.shape[1] + 1, dtype=np.float64)
        for r in range(g.shape[0]):
            axes[0].plot(
                k,
                g[r] / (GAMMA0_HZ * k),
                color=colours[name],
                ls=styles[r % 4],
                lw=1.0,
                label=f"{name} r{r}",
            )
            axes[1].plot(
                k,
                v["profile"][r],
                color=colours[name],
                ls=styles[r % 4],
                lw=1.0,
                label=f"{name} r{r}",
            )
        sel = v["f"] <= fmax
        axes[2].plot(v["f"][sel], v["floor"][sel], color=colours[name], lw=1.2, label=name)
    axes[0].axhline(WIDTH_BAR, color="k", lw=0.8, ls=":", label=f"{WIDTH_BAR:g} gamma_0 k")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("order k")
    axes[0].set_ylabel("gamma_rk / (0.01 k)")
    axes[0].set_title("line half width in units of the bench law gamma_0 k")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("order k")
    axes[1].set_ylabel("profile_db (line power at f_ref, dB)")
    axes[1].set_title("per-order profile")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Hz")
    axes[2].set_ylabel("dB (mic mean)")
    axes[2].set_title("floor: expectation with the comb off")
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7, ncol=2)
    axes[2].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--fit",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="a fit JSON under a column label; repeat (e.g. v3=... v2=...)",
    )
    ap.add_argument("--rps", type=float, default=80.0, help="the one speed every rotor is drawn at")
    ap.add_argument("--fmax", type=float, default=8000.0)
    ap.add_argument("--name", required=True, help="figure stem, e.g. dregon")
    ap.add_argument("--out", type=Path, default=Path("results/noise_v3/param_view"))
    args = ap.parse_args(argv)

    fits: dict[str, dict[str, Any]] = {}
    for spec in args.fit:
        label, sep, path = str(spec).partition("=")
        if not sep:
            raise SystemExit(f"--fit wants LABEL=PATH, got {spec!r}")
        fits[label] = json.loads(Path(path).read_text())
    n_rotors = max(
        np.atleast_2d(np.asarray(f["params"]["profile"]["profile_db"])).shape[0]
        for f in fits.values()
    )
    probe = ModelProbe(n_rotors=n_rotors, n_mics=8, speeds=(float(args.rps),))
    views = {label: view(fit, probe, float(args.rps)) for label, fit in fits.items()}
    title = f"{args.name} — " + ", ".join(
        f"{k}: {v.get('schema')} {v.get('mode', '')}" for k, v in fits.items()
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for stem, fig in (
        ("comb", figure_comb(views, float(args.rps), float(args.fmax), args.name)),
        ("params", figure_params(views, title, float(args.fmax))),
    ):
        path = out / f"{args.name}_{stem}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
