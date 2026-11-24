"""Static, method-masked matched samples; no best-draw selection."""

import json
from pathlib import Path
from zlib import crc32

import matplotlib.pyplot as plt
import numpy as np

from plots.timeframe.renderers import ROTOR_COLORS

FLIGHTS = {
    "michaels": "FLY124",
    "vid_m100": "indoor_loaded_8",
    "dregon": "free-flight_nosource_room1",
}


def review_plates(
    unified: Path, out: Path, families: list[str], targets: list[dict], fs: float = 25.0
) -> None:
    # Same recording on each pass, chosen before inspecting candidate scores.
    original = ("ours", "phasevar4", "phasevar2+hsmm3")
    all_methods = [*families, *original]
    phases = ("startup", "standby", "takeoff", "cruise", "landing")
    rng = np.random.default_rng(147203)
    answer = {}
    with np.load(unified / "baseline_draw0.npz") as base:
        for rig, name in FLIGHTS.items():
            (target,) = [t for t in targets if (t["rig"], t["flight"]) == (rig, name)]
            key = rig + "__" + target["fold"]
            fig, axes = plt.subplots(
                5, 1 + len(all_methods), figsize=(3 * (1 + len(all_methods)), 9.2), sharey="row"
            )
            fig.subplots_adjust(
                left=0.053, right=0.995, top=0.90, bottom=0.06, hspace=0.57, wspace=0.08
            )
            data = {
                p: {m: base[f"{key}__{name}__{p}__{m}"] for m in ("real", *original)}
                for p in phases
            }
            for method in families:
                with np.load(out / "draws" / f"{key}__{method}.npz") as z:
                    for phase in phases:
                        x = z[f"{name}__{phase}__draws"]
                        data[phase][method] = x[len(x) // 2, 0]
            for i, phase in enumerate(phases):
                order = rng.permutation(all_methods).tolist()
                answer[rig + "/" + name + "/" + phase] = order
                for j, m in enumerate(("real", *order)):
                    ax = axes[i, j]
                    values = data[phase][m]
                    time = np.arange(values.shape[1]) / fs
                    for y, c in zip(values, ROTOR_COLORS, strict=True):
                        ax.plot(time, y, color=c, lw=0.85)
                    ax.spines[["top", "right"]].set_visible(False)
                    ax.set_xlim(0, time[-1])
                    ax.grid(axis="y", alpha=0.23)
                    ax.tick_params(labelsize=8)
                    if j == 0:
                        ax.set_ylabel(phase.capitalize() + "\nrev/s", fontsize=10)
                    if i == 0:
                        ax.set_title(
                            "Measured" if j == 0 else chr(ord("A") + j - 1),
                            fontsize=13,
                            weight="bold",
                        )
                    if i == 4:
                        ax.set_xlabel("Window time (s)", fontsize=9)
            fig.suptitle(
                f"{rig} / {name}  ·  independently shuffled methods on each row",
                fontsize=16,
                weight="bold",
            )
            fig.savefig(out / f"blind_{rig}.png", dpi=155, facecolor="white")
            plt.close(fig)
    (out / "blind_key.json").write_text(json.dumps(answer, indent=1) + "\n")


def full_plates(
    out: Path,
    families: list[str],
    original: tuple[str, ...],
    fs: float = 25.0,
    *,
    all_flights: bool = False,
) -> None:
    """Complete chronological traces, one independent shuffle for each rig."""
    review = out / "complete_review" / ("+".join(families))
    if all_flights:
        review = review / "all_flights"
    review.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng([55392, crc32("|".join(families).encode())])
    key = {}
    with np.load(
        out / ("full_panel_draw0.npz" if all_flights else "full_baseline_draw0.npz")
    ) as old:
        targets = [
            s.removesuffix("__real").split("__", 2) for s in old.files if s.endswith("__real")
        ]
        if not all_flights:
            targets = [
                next(t for t in targets if t[0] == r and t[2] == f) for r, f in FLIGHTS.items()
            ]
        for rig, fold, flight in targets:
            prefix = f"{rig}__{fold}__{flight}__"
            traces = {name: old[prefix + name] for name in ("real", *original)}
            for name in families:
                with np.load(out / "draws" / f"{rig}__{fold}__{name}.npz") as z:
                    traces[name] = z[flight + "__paths"][0]
            order = rng.permutation([*original, *families]).tolist()
            key[rig + "/" + flight] = order
            labels = ["Measured", *[chr(ord("A") + j) for j in range(len(order))]]
            fig, axes = plt.subplots(
                len(labels), 1, figsize=(12, 1.38 * len(labels) + 1), sharex=True, sharey=True
            )
            for ax, method, label in zip(axes, ("real", *order), labels, strict=True):
                values = traces[method]
                sec = np.arange(values.shape[1]) / fs
                for y, color in zip(values, ROTOR_COLORS, strict=True):
                    ax.plot(sec, y, color=color, lw=0.65)
                ax.set_ylabel(label, rotation=0, labelpad=28)
                ax.spines[["top", "right"]].set_visible(False)
                ax.set_xlim(0, sec[-1])
                ax.grid(axis="y", alpha=0.22)
            axes[-1].set_xlabel("Time from recording start (s)")
            axes[0].set_title(f"{rig} / {flight} · complete motor speeds (rev/s) · first draw")
            fig.subplots_adjust(left=0.092, right=0.985, hspace=0.27, top=0.95, bottom=0.065)
            suffix = f"{rig}__{flight}" if all_flights else rig
            fig.savefig(review / f"complete_{suffix}.png", dpi=140, facecolor="white")
            plt.close(fig)
    (review / "complete_key.json").write_text(json.dumps(key, indent=1) + "\n")
