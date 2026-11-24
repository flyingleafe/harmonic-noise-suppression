"""All held-out motor trajectories for the per-drone train/validation comparison."""

from __future__ import annotations

import base64
import csv
import json
from contextlib import ExitStack
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plotly.offline import get_plotlyjs

from plots.timeframe.renderers import ROTOR_COLORS

RIG_NAMES = {
    "michaels": "MD2",
    "vid_m100": "VID M100",
    "dregon": "DREGON",
    "blackbird_quad": "Blackbird",
    "neurobem_quad": "NeuroBEM",
    "pitcn_quad": "PI-TCN",
    "nanobench_cf21b": "NanoBench CF2.1B",
}
METHODS = {
    "real": "Measured",
    "regulated_bridge": "Regulated bridge",
    "template": "Template",
    "phasevar2": "Phase VAR(2)",
    "phasevar4": "Phase VAR(4)",
    "msvar3": "MS-VAR hybrid",
    "hsmm3": "VAR-HSMM hybrid",
}


def _bytes(x):
    return base64.b64encode(np.ascontiguousarray(x, dtype="<f4").tobytes()).decode("ascii")


def export(root: Path, assets: Path):
    """Freeze draw zero for all targets; preview the first ID lasting >=16 s, or the first ID."""
    assets.mkdir(parents=True, exist_ok=True)
    payload = {"fs": 25, "colors": list(ROTOR_COLORS), "methods": METHODS, "rigs": {}}
    views = {}
    for rig, label in RIG_NAMES.items():
        m = json.loads((root / "pack" / f"{rig}.train.json").read_text())
        trs = m["train"]
        metadata = {
            "label": label,
            "train_records": len(set(t["recording"] for t in trs)),
            "valid_records": len(set(t["recording"] for t in m["targets"])),
            "targets": [],
        }
        payload["rigs"][rig] = metadata
        targets = sorted(
            enumerate(m["targets"]), key=lambda pair: (pair[1]["recording"], pair[1]["id"])
        )
        with ExitStack() as stack:
            z = {"real": stack.enter_context(np.load(root / "pack" / f"{rig}.valid.npz"))}
            for arm in list(METHODS)[1:]:
                z[arm] = stack.enter_context(np.load(root / "runs" / rig / f"{arm}.draws.npz"))
            for i, t in targets:
                n = t["n"]
                k = f"t{i}"
                paths = {name: (src[k] if name == "real" else src[k][0]) for name, src in z.items()}
                if not all(x.shape == (4, n) for x in paths.values()):
                    raise ValueError(f"unmatched lengths: {rig}/{t['id']}")
                if any(np.isinf(x).any() for x in paths.values()):
                    raise ValueError(f"diverged path: {rig}/{t['id']}")
                rec = dict(
                    id=t["id"],
                    recording=t["recording"],
                    n=n,
                    events=t["events"],
                    paths={a: _bytes(x) for a, x in paths.items()},
                )
                metadata["targets"].append(rec)
                if rig not in views and n >= 400:
                    views[rig] = (t["id"], paths)
            if rig not in views:
                i, t = targets[0]
                views[rig] = (
                    t["id"],
                    {a: (s[f"t{i}"] if a == "real" else s[f"t{i}"][0]) for a, s in z.items()},
                )
    (assets / "trajectories.json").write_text(
        json.dumps(payload, separators=(",", ":"), allow_nan=False)
    )
    (assets / "plotly.html").write_text("<script>" + get_plotlyjs() + "</script>\n")
    overview(
        views,
        ("real", "regulated_bridge", "template", "hsmm3"),
        METHODS,
        assets / "every-drone.png",
    )
    return payload


def overview(views, choices, labels, path: Path) -> None:
    """Render preselected matched trajectories with shared physical axes per drone."""
    fig, axes = plt.subplots(
        len(views),
        len(choices),
        figsize=(3.5 * len(choices), 11.6),
        sharex="row",
        sharey="row",
        layout="constrained",
    )
    for i, (rig, (_, paths)) in enumerate(views.items()):
        for j, arm in enumerate(choices):
            ax = axes[i, j]
            x = paths[arm]
            sec = np.arange(x.shape[1]) / 25
            for h, y in enumerate(x):
                ax.plot(sec, y, color=ROTOR_COLORS[h], lw=0.65)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.18)
            ax.set_xlim(0, sec[-1])
            ax.tick_params(labelsize=8)
            if i == 0:
                ax.set_title(labels[arm], fontsize=12)
            if j == 0:
                ax.set_ylabel(RIG_NAMES[rig] + "\nrev/s", fontsize=9)
            if i == len(views) - 1:
                ax.set_xlabel("Time (s)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def tables(root: Path, assets: Path, payload: dict):
    """Generate Markdown tables without rounding before deciding a winner."""
    with (root / "phase_scores.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    arms = list(METHODS)[1:]
    full = [
        "| Drone | Phase | Validation recordings | " + " | ".join(METHODS[a] for a in arms) + " |",
        "|---|---|---:|" + "---:|" * len(arms),
    ]
    summary = [
        "| Drone | Train / valid recordings | Scored phases | "
        + " | ".join(METHODS[a] for a in arms)
        + " |",
        "|---|---:|---|" + "---:|" * len(arms),
    ]
    with (root / "whole_path_scores.csv").open() as handle:
        whole = {r["rig"]: r for r in csv.DictReader(handle)}
    raw = [
        "| Drone | Recordings | Common time | " + " | ".join(METHODS[a] for a in arms) + " |",
        "|---|---:|---:|" + "---:|" * len(arms),
    ]
    sizes = [
        "| Drone | " + " | ".join(METHODS[a] for a in arms) + " |",
        "|---|" + "---:|" * len(arms),
    ]
    for rig, meta in payload["rigs"].items():
        allrows = [r for r in rows if r["rig"] == rig]
        phases = [
            r
            for r in allrows
            if r["phase"].lower() not in ("balanced", "all", "overall") and int(r["n_valid"]) > 0
        ]
        if not phases:
            raise ValueError(f"no scored phases: {rig}")

        def fmt(vals):
            best = min(vals)
            return " | ".join(f"**{v:.3f}**" if abs(v - best) < 1e-12 else f"{v:.3f}" for v in vals)

        for row in phases:
            full.append(
                "| "
                + meta["label"]
                + " | "
                + row["phase"]
                + " | "
                + row["n_valid"]
                + " | "
                + fmt([float(row[a]) for a in arms])
                + " |"
            )
        values = [sum(float(r[a]) for r in phases) / len(phases) for a in arms]
        counts = (
            f"{meta['train_records']} / {meta['valid_records']}"
            if rig != "blackbird_quad"
            else "1 / 1 (same flight)"
        )
        summary.append(
            "| "
            + meta["label"]
            + " | "
            + counts
            + " | "
            + str(len(phases))
            + " | "
            + fmt(values)
            + " |"
        )
        w = whole[rig]
        coverage = int(w["common_samples"]) / int(w["total_samples"])
        raw.append(
            f"| {meta['label']} | {w['n_valid']} | {coverage:.1%} | "
            + fmt([float(w[a]) for a in arms])
            + " |"
        )
        vals = [
            json.loads((root / "runs" / rig / (arm + ".fit.json")).read_text())["n_params"]
            for arm in arms
        ]
        sizes.append(f"| {meta['label']} | " + " | ".join(str(v) for v in vals) + " |")
    (assets / "whole-path.md").write_text("\n".join(raw) + "\n")
    (assets / "model-sizes.md").write_text("\n".join(sizes) + "\n")
    (assets / "balanced.md").write_text("\n".join(summary) + "\n")
    (assets / "phase-scores.md").write_text("\n".join(full) + "\n")
