#!/usr/bin/env python
"""Trajectory statistics of the training streams against the real validation.

Candidate (2) of ``docs/explainers/noise-model-v3-latent-runaway.qmd``: is the
HARD trajectory sampler of the v2/v3 streams too wild? Every stream window is
drawn by the arm's own pool (the code the training stream calls), built from
its committed policy file exactly as ``notebooks/noise_lab.py``'s
``_stream_rps`` builds it, with ONE change: ``flight_reuse: 1``, so every
window comes from a fresh flight. The per-window distribution is unchanged
(the stream places each window uniformly in its flight and draws
``rps_scale`` per window); only the correlation between windows sharing a
flight goes.

The v3 and v3r3 streams (``conf/online_mix/noise_v3{,r3}_{easy,hard}_5050.yaml``)
differ from the v2 ones in the ``preset_bank`` line only, and every bank of a
preset carries the same ``traj_rig`` split (easy: 1024 dregon + 1024
michaels; hard: 2048 null), so their trajectory draw IS the v2 draw.

References:

* ``real`` -- the 37 clips of ``dload:DREGON-LM-V4-michaels-valid-full``
  (``rps.npy``, shape-stretched over the 8 s clip as ``DregonLMFrameDataset``
  does), each cut into four 2 s windows;
* ``synth_valid`` -- the trajectory recipe of the synthetic validation parts
  (``static_*`` = ``salv2_comb.yaml``, ``stochastic_*`` = ``salv2_stoch.yaml``;
  both are ``full_flight`` + ``synthetic_intermittent`` at weight 1:1,
  ``rps_scale_range`` [0.45, 1.2]).

Window length is 2 s, the training clip length of every SCv2 arm
(``conf/experiment/rig_easy_scv2_unified.yaml``, ``duration_s: 2.0``).

    PYTHONPATH=src python scripts/noise_v3_traj_stats.py            # draw + plot
    PYTHONPATH=src python scripts/noise_v3_traj_stats.py --plot-only

Writes ``docs/explainers/noise-model-v3-latent-runaway/traj_stats.json`` and
``traj_ecdf.png``.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "notebooks")]

OUT = ROOT / "docs/explainers/noise-model-v3-latent-runaway"
DATA = OUT / "traj_stats.json"
FIG = OUT / "traj_ecdf.png"

N_WINDOWS = 512
DURATION_S = 2.0
SEED = 20260926
GRID_HZ = 100.0  # statistics grid
RATE_HZ = 10.0  # rate of change on 0.1 s block means
LOW_RPS, HIGH_RPS = 40.0, 90.0
REAL = "dload:DREGON-LM-V4-michaels-valid-full"
REAL_CLIP_S = 8.0

#: name -> (label, kind, policy)
ARMS = {
    "legacy_easy": ("legacy easy", "legacy_stream", "rig_easy_5050"),
    "legacy_hard": ("legacy hard", "legacy_stream", "rig_hard_5050"),
    "v2_easy": ("v2/v3 easy", "v2_stream", "noise_v2_easy_5050"),
    "v2_hard": ("v2/v3 hard", "v2_stream", "noise_v2_hard_5050"),
}
REFS = {"real": "real valid", "synth_valid": "synthetic valid"}

STATS = {
    "mean": ("mean speed", "rev/s"),
    "range": ("range (max − min)", "rev/s"),
    "min": ("min", "rev/s"),
    "max": ("max", "rev/s"),
    "std_time": ("std over time, rotor mean", "rev/s"),
    "std_rotor": ("std over time, per rotor", "rev/s"),
    "sep_mean": ("mean rotor separation", "rev/s"),
    "sep_max": ("max rotor separation", "rev/s"),
    "rate_med": ("|df/dt| median", "rev/s²"),
    "rate_p95": ("|df/dt| 95th pct", "rev/s²"),
    "frac_low": (f"time below {LOW_RPS:.0f} rev/s", "fraction"),
    "frac_high": (f"time above {HIGH_RPS:.0f} rev/s", "fraction"),
}

STYLE = {
    "legacy_easy": dict(color="#1f77b4", ls="--", lw=1.4),
    "legacy_hard": dict(color="#1f77b4", ls="-", lw=2.0),
    "v2_easy": dict(color="#d62728", ls="--", lw=1.4),
    "v2_hard": dict(color="#d62728", ls="-", lw=2.0),
    "real": dict(color="#000000", ls="-", lw=2.6),
    "synth_valid": dict(color="#7f7f7f", ls=":", lw=2.0),
}

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


# ── per-window statistics ─────────────────────────────────────────────────


def window_stats(rps: np.ndarray) -> dict[str, float]:
    """All statistics of one ``(R, T)`` window on the :data:`GRID_HZ` grid."""
    r = np.asarray(rps, dtype=np.float64)
    n_rot = r.shape[0]
    pairs = [(i, j) for i in range(n_rot) for j in range(i + 1, n_rot)]
    sep = np.stack([np.abs(r[i] - r[j]) for i, j in pairs])
    block = int(round(GRID_HZ / RATE_HZ))
    n_blk = r.shape[1] // block
    coarse = r[:, : n_blk * block].reshape(n_rot, n_blk, block).mean(-1)
    rate = np.abs(np.diff(coarse, axis=1)).ravel() * RATE_HZ
    return {
        "mean": float(r.mean()),
        "range": float(r.max() - r.min()),
        "min": float(r.min()),
        "max": float(r.max()),
        "std_time": float(r.mean(0).std()),
        "std_rotor": float(r.std(1).mean()),
        "sep_mean": float(sep.mean()),
        "sep_max": float(sep.max()),
        "rate_med": float(np.median(rate)),
        "rate_p95": float(np.percentile(rate, 95)),
        "frac_low": float((r < LOW_RPS).mean()),
        "frac_high": float((r > HIGH_RPS).mean()),
    }


def _to_grid(rps_audio: np.ndarray, sr: int) -> np.ndarray:
    step = int(round(sr / GRID_HZ))
    return np.asarray(rps_audio)[:, ::step]


# ── draws ─────────────────────────────────────────────────────────────────


def draw_arm(kind: str, policy: str, n: int, seed: int) -> list[np.ndarray]:
    """``n`` independent 2 s windows of one arm's stream, each on a fresh flight."""
    import noise_lab as nl

    cfg = copy.deepcopy(nl._policy_noise_source(policy))
    cfg["rps"]["flight_reuse"] = 1
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    if kind == "legacy_stream":
        from data_processing.stochastic_rotor_noise import StochasticNoisePool

        cfg.pop("preset_bank", None)
        pool: Any = StochasticNoisePool.from_config(cfg, duration_s=DURATION_S, sample_rate=nl.SR)
        for _ in range(n):
            out.append(_to_grid(pool.sample_rps(rng, DURATION_S), nl.SR))
    else:
        from data_processing.noise_v2_pool import NoiseV2Pool

        cfg["preset_bank"] = str(nl.ROOT / nl.V2_BANKS["hard" if "hard" in policy else "easy"])
        pool = NoiseV2Pool.from_config(cfg, duration_s=DURATION_S, sample_rate=nl.SR)
        for _ in range(n):
            entry = pool.entries[int(rng.integers(len(pool.entries)))]
            out.append(_to_grid(pool.sample_rps(rng, DURATION_S, entry), nl.SR))
    return out


def draw_synth_valid(n: int, seed: int) -> list[np.ndarray]:
    """The synthetic validation parts' trajectory recipe (salv2_stoch == salv2_comb)."""
    import noise_lab as nl

    from data_processing.stochastic_rotor_noise import StochasticNoisePool

    cfg = yaml.safe_load((ROOT / "conf/online_mix/salv2_stoch.yaml").read_text())
    pools = []
    for src in cfg["sources"]["noise"]:
        if src.get("kind") != "stochastic":
            continue
        src = copy.deepcopy(src)
        src["rps"]["flight_reuse"] = 1
        pools.append(StochasticNoisePool.from_config(src, duration_s=DURATION_S, sample_rate=nl.SR))
    rng = np.random.default_rng(seed)
    return [
        _to_grid(pools[int(rng.integers(len(pools)))].sample_rps(rng, DURATION_S), nl.SR)
        for _ in range(n)
    ]


def real_windows() -> tuple[list[np.ndarray], list[str]]:
    """Every validation clip, on the stats grid, cut into 2 s windows."""
    from data_processing.streams import resolve_source

    root = Path(resolve_source(REAL))
    meta = json.loads((root / "metadata.json").read_text())["valid"]
    n_grid = int(round(REAL_CLIP_S * GRID_HZ))
    t = (np.arange(n_grid) + 0.5) / GRID_HZ
    win = int(round(DURATION_S * GRID_HZ))
    out, rigs = [], []
    for m in meta:
        raw = np.load(root / m["id"] / "rps.npy").astype(np.float64)
        t_src = np.linspace(0.0, float(m["duration"]), raw.shape[1])
        grid = np.stack([np.interp(t, t_src, row) for row in raw])
        rig = "michaels" if str(m["recording_id"]).startswith("michaels") else "dregon"
        for s in range(0, n_grid - win + 1, win):
            out.append(grid[:, s : s + win])
            rigs.append(rig)
    return out, rigs


# ── summaries ─────────────────────────────────────────────────────────────


def _summary(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    out = {}
    for key in STATS:
        v = np.array([r[key] for r in rows])
        q = np.percentile(v, [5, 25, 50, 75, 95])
        out[key] = {
            "p05": float(q[0]),
            "q1": float(q[1]),
            "median": float(q[2]),
            "q3": float(q[3]),
            "p95": float(q[4]),
            "mean": float(v.mean()),
        }
    return out


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    grid = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), grid, side="right") / a.size
    cb = np.searchsorted(np.sort(b), grid, side="right") / b.size
    return float(np.abs(ca - cb).max())


def evaluate() -> dict[str, Any]:
    t0 = time.time()
    groups: dict[str, list[dict[str, float]]] = {}
    for i, (name, (_, kind, policy)) in enumerate(ARMS.items()):
        groups[name] = [window_stats(w) for w in draw_arm(kind, policy, N_WINDOWS, SEED + i)]
        print(f"{name}: {time.time() - t0:.1f} s", flush=True)
    groups["synth_valid"] = [window_stats(w) for w in draw_synth_valid(N_WINDOWS, SEED + 99)]
    real, rigs = real_windows()
    groups["real"] = [window_stats(w) for w in real]
    rig_arr = np.array(rigs)
    real_by_rig = {
        rig: _summary([s for s, g in zip(groups["real"], rig_arr, strict=True) if g == rig])
        for rig in ("dregon", "michaels")
    }
    ks = {
        name: {
            key: _ks(
                np.array([r[key] for r in rows]),
                np.array([r[key] for r in groups["real"]]),
            )
            for key in STATS
        }
        for name, rows in groups.items()
        if name != "real"
    }
    return {
        "provenance": {
            "script": "scripts/noise_v3_traj_stats.py",
            "n_windows": N_WINDOWS,
            "n_real_windows": len(real),
            "real_windows_by_rig": {r: int((rig_arr == r).sum()) for r in ("dregon", "michaels")},
            "duration_s": DURATION_S,
            "seed": SEED,
            "grid_hz": GRID_HZ,
            "rate_grid_hz": RATE_HZ,
            "low_high_rps": [LOW_RPS, HIGH_RPS],
            "real": REAL,
            "flight_reuse": 1,
            "policies": {name: f"conf/online_mix/{p}.yaml" for name, (_, _, p) in ARMS.items()},
            "v3_same_as_v2": (
                "noise_v3{,r3}_{easy,hard}_5050.yaml differ from noise_v2_* in "
                "`preset_bank` only; every bank's traj_rig split is identical "
                "(easy 1024 dregon + 1024 michaels, hard 2048 null)"
            ),
            "legacy_easy_same_as_hard": (
                "rig_easy_5050.yaml and rig_hard_5050.yaml differ in `preset_bank` only; "
                "both fly full_flight, aggressiveness 1.0, rps_scale_range [0.45, 1.2]"
            ),
        },
        "labels": {**{k: v[0] for k, v in ARMS.items()}, **REFS},
        "stats": {k: {"label": v[0], "unit": v[1]} for k, v in STATS.items()},
        "summary": {name: _summary(rows) for name, rows in groups.items()},
        "real_by_rig": real_by_rig,
        "ks_to_real": ks,
        "windows": {
            name: {k: [round(r[k], 4) for r in rows] for k in STATS}
            for name, rows in groups.items()
        },
    }


# ── figure ────────────────────────────────────────────────────────────────

PANELS = ("mean", "range", "std_time", "sep_mean", "rate_med", "rate_p95")
ORDER = ("legacy_easy", "legacy_hard", "v2_easy", "v2_hard", "synth_valid", "real")


def plot(data: dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    for ax, key in zip(axes.ravel(), PANELS, strict=True):
        label, unit = STATS[key]
        for name in ORDER:
            v = np.sort(np.asarray(data["windows"][name][key]))
            y = np.arange(1, v.size + 1) / v.size
            ax.step(v, y, where="post", label=data["labels"][name], **STYLE[name])
        if key.startswith("rate") or key in ("range", "std_time", "sep_mean"):
            ax.set_xscale("symlog", linthresh=1.0)
        ax.set_title(label)
        ax.set_xlabel(unit)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction of 2 s windows")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.03)
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG)
    plt.close(fig)
    print(f"wrote {FIG}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Trajectory statistics of the training streams.")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        data = json.loads(DATA.read_text())
    else:
        data = evaluate()
        DATA.write_text(json.dumps(data))
        print(f"wrote {DATA}")
    plot(data)
    for key in STATS:
        row = "  ".join(f"{name}={data['summary'][name][key]['median']:.2f}" for name in ORDER)
        print(f"{key:10s} {row}")


if __name__ == "__main__":
    main()
