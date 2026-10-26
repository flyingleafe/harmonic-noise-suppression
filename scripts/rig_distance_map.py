"""Map the wide-prior training bank, the two fits and the validation sectors
under the rotor-permutation-invariant spectral distance
(:mod:`experiments.noise_model.rig_distance`; Dmitrii, 2026-10-26).

    # compute (uni-cpu): features in parallel, three distance matrices, metric MDS
    PYTHONPATH=src python scripts/rig_distance_map.py --workers 32
    # plot from the written npz (laptop)
    PYTHONPATH=src python scripts/rig_distance_map.py --plot

Rigs: a random ``--n-bank`` subsample of ``prior-v4-banks`` (seed ``--seed``),
the DREGON bench fit and the MD2 standby/cruise pair
(:data:`experiments.noise_model.sectors_v4.FITS`), and every rig of the six
``validation-sectors-v4`` sectors. Writes ``<out>/distances.npz`` (group and
name per rig, the harmonic / broadband / total distance matrices, a 2-D metric
MDS of each and its stress-1) and ``<out>/summary.json`` (distances from the
bank and from each sector to each fit, measured in the full distance, not on
the map); ``--plot`` writes ``<out>/map.{png,pdf}``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

BANK = "dload:prior-v4-banks/prior_v4_n16384.npz"
SECTORS = "dload:validation-sectors-v4"
NPZ_SECTORS = ("uniform", "weak_comb", "strong_motor", "wide_pedestal")
JSON_SECTORS = ("near_dregon", "near_md2")
COMPONENTS = ("total", "harmonic", "broadband")

_CACHE: dict[str, Any] = {}


def _source(path: str) -> Any:
    """A bank (lazy) or a json sector, loaded once per process."""
    if path not in _CACHE:
        from data_processing.noise_model.prior_bank_v4 import PriorBankV4
        from data_processing.streams import resolve_source

        local = Path(resolve_source(path))
        _CACHE[path] = (
            PriorBankV4(local)
            if local.suffix == ".npz"
            else json.loads(local.read_text())["entries"]
        )
    return _CACHE[path]


def _rig(job: tuple[str, str, Any]) -> Any:
    group, path, key = job
    if group == "fit":
        from experiments.noise_model.sectors_v4 import fit_regimes

        return fit_regimes(key, root=ROOT)
    src = _source(path)
    return src.payload(int(key)) if hasattr(src, "payload") else src[int(key)]


def _init_worker() -> None:
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)


def _features(job: tuple[str, str, Any]) -> tuple[np.ndarray, np.ndarray]:
    from experiments.noise_model.rig_distance import features

    return features(_rig(job))


def _jobs(n_bank: int, seed: int) -> tuple[list[tuple[str, str, Any]], list[str], list[str]]:
    jobs: list[tuple[str, str, Any]] = [("fit", "", "dregon"), ("fit", "", "md2")]
    groups, names = ["fit_dregon", "fit_md2"], ["dregon", "md2"]
    n_total = len(_source(BANK))
    pick = np.sort(np.random.default_rng(seed).choice(n_total, n_bank, replace=False))
    jobs += [("bank", BANK, int(i)) for i in pick]
    groups += ["bank"] * n_bank
    names += [f"bank_{i}" for i in pick]
    for sec in NPZ_SECTORS + JSON_SECTORS:
        path = f"{SECTORS}/{sec}.{'npz' if sec in NPZ_SECTORS else 'json'}"
        n = len(_source(path))
        jobs += [(sec, path, i) for i in range(n)]
        groups += [sec] * n
        names += [f"{sec}_{i}" for i in range(n)]
    return jobs, groups, names


def _q(x: np.ndarray) -> dict[str, float]:
    return {
        "min": float(x.min()),
        "p1": float(np.quantile(x, 0.01)),
        "p5": float(np.quantile(x, 0.05)),
        "median": float(np.median(x)),
    }


def compute(a: argparse.Namespace) -> None:
    from experiments.noise_model.rig_distance import metric_mds, pairwise

    t0 = time.time()
    jobs, groups, names = _jobs(a.n_bank, a.seed)
    with mp.get_context("fork").Pool(a.workers, initializer=_init_worker) as pool:
        feats = pool.map(_features, jobs, chunksize=4)
    t_feat = time.time() - t0
    print(f"features of {len(jobs)} rigs in {t_feat:.0f} s", flush=True)
    harm = np.stack([f[0] for f in feats])
    broad = np.stack([f[1] for f in feats])
    del feats
    dist = pairwise(harm, broad)
    del harm, broad
    print(f"distances in {time.time() - t0 - t_feat:.0f} s", flush=True)
    g = np.asarray(groups)
    maps, stress = {}, {}
    for comp in COMPONENTS:
        maps[comp], stress[comp] = metric_mds(dist[comp])
        print(f"{comp}: stress-1 {stress[comp]:.3f}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "distances.npz",
        groups=g,
        names=np.asarray(names),
        **{f"d_{c}": dist[c] for c in COMPONENTS},
        **{f"xy_{c}": maps[c] for c in COMPONENTS},
        stress=np.asarray([stress[c] for c in COMPONENTS]),
    )
    summary: dict[str, Any] = {
        "rigs": {k: int((g == k).sum()) for k in dict.fromkeys(groups)},
        "stress1": stress,
        "fit_to_fit": {c: float(dist[c][0, 1]) for c in COMPONENTS},
        "to_fit": {},
        "seconds": {"features": round(t_feat, 1), "total": round(time.time() - t0, 1)},
        "args": vars(a),
    }
    for i, fit in enumerate(("dregon", "md2")):
        row: dict[str, Any] = {}
        for c in COMPONENTS:
            d = dist[c][i]
            row[c] = {"bank": _q(d[g == "bank"])}
            for sec in NPZ_SECTORS + JSON_SECTORS:
                row[c][sec] = _q(d[g == sec])
        summary["to_fit"][fit] = row
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("rigs", "stress1", "fit_to_fit")}, indent=1))
    for fit, row in summary["to_fit"].items():
        for c in COMPONENTS:
            cells = "  ".join(f"{k} {v['min']:.2f}/{v['median']:.2f}" for k, v in row[c].items())
            print(f"to {fit:6s} {c:9s} (min/median) {cells}", flush=True)


SECTOR_STYLE = {
    "uniform": "#1f77b4",
    "weak_comb": "#2ca02c",
    "strong_motor": "#9467bd",
    "wide_pedestal": "#8c564b",
    "near_dregon": "#d62728",
    "near_md2": "#ff7f0e",
}


#: Groups the zoomed row frames: the bank, its sectors and the MD2 neighbourhood
#: (the DREGON neighbourhood lies ~35 dB away and would flatten the rest).
ZOOM_GROUPS = (
    "bank",
    "uniform",
    "weak_comb",
    "strong_motor",
    "wide_pedestal",
    "near_md2",
    "fit_md2",
)


def _draw(ax: Any, xy: np.ndarray, g: np.ndarray) -> None:
    from scipy.spatial import ConvexHull

    b = g == "bank"
    ax.scatter(
        xy[b, 0], xy[b, 1], s=5, c="0.55", alpha=0.4, lw=0, label="training bank (2048 of 16 384)"
    )
    for sec, colour in SECTOR_STYLE.items():
        pts = xy[g == sec]
        ax.scatter(
            pts[:, 0], pts[:, 1], s=12, c=colour, lw=0, label=f"sector: {sec.replace('_', ' ')}"
        )
        loop = ConvexHull(pts).vertices
        loop = np.append(loop, loop[0])
        ax.fill(pts[loop, 0], pts[loop, 1], color=colour, alpha=0.08, lw=0)
        ax.plot(pts[loop, 0], pts[loop, 1], color=colour, lw=1.4)
    for fit, marker, colour in (("fit_dregon", "*", "#d62728"), ("fit_md2", "D", "#ff7f0e")):
        p = xy[g == fit][0]
        ax.scatter(
            *p,
            s=260 if marker == "*" else 90,
            marker=marker,
            c=colour,
            edgecolors="k",
            linewidths=1.2,
            zorder=5,
            label=fit.replace("fit_", "").upper() + " fit",
        )
    ax.set_xlabel("MDS 1 (dB)", fontsize=10)
    ax.set_ylabel("MDS 2 (dB)", fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25)


def plot(a: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(a.out)
    z = np.load(out / "distances.npz")
    g = z["groups"]
    stress = dict(zip(COMPONENTS, z["stress"], strict=True))
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))
    for col, comp in enumerate(COMPONENTS):
        xy = z[f"xy_{comp}"]
        _draw(axes[0, col], xy, g)
        axes[0, col].set_title(
            f"{comp} distance: metric MDS, stress-1 {stress[comp]:.2f}", fontsize=11
        )
        _draw(axes[1, col], xy, g)
        sel = xy[np.isin(g, ZOOM_GROUPS)]
        lo, hi = sel.min(axis=0), sel.max(axis=0)
        pad = 0.05 * (hi - lo)
        axes[1, col].set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        axes[1, col].set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        axes[1, col].set_aspect("equal", adjustable="box")
        axes[1, col].set_title(f"{comp}: zoom on the bank and MD2 (DREGON off-frame)", fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=5, fontsize=10, markerscale=1.6, frameon=False
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"map.{ext}", dpi=170, bbox_inches="tight")
    print(f"wrote {out / 'map.png'}")


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n-bank", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results/rig_distance_v4")
    ap.add_argument("--plot", action="store_true", help="plot from <out>/distances.npz only")
    return ap.parse_args()


def main() -> None:
    a = _args()
    if a.plot:
        plot(a)
    else:
        compute(a)


if __name__ == "__main__":
    main()
