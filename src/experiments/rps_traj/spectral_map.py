"""Real 4-motor cruise: does the joint spectral distance separate the rigs?

Input is an evaluation-only table of cruise windows (``rows``: ``rig``,
``flight``, ``start_s``, ``source``) and its joint operating-speed and
cross-spectral features. Distances are Euclidean; the maps do not rescale
or re-weight the calibrated features.

**Unit of independence.** A flight (one physical recording) is the unit:
windows of one flight share conditions and may overlap in time, so they are
never evidence that a rig is self-consistent. Every "same-rig" comparison
pairs a target window with OTHER flights of its rig. Flights are keyed by
``(rig, flight)``.

**Same rig vs other rigs** (target = one window of a rig with >= 2 flights):

* ``within`` — mean over the rig's other flights of the mean distance from the
  target to that flight's windows (flight-balanced);
* ``to_rig[s]`` — the same against every flight of other rig ``s``;
  ``between`` — mean of ``to_rig`` over the other rigs (rig-balanced);
* ``rig_win`` — fraction of other rigs with ``within < to_rig[s]`` (tie 1/2);
* ``auc`` — P(distance to a same-rig other flight < distance to an other-rig
  flight), ties 1/2, over flight-mean distances, each same-rig flight weighted
  equally and each other rig weighted equally (its flights share its weight).

Window values are averaged within the target flight, flights averaged within
a rig, and rigs averaged into the macro figure. Rigs with a single flight
(e.g. Blackbird) contribute no targets but stay as comparators.

**Rig identification** — nearest rig centroid, strictly leave the entire
target flight out: a rig centroid is the mean of its flights' mean features,
recomputed without the target flight. Single-flight rigs are not targets
(their centroid would vanish) but remain candidate answers for every other
target. Reported per segment window (averaged per flight, per rig, macro) and,
separately, per flight: the held-out flight's mean feature against each rig
centroid computed without that entire flight.

**Rig separation.** An evaluable rig is separated iff its balanced segment win
(``auc`` above) exceeds 1/2 AND a strict majority of its held-out flights is
identified correctly at flight level.

**Within-record** distances (pairs of windows of one flight) are reported for
every rig, including single-flight rigs, as a separate optimistic floor.

**Map.** Rig centroids (mean of flight-mean features) give a true Euclidean
distance matrix (CSV) and a metric-MDS 2D picture (PNG). Start at the classical
solution and minimize squared pairwise distance errors. The map reports
normalized raw stress ``sqrt(sum (d2 - d)^2 / sum d^2)``, the pairwise
correlations and the worst-compressed pair are reported with it. The full
segment distance matrix (condensed, float32, input row order) is saved as a
compressed NPZ next to the centroid CSV.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import pearsonr, spearmanr

#: Output file names, inside ``out``.
CENTROID_CSV = "rig_centroid_distance.csv"
MDS_PNG = "rig_centroid_mds.png"
SEGMENT_NPZ = "segment_distance.npz"
MDS_DPI = 200
RIG_NAMES = {
    "blackbird_quad": "Blackbird",
    "dregon": "DREGON",
    "michaels": "MD2",
    "nanobench_cf21b": "NanoBench",
    "neurobem_quad": "NeuroBEM",
    "pitcn_quad": "PI-TCN",
    "vid_m100": "VID M100",
}

#: Distance entries per row block of the n x n pass (bounds its memory).
_BLOCK_ENTRIES = 1 << 24


def _num(x: Any) -> float | None:
    """JSON-safe float: NaN/inf become ``None``."""
    v = float(x)
    return v if math.isfinite(v) else None


def _mean(xs: list[float] | np.ndarray) -> float | None:
    arr = np.asarray([x for x in xs if x is not None], dtype=float)
    return _num(arr.mean()) if arr.size else None


def _segment_distances(x: np.ndarray, members: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """One pass over all segment pairs.

    Returns ``M`` with ``M[i, f]`` = mean distance from segment ``i`` to the
    segments of flight ``f`` (a segment's distance to itself excluded; NaN for
    a single-segment flight against its own segment), and the full condensed
    (``scipy.spatial.distance.squareform`` order) distance vector as float32.
    """
    n = x.shape[0]
    sums = np.zeros((n, len(members)))
    cond = np.empty(n * (n - 1) // 2, dtype=np.float32)
    step = max(1, _BLOCK_ENTRIES // n)
    for lo in range(0, n, step):
        hi = min(n, lo + step)
        blk = cdist(x[lo:hi], x)
        for f, idx in enumerate(members):
            sums[lo:hi, f] = blk[:, idx].sum(axis=1)
        for i in range(lo, hi):
            start = i * n - i * (i + 1) // 2
            cond[start : start + n - i - 1] = blk[i - lo, i + 1 :]
    cnt = np.array([float(idx.size) for idx in members])[None, :].repeat(n, axis=0)
    for f, idx in enumerate(members):
        cnt[idx, f] -= 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        m_out = np.where(cnt > 0, sums / np.maximum(cnt, 1.0), np.nan)
    return m_out, cond


def _win(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 where ``a < b``, 1/2 on ties, else 0 (elementwise, broadcast)."""
    return (a < b).astype(float) + 0.5 * (a == b)


def classical_mds(d: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Torgerson MDS of a distance matrix: ``(coords (n, k), eigenvalues desc)``.

    Each axis's sign is fixed so its largest-magnitude coordinate is positive.
    """
    n = d.shape[0]
    j = np.eye(n) - 1.0 / n
    b = -0.5 * j @ (d**2) @ j
    w, v = np.linalg.eigh((b + b.T) / 2)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    coords = v[:, :k] * np.sqrt(np.clip(w[:k], 0.0, None))
    for c in range(k):
        if coords[np.argmax(np.abs(coords[:, c])), c] < 0:
            coords[:, c] *= -1
    return coords, w


def mds_fidelity(d_true: np.ndarray, coords: np.ndarray, names: list[str]) -> dict[str, Any]:
    """How faithfully the 2D picture shows the true distances."""
    iu = np.triu_indices(d_true.shape[0], 1)
    dt = d_true[iu]
    d2 = squareform(pdist(coords))[iu]
    ratio = d2 / dt
    worst = int(np.argmin(ratio))
    most = int(np.argmax(ratio))
    d2_full = squareform(pdist(coords))
    nn_true = np.argmin(d_true + np.diag(np.full(len(names), np.inf)), axis=1)
    nn_2d = np.argmin(d2_full + np.diag(np.full(len(names), np.inf)), axis=1)
    return {
        "normalized_stress": _num(np.sqrt(((d2 - dt) ** 2).sum() / (dt**2).sum())),
        "pearson_r": _num(pearsonr(dt, d2)[0]),
        "spearman_rho": _num(spearmanr(dt, d2)[0]),
        "worst_compression": {
            "pair": [names[iu[0][worst]], names[iu[1][worst]]],
            "ratio_2d_over_true": _num(ratio[worst]),
            "true": _num(dt[worst]),
            "mds_2d": _num(d2[worst]),
        },
        "max_ratio_2d_over_true": {
            "pair": [names[iu[0][most]], names[iu[1][most]]],
            "ratio": _num(ratio[most]),
        },
        "median_ratio_2d_over_true": _num(np.median(ratio)),
        "nearest_neighbour_preserved": int((nn_true == nn_2d).sum()),
        "nearest_neighbour_of": len(names),
    }


def _metric_mds(distances: np.ndarray) -> np.ndarray:
    from sklearn.manifold import smacof

    initial, _ = classical_mds(distances)
    # sklearn 1.8 types n_init by its "warn" sentinel default; an int is the documented value.
    out = smacof(distances, init=initial, n_init=1, n_components=2, max_iter=1000, eps=1e-10)  # pyright: ignore[reportArgumentType]
    return cast(np.ndarray, out[0])


def map_all_flights(rows: list[dict], joint: np.ndarray, means: np.ndarray, out: Path) -> dict:
    """Descriptive map: include calibration flights; never use 2D for scoring."""
    from plots.flight_distribution import write_flight_map

    groups: dict[tuple[str, str], list[int]] = {}
    for i, row in enumerate(rows):
        groups.setdefault((row["rig"], row["flight"]), []).append(i)
    keys = sorted(groups)
    vectors = np.stack([joint[groups[k]].mean(axis=0) for k in keys])
    distances = squareform(pdist(vectors))
    xy = _metric_mds(distances)
    records = []
    for (rig, flight), (x, y) in zip(keys, xy):
        ids = groups[rig, flight]
        speeds = means[ids].mean(axis=0)
        records.append(
            dict(
                rig=rig,
                flight=flight,
                role="calibration" if rows[ids[0]]["calibration"] else "evaluation",
                n_windows=len(ids),
                mds_1=x,
                mds_2=y,
                **dict(zip(("front_right", "front_left", "rear_left", "rear_right"), speeds)),
            )
        )
    with (out / "flight_mds_coordinates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    write_flight_map(records, RIG_NAMES, out / "flight_mds.html")
    return mds_fidelity(distances, xy, [f"{r}/{f}" for r, f in keys])


def render_centroids(d: np.ndarray, rigs: list[str], counts: dict, path: Path) -> dict:
    """Fit only the 2D display. Never use its coordinates for the identity test."""
    from plots.labeled_map import plot_labeled_map

    xy = _metric_mds(d)
    fit = mds_fidelity(d, xy, rigs)
    labels = [RIG_NAMES.get(r, r) for r in rigs]
    legend = [
        f"{label}  ·  {n} {'flight' if n == 1 else 'flights'}"
        for label, r in zip(labels, rigs)
        for n in [counts[r]["n_flights"]]
    ]
    plot_labeled_map(
        xy,
        labels,
        path,
        legend_labels=legend,
        info_lines=[
            "2D map fidelity",
            f"Normalized stress   {fit['normalized_stress']:.3f}",
            f"Distance rank corr. {fit['spearman_rho']:.3f}",
            f"Nearest rig kept    {fit['nearest_neighbour_preserved']}/{len(rigs)}",
        ],
        title="Cruise operating speed and joint motor dynamics",
        axis_labels=("Metric MDS 1", "Metric MDS 2"),
        dpi=MDS_DPI,
    )
    return {
        "png": str(path),
        "dpi": MDS_DPI,
        "coords": {r: xy[i].tolist() for i, r in enumerate(rigs)},
        **fit,
        "method": "Metric MDS (SMACOF, classical initialization); distances may shrink or expand.",
    }


def analyze(rows: list[dict], features: np.ndarray, out: Path) -> dict[str, Any]:
    """Same-rig vs other-rig spectral distances, LOFO rig ID and the rig map.

    Writes :data:`CENTROID_CSV` and :data:`MDS_PNG` into ``out`` and returns a
    JSON-serializable summary (see the module docstring for every definition).
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != len(rows) or x.shape[0] == 0:
        raise ValueError(f"features {x.shape} do not match {len(rows)} rows")
    if not np.isfinite(x).all():
        raise ValueError("features contain non-finite values")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n, p = x.shape

    win_rig = [str(r["rig"]) for r in rows]
    win_flight = [(str(r["rig"]), str(r["flight"])) for r in rows]
    starts = np.array([float(r["start_s"]) for r in rows])
    rigs = sorted(set(win_rig))
    rig_ix = {r: i for i, r in enumerate(rigs)}
    flights = sorted(set(win_flight))
    flight_ix = {f: i for i, f in enumerate(flights)}
    fidx = np.array([flight_ix[f] for f in win_flight])
    members = [np.flatnonzero(fidx == f) for f in range(len(flights))]
    flight_rig = [f[0] for f in flights]
    flights_of = {r: [f for f in range(len(flights)) if flight_rig[f] == r] for r in rigs}
    eligible = [r for r in rigs if len(flights_of[r]) >= 2]
    single = [r for r in rigs if len(flights_of[r]) < 2]
    n_rigs = len(rigs)

    counts = {
        r: {
            "n_flights": len(flights_of[r]),
            "n_windows": int(sum(members[f].size for f in flights_of[r])),
            "sources": sorted({str(rows[i]["source"]) for f in flights_of[r] for i in members[f]}),
            "independent_target": r in eligible,
        }
        for r in rigs
    }

    m, cond = _segment_distances(x, members)
    npz_path = out / SEGMENT_NPZ
    np.savez_compressed(
        npz_path,
        distance_condensed=cond,
        rig=np.array(win_rig),
        flight=np.array([f[1] for f in win_flight]),
        start_s=starts,
        source=np.array([str(r["source"]) for r in rows]),
    )
    del cond
    flight_mean = np.stack([x[idx].mean(axis=0) for idx in members])

    # ---- within-record (all rigs) -----------------------------------------
    per_flight: list[dict[str, Any]] = []
    rec_pairs: dict[str, list[np.ndarray]] = {r: [] for r in rigs}
    rec_flight_med: dict[str, list[float]] = {r: [] for r in rigs}
    for f, idx in enumerate(members):
        rig, fl = flights[f]
        d_rec = pdist(x[idx]) if idx.size >= 2 else np.empty(0)
        rec_pairs[rig].append(d_rec)
        if d_rec.size:
            rec_flight_med[rig].append(float(np.median(d_rec)))
        per_flight.append(
            {
                "rig": rig,
                "flight": fl,
                "n_windows": int(idx.size),
                "start_span_s": _num(np.ptp(starts[idx])) if idx.size else None,
                "within_record_median": _num(np.median(d_rec)) if d_rec.size else None,
                "within_record_pairs": int(d_rec.size),
            }
        )
    within_record = {}
    for r in rigs:
        pooled = np.concatenate(rec_pairs[r]) if rec_pairs[r] else np.empty(0)
        within_record[r] = {
            "n_flights": len(flights_of[r]),
            "n_pairs": int(pooled.size),
            "median_pooled": _num(np.median(pooled)) if pooled.size else None,
            "median_of_flight_medians": _num(np.median(rec_flight_med[r]))
            if rec_flight_med[r]
            else None,
        }

    # ---- same rig (other flights) vs other rigs, and LOFO identification ---
    conf = np.zeros((len(eligible), n_rigs))
    per_rig: dict[str, dict[str, Any]] = {}
    for e, r in enumerate(eligible):
        ri = rig_ix[r]
        flight_rows = []
        within_all, between_all, recs = [], [], []
        to_rig_all: dict[str, list[np.ndarray]] = {s: [] for s in rigs if s != r}
        for f in flights_of[r]:
            idx = members[f]
            same = [g for g in flights_of[r] if g != f]
            a = m[np.ix_(idx, np.asarray(same))]  # (k, a)
            within = a.mean(axis=1)
            others = [s for s in rigs if s != r]
            to_rig = {s: m[np.ix_(idx, np.asarray(flights_of[s]))].mean(axis=1) for s in others}
            between = np.mean(np.stack([to_rig[s] for s in others]), axis=0)
            rig_win = np.mean(np.stack([_win(within, to_rig[s]) for s in others]), axis=0)
            auc = np.zeros(idx.size)
            for s in others:
                b = m[np.ix_(idx, np.asarray(flights_of[s]))]  # (k, b)
                auc += _win(a[:, :, None], b[:, None, :]).mean(axis=(1, 2)) / len(others)

            # Leave the whole target flight out of every centroid.
            cents = np.stack(
                [flight_mean[[g for g in flights_of[s] if g != f]].mean(axis=0) for s in rigs]
            )
            dw = cdist(x[idx], cents)
            pred = np.argmin(dw, axis=1)
            rank = 1 + (dw < dw[:, [ri]]).sum(axis=1)
            fpred = int(np.argmin(cdist(flight_mean[[f]], cents)[0]))
            conf[e] += np.bincount(pred, minlength=n_rigs) / idx.size

            within_all.append(within)
            between_all.append(between)
            recs.append(m[idx, f])
            for s in others:
                to_rig_all[s].append(to_rig[s])
            row = {
                "auc": float(auc.mean()),
                "rig_win_fraction": float(rig_win.mean()),
                "within_median": float(np.median(within)),
                "between_median": float(np.median(between)),
                "id_window_accuracy": float((pred == ri).mean()),
                "id_window_mean_rank": float(rank.mean()),
                "id_flight_pred": rigs[fpred],
                "id_flight_correct": fpred == ri,
            }
            flight_rows.append(row)
            per_flight[f].update(
                {k: (_num(v) if isinstance(v, float) else v) for k, v in row.items()}
            )
        conf[e] /= len(flights_of[r])
        within_cat = np.concatenate(within_all)
        between_cat = np.concatenate(between_all)
        rec_cat = np.concatenate(recs)
        rec_cat = rec_cat[np.isfinite(rec_cat)]
        per_rig[r] = {
            "n_target_flights": len(flight_rows),
            "n_target_windows": int(within_cat.size),
            "auc": _mean([fr["auc"] for fr in flight_rows]),
            "rig_win_fraction": _mean([fr["rig_win_fraction"] for fr in flight_rows]),
            "flights_auc_above_half": int(sum(fr["auc"] > 0.5 for fr in flight_rows)),
            "within_median": _num(np.median(within_cat)),
            "between_median": _num(np.median(between_cat)),
            "within_record_median": _num(np.median(rec_cat)) if rec_cat.size else None,
            "between_over_within": _num(np.median(between_cat) / np.median(within_cat)),
            "to_rig_median": {s: _num(np.median(np.concatenate(v))) for s, v in to_rig_all.items()},
            "id_window_accuracy": _mean([fr["id_window_accuracy"] for fr in flight_rows]),
            "id_window_mean_rank": _mean([fr["id_window_mean_rank"] for fr in flight_rows]),
            "id_flight_accuracy": _mean([float(fr["id_flight_correct"]) for fr in flight_rows]),
            "id_flights_correct": int(sum(fr["id_flight_correct"] for fr in flight_rows)),
        }
        pr = per_rig[r]
        pr["separated"] = bool(
            pr["auc"] is not None
            and pr["auc"] > 0.5
            and 2 * pr["id_flights_correct"] > pr["n_target_flights"]
        )

    def macro(key: str) -> float | None:
        return _mean([per_rig[r][key] for r in eligible])

    target_flights = [pf for pf in per_flight if pf["rig"] in eligible]

    def flight_pooled(key: str) -> float | None:
        return _mean([float(pf[key]) for pf in target_flights])

    same_vs_other = {
        "unit": "independent flight; windows averaged per target flight, flights per rig, rigs macro",
        "targets": eligible,
        "comparators": rigs,
        "macro": {
            "auc": macro("auc"),
            "rig_win_fraction": macro("rig_win_fraction"),
            "within_median": macro("within_median"),
            "between_median": macro("between_median"),
            "between_over_within": macro("between_over_within"),
        },
        "flight_pooled": {
            "auc": flight_pooled("auc"),
            "rig_win_fraction": flight_pooled("rig_win_fraction"),
        },
        "flights_auc_above_half": int(sum(pf["auc"] > 0.5 for pf in target_flights)),
        "n_target_flights": len(target_flights),
        "n_target_windows": int(sum(pf["n_windows"] for pf in target_flights)),
        "per_rig": {
            r: {k: v for k, v in per_rig[r].items() if not k.startswith("id_")} for r in eligible
        },
    }
    identification = {
        "method": "nearest rig centroid (mean of flight-mean features), entire target flight left out",
        "chance": 1.0 / n_rigs,
        "candidates": rigs,
        "targets": eligible,
        "excluded_targets_single_flight": single,
        "macro_window_accuracy": macro("id_window_accuracy"),
        "macro_flight_accuracy": macro("id_flight_accuracy"),
        "macro_window_mean_rank": macro("id_window_mean_rank"),
        "flights_correct": int(sum(bool(pf["id_flight_correct"]) for pf in target_flights)),
        "n_target_flights": len(target_flights),
        "per_rig": {
            r: {
                "n_target_flights": per_rig[r]["n_target_flights"],
                "n_target_windows": per_rig[r]["n_target_windows"],
                "window_accuracy": per_rig[r]["id_window_accuracy"],
                "window_mean_rank": per_rig[r]["id_window_mean_rank"],
                "flight_accuracy": per_rig[r]["id_flight_accuracy"],
                "flights_correct": per_rig[r]["id_flights_correct"],
            }
            for r in eligible
        },
        "confusion_flight_balanced": {
            r: {s: _num(conf[e, rig_ix[s]]) for s in rigs} for e, r in enumerate(eligible)
        },
    }

    # ---- rig centroid map ----------------------------------------------------
    rig_centroid = np.stack([flight_mean[flights_of[r]].mean(axis=0) for r in rigs])
    d_true = cdist(rig_centroid, rig_centroid)
    csv_path = out / CENTROID_CSV
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rig", *rigs])
        for r, drow in zip(rigs, d_true):
            w.writerow([r, *(f"{v:.6g}" for v in drow)])

    mds: dict[str, Any] = {"png": None}
    files = {"centroid_csv": str(csv_path), "mds_png": None, "segment_npz": str(npz_path)}
    if n_rigs >= 3:
        mds = render_centroids(d_true, rigs, counts, out / MDS_PNG)
        files["mds_png"] = mds["png"]

    return {
        "distance": "Quadrature of calibrated four-motor mean-speed RMS and complex spectral distance",
        "counts": {
            "n_windows": n,
            "n_features": p,
            "n_rigs": n_rigs,
            "n_flights": len(flights),
            "per_rig": counts,
        },
        "single_flight_rigs": single,
        "notes": [
            "single-flight rigs are never independent targets (no other flight of theirs "
            "exists); they stay as comparators and candidate answers",
            "within-record distances compare distinct non-overlapping windows, "
            "not independent recordings",
        ],
        "same_vs_other": same_vs_other,
        "identification": identification,
        "rig_separation": {
            "criterion": "rig separated iff balanced segment win (same_vs_other auc, ties 1/2) > 0.5 "
            "AND a strict majority of its held-out flights is classified correctly by the "
            "flight-level LOFO nearest centroid",
            "per_rig": {r: per_rig[r]["separated"] for r in eligible},
            "n_separated": int(sum(per_rig[r]["separated"] for r in eligible)),
            "n_evaluable": len(eligible),
            "not_evaluable": single,
        },
        "within_record": within_record,
        "per_flight": per_flight,
        "centroid_distance": {
            "rigs": rigs,
            "matrix": [[_num(v) for v in drow] for drow in d_true],
            "definition": "Euclidean distance between rig centroids = mean of the rig's flight-mean features",
            "csv": str(csv_path),
        },
        "segment_distance": {
            "npz": str(npz_path),
            "layout": "distance_condensed (float32, scipy squareform order) over the input rows "
            "in their given order, plus per-row rig, flight, start_s, source",
        },
        "mds": mds,
        "files": files,
    }
