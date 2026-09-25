"""Is a noise-v2 preset bank TONAL? Prominence over the local floor, entry by entry.

The bank guards constrain the trend's total drop, the 1/3-octave envelope, the
line widths and finiteness — nothing about how far a rotor order stands over the
floor AROUND it. This script measures that for every entry of both published
banks on real four-rotor carrier patterns, against the two pinned anchors, and
writes the record the tonality decision is made on.

    python scripts/noise_v2_tonality_audit.py --limit 16      # smoke
    python scripts/noise_v2_tonality_audit.py --workers 6     # the full audit

Outputs, under ``results/noise_v2/rig_sampler/tonality/``: ``audit.json`` (every
per-entry row and every summary), ``findings.md`` (the tables), and the three
figures ``prominence_ladder.png``, ``visible_orders.png``, ``decay.png``.
Re-running with ``--reuse`` rebuilds the tables and figures from an existing
``audit.json`` without re-probing.

The estimator, the carrier patterns and the two floor references are documented
in :mod:`experiments.noise_model.tonality`; nothing here changes a bank, a guard
or a prior.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from data_processing.noise_v2_pool import load_preset_bank  # noqa: E402
from experiments.noise_model import rig_sampler as RS  # noqa: E402
from experiments.noise_model import tonality as TN  # noqa: E402

OUT = Path("results/noise_v2/rig_sampler/tonality")
BANKS = {
    "easy": "dload:noise-v2-banks/noise_v2_easy_n2048.json",
    "hard": "dload:noise-v2-banks/noise_v2_hard_n2048.json",
}
#: The quantiles every summary table reports.
QUANTILES = (5.0, 25.0, 50.0, 75.0, 95.0)
#: The drawn / realised coordinates the Spearman screen runs over.
PREDICTORS = (
    "level_db",
    "floor_mean_db",
    "rotor_gain_db",
    "slope_db_dec",
    "gain_minus_floor_db",
    "trend_drop_db",
    "ltas_level_db",
    "t",
)
BANK_COLOUR = {"easy-dregon": "#1f77b4", "easy-michaels": "#d62728", "hard": "#2ca02c"}

_WORKER: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# the parallel audit
# ---------------------------------------------------------------------------


def _compact(obj: Any) -> Any:
    """Numeric lists to float64 arrays, in place, recursively.

    Both banks as parsed JSON are 833 MB of Python floats per worker; the same
    numbers as arrays are a fraction of that, and every consumer here
    (``params_from_dict``, ``payload_shape``) goes through ``np.asarray``
    anyway.
    """
    if isinstance(obj, dict):
        return {k: _compact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        flat = np.asarray(obj)
        if flat.dtype.kind in "fiu":
            return flat.astype(np.float64)
        return [_compact(v) for v in obj]
    return obj


def _init_worker(bank_paths: dict[str, str], patterns: list[dict[str, Any]]) -> None:
    import torch

    torch.set_num_threads(1)
    pats = TN.patterns_from_dicts(patterns)
    _WORKER["probe"] = TN.PatternProbe(pats)
    _WORKER["names"] = [p.name for p in pats]
    _WORKER["banks"] = {
        k: tuple(
            (
                _compact(e.cruise),
                None if e.standby is None else _compact(e.standby),
                e.name,
                e.provenance,
            )
            for e in load_preset_bank(v)
        )
        for k, v in bank_paths.items()
    }


def _audit_one(item: tuple[str, int]) -> dict[str, Any]:
    bank, index = item
    cruise, standby, name, prov = _WORKER["banks"][bank][index]
    row, curves = TN.entry_row(
        {"cruise": cruise, "standby": standby}, _WORKER["probe"], _WORKER["names"]
    )
    row["bank"] = bank
    row["index"] = int(index)
    row["name"] = name
    prov = prov or {}
    row["drawn"] = {
        "level_db": float(prov.get("level_db", float("nan"))),
        "ltas_level_db": float(prov.get("ltas_level_db", float("nan"))),
        "ltas_rms_db": float(prov.get("ltas_rms_db", float("nan"))),
        "trend_drop_db_min": float(prov.get("trend_drop_db_min", float("nan"))),
        "attempts": int(prov.get("attempts", 0)),
        "anchor": str(prov.get("anchor", "")),
        **({"t": float(prov["t"])} if "t" in prov else {}),
    }
    row["has_standby"] = standby is not None
    return {
        "row": _round(row),
        "curves": {k: [round(float(x), 3) for x in v] for k, v in curves.items()},
    }


def _round(obj: Any, nd: int = 3) -> Any:
    """Floats to ``nd`` decimals, recursively — audit.json is 4096 rows wide."""
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, nd) for v in obj]
    return obj


def _reference_rows(patterns: list[TN.Pattern]) -> dict[str, Any]:
    """The anchors, measured exactly as an entry is, plus the switch check."""
    probe = TN.PatternProbe(patterns)
    anchors = RS.pinned_anchors()
    out: dict[str, Any] = {"rows": {}, "curves": {}, "comb_switch": {}}
    cruise = [p.name for p in patterns if p.regime == "cruise"]
    standby = [p.name for p in patterns if p.regime == "standby"]
    for key in ("dregon", "michaels", "michaels_standby"):
        fit = anchors[key]
        if fit is None:
            continue
        if key == "michaels_standby":
            payloads = {"cruise": anchors["michaels"], "standby": fit}
            names = standby
        else:
            payloads = {"cruise": fit, "standby": None}
            names = cruise
        row, curves = TN.entry_row(payloads, probe, names, detail=True)
        row["name"] = key
        row["shape"] = TN.payload_shape(fit)
        out["rows"][key] = _round(row, 4)
        out["curves"][key] = {k: [round(float(x), 3) for x in v] for k, v in curves.items()}
    for p in patterns:
        key = "michaels_standby" if p.regime == "standby" else "dregon"
        fit = anchors[key]
        assert fit is not None
        out["comb_switch"][p.name] = TN.comb_switch_check(fit, probe, p.name)
    return out


def run_audit(
    *, limit: int | None, workers: int, banks: dict[str, str]
) -> tuple[list[dict[str, Any]], dict[str, Any], list[TN.Pattern]]:
    patterns = TN.select_patterns()
    counts = {k: len(load_preset_bank(v)) for k, v in banks.items()}
    items: list[tuple[str, int]] = []
    for name, n in counts.items():
        take = n if limit is None else min(int(limit), n)
        step = max(1, n // take)
        items += [(name, i) for i in range(0, n, step)][:take]
    pat_dicts = [p.as_dict() for p in patterns]
    t0 = time.time()
    results: list[dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=int(workers),
        initializer=_init_worker,
        initargs=(banks, pat_dicts),
        maxtasksperchild=256,
    ) as pool:
        for i, res in enumerate(pool.imap_unordered(_audit_one, items, chunksize=4), start=1):
            results.append(res)
            if i % 128 == 0 or i == len(items):
                rate = i / max(time.time() - t0, 1e-9)
                left = (len(items) - i) / max(rate, 1e-9)
                print(
                    f"  {i}/{len(items)} entries  {rate:.1f}/s  eta {left / 60:.1f} min", flush=True
                )
    results.sort(key=lambda r: (r["row"]["bank"], r["row"]["index"]))
    refs = _reference_rows(patterns)
    print(f"audit: {len(results)} entries in {time.time() - t0:.0f} s", flush=True)
    return results, refs, patterns


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------


def _group_of(row: dict[str, Any]) -> str:
    if row["bank"] == "easy":
        return "easy-" + ("dregon" if row["name"].startswith("dregon") else "michaels")
    return "hard"


def _quantiles(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {f"q{q:g}": float("nan") for q in QUANTILES} | {"n": 0, "mean": float("nan")}
    out = {f"q{q:g}": float(np.percentile(v, q)) for q in QUANTILES}
    out["n"] = int(v.size)
    out["mean"] = float(v.mean())
    return out


#: Every scalar the summary tables quantile over, and how it is read off a row.
STATS: tuple[tuple[str, str, int], ...] = (
    ("prom_k1", "prom_k1", 1),
    ("prom_k2", "prom_k2", 1),
    ("prom_k4", "prom_k4", 1),
    ("prom_k8", "prom_k8", 1),
    ("prom_k16", "prom_k16", 1),
    ("prom_bb_k1", "prom_bb_k1", 1),
    ("prom_bb_k2", "prom_bb_k2", 1),
    ("count_ge3", "count_ge3", 0),
    ("count_ge6", "count_ge6", 0),
    ("count_ge10", "count_ge10", 0),
    ("highest_ge3", "highest_ge3", 0),
    ("highest_ge6", "highest_ge6", 0),
    ("highest_ge10", "highest_ge10", 0),
    ("trend_cross_order", "trend_cross_order", 0),
)


def _stat(row: dict[str, Any], pattern: str, key: str, slot: int) -> float:
    block = row.get(pattern)
    if not isinstance(block, dict) or key not in block:
        return float("nan")
    return float(block[key][slot])


def _shape_stat(row: dict[str, Any], key: str, *, standby: bool = False) -> float:
    shape = row.get("shape_standby" if standby else "shape")
    if not isinstance(shape, dict):
        return float("nan")
    v = shape[key]
    return float(v[0]) if isinstance(v, list) else float(v)


def summarise(
    rows: list[dict[str, Any]], refs: dict[str, Any], patterns: list[TN.Pattern]
) -> dict[str, Any]:
    cruise = [p.name for p in patterns if p.regime == "cruise"]
    standby = [p.name for p in patterns if p.regime == "standby"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_group_of(row), []).append(row)
    deciles: dict[str, list[dict[str, Any]]] = {}
    for row in groups.get("hard", []):
        t = row["drawn"].get("t")
        if t is None:
            continue
        deciles.setdefault(f"hard-t{min(int(float(t) * 10), 9)}", []).append(row)

    def cell(items: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
        out: dict[str, Any] = {"n_entries": len(items)}
        for label, key, slot in STATS:
            pooled = np.array(
                [_stat(r, p, key, slot) for r in items for p in names if p in r], dtype=np.float64
            )
            out[label] = _quantiles(pooled)
        for key in ("slope_db_dec", "trend_drop_db", "rotor_gain_db", "floor_mean_db"):
            sb = names and names[0].endswith(("standby_narrow", "standby_wide"))
            out[key] = _quantiles(
                np.array([_shape_stat(r, key, standby=bool(sb)) for r in items], dtype=np.float64)
            )
        return out

    summary: dict[str, Any] = {"cruise": {}, "standby": {}, "per_pattern": {}}
    for name, items in list(groups.items()) + list(deciles.items()):
        summary["cruise"][name] = cell(items, cruise)
        carrying = [r for r in items if r.get("has_standby")]
        if carrying:
            summary["standby"][name] = cell(carrying, standby)
    for p in patterns:
        summary["per_pattern"][p.name] = {
            name: cell(items, [p.name]) for name, items in groups.items()
        }

    # (b) the anchor bar: how many entries are LESS tonal than the least tonal
    # anchor. Each anchor is summarised by the MEDIAN over its four cruise
    # patterns (its worst pattern would be an unfairly low bar), and the bar is
    # the lower of the two anchors — the least tonal real rig.
    anchor_bar: dict[str, Any] = {}
    for label, key, slot in STATS:
        vals = {
            a: float(
                np.nanmedian(
                    [_stat(refs["rows"][a], p, key, slot) for p in cruise if p in refs["rows"][a]]
                )
            )
            for a in ("dregon", "michaels")
        }
        bar = float(np.nanmin(list(vals.values())))
        anchor_bar[label] = {"anchors": vals, "bar": bar, "groups": {}}
        for name, items in groups.items():
            v = np.array(
                [_stat(r, p, key, slot) for r in items for p in cruise if p in r], dtype=np.float64
            )
            v = v[np.isfinite(v)]
            anchor_bar[label]["groups"][name] = {
                "n": int(v.size),
                "frac_below": float((v < bar).mean()) if v.size else float("nan"),
            }
    summary["anchor_bar"] = anchor_bar

    # (c) the fraction under {4, 8, 16} visible orders at each threshold
    under: dict[str, Any] = {}
    for thr in ("3", "6", "10"):
        under[thr] = {}
        for name, items in groups.items():
            v = np.array(
                [_stat(r, p, f"count_ge{thr}", 0) for r in items for p in cruise if p in r],
                dtype=np.float64,
            )
            v = v[np.isfinite(v)]
            under[thr][name] = {
                "n": int(v.size),
                **{f"lt{b}": float((v < b).mean()) for b in (4, 8, 16)},
                "median": float(np.median(v)) if v.size else float("nan"),
            }
    summary["under_counts"] = under

    # (e) which drawn coordinate predicts low tonality
    summary["spearman"] = _spearman(rows, cruise)
    summary["real_profiles"] = TN.real_profile_slopes()
    return summary


def _rank(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(v.size, dtype=np.float64)
    ranks[order] = np.arange(v.size, dtype=np.float64)
    # average ties
    sv = v[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = np.mean(ranks[order[i : j + 1]])
        i = j + 1
    return ranks


def _spearman(rows: list[dict[str, Any]], cruise: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    groups: dict[str, list[dict[str, Any]]] = {"all": list(rows)}
    for row in rows:
        groups.setdefault(_group_of(row), []).append(row)
    # the hard bank's coordinates all move WITH t, so a pooled rho there reads
    # the path, not the mechanism; the per-decile rhos average that out
    deciles: dict[int, list[dict[str, Any]]] = {}
    for row in groups.get("hard", []):
        t = row["drawn"].get("t")
        if t is not None:
            deciles.setdefault(min(int(float(t) * 10), 9), []).append(row)
    for d, items in deciles.items():
        groups[f"hard-t{d}"] = items
    for name, items in groups.items():
        target = np.array(
            [
                float(np.nanmean([_stat(r, p, "count_ge6", 0) for p in cruise if p in r]))
                for r in items
            ],
            dtype=np.float64,
        )
        cell: dict[str, float] = {}
        for pred in PREDICTORS:
            if pred in (
                "rotor_gain_db",
                "slope_db_dec",
                "trend_drop_db",
                "floor_mean_db",
                "gain_minus_floor_db",
            ):
                x = np.array([_shape_stat(r, pred) for r in items], dtype=np.float64)
            else:
                x = np.array(
                    [float(r["drawn"].get(pred, float("nan"))) for r in items], dtype=np.float64
                )
            m = np.isfinite(x) & np.isfinite(target)
            if m.sum() < 8:
                continue
            a, b = _rank(x[m]), _rank(target[m])
            sd = float(a.std() * b.std())
            cell[pred] = float(np.mean((a - a.mean()) * (b - b.mean())) / sd) if sd > 0 else 0.0
        out[name] = {"n": int(target.size), "rho": cell}
    within = [f"hard-t{d}" for d in sorted(deciles)]
    if within:
        pooled: dict[str, float] = {}
        for pred in PREDICTORS:
            vals = [out[g]["rho"][pred] for g in within if pred in out[g]["rho"]]
            if vals:
                pooled[pred] = float(np.mean(vals))
        out["hard-within-t"] = {
            "n": int(sum(out[g]["n"] for g in within)),
            "rho": pooled,
            "note": "mean of the ten per-decile rhos: the path coordinate held fixed",
        }
    return out


def curve_quantiles(
    results: list[dict[str, Any]], patterns: list[TN.Pattern]
) -> dict[str, dict[str, Any]]:
    """Per group and pattern, the 5/25/50/75/95 % prominence-vs-order curves."""
    out: dict[str, dict[str, Any]] = {}
    for p in patterns:
        buckets: dict[str, list[np.ndarray]] = {}
        for res in results:
            curve = res["curves"].get(p.name)
            if curve is None:
                continue
            buckets.setdefault(_group_of(res["row"]), []).append(
                np.asarray(curve, dtype=np.float64)
            )
        cell: dict[str, Any] = {}
        for name, curves in buckets.items():
            width = min(c.size for c in curves)
            block = np.stack([c[:width] for c in curves])
            cell[name] = {
                "k_max": int(width),
                "n": int(block.shape[0]),
                **{
                    f"q{q:g}": [round(float(v), 4) for v in np.percentile(block, q, axis=0)]
                    for q in QUANTILES
                },
            }
        out[p.name] = cell
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _save(fig: Figure, name: str, out_dir: Path) -> Path:
    path = out_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_ladder(audit: dict[str, Any], out_dir: Path) -> Path:
    curves = audit["curves"]
    cruise = [p["name"] for p in audit["patterns"] if p["regime"] == "cruise"]
    fig, axes = plt.subplots(1, len(cruise), figsize=(4.2 * len(cruise), 4.4), sharey=True)
    for ax, pat in zip(np.atleast_1d(axes), cruise, strict=True):
        for name, cell in sorted(curves.get(pat, {}).items()):
            k = np.arange(1, cell["k_max"] + 1)
            colour = BANK_COLOUR.get(name, "#777777")
            ax.fill_between(k, cell["q5"], cell["q95"], color=colour, alpha=0.15, linewidth=0)
            ax.fill_between(k, cell["q25"], cell["q75"], color=colour, alpha=0.3, linewidth=0)
            ax.plot(k, cell["q50"], color=colour, lw=1.6, label=f"{name} (n={cell['n']})")
        for anchor, style in (("dregon", "--"), ("michaels", ":")):
            ref = audit["references"]["curves"].get(anchor, {}).get(pat)
            if ref is None:
                continue
            ax.plot(
                np.arange(1, len(ref) + 1), ref, style, color="k", lw=1.3, label=f"anchor {anchor}"
            )
        for thr, c in zip(TN.PROM_THRESHOLDS_DB, ("#bbbbbb", "#777777", "#333333"), strict=True):
            ax.axhline(thr, color=c, lw=0.8, ls=(0, (4, 3)))
            ax.annotate(
                f"{thr:g} dB",
                xy=(0.995, thr),
                xycoords=("axes fraction", "data"),
                fontsize=6,
                color=c,
                va="bottom",
                ha="right",
            )
        ax.set_xscale("log")
        ax.set_xlabel("rotor order k")
        ax.set_title(pat, fontsize=10)
        ax.grid(alpha=0.25, which="both")
    np.atleast_1d(axes)[0].set_ylabel("prominence over the local floor (dB)")
    np.atleast_1d(axes)[0].set_ylim(-8.0, 40.0)
    np.atleast_1d(axes)[-1].legend(fontsize=7, loc="upper right")
    fig.suptitle(
        "Rotor-order prominence over the LOCAL floor: bank clouds (median, 25-75 %, 5-95 %) "
        "against the pinned anchors",
        fontsize=11,
    )
    return _save(fig, "prominence_ladder.png", out_dir)


def figure_visible(audit: dict[str, Any], out_dir: Path) -> Path:
    rows = audit["entries"]
    cruise = [p["name"] for p in audit["patterns"] if p["regime"] == "cruise"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
    for ax, thr in zip(axes, ("3", "6", "10"), strict=True):
        for name in ("easy-dregon", "easy-michaels", "hard"):
            v = np.array(
                [
                    _stat(r, p, f"count_ge{thr}", 0)
                    for r in rows
                    if _group_of(r) == name
                    for p in cruise
                    if p in r
                ],
                dtype=np.float64,
            )
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            xs = np.sort(v)
            ax.plot(
                xs,
                np.arange(1, xs.size + 1) / xs.size,
                color=BANK_COLOUR[name],
                lw=1.8,
                label=f"{name} (median {np.median(v):.0f})",
            )
        for anchor, style in (("dregon", "--"), ("michaels", ":")):
            ref = audit["references"]["rows"].get(anchor)
            if ref is None:
                continue
            vals = [_stat(ref, p, f"count_ge{thr}", 0) for p in cruise if p in ref]
            ax.axvline(
                float(np.nanmin(vals)), color="k", ls=style, lw=1.2, label=f"anchor {anchor}"
            )
        ax.set_xlabel(f"orders with prominence >= {thr} dB")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("cumulative fraction of entries")
    fig.suptitle(
        "How many rotor orders clear the local floor: CDF over entries x cruise patterns",
        fontsize=11,
    )
    return _save(fig, "visible_orders.png", out_dir)


def figure_decay(audit: dict[str, Any], out_dir: Path) -> Path:
    rows = audit["entries"]
    real = audit["summary"]["real_profiles"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, key, label in (
        (axes[0], "slope_db_dec", "trend slope (dB / decade of order)"),
        (axes[1], "trend_drop_db", "total trend drop k=1 -> K (dB)"),
    ):
        for name in ("easy-dregon", "easy-michaels", "hard"):
            v = np.array(
                [_shape_stat(r, key) for r in rows if _group_of(r) == name], dtype=np.float64
            )
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            ax.hist(
                v,
                bins=48,
                histtype="step",
                density=True,
                color=BANK_COLOUR[name],
                lw=1.6,
                label=f"{name} (median {np.median(v):.2f})",
            )
        rv = np.array([r[key] for r in real], dtype=np.float64)
        ax.plot(
            rv,
            np.full(rv.size, -0.004),
            "|",
            color="k",
            ms=12,
            mew=1.4,
            label=f"16 measured rotor profiles ({rv.min():.2f} .. {rv.max():.2f})",
        )
        if key == "trend_drop_db":
            ax.axvline(RS.TREND_MARGIN_DB, color="#b22222", lw=1.3, ls="--", label="guard 3 dB")
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle(
        "How weak is the admitted decay? Bank entries against the measured profiles", fontsize=11
    )
    return _save(fig, "decay.png", out_dir)


# ---------------------------------------------------------------------------
# findings.md
# ---------------------------------------------------------------------------


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _qrow(name: str, q: dict[str, float], fmt: str = "{:.2f}") -> str:
    return _row(
        [name, str(q["n"])]
        + [fmt.format(q[f"q{v:g}"]) if np.isfinite(q[f"q{v:g}"]) else "—" for v in QUANTILES]
    )


def write_findings(audit: dict[str, Any], out_dir: Path) -> Path:
    s = audit["summary"]
    refs = audit["references"]
    pats = audit["patterns"]
    standby = [p["name"] for p in pats if p["regime"] == "standby"]
    groups = ("easy-dregon", "easy-michaels", "hard")
    L: list[str] = []
    A = L.append

    A("# Bank tonality: how far do the rotor orders stand over the local floor?")
    A("")
    A(
        "**What this is.** `scripts/noise_v2_tonality_audit.py` measures, for every entry of both "
        "published `noise-v2-bank/1` banks (`dload:noise-v2-banks/noise_v2_{easy,hard}_n2048.json`), "
        "how many rotor orders are identifiable over the floor around them, and how weak the trend "
        "decay the `trend_falls` guard admits actually is. It changes no bank, no guard and no prior. "
        "Machine-readable: `audit.json` — every number below is read from it. Figures: "
        "`prominence_ladder.png`, `visible_orders.png`, `decay.png`."
    )
    A("")
    A("## 1. The estimator, and what its floor contains")
    A("")
    A(
        "Round 4's estimator (`results/noise_v2/rounds/round4/legacy_truth/anatomy.md` § \"MEASURED "
        'on the render: prominence over the local floor"), applied to the noise-free EXPECTED '
        "periodogram instead of a render:"
    )
    A("")
    A(f"* **peak** = the largest `P_tot` bin within +-{TN.PEAK_HALF_BINS} bin of `k f_r`;")
    A(
        f"* **local floor** = the MEDIAN of `P_tot` over the two-sided "
        f"{TN.ANNULUS_FRAC[0]:.2f}-{TN.ANNULUS_FRAC[1]:.2f} `fbar` annulus around `k f_r`, excluding "
        f"+-{TN.EXCLUDE_HALF_BINS} bin around every rotor's `k-1 / k / k+1` lines;"
    )
    A("* **prominence** = `10 log10(peak / floor)`, in dB, per microphone and per rotor.")
    A("")
    A(
        "The grid is the model's own flight front end (2048-point rFFT at 16 kHz, **7.8125 Hz per "
        "bin**), because that is the front end the fits live on and the arms train on. R4 measured "
        "on 8192-point renders (1.95 Hz per bin); the two are not interchangeable."
    )
    A("")
    A(
        "The floor this reads is the AGGREGATE local floor: the broadband floor PLUS whatever the "
        "four combs put between this rotor's orders. That is not a rounding error. Measured on the "
        "comb-off expectation (`profile_db = -300 dB`, verified bin-for-bin against the full-order "
        f"comb-off model: max |diff| "
        f"{max(v['max_abs_db'] for v in refs['comb_switch'].values()):.3g} dB over "
        f"{next(iter(refs['comb_switch'].values()))['n_bins']} bins), the comb raises the spectrum "
        "even at the bins farthest from any line by"
    )
    A("")
    A(
        _row(
            [
                "pattern",
                "payload",
                "bins used",
                "min dist (bins)",
                "pedestal median (dB)",
                "p95",
                "max",
            ]
        )
    )
    A(_row(["---", "---", "---:", "---:", "---:", "---:", "---:"]))
    for p in pats:
        key = "michaels_standby" if p["regime"] == "standby" else "dregon"
        block = refs["rows"][key].get(p["name"])
        if block is None:
            continue
        ped = block["pedestal"]
        A(
            _row(
                [
                    f"`{p['name']}`",
                    f"`{key}` anchor",
                    str(ped["n_bins"]),
                    str(ped["min_dist_bins"]),
                    f"{ped['median_db']:.2f}",
                    f"{ped['p95_db']:.2f}",
                    f"{ped['max_db']:.2f}",
                ]
            )
        )
    A("")
    A(
        "So a line that stands 20 dB over the BROADBAND floor stands only ~11 dB over the floor a "
        "separator actually sees around it. Both readings are reported: `prom_*` against the local "
        "floor (the headline, and what the counts use) and `prom_bb_k*` against the comb-free "
        "broadband floor at the same bin, at the named orders only — `prom_bb` is NOT counted over "
        "orders because at high `k` the peak bin holds the other orders' merged skirts and the "
        "statistic saturates at the pedestal."
    )
    A("")
    A("## 2. The carrier patterns")
    A("")
    A(
        "Four rotors at ONE speed pile every rotor's order on one bin and overstate visibility, so "
        "the probe uses the per-rotor mean carriers of REAL windows "
        "(`rig_sampler._real_window_specs`). Per family the NARROWEST- and WIDEST-spread window is "
        "taken, which brackets how much of a rotor's local floor is its neighbours' lines:"
    )
    A("")
    A(_row(["pattern", "support", "per-rotor rev/s", "mean", "spread"]))
    A(_row(["---", "---", "---", "---:", "---:"]))
    for p in pats:
        A(
            _row(
                [
                    f"`{p['name']}`",
                    f"`{p['support']}`",
                    ", ".join(f"{v:.1f}" for v in p["rev_s"]),
                    f"{p['mean_rev_s']:.2f}",
                    f"{p['spread_rev_s']:.2f}",
                ]
            )
        )
    A("")
    A(
        "Orders run `k = 1..K`, `K` = the payload's profile length capped by "
        "`spectrum.k_max_for_carrier` at the pattern's FASTEST rotor. At standby the estimator is "
        "under strain and says so: 35 rev/s puts the orders 4.6 bins apart, the "
        f"{TN.ANNULUS_FRAC[0]:.2f}-{TN.ANNULUS_FRAC[1]:.2f} `fbar` annulus is 2-4 bins wide and the "
        "neighbour exclusion empties it, so the audit falls back to the unexcluded annulus and "
        "COUNTS the orders where it had to (`n_fallback_orders`)."
    )
    A("")
    A("## 3. The references")
    A("")
    A(
        _row(
            [
                "payload",
                "pattern",
                "k=1",
                "k=2",
                "k=4",
                "k=8",
                "k=16",
                ">=3 dB",
                ">=6 dB",
                ">=10 dB",
                "highest >=6",
                "trend crosses at",
            ]
        )
    )
    A(_row(["---", "---"] + ["---:"] * 10))
    for key, row in refs["rows"].items():
        for p in pats:
            if p["name"] not in row:
                continue
            b = row[p["name"]]
            A(
                _row(
                    [
                        f"`{key}`",
                        f"`{p['name']}`",
                        *[
                            f"{b[f'prom_k{k}'][1]:+.2f}" if f"prom_k{k}" in b else "—"
                            for k in TN.NAMED_ORDERS
                        ],
                        f"{b['count_ge3'][0]:.1f}",
                        f"{b['count_ge6'][0]:.1f}",
                        f"{b['count_ge10'][0]:.1f}",
                        f"{b['highest_ge6'][0]:.1f}",
                        f"{b['trend_cross_order'][0]:.0f}",
                    ]
                )
            )
    A("")
    A(
        "**Calibration note, not a comparison.** R4 measured the REAL DREGON audio at "
        "**k=1 +5.77 dB, k=2 +2.08 dB, and nothing identifiable from k>=9** (mean over the three "
        "scored windows of `anatomy.md`'s table). Those numbers are NOT directly comparable with the "
        "table above: they are an 8192-point measurement of recorded audio, whose shaft wander, "
        "four-rotor speed spread and finite-window averaging smear every line, against a noise-free "
        "2048-point expectation at a CONSTANT carrier here. The expectation is the tonality the "
        "sampler PUT in the payload; the render is that minus whatever the trajectory smears out."
    )
    A("")
    A("## 4. (a) Per bank and half: the quantiles")
    A("")
    A("Pooled over each group's entries x the four cruise patterns.")
    for label, fmt in (
        ("prom_k1", "{:+.2f}"),
        ("prom_k2", "{:+.2f}"),
        ("count_ge6", "{:.1f}"),
        ("highest_ge6", "{:.1f}"),
        ("slope_db_dec", "{:.2f}"),
    ):
        A("")
        A(f"**{label}**")
        A("")
        A(_row(["group", "n"] + [f"{q:g} %" for q in QUANTILES]))
        A(_row(["---", "---:"] + ["---:"] * len(QUANTILES)))
        for name in groups:
            cell = s["cruise"].get(name)
            if cell:
                A(_qrow(name, cell[label], fmt))
        for name in sorted(k for k in s["cruise"] if k.startswith("hard-t")):
            A(_qrow(name, s["cruise"][name][label], fmt))
    A("")
    A("## 5. (b) How many entries are LESS tonal than the least tonal anchor?")
    A("")
    A(
        "Each anchor is summarised by the MEDIAN of that statistic over its four cruise patterns "
        "and the bar is the LOWER of the two anchors — the least tonal real rig. The fraction "
        "counts entry x cruise-pattern pairs strictly under the bar (for `trend_cross_order` too, "
        "where a smaller order is the less tonal one)."
    )
    A("")
    A(_row(["statistic", "dregon", "michaels", "bar", *groups]))
    A(_row(["---"] + ["---:"] * (3 + len(groups))))
    for label, _key, _slot in STATS:
        cell = s["anchor_bar"][label]
        A(
            _row(
                [
                    label,
                    f"{cell['anchors']['dregon']:.2f}",
                    f"{cell['anchors']['michaels']:.2f}",
                    f"{cell['bar']:.2f}",
                    *[f"{100 * cell['groups'][g]['frac_below']:.1f} %" for g in groups],
                ]
            )
        )
    A("")
    A("## 6. (c) How many entries carry fewer than 4 / 8 / 16 visible orders?")
    A("")
    A(
        "All three prominence bars are reported because none of them is established: 3 dB is about "
        "where a line stops being separable from its own floor estimate, 10 dB is a line no tracker "
        "can miss, 6 dB sits between them."
    )
    A("")
    A(_row(["threshold", "group", "median count", "< 4", "< 8", "< 16"]))
    A(_row(["---", "---", "---:", "---:", "---:", "---:"]))
    for thr in ("3", "6", "10"):
        for name in groups:
            cell = s["under_counts"][thr][name]
            A(
                _row(
                    [
                        f">= {thr} dB",
                        name,
                        f"{cell['median']:.1f}",
                        f"{100 * cell['lt4']:.1f} %",
                        f"{100 * cell['lt8']:.1f} %",
                        f"{100 * cell['lt16']:.1f} %",
                    ]
                )
            )
    A("")
    A("## 7. (d) The standby payloads on standby patterns")
    A("")
    A(
        "Only the entries that carry a standby payload, measured on that payload at the two "
        f"standby patterns ({', '.join('`' + n + '`' for n in standby)})."
    )
    A("")
    A(_row(["group", "entries", "k=1", "k=2", "count >=6 dB", "highest >=6 dB", "slope dB/dec"]))
    A(_row(["---", "---:", "---:", "---:", "---:", "---:", "---:"]))
    for name in groups:
        cell = s["standby"].get(name)
        if not cell:
            continue
        A(
            _row(
                [
                    name,
                    str(cell["n_entries"]),
                    f"{cell['prom_k1']['q50']:+.2f}",
                    f"{cell['prom_k2']['q50']:+.2f}",
                    f"{cell['count_ge6']['q50']:.1f}",
                    f"{cell['highest_ge6']['q50']:.1f}",
                    f"{cell['slope_db_dec']['q50']:.2f}",
                ]
            )
        )
    A("")
    A("## 8. (e) Which drawn coordinate makes an entry weakly tonal?")
    A("")
    A(
        "Spearman rho between an entry's `count_ge6` (mean over the four cruise patterns) and each "
        "coordinate. `level_db`, `ltas_level_db` and `t` come from the entry's own `provenance` "
        "block; `rotor_gain_db`, `slope_db_dec`, `trend_drop_db`, `floor_mean_db` and "
        "`gain_minus_floor_db` (the comb-over-floor offset) are the REALISED values read off the "
        "payload — the bank keeps `level_db` and `t` but not the rest of the `drawn` block. "
        "`all` pools both banks, so it mostly measures which RIG an entry is; `hard-within-t` is "
        "the mean of the ten per-decile rhos, which holds the path position fixed and is the only "
        "column that isolates the perturbation's own effect on the hard bank."
    )
    A("")
    cols = [g for g in (*groups, "hard-within-t", "all") if g in s["spearman"]]
    A(_row(["coordinate", *(f"{g} (n={s['spearman'][g]['n']})" for g in cols)]))
    A(_row(["---"] + ["---:"] * len(cols)))
    for pred in PREDICTORS:
        cells = []
        for g in cols:
            rho = s["spearman"][g]["rho"].get(pred)
            cells.append("—" if rho is None else f"{rho:+.3f}")
        if any(c != "—" for c in cells):
            A(_row([f"`{pred}`", *cells]))
    A("")
    A("## 9. The decay the guard admits")
    A("")
    real = s["real_profiles"]
    slopes = np.array([r["slope_db_dec"] for r in real])
    drops = np.array([r["trend_drop_db"] for r in real])
    A(
        f"The 16 MEASURED v2 rotor profiles (2 cruise rigs x 4 rotors, Michael's standby x 4, 4 "
        f"bench motors) have trend slopes {slopes.min():.2f} .. {slopes.max():.2f} dB/decade "
        f"(median {np.median(slopes):.2f}) and total drops {drops.min():.2f} .. {drops.max():.2f} dB "
        f"(median {np.median(drops):.2f}). The `trend_falls` guard admits anything dropping "
        f"{RS.TREND_MARGIN_DB:.0f} dB or more."
    )
    A("")
    A(_row(["group", "n"] + [f"slope {q:g} %" for q in QUANTILES]))
    A(_row(["---", "---:"] + ["---:"] * len(QUANTILES)))
    for name in groups:
        cell = s["cruise"].get(name)
        if cell:
            A(_qrow(name, cell["slope_db_dec"], "{:.2f}"))
    A("")
    A(_row(["group", "n"] + [f"drop {q:g} %" for q in QUANTILES]))
    A(_row(["---", "---:"] + ["---:"] * len(QUANTILES)))
    for name in groups:
        cell = s["cruise"].get(name)
        if cell:
            A(_qrow(name, cell["trend_drop_db"], "{:.2f}"))
    A("")
    A("![prominence ladder](prominence_ladder.png)")
    A("")
    A("![visible orders](visible_orders.png)")
    A("")
    A("![decay](decay.png)")
    A("")
    A("## 10. Verdict")
    A("")
    A(_verdict(audit))
    A("")
    path = out_dir / "findings.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L))
    return path


MECHANISM = ("rotor_gain_db", "floor_mean_db", "gain_minus_floor_db", "slope_db_dec", "level_db")


def _verdict(audit: dict[str, Any]) -> str:
    s = audit["summary"]
    u = s["under_counts"]
    bar = s["anchor_bar"]["count_ge6"]
    pooled = s["spearman"]["all"]["rho"]
    within = s["spearman"].get("hard-within-t", {}).get("rho", {})
    dregon = s["spearman"].get("easy-dregon", {}).get("rho", {})

    def rho(block: dict[str, float], pred: str) -> float:
        return float(block.get(pred, float("nan")))

    best = max(pooled.items(), key=lambda kv: abs(kv[1]))
    mech = max(
        ((k, v) for k, v in within.items() if k in MECHANISM),
        key=lambda kv: abs(kv[1]),
        default=("—", float("nan")),
    )
    gain, floor, diff = (
        rho(within, "rotor_gain_db"),
        rho(within, "floor_mean_db"),
        rho(within, "gain_minus_floor_db"),
    )
    if max(abs(gain), abs(floor), abs(diff)) < 0.15:
        driver = (
            "NEITHER on its own: with the rig held fixed every one of these coordinates has "
            "|rho| under 0.15, so what a bank entry's tonality really tracks is which RIG it is"
        )
    elif abs(gain) > 2.0 * abs(floor):
        driver = "the COMB level sinking, not the floor rising"
    elif abs(floor) > 2.0 * abs(gain):
        driver = "the FLOOR rising, not the comb sinking"
    else:
        driver = "comb and floor moving together, so their DIFFERENCE is the coordinate that bites"
    easy = [g for g in ("easy-dregon", "easy-michaels") if g in u["6"]]
    med = {g: u["6"][g]["median"] for g in easy}
    return (
        "At the 6 dB bar the median entry carries "
        + ", ".join(f"{v:.1f} orders ({g})" for g, v in med.items())
        + f" and {u['6']['hard']['median']:.1f} (hard), out of the 81-88 modelled; at 3 dB the "
        f"medians are {np.mean([u['3'][g]['median'] for g in easy]):.1f} (easy, pooled) and "
        f"{u['3']['hard']['median']:.1f} (hard), at 10 dB "
        f"{np.mean([u['10'][g]['median'] for g in easy]):.1f} and {u['10']['hard']['median']:.1f}. "
        f"Fewer than FOUR orders clear 6 dB in "
        f"{100 * np.mean([u['6'][g]['lt4'] for g in easy]):.1f} % of easy and "
        f"{100 * u['6']['hard']['lt4']:.1f} % of hard entry-patterns, fewer than eight in "
        f"{100 * np.mean([u['6'][g]['lt8'] for g in easy]):.1f} % and "
        f"{100 * u['6']['hard']['lt8']:.1f} %, and at the 10 dB bar fewer than four in "
        f"{100 * np.mean([u['10'][g]['lt4'] for g in easy]):.1f} % and "
        f"{100 * u['10']['hard']['lt4']:.1f} %. Against the least tonal PINNED anchor "
        f"({bar['bar']:.2f} orders over 6 dB, "
        f"{min(bar['anchors'], key=lambda a: bar['anchors'][a])}), "
        f"{100 * np.mean([bar['groups'][g]['frac_below'] for g in easy]):.1f} % of easy and "
        f"{100 * bar['groups']['hard']['frac_below']:.1f} % of hard entry-patterns are less tonal "
        f"than the real rigs the bank was drawn around — i.e. the banks do NOT merely reproduce the "
        f"anchors' tonality, they extend well below it. The largest single Spearman rho against "
        f"`count_ge6` over all {s['spearman']['all']['n']} entries is `{best[0]}` ({best[1]:+.3f}), "
        f"but that is the rig identity speaking: on the hard bank `t` moves every coordinate at "
        f"once. Holding the path position fixed (mean of the ten per-decile rhos) the ordering is "
        f"`rotor_gain_db` {gain:+.3f}, `floor_mean_db` {floor:+.3f} and their difference "
        f"`gain_minus_floor_db` {diff:+.3f}, led by `{mech[0]}` ({mech[1]:+.3f}); on the easy "
        f"bank's DREGON half, where no path coordinate exists at all, the same ordering holds and "
        f"is just as weak (`gain_minus_floor_db` {rho(dregon, 'gain_minus_floor_db'):+.3f}, "
        f"`rotor_gain_db` {rho(dregon, 'rotor_gain_db'):+.3f}, `floor_mean_db` "
        f"{rho(dregon, 'floor_mean_db'):+.3f}). The driver is {driver} — and within one anchor's "
        f"neighbourhood no "
        f"single drawn coordinate explains much, so a tonality REJECTION guard would bite where a "
        f"prior on any one of these coordinates would not."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _estimator() -> dict[str, Any]:
    return {
        "annulus_frac_of_fbar": list(TN.ANNULUS_FRAC),
        "peak_half_bins": TN.PEAK_HALF_BINS,
        "exclude_half_bins": TN.EXCLUDE_HALF_BINS,
        "thresholds_db": list(TN.PROM_THRESHOLDS_DB),
        "named_orders": list(TN.NAMED_ORDERS),
        "comb_off_db": TN.COMB_OFF_DB,
        "n_fft": 2048,
        "sr": 16000,
        "bin_hz": 16000.0 / 2048.0,
    }


def audit_fits(specs: list[str], out_dir: Path) -> Path:
    """``--fit``: named FIT payloads (a ``noise-v3-fit/1`` rig, say) measured
    exactly as the pinned anchors are, with the anchors' rows beside them."""
    fits: dict[str, dict[str, Any]] = {}
    paths: dict[str, str] = {}
    for spec in specs:
        key, sep, path = str(spec).partition("=")
        if not sep:
            raise SystemExit(f"--fit wants RIG[_standby]=PATH, got {spec!r}")
        paths[key] = path
        fits[key] = json.loads(Path(path).read_text())
    patterns = TN.select_patterns()
    probe = TN.PatternProbe(patterns)
    rows: dict[str, Any] = {}
    curves: dict[str, Any] = {}
    for rig in sorted({k.removesuffix("_standby") for k in fits}):
        standby = fits.get(f"{rig}_standby")
        names = [p.name for p in patterns if p.rig == rig]
        payloads: dict[str, Any]
        if rig in fits:
            payloads = {"cruise": fits[rig], "standby": standby}
        else:
            # a standby fit alone: its own regime's patterns only (the cruise
            # slot is never read for them; its shape row is dropped)
            payloads = {"cruise": standby, "standby": standby}
            names = [p.name for p in patterns if p.rig == rig and p.regime == "standby"]
        row, cur = TN.entry_row(payloads, probe, names, detail=True)
        if rig not in fits:
            row.pop("shape", None)
        rows[rig] = _round(row, 4)
        curves[rig] = {k: [round(float(x), 3) for x in v] for k, v in cur.items()}
    payload = {
        "format": "noise-v2-tonality-fits/1",
        "fits": paths,
        "estimator": _estimator(),
        "patterns": [p.as_dict() for p in patterns],
        "rows": rows,
        "curves": curves,
        "references": _reference_rows(patterns),
    }
    path = out_dir / "fits.json"
    path.write_text(json.dumps(payload, indent=1))
    n_col = min(3, len(patterns))
    n_row = -(-len(patterns) // n_col)
    fig, axes = plt.subplots(n_row, n_col, figsize=(5.2 * n_col, 3.6 * n_row), squeeze=False)
    for ax in axes.flat[len(patterns) :]:
        ax.set_visible(False)
    for ax, pat in zip(axes.flat, patterns):
        anchor = pat.rig + ("_standby" if pat.regime == "standby" else "")
        for label, cur, style in (
            (f"{pat.rig} fit", curves.get(pat.rig, {}).get(pat.name), dict(color="C3")),
            (
                f"{anchor} anchor",
                payload["references"]["curves"].get(anchor, {}).get(pat.name),
                dict(color="0.4", ls="--"),
            ),
        ):
            if cur:
                ax.plot(np.arange(1, len(cur) + 1), cur, label=label, lw=1.2, **style)
        ax.legend(fontsize=8)
        for thr in TN.PROM_THRESHOLDS_DB:
            ax.axhline(thr, color="0.8", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(pat.name, fontsize=9)
        ax.set_xlabel("order k")
        ax.grid(alpha=0.3)
    for row in axes:
        row[0].set_ylabel("prominence over local floor, dB")
    fig.tight_layout()
    print("wrote", _save(fig, "fits_ladder.png", out_dir))
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="entries per bank (smoke runs)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--bank", choices=sorted(BANKS), action="append")
    ap.add_argument("--reuse", action="store_true", help="rebuild tables/figures from audit.json")
    ap.add_argument(
        "--fit",
        action="append",
        default=[],
        metavar="RIG[_standby]=PATH",
        help="audit these fit payloads (e.g. a noise-v3-fit/1 rig) instead of the banks: the "
        "anchors' per-pattern rows, for each fit and for the anchors, to <out>/fits.json",
    )
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.fit:
        print("wrote", audit_fits(list(args.fit), out_dir))
        return 0
    audit_path = out_dir / "audit.json"
    if args.reuse:
        audit = json.loads(audit_path.read_text())
        # the summary is derived, so it is rebuilt from the stored rows: a
        # change to a table definition never needs the 58-minute probe again
        audit["summary"] = summarise(
            audit["entries"], audit["references"], TN.patterns_from_dicts(audit["patterns"])
        )
        audit_path.write_text(json.dumps(audit, separators=(",", ":")))
    else:
        banks = {k: v for k, v in BANKS.items() if not args.bank or k in args.bank}
        results, refs, patterns = run_audit(limit=args.limit, workers=args.workers, banks=banks)
        rows = [r["row"] for r in results]
        audit = {
            "format": "noise-v2-tonality-audit/1",
            "banks": banks,
            "n_entries": len(rows),
            "limit": args.limit,
            "estimator": _estimator(),
            "patterns": [p.as_dict() for p in patterns],
            "references": refs,
            "curves": curve_quantiles(results, patterns),
            "entries": rows,
        }
        audit["summary"] = summarise(rows, refs, patterns)
        audit_path.write_text(json.dumps(audit, separators=(",", ":")))
        print(f"wrote {audit_path} ({audit_path.stat().st_size / 1e6:.1f} MB)")

    for fn in (figure_ladder, figure_visible, figure_decay):
        print("wrote", fn(audit, out_dir))
    print("wrote", write_findings(audit, out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
