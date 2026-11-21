#!/usr/bin/env python3
"""Flight-regime campaign driver: fit, score, aggregate.

Thin CLI over :mod:`experiments.rps_traj.regime_campaign`; the write-up is
``docs/experiments/rps-flight-regimes.md``.  One unit = one JSON under
``<out>/raw`` (``utils.gridrun``: restartable, a failed unit leaves an
``.err`` and the rest go on).

    # held-out comparison of the cruise models, every rig
    python scripts/rps_traj_regimes.py run --stage cruise --jobs 4
    # whole flights on the ground-to-ground rigs (reads the cruise units)
    python scripts/rps_traj_regimes.py run --stage flight --jobs 4
    # the per-drone parameter sets: every flight, no held-out score
    python scripts/rps_traj_regimes.py run --stage cruise --final
    python scripts/rps_traj_regimes.py run --stage flight --final --arms ours
    # tables, figure and params/<rig>.json from whatever has run
    python scripts/rps_traj_regimes.py table
    # does the Gaussian fit run faster on a GPU?
    python scripts/rps_traj_regimes.py bench --rigs michaels pitcn_quad
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if (_HERE.parent / "src").is_dir() else Path.cwd().resolve()
sys.path.insert(0, str(_ROOT / "src"))

from experiments.rps_traj import phase_dynamics as pd_  # noqa: E402  (after the path pin)
from experiments.rps_traj import regime_campaign as rc  # noqa: E402
from utils.gridrun import run_grid  # noqa: E402

DEFAULT_OUT = "results/rps_traj/regimes"


def _run(args: argparse.Namespace) -> int:
    rigs = args.rigs or list(rc.RIGS)
    worker = rc.run_unit
    if args.stage == "cruise":
        units = rc.cruise_units(
            rigs, args.families or rc.FAMILIES, final=args.final, threads=args.threads
        )
    elif args.stage == "flight":
        units = rc.flight_units(
            rigs, args.arms or rc.FLIGHT_ARMS, args.out, final=args.final, threads=args.threads
        )
    else:  # dynamics inside each phase, on the held-out flight's own schedule
        units = pd_.units(rigs, args.arms or pd_.ARMS, args.out, threads=args.threads)
        worker = pd_.run_unit
    if args.folds:
        keep = set(args.folds)
        units = [u for u in units if str(u.params["fold"]) in keep]
    result = run_grid(
        units,
        worker,
        args.out,
        jobs=args.jobs,
        resume=not args.no_resume,
        blas_threads=args.threads,
        mp_context="spawn",
        summarize=lambda rows: {"n_units": len(rows)},
        summary_name="units.json",
    )
    return result.exit_code


def _figure(summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cruise = summary["cruise"]
    rigs = [r for r in rc.RIGS if r in cruise]
    keys = [
        "logpdf",
        "es_skill_0.2",
        "es_skill_1",
        "inc_w1_0.2",
        "inc_w1_1",
        "inc_kurt_diff_1",
        "acf_rmse",
        "psd_lsd_db",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.4))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, key in zip(axes.ravel(), keys):
        label, higher = rc.CRUISE_METRICS[key]
        for a, arm in enumerate(rc.CRUISE_ARMS):
            xs, ys, es = [], [], []
            for i, rig in enumerate(rigs):
                stat = cruise[rig].get(arm, {}).get("mean", {}).get(key)
                if not stat or not stat["n"]:
                    continue
                ref = cruise[rig]["gauss"]["mean"][key]["mean"] if key == "logpdf" else 0.0
                xs.append(i + (a - 3.5) * 0.09)
                ys.append(stat["mean"] - ref)
                es.append(stat["sd"] / max(np.sqrt(stat["n"]), 1.0))
            marker = "o" if a < 3 else "s"
            ax.errorbar(xs, ys, yerr=es, fmt=marker, ms=5, lw=0.8, color=colors[a], label=arm)
        title = label + (" (minus gauss)" if key == "logpdf" else "")
        ax.set_title(f"{title}\n{'higher' if higher else 'lower'} is better", fontsize=9)
        ax.set_xticks(range(len(rigs)))
        ax.set_xticklabels(
            [r.replace("_quad", "").replace("_cf21b", "") for r in rigs], rotation=25, fontsize=8
        )
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(
        "Held-out cruise scores: mean over folds, bar = standard error over folds", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _table(args: argparse.Namespace) -> int:
    out = Path(args.out)
    rows = [json.loads(p.read_text()) for p in sorted((out / "raw").glob("*.json"))]
    summary = rc.aggregate(rows)
    params = summary.pop("params")
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / "tables.md").write_text(rc.markdown(summary) + "\n")
    (out / "params").mkdir(exist_ok=True)
    for rig, payload in params.items():
        (out / "params" / f"{rig}.json").write_text(json.dumps(payload, indent=1) + "\n")
    if summary["cruise"]:
        _figure(summary, out / "cruise_scores.png")
    print(rc.markdown(summary))
    phase = pd_.aggregate(rows)
    if phase:
        (out / "phase_summary.json").write_text(json.dumps(phase, indent=1) + "\n")
        (out / "phase_tables.md").write_text(pd_.markdown(phase) + "\n")
        print(pd_.markdown(phase))
    print(f"{len(rows)} units -> {out}/summary.json, tables.md, params/ ({len(params)} rigs)")
    return 0


def _bench(args: argparse.Namespace) -> int:
    import torch

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    rows = []
    for rig in args.rigs or ["michaels", "pitcn_quad"]:
        for device in devices:
            row = rc.bench_gaussian_fit(rig, device)
            row["threads"] = torch.get_num_threads()
            row["gpu"] = torch.cuda.get_device_name(0) if device == "cuda" else None
            rows.append(row)
            print(json.dumps(row), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench_device.json").write_text(json.dumps(rows, indent=1) + "\n")
    return 0


def _examples(args: argparse.Namespace) -> int:
    """Generated flights of every given-schedule arm beside the held-out flight
    whose schedule they follow, one zoomable figure a rig."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from experiments.rps_traj.phases import PHASES

    rows = [("real", "held-out flight"), ("ours", "ours"), ("phasevar2", "per-phase VAR(2)")]
    rows.append(("phasevar4", "per-phase VAR(4)"))
    rotor = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    shade = ["#ffffff", "#fde0c5", "#e8e8e8", "#c9e4f5", "#ffffff", "#d9f0d3"]
    parts = []
    for rig in args.rigs or list(rc.FLIGHT_RIGS):
        ex = pd_.example_flights(rig, args.out, n=args.draws, fold=args.fold)
        t, labels = ex["t"], ex["labels"]
        fig = make_subplots(
            rows=len(rows),
            cols=1,
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0.03,
            subplot_titles=[title for _, title in rows],
        )
        draw_of = []  # per trace: the draw it belongs to (-1: always shown)
        for r, (key, _) in enumerate(rows, start=1):
            paths = [ex["real"]] if key == "real" else ex[key]
            for k, x in enumerate(paths):
                for i in range(4):
                    fig.add_trace(
                        go.Scattergl(
                            x=t,
                            y=x[i],
                            mode="lines",
                            line={"width": 1, "color": rotor[i]},
                            name=f"rotor {i + 1}",
                            legendgroup=f"rotor {i + 1}",
                            showlegend=(r == 1),
                            visible=(key == "real" or k == 0),
                        ),
                        row=r,
                        col=1,
                    )
                    draw_of.append(-1 if key == "real" else k)
        edges = [0, *(int(i) + 1 for i in (labels[1:] != labels[:-1]).nonzero()[0]), len(labels)]
        for a, b in zip(edges[:-1], edges[1:]):
            phase = int(labels[a])
            fig.add_vrect(
                x0=t[a],
                x1=t[b - 1],
                fillcolor=shade[phase],
                opacity=0.45,
                line_width=0,
                layer="below",
                annotation_text=PHASES[phase] if b - a > 8 else "",
                annotation_position="top left",
                annotation_font_size=9,
            )
        for when in ex["events"].values():
            if when is not None:
                fig.add_vline(x=when, line={"width": 1, "dash": "dot", "color": "#555"})
        fig.update_layout(
            title=f"{rig}: schedule of held-out flight {ex['flight']} (25 Hz); "
            "dotted lines: motors on, take-off, landing",
            height=230 * len(rows) + 120,
            margin={"l": 60, "r": 20, "t": 90, "b": 40},
            hovermode="x unified",
            updatemenus=[
                {
                    "type": "buttons",
                    "direction": "right",
                    "x": 1.0,
                    "xanchor": "right",
                    "y": 1.09,
                    "buttons": [
                        {
                            "label": f"draw {k + 1}",
                            "method": "update",
                            "args": [{"visible": [d in (-1, k) for d in draw_of]}],
                        }
                        for k in range(args.draws)
                    ],
                }
            ],
        )
        fig.update_yaxes(title_text="rev/s")
        fig.update_xaxes(title_text="s", row=len(rows), col=1)
        parts.append(fig.to_html(full_html=False, include_plotlyjs=not parts))
    path = Path(args.out) / "generated_flights.html"
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Generated flights on a given schedule</title>"
        "<body style='font-family:sans-serif;max-width:1500px;margin:auto'>"
        "<p>Every generated flight follows the held-out flight's motors-on, take-off and landing "
        "times; models are fitted on the other flights of the rig. Drag to zoom (all rows "
        "together), double-click to reset, buttons switch the draw.</p>" + "\n".join(parts)
    )
    print(path)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="fit and score units")
    run.add_argument("--stage", choices=("cruise", "flight", "phase"), required=True)
    run.add_argument("--rigs", nargs="*", default=None)
    run.add_argument("--families", nargs="*", default=None, help=f"cruise families {rc.FAMILIES}")
    run.add_argument("--arms", nargs="*", default=None, help=f"whole-flight arms {rc.FLIGHT_ARMS}")
    run.add_argument("--folds", nargs="*", default=None, help="only these folds (e.g. 0 1)")
    run.add_argument("--final", action="store_true", help="fit on every flight (fold 'all')")
    run.add_argument("--jobs", type=int, default=1, help="units in parallel")
    run.add_argument("--threads", type=int, default=1, help="threads inside one unit")
    run.add_argument("--no-resume", action="store_true")
    run.set_defaults(func=_run)
    table = sub.add_parser("table", help="aggregate the unit JSONs")
    table.set_defaults(func=_table)
    bench = sub.add_parser("bench", help="time the Gaussian fit's objective on CPU and GPU")
    bench.add_argument("--rigs", nargs="*", default=None)
    bench.set_defaults(func=_bench)
    examples = sub.add_parser("examples", help="zoomable generated flights on one schedule")
    examples.add_argument("--rigs", nargs="*", default=None)
    examples.add_argument("--fold", default="0")
    examples.add_argument("--draws", type=int, default=3)
    examples.set_defaults(func=_examples)
    for p in (run, table, bench, examples):
        p.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
