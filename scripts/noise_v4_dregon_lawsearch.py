#!/usr/bin/env python
"""Noise model v4: search the DREGON flight payload's speed laws and levels
DIRECTLY on the official scorer's metrics (HPPNet-L2 PIT MAE and LTAS).

Module: :mod:`experiments.noise_model.dregon_lawsearch_v4` (the space, the
composition, the tables); this CLI owns the frozen scorer (the scripts tree's
``stochastic_fit_revised_eval.Tracker``) and the restartable grids.

    # the full search (a CPU node; one HPPNet pass per (point, segment))
    OMP_NUM_THREADS=2 python scripts/noise_v4_dregon_lawsearch.py \\
        --base results/noise_v4/flight/wind/dregon_room2__flight_v4__dregon__kappa0__r5__s2__scwind.json \\
        --workers 16 --threads 2
    # a smoke: 2 points x 1 segment, no stage 2
    python scripts/noise_v4_dregon_lawsearch.py --base <payload> --points 2 --max-segments 1 \\
        --stage2 0 --workers 2 --out /tmp/nv4_laws_smoke

Steps, each a :mod:`utils.gridrun` grid under ``<out>/<base stem>__laws/``
(``config.json`` refuses another base, box, seed or segment count):

1. ``components/``: per segment the render parts at the render seed
   (:func:`~experiments.noise_model.dregon_lawsearch_v4.build_components`);
2. ``refs/``: per segment the real clip's PIT, nv3r4's and the base payload's
   PIT and LTAS (direct renders), and the composition check on the first
   segment (the base point and three random points: the composed render
   against ``render_noise`` of the edited payload, and the PIT/LTAS of both);
3. ``stage1/``: ``--stage1`` scrambled Sobol points x segments;
4. ``stage2/``: ``--stage2`` Sobol points in the box shrunk around stage 1's
   best feasible points (``stage2_box.json``, written once);
5. ``<out>/<base stem>__laws.table.json`` (every point, references, Pareto
   front, main effects, verification, timings) and, when a point beats nv3r4
   on both tuning-set means, ``<out>/<base stem>__laws.json``: the base payload
   with that point written in and a ``law_search`` provenance block.

``--points N`` caps both stages at their first N points (a smoke);
``--device cuda`` runs the tracker on the GPU.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.gridrun import GridResult, Unit, run_grid  # noqa: E402

DEFAULT_OUT = "results/noise_v4/flight/lawsearch"
_TRACKER: Any = None


# ── the worker (module level: the spawn pool pickles it by reference) ──────


def _tracker(threads: int, device: str) -> Any:
    """The frozen scorer of ``noise_v2_round_score.Probe``, once per process."""
    global _TRACKER
    if _TRACKER is None:
        import torch

        torch.set_num_threads(max(1, int(threads)))
        import stochastic_fit_revised_eval as ev

        from experiments.noise_model import gates as GT

        tracker = ev.Tracker(dict(experiment=GT.SCORER_EXPERIMENT, ckpt=GT.SCORER_CKPT))
        if tracker.record["sha256"] != GT.SCORER_SHA256:
            raise RuntimeError(
                f"frozen scorer digest mismatch: {tracker.record['sha256']} != {GT.SCORER_SHA256}"
            )
        if device != "cpu":
            import zoo

            tracker.fm = zoo.load(GT.SCORER_EXPERIMENT, ckpt=GT.SCORER_CKPT, device=device)
        _TRACKER = tracker
    return _TRACKER


@functools.lru_cache(maxsize=4)
def _payload(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _components_path(work: str, segment: int) -> Path:
    return Path(work) / "components" / f"seg{int(segment)}.npz"


@functools.cache
def _components(work: str, segment: int, base: str) -> Any:
    from experiments.noise_model import dregon_lawsearch_v4 as LS

    return LS.load_components(_components_path(work, segment), _payload(base))


def unit_worker(unit: Unit) -> dict[str, Any]:
    import numpy as np

    from data_processing.noise_model.render import render_noise
    from experiments.noise_model import dregon_lawsearch_v4 as LS

    p = unit.params
    kind, work, base_path = str(p["kind"]), str(p["work"]), str(p["base"])
    seg = int(p["segment"])
    base = _payload(base_path)
    head = dict(uid=unit.uid, kind=kind, segment=seg)
    if kind == "components":
        t0 = time.time()
        arrays = LS.build_components(base, seg, seed=int(p["seed"]))
        LS.save_components(_components_path(work, seg), arrays)
        return head | dict(
            name=str(arrays["name"]),
            path=str(_components_path(work, seg)),
            wall_s=time.time() - t0,
            part_wall_s=arrays["wall_s"],
        )
    comp = _components(work, seg, base_path)
    tracker = _tracker(int(p["threads"]), str(p["device"]))
    head["name"] = comp.name
    if kind == "ref":
        arm = str(p["arm"])
        t0 = time.time()
        if arm == "real":
            audio = comp.real.astype(np.float64)
        else:
            fit = _payload(LS.NV3R4) if arm == "nv3r4" else base
            audio = np.asarray(
                render_noise(fit, comp.rps, n_mics=comp.n_mics, seed=comp.seed), dtype=np.float64
            )[: comp.n_mics]
        render_s = time.time() - t0
        return head | dict(arm=arm, render_s=render_s, **LS.score(tracker, comp, audio))
    point = {k: float(v) for k, v in dict(p["point"]).items()}
    if kind == "verify":
        t0 = time.time()
        direct = np.asarray(
            render_noise(
                LS.edit_payload(base, point), comp.rps, n_mics=comp.n_mics, seed=comp.seed
            ),
            dtype=np.float64,
        )
        render_s = time.time() - t0
        t0 = time.time()
        composed = LS.compose(comp, point)
        compose_s = time.time() - t0
        err = composed - direct
        rms = np.sqrt((direct**2).mean(axis=1))
        s_direct = LS.score(tracker, comp, direct)
        s_comp = LS.score(tracker, comp, composed)
        return head | dict(
            idx=int(p["idx"]),
            point=point,
            render_s=render_s,
            compose_s=compose_s,
            err_max_rel=float(np.abs(err).max() / np.sqrt((direct**2).mean())),
            err_rms_rel=float(np.sqrt((err**2).mean() / (direct**2).mean())),
            err_max_rel_per_mic=(np.abs(err).max(axis=1) / rms).tolist(),
            direct=s_direct,
            composed=s_comp,
            d_pit_mae=s_comp["pit_mae"] - s_direct["pit_mae"],
            d_ltas_abs_db=s_comp["ltas_abs_db"] - s_direct["ltas_abs_db"],
        )
    if kind == "point":
        t0 = time.time()
        audio = LS.compose(comp, point)
        compose_s = time.time() - t0
        return head | dict(
            stage=int(p["stage"]),
            idx=int(p["idx"]),
            point=point,
            compose_s=compose_s,
            **LS.score(tracker, comp, audio),
        )
    raise ValueError(f"unknown unit kind {kind!r}")


# ── the driver ──────────────────────────────────────────────────────────────


def _die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = sorted(float(r[key]) for r in rows if r.get(key) is not None)
    return vals[len(vals) // 2] if vals else None


def _timings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return dict(
        n_units=len(rows),
        median_compose_s=_median(rows, "compose_s"),
        median_pit_s=_median(rows, "pit_s"),
        median_ltas_s=_median(rows, "ltas_s"),
        median_render_s=_median(rows, "render_s"),
    )


def _verification(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vs = sorted((r for r in rows if r["kind"] == "verify"), key=lambda r: r["idx"])
    return dict(
        rule="compose(segment parts, point) against render_noise(edit_payload(base, point)) at "
        "the render seed, first segment; idx 0 is the base point, the rest uniform draws",
        points=[
            dict(
                idx=r["idx"],
                point=r["point"],
                err_max_rel=r["err_max_rel"],
                err_rms_rel=r["err_rms_rel"],
                pit_direct=r["direct"]["pit_mae"],
                pit_composed=r["composed"]["pit_mae"],
                d_pit_mae=r["d_pit_mae"],
                ltas_direct=r["direct"]["ltas_abs_db"],
                ltas_composed=r["composed"]["ltas_abs_db"],
                d_ltas_abs_db=r["d_ltas_abs_db"],
                render_s=r["render_s"],
                compose_s=r["compose_s"],
            )
            for r in vs
        ],
        max_err_max_rel=max((r["err_max_rel"] for r in vs), default=None),
        max_abs_d_pit_mae=max((abs(r["d_pit_mae"]) for r in vs), default=None),
        max_abs_d_ltas_abs_db=max((abs(r["d_ltas_abs_db"]) for r in vs), default=None),
    )


def _check_grid(result: GridResult, what: str) -> None:
    if result.n_failed:
        _die(f"{result.n_failed} {what} unit(s) failed; see {result.out_dir}/raw/*.err")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True, help="the noise-v4-fit/3 DREGON flight payload")
    ap.add_argument("--stage1", type=int, default=256, help="stage-1 Sobol points")
    ap.add_argument("--stage2", type=int, default=128, help="stage-2 Sobol points (0: none)")
    ap.add_argument("--workers", type=int, default=4, help="parallel worker processes")
    ap.add_argument("--threads", type=int, default=2, help="torch/BLAS threads per worker")
    ap.add_argument("--device", default="cpu", help="the tracker's device (cpu | cuda)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--max-segments", type=int, default=None, help="first N tuning segments")
    ap.add_argument("--points", type=int, default=None, help="cap both stages (smoke)")
    ap.add_argument("--no-resume", action="store_true", help="re-run every unit")
    ap.add_argument(
        "--free",
        default=None,
        help="search only these numbers, the rest pinned at the base payload's: "
        "'g_a=-3:9,g_a1=-3:9,g_f=-9:3' (default: the full BOX)",
    )
    ap.add_argument("--gate", default="pit+ltas", help="feasibility: 'pit+ltas' or 'pit'")
    ap.add_argument(
        "--select", default="min-pit", help="among feasible points: 'min-pit' or 'min-ltas'"
    )
    args = ap.parse_args(argv)
    os.environ.setdefault("NUMBA_NUM_THREADS", str(args.threads))

    from experiments.noise_model import dregon_flight_v4 as DF
    from experiments.noise_model import dregon_lawsearch_v4 as LS
    from experiments.noise_model import flight_v4 as FV
    from experiments.noise_model import gates as GT

    t_start = time.time()
    base_path = Path(args.base)
    raw = base_path.read_bytes()
    base = json.loads(raw)
    LS.check_base(base)
    if args.gate not in LS.GATES or args.select not in LS.SELECTIONS:
        _die(f"--gate in {LS.GATES}, --select in {LS.SELECTIONS}")
    box: dict[str, tuple[float, float]] = dict(LS.BOX)
    if args.free:
        free = {}
        for item in str(args.free).split(","):
            name, _, rng = item.partition("=")
            lo, _, hi = rng.partition(":")
            free[name.strip()] = (float(lo), float(hi))
        box = LS.pinned_box(base, free)
    stem = base_path.stem
    out = Path(args.out)
    work = out / f"{stem}__laws"
    n_specs = len(DF.pool_specs())
    n_seg = n_specs if args.max_segments is None else min(int(args.max_segments), n_specs)
    n1 = int(args.stage1) if args.points is None else min(int(args.stage1), int(args.points))
    n2 = int(args.stage2) if args.points is None else min(int(args.stage2), int(args.points))
    config = dict(
        stage=LS.STAGE,
        base=str(base_path),
        base_sha256=hashlib.sha256(raw).hexdigest(),
        n_segments=n_seg,
        render_seed=LS.RENDER_SEED,
        params=list(LS.PARAMS),
        box={k: list(v) for k, v in box.items()},
        gate=args.gate,
        select=args.select,
        sobol_seeds=list(LS.SOBOL_SEEDS),
        verify_seed=LS.VERIFY_SEED,
        nv3r4=LS.NV3R4,
    )
    cfg_path = work / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        if old != config:
            _die(f"{cfg_path} holds another configuration; use another --out")
    else:
        _atomic_json(cfg_path, config)
    common = dict(work=str(work), base=str(base_path), threads=args.threads, device=args.device)
    grid = dict(
        resume=not args.no_resume,
        blas_threads=int(args.threads),
        mp_context="spawn",
    )

    # 1. the parts
    comps = [
        Unit(f"comp__seg{i}", dict(common, kind="components", segment=i, seed=LS.RENDER_SEED))
        for i in range(n_seg)
    ]
    res = run_grid(
        comps,
        unit_worker,
        work / "components",
        jobs=min(int(args.workers), n_seg),
        summarize=lambda rows: dict(
            n_units=len(rows), wall_s=[round(float(r["wall_s"]), 1) for r in rows]
        ),
        **grid,
    )
    _check_grid(res, "component")

    # 2. references and the composition check
    refs = [
        Unit(f"ref__{arm}__seg{i}", dict(common, kind="ref", arm=arm, segment=i))
        for i in range(n_seg)
        for arm in ("real", "nv3r4", "base")
    ]
    refs += [
        Unit(f"verify__v{j}__seg0", dict(common, kind="verify", idx=j, point=pt, segment=0))
        for j, pt in enumerate(LS.verify_points(base, box))
    ]
    res = run_grid(refs, unit_worker, work / "refs", jobs=int(args.workers), **grid)
    _check_grid(res, "reference")
    ref_rows = LS.read_rows(work / "refs" / "raw")
    references = LS.bars([r for r in ref_rows if r["kind"] == "ref"], n_seg)
    verification = _verification(ref_rows)
    bar = references["nv3r4"]

    # 3. stage 1
    def point_units(stage: int, points: list[dict[str, float]]) -> list[Unit]:
        return [
            Unit(
                f"s{stage}__p{j:04d}__seg{i}",
                dict(common, kind="point", stage=stage, idx=j, point=pt, segment=i),
            )
            for j, pt in enumerate(points)
            for i in range(n_seg)
        ]

    def rows_of(stage: int, n: int) -> list[dict[str, Any]]:
        rows = LS.read_rows(work / f"stage{stage}" / "raw")
        return [r for r in rows if r["idx"] < n and r["segment"] < n_seg]

    pts1 = LS.sobol_points(n1, box, LS.SOBOL_SEEDS[0])
    res = run_grid(
        point_units(1, pts1),
        unit_worker,
        work / "stage1",
        jobs=int(args.workers),
        summarize=_timings,
        **grid,
    )
    _check_grid(res, "stage-1")
    table = LS.point_table(rows_of(1, n1), n_seg)
    LS.mark_feasible(table, bar, args.gate)

    # 4. stage 2
    s2: dict[str, Any] | None = None
    if n2 > 0:
        s2_path = work / "stage2_box.json"
        if s2_path.exists():
            s2 = json.loads(s2_path.read_text())
            if int(s2["n_stage1"]) != n1:
                _die(f"{s2_path} was drawn from {s2['n_stage1']} stage-1 points, not {n1}")
        else:
            s2 = LS.stage2_box(table, bar, box=box, gate=args.gate, selection=args.select)
            _atomic_json(s2_path, s2)
        pts2 = LS.sobol_points(n2, s2["box"], LS.SOBOL_SEEDS[1])
        res = run_grid(
            point_units(2, pts2),
            unit_worker,
            work / "stage2",
            jobs=int(args.workers),
            summarize=_timings,
            **grid,
        )
        _check_grid(res, "stage-2")
        table += LS.point_table(rows_of(2, n2), n_seg)
        LS.mark_feasible(table, bar, args.gate)

    # 5. the table, the selection, the export
    chosen = LS.select(table, args.select)
    point_rows = rows_of(1, n1) + (rows_of(2, n2) if n2 > 0 else [])
    comp_rows = LS.read_rows(work / "components" / "raw")
    timings = dict(
        per_point_segment=_timings(point_rows),
        components_wall_s=[round(float(r["wall_s"]), 1) for r in comp_rows],
        verify=dict(
            median_direct_render_s=_median(
                [r for r in ref_rows if r["kind"] == "verify"], "render_s"
            ),
            median_compose_s=_median([r for r in ref_rows if r["kind"] == "verify"], "compose_s"),
        ),
    )
    table_path = out / f"{stem}__laws.table.json"
    payload_path = out / f"{stem}__laws.json"
    summary = dict(
        config=config,
        n_stage1=n1,
        n_stage2=n2,
        tuning_set=references["real"]["segments"],
        references=references,
        bar=dict(arm="nv3r4", mean_pit_mae=bar["mean_pit_mae"], mean_ltas=bar["mean_ltas_abs_db"]),
        verification=verification,
        stage2=s2,
        main_effects=LS.main_effects(table, box),
        pareto=LS.pareto(table),
        n_feasible=sum(1 for e in table if e.get("feasible")),
        chosen=chosen,
        payload=str(payload_path) if chosen else None,
        timings=timings,
        git=FV._git_sha(),
        table=table,
    )
    _atomic_json(table_path, summary)
    if chosen is not None:
        payload = LS.edit_payload(base, chosen["point"])
        payload["law_search"] = dict(
            stage=LS.STAGE,
            module="experiments.noise_model.dregon_lawsearch_v4",
            script="scripts/noise_v4_dregon_lawsearch.py",
            base=str(base_path),
            base_sha256=config["base_sha256"],
            params=list(LS.PARAMS),
            box=config["box"],
            sampler=dict(
                kind="scipy.stats.qmc.Sobol, scrambled, random_base2 prefix",
                seeds=list(LS.SOBOL_SEEDS),
                n_stage1=n1,
                n_stage2=n2,
            ),
            stage2=s2,
            selection=f"gate {args.gate!r} against nv3r4's tuning-set means, then {args.select}",
            scorer=dict(
                experiment=GT.SCORER_EXPERIMENT,
                ckpt=GT.SCORER_CKPT,
                sha256=GT.SCORER_SHA256,
                pit="tracker.pit, all mics, raw-telemetry cruise support",
                ltas="revised_eval.ltas_deviation_db(real, arm, mic=0).mean_abs_db",
                device=args.device,
            ),
            tuning_set=references["real"]["segments"],
            render_seed=LS.RENDER_SEED,
            references=references,
            chosen=chosen,
            verification=dict(
                max_err_max_rel=verification["max_err_max_rel"],
                max_abs_d_pit_mae=verification["max_abs_d_pit_mae"],
                max_abs_d_ltas_abs_db=verification["max_abs_d_ltas_abs_db"],
            ),
            table=str(table_path),
            git=summary["git"],
        )
        _atomic_json(payload_path, payload)

    # the report
    print(f"\n== references (tuning set, seed {LS.RENDER_SEED}) ==")
    for arm, r in references.items():
        print(
            f"{arm:6s} mean PIT {r['mean_pit_mae']:.3f}  mean LTAS {r['mean_ltas_abs_db']:.3f}  "
            f"per segment PIT {[round(v, 3) for v in r['pit_mae']]}  "
            f"LTAS {[round(v, 3) for v in r['ltas_abs_db']]}"
        )
    print("\n== composition check ==")
    for v in verification["points"]:
        print(
            f"v{v['idx']}: max|err|/rms {v['err_max_rel']:.2e}  rms err/rms {v['err_rms_rel']:.2e}  "
            f"PIT {v['pit_direct']:.4f} -> {v['pit_composed']:.4f}  "
            f"LTAS {v['ltas_direct']:.4f} -> {v['ltas_composed']:.4f}  "
            f"direct render {v['render_s']:.2f}s compose {v['compose_s']:.2f}s"
        )
    print("\n== timings ==")
    print(json.dumps(timings, indent=1))
    print(f"\n== points: {len(table)}, feasible {summary['n_feasible']} ==")
    for e in sorted((e for e in table if e["complete"]), key=lambda e: e["mean_pit_mae"])[:10]:
        pt = " ".join(f"{k}={e['point'][k]:.2f}" for k in LS.PARAMS)
        print(
            f"s{e['stage']} p{e['idx']:04d}  PIT {e['mean_pit_mae']:.3f}  "
            f"LTAS {e['mean_ltas_abs_db']:.3f}  feasible {e['feasible']}  {pt}"
        )
    print(f"\nchosen: {chosen and dict(stage=chosen['stage'], idx=chosen['idx'])}")
    print(f"table: {table_path}" + (f"\npayload: {payload_path}" if chosen else ""))
    print(f"wall {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
