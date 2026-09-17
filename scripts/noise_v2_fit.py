#!/usr/bin/env python
"""Fit the noise model v2 on one support, one support set, or one flight pool.

    # 20 DREGON single-motor bench supports plus the four-motor validation,
    # one restartable unit-JSON per fit
    python scripts/noise_v2_fit.py bench --set dregon-bench --jobs 8
    # the 135 derived survey bench points, same harness
    python scripts/noise_v2_fit.py bench --set bench-points --jobs 16 --adam-steps 400
    # one support by name
    python scripts/noise_v2_fit.py bench --support bench_dregon_allMotors_70
    # Michael's FLY125 cruise: eight fit windows only (not frozen FLY124 score windows)
    python scripts/noise_v2_fit.py flight --set michaels-cruise --name michaels_fly125_cruise
    # DREGON room2 floor: five disjoint 8 s fit supports only, with its comb
    # FROZEN at the mean of the four Motor*_70 bench fits
    python scripts/noise_v2_fit.py flight --set dregon-floor --name dregon_room2_floor \
        --floor-only --frozen-mean results/noise_v2/rounds/round1/fits/bench_dregon_Motor*_70__bench.json
    # the round-1 fit findings table over everything already written
    python scripts/noise_v2_fit.py findings

Every fit lands at ``<out>/<support>__<mode>.json`` in the ``noise-v2-fit/1``
schema (:func:`experiments.noise_model.fit.write_fit`), and the bench modes run
through :mod:`utils.gridrun` so 135 points are one restartable unit grid: a
unit whose JSON already exists is skipped, a unit that raises leaves a ``.err``
beside it and does not kill the grid.

``PYTHONPATH=src:scripts``. Anything over a couple of CPU-minutes belongs on
``omnirun submit --backend uni-cpu --gpus 0``; the support ``.npz`` caches are
gitignored, so a remote job must run ``scripts/noise_v2_supports.py build``
first, in the same job.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.gridrun import Unit, add_gridrun_args, gridrun_from_args, unit_path  # noqa: E402

OUT_DIR = "results/noise_v2/rounds/round1/fits"
#: profile width. 130 is C4's ``k_cap``; the forward model additionally caps
#: every support at the highest order below its own Nyquist.
K_CAP = 130


# ── one unit ────────────────────────────────────────────────────────────────


def worker(unit: Unit) -> dict[str, Any]:
    """Fit ONE support (bench) or ONE pool (flight) and write its fit JSON."""
    import numpy as np
    import torch

    from experiments.noise_model import fit as FT
    from experiments.noise_model import model as MD
    from experiments.noise_model import supports as SU

    torch.set_num_threads(1)
    p = dict(unit.params)
    mode = str(p["mode"])
    out_dir = Path(p["out_dir"])
    optim = FT.OptimSpec(**p["optim"])
    frozen = p.get("frozen")

    if mode == "bench":
        support = SU.load_support(str(p["spec"]))
        batch = MD.bench_batch(
            name=support.name,
            power=np.asarray(support.power, dtype=np.float64),
            sr=int(support.sr),
            carrier_mean=np.asarray(support.carrier_rev_s, dtype=np.float64).mean(axis=1),
            k_cap=K_CAP,
        )
        name, kind = support.name, support.kind
    else:
        members = []
        specs = list(p["specs"])
        for spec in specs:
            s = SU.load_support(str(spec))
            members.append(
                (
                    s.name,
                    np.asarray(s.power, dtype=np.float64),
                    np.asarray(s.carrier_rev_s_audio, dtype=np.float64),
                    np.asarray(s.frame_starts, dtype=np.int64),
                )
            )
        first = SU.load_support(str(specs[0]))
        batch = MD.flight_batch(
            name=str(p["name"]),
            members=members,
            sr=int(first.sr),
            n_fft=int(first.n_fft),
            hop=int(first.hop),
            k_cap=K_CAP,
            frame_stride=int(p.get("frame_stride", 4)),
            max_frames=p.get("max_frames"),
        )
        name, kind = str(p["name"]), "flight"

    outcome = FT.fit_support(
        batch, mode=mode, frozen=frozen, optim=optim, progress=int(p.get("progress", 0))
    )
    path = out_dir / f"{name}__{mode}.json"
    FT.write_fit(
        path,
        support=name,
        kind=kind,
        mode=mode,
        outcome=outcome,
        batch=batch,
        extra={"frozen_from": p.get("frozen_from")} if p.get("frozen_from") else None,
    )
    d = MD.params_to_dict(outcome.params)
    return dict(
        uid=unit.uid,
        support=name,
        mode=mode,
        fit_json=str(path),
        converged=outcome.converged,
        wall_s=outcome.optimiser["wall_s"],
        whittle_nats=outcome.objective["whittle_nats"],
        per_band=outcome.objective["per_band"],
        n_cells=outcome.objective["n_cells"],
        sigma_nu=d["sigma_nu"],
        lam=d["lam"],
        sigma_eps_even=d["sigma_eps_even"],
        sigma_eps_odd=d["sigma_eps_odd"],
        lam_eps_even=d["lam_eps_even"],
        lam_eps_odd=d["lam_eps_odd"],
        carrier_rev_s=d["carrier_rev_s"],
        n_rotors=batch.n_rotors,
        k_max=batch.k_max,
    )


# ── frozen comb ─────────────────────────────────────────────────────────────


def mean_comb(paths: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The MEAN comb of several bench fits, as a ``frozen`` mapping.

    The DREGON floor fit freezes the comb at the mean of the four
    ``Motor{1-4}_70`` single-motor fits: the flight support has four rotors on
    one airframe and no bench fit of the four TOGETHER is a fit (the four-motor
    static record is the VALIDATION support, not a fitting one). Dynamics and
    the per-order profile are averaged in their own natural scale — rates and
    scales in log, dB levels in dB — over the fits that carry them.
    """
    import numpy as np

    fits = [json.loads(Path(p).read_text()) for p in paths]
    if not fits:
        raise ValueError("--frozen-mean needs at least one bench fit JSON")
    par = [f["params"] for f in fits]
    log_mean = lambda key: float(np.exp(np.mean([np.log(float(q[key])) for q in par])))  # noqa: E731
    prof = np.stack(
        [np.asarray(q["profile"]["profile_db"], dtype=np.float64)[0] for q in par]
    )  # (n_fits, K)
    frozen = dict(
        sigma_nu=log_mean("sigma_nu"),
        lam=log_mean("lam"),
        sigma_eps=[log_mean("sigma_eps_even"), log_mean("sigma_eps_odd")],
        lam_eps=[log_mean("lam_eps_even"), log_mean("lam_eps_odd")],
        profile_db=prof.mean(axis=0)[None, :].tolist(),
        amp_exp=float(np.mean([float(q["profile"]["amp_exp"]) for q in par])),
    )
    return frozen, dict(
        frozen_comb_from=[f["support"] for f in fits],
        frozen_comb_paths=[str(p) for p in paths],
        rule="log-mean of the rates and scales, dB-mean of the per-order profile",
    )


def expand_frozen(frozen: dict[str, Any], *, n_rotors: int) -> dict[str, Any]:
    """Repeat a one-rotor frozen comb over ``n_rotors`` rotors of one airframe."""
    import numpy as np

    out = dict(frozen)
    prof = np.asarray(frozen["profile_db"], dtype=np.float64)
    if prof.shape[0] != n_rotors:
        out["profile_db"] = np.repeat(prof[:1], n_rotors, axis=0).tolist()
    return out


# ── findings ────────────────────────────────────────────────────────────────


def findings(out_dir: Path) -> str:
    """The per-support parameter table and the population summary."""
    import numpy as np

    rows = []
    for path in sorted(out_dir.glob("*.json")):
        try:
            f = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if f.get("schema") != "noise-v2-fit/1":
            continue
        p, o, q = f["params"], f["objective"], f["optimiser"]
        rows.append(
            dict(
                support=f["support"],
                mode=f["mode"],
                n_rotors=f["n_rotors"],
                k_max=f["k_max"],
                sigma_nu=p["sigma_nu"],
                lam=p["lam"],
                sigma_eps_even=p["sigma_eps_even"],
                sigma_eps_odd=p["sigma_eps_odd"],
                lam_eps_even=p["lam_eps_even"],
                lam_eps_odd=p["lam_eps_odd"],
                carrier=p["carrier_rev_s"],
                floor_level_db=f["diagnostics"].get("floor_level_db"),
                whittle=o["whittle_nats"],
                comb=o["per_band"]["comb"],
                floor=o["per_band"]["floor"],
                n_cells=o["n_cells"],
                converged=q["converged"],
                wall_s=q["wall_s"],
                path=str(path),
            )
        )
    if not rows:
        return "# Noise model v2 — round 1 fits\n\nNo fit JSON found.\n"

    def stats(key: str, subset: list[dict[str, Any]]) -> str:
        v = np.asarray([r[key] for r in subset], dtype=np.float64)
        v = v[np.isfinite(v)]
        if not v.size:
            return "—"
        return f"{np.median(v):.4g} [{np.quantile(v, 0.25):.4g}, {np.quantile(v, 0.75):.4g}]"

    lines = [
        "# Noise model v2 — round 1 fits",
        "",
        f"{len(rows)} fit JSON(s) under `{out_dir}`. Every number below is read from a "
        "`noise-v2-fit/1` payload in that directory; nothing is recomputed here.",
        "",
        "## Per-support parameters",
        "",
        "| support | mode | R | k_max | sigma_nu | lam | sigma_eps even/odd | lam_eps even/odd |"
        " carrier rev/s | floor dB | whittle nats | comb | floor | cells | conv | wall s |",
        "|---|---|--:|--:|--:|--:|---|---|---|--:|--:|--:|--:|--:|:-:|--:|",
    ]
    for r in rows:
        car = ", ".join(f"{c:.3f}" for c in r["carrier"]) if r["carrier"] else "label"
        lines.append(
            f"| `{r['support']}` | {r['mode']} | {r['n_rotors']} | {r['k_max']} |"
            f" {r['sigma_nu']:.4f} | {r['lam']:.3f} |"
            f" {r['sigma_eps_even']:.4f} / {r['sigma_eps_odd']:.4f} |"
            f" {r['lam_eps_even']:.3f} / {r['lam_eps_odd']:.3f} | {car} |"
            f" {format(r['floor_level_db'], '.2f') if r['floor_level_db'] is not None else '—'} |"
            f" {r['whittle']:.6g} | {r['comb']:.6g} | {r['floor']:.6g} | {r['n_cells']} |"
            f" {'y' if r['converged'] else 'N'} | {r['wall_s']:.0f} |"
        )

    groups = {
        "DREGON single-motor bench (all throttles)": [
            r for r in rows if r["support"].startswith("bench_dregon_Motor")
        ],
        "survey bench points": [r for r in rows if r["support"].startswith("bench_point")],
        "flight": [r for r in rows if r["mode"].startswith("flight")],
    }
    lines += [
        "",
        "## Population median [IQR]",
        "",
        "| set | n | sigma_nu | lam | sigma_eps even |"
        " sigma_eps odd | lam_eps even | lam_eps odd |",
        "|---|--:|---|---|---|---|---|---|",
    ]
    for gname, subset in groups.items():
        if not subset:
            continue
        lines.append(
            f"| {gname} | {len(subset)} | {stats('sigma_nu', subset)} | {stats('lam', subset)} |"
            f" {stats('sigma_eps_even', subset)} | {stats('sigma_eps_odd', subset)} |"
            f" {stats('lam_eps_even', subset)} | {stats('lam_eps_odd', subset)} |"
        )

    quad = [r for r in rows if "allMotors" in r["support"]]
    singles = [
        r
        for r in rows
        if r["support"].startswith("bench_dregon_Motor") and r["support"].endswith("_70")
    ]
    if quad and singles:
        q = quad[0]
        lines += [
            "",
            "## Four-motor validation",
            "",
            "`motor_allMotors_70` against the four `Motor{1-4}_70` single-motor fits: the "
            "ratio of the four-rotor value to the geometric mean of the per-rotor ones, and "
            "the per-rotor spread that decision 9 of 2026-09-17 freezes as the tolerance.",
            "",
            "| parameter | four-motor | per-rotor geo-mean | ratio | per-rotor min/max |",
            "|---|--:|--:|--:|---|",
        ]
        for key in (
            "sigma_nu",
            "lam",
            "sigma_eps_even",
            "sigma_eps_odd",
            "lam_eps_even",
            "lam_eps_odd",
        ):
            v = np.asarray([r[key] for r in singles], dtype=np.float64)
            geo = float(np.exp(np.mean(np.log(v))))
            lines.append(
                f"| `{key}` | {q[key]:.4g} | {geo:.4g} | {q[key] / geo:.3f} |"
                f" {v.min():.4g} / {v.max():.4g} |"
            )

    edges = []
    for r in rows:
        if r["lam"] > 30.0:
            edges.append(
                f"`{r['support']}`: lam = {r['lam']:.2f} > 30 (label-chain regime, see the "
                "explainer's third consequence of the shaft prior)"
            )
        if not (0.11 <= r["sigma_nu"] <= 1.8):
            edges.append(
                f"`{r['support']}`: sigma_nu = {r['sigma_nu']:.4f} outside the prior's central 95 %"
            )
        if not r["converged"]:
            edges.append(f"`{r['support']}`: NOT converged (L-BFGS restart still improving)")
    lines += ["", "## Prior edges and non-convergence", ""]
    lines += [f"- {e}" for e in edges] if edges else ["- none"]
    lines.append("")
    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────────


def _optim_from_args(args: argparse.Namespace) -> dict[str, Any]:
    spec: dict[str, Any] = dict(
        adam_steps=int(args.adam_steps),
        adam_lr=float(args.adam_lr),
        lbfgs_iters=int(args.lbfgs_iters),
        seed=int(args.seed),
    )
    if getattr(args, "adam_batch", None) is not None:
        spec["adam_batch"] = None if int(args.adam_batch) <= 0 else int(args.adam_batch)
    if getattr(args, "lbfgs_frames", None) is not None:
        spec["lbfgs_frames"] = None if int(args.lbfgs_frames) <= 0 else int(args.lbfgs_frames)
    return spec


def _specs(args: argparse.Namespace) -> list[str]:
    from experiments.noise_model import supports as SU

    if args.support:
        return [str(s) for s in args.support]
    if not args.set:
        raise SystemExit("name --support ... or --set <support set>")
    specs = SU.support_set(str(args.set))
    # The named flight sets deliberately include their *frozen scoring* members
    # so one cache index describes the complete round.  Fits must not train on
    # those members: R1's Michael fit is FLY125 only and its DREGON floor fit
    # uses the guarded 8 s room-2 segments, not the five score windows.
    if args.set == "michaels-cruise":
        specs = [s for s in specs if s.args.get("recording") == "FLY125"]
    elif args.set == "dregon-floor":
        specs = [s for s in specs if float(s.args.get("dur_s", 0.0)) == 8.0]
    return [s.text if hasattr(s, "text") else str(s) for s in specs]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("bench", "flight"):
        p = sub.add_parser(name)
        p.add_argument("--support", nargs="*", help="support spec string(s)")
        p.add_argument("--set", help="named support set (dregon-bench, bench-points, ...)")
        p.add_argument("--out", default=OUT_DIR)
        p.add_argument(
            "--grid-dir", default=None, help="gridrun unit directory (default <out>/grid)"
        )
        p.add_argument("--adam-steps", type=int, default=1500)
        p.add_argument("--adam-lr", type=float, default=0.02)
        p.add_argument("--lbfgs-iters", type=int, default=200)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--progress", type=int, default=0, help="print the Adam loss every N steps")
        add_gridrun_args(p, jobs=4)
        if name == "flight":
            p.add_argument(
                "--name", required=True, help="pooled fit name, e.g. michaels_fly125_cruise"
            )
            p.add_argument("--floor-only", action="store_true")
            p.add_argument(
                "--frozen-mean",
                nargs="*",
                default=None,
                help="bench fit JSONs to freeze the comb at",
            )
            p.add_argument("--frame-stride", type=int, default=4)
            p.add_argument("--max-frames", type=int, default=256)
            p.add_argument("--adam-batch", type=int, default=8)
            p.add_argument("--lbfgs-frames", type=int, default=64)

    f = sub.add_parser("findings")
    f.add_argument("--out", default=OUT_DIR)

    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cmd == "findings":
        text = findings(out_dir)
        (out_dir / "findings.md").write_text(text)
        print(text)
        return 0

    specs = _specs(args)
    grid_dir = Path(args.grid_dir) if args.grid_dir else out_dir / "grid"

    if args.cmd == "bench":
        units = [
            Unit(
                uid=spec.replace(":", "_"),
                params=dict(
                    mode="bench",
                    spec=spec,
                    out_dir=str(out_dir),
                    optim=_optim_from_args(args),
                    progress=int(args.progress),
                ),
            )
            for spec in specs
        ]
    else:
        frozen = None
        frozen_from = None
        if args.frozen_mean:
            from experiments.noise_model import supports as SU

            frozen, frozen_from = mean_comb(list(args.frozen_mean))
            n_rotors = int(SU.load_support(specs[0]).carrier_rev_s.shape[0])
            frozen = expand_frozen(frozen, n_rotors=n_rotors)
        mode = "flight_floor_only" if args.floor_only else "flight"
        if args.floor_only and frozen is None:
            raise SystemExit("--floor-only needs --frozen-mean <bench fit JSONs>")
        units = [
            Unit(
                uid=f"{args.name}__{mode}",
                params=dict(
                    mode=mode,
                    specs=specs,
                    name=str(args.name),
                    out_dir=str(out_dir),
                    optim=_optim_from_args(args),
                    frozen=frozen,
                    frozen_from=frozen_from,
                    frame_stride=int(args.frame_stride),
                    max_frames=None if int(args.max_frames) <= 0 else int(args.max_frames),
                    progress=int(args.progress),
                ),
            )
        ]

    def summarize(payloads: list[dict[str, Any]]) -> dict[str, Any]:
        ok = [r for r in payloads if r.get("converged")]
        return dict(
            n_units=len(payloads),
            n_converged=len(ok),
            not_converged=[r["support"] for r in payloads if not r.get("converged")],
            slowest_s=max((r.get("wall_s", 0.0) for r in payloads), default=0.0),
            fits=[r.get("fit_json") for r in payloads],
        )

    result = gridrun_from_args(args, units, worker, grid_dir, summarize=summarize)
    text = findings(out_dir)
    (out_dir / "findings.md").write_text(text)
    print(f"# wrote {out_dir / 'findings.md'}", flush=True)
    for u in units:
        err = unit_path(grid_dir, u.uid).with_suffix(".err")
        if err.exists():
            print(f"!! {u.uid}: {err}", flush=True)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
