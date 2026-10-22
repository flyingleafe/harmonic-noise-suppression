"""Noise model v4 — the DREGON bench rig fit driver (Procedure A of
``docs/explainers/noise-model-v4.qmd``): one joint fit over the 20 single-rotor
recordings, its restarts and the planted check::

    python scripts/noise_v4_bench.py fit --seeds 4 --kappa K --device cuda [--flat-shaft]
    python scripts/noise_v4_bench.py reduce
    python scripts/noise_v4_bench.py planted --truth <payload.json> --kappa K --device cuda
    python scripts/noise_v4_bench.py refit --from <payload.json> --flat-shaft --device cuda

Outputs under ``--out`` (default ``results/noise_v4/bench``): ``restarts/`` one
``noise-v4-fit/3`` payload per restart; ``raw/`` the gridrun unit records;
``fits/<name>.json`` the selected restart (lowest objective) with the restart
spread per parameter group; ``planted/<name>.json`` the recovery report;
``restarts/<source>__flat.json`` + ``refit/<source>__flat.json`` the nested
flat-shaft refit (one ``(sigma_nu, lam)`` for every speed) and its comparison.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.gridrun import Unit, run_grid  # noqa: E402

OUT = Path("results/noise_v4/bench")
ALL_MOTORS = ("Motor1", "Motor2", "Motor3", "Motor4")
ALL_THROTTLES = (50, 60, 70, 80, 90)


def _git() -> str | None:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or None


def _rig_name(
    motors: list[str], throttles: list[int], kappa: float, tag: str, flat: bool = False
) -> str:
    """``dregon[-<motors>][-t<throttles>]__kappa<k>[__flat][__tag]``."""
    name = "dregon"
    if tuple(motors) != ALL_MOTORS:
        name += "-" + "-".join(motors)
    if tuple(throttles) != ALL_THROTTLES:
        name += "-t" + "-".join(str(t) for t in throttles)
    return f"{name}__kappa{kappa:g}" + ("__flat" if flat else "") + (f"__{tag}" if tag else "")


def _rig_of_payload(pay: dict[str, Any]) -> Any:
    """The real-data rig of a payload's recordings (cached settings)."""
    from experiments.noise_model import bench_v4 as B

    keys = [r["key"] for r in pay["recordings"]]
    motors = tuple(dict.fromkeys(r["motor"] for r in pay["recordings"]))
    pairs = [re.fullmatch(r"motor_(\w+?)_(\d+)", k) for k in keys]
    settings = [B.load_setting(m.group(1), int(m.group(2))) for m in pairs if m is not None]
    return B.build_rig(motors, settings=settings)


def _fit_worker(unit: Unit) -> dict[str, Any]:
    import torch

    from experiments.noise_model import bench_v4 as B

    prm = unit.params
    torch.set_num_threads(int(prm["threads"]))
    rig = B.build_rig(tuple(prm["motors"]), throttles=tuple(prm["throttles"]))
    model, rec = B.fit_rig(
        rig,
        seed=int(prm["seed"]),
        kappa=float(prm["kappa"]),
        adam_steps=int(prm["adam_steps"]),
        lbfgs_iters=int(prm["lbfgs_iters"]),
        device=prm["device"],
        flat_shaft=bool(prm["flat_shaft"]),
        deadline_s=60.0 * float(prm["deadline_min"]) if prm.get("deadline_min") else None,
    )
    pay = B.payload(model, rec, git=prm["git"])
    path = Path(prm["out"]) / "restarts" / f"{unit.uid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pay))
    return dict(
        uid=unit.uid,
        seed=int(prm["seed"]),
        kappa=float(prm["kappa"]),
        objective=rec["objective"],
        converged=rec["converged"],
        restart_gain_per_cell=rec["restart_gain_per_cell"],
        restart_movement=rec["restart_movement"],
        stop=rec["stop"],
        n_convergence_passes=len(rec["convergence_passes"]),
        held_orders=rec["evidence"]["held_orders"],
        alpha_aero=pay["params"]["alpha_aero"],
        alpha_motor=pay["params"]["alpha_motor"],
        b0=pay["params"]["floor"]["b0"],
        b1=pay["params"]["floor"]["b1"],
        shaft=pay["params"]["shaft"],
        eval_s=rec["eval_s"],
        wall_s=rec["wall_s"],
        path=str(path),
    )


def cmd_fit(args: argparse.Namespace) -> int:
    git = _git()
    name = _rig_name(args.motors, args.throttles, args.kappa, args.tag, args.flat_shaft)
    units = [
        Unit(
            uid=f"{name}__s{s}",
            params=dict(
                motors=list(args.motors),
                throttles=list(args.throttles),
                seed=s,
                kappa=args.kappa,
                flat_shaft=bool(args.flat_shaft),
                adam_steps=args.adam_steps,
                lbfgs_iters=args.lbfgs_iters,
                deadline_min=args.deadline_min,
                device=args.device,
                threads=args.threads,
                out=str(args.out),
                git=git,
            ),
        )
        for s in range(args.seed_start, args.seed_start + args.seeds)
    ]
    res = run_grid(units, _fit_worker, str(args.out), jobs=args.jobs, mp_context="spawn")
    return res.exit_code


def cmd_reduce(args: argparse.Namespace) -> int:
    import numpy as np

    from experiments.noise_model import bench_v4 as B

    # per restart (seed), its latest continuation ``__c<n>`` stands for it
    latest: dict[str, dict[int, tuple[int, Path]]] = {}
    for p in sorted((args.out / "restarts").glob("*.json")):
        m = re.fullmatch(r"(.*)__s(\d+)(?:__c(\d+))?", p.stem)
        if m is None:
            continue
        name, seed, c = m.group(1), int(m.group(2)), int(m.group(3) or 0)
        if c >= latest.setdefault(name, {}).get(seed, (-1, p))[0]:
            latest[name][seed] = (c, p)
    groups: dict[str, list[dict[str, Any]]] = {}
    for name, by_seed in latest.items():
        for _, p in sorted(by_seed.values(), key=lambda cp: str(cp[1])):
            pay = json.loads(p.read_text())
            if pay.get("schema") == "noise-v4-fit/3":
                groups.setdefault(name, []).append(dict(path=str(p), pay=pay))
    (args.out / "fits").mkdir(parents=True, exist_ok=True)
    for name, items in groups.items():
        obj = np.array([it["pay"]["objective"]["total_nats"] for it in items])
        n_cells = items[0]["pay"]["objective"]["n_cells"]
        best = items[int(np.argmin(obj))]["pay"]
        vecs = [B.payload_group_vectors(it["pay"]) for it in items]
        lo = {g: np.min([v[g] for v in vecs], axis=0) for g in vecs[0]}
        hi = {g: np.max([v[g] for v in vecs], axis=0) for g in vecs[0]}
        opt = [it["pay"]["optimiser"] for it in items]
        best["restarts"] = dict(
            paths=[it["path"] for it in items],
            seeds=[o["seed"] for o in opt],
            objective=obj.tolist(),
            converged=[o["converged"] for o in opt],
            restart_gain_per_cell=[o["restart_gain_per_cell"] for o in opt],
            stop=[o.get("stop") for o in opt],
            best_minus_median_per_cell=float((np.median(obj) - obj.min()) / n_cells),
            best_minus_worst_per_cell=float((obj.max() - obj.min()) / n_cells),
            spread=B.group_movement(lo, hi),
            alpha_aero=[it["pay"]["params"]["alpha_aero"] for it in items],
            alpha_motor=[it["pay"]["params"]["alpha_motor"] for it in items],
            b0=[it["pay"]["params"]["floor"]["b0"] for it in items],
            shaft=[it["pay"]["params"]["shaft"] for it in items],
            rule="reported fit = restart with the lowest total objective; spread = "
            "per-group range over restarts (levels median, others as the movement rule)",
        )
        (args.out / "fits" / f"{name}.json").write_text(json.dumps(best))
        print(
            f"{name}: {len(items)} restarts, best {obj.min():.1f}, spread "
            f"{best['restarts']['best_minus_worst_per_cell']:.2e}/cell, converged "
            f"{sum(best['restarts']['converged'])}/{len(items)}"
        )
    return 0


def _within(
    fit: float, truth: float, *, abs_tol: float | None = None, rel_tol: float | None = None
) -> dict[str, Any]:
    err = fit - truth
    ok = abs(err) <= abs_tol if abs_tol is not None else abs(err / truth) <= float(rel_tol or 0.0)
    return dict(truth=truth, fit=fit, err=err, rel_err=err / truth if truth else None, ok=bool(ok))


def _planted(
    truth_path: str | Path,
    *,
    floor_below: float | None,
    render_seed: int,
    kappa: float,
    device: str,
) -> tuple[Any, Any, dict[str, Any] | None]:
    """``(truth model, planted rig, floor patch)``: the truth payload, its floor
    below ``floor_below`` Hz replaced by the measured inter-line floor (the
    data-only initial control values), rendered with ``render_seed``."""
    import numpy as np
    import torch

    from experiments.noise_model import bench_v4 as B

    pay_t = json.loads(Path(truth_path).read_text())
    real = _rig_of_payload(pay_t)
    truth = B.RigModel(real, device=device, kappa=kappa)
    truth.load_payload(pay_t)
    floor_patch = None
    if floor_below is not None:
        ctrl_hz = np.asarray(truth.frozen[0].grid.floor.ctrl_hz, dtype=np.float64)
        with torch.no_grad():
            target = truth.cbar().cpu().numpy()
        low = ctrl_hz < float(floor_below)
        target[low] = np.asarray(real.init["floor_ctrl_db"])[low]
        err = truth.set_floor_ctrl(target)
        floor_patch = dict(
            below_hz=float(floor_below),
            ctrl_hz=ctrl_hz[low].tolist(),
            ctrl_db=target[low].tolist(),
            max_reproduction_err_db=err,
        )
    planted = B.render_planted(truth, seed=render_seed)
    return truth, B.build_rig(real.motors, settings=planted), floor_patch


def cmd_planted(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    from experiments.noise_model import bench_v4 as B

    torch.set_num_threads(args.threads)
    truth, rig, floor_patch = _planted(
        args.truth,
        floor_below=args.floor_data_below,
        render_seed=args.render_seed,
        kappa=args.kappa,
        device=args.device,
    )
    start = json.loads(Path(args.resume_from).read_text()) if args.resume_from else None
    model, rec = B.fit_rig(
        rig,
        seed=args.seed,
        kappa=args.kappa,
        adam_steps=args.adam_steps,
        lbfgs_iters=args.lbfgs_iters,
        device=args.device,
        start=start,
        resume=start is not None,
        deadline_s=60.0 * args.deadline_min,
    )

    def P(m: Any, name: str) -> np.ndarray:
        if name in ("b0", "b1"):
            with torch.no_grad():
                return m.floor_exps()[int(name == "b1")].cpu().numpy()
        return m.params[name].detach().cpu().numpy()

    # line visibility on the planted data: per (rotor, order) the median over
    # the rotor's recordings of the mic-median line SNR
    R, K = rig.n_rotors, rig.k_prof
    snr = np.full((R, K), np.nan)
    for r in range(R):
        rows = []
        for j, s in enumerate(rig.settings):
            if rig.rotors[j] == r:
                row = np.full(K, np.nan)
                row[: s.k_max] = np.median(rig.tables[j]["snr_db"], axis=0)
                rows.append(row)
        # orders above every recording of the rotor are all-NaN by intent
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            snr[r] = np.nanmedian(np.asarray(rows), axis=0)
    snr = np.nan_to_num(snr, nan=-np.inf)
    pa_t, pa_f = P(truth, "p_aero"), P(model, "p_aero")
    v10 = snr >= 10.0
    err10 = np.abs(pa_f - pa_t)[v10]
    with torch.no_grad():
        W_t, W_f = truth.width_phase().cpu().numpy(), model.width_phase().cpu().numpy()
        basis = model.frozen[0].grid.floor
        f_psd = model.frozen[0].floor_f.cpu().numpy()
        inband = (f_psd >= B.BAND_HZ[0]) & (f_psd <= B.BAND_HZ[1])
        fl_t = 10 * np.log10(basis.psd(truth._floor_params()).cpu().numpy()[inband])
        fl_f = 10 * np.log10(basis.psd(model._floor_params()).cpu().numpy()[inband])
    v15 = snr.max(axis=0) >= 15.0
    werr = np.abs(W_f[v15] / W_t[v15] - 1.0)
    eps = (float(np.exp(P(truth, "log_eps"))), float(np.exp(P(model, "log_eps"))))
    lam = (float(np.exp(P(truth, "log_lam_ref"))), float(np.exp(P(model, "log_lam_ref"))))
    checks: dict[str, Any] = dict(
        profiles_10db=dict(
            n_lines=int(v10.sum()),
            median_abs_err_db=float(np.median(err10)) if err10.size else None,
            frac_within_1db=float(np.mean(err10 <= 1.0)) if err10.size else None,
            ok=bool(err10.size and np.all(err10 <= 1.0)),
        ),
        alpha_aero=_within(
            float(P(model, "alpha_aero")), float(P(truth, "alpha_aero")), abs_tol=0.3
        ),
        b0=_within(float(P(model, "b0")), float(P(truth, "b0")), abs_tol=0.3),
        floor=dict(
            max_abs_err_db=float(np.max(np.abs(fl_f - fl_t))),
            rms_err_db=float(np.sqrt(np.mean((fl_f - fl_t) ** 2))),
            mean_err_db=float(np.mean(fl_f - fl_t)),
            ok=bool(np.max(np.abs(fl_f - fl_t)) <= 1.0),
        ),
        eps=_within(eps[1], eps[0], rel_tol=0.2),
        lam_ref=_within(lam[1], lam[0], rel_tol=0.2),
        width_phase_15db=dict(
            orders=(np.flatnonzero(v15) + 1).tolist(),
            median_rel_err=float(np.median(werr)) if werr.size else None,
            frac_within_30pct=float(np.mean(werr <= 0.3)) if werr.size else None,
            ok=bool(werr.size and np.all(werr <= 0.3)),
        ),
    )
    checks["all_ok"] = all(bool(v["ok"]) for v in checks.values() if isinstance(v, dict))
    out = dict(
        truth=str(args.truth),
        floor_patch=floor_patch,
        kappa=args.kappa,
        render_seed=args.render_seed,
        seed=args.seed,
        checks=checks,
        info=dict(
            alpha_motor=_within(
                float(P(model, "alpha_motor")), float(P(truth, "alpha_motor")), abs_tol=0.3
            ),
            b1=_within(float(P(model, "b1")), float(P(truth, "b1")), abs_tol=0.1),
            a_sigma=_within(float(P(model, "a_sigma")), float(P(truth, "a_sigma")), abs_tol=0.2),
            b_lam=_within(float(P(model, "b_lam")), float(P(truth, "b_lam")), abs_tol=0.2),
        ),
        objective=rec["objective"],
        converged=rec["converged"],
        restart_gain_per_cell=rec["restart_gain_per_cell"],
        held_orders=rec["evidence"]["held_orders"],
        wall_s=rec["wall_s"],
        git=_git(),
    )
    stem = Path(args.truth).stem + (
        f"__floor{args.floor_data_below:g}" if args.floor_data_below is not None else ""
    )
    path = args.out / "planted" / f"{stem}__kappa{args.kappa:g}__r{args.render_seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    path.with_suffix(".fit.json").write_text(json.dumps(B.payload(model, rec, git=_git())))
    print(json.dumps(out["checks"], indent=1))
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    """A continuation: the convergence loop of :func:`bench_v4.fit_rig` from a
    payload (no Adam, no evidence freeze), written as ``<stem>__c<n+1>``;
    ``--kappa`` continues under another floor push (``__kappa<K>`` in the stem)."""
    import torch

    from experiments.noise_model import bench_v4 as B

    torch.set_num_threads(args.threads)
    src_path = Path(args.source)
    src = json.loads(src_path.read_text())
    rig = _rig_of_payload(src)
    model, rec = B.fit_rig(
        rig,
        seed=int(src["optimiser"]["seed"]),
        kappa=float(src["priors"]["floor_push_kappa"] if args.kappa is None else args.kappa),
        lbfgs_iters=args.lbfgs_iters,
        device=args.device,
        flat_shaft=bool(src["params"]["shaft"]["pinned"]),
        start=src,
        resume=True,
        deadline_s=60.0 * args.deadline_min,
    )
    pay = B.payload(model, rec, git=_git())
    pay["optimiser"]["resumed_from"] = str(src_path)
    m = re.fullmatch(r"(.*?)(?:__c(\d+))?", src_path.stem)
    assert m is not None
    stem = m.group(1)
    if args.kappa is not None:
        stem = re.sub(r"__kappa[^_]+", f"__kappa{args.kappa:g}", stem)
    path = args.out / "restarts" / f"{stem}__c{int(m.group(2) or 0) + 1}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pay))
    print(
        f"wrote {path}: {rec['objective']:.1f} nats (from {rec['objective_start']:.1f}); "
        f"converged {rec['converged']} ({rec['stop']}, {len(rec['convergence_passes'])} passes, "
        f"last gain {rec['restart_gain_per_cell']})",
        flush=True,
    )
    return 0


def cmd_refit(args: argparse.Namespace) -> int:
    """The nested test of the shaft law: re-optimise a fitted payload with
    ``a_sigma = b_lam = 0`` (one ``(sigma_nu, lam)`` for every speed) under
    :func:`bench_v4.fit_rig`'s schedule, and compare."""
    import torch

    from experiments.noise_model import bench_v4 as B

    torch.set_num_threads(args.threads)
    src_path = Path(args.source)
    src = json.loads(src_path.read_text())
    kappa = float(src["priors"]["floor_push_kappa"] if args.kappa is None else args.kappa)
    rig = _rig_of_payload(src)
    model, rec = B.fit_rig(
        rig,
        seed=int(src["optimiser"]["seed"]),
        kappa=kappa,
        adam_steps=args.adam_steps,
        lbfgs_iters=args.lbfgs_iters,
        device=args.device,
        flat_shaft=bool(args.flat_shaft),
        start=src,
        deadline_s=60.0 * args.deadline_min,
    )
    pay = B.payload(model, rec, git=_git())
    uid = src_path.stem + ("__flat" if args.flat_shaft else "__refit")
    path = args.out / "restarts" / f"{uid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pay))
    o_src, o_new = src["objective"], pay["objective"]
    n = int(o_new["n_cells"])
    mv = B.group_movement(B.payload_group_vectors(src), B.payload_group_vectors(pay))
    out = dict(
        source=str(src_path),
        refit=str(path),
        flat_shaft=bool(args.flat_shaft),
        kappa=kappa,
        source_kappa=src["priors"]["floor_push_kappa"],
        n_cells=n,
        objective_source=o_src["total_nats"],
        objective_refit_start=rec["objective_start"],
        objective_refit=o_new["total_nats"],
        delta_total_nats=o_new["total_nats"] - o_src["total_nats"],
        delta_total_per_cell=(o_new["total_nats"] - o_src["total_nats"]) / n,
        delta_whittle_nats=o_new["whittle_nats"] - o_src["whittle_nats"],
        delta_whittle_per_cell=(o_new["whittle_nats"] - o_src["whittle_nats"]) / n,
        note="the flat refit drops the a_sigma and b_lam prior terms, so its total "
        "objective is not strictly nested; the Whittle delta is the like-for-like term",
        movement=dict(
            levels=mv["levels"],
            exponents_max=mv["exponents"],
            floor=mv["floor"],
            shaft_logs=mv["shaft_logs"],
            pedestal_logs=mv["pedestal_logs"],
        ),
        shaft_source=src["params"]["shaft"],
        shaft_refit=pay["params"]["shaft"],
        converged=rec["converged"],
        restart_gain_per_cell=rec["restart_gain_per_cell"],
        wall_s=rec["wall_s"],
    )
    rep = args.out / "refit" / f"{uid}.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


def cmd_eta(args: argparse.Namespace) -> int:
    """Step 3: the likelihood weight ``eta`` by block jackknife at a fitted
    payload, on the real recordings or (``--planted``) on the planted twin the
    planted report describes."""
    import torch

    from experiments.noise_model import bench_eta as E
    from experiments.noise_model import bench_v4 as B

    torch.set_num_threads(args.threads)
    pay = json.loads(Path(args.fit).read_text())
    kappa = float(pay["priors"]["floor_push_kappa"])
    if args.planted:
        rep = json.loads(Path(args.planted).read_text())
        patch = rep.get("floor_patch")
        _, rig, _ = _planted(
            rep["truth"],
            floor_below=None if patch is None else patch["below_hz"],
            render_seed=int(rep["render_seed"]),
            kappa=kappa,
            device=args.device,
        )
    else:
        rig = _rig_of_payload(pay)
    model = B.RigModel(
        rig, device=args.device, kappa=kappa, flat_shaft=bool(pay["params"]["shaft"]["pinned"])
    )
    model.load_payload(pay)
    path = args.out / "eta" / (Path(args.fit).stem + ".json")
    rep_out = E.eta_report(model, band_hz=args.band_hz, save=path.with_suffix(".npz"))
    rep_out.update(fit=str(args.fit), planted=args.planted, git=_git())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rep_out, indent=1))
    print(json.dumps({k: v for k, v in rep_out.items() if k != "params"}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    kappa_help = "floor push (nats per dB per control value); set by the planted check"
    p = sub.add_parser("fit")
    p.add_argument("--motors", nargs="+", default=list(ALL_MOTORS))
    p.add_argument("--throttles", nargs="+", type=int, default=list(ALL_THROTTLES))
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--kappa", type=float, required=True, help=kappa_help)
    p.add_argument("--tag", default="", help="suffix of a schedule variant")
    p.add_argument("--adam-steps", type=int, default=1500)
    p.add_argument("--lbfgs-iters", type=int, default=1200)
    p.add_argument("--deadline-min", type=float, default=None, help="stop the loop before this")
    p.add_argument("--device", default="cpu")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument(
        "--flat-shaft",
        action="store_true",
        help="pin a_sigma = b_lam = 0: one shaft for all speeds",
    )
    p.set_defaults(func=cmd_fit)
    p = sub.add_parser("reduce")
    p.set_defaults(func=cmd_reduce)
    p = sub.add_parser("planted")
    p.add_argument("--truth", required=True, help="a noise-v4-fit/3 bench payload")
    p.add_argument("--kappa", type=float, required=True, help=kappa_help)
    p.add_argument("--render-seed", type=int, default=1)
    p.add_argument("--seed", type=int, default=0, help="the fit's restart index")
    p.add_argument("--adam-steps", type=int, default=1500)
    p.add_argument("--lbfgs-iters", type=int, default=1200)
    p.add_argument("--deadline-min", type=float, default=50.0)
    p.add_argument(
        "--resume-from",
        default=None,
        help="continue this planted fit (its .fit.json) on the same re-rendered twin",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument(
        "--floor-data-below",
        type=float,
        default=None,
        metavar="HZ",
        help="replace the truth's floor below HZ by the measured inter-line floor",
    )
    p.set_defaults(func=cmd_planted)
    p = sub.add_parser("refit", help="nested refit of a payload with the shaft law pinned flat")
    p.add_argument("--from", dest="source", required=True, help="a noise-v4-fit/3 bench payload")
    p.add_argument("--flat-shaft", action="store_true", help="pin a_sigma = b_lam = 0")
    p.add_argument("--kappa", type=float, default=None, help="default: the source's kappa")
    p.add_argument("--adam-steps", type=int, default=300)
    p.add_argument("--lbfgs-iters", type=int, default=1200)
    p.add_argument("--deadline-min", type=float, default=50.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.set_defaults(func=cmd_refit)
    p = sub.add_parser("resume", help="a continuation of a payload's convergence loop")
    p.add_argument("--from", dest="source", required=True, help="a noise-v4-fit/3 bench payload")
    p.add_argument("--lbfgs-iters", type=int, default=1200, help="loop passes of a quarter")
    p.add_argument("--deadline-min", type=float, default=52.0)
    p.add_argument("--kappa", type=float, default=None, help="default: the source's kappa")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.set_defaults(func=cmd_resume)
    p = sub.add_parser("eta", help="step 3: the likelihood weight by block jackknife")
    p.add_argument("--fit", required=True, help="a noise-v4-fit/3 bench payload")
    p.add_argument("--planted", default=None, help="the planted report of a planted twin's fit")
    p.add_argument("--band-hz", type=float, default=200.0, help="the finest block's band")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.set_defaults(func=cmd_eta)
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
