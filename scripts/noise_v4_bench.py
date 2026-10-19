"""Noise model v4 — the bench fit driver (``docs/experiments/noise-model-v4-bench.md``).

Fit procedure A of ``docs/explainers/noise-model-v4.qmd`` on the DREGON
single-rotor ladders, the planted-synthetic check and the acceptance checks::

    python scripts/noise_v4_bench.py fit --motors Motor1 Motor2 --seeds 4 --device cuda
    python scripts/noise_v4_bench.py fit --motors Motor1 --seeds 4 --exclude-windy
    python scripts/noise_v4_bench.py planted --motor Motor1 [--truth <fit.json>]
    python scripts/noise_v4_bench.py reduce
    python scripts/noise_v4_bench.py check

Outputs under ``--out`` (default ``results/noise_v4/bench``): ``restarts/`` one
``noise-v4-fit/1`` payload per (motor, seed[, no-windy]); ``raw/`` the gridrun
unit records; ``fits/<motor>[__nowindy].json`` the selected restart (lowest
objective) with the restart spread; ``planted/<motor>.json``;
``checks.json`` + ``checks.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.gridrun import Unit, run_grid  # noqa: E402

OUT = Path("results/noise_v4/bench")
WIDTH_LAW = Path("results/static_rig/single_rotor/width_law.json")
#: The recordings whose quick fit fell into the random-walk corner (lam < 3).
RW_CORNER = ("motor_Motor1_90", "motor_Motor2_80", "motor_Motor3_90", "motor_Motor4_90")


def _git() -> str | None:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or None


def _variant(exclude_windy: bool) -> str:
    return "__nowindy" if exclude_windy else ""


def _fit_worker(unit: Unit) -> dict[str, Any]:
    import torch

    from experiments.noise_model import bench_v4 as B

    prm = unit.params
    torch.set_num_threads(int(prm["threads"]))
    lad = B.build_ladder(prm["motor"])
    exclude = B.WIND_MICS[prm["motor"]] if prm["exclude_windy"] else ()
    model, rec = B.fit_ladder(
        lad,
        seed=int(prm["seed"]),
        adam_steps=int(prm["adam_steps"]),
        lbfgs_iters=int(prm["lbfgs_iters"]),
        device=prm["device"],
        exclude_mics=exclude,
    )
    pay = B.payload(model, rec, git=prm["git"])
    path = Path(prm["out"]) / "restarts" / f"{unit.uid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pay))
    return dict(
        uid=unit.uid,
        motor=prm["motor"],
        seed=int(prm["seed"]),
        exclude_windy=bool(prm["exclude_windy"]),
        objective=rec["objective"],
        converged=rec["converged"],
        restart_gain_per_cell=rec["restart_gain_per_cell"],
        alpha_aero=pay["shared"]["alpha_aero"],
        alpha_motor=pay["shared"]["alpha_motor"],
        b0=pay["shared"]["floor"]["b0"],
        b1=pay["shared"]["floor"]["b1"],
        sigma_nu=[s["sigma_nu_rad_s"] for s in pay["settings"]],
        lam=[s["lam_per_s"] for s in pay["settings"]],
        eval_s=rec["eval_s"],
        wall_s=rec["wall_s"],
        path=str(path),
    )


def cmd_fit(args: argparse.Namespace) -> int:
    git = _git()
    units = [
        Unit(
            uid=f"{m}__s{s}{_variant(args.exclude_windy)}",
            params=dict(
                motor=m,
                seed=s,
                exclude_windy=bool(args.exclude_windy),
                adam_steps=args.adam_steps,
                lbfgs_iters=args.lbfgs_iters,
                device=args.device,
                threads=args.threads,
                out=str(args.out),
                git=git,
            ),
        )
        for m in args.motors
        for s in range(args.seeds)
    ]
    res = run_grid(units, _fit_worker, str(args.out), jobs=args.jobs, mp_context="spawn")
    return res.exit_code


def cmd_reduce(args: argparse.Namespace) -> int:
    import numpy as np

    groups: dict[str, list[dict[str, Any]]] = {}
    for p in sorted((args.out / "restarts").glob("*.json")):
        pay = json.loads(p.read_text())
        name = p.stem.split("__s")[0] + ("__nowindy" if p.stem.endswith("__nowindy") else "")
        groups.setdefault(name, []).append(dict(path=str(p), pay=pay))
    (args.out / "fits").mkdir(parents=True, exist_ok=True)
    for name, items in groups.items():
        obj = np.array([it["pay"]["objective"]["total_nats"] for it in items])
        n_cells = items[0]["pay"]["objective"]["n_cells"]
        best = items[int(np.argmin(obj))]["pay"]
        best["restarts"] = dict(
            paths=[it["path"] for it in items],
            seeds=[it["pay"]["optimiser"]["seed"] for it in items],
            objective=obj.tolist(),
            converged=[it["pay"]["optimiser"]["converged"] for it in items],
            restart_gain_per_cell=[it["pay"]["optimiser"]["restart_gain_per_cell"] for it in items],
            best_minus_median_per_cell=float((np.median(obj) - obj.min()) / n_cells),
            best_minus_worst_per_cell=float((obj.max() - obj.min()) / n_cells),
            alpha_aero=[it["pay"]["shared"]["alpha_aero"] for it in items],
            alpha_motor=[it["pay"]["shared"]["alpha_motor"] for it in items],
            sigma_nu=[[s["sigma_nu_rad_s"] for s in it["pay"]["settings"]] for it in items],
            lam=[[s["lam_per_s"] for s in it["pay"]["settings"]] for it in items],
            rule="reported fit = restart with the lowest total objective",
        )
        (args.out / "fits" / f"{name}.json").write_text(json.dumps(best))
        print(
            f"{name}: {len(items)} restarts, best {obj.min():.1f}, spread "
            f"{best['restarts']['best_minus_worst_per_cell']:.2e}/cell, converged "
            f"{sum(best['restarts']['converged'])}/{len(items)}"
        )
    return 0


def cmd_planted(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    from experiments.noise_model import bench_v4 as B

    torch.set_num_threads(args.threads)
    real = B.build_ladder(args.motor)
    truth = B.LadderModel(real, device=args.device)
    if args.truth:
        truth.load_payload(json.loads(Path(args.truth).read_text()))
    planted = B.render_planted(truth, seed=args.render_seed)
    lad = B.build_ladder(args.motor, settings=planted)
    model, rec = B.fit_ladder(
        lad,
        seed=0,
        adam_steps=args.adam_steps,
        lbfgs_iters=args.lbfgs_iters,
        device=args.device,
    )
    t = {k: v.detach().cpu().numpy() for k, v in truth.params.items()}
    f = {k: v.detach().cpu().numpy() for k, v in model.params.items()}
    s2_t = truth.sigma2().detach().cpu().numpy()
    s2_f = model.sigma2().detach().cpu().numpy()
    rows = []
    for j, s in enumerate(lad.settings):
        tab = lad.tables[j]
        snr = np.median(tab["snr_db"], axis=0)
        K = s.k_max
        prof_t = t["p_aero"][:K]
        prof_f = f["p_aero"][:K]
        vis10 = snr >= 10.0
        vis15 = snr >= 15.0
        rows.append(
            dict(
                key=s.key,
                sigma_nu_rel_err=float(math.exp(f["log_sigma_nu"][j] - t["log_sigma_nu"][j]) - 1.0),
                lam_rel_err=float(math.exp(f["log_lam"][j] - t["log_lam"][j]) - 1.0),
                n_lines_15db=int(vis15.sum()),
                sigma2_rel_err_median_15db=float(
                    np.median(np.abs(s2_f[:K][vis15] / s2_t[:K][vis15] - 1.0))
                )
                if vis15.any()
                else None,
                sigma2_within_30pct_15db=float(
                    np.mean(np.abs(s2_f[:K][vis15] / s2_t[:K][vis15] - 1.0) <= 0.3)
                )
                if vis15.any()
                else None,
                profile_abs_err_db_median_10db=float(
                    np.median(np.abs(prof_f[vis10] - prof_t[vis10]))
                )
                if vis10.any()
                else None,
                profile_within_1db_10db=float(np.mean(np.abs(prof_f[vis10] - prof_t[vis10]) <= 1.0))
                if vis10.any()
                else None,
            )
        )
    out = dict(
        motor=args.motor,
        truth=args.truth or "seed model of the real ladder",
        render_seed=args.render_seed,
        alpha_aero=dict(truth=float(t["alpha_aero"]), fit=float(f["alpha_aero"])),
        alpha_motor=dict(truth=float(t["alpha_motor"]), fit=float(f["alpha_motor"])),
        b0=dict(truth=float(t["b0"]), fit=float(f["b0"])),
        b1=dict(truth=float(t["b1"]), fit=float(f["b1"])),
        settings=rows,
        objective=rec["objective"],
        converged=rec["converged"],
        restart_gain_per_cell=rec["restart_gain_per_cell"],
        wall_s=rec["wall_s"],
        git=_git(),
    )
    path = args.out / "planted" / f"{args.motor}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


def _hwhm_hz(k: float, sigma_nu: float, lam: float, s2: float, gam: float) -> float:
    """Half-width at half-maximum of the v4 line (no window) at order ``k``."""
    import numpy as np

    from data_processing.noise_model.lag import r_tau_v4

    dt = 1e-4
    tau = np.arange(0, 60.0, dt)
    r = r_tau_v4(tau, k, sigma_nu=sigma_nu, lam=lam, sigma2=s2, gamma_am_hz=gam)
    # the coherent core never decays when s2 < inf and the shaft is bounded at
    # low k; the half-width is read where the shape falls to half its peak
    nfft = 1 << 22
    shape = 2.0 * np.fft.rfft(r * dt, n=nfft).real - r[0] * dt
    f = np.fft.rfftfreq(nfft, d=dt)
    half = shape[0] / 2.0
    i = int(np.argmax(shape < half))
    return float(f[i])


def cmd_check(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    from experiments.noise_model import bench_v4 as B

    torch.set_num_threads(args.threads)
    width_law = json.loads(WIDTH_LAW.read_text())
    report: dict[str, Any] = dict(motors={}, git=_git())
    for fit_path in sorted((args.out / "fits").glob("Motor?.json")):
        pay = json.loads(fit_path.read_text())
        motor = pay["motor"]
        lad = B.build_ladder(motor)
        model = B.LadderModel(lad, device=args.device)
        model.load_payload(pay)
        sh = pay["shared"]
        s2 = np.asarray(sh["pedestal"]["sigma2"])
        gm = np.asarray(sh["pedestal"]["gamma_hz"])
        rows = []
        for j, (s, st) in enumerate(zip(lad.settings, pay["settings"])):
            with torch.no_grad():
                Mx = model.expected(j).cpu().numpy()
                w_aero, w_motor, motor_k = model.line_weights(j)
            f = s.freqs
            pw = s.power
            bands = {}
            for a, b in zip((30, 100, 300, 1000, 2000, 4000), (100, 300, 1000, 2000, 4000, 7200)):
                sel = (f >= a) & (f < b)
                r = pw[:, sel] / Mx[:, sel]
                bands[f"{a}-{b}"] = dict(
                    mean_ratio=float(r.mean()),
                    median_ratio_over_ln2=float(np.median(r) / math.log(2.0)),
                    level_db=float(10 * np.log10(pw[:, sel].mean() / Mx[:, sel].mean())),
                )
            # the line bins: data over model power in +-1 Hz around k f_r
            line_db = {}
            for k0, k1 in ((1, 4), (5, 10), (11, 20), (21, 40), (41, 80), (81, 160)):
                sel = np.zeros(f.size, bool)
                for k in range(k0, min(k1, s.k_max) + 1):
                    sel |= np.abs(f - k * s.carrier) <= 1.0
                if sel.any():
                    line_db[f"{k0}-{k1}"] = float(
                        10 * np.log10(pw[:, sel].sum() / Mx[:, sel].sum())
                    )
            # quick-fit agreement
            quick = B._quick_shaft(s.key)
            ou = json.loads((B.OU_DIR / f"{s.key}.json").read_text())
            mm = w_aero.cpu().numpy().mean(axis=0)
            if motor_k.size:
                mm[motor_k - 1] += w_motor.cpu().numpy().mean(axis=0)
            fit_db = 10 * np.log10(mm)
            ou_db = 10 * np.log10(np.asarray(ou["amp2"], dtype=float).mean(axis=0) / 2.0)
            snr = np.median(lad.tables[j]["snr_db"], axis=0)
            Kc = min(fit_db.size, ou_db.size)
            vis = snr[:Kc] >= 10.0
            prof_diff = fit_db[:Kc][vis] - ou_db[:Kc][vis]
            sig, lam = st["sigma_nu_rad_s"], st["lam_per_s"]
            row = dict(
                key=s.key,
                sigma_nu_rad_s=sig,
                lam_per_s=lam,
                quick=None
                if quick is None
                else dict(
                    sigma_nu_rad_s=quick[0],
                    lam_per_s=quick[1],
                    dlog_sigma=float(math.log(sig / quick[0])),
                    lam_ratio=float(lam / quick[1]),
                    pass_sigma=bool(abs(math.log(sig / quick[0])) <= 0.3),
                    pass_lam=bool(1 / 1.5 <= lam / quick[1] <= 1.5),
                ),
                profile_vs_quick_db=dict(
                    n_lines=int(vis.sum()),
                    median_abs=float(np.median(np.abs(prof_diff))) if vis.any() else None,
                    within_2db=float(np.mean(np.abs(prof_diff) <= 2.0)) if vis.any() else None,
                ),
                bands=bands,
                line_bins_db=line_db,
                whittle_per_cell=st["whittle_nats"] / st["n_cells"],
            )
            if s.key in RW_CORNER:
                k80 = np.asarray(width_law[s.key]["gamma_k"], dtype=float)
                ok = np.asarray(width_law[s.key]["ok"], dtype=bool)
                ks = np.arange(1, k80.size + 1)
                sel = ok & (ks >= 75) & (ks <= 85)
                free = float(np.median(k80[sel])) if sel.any() else None
                kk = min(80, s2.size)
                model_hw = _hwhm_hz(80.0, sig, lam, float(s2[kk - 1]), float(gm[kk - 1]))
                row["rw_corner"] = dict(
                    lam_in_5_30=bool(5.0 <= lam <= 30.0),
                    hwhm_k80_model_hz=model_hw,
                    hwhm_k80_free_hz=free,
                    hwhm_ratio=None if not free else model_hw / free,
                    pass_hwhm=None if not free else bool(1 / 1.5 <= model_hw / free <= 1.5),
                )
            rows.append(row)
        # the BPF line (k = 2) against its own law over the ladder
        bpf = []
        for j, s in enumerate(lad.settings):
            lev = float(lad.tables[j]["level_db"][:, 1].mean())
            law = sh["profile_aero_db"][1] + sh["alpha_aero"] * 10 * math.log10(s.carrier / B.F_REF)
            bpf.append(lev - law)
        bpf = np.asarray(bpf)
        exps = dict(
            alpha_aero=sh["alpha_aero"],
            alpha_motor=sh["alpha_motor"],
            b0=sh["floor"]["b0"],
            b1=sh["floor"]["b1"],
            pass_alpha_aero=bool(4.5 <= sh["alpha_aero"] <= 6.5),
            pass_alpha_motor=bool(-3.0 <= sh["alpha_motor"] <= 2.0),
            bpf_rms_db=float(np.sqrt(np.mean((bpf - bpf.mean()) ** 2))),
            pass_bpf=bool(np.sqrt(np.mean((bpf - bpf.mean()) ** 2)) <= 1.0),
        )
        rep = dict(
            settings=rows,
            exponents=exps,
            restarts=pay.get("restarts"),
            optimiser=dict(
                converged=pay["optimiser"]["converged"],
                restart_gain_per_cell=pay["optimiser"]["restart_gain_per_cell"],
                eval_s=pay["optimiser"]["eval_s"],
                wall_s=pay["optimiser"]["wall_s"],
            ),
            lag_support=pay["lag_support"],
        )
        nowindy = args.out / "fits" / f"{motor}__nowindy.json"
        if nowindy.exists():
            pw = json.loads(nowindy.read_text())
            a = np.asarray(sh["profile_aero_db"])
            b = np.asarray(pw["shared"]["profile_aero_db"])
            snr = np.median(lad.tables[2]["snr_db"], axis=0)
            vis = snr >= 10.0
            K = min(vis.size, a.size)
            d = a[:K][vis[:K]] - b[:K][vis[:K]]
            rep["wind"] = dict(
                n_lines=int(vis[:K].sum()),
                median_abs_db=float(np.median(np.abs(d))),
                max_abs_db=float(np.max(np.abs(d))),
                pass_median_0p5db=bool(np.median(np.abs(d)) < 0.5),
            )
        report["motors"][motor] = rep
    (args.out / "checks.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1)[:20000])
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--motors", nargs="+", default=["Motor1", "Motor2", "Motor3", "Motor4"])
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--exclude-windy", action="store_true")
    p.add_argument("--adam-steps", type=int, default=1500)
    p.add_argument("--lbfgs-iters", type=int, default=400)
    p.add_argument("--device", default="cpu")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--threads", type=int, default=8)
    p.set_defaults(func=cmd_fit)
    p = sub.add_parser("reduce")
    p.set_defaults(func=cmd_reduce)
    p = sub.add_parser("planted")
    p.add_argument("--motor", default="Motor1")
    p.add_argument("--truth", default=None, help="a fit payload; default: the seed model")
    p.add_argument("--render-seed", type=int, default=1)
    p.add_argument("--adam-steps", type=int, default=1500)
    p.add_argument("--lbfgs-iters", type=int, default=400)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.set_defaults(func=cmd_planted)
    p = sub.add_parser("check")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.set_defaults(func=cmd_check)
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
