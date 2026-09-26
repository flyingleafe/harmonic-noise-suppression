"""Noise v3 diagnosis, the GPU half: the rig step to its true optimum, the
exact ridge step, the zero-mean projection and the u_j notch decomposition.

Runs on ONE fitted pool (a ``noise-v3-fit/1`` JSON with its latents) whose
pool it rebuilds exactly as ``scripts/noise_v2_fit.py flight --mode flight_v3``
built it (channel-normalised, the fit's work rate, stride 4, every frame).
Every state it visits is priced the same way (:func:`price`): the Whittle term
in total and per band, the rig ``-log prior`` per site, the OU ``-log prior``
per family (quadratic part per family, full total), and the static share of
every latent family (the pool mean per track). Experiments (``--steps``):

``fit``      the recorded fit, rebuilt (checks the reconstruction).
``rig``      step 2: the rig's L-BFGS on every frame with the latents FROZEN,
             run to convergence (no relative stop; chunks of ``--chunk``
             iterations, each chunk's restart kept, until a chunk gains less
             than ``--tol-cell`` per cell), with its trace; plus where the
             production rule (``|df| <= 1e-5 |f|``) would have stopped.
``ridge``    the exact prior minimum along the Whittle-invariant ridge
             (:func:`experiments.noise_model.fit.static_ridge_step`) from the fit.
``alt``      from ``ridge``: ``--rounds`` rounds of latent step (converged),
             ridge step, rig step (converged).
``zeromean`` step 2 reverse: every latent track minus its pool mean, rig
             fixed (priced), then the latent step re-run (converged).
``notch``    step 3: the static part of ``u_j`` removed, rig fixed (priced);
             then the rig re-fitted to convergence with those latents; then
             the latent step re-run from the removed state.

Writes ``<out>/<pool>.json`` after every experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data_processing.noise_model.v3 import Wander
from experiments.noise_model import fit as FT
from experiments.noise_model import model as MD
from experiments.noise_model import supports as SU
from experiments.stochastic_fit.revised_phase import composite_risk

WHITTLE_BANDS_HZ = ((30, 300), (300, 700), (700, 1500), (1500, 3000), (3000, 5000), (5000, 8000))
PRODUCTION_RTOL = 1e-5
CHANNEL_GAINS = "results/noise_v2/mic_gains/mic_gains.json"
K_CAP = 130


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── the pool, the rig and the latents of a fit ─────────────────────────────


def build(
    fit: dict[str, Any],
    set_name: str,
    *,
    wind: bool,
    device: str,
    mics: list[int] | None = None,
    k_cap: int = K_CAP,
) -> dict[str, Any]:
    rig = "dregon" if fit["support"].startswith("dregon") else "michaels"
    specs = {s.name: s for s in SU.support_set(set_name)}
    members = []
    sups = [SU.load_cached(n) or SU.load_support(specs[n]) for n in fit["supports"]]
    for s in sups:
        power = np.asarray(s.power, dtype=np.float64)
        members.append(
            (
                s.name,
                power if mics is None else power[np.asarray(mics, dtype=np.int64)],
                np.asarray(s.carrier_rev_s_audio, dtype=np.float64),
                np.asarray(s.frame_starts, dtype=np.int64),
            )
        )
    batch = MD.flight_batch(
        name=str(fit["support"]),
        members=members,
        sr=int(sups[0].sr),
        n_fft=int(sups[0].n_fft),
        hop=int(sups[0].hop),
        k_cap=int(k_cap),
        frame_stride=int(fit["diagnostics"]["batch"]["frame_stride"]),
        max_frames=None,
        channel_gains=FT.load_channel_gains(CHANNEL_GAINS, rig=rig, mics=mics),
        device=device,
        chunk_frames=int(fit["optimiser"]["chunk_frames"]),
        sr_work=int(fit["front_end"]["sr_work"]),
    )
    record = dict(fit["priors"]["wander"])
    wander = Wander.from_mapping(record)
    priors = MD.PriorsV3(wind=wind, wander=wander, wander_record=record)
    blocked = MD.with_blocks(replace(batch, latents=None), wander.block_s)
    start = FT.seeds(blocked, mode=MD.V3_MODE, priors=priors, pin=None, profile_init=None)
    measured_batch = replace(blocked, measured=start.measured)
    return dict(
        batch=blocked,
        measured_batch=measured_batch,
        measured=start.measured,
        priors=priors,
        wander=wander,
    )


def sites_of(fit: dict[str, Any], device: str) -> dict[str, torch.Tensor]:
    p = fit["params"]
    pinned = set(fit["diagnostics"].get("span_pins", {}).get("pinned", []))

    def t(v: Any) -> torch.Tensor:
        return torch.as_tensor(np.asarray(v, dtype=np.float64), dtype=torch.float64, device=device)

    out = dict(
        sigma_nu=t(p["sigma_nu"]),
        gamma_hz=t(p["gamma_hz"]),
        profile_db=t(p["profile"]["profile_db"]),
        floor_shape_z=t(p["floor"]["floor_shape_z"]),
    )
    if "amp_exp" not in pinned:
        out["amp_exp"] = t(p["profile"]["amp_exp"])
    if "floor_exp" not in pinned:
        out["floor_exp"] = t(p["floor"]["floor_exp"])
    if "floor_static_rel" not in pinned:
        out["floor_static_rel"] = t(p["floor"]["floor_static_rel"])
    if p.get("wind") is not None:
        out["wind_db"] = t(p["wind"]["wind_db"])
    return out


def latents_of(fit: dict[str, Any], device: str) -> dict[int, MD.WindowLatents]:
    def t(v: Any) -> torch.Tensor:
        return torch.as_tensor(np.asarray(v, dtype=np.float64), dtype=torch.float64, device=device)

    return {
        int(w["window"]): MD.WindowLatents(d=t(w["d"]), v=t(w["v"]), u=t(w["u"]), uj=t(w["uj"]))
        for w in fit["latents"]["windows"]
    }


# ── pricing ────────────────────────────────────────────────────────────────


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float64)


def static_stats(latents: dict[int, MD.WindowLatents]) -> dict[str, Any]:
    """Static share (pool mean per track) of each family's mean square."""
    out: dict[str, Any] = {}
    for name in ("d", "v", "u", "uj"):
        xs = [
            _np(getattr(latents[w], name))
            for w in sorted(latents)
            if getattr(latents[w], name) is not None
        ]
        if not xs:
            continue
        cat = np.concatenate(xs, axis=-1)
        ms = float((cat**2).mean())
        mean = cat.mean(axis=-1, keepdims=True)
        st = float((np.broadcast_to(mean, cat.shape) ** 2).mean())
        out[name] = dict(
            rms_db=math.sqrt(ms),
            static_share=st / ms if ms > 0 else 0.0,
            static_rms_db=math.sqrt(st),
        )
    return out


def ou_by_family(latents: dict[int, MD.WindowLatents], wander: Wander) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in ("d", "v", "u", "uj"):
        sub = {w: MD.WindowLatents(**{name: getattr(latents[w], name)}) for w in latents}
        out[name] = MD.ou_prior_nats(sub, wander)
    out["total"] = MD.ou_prior_nats(latents, wander)
    return out


def rig_prior_by_site(
    batch: MD.SupportBatch, sites: dict[str, torch.Tensor], priors: Any
) -> dict[str, float]:
    acc: dict[str, float] = {}

    def lookup(name: str, d: Any) -> torch.Tensor:
        v = torch.as_tensor(sites[name], dtype=torch.float64, device=batch.power.device)
        acc[name] = -float(d.log_prob(v).sum())
        return v

    MD.sample_params(batch, mode=MD.V3_MODE, priors=priors, site=lookup)
    acc["total"] = float(sum(acc.values()))
    return acc


def price(
    ctx: dict[str, Any], sites: dict[str, torch.Tensor], latents: dict[int, MD.WindowLatents]
) -> dict[str, Any]:
    mb = ctx["measured_batch"]
    params = MD.sample_params_from_values(mb, mode=MD.V3_MODE, values=sites, priors=ctx["priors"])
    b = replace(ctx["batch"], latents=latents)
    with torch.no_grad():
        m = MD.forward(b, params, unit_autocorr=True)
        whittle = float(MD.whittle_risk(b, m))
        freqs = torch.as_tensor(np.asarray(b.grid.freqs_hz, dtype=np.float64), device=m.device)
        bands = {}
        for lo, hi in WHITTLE_BANDS_HZ:
            mask = b.band & (freqs >= lo) & (freqs < hi)
            bands[f"{lo}-{hi}"] = float(composite_risk(b.power, m, b.weights, band=mask))
    rig = rig_prior_by_site(mb, sites, ctx["priors"])
    ou = ou_by_family(latents, ctx["wander"])
    return dict(
        whittle_nats=whittle,
        whittle_band_nats=bands,
        rig_prior_nats=rig,
        ou_prior_nats=ou,
        total_nats=whittle + rig["total"] + ou["total"],
        static=static_stats(latents),
    )


def delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """``b - a`` of every priced term."""
    return dict(
        whittle_nats=b["whittle_nats"] - a["whittle_nats"],
        whittle_band_nats={
            k: b["whittle_band_nats"][k] - a["whittle_band_nats"][k] for k in a["whittle_band_nats"]
        },
        rig_prior_nats={
            k: b["rig_prior_nats"][k] - a["rig_prior_nats"][k] for k in a["rig_prior_nats"]
        },
        ou_prior_nats={
            k: b["ou_prior_nats"][k] - a["ou_prior_nats"][k] for k in a["ou_prior_nats"]
        },
        total_nats=b["total_nats"] - a["total_nats"],
    )


def rig_view(ctx: dict[str, Any], sites: dict[str, torch.Tensor]) -> dict[str, Any]:
    """The rig numbers the moves are read in: profile, control values, widths."""
    sb = float(ctx["measured"].floor_shape_sd_db)
    chol = _np(ctx["batch"].grid.floor.shape_chol)
    return dict(
        profile_db=_np(sites["profile_db"]).tolist(),
        ctrl_db=(sb * chol @ _np(sites["floor_shape_z"])).tolist(),
        gamma_hz=_np(sites["gamma_hz"]).tolist(),
        sigma_nu=float(_np(sites["sigma_nu"])),
    )


# ── steps ──────────────────────────────────────────────────────────────────


def rig_converge(
    ctx: dict[str, Any],
    sites: dict[str, torch.Tensor],
    latents: dict[int, MD.WindowLatents],
    *,
    chunk: int,
    max_iters: int,
    tol_cell: float,
    wall_s: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """The rig's all-frames L-BFGS with the latents fixed, no relative stop,
    in chunks (each a fresh L-BFGS pass + its half-budget restart) until a
    chunk gains less than ``tol_cell`` per cell or a cap is hit."""
    b = replace(ctx["batch"], latents=latents)
    n_cells = int(b.n_cells)
    optim = FT.OptimSpec(
        adam_steps=0,
        lbfgs_iters=int(chunk),
        lbfgs_frames=None,
        lbfgs_rtol=None,
        init_jitter=0.0,
        seed=0,
    )
    cur = {k: v.detach().clone() for k, v in sites.items()}
    trace: list[dict[str, Any]] = []
    t0 = time.time()
    iters = 0
    stop = "max_iters"
    while iters < max_iters:
        t1 = time.time()
        out = FT.fit_support(
            b,
            mode=MD.V3_MODE,
            priors=ctx["priors"],
            pin=None,
            optim=optim,
            forward_kw=dict(unit_autocorr=True),
            profile_init=None,
            progress=0,
            start=FT.Seeds(measured=ctx["measured"], init=cur),
        )
        o = out.optimiser
        n_it = int(o["lbfgs_iters_used"]) + int(o["lbfgs_restart_iters_used"])
        iters += n_it
        gain = float(o["lbfgs_loss_before"]) - float(o["lbfgs_loss_after"])
        trace.append(
            dict(
                loss_before=float(o["lbfgs_loss_before"]),
                loss_first_pass=float(o["lbfgs_loss_first_pass"]),
                loss_after=float(o["lbfgs_loss_after"]),
                iters=n_it,
                evals=int(o["lbfgs_evals"]) + int(o["lbfgs_restart_evals"]),
                gain_nats=gain,
                grad_norm=float(o["grad_norm"]),
                wall_s=time.time() - t1,
            )
        )
        cur = {k: v.detach().clone() for k, v in out.sites.items()}
        log(
            f"    rig chunk {len(trace)}: {n_it} iters, gain {gain:.1f} nats ({gain / n_cells:.2e}/cell), "
            f"loss {o['lbfgs_loss_after']:.2f}, {time.time() - t1:.0f} s"
        )
        if gain < tol_cell * n_cells:
            stop = "tol"
            break
        if time.time() - t0 > wall_s:
            stop = "wall"
            break
    return cur, dict(
        trace=trace,
        iters=iters,
        stop=stop,
        wall_s=time.time() - t0,
        s_per_eval=sum(t["wall_s"] for t in trace) / max(1, sum(t["evals"] for t in trace)),
        gain_nats=trace[0]["loss_before"] - trace[-1]["loss_after"],
    )


def production_stop(
    ctx: dict[str, Any],
    sites: dict[str, torch.Tensor],
    latents: dict[int, MD.WindowLatents],
    iters: int,
) -> dict[str, Any]:
    """Where the production stop fires on the rig's all-frames L-BFGS from
    ``sites``: the production call itself (``lbfgs_rtol`` 1e-5, 500
    iterations and its restart, as the polish runs it) and the same pass
    without the relative stop for ``iters`` iterations, each with its
    per-iteration objective trace, and on the latter every iteration's move
    against the threshold ``1e-5 max(|f_k|, |f_{k-1}|)``."""
    b = replace(ctx["batch"], latents=latents)

    def run(n: int, rtol: float | None) -> dict[str, Any]:
        out = FT.fit_support(
            b,
            mode=MD.V3_MODE,
            priors=ctx["priors"],
            optim=FT.OptimSpec(
                adam_steps=0,
                lbfgs_iters=int(n),
                lbfgs_frames=None,
                lbfgs_rtol=rtol,
                init_jitter=0.0,
                seed=0,
            ),
            forward_kw=dict(unit_autocorr=True),
            start=FT.Seeds(measured=ctx["measured"], init=sites),
        )
        return out.optimiser

    prod = run(500, PRODUCTION_RTOL)
    free = run(int(iters), None)
    losses = [float(x) for x in free["lbfgs_trace_first_pass"]]
    steps = [losses[i - 1] - losses[i] for i in range(1, len(losses))]
    thresh = [
        PRODUCTION_RTOL * max(abs(losses[i]), abs(losses[i - 1]), 1.0)
        for i in range(1, len(losses))
    ]
    fired = next(
        (i + 1 for i, (s, th) in enumerate(zip(steps, thresh, strict=True)) if abs(s) <= th), None
    )
    log(
        f"    production: {prod['lbfgs_iters_used']}+{prod['lbfgs_restart_iters_used']} iterations; "
        f"free pass: {len(losses) - 1} iterations, rtol would fire at {fired}"
    )
    return dict(
        production=dict(
            (k, prod[k])
            for k in (
                "lbfgs_loss_before",
                "lbfgs_loss_first_pass",
                "lbfgs_loss_after",
                "lbfgs_iters_used",
                "lbfgs_restart_iters_used",
                "lbfgs_restart_gain_per_cell",
                "converged",
                "grad_norm",
                "lbfgs_trace_first_pass",
                "lbfgs_trace_restart",
            )
        ),
        losses=losses,
        step_nats=steps,
        rtol_threshold_nats=thresh,
        rtol_fires_at_iter=fired,
    )


def latent_converge(
    ctx: dict[str, Any],
    sites: dict[str, torch.Tensor],
    latents: dict[int, MD.WindowLatents],
    *,
    iters: int,
    passes: int,
) -> tuple[dict[int, MD.WindowLatents], dict[str, Any]]:
    params = MD.detach_params(
        MD.sample_params_from_values(
            ctx["measured_batch"], mode=MD.V3_MODE, values=sites, priors=ctx["priors"]
        )
    )
    recs = []
    cur = latents
    for _ in range(int(passes)):
        cur, rec = FT.fit_latents(
            ctx["measured_batch"], params, wander=ctx["wander"], init=cur, iters=int(iters)
        )
        recs.append(
            {k: rec[k] for k in ("loss_before", "loss_after", "evals", "wall_s") if k in rec}
        )
        log(
            f"    latent pass: {rec['loss_before']:.2f} -> {rec['loss_after']:.2f} ({rec['evals']} evals)"
        )
        if rec["loss_before"] - rec["loss_after"] < 1e-6 * int(ctx["batch"].n_cells):
            break
    return cur, dict(passes=recs)


def ridge(
    ctx: dict[str, Any],
    sites: dict[str, torch.Tensor],
    latents: dict[int, MD.WindowLatents],
    families: tuple[str, ...] = ("d", "v", "u", "uj"),
) -> tuple[dict[str, torch.Tensor], dict[int, MD.WindowLatents], dict[str, Any]]:
    return FT.static_ridge_step(
        sites,
        latents,
        wander=ctx["wander"],
        measured=ctx["measured"],
        priors=ctx["priors"],
        shape_chol=ctx["batch"].grid.floor.shape_chol,
        families=families,
    )


def remove_static(
    latents: dict[int, MD.WindowLatents], names: tuple[str, ...], index: int | None = None
) -> dict[int, MD.WindowLatents]:
    """Every track of ``names`` minus its pool mean (``index``: only that
    control point of ``uj``)."""
    ws = sorted(latents)
    out = {w: MD.WindowLatents(**latents[w].tracks()) for w in ws}
    for name in names:
        xs = [getattr(latents[w], name) for w in ws]
        if xs[0] is None:
            continue
        mean = torch.cat(xs, dim=-1).mean(dim=-1, keepdim=True)
        if index is not None:
            keep = torch.zeros_like(mean)
            keep[index] = mean[index]
            mean = keep
        for w in ws:
            setattr(out[w], name, getattr(latents[w], name) - mean)
    return out


# ── driver ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fit", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--wind", action="store_true")
    ap.add_argument("--out", default="results/noise_v3/diag/rig")
    ap.add_argument("--steps", default="fit,rig,ridge,alt,zeromean,notch")
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--max-iters", type=int, default=1500)
    ap.add_argument("--tol-cell", type=float, default=1e-5)
    ap.add_argument("--rig-wall-s", type=float, default=1800.0)
    ap.add_argument("--latent-iters", type=int, default=300)
    ap.add_argument("--latent-passes", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--rtol-trace-iters", type=int, default=60)
    ap.add_argument("--notch-index", type=int, default=11)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mics", default=None, help="comma list (smoke)")
    ap.add_argument("--k-cap", type=int, default=K_CAP)
    args = ap.parse_args()
    steps = [s for s in args.steps.split(",") if s]

    fit = json.loads(Path(args.fit).read_text())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{fit['support']}.json"
    res: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    res.update(fit=args.fit, set=args.set, device=args.device, fit_git=fit.get("git"))

    def save() -> None:
        path.write_text(json.dumps(res, indent=1))

    log(f"building {fit['support']} on {args.device}")
    t0 = time.time()
    ctx = build(
        fit,
        args.set,
        wind=args.wind,
        device=args.device,
        mics=[int(m) for m in args.mics.split(",")] if args.mics else None,
        k_cap=args.k_cap,
    )
    n_cells = int(ctx["batch"].n_cells)
    res["n_cells"] = n_cells
    res["build_s"] = time.time() - t0
    sites = sites_of(fit, args.device)
    latents = latents_of(fit, args.device)
    base = price(ctx, sites, latents)
    res["fit_state"] = dict(
        price=base,
        recorded=dict(
            whittle_nats=fit["objective"]["whittle_nats"],
            rig_neg_log_prior_nats=fit["objective"]["rig_neg_log_prior_nats"],
            ou_neg_log_prior_nats=fit["objective"]["ou_neg_log_prior_nats"],
            total_nats=fit["objective"]["total_nats"],
        ),
        rig=rig_view(ctx, sites),
    )
    log(
        f"fit rebuilt: whittle {base['whittle_nats']:.2f} (recorded {fit['objective']['whittle_nats']:.2f}), "
        f"rig {base['rig_prior_nats']['total']:.2f}, ou {base['ou_prior_nats']['total']:.2f}"
    )
    save()

    if "rig" in steps:
        log("step 2: production stop rule trace")
        res["production_stop"] = production_stop(ctx, sites, latents, args.rtol_trace_iters)
        save()
        log("step 2: rig to convergence, latents frozen")
        s_rig, rec = rig_converge(
            ctx,
            sites,
            latents,
            chunk=args.chunk,
            max_iters=args.max_iters,
            tol_cell=args.tol_cell,
            wall_s=args.rig_wall_s,
        )
        p_rig = price(ctx, s_rig, latents)
        res["rig_converged"] = dict(
            optimiser=rec, price=p_rig, change=delta(base, p_rig), rig=rig_view(ctx, s_rig)
        )
        save()

    if "ridge" in steps or "alt" in steps:
        log("ridge step from the fit")
        s_r, l_r, rec = ridge(ctx, sites, latents)
        p_r = price(ctx, s_r, l_r)
        res["ridge"] = dict(step=rec, price=p_r, change=delta(base, p_r), rig=rig_view(ctx, s_r))
        save()
        if "alt" in steps:
            rounds = []
            s_a, l_a = s_r, l_r
            for rnd in range(1, args.rounds + 1):
                log(f"alt round {rnd}: latent step")
                l_a, lrec = latent_converge(
                    ctx, s_a, l_a, iters=args.latent_iters, passes=args.latent_passes
                )
                s_a, l_a, rrec = ridge(ctx, s_a, l_a)
                log(f"alt round {rnd}: ridge {rrec['total_prior_change_nats']:.1f} nats; rig step")
                s_a, grec = rig_converge(
                    ctx,
                    s_a,
                    l_a,
                    chunk=args.chunk,
                    max_iters=args.max_iters,
                    tol_cell=args.tol_cell,
                    wall_s=args.rig_wall_s,
                )
                p_a = price(ctx, s_a, l_a)
                rounds.append(
                    dict(
                        round=rnd,
                        latent=lrec,
                        ridge=rrec,
                        rig=dict((k, grec[k]) for k in ("iters", "stop", "wall_s", "gain_nats")),
                        price=p_a,
                        change=delta(base, p_a),
                    )
                )
                res["alt"] = dict(rounds=rounds, rig=rig_view(ctx, s_a))
                save()
                log(
                    f"alt round {rnd}: total {p_a['total_nats']:.2f} (fit {base['total_nats']:.2f})"
                )

    if "zeromean" in steps:
        log("step 2 reverse: zero-mean projection, rig fixed")
        l_z = remove_static(latents, ("d", "v", "u", "uj"))
        p_z = price(ctx, sites, l_z)
        l_z2, lrec = latent_converge(
            ctx, sites, l_z, iters=args.latent_iters, passes=args.latent_passes
        )
        p_z2 = price(ctx, sites, l_z2)
        res["zeromean"] = dict(
            projected=dict(price=p_z, change=delta(base, p_z)),
            relatent=dict(latent=lrec, price=p_z2, change=delta(base, p_z2)),
        )
        save()

    if "notch" in steps:
        log("step 3: the static u_j removed")
        out: dict[str, Any] = {}
        for tag, idx in (("all_points", None), (f"point_{args.notch_index}", args.notch_index)):
            l_n = remove_static(latents, ("uj",), index=idx)
            p_n = price(ctx, sites, l_n)
            out[tag] = dict(removed=dict(price=p_n, change=delta(base, p_n)))
            save()
            if idx is not None:
                log("step 3: rig refit with the notch removed")
                s_n, grec = rig_converge(
                    ctx,
                    sites,
                    l_n,
                    chunk=args.chunk,
                    max_iters=args.max_iters,
                    tol_cell=args.tol_cell,
                    wall_s=args.rig_wall_s,
                )
                p_n2 = price(ctx, s_n, l_n)
                out[tag]["rig_refit"] = dict(
                    optimiser=dict((k, grec[k]) for k in ("iters", "stop", "wall_s", "gain_nats")),
                    price=p_n2,
                    change=delta(base, p_n2),
                    rig=rig_view(ctx, s_n),
                )
                log("step 3: latent step from the removed state")
                l_n3, lrec = latent_converge(
                    ctx, sites, l_n, iters=args.latent_iters, passes=args.latent_passes
                )
                p_n3 = price(ctx, sites, l_n3)
                out[tag]["relatent"] = dict(latent=lrec, price=p_n3, change=delta(base, p_n3))
            res["notch"] = out
            save()
    res["wall_s"] = time.time() - t0
    save()
    log(f"done in {time.time() - t0:.0f} s -> {path}")


if __name__ == "__main__":
    main()
