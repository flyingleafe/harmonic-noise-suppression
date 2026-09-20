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
    # the same with a per-order gain for the orders the score windows resolve
    python scripts/noise_v2_fit.py flight --set dregon-floor --name dregon_room2_floor \
        --floor-low-k --low-orders 8 --frozen-mean results/noise_v2/rounds/round3/fits/bench_dregon_Motor*_70__bench.json
    # the round-1 fit findings table over everything already written
    python scripts/noise_v2_fit.py findings

Every fit lands at ``<out>/<support>__<mode>.json`` in the ``noise-v2-fit/2``
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
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.gridrun import Unit, add_gridrun_args, gridrun_from_args, unit_path  # noqa: E402

OUT_DIR = "results/noise_v2/rounds/round3/fits"
#: profile width. 130 is C4's ``k_cap``; the forward model additionally caps
#: every support at the highest order below its own Nyquist.
K_CAP = 130
#: The orders every summary quotes ``gamma_rk`` at. Low orders are where the
#: shaft-absorption check lives, high ones are where a free width is supposed
#: to earn its place.
GAMMA_LADDER = (1, 2, 4, 8, 16, 32)
#: The fit schemas the reductions read: this round's and R1/R2's.
SCHEMAS = ("noise-v2-fit/2", "noise-v2-fit/1")


def gamma_ladder(gamma_hz: Any) -> dict[str, Any]:
    """``gamma_rk`` at :data:`GAMMA_LADDER`, per rotor, plus its log-mean."""
    import numpy as np

    g = np.atleast_2d(np.asarray(gamma_hz, dtype=np.float64))
    ks = [k for k in GAMMA_LADDER if k <= int(g.shape[1])]
    return dict(
        k=ks,
        value=[g[:, k - 1].tolist() for k in ks],
        log_mean=float(np.exp(np.mean(np.log(np.maximum(g, 1e-12))))),
    )


def load_profile_init(p: dict[str, Any]) -> Any:
    """The unit's ``--profile-init`` file, with ``--profile-prior-sigma`` applied.

    The ladder is ``k:sd`` pairs read left to right: ``16:1,48:3`` gives 1 dB
    up to k = 16, 3 dB from 17 to 48 and NaN (the model's own prior) above. It
    only ever REPLACES the file's widths, and only where the file supplies a
    centre — an order the estimator never measured must not acquire a prior.
    """
    import numpy as np

    from experiments.noise_model import fit as FT

    path = p.get("profile_init")
    if not path:
        return None
    init = FT.load_profile_init(path)
    spec = p.get("profile_prior_sigma")
    if not spec:
        return init
    db = np.atleast_2d(np.asarray(init.profile_db, dtype=np.float64))
    k = np.arange(1, db.shape[1] + 1)
    sigma = np.full(db.shape[1], np.nan)
    lo = 1
    for tier in str(spec).split(","):
        hi_s, _, sd_s = tier.partition(":")
        hi = int(hi_s)
        sigma[(k >= lo) & (k <= hi)] = float(sd_s)
        lo = hi + 1
    import dataclasses

    return dataclasses.replace(
        init, sigma_db=np.where(np.isfinite(db), np.broadcast_to(sigma[None, :], db.shape), np.nan)
    )


# ── one unit ────────────────────────────────────────────────────────────────


def worker(unit: Unit) -> dict[str, Any]:
    """Fit ONE support (bench) or ONE pool (flight) and write its fit JSON."""
    import numpy as np
    import torch

    from experiments.noise_model import fit as FT
    from experiments.noise_model import model as MD
    from experiments.noise_model import supports as SU

    p = dict(unit.params)
    # one unit per core by default: the bench grid runs `--jobs` units at once
    # and a torch thread pool per unit would oversubscribe the node. A single
    # pooled FLIGHT fit is one unit, so there `--threads` is the whole node.
    torch.set_num_threads(max(1, int(p.get("threads", 1))))
    mode = str(p["mode"])
    out_dir = Path(p["out_dir"])
    optim = FT.OptimSpec(**p["optim"])
    frozen = p.get("frozen")
    prof_init = load_profile_init(p)

    if mode == "bench":
        support = SU.load_support(str(p["spec"]))
        carrier = np.asarray(support.carrier_rev_s, dtype=np.float64).mean(axis=1)
        if p.get("apply_carrier_offset"):
            if prof_init is None or prof_init.carrier_offset_rev_s is None:
                raise SystemExit("--apply-carrier-offset needs a --profile-init with offsets")
            carrier = carrier + prof_init.carrier_offset_rev_s
        batch = MD.bench_batch(
            name=support.name,
            power=np.asarray(support.power, dtype=np.float64),
            sr=int(support.sr),
            carrier_mean=carrier,
            # the support's own sample count: recovering it as 2 (F - 1) is
            # n - 1 for an odd segment and stretches the model's bin grid
            n_samples=int(support.n_fft),
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
        batch,
        mode=mode,
        frozen=frozen,
        pin=MD.dynamics_pin(p.get("pin")),
        low_orders=p.get("low_orders"),
        optim=optim,
        profile_init=prof_init,
        progress=int(p.get("progress", 0)),
    )
    # a RESTART lands beside its siblings and is reduced to one reported fit
    # afterwards; a single-seed fit is the reported fit itself
    tag = p.get("restart_tag")
    if tag:
        path = out_dir / "restarts" / f"{name}__{mode}__{tag}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
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
        seed=int(optim.seed),
        sigma_nu=d["sigma_nu"],
        lam=d["lam"],
        gamma_hz=gamma_ladder(np.asarray(d["gamma_hz"], dtype=np.float64)),
        gamma_low_order_check=outcome.diagnostics.get("gamma_low_order_check"),
        span_pins=outcome.diagnostics.get("span_pins"),
        carrier_rev_s=d["carrier_rev_s"],
        comb_gain_db=outcome.comb_gain_db,
        low_order_gain_db=outcome.low_order_gain_db,
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
    the per-order profile are averaged in their own natural scale — rates,
    scales and the per-line WIDTHS in log, dB levels in dB — over the fits
    that carry them.

    "The fits that carry them" is literal for the profile and the widths:
    every support caps its orders at its OWN Nyquist, so the four Motor*_70
    fits are 116 to 118 orders wide and stacking them ragged. Order ``k`` is
    therefore the mean over exactly the fits that reach ``k``, and the frozen
    comb is as wide as the widest of them.
    """
    import numpy as np

    from experiments.noise_model import model as MD

    fits = [json.loads(Path(p).read_text()) for p in paths]
    if not fits:
        raise ValueError("--frozen-mean needs at least one bench fit JSON")
    par = [f["params"] for f in fits]
    log_mean = lambda key: float(np.exp(np.mean([np.log(float(q[key])) for q in par])))  # noqa: E731
    widest = max(
        int(np.asarray(q["profile"]["profile_db"], dtype=np.float64).shape[1]) for q in par
    )
    prof = np.full((len(par), widest), np.nan, dtype=np.float64)
    gam = np.full((len(par), widest), np.nan, dtype=np.float64)
    for i, q in enumerate(par):
        row = np.asarray(q["profile"]["profile_db"], dtype=np.float64)[0]
        prof[i, : row.size] = row
        # a /1 fit's per-order OU is mapped onto its equivalent width first,
        # so a frozen comb may mix rounds without mixing laws
        grow = MD.gamma_from_params(q)[0]
        gam[i, : grow.size] = np.maximum(grow, 1e-12)
    frozen = dict(
        sigma_nu=log_mean("sigma_nu"),
        lam=log_mean("lam"),
        gamma_hz=np.exp(np.nanmean(np.log(gam), axis=0))[None, :].tolist(),
        profile_db=np.nanmean(prof, axis=0)[None, :].tolist(),
        amp_exp=float(np.mean([float(q["profile"]["amp_exp"]) for q in par])),
    )
    return frozen, dict(
        frozen_comb_from=[f["support"] for f in fits],
        frozen_comb_paths=[str(p) for p in paths],
        rule="log-mean of the rates, scales and per-line widths, dB-mean of the per-order "
        "profile, over the fits that reach each order",
        profile_orders=widest,
        profile_orders_per_fit=[
            int(np.asarray(q["profile"]["profile_db"], dtype=np.float64).shape[1]) for q in par
        ],
    )


def expand_frozen(frozen: dict[str, Any], *, n_rotors: int) -> dict[str, Any]:
    """Repeat a one-rotor frozen comb over ``n_rotors`` rotors of one airframe."""
    import numpy as np

    out = dict(frozen)
    prof = np.asarray(frozen["profile_db"], dtype=np.float64)
    if prof.shape[0] != n_rotors:
        out["profile_db"] = np.repeat(prof[:1], n_rotors, axis=0).tolist()
    gam = np.asarray(frozen["gamma_hz"], dtype=np.float64)
    if gam.shape[0] != n_rotors:
        out["gamma_hz"] = np.repeat(gam[:1], n_rotors, axis=0).tolist()
    return out


# ── restarts ────────────────────────────────────────────────────────────────


DYN_KEYS = ("sigma_nu", "lam")


def reduce_restarts(out_dir: Path, *, mode: str = "bench") -> list[dict[str, Any]]:
    """Collapse each support's per-seed restarts into ONE reported fit.

    The reported fit is the best restart — the lowest polished objective — and
    it carries a ``restarts`` block: every restart's objective and dynamics,
    the best-minus-median and best-minus-worst objective PER OBSERVED CELL,
    the min/median/max of each dynamics SCALAR over the restarts and the same
    for ``gamma_rk`` at :data:`GAMMA_LADDER` (a whole ``(R, K)`` block per
    restart does not belong in a summary, but the ladder and the log-mean
    over orders say whether the widths agree). A support whose restarts
    disagree by far more than the convergence tolerance (1e-4 nats/cell) was
    not fitted so much as sampled, and this block is the evidence for it
    rather than a claim about it.
    """
    import numpy as np

    from experiments.noise_model import model as MD

    rdir = out_dir / "restarts"
    groups: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(rdir.glob(f"*__{mode}__s*.json")):
        try:
            f = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if f.get("schema") not in SCHEMAS:
            continue
        f["_path"] = str(path)
        groups.setdefault(str(f["support"]), []).append(f)

    written: list[dict[str, Any]] = []
    for support, items in sorted(groups.items()):
        loss = np.asarray([float(f["optimiser"]["lbfgs_loss_after"]) for f in items])
        best = items[int(np.argmin(loss))]
        cells = max(1, int(best["objective"]["n_cells"]))
        params = {}
        for key in DYN_KEYS:
            v = np.asarray([float(f["params"][key]) for f in items], dtype=np.float64)
            params[key] = dict(
                values=v.tolist(),
                min=float(v.min()),
                median=float(np.median(v)),
                max=float(v.max()),
                max_over_min=float(v.max() / v.min()) if v.min() > 0.0 else None,
            )
        gam = [gamma_ladder(MD.gamma_from_params(f["params"])) for f in items]
        ladder_k = gam[0]["k"]
        params["gamma_hz"] = dict(
            k=ladder_k,
            # rotor-max at each ladder order, one row per restart
            values=[[max(v) for v in g["value"]] for g in gam],
            log_mean=dict(
                values=[g["log_mean"] for g in gam],
                min=float(np.min([g["log_mean"] for g in gam])),
                median=float(np.median([g["log_mean"] for g in gam])),
                max=float(np.max([g["log_mean"] for g in gam])),
                max_over_min=float(
                    np.max([g["log_mean"] for g in gam])
                    / max(np.min([g["log_mean"] for g in gam]), 1e-12)
                ),
            ),
        )
        block = dict(
            n_restarts=len(items),
            seeds=[int(f["optimiser"]["seed"]) for f in items],
            selected_seed=int(best["optimiser"]["seed"]),
            paths=[f["_path"] for f in items],
            loss=loss.tolist(),
            whittle_nats=[float(f["objective"]["whittle_nats"]) for f in items],
            converged=[bool(f["optimiser"]["converged"]) for f in items],
            loss_best=float(loss.min()),
            loss_median=float(np.median(loss)),
            loss_worst=float(loss.max()),
            best_minus_median_per_cell=float((np.median(loss) - loss.min()) / cells),
            best_minus_worst_per_cell=float((loss.max() - loss.min()) / cells),
            params=params,
            rule="reported fit = restart with the lowest polished objective",
        )
        payload = {k: v for k, v in best.items() if k != "_path"}
        payload["restarts"] = block
        path = out_dir / f"{support}__{mode}.json"
        path.write_text(json.dumps(payload, indent=1) + "\n")
        written.append(
            dict(
                support=support,
                path=str(path),
                n_restarts=len(items),
                selected_seed=block["selected_seed"],
                best_minus_median_per_cell=block["best_minus_median_per_cell"],
                best_minus_worst_per_cell=block["best_minus_worst_per_cell"],
            )
        )
    return written


# ── four-motor validation ───────────────────────────────────────────────────


def four_motor_validation(spec: str, quad_path: str, single_paths: list[str]) -> dict[str, Any]:
    """Predict ``motor_allMotors_70`` from the single-motor fits and MEASURE the gap.

    The four-motor static record is the validation support, never a fitting
    one, so the question is not whether two parameter vectors look alike — the
    MAP problem is badly identified and they need not — but how much objective
    a four-rotor forward model built from the FOUR SINGLE-MOTOR fits gives away
    against the four-motor support's own fit on the same support.

    What comes from the single-motor fits is what the airframe is supposed to
    carry from rig to rig: the shared shaft dynamics (log-mean over the four,
    the same rule the floor fit's frozen comb uses) and one per-order profile
    per rotor (rotor ``r`` gets ``Motor{r}``'s own profile, truncated to the
    validation support's order cap). What comes from the four-motor fit is
    everything that is a property of THAT recording and not of a rotor: the
    refined carriers, the mic line gains, the floor and the mic gains. The
    reported discrepancy is therefore the comb's, not the room's.
    """
    import numpy as np
    import torch

    from experiments.noise_model import model as MD
    from experiments.noise_model import supports as SU

    quad = json.loads(Path(quad_path).read_text())
    singles = [json.loads(Path(p).read_text()) for p in single_paths]
    if not singles:
        raise ValueError("--singles needs at least one single-motor bench fit JSON")

    support = SU.load_support(spec)
    batch = MD.bench_batch(
        name=support.name,
        power=np.asarray(support.power, dtype=np.float64),
        sr=int(support.sr),
        carrier_mean=np.asarray(support.carrier_rev_s, dtype=np.float64).mean(axis=1),
        k_cap=K_CAP,
    )
    n_rotors, k_max = batch.n_rotors, batch.k_max

    own = dict(quad["params"])
    pred = dict(own)
    par = [f["params"] for f in singles]
    log_mean = lambda key: float(np.exp(np.mean([np.log(float(q[key])) for q in par])))  # noqa: E731
    pred["sigma_nu"] = log_mean("sigma_nu")
    pred["lam"] = log_mean("lam")
    # the widths travel per ORDER, rotor r taking Motor{r}'s own line widths
    # exactly as it takes Motor{r}'s profile; a /1 single-motor fit is mapped
    gam = np.zeros((n_rotors, k_max), dtype=np.float64)
    own_gam = MD.gamma_from_params(own)
    prof = np.zeros((n_rotors, k_max), dtype=np.float64)
    own_prof = np.asarray(own["profile"]["profile_db"], dtype=np.float64)
    for r in range(n_rotors):
        src = np.asarray(par[r % len(par)]["profile"]["profile_db"], dtype=np.float64)[0]
        take = min(k_max, src.size)
        prof[r, :take] = src[:take]
        if take < k_max:  # the single-motor rig ran below this support's cap
            prof[r, take:] = own_prof[r, take:]
        src_g = MD.gamma_from_params(par[r % len(par)])[0]
        gam[r, : min(k_max, src_g.size)] = src_g[: min(k_max, src_g.size)]
        if src_g.size < k_max:
            gam[r, src_g.size :] = own_gam[r, src_g.size :]
    pred["gamma_hz"] = gam.tolist()
    pred["profile"] = dict(own["profile"])
    pred["profile"]["profile_db"] = prof.tolist()
    pred["profile"]["amp_exp"] = float(np.mean([float(q["profile"]["amp_exp"]) for q in par]))

    def score(d: dict[str, Any]) -> dict[str, Any]:
        params = MD.params_from_dict(d)
        with torch.no_grad():
            m = MD.forward(batch, params)
            obj = MD.objective_breakdown(batch, m)
            power = batch.power
            out: dict[str, Any] = dict(obj)
            for name, band in (("floor", batch.band_lo), ("comb", batch.band_hi)):
                i_sum = float(power[:, :, band].sum())
                m_sum = float(m[:, :, band].sum())
                out[f"level_ratio_db_{name}"] = 10.0 * float(np.log10(i_sum / m_sum))
            out["nats_per_cell"] = obj["whittle_nats"] / max(1, obj["n_cells"])
        return out

    own_score, pred_score = score(own), score(pred)
    keys = ("sigma_nu", "lam")
    params_table = {}
    for key in keys:
        v = np.asarray([float(q[key]) for q in par], dtype=np.float64)
        geo = float(np.exp(np.mean(np.log(v))))
        params_table[key] = dict(
            four_motor=float(own[key]),
            single_geo_mean=geo,
            ratio=float(own[key]) / geo,
            per_rotor=v.tolist(),
            per_rotor_spread=float(v.max() / v.min()),
        )
    gap = pred_score["nats_per_cell"] - own_score["nats_per_cell"]
    return dict(
        schema="noise-v2-four-motor/1",
        support=support.name,
        spec=spec,
        n_rotors=n_rotors,
        k_max=k_max,
        quad_fit=str(quad_path),
        single_fits=[str(p) for p in single_paths],
        single_supports=[f["support"] for f in singles],
        rule=(
            "dynamics = log-mean over the single-motor fits, rotor r's profile = Motor{r}'s "
            "own profile truncated to this support's order cap; carriers, mic line gains, "
            "floor and mic gains from the four-motor fit itself"
        ),
        own_fit=own_score,
        prediction=pred_score,
        params=params_table,
        tolerance=dict(
            nats_per_cell=float(gap),
            nats_total=float(pred_score["whittle_nats"] - own_score["whittle_nats"]),
            level_ratio_db_comb=float(pred_score["level_ratio_db_comb"]),
            level_ratio_db_floor=float(pred_score["level_ratio_db_floor"]),
            own_level_ratio_db_comb=float(own_score["level_ratio_db_comb"]),
            own_level_ratio_db_floor=float(own_score["level_ratio_db_floor"]),
            note=(
                "R1 four-motor tolerance, FROZEN at the measurement: the Whittle cost per "
                "observed cell a four-rotor forward model built from the single-motor fits "
                "gives away against the four-motor support's own fit, plus the band level "
                "ratios of both. A later round predicting this support must not exceed it."
            ),
        ),
    )


# ── findings ────────────────────────────────────────────────────────────────


def findings(out_dir: Path) -> str:
    """The per-support parameter table and the population summary."""
    import numpy as np

    from experiments.noise_model import model as MD

    rows = []
    for path in sorted(out_dir.glob("*.json")):
        try:
            f = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if f.get("schema") not in SCHEMAS:
            continue
        p, o, q = f["params"], f["objective"], f["optimiser"]
        gam = MD.gamma_from_params(p)
        check = f["diagnostics"].get("gamma_low_order_check") or {}
        pins = f["diagnostics"].get("span_pins") or {}
        rows.append(
            dict(
                support=f["support"],
                mode=f["mode"],
                n_rotors=f["n_rotors"],
                k_max=f["k_max"],
                sigma_nu=p["sigma_nu"],
                lam=p["lam"],
                gamma=gam,
                gamma_log_mean=float(np.exp(np.mean(np.log(np.maximum(gam, 1e-12))))),
                gamma_check=check.get("verdict", "—"),
                gamma_k4_over_k1=check.get("k4_over_k1"),
                gamma_over_res=check.get("max_over_resolution"),
                span_pinned=list(pins.get("pinned") or []),
                speed_span=pins.get("speed_span"),
                carrier=p["carrier_rev_s"],
                floor_level_db=f["diagnostics"].get("floor_level_db"),
                whittle=o["whittle_nats"],
                comb=o["per_band"]["comb"],
                floor=o["per_band"]["floor"],
                n_cells=o["n_cells"],
                converged=q["converged"],
                wall_s=q["wall_s"],
                restarts=f.get("restarts"),
                path=str(path),
            )
        )
    if not rows:
        return "# Noise model v2 — round 3 fits\n\nNo fit JSON found.\n"

    def stats(key: str, subset: list[dict[str, Any]]) -> str:
        v = np.asarray([r[key] for r in subset], dtype=np.float64)
        v = v[np.isfinite(v)]
        if not v.size:
            return "—"
        return f"{np.median(v):.4g} [{np.quantile(v, 0.25):.4g}, {np.quantile(v, 0.75):.4g}]"

    def gam_at(r: dict[str, Any], k: int) -> str:
        g = np.asarray(r["gamma"], dtype=np.float64)
        return f"{float(np.max(g[:, k - 1])):.4g}" if g.shape[1] >= k else "—"

    lines = [
        "# Noise model v2 — round 3 fits (Model R3)",
        "",
        f"{len(rows)} fit JSON(s) under `{out_dir}`. Every number below is read from a "
        "`noise-v2-fit/2` payload in that directory (a `/1` payload's per-order OU is mapped "
        "onto its equivalent Lorentzian width first); nothing is recomputed here. The "
        "`gamma` columns are the rotor-max width in Hz at that order.",
        "",
        "## Per-support parameters",
        "",
        "| support | mode | R | k_max | sigma_nu | lam | "
        + " | ".join(f"g(k={k})" for k in GAMMA_LADDER)
        + " | low-k check | pinned | carrier rev/s | floor dB | whittle nats | comb | floor |"
        " cells | conv | wall s |",
        "|---|---|--:|--:|--:|--:|" + "--:|" * len(GAMMA_LADDER) + ":-:|---|---|--:|--:|--:|--:|"
        "--:|:-:|--:|",
    ]
    for r in rows:
        car = ", ".join(f"{c:.3f}" for c in r["carrier"]) if r["carrier"] else "label"
        lines.append(
            f"| `{r['support']}` | {r['mode']} | {r['n_rotors']} | {r['k_max']} |"
            f" {r['sigma_nu']:.4f} | {r['lam']:.3f} | "
            + " | ".join(gam_at(r, k) for k in GAMMA_LADDER)
            + f" | {r['gamma_check']} | {','.join(r['span_pinned']) or '—'} | {car} |"
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
        "| set | n | sigma_nu | lam | gamma log-mean over orders (Hz) |",
        "|---|--:|---|---|---|",
    ]
    for gname, subset in groups.items():
        if not subset:
            continue
        lines.append(
            f"| {gname} | {len(subset)} | {stats('sigma_nu', subset)} | {stats('lam', subset)} |"
            f" {stats('gamma_log_mean', subset)} |"
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
            "ratio of the four-rotor value to the geometric mean of the per-rotor ones and "
            "the per-rotor spread, as read off the fits. This is a raw parameter comparison "
            "only — what the four-motor support does or does not validate is measured in "
            "its own workstream, not here.",
            "",
            "| parameter | four-motor | per-rotor geo-mean | ratio | per-rotor min/max |",
            "|---|--:|--:|--:|---|",
        ]
        for key in ("sigma_nu", "lam", "gamma_log_mean"):
            v = np.asarray([r[key] for r in singles], dtype=np.float64)
            geo = float(np.exp(np.mean(np.log(v))))
            lines.append(
                f"| `{key}` | {q[key]:.4g} | {geo:.4g} | {q[key] / geo:.3f} |"
                f" {v.min():.4g} / {v.max():.4g} |"
            )

    multi = [r for r in rows if r.get("restarts")]
    if multi:
        tol = 1e-4
        lines += [
            "",
            "## Multi-start restarts",
            "",
            "Each support was fitted from several starts: start 0 from the data-driven "
            "initialisation, the others from a log-normal perturbation of the dynamics "
            "init (`OptimSpec.init_jitter`; a bare seed change is a no-op because a bench "
            "fit is deterministic). The REPORTED fit above is the start with the lowest "
            "polished objective. `best-median` and `best-worst` are that objective's "
            "advantage over the median and the worst start, per observed cell, against "
            f"the same {tol:g} nats/cell tolerance the convergence test uses; "
            "`starts agree` is yes only when even the worst start is inside it. The "
            "dynamics columns are min / median / max over the starts.",
            "",
            "| support | starts | best nats/cell | best-median | best-worst | starts agree |"
            " L-BFGS conv | sigma_nu | lam | gamma log-mean |",
            "|---|--:|--:|--:|--:|:-:|:-:|---|---|---|",
        ]
        for r in sorted(multi, key=lambda x: x["support"]):
            b = r["restarts"]
            cells = max(1, int(r["n_cells"]))
            cols = []
            for key in DYN_KEYS:
                s = b["params"][key]
                cols.append(f"{s['min']:.3g} / {s['median']:.3g} / {s['max']:.3g}")
            g = (b["params"].get("gamma_hz") or {}).get("log_mean")
            cols.append(f"{g['min']:.3g} / {g['median']:.3g} / {g['max']:.3g}" if g else "—")
            agree = b["best_minus_worst_per_cell"] < tol
            lines.append(
                f"| `{r['support']}` | {b['n_restarts']} |"
                f" {r['whittle'] / cells:.4f} | {b['best_minus_median_per_cell']:.3g} |"
                f" {b['best_minus_worst_per_cell']:.3g} | {'y' if agree else 'N'} |"
                f" {'y' if r['converged'] else 'N'} | " + " | ".join(cols) + " |"
            )
        agree_n = sum(1 for r in multi if r["restarts"]["best_minus_worst_per_cell"] < tol)
        worst = max(multi, key=lambda r: r["restarts"]["best_minus_worst_per_cell"])
        spreads = np.asarray(
            [
                r["restarts"]["params"]["lam"]["max_over_min"] or np.nan
                for r in multi
                if r["restarts"]["params"]["lam"]["max_over_min"]
            ],
            dtype=np.float64,
        )
        lines += [
            "",
            f"{agree_n} of {len(multi)} supports have every start inside the tolerance. The "
            f"widest disagreement is `{worst['support']}` at "
            f"{worst['restarts']['best_minus_worst_per_cell']:.3g} nats/cell, "
            f"{worst['restarts']['best_minus_worst_per_cell'] / tol:.0f} times the tolerance. "
            f"`lam` alone spans a factor of {np.nanmedian(spreads):.3g} (median over supports) "
            f"and up to {np.nanmax(spreads):.3g} across the starts of one support: on this "
            "evidence the R1 bench MAP problem is multi-modal, and a number computed from a "
            "single start is a draw from that multiplicity rather than an estimate.",
        ]

    # the low-order check that decides whether the shaft was absorbed
    lines += [
        "",
        "## The low-order `gamma_rk` check",
        "",
        "R3's one possible degeneracy: in the Brownian limit the shaft term is Lorentzian "
        "too, so a `gamma_rk` ramping as `k^2` would absorb it. PASS is every fitted width "
        "at `k <= 4` sitting within a factor of 3 of the window's resolution floor "
        "`1 / (2 T)`; `fail_k2_ramp` is `gamma_4 / gamma_1 >= 8`, and then `lam` takes the "
        "long-lag pin instead of the bench value.",
        "",
        "| support | verdict | max gamma(k<=4) / resolution | gamma_4 / gamma_1 |",
        "|---|:-:|--:|--:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['support']}` | {r['gamma_check']} |"
            f" {format(r['gamma_over_res'], '.3g') if r['gamma_over_res'] is not None else '—'} |"
            f" {format(r['gamma_k4_over_k1'], '.3g') if r['gamma_k4_over_k1'] is not None else '—'} |"
        )
    failed = [r["support"] for r in rows if r["gamma_check"] not in ("pass", "—")]
    lines += [
        "",
        (
            f"{len(rows) - len(failed)} of {len(rows)} fits pass. Failing: "
            + ", ".join(f"`{s}`" for s in failed)
            if failed
            else "Every fit passes the check."
        ),
    ]
    edges = []
    # the R3 priors' own central 95 %, not a remembered number
    lam_lo, lam_hi = (
        math.exp(MD.PRIORS.log_lam[0] - 2.0 * MD.PRIORS.log_lam[1]),
        math.exp(MD.PRIORS.log_lam[0] + 2.0 * MD.PRIORS.log_lam[1]),
    )
    for r in rows:
        flight = str(r["mode"]).startswith("flight")
        s_lo, s_hi = (
            math.exp(
                MD.PRIORS.sigma_nu_prior(flight)[0] - 2.0 * MD.PRIORS.sigma_nu_prior(flight)[1]
            ),
            math.exp(
                MD.PRIORS.sigma_nu_prior(flight)[0] + 2.0 * MD.PRIORS.sigma_nu_prior(flight)[1]
            ),
        )
        if not flight and not (lam_lo <= r["lam"] <= lam_hi):
            edges.append(
                f"`{r['support']}`: lam = {r['lam']:.2f} outside the prior's central 95 % "
                f"[{lam_lo:.3g}, {lam_hi:.3g}]"
            )
        if not (s_lo <= r["sigma_nu"] <= s_hi):
            edges.append(
                f"`{r['support']}`: sigma_nu = {r['sigma_nu']:.4f} outside the prior's "
                f"central 95 % [{s_lo:.3g}, {s_hi:.3g}]"
            )
        if r["gamma_check"] not in ("pass", "—"):
            edges.append(f"`{r['support']}`: low-order gamma check {r['gamma_check']}")
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


def _pin_from_args(args: argparse.Namespace) -> dict[str, float] | None:
    """``["lam=0.5"]`` -> ``{"lam": 0.5}``."""
    items = getattr(args, "pin", None)
    if not items:
        return None
    out: dict[str, float] = {}
    for item in items:
        key, sep, value = str(item).partition("=")
        if not sep:
            raise SystemExit(f"--pin wants NAME=VALUE, got {item!r}")
        out[key.strip()] = float(value)
    return out


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
        p.add_argument(
            "--seeds",
            type=int,
            default=1,
            help="RESTARTS per support (bench) or per pool (flight): seeds <seed> .. "
            "<seed>+N-1, reduced to the best fit plus a restart-spread block",
        )
        p.add_argument(
            "--init-jitter",
            type=float,
            default=0.8,
            help="log-space sd of the multi-start perturbation of the dynamics init, "
            "applied to every restart but the first",
        )
        p.add_argument("--progress", type=int, default=0, help="print the Adam loss every N steps")
        p.add_argument(
            "--pin",
            nargs="*",
            default=None,
            metavar="NAME=VALUE",
            help="hold one dynamics coordinate FIXED while the rest of the block is fitted, "
            "e.g. --pin lam=0.5. Names: sigma_nu, lam (R3's two dynamics scalars; the "
            "per-line gamma_hz block is frozen wholesale, never pinned coordinate-wise). "
            "Used when a rate is not identifiable from the data and the "
            "identified ridge coordinate is what the fit should move along",
        )
        p.add_argument(
            "--profile-init",
            default=None,
            metavar="NPZ",
            help="an .npz (or .json) carrying profile_db (R, K) in the model's units and, "
            "optionally, sigma_db (R, K) and carrier_offset_rev_s (R,). The profile is "
            "INITIALISED there wherever the centre is finite, and the Gaussian profile prior "
            "is re-centred and re-scaled wherever the width is finite too; NaN leaves the "
            "model's own initialisation and two-regime prior untouched. Written by "
            "scripts/noise_v2_fourmotor.py from the multi-rotor estimator",
        )
        p.add_argument(
            "--profile-prior-sigma",
            default=None,
            metavar="K:SD,...",
            help="override the file's per-order prior widths with a piecewise-constant "
            "ladder, e.g. '16:1,48:3' = 1 dB for k <= 16, 3 dB for 17 <= k <= 48 and the "
            "model's own prior above. Only orders the file gives a centre for are affected",
        )
        p.add_argument(
            "--apply-carrier-offset",
            action="store_true",
            help="add the --profile-init file's carrier_offset_rev_s to the support's frozen "
            "bench carriers (bench mode only)",
        )
        p.add_argument(
            "--threads",
            type=int,
            default=1,
            help="torch threads per unit (a one-unit flight fit wants the whole node)",
        )
        add_gridrun_args(p, jobs=4)
        if name == "flight":
            p.add_argument(
                "--name", required=True, help="pooled fit name, e.g. michaels_fly125_cruise"
            )
            p.add_argument(
                "--floor-only",
                action="store_true",
                help="freeze the comb's shape and dynamics from --frozen-mean and fit the "
                "floor, the mic gains and ONE shared comb level (comb_gain_db)",
            )
            p.add_argument(
                "--floor-low-k",
                action="store_true",
                help="mode flight_floor_lowk: --floor-only PLUS one gain per ORDER below "
                "--low-orders, shared across rotors. The transplanted comb keeps its bench "
                "shape above the cut; below it the flight recording sets the level, which is "
                "what a score window with no resolvable comb above k ~ 8 needs",
            )
            p.add_argument(
                "--low-orders",
                type=int,
                default=8,
                help="width of the per-order low-order gain block of --floor-low-k",
            )
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
            p.add_argument(
                "--restart-tag",
                default=None,
                help="write this fit as a RESTART under <out>/restarts/<name>__<mode>__<tag>."
                "json even though it is a single seed. A pooled flight fit can need the whole "
                "node, so its restarts run as one CLUSTER JOB EACH; `reduce` then collapses "
                "them the way --seeds would have",
            )

    f = sub.add_parser("findings")
    f.add_argument("--out", default=OUT_DIR)

    r = sub.add_parser(
        "reduce", help="collapse <out>/restarts/*__<mode>__s*.json into one reported fit"
    )
    r.add_argument("--out", default=OUT_DIR)
    r.add_argument("--mode", default="bench")

    v = sub.add_parser("validate", help="four-motor validation of the single-motor fits")
    v.add_argument("--out", default=OUT_DIR)
    v.add_argument("--support", default="bench_dregon_motor:allMotors:70")
    v.add_argument("--quad", default=f"{OUT_DIR}/bench_dregon_allMotors_70__bench.json")
    v.add_argument("--singles", nargs="+", required=True, help="single-motor bench fit JSONs")

    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cmd == "findings":
        text = findings(out_dir)
        (out_dir / "findings.md").write_text(text)
        print(text)
        return 0

    if args.cmd == "reduce":
        rows = reduce_restarts(out_dir, mode=str(args.mode))
        if not rows:
            raise SystemExit(f"{out_dir / 'restarts'}: no *__{args.mode}__s*.json to reduce")
        for row in rows:
            print(json.dumps(row))
        return 0

    if args.cmd == "validate":
        payload = four_motor_validation(str(args.support), str(args.quad), list(args.singles))
        path = out_dir / "four_motor_validation.json"
        path.write_text(json.dumps(payload, indent=1) + "\n")
        print(json.dumps(payload["tolerance"], indent=1))
        print(f"# wrote {path}", flush=True)
        return 0

    specs = _specs(args)
    grid_dir = Path(args.grid_dir) if args.grid_dir else out_dir / "grid"

    seeds = [int(args.seed) + i for i in range(max(1, int(getattr(args, "seeds", 1))))]
    if args.cmd == "bench":
        units = []
        for spec in specs:
            for s in seeds:
                optim = _optim_from_args(args)
                optim["seed"] = s
                optim["init_jitter"] = 0.0 if s == seeds[0] else float(args.init_jitter)
                tag = f"s{s}" if len(seeds) > 1 else None
                units.append(
                    Unit(
                        uid=spec.replace(":", "_") + (f"__{tag}" if tag else ""),
                        params=dict(
                            mode="bench",
                            spec=spec,
                            out_dir=str(out_dir),
                            optim=optim,
                            progress=int(args.progress),
                            threads=int(args.threads),
                            pin=_pin_from_args(args),
                            profile_init=args.profile_init,
                            profile_prior_sigma=args.profile_prior_sigma,
                            apply_carrier_offset=bool(args.apply_carrier_offset),
                            restart_tag=tag,
                        ),
                    )
                )
    else:
        frozen = None
        frozen_from = None
        if args.frozen_mean:
            from experiments.noise_model import supports as SU

            frozen, frozen_from = mean_comb(list(args.frozen_mean))
            n_rotors = int(SU.load_support(specs[0]).carrier_rev_s.shape[0])
            frozen = expand_frozen(frozen, n_rotors=n_rotors)
        low_k = bool(getattr(args, "floor_low_k", False))
        floor_only = bool(args.floor_only) or low_k
        mode = ("flight_floor_lowk" if low_k else "flight_floor_only") if floor_only else "flight"
        if floor_only and frozen is None:
            raise SystemExit("--floor-only / --floor-low-k needs --frozen-mean <bench fit JSONs>")
        units = []
        for s in seeds:
            optim = _optim_from_args(args)
            optim["seed"] = s
            optim["init_jitter"] = 0.0 if s == seeds[0] else float(args.init_jitter)
            tag = f"s{s}" if len(seeds) > 1 else (args.restart_tag or None)
            units.append(
                Unit(
                    uid=f"{args.name}__{mode}" + (f"__{tag}" if tag else ""),
                    params=dict(
                        mode=mode,
                        specs=specs,
                        name=str(args.name),
                        out_dir=str(out_dir),
                        optim=optim,
                        frozen=frozen,
                        frozen_from=frozen_from,
                        frame_stride=int(args.frame_stride),
                        max_frames=None if int(args.max_frames) <= 0 else int(args.max_frames),
                        progress=int(args.progress),
                        threads=int(args.threads),
                        pin=_pin_from_args(args),
                        profile_init=args.profile_init,
                        profile_prior_sigma=args.profile_prior_sigma,
                        low_orders=int(args.low_orders) if low_k else None,
                        restart_tag=tag,
                    ),
                )
            )

    def summarize(payloads: list[dict[str, Any]]) -> dict[str, Any]:
        ok = [r for r in payloads if r.get("converged")]
        return dict(
            n_units=len(payloads),
            n_converged=len(ok),
            not_converged=[r["support"] for r in payloads if not r.get("converged")],
            slowest_s=max((r.get("wall_s", 0.0) for r in payloads), default=0.0),
            fits=[r.get("fit_json") for r in payloads],
        )

    # the BLAS pools get the same width as the torch pool, or a `--threads 8`
    # flight fit would still do its linear algebra on one core
    result = gridrun_from_args(
        args, units, worker, grid_dir, summarize=summarize, blas_threads=int(args.threads)
    )
    if len(seeds) > 1:
        for row in reduce_restarts(out_dir, mode=("bench" if args.cmd == "bench" else mode)):
            print(f"# reduced {row['support']}: {json.dumps(row)}", flush=True)
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
