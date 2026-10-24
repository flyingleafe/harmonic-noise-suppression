#!/usr/bin/env python
"""Noise model v4, procedure B: fit MD2 (Michael's drone) in free flight; the
DREGON flight stage (wind, floor and per-rotor line gains from flight on a frozen bench rig);
and the DREGON wind stage (procedure C-wind: the SC gust wind per capsule on a flight payload).

Spec: ``docs/explainers/noise-model-v4.qmd`` § Procedure B; module
:mod:`experiments.noise_model.flight_v4`. One restart per call; the restarts
are reduced afterwards to the lowest objective.

    # one restart (seed 0 = prior medians; seeds 1-3 draw sigma_nu from its prior)
    python scripts/noise_v4_flight.py fit --seed 0 --device cuda --out results/noise_v4/flight
    # the four restarts reduced to one payload
    python scripts/noise_v4_flight.py reduce --out results/noise_v4/flight
    # one regime with the phase pedestal fitted per order (held at the prior medians otherwise)
    python scripts/noise_v4_flight.py fit --sets michaels-cruise --name md2_fly125_cruise \\
        --free-phase-pedestal --device cuda --out results/noise_v4/flight
    # a CPU smoke: one standby + one cruise window, mics 0,1, 24 orders
    python scripts/noise_v4_flight.py fit --max-windows 2 --mics 0,1 --k-cap 24 \\
        --adam-steps 20 --lbfgs-iters 10 --polish-iters 4 --out /tmp/nv4_smoke
    # DREGON flight stage on a bench payload (module experiments.noise_model.dregon_flight_v4)
    python scripts/noise_v4_flight.py dregon \\
        --from results/noise_v4/bench/restarts/dregon__kappa0__r5__s2.json --threads 8
    # ... with the flight smear allowed only into the shaft wander eps (GPU)
    python scripts/noise_v4_flight.py dregon --coherence shaft --device cuda --chunk-frames 64 \\
        --from results/noise_v4/bench/restarts/dregon__kappa10__r5__s2__c1.json --progress 1
    # ... with the phase pedestal per order and eps refitted in flight (GPU)
    python scripts/noise_v4_flight.py dregon --coherence full --device cuda --chunk-frames 64 \\
        --from results/noise_v4/bench/restarts/dregon__kappa10__r5__s2__c1.json --progress 1
    # LINE REFIT on the wide pool from a flight payload (its sc wind's shape held):
    # profiles + phase pedestal of k <= 30, eps, mic gains, floor, wind levels; a
    # deadline-bounded GPU job, then continuations (<...>__lines__c1, __c2, ...)
    python scripts/noise_v4_flight.py dregon --lines --pool wide --device cuda \\
        --chunk-frames 64 --deadline-min 55 --progress 1 --from \\
        results/noise_v4/flight/wind/dregon_room2__flight_v4__dregon__kappa0__r5__s2__scwind.json
    python scripts/noise_v4_flight.py dregon --device cuda --chunk-frames 64 --deadline-min 55 \\
        --resume results/noise_v4/flight/dregon_room2__flight_v4__<stem>__wide__lines.json
    # DREGON wind stage on a flight-stage payload: the SC gust wind per capsule
    # (module experiments.noise_model.dregon_wind_v4); a 6-CPU node, single-threaded workers
    OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python scripts/noise_v4_flight.py wind \\
        --from results/noise_v4/flight/dregon_room2__flight_v4__dregon__kappa0__r5__s2.json \\
        --workers 6 --progress 1
    # ... a smoke: one grid point, one seed, one segment
    python scripts/noise_v4_flight.py wind --from <payload> --offsets 2 --scales 1 --rates 1 \\
        --seeds 1 --max-segments 1 --out /tmp/nv4_wind_smoke
    # gust-law search by the time-aware battery (inner refit: per-mic levels + floor mean_db)
    OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python scripts/noise_v4_flight.py wind --targets battery \\
        --from <sc-wind or line-refit payload> --workers 8 --deadline-min 170 --progress 1

The pool is FLY125 standby (``michaels-standby``, 3 x 4 s) and cruise
(``michaels-cruise`` FLY125 rows, 8 x 8 s), labels ``rps_refined``, the nv3r4
front end (frames thinned by 4, 32 kHz work grid, 16-frame chunks) with the
channels divided by the measured array response (``--channel-gains-band
transfer``). Supports come from the ``.npz`` cache or are built from the data
(``scripts/noise_v2_supports.py build`` fills the cache in a remote job).

Writes ``<out>/restarts/<name>__flight_v4__s<seed>.json`` (``noise-v4-fit/3``);
``reduce`` writes ``<out>/<name>__flight_v4.json``; ``dregon`` writes
``<out>/dregon_room2__flight_v4__<source stem>[__wide]<mode>.json`` (a flight
source's own ``dregon_room2__flight_v4__`` prefix dropped): the source payload
with ``kind="flight"``, a per-rotor line gain written into both profile rows, a
per-mic line gain (sum 0) into the pair EQ, the floor refitted and the per-mic
wind levels (static, or the source's ``sc`` gust wind with its law and shape
held), fitted on the five 8 s ``dregon-floor`` segments (``--pool wide``: the
18 windows of ``dregon-flight-wide``) with every source line-shape, pedestal,
shaft and pair-EQ shape parameter frozen (``--no-line-gain`` /
``--no-mic-gain`` hold those gains at 0, ``--max-frames`` for a smoke run).
``--coherence shaft`` also fits the shaft ``eps`` (the bench phase pedestal
held; mode ``__shaft``), ``--coherence full`` the phase pedestal per
aerodynamic order and ``eps`` (``__coh``), ``--lines`` (``--coherence lines``,
``__lines``) the profile entries and the phase pedestal of the orders
``k <= --k-free`` and ``eps`` (line gains held: the profiles carry the level).
``--deadline-min`` stops the fit in time to write the payload; ``--resume
<payload>`` continues it (source and settings read from the payload) as
``<payload stem>__c<n+1>.json``. ``wind`` writes ``<out>/<stem>__scwind.json``: the
``--from`` payload with ``params.wind`` the SC gust wind (kind ``sc``: per-mic
levels by likelihood, the gust law by simulated moments over the
``--offsets`` x ``--scales`` x ``--rates`` grid) and a ``wind_stage`` block, plus
``<stem>__scwind.grid.json`` (the whole grid); its restartable work (one
``utils.gridrun`` unit per base render and grid point) is ``<out>/<stem>__scwind/``.
``wind --targets battery`` writes ``<out>/<stem>__gust.json`` and
``<stem>__gust.grid.json`` instead: the gust law chosen by the seed-floor-normalised
spectro-temporal / level-distribution distances on the tuning segments, the
per-mic levels and the floor ``mean_db`` refitted by Whittle at each law
(``--pool floor|wide``); work in ``<out>/<stem>__gust/``; exit 3 when
``--deadline-min`` cut it (rerun the same command to resume).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

NAME = "md2_fly125"
SETS = ("michaels-standby", "michaels-cruise")
K_CAP = 130
CHANNEL_GAINS = "results/noise_v2/mic_gains/mic_gains.json"
#: ``dregon --coherence`` modes (``dregon_flight_v4.COHERENCE_MODES``) and their output suffixes.
COHERENCE_TAGS = {"none": "", "shaft": "__shaft", "full": "__coh", "lines": "__lines"}
#: ``dregon --pool`` choices (``dregon_flight_v4.POOLS``) and their output suffixes.
POOL_TAGS = {"floor": "", "wide": "__wide"}
#: ``dregon``'s exit status when ``--deadline-min`` stopped the fit (resume it).
EXIT_DEADLINE = 3


def _pool_specs(sets: list[str], max_windows: int | None) -> list[str]:
    """FLY125 windows of ``sets``; ``max_windows`` takes them round-robin
    across the sets, so a two-window smoke still joins standby and cruise."""
    from experiments.noise_model import supports as SU

    per_set = []
    for name in sets:
        specs = [s for s in SU.support_set(name) if s.args.get("recording") == "FLY125"]
        per_set.append([s.text if hasattr(s, "text") else str(s) for s in specs])
    out: list[str] = []
    for i in range(max(len(s) for s in per_set)):
        out.extend(s[i] for s in per_set if i < len(s))
    return out if max_windows is None else out[: int(max_windows)]


def _build(args: argparse.Namespace) -> tuple[Any, Any]:
    import numpy as np
    import torch

    from experiments.noise_model import fit as FT
    from experiments.noise_model import flight_v4 as FV
    from experiments.noise_model import model as MD
    from experiments.noise_model import supports as SU

    torch.set_num_threads(max(1, int(args.threads)))
    mics = [int(v) for v in str(args.mics).split(",")] if args.mics else None
    specs = list(args.support) if args.support else _pool_specs(list(args.sets), args.max_windows)
    members = []
    first = None
    for spec in specs:
        s = SU.load_support(spec)
        first = first or s
        power = np.asarray(s.power, dtype=np.float64)
        members.append(
            (
                s.name,
                power if mics is None else power[np.asarray(mics, dtype=np.int64)],
                np.asarray(s.carrier_rev_s_audio, dtype=np.float64),
                np.asarray(s.frame_starts, dtype=np.int64),
            )
        )
    assert first is not None
    gains = FT.load_channel_gains(
        str(args.channel_gains), rig="michaels", mics=mics, band=str(args.channel_gains_band)
    )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    batch = MD.flight_batch(
        name=str(args.name),
        members=members,
        sr=int(first.sr),
        n_fft=int(first.n_fft),
        hop=int(first.hop),
        k_cap=int(args.k_cap),
        frame_stride=int(args.frame_stride),
        channel_gains=gains,
        device=device,
        chunk_frames=None if int(args.chunk_frames) <= 0 else int(args.chunk_frames),
        sr_work=int(args.work_rate),
    )
    data = FV.prepare(batch, k_cap=int(args.k_cap), mics=mics, wander_path=str(args.wander))
    return data, gains


def cmd_fit(args: argparse.Namespace) -> int:
    from experiments.noise_model import flight_v4 as FV

    t0 = time.time()
    data, gains = _build(args)
    b = data.batch
    print(
        f"pool {b.name}: {len(b.members)} windows, {int(b.power.shape[1])} frames, "
        f"{b.n_mics} mics, K {data.k_max}, motor orders {data.motor_orders.tolist()}, "
        f"device {b.power.device}, build {time.time() - t0:.1f}s",
        flush=True,
    )
    t1 = time.time()
    measured = FV.measure(data)
    print(
        f"measured: mu {measured.floor_mean_db:.2f} dB  sigma_B {measured.floor_shape_sd_db:.2f} dB  "
        f"({time.time() - t1:.1f}s)",
        flush=True,
    )
    if args.time_eval:
        import torch

        model = FV.FlightV4Model(
            data,
            measured,
            kappa=float(args.kappa),
            free_phase_pedestal=bool(args.free_phase_pedestal),
        )
        t2 = time.time()
        loss = float(model.objective(b, grad=True))
        FV._sync(b.power.device)
        dt = time.time() - t2
        grads = sum(float((p.grad**2).sum()) for p in model.params.values() if p.grad is not None)
        print(
            f"ONE objective call (forward + gradient, {int(b.power.shape[1])} frames, "
            f"{b.n_cells} cells): {dt:.2f}s  loss {loss:.6g}  |grad|^2 {grads:.3g}  "
            f"threads {torch.get_num_threads()}",
            flush=True,
        )
        return 0
    optim = FV.OptimV4(
        adam_steps=int(args.adam_steps),
        adam_lr=float(args.adam_lr),
        adam_batch=int(args.adam_batch),
        lbfgs_iters=int(args.lbfgs_iters),
        lbfgs_frames=int(args.lbfgs_frames),
        polish_iters=int(args.polish_iters),
        seed=int(args.seed),
    )
    start = json.loads(Path(args.start).read_text()) if args.start else None
    model, record = FV.fit_flight(
        data,
        measured,
        kappa=float(args.kappa),
        optim=optim,
        progress=int(args.progress),
        start=start,
        deadline_s=60.0 * float(args.deadline_min) if args.deadline_min else None,
        free_phase_pedestal=bool(args.free_phase_pedestal),
    )
    pay = FV.payload(model, record, channel_gains=gains)
    pay["pool"]["specs"] = (
        list(args.support) if args.support else _pool_specs(list(args.sets), args.max_windows)
    )
    uid = f"{args.name}__{FV.FLIGHT_V4_MODE}__s{int(args.seed)}" + (
        f"__{args.tag}" if args.tag else ""
    )
    out = Path(args.out) / "restarts" / f"{uid}.json"
    if start is not None:
        pay["optimiser"]["start_payload"] = str(args.start)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pay, indent=1))
    conv = record["optimiser"]["convergence"]
    print(
        f"wrote {out}: total {record['objective']['total_nats']:.6g} nats "
        f"(start {record['objective']['start_total_nats']:.6g}); converged {conv['converged']} "
        f"({conv['stop']}, {conv['n_restart_passes']} restart passes, gain/cell "
        f"{conv['gain_per_cell']}, moved {conv['movement']}); "
        f"wall {time.time() - t0:.1f}s",
        flush=True,
    )
    return 0


def cmd_reduce(args: argparse.Namespace) -> int:
    from experiments.noise_model import flight_v4 as FV

    paths = sorted((Path(args.out) / "restarts").glob(f"{args.name}__{FV.FLIGHT_V4_MODE}__s*.json"))
    if not paths:
        raise SystemExit(f"no restarts under {Path(args.out) / 'restarts'}")
    payloads = [json.loads(p.read_text()) for p in paths]
    red = FV.reduce_payloads(payloads)
    red["restarts"]["paths"] = [str(p) for p in paths]
    out = Path(args.out) / f"{args.name}__{FV.FLIGHT_V4_MODE}.json"
    out.write_text(json.dumps(red, indent=1))
    r = red["restarts"]
    print(
        f"wrote {out}: seed {r['selected_seed']} of {r['seeds']}; totals {r['total_nats']}; "
        f"converged {r['converged']}",
        flush=True,
    )
    return 0


def _dregon_out(args: argparse.Namespace, source_path: Path) -> Path:
    """``<out>/dregon_room2__flight_v4__<source stem><pool><mode>.json`` (a
    flight source's own prefix dropped), or ``<resume stem>__c<n+1>.json``."""
    import re

    from experiments.noise_model import dregon_flight_v4 as DF

    if args.resume:
        m = re.fullmatch(r"(.*?)(?:__c(\d+))?", Path(args.resume).stem)
        assert m is not None
        return Path(args.out) / f"{m.group(1)}__c{int(m.group(2) or 0) + 1}.json"
    prefix = f"{DF.NAME}__flight_v4__"
    stem = source_path.stem.removeprefix(prefix)
    tags = POOL_TAGS[args.pool] + COHERENCE_TAGS[args.coherence]
    return Path(args.out) / f"{prefix}{stem}{tags}.json"


def cmd_dregon(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    from experiments.noise_model import dregon_flight_v4 as DF

    torch.set_num_threads(max(1, int(args.threads)))
    t0 = time.time()
    resume = json.loads(Path(args.resume).read_text()) if args.resume else None
    if resume is not None:
        # a continuation: the source and every setting are the payload's
        rs = resume["flight_stage"]
        args.source = args.source or rs["source"]
        s = rs["settings"]
        args.pool, args.frames_thin, args.max_frames = s["pool"], s["stride"], s["max_frames"]
        args.coherence, args.kappa = s["coherence"], s["kappa"]
        args.no_line_gain, args.no_mic_gain = not s["line_gain"], not s["mic_gain"]
        args.k_free, args.profile_sd_db = s["k_free"], s["profile_sd_db"]
        args.hold_eps = bool(s.get("hold_eps"))
        args.hold_pedestal = bool(s.get("hold_pedestal"))
        args.wind_from = rs.get("wind_from")
    if args.k_free is None:
        args.k_free = DF.K_FREE
    if args.profile_sd_db is None:
        args.profile_sd_db = DF.PROFILE_FLIGHT_SD_DB
    if not args.source:
        raise SystemExit("dregon needs --from <payload> (or --resume <payload>)")
    source_path = Path(args.source)
    source = json.loads(source_path.read_text())
    if args.wind_from:
        # the gust law and levels of another payload (the battery gust search's)
        source["params"]["wind"] = json.loads(Path(args.wind_from).read_text())["params"]["wind"]
    pay = DF.run_stage(
        source,
        source_path=str(source_path),
        kappa=args.kappa,
        pool=str(args.pool),
        stride=int(args.frames_thin),
        max_frames=args.max_frames,
        wind_start_path=str(args.wind_start),
        chunk_frames=int(args.chunk_frames),
        device=str(args.device),
        iters=int(args.iters),
        restart_iters=int(args.restart_iters),
        max_passes=int(args.max_passes),
        progress=int(args.progress),
        line_gain=False if args.no_line_gain else None,
        mic_gain=not args.no_mic_gain,
        coherence=str(args.coherence),
        k_free=int(args.k_free),
        profile_sd_db=float(args.profile_sd_db),
        hold_eps=bool(args.hold_eps),
        hold_pedestal=bool(args.hold_pedestal),
        deadline_s=None if args.deadline_min is None else 60.0 * float(args.deadline_min),
        resume=resume,
        resume_path=args.resume,
    )
    if args.wind_from:
        pay["flight_stage"]["wind_from"] = str(args.wind_from)
    out = _dregon_out(args, source_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pay, indent=1))
    st, fl = pay["flight_stage"], pay["params"]["floor"]
    obj, conv = st["objective"], st["optimiser"]["convergence"]
    coh = st["coherence"]
    coh_line = (
        f"coherence {coh['mode']}: eps {coh['start']['eps']:.4g} -> {coh['fitted']['eps']:.4g} "
        f"(sigma_ref {coh['fitted']['sigma_ref_rad_s']:.3g} rad/s; source "
        f"{coh['bench']['eps']:.4g}), median sigma2_phase "
        f"{np.median(coh['fitted']['sigma2_phase']):.3g}, median W "
        f"{np.median(coh['fitted']['width_phase_hz']):.3g} Hz"
        if coh["free"]
        else "coherence none: frozen at the bench's"
    )
    if st["lines"] is not None:
        d = np.asarray(st["lines"]["fitted_db"]) - np.asarray(st["lines"]["prior"]["centre_db"])
        coh_line += (
            f"\nprofiles k <= {st['lines']['k_free']} vs centre: median |d| per rotor "
            f"{np.round(np.median(np.abs(d), axis=1), 2).tolist()} dB, max "
            f"{np.round(np.abs(d).max(axis=1), 2).tolist()} dB"
        )
    print(
        f"wrote {out}\n"
        f"pool {pay['support']}: {len(pay['supports'])} windows, "
        f"{st['pool']['n_frames_used']} frames; wind {st['wind']['kind']}\n"
        f"objective {obj['total_nats']:.10g} nats (whittle {obj['whittle_nats']:.10g}, "
        f"-log prior {obj['neg_log_prior_nats']:.6g}, start {obj['start_total_nats']:.10g}, "
        f"{obj['n_cells']} cells)\n"
        f"converged {conv['converged']} ({conv['stop']}, {conv['n_passes']} passes, last gain/cell "
        f"{conv['gain_per_cell']:.3g}, moved {conv['movement']}); objective + gradient "
        f"{st['optimiser']['eval_grad_s']:.2f}s\n"
        f"renderer check: max rel err {st['renderer_check']['max_rel_err']:.3g}\n"
        f"floor mean_db {fl['mean_db']:.3f}  b0 {fl['b0']:.4f}  b1 {fl['b1']:.4f}  "
        f"static_rel {fl['static_rel']:.4g}\n"
        f"wind levels {[round(w, 2) for w in st['wind']['level_db']]}\n"
        f"line_gain_db {[round(g, 3) for g in st['line_gain']['gain_db']]} "
        f"(free {st['line_gain']['free']})\n"
        f"mic_gain_db {[round(g, 3) for g in st['mic_gain']['gain_db']]} "
        f"(free {st['mic_gain']['free']})\n"
        f"{coh_line}\n"
        f"wall {time.time() - t0:.1f}s",
        flush=True,
    )
    return EXIT_DEADLINE if conv["stop"] == "deadline" else 0


def cmd_wind(args: argparse.Namespace) -> int:
    import numpy as np

    from experiments.noise_model import dregon_wind_v4 as DW

    battery = args.targets == "battery"
    defaults = dict(
        offsets=DW.BATTERY_OFFSETS_MS if battery else DW.OFFSETS_MS,
        scales=DW.BATTERY_SCALES_MS if battery else DW.SCALES_MS,
        rates=DW.BATTERY_RATES_HZ if battery else DW.RATES_HZ,
    )
    grid = {k: tuple(getattr(args, k) or v) for k, v in defaults.items()}
    if battery:
        return cmd_gust(args, grid)
    if args.pool != "floor" or args.deadline_min is not None or args.floor_pairs is not None:
        raise SystemExit("--pool, --deadline-min and --floor-pairs are --targets battery options")
    t0 = time.time()
    pay, report, pay_path, report_path = DW.run_wind_stage(
        args.source,
        out=args.out,
        offsets=grid["offsets"],
        scales=grid["scales"],
        rates=grid["rates"],
        seeds=int(args.seeds),
        workers=int(args.workers),
        max_segments=args.max_segments,
        n_mc=int(args.n_mc),
        check_seeds=int(args.check_seeds),
        resume=not args.no_resume,
        progress=int(args.progress),
    )
    st, w = pay["wind_stage"], pay["params"]["wind"]
    ch, chk = st["gust_law"]["chosen"], st["renderer_check"]
    centre = [r["centre_db"] for r in st["amplitude"]["fit"]["mics"]]
    print(
        f"wrote {pay_path}\n      {report_path}\n"
        f"chosen law: offset {ch['speed_offset_ms']:g} m/s, scale {ch['speed_scale_ms']:g} m/s, "
        f"rate {ch['gust_rate_hz']:g} Hz (distance {ch['distance']:.4g}; "
        f"{st['gust_law']['n_finished']}/{st['gust_law']['n_points']} grid points)\n"
        f"level_db {[round(x, 2) for x in w['level_db']]}\n"
        f"   prior centre (source wind, 20-100 Hz) {[round(x, 2) for x in centre]}\n"
        f"distance by statistic { ({k: round(v, 2) for k, v in ch['terms']['by_stat'].items()}) }\n"
        + "\n".join(DW.stats_table(st["real"], st["fitted"]))
        + f"\nwind renderer check: max |band err| {chk['max_abs_err_db']:.3f} dB "
        f"(tol {chk['tol_db']:g}, {chk['n_seeds']} seeds) passed {chk['passed']}\n"
        f"grid point wall: median {st['point_wall_s']['median']:.1f}s, max "
        f"{st['point_wall_s']['max']:.1f}s; best 3 "
        f"{[(t['uid'], round(t['distance'], 2)) for t in report['table'][:3]]}\n"
        f"wall {time.time() - t0:.1f}s (mean level {np.mean(w['level_db']):.2f} dB)",
        flush=True,
    )
    return 0 if chk["passed"] else 1


def cmd_gust(args: argparse.Namespace, grid: dict[str, tuple[float, ...]]) -> int:
    """``wind --targets battery``: :func:`dregon_wind_v4.run_gust_stage`."""
    import numpy as np

    from experiments.noise_model import dregon_wind_v4 as DW

    t0 = time.time()
    res = DW.run_gust_stage(
        args.source,
        out=args.out,
        offsets=grid["offsets"],
        scales=grid["scales"],
        rates=grid["rates"],
        seeds=int(args.seeds),
        floor_pairs=int(args.floor_pairs or DW.N_FLOOR_PAIRS),
        pool=str(args.pool),
        workers=int(args.workers),
        max_segments=args.max_segments,
        n_mc=int(args.n_mc),
        check_seeds=int(args.check_seeds),
        resume=not args.no_resume,
        deadline_min=args.deadline_min,
        progress=int(args.progress),
    )
    if "payload" not in res:
        print(
            f"deadline: stopped in the {res['phase']} phase, no grid point yet; rerun the same "
            f"command to resume ({res['work_dir']})",
            flush=True,
        )
        return 3
    st = res["payload"]["wind_stage"]
    gl, chk, inner = st["gust_law"], st["renderer_check"], st["inner"]
    ch, sp = gl["chosen"], gl["source_point"]
    floors = st["targets"]["seed_floor"]["floor"]
    note = f"; {inner['pool_note']}" if inner["pool_note"] else ""

    def law(p: dict[str, Any]) -> str:
        return (
            f"offset {p['speed_offset_ms']:g} m/s, scale {p['speed_scale_ms']:g} m/s, rate "
            f"{p['gust_rate_hz']:g} Hz"
        )

    rows = [
        f"wrote {res['payload_path']}\n      {res['report_path']}",
        f"chosen law: {law(ch)} (objective {ch['objective']:.4g}; {gl['n_finished']}/"
        f"{gl['n_points']} grid points{'' if gl['complete'] else ', INCOMPLETE'})",
        f"level_db {[round(x, 2) for x in ch['level_db']]}\n"
        f"   prior centre {[round(r['centre_db'], 2) for r in ch['inner']['mics']]}",
        f"floor mean_db {ch['floor_mean_db']:.3f} (source {st['source_floor_mean_db']:.3f}); "
        f"inner converged {ch['inner']['converged']} in {ch['inner']['n_sweeps']} sweeps; "
        f"pool {inner['pool']} (asked {inner['pool_requested']}){note}",
    ]
    if sp is not None:
        rows.append(f"source law: {law(sp)}: objective {sp['objective']:.4g}, rank {sp['rank']}")
    rows.append(
        f"{'term':<42} {'seed floor':>10} {'chosen':>9} {'ratio':>6}"
        + ("" if sp is None else f" {'source':>9} {'ratio':>6}")
    )
    for j in DW.BATTERY_TERMS:
        c = ch["terms"]["by_term"][j]
        row = f"{j:<42} {floors[j]:10.4g} {c['mean']:9.4g} {c['ratio']:6.2f}"
        if sp is not None:
            s = sp["terms"]["by_term"][j]
            row += f" {s['mean']:9.4g} {s['ratio']:6.2f}"
        rows.append(row)
    parts = st["point_wall_s"]["parts_median"]
    rows += [
        f"wind renderer check: max |band err| {chk['max_abs_err_db']:.3f} dB (tol "
        f"{chk['tol_db']:g}, {chk['n_seeds']} seeds) passed {chk['passed']}",
        f"grid point wall: median {st['point_wall_s']['median']:.1f}s, max "
        f"{st['point_wall_s']['max']:.1f}s; median parts "
        + ", ".join(f"{k} {v:.1f}" for k, v in parts.items()),
        f"wall {time.time() - t0:.1f}s (mean level {np.mean(ch['level_db']):.2f} dB)",
    ]
    print("\n".join(rows), flush=True)
    if not gl["complete"]:
        return 3
    return 0 if chk["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    from experiments.noise_model import flight_v4 as FV

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="one restart")
    f.add_argument("--out", required=True)
    f.add_argument("--name", default=NAME)
    f.add_argument("--sets", nargs="+", default=list(SETS))
    f.add_argument("--support", nargs="*", default=None, help="explicit support specs instead")
    f.add_argument("--max-windows", type=int, default=None)
    f.add_argument("--mics", default=None, metavar="I,J,...")
    f.add_argument("--k-cap", type=int, default=K_CAP)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--device", choices=("cuda", "cpu"), default=None)
    f.add_argument("--kappa", type=float, default=0.0, help="floor push (spec, calibrated in A)")
    f.add_argument("--wander", default=FV.DEFAULT_WANDER)
    f.add_argument("--channel-gains", default=CHANNEL_GAINS)
    f.add_argument(
        "--channel-gains-band", choices=("above_500", "full", "transfer"), default="transfer"
    )
    f.add_argument("--frame-stride", type=int, default=4)
    f.add_argument("--chunk-frames", type=int, default=16)
    f.add_argument("--work-rate", type=int, default=32000)
    f.add_argument("--threads", type=int, default=4)
    f.add_argument("--adam-steps", type=int, default=1500)
    f.add_argument("--adam-lr", type=float, default=0.02)
    f.add_argument("--adam-batch", type=int, default=8)
    f.add_argument("--lbfgs-iters", type=int, default=600)
    f.add_argument("--lbfgs-frames", type=int, default=64)
    f.add_argument("--polish-iters", type=int, default=150)
    f.add_argument("--progress", type=int, default=0)
    f.add_argument(
        "--start",
        default=None,
        help="continue from this flight payload (same pool); use with --adam-steps 0 "
        "--lbfgs-iters 0 so only the all-frames polish and its restart run",
    )
    f.add_argument("--tag", default="", help="suffix of the restart file (a continuation)")
    f.add_argument(
        "--deadline-min", type=float, default=None, help="no restart pass ends after this"
    )
    f.add_argument(
        "--time-eval", action="store_true", help="time ONE objective call on the pool and exit"
    )
    f.add_argument(
        "--free-phase-pedestal",
        action="store_true",
        help="fit sigma2_phi and W_phi per order under the bench priors (default: held at "
        "the prior medians); a --start payload that fitted them needs it",
    )
    r = sub.add_parser("reduce", help="lowest objective over the restarts")
    r.add_argument("--out", required=True)
    r.add_argument("--name", default=NAME)
    d = sub.add_parser(
        "dregon",
        help="DREGON flight stage: line gains (or, --lines, the low-order lines) + wind + floor "
        "on a bench or flight payload",
    )
    d.add_argument(
        "--from",
        "--bench",
        dest="source",
        default=None,
        help="a DREGON noise-v4-fit/3 bench payload, or a flight payload (its floor and wind "
        "levels the start, its sc wind's shape held); not needed with --resume",
    )
    d.add_argument(
        "--pool",
        choices=tuple(POOL_TAGS),
        default="floor",
        help="floor: the five 8 s dregon-floor segments; wide: the 18 dregon-flight-wide "
        "windows (output <...>__wide<mode>.json)",
    )
    d.add_argument(
        "--kappa", type=float, default=None, help="floor push (default: the source payload's)"
    )
    d.add_argument("--frames-thin", type=int, default=4, help="keep every N-th frame")
    d.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    d.add_argument("--threads", type=int, default=4)
    d.add_argument("--out", default="results/noise_v4/flight")
    d.add_argument("--max-frames", type=int, default=None, help="first N frames only (smoke)")
    d.add_argument("--chunk-frames", type=int, default=16)
    d.add_argument("--iters", type=int, default=200, help="L-BFGS iterations of the first pass")
    d.add_argument("--restart-iters", type=int, default=100, help="... of each fresh pass")
    d.add_argument("--max-passes", type=int, default=20)
    d.add_argument(
        "--wind-start", default=None, help="nv3r4 DREGON payload (a bench source's wind start)"
    )
    d.add_argument("--progress", type=int, default=0, help="print every N frozen frames")
    d.add_argument(
        "--no-line-gain",
        action="store_true",
        help="hold the per-rotor line gains at 0 (the bench line levels; --lines does)",
    )
    d.add_argument(
        "--no-mic-gain",
        action="store_true",
        help="hold the per-mic line gains at 0 (the bench pair EQ levels)",
    )
    d.add_argument(
        "--coherence",
        choices=tuple(COHERENCE_TAGS),
        default="none",
        help="the line coherence in flight: none (the bench's, frozen L_r), shaft (fit the "
        "shaft eps only, the bench phase pedestal held; output <...>__shaft.json), full "
        "(fit the phase pedestal per aerodynamic order and eps; output <...>__coh.json) or "
        "lines (= --lines)",
    )
    d.add_argument(
        "--lines",
        dest="coherence",
        action="store_const",
        const="lines",
        help="the line refit (--coherence lines): profile_aero_db[:, :k_free] under "
        "N(bench, --profile-sd-db), the phase pedestal of k <= k_free under the bench priors "
        "and eps free; line gains held; output <...>__lines.json",
    )
    d.add_argument(
        "--k-free", type=int, default=None, help="--lines: the free orders (default K_FREE = 30)"
    )
    d.add_argument(
        "--profile-sd-db",
        type=float,
        default=None,
        help="--lines: the profile prior's sd (default PROFILE_FLIGHT_SD_DB = 6 dB)",
    )
    d.add_argument(
        "--hold-eps",
        action="store_true",
        help="--lines: hold the shaft scale eps at the source's (the free low orders "
        "stay inside one bin despite the label error)",
    )
    d.add_argument(
        "--hold-pedestal",
        action="store_true",
        help="--lines: hold the phase pedestal of every order at the source's (thin lines)",
    )
    d.add_argument(
        "--wind-from",
        default=None,
        help="replace the source's params.wind (sc law, shape, levels) by this payload's",
    )
    d.add_argument(
        "--deadline-min",
        type=float,
        default=None,
        help="stop the fit in time to write the payload (exit 3: resume it)",
    )
    d.add_argument(
        "--resume",
        default=None,
        help="continue this stage payload (its source and settings) as <stem>__c<n+1>.json",
    )
    from experiments.noise_model import dregon_wind_v4 as DW

    wd = sub.add_parser(
        "wind", help="DREGON wind stage: the SC gust wind per capsule on a flight-stage payload"
    )
    wd.add_argument("--from", dest="source", required=True, help="a DREGON flight payload")
    wd.add_argument(
        "--targets",
        choices=DW.TARGETS,
        default="envelope",
        help="envelope: C-wind's floor-envelope moments (thrust-regressed, 50-500 Hz); battery: "
        "the seed-floor-normalised spectro-temporal + level-distribution distances, full band, "
        "with the floor mean_db refitted at each law",
    )
    wd.add_argument("--offsets", type=float, nargs="+", default=None, help="default: the mode's")
    wd.add_argument("--scales", type=float, nargs="+", default=None, help="default: the mode's")
    wd.add_argument("--rates", type=float, nargs="+", default=None, help="default: the mode's")
    wd.add_argument("--seeds", type=int, default=DW.N_SEEDS, help="sim seeds per segment")
    wd.add_argument("--workers", type=int, default=1, help="parallel grid processes")
    wd.add_argument("--out", default="results/noise_v4/flight/wind")
    wd.add_argument("--max-segments", type=int, default=None, help="first N segments (smoke)")
    wd.add_argument("--n-mc", type=int, default=DW.SHAPE_N_MC, help="shape Monte-Carlo clips")
    wd.add_argument("--check-seeds", type=int, default=DW.CHECK_SEEDS)
    wd.add_argument("--no-resume", action="store_true", help="re-run every finished unit")
    wd.add_argument("--progress", type=int, default=0)
    wd.add_argument(
        "--pool",
        choices=DW.POOLS,
        default="floor",
        help="battery: the inner Whittle pool (wide: dregon_flight_v4.wide_pool_specs(); falls "
        "back to floor without it)",
    )
    wd.add_argument(
        "--floor-pairs",
        type=int,
        default=None,
        help=f"battery: seed-floor render pairs per segment (default {DW.N_FLOOR_PAIRS})",
    )
    wd.add_argument(
        "--deadline-min",
        type=float,
        default=None,
        help="battery: launch no unit chunk that would end after this; exit 3, rerun to resume",
    )
    args = ap.parse_args(argv)
    if args.cmd == "wind":
        return cmd_wind(args)
    if args.cmd == "dregon":
        if args.wind_start is None:
            from experiments.noise_model import dregon_flight_v4 as DF

            args.wind_start = DF.WIND_START
        return cmd_dregon(args)
    return cmd_fit(args) if args.cmd == "fit" else cmd_reduce(args)


if __name__ == "__main__":
    raise SystemExit(main())
