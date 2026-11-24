"""Fixed-schedule test of the unified whole-flight generator, restartable and portable.

A separate experiment beside :mod:`experiments.rps_traj.schedule_eval`: it
reuses that module's frozen targets, folds, windows, donors, hover, cruise
metric and seeds, writes under its own root (default ``results/rps_traj/unified``)
and never touches the baseline records in ``<campaign>/schedule_eval/``.

Commands (``python -m experiments.rps_traj.unified_experiment ...``)::

    pack    [--campaign results/rps_traj/regimes] [--baseline <campaign>/schedule_eval/schedule_models.json] [--out OUT]
    fit     --rig R --fold F --arm hierarchical|flat [--out OUT] [--max-iter 30] [--force] [--score]
    collect [--out OUT] [--rig R] [--fold F] [--arm A] [--fit-missing]

* ``pack`` (local, needs the datasets and the campaign stack) runs
  ``schedule_eval.plan`` per rig and checks it against the frozen
  ``schedule_models.json`` (same 13 measured targets on 3 rigs, the same 65
  target x phase cells, held-out IDs, supports, donor counts, hover, seeds
  and exclusions); a manifest is published only if all checks pass.
* ``fit`` (portable worker) reads only ``pack/<rig>__<fold>.train.npz`` and
  ``pack/<rig>__<fold>.json``, needs NumPy plus whatever
  :mod:`experiments.rps_traj.unified_response` needs, fits one arm
  (``hierarchical`` / ``flat``), draws :data:`N_DRAWS` whole flights per target
  on the target's own clock and saves the fit, the paths and the scored
  windows.  It never opens the evaluation file and does not score unless
  ``--score`` is given (scoring imports the campaign stack, torch included).
  A unit whose outputs exist for the same packed inputs is skipped.
* ``collect`` scores every available unit with
  :func:`experiments.rps_traj.schedule_eval._score` (hence
  :func:`~experiments.rps_traj.schedule_score.score_phase`, origin-averaged as
  the baselines), pairs each cell with the baseline rows, verifies the donor
  scales against the baseline diagnostics and writes ``unified_models.json``.

Training tracks: every recording of the rig outside the fold's held-out set,
whole (ground to ground), on the campaign grid (100 Hz, NaN linearly bridged,
4-sample block means to 25 Hz).  DREGON's command-only recordings
(``motors_command``, standby recovered from the audio) stay in the training
split as the campaign and the baselines had them (legacy, flagged
``measured: false``); they are never targets or donors.  A recording whose
landmarks cannot be found (no airborne segment) has no schedule and is left
out of training, listed in ``train_excluded``; the baselines' labelled phase
VARs drop the same recordings.

Schedule given to the model: only the coarse commanded events
``motors_on``, ``takeoff`` and ``landing`` of the recording, as one int8 code
per 25 Hz sample (:func:`schedule_codes`): 0 before motors-on, 1 from
motors-on to take-off (start-up and standby), 2 from take-off to the landing
or the end (take-off and cruise), 3 from the landing to the end (rundown and
rest).  A recording that starts airborne is 2 from its first sample; one
without a landing stays 2 to its end.  No arrival, standby level, motors-off,
hover or target speed reaches the model, and the codes carry no reset: the
model sees one continuous track.

Generated paths have exactly the target's length ``T``; sample ``t`` of a path
is sample ``t`` of the target, so windows are cut at the frozen target starts
with no shift and no padding.  One seed per (rig, fold, target), the frozen
``schedule_eval`` key ``(rig, fold, flight, 'draws')``, is shared by both arms.

Layout under OUT (``KEY = <rig>__<fold>``, ``F`` a flight id, ``P`` a phase,
rev/s at 25 Hz, ``O`` origins: 3 for cruise, 1 otherwise, ``D`` donors)::

    pack/manifest.json        protocol, metric params, exclusions, verification, baseline paths and param counts
    pack/KEY.train.npz        train_x (4, N) f64, train_phase (N,) int8, train_offsets (k+1,) int64
    pack/KEY.eval.npz         F__x (4, T), F__phase (T,), F__P__obs (O, 4, n), F__P__refs (O, D, 4, n)
    pack/KEY.json             training tracks, targets (events, seed, cells), baseline rows per cell
    fits/KEY__ARM.json        to_dict(), history, n_params, counts, per-target draw status
    draws/KEY__ARM.npz        F__paths (32, 4, T) f32, F__P__draws (O, 32, 4, n) f64, F__phase (T,) int8
    scores/KEY__ARM.json      rows in ``schedule_eval._row`` format, arm ``unified_<ARM>``
    unified_models.json       every row, fits summary, verification
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

#: Frozen copies of the ``schedule_eval`` constants (checked by ``pack``).
FS = 25.0
N_DRAWS = 32
PHASES = ("startup", "standby", "takeoff", "cruise", "landing")
RIGS = ("michaels", "vid_m100", "dregon")
N_TARGETS = 13
N_CELLS = 65
ARMS = {"hierarchical": True, "flat": False}
MAX_ITER = 30
EVENTS = ("motors_on", "takeoff", "landing")
CODES = {
    0: "before motors-on",
    1: "motors-on to take-off",
    2: "take-off to landing/end",
    3: "landing to end",
}
#: Donor-only scale diagnostics of ``score_phase``: equal to the baseline's when the refs and hover are.
SCALE_KEYS = {"cruise": ("joint_scale",), "short": ("path_scale_rps", "summary_scale_rps")}
DEFAULT_OUT = Path("results/rps_traj/unified")
DEFAULT_CAMPAIGN = Path("results/rps_traj/regimes")


# ─── portable helpers (NumPy only) ────────────────────────────────────────────


def schedule_codes(n: int, events: Mapping[str, int | None]) -> np.ndarray:
    """``(n,)`` int8 phase codes of a recording from its coarse events."""
    on, takeoff, landing = (events.get(k) for k in EVENTS)
    if (on is None) != (takeoff is None):
        raise ValueError(f"motors-on and take-off must both be known or both absent: {events}")
    marks = [m for m in (on, takeoff, landing) if m is not None]
    if marks != sorted(marks) or any(m < 0 or m > n for m in marks):
        raise ValueError(f"events out of order or outside [0, {n}]: {events}")
    codes = np.zeros(n, dtype=np.int8)
    if on is None:
        codes[:] = 2
    else:
        codes[on:takeoff] = 1
        codes[takeoff:] = 2
    if landing is not None:
        codes[landing:] = 3
    return codes


def _key(rig: str, fold: str) -> str:
    return f"{rig}__{fold}"


def _paths(out: Path, rig: str, fold: str, arm: str | None = None) -> dict[str, Path]:
    k = _key(rig, fold)
    p = {
        "meta": out / "pack" / f"{k}.json",
        "train": out / "pack" / f"{k}.train.npz",
        "eval": out / "pack" / f"{k}.eval.npz",
    }
    if arm is not None:
        p |= {
            "fit": out / "fits" / f"{k}__{arm}.json",
            "draws": out / "draws" / f"{k}__{arm}.npz",
            "scores": out / "scores" / f"{k}__{arm}.json",
        }
    return p


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: Any, *, allow_nan: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=1, allow_nan=allow_nan) + "\n")
    tmp.replace(path)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, allow_pickle=True, **arrays)
    tmp.replace(path)


def _jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x))


# ─── pack (local: datasets + campaign stack) ──────────────────────────────────


def _events(marks: Any) -> dict[str, int | None]:
    return {k: None if getattr(marks, k) is None else int(getattr(marks, k)) for k in EVENTS}


def _numeric_size(x: Any) -> int:
    if x is None or isinstance(x, (bool, str)):
        return 0
    if isinstance(x, np.ndarray):
        return int(x.size)
    if isinstance(x, (tuple, list)):
        return sum(_numeric_size(v) for v in x)
    return 1 if isinstance(x, (int, float, np.number)) else 0


def baseline_param_counts(campaign: Path, rig: str, fold: str) -> dict[str, Any]:
    """Free parameters of the five baseline arms of one fold.

    ``phasevar{p}``: six per-phase Gaussian VAR(p) with intercept, full noise
    covariance (``6 x (4 + 10 + 16p)``; the labelled fit's initial and
    transition probabilities are unused when the labels are given).
    Count the learned cap once for each VAR arm. The template count includes
    the nine moment-fit hover parameters omitted from the cruise BIC count,
    plus the phase layer, including its stored time samples (not independent
    statistical degrees of freedom). Do not refit models to count parameters.
    """
    from utils.gridrun import unit_path

    def cached(family: str, arm: str) -> tuple[int, str]:
        path = unit_path(campaign, f"cruise__{rig}__{fold}__{family}")
        return int(json.loads(path.read_text())["arms"][arm]["n_params"]), str(path)

    var = {p: 6 * (4 + 10 + 16 * p) + 1 for p in (2, 4)}
    out: dict[str, Any] = {
        f"phasevar{p}": {"n_params": n, "source": "6 x (4 + 10 + 16p) + training-derived cap"}
        for p, n in var.items()
    }
    for arm, (family, inner) in {
        "phasevar2+msvar3": ("msvar", "msvar3"),
        "phasevar2+hsmm3": ("hsmm", "hsmm3"),
    }.items():
        n, src = cached(family, inner)
        out[arm] = {
            "n_params": var[2] + n + 4,
            "source": f"phasevar2 + {src}:arms.{inner}.n_params + 4 level",
            "inner_n_params": n,
        }
    n, src = cached("ours", "switch2")
    phase_path = unit_path(campaign, f"flight__{rig}__{fold}__ours")
    layer = json.loads(phase_path.read_text())["params"]
    layer_n = sum(_numeric_size(v) for k, v in layer.items() if k not in ("fs", "counts"))
    out["ours"] = {
        "n_params": n + 9 + layer_n,
        "source": f"{src}, 9 hover scalars and {phase_path}",
        "cruise_n_params": n,
        "hover_parameters": 9,
        "phase_layer_values": layer_n,
    }
    return out


def _check(ok: bool, what: str) -> None:
    if not ok:
        raise AssertionError(f"pack: frozen protocol mismatch: {what}")


def pack(campaign: Path, baseline: Path, out: Path) -> list[Path]:
    """Freeze every portable input of the experiment under ``out/pack``."""
    from experiments.rps_traj import regime_campaign as rc
    from experiments.rps_traj import schedule_eval as se
    from experiments.rps_traj.phases import landmarks

    _check(
        se.FS == FS and se.N_DRAWS == N_DRAWS and tuple(se.PHASES) == PHASES,
        "schedule_eval constants",
    )
    _check(tuple(rc.FLIGHT_RIGS) == RIGS, "flight rigs")
    old = json.loads(baseline.read_text())
    controls_path = baseline.parent / "schedule_controls.json"
    controls = json.loads(controls_path.read_text())
    old_cells: dict[tuple, dict[str, dict]] = {}
    for row in old["rows"]:
        k = (row["rig"], row["fold"], row["flight"], row["phase"])
        _check(
            row["arm"] not in old_cells.setdefault(k, {}),
            f"duplicate baseline row {row['arm']} {k}",
        )
        old_cells[k][row["arm"]] = row
    real = {
        (r["rig"], r["fold"], r["flight"], r["phase"]): r
        for r in controls["rows"]
        if r["arm"] == "real"
    }

    metrics, metric_meta = se.cruise_metrics(RIGS)
    excluded: list[dict] = []
    written: list[Path] = []
    folds_meta: dict[str, Any] = {}
    cells_seen: set[tuple] = set()
    targets_seen: set[tuple] = set()
    for rig in RIGS:
        plans, exc = se.plan(rig, campaign, require_caches=False)
        excluded += exc
        data = rc.rig_data(rig)
        cal = {tuple(x) for x in metric_meta["rigs"][rig]["calibration_recordings"]}
        for fp in plans:
            _check(
                not any((rig, t.name) in cal for t in fp.targets),
                f"{rig}/{fp.fold} target in calibration",
            )
            old_fold = old["meta"]["folds"][f"{rig}/{fp.fold}"]
            _check(old_fold["held_out"] == fp.meta["held_out"], f"{rig}/{fp.fold} held-out ids")
            _check(old_fold["targets"] == fp.meta["targets"], f"{rig}/{fp.fold} targets")
            _check(
                old_fold["train_hover_rps"] == fp.meta["train_hover_rps"], f"{rig}/{fp.fold} hover"
            )
            held = set(fp.meta["held_out"])

            # Training tracks: whole recordings outside the fold, campaign grid.
            xs, codes, train_rows, train_excl = [], [], [], []
            offset = 0
            for f in data.flights:
                if f.flight in held:
                    continue
                x = rc._to_flight_rate(f.rps, f.fs)
                measured = "motors_command" not in f.source
                marks = landmarks(x, FS)
                if marks is None:
                    train_excl.append(
                        {
                            "flight": f.flight,
                            "source": f.source,
                            "measured": measured,
                            "samples": int(x.shape[1]),
                            "reason": "no landmarks (no airborne segment)",
                        }
                    )
                    continue
                ev = _events(marks)
                c = schedule_codes(x.shape[1], ev)
                bad = int((~np.isfinite(np.asarray(f.rps))).any(axis=0).sum())
                train_rows.append(
                    {
                        "flight": f.flight,
                        "source": f.source,
                        "measured": measured,
                        "offset": offset,
                        "samples": int(x.shape[1]),
                        "seconds": x.shape[1] / FS,
                        "events": ev,
                        "code_samples": np.bincount(c, minlength=4).tolist(),
                        "nonfinite_grid_samples_bridged": bad,
                    }
                )
                xs.append(x)
                codes.append(c)
                offset += x.shape[1]
            _check(bool(xs), f"{rig}/{fp.fold} has no training track")
            paths = _paths(out, rig, fp.fold)
            _write_npz(
                paths["train"],
                {
                    "train_x": np.concatenate(xs, axis=1),
                    "train_phase": np.concatenate(codes),
                    "train_offsets": np.array(
                        [r["offset"] for r in train_rows] + [offset], dtype=np.int64
                    ),
                },
            )

            # Targets and cells, checked against the frozen baseline rows.
            ev_arrays: dict[str, np.ndarray] = {}
            targets = []
            for t in fp.targets:
                ev = _events(t.marks)
                _check(
                    all(v is not None for v in ev.values()),
                    f"{rig}/{t.name} lacks a coarse event: {ev}",
                )
                c = schedule_codes(t.marks.n, ev)
                ev_arrays[f"{t.name}__x"] = np.asarray(t.x, dtype=np.float64)
                ev_arrays[f"{t.name}__phase"] = c
                _, seed = se._rng(rig, fp.fold, t.name, "draws")
                cells = []
                for g in (g for g in fp.groups if g.flight == t.name):
                    k = (rig, fp.fold, t.name, g.phase)
                    rows = old_cells.get(k, {})
                    _check(set(rows) == set(se.ARMS), f"baseline arms of {k}: {sorted(rows)}")
                    for arm, row in rows.items():
                        _check(row["support"] == _jsonable(g.support), f"support of {k} ({arm})")
                        _check(row["n_refs"] == len(g.refs[0]), f"n_refs of {k} ({arm})")
                        _check(row["train_hover_rps"] == g.hover, f"hover of {k} ({arm})")
                        _check(row["seed"] == seed, f"seed of {k} ({arm})")
                        _check(row["n_draws"] == N_DRAWS, f"n_draws of {k} ({arm})")
                    _check(k in real, f"real control of {k}")
                    ev_arrays[f"{t.name}__{g.phase}__obs"] = np.stack(g.obs)
                    ev_arrays[f"{t.name}__{g.phase}__refs"] = np.stack(g.refs)
                    cells.append(
                        {
                            "phase": g.phase,
                            "starts": list(g.starts),
                            "n": g.n,
                            "hover": g.hover,
                            "n_refs": int(len(g.refs[0])),
                            "support": g.support,
                            "baseline": {
                                arm: {
                                    "score": r["score"],
                                    "status": r["status"],
                                    "diagnostics": r["diagnostics"],
                                }
                                for arm, r in sorted(rows.items())
                            }
                            | {
                                "real": {
                                    "score": real[k]["score"],
                                    "status": real[k]["status"],
                                    "diagnostics": real[k]["diagnostics"],
                                }
                            },
                        }
                    )
                    cells_seen.add(k)
                targets.append(
                    {
                        "flight": t.name,
                        "samples": int(t.marks.n),
                        "events": ev,
                        "seed": seed,
                        "code_samples": np.bincount(c, minlength=4).tolist(),
                        "cells": cells,
                    }
                )
                targets_seen.add((rig, t.name))
            _write_npz(paths["eval"], ev_arrays)
            counts = {
                "n_tracks": len(train_rows),
                "n_measured": sum(r["measured"] for r in train_rows),
                "n_command_only": sum(not r["measured"] for r in train_rows),
                "n_excluded": len(train_excl),
                "samples": offset,
                "seconds": offset / FS,
                "code_samples": np.bincount(np.concatenate(codes), minlength=4).tolist(),
            }
            meta = {
                "rig": rig,
                "fold": fp.fold,
                "fs_hz": FS,
                "n_draws": N_DRAWS,
                "held_out": fp.meta["held_out"],
                "train_hover_rps": fp.meta["train_hover_rps"],
                "train": train_rows,
                "train_excluded": train_excl,
                "train_counts": counts,
                "train_sha256": _sha256(paths["train"]),
                "eval_sha256": _sha256(paths["eval"]),
                "targets": targets,
                "baseline_param_counts": baseline_param_counts(campaign, rig, fp.fold),
                "schedule_eval_fold_meta": fp.meta,
            }
            _write_json(paths["meta"], meta)
            folds_meta[_key(rig, fp.fold)] = {
                k: meta[k]
                for k in (
                    "held_out",
                    "train_counts",
                    "train_sha256",
                    "eval_sha256",
                    "baseline_param_counts",
                )
            } | {"targets": [t["flight"] for t in targets]}
            written += [paths["train"], paths["eval"], paths["meta"]]

    _check(_jsonable(excluded) == old["exclusions"], "exclusions")
    _check(
        cells_seen == set(old_cells),
        f"cells: missing {set(old_cells) - cells_seen}, extra {cells_seen - set(old_cells)}",
    )
    _check(len(cells_seen) == N_CELLS and len(targets_seen) == N_TARGETS, "13 targets / 65 cells")
    manifest = {
        "protocol": {
            "fs_hz": FS,
            "n_draws": N_DRAWS,
            "phases": list(PHASES),
            "rigs": list(RIGS),
            "arms": {a: {"hierarchical": h, "row_arm": f"unified_{a}"} for a, h in ARMS.items()},
            "max_iter_default": MAX_ITER,
            "schedule_codes": CODES,
            "schedule_events": list(EVENTS),
            "schedule_policy": "codes from motors_on/takeoff/landing only; starts airborne -> 2 from 0; "
            "no landing -> 2 to the end; no arrival, standby level, motors-off, hover or "
            "target speed given to the model; no reset of the model across codes",
            "training": "whole recordings outside the fold's held-out ids on the campaign grid "
            "(regime_campaign._to_flight_rate); DREGON command-only recordings (standby recovered "
            "from audio) kept as in the campaign train split, flagged measured=false; recordings "
            "without landmarks excluded and listed per fold",
            "generation_clock": "paths have the target's length; sample t is the target's sample t; windows "
            "cut at the frozen schedule_eval starts, no shift, no padding",
            "seeds": "default_rng(SeedSequence(seed)) with the frozen schedule_eval key "
            "[2026, crc32('rig|fold|flight|draws')], shared by both arms",
            "scoring": "schedule_eval._score (score_phase, key-wise mean over cruise origins) with the frozen "
            "per-rig CruiseMetric below",
        },
        "metric": {
            rig: {
                "band_power": metrics[rig].band_power.tolist(),
                "level_scale_rps": metrics[rig].level_scale_rps,
                "spectrum_scale": metrics[rig].spectrum_scale,
            }
            for rig in RIGS
        },
        "metric_meta": metric_meta,
        "baseline": {
            "models": str(baseline),
            "controls": str(controls_path),
            "campaign": str(campaign),
            "models_sha256": _sha256(baseline),
            "controls_sha256": _sha256(controls_path),
            "arms": list(se.ARMS),
            "param_counts": "pack/<rig>__<fold>.json:baseline_param_counts",
        },
        "exclusions": excluded,
        "verification": {
            "targets": len(targets_seen),
            "cells": len(cells_seen),
            "rigs": len(RIGS),
            "matched": "held-out ids, targets, hover, supports, n_refs, seeds, n_draws, exclusions "
            "against the baseline schedule_models.json",
        },
        "folds": folds_meta,
    }
    path = out / "pack" / "manifest.json"
    _write_json(path, manifest)
    return [*written, path]


# ─── fit (portable worker) ────────────────────────────────────────────────────


def _done(
    paths: Mapping[str, Path], train_sha: str, max_iter: int, source_sha: str, select_response: bool
) -> bool:
    if not (paths["fit"].exists() and paths["draws"].exists()):
        return False
    fit = json.loads(paths["fit"].read_text())
    return (
        fit.get("train_sha256") == train_sha
        and fit.get("max_iter") == max_iter
        and fit.get("select_response", False) == select_response
        and fit.get("source_sha256") == source_sha
        and fit.get("draws_sha256") == _sha256(paths["draws"])
    )


def fit_unit(
    out: Path,
    rig: str,
    fold: str,
    arm: str,
    *,
    max_iter: int = MAX_ITER,
    force: bool = False,
    score: bool = False,
    initial_fits: Path | None = None,
    select_response: bool = False,
) -> dict[str, Path]:
    """Fit one arm on one fold's training tracks and draw every target."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {sorted(ARMS)}")
    paths = _paths(out, rig, fold, arm)
    meta = json.loads(paths["meta"].read_text())
    if _sha256(paths["train"]) != meta["train_sha256"]:
        raise ValueError(f"{paths['train']} does not match its pack metadata")
    module = Path(__file__)
    source_sha = hashlib.sha256(
        "".join(
            _sha256(module.with_name(name))
            for name in ("unified_grammar.py", "unified_response.py", "unified_experiment.py")
        ).encode()
    ).hexdigest()
    if force or not _done(paths, meta["train_sha256"], max_iter, source_sha, select_response):
        from experiments.rps_traj import unified_response as ur

        with np.load(paths["train"]) as z:
            x, codes, offsets = z["train_x"], z["train_phase"], z["train_offsets"]
        tracks = [
            (np.ascontiguousarray(x[:, a:b]), np.ascontiguousarray(codes[a:b]))
            for a, b in zip(offsets[:-1], offsets[1:])
        ]
        t0 = time.perf_counter()
        initial = None
        if initial_fits is not None:
            checkpoint = json.loads((initial_fits / paths["fit"].name).read_text())
            if checkpoint["train_sha256"] != meta["train_sha256"] or checkpoint["arm"] != arm:
                raise ValueError("restart checkpoint belongs to a different fold or arm")
            initial = ur.UnifiedResponseModel.from_dict(checkpoint["params"])
        model = ur.fit(tracks, hierarchical=ARMS[arm], max_iter=max_iter, initial=initial)
        selection = {}
        if select_response:
            # Same anchors, grammar, initialization seed, priors and sample
            # support; only the response poles for transient actions may differ.
            split = ur.fit(tracks, hierarchical=ARMS[arm], max_iter=max_iter, split_response=True)
            for name, candidate in (("shared", model), ("split", split)):
                info = candidate.fit_info
                ll = info["trace"][-1]["data_loglik"]
                selection[name] = {
                    "n_params": candidate.n_params,
                    "train_loglik": ll,
                    "bic": float(-2 * ll + candidate.n_params * np.log(info["n_valid"])),
                    "iterations": info["n_iter"],
                    "stop": info["stop_reason"],
                }
            if selection["split"]["bic"] < selection["shared"]["bic"]:
                model = split
        fit_s = time.perf_counter() - t0
        arrays: dict[str, np.ndarray] = {}
        draws_meta = []
        t0 = time.perf_counter()
        for t in meta["targets"]:
            c = schedule_codes(t["samples"], t["events"])
            arrays[f"{t['flight']}__phase"] = c
            rng = np.random.default_rng(np.random.SeedSequence(t["seed"]))
            status = "ok"
            try:
                paths_ = np.asarray(model.sample(c, N_DRAWS, rng), dtype=np.float64)
                if paths_.shape != (N_DRAWS, 4, t["samples"]):
                    status = f"failed: shape {paths_.shape} != {(N_DRAWS, 4, t['samples'])}"
                elif not np.isfinite(paths_).all():
                    status = "failed: non-finite generated path"
            except Exception as exc:  # reported per row by collect, never dropped
                status = f"failed: generate: {type(exc).__name__}: {exc}"
            if status == "ok":
                arrays[f"{t['flight']}__paths"] = paths_.astype(np.float32)
                for cell in t["cells"]:
                    n = cell["n"]
                    arrays[f"{t['flight']}__{cell['phase']}__draws"] = np.stack(
                        [paths_[:, :, s : s + n] for s in cell["starts"]]
                    )
            draws_meta.append(
                {
                    "flight": t["flight"],
                    "seed": t["seed"],
                    "status": status,
                    "clip_fraction": float(model.last_clip_fraction) if status == "ok" else None,
                }
            )
        sample_s = time.perf_counter() - t0
        _write_npz(paths["draws"], arrays)
        _write_json(
            paths["fit"],
            {
                "rig": rig,
                "fold": fold,
                "arm": arm,
                "hierarchical": ARMS[arm],
                "max_iter": max_iter,
                "source_sha256": source_sha,
                "select_response": select_response,
                "training_bic": selection,
                "n_params": int(model.n_params),
                "history": [float(v) for v in model.history],
                "train_sha256": meta["train_sha256"],
                "train_counts": meta["train_counts"],
                "n_draws": N_DRAWS,
                "draws": draws_meta,
                "draws_sha256": _sha256(paths["draws"]),
                "fit_seconds": fit_s,
                "sample_seconds": sample_s,
                "versions": {"python": sys.version.split()[0], "numpy": np.__version__},
                "params": model.to_dict(),
            },
            allow_nan=True,
        )
    if score:
        score_unit(out, rig, fold, arm)
    return paths


# ─── collect (local: campaign stack) ──────────────────────────────────────────


def _metric(manifest: Mapping[str, Any], rig: str) -> Any:
    from experiments.rps_traj.spectral_validation import CruiseMetric

    m = manifest["metric"][rig]
    return CruiseMetric(
        np.asarray(m["band_power"], dtype=np.float64),
        float(m["level_scale_rps"]),
        float(m["spectrum_scale"]),
    )


def score_unit(out: Path, rig: str, fold: str, arm: str) -> Path:
    """Rows of one unit in ``schedule_eval._row`` format, one per frozen cell."""
    from experiments.rps_traj import schedule_eval as se

    manifest = json.loads((out / "pack" / "manifest.json").read_text())
    paths = _paths(out, rig, fold, arm)
    meta = json.loads(paths["meta"].read_text())
    fit = json.loads(paths["fit"].read_text())
    if (
        fit["train_sha256"] != meta["train_sha256"]
        or _sha256(paths["draws"]) != fit["draws_sha256"]
    ):
        raise ValueError(f"{paths['fit']}: stale against its pack or draws")
    if _sha256(paths["eval"]) != meta["eval_sha256"]:
        raise ValueError(f"{paths['eval']} does not match its pack metadata")
    metric = _metric(manifest, rig)
    status_of = {d["flight"]: d["status"] for d in fit["draws"]}
    rows = []
    with np.load(paths["eval"]) as ev, np.load(paths["draws"]) as dr:
        for t in meta["targets"]:
            f = t["flight"]
            for cell in t["cells"]:
                p = cell["phase"]
                g = se.Group(
                    rig,
                    fold,
                    f,
                    p,
                    cell["hover"],
                    tuple(cell["starts"]),
                    cell["n"],
                    tuple(se._frozen(o) for o in ev[f"{f}__{p}__obs"]),
                    tuple(se._frozen(r) for r in ev[f"{f}__{p}__refs"]),
                    cell["support"],
                )
                status, score, diag = status_of[f], None, {}
                if status == "ok":
                    status, score, diag = se._score(g, list(dr[f"{f}__{p}__draws"]), metric)
                base = cell["baseline"]
                rows.append(
                    se._row(
                        "model",
                        f"unified_{arm}",
                        g,
                        N_DRAWS,
                        status,
                        score,
                        diag,
                        t["seed"],
                        n_params=fit["n_params"],
                        baseline={a: b["score"] for a, b in base.items()},
                        donor_scales_match=_scales_match(p, diag, base["ours"]["diagnostics"])
                        if status == "ok"
                        else None,
                    )
                )
    _write_json(
        paths["scores"],
        {
            "rig": rig,
            "fold": fold,
            "arm": arm,
            "fit": str(paths["fit"]),
            "draws": str(paths["draws"]),
            "draws_sha256": fit["draws_sha256"],
            "rows": rows,
        },
    )
    return paths["scores"]


def _scales_match(phase: str, diag: Mapping[str, Any], old: Mapping[str, Any]) -> bool | None:
    if not diag:
        return None
    keys = SCALE_KEYS["cruise" if phase == "cruise" else "short"]
    return all(bool(np.isclose(diag[k], old[k], rtol=1e-9, atol=0.0)) for k in keys)


def collect(
    out: Path,
    *,
    rigs: Sequence[str] | None,
    folds: Sequence[str] | None,
    arms: Sequence[str] | None,
    fit_missing: bool,
    max_iter: int = MAX_ITER,
) -> tuple[Path, bool]:
    manifest = json.loads((out / "pack" / "manifest.json").read_text())
    units = [(k.rsplit("__", 1)[0], k.rsplit("__", 1)[1]) for k in manifest["folds"]]
    chosen = [
        (r, f, a)
        for r, f in units
        for a in ARMS
        if (rigs is None or r in rigs)
        and (folds is None or f in folds)
        and (arms is None or a in arms)
    ]
    for rig, fold, arm in chosen:  # serial: one unit at a time
        paths = _paths(out, rig, fold, arm)
        if not paths["fit"].exists() and fit_missing:
            fit_unit(out, rig, fold, arm, max_iter=max_iter)
        if paths["fit"].exists():
            score_unit(out, rig, fold, arm)

    rows, fits, missing = [], {}, []
    expected = {
        (r, f, t["flight"], c["phase"])
        for r, f in units
        for t in json.loads(_paths(out, r, f)["meta"].read_text())["targets"]
        for c in t["cells"]
    }
    for rig, fold in units:
        for arm in ARMS:
            paths = _paths(out, rig, fold, arm)
            if not paths["scores"].exists():
                missing.append(f"{_key(rig, fold)}__{arm}")
                continue
            unit = json.loads(paths["scores"].read_text())
            fit = json.loads(paths["fit"].read_text())
            if unit["draws_sha256"] != fit["draws_sha256"]:
                missing.append(f"{_key(rig, fold)}__{arm} (scores stale: rerun collect for it)")
                continue
            rows += unit["rows"]
            fits[f"{_key(rig, fold)}__{arm}"] = {
                "n_params": fit["n_params"],
                "history": fit["history"],
                "max_iter": fit["max_iter"],
                "fit_seconds": fit["fit_seconds"],
                "draws": fit["draws"],
                "path": str(paths["fit"]),
            }
    by_arm: dict[str, set] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], set()).add((r["rig"], r["fold"], r["flight"], r["phase"]))
    cells = {
        a: {"n": len(s), "missing": sorted(map(list, expected - s))} for a, s in by_arm.items()
    }
    mismatch = [
        [r["arm"], r["rig"], r["fold"], r["flight"], r["phase"]]
        for r in rows
        if r["donor_scales_match"] is False
    ]
    failed = [
        [r["arm"], r["rig"], r["fold"], r["flight"], r["phase"], r["status"]]
        for r in rows
        if r["status"] != "ok"
    ]
    complete = (
        not missing
        and len(expected) == N_CELLS
        and set(by_arm) == {f"unified_{a}" for a in ARMS}
        and all(c["n"] == N_CELLS for c in cells.values())
    )
    verification = {
        "complete": complete,
        "expected_cells": len(expected),
        "cells_by_arm": cells,
        "missing_units": missing,
        "donor_scale_mismatch": mismatch,
        "failed_rows": failed,
    }
    path = out / "unified_models.json"
    _write_json(
        path,
        {
            "meta": {
                "protocol": manifest["protocol"],
                "baseline": manifest["baseline"],
                "baseline_param_counts": {
                    k: v["baseline_param_counts"] for k, v in manifest["folds"].items()
                },
                "folds": {
                    k: {kk: v[kk] for kk in ("held_out", "targets", "train_counts")}
                    for k, v in manifest["folds"].items()
                },
                "fits": fits,
            },
            "exclusions": manifest["exclusions"],
            "verification": verification,
            "rows": rows,
        },
    )
    return path, complete and not mismatch and not failed


# ─── CLI ──────────────────────────────────────────────────────────────────────


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack", help="freeze portable inputs (local; datasets needed)")
    p.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    p.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="frozen schedule_models.json (default <campaign>/schedule_eval/schedule_models.json)",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    f = sub.add_parser("fit", help="fit one arm on one fold and draw its targets (portable)")
    f.add_argument("--rig", required=True, choices=RIGS)
    f.add_argument("--fold", required=True)
    f.add_argument("--arm", required=True, choices=sorted(ARMS))
    f.add_argument("--out", type=Path, default=DEFAULT_OUT)
    f.add_argument("--max-iter", type=int, default=MAX_ITER)
    f.add_argument("--force", action="store_true", help="refit even if outputs for this pack exist")
    f.add_argument(
        "--initial-fits", type=Path, help="continue same-fold fits to the total --max-iter budget"
    )
    f.add_argument(
        "--select-response",
        action="store_true",
        help="select shared or transient/cruise poles by training BIC",
    )
    f.add_argument("--score", action="store_true", help="also score (imports the campaign stack)")
    c = sub.add_parser(
        "collect", help="score units, pair with baselines, write unified_models.json"
    )
    c.add_argument("--out", type=Path, default=DEFAULT_OUT)
    c.add_argument("--rig", action="append", choices=RIGS)
    c.add_argument("--fold", action="append")
    c.add_argument("--arm", action="append", choices=sorted(ARMS))
    c.add_argument("--fit-missing", action="store_true", help="fit missing units here, serially")
    c.add_argument("--max-iter", type=int, default=MAX_ITER)
    args = parser.parse_args(argv)
    if args.cmd == "pack":
        baseline = args.baseline or args.campaign / "schedule_eval" / "schedule_models.json"
        for path in pack(args.campaign, baseline, args.out):
            print(path)
        return 0
    if args.cmd == "fit":
        for path in fit_unit(
            args.out,
            args.rig,
            args.fold,
            args.arm,
            max_iter=args.max_iter,
            force=args.force,
            score=args.score,
            initial_fits=args.initial_fits,
            select_response=args.select_response,
        ).values():
            if path.exists() and path.suffix in (".json", ".npz") and path.parent.name != "pack":
                print(path)
        return 0
    path, ok = collect(
        args.out,
        rigs=args.rig,
        folds=args.fold,
        arms=args.arm,
        fit_missing=args.fit_missing,
        max_iter=args.max_iter,
    )
    print(path)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
