"""Fixed-schedule evaluation: real-data controls, then the five example arms.

Every held-out MEASURED flight of a flight rig (:data:`regime_campaign.FLIGHT_RIGS`)
fixes a schedule; each phase window of it is scored by
:func:`experiments.rps_traj.schedule_score.score_phase` against an ensemble:

* **Targets.** Measured flights held out by ``rig_data(rig).held_out`` (the
  campaign folds).  DREGON's command-only recordings (``motors_command``)
  are never targets and never donors; flights without settled cruise are in
  no fold, so they are never targets.
* **Donors (``refs``).** Measured flights of the same rig OUTSIDE the target's
  fold with landmarks, one window per donor, the same unmodified array for
  every control and arm of a (flight, phase, origin) and set read-only.  The
  target is never its own donor.  ``hover`` is the median ``Landmarks.hover``
  of the fold's measured donors (training-only).
* **Preprocessing.** Targets, donors and generators live on the campaign
  grid: 100 Hz common grid, NaN linearly bridged, 4-sample block means to
  25 Hz (``rc._to_flight_rate``).  The cruise metric is calibrated on the
  pilot flight-map windows instead (``spectral_inputs.load_segments``:
  ``resample_poly`` to 25 Hz, no bridging, edges cut).
* **Cruise metric.** One :class:`CruiseMetric` per target rig, fitted on the
  reserved calibration recordings of the OTHER rigs only.
* **Short phases.** ``startup`` (motors-on), ``standby`` (its end = take-off),
  ``takeoff``, ``landing`` from :func:`phase_dynamics.windows`; the window
  offsets of the target and of every donor having the phase are intersected,
  so all real paths are cut inside their own records (never padded).
* **Cruise.** Settled cruise (:meth:`multiscale.Trace.spans`) of each flight,
  narrowed to its longest run free of bridged samples when that run holds
  :data:`CRUISE_N` samples (otherwise the whole span, its bridged samples
  counted); :data:`CRUISE_N` samples at relative origins 0, 0.5, 1 of each
  flight's own span; the three origin scores are averaged per flight.
* **Generated paths.** :data:`N_DRAWS` paths per arm and target from
  ``fit_arm(..., same_clock=True)`` with ONE named seed per (rig, fold,
  flight) shared by all arms (the phasevar2 arms draw identical outer
  innovations).  A path is aligned to the target's clock ONLY by the
  difference of the commanded motors-on events; past its end it is at rest
  (zeros) only when its commanded landing lies inside it and its last
  0.5 s rotor mean is under ``STOPPED_FRACTION`` of the training hover,
  otherwise the arm fails for that phase (reported, never dropped).

Controls are scored on every target before any model (``--controls-only``
stops there).  The model pass reads the fold's cached cruise fits and fails
when one is missing instead of refitting.
"""

from __future__ import annotations

import json
import os
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from experiments.rps_traj import multiscale as ms
from experiments.rps_traj import phase_dynamics as pdyn
from experiments.rps_traj import regime_campaign as rc
from experiments.rps_traj.phases import STOPPED_FRACTION, Landmarks, block_mean, landmarks
from experiments.rps_traj.spectral_inputs import COMMAND_SOURCE
from experiments.rps_traj.spectral_inputs import WINDOW_SAMPLES as CRUISE_N
from experiments.rps_traj.spectral_validation import CruiseMetric, augment
from experiments.rps_traj.switching import training_blocks
from utils.gridrun import unit_path

FS = pdyn.FS
BASE_SEED = 2026
N_DRAWS = 32
MIN_DONORS = 2
ORIGINS = (0.0, 0.5, 1.0)
PHASES = ("startup", "standby", "takeoff", "cruise", "landing")
ARMS = pdyn.EXAMPLE_ARMS
#: The event each short phase is aligned to (``standby`` ends at take-off).
EVENTS = {
    "startup": "motors_on",
    "standby": "standby_end",
    "takeoff": "takeoff",
    "landing": "landing",
}
MIN_SAMPLES = pdyn._n(pdyn.INC_S) + 2
DC_STEP = 0.10
REST_S = 0.5
#: control -> (phases, role)
CONTROLS = {
    "real": ("all", "reference"),
    "dc_plus10": ("all", "positive"),
    "flatten_time": ("short", "positive"),
    "collective": ("short", "positive"),
    "independent_phase": ("cruise", "positive"),
    "shared_phase": ("cruise", "negative"),
    "fast_independent": ("cruise", "positive"),
}
#: cruise unit family -> arm whose params the generators read
CACHES = {"ours": "switch2", "msvar": "msvar3", "hsmm": "hsmm3"}
#: The manuscript's single illustrative trace, fixed before any model score:
#: MD2 FLY124 (an atypical take-off), the arms drawn, the phases saved.
EXAMPLE = ("michaels", "FLY124")
EXAMPLE_ARMS = ("ours", "phasevar4")
EXAMPLE_PHASES = ("startup_standby", "takeoff", "cruise", "landing")


@dataclass(frozen=True)
class Rec:
    """One flight on the campaign grid; ``bridged[t]``: sample t saw a NaN."""

    name: str
    source: str
    x: np.ndarray
    marks: Landmarks
    bridged: np.ndarray


@dataclass(frozen=True)
class Group:
    """One target flight x phase, fixed before any score."""

    rig: str
    fold: str
    flight: str
    phase: str
    hover: float
    starts: tuple[int, ...]  # window start per origin, on the target's clock
    n: int
    obs: tuple[np.ndarray, ...]  # (4, n) per origin
    refs: tuple[np.ndarray, ...]  # (D, 4, n) per origin, read-only
    support: dict[str, Any]


@dataclass(frozen=True)
class FoldPlan:
    rig: str
    fold: str
    targets: tuple[Rec, ...]
    groups: tuple[Group, ...]
    meta: dict[str, Any]


def _rng(*parts: object) -> tuple[np.random.Generator, list[int]]:
    """A generator named by ``parts``: ``[BASE_SEED, crc32("a|b|..")]``."""
    key = [BASE_SEED, zlib.crc32("|".join(map(str, parts)).encode())]
    return np.random.default_rng(np.random.SeedSequence(key)), key


def _measured(source: str) -> bool:
    return COMMAND_SOURCE not in source


def _record(f: Any) -> Rec | None:
    x = rc._to_flight_rate(f.rps, f.fs)
    marks = landmarks(x, FS)
    if marks is None:
        return None
    bad = cast(np.ndarray, (~np.isfinite(np.asarray(f.rps))).any(axis=0)).astype(np.float64)
    bridged = block_mean(bad, f.fs, FS) > 0
    assert bridged.shape[0] == x.shape[1]
    return Rec(f.flight, f.source, x, marks, bridged)


def _window(marks: Landmarks, phase: str) -> tuple[int, int, int] | None:
    """``(event index, lo, hi)`` of a short phase."""
    w = pdyn.windows(marks).get(phase)
    if w is None:
        return None
    event, lo, hi = w
    i = getattr(marks, event)
    if phase == "standby":
        assert marks.standby is not None and marks.standby[1] == i
    return int(i), int(lo), int(hi)


def _cruise_span(rec: Rec) -> tuple[int, int, bool] | None:
    """Settled cruise ``(a, b, has_bridged)``: the longest bridge-free run of
    it when that holds :data:`CRUISE_N` samples, else the whole span."""
    span = ms.Trace(rec.name, rec.x, rec.marks).spans().get("cruise")
    if span is None or span[1] - span[0] < CRUISE_N:
        return None
    a, b = int(span[0]), int(span[1])
    ok = ~rec.bridged[a:b]
    edges = np.flatnonzero(np.diff(np.concatenate([[0], ok.astype(np.int8), [0]])))
    runs = list(zip(edges[::2], edges[1::2]))
    if runs:
        lo, hi = max(runs, key=lambda r: r[1] - r[0])
        if hi - lo >= CRUISE_N:
            return a + int(lo), a + int(hi), False
    return a, b, True


def _starts(a: int, b: int) -> list[int]:
    return [a + int(round(u * (b - a - CRUISE_N))) for u in ORIGINS]


def _frozen(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.float64)
    a.flags.writeable = False
    return a


def _short_group(
    rig: str, fold: str, target: Rec, donors: Sequence[Rec], phase: str, hover: float
) -> Group | str:
    own = _window(target.marks, phase)
    if own is None:
        return f"target has no {phase} window"
    pool = [(d, w) for d in donors if (w := _window(d.marks, phase)) is not None]
    if len(pool) < MIN_DONORS:
        return f"{len(pool)} donors with a {phase} window < {MIN_DONORS}"
    lo = max(own[1], *(w[1] for _, w in pool))
    hi = min(own[2], *(w[2] for _, w in pool))
    if hi - lo < MIN_SAMPLES:
        return f"shared support {hi - lo} samples < {MIN_SAMPLES}"
    cuts = []
    for rec, (i, _, _) in [(target, own), *pool]:
        assert i + lo >= 0 and i + hi <= rec.x.shape[1], "real window outside its record"
        cuts.append((rec.x[:, i + lo : i + hi], int(rec.bridged[i + lo : i + hi].sum())))
    support = {
        "event": EVENTS[phase],
        "lo": lo,
        "hi": hi,
        "samples": hi - lo,
        "seconds": (hi - lo) / FS,
        "origins": [0.0],
        "donors": [d.name for d, _ in pool],
        "donors_without_phase": sorted({d.name for d in donors} - {d.name for d, _ in pool}),
        "target_bridged_samples": cuts[0][1],
        "donor_bridged_samples": [c[1] for c in cuts[1:]],
    }
    return Group(
        rig,
        fold,
        target.name,
        phase,
        hover,
        (own[0] + lo,),
        hi - lo,
        (_frozen(cuts[0][0]),),
        (_frozen(np.stack([c[0] for c in cuts[1:]])),),
        support,
    )


def _cruise_group(
    rig: str, fold: str, target: Rec, donors: Sequence[Rec], hover: float
) -> Group | str:
    own = _cruise_span(target)
    if own is None:
        return f"target settled cruise shorter than {CRUISE_N} samples or absent"
    pool = [(d, s) for d in donors if (s := _cruise_span(d)) is not None]
    if len(pool) < MIN_DONORS:
        return f"{len(pool)} donors with >= {CRUISE_N} settled-cruise samples < {MIN_DONORS}"
    starts = _starts(own[0], own[1])
    donor_starts = [_starts(s[0], s[1]) for _, s in pool]
    obs = tuple(_frozen(target.x[:, t : t + CRUISE_N]) for t in starts)
    refs = tuple(
        _frozen(
            np.stack([d.x[:, ds[k] : ds[k] + CRUISE_N] for (d, _), ds in zip(pool, donor_starts)])
        )
        for k in range(len(ORIGINS))
    )

    def bridged(rec: Rec, ts: Sequence[int]) -> int:
        return int(sum(rec.bridged[t : t + CRUISE_N].sum() for t in ts))

    support = {
        "event": "settled_cruise",
        "samples": CRUISE_N,
        "seconds": CRUISE_N / FS,
        "origins": list(ORIGINS),
        "target_span": [own[0], own[1]],
        "target_starts": starts,
        "target_span_has_bridged": own[2],
        "target_bridged_samples": bridged(target, starts),
        "donors": [d.name for d, _ in pool],
        "donors_without_phase": sorted({d.name for d in donors} - {d.name for d, _ in pool}),
        "donor_spans": [[s[0], s[1]] for _, s in pool],
        "donor_span_has_bridged": [s[2] for _, s in pool],
        "donor_bridged_samples": [bridged(d, ds) for (d, _), ds in zip(pool, donor_starts)],
    }
    return Group(
        rig, fold, target.name, "cruise", hover, tuple(starts), CRUISE_N, obs, refs, support
    )


def provenance(out_dir: Path, rig: str, fold: str, *, required: bool) -> dict[str, Any]:
    """Check the fold's cached cruise fits against this split (held-out ids,
    training block count and seconds).  Missing caches raise when
    ``required``; any mismatch raises."""
    data = rc.rig_data(rig)
    held = sorted(data.held_out[int(fold)])
    train, _ = data.split(fold)
    blocks = training_blocks(train, data.rate_hz)
    train_s = float(sum(b.shape[1] for b in blocks) / data.rate_hz)
    out: dict[str, Any] = {"held_out": held, "train_groups": sorted(train), "caches": {}}
    for family, arm in CACHES.items():
        path = unit_path(out_dir, f"cruise__{rig}__{fold}__{family}")
        if not path.exists():
            if required:
                raise FileNotFoundError(f"missing fold cache {path}; refusing to refit")
            out["caches"][family] = {"path": str(path), "status": "missing"}
            continue
        unit = json.loads(path.read_text())
        problems = [
            k
            for k, ok in (
                ("rig", unit.get("rig") == rig),
                ("fold", str(unit.get("fold")) == fold),
                ("family", unit.get("family") == family),
                ("held_out", sorted(unit.get("held_out", [])) == held),
                ("n_train_blocks", unit.get("n_train_blocks") == len(blocks)),
                ("train_s", np.isclose(unit.get("train_s", np.nan), train_s, rtol=1e-9)),
                (f"arms.{arm}.params", "params" in unit.get("arms", {}).get(arm, {})),
            )
            if not ok
        ]
        if problems:
            raise ValueError(f"{path}: provenance mismatch in {problems}")
        out["caches"][family] = {
            "path": str(path),
            "status": "verified",
            "arm": arm,
            "unit_keys": sorted(unit),
            "arms": sorted(unit["arms"]),
            "n_train_blocks": len(blocks),
            "train_s": train_s,
        }
    return out


def plan(rig: str, out_dir: Path, *, require_caches: bool) -> tuple[list[FoldPlan], list[dict]]:
    """Every (target, phase) group of a rig, plus explicit exclusions."""
    data = rc.rig_data(rig)
    recs: dict[str, Rec | None] = {f.flight: _record(f) for f in data.flights}
    source = {f.flight: f.source for f in data.flights}
    plans, excluded = [], []
    for k, held in enumerate(data.held_out):
        fold = str(k)
        held_set = set(held)

        def skip(flight: str, phase: str, reason: str, fold: str = fold) -> None:
            excluded.append(
                {"rig": rig, "fold": fold, "flight": flight, "phase": phase, "reason": reason}
            )

        donors = [
            r
            for name, r in recs.items()
            if r is not None and name not in held_set and _measured(source[name])
        ]
        if not donors:
            raise ValueError(f"{rig} fold {fold}: no measured donor with landmarks")
        hover = float(np.median([d.marks.hover for d in donors]))
        targets: list[Rec] = []
        groups: list[Group] = []
        for name in sorted(held):
            if not _measured(source[name]):
                skip(name, "all", f"command-only source {source[name]!r}: not a measured target")
                continue
            rec = recs[name]
            if rec is None:
                skip(name, "all", "no landmarks on the campaign grid")
                continue
            made = 0
            for phase in PHASES:
                g = (
                    _cruise_group(rig, fold, rec, donors, hover)
                    if phase == "cruise"
                    else _short_group(rig, fold, rec, donors, phase, hover)
                )
                if isinstance(g, str):
                    skip(name, phase, g)
                else:
                    groups.append(g)
                    made += 1
            if made:
                targets.append(rec)
        train = [f for f in data.flights if f.flight not in held_set]
        meta = {
            "held_out": sorted(held),
            "targets": [t.name for t in targets],
            "train_hover_rps": hover,
            "donor_pool": [{"flight": d.name, "source": d.source} for d in donors],
            "donor_pool_excluded": [
                {
                    "flight": n,
                    "source": source[n],
                    "reason": "command-only" if not _measured(source[n]) else "no landmarks",
                }
                for n in sorted(recs)
                if n not in held_set and (recs[n] is None or not _measured(source[n]))
            ],
            "generator_training_flights": [
                {
                    "flight": f.flight,
                    "source": f.source,
                    "measured": _measured(f.source),
                    "has_landmarks": recs[f.flight] is not None,
                }
                for f in train
            ],
            "provenance": provenance(out_dir, rig, fold, required=require_caches),
        }
        plans.append(FoldPlan(rig, fold, tuple(targets), tuple(groups), meta))
    return plans, excluded


def cruise_metrics(rigs: Sequence[str]) -> tuple[dict[str, CruiseMetric], dict[str, Any]]:
    """One metric per target rig, on the OTHER rigs' calibration recordings."""
    from experiments.rps_traj.spectral_inputs import load_segments

    rows, waves, inventory = load_segments()
    cal = np.array([r["calibration"] for r in rows], dtype=bool)
    rig_of = np.array([r["rig"] for r in rows])
    metrics, meta = {}, {"dataset_pins": inventory["dataset_pins"], "rigs": {}}
    for rig in rigs:
        keep = np.flatnonzero(cal & (rig_of != rig))
        chosen = [rows[i] for i in keep]
        if any(r["rig"] == rig for r in chosen):
            raise AssertionError("target rig calibration leaked into its metric")
        metric = CruiseMetric.fit(waves[keep], chosen)
        metrics[rig] = metric
        meta["rigs"][rig] = {
            "calibration_recordings": sorted({(r["rig"], r["flight"]) for r in chosen}),
            "calibration_windows": len(chosen),
            "level_scale_rps": metric.level_scale_rps,
            "spectrum_scale": metric.spectrum_scale,
            "band_power": metric.band_power.tolist(),
        }
    return metrics, meta


def _score(
    g: Group, draws: Sequence[np.ndarray], metric: CruiseMetric
) -> tuple[str, float | None, dict]:
    """Origin-averaged ``score_phase`` dict; ``status`` flags inconsistent or
    non-finite output instead of dropping the row."""
    from experiments.rps_traj.schedule_score import score_phase

    outs = [
        score_phase(g.phase, o, r, d, g.hover, metric)
        for o, r, d in zip(g.obs, g.refs, draws, strict=True)
    ]
    keys = set(outs[0])
    if any(set(o) != keys for o in outs) or "score" not in keys:
        return f"failed: inconsistent score keys {sorted(keys)}", None, {}
    mean = {k: float(np.mean([o[k] for o in outs])) for k in sorted(keys)}
    bad = sorted(k for k, v in mean.items() if not np.isfinite(v))
    if bad:
        return f"failed: non-finite {bad}", None, {k: v for k, v in mean.items() if k not in bad}
    score = mean.pop("score")
    if len(outs) > 1:
        mean["score_per_origin"] = [float(o["score"]) for o in outs]  # type: ignore[assignment]
    return "ok", score, mean


def _row(
    kind: str,
    arm: str,
    g: Group,
    n_draws: int,
    status: str,
    score: float | None,
    diag: dict,
    seed: list[int] | None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "arm": arm,
        "rig": g.rig,
        "fold": g.fold,
        "flight": g.flight,
        "phase": g.phase,
        "n_draws": n_draws,
        "n_refs": int(len(g.refs[0])),
        "train_hover_rps": g.hover,
        "support": g.support,
        "status": status,
        "score": score,
        "diagnostics": diag,
        "seed": seed,
        **extra,
    }


def _control(name: str, refs: np.ndarray, hover: float, rng: np.random.Generator) -> np.ndarray:
    if name == "real":
        return refs.copy()
    if name == "dc_plus10":
        return refs + DC_STEP * hover
    if name == "flatten_time":
        return np.broadcast_to(refs.mean(axis=-1, keepdims=True), refs.shape).copy()
    if name == "collective":
        return np.repeat(refs.mean(axis=1, keepdims=True), refs.shape[1], axis=1)
    return augment(refs, name, rng)


def control_rows(g: Group, metric: CruiseMetric) -> list[dict[str, Any]]:
    rows = []
    for name, (where, role) in CONTROLS.items():
        if where == "cruise" and g.phase != "cruise" or where == "short" and g.phase == "cruise":
            continue
        rng, seed = _rng(g.rig, g.fold, g.flight, g.phase, name)
        draws = [_control(name, r, g.hover, rng) for r in g.refs]
        status, score, diag = _score(g, draws, metric)
        rows.append(
            _row("control", name, g, int(len(g.refs[0])), status, score, diag, seed, role=role)
        )
    return rows


def _at_rest(path: np.ndarray, events: Mapping[str, Any], hover: float) -> bool:
    landing = events.get("landing")
    tail = path[:, -pdyn._n(REST_S) :].mean(axis=0)
    return (
        landing is not None
        and landing < path.shape[1]
        and bool(tail.max() < STOPPED_FRACTION * hover)
    )


def _cut_generated(
    paths: Sequence[tuple[np.ndarray, Mapping[str, Any]]], target: Rec, g: Group
) -> tuple[list[np.ndarray] | None, str, dict[str, float]]:
    """Per origin ``(N, 4, n)`` on the target clock via the motors-on shift."""
    per_origin: list[list[np.ndarray]] = [[] for _ in g.starts]
    padded_draws, padded_samples = 0, 0
    for path, events in paths:
        on_gen, on_real = events.get("motors_on"), target.marks.motors_on
        if (on_gen is None) != (on_real is None):
            return None, "failed: commanded motors-on missing on one side", {}
        shift = 0 if on_gen is None or on_real is None else int(on_gen) - int(on_real)
        padded = False
        for k, start in enumerate(g.starts):
            lo, hi = start + shift, start + shift + g.n
            if lo < 0:
                return None, "failed: generated path starts after the window", {}
            w = np.zeros((path.shape[0], g.n))
            part = path[:, lo:hi]
            if part.shape[1] < g.n:
                if not _at_rest(path, events, g.hover):
                    return None, "failed: window past the end of a path not demonstrably landed", {}
                padded = True
                padded_samples += g.n - part.shape[1]
            w[:, : part.shape[1]] = part
            per_origin[k].append(w)
        padded_draws += padded
    diag = {"padded_draws": float(padded_draws), "padded_samples": float(padded_samples)}
    return [np.stack(d) for d in per_origin], "ok", diag


def _capture_example(example, target, groups, paths, arm):
    """Display windows only; do not change the windows used for scoring."""
    on, takeoff = target.marks.motors_on, target.marks.takeoff
    if on is None or takeoff is None or takeoff <= on:
        raise ValueError("illustrative flight needs a pre-takeoff interval")
    by_phase = {g.phase: g for g in groups}
    for phase in EXAMPLE_PHASES:
        if phase == "startup_standby":
            g = replace(by_phase["startup"], starts=(on,), n=takeoff - on)
            lo = 0
        elif phase == "cruise":
            cruise = by_phase[phase]
            g = replace(cruise, starts=(cruise.starts[len(cruise.starts) // 2],))
            lo = 0  # seconds from the start of the middle scored cruise window
        else:
            g = by_phase[phase]
            lo = g.support["lo"]
        draws, status, _ = _cut_generated(paths, target, g)
        if draws is None:
            raise ValueError(f"{target.name}/{phase}/{arm}: {status}")
        (start,) = g.starts
        example[f"{phase}__real"] = target.x[:, start : start + g.n].copy()
        example[f"{phase}__{arm}"] = draws[0]
        example[f"{phase}__lo"] = np.array(lo)


def model_rows(
    fp: FoldPlan, out_dir: Path, metric: CruiseMetric, example: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    """Every arm on every target of a fold, from the fold's cached fits; the
    :data:`EXAMPLE` flight's windows of :data:`EXAMPLE_ARMS` go to ``example``."""
    generators: dict[str, Any] = {}
    fit_errors: dict[str, str] = {}
    for arm in ARMS:
        try:
            generators[arm] = pdyn.fit_arm(arm, fp.rig, fp.fold, str(out_dir), 0, same_clock=True)
        except Exception as exc:  # reported per row, never skipped
            fit_errors[arm] = f"failed: fit: {type(exc).__name__}: {exc}"
    rows = []
    for target in fp.targets:
        groups = [g for g in fp.groups if g.flight == target.name]
        for arm in ARMS:
            rng, seed = _rng(fp.rig, fp.fold, target.name, "draws")
            status, paths = fit_errors.get(arm, "ok"), None
            if arm in generators:
                try:
                    paths = generators[arm](target.marks, N_DRAWS, rng)
                except Exception as exc:  # reported per row, never skipped
                    status = f"failed: generate: {type(exc).__name__}: {exc}"
                if status == "ok" and (paths is None or len(paths) != N_DRAWS):
                    status = f"failed: {0 if paths is None else len(paths)} of {N_DRAWS} paths"
            if status == "ok" and (fp.rig, target.name) == EXAMPLE and arm in EXAMPLE_ARMS:
                _capture_example(example, target, groups, paths, arm)
            for g in groups:
                if status != "ok":
                    rows.append(_row("model", arm, g, N_DRAWS, status, None, {}, seed))
                    continue
                assert paths is not None  # status "ok" means the arm generated its paths
                draws, cut_status, cut_diag = _cut_generated(paths, target, g)
                if draws is None:
                    rows.append(_row("model", arm, g, N_DRAWS, cut_status, None, {}, seed))
                    continue
                st, score, diag = _score(g, draws, metric)
                rows.append(_row("model", arm, g, N_DRAWS, st, score, {**diag, **cut_diag}, seed))
    return rows


def _meta(metric_meta: dict, folds: dict) -> dict[str, Any]:
    return {
        "base_seed": BASE_SEED,
        "seeds": "default_rng(SeedSequence([base_seed, crc32('|'.join(parts))])); controls: "
        "(rig, fold, flight, phase, control); models: (rig, fold, flight, 'draws') shared by all arms",
        "fs_hz": FS,
        "n_draws": N_DRAWS,
        "arms": list(ARMS),
        "controls": {k: {"phases": w, "role": r} for k, (w, r) in CONTROLS.items()},
        "dc_step_of_train_hover": DC_STEP,
        "phases": list(PHASES),
        "events": EVENTS,
        "cruise": {
            "samples": CRUISE_N,
            "origins": list(ORIGINS),
            "average": "key-wise mean over origins per flight",
        },
        "preprocessing": {
            "targets_donors": "common 100 Hz grid -> NaN linearly bridged -> 4-sample block means to 25 Hz "
            "(regime_campaign._to_flight_rate)",
            "generated": "phasevar* simulated at 25 Hz on the target's phase labels; ours sampled at the phase "
            "layer rate and block-meaned to 25 Hz; inner msvar3/hsmm3 cruise brought to 25 Hz by "
            "block means (rate > 25 Hz) or linear interpolation (rate < 25 Hz, e.g. michaels 12.5 Hz)",
            "metric_calibration": "pilot flight map windows (spectral_inputs.load_segments): resample_poly to "
            "25 Hz, no NaN bridging, 0.5 s run edges cut; differs from the block means above",
        },
        "support_policy": "short phases: intersection of the target's and every donor's window offsets around "
        "the event, real paths cut inside their records, never padded; cruise: each flight's "
        "own settled span, origins at 0/0.5/1",
        "gap_policy": "cruise span narrowed to its longest bridge-free run when >= samples, else whole span with "
        "bridged samples counted in support; short phases count bridged samples in support",
        "generation_clock": "aligned to the target only by the commanded motors-on difference; past the path end "
        "zeros only if its commanded landing is inside it and the last 0.5 s rotor mean "
        "max < STOPPED_FRACTION x train hover, else the arm row fails",
        "targets": "measured held-out flights of rig_data folds; command-only (DREGON motors_command) excluded",
        "hover": "median Landmarks.hover of the fold's measured donors (training only)",
        "metric": metric_meta,
        "folds": folds,
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    tmp.replace(path)


def run(out_dir: Path, *, controls_only: bool, rigs: Sequence[str] = rc.FLIGHT_RIGS) -> list[Path]:
    """Controls on every target, written; then (unless ``controls_only``) the
    five arms from the cached fold fits.  ``out_dir`` holds ``raw/`` caches;
    results go to ``out_dir/schedule_eval/``."""
    metrics, metric_meta = cruise_metrics(rigs)
    plans: list[FoldPlan] = []
    excluded: list[dict] = []
    for rig in rigs:
        p, e = plan(rig, out_dir, require_caches=not controls_only)
        plans += p
        excluded += e
    for fp in plans:
        cal = {tuple(x) for x in metric_meta["rigs"][fp.rig]["calibration_recordings"]}
        if any((fp.rig, t.name) in cal for t in fp.targets):
            raise AssertionError("a target flight is in its own metric's calibration")
    folds = {f"{fp.rig}/{fp.fold}": fp.meta for fp in plans}
    meta = _meta(metric_meta, folds)
    dest = out_dir / "schedule_eval"
    controls = [row for fp in plans for g in fp.groups for row in control_rows(g, metrics[fp.rig])]
    written = [dest / "schedule_controls.json"]
    _write(written[0], {"meta": meta, "exclusions": excluded, "rows": controls})
    if controls_only:
        return written
    example: dict[str, np.ndarray] = {}
    models = [row for fp in plans for row in model_rows(fp, out_dir, metrics[fp.rig], example)]
    expected = [f"{p}__{k}" for p in EXAMPLE_PHASES for k in ("real", "lo", *EXAMPLE_ARMS)]
    if EXAMPLE[0] in rigs:
        npz = dest / f"example_{EXAMPLE[0]}_{EXAMPLE[1]}.npz"
        dest.mkdir(parents=True, exist_ok=True)
        np.savez(npz, allow_pickle=True, **example)
        written.append(npz)
        meta = {
            **meta,
            "example": {
                "path": str(npz),
                "rig": EXAMPLE[0],
                "flight": EXAMPLE[1],
                "arms": list(EXAMPLE_ARMS),
                "keys": sorted(example),
                "missing": [k for k in expected if k not in example],
                "units": "rev/s at 25 Hz; lo is a sample offset from the labeled window origin",
            },
        }
    written.insert(1, dest / "schedule_models.json")
    _write(written[1], {"meta": meta, "exclusions": excluded, "rows": models})
    return written


__all__ = [
    "ARMS",
    "CONTROLS",
    "N_DRAWS",
    "ORIGINS",
    "PHASES",
    "control_rows",
    "model_rows",
    "plan",
    "run",
]
