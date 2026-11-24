"""Per-drone baselines: the historical generators refitted on one rig's training split.

Portable worker of the per-drone comparison.  It reads ONLY
``ROOT/pack/{rig}.train.npz`` (``x (4, T)`` float64 on the 100 Hz common grid,
NaN not bridged; ``offsets (k+1,)``) and ``ROOT/pack/{rig}.train.json``
(``train[i]``: one training record each; ``targets[k]``: ``{index, id, n,
events, marks, seed}`` of a validation recording WITHOUT its speeds), fits one
arm on exactly those training records and writes

* ``ROOT/runs/{rig}/{arm}.draws.npz`` - ``t{k}``: ``(32, 4, n_k)`` float32,
  sample 0 the first sample of target ``k`` at 25 Hz; ``NaN`` where the arm
  has no fitted law (see "availability");
* ``ROOT/runs/{rig}/{arm}.fit.json`` - fitted parameters, ``n_params``,
  provenance (``train_sha256``, ``train_ids``, this file's sha256) and one
  status record per target.  Written last: its presence with matching hashes
  is the completion mark, so a killed unit is simply run again.

Arms (the historical fixed-schedule arms of
:mod:`experiments.rps_traj.schedule_eval`, renamed):

* ``template`` - ``ours``: the phase layer
  (:func:`~experiments.rps_traj.transients.fit_phase_layer`, 100 Hz) around
  the two-level cruise model (``fit_gaussian`` + ``fit_levels(2)`` on the
  settled cruise at the rig's campaign rate), sampled with
  ``same_clock=True`` from the target's COARSE events only (motors-on,
  take-off, landing) and aligned by the motors-on difference, as
  ``phase_dynamics.example_flights`` does.
* ``phasevar2`` / ``phasevar4`` - one stable Gaussian VAR(p) per phase,
  labelled maximum likelihood on the training phases, simulated along the
  target's phase labels (which use the finer diagnostic landmarks: standby,
  arrival, motors-off - disclosed, as before) and clamped to ``[0, 1.25 x the
  fastest training sample]``.
* ``msvar3`` / ``hsmm3`` - ``phasevar2+msvar3`` / ``phasevar2+hsmm3``: the
  per-phase VAR(2) whose settled cruise is the cruise campaign's three-state
  Markov-switching VAR (order by BIC among 1, 2, 4) / VAR-HSMM (that order,
  regularisation by one-step MSFE on every fifth TRAINING block), placed at
  the training cruise level.

Availability.  Nothing a training split does not contain is invented:

* a target recorded airborne at both ends gets, from ``template``, the
  layer's own cruise law only (absolute level ``mu + s_c z + s_r z_4``, the
  law of :meth:`PhaseLayer.sample`, plus the shaft fluctuation sampled at
  100 Hz and block-averaged to 25 Hz) and, from the VAR arms, the cruise
  state along its all-cruise labels;
* a target that starts airborne and lands is cut from a layer draw whose
  airborne span covers the target's pre-landing part plus the settling, by
  the landing difference (no take-off enters the window);
* a target that ends airborne is drawn with its landing pushed past the end
  of the recording (no landing curve enters the window);
* a phase no training record contains (e.g. a landing when every training
  record ends airborne) makes the draws ``NaN`` from where it starts; a
  generated path that ends before the target ends is padded with rest
  (zeros) ONLY when it demonstrably landed (``schedule_eval._at_rest``),
  ``NaN`` otherwise.  Front (pre-motors-on rest) and tail padding are
  counted per target.

Costs (CPU, one thread): ``template`` - ``fit_gaussian`` (4 L-BFGS restarts
of the exact Kalman likelihood, autograd) dominates; minutes on the large
rigs, plus the IMM level fit.  ``msvar3`` - three EM fits (3 restarts, up to
200 iterations each).  ``hsmm3`` - the same three for the order, six
regularisation fits and the final fit, all explicit-duration with
``D = 5 s x rate`` (250 at 50 Hz): the slowest arm, tens of minutes to hours
on NeuroBEM-sized training sets.  ``phasevar*`` - closed form, seconds.

CLI::

    PYTHONPATH=src python -m experiments.rps_traj.per_drone_baselines \
        --root results/rps_traj/per_drone --rig michaels --arm template
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.rps_traj import phase_dynamics as pdyn
from experiments.rps_traj import regime_campaign as rc
from experiments.rps_traj import unified_experiment as ue
from experiments.rps_traj.cruise_api import NUM_ROTORS
from experiments.rps_traj.data import RATE_HZ, Flight
from experiments.rps_traj.phases import (
    PHASES,
    SETTLE_AFTER_ARRIVAL_S,
    Landmarks,
    block_mean,
    campaign_rate,
    cruise_groups,
    landmarks,
    phase_labels,
)
from experiments.rps_traj.schedule_eval import _at_rest
from experiments.rps_traj.spectral_inputs import COMMAND_SOURCE
from experiments.rps_traj.switching import (
    SwitchingCruise,
    fit_gaussian,
    fit_levels,
    training_blocks,
)
from experiments.rps_traj.transients import LANDING_PRE_S, PhaseLayer, fit_phase_layer
from experiments.rps_traj.var_models import (
    SwitchingVarModel,
    VarModel,
    _simulate_kernel,
    fit_switching_var,
    select_regularization,
)

ARMS = ("template", "phasevar2", "phasevar4", "msvar3", "hsmm3")
FS = pdyn.FS
N_DRAWS = 32
#: The fits' seed (the campaign's ``seed=0``).
FIT_SEED = 0
#: Index-only landmark fields a target may carry; speed levels never.
MARK_KEYS = (
    "fs",
    "n",
    "starts_ground",
    "ends_ground",
    "motors_on",
    "standby",
    "takeoff",
    "arrival",
    "landing",
    "motors_off",
)
LEVEL_KEYS = ("hover", "standby_level", "hover_in", "hover_out")
#: Margin (s) by which a drawn landing is pushed past a target that ends airborne.
AIRBORNE_END_MARGIN_S = 1.0


# ─── inputs ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Target:
    """One validation recording as a worker sees it: its schedule, no speeds."""

    index: int
    id: str
    n: int
    events: dict[str, int | None]
    marks: Landmarks
    seed: list[int]
    unmarked: bool


@dataclass(frozen=True)
class Pack:
    rig: str
    sha256: str
    flights: tuple[Flight, ...]
    targets: tuple[Target, ...]
    hover: float
    #: Training rows the pack marked ``coarse_initial`` (kept in every arm).
    coarse_initial: tuple[str, ...] = ()


def _int(v: Any) -> int | None:
    return None if v is None else int(v)


def _marks(
    payload: Mapping[str, Any] | None, n: int, events: Mapping[str, Any], hover: float
) -> tuple[Landmarks, bool]:
    """A target's index-only landmarks; ``hover`` is the TRAINING hover.  A
    target without marks and without events is airborne throughout."""
    if payload is None:
        if any(events.get(k) is not None for k in ("motors_on", "takeoff", "landing")):
            raise ValueError("a target with events but no marks")
        return Landmarks(fs=FS, n=n, hover=hover, starts_ground=False, ends_ground=False), True
    leaked = [k for k in LEVEL_KEYS if k in payload]
    if leaked:
        raise ValueError(f"target marks carry speed levels {leaked}")
    unknown = sorted(set(payload) - set(MARK_KEYS))
    if unknown:
        raise ValueError(f"unknown mark fields {unknown}")
    if float(payload.get("fs", FS)) != FS or int(payload.get("n", n)) != n:
        raise ValueError(f"marks at {payload.get('fs')} Hz / n={payload.get('n')}, target n={n}")
    standby = payload.get("standby")
    marks = Landmarks(
        fs=FS,
        n=n,
        hover=hover,
        starts_ground=bool(payload.get("starts_ground", payload.get("motors_on") is not None)),
        ends_ground=bool(payload.get("ends_ground", payload.get("landing") is not None)),
        motors_on=_int(payload.get("motors_on")),
        standby=None if not standby else (int(standby[0]), int(standby[1])),
        takeoff=_int(payload.get("takeoff")),
        arrival=_int(payload.get("arrival")),
        landing=_int(payload.get("landing")),
        motors_off=_int(payload.get("motors_off")),
    )
    return marks, False


def _training_hover(flights: Sequence[Flight]) -> float:
    """Median landmark hover of the training records at 25 Hz (the hover
    ``schedule_eval`` takes from a fold's donors)."""
    found = [
        m.hover
        for f in flights
        if (m := landmarks(rc._to_flight_rate(f.rps, f.fs), FS)) is not None
    ]
    if not found:
        raise ValueError("no training record has an airborne segment")
    return float(np.median(found))


def load_pack(root: Path, rig: str) -> Pack:
    """The training records and target schedules of ``rig``, hash-checked."""
    npz = root / "pack" / f"{rig}.train.npz"
    meta = json.loads((root / "pack" / f"{rig}.train.json").read_text())
    sha = ue._sha256(npz)
    if meta.get("rig") != rig:
        raise ValueError(f"pack is for rig {meta.get('rig')!r}, not {rig!r}")
    if sha != meta["train_sha256"]:
        raise ValueError(f"{npz}: sha256 {sha} != train_sha256 {meta['train_sha256']}")
    with np.load(npz) as buf:
        x = np.asarray(buf["x"], dtype=np.float64)
        offsets = np.asarray(buf["offsets"], dtype=np.int64)
    records = list(meta["train"])
    if x.shape[0] != NUM_ROTORS or offsets.size != len(records) + 1:
        raise ValueError(f"x {x.shape} / offsets {offsets.size} do not fit {len(records)} records")
    if offsets[0] != 0 or offsets[-1] != x.shape[1] or np.any(np.diff(offsets) <= 0):
        raise ValueError("offsets do not partition x")
    ids = [str(r["id"]) for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate training ids")
    flights = []
    for r, a, b in zip(records, offsets[:-1], offsets[1:]):
        if "offset" in r and int(r["offset"]) != int(a):
            raise ValueError(f"{r['id']}: offset {r['offset']} != {a}")
        if "samples_100" in r and int(r["samples_100"]) != int(b - a):
            raise ValueError(f"{r['id']}: samples_100 {r['samples_100']} != {b - a}")
        source = str(r.get("source", ""))
        if COMMAND_SOURCE in source:
            raise ValueError(f"{r['id']}: command-only recordings are excluded from every arm")
        flights.append(
            Flight(
                rig=rig,
                flight=str(r["id"]),
                fs=RATE_HZ,
                t0=float(r.get("t0_s", 0.0)),
                rps=np.ascontiguousarray(x[:, a:b]),
                source=source,
            )
        )
    hover = _training_hover(flights)
    targets = []
    for k, t in enumerate(meta["targets"]):
        if int(t.get("index", k)) != k:
            raise ValueError(f"target {k} carries index {t.get('index')}")
        n = int(t["n"])
        events = {e: _int(t.get("events", {}).get(e)) for e in ("motors_on", "takeoff", "landing")}
        marks, unmarked = _marks(t.get("marks"), n, events, hover)
        targets.append(
            Target(k, str(t["id"]), n, events, marks, [int(s) for s in t["seed"]], unmarked)
        )
    coarse = tuple(str(r["id"]) for r in records if r.get("schedule_status") == "coarse_initial")
    return Pack(rig, sha, tuple(flights), tuple(targets), hover, coarse)


# ─── template: phase layer + two-level cruise ─────────────────────────────────


def _layer_size(layer: PhaseLayer) -> int:
    """Numbers the layer stores (curves, levels, log-normal laws, ramp rate)."""
    total = 1  # takeoff_rate
    for key, value in layer.__dict__.items():
        if isinstance(value, np.ndarray):
            total += int(value.size)
        elif key in ("pre_takeoff_log_s", "airborne_log_s"):
            total += 2
    return total


def _cruise_only(cruise: SwitchingCruise, n: int, rng: np.random.Generator) -> np.ndarray:
    """``(4, n)`` at 25 Hz of the layer's airborne law: an absolute level as
    :meth:`PhaseLayer.sample` draws it plus the shaft fluctuation at 100 Hz."""
    p = cruise.params
    hover = p.mu + rng.standard_normal() * p.s_c + rng.standard_normal(NUM_ROTORS) * p.s_r
    k = int(round(RATE_HZ / FS))
    fluct = cruise.sample(n * k, rng, rate_hz=RATE_HZ, shaft_only=True)
    return rc._to_flight_rate(np.clip(hover[:, None] + fluct, 0.0, None), RATE_HZ)


def _align(x: np.ndarray, shift: int, n: int, rest_tail: bool) -> tuple[np.ndarray, int, int, int]:
    """``y[t] = x[t + shift]`` for ``t < n``; before ``x`` starts the rotors
    are at rest (zeros), past its end at rest only when ``rest_tail``, NaN
    otherwise.  ``(y, front padded, tail padded, tail missing)``."""
    y = np.full((x.shape[0], n), np.nan)
    src = np.arange(n) + int(shift)
    front, tail = src < 0, src >= x.shape[1]
    inside = ~front & ~tail
    y[:, front] = 0.0
    y[:, inside] = x[:, src[inside]]
    if rest_tail:
        y[:, tail] = 0.0
    n_tail = int(tail.sum())
    return y, int(front.sum()), n_tail if rest_tail else 0, 0 if rest_tail else n_tail


def _template_one(
    layer: PhaseLayer | None,
    cruise: SwitchingCruise,
    target: Target,
    hover: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """One draw on the target clock and its diagnostics."""
    ev, n = target.events, target.n
    on, to, ld = ev["motors_on"], ev["takeoff"], ev["landing"]
    lands = layer is not None and layer.landing_common is not None
    ld_window = None if ld is None else max(ld - int(round(LANDING_PRE_S * FS)), 0)
    scale = FS / RATE_HZ

    def events_of(draw: Mapping[str, Any]) -> dict[str, int | None]:
        return {
            k: None if draw[k] is None else int(round(draw[k] * scale))
            for k in ("motors_on", "takeoff", "landing")
        }

    diag: dict[str, Any] = {"nan_from": None}
    if on is None and to is None and (ld is None or not lands):
        # airborne at the start, and either airborne at the end or landing
        # with no training landing: the cruise law, NaN from the landing window
        diag["case"] = "airborne" if ld is None else "airborne_start_landing_unsupported"
        y = _cruise_only(cruise, n, rng)
        if ld_window is not None:
            y[:, ld_window:] = np.nan
            diag["nan_from"] = ld_window
        return y, {**diag, "front_pad": 0, "tail_pad": 0, "tail_missing": 0}
    if layer is None:
        diag["case"] = "ground_start_unsupported"
        return np.full((NUM_ROTORS, n), np.nan), {
            **diag,
            "nan_from": 0,
            "front_pad": 0,
            "tail_pad": 0,
            "tail_missing": 0,
        }

    if on is None:  # starts airborne and lands: cut by the landing difference
        assert ld is not None and to is None
        diag["case"] = "airborne_start_landing"
        air = ld / FS + SETTLE_AFTER_ARRIVAL_S + AIRBORNE_END_MARGIN_S
        draw = layer.sample(cruise, rng, airborne_s=air)
        x, gev = rc._to_flight_rate(draw["rps"], layer.fs), events_of(draw)
        assert gev["landing"] is not None
        shift = gev["landing"] - ld
        arrival = int(round(draw["arrival"] * scale))
        if shift < arrival + int(round(SETTLE_AFTER_ARRIVAL_S * FS)):
            raise AssertionError("an airborne-start window reaches into the drawn take-off")
        y, front, tail, missing = _align(x, shift, n, _at_rest(x, gev, hover))
        diag["landing_err"] = gev["landing"] - shift - ld
    else:
        if to is None:
            raise ValueError(f"target {target.index}: motors-on without take-off")
        diag["case"] = "ground_start"
        pre = (to - on) / FS
        supported_landing = ld is not None and lands
        if supported_landing:
            assert ld is not None
            end_s = (ld - to) / FS
        else:  # ends airborne, or lands with no training landing: land after the end
            end_s = (n - to) / FS + LANDING_PRE_S + AIRBORNE_END_MARGIN_S
            diag["case"] = (
                "ground_start_airborne_end" if ld is None else "ground_start_landing_unsupported"
            )
        state = int(rng.integers(2**62))
        first = layer.sample(
            cruise, np.random.default_rng(state), airborne_s=max(end_s, 1.0), pre_takeoff_s=pre
        )
        ramp = (first["arrival"] - first["takeoff"]) / layer.fs
        draw = layer.sample(
            cruise,
            np.random.default_rng(state),
            airborne_s=max(end_s - ramp, 1.0),
            pre_takeoff_s=pre,
        )
        x, gev = rc._to_flight_rate(draw["rps"], layer.fs), events_of(draw)
        assert gev["motors_on"] is not None and gev["takeoff"] is not None
        shift = gev["motors_on"] - on
        y, front, tail, missing = _align(x, shift, n, _at_rest(x, gev, hover))
        diag["takeoff_err"] = gev["takeoff"] - shift - to
        if supported_landing:
            assert gev["landing"] is not None and ld is not None
            diag["landing_err"] = gev["landing"] - shift - ld
        elif ld_window is not None:
            y[:, ld_window:] = np.nan
            diag["nan_from"] = ld_window
    if missing:
        first_missing = n - missing
        diag["nan_from"] = (
            first_missing if diag["nan_from"] is None else min(diag["nan_from"], first_missing)
        )
    return y, {**diag, "front_pad": front, "tail_pad": tail, "tail_missing": missing}


def fit_template(flights: Sequence[Flight], rig: str) -> tuple[dict[str, Any], Any]:
    """The template's fitted parts (JSON) and its generator."""
    rate = campaign_rate(rig)
    groups = cruise_groups(flights, rate)
    params, info = fit_gaussian(groups, rate, seed=FIT_SEED)
    cruise = fit_levels(params, training_blocks(groups, rate), rate, 2)
    layer: PhaseLayer | None = None
    layer_note = None
    try:
        layer = fit_phase_layer(flights)
    except ValueError as exc:  # no training take-off: the cruise law alone is supported
        layer_note = str(exc)
    n_layer = 0 if layer is None else _layer_size(layer)
    fitted = {
        "cruise_rate_hz": rate,
        "cruise_groups": sorted(groups),
        "cruise": cruise.to_dict(),
        "gaussian_fit": info,
        "layer": None if layer is None else layer.to_dict(),
        "layer_note": layer_note,
        "n_params": int(cruise.n_params + 9 + n_layer),
        "n_params_detail": {
            "cruise": int(cruise.n_params),
            "hover_population": 9,
            "layer_values": n_layer,
        },
    }

    def generate(
        target: Target, hover: float, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict[str, Any]]:
        ys, diags = zip(*(_template_one(layer, cruise, target, hover, rng) for _ in range(N_DRAWS)))
        return np.stack(ys), _summary(diags)

    return fitted, generate


def _summary(diags: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cases = sorted({str(d["case"]) for d in diags})
    nan_from = [d["nan_from"] for d in diags if d["nan_from"] is not None]
    out: dict[str, Any] = {
        "case": cases[0] if len(cases) == 1 else cases,
        "nan_from_min": min(nan_from) if nan_from else None,
        "draws_with_nan": len(nan_from),
        "front_pad_samples": int(sum(d["front_pad"] for d in diags)),
        "front_pad_draws": int(sum(d["front_pad"] > 0 for d in diags)),
        "tail_pad_samples": int(sum(d["tail_pad"] for d in diags)),
        "tail_pad_draws": int(sum(d["tail_pad"] > 0 for d in diags)),
        "tail_missing_draws": int(sum(d["tail_missing"] > 0 for d in diags)),
    }
    for key in ("takeoff_err", "landing_err"):
        errs = [abs(int(d[key])) for d in diags if key in d]
        if errs:
            out[f"max_abs_{key}"] = max(errs)
    return out


# ─── per-phase VARs, optionally with a switching cruise ──────────────────────


def _hsmm3(blocks: list[np.ndarray], rate: float) -> tuple[SwitchingVarModel, dict[str, Any]]:
    """``regime_campaign._fit_hsmm`` for three states only."""
    m = 3
    order = rc._switch_order(blocks, rate, m, FIT_SEED)[0].order
    kw: dict[str, Any] = dict(
        n_states=m, order=order, durations="explicit", max_dur_s=rc.MAX_DUR_S, seed=FIT_SEED
    )
    val = blocks[4::5]
    inner = [b for i, b in enumerate(blocks) if i % 5 != 4]
    lam_a, lam_sigma, table = 0.0, 0.0, []
    if val and inner:
        rows = sum(max(b.shape[1] - order, 0) for b in inner)
        picked, table = select_regularization(
            inner,
            val,
            rate,
            grid_a=[c * rows for c in rc.GRID_A_PER_ROW],
            grid_sigma=rc.GRID_SIGMA,
            n_init=1,
            **kw,
        )
        lam_a = float(picked.lam_a) * sum(max(b.shape[1] - order, 0) for b in blocks) / rows
        lam_sigma = float(picked.lam_sigma)
    model = fit_switching_var(blocks, rate, lam_a=lam_a, lam_sigma=lam_sigma, name=f"hsmm{m}", **kw)
    return model, {"lam_a": lam_a, "lam_sigma": lam_sigma, "grid": table, **rc._switch_note(model)}


def _settled(
    x: np.ndarray,
    lab: np.ndarray,
    model: SwitchingVarModel,
    level: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Transcribed from ``phase_dynamics.fit_arm``: the settled cruise (from
    ``SETTLE_S`` after its start to its end) cross-faded over a second into
    the training cruise level plus the switching VAR's fluctuation."""
    idx = np.flatnonzero(lab == pdyn.CRUISE)
    if idx.size < pdyn._n(pdyn.SETTLE_S + 4.0):
        return x
    a0, a1 = int(idx[0]) + pdyn._n(pdyn.SETTLE_S), int(idx[-1]) + 1
    rate = float(model.rate_hz)
    draw = model.sample(int(np.ceil((a1 - a0) * rate / FS)) + 2, rng)
    if rate > FS:
        draw = block_mean(draw, rate, FS)
    elif rate < FS:
        t = np.arange(a1 - a0) / FS
        draw = np.stack([np.interp(t, np.arange(row.size) / rate, row) for row in draw])
    new = level[:, None] + draw[:, : a1 - a0]
    k = np.arange(a1 - a0)
    w = np.clip(np.minimum(k + 1, (a1 - a0) - k) / pdyn._n(1.0), 0.0, 1.0)
    out = x.copy()
    out[:, a0:a1] = (1.0 - w) * x[:, a0:a1] + w * new
    return np.clip(out, 0.0, None)


def fit_phasevar(flights: Sequence[Flight], rig: str, arm: str) -> tuple[dict[str, Any], Any]:
    """``phasevarP`` or ``phasevar2`` + a three-state switching cruise.

    The per-phase VAR is the labelled maximum-likelihood fit of
    ``phase_dynamics.fit_arm`` restricted to the phases the training records
    contain: each state's estimate uses only its own samples and the common
    scale, so the states fitted are those of the six-state call, which is
    what it is when every phase is present."""
    order = 2 if arm in ("msvar3", "hsmm3") else int(arm.removeprefix("phasevar"))
    seqs = [rc._to_flight_rate(f.rps, f.fs) for f in flights]
    every = [(f.flight, s, landmarks(s, FS)) for f, s in zip(flights, seqs)]
    marked = [
        (
            name,
            s,
            m if m is not None else Landmarks(FS, s.shape[1], float(np.mean(s)), False, False),
        )
        for name, s, m in every
    ]
    labels = [phase_labels(m) for _, _, m in marked]
    counts = np.zeros(len(PHASES), dtype=np.int64)
    for lab in labels:
        np.add.at(counts, lab[order:], 1)
    present = [j for j in range(len(PHASES)) if counts[j] > 0]
    code = np.full(len(PHASES), -1, dtype=np.int64)
    code[present] = np.arange(len(present))
    model = fit_switching_var(
        [s for _, s, _ in marked],
        FS,
        n_states=len(present),
        order=order,
        intercept=True,
        sigma_floor=rc.FLIGHT_SIGMA_FLOOR,
        n_init=1,
        max_iter=0,  # the states are known: the labelled fit is the estimate
        seed=FIT_SEED,
        init_labels=[code[lab] for lab in labels],
        stable=True,
    )
    cap = rc.FLIGHT_CAP * max(float(s.max()) for s in seqs)
    per_state = NUM_ROTORS + NUM_ROTORS * NUM_ROTORS * order + NUM_ROTORS * (NUM_ROTORS + 1) // 2
    fitted: dict[str, Any] = {
        "states": [PHASES[j] for j in present],
        "train_samples_per_phase": {PHASES[j]: int(counts[j]) for j in range(len(PHASES))},
        "unlabelled_train_ids": [name for name, _, m in every if m is None],
        "cap": cap,
        "phasevar": model.to_dict(),
        "n_params_detail": {"phasevar": per_state * len(present), "training_speed_cap": 1},
    }
    cruise_model: SwitchingVarModel | None = None
    level = np.zeros(NUM_ROTORS)
    if arm in ("msvar3", "hsmm3"):
        rate = campaign_rate(rig)
        groups = cruise_groups(flights, rate)
        blocks = training_blocks(groups, rate)
        if arm == "msvar3":
            cruise_model, note = rc._switch_order(blocks, rate, 3, FIT_SEED)
            note = {**note, **rc._switch_note(cruise_model)}
        else:
            cruise_model, note = _hsmm3(blocks, rate)
        level = np.concatenate(
            [s[:, lab == pdyn.CRUISE] for (_, s, _), lab in zip(marked, labels)], axis=1
        ).mean(axis=1)
        fitted.update(
            cruise_rate_hz=rate,
            cruise_groups=sorted(groups),
            cruise=cruise_model.to_dict(),
            cruise_note=note,
            cruise_level=level.tolist(),
        )
        fitted["n_params_detail"].update(cruise=int(cruise_model.n_params), cruise_level=NUM_ROTORS)
    fitted["n_params"] = int(sum(fitted["n_params_detail"].values()))
    initial = None
    if int(code[pdyn.CRUISE]) >= 0:
        j = int(code[pdyn.CRUISE])
        # A recording which begins during flight has no preceding rest.
        # Use the fitted stationary joint lag law, not the target's first
        # speeds and not a zero initialization that spuriously launches it.
        initial = VarModel(
            "airborne entry", FS, model.coefs[j], model.mu[j], model.sigma[j], True
        )._stationary_root

    def generate(
        target: Target, hover: float, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict[str, Any]]:
        lab = phase_labels(target.marks)
        bad = np.flatnonzero(code[lab] < 0)
        stop = int(bad[0]) if bad.size else target.n
        eps = rng.standard_normal((N_DRAWS, target.n, NUM_ROTORS))
        lags = np.zeros((N_DRAWS, model.order, NUM_ROTORS))
        if stop and lab[0] == pdyn.CRUISE and initial is not None:
            mean, root = initial
            lags[:] = (mean + rng.standard_normal((N_DRAWS, len(mean))) @ root.T).reshape(
                lags.shape
            )
        flights_ = np.full((N_DRAWS, NUM_ROTORS, target.n), np.nan)
        if stop > 0:
            paths = _simulate_kernel(
                np.ascontiguousarray(np.tile(code[lab[:stop]], (N_DRAWS, 1))),
                model.mu,
                model.coefs,
                model._chol,
                np.ascontiguousarray(eps[:, :stop]),
                lags,
                0.0,
                cap,
            )
            flights_[:, :, :stop] = paths.transpose(0, 2, 1)
        if cruise_model is not None:
            for i in range(N_DRAWS):
                flights_[i] = _settled(flights_[i], lab, cruise_model, level, rng)
                flights_[i][:, stop:] = np.nan
        missing = sorted({PHASES[j] for j in lab[stop:] if code[j] < 0})
        return flights_, {
            "case": "labels",
            "nan_from_min": None if stop == target.n else stop,
            "unsupported_phases": missing,
            "draws_with_nan": 0 if stop == target.n else N_DRAWS,
            "front_pad_samples": 0,
            "tail_pad_samples": 0,
        }

    return fitted, generate


# ─── the worker ───────────────────────────────────────────────────────────────


def run_paths(root: Path, rig: str, arm: str) -> dict[str, Path]:
    run = root / "runs" / rig
    return {"fit": run / f"{arm}.fit.json", "draws": run / f"{arm}.draws.npz"}


def _source_sha() -> str:
    return ue._sha256(Path(__file__))


def done(root: Path, rig: str, arm: str, train_sha256: str) -> bool:
    """A finished unit for this training split and this code."""
    paths = run_paths(root, rig, arm)
    if not (paths["fit"].exists() and paths["draws"].exists()):
        return False
    fit_ = json.loads(paths["fit"].read_text())
    return (
        fit_.get("train_sha256") == train_sha256
        and fit_.get("arm") == arm
        and fit_.get("source_sha256") == _source_sha()
        and fit_.get("draws_sha256") == ue._sha256(paths["draws"])
    )


def fit(root: Path, rig: str, arm: str, *, force: bool = False, threads: int = 1) -> Path:
    """Fit ``arm`` on ``rig``'s training records, draw every target, write both
    files; a unit already done for this split and code is left alone."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; one of {ARMS}")
    root = Path(root)
    rc._threads(threads)
    pack = load_pack(root, rig)
    paths = run_paths(root, rig, arm)
    if not force and done(root, rig, arm, pack.sha256):
        return paths["fit"]
    t0 = time.perf_counter()
    fitted, generate = (
        fit_template(pack.flights, rig)
        if arm == "template"
        else fit_phasevar(pack.flights, rig, arm)
    )
    fit_s = time.perf_counter() - t0
    draws: dict[str, np.ndarray] = {}
    status = []
    for target in pack.targets:
        rng = np.random.default_rng(np.random.SeedSequence(target.seed))
        y, diag = generate(target, pack.hover, rng)
        assert y.shape == (N_DRAWS, NUM_ROTORS, target.n)
        draws[f"t{target.index}"] = y.astype(np.float32)
        status.append(
            {
                "index": target.index,
                "id": target.id,
                "n": target.n,
                "status": "ok" if diag.get("nan_from_min") is None else "partial",
                "unmarked_airborne": target.unmarked,
                **diag,
            }
        )
    ue._write_npz(paths["draws"], draws)
    payload = {
        "rig": rig,
        "arm": arm,
        "train_sha256": pack.sha256,
        "train_ids": [f.flight for f in pack.flights],
        "coarse_initial_train_ids": list(pack.coarse_initial),
        "n_params": fitted["n_params"],
        "fit_seed": FIT_SEED,
        "n_draws": N_DRAWS,
        "fs_hz": FS,
        "training_hover": pack.hover,
        "fit_s": fit_s,
        "gen_s": time.perf_counter() - t0 - fit_s,
        "source_sha256": _source_sha(),
        "draws_sha256": ue._sha256(paths["draws"]),
        "params": fitted,
        "targets": status,
    }
    ue._write_json(paths["fit"], payload, allow_nan=True)
    return paths["fit"]


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--rig", required=True)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    print(fit(args.root, args.rig, args.arm, force=args.force, threads=args.threads))


if __name__ == "__main__":
    main()


__all__ = [
    "ARMS",
    "N_DRAWS",
    "Pack",
    "Target",
    "done",
    "fit",
    "fit_phasevar",
    "fit_template",
    "load_pack",
    "main",
]
