"""The flight-regime campaign: its units of work, scoring and aggregation.

One unit is one (rig, fold, family) fit-and-score job writing one JSON
(``utils.gridrun``); the thin CLI is ``scripts/rps_traj_regimes.py`` and the
write-up ``docs/experiments/rps-flight-regimes.md``.

Two comparisons, both on HELD-OUT flights:

* **cruise** (all seven rigs) — the fluctuation of settled cruise.  Arms: the
  structured model with one, two and three noise levels (``gauss``,
  ``switch2``, ``switch3``; :mod:`~experiments.rps_traj.switching`) against
  the plain VAR, the Markov-switching VAR and the regularised VAR-HSMM
  (:mod:`~experiments.rps_traj.var_models`).  Scores: the one-step log
  predictive density, the energy score and CRPS of sampled forecasts, and
  distances between generated and held-out blocks
  (:mod:`~experiments.rps_traj.scores`).
* **flight** (the rigs recorded ground to ground) — whole generated flights.
  Arms: the fitted phase layer (:mod:`~experiments.rps_traj.transients`)
  against the hand-set scaffold it replaces and against the switching VARs
  fitted to whole flights.  Scores: the CRPS of each whole-flight read-out
  (:func:`~experiments.rps_traj.phases.readouts`) and Wasserstein distances
  of the speed and increment distributions.

Fold ``"all"`` fits on every flight and scores nothing: those are the
per-drone parameter sets.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from experiments.rps_traj import model as M
from experiments.rps_traj.cruise_api import CruiseModel, burn_samples
from experiments.rps_traj.data import RATE_HZ, Flight
from experiments.rps_traj.phases import (
    PHASES,
    READOUTS,
    block_mean,
    bridge_nans,
    campaign_rate,
    cruise_groups,
    folds,
    landmarks,
    load_flights,
    phase_labels,
    readouts,
)
from experiments.rps_traj.scores import crps, forecast_scores, sample_scores
from experiments.rps_traj.switching import (
    SwitchingCruise,
    fit_gaussian,
    fit_levels,
    training_blocks,
)
from experiments.rps_traj.transients import PhaseLayer, fit_phase_layer, replay_error
from experiments.rps_traj.var_models import (
    SwitchingVarModel,
    fit_switching_var,
    fit_var,
    select_regularization,
)
from utils.gridrun import Unit, unit_path

#: The campaign's rigs, the ones with whole flights, and the model families.
RIGS = (
    "michaels",
    "dregon",
    "vid_m100",
    "blackbird_quad",
    "nanobench_cf21b",
    "neurobem_quad",
    "pitcn_quad",
)
FLIGHT_RIGS = ("michaels", "vid_m100", "dregon")
FAMILIES = ("ours", "var", "msvar", "hsmm")
FLIGHT_ARMS = ("ours", "scaffold", "msvar", "hsmm")

#: Forecast horizons (s), origins every ``STRIDE_S`` and at most
#: ``MAX_ORIGINS`` a fold, ``N_PATHS`` sampled paths an origin.
HORIZONS_S = (0.2, 1.0, 2.0)
STRIDE_S = 2.0
MAX_ORIGINS = 200
N_PATHS = 100

#: Generated material for the sample distances: this many times the held-out
#: duration, at most :data:`GEN_MAX_SAMPLES`.
GEN_REPEATS = 4
GEN_MAX_SAMPLES = 200_000

#: Baseline settings.  VAR order by BIC up to one second of lags (24 at most);
#: switching VAR order by BIC among these; state durations up to this long;
#: the regularisation grid of the VAR-HSMM, chosen by one-step MSFE on every
#: fifth training block (Xu and Liu's criterion).  Their LASSO weight
#: multiplies an UNNORMALISED sum of squares, so a weight means nothing
#: without the sample count: the grid here is per training row (their grid of
#: 1e-4 to 1 on 303 days of 50 stocks is 1e-8 to 1e-4 per squared residual),
#: and 0 is added to both grids because four channels meet 1e4-1e5 samples.
VAR_P_MAX = 24
SWITCH_ORDERS = (1, 2, 4)
SWITCH_STATES = (2, 3)
MAX_DUR_S = 5.0
GRID_A_PER_ROW = (0.0, 1e-6, 1e-4)
GRID_SIGMA = (0.0, 0.1)

#: A generated path further than this many pooled standard deviations of the
#: held-out blocks from zero has left the data's range for good: a switching
#: VAR may have an explosive state, and its sample scores are then reported
#: as missing instead of as numbers.
DIVERGED_STD = 1e3

#: Whole flights: the common grid generated flights are compared on, how many
#: are generated, and the switching VARs fitted to whole flights: one state
#: per phase, order 2 so that a state can ramp, a covariance floor so that a
#: motionless phase does not collapse one, and EM STARTED AT THE PHASES of the
#: training flights (:func:`~experiments.rps_traj.phases.phase_labels`).  That
#: start is the most favourable one for a generator of whole flights: from a
#: variance-based labelling, likelihood prefers states that flicker (dwell
#: 0.04-2 s on MD2) and 2-8 % of the draws were one valid flight.  The dwell
#: law spans a whole flight, which makes one EM pass cost seconds.
FLIGHT_RATE_HZ = 25.0
N_FLIGHTS_GEN = 64
FLIGHT_ORDER = 2
FLIGHT_SIGMA_FLOOR = 1e-6
FLIGHT_MAX_ITER = 100


# ─── data ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RigData:
    """One rig, as every unit sees it."""

    rig: str
    rate_hz: float
    flights: tuple[Flight, ...]
    groups: dict[str, list[np.ndarray]]
    held_out: tuple[tuple[str, ...], ...]

    def split(self, fold: str) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
        """``(train, test)`` cruise groups of a fold (``"all"``: no test)."""
        held = set() if fold == "all" else set(self.held_out[int(fold)])
        train = {g: s for g, s in self.groups.items() if g not in held}
        return train, {g: self.groups[g] for g in sorted(held)}


@cache
def rig_data(rig: str) -> RigData:
    flights = tuple(load_flights(rig))
    rate = campaign_rate(rig)
    groups = cruise_groups(flights, rate)
    source = {f.flight: f.source for f in flights}
    names = list(groups)
    strata = None
    if rig == "dregon":
        strata = ["measured" if "measured" in source[g] else "command" for g in names]
    held = tuple(tuple(h) for h in folds(names, strata))
    return RigData(rig=rig, rate_hz=rate, flights=flights, groups=groups, held_out=held)


def fold_ids(rig: str) -> list[str]:
    """``["0", .., "k-1"]`` of a rig (loads the rig)."""
    return [str(i) for i in range(len(rig_data(rig).held_out))]


# ─── fitting the cruise families ──────────────────────────────────────────────

Fitted = dict[str, tuple[CruiseModel, float, dict[str, Any]]]


def _fit_ours(
    train: Mapping[str, Sequence[np.ndarray]], blocks: list[np.ndarray], rate: float, seed: int
) -> Fitted:
    t0 = time.perf_counter()
    params, info = fit_gaussian(train, rate, seed=seed)
    gauss = fit_levels(params, blocks, rate, 1)
    t1 = time.perf_counter()
    two = fit_levels(params, blocks, rate, 2)
    t2 = time.perf_counter()
    three = fit_levels(params, blocks, rate, 3, warm=two)
    t3 = time.perf_counter()
    return {
        "gauss": (gauss, t1 - t0, info),
        "switch2": (two, t2 - t1, dict(two.info)),
        "switch3": (three, t3 - t2, dict(three.info)),
    }


def _fit_var(
    train: Mapping[str, Sequence[np.ndarray]], blocks: list[np.ndarray], rate: float, seed: int
) -> Fitted:
    t0 = time.perf_counter()
    model = fit_var(blocks, rate, p_max=min(burn_samples(rate), VAR_P_MAX))
    note = {"order": model.order, "shrunk": bool(model.shrunk), "radius": model.spectral_radius}
    return {"var": (model, time.perf_counter() - t0, note)}


def _bic(model: SwitchingVarModel, n_scored: int) -> float:
    return -2.0 * float(model.loglik_) + model.n_params * float(np.log(max(n_scored, 2)))


def _switch_order(
    blocks: list[np.ndarray], rate: float, m: int, seed: int
) -> tuple[SwitchingVarModel, dict[str, Any]]:
    """The Markov-switching VAR with ``m`` states at its BIC order."""
    best: tuple[float, SwitchingVarModel] | None = None
    table = {}
    for p in SWITCH_ORDERS:
        model = fit_switching_var(
            blocks, rate, n_states=m, order=p, durations="geometric", seed=seed, name=f"msvar{m}"
        )
        n_scored = int(sum(max(b.shape[1] - p, 0) for b in blocks))
        table[str(p)] = _bic(model, n_scored)
        if best is None or table[str(p)] < best[0]:
            best = (table[str(p)], model)
    assert best is not None
    return best[1], {"bic": table, "order": best[1].order}


def _switch_note(model: SwitchingVarModel) -> dict[str, Any]:
    return {
        "order": model.order,
        "converged": bool(model.converged_),
        "n_iter": int(model.n_iter_),
        "occupancy": [float(v) for v in model.stationary_occupancy],
        "dwell_s": [float(v) for v in model.mean_dwell_s],
    }


def _fit_msvar(
    train: Mapping[str, Sequence[np.ndarray]], blocks: list[np.ndarray], rate: float, seed: int
) -> Fitted:
    out: Fitted = {}
    for m in SWITCH_STATES:
        t0 = time.perf_counter()
        model, note = _switch_order(blocks, rate, m, seed)
        out[f"msvar{m}"] = (model, time.perf_counter() - t0, {**note, **_switch_note(model)})
    return out


def _fit_hsmm(
    train: Mapping[str, Sequence[np.ndarray]], blocks: list[np.ndarray], rate: float, seed: int
) -> Fitted:
    out: Fitted = {}
    val = blocks[4::5]
    inner = [b for i, b in enumerate(blocks) if i % 5 != 4]
    for m in SWITCH_STATES:
        t0 = time.perf_counter()
        order = _switch_order(blocks, rate, m, seed)[0].order
        kw: dict[str, Any] = dict(
            n_states=m, order=order, durations="explicit", max_dur_s=MAX_DUR_S, seed=seed
        )
        lam_a, lam_sigma, table = 0.0, 0.0, []
        if val and inner:
            rows = sum(max(b.shape[1] - order, 0) for b in inner)
            picked, table = select_regularization(
                inner,
                val,
                rate,
                grid_a=[c * rows for c in GRID_A_PER_ROW],
                grid_sigma=GRID_SIGMA,
                n_init=1,
                **kw,
            )
            lam_a = float(picked.lam_a) * sum(max(b.shape[1] - order, 0) for b in blocks) / rows
            lam_sigma = float(picked.lam_sigma)
        model = fit_switching_var(
            blocks, rate, lam_a=lam_a, lam_sigma=lam_sigma, name=f"hsmm{m}", **kw
        )
        note = {"lam_a": lam_a, "lam_sigma": lam_sigma, "grid": table, **_switch_note(model)}
        out[f"hsmm{m}"] = (model, time.perf_counter() - t0, note)
    return out


_FIT: dict[str, Callable[..., Fitted]] = {
    "ours": _fit_ours,
    "var": _fit_var,
    "msvar": _fit_msvar,
    "hsmm": _fit_hsmm,
}


# ─── scoring a cruise model ───────────────────────────────────────────────────


def mean_logpdf(model: CruiseModel, blocks: Sequence[np.ndarray], burn: int) -> tuple[float, int]:
    """``(mean one-step log predictive density, scored samples)`` of blocks."""
    total, n = 0.0, 0
    for block in blocks:
        if block.shape[1] > burn:
            lp = model.logpdf(block, burn)
            total += float(lp.sum())
            n += int(lp.size)
    return (total / n if n else float("nan")), n


def generated_blocks(
    model: CruiseModel, like: Sequence[np.ndarray], rng: np.random.Generator
) -> list[np.ndarray]:
    """Generated blocks with the lengths of ``like``, repeated, each demeaned:
    one long stationary draw cut the way the real segments are."""
    lengths = [b.shape[1] for b in like]
    total = int(sum(lengths))
    repeats = int(np.clip(GEN_MAX_SAMPLES // max(total, 1), 1, GEN_REPEATS))
    path = model.sample(total * repeats, rng)
    out, start = [], 0
    for _ in range(repeats):
        for n in lengths:
            block = path[:, start : start + n]
            out.append(np.ascontiguousarray(block - block.mean(axis=1, keepdims=True)))
            start += n
    return out


def score_cruise(
    model: CruiseModel, blocks: Sequence[np.ndarray], rate: float, seed: int
) -> dict[str, Any]:
    """Every held-out score of one cruise model.  A model whose sampler leaves
    the data's range (:data:`DIVERGED_STD`) gets no sample or forecast score."""
    burn = burn_samples(rate)
    logpdf, n = mean_logpdf(model, blocks, burn)
    out: dict[str, Any] = {"logpdf": logpdf, "n_scored": n, "diverged": False}
    rng = np.random.default_rng(seed + 1)
    limit = DIVERGED_STD * float(np.sqrt(np.mean([np.mean(b * b) for b in blocks])))
    gen = generated_blocks(model, blocks, rng)
    if not all(np.isfinite(g).all() and np.abs(g).max() < limit for g in gen):
        out["diverged"] = True
        return out
    out["sample"] = sample_scores(list(blocks), gen, rate, rng=np.random.default_rng(seed + 2))
    with np.errstate(all="ignore"):
        forecast = forecast_scores(
            model,
            blocks,
            horizons_s=HORIZONS_S,
            stride_s=STRIDE_S,
            n_paths=N_PATHS,
            rng=rng,
            max_origins=MAX_ORIGINS,
        )
    out["forecast"] = forecast
    return out


def run_cruise(unit: Unit) -> dict[str, Any]:
    """One cruise unit: fit a family on the training flights of a fold, score
    every arm of it on the held-out ones."""
    p = unit.params
    rig, fold, family, seed = str(p["rig"]), str(p["fold"]), str(p["family"]), int(p.get("seed", 0))
    _threads(int(p.get("threads", 1)))
    data = rig_data(rig)
    rate = data.rate_hz
    train, test = data.split(fold)
    train_blocks, test_blocks = training_blocks(train, rate), training_blocks(test, rate)
    burn = burn_samples(rate)
    out: dict[str, Any] = {
        "kind": "cruise",
        "rig": rig,
        "fold": fold,
        "family": family,
        "rate_hz": rate,
        "held_out": sorted(test),
        "n_train_blocks": len(train_blocks),
        "n_test_blocks": len(test_blocks),
        "train_s": float(sum(b.shape[1] for b in train_blocks) / rate),
        "test_s": float(sum(b.shape[1] for b in test_blocks) / rate),
        "arms": {},
    }
    for name, (model, fit_s, note) in _FIT[family](train, train_blocks, rate, seed).items():
        t0 = time.perf_counter()
        arm: dict[str, Any] = {
            "n_params": int(model.n_params),
            "fit_s": float(fit_s),
            "note": note,
            "train_logpdf": mean_logpdf(model, train_blocks, burn)[0],
            "params": model.to_dict(),
        }
        if test_blocks:
            arm["scores"] = score_cruise(model, test_blocks, rate, seed)
        arm["score_s"] = time.perf_counter() - t0
        out["arms"][name] = arm
    return out


def _threads(n: int) -> None:
    """Threads of the numerical libraries inside one unit."""
    import numba
    import torch

    with contextlib.suppress(ValueError):  # more than the pool was launched with: keep its size
        numba.set_num_threads(max(1, int(n)))
    torch.set_num_threads(max(1, int(n)))


# ─── whole flights ────────────────────────────────────────────────────────────


def _to_flight_rate(rps: np.ndarray, fs: float) -> np.ndarray:
    """A whole flight on the comparison grid (gaps bridged, block means)."""
    return block_mean(bridge_nans(np.asarray(rps, dtype=np.float64)), fs, FLIGHT_RATE_HZ)


def first_pass(rps: np.ndarray, fs: float, hover: float) -> np.ndarray:
    """A free-running sample cut after its first flight: one second past the
    first time the rotor mean stays under a tenth of ``hover`` for a second
    after having been above half of it.  Unchanged when that never happens."""
    m = rps.mean(axis=0)
    up = np.flatnonzero(m > 0.5 * hover)
    if up.size == 0:
        return rps
    low = m < 0.1 * hover
    run = int(round(fs))
    count = 0
    for i in range(int(up[0]), m.size):
        count = count + 1 if low[i] else 0
        if count >= run:
            return rps[:, : min(i + run, m.size)]
    return rps


def _cruise_for_flights(out_dir: str | None, rig: str, fold: str, seed: int) -> SwitchingCruise:
    """The two-level cruise model of a fold: read from that fold's cruise unit
    when it has run, fitted here otherwise."""
    if out_dir is not None:
        path = unit_path(out_dir, f"cruise__{rig}__{fold}__ours")
        if path.exists():
            import json

            return SwitchingCruise.from_dict(
                json.loads(path.read_text())["arms"]["switch2"]["params"]
            )
    data = rig_data(rig)
    train, _ = data.split(fold)
    params, _ = fit_gaussian(train, data.rate_hz, seed=seed)
    return fit_levels(params, training_blocks(train, data.rate_hz), data.rate_hz, 2)


def _scaffold_flight(
    cruise: SwitchingCruise, train: Sequence[Flight], rng: np.random.Generator
) -> np.ndarray:
    """One flight from the scaffold in use today
    (:func:`data_processing.trajectory_model.wrap_airborne`: raised-cosine
    ramps, hand-set duration ranges, the stored idle rule), around the same
    cruise process."""
    from data_processing.rps_synthesis import FlightPhaseRanges
    from data_processing.trajectory_model import wrap_airborne

    params = cruise.params
    idle = M._idle_level(list(train), params.mu)
    level = params.mu + rng.standard_normal() * params.s_c + rng.standard_normal(4) * params.s_r
    ranges = FlightPhaseRanges()
    names = (
        "pre_ground_s",
        "spinup_s",
        "warmup_s",
        "takeoff_s",
        "landing_s",
        "spindown_s",
        "post_ground_s",
    )
    total = sum(float(np.mean(getattr(ranges, k))) for k in names) + float(
        rng.uniform(*ranges.cruise_s)
    )

    def airborne(n: int, rng_: np.random.Generator, fs: float = RATE_HZ) -> np.ndarray:
        return level[:, None] + cruise.sample(n, rng_, rate_hz=fs, shaft_only=True)

    return wrap_airborne(airborne, total, RATE_HZ, rng, idle=idle, phases=ranges)


def _w1(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import wasserstein_distance

    return float(wasserstein_distance(a, b))


def score_flights(
    gen: Sequence[np.ndarray], held: Sequence[np.ndarray], fs: float
) -> dict[str, Any]:
    """Generated whole flights against the held-out ones, all ``(4, n)`` at
    ``fs``: the CRPS of every read-out (the generated flights are the
    predictive ensemble, the held-out flight the observation), the share of
    generated flights that are one valid pass, and Wasserstein distances of
    speed and increment distributions, speeds relative to each flight's hover.

    A generated flight that is not finite or exceeds ten times the fastest
    held-out rotor has diverged (a switching VAR can have an explosive state):
    it counts as invalid and enters no other statistic.
    """
    limit = 10.0 * max(float(np.max(x)) for x in held)
    ok = [x for x in gen if np.isfinite(x).all() and float(np.max(x)) < limit]
    g = [readouts(x, fs) for x in ok]
    h = [readouts(x, fs) for x in held]
    out: dict[str, Any] = {
        "valid_rate": float(sum(r["valid"] for r in g)) / max(len(gen), 1),
        "diverged_rate": 1.0 - len(ok) / max(len(gen), 1),
        "readouts": {},
    }
    for key in READOUTS:
        draws = np.array([r[key] for r in g if np.isfinite(r[key])])
        obs = [r[key] for r in h if np.isfinite(r[key])]
        entry: dict[str, Any] = {"n_gen": int(draws.size), "observed": obs}
        if draws.size >= 2 and obs:
            entry["crps"] = float(np.mean([crps(draws, o) for o in obs]))
            entry["median"] = float(np.median(draws))
            entry["abs_err"] = float(np.mean([abs(np.median(draws) - o) for o in obs]))
        out["readouts"][key] = entry
    if not ok:
        return out

    def rel(xs: Sequence[np.ndarray]) -> list[np.ndarray]:
        scaled = []
        for x in xs:
            marks = landmarks(x, fs)
            hover = marks.hover if marks is not None else float(np.percentile(x.mean(axis=0), 90))
            scaled.append(x / max(hover, 1e-9))
        return scaled

    gs, hs = rel(ok), rel(held)
    out["speed_w1"] = _w1(
        np.concatenate([x.ravel() for x in gs]), np.concatenate([x.ravel() for x in hs])
    )
    for lag_s in (0.2, 1.0):
        lag = int(round(lag_s * fs))
        a = np.concatenate([(x[:, lag:] - x[:, :-lag]).ravel() for x in gs])
        b = np.concatenate([(x[:, lag:] - x[:, :-lag]).ravel() for x in hs])
        out[f"inc_w1_{lag_s:g}"] = _w1(a, b)
        out[f"inc_q999_gen_{lag_s:g}"] = float(np.quantile(np.abs(a), 0.999))
        out[f"inc_q999_real_{lag_s:g}"] = float(np.quantile(np.abs(b), 0.999))
    return out


def run_flight(unit: Unit) -> dict[str, Any]:
    """One whole-flight unit: fit an arm on the training flights of a fold,
    generate flights, score them on the held-out ones."""
    p = unit.params
    rig, fold, arm, seed = str(p["rig"]), str(p["fold"]), str(p["arm"]), int(p.get("seed", 0))
    _threads(int(p.get("threads", 1)))
    data = rig_data(rig)
    held_ids = set() if fold == "all" else set(data.held_out[int(fold)])
    train = [f for f in data.flights if f.flight not in held_ids]
    held = [f for f in data.flights if f.flight in held_ids]
    rng = np.random.default_rng(seed + 10)
    out: dict[str, Any] = {
        "kind": "flight",
        "rig": rig,
        "fold": fold,
        "arm": arm,
        "held_out": sorted(held_ids),
    }
    t0 = time.perf_counter()
    gen: list[np.ndarray] = []
    if arm in ("ours", "scaffold"):
        cruise = _cruise_for_flights(p.get("out_dir"), rig, fold, seed)
        if arm == "ours":
            layer = fit_phase_layer(train)
            out["params"] = layer.to_dict()
            out["replay_error"] = {f.flight: replay_error(layer, f) for f in held}
            gen = [
                _to_flight_rate(layer.sample(cruise, rng)["rps"], layer.fs)
                for _ in range(N_FLIGHTS_GEN)
            ]
        else:
            gen = [
                _to_flight_rate(_scaffold_flight(cruise, train, rng), RATE_HZ)
                for _ in range(N_FLIGHTS_GEN)
            ]
    elif arm in ("msvar", "hsmm"):
        seqs = [_to_flight_rate(f.rps, f.fs) for f in train]
        longest = max(s.shape[1] for s in seqs)
        marks = [landmarks(s, FLIGHT_RATE_HZ) for s in seqs]
        if any(m is None for m in marks):
            raise ValueError("a training flight has no airborne segment")
        hover = float(np.median([m.hover for m in marks if m is not None]))
        model = fit_switching_var(
            seqs,
            FLIGHT_RATE_HZ,
            n_states=len(PHASES),
            order=FLIGHT_ORDER,
            durations="geometric" if arm == "msvar" else "explicit",
            max_dur_s=1.25 * longest / FLIGHT_RATE_HZ,
            intercept=True,
            sigma_floor=FLIGHT_SIGMA_FLOOR,
            n_init=1,
            max_iter=FLIGHT_MAX_ITER,
            seed=seed,
            name=f"{arm}_flight",
            init_labels=[phase_labels(m) for m in marks if m is not None],
        )
        out["params"] = model.to_dict()
        out["note"] = _switch_note(model)
        for _ in range(N_FLIGHTS_GEN):
            path = np.clip(model.sample(int(1.5 * longest), rng, start="initial"), 0.0, None)
            gen.append(first_pass(path, FLIGHT_RATE_HZ, hover))
    else:
        raise ValueError(f"unknown flight arm {arm!r}")
    out["fit_s"] = time.perf_counter() - t0
    if held:
        real = [_to_flight_rate(f.rps, f.fs) for f in held]
        out["scores"] = score_flights(gen, real, FLIGHT_RATE_HZ)
    return out


# ─── units ────────────────────────────────────────────────────────────────────


def cruise_units(
    rigs: Sequence[str], families: Sequence[str], *, final: bool, threads: int = 1
) -> list[Unit]:
    units = []
    for rig in rigs:
        for fold in ["all"] if final else fold_ids(rig):
            for family in families:
                units.append(
                    Unit(
                        uid=f"cruise__{rig}__{fold}__{family}",
                        params={"rig": rig, "fold": fold, "family": family, "threads": threads},
                    )
                )
    return units


def flight_units(
    rigs: Sequence[str], arms: Sequence[str], out_dir: str | Path, *, final: bool, threads: int = 1
) -> list[Unit]:
    units = []
    for rig in rigs:
        if rig not in FLIGHT_RIGS:
            continue
        for fold in ["all"] if final else fold_ids(rig):
            for arm in arms:
                units.append(
                    Unit(
                        uid=f"flight__{rig}__{fold}__{arm}",
                        params={
                            "rig": rig,
                            "fold": fold,
                            "arm": arm,
                            "threads": threads,
                            "out_dir": str(out_dir),
                        },
                    )
                )
    return units


def run_unit(unit: Unit) -> dict[str, Any]:
    """The gridrun worker: dispatch on the unit id."""
    return run_cruise(unit) if unit.uid.startswith("cruise__") else run_flight(unit)


# ─── aggregation ──────────────────────────────────────────────────────────────

#: Cruise arms in report order, and the scores of the report:
#: ``key -> (label, higher is better)``.
CRUISE_ARMS = ("gauss", "switch2", "switch3", "var", "msvar2", "msvar3", "hsmm2", "hsmm3")
CRUISE_METRICS: dict[str, tuple[str, bool]] = {
    "logpdf": ("one-step log predictive density, nats/sample", True),
    "es_skill_0.2": ("energy-score skill over climatology, 0.2 s ahead", True),
    "es_skill_1": ("energy-score skill over climatology, 1 s ahead", True),
    "es_skill_2": ("energy-score skill over climatology, 2 s ahead", True),
    "crps_skill_0.2": ("CRPS skill over climatology, 0.2 s ahead", True),
    "inc_w1_0.2": ("Wasserstein-1 of 0.2 s increments / their std", False),
    "inc_w1_1": ("Wasserstein-1 of 1 s increments / their std", False),
    "inc_kurt_diff_0.2": ("excess-kurtosis error of 0.2 s increments", False),
    "inc_kurt_diff_1": ("excess-kurtosis error of 1 s increments", False),
    "acf_rmse": ("autocorrelation RMSE, lags to 5 s", False),
    "xcorr_mae": ("inter-rotor correlation error", False),
    "var_log_ratio": ("abs log variance ratio", False),
    "psd_lsd_db": ("log-spectral distance of mixer modes, dB", False),
    "mmd2": ("squared MMD of 2 s windows", False),
}


def flat_cruise(scores: Mapping[str, Any]) -> dict[str, float]:
    """The report's cruise scores of one arm on one fold (missing: ``nan``)."""
    out = dict.fromkeys(CRUISE_METRICS, float("nan"))
    out["logpdf"] = float(scores["logpdf"])
    for h, v in scores.get("forecast", {}).get("horizons", {}).items():
        if f"es_skill_{h}" in out:
            out[f"es_skill_{h}"] = 1.0 - float(v["energy"]) / float(v["energy_clim"])
        if f"crps_skill_{h}" in out:
            out[f"crps_skill_{h}"] = 1.0 - float(v["crps"]) / float(v["crps_clim"])
    for key, value in scores.get("sample", {}).items():
        if key in out:
            out[key] = float(value)
    return out


def _stat(values: Sequence[float]) -> dict[str, Any]:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0}
    return {
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "n": int(v.size),
    }


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Every unit JSON into the report: per rig and arm the mean and spread
    over folds of every score, the per-fold values, the mean rank of every
    arm over the rigs, and the per-drone parameter sets (fold ``"all"``)."""
    cruise: dict[str, dict[str, dict[str, Any]]] = {}
    flight: dict[str, dict[str, dict[str, Any]]] = {}
    params: dict[str, dict[str, Any]] = {}
    for row in rows:
        rig, fold = str(row["rig"]), str(row["fold"])
        if row.get("kind") == "cruise":
            for name, arm in row["arms"].items():
                if fold == "all":
                    params.setdefault(rig, {}).setdefault("cruise", {})[name] = arm["params"]
                    continue
                slot = cruise.setdefault(rig, {}).setdefault(
                    name, {"folds": {}, "n_params": arm["n_params"], "fit_s": [], "diverged": 0}
                )
                slot["folds"][fold] = flat_cruise(arm["scores"])
                slot["fit_s"].append(float(arm["fit_s"]))
                slot["diverged"] += int(bool(arm["scores"].get("diverged", False)))
                slot["rate_hz"] = row["rate_hz"]
        elif row.get("kind") == "flight":
            arm = str(row["arm"])
            if fold == "all":
                if "params" in row:
                    params.setdefault(rig, {}).setdefault("flight", {})[arm] = row["params"]
                continue
            slot = flight.setdefault(rig, {}).setdefault(arm, {"folds": {}, "replay_error": {}})
            sc = row["scores"]
            flat = {
                "valid_rate": sc["valid_rate"],
                "diverged_rate": sc.get("diverged_rate", 0.0),
                "speed_w1": sc.get("speed_w1", float("nan")),
                "inc_w1_0.2": sc.get("inc_w1_0.2", float("nan")),
                "inc_w1_1": sc.get("inc_w1_1", float("nan")),
                "inc_q999_ratio_0.2": sc.get("inc_q999_gen_0.2", float("nan"))
                / sc.get("inc_q999_real_0.2", float("nan")),
            }
            for key, entry in sc["readouts"].items():
                flat[f"crps_{key}"] = entry.get("crps", float("nan"))
            slot["folds"][fold] = flat
            slot["replay_error"].update(row.get("replay_error", {}))
    for table in (cruise, flight):
        for arms in table.values():
            for slot in arms.values():
                keys = sorted({k for f in slot["folds"].values() for k in f})
                slot["mean"] = {
                    k: _stat([f.get(k, float("nan")) for f in slot["folds"].values()]) for k in keys
                }
    ranks: dict[str, dict[str, float]] = {}
    for key, (_, higher) in CRUISE_METRICS.items():
        per_arm: dict[str, list[float]] = {}
        for arms in cruise.values():
            vals = {a: s["mean"][key]["mean"] for a, s in arms.items() if key in s["mean"]}
            vals = {a: v for a, v in vals.items() if np.isfinite(v)}
            order = sorted(vals, key=lambda a: -vals[a] if higher else vals[a])
            for rank, arm in enumerate(order, start=1):
                per_arm.setdefault(arm, []).append(float(rank))
        ranks[key] = {a: float(np.mean(r)) for a, r in per_arm.items()}
    return {"cruise": cruise, "flight": flight, "rank": ranks, "params": params}


def _cell(stat: Mapping[str, Any], digits: int = 3) -> str:
    return "–" if not stat or not stat.get("n") else f"{stat['mean']:.{digits}f}"


def markdown(summary: Mapping[str, Any]) -> str:
    """The report tables: one per cruise score (arms by rigs, mean over the
    held-out folds, mean rank last) and one per whole-flight rig."""
    cruise, flight = summary["cruise"], summary["flight"]
    rigs = [r for r in RIGS if r in cruise]
    lines: list[str] = []
    for key, (label, higher) in CRUISE_METRICS.items():
        lines += [f"### {label} ({'higher' if higher else 'lower'} is better)", ""]
        lines.append("| arm | free parameters | " + " | ".join(rigs) + " | mean rank |")
        lines.append("|---|---|" + "---|" * (len(rigs) + 1))
        for arm in CRUISE_ARMS:
            if not any(arm in cruise[r] for r in rigs):
                continue
            n_par = sorted({cruise[r][arm]["n_params"] for r in rigs if arm in cruise[r]})
            cells = [_cell(cruise[r].get(arm, {}).get("mean", {}).get(key, {})) for r in rigs]
            rank = summary["rank"].get(key, {}).get(arm, float("nan"))
            par = str(n_par[0]) if len(n_par) == 1 else f"{n_par[0]}–{n_par[-1]}"
            lines.append(f"| {arm} | {par} | " + " | ".join(cells) + f" | {rank:.2f} |")
        lines.append("")
    for rig in [r for r in FLIGHT_RIGS if r in flight]:
        arms = [a for a in FLIGHT_ARMS if a in flight[rig]]
        keys = sorted({k for a in arms for k in flight[rig][a]["mean"]})
        lines += [f"### whole flights, {rig} (mean over held-out folds)", ""]
        lines.append("| score | " + " | ".join(arms) + " |")
        lines.append("|---|" + "---|" * len(arms))
        for key in keys:
            cells = [_cell(flight[rig][a]["mean"].get(key, {})) for a in arms]
            lines.append(f"| {key} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


# ─── does the Gaussian fit run faster on a GPU? ───────────────────────────────


def bench_gaussian_fit(rig: str, device: str, repeats: int = 5) -> dict[str, Any]:
    """Wall time of one stage-1 objective-and-gradient evaluation on a device.

    The objective is :func:`experiments.rps_traj.model._neg_log_post` on the
    rig's settled-cruise blocks: a Riccati fixed point and then one sequential
    recursion over the block length for all blocks at once.
    """
    import torch

    from experiments.rps_traj.switching import _data_start, gaussian_batches, level_stats

    data = rig_data(rig)
    segments = [s for segs in data.groups.values() for s in segs]
    mu, s_c, s_r = level_stats(data.groups)
    start = _data_start(segments, mu, s_c, s_r)
    previous = torch.get_default_device()
    torch.set_default_device(device)
    try:
        batches = gaussian_batches(segments, data.rate_hz)
        vec0 = M._fit_vector(start)
        times = []
        for _ in range(repeats + 1):
            vec = vec0.clone().requires_grad_(True)
            if device != "cpu":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            value = M._neg_log_post(vec, batches, data.rate_hz)
            (grad,) = torch.autograd.grad(value, vec)
            if device != "cpu":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        return {
            "rig": rig,
            "device": device,
            "rate_hz": data.rate_hz,
            "n_blocks": int(sum(b.n_blocks for b in batches)),
            "n_scored": int(sum(b.n_scored for b in batches)),
            "seconds": float(np.median(times[1:])),
            "value": float(value.item()),
            "grad_norm": float(grad.norm().item()),
        }
    finally:
        torch.set_default_device(previous)


__all__ = [
    "CRUISE_ARMS",
    "CRUISE_METRICS",
    "FAMILIES",
    "FLIGHT_ARMS",
    "FLIGHT_RIGS",
    "HORIZONS_S",
    "RIGS",
    "PhaseLayer",
    "RigData",
    "aggregate",
    "bench_gaussian_fit",
    "cruise_units",
    "first_pass",
    "flat_cruise",
    "flight_units",
    "fold_ids",
    "generated_blocks",
    "markdown",
    "mean_logpdf",
    "rig_data",
    "run_cruise",
    "run_flight",
    "run_unit",
    "score_cruise",
    "score_flights",
]
