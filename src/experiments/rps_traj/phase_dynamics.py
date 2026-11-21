"""Dynamics inside each phase when the schedule is given.

The whole-flight comparison of :mod:`experiments.rps_traj.regime_campaign`
lets every generator choose when to change phase, and the switching VARs fail
at that.  Here the schedule is not the model's business: every arm is told
when the motors start, when the take-off begins and when the landing begins,
and is scored on what the rotors do inside windows aligned to those events.

Arms:

* ``ours`` - the fitted phase layer around the two-level cruise model
  (:meth:`~experiments.rps_traj.transients.PhaseLayer.sample` with the
  held-out flight's time on the ground and time airborne);
* ``phasevar2``, ``phasevar4`` - one Gaussian VAR(p) per phase, fitted to the
  training samples of that phase (the maximum-likelihood switching VAR when
  the states are known; with the states given, the Markov-switching VAR and
  the VAR-HSMM are this same model), kept stable, simulated along the
  held-out flight's own phase labels and clamped as in the campaign.  They
  are given more than ``ours``: the labels also say when each transient ENDS.

Scores, per window and held-out flight, 64 generated flights the ensemble:

* ``es_path`` - energy score of the four rotor speeds at each instant of the
  window, mean over the window, relative to hover;
* ``es_inc`` - the same for the 0.2 s increments: the proper score of the
  dynamics rather than of the level;
* ``inc_w1`` - Wasserstein-1 between generated and held-out 0.2 s increments
  pooled over the window, relative to hover;
* ``inc_sd_gen``, ``inc_sd_real`` - their standard deviations, relative to
  hover (a generator that is too smooth shows here).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import wasserstein_distance

from experiments.rps_traj import regime_campaign as rc
from experiments.rps_traj.cruise_api import NUM_ROTORS
from experiments.rps_traj.phases import PHASES, Landmarks, landmarks, phase_labels
from experiments.rps_traj.scores import _energy_scores
from experiments.rps_traj.transients import fit_phase_layer
from experiments.rps_traj.var_models import _simulate_kernel, fit_switching_var
from utils.gridrun import Unit

ARMS = ("ours", "phasevar2", "phasevar4")
WINDOWS = ("startup", "standby", "takeoff", "landing")
METRICS = ("es_path", "es_inc", "inc_w1", "inc_sd_gen", "inc_sd_real")

#: Window lengths (s): after motors-on, before take-off, after take-off, and
#: around the landing mark; the lag of the increments.
STARTUP_S = 3.0
STANDBY_S = 2.0
TAKEOFF_S = 4.0
LANDING_PRE_S = 0.5
LANDING_S = 4.0
INC_S = 0.2
N_GEN = 64
FS = rc.FLIGHT_RATE_HZ


def _n(seconds: float) -> int:
    return int(round(seconds * FS))


def windows(marks: Landmarks) -> dict[str, tuple[str, int, int]]:
    """``window -> (event, lo, hi)``: sample offsets from an event of a
    recording at :data:`FS`, for the windows its landmarks allow."""
    out: dict[str, tuple[str, int, int]] = {}
    on, takeoff, landing = marks.motors_on, marks.takeoff, marks.landing
    if on is not None and takeoff is not None:
        out["startup"] = ("motors_on", 0, min(_n(STARTUP_S), takeoff - on))
        if marks.standby is not None:
            out["standby"] = ("takeoff", -min(_n(STANDBY_S), takeoff - marks.standby[0]), 0)
    if takeoff is not None:
        end = marks.n if landing is None else landing
        out["takeoff"] = ("takeoff", 0, min(_n(TAKEOFF_S), end - takeoff))
    if landing is not None:
        start = takeoff if marks.arrival is None else marks.arrival
        pre = min(_n(LANDING_PRE_S), landing - (start or 0))
        out["landing"] = ("landing", -pre, min(_n(LANDING_S), marks.n - landing))
    lag = _n(INC_S)
    return {k: v for k, v in out.items() if v[2] - v[1] >= lag + 2}


def cut(x: np.ndarray, event: int | None, lo: int, hi: int) -> np.ndarray | None:
    """``x[:, event + lo : event + hi]``; past its end a recording is at rest."""
    if event is None or event + lo < 0:
        return None
    out = np.zeros((x.shape[0], hi - lo))
    part = x[:, event + lo : event + hi]
    out[:, : part.shape[1]] = part
    return out


def window_scores(real: np.ndarray, gens: np.ndarray, hover: float) -> dict[str, float]:
    """Scores of an ensemble ``(N, 4, L)`` against a held-out window ``(4, L)``."""
    lag = _n(INC_S)
    dr = real[:, lag:] - real[:, :-lag]
    dg = gens[:, :, lag:] - gens[:, :, :-lag]
    return {
        "es_path": float(_energy_scores(gens.transpose(0, 2, 1), real.T).mean()) / hover,
        "es_inc": float(_energy_scores(dg.transpose(0, 2, 1), dr.T).mean()) / hover,
        "inc_w1": float(wasserstein_distance(dg.ravel(), dr.ravel())) / hover,
        "inc_sd_gen": float(dg.std()) / hover,
        "inc_sd_real": float(dr.std()) / hover,
    }


def score_flight(
    real: np.ndarray, marks: Landmarks, gens: Sequence[tuple[np.ndarray, Mapping[str, Any]]]
) -> dict[str, dict[str, float]]:
    """Every window of one held-out flight ``(4, n)`` at :data:`FS` against
    generated flights, each cut at ITS OWN events."""
    out: dict[str, dict[str, float]] = {}
    for name, (event, lo, hi) in windows(marks).items():
        truth = cut(real, getattr(marks, event), lo, hi)
        cuts = [cut(x, events.get(event), lo, hi) for x, events in gens]
        kept = [c for c in cuts if c is not None]
        if truth is None or len(kept) < 2:
            continue
        out[name] = window_scores(truth, np.stack(kept), marks.hover)
        out[name]["n_gen"] = float(len(kept))
    return out


def _events(marks: Landmarks) -> dict[str, int | None]:
    return {"motors_on": marks.motors_on, "takeoff": marks.takeoff, "landing": marks.landing}


Generated = list[tuple[np.ndarray, Mapping[str, Any]]]


def fit_arm(
    arm: str, rig: str, fold: str, out_dir: str | None, seed: int = 0, *, same_clock: bool = False
) -> Callable[[Landmarks, int, np.random.Generator], Generated | None]:
    """An arm fitted on the training flights of a fold, as a generator of
    ``n`` flights ``(rps (4, .) at FS, events)`` on the schedule of a held-out
    flight's landmarks (``None`` when the flight has no take-off to follow).

    ``same_clock`` makes ``ours`` land at the held-out flight's landing time as
    well: its take-off lasts as long as its own ramp does, so the time airborne
    is shortened or lengthened by the difference (the levels, hence the ramp,
    are the first things a draw fixes: the same seed gives them again).
    """
    data = rc.rig_data(rig)
    held_ids = set(data.held_out[int(fold)])
    train = [f for f in data.flights if f.flight not in held_ids]

    if arm == "ours":
        cruise = rc._cruise_for_flights(out_dir, rig, fold, seed)
        layer = fit_phase_layer(train)
        scale = FS / layer.fs

        def ours(marks: Landmarks, n: int, rng: np.random.Generator) -> Generated | None:
            if marks.motors_on is None or marks.takeoff is None:
                return None
            arrival = marks.takeoff if marks.arrival is None else marks.arrival
            end = marks.n if marks.landing is None else marks.landing
            pre = (marks.takeoff - marks.motors_on) / FS
            air = max((end - arrival) / FS, 1.0)
            gens: Generated = []
            for _ in range(n):
                if same_clock:
                    state = int(rng.integers(2**62))
                    first = layer.sample(
                        cruise, np.random.default_rng(state), airborne_s=air, pre_takeoff_s=pre
                    )
                    ramp = (first["arrival"] - first["takeoff"]) / layer.fs
                    draw = layer.sample(
                        cruise,
                        np.random.default_rng(state),
                        airborne_s=max((end - marks.takeoff) / FS - ramp, 1.0),
                        pre_takeoff_s=pre,
                    )
                else:
                    draw = layer.sample(cruise, rng, airborne_s=air, pre_takeoff_s=pre)
                events = {
                    k: None if draw[k] is None else int(round(draw[k] * scale))
                    for k in ("motors_on", "takeoff", "landing")
                }
                gens.append((rc._to_flight_rate(draw["rps"], layer.fs), events))
            return gens

        return ours

    seqs = [rc._to_flight_rate(f.rps, f.fs) for f in train]
    labelled = [(s, landmarks(s, FS)) for s in seqs]
    model = fit_switching_var(
        [s for s, m in labelled if m is not None],
        FS,
        n_states=len(PHASES),
        order=int(arm.removeprefix("phasevar")),
        intercept=True,
        sigma_floor=rc.FLIGHT_SIGMA_FLOOR,
        n_init=1,
        max_iter=0,  # the states are known: the labelled fit is the estimate
        seed=seed,
        init_labels=[phase_labels(m) for _, m in labelled if m is not None],
        stable=True,
    )
    cap = rc.FLIGHT_CAP * max(float(s.max()) for s in seqs)

    def phasevar(marks: Landmarks, n: int, rng: np.random.Generator) -> Generated | None:
        states = np.tile(phase_labels(marks), (n, 1))
        eps = rng.standard_normal((n, marks.n, NUM_ROTORS))
        lags = np.zeros((n, model.order, NUM_ROTORS))
        paths = _simulate_kernel(states, model.mu, model.coefs, model._chol, eps, lags, 0.0, cap)
        return [(path.T, _events(marks)) for path in paths]

    return phasevar


def run_unit(unit: Unit) -> dict[str, Any]:
    """One unit: an arm fitted on the training flights of a fold and scored
    on every window of the held-out ones."""
    p = unit.params
    rig, fold, arm, seed = str(p["rig"]), str(p["fold"]), str(p["arm"]), int(p.get("seed", 0))
    rc._threads(int(p.get("threads", 1)))
    data = rc.rig_data(rig)
    held_ids = set(data.held_out[int(fold)])
    rng = np.random.default_rng(seed + 20)
    out: dict[str, Any] = {"kind": "phase", "rig": rig, "fold": fold, "arm": arm, "flights": {}}
    generate = fit_arm(arm, rig, fold, p.get("out_dir"), seed)
    for flight in (f for f in data.flights if f.flight in held_ids):
        real = rc._to_flight_rate(flight.rps, flight.fs)
        marks = landmarks(real, FS)
        gens = None if marks is None else generate(marks, N_GEN, rng)
        if marks is not None and gens is not None:
            out["flights"][flight.flight] = score_flight(real, marks, gens)
    return out


def example_flights(
    rig: str, out_dir: str | None, n: int = 3, fold: str = "0", seed: int = 0
) -> dict[str, Any]:
    """Generated flights of every arm on ONE schedule: that of the first
    held-out flight of a fold that goes from the ground to the ground (the
    first one otherwise).  ``{"flight", "t", "events" (s), "labels", "real",
    arm: [rps (4, n_t), ...]}``, everything at :data:`FS` on the held-out
    flight's own clock."""
    data = rc.rig_data(rig)
    held = [f for f in data.flights if f.flight in set(data.held_out[int(fold)])]
    marked = [(f, rc._to_flight_rate(f.rps, f.fs)) for f in held]
    marked = [(f, x, landmarks(x, FS)) for f, x in marked]
    whole = [(f, x, m) for f, x, m in marked if m is not None and m.whole]
    flight, real, marks = (whole or [e for e in marked if e[2] is not None])[0]
    assert marks is not None
    out: dict[str, Any] = {
        "flight": flight.flight,
        "t": np.arange(marks.n) / FS,
        "events": {k: None if v is None else v / FS for k, v in _events(marks).items()},
        "labels": phase_labels(marks),
        "real": real,
    }
    for arm in ARMS:
        gens = fit_arm(arm, rig, fold, out_dir, seed, same_clock=True)(
            marks, n, np.random.default_rng(seed + 30)
        )
        paths = []
        for x, events in gens or []:
            shift = (marks.motors_on or 0) - (events.get("motors_on") or 0)
            y = np.zeros_like(real)
            src = x[:, max(-shift, 0) : max(-shift, 0) + marks.n - max(shift, 0)]
            y[:, max(shift, 0) : max(shift, 0) + src.shape[1]] = src
            paths.append(y)
        out[arm] = paths
    return out


def units(rigs: Sequence[str], arms: Sequence[str], out_dir: str, threads: int = 1) -> list[Unit]:
    return [
        Unit(
            uid=f"phase__{rig}__{fold}__{arm}",
            params={"rig": rig, "fold": fold, "arm": arm, "threads": threads, "out_dir": out_dir},
        )
        for rig in rigs
        if rig in rc.FLIGHT_RIGS
        for fold in rc.fold_ids(rig)
        for arm in arms
    ]


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``rig -> window -> arm -> metric -> mean over held-out flights``."""
    acc: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
    for row in rows:
        if row.get("kind") != "phase":
            continue
        for scores in row["flights"].values():
            for window, values in scores.items():
                slot = (
                    acc.setdefault(row["rig"], {}).setdefault(window, {}).setdefault(row["arm"], {})
                )
                for key, value in values.items():
                    slot.setdefault(key, []).append(float(value))
    return {
        rig: {
            w: {
                a: {k: float(np.mean(v)) for k, v in m.items()} | {"n": len(m["es_path"])}
                for a, m in arms.items()
            }
            for w, arms in wins.items()
        }
        for rig, wins in acc.items()
    }


def markdown(summary: Mapping[str, Any]) -> str:
    """One table per rig: windows by arms, every score x 100 (percent of hover)."""
    lines: list[str] = []
    for rig in [r for r in rc.FLIGHT_RIGS if r in summary]:
        lines += [
            f"### dynamics on a given schedule, {rig} (percent of hover, mean over held-out flights)",
            "",
        ]
        lines.append("| window | flights | score | " + " | ".join(ARMS) + " |")
        lines.append("|---|---|---|" + "---|" * len(ARMS))
        for window in [w for w in WINDOWS if w in summary[rig]]:
            arms = summary[rig][window]
            n = max(int(a["n"]) for a in arms.values())
            for key in ("es_path", "es_inc", "inc_w1", "inc_sd_gen"):
                cells = [f"{100 * arms[a][key]:.2f}" if a in arms else "–" for a in ARMS]
                lines.append(f"| {window} | {n} | {key} | " + " | ".join(cells) + " |")
            real = next(iter(arms.values()))["inc_sd_real"]
            lines.append(
                f"| {window} | {n} | inc_sd_real | "
                + " | ".join([f"{100 * real:.2f}"] * len(ARMS))
                + " |"
            )
        lines.append("")
    return "\n".join(lines)
