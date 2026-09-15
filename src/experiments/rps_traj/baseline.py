"""The BASELINE rotor-speed trajectory model: the current synthesiser, fitted.

What the campaign's training streams sample today is
:func:`data_processing.rps_synthesis.generate_full_flight`, whose airborne part
is :func:`~data_processing.rps_synthesis.generate_intermittent` driven by a
hand-calibrated :class:`~data_processing.rps_synthesis.DroneProfile` (DREGON's
or Michael's).  That model is the thing a new one has to beat, so it must be
given its best shot on every rig: this module fits the *same* generative model
per rig and writes the fit out as the baseline artefact.

The intermittent model has no tractable likelihood (Poisson-gated rectangular
pulses through a first-order lag, plus OU jitter, mixed through :data:`MIXER`),
so it is fitted by **simulated moment matching** against the frozen statistics
of :mod:`experiments.rps_traj.stats`:

* the free vector is, per control mode, ``(trim, cruise_std, maneuver_std,
  rate_hz, mean_maneuver_s)`` plus the airframe's ``motor_tau`` and
  ``cruise_tau`` — 22 numbers (see :data:`PARAM_NAMES`).  Positive fields are
  optimised in log space, the roll/pitch/yaw trims (which may be negative) in
  linear space;
* ``aggressiveness`` is held at 1.0 (it multiplies maneuver rate AND amplitude,
  both of which are already free, so it is exactly redundant) and
  ``rps_min``/``rps_max`` are held at a wide bracket around the rig's own
  airborne range, where the clip is inactive;
* the loss is the frozen five-family discrepancy collapsed by the fixed weights
  of :data:`OBJECTIVE_WEIGHTS` — ``rotor_var + acf + xcorr + 0.1 rotor_mean +
  0.1 overall_mean``.  The three unit-weighted families are dimensionless
  (a log-variance ratio and two correlations, all O(0.1-1)); the two mean
  families are in rev/s and would otherwise dominate at 10-100x, so they carry
  0.1 — enough to pin the trims (they are the only families that see them) and
  not enough to buy a mean at the cost of the dynamics;
* every evaluation simulates the REAL flight durations (``n_rep`` times over)
  with the SAME seed — common random numbers.  That makes the objective an
  exactly deterministic function of the parameter vector, which is what
  the derivative-free search needs; it does not make it smooth, because the
  number of random draws depends on the maneuver rate, so the pulse stream
  re-shuffles as the parameters move.  The optimiser is therefore run as a
  pattern search, not as a gradient method, and the returned fit is
  re-evaluated with a larger ``n_rep`` to get a less noisy objective.

Fitting the baseline is NOT tuning the acceptance rule: the rule
(:func:`experiments.rps_traj.stats.passes`) compares a candidate against this
fitted baseline, so any effort spent here can only make the campaign's bar
harder to clear.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.optimize import minimize

from data_processing.rps_synthesis import (
    DREGON_PROFILE,
    MICHAELS_PROFILE,
    DroneProfile,
    FlightPhaseRanges,
    ManeuverModeParams,
    generate_full_flight,
    generate_intermittent,
)
from experiments.rps_traj.data import (
    ERODE_S,
    MIN_RUN_S,
    RATE_HZ,
    Flight,
    airborne_segments,
)
from experiments.rps_traj.stats import Sampler, compute_stats, discrepancy, stats_from_samples
from tracking.rotors import MODE_NAMES, modes_from_rps

_LOG = logging.getLogger(__name__)

#: Fixed collapse of the five frozen discrepancy families into one number.
#: See the module docstring for why the two rev/s families carry 0.1.
OBJECTIVE_WEIGHTS: dict[str, float] = {
    "rotor_var": 1.0,
    "acf": 1.0,
    "xcorr": 1.0,
    "rotor_mean": 0.1,
    "overall_mean": 0.1,
}

#: Value the objective returns when a parameter vector cannot be simulated or
#: scored (non-finite statistics). Far above any achievable loss, so the
#: simplex is pushed back into the feasible region.
OBJECTIVE_PENALTY = 1e3

#: ``n_rep`` used during the search (cheap) and for the final, reported
#: objective (less noisy).
N_REP_SEARCH = 2
N_REP_FINAL = 5

#: Simulated seconds per search evaluation the rig budget allows. The
#: generator runs at roughly 2000x realtime on one core (its motor lag and OU
#: jitter are Python loops), so 6000 simulated seconds is ~1 s per objective
#: evaluation and ~5 min for a 300-evaluation fit — the campaign's per-rig
#: laptop budget. Corpora big enough to blow it (NeuroBEM's 68 min, PITCN's
#: 61 min) drop to ``n_rep = 1``: fewer replicates of the same durations, NOT
#: shorter flights, because the frozen ACF is segment-length weighted and
#: shortening a duration would change the target itself.
SEARCH_SIM_BUDGET_S = 6000.0

#: Hard brackets every parameter is clipped into when a vector is turned back
#: into a :class:`DroneProfile`. These are simulator sanity limits, not priors:
#: ``rate_hz`` above a few events per second stops being "intermittent" and
#: makes the pulse loop the dominant cost, and a ``tau`` below a sample step is
#: indistinguishable from zero on the 100 Hz grid.
BOUNDS: dict[str, tuple[float, float]] = {
    "trim": (-1e4, 1e4),
    "cruise_std": (1e-3, 500.0),
    "maneuver_std": (1e-3, 1000.0),
    "rate_hz": (1e-3, 5.0),
    "mean_maneuver_s": (0.02, 30.0),
    "motor_tau": (1e-3, 5.0),
    "cruise_tau": (1e-3, 30.0),
}

#: The positive per-mode fields, optimised in log space, in vector order.
_LOG_MODE_FIELDS = ("cruise_std", "maneuver_std", "rate_hz", "mean_maneuver_s")

#: Names of the 22 free parameters, in vector order (diagnostics only).
PARAM_NAMES: tuple[str, ...] = (
    ("log_common_trim",)
    + tuple(f"{m}_trim" for m in MODE_NAMES[1:])
    + tuple(f"log_{m}_{f}" for m in MODE_NAMES for f in _LOG_MODE_FIELDS)
    + ("log_motor_tau", "log_cruise_tau")
)

#: Per-coordinate first step of the search. In log space 0.3 is a factor of
#: 1.35; the linear differential trims get a rev/s step scaled to the rig (see
#: :func:`_search_steps`), because they can start at zero and a relative step
#: would then be no step at all.
_LOG_STEP = 0.3
_TRIM_STEP_REL = 0.05

#: The derivative-free optimiser. The objective is deterministic but not
#: smooth (the pulse stream re-shuffles as the maneuver rate moves), so only
#: pattern searches apply. MEASURED at the campaign's 300-evaluation budget:
#: Powell reaches 0.693 on michaels and 0.494 on dregon where Nelder-Mead
#: reaches 1.161 and 0.565 at the same cost (~140 s per rig), because a
#: 23-point simplex in 22 dimensions spends its whole budget being built
#: whereas Powell's line searches bank a real improvement per direction. The
#: constant exists so that choice can be re-measured rather than re-argued.
SEARCH_METHOD = "Powell"


# ─── parameter vector <-> profile ─────────────────────────────────────────────


def _clip(name: str, value: float) -> float:
    lo, hi = BOUNDS[name]
    if not np.isfinite(value):
        return lo
    return float(min(max(value, lo), hi))


def profile_to_vector(profile: DroneProfile) -> np.ndarray:
    """The 22 free parameters of ``profile`` in optimiser space."""
    out: list[float] = [float(np.log(max(profile.common.trim, 1e-6)))]
    out += [float(m.trim) for m in profile.modes[1:]]
    for mode in profile.modes:
        out += [float(np.log(max(getattr(mode, f), 1e-9))) for f in _LOG_MODE_FIELDS]
    out += [
        float(np.log(max(profile.motor_tau, 1e-9))),
        float(np.log(max(profile.cruise_tau, 1e-9))),
    ]
    return np.asarray(out, dtype=np.float64)


def vector_to_profile(x: np.ndarray, template: DroneProfile) -> DroneProfile:
    """Optimiser vector → :class:`DroneProfile`, clipped into :data:`BOUNDS`.

    ``template`` supplies the fields that are NOT fitted (``rps_min``,
    ``rps_max``).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size != len(PARAM_NAMES):
        raise ValueError(f"expected {len(PARAM_NAMES)} parameters, got {x.size}")
    trims = [_clip("trim", float(np.exp(min(x[0], 20.0))))] + [_clip("trim", v) for v in x[1:4]]
    modes: list[ManeuverModeParams] = []
    for i, trim in enumerate(trims):
        block = x[4 + 4 * i : 8 + 4 * i]
        values = {
            f: _clip(f, float(np.exp(min(v, 20.0))))
            for f, v in zip(_LOG_MODE_FIELDS, block, strict=True)
        }
        modes.append(ManeuverModeParams(trim=trim, **values))
    return replace(
        template,
        common=modes[0],
        roll=modes[1],
        pitch=modes[2],
        yaw=modes[3],
        motor_tau=_clip("motor_tau", float(np.exp(min(x[-2], 5.0)))),
        cruise_tau=_clip("cruise_tau", float(np.exp(min(x[-1], 5.0)))),
    )


# ─── the fit ──────────────────────────────────────────────────────────────────


def _profile_json(profile: DroneProfile) -> dict[str, Any]:
    def mode(m: ManeuverModeParams) -> dict[str, float]:
        return {
            "trim": float(m.trim),
            "cruise_std": float(m.cruise_std),
            "maneuver_std": float(m.maneuver_std),
            "rate_hz": float(m.rate_hz),
            "mean_maneuver_s": float(m.mean_maneuver_s),
        }

    return {
        "common": mode(profile.common),
        "roll": mode(profile.roll),
        "pitch": mode(profile.pitch),
        "yaw": mode(profile.yaw),
        "motor_tau": float(profile.motor_tau),
        "cruise_tau": float(profile.cruise_tau),
        "rps_min": float(profile.rps_min),
        "rps_max": float(profile.rps_max),
    }


def _profile_from_json(payload: dict[str, Any]) -> DroneProfile:
    return DroneProfile(
        common=ManeuverModeParams(**{k: float(v) for k, v in payload["common"].items()}),
        roll=ManeuverModeParams(**{k: float(v) for k, v in payload["roll"].items()}),
        pitch=ManeuverModeParams(**{k: float(v) for k, v in payload["pitch"].items()}),
        yaw=ManeuverModeParams(**{k: float(v) for k, v in payload["yaw"].items()}),
        motor_tau=float(payload["motor_tau"]),
        cruise_tau=float(payload["cruise_tau"]),
        rps_min=float(payload["rps_min"]),
        rps_max=float(payload["rps_max"]),
    )


@dataclass
class BaselineFit:
    """A per-rig fit of the CURRENT intermittent/full-flight synthesiser.

    Attributes:
        rig: the rig id this was fitted to.
        profile: the fitted :class:`DroneProfile` (all fields, including the
            held-fixed ``rps_min``/``rps_max``).
        aggressiveness: the maneuver rate/amplitude multiplier passed to the
            generator. Always 1.0 for a fit — it is redundant with the free
            per-mode ``rate_hz``/``maneuver_std`` — and kept as a field only
            because the generator's API takes it.
        idle_rps: warm-up/idle common-mode level (rev/s) for
            :meth:`full_flight`. NOT fitted by the moment matching: the frozen
            airborne rule erases the ground and warm-up phases before any
            statistic is taken, so idle is unidentified by the objective and is
            instead read off the real flights' pre-takeoff plateau by
            :func:`estimate_idle_rps`.
        objective: the weighted discrepancy of the fitted profile, evaluated
            with ``n_rep = N_REP_FINAL``.
        n_evals: number of objective evaluations the search spent.
    """

    rig: str
    profile: DroneProfile
    aggressiveness: float
    idle_rps: float
    objective: float
    n_evals: int

    # ─── contract 4 (same shape as TrajStats: dict + text) ───

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict; every profile field is written explicitly."""
        return {
            "model": "baseline_intermittent",
            "rig": self.rig,
            "profile": _profile_json(self.profile),
            "aggressiveness": float(self.aggressiveness),
            "idle_rps": float(self.idle_rps),
            "objective": float(self.objective),
            "n_evals": int(self.n_evals),
        }

    def to_json(self) -> str:
        """JSON text, as :meth:`TrajStats.to_json`."""
        return json.dumps(self.to_dict(), indent=2, allow_nan=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BaselineFit:
        return cls(
            rig=str(payload["rig"]),
            profile=_profile_from_json(payload["profile"]),
            aggressiveness=float(payload["aggressiveness"]),
            idle_rps=float(payload["idle_rps"]),
            objective=float(payload["objective"]),
            n_evals=int(payload["n_evals"]),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | dict[str, Any]) -> BaselineFit:
        """JSON text (or an already-parsed dict) → :class:`BaselineFit`."""
        parsed = json.loads(payload) if isinstance(payload, (str, bytes)) else payload
        return cls.from_dict(parsed)

    def sampler(self, fs: float = RATE_HZ) -> Sampler:
        """The frozen sampler protocol: ``(n_samples, rng) -> (4, n_samples)``.

        This is the airborne/cruise model — exactly what the frozen statistics
        are computed on.
        """
        profile = self.profile
        aggressiveness = float(self.aggressiveness)

        def sample(n_samples: int, rng: np.random.Generator) -> np.ndarray:
            return generate_intermittent(
                n_samples / float(fs),
                float(fs),
                profile=profile,
                aggressiveness=aggressiveness,
                rng=rng,
            )

        return sample

    def full_flight(
        self, duration_s: float, fs: float = RATE_HZ, rng: np.random.Generator | int | None = None
    ) -> np.ndarray:
        """A whole flight: ground → spin-up → warm-up → takeoff → cruise →
        landing → ground, ``(4, n)`` rev/s at ``fs``.

        The idle plateau is pinned to :attr:`idle_rps` through
        :class:`FlightPhaseRanges`' ``idle_frac`` (the generator samples the
        idle level as a fraction of the hover trim, so a degenerate range fixes
        it exactly). For short requests the non-cruise phase ranges are shrunk
        proportionally so that cruise still gets ~40 % of the flight.
        """
        return generate_full_flight(
            float(duration_s),
            float(fs),
            profile=self.profile,
            aggressiveness=float(self.aggressiveness),
            phases=self._phases(float(duration_s)),
            rng=rng,
        )

    def _phases(self, duration_s: float) -> FlightPhaseRanges:
        frac = float(np.clip(self.idle_rps / max(self.profile.common.trim, 1e-6), 0.02, 0.95))
        ranges = FlightPhaseRanges()
        fields = (
            "pre_ground_s",
            "spinup_s",
            "warmup_s",
            "takeoff_s",
            "landing_s",
            "spindown_s",
            "post_ground_s",
        )
        fixed_max = sum(getattr(ranges, f)[1] for f in fields)
        budget = 0.6 * duration_s
        if fixed_max > budget:
            shrink = budget / fixed_max
            ranges = replace(
                ranges,
                **{f: tuple(v * shrink for v in getattr(ranges, f)) for f in fields},  # type: ignore[arg-type]
            )
        return replace(ranges, idle_frac=(frac, frac))


def objective_from_discrepancy(d: dict[str, float]) -> float:
    """Collapse the five frozen families with :data:`OBJECTIVE_WEIGHTS`."""
    total = 0.0
    for family, weight in OBJECTIVE_WEIGHTS.items():
        value = float(d[family])
        if not np.isfinite(value):
            return OBJECTIVE_PENALTY
        total += weight * value
    return float(min(total, OBJECTIVE_PENALTY))


def estimate_idle_rps(flights: Sequence[Flight]) -> float:
    """The real warm-up/idle common-mode level (rev/s), pooled over ``flights``.

    Idle is whatever the rotors hold while NOT airborne and NOT stopped: the
    median rotor-mean speed over the samples that the frozen airborne rule
    rejected but that still sit between 15 % and 80 % of the airborne hover
    level. Rigs whose recordings contain no ground phase at all (bench and
    simulation rigs) fall back to 45 % of hover, the midpoint of the
    generator's own default ``idle_frac``.
    """
    idle: list[np.ndarray] = []
    hovers: list[float] = []
    for flight in flights:
        rps = np.atleast_2d(np.asarray(flight.rps, dtype=np.float64))
        with np.errstate(invalid="ignore"):
            level = np.nanmean(rps, axis=0)
        segments = airborne_segments(rps, flight.fs)
        if not segments:
            continue
        air = np.zeros(level.size, dtype=bool)
        for sl in segments:
            air[sl] = True
        hover = float(np.nanmedian(level[air]))
        if not np.isfinite(hover) or hover <= 0.0:
            continue
        hovers.append(hover)
        candidates = level[~air & np.isfinite(level)]
        idle.append(candidates[(candidates > 0.15 * hover) & (candidates < 0.80 * hover)])
    if not hovers:
        raise ValueError("no airborne segments — cannot estimate an idle level")
    hover = float(np.mean(hovers))
    pooled = np.concatenate(idle) if idle else np.empty(0)
    if pooled.size < 0.5 * RATE_HZ:  # less than half a second of idle anywhere
        return 0.45 * hover
    return float(np.median(pooled))


def starting_profile(flights: Sequence[Flight]) -> DroneProfile:
    """The search's start point: the closer stock profile, moved onto this rig.

    Picks whichever of :data:`DREGON_PROFILE` / :data:`MICHAELS_PROFILE` has
    the closer hover level, then (i) sets the common trim to the rig's real
    overall mean and the differential trims to the mode projection of the real
    rotor means, ``MIXER.T mu / 4``, and (ii) scales every positive amplitude
    (``cruise_std``, ``maneuver_std``) by the hover-level ratio, because the
    stock profiles are calibrated for ~80 rev/s quads and the corpus spans a
    Crazyflie to a 300 rev/s bench rig — a 4x scale error in the trims would be
    a 4x scale error in the amplitudes too. Everything else (rates, durations,
    taus) is scale-free and starts at the stock value.
    """
    real = compute_stats(flights)
    mu = np.asarray(real.rotor_mean, dtype=np.float64)
    overall = float(real.overall_mean)
    stock = min((DREGON_PROFILE, MICHAELS_PROFILE), key=lambda p: abs(p.common.trim - overall))
    scale = overall / stock.common.trim if stock.common.trim > 0 else 1.0

    modes = modes_from_rps(mu[:, None])[:, 0] if mu.size == 4 else np.zeros(4)
    trims = (overall, float(modes[1]), float(modes[2]), float(modes[3]))
    rescaled = [
        replace(
            params,
            trim=trim,
            cruise_std=params.cruise_std * scale,
            maneuver_std=params.maneuver_std * scale,
        )
        for params, trim in zip(stock.modes, trims, strict=True)
    ]

    # The clip bracket is FIXED, not fitted: a span's worth of headroom either
    # side of everything the rig was ever observed to do, so the clip never
    # bites during the search (a biting clip would make the objective flat in
    # the amplitude directions).
    observed = np.concatenate([np.asarray(f.rps, dtype=np.float64).reshape(-1) for f in flights])
    observed = observed[np.isfinite(observed)]
    if observed.size == 0:
        raise ValueError("flights carry no finite rotor speeds")
    lo, hi = float(observed.min()), float(observed.max())
    span = max(hi - lo, 1.0)
    return replace(
        stock,
        common=rescaled[0],
        roll=rescaled[1],
        pitch=rescaled[2],
        yaw=rescaled[3],
        rps_min=max(0.0, lo - span),
        rps_max=hi + span,
    )


def _search_steps(x0: np.ndarray, trim_scale: float) -> np.ndarray:
    """Per-coordinate first step of the search: absolute and unit-aware.

    scipy's own defaults perturb each coordinate by 5 % of its VALUE, which is
    no perturbation at all for a differential trim that starts near zero —
    hence the explicit construction. In log space the step is a fixed factor;
    the three linear trims get a rev/s step scaled to the rig's own level.
    """
    step = np.full(x0.size, _LOG_STEP)
    step[1:4] = max(_TRIM_STEP_REL * trim_scale, 0.25)
    return step


def fitting_durations(flights: Sequence[Flight]) -> list[float]:
    """The flight durations worth simulating, in seconds.

    A trajectory shorter than ``MIN_RUN_S + 2 ERODE_S`` = 7 s cannot survive
    the frozen airborne rule (a 5 s run after 1 s of erosion at each end), so
    :func:`~experiments.rps_traj.stats.compute_stats` discards it whatever it
    contains. Dropping those durations is exact, not an approximation, and on
    NeuroBEM (115 of 247 flights are shorter) it is free speed.
    """
    floor = MIN_RUN_S + 2.0 * ERODE_S
    kept = [float(f.duration_s) for f in flights if float(f.duration_s) >= floor]
    if not kept:
        raise ValueError(
            f"every flight is shorter than {floor:.0f}s — none can be airborne under the "
            "frozen rule, so there is nothing to fit"
        )
    return kept


def search_n_rep(durations: Sequence[float]) -> int:
    """Replicates per search evaluation: :data:`N_REP_SEARCH`, or fewer when
    the corpus is long enough to blow :data:`SEARCH_SIM_BUDGET_S`."""
    total = float(sum(durations))
    if total <= 0.0:
        return N_REP_SEARCH
    return int(max(1, min(N_REP_SEARCH, int(SEARCH_SIM_BUDGET_S // total))))


def fit_baseline(
    flights: Sequence[Flight],
    rig: str,
    *,
    seed: int = 0,
    n_evals: int = 300,
) -> BaselineFit:
    """Fit the current intermittent synthesiser to ``flights`` by moment matching.

    Args:
        flights: the rig's real flights (see
            :func:`experiments.rps_traj.data.load_rig`).
        rig: the rig id recorded in the fit.
        seed: the COMMON RANDOM NUMBER seed — every objective evaluation, and
            the final re-evaluation, simulate with this same seed, so the
            objective is a deterministic function of the parameters.
        n_evals: budget of objective evaluations for the :data:`SEARCH_METHOD`
            search (``maxfev``). Each evaluation simulates
            :func:`search_n_rep` copies of every real flight duration long
            enough to be airborne.

    Returns:
        The :class:`BaselineFit` with the better of (start point, best profile
        the search saw) judged at ``N_REP_FINAL`` — the search optimises a
        2-replicate estimate of the objective, so a marginal "improvement" can
        be simulation noise, and the baseline must never be handed to the
        acceptance rule in a state worse than the stock profile it started
        from. Both objectives, per family, are logged at INFO.
    """
    if not flights:
        raise ValueError("no flights to fit")
    real = compute_stats(flights)
    durations = fitting_durations(flights)
    n_rep_search = search_n_rep(durations)
    template = starting_profile(flights)
    x0 = profile_to_vector(template)

    def score(x: np.ndarray, n_rep: int) -> tuple[float, dict[str, float]]:
        profile = vector_to_profile(x, template)
        candidate = BaselineFit(
            rig=rig,
            profile=profile,
            aggressiveness=1.0,
            idle_rps=0.0,
            objective=float("nan"),
            n_evals=0,
        )
        try:
            model = stats_from_samples(
                candidate.sampler(),
                durations,
                n_rep=n_rep,
                seed=int(seed),
            )
        except (ValueError, FloatingPointError):
            # A degenerate profile can leave no airborne segment at all.
            return OBJECTIVE_PENALTY, dict.fromkeys(OBJECTIVE_WEIGHTS, float("inf"))
        families = discrepancy(model, real)
        return objective_from_discrepancy(families), families

    calls = 0
    best = [float("inf"), x0.copy()]

    def loss(x: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        value, _ = score(x, n_rep_search)
        if value < best[0]:
            best[0], best[1] = value, np.asarray(x, dtype=np.float64).copy()
        return value

    start_search, start_families = score(x0, n_rep_search)
    best[0] = start_search
    _LOG.info(
        "%s: %d flights (%.0fs, n_rep=%d) start objective %.4f %s",
        rig,
        len(durations),
        sum(durations),
        n_rep_search,
        start_search,
        _fmt_families(start_families),
    )

    t0 = time.perf_counter()
    step = _search_steps(x0, abs(float(real.overall_mean)))
    if SEARCH_METHOD == "Nelder-Mead":
        options = {
            "maxfev": int(n_evals),
            "initial_simplex": np.vstack([x0, x0 + np.diag(step)]),
            "xatol": 1e-3,
            "fatol": 1e-4,
        }
    else:
        options = {"maxfev": int(n_evals), "direc": np.diag(step), "xtol": 1e-3, "ftol": 1e-4}
    result = minimize(loss, x0, method=SEARCH_METHOD, options={**options, "disp": False})
    wall = time.perf_counter() - t0

    x_search = np.asarray(result.x, dtype=np.float64)
    if not np.isfinite(result.fun) or float(result.fun) > best[0]:
        x_search = best[1]

    start_final, start_final_families = score(x0, N_REP_FINAL)
    search_final, search_final_families = score(x_search, N_REP_FINAL)
    if search_final <= start_final:
        x_best, final, final_families = x_search, search_final, search_final_families
    else:
        x_best, final, final_families = x0, start_final, start_final_families
    _LOG.info(
        "%s: %d evals in %.1fs; search(n_rep=%d) %.4f -> %.4f; "
        "final(n_rep=%d) start %.4f %s -> fitted %.4f %s",
        rig,
        calls,
        wall,
        n_rep_search,
        start_search,
        float(best[0]),
        N_REP_FINAL,
        start_final,
        _fmt_families(start_final_families),
        final,
        _fmt_families(final_families),
    )

    profile = vector_to_profile(x_best, template)
    return BaselineFit(
        rig=rig,
        profile=profile,
        aggressiveness=1.0,
        idle_rps=estimate_idle_rps(flights),
        objective=final,
        n_evals=calls,
    )


def _fmt_families(families: dict[str, float]) -> str:
    return "[" + " ".join(f"{k}={families[k]:.4f}" for k in OBJECTIVE_WEIGHTS) + "]"


__all__ = [
    "BOUNDS",
    "N_REP_FINAL",
    "N_REP_SEARCH",
    "OBJECTIVE_PENALTY",
    "OBJECTIVE_WEIGHTS",
    "PARAM_NAMES",
    "SEARCH_METHOD",
    "SEARCH_SIM_BUDGET_S",
    "BaselineFit",
    "estimate_idle_rps",
    "fit_baseline",
    "fitting_durations",
    "objective_from_discrepancy",
    "profile_to_vector",
    "search_n_rep",
    "starting_profile",
    "vector_to_profile",
]
