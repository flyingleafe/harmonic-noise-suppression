"""``rps.kind: fitted_traj`` — training trajectories from the FITTED model.

The incumbent ``full_flight`` trajectory (:mod:`data_processing.rps_synthesis`)
is a hand-written scaffold: telegraph set-points plus an OU jitter, with ranges
chosen by eye.  This source replaces the airborne part of it with the
exact-likelihood MAP fits of the campaign's generative model
(``docs/experiments/rps-trajectory-model.md``), published as the
``rps-traj-fits`` dataset: seven per-rig fits plus the GLOBAL fit — the
diagonal Gaussian over all seven rigs in scale-free coordinates, from which a
FRESH drone can be drawn per flight.

One noise source's ``rps`` block::

    rps:
      kind: fitted_traj
      fits: dload:rps-traj-fits   # or a plain directory with the same layout
      rigs:                       # a weighted mixture; weights need not sum to 1
        michaels: 1.0
        dregon: 1.0
        posterior: 2.0            # the reserved name: a fresh draw per flight
      mean_shift: [-10.0, 10.0]   # additive rev/s, one draw per flight
      mean_scale: [1.0, 1.0]      # multiplicative, one draw per flight
      flight_fs: 200              # like full_flight: the trajectory's own rate
      flight_reuse: 32            # like full_flight: windows per flight
      measurement_noise: true     # keep the per-rotor measurement OU term
      antithetic_offsets: false
      rps_max: 150                # OPTIONAL cap (rev/s): reject-and-redraw a
                                  # flight any rotor of which exceeds it
      phases: {warmup_s: [3.0, 25.0]}   # optional FlightPhaseRanges overrides

Per flight the source picks a rig by weight, turns it into
:class:`~data_processing.trajectory_model.params.Params` (a stored fit, or a
posterior draw), applies ``mean_scale`` then ``mean_shift`` to the hover level —
the ESC clamp and the warm-up idle level move with it — and wraps one stationary
airborne realisation in the full-flight envelope
(:func:`~data_processing.trajectory_model.flight.wrap_airborne`: ground,
spin-up, idle, take-off, airborne, landing, spin-down, ground).  The pools cache
that flight and window it exactly as they window a ``full_flight`` one.

``rps_max`` is a HARD CEILING on the whole flight, enforced by rejection: a
flight in which any rotor exceeds it anywhere is thrown away and the attempt
redrawn (a fresh drone for ``posterior``, a fresh hover shift and realisation
for a stored rig), up to :data:`MAX_RPS_REDRAWS` times, after which the source
raises rather than silently returning an out-of-range flight.  It exists
because the rig posterior is a hyperprior over SEVEN rigs and draws hover
levels far above what a downstream model's speed grid covers (the salience
trunks are built on 0-150 rev/s), and because clipping a trajectory to the
ceiling would invent a flat-topped flight no rig flies.  What it gives up is
stated plainly: the accepted population is the drawn one CONDITIONED on
staying under the cap — a documented truncation of the hyperprior — and
``FittedTrajectorySource.stats`` counts accepted and rejected flights so the
truncation can be reported.

``flight_fs``/``flight_reuse`` stay the POOL's knobs (they are the same keys the
incumbent kind uses); everything else above belongs to this source, and every
draw comes from the pool's own rng, so a stream stays reproducible per
``(base_seed, epoch, worker, position)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from data_processing.trajectory_model.flight import FlightPhaseRanges, wrap_airborne
from data_processing.trajectory_model.params import Params
from data_processing.trajectory_model.posterior import Posterior
from data_processing.trajectory_model.sampler import NewFit
from tracking.rotors import NUM_ROTORS

#: The reserved mixture name that draws a fresh drone from the global fit
#: (the rig posterior) for every flight, instead of reusing a stored rig.
POSTERIOR_RIG = "posterior"

#: The ``rps.kind`` values that window a CACHED whole flight instead of drawing
#: a cruise window per clip: the hand-written scaffold
#: (:func:`data_processing.rps_synthesis.generate_full_flight`) and this module.
#: Every pool caches and windows both with the same code.
FLIGHT_KINDS = ("full_flight", "fitted_traj")

#: The ``rps.kind`` this module serves.
FITTED_KIND = "fitted_traj"

#: A posterior draw has no measured ESC floor and ceiling, so it takes this
#: clamp, relative to its own hover level.  The stored rigs sit at 0.50-0.70 of
#: their mean at the floor and 1.15-2.05 at the ceiling, so these bounds are
#: looser than every measured floor and above all but one measured ceiling:
#: they clip the Gaussian's far tail and nothing a real rig does.
POSTERIOR_CLAMP_REL = (0.35, 1.6)

#: A posterior draw has no measured warm-up idle level either, so it idles at
#: this fraction of its own hover level, spread over the rotors by its own trim
#: — the same fallback the fitting campaign uses for a rig whose recordings
#: start already airborne.  The stored rigs idle at 0.11-0.49 of their mean.
POSTERIOR_IDLE_REL = 0.35

#: Files the dataset (or a plain directory) must contain.
FITS_SUBDIR = "fits"
POSTERIOR_FILE = "posterior.json"


@dataclass(frozen=True)
class FitBundle:
    """Everything one ``rps-traj-fits`` tree holds."""

    root: Path
    fits: Mapping[str, NewFit]
    posterior: Posterior | None

    @property
    def names(self) -> tuple[str, ...]:
        """Every name a mixture may weight, stored rigs first."""
        extra = (POSTERIOR_RIG,) if self.posterior is not None else ()
        return tuple(sorted(self.fits)) + extra


#: How many times :meth:`FittedTrajectorySource.flight` may redraw a flight
#: that violates ``rps_max`` before it gives up. A cap the mixture cannot
#: clear at all would otherwise spin forever in a DataLoader worker, so the
#: budget is finite; it is 64 rather than 32 because the number that matters
#: is the chance of killing a LONG RUN on a cap that is merely tight. The
#: measured acceptance of the rig posterior at ``rps_max: 150`` is 0.309
#: (256 accepted of 828 attempts, ``mean_shift: [-5, 5]``), at which 33
#: attempts fail once per 2.1e5 flights — about one job in seven over a
#: 31k-flight run — while 65 attempts fail once per 4.7e9.
MAX_RPS_REDRAWS = 64


@dataclass(frozen=True)
class RigDraw:
    """One flight's drone: where it came from and what it is."""

    rig: str
    params: Params
    idle_rps: np.ndarray
    clip: tuple[float, float]
    mean_scale: float
    mean_shift: float
    #: Flights thrown away by the ``rps_max`` cap before this one was accepted
    #: (0 when the first attempt passed, or when no cap is set).
    redraws: int = 0


@lru_cache(maxsize=8)
def _load_bundle(source: str) -> FitBundle:
    from data_processing.streams import resolve_source  # noqa: PLC0415 (keeps dload optional)

    root = Path(resolve_source(source))
    fit_dir = root / FITS_SUBDIR
    if not fit_dir.is_dir():
        raise FileNotFoundError(
            f"{source!r} resolves to {root}, which has no {FITS_SUBDIR}/ directory — "
            "expected the rps-traj-fits layout (fits/<rig>.json + posterior.json)"
        )
    fits = {
        path.stem: NewFit.from_json(path.read_text(encoding="utf-8"))
        for path in sorted(fit_dir.glob("*.json"))
    }
    if not fits:
        raise FileNotFoundError(f"no fits in {fit_dir}")
    posterior_path = root / POSTERIOR_FILE
    posterior = (
        Posterior.from_json(posterior_path.read_text(encoding="utf-8"))
        if posterior_path.is_file()
        else None
    )
    return FitBundle(root=root, fits=fits, posterior=posterior)


def load_bundle(source: str | Path) -> FitBundle:
    """Load a fits tree, memoized per location (the files never change: a dload
    version is content-addressed and a raw tree is committed once)."""
    return _load_bundle(str(source))


class FittedTrajectorySource:
    """A weighted mixture of fitted rigs, drawn one drone per flight.

    Stateful in exactly three ways, all documented: :attr:`last_draw` (the
    drone of the most recently generated flight, which is what a pool reports
    as the window's provenance), the antithetic offset pairing, and
    :attr:`stats` (the ``rps_max`` accept/reject tally).
    """

    def __init__(
        self,
        bundle: FitBundle,
        weights: Mapping[str, float],
        *,
        mean_shift: tuple[float, float] = (0.0, 0.0),
        mean_scale: tuple[float, float] = (1.0, 1.0),
        measurement_noise: bool = True,
        antithetic_offsets: bool = False,
        rps_max: float | None = None,
        phases: FlightPhaseRanges | None = None,
    ):
        if not weights:
            raise ValueError("rps.rigs is empty: name at least one rig to draw from")
        unknown = [name for name in weights if name not in bundle.names]
        if unknown:
            raise ValueError(
                f"unknown rig(s) {unknown} in rps.rigs; {bundle.root} offers {bundle.names}"
            )
        negative = {name: w for name, w in weights.items() if not (float(w) >= 0.0)}
        if negative:
            raise ValueError(f"rps.rigs weights must be non-negative, got {negative}")
        total = float(sum(float(w) for w in weights.values()))
        if total <= 0.0:
            raise ValueError(f"rps.rigs weights sum to {total}: nothing to draw")
        self.bundle = bundle
        self.names: tuple[str, ...] = tuple(weights)
        self.probs = np.array([float(weights[n]) for n in self.names]) / total
        self.mean_shift = (float(mean_shift[0]), float(mean_shift[1]))
        self.mean_scale = (float(mean_scale[0]), float(mean_scale[1]))
        self.measurement_noise = bool(measurement_noise)
        self.antithetic_offsets = bool(antithetic_offsets)
        if rps_max is not None and not (float(rps_max) > 0.0):
            raise ValueError(f"rps.rps_max must be a positive rotor speed, got {rps_max!r}")
        self.rps_max = None if rps_max is None else float(rps_max)
        self.phases = phases or FlightPhaseRanges()
        self.last_draw: RigDraw | None = None
        self._last_z: np.ndarray | None = None
        #: The ``rps_max`` tally over this source's lifetime: flights returned
        #: and flights thrown away by the cap. A pool (or a study) reads it to
        #: report how much of the drawn population the cap truncates.
        self.stats: dict[str, int] = {"flights": 0, "rejected": 0}

    @property
    def last_rig(self) -> str | None:
        """The rig the most recent flight was drawn from."""
        return None if self.last_draw is None else self.last_draw.rig

    # ── the drone of one flight ──
    def draw(self, rng: np.random.Generator) -> RigDraw:
        """Pick a rig by weight and move its hover level."""
        rig = str(self.names[int(rng.choice(len(self.names), p=self.probs))])
        if rig == POSTERIOR_RIG:
            assert self.bundle.posterior is not None  # guaranteed by bundle.names
            params = self.bundle.posterior.sample(rng)
            hover = float(np.mean(params.mu))
            idle = POSTERIOR_IDLE_REL * params.mu
            clip = (POSTERIOR_CLAMP_REL[0] * hover, POSTERIOR_CLAMP_REL[1] * hover)
        else:
            fit = self.bundle.fits[rig]
            params, idle, clip = fit.params, fit.idle_rps, fit.clip
        scale = float(rng.uniform(*self.mean_scale))
        shift = float(rng.uniform(*self.mean_shift))
        mu = params.mu * scale + shift
        # The idle level is a fraction of the hover level, so it travels with
        # the mean instead of sitting where the unshifted rig left it.
        ratio = float(np.mean(mu)) / float(np.mean(params.mu))
        # The ESC limits are speeds, so they scale and shift like the mean; the
        # floor cannot go below a stopped rotor.
        clip = (max(clip[0] * scale + shift, 0.0), clip[1] * scale + shift)
        return RigDraw(
            rig=rig,
            # Dropping the measurement process is what the "measurement" label
            # is for: it models sample-and-hold, quantisation and ESC jitter,
            # which a training stream renders audio from and should not hear.
            # Zeroing its std keeps the random stream identical.
            params=replace(
                params, mu=mu, sigma_e=params.sigma_e if self.measurement_noise else 0.0
            ),
            idle_rps=np.asarray(idle, dtype=np.float64) * ratio,
            clip=clip,
            mean_scale=scale,
            mean_shift=shift,
        )

    def _offset(self, params: Params, rng: np.random.Generator) -> np.ndarray | None:
        """The per-flight level offset, or ``None`` to let the sampler draw it.

        With ``antithetic_offsets`` the offset is drawn here as five standard
        normals and the NEXT flight reuses their negation, rescaled by its own
        rig's ``(s_c, s_r)``: each flight's offset keeps its marginal law, so no
        statistic changes in distribution, but the offsets of an even number of
        flights cancel instead of leaving an ``s / sqrt(N)`` error in the
        stream's pooled level.
        """
        if not self.antithetic_offsets:
            return None
        if self._last_z is None:
            z = rng.standard_normal(1 + NUM_ROTORS)
            self._last_z = z
        else:
            z = -self._last_z  # the paired flight: the first flight's draw, negated
            self._last_z = None
        return z[0] * params.s_c + z[1:] * params.s_r

    # ── the flight ──
    def flight_duration_s(self, rng: np.random.Generator) -> float:
        """A plausible whole-flight length (s): the fixed phases at their
        mid-range, plus one draw from ``cruise_s``.

        The airborne share then varies because :func:`wrap_airborne` draws the
        fixed phases itself, from the same rng.
        """
        names = (
            "pre_ground_s",
            "spinup_s",
            "warmup_s",
            "takeoff_s",
            "landing_s",
            "spindown_s",
            "post_ground_s",
        )
        fixed = sum(float(np.mean(getattr(self.phases, name))) for name in names)
        return fixed + float(rng.uniform(*self.phases.cruise_s))

    def flight(
        self, rng: np.random.Generator, fs: float, duration_s: float | None = None
    ) -> np.ndarray:
        """``(4, n)`` whole flight at ``fs`` Hz from one drawn drone.

        Records the drone in :attr:`last_draw`, so a caller can report which rig
        the windows it cuts from this flight came from.

        With ``rps_max`` set, a flight in which ANY rotor exceeds the cap
        anywhere — take-off overshoot included, not just the hover level — is
        thrown away whole and redrawn, up to :data:`MAX_RPS_REDRAWS` times. The
        whole attempt is redrawn, so a ``posterior`` mixture gets a fresh drone
        and a stored rig gets a fresh hover shift and a fresh realisation. This
        is REJECTION SAMPLING and therefore a TRUNCATION of the drawn
        population, not a clip of one flight: the accepted distribution is the
        drawn one conditioned on staying under the cap, and :attr:`stats`
        records what that costs.
        """
        for attempt in range(MAX_RPS_REDRAWS + 1):
            # A rejected attempt must not consume the antithetic pairing: the
            # pair is a property of two ACCEPTED flights.
            paired_z = None if self._last_z is None else self._last_z.copy()
            draw = self.draw(rng)
            total = self.flight_duration_s(rng) if duration_s is None else float(duration_s)
            params, clip = draw.params, draw.clip
            offset = self._offset(params, rng)

            def airborne(
                n: int,
                rng_: np.random.Generator,
                fs: float = fs,
                params: Params = params,
                offset: np.ndarray | None = offset,
                clip: tuple[float, float] = clip,
            ) -> np.ndarray:
                return params.sample_airborne(n, rng_, fs=fs, offset=offset, clip=clip)

            track = wrap_airborne(airborne, total, fs, rng, idle=draw.idle_rps, phases=self.phases)
            if self.rps_max is None or float(np.max(track)) <= self.rps_max:
                self.last_draw = replace(draw, redraws=attempt)
                self.stats["flights"] += 1
                return track
            self.stats["rejected"] += 1
            self._last_z = paired_z
        raise RuntimeError(
            f"rps.rps_max {self.rps_max} rejected {MAX_RPS_REDRAWS + 1} consecutive flights "
            f"from rigs {self.names}: the cap is below what this mixture flies at all. "
            "Raise rps_max, or drop the rigs whose hover level exceeds it."
        )


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _pair(cfg: Any, key: str, default: tuple[float, float]) -> tuple[float, float]:
    value = _get(cfg, key)
    if value is None:
        return default
    if len(value) != 2:
        raise ValueError(f"rps.{key} must be a [low, high] pair, got {list(value)!r}")
    return (float(value[0]), float(value[1]))


def build_from_config(cfg: Any) -> FittedTrajectorySource:
    """Build the source from one noise source's ``rps`` block.

    Every error a policy can make — a missing ``fits``, an empty mixture, an
    unknown rig name, a negative weight — is raised HERE, when the pool is
    built, and not in a DataLoader worker on some later window.
    """
    fits = _get(cfg, "fits")
    if not fits:
        raise ValueError(
            "rps.kind 'fitted_traj' requires rps.fits: a dload URI "
            "(dload:rps-traj-fits) or a directory with fits/<rig>.json"
        )
    rigs = _get(cfg, "rigs")
    if rigs is None:
        raise ValueError("rps.kind 'fitted_traj' requires rps.rigs: a rig -> weight mapping")
    weights = {str(name): float(weight) for name, weight in dict(rigs).items()}
    phases_cfg = _get(cfg, "phases")
    phases = (
        FlightPhaseRanges(
            **{str(k): (float(v[0]), float(v[1])) for k, v in dict(phases_cfg).items()}
        )
        if phases_cfg
        else None
    )
    return FittedTrajectorySource(
        load_bundle(str(fits)),
        weights,
        mean_shift=_pair(cfg, "mean_shift", (0.0, 0.0)),
        mean_scale=_pair(cfg, "mean_scale", (1.0, 1.0)),
        measurement_noise=bool(_get(cfg, "measurement_noise", True)),
        antithetic_offsets=bool(_get(cfg, "antithetic_offsets", False)),
        rps_max=(None if _get(cfg, "rps_max") is None else float(_get(cfg, "rps_max"))),
        phases=phases,
    )


__all__ = [
    "FITS_SUBDIR",
    "FITTED_KIND",
    "FLIGHT_KINDS",
    "MAX_RPS_REDRAWS",
    "POSTERIOR_CLAMP_REL",
    "POSTERIOR_FILE",
    "POSTERIOR_IDLE_REL",
    "POSTERIOR_RIG",
    "FitBundle",
    "FittedTrajectorySource",
    "RigDraw",
    "build_from_config",
    "load_bundle",
]
