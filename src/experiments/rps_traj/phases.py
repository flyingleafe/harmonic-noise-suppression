"""Whole-flight phases of a rotor-speed recording: landmarks, settled cruise, folds.

On the rigs recorded ground to ground a flight is ONE pass through six phases,
in one order::

    zero -> start-up -> standby -> take-off -> cruise -> landing -> zero

(all 13 flights with a measured channel on MD2, VID M100 and DREGON; campaign
doc ``docs/experiments/rps-flight-regimes.md``).  Nothing is hidden and nothing
branches, so the phases are READ OFF the speed level by the fixed rule in
:func:`landmarks` rather than inferred, and everything downstream is built on
those landmarks:

* :func:`settled_cruise` — the frozen airborne segments with the take-off and
  landing transients cut off.  The frozen rule's one-second erosion leaves the
  top of an MD2 take-off inside its "airborne" segment (3-8 % of the scored
  samples carrying 7-50 % of the squared prediction error of the cruise fit),
  so the cruise models of this campaign are fitted and scored on these.
* :func:`cruise_groups` and :func:`folds` — the held-out protocol: whole
  flights are held out, never parts of one (a rig with a single flight is
  split in two halves in time and says so).
* :func:`readouts` — the scalar whole-flight statistics a generated flight is
  compared on.
* :func:`load_flights` — :func:`~experiments.rps_traj.data.load_rig` plus the
  standby the DREGON command channel does not show: there the command reads 0
  while the rotors idle, the measured speed being ``max(idle, command)``, and
  the idle interval of the five command-only recordings was recovered from the
  audio (:mod:`experiments.rps_traj.standby_audio`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter1d

from experiments.rps_traj.cruise_api import NUM_ROTORS
from experiments.rps_traj.data import (
    MIN_RUN_S,
    RATE_HZ,
    Flight,
    _airborne_mask,
    _runs,
    airborne_segments,
    load_rig,
)

#: The rate (Hz) every cruise model of a rig is fitted and scored at.
#:
#: It is BELOW the common grid's rate on every rig, reached by block means
#: (:func:`block_mean`), for two reasons.  (1) The one-step log predictive
#: density rewards a model for learning the preprocessing: a zero-phase
#: Butterworth at 0.4 x rate leaves a 17 dB dip below the new Nyquist
#: frequency, which a free-form autoregression learns and a continuous-time
#: model cannot; on MD2 it was worth 0.47 nats/sample to a VAR over the
#: Gaussian structured model, against 0.07 on block means.  The common grid
#: itself carries that dip at 40-50 Hz wherever the telemetry is faster than
#: 100 Hz, so the fast rigs are scored at 50 Hz, where it folds away.  (2) The
#: rate must be one the telemetry is believable at for MORE than second
#: moments.  MD2 is logged at 29.41 Hz with holds (effective update 17 Hz);
#: at 25 Hz its squared prediction errors alternate sample to sample, which a
#: noise-level switch learns as 0.14 s "bursts", and at 12.5 Hz they do not.
#: DREGON's measured channel is a 45 Hz sample-and-hold.
CAMPAIGN_RATE_HZ: dict[str, float] = {"michaels": 12.5, "dregon": 25.0}
DEFAULT_RATE_HZ = 50.0

#: Settled cruise starts this long after the arrival at hover level and ends
#: this long before the landing (s).
SETTLE_AFTER_ARRIVAL_S = 3.0
CLEAR_BEFORE_LANDING_S = 2.5

#: A recording starts / ends on the ground when its rotor mean there is under
#: this fraction of the hover level; motors count as running above
#: :data:`MOVING_RPS` rev/s, and as stopped again after a landing once the
#: rotor mean is under :data:`STOPPED_FRACTION` of the level it left (MD2's
#: log can end at 2-5 rev/s, never reaching zero).
GROUND_FRACTION = 0.3
MOVING_RPS = 1.0
STOPPED_FRACTION = 0.1

#: Standby is the level the running motors dwell at longest before take-off;
#: a sample belongs to it within this tolerance, and the plateau must last
#: :data:`MIN_STANDBY_S`.
STANDBY_TOL_REL = 0.08
STANDBY_TOL_MIN_RPS = 1.0
MIN_STANDBY_S = 0.5

#: Arrival is the first time the rotor mean covers this fraction of the step
#: from standby to the settled hover level; a landing is timed where the rotor
#: mean last exceeds :data:`LANDING_FRACTION` of the level it leaves: inside
#: the first, steep drop on every rig (a pre-landing dip stays above it, and
#: the shelf MD2 and VID pass through on the way down sits below it, at about
#: 0.5 and 0.25 of the level left).
ARRIVAL_FRACTION = 0.9
LANDING_FRACTION = 0.75

#: Folds per rig at most; a rig with fewer flights is left one flight out.
MAX_FOLDS = 5

#: DREGON's idle speed (rev/s, room1 ``motors_measured`` plateau) and the
#: standby interval of the command-only recordings, seconds from the first
#: motor stamp, recovered from the audio by
#: ``python -m experiments.rps_traj.standby_audio`` (worst error against the
#: measured plateau on the five recordings that have one: 0.21 s).
DREGON_IDLE_RPS = 15.27
DREGON_AUDIO_STANDBY_S: dict[str, tuple[float, float]] = {
    "free-flight_nosource_room2": (1.092, 2.406),
    "hovering_nosource_room2": (1.451, 2.477),
    "rectangle_nosource_room2": (1.062, 2.354),
    "spinning_nosource_room2": (0.673, 2.324),
    "updown_nosource_room2": (1.381, 2.762),
}


def campaign_rate(rig: str) -> float:
    """The cruise fit-and-score rate of ``rig`` (Hz)."""
    return CAMPAIGN_RATE_HZ.get(rig, DEFAULT_RATE_HZ)


def block_mean(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """``(..., n)`` at ``fs_in`` to ``fs_out`` by an integer factor: each output
    sample is the mean of the input samples it spans (a ragged tail is
    dropped).  The gentlest decimation there is: a sinc roll-off 3.9 dB down
    at the new Nyquist frequency, no ringing, no parameter."""
    factor = int(round(float(fs_in) / float(fs_out)))
    if factor <= 1:
        return np.ascontiguousarray(x)
    n = (x.shape[-1] // factor) * factor
    return np.ascontiguousarray(
        x[..., :n].reshape(*x.shape[:-1], n // factor, factor).mean(axis=-1)
    )


# ─── loading ──────────────────────────────────────────────────────────────────


def with_recovered_standby(flight: Flight) -> Flight:
    """A DREGON command-only flight with its audio-recovered standby put back.

    From the recovered motor start on the rotors run at ``max(idle, command)``
    (what the measured channel shows on every recording that has both); before
    it the command is kept.  Any other flight is returned unchanged.
    """
    interval = DREGON_AUDIO_STANDBY_S.get(flight.flight)
    if interval is None or "motors_command" not in flight.source:
        return flight
    rps = np.array(flight.rps, dtype=np.float64)
    start = int(round(interval[0] * flight.fs))
    rps[:, start:] = np.maximum(rps[:, start:], DREGON_IDLE_RPS)
    return replace(flight, rps=rps, source=flight.source + "+audio_standby")


def load_flights(rig: str) -> list[Flight]:
    """Every real flight of ``rig`` on the common grid, standby recovered."""
    return [with_recovered_standby(f) for f in load_rig(rig)]


# ─── landmarks ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Landmarks:
    """Where one recording changes phase (sample indices at ``fs``).

    ``standby`` is ``[start, stop)`` and ``stop == takeoff``.  A recording that
    starts airborne has no start-up landmarks, one that ends airborne no
    landing ones; ``hover_in`` / ``hover_out`` are the per-rotor levels the
    take-off settles at and the landing leaves.
    """

    fs: float
    n: int
    hover: float
    starts_ground: bool
    ends_ground: bool
    motors_on: int | None = None
    standby: tuple[int, int] | None = None
    standby_level: np.ndarray | None = None
    takeoff: int | None = None
    arrival: int | None = None
    hover_in: np.ndarray | None = None
    landing: int | None = None
    motors_off: int | None = None
    hover_out: np.ndarray | None = None

    @property
    def whole(self) -> bool:
        """Ground to ground with every landmark found, in order."""
        marks = (self.motors_on, self.takeoff, self.arrival, self.landing, self.motors_off)
        if any(m is None for m in marks):
            return False
        ints = [int(m) for m in marks if m is not None]
        return ints == sorted(ints) and ints[1] < ints[2] < ints[3]


def bridge_nans(rps: np.ndarray) -> np.ndarray:
    """``rps`` with every non-finite sample linearly bridged, per rotor."""
    out = np.array(rps, dtype=np.float64)
    idx = np.arange(out.shape[-1])
    for row in out:
        bad = ~np.isfinite(row)
        if bad.any() and not bad.all():
            row[bad] = np.interp(idx[bad], idx[~bad], row[~bad])
    return np.nan_to_num(out, nan=0.0)


def _n(seconds: float, fs: float) -> int:
    return max(int(round(seconds * fs)), 1)


def _takeoff_marks(x: np.ndarray, ms: np.ndarray, fs: float, a0: int, a1: int) -> dict[str, Any]:
    running = np.flatnonzero(ms[: a0 + 1] > MOVING_RPS)
    on = int(running[0]) if running.size else 0
    stop = min(a0 + _n(13.0, fs), a1)
    lo = min(a0 + _n(3.0, fs), stop - 1)
    hover_in = np.median(x[:, lo:stop], axis=1)
    h0 = float(hover_in.mean())

    takeoff, standby, level = on, None, None
    pre = ms[on:a0]
    if pre.size >= _n(MIN_STANDBY_S, fs):
        width = max(0.5, 0.02 * h0)
        hist, edges = np.histogram(pre, bins=np.arange(0.0, float(pre.max()) + 2.0 * width, width))
        k = int(np.argmax(hist))
        centre = 0.5 * float(edges[k] + edges[k + 1])
        centre = float(np.median(pre[np.abs(pre - centre) <= width]))
        tol = max(STANDBY_TOL_MIN_RPS, STANDBY_TOL_REL * centre)
        inside = np.abs(ms[:a0] - centre) <= tol
        hits = np.flatnonzero(inside[on:])
        if hits.size:
            end = on + int(hits[-1]) + 1
            start = end
            while start > on and inside[start - 1]:
                start -= 1
            if end - start >= _n(MIN_STANDBY_S, fs) and centre > MOVING_RPS:
                standby = (start, end)
                level = np.median(x[:, max(end - _n(2.0, fs), start) : end], axis=1)
                takeoff = end
    base = float(level.mean()) if level is not None else 0.0
    rise = np.flatnonzero(ms[takeoff:a1] >= base + ARRIVAL_FRACTION * (h0 - base))
    return {
        "motors_on": on,
        "standby": standby,
        "standby_level": level,
        "takeoff": takeoff,
        "arrival": takeoff + int(rise[0]) if rise.size else None,
        "hover_in": hover_in,
    }


def _landing_marks(x: np.ndarray, ms: np.ndarray, fs: float, a0: int, a1: int) -> dict[str, Any]:
    lo = max(a1 - _n(13.0, fs), a0)
    hi = max(a1 - _n(3.0, fs), lo + 1)
    hover_out = np.median(x[:, lo:hi], axis=1)
    h1 = float(hover_out.mean())
    j = max(a1 - _n(5.0, fs), a0)
    above = np.flatnonzero(ms[j:] > LANDING_FRACTION * h1)
    landing = j + int(above[-1]) if above.size else None
    off = None
    if landing is not None:
        still = np.flatnonzero(ms[landing:] < STOPPED_FRACTION * h1)
        off = landing + int(still[0]) if still.size else None
    return {"landing": landing, "motors_off": off, "hover_out": hover_out}


def landmarks(rps: np.ndarray, fs: float = RATE_HZ) -> Landmarks | None:
    """The phase landmarks of one recording ``(4, n)``, or ``None`` when the
    frozen airborne rule finds no airborne segment in it.

    The airborne span is the frozen rule's own test (every rotor above half
    the 90th percentile of the rotor mean) on the gap-bridged track, runs of
    at least :data:`~experiments.rps_traj.data.MIN_RUN_S`, from the first to
    the last: a telemetry gap right after take-off (VID M100) must not move
    the take-off.
    """
    rps = np.atleast_2d(np.asarray(rps, dtype=np.float64))
    segs = airborne_segments(rps, fs)
    if not segs:
        return None
    x = bridge_nans(rps)
    m = x.mean(axis=0)
    ms = uniform_filter1d(m, _n(0.1, fs))
    n = int(m.size)
    least = int(round(MIN_RUN_S * fs))
    runs = [(s, e) for s, e in _runs(_airborne_mask(x)) if e - s >= least]
    if not runs:
        return None
    a0, a1 = runs[0][0], runs[-1][1]
    hover = float(np.median(np.concatenate([m[s] for s in segs])))
    edge = _n(0.3, fs)
    starts = bool(np.median(m[:edge]) < GROUND_FRACTION * hover)
    ends = bool(np.median(m[-edge:]) < GROUND_FRACTION * hover)
    marks: dict[str, Any] = {}
    if starts:
        marks.update(_takeoff_marks(x, ms, fs, a0, a1))
    if ends:
        marks.update(_landing_marks(x, ms, fs, a0, a1))
    return Landmarks(
        fs=float(fs), n=n, hover=hover, starts_ground=starts, ends_ground=ends, **marks
    )


def settled_cruise(
    rps: np.ndarray, fs: float = RATE_HZ, marks: Landmarks | None = None
) -> list[slice]:
    """The frozen airborne segments with the take-off and landing cut off.

    A segment keeps the frozen rule's bounds unless a transition is next to
    it: it then starts :data:`SETTLE_AFTER_ARRIVAL_S` after the arrival and
    ends :data:`CLEAR_BEFORE_LANDING_S` before the landing.  What is left
    shorter than the frozen minimum run is dropped.
    """
    rps = np.atleast_2d(np.asarray(rps, dtype=np.float64))
    marks = landmarks(rps, fs) if marks is None else marks
    if marks is None:
        return []
    least = int(round(MIN_RUN_S * fs))
    lo = None if marks.arrival is None else marks.arrival + _n(SETTLE_AFTER_ARRIVAL_S, fs)
    hi = None if marks.landing is None else marks.landing - _n(CLEAR_BEFORE_LANDING_S, fs)
    out = []
    for sl in airborne_segments(rps, fs):
        start = sl.start if lo is None else max(sl.start, lo)
        stop = sl.stop if hi is None else min(sl.stop, hi)
        if stop - start >= least:
            out.append(slice(start, stop))
    return out


# ─── the held-out protocol ────────────────────────────────────────────────────


def cruise_groups(flights: Sequence[Flight], rate_hz: float) -> dict[str, list[np.ndarray]]:
    """``{group: [settled-cruise segments (4, n) at rate_hz]}``, NOT demeaned.

    A group is what is held out together: one flight.  A rig with a single
    usable flight is split into two groups, the halves in time of its settled
    cruise (``<flight>#a`` and ``#b``), which is then the only split there is.
    """
    out: dict[str, list[np.ndarray]] = {}
    for flight in flights:
        rps = np.asarray(flight.rps, dtype=np.float64)
        segs = [rps[:, sl] for sl in settled_cruise(rps, flight.fs)]
        segs = [block_mean(s, flight.fs, rate_hz) for s in segs if np.isfinite(s).all()]
        if segs:
            out[flight.flight] = segs
    if len(out) == 1:
        ((name, segs),) = out.items()
        halves: dict[str, list[np.ndarray]] = {f"{name}#a": [], f"{name}#b": []}
        for seg in segs:
            mid = seg.shape[1] // 2
            halves[f"{name}#a"].append(np.ascontiguousarray(seg[:, :mid]))
            halves[f"{name}#b"].append(np.ascontiguousarray(seg[:, mid:]))
        return halves
    return out


def folds(groups: Sequence[str], strata: Sequence[str] | None = None) -> list[list[str]]:
    """Held-out groups per fold: leave-one-out up to :data:`MAX_FOLDS` groups,
    otherwise :data:`MAX_FOLDS` folds dealt round-robin over the groups in a
    fixed pseudo-random order (seed 0), stratum by stratum when given (DREGON:
    the measured and the command-only recordings spread evenly)."""
    names = sorted(groups)
    k = min(MAX_FOLDS, len(names))
    if k < 2:
        raise ValueError(f"need at least two groups to hold one out, got {names}")
    rng = np.random.default_rng(0)
    order = [names[i] for i in rng.permutation(len(names))]
    if strata is not None:
        label = dict(zip(groups, strata))
        order = sorted(order, key=lambda g: label[g])
    out: list[list[str]] = [[] for _ in range(k)]
    for i, name in enumerate(order):
        out[i % k].append(name)
    return out


#: The six phases, in order; :func:`phase_labels` returns indices into this.
PHASES = ("zero", "startup", "standby", "takeoff", "cruise", "landing")


def phase_labels(marks: Landmarks) -> np.ndarray:
    """``(n,)`` phase index of every sample of a recording (:data:`PHASES`).

    Start-up runs from motors-on to the standby plateau, take-off from its
    onset to the arrival, landing from 0.3 s before its mark to motors-off; a
    recording that starts or ends airborne is cruise there.
    """
    lab = np.zeros(marks.n, dtype=np.int64)
    start = 0
    if marks.motors_on is not None and marks.takeoff is not None:
        lab[marks.motors_on : marks.takeoff] = 1
        if marks.standby is not None:
            lab[marks.standby[0] : marks.standby[1]] = 2
        start = marks.takeoff if marks.arrival is None else marks.arrival
        lab[marks.takeoff : start] = 3
    stop = marks.n
    if marks.landing is not None:
        stop = max(marks.landing - _n(0.3, marks.fs), start)
        off = marks.n if marks.motors_off is None else marks.motors_off
        lab[stop:off] = 5
    lab[start:stop] = 4
    return lab


# ─── whole-flight read-outs ───────────────────────────────────────────────────

#: The scalar statistics of one whole flight, as :func:`readouts` keys.
READOUTS = (
    "startup_s",
    "startup_peak_rel",
    "standby_s",
    "standby_rel",
    "standby_spread_rel",
    "takeoff_rise_s",
    "takeoff_overshoot_rel",
    "airborne_s",
    "landing_fall_s",
    "landing_spread_rel",
    "midband_s",
)


def readouts(rps: np.ndarray, fs: float = RATE_HZ) -> dict[str, float]:
    """The whole-flight statistics of one recording; ``nan`` where its
    landmarks do not exist.  ``valid`` is 1.0 for one ground-to-ground pass
    in the right order.  Levels are relative to the hover level, the take-off
    overshoot to the step from standby, so rigs of different size compare.
    """
    out: dict[str, float] = dict.fromkeys(READOUTS, float("nan"))
    out["valid"] = 0.0
    marks = landmarks(rps, fs)
    if marks is None:
        return out
    x = bridge_nans(np.atleast_2d(np.asarray(rps, dtype=np.float64)))
    ms = uniform_filter1d(x.mean(axis=0), _n(0.1, fs))
    out["valid"] = float(marks.whole)
    mid = (ms > 0.55 * marks.hover) & (ms < 0.80 * marks.hover)
    out["midband_s"] = float(mid.sum()) / fs
    if marks.takeoff is not None and marks.motors_on is not None:
        base = 0.0
        if marks.standby is not None and marks.standby_level is not None:
            start, stop = marks.standby
            base = float(marks.standby_level.mean())
            out["startup_s"] = (start - marks.motors_on) / fs
            if start > marks.motors_on and base > 0.0:
                out["startup_peak_rel"] = float(ms[marks.motors_on : start].max()) / base
            out["standby_s"] = (stop - start) / fs
            out["standby_rel"] = base / marks.hover
            out["standby_spread_rel"] = float(np.ptp(marks.standby_level)) / marks.hover
        if marks.arrival is not None and marks.hover_in is not None:
            h0 = float(marks.hover_in.mean())
            out["takeoff_rise_s"] = (marks.arrival - marks.takeoff) / fs
            peak = float(ms[marks.arrival : marks.arrival + _n(3.0, fs)].max())
            out["takeoff_overshoot_rel"] = (peak - h0) / max(h0 - base, 1e-9)
    if marks.arrival is not None and marks.landing is not None:
        out["airborne_s"] = (marks.landing - marks.arrival) / fs
    if marks.landing is not None and marks.hover_out is not None:
        h1 = float(marks.hover_out.mean())
        stop = marks.n if marks.motors_off is None else marks.motors_off
        if marks.motors_off is not None:
            out["landing_fall_s"] = (marks.motors_off - marks.landing) / fs
        win = x[:, max(marks.landing - _n(0.5, fs), 0) : max(stop, marks.landing + 1)]
        out["landing_spread_rel"] = float(np.ptp(win, axis=0).max()) / h1
    return out


__all__ = [
    "ARRIVAL_FRACTION",
    "CAMPAIGN_RATE_HZ",
    "CLEAR_BEFORE_LANDING_S",
    "DEFAULT_RATE_HZ",
    "DREGON_AUDIO_STANDBY_S",
    "DREGON_IDLE_RPS",
    "LANDING_FRACTION",
    "MAX_FOLDS",
    "NUM_ROTORS",
    "PHASES",
    "READOUTS",
    "SETTLE_AFTER_ARRIVAL_S",
    "Landmarks",
    "block_mean",
    "bridge_nans",
    "campaign_rate",
    "cruise_groups",
    "folds",
    "landmarks",
    "load_flights",
    "phase_labels",
    "readouts",
    "settled_cruise",
    "with_recovered_standby",
]
