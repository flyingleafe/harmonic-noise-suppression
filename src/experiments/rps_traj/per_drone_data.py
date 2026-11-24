"""Per-drone train/validation split, portable training pack and validation scoring.

Commands (``PYTHONPATH=src python -m experiments.rps_traj.per_drone_data ...``)::

    prepare --root ROOT [--reserved-raw results/rps_traj/confirmation/raw]
    score   --root ROOT

``prepare`` (local: the published ``*-frames`` datasets pinned in ``dload.lock``
plus the two reserved VID ``.motors.npz`` files) splits every rig of
:data:`experiments.rps_traj.data.RIGS` into training and validation
recordings, freezes the generators' training input and the evaluator's
validation cells under ``ROOT/pack``.  ``score`` reads only ``ROOT/pack`` and
``ROOT/runs`` and writes ``ROOT/scores.json`` and ``ROOT/phase_scores.csv``.

Split (one fixed split per rig, never refitted, never informed by a score)
--------------------------------------------------------------------------
* **Unit.** The PHYSICAL recording (:func:`spectral_inputs.recording_id`):
  every NeuroBEM ``<flight>_seg_<k>`` segment of a flight is on the same side.
* **Usable.** Measured 4-rotor telemetry on the 100 Hz common grid
  (:func:`phases.load_flights`, MIXER rotor order).  DREGON's command-only
  recordings (``motors_command``) are excluded from every arm, training
  included.  A recording shorter than one 25 Hz sample is excluded.
* **MD2:** the two ``michaels-frames`` recordings FLY124/FLY125 train;
  the two ``michaels-test-frames`` FLY103/FLY108 validate, from the same airframe.
* **Seeded rigs** (``dregon``, ``neurobem_quad``, ``pitcn_quad``):
  split physical IDs without consulting speeds or landmarks; reserve
  ``max(1, round(0.2 n))`` with seed ``[2026, crc32('rig|per_drone|split')]``.
  Tracks without identifiable events remain on their assigned side.
* **vid_m100.** The four ``VID-frames`` flights train; the two reserved round
  flights (``confirm_flights.RESERVED``, converted from their raw
  ``.motors.npz`` exactly as ``VID-frames`` is: ``vid._frame_from_arrays`` ->
  ``data._flight_from_frame`` -> ``ROTOR_TO_MIXER``) are the validation set.
* **nanobench_cf21b.** The publisher's split, read from the pinned source
  definition ``nanobench.CSV_FILES``: ``train/*`` train, ``test/*`` (melon)
  validate.
* **blackbird_quad.** One recording, split WITHIN the flight: the first
  ``floor(0.7 T / 4) * 4`` common-grid samples train, the samples from
  :data:`GUARD_S` (16 s) later to the end validate.  Its rotor permutation
  (unpublished) is inferred by :func:`data.infer_rotor_to_mixer` on the
  training span only.

Every split is asserted: no shared physical recording (except Blackbird,
whose two spans are >= 16 s apart on the recording clock), no shared id
outside Blackbird, no bit-identical track on both sides, the VID and NanoBench
sets equal their definitions.

Worker pack (``pack/{rig}.train.npz`` / ``pack/{rig}.train.json``)
-----------------------------------------------------------------
``train.npz``: ``x`` ``(4, T)`` float64, the training tracks on the 100 Hz
common grid, MIXER order, NaN NOT bridged; ``offsets`` ``(k + 1,)`` int64.
``train.json``: ``rig``, ``fs_train_hz`` (100), ``fs_hz`` (25), ``n_draws``,
``train[i]`` = ``{index, id, recording, source, t0_s, span_100, offset,
samples_100, n, schedule_status, events, marks, code_samples}`` with ``n =
samples_100 // 4`` (length of ``regime_campaign._to_flight_rate``), ``events``
the 25 Hz ``motors_on/takeoff/landing`` of :func:`phases.landmarks` on that
grid, ``schedule_status`` ``"landmarks"`` or ``"coarse_initial"`` (no
landmarks: every event ``None``, schedule code 2 throughout).  ``targets[k]``
= ``{index, id, recording, n, events, marks, seed}``.  ``marks`` are
INDEX-ONLY (:func:`marks_index`): no measured speed level of any recording.
``train_sha256`` is the sha256 of ``train.npz``.

Evaluator-local files (never shipped to workers)
------------------------------------------------
``pack/{rig}.valid.npz``: ``t{k}`` ``(4, n)`` float64, measured target ``k``
at 25 Hz (NaN bridged, 4-sample block means), for plots.
``pack/{rig}.eval.npz``: ``t{k}__{phase}__obs`` ``(O, 4, n)`` and
``t{k}__{phase}__refs`` ``(O, D, 4, n)``.  ``pack/{rig}.valid.json``: targets
(index-only landmarks: no measured hover/trim of a target anywhere), cells, exclusions, training hover, split and
calibration provenance.  ``pack/manifest.json``: protocol, per-rig
:class:`CruiseMetric`, coverage, provenance.

Validation cells (the ``schedule_eval`` protocol, unchanged scorer)
-------------------------------------------------------------------
* **Short phases** (``startup``, ``standby``, ``takeoff``, ``landing``):
  ``schedule_eval._short_group`` with ONE donor per training physical
  recording (its first segment holding the phase).  A phase whose event the
  target does not have is ``absent`` and never scored (airborne-only
  recordings have no ground phases); one it has but cannot be supported is
  ``unscorable``.
* **Cruise**: a recording with an arrival uses ``schedule_eval._cruise_span``
  unchanged; a recording that STARTS AIRBORNE has no arrival, so its span is
  the longest frozen settled-cruise slice (:func:`phases.settled_cruise`,
  landing clearance kept) narrowed to its longest bridge-free run exactly as
  ``_cruise_span`` does (``span_rule`` in the support).  16 s at origins 0,
  0.5, 1; donors: one per training physical recording (its longest span).
  A rig with ONE training recording (Blackbird) uses up to four disjoint 16 s
  windows of that recording's training span as refs, the same for every
  origin, flagged ``refs_independent: false``.
* **Hover** (floor of the normalisers): median over training recordings of
  each recording's median landmark hover.
* **Cruise metric**: one :class:`CruiseMetric` per target rig, fitted on the
  OTHER rigs' TRAINING recordings only: per rig with at least two
  training recordings that yield a 16 s window, the lexicographically first
  such recording (the ``spectral_inputs.load_segments`` rule applied to the
  training split: same windows, same cap).  A single-recording rig
  (Blackbird) contributes none, as in ``load_segments``.
* **Score**: ``schedule_eval._score`` (``score_phase``, origin-averaged), on
  windows cut from each arm's draws at the frozen starts (sample 0 is sample
  0 of the target, no shift, no padding); segments are averaged per physical
  recording with equal weight.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from experiments.rps_traj import data
from experiments.rps_traj import regime_campaign as rc
from experiments.rps_traj import schedule_eval as se
from experiments.rps_traj import spectral_inputs as si
from experiments.rps_traj import unified_experiment as ue
from experiments.rps_traj.phases import Landmarks, load_flights, settled_cruise
from experiments.rps_traj.spectral_validation import CruiseMetric

#: Arms scored, in report order (``phase_scores.csv`` columns).
ARMS = ("regulated_bridge", "template", "phasevar2", "phasevar4", "msvar3", "hsmm3")
FS = se.FS
GRID_HZ = float(data.RATE_HZ)
DECIM = int(round(GRID_HZ / FS))
N_DRAWS = se.N_DRAWS
CRUISE_N = se.CRUISE_N
FOLD = "per_drone"
VALID_FRACTION = 0.2
MIN_TRAIN = 2
WITHIN_FLIGHT_RIG = "blackbird_quad"
WITHIN_FLIGHT_TRAIN_FRACTION = 0.7
#: Clock gap between the training and validation spans of one recording (s).
GUARD_S = si.WINDOW_S
SEEDED_RIGS = ("dregon", "neurobem_quad", "pitcn_quad")
VID_RIG = "vid_m100"
NANOBENCH_RIG = "nanobench_cf21b"
DEFAULT_RESERVED_RAW = Path("results/rps_traj/confirmation/raw")
DEFAULT_ROOT = Path("results/rps_traj/per_drone")
#: Source modules whose code defines the pack (hashed into the manifest).
SOURCE_MODULES = (
    "per_drone_data.py",
    "schedule_eval.py",
    "schedule_score.py",
    "spectral_inputs.py",
    "spectral_validation.py",
    "phases.py",
    "data.py",
    "phase_dynamics.py",
    "multiscale.py",
    "regime_campaign.py",
    "unified_experiment.py",
    "scores.py",
)


# ─── records ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Piece:
    """One usable track: a published frame or (Blackbird) a span of it.

    ``flight`` is on the 100 Hz common grid, MIXER order, NaN kept;
    ``flight.t0`` is the piece's first sample on the recording's own clock;
    ``span`` is ``[lo, hi)`` on the published frame's common grid.
    """

    id: str
    recording: str
    flight: data.Flight
    span: tuple[int, int]

    @property
    def duration_s(self) -> float:
        return self.flight.rps.shape[1] / GRID_HZ


def marks_index(m: Landmarks) -> dict[str, Any]:
    """The landmark INDICES of a recording: no speed level of it."""
    out: dict[str, Any] = {
        "fs": float(m.fs),
        "n": int(m.n),
        "starts_ground": bool(m.starts_ground),
        "ends_ground": bool(m.ends_ground),
        "standby": None if m.standby is None else [int(m.standby[0]), int(m.standby[1])],
    }
    for k in ("motors_on", "takeoff", "arrival", "landing", "motors_off"):
        v = getattr(m, k)
        out[k] = None if v is None else int(v)
    return out


def _events(m: Landmarks | None) -> dict[str, int | None]:
    return dict.fromkeys(ue.EVENTS) if m is None else ue._events(m)


def _track_sha(rps: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(rps, dtype=np.float64).tobytes()).hexdigest()


def _vid_reserved(path: Path, name: str) -> data.Flight:
    """A reserved VID flight from its raw ``.motors.npz``, converted as ``VID-frames`` is."""
    from data_processing.sources import vid

    with np.load(path) as raw:
        frame = vid._frame_from_arrays(name, dict(raw), "VID")
    f = data._flight_from_frame(frame, "VID-frames", ("rps",), VID_RIG)
    if f is None or f.flight != name or f.n_rotors != data.NUM_ROTORS:
        raise ValueError(f"{path}: no 4-rotor measured trajectory for {name}")
    rows = list(data.ROTOR_TO_MIXER[VID_RIG])
    return replace(f, rps=np.ascontiguousarray(f.rps[rows]), source=f"{path.name}:rps")


def _load(rig: str, reserved_raw: Path) -> tuple[list[Piece], list[dict], dict[str, Any]]:
    """Every usable whole recording of ``rig`` (Blackbird: published rotor order)."""
    info: dict[str, Any] = {}
    if rig == WITHIN_FLIGHT_RIG:
        flights = [
            f
            for ds, keys in data.RIG_SOURCES[rig]
            for f in data._flights_from_dataset(ds, keys, data._rig_id(rig, ds), match_rig=rig)
            if f.n_rotors == data.NUM_ROTORS
        ]
    else:
        flights = load_flights(rig)
    if rig == VID_RIG:
        from experiments.rps_traj.confirm_flights import RESERVED

        clash = sorted({f.flight for f in flights} & set(RESERVED))
        if clash:
            raise AssertionError(f"reserved VID flights {clash} are already in VID-frames")
        info["reserved_raw"] = {}
        for name in RESERVED:
            path = reserved_raw / f"{name}.motors.npz"
            flights.append(_vid_reserved(path, name))
            info["reserved_raw"][name] = {"path": str(path), "sha256": ue._sha256(path)}
    excluded, pieces = [], []
    for f in sorted(flights, key=lambda f: f.flight):
        rec = si.recording_id(rig, f.flight)
        if si.COMMAND_SOURCE in f.source:
            excluded.append(
                {
                    "id": f.flight,
                    "recording": rec,
                    "source": f.source,
                    "reason": "command-only rotor speed (not measured): excluded from every arm",
                }
            )
            continue
        if f.rps.shape[1] < DECIM:
            excluded.append(
                {
                    "id": f.flight,
                    "recording": rec,
                    "source": f.source,
                    "reason": f"{f.rps.shape[1]} common-grid samples: shorter than one 25 Hz sample",
                }
            )
            continue
        pieces.append(Piece(f.flight, rec, f, (0, int(f.rps.shape[1]))))
    info["frames_loaded"] = len(flights)
    return pieces, excluded, info


# ─── split ────────────────────────────────────────────────────────────────────


def seeded_validation(rig: str, eligible: Sequence[str]) -> list[str]:
    """The validation recordings of a seeded rig: ``max(1, round(0.2 n))``,
    leaving at least :data:`MIN_TRAIN` when ``n >= 3``; none when ``n < 2``."""
    names = sorted(set(eligible))
    n = len(names)
    if n < 2:
        return []
    k = max(1, round(VALID_FRACTION * n))
    if n >= MIN_TRAIN + 1:
        k = min(k, n - MIN_TRAIN)
    rng, _ = se._rng(rig, FOLD, "split")
    return sorted(names[int(i)] for i in rng.permutation(n)[:k])


def within_flight_cut(n100: int) -> tuple[int, int]:
    """``(train_stop, valid_start)`` on the 100 Hz grid, both multiples of the
    25 Hz block, :data:`GUARD_S` apart."""
    a = int(n100 * WITHIN_FLIGHT_TRAIN_FRACTION) // DECIM * DECIM
    b = a + int(round(GUARD_S * GRID_HZ))
    if a <= 0 or b >= n100:
        raise ValueError(f"recording of {n100} samples cannot hold a {GUARD_S} s guard")
    return a, b


def _nanobench_split() -> tuple[set[str], set[str]]:
    from data_processing.sources import nanobench

    split: dict[str, set[str]] = {"train": set(), "test": set()}
    for rel in nanobench.CSV_FILES:
        part, name = rel.split("/", 1)
        split[part].add(name.removesuffix(".csv"))
    return split["train"], split["test"]


def _split(rig: str, pieces: Sequence[Piece]) -> tuple[list[Piece], list[Piece], dict[str, Any]]:
    recs = sorted({p.recording for p in pieces})
    if rig == WITHIN_FLIGHT_RIG:
        if len(pieces) != 1:
            raise AssertionError(
                f"{rig}: within-flight split needs exactly one track, got {len(pieces)}"
            )
        (p,) = pieces
        f = p.flight
        a, b = within_flight_cut(f.rps.shape[1])
        train_f = replace(f, rps=np.ascontiguousarray(f.rps[:, :a]))
        perm = data.infer_rotor_to_mixer(data._airborne_block([train_f]))
        rows = list(perm)
        tr = Piece(
            p.id, p.recording, replace(train_f, rps=np.ascontiguousarray(train_f.rps[rows])), (0, a)
        )
        va = Piece(
            p.id,
            p.recording,
            replace(f, rps=np.ascontiguousarray(f.rps[rows, b:]), t0=f.t0 + b / GRID_HZ),
            (b, int(f.rps.shape[1])),
        )
        meta = {
            "rule": "within-flight: earlier span trains, later span validates",
            "train_fraction": WITHIN_FLIGHT_TRAIN_FRACTION,
            "guard_s": GUARD_S,
            "train_span_100": [0, a],
            "valid_span_100": [b, int(f.rps.shape[1])],
            "rotor_to_mixer_train_only": list(perm),
            "train_recordings": [p.recording],
            "valid_recordings": [p.recording],
            "independence": "validation is a later span of the training recording: not an independent flight",
        }
        return [tr], [va], meta
    if rig == VID_RIG:
        from experiments.rps_traj.confirm_flights import RESERVED

        valid = set(RESERVED)
        rule = "VID-frames flights train; the two reserved round flights validate"
        if not valid <= set(recs) or len(set(recs) - valid) != 4:
            raise AssertionError(f"{rig}: expected 4 frames + reserved {sorted(valid)}, got {recs}")
    elif rig == "michaels":
        valid = {p.recording for p in pieces if p.flight.source.startswith("michaels-test-frames:")}
        if valid != {"FLY103", "FLY108"} or set(recs) - valid != {"FLY124", "FLY125"}:
            raise ValueError("unexpected MD2 source train/test inventory")
        rule = "MD2 source split: FLY124/FLY125 train; FLY103/FLY108 validate"
    elif rig == NANOBENCH_RIG:
        train_ids, test_ids = _nanobench_split()
        valid = test_ids
        rule = "publisher split (nanobench.CSV_FILES): train/* train, test/* validate"
        if set(recs) != train_ids | test_ids:
            raise AssertionError(
                f"{rig}: loaded {recs} != published {sorted(train_ids | test_ids)}"
            )
    elif rig in SEEDED_RIGS:
        eligible = recs
        valid = set(seeded_validation(rig, eligible))
        rule = (
            f"seeded: k = max(1, round({VALID_FRACTION} n)) of all n physical recordings, "
            f"at most n - {MIN_TRAIN} (n >= 3); seed [2026, crc32('{rig}|{FOLD}|split')]"
        )
    else:
        raise KeyError(f"no split rule for rig {rig!r}")
    train = [p for p in pieces if p.recording not in valid]
    validp = [p for p in pieces if p.recording in valid]
    meta = {
        "rule": rule,
        "train_recordings": sorted({p.recording for p in train}),
        "valid_recordings": sorted({p.recording for p in validp}),
    }
    if rig in SEEDED_RIGS:
        meta["eligible_recordings"] = eligible
        meta["seed"] = se._rng(rig, FOLD, "split")[1]
    return train, validp, meta


def assert_disjoint(rig: str, train: Sequence[Piece], valid: Sequence[Piece]) -> dict[str, Any]:
    """Physical recordings, ids, clocks and track contents of the two sides."""
    tr_recs = {p.recording for p in train}
    va_recs = {p.recording for p in valid}
    shared = tr_recs & va_recs
    if shared and rig != WITHIN_FLIGHT_RIG:
        raise AssertionError(f"{rig}: physical recordings on both sides: {sorted(shared)}")
    if rig != WITHIN_FLIGHT_RIG and {p.id for p in train} & {p.id for p in valid}:
        raise AssertionError(f"{rig}: ids on both sides")
    gaps = []
    guard = int(round(GUARD_S * GRID_HZ))
    for p in train:
        for q in valid:
            if p.recording != q.recording:
                continue
            gap_s = max(
                q.flight.t0 - (p.flight.t0 + p.duration_s),
                p.flight.t0 - (q.flight.t0 + q.duration_s),
            )
            gap_n = max(q.span[0] - p.span[1], p.span[0] - q.span[1])
            if gap_n < guard or gap_s < GUARD_S - 0.5 / GRID_HZ:
                raise AssertionError(
                    f"{rig}/{p.recording}: spans {p.span} {q.span} closer than {GUARD_S} s"
                )
            gaps.append(
                {"recording": p.recording, "gap_s": float(gap_s), "gap_samples_100": int(gap_n)}
            )
    tr_sha = {_track_sha(p.flight.rps) for p in train}
    if any(_track_sha(q.flight.rps) in tr_sha for q in valid):
        raise AssertionError(f"{rig}: a validation track is bit-identical to a training track")
    return {
        "shared_recordings": sorted(shared),
        "clock_gaps": gaps,
        "checks": "recordings disjoint (within-flight rig: >= guard apart), ids disjoint, no identical track",
    }


# ─── validation cells ─────────────────────────────────────────────────────────


def _narrow(rec: se.Rec, a: int, b: int) -> tuple[int, int, bool]:
    """``schedule_eval._cruise_span``'s bridge-free narrowing of ``[a, b)``."""
    ok = ~rec.bridged[a:b]
    edges = np.flatnonzero(np.diff(np.concatenate([[0], ok.astype(np.int8), [0]])))
    runs = list(zip(edges[::2], edges[1::2]))
    if runs:
        lo, hi = max(runs, key=lambda r: r[1] - r[0])
        if hi - lo >= CRUISE_N:
            return a + int(lo), a + int(hi), False
    return a, b, True


def cruise_span(rec: se.Rec) -> tuple[int, int, bool, str] | str:
    """``(a, b, has_bridged, rule)`` of a recording's settled cruise, or why not."""
    m = rec.marks
    if m.arrival is not None:
        s = se._cruise_span(rec)
        if s is None:
            return f"settled cruise after the arrival shorter than {CRUISE_N} samples"
        return (*s, "schedule_eval")
    if m.starts_ground:
        return "starts on the ground but never arrives at hover level: no settled cruise"
    slices = settled_cruise(rec.x, FS, m)
    if not slices:
        return "starts airborne; no settled-cruise slice of the frozen minimum run"
    sl = max(slices, key=lambda s: s.stop - s.start)
    if sl.stop - sl.start < CRUISE_N:
        return f"starts airborne; longest settled-cruise slice {sl.stop - sl.start} samples < {CRUISE_N}"
    return (*_narrow(rec, int(sl.start), int(sl.stop)), "airborne_settled")


def disjoint_starts(a: int, b: int) -> list[int]:
    """Starts of the disjoint 16 s windows of ``[a, b)``, remainder split between
    the ends, at most ``MAX_WINDOWS_PER_RECORDING`` evenly spaced."""
    k = (b - a) // CRUISE_N
    lo = a + ((b - a) - k * CRUISE_N) // 2
    starts = [lo + j * CRUISE_N for j in range(k)]
    if len(starts) > si.MAX_WINDOWS_PER_RECORDING:
        idx = np.round(np.linspace(0, len(starts) - 1, si.MAX_WINDOWS_PER_RECORDING)).astype(int)
        starts = [starts[i] for i in idx]
    return starts


def _phase_present(m: Landmarks, phase: str) -> bool:
    if phase == "startup":
        return m.motors_on is not None and m.takeoff is not None
    if phase == "standby":
        return m.standby is not None
    return getattr(m, phase) is not None


def _cruise_group(
    rig: str, target: se.Rec, donors: Mapping[str, Sequence[se.Rec]], hover: float
) -> se.Group | str:
    own = cruise_span(target)
    if isinstance(own, str):
        return f"target: {own}"
    pool = []
    for recording, recs in donors.items():
        spans = [(r, s) for r in recs if not isinstance(s := cruise_span(r), str)]
        if spans:
            r, s = max(spans, key=lambda rs: rs[1][1] - rs[1][0])
            pool.append((recording, r, s))
    starts = se._starts(own[0], own[1])
    obs = tuple(se._frozen(target.x[:, t : t + CRUISE_N]) for t in starts)

    def bridged(rec: se.Rec, ts: Sequence[int]) -> int:
        return int(sum(rec.bridged[t : t + CRUISE_N].sum() for t in ts))

    support: dict[str, Any] = {
        "event": "settled_cruise",
        "samples": CRUISE_N,
        "seconds": CRUISE_N / FS,
        "origins": list(se.ORIGINS),
        "span_rule": own[3],
        "target_span": [own[0], own[1]],
        "target_starts": starts,
        "target_span_has_bridged": own[2],
        "target_bridged_samples": bridged(target, starts),
        "donors_without_phase": sorted(set(donors) - {d for d, _, _ in pool}),
    }
    if len(pool) >= se.MIN_DONORS:
        donor_starts = [se._starts(s[0], s[1]) for _, _, s in pool]
        refs = tuple(
            se._frozen(
                np.stack(
                    [r.x[:, ds[k] : ds[k] + CRUISE_N] for (_, r, _), ds in zip(pool, donor_starts)]
                )
            )
            for k in range(len(se.ORIGINS))
        )
        support |= {
            "refs_independent": True,
            "donors": [r.name for _, r, _ in pool],
            "donor_recordings": [d for d, _, _ in pool],
            "donor_span_rules": [s[3] for _, _, s in pool],
            "donor_spans": [[s[0], s[1]] for _, _, s in pool],
            "donor_span_has_bridged": [s[2] for _, _, s in pool],
            "donor_bridged_samples": [bridged(r, ds) for (_, r, _), ds in zip(pool, donor_starts)],
        }
    elif len(donors) == 1 and pool:
        ((recording, r, s),) = pool
        ws = disjoint_starts(s[0], s[1])
        if len(ws) < se.MIN_DONORS:
            return f"single training recording holds {len(ws)} disjoint {CRUISE_N}-sample windows < {se.MIN_DONORS}"
        block = se._frozen(np.stack([r.x[:, w : w + CRUISE_N] for w in ws]))
        refs = tuple(block for _ in se.ORIGINS)
        support |= {
            "refs_independent": False,
            "refs_note": "disjoint windows of the rig's single training recording: not independent flights",
            "donors": [r.name] * len(ws),
            "donor_recordings": [recording] * len(ws),
            "donor_span_rules": [s[3]],
            "donor_spans": [[s[0], s[1]]],
            "donor_span_has_bridged": [s[2]],
            "donor_window_starts": ws,
            "donor_bridged_samples": [bridged(r, [w]) for w in ws],
        }
    else:
        return f"{len(pool)} training recordings with >= {CRUISE_N} settled-cruise samples < {se.MIN_DONORS}"
    return se.Group(
        rig, FOLD, target.name, "cruise", hover, tuple(starts), CRUISE_N, obs, refs, support
    )


def _short_group(
    rig: str, target: se.Rec, donors: Mapping[str, Sequence[se.Rec]], phase: str, hover: float
) -> se.Group | str:
    chosen = [
        (recording, next((r for r in recs if se._window(r.marks, phase) is not None), recs[0]))
        for recording, recs in donors.items()
    ]
    g = se._short_group(rig, FOLD, target, [r for _, r in chosen], phase, hover)
    if isinstance(g, str):
        return g
    g.support["refs_independent"] = True
    g.support["donor_recordings"] = [d for d, r in chosen if se._window(r.marks, phase) is not None]
    return g


def _cells(
    rig: str, target: se.Rec, donors: Mapping[str, Sequence[se.Rec]], hover: float
) -> tuple[list[se.Group], list[dict[str, str]]]:
    groups, skipped = [], []
    for phase in se.PHASES:
        if phase == "cruise":
            g = _cruise_group(rig, target, donors, hover)
        elif not _phase_present(target.marks, phase):
            skipped.append(
                {"phase": phase, "kind": "absent", "reason": f"target recording has no {phase}"}
            )
            continue
        else:
            g = _short_group(rig, target, donors, phase, hover)
        if isinstance(g, str):
            skipped.append({"phase": phase, "kind": "unscorable", "reason": g})
        else:
            groups.append(g)
    return groups, skipped


# ─── cruise metric (off-rig, training recordings only) ───────────────────────


def _calibration_source(
    rig: str, train: Sequence[Piece]
) -> tuple[list[dict], list[np.ndarray], dict]:
    """``load_segments``' calibration rule on the training split of ``rig``."""
    stats: dict[str, int] = dict.fromkeys(
        ("settled_slices", "nonfinite_samples", "finite_runs", "short_runs"), 0
    )
    by_rec: dict[str, list[Piece]] = {}
    for p in train:
        by_rec.setdefault(p.recording, []).append(p)
    found: list[tuple[str, list[si._Window]]] = []
    for recording in sorted(by_rec):
        runs = [
            r
            for p in sorted(by_rec[recording], key=lambda p: p.id)
            for r in si._runs(p.flight, stats)
        ]
        if runs:
            found.append((recording, si._windows(runs)))
        if len(found) == 2:
            break
    if len(found) < 2:
        return (
            [],
            [],
            {
                "rig": rig,
                "calibration": None,
                "reason": "fewer than two training recordings with a 16 s window (load_segments rule)",
            },
        )
    recording, windows = found[0]
    kept = si._cap(windows)
    rows = [
        {
            "rig": rig,
            "flight": recording,
            "segment": w.segment,
            "start_s": round(float(w.start_s), 4),
            "source": w.source,
            "calibration": True,
        }
        for w in kept
    ]
    return rows, [w.x for w in kept], {"rig": rig, "calibration": recording, "windows": len(kept)}


def _metrics(
    splits: Mapping[str, tuple[list[Piece], list[Piece]]],
) -> tuple[dict[str, CruiseMetric], dict[str, Any]]:
    sources = {rig: _calibration_source(rig, train) for rig, (train, _) in splits.items()}
    held = {(rig, p.recording) for rig, (_, valid) in splits.items() for p in valid}
    metrics, meta = {}, {"sources": {rig: s[2] for rig, s in sources.items()}, "rigs": {}}
    for rig in splits:
        rows = [r for q, s in sources.items() if q != rig for r in s[0]]
        waves = [w for q, s in sources.items() if q != rig for w in s[1]]
        used = sorted({(r["rig"], r["flight"]) for r in rows})
        if any(q == rig for q, _ in used) or any(u in held for u in used):
            raise AssertionError(
                f"{rig}: metric calibration includes its own rig or a validation recording"
            )
        m = CruiseMetric.fit(np.stack(waves), rows)
        metrics[rig] = m
        meta["rigs"][rig] = {
            "calibration_recordings": [list(u) for u in used],
            "calibration_windows": len(rows),
            "level_scale_rps": m.level_scale_rps,
            "spectrum_scale": m.spectrum_scale,
            "band_power": m.band_power.tolist(),
        }
    return metrics, meta


# ─── prepare ──────────────────────────────────────────────────────────────────


def _default(x: Any) -> Any:
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"not JSON serialisable: {type(x).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=1, allow_nan=False, default=_default) + "\n")
    tmp.replace(path)


def paths(root: Path, rig: str) -> dict[str, Path]:
    pack = root / "pack"
    return {
        k: pack / f"{rig}.{k}.{ext}"
        for k, ext in (("train", "npz"), ("valid", "npz"), ("eval", "npz"))
    } | {"train_meta": pack / f"{rig}.train.json", "valid_meta": pack / f"{rig}.valid.json"}


def run_paths(root: Path, rig: str, arm: str) -> dict[str, Path]:
    d = root / "runs" / rig
    return {"fit": d / f"{arm}.fit.json", "draws": d / f"{arm}.draws.npz"}


def _write_rig(
    root: Path,
    rig: str,
    train: Sequence[Piece],
    valid: Sequence[Piece],
    excluded: list[dict],
    split_meta: dict[str, Any],
    metric_meta: dict[str, Any],
) -> dict[str, Any]:
    p = paths(root, rig)
    # Training tracks (worker input).
    rows, xs, offset = [], [], 0
    donors: dict[str, list[se.Rec]] = {}
    for i, piece in enumerate(sorted(train, key=lambda q: (q.recording, q.id))):
        f = piece.flight
        n = f.rps.shape[1] // DECIM
        rec = se._record(f)
        if rec is not None:
            if rec.marks.n != n:
                raise AssertionError(f"{rig}/{piece.id}: 25 Hz length {rec.marks.n} != {n}")
            donors.setdefault(piece.recording, []).append(rec)
        ev = _events(None if rec is None else rec.marks)
        codes = ue.schedule_codes(n, ev)
        rows.append(
            {
                "index": i,
                "id": piece.id,
                "recording": piece.recording,
                "source": f.source,
                "t0_s": float(f.t0),
                "span_100": list(piece.span),
                "offset": offset,
                "samples_100": int(f.rps.shape[1]),
                "n": n,
                "schedule_status": "coarse_initial" if rec is None else "landmarks",
                "events": ev,
                "marks": None if rec is None else marks_index(rec.marks),
                "code_samples": np.bincount(codes, minlength=4).tolist(),
            }
        )
        xs.append(np.asarray(f.rps, dtype=np.float64))
        offset += f.rps.shape[1]
    if not donors:
        raise AssertionError(f"{rig}: no training recording has landmarks")
    hover = float(
        np.median(
            np.asarray([np.median([r.marks.hover for r in recs]) for recs in donors.values()])
        )
    )
    ue._write_npz(
        p["train"],
        {
            "x": np.concatenate(xs, axis=1),
            "offsets": np.array([r["offset"] for r in rows] + [offset], dtype=np.int64),
        },
    )
    train_sha = ue._sha256(p["train"])

    # Validation targets and cells (evaluator-local).
    targets, vtargets, target_excl, cell_excl = [], [], [], []
    valid_arrays: dict[str, np.ndarray] = {}
    eval_arrays: dict[str, np.ndarray] = {}
    for piece in sorted(valid, key=lambda q: (q.recording, q.id)):
        rec = se._record(piece.flight)
        if rec is None:
            # Preserve the ID and trace even if the frozen score has no
            # qualifying window. No unavailable landmark goes to a generator.
            k = len(targets)
            observed = rc._to_flight_rate(piece.flight.rps, GRID_HZ)
            n = observed.shape[1]
            _, seed = se._rng(rig, FOLD, piece.id, "draws")
            valid_arrays[f"t{k}"] = observed
            target = {
                "index": k,
                "id": piece.id,
                "recording": piece.recording,
                "n": n,
                "events": _events(None),
                "marks": None,
                "seed": seed,
            }
            targets.append(target)
            vtargets.append(
                {
                    **target,
                    "source": piece.flight.source,
                    "t0_s": float(piece.flight.t0),
                    "span_100": list(piece.span),
                    "code_samples": [0, 0, n, 0],
                    "cells": [],
                }
            )
            target_excl.append(
                {
                    "id": piece.id,
                    "recording": piece.recording,
                    "reason": "no landmarks: shown but not scored",
                }
            )
            continue
        k = len(targets)
        ev = _events(rec.marks)
        codes = ue.schedule_codes(rec.marks.n, ev)
        _, seed = se._rng(rig, FOLD, piece.id, "draws")
        groups, skipped = _cells(rig, rec, donors, hover)
        cell_excl += [
            {"target": k, "id": piece.id, "recording": piece.recording, **s} for s in skipped
        ]
        valid_arrays[f"t{k}"] = np.asarray(rec.x, dtype=np.float64)
        cells = []
        for g in groups:
            eval_arrays[f"t{k}__{g.phase}__obs"] = np.stack(g.obs)
            eval_arrays[f"t{k}__{g.phase}__refs"] = np.stack(g.refs)
            cells.append(
                {
                    "phase": g.phase,
                    "starts": [int(s) for s in g.starts],
                    "n": int(g.n),
                    "hover": g.hover,
                    "n_refs": int(len(g.refs[0])),
                    "support": g.support,
                }
            )
        targets.append(
            {
                "index": k,
                "id": piece.id,
                "recording": piece.recording,
                "n": int(rec.marks.n),
                "events": ev,
                "marks": marks_index(rec.marks),
                "seed": seed,
            }
        )
        vtargets.append(
            {
                **targets[-1],
                "source": piece.flight.source,
                "t0_s": float(piece.flight.t0),
                "span_100": list(piece.span),
                "code_samples": np.bincount(codes, minlength=4).tolist(),
                "cells": cells,
            }
        )
    if not targets:
        raise AssertionError(f"{rig}: no validation target with landmarks")
    if len(eval_arrays) == 0:
        eval_arrays["_empty"] = np.zeros(0)
    ue._write_npz(p["valid"], valid_arrays)
    ue._write_npz(p["eval"], eval_arrays)

    train_meta = {
        "rig": rig,
        "fs_train_hz": GRID_HZ,
        "fs_hz": FS,
        "n_draws": N_DRAWS,
        "train": rows,
        "targets": targets,
        "train_sha256": train_sha,
        "schedule_events": list(ue.EVENTS),
        "schedule_codes": ue.CODES,
        "marks_note": "index-only landmarks (25 Hz): no measured speed level of any recording",
    }
    _write_json(p["train_meta"], train_meta)
    validation_recordings: dict[str, list[str]] = {}
    for q in sorted(valid, key=lambda q: (q.recording, q.id)):
        validation_recordings.setdefault(q.recording, []).append(q.id)
    by_phase = {
        ph: sum(any(c["phase"] == ph for c in t["cells"]) for t in vtargets) for ph in se.PHASES
    }
    coverage = {
        "train_tracks": len(rows),
        "train_recordings": len({r["recording"] for r in rows}),
        "train_coarse_initial": [r["id"] for r in rows if r["schedule_status"] == "coarse_initial"],
        "train_seconds": offset / GRID_HZ,
        "valid_tracks": len(valid),
        "valid_recordings": len(validation_recordings),
        "targets": len(targets),
        "target_cells_by_phase": by_phase,
        "excluded": excluded,
        "target_exclusions": target_excl,
        "cells_absent": sum(c["kind"] == "absent" for c in cell_excl),
        "cells_unscorable": sum(c["kind"] == "unscorable" for c in cell_excl),
    }
    valid_meta = {
        "rig": rig,
        "fs_hz": FS,
        "n_draws": N_DRAWS,
        "fold": FOLD,
        "train_hover_rps": hover,
        "train_sha256": train_sha,
        "train_meta_sha256": ue._sha256(p["train_meta"]),
        "valid_sha256": ue._sha256(p["valid"]),
        "eval_sha256": ue._sha256(p["eval"]),
        "train_ids": [r["id"] for r in rows],
        "validation_recordings": validation_recordings,
        "split": split_meta,
        "targets": vtargets,
        "target_exclusions": target_excl,
        "cell_exclusions": cell_excl,
        "metric": metric_meta,
        "coverage": coverage,
    }
    _write_json(p["valid_meta"], valid_meta)
    return {"train_sha256": train_sha, "coverage": coverage, "split": split_meta}


def _provenance(reserved_raw: Path) -> dict[str, Any]:
    here = Path(__file__).parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=here
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "dataset_pins": si.dataset_pins(),
        "reserved_raw_dir": str(reserved_raw),
        "git_commit": commit,
        "source_sha256": {m: ue._sha256(here / m) for m in SOURCE_MODULES},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__},
    }


def prepare(root: Path, reserved_raw: Path = DEFAULT_RESERVED_RAW) -> list[Path]:
    """Split every rig, freeze the worker pack and the validation cells."""
    loaded: dict[str, Any] = {}
    splits: dict[str, tuple[list[Piece], list[Piece]]] = {}
    for rig in data.RIGS:
        pieces, excluded, info = _load(rig, reserved_raw)
        train, valid, split_meta = _split(rig, pieces)
        split_meta |= {"load": info, "disjointness": assert_disjoint(rig, train, valid)}
        splits[rig] = (train, valid)
        loaded[rig] = (excluded, split_meta)
    metrics, metric_meta = _metrics(splits)
    written: list[Path] = []
    rigs_meta = {}
    for rig in data.RIGS:
        train, valid = splits[rig]
        excluded, split_meta = loaded[rig]
        rigs_meta[rig] = _write_rig(
            root, rig, train, valid, excluded, split_meta, metric_meta["rigs"][rig]
        )
        written += [p for k, p in paths(root, rig).items()]
    manifest = {
        "protocol": {
            "fs_hz": FS,
            "fs_train_hz": GRID_HZ,
            "n_draws": N_DRAWS,
            "phases": list(se.PHASES),
            "arms": list(ARMS),
            "rigs": list(data.RIGS),
            "fold_label": FOLD,
            "split": "physical recordings disjoint; seeded 20 % for dregon/neurobem_quad/pitcn_quad; "
            "MD2 published train/test; VID 4 original flights train / 2 round validate; NanoBench publisher split; Blackbird "
            f"within-flight {WITHIN_FLIGHT_TRAIN_FRACTION:.0%} earlier / later after a {GUARD_S} s guard",
            "schedule": "codes from 25 Hz motors_on/takeoff/landing only (unified_experiment.schedule_codes); "
            "training tracks without landmarks are coarse_initial (code 2 throughout)",
            "seeds": f"default_rng(SeedSequence(seed)), seed = [2026, crc32('rig|{FOLD}|id|draws')], shared by arms",
            "draws": "runs/{rig}/{arm}.draws.npz key t{k}: (32, 4, n) float32, sample 0 = sample 0 of target k",
            "scoring": "schedule_eval._score (score_phase, origin-averaged); segments averaged per physical "
            "recording; cruise metric off-rig on training recordings only",
            "excluded_everywhere": "DREGON command-only recordings",
        },
        "metric": {
            rig: {
                "band_power": m.band_power.tolist(),
                "level_scale_rps": m.level_scale_rps,
                "spectrum_scale": m.spectrum_scale,
            }
            for rig, m in metrics.items()
        },
        "metric_meta": metric_meta,
        "rigs": rigs_meta,
        "provenance": _provenance(reserved_raw),
    }
    path = root / "pack" / "manifest.json"
    _write_json(path, manifest)
    return [*written, path]


# ─── score ────────────────────────────────────────────────────────────────────


def _fit_problems(
    fit: Mapping[str, Any], rig: str, arm: str, train_sha: str, train_ids: list[str]
) -> list[str]:
    return [
        k
        for k, ok in (
            ("rig", fit.get("rig") == rig),
            ("arm", fit.get("arm") == arm),
            ("train_sha256", fit.get("train_sha256") == train_sha),
            ("train_ids", list(fit.get("train_ids", [])) == train_ids),
            ("n_params", "n_params" in fit),
        )
        if not ok
    ]


def _segment_rows(
    root: Path,
    rig: str,
    arm: str,
    vmeta: Mapping[str, Any],
    groups: Mapping[tuple, se.Group],
    metric: CruiseMetric,
) -> list[dict[str, Any]] | None:
    rp = run_paths(root, rig, arm)
    if not (rp["fit"].exists() and rp["draws"].exists()):
        return None
    fit = json.loads(rp["fit"].read_text())
    problems = _fit_problems(fit, rig, arm, vmeta["train_sha256"], vmeta["train_ids"])
    rows = []
    with np.load(rp["draws"]) as dr:
        for t in vmeta["targets"]:
            k = t["index"]
            status, paths_ = "ok", None
            if problems:
                status = f"failed: fit provenance mismatch {problems}"
            elif f"t{k}" not in dr.files:
                status = "failed: no draws for target"
            else:
                paths_ = np.asarray(dr[f"t{k}"], dtype=np.float64)
                if paths_.shape != (N_DRAWS, data.NUM_ROTORS, t["n"]):
                    status = f"failed: draws shape {paths_.shape} != {(N_DRAWS, data.NUM_ROTORS, t['n'])}"
            # A NaN marks samples the arm has no fitted law for (the worker's
            # declared choice), an infinity a diverged sampler (float32
            # overflow); only the windows that touch one fail, each reported.
            wt = fit.get("targets")
            worker = wt[k] if isinstance(wt, list) and k < len(wt) else None
            for cell in t["cells"]:
                g = groups[(k, cell["phase"])]
                score, diag, st = None, {}, status
                if st == "ok":
                    assert paths_ is not None  # status stays "ok" only once draws loaded
                    cuts = [paths_[:, :, s : s + g.n] for s in g.starts]
                    if any(np.isinf(c).any() for c in cuts):
                        st = "failed: diverged (infinite value in the generated window)"
                    elif any(np.isnan(c).any() for c in cuts):
                        st = "failed: unsupported (NaN in the generated window)"
                    else:
                        st, score, diag = se._score(g, cuts, metric)
                rows.append(
                    {
                        "rig": rig,
                        "recording": t["recording"],
                        "id": t["id"],
                        "target": k,
                        "phase": g.phase,
                        "arm": arm,
                        "status": st,
                        "score": score,
                        "diagnostics": diag,
                        "n_refs": int(len(g.refs[0])),
                        "refs_independent": g.support["refs_independent"],
                        "n_draws": N_DRAWS,
                        "seed": t["seed"],
                        "n_params": fit.get("n_params"),
                        "worker_target": worker,
                    }
                )
    return rows


def _flight_rows(seg: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Segments of one physical recording averaged with equal weight."""
    by: dict[tuple, list[Mapping[str, Any]]] = {}
    for r in seg:
        by.setdefault((r["rig"], r["recording"], r["phase"], r["arm"]), []).append(r)
    out = []
    for (rig, recording, phase, arm), rs in sorted(by.items()):
        ok = all(r["status"] == "ok" for r in rs)
        out.append(
            {
                "rig": rig,
                "recording": recording,
                "id": ",".join(r["id"] for r in rs),
                "ids": [r["id"] for r in rs],
                "phase": phase,
                "arm": arm,
                "status": "ok" if ok else next(r["status"] for r in rs if r["status"] != "ok"),
                "score": float(np.mean([r["score"] for r in rs])) if ok else None,
                "n_segments": len(rs),
                "refs_independent": all(r["refs_independent"] for r in rs),
                "n_params": rs[0]["n_params"],
            }
        )
    return out


def summarise(
    rows: Sequence[Mapping[str, Any]], arms_by_rig: Mapping[str, Sequence[str]]
) -> dict[str, dict[str, Any]]:
    """Per rig and phase: means over the recordings every present arm scored;
    ``balanced``: equal-weight mean of those phase means."""
    out: dict[str, dict[str, Any]] = {}
    for rig, arms in arms_by_rig.items():
        rig_rows = [r for r in rows if r["rig"] == rig]
        phases: dict[str, Any] = {}
        for phase in se.PHASES:
            cell = {(r["recording"], r["arm"]): r for r in rig_rows if r["phase"] == phase}
            recs = sorted({rec for rec, _ in cell})
            if not recs:
                continue
            paired = [
                rec
                for rec in recs
                if arms and all((rec, a) in cell and cell[(rec, a)]["status"] == "ok" for a in arms)
            ]
            phases[phase] = {
                "n_cells": len(recs),
                "n_valid": len(paired),
                "paired_recordings": paired,
                "means": {
                    a: float(np.mean([cell[(rec, a)]["score"] for rec in paired]))
                    if paired
                    else None
                    for a in arms
                },
                "n_ok": {
                    a: sum((rec, a) in cell and cell[(rec, a)]["status"] == "ok" for rec in recs)
                    for a in arms
                },
                "failed": [
                    {"recording": rec, "arm": a, "status": cell[(rec, a)]["status"]}
                    for rec in recs
                    for a in arms
                    if (rec, a) in cell and cell[(rec, a)]["status"] != "ok"
                ],
            }
        used = [ph for ph, v in phases.items() if v["n_valid"]]
        balanced = {
            a: float(np.mean([phases[ph]["means"][a] for ph in used])) if used else None
            for a in arms
        }
        out[rig] = {
            "arms": list(arms),
            "phases": phases,
            "balanced": {"phases": used, "means": balanced},
        }
    return out


def score(root: Path, *, arms: Sequence[str] = ARMS, destination: Path | None = None) -> list[Path]:
    """Score every available run on the frozen cells; write ``scores.json`` and ``phase_scores.csv``."""
    manifest = json.loads((root / "pack" / "manifest.json").read_text())
    destination = root if destination is None else destination
    destination.mkdir(parents=True, exist_ok=True)
    seg_rows: list[dict[str, Any]] = []
    missing, arms_by_rig, exclusions = [], {}, {}
    for rig in manifest["protocol"]["rigs"]:
        p = paths(root, rig)
        vmeta = json.loads(p["valid_meta"].read_text())
        for key, sha in (
            ("train", "train_sha256"),
            ("train_meta", "train_meta_sha256"),
            ("eval", "eval_sha256"),
        ):
            if ue._sha256(p[key]) != vmeta[sha]:
                raise ValueError(f"{p[key]} does not match {p['valid_meta']}")
        metric = ue._metric(manifest, rig)
        with np.load(p["eval"]) as ev:
            groups = {
                (t["index"], c["phase"]): se.Group(
                    rig,
                    FOLD,
                    t["id"],
                    c["phase"],
                    c["hover"],
                    tuple(c["starts"]),
                    c["n"],
                    tuple(se._frozen(o) for o in ev[f"t{t['index']}__{c['phase']}__obs"]),
                    tuple(se._frozen(r) for r in ev[f"t{t['index']}__{c['phase']}__refs"]),
                    c["support"],
                )
                for t in vmeta["targets"]
                for c in t["cells"]
            }
        present = []
        for arm in arms:
            rows = _segment_rows(root, rig, arm, vmeta, groups, metric)
            if rows is None:
                missing.append({"rig": rig, "arm": arm})
                continue
            present.append(arm)
            seg_rows += rows
        arms_by_rig[rig] = present
        exclusions[rig] = {"targets": vmeta["target_exclusions"], "cells": vmeta["cell_exclusions"]}
    rows = _flight_rows(seg_rows)
    summary = summarise(rows, arms_by_rig)
    out_json = destination / "scores.json"
    _write_json(
        out_json,
        {
            "protocol": manifest["protocol"] | {"arms": list(arms)},
            "manifest_sha256": ue._sha256(root / "pack/manifest.json"),
            "rows": rows,
            "segment_rows": seg_rows,
            "summary": summary,
            "missing_runs": missing,
            "exclusions": exclusions,
            "csv_note": "phase=balanced: equal-weight mean of the rig's phase means; its n_valid "
            "is the number of phases averaged",
        },
    )
    out_csv = destination / "phase_scores.csv"
    tmp = out_csv.with_name(f"{out_csv.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rig", "phase", "n_valid", *arms])
        for rig, s in summary.items():
            for phase, v in s["phases"].items():
                w.writerow(
                    [
                        rig,
                        phase,
                        v["n_valid"],
                        *("" if v["means"].get(a) is None else repr(v["means"][a]) for a in arms),
                    ]
                )
            b = s["balanced"]
            w.writerow(
                [
                    rig,
                    "balanced",
                    len(b["phases"]),
                    *("" if b["means"].get(a) is None else repr(b["means"][a]) for a in arms),
                ]
            )
    tmp.replace(out_csv)
    return [out_json, out_csv]


# ─── CLI ──────────────────────────────────────────────────────────────────────


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="split every rig and freeze ROOT/pack")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--reserved-raw", type=Path, default=DEFAULT_RESERVED_RAW)
    s = sub.add_parser("score", help="score ROOT/runs on ROOT/pack")
    s.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    out = prepare(args.root, args.reserved_raw) if args.cmd == "prepare" else score(args.root)
    for path in out:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
