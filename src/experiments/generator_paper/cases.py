"""Real flight windows for the generator paper, cut from pinned published frames.

One loader, :func:`load_flight_clips`, turns a ``(rig, split)`` pair into a
deterministic list of :class:`FlightClip` windows; :func:`load_support_clips`
cuts EXPLICIT ``(recording, start, duration)`` windows of a split (the fitted
supports of the matched-regime baselines) through the same decode, resampling
and slicing path. Everything comes from the
published ``tdframe-v1`` datasets through the existing decode path
(:mod:`data_processing.streams` + :mod:`experiments.stochastic_fit.clips`);
no raw WAV/CSV is parsed and no timing or rev/s correction is applied here.

Contracts:

* **Pinned sources, fixed splits.** ``md2`` (alias ``michaels``) reads
  ``michaels-frames@8e9d1495…``: calibration is FLY125 only, evaluation is
  FLY124 only; the FLY103/FLY108 TEST recordings are refused. ``dregon``
  reads ``DREGON-frames@261b0997…``: calibration is the five ``*_room2``
  pure-noise flights, evaluation is ``free-flight_nosource_room1``; the
  DREGON TEST ids are refused. Recordings are selected by ``recording_id``,
  never by ``meta.split`` (the published DREGON split field predates the
  test split).
* **One label, published as-is.** The rotor speed is ``rps_refined``, which
  already carries the calibration (Michael's ``rps_scale``/alignment) and the
  regime gating. A frame without it is an error: there is no fallback to raw
  telemetry.
* **Shared coverage only.** Windows lie inside
  :attr:`clips.Recording.coverage` (audio AND label both present); labels are
  never extrapolated past their own stamps.
* **Native audio travels.** Each window carries the untouched native-rate
  audio (44.1 kHz on both rigs) beside its 16 kHz copy. The 16 kHz audio is
  the whole recording decimated ONCE (``decimate_audio``) and then sliced, so
  windows carry no per-window filter edges. Window starts sit on the common
  ``1 / gcd(native_sr, 16000)`` grid of the audio origin, so both copies start
  on the same instant with integer sample offsets.
* **Deterministic windows.** Full, non-overlapping ``duration_s`` windows on
  one grid anchored at the first jointly covered instant where every rotor is
  at least ``min_rps``; a window is kept only if every rotor stays at least
  ``min_rps`` over its whole span (exact over the linearly interpolated
  label). Failing windows are dropped, never shifted.

FLY124 was used during historical model development, so the md2 evaluation
split is a recording-held-out parameter-transfer diagnostic, not a blind test;
every md2 evaluation clip says so in its provenance.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data_processing.noise_model.resample import decimate_audio
from data_processing.sources.dregon import SPLIT_RECORDINGS as DREGON_SPLIT_RECORDINGS
from data_processing.sources.dregon import TEST_RECORDING_IDS as DREGON_TEST_RECORDING_IDS
from data_processing.sources.michaels import MICHAELS_TEST_FILES
from experiments.stochastic_fit import clips as C

#: Analysis rate of every :class:`FlightClip`'s ``audio``/``rps``.
SR = 16000
#: The published label: calibrated + regime-gated; never re-corrected here.
LABEL_KEY = "rps_refined"
#: Regime rule on the window's per-rotor label extrema (rev/s).
STANDBY_MAX_RPS = 45.0
CRUISE_MIN_RPS = 65.0
REGIMES = ("standby", "transition", "cruise")
SPLITS = ("calibration", "evaluation")

#: Relative tolerance for "is this time a whole number of grid ticks".
_TICK_TOL = 1e-6
#: Slack, in seconds, on coverage comparisons (DREGON clocks are unix seconds).
_TIME_EPS = 1e-6


@dataclass(frozen=True)
class FlightClip:
    """One evaluation/calibration window. Arrays are time-last and read-only.

    ``start_s`` is relative to the recording's audio origin (audio sample 0).
    ``audio`` ``(M, T)`` and ``rps`` ``(R, T)`` share the ``sr`` grid;
    ``native_audio`` ``(M, T_native)`` covers the same span at ``native_sr``.
    """

    key: str
    rig: str
    recording_id: str
    start_s: float
    sr: int
    audio: np.ndarray
    rps: np.ndarray
    native_sr: int
    native_audio: np.ndarray
    regime: str
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        if self.audio.ndim != 2 or self.rps.ndim != 2 or self.native_audio.ndim != 2:
            raise ValueError(f"{self.key}: audio, rps and native_audio must be 2-D, time-last")
        if self.rps.shape[-1] != self.audio.shape[-1]:
            raise ValueError(
                f"{self.key}: rps {self.rps.shape} is not on the audio grid {self.audio.shape}"
            )
        if self.native_audio.shape[0] != self.audio.shape[0]:
            raise ValueError(
                f"{self.key}: native_audio has {self.native_audio.shape[0]} channels, "
                f"audio has {self.audio.shape[0]}"
            )
        if self.native_audio.shape[-1] * int(self.sr) != self.audio.shape[-1] * int(self.native_sr):
            raise ValueError(
                f"{self.key}: native ({self.native_audio.shape[-1]} @ {self.native_sr} Hz) and "
                f"analysis ({self.audio.shape[-1]} @ {self.sr} Hz) spans differ"
            )
        if self.regime not in REGIMES:
            raise ValueError(f"{self.key}: regime {self.regime!r} not in {REGIMES}")

    @property
    def duration_s(self) -> float:
        return float(self.audio.shape[-1] / self.sr)


@dataclass(frozen=True)
class _Rig:
    name: str
    dataset: str
    version: str
    n_mics: int
    n_rotors: int
    splits: Mapping[str, tuple[str, ...]]
    forbidden: tuple[str, ...]
    evaluation_status: str

    @property
    def pin(self) -> str:
        return f"{self.dataset}@{self.version}"


_MD2 = _Rig(
    name="md2",
    dataset="michaels-frames",
    version="8e9d149560dd5d5fa8aaa87e7ea537d7e4d96e829e874dab7d9c92621cca6e46",
    n_mics=8,
    n_rotors=4,
    splits={"calibration": ("FLY125",), "evaluation": ("FLY124",)},
    forbidden=tuple(Path(csv).stem for _, csv, *_ in MICHAELS_TEST_FILES),
    evaluation_status=(
        "FLY124 was used in historical model development: recording-held-out "
        "parameter-transfer diagnostic, not a blind test"
    ),
)
_DREGON = _Rig(
    name="dregon",
    dataset="DREGON-frames",
    version="261b09971c8ace16e3434f4a89406c6043403f8ac89a3f1202566cb8c712ba89",
    n_mics=8,
    n_rotors=4,
    splits={
        "calibration": tuple(
            r for r in DREGON_SPLIT_RECORDINGS["in_flight_noise"] if r.endswith("_room2")
        ),
        "evaluation": ("free-flight_nosource_room1",),
    },
    forbidden=tuple(DREGON_TEST_RECORDING_IDS),
    evaluation_status="room1 pure-noise flight held out from the room2 calibration recordings",
)
_RIGS: dict[str, _Rig] = {"md2": _MD2, "michaels": _MD2, "dregon": _DREGON}


def _check_rig_splits(rig: _Rig) -> None:
    cal, ev = set(rig.splits["calibration"]), set(rig.splits["evaluation"])
    if not cal or not ev or cal & ev:
        raise RuntimeError(
            f"{rig.name}: calibration/evaluation recordings must be non-empty and disjoint"
        )
    leaked = (cal | ev) & set(rig.forbidden)
    if leaked:
        raise RuntimeError(f"{rig.name}: split names held-out TEST recordings {sorted(leaked)}")


_check_rig_splits(_MD2)
_check_rig_splits(_DREGON)
if len(_DREGON.splits["calibration"]) != 5:
    raise RuntimeError(
        "DREGON calibration must be the five room2 pure-noise flights, "
        f"got {_DREGON.splits['calibration']}"
    )


def resolve_rig(rig: str) -> str:
    """Canonical rig name (``"md2"``/``"dregon"``) for ``rig`` or its alias."""
    spec = _RIGS.get(str(rig).strip().lower())
    if spec is None:
        raise ValueError(f"unsupported rig {rig!r}; supported: {sorted(_RIGS)}")
    return spec.name


def regime_of(rotor_min: np.ndarray, rotor_max: np.ndarray) -> str:
    """``standby`` if every rotor <= 45, ``cruise`` if every rotor >= 65, else ``transition``."""
    if float(np.max(rotor_max)) <= STANDBY_MAX_RPS:
        return "standby"
    if float(np.min(rotor_min)) >= CRUISE_MIN_RPS:
        return "cruise"
    return "transition"


def _check_time_last(series: Any, what: str) -> None:
    axis = getattr(series, "time_axis", None)
    ndim = getattr(series, "ndim", None)
    if axis is None or ndim is None or axis != ndim - 1:
        dims = getattr(series, "dims", None)
        raise ValueError(f"{what}: expected time on the last axis, got dims {dims}")


def _geometry(frame: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("mic_pos", "rotor_pos"):
        if name in frame:
            out[name] = {"available": True, "shape": list(np.shape(frame[name].data))}
        else:
            out[name] = {"available": False}
    return out


def _decode(rig: _Rig, ids: tuple[str, ...]) -> dict[str, tuple[C.Recording, dict[str, Any]]]:
    """Decode exactly ``ids`` from the pinned dataset; skip every other sample undecoded."""
    from data_processing.frames import meta_dict
    from data_processing.streams import decode_tdframe, is_data_sample, open_repository

    wanted = set(ids)
    out: dict[str, tuple[C.Recording, dict[str, Any]]] = {}
    ds = open_repository().dataset(rig.dataset, rig.version)
    for sample in ds.samples():
        if not is_data_sample(sample):
            continue
        key = str(sample[0])
        if key not in wanted or key in out:
            continue
        frame = decode_tdframe(sample)
        meta = meta_dict(frame)
        rid = str(meta.setdefault("recording_id", key))
        if rid != key:
            raise ValueError(f"{rig.pin}: sample {key!r} carries recording_id {rid!r}")
        if "audio" not in frame:
            raise KeyError(f"{rig.pin}:{key} publishes no 'audio'")
        if LABEL_KEY not in frame:
            raise KeyError(
                f"{rig.pin}:{key} publishes no {LABEL_KEY!r} track; refusing raw fallback"
            )
        _check_time_last(frame["audio"], f"{rig.pin}:{key}:audio")
        _check_time_last(frame[LABEL_KEY], f"{rig.pin}:{key}:{LABEL_KEY}")
        rec = C._recording_from_frame(rig.dataset, rig.version, frame, meta, LABEL_KEY)
        out[key] = (rec, _geometry(frame))
        if len(out) == len(wanted):
            break
    missing = sorted(wanted - set(out))
    if missing:
        raise KeyError(f"{rig.pin}: recordings {missing} not found")
    return out


def _validate_recording(rig: _Rig, rec: C.Recording) -> tuple[int, float, float]:
    """``(native_sr, cov0, cov1)`` after checking shapes, rate, stamps and coverage."""
    rid = rec.recording_id
    if rec.n_channels != rig.n_mics:
        raise ValueError(f"{rig.pin}:{rid}: {rec.n_channels} mic channels, expected {rig.n_mics}")
    if rec.rps is None or rec.rps_t is None:
        raise ValueError(f"{rig.pin}:{rid}: no {LABEL_KEY} values")
    if rec.rps.shape[0] != rig.n_rotors:
        raise ValueError(f"{rig.pin}:{rid}: {rec.rps.shape[0]} rotors, expected {rig.n_rotors}")
    if rec.rps_t.size < 2 or not np.all(np.diff(rec.rps_t) > 0.0):
        raise ValueError(f"{rig.pin}:{rid}: {LABEL_KEY} stamps are not strictly increasing")
    native_sr = int(round(rec.sr))
    if native_sr <= 0 or abs(rec.sr - native_sr) > 1e-6:
        raise ValueError(f"{rig.pin}:{rid}: native rate {rec.sr!r} is not an integer grid")
    cov0, cov1 = rec.coverage
    if not cov1 > cov0:
        raise ValueError(f"{rig.pin}:{rid}: empty audio/{LABEL_KEY} coverage [{cov0}, {cov1}]")
    inside = (rec.rps_t >= cov0) & (rec.rps_t <= cov1)
    if not np.all(np.isfinite(rec.rps[:, inside])):
        raise ValueError(f"{rig.pin}:{rid}: non-finite {LABEL_KEY} values inside coverage")
    return native_sr, float(cov0), float(cov1)


def _extrema(rec: C.Recording, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact per-rotor min/max of the linearly interpolated label over ``[t0, t1]``."""
    assert rec.rps is not None and rec.rps_t is not None
    i = int(np.searchsorted(rec.rps_t, t0, side="right"))
    j = int(np.searchsorted(rec.rps_t, t1, side="left"))
    ends = rec.rps_at(np.array([t0, t1]))
    inner = rec.rps[:, i:j]
    if inner.shape[-1]:
        return (
            np.minimum(inner.min(axis=1), ends.min(axis=1)),
            np.maximum(inner.max(axis=1), ends.max(axis=1)),
        )
    return ends.min(axis=1), ends.max(axis=1)


@dataclass(frozen=True)
class _Window:
    index: int
    tick: int
    n_ticks: int
    t0: float
    t1: float
    rotor_min: np.ndarray
    rotor_max: np.ndarray


def _windows(
    rec: C.Recording,
    *,
    cov0: float,
    cov1: float,
    ticks_per_s: int,
    n_ticks: int,
    min_rps: float,
) -> tuple[float, list[_Window]]:
    """``(anchor_s, kept windows)`` on the grid anchored at the first rotor-on instant."""
    assert rec.rps is not None and rec.rps_t is not None
    inside = np.flatnonzero((rec.rps_t >= cov0) & (rec.rps_t <= cov1))
    on = np.all(rec.rps[:, inside] >= min_rps, axis=0)
    if not on.any():
        raise ValueError(
            f"{rec.recording_id}: no covered instant with every rotor >= {min_rps} rev/s"
        )
    anchor = float(rec.rps_t[inside[int(np.argmax(on))]])
    tick = int(math.ceil((anchor - rec.t_start) * ticks_per_s - _TICK_TOL))
    out: list[_Window] = []
    index = 0
    while True:
        t0 = rec.t_start + tick / ticks_per_s
        t1 = rec.t_start + (tick + n_ticks) / ticks_per_s
        if t1 > cov1 + _TIME_EPS:
            break
        if t0 >= cov0 - _TIME_EPS:
            lo, hi = _extrema(rec, t0, t1)
            if float(lo.min()) >= min_rps:
                out.append(_Window(index, tick, n_ticks, t0, t1, lo, hi))
        tick += n_ticks
        index += 1
    return anchor, out


def _whole_ticks(seconds: float, ticks_per_s: int, what: str) -> int:
    n = float(seconds) * ticks_per_s
    k = int(round(n))
    if k <= 0 or abs(n - k) > _TICK_TOL * max(1.0, n):
        raise ValueError(
            f"{what}={seconds!r} s is not a positive multiple of the {1.0 / ticks_per_s:g} s "
            "grid shared by the native and 16 kHz sample clocks"
        )
    return k


def _resolve_split(rig: str, split: str) -> tuple[_Rig, tuple[str, ...]]:
    spec = _RIGS.get(str(rig).strip().lower())
    if spec is None:
        raise ValueError(f"unsupported rig {rig!r}; supported: {sorted(_RIGS)}")
    if split not in SPLITS:
        raise ValueError(f"unsupported split {split!r}; supported: {SPLITS}")
    ids = spec.splits[split]
    leaked = set(ids) & set(spec.forbidden)
    if leaked:  # pragma: no cover - guarded at import
        raise RuntimeError(f"{spec.name}:{split} names held-out TEST recordings {sorted(leaked)}")
    return spec, ids


def _recording_record(
    spec: _Rig,
    rec: C.Recording,
    *,
    cov0: float,
    cov1: float,
    ticks_per_s: int,
    native_sr: int,
    geometry: dict[str, Any],
    window_rule: dict[str, Any],
) -> dict[str, Any]:
    rps_t = np.asarray(rec.rps_t)
    return {
        "dataset": spec.dataset,
        "dataset_version": spec.version,
        "dataset_pin": spec.pin,
        "recording_id": rec.recording_id,
        "label_key": LABEL_KEY,
        "label_span_s": [float(rps_t[0]), float(rps_t[-1])],
        "label_median_step_s": float(np.median(np.diff(rps_t))),
        "clock_origin_s": float(rec.t_start),
        "coverage_s": [cov0, cov1],
        **window_rule,
        "grid_step_s": 1.0 / ticks_per_s,
        "native_sr": native_sr,
        "geometry": geometry,
        "source_meta": dict(rec.meta),
    }


def _materialize(
    spec: _Rig,
    rec: C.Recording,
    *,
    native_sr: int,
    ticks_per_s: int,
    recording: dict[str, Any],
    windows: list[tuple[_Window, dict[str, Any], dict[str, Any]]],
) -> list[FlightClip]:
    """Cut ``(window, selection, criteria)`` entries from ONE whole-recording decimation.

    ``selection`` follows the recording record in each provenance, ``criteria``
    precedes the window's rotor extrema; both keep the loader's key order.
    """
    native = rec.audio
    native.flags.writeable = False
    audio16 = decimate_audio(native, native_sr, SR)
    audio16.flags.writeable = False
    nat_per_tick, sr_per_tick = native_sr // ticks_per_s, SR // ticks_per_s
    clips: list[FlightClip] = []
    for w, selection, criteria in windows:
        n_tick = w.n_ticks
        i0, i1 = w.tick * nat_per_tick, (w.tick + n_tick) * nat_per_tick
        j0, j1 = w.tick * sr_per_tick, (w.tick + n_tick) * sr_per_tick
        if i1 > native.shape[-1] or j1 > audio16.shape[-1]:
            raise ValueError(
                f"{spec.pin}:{rec.recording_id}: window [{w.t0}, {w.t1}] runs past the audio"
            )
        rps = rec.rps_at(rec.t_start + np.arange(j0, j1) / SR)
        rps.flags.writeable = False
        rel0 = w.tick / ticks_per_s
        rel1 = (w.tick + n_tick) / ticks_per_s
        regime = regime_of(w.rotor_min, w.rotor_max)
        key = f"{spec.name}:{rec.recording_id}@{rel0:.3f}+{rel1 - rel0:g}"
        provenance = {
            **recording,
            **selection,
            "absolute_start_s": float(w.t0),
            "absolute_end_s": float(w.t1),
            "relative_start_s": float(rel0),
            "relative_end_s": float(rel1),
            "native_start_sample": int(i0),
            "native_n_samples": int(i1 - i0),
            "start_sample": int(j0),
            "n_samples": int(j1 - j0),
            "resampling": (
                "whole-recording scipy.signal.resample_poly "
                "(data_processing.noise_model.resample.decimate_audio), then sliced"
            ),
            "rps_interpolation": f"linear in {LABEL_KEY} stamps, no extrapolation",
            **criteria,
            "rotor_min_rps": [float(v) for v in w.rotor_min],
            "rotor_max_rps": [float(v) for v in w.rotor_max],
            "regime_rule": {
                "standby_max_rps": STANDBY_MAX_RPS,
                "cruise_min_rps": CRUISE_MIN_RPS,
            },
        }
        clips.append(
            FlightClip(
                key=key,
                rig=spec.name,
                recording_id=rec.recording_id,
                start_s=float(rel0),
                sr=SR,
                audio=audio16[:, j0:j1],
                rps=rps,
                native_sr=native_sr,
                native_audio=native[:, i0:i1],
                regime=regime,
                provenance=provenance,
            )
        )
    return clips


def load_flight_clips(
    rig: str,
    split: str,
    *,
    duration_s: float,
    min_rps: float = 20.0,
    limit: int | None = None,
) -> list[FlightClip]:
    """Deterministic non-overlapping flight windows of one rig split.

    ``rig``: ``"md2"``/``"michaels"`` or ``"dregon"``. ``split``:
    ``"calibration"`` or ``"evaluation"``. Clips come in the split's declared
    recording order, then by time; ``limit`` keeps the first ``limit``.
    Every recording of the split is decoded and windowed even when ``limit``
    is set, so each clip's ``split_window_counts`` describes the full split.
    """
    spec, ids = _resolve_split(rig, split)
    if not (math.isfinite(float(duration_s)) and float(duration_s) > 0.0):
        raise ValueError(f"duration_s must be positive and finite, got {duration_s!r}")
    if not (math.isfinite(float(min_rps)) and float(min_rps) >= 0.0):
        raise ValueError(f"min_rps must be finite and >= 0, got {min_rps!r}")
    if limit is not None and int(limit) <= 0:
        raise ValueError(f"limit must be positive or None, got {limit!r}")
    min_rps = float(min_rps)

    decoded = _decode(spec, ids)

    plans: list[tuple[C.Recording, dict[str, Any], int, int, list[_Window]]] = []
    counts: dict[str, int] = {}
    for rid in ids:
        rec, geometry = decoded[rid]
        native_sr, cov0, cov1 = _validate_recording(spec, rec)
        ticks_per_s = math.gcd(native_sr, SR)
        n_ticks = _whole_ticks(duration_s, ticks_per_s, "duration_s")
        anchor, wins = _windows(
            rec, cov0=cov0, cov1=cov1, ticks_per_s=ticks_per_s, n_ticks=n_ticks, min_rps=min_rps
        )
        counts[rid] = len(wins)
        recording = _recording_record(
            spec,
            rec,
            cov0=cov0,
            cov1=cov1,
            ticks_per_s=ticks_per_s,
            native_sr=native_sr,
            geometry=geometry,
            window_rule={
                "anchor_s": anchor,
                "anchor_rule": f"first covered {LABEL_KEY} stamp with every rotor >= {min_rps} rev/s",
            },
        )
        plans.append((rec, recording, native_sr, ticks_per_s, wins))

    if not any(counts.values()):
        raise ValueError(
            f"{spec.pin} {split}: no full {duration_s:g} s window with every rotor >= {min_rps} "
            f"rev/s inside coverage (recordings {list(ids)})"
        )

    clips: list[FlightClip] = []
    for rec, recording, native_sr, ticks_per_s, wins in plans:
        if limit is not None:
            wins = wins[: max(int(limit) - len(clips), 0)]
        if not wins:
            continue
        selected = [
            (
                w,
                {
                    "split": split,
                    "split_recordings": list(ids),
                    "split_window_counts": dict(counts),
                    "evaluation_status": spec.evaluation_status if split == "evaluation" else None,
                    "window_index": int(w.index),
                },
                {"min_rps": min_rps},
            )
            for w in wins
        ]
        clips += _materialize(
            spec,
            rec,
            native_sr=native_sr,
            ticks_per_s=ticks_per_s,
            recording=recording,
            windows=selected,
        )
    return clips


def load_support_clips(
    rig: str,
    split: str,
    windows: Sequence[tuple[str, float, float]],
    *,
    labels: Sequence[Mapping[str, Any]] | None = None,
) -> list[FlightClip]:
    """Explicit ``(recording_id, start_s, duration_s)`` windows of one rig split, in order.

    ``start_s`` is relative to the recording's audio origin (as
    :attr:`FlightClip.start_s`). The audio, label and resampling path is that
    of :func:`load_flight_clips` (whole-recording decimation, then sliced), so a
    window that is also a grid window is bit-identical to it. A window off the
    shared sample grid, outside the joint audio/label coverage, or on a
    recording outside the split is refused, never shifted or dropped.
    ``labels`` (one mapping per window) is merged into each window's
    provenance, e.g. the support spec the window was read from.
    """
    spec, ids = _resolve_split(rig, split)
    windows = [(str(r), float(s), float(d)) for r, s, d in windows]
    if not windows:
        raise ValueError("load_support_clips needs at least one window")
    if labels is not None and len(labels) != len(windows):
        raise ValueError(f"{len(labels)} labels for {len(windows)} windows")
    outside = sorted({r for r, _, _ in windows} - set(ids))
    if outside:
        raise ValueError(f"{spec.name}:{split} holds {list(ids)}; windows name {outside}")
    order = list(dict.fromkeys(r for r, _, _ in windows))
    decoded = _decode(spec, tuple(order))

    per_recording: dict[str, list[tuple[int, _Window]]] = {rid: [] for rid in order}
    context: dict[str, tuple[C.Recording, int, int, dict[str, Any]]] = {}
    for rid in order:
        rec, geometry = decoded[rid]
        native_sr, cov0, cov1 = _validate_recording(spec, rec)
        ticks_per_s = math.gcd(native_sr, SR)
        recording = _recording_record(
            spec,
            rec,
            cov0=cov0,
            cov1=cov1,
            ticks_per_s=ticks_per_s,
            native_sr=native_sr,
            geometry=geometry,
            window_rule={
                "window_rule": "explicit support window on the audio clock; not the regular grid"
            },
        )
        context[rid] = (rec, native_sr, ticks_per_s, recording)
    for index, (rid, start_s, duration_s) in enumerate(windows):
        rec, _, ticks_per_s, _ = context[rid]
        n_ticks = _whole_ticks(duration_s, ticks_per_s, "duration_s")
        start = start_s * ticks_per_s
        tick = int(round(start))
        if tick < 0 or abs(start - tick) > _TICK_TOL * max(1.0, abs(start)):
            raise ValueError(
                f"{rid}: start_s={start_s!r} is not a non-negative multiple of the "
                f"{1.0 / ticks_per_s:g} s shared sample grid"
            )
        t0 = rec.t_start + tick / ticks_per_s
        t1 = rec.t_start + (tick + n_ticks) / ticks_per_s
        cov0, cov1 = context[rid][3]["coverage_s"]
        if t0 < cov0 - _TIME_EPS or t1 > cov1 + _TIME_EPS:
            raise ValueError(
                f"{spec.pin}:{rid}: window [{t0}, {t1}] leaves the audio/{LABEL_KEY} coverage "
                f"[{cov0}, {cov1}]"
            )
        lo, hi = _extrema(rec, t0, t1)
        per_recording[rid].append((index, _Window(index, tick, n_ticks, t0, t1, lo, hi)))

    out: dict[int, FlightClip] = {}
    for rid in order:
        rec, native_sr, ticks_per_s, recording = context[rid]
        entries = per_recording[rid]
        cut = _materialize(
            spec,
            rec,
            native_sr=native_sr,
            ticks_per_s=ticks_per_s,
            recording=recording,
            windows=[
                (
                    w,
                    {
                        "split": split,
                        "split_recordings": list(ids),
                        "evaluation_status": spec.evaluation_status
                        if split == "evaluation"
                        else None,
                        "support_index": int(index),
                        **(dict(labels[index]) if labels is not None else {}),
                    },
                    {},
                )
                for index, w in entries
            ],
        )
        out.update({index: clip for (index, _), clip in zip(entries, cut, strict=True)})
    clips = [out[i] for i in range(len(windows))]
    if len({c.key for c in clips}) != len(clips):
        raise ValueError(f"duplicate support windows: {[c.key for c in clips]}")
    return clips


__all__ = [
    "CRUISE_MIN_RPS",
    "FlightClip",
    "LABEL_KEY",
    "REGIMES",
    "SPLITS",
    "SR",
    "STANDBY_MAX_RPS",
    "load_flight_clips",
    "load_support_clips",
    "regime_of",
    "resolve_rig",
]
