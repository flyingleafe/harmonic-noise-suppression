"""Fixed-length settled-cruise windows of every campaign rig, for spectral comparison.

The unit of the joint spectral-distance analysis is one WINDOW: ``(4, 400)``
rotor speeds in rev/s, mixer rotor order ``[RFront, LFront, LBack, RBack]``,
:data:`WINDOW_S` = 16 s at :data:`RATE_HZ_OUT` = 25 Hz, NOT demeaned.
:func:`load_segments` is the only place a window is made:

* **Source.** :func:`~experiments.rps_traj.phases.load_flights` for every rig
  of :data:`~experiments.rps_traj.data.RIGS`, on the 100 Hz common grid.  Only
  MEASURED rotor speed enters: DREGON's command-only recordings
  (``motors_command``, audio-recovered standby) are excluded.
* **Samples.** :func:`~experiments.rps_traj.phases.settled_cruise` slices,
  split into their all-rotor-finite runs; a NaN is never bridged.
* **Rate.** Each run is brought to 25 Hz by ``scipy.signal.resample_poly``
  (polyphase, its own Kaiser anti-alias FIR, 40 input taps per side at 100 Hz)
  and :data:`EDGE_S` is cut from both ends, so no window sample sees the
  filter's padding beyond the run.
* **Windows.** A run holds ``floor(L / 400)`` consecutive windows, the
  remainder split evenly between its two ends; of a recording's windows (in
  time order, over all its runs) at most :data:`MAX_WINDOWS_PER_RECORDING`
  evenly spaced ones are kept, so a long flight cannot dominate a centroid.
* **Recordings.** ``flight`` is the PHYSICAL recording: NeuroBEM publishes one
  frame per Vicon-dropout-free segment (``<flight>_seg_<k>``), and all
  segments of a flight are one recording (one fold unit, one cap).
* **Calibration.** In a rig with at least two eligible recordings the
  lexicographically first one is reserved whole for calibration
  (``calibration=True``) and never validates; a single-recording rig
  (Blackbird) has none.

The returned inventory is JSON-serialisable: dataset pins, per-rig exclusions
and short counts, and the recording ids per role.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from experiments.rps_traj.cruise_api import NUM_ROTORS
from experiments.rps_traj.data import RIG_SOURCES, RIGS, Flight
from experiments.rps_traj.phases import load_flights, settled_cruise

#: Window length (s), output rate (Hz) and the samples a window holds.
WINDOW_S = 16.0
RATE_HZ_OUT = 25.0
WINDOW_SAMPLES = int(round(WINDOW_S * RATE_HZ_OUT))

#: Cut from both ends of every downsampled run (s): covers the support of
#: ``resample_poly``'s filter (0.4 s per side at 100 -> 25 Hz).
EDGE_S = 0.5

#: Most windows one physical recording contributes.
MAX_WINDOWS_PER_RECORDING = 4

#: DREGON's command channel is not a measured rotor speed.
COMMAND_SOURCE = "motors_command"

_SEGMENT_RE = re.compile(r"^(?P<flight>.+)_seg_(?P<seg>\d+)$")

#: Rigs whose published frames are pieces of one physical recording.
SEGMENTED_RIGS = frozenset({"neurobem_quad"})


def recording_id(rig: str, segment: str) -> str:
    """The physical recording a published frame id belongs to."""
    if rig in SEGMENTED_RIGS:
        m = _SEGMENT_RE.match(segment)
        if m is None:
            raise ValueError(f"{rig} frame {segment!r} is not '<flight>_seg_<k>'")
        return m.group("flight")
    return segment


@dataclass(frozen=True)
class _Run:
    """One all-finite settled-cruise run at :data:`RATE_HZ_OUT`, edges cut."""

    segment: str
    source: str
    start_s: float  # time of x[:, 0] from the segment's first grid sample
    x: np.ndarray  # (4, n)


@dataclass(frozen=True)
class _Window:
    segment: str
    source: str
    start_s: float
    x: np.ndarray  # (4, WINDOW_SAMPLES)


def _finite_runs(rps: np.ndarray, sl: slice) -> list[tuple[int, int]]:
    """``[(start, stop))`` absolute indices of the all-finite runs of ``sl``."""
    ok = np.isfinite(rps[:, sl]).all(axis=0)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], ok.astype(np.int8), [0]])))
    return [(sl.start + int(a), sl.start + int(b)) for a, b in zip(edges[::2], edges[1::2])]


def _downsample(x: np.ndarray, fs: float) -> np.ndarray:
    """``x`` ``(4, n)`` at ``fs`` to :data:`RATE_HZ_OUT`, :data:`EDGE_S` cut per end."""
    ratio = Fraction(RATE_HZ_OUT).limit_denominator(1000) / Fraction(fs).limit_denominator(1000)
    y = resample_poly(x, ratio.numerator, ratio.denominator, axis=-1, padtype="line")
    edge = int(np.ceil(EDGE_S * RATE_HZ_OUT))
    return y[:, edge : y.shape[1] - edge] if y.shape[1] > 2 * edge else y[:, :0]


def _runs(flight: Flight, stats: dict[str, int]) -> list[_Run]:
    rps = np.asarray(flight.rps, dtype=np.float64)
    fs = float(flight.fs)
    edge = int(np.ceil(EDGE_S * RATE_HZ_OUT))
    out = []
    for sl in settled_cruise(rps, fs):
        stats["settled_slices"] += 1
        stats["nonfinite_samples"] += int((~np.isfinite(rps[:, sl]).all(axis=0)).sum())
        for a, b in _finite_runs(rps, sl):
            stats["finite_runs"] += 1
            x = _downsample(rps[:, a:b], fs)
            if x.shape[1] < WINDOW_SAMPLES:
                stats["short_runs"] += 1
                continue
            out.append(_Run(flight.flight, flight.source, a / fs + edge / RATE_HZ_OUT, x))
    return out


def _windows(runs: Sequence[_Run]) -> list[_Window]:
    """Every window of ``runs`` in time order, remainder split per run."""
    out = []
    for run in runs:
        n = run.x.shape[1]
        k = n // WINDOW_SAMPLES
        lo = (n - k * WINDOW_SAMPLES) // 2
        for j in range(k):
            a = lo + j * WINDOW_SAMPLES
            out.append(
                _Window(
                    run.segment,
                    run.source,
                    run.start_s + a / RATE_HZ_OUT,
                    run.x[:, a : a + WINDOW_SAMPLES],
                )
            )
    return out


def _cap(windows: Sequence[_Window]) -> list[_Window]:
    """At most :data:`MAX_WINDOWS_PER_RECORDING`, evenly spaced in time order."""
    if len(windows) <= MAX_WINDOWS_PER_RECORDING:
        return list(windows)
    idx = np.round(np.linspace(0, len(windows) - 1, MAX_WINDOWS_PER_RECORDING)).astype(int)
    return [windows[i] for i in idx]


def dataset_pins() -> dict[str, str | None]:
    """The ``dload.lock`` pin of every dataset behind the campaign rigs."""
    from data_processing.streams import REPO_ROOT

    with open(REPO_ROOT / "dload.lock", "rb") as fh:
        pins = tomllib.load(fh).get("datasets", {})
    names = sorted({name for sources in RIG_SOURCES.values() for name, _ in sources})
    return {name: pins.get(name) for name in names}


def load_segments(
    rigs: Sequence[str] = RIGS,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    """``(rows, waves, inventory)`` of every settled-cruise window of ``rigs``.

    ``waves[i]`` is window ``i``: ``(4, WINDOW_SAMPLES)`` rev/s at
    :data:`RATE_HZ_OUT`, mixer rotor order, not demeaned.  ``rows[i]`` names it:
    ``rig``, ``flight`` (physical recording id, unique across rigs),
    ``segment`` (published frame id), ``start_s`` (window start, seconds from
    the segment's first common-grid sample), ``source`` (dataset:channel) and
    ``calibration``.  Rows run rig by rig in ``rigs`` order, then by recording
    and time.  ``inventory`` is JSON-serialisable (see the module docstring).
    """
    rows: list[dict[str, Any]] = []
    waves: list[np.ndarray] = []
    owner: dict[str, str] = {}
    per_rig: dict[str, Any] = {}
    for rig in rigs:
        flights = load_flights(rig)
        stats: dict[str, int] = dict.fromkeys(
            ("settled_slices", "nonfinite_samples", "finite_runs", "short_runs"), 0
        )
        command_only = sorted(f.flight for f in flights if COMMAND_SOURCE in f.source)
        measured = [f for f in flights if COMMAND_SOURCE not in f.source]
        if any(f.n_rotors != NUM_ROTORS for f in measured):
            raise ValueError(f"{rig}: a flight without {NUM_ROTORS} rotors reached the loader")

        segments: dict[str, list[str]] = {}
        runs: dict[str, list[_Run]] = {}
        slices: dict[str, int] = {}
        for f in measured:
            rec = recording_id(rig, f.flight)
            segments.setdefault(rec, []).append(f.flight)
            before = stats["settled_slices"]
            runs.setdefault(rec, []).extend(_runs(f, stats))
            slices[rec] = slices.get(rec, 0) + stats["settled_slices"] - before
        for rec, segs in segments.items():
            if len(set(segs)) != len(segs):
                raise ValueError(f"{rig}: recording {rec!r} has repeated segments {segs}")
            if rec in owner:
                raise ValueError(f"recording id {rec!r} is in both {owner[rec]} and {rig}")
            owner[rec] = rig
        no_cruise = sorted(rec for rec in segments if not slices[rec])
        short = sorted(rec for rec in segments if slices[rec] and not runs[rec])

        available = {rec: _windows(rs) for rec, rs in sorted(runs.items()) if rs}
        eligible = sorted(available)
        calibration = eligible[0] if len(eligible) >= 2 else None

        kept: dict[str, int] = {}
        for rec in eligible:
            for w in _cap(available[rec]):
                rows.append(
                    {
                        "rig": rig,
                        "flight": rec,
                        "segment": w.segment,
                        "start_s": round(float(w.start_s), 4),
                        "source": w.source,
                        "calibration": rec == calibration,
                    }
                )
                waves.append(w.x)
                kept[rec] = kept.get(rec, 0) + 1

        n_avail = sum(len(available[rec]) for rec in eligible)
        per_rig[rig] = {
            "datasets": [name for name, _ in RIG_SOURCES[rig]],
            "frames": len(flights),
            "excluded_command_only": command_only,
            "recordings": len(segments),
            "multi_segment_recordings": {
                rec: sorted(segs) for rec, segs in sorted(segments.items()) if len(segs) > 1
            },
            "no_settled_cruise": no_cruise,
            "short_recordings": short,
            **stats,
            "eligible": eligible,
            "calibration": calibration,
            "validation": [rec for rec in eligible if rec != calibration],
            "windows_available": n_avail,
            "windows_capped_out": n_avail - sum(kept.values()),
            "windows": sum(kept.values()),
            "calibration_windows": kept.get(calibration, 0) if calibration else 0,
            "windows_per_recording": kept,
        }

    inventory = {
        "params": {
            "window_s": WINDOW_S,
            "rate_hz": RATE_HZ_OUT,
            "window_samples": WINDOW_SAMPLES,
            "edge_s": EDGE_S,
            "max_windows_per_recording": MAX_WINDOWS_PER_RECORDING,
            "resampler": "scipy.signal.resample_poly(padtype='line')",
        },
        "dataset_pins": dataset_pins(),
        "rigs": per_rig,
        "totals": {
            "windows": len(rows),
            "calibration_windows": sum(r["calibration"] for r in rows),
            "recordings": len({(r["rig"], r["flight"]) for r in rows}),
        },
    }
    out = np.stack(waves) if waves else np.zeros((0, NUM_ROTORS, WINDOW_SAMPLES), dtype=np.float64)
    return rows, out, inventory


__all__ = [
    "EDGE_S",
    "MAX_WINDOWS_PER_RECORDING",
    "RATE_HZ_OUT",
    "WINDOW_S",
    "WINDOW_SAMPLES",
    "dataset_pins",
    "load_segments",
    "recording_id",
]
