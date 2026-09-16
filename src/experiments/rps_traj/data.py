"""REAL rotor-speed trajectories on the campaign's one common analysis grid.

Every rig the campaign models arrives here as a list of :class:`Flight`: a
``(4, T)`` rev/s array on a :data:`RATE_HZ` = 100 Hz grid, with NaN wherever
telemetry is missing. :func:`load_rig` serves every rig in :data:`RIGS` off
the project's published ``*-frames`` datasets; :func:`load_frames_dataset` is
the generic adapter behind it (contract: a ``td.Frame`` with an ``rps``
Series of dims ``("rotor", "time")`` in rev/s, plus ``meta.system.rig`` and
``meta.system.native_rate_hz``).

Frozen conventions — a model is only comparable to real data if it is judged
on the same grid and the same samples:

- **Common grid.** 100 Hz, linear interpolation from the first native sample
  (:func:`to_common_grid`). Telemetry faster than 100 Hz is first low-passed
  at 40 Hz (zero-phase 4th-order Butterworth) so that decimation cannot fold
  rotor-speed ripple down into the band we model. Michael's DJI rig logs at
  ~29.5 Hz and is interpolated as is: for that rig **no content above
  ~15 Hz exists** (its own Nyquist), so any model comparison for `michaels`
  is informative only below ~15 Hz.
- **Airborne samples.** :func:`airborne_segments`, the frozen rule below.
  Nothing outside those slices enters a statistic.
- **Rotor order.** Every :attr:`Flight.rps` out of :func:`load_rig` is
  permuted to the MIXER order ``[RFront, LFront, LBack, RBack]`` of
  :data:`tracking.rotors.MIXER`, so the mode projection
  ``MIXER.T @ w / 4`` means the same thing on every rig
  (:data:`ROTOR_TO_MIXER`, :func:`infer_rotor_to_mixer`).

NaNs survive interpolation: a gap wider than two native sample periods stays
NaN on the common grid instead of being bridged by a straight line.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from itertools import permutations
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

# The campaign's common analysis rate (Hz). Frozen, and the grid the shipped
# fits are stated on, so it has exactly one definition — in the model package —
# and stays part of this module's vocabulary (``RATE_HZ`` in ``__all__``).
from data_processing.trajectory_model import RATE_HZ

#: Anti-alias corner (Hz) applied before decimating telemetry faster than
#: :data:`RATE_HZ` onto the common grid, and its Butterworth order.
ANTIALIAS_HZ = 40.0
ANTIALIAS_ORDER = 4

#: Airborne rule (frozen): threshold is half the 90th percentile of the
#: rotor-mean speed, every rotor must exceed it, runs are eroded by
#: :data:`ERODE_S` at both ends and kept only if at least :data:`MIN_RUN_S`
#: long.
ERODE_S = 1.0
MIN_RUN_S = 5.0
AIRBORNE_PERCENTILE = 90.0
AIRBORNE_FRACTION = 0.5

#: FLY103/FLY108 (``michaels-test-frames``) are the same PHYSICAL vehicle as
#: FLY124/FLY125 (``michaels-frames``) — all four DatCon logs carry flight
#: controller serial ``mcID(SN)|041DE70827``, ``ACType|M100`` and
#: ``mcVer|v3.1.6.108``, and the yaw trim (the airframe's own torque
#: imbalance) agrees to a 1.20 rev/s spread about 5.43 rev/s
#: (``results/rps_traj/explore/summary.json``, ``michaels_same_drone``) — so
#: by default they are pooled into the single rig ``michaels``. Flip to
#: ``False`` to model the held-out recordings as their own rig
#: :data:`MICHAELS_TEST_RIG`.
MICHAELS_TEST_IS_SAME_RIG = True

#: The held-out michaels recordings' own rig id when they are NOT pooled.
MICHAELS_TEST_RIG = "michaels_test"

#: The rigs of the campaign, in report order. VID's single-motor bench runs
#: (``vid_m3508_bench``, 1 rotor) ship inside ``VID-frames`` but are not a
#: rig: they are not the vehicle, so :func:`load_rig` drops them.
RIGS = (
    "michaels",
    "dregon",
    "neurobem_quad",
    "pitcn_quad",
    "nanobench_cf21b",
    "vid_m100",
    "blackbird_quad",
)

#: Number of rotors a campaign rig must have.
NUM_ROTORS = 4

#: Rotor axis permutation per rig: ``rps[list(ROTOR_TO_MIXER[rig])]`` is the
#: MIXER order ``[RFront, LFront, LBack, RBack]`` of
#: :data:`tracking.rotors.MIXER`. Sources publish their own array order, so
#: without this the mode projection would mean a different thing per rig.
#:
#: - ``michaels`` — ``sources.michaels.ROTOR_ORDER`` IS the mixer order.
#: - ``dregon`` — motor rows reindexed from ``coordinates.mat['rotorsPos']``
#:   (``results/rps_traj/explore/findings.md``).
#: - ``neurobem_quad`` — published order ``[back_right, front_right,
#:   back_left, front_left]`` (README columns ``mot 1..4``).
#: - ``pitcn_quad`` — ``sources.pitcn.ROTOR_LAYOUT`` ``[front_left, back_left,
#:   back_right, front_right]`` (derived from the publisher's torque map; only
#:   the diagonal pairing is certain).
#: - ``nanobench_cf21b`` — ``sources.nanobench.ROTOR_LAYOUT`` ``[front_right,
#:   back_right, back_left, front_left]`` (stock Crazyflie X numbering).
#: - ``vid_m100`` — DJI M100 numbering, same as Michael's M100; its rotor
#:   means ``[87.9, 78.2, 82.5, 74.7]`` reproduce that rig's trim pattern.
#:
#: ``blackbird_quad`` is deliberately absent: the order of MIT's MotorRPM
#: array is undocumented (repo issue #26, unanswered), so it is inferred from
#: the data by :func:`infer_rotor_to_mixer`.
ROTOR_TO_MIXER: dict[str, tuple[int, int, int, int]] = {
    "michaels": (0, 1, 2, 3),
    MICHAELS_TEST_RIG: (0, 1, 2, 3),
    "dregon": (2, 3, 0, 1),
    "neurobem_quad": (1, 3, 2, 0),
    "pitcn_quad": (3, 0, 1, 2),
    "nanobench_cf21b": (0, 3, 2, 1),
    "vid_m100": (0, 1, 2, 3),
}

#: Pre-airborne samples count as idle/standby (rather than motors-off) when
#: the rotor-mean speed exceeds this. See :func:`ground_level`.
IDLE_MIN_RPS = 5.0

#: Published frames datasets behind :func:`load_rig`, with the rotor-speed
#: entry names tried in order (first present wins; the chosen one is recorded
#: in :attr:`Flight.source`). A dataset may carry several rigs, so frames that
#: declare a ``meta.system.rig`` other than the requested one are dropped
#: (michaels/dregon frames declare none — the dataset itself is the rig).
RIG_SOURCES: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "michaels": (
        ("michaels-frames", ("rps",)),
        ("michaels-test-frames", ("rps",)),
    ),
    "dregon": (("DREGON-frames", ("motors_measured", "motors_command")),),
    "neurobem_quad": (("NeuroBEM-frames", ("rps",)),),
    "pitcn_quad": (("PITCN-frames", ("rps",)),),
    "nanobench_cf21b": (("NanoBench-frames", ("rps",)),),
    "vid_m100": (("VID-frames", ("rps",)),),
    "blackbird_quad": (("Blackbird-frames", ("rps",)),),
}


def _rig_id(rig: str, dataset: str) -> str:
    """The rig label a dataset's flights get — read at CALL time, so flipping
    :data:`MICHAELS_TEST_IS_SAME_RIG` takes effect without reimporting."""
    if dataset == "michaels-test-frames" and not MICHAELS_TEST_IS_SAME_RIG:
        return MICHAELS_TEST_RIG
    return rig


@dataclass(frozen=True)
class Flight:
    """One flight's rotor-speed trajectory on the common grid.

    ``rps`` is ``(n_rotors, T)`` rev/s at ``fs`` Hz, float64, NaN where
    telemetry is missing; ``t0`` is the time of the first sample on the
    recording's own clock (seconds), kept only for traceability back to the
    source recording. ``source`` names where the numbers came from, e.g.
    ``"DREGON-frames:motors_measured"``.
    """

    rig: str
    flight: str
    fs: float
    t0: float
    rps: np.ndarray
    source: str

    @property
    def n_rotors(self) -> int:
        return int(self.rps.shape[0])

    @property
    def duration_s(self) -> float:
        return float(self.rps.shape[1]) / float(self.fs)


# ─── The common grid ──────────────────────────────────────────────────────────


def _interp_nans(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(y with NaNs linearly bridged, nan mask)`` for one 1-D track."""
    bad = np.isnan(y)
    if not bad.any() or bad.all():
        return y, bad
    idx = np.arange(y.size, dtype=np.float64)
    filled = y.copy()
    filled[bad] = np.interp(idx[bad], idx[~bad], y[~bad])
    return filled, bad


def to_common_grid(t: np.ndarray, x: np.ndarray, native_rate_hz: float | None) -> np.ndarray:
    """Resample ``x`` ``(n_rotors, T)`` sampled at times ``t`` onto 100 Hz.

    The grid starts at ``t[0]`` and steps ``1 / RATE_HZ``. Faster-than-100 Hz
    telemetry is first passed through a zero-phase 4th-order Butterworth
    low-pass at :data:`ANTIALIAS_HZ` (``sosfiltfilt``); NaN runs are bridged
    for the filter and re-masked afterwards, because ``sosfiltfilt`` would
    otherwise smear a single NaN over the whole track. ``native_rate_hz``
    ``None`` is estimated from the median sample spacing of ``t``.

    Missing telemetry stays missing: a grid point whose NEAREST source sample
    is NaN comes out NaN, so an interior dropout is not bridged by a straight
    line. Timestamp jitter is not missingness — DREGON's ~1 kHz motor stamps
    wander by ±1 ms and occasionally skip 24 ms, which must not punch holes
    in the 100 Hz output.
    """
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if t.size != x.shape[-1]:
        raise ValueError(f"timestamps {t.size} do not match samples {x.shape[-1]}")
    if t.size == 0:
        return np.zeros((x.shape[0], 0), dtype=np.float64)

    dt = np.diff(t)
    if native_rate_hz:
        native = float(native_rate_hz)
    else:
        native = 1.0 / float(np.median(dt)) if dt.size else RATE_HZ
    if not np.isfinite(native) or native <= 0.0:
        raise ValueError(f"nonsensical native rate {native!r}")

    if native > RATE_HZ:
        sos = butter(ANTIALIAS_ORDER, ANTIALIAS_HZ, btype="low", fs=native, output="sos")
        # sosfiltfilt needs more samples than its padlen; short tracks are
        # left unfiltered (they carry no resolvable low-frequency content
        # anyway) rather than raising.
        padlen = 3 * (2 * ANTIALIAS_ORDER + 1)
        if x.shape[-1] > padlen:
            filtered = np.empty_like(x)
            for r in range(x.shape[0]):
                filled, bad = _interp_nans(x[r])
                if bad.all():
                    filtered[r] = np.nan
                    continue
                row = sosfiltfilt(sos, filled)
                row[bad] = np.nan
                filtered[r] = row
            x = filtered

    n = int(np.floor((t[-1] - t[0]) * RATE_HZ)) + 1
    grid = t[0] + np.arange(n, dtype=np.float64) / RATE_HZ
    nearest = np.clip(np.searchsorted(t, grid, side="left"), 0, t.size - 1)
    left = np.maximum(nearest - 1, 0)
    closer = np.abs(t[left] - grid) < np.abs(t[nearest] - grid)
    nearest = np.where(closer, left, nearest)
    out = np.full((x.shape[0], n), np.nan, dtype=np.float64)
    for r in range(x.shape[0]):
        valid = ~np.isnan(x[r])
        if valid.sum() < 2:
            continue
        out[r] = np.interp(grid, t[valid], x[r, valid], left=np.nan, right=np.nan)
        out[r, ~valid[nearest]] = np.nan
    return out


def _trim_all_nan(rps: np.ndarray) -> tuple[np.ndarray, int]:
    """Drop leading/trailing columns that are NaN in every rotor.

    Returns the trimmed array and the number of dropped leading columns (so
    the caller can keep ``t0`` honest).
    """
    if rps.size == 0:
        return rps, 0
    good = np.flatnonzero(~np.isnan(rps).all(axis=0))
    if good.size == 0:
        return rps[:, :0], 0
    lo, hi = int(good[0]), int(good[-1]) + 1
    return np.ascontiguousarray(rps[:, lo:hi]), lo


# ─── The frozen airborne rule ─────────────────────────────────────────────────


def _runs(mask: np.ndarray) -> Iterator[tuple[int, int]]:
    """``(start, stop)`` of every maximal True run of a 1-D boolean mask."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for start, stop in zip(edges[::2], edges[1::2]):
        yield int(start), int(stop)


def _rotor_mean(rps: np.ndarray) -> np.ndarray:
    """``nanmean`` over rotors, spelled out: all-NaN columns are normal in
    real telemetry and must not raise "Mean of empty slice" noise."""
    finite = np.isfinite(rps)
    count = finite.sum(axis=0)
    return np.where(
        count > 0, np.where(finite, rps, 0.0).sum(axis=0) / np.maximum(count, 1), np.nan
    )


def _airborne_mask(rps: np.ndarray) -> np.ndarray:
    """The frozen per-sample airborne test, BEFORE erosion: every rotor
    strictly above half the 90th percentile of the rotor-mean speed (a NaN
    rotor sample is ``False``). Empty array when nothing is finite."""
    if rps.shape[-1] == 0:
        return np.zeros(rps.shape[-1], dtype=bool)
    finite = np.isfinite(rps)
    if not finite.any():
        return np.zeros(rps.shape[-1], dtype=bool)
    m = _rotor_mean(rps)
    thr = AIRBORNE_FRACTION * float(np.percentile(m[np.isfinite(m)], AIRBORNE_PERCENTILE))
    return np.asarray((finite & (np.where(finite, rps, 0.0) > thr)).all(axis=0))


def airborne_segments(rps: np.ndarray, fs: float) -> list[slice]:
    """The airborne slices of a flight — FROZEN RULE, never tuned per rig.

    ``m = nanmean over rotors``; ``thr = 0.5 * nanpercentile(m, 90)``;
    ``mask`` = every rotor strictly above ``thr`` (a NaN rotor sample is
    ``False``); every True run is eroded by :data:`ERODE_S` = 1 s at both
    ends (dropping spin-up/spin-down transients that no stationary model
    should be asked to match) and kept only if what survives is at least
    :data:`MIN_RUN_S` = 5 s long.
    """
    rps = np.atleast_2d(np.asarray(rps, dtype=np.float64))
    mask = _airborne_mask(rps)
    erode = int(round(ERODE_S * float(fs)))
    min_len = int(round(MIN_RUN_S * float(fs)))
    out: list[slice] = []
    for start, stop in _runs(mask):
        lo, hi = start + erode, stop - erode
        if hi - lo >= min_len:
            out.append(slice(lo, hi))
    return out


def ground_level(flight: Flight) -> float | None:
    """The rig's idle/standby rotor speed in ``flight``, or ``None``.

    The median rotor speed over the PRE-AIRBORNE samples whose rotor-mean
    speed exceeds :data:`IDLE_MIN_RPS` = 5 rev/s — the level the motors hold
    on the ground with the props armed but not lifting. "Pre-airborne" is
    everything before the first sample that passes :func:`_airborne_mask`
    (the UN-eroded test, so the erosion margin at the start of a run is not
    mistaken for ground). ``None`` when the flight has no such part: the
    recordings that start already airborne (NeuroBEM segments, PI-TCN bags,
    Blackbird) and any whose ground part is motors-off only.
    """
    rps = np.asarray(flight.rps, dtype=np.float64)
    airborne = np.flatnonzero(_airborne_mask(rps))
    pre = rps[:, : int(airborne[0])] if airborne.size else rps
    if pre.shape[-1] == 0:
        return None
    mean = _rotor_mean(pre)
    idle = pre[:, np.isfinite(mean) & (mean > IDLE_MIN_RPS)]
    if idle.size == 0 or not np.isfinite(idle).any():
        return None
    return float(np.nanmedian(idle))


# ─── Rotor order ──────────────────────────────────────────────────────────────

#: The two diagonal rotor pairs in MIXER order: rotors on the same rotation
#: sense, which a quad trims yaw with. Their zero-lag correlation is well
#: above the adjacent pairs' on every rig measured so far
#: (``results/rps_traj/explore/findings.md``), which is what makes the rotor
#: order identifiable from data at all.
DIAGONAL_PAIRS = ((0, 2), (1, 3))

#: How much a candidate diagonal pairing must beat the published order by,
#: in summed zero-lag correlation, before :func:`infer_rotor_to_mixer`
#: reorders the rotors. Sampling noise alone moves the score by ~0.02 on a
#: 200 s flight of uncoupled rotors, while a real diagonal signature is worth
#: 1-2 (Blackbird: 1.72 for the published order against -0.57 / -0.61 for the
#: other two pairings) — so anything in between is "no evidence", and no
#: evidence means keep the order the publisher shipped.
PAIRING_MARGIN = 0.05


def _zero_lag_corr(rps: np.ndarray) -> np.ndarray:
    """``(4, 4)`` zero-lag Pearson correlation, per-rotor demeaned, NaN-safe."""
    x = np.asarray(rps, dtype=np.float64)
    x = x - np.nanmean(x, axis=1, keepdims=True)
    x = np.where(np.isfinite(x), x, 0.0)
    cov = x @ x.T
    scale = np.sqrt(np.maximum(np.diag(cov), np.finfo(np.float64).tiny))
    return cov / np.outer(scale, scale)


def infer_rotor_to_mixer(rps: np.ndarray) -> tuple[int, int, int, int]:
    """Infer a rig's rotor permutation from ``(4, T)`` rev/s, correlation-only.

    Picks the permutation whose two :data:`DIAGONAL_PAIRS` carry the largest
    summed zero-lag correlation, and only if it wins by :data:`PAIRING_MARGIN`
    — otherwise the published order stands.

    **Only the diagonal PAIRING is identifiable** this way: the data cannot
    say which diagonal is ``RFront``/``LBack`` and which is
    ``LFront``/``RBack``, nor which member of a pair is front. So the result
    is the canonical representative of the winning pairing (rotor 0 stays in
    slot 0). That is enough for every campaign statistic — the mixer's
    roll/pitch columns swap sign or role under the unresolved freedom, while
    ``common``/``yaw`` (~78 % of the variance) and the diagonal-vs-adjacent
    correlation contrast do not.
    """
    rps = np.asarray(rps, dtype=np.float64)
    if rps.shape[0] != NUM_ROTORS:
        raise ValueError(f"expected {NUM_ROTORS} rotors, got {rps.shape[0]}")
    corr = _zero_lag_corr(rps)
    identity = (0, 1, 2, 3)

    def score(perm: tuple[int, ...]) -> float:
        return float(sum(corr[perm[i], perm[j]] for i, j in DIAGONAL_PAIRS))

    # One representative per distinct diagonal pairing; the identity's own
    # pairing is the incumbent and is never re-tested.
    incumbent = score(identity)
    seen = {frozenset(frozenset(pair) for pair in DIAGONAL_PAIRS)}
    best, best_score = identity, incumbent + PAIRING_MARGIN
    for perm in permutations(range(NUM_ROTORS)):
        pairing = frozenset(frozenset(perm[i] for i in pair) for pair in DIAGONAL_PAIRS)
        if pairing in seen:
            continue
        seen.add(pairing)
        if (candidate := score(perm)) > best_score:
            best, best_score = perm, candidate
    return best  # type: ignore[return-value]


def _airborne_block(flights: Sequence[Flight]) -> np.ndarray:
    """Every airborne segment of ``flights``, per-segment per-rotor demeaned
    and concatenated — the sample set :func:`infer_rotor_to_mixer` reads, with
    between-flight level offsets removed so they cannot fake a correlation."""
    blocks = [
        seg - np.nanmean(seg, axis=1, keepdims=True)
        for f in flights
        for seg in (f.rps[:, s] for s in airborne_segments(f.rps, f.fs))
    ]
    if not blocks:
        return np.empty((NUM_ROTORS, 0))
    return np.concatenate(blocks, axis=1)


def rotor_to_mixer(rig: str, flights: Sequence[Flight]) -> tuple[int, int, int, int]:
    """``ROTOR_TO_MIXER[rig]`` when the layout is published, else inferred."""
    known = ROTOR_TO_MIXER.get(rig)
    if known is not None:
        return known
    block = _airborne_block(flights)
    if block.shape[-1] == 0:
        return (0, 1, 2, 3)
    return infer_rotor_to_mixer(block)


# ─── Frames-dataset adapters ──────────────────────────────────────────────────


def _time_index(series: Any) -> Any:
    try:
        return series.indexes["time"]
    except (AttributeError, KeyError, TypeError) as exc:  # pragma: no cover - guard
        raise ValueError(f"series carries no time index: {series!r}") from exc


def _series_rate(series: Any) -> float:
    """Sample rate of a ``td.Series``: exact on a grid, implied by stamps."""
    idx = _time_index(series)
    rate = getattr(idx, "sr", None)
    if rate:
        return float(rate)
    stamps = np.asarray(idx.timestamps, dtype=np.float64)
    if stamps.size > 1:
        return float((stamps.size - 1) / (stamps[-1] - stamps[0]))
    raise ValueError(f"cannot determine the sample rate of a {type(idx).__name__}")


def _series_times(series: Any, n: int) -> np.ndarray:
    """Absolute sample times of a ``td.Series`` on the recording's clock."""
    idx = _time_index(series)
    getter = getattr(idx, "sample_times", None)
    if callable(getter):
        return np.asarray(getter(), dtype=np.float64)
    stamps = np.asarray(idx.timestamps, dtype=np.float64)
    if stamps.size == n:
        return stamps
    return float(getattr(idx, "t_start", 0.0)) + np.arange(n) / _series_rate(series)


def _meta_group(frame: Any, group: str) -> dict[str, Any]:
    if "meta" not in frame:
        return {}
    meta = frame["meta"]
    if group not in meta:
        return {}
    sub = meta[group]
    try:
        return {k: sub[k] for k in sub}
    except TypeError:  # pragma: no cover - a scalar under a group name
        return {}


def _meta_scalar(frame: Any, key: str, default: Any = None) -> Any:
    if "meta" not in frame:
        return default
    meta = frame["meta"]
    # Not `meta.get`: a tdframe meta group is a mapping-LIKE container whose
    # only guaranteed protocol is `in` and `[]`.
    return meta[key] if key in meta else default  # noqa: SIM401


def _flight_from_frame(
    frame: Any, dataset: str, rps_keys: Sequence[str], rig: str | None
) -> Flight | None:
    """One published frame → one :class:`Flight`, or ``None`` if it carries no
    rotor-speed track (DREGON's bench and clean-source samples do not)."""
    key = next((k for k in rps_keys if k in frame), None)
    if key is None:
        return None
    series = frame[key]
    values = np.asarray(series.data, dtype=np.float64)
    values = values[None, :] if values.ndim == 1 else values
    if values.shape[-1] < 2:
        return None
    times = _series_times(series, values.shape[-1])
    system = _meta_group(frame, "system")
    native = system.get("native_rate_hz")
    native = float(native) if native else _series_rate(series)

    grid = to_common_grid(times - times[0], values, native)
    grid, dropped = _trim_all_nan(grid)
    if grid.shape[-1] == 0:
        return None
    flight = str(_meta_scalar(frame, "recording_id", "unknown"))
    rig_id = str(rig or system.get("rig") or dataset)
    return Flight(
        rig=rig_id,
        flight=flight,
        fs=RATE_HZ,
        t0=float(times[0]) + dropped / RATE_HZ,
        rps=grid,
        source=f"{dataset}:{key}",
    )


def _flights_from_dataset(
    dataset: str,
    rps_keys: Sequence[str],
    rig: str | None = None,
    *,
    match_rig: str | None = None,
) -> list[Flight]:
    """Stream one published ``tdframe-v1`` dataset into :class:`Flight`s.

    One frame is decoded at a time (``iter_published_frames``); the returned
    list is sorted by flight id so every downstream statistic is
    order-deterministic. ``match_rig`` drops frames that declare a different
    ``meta.system.rig`` (a dataset may carry several rigs; frames declaring
    none are always kept, which is how michaels/dregon work).
    """
    from data_processing.streams import iter_published_frames

    flights: list[Flight] = []
    for frame in iter_published_frames(dataset):
        if match_rig is not None:
            declared = _meta_group(frame, "system").get("rig")
            if declared is not None and str(declared) != match_rig:
                continue
        flight = _flight_from_frame(frame, dataset, rps_keys, rig)
        if flight is not None:
            flights.append(flight)
    return sorted(flights, key=lambda f: f.flight)


def load_frames_dataset(name: str, rps_key: str = "rps") -> list[Flight]:
    """Every flight of a telemetry frames dataset, on the common grid.

    The generic adapter for the rigs being ingested into the campaign: the
    dataset's frames must carry ``rps`` (dims ``("rotor", "time")``, rev/s)
    and ``meta.system.rig`` / ``meta.system.native_rate_hz``. Each frame's
    ``meta.system.rig`` becomes :attr:`Flight.rig`, so one dataset may hold
    several rigs.
    """
    return _flights_from_dataset(name, (rps_key,))


def load_rig(rig: str) -> list[Flight]:
    """Every real flight of ``rig``, on the common grid, in MIXER rotor order.

    ``"michaels"`` — the DJI Matrice 100 rig: FLY124/FLY125 from
    ``michaels-frames`` plus the held-out FLY103/FLY108 from
    ``michaels-test-frames`` (same airframe, pooled unless
    :data:`MICHAELS_TEST_IS_SAME_RIG` is flipped). Rotor speeds come from the
    calibrated ``rps`` track logged at ~29.5 Hz.

    ``"dregon"`` — the DREGON Mikrokopter: ``DREGON-frames``, preferring the
    tachometer ``motors_measured`` and falling back to the (publish-time
    cleaned) ``motors_command``; :attr:`Flight.source` records which.

    The five public rigs (``neurobem_quad``, ``pitcn_quad``,
    ``nanobench_cf21b``, ``vid_m100``, ``blackbird_quad``) read their own
    telemetry-only frames dataset. Frames that are not this rig's, and frames
    that do not carry exactly :data:`NUM_ROTORS` rotors (VID's single-motor
    bench runs), are dropped.

    The rotor axis is permuted to the mixer order ``[RFront, LFront, LBack,
    RBack]`` — from :data:`ROTOR_TO_MIXER` where the layout is published, and
    from the rig's own diagonal-pair correlations otherwise
    (:func:`infer_rotor_to_mixer`; ``blackbird_quad`` only).
    """
    try:
        sources = RIG_SOURCES[rig]
    except KeyError:
        raise KeyError(f"unknown rig {rig!r} (known: {sorted(RIG_SOURCES)})") from None
    out: list[Flight] = []
    for dataset, rps_keys in sources:
        out.extend(
            f
            for f in _flights_from_dataset(dataset, rps_keys, _rig_id(rig, dataset), match_rig=rig)
            if f.n_rotors == NUM_ROTORS
        )
    perm = rotor_to_mixer(rig, out)
    if perm != (0, 1, 2, 3):
        rows = list(perm)
        out = [replace(f, rps=np.ascontiguousarray(f.rps[rows])) for f in out]
    return sorted(out, key=lambda f: (f.rig, f.flight))


__all__ = [
    "AIRBORNE_FRACTION",
    "AIRBORNE_PERCENTILE",
    "ANTIALIAS_HZ",
    "ANTIALIAS_ORDER",
    "DIAGONAL_PAIRS",
    "ERODE_S",
    "IDLE_MIN_RPS",
    "MICHAELS_TEST_IS_SAME_RIG",
    "MICHAELS_TEST_RIG",
    "MIN_RUN_S",
    "NUM_ROTORS",
    "PAIRING_MARGIN",
    "RATE_HZ",
    "RIGS",
    "RIG_SOURCES",
    "ROTOR_TO_MIXER",
    "Flight",
    "airborne_segments",
    "ground_level",
    "infer_rotor_to_mixer",
    "load_frames_dataset",
    "load_rig",
    "rotor_to_mixer",
    "to_common_grid",
]
