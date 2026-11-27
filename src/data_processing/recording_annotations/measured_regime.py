"""Assess a take's speed regime from documented, audio-clock shaft telemetry.

This is annotation evidence, never a tachometry prior. No waveform is read; no
resampling, forward fill, motor sorting, or substitution of commanded speed.
``spans_s`` use the original Frame clock and list simultaneous plateau spans,
not speed tracks. Slow drift within a plateau is allowed.
"""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from .scan import SampleScan, SeriesRef


DATASETS = frozenset(
    {
        "DREGON-frames",
        "michaels-frames",
        "michaels-test-frames",
        "new_motor_data-frames",
        "D3SET-frames",
    }
)
RECIPE = "measured-shaft-regime/3"
# Same physical observation duration, slew and change-band thresholds as the
# acoustic analysis (tracking.steady_tachometry._MIN_S, _confirm, _gap_regime).
_MIN_WINDOW_S = 1.5
_SLEW_FLOOR = 0.25  # rev/s per s
_SLEW_REL = 0.006  # relative per s
_HALF_AGREEMENT = 0.18  # rev/s: upper end of acoustic half-window tolerance
_CHANGE_FLOOR = 0.8  # rev/s
_CHANGE_REL = 0.027
# Timestamp/coverage gates specific to telemetry, not motor physics.
# A sub-window cadence variation of 25 ms or less is negligible on the
# >=1.5 s regime scale, even for nominally kHz bursty telemetry (DREGON).
_CADENCE_ALLOWANCE_S = 0.025
_MAX_GAP_S = 0.25
_MIN_COVERAGE = 0.8
_STRONG_COVERAGE = 0.95
_MIXED_FRACTION = 0.15
# A reference is a post-fit check, not an estimate on a sparse track. An
# unobserved window edge is allowed only within the logger cadence.
_REFERENCE_COVERAGE = 0.9


class _RangeReader(Protocol):
    def read(self, offset: int, length: int) -> bytes: ...


def _tolerance(center: np.ndarray) -> np.ndarray:
    return np.maximum(_CHANGE_FLOOR, _CHANGE_REL * center)


def _runs(mask: np.ndarray):
    edges = np.flatnonzero(np.diff(mask, prepend=False, append=False))
    return zip(edges[::2], edges[1::2], strict=True)


def _finish(result: dict, regime: str, reason: str, note: str) -> dict:
    result.update(
        suggested=regime, speed_regime=regime, reason=reason, evidence=f"{result['source']}: {note}"
    )
    return result


def assess(
    rps_ct,
    times_s,
    *,
    audio_start_s: float,
    audio_end_s: float,
    role: str,
    provenance: Mapping[str, object],
) -> dict:
    """Return regime plus quantitative evidence, without looking at PCM.

    ``rps_ct`` is (physical shaft, time) in rev/s; ``times_s`` uses the given
    audio role's clock. Provenance supplies ``source``, ``basis``, the rig's
    ``expected_rotors``, and whether this observation is ``authoritative``.
    Missing/negative values and long timestamp gaps provide no support. Exact
    repeats at one timestamp count once; conflicting duplicate shaft values
    break support on that shaft. Out-of-order rows are never reordered.
    """
    expected = provenance.get("expected_rotors")
    result: dict = {
        "recipe": RECIPE,
        "role": role,
        "source": str(provenance.get("source", "shaft telemetry")),
        "basis": str(provenance.get("basis", "unknown")),
        "n_expected": expected,
        "strong": False,
        "spans_s": [],
        "coverage": 0.0,
    }
    t, x = np.asarray(times_s, dtype=np.float64), np.asarray(rps_ct)
    duration = audio_end_s - audio_start_s
    if (
        not np.isfinite(audio_start_s)
        or not np.isfinite(audio_end_s)
        or duration <= 0
        or t.ndim != 1
        or x.ndim != 2
        or x.shape[0] < 1
        or x.shape[1] != t.size
        or x.dtype.kind not in "fiu"
    ):
        return _finish(
            result,
            "unknown",
            "invalid_track_or_audio_span",
            "invalid recording span or telemetry shape",
        )
    if not np.all(np.isfinite(t)):
        return _finish(
            result,
            "unknown",
            "invalid_timestamps",
            "nonfinite timestamps cannot be placed on the audio clock",
        )
    if t.size and (t[0] < audio_start_s or t[-1] >= audio_end_s or np.any(t[1:] < t[:-1])):
        keep = (t >= audio_start_s) & (t < audio_end_s)
        t, x = t[keep], x[:, keep]
    t = t - audio_start_s  # work near zero, including sources with Unix clocks
    dt = np.diff(t)
    result.update(n_measured=int(x.shape[0]), n_in_audio=int(t.size))
    if np.any(dt < 0):
        return _finish(
            result,
            "unknown",
            "unordered_timestamps",
            "out-of-order timestamps inside the audio span; no time reordering",
        )
    disputed = None
    if t.size and np.any(dt == 0):
        first = np.flatnonzero(np.r_[True, dt > 0])
        last = np.r_[first[1:], t.size]
        disputed = np.zeros((x.shape[0], first.size), dtype=bool)
        for k in np.flatnonzero(last - first > 1):
            a, b = first[k], last[k]
            disputed[:, k] = np.any(x[:, a:b] != x[:, a : a + 1], axis=1)
        result.update(
            duplicate_rows=int(t.size - first.size),
            disputed_timestamps=int(np.any(disputed, axis=0).sum()),
        )
        t, x = t[first], x[:, first]
        dt = np.diff(t)
    if dt.size < 5:
        return _finish(
            result, "unknown", "insufficient_samples", "fewer than six distinct in-audio samples"
        )
    nominal_gap = float(np.median(dt))
    gap = min(_MAX_GAP_S, max(_CADENCE_ALLOWANCE_S, 4 * nominal_gap))
    valid = np.isfinite(x) & (x >= 0)
    if disputed is not None:
        valid &= ~disputed
    intervals = (dt <= gap) & valid[:, :-1] & valid[:, 1:]
    joint = np.all(intervals, axis=0)
    coverage_s = float(dt[joint].sum())
    coverage = min(1.0, coverage_s / duration)
    rows = []
    for i, (values, finite, support) in enumerate(zip(x, valid, intervals, strict=True)):
        supported = values[finite]
        rows.append(
            {
                "shaft_channel": i,
                "coverage": float(dt[support].sum()) / duration,
                "median_rev_s": float(np.median(supported)) if supported.size else None,
                "plateau_s": 0.0,
                "variable_s": 0.0,
                "off_s": 0.0,
            }
        )
    result.update(coverage=coverage, coverage_s=coverage_s, maximum_interstamp_s=gap, shafts=rows)
    complete = (
        isinstance(expected, (int, np.integer))
        and not isinstance(expected, bool)
        and expected == x.shape[0]
    )

    spans: list[list[float]] = []
    totals = {"plateau_s": 0.0, "variable_s": 0.0, "off_s": 0.0}
    first_center: np.ndarray | None = None
    span_center: np.ndarray | None = None
    different_bands = False
    for a, b in _runs(joint):
        elapsed = float(t[b] - t[a])
        if elapsed < _MIN_WINDOW_S:
            continue
        n = int(elapsed / _MIN_WINDOW_S)
        # Partition a contiguous run into near-equal >=1.5 s windows. No gap is
        # bridged, and the last window keeps the sub-window remainder.
        edges = np.searchsorted(t[a : b + 1], np.linspace(t[a], t[b], n + 1)) + a
        for i, j in zip(edges[:-1], edges[1:], strict=True):
            elapsed = float(t[j] - t[i])
            mid = i + int(np.searchsorted(t[i : j + 1], (t[i] + t[j]) / 2))
            if min(mid - i, j + 1 - mid) < 3:
                continue
            block = x[:, i : j + 1]
            q10, center, q90 = np.quantile(block, (0.1, 0.5, 0.9), axis=1)
            left, right = np.median(x[:, i:mid], axis=1), np.median(x[:, mid : j + 1], axis=1)
            half_delta = np.abs(left - right)
            slew = np.maximum(_SLEW_FLOOR, _SLEW_REL * center)
            half_elapsed = (t[mid] + t[j] - t[i] - t[mid - 1]) / 2
            off = np.all(block == 0, axis=1)
            crosses_zero = ~off & np.any(block == 0, axis=1)
            steady = (
                ~off
                & ~crosses_zero
                & (q90 - q10 <= _tolerance(center))
                & (half_delta <= _HALF_AGREEMENT + slew * half_elapsed)
            )
            for k, row in enumerate(rows):
                state = "off_s" if off[k] else "plateau_s" if steady[k] else "variable_s"
                row[state] += elapsed
            state = (
                "off_s" if np.all(off) else "plateau_s" if np.all(off | steady) else "variable_s"
            )
            totals[state] += elapsed
            if state != "plateau_s" or not complete:
                continue
            if first_center is None:
                first_center = center
            elif np.any(np.abs(center - first_center) > _tolerance(first_center)):
                different_bands = True
            start, end = float(t[i] + audio_start_s), float(t[j] + audio_start_s)
            if (
                spans
                and start == spans[-1][1]
                and span_center is not None
                and np.all(np.abs(center - span_center) <= _tolerance(span_center))
            ):
                spans[-1][1] = end
            else:
                span_center = center
                spans.append([start, end])
    analyzed = sum(totals.values())
    result.update(
        totals,
        analyzed_s=analyzed,
        spans_s=spans,
        thresholds={
            "min_window_s": _MIN_WINDOW_S,
            "half_agreement_rev_s": _HALF_AGREEMENT,
            "slew_rev_s_per_s": [_SLEW_FLOOR, _SLEW_REL],
            "plateau_band_rev_s": [_CHANGE_FLOOR, _CHANGE_REL],
        },
    )
    if not complete:
        return _finish(
            result,
            "unknown",
            "incomplete_rotor_set",
            f"{x.shape[0]} usable shaft channel(s) for {expected!r} expected; no full-rig conclusion",
        )
    if coverage < _MIN_COVERAGE or analyzed < max(_MIN_WINDOW_S, 0.7 * duration):
        return _finish(
            result,
            "unknown",
            "sparse_joint_coverage",
            f"{coverage:.1%} joint valid coverage; only {analyzed:.2f} of {duration:.2f} s have sustained evidence",
        )
    off, varying, plateau = totals["off_s"], totals["variable_s"], totals["plateau_s"]
    if (
        off >= 0.99 * duration
        and not np.any(x[valid] > 0)
        and not result.get("disputed_timestamps")
    ):
        regime, reason = "off", "all_shafts_documented_zero"
    elif (
        plateau > 0
        and (different_bands or off + varying >= _MIXED_FRACTION * analyzed)
        or min(off, varying) >= _MIXED_FRACTION * analyzed
    ):
        regime, reason = "mixed", "distinct_speed_bands_or_phase_types"
    elif varying >= 0.6 * analyzed:
        regime, reason = "variable", "sustained_shaft_variation"
    elif (
        plateau >= 0.7 * analyzed and coverage >= 0.9 and off + varying < _MIXED_FRACTION * analyzed
    ):
        regime, reason = "constant", "shaft_plateaux_with_bounded_drift"
    else:
        regime, reason = "unknown", "inconclusive_phase_coverage"
    result["strong"] = (
        regime != "unknown"
        and provenance.get("authoritative") is True
        and coverage >= _STRONG_COVERAGE
        and analyzed >= 0.9 * duration
    )
    return _finish(
        result,
        regime,
        reason,
        f"{x.shape[0]} shafts, {coverage:.1%} joint coverage; {plateau:.2f} s plateau, "
        f"{varying:.2f} s varying, {off:.2f} s documented zero over {duration:.2f} s. "
        "Plateaux use quantile spread and split-window slew tolerances, not exact equality.",
    )


def _track(sc: SampleScan, dataset: str) -> tuple[str, str, float, dict] | None:
    path, role, scale = "rps", "audio", 1.0
    basis, authoritative = "measured", True
    if dataset == "DREGON-frames":
        path = "motors_measured"
        source = "DREGON native motor.measured (rev/s, DREGON release and data catalog)"
    elif dataset in ("michaels-frames", "michaels-test-frames"):
        p = sc.meta.get("provenance", {})
        gain = sc.meta.get("rps_scale")
        if (
            p.get("builder") != "data_processing.sources.michaels.build_frame"
            or not isinstance(gain, (int, float))
            or not np.isfinite(gain)
            or gain <= 0
            or gain == 1
            or not p.get("alignment")
            or not p.get("calibration")
        ):
            return None
        source = "Michael's Motor:Speed:* (RPM/60 with published acoustic alignment and scale correction)"
        basis = "measured_acoustically_calibrated"
    elif dataset == "new_motor_data-frames":
        clock, rps = sc.meta.get("clock", {}), sc.meta.get("rps", {})
        if (
            clock.get("source") != "data_processing.sources.new_motor_data.AUDIO_CLOCK"
            or rps.get("unit") != "rev/s"
            or not str(rps.get("source", "")).startswith("dat Motor")
        ):
            return None
        source = "November DAT Motor record 10090 (RPM/60; AUDIO_CLOCK alignment)"
        basis = "measured_acoustically_aligned"
    elif dataset == "D3SET-frames":
        track = sc.meta.get("telemetry_tracks", {}).get("bpf_label", {})
        if track.get("on_audio_clock") is not True or track.get("unit") != "Hz":
            return None
        path, scale = "telemetry/bpf_label", 0.5
        role = "audio" if "audio" in sc.series else "array_audio"
        source = "D3SET aligned actuator BPF/2 (optical PWM calibration plus manual acoustic offset; not measured in flight)"
        basis, authoritative = "acoustically_calibrated_bpf", False
    else:
        return None
    return (
        path,
        role,
        scale,
        {"source": source, "basis": basis, "expected_rotors": 4, "authoritative": authoritative},
    )


def _stamps(shard: _RangeReader, sc: SampleScan, ref: SeriesRef) -> np.ndarray | None:
    if ref.index == "grid" and ref.rate_hz and ref.rate_hz > 0 and ref.t_start_s is not None:
        return ref.t_start_s + np.arange(ref.n_time, dtype=np.float64) / ref.rate_hz
    if ref.index == "stamps" and ref.time_axis is not None and ref.index_start_ticks is not None:
        stamp_field = f"{ref.path}#{ref.dims[ref.time_axis]}.stamps"
        span = sc.fields.get(stamp_field)
        if span is None:
            return None
        ticks = np.load(io.BytesIO(shard.read(*span)), allow_pickle=False)
        if ticks.ndim == 1 and ticks.shape == (ref.n_time,) and ticks.dtype.kind == "i":
            # The tdframe-v1 codec stores relative StampIndex ticks; add the
            # signed origin before floating conversion (including Unix clocks).
            return (ticks + ref.index_start_ticks) / 1e9
    return None


def _read_shaft_rows(shard: _RangeReader, ref: SeriesRef, start: int, stop: int) -> np.ndarray:
    dtype = np.dtype(ref.dtype)
    size, n, nc = dtype.itemsize, stop - start, ref.n_channels
    offset = ref.data_offset
    if (ref.time_axis == 1) != ref.fortran:
        assert ref.n_time is not None
        values = np.empty((nc, n), dtype=dtype)
        for c in range(nc):
            values[c] = np.frombuffer(
                shard.read(offset + (c * ref.n_time + start) * size, n * size), dtype
            )
        return values
    return (
        np.frombuffer(shard.read(offset + start * nc * size, n * nc * size), dtype).reshape(n, nc).T
    )


def _shaft_ref(sc: SampleScan, path: str) -> SeriesRef | None:
    ref = sc.series.get(path)
    if (
        ref is None
        or ref.container != "npy"
        or len(ref.shape) != 2
        or ref.time_axis not in (0, 1)
        or not ref.n_time
        or np.dtype(ref.dtype).kind not in "fiu"
        or not (
            ref.offset <= ref.data_offset
            and ref.data_offset + ref.n_time * ref.n_channels * np.dtype(ref.dtype).itemsize
            <= ref.offset + ref.length
        )
    ):
        return None
    return ref


def from_scan(shard: _RangeReader, sc: SampleScan, dataset: str) -> dict | None:
    """Read only the eligible native NPY series and its clock, no audio bytes.

    The entire native stamp field is needed to locate/verify the overlap, but
    values are read only inside the audio role's support, using exact shard
    ranges (not coalesced across fields). An absent/unproven path returns None.
    """
    selected = _track(sc, dataset)
    if selected is None:
        return None
    path, role, scale, provenance = selected
    ref, audio = _shaft_ref(sc, path), sc.series.get(role)
    if (
        ref is None
        or audio is None
        or not audio.is_audio
        or audio.t_start_s is None
        or audio.t_end_s is None
    ):
        return None
    stamps = _stamps(shard, sc, ref)
    if stamps is None or not np.all(np.isfinite(stamps)):
        return None
    a, b = audio.t_start_s, audio.t_end_s
    if role == "array_audio":
        stream = sc.meta.get("audio_streams", {}).get(role, {})
        if (sr := stream.get("sample_rate")) and (n := stream.get("num_samples")):
            # D3SET padded matrix.wav to align with drone.wav. Those zeros are
            # not recorder support and cannot qualify as motor-off evidence.
            first = stream.get("leading_digital_zero_samples", 0)
            last = n - stream.get("trailing_digital_zero_samples", 0)
            if not 0 <= first < last <= n:
                return None
            a, b = a + first / sr, min(b, a + last / sr)
    if np.any(stamps[1:] < stamps[:-1]):
        return _finish(
            {
                "coverage": 0.0,
                "strong": False,
                "spans_s": [],
                "role": role,
                "source": provenance["source"],
                "basis": provenance["basis"],
                "recipe": RECIPE,
            },
            "unknown",
            "unordered_native_clock",
            "cannot bound an unordered native stamp field",
        )
    start, stop = np.searchsorted(stamps, [a, b]).tolist()
    values = (
        _read_shaft_rows(shard, ref, start, stop) if stop > start else np.empty((ref.n_channels, 0))
    )
    if scale != 1.0:
        values = values * scale
    result = assess(
        values, stamps[start:stop], audio_start_s=a, audio_end_s=b, role=role, provenance=provenance
    )
    result["entry"] = path
    return result


def _reference_source(sc: SampleScan, dataset: str) -> tuple[str, float, str, str] | None:
    # Only actual shaft feedback has independent speed values. In particular,
    # the publisher's acoustically adjusted D3SET BPF is not truth for a fit.
    if dataset == "DREGON-frames":
        return (
            "motors_measured",
            1.0,
            "native_shaft_feedback",
            "DREGON motor.measured in rev/s; the publisher manually aligned audio timestamps"
            " to motor logs with stated error <0.1 s",
        )
    if dataset in ("michaels-frames", "michaels-test-frames"):
        p = sc.meta.get("provenance", {})
        if p.get("builder") != "data_processing.sources.michaels.build_frame" or not p.get(
            "alignment"
        ):
            return None
        return (
            "motor_speed",
            1 / 60,
            "native_feedback_acoustically_aligned",
            "Michael's original DatCon Motor:Speed:* RPM/60, without the audio-fitted rps_scale;"
            " the published audio alignment was estimated acoustically",
        )
    if dataset == "new_motor_data-frames" and (selected := _track(sc, dataset)) is not None:
        path, _, scale, _ = selected
        return (
            path,
            scale,
            "native_feedback_acoustically_aligned",
            "DJI DAT Motor record 10090 RPM/60, no acoustic gain correction;"
            " the WAV clock alignment was estimated acoustically",
        )
    return None


def _reference_stats(
    values: np.ndarray, stamps: np.ndarray, a: float, b: float, gap_limit: float
) -> tuple[np.ndarray, float, float] | None:
    if not np.all(np.isfinite(values) & (values >= 0)):
        return None
    dt = np.diff(stamps)
    duplicate = dt == 0
    if np.any(duplicate):
        if np.any(values[:, 1:][:, duplicate] != values[:, :-1][:, duplicate]):
            return None
        values = values[:, np.r_[True, ~duplicate]]
        dt = dt[~duplicate]
    if dt.size < 5:
        return None
    gap = float(np.max(dt))
    coverage = min(1.0, float(stamps[-1] - stamps[0]) / (b - a))
    if coverage < _REFERENCE_COVERAGE or max(gap, stamps[0] - a, b - stamps[-1]) > gap_limit:
        return None
    return np.quantile(values, (0.1, 0.5, 0.9), axis=1), coverage, gap


def reference_windows(
    shard: _RangeReader,
    sc: SampleScan,
    dataset: str,
    windows: Sequence[Mapping[str, object]],
) -> list[dict | None]:
    """Summarize independent shaft feedback for matching, already fitted PCM windows.

    One result per input window, including ``None`` for unsupported windows.
    This accessor is only for the primary ``audio`` role. It uses each
    window's actual in-audio stamps and values, never interpolation,
    out-of-audio values, a setpoint or the acoustic fit rates themselves.
    Centers/ranges retain native channel order; they do not label the PCM fit.
    """
    out: list[dict | None] = [None] * len(windows)
    if not windows or (selected := _reference_source(sc, dataset)) is None:
        return out
    path, scale, basis, source = selected
    ref, audio = _shaft_ref(sc, path), sc.series.get("audio")
    if (
        ref is None
        or ref.n_channels != 4
        or audio is None
        or not audio.is_audio
        or audio.t_start_s is None
        or audio.t_end_s is None
    ):
        return out
    stamps = _stamps(shard, sc, ref)
    if stamps is None or not np.all(np.isfinite(stamps)) or np.any(stamps[1:] < stamps[:-1]):
        return out
    first, last = map(int, np.searchsorted(stamps, [audio.t_start_s, audio.t_end_s]))
    cadence = np.diff(stamps[first:last])
    positive = cadence[cadence > 0]
    if positive.size < 5:
        return out
    gap_limit = min(_MAX_GAP_S, max(_CADENCE_ALLOWANCE_S, 4 * float(np.median(positive))))
    slices: list[tuple[int, int, int, float, float]] = []
    for k, window in enumerate(windows):
        a, b = window.get("start_s"), window.get("end_s")
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            continue
        if not (audio.t_start_s <= a < b <= audio.t_end_s):
            continue
        i, j = map(int, np.searchsorted(stamps, [a, b]))
        if j - i >= 6:
            slices.append((i, j, k, a, b))
    # Consecutive or overlapping fit windows can share an exact NPY slab.
    # Do not coalesce byte ranges across fields or read unrelated channels.
    slices.sort()
    left = 0
    while left < len(slices):
        start, stop = slices[left][:2]
        right = left + 1
        while right < len(slices) and slices[right][0] <= stop:
            stop = max(stop, slices[right][1])
            right += 1
        raw = _read_shaft_rows(shard, ref, start, stop)
        for i, j, k, a, b in slices[left:right]:
            stats = _reference_stats(raw[:, i - start : j - start], stamps[i:j], a, b, gap_limit)
            if stats is None:
                continue
            quantiles, coverage, gap = stats
            quantiles *= scale
            out[k] = {
                "source": source,
                "basis": basis,
                "entry": path,
                "coverage": coverage,
                "maximum_interstamp_s": gap,
                "rates_rev_s": quantiles[1].tolist(),
                "ranges_rev_s": quantiles[[0, 2]].T.tolist(),
                "shaft_channels": list(range(ref.n_channels)),
            }
        left = right
    return out
