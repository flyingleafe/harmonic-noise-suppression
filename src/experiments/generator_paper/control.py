"""Waveform-level rotor-speed controllability probe for the generator paper.

Every probed generator receives three constant per-rotor commands that differ
only in rotor 0 (70, 72 or 74 rev/s; rotors 1-3 fixed at 80, 88, 96 rev/s).
Commanded blade-pass frequencies (two blades, ``2 * rps``) are then READ BACK
FROM THE GENERATED AUDIO: mic-averaged Welch PSD, prominent-peak detection in
130-202 Hz, log-power quadratic interpolation and a unique peak-to-rotor
assignment with an explicit tolerance. Absent peaks stay missing; nothing is
copied from the command labels.

This is an engineering controllability measurement of the generators. It is
not a perceptual-fidelity measurement and not a flight-controller/airframe
dynamics model: commands are constant, all inside the cruise regime, and no
start/stop or out-of-support speeds are requested.

Fits are never re-estimated here. Published-family adaptations are rebuilt from
the parent's ``final_fit_reports.json``; Heutschi's native reference is recovered
from the pinned FLY125 calibration window named by its report. The proposed
model uses the frozen accepted fits of ``input_provenance.json``. The GP and
EONE-RPS adaptations reload the evaluator's frozen checkpoints. Standby/cruise
matched ``*_regimes`` variants reload the frozen ``regime_baselines.json``
manifest prepared by the evaluator (never fitted here).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import optimize, signal

from experiments.generator_paper.baselines import (
    EQ_CENTRES,
    UPSTREAM_SHA,
    YANG_SOURCE_SR,
    HeutschiFit,
    YangFit,
    render_heutschi,
    render_yang,
)
from experiments.generator_paper.cases import FlightClip, load_flight_clips
from experiments.generator_paper.metrics import _jsonable
from experiments.generator_paper.regime_baselines import REGIME_METHODS, regime_render
from experiments.generator_paper.run import (
    EONE_METHODS,
    GP_METHODS,
    SOURCE_ROOT,
    _input_models,
    _render_stochastic,
    eone_render,
    gp_render,
    regime_manifest_info,
    regime_source_files,
)

SCHEMA = "generator-paper-control-probe/1"
METHODS = ("stochastic", "heutschi2020", "yang2025", "yang2025_phase")
ALL_METHODS = METHODS + GP_METHODS + EONE_METHODS + REGIME_METHODS
REPORT_IDS = ("heutschi2020", "yang2025", "yang2025_phase")
RIG = "md2"
SR = 16000
DURATION_S = 4.0
FIXED_RPS = (80.0, 88.0, 96.0)
ROTOR0_RPS = (70.0, 72.0, 74.0)
BLADES = 2
SEED_DOMAIN = 5000

# Readout protocol. With these commands the second shaft order of every rotor
# lies in BAND_HZ while first (70-96 Hz) and third (210-288 Hz) orders do not.
WELCH_NPERSEG = 16384
WELCH_HOP = 4096
BAND_HZ = (130.0, 202.0)
PROMINENCE_DB = 6.0
MIN_PEAK_SEPARATION_HZ = 4.0
PROMINENCE_HALF_WINDOW_HZ = 12.0
MATCH_TOLERANCE_HZ = 6.0  # half of the smallest commanded BPF gap (148 -> 160 Hz)

LABELS = {
    "stochastic": (
        "Proposed calibrated rotor-synchronous stochastic generator; frozen accepted FLY125 "
        "fits from input_provenance.json, not refitted for this probe."
    ),
    "heutschi2020": (
        "Telemetry/target-calibrated SOURCE-STAGE ADAPTATION of the Heutschi et al. (2020) "
        "family (DOI 10.1051/aacus/2020023); not an exact full-system reproduction. "
        "Resampling is driven by the mean of the per-rotor commands, as in the adaptation."
    ),
    "yang2025": (
        "Telemetry/target-calibrated SOURCE-STAGE ADAPTATION of the Yang et al. (2025) "
        "family (DOI 10.23919/EUSIPCO63237.2025.11226465) with published integer-period "
        "event timing; not an exact full-system reproduction."
    ),
    "yang2025_phase": (
        "Stronger IMPLEMENTATION VARIANT of the Yang 2025 source-stage adaptation: "
        "fractional-period phase carry removes integer-period quantization. Not part of "
        "the published method and not an exact full-system reproduction."
    ),
    "lee2026_gp": (
        "Matched-calibration coefficient-space GP ADAPTATION/CONTROL inspired by Lee et al. (2026) "
        "(DOI 10.1121/10.0043469), white Gaussian broadband; frozen FLY125 calibration-only "
        "checkpoint, not refitted for this probe; not a paper-exact reproduction."
    ),
    "lee2026_gp_colored": (
        "Stronger IMPLEMENTATION VARIANT of the lee2026_gp adaptation: same frozen checkpoint, "
        "broadband residual coloured by the calibration residual PSD at the same predicted variance."
    ),
    "doerfler2026_eone": (
        "EONE-RPS curve bank: rotor-RPS SOURCE-STAGE ADAPTATION of Doerfler, Kuntz and Zimmer (2026) "
        "(arXiv:2606.21521) parametric harmonic-plus-noise synthesis, directly fitted to FLY125 "
        "calibration audio without the unavailable pretrained encoder, torque fixed to 1; frozen "
        "checkpoint, not refitted for this probe; not a paper-exact reproduction."
    ),
    **{
        method_id: (
            f"Standby/cruise MATCHED-FIT variant of {method_id.removesuffix('_regimes')}: independent frozen "
            "experts fitted on the FLY125 standby and cruise supports and blended with the proposed model's "
            "rotor-speed power gate; not refitted for this probe; not a paper-exact reproduction."
        )
        for method_id in REGIME_METHODS
    },
}

SOURCES = (
    "src/experiments/generator_paper/control.py",
    "src/experiments/generator_paper/run.py",
    "src/experiments/generator_paper/cases.py",
    "src/experiments/generator_paper/baselines.py",
    "src/experiments/generator_paper/metrics.py",
    "src/data_processing/noise_model/render.py",
    "src/data_processing/noise_model/ou.py",
    "src/data_processing/noise_model/v4.py",
    "src/data_processing/noise_model/resample.py",
    "src/experiments/stochastic_fit/clips.py",
    "pyproject.toml",
    "uv.lock",
)


def conditions() -> list[dict[str, Any]]:
    out = []
    for rotor0 in ROTOR0_RPS:
        rps = (rotor0, *FIXED_RPS)
        out.append(
            {
                "name": f"r0_{rotor0:g}",
                "rotor_rps": list(rps),
                "commanded_bpf_hz": [BLADES * v for v in rps],
            }
        )
    return out


def command_track(rotor_rps: list[float]) -> np.ndarray:
    n = int(round(DURATION_S * SR))
    return np.repeat(np.asarray(rotor_rps, dtype=np.float64)[:, None], n, axis=1)


# ── fit reconstruction ───────────────────────────────────────────────────────


def _require(fit: dict[str, Any], keys: tuple[str, ...], method_id: str) -> None:
    missing = [k for k in keys if k not in fit]
    if missing:
        raise ValueError(
            f"{method_id}: fit report lacks {missing}; refusing to refit or substitute"
        )


def _expect(method_id: str, name: str, value: Any, expected: Any) -> None:
    if value != expected:
        raise ValueError(f"{method_id}: incompatible {name}={value!r}, expected {expected!r}")


def _matrix(method_id: str, name: str, value: Any, n_mics: int) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.shape != (n_mics, EQ_CENTRES.size) or not np.isfinite(out).all():
        raise ValueError(
            f"{method_id}: {name} must be finite with shape {(n_mics, EQ_CENTRES.size)}, got {out.shape}"
        )
    return out


def _positive(method_id: str, name: str, value: Any, *, allow_zero: bool = False) -> float:
    out = float(value)
    if not np.isfinite(out) or out < 0 or (out == 0 and not allow_zero):
        raise ValueError(f"{method_id}: invalid {name}={value!r}")
    return out


def load_fit_reports(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    raw = path.read_bytes()
    data = json.loads(raw)
    methods = data.get("methods") if isinstance(data, dict) else None
    if not isinstance(methods, dict):
        raise ValueError(f"{path}: expected an object with a 'methods' mapping")
    unknown = sorted(set(methods) - set(REPORT_IDS))
    missing = [m for m in REPORT_IDS if m not in methods]
    if unknown or missing:
        raise ValueError(f"{path}: fit report ids missing {missing}, incompatible {unknown}")
    for method_id in REPORT_IDS:
        entry = methods[method_id]
        if not isinstance(entry, dict) or not isinstance(entry.get("fit"), dict):
            raise ValueError(f"{path}: {method_id} has no raw calibration 'fit' object")
        for name in ("source_job", "raw_sha256"):
            if not isinstance(entry.get(name), str) or not entry[name]:
                raise ValueError(f"{path}: {method_id} lacks source reference {name!r}")
    return methods, hashlib.sha256(raw).hexdigest()


def heutschi_from_report(
    fit: dict[str, Any],
    clips: list[FlightClip],
) -> tuple[HeutschiFit, dict[str, Any]]:
    """Rebuild the fitted adaptation; reference = fit_heutschi's float64 native copy."""
    method_id = "heutschi2020"
    _require(
        fit,
        (
            "method",
            "doi",
            "reference_clip",
            "native_sr",
            "reference_rps",
            "eq_centres_hz",
            "slope_db_per_rpm",
            "slope_identified",
            "calibration_clip_keys",
            "reference_looping",
        ),
        method_id,
    )
    _expect(method_id, "method", fit["method"], "Heutschi 2020 source-stage adaptation")
    _expect(method_id, "doi", fit["doi"], "10.1051/aacus/2020023")
    _expect(method_id, "reference_looping", fit["reference_looping"], False)
    _expect(
        method_id, "eq_centres_hz", [float(v) for v in fit["eq_centres_hz"]], EQ_CENTRES.tolist()
    )
    _expect(
        method_id,
        "calibration_clip_keys",
        list(fit["calibration_clip_keys"]),
        [c.key for c in clips],
    )
    found = [c for c in clips if c.key == fit["reference_clip"]]
    if len(found) != 1:
        raise ValueError(
            f"{method_id}: reference clip {fit['reference_clip']!r} not in pinned calibration windows"
        )
    clip = found[0]
    _expect(method_id, "native_sr", int(fit["native_sr"]), int(clip.native_sr))
    reference_rps = _positive(method_id, "reference_rps", fit["reference_rps"])
    recovered_rps = float(clip.rps.mean())
    if abs(recovered_rps - reference_rps) > 1e-9:
        raise ValueError(
            f"{method_id}: recovered reference RPS {recovered_rps!r} != reported {reference_rps!r}"
        )
    slopes = _matrix(method_id, "slope_db_per_rpm", fit["slope_db_per_rpm"], clip.audio.shape[0])
    reference = np.array(clip.native_audio, dtype=np.float64, copy=True)
    source = {
        "reference_clip": clip.key,
        "reference_provenance": clip.provenance,
        "reference_shape": list(reference.shape),
        "native_sr": int(clip.native_sr),
        "reference_float64_sha256": hashlib.sha256(
            np.ascontiguousarray(reference).tobytes()
        ).hexdigest(),
        "reference_rps_reported": reference_rps,
        "reference_rps_recovered": recovered_rps,
        "slope_identified": bool(fit["slope_identified"]),
    }
    return HeutschiFit(reference, int(clip.native_sr), reference_rps, slopes, fit), source


def yang_from_report(fit: dict[str, Any], *, method_id: str, n_mics: int) -> YangFit:
    _require(
        fit,
        (
            "method",
            "doi",
            "upstream_sha",
            "source_sr",
            "pulse_shape",
            "pulse_width_source_samples",
            "blades",
            "phase_carry",
            "modulation_range_hz",
            "white_ratio",
            "relative_wind_speed_jitter",
            "lowpass_hz",
            "lowpass_order",
            "reference_rps",
            "offset_db",
            "slope_db_per_rpm",
            "eq_centres_hz",
        ),
        method_id,
    )
    _expect(method_id, "method", fit["method"], "Yang 2025 source-stage adaptation")
    _expect(method_id, "doi", fit["doi"], "10.23919/EUSIPCO63237.2025.11226465")
    _expect(method_id, "upstream_sha", fit["upstream_sha"], UPSTREAM_SHA)
    _expect(method_id, "source_sr", int(fit["source_sr"]), YANG_SOURCE_SR)
    _expect(method_id, "pulse_shape", fit["pulse_shape"], "rectangular")
    _expect(method_id, "pulse_width_source_samples", int(fit["pulse_width_source_samples"]), 3)
    _expect(method_id, "blades", int(fit["blades"]), BLADES)
    _expect(method_id, "relative_wind_speed_jitter", float(fit["relative_wind_speed_jitter"]), 0.0)
    _expect(method_id, "lowpass_hz", float(fit["lowpass_hz"]), 2400.0)
    _expect(method_id, "lowpass_order", int(fit["lowpass_order"]), 2)
    _expect(
        method_id, "eq_centres_hz", [float(v) for v in fit["eq_centres_hz"]], EQ_CENTRES.tolist()
    )
    phase_carry = fit["phase_carry"]
    if not isinstance(phase_carry, bool):
        raise ValueError(f"{method_id}: phase_carry must be boolean, got {phase_carry!r}")
    _expect(method_id, "phase_carry", phase_carry, method_id == "yang2025_phase")
    return YangFit(
        reference_rps=_positive(method_id, "reference_rps", fit["reference_rps"]),
        offset_db=_matrix(method_id, "offset_db", fit["offset_db"], n_mics),
        slope_db_per_rpm=_matrix(method_id, "slope_db_per_rpm", fit["slope_db_per_rpm"], n_mics),
        phase_carry=phase_carry,
        modulation_range_hz=_positive(
            method_id, "modulation_range_hz", fit["modulation_range_hz"], allow_zero=True
        ),
        white_ratio=_positive(method_id, "white_ratio", fit["white_ratio"], allow_zero=True),
        report=fit,
    )


# ── frequency readout ────────────────────────────────────────────────────────


def mean_psd(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    f, p = signal.welch(
        np.atleast_2d(audio),
        fs=sr,
        window="hann",
        nperseg=WELCH_NPERSEG,
        noverlap=WELCH_NPERSEG - WELCH_HOP,
        detrend=False,  # pyright: ignore[reportArgumentType]
        scaling="density",
        axis=-1,
    )
    return f, p.mean(axis=0)


def detect_peaks(f: np.ndarray, psd: np.ndarray) -> list[dict[str, float]]:
    """Prominent mic-averaged PSD peaks in BAND_HZ with log-quadratic refinement."""
    df = float(f[1] - f[0])
    top = float(psd.max())
    if not np.isfinite(top) or top <= 0:
        return []
    level = 10.0 * np.log10(np.maximum(psd, top * 1e-15))
    distance = max(1, int(np.ceil(MIN_PEAK_SEPARATION_HZ / df)))
    wlen = 2 * int(np.ceil(PROMINENCE_HALF_WINDOW_HZ / df)) + 1
    index, props = signal.find_peaks(level, prominence=PROMINENCE_DB, distance=distance, wlen=wlen)
    peaks = []
    for k, prominence in zip(index, props["prominences"], strict=True):
        if k == 0 or k == level.size - 1:
            continue
        a, b, c = level[k - 1], level[k], level[k + 1]
        curvature = a - 2.0 * b + c
        delta = 0.5 * (a - c) / curvature if curvature < 0 else 0.0
        frequency = float(f[k] + delta * df)
        if BAND_HZ[0] <= frequency <= BAND_HZ[1]:
            peaks.append(
                {
                    "frequency_hz": frequency,
                    "bin_frequency_hz": float(f[k]),
                    "level_db": float(b - 0.25 * (a - c) * delta),
                    "prominence_db": float(prominence),
                }
            )
    return peaks


def assign_peaks(peaks: list[dict[str, float]], rotor_rps: list[float]) -> list[dict[str, Any]]:
    """Unique peak-to-rotor assignment maximizing in-tolerance matches, then minimizing error."""
    targets = BLADES * np.asarray(rotor_rps, dtype=np.float64)
    rotors = [
        {
            "rotor": r,
            "commanded_rps": float(rotor_rps[r]),
            "commanded_bpf_hz": float(targets[r]),
            "matched": False,
            "peak_index": None,
            "measured_bpf_hz": None,
            "measured_rps": None,
            "error_rps": None,
            "prominence_db": None,
        }
        for r in range(targets.size)
    ]
    if not peaks:
        return rotors
    found = np.array([p["frequency_hz"] for p in peaks])
    distance = np.abs(found[:, None] - targets[None, :])
    # Out-of-tolerance pairs carry a cost larger than any in-tolerance assignment total.
    cost = np.where(distance <= MATCH_TOLERANCE_HZ, distance, 1e6 + distance)
    for i, r in zip(*optimize.linear_sum_assignment(cost), strict=True):
        if distance[i, r] > MATCH_TOLERANCE_HZ:
            continue
        measured = float(found[i]) / BLADES
        rotors[r].update(
            {
                "matched": True,
                "peak_index": int(i),
                "measured_bpf_hz": float(found[i]),
                "measured_rps": measured,
                "error_rps": measured - float(rotor_rps[r]),
                "prominence_db": peaks[i]["prominence_db"],
            }
        )
    return rotors


# ── support-aware aggregation ────────────────────────────────────────────────


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "sample_sd": None, "min": None, "max": None, "values": []}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "sample_sd": float(arr.std(ddof=1)) if arr.size > 1 else None,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "values": arr.tolist(),
    }


def _support(cells: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [c for c in cells if c["matched"]]
    errors = [c["error_rps"] for c in matched]
    return {
        "n_commands": len(cells),
        "n_matched": len(matched),
        "matched_command_fraction": len(matched) / len(cells) if cells else None,
        "rps_mae_given_match": float(np.mean(np.abs(errors))) if errors else None,
        "rps_bias_given_match": float(np.mean(errors)) if errors else None,
    }


def summarize_method(records: list[dict[str, Any]], draws: int) -> dict[str, Any]:
    cells = [
        dict(r, condition=rec["condition"], draw=rec["draw"])
        for rec in records
        for r in rec["rotors"]
    ]
    per_draw = []
    for draw in range(draws):
        support = _support([c for c in cells if c["draw"] == draw])
        support["n_unassigned_peaks"] = sum(
            rec["n_unassigned_peaks"] for rec in records if rec["draw"] == draw
        )
        per_draw.append({"draw": draw, **support})
    names = [c["name"] for c in conditions()]
    commands0 = np.asarray(ROTOR0_RPS)
    slopes = []
    for rotor in range(len(FIXED_RPS) + 1):
        draw_rows, values = [], []
        for draw in range(draws):
            by_condition = {
                c["condition"]: c for c in cells if c["draw"] == draw and c["rotor"] == rotor
            }
            missing = [n for n in names if not by_condition[n]["matched"]]
            if missing:
                draw_rows.append({"draw": draw, "slope": None, "missing_conditions": missing})
                continue
            measured = np.array([by_condition[n]["measured_rps"] for n in names])
            slope = float(np.polyfit(commands0, measured, 1)[0])
            values.append(slope)
            draw_rows.append(
                {
                    "draw": draw,
                    "slope": slope,
                    "measured_rps": measured.tolist(),
                    "missing_conditions": [],
                }
            )
        target = 1.0 if rotor == 0 else 0.0
        slopes.append(
            {
                "rotor": rotor,
                "target_slope": target,
                "supported_draw_fraction": len(values) / draws,
                "slope": _stats(values),
                "mean_abs_deviation_from_target": float(
                    np.mean(np.abs(np.asarray(values) - target))
                )
                if values
                else None,
                "per_draw": draw_rows,
            }
        )
    return {
        "pooled": {
            **_support(cells),
            "n_unassigned_peaks": sum(r["n_unassigned_peaks"] for r in records),
        },
        "across_draws": {
            "matched_command_fraction": _stats([d["matched_command_fraction"] for d in per_draw]),
            "rps_mae_given_match": _stats(
                [d["rps_mae_given_match"] for d in per_draw if d["rps_mae_given_match"] is not None]
            ),
        },
        "per_draw": per_draw,
        "by_condition": {n: _support([c for c in cells if c["condition"] == n]) for n in names},
        "by_rotor": {
            str(r): _support([c for c in cells if c["rotor"] == r])
            for r in range(len(FIXED_RPS) + 1)
        },
        "rotor0_command_response_slopes": slopes,
    }


# ── probe ────────────────────────────────────────────────────────────────────


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_probe(
    reports_path: Path,
    inputs: Path,
    out: Path,
    *,
    seed: int,
    draws: int,
    methods: tuple[str, ...] = METHODS,
    gp_checkpoint: Path | None = None,
    eone_checkpoint: Path | None = None,
    regime_fits: Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    methods = tuple(methods)
    clips = (
        load_flight_clips(RIG, "calibration", duration_s=8.0)
        if any(m not in REGIME_METHODS for m in methods)
        else []
    )
    n_mics = clips[0].audio.shape[0] if clips else None
    calibration_keys = [c.key for c in clips]
    renderers: dict[str, Any] = {}
    fit_sources: dict[str, Any] = {}
    reports_sha = None
    if any(m in REPORT_IDS for m in methods):
        reports, reports_sha = load_fit_reports(reports_path)
        if "heutschi2020" in methods:
            heutschi, heutschi_source = heutschi_from_report(reports["heutschi2020"]["fit"], clips)
            renderers["heutschi2020"] = lambda rps, rng: render_heutschi(
                heutschi, rps, sr=SR, rng=rng
            )
            fit_sources["heutschi2020"] = {
                **{k: reports["heutschi2020"][k] for k in ("source_job", "raw_sha256")},
                **heutschi_source,
            }
        for method_id in ("yang2025", "yang2025_phase"):
            if method_id in methods:
                assert n_mics is not None
                fit = yang_from_report(
                    reports[method_id]["fit"], method_id=method_id, n_mics=n_mics
                )
                renderers[method_id] = lambda rps, rng, fit=fit: render_yang(
                    fit, rps, sr=SR, rng=rng
                )
                fit_sources[method_id] = {
                    k: reports[method_id][k] for k in ("source_job", "raw_sha256")
                }
    del clips
    if "stochastic" in methods:
        assert n_mics is not None
        stochastic_fits, stochastic_report = _input_models(inputs, RIG, False)
        renderers["stochastic"] = lambda rps, rng, n_mics=n_mics: _render_stochastic(
            stochastic_fits, rps, sr=SR, rng=rng, rig=RIG, n_mics=n_mics
        )
        fit_sources["stochastic"] = stochastic_report
    gp_sha = None
    for method_id in GP_METHODS:
        if method_id in methods:
            assert gp_checkpoint is not None
            render, report = gp_render(method_id, gp_checkpoint.resolve())
            renderers[method_id] = lambda rps, rng, render=render: render(rps, sr=SR, rng=rng)
            fit_sources[method_id] = report
            gp_sha = report["checkpoint_file_sha256"]
    eone_sha = None
    if any(m in EONE_METHODS for m in methods):
        assert eone_checkpoint is not None
        render, report = eone_render(eone_checkpoint.resolve())
        for method_id in EONE_METHODS:
            if method_id in methods:
                renderers[method_id] = lambda rps, rng, render=render: render(rps, sr=SR, rng=rng)
                fit_sources[method_id] = report
        eone_sha = report["checkpoint_file_sha256"]
    regime_methods = tuple(m for m in methods if m in REGIME_METHODS)
    regime_info = None
    if regime_methods:
        assert regime_fits is not None
        regime_info = regime_manifest_info(regime_fits, regime_methods)
        for method_id in regime_methods:
            render, report = regime_render(method_id, Path(regime_info["path"]))
            if report.get("manifest_sha256") != regime_info["sha256"]:
                raise ValueError(f"{method_id}: regime manifest hash mismatch on reload")
            expert_mics = int(report["n_mics"])
            if n_mics is not None and expert_mics != n_mics:
                raise ValueError(f"{method_id}: {expert_mics} microphones differ from {n_mics}")
            n_mics = expert_mics
            for window in report["calibration_windows"]:
                if window["key"] not in calibration_keys:
                    calibration_keys.append(window["key"])
            renderers[method_id] = lambda rps, rng, render=render: render(rps, sr=SR, rng=rng)
            fit_sources[method_id] = report
    extra_sources: tuple[str, ...] = ()
    if gp_sha:
        from experiments.generator_paper.gp_baseline import GP_SOURCE_FILES

        extra_sources += tuple(GP_SOURCE_FILES)
    if eone_sha:
        from experiments.generator_paper.eone_baseline import EONE_SOURCE_FILES

        extra_sources += tuple(EONE_SOURCE_FILES)
    if regime_methods:
        extra_sources += tuple(regime_source_files(regime_methods))
    conds = conditions()
    n = int(round(DURATION_S * SR))
    freq = None
    psd_store = np.empty((len(methods), len(conds), draws, WELCH_NPERSEG // 2 + 1))
    examples: dict[tuple[str, str], np.ndarray] = {}
    methods_out: dict[str, Any] = {}
    render_seconds: dict[str, list[float]] = {m: [] for m in methods}
    for m_index, method_id in enumerate(methods):
        records = []
        for draw in range(draws):
            for c_index, cond in enumerate(conds):
                # Identical stream for a draw across setpoints: only the command changes.
                rng = np.random.default_rng(np.random.SeedSequence([seed, SEED_DOMAIN, draw]))
                rps = command_track(cond["rotor_rps"])
                tic = time.monotonic()
                audio = np.asarray(renderers[method_id](rps, rng), dtype=np.float32)
                render_seconds[method_id].append(time.monotonic() - tic)
                if audio.shape != (n_mics, n) or not np.isfinite(audio).all():
                    raise ValueError(
                        f"{method_id}/{cond['name']}/draw{draw}: invalid audio {audio.shape}"
                    )
                f, psd = mean_psd(audio.astype(np.float64), SR)
                freq = f
                psd_store[m_index, c_index, draw] = psd
                peaks: list[dict[str, Any]] = detect_peaks(f, psd)
                rotors = assign_peaks(peaks, cond["rotor_rps"])
                assigned = {r["peak_index"]: r["rotor"] for r in rotors if r["matched"]}
                for i, peak in enumerate(peaks):
                    peak["assigned_rotor"] = assigned.get(i)
                records.append(
                    {
                        "condition": cond["name"],
                        "draw": draw,
                        "detected_peaks": peaks,
                        "n_unassigned_peaks": len(peaks) - len(assigned),
                        "rotors": rotors,
                        "audio_rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
                    }
                )
                if draw == 0:
                    examples[(method_id, cond["name"])] = audio
                print(
                    f"{method_id} {cond['name']} draw {draw}: {len(assigned)}/4 matched, "
                    f"{len(peaks)} peaks",
                    flush=True,
                )
        methods_out[method_id] = {
            "label": LABELS[method_id],
            "fit_source": fit_sources[method_id],
            "summary": summarize_method(records, draws),
            "records": records,
        }
    assert freq is not None, "control probe requires at least one render"
    out.mkdir(parents=True, exist_ok=True)
    example_dir = out / "examples"
    example_dir.mkdir(exist_ok=True)
    example_files = []
    for (method_id, name), audio in examples.items():
        rotor_rps = next(c["rotor_rps"] for c in conds if c["name"] == name)
        path = example_dir / f"{method_id}__{name}__draw0.npz"
        np.savez_compressed(
            path,
            generated=audio,
            sr=SR,
            rotor_rps=np.asarray(rotor_rps),
            method=method_id,
            condition=name,
            seed=seed,
            draw=0,
            seed_entropy=np.asarray([seed, SEED_DOMAIN, 0]),
        )
        example_files.append(str(path.relative_to(out)))
    np.savez_compressed(
        out / "control_measurements.npz",
        frequency_hz=freq,
        mean_psd=psd_store,
        methods=np.asarray(methods),
        conditions=np.asarray([c["name"] for c in conds]),
        rotor_rps=np.asarray([c["rotor_rps"] for c in conds]),
        draws=np.arange(draws),
    )
    result = {
        "schema": SCHEMA,
        "interpretation": (
            "Engineering controllability measurement: generated blade-pass frequencies read back "
            "from audio under constant cruise-range commands. Not perceptual fidelity, not a "
            "flight-controller or airframe dynamics model, no start/stop extrapolation. Draw "
            "spread reflects generator stochasticity only, not airframe or training-seed variation."
        ),
        "rig": RIG,
        "sr": SR,
        "duration_s": DURATION_S,
        "n_mics": n_mics,
        "seed": seed,
        "n_draws": draws,
        "methods_probed": list(methods),
        "conditions": conds,
        "random_streams": {
            "seed_sequence": "SeedSequence([seed, 5000, draw]), recreated for every setpoint condition",
            "note": (
                "Each draw uses the identical stream across setpoints so only the command changes. "
                "Heutschi's reference start offset is uniform on a command-dependent admissible "
                "interval and Yang's first-event time depends on the command, so identical streams "
                "do not imply identical sample alignment."
            ),
        },
        "measurement": {
            "psd": "Welch, Hann window, scaling=density, mean over microphones",
            "nperseg": WELCH_NPERSEG,
            "hop": WELCH_HOP,
            "band_hz": list(BAND_HZ),
            "frequency_resolution_hz": float(SR / WELCH_NPERSEG),
            "peak_detection": "scipy.signal.find_peaks on 10*log10 PSD over the full spectrum, kept if refined frequency lies in band_hz",
            "prominence_db": PROMINENCE_DB,
            "min_peak_separation_hz": MIN_PEAK_SEPARATION_HZ,
            "prominence_window_hz": 2 * PROMINENCE_HALF_WINDOW_HZ,
            "refinement": "three-point quadratic interpolation in log power",
            "assignment": "scipy.optimize.linear_sum_assignment of peaks to commanded 2*rps; maximize in-tolerance matches, then minimize |error|",
            "match_tolerance_hz": MATCH_TOLERANCE_HZ,
            "blades": BLADES,
            "measured_rps": "matched peak frequency / blades",
            "missing": "rotors without an in-tolerance peak stay unmatched; no fallback values",
            "orders_in_band": "second shaft order only for all commands (first and third orders lie outside)",
            "slope": "least-squares slope of measured rotor rps vs rotor-0 command over all three setpoints; defined only for draws matched in every setpoint; target 1 for rotor 0, 0 otherwise",
        },
        "methods": methods_out,
        "outputs": {"examples": sorted(example_files), "measurements": "control_measurements.npz"},
        "provenance": {
            "fit_reports": str(reports_path.resolve()) if reports_sha else None,
            "fit_reports_sha256": reports_sha,
            "gp_checkpoint": str(gp_checkpoint.resolve()) if gp_checkpoint and gp_sha else None,
            "gp_checkpoint_sha256": gp_sha,
            "eone_checkpoint": str(eone_checkpoint.resolve())
            if eone_checkpoint and eone_sha
            else None,
            "eone_checkpoint_sha256": eone_sha,
            "regime_fits": regime_info["path"] if regime_info else None,
            "regime_fits_sha256": regime_info["sha256"] if regime_info else None,
            "regime_fits_methods": regime_info["methods"] if regime_info else None,
            "inputs": str(inputs.resolve()) if "stochastic" in methods else None,
            "input_provenance_sha256": _sha(inputs.parent / "input_provenance.json")
            if "stochastic" in methods
            else None,
            "calibration_clip_keys": calibration_keys,
            "refitted_in_this_probe": False,
            "source_sha256": {
                name: _sha(SOURCE_ROOT / name) for name in dict.fromkeys(SOURCES + extra_sources)
            },
            "snapshot_sha256": os.environ.get("NG_SNAPSHOT_SHA256"),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "timing": {
            "render_seconds": render_seconds,
            "total_wall_seconds": time.monotonic() - started,
        },
    }
    (out / "control_probe.json").write_text(json.dumps(_jsonable(result), indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--reports", type=Path, default=Path("results/noise_generator_paper/final_fit_reports.json")
    )
    parser.add_argument("--inputs", type=Path, default=Path("results/noise_generator_paper/inputs"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=list(METHODS))
    parser.add_argument(
        "--gp-checkpoint", type=Path, help="frozen lee2026_gp checkpoint (required for GP methods)"
    )
    parser.add_argument(
        "--eone-checkpoint",
        type=Path,
        help="frozen doerfler2026_eone checkpoint from the evaluator (required for that method)",
    )
    parser.add_argument(
        "--regime-fits",
        type=Path,
        help="frozen regime_baselines.json manifest from the evaluator (required for *_regimes methods)",
    )
    args = parser.parse_args(argv)
    if args.seed < 0 or args.draws < 1:
        parser.error("seed must be non-negative and draws positive")
    if len(set(args.methods)) != len(args.methods):
        parser.error("methods must be unique")
    if any(m in GP_METHODS for m in args.methods) != (args.gp_checkpoint is not None):
        parser.error("--gp-checkpoint is required exactly when a lee2026_gp method is selected")
    if any(m in EONE_METHODS for m in args.methods) != (args.eone_checkpoint is not None):
        parser.error("--eone-checkpoint is required exactly when doerfler2026_eone is selected")
    if any(m in REGIME_METHODS for m in args.methods) != (args.regime_fits is not None):
        parser.error("--regime-fits is required exactly when a *_regimes method is selected")
    if (args.out / "control_probe.json").exists():
        parser.error(f"{args.out} already holds a control probe; use a new output directory")
    run_probe(
        args.reports,
        args.inputs,
        args.out,
        seed=args.seed,
        draws=args.draws,
        methods=tuple(args.methods),
        gp_checkpoint=args.gp_checkpoint,
        eone_checkpoint=args.eone_checkpoint,
        regime_fits=args.regime_fits,
    )


if __name__ == "__main__":
    main()
