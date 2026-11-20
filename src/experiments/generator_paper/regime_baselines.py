"""Standby/cruise-matched fitting and blending of the external noise-generator baselines.

Our generator renders md2 audio from TWO accepted FLY125 fits, one on 12 s of
standby support and one on 64 s of cruise support, blended per sample by the
slowest-rotor gate. This module gives every external comparator the same
treatment, as six additional method ids (the pooled ids stay unchanged):

* **Supports.** The exact windows are read from the hash-verified accepted md2
  fits named by ``input_provenance.json`` (``pool.specs``), never re-derived
  from the regular calibration grid: standby FLY125@2+4, @6+4, @10+4 and
  cruise FLY125@16+8, @32+8, @48+8, @64+8, @96+8, @112+8, @128+8, @144+8
  (seconds on the recording's audio clock, ``rps_refined``). They are cut by
  :func:`experiments.generator_paper.cases.load_support_clips`, the same
  pinned decode and whole-recording resampling as the evaluation loader.
  These are the core fit supports only: ours also used auxiliary FLY125
  calibration and external priors, so the comparison is NOT controlled for
  total data exposure.
* **Experts.** Each family is fitted independently on the standby and on the
  cruise support with its unchanged hyperparameters, selection and loss. The
  two GP broadband variants share one checkpoint per regime. Heutschi's
  standby reference is the exact concatenation of the three contiguous 4 s
  supports (2-14 s; sample-adjacent slices of one resampling, all
  constituents recorded), because a 4 s reference cannot cover a 4 s
  evaluation trajectory; no looping, padding or extra audio. Cruise windows
  (8 s) are used as supplied.
* **Frozen states.** Every completed expert is written atomically (state +
  record with checksums) before any evaluation; workers only reload, verify
  and render. Paths in the manifest are relative to its directory, so a
  pulled output tree stays loadable.
* **Blend.** ``sqrt(1 - w) * standby + sqrt(w) * cruise`` with
  ``w = render.regime_blend_weight(rps)`` (0 at <= 45, 1 at >= 65 rev/s of the
  slowest rotor) and independent expert streams from ``render.regime_seeds``:
  the experts' POWERS are interpolated, not their parameters. An expert whose
  gain is zero over the whole track is not rendered.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from data_processing.noise_model.render import REGIME_ORDER, regime_blend_weight, regime_seeds
from experiments.generator_paper.baselines import (
    HeutschiFit,
    YangFit,
    fit_heutschi,
    fit_yang,
    render_heutschi,
    render_yang,
)
from experiments.generator_paper.cases import (
    CRUISE_MIN_RPS,
    LABEL_KEY,
    STANDBY_MAX_RPS,
    FlightClip,
    load_support_clips,
    regime_of,
)
from experiments.generator_paper.metrics import _jsonable

SCHEMA = "generator-paper-regime-baselines/1"
RECORD_SCHEMA = "generator-paper-regime-expert/1"
FITTING_TREATMENT = "standby_cruise_matched"
REGIME_METHODS = (
    "heutschi2020_regimes",
    "yang2025_regimes",
    "yang2025_phase_regimes",
    "lee2026_gp_regimes",
    "lee2026_gp_colored_regimes",
    "doerfler2026_eone_regimes",
)
#: Matched id -> the pooled method it pairs with.
BASE_METHOD = {
    "heutschi2020_regimes": "heutschi2020",
    "yang2025_regimes": "yang2025",
    "yang2025_phase_regimes": "yang2025_phase",
    "lee2026_gp_regimes": "lee2026_gp",
    "lee2026_gp_colored_regimes": "lee2026_gp_colored",
    "doerfler2026_eone_regimes": "doerfler2026_eone",
}
#: Matched id -> the family whose two fitted experts it renders from.
FAMILY = {
    "heutschi2020_regimes": "heutschi2020",
    "yang2025_regimes": "yang2025",
    "yang2025_phase_regimes": "yang2025_phase",
    "lee2026_gp_regimes": "lee2026_gp",
    "lee2026_gp_colored_regimes": "lee2026_gp",
    "doerfler2026_eone_regimes": "doerfler2026_eone",
}
FAMILIES = tuple(dict.fromkeys(FAMILY.values()))
MANIFEST = Path("fits") / "regime_baselines.json"
#: Expert states live here, relative to the manifest's directory.
STATE_DIR = "regime_baselines"
REGIME_SOURCE_FILES = (
    "src/experiments/generator_paper/regime_baselines.py",
    "src/experiments/generator_paper/cases.py",
    "src/experiments/generator_paper/baselines.py",
    "src/experiments/generator_paper/metrics.py",
    "src/experiments/noise_model/supports.py",
    "src/experiments/stochastic_fit/clips.py",
    "src/data_processing/noise_model/render.py",
    "src/data_processing/rps_gating.py",
)
MICHAELS_DATASET = "michaels-frames"
CALIBRATION_ONLY = (
    "md2 FLY125 standby/cruise core fit supports of the accepted md2 fits "
    "(input_provenance.json accepted_md2_fits -> pool.specs); no evaluation access"
)
BLEND = {
    "rule": "out = sqrt(1 - w) * standby + sqrt(w) * cruise",
    "interpolates": "expert output POWER (independent streams), not parameter vectors",
    "gate": "data_processing.noise_model.render.regime_blend_weight (smoothstep in the slowest rotor; "
    "settle_s=0, no time filter)",
    "standby_max_rps": STANDBY_MAX_RPS,
    "cruise_min_rps": CRUISE_MIN_RPS,
    "streams": "seed = rng.integers(0, 2**32 - 1); data_processing.noise_model.render.regime_seeds(seed) -> "
    "one independent numpy Generator per expert",
    "inactive_expert": "an expert whose gain is zero over the whole track is not rendered",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _atomic_json(path: Path, obj: Any) -> None:
    _atomic_bytes(path, (json.dumps(_jsonable(obj), indent=2, allow_nan=False) + "\n").encode())


def _digest(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(_jsonable(obj), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


# ── supports ───────────────────────────────────────────────────────────────


def support_windows(inputs: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """The accepted fits' exact support windows per regime, without decoding audio.

    Reads ``inputs.parent / input_provenance.json``, verifies each accepted md2
    fit file in ``inputs`` against its recorded sha256 and parses its
    ``pool.specs`` (cross-checked against ``pool.members``). Returns
    ``({regime: [window]}, provenance)``.
    """
    from experiments.noise_model.supports import parse_spec

    inputs = Path(inputs)
    provenance_path = inputs.parent / "input_provenance.json"
    raw = provenance_path.read_bytes()
    provenance = json.loads(raw)
    pin = provenance["dataset_pins"][MICHAELS_DATASET]
    records = {str(r["regime"]): r for r in provenance["accepted_md2_fits"]}
    if sorted(records) != sorted(REGIME_ORDER) or len(records) != len(
        provenance["accepted_md2_fits"]
    ):
        raise ValueError(
            f"accepted_md2_fits must hold one fit per regime {REGIME_ORDER}, got "
            f"{[r.get('regime') for r in provenance['accepted_md2_fits']]}"
        )
    windows: dict[str, list[dict[str, Any]]] = {}
    accepted: dict[str, Any] = {}
    for regime in REGIME_ORDER:
        record = records[regime]
        path = inputs / Path(record["local_path"]).name
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"accepted {regime} fit hash mismatch: {path}")
        pool = json.loads(data)["pool"]
        specs, members = list(pool["specs"]), list(pool["members"])
        if not specs or len(specs) != len(members):
            raise ValueError(
                f"{path.name}: pool.specs/members disagree ({len(specs)} vs {len(members)})"
            )
        rows = []
        for text, member in zip(specs, members, strict=True):
            spec = parse_spec(text)
            if spec.family != "flight_michaels" or spec.name != member:
                raise ValueError(
                    f"{path.name}: support {text!r} is not the flight_michaels member {member!r}"
                )
            if spec.args["rps_key"] != LABEL_KEY or spec.args["dataset"] != MICHAELS_DATASET:
                raise ValueError(
                    f"{path.name}: support {text!r} is not {MICHAELS_DATASET}/{LABEL_KEY}"
                )
            rows.append(
                {
                    "recording_id": spec.args["recording"],
                    "start_s": spec.args["start_s"],
                    "duration_s": spec.args["dur_s"],
                    "support_spec": text,
                    "support_member": member,
                    "fit_regime": regime,
                }
            )
        windows[regime] = rows
        accepted[regime] = {
            "source": record["source"],
            "sha256": digest,
            "file": path.name,
            "pool_specs": specs,
            "pool_members": members,
        }
    report = {
        "input_provenance": {
            "file": provenance_path.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "dataset_pin": f"{MICHAELS_DATASET}@{pin}",
        "label_key": LABEL_KEY,
        "accepted_fits": accepted,
        "supports": windows,
        "support_seconds": {
            r: float(sum(w["duration_s"] for w in rows)) for r, rows in windows.items()
        },
        "time_reference": "start_s relative to the recording's audio origin (audio sample 0)",
        "role_labels": (
            "fit_regime = the accepted fit the window supports; regime = cases.regime_of "
            "whole-window classification; both kept"
        ),
        "exposure": (
            "core fit supports only; our generator additionally used auxiliary FLY125 "
            "calibration and external priors, so this is not a total-data-exposure control"
        ),
    }
    return windows, report


def load_regime_calibration(inputs: Path) -> tuple[dict[str, list[FlightClip]], dict[str, Any]]:
    """Exact standby/cruise support clips (cases decode/resampling) and their provenance."""
    windows, provenance = support_windows(inputs)
    flat = [w for regime in REGIME_ORDER for w in windows[regime]]
    clips = load_support_clips(
        "md2",
        "calibration",
        [(w["recording_id"], w["start_s"], w["duration_s"]) for w in flat],
        labels=[{k: w[k] for k in ("support_spec", "support_member", "fit_regime")} for w in flat],
    )
    expected_pin = provenance["dataset_pin"].partition("@")[2]
    for clip in clips:
        if clip.provenance["dataset_version"] != expected_pin:
            raise ValueError(
                f"{clip.key}: decoded {clip.provenance['dataset_pin']}, inputs pin {expected_pin}"
            )
    out: dict[str, list[FlightClip]] = {}
    i = 0
    for regime in REGIME_ORDER:
        out[regime] = clips[i : i + len(windows[regime])]
        i += len(windows[regime])
    return out, provenance


def merge_contiguous(clips: list[FlightClip]) -> FlightClip:
    """Concatenate sample-adjacent windows of ONE recording into one clip, exactly.

    The windows must abut on both the 16 kHz and the native grid (consecutive
    slices of the same whole-recording resampling), so the result is the
    contiguous span itself; no audio is added, looped or padded. The merged
    provenance keeps every constituent window's provenance.
    """
    if len(clips) < 2:
        raise ValueError("merge_contiguous needs at least two windows")
    first, last = clips[0], clips[-1]
    for name in ("rig", "recording_id", "sr", "native_sr"):
        values = {getattr(c, name) for c in clips}
        if len(values) != 1:
            raise ValueError(f"windows disagree on {name}: {sorted(map(str, values))}")
    for name in ("split", "dataset_pin", "label_key"):
        values = {str(c.provenance.get(name)) for c in clips}
        if len(values) != 1:
            raise ValueError(f"windows disagree on {name}: {sorted(values)}")
    for a, b in zip(clips, clips[1:], strict=False):
        pa, pb = a.provenance, b.provenance
        if (
            pa["start_sample"] + pa["n_samples"] != pb["start_sample"]
            or pa["native_start_sample"] + pa["native_n_samples"] != pb["native_start_sample"]
        ):
            raise ValueError(f"{a.key} and {b.key} are not sample-adjacent; refusing to splice")
    audio = np.concatenate([c.audio for c in clips], axis=-1)
    rps = np.concatenate([c.rps for c in clips], axis=-1)
    native = np.concatenate([c.native_audio for c in clips], axis=-1)
    for arr in (audio, rps, native):
        arr.flags.writeable = False
    lo = np.min([c.provenance["rotor_min_rps"] for c in clips], axis=0)
    hi = np.max([c.provenance["rotor_max_rps"] for c in clips], axis=0)
    p0, p1 = first.provenance, last.provenance
    duration = audio.shape[-1] / first.sr
    provenance = {
        "dataset": p0["dataset"],
        "dataset_version": p0["dataset_version"],
        "dataset_pin": p0["dataset_pin"],
        "recording_id": first.recording_id,
        "label_key": p0["label_key"],
        "split": p0["split"],
        "native_sr": first.native_sr,
        "absolute_start_s": p0["absolute_start_s"],
        "absolute_end_s": p1["absolute_end_s"],
        "relative_start_s": p0["relative_start_s"],
        "relative_end_s": p1["relative_end_s"],
        "native_start_sample": p0["native_start_sample"],
        "native_n_samples": int(native.shape[-1]),
        "start_sample": p0["start_sample"],
        "n_samples": int(audio.shape[-1]),
        "rotor_min_rps": [float(v) for v in lo],
        "rotor_max_rps": [float(v) for v in hi],
        "regime_rule": p0["regime_rule"],
        "merge_rule": (
            "exact concatenation of sample-adjacent supplied support windows (consecutive "
            "slices of one whole-recording resampling); no looping, padding or additional audio"
        ),
        "merged_from": [
            {"key": c.key, "regime": c.regime, "provenance": c.provenance} for c in clips
        ],
    }
    return FlightClip(
        key=f"{first.rig}:{first.recording_id}@{first.start_s:.3f}+{duration:g}",
        rig=first.rig,
        recording_id=first.recording_id,
        start_s=first.start_s,
        sr=first.sr,
        audio=audio,
        rps=rps,
        native_sr=first.native_sr,
        native_audio=native,
        regime=regime_of(lo, hi),
        provenance=provenance,
    )


def _window_entry(clip: FlightClip) -> dict[str, Any]:
    return {
        "key": clip.key,
        "regime": clip.regime,
        "fit_regime": clip.provenance.get("fit_regime"),
        "provenance": clip.provenance,
    }


def _fit_inputs(family: str, regime: str, supports: list[FlightClip]) -> list[FlightClip]:
    """Heutschi's standby reference is the merged 12 s span; everything else as supplied."""
    if family == "heutschi2020" and regime == "standby":
        merged = merge_contiguous(supports)
        return [
            FlightClip(
                **{**merged.__dict__, "provenance": {**merged.provenance, "fit_regime": regime}}
            )
        ]
    return list(supports)


# ── expert states ──────────────────────────────────────────────────────────


_STATE_SUFFIX = {
    "heutschi2020": ".npz",
    "yang2025": ".npz",
    "yang2025_phase": ".npz",
    "lee2026_gp": ".pt",
    "doerfler2026_eone": ".pt",
}


def _save_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def save_heutschi(fit: HeutschiFit, path: Path) -> None:
    """The render state: native reference (stored float32 only when exact), speed and EQ slopes."""
    reference = np.asarray(fit.reference)
    narrow = reference.astype(np.float32)
    stored = narrow if np.array_equal(narrow.astype(reference.dtype), reference) else reference
    _save_npz(
        path,
        {
            "reference": stored,
            "native_sr": np.int64(fit.native_sr),
            "reference_rps": np.float64(fit.reference_rps),
            "slope_db_per_rpm": np.asarray(fit.slope_db_per_rpm, dtype=np.float64),
        },
    )


def load_heutschi(path: Path, report: dict[str, Any]) -> HeutschiFit:
    with np.load(path, allow_pickle=False) as z:
        return HeutschiFit(
            reference=np.asarray(z["reference"], dtype=np.float64),
            native_sr=int(z["native_sr"]),
            reference_rps=float(z["reference_rps"]),
            slope_db_per_rpm=np.array(z["slope_db_per_rpm"]),
            report=report,
        )


def save_yang(fit: YangFit, path: Path) -> None:
    _save_npz(
        path,
        {
            "reference_rps": np.float64(fit.reference_rps),
            "offset_db": np.asarray(fit.offset_db, dtype=np.float64),
            "slope_db_per_rpm": np.asarray(fit.slope_db_per_rpm, dtype=np.float64),
            "phase_carry": np.bool_(fit.phase_carry),
            "modulation_range_hz": np.float64(fit.modulation_range_hz),
            "white_ratio": np.float64(fit.white_ratio),
        },
    )


def load_yang(path: Path, report: dict[str, Any]) -> YangFit:
    with np.load(path, allow_pickle=False) as z:
        return YangFit(
            reference_rps=float(z["reference_rps"]),
            offset_db=np.array(z["offset_db"]),
            slope_db_per_rpm=np.array(z["slope_db_per_rpm"]),
            phase_carry=bool(z["phase_carry"]),
            modulation_range_hz=float(z["modulation_range_hz"]),
            white_ratio=float(z["white_ratio"]),
            report=report,
        )


def _fit_expert(
    family: str,
    clips: list[FlightClip],
    state: Path,
    *,
    seed: int,
    eone_device: str | None,
) -> tuple[dict[str, Any], str]:
    """Fit one expert, write its state atomically; ``(fit report, device)``."""
    if family == "heutschi2020":
        fit = fit_heutschi(clips, seed=seed)
        save_heutschi(fit, state)
        return fit.report, "cpu"
    if family in ("yang2025", "yang2025_phase"):
        fit = fit_yang(clips, seed=seed, phase_carry=family == "yang2025_phase")
        save_yang(fit, state)
        return fit.report, "cpu"
    if family == "lee2026_gp":
        from experiments.generator_paper.gp_baseline import fit_lee

        return fit_lee(clips, seed=seed, checkpoint=state).report, "cpu"
    if family == "doerfler2026_eone":
        from experiments.generator_paper.eone_baseline import fit_eone

        if eone_device is None:
            raise ValueError("fitting doerfler2026_eone_regimes requires an explicit eone_device")
        return fit_eone(clips, seed=seed, checkpoint=state, device=eone_device).report, eone_device
    raise ValueError(f"unknown family {family!r}")


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _resolve(root: Path, rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"manifest path {rel!r} must be relative to the manifest directory")
    return root / p


def _verified(root: Path, entry: Mapping[str, Any], what: str) -> Path:
    path = _resolve(root, entry["path"])
    if _sha256(path) != entry["sha256"]:
        raise ValueError(f"{what}: {path} does not match its recorded sha256")
    return path


def _reusable_record(
    record_path: Path, root: Path, *, family: str, regime: str, seed: int, calibration_digest: str
) -> dict[str, Any] | None:
    """A completed expert from an interrupted preparation, if it matches exactly."""
    if not record_path.exists():
        return None
    record = json.loads(record_path.read_text())
    if (
        record.get("schema") != RECORD_SCHEMA
        or record.get("family") != family
        or record.get("regime") != regime
        or record.get("seed") != seed
        or record.get("calibration_digest") != calibration_digest
    ):
        return None
    state = _resolve(root, record["state"]["path"])
    if not state.exists() or _sha256(state) != record["state"]["sha256"]:
        return None
    return record


# ── preparation ────────────────────────────────────────────────────────────


def _check_methods(methods: tuple[str, ...]) -> tuple[str, ...]:
    methods = tuple(methods)
    unknown = [m for m in methods if m not in REGIME_METHODS]
    if not methods or unknown or len(set(methods)) != len(methods):
        raise ValueError(f"methods must be distinct ids from {REGIME_METHODS}, got {methods}")
    return tuple(m for m in REGIME_METHODS if m in methods)


def _info(path: Path, manifest: Mapping[str, Any], origin: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "methods": list(manifest["methods"]),
        "calibration_only": manifest["calibration_only"],
        "origin": origin,
        "seed": manifest["seed"],
        "families": sorted(manifest["families"]),
    }


def prepare_regime_fits(
    out: Path,
    inputs: Path,
    methods: tuple[str, ...],
    *,
    seed: int,
    eone_device: str | None = None,
) -> dict[str, Any]:
    """Fit the requested matched families once and freeze them under ``out/fits``.

    An existing ``out/fits/regime_baselines.json`` is reused only if it covers
    every requested method with the same seed and supports (all artifacts
    re-verified); otherwise it is refused, never extended. Without a manifest,
    experts completed by an interrupted run (state + record checksums, same
    seed and supports) are kept and only the missing ones are fitted. Returns
    ``{path, sha256, methods, calibration_only, origin, seed, families}``.
    """
    methods = _check_methods(methods)
    seed = int(seed)
    manifest_path = (Path(out) / MANIFEST).resolve()
    root = manifest_path.parent
    _, support_report = support_windows(inputs)
    calibration_digest = _digest(support_report)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_bytes())
        if manifest.get("schema") != SCHEMA:
            raise ValueError(
                f"{manifest_path}: schema {manifest.get('schema')!r} is not {SCHEMA!r}"
            )
        missing = [m for m in methods if m not in manifest["methods"]]
        if missing:
            raise ValueError(
                f"{manifest_path} is frozen without {missing}; use a new output directory "
                "(manifests are never extended)"
            )
        if manifest["seed"] != seed or manifest["calibration_digest"] != calibration_digest:
            raise ValueError(f"{manifest_path} was fitted with another seed or support set")
        for family, entry in manifest["families"].items():
            for regime, expert in entry["experts"].items():
                _verified(root, expert["record"], f"{family}/{regime} record")
                _verified(root, expert["state"], f"{family}/{regime} state")
        return _info(manifest_path, manifest, "campaign_manifest_reused_on_restart")

    families = tuple(dict.fromkeys(FAMILY[m] for m in methods))
    record_paths = {
        (f, r): root / STATE_DIR / f"{f}__{r}.json" for f in families for r in REGIME_ORDER
    }
    done = {
        key: _reusable_record(
            path,
            root,
            family=key[0],
            regime=key[1],
            seed=seed,
            calibration_digest=calibration_digest,
        )
        for key, path in record_paths.items()
    }
    if (
        any(
            f == "doerfler2026_eone" and done[(f, r)] is None
            for f in families
            for r in REGIME_ORDER
        )
        and eone_device is None
    ):
        raise ValueError("fitting doerfler2026_eone_regimes requires an explicit eone_device")

    supports, provenance = load_regime_calibration(inputs)
    if _digest(provenance) != calibration_digest:  # pragma: no cover - same files read twice
        raise RuntimeError("support provenance changed while loading")
    n_mics = {c.audio.shape[0] for clips in supports.values() for c in clips}
    n_rotors = {c.rps.shape[0] for clips in supports.values() for c in clips}
    if len(n_mics) != 1 or len(n_rotors) != 1:
        raise ValueError(f"support windows disagree on microphones {n_mics} or rotors {n_rotors}")
    windows = [_window_entry(c) for r in REGIME_ORDER for c in supports[r]]

    family_entries: dict[str, Any] = {}
    for family in families:
        experts: dict[str, Any] = {}
        for regime in REGIME_ORDER:
            record_path = record_paths[(family, regime)]
            record = done[(family, regime)]
            if record is None:
                state = root / STATE_DIR / f"{family}__{regime}{_STATE_SUFFIX[family]}"
                clips = _fit_inputs(family, regime, supports[regime])
                print(f"regime fits: {family}/{regime} on {[c.key for c in clips]}", flush=True)
                started = time.monotonic()
                report, device = _fit_expert(
                    family, clips, state, seed=seed, eone_device=eone_device
                )
                record = {
                    "schema": RECORD_SCHEMA,
                    "family": family,
                    "regime": regime,
                    "seed": seed,
                    "calibration_digest": calibration_digest,
                    "state": {"path": _rel(state, root), "sha256": _sha256(state)},
                    "fit": report,
                    "fit_inputs": [_window_entry(c) for c in clips],
                    "fit_wall_seconds": time.monotonic() - started,
                    "fit_device": device,
                    "completed_utc": datetime.now(UTC).isoformat(),
                }
                _atomic_json(record_path, record)
            experts[regime] = {
                "record": {"path": _rel(record_path, root), "sha256": _sha256(record_path)},
                "state": record["state"],
            }
        family_entries[family] = {
            "methods": [m for m in methods if FAMILY[m] == family],
            "shared_state": [m for m in REGIME_METHODS if FAMILY[m] == family],
            "experts": experts,
        }
    manifest = {
        "schema": SCHEMA,
        "fitting_treatment": FITTING_TREATMENT,
        "methods": list(methods),
        "base_methods": {m: BASE_METHOD[m] for m in methods},
        "seed": seed,
        "calibration_only": CALIBRATION_ONLY,
        "calibration": provenance,
        "calibration_digest": calibration_digest,
        "calibration_windows": windows,
        "n_mics": n_mics.pop(),
        "n_rotors": n_rotors.pop(),
        "regimes": list(REGIME_ORDER),
        "blend": BLEND,
        "families": family_entries,
        "paths": "relative to this manifest's directory",
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "created_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(manifest_path, manifest)
    return _info(manifest_path, manifest, "fitted_in_this_campaign_before_method_workers")


# ── reload + blend ─────────────────────────────────────────────────────────


def blend_experts(
    experts: Mapping[str, Callable[..., np.ndarray]],
    rps: np.ndarray,
    *,
    sr: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """``sqrt(1 - w) * standby + sqrt(w) * cruise`` from independent expert streams.

    One draw from ``rng`` seeds :func:`regime_seeds`; each ACTIVE expert renders
    the whole track with its own Generator. Returns float32 ``(mics, T)``.
    """
    w = regime_blend_weight(rps, sr=int(sr))
    gains = {"standby": np.sqrt(np.maximum(1.0 - w, 0.0)), "cruise": np.sqrt(np.maximum(w, 0.0))}
    seeds = regime_seeds(int(rng.integers(0, 2**32 - 1)), REGIME_ORDER)
    out: np.ndarray | None = None
    for regime in REGIME_ORDER:
        gain = gains[regime]
        if not bool(np.any(gain > 0.0)):
            continue
        audio = np.asarray(
            experts[regime](rps, sr=sr, rng=np.random.default_rng(seeds[regime])), dtype=np.float64
        )
        out = audio * gain if out is None else out + audio * gain
    assert out is not None  # w in [0, 1]: at least one gain is positive
    return np.ascontiguousarray(out, dtype=np.float32)


def _require_production(report: Mapping[str, Any], label: str) -> None:
    if not report.get("production_settings"):
        raise ValueError(f"{label}: matched expert is not a production-settings fit")
    if report.get("calibration_recordings") != ["FLY125"] or report.get("calibration_splits") != [
        "calibration"
    ]:
        raise ValueError(
            f"{label}: matched expert must be calibrated on FLY125 calibration supports only"
        )


def _load_expert(
    family: str, method: str, state: Path, record: Mapping[str, Any]
) -> tuple[Callable[..., np.ndarray], int, dict[str, Any]]:
    """``(render(rps, *, sr, rng), n_mics, loaded fit report)`` for one frozen expert."""
    label = f"{method}/{record['regime']}"
    if family == "heutschi2020":
        fit = load_heutschi(state, record["fit"])
        return partial(render_heutschi, fit), int(fit.reference.shape[0]), fit.report
    if family in ("yang2025", "yang2025_phase"):
        fit = load_yang(state, record["fit"])
        if fit.phase_carry != (family == "yang2025_phase"):
            raise ValueError(
                f"{label}: stored phase_carry={fit.phase_carry} does not match the family"
            )
        return partial(render_yang, fit), int(fit.offset_db.shape[0]), fit.report
    if family == "lee2026_gp":
        from experiments.generator_paper.gp_baseline import LeeFit, render_lee

        gp = LeeFit.load(state)
        _require_production(gp.report, label)
        return (
            partial(render_lee, gp, colored=method == "lee2026_gp_colored_regimes"),
            gp.n_mics,
            gp.report,
        )
    if family == "doerfler2026_eone":
        from experiments.generator_paper.eone_baseline import EoneFit, render_eone

        eone = EoneFit.load(state)
        _require_production(eone.report, label)
        return partial(render_eone, eone), eone.n_mics, eone.report
    raise ValueError(f"unknown family {family!r}")


def regime_render(
    method: str, manifest_path: Path
) -> tuple[Callable[..., np.ndarray], dict[str, Any]]:
    """Reload one matched method's two frozen experts (no refit) and its blend.

    Every record/state checksum and the method/support compatibility are
    verified. The callable is ``render(rps, *, sr, rng) -> float32 (mics, T)``.
    """
    if method not in REGIME_METHODS:
        raise ValueError(f"{method!r} is not a matched-regime method {REGIME_METHODS}")
    manifest_path = Path(manifest_path).resolve()
    raw = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw)
    if manifest.get("schema") != SCHEMA or manifest.get("fitting_treatment") != FITTING_TREATMENT:
        raise ValueError(f"{manifest_path}: not a {SCHEMA} manifest")
    if method not in manifest["methods"]:
        raise ValueError(f"{manifest_path} holds {manifest['methods']}, not {method}")
    family = FAMILY[method]
    entry = manifest["families"][family]
    if method not in entry["methods"]:
        raise ValueError(f"{manifest_path}: family {family} does not list {method}")
    support_keys = {
        r: [w["key"] for w in manifest["calibration_windows"] if w["fit_regime"] == r]
        for r in REGIME_ORDER
    }
    root = manifest_path.parent
    experts: dict[str, Callable[..., np.ndarray]] = {}
    regimes: dict[str, Any] = {}
    for regime in REGIME_ORDER:
        expert = entry["experts"][regime]
        record_path = _verified(root, expert["record"], f"{method}/{regime} record")
        record = json.loads(record_path.read_text())
        if (
            record.get("schema") != RECORD_SCHEMA
            or record["family"] != family
            or record["regime"] != regime
            or record["seed"] != manifest["seed"]
            or record["calibration_digest"] != manifest["calibration_digest"]
            or record["state"] != expert["state"]
        ):
            raise ValueError(
                f"{record_path}: record does not belong to this manifest's {family}/{regime}"
            )
        constituents = [
            k
            for w in record["fit_inputs"]
            for k in (
                [m["key"] for m in w["provenance"]["merged_from"]]
                if "merged_from" in w["provenance"]
                else [w["key"]]
            )
        ]
        if constituents != support_keys[regime]:
            raise ValueError(
                f"{record_path}: fit inputs {constituents} are not the {regime} supports "
                f"{support_keys[regime]}"
            )
        fit_keys = [w["key"] for w in record["fit_inputs"]]
        if record["fit"].get("calibration_clip_keys") != fit_keys:
            raise ValueError(
                f"{record_path}: fit report was calibrated on "
                f"{record['fit'].get('calibration_clip_keys')}, record lists {fit_keys}"
            )
        state = _verified(root, expert["state"], f"{method}/{regime} state")
        render, n_mics, fit_report = _load_expert(family, method, state, record)
        if n_mics != manifest["n_mics"]:
            raise ValueError(
                f"{method}/{regime}: expert renders {n_mics} microphones, supports have "
                f"{manifest['n_mics']}"
            )
        experts[regime] = render
        regimes[regime] = {
            "fit": fit_report,
            "fit_inputs": record["fit_inputs"],
            "artifacts": {"record": expert["record"], "state": expert["state"]},
            "fit_seed": record["seed"],
            "fit_wall_seconds": record["fit_wall_seconds"],
            "fit_device": record["fit_device"],
            "completed_utc": record["completed_utc"],
        }
    report = {
        "method": method,
        "base_method": BASE_METHOD[method],
        "family": family,
        "fitting_treatment": FITTING_TREATMENT,
        "shared_state": entry["shared_state"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "calibration_only": manifest["calibration_only"],
        "calibration_provenance": manifest["calibration"],
        "calibration_windows": manifest["calibration_windows"],
        "seed": manifest["seed"],
        "n_mics": manifest["n_mics"],
        "n_rotors": manifest["n_rotors"],
        "blend": manifest["blend"],
        "regimes": regimes,
        "refitted_in_this_worker": False,
        **(
            {"broadband_variant": "colored" if method == "lee2026_gp_colored_regimes" else "white"}
            if family == "lee2026_gp"
            else {}
        ),
    }
    return partial(blend_experts, experts), report


__all__ = [
    "BASE_METHOD",
    "BLEND",
    "FAMILY",
    "FITTING_TREATMENT",
    "MANIFEST",
    "REGIME_METHODS",
    "REGIME_SOURCE_FILES",
    "blend_experts",
    "load_regime_calibration",
    "merge_contiguous",
    "prepare_regime_fits",
    "regime_render",
    "support_windows",
]
