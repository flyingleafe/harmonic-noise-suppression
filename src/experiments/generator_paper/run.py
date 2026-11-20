"""Restartable, artifact-provenanced comparisons of calibrated source models."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from data_processing.noise_model.render import render_noise, render_noise_regimes
from experiments.generator_paper.baselines import (
    fit_heutschi,
    fit_yang,
    render_heutschi,
    render_yang,
)
from experiments.generator_paper.cases import load_flight_clips
from experiments.generator_paper.metrics import _jsonable, evaluate_pair
from experiments.generator_paper.regime_baselines import MANIFEST as REGIME_MANIFEST
from experiments.generator_paper.regime_baselines import (
    REGIME_METHODS,
    prepare_regime_fits,
    regime_render,
)
from utils.gridrun import Unit, run_grid

METHODS = ("stochastic", "stochastic_coherent", "heutschi2020", "yang2025", "yang2025_phase")
GP_METHODS = ("lee2026_gp", "lee2026_gp_colored")
GP_CHECKPOINT = Path("fits") / "lee2026_gp.pt"
GP_FIT_REPORT = Path("fits") / "lee2026_gp_fit_report.json"
EONE_METHODS = ("doerfler2026_eone",)
EONE_CHECKPOINT = Path("fits") / "doerfler2026_eone.pt"
EONE_FIT_REPORT = Path("fits") / "doerfler2026_eone_fit_report.json"
REGIME_GP_METHODS = ("lee2026_gp_regimes", "lee2026_gp_colored_regimes")
REGIME_EONE_METHODS = ("doerfler2026_eone_regimes",)
SOURCE_ROOT = Path(__file__).resolve().parents[3]
CALIBRATION_ONLY = "md2 calibration windows (load_flight_clips('md2','calibration',duration_s=8)); no evaluation access"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_benchmark_fit(report: dict[str, Any], label: str) -> None:
    if not report["production_settings"]:
        raise ValueError(f"benchmark {label} requires production settings, not a tiny smoke fit")
    if report["calibration_recordings"] != ["FLY125"] or report["calibration_splits"] != [
        "calibration"
    ]:
        raise ValueError(f"benchmark {label} must be calibrated on FLY125 only")


def _prepare_checkpoint(
    out: Path,
    supplied: Path | None,
    *,
    label: str,
    checkpoint: Path,
    fit_report: Path,
    load: Callable[[Path], Any],
    fit: Callable[[list[Any], Path], Any],
    shared_by: tuple[str, ...],
) -> dict[str, Any]:
    if supplied is not None:
        path = supplied.resolve()
        model = load(path)
        origin = "supplied_frozen_checkpoint"
    else:
        path = (out / checkpoint).resolve()
        if path.exists():
            model = load(path)
            origin = "campaign_checkpoint_reused_on_restart"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            model = fit(load_flight_clips("md2", "calibration", duration_s=8.0), path)
            origin = "fitted_in_this_campaign_before_method_workers"
    _require_benchmark_fit(model.report, label)
    report_path = (out / fit_report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(_jsonable(model.report), indent=2) + "\n")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "origin": origin,
        "fit_seed": int(model.report["seed"]),
        "fit_report": str(report_path),
        "calibration_only": CALIBRATION_ONLY,
        "shared_by": list(shared_by),
    }


def prepare_gp_checkpoint(out: Path, supplied: Path | None, *, seed: int) -> dict[str, Any]:
    """Fit the shared calibration-only GP checkpoint once, or adopt a frozen one.

    Fitting reads only the pinned md2 calibration windows. A checkpoint already in
    ``out/fits`` (restart) is reused, never refitted.
    """
    from experiments.generator_paper.gp_baseline import LeeFit, fit_lee

    return _prepare_checkpoint(
        out,
        supplied,
        label="GP",
        checkpoint=GP_CHECKPOINT,
        fit_report=GP_FIT_REPORT,
        load=LeeFit.load,
        fit=lambda clips, path: fit_lee(clips, seed=seed, checkpoint=path),
        shared_by=GP_METHODS,
    )


def prepare_eone_checkpoint(
    out: Path, supplied: Path | None, *, seed: int, device: str | None
) -> dict[str, Any]:
    """Fit the calibration-only EONE-RPS curve bank once on ``device``, or adopt a frozen one.

    Same rules as :func:`prepare_gp_checkpoint`: only the pinned md2 calibration
    windows reach the fit, a restart reuses ``out/fits``, scoring workers reload.
    """
    from experiments.generator_paper.eone_baseline import EONE_METHOD_ID, EoneFit, fit_eone

    if EONE_METHODS != (EONE_METHOD_ID,):
        raise ValueError(f"EONE method id mismatch: {EONE_METHODS} vs {EONE_METHOD_ID!r}")

    def fit(clips: list[Any], path: Path) -> Any:
        if device is None:
            raise ValueError("fitting doerfler2026_eone requires an explicit --eone-device")
        return fit_eone(clips, seed=seed, checkpoint=path, device=device)

    info = _prepare_checkpoint(
        out,
        supplied,
        label="EONE",
        checkpoint=EONE_CHECKPOINT,
        fit_report=EONE_FIT_REPORT,
        load=EoneFit.load,
        fit=fit,
        shared_by=EONE_METHODS,
    )
    info["requested_fit_device"] = (
        device if info["origin"] == "fitted_in_this_campaign_before_method_workers" else None
    )
    return info


def gp_render(method: str, checkpoint: Path) -> tuple[Callable[..., np.ndarray], dict[str, Any]]:
    from experiments.generator_paper.gp_baseline import LeeFit, render_lee

    fit = LeeFit.load(checkpoint)
    _require_benchmark_fit(fit.report, "GP")
    report = {
        **fit.report,
        "checkpoint_path": str(checkpoint),
        "checkpoint_file_sha256": _sha256(checkpoint),
        "broadband_variant": "colored" if method == "lee2026_gp_colored" else "white",
        "refitted_in_this_worker": False,
    }
    return partial(render_lee, fit, colored=method == "lee2026_gp_colored"), report


def eone_render(checkpoint: Path) -> tuple[Callable[..., np.ndarray], dict[str, Any]]:
    """Reload the frozen EONE-RPS checkpoint for rendering only (no refit)."""
    from experiments.generator_paper.eone_baseline import EoneFit, render_eone

    fit = EoneFit.load(checkpoint)
    _require_benchmark_fit(fit.report, "EONE")
    report = {
        **fit.report,
        "checkpoint_path": str(checkpoint),
        "checkpoint_file_sha256": _sha256(checkpoint),
        "refitted_in_this_worker": False,
    }
    return partial(render_eone, fit), report


def regime_manifest_info(manifest: Path, methods: tuple[str, ...]) -> dict[str, Any]:
    """Hash a frozen matched-regime manifest and check it covers ``methods`` (no fitting)."""
    path = manifest.resolve()
    data = json.loads(path.read_bytes())
    fitted = data.get("methods") if isinstance(data, dict) else None
    if not isinstance(fitted, list):
        raise ValueError(f"{path}: regime manifest lacks a 'methods' list")
    missing = [m for m in methods if m not in fitted]
    if missing:
        raise ValueError(f"{path}: regime manifest does not contain fitted methods {missing}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "methods": list(fitted),
        "calibration_only": data.get("calibration_only"),
    }


def regime_source_files(methods: tuple[str, ...]) -> list[str]:
    """Runtime sources of the matched-regime branch (core module, loaders, family modules)."""
    from experiments.generator_paper import regime_baselines

    names = [
        "src/experiments/generator_paper/regime_baselines.py",
        "src/experiments/generator_paper/cases.py",
        "src/experiments/generator_paper/baselines.py",
    ]
    names += list(regime_baselines.REGIME_SOURCE_FILES)
    if any(m in REGIME_GP_METHODS for m in methods):
        from experiments.generator_paper.gp_baseline import GP_SOURCE_FILES

        names += list(GP_SOURCE_FILES)
    if any(m in REGIME_EONE_METHODS for m in methods):
        from experiments.generator_paper.eone_baseline import EONE_SOURCE_FILES

        names += list(EONE_SOURCE_FILES)
    return list(dict.fromkeys(names))


def _input_models(root: Path, rig: str, coherent: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    provenance = json.loads((root.parent / "input_provenance.json").read_text())
    records = (
        provenance["accepted_md2_fits"] if rig == "md2" else [provenance["accepted_dregon_fit"]]
    )
    fits, sources = {}, []
    for record in records:
        path = root / Path(record["local_path"]).name
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"calibration artifact hash mismatch: {path}")
        fit = json.loads(raw)
        if coherent:
            fit = copy.deepcopy(fit)
            fit["params"]["shaft"]["eps"] = 0.0
            for name in ("sigma2_amp", "sigma2_phase"):
                values = fit["params"]["pedestal"][name]
                fit["params"]["pedestal"][name] = [0.0] * len(values)
        fits[record.get("regime", "single")] = fit
        sources.append({"source": record["source"], "sha256": digest, "path": str(path)})
    report = {
        "sources": sources,
        "refitted_in_this_evaluation": False,
        "ablation": "shaft, log-AM and phase variances set to zero; no recalibration"
        if coherent
        else None,
        "calibration_exposure": (
            "Accepted FLY125 fits; auxiliary response/amplitude measurements also used FLY125. "
            "External DREGON-informed prior choices are part of the fixed method, not few-shot target calibration."
            if rig == "md2"
            else "Accepted DREGON bench fit plus flight-fitted wind. Baselines use room2 flights: "
            "this secondary diagnostic is not a matched-calibration architecture comparison."
        ),
    }
    return (fits if rig == "md2" else fits["single"]), report


def _case_means(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["key"], []).append(row)
    out = []
    for key, case_rows in grouped.items():
        keys = sorted(set().union(*(r["summary"] for r in case_rows)))
        metrics = {}
        for name in keys:
            values = [r["summary"][name] for r in case_rows if name in r["summary"]]
            metrics[name] = {
                "mean": float(np.mean(values)),
                "draw_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "n_draws_defined": len(values),
            }
        out.append(
            {
                "key": key,
                "regime": case_rows[0]["regime"],
                "n_draws": len(case_rows),
                "metrics": metrics,
            }
        )
    return out


def _aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {}
    for regime in ("all", "standby", "transition", "cruise"):
        selected = [r for r in cases if regime == "all" or r["regime"] == regime]
        metrics = {}
        for name in sorted(set().union(*(r["metrics"] for r in selected))):
            values = [r["metrics"][name]["mean"] for r in selected if name in r["metrics"]]
            metrics[name] = {
                "mean": float(np.mean(values)),
                "sample_sd_across_windows": float(np.std(values, ddof=1))
                if len(values) > 1
                else None,
                "n_windows_defined": len(values),
            }
        groups[regime] = {"n_windows": len(selected), "metrics": metrics}
    return groups


def _render_stochastic(
    fits: dict[str, Any],
    rps: np.ndarray,
    *,
    sr: int,
    rng: np.random.Generator,
    rig: str,
    n_mics: int,
) -> np.ndarray:
    seed = int(rng.integers(0, 2**32 - 1))
    if rig == "md2":
        audio = render_noise_regimes(fits, rps, sr=sr, n_mics=n_mics, seed=seed)
    else:
        audio = render_noise(fits, rps, sr=sr, n_mics=n_mics, seed=seed)
    return np.asarray(audio, dtype=np.float32)


def worker(unit: Unit) -> dict[str, Any]:
    p = dict(unit.params)
    method, rig = p["method"], p["rig"]
    started = time.monotonic()
    regime = method in REGIME_METHODS
    # Matched-regime workers reload frozen standby/cruise states; the pooled 21-window
    # calibration set is neither loaded nor reported for them.
    calibration = [] if regime else load_flight_clips(rig, "calibration", duration_s=8.0)
    evaluation = load_flight_clips(rig, "evaluation", duration_s=4.0, limit=p["evaluation_limit"])
    calibration_description = (
        "frozen paired fits" if regime else f"{len(calibration)} calibration windows"
    )
    print(
        f"{rig}/{method}: {calibration_description}, {len(evaluation)} evaluation windows",
        flush=True,
    )
    fit_start = time.monotonic()
    render: Callable[..., np.ndarray]
    if regime:
        render, fit_report = regime_render(method, Path(p["regime_fits"]))
        if fit_report.get("manifest_sha256") != p["regime_fits_sha256"]:
            raise ValueError(f"{method}: regime manifest changed after protocol was frozen")
    elif method.startswith("stochastic"):
        stochastic_fit, fit_report = _input_models(
            Path(p["inputs"]), rig, method == "stochastic_coherent"
        )
        render = partial(
            _render_stochastic, stochastic_fit, rig=rig, n_mics=calibration[0].audio.shape[0]
        )
    elif method in GP_METHODS:
        render, fit_report = gp_render(method, Path(p["gp_checkpoint"]))
    elif method in EONE_METHODS:
        render, fit_report = eone_render(Path(p["eone_checkpoint"]))
    elif method == "heutschi2020":
        heutschi_fit = fit_heutschi(calibration, seed=p["seed"])
        fit_report = heutschi_fit.report
        render = partial(render_heutschi, heutschi_fit)
    else:
        yang_fit = fit_yang(calibration, seed=p["seed"], phase_carry=method == "yang2025_phase")
        fit_report = yang_fit.report
        render = partial(render_yang, yang_fit)
    calibration_seconds = time.monotonic() - fit_start
    if regime:
        calibration_manifest = fit_report["calibration_windows"]
    else:
        calibration_manifest = [
            {"key": c.key, "regime": c.regime, "provenance": c.provenance} for c in calibration
        ]
    # One representative per regime; all measured windows still enter the summaries.
    example_indices = []
    for regime in ("standby", "transition", "cruise"):
        first = next((i for i, c in enumerate(evaluation) if c.regime == regime), None)
        if first is not None:
            example_indices.append(first)
    examples = Path(p["out"]) / "examples"
    examples.mkdir(parents=True, exist_ok=True)
    rows, metadata, render_times, metric_times = [], None, [], []
    for i, clip in enumerate(evaluation):
        for draw in range(p["draws"]):
            rng = np.random.default_rng(np.random.SeedSequence([p["seed"], i, draw]))
            render_start = time.monotonic()
            generated = render(clip.rps, sr=clip.sr, rng=rng)
            render_times.append(time.monotonic() - render_start)
            generated = np.asarray(generated, dtype=np.float32)
            if generated.shape != clip.audio.shape or not np.isfinite(generated).all():
                raise ValueError(f"{method}/{clip.key}: invalid generated audio {generated.shape}")
            metric_start = time.monotonic()
            result = evaluate_pair(clip.audio, generated, clip.rps, sr=clip.sr)
            metric_times.append(time.monotonic() - metric_start)
            if "lsd_db" not in result["summary"] or "third_octave_mae_db" not in result["summary"]:
                raise ValueError(f"{method}/{clip.key}: primary metrics are undefined")
            metadata = result["metadata"]
            rows.append(
                {
                    "key": clip.key,
                    "recording_id": clip.recording_id,
                    "regime": clip.regime,
                    "start_s": clip.start_s,
                    "draw": draw,
                    "summary": result["summary"],
                    "undefined": result["details"]["undefined"],
                    "coverage": result["details"]["coverage"],
                    "details": result["details"] if draw == 0 and i in example_indices else None,
                }
            )
            if draw == 0 and i in example_indices:
                np.savez_compressed(
                    examples / f"{method}__{i:03d}.npz",
                    generated=generated,
                    sr=clip.sr,
                    key=clip.key,
                )
                if method == p["reference_owner"]:
                    np.savez_compressed(
                        examples / f"reference__{i:03d}.npz",
                        real=clip.audio,
                        rps=clip.rps,
                        sr=clip.sr,
                        key=clip.key,
                        regime=clip.regime,
                    )
            print(
                f"{method} {i + 1}/{len(evaluation)} draw {draw + 1}/{p['draws']}: LSD {result['summary']['lsd_db']:.3f} dB",
                flush=True,
            )
    cases = _case_means(rows)
    return _jsonable(
        {
            "schema": "generator-paper-evaluation/1",
            "method": method,
            "rig": rig,
            "seed": p["seed"],
            "n_draws": p["draws"],
            "protocol_fingerprint": p["fingerprint"],
            "snapshot_sha256": os.environ.get("NG_SNAPSHOT_SHA256"),
            "python": platform.python_version(),
            "calibration": fit_report,
            "calibration_wall_seconds": calibration_seconds,
            "calibration_windows": calibration_manifest,
            "evaluation_windows": [{"key": c.key, "provenance": c.provenance} for c in evaluation],
            "metric_metadata": metadata,
            "pairs": rows,
            "case_means": cases,
            "by_regime": _aggregate(cases),
            "timing": {
                "render_seconds": render_times,
                "metric_seconds": metric_times,
                "total_wall_seconds": time.monotonic() - started,
            },
            "interpretation": "Window SD is not a population uncertainty over airframes or independent training seeds.",
        }
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "generator-paper-summary/1",
        "n_methods": len(rows),
        "methods": {
            row["method"]: {
                "rig": row["rig"],
                "by_regime": row["by_regime"],
                "timing": row["timing"],
            }
            for row in rows
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=Path("results/noise_generator_paper/inputs"))
    parser.add_argument("--rig", choices=("md2", "dregon"), default="md2")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS + GP_METHODS + EONE_METHODS + REGIME_METHODS,
        default=list(METHODS),
    )
    parser.add_argument(
        "--regime-fits",
        type=Path,
        help="frozen standby/cruise manifest (fits/regime_baselines.json); otherwise the requested "
        "*_regimes states are fitted once into OUT/fits before method workers",
    )
    parser.add_argument(
        "--gp-checkpoint",
        type=Path,
        help="frozen lee2026_gp checkpoint; otherwise fitted once into OUT/fits/lee2026_gp.pt",
    )
    parser.add_argument(
        "--eone-checkpoint",
        type=Path,
        help="frozen doerfler2026_eone checkpoint; otherwise fitted once into OUT/fits/doerfler2026_eone.pt",
    )
    parser.add_argument(
        "--eone-device",
        help="torch device for fitting doerfler2026_eone[_regimes] (e.g. cuda, cpu); required when fitting",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--evaluation-limit", type=int)
    args = parser.parse_args()
    if args.draws < 1 or args.jobs < 1 or len(set(args.methods)) != len(args.methods):
        parser.error("draws/jobs must be positive and methods must be unique")
    use_gp = any(m in GP_METHODS for m in args.methods)
    if use_gp and args.rig != "md2":
        parser.error("lee2026_gp methods are calibrated on md2 only")
    if args.gp_checkpoint is not None and not use_gp:
        parser.error("--gp-checkpoint requires a lee2026_gp method")
    use_eone = any(m in EONE_METHODS for m in args.methods)
    if use_eone and args.rig != "md2":
        parser.error("doerfler2026_eone is calibrated on md2 only")
    regime_methods = tuple(m for m in args.methods if m in REGIME_METHODS)
    if regime_methods and args.rig != "md2":
        parser.error("*_regimes methods are calibrated on md2 standby/cruise supports only")
    if args.regime_fits is not None and not regime_methods:
        parser.error("--regime-fits requires a *_regimes method")
    fit_regime_eone = (
        any(m in REGIME_EONE_METHODS for m in regime_methods)
        and args.regime_fits is None
        and not (args.out / REGIME_MANIFEST).exists()
    )
    fit_pooled_eone = (
        use_eone and args.eone_checkpoint is None and not (args.out / EONE_CHECKPOINT).exists()
    )
    if args.eone_device is not None and not (fit_pooled_eone or fit_regime_eone):
        parser.error(
            "--eone-device applies only when an EONE fit is performed; frozen checkpoints/manifests "
            "are never refitted"
        )
    if args.eone_checkpoint is not None and not use_eone:
        parser.error("--eone-checkpoint requires the doerfler2026_eone method")
    if (fit_pooled_eone or fit_regime_eone) and args.eone_device is None:
        parser.error(
            "fitting EONE requires an explicit --eone-device (or a frozen --eone-checkpoint/--regime-fits)"
        )
    sources = [
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
    ]
    if use_gp:
        from experiments.generator_paper.gp_baseline import GP_SOURCE_FILES

        sources += [name for name in GP_SOURCE_FILES if name not in sources]
    if use_eone:
        from experiments.generator_paper.eone_baseline import EONE_SOURCE_FILES

        sources += [name for name in EONE_SOURCE_FILES if name not in sources]
    if regime_methods:
        sources += [name for name in regime_source_files(regime_methods) if name not in sources]
    protocol = {
        "rig": args.rig,
        "methods": args.methods,
        "seed": args.seed,
        "draws": args.draws,
        "calibration_duration_s": 8.0,
        "evaluation_duration_s": 4.0,
        "minimum_all_rotor_rps": 20.0,
        "evaluation_limit": args.evaluation_limit,
        "inputs": str(args.inputs.resolve()),
        "input_provenance_sha256": hashlib.sha256(
            (args.inputs.parent / "input_provenance.json").read_bytes()
        ).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((SOURCE_ROOT / name).read_bytes()).hexdigest() for name in sources
        },
        "snapshot_sha256": os.environ.get("NG_SNAPSHOT_SHA256"),
        "primary_regime": "cruise",
        "artifact_selection": "calibration only; no evaluation-based parameter selection",
        "aggregation": "mics within pair; draws within window; unweighted windows within regime",
    }
    gp_checkpoint = eone_checkpoint = None
    if use_gp:
        args.out.mkdir(parents=True, exist_ok=True)
        gp_checkpoint = prepare_gp_checkpoint(args.out, args.gp_checkpoint, seed=args.seed)
        protocol["gp_checkpoint"] = {
            k: gp_checkpoint[k] for k in ("path", "sha256", "fit_seed", "calibration_only")
        }
        (args.out / "gp_checkpoint_provenance.json").write_text(
            json.dumps(gp_checkpoint, indent=2) + "\n"
        )
    if use_eone:
        args.out.mkdir(parents=True, exist_ok=True)
        eone_checkpoint = prepare_eone_checkpoint(
            args.out, args.eone_checkpoint, seed=args.seed, device=args.eone_device
        )
        protocol["eone_checkpoint"] = {
            k: eone_checkpoint[k] for k in ("path", "sha256", "fit_seed", "calibration_only")
        }
        (args.out / "eone_checkpoint_provenance.json").write_text(
            json.dumps(eone_checkpoint, indent=2) + "\n"
        )
    regime_fits = None
    if regime_methods:
        args.out.mkdir(parents=True, exist_ok=True)
        if args.regime_fits is not None:
            regime_fits = regime_manifest_info(args.regime_fits, regime_methods)
            regime_fits["origin"] = "supplied_frozen_manifest"
        else:
            prepared = prepare_regime_fits(
                args.out,
                args.inputs,
                regime_methods,
                seed=args.seed,
                eone_device=args.eone_device if fit_regime_eone else None,
            )
            regime_fits = {
                **regime_manifest_info(Path(prepared["path"]), regime_methods),
                "origin": "fitted_or_reused_in_this_campaign_before_method_workers",
            }
            if prepared["sha256"] != regime_fits["sha256"]:
                raise ValueError("regime manifest hash differs from prepare_regime_fits report")
        protocol["regime_fits"] = {
            k: regime_fits[k] for k in ("path", "sha256", "methods", "calibration_only")
        }
        protocol["regime_fits"]["requested_methods"] = list(regime_methods)
        (args.out / "regime_fits_provenance.json").write_text(
            json.dumps(regime_fits, indent=2) + "\n"
        )
    if use_gp or any(m in REGIME_GP_METHODS for m in regime_methods):
        # The full fit uses the allocated CPU threads; each spawned scoring unit
        # gets one thread so the two broadband variants do not oversubscribe.
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMBA_NUM_THREADS",
        ):
            os.environ[name] = "1"
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "protocol.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != protocol:
        raise ValueError(
            "output directory contains a different protocol; use a new campaign directory"
        )
    manifest_path.write_text(json.dumps(protocol, indent=2) + "\n")
    units = [
        Unit(
            uid=name,
            params={
                "method": name,
                "rig": args.rig,
                "seed": args.seed,
                "draws": args.draws,
                "inputs": str(args.inputs.resolve()),
                "out": str(args.out.resolve()),
                "evaluation_limit": args.evaluation_limit,
                "fingerprint": fingerprint,
                "reference_owner": args.methods[0],
                **({"gp_checkpoint": gp_checkpoint["path"]} if gp_checkpoint else {}),
                **({"eone_checkpoint": eone_checkpoint["path"]} if eone_checkpoint else {}),
                **(
                    {
                        "regime_fits": regime_fits["path"],
                        "regime_fits_sha256": regime_fits["sha256"],
                    }
                    if regime_fits
                    else {}
                ),
            },
        )
        for name in args.methods
    ]
    result = run_grid(
        units, worker, args.out, jobs=args.jobs, summarize=summarize, mp_context="spawn"
    )
    raise SystemExit(result.exit_code)


if __name__ == "__main__":
    main()
