"""Score genuine native-framework outputs and extend, never replace, old evidence."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.generator_paper.neural_run import (
    METHODS,
    ROOT,
    calibration_report,
    load_json,
    plain,
    sha256,
    write_json,
)
from utils.gridrun import Unit, run_grid


def _generated(root: Path, request: dict, rendered: dict, manifest_sha: str) -> np.ndarray:
    entry = rendered[request["id"]]
    path = root / entry["file"]
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"Stored native waveform changed: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        if (
            str(arrays["manifest_sha256"]) != manifest_sha
            or str(arrays["request_id"]) != request["id"]
        ):
            raise ValueError(f"Stored native waveform has wrong model or trajectory: {path}")
        return arrays["generated"]


def _score_window(unit: Unit) -> dict:
    from experiments.generator_paper.metrics import evaluate_pair

    p = unit.params
    dataset, rendered_root, out = Path(p["dataset"]), Path(p["rendered"]), Path(p["out"])
    case = p["case"]
    reference = dataset / case["file"]
    if sha256(reference) != case["sha256"]:
        raise ValueError(f"Evaluation reference changed: {reference}")
    with np.load(reference, allow_pickle=False) as arrays:
        real, rps = arrays["real"], arrays["rps"]
    rows, timings, metadata = [], [], None
    for request in p["requests"]:
        generated = _generated(rendered_root, request, p["render_records"], p["manifest_sha256"])
        if generated.shape != real.shape or not np.isfinite(generated).all():
            raise ValueError(f"Invalid native output shape/values for {request['id']}")
        started = time.monotonic()
        result = evaluate_pair(real, generated, rps, sr=int(case["sr"]))
        timings.append(time.monotonic() - started)
        if "lsd_db" not in result["summary"] or "third_octave_mae_db" not in result["summary"]:
            raise ValueError(f"Primary metric undefined for {request['id']}")
        metadata = result["metadata"]
        example = case["index"] in p["example_indices"]
        rows.append(
            {
                "key": case["key"],
                "recording_id": case["recording_id"],
                "regime": case["regime"],
                "start_s": case["start_s"],
                "draw": request["draw"],
                "summary": result["summary"],
                "undefined": result["details"]["undefined"],
                "coverage": result["details"]["coverage"],
                "details": result["details"] if request["draw"] == 0 and example else None,
            }
        )
        if request["draw"] == 0 and example:
            folder = out / "examples"
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                folder / f"{p['method']}__{case['index']:03d}.npz",
                generated=generated,
                sr=case["sr"],
                key=case["key"],
            )
            np.savez_compressed(
                folder / f"reference__{case['index']:03d}.npz",
                real=real,
                rps=rps,
                sr=case["sr"],
                key=case["key"],
                regime=case["regime"],
            )
        print(
            f"{p['method']} {case['index'] + 1} draw {request['draw'] + 1}: "
            f"LSD {result['summary']['lsd_db']:.3f} dB",
            flush=True,
        )
    return plain(
        {
            "index": case["index"],
            "pairs": rows,
            "metric_metadata": metadata,
            "metric_seconds": timings,
        }
    )


def _control_probe(
    method: str, dataset: Path, manifest: Path, rendered_root: Path, rendered: dict, out: Path
) -> dict:
    from experiments.generator_paper import control as c

    inputs = load_json(dataset / "evaluation.json")
    requests = load_json(dataset / "requests.json")
    conditions, draws = inputs["conditions"], int(inputs["draws"])
    selected = {
        (row["condition"], row["draw"]): row
        for row in requests["requests"]
        if row["kind"] == "control"
    }
    records, examples = [], []
    spectra = np.empty((1, len(conditions), draws, c.WELCH_NPERSEG // 2 + 1), dtype=np.float64)
    frequency = None
    manifest_sha = sha256(manifest)
    out.mkdir(parents=True, exist_ok=True)
    (out / "examples").mkdir(exist_ok=True)
    for draw in range(draws):
        for index, condition in enumerate(conditions):
            request = selected[(condition["name"], draw)]
            audio = _generated(rendered_root, request, rendered, manifest_sha)
            if audio.shape != (inputs["n_mics"], int(c.DURATION_S * c.SR)):
                raise ValueError("Control output disagrees with the frozen benchmark dimensions")
            frequency, psd = c.mean_psd(audio.astype(np.float64), c.SR)
            spectra[0, index, draw] = psd
            peaks: list[dict[str, Any]] = c.detect_peaks(frequency, psd)
            rotors = c.assign_peaks(peaks, condition["rotor_rps"])
            assigned = {row["peak_index"]: row["rotor"] for row in rotors if row["matched"]}
            for i, peak in enumerate(peaks):
                peak["assigned_rotor"] = assigned.get(i)
            records.append(
                {
                    "condition": condition["name"],
                    "draw": draw,
                    "detected_peaks": peaks,
                    "n_unassigned_peaks": len(peaks) - len(assigned),
                    "rotors": rotors,
                    "audio_rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
                }
            )
            if draw == 0:
                path = out / "examples" / f"{method}__{condition['name']}__draw0.npz"
                np.savez_compressed(
                    path,
                    generated=audio,
                    sr=c.SR,
                    rotor_rps=np.asarray(condition["rotor_rps"]),
                    method=method,
                    condition=condition["name"],
                    seed=inputs["seed"],
                    draw=0,
                    seed_entropy=np.asarray(request["seed_entropy"]),
                )
                examples.append(str(path.relative_to(out)))
    if frequency is None:
        raise ValueError("Control request set is empty")
    np.savez_compressed(
        out / "control_measurements.npz",
        frequency_hz=frequency,
        mean_psd=spectra,
        methods=np.asarray([method]),
        conditions=np.asarray([row["name"] for row in conditions]),
        rotor_rps=np.asarray([row["rotor_rps"] for row in conditions]),
        draws=np.arange(draws),
    )
    fit = calibration_report(manifest)
    result = {
        "schema": c.SCHEMA,
        "interpretation": "Engineering controllability read back from genuinely generated audio; not perceptual "
        "fidelity, flight-controller simulation or a new airframe sample.",
        "rig": "md2",
        "sr": c.SR,
        "duration_s": c.DURATION_S,
        "n_mics": inputs["n_mics"],
        "seed": inputs["seed"],
        "n_draws": draws,
        "methods_probed": [method],
        "conditions": conditions,
        "random_streams": {
            "seed_sequence": "SeedSequence([seed, 5000, draw]), recreated for every setpoint condition",
            "note": "Same independent expert seeds per draw across setpoints; source-native RNGs documented by fits.",
        },
        "measurement": {
            "psd": "Welch, Hann window, scaling=density, mean over microphones",
            "nperseg": c.WELCH_NPERSEG,
            "hop": c.WELCH_HOP,
            "band_hz": list(c.BAND_HZ),
            "frequency_resolution_hz": c.SR / c.WELCH_NPERSEG,
            "peak_detection": "scipy.signal.find_peaks on 10*log10 PSD over the full spectrum, kept if refined frequency lies in band_hz",
            "prominence_db": c.PROMINENCE_DB,
            "min_peak_separation_hz": c.MIN_PEAK_SEPARATION_HZ,
            "prominence_window_hz": 2 * c.PROMINENCE_HALF_WINDOW_HZ,
            "refinement": "three-point quadratic interpolation in log power",
            "assignment": "scipy.optimize.linear_sum_assignment of peaks to commanded 2*rps; maximize in-tolerance matches, then minimize |error|",
            "match_tolerance_hz": c.MATCH_TOLERANCE_HZ,
            "blades": c.BLADES,
            "measured_rps": "matched peak frequency / blades",
            "missing": "rotors without an in-tolerance peak stay unmatched; no fallback values",
            "orders_in_band": "second shaft order only for all commands (first and third orders lie outside)",
            "slope": "least-squares slope of measured rotor rps vs rotor-0 command over all three setpoints; defined only for draws matched in every setpoint; target 1 for rotor 0, 0 otherwise",
        },
        "methods": {
            method: {
                "label": METHODS[method][1],
                "fit_source": fit,
                "summary": c.summarize_method(records, draws),
                "records": records,
            }
        },
        "outputs": {"examples": examples, "measurements": "control_measurements.npz"},
        "provenance": {
            "regime_fits": str(manifest.resolve()),
            "regime_fits_sha256": manifest_sha,
            "regime_fits_methods": [method],
            "refitted_in_this_probe": False,
            "source_sha256": fit["source_sha256"],
            "snapshot_sha256": os.environ.get("NG_SNAPSHOT_SHA256"),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "timing": {
            "render_seconds": {
                method: [rendered[row["id"]]["seconds"] for row in selected.values()]
            }
        },
    }
    write_json(out / "control_probe.json", result)
    return result


def score_outputs(
    method: str, dataset: Path, manifest: Path, rendered: Path, out: Path, *, jobs: int = 4
) -> dict:
    from experiments.generator_paper.run import _aggregate, _case_means

    dataset, manifest, rendered, out = (
        path.resolve() for path in (dataset, manifest, rendered, out)
    )
    fit = calibration_report(manifest)
    if not fit["production_settings"]:
        raise ValueError("Smoke fits cannot enter the scientific comparison")
    evaluation = load_json(dataset / "evaluation.json")
    requests = load_json(dataset / "requests.json")
    render_info = load_json(rendered / "render_manifest.json")
    if (
        render_info["method"] != method
        or render_info["manifest_sha256"] != sha256(manifest)
        or render_info["requests_sha256"] != sha256(dataset / "requests.json")
    ):
        raise ValueError("Rendered outputs do not belong to this frozen model/input export")
    records = {row["id"]: row for row in render_info["records"]}
    expected = {row["id"] for row in requests["requests"]}
    if set(records) != expected or len(records) != len(render_info["records"]):
        raise ValueError("Native render set is missing or duplicates fidelity/control requests")
    protocol = {
        "method": method,
        "rig": "md2",
        "seed": evaluation["seed"],
        "draws": evaluation["draws"],
        "calibration_sha256": evaluation["calibration_sha256"],
        "manifest_sha256": sha256(manifest),
        "requests_sha256": sha256(dataset / "requests.json"),
        "render_sha256": sha256(rendered / "render_manifest.json"),
        "artifact_selection": "fixed calibration-only fitting budget; no evaluation selection",
        "source_sha256": fit["source_sha256"],
        "snapshot_sha256": os.environ.get("NG_SNAPSHOT_SHA256"),
    }
    fingerprint = hashlib_json(protocol)
    write_json(out / "eval" / "protocol.json", protocol)
    units = []
    for case in evaluation["cases"]:
        these = sorted(
            (
                row
                for row in requests["requests"]
                if row["kind"] == "evaluation" and row["index"] == case["index"]
            ),
            key=lambda row: row["draw"],
        )
        units.append(
            Unit(
                uid=f"window__{case['index']:03d}",
                params={
                    "dataset": str(dataset),
                    "rendered": str(rendered),
                    "out": str(out / "eval"),
                    "method": method,
                    "case": case,
                    "requests": these,
                    "render_records": records,
                    "manifest_sha256": sha256(manifest),
                    "example_indices": evaluation["example_indices"],
                },
            )
        )
    completed = run_grid(
        units, _score_window, out / "eval" / "units", jobs=jobs, mp_context="spawn"
    )
    if completed.exit_code:
        raise RuntimeError("A native waveform scoring unit failed; inspect units/raw/*.err")
    windows = [load_json(path) for path in sorted((out / "eval" / "units" / "raw").glob("*.json"))]
    if len(windows) != len(evaluation["cases"]):
        raise ValueError("Scored window set differs from the fixed evaluation export")
    pairs = [pair for window in windows for pair in window["pairs"]]
    if len(pairs) != len(evaluation["cases"]) * evaluation["draws"]:
        raise ValueError("Scored pair count differs from the complete draw/window grid")
    cases = _case_means(pairs)
    raw = {
        "schema": "generator-paper-evaluation/1",
        "method": method,
        "rig": "md2",
        "seed": evaluation["seed"],
        "n_draws": evaluation["draws"],
        "protocol_fingerprint": fingerprint,
        "snapshot_sha256": os.environ.get("NG_SNAPSHOT_SHA256"),
        "python": platform.python_version(),
        "calibration": fit,
        "calibration_wall_seconds": sum(row["fit_wall_seconds"] for row in fit["regimes"].values()),
        "calibration_windows": fit["calibration_windows"],
        "evaluation_windows": [
            {"key": row["key"], "provenance": row["provenance"]} for row in evaluation["cases"]
        ],
        "metric_metadata": windows[0]["metric_metadata"],
        "pairs": pairs,
        "case_means": cases,
        "by_regime": _aggregate(cases),
        "timing": {
            "render_seconds": [
                row["seconds"] for row in render_info["records"] if row["kind"] == "evaluation"
            ],
            "metric_seconds": [value for window in windows for value in window["metric_seconds"]],
        },
        "interpretation": "Means/SD aggregate microphones and stochastic draws within recording windows; "
        "not population-level uncertainty, blind generalization or independent training seeds.",
    }
    path = out / "eval" / "raw" / f"{method}.json"
    write_json(path, raw)
    probe = _control_probe(method, dataset, manifest, rendered, records, out / "control")
    result = {
        "method": method,
        "raw": str(path.relative_to(out)),
        "raw_sha256": sha256(path),
        "control": "control",
        "control_sha256": sha256(out / "control" / "control_probe.json"),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "fidelity_pairs": len(pairs),
        "control_commands": len(probe["methods"][method]["records"]) * 4,
    }
    write_json(out / "extension_result.json", result)
    return result


def hashlib_json(value: Any) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(plain(value), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def statistics(raw: dict) -> dict:
    from experiments.generator_paper.run import _case_means

    cases = _case_means(raw["pairs"])
    result = {}
    for regime in ("all", "cruise", "standby", "transition"):
        selected = [case for case in cases if regime == "all" or case["regime"] == regime]
        metrics = {}
        for name in sorted(set().union(*(case["metrics"] for case in selected))):
            values = [case["metrics"][name]["mean"] for case in selected if name in case["metrics"]]
            metrics[name] = {
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "n": len(values),
                "pairs": sum(
                    case["metrics"][name]["n_draws_defined"]
                    for case in selected
                    if name in case["metrics"]
                ),
            }
        result[regime] = {"n_windows": len(selected), "n_draws": raw["n_draws"], "metrics": metrics}
    return result


def verify_frozen_eone(freeze: dict) -> None:
    for name, digest in freeze["files"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Frozen historical/EONE file changed: {name}")


def merge_evidence(base: Path, sources: list[tuple[str, Path]], output: Path) -> dict:
    freeze = load_json(ROOT / "results/noise_generator_paper/neural_extension/eone_frozen.json")
    verify_frozen_eone(freeze)
    evidence = copy.deepcopy(load_json(base))
    old_control_root = ROOT / evidence["control_root"]
    control = load_json(old_control_root / "control_probe.json")
    with np.load(old_control_root / "control_measurements.npz", allow_pickle=False) as arrays:
        measurements = {key: arrays[key] for key in arrays.files}
    combined_root = ROOT / "results/noise_generator_paper/neural_extension/control_combined"
    combined_root.mkdir(parents=True, exist_ok=True)
    new_roots = {}
    for job, source in sources:
        source = source.resolve()
        result = load_json(source / "extension_result.json")
        method = result["method"]
        if method not in METHODS or method in new_roots:
            raise ValueError(f"Unknown or duplicated extension method {method}")
        raw_path = source / result["raw"]
        if sha256(raw_path) != result["raw_sha256"]:
            raise ValueError(f"Scored native output changed: {raw_path}")
        raw = load_json(raw_path)
        probe_root = source / result["control"]
        probe = load_json(probe_root / "control_probe.json")
        if sha256(probe_root / "control_probe.json") != result["control_sha256"]:
            raise ValueError("Native control output changed")
        control["methods"].update(probe["methods"])
        control["methods_probed"].append(method)
        with np.load(probe_root / "control_measurements.npz", allow_pickle=False) as arrays:
            for key in ("frequency_hz", "conditions", "rotor_rps", "draws"):
                if not np.array_equal(measurements[key], arrays[key]):
                    raise ValueError(f"Control measurement axis differs: {key}")
            measurements["mean_psd"] = np.concatenate(
                (measurements["mean_psd"], arrays["mean_psd"]), axis=0
            )
            measurements["methods"] = np.concatenate(
                (measurements["methods"], arrays["methods"]), axis=0
            )
        for name in probe["outputs"]["examples"]:
            destination = combined_root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((probe_root / name).read_bytes())
        evidence["method_order"].append(method)
        evidence["method_sources"][method] = {
            "job": job,
            "raw_path": str(raw_path.relative_to(ROOT)),
            "sha256": sha256(raw_path),
            "examples_root": str((source / "eval" / "examples").relative_to(ROOT)),
            "label": METHODS[method][1],
            "short_label": METHODS[method][2],
            "fitting_treatment": "standby_cruise_matched",
        }
        evidence["statistics"][method] = statistics(raw)
        summary = probe["methods"][method]["summary"]
        pooled = summary["pooled"]
        evidence["control"][method] = {
            "matches": pooled["n_matched"],
            "commands": pooled["n_commands"],
            "fraction": pooled["matched_command_fraction"],
            "mae_given_match_rps": pooled["rps_mae_given_match"],
            "response_slopes": summary["rotor0_command_response_slopes"],
        }
        for key in ("primary_methods", "matched_methods"):
            evidence["regime_comparison"][key].append(method)
        artifact = raw["calibration"]["regimes"]["standby"]["artifacts"]["state"]["path"]
        # Every fitted state lives beside this frozen, pulled manifest.
        manifests = list(source.rglob("neural_regimes.json"))
        manifests = [path for path in manifests if sha256(path) == result["manifest_sha256"]]
        if len(manifests) != 1 or not (manifests[0].parent / artifact).exists():
            raise ValueError("Cannot identify the unique pulled native fit manifest and states")
        evidence["regime_comparison"]["manifest_sources"][method] = {
            "path": str(manifests[0].relative_to(ROOT)),
            "sha256": result["manifest_sha256"],
        }
        new_roots[method] = {"job": job, "root": str(source.relative_to(ROOT)), **result}
        evidence["source_jobs"][f"neural_{method}"] = job
    if set(new_roots) != set(METHODS):
        raise ValueError(
            "The published comparison requires all three selected additional candidates"
        )
    # Legacy combined evidence keeps archived source_example_roots, not local
    # example files. Retain those references alongside the newly copied examples.
    control["outputs"]["examples"] = sorted(
        str(path.relative_to(combined_root)) for path in (combined_root / "examples").glob("*.npz")
    )
    control["provenance"]["neural_extension"] = new_roots
    control["provenance"]["refitted_in_this_probe"] = False
    np.savez_compressed(combined_root / "control_measurements.npz", **measurements)
    write_json(combined_root / "control_probe.json", control)
    evidence["control_root"] = str(combined_root.relative_to(ROOT))
    evidence["control_source_sha256"] = sha256(combined_root / "control_probe.json")
    evidence["neural_comparison"] = {
        "authorization": "2026-11-19: add PTR, published HPN and DDSP-Piano; freeze EONE; remove recording warp from presentation",
        "methods": list(METHODS),
        "sources": new_roots,
        "excluded_presentation_methods": ["heutschi2020", "heutschi2020_regimes"],
        "eone_freeze": "results/noise_generator_paper/neural_extension/eone_frozen.json",
        "prior_evidence": str(base.relative_to(ROOT)),
        "prior_evidence_sha256": sha256(base),
        "new_fidelity_pairs": sum(row["fidelity_pairs"] for row in new_roots.values()),
        "new_control_commands": sum(row["control_commands"] for row in new_roots.values()),
        "selection": "One seed, fixed fitting settings, no evaluation-based selection; original EONE and other raw rows retained unchanged",
    }
    evidence["neural_comparison"]["verification"] = {
        "complete_methods": list(METHODS),
        "eone_frozen_files": len(freeze["files"]),
        "old_method_rows_retained": True,
    }
    verify_frozen_eone(freeze)
    write_json(output, evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=ROOT / "results/noise_generator_paper/final_evidence.json"
    )
    parser.add_argument(
        "--source", action="append", required=True, help="JOB_ID=PATH_TO_PULLED_METHOD_OUTPUT"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/noise_generator_paper/neural_extension_evidence.json",
    )
    args = parser.parse_args()
    sources = []
    for text in args.source:
        job, separator, path = text.partition("=")
        if not separator:
            parser.error("--source requires JOB_ID=PATH")
        sources.append((job, Path(path)))
    evidence = merge_evidence(args.base.resolve(), sources, args.out.resolve())
    print(json.dumps(evidence["neural_comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()
