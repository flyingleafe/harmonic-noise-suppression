"""Native-framework generator comparison stages, with a Python-3.10-safe array boundary.

Export pinned project data in the project runtime, fit/render in the authors'
native framework, then score stored outputs in the project runtime. The native
render stage receives speed trajectories, blend gains and frozen checkpoints;
its manifest contains no evaluation audio. Historical benchmark/EONE files are
never written by these stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REGIMES = ("standby", "cruise")
METHODS = {
    "doerfler2026_ptr_regimes": ("doerfler2026_ptr", "PTR, 2-expert", "PTR"),
    "doerfler2026_hpn_regimes": ("doerfler2026_hpn", "HPN, 2-expert", "HPN"),
    "renault2023_piano_regimes": ("renault2023_piano", "DDSP-Piano, 2-expert", "Piano"),
}
CALIBRATION_ONLY = "md2 FLY125 accepted standby/cruise core supports only; no evaluation access"
SOURCE_FILES = (
    "src/experiments/generator_paper/neural_run.py",
    "src/experiments/generator_paper/neural_analysis.py",
    "src/experiments/generator_paper/engine_neural.py",
    "src/experiments/generator_paper/piano_baseline.py",
    "src/experiments/generator_paper/cases.py",
    "src/experiments/generator_paper/metrics.py",
    "src/experiments/generator_paper/control.py",
    "src/experiments/generator_paper/regime_baselines.py",
    "src/data_processing/noise_model/render.py",
    "dload.lock",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def source_hashes() -> dict[str, str]:
    return {name: sha256(ROOT / name) for name in SOURCE_FILES}


def _clip_meta(clip: Any, *, fit_regime: str | None = None) -> dict:
    result = {
        "key": clip.key,
        "rig": clip.rig,
        "recording_id": clip.recording_id,
        "start_s": clip.start_s,
        "sr": clip.sr,
        "regime": clip.regime,
        "provenance": clip.provenance,
    }
    if fit_regime is not None:
        result["fit_regime"] = fit_regime
    return result


def export_data(
    out: Path, inputs: Path, *, seed: int = 0, draws: int = 4, evaluation_limit: int | None = 26
) -> dict:
    from data_processing.noise_model.render import regime_blend_weight, regime_seeds
    from experiments.generator_paper.cases import load_flight_clips
    from experiments.generator_paper.control import command_track, conditions
    from experiments.generator_paper.regime_baselines import BLEND, load_regime_calibration

    if draws < 1:
        raise ValueError("draws must be positive")
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "calibration.json").exists() or (out / "requests.json").exists():
        raise FileExistsError(
            "Use an empty dataset export directory; exported inputs are immutable"
        )
    supports, provenance = load_regime_calibration(inputs)
    calibration: dict[str, list[dict]] = {}
    n_mics = supports["standby"][0].audio.shape[0]
    for regime in REGIMES:
        calibration[regime] = []
        for i, clip in enumerate(supports[regime]):
            path = out / "calibration" / f"{regime}__{i:03d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, audio=clip.audio, rps=clip.rps)
            calibration[regime].append(
                {
                    **_clip_meta(clip, fit_regime=regime),
                    "file": str(path.relative_to(out)),
                    "sha256": sha256(path),
                }
            )
    cal = {
        "schema": "generator-paper-neural-calibration/1",
        "seed": seed,
        "sample_rate": 16000,
        "n_mics": n_mics,
        "n_rotors": 4,
        "calibration_only": CALIBRATION_ONLY,
        "regimes": calibration,
        "calibration_provenance": provenance,
        "blend": BLEND,
        "source_sha256": source_hashes(),
        "snapshot_sha256": os.environ.get("NG_SNAPSHOT_SHA256"),
    }
    write_json(out / "calibration.json", cal)
    evaluation = load_flight_clips("md2", "evaluation", duration_s=4.0, limit=evaluation_limit)
    references, requests = [], []
    example_indices = [
        next(i for i, clip in enumerate(evaluation) if clip.regime == regime)
        for regime in ("standby", "transition", "cruise")
        if any(clip.regime == regime for clip in evaluation)
    ]

    def trajectory(name: str, tracks: np.ndarray) -> tuple[str, str, str, str]:
        rps_path = out / "requests" / f"{name}.npy"
        gains_path = out / "requests" / f"{name}__gains.npy"
        rps_path.parent.mkdir(parents=True, exist_ok=True)
        weight = regime_blend_weight(tracks, sr=16000)
        gains = np.stack((np.sqrt(np.maximum(1.0 - weight, 0.0)), np.sqrt(np.maximum(weight, 0.0))))
        np.save(rps_path, tracks, allow_pickle=False)
        np.save(gains_path, gains, allow_pickle=False)
        return (
            str(rps_path.relative_to(out)),
            sha256(rps_path),
            str(gains_path.relative_to(out)),
            sha256(gains_path),
        )

    def request(
        name: str, kind: str, tracks: tuple[str, str, str, str], entropy: list[int], **metadata: Any
    ) -> None:
        rng = np.random.default_rng(np.random.SeedSequence(entropy))
        streams = regime_seeds(int(rng.integers(0, 2**32 - 1)), REGIMES)
        requests.append(
            {
                "id": name,
                "kind": kind,
                "rps_file": tracks[0],
                "rps_sha256": tracks[1],
                "gains_file": tracks[2],
                "gains_sha256": tracks[3],
                "seed_entropy": entropy,
                "expert_seeds": streams,
                "output": name + ".npz",
                **metadata,
            }
        )

    for i, clip in enumerate(evaluation):
        path = out / "references" / f"reference__{i:03d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, real=clip.audio, rps=clip.rps, sr=clip.sr, key=clip.key, regime=clip.regime
        )
        references.append(
            {
                **_clip_meta(clip),
                "file": str(path.relative_to(out)),
                "sha256": sha256(path),
                "index": i,
            }
        )
        tracks = trajectory(f"evaluation__{i:03d}", clip.rps)
        for draw in range(draws):
            request(
                f"evaluation__{i:03d}__draw{draw}",
                "evaluation",
                tracks,
                [seed, i, draw],
                index=i,
                draw=draw,
                key=clip.key,
                regime=clip.regime,
            )
    conds = conditions()
    for condition in conds:
        tracks = trajectory(condition["name"], command_track(condition["rotor_rps"]))
        for draw in range(draws):
            request(
                f"control__{condition['name']}__draw{draw}",
                "control",
                tracks,
                [seed, 5000, draw],
                condition=condition["name"],
                draw=draw,
                rotor_rps=condition["rotor_rps"],
            )
    write_json(
        out / "evaluation.json",
        {
            "schema": "generator-paper-neural-evaluation-inputs/1",
            "seed": seed,
            "draws": draws,
            "sample_rate": 16000,
            "n_mics": n_mics,
            "cases": references,
            "example_indices": example_indices,
            "conditions": conds,
            "calibration_sha256": sha256(out / "calibration.json"),
        },
    )
    # Deliberately does not expose reference-audio files to native rendering.
    write_json(
        out / "requests.json",
        {
            "schema": "generator-paper-neural-render-requests/1",
            "sample_rate": 16000,
            "n_mics": n_mics,
            "seed": seed,
            "draws": draws,
            "blend": BLEND,
            "requests": requests,
        },
    )
    result = {
        "calibration_sha256": sha256(out / "calibration.json"),
        "evaluation_sha256": sha256(out / "evaluation.json"),
        "requests_sha256": sha256(out / "requests.json"),
        "evaluation_windows": len(evaluation),
        "requests": len(requests),
    }
    write_json(out / "export.json", result)
    print(json.dumps(result), flush=True)
    return result


def _native_clips(dataset: Path, entries: list[dict]) -> list[Any]:
    clips = []
    for entry in entries:
        path = dataset / entry["file"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Calibration array hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as arrays:
            audio, rps = arrays["audio"], arrays["rps"]
        provenance = dict(entry["provenance"])
        provenance["fit_regime"] = entry["fit_regime"]
        clips.append(
            SimpleNamespace(
                **{
                    key: entry[key]
                    for key in ("key", "rig", "recording_id", "start_s", "sr", "regime")
                },
                provenance=provenance,
                audio=audio,
                rps=rps,
            )
        )
    return clips


def _native_api(method: str) -> tuple[Any, Any, dict[str, Any]]:
    if method not in METHODS:
        raise ValueError(f"Unknown neural method {method}")
    if method == "renault2023_piano_regimes":
        from experiments.generator_paper.piano_baseline import fit_piano, load_piano

        return fit_piano, load_piano, {}
    from experiments.generator_paper.engine_neural import fit_engine, load_engine

    return fit_engine, load_engine, {"method": "ptr" if "_ptr_" in method else "hpn"}


def fit_experts(
    method: str, dataset: Path, out: Path, *, device: str, upstream: Path, smoke: bool = False
) -> Path:
    dataset, out = dataset.resolve(), out.resolve()
    cal = load_json(dataset / "calibration.json")
    seed = int(cal["seed"])
    fitter, loader, kwargs = _native_api(method)
    base = METHODS[method][0]
    out.mkdir(parents=True, exist_ok=True)
    records = {}
    for regime in REGIMES:
        state = out / f"{base}__{regime}.json"
        record_path = out / f"{base}__{regime}.record.json"
        entries = cal["regimes"][regime]
        if smoke:
            entries = entries[:1]
        if record_path.exists():
            record = load_json(record_path)
            if (
                record["calibration_sha256"] != sha256(dataset / "calibration.json")
                or record["smoke"] != smoke
                or record["fit_inputs"] != entries
            ):
                raise ValueError("Existing fit record has different input/protocol")
            loaded = loader(state, device=device, upstream=upstream)
            if loaded.report != record["fit"]:
                raise ValueError("Existing native fit differs from its record")
            print(f"Reusing frozen {method}/{regime}", flush=True)
        else:
            clips = _native_clips(dataset, entries)
            if smoke:
                for clip in clips:
                    clip.audio = clip.audio[:1]
                if method == "renault2023_piano_regimes":
                    from experiments.generator_paper.piano_baseline import PianoSettings

                    kwargs["settings"] = PianoSettings(
                        epochs=(1, 1, 1), steps_per_epoch=1, batch_sizes=(1, 1, 1)
                    )
                else:
                    from experiments.generator_paper.engine_neural import EngineSettings

                    kwargs["settings"] = EngineSettings(epochs=1, batch_size=1)
            started = time.monotonic()
            report = fitter(clips, state, seed=seed, device=device, upstream=upstream, **kwargs)
            if report["status"] != "completed" or bool(report["production_settings"]) == smoke:
                raise ValueError(
                    "Native fit did not complete the requested production/smoke protocol"
                )
            fit_inputs = [
                {key: entry[key] for key in ("key", "regime", "fit_regime", "provenance")}
                for entry in entries
            ]
            record = {
                "fit": report,
                "fit_inputs": entries,
                "benchmark_fit_inputs": fit_inputs,
                "fit_seed": seed,
                "fit_device": device,
                "fit_wall_seconds": time.monotonic() - started,
                # Native TensorFlow runs on Python 3.10, which has no datetime.UTC.
                "completed_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
                "smoke": smoke,
                "calibration_sha256": sha256(dataset / "calibration.json"),
            }
            write_json(record_path, record)
        records[regime] = {
            "fit": record["fit"],
            "fit_inputs": record["benchmark_fit_inputs"],
            **{
                key: record[key]
                for key in ("fit_seed", "fit_device", "fit_wall_seconds", "completed_utc")
            },
            "artifacts": {
                "state": {"path": state.name, "sha256": sha256(state)},
                "record": {"path": record_path.name, "sha256": sha256(record_path)},
            },
        }
    manifest = {
        "schema": "generator-paper-neural-regimes/1",
        "methods": [method],
        "method": method,
        "base_method": base,
        "family": base,
        "fitting_treatment": "standby_cruise_matched",
        "shared_state": [method],
        "calibration_only": CALIBRATION_ONLY,
        "calibration_provenance": cal["calibration_provenance"],
        "calibration_windows": [
            entry for regime in REGIMES for entry in records[regime]["fit_inputs"]
        ],
        "seed": seed,
        "n_mics": 1 if smoke else cal["n_mics"],
        "n_rotors": 4,
        "sample_rate": cal["sample_rate"],
        "blend": cal["blend"],
        "regimes": records,
        "source_sha256": cal["source_sha256"],
        "calibration_sha256": sha256(dataset / "calibration.json"),
        "production_settings": not smoke,
    }
    path = out / "neural_regimes.json"
    write_json(path, manifest)
    return path


def calibration_report(manifest_path: Path) -> dict:
    manifest = load_json(manifest_path)
    return {
        **manifest,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "refitted_in_this_worker": False,
    }


def render_requests(
    method: str,
    manifest_path: Path,
    requests_path: Path,
    out: Path,
    *,
    device: str,
    upstream: Path,
    kinds: tuple[str, ...] = ("evaluation", "control"),
) -> dict:
    manifest_path, requests_path, out = (
        manifest_path.resolve(),
        requests_path.resolve(),
        out.resolve(),
    )
    manifest, tasks = load_json(manifest_path), load_json(requests_path)
    if manifest["method"] != method or manifest["blend"] != tasks["blend"]:
        raise ValueError("Native render requests and fitted model disagree")
    _, loader, _ = _native_api(method)
    models = {}
    for regime in REGIMES:
        entry = manifest["regimes"][regime]["artifacts"]["state"]
        path = manifest_path.parent / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Frozen native state hash mismatch: {path}")
        models[regime] = loader(path, device=device, upstream=upstream)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for request in tasks["requests"]:
        if request["kind"] not in kinds:
            continue
        destination = out / request["output"]
        rps_path = requests_path.parent / request["rps_file"]
        gains_path = requests_path.parent / request["gains_file"]
        if (
            sha256(rps_path) != request["rps_sha256"]
            or sha256(gains_path) != request["gains_sha256"]
        ):
            raise ValueError("Speed-only render input hash mismatch")
        rps = np.load(rps_path, allow_pickle=False)
        gains = np.load(gains_path, allow_pickle=False)
        expected_shape = (manifest["n_mics"], rps.shape[-1])
        started = time.monotonic()
        audio = None
        for i, regime in enumerate(REGIMES):
            gain = gains[i]
            if not np.any(gain > 0):
                continue
            native = np.asarray(
                models[regime].render(
                    rps,
                    sr=tasks["sample_rate"],
                    rng=np.random.default_rng(request["expert_seeds"][regime]),
                ),
                dtype=np.float64,
            )
            if native.shape != expected_shape or not np.isfinite(native).all():
                raise ValueError(f"{method}/{request['id']}: invalid native output {native.shape}")
            audio = native * gain if audio is None else audio + native * gain
        if audio is None:
            raise ValueError("No active expert for a valid blend")
        generated = np.ascontiguousarray(audio, dtype=np.float32)
        np.savez_compressed(
            destination,
            generated=generated,
            sr=tasks["sample_rate"],
            request_id=request["id"],
            manifest_sha256=sha256(manifest_path),
        )
        records.append(
            {
                "id": request["id"],
                "file": destination.name,
                "sha256": sha256(destination),
                "seconds": time.monotonic() - started,
                "rps_sha256": request["rps_sha256"],
                "kind": request["kind"],
            }
        )
        print(f"Rendered {method}/{request['id']} {generated.shape}", flush=True)
    report = {
        "schema": "generator-paper-neural-rendered/1",
        "method": method,
        "manifest_sha256": sha256(manifest_path),
        "requests_sha256": sha256(requests_path),
        "device": device,
        "python": platform.python_version(),
        "records": records,
        "evaluation_audio_access": False,
        "output_level_adjustment": False,
    }
    write_json(out / "render_manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--inputs", type=Path, default=Path("results/noise_generator_paper/inputs"))
    export.add_argument("--seed", type=int, default=0)
    export.add_argument("--draws", type=int, default=4)
    export.add_argument("--evaluation-limit", type=int, default=26)
    for command in ("fit", "render"):
        child = sub.add_parser(command)
        child.add_argument("--method", choices=tuple(METHODS), required=True)
        child.add_argument("--out", type=Path, required=True)
        child.add_argument("--upstream", type=Path, required=True)
        child.add_argument("--device", default="cuda")
        if command == "fit":
            child.add_argument("--dataset", type=Path, required=True)
            child.add_argument("--smoke", action="store_true")
        else:
            child.add_argument("--manifest", type=Path, required=True)
            child.add_argument("--requests", type=Path, required=True)
            child.add_argument(
                "--kinds",
                nargs="+",
                choices=("evaluation", "control"),
                default=["evaluation", "control"],
            )
    score = sub.add_parser("score")
    score.add_argument("--method", choices=tuple(METHODS), required=True)
    score.add_argument("--dataset", type=Path, required=True)
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--rendered", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        export_data(
            args.out,
            args.inputs,
            seed=args.seed,
            draws=args.draws,
            evaluation_limit=args.evaluation_limit,
        )
    elif args.command == "fit":
        print(
            fit_experts(
                args.method,
                args.dataset,
                args.out,
                device=args.device,
                upstream=args.upstream,
                smoke=args.smoke,
            ),
            flush=True,
        )
    elif args.command == "render":
        render_requests(
            args.method,
            args.manifest,
            args.requests,
            args.out,
            device=args.device,
            upstream=args.upstream,
            kinds=tuple(args.kinds),
        )
    else:
        from experiments.generator_paper.neural_analysis import score_outputs

        score_outputs(args.method, args.dataset, args.manifest, args.rendered, args.out)


if __name__ == "__main__":
    main()
