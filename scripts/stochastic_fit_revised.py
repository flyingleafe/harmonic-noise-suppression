"""Fit the REVISED rotor-phase model (candidate 1) — the candidate's only entry point.

One shared shaft OU per rotor plus one scalar independent harmonic diffusion,
initialized from phase moments and fitted by ordinary Adam on a single plug-in
composite MAP objective (:mod:`experiments.stochastic_fit.revised_phase`).

    python scripts/stochastic_fit_revised.py --manifest <frozen-manifest> \
        --rig michaels --device cuda --out results/revised/michaels.json

    # the exact manifest schema this CLI requires, so a manifest can be frozen
    # against it instead of guessed at
    python scripts/stochastic_fit_revised.py --schema

EVERYTHING the fit needs comes from the frozen manifest: front end, state grid,
optimizer, composite temperature, priors, moment front end, and per rig the
dataset, rotor-track key, order cap, telemetry delays and clip list. A missing
key is an error naming the key — no value is ever invented here.

WHAT IT REFUSES, before anything expensive runs. The rotor track must be a RAW
telemetry key (``rps`` / ``motors_measured`` / ``motors_command``): ``auto`` and
``rps_refined`` are rejected, because candidate 1 is never fitted on labels this
project inferred. The clip list must be the FROZEN cohort of its rig —
Michael's FLY125 standby + ramp + cruise JOINTLY, DREGON's five room2
pure-noise 16 s cruise windows — with no duplicated or overlapping training
window, and held-out supports (FLY124, DREGON room1) are refused outright.
``--rig bench`` is allowed for planted/physical diagnostics only and is marked
in the export as not a scored arm.

If stage 1 does not identify the shaft dynamics, the moment diagnostic is
written and the run STOPS with a non-zero exit: an export with no valid
dynamics estimate would be rejected by the evaluator anyway.

Nothing here scores anything: acceptance gates live in
``scripts/stochastic_fit_revised_eval.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from experiments.stochastic_fit import clips as C
from experiments.stochastic_fit import revised_phase as RP
from experiments.stochastic_fit.stage2 import HELD_OUT_RECORDING, save

#: Waveform supports that must never be fitted. Michael's held-out recording is
#: ``stage2.HELD_OUT_RECORDING`` (FLY124); DREGON's room1 free-flight is held
#: out of the tracker's training pool, room2 is the fit pool.
HELD_OUT_TOKENS: dict[str, tuple[str, ...]] = {
    "michaels": (HELD_OUT_RECORDING,),
    "dregon": ("room1",),
    "bench": (),
}

MANIFEST_SCHEMA: dict[str, Any] = {
    "front_end": {
        "sr": "int, audio rate of the clips the fit consumes (16000)",
        "sample_rate_work": "int, the DECLARED integer-oversampled model grid the atoms are "
        "synthesized on (64000 = 4 x 16000); an integer multiple of sr, never called 'native'",
        "n_fft": "int, spectral window (16384)",
        "hop": "int, spectral hop (1024)",
        "band_hz": "[float, float], the fixed fit band ([30.0, 7900.0])",
    },
    "state": {"rate_hz": "float, latent state grid rate, independent of hop (e.g. 500.0)"},
    "optimizer": {
        "harmonic_chunk": "optional int or null, orders per kernel call "
        "(null = all at once); a memory device only, no order is ever dropped",
        "iters": "int, stage-1 marginal Adam iterations",
        "lr": "float, stage-1 marginal Adam learning rate",
        "carrier_iters": "int, stage-2 carrier-only Adam iterations",
        "carrier_lr": "float, stage-2 carrier-only Adam learning rate",
        "seed": "int",
        "frame_chunk": "int, frames per backward chunk (memory bound, 1 at n_fft 16384)",
        "frames_per_step": "int or null, frames sampled per step (null = all)",
        "training_recipe": "optional 'full' or 'band_energy_ladder'; the latter fits active "
        "orders 1..16, then 1..48, then full K during stage 1",
        "atom_dtype": "optional 'float32' (default) or 'float64'",
        "fit_method": "optional 'marginal_then_carrier' (C3/default) or "
        "'alternating_conditional_map' (C4 safeguarded conditional MAP)",
        "alternating_cycles": "optional int, C4 cycles (3)",
        "alternating_block_iters": "optional int, C4 Adam proposal steps per block (20)",
        "alternating_lr": "optional float, C4 proposal learning rate (0.05)",
        "alternating_backtracks": "optional int, bounded half-steps on rejection (4)",
        "warm_start_export": "optional path to a previous revised export for C4 training-only warm start",
        "fixed_lambda": "optional positive float, C4 fixed empirical OU lambda",
        "fixed_sigma": "optional positive float, C4 fixed empirical OU sigma",
        "initial_d": "optional positive float, C4 D reset after warm start (initialization only)",
    },
    "composite": {
        "temperature": "float > 0, the frozen composite temperature T = J/H "
        "(J = Var(U) over fixed baseline predictive draws, U = sum_i a_i (1 - I/M), "
        "H = sum_i a_i) on the UNNORMALIZED fit risk; the objective divides by it. "
        "Calibrated and frozen elsewhere; read, never tuned."
    },
    "priors": {
        "bias_std_hz": "float, per-clip bias prior std around the rotor population mean",
        "log_d_mean": "float, broad Gaussian prior on fitted log D",
        "log_d_std": "float",
        "log_sigma_mean": "optional float, default 0.0 (sigma prior mean = 1 rad/s)",
        "log_sigma_std": "optional float, default 2.0",
        "bias_mean_std_hz": f"optional float, default {RP.BIAS_MEAN_PRIOR_STD_HZ}",
    },
    "moments": {
        "window": "int, the moment stage's own (shorter) window",
        "hop": "int, the moment stage's own hop",
        "lags": "[int], frame lags whose increment variance is fitted",
        "gate": "optional; if present it must EQUAL the preregistered gate "
        f"{RP.MOMENT_GATE.as_dict()} — it records the gate, it cannot change it. The gate is "
        "applied per MICROPHONE and per FRAME, and lag pairs never straddle a gated frame",
    },
    "rigs": {
        "<michaels|dregon|bench>": {
            "dataset": "str, frames dataset NAME[@VERSION]",
            "rps_key": "str, the RAW telemetry track "
            f"({' / '.join(RP.RAW_RPS_KEYS)}); 'rps_refined' and 'auto' are REFUSED",
            "k_cap": "int, highest order modelled (130 in cruise)",
            "delay_s": "[float] one telemetry delay per rotor (or a single float)",
            "clips": [
                {
                    "recording": "str, recording id",
                    "regime": "str, standby | ramp | cruise",
                    "start_s": "float, on the recording's own clock",
                    "seconds": "float, window length",
                    "channels": "str or null, e.g. '0-3' (null = all)",
                }
            ],
            "cohort": "ENFORCED, not read: the frozen training cohort of this rig — "
            + "; ".join(f"{r}: {c['description']}" for r, c in RP.TRAINING_COHORT.items())
            + ". Duplicated or overlapping training windows are refused.",
        }
    },
    "heldout": {"recordings": "optional [str], extra supports the fit must refuse"},
}


def _device(name: str) -> str:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda asked for, but torch.cuda.is_available() is False")
    label = torch.cuda.get_device_name(0) if name == "cuda" else "cpu"
    print(f"device {name} ({label})  torch threads {torch.get_num_threads()}", flush=True)
    return name


def _req(manifest: dict[str, Any], path: str, where: Path) -> Any:
    """The manifest value at a dotted ``path``, or a precise error."""
    node: Any = manifest
    for i, part in enumerate(path.split(".")):
        if not isinstance(node, dict) or part not in node:
            seen = sorted(node) if isinstance(node, dict) else type(node).__name__
            raise ValueError(
                f"{where}: missing required manifest key '{path}' (at '{'.'.join(path.split('.')[: i + 1])}'; "
                f"present: {seen}). Run --schema for the schema this CLI requires."
            )
        node = node[part]
    return node


def _config(manifest: dict[str, Any], rig: str, where: Path, sha: str) -> RP.FitConfig:
    opt = _req(manifest, "optimizer", where)
    mom = _req(manifest, "moments", where)
    rig_node = _req(manifest, f"rigs.{rig}", where)
    gate = RP.MomentGate()
    if "gate" in mom:
        recorded = dict(mom["gate"])
        expected = gate.as_dict()
        if {k: recorded.get(k) for k in expected} != expected:
            raise ValueError(
                f"{where}: moments.gate disagrees with the PREREGISTERED gate.\n"
                f"  manifest: {recorded}\n  preregistered: {expected}\n"
                "The gate was fixed before any estimate was computed and is not a knob."
            )
    RP.check_rotor_track_key(
        str(_req(manifest, f"rigs.{rig}.rps_key", where)), rig=rig, where=str(where)
    )
    delay = rig_node.get("delay_s")
    if delay is None:
        raise ValueError(f"{where}: missing required manifest key 'rigs.{rig}.delay_s'")
    delay_s = tuple(float(v) for v in (delay if isinstance(delay, (list, tuple)) else [delay]))
    return RP.FitConfig(
        n_fft=int(_req(manifest, "front_end.n_fft", where)),
        hop=int(_req(manifest, "front_end.hop", where)),
        sr=int(_req(manifest, "front_end.sr", where)),
        sample_rate_work=int(_req(manifest, "front_end.sample_rate_work", where)),
        band_hz=tuple(float(v) for v in _req(manifest, "front_end.band_hz", where)),  # type: ignore[arg-type]
        state_rate_hz=float(_req(manifest, "state.rate_hz", where)),
        k_cap=int(_req(manifest, f"rigs.{rig}.k_cap", where)),
        delay_s=delay_s,
        iters=int(_req(manifest, "optimizer.iters", where)),
        lr=float(_req(manifest, "optimizer.lr", where)),
        carrier_iters=int(opt["carrier_iters"]) if "carrier_iters" in opt else None,
        carrier_lr=float(opt["carrier_lr"]) if "carrier_lr" in opt else None,
        seed=int(_req(manifest, "optimizer.seed", where)),
        frame_chunk=int(_req(manifest, "optimizer.frame_chunk", where)),
        frames_per_step=(
            None
            if _req(manifest, "optimizer.frames_per_step", where) is None
            else int(opt["frames_per_step"])
        ),
        training_recipe=str(opt.get("training_recipe", "full")),
        temperature=float(_req(manifest, "composite.temperature", where)),
        bias_std_hz=float(_req(manifest, "priors.bias_std_hz", where)),
        bias_mean_std_hz=float(
            manifest["priors"].get("bias_mean_std_hz", RP.BIAS_MEAN_PRIOR_STD_HZ)
        ),
        log_d_mean=float(_req(manifest, "priors.log_d_mean", where)),
        log_d_std=float(_req(manifest, "priors.log_d_std", where)),
        log_sigma_mean=float(manifest["priors"].get("log_sigma_mean", 0.0)),
        log_sigma_std=float(manifest["priors"].get("log_sigma_std", 2.0)),
        atom_dtype=str(opt.get("atom_dtype", "float32")),
        fit_method=str(opt.get("fit_method", "marginal_then_carrier")),
        alternating_cycles=int(opt.get("alternating_cycles", 3)),
        alternating_block_iters=int(opt.get("alternating_block_iters", 20)),
        alternating_lr=float(opt.get("alternating_lr", 0.05)),
        alternating_backtracks=int(opt.get("alternating_backtracks", 4)),
        warm_start_export=opt.get("warm_start_export"),
        fixed_lambda=(None if opt.get("fixed_lambda") is None else float(opt.get("fixed_lambda"))),
        fixed_sigma=(None if opt.get("fixed_sigma") is None else float(opt.get("fixed_sigma"))),
        initial_d=(None if opt.get("initial_d") is None else float(opt.get("initial_d"))),
        harmonic_chunk=opt.get("harmonic_chunk", 32),
        moments=RP.MomentConfig(
            window=int(_req(manifest, "moments.window", where)),
            hop=int(_req(manifest, "moments.hop", where)),
            lags=tuple(int(v) for v in _req(manifest, "moments.lags", where)),
        ),
        gate=gate,
        provenance=dict(
            manifest_path=str(where),
            manifest_sha256=sha,
            front_end_source="manifest",
            rig=rig,
            dataset=str(_req(manifest, f"rigs.{rig}.dataset", where)),
            rps_key=str(_req(manifest, f"rigs.{rig}.rps_key", where)),
            bench_diagnostic_only=rig == "bench",
            scored_arm=rig != "bench",
        ),
    )


def _refuse_heldout(manifest: dict[str, Any], rig: str, recordings: list[str], where: Path) -> None:
    tokens = list(HELD_OUT_TOKENS.get(rig, ()))
    tokens += [str(r) for r in manifest.get("heldout", {}).get("recordings", [])]
    hits = sorted(
        {rec for rec in recordings for tok in tokens if tok and tok.lower() in rec.lower()}
    )
    if hits:
        raise SystemExit(
            f"{where}: rigs.{rig}.clips names HELD-OUT support(s) {hits} "
            f"(held-out tokens for {rig}: {tokens}). Held-out waveforms are never fitted."
        )


def _load_rows(
    manifest: dict[str, Any], rig: str, cfg: RP.FitConfig, where: Path
) -> tuple[list[tuple[str, Any]], dict[str, str]]:
    rig_node = _req(manifest, f"rigs.{rig}", where)
    dataset = str(rig_node["dataset"])
    rps_key = str(rig_node["rps_key"])
    entries = _req(manifest, f"rigs.{rig}.clips", where)
    if not entries:
        raise ValueError(f"{where}: rigs.{rig}.clips is empty")
    _refuse_heldout(manifest, rig, [str(e.get("recording")) for e in entries], where)
    for i, entry in enumerate(entries):
        for key in ("recording", "regime", "start_s", "seconds"):
            if key not in entry:
                raise ValueError(f"{where}: rigs.{rig}.clips[{i}] is missing '{key}'")
    # THE FROZEN COHORT AND ITS DISJOINTNESS, before a single clip is decoded
    # and long before the moment stage: a non-cohort manifest is a different
    # experiment, not a cheaper one.
    RP.check_training_cohort(rig, entries, where=str(where))
    RP.check_manifest_supports(rig, entries, where=str(where))
    rows: list[tuple[str, Any]] = []
    regimes: dict[str, str] = {}
    for entry in entries:
        clip = C.load_clip(
            dataset,
            str(entry["recording"]),
            float(entry["start_s"]),
            float(entry["seconds"]),
            channels=entry.get("channels"),
            rps_key=rps_key,
        )
        clip = C.decimate(clip, cfg.sr)
        rows.append((clip.clip_id, clip))
        regimes[clip.clip_id] = str(entry["regime"])
        print(
            f"clip {clip.clip_id}  {entry['regime']:8s}  {clip.audio.shape[0]}ch  "
            f"{clip.duration_s:.2f}s  rotors {clip.rps.shape[0]}",
            flush=True,
        )
    return rows, regimes


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--manifest", type=Path, default=None, help="the FROZEN manifest (required)")
    ap.add_argument("--rig", default=None, choices=sorted(HELD_OUT_TOKENS))
    ap.add_argument("--out", type=Path, default=None, help="export path (required)")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument(
        "--schema",
        action="store_true",
        help="print the exact manifest schema this CLI requires (JSON) and exit",
    )
    args = ap.parse_args()

    if args.schema:
        print(json.dumps(MANIFEST_SCHEMA, indent=2))
        return
    missing = [
        flag
        for flag, value in (("--manifest", args.manifest), ("--rig", args.rig), ("--out", args.out))
        if value is None
    ]
    if missing:
        raise SystemExit(f"{' and '.join(missing)} required (or --schema to print the schema)")

    path = Path(args.manifest)
    if not path.exists():
        raise SystemExit(f"manifest {path} does not exist; freeze one against --schema first")
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw.decode())
    if args.rig not in _req(manifest, "rigs", path):
        raise SystemExit(
            f"{path}: no 'rigs.{args.rig}' block (present: {sorted(manifest['rigs'])})"
        )

    cfg = _config(manifest, args.rig, path, sha)
    device = _device(args.device)
    if args.rig == "bench":
        print(
            "BENCH: planted/physical diagnostics only — this export is marked "
            "scored_arm=false and must not become a scored candidate arm",
            flush=True,
        )
    t0 = time.time()
    rows, regimes = _load_rows(manifest, args.rig, cfg, path)
    cfg.provenance["regimes"] = regimes

    dynamics = RP.ShaftDynamics(
        lam=RP.FIXED_REFERENCE_LAMBDA,
        sigma=RP.INITIAL_SIGMA,
        d_init=math.exp(float(cfg.log_d_mean)),
        identified=False,
        diagnostics=dict(
            estimator_stage="marginal_then_carrier",
            lambda_source="fixed_reference",
            lambda_assumption_s_inv=RP.FIXED_REFERENCE_LAMBDA,
            initial_sigma_rad_s=RP.INITIAL_SIGMA,
            shared_phase_evidence="not_identified_by_marginal_score",
            note="phase-moment estimates are diagnostics only in round 2; no moment gate controls export validity",
        ),
    )
    print(
        f"\nstage 1: marginal expected-periodogram quasi-MAP over {len(rows)} clip(s)\n"
        f"  fixed lambda {dynamics.lam:.4g} 1/s (reference assumption, not measured); "
        f"sigma init {dynamics.sigma:.4g} rad/s; D init {dynamics.d_init:.4g} rad^2/s",
        flush=True,
    )

    print("\nstage 2: frozen-global carrier diagnostics", flush=True)
    export = RP.fit_revised(
        rows,
        rig_id=args.rig,
        dynamics=dynamics,
        config=cfg,
        device=device,
        progress=lambda m: print(f"  {m}", flush=True),
    )
    out = save(export, args.out)
    sens = export["diagnostics"]["coarsening_sensitivity"]
    flagged = (
        "\n  FLAGGED: " + sens["needs_real_grid_refinement_reason"] + "; this export needs the "
        "real planted 500 vs 1000 Hz grid comparison before acceptance"
        if sens["needs_real_grid_refinement"]
        else ""
    )
    mf = export["diagnostics"]["marginal_fit"]
    cf = export["diagnostics"]["carrier_fit"]
    print(
        f"\nwrote {out}  ({export['diagnostics']['runtime_s']:.0f}s fit, "
        f"{time.time() - t0:.0f}s total)\n"
        f"  marginal quasi-risk {mf['objective_start']:.4f} -> {mf['objective_end']:.4f} "
        f"valid={mf['valid']}\n"
        f"  carrier diagnostic MAP {cf['objective_start']:.4f} -> {cf['objective_end']:.4f} "
        f"valid={cf['valid']}\n"
        f"  fixed lambda {export['parameters']['lam']:.4g} 1/s; "
        f"sigma {export['parameters']['sigma']:.4g} rad/s; "
        f"D {export['parameters']['d_scalar']:.4g} rad^2/s\n"
        f"  shared_phase_evidence={export['shared_phase_evidence']}\n"
        f"  coarsening sensitivity (a sanity indicator, NOT convergence): subsampling the "
        f"fitted {sens['rate_hz']:.0f} Hz MAP path by two moves one frame's predicted spectrum "
        f"by {sens['band_l1_rel_change'] * 100:.2f}% (band L1)" + flagged,
        flush=True,
    )


if __name__ == "__main__":
    main()
