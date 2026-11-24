"""Fit the locked regulated bridge from one drone's explicit training pack.

This worker reads only the training array and the timing-only manifest. Validation
waveforms, donor normalizers and scores never enter this process.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from zlib import crc32

import numpy as np

from experiments.rps_traj import regulated_bridge as regulated
from experiments.rps_traj import unified_experiment as ue
from experiments.rps_traj.phases import block_mean, bridge_nans

ARM = "regulated_bridge"


def fit(root: Path, rig: str, max_iter: int = 80, *, arm: str = ARM) -> None:
    manifest_file = root / "pack" / f"{rig}.train.json"
    manifest = json.loads(manifest_file.read_text())
    array = root / "pack" / f"{rig}.train.npz"
    if ue._sha256(array) != manifest["train_sha256"] or manifest["rig"] != rig:
        raise ValueError(f"{rig}: stale or misidentified training split")
    records = manifest["train"]
    with np.load(array, allow_pickle=False) as src:
        values, offsets = src["x"], src["offsets"]
    if (
        values.shape[0] != 4
        or offsets[0] != 0
        or offsets[-1] != values.shape[1]
        or len(offsets) != len(records) + 1
    ):
        raise ValueError("invalid packed row offsets")
    tracks = []
    for row, lo, hi in zip(records, offsets[:-1], offsets[1:], strict=True):
        speeds = block_mean(bridge_nans(values[:, int(lo) : int(hi)]), 100.0, ue.FS)
        phase = ue.schedule_codes(speeds.shape[1], row["events"])
        tracks.append((speeds, phase))
    del values
    start = time.monotonic()
    if arm == ARM:
        law = regulated.fit(tracks, max_iter=max_iter)
        parameters = law.coefficients().tolist()
        model = regulated.RegulatedBridge.from_coefficients(parameters)
    elif arm == "lifting_control":
        from experiments.rps_traj import lifting_control

        law = lifting_control.fit(tracks, max_iter=max_iter)
        parameters = law.to_dict()
        parameters.pop("history", None)  # optimizer trace is recorded once outside the portable law
        model = lifting_control.LiftingControl.from_dict(parameters)
    else:
        raise ValueError(f"unknown whole-flight model {arm}")
    elapsed = time.monotonic() - start
    codes_train = np.bincount(np.concatenate([c for _, c in tracks]), minlength=4)
    level_checks = []
    if arm == "lifting_control":
        for row, (w, c) in zip(records, tracks, strict=True):
            start, stop = row["events"]["takeoff"], row["events"]["landing"]
            lo = 0 if start is None else start + 100
            hi = len(c) if stop is None else stop - 50
            if hi - lo < 25:
                continue
            seed = [2026, crc32(f"{rig}|train-check|{row['id']}".encode())]
            sim = model.sample(c, 8, np.random.default_rng(np.random.SeedSequence(seed)))
            level_checks.append(
                dict(
                    id=row["id"],
                    recording=row["recording"],
                    observed=w[:, lo:hi].mean(axis=1).tolist(),
                    generated=sim[:, :, lo:hi].mean(axis=(0, 2)).tolist(),
                )
            )
    draws = {}
    for i, t in enumerate(manifest["targets"]):
        schedule = ue.schedule_codes(t["n"], t["events"])
        ensemble = model.sample(
            schedule, ue.N_DRAWS, np.random.default_rng(np.random.SeedSequence(t["seed"]))
        )
        if (
            ensemble.shape != (ue.N_DRAWS, 4, len(schedule))
            or not np.isfinite(ensemble).all()
            or (ensemble < 0).any()
        ):
            raise FloatingPointError(f"{rig}/{t['id']}: invalid generated speeds")
        draws[f"t{i}"] = ensemble.astype(np.float32)
    out = root / "runs" / rig
    ue._write_npz(out / f"{arm}.draws.npz", draws)
    ue._write_json(
        out / f"{arm}.fit.json",
        dict(
            rig=rig,
            arm=arm,
            train_sha256=manifest["train_sha256"],
            manifest_sha256=ue._sha256(manifest_file),
            train_ids=[r["id"] for r in records],
            n_params=law.n_params,
            fit_seconds=elapsed,
            fit_iter=len(law.history) - 1,
            objective=law.history,
            parameters=parameters,
            training_level_check=level_checks,
            training_phase_samples=codes_train.tolist(),
            targets=[t["id"] for t in manifest["targets"]],
            draws_sha256=ue._sha256(out / f"{arm}.draws.npz"),
        ),
    )
    print(
        f"{rig}: {len(records)} training tracks, {len(draws)} validation tracks, {law.n_params} coefficients, {elapsed:.1f} s",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rig", required=True)
    args = parser.parse_args()
    fit(args.root, args.rig)


if __name__ == "__main__":
    main()
