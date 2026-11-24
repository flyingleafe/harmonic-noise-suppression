"""Build full-development fit inputs, export a compact law, or draw a chosen schedule.

The three all-data rigs are for subsequent use, never for reporting a score on
the development or reserved targets. Reproduction examples in the campaign log.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.rps_traj import unified_experiment as ue
from experiments.rps_traj.clock_experiment import ROOT
from experiments.rps_traj.regulated_bridge import RegulatedBridge

FULL = ROOT / "deployment"


def prepare(out: Path, source: Path) -> None:
    for rig in ue.RIGS:
        records = {}
        for path in sorted(source.glob(f"{rig}__*.json")):
            meta = json.loads(path.read_text())
            raw = path.with_name(path.stem + ".train.npz")
            if ue._sha256(raw) != meta["train_sha256"]:
                raise ValueError(f"changed training data: {raw}")
            with np.load(raw) as buf:
                all_x, all_c = buf["train_x"], buf["train_phase"]
                for r in meta["train"]:
                    if not r["measured"]:
                        continue
                    start = r["offset"]
                    end = start + r["samples"]
                    name = r["flight"]
                    x = all_x[:, start:end]
                    c = all_c[start:end]
                    if name in records:
                        if not np.array_equal(x, records[name][0]) or not np.array_equal(
                            c, records[name][1]
                        ):
                            raise ValueError(f"inconsistent copies of {rig}/{name}")
                    else:
                        records[name] = (
                            x.copy(),
                            c.copy(),
                            {k: r[k] for k in ("flight", "source", "measured", "samples")},
                        )
        expected = {"michaels": 4, "vid_m100": 4, "dregon": 5}[rig]
        if len(records) != expected:
            raise ValueError(f"{rig}: expect {expected} measured development flights")
        rows = [records[k] for k in sorted(records)]
        fn = out / "pack" / f"{rig}__all.train.npz"
        ue._write_npz(
            fn,
            dict(
                train_x=np.concatenate([r[0] for r in rows], axis=1),
                train_phase=np.concatenate([r[1] for r in rows]),
                train_offsets=np.r_[0, np.cumsum([len(r[1]) for r in rows])],
            ),
        )
        ue._write_json(
            out / "pack" / f"{rig}__all.json",
            dict(
                rig=rig,
                fold="all",
                train_sha256=ue._sha256(fn),
                train=[r[2] for r in rows],
                train_counts={"n_measured": len(rows)},
                targets=[],
            ),
        )


def export(out: Path) -> None:
    dest = out / "laws"
    dest.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for rig in ue.RIGS:
        fit = out / "fits" / f"{rig}__all__regulated_bridge.json"
        info = json.loads(fit.read_text())
        m = RegulatedBridge.from_dict(info["params"])
        coef = m.coefficients()
        path = dest / f"{rig}.npy"
        np.save(path, coef, allow_pickle=False)
        compact = RegulatedBridge.from_coefficients(np.load(path))
        c = np.repeat(np.arange(4), [10, 180, 400, 160])
        np.testing.assert_array_equal(
            m.sample(c, 32, np.random.default_rng(40)),
            compact.sample(c, 32, np.random.default_rng(40)),
        )
        metadata[rig] = dict(
            parameters=len(coef),
            parameter_bytes=coef.nbytes,
            fit_sha256=ue._sha256(fit),
            train_sha256=info["train_sha256"],
            law_sha256=ue._sha256(path),
            flights=json.loads((out / "pack" / f"{rig}__all.json").read_text())["train"],
        )
    ue._write_json(dest / "manifest.json", metadata)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", type=Path, default=ROOT / "pack")
    exp = sub.add_parser("export")
    for a in (prep, exp):
        a.add_argument("--out", type=Path, default=FULL)
    draw = sub.add_parser("draw")
    draw.add_argument("law", type=Path)
    draw.add_argument("output", type=Path)
    draw.add_argument(
        "--seconds",
        nargs=4,
        type=float,
        metavar=("OFF", "ARMED", "AIRBORNE", "RUNDOWN"),
        required=True,
    )
    draw.add_argument("--draws", type=int, default=1)
    draw.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args.out, args.source)
    elif args.command == "export":
        export(args.out)
    else:
        if not np.isfinite(args.seconds).all() or min(args.seconds) < 0 or args.draws < 1:
            p.error("durations must be finite and nonnegative; draws must be positive")
        c = np.repeat(
            np.arange(4, dtype=np.int8), np.rint(np.asarray(args.seconds) * ue.FS).astype(int)
        )
        if not len(c):
            p.error("empty schedule")
        m = RegulatedBridge.from_coefficients(np.load(args.law, allow_pickle=False))
        x = m.sample(c, args.draws, np.random.default_rng(args.seed))
        ue._write_npz(args.output, {"rps": x, "phase": c, "fs_hz": np.asarray(ue.FS)})
        print(f"{args.output}: {args.draws} x 4 x {len(c)} at {ue.FS:g} Hz")


if __name__ == "__main__":
    main()
