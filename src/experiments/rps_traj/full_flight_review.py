"""Recreate complete comparison draws and verify against archived scored windows.

`references` runs once; `render` can be repeated for a new candidate. Missing
baseline time before its first sample or after its last sample stays blank. A
physically verified padded scored window remains in the existing phase panels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.rps_traj.clock_experiment import CAMPAIGN, ROOT
from experiments.rps_traj.phase_dynamics import fit_arm
from experiments.rps_traj.phases import landmarks
from experiments.rps_traj.unified_experiment import FS, N_DRAWS
from plots.flight_review import FLIGHTS, full_plates

OLD = ("ours", "phasevar4", "phasevar2+hsmm3")


def references(out: Path, all_flights: bool = False):
    buf = {}
    previous = out / "full_baseline_draw0.npz"
    if all_flights and previous.exists():
        with np.load(previous) as stored:
            buf.update({key: stored[key] for key in stored.files})
    with np.load(out / "baseline_draw0.npz") as archive:
        for path in sorted((out / "pack").glob("*.json")):
            if path.stem == "manifest":
                continue
            meta = json.loads(path.read_text())
            for target in meta["targets"]:
                flight = target["flight"]
                if not all_flights and FLIGHTS[meta["rig"]] != flight:
                    continue
                if f"{path.stem}__{flight}__real" in buf:
                    continue
                with np.load(path.with_name(path.stem + ".eval.npz")) as ref:
                    real = ref[flight + "__x"]
                marks = landmarks(real, FS)
                if marks is None or marks.motors_on is None:
                    raise ValueError(f"no onset in {flight}")
                key = f"{path.stem}__{flight}__"
                buf[key + "real"] = real.astype("float32")
                for arm in OLD:
                    gen = fit_arm(arm, meta["rig"], meta["fold"], str(CAMPAIGN), same_clock=True)
                    ensemble = gen(
                        marks,
                        N_DRAWS,
                        np.random.default_rng(np.random.SeedSequence(target["seed"])),
                    )
                    if not ensemble:
                        raise ValueError(f"no paths in {flight}/{arm}")
                    series, ev = ensemble[0]
                    shift = int(ev["motors_on"]) - marks.motors_on
                    arr = np.full(real.shape, np.nan, dtype="float32")
                    start, end = max(0, -shift), min(arr.shape[1], series.shape[1] - shift)
                    arr[:, start:end] = series[:, start + shift : end + shift]
                    for cell in target["cells"]:
                        start = cell["starts"][len(cell["starts"]) // 2]
                        piece = arr[:, start : start + cell["n"]]
                        scored = archive[key + cell["phase"] + "__" + arm]
                        mask = np.isfinite(piece)
                        if not mask.any() or not np.allclose(
                            scored[mask], piece[mask], rtol=1e-5, atol=2e-5
                        ):
                            raise ValueError(f"wrong complete draw: {flight}/{arm}/{cell['phase']}")
                    buf[key + arm] = arr
    if len(buf) != (13 if all_flights else len(FLIGHTS)) * (1 + len(OLD)):
        raise ValueError("missing complete flight")
    np.savez_compressed(out / ("full_panel_draw0.npz" if all_flights else previous.name), **buf)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["references", "render"])
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument("--families", nargs="+", default=["bridge_program"])
    parser.add_argument("--all-flights", action="store_true")
    args = parser.parse_args()
    if args.command == "references":
        references(args.out, args.all_flights)
    else:
        full_plates(args.out, args.families, OLD, all_flights=args.all_flights)


if __name__ == "__main__":
    main()
