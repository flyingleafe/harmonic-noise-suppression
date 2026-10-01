"""Joint Bretthorst rotor-speed search on fetched static-rig recordings.

    python scripts/static_rig_joint.py --work /tmp/sr_avq --out results/static_rig/joint S1_seq1 S1_seq2

One JSON per recording (window tree, modes, linked tracks) and a printed
track table. Light enough for the laptop on AVQ/DREGON/SPCUP.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from experiments.static_rig import spectra as S  # noqa: E402
from experiments.static_rig.joint_speeds import JointParams, analyse  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=_ROOT / "results/static_rig/joint")
    ap.add_argument("--min-win", type=float, default=4.0)
    ap.add_argument("keys", nargs="+")
    args = ap.parse_args(argv)
    units = {u["key"]: u for u in json.loads((args.work / "units.json").read_text())}
    args.out.mkdir(parents=True, exist_ok=True)
    p = JointParams(min_win_s=args.min_win)
    for key in args.keys:
        u = units[key]
        fs = int(u["fs"])
        x = np.load(u["audio"], mmap_mode="r")
        a, b = S.motor_on_span(np.asarray(x), fs, S.Params())
        t0 = time.time()
        res = analyse(np.asarray(x[:, a:b], dtype=np.float64), fs, int(u["n_rotors"]), p)
        res.update(key=key, dataset=u["dataset"], span_s=[a / fs, b / fs], seconds=time.time() - t0)
        (args.out / f"{u['uid']}.json").write_text(json.dumps(res, indent=1))
        print(f"== {key}: {res['seconds']:.0f} s, {len(res['linked'])} leaves")
        for lf in res["linked"]:
            t0s, t1s = lf["t0"] + a / fs, lf["t1"] + a / fs
            print(
                f"  {t0s:6.1f}-{t1s:6.1f}s d{lf['depth']} "
                + " ".join(f"{s:8.3f}" for s in lf["speeds"])
                + f"  ev {lf['evidence']:9.0f}  gap {lf['runner_up_gap'] or 0:7.0f}"
            )
        for r, rot in enumerate(res["rotors"]):
            print(
                f"  rotor {r}: {rot['min']:.2f}-{rot['max']:.2f} (mean {rot['mean']:.2f}, hop scatter {rot['hop_scatter']:.3f})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
