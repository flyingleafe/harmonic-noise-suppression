"""Throwaway: the rotor-lines guard on round-4 hard path draws, by t.

    PYTHONPATH=src python scripts/_nv3_r4_guard_calib.py [--pin 6,6] [--n 3]

For t on a fixed grid, ``--n`` perturbed path points per t (the bank's own
spread and widths; the ordinary guards NOT applied, so the tonality guard is
measured on its own), each with and without the standby payload: the guard's
verdict and the min over patterns of the shared-order count. JSON to --out.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.noise_model import rig_sampler as RS

T_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def draw_rows(pin: dict[str, float] | None, n: int, standby_only: bool = False) -> list[dict]:
    anchors = RS.pinned_anchors("v3r4", pin=pin)
    probe = RS.RotorLinesProbe()
    a, b, sb = anchors["dregon"], anchors["michaels"], anchors["michaels_standby"]
    assert a is not None and b is not None and sb is not None
    rows = []
    for t in T_GRID:
        for i in range(n):
            rng = np.random.default_rng([20260927, int(t * 1000), i])
            mid = RS.interpolate_fits(a, b, t, k_max=RS.PATH_K_MAX)
            mid.pop("_path", None)
            cand, _drawn = RS.draw_fit(mid, rng, 3.0, RS.WIDTHS)
            for carried in (True,) if standby_only else (False, True):
                st = RS.draw_fit(sb, rng, 3.0, RS.WIDTHS)[0] if carried else None
                r = probe.measure(cand, st)
                worst = min(r["rotor_lines_shared"], key=r["rotor_lines_shared"].get)
                rows.append(
                    dict(
                        t=t,
                        i=i,
                        standby=carried,
                        ok=r["rotor_lines"],
                        worst=worst,
                        min_comparable=min(min(c) for c in r["rotor_lines_counts"].values()),
                        counts=r["rotor_lines_counts"],
                        shared=r["rotor_lines_shared"],
                        spread_db=r["rotor_lines_spread_db"],
                    )
                )
                print(
                    f"t={t:.2f} i={i} standby={int(carried)} ok={int(r['rotor_lines'])} "
                    f"min shared {r['rotor_lines_shared'][worst]} at {worst}",
                    flush=True,
                )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", default=None, help="amp,floor exponents pinned on both endpoints")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--standby-only", action="store_true")
    ap.add_argument("--out", default="results/noise_v3/rig_sampler/guard_calib_r4.json")
    args = ap.parse_args()
    pin = None
    if args.pin:
        amp, floor = (float(v) for v in args.pin.split(","))
        pin = {"amp_exp": amp, "floor_exp": floor}
    t0 = time.time()
    rows = draw_rows(pin, int(args.n), bool(args.standby_only))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(pin=pin, n=args.n, t_grid=T_GRID, rows=rows), indent=1))
    print(f"wrote {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
