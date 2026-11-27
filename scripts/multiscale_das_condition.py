"""DroneAudioSet-only experiment: resolve harmonic order with a leave-scene-out rig prior.

A throttle setting is not a speed. September's independently analyzed drone-only
spectra establish a distribution for each airframe/setting, not a rotor count or
a speed label. Never import the repeated placeholder rotors from a bench point.
The prior can select a harmonic *order* only; every finite speed is still derived
from this exact source's observed frequency. Assess on withheld upper/lower pairs.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from multiscale_corpus import coverage, read_rows, write_rows


def signature(key):
    flat = key.replace("/", "__")
    drone = re.search(r"__(drone[12])(?:-only)?__", flat)
    throttle = re.search(r"__throttle-(high|low)__", flat)
    assert drone and throttle
    scene = re.sub(r"__mic[123]_[^/]*?-(File\d|silence)(?:\.wav)?$", r"__\1", flat)
    scene = re.sub(r"^\d{6}_", "", scene)
    return (drone[1], throttle[1]), scene


def correct(track, bounds, center):
    rates = np.array(track["rates_rps"], float)
    err = np.array(track["uncertainty_rps"], float)
    state = np.array(track["provenance"], dtype=object)
    ids = np.array(track["track_id"], dtype=object)
    orders = [[list(v) for v in r] for r in track["order_support"]]
    diag = dict(track["diagnostics"])
    dec = []
    factors = np.array([1.0, 2.0, 0.5, 3.0, 1 / 3])
    index = defaultdict(list)
    for i, j in zip(*np.nonzero(np.isfinite(rates)), strict=True):
        index[ids[i, j]].append((i, j))
    for tid, pairs in index.items():
        ii, jj = np.array(pairs).T
        s = rates[ii, jj]
        lo, hi = np.quantile(s, [0.1, 0.9])
        choices = np.flatnonzero((lo * factors >= bounds[0]) & (hi * factors <= bounds[1]))
        if not len(choices):
            continue
        dist = np.abs(np.log(np.median(s) * factors[choices] / center))
        order = np.argsort(dist)
        win = int(choices[order[0]])
        if len(choices) > 1 and dist[order[1]] - dist[order[0]] < 0.08:
            continue
        gain = float(factors[win])
        dec.append({"id": tid, "multiplier": gain})
        if gain == 1.0:
            continue
        for i, j in pairs:
            clean = [
                round(k / gain)
                for k in orders[i][j]
                if k / gain >= 1 and abs(k / gain - round(k / gain)) < 1e-6
            ]
            if state[i, j] != "interpolated" and len(clean) < 2:
                rates[i, j] = err[i, j] = np.nan
                ids[i, j] = None
                state[i, j] = "unresolved"
                orders[i][j] = []
                continue
            orders[i][j] = clean
            rates[i, j] *= gain
            err[i, j] = max(0.3, err[i, j] * gain)
            state[i, j] = "interpolated" if state[i, j] == "interpolated" else "track_supported"
    for i in range(len(rates)):
        seen = []
        for j in sorted(
            np.flatnonzero(np.isfinite(rates[i])), key=lambda j: (-len(orders[i][j]), err[i, j])
        ):
            if any(abs(rates[i, j] - v) < 0.15 for v in seen):
                rates[i, j] = err[i, j] = np.nan
                ids[i, j] = None
                state[i, j] = "unresolved"
                orders[i][j] = []
            else:
                seen.append(rates[i, j])
    null = np.array(None, dtype=object)
    diag["condition_order_prior"] = {
        "source": "archived_leave_scene_out_das",
        "center_rps": center,
        "band_rps": bounds,
        "decisions": dec,
    }
    return dict(
        track,
        rates_rps=np.where(np.isfinite(rates), rates, null).tolist(),
        uncertainty_rps=np.where(np.isfinite(err), err, null).tolist(),
        provenance=state.tolist(),
        track_id=ids.tolist(),
        order_support=orders,
        diagnostics=diag,
    )


def load_cells(calib):
    cells = defaultdict(lambda: defaultdict(list))
    for p in json.loads(calib.read_text())["points"]:
        if p["corpus"] != "DroneAudioSet":
            continue
        cell, scene = signature(p["source_id"])
        s = np.array(p["speed_rev_s"][: p["n_resolved"]])
        if len(s):
            cells[cell][scene].append(float(np.median(s)))
    return cells


def from_condition(track, cells, key):
    cell, scene = signature(key)
    pool = np.array([np.median(v) for s, v in cells[cell].items() if s != scene])
    if len(pool) < 3:
        raise ValueError(
            "fewer than three independently named calibration segments for this condition"
        )
    med = float(np.median(pool))
    mad = float(np.median(abs(pool - med)))
    radius = max(0.18 * med, 3 * 1.4826 * mad)
    bounds = (max(0.72 * med, med - radius), min(1.28 * med, med + radius))
    return correct(track, bounds, med)


def run(root, out, calib, clock):
    cells = load_cells(calib)
    allow = {
        c["scene"]
        for c in json.loads(clock.read_text())["rows"]
        if c["pairs"]["mic2_8array-down/mic3_8array-up"]["screen_pass"]
    }
    rows = []
    for f in sorted((root / "tracks").glob("*.jsonl.gz")):
        for row in read_rows(f):
            if row["dataset"] != "DroneAudioSet" or row["recording"] not in allow:
                continue
            track = from_condition(row["track"], cells, row["key"])
            rows.append(
                {
                    **row,
                    "track": track,
                    "coverage": coverage(track, 4, row["duration_s"]),
                    "review_status": "rig_condition_prior_experiment",
                }
            )
    assert len(rows) == 69
    write_rows(out / "tracks/part-000.jsonl.gz", rows)
    info = json.loads((root / "summary.json").read_text())
    info["condition_calibration"] = (
        "independent historical rig spectra; held-out scene not used in prior"
    )
    (out / "summary.json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("bench", type=Path)
    p.add_argument("clocks", type=Path)
    a = p.parse_args()
    run(a.root, a.out, a.bench, a.clocks)
