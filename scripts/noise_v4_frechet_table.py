#!/usr/bin/env python3
"""The Fréchet table of the wrap-up paper (``tab:similarity``): how close the
accepted v4 noise fits sound to their own drone, against two references::

    OMP_NUM_THREADS=4 PYTHONPATH=src:scripts python scripts/noise_v4_frechet_table.py \\
        --out results/noise_v4/similarity/frechet_table/

Material: every 4 s cruise window (all rotors >= 65 rev/s, the scorer's band)
of one HELD-OUT noise-only recording per drone — DREGON
``free-flight_nosource_room1`` (the validation room; the wind was tuned in
room 2) and MD2 ``FLY124`` (the validation flight; the fit is on FLY125) —
on the scorer's raw label (``gates.RAW_RPS_KEY``), all eight microphones,
16 kHz. The windows of a drone are split by alternation into two halves A
and B. Each window is scaled to unit RMS over its microphones, as the
training streams do, so recording gain does not count.

Measure: :func:`similarity._third_octave_levels` (128 ms periodic-Hann
frames, 16 ms hop, 24 third-octave bands 31.5 Hz-7.9 kHz, dB) pooled over
the windows and the microphones of a set, and the Fréchet distance
(:func:`similarity._frechet`, mean term + covariance term, dB^2) between
Gaussians fitted to the two sets of frame vectors.

Rows, per drone ``d`` (``o`` = the other drone), each the mean over the two
halves:

- ``real_real``: real A of ``d`` against real B of ``d`` — the distance
  between two disjoint sets of the drone's own windows;
- ``render``: real A against the render of the fit on the labels of B, and B
  against the render on A — the same trajectories as ``real_real``, so the
  row differs from it only by real versus synthetic audio;
- ``other_drone``: real A of ``d`` against real B of ``o`` and B against A.

``render_same_labels`` (real A against the render on A's own labels) is
written to the JSON as a diagnostic. The DREGON fit renders through
``render_noise`` (its own gust wind); the MD2 fit is the standby/cruise pair
through ``render_noise_regimes`` (all-cruise windows take the cruise fit).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SR = 16000
N_MICS = 8
WINDOW_S = 4.0
SEED = 2001
DRONES: dict[str, dict[str, Any]] = {
    "dregon": dict(
        dataset="DREGON-frames",
        recording="free-flight_nosource_room1",
        fits=dict(single="results/noise_v4/flight/benchwind/dregon__c7__benchwind_gust.json"),
    ),
    "md2": dict(
        dataset="michaels-frames",
        recording="FLY124",
        fits=dict(
            standby="results/noise_v4/flight/restarts/md2_fly125_standby_fpb__flight_v4__s0__r6c1.json",
            cruise="results/noise_v4/flight/restarts/md2_fly125_cruise_fpb__flight_v4__s0__r6c3.json",
        ),
    ),
}
RIG = {"dregon": "dregon", "md2": "michaels"}


def _render(fits: dict[str, dict[str, Any]], rps: np.ndarray, n: int) -> np.ndarray:
    from data_processing.noise_model.render import render_noise, render_noise_regimes

    if set(fits) == {"standby", "cruise"}:
        x = render_noise_regimes(fits, rps, sr=SR, n_mics=N_MICS, seed=SEED)
    else:
        x = render_noise(fits["single"], rps, sr=SR, n_mics=N_MICS, seed=SEED)
    return np.asarray(x, dtype=np.float64)[:, :n]


def _unit_rms(x: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.mean(x**2))


def _levels(x: np.ndarray) -> np.ndarray:
    """``(M * N, B3)`` third-octave frame levels of one window, mics pooled."""
    from experiments.noise_model import similarity as SIM

    lv = SIM._third_octave_levels(x, SR)
    return lv.reshape(-1, lv.shape[-1])


def _fd(a: list[np.ndarray], b: list[np.ndarray]) -> dict[str, float]:
    from experiments.noise_model import similarity as SIM

    mean_t, cov_t = SIM._frechet(np.concatenate(a), np.concatenate(b))
    return dict(total=mean_t + cov_t, mean_term=mean_t, cov_term=cov_t)


def _avg(*rows: dict[str, float]) -> dict[str, float]:
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def drone_levels(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Real and rendered frame levels of every cruise window of one drone."""
    from experiments.noise_model import gates as GT
    from experiments.stochastic_fit import clips as C
    from experiments.stochastic_fit import revised_eval as RE

    key = GT.RAW_RPS_KEY[RIG[name]]
    rec = next(iter(C.iter_recordings(spec["dataset"], (spec["recording"],), None, key)))
    found = C.windows(rec, seconds=WINDOW_S, max_clips=10_000, min_rps=GT.REGIME_BANDS["cruise"][0])
    fits = {k: json.loads(Path(p).read_text()) for k, p in spec["fits"].items()}
    real, rend, rows = [], [], []
    for t0, dur in found:
        win = RE.Window(spec["recording"], float(t0), float(dur), regime="cruise")
        clip = RE.load_window(
            win, dataset=spec["dataset"], version=None, channels=None, rps_key=key
        )
        x = np.asarray(clip.audio, dtype=np.float64)[:N_MICS]
        rps = np.atleast_2d(np.asarray(clip.rps, dtype=np.float64))
        y = _render(fits, rps, x.shape[1])
        real.append(_levels(_unit_rms(x)))
        rend.append(_levels(_unit_rms(y)))
        rows.append(
            dict(
                start_s=float(t0 - rec.t_start),
                duration_s=float(dur),
                mean_rps=float(rps.mean()),
                real_rms=float(np.sqrt(np.mean(x**2))),
                render_rms=float(np.sqrt(np.mean(y**2))),
            )
        )
        print(
            f"{name} @{rows[-1]['start_s']:.1f}s mean {rows[-1]['mean_rps']:.1f} rev/s", flush=True
        )
    return dict(rps_key=key, windows=rows, real=real, render=rend)


def table(lv: dict[str, dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    def half(x: list[np.ndarray], h: int) -> list[np.ndarray]:
        return x[h::2]

    out: dict[str, dict[str, dict[str, float]]] = {}
    for d, o in (("dregon", "md2"), ("md2", "dregon")):
        r, s, ro = lv[d]["real"], lv[d]["render"], lv[o]["real"]
        a, b = half(r, 0), half(r, 1)
        out[d] = dict(
            real_real=_fd(a, b),
            render=_avg(_fd(a, half(s, 1)), _fd(b, half(s, 0))),
            other_drone=_avg(_fd(a, half(ro, 1)), _fd(b, half(ro, 0))),
            render_same_labels=_avg(_fd(a, half(s, 0)), _fd(b, half(s, 1))),
        )
    return out


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--out", default="results/noise_v4/similarity/frechet_table/")
    args = ap.parse_args()
    t0 = time.time()
    lv = {d: drone_levels(d, spec) for d, spec in DRONES.items()}
    tab = table(lv)
    git = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    report = dict(
        git=git,
        seed=SEED,
        window_s=WINDOW_S,
        sr=SR,
        n_mics=N_MICS,
        drones={
            d: dict(
                dataset=spec["dataset"],
                recording=spec["recording"],
                rps_key=lv[d]["rps_key"],
                fits={k: dict(path=p, sha256=_sha256(p)) for k, p in spec["fits"].items()},
                n_windows=len(lv[d]["windows"]),
                windows=lv[d]["windows"],
            )
            for d, spec in DRONES.items()
        },
        table=tab,
        wall_s=time.time() - t0,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    md = [
        "# Fréchet table (tab:similarity)",
        "",
        f"git `{git}`, render seed {SEED}, {WINDOW_S:g} s cruise windows, unit RMS per window, "
        "mics pooled; dB^2 as total (mean term + covariance term).",
        "",
        "| row | DREGON | MD2 |",
        "|---|---:|---:|",
    ]
    for row in ("real_real", "render", "other_drone", "render_same_labels"):
        cells = [
            f"{tab[d][row]['total']:.1f} ({tab[d][row]['mean_term']:.1f} + "
            f"{tab[d][row]['cov_term']:.1f})"
            for d in ("dregon", "md2")
        ]
        md.append(f"| {row} | {cells[0]} | {cells[1]} |")
    md.append("")
    md.append(
        "Windows: "
        + ", ".join(f"{d} {len(lv[d]['windows'])}" for d in DRONES)
        + f"; wall {time.time() - t0:.0f} s."
    )
    (out / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
