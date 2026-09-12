"""Does the frozen RPS model track SYNTHETIC cruise from the Stage-2 fit?

The acceptance gate for Stage 2. ``hppnet_l2_r2_s0`` reaches 0.99 rev/s on
Michael's held-out cruise, so a synthetic clip that carries Michael's comb
should be tracked comparably. The probe is known to be discriminative: on the
deliberately diverse control policy the same checkpoint reads 19.6 rev/s, and
on the previous fitted stream 13.8 rev/s with 7 of 24 draws refused outright.

Three numbers are reported, and all three are preregistered:

* **refusal rate** — a draw is refused when the mic-0 error is within 3 % of the
  clip's own mean true speed, i.e. the model output carries no speed at all.
  Target 0 of 12.
* **cross-channel spread** — the spread of the 8 per-microphone errors. Real
  clips read 0.01-0.17 rev/s. Target median <= 0.5.
* **8-mic PIT MAE** — target median <= 1.6 rev/s, twice the real held-out 0.79.

Every clip is an independent render: its own seed, its own rotor trajectory
taken from a real cruise window, and no reuse of any draw.

    python scripts/_stage2_probe.py --fit results/S2/cruise_8clip.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

import zoo
from experiments.stochastic_fit import native
from experiments.stochastic_fit import stage2 as S2
from metrics.salience_layers import LayerPeakRPSMetric

OUT = Path("results/S2/probe")
MODEL = "hppnet_l2_r2_s0"
N_CLIPS = 12
REFUSAL_TOL = 0.03
SPREAD_MAX = 0.5
MAE_MAX = 1.6


def _probe_helpers() -> Any:
    """The scoring and figure code from the existing probe, reused as-is."""
    path = Path(__file__).resolve().parent / "_synthetic_probe.py"
    spec = importlib.util.spec_from_file_location("_synthetic_probe", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def real_windows(
    recording: str, seconds: float, n: int, regime: str = "cruise"
) -> list[tuple[float, float]]:
    """Real windows of one regime, to supply the rotor trajectories.

    Standby holds exactly one contiguous run in FLY125, so its twelve draws
    share one trajectory and differ only in their random seed. That is stated
    where the numbers are reported: the draws are independent renders, not
    independent flights.
    """
    band = S2.REGIMES[regime]
    return S2.cruise_windows(
        recording,
        seconds=seconds,
        max_clips=n,
        min_rps=float(band["min_rps"]),
        max_rps=band["max_rps"],
        stride_s=band["stride_s"] or seconds,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=Path, default=Path("results/S2/cruise_8clip.json"))
    ap.add_argument("--clips", type=int, default=N_CLIPS)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--real", action="store_true", help="also score the REAL clips")
    ap.add_argument("--recording", default=S2.FIT_RECORDING)
    ap.add_argument("--regime", default="cruise", choices=sorted(S2.REGIMES))
    ap.add_argument("--out", type=Path, default=OUT / "probe.json")
    args = ap.parse_args()

    helpers = _probe_helpers()
    summary = json.loads(args.fit.read_text())
    entries = list(summary["clips"].values())
    fm = zoo.load(MODEL, ckpt="best")
    metric = LayerPeakRPSMetric()
    OUT.mkdir(parents=True, exist_ok=True)

    windows = real_windows(args.recording, args.seconds, args.clips, args.regime)
    rows: list[dict[str, Any]] = []
    for i in range(args.clips):
        start_s, dur = windows[i % len(windows)]
        real = native.decimate(
            native.load_native_clip(
                args.recording,
                start_s,
                dur,
                # The cache key must name the DATA, not the loop index. Keyed
                # "probe_00" the standby run read the cruise clips the cruise
                # run had left in the cache, and reported 80 rev/s windows for
                # a standby probe.
                clip_id=f"probe_{args.recording}_{args.regime}_{start_s:.2f}_{dur:g}",
            ),
            S2.SR,
        )
        rps = np.asarray(real.rps, dtype=np.float64)
        if args.real:
            # the same pipeline on the REAL clip: the reference the synthetic
            # numbers are read against, measured here rather than quoted
            audio = np.asarray(real.audio, dtype=np.float64)
        else:
            export = entries[i % len(entries)]["params"]
            audio = S2.render_from_export(
                export, rps, n_mics=int(real.audio.shape[0]), seed=1000 + i
            )
        n = min(audio.shape[-1], rps.shape[-1])
        audio, rps_i = audio[:, :n], rps[:, :n]

        per_mic = []
        pred0 = truth0 = None
        for mic in range(audio.shape[0]):
            pred, truth, mae = helpers.score(fm, metric, audio, rps_i, S2.SR, mic)
            per_mic.append(mae)
            if mic == 0:
                pred0, truth0 = pred, truth
        mean_true = float(np.mean(rps_i))
        refused = bool(abs(per_mic[0] - mean_true) <= REFUSAL_TOL * mean_true)
        row = dict(
            clip=i,
            start_s=start_s,
            mean_true_rps=round(mean_true, 2),
            mae_mic0=round(per_mic[0], 3),
            mae_8mic=round(float(np.mean(per_mic)), 3),
            spread=round(float(np.max(per_mic) - np.min(per_mic)), 3),
            refused=refused,
            per_mic=[round(v, 3) for v in per_mic],
        )
        assert pred0 is not None and truth0 is not None
        tag = f"{args.regime}_{'real' if args.real else 'fitted'}"
        row["figure"] = helpers.figure(tag, i, audio, rps_i, S2.SR, pred0, truth0, per_mic)
        rows.append(row)
        print(row, flush=True)

    mae = float(np.median([r["mae_8mic"] for r in rows]))
    spread = float(np.median([r["spread"] for r in rows]))
    refusals = int(sum(r["refused"] for r in rows))
    verdict = dict(
        n=len(rows),
        median_mae_8mic=round(mae, 3),
        median_spread=round(spread, 3),
        refusals=refusals,
        passes=bool(refusals == 0 and spread <= SPREAD_MAX and mae <= MAE_MAX),
        thresholds=dict(refusals=0, spread=SPREAD_MAX, mae=MAE_MAX),
    )
    print("\n" + json.dumps(verdict, indent=1), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(verdict=verdict, clips=rows), indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
