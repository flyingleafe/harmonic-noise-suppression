"""PIT MAE of DREGON flight payloads under a chosen frozen tracker.

Sets: ``tuning`` = the five 8 s ``dregon-floor`` segments (``dregon_flight_v4.pool_specs``);
``scored`` = the five 4 s official cruise windows (``gates.DREGON_CRUISE_SUPPORTS``; report
only, never tune on them). Cruise support, all 8 mics; the arm ``real`` scores the recording.
Trackers: ``hppnet`` = the frozen nv3r4 scorer (``gates.SCORER_*``); ``hrspnet`` = HRSPNet
``rp_pyrseplstm_k64_alllab`` (rp-protocol grid, checkpoint ``--ckpt``, default
``best_real_median``).

    python scripts/noise_v4_pit_probe.py --tracker hrspnet --set scored --seeds 2001-2004 \
        nv3r4=results/noise_v3/fits_r4/dregon_room2_floor__flight_v3.json v4=<payload>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import stochastic_fit_revised_eval as ev

from data_processing.noise_model.render import render_noise
from experiments.noise_model import dregon_flight_v4 as DF
from experiments.noise_model import dregon_lawsearch_v4 as L
from experiments.noise_model import gates as GT
from experiments.noise_model import supports as SU
from experiments.stochastic_fit import revised_eval as RE

HRSPNET = "rp_pyrseplstm_k64_alllab"
#: conf/metrics/salience_layers_rp.yaml (the rp-protocol output grid)
RP_METRIC = dict(
    grid=dict(kind="linear", fmin=0.0, fmax=159.1796875, bins=164),
    n_layers=4,
    sigma_rev_s=1.0,
    rate=(16000, 512),
)


def tracker(name: str, ckpt: str | None) -> ev.Tracker:
    if name == "hppnet":
        return ev.Tracker(dict(experiment=GT.SCORER_EXPERIMENT, ckpt=ckpt or GT.SCORER_CKPT))
    return ev.Tracker(dict(experiment=HRSPNET, ckpt=ckpt or "best_real_median", metric=RP_METRIC))


def windows(which: str):
    """(label, window, real (8, T), raw rps (R, T)) per window of the set."""
    if which == "tuning":
        for spec in DF.pool_specs():
            clip = SU.flight_clip(spec)
            a = spec.args
            win = RE.Window(str(a["recording"]), float(a["start_s"]), float(a["dur_s"]), "cruise")
            yield str(a["recording"]), win, clip
        return
    for sup in GT.DREGON_CRUISE_SUPPORTS:
        clip = RE.load_window(
            sup.window,
            dataset=GT.DATASET["dregon"],
            version=None,
            channels=None,
            rps_key=GT.RAW_RPS_KEY["dregon"],
        )
        yield sup.recording, sup.window, clip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--tracker", choices=("hppnet", "hrspnet"), default="hrspnet")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--set", choices=("tuning", "scored"), default="tuning")
    ap.add_argument("--seeds", default="2001", help="one seed or a range a-b")
    ap.add_argument("--out", default=None, help="write the per-window table as JSON")
    ap.add_argument("arms", nargs="+", help="name=payload")
    args = ap.parse_args()
    lo, _, hi = args.seeds.partition("-")
    seeds = list(range(int(lo), int(hi or lo) + 1))
    tr = tracker(args.tracker, args.ckpt)
    fits = {
        "real": None,
        **{k: json.loads(Path(v).read_text()) for k, v in (a.split("=", 1) for a in args.arms)},
    }
    rows: list[dict] = []
    for label, win, clip in windows(args.set):
        real = np.asarray(clip.audio, dtype=np.float64)[:8]
        rps = np.atleast_2d(np.asarray(clip.rps, dtype=np.float64))
        n = real.shape[1]
        sup = RE.regime_support(
            win,
            rps,
            regime="cruise",
            min_rps=L.CRUISE_BAND[0],
            max_rps=L.CRUISE_BAND[1],
            sr=clip.sr,
        )
        pit: dict[str, float] = {}
        for k, fit in fits.items():
            vals = []
            for s in seeds if fit is not None else seeds[:1]:
                x = (
                    real
                    if fit is None
                    else np.asarray(render_noise(fit, rps, n_mics=8, seed=s))[:, :n]
                )
                vals.append(
                    float(
                        tr.pit(x, rps, mics=list(range(8)), expected_samples=n, support=sup)["mae"]
                    )
                )
            pit[k] = float(np.mean(vals))
        rows.append(dict(window=label, start_s=float(win.start_s), pit=pit))
        print(label, {k: round(v, 2) for k, v in pit.items()}, flush=True)
    means = {k: float(np.mean([r["pit"][k] for r in rows])) for k in fits}
    print(
        f"tracker {args.tracker} ({args.ckpt or 'default ckpt'}) set {args.set} seeds {args.seeds}"
    )
    for k in fits:
        print(
            f"{k:16s} " + " ".join(f"{r['pit'][k]:5.2f}" for r in rows) + f"  mean {means[k]:.3f}"
        )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                dict(tracker=tr.record, set=args.set, seeds=seeds, rows=rows, mean=means), indent=1
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
