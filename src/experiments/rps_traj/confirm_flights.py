"""Lock one development winner before evaluating the two reserved VID flights.

`lock FAMILY` requires a win on all five frozen development cells. `evaluate`
reads only the immutable lock, verifies the decoder reproduces a development
ensemble, then opens the two new flights. No model search on these results.
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.rps_traj import data, phase_dynamics, regime_campaign, schedule_eval
from experiments.rps_traj import unified_experiment as ue
from experiments.rps_traj.clock_experiment import CAMPAIGN, ROOT

RESERVED = ("indoor_loaded_round", "indoor_loadless_round")
DEST = Path("results/rps_traj/confirmation")
RIG, FOLD = "vid_m100", "0"


def lock(family):
    comparison = ROOT / "trials" / family / "comparison.json"
    if not comparison.is_file():
        comparison = ROOT / "comparison.json"
    scores = json.loads(comparison.read_text())
    if not all(scores["acceptance"][family].values()) or scores["max_parameters"][family] >= 277:
        raise ValueError(
            "no all-regime compact development winner: reserved speeds remain unopened"
        )
    source = ROOT / "fits" / f"{RIG}__{FOLD}__{family}.json"
    evidence = json.loads(source.read_text())
    record = dict(
        family=family,
        fit_path=str(source),
        fit_sha256=ue._sha256(source),
        worker_sha256=evidence["source_sha256"],
        comparison=str(comparison),
        comparison_sha256=ue._sha256(comparison),
        scores=scores["score_by_phase"],
        rig=RIG,
        fold=FOLD,
        reserved=list(RESERVED),
    )
    DEST.mkdir(exist_ok=True, parents=True)
    # Exclusive create: a second candidate cannot turn the same two flights
    # into purported independently-confirming evidence after a failure.
    with (DEST / "lock.json").open("x") as handle:
        json.dump(record, handle, indent=2)


def _model(lock, meta):
    path = Path(lock["fit_path"])
    if ue._sha256(path) != lock["fit_sha256"]:
        raise ValueError("frozen fit changed")
    family = lock["family"]
    module = importlib.import_module("experiments.rps_traj." + family)
    impl = getattr(module, "MODEL_CLASS", None)
    if impl is None:
        (impl,) = [
            x
            for x in vars(module).values()
            if isinstance(x, type)
            and x.__module__ == module.__name__
            and callable(getattr(x, "from_dict", None))
            and callable(getattr(x, "sample", None))
        ]
    m = impl.from_dict(json.loads(path.read_text())["params"])
    (original,) = meta["targets"]
    c = ue.schedule_codes(original["samples"], original["events"])
    clone = m.sample(c, ue.N_DRAWS, np.random.default_rng(np.random.SeedSequence(original["seed"])))
    with np.load(ROOT / "draws" / f"{RIG}__{FOLD}__{family}.npz") as z:
        np.testing.assert_array_equal(clone.astype(np.float32), z[original["flight"] + "__paths"])
    return m


def _record(path, name):
    from data_processing.sources import vid

    with np.load(path) as raw:
        frame = vid._frame_from_arrays(name, dict(raw), "VID")
    f = data._flight_from_frame(frame, "VID-frames", ("rps",), RIG)
    if f is None or f.flight != name:
        raise ValueError(f"no measured trajectory for {name}")
    f = replace(f, rps=f.rps[list(data.ROTOR_TO_MIXER[RIG])])
    return schedule_eval._record(f)


def evaluate():
    frozen = json.loads((DEST / "lock.json").read_text())
    family = frozen["family"]
    if frozen["reserved"] != list(RESERVED) or frozen["fold"] != FOLD or frozen["rig"] != RIG:
        raise ValueError("unexpected lock protocol")
    meta = json.loads((ROOT / "pack" / f"{RIG}__{FOLD}.json").read_text())
    m = _model(frozen, meta)
    donor_names = {r["flight"] for r in meta["train"] if r["measured"]}
    donors = [
        d
        for f in regime_campaign.rig_data(RIG).flights
        if f.flight in donor_names and (d := schedule_eval._record(f)) is not None
    ]
    if len(donors) != 3 or donor_names.intersection(RESERVED):
        raise ValueError("incorrect prospective training split")
    metric = ue._metric(json.loads((ROOT / "pack/manifest.json").read_text()), RIG)
    hover = meta["train_hover_rps"]
    old = {
        arm: phase_dynamics.fit_arm(arm, RIG, FOLD, str(CAMPAIGN), same_clock=True)
        for arm in phase_dynamics.EXAMPLE_ARMS
    }
    rows = []
    excluded = []
    for name in RESERVED:
        t = _record(DEST / "raw" / (name + ".motors.npz"), name)
        if t is None:
            excluded.append(dict(flight=name, phase="all", reason="no landmarks"))
            continue
        _, seed = schedule_eval._rng(RIG, FOLD, name, "draws")

        def generator(seed=seed):
            return np.random.default_rng(np.random.SeedSequence(seed))

        schedule = ue.schedule_codes(t.x.shape[1], ue._events(t.marks))
        new = m.sample(schedule, ue.N_DRAWS, generator())
        paths = {a: f(t.marks, ue.N_DRAWS, generator()) for a, f in old.items()}
        if not np.isfinite(new).all() or (new < 0).any():
            raise ValueError("nonphysical prospective ensemble")
        for phase in ue.PHASES:
            g = (
                schedule_eval._cruise_group(RIG, FOLD, t, donors, hover)
                if phase == "cruise"
                else schedule_eval._short_group(RIG, FOLD, t, donors, phase, hover)
            )
            if isinstance(g, str):
                excluded.append(dict(flight=name, phase=phase, reason=g))
                continue
            draws = {family: [new[:, :, i : i + g.n] for i in g.starts]}
            for arm, series in paths.items():
                if series is None:
                    raise ValueError(f"no prospective draw for {arm}")
                cuts, status, _ = schedule_eval._cut_generated(series, t, g)
                if status != "ok" or cuts is None:
                    raise ValueError((arm, name, phase, status))
                draws[arm] = cuts
            for arm, ensemble in draws.items():
                status, score, diag = schedule_eval._score(g, ensemble, metric)
                if status != "ok":
                    raise ValueError((arm, name, phase, status))
                rows.append(
                    dict(
                        flight=name,
                        phase=phase,
                        arm=arm,
                        score=score,
                        diagnostics=diag,
                        seed=seed,
                        support=g.support,
                    )
                )
    summary = {}
    for phase in ue.PHASES:
        paired = {r["flight"] for r in rows if r["phase"] == phase}
        if not paired:
            continue
        means = {
            arm: float(
                np.mean([r["score"] for r in rows if r["phase"] == phase and r["arm"] == arm])
            )
            for arm in (family, *old)
        }
        best = min(old, key=means.__getitem__)
        summary[phase] = dict(
            n_flights=len(paired),
            means=means,
            best_historical=best,
            historical_minus_frozen=means[best] - means[family],
        )
    ue._write_json(
        DEST / "scores.json", dict(lock=frozen, rows=rows, excluded=excluded, summary=summary)
    )
    print(json.dumps(dict(summary=summary, excluded=excluded), indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmds = parser.add_subparsers(required=True, dest="cmd")
    cmds.add_parser("evaluate")
    cmds.add_parser("lock").add_argument("family")
    args = parser.parse_args()
    if args.cmd == "lock":
        lock(args.family)
    else:
        evaluate()


if __name__ == "__main__":
    main()
