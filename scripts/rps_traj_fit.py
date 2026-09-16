#!/usr/bin/env python3
"""Fit both rotor-speed models to every rig and print the PASS/FAIL comparison.

One CLI, no laptop state: it loads each rig's real flights, writes the real
statistics, fits the NEW model (``experiments.rps_traj.model.fit_rig``) and the
moment-matched BASELINE (``experiments.rps_traj.baseline.fit_baseline``), draws
the frozen statistics from both samplers and scores them with the frozen
discrepancy and acceptance rule.

Artefacts under ``--out`` (default ``results/rps_traj``)::

    fits/new/<rig>.json     fitted new-model parameters and fit provenance
    fits/base/<rig>.json    fitted baseline
    stats/real/<rig>.json   real TrajStats (identical to rps_traj_real_stats.py)
    stats/new/<rig>.json     model TrajStats from the new sampler
    stats/base/<rig>.json    model TrajStats from the baseline sampler
    summary.json            per-rig discrepancies, verdicts, NLL, wall times

Sample durations.  The frozen statistics are segment-length weighted, so a
model is asked for the rig's OWN airborne segment lengths — plus the 2 s the
airborne rule erodes from a sample's ends, so that what survives the rule has
exactly the real length.

``--n-rep`` applies to BOTH models, and the new model's statistics use its
antithetic-offset sampler (see ``Params.sampler``): the per-flight offset is
Monte-Carlo noise on the pooled mean, and the baseline has no offset term, so
without both the new model would be charged for a term the comparison cannot
see.  Each record also carries the airborne-rule RETENTION of the sampled
flights, which is the diagnostic behind a model whose mean looks biased.

Exit status is 0 whenever the fits and records were written — a FAIL verdict is
a finding, not a failure.  ``scripts/rps_traj_compare.py`` is the judge and
returns non-zero when a rig does not pass.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if (_HERE.parent / "src").is_dir() else Path.cwd().resolve()
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from rps_traj_compare import report_rig  # noqa: E402  (sibling script, after the path pin)

from data_processing.trajectory_model import NewFit  # noqa: E402
from experiments.rps_traj.data import (  # noqa: E402
    RATE_HZ,
    Flight,
    airborne_segments,
    load_rig,
)
from experiments.rps_traj.model import SAMPLE_PAD_S, fit_rig  # noqa: E402
from experiments.rps_traj.stats import (  # noqa: E402
    FAMILIES,
    TrajStats,
    compute_stats,
    discrepancy,
    stats_from_samples,
)

#: The campaign's rigs, in report order.
RIGS = (
    "michaels",
    "dregon",
    "neurobem_quad",
    "pitcn_quad",
    "nanobench_cf21b",
    "vid_m100",
    "blackbird_quad",
)


def sample_durations(flights: list[Flight]) -> list[float]:
    """The rig's airborne segment lengths, padded by the rule's erosion.

    ``stats_from_samples`` runs a model sample through the same airborne rule
    as real telemetry, which erodes 1 s from each end; asking for
    ``segment + 2 s`` therefore makes the model's SURVIVING segments the same
    length as the real ones, which is what the duration-weighted ACF compares.
    """
    out: list[float] = []
    for flight in flights:
        for sl in airborne_segments(flight.rps, flight.fs):
            out.append((sl.stop - sl.start) / float(flight.fs) + SAMPLE_PAD_S)
    return out


def _write(path: Path, payload: str | dict[str, Any]) -> Path:
    """Write JSON text; a ``to_json()`` that handed back a dict is serialised."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=2)
    path.write_text(text)
    return path


def _load_or_fit_baseline(
    flights: list[Flight], rig: str, seed: int, out: Path, reuse: bool
) -> Any:
    """Import the baseline lazily so ``--skip-base`` works before it lands.

    With ``reuse`` an existing ``fits/base/<rig>.json`` is loaded instead of
    refitting — the baseline search is the expensive half of a run and its fit
    does not depend on anything the new model changes.
    """
    from experiments.rps_traj.baseline import BaselineFit, fit_baseline  # noqa: PLC0415

    cached = out / "fits" / "base" / f"{rig}.json"
    if reuse and cached.is_file():
        print(f"  [{rig}] reusing baseline fit {cached}")
        return BaselineFit.from_json(cached.read_text())
    return fit_baseline(flights, rig, seed=seed)


def warm_start(round_path: Path, rig: str) -> Any:
    """The previous round's fitted ``Params`` for ``rig``, or ``None``.

    Round 4's likelihood is expensive, so restart 0 starts from round 3's answer
    instead of the data-driven guess; restarts 1-3 are jittered around it.  A
    round-3 dict has ``sigma_w`` where round 4 has ``(tau_e, sigma_e)``, so the
    white term is re-expressed as a measurement OU with a 5 ms time constant —
    white on any grid this campaign uses — and everything else carries over
    unchanged.
    """
    from data_processing.trajectory_model import Params  # noqa: PLC0415

    if not round_path.is_file():
        return None
    for record in json.loads(round_path.read_text()).get("rigs", []):
        if record.get("rig") != rig or "params" not in record:
            continue
        payload = dict(record["params"])
        if "tau_e" not in payload:
            if "sigma_w" not in payload or "f0" not in payload:
                return None
            payload["tau_e"] = 0.005
            payload["sigma_e"] = max(float(payload.pop("sigma_w")), 1e-3)
        return Params.from_dict(payload)
    return None


def _params_line(fit: Any) -> str:
    p = fit.params

    def row(label: str, values: Any, precision: int = 2) -> str:
        body = np.array2string(
            np.asarray(values, dtype=np.float64), precision=precision, floatmode="fixed"
        )
        return f"    {label:<9} {body}"

    return "\n".join(
        [
            f"    theta {p.theta:+.3f} rad  tau_e {p.tau_e:.4f} s  sigma_e {p.sigma_e:.4f} rev/s",
            row("tau_slow", p.tau_slow) + " s",
            row("sig_slow", p.sigma_slow) + " rev/s",
            row("f0", p.f0) + " Hz",
            row("zeta", p.zeta),
            row("sig_osc", p.sigma_osc) + " rev/s",
            f"    s_c {p.s_c:.2f} rev/s (common)",
            row("s_r", p.s_r) + " rev/s (per rotor)",
        ]
    )


def airborne_retention(
    sampler: Any, durations: list[float], fs: float, n_rep: int, seed: int
) -> float:
    """Fraction of a model's sampled rotor samples the frozen airborne rule keeps.

    A DIAGNOSTIC, not a score.  ``stats_from_samples`` runs model samples through
    :func:`airborne_segments` exactly like real telemetry, and on a pure-airborne
    sample the threshold ``0.5 * p90(rotor mean)`` sits only a few sigma below
    the sample's own mean when a rig's relative fluctuation is large.  Samples
    are then dropped from the LOW side only, which biases the retained mean UP:
    round 1 lost 34 % of neurobem's samples this way and its retained mean came
    out 26.8 rev/s above ``mu``.  Real flights escape it because their ``p90``
    is taken over the whole recording, so their threshold is half of HOVER.
    """
    kept = total = 0
    for rep in range(int(n_rep)):
        for i, duration in enumerate(durations):
            n = int(round(float(duration) * fs))
            if n <= 0:
                continue
            rps = np.asarray(sampler(n, np.random.default_rng([int(seed), rep, i])))
            total += rps.size
            kept += sum(rps[:, sl].size for sl in airborne_segments(rps, fs))
    return float(kept) / float(total) if total else float("nan")


def rescore_rig(rig: str, out: Path, *, seed: int, n_rep: int) -> dict[str, Any]:
    """Re-score a rig from its EXISTING fits, with only the offset model changed.

    Round 5 changed nothing about the dynamics — it split the per-flight offset
    into a common and a per-rotor part (:func:`model.pooled_means`) — so
    refitting the 23 dynamic parameters would burn hours to land on the same
    answer.  This path therefore loads the previous round's ``fits/new`` and
    ``fits/base``, re-estimates ONLY ``(s_c, s_r)`` from the flights' airborne
    means, and recomputes the sampled statistics and the verdict.  The record
    it writes says so in ``dynamics_from``.
    """
    from experiments.rps_traj.model import pooled_means  # noqa: PLC0415

    flights = load_rig(rig)
    durations = sample_durations(flights)
    fs = float(flights[0].fs)
    real = compute_stats(flights)
    _write(out / "stats" / "real" / f"{rig}.json", real.to_json())

    payload = json.loads((out / "fits" / "new" / f"{rig}.json").read_text())
    mu, s_c, s_r = pooled_means(flights, fs)
    params = dict(payload["params"])
    params.pop("s", None)
    params["mu"] = mu.tolist()
    params["s_c"], params["s_r"] = s_c, s_r.tolist()
    payload["params"] = params
    new_fit = NewFit.from_dict(payload)
    _write(out / "fits" / "new" / f"{rig}.json", new_fit.to_json())

    new_stats = stats_from_samples(
        new_fit.sampler(fs, antithetic_offsets=True), durations, fs=fs, n_rep=n_rep, seed=seed
    )
    _write(out / "stats" / "new" / f"{rig}.json", new_stats.to_json())

    base_fit = _load_or_fit_baseline(flights, rig, seed, out, reuse=True)
    base_stats = stats_from_samples(base_fit.sampler(fs), durations, fs=fs, n_rep=n_rep, seed=seed)
    _write(out / "stats" / "base" / f"{rig}.json", base_stats.to_json())

    record: dict[str, Any] = {
        "rig": rig,
        "dynamics_from": payload.get("round", "round4"),
        "rescored": True,
        "n_flights": len(flights),
        "airborne_s": float(sum(durations) - SAMPLE_PAD_S * len(durations)),
        "n_segments": len(durations),
        "fit_rate_hz": new_fit.fit_rate_hz,
        "n_scored": new_fit.n_scored,
        "nll": new_fit.nll,
        "n_iter": new_fit.n_iter,
        "wall_s": new_fit.wall_s,
        "n_blocks": new_fit.n_blocks,
        "params": new_fit.params.to_dict(),
        "new": discrepancy(new_stats, real),
        "base": discrepancy(base_stats, real),
        "new_retention": airborne_retention(
            new_fit.sampler(fs, antithetic_offsets=True), durations, fs, n_rep, seed
        ),
        "base_retention": airborne_retention(base_fit.sampler(fs), durations, fs, n_rep, seed),
    }
    print(
        f"\n[{rig}] RESCORED from {record['dynamics_from']} dynamics; "
        f"{len(flights)} flights, {record['airborne_s']:.0f} s airborne\n"
        + _params_line(new_fit)
        + f"\n    airborne-rule retention: new {record['new_retention']:.1%}"
        f", base {record['base_retention']:.1%}"
    )
    record["verdict"] = report_rig(rig, real, base_stats, new_stats)
    return record


def run_rig(
    rig: str,
    out: Path,
    *,
    seed: int,
    n_rep: int,
    n_restarts: int,
    skip_base: bool,
    reuse_base: bool,
    warm: Path,
) -> dict[str, Any]:
    """Fit, sample and score one rig; returns its summary record."""
    flights = load_rig(rig)
    if not flights:
        raise SystemExit(f"rig {rig!r} yielded no flights")
    durations = sample_durations(flights)
    if not durations:
        raise SystemExit(f"rig {rig!r} has no airborne segments")
    fs = float(flights[0].fs)

    real = compute_stats(flights)
    _write(out / "stats" / "real" / f"{rig}.json", real.to_json())

    new_fit = fit_rig(flights, rig, seed=seed, n_restarts=n_restarts, start=warm_start(warm, rig))
    _write(out / "fits" / "new" / f"{rig}.json", new_fit.to_json())
    # Antithetic offsets: MC variance reduction on the pooled mean only (see
    # Params.sampler).  One sampler instance per estimator, since it is stateful.
    new_stats = stats_from_samples(
        new_fit.sampler(fs, antithetic_offsets=True), durations, fs=fs, n_rep=n_rep, seed=seed
    )
    _write(out / "stats" / "new" / f"{rig}.json", new_stats.to_json())

    record: dict[str, Any] = {
        "rig": rig,
        "n_flights": len(flights),
        "airborne_s": float(sum(durations) - SAMPLE_PAD_S * len(durations)),
        "n_segments": len(durations),
        "fit_rate_hz": new_fit.fit_rate_hz,
        "n_scored": new_fit.n_scored,
        "nll": new_fit.nll,
        "n_iter": new_fit.n_iter,
        "wall_s": new_fit.wall_s,
        "n_blocks": new_fit.n_blocks,
        "params": new_fit.params.to_dict(),
        "new": discrepancy(new_stats, real),
        "new_retention": airborne_retention(
            new_fit.sampler(fs, antithetic_offsets=True), durations, fs, n_rep, seed
        ),
    }

    base_stats: TrajStats | None = None
    if not skip_base:
        t0 = time.perf_counter()
        try:
            base_fit = _load_or_fit_baseline(flights, rig, seed, out, reuse_base)
        except ImportError as exc:
            print(f"  [{rig}] baseline unavailable ({exc}); pass --skip-base to silence")
        else:
            _write(out / "fits" / "base" / f"{rig}.json", base_fit.to_json())
            base_stats = stats_from_samples(
                base_fit.sampler(fs), durations, fs=fs, n_rep=n_rep, seed=seed
            )
            _write(out / "stats" / "base" / f"{rig}.json", base_stats.to_json())
            record["base"] = discrepancy(base_stats, real)
            record["base_wall_s"] = time.perf_counter() - t0
            record["base_retention"] = airborne_retention(
                base_fit.sampler(fs), durations, fs, n_rep, seed
            )

    print(
        f"\n[{rig}] {len(flights)} flights, {record['airborne_s']:.0f} s airborne, "
        f"fit rate {new_fit.fit_rate_hz:g} Hz, {new_fit.n_blocks} blocks, "
        f"{new_fit.n_scored} scored samples\n"
        f"  new fit: NLL {new_fit.nll:.1f} in {new_fit.n_iter} iters, "
        f"{new_fit.wall_s:.1f} s wall ({new_fit.n_restarts} restarts)\n"
        + _params_line(new_fit)
        + f"\n    airborne-rule retention: new {record['new_retention']:.1%}"
        + (f", base {record['base_retention']:.1%}" if "base_retention" in record else "")
    )
    if base_stats is not None:
        record["verdict"] = report_rig(rig, real, base_stats, new_stats)
    else:
        print(
            f"  {rig}: new-vs-real discrepancy "
            + "  ".join(f"{f}={record['new'][f]:.5f}" for f in FAMILIES)
        )
        record["verdict"] = None
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--rigs", default=",".join(RIGS), help="comma-separated rig ids")
    ap.add_argument("--out", type=Path, default=Path("results/rps_traj"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--n-rep",
        type=int,
        default=40,
        help="sampler repetitions per duration, for BOTH models (round 2 default)",
    )
    ap.add_argument("--n-restarts", type=int, default=4, help="L-BFGS-B restarts per rig")
    ap.add_argument(
        "--warm-from",
        type=Path,
        default=Path("results/rps_traj/rounds/round3.json"),
        help="previous round snapshot to warm-start restart 0 from",
    )
    ap.add_argument("--skip-base", action="store_true", help="fit the new model only")
    ap.add_argument(
        "--reuse-base",
        action="store_true",
        help="load fits/base/<rig>.json instead of refitting the baseline",
    )
    ap.add_argument("--round", help="also snapshot the summary to rounds/<name>.json")
    ap.add_argument(
        "--rescore",
        action="store_true",
        help="re-score from the existing fits (offset model only; no refit)",
    )
    args = ap.parse_args()

    rigs = [r.strip() for r in str(args.rigs).split(",") if r.strip()]
    out = Path(args.out)
    records = [
        rescore_rig(rig, out, seed=args.seed, n_rep=args.n_rep)
        if args.rescore
        else run_rig(
            rig,
            out,
            seed=args.seed,
            n_rep=args.n_rep,
            n_restarts=args.n_restarts,
            skip_base=args.skip_base,
            reuse_base=args.reuse_base,
            warm=args.warm_from,
        )
        for rig in rigs
    ]

    verdicts = {r["rig"]: r["verdict"] for r in records}
    scored = {rig: v for rig, v in verdicts.items() if v is not None}
    overall = bool(scored) and all(scored.values())
    if scored:
        print(
            f"\noverall: {'PASS' if overall else 'FAIL'} "
            f"({sum(scored.values())}/{len(scored)} rigs pass: "
            + ", ".join(f"{rig}={'PASS' if v else 'FAIL'}" for rig, v in scored.items())
            + ")"
        )
    else:
        print("\noverall: NOT SCORED (no baseline statistics; --skip-base)")
    summary = json.dumps(
        {
            "rate_hz": RATE_HZ,
            "seed": args.seed,
            "n_rep": args.n_rep,
            "rescored": bool(args.rescore),
            "overall": overall if scored else None,
            "rigs": records,
        },
        indent=2,
        allow_nan=False,
    )
    _write(out / "summary.json", summary)
    print(f"wrote {out / 'summary.json'}")
    if args.round:
        snapshot = _write(out / "rounds" / f"{args.round}.json", summary)
        print(f"wrote {snapshot}")
        # ALSO one record per rig.  The campaign's expensive rounds run as one
        # job per rig, and seven jobs would each write their own single-rig
        # summary.json over the last one; per-rig records let
        # scripts/rps_traj_merge_round.py rebuild the combined summary exactly,
        # including the provenance (NLL, iterations, wall time, retention) that
        # only the job that did the fit knows.
        for record in records:
            part = _write(
                out / "rounds" / args.round / f"{record['rig']}.json",
                json.dumps(record, indent=2, allow_nan=False),
            )
            print(f"wrote {part}")
    # ALWAYS 0: this script's job is to FIT and RECORD, and it did.  A FAIL
    # verdict is a finding, not a failure, and returning 1 for it made every
    # honest single-rig job show up as "failed" in ``omnirun ps``.  The
    # PASS/FAIL exit code belongs to the judge, ``rps_traj_compare.py``.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
