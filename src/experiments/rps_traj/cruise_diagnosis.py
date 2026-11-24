"""Replay only the two cruise arms; attribute the published score, without a refit of cruise dynamics.

Run: python -m experiments.rps_traj.cruise_diagnosis [--out results/rps_traj/regimes]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

from experiments.rps_traj import phase_dynamics as pdyn
from experiments.rps_traj import schedule_eval as se
from experiments.rps_traj import spectral_validation as sv

ARMS = ("ours", "phasevar2+hsmm3")
BANDS = [f"{lo:g}–{hi:g} Hz" for lo, hi in zip(sv.EDGES, sv.EDGES[1:])]
BLOCKS = ["mean_speed", *(f"{band} {kind}" for band in BANDS for kind in ("auto", "cross"))]


def breakdown(metric, obs, draws, refs, hover):
    """Split each Euclidean norm into ||delta_block||²/||delta||, then apply fair ES.

    This allocates the existing score exactly; it is not a set of new scoring rules.
    Every pairwise term uses the joint norm, not a block-specific norm.
    """
    x = np.concatenate((obs[None], draws))
    s = sv.spectra(x)
    f = sv.embed(s, metric.band_power) / metric.spectrum_scale
    level = x.mean(axis=-1) / (2 * metric.level_scale_rps)
    blocks = [level]
    powers = []
    for b in range(len(BANDS)):
        band = sv._octave(b)
        blocks.extend([f[:, band, :4].reshape(len(x), -1), f[:, band, 4:].reshape(len(x), -1)])
        # Mean per-motor band power, in (rev/s)^2, integrated on the scoring grid.
        powers.append(
            s[:, band].diagonal(axis1=2, axis2=3).real.mean(axis=2).sum(axis=1) * sv.FS / sv.N
        )
    sq_target = np.stack([np.square(v[1:] - v[0]).sum(axis=1) for v in blocks], axis=1)
    sq_pairs = np.stack([pdist(v[1:], metric="sqeuclidean") for v in blocks], axis=1)

    def allocate(sq):
        norm = np.sqrt(sq.sum(axis=1, keepdims=True))
        return np.divide(sq, norm, out=np.zeros_like(sq), where=norm > 0).mean(axis=0)

    scale = max(
        float(np.median(pdist(metric.transform(refs)))), 0.01 * hover / metric.level_scale_rps
    )
    fit, spread = allocate(sq_target) / scale, 0.5 * allocate(sq_pairs) / scale
    powers = np.stack(powers, axis=1)
    return {
        "score": float((fit - spread).sum()),
        "parts": (fit - spread).tolist(),
        "distance_to_real": float(fit.sum()),
        "spread_credit": float(spread.sum()),
        "power_real": powers[0].tolist(),
        "power_draw_median": np.median(powers[1:], axis=0).tolist(),
    }


def replay(out):
    dest = out / "schedule_eval"
    report = json.loads((dest / "schedule_models.json").read_text())
    saved = {(r["rig"], r["flight"], r["arm"]): r for r in report["rows"] if r["phase"] == "cruise"}
    labels, times, examples, rows = [], [], [], []
    for rig, cal in report["meta"]["metric"]["rigs"].items():
        metric = sv.CruiseMetric(
            np.array(cal["band_power"]), cal["level_scale_rps"], cal["spectrum_scale"]
        )
        plans, _ = se.plan(rig, out, require_caches=True)
        for fp in plans:
            for target in fp.targets:
                (g,) = [g for g in fp.groups if g.flight == target.name and g.phase == "cruise"]
                key = rig, target.name
                labels.append(key)
                times.append(np.array(g.starts) / se.FS)
                waves = [np.stack(g.obs)]
                for arm in ARMS:
                    reference = saved[*key, arm]
                    generate = pdyn.fit_arm(arm, rig, fp.fold, str(out), 0, same_clock=True)
                    rng = np.random.default_rng(np.random.SeedSequence(reference["seed"]))
                    paths = generate(target.marks, reference["n_draws"], rng)
                    if paths is None or len(paths) != reference["n_draws"]:
                        raise ValueError(f"{key} {arm}: incomplete generated ensemble")
                    draws, status, _ = se._cut_generated(paths, target, g)
                    if draws is None or status != "ok":
                        raise ValueError((key, arm, status))
                    origins = [
                        breakdown(metric, o, d, r, g.hover)
                        for o, d, r in zip(g.obs, draws, g.refs, strict=True)
                    ]
                    np.testing.assert_allclose(
                        [o["score"] for o in origins],
                        reference["diagnostics"]["score_per_origin"],
                        rtol=1e-9,
                        atol=1e-10,
                    )
                    waves.append(np.stack([d[0] for d in draws]))
                    rows.append(
                        {
                            "rig": rig,
                            "flight": target.name,
                            "arm": arm,
                            "score": reference["score"],
                            "level_es_rps": reference["diagnostics"]["level_es_rps"],
                            "origins": origins,
                        }
                    )
                examples.append(np.stack(waves))  # source, origin, rotor, sample
    np.savez_compressed(
        dest / "cruise_draw0.npz",
        labels=np.array(labels),
        starts_s=np.array(times),
        waves=np.stack(examples),
        fs=se.FS,
    )
    return rows


def summarize(rows):
    rigs = sorted({r["rig"] for r in rows})
    summary, gaps = {}, []
    for rig in rigs:
        by_arm = {}
        for arm in ARMS:
            records = sorted(
                [r for r in rows if (r["rig"], r["arm"]) == (rig, arm)], key=lambda r: r["flight"]
            )
            orgs = [o for r in records for o in r["origins"]]
            ratio = np.array([np.array(o["power_draw_median"]) / o["power_real"] for o in orgs])
            by_arm[arm] = {
                "score": float(np.mean([r["score"] for r in records])),
                "level_es_rps": float(np.mean([r["level_es_rps"] for r in records])),
                "parts": np.mean([o["parts"] for o in orgs], axis=0).tolist(),
                "power_ratio_median": np.median(ratio, axis=0).tolist(),
                **{
                    k: float(np.mean([o[k] for o in orgs]))
                    for k in ("distance_to_real", "spread_credit")
                },
            }
        o, h = ARMS
        paired = {
            f: next(r["score"] for r in rows if (r["rig"], r["flight"], r["arm"]) == (rig, f, o))
            - next(r["score"] for r in rows if (r["rig"], r["flight"], r["arm"]) == (rig, f, h))
            for f in sorted({r["flight"] for r in rows if r["rig"] == rig})
        }
        gaps.append(np.array(list(paired.values())))
        summary[rig] = {"arms": by_arm, "gap_by_flight": paired}
    parts = np.mean(
        [
            np.subtract(s["arms"][ARMS[0]]["parts"], s["arms"][ARMS[1]]["parts"])
            for s in summary.values()
        ],
        axis=0,
    )
    rng = np.random.default_rng(2026)
    boot = np.mean(
        [rng.choice(v, (10000, len(v)), replace=True).mean(axis=1) for v in gaps], axis=0
    )
    gap = float(np.mean([v.mean() for v in gaps]))
    np.testing.assert_allclose(gap, parts.sum(), atol=1e-10)
    return {
        "rigs": summary,
        "gap_ours_minus_hybrid": gap,
        "conditional_paired_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "parts_of_gap": dict(zip(BLOCKS, parts.tolist())),
    }


def main():
    from plots.cruise_sections import render

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/rps_traj/regimes"))
    out = ap.parse_args().out
    rows = replay(out)
    result = summarize(rows)
    (out / "schedule_eval" / "cruise_diagnosis.json").write_text(
        json.dumps({"blocks": BLOCKS, "summary": result, "rows": rows}, indent=1, allow_nan=False)
        + "\n"
    )
    render(out / "schedule_eval")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
