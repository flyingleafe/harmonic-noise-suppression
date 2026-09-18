#!/usr/bin/env python
"""Stage-2 synthetic sweep: when are per-rotor profiles recoverable at all?

Runs :mod:`experiments.noise_model.multirotor` over a sweep in the minimum
carrier spacing ``delta`` and writes, into
``results/noise_v2/multirotor_synth/``:

* ``sweep.json`` -- every per-(rotor, order) cell of every configuration,
  seed and estimator: the dB profile error against the PLANTED truth, the
  line's own SNR, the loudest interferer's contrast, the realised beat cycles
  of the least-squares window, and the three candidate resolvability scores;
* ``summary.json`` -- the delta x estimator table and the failing-order ranges;
* ``findings.md`` -- written by hand from those two files;
* two figures.

THE TRUTH IS THE COMMITTED FIT. Every rotor's profile, carrier and dynamics
are read out of ``results/noise_v2/rounds/round1/fits/bench_dregon_Motor{r}_70
__bench.json`` -- so "can the estimator recover the profile" is literally "can
it recover the number the single-motor fit reports". Those five fits all carry
``optimiser.converged = false`` (``which_converged = "none"``), which is
recorded in the provenance block of both JSONs and quoted in the findings next
to every number taken from them.

Usage::

    PYTHONPATH=src python scripts/noise_v2_multirotor_synth.py            # full sweep
    PYTHONPATH=src python scripts/noise_v2_multirotor_synth.py --quick    # 1 seed, 3 configs
    PYTHONPATH=src python scripts/noise_v2_multirotor_synth.py --figures-only
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from experiments.noise_model import multirotor as MR

SCHEMA = "noise-v2-multirotor-synth/1"
FIT_DIR = Path("results/noise_v2/rounds/round1/fits")
OUT_DIR = Path("results/noise_v2/multirotor_synth")
RIG_FIT = FIT_DIR / "bench_dregon_allMotors_70__bench.json"

#: The refined four-rotor carriers of ``bench_dregon_allMotors_70`` (gaps
#: 3.02 / 1.08 / 0.83 Hz). Every configuration rescales the offsets from their
#: mean so the PATTERN is kept and only the minimum spacing moves.
BASE_CARRIERS = np.array([64.639557, 67.661824, 68.737936, 69.571794])

#: Orders scored. The comb is SYNTHESISED over every order below the Nyquist
#: (about 110); this ladder is where the estimators are run, chosen to cover
#: the decade in ``k`` at 15 points instead of 110.
LADDER = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 80, 96, 104])

#: A cell whose planted line is under this much SNR in its own analysis band
#: is not a test of resolvability -- the line is not in the recording. Three
#: tiers are reported; the figures and the headline use :data:`SNR_GATE_DB`.
SNR_TIERS_DB = (6.0, 20.0, 40.0)
SNR_GATE_DB = 20.0

#: Above this the recovered profile is wrong enough to break a rig fit: 3 dB
#: is a factor two in line power.
FAIL_DB = 3.0

SEEDS = (11, 12, 13)


@dataclass(frozen=True)
class Case:
    """One sweep configuration."""

    name: str
    delta_hz: float
    comb_to_floor_db: float = 80.0
    track_drift_std_hz: float = 0.24
    identical_steering: bool = False
    note: str = ""


#: The real comb-to-floor ratio is measured from the committed rig fit (see
#: :func:`rig_comb_to_floor_db`) and is about 4.5 dB TOTAL over the band. The
#: main sweep runs 80 dB instead, and deliberately: the fitted profiles span
#: 40 dB across orders, so at the real level most single orders sit under the
#: floor of a single 24 s support and an error there measures the floor, not
#: the separation. 80 dB puts most of the ladder over the 20 dB tier, which
#: makes the remaining failures resolvability failures. What the real level
#: costs on its own is the ``delta_0p83_real_snr`` case.
CASES: tuple[Case, ...] = (
    Case("delta_0p01", 0.01, note="below the record-length bound 1/(k T): E2's own floor"),
    Case("delta_0p05", 0.05, note="mandated sweep point"),
    Case("delta_0p20", 0.20, note="mandated sweep point"),
    Case("delta_0p83", 0.83, note="the real allMotors_70 minimum spacing"),
    Case("delta_3p00", 3.00, note="mandated sweep point"),
    Case("delta_0p83_identical_steering", 0.83, identical_steering=True, note="kills E2-multi"),
    Case("delta_0p83_drift_2x", 0.83, track_drift_std_hz=0.48, note="planted track drift doubled"),
    Case("delta_0p83_real_snr", 0.83, comb_to_floor_db=-1.0, note="measured rig comb-to-floor"),
)

ESTIMATORS = ("E1", "E2-single", "E2-multi", "E2-multi-oracle")


# ── provenance ──────────────────────────────────────────────────────────────


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # pragma: no cover - provenance only
        return "unknown"


def load_rotors() -> tuple[tuple[MR.RotorSpec, ...], list[dict[str, Any]]]:
    """The four single-motor 70 % fits, verbatim, plus their provenance."""
    specs: list[MR.RotorSpec] = []
    prov: list[dict[str, Any]] = []
    for r in range(1, 5):
        path = FIT_DIR / f"bench_dregon_Motor{r}_70__bench.json"
        fit = json.loads(path.read_text())
        specs.append(MR.rotor_from_fit(fit, name=f"Motor{r}"))
        p = fit["params"]
        prov.append(
            dict(
                rotor=f"Motor{r}",
                path=str(path),
                fitted_carrier_rev_s=float(np.atleast_1d(p["carrier_rev_s"])[0]),
                sigma_nu=float(p["sigma_nu"]),
                lam=float(p["lam"]),
                gamma_hz_ladder={
                    int(k): float(MR.MD.gamma_from_params(p)[0, int(k) - 1]) for k in LADDER
                },
                k_max=int(fit["k_max"]),
                n_orders=int(np.asarray(p["profile"]["profile_db"]).shape[1]),
                profile_db_ladder={
                    int(k): float(np.asarray(p["profile"]["profile_db"])[0, int(k) - 1])
                    for k in LADDER
                },
                converged=bool(fit["optimiser"].get("converged", False)),
                which_converged=fit["optimiser"].get("which_converged"),
                grad_norm=float(fit["optimiser"].get("grad_norm", float("nan"))),
                within_record_sigma_nu=float(MR.within_record_spec(specs[-1], 24.0).sigma_nu),
                git=fit.get("git"),
            )
        )
    return tuple(specs), prov


def load_floor() -> tuple[MR.FloorSpec, dict[str, Any]]:
    """The rig fit's floor SHAPE (its level is set by the comb-to-floor ratio)."""
    fit = json.loads(RIG_FIT.read_text())
    fl = fit["params"]["floor"]
    spec = MR.FloorSpec(
        shape_z=np.asarray(fl["floor_shape_z"], dtype=np.float64),
        tilt_db_oct=float(fl["floor_tilt_db_oct"]),
    )
    prov = dict(
        path=str(RIG_FIT),
        floor_tilt_db_oct=float(fl["floor_tilt_db_oct"]),
        floor_mean_db=float(fl["floor_mean_db"]),
        rig_carriers_rev_s=[float(v) for v in fit["params"]["carrier_rev_s"]],
        converged=bool(fit["optimiser"].get("converged", False)),
        which_converged=fit["optimiser"].get("which_converged"),
    )
    return spec, prov


def rig_comb_to_floor_db(sr: int = 16000, n: int = 384000) -> float:
    """In-band comb power over floor power of the committed rig fit, dB.

    Both sides in the units :func:`multirotor.simulate_scene` uses: the comb
    is ``sum_rk 10^(profile_db / 10)`` and the floor is
    ``(2 / N) sum_band psd``.
    """
    from experiments.noise_model import spectrum as SP
    from experiments.stochastic_fit.revised_phase import floor_geometry, floor_power_spectrum

    fit = json.loads(RIG_FIT.read_text())
    fl = fit["params"]["floor"]
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    shape_mat, tilt_oct = floor_geometry(freqs, SP.floor_ctrl_hz(sr))
    psd = floor_power_spectrum(
        shape_mat,
        tilt_oct,
        mean_db=float(fl["floor_mean_db"]),
        ctrl_db=SP.floor_shape_db(fl["floor_shape_z"], sr=sr),
        tilt_db_oct=float(fl["floor_tilt_db_oct"]),
        rate_factor=1.0,
    )
    band = SP.bench_band(freqs, sr)
    comb = float((10.0 ** (np.asarray(fit["params"]["profile"]["profile_db"]) / 10.0)).sum())
    floor = float(2.0 * np.asarray(psd)[band].sum() / n)
    return float(10.0 * math.log10(comb / floor))


# ── one run ─────────────────────────────────────────────────────────────────


def oracle_steering(scene: MR.MultiRotorScene) -> tuple[np.ndarray, np.ndarray]:
    """The planted mic-relative steering: the ceiling of the spatial lever.

    ``E2-multi`` has to estimate this from the data; comparing the two says
    whether a shortfall is the lever's or the estimate's.
    """
    st = scene.steering[:, :, 0]
    gain = np.abs(st) / np.abs(st[0])[None]
    gain = gain / np.sqrt((gain**2).mean(axis=0, keepdims=True))
    return gain, scene.delays_s - scene.delays_s[0][None]


def run_one(args: tuple[Case, int, float]) -> dict[str, Any]:
    """One (configuration, seed): generate, refine, run all four estimators."""
    case, seed, real_ratio_db = args
    specs, _ = load_rotors()
    floor, _ = load_floor()
    cfg = MR.make_config(
        specs,
        delta_hz=case.delta_hz,
        base_carriers=BASE_CARRIERS,
        n_mics=8,
        sr=16000,
        duration_s=24.0,
        track_drift_std_hz=case.track_drift_std_hz,
        comb_to_floor_db=(real_ratio_db if case.comb_to_floor_db < 0.0 else case.comb_to_floor_db),
        identical_steering=case.identical_steering,
        floor=floor,
        k_cap=110,
        seed=seed,
    )
    scene = MR.simulate_scene(cfg)
    refine = MR.refine_offsets(scene, np.array([4, 8, 12, 16, 24, 32, 40]))
    off = refine["offsets_rev_s"]
    truth = scene.power_true[:, LADDER - 1]

    results: dict[str, Any] = {}
    e1 = MR.estimate_e1(scene, LADDER, offsets_rev_s=off)
    e2s = MR.estimate_e2(scene, LADDER, offsets_rev_s=off)
    e2m = MR.estimate_e2(scene, LADDER, offsets_rev_s=off, multi=True)
    e2o = MR.estimate_e2(
        scene, LADDER, offsets_rev_s=off, multi=True, steering=oracle_steering(scene)
    )
    snr = MR.line_snr_db(scene, LADDER, e2s["bandwidth_hz"])
    contrast = MR.neighbour_contrast_db(scene, LADDER, e2s["bandwidth_hz"])
    for name, res in (("E1", e1), ("E2-single", e2s), ("E2-multi", e2m), ("E2-multi-oracle", e2o)):
        results[name] = np.round(MR.profile_error_db(res["power"], truth), 3).tolist()

    carriers = cfg.carriers_rev_s
    pair_delta = np.array(
        [
            min(abs(carriers[r] - carriers[s]) for s in range(cfg.n_rotors) if s != r)
            for r in range(cfg.n_rotors)
        ]
    )
    score_within = MR.per_rotor_scores(cfg.rotors, LADDER, duration_s=24.0)
    score_ens = MR.per_rotor_scores(cfg.rotors, LADDER, duration_s=24.0, within_record=False)
    score_rec = np.array(
        [
            [MR.record_resolvability_score(float(k), pair_delta[r], 24.0) for k in LADDER]
            for r in range(cfg.n_rotors)
        ]
    )
    return dict(
        case=case.name,
        seed=int(seed),
        delta_hz=float(case.delta_hz),
        carriers_rev_s=[float(v) for v in carriers],
        pair_delta_hz=[float(v) for v in pair_delta],
        comb_to_floor_db=float(cfg.comb_to_floor_db),
        track_drift_std_hz=float(case.track_drift_std_hz),
        identical_steering=bool(case.identical_steering),
        k_max=int(scene.k_max),
        error_db=results,
        snr_db=np.round(snr, 2).tolist(),
        contrast_db=np.round(np.where(np.isfinite(contrast), contrast, -999.0), 2).tolist(),
        beat_cycles=np.round(e2s["beat_cycles"], 4).tolist(),
        nearest_line_hz=np.round(
            np.where(np.isfinite(e2s["nearest_line_hz"]), e2s["nearest_line_hz"], -1.0), 4
        ).tolist(),
        window_s=np.round(e2s["window_s"], 5).tolist(),
        coherence_time_s=np.round(e2s["coherence_time_s"], 5).tolist(),
        n_columns=e2s["n_columns"].tolist(),
        score_within_record=np.round(score_within, 4).tolist(),
        score_ensemble=np.round(score_ens, 4).tolist(),
        score_record=np.round(score_rec, 4).tolist(),
        offset_spread_ratio=float(MR.offset_spread_ratio(cfg.rotors, delta_hz=case.delta_hz)),
        shaft_offset_true_rev_s=np.round(scene.shaft_offset_rev_s, 5).tolist(),
        shaft_offset_est_rev_s=np.round(off, 5).tolist(),
        offset_clipped=[bool(v) for v in refine["clipped"]],
        offset_room_rev_s=np.round(refine["room_rev_s"], 4).tolist(),
        steering_orders=[int(v) for v in np.atleast_1d(e2m["steering_orders"])],
        steering_delay_err_us=np.round(
            (e2m["steering_delay"] - oracle_steering(scene)[1]) * 1e6, 2
        ).tolist(),
        floor_band_power=float(scene.diagnostics["floor_band_power"]),
        comb_power=float(scene.diagnostics["comb_power"]),
    )


# ── aggregation ─────────────────────────────────────────────────────────────


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The delta x estimator table, at each SNR tier.

    THREE TIERS, because the fitted profiles span 40 dB and a cell 6 dB over
    the floor carries several dB of the estimator's own noise however well the
    rotors are separated. The 20 dB tier is where a residual error IS a
    separation error; the 6 dB tier shows what the whole ladder does.

    Collapsed cells (the estimate floored to zero, sentinel ``-200`` dB) are
    clipped to 60 dB so a percentile stays a number, and counted separately.
    """
    by_case: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_case.setdefault(rec["case"], []).append(rec)
    out: dict[str, Any] = {}
    for case, recs in by_case.items():
        snr = np.stack([np.asarray(r["snr_db"]) for r in recs])  # (S, R, K)
        entry: dict[str, Any] = dict(
            delta_hz=recs[0]["delta_hz"],
            carriers_rev_s=recs[0]["carriers_rev_s"],
            comb_to_floor_db=recs[0]["comb_to_floor_db"],
            track_drift_std_hz=recs[0]["track_drift_std_hz"],
            identical_steering=recs[0]["identical_steering"],
            n_seeds=len(recs),
            total_cells=int(snr.size),
            offset_spread_ratio=recs[0]["offset_spread_ratio"],
            offset_abs_err_rev_s=float(
                np.max(
                    [
                        np.abs(
                            np.asarray(r["shaft_offset_est_rev_s"])
                            - np.asarray(r["shaft_offset_true_rev_s"])
                        ).max()
                        for r in recs
                    ]
                )
            ),
            beat_cycles_min=float(np.min([np.min(r["beat_cycles"]) for r in recs])),
            beat_cycles_median=float(np.median([np.median(r["beat_cycles"]) for r in recs])),
            #: ladder cells whose nearest OTHER comb line is inside 0.1 Hz --
            #: an accidental collision of the dense 440-line comb, a separate
            #: failure mode from the order-k spacing itself
            collisions_under_0p1hz=int(
                sum(
                    int(
                        (
                            (np.asarray(r["nearest_line_hz"]) >= 0.0)
                            & (np.asarray(r["nearest_line_hz"]) < 0.1)
                        ).sum()
                    )
                    for r in recs
                )
            ),
            tiers={},
        )
        for tier in SNR_TIERS_DB:
            gate = snr >= tier
            tier_entry: dict[str, Any] = dict(gated_cells=int(gate.sum()), estimators={})
            for est in ESTIMATORS:
                raw = np.stack([np.asarray(r["error_db"][est]) for r in recs])
                err = np.minimum(np.abs(raw), 60.0)
                vals = err[gate]
                failing = []
                for q, k in enumerate(LADDER):
                    cell = err[:, :, q][gate[:, :, q]]
                    if cell.size and float(np.median(cell)) > FAIL_DB:
                        failing.append(int(k))
                tier_entry["estimators"][est] = dict(
                    median_abs_db=float(np.median(vals)) if vals.size else None,
                    p90_abs_db=float(np.percentile(vals, 90)) if vals.size else None,
                    max_abs_db=float(vals.max()) if vals.size else None,
                    failing_orders=failing,
                    frac_cells_over_3db=float((vals > FAIL_DB).mean()) if vals.size else None,
                    frac_collapsed=float((raw[gate] <= -199.0).mean()) if vals.size else None,
                )
            entry["tiers"][str(int(tier))] = tier_entry
        out[case] = entry
    return out


# ── figures ─────────────────────────────────────────────────────────────────


def figures(records: list[dict[str, Any]], out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_case: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_case.setdefault(rec["case"], []).append(rec)
    delta_cases = [c.name for c in CASES if c.name.startswith("delta_") and "_" not in c.name[6:]]
    delta_cases = [c for c in delta_cases if c in by_case]
    colours = {
        "E1": "#c1272d",
        "E2-single": "#0b6e99",
        "E2-multi": "#1b7837",
        "E2-multi-oracle": "#7a6ff0",
    }

    # ── figure 1: profile error vs order, one panel per delta ──
    n = len(delta_cases)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, case in zip(axes, delta_cases, strict=True):
        recs = by_case[case]
        gate = np.stack([np.asarray(r["snr_db"]) >= SNR_GATE_DB for r in recs])
        for est in ESTIMATORS:
            err = np.stack([np.abs(np.asarray(r["error_db"][est])) for r in recs])
            med = []
            for q in range(LADDER.size):
                cell = err[:, :, q][gate[:, :, q]]
                med.append(float(np.median(cell)) if cell.size else np.nan)
            ax.plot(LADDER, med, "o-", ms=3, lw=1.3, color=colours[est], label=est)
        ax.axhline(FAIL_DB, color="0.4", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel("order $k$")
        ax.set_title(f"$\\delta$ = {recs[0]['delta_hz']:g} Hz", fontsize=10)
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("median $|$profile error$|$  (dB)")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(
        "Per-rotor profile recovery vs order, 4 rotors / 8 mics / 24 s, SNR-gated cells only",
        fontsize=10,
    )
    fig.tight_layout()
    p1 = out_dir / "fig_profile_error.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # ── figure 2: the criterion ──
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for case in delta_cases:
        recs = by_case[case]
        sc = np.asarray(recs[0]["score_within_record"]).min(axis=0)
        axa.plot(LADDER, sc, "o-", ms=3, lw=1.3, label=f"$\\delta$={recs[0]['delta_hz']:g} Hz")
    axa.axhline(1.0, color="k", ls="--", lw=1)
    axa.set_xscale("log")
    axa.set_yscale("log")
    axa.set_xlabel("order $k$")
    axa.set_ylabel(r"$k\,\delta\,\min(\tau_c(k), T)$")
    axa.set_title("the criterion is flat in $k$", fontsize=10)
    axa.grid(alpha=0.25, which="both")
    axa.legend(fontsize=7)

    bins = np.array([5e-3, 2e-2, 1e-1, 3e-1, 1.0, 3.0])
    centres = np.sqrt(bins[:-1] * bins[1:])
    for est in ESTIMATORS:
        xs, ys = [], []
        for case in delta_cases:
            for rec in by_case[case]:
                gate = np.asarray(rec["snr_db"]) >= SNR_GATE_DB
                cyc = np.asarray(rec["beat_cycles"])
                err = np.minimum(np.abs(np.asarray(rec["error_db"][est])), 60.0)
                sel = gate & np.isfinite(cyc)
                xs.append(cyc[sel])
                ys.append(np.maximum(err[sel], 1e-2))
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        med = np.array(
            [
                np.median(y[(x >= a) & (x < b)]) if ((x >= a) & (x < b)).any() else np.nan
                for a, b in zip(bins[:-1], bins[1:], strict=True)
            ]
        )
        axb.plot(centres, med, "o-", ms=4, lw=1.4, color=colours[est], label=est)
        axb.scatter(x, y, s=4, alpha=0.12, color=colours[est], edgecolors="none")
    axb.axvline(1.0, color="k", ls="--", lw=1)
    axb.axhline(FAIL_DB, color="0.4", ls=":", lw=1)
    axb.set_xscale("log")
    axb.set_yscale("log")
    axb.set_xlabel("realised beat cycles in the LS window  $\\delta_{near} T_w$")
    axb.set_ylabel("$|$profile error$|$  (dB)")
    axb.set_title("error vs the observed beat", fontsize=10)
    axb.grid(alpha=0.25, which="both")
    axb.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    p2 = out_dir / "fig_criterion.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    return [str(p1), str(p2)]


# ── main ────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="one seed, three configurations")
    ap.add_argument("--figures-only", action="store_true", help="re-read sweep.json and re-plot")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sweep_path = args.out / "sweep.json"

    if args.figures_only:
        payload = json.loads(sweep_path.read_text())
        paths = figures(payload["records"], args.out)
        print("figures:", paths)
        return

    specs, rotor_prov = load_rotors()
    _floor, floor_prov = load_floor()
    real_ratio = rig_comb_to_floor_db()
    cases = CASES[:4] if args.quick else CASES
    seeds = SEEDS[:1] if args.quick else SEEDS
    jobs = [(case, seed, real_ratio) for case in cases for seed in seeds]
    print(f"{len(jobs)} runs, {args.jobs} workers, real comb-to-floor {real_ratio:.2f} dB")

    records: list[dict[str, Any]] = []
    if args.jobs <= 1:
        for job in jobs:
            records.append(run_one(job))
            print("  done", records[-1]["case"], records[-1]["seed"], flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for rec in pool.map(run_one, jobs):
                records.append(rec)
                print("  done", rec["case"], rec["seed"], flush=True)

    payload = dict(
        schema=SCHEMA,
        git=_git_sha(),
        ladder=[int(k) for k in LADDER],
        snr_gate_db=SNR_GATE_DB,
        fail_db=FAIL_DB,
        seeds=[int(s) for s in seeds],
        sr=16000,
        duration_s=24.0,
        n_mics=8,
        n_rotors=4,
        base_carriers_rev_s=[float(v) for v in BASE_CARRIERS],
        real_comb_to_floor_db=real_ratio,
        cases=[asdict(replace(c)) for c in cases],
        provenance=dict(
            rotor_fits=rotor_prov,
            floor_fit=floor_prov,
            note=(
                "every fit read here reports optimiser.converged = false "
                "(which_converged = 'none'); the planted truth is therefore the "
                "fit's reported MAP, not a converged optimum"
            ),
        ),
        records=records,
    )
    sweep_path.write_text(json.dumps(payload, indent=1))
    summary = dict(
        schema=SCHEMA,
        git=payload["git"],
        snr_gate_db=SNR_GATE_DB,
        fail_db=FAIL_DB,
        ladder=payload["ladder"],
        real_comb_to_floor_db=real_ratio,
        provenance=payload["provenance"],
        cases=summarise(records),
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    paths = figures(records, args.out)
    print("wrote", sweep_path, args.out / "summary.json", *paths)


if __name__ == "__main__":
    main()
