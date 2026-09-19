#!/usr/bin/env python
"""Stage 2/3 of the four-motor transfer: the multi-rotor estimators on the REAL
``bench_dregon_allMotors_70`` support, and the rig fit that has to reproduce the
four per-rotor fits.

STAGE 2 (``estimate``). The synthetic study
(``results/noise_v2/multirotor_synth/findings.md``) measured all three
estimators against a KNOWN per-rotor profile at the real carrier pattern. Here
the same three run on the real recording, where the truth is unknown: the
comparison target is the four committed single-motor fits
``bench_dregon_Motor{1..4}_70__bench.json``, which are a DIFFERENT recording of
the same motors (one motor at a time, different mounting/airflow, no
neighbours). They are the stage-3 target, not ground truth.

The scene the estimators take is assembled from the support itself:

* the audio is the support's own material -- the recording cut to the index's
  window and decimated by :func:`supports.bench_support`'s route, checked
  bit-for-bit against the cached periodogram (``audio_vs_support_max_rel``);
* the tracks are the bench convention, one CONSTANT per rotor, at the index's
  window-refined carriers, so the phase-aware lever here is the per-line LS and
  the steering, never track demodulation (a bench track has no drift to remove);
* the per-rotor dynamics (``sigma_nu``, ``lam``, ``gamma_hz``) come from the
  four single-motor R3 fits. They set analysis BANDWIDTHS, LS window lengths and
  the resolvability score only -- ``profile_db`` is never read by an estimator,
  so no truth leaks into the estimate.

THE UNITS. An estimator returns the mic-averaged line power ``P_rk`` of the
audio. The model's ``profile_db`` is the same power BEFORE the render transfer
and the mic gains (:func:`spectrum.bench_model`: the mic-``m`` comb is
``line_gain[m, r] * gain_all[m] * transfer(f)`` times the per-rotor line), so

    profile_db_est = 10 log10 P_rk - 10 log10 T(k f_r)
                     - 10 log10 mean_m (line_gain[m, r] gain_all[m])

with both gain blocks mean-pinned in dB exactly as the model pins them and read
off the four-motor rig fit -- the frame a profile INIT for that fit has to be
in. In band the whole correction is 0.2-0.9 dB.

THE CARRIER REFINEMENT. :func:`multirotor.refine_offsets` at its default
``bound_rev_s = 0.5`` locks two of the four rotors onto the wrong comb (see the
findings); the bound is a search room, not a measurement, so it is tightened to
``--offset-bound``, which the support index's own half-window drift justifies.
Both are reported.

STAGE 3 (``compare``). Reads the arm fits and scores each against the four
per-rotor fits: per rotor the median ``|profile_db(fit) - profile_db(Motor r)|``
over in-band orders ``k <= 48`` and the fraction of them inside 3 dB, plus the
dynamics ladder against the per-rotor log-mean.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.noise_model import model as MD  # noqa: E402
from experiments.noise_model import multirotor as MR  # noqa: E402
from experiments.noise_model import spectrum as SP  # noqa: E402
from experiments.noise_model import supports as SU  # noqa: E402
from experiments.stochastic_fit import clips as C  # noqa: E402
from experiments.stochastic_fit.phase_kernel import hann_window  # noqa: E402
from experiments.stochastic_fit.stage2 import render_transfer_power  # noqa: E402

SCHEMA = "noise-v2-fourmotor/1"
OUT_DIR = Path("results/noise_v2/rounds/round3/fourmotor")
FIT_DIR = Path("results/noise_v2/rounds/round3/fits")
SUPPORT = "bench_dregon_motor:allMotors:70"
SUPPORT_NAME = "bench_dregon_allMotors_70"
SUPPORT_CACHE = Path("results/noise_v2/rounds/round2/supports")
RIG_FIT = FIT_DIR / f"{SUPPORT_NAME}__bench.json"
ROTOR_FITS = tuple(FIT_DIR / f"bench_dregon_Motor{r}_70__bench.json" for r in (1, 2, 3, 4))

#: The band the likelihood sees (``spectrum.bench_band`` at 16 kHz). Every
#: comparison is restricted to orders whose line is inside it
#: (``multirotor_synth/findings.md`` section 5).
BAND_TOP_HZ = min(SP.BAND_F_MAX, 0.45 * 16000.0)

#: The per-order prior widths the synthetic study measured, as the stage-3
#: recommendation states them: 1 dB to k = 16, 3 dB to k = 48, the model's own
#: two-regime prior above (entered as NaN, i.e. "leave the default alone").
SIGMA_TIERS: tuple[tuple[int, float], ...] = ((16, 1.0), (48, 3.0))

#: The three order bands every comparison is cut into.
BANDS: tuple[tuple[str, int, int], ...] = (
    ("k<=16", 1, 16),
    ("k17-48", 17, 48),
    ("k49+", 49, 10**6),
)

#: The orders :func:`multirotor.refine_offsets` is given, and the alternative
#: ladders the offset's stability is measured over.
OFFSET_ORDERS = np.array([4, 8, 12, 16, 24, 32, 40])
OFFSET_LADDERS: dict[str, np.ndarray] = {
    "k4-40 (campaign)": OFFSET_ORDERS,
    "k4-48": np.arange(4, 49, 4),
    "k8-96": np.arange(8, 97, 8),
    "k20-100": np.arange(20, 101, 10),
}

#: The stage-3 criterion. "Harmonic profiles largely restored" means both.
CRIT_MEDIAN_DB = 3.0
CRIT_FRACTION = 0.70
CRIT_K_MAX = 48

GAMMA_LADDER = (1, 2, 4, 8, 16)


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # pragma: no cover - provenance only
        return "unknown"


# ── the scene ───────────────────────────────────────────────────────────────


def support_audio(support: SU.Support) -> tuple[np.ndarray, float]:
    """The support's OWN audio, plus the max relative periodogram mismatch.

    :func:`supports.bench_support` keeps only the periodogram, so the audio is
    rebuilt by the same route (cut the index window out of the native-rate
    recording, decimate to 16 kHz) and then checked against the cached power.
    """
    rec = C.load_recording(
        SU.DREGON_DATASET, str(support.meta["recording_id"]), None, SU.DREGON_RPS_KEY
    )
    sr_native = float(rec.sr)
    i0 = int(round(support.segment[0] * sr_native))
    i1 = int(round(support.segment[1] * sr_native))
    clip = SU._bench_clip(
        np.asarray(rec.audio)[:, i0:i1],
        sr_native,
        [float(v) for v in support.carrier_rev_s[:, 0]],
        clip_id=support.name,
        group="bench",
    )
    clip = C.decimate(clip, SU.SR)
    audio = np.asarray(clip.audio, dtype=np.float64)
    w = hann_window(audio.shape[1])
    power = (np.abs(np.fft.rfft(audio * w[None, :], axis=1)) ** 2) / float((w**2).sum())
    ref = np.asarray(support.power[:, 0, :], dtype=np.float64)
    rel = float(np.max(np.abs(power - ref) / np.maximum(ref, 1e-30)))
    return audio, rel


def rotor_specs(carriers: np.ndarray) -> tuple[MR.RotorSpec, ...]:
    """The four single-motor R3 fits' dynamics ON the four-motor carriers."""
    out = []
    for i, path in enumerate(ROTOR_FITS):
        fit = json.loads(Path(path).read_text())
        spec = MR.rotor_from_fit(fit, name=Path(path).stem)
        out.append(
            MR.RotorSpec(
                name=f"Motor{i + 1}",
                carrier_rev_s=float(carriers[i]),
                profile_db=spec.profile_db,
                sigma_nu=spec.sigma_nu,
                lam=spec.lam,
                gamma_hz=spec.gamma_hz,
            )
        )
    return tuple(out)


def real_scene(support: SU.Support, audio: np.ndarray, k_max: int) -> MR.MultiRotorScene:
    """A :class:`multirotor.MultiRotorScene` whose audio is a RECORDING.

    Everything the estimators read is real (audio, carriers, tracks, dynamics);
    the planted fields a synthetic scene carries (``power_true``, ``steering``,
    ``delays_s``, ``shaft_offset_rev_s``) are left empty because there is no
    truth, and no estimator touches them -- only ``diagnostics.delay_bound_s``,
    the array's own delay bound, is read (by the steering matched filter).
    """
    carriers = np.asarray(support.carrier_rev_s[:, 0], dtype=np.float64)
    n = audio.shape[1]
    cfg = MR.SceneConfig(
        rotors=rotor_specs(carriers),
        n_mics=int(audio.shape[0]),
        sr=int(support.sr),
        duration_s=n / float(support.sr),
        k_cap=int(k_max),
    )
    t = np.arange(n, dtype=np.float64) / float(support.sr)
    return MR.MultiRotorScene(
        cfg=cfg,
        audio=audio,
        track_rev_s=np.repeat(carriers[:, None], n, axis=1),
        track_turns=carriers[:, None] * t[None, :],
        power_true=np.zeros((carriers.size, k_max)),
        steering=np.zeros((audio.shape[0], carriers.size, k_max), dtype=complex),
        delays_s=np.zeros((audio.shape[0], carriers.size)),
        shaft_offset_rev_s=np.zeros(carriers.size),
        k_max=int(k_max),
        diagnostics=dict(delay_bound_s=float(cfg.array_aperture_m) / MR.SPEED_OF_SOUND),
    )


# ── units ───────────────────────────────────────────────────────────────────


def calibration_db(carriers: np.ndarray, k_max: int, rig: dict[str, Any]) -> np.ndarray:
    """``(R, K)`` dB to SUBTRACT from a measured line power to get ``profile_db``.

    The model's mic-``m`` comb is ``line_gain[m, r] gain_all[m] transfer(f)``
    times the per-rotor line power, both gain blocks mean-pinned in dB
    (:func:`spectrum._mean_pinned_db`), and an estimator reports the arithmetic
    mic MEAN of the line power -- so the factor is the mic mean of the product,
    times the transfer at the line.
    """
    p = rig["params"]
    mlg = np.asarray(p["profile"]["mic_line_gain_db"], dtype=np.float64)
    mg = np.asarray(p["mic_gains_db"], dtype=np.float64)
    g_line = 10.0 ** ((mlg - mlg.mean(axis=0, keepdims=True)) / 10.0)
    g_all = 10.0 ** ((mg - mg.mean()) / 10.0)
    mic_db = 10.0 * np.log10((g_line * g_all[:, None]).mean(axis=0))  # (R,)
    k = np.arange(1, int(k_max) + 1, dtype=np.float64)
    lines = k[None, :] * np.asarray(carriers, dtype=np.float64)[:, None]
    transfer = np.asarray(
        render_transfer_power(
            lines.ravel(), sample_rate_work=SP.SAMPLE_RATE_WORK, sample_rate_out=SU.SR
        ),
        dtype=np.float64,
    ).reshape(lines.shape)
    return 10.0 * np.log10(np.maximum(transfer, 1e-30)) + mic_db[:, None]


def to_profile_db(power: np.ndarray, cal_db: np.ndarray, floor_db: float = -200.0) -> np.ndarray:
    """A measured ``(R, K)`` line power as the model's ``profile_db``."""
    with np.errstate(divide="ignore"):
        db = 10.0 * np.log10(np.maximum(np.asarray(power, dtype=np.float64), 0.0))
    return np.where(np.isfinite(db), db - cal_db, floor_db)


def in_band_mask(carriers: np.ndarray, k_max: int) -> np.ndarray:
    k = np.arange(1, int(k_max) + 1, dtype=np.float64)
    return k[None, :] * np.asarray(carriers, dtype=np.float64)[:, None] <= BAND_TOP_HZ


# ── comparison ──────────────────────────────────────────────────────────────


def band_stats(delta: np.ndarray, ok: np.ndarray) -> list[dict[str, Any]]:
    """Median ``|delta|``, and the fraction inside 3 dB, per :data:`BANDS`."""
    k = np.arange(1, delta.shape[1] + 1)
    out = []
    for name, lo, hi in BANDS:
        sel = ok & (k[None, :] >= lo) & (k[None, :] <= hi)
        rows = []
        for r in range(delta.shape[0]):
            d = np.abs(delta[r][sel[r]])
            rows.append(
                dict(
                    rotor=r + 1,
                    n=int(d.size),
                    median_abs_db=float(np.median(d)) if d.size else None,
                    frac_within_3db=float(np.mean(d <= 3.0)) if d.size else None,
                )
            )
        out.append(dict(band=name, per_rotor=rows))
    return out


def rotor_profiles(k_max: int) -> np.ndarray:
    """``(4, k_max)`` ``profile_db`` of the four single-motor R3 fits."""
    rows = []
    for path in ROTOR_FITS:
        fit = json.loads(Path(path).read_text())
        prof = np.asarray(fit["params"]["profile"]["profile_db"], dtype=np.float64)[0]
        row = np.full(k_max, np.nan)
        m = min(k_max, prof.size)
        row[:m] = prof[:m]
        rows.append(row)
    return np.stack(rows)


def rotor_dynamics() -> dict[str, Any]:
    """The four per-rotor fits' dynamics, and the log-mean combination."""
    sig, lam, gam = [], [], []
    for path in ROTOR_FITS:
        fit = json.loads(Path(path).read_text())
        p = fit["params"]
        sig.append(float(p["sigma_nu"]))
        lam.append(float(p["lam"]))
        gam.append(MD.gamma_from_params(p)[0])
    width = min(g.size for g in gam)
    g = np.stack([x[:width] for x in gam])
    return dict(
        sigma_nu=sig,
        lam=lam,
        sigma_nu_log_mean=float(np.exp(np.mean(np.log(sig)))),
        lam_log_mean=float(np.exp(np.mean(np.log(lam)))),
        gamma_log_mean_per_k={
            str(k): float(np.exp(np.mean(np.log(np.maximum(g[:, k - 1], 1e-12)))))
            for k in GAMMA_LADDER
        },
        gamma_log_mean_all=float(np.exp(np.mean(np.log(np.maximum(g, 1e-12))))),
    )


# ── stage 2 ─────────────────────────────────────────────────────────────────


def offsets_block(scene: MR.MultiRotorScene, bound: float) -> dict[str, Any]:
    """The carrier refinement at the default bound, at ``bound``, and its
    stability over four order ladders."""

    def one(orders: np.ndarray, b: float) -> dict[str, Any]:
        res = MR.refine_offsets(scene, orders, bound_rev_s=b)
        return {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in res.items()
            if k != "bound_rev_s"
        } | dict(bound_rev_s=float(b), orders=[int(v) for v in orders])

    default = one(OFFSET_ORDERS, 0.5)
    tight = {name: one(orders, bound) for name, orders in OFFSET_LADDERS.items()}
    stack = np.stack([np.asarray(v["offsets_rev_s"]) for v in tight.values()])
    chosen = np.asarray(tight["k4-40 (campaign)"]["offsets_rev_s"], dtype=np.float64)
    return dict(
        default_bound=default,
        tight=tight,
        chosen_rev_s=chosen.tolist(),
        ladder_spread_rev_s=stack.std(axis=0).tolist(),
        ladder_min_rev_s=stack.min(axis=0).tolist(),
        ladder_max_rev_s=stack.max(axis=0).tolist(),
    )


def estimate(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    support = SU.load_support(SUPPORT, cache_dir=SUPPORT_CACHE)
    audio, audio_rel = support_audio(support)
    carriers = np.asarray(support.carrier_rev_s[:, 0], dtype=np.float64)
    rig = json.loads(RIG_FIT.read_text())
    k_max = int(args.k_max or SP.k_max_for_carrier(carriers, int(support.sr), k_cap=130))
    scene = real_scene(support, audio, k_max)
    orders = np.arange(1, k_max + 1, dtype=np.int64)
    duration_s = scene.cfg.duration_s

    print(f"# support {support.name}: {duration_s:.2f} s, {audio.shape[0]} mics, k_max {k_max}")
    print(f"# audio vs cached periodogram, max relative error {audio_rel:.2e}", flush=True)

    off = offsets_block(scene, float(args.offset_bound))
    chosen = np.asarray(off["chosen_rev_s"], dtype=np.float64)
    bin_hz = float(support.sr) / audio.shape[1]
    print(f"# offsets rev/s {np.round(chosen, 5).tolist()} ({np.round(chosen / bin_hz, 2)} bins)")

    e1 = MR.estimate_e1(scene, orders, offsets_rev_s=chosen)
    print("# E1 done", flush=True)
    e2s = MR.estimate_e2(scene, orders, offsets_rev_s=chosen)
    print("# E2-single done", flush=True)
    e2m = MR.estimate_e2(scene, orders, offsets_rev_s=chosen, multi=True)
    print("# E2-multi done", flush=True)
    # the no-refinement control: how much the (uncertain) offsets are worth
    e2m0 = MR.estimate_e2(scene, orders, multi=True)
    print("# E2-multi (no offsets) done", flush=True)

    cal = calibration_db(carriers, k_max, rig)
    ok = in_band_mask(carriers, k_max)
    target = rotor_profiles(k_max)
    est = {
        "E1": to_profile_db(e1["power"], cal),
        "E2-single": to_profile_db(e2s["power"], cal),
        "E2-multi": to_profile_db(e2m["power"], cal),
        "E2-multi-no-offset": to_profile_db(e2m0["power"], cal),
    }
    rig_prof = np.asarray(rig["params"]["profile"]["profile_db"], dtype=np.float64)[:, :k_max]
    est["R3-rig-fit"] = rig_prof

    scores = MR.per_rotor_scores(scene.cfg.rotors, orders, duration_s=duration_s)
    comparison = {name: band_stats(v - target, ok & np.isfinite(target)) for name, v in est.items()}
    collapsed = {
        name: int(np.sum((v <= -199.0) & ok)) for name, v in est.items() if name != "R3-rig-fit"
    }

    payload = dict(
        schema=SCHEMA,
        stage="2",
        git_sha=git_sha(),
        support=support.name,
        support_spec=SUPPORT,
        duration_s=duration_s,
        n_mics=int(audio.shape[0]),
        sr=int(support.sr),
        k_max=k_max,
        bin_hz=bin_hz,
        band_top_hz=BAND_TOP_HZ,
        in_band_k_max=[int(v) for v in ok.sum(axis=1)],
        carriers_rev_s=carriers.tolist(),
        min_spacing_hz=float(np.min(np.diff(np.sort(carriers)))),
        audio_vs_support_max_rel=audio_rel,
        rig_fit=str(RIG_FIT),
        rotor_fits=[str(p) for p in ROTOR_FITS],
        rotor_dynamics=rotor_dynamics(),
        offsets=off,
        offset_spread_ratio=float(
            MR.offset_spread_ratio(
                scene.cfg.rotors, delta_hz=float(np.min(np.diff(np.sort(carriers))))
            )
        ),
        calibration_db=np.round(cal, 4).tolist(),
        orders=[int(v) for v in orders],
        profile_db={k: np.round(v, 3).tolist() for k, v in est.items()},
        target_profile_db=np.round(np.where(np.isfinite(target), target, -999.0), 3).tolist(),
        in_band=ok.tolist(),
        resolvability_score=np.round(scores, 4).tolist(),
        comparison=comparison,
        collapsed_cells=collapsed,
        e2_diagnostics=dict(
            steering_orders=[int(v) for v in np.atleast_1d(e2m["steering_orders"])],
            steering_gain=np.round(np.abs(e2m["steering_gain"]), 4).tolist(),
            steering_delay_us=np.round(e2m["steering_delay"] * 1e6, 3).tolist(),
            bandwidth_hz=np.round(e2s["bandwidth_hz"], 3).tolist(),
            nearest_line_hz=np.round(
                np.where(np.isfinite(e2s["nearest_line_hz"]), e2s["nearest_line_hz"], -1.0), 4
            ).tolist(),
            beat_cycles=np.round(
                np.where(np.isfinite(e2s["beat_cycles"]), e2s["beat_cycles"], -1.0), 4
            ).tolist(),
            n_columns=e2s["n_columns"].tolist(),
            window_s=np.round(e2s["window_s"], 5).tolist(),
            coherence_time_s=np.round(e2s["coherence_time_s"], 5).tolist(),
            cond_median=float(np.median(e2m["cond"])),
        ),
    )
    path = out_dir / "estimators.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")

    init_path = write_profile_init(out_dir, est["E2-multi"], ok, carriers, chosen, k_max)
    fig = figure(out_dir, payload)
    print(f"# wrote {path}\n# wrote {init_path}\n# wrote {fig}")
    for name, rows in comparison.items():
        for band in rows:
            med = [b["median_abs_db"] for b in band["per_rotor"]]
            print(
                f"{name:20s} {band['band']:8s} median|d| per rotor "
                f"{[None if v is None else round(v, 2) for v in med]}"
            )
    return 0


def write_profile_init(
    out_dir: Path,
    profile_db: np.ndarray,
    ok: np.ndarray,
    carriers: np.ndarray,
    offsets: np.ndarray,
    k_max: int,
) -> Path:
    """The ``--profile-init`` payload of the stage-3 fit.

    ``profile_db`` is NaN where the estimator has nothing to say (out of band,
    or a collapsed cell) and ``sigma_db`` is NaN wherever the fit should keep
    its own two-regime prior -- above :data:`SIGMA_TIERS`' last tier, and on
    every NaN centre. NaN is the "leave the default alone" marker, so the file
    can never silently pin an order the estimator never measured.
    """
    prof = np.where(ok & (profile_db > -199.0), profile_db, np.nan)
    k = np.arange(1, k_max + 1)
    sigma = np.full(k_max, np.nan)
    lo = 1
    for hi, sd in SIGMA_TIERS:
        sigma[(k >= lo) & (k <= hi)] = sd
        lo = hi + 1
    sigma_db = np.where(np.isfinite(prof), np.broadcast_to(sigma[None, :], prof.shape), np.nan)
    path = out_dir / "profile_init_e2multi.npz"
    np.savez(
        path,
        profile_db=prof,
        sigma_db=sigma_db,
        carrier_rev_s=np.asarray(carriers, dtype=np.float64),
        carrier_offset_rev_s=np.asarray(offsets, dtype=np.float64),
        note=np.array(
            json.dumps(
                dict(
                    source="estimate_e2(multi=True) on bench_dregon_allMotors_70",
                    units="model profile_db (transfer and mic gains removed)",
                    sigma_tiers=[list(t) for t in SIGMA_TIERS],
                    nan="keep the model's own init / prior",
                )
            )
        ),
    )
    return path


# ── the figure ──────────────────────────────────────────────────────────────


def figure(out_dir: Path, payload: dict[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    orders = np.asarray(payload["orders"], dtype=float)
    ok = np.asarray(payload["in_band"], dtype=bool)
    target = np.asarray(payload["target_profile_db"], dtype=float)
    series = [
        ("R3 per-rotor fit", target, "k", 2.0),
        ("E1", np.asarray(payload["profile_db"]["E1"], dtype=float), "tab:red", 1.0),
        (
            "E2-single",
            np.asarray(payload["profile_db"]["E2-single"], dtype=float),
            "tab:orange",
            1.0,
        ),
        ("E2-multi", np.asarray(payload["profile_db"]["E2-multi"], dtype=float), "tab:blue", 1.4),
        (
            "R3 rig fit",
            np.asarray(payload["profile_db"]["R3-rig-fit"], dtype=float),
            "tab:green",
            1.0,
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6), sharex=True, sharey=True)
    for r, ax in enumerate(axes.ravel()):
        for name, arr, colour, lw in series:
            y = np.where(ok[r] & (arr[r] > -199.0), arr[r], np.nan)
            ax.plot(orders, y, color=colour, lw=lw, label=name if r == 0 else None)
        cap = int(ok[r].sum())
        ax.axvline(cap + 0.5, color="0.6", ls=":", lw=1)
        ax.set_title(
            f"rotor {r + 1} — carrier {payload['carriers_rev_s'][r]:.3f} rev/s, in band to k = {cap}"
        )
        ax.grid(alpha=0.3)
        ax.set_xlim(0, orders.max())
    for ax in axes[-1]:
        ax.set_xlabel("order k")
    for ax in axes[:, 0]:
        ax.set_ylabel("profile_db (dB)")
    fig.suptitle(
        f"{payload['support']} — per-rotor profile: the single-motor R3 fits against the three "
        f"estimators and the R3 rig fit",
        fontsize=11,
    )
    fig.legend(loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    path = out_dir / "fig_profiles.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ── stage 3 ─────────────────────────────────────────────────────────────────


def arm_row(name: str, path: Path, k_max_cap: int) -> dict[str, Any]:
    """One fit arm scored against the four per-rotor fits."""
    fit = json.loads(Path(path).read_text())
    p = fit["params"]
    carriers = np.asarray(p["carrier_rev_s"], dtype=np.float64).reshape(-1)
    prof = np.asarray(p["profile"]["profile_db"], dtype=np.float64)
    k_max = min(int(prof.shape[1]), k_max_cap)
    prof = prof[:, :k_max]
    target = rotor_profiles(k_max)
    ok = in_band_mask(carriers, k_max) & np.isfinite(target)
    gamma = MD.gamma_from_params(p)[:, :k_max]
    k = np.arange(1, k_max + 1)
    sel = ok & (k[None, :] <= CRIT_K_MAX)
    per_rotor = []
    for r in range(prof.shape[0]):
        d = np.abs(prof[r][sel[r]] - target[r][sel[r]])
        per_rotor.append(
            dict(
                rotor=r + 1,
                n=int(d.size),
                median_abs_db=float(np.median(d)) if d.size else None,
                frac_within_3db=float(np.mean(d <= 3.0)) if d.size else None,
                passes=bool(
                    d.size and np.median(d) <= CRIT_MEDIAN_DB and np.mean(d <= 3.0) >= CRIT_FRACTION
                ),
            )
        )
    return dict(
        arm=name,
        fit=str(path),
        support=fit.get("support"),
        k_max=int(fit.get("k_max", k_max)),
        converged=bool(fit["optimiser"]["converged"]),
        which_converged=fit["optimiser"].get("which_converged"),
        grad_norm=float(fit["optimiser"]["grad_norm"]),
        wall_s=float(fit["optimiser"]["wall_s"]),
        whittle_nats=float(fit["objective"]["whittle_nats"]),
        n_cells=int(fit["objective"]["n_cells"]),
        nats_per_cell=float(fit["objective"]["whittle_nats"]) / float(fit["objective"]["n_cells"]),
        sigma_nu=float(p["sigma_nu"]),
        lam=float(p["lam"]),
        carrier_rev_s=carriers.tolist(),
        gamma_ladder={str(j): float(np.max(gamma[:, j - 1])) for j in GAMMA_LADDER},
        gamma_log_mean=float(np.exp(np.mean(np.log(np.maximum(gamma, 1e-12))))),
        gamma_low_order_check=(fit.get("diagnostics") or {}).get("gamma_low_order_check"),
        criterion=dict(
            k_max=CRIT_K_MAX,
            median_db=CRIT_MEDIAN_DB,
            fraction=CRIT_FRACTION,
            per_rotor=per_rotor,
            verdict=bool(per_rotor and all(row["passes"] for row in per_rotor)),
        ),
        bands=band_stats(prof - target, ok),
    )


def compare(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in args.arm:
        name, _, path = item.partition("=")
        rows.append(arm_row(name, Path(path), int(args.k_max_cap)))
    payload = dict(
        schema=SCHEMA,
        stage="3",
        git_sha=git_sha(),
        criterion=dict(
            k_max=CRIT_K_MAX,
            median_abs_db=CRIT_MEDIAN_DB,
            frac_within_3db=CRIT_FRACTION,
            note="in-band orders only; target is the four bench_dregon_Motor{r}_70 R3 fits",
        ),
        rotor_dynamics=rotor_dynamics(),
        arms=rows,
    )
    path = out_dir / "stage3_arms.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"# wrote {path}")
    for row in rows:
        med = [r["median_abs_db"] for r in row["criterion"]["per_rotor"]]
        frac = [r["frac_within_3db"] for r in row["criterion"]["per_rotor"]]
        print(
            f"{row['arm']:4s} conv={row['converged']} nats/cell={row['nats_per_cell']:.5f} "
            f"sigma_nu={row['sigma_nu']:.4f} lam={row['lam']:.3f} "
            f"median|d|={[None if v is None else round(v, 2) for v in med]} "
            f"frac3={[None if v is None else round(v, 2) for v in frac]} "
            f"verdict={row['criterion']['verdict']}"
        )
    return 0


# ── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("estimate", help="stage 2: the estimators on the real support")
    e.add_argument("--out", default=str(OUT_DIR))
    e.add_argument("--k-max", type=int, default=0, help="0 = the carrier geometry's own cap")
    e.add_argument(
        "--offset-bound",
        type=float,
        default=0.05,
        help="refine_offsets search room in rev/s. The default 0.5 reaches the next rotor's "
        "comb and locks onto it; 0.05 is 1.4x the support index's worst half-window drift",
    )

    c = sub.add_parser("compare", help="stage 3: the fit arms against the per-rotor fits")
    c.add_argument("--out", default=str(OUT_DIR))
    c.add_argument("--arm", nargs="+", required=True, metavar="NAME=FIT.json")
    c.add_argument("--k-max-cap", type=int, default=130)

    args = ap.parse_args(argv)
    return {"estimate": estimate, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
