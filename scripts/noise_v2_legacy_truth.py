#!/usr/bin/env python
"""R4: the LEGACY DREGON render as the ground truth the tracker accepts.

HPPNet tracks the real DREGON room-2 cruise clips at 1.07 rev/s PIT MAE and
the LEGACY model's render of the same labels at 1.96; the round-3 v2 candidate
fitted to the same real recording renders at 69.6. Round 3 established that the
v2 render is faithful to its own forward model (``render_vs_model.md``), that
the fit is within ~2 dB of the real periodogram in every cell class, and that
neither line width nor phase structure moves the tracker at the fitted level.
What is left is the question this runner asks: treat the LEGACY render as the
data and ask what v2 must carry to reproduce it.

Three sub-studies, one subcommand each:

``anatomy``
    (A) The legacy DREGON export, the v2 R3 bench fits and the v2 R3 flight fit
    side by side — per-order level, width law, coherence model, floor model and
    the bench -> flight composition of each — plus, MEASURED on the renders of
    the three score windows, the k <= 32 prominence ladder over the local
    floor, the -3 dB widths, the floor slope and the cross-order phase
    coherence ``|E[exp(i(phi_2k - 2 phi_k))]|``.

``supports``
    (B) Render the legacy model on the FIVE 8 s windows the R3 v2 flight fit
    pooled, with the frozen ``motors_command`` labels, and write v2 supports
    built from that audio (``supports.synthetic_support``) to a cache directory
    a fit can read.

``score``
    (B3, C) Render named fits on the three score windows and score the frozen
    HPPNet, plus the parameter-group swaps between two fits.

Every number this runner writes comes from one of its own JSON payloads.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.noise_model import gates as GT  # noqa: E402
from experiments.noise_model import render as RD  # noqa: E402
from experiments.noise_model import supports as SU  # noqa: E402
from experiments.stochastic_fit import revised_eval as RE  # noqa: E402

SCHEMA = "noise-v2-legacy-truth/1"
OUT_DEFAULT = Path("results/noise_v2/rounds/round4/legacy_truth")
R3_FITS = Path("results/noise_v2/rounds/round3/fits")
FIT_REAL = R3_FITS / "dregon_room2_floor__flight_floor_lowk.json"
BENCH_FITS = tuple(R3_FITS / f"bench_dregon_Motor{i}_70__bench.json" for i in (1, 2, 3, 4))
LEGACY_EXPORT = "results/S2/dregon_room2_cruise_refined.json"

#: The three CRUISE score windows every round-3 DREGON reading is on.
RECORDINGS = (
    "free-flight_nosource_room2",
    "hovering_nosource_room2",
    "updown_nosource_room2",
)
#: The frozen render seed of the round-3 readings this file is read against.
SEED = 2001
#: The prominence ladder and the orders the -3 dB width is read at.
PROM_ORDERS = tuple(range(1, 33))
WIDTH_ORDERS = (1, 2, 4, 8, 16)
#: The orders the legacy width law and the v2 gamma ladder are tabulated at.
GAMMA_LADDER = (1, 2, 4, 8, 16, 32)
#: ``|E[exp(i(phi_2k - 2 phi_k))]|`` is read at these k (2k <= 8).
BISPEC_ORDERS = (1, 2, 3, 4)


def die(m: str) -> NoReturn:
    raise SystemExit(f"error: {m}")


def _module(name: str) -> Any:
    path = Path("scripts") / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        die(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


# ── the shared window machinery ─────────────────────────────────────────────


def scored_support(recording: str) -> GT.ScoredSupport:
    found = [s for s in GT.DREGON_CRUISE_SUPPORTS if s.recording == recording]
    if not found:
        die(f"no frozen DREGON cruise support carries recording {recording!r}")
    return found[0]


def load_clip(support: GT.ScoredSupport) -> Any:
    return RE.load_window(
        support.window,
        dataset=GT.DATASET["dregon"],
        version=None,
        channels=None,
        rps_key=GT.RAW_RPS_KEY["dregon"],
    )


def legacy_params(recording: str) -> Any:
    """The identity-matched legacy parameter set of one recording, PHYSICAL."""
    rs = _module("noise_v2_round_score")
    arm = rs.legacy_arm("dregon")
    return RE.to_renderer_units(
        rs.legacy_identity_params(arm, regime="cruise", recording=recording)
    )


def legacy_render(recording: str, rps: np.ndarray, *, n_mics: int = 8, seed: int = SEED) -> Any:
    """The legacy arm's own render route, identity-matched to ``recording``."""
    rs = _module("noise_v2_round_score")
    arm = rs._arm_for_recording(rs.legacy_arm("dregon"), scored_support(recording))
    return np.asarray(
        arm.render(np.atleast_2d(rps), regime="cruise", n_mics=int(n_mics), seed=int(seed)),
        dtype=np.float64,
    )


# ── the order-tracked read (the round-3 estimator, one STFT per mic) ────────

WIDE_N, WIDE_HOP = 8192, 1024


def stft_power(x: np.ndarray, *, n: int = WIDE_N, hop: int = WIDE_HOP) -> tuple[Any, Any]:
    x = np.asarray(x, dtype=np.float64)
    w = np.hanning(n)
    starts = np.arange(0, x.size - n + 1, hop)
    power = np.stack([np.abs(np.fft.rfft(x[s : s + n] * w)) ** 2 for s in starts])
    return power / float((w**2).sum()), starts + n // 2


def order_profiles(
    audio: np.ndarray, f0_tracks: np.ndarray, orders: tuple[int, ...], *, sr: int
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """The order-tracked band profile of every order in ONE pass per mic.

    Identical in definition to ``noise_v2_widen_dregon.line_width_db3``'s own
    accumulation — each frame read on that frame's own label carrier, averaged
    over mics, rotors and frames — but the 8192-point STFT is computed once per
    microphone instead of once per (microphone, order). The agreement is
    asserted at ``k = 1`` in the payload's ``estimator_check``.
    """
    freqs = np.fft.rfftfreq(WIDE_N, d=1.0 / float(sr))
    f0 = np.asarray(f0_tracks, dtype=np.float64)
    fbar = float(f0.mean())
    grid = np.arange(-0.75 * fbar, 0.75 * fbar + 1e-9, 0.5)
    acc = {k: np.zeros(grid.size) for k in orders}
    n_used = 0
    for m in range(int(audio.shape[0])):
        power, centres = stft_power(audio[m])
        idx = np.clip(centres.astype(np.int64), 0, int(f0.shape[1]) - 1)
        for r in range(int(f0.shape[0])):
            car = f0[r][idx]
            for j, c in enumerate(car):
                for k in orders:
                    acc[k] += np.interp(float(k) * c + grid, freqs, power[j])
                n_used += 1
    return grid, {k: v / max(n_used, 1) for k, v in acc.items()}


def profile_stats(prof: np.ndarray, grid: np.ndarray, fbar: float, *, sr: int) -> dict[str, Any]:
    """``peak_over_base_db`` and the -3 dB width of one order-tracked profile."""
    side = (np.abs(grid) >= 0.45 * fbar) & (np.abs(grid) <= 0.7 * fbar)
    base = float(np.median(prof[side])) if side.any() else float(np.median(prof))
    ex = prof - base
    i0 = int(np.argmin(np.abs(grid)))
    peak = float(ex[max(i0 - 2, 0) : i0 + 3].max())
    if peak <= 0:
        return dict(
            width_hz=None,
            peak_over_base_db=float("nan"),
            base_power=base,
            resolution_hz=float(sr) / WIDE_N,
        )
    half = peak / 2.0

    def cross(step: int) -> float:
        j = i0
        while 0 <= j + step < ex.size and abs(float(grid[j + step])) <= 0.5 * fbar:
            if ex[j + step] <= half:
                y0, y1 = float(ex[j]), float(ex[j + step])
                if y1 == y0:
                    return float(grid[j + step])
                return float(grid[j]) + (half - y0) * (float(grid[j + step]) - float(grid[j])) / (
                    y1 - y0
                )
            j += step
        return float("nan")

    lo, hi = cross(-1), cross(1)
    return dict(
        width_hz=(float(hi - lo) if np.isfinite(lo) and np.isfinite(hi) else None),
        peak_over_base_db=float(10.0 * np.log10(max(peak + base, 1e-300) / max(base, 1e-300))),
        base_power=base,
        resolution_hz=float(sr) / WIDE_N,
    )


def ladder(audio: np.ndarray, f0: np.ndarray, *, sr: int) -> dict[str, Any]:
    """The k <= 32 prominence ladder, the widths, and the LOCAL FLOOR slope.

    The local floor of order ``k`` is the same two-sided annulus median the
    prominence is read against, so the slope is the floor the comb is read
    against and not a separate estimator.
    """
    fbar = float(np.asarray(f0, dtype=np.float64).mean())
    grid, profs = order_profiles(audio, f0, PROM_ORDERS, sr=sr)
    rows = {k: profile_stats(profs[k], grid, fbar, sr=sr) for k in PROM_ORDERS}
    ks = np.array([k for k in PROM_ORDERS if rows[k]["base_power"] > 0.0], dtype=np.float64)
    base_db = np.array([10.0 * np.log10(rows[int(k)]["base_power"]) for k in ks], dtype=np.float64)
    slope = float("nan")
    if ks.size >= 4:
        x = np.log2(ks * fbar)
        slope = float(np.polyfit(x, base_db, 1)[0])
    return dict(
        fbar_rev_s=fbar,
        prominence_db={str(k): rows[k]["peak_over_base_db"] for k in PROM_ORDERS},
        width_hz={str(k): rows[k]["width_hz"] for k in WIDTH_ORDERS},
        local_floor_db={
            str(k): (10.0 * np.log10(max(rows[k]["base_power"], 1e-300))) for k in PROM_ORDERS
        },
        floor_slope_db_oct=slope,
    )


# ── cross-order phase coherence ─────────────────────────────────────────────


def bispectral_coherence(audio: np.ndarray, f0: np.ndarray, *, sr: int) -> dict[str, Any]:
    """``|E_t[exp(i(phi_2k - 2 phi_k))]|`` on the label's own carriers.

    ``z_k`` is the complex envelope of order ``k`` demodulated at ``k f_r(t)``
    (the round-3 coherence probe's own +-16 Hz time-domain demodulation), so
    ``arg z_k`` IS the order's residual phase. A comb whose orders ride one
    shaft has ``arg z_2k - 2 arg z_k`` constant in time and the statistic is 1;
    independent per-order phase noise or a floor-dominated band drives it to 0.
    The adjacent-pair statistic of ``coherence.md`` is reported beside it so
    the two rounds' numbers can be read against each other.
    """
    cd = _module("noise_v2_coherence_dregon")
    f0 = np.asarray(f0, dtype=np.float64)
    bis: dict[int, list[float]] = {k: [] for k in BISPEC_ORDERS}
    adj: list[float] = []
    for r in range(int(f0.shape[0])):
        zs = {
            k: cd.demodulate(audio, f0[r], k, sr=int(sr))
            for k in sorted(set(BISPEC_ORDERS) | {2 * k for k in BISPEC_ORDERS} | set(range(1, 7)))
        }
        for k in BISPEC_ORDERS:
            zk, z2k = zs[k], zs[2 * k]
            num = z2k * np.conj(zk) ** 2
            den = np.abs(z2k) * np.abs(zk) ** 2
            u = np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)
            bis[k].extend(np.abs(u.mean(axis=-1)).tolist())
        for k in range(1, 6):
            u = zs[k + 1] * np.conj(zs[k])
            u = u / np.maximum(np.abs(u), 1e-300)
            adj.extend(np.abs(u.mean(axis=-1)).tolist())
    return dict(
        bispec={str(k): float(np.mean(v)) for k, v in bis.items()},
        adjacent_pair_mean=float(np.mean(adj)),
        protocol=dict(bw_hz=cd.DEMOD_BW_HZ, hop=cd.DEMOD_HOP, orders=list(BISPEC_ORDERS)),
    )


# ── (A) the parameter anatomy ───────────────────────────────────────────────


def legacy_anatomy(recording: str) -> dict[str, Any]:
    """The legacy export's parameters, in its own units and in v2's."""
    mp = legacy_params(recording)
    p = mp.params
    prof = np.asarray(p["profile_db"], dtype=np.float64)
    prof_cand = RE.legacy_profile_db_to_candidate(prof)
    g0 = np.asarray(p["gamma0"], dtype=np.float64)
    gs = np.asarray(p["gamma_slope"], dtype=np.float64)
    k_half = float(p["coherence_k_half"])
    ks = np.arange(1, prof.shape[1] + 1, dtype=np.float64)
    w = np.exp(-((ks / max(k_half, 1e-3)) ** 2))
    return dict(
        source=dict(mp.source),
        export=LEGACY_EXPORT,
        n_rotors=int(prof.shape[0]),
        n_orders=int(prof.shape[1]),
        profile_db_legacy_units={
            str(k): prof[:, k - 1].tolist() for k in PROM_ORDERS if k <= prof.shape[1]
        },
        profile_db_candidate_units={
            str(k): prof_cand[:, k - 1].tolist() for k in PROM_ORDERS if k <= prof.shape[1]
        },
        profile_units=dict(
            legacy="band weight in periodogram*Hz (model.CombSpectrum.line_power)",
            candidate="tone mean square, A = sqrt(2 P)",
            conversion_db=float(10.0 * np.log10(2.0 / 16000.0)),
            conversion="revised_eval.legacy_profile_db_to_candidate",
        ),
        gamma_law=dict(
            rule="gamma_rk = gamma0_r + gamma_slope_r * k (width_power 1.0)",
            gamma0_hz=g0.tolist(),
            gamma_slope_hz=gs.tolist(),
            gamma_hz={str(k): (g0 + gs * float(k)).tolist() for k in GAMMA_LADDER},
        ),
        coherence=dict(
            rule="w_k = exp(-(k / coherence_k_half)^2): the COHERENT share of order k's "
            "power rides the tone bank as a needle carrying the analysis window's own "
            "|W(f - f0)|^2; the remaining 1 - w_k is rendered as narrowband NOISE of "
            "half-width gamma_rk alongside the floor",
            coherence_k_half=k_half,
            w_k={str(k): float(w[k - 1]) for k in GAMMA_LADDER if k <= w.size},
        ),
        floor=dict(
            floor_mean_db=float(p["floor_mean_db"]),
            floor_tilt_db_oct=float(p["floor_tilt_db_oct"]),
            floor_shape_db=np.asarray(p["floor_shape_db"], dtype=np.float64).tolist(),
            floor_ctrl_hz=np.asarray(p["floor_ctrl_hz"], dtype=np.float64).tolist(),
            floor_static_rel=float(p["floor_static_rel"]),
            floor_exp=float(p["floor_exp"]),
            mic_floor_db=np.asarray(p["mic_floor_db"], dtype=np.float64).tolist(),
        ),
        speed_law=dict(
            amp_exp=float(p["amp_exp"]),
            floor_exp=float(p["floor_exp"]),
            amp_rps_ref=80.0,
            rule="line power scales (f_r / 80)^amp_exp, floor power "
            "mean_r (f_r / 80)^floor_exp + floor_static_rel",
        ),
        mic=dict(
            mic_gain_db=np.asarray(p["mic_gain_db"], dtype=np.float64).tolist(),
            gain_all_db=np.asarray(p["gain_all_db"], dtype=np.float64).tolist(),
        ),
    )


def v2_anatomy() -> dict[str, Any]:
    """The R3 v2 bench fits and the R3 v2 DREGON flight fit."""
    bench = {}
    for path in BENCH_FITS:
        d = json.loads(Path(path).read_text())
        p = d["params"]
        prof = np.asarray(p["profile"]["profile_db"], dtype=np.float64)
        gam = np.asarray(p["gamma_hz"], dtype=np.float64)
        bench[Path(path).stem] = dict(
            path=str(path),
            converged=d["optimiser"]["converged"],
            which_converged=d["optimiser"].get("which_converged"),
            sigma_nu=float(p["sigma_nu"]),
            lam=float(p["lam"]),
            carrier_rev_s=p.get("carrier_rev_s"),
            amp_exp=float(p["profile"]["amp_exp"]),
            profile_db={str(k): prof[:, k - 1].tolist() for k in PROM_ORDERS if k <= prof.shape[1]},
            gamma_hz={str(k): gam[:, k - 1].tolist() for k in GAMMA_LADDER if k <= gam.shape[1]},
            floor={k: v for k, v in p["floor"].items() if not isinstance(v, list)},
            whittle_nats_per_cell=float(d["objective"]["whittle_nats"])
            / float(d["objective"]["n_cells"]),
        )
    d = json.loads(FIT_REAL.read_text())
    p = d["params"]
    prof = np.asarray(p["profile"]["profile_db"], dtype=np.float64)
    gam = np.asarray(p["gamma_hz"], dtype=np.float64)
    flight = dict(
        path=str(FIT_REAL),
        mode=d["mode"],
        supports=d["supports"],
        converged=d["optimiser"]["converged"],
        which_converged=d["optimiser"].get("which_converged"),
        sigma_nu=float(p["sigma_nu"]),
        lam=float(p["lam"]),
        comb_gain_db=float(p["profile"]["comb_gain_db"]),
        low_order_gain_db=[float(v) for v in p["profile"]["low_order_gain_db"]],
        amp_exp=float(p["profile"]["amp_exp"]),
        profile_db={str(k): prof[:, k - 1].tolist() for k in PROM_ORDERS if k <= prof.shape[1]},
        gamma_hz={str(k): gam[:, k - 1].tolist() for k in GAMMA_LADDER if k <= gam.shape[1]},
        floor={k: v for k, v in p["floor"].items() if not isinstance(v, list)},
        floor_shape_z=list(p["floor"]["floor_shape_z"]),
        mic_floor_db=list(p["floor"]["mic_floor_db"]),
        whittle_nats_per_cell=float(d["objective"]["whittle_nats"])
        / float(d["objective"]["n_cells"]),
        frozen_from=d.get("frozen_from"),
        span_pins=(d.get("diagnostics") or {}).get("span_pins"),
    )
    return dict(bench=bench, flight=flight)


def composition() -> dict[str, Any]:
    """How each model maps its fitted parameters onto a FREE-FLIGHT comb."""
    return dict(
        legacy=dict(
            fitted_on="the flight recording ITSELF — results/S2/dregon_room2_cruise_refined.json "
            "carries one MAP fit per room-2 flight clip and the arm is identity-matched to the "
            "scored recording, so there is no bench -> flight transplant at all",
            line_power="P_rk(t) = 10^(profile_db[r,k]/10) * (f_r(t) / 80)^amp_exp, split by "
            "w_k = exp(-(k/k_half)^2) into a coherent needle (w_k) and a Rayleigh pedestal "
            "of half-width gamma0_r + gamma_slope_r k (1 - w_k)",
            floor_power="floor shape (14 control points, tilt dB/oct) * "
            "(mean_r (f_r/80)^floor_exp + floor_static_rel)",
            level="every level is that clip's own MAP value with its power_scale folded in "
            "and the +6.02 dB work-rate term applied (revised_eval.to_renderer_units)",
        ),
        v2=dict(
            fitted_on="the bench comb of four DREGON single-motor 70% recordings, TRANSPLANTED "
            "onto the flight pool; the flight fit moves only the floor, the mic gains, one "
            "shared comb_gain_db and one gain per order for k <= low_orders",
            line_power="P_rk(t) = 10^(profile_db[r,k]/10) * (f_r(t)/80)^amp_exp with "
            "profile_db ALREADY carrying comb_gain_db and low_order_gain_db; every order is a "
            "single Wiener-phase line of half-width gamma_rk (no coherent/incoherent split)",
            floor_power="floor shape (14 z control points, tilt dB/oct) * "
            "(mean_r (f_r/80)^floor_exp + floor_static_rel)",
            transplant="log-mean of the four bench fits' rates, scales and per-line widths, "
            "dB-mean of the per-order profile, over the fits that reach each order",
        ),
    )


def run_anatomy(*, out: Path, recordings: tuple[str, ...]) -> dict[str, Any]:
    wd = _module("noise_v2_widen_dregon")
    fit_v2 = json.loads(FIT_REAL.read_text())
    payload: dict[str, Any] = dict(
        schema=SCHEMA,
        study="A",
        git=git_rev(),
        protocol=dict(
            seed=SEED,
            n_mics=8,
            wide=dict(n_fft=WIDE_N, hop=WIDE_HOP, resolution_hz=16000.0 / WIDE_N),
            prominence_orders=list(PROM_ORDERS),
            width_orders=list(WIDTH_ORDERS),
            estimator="noise_v2_widen_dregon.line_width_db3, one STFT per microphone",
        ),
        composition=composition(),
        v2=v2_anatomy(),
        legacy={},
        windows={},
    )
    figures: dict[str, Any] = {}
    for recording in recordings:
        support = scored_support(recording)
        clip = load_clip(support)
        sr = int(clip.sr)
        real = np.asarray(clip.audio, dtype=np.float64)[:8]
        rps = np.atleast_2d(np.asarray(clip.rps, dtype=np.float64))
        payload["legacy"][recording] = legacy_anatomy(recording)
        # the +21 dB arm is the CONTROL on every statistic below: it is the v2
        # model with nothing changed but the comb level, so anything it moves
        # as far as legacy does is a level artefact and not a model quality
        fit_v2_up = copy.deepcopy(fit_v2)
        fit_v2_up["params"]["profile"]["profile_db"] = (
            np.asarray(fit_v2["params"]["profile"]["profile_db"], dtype=np.float64) + 21.0
        ).tolist()
        arms = dict(
            real=real,
            legacy=np.asarray(legacy_render(recording, rps), dtype=np.float64)[:8],
            v2=np.asarray(RD.render_noise(fit_v2, rps, n_mics=8, seed=SEED), dtype=np.float64)[:8],
            v2_plus21db=np.asarray(
                RD.render_noise(fit_v2_up, rps, n_mics=8, seed=SEED), dtype=np.float64
            )[:8],
        )
        fbar = float(rps.mean())
        row: dict[str, Any] = dict(
            support=support.as_dict(),
            carrier=dict(
                fbar_rev_s=fbar,
                min_rev_s=float(rps.min()),
                max_rev_s=float(rps.max()),
                legacy_speed_factor_db=float(
                    payload["legacy"][recording]["speed_law"]["amp_exp"]
                    * 10.0
                    * np.log10(fbar / 80.0)
                ),
                v2_speed_factor_db=float(
                    payload["v2"]["flight"]["amp_exp"] * 10.0 * np.log10(fbar / 80.0)
                ),
            ),
            arms={},
        )
        for name, audio in arms.items():
            lad = ladder(audio, rps, sr=sr)
            check = wd.line_width_db3(audio, rps, 1, sr=sr)
            lad["estimator_check"] = dict(
                k=1,
                widen_peak_over_base_db=float(check["peak_over_base_db"]),
                ladder_peak_over_base_db=float(lad["prominence_db"]["1"]),
                widen_width_hz=check["width_hz"],
                ladder_width_hz=lad["width_hz"]["1"],
            )
            lad["coherence"] = bispectral_coherence(audio, rps, sr=sr)
            row["arms"][name] = lad
            print(
                f"[{recording}] {name}: k1 {lad['prominence_db']['1']:+.2f} dB "
                f"k2 {lad['prominence_db']['2']:+.2f} floor {lad['floor_slope_db_oct']:+.2f} dB/oct "
                f"B1 {lad['coherence']['bispec']['1']:.3f}",
                flush=True,
            )
        payload["windows"][support.key] = row
        figures[recording] = {n: row["arms"][n]["prominence_db"] for n in arms}
    Path(out).mkdir(parents=True, exist_ok=True)
    payload["_figures"] = figures
    return payload


def write_ladder_figure(figures: dict[str, Any], out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = dict(real="#222222", legacy="#c1440e", v2="#1f6fb4", v2_plus21db="#2e8b57")
    styles = dict(v2_plus21db="--")
    fig, axes = plt.subplots(1, len(figures), figsize=(4.2 * len(figures), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    ks = np.array(PROM_ORDERS, dtype=float)
    for ax, (recording, arms) in zip(axes, figures.items(), strict=False):
        for name, prom in arms.items():
            y = np.array([prom[str(int(k))] for k in ks], dtype=float)
            ax.plot(
                ks,
                y,
                marker="o",
                ms=3,
                lw=1.4,
                ls=styles.get(name, "-"),
                color=colours.get(name, "#888888"),
                label=name,
            )
        ax.axhline(0.0, color="#999999", lw=0.8, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("order k")
        ax.set_title(recording.replace("_nosource_room2", ""))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("order prominence over local floor (dB)")
    axes[0].legend(fontsize=8)
    fig.suptitle("DREGON cruise: order prominence ladder, real / legacy / v2 R3 (seed 2001)")
    fig.tight_layout()
    path = Path(out) / "prominence_ladder.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [str(path)]


def _db(v: Any) -> str:
    """A dB cell, or ``—`` where the estimator identified nothing."""
    x = float("nan") if v is None else float(v)
    return "—" if not np.isfinite(x) else f"{x:+.2f}"


def anatomy_findings(payload: dict[str, Any]) -> str:
    o: list[str] = []
    o.append("# R4 legacy-truth A: anatomy of the legacy DREGON fit against v2 R3")
    o.append("")
    o.append(
        f"Windows {', '.join(r.replace('_nosource_room2', '') for r in RECORDINGS)}; seed "
        f"{SEED}, 8 mics; legacy export `{LEGACY_EXPORT}` (identity-matched per recording), "
        f"v2 flight fit `{FIT_REAL}` (`converged` "
        f"{payload['v2']['flight']['converged']}); git `{payload['git']}`."
    )
    o.append("")
    o.append("## Per-order comb level (dB)")
    o.append("")
    o.append(
        "Legacy `profile_db` is a band weight in periodogram*Hz and v2's is a tone mean square; "
        "the column `legacy (v2 units)` applies the ONE stated conversion "
        "`revised_eval.legacy_profile_db_to_candidate` (+10 log10(2/16000) = -39.03 dB). "
        "`legacy needle` is that level times the coherent share `w_k`; the rest of the order's "
        "power is a Rayleigh pedestal of half-width `gamma_rk`. Rotor-mean in linear power."
    )
    o.append("")
    lg = payload["legacy"][RECORDINGS[0]]
    fl = payload["v2"]["flight"]
    o.append(
        "| k | legacy (own units) | legacy (v2 units) | w_k | legacy needle (v2 units) | "
        "v2 flight profile | v2 low-order gain | legacy - v2 |"
    )
    o.append("|---:|---:|---:|---:|---:|---:|---:|---:|")

    def dbmean(v: list[float]) -> float:
        return float(10.0 * np.log10(np.mean(10.0 ** (np.asarray(v, dtype=np.float64) / 10.0))))

    k_half = float(lg["coherence"]["coherence_k_half"])
    low = fl["low_order_gain_db"]
    for k in PROM_ORDERS:
        s = str(k)
        if s not in lg["profile_db_legacy_units"]:
            continue
        a = dbmean(lg["profile_db_legacy_units"][s])
        b = dbmean(lg["profile_db_candidate_units"][s])
        w = float(np.exp(-((k / max(k_half, 1e-3)) ** 2)))
        c = dbmean(fl["profile_db"][s]) if s in fl["profile_db"] else float("nan")
        lo = f"{low[k - 1]:+.2f}" if k <= len(low) else "—"
        o.append(
            f"| {k} | {a:+.2f} | {b:+.2f} | {w:.4f} | {b + 10.0 * np.log10(max(w, 1e-12)):+.2f} "
            f"| {c:+.2f} | {lo} | {b - c:+.2f} |"
        )
    o.append("")
    o.append(
        "Both families scale the comb by `(f_r / 80)^amp_exp`, so at a window's own mean "
        "carrier the legacy comb carries an extra "
        + ", ".join(
            f"{row['carrier']['legacy_speed_factor_db']:+.2f} dB "
            f"(fbar {row['carrier']['fbar_rev_s']:.1f} rev/s, "
            f"{key.split('@')[0].replace('_nosource_room2', '')})"
            for key, row in payload["windows"].items()
        )
        + " and the v2 comb exactly 0 dB (`amp_exp` fitted at 0)."
    )
    o.append("")
    o.append("## Line width, coherence, floor and speed law")
    o.append("")
    o.append("| quantity | legacy | v2 R3 flight |")
    o.append("|---|---|---|")
    gl = lg["gamma_law"]["gamma_hz"]
    gv = fl["gamma_hz"]
    for k in GAMMA_LADDER:
        s = str(k)
        lv = float(np.mean(gl[s])) if s in gl else float("nan")
        vv = float(np.mean(gv[s])) if s in gv else float("nan")
        o.append(f"| line half-width gamma at k={k} (Hz) | {lv:.3f} | {vv:.4f} |")
    o.append(
        f"| line model | needle (share w_k) + Rayleigh pedestal, k_half {k_half:.3f} "
        "| ONE Wiener-phase line per order, no split |"
    )
    o.append(
        f"| shaft model | label track only (shaft_jitter 0, phase diffusion 0) | "
        f"integrated OU, sigma_nu {fl['sigma_nu']:.4f}, lam {fl['lam']:.4f} |"
    )
    o.append(
        f"| amp_exp (line power ~ (f/80)^a) | {lg['speed_law']['amp_exp']:.3f} "
        f"| {fl['amp_exp']:.3f} |"
    )
    o.append(
        f"| floor_exp | {lg['floor']['floor_exp']:.3f} | {float(fl['floor']['floor_exp']):.3f} |"
    )
    o.append(
        f"| floor_static_rel | {lg['floor']['floor_static_rel']:.4f} "
        f"| {float(fl['floor']['floor_static_rel']):.4f} |"
    )
    o.append(
        f"| floor_mean_db | {lg['floor']['floor_mean_db']:.2f} "
        f"| {float(fl['floor']['floor_mean_db']):.2f} |"
    )
    o.append(
        f"| floor_tilt_db_oct | {lg['floor']['floor_tilt_db_oct']:.3f} "
        f"| {float(fl['floor']['floor_tilt_db_oct']):.3f} |"
    )
    o.append(f"| comb_gain_db | — (no transplant) | {fl['comb_gain_db']:+.3f} |")
    o.append("")
    o.append("## MEASURED on the render: prominence over the local floor (dB)")
    o.append("")
    o.append(
        "Order-tracked on each frame's own label carrier, 8192-point (1.95 Hz per bin), mic- "
        "and rotor-averaged; the local floor is the two-sided 0.45-0.7 fbar annulus median. "
        "This is `noise_v2_widen_dregon.line_width_db3`'s own statistic, checked against it at "
        "k=1 in `anatomy.json` (`estimator_check`)."
    )
    o.append(
        "An order whose band peak does not clear its own local floor carries no identifiable "
        "prominence and is `—`."
    )
    o.append("")
    head = "| window | arm | " + " | ".join(f"k={k}" for k in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32))
    o.append(head + " | floor slope dB/oct |")
    o.append("|---|---|" + "---:|" * 11)
    for key, row in payload["windows"].items():
        w = key.split("@")[0].replace("_nosource_room2", "")
        for name, lad in row["arms"].items():
            cells = " | ".join(
                _db(lad["prominence_db"][str(k)]) for k in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
            )
            o.append(f"| `{w}` | `{name}` | {cells} | {lad['floor_slope_db_oct']:+.2f} |")
    o.append("")
    o.append("## MEASURED: -3 dB width and cross-order phase coherence")
    o.append("")
    o.append(
        "`|E[exp(i(phi_2k - 2 phi_k))]|` on the +-16 Hz demodulated envelopes of the label's own "
        "carriers, averaged over 4 rotors and 8 microphones; 1 = the two orders keep a fixed "
        "phase relation over the window, 0 = they do not. `adj` is round 3's adjacent-pair "
        "statistic on the same envelopes."
    )
    o.append("")
    o.append(
        "| window | arm | w(k=1) | w(k=2) | w(k=4) | w(k=8) | w(k=16) | B(1,2) | B(2,4) | "
        "B(3,6) | B(4,8) | adj |"
    )
    o.append("|---|---|" + "---:|" * 10)
    for key, row in payload["windows"].items():
        w = key.split("@")[0].replace("_nosource_room2", "")
        for name, lad in row["arms"].items():
            widths = " | ".join(
                ("—" if lad["width_hz"][str(k)] is None else f"{lad['width_hz'][str(k)]:.1f}")
                for k in WIDTH_ORDERS
            )
            b = lad["coherence"]["bispec"]
            bs = " | ".join(f"{b[str(k)]:.3f}" for k in BISPEC_ORDERS)
            o.append(
                f"| `{w}` | `{name}` | {widths} | {bs} "
                f"| {lad['coherence']['adjacent_pair_mean']:.3f} |"
            )
    o.append("")
    return "\n".join(o)


# ── (B) the synthetic supports ──────────────────────────────────────────────


def run_supports(*, out: Path, cache: Path, specs: list[str], prefix: str) -> dict[str, Any]:
    """Render the legacy model on each spec's window and cache a v2 support."""
    rows = []
    for text in specs:
        spec = SU.as_spec(text)
        args = spec.args
        real = SU.load_support(spec)
        recording = str(args["recording"])
        rps = np.asarray(real.carrier_rev_s_audio, dtype=np.float64)
        audio = legacy_render(recording, rps, n_mics=int(real.n_mics), seed=SEED)
        n = int(rps.shape[-1])
        if audio.shape[-1] < n:
            die(f"{spec.name}: legacy render is {audio.shape[-1]} samples, support wants {n}")
        support = SU.synthetic_support(
            f"{prefix}{spec.name}",
            audio[:, :n],
            rps,
            sr=int(real.sr),
            segment=real.segment,
            meta=dict(
                real.meta,
                synthetic=dict(
                    source="legacy stage-2 identity-matched render",
                    export=LEGACY_EXPORT,
                    recording=recording,
                    seed=SEED,
                    from_support=spec.name,
                    from_spec=spec.text,
                    renderer="stage2.render_from_export via round_score.Arm.render",
                ),
            ),
        )
        path = SU.save_support(support, out_dir=cache)
        rows.append(
            dict(
                name=support.name,
                spec=support.meta["spec"],
                from_spec=spec.text,
                recording=recording,
                npz=str(path),
                n_mics=int(support.n_mics),
                n_frames=int(support.power.shape[1]),
                seconds=float(support.n_samples / support.sr)
                if hasattr(support, "n_samples")
                else None,
                carrier_rev_s_mean=float(np.mean(support.carrier_rev_s)),
                band_power_db=float(10.0 * np.log10(float(np.mean(support.power)))),
                real_band_power_db=float(10.0 * np.log10(float(np.mean(real.power)))),
            )
        )
        print(f"[supports] {support.name}: {rows[-1]['n_frames']} frames -> {path}", flush=True)
    Path(out).mkdir(parents=True, exist_ok=True)
    return dict(
        schema=SCHEMA,
        study="B-supports",
        git=git_rev(),
        cache_dir=str(cache),
        seed=SEED,
        supports=rows,
    )


# ── (B3, C) rendering and scoring named fits and parameter swaps ────────────

#: The parameter GROUPS study C swaps one at a time between two fits.
SWAP_GROUPS: dict[str, tuple[str, ...]] = {
    "comb_gain": ("profile.comb_gain_db",),
    "low_order": ("profile.low_order_gain_db",),
    "gamma": ("gamma_hz",),
    "dynamics": ("sigma_nu", "lam"),
    "floor": ("floor",),
}


def _get(params: dict[str, Any], dotted: str) -> Any:
    node: Any = params
    for part in dotted.split("."):
        node = node[part]
    return copy.deepcopy(node)


def _set(params: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = params
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = copy.deepcopy(value)


def apply_profile_gains(fit: dict[str, Any], src: dict[str, Any], group: str) -> dict[str, Any]:
    """Swap ONE parameter group of ``fit`` for ``src``'s.

    ``profile_db`` already carries ``comb_gain_db`` and ``low_order_gain_db``
    (``fit.write_fit``'s own rule: never re-apply them), so swapping either
    gain means re-folding the DIFFERENCE into ``profile_db`` as well — that is
    what makes the swap a swap of the rendered comb and not of a record field.
    """
    out = copy.deepcopy(fit)
    po, ps = out["params"], src["params"]
    if group == "comb_gain":
        d = float(ps["profile"]["comb_gain_db"]) - float(po["profile"]["comb_gain_db"])
        prof = np.asarray(po["profile"]["profile_db"], dtype=np.float64) + d
        po["profile"]["profile_db"] = prof.tolist()
        po["profile"]["comb_gain_db"] = float(ps["profile"]["comb_gain_db"])
        return out
    if group == "low_order":
        a = np.asarray(po["profile"]["low_order_gain_db"], dtype=np.float64)
        b = np.asarray(ps["profile"]["low_order_gain_db"], dtype=np.float64)
        n = min(a.size, b.size)
        prof = np.asarray(po["profile"]["profile_db"], dtype=np.float64)
        prof[:, :n] += b[:n] - a[:n]
        if b.size > n:
            prof[:, n : b.size] += b[n : b.size]
        if a.size > n:
            prof[:, n : a.size] -= a[n : a.size]
        po["profile"]["profile_db"] = prof.tolist()
        po["profile"]["low_order_gain_db"] = b.tolist()
        return out
    for key in SWAP_GROUPS[group]:
        _set(po, key, _get(ps, key))
    return out


def swap_arms(
    p_real: dict[str, Any], p_legacy: dict[str, Any], groups: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    """Both directions, one group at a time, plus ``all``."""
    out: dict[str, dict[str, Any]] = {"P_real": p_real, "P_legacy": p_legacy}
    for g in groups:
        out[f"real_to_legacy__{g}"] = apply_profile_gains(p_real, p_legacy, g)
        out[f"legacy_to_real__{g}"] = apply_profile_gains(p_legacy, p_real, g)
    a, b = p_real, p_legacy
    for g in groups:
        a = apply_profile_gains(a, p_legacy, g)
        b = apply_profile_gains(b, p_real, g)
    out["real_to_legacy__all"] = a
    out["legacy_to_real__all"] = b
    return out


def run_score(
    *,
    out: Path,
    arms: dict[str, dict[str, Any] | str],
    recordings: tuple[str, ...],
    probe: bool,
    stem: str,
    ladder_arms: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Render every arm on the score windows and score the frozen HPPNet."""
    rs = _module("noise_v2_round_score")
    probe_obj = rs.Probe.load() if probe else None
    payload: dict[str, Any] = dict(
        schema=SCHEMA,
        study=stem,
        git=git_rev(),
        protocol=dict(
            seed=SEED,
            n_mics=8,
            scorer=(probe_obj.record if probe_obj is not None else None),
            arms={k: (v if isinstance(v, str) else v.get("_label", k)) for k, v in arms.items()},
        ),
        windows={},
    )
    for recording in recordings:
        support = scored_support(recording)
        clip = load_clip(support)
        sr = int(clip.sr)
        real = np.asarray(clip.audio, dtype=np.float64)[:8]
        rps = np.atleast_2d(np.asarray(clip.rps, dtype=np.float64))
        rsupport = RE.regime_support(
            support.window,
            rps,
            regime=support.regime,
            min_rps=support.min_rps,
            max_rps=support.max_rps,
            sr=sr,
        )
        row: dict[str, Any] = dict(support=support.as_dict(), arms={})
        for name, arm in arms.items():
            if arm == "real":
                audio = real
            elif arm == "legacy":
                audio = np.asarray(legacy_render(recording, rps), dtype=np.float64)[:8]
            else:
                assert isinstance(arm, dict)
                audio = np.asarray(
                    RD.render_noise(arm, rps, n_mics=8, seed=SEED), dtype=np.float64
                )[:8]
            entry: dict[str, Any] = {}
            if probe_obj is not None:
                pit = probe_obj.tracker.pit(
                    audio,
                    rps,
                    mics=list(range(8)),
                    expected_samples=int(real.shape[-1]),
                    support=rsupport,
                )
                entry["pit_mae"] = float(pit["mae"])
            if name in ladder_arms:
                entry |= ladder(audio, rps, sr=sr)
            row["arms"][name] = entry
            print(f"[{recording}] {name}: pit={entry.get('pit_mae')}", flush=True)
        payload["windows"][support.key] = row
    Path(out).mkdir(parents=True, exist_ok=True)
    return payload


def score_table(payload: dict[str, Any]) -> str:
    keys = list(payload["windows"])
    names = list(next(iter(payload["windows"].values()))["arms"])
    o = [
        "| arm | "
        + " | ".join(k.split("@")[0].replace("_nosource_room2", "") for k in keys)
        + " | mean |",
        "|---|" + "---:|" * (len(keys) + 1),
    ]
    for n in names:
        vals = [payload["windows"][k]["arms"][n].get("pit_mae") for k in keys]
        if any(v is None for v in vals):
            continue
        cells = " | ".join(f"{v:.3f}" for v in vals)
        o.append(f"| `{n}` | {cells} | {float(np.mean(vals)):.3f} |")
    return "\n".join(o)


# ── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("anatomy", help="(A) parameters and measured render anatomy")
    a.add_argument("--out", default=str(OUT_DEFAULT))
    a.add_argument("--recordings", default=",".join(RECORDINGS))
    a.add_argument("--figures", action="store_true")

    s = sub.add_parser("supports", help="(B) legacy render -> v2 supports")
    s.add_argument("--out", default=str(OUT_DEFAULT))
    s.add_argument("--cache", default=str(OUT_DEFAULT / "supports"))
    s.add_argument("--prefix", default="legacy_")
    s.add_argument(
        "--from-fit",
        default=str(FIT_REAL),
        help="take the support pool from this fit's own supports list",
    )

    c = sub.add_parser("score", help="(B3, C) render named fits and score HPPNet")
    c.add_argument("--out", default=str(OUT_DEFAULT))
    c.add_argument("--stem", default="score")
    c.add_argument("--probe", action="store_true")
    c.add_argument("--recordings", default=",".join(RECORDINGS))
    c.add_argument(
        "--fit",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="a v2 fit JSON to render and score; 'real' and 'legacy' are reserved arm names",
    )
    c.add_argument("--with-real", action="store_true", help="add the real and legacy arms")
    c.add_argument(
        "--ladder", default="", help="comma-separated arms to also measure the ladder on"
    )
    c.add_argument(
        "--swap",
        nargs=2,
        default=None,
        metavar=("P_REAL", "P_LEGACY"),
        help="two fit JSONs: score every one-group swap in both directions",
    )
    c.add_argument("--swap-groups", default=",".join(SWAP_GROUPS))

    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.cmd == "anatomy":
        recs = tuple(r for r in str(args.recordings).split(",") if r)
        payload = run_anatomy(out=out, recordings=recs)
        figures = payload.pop("_figures")
        written = write_ladder_figure(figures, out) if args.figures else []
        payload["figures"] = written
        (out / "anatomy.json").write_text(json.dumps(payload, indent=1) + "\n")
        (out / "anatomy.md").write_text(anatomy_findings(payload) + "\n")
        print(f"# wrote {out / 'anatomy.json'} and {out / 'anatomy.md'}")
        for p in written:
            print(f"# wrote {p}")
        return 0

    if args.cmd == "supports":
        specs = [
            SU.parse_spec(_spec_of(name)).text
            for name in json.loads(Path(args.from_fit).read_text())["supports"]
        ]
        payload = run_supports(
            out=out, cache=Path(args.cache), specs=specs, prefix=str(args.prefix)
        )
        (out / "supports.json").write_text(json.dumps(payload, indent=1) + "\n")
        print(f"# wrote {out / 'supports.json'}")
        return 0

    recs = tuple(r for r in str(args.recordings).split(",") if r)
    arms: dict[str, dict[str, Any] | str] = {}
    if args.with_real:
        arms["real"] = "real"
        arms["legacy"] = "legacy"
    for entry in args.fit:
        name, _, path = str(entry).partition("=")
        if not path:
            die(f"--fit wants NAME=PATH, got {entry!r}")
        arms[name] = json.loads(Path(path).read_text())
    if args.swap:
        p_real = json.loads(Path(args.swap[0]).read_text())
        p_legacy = json.loads(Path(args.swap[1]).read_text())
        groups = tuple(g for g in str(args.swap_groups).split(",") if g)
        arms |= swap_arms(p_real, p_legacy, groups)
    if not arms:
        die("nothing to score: pass --fit, --swap or --with-real")
    payload = run_score(
        out=out,
        arms=arms,
        recordings=recs,
        probe=bool(args.probe),
        stem=str(args.stem),
        ladder_arms=tuple(g for g in str(args.ladder).split(",") if g),
    )
    (out / f"{args.stem}.json").write_text(json.dumps(payload, indent=1) + "\n")
    print(score_table(payload))
    print(f"# wrote {out / f'{args.stem}.json'}")
    return 0


def _spec_of(support_name: str) -> str:
    """``flight_dregon_<rec>@<start>+<dur>_<key>`` -> its spec text."""
    stem = str(support_name)
    if not stem.startswith("flight_dregon_"):
        die(f"{support_name!r}: only DREGON flight supports are handled")
    body = stem[len("flight_dregon_") :]
    rec, _, rest = body.partition("@")
    start, _, rest2 = rest.partition("+")
    dur, _, key = rest2.partition("_")
    return f"flight_dregon:{rec}:{start}:{dur}:{key}"


if __name__ == "__main__":
    raise SystemExit(main())
