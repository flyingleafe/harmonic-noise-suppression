#!/usr/bin/env python
"""Why the round-1 DREGON bench fits carry no comb: registration, scale, init.

    # the 21-support window-level census (which supports are post-spin-down silence)
    python scripts/noise_v2_bench_diag.py census
    # per-support geometry / scale / registration / line-SNR / initialiser
    python scripts/noise_v2_bench_diag.py diag
    # objective decomposition: fitted vs floor-only vs oracle comb, per carrier
    python scripts/noise_v2_bench_diag.py objective
    # the four figures
    python scripts/noise_v2_bench_diag.py figures
    python scripts/noise_v2_bench_diag.py all

Everything lands in ``results/noise_v2/rounds/round1/bench_diag/``. Read-only
with respect to the frozen supports index and the committed fits; the support
``.npz`` caches are (re)built through :func:`supports.load_support`, which
reproduces ``index.json`` exactly (verified: segment, carrier, ``n_fft``).

``PYTHONPATH=src``. Three 7-21 s supports, a handful of periodograms and a few
hundred forward evaluations: this runs on a laptop in minutes, no submission.

UNITS, because the round-1 fit JSONs are read wrongly without them.
``profile_db`` is the per-order line VARIANCE in the support's periodogram
units, not a periodogram level: the line's PEAK bin sits
``+10 log10(max_j lines_unit_j)`` above it, where ``lines_unit`` is the model's
own unit-profile line response (``+46 dB`` at ``k = 1`` on a 7.3 s support,
falling to ``+23 dB`` by ``k = 80`` as the shaft jitter spreads the line). A
``profile_db`` of -88 dB against a ``floor_mean_db`` of -77 dB is therefore a
line ~34 dB ABOVE the floor, not 11 dB below it. Every level this script
reports is a periodogram level in dB and is directly comparable to the data.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from scipy.ndimage import median_filter  # noqa: E402

from experiments.noise_model import fit as FT  # noqa: E402
from experiments.noise_model import model as MD  # noqa: E402
from experiments.noise_model import spectrum as SP  # noqa: E402
from experiments.noise_model import supports as SU  # noqa: E402
from experiments.stochastic_fit import clips as C  # noqa: E402
from experiments.stochastic_fit.revised_phase import composite_risk  # noqa: E402

OUT_DIR = Path("results/noise_v2/rounds/round1/bench_diag")
INDEX = Path("results/noise_v2/rounds/round1/supports/index.json")
FITS = Path("results/noise_v2/rounds/round1/fits")
SCHEMA = "noise-v2-bench-diag/1"

#: The three supports of the assignment plus the motor-on control window.
TARGETS = ("bench_dregon_Motor1_70", "bench_dregon_Motor2_60", "bench_dregon_Motor1_80")
CONTROL_SUFFIX = "__motor_on"

#: Registration grid: +-1.5 rev/s around the index carrier, 0.002 rev/s.
REG_HALF_WIDTH = 1.5
REG_STEP = 0.002
#: Local-floor annulus of the per-order line SNR, in bins.
FLOOR_BINS = 50
#: Half-width of the line window, in bins plus a jitter term. ``spec`` is the
#: assignment's literal ``2 bins + 0.3 k residual_std``, which reads
#: ``residual_std`` as the index's per-ORDER residual (0.15 Hz at k~62) and so
#: over-widens the window by the order itself; ``tight`` is bins only and
#: ``jitter`` uses the per-REV residual ``residual_std / order``, which is the
#: line's own smear at order k. All three are reported.
WINDOWS = ("spec", "tight", "jitter")
JITTER_SIGMAS = 3.0
#: Cap on one order's contribution to the capped registration score, in dB.
#: Four fixed-frequency interferer lines at 22-28 dB would otherwise decide a
#: 103-order sum in which the rotor's own lines carry 2-9 dB.
SNR_CAP_DB = 10.0
LINE_SNR_THRESHOLD_DB = 6.0
ENVELOPE_S = 0.5


# ── small helpers ───────────────────────────────────────────────────────────


def index_rows() -> dict[str, dict[str, Any]]:
    idx = json.loads(INDEX.read_text())
    return {r["name"]: r for r in idx["sets"]["dregon-bench"]["supports"]}


def fit_json(name: str) -> dict[str, Any] | None:
    path = FITS / f"{name}__bench.json"
    return json.loads(path.read_text()) if path.exists() else None


def envelope_db(audio: np.ndarray, sr: float, step_s: float = ENVELOPE_S) -> np.ndarray:
    """Mic-mean RMS in dB on a ``step_s`` grid — the motor's on/off trace."""
    x = np.asarray(audio, dtype=np.float64)
    x = x.mean(axis=0) if x.ndim > 1 else x
    w = int(round(step_s * sr))
    n = x.size // w
    return 20.0 * np.log10(np.sqrt((x[: n * w].reshape(n, w) ** 2).mean(axis=1)) + 1e-20)


def loudest_window(
    env: np.ndarray, dur_s: float, step_s: float = ENVELOPE_S
) -> tuple[float, float]:
    """``(start_s, mean_db)`` of the loudest ``dur_s`` window on the grid."""
    nb = max(1, int(round(dur_s / step_s)))
    if env.size <= nb:
        return 0.0, float(env.mean())
    means = np.array([env[i : i + nb].mean() for i in range(env.size - nb + 1)])
    i = int(np.argmax(means))
    return i * step_s, float(means[i])


def _fwhm_hz(arr: np.ndarray, j: int, span: int, df: float) -> float:
    """Half-power width in Hz of the feature at bin ``j``, searched +-``span``.

    The peak is taken over +-``span`` so a model line that is centred a bin or
    two away is still measured at its own peak; the width is the run of bins
    around that peak above half its power.
    """
    a0, b0 = max(0, j - span), min(arr.size, j + span + 1)
    jp = a0 + int(np.argmax(arr[a0:b0]))
    thr = arr[jp] / 2.0
    a = jp
    while a > a0 and arr[a - 1] > thr:
        a -= 1
    b = jp
    while b < b0 - 1 and arr[b + 1] > thr:
        b += 1
    return float((b - a + 1) * df)


def bgrid(batch: MD.SupportBatch) -> SP.BenchGrid:
    """The batch's bench grid, narrowed for the type checker."""
    assert isinstance(batch.grid, SP.BenchGrid), f"{batch.name} is not a bench batch"
    return batch.grid


def band_of(support: SU.Support) -> np.ndarray:
    return SP.bench_band(support.freqs_hz, support.sr)


def mic_mean(power: np.ndarray) -> np.ndarray:
    return np.asarray(power, dtype=np.float64)[:, 0, :].mean(axis=0)


def db(v: Any) -> float:
    return float(10.0 * np.log10(np.maximum(np.asarray(v, dtype=np.float64), 1e-300)))


# ── supports ────────────────────────────────────────────────────────────────


def load_target(name: str, rows: dict[str, dict[str, Any]]) -> tuple[SU.Support, dict[str, Any]]:
    """One target support plus the provenance of the window it covers.

    ``<name>__motor_on`` is the CONTROL: the loudest window of the same length
    in the same recording, on a 0.5 s grid. Same loader, same decimation, same
    one-frame periodogram — the only difference is where in the recording the
    window sits.
    """
    base = name.removesuffix(CONTROL_SUFFIX)
    row = rows[base]
    if not name.endswith(CONTROL_SUFFIX):
        return SU.load_support(row["spec"]), dict(window="index", segment=row["segment"])
    rec = C.load_recording(SU.DREGON_DATASET, row["recording_id"], None, SU.DREGON_RPS_KEY)
    dur = float(row["duration_s"])
    start, level = loudest_window(envelope_db(rec.audio, rec.sr), dur)
    sup = SU.bench_support(
        name,
        rec.audio,
        rec.sr,
        [float(row["carriers_rev_s"][0])],
        segment=(start, start + dur),
        meta=dict(row["spec"] and {"spec": row["spec"]}, control="loudest same-length window"),
    )
    return sup, dict(window="loudest_same_length", segment=[start, start + dur], level_db=level)


def batch_for(support: SU.Support, *, k_cap: int = 130) -> MD.SupportBatch:
    """The batch the fit driver builds (``scripts/noise_v2_fit.py:74-80``)."""
    return MD.bench_batch(
        name=support.name,
        power=np.asarray(support.power, dtype=np.float64),
        sr=int(support.sr),
        carrier_mean=np.asarray(support.carrier_rev_s, dtype=np.float64).mean(axis=1),
        k_cap=k_cap,
    )


# ── registration ────────────────────────────────────────────────────────────


def half_width_bins(k: int, df: float, kind: str, sigma_hz: float, sigma_rev: float) -> int:
    if kind == "tight":
        extra = 0.0
    elif kind == "spec":
        extra = 0.3 * k * sigma_hz
    elif kind == "jitter":
        extra = JITTER_SIGMAS * k * sigma_rev
    else:
        raise ValueError(kind)
    return int(math.ceil((2.0 * df + extra) / df))


def line_table(
    pm: np.ndarray,
    med: np.ndarray,
    df: float,
    f0: float,
    ks: np.ndarray,
    *,
    kind: str,
    sigma_hz: float,
    sigma_rev: float,
) -> dict[str, Any]:
    """Per-order peak level, local floor, SNR and peak position at carrier ``f0``."""
    peak_db, floor_db, snr_db, peak_hz = [], [], [], []
    for k in ks:
        half = half_width_bins(int(k), df, kind, sigma_hz, sigma_rev)
        j = int(round(k * f0 / df))
        a, b = max(0, j - half), min(pm.size, j + half + 1)
        if a >= b:
            continue
        jj = a + int(np.argmax(pm[a:b]))
        peak_db.append(db(pm[jj]))
        floor_db.append(db(med[min(j, med.size - 1)]))
        snr_db.append(peak_db[-1] - floor_db[-1])
        peak_hz.append(float(jj * df))
    snr = np.asarray(snr_db)
    kk = np.asarray(ks[: snr.size], dtype=np.int64)
    implied = np.asarray(peak_hz) / np.maximum(kk, 1)
    strong = snr >= LINE_SNR_THRESHOLD_DB
    return dict(
        carrier_rev_s=float(f0),
        window=kind,
        k=kk.tolist(),
        peak_db=[round(v, 3) for v in peak_db],
        floor_db=[round(v, 3) for v in floor_db],
        snr_db=[round(v, 3) for v in snr_db],
        n_orders=int(kk.size),
        n_above_6db=int(strong.sum()),
        snr_sum_db=round(float(snr.sum()), 2),
        snr_sum_capped_db=round(float(np.minimum(snr, SNR_CAP_DB).sum()), 2),
        snr_median_db=round(float(np.median(snr)), 3),
        implied_carrier_rev_s=(
            round(float(np.median(implied[strong])), 5) if strong.any() else None
        ),
        implied_carrier_strong_k=kk[strong].tolist(),
    )


def score_curve(
    pm: np.ndarray,
    med: np.ndarray,
    df: float,
    grid: np.ndarray,
    ks: np.ndarray,
    *,
    kind: str,
    sigma_hz: float,
    sigma_rev: float,
    cap_db: float | None,
) -> np.ndarray:
    """``S(f) = sum_k 10log10(max P in the order window)``, optionally as a
    floor-normalised per-order SNR capped at ``cap_db``."""
    from scipy.ndimage import maximum_filter1d

    s = np.zeros(grid.size)
    for k in ks:
        half = half_width_bins(int(k), df, kind, sigma_hz, sigma_rev)
        mx = maximum_filter1d(pm, size=2 * half + 1, mode="nearest")
        idx = np.clip(np.rint(k * grid / df).astype(np.int64), 0, pm.size - 1)
        v = 10.0 * np.log10(np.maximum(mx[idx], 1e-300))
        if cap_db is not None:
            v = np.minimum(v - 10.0 * np.log10(np.maximum(med[idx], 1e-300)), cap_db)
        s += v
    return s


def registration(
    support: SU.Support, row: dict[str, Any], fit: dict[str, Any] | None
) -> dict[str, Any]:
    pm = mic_mean(support.power)
    med = median_filter(pm, size=2 * FLOOR_BINS + 1, mode="nearest")
    df = float(support.freqs_hz[1] - support.freqs_hz[0])
    f_index = float(row["carriers_rev_s"][0])
    sigma_hz = float(row["residual_std_hz"][0])
    sigma_rev = sigma_hz / max(1, int(row["orders"][0]))
    f_top = min(SP.BAND_F_MAX, 0.45 * support.sr)
    lo, hi = f_index - REG_HALF_WIDTH, f_index + REG_HALF_WIDTH
    ks = np.arange(max(1, int(math.ceil(SP.BAND_F_MIN / lo))), int(math.floor(f_top / hi)) + 1)
    grid = f_index + np.arange(-REG_HALF_WIDTH, REG_HALF_WIDTH + 1e-12, REG_STEP)

    carriers = {"index": f_index, "survey": float(row["survey_rev_s"][0])}
    if fit is not None:
        carriers["fit_refined"] = float(fit["diagnostics"]["batch"]["carrier_init_rev_s"][0])
        carriers["fitted"] = float(fit["params"]["carrier_rev_s"][0])

    variants: dict[str, Any] = {}
    for kind in WINDOWS:
        for cap, tag in ((None, "raw"), (SNR_CAP_DB, "capped_snr")):
            s = score_curve(
                pm, med, df, grid, ks, kind=kind, sigma_hz=sigma_hz, sigma_rev=sigma_rev, cap_db=cap
            )
            i = int(np.argmax(s))
            at = {
                nm: round(
                    float(
                        score_curve(
                            pm,
                            med,
                            df,
                            np.array([f]),
                            ks,
                            kind=kind,
                            sigma_hz=sigma_hz,
                            sigma_rev=sigma_rev,
                            cap_db=cap,
                        )[0]
                    ),
                    2,
                )
                for nm, f in carriers.items()
            }
            variants[f"{kind}__{tag}"] = dict(
                argmax_rev_s=round(float(grid[i]), 5),
                S_argmax=round(float(s[i])),
                S_at=at,
                curve_step_rev_s=0.01,
                curve=[round(float(v), 2) for v in s[::5]],
                curve_f0=round(float(grid[0]), 5),
            )

    lines = {
        nm: line_table(pm, med, df, f, ks, kind="jitter", sigma_hz=sigma_hz, sigma_rev=sigma_rev)
        for nm, f in carriers.items()
    }
    return dict(
        carriers_rev_s={k: round(v, 6) for k, v in carriers.items()},
        residual_std_hz=sigma_hz,
        residual_std_rev_s=round(sigma_rev, 8),
        index_order=int(row["orders"][0]),
        orders_scored=[int(ks[0]), int(ks[-1])],
        grid=dict(half_width_rev_s=REG_HALF_WIDTH, step_rev_s=REG_STEP, n=int(grid.size)),
        scores=variants,
        lines=lines,
        carrier_prior_sigmas={
            nm: round((f - float(row["survey_rev_s"][0])) / 0.5, 3) for nm, f in carriers.items()
        },
        interferer=interferer_comb(pm, med, df, support.sr, carriers, sigma_rev=sigma_rev),
    )


def interferer_comb(
    pm: np.ndarray,
    med: np.ndarray,
    df: float,
    sr: int,
    carriers: dict[str, float],
    *,
    sigma_rev: float = 0.0,
) -> dict[str, Any]:
    """The loudest narrow lines and whether a fitted carrier aliases onto them.

    A fixed-frequency interferer at a constant absolute level dominates a
    support whose motor is off, and any carrier ``f`` with ``k f`` on such a
    line collects 20-28 dB where the rotor's own orders carry 2-9 dB.
    """
    band = SP.bench_band(np.arange(pm.size) * df, sr)
    ratio = pm / np.maximum(med, 1e-300)
    inb = np.where(band)[0]
    top = np.sort(inb[np.argsort(ratio[inb])[::-1][:400]])
    groups: list[list[int]] = [[int(top[0])]]
    for j in top[1:]:
        (groups[-1].append(int(j)) if j - groups[-1][-1] <= 4 else groups.append([int(j)]))
    peaks = sorted(
        ((float(pm[g].max()), int(g[int(np.argmax(pm[g]))])) for g in groups), reverse=True
    )[:8]
    lines = [
        dict(f_hz=round(j * df, 3), level_db=round(db(pm[j]), 2), snr_db=round(db(ratio[j]), 2))
        for _, j in peaks
    ]
    alias = {}
    for nm, f in carriers.items():
        hits = []
        for ln in lines:
            k = ln["f_hz"] / f
            kk = int(round(k))
            # the SAME window the per-order line table uses, so a carrier whose
            # order k is inside that order's own read window counts as a hit
            tol = 2.0 * df + JITTER_SIGMAS * kk * sigma_rev
            if kk >= 1 and abs(kk * f - ln["f_hz"]) <= tol:
                hits.append(
                    dict(
                        k=kk,
                        f_hz=ln["f_hz"],
                        snr_db=ln["snr_db"],
                        offset_hz=round(abs(kk * f - ln["f_hz"]), 3),
                        offset_bins=round(abs(kk * f - ln["f_hz"]) / df, 2),
                    )
                )
        alias[nm] = hits
    return dict(loudest_lines=lines, aliased_orders=alias)


def refine_score_audit(
    support: SU.Support, row: dict[str, Any], fit: dict[str, Any] | None
) -> dict[str, Any]:
    """``refine_bench_carrier``'s OWN score at each candidate carrier.

    The score is ``sum_{k >= k_min} log P(k f)`` at the nearest bin, exactly as
    :func:`spectrum.refine_bench_carrier` computes it. Two forms are reported:
    the function's own (whose highest order is ``floor(f_top / f)``, so the
    number of summands moves with ``f``) and a common-order form that sums the
    same ``k`` range for every candidate. Whether the refinement's mode is a
    genuinely higher score or an artefact of the moving order count is then a
    fact on the record rather than an inference.
    """
    p = mic_mean(support.power)
    df = float(support.freqs_hz[1] - support.freqs_hz[0])
    lp = np.log(np.maximum(p, 1e-24))
    f_top = min(SP.BAND_F_MAX, 0.45 * float(support.sr))
    k_min = 4

    def own(f0: float, k_top: int | None = None) -> tuple[float, int]:
        kt = int(math.floor(f_top / f0)) if k_top is None else int(k_top)
        if kt < k_min:
            return float("nan"), 0
        idx = np.rint(np.arange(k_min, kt + 1) * f0 / df).astype(np.int64)
        ok = (idx > 0) & (idx < lp.size)
        return float(lp[idx[ok]].sum()), int(ok.sum())

    best, diag = SP.refine_bench_carrier(
        np.asarray(support.power, dtype=np.float64)[:, 0, :],
        support.freqs_hz,
        float(row["survey_rev_s"][0]),
        sr=int(support.sr),
    )
    from_index, diag_index = SP.refine_bench_carrier(
        np.asarray(support.power, dtype=np.float64)[:, 0, :],
        support.freqs_hz,
        float(row["carriers_rev_s"][0]),
        sr=int(support.sr),
        half_width_rev_s=0.35,
        n_grid=1401,
    )
    cands = {
        "survey": float(row["survey_rev_s"][0]),
        "index": float(row["carriers_rev_s"][0]),
        "coarse": float(diag["coarse_rev_s"]),
        "refined_from_survey": float(best),
        "refined_from_index_tight": float(from_index),
    }
    if fit is not None:
        cands["fitted"] = float(fit["params"]["carrier_rev_s"][0])
    k_common = min(int(math.floor(f_top / f)) for f in cands.values())
    scores = {}
    for nm, f0 in cands.items():
        s_own, n_own = own(f0)
        s_com, n_com = own(f0, k_common)
        scores[nm] = dict(
            carrier_rev_s=round(f0, 6),
            score=round(s_own, 3),
            n_orders=n_own,
            score_common_orders=round(s_com, 3),
            n_orders_common=n_com,
        )
    return dict(
        k_min=k_min,
        k_common=k_common,
        note="sum of log P at the nearest bin of k f, refine_bench_carrier's own score",
        harmonic_score_reported=float(diag["harmonic_score"]),
        shift_from_survey_rev_s=float(diag["shift_rev_s"]),
        shift_from_index_tight_rev_s=float(diag_index["shift_rev_s"]),
        scores=scores,
        score_of_refined_minus_index=round(
            scores["refined_from_survey"]["score_common_orders"]
            - scores["index"]["score_common_orders"],
            3,
        ),
    )


# ── scale ───────────────────────────────────────────────────────────────────


def scale_block(
    support: SU.Support, row: dict[str, Any], fit: dict[str, Any] | None, *, control_dur_s: float
) -> dict[str, Any]:
    """Data levels, raw-audio RMS and the length control of the same segment."""
    pm = mic_mean(support.power)
    band = band_of(support)
    per_mic = np.asarray(support.power, dtype=np.float64)[:, 0, :][:, band]
    rec = C.load_recording(SU.DREGON_DATASET, row["recording_id"], None, SU.DREGON_RPS_KEY)
    sr = float(rec.sr)
    x = np.asarray(rec.audio, dtype=np.float64)
    i0, i1 = int(round(support.segment[0] * sr)), int(round(support.segment[1] * sr))
    seg = x[:, i0:i1]
    env = envelope_db(x, sr)
    dur = float(support.segment[1] - support.segment[0])
    ls, ll = loudest_window(env, dur)
    _, l7 = loudest_window(env, control_dur_s)
    e0, e1 = (
        int(support.segment[0] / ENVELOPE_S),
        max(int(support.segment[0] / ENVELOPE_S) + 1, int(support.segment[1] / ENVELOPE_S)),
    )
    # length control: the SAME audio start, a shorter window, same loader
    short = SU.bench_support(
        f"{support.name}__len_control",
        x,
        sr,
        [float(row["carriers_rev_s"][0])],
        segment=(support.segment[0], min(support.segment[0] + control_dur_s, x.shape[-1] / sr)),
        meta={},
    )
    p_short = mic_mean(short.power)
    out = dict(
        data_median_in_band_db=round(db(np.median(pm[band])), 3),
        data_median_per_mic_db=[round(db(np.median(v)), 2) for v in per_mic],
        data_mean_in_band_db=round(db(pm[band].mean()), 3),
        audio_segment_rms_db=round(20.0 * np.log10(np.sqrt((seg**2).mean())), 3),
        audio_segment_rms_per_mic_db=[
            round(20.0 * np.log10(np.sqrt((v**2).mean())), 2) for v in seg
        ],
        audio_recording_rms_db=round(20.0 * np.log10(np.sqrt((x**2).mean())), 3),
        recording_duration_s=round(x.shape[-1] / sr, 3),
        envelope_step_s=ENVELOPE_S,
        envelope_db=[round(float(v), 2) for v in env],
        window_envelope_mean_db=round(float(env[e0:e1].mean()), 2),
        loudest_same_length_window=dict(start_s=ls, mean_db=round(ll, 2)),
        loudest_short_window=dict(dur_s=control_dur_s, mean_db=round(l7, 2)),
        level_deficit_db=round(float(ll - env[e0:e1].mean()), 2),
        length_control=dict(
            note="same segment start, shorter window, same loader/periodogram",
            dur_s=round(float(short.segment[1] - short.segment[0]), 4),
            n_fft=int(short.n_fft),
            median_in_band_db=round(db(np.median(p_short[band_of(short)])), 3),
            level_change_db=round(
                db(np.median(p_short[band_of(short)])) - db(np.median(pm[band])), 3
            ),
            length_ratio_db=round(10.0 * np.log10(short.n_fft / support.n_fft), 3),
        ),
    )
    if fit is not None:
        out["fit_init_floor_mean_db"] = fit["diagnostics"]["init_floor_mean_db"]
        out["fit_floor_mean_db"] = fit["params"]["floor"]["floor_mean_db"]
        out["fit_floor_level_db"] = fit["diagnostics"]["floor_level_db"]
        out["fit_floor_mean_moved_from_init"] = bool(
            fit["diagnostics"]["init_floor_mean_db"] != fit["params"]["floor"]["floor_mean_db"]
        )
    return out


# ── model / objective ───────────────────────────────────────────────────────


def with_profile(par: SP.V2Params, level_db: float) -> SP.V2Params:
    prof = torch.full_like(torch.as_tensor(par.profile_db, dtype=torch.float64), float(level_db))
    return dataclasses.replace(par, profile_db=prof)


def objective_of(
    batch: MD.SupportBatch,
    par: SP.V2Params,
    *,
    k_max: int | None = None,
    groups: Any = None,
) -> dict[str, Any]:
    kk = batch.k_max if k_max is None else int(k_max)
    gg = batch.bench_order_groups if groups is None else groups
    with torch.no_grad():
        m = SP.bench_model(bgrid(batch), par, k_max=kk, groups=gg)
        total = float(MD.whittle_risk(batch, m))
        comb = float(composite_risk(batch.power, m, batch.weights, band=batch.band_hi))
        floor = float(composite_risk(batch.power, m, batch.weights, band=batch.band_lo))
    n_comb = int(batch.power.shape[0]) * int(batch.power.shape[1]) * int(batch.band_hi.sum())
    return dict(
        whittle_nats=total,
        nats_per_cell=round(total / batch.n_cells, 5),
        comb_band_nats=comb,
        comb_band_nats_per_cell=round(comb / n_comb, 5),
        floor_band_nats=floor,
        n_cells=batch.n_cells,
        n_cells_comb=n_comb,
    )


def lines_unit_response(
    batch: MD.SupportBatch, par: SP.V2Params, f0: float, *, k_cap: int = 130
) -> tuple[SP.V2Params, int, Any, np.ndarray, np.ndarray]:
    """``(base params at f0, k_max, groups, floor-only spectrum, unit-line spectrum)``."""
    sr = bgrid(batch).sr
    kk = int(SP.k_max_for_carrier(np.array([f0]), sr, k_cap=k_cap))
    width = max(kk, int(np.asarray(par.profile_db).shape[1]))
    base = dataclasses.replace(
        par,
        carrier_rev_s=torch.as_tensor([float(f0)], dtype=torch.float64),
        profile_db=torch.zeros(1, width, dtype=torch.float64),
    )
    groups = SP.order_groups(
        kk,
        sigma_nu=math.exp(MD.PRIORS.log_sigma_nu[0]),
        lam=math.exp(MD.PRIORS.log_lam[0]),
        sr=sr,
        n=bgrid(batch).n,
    )
    with torch.no_grad():
        quiet = (
            SP.bench_model(bgrid(batch), with_profile(base, -300.0), k_max=kk, groups=groups)
            .mean(dim=(0, 1))
            .numpy()
        )
        unit = (
            SP.bench_model(bgrid(batch), with_profile(base, 0.0), k_max=kk, groups=groups)
            .mean(dim=(0, 1))
            .numpy()
        )
    return base, kk, groups, quiet, np.maximum(unit - quiet, 1e-300)


def oracle_comb(
    batch: MD.SupportBatch,
    par: SP.V2Params,
    f0: float,
    sigma_rev: float,
    *,
    mode: str = "energy",
    k_cap: int = 130,
) -> dict[str, Any]:
    """The comb read straight off the data at carrier ``f0``.

    Per in-band order, the excess of the mic-mean periodogram over the model's
    own floor is matched to the model's unit-profile line response — the
    initialiser's construction (``fit.initial_values``), with the line window
    widened to the order's own jitter and with the carrier supplied rather than
    refined. ``energy`` matches the window's summed excess (a matched-energy
    estimate, which does not over-inject at the high orders whose line is a
    wide pedestal); ``peak`` matches the peak bin, which is what the
    initialiser does.
    """
    base, kk, groups, quiet, lines_unit = lines_unit_response(batch, par, f0, k_cap=k_cap)
    obs = batch.power.numpy().mean(axis=(0, 1))
    excess = obs - quiet
    df = float(bgrid(batch).freqs_hz[1] - bgrid(batch).freqs_hz[0])
    f_top = float(bgrid(batch).freqs_hz[np.asarray(batch.band)].max())
    width = int(np.asarray(base.profile_db).shape[1])
    prof = np.full((1, width), -300.0)
    measured: dict[int, float] = {}
    for k in range(1, kk + 1):
        fk = k * f0
        if fk < SP.BAND_F_MIN or fk > f_top:
            continue
        half = int(math.ceil((2.0 * df + JITTER_SIGMAS * k * sigma_rev) / df))
        j = int(round(fk / df))
        a, b = max(0, j - half), min(obs.size, j + half + 1)
        if mode == "energy":
            num, den = float(excess[a:b].sum()), float(lines_unit[a:b].sum())
        else:
            num, den = float(np.maximum(excess[a:b], 1e-30).max()), float(lines_unit[a:b].max())
        if den > 0.0 and num > 0.0:
            v = float(np.clip(10.0 * math.log10(num / den), -120.0, 40.0))
            prof[0, k - 1] = v
            measured[k] = v
    par_o = dataclasses.replace(base, profile_db=torch.as_tensor(prof, dtype=torch.float64))
    best = None
    for shift in np.arange(-24.0, 12.001, 2.0):
        pp = np.where(prof > -299.0, prof + shift, prof)
        obj = objective_of(
            batch,
            dataclasses.replace(par_o, profile_db=torch.as_tensor(pp, dtype=torch.float64)),
            k_max=kk,
            groups=groups,
        )
        if best is None or obj["nats_per_cell"] < best[1]["nats_per_cell"]:
            best = (float(shift), obj, pp)
    assert best is not None
    raw = objective_of(batch, par_o, k_max=kk, groups=groups)
    peak_scale = {
        str(k): round(db(lines_unit[int(round(k * f0 / df))]), 2)
        for k in (1, 2, 5, 10, 20, 40, 80)
        if int(round(k * f0 / df)) < lines_unit.size
    }
    return dict(
        carrier_rev_s=round(float(f0), 6),
        mode=mode,
        k_max=kk,
        n_orders_measured=len(measured),
        profile_db=[round(float(v), 3) for v in prof[0]],
        profile_db_k1_10=[round(float(v), 2) for v in prof[0][:10]],
        objective=raw,
        best_offset_db=best[0],
        objective_at_best_offset=best[1],
        peak_scale_db=peak_scale,
    )


def initialiser_block(batch: MD.SupportBatch, sigma_rev: float) -> dict[str, Any]:
    """What ``fit.initial_values`` actually reads on this support."""
    init = FT.initial_values(batch, mode="bench")
    prof = init["profile_db"].numpy()[0]
    seed = MD.sample_params_from_values(batch, mode="bench", values=init)
    f_init = float(np.asarray(init["carrier_rev_s"])[0])
    _, kk, _, _, lines_unit = lines_unit_response(batch, seed, f_init)
    df = float(bgrid(batch).freqs_hz[1] - bgrid(batch).freqs_hz[0])
    obs = batch.power.numpy().mean(axis=(0, 1))
    med = median_filter(obs, size=2 * FLOOR_BINS + 1, mode="nearest")
    peak_pred, peak_data, snr_data = [], [], []
    for k in range(1, min(kk, prof.size) + 1):
        j = int(round(k * f_init / df))
        if j >= lines_unit.size:
            break
        peak_pred.append(round(float(prof[k - 1] + db(lines_unit[j])), 2))
        half = int(math.ceil((2.0 * df + JITTER_SIGMAS * k * sigma_rev) / df))
        a, b = max(0, j - half), min(obs.size, j + half + 1)
        peak_data.append(round(db(obs[a:b].max()), 2))
        snr_data.append(round(peak_data[-1] - db(med[j]), 2))
    # half-power width of the model's line against the data's, at the SAME
    # carrier and dynamics: the line the model draws at the prior-centre
    # dynamics is much wider than the one in the data
    with torch.no_grad():
        m_init = MD.forward(batch, seed).mean(dim=(0, 1)).numpy()
    widths = {}
    for k in (2, 5, 10, 20, 40):
        j = int(round(k * f_init / df))
        span = int(math.ceil(30.0 / df))
        if j - span < 0 or j + span >= obs.size:
            continue
        widths[str(k)] = dict(
            model_fwhm_hz=round(_fwhm_hz(m_init, j, span, df), 3),
            data_fwhm_hz=round(_fwhm_hz(obs, j, span, df), 3),
        )
    return dict(
        carrier_rev_s=round(f_init, 6),
        floor_mean_db=round(float(init["floor_mean_db"]), 4),
        profile_db=[round(float(v), 3) for v in prof],
        profile_db_k1_10=[round(float(v), 2) for v in prof[:10]],
        profile_db_min=round(float(prof.min()), 2),
        profile_db_max=round(float(prof.max()), 2),
        profile_db_median=round(float(np.median(prof)), 2),
        predicted_line_peak_db=peak_pred,
        predicted_line_peak_db_k1_10=peak_pred[:10],
        # the SAME orders read off the data at the SAME (init) carrier, so the
        # initialiser's profile can be checked without a carrier mismatch
        data_line_peak_db=peak_data,
        data_line_peak_db_k1_10=peak_data[:10],
        data_line_snr_db_k1_10=snr_data[:10],
        line_fwhm_hz=widths,
        line_fwhm_dynamics=dict(
            sigma_nu=float(np.asarray(seed.sigma_nu)), lam=float(np.asarray(seed.lam))
        ),
        init_window_half_bins=3,
        note=(
            "the initialiser reads +-3 bins around k * carrier_init; "
            f"one bin is {df:.6f} Hz and the order's own jitter is "
            f"{sigma_rev:.6f} rev/s * k"
        ),
    )


# ── patch 1, run offline ────────────────────────────────────────────────────


def patched_stationary_segment(
    audio: np.ndarray, sr: float, survey_rev_s: list[float], *, level_tol_db: float = 6.0
) -> dict[str, Any]:
    """:func:`supports.stationary_segment` PLUS the in-window level test.

    A verbatim copy of the approved rule with the two changes patch 1 proposes:
    the window must also carry the motor (mic-mean power within
    ``level_tol_db`` of the recording's loudest window on the same smoothing
    grid), and the fallback ranks only windows that do. Run here rather than in
    ``supports.py`` so the frozen index keeps building bit-identically while
    the fix is being reviewed.
    """
    from utils.demod import demodulate, residual_frequency

    x = np.asarray(audio, dtype=np.float64)
    x = x[None, :] if x.ndim == 1 else x
    n = x.shape[-1]
    edge = int(round(SU.BENCH_EDGE_S * sr))
    smooth = int(round(SU.BENCH_RESIDUAL_SMOOTH_S * sr))
    inner = slice(edge, n - edge)
    orders, carriers, margins, narrow, wide = [], [], [], [], []
    for f_survey in survey_rev_s:
        k, margin_db = SU.select_order(x, sr, float(f_survey))
        f0 = SU.refine_carrier(x[0], sr, float(f_survey), k)
        carrier = np.full(n, f0)
        res = []
        for band in (SU.BENCH_DEMOD_BAND_HZ, SU.BENCH_WIDE_BAND_FRAC * f0):
            z = demodulate(x, carrier, float(band), sr, order=k)
            mm = np.mean(residual_frequency(z, sr, smooth_s=0.0), axis=0)
            res.append(SU._moving_mean(mm, smooth)[inner])
        orders.append(k)
        carriers.append(f0)
        margins.append(margin_db)
        narrow.append(res[0])
        wide.append(res[1])
    worst = np.max(np.abs(np.stack(narrow)), axis=0)
    worst_wide = np.max(np.abs(np.stack(wide)), axis=0)
    level = SU._moving_mean(np.mean(x**2, axis=0), smooth)[inner]
    level_db = 10.0 * np.log10(np.maximum(level, 1e-30))
    motor_on = level_db >= level_db.max() - float(level_tol_db)
    inside = (worst <= SU.BENCH_RESIDUAL_TOL_HZ) & (worst_wide <= SU.BENCH_WIDE_TOL_HZ) & motor_on
    a, b = SU._longest_run(inside)
    longest = (b - a) / sr
    if longest < SU.BENCH_MIN_SEGMENT_S:
        m = min(int(round(SU.BENCH_MIN_SEGMENT_S * sr)), worst.size)
        step = max(1, int(round(SU.BENCH_FALLBACK_STEP_S * sr)))
        offsets = np.arange(0, worst.size - m + 1, step)
        ok = np.array([bool(motor_on[i : i + m].all()) for i in offsets])
        pool = offsets[ok] if ok.any() else offsets
        key = (
            [worst[i : i + m].max() for i in pool]
            if ok.any()
            else [-level_db[i : i + m].mean() for i in pool]
        )
        a = int(pool[int(np.argmin(key))])
        b = a + m
    i0, i1 = int(round(edge + a)), int(round(edge + b))
    return dict(
        start_s=(edge + a) / sr,
        end_s=(edge + b) / sr,
        duration_s=(b - a) / sr,
        longest_inside_s=longest,
        level_db=float(level_db[a:b].mean()),
        level_deficit_db=float(level_db.max() - level_db[a:b].mean()),
        level_tol_db=float(level_tol_db),
        orders=[int(k) for k in orders],
        carrier_rev_s=[float(v) for v in carriers],
        margin_whole_recording_db=float(min(margins)),
        # the margin the rule SHOULD read: inside the selected window
        margin_in_window_db=float(SU.select_order(x[:, i0:i1], sr, float(survey_rev_s[0]))[1]),
    )


def cmd_verify_patch1(args: argparse.Namespace) -> None:
    """Run the patched rule on all 21 recordings and record what it selects."""
    rows = index_rows()
    out = []
    for name, row in rows.items():
        rec = C.load_recording(SU.DREGON_DATASET, row["recording_id"], None, SU.DREGON_RPS_KEY)
        point = SU.bench_manifest()[SU.dregon_bench_point_key(row["recording_id"])]
        res = patched_stationary_segment(
            rec.audio,
            rec.sr,
            [float(v) for v in point["speed_rev_s"]],
            level_tol_db=float(args.level_tol_db),
        )
        out.append(
            dict(
                name=name,
                old_segment=[round(v, 3) for v in row["segment"]],
                old_duration_s=round(float(row["duration_s"]), 3),
                old_stationary_pass=bool(row["stationary_pass"]),
                new_segment=[round(res["start_s"], 3), round(res["end_s"], 3)],
                new_duration_s=round(res["duration_s"], 3),
                new_longest_inside_s=round(res["longest_inside_s"], 3),
                new_level_deficit_db=round(res["level_deficit_db"], 2),
                margin_whole_recording_db=round(res["margin_whole_recording_db"], 2),
                margin_in_window_db=round(res["margin_in_window_db"], 2),
                orders=res["orders"],
                carrier_rev_s=[round(v, 4) for v in res["carrier_rev_s"]],
            )
        )
        print(f"  {name}: {out[-1]['old_segment']} -> {out[-1]['new_segment']}", flush=True)
        del rec
    d = np.array([r["new_level_deficit_db"] for r in out])
    mw = np.array([r["margin_in_window_db"] for r in out])
    li = np.array([r["new_longest_inside_s"] for r in out])
    write(
        OUT_DIR / "patch1_check.json",
        dict(
            schema=SCHEMA,
            what="supports.stationary_segment + the in-window level test of patch 1, on all 21 recordings",
            level_tol_db=float(args.level_tol_db),
            level_deficit_db=dict(min=round(float(d.min()), 2), max=round(float(d.max()), 2)),
            margin_in_window_db=dict(min=round(float(mw.min()), 2), max=round(float(mw.max()), 2)),
            n_margin_in_window_below_3db=int((mw < SU.BENCH_LINE_MARGIN_DB).sum()),
            n_longest_inside_below_min=int((li < SU.BENCH_MIN_SEGMENT_S).sum()),
            supports=out,
        ),
    )


# ── commands ────────────────────────────────────────────────────────────────


def cmd_census(_args: argparse.Namespace) -> None:
    """Every DREGON bench support: is its window on the motor or after it?"""
    rows = index_rows()
    out = []
    for name, row in rows.items():
        rec = C.load_recording(SU.DREGON_DATASET, row["recording_id"], None, SU.DREGON_RPS_KEY)
        sr = float(rec.sr)
        env = envelope_db(rec.audio, sr)
        dur = float(row["duration_s"])
        e0 = int(row["segment"][0] / ENVELOPE_S)
        e1 = max(e0 + 1, int(row["segment"][1] / ENVELOPE_S))
        win = float(env[e0:e1].mean())
        ls, ll = loudest_window(env, dur)
        s7, l7 = loudest_window(env, 7.0)
        out.append(
            dict(
                name=name,
                stationary_pass=bool(row["stationary_pass"]),
                duration_s=round(dur, 3),
                segment=[round(v, 3) for v in row["segment"]],
                recording_s=round(rec.audio.shape[-1] / sr, 2),
                window_mean_db=round(win, 2),
                loudest_same_length=dict(start_s=ls, mean_db=round(ll, 2)),
                loudest_7s=dict(start_s=s7, mean_db=round(l7, 2)),
                deficit_db=round(ll - win, 2),
                deficit_vs_7s_db=round(l7 - win, 2),
            )
        )
        del rec
    d = np.array([r["deficit_db"] for r in out])
    payload = dict(
        schema=SCHEMA,
        what="window level of every dregon-bench support vs the loudest window of its recording",
        envelope_step_s=ENVELOPE_S,
        n_supports=len(out),
        n_deficit_ge_6db=int((d >= 6).sum()),
        n_deficit_ge_10db=int((d >= 10).sum()),
        n_deficit_ge_20db=int((d >= 20).sum()),
        supports=out,
    )
    write(OUT_DIR / "census.json", payload)


def cmd_diag(args: argparse.Namespace) -> None:
    rows = index_rows()
    for name in args.support:
        t0 = time.time()
        support, prov = load_target(name, rows)
        base = name.removesuffix(CONTROL_SUFFIX)
        row, fit = rows[base], fit_json(base)
        batch = batch_for(support)
        sigma_rev = float(row["residual_std_hz"][0]) / max(1, int(row["orders"][0]))
        payload = dict(
            schema=SCHEMA,
            support=name,
            base_support=base,
            provenance=prov,
            fit_git=(fit or {}).get("git"),
            fit_selected_seed=((fit or {}).get("restarts") or {}).get("selected_seed"),
            geometry=dict(
                sr=int(support.sr),
                n_samples_periodogram=int(support.n_fft),
                n_samples_model_grid=int(bgrid(batch).n),
                bin_hz_data=float(support.freqs_hz[1] - support.freqs_hz[0]),
                bin_hz_model=float(bgrid(batch).diagnostics["bin_hz"]),
                bin_hz_relative_error=float(
                    bgrid(batch).diagnostics["bin_hz"] / (support.freqs_hz[1] - support.freqs_hz[0])
                    - 1.0
                ),
                duration_s=round(support.duration_s, 6),
                segment=[round(v, 6) for v in support.segment],
                k_max=int(batch.k_max),
                n_band_bins=int(np.asarray(batch.band).sum()),
                n_cells=batch.n_cells,
            ),
            stationarity=support.meta.get("stationarity"),
            scale=scale_block(
                support, row, fit if not name.endswith(CONTROL_SUFFIX) else None, control_dur_s=7.0
            ),
            registration=registration(
                support, row, fit if not name.endswith(CONTROL_SUFFIX) else None
            ),
            refine_score=refine_score_audit(
                support, row, fit if not name.endswith(CONTROL_SUFFIX) else None
            ),
            initialiser=initialiser_block(batch, sigma_rev),
            wall_s=round(time.time() - t0, 1),
        )
        write(OUT_DIR / f"{name}.json", payload)


def cmd_objective(args: argparse.Namespace) -> None:
    rows = index_rows()
    for name in args.support:
        t0 = time.time()
        support, prov = load_target(name, rows)
        base = name.removesuffix(CONTROL_SUFFIX)
        row = rows[base]
        fit = fit_json(base)
        batch = batch_for(support)
        sigma_rev = float(row["residual_std_hz"][0]) / max(1, int(row["orders"][0]))
        control = name.endswith(CONTROL_SUFFIX)
        if control or fit is None:
            # no fit exists for a control window: the reference parameter set
            # is the initialiser's own (floor + mic + dynamics centres), which
            # is what the fit would start from
            init = FT.initial_values(batch, mode="bench")
            par = MD.sample_params_from_values(batch, mode="bench", values=init)
            ref = "initial_values"
        else:
            par = MD.params_from_dict(fit["params"])
            ref = "committed fit params"
        fitted = objective_of(batch, par)
        floor_only = objective_of(batch, with_profile(par, -300.0))
        split = {}
        if not control and fit is not None:
            prof = np.asarray(fit["params"]["profile"]["profile_db"], dtype=np.float64)
            f_top = float(bgrid(batch).freqs_hz[np.asarray(batch.band)].max())
            k_in = int(math.floor(f_top / float(fit["params"]["carrier_rev_s"][0])))
            for tag, sl in (
                ("in_band_orders_only", slice(k_in, None)),
                ("out_of_band_orders_only", slice(0, k_in)),
            ):
                pp = prof.copy()
                pp[0, sl] = -300.0
                split[tag] = objective_of(
                    batch,
                    dataclasses.replace(par, profile_db=torch.as_tensor(pp, dtype=torch.float64)),
                )
            split["k_in_band"] = k_in
        carriers = {
            "index": float(row["carriers_rev_s"][0]),
            "survey": float(row["survey_rev_s"][0]),
        }
        if fit is not None and not control:
            carriers["fit_refined"] = float(fit["diagnostics"]["batch"]["carrier_init_rev_s"][0])
            carriers["fitted"] = float(fit["params"]["carrier_rev_s"][0])
        carriers["batch_refined"] = float(np.asarray(batch.carrier_init)[0])
        # the measured carrier: the median of k f over the orders that carry a line
        reg = registration(support, row, fit if not control else None)
        implied = reg["lines"]["index"]["implied_carrier_rev_s"]
        if implied:
            carriers["measured_from_lines"] = float(implied)
        oracle = {nm: oracle_comb(batch, par, f0, sigma_rev) for nm, f0 in carriers.items()}
        oracle_peak = oracle_comb(
            batch,
            par,
            carriers.get("measured_from_lines", carriers["index"]),
            sigma_rev,
            mode="peak",
        )
        degenerate = {}
        if not control and fit is not None:
            dyn = dict(
                sigma_nu=torch.as_tensor(2.0 * math.pi * float(row["residual_std_hz"][0])),
                lam=torch.as_tensor(0.5),
            )
            f0 = carriers.get("measured_from_lines", carriers["index"])
            degenerate = oracle_comb(batch, dataclasses.replace(par, **dyn), f0, sigma_rev)
            degenerate["dynamics"] = dict(
                sigma_nu=float(dyn["sigma_nu"]), lam=0.5, note="index residual std in rad/s"
            )
        scan_f = np.round(
            np.arange(carriers["index"] - 0.30, carriers["index"] + 0.30 + 1e-9, 0.02), 6
        )
        scan = [
            dict(
                carrier_rev_s=float(f0),
                nats_per_cell=oracle_comb(batch, par, float(f0), sigma_rev)["objective"][
                    "nats_per_cell"
                ],
            )
            for f0 in scan_f
        ]
        # (d) THE PIPELINE MAIN PROPOSES: no in-fit refinement at all. The batch
        # geometry comes from the index carrier (``refine_carrier=False``, so
        # ``k_max`` and the order groups are the index carrier's), the comb is
        # read off the data at that carrier, and everything else is as fitted.
        d_batch = MD.bench_batch(
            name=support.name + "__no_refine",
            power=np.asarray(support.power, dtype=np.float64),
            sr=int(support.sr),
            carrier_mean=np.asarray(support.carrier_rev_s, dtype=np.float64).mean(axis=1),
            k_cap=130,
            refine_carrier=False,
        )
        d_arm = oracle_comb(d_batch, par, carriers["index"], sigma_rev)
        d_arm["batch_k_max"] = int(d_batch.k_max)
        d_arm["batch_carrier_init_rev_s"] = float(np.asarray(d_batch.carrier_init)[0])
        # the index carrier can reach one order further than the fitted profile
        # is wide (k_max 117 against 116 on Motor1_70), so the two reference
        # evaluations that reuse the FITTED profile stay at its width
        d_k = min(int(d_batch.k_max), int(np.asarray(par.profile_db).shape[1]))
        d_arm["reference_k_max"] = d_k
        d_groups = SP.order_groups(
            d_k,
            sigma_nu=math.exp(MD.PRIORS.log_sigma_nu[0]),
            lam=math.exp(MD.PRIORS.log_lam[0]),
            sr=bgrid(d_batch).sr,
            n=bgrid(d_batch).n,
        )
        d_arm["floor_only"] = objective_of(
            d_batch, with_profile(par, -300.0), k_max=d_k, groups=d_groups
        )
        d_arm["as_fitted_on_this_batch"] = objective_of(d_batch, par, k_max=d_k, groups=d_groups)
        payload = dict(
            schema=SCHEMA,
            support=name,
            base_support=base,
            provenance=prov,
            reference_params=ref,
            a_fitted=fitted,
            b_floor_only=floor_only,
            c_oracle=oracle,
            c_oracle_peak_matched=oracle_peak,
            c_oracle_degenerate_dynamics=degenerate,
            d_index_no_refine=d_arm,
            comb_block_split=split,
            oracle_carrier_scan=scan,
            # a control window has no fit of its own; quoting the base
            # support's objective beside a control's would invite a comparison
            # across two different windows, which the log M term forbids
            fit_json_whittle_nats=(
                fit["objective"]["whittle_nats"] if (fit and not control) else None
            ),
            fit_git=None if control else (fit or {}).get("git"),
            fit_selected_seed=(
                None if control else ((fit or {}).get("restarts") or {}).get("selected_seed")
            ),
            wall_s=round(time.time() - t0, 1),
        )
        write(OUT_DIR / f"objective_{name}.json", payload)


def cmd_figures(_args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = index_rows()
    name = "bench_dregon_Motor1_70"
    sup, _prov = load_target(name, rows)
    on, _prov_on = load_target(name + CONTROL_SUFFIX, rows)
    row, fit = rows[name], fit_json(name)
    assert fit is not None, f"{name} has no committed fit JSON"
    par = MD.params_from_dict(fit["params"])
    batch = batch_for(sup)
    sigma_rev = float(row["residual_std_hz"][0]) / int(row["orders"][0])
    diag = json.loads((OUT_DIR / f"{name}.json").read_text())
    obj = json.loads((OUT_DIR / f"objective_{name}.json").read_text())
    f_index = float(row["carriers_rev_s"][0])
    f_fit = float(fit["params"]["carrier_rev_s"][0])
    f_ref = float(fit["diagnostics"]["batch"]["carrier_init_rev_s"][0])
    f_meas = obj["c_oracle"].get("measured_from_lines", {}).get("carrier_rev_s", f_index)

    pm = mic_mean(sup.power)
    df = float(sup.freqs_hz[1] - sup.freqs_hz[0])
    with torch.no_grad():
        m_fit = (
            SP.bench_model(bgrid(batch), par, k_max=batch.k_max, groups=batch.bench_order_groups)
            .mean(dim=(0, 1))
            .numpy()
        )
    orc = oracle_comb(batch, par, f_meas, sigma_rev)
    par_o = dataclasses.replace(
        par,
        carrier_rev_s=torch.as_tensor([f_meas], dtype=torch.float64),
        profile_db=torch.as_tensor(np.asarray(orc["profile_db"])[None, :], dtype=torch.float64),
    )
    groups = SP.order_groups(
        orc["k_max"],
        sigma_nu=math.exp(MD.PRIORS.log_sigma_nu[0]),
        lam=math.exp(MD.PRIORS.log_lam[0]),
        sr=bgrid(batch).sr,
        n=bgrid(batch).n,
    )
    with torch.no_grad():
        m_orc = (
            SP.bench_model(bgrid(batch), par_o, k_max=orc["k_max"], groups=groups)
            .mean(dim=(0, 1))
            .numpy()
        )

    # 1. data vs model around the first orders, committed support
    ks = (1, 2, 4, 10, 20, 40)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, k in zip(axes.ravel(), ks):
        c = k * f_index
        span = max(3.0, 6.0 * k * sigma_rev, min(40.0, abs(k * (f_fit - f_index)) + 1.0))
        half = int(math.ceil(span / df))
        j = int(round(c / df))
        sl = slice(max(0, j - half), min(pm.size, j + half + 1))
        f = sup.freqs_hz[sl]
        ax.plot(f, 10 * np.log10(pm[sl]), color="0.3", lw=0.8, label="data (mic mean)")
        ax.plot(f, 10 * np.log10(m_fit[sl]), color="tab:red", lw=1.2, label="fitted model")
        ax.plot(f, 10 * np.log10(m_orc[sl]), color="tab:blue", lw=1.2, label="oracle comb")
        for fc, col, lab in ((f_index, "tab:green", "index"), (f_fit, "tab:red", "fitted")):
            if f[0] <= k * fc <= f[-1]:
                ax.axvline(
                    k * fc, color=col, ls=":", lw=1.0, label=f"k x {lab}" if k == 1 else None
                )
        ax.set_xlim(float(f[0]), float(f[-1]))
        ax.set_title(
            f"k = {k}  ({c:.1f} Hz)"
            + ("" if f[0] <= k * f_fit <= f[-1] else "  [k x fitted off-panel]")
        )
        ax.set_xlabel("Hz")
        ax.set_ylabel("dB")
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle(
        f"{name}: committed support (post-spin-down window) — data vs fitted vs oracle comb"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_overlay_Motor1_70.png", dpi=110)
    plt.close(fig)

    # 2. same orders on the motor-on control window
    bon = batch_for(on)
    init = FT.initial_values(bon, mode="bench")
    par_on = MD.sample_params_from_values(bon, mode="bench", values=init)
    pon = mic_mean(on.power)
    f_on = float(np.asarray(bon.carrier_init)[0])
    orc_on = oracle_comb(bon, par_on, f_on, sigma_rev)
    par_on_o = dataclasses.replace(
        par_on,
        carrier_rev_s=torch.as_tensor([f_on], dtype=torch.float64),
        profile_db=torch.as_tensor(np.asarray(orc_on["profile_db"])[None, :], dtype=torch.float64),
    )
    g_on = SP.order_groups(
        orc_on["k_max"],
        sigma_nu=math.exp(MD.PRIORS.log_sigma_nu[0]),
        lam=math.exp(MD.PRIORS.log_lam[0]),
        sr=bgrid(bon).sr,
        n=bgrid(bon).n,
    )
    with torch.no_grad():
        m_on = (
            SP.bench_model(bgrid(bon), par_on_o, k_max=orc_on["k_max"], groups=g_on)
            .mean(dim=(0, 1))
            .numpy()
        )
        m_on_fl = (
            SP.bench_model(
                bgrid(bon), with_profile(par_on_o, -300.0), k_max=orc_on["k_max"], groups=g_on
            )
            .mean(dim=(0, 1))
            .numpy()
        )
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    d_on = float(on.freqs_hz[1] - on.freqs_hz[0])
    for ax, k in zip(axes.ravel(), ks):
        half = int(math.ceil(max(3.0, 6.0 * k * sigma_rev) / d_on))
        j = int(round(k * f_on / d_on))
        sl = slice(max(0, j - half), min(pon.size, j + half + 1))
        ax.plot(on.freqs_hz[sl], 10 * np.log10(pon[sl]), color="0.3", lw=0.8, label="data")
        ax.plot(
            on.freqs_hz[sl], 10 * np.log10(m_on[sl]), color="tab:blue", lw=1.2, label="oracle comb"
        )
        ax.plot(
            on.freqs_hz[sl],
            10 * np.log10(m_on_fl[sl]),
            color="tab:orange",
            lw=1.0,
            label="floor only",
        )
        ax.axvline(k * f_on, color="tab:blue", ls=":", lw=1.0)
        ax.set_title(f"k = {k}  ({k * f_on:.1f} Hz)")
        ax.set_xlabel("Hz")
        ax.set_ylabel("dB")
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle(
        f"{name}{CONTROL_SUFFIX}: same recording, loudest {on.duration_s:.2f} s window "
        f"[{on.segment[0]:.1f}, {on.segment[1]:.1f}] s — the comb the fit never saw"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_overlay_Motor1_70_motor_on.png", dpi=110)
    plt.close(fig)

    # 3. registration score, per-order line SNR and the objective bars
    diag_on = json.loads((OUT_DIR / f"{name}{CONTROL_SUFFIX}.json").read_text())
    obj_on = json.loads((OUT_DIR / f"objective_{name}{CONTROL_SUFFIX}.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    tag = "jitter__capped_snr"
    for d_, lab, col in (
        (diag, "committed window", "tab:red"),
        (diag_on, "motor-on control", "tab:blue"),
    ):
        v = d_["registration"]["scores"][tag]
        f = v["curve_f0"] + np.arange(len(v["curve"])) * v["curve_step_rev_s"]
        y = np.asarray(v["curve"], dtype=np.float64)
        axes[0].plot(
            f, y - np.median(y), color=col, lw=1.0, label=f"{lab} (argmax {v['argmax_rev_s']:.3f})"
        )
    for fc, col, lab in (
        (f_index, "tab:green", f"index {f_index:.3f}"),
        (f_ref, "tab:orange", f"fit refined {f_ref:.3f}"),
        (f_fit, "0.2", f"fitted {f_fit:.3f}"),
    ):
        axes[0].axvline(fc, color=col, ls="--", lw=1.2, label=lab)
    axes[0].set_xlabel("carrier [rev/s]")
    axes[0].set_ylabel("S(f) - median S  [dB, capped per order]")
    axes[0].set_title(f"{name}: comb registration score")
    axes[0].legend(fontsize=7, loc="upper left")
    for d_, key, lab, col in (
        (diag, "index", "committed @ index", "tab:green"),
        (diag, "fitted", "committed @ fitted", "0.3"),
        (diag_on, "index", "motor-on @ index", "tab:blue"),
    ):
        t = d_["registration"]["lines"][key]
        axes[1].plot(
            t["k"],
            t["snr_db"],
            ".-",
            ms=3,
            lw=0.7,
            color=col,
            label=f"{lab}: {t['n_above_6db']}/{t['n_orders']} orders > 6 dB",
        )
    axes[1].axhline(LINE_SNR_THRESHOLD_DB, color="0.5", ls="--", lw=1.0)
    axes[1].set_xlabel("order k")
    axes[1].set_ylabel("line SNR [dB over local median floor]")
    axes[1].set_title("per-order line SNR")
    axes[1].legend(fontsize=7)
    bars: list[tuple[str, float, str]] = [
        ("(a) fitted", obj["a_fitted"]["nats_per_cell"], "tab:red"),
        ("(b) floor only", obj["b_floor_only"]["nats_per_cell"], "tab:orange"),
        (
            "(c) oracle@index",
            obj["c_oracle"]["index"]["objective_at_best_offset"]["nats_per_cell"],
            "tab:green",
        ),
        (
            "(c) oracle@fitted",
            obj["c_oracle"]["fitted"]["objective_at_best_offset"]["nats_per_cell"],
            "0.3",
        ),
        ("CTRL floor only", obj_on["b_floor_only"]["nats_per_cell"], "tab:orange"),
        (
            "CTRL oracle",
            obj_on["c_oracle"]["batch_refined"]["objective_at_best_offset"]["nats_per_cell"],
            "tab:blue",
        ),
    ]
    axes[2].bar(range(len(bars)), [v for _, v, _ in bars], color=[c for _, _, c in bars])
    axes[2].set_xticks(range(len(bars)))
    axes[2].set_xticklabels([n for n, _, _ in bars], rotation=30, ha="right", fontsize=7)
    for i, (_, v, _) in enumerate(bars):
        axes[2].annotate(
            f"{v:.2f}", (i, v), ha="center", va="bottom" if v < 0 else "top", fontsize=7
        )
    axes[2].set_ylabel("Whittle nats / cell (lower is better)")
    axes[2].set_title(
        "objective: committed support vs motor-on control\n(levels differ between windows; compare within a window)"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_registration_Motor1_70.png", dpi=110)
    plt.close(fig)

    # 4. the census: which windows sit on the motor
    cen = json.loads((OUT_DIR / "census.json").read_text())
    fig, ax = plt.subplots(figsize=(11, 5))
    names = [r["name"].replace("bench_dregon_", "") for r in cen["supports"]]
    d = [r["deficit_db"] for r in cen["supports"]]
    colours = ["tab:red" if v >= 10 else ("tab:orange" if v >= 6 else "tab:green") for v in d]
    ax.bar(names, d, color=colours)
    ax.axhline(6, color="0.4", ls="--", lw=1.0)
    ax.axhline(10, color="0.2", ls="--", lw=1.0)
    ax.set_ylabel("loudest same-length window - support window [dB]")
    ax.set_title(
        "dregon-bench supports: how far below the motor-on level the frozen window sits "
        f"({cen['n_deficit_ge_10db']} of {cen['n_supports']} are >= 10 dB down)"
    )
    ax.tick_params(axis="x", rotation=75, labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_census.png", dpi=110)
    plt.close(fig)
    print(f"wrote 4 figures to {OUT_DIR}")


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))
    print(f"wrote {path} ({path.stat().st_size} bytes)")


DIAG_DEFAULT = list(TARGETS) + [t + CONTROL_SUFFIX for t in TARGETS]
OBJ_DEFAULT = [TARGETS[0], TARGETS[2], TARGETS[0] + CONTROL_SUFFIX]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("census").set_defaults(fn=cmd_census)
    for cmd, fn, default in (
        ("diag", cmd_diag, DIAG_DEFAULT),
        ("objective", cmd_objective, OBJ_DEFAULT),
    ):
        p = sub.add_parser(cmd)
        p.add_argument("--support", action="append", default=None)
        p.set_defaults(fn=fn, _default_supports=default)
    p = sub.add_parser("verify-patch1")
    p.add_argument("--level-tol-db", type=float, default=6.0)
    p.set_defaults(fn=cmd_verify_patch1)
    sub.add_parser("figures").set_defaults(fn=cmd_figures)
    sub.add_parser("all").set_defaults(fn=None, level_tol_db=6.0)
    args = ap.parse_args(argv)
    torch.set_num_threads(4)
    if args.cmd == "all":
        cmd_census(args)
        cmd_diag(argparse.Namespace(support=DIAG_DEFAULT))
        cmd_objective(argparse.Namespace(support=OBJ_DEFAULT))
        cmd_verify_patch1(args)
        cmd_figures(args)
        return 0
    if getattr(args, "support", None) is None and hasattr(args, "_default_supports"):
        args.support = args._default_supports
    args.fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
