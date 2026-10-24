"""Noise model v4, the DREGON WIND STAGE (procedure C-wind): the flight wind
as the SC gust generator per capsule, its amplitude by likelihood and its gust
law by simulated moments, on a frozen flight-stage payload.

WHY. The flight stage (:mod:`.dregon_flight_v4`) fits a STATIC per-mic wind:
stationary Gaussian noise of v3's ``wind_shape``. Real DREGON wind is GUSTY and
independent per capsule: its between-line 20-400 Hz level moves with sd
2.9-3.4 dB over 128 ms frames at a cross-mic correlation of 0.06-0.2, where
the static wind reads 1.3-1.5 dB (pure chi-square scatter). A static wind at
the fitted mean power masks the low rotor orders in EVERY frame; a probe that
replaced it by the SC gust generator at the same mean band power cut the
HPPNet tracker error from 22.3 to 2.1 rev/s (hovering window). So the wind
amplitude is fitted separately with the wind noise model, and rendered with it
(``params.wind.kind = "sc"``, :class:`data_processing.noise_model.v4.Wind`).

THE WIND MODEL. Per capsule an INDEPENDENT draw of the SC generator
(:func:`data_processing.noise_model.wind_sc.wind_noise`, Mirabilii et al.,
IWAENC 2022): wind-speed knots ``speed_offset_ms + speed_scale_ms *
Weibull(2)``, ``max(1, round(gust_rate_hz * duration))`` per clip, resampled
over the clip as SC does, plus SC's 100 ms fluctuations; the long-term gain
(8 dB per m/s), the GARCH short-term envelope and the speed-dependent LPC;
unit RMS. ``shape.psd`` is the generator's long-run mean PSD at the law
(:func:`sc_shape`: ``n_mc`` clips of ``clip_s`` from a fixed seed, the mean
periodic-Hann periodogram), divided by its 20-100 Hz mean ``unit_psd_ref``;
``level_db[m]`` is 10 log10 of capsule ``m``'s expected wind PSD averaged over
20-100 Hz, in the model's periodogram units (the static ``wind_db``'s on its
flat part). The renderer multiplies a unit-RMS draw by ``sqrt(10^{L/10} /
unit_psd_ref)``, so its expected PSD is ``10^{L/10} shape.psd``; the array
response multiplies it as it does the static wind.

THE PROCEDURE. Lines, floor, per-rotor/per-mic gains and coherence are the
source payload's, frozen; only the wind changes.

1. REAL TARGETS (:func:`wind_statistics`), per pool segment
   (:func:`.dregon_flight_v4.pool_specs`: the five 8 s ``dregon-floor``
   segments, never the scoring windows), from the raw audio and the
   ``motors_command`` label (:func:`.supports.flight_clip`, the loader of the
   flight stage's supports): the 8192 / 1024 periodic-Hann periodogram; the
   floor envelope (:func:`experiments.stochastic_fit.diagnostics.floor_envelope`,
   floor cells of 50-500 Hz), regressed per mic on the SAME mic's 1-4 kHz band
   level (dB, same frames) and kept as the residual re its median (the common
   thrust modulation is not wind; the simulations get the same); its
   per-capsule std, p5, p95, frac_burst, run_median, lag1
   (:func:`~experiments.stochastic_fit.diagnostics.floor_envelope_capsule_stats`)
   and its cross-mic correlation (``floor_envelope_stats``' ``xmic_corr``); and
   the mean between-line spectrum (the floor cells' mean power, dB) in the
   twelve base-2 third-octave bands 50-630 Hz (:data:`SPEC_CENTRES_HZ`). The
   SCALES (:func:`stat_scales`) are the across-segment sd of each statistic,
   pooled over the mics (per band for the spectrum), floored at
   :data:`STAT_FLOORS` (with one segment, the floors alone).
2. AMPLITUDE (:func:`fit_levels`) at a law ``(offset, scale, rate)``: its shape
   (:func:`sc_shape`), then per mic ONE level by the Whittle likelihood on the
   flight stage's own pool and front end (:func:`.dregon_flight_v4.load_pool`:
   2048 / 512, frames thinned by 4, composite exposure weights, the payload's
   ``band_hz``): ``E = E0 + 10^{L_m/10} T_m``, ``E0`` the source payload's
   expected periodogram with its wind removed (lines + floor + static share;
   :func:`frozen_expectation`, the stage model's ``E`` at its fitted values, which
   the stage's renderer check holds to 1e-6 of ``expected_periodogram``) and
   ``T_m`` the window transform of ``shape.psd`` times the array response at
   0 dB (:func:`wind_transform`, ``expected_periodogram`` of the wind alone).
   Prior ``L_m ~ N(source wind's 20-100 Hz mean level, 6 dB)``. The 1-D search
   is a 1 dB scan over +-30 dB around the centre, then bounded Brent on the
   best bracket (:data:`LEVEL_SPAN_DB`).
3. GUST LAW (:func:`grid_point`) by simulated moments over the grid
   :data:`OFFSETS_MS` x :data:`SCALES_MS` x :data:`RATES_HZ` (CLI-configurable):
   at each point step 2, then per segment and seed the source payload WITHOUT
   its wind rendered ONCE (:func:`base_unit`, cached and reused across the grid:
   the wind is additive and independent) plus the SC wind at the fitted levels
   (:func:`data_processing.noise_model.render.render_v4_wind`, the renderer's
   own wind path), the step-1 statistics of the sum, averaged over the seeds
   (common random numbers: the same render and wind seeds at every point).
   DISTANCE (:func:`distance`): the sum over segments of squared standardised
   differences ``((sim - real) / scale)^2`` where each statistic carries one
   unit per (segment, mic): the six per-capsule envelope statistics one each,
   ``xmic_corr`` (one per segment) weighed by ``M``, and the spectrum as the
   mean over its bands per mic. The minimum is chosen; the whole grid is kept.
4. EXPORT (:func:`run_wind_stage`): the source payload with ``params.wind`` the
   ``sc`` block at the chosen point, a ``wind_stage`` provenance block and its
   :func:`wind_renderer_check` (the rendered wind's mean periodogram over many
   seeds against ``10^{L/10} shape.psd`` times the array response, third
   octaves 20-500 Hz, within :data:`CHECK_TOL_DB`).

Restartable and parallel: ``<out>/<stem>__scwind/`` holds ``config.json``
(refuses a different configuration), ``context.npz``/``context.json`` (targets,
scales, the pool and ``E0``), ``base/`` (the wind-free renders, one
:mod:`utils.gridrun` unit per segment and seed) and ``grid/raw/`` (one unit per
grid point).

BATTERY MODE (:func:`run_gust_stage`, CLI ``wind --targets battery``; one
step of the alternating DREGON fit). The model and the renderer are unchanged;
the source's lines (profiles, pedestal, shaft, gains, coherence) and the
floor's shape, exponent and static share stay frozen.

1. SOURCE: a DREGON v4 flight payload. One whose wind is already ``sc`` (a
   line-refit payload) centres the level priors on its ``level_db`` and adds
   its own gust law to the grid as one point, with its own ``shape``, so that
   point reproduces the source when the inner fit returns the source's values.
2. INNER (:func:`fit_levels_floor`) at a law: the per-mic levels ``L_m`` and
   the floor's ``mean_db`` ``mu`` (one scalar) by the composite Whittle risk on
   the stage pool (``pool="floor"``: the tuning segments' frames, as step 2;
   ``"wide"``: ``dregon_flight_v4.wide_pool_specs()``, every frame) for ``E =
   E_lines + 10^{(mu - mu_src)/10} E_floor + 10^{L_m/10} T_m``. The floor and
   its static share scale exactly with ``10^{mean_db/10}``. Priors ``L_m ~
   N(centre_m, 6 dB)`` and ``mu ~ N(mu_src, 6 dB)``. Coordinate descent: the
   levels at ``mu``, then ``mu`` at the levels, each a 1 dB scan over +-30 dB
   around its prior centre (later sweeps: +-2 dB around the current value
   first) and bounded Brent, until nothing moves by more than
   :data:`INNER_TOL_DB`.
3. OUTER (:func:`battery_point`): per tuning segment ``i`` (the five
   ``pool_specs()`` 8 s segments) and seed ``k``, the sim ``y_ik = lines_ik +
   10^{(mu - mu_src)/20} floor_ik + wind_ik``. The source without its wind is
   rendered ONCE per ``(i, k)`` and split into its lines (floor silenced, same
   seed) and its floor (the rest: the rendered floor is linear in
   ``10^{mean_db/20}``). ``wind_ik`` is the sc wind at the law and the fitted
   levels (:func:`render_v4_wind`, wind seed ``(i, k)``, common random numbers).
   Each distance of :data:`BATTERY_TERMS` (:func:`battery_terms`:
   ``similarity.spectro_temporal`` and ``similarity.level_distribution``, FULL
   band, real as reference, NO thrust regression) is divided by its SEED FLOOR
   (:func:`seed_floors`: the SOURCE payload as it is, rendered at two seeds on
   the segment's label, :data:`N_FLOOR_PAIRS` pairs per segment, once per
   run), then averaged over segments and seeds and summed with
   :data:`BATTERY_WEIGHTS` (:data:`BATTERY_FORMULA`, :func:`battery_objective`).
   The grid is :data:`BATTERY_OFFSETS_MS` x :data:`BATTERY_SCALES_MS` x
   :data:`BATTERY_RATES_HZ` plus the source's law; the minimum is chosen.
4. EXPORT: ``<out>/<stem>__gust.json`` (the source with the chosen ``sc``
   block and floor ``mean_db``, a ``wind_stage`` block, the wind renderer
   check) and ``<stem>__gust.grid.json`` (every point and term). The work dir
   ``<out>/<stem>__gust/`` holds ``base/`` (lines and floor renders), ``floor/``
   (the seed-floor pairs) and ``grid/`` (one unit per law). ``deadline_min``
   stops launching grid points; a rerun resumes from the unit files.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from data_processing.noise_model import FIT_SCHEMA_V4
from data_processing.noise_model import v4 as V4
from data_processing.noise_model.render import render_noise, render_v4_wind
from data_processing.noise_model.wind_sc import wind_noise
from experiments.stochastic_fit.data import Clip, periodogram
from experiments.stochastic_fit.diagnostics import (
    floor_cells,
    floor_envelope,
    floor_envelope_capsule_stats,
    floor_envelope_stats,
)
from experiments.stochastic_fit.revised_phase import composite_risk
from utils.gridrun import Unit, run_grid, unit_path

from . import dregon_flight_v4 as DF
from . import flight_v4 as FV
from . import similarity as SIM
from . import supports as SU

__all__ = [
    "BATTERY_TERMS",
    "BatteryContext",
    "LevelData",
    "Segment",
    "base_unit",
    "battery_objective",
    "battery_point",
    "battery_terms",
    "between_line_spectrum",
    "distance",
    "fit_levels",
    "fit_levels_floor",
    "frozen_expectation",
    "grid_laws",
    "grid_point",
    "grid_unit",
    "gust_payload",
    "load_segments",
    "mean_periodogram",
    "run_gust_stage",
    "run_wind_stage",
    "sc_block",
    "sc_shape",
    "seed_floors",
    "source_law",
    "stat_scales",
    "stats_table",
    "thrust_residual",
    "wind_renderer_check",
    "wind_statistics",
    "wind_transform",
    "with_wind",
]

STAGE = "dregon_wind_v4"
#: The gust instrument's front end (docs/experiments/stochastic-fit.md, "DREGON's
#: low-band floor"): 1.95 Hz bins, 64 ms hop, floor cells of 50-500 Hz.
ENV_N_FFT = 8192
ENV_HOP = 1024
ENV_BAND_HZ = (50.0, 500.0)
#: The band whose level carries the common thrust modulation the envelope is
#: regressed on (the lines and floor of every rotor, little wind).
THRUST_BAND_HZ = (1000.0, 4000.0)
#: Base-2 third-octave centres 49.6 ... 630 Hz (edges ``c 2^{+-1/6}``, contiguous).
SPEC_CENTRES_HZ = tuple(float(1000.0 * 2.0 ** (k / 3.0)) for k in range(-13, -1))
#: The per-capsule envelope statistics of the distance.
MIC_STATS = ("std", "p5", "p95", "frac_burst", "run_median", "lag1")
#: The scale floors: dB statistics 0.3 dB, fractions and correlations 0.05, the
#: run median one 64 ms frame, the spectrum 0.3 dB.
STAT_FLOORS = dict(
    std=0.3,
    p5=0.3,
    p95=0.3,
    frac_burst=0.05,
    run_median=1.0,
    lag1=0.05,
    xmic_corr=0.05,
    spectrum_db=0.3,
)
#: The level prior's sd (dB), centred on the source wind's 20-100 Hz level.
LEVEL_PRIOR_SD_DB = 6.0
#: The level search: a 1 dB scan over +-LEVEL_SPAN_DB around the prior centre.
LEVEL_SPAN_DB = 30.0
#: The default gust-law grid (CLI ``--offsets``/``--scales``/``--rates``).
OFFSETS_MS = (0.0, 1.0, 2.0, 4.0, 6.0)
SCALES_MS = (0.25, 0.5, 1.0, 2.0, 3.0)
RATES_HZ = (0.25, 0.5, 1.0, 2.0, 4.0)
#: Simulation seeds per segment and grid point.
N_SEEDS = 4
#: The shape Monte-Carlo: clips, seed, frame and clip length (the pool segments').
SHAPE_N_MC = 64
SHAPE_SEED = 20261023
SHAPE_N_FFT = 4096
SHAPE_CLIP_S = float(SU.DREGON_FLOOR_DUR_S)
#: Render seeds of the wind-free base renders and of the sim winds:
#: ``seed + 100 * segment + k``.
BASE_SEED = 9100
WIND_SEED = 9500
#: The wind renderer check: seeds, seed offset, band and tolerance (max |error|
#: of the third-octave band power, dB, per mic and band).
CHECK_SEEDS = 64
CHECK_SEED = 9900
CHECK_BAND_HZ = (20.0, 500.0)
CHECK_TOL_DB = 1.0
#: Battery mode (:func:`run_gust_stage`): the target modes and inner pools.
TARGETS = ("envelope", "battery")
POOLS = ("floor", "wide")
#: The battery mode's default gust-law grid (175 points).
BATTERY_OFFSETS_MS = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 14.0)
BATTERY_SCALES_MS = (0.25, 0.5, 1.0, 2.0, 4.0)
BATTERY_RATES_HZ = (0.5, 1.0, 2.0, 4.0, 8.0)
#: The floor mean's prior sd (dB), centred on the source's ``floor.mean_db``.
FLOOR_MEAN_PRIOR_SD_DB = 6.0
#: The inner coordinate descent: stop when no value moves by more than this (dB).
INNER_TOL_DB = 0.01
INNER_MAX_SWEEPS = 20
#: Later sweeps scan +- this around the current value first (the full scan
#: around the prior centre only when its best point is an edge).
INNER_LOCAL_SPAN_DB = 2.0
#: Seed-floor pairs per segment; pair p of segment i renders the source at
#: ``FLOOR_SEED + 100 i + 2 p`` and ``... + 1`` (no overlap with the grid's seeds).
N_FLOOR_PAIRS = 4
FLOOR_SEED = 30100
#: The outer objective's distances: ``<similarity function>.<summary key>``, as
#: ``scripts/noise_v4_similarity_report.py`` names them; ``offdiag_corr_abs_diff``
#: is ``|arm_offdiag_corr - real_offdiag_corr|`` of the mic means.
BATTERY_TERMS = (
    "spectro_temporal.frechet_db2",
    "spectro_temporal.frechet_mean_term_db2",
    "spectro_temporal.frechet_cov_term_db2",
    "spectro_temporal.comod_frobenius",
    "spectro_temporal.offdiag_corr_abs_diff",
    "spectro_temporal.sd_abs_diff_db",
    "level_distribution.w1_centered_db",
    "level_distribution.sd_abs_diff_db",
    "level_distribution.spread_abs_diff_db",
    "level_distribution.mod_norm_abs_db",
)
BATTERY_WEIGHTS = dict.fromkeys(BATTERY_TERMS, 1.0)
BATTERY_FORMULA = (
    "D(theta) = sum_j w_j dbar_j(theta) / f_j over the terms j; dbar_j(theta) = (1 / (S K)) "
    "sum_i sum_k d_j(x_i, y_ik(theta)), x_i the real tuning segment i (8 mics, motors_command "
    "label), y_ik(theta) = lines_ik + 10^{(mu - mu_src)/20} floor_ik + wind_ik(theta, L): the "
    "source without its wind rendered at base seed (i, k) and split into lines and floor, plus "
    "the sc wind at the law theta and the inner-fitted levels L at wind seed (i, k) (common "
    "random numbers), mu the inner-fitted floor mean_db; f_j = (1 / (S P)) sum_i sum_p "
    "d_j(z_ipA, z_ipB), the source payload rendered at two seeds on segment i's label (the "
    "seed floor, once per run); d_j(real, arm) the named similarity.py summary (means over "
    "mics; real = the reference); w_j = BATTERY_WEIGHTS[j]; a perfect generator scores about "
    "sum_j w_j"
)


def _plain(x: Any) -> Any:
    """JSON-able copy: arrays to lists, numpy scalars to Python."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(_plain(obj), indent=1))
    os.replace(tmp, path)


def _third_octave_edges(centres: Any) -> np.ndarray:
    c = np.asarray(centres, dtype=np.float64)
    return np.stack([c * 2.0 ** (-1.0 / 6.0), c * 2.0 ** (1.0 / 6.0)], axis=1)


def mean_periodogram(x: Any, n_fft: int, hop: int | None = None) -> np.ndarray:
    """``(M, n_fft // 2 + 1)`` mean periodic-Hann periodogram ``|rfft(w x)|^2 /
    sum w^2`` (the model's units) of ``(M, T)`` over frames fully inside,
    ``hop`` ``n_fft // 2`` by default."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    n, hop = int(n_fft), int(hop or n_fft // 2)
    w = V4.hann_window(n)
    frames = np.lib.stride_tricks.sliding_window_view(x, n, axis=-1)[:, ::hop]
    spec = np.fft.rfft(frames * w, axis=-1)
    return (spec.real**2 + spec.imag**2).mean(axis=1) / float(np.sum(w**2))


# ── the segments and the statistics ─────────────────────────────────────────


@dataclass
class Segment:
    """One pool segment's raw window."""

    name: str
    audio: np.ndarray  #: (M, T) at ``sr``
    rps: np.ndarray  #: (R, T) the ``motors_command`` label on the audio grid
    sr: int


def load_segments(max_segments: int | None = None) -> list[Segment]:
    """The flight stage's pool segments (:func:`.dregon_flight_v4.pool_specs`),
    in pool order, read as its supports are (:func:`.supports.flight_clip`)."""
    specs = DF.pool_specs()
    if max_segments is not None:
        specs = specs[: int(max_segments)]
    out = []
    for spec in specs:
        clip = SU.flight_clip(spec)
        out.append(
            Segment(
                spec.name,
                np.asarray(clip.audio, dtype=np.float32),
                np.asarray(clip.rps, dtype=np.float64),
                int(clip.sr),
            )
        )
    return out


def thrust_residual(env_db: np.ndarray, thrust_db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(M, N)`` each mic's envelope minus its least-squares fit on ``[1,
    thrust_db[m]]`` (the same mic's thrust-band level, same frames), re its
    median; and ``(M,)`` the slopes (dB per dB)."""
    env = np.asarray(env_db, dtype=np.float64)
    out = np.empty_like(env)
    slope = np.empty(env.shape[0])
    for m in range(env.shape[0]):
        x = np.stack([np.ones(env.shape[1]), np.asarray(thrust_db[m], dtype=np.float64)], axis=1)
        coef = np.linalg.lstsq(x, env[m], rcond=None)[0]
        r = env[m] - x @ coef
        out[m] = r - np.median(r)
        slope[m] = coef[1]
    return out, slope


def between_line_spectrum(
    power: np.ndarray,
    freqs: np.ndarray,
    rps: np.ndarray,
    centres: Any = SPEC_CENTRES_HZ,
) -> np.ndarray:
    """``(M, B)`` the mean power (dB) of the floor cells (:func:`floor_cells`,
    per frame at its rotor speeds) in each third-octave band, over every frame
    and cell; ``nan`` for a band without a floor cell in any frame."""
    m_, n_, _ = power.shape
    out = np.full((m_, len(centres)), np.nan)
    for b, (lo, hi) in enumerate(_third_octave_edges(centres)):
        total = np.zeros(m_)
        count = 0
        for n in range(n_):
            mask = floor_cells(freqs, rps[:, n], float(lo), float(hi))
            if mask.any():
                total += power[:, n, mask].sum(axis=-1)
                count += int(mask.sum())
        if count:
            out[:, b] = 10.0 * np.log10(np.maximum(total / count, 1e-300))
    return out


def wind_statistics(audio: np.ndarray, rps: np.ndarray, sr: int = SU.SR) -> dict[str, Any]:
    """The wind stage's statistics of one ``(M, T)`` window on its label ``rps
    (R, T)`` (module docstring, step 1): ``mic`` (each of :data:`MIC_STATS`,
    ``(M,)``), ``xmic_corr``, ``spectrum_db`` ``(M, B)``, and for the report
    ``pooled`` (``floor_envelope_stats`` of the residual), ``raw_pooled`` (of
    the envelope before the regression), the floor-cell ``cover`` and the
    regression ``thrust_slope``."""
    pg = periodogram(Clip("wind_stats", "dregon_room2", audio, rps, int(sr)), ENV_N_FFT, ENV_HOP)
    power = np.asarray(pg.power, dtype=np.float64)
    env, cover = floor_envelope(power, pg.freqs, pg.rps, *ENV_BAND_HZ)
    hi = (pg.freqs >= THRUST_BAND_HZ[0]) & (pg.freqs < THRUST_BAND_HZ[1])
    thrust = 10.0 * np.log10(np.maximum(power[:, :, hi].mean(axis=-1), 1e-300))
    resid, slope = thrust_residual(env, thrust)
    pooled = floor_envelope_stats(resid)
    caps = floor_envelope_capsule_stats(resid)
    return dict(
        mic={k: caps[k] for k in MIC_STATS},
        xmic_corr=float(pooled["xmic_corr"]),
        spectrum_db=between_line_spectrum(power, pg.freqs, pg.rps),
        pooled=pooled,
        raw_pooled=floor_envelope_stats(env),
        cover=float(cover.mean()),
        thrust_slope=slope,
    )


def mean_statistics(stats: list[dict[str, Any]]) -> dict[str, Any]:
    """The seed average of :func:`wind_statistics` outputs of one segment."""

    def avg(get: Any) -> np.ndarray:
        return np.mean([np.asarray(get(s), dtype=np.float64) for s in stats], axis=0)

    return dict(
        mic={k: avg(lambda s, k=k: s["mic"][k]) for k in MIC_STATS},
        xmic_corr=float(avg(lambda s: s["xmic_corr"])),
        spectrum_db=avg(lambda s: s["spectrum_db"]),
        pooled={k: float(avg(lambda s, k=k: s["pooled"][k])) for k in stats[0]["pooled"]},
        raw_pooled={k: float(avg(lambda s, k=k: s["raw_pooled"][k])) for k in stats[0]["pooled"]},
        cover=float(avg(lambda s: s["cover"])),
        thrust_slope=avg(lambda s: s["thrust_slope"]),
        n_seeds=len(stats),
    )


def _arrays(st: dict[str, Any]) -> tuple[np.ndarray, float, np.ndarray]:
    mic = np.stack([np.asarray(st["mic"][k], dtype=np.float64) for k in MIC_STATS])  # (K, M)
    spec = np.asarray(st["spectrum_db"], dtype=np.float64)  # (M, B)
    return mic, float(st["xmic_corr"]), spec


def stat_scales(real: list[dict[str, Any]]) -> dict[str, Any]:
    """The distance's scales: the across-segment sd (``ddof=1``) of each
    per-capsule statistic pooled over the mics (root-mean variance), of
    ``xmic_corr``, and of the spectrum per band pooled over the mics, each
    floored at :data:`STAT_FLOORS` (one segment: the floors)."""
    mic = np.stack([_arrays(r)[0] for r in real])  # (S, K, M)
    xm = np.array([_arrays(r)[1] for r in real])
    spec = np.stack([_arrays(r)[2] for r in real])  # (S, M, B)
    if len(real) >= 2:
        sd_mic = np.sqrt(np.var(mic, axis=0, ddof=1).mean(axis=-1))
        sd_x = float(np.std(xm, ddof=1))
        sd_spec = np.sqrt(np.nanmean(np.var(spec, axis=0, ddof=1), axis=0))
    else:
        sd_mic, sd_x, sd_spec = np.zeros(len(MIC_STATS)), 0.0, np.zeros(spec.shape[-1])
    sd_spec = np.where(np.isfinite(sd_spec), sd_spec, 0.0)
    return dict(
        mic={k: max(float(s), STAT_FLOORS[k]) for k, s in zip(MIC_STATS, sd_mic, strict=True)},
        xmic_corr=max(sd_x, STAT_FLOORS["xmic_corr"]),
        spectrum_db=np.maximum(sd_spec, STAT_FLOORS["spectrum_db"]).tolist(),
        n_segments=len(real),
        floors=dict(STAT_FLOORS),
        rule="across-segment sd (ddof 1) of each statistic, the per-capsule ones pooled over the "
        "mics as the root-mean variance, the spectrum per band; floored at the floors (one "
        "segment: the floors alone)",
    )


def distance(
    real: list[dict[str, Any]], sim: list[dict[str, Any]], scales: dict[str, Any]
) -> dict[str, Any]:
    """The simulated-moment distance (module docstring, step 3): per segment
    ``sum_k sum_m ((sim - real) / s_k)^2`` over :data:`MIC_STATS`, ``M ((sim -
    real) / s_x)^2`` for ``xmic_corr`` and ``sum_m mean_b ((sim - real) /
    s_b)^2`` for the spectrum (bands without floor cells skipped), summed over
    the segments; with the total per statistic and per segment."""
    s_mic = np.array([scales["mic"][k] for k in MIC_STATS])[:, None]
    s_x = float(scales["xmic_corr"])
    s_spec = np.asarray(scales["spectrum_db"], dtype=np.float64)[None, :]
    by_stat = dict.fromkeys((*MIC_STATS, "xmic_corr", "spectrum_db"), 0.0)
    by_segment = []
    for r, s in zip(real, sim, strict=True):
        mr, xr, sr_ = _arrays(r)
        ms, xs, ss = _arrays(s)
        mic = (((ms - mr) / s_mic) ** 2).sum(axis=1)  # (K,)
        xmic = mr.shape[1] * ((xs - xr) / s_x) ** 2
        d2 = ((ss - sr_) / s_spec) ** 2
        ok = np.isfinite(d2)
        spec = float((np.where(ok, d2, 0.0).sum(axis=1) / np.maximum(ok.sum(axis=1), 1)).sum())
        for k, v in zip(MIC_STATS, mic, strict=True):
            by_stat[k] += float(v)
        by_stat["xmic_corr"] += float(xmic)
        by_stat["spectrum_db"] += spec
        by_segment.append(float(mic.sum() + xmic + spec))
    return dict(total=float(sum(by_segment)), by_stat=by_stat, by_segment=by_segment)


# ── the wind block, the frozen part and the amplitude ───────────────────────


def sc_shape(
    offset_ms: float,
    scale_ms: float,
    rate_hz: float,
    *,
    sr: int,
    clip_s: float = SHAPE_CLIP_S,
    n_mc: int = SHAPE_N_MC,
    seed: int = SHAPE_SEED,
    n_fft: int = SHAPE_N_FFT,
) -> dict[str, Any]:
    """The ``sc`` block's ``shape``: the mean periodic-Hann periodogram of
    ``n_mc`` unit-RMS :func:`.wind_sc.wind_noise` clips of ``clip_s`` at the law
    (one generator of ``seed``), divided by its mean over
    :data:`.v4.WIND_REF_BAND_HZ` (``unit_psd_ref``)."""
    rng = np.random.default_rng(int(seed))
    acc = np.zeros(int(n_fft) // 2 + 1)
    for _ in range(int(n_mc)):
        x, _ = wind_noise(
            rng,
            int(sr),
            float(clip_s),
            speed_offset_ms=float(offset_ms),
            speed_scale_ms=float(scale_ms),
            gust_rate_hz=float(rate_hz),
        )
        acc += mean_periodogram(x, int(n_fft))[0]
    p = acc / int(n_mc)
    f = np.fft.rfftfreq(int(n_fft), d=1.0 / float(sr))
    lo, hi = V4.WIND_REF_BAND_HZ
    unit = float(p[(f >= lo) & (f <= hi)].mean())
    return dict(
        freqs_hz=f.tolist(),
        psd=(p / unit).tolist(),
        ref_band_hz=list(V4.WIND_REF_BAND_HZ),
        unit_psd_ref=unit,
        n_mc=int(n_mc),
        seed=int(seed),
        clip_s=float(clip_s),
        n_fft=int(n_fft),
        sr=int(sr),
        rule="psd = the mean periodic-Hann periodogram |rfft(w x)|^2 / sum w^2 (hop n_fft/2) of "
        "n_mc unit-RMS wind_sc.wind_noise clips of clip_s at this law, divided by its mean over "
        "ref_band_hz (= unit_psd_ref); the renderer scales a unit-RMS draw by "
        "sqrt(10^{level_db/10} / unit_psd_ref), so its expected PSD is 10^{level_db/10} psd",
    )


def sc_block(
    level_db: Any, offset_ms: float, scale_ms: float, rate_hz: float, shape: dict[str, Any]
) -> dict[str, Any]:
    """The payload's ``params.wind`` of kind ``sc``."""
    return dict(
        kind=V4.WIND_SC,
        level_db=[float(x) for x in np.asarray(level_db, dtype=np.float64)],
        speed_offset_ms=float(offset_ms),
        speed_scale_ms=float(scale_ms),
        gust_rate_hz=float(rate_hz),
        shape=shape,
        level="level_db[m] is 10 log10 of capsule m's expected wind PSD averaged over "
        "shape.ref_band_hz (20-100 Hz), in the model's periodogram units (the static wind_db's "
        "on its flat part), before the array response; one independent SC draw per capsule",
    )


def with_wind(payload: dict[str, Any], wind: dict[str, Any] | None) -> dict[str, Any]:
    """``payload`` with ``params.wind`` replaced (``None``: no wind)."""
    out = copy.deepcopy(payload)
    out["params"]["wind"] = None if wind is None else copy.deepcopy(wind)
    return out


def wind_only(payload: dict[str, Any], wind: dict[str, Any]) -> dict[str, Any]:
    """``payload`` with the wind block ``wind`` and every line profile, the
    floor and its static share silenced (:data:`.dregon_flight_v4.SILENT_DB`:
    their powers are exactly 0)."""
    out = DF.silent_floor(with_wind(payload, wind))
    p = out["params"]
    for key in ("profile_aero_db", "profile_motor_db"):
        p[key] = np.full(np.shape(p[key]), DF.SILENT_DB).tolist()
    return out


def source_level_db(payload: dict[str, Any]) -> np.ndarray:
    """``(M,)`` the source wind's levels, the level prior's centre: an ``sc``
    block's ``level_db``; a static one's 10 log10 of its PSD averaged over
    20-100 Hz (the ``wind_db`` itself: its shape is one there)."""
    par = V4.V4Params.from_payload(payload)
    if par.wind is None:
        raise ValueError("the source payload carries no flight wind")
    if par.wind.kind == V4.WIND_SC:
        return np.asarray(par.wind.level_db, dtype=np.float64).copy()
    f = np.linspace(*V4.WIND_REF_BAND_HZ, 801)
    return 10.0 * np.log10(par.wind.psd(f).mean(axis=1))


@dataclass
class LevelData:
    """The amplitude fit's data: the stage pool's band cells and ``E0``."""

    power: np.ndarray  #: (M, N, Fb) the pool periodograms on the band
    frozen: np.ndarray  #: (M, N, Fb) ``E0``: the source payload without its wind
    weights: np.ndarray  #: (N,) the pool's composite exposure weights
    band: np.ndarray  #: (Fb,) the band's bin indices on the pool grid
    centre_db: np.ndarray  #: (M,) the level prior's centre
    n_fft: int
    sr: int
    #: (M, N, Fb) ``E_floor``: the floor and its static share of ``E0`` at the
    #: source's ``mean_db`` (battery mode's floor-mean refit; ``None``: held)
    floor: np.ndarray | None = None
    floor_centre_db: float | None = None  #: the source's ``floor.mean_db``


def frozen_expectation(
    payload_nowind: dict[str, Any],
    rps: np.ndarray,
    *,
    n_fft: int,
    sr: int,
    band: np.ndarray,
    progress: int = 0,
) -> np.ndarray:
    """``(M, N, Fb)`` ``v4.expected_periodogram`` of the wind-free payload at
    each frame's speeds ``rps (R, N)``, on the band's bins."""
    out = []
    t0 = time.time()
    for i in range(rps.shape[1]):
        e = V4.expected_periodogram(payload_nowind, rps[:, i], n_fft=int(n_fft), sr=int(sr))
        out.append(e[:, band])
        if progress and (i + 1) % int(progress) == 0:
            print(f"  E0 {i + 1}/{rps.shape[1]}  {time.time() - t0:.1f}s", flush=True)
    return np.stack(out, axis=1)


def wind_transform(
    payload: dict[str, Any], wind: dict[str, Any], *, n_fft: int, sr: int, band: np.ndarray
) -> np.ndarray:
    """``(M, Fb)`` the expected periodogram of ``wind`` at 0 dB on every mic
    (its window transform times the array response): ``expected_periodogram``
    of the payload with the wind alone, on the band's bins."""
    unit = dict(wind, level_db=[0.0] * len(wind["level_db"]))
    pay = wind_only(payload, unit)
    rps = np.full(int(pay["n_rotors"]), float(pay.get("f_ref_rev_s", 80.0)))
    return V4.expected_periodogram(pay, rps, n_fft=int(n_fft), sr=int(sr))[:, band]


def _scan_brent(nll: Any, centre: float, start: float | None = None) -> tuple[float, float, bool]:
    """The 1-D search of the inner fits: a 1 dB scan over ``centre +-
    LEVEL_SPAN_DB``, then bounded Brent on the best point's bracket; ``(x,
    nll(x), the scan's best point was an edge)``. With ``start`` (a later
    sweep) the scan is ``start +- INNER_LOCAL_SPAN_DB`` first, the full one only
    when its best point is an edge."""
    for c, span in ((start, INNER_LOCAL_SPAN_DB), (centre, LEVEL_SPAN_DB)):
        if c is None:
            continue
        scan = float(c) + np.arange(-span, span + 0.5, 1.0)
        vals = np.array([nll(x) for x in scan])
        j = int(np.argmin(vals))
        edge = bool(j in (0, scan.size - 1))
        if edge and span != LEVEL_SPAN_DB:
            continue
        lo, hi = scan[max(j - 1, 0)], scan[min(j + 1, scan.size - 1)]
        res = minimize_scalar(nll, bounds=(float(lo), float(hi)), method="bounded")
        return float(res.x), float(res.fun), edge
    raise AssertionError("unreachable: the full scan always returns")


def fit_levels(
    data: LevelData, transform: np.ndarray, *, start: np.ndarray | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Per mic the MAP level ``L_m`` of ``E = E0 + 10^{L_m/10} T_m`` under the
    composite Whittle risk on the pool (``composite_risk``) and ``L_m ~
    N(centre_m, LEVEL_PRIOR_SD_DB)``: a 1 dB scan over ``centre +-
    LEVEL_SPAN_DB`` (``start``: first around it, :func:`_scan_brent`), then
    bounded Brent on the best point's bracket."""
    levels, rows = [], []
    sd = LEVEL_PRIOR_SD_DB
    for m in range(data.power.shape[0]):
        c = float(data.centre_db[m])
        p, e0 = data.power[m : m + 1], data.frozen[m : m + 1]
        t = np.asarray(transform[m], dtype=np.float64)[None, None, :]

        def nll(
            level: float,
            p: np.ndarray = p,
            e0: np.ndarray = e0,
            t: np.ndarray = t,
            c: float = c,
        ) -> float:
            model = e0 + 10.0 ** (float(level) / 10.0) * t
            return float(composite_risk(p, model, data.weights)) + 0.5 * ((level - c) / sd) ** 2

        level, fun, edge = _scan_brent(nll, c, None if start is None else float(start[m]))
        levels.append(level)
        rows.append(
            dict(
                mic=m,
                level_db=level,
                centre_db=c,
                objective=fun,
                gain_vs_centre=float(nll(c) - fun),
                at_scan_edge=edge,
            )
        )
    return np.asarray(levels), dict(
        mics=rows,
        prior=dict(family="Normal", centre="the source wind's 20-100 Hz level", sd_db=sd),
        search=f"1 dB scan over centre +-{LEVEL_SPAN_DB:g} dB, bounded Brent on the bracket",
        rule="E = E0 + 10^{L/10} T, E0 the source payload's expected periodogram without its "
        "wind, T the sc wind at 0 dB (window transform x array response); composite Whittle "
        "risk on the flight stage pool and band, plus the prior; per mic",
    )


def fit_levels_floor(
    data: LevelData, transform: np.ndarray
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Battery mode's inner fit (module docstring, battery step 2): the per-mic
    levels ``L_m`` and the one floor ``mean_db`` ``mu`` of ``E = E_lines +
    10^{(mu - mu_src)/10} E_floor + 10^{L_m/10} T_m`` (``E_floor`` =
    ``data.floor``, ``E_lines = E0 - E_floor``) under the composite Whittle risk
    plus the priors ``L_m ~ N(centre_m, LEVEL_PRIOR_SD_DB)`` and ``mu ~
    N(mu_src, FLOOR_MEAN_PRIOR_SD_DB)``. Coordinate descent: :func:`fit_levels`
    at ``mu``, then ``mu`` at those levels (the same scan + Brent around
    ``mu_src``), until no value moves by more than :data:`INNER_TOL_DB` or
    :data:`INNER_MAX_SWEEPS` sweeps; ``(levels, mu, record)``."""
    if data.floor is None or data.floor_centre_db is None:
        raise ValueError("the floor-mean refit needs LevelData.floor and floor_centre_db")
    mu0, sd_f = float(data.floor_centre_db), FLOOR_MEAN_PRIOR_SD_DB
    centre = np.asarray(data.centre_db, dtype=np.float64)
    lines = data.frozen - data.floor
    t = np.asarray(transform, dtype=np.float64)[:, None, :]

    def floor_gain(mu: float) -> float:
        return 10.0 ** ((float(mu) - mu0) / 10.0)

    def total(levels: np.ndarray, mu: float) -> float:
        model = lines + floor_gain(mu) * data.floor + 10.0 ** (levels / 10.0)[:, None, None] * t
        prior = 0.5 * float(np.sum(((levels - centre) / LEVEL_PRIOR_SD_DB) ** 2))
        prior += 0.5 * ((float(mu) - mu0) / sd_f) ** 2
        return float(composite_risk(data.power, model, data.weights)) + prior

    levels, mu = centre.copy(), mu0
    sweeps: list[dict[str, Any]] = []
    fit: dict[str, Any] = {}
    mu_fun, mu_edge, converged = math.nan, False, False
    for sweep in range(INNER_MAX_SWEEPS):
        frozen = lines + floor_gain(mu) * data.floor
        local = None if sweep == 0 else levels
        new_levels, fit = fit_levels(replace(data, frozen=frozen), transform, start=local)
        rest = lines + 10.0 ** (new_levels / 10.0)[:, None, None] * t

        def nll_mu(x: float, rest: np.ndarray = rest) -> float:
            model = rest + floor_gain(x) * data.floor
            return float(composite_risk(data.power, model, data.weights)) + (
                0.5 * ((float(x) - mu0) / sd_f) ** 2
            )

        new_mu, mu_fun, mu_edge = _scan_brent(nll_mu, mu0, None if sweep == 0 else mu)
        move = max(float(np.abs(new_levels - levels).max()), abs(new_mu - mu))
        levels, mu = new_levels, new_mu
        sweeps.append(
            dict(sweep=sweep, floor_mean_db=mu, level_db=levels.tolist(), max_move_db=move)
        )
        if move < INNER_TOL_DB:
            converged = True
            break
    record = dict(
        mics=fit["mics"],
        floor_mean=dict(
            mean_db=mu,
            centre_db=mu0,
            shift_db=mu - mu0,
            sd_db=sd_f,
            objective=mu_fun,
            at_scan_edge=mu_edge,
        ),
        objective=total(levels, mu),
        objective_at_source=total(centre, mu0),
        sweeps=sweeps,
        n_sweeps=len(sweeps),
        converged=converged,
        tol_db=INNER_TOL_DB,
        prior=dict(
            family="Normal",
            levels=dict(centre="the source wind's levels", sd_db=LEVEL_PRIOR_SD_DB),
            floor_mean=dict(centre="the source's floor.mean_db", sd_db=sd_f),
        ),
        search=f"coordinate descent (levels at mu, then mu at the levels), each a 1 dB scan over "
        f"its prior centre +-{LEVEL_SPAN_DB:g} dB (later sweeps: +-{INNER_LOCAL_SPAN_DB:g} dB "
        "around the current value first, the full scan when its best point is an edge) and "
        f"bounded Brent on the bracket; stop when nothing moves by more than {INNER_TOL_DB:g} dB "
        f"(at most {INNER_MAX_SWEEPS} sweeps)",
        rule="E = E_lines + 10^{(mu - mu_src)/10} E_floor + 10^{L_m/10} T_m: E_floor the floor "
        "and its static share of the source's expected periodogram at its mean_db, E_lines the "
        "rest without the wind, T_m the sc wind at 0 dB; composite Whittle risk on the pool and "
        "band, plus the priors",
    )
    return levels, mu, record


# ── the restartable grid ────────────────────────────────────────────────────


def base_seed(segment: int, k: int) -> int:
    return BASE_SEED + 100 * int(segment) + int(k)


def wind_seed(segment: int, k: int) -> int:
    return WIND_SEED + 100 * int(segment) + int(k)


def _uid(offset: float, scale: float, rate: float) -> str:
    return f"o{offset:g}__s{scale:g}__r{rate:g}"


@dataclass
class Context:
    """Everything a grid unit reads: the source payload, the segments' labels,
    the real targets, the scales and the level data."""

    config: dict[str, Any]
    stage: dict[str, Any]
    names: list[str]
    rps: np.ndarray  #: (S, R, T) the segments' labels on the audio grid
    sr: int
    n_mics: int
    real: list[dict[str, Any]]
    scales: dict[str, Any]
    level: LevelData
    meta: dict[str, Any]


def _source_check(stage: dict[str, Any]) -> None:
    if stage.get("schema") != FIT_SCHEMA_V4 or stage.get("kind") != "flight":
        raise ValueError(
            f"the wind stage starts from a v4 FLIGHT payload, got schema {stage.get('schema')!r} "
            f"kind {stage.get('kind')!r}"
        )
    if stage.get("rig") != "dregon":
        raise ValueError(f"the wind stage is DREGON's, got rig {stage.get('rig')!r}")
    if V4.V4Params.from_payload(stage).wind is None:
        raise ValueError("the source payload carries no flight wind (the level prior's centre)")


def build_context(
    work: Path, stage: dict[str, Any], *, max_segments: int | None, progress: int = 0
) -> None:
    """Write ``context.npz`` / ``context.json`` unless they exist: the segments'
    labels and real statistics (step 1), the scales, and the level data (the
    pool frames of those segments, ``E0``, the prior centres)."""
    if (work / "context.npz").exists() and (work / "context.json").exists():
        return
    t0 = time.time()
    segs = load_segments(max_segments)
    if len({s.audio.shape[1] for s in segs}) != 1:
        raise ValueError("the pool segments differ in length")
    real = [_plain(wind_statistics(s.audio, s.rps, s.sr)) for s in segs]
    if progress:
        print(f"real targets of {len(segs)} segments  {time.time() - t0:.1f}s", flush=True)
    pool = DF.load_pool(stride=DF.FRAME_STRIDE)
    if list(pool.names[: len(segs)]) != [s.name for s in segs]:
        raise AssertionError(
            f"pool order {pool.names} is not the segments' {[s.name for s in segs]}"
        )
    keep = pool.frame_window < len(segs)
    f = np.asarray(pool.freqs_hz, dtype=np.float64)
    band_hz = (float(stage["band_hz"][0]), float(stage["band_hz"][1]))
    band = np.flatnonzero((f >= band_hz[0]) & (f <= band_hz[1]))
    frozen = frozen_expectation(
        with_wind(stage, None),
        pool.rps[:, keep],
        n_fft=pool.n_fft,
        sr=pool.sr,
        band=band,
        progress=progress * 50,
    )
    if progress:
        print(f"E0 on {int(keep.sum())} pool frames  {time.time() - t0:.1f}s", flush=True)
    tmp = work / f"context.{os.getpid()}.tmp.npz"
    np.savez(
        tmp,
        rps=np.stack([s.rps for s in segs]),
        power=np.asarray(pool.power, dtype=np.float64)[:, keep][:, :, band],
        frozen=frozen,
        weights=np.asarray(pool.weights, dtype=np.float64)[keep],
        band=band,
        centre=source_level_db(stage),
    )
    os.replace(tmp, work / "context.npz")
    diag = {k: v for k, v in pool.diagnostics().items() if k != "frames"}
    _atomic_json(
        work / "context.json",
        dict(
            segments=[s.name for s in segs],
            sr=int(segs[0].sr),
            n_mics=int(segs[0].audio.shape[0]),
            n_samples=int(segs[0].audio.shape[1]),
            real=real,
            scales=stat_scales(real),
            pool=dict(diag, n_frames_wind_stage=int(keep.sum())),
            band_hz=list(band_hz),
            n_fft=int(pool.n_fft),
            wall_s=time.time() - t0,
        ),
    )


@functools.lru_cache(maxsize=2)
def _context(work: str) -> Context:
    w = Path(work)
    config = json.loads((w / "config.json").read_text())
    meta = json.loads((w / "context.json").read_text())
    stage = json.loads(Path(config["source"]).read_text())
    with np.load(w / "context.npz") as z:
        arrays = {k: np.asarray(z[k]) for k in z.files}
    level = LevelData(
        power=arrays["power"],
        frozen=arrays["frozen"],
        weights=arrays["weights"],
        band=arrays["band"],
        centre_db=arrays["centre"],
        n_fft=int(meta["n_fft"]),
        sr=int(meta["sr"]),
    )
    return Context(
        config=config,
        stage=stage,
        names=list(meta["segments"]),
        rps=arrays["rps"],
        sr=int(meta["sr"]),
        n_mics=int(meta["n_mics"]),
        real=meta["real"],
        scales=meta["scales"],
        level=level,
        meta=meta,
    )


def _base_path(work: str | Path, segment: int, k: int) -> Path:
    return Path(work) / "base" / "audio" / f"seg{int(segment)}__seed{int(k)}.npy"


@functools.lru_cache(maxsize=64)
def _base_audio(work: str, segment: int, k: int) -> np.ndarray:
    return np.load(_base_path(work, segment, k))


def base_unit(unit: Unit) -> dict[str, Any]:
    """One wind-free render of the source payload on a segment's label (the
    grid reuses it: the wind is additive and independent)."""
    work, i, k = str(unit.params["work"]), int(unit.params["segment"]), int(unit.params["seed"])
    ctx = _context(work)
    t0 = time.time()
    seed = base_seed(i, k)
    x = render_noise(
        with_wind(ctx.stage, None), ctx.rps[i], sr=ctx.sr, n_mics=ctx.n_mics, seed=seed
    )
    path = _base_path(work, i, k)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp, x.astype(np.float32))
    os.replace(tmp, path)
    return dict(
        segment=i,
        name=ctx.names[i],
        seed_index=k,
        render_seed=seed,
        path=str(path),
        rms=np.sqrt((x**2).mean(axis=1)).tolist(),
        wall_s=time.time() - t0,
    )


def grid_point(ctx: Context, offset: float, scale: float, rate: float, *, work: str) -> dict:
    """Steps 2 and 3 at one law: the shape, the levels, the simulated
    statistics per segment (averaged over the seeds) and the distance."""
    t0 = time.time()
    cfg = ctx.config
    shape = sc_shape(
        offset, scale, rate, sr=ctx.sr, n_mc=int(cfg["n_mc"]), seed=int(cfg["shape_seed"])
    )
    unit_block = sc_block(np.zeros(ctx.n_mics), offset, scale, rate, shape)
    tw = wind_transform(
        ctx.stage, unit_block, n_fft=ctx.level.n_fft, sr=ctx.level.sr, band=ctx.level.band
    )
    levels, level_fit = fit_levels(ctx.level, tw)
    t_level = time.time() - t0
    pay = with_wind(ctx.stage, sc_block(levels, offset, scale, rate, shape))
    n = int(ctx.rps.shape[-1])
    sims = []
    for i in range(len(ctx.names)):
        per_seed = []
        for k in range(int(cfg["seeds"])):
            audio = _base_audio(work, i, k).astype(np.float64) + render_v4_wind(
                pay, n, sr=ctx.sr, n_mics=ctx.n_mics, seed=wind_seed(i, k)
            )
            per_seed.append(wind_statistics(audio, ctx.rps[i], ctx.sr))
        sims.append(mean_statistics(per_seed))
    d = distance(ctx.real, sims, ctx.scales)
    return dict(
        speed_offset_ms=float(offset),
        speed_scale_ms=float(scale),
        gust_rate_hz=float(rate),
        level_db=levels.tolist(),
        level_fit=level_fit,
        distance=d["total"],
        distance_terms=d,
        sim=_plain(sims),
        shape=shape,
        wall_s=time.time() - t0,
        level_wall_s=t_level,
    )


def grid_unit(unit: Unit) -> dict[str, Any]:
    """:func:`grid_point` as a :mod:`utils.gridrun` unit."""
    p = unit.params
    ctx = _context(str(p["work"]))
    return dict(
        uid=unit.uid,
        **grid_point(
            ctx, float(p["offset"]), float(p["scale"]), float(p["rate"]), work=str(p["work"])
        ),
    )


def _grid_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    table = sorted(
        (
            dict(
                uid=r["uid"],
                law=[r["speed_offset_ms"], r["speed_scale_ms"], r["gust_rate_hz"]],
                distance=r["distance"],
                wall_s=round(float(r["wall_s"]), 1),
            )
            for r in rows
        ),
        key=lambda r: r["distance"],
    )
    return dict(n_units=len(rows), best=table[:5])


# ── the renderer check and the stage ────────────────────────────────────────


def wind_renderer_check(
    payload: dict[str, Any],
    *,
    n_seeds: int = CHECK_SEEDS,
    duration_s: float = SHAPE_CLIP_S,
    seed: int = CHECK_SEED,
    n_fft: int = SHAPE_N_FFT,
    band_hz: tuple[float, float] = CHECK_BAND_HZ,
    tol_db: float = CHECK_TOL_DB,
) -> dict[str, Any]:
    """The rendered wind (:func:`render_v4_wind`, the renderer's own path) of
    ``n_seeds`` clips: its mean periodogram (``n_fft``, hop ``n_fft / 2``)
    against ``10^{L/10} shape.psd`` times the array response, as band power in
    the third octaves inside ``band_hz``; passes when every band of every mic is
    within ``tol_db``."""
    par = V4.V4Params.from_payload(payload)
    if par.wind is None or par.wind.gust is None:
        raise ValueError("the wind renderer check needs an sc wind block")
    sr, m_ = int(par.sr), int(par.n_mics)
    n = int(round(float(duration_s) * sr))
    acc = np.zeros((m_, int(n_fft) // 2 + 1))
    for k in range(int(n_seeds)):
        acc += mean_periodogram(render_v4_wind(payload, n, sr=sr, n_mics=m_, seed=seed + k), n_fft)
    got = acc / int(n_seeds)
    f = np.fft.rfftfreq(int(n_fft), d=1.0 / float(sr))
    want = par.wind.psd(f) * 10.0 ** (par.mic.array_db(f) / 10.0)
    centres = [
        c
        for c in 1000.0 * 2.0 ** (np.arange(-17, 0) / 3.0)
        if c * 2.0 ** (-1.0 / 6.0) >= band_hz[0] and c <= band_hz[1]
    ]
    err = np.zeros((m_, len(centres)))
    for b, (lo, hi) in enumerate(_third_octave_edges(centres)):
        sel = (f >= lo) & (f < min(hi, band_hz[1]))
        err[:, b] = 10.0 * np.log10(got[:, sel].sum(axis=1) / want[:, sel].sum(axis=1))
    sel = (f >= band_hz[0]) & (f <= band_hz[1])
    return dict(
        n_seeds=int(n_seeds),
        seed=int(seed),
        duration_s=float(duration_s),
        n_fft=int(n_fft),
        band_hz=list(band_hz),
        band_centres_hz=[float(c) for c in centres],
        band_err_db=err.tolist(),
        max_abs_err_db=float(np.abs(err).max()),
        bin_rms_err_db=float(np.sqrt(np.mean((10.0 * np.log10(got[:, sel] / want[:, sel])) ** 2))),
        tol_db=float(tol_db),
        passed=bool(np.abs(err).max() < float(tol_db)),
        rule="render_v4_wind (the renderer's wind path: its own stream of each seed, the array "
        "response and the anti-alias) of n_seeds clips; mean periodic-Hann periodogram vs "
        "10^{level_db/10} shape.psd x 10^{A_m/10}, third-octave band power inside band_hz, "
        "max |error| over mics and bands < tol_db",
    )


def _prepare(work: Path, config: dict[str, Any]) -> None:
    path = work / "config.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old != config:
            raise ValueError(
                f"{work} holds another configuration ({old}); this run is {config}: pass "
                "another --out"
            )
        return
    _atomic_json(path, config)


def stats_table(real: list[dict[str, Any]], sim: list[dict[str, Any]]) -> list[str]:
    """Real vs simulated statistics, each the mean over segments and mics, and
    the spectrum's mean difference per band (dB)."""
    rows = [f"{'statistic':<14} {'real':>8} {'sim':>8}"]
    for k in MIC_STATS:
        r = float(np.mean([np.mean(s["mic"][k]) for s in real]))
        s_ = float(np.mean([np.mean(s["mic"][k]) for s in sim]))
        rows.append(f"{k:<14} {r:8.3f} {s_:8.3f}")
    r = float(np.mean([s["xmic_corr"] for s in real]))
    s_ = float(np.mean([s["xmic_corr"] for s in sim]))
    rows.append(f"{'xmic_corr':<14} {r:8.3f} {s_:8.3f}")
    diff = np.nanmean(
        [
            np.asarray(b["spectrum_db"]) - np.asarray(a["spectrum_db"])
            for a, b in zip(real, sim, strict=True)
        ],
        axis=(0, 1),
    )
    rows.append(
        "spectrum sim-real dB @ "
        + " ".join(f"{c:.0f}:{d:+.1f}" for c, d in zip(SPEC_CENTRES_HZ, diff, strict=True))
    )
    return rows


def run_wind_stage(
    source: str | Path,
    *,
    out: str | Path,
    offsets: tuple[float, ...] = OFFSETS_MS,
    scales: tuple[float, ...] = SCALES_MS,
    rates: tuple[float, ...] = RATES_HZ,
    seeds: int = N_SEEDS,
    workers: int = 1,
    max_segments: int | None = None,
    n_mc: int = SHAPE_N_MC,
    check_seeds: int = CHECK_SEEDS,
    resume: bool = True,
    progress: int = 0,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    """The whole stage on the source payload ``source``; writes and returns
    ``(payload, grid report, payload path, report path)``: ``<out>/<stem>__scwind.json``
    and ``<out>/<stem>__scwind.grid.json``, the work under ``<out>/<stem>__scwind/``."""
    t0 = time.time()
    source = Path(source)
    raw = source.read_bytes()
    stage = json.loads(raw)
    _source_check(stage)
    out = Path(out)
    work = out / f"{source.stem}__scwind"
    config = dict(
        source=str(source),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        max_segments=None if max_segments is None else int(max_segments),
        frame_stride=DF.FRAME_STRIDE,
        seeds=int(seeds),
        n_mc=int(n_mc),
        shape_seed=SHAPE_SEED,
        base_seed=BASE_SEED,
        wind_seed=WIND_SEED,
    )
    _prepare(work, config)
    build_context(work, stage, max_segments=max_segments, progress=progress)
    ctx = _context(str(work))
    n_seg = len(ctx.names)
    base = run_grid(
        [
            Unit(f"seg{i}__seed{k}", dict(work=str(work), segment=i, seed=k))
            for i in range(n_seg)
            for k in range(int(seeds))
        ],
        base_unit,
        work / "base",
        jobs=int(workers),
        resume=resume,
        verbose=bool(progress),
    )
    if base.n_failed:
        raise RuntimeError(f"{base.n_failed} base render(s) failed: see {work}/base/raw/*.err")
    units = [
        Unit(_uid(o, s, r), dict(work=str(work), offset=o, scale=s, rate=r))
        for o in offsets
        for s in scales
        for r in rates
    ]
    grid = run_grid(
        units,
        grid_unit,
        work / "grid",
        jobs=int(workers),
        resume=resume,
        summarize=_grid_summary,
        verbose=bool(progress),
    )
    rows = [
        json.loads(unit_path(work / "grid", u.uid).read_text())
        for u in units
        if unit_path(work / "grid", u.uid).exists()
    ]
    rows = [r for r in rows if math.isfinite(float(r["distance"]))]
    if not rows:
        raise RuntimeError(f"no grid point finished: see {work}/grid/raw/*.err")
    best = min(rows, key=lambda r: float(r["distance"]))
    law = (best["speed_offset_ms"], best["speed_scale_ms"], best["gust_rate_hz"])
    block = sc_block(best["level_db"], *law, best["shape"])
    pay = with_wind(stage, block)
    check = wind_renderer_check(pay, n_seeds=int(check_seeds))
    walls = np.array([float(r["wall_s"]) for r in rows])
    table = sorted(
        (
            dict(
                uid=r["uid"],
                speed_offset_ms=r["speed_offset_ms"],
                speed_scale_ms=r["speed_scale_ms"],
                gust_rate_hz=r["gust_rate_hz"],
                level_db=r["level_db"],
                distance=r["distance"],
                by_stat=r["distance_terms"]["by_stat"],
                by_segment=r["distance_terms"]["by_segment"],
                wall_s=r["wall_s"],
            )
            for r in rows
        ),
        key=lambda r: float(r["distance"]),
    )
    stem = f"{source.stem}__scwind"
    pay_path, report_path = out / f"{stem}.json", out / f"{stem}.grid.json"
    pay["wind_stage"] = dict(
        stage=STAGE,
        module=__name__,
        source=str(source),
        source_sha256=config["source_sha256"],
        source_git=stage.get("git"),
        source_wind=stage["params"]["wind"],
        frozen="every params block but wind: the source payload's (lines, floor, per-rotor and "
        "per-mic gains, coherence)",
        segments=ctx.names,
        targets=dict(
            front_end=dict(n_fft=ENV_N_FFT, hop=ENV_HOP, window="periodic Hann", sr=ctx.sr),
            envelope="experiments.stochastic_fit.diagnostics.floor_envelope, floor cells of "
            f"{ENV_BAND_HZ[0]:g}-{ENV_BAND_HZ[1]:g} Hz (+-5 Hz around every k r excluded)",
            thrust_regression=f"per mic OLS of the envelope on [1, the same mic's "
            f"{THRUST_BAND_HZ[0]:g}-{THRUST_BAND_HZ[1]:g} Hz band level in dB], residual re its "
            "median (real and simulated alike)",
            statistics=list(MIC_STATS) + ["xmic_corr", "spectrum_db"],
            spectrum_centres_hz=list(SPEC_CENTRES_HZ),
            spectrum="mean floor-cell power per third-octave band over every frame and cell, dB",
        ),
        real=ctx.real,
        scales=ctx.scales,
        amplitude=dict(
            level_db=best["level_db"],
            fit=best["level_fit"],
            band_hz=ctx.meta["band_hz"],
            n_fft=ctx.meta["n_fft"],
            pool=ctx.meta["pool"],
        ),
        gust_law=dict(
            grid=dict(
                speed_offset_ms=list(offsets),
                speed_scale_ms=list(scales),
                gust_rate_hz=list(rates),
            ),
            seeds=int(seeds),
            base_seed=BASE_SEED,
            wind_seed=WIND_SEED,
            seed_rule="segment i, seed k: base render seed base_seed + 100 i + k, sim wind seed "
            "wind_seed + 100 i + k, the same at every grid point (common random numbers)",
            distance_rule="sum over segments of ((sim - real) / scale)^2: each per-capsule "
            "statistic summed over the mics, xmic_corr x M, the spectrum as the mean over its "
            "bands per mic summed over the mics; sim = the seed average",
            chosen=dict(
                uid=best["uid"],
                speed_offset_ms=law[0],
                speed_scale_ms=law[1],
                gust_rate_hz=law[2],
                distance=best["distance"],
                terms=best["distance_terms"],
            ),
            n_points=len(units),
            n_finished=len(rows),
            table=[dict(uid=t["uid"], distance=t["distance"]) for t in table],
        ),
        fitted=best["sim"],
        renderer_check=check,
        grid_report=str(report_path),
        work_dir=str(work),
        point_wall_s=dict(median=float(np.median(walls)), max=float(walls.max())),
        wall_s=time.time() - t0,
        git=FV._git_sha(),
    )
    report = dict(
        stage=STAGE,
        source=str(source),
        payload=str(pay_path),
        config=config,
        real=ctx.real,
        scales=ctx.scales,
        table=table,
        points=[{k: v for k, v in r.items() if k != "shape"} for r in rows],
        grid_summary=grid.summary,
    )
    _atomic_json(pay_path, pay)
    _atomic_json(report_path, report)
    return pay, report, pay_path, report_path


# ── battery mode (module docstring, BATTERY MODE): the time-aware search ────


def battery_terms(
    real: np.ndarray, arm: np.ndarray, *, sr: int = SU.SR
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """The distances :data:`BATTERY_TERMS` of ``arm`` against the reference
    ``real`` (both ``(M, T)``, full band) and the whole ``summary`` of the two
    measures (``spectro_temporal``, ``level_distribution``)."""
    real = np.asarray(real, dtype=np.float64)
    arm = np.asarray(arm, dtype=np.float64)
    summ = dict(
        spectro_temporal=_plain(SIM.spectro_temporal(real, arm, sr=sr)["summary"]),
        level_distribution=_plain(SIM.level_distribution(real, arm, sr=sr)["summary"]),
    )
    terms = {}
    for name in BATTERY_TERMS:
        meas, key = name.split(".", 1)
        if key == "offdiag_corr_abs_diff":
            st = summ["spectro_temporal"]
            terms[name] = abs(float(st["arm_offdiag_corr"]) - float(st["real_offdiag_corr"]))
        else:
            terms[name] = float(summ[meas][key])
    return terms, summ


def _mean_summaries(rows: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    """The seed average of :func:`battery_terms` summaries."""
    return {
        meas: {k: float(np.mean([r[meas][k] for r in rows])) for k in rows[0][meas]}
        for meas in rows[0]
    }


def seed_floors(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """The seed floor ``f_j`` of every term (:data:`BATTERY_FORMULA`): its mean
    over the ``(segment, pair)`` rows of :func:`seed_floor_unit` (each
    ``segment``, ``pair`` and ``terms``), and per segment; every floor must be
    positive and finite (it divides the objective)."""
    segs = sorted({int(p["segment"]) for p in pairs})
    floor, by_segment = {}, {}
    for j in BATTERY_TERMS:
        v = float(np.mean([float(p["terms"][j]) for p in pairs]))
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError(f"the seed floor of {j} is {v}: it must be positive and finite")
        floor[j] = v
        by_segment[j] = [
            float(np.mean([float(p["terms"][j]) for p in pairs if int(p["segment"]) == i]))
            for i in segs
        ]
    return dict(floor=floor, by_segment=by_segment, segments=segs, n_pairs=len(pairs))


def battery_objective(
    terms: list[list[dict[str, float]]],
    floors: dict[str, float],
    weights: dict[str, float] = BATTERY_WEIGHTS,
) -> dict[str, Any]:
    """:data:`BATTERY_FORMULA` of the distances ``terms[i][k]`` (segment ``i``,
    seed ``k``, each a :func:`battery_terms` dict): ``total``, and per term its
    mean ``dbar_j``, floor, ``ratio = dbar_j / f_j``, weight and contribution
    ``w_j ratio``, and per segment ``D_i`` (``total`` is their mean)."""
    by_term = {}
    for j in BATTERY_TERMS:
        mean = float(np.mean([[float(t[j]) for t in seg] for seg in terms]))
        ratio = mean / float(floors[j])
        by_term[j] = dict(
            mean=mean,
            floor=float(floors[j]),
            ratio=ratio,
            weight=float(weights[j]),
            contribution=float(weights[j]) * ratio,
        )
    by_segment = [
        float(
            sum(
                float(weights[j]) * float(np.mean([float(t[j]) for t in seg])) / float(floors[j])
                for j in BATTERY_TERMS
            )
        )
        for seg in terms
    ]
    return dict(
        total=float(sum(v["contribution"] for v in by_term.values())),
        by_term=by_term,
        by_segment=by_segment,
    )


def source_law(stage: dict[str, Any]) -> tuple[float, float, float] | None:
    """The source's own gust law ``(offset, scale, rate)`` when its wind is ``sc``."""
    wind = stage["params"].get("wind") or {}
    if wind.get("kind") != V4.WIND_SC:
        return None
    return (
        float(wind["speed_offset_ms"]),
        float(wind["speed_scale_ms"]),
        float(wind["gust_rate_hz"]),
    )


def grid_laws(offsets: Any, scales: Any, rates: Any, stage: dict[str, Any]) -> list[dict[str, Any]]:
    """The search's laws in grid order, each ``uid``, ``law`` and ``source``:
    the point whose uid is the sc source's own law carries the source's exact
    law (and later its shape); the source's law is appended when the grid
    misses it."""
    src = source_law(stage)
    src_uid = None if src is None else _uid(*src)
    out = []
    for o in offsets:
        for s in scales:
            for r in rates:
                law = (float(o), float(s), float(r))
                is_src = _uid(*law) == src_uid
                out.append(dict(uid=_uid(*law), law=src if is_src else law, source=is_src))
    if src is not None and not any(x["source"] for x in out):
        out.append(dict(uid=src_uid, law=src, source=True))
    return out


def gust_payload(stage: dict[str, Any], point: dict[str, Any]) -> dict[str, Any]:
    """The source payload at a battery point: ``params.wind`` its ``sc`` block
    (levels, law, shape) and ``params.floor.mean_db`` its floor mean."""
    law = (point["speed_offset_ms"], point["speed_scale_ms"], point["gust_rate_hz"])
    pay = with_wind(stage, sc_block(point["level_db"], *law, point["shape"]))
    pay["params"]["floor"]["mean_db"] = float(point["floor_mean_db"])
    return pay


@dataclass
class BatteryContext:
    """Everything a battery grid point reads (:func:`battery_point`)."""

    stage: dict[str, Any]  #: the source payload
    names: list[str]
    rps: np.ndarray  #: (S, R, T) the segments' labels on the audio grid
    real: np.ndarray  #: (S, M, T) the real segments
    sr: int
    n_mics: int
    seeds: int
    n_mc: int
    shape_seed: int
    level: LevelData  #: with ``floor`` and ``floor_centre_db``
    floors: dict[str, float]  #: the seed floor of every term
    weights: dict[str, float]
    #: ``(segment, seed index) -> (lines, floor)``, ``(M, T)`` float64: the
    #: source without its wind at base seed ``(i, k)``, split
    base: Any


def battery_point(
    ctx: BatteryContext, offset: float, scale: float, rate: float, *, source: bool = False
) -> dict[str, Any]:
    """Battery steps 2 and 3 at one law: its shape (``source``: the source
    payload's own), the inner fit (:func:`fit_levels_floor`), the sims of every
    segment and seed and the objective (:func:`battery_objective`)."""
    t0 = time.time()
    if source:
        shape = copy.deepcopy(ctx.stage["params"]["wind"]["shape"])
    else:
        shape = sc_shape(offset, scale, rate, sr=ctx.sr, n_mc=ctx.n_mc, seed=ctx.shape_seed)
    t_shape = time.time() - t0
    unit_block = sc_block(np.zeros(ctx.n_mics), offset, scale, rate, shape)
    tw = wind_transform(
        ctx.stage, unit_block, n_fft=ctx.level.n_fft, sr=ctx.level.sr, band=ctx.level.band
    )
    levels, mu, inner = fit_levels_floor(ctx.level, tw)
    t_inner = time.time() - t0 - t_shape
    pay = with_wind(ctx.stage, sc_block(levels, offset, scale, rate, shape))
    amp = 10.0 ** ((mu - float(ctx.level.floor_centre_db)) / 20.0)
    n = int(ctx.real.shape[-1])
    terms, summaries = [], []
    t_render = t_battery = 0.0
    for i in range(len(ctx.names)):
        real = np.asarray(ctx.real[i], dtype=np.float64)
        seg_terms, seg_summ = [], []
        for k in range(int(ctx.seeds)):
            t1 = time.time()
            lines, floor = ctx.base(i, k)
            wind = render_v4_wind(pay, n, sr=ctx.sr, n_mics=ctx.n_mics, seed=wind_seed(i, k))
            sim = lines + amp * floor + wind
            t2 = time.time()
            tj, sj = battery_terms(real, sim, sr=ctx.sr)
            t_render += t2 - t1
            t_battery += time.time() - t2
            seg_terms.append(tj)
            seg_summ.append(sj)
        terms.append(seg_terms)
        summaries.append(_mean_summaries(seg_summ))
    obj = battery_objective(terms, ctx.floors, ctx.weights)
    return _plain(
        dict(
            speed_offset_ms=float(offset),
            speed_scale_ms=float(scale),
            gust_rate_hz=float(rate),
            source_law=bool(source),
            level_db=levels,
            floor_mean_db=float(mu),
            floor_shift_db=float(mu - float(ctx.level.floor_centre_db)),
            inner=inner,
            objective=obj["total"],
            objective_terms=obj,
            terms=terms,
            summaries=summaries,
            shape=shape,
            wall_s=time.time() - t0,
            wall_parts=dict(
                shape_s=t_shape, inner_s=t_inner, render_s=t_render, battery_s=t_battery
            ),
        )
    )


def resolve_pool(pool: str) -> tuple[str, str | None]:
    """``(the pool used, a fallback note)``: ``wide`` needs
    ``dregon_flight_v4.wide_pool_specs``; without it the floor pool is used."""
    if pool not in POOLS:
        raise ValueError(f"pool must be one of {POOLS}, got {pool!r}")
    if pool == "wide" and not hasattr(DF, "wide_pool_specs"):
        return "floor", (
            "dregon_flight_v4.wide_pool_specs() does not exist in this checkout: the inner fit "
            "fell back to the floor pool"
        )
    return pool, None


def inner_pool(pool: str, names: list[str]) -> tuple[DF.FlightPool, np.ndarray, dict[str, Any]]:
    """The inner fit's pool, its frame mask and a description: ``floor`` the
    tuning segments' frames (``names``, the first of ``pool_specs()``),
    ``wide`` every frame of ``dregon_flight_v4.wide_pool_specs()``."""
    if pool == "wide":
        fp = DF.load_pool(stride=DF.FRAME_STRIDE, specs=DF.wide_pool_specs())
        keep = np.ones(fp.n_frames, dtype=bool)
        return fp, keep, dict(pool="wide", frames="every frame of wide_pool_specs()")
    fp = DF.load_pool(stride=DF.FRAME_STRIDE)
    if list(fp.names[: len(names)]) != list(names):
        raise AssertionError(f"pool order {fp.names} is not the segments' {names}")
    keep = fp.frame_window < len(names)
    return fp, keep, dict(pool="floor", frames="the tuning segments' frames of pool_specs()")


def _save_npy(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp, x)
    os.replace(tmp, path)


def build_battery_context(
    work: Path,
    stage: dict[str, Any],
    *,
    max_segments: int | None,
    pool: str,
    progress: int = 0,
) -> None:
    """Write ``context.npz`` / ``context.json`` unless they exist: the tuning
    segments (audio and labels) and the inner fit's data (the pool's band
    cells, ``E0`` and its floor share ``E_floor``, the prior centres)."""
    if (work / "context.npz").exists() and (work / "context.json").exists():
        return
    t0 = time.time()
    segs = load_segments(max_segments)
    if len({s.audio.shape for s in segs}) != 1:
        raise ValueError("the tuning segments differ in shape")
    fp, keep, pool_info = inner_pool(pool, [s.name for s in segs])
    f = np.asarray(fp.freqs_hz, dtype=np.float64)
    band_hz = (float(stage["band_hz"][0]), float(stage["band_hz"][1]))
    band = np.flatnonzero((f >= band_hz[0]) & (f <= band_hz[1]))
    nowind = with_wind(stage, None)
    kw = dict(n_fft=fp.n_fft, sr=fp.sr, band=band, progress=progress * 50)
    frozen = frozen_expectation(nowind, fp.rps[:, keep], **kw)
    lines = frozen_expectation(DF.silent_floor(nowind), fp.rps[:, keep], **kw)
    if progress:
        print(f"E0 and E_lines on {int(keep.sum())} pool frames  {time.time() - t0:.1f}s")
    tmp = work / f"context.{os.getpid()}.tmp.npz"
    np.savez(
        tmp,
        rps=np.stack([s.rps for s in segs]),
        real=np.stack([s.audio for s in segs]).astype(np.float32),
        power=np.asarray(fp.power, dtype=np.float64)[:, keep][:, :, band],
        frozen=frozen,
        floor=frozen - lines,
        weights=np.asarray(fp.weights, dtype=np.float64)[keep],
        band=band,
        centre=source_level_db(stage),
    )
    os.replace(tmp, work / "context.npz")
    diag = {k: v for k, v in fp.diagnostics().items() if k != "frames"}
    _atomic_json(
        work / "context.json",
        dict(
            segments=[s.name for s in segs],
            sr=int(segs[0].sr),
            n_mics=int(segs[0].audio.shape[0]),
            n_samples=int(segs[0].audio.shape[1]),
            pool=dict(diag, **pool_info, n_frames_inner=int(keep.sum())),
            band_hz=list(band_hz),
            n_fft=int(fp.n_fft),
            floor_centre_db=float(stage["params"]["floor"]["mean_db"]),
            wall_s=time.time() - t0,
        ),
    )


@functools.lru_cache(maxsize=2)
def _battery_inputs(work: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """``(config, context.json, the source payload)`` of a battery work dir."""
    w = Path(work)
    config = json.loads((w / "config.json").read_text())
    meta = json.loads((w / "context.json").read_text())
    return config, meta, json.loads(Path(config["source"]).read_text())


@functools.lru_cache(maxsize=2)
def _battery_arrays(work: str) -> dict[str, np.ndarray]:
    with np.load(Path(work) / "context.npz") as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _split_paths(work: str | Path, segment: int, k: int) -> tuple[Path, Path]:
    name = f"seg{int(segment)}__seed{int(k)}.npy"
    return Path(work) / "base" / "lines" / name, Path(work) / "base" / "floor" / name


@functools.lru_cache(maxsize=64)
def _split_audio(work: str, segment: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    lines, floor = _split_paths(work, segment, k)
    return np.load(lines), np.load(floor)


def _split_base(work: str, segment: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    lines, floor = _split_audio(work, int(segment), int(k))
    return lines.astype(np.float64), floor.astype(np.float64)


def split_base_unit(unit: Unit) -> dict[str, Any]:
    """One render of the source without its wind on a segment's label, split
    into its lines (the same seed with the floor silenced) and its floor (the
    rest)."""
    work, i, k = str(unit.params["work"]), int(unit.params["segment"]), int(unit.params["seed"])
    _, meta, stage = _battery_inputs(work)
    rps = _battery_arrays(work)["rps"][i]
    t0 = time.time()
    seed = base_seed(i, k)
    nowind = with_wind(stage, None)
    kw = dict(sr=int(meta["sr"]), n_mics=int(meta["n_mics"]), seed=seed)
    full = render_noise(nowind, rps, **kw)
    lines = render_noise(DF.silent_floor(nowind), rps, **kw)
    floor = full - lines
    p_lines, p_floor = _split_paths(work, i, k)
    _save_npy(p_lines, lines.astype(np.float32))
    _save_npy(p_floor, floor.astype(np.float32))
    return dict(
        segment=i,
        name=meta["segments"][i],
        seed_index=k,
        render_seed=seed,
        rms_lines=np.sqrt((lines**2).mean(axis=1)).tolist(),
        rms_floor=np.sqrt((floor**2).mean(axis=1)).tolist(),
        wall_s=time.time() - t0,
    )


def floor_seed(segment: int, q: int) -> int:
    return FLOOR_SEED + 100 * int(segment) + int(q)


def seed_floor_unit(unit: Unit) -> dict[str, Any]:
    """One seed-floor pair: the source payload as it is rendered at
    ``floor_seed(i, 2p)`` and ``floor_seed(i, 2p + 1)`` on segment ``i``'s
    label, the first as the reference (:func:`battery_terms`)."""
    work, i, p = str(unit.params["work"]), int(unit.params["segment"]), int(unit.params["pair"])
    _, meta, stage = _battery_inputs(work)
    rps = _battery_arrays(work)["rps"][i]
    t0 = time.time()
    seeds = [floor_seed(i, 2 * p), floor_seed(i, 2 * p + 1)]
    za, zb = (
        render_noise(stage, rps, sr=int(meta["sr"]), n_mics=int(meta["n_mics"]), seed=s)
        for s in seeds
    )
    terms, summ = battery_terms(za, zb, sr=int(meta["sr"]))
    return dict(
        segment=i,
        name=meta["segments"][i],
        pair=p,
        render_seeds=seeds,
        terms=terms,
        summaries=summ,
        wall_s=time.time() - t0,
    )


@functools.lru_cache(maxsize=2)
def _battery_context(work: str) -> BatteryContext:
    config, meta, stage = _battery_inputs(work)
    arrays = _battery_arrays(work)
    floors = json.loads((Path(work) / "seed_floor.json").read_text())
    level = LevelData(
        power=arrays["power"],
        frozen=arrays["frozen"],
        weights=arrays["weights"],
        band=arrays["band"],
        centre_db=arrays["centre"],
        n_fft=int(meta["n_fft"]),
        sr=int(meta["sr"]),
        floor=arrays["floor"],
        floor_centre_db=float(meta["floor_centre_db"]),
    )
    return BatteryContext(
        stage=stage,
        names=list(meta["segments"]),
        rps=arrays["rps"],
        real=arrays["real"],
        sr=int(meta["sr"]),
        n_mics=int(meta["n_mics"]),
        seeds=int(config["seeds"]),
        n_mc=int(config["n_mc"]),
        shape_seed=int(config["shape_seed"]),
        level=level,
        floors=floors["floor"],
        weights=config["weights"],
        base=functools.partial(_split_base, work),
    )


def battery_unit(unit: Unit) -> dict[str, Any]:
    """:func:`battery_point` as a :mod:`utils.gridrun` unit."""
    p = unit.params
    ctx = _battery_context(str(p["work"]))
    law = (float(p["offset"]), float(p["scale"]), float(p["rate"]))
    return dict(uid=unit.uid, **battery_point(ctx, *law, source=bool(p["source"])))


def _battery_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    table = sorted(
        (
            dict(
                uid=r["uid"],
                law=[r["speed_offset_ms"], r["speed_scale_ms"], r["gust_rate_hz"]],
                objective=r["objective"],
                source_law=r["source_law"],
                wall_s=round(float(r["wall_s"]), 1),
            )
            for r in rows
        ),
        key=lambda r: r["objective"] if math.isfinite(r["objective"]) else math.inf,
    )
    return dict(n_units=len(rows), best=table[:5])


def _run_until(
    units: list[Unit],
    worker: Any,
    out_dir: Path,
    *,
    workers: int,
    resume: bool,
    deadline: float | None,
    verbose: bool,
    summarize: Any = None,
) -> tuple[bool, int]:
    """``run_grid`` over ``units``; with a ``deadline`` (a ``time.time()``
    instant) in chunks of ``workers`` units, none launched when it would end
    past the deadline (its estimate: the slowest chunk so far). ``(every unit
    finished or failed, the number failed)``; a rerun resumes."""
    kw = dict(jobs=int(workers), resume=resume, summarize=summarize, verbose=verbose)
    if deadline is None:
        return True, run_grid(units, worker, out_dir, **kw).n_failed
    todo = [u for u in units if not (resume and unit_path(out_dir, u.uid).exists())]
    est, failed, step = 0.0, 0, max(1, int(workers))
    for c in range(0, len(todo), step):
        if time.time() + est > deadline:
            return False, failed
        t = time.time()
        failed += run_grid(todo[c : c + step], worker, out_dir, **kw).n_failed
        est = max(est, time.time() - t)
    return True, failed


def _gust_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every finished point, best first: its law, inner fit, objective and every term."""
    table = [
        dict(
            uid=r["uid"],
            speed_offset_ms=r["speed_offset_ms"],
            speed_scale_ms=r["speed_scale_ms"],
            gust_rate_hz=r["gust_rate_hz"],
            source_law=r["source_law"],
            level_db=r["level_db"],
            floor_mean_db=r["floor_mean_db"],
            floor_shift_db=r["floor_shift_db"],
            inner_converged=r["inner"]["converged"],
            inner_sweeps=r["inner"]["n_sweeps"],
            objective=r["objective"],
            by_term=r["objective_terms"]["by_term"],
            by_segment=r["objective_terms"]["by_segment"],
            wall_s=r["wall_s"],
            wall_parts=r["wall_parts"],
        )
        for r in rows
    ]
    return sorted(table, key=lambda r: float(r["objective"]))


def run_gust_stage(
    source: str | Path,
    *,
    out: str | Path,
    offsets: tuple[float, ...] = BATTERY_OFFSETS_MS,
    scales: tuple[float, ...] = BATTERY_SCALES_MS,
    rates: tuple[float, ...] = BATTERY_RATES_HZ,
    seeds: int = N_SEEDS,
    floor_pairs: int = N_FLOOR_PAIRS,
    pool: str = "floor",
    workers: int = 1,
    max_segments: int | None = None,
    n_mc: int = SHAPE_N_MC,
    check_seeds: int = CHECK_SEEDS,
    resume: bool = True,
    deadline_min: float | None = None,
    progress: int = 0,
) -> dict[str, Any]:
    """Battery mode (module docstring, BATTERY MODE) on the source payload
    ``source``. Returns ``complete`` (``False``: ``deadline_min`` stopped a
    phase, named ``phase``; rerun the same command to resume) and, once any
    grid point finished, ``payload``, ``report``, ``payload_path`` and
    ``report_path`` (``<out>/<stem>__gust.json`` and ``<stem>__gust.grid.json``,
    written; a partial grid is marked ``gust_law.complete = false``)."""
    t0 = time.time()
    deadline = None if deadline_min is None else t0 + 60.0 * float(deadline_min)
    source = Path(source)
    raw = source.read_bytes()
    stage = json.loads(raw)
    _source_check(stage)
    if int(floor_pairs) < 1 or int(seeds) < 1:
        raise ValueError("battery mode needs at least one seed and one seed-floor pair")
    pool_used, pool_note = resolve_pool(pool)
    if pool_note and progress:
        print(pool_note, flush=True)
    out = Path(out)
    stem = f"{source.stem}__gust"
    work = out / stem
    config = dict(
        targets="battery",
        source=str(source),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        max_segments=None if max_segments is None else int(max_segments),
        frame_stride=DF.FRAME_STRIDE,
        seeds=int(seeds),
        floor_pairs=int(floor_pairs),
        n_mc=int(n_mc),
        shape_seed=SHAPE_SEED,
        base_seed=BASE_SEED,
        wind_seed=WIND_SEED,
        floor_seed=FLOOR_SEED,
        pool=pool_used,
        pool_requested=str(pool),
        inner="per-mic sc levels + floor.mean_db",
        terms=list(BATTERY_TERMS),
        weights=dict(BATTERY_WEIGHTS),
    )
    _prepare(work, config)
    build_battery_context(work, stage, max_segments=max_segments, pool=pool_used, progress=progress)
    _, meta, _ = _battery_inputs(str(work))
    n_seg, verbose = len(meta["segments"]), bool(progress)
    run = dict(workers=workers, resume=resume, deadline=deadline, verbose=verbose)
    base_units = [
        Unit(f"seg{i}__seed{k}", dict(work=str(work), segment=i, seed=k))
        for i in range(n_seg)
        for k in range(int(seeds))
    ]
    floor_units = [
        Unit(f"seg{i}__pair{p}", dict(work=str(work), segment=i, pair=p))
        for i in range(n_seg)
        for p in range(int(floor_pairs))
    ]
    for phase, units, worker in (
        ("base", base_units, split_base_unit),
        ("floor", floor_units, seed_floor_unit),
    ):
        done, failed = _run_until(units, worker, work / phase, **run)
        if failed:
            raise RuntimeError(f"{failed} {phase} unit(s) failed: see {work}/{phase}/raw/*.err")
        if not done:
            return dict(complete=False, phase=phase, work_dir=str(work))
    pairs = [json.loads(unit_path(work / "floor", u.uid).read_text()) for u in floor_units]
    floors = seed_floors(pairs)
    _atomic_json(work / "seed_floor.json", floors)
    laws = grid_laws(offsets, scales, rates, stage)
    units = []
    for x in laws:
        o, s, r = x["law"]
        units.append(
            Unit(x["uid"], dict(work=str(work), offset=o, scale=s, rate=r, source=x["source"]))
        )
    complete, _ = _run_until(units, battery_unit, work / "grid", summarize=_battery_summary, **run)
    rows = [
        json.loads(unit_path(work / "grid", u.uid).read_text())
        for u in units
        if unit_path(work / "grid", u.uid).exists()
    ]
    rows = [r for r in rows if math.isfinite(float(r["objective"]))]
    if not rows:
        if not complete:
            return dict(complete=False, phase="grid", work_dir=str(work))
        raise RuntimeError(f"no grid point finished: see {work}/grid/raw/*.err")
    complete = complete and len(rows) == len(units)
    table = _gust_table(rows)
    best = min(rows, key=lambda r: float(r["objective"]))
    src_rows = [r for r in rows if r["source_law"]]
    src_row = src_rows[0] if src_rows else None
    pay = gust_payload(stage, best)
    check = wind_renderer_check(pay, n_seeds=int(check_seeds))
    pay_path, report_path = out / f"{stem}.json", out / f"{stem}.grid.json"

    def point(r: dict[str, Any]) -> dict[str, Any]:
        return dict(
            uid=r["uid"],
            speed_offset_ms=r["speed_offset_ms"],
            speed_scale_ms=r["speed_scale_ms"],
            gust_rate_hz=r["gust_rate_hz"],
            level_db=r["level_db"],
            floor_mean_db=r["floor_mean_db"],
            objective=r["objective"],
            rank=1 + [t["uid"] for t in table].index(r["uid"]),
            terms=r["objective_terms"],
            inner=r["inner"],
        )

    walls = {
        k: float(np.median([float(r["wall_parts"][k]) for r in rows]))
        for k in rows[0]["wall_parts"]
    }
    point_wall = np.array([float(r["wall_s"]) for r in rows])
    pay["wind_stage"] = dict(
        stage=STAGE,
        mode="battery",
        module=__name__,
        source=str(source),
        source_sha256=config["source_sha256"],
        source_git=stage.get("git"),
        source_wind=stage["params"]["wind"],
        source_floor_mean_db=float(stage["params"]["floor"]["mean_db"]),
        frozen="every params block but wind and floor.mean_db: the source payload's (lines, the "
        "floor's shape, exponent and static share, per-rotor and per-mic gains, coherence)",
        refitted="params.wind (sc: the gust law by the outer battery search, the per-mic levels "
        "by the inner Whittle fit) and params.floor.mean_db (inner Whittle fit)",
        segments=meta["segments"],
        targets=dict(
            mode="battery",
            measures="experiments.noise_model.similarity.spectro_temporal and "
            "level_distribution, FULL band, real as the reference, no thrust regression",
            terms=list(BATTERY_TERMS),
            weights=dict(BATTERY_WEIGHTS),
            formula=BATTERY_FORMULA,
            seed_floor=dict(
                floors,
                pairs_per_segment=int(floor_pairs),
                seed_rule="pair p of segment i: the source payload rendered at floor_seed + "
                "100 i + 2 p (reference) and + 1, on segment i's motors_command label",
                floor_seed=FLOOR_SEED,
            ),
        ),
        inner=dict(
            pool=pool_used,
            pool_requested=str(pool),
            pool_note=pool_note,
            pool_meta=meta["pool"],
            band_hz=meta["band_hz"],
            n_fft=meta["n_fft"],
            fit=best["inner"],
        ),
        gust_law=dict(
            grid=dict(
                speed_offset_ms=list(offsets),
                speed_scale_ms=list(scales),
                gust_rate_hz=list(rates),
            ),
            source_law=None if src_row is None else list(source_law(stage)),
            seeds=int(seeds),
            base_seed=BASE_SEED,
            wind_seed=WIND_SEED,
            seed_rule="segment i, seed k: base render seed base_seed + 100 i + k (split into lines "
            "and floor), sim wind seed wind_seed + 100 i + k, the same at every grid point "
            "(common random numbers)",
            chosen=point(best),
            source_point=None if src_row is None else point(src_row),
            n_points=len(units),
            n_finished=len(rows),
            complete=bool(complete),
            table=[dict(uid=t["uid"], objective=t["objective"]) for t in table],
        ),
        fitted=best["summaries"],
        renderer_check=check,
        grid_report=str(report_path),
        work_dir=str(work),
        point_wall_s=dict(
            median=float(np.median(point_wall)), max=float(point_wall.max()), parts_median=walls
        ),
        wall_s=time.time() - t0,
        git=FV._git_sha(),
    )
    report = dict(
        stage=STAGE,
        mode="battery",
        source=str(source),
        payload=str(pay_path),
        config=config,
        formula=BATTERY_FORMULA,
        seed_floor=dict(floors, pairs=pairs),
        complete=bool(complete),
        table=table,
        points=[{k: v for k, v in r.items() if k != "shape"} for r in rows],
    )
    _atomic_json(pay_path, pay)
    _atomic_json(report_path, report)
    return dict(
        complete=bool(complete),
        phase="done" if complete else "grid",
        payload=pay,
        report=report,
        payload_path=pay_path,
        report_path=report_path,
        work_dir=str(work),
    )
