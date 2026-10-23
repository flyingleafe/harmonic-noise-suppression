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
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
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
from . import supports as SU

__all__ = [
    "LevelData",
    "Segment",
    "base_unit",
    "between_line_spectrum",
    "distance",
    "fit_levels",
    "frozen_expectation",
    "grid_point",
    "grid_unit",
    "load_segments",
    "mean_periodogram",
    "run_wind_stage",
    "sc_block",
    "sc_shape",
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
    """``(M,)`` 10 log10 of the payload's wind PSD averaged over 20-100 Hz
    (the static ``wind_db`` itself: its shape is one there)."""
    par = V4.V4Params.from_payload(payload)
    if par.wind is None:
        raise ValueError("the source payload carries no flight wind")
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


def fit_levels(data: LevelData, transform: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Per mic the MAP level ``L_m`` of ``E = E0 + 10^{L_m/10} T_m`` under the
    composite Whittle risk on the pool (``composite_risk``) and ``L_m ~
    N(centre_m, LEVEL_PRIOR_SD_DB)``: a 1 dB scan over ``centre +-
    LEVEL_SPAN_DB``, then bounded Brent on the best point's bracket."""
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

        scan = c + np.arange(-LEVEL_SPAN_DB, LEVEL_SPAN_DB + 0.5, 1.0)
        vals = np.array([nll(x) for x in scan])
        j = int(np.argmin(vals))
        lo, hi = scan[max(j - 1, 0)], scan[min(j + 1, scan.size - 1)]
        res = minimize_scalar(nll, bounds=(float(lo), float(hi)), method="bounded")
        level = float(res.x)
        levels.append(level)
        rows.append(
            dict(
                mic=m,
                level_db=level,
                centre_db=c,
                objective=float(res.fun),
                gain_vs_centre=float(nll(c) - res.fun),
                at_scan_edge=bool(j in (0, scan.size - 1)),
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
