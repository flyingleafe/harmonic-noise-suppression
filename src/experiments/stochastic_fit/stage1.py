"""S1: the single-rotor, single-microphone, windless fit (DREGON's motor bench).

The simplest case the project has, and the one the whole staged plan starts
from: one rotor at a fixed setpoint, nothing moving, no telemetry, no rotor
mixture, no wind. If the generative model cannot be made indistinguishable
here, nothing downstream is worth fitting.

Geometry and discipline
-----------------------
* Source is the RAW 44.1 kHz bench WAV (``native.bench_clip``), decimated to
  16 kHz by ``native.decimate`` — never the published 16 kHz datasets, whose
  88-90 dB brick wall at 7.9 kHz would be fitted as if it were the rig.
* One channel only. Channel 7 carries the largest band-integrated comb margin
  over orders 2-30 (18.54 dB, against 16.59-18.11 dB for the rest), measured
  with ``bench.read_orders`` over six recordings.
* ``Motor1``-``Motor3`` (15 cells) are the FIT set; ``Motor4`` (5 cells) is
  HELD OUT and is scored, never fitted — the same rule the flight stages apply
  to FLY124 and ``dregon_room1``.

What is measured, and what is assumed
-------------------------------------
The bench is stationary, so every quantity the flight fit confounds is
separately visible here: the per-order line level (band-integrated excess over
a local floor), the per-order line WIDTH (a shape fit, no rate wander to
inflate it), the floor's own shape, and — across five setpoints per motor —
the speed law of both. Nothing about shaft jitter or amplitude modulation is
assumed: the measured widths go straight into ``gamma0``/``gamma_slope``.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from data_processing import stochastic_rotor_noise as srn
from experiments.stochastic_fit import bench, native
from experiments.stochastic_fit.data import Clip

#: The bench cells. Setpoints are NOT rev/s — the rate is measured per cell.
MOTORS = (1, 2, 3, 4)
SPEEDS = (50, 60, 70, 80, 90)
FIT_MOTORS = (1, 2, 3)
HELD_OUT_MOTORS = (4,)

#: Channel with the largest comb margin; see the module docstring.
CHANNEL = 7
SR = 16000
#: 16 s of audio at 16 kHz with a 65536-point window: 0.244 Hz bins and ~7
#: averages. The narrowest bench line measured at 44.1 kHz is 0.28 Hz, so this
#: resolves it; a coarser grid would report the window instead of the line.
SECONDS = 16.0
N_ANALYSIS = 1 << 16
K_MAX = 200
#: Orders are read to 200: the slowest cell (49 rev/s) needs order 160 to
#: reach 7.9 kHz, and holding a flat pad past order 100 rendered +7.8 dB of
#: excess in the top band. Above Nyquist ``read_orders`` stops on its own.
#: Floor-shape knots, log-spaced over the modelled band.
N_FLOOR_KNOTS = 14
FLOOR_LO_HZ, FLOOR_HI_HZ = 40.0, 7900.0


@dataclass
class CellMeasurement:
    """Everything one bench recording says, on one channel."""

    motor: int
    setpoint: int
    rate_rps: float
    line_db: np.ndarray  # (K,) band-integrated excess over the local floor
    floor_db: np.ndarray  # (K,) local floor at each order
    width_hz: np.ndarray  # (K,) fitted line half width
    margin_db: np.ndarray  # (K,) peak margin, the detection statistic
    ctrl_hz: np.ndarray  # (J,) floor-shape knots
    ctrl_db: np.ndarray  # (J,) floor level at the knots
    df: float
    n_seconds: float

    @property
    def cell(self) -> tuple[int, int]:
        return (self.motor, self.setpoint)


def _welch(x: np.ndarray, n_fft: int) -> np.ndarray:
    """The bench instrument's own periodogram, reused verbatim."""
    return bench._welch(x, n_fft)


def floor_knots(psd: np.ndarray, df: float) -> tuple[np.ndarray, np.ndarray]:
    """Floor level at log-spaced knots: the median PSD in a band around each.

    A median over a band whose width exceeds the line spacing is insensitive to
    the comb (lines occupy a few bins in tens), so this reads the broadband
    floor without having to mask the lines.
    """
    ctrl = np.geomspace(FLOOR_LO_HZ, FLOOR_HI_HZ, N_FLOOR_KNOTS)
    edges = np.sqrt(ctrl[:-1] * ctrl[1:])
    lo = np.concatenate([[ctrl[0] / np.sqrt(ctrl[1] / ctrl[0])], edges])
    hi = np.concatenate([edges, [ctrl[-1] * np.sqrt(ctrl[-1] / ctrl[-2])]])
    freqs = np.arange(psd.size) * df
    out = np.full(ctrl.size, np.nan)
    for j, (a, b) in enumerate(zip(lo, hi, strict=True)):
        m = (freqs >= a) & (freqs < b)
        if m.sum() >= 8:
            out[j] = 10.0 * np.log10(max(float(np.median(psd[m])), 1e-300))
    ok = np.isfinite(out)
    if not ok.all():  # extrapolate flat at the edges rather than emit NaN
        out = np.interp(np.log(ctrl), np.log(ctrl[ok]), out[ok])
    return ctrl, out


def measure_cell(motor: int, setpoint: int, *, channel: int = CHANNEL) -> CellMeasurement:
    """Measure one bench recording on one channel, natively sourced."""
    clip = native.decimate(native.bench_clip(motor, setpoint, duration_s=SECONDS, start_s=3.0), SR)
    x = clip.audio[channel].astype(np.float64)
    x = x - x.mean()
    psd = _welch(x, N_ANALYSIS)
    df = SR / N_ANALYSIS
    rate = bench.estimate_rate(psd, df)
    line, floor, width, margin, _ = bench.read_orders(psd[None], df, rate, K_MAX)
    ctrl_hz, ctrl_db = floor_knots(psd, df)
    return CellMeasurement(
        motor=motor,
        setpoint=setpoint,
        rate_rps=float(rate),
        line_db=line[0],
        floor_db=floor[0],
        width_hz=width,
        margin_db=margin[0],
        ctrl_hz=ctrl_hz,
        ctrl_db=ctrl_db,
        df=float(df),
        n_seconds=float(clip.duration_s),
    )


def measure_all(*, channel: int = CHANNEL) -> list[CellMeasurement]:
    return [measure_cell(m, s, channel=channel) for m in MOTORS for s in SPEEDS]


# =============================================================================
# The population fit
# =============================================================================

#: An order counts as measured only where its peak clears the local floor.
#: 6 dB is the bench instrument's own profile bar (``PROFILE_MARGIN_MIN_DB``).
MARGIN_MIN_DB = 6.0
#: Reference order for the profile's centre and for the level speed law.
K_REF = 2
#: Reference rate of the speed laws, rev/s.
RATE_REF = 60.0


@dataclass
class Stage1Fit:
    """The generative parameters S1 identifies, all from the fit motors."""

    profile_db: np.ndarray  # (K,) line level relative to order K_REF
    motor_dev_db: dict[int, list[float]]  # per-motor deviation from profile
    gamma0_hz: float
    gamma_slope_hz: float
    amp_exp: float  # line level ~ rate**amp_exp
    floor_exp: float  # floor level ~ rate**floor_exp
    floor_shape_db: np.ndarray  # (J,) floor shape at the knots, centred
    floor_ctrl_hz: np.ndarray  # (J,)
    comb_offset_db: float  # line level at K_REF minus floor at K_REF, at RATE_REF
    k_support: int  # last order with a measured level
    rates: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def detection_limit_db(m: CellMeasurement) -> np.ndarray:
    """Per order, the line level this cell could just barely have detected.

    Among DETECTED orders the band-integrated level and the peak margin measure
    the same line, so ``line_db - floor_db`` regresses on ``margin_db`` with
    unit-correct scale. Evaluating that regression at ``MARGIN_MIN_DB`` gives
    the level at which this cell's detection would have failed — the censoring
    limit. Returned in the same units as ``line_db``.
    """
    line = np.asarray(m.line_db, dtype=np.float64)
    floor = np.asarray(m.floor_db, dtype=np.float64)
    marg = np.asarray(m.margin_db, dtype=np.float64)
    det = np.isfinite(line) & np.isfinite(marg) & (marg >= MARGIN_MIN_DB)
    if det.sum() >= 4:
        a = np.stack([marg[det], np.ones(int(det.sum()))]).T
        coef, *_ = np.linalg.lstsq(a, (line - floor)[det], rcond=None)
        rel_limit = float(coef[0] * MARGIN_MIN_DB + coef[1])
    else:  # no regression possible: the band integral of a just-detectable line
        rel_limit = MARGIN_MIN_DB
    return floor + rel_limit


def _masked(m: CellMeasurement, *, censor: bool = True) -> np.ndarray:
    """Line levels; undetected orders replaced by this cell's detection limit.

    Dropping undetected orders (the obvious choice) biases the pooled profile
    UPWARDS at high order, because only the cells loud enough to detect that
    order contribute — the first S1 gate run showed exactly that, +6.4 dB of
    synthetic excess at 7-7.9 kHz and +6.4 dB of comb excess over orders
    30-100. An undetected line is an UPPER BOUND, not missing data, so it
    enters at its own detection limit and the pooled median becomes a
    conservative estimate instead of a selected one.
    """
    out = np.array(m.line_db, dtype=np.float64, copy=True)
    undetected = ~(np.asarray(m.margin_db) >= MARGIN_MIN_DB)
    if censor:
        limit = detection_limit_db(m)
        in_band = np.isfinite(np.asarray(m.floor_db))
        out[undetected & in_band] = limit[undetected & in_band]
        out[~in_band] = np.nan
    else:
        out[undetected] = np.nan
    return out


def fit_population(measurements: list[CellMeasurement]) -> Stage1Fit:
    """Fit the S1 population on the FIT motors only."""
    fit_set = [m for m in measurements if m.motor in FIT_MOTORS]
    if not fit_set:
        raise ValueError("no fit-set cells")
    k = np.arange(1, K_MAX + 1, dtype=np.float64)

    # --- level: shared profile shape, per-motor deviation, speed exponent ----
    # Every cell's profile is centred on its own K_REF, so the speed law and the
    # timbre do not contaminate each other.
    #
    # Two populations per order: DETECTED levels, and detection LIMITS for the
    # cells where that order is invisible. The profile is the detected median
    # where a majority of cells detect the order; beyond that the measured
    # roll-off is extrapolated in log k and CLIPPED to stay below the median
    # detection limit — an order nobody detected cannot sit at a level that
    # would have been detected. Holding the last measured value instead (or
    # substituting the limit) renders 100 detectable orders where the real
    # bench clip has 61, which is what the first two S1 gate runs reported.
    det_rows, lim_rows = [], []
    for m in fit_set:
        ref = np.asarray(m.line_db, dtype=np.float64)[K_REF - 1]
        if not np.isfinite(ref):
            continue
        det_rows.append(_masked(m, censor=False) - ref)
        lim_rows.append(detection_limit_db(m) - ref)
    det_arr, lim_arr = np.array(det_rows), np.array(lim_rows)
    n_cells = det_arr.shape[0]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        det_med = np.nanmedian(det_arr, axis=0)
        lim_med = np.nanmedian(lim_arr, axis=0)
        n_det = np.sum(np.isfinite(det_arr), axis=0)
    measured = n_det > n_cells / 2
    support = int(np.max(np.flatnonzero(measured)) + 1) if measured.any() else 0

    profile = np.array(det_med, dtype=np.float64, copy=True)
    if support >= 8:
        # The roll-off is fitted over the WHOLE measured range from order 4 up,
        # not over its upper half: at high order only the loud cells detect the
        # line, so a fit restricted to the top of the support reads the
        # selection instead of the decay and returned +3.3 dB per log-k —
        # a comb RISING with order, which then jumped the extrapolated profile
        # from -43 dB at order 160 to -21 dB at order 200.
        seg = np.flatnonzero(measured & (k >= 4.0))
        coef = np.polyfit(np.log(k[seg]), det_med[seg], 1)
        tail = np.arange(support, K_MAX)
        extrap = np.polyval(coef, np.log(k[tail]))
        cap = np.where(np.isfinite(lim_med[tail]), lim_med[tail], extrap)
        profile[tail] = np.minimum(extrap, cap)
        profile[:support] = np.where(
            measured[:support],
            det_med[:support],
            np.minimum(
                np.polyval(coef, np.log(k[:support])),
                np.where(np.isfinite(lim_med[:support]), lim_med[:support], 0.0),
            ),
        )
        # Past the measured support the comb cannot rise: no mechanism makes a
        # blade-passage harmonic louder at higher order, and the detection
        # limits are only upper bounds.
        if support < K_MAX:
            profile[support - 1 :] = np.minimum.accumulate(profile[support - 1 :])
        rolloff_db_per_log_k = float(coef[0])
    else:
        rolloff_db_per_log_k = float("nan")

    motor_dev: dict[int, list[float]] = {}
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for mot in FIT_MOTORS:
            rows = [d for d, m in zip(det_arr, fit_set, strict=True) if m.motor == mot]
            dev = np.nanmedian(np.array(rows), axis=0) - profile
            dev[~measured] = 0.0  # no per-motor timbre where nobody measured it
            motor_dev[mot] = np.nan_to_num(dev, nan=0.0).tolist()

    # --- speed laws: level and floor at K_REF against the measured rate ------
    # ONE exponent shared by the motors, but a FREE INTERCEPT PER MOTOR: the
    # four rotors differ by up to 7 dB in absolute level at the same setpoint
    # (Motor1 35.7 dB vs Motor4 29.0 dB at ~49 rev/s), and pooling them into a
    # single intercept charges that offset to the speed exponent.
    motors_of = [m.motor for m in fit_set]
    uniq = sorted(set(motors_of))

    def _exp(values: list[float], rates: list[float]) -> tuple[float, dict[int, float]]:
        v = np.asarray(values, dtype=np.float64)
        r = np.asarray(rates, dtype=np.float64)
        ok = np.isfinite(v) & np.isfinite(r) & (r > 0)
        if ok.sum() < len(uniq) + 2:
            return float("nan"), {}
        cols = [10.0 * np.log10(r[ok])]
        for mot in uniq:
            cols.append(np.array([1.0 if m == mot else 0.0 for m, o in zip(motors_of, ok) if o]))
        design = np.stack(cols).T
        coef, *_ = np.linalg.lstsq(design, v[ok], rcond=None)
        return float(coef[0]), {mot: float(c) for mot, c in zip(uniq, coef[1:], strict=True)}

    lvl = [float(_masked(m)[K_REF - 1]) for m in fit_set]
    flr = [float(np.asarray(m.floor_db)[K_REF - 1]) for m in fit_set]
    rates = [m.rate_rps for m in fit_set]
    amp_exp, amp_c = _exp(lvl, rates)
    floor_exp, floor_c = _exp(flr, rates)

    # --- width law: gamma = gamma0 + c k, over lines that stand clear --------
    wk, wv = [], []
    for m in fit_set:
        w = np.asarray(m.width_hz, dtype=np.float64)
        ok = np.isfinite(w) & (np.asarray(m.margin_db) >= bench.WIDTH_MARGIN_MIN_DB)
        wk.append(k[ok])
        wv.append(w[ok])
    wk_a, wv_a = np.concatenate(wk), np.concatenate(wv)
    if wk_a.size >= 4:
        a = np.stack([wk_a, np.ones_like(wk_a)])
        coef, *_ = np.linalg.lstsq(a.T, wv_a, rcond=None)
        gamma_slope, gamma0 = float(coef[0]), float(coef[1])
    else:
        gamma_slope, gamma0 = float("nan"), float("nan")

    # --- floor shape: centred median over the fit cells ----------------------
    shape = np.median(np.array([m.ctrl_db for m in fit_set]), axis=0)
    ctrl_hz = fit_set[0].ctrl_hz
    shape_centred = shape - np.median(shape)

    # --- comb-to-floor offset at the reference rate --------------------------
    offsets = [lv - fl for lv, fl in zip(lvl, flr, strict=True)]
    at_ref = [
        o - (amp_exp - floor_exp) * 10.0 * np.log10(r / RATE_REF)
        for o, r in zip(offsets, rates, strict=True)
        if np.isfinite(o)
    ]
    comb_offset = float(np.median(at_ref)) if at_ref else float("nan")

    return Stage1Fit(
        profile_db=np.nan_to_num(profile, nan=-60.0),
        motor_dev_db=motor_dev,
        gamma0_hz=gamma0,
        gamma_slope_hz=gamma_slope,
        amp_exp=amp_exp,
        floor_exp=floor_exp,
        floor_shape_db=shape_centred,
        floor_ctrl_hz=ctrl_hz,
        comb_offset_db=comb_offset,
        k_support=support,
        rates={f"Motor{m.motor}_{m.setpoint}": m.rate_rps for m in measurements},
        diagnostics={
            "n_fit_cells": len(fit_set),
            "amp_intercept_db_per_motor": amp_c,
            "floor_intercept_db_per_motor": floor_c,
            "width_lines_used": int(wk_a.size),
            "profile_support_orders": support,
            "rolloff_db_per_log_k": rolloff_db_per_log_k,
            "channel": CHANNEL,
            "margin_min_db": MARGIN_MIN_DB,
        },
    )


# =============================================================================
# Rendering one S1 clip
# =============================================================================


def params_for(
    fit: Stage1Fit, rate_rps: float, *, motor: int | None = None, sample_rate: int = SR
) -> srn.StochasticParams:
    """A single-rotor renderer parameter set at ``rate_rps``.

    Only shapes and ratios are set: ``synthesize`` normalises the output RMS and
    every acceptance statistic is referenced to the clip's own 200-400 Hz band,
    so an absolute level would be unobservable anyway. The width comes straight
    from the measured law, with no shaft jitter and no amplitude-modulation
    broadening — on a bench there is nothing to jitter.
    """
    k_max = max(2, int(np.floor((sample_rate / 2) / max(rate_rps, 1.0))))
    prof = np.asarray(fit.profile_db, dtype=np.float64)[:k_max].copy()
    if motor is not None and motor in fit.motor_dev_db:
        dev = np.asarray(fit.motor_dev_db[motor], dtype=np.float64)[:k_max]
        prof = prof + dev[: prof.size]
    if prof.size < k_max:  # below the measured support: hold the last measured
        prof = np.concatenate([prof, np.full(k_max - prof.size, prof[-1])])
    return srn.StochasticParams(
        sample_rate=int(sample_rate),
        n_rotors=1,
        n_harmonics=k_max,
        profile_db=prof[None, :],
        gamma0=np.array([fit.gamma0_hz]),
        gamma_slope=np.array([fit.gamma_slope_hz]),
        floor_ctrl_hz=np.asarray(fit.floor_ctrl_hz, dtype=np.float64),
        floor_ctrl_db=np.asarray(fit.floor_shape_db, dtype=np.float64),
        floor_tilt_db_oct=0.0,
        harm_mean_db=float(fit.comb_offset_db),
        floor_mean_db=0.0,
        harm_gp_std_db=0.0,
        harm_gp_tau_s=1.0,
        harm_coherence=0.0,
        floor_gp_std_db=0.0,
        floor_gp_tau_s=1.0,
        floor_tilt_gp_std=0.0,
        floor_tilt_gp_tau_s=1.0,
        line_bin_integrate=True,
        floor_static_rel=0.0,
        amp_rps_exponent=float(fit.amp_exp),
        amp_rps_exponent_floor=float(fit.floor_exp),
        amp_rps_ref=RATE_REF,
        shaft_jitter_rps=float(fit.gamma_slope_hz) / 1.177,
        shaft_jitter_tau_s=2.0,
        phase_diffusion_hz_per_order=0.0,
        shaft_offset_rps=0.0,
        umod_std_db=0.0,
        umod_tau_s=1.0,
        umod_corner_hz=200.0,
        mic_gain_all_db=0.0,
        mic_floor_std_db=0.0,
    )


def render(
    fit: Stage1Fit,
    rate_rps: float,
    *,
    seconds: float = SECONDS,
    seed: int = 0,
    motor: int | None = None,
) -> np.ndarray:
    """One synthetic S1 clip: ``(T,)`` mono at :data:`SR`.

    Rendered at the RAW 44.1 kHz rate and decimated by ``native.decimate``, the
    same path every real clip takes. Generating directly at 16 kHz instead left
    the synthetic clip without the resampler's transition band while the real
    one had it, which alone put +7.8 dB of spurious deviation into the
    7-7.9 kHz acceptance band.

    ``line_mode="fm"``, with the measured width carried by SHAFT JITTER rather
    than by ``gamma0``/``gamma_slope``, which that mode ignores. The choice is
    forced by two measurements: with zero jitter the FM mode renders 0.00 Hz
    lines at every order (real: 0.20-7.92 Hz, peak margin 27-38 dB against a
    real 2-20 dB), while ``line_mode="stochastic"`` cannot go below one STFT
    bin and rendered 2.6-6.9 Hz lines where the bench has 0.2-1 Hz. The
    measured width law is LINEAR in k (0.0385 Hz/order), which is exactly what
    a quasi-static shaft-speed wander of ``slope / 1.177`` = 0.033 rev/s
    produces — ESC control ripple on a clamped motor, not a modelling fudge.
    """
    params = params_for(fit, rate_rps, motor=motor, sample_rate=native.NATIVE_SR)
    n = int(round(seconds * native.NATIVE_SR))
    rps = np.full((1, n), float(rate_rps))
    audio, _ = srn.synthesize(
        params, rps, rng=np.random.default_rng(seed), n_mics=1, line_mode="fm"
    )
    clip = Clip(
        "synthetic",
        "synthetic",
        np.asarray(audio, dtype=np.float32),
        rps,
        native.NATIVE_SR,
        None,
        {"synthetic": True},
    )
    return np.asarray(native.decimate(clip, SR).audio[0], dtype=np.float64)


def calibrate_profile(
    fit: Stage1Fit,
    measurements: list[CellMeasurement],
    *,
    rounds: int = 3,
    seed: int = 7,
) -> Stage1Fit:
    """Map the measured profile into the renderer's own level convention.

    ``profile_db`` and ``harm_mean_db`` are the renderer's parameters, not the
    estimator's readings: the renderer normalises the comb against the floor's
    MEAN spectrum over the whole band, so feeding it a profile centred on
    order 2 renders a comb 5 dB too strong at low order and 10 dB too strong
    at high order (measured on the fit cells: +5.3 dB over orders 2-15, +9.9 dB
    over 30-100, 100 detectable orders against a real 42-54).

    The correction is the one affine function of ``log k`` that removes that
    discrepancy, fitted ONLY on the fit motors and applied to the profile. The
    held-out motor never enters, so the gate stays a test.
    """
    from experiments.stochastic_fit import accept_stats as stats

    cells = [m for m in measurements if m.motor in FIT_MOTORS]
    k = np.arange(1, K_MAX + 1, dtype=np.float64)
    out = fit
    history: list[dict[str, float]] = []
    for r in range(rounds):
        deltas = []
        for m in cells:
            real = native.decimate(
                native.bench_clip(m.motor, m.setpoint, duration_s=SECONDS, start_s=3.0), SR
            )
            xr = real.audio[CHANNEL].astype(np.float64)
            xs = render(out, m.rate_rps, seconds=SECONDS, seed=seed + r, motor=m.motor)
            er, mr = stats.order_profile(
                xr, m.rate_rps, gamma0=out.gamma0_hz, gamma_slope=out.gamma_slope_hz
            )
            es, _ = stats.order_profile(
                xs, m.rate_rps, gamma0=out.gamma0_hz, gamma_slope=out.gamma_slope_hz
            )
            d = np.full(K_MAX, np.nan)
            n = min(er.size, es.size, K_MAX)
            usable = np.isfinite(er[:n]) & np.isfinite(es[:n]) & (mr[:n] >= stats.MARGIN_MIN_DB)
            d[:n][usable] = (es[:n] - er[:n])[usable]
            deltas.append(d)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(np.array(deltas), axis=0)
        ok = np.isfinite(med) & (k >= 2)
        if ok.sum() < 6:
            break
        coef = np.polyfit(np.log(k[ok]), med[ok], 1)
        correction = np.polyval(coef, np.log(k))
        history.append(
            {
                "round": r,
                "median_delta_db": float(np.nanmedian(med[ok])),
                "slope_db_per_log_k": float(coef[0]),
                "intercept_db": float(coef[1]),
                "orders_used": int(ok.sum()),
            }
        )
        prof = np.asarray(out.profile_db, dtype=np.float64) - correction
        prof = prof - prof[K_REF - 1]  # keep the K_REF centring of the profile
        out = Stage1Fit(
            profile_db=prof,
            motor_dev_db=out.motor_dev_db,
            gamma0_hz=out.gamma0_hz,
            gamma_slope_hz=out.gamma_slope_hz,
            amp_exp=out.amp_exp,
            floor_exp=out.floor_exp,
            floor_shape_db=out.floor_shape_db,
            floor_ctrl_hz=out.floor_ctrl_hz,
            comb_offset_db=out.comb_offset_db - float(np.polyval(coef, np.log(K_REF))),
            k_support=out.k_support,
            rates=out.rates,
            diagnostics={**out.diagnostics, "calibration": history},
        )
    return out


def save(fit: Stage1Fit, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    d = asdict(fit)
    for key, val in list(d.items()):
        if isinstance(val, np.ndarray):
            d[key] = val.tolist()
    p.write_text(json.dumps(d, indent=1))
    return p


def load(path: str | Path) -> Stage1Fit:
    d = json.loads(Path(path).read_text())
    for key in ("profile_db", "floor_shape_db", "floor_ctrl_hz"):
        d[key] = np.asarray(d[key], dtype=np.float64)
    d["motor_dev_db"] = {int(k): v for k, v in d["motor_dev_db"].items()}
    return Stage1Fit(**d)
