"""Noise model v4 — the bench fit (Fit procedure A, ``docs/explainers/noise-model-v4.qmd``).

ONE maximum-a-posteriori fit per DREGON motor, JOINT over its five-setting
throttle ladder (``motor_Motor{i}_{50..90}``), on one whole-span periodic-Hann
periodogram per setting and microphone (16 kHz, 0.03-0.1 Hz resolution).

The expected periodogram of setting ``j``, microphone ``m``::

    M_jm(f) = sum_k h_jmk [ P^a_jk L^a_jk(f) + P^m_jk L^m_jk(f) ]      (the comb)
            + g_m F(f) r_j^{b(f)}                                     (the floor)
            + 10^{w_jm/10} s_W(f)                                     (the wind)
            + N_jm(f)                                                 (the ambient, measured)

* ``P^a_jk = 10^{(p^a_k + 10 alpha^a log10 r_j)/10}``, ``r_j = f_j / 80 rev/s``;
  ``P^m`` the same with ``p^m_n``, ``alpha^m`` at the motor orders ``k = 3 p n``;
* ``L_jk`` is the window-convolved line of lag law :func:`lag.r_tau_v4`: the
  shaft OU ``(sigma_nu(f_j), lam(f_j))`` times the per-order pedestal
  ``(sigma2_k, gamma_k)`` for the aerodynamic family, the shaft alone for the
  motor family. The shaft follows ONE speed law per motor (:meth:`LadderModel.shaft`):
  ``sigma_nu = 2 pi f_ref eps r^a`` (``eps`` the relative speed jitter at
  ``f_ref``, half-normal in linear space) and ``lam = lam_ref r^b``;
* ``h_jmk = 10^{(E_jmk + delta_jmk)/10}``: the pair EQ ``E`` (a constant,
  measured on the OTHER settings of the ladder, :func:`pair_eq`) plus the
  per-run, per-line scatter ``delta ~ N(0, 2.4 dB)`` (6 dB below ``k = 8`` on
  the windy microphones);
* ``F(f)`` the 14-point floor spline of nv3r4 at the reference speed,
  ``b(f) = b0 + b1 log2(f / 2 kHz)``, ``g_m`` mean-pinned microphone gains;
* ``s_W`` the fixed v3 wind shape, one level per setting and microphone;
* ``N_jm`` the Welch spectrum of the recording's quiet (rotors stopped) tail.

Shared across the ladder: ``p^a, p^m, alpha^a, alpha^m, b0, b1, F, g, sigma2,
gamma`` and the shaft law ``(eps, a, lam_ref, b)``. Per setting: the carrier
(frozen, the measured core speed), ``delta``, the wind and the ambient.

Deviations from the explainer, each forced by the data: (i) the ambient is
read on the QUIET run of the recording (the rotors stopped), not on "the span
start minus 0.5 s" — the pre-span segment holds the armed motor idling at
-64 dB, 8 dB over the true ambient; (ii) the wind level is per setting, not per
microphone only — the downwash grows with the throttle.

The lag law of every line is summed in the LAG domain (exact for a constant
carrier, :func:`spectrum.bench_model`) over order groups whose lag support is
fixed before the fit from the seeds (:func:`spectrum.order_groups`), with a
margin; :func:`lag_support_check` re-reads the support at the fitted values.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from torch import Tensor

from data_processing.noise_model.lag import order_lag_support_s, r_tau_v4
from data_processing.noise_model.resample import decimate_audio
from data_processing.noise_model.v3 import wind_shape
from experiments.stochastic_fit import clips as C
from experiments.stochastic_fit.model import AMP_RPS_REF
from experiments.stochastic_fit.phase_kernel import hann_window
from experiments.stochastic_fit.revised_phase import MODEL_FLOOR

from . import spectrum as SP
from .lbfgs import LBFGS

__all__ = [
    "FIT_SCHEMA",
    "Ladder",
    "LadderModel",
    "Setting",
    "build_ladder",
    "fit_ladder",
    "lag_support_check",
    "payload",
    "prepare_setting",
]

FIT_SCHEMA = "noise-v4-fit/2"
DATASET = "DREGON-frames"
RPS_KEY = "motors_command"
SR = 16000
F_REF = float(AMP_RPS_REF)
THROTTLES = (50, 60, 70, 80, 90)
MOTORS = ("Motor1", "Motor2", "Motor3", "Motor4")
#: DREGON: two blades, seven pole pairs, so the motor family sits at 21 n.
POLE_PAIRS = 7
#: The downwash microphones per motor (``docs/experiments/static-rig-profiles.md``).
WIND_MICS = {"Motor1": (5, 6), "Motor2": (0,), "Motor3": (1, 2), "Motor4": (4,)}

SINGLES_PATH = Path("results/static_rig/single_rotor/dregon_singles.json")
OU_DIR = Path("results/static_rig/single_rotor/ou")
AM_PATH = Path("results/static_rig/single_rotor/amplitude_jitter.json")
CACHE_DIR = Path("results/noise_v4/bench/settings")

#: The likelihood band: 30 Hz to 0.45 sr (7.2 kHz), clear of the decimator.
BAND_HZ = (SP.BAND_F_MIN, min(SP.BAND_F_MAX, 0.45 * SR))
LAG_NATS = SP.LAG_SUPPORT_NATS
#: The lag grid is cut at the seeds with this margin: sigma_nu / 1.5, lam x 1.5.
LAG_SIGMA_MARGIN = 1.5
LAG_LAM_MARGIN = 1.5

# ── priors (explainer, tbl-fit-priors, bench column) ─────────────────────────
PROFILE_SD_DB = 10.0
MOTOR_BOOST_DB = 6.0
ALPHA_AERO = (5.5, 1.0)
ALPHA_MOTOR = (0.0, 2.0)
B0 = (3.0, 1.0)
B1 = (0.5, 0.25)
#: The shaft law. ``eps``: relative rms speed jitter at f_ref, HalfNormal in
#: linear space (a percent-level speed regulation error). ``a``, ``b``: speed
#: exponents of sigma_nu and lam; 1 is the aero-damped motor whose disturbance
#: torque scales with its load (constant relative jitter, damping ~ speed).
EPS_HALF_NORMAL = 0.005
A_SIGMA = (1.0, 0.5)
B_LAM = (1.0, 0.5)
#: Seed of a setting whose quick fit fell into the random-walk corner.
SHAFT_SEED = (0.4, 12.0)  # (rad/s, 1/s)
SIGMA2_CAP = 0.6
SCATTER_SD_DB = 2.4
SCATTER_WINDY_SD_DB = 6.0
SCATTER_WINDY_BELOW_K = 8
FLOOR_MEAN_SD_DB = 10.0
MIC_FLOOR_SD_DB = 6.0
WIND_SD_DB = 6.0
#: Convergence: a fresh L-BFGS restart gaining less than this per cell.
TOL_NATS_PER_CELL = 1e-4


def _log_sigma2_prior(k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(mu, sd)`` of ``log sigma2_k``: the four tiers of tbl-prior-laws."""
    mu = np.where(
        k <= 3,
        math.log(0.003),
        np.where(k <= 8, math.log(0.05), np.where(k <= 20, math.log(0.3), math.log(0.5))),
    )
    sd = np.where(k <= 8, 1.0, np.where(k <= 20, 0.7, 0.5))
    return mu.astype(np.float64), sd.astype(np.float64)


def _log_gamma_prior(k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(mu, sd)`` of ``log gamma_k`` (Hz)."""
    mu = np.where(k <= 8, math.log(0.4), math.log(1.0))
    return mu.astype(np.float64), np.full(k.shape, 0.6)


def _capped_sigma2(u: Tensor, cap: float | None) -> Tensor:
    """``log sigma2 = log cap - softplus(log cap - u)``: a smooth cap in log
    space, equal to ``e^u`` far below the cap and never above it; ``e^u`` when
    ``cap`` is None."""
    if cap is None:
        return torch.exp(u)
    log_cap = math.log(cap)
    return torch.exp(log_cap - torch.nn.functional.softplus(log_cap - u))


# ── data ─────────────────────────────────────────────────────────────────────


@dataclass
class Setting:
    """One stationary recording of the ladder, everything the fit holds fixed."""

    key: str
    motor: str
    throttle: int
    carrier: float  #: rev/s, frozen (core peaks of orders 2-7, +-0.003)
    span_s: tuple[float, float]
    ambient_runs_s: list[tuple[float, float]]
    power: np.ndarray  #: (M, F) periodogram, ``|rfft(w x)|^2 / sum w^2``
    ambient: np.ndarray  #: (M, F) the quiet-run Welch spectrum on the same bins
    n: int  #: samples of the span at 16 kHz (window = FFT length)

    @property
    def freqs(self) -> np.ndarray:
        return np.fft.rfftfreq(self.n, d=1.0 / SR)

    @property
    def windy(self) -> tuple[int, ...]:
        return WIND_MICS[self.motor]

    @property
    def k_max(self) -> int:
        return int(BAND_HZ[1] // self.carrier)


def _band_level_db(x: np.ndarray, block_s: float = 0.25) -> np.ndarray:
    """Mic-median band level (dB) of ``(M, T)`` 16 kHz audio per block."""
    sos = butter(4, list(BAND_HZ), btype="band", fs=SR, output="sos")
    y = sosfiltfilt(sos, np.asarray(x, dtype=np.float64), axis=-1)
    nb = int(block_s * SR)
    n = y.shape[-1] // nb
    e = (y[:, : n * nb].reshape(y.shape[0], n, nb) ** 2).mean(-1)
    return 10.0 * np.log10(np.median(e, axis=0) + 1e-30)


def ambient_runs(
    x: np.ndarray,
    span_s: tuple[float, float],
    *,
    block_s: float = 0.25,
    tol_db: float = 2.0,
    guard_s: float = 1.0,
    trim_s: float = 0.25,
    min_run_s: float = 1.0,
    min_total_s: float = 2.0,
) -> list[tuple[float, float]]:
    """Every run of blocks within ``tol_db`` of the recording's quietest level
    (5th percentile), at least ``guard_s`` away from the rotor-on span and at
    least ``min_run_s`` long after trimming: the rotors stopped, the room alone.
    The armed, idling motor before and after the span sits 3-8 dB higher and is
    excluded by the tolerance."""
    lev = _band_level_db(x, block_s)
    t0 = np.arange(lev.size) * block_s
    quiet = lev <= np.percentile(lev, 5) + tol_db
    quiet &= (t0 + block_s <= span_s[0] - guard_s) | (t0 >= span_s[1] + guard_s)
    edges = np.diff(np.concatenate([[0], quiet.astype(np.int8), [0]]))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    runs = [
        (float(a * block_s + trim_s), float(b * block_s - trim_s)) for a, b in zip(starts, stops)
    ]
    runs = [r for r in runs if r[1] - r[0] >= min_run_s]
    total = sum(b - a for a, b in runs)
    if total < min_total_s:
        raise ValueError(f"{total:.2f} s of quiet runs outside the span, under {min_total_s} s")
    return runs


def _periodogram(x: np.ndarray) -> np.ndarray:
    """``|rfft(w x)|^2 / sum w^2`` with the periodic Hann over the whole segment."""
    w = hann_window(x.shape[-1])
    spec = np.fft.rfft(np.asarray(x, dtype=np.float64) * w, axis=-1)
    return (spec.real**2 + spec.imag**2) / float(np.sum(w**2))


def _welch(runs: list[np.ndarray], seg: int = SR, hop: int = SR // 2) -> np.ndarray:
    """Mean of :func:`_periodogram` over every ``seg``-sample segment of every
    run: the same units as the whole-span periodogram."""
    parts = [
        _periodogram(x[:, i * hop : i * hop + seg])
        for x in runs
        for i in range(1 + (x.shape[-1] - seg) // hop)
    ]
    return np.mean(parts, axis=0)


def prepare_setting(motor: str, throttle: int) -> Setting:
    """Load one recording, decimate it to 16 kHz, cut the strict span and the
    quiet run, and read the periodogram and the ambient."""
    key = f"motor_{motor}_{int(throttle)}"
    singles = json.loads(SINGLES_PATH.read_text())["recordings"][key]
    rec = C.load_recording(DATASET, key, None, RPS_KEY)
    x = decimate_audio(np.asarray(rec.audio, dtype=np.float64), int(rec.sr), SR)
    span = (float(singles["span_s"][0]), float(singles["span_s"][1]))
    seg = x[:, int(round(span[0] * SR)) : int(round(span[1] * SR))]
    amb = ambient_runs(x, span)
    quiet = [x[:, int(round(a * SR)) : int(round(b * SR))] for a, b in amb]
    n = int(seg.shape[-1])
    f_fine = np.fft.rfftfreq(n, d=1.0 / SR)
    welch = _welch(quiet)
    f_welch = np.fft.rfftfreq(SR, d=1.0 / SR)
    ambient = np.stack([np.interp(f_fine, f_welch, row) for row in welch])
    return Setting(
        key=key,
        motor=motor,
        throttle=int(throttle),
        carrier=float(singles["s_bar_rev_s"]),
        span_s=span,
        ambient_runs_s=amb,
        power=_periodogram(seg),
        ambient=ambient,
        n=n,
    )


def save_setting(s: Setting, cache_dir: Path = CACHE_DIR) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{s.key}.npz"
    np.savez(
        path,
        power=s.power.astype(np.float32),
        ambient=s.ambient.astype(np.float32),
        meta=json.dumps(
            dict(
                key=s.key,
                motor=s.motor,
                throttle=s.throttle,
                carrier=s.carrier,
                span_s=s.span_s,
                ambient_runs_s=s.ambient_runs_s,
                n=s.n,
            )
        ),
    )
    return path


def load_setting(motor: str, throttle: int, cache_dir: Path = CACHE_DIR) -> Setting:
    path = cache_dir / f"motor_{motor}_{int(throttle)}.npz"
    if not path.exists():
        s = prepare_setting(motor, throttle)
        save_setting(s, cache_dir)
        return s
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    return Setting(
        key=meta["key"],
        motor=meta["motor"],
        throttle=int(meta["throttle"]),
        carrier=float(meta["carrier"]),
        span_s=tuple(meta["span_s"]),
        ambient_runs_s=[tuple(r) for r in meta["ambient_runs_s"]],
        power=z["power"].astype(np.float64),
        ambient=z["ambient"].astype(np.float64),
        n=int(meta["n"]),
    )


# ── measurements (seeds, prior centres, the pair EQ) ─────────────────────────


def line_table(s: Setting, k_max: int) -> dict[str, np.ndarray]:
    """Per mic and order: line power (dB, the model's ``10^{p/10}`` units),
    the floor power in the same band (dB) and their ratio (``snr_db``).

    ``P = (2/n) sum_band (I - F)`` over ``|f - k f_r| <= max(1 Hz, 3 bins)``;
    ``F`` is the annulus (6-18 Hz away) median over ``ln 2`` (the exponential
    median-to-mean factor)."""
    f = s.freqs
    df = float(f[1])
    half = max(1.0, 3.0 * df)
    M = s.power.shape[0]
    level = np.full((M, k_max), np.nan)
    floor = np.full((M, k_max), np.nan)
    for k in range(1, k_max + 1):
        fc = k * s.carrier
        band = np.abs(f - fc) <= half
        ann = (np.abs(f - fc) >= 6.0) & (np.abs(f - fc) <= 18.0)
        F = np.median(s.power[:, ann], axis=1) / math.log(2.0)
        tot = s.power[:, band].sum(axis=1)
        flo = F * band.sum()
        level[:, k - 1] = (2.0 / s.n) * np.maximum(tot - flo, 1e-3 * flo)
        floor[:, k - 1] = (2.0 / s.n) * flo
    return dict(
        level_db=10.0 * np.log10(level),
        floor_db=10.0 * np.log10(floor),
        snr_db=10.0 * np.log10(level / floor + 1.0),
    )


def pair_eq(
    settings: list[Setting],
    tables: list[dict[str, np.ndarray]],
    target: int,
    *,
    min_snr_db: float = 10.0,
    width_hz: float = 200.0,
) -> np.ndarray:
    """``E (M, K_target)`` dB: each mic's line level relative to the mic mean,
    pooled over the OTHER settings on the frequency axis and smoothed with a
    ``width_hz`` Gaussian, read at the target setting's line frequencies. A
    frequency with no nearby evidence shrinks to zero (one pseudo-observation
    of weight one at zero)."""
    fs, devs = [], []
    for i, (s, t) in enumerate(zip(settings, tables)):
        if i == target:
            continue
        ok = np.all(t["snr_db"] >= min_snr_db, axis=0)
        lev = t["level_db"][:, ok]
        devs.append(lev - lev.mean(axis=0, keepdims=True))
        fs.append(s.carrier * (np.flatnonzero(ok) + 1.0))
    f_obs = np.concatenate(fs)
    d_obs = np.concatenate(devs, axis=1)  # (M, N)
    tgt = settings[target]
    f_tgt = tgt.carrier * np.arange(1, tgt.k_max + 1, dtype=np.float64)
    w = np.exp(-0.5 * ((f_tgt[:, None] - f_obs[None, :]) / width_hz) ** 2)  # (K, N)
    return (d_obs @ w.T) / (w.sum(axis=1)[None, :] + 1.0)


def _quick_shaft(key: str) -> tuple[float, float] | None:
    """``(sigma_nu rad/s, lam 1/s)`` of the width-law quick fit, or ``None``
    where it fell into the random-walk corner (``lam < 3``)."""
    path = OU_DIR / f"{key}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    if float(d["lam"]) < 3.0:
        return None
    return 2.0 * math.pi * float(d["sigma_nu"]), float(d["lam"])


def _pedestal_seed(keys: list[str], k_prof: int) -> tuple[np.ndarray, np.ndarray]:
    """``(sigma2, gamma)`` per order: the median over the ladder of the
    good-mic median of the direct amplitude-jitter reading; prior medians
    where it is missing."""
    aj = json.loads(AM_PATH.read_text())
    s2, gm = [], []
    # orders a recording never read (SNR < 10 dB) are NaN on every mic: their
    # all-NaN medians are NaN by intent and fall back to the prior below
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for key in keys:
            r = aj[key]
            good = list(r["mics_good"])
            s2.append(np.nanmedian(np.array(r["sigma_m2"], dtype=float)[good], axis=0))
            gm.append(np.nanmedian(np.array(r["gamma_m"], dtype=float)[good], axis=0))
        s2m = np.nanmedian(np.stack(s2), axis=0)
        gmm = np.nanmedian(np.stack(gm), axis=0)
    k = np.arange(1, k_prof + 1, dtype=np.float64)
    mu_s, _ = _log_sigma2_prior(k)
    mu_g, _ = _log_gamma_prior(k)
    out_s = np.exp(mu_s)
    out_g = np.exp(mu_g)
    n = min(k_prof, s2m.size)
    ok_s = np.isfinite(s2m[:n]) & (s2m[:n] > 0)
    ok_g = np.isfinite(gmm[:n]) & (gmm[:n] > 0)
    out_s[:n][ok_s] = np.clip(s2m[:n][ok_s], 1e-4, 0.9 * SIGMA2_CAP)
    out_g[:n][ok_g] = np.clip(gmm[:n][ok_g], 0.05, 3.0)
    return out_s, out_g


# ── the ladder ───────────────────────────────────────────────────────────────


@dataclass
class _Frozen:
    """Per-setting tensors the fit never changes."""

    grid: SP.BenchGrid
    power: Tensor  # (M, F)
    ambient: Tensor  # (M, F)
    band: Tensor  # (F,) bool
    wind: Tensor  # (F,)
    eq_db: Tensor  # (M, K)
    log_r: float
    groups: list[tuple[int, np.ndarray]]
    cos: list[Tensor]  # per group (Kg, L)
    motor_rows: list[np.ndarray]  # per group: indices into the group's orders that are motor
    floor_f: Tensor  # the floor basis' PSD-grid frequencies


@dataclass
class Ladder:
    """One motor's ladder: data, measured seeds and prior centres."""

    motor: str
    settings: list[Setting]
    tables: list[dict[str, np.ndarray]]
    eq: list[np.ndarray]
    k_prof: int
    motor_orders: np.ndarray  # (n_m,) orders 3 p n <= k_prof
    seeds: dict[str, Any]
    centres: dict[str, Any]


def _floor_ctrl_measured(lad_settings: list[Setting], basis_ctrl_hz: np.ndarray) -> np.ndarray:
    """The floor at the control points, in dB, at the REFERENCE speed: per
    non-windy mic, the lower envelope over +-10 % in frequency (20th percentile
    over 0.2231, the exponential's 20th-percentile-to-mean factor) less the
    mean ambient there (kept >= 10 % of the envelope), averaged over the mics,
    moved to 80 rev/s with the prior-median ``b(f)``, then averaged over the
    ladder in dB."""
    out = []
    for s in lad_settings:
        mics = [m for m in range(s.power.shape[0]) if m not in s.windy]
        f = s.freqs
        r = s.carrier / F_REF
        vals = []
        for fc in basis_ctrl_hz:
            sel = (f >= max(fc / 1.1, BAND_HZ[0])) & (f <= min(fc * 1.1, BAND_HZ[1]))
            if not sel.any():
                sel = (f >= BAND_HZ[1] / 1.2) & (f <= BAND_HZ[1])
            env = np.percentile(s.power[mics][:, sel], 20, axis=1) / 0.2231
            net = np.maximum(env - s.ambient[mics][:, sel].mean(axis=1), 0.1 * env)
            b = B0[0] + B1[0] * math.log2(max(fc, BAND_HZ[0]) / 2000.0)
            vals.append(10.0 * math.log10(float(net.mean())) - 10.0 * b * math.log10(r))
        out.append(vals)
    return np.mean(np.asarray(out), axis=0)


def build_ladder(motor: str, settings: list[Setting] | None = None) -> Ladder:
    """The ladder of ``motor`` from the cached real settings, or from
    ``settings`` (a planted render); seeds and prior centres measured on them."""
    if settings is None:
        settings = [load_setting(motor, t) for t in THROTTLES]
    tables = [line_table(s, s.k_max) for s in settings]
    eq = [pair_eq(settings, tables, j) for j in range(len(settings))]
    k_prof = max(s.k_max for s in settings)
    motor_orders = np.arange(3 * POLE_PAIRS, k_prof + 1, 3 * POLE_PAIRS)

    # profile centres at the reference speed: the mic-mean level of each
    # setting moved with the prior-median exponent, averaged where the line is
    # visible (snr >= 3 dB at the mic median); else 10 dB under its floor band
    num = np.zeros(k_prof)
    cnt = np.zeros(k_prof)
    flo = np.full(k_prof, np.nan)
    for s, t in zip(settings, tables):
        lr = 10.0 * math.log10(s.carrier / F_REF)
        lev = t["level_db"].mean(axis=0) - ALPHA_AERO[0] * lr
        vis = np.median(t["snr_db"], axis=0) >= 3.0
        K = lev.size
        num[:K] += np.where(vis, lev, 0.0)
        cnt[:K] += vis
        fl = t["floor_db"].mean(axis=0) - ALPHA_AERO[0] * lr - 10.0
        flo[:K] = np.where(np.isnan(flo[:K]), fl, np.fmin(flo[:K], fl))
    p_hat = np.where(cnt > 0, num / np.maximum(cnt, 1), flo)

    basis_ctrl = SP.bench_grid(n=settings[0].n, sr=SR, apply_transfer=False).floor.ctrl_hz
    ctrl_db = _floor_ctrl_measured(settings, basis_ctrl)
    floor_mean = float(ctrl_db.mean())
    ctrl_rel = ctrl_db - floor_mean
    sigma_b = float(max(np.std(ctrl_rel), 1.0))

    keys = [s.key for s in settings]
    s2_seed, g_seed = _pedestal_seed(keys, k_prof)
    shaft = []
    for s in settings:
        q = _quick_shaft(s.key)
        shaft.append(q if q is not None else SHAFT_SEED)

    # wind centres: each mic's lower envelope at 30-90 Hz (lines excluded by the
    # percentile) over the calm-mic median; a calm mic sits 20 dB under it
    wind_c = []
    for s in settings:
        f = s.freqs
        low = (f >= BAND_HZ[0]) & (f <= 90.0)
        lev = np.percentile(s.power[:, low], 20, axis=1) / 0.2231
        calm = [m for m in range(lev.size) if m not in s.windy]
        ref = float(np.median(lev[calm]))
        wind_c.append(10.0 * np.log10(np.maximum(lev - ref, 1e-2 * ref)))

    m_idx = motor_orders - 1
    centres = dict(
        p_aero=p_hat,
        p_motor=p_hat[m_idx] + MOTOR_BOOST_DB,
        floor_mean_db=floor_mean,
        sigma_b_db=sigma_b,
        wind_db=wind_c,
    )
    # seed: split the measured sum at a shared order 1:4 (aero:motor) in power
    p_aero0 = p_hat.copy()
    p_aero0[m_idx] = p_hat[m_idx] - 7.0
    seeds = dict(
        p_aero=p_aero0,
        p_motor=p_hat[m_idx] - 1.0,
        floor_ctrl_rel_db=ctrl_rel,
        sigma2=s2_seed,
        gamma_am=g_seed,
        shaft=shaft,
    )
    return Ladder(
        motor=motor,
        settings=settings,
        tables=tables,
        eq=eq,
        k_prof=k_prof,
        motor_orders=motor_orders,
        seeds=seeds,
        centres=centres,
    )


def _lag_dtft(q: Tensor, n: int) -> Tensor:
    """``sum_{|tau| < n} q(|tau|) e^{-2 pi i j tau / n}``, ``j = 0 .. n // 2``,
    for a real even lag sequence given on ``tau = 0 .. L - 1`` (``L <= n``):
    ``2 Re rfft_n(q) - q(0)``. Equal to :func:`spectrum._even_bin_dtft` (its
    ``2 n`` transform read at the even bins IS the ``n``-point one) at half the
    transform length."""
    return 2.0 * torch.fft.rfft(q, n=n, dim=-1).real - q[..., 0:1]


def _ridge_z(chol: np.ndarray, sigma_b: float, ctrl_rel: np.ndarray) -> np.ndarray:
    """The ``z`` with ``sigma_b L z ~= ctrl``: ridge solution under ``N(0, I)``."""
    A = sigma_b * chol
    return np.linalg.solve(A.T @ A + np.eye(A.shape[1]), A.T @ ctrl_rel)


class LadderModel:
    """The v4 bench forward model and MAP objective of one ladder."""

    def __init__(
        self,
        lad: Ladder,
        *,
        device: Any = "cpu",
        seed: int = 0,
        jitter: float = 0.0,
        exclude_mics: tuple[int, ...] = (),
        pin_lam: float | None = None,
        sigma2_cap: float | None = SIGMA2_CAP,
        center_delta: bool = False,
    ):
        self.lad = lad
        #: the pedestal cap (explainer § am-pm, a RENDER limit); None fits it free
        self.sigma2_cap = sigma2_cap
        #: ``center_delta``: the scatter delta of a setting has zero mean over
        #: (mic, line), so a ladder-wide level trend can only go through alpha
        self.center_delta = center_delta
        self.device = torch.device(device)
        dd: dict[str, Any] = dict(dtype=torch.float64, device=self.device)
        rng = np.random.default_rng(seed)

        def jit(size: int) -> np.ndarray:
            return rng.normal(0.0, jitter, size) if jitter > 0 else np.zeros(size)

        K = lad.k_prof
        k = np.arange(1, K + 1, dtype=np.float64)
        self.k = k
        self.frozen: list[_Frozen] = []
        for j, s in enumerate(lad.settings):
            grid = SP.bench_grid(n=s.n, sr=SR, device=self.device, apply_transfer=False)
            f = grid.freqs_hz
            band = (f >= BAND_HZ[0]) & (f <= BAND_HZ[1])
            sig0, lam0 = lad.seeds["shaft"][j]
            groups = SP.order_groups(
                s.k_max,
                sigma_nu=sig0 / LAG_SIGMA_MARGIN,
                lam=min(max(lam0, 5.0), 30.0) * LAG_LAM_MARGIN,
                sr=SR,
                n=s.n,
                threshold_nats=LAG_NATS,
            )
            cos, mrows = [], []
            for length, orders in groups:
                tau = grid.tau_s[:length]
                kk = torch.as_tensor(orders, **dd)
                ang = torch.remainder(kk[:, None] * s.carrier * tau[None, :], 1.0)
                cos.append(torch.cos(2.0 * math.pi * ang))
                mrows.append(np.flatnonzero(orders % (3 * POLE_PAIRS) == 0))
            psd_len = grid.floor.psd_len
            self.frozen.append(
                _Frozen(
                    grid=grid,
                    power=torch.as_tensor(s.power, **dd),
                    ambient=torch.as_tensor(s.ambient, **dd),
                    band=torch.as_tensor(band, device=self.device),
                    wind=torch.as_tensor(wind_shape(f), **dd),
                    eq_db=torch.as_tensor(lad.eq[j], **dd),
                    log_r=math.log(s.carrier / F_REF),
                    groups=groups,
                    cos=cos,
                    motor_rows=mrows,
                    floor_f=torch.as_tensor(np.fft.rfftfreq(psd_len, d=1.0 / SR), **dd),
                )
            )
        chol = self.frozen[0].grid.floor.shape_chol.cpu().numpy()
        sigma_b = lad.centres["sigma_b_db"]
        z0 = _ridge_z(chol, sigma_b, lad.seeds["floor_ctrl_rel_db"])
        M = lad.settings[0].power.shape[0]
        n_m = lad.motor_orders.size

        def P(v: Any) -> Tensor:
            return torch.as_tensor(np.asarray(v, dtype=np.float64), **dd).requires_grad_(True)

        # the shaft law, seeded at a = b = 1 through the per-setting seeds
        sh = np.asarray(lad.seeds["shaft"], dtype=np.float64)
        log_r = np.log([s.carrier / F_REF for s in lad.settings])
        log_eps0 = float(np.median(np.log(sh[:, 0]) - log_r)) - math.log(2.0 * math.pi * F_REF)
        log_lam0 = float(np.median(np.log(sh[:, 1]) - log_r))
        self.params: dict[str, Tensor] = dict(
            p_aero=P(lad.seeds["p_aero"]),
            p_motor=P(lad.seeds["p_motor"]),
            alpha_aero=P(ALPHA_AERO[0]),
            alpha_motor=P(ALPHA_MOTOR[0]),
            b0=P(B0[0]),
            b1=P(B1[0]),
            u_sigma2=P(np.log(lad.seeds["sigma2"]) + jit(K)),
            log_gamma=P(np.log(lad.seeds["gamma_am"]) + jit(K)),
            floor_mean_db=P(lad.centres["floor_mean_db"]),
            floor_z=P(z0),
            mic_floor_db=P(np.zeros(M)),
            log_eps=P(log_eps0 + jit(1)[0]),
            a_sigma=P(A_SIGMA[0]),
            **{f"delta_{j}": P(np.zeros((M, s.k_max))) for j, s in enumerate(lad.settings)},
            **{f"wind_{j}": P(lad.centres["wind_db"][j]) for j in range(len(lad.settings))},
        )
        #: ``pin_lam``: lam = pin_lam r^1, both held fixed (the pinned-rate arm)
        self.pinned: dict[str, Tensor] = {}
        if pin_lam is None:
            self.params["log_lam_ref"] = P(log_lam0 + jit(1)[0])
            self.params["b_lam"] = P(B_LAM[0])
        else:
            self.pinned["log_lam_ref"] = torch.as_tensor(math.log(pin_lam), **dd)
            self.pinned["b_lam"] = torch.as_tensor(B_LAM[0], **dd)
        assert self.params["p_motor"].numel() == n_m
        #: the microphones the likelihood reads (all unless ``exclude_mics``)
        self.mics = torch.as_tensor(
            [m for m in range(M) if m not in set(exclude_mics)], device=self.device
        )
        self.n_cells = int(sum(int(fz.band.sum()) * int(self.mics.numel()) for fz in self.frozen))

    # ── pieces ──
    def sigma2(self) -> Tensor:
        return _capped_sigma2(self.params["u_sigma2"], self.sigma2_cap)

    def gamma_am(self) -> Tensor:
        return torch.exp(self.params["log_gamma"])

    def delta(self, j: int) -> Tensor:
        """``(M, K)`` per-run line scatter of setting ``j`` (dB)."""
        d = self.params[f"delta_{j}"]
        return d - d.mean() if self.center_delta else d

    def shaft(self, j: int) -> tuple[Tensor, Tensor]:
        """``(sigma_nu rad/s, lam 1/s)`` of setting ``j`` from the ladder's
        shaft law: ``sigma_nu = 2 pi f_ref eps r^a``, ``lam = lam_ref r^b``."""
        p, lr = {**self.pinned, **self.params}, self.frozen[j].log_r
        sig = 2.0 * math.pi * F_REF * torch.exp(p["log_eps"] + p["a_sigma"] * lr)
        lam = torch.exp(p["log_lam_ref"] + p["b_lam"] * lr)
        return sig, lam

    def _floor_params(self) -> SP.FloorParams:
        p = self.params
        return SP.FloorParams(
            mean_db=p["floor_mean_db"],
            shape_z=p["floor_z"],
            tilt_db_oct=torch.zeros((), dtype=torch.float64, device=self.device),
            mic_floor_db=torch.zeros(1, dtype=torch.float64, device=self.device),
            shape_sd_db=torch.as_tensor(
                self.lad.centres["sigma_b_db"], dtype=torch.float64, device=self.device
            ),
        )

    def line_weights(self, j: int) -> tuple[Tensor, Tensor, np.ndarray]:
        """``(w_aero (M, K), w_motor (M, n_m), motor_k)``: each line's mean power
        at setting ``j`` per mic, ``10^{(p + 10 alpha log10 r + E + delta)/10}``."""
        p, fz, s = self.params, self.frozen[j], self.lad.settings[j]
        K = s.k_max
        lr10 = 10.0 * fz.log_r / math.log(10.0)
        mic_db = fz.eq_db + self.delta(j)  # (M, K)
        w_aero = 10.0 ** ((p["p_aero"][:K][None, :] + p["alpha_aero"] * lr10 + mic_db) / 10.0)
        motor_k = self.lad.motor_orders[self.lad.motor_orders <= K]
        w_motor = 10.0 ** (
            (
                p["p_motor"][: motor_k.size][None, :]
                + p["alpha_motor"] * lr10
                + mic_db[:, motor_k - 1]
            )
            / 10.0
        )
        return w_aero, w_motor, motor_k

    def components(self, j: int) -> dict[str, Tensor]:
        """Setting ``j``'s modelled parts, each ``(M, F)``: ``lines``, ``floor``
        (with the mic gains) and ``wind``. The ambient is data, not a part."""
        p, fz, s = self.params, self.frozen[j], self.lad.settings[j]
        grid = fz.grid
        n = grid.n
        M = s.power.shape[0]
        sig, lam = self.shaft(j)
        s2 = self.sigma2()
        gam = self.gamma_am()
        w_aero, w_motor, motor_k = self.line_weights(j)
        acov = torch.zeros(M, n, dtype=torch.float64, device=self.device)
        motor_pos = {int(kk): i for i, kk in enumerate(motor_k)}
        for (length, orders), cos, mrows in zip(fz.groups, fz.cos, fz.motor_rows):
            idx = torch.as_tensor(orders - 1, device=self.device)
            tau = grid.tau_s[:length][None, :]
            kk = torch.as_tensor(orders, dtype=torch.float64, device=self.device)[:, None]
            rho = r_tau_v4(
                tau,
                kk,
                sigma_nu=sig,
                lam=lam,
                sigma2=s2[idx][:, None],
                gamma_am_hz=gam[idx][:, None],
            )
            block = w_aero[:, idx] @ (rho * cos)  # (M, L)
            if mrows.size:
                rows = torch.as_tensor(mrows, device=self.device)
                shaft = r_tau_v4(tau, kk[rows], sigma_nu=sig, lam=lam, sigma2=0.0, gamma_am_hz=0.0)
                cols = [motor_pos[int(orders[r])] for r in mrows]
                block = block + w_motor[:, cols] @ (shaft * cos[rows])
            acov[:, :length] += block
        lines = _lag_dtft(grid.win_acf[None, :] * acov, n) / grid.window_sumsq

        basis = grid.floor
        psd = basis.psd(self._floor_params())
        b = p["b0"] + p["b1"] * torch.log2(torch.clamp(fz.floor_f, min=BAND_HZ[0]) / 2000.0)
        psd = psd * torch.exp(b * fz.log_r)
        c = torch.fft.irfft(psd.to(torch.complex128), n=basis.psd_len).real[: basis.lag_len]
        floor = _lag_dtft(grid.win_acf[: basis.lag_len] * c, n) / grid.window_sumsq  # (F,)
        mf = p["mic_floor_db"]
        g = 10.0 ** ((mf - mf.mean()) / 10.0)
        wind = 10.0 ** (p[f"wind_{j}"][:, None] / 10.0) * fz.wind[None, :]
        return dict(lines=lines, floor=g[:, None] * floor[None, :], wind=wind)

    def expected(self, j: int) -> Tensor:
        """``(M, F)`` expected periodogram of setting ``j``."""
        c = self.components(j)
        out = c["lines"] + c["floor"] + c["wind"] + self.frozen[j].ambient
        return out.clamp_min(MODEL_FLOOR)

    def load_payload(self, pay: dict[str, Any]) -> None:
        """Set every parameter from a ``noise-v4-fit/2`` payload of the same
        ladder geometry (the planted check's truth); a pinned rate takes the
        payload's value too."""
        sh = pay["shared"]
        s2 = np.asarray(sh["pedestal"]["sigma2"], dtype=np.float64)
        if self.sigma2_cap is None:
            u = np.log(s2)
        else:
            # inverse of _capped_sigma2: u = log cap - softplus^-1(log cap - log s2)
            log_cap = math.log(self.sigma2_cap)
            u = log_cap - np.log(np.expm1(np.maximum(log_cap - np.log(s2), 1e-12)))
        vals = dict(
            p_aero=sh["profile_aero_db"],
            p_motor=sh["profile_motor_db"],
            alpha_aero=sh["alpha_aero"],
            alpha_motor=sh["alpha_motor"],
            b0=sh["floor"]["b0"],
            b1=sh["floor"]["b1"],
            u_sigma2=u,
            log_gamma=np.log(sh["pedestal"]["gamma_hz"]),
            floor_mean_db=sh["floor"]["mean_db"],
            floor_z=sh["floor"]["shape_z"],
            mic_floor_db=sh["floor"]["mic_floor_db"],
            log_eps=math.log(sh["shaft"]["eps"]),
            a_sigma=sh["shaft"]["a_sigma"],
            log_lam_ref=math.log(sh["shaft"]["lam_ref_per_s"]),
            b_lam=sh["shaft"]["b_lam"],
        )
        for j, st in enumerate(pay["settings"]):
            vals[f"delta_{j}"] = st["delta_db"]
            vals[f"wind_{j}"] = st["wind_db"]
        with torch.no_grad():
            for k, v in vals.items():
                dst = self.params[k] if k in self.params else self.pinned[k]
                dst.copy_(torch.as_tensor(np.asarray(v, dtype=np.float64)))

    def whittle(self, j: int) -> Tensor:
        fz = self.frozen[j]
        m = self.expected(j)[self.mics][:, fz.band]
        i = fz.power[self.mics][:, fz.band]
        return (torch.log(m) + i / m).sum()

    def neg_log_prior(self) -> Tensor:
        p, lad = self.params, self.lad
        dd: dict[str, Any] = dict(dtype=torch.float64, device=self.device)

        def nrm(x: Tensor, mu: Any, sd: Any) -> Tensor:
            mu_t = torch.as_tensor(mu, **dd)
            sd_t = torch.as_tensor(sd, **dd)
            return (0.5 * ((x - mu_t) / sd_t) ** 2).sum()

        k = self.k
        mu_s, sd_s = _log_sigma2_prior(k)
        mu_g, sd_g = _log_gamma_prior(k)
        out = (
            nrm(p["p_aero"], lad.centres["p_aero"], PROFILE_SD_DB)
            + nrm(p["p_motor"], lad.centres["p_motor"], PROFILE_SD_DB)
            + nrm(p["alpha_aero"], *ALPHA_AERO)
            + nrm(p["alpha_motor"], *ALPHA_MOTOR)
            + nrm(p["b0"], *B0)
            + nrm(p["b1"], *B1)
            + nrm(p["u_sigma2"], mu_s, sd_s)
            + nrm(p["log_gamma"], mu_g, sd_g)
            + nrm(p["floor_mean_db"], lad.centres["floor_mean_db"], FLOOR_MEAN_SD_DB)
            + nrm(p["floor_z"], 0.0, 1.0)
            + nrm(p["mic_floor_db"], 0.0, MIC_FLOOR_SD_DB)
            + 0.5 * (torch.exp(p["log_eps"]) / EPS_HALF_NORMAL) ** 2
            + nrm(p["a_sigma"], *A_SIGMA)
        )
        if "b_lam" in p:
            out = out + nrm(p["b_lam"], *B_LAM)
        for j, s in enumerate(lad.settings):
            M, K = s.power.shape[0], s.k_max
            sd = np.full((M, K), SCATTER_SD_DB)
            for m in s.windy:
                sd[m, : SCATTER_WINDY_BELOW_K - 1] = SCATTER_WINDY_SD_DB
            out = out + nrm(p[f"delta_{j}"], 0.0, sd)
            out = out + nrm(p[f"wind_{j}"], lad.centres["wind_db"][j], WIND_SD_DB)
        return out

    def objective_and_grad(self) -> tuple[float, list[float]]:
        """Total objective (nats), gradients left in ``.grad``; one setting at
        a time so only one setting's graph is alive."""
        for t in self.params.values():
            t.grad = None
        prior = self.neg_log_prior()
        prior.backward()
        parts = []
        for j in range(len(self.frozen)):
            w = self.whittle(j)
            w.backward()
            parts.append(float(w.detach()))
        return float(prior.detach()) + sum(parts), parts


def lag_support_check(model: LadderModel, *, nats: float = 20.0) -> list[dict[str, Any]]:
    """Per setting: the orders whose FITTED shaft factor is still above
    ``e^-nats`` where the frozen lag grid cuts them (a grid that was too short)."""
    out = []
    for j, (fz, s) in enumerate(zip(model.frozen, model.lad.settings)):
        sig_t, lam_t = model.shaft(j)
        sig, lam = float(sig_t), float(lam_t)
        short = []
        for length, orders in fz.groups:
            need = order_lag_support_s(
                orders.astype(np.float64),
                sigma_nu=sig,
                lam=lam,
                threshold_nats=nats,
                tau_cap_s=s.n / SR,
            )
            bad = orders[np.ceil(np.atleast_1d(need) * SR) > length]
            short += bad.tolist()
        out.append(dict(key=s.key, sigma_nu=sig, lam=lam, short_orders=short))
    return out


# ── the fit ──────────────────────────────────────────────────────────────────


def _run_lbfgs(model: LadderModel, iters: int, history: int = 10) -> dict[str, Any]:
    params = list(model.params.values())
    opt = LBFGS(
        params,
        lr=1.0,
        max_iter=int(iters),
        max_eval=int(iters * 1.25) + 1,
        history_size=history,
        line_search_fn="strong_wolfe",
        tolerance_grad=0.0,
        tolerance_change=0.0,
    )
    n_eval = [0]

    def closure() -> Tensor:
        total, _ = model.objective_and_grad()
        n_eval[0] += 1
        return torch.as_tensor(total, dtype=torch.float64)

    t0 = time.perf_counter()
    opt.step(closure)
    return dict(
        iters=int(opt.n_iter),
        evals=n_eval[0],
        trace=[float(v) for v in opt.trace],
        wall_s=time.perf_counter() - t0,
    )


def fit_ladder(
    lad: Ladder,
    *,
    seed: int = 0,
    jitter: float = 0.8,
    adam_steps: int = 300,
    adam_lr: float = 0.02,
    lbfgs_iters: int = 400,
    device: Any = "cpu",
    exclude_mics: tuple[int, ...] = (),
    pin_lam: float | None = None,
    sigma2_cap: float | None = SIGMA2_CAP,
    center_delta: bool = False,
    log: Any = print,
) -> tuple[LadderModel, dict[str, Any]]:
    """MAP of one ladder: Adam, then L-BFGS, then a fresh-memory L-BFGS restart
    with half the budget; converged iff the restart gains less than
    ``TOL_NATS_PER_CELL`` per cell."""
    model = LadderModel(
        lad,
        device=device,
        seed=seed,
        jitter=jitter if seed > 0 else 0.0,
        exclude_mics=exclude_mics,
        pin_lam=pin_lam,
        sigma2_cap=sigma2_cap,
        center_delta=center_delta,
    )
    t_all = time.perf_counter()
    t0 = time.perf_counter()
    first, _ = model.objective_and_grad()
    eval_s = time.perf_counter() - t0
    log(
        f"[{lad.motor} s{seed}] start {first:.1f} nats ({model.n_cells} cells, {eval_s:.2f} s/eval)"
    )
    adam_trace = []
    if adam_steps > 0:
        opt = torch.optim.Adam(list(model.params.values()), lr=adam_lr)
        for step in range(int(adam_steps)):
            total, _ = model.objective_and_grad()
            opt.step()
            adam_trace.append(total)
            if step % 50 == 0:
                log(f"[{lad.motor} s{seed}] adam {step} {total:.1f}")
    pass1 = _run_lbfgs(model, lbfgs_iters)
    log(f"[{lad.motor} s{seed}] lbfgs pass {pass1['trace'][-1]:.1f} ({pass1['iters']} it)")
    restart = _run_lbfgs(model, max(1, lbfgs_iters // 2))
    gain = pass1["trace"][-1] - restart["trace"][-1]
    total, parts = model.objective_and_grad()
    grad_norm = math.sqrt(
        sum(float((t.grad**2).sum()) for t in model.params.values() if t.grad is not None)
    )
    record = dict(
        seed=int(seed),
        jitter=float(jitter if seed > 0 else 0.0),
        pin_lam=pin_lam,
        sigma2_cap=sigma2_cap,
        center_delta=center_delta,
        adam_steps=int(adam_steps),
        adam_lr=float(adam_lr),
        lbfgs_iters=int(lbfgs_iters),
        objective_start=first,
        objective=total,
        whittle_per_setting=parts,
        n_cells=model.n_cells,
        eval_s=eval_s,
        adam_trace=adam_trace,
        lbfgs_pass=pass1,
        lbfgs_restart=restart,
        restart_gain_per_cell=gain / model.n_cells,
        converged=bool(gain / model.n_cells < TOL_NATS_PER_CELL),
        grad_norm=grad_norm,
        wall_s=time.perf_counter() - t_all,
    )
    log(
        f"[{lad.motor} s{seed}] done {total:.1f} nats, restart gain "
        f"{record['restart_gain_per_cell']:.2e}/cell, converged={record['converged']}"
    )
    return model, record


def render_planted(truth: LadderModel, *, seed: int = 0) -> list[Setting]:
    """Time-domain renders of ``truth``'s ladder at its own geometry (carrier,
    span length, mics): the aerodynamic lines with their pedestal and the motor
    lines on ONE shared OU shaft per setting
    (:func:`static_rig.single_rotor.synthesise`, same seed for both families so
    they share the shaft), plus Gaussian noise shaped to the model's floor and
    wind, plus Gaussian noise shaped to the measured ambient.

    ``synthesise`` writes ``sqrt(amp2) cos(.)`` (mean power ``amp2 / 2``) and
    normalises its pedestal to unit mean AMPLITUDE (mean power ``e^{sigma2}``);
    the fit's ``10^{p/10}`` is the mean POWER, so ``amp2 = 2 P e^{-sigma2}``.
    """
    from experiments.static_rig.single_rotor import OUFit, synthesise

    rng = np.random.default_rng(seed)
    s2 = truth.sigma2().detach().cpu().numpy()
    gam = truth.gamma_am().detach().cpu().numpy()
    out = []
    for j, s in enumerate(truth.lad.settings):
        with torch.no_grad():
            w_aero, w_motor, motor_k = truth.line_weights(j)
            comp = truth.components(j)
        K = s.k_max
        sig_t, lam_t = truth.shaft(j)
        sig, lam = float(sig_t) / (2.0 * math.pi), float(lam_t)
        shaft_seed = int(rng.integers(2**31))
        aero = OUFit(
            s=s.carrier,
            sigma_nu=sig,
            lam=lam,
            orders=np.arange(1, K + 1),
            amp2=2.0 * w_aero.cpu().numpy() * np.exp(-s2[:K])[None, :],
            T=s.n / SR,
        )
        x = synthesise(aero, SR, s.n, seed=shaft_seed, am=(s2[:K], gam[:K])).astype(np.float64)
        if motor_k.size:
            motor = OUFit(
                s=s.carrier,
                sigma_nu=sig,
                lam=lam,
                orders=motor_k,
                amp2=2.0 * w_motor.cpu().numpy(),
                T=s.n / SR,
            )
            x += synthesise(motor, SR, s.n, seed=shaft_seed).astype(np.float64)
        shaped = (comp["floor"] + comp["wind"]).cpu().numpy()
        for spec in (shaped, s.ambient):
            white = np.fft.rfft(rng.standard_normal(x.shape), axis=-1)
            x += np.fft.irfft(white * np.sqrt(np.maximum(spec, 0.0)), n=s.n, axis=-1)
        out.append(
            Setting(
                key=s.key,
                motor=s.motor,
                throttle=s.throttle,
                carrier=s.carrier,
                span_s=s.span_s,
                ambient_runs_s=s.ambient_runs_s,
                power=_periodogram(x),
                ambient=s.ambient,
                n=s.n,
            )
        )
    return out


def payload(
    model: LadderModel, record: dict[str, Any], *, git: str | None = None
) -> dict[str, Any]:
    """The ``noise-v4-fit/2`` bench payload of one ladder."""
    p = {k: v.detach().cpu().numpy() for k, v in {**model.pinned, **model.params}.items()}
    lad = model.lad
    s2 = model.sigma2().detach().cpu().numpy()
    gm = model.gamma_am().detach().cpu().numpy()
    settings = []
    for j, s in enumerate(lad.settings):
        sig_t, lam_t = model.shaft(j)
        settings.append(
            dict(
                key=s.key,
                throttle=s.throttle,
                carrier_rev_s=s.carrier,
                span_s=list(s.span_s),
                ambient_runs_s=[list(r) for r in s.ambient_runs_s],
                n_samples=s.n,
                k_max=s.k_max,
                sigma_nu_rad_s=float(sig_t),
                lam_per_s=float(lam_t),
                wind_db=p[f"wind_{j}"].tolist(),
                delta_db=model.delta(j).detach().cpu().numpy().tolist(),
                eq_db=lad.eq[j].tolist(),
                whittle_nats=record["whittle_per_setting"][j],
                n_cells=int(model.frozen[j].band.sum()) * int(model.mics.numel()),
            )
        )
    floor = model.frozen[0].grid.floor
    return dict(
        schema=FIT_SCHEMA,
        kind="bench_ladder",
        mics_fitted=model.mics.cpu().tolist(),
        motor=lad.motor,
        rig="dregon",
        blades=2,
        pole_pairs=POLE_PAIRS,
        f_ref_rev_s=F_REF,
        sr=SR,
        band_hz=list(BAND_HZ),
        shared=dict(
            profile_aero_db=p["p_aero"].tolist(),
            motor_orders=lad.motor_orders.tolist(),
            profile_motor_db=p["p_motor"].tolist(),
            alpha_aero=float(p["alpha_aero"]),
            alpha_motor=float(p["alpha_motor"]),
            pedestal=dict(sigma2=s2.tolist(), gamma_hz=gm.tolist(), cap=model.sigma2_cap),
            shaft=dict(
                eps=float(np.exp(p["log_eps"])),
                a_sigma=float(p["a_sigma"]),
                lam_ref_per_s=float(np.exp(p["log_lam_ref"])),
                b_lam=float(p["b_lam"]),
                f_ref_rev_s=F_REF,
                pinned=sorted(model.pinned),
                law="sigma_nu = 2 pi f_ref eps r^a_sigma, lam = lam_ref r^b_lam, r = f / f_ref",
            ),
            floor=dict(
                mean_db=float(p["floor_mean_db"]),
                shape_z=p["floor_z"].tolist(),
                shape_sd_db=float(lad.centres["sigma_b_db"]),
                ctrl_hz=floor.ctrl_hz.tolist(),
                b0=float(p["b0"]),
                b1=float(p["b1"]),
                mic_floor_db=(p["mic_floor_db"] - p["mic_floor_db"].mean()).tolist(),
            ),
        ),
        settings=settings,
        objective=dict(
            total_nats=record["objective"],
            whittle_nats=float(sum(record["whittle_per_setting"])),
            n_cells=model.n_cells,
        ),
        optimiser={k: v for k, v in record.items() if k not in ("whittle_per_setting",)},
        lag_support=lag_support_check(model),
        centres=dict(
            p_aero=np.asarray(lad.centres["p_aero"]).tolist(),
            p_motor=np.asarray(lad.centres["p_motor"]).tolist(),
            floor_mean_db=lad.centres["floor_mean_db"],
            sigma_b_db=lad.centres["sigma_b_db"],
            wind_db=[np.asarray(w).tolist() for w in lad.centres["wind_db"]],
        ),
        git=git,
    )
