"""Noise model v4 — Fit procedure A, the DREGON bench rig fit
(``docs/explainers/noise-model-v4.qmd`` § Procedure A).

ONE MAP fit of the DREGON drone over its 20 single-rotor recordings
(``motor_Motor{1-4}_{50..90}``), one whole-span periodic-Hann periodogram per
recording and microphone (16 kHz, 0.03-0.1 Hz bins). Recording ``j`` of rotor
``r``, microphone ``m``::

    M_jm(f) = sum_k h_rm(k f_j) [P^a_rjk L^a_jk(f) + P^m_rjk L^m_jk(f)]   comb
            + h_rm(f) F(f) s_j^{b(f)}                                    rotor floor
            + 10^{w_jm/10} s_W(f)   (windy microphones only)            wind
            + N_jm(f)                                                    ambient (data)

``P = 10^{(p_rk + 10 alpha log10 s_j)/10}``, ``s_j = f_j / 80``; the four rotor
profiles share a centre (``p_rk = pbar_k + c_r + e_rk``); ``alpha^a``,
``alpha^m``, ``F``, ``b(f)``, the pedestal and the shaft law are shared by the
rotors. ``L`` is the window-convolved line of the spec's lag law: shaft OU x
log-amplitude OU x phase OU (motor lines: shaft only). ``h_rm = 10^{E_rm/10}``
is the pair EQ, measured on the rotor's OTHER recordings. No per-line scatter,
no free microphone gains (decisions of 2026-10-21).

Every line is summed in the LAG domain (exact for a constant carrier) over
order groups whose lag support is fixed before the fit for a narrow shaft
(:data:`LAG_SHAFT`); :func:`lag_support_check` re-reads the support at the
fitted shaft. The optimiser is the spec's (§ The optimiser): Adam, L-BFGS, the
evidence freeze of the phase pedestal (:func:`phase_evidence`), a fresh-memory
L-BFGS restart pass with the hold, and a final pass with the hold released
(:func:`fit_rig`).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from torch import Tensor

from data_processing.noise_model.lag import order_lag_support_s, r_tau_v4
from data_processing.noise_model.resample import decimate_audio
from data_processing.noise_model.v3 import wind_shape, wind_shape_spec
from experiments.stochastic_fit import clips as C
from experiments.stochastic_fit.model import AMP_RPS_REF
from experiments.stochastic_fit.phase_kernel import hann_window
from experiments.stochastic_fit.revised_phase import MODEL_FLOOR

from . import spectrum as SP
from . import v4_priors as PR
from .lbfgs import LBFGS

__all__ = [
    "MOTORS",
    "THROTTLES",
    "WIND_MICS",
    "Rig",
    "RigModel",
    "Setting",
    "build_rig",
    "fit_rig",
    "group_movement",
    "group_vectors",
    "lag_support_check",
    "load_setting",
    "pair_eq",
    "payload",
    "payload_group_vectors",
    "phase_evidence",
    "prepare_setting",
    "render_planted",
    "render_setting",
]

DATASET = "DREGON-frames"
RPS_KEY = "motors_command"
SR = 16000
F_REF = float(AMP_RPS_REF)
THROTTLES = (50, 60, 70, 80, 90)
MOTORS = ("Motor1", "Motor2", "Motor3", "Motor4")
#: DREGON: two blades, seven pole pairs, so the motor family sits at 21 n.
BLADES = 2
POLE_PAIRS = 7
#: The downwash microphones per motor (spec step A4).
WIND_MICS = {"Motor1": (5, 6), "Motor2": (0,), "Motor3": (1, 2), "Motor4": (4,)}

SINGLES_PATH = Path("results/static_rig/single_rotor/dregon_singles.json")
CACHE_DIR = Path("results/noise_v4/bench/settings")

#: The likelihood band: 30 Hz to 0.45 sr (7.2 kHz), clear of the decimator.
BAND_HZ = (SP.BAND_F_MIN, min(SP.BAND_F_MAX, 0.45 * SR))
LAG_NATS = SP.LAG_SUPPORT_NATS
#: The narrow shaft (sigma_nu rad/s, lam 1/s) that fixes the lag support of
#: every line before the fit: a wider fitted shaft only shortens it.
LAG_SHAFT = (0.2, 45.0)
#: The pair EQ: Gaussian smoothing width and the payload's frequency grid.
EQ_WIDTH_HZ = 200.0
EQ_GRID_HZ = np.arange(0.0, SR / 2 + 10.0, 10.0)
#: Profile visibility for the level initialisation (mic-median line SNR).
VISIBLE_SNR_DB = 3.0
#: Floor-mean and wind priors (nv3r4).
FLOOR_MEAN_SD_DB = 10.0
WIND_SD_DB = 6.0
#: Prior medians: a half-normal's median is 0.674 of its scale.
HALF_NORMAL_MEDIAN = 0.6745
#: The evidence freeze holds a phase-pedestal parameter where the data shrink
#: its prior variance by less than 10 %: Fisher < prior precision / 9.
EVIDENCE_RATIO = 1.0 / 9.0
#: Convergence (spec § The optimiser): gain per cell AND per-group movement.
TOL_NATS_PER_CELL = 1e-4
TOL_MOVE = dict(levels=0.1, exponents=0.02, floor=0.1, dynamics=0.05)


def _capped(u: Tensor, log_cap: Tensor | float) -> Tensor:
    """``exp(log cap - softplus(log cap - u))``: ``e^u`` far below the cap,
    never above it (the truncation of a prior's support)."""
    return torch.exp(log_cap - torch.nn.functional.softplus(log_cap - u))


def _capped_np(u: np.ndarray, cap: np.ndarray | float) -> np.ndarray:
    """:func:`_capped` on numpy arrays."""
    lc = np.log(cap)
    return np.exp(lc - np.logaddexp(0.0, lc - u))


def _uncapped(s2: np.ndarray, cap: np.ndarray | float) -> np.ndarray:
    """Inverse of :func:`_capped`."""
    lc = np.log(cap)
    return lc - np.log(np.expm1(np.maximum(lc - np.log(s2), 1e-12)))


def _under(v: np.ndarray, cap: float) -> np.ndarray:
    """``v`` with entries at or over ``cap`` moved to ``0.99 cap``."""
    return np.where(v >= cap, 0.99 * cap, v)


# ── data ─────────────────────────────────────────────────────────────────────


@dataclass
class Setting:
    """One stationary recording, everything the fit holds fixed."""

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
    def calm(self) -> list[int]:
        return [m for m in range(self.power.shape[0]) if m not in self.windy]

    @property
    def k_max(self) -> int:
        return int(BAND_HZ[1] // self.carrier)

    @property
    def speed(self) -> float:
        """``s_j = f_j / f_ref``."""
        return self.carrier / F_REF


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


def load_audio(motor: str, throttle: int) -> tuple[np.ndarray, tuple[float, float]]:
    """``(x (M, T) at 16 kHz, strict rotor-on span in s)`` of one recording."""
    key = f"motor_{motor}_{int(throttle)}"
    singles = json.loads(SINGLES_PATH.read_text())["recordings"][key]
    rec = C.load_recording(DATASET, key, None, RPS_KEY)
    x = decimate_audio(np.asarray(rec.audio, dtype=np.float64), int(rec.sr), SR)
    return x, (float(singles["span_s"][0]), float(singles["span_s"][1]))


def prepare_setting(motor: str, throttle: int) -> Setting:
    """Load one recording, decimate it to 16 kHz, cut the strict span and the
    quiet run, and read the periodogram and the ambient."""
    key = f"motor_{motor}_{int(throttle)}"
    singles = json.loads(SINGLES_PATH.read_text())["recordings"][key]
    x, span = load_audio(motor, throttle)
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


# ── measurements (pair EQ, prior centres, initial values) ────────────────────


def line_table(s: Setting, k_max: int) -> dict[str, np.ndarray]:
    """Per mic and order: line power (dB, the model's ``10^{p/10}`` units),
    the floor power in the same band (dB), their ratio (``snr_db``) and the
    floor of ONE bin in line units (``floor_bin_db``, ``(2/n) F``).

    ``P = (2/n) sum_band (I - F)`` over ``|f - k f_r| <= max(1 Hz, 3 bins)``;
    ``F`` is the annulus (6-18 Hz away) median over ``ln 2`` (the exponential
    median-to-mean factor)."""
    f = s.freqs
    df = float(f[1])
    half = max(1.0, 3.0 * df)
    M = s.power.shape[0]
    level = np.full((M, k_max), np.nan)
    floor = np.full((M, k_max), np.nan)
    floor_bin = np.full((M, k_max), np.nan)
    for k in range(1, k_max + 1):
        fc = k * s.carrier
        band = np.abs(f - fc) <= half
        ann = (np.abs(f - fc) >= 6.0) & (np.abs(f - fc) <= 18.0)
        F = np.median(s.power[:, ann], axis=1) / math.log(2.0)
        tot = s.power[:, band].sum(axis=1)
        flo = F * band.sum()
        level[:, k - 1] = (2.0 / s.n) * np.maximum(tot - flo, 1e-3 * flo)
        floor[:, k - 1] = (2.0 / s.n) * flo
        floor_bin[:, k - 1] = (2.0 / s.n) * F
    return dict(
        level_db=10.0 * np.log10(level),
        floor_db=10.0 * np.log10(floor),
        floor_bin_db=10.0 * np.log10(floor_bin),
        snr_db=10.0 * np.log10(level / floor + 1.0),
    )


def pair_eq(
    settings: list[Setting],
    tables: list[dict[str, np.ndarray]],
    freqs_hz: np.ndarray,
    *,
    use: list[int],
    min_snr_db: float = 10.0,
    width_hz: float = EQ_WIDTH_HZ,
) -> np.ndarray:
    """``E (M, F)`` dB at ``freqs_hz``: each mic's line level relative to the
    mic mean, pooled over the recordings ``use`` (lines >= ``min_snr_db`` on
    every mic) on the frequency axis and smoothed with a ``width_hz``
    Gaussian. A frequency with no nearby evidence shrinks to zero (one
    pseudo-observation of weight one at zero)."""
    fs, devs = [], []
    for i in use:
        s, t = settings[i], tables[i]
        ok = np.all(t["snr_db"] >= min_snr_db, axis=0)
        lev = t["level_db"][:, ok]
        devs.append(lev - lev.mean(axis=0, keepdims=True))
        fs.append(s.carrier * (np.flatnonzero(ok) + 1.0))
    f_obs = np.concatenate(fs)
    d_obs = np.concatenate(devs, axis=1)  # (M, N)
    f_tgt = np.asarray(freqs_hz, dtype=np.float64)
    w = np.exp(-0.5 * ((f_tgt[:, None] - f_obs[None, :]) / width_hz) ** 2)  # (F, N)
    return (d_obs @ w.T) / (w.sum(axis=1)[None, :] + 1.0)


def _interp_rows(f: np.ndarray, grid: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Linear interpolation of every row of ``rows`` (on ``grid``) at ``f``,
    flat outside."""
    return np.stack([np.interp(f, grid, r) for r in rows])


def _floor_ctrl_init(settings: list[Setting], eq_grid_db: list[np.ndarray]) -> np.ndarray:
    """The floor at the 14 control points (dB, one rotor at f_ref), spec
    § Initialisation step 1: per recording, the calm-mic mean periodogram
    smoothed over 1 Hz; its minimum in each gap between consecutive harmonics,
    less the calm-mic ambient there (kept >= 10 %), divided by the calm-mic
    pair EQ and by ``s_j^{b(f)}`` at the prior-centre ``b``; the median within
    each spline cell; the mean over recordings in dB. Cells with no gap are
    interpolated in log frequency."""
    from scipy.ndimage import uniform_filter1d

    ctrl = SP.bench_grid(n=settings[0].n, sr=SR, apply_transfer=False).floor.ctrl_hz
    edges = np.concatenate([[0.0], np.sqrt(ctrl[:-1] * ctrl[1:]), [np.inf]])
    rows = []
    for s, eqg in zip(settings, eq_grid_db):
        f = s.freqs
        calm = s.calm
        width = max(3, int(math.ceil(1.0 / float(f[1]))))
        spec = uniform_filter1d(s.power[calm].mean(axis=0), size=width, mode="nearest")
        amb = s.ambient[calm].mean(axis=0)
        fc, val = [], []
        for k in range(1, s.k_max):
            lo, hi = k * s.carrier, (k + 1) * s.carrier
            if lo < BAND_HZ[0] or hi > BAND_HZ[1]:
                continue
            a, b = int(np.searchsorted(f, lo, "right")), int(np.searchsorted(f, hi, "left"))
            i = a + int(np.argmin(spec[a:b]))
            v = float(spec[i])
            fc.append(0.5 * (lo + hi))
            val.append(max(v - float(amb[i]), 0.1 * v))
        fc_a, val_a = np.asarray(fc), np.asarray(val)
        b_exp = PR.B0[0] + PR.B1[0] * np.log2(np.maximum(fc_a, BAND_HZ[0]) / 2000.0)
        h = (10.0 ** (_interp_rows(fc_a, EQ_GRID_HZ, eqg[calm]) / 10.0)).mean(axis=0)
        db = 10.0 * np.log10(val_a / h) - 10.0 * b_exp * math.log10(s.speed)
        cell = np.searchsorted(edges, fc_a, "right") - 1
        row = np.full(ctrl.size, np.nan)
        for c in np.unique(cell):
            row[c] = float(np.median(db[cell == c]))
        rows.append(row)
    rows_a = np.asarray(rows)
    seen = np.isfinite(rows_a).any(axis=0)
    out = np.full(ctrl.size, np.nan)
    out[seen] = np.nanmean(rows_a[:, seen], axis=0)
    ok = np.isfinite(out)
    return np.interp(np.log(ctrl), np.log(ctrl[ok]), out[ok])


def _wind_centres(settings: list[Setting]) -> list[np.ndarray]:
    """Per recording, the windy mics' measured wind level (dB): each mic's lower
    envelope at 30-90 Hz (lines excluded by the percentile) over the calm-mic
    median."""
    out = []
    for s in settings:
        f = s.freqs
        low = (f >= BAND_HZ[0]) & (f <= 90.0)
        lev = np.percentile(s.power[:, low], 20, axis=1) / 0.2231
        ref = float(np.median(lev[s.calm]))
        w = 10.0 * np.log10(np.maximum(lev - ref, 1e-2 * ref))
        out.append(w[list(s.windy)])
    return out


def _band_fraction(
    k: np.ndarray,
    *,
    n: int,
    half_hz: float,
    sigma_nu: float,
    lam: float,
    s2A: np.ndarray | float,
    gA: np.ndarray | float,
    s2P: np.ndarray | float,
    gP: np.ndarray | float,
    step: int = 4,
) -> np.ndarray:
    """Share of a line's power inside ``|f - k f_r| <= half_hz`` (the
    :func:`line_table` band) under its window-convolved lag law:
    ``2 int_0^T R(tau) a(tau) sin(2 pi h tau) / (pi tau) dtau`` with ``a`` the
    normalised Hann autocorrelation. Clipped to ``[0.01, 1]``."""
    w = hann_window(n)
    spec = np.fft.rfft(w, n=2 * n)
    acf = np.fft.irfft(spec.real**2 + spec.imag**2, n=2 * n)[:n:step]
    tau = np.arange(acf.size) * (step / SR)
    dt = step / SR
    kern = acf / acf[0] * 2.0 * half_hz * np.sinc(2.0 * half_hz * tau) * dt
    kern[0] *= 0.5
    k = np.asarray(k, dtype=np.float64)
    s2A, gA = np.broadcast_to(s2A, k.shape), np.broadcast_to(gA, k.shape)
    s2P, gP = np.broadcast_to(s2P, k.shape), np.broadcast_to(gP, k.shape)
    out = np.empty(k.size)
    for a in range(0, k.size, 16):
        sl = slice(a, a + 16)
        rho = r_tau_v4(
            tau[None, :],
            k[sl, None],
            sigma_nu=sigma_nu,
            lam=lam,
            sigma2=s2A[sl, None],
            gamma_am_hz=gA[sl, None],
        ) * np.exp(s2P[sl, None] * np.expm1(-2.0 * math.pi * gP[sl, None] * tau[None, :]))
        out[sl] = 2.0 * (rho @ kern)
    return np.clip(out, 0.01, 1.0)


# ── the rig ──────────────────────────────────────────────────────────────────


@dataclass
class Rig:
    """The DREGON bench rig: every recording, the measured constants (pair EQ,
    ambient, prior centres) and the data-only initial values."""

    motors: tuple[str, ...]
    settings: list[Setting]
    rotors: list[int]  #: rotor index (into ``motors``) per recording
    tables: list[dict[str, np.ndarray]]
    eq_line: list[np.ndarray]  #: (M, K_j) dB, leave-one-out, at the lines
    eq_grid: list[np.ndarray]  #: (M, Fq) dB, leave-one-out, on EQ_GRID_HZ
    eq_all: np.ndarray  #: (R, M, Fq) dB, all the rotor's recordings (payload)
    k_prof: int
    motor_orders: np.ndarray  #: (n_m,) 3 p n <= k_prof
    centres: dict[str, Any]
    init: dict[str, Any]

    @property
    def n_rotors(self) -> int:
        return len(self.motors)

    @property
    def n_mics(self) -> int:
        return int(self.settings[0].power.shape[0])


def build_rig(
    motors: tuple[str, ...] = MOTORS,
    settings: list[Setting] | None = None,
    *,
    throttles: tuple[int, ...] = THROTTLES,
) -> Rig:
    """The rig of ``motors`` from the cached real recordings, or from
    ``settings`` (a planted render; ``motors`` then follows them); the pair EQ,
    the prior centres and the data-only initial values measured on them."""
    if settings is None:
        settings = [load_setting(m, t) for m in motors for t in throttles]
    else:
        motors = tuple(dict.fromkeys(s.motor for s in settings))
    rotors = [motors.index(s.motor) for s in settings]
    tables = [line_table(s, s.k_max) for s in settings]
    same = [[i for i, q in enumerate(rotors) if q == r] for r in range(len(motors))]
    eq_line, eq_grid = [], []
    for j, s in enumerate(settings):
        loo = [i for i in same[rotors[j]] if i != j] or [j]
        f_lines = s.carrier * np.arange(1, s.k_max + 1, dtype=np.float64)
        eq_line.append(pair_eq(settings, tables, f_lines, use=loo))
        eq_grid.append(pair_eq(settings, tables, EQ_GRID_HZ, use=loo))
    eq_all = np.stack([pair_eq(settings, tables, EQ_GRID_HZ, use=u) for u in same])
    k_prof = max(s.k_max for s in settings)
    motor_orders = np.arange(3 * POLE_PAIRS, k_prof + 1, 3 * POLE_PAIRS)
    k = np.arange(1, k_prof + 1)

    # C4: the measured floor of one bin at k f_j in line units, moved to f_ref
    # with the prior-centre exponent, calm-mic mean, mean over recordings
    fh = np.full((len(settings), k_prof), np.nan)
    obs = []
    for j, (s, t) in enumerate(zip(settings, tables)):
        lr10 = 10.0 * math.log10(s.speed)
        K = s.k_max
        fh[j, :K] = t["floor_bin_db"][s.calm].mean(axis=0) - PR.ALPHA_AERO[0] * lr10
        vis = np.median(t["snr_db"], axis=0) >= VISIBLE_SNR_DB
        lev = t["level_db"][s.calm].mean(axis=0) - eq_line[j][s.calm].mean(axis=0)
        obs.append(np.where(vis, lev, np.nan))
    fhat = np.nanmean(fh, axis=0)
    c4_aero = fhat - PR.C4_OFFSET_DB - PR.C4_ODD_PENALTY_DB * (k % BLADES != 0)
    c4_motor = fhat[motor_orders - 1] - PR.C4_OFFSET_DB

    ctrl_db = _floor_ctrl_init(settings, eq_grid)
    floor_mean = float(ctrl_db.mean())
    sigma_b = float(max(np.std(ctrl_db - floor_mean), 1.0))
    centres = dict(
        fhat_db=fhat,
        c4_aero=c4_aero,
        c4_motor=c4_motor,
        floor_mean_db=floor_mean,
        sigma_b_db=sigma_b,
        wind_db=_wind_centres(settings),
    )
    return Rig(
        motors=tuple(motors),
        settings=settings,
        rotors=rotors,
        tables=tables,
        eq_line=eq_line,
        eq_grid=eq_grid,
        eq_all=eq_all,
        k_prof=k_prof,
        motor_orders=motor_orders,
        centres=centres,
        init=dict(floor_ctrl_db=ctrl_db, obs_db=obs),
    )


# ── the model ────────────────────────────────────────────────────────────────


def _lag_dtft(q: Tensor, n: int) -> Tensor:
    """``sum_{|tau| < n} q(|tau|) e^{-2 pi i j tau / n}``, ``j = 0 .. n // 2``,
    for a real even lag sequence given on ``tau = 0 .. L - 1`` (``L <= n``):
    ``2 Re rfft_n(q) - q(0)``."""
    return 2.0 * torch.fft.rfft(q, n=n, dim=-1).real - q[..., 0:1]


def _ridge_z(chol: np.ndarray, sigma_b: float, ctrl_rel: np.ndarray) -> np.ndarray:
    """The ``z`` with ``sigma_b L z ~= ctrl``: ridge solution under ``N(0, I)``."""
    A = sigma_b * chol
    return np.linalg.solve(A.T @ A + np.eye(A.shape[1]), A.T @ ctrl_rel)


def _gamma_from_width(W: Any, s2: Any) -> Any:
    """``gamma_phi`` of the pedestal of width ``W`` (eq-width) at variance
    ``s2``: ``W (1 - e^{-s2}) / s2``; numpy arrays or torch tensors."""
    if isinstance(s2, Tensor):
        return W * (-torch.expm1(-s2)) / s2
    return W * (-np.expm1(-s2)) / s2


@dataclass
class _Frozen:
    """Per-recording tensors the fit never changes."""

    grid: SP.BenchGrid
    power: Tensor  # (M, F)
    ambient: Tensor  # (M, F)
    band: Tensor  # (F,) bool
    wind_shape: Tensor  # (F,)
    wind_rows: Tensor  # (W,) the rotor's windy mics
    eq_line: Tensor  # (M, K_j) dB
    eq_floor: Tensor  # (M, F) linear power gain of the floor
    log_s: float
    rotor: int
    k_max: int
    groups: list[tuple[int, np.ndarray]]
    motor_rows: list[np.ndarray]  # per group: rows of the group's orders that are motor
    motor_k: np.ndarray  # motor orders <= k_max
    floor_f: Tensor  # the floor basis' PSD-grid frequencies


class RigModel:
    """The v4 bench forward model and MAP objective of the whole rig.

    ``seed`` is the restart: 0 starts the pedestal and the shaft at the prior
    medians, ``seed >= 1`` draws them from the priors (spec § Initialisation).
    ``kappa`` is the floor push (nats per dB of each control value)."""

    def __init__(
        self,
        rig: Rig,
        *,
        device: Any = "cpu",
        seed: int = 0,
        kappa: float = 0.0,
        flat_shaft: bool = False,
    ):
        self.rig = rig
        self.seed = int(seed)
        self.kappa = float(kappa)
        #: ``flat_shaft``: one (sigma_nu, lam) for every speed, a_sigma = b_lam = 0
        #: held (gradient zeroed, no prior term)
        self.pinned: tuple[str, ...] = ("a_sigma", "b_lam") if flat_shaft else ()
        self.device = torch.device(device)
        dd: dict[str, Any] = dict(dtype=torch.float64, device=self.device)
        self.dd = dd
        K = rig.k_prof
        k = np.arange(1, K + 1)
        self.k = k
        self.cap = PR.amp_cap(k)
        self.log_cap = torch.as_tensor(np.log(self.cap), **dd)
        self.c_w = PR.width_scale(k)
        self.sigma_b = float(rig.centres["sigma_b_db"])
        self.frozen: list[_Frozen] = []
        for j, s in enumerate(rig.settings):
            grid = SP.bench_grid(n=s.n, sr=SR, device=self.device, apply_transfer=False)
            f = grid.freqs_hz
            band = (f >= BAND_HZ[0]) & (f <= BAND_HZ[1])
            groups = SP.order_groups(
                s.k_max,
                sigma_nu=LAG_SHAFT[0],
                lam=LAG_SHAFT[1],
                sr=SR,
                n=s.n,
                threshold_nats=LAG_NATS,
            )
            eq_f = _interp_rows(f, EQ_GRID_HZ, rig.eq_grid[j])
            self.frozen.append(
                _Frozen(
                    grid=grid,
                    power=torch.as_tensor(s.power, **dd),
                    ambient=torch.as_tensor(s.ambient, **dd),
                    band=torch.as_tensor(band, device=self.device),
                    wind_shape=torch.as_tensor(wind_shape(f), **dd),
                    wind_rows=torch.as_tensor(list(s.windy), device=self.device),
                    eq_line=torch.as_tensor(rig.eq_line[j], **dd),
                    eq_floor=torch.as_tensor(10.0 ** (eq_f / 10.0), **dd),
                    log_s=math.log(s.speed),
                    rotor=rig.rotors[j],
                    k_max=s.k_max,
                    groups=groups,
                    motor_rows=[np.flatnonzero(o % (3 * POLE_PAIRS) == 0) for _, o in groups],
                    motor_k=rig.motor_orders[rig.motor_orders <= s.k_max],
                    floor_f=torch.as_tensor(np.fft.rfftfreq(grid.floor.psd_len, d=1.0 / SR), **dd),
                )
            )
        self.n_cells = int(sum(int(fz.band.sum()) * rig.n_mics for fz in self.frozen))

        dyn = self._initial_dynamics(np.random.default_rng(self.seed))
        for name in self.pinned:
            dyn[name] = np.asarray(0.0)
        p_aero, p_motor = self._initial_levels(dyn)
        ctrl = rig.init["floor_ctrl_db"]
        floor_mean = float(ctrl.mean())
        chol = self.frozen[0].grid.floor.shape_chol.cpu().numpy()
        pbar_a, pbar_m = p_aero.mean(axis=0), p_motor.mean(axis=0)

        def P(v: Any) -> Tensor:
            # torch.tensor COPIES: two models of one rig never share parameters
            return torch.tensor(np.asarray(v, dtype=np.float64), **dd).requires_grad_(True)

        self.params: dict[str, Tensor] = dict(
            p_aero=P(p_aero),
            pbar_aero=P(pbar_a),
            c_aero=P((p_aero - pbar_a).mean(axis=1)),
            p_motor=P(p_motor),
            pbar_motor=P(pbar_m),
            c_motor=P((p_motor - pbar_m).mean(axis=1)),
            alpha_aero=P(PR.ALPHA_AERO[0]),
            alpha_motor=P(PR.ALPHA_MOTOR[0]),
            b0=P(PR.B0[0]),
            b1=P(PR.B1[0]),
            floor_mean_db=P(floor_mean),
            floor_z=P(_ridge_z(chol, self.sigma_b, ctrl - floor_mean)),
            **{name: P(v) for name, v in dyn.items()},
            **{f"wind_{j}": P(w) for j, w in enumerate(rig.centres["wind_db"])},
        )
        #: gradient masks of the evidence freeze (1 free, 0 held)
        self.hold: dict[str, Tensor] = {}

    # ── initialisation ──
    def _initial_dynamics(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Pedestal and shaft: prior medians (restart 0) or prior draws; the
        capped variances and width in their coordinates ``u``."""
        k, c_w = self.k, self.c_w
        mu_u, mu_g = PR.amp_log_median(k), PR.gamma_amp_log_median(k)
        log_cap = np.log(self.cap)
        if self.seed == 0:
            return dict(
                u_amp=_uncapped(np.exp(mu_u), self.cap),
                log_gA=mu_g,
                u_s2P=_uncapped(
                    np.full(k.size, math.exp(PR.PHASE_SIGMA2_LN[0])), PR.PHASE_SIGMA2_MAX
                ),
                u_W=_uncapped(HALF_NORMAL_MEDIAN * c_w, PR.W_MAX_HZ),
                log_eps=np.asarray(math.log(HALF_NORMAL_MEDIAN * PR.EPS_HALF_NORMAL)),
                a_sigma=np.asarray(PR.A_SIGMA[0]),
                log_lam_ref=np.asarray(PR.LAM_REF_LN[0]),
                b_lam=np.asarray(PR.B_LAM[0]),
            )

        def truncated(mu: Any, sd: Any, cap: Any, fold: bool = False) -> np.ndarray:
            # rejection sampling of N(mu, sd) (|.| if ``fold``) truncated at ``cap``
            mu, sd = np.broadcast_to(mu, k.shape), np.broadcast_to(sd, k.shape)

            def draw(m: np.ndarray) -> np.ndarray:
                x = mu[m] + sd[m] * rng.standard_normal(int(m.sum()))
                return np.abs(x) if fold else x

            x = draw(np.ones(k.shape, dtype=bool))
            while np.any(over := x >= cap):
                x[over] = draw(over)
            return x

        ls2A = truncated(mu_u, PR.AMP_LOG_SD, log_cap)
        ls2P = truncated(
            PR.PHASE_SIGMA2_LN[0], PR.PHASE_SIGMA2_LN[1], math.log(PR.PHASE_SIGMA2_MAX)
        )
        W = truncated(0.0, c_w, PR.W_MAX_HZ, fold=True)
        return dict(
            u_amp=_uncapped(np.exp(ls2A), self.cap),
            log_gA=mu_g + PR.GAMMA_AMP_LOG_SD * rng.standard_normal(k.size),
            u_s2P=_uncapped(np.exp(ls2P), PR.PHASE_SIGMA2_MAX),
            u_W=_uncapped(np.maximum(W, 1e-3), PR.W_MAX_HZ),
            log_eps=np.asarray(math.log(abs(PR.EPS_HALF_NORMAL * rng.standard_normal()))),
            a_sigma=np.asarray(PR.A_SIGMA[0] + PR.A_SIGMA[1] * rng.standard_normal()),
            log_lam_ref=np.asarray(PR.LAM_REF_LN[0] + PR.LAM_REF_LN[1] * rng.standard_normal()),
            b_lam=np.asarray(PR.B_LAM[0] + PR.B_LAM[1] * rng.standard_normal()),
        )

    def _initial_levels(self, dyn: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """``(p_aero (R, K), p_motor (R, n_m))`` from the data (spec
        § Initialisation steps 2-3): each visible line's calm-mic band power,
        less the calm-mic pair EQ, divided by the share of the line's power the
        band holds under the initial kernel, moved to f_ref with the prior-centre
        exponent, averaged over the rotor's recordings in dB; the C4 centre
        where no recording sees the line. At ``k = 3 p n`` the aerodynamic level
        is the mean of ``k +- B`` and the motor line takes the rest."""
        rig, K = self.rig, self.rig.k_prof
        R, mo = rig.n_rotors, rig.motor_orders
        s2A = _capped_np(dyn["u_amp"], self.cap)
        gA, s2P = np.exp(dyn["log_gA"]), _capped_np(dyn["u_s2P"], PR.PHASE_SIGMA2_MAX)
        gP = _gamma_from_width(_capped_np(dyn["u_W"], PR.W_MAX_HZ), s2P)
        num, cnt = np.zeros((R, K)), np.zeros((R, K))
        phis = []
        for j, s in enumerate(rig.settings):
            sig, lam = self._shaft_np(dyn, s.speed)
            kk = np.arange(1, s.k_max + 1)
            half = max(1.0, 3.0 * SR / s.n)
            phi = _band_fraction(
                kk, n=s.n, half_hz=half, sigma_nu=sig, lam=lam,
                s2A=s2A[: s.k_max], gA=gA[: s.k_max], s2P=s2P[: s.k_max], gP=gP[: s.k_max],
            )  # fmt: skip
            phi_m = _band_fraction(
                mo[mo <= s.k_max], n=s.n, half_hz=half, sigma_nu=sig, lam=lam,
                s2A=0.0, gA=0.0, s2P=0.0, gP=0.0,
            )  # fmt: skip
            phis.append((phi, phi_m))
            pa = rig.init["obs_db"][j] - 10.0 * np.log10(phi)
            pa = pa - PR.ALPHA_AERO[0] * 10.0 * math.log10(s.speed)
            ok = np.isfinite(pa)
            r = rig.rotors[j]
            num[r, : s.k_max] += np.where(ok, pa, 0.0)
            cnt[r, : s.k_max] += ok
        p_aero = np.where(cnt > 0, num / np.maximum(cnt, 1), rig.centres["c4_aero"][None, :])
        for km in mo:
            nb = [q for q in (km - BLADES, km + BLADES) if 1 <= q <= K]
            p_aero[:, km - 1] = p_aero[:, np.asarray(nb) - 1].mean(axis=1)
        num_m, cnt_m = np.zeros((R, mo.size)), np.zeros((R, mo.size))
        for j, s in enumerate(rig.settings):
            r, (phi, phi_m) = rig.rotors[j], phis[j]
            lr10 = 10.0 * math.log10(s.speed)
            for i, km in enumerate(mo[mo <= s.k_max]):
                o = rig.init["obs_db"][j][km - 1]
                if not np.isfinite(o):
                    continue
                aero = phi[km - 1] * 10.0 ** ((p_aero[r, km - 1] + PR.ALPHA_AERO[0] * lr10) / 10.0)
                rest = 10.0 ** (o / 10.0) - aero
                if rest > 0.0:
                    num_m[r, i] += 10.0 * math.log10(rest / phi_m[i]) - PR.ALPHA_MOTOR[0] * lr10
                    cnt_m[r, i] += 1
        p_motor = np.where(
            cnt_m > 0, num_m / np.maximum(cnt_m, 1), rig.centres["c4_motor"][None, :]
        )
        return p_aero, p_motor

    @staticmethod
    def _shaft_np(dyn: dict[str, np.ndarray], speed: float) -> tuple[float, float]:
        ls = math.log(speed)
        sig = 2.0 * math.pi * F_REF * math.exp(float(dyn["log_eps"]) + float(dyn["a_sigma"]) * ls)
        lam = math.exp(float(dyn["log_lam_ref"]) + float(dyn["b_lam"]) * ls)
        return sig, lam

    # ── pieces ──
    def s2A(self) -> Tensor:
        return _capped(self.params["u_amp"], self.log_cap)

    def gA(self) -> Tensor:
        return torch.exp(self.params["log_gA"])

    def s2P(self) -> Tensor:
        """``sigma2_phi`` per order, capped at :data:`.v4_priors.PHASE_SIGMA2_MAX`."""
        return _capped(self.params["u_s2P"], math.log(PR.PHASE_SIGMA2_MAX))

    def gP(self) -> Tensor:
        """``gamma_phi`` per order, derived from the fitted width and variance."""
        return _gamma_from_width(self.width_phase(), self.s2P())

    def width_phase(self) -> Tensor:
        """``W_phi`` per order (Hz, eq-width), capped at :data:`.v4_priors.W_MAX_HZ`."""
        return _capped(self.params["u_W"], math.log(PR.W_MAX_HZ))

    def shaft(self, j: int) -> tuple[Tensor, Tensor]:
        """``(sigma_nu rad/s, lam 1/s)`` of recording ``j`` from the rig's
        shaft law ``sigma_nu = 2 pi f_ref eps s^a``, ``lam = lam_ref s^b``."""
        p, ls = self.params, self.frozen[j].log_s
        sig = 2.0 * math.pi * F_REF * torch.exp(p["log_eps"] + p["a_sigma"] * ls)
        lam = torch.exp(p["log_lam_ref"] + p["b_lam"] * ls)
        return sig, lam

    def _floor_params(self) -> SP.FloorParams:
        p = self.params
        return SP.FloorParams(
            mean_db=p["floor_mean_db"],
            shape_z=p["floor_z"],
            tilt_db_oct=torch.zeros((), **self.dd),
            mic_floor_db=torch.zeros(1, **self.dd),
            shape_sd_db=torch.as_tensor(self.sigma_b, **self.dd),
        )

    def cbar(self) -> Tensor:
        """``(14,)`` the floor's control values (dB)."""
        basis = self.frozen[0].grid.floor
        return self.params["floor_mean_db"] + basis.shape_db(self.params["floor_z"], self.sigma_b)

    @torch.no_grad()
    def set_floor_ctrl(self, ctrl_db: np.ndarray) -> float:
        """Set the floor to the control values ``ctrl_db`` (dB): ``mean`` their
        mean, ``z`` the least-squares solve of the (linear) shape map. Returns
        the largest reproduction error (dB); a planted truth's floor."""
        basis = self.frozen[0].grid.floor
        target = np.asarray(ctrl_db, dtype=np.float64)
        mean = float(target.mean())
        z0 = torch.zeros_like(self.params["floor_z"])
        with torch.enable_grad():
            A = torch.autograd.functional.jacobian(lambda z: basis.shape_db(z, self.sigma_b), z0)
        z = np.linalg.lstsq(A.cpu().numpy(), target - mean, rcond=None)[0]
        self.params["floor_mean_db"].copy_(torch.as_tensor(mean, **self.dd))
        self.params["floor_z"].copy_(torch.as_tensor(z, **self.dd))
        return float(np.max(np.abs(self.cbar().cpu().numpy() - target)))

    def line_weights(self, j: int) -> tuple[Tensor, Tensor, np.ndarray]:
        """``(w_aero (M, K_j), w_motor (M, n_m_j), motor_k)``: each line's mean
        power at recording ``j`` per mic, ``10^{(p + 10 alpha log10 s + E)/10}``."""
        p, fz = self.params, self.frozen[j]
        r, K = fz.rotor, fz.k_max
        lr10 = 10.0 * fz.log_s / math.log(10.0)
        w_aero = 10.0 ** (
            (p["p_aero"][r, :K][None, :] + p["alpha_aero"] * lr10 + fz.eq_line) / 10.0
        )
        mk = fz.motor_k
        w_motor = 10.0 ** (
            (p["p_motor"][r, : mk.size][None, :] + p["alpha_motor"] * lr10 + fz.eq_line[:, mk - 1])
            / 10.0
        )
        return w_aero, w_motor, mk

    def _rho_aero(self, tau: Tensor, kk: Tensor, idx: Tensor, sig: Tensor, lam: Tensor) -> Tensor:
        """``(Kg, L)`` the aerodynamic lag law: shaft x amplitude OU x phase OU."""
        rho = r_tau_v4(
            tau, kk, sigma_nu=sig, lam=lam,
            sigma2=self.s2A()[idx][:, None], gamma_am_hz=self.gA()[idx][:, None],
        )  # fmt: skip
        s2P, gP = self.s2P()[idx][:, None], self.gP()[idx][:, None]
        return rho * torch.exp(s2P * torch.expm1(-2.0 * math.pi * gP * tau))

    def _cos(self, j: int, orders: np.ndarray, length: int) -> Tensor:
        fz = self.frozen[j]
        kk = torch.as_tensor(orders, **self.dd)[:, None]
        ang = torch.remainder(
            kk * self.rig.settings[j].carrier * fz.grid.tau_s[:length][None, :], 1.0
        )
        return torch.cos(2.0 * math.pi * ang)

    def floor_psd(self, j: int) -> Tensor:
        """One rotor's floor at recording ``j``'s speed on the PSD grid."""
        p, fz = self.params, self.frozen[j]
        psd = fz.grid.floor.psd(self._floor_params())
        b = p["b0"] + p["b1"] * torch.log2(torch.clamp(fz.floor_f, min=BAND_HZ[0]) / 2000.0)
        return psd * torch.exp(b * fz.log_s)

    def components(self, j: int) -> dict[str, Tensor]:
        """Recording ``j``'s modelled parts, each ``(M, F)``: ``lines``,
        ``floor`` (with the pair EQ) and ``wind``. The ambient is data."""
        p, fz = self.params, self.frozen[j]
        grid, n = fz.grid, fz.grid.n
        M = self.rig.n_mics
        sig, lam = self.shaft(j)
        w_aero, w_motor, motor_k = self.line_weights(j)
        motor_pos = {int(kk): i for i, kk in enumerate(motor_k)}
        acov = torch.zeros(M, n, **self.dd)
        for (length, orders), mrows in zip(fz.groups, fz.motor_rows):
            idx = torch.as_tensor(orders - 1, device=self.device)
            tau = grid.tau_s[:length][None, :]
            kk = torch.as_tensor(orders, **self.dd)[:, None]
            cos = self._cos(j, orders, length)
            block = w_aero[:, idx] @ (self._rho_aero(tau, kk, idx, sig, lam) * cos)  # (M, L)
            if mrows.size:
                rows = torch.as_tensor(mrows, device=self.device)
                shaft = r_tau_v4(tau, kk[rows], sigma_nu=sig, lam=lam, sigma2=0.0, gamma_am_hz=0.0)
                cols = [motor_pos[int(orders[r])] for r in mrows]
                block = block + w_motor[:, cols] @ (shaft * cos[rows])
            acov[:, :length] += block
        lines = _lag_dtft(grid.win_acf[None, :] * acov, n) / grid.window_sumsq

        basis = grid.floor
        c = torch.fft.irfft(self.floor_psd(j).to(torch.complex128), n=basis.psd_len).real
        floor = _lag_dtft(grid.win_acf[: basis.lag_len] * c[: basis.lag_len], n) / grid.window_sumsq
        lvl = torch.zeros(M, **self.dd).index_put((fz.wind_rows,), 10.0 ** (p[f"wind_{j}"] / 10.0))
        return dict(
            lines=lines,
            floor=fz.eq_floor * floor[None, :],
            wind=lvl[:, None] * fz.wind_shape[None, :],
        )

    def expected(self, j: int) -> Tensor:
        """``(M, F)`` expected periodogram of recording ``j``."""
        c = self.components(j)
        out = c["lines"] + c["floor"] + c["wind"] + self.frozen[j].ambient
        return out.clamp_min(MODEL_FLOOR)

    def whittle(self, j: int) -> Tensor:
        fz = self.frozen[j]
        m = self.expected(j)[:, fz.band]
        return (torch.log(m) + fz.power[:, fz.band] / m).sum()

    def neg_log_prior(self) -> Tensor:
        """``-log p`` of tbl-priors, plus the floor push ``-kappa sum cbar``."""
        p, rig, k = self.params, self.rig, self.k
        dd = self.dd

        def nrm(x: Tensor, mu: Any, sd: Any) -> Tensor:
            return (0.5 * ((x - torch.as_tensor(mu, **dd)) / torch.as_tensor(sd, **dd)) ** 2).sum()

        out = torch.zeros((), **dd)
        for fam, c4 in (("aero", rig.centres["c4_aero"]), ("motor", rig.centres["c4_motor"])):
            c = p[f"c_{fam}"]
            resid = p[f"p_{fam}"] - p[f"pbar_{fam}"][None, :] - (c - c.mean())[:, None]
            out = (
                out
                + nrm(p[f"pbar_{fam}"], c4, PR.C4_SD_DB)
                + nrm(resid, 0.0, PR.PROFILE_LINE_SD_DB)
                + nrm(c, 0.0, PR.PROFILE_ROTOR_SD_DB)
            )
        out = (
            out
            + nrm(p["alpha_aero"], *PR.ALPHA_AERO)
            + nrm(p["alpha_motor"], *PR.ALPHA_MOTOR)
            + nrm(p["b0"], *PR.B0)
            + nrm(p["b1"], *PR.B1)
            + nrm(p["floor_mean_db"], rig.centres["floor_mean_db"], FLOOR_MEAN_SD_DB)
            + nrm(p["floor_z"], 0.0, 1.0)
            - self.kappa * self.cbar().sum()
            # the truncated priors act on the capped quantities themselves; MAP
            # in log s2 / W needs no Jacobian (each u -> capped value is a
            # monotone bijection onto (0, cap), so both coordinates share the
            # optimum)
            + nrm(torch.log(self.s2A()), PR.amp_log_median(k), PR.AMP_LOG_SD)
            + nrm(p["log_gA"], PR.gamma_amp_log_median(k), PR.GAMMA_AMP_LOG_SD)
            + nrm(torch.log(self.s2P()), *PR.PHASE_SIGMA2_LN)
            + nrm(self.width_phase(), 0.0, self.c_w)
            + nrm(torch.exp(p["log_eps"]), 0.0, PR.EPS_HALF_NORMAL)
            + nrm(p["log_lam_ref"], *PR.LAM_REF_LN)
        )
        if "a_sigma" not in self.pinned:
            out = out + nrm(p["a_sigma"], *PR.A_SIGMA)
        if "b_lam" not in self.pinned:
            out = out + nrm(p["b_lam"], *PR.B_LAM)
        for j, w in enumerate(rig.centres["wind_db"]):
            out = out + nrm(p[f"wind_{j}"], w, WIND_SD_DB)
        return out

    def objective_and_grad(self) -> tuple[float, list[float]]:
        """Total objective (nats), gradients left in ``.grad`` (held entries
        zeroed); one recording at a time so only one recording's graph is alive."""
        for t in self.params.values():
            t.grad = None
        prior = self.neg_log_prior()
        prior.backward()
        parts = []
        for j in range(len(self.frozen)):
            w = self.whittle(j)
            w.backward()
            parts.append(float(w.detach()))
        for name, mask in self.hold.items():
            g = self.params[name].grad
            if g is not None:
                g.mul_(mask)
        for name in self.pinned:
            g = self.params[name].grad
            if g is not None:
                g.zero_()
        return float(prior.detach()) + sum(parts), parts

    def load_payload(self, pay: dict[str, Any]) -> None:
        """Set every parameter from a ``noise-v4-fit/3`` bench payload of the
        same rig geometry (the planted truth, a refit's start), and its pair
        EQ; pinned shaft exponents stay at 0."""
        pr = pay["params"]
        ped, fl, sh = pr["pedestal"], pr["floor"], pr["shaft"]
        hier = pr.get("hierarchy", {})
        K = self.rig.k_prof

        def a(v: Any) -> np.ndarray:
            return np.asarray(v, dtype=np.float64)

        p_aero, p_motor = a(pr["profile_aero_db"])[:, :K], a(pr["profile_motor_db"])
        vals: dict[str, Any] = dict(
            p_aero=p_aero,
            pbar_aero=hier.get("pbar_aero_db", p_aero.mean(axis=0)),
            c_aero=hier.get("c_aero_db", (p_aero - p_aero.mean(axis=0)).mean(axis=1)),
            p_motor=p_motor,
            pbar_motor=hier.get("pbar_motor_db", p_motor.mean(axis=0)),
            c_motor=hier.get("c_motor_db", (p_motor - p_motor.mean(axis=0)).mean(axis=1)),
            alpha_aero=pr["alpha_aero"],
            alpha_motor=pr["alpha_motor"],
            b0=fl["b0"],
            b1=fl["b1"],
            floor_mean_db=fl["mean_db"],
            floor_z=fl["shape_z"],
            u_amp=_uncapped(a(ped["sigma2_amp"])[:K], self.cap),
            log_gA=np.log(a(ped["gamma_amp_hz"])[:K]),
            # a payload at or over a cap (pre-cap fits) starts just under it;
            # values under the cap load exactly
            u_s2P=_uncapped(
                _under(a(ped["sigma2_phase"])[:K], PR.PHASE_SIGMA2_MAX), PR.PHASE_SIGMA2_MAX
            ),
            u_W=_uncapped(
                _under(
                    a(ped["width_phase_hz"])[:K]
                    if "width_phase_hz" in ped
                    else PR.pedestal_width(a(ped["gamma_phase_hz"]), a(ped["sigma2_phase"]))[:K],
                    PR.W_MAX_HZ,
                ),
                PR.W_MAX_HZ,
            ),
            log_eps=math.log(sh["eps"]),
            a_sigma=sh["a_sigma"],
            log_lam_ref=math.log(sh["lam_ref_per_s"]),
            b_lam=sh["b_lam"],
        )
        by_key = {r["key"]: r for r in pay["recordings"]}
        for j, s in enumerate(self.rig.settings):
            vals[f"wind_{j}"] = by_key[s.key]["wind_db"]
        self.sigma_b = float(fl["shape_sd_db"])
        eq = pr["mic_response"]["pair_eq"]
        grid, db = a(eq["freqs_hz"]), a(eq["db"])
        for j, (s, fz) in enumerate(zip(self.rig.settings, self.frozen)):
            rows = db[self.rig.rotors[j]]
            f_lines = s.carrier * np.arange(1, s.k_max + 1, dtype=np.float64)
            fz.eq_line = torch.as_tensor(_interp_rows(f_lines, grid, rows), **self.dd)
            e_f = _interp_rows(fz.grid.freqs_hz, grid, rows)
            fz.eq_floor = torch.as_tensor(10.0 ** (e_f / 10.0), **self.dd)
        with torch.no_grad():
            for name, v in vals.items():
                self.params[name].copy_(torch.as_tensor(np.asarray(v, dtype=np.float64)))
            for name in self.pinned:
                self.params[name].zero_()


def lag_support_check(model: RigModel, *, nats: float = 20.0) -> list[dict[str, Any]]:
    """Per recording: the orders whose FITTED shaft factor is still above
    ``e^-nats`` where the frozen lag grid cuts them (a grid that was too short)."""
    out = []
    for j, (fz, s) in enumerate(zip(model.frozen, model.rig.settings)):
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


# ── the evidence freeze ──────────────────────────────────────────────────────


@torch.no_grad()
def phase_evidence(model: RigModel, *, chunk: int = 8) -> dict[str, Any]:
    """Per order, the expected Fisher information of ``log s2P`` and
    ``log W`` (``gamma_phi`` derived from both) from the Whittle term,
    ``sum_cells (dM/dtheta / M)^2``, against the prior precision (``1/1.5^2``;
    ``2 (W/c_W)^2`` for ``log W``, the curvature of the half-normal on ``W``).
    ``hold_*`` marks the orders where the data shrink the prior variance by
    less than 10 %."""
    K = model.rig.k_prof
    fis = {"s2P": np.zeros(K), "W": np.zeros(K)}
    s2P_all, gP_all = model.s2P(), model.gP()
    two_pi = 2.0 * math.pi
    for j, fz in enumerate(model.frozen):
        grid, n = fz.grid, fz.grid.n
        inv2 = model.expected(j)[:, fz.band].reciprocal().square()  # (M, Fb)
        w_aero, _, _ = model.line_weights(j)
        sig, lam = model.shaft(j)
        for length, orders in fz.groups:
            for a in range(0, orders.size, chunk):
                o = orders[a : a + chunk]
                idx = torch.as_tensor(o - 1, device=model.device)
                tau = grid.tau_s[:length][None, :]
                kk = torch.as_tensor(o, **model.dd)[:, None]
                base = model._rho_aero(tau, kk, idx, sig, lam) * model._cos(j, o, length)
                base = base * grid.win_acf[None, :length] / grid.window_sumsq
                s2P, gP = s2P_all[idx][:, None], gP_all[idx][:, None]
                e = torch.exp(-two_pi * gP * tau)
                # d log rho_phi / d log W  (gamma ~ W at fixed s2P)
                d_w = -s2P * two_pi * gP * tau * e
                # d log rho_phi / d log s2P, through s2P and gamma(W, s2P)
                dlg = s2P / torch.expm1(s2P) - 1.0
                d_s = s2P * (e - 1.0) + d_w * dlg
                w2 = (w_aero[:, idx] ** 2).T @ inv2  # (Kc, Fb)
                for name, d in (("s2P", d_s), ("W", d_w)):
                    D = _lag_dtft(base * d, n)[:, fz.band]
                    fis[name][o - 1] += (w2 * D.square()).sum(dim=-1).cpu().numpy()
    W = model.width_phase().cpu().numpy()
    prec = {
        "s2P": np.full(K, 1.0 / PR.PHASE_SIGMA2_LN[1] ** 2),
        "W": 2.0 * (W / model.c_w) ** 2,
    }
    hold = {name: fis[name] < EVIDENCE_RATIO * prec[name] for name in fis}
    return dict(
        fisher_log_s2P=fis["s2P"].tolist(),
        fisher_log_W=fis["W"].tolist(),
        prior_prec_log_s2P=prec["s2P"].tolist(),
        prior_prec_log_W=prec["W"].tolist(),
        hold_log_s2P=hold["s2P"],
        hold_log_W=hold["W"],
    )


def _apply_hold(model: RigModel, ev: dict[str, Any]) -> dict[str, list[int]]:
    """Hold the flagged phase-pedestal parameters at their prior medians
    (``s2P = 1``; ``W = 0.674 c_W``, so ``gamma_phi`` follows ``s2P`` and the
    held width stays put) and mask their gradients."""
    hs, hw = np.asarray(ev["hold_log_s2P"]), np.asarray(ev["hold_log_W"])
    p = model.params
    u_s2P_med = _uncapped(np.exp(PR.PHASE_SIGMA2_LN[0]), PR.PHASE_SIGMA2_MAX)
    u_W_med = _uncapped(HALF_NORMAL_MEDIAN * model.c_w, PR.W_MAX_HZ)
    with torch.no_grad():
        us = p["u_s2P"].detach().cpu().numpy().copy()
        us[hs] = u_s2P_med
        uw = p["u_W"].detach().cpu().numpy().copy()
        uw[hw] = u_W_med[hw]
        p["u_s2P"].copy_(torch.as_tensor(us))
        p["u_W"].copy_(torch.as_tensor(uw))
    model.hold = {
        "u_s2P": torch.as_tensor(~hs, **model.dd),
        "u_W": torch.as_tensor(~hw, **model.dd),
    }
    k = model.k
    return dict(log_s2P=k[hs].tolist(), log_W=k[hw].tolist())


# ── convergence bookkeeping ──────────────────────────────────────────────────


def group_vectors(model: RigModel) -> dict[str, np.ndarray]:
    """The parameter groups of the convergence rule, as numpy vectors."""
    p = {k: v.detach().cpu().numpy() for k, v in model.params.items()}
    with torch.no_grad():
        ped = np.stack(
            [
                np.log(model.s2A().cpu().numpy()),
                p["log_gA"],
                np.log(model.s2P().cpu().numpy()),
                np.log(model.width_phase().cpu().numpy()),
            ]
        )
        cbar = model.cbar().cpu().numpy()
    return dict(
        levels=np.concatenate([p["p_aero"].ravel(), p["p_motor"].ravel()]),
        exponents=np.array(
            [float(p[n]) for n in ("alpha_aero", "alpha_motor", "b0", "b1", "a_sigma", "b_lam")]
        ),
        floor=cbar,
        shaft_logs=np.array([float(p["log_eps"]), float(p["log_lam_ref"])]),
        pedestal_logs=ped,
    )


def payload_group_vectors(pay: dict[str, Any]) -> dict[str, np.ndarray]:
    """:func:`group_vectors` read back from a payload."""
    pr = pay["params"]
    ped, sh, fl = pr["pedestal"], pr["shaft"], pr["floor"]
    return dict(
        levels=np.concatenate([np.ravel(pr["profile_aero_db"]), np.ravel(pr["profile_motor_db"])]),
        exponents=np.array(
            [pr["alpha_aero"], pr["alpha_motor"], fl["b0"], fl["b1"], sh["a_sigma"], sh["b_lam"]]
        ),
        floor=np.asarray(fl["ctrl_db"]),
        shaft_logs=np.log([sh["eps"], sh["lam_ref_per_s"]]),
        pedestal_logs=np.log(
            np.stack(
                [
                    ped["sigma2_amp"],
                    ped["gamma_amp_hz"],
                    ped["sigma2_phase"],
                    ped["width_phase_hz"],
                ]
            )
        ),
    )


def group_movement(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, Any]:
    """Per-group movement between two :func:`group_vectors` (spec § The
    optimiser): levels median |d| (dB), exponents max, floor control values
    max (dB), log-dynamics = max(shaft logs max, max over the four pedestal
    parameters of the median over orders); and whether each is in tolerance."""
    d = {k: np.abs(np.asarray(b[k]) - np.asarray(a[k])) for k in a}
    ped = [float(v) for v in np.median(d["pedestal_logs"], axis=1)]
    tops: dict[str, float] = dict(
        levels=float(np.median(d["levels"])),
        exponents=float(d["exponents"].max()),
        floor=float(d["floor"].max()),
        shaft_logs=float(d["shaft_logs"].max()),
    )
    tops["dynamics"] = max(tops["shaft_logs"], *ped)
    mv: dict[str, Any] = dict(tops, pedestal_logs=ped)
    mv["within"] = {g: bool(tops[g] <= tol) for g, tol in TOL_MOVE.items()}
    return mv


# ── the fit ──────────────────────────────────────────────────────────────────


def _run_lbfgs(model: RigModel, iters: int, history: int = 10) -> dict[str, Any]:
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


def _pass(model: RigModel, iters: int, n_cells: int) -> dict[str, Any]:
    """One fresh-memory L-BFGS pass with its gain per cell and movement."""
    before = group_vectors(model)
    out = _run_lbfgs(model, max(1, int(iters)))
    out["gain_per_cell"] = (out["trace"][0] - out["trace"][-1]) / n_cells
    out["movement"] = group_movement(before, group_vectors(model))
    return out


def fit_rig(
    rig: Rig,
    *,
    seed: int = 0,
    kappa: float = 0.0,
    adam_steps: int = 1500,
    adam_lr: float = 0.02,
    lbfgs_iters: int = 1200,
    device: Any = "cpu",
    flat_shaft: bool = False,
    start: dict[str, Any] | None = None,
    log: Any = print,
) -> tuple[RigModel, dict[str, Any]]:
    """MAP of the rig (spec § The optimiser): Adam; L-BFGS on every cell; the
    evidence freeze of the phase pedestal; a fresh-memory L-BFGS restart pass
    with the hold (``lbfgs_iters // 2``), converged iff it gains less than
    ``TOL_NATS_PER_CELL`` per cell AND moves no group beyond ``TOL_MOVE``; a
    final pass with the hold released (``lbfgs_iters // 4``). ``flat_shaft``
    pins ``a_sigma = b_lam = 0``; ``start`` (a payload of the same rig) replaces
    the data-only initialisation (the nested refit)."""
    tag = f"[rig s{seed}{' flat' if flat_shaft else ''}]"
    t_all = time.perf_counter()
    model = RigModel(rig, device=device, seed=seed, kappa=kappa, flat_shaft=flat_shaft)
    if start is not None:
        model.load_payload(start)
    init_s = time.perf_counter() - t_all
    t0 = time.perf_counter()
    first, _ = model.objective_and_grad()
    eval_s = time.perf_counter() - t0
    nc = model.n_cells
    log(f"{tag} start {first:.1f} nats ({nc} cells, {eval_s:.2f} s/eval, init {init_s:.1f} s)")
    adam_trace = []
    if adam_steps > 0:
        opt = torch.optim.Adam(list(model.params.values()), lr=adam_lr)
        for step in range(int(adam_steps)):
            total, _ = model.objective_and_grad()
            opt.step()
            adam_trace.append(total)
            if step % 50 == 0:
                log(f"{tag} adam {step} {total:.1f}")
    pass1 = _pass(model, lbfgs_iters, nc)
    log(f"{tag} lbfgs {pass1['trace'][-1]:.1f} ({pass1['iters']} it)")
    t0 = time.perf_counter()
    ev = phase_evidence(model)
    evidence_s = time.perf_counter() - t0
    held = _apply_hold(model, ev)
    log(f"{tag} evidence freeze: held s2P {held['log_s2P']}, W {held['log_W']}")
    restart = _pass(model, lbfgs_iters // 2, nc)
    mv = restart["movement"]
    converged = bool(restart["gain_per_cell"] < TOL_NATS_PER_CELL and all(mv["within"].values()))
    log(f"{tag} restart gain {restart['gain_per_cell']:.2e}/cell, movement {mv}")
    model.hold = {}
    release = _pass(model, lbfgs_iters // 4, nc)
    log(f"{tag} release gain {release['gain_per_cell']:.2e}/cell")
    total, parts = model.objective_and_grad()
    record = dict(
        seed=int(seed),
        kappa=float(kappa),
        flat_shaft=bool(flat_shaft),
        started_from_payload=start is not None,
        adam_steps=int(adam_steps),
        adam_lr=float(adam_lr),
        lbfgs_iters=int(lbfgs_iters),
        objective_start=first,
        objective=total,
        whittle_per_recording=parts,
        n_cells=nc,
        eval_s=eval_s,
        init_s=init_s,
        adam_trace=adam_trace,
        lbfgs_pass=pass1,
        evidence={
            **{k: v for k, v in ev.items() if not k.startswith("hold_")},
            "held_orders": held,
            "wall_s": evidence_s,
        },
        restart_pass=restart,
        restart_gain_per_cell=restart["gain_per_cell"],
        converged=converged,
        release_pass=release,
        wall_s=time.perf_counter() - t_all,
    )
    log(f"{tag} done {total:.1f} nats, converged={converged}")
    return model, record


# ── planted renders ──────────────────────────────────────────────────────────


def render_setting(truth: RigModel, j: int, rng: np.random.Generator) -> np.ndarray:
    """``(M, n)`` time-domain render of recording ``j`` of ``truth`` at its own
    geometry: the aerodynamic lines with both pedestal processes and the motor
    lines on ONE shared OU shaft (:func:`static_rig.single_rotor.synthesise`,
    same seed for both families so they share the shaft), plus Gaussian noise
    shaped to the model's floor (with the pair EQ) and wind, plus Gaussian
    noise shaped to the measured ambient.

    ``synthesise`` normalises the amplitude OU to unit mean AMPLITUDE (mean
    power ``e^{s2A}``); the model's ``10^{p/10}`` is the mean POWER, so
    ``amp2 = 2 w e^{-s2A}``."""
    from experiments.static_rig.single_rotor import OUFit, synthesise

    s = truth.rig.settings[j]
    K = s.k_max
    with torch.no_grad():
        s2A = truth.s2A().cpu().numpy()[:K]
        gA = truth.gA().cpu().numpy()[:K]
        s2P = truth.s2P().cpu().numpy()[:K]
        gP = truth.gP().cpu().numpy()[:K]
        w_aero, w_motor, motor_k = truth.line_weights(j)
        comp = truth.components(j)
        sig_t, lam_t = truth.shaft(j)
    sig, lam = float(sig_t) / (2.0 * math.pi), float(lam_t)
    shaft_seed = int(rng.integers(2**31))
    aero = OUFit(
        s=s.carrier,
        sigma_nu=sig,
        lam=lam,
        orders=np.arange(1, K + 1),
        amp2=2.0 * w_aero.cpu().numpy() * np.exp(-s2A)[None, :],
        T=s.n / SR,
    )
    x = synthesise(
        aero, SR, s.n, seed=shaft_seed, am=(s2A, gA), pm=(s2P, gP), env_rate=1000
    ).astype(np.float64)
    if motor_k.size:
        motor = OUFit(
            s=s.carrier,
            sigma_nu=sig,
            lam=lam,
            orders=motor_k,
            amp2=2.0 * w_motor.cpu().numpy(),
            T=s.n / SR,
        )
        x += synthesise(motor, SR, s.n, seed=shaft_seed, env_rate=1000).astype(np.float64)
    shaped = (comp["floor"] + comp["wind"]).cpu().numpy()
    for spec in (shaped, s.ambient):
        white = np.fft.rfft(rng.standard_normal(x.shape), axis=-1)
        x += np.fft.irfft(white * np.sqrt(np.maximum(spec, 0.0)), n=s.n, axis=-1)
    return x


def render_planted(truth: RigModel, *, seed: int = 0) -> list[Setting]:
    """:func:`render_setting` of every recording of ``truth``'s rig, as
    :class:`Setting` periodograms (the planted check's data)."""
    rng = np.random.default_rng(seed)
    out = []
    for j, s in enumerate(truth.rig.settings):
        x = render_setting(truth, j, rng)
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


# ── the payload ──────────────────────────────────────────────────────────────


def _priors_block(model: RigModel) -> dict[str, Any]:
    """Every prior value the fit used (tbl-priors)."""
    k = model.k
    return dict(
        c4=dict(offset_db=PR.C4_OFFSET_DB, odd_penalty_db=PR.C4_ODD_PENALTY_DB, sd_db=PR.C4_SD_DB),
        profile_line_sd_db=PR.PROFILE_LINE_SD_DB,
        profile_rotor_sd_db=PR.PROFILE_ROTOR_SD_DB,
        alpha_aero=list(PR.ALPHA_AERO),
        alpha_motor=list(PR.ALPHA_MOTOR),
        b0=list(PR.B0),
        b1=list(PR.B1),
        floor_mean_sd_db=FLOOR_MEAN_SD_DB,
        floor_push_kappa=model.kappa,
        amp_groups=[list(g) for g in PR.AMP_GROUPS],
        amp_log_sd=PR.AMP_LOG_SD,
        amp_cap=model.cap.tolist(),
        gamma_amp_log_median=PR.gamma_amp_log_median(k).tolist(),
        gamma_amp_log_sd=PR.GAMMA_AMP_LOG_SD,
        phase_sigma2_ln=list(PR.PHASE_SIGMA2_LN),
        phase_sigma2_max=PR.PHASE_SIGMA2_MAX,
        c_w=PR.C_W,
        w_max_hz=PR.W_MAX_HZ,
        eps_half_normal=PR.EPS_HALF_NORMAL,
        a_sigma=list(PR.A_SIGMA),
        b_lam=list(PR.B_LAM),
        lam_ref_ln=list(PR.LAM_REF_LN),
        wind_sd_db=WIND_SD_DB,
        evidence_ratio=EVIDENCE_RATIO,
        tolerances=dict(nats_per_cell=TOL_NATS_PER_CELL, **TOL_MOVE),
    )


def payload(model: RigModel, record: dict[str, Any], *, git: str | None = None) -> dict[str, Any]:
    """The ``noise-v4-fit/3`` bench payload of the DREGON rig."""
    p = {k: v.detach().cpu().numpy() for k, v in model.params.items()}
    rig = model.rig
    with torch.no_grad():
        s2A, gA = model.s2A().cpu().numpy(), model.gA().cpu().numpy()
        s2P, gP = model.s2P().cpu().numpy(), model.gP().cpu().numpy()
        W = model.width_phase().cpu().numpy()
        cbar = model.cbar().cpu().numpy()
    recordings = []
    for j, s in enumerate(rig.settings):
        sig_t, lam_t = model.shaft(j)
        recordings.append(
            dict(
                key=s.key,
                rotor=rig.rotors[j],
                motor=s.motor,
                throttle=s.throttle,
                carrier_rev_s=s.carrier,
                speed_rel=s.speed,
                span_s=list(s.span_s),
                n_samples=s.n,
                k_max=s.k_max,
                sigma_nu_rad_s=float(sig_t),
                lam_per_s=float(lam_t),
                wind_mics=list(s.windy),
                wind_db=p[f"wind_{j}"].tolist(),
                ambient=dict(
                    ref=str(CACHE_DIR / f"{s.key}.npz"),
                    runs_s=[list(r) for r in s.ambient_runs_s],
                    rule="Welch of the rotor-stopped runs (bench_v4.ambient_runs)",
                ),
                whittle_nats=record["whittle_per_recording"][j],
                n_cells=int(model.frozen[j].band.sum()) * rig.n_mics,
            )
        )
    floor = model.frozen[0].grid.floor
    held = record["evidence"]["held_orders"]
    return dict(
        schema=PR.FIT_SCHEMA,
        kind="bench",
        rig="dregon",
        n_rotors=rig.n_rotors,
        n_mics=rig.n_mics,
        blades=BLADES,
        pole_pairs=POLE_PAIRS,
        f_ref_rev_s=F_REF,
        sr=SR,
        band_hz=list(BAND_HZ),
        motors=list(rig.motors),
        params=dict(
            profile_aero_db=p["p_aero"].tolist(),
            motor_orders=rig.motor_orders.tolist(),
            profile_motor_db=p["p_motor"].tolist(),
            alpha_aero=float(p["alpha_aero"]),
            alpha_motor=float(p["alpha_motor"]),
            hierarchy=dict(
                pbar_aero_db=p["pbar_aero"].tolist(),
                c_aero_db=(p["c_aero"] - p["c_aero"].mean()).tolist(),
                pbar_motor_db=p["pbar_motor"].tolist(),
                c_motor_db=(p["c_motor"] - p["c_motor"].mean()).tolist(),
            ),
            pedestal=dict(
                sigma2_amp=s2A.tolist(),
                gamma_amp_hz=gA.tolist(),
                sigma2_phase=s2P.tolist(),
                gamma_phase_hz=gP.tolist(),
                width_phase_hz=W.tolist(),
                held_by_evidence=held,
            ),
            shaft=dict(
                eps=float(np.exp(p["log_eps"])),
                a_sigma=float(p["a_sigma"]),
                lam_ref_per_s=float(np.exp(p["log_lam_ref"])),
                b_lam=float(p["b_lam"]),
                f_ref_rev_s=F_REF,
                pinned=list(model.pinned),
            ),
            floor=dict(
                mean_db=float(p["floor_mean_db"]),
                shape_z=p["floor_z"].tolist(),
                shape_sd_db=float(model.sigma_b),
                ctrl_hz=floor.ctrl_hz.tolist(),
                ctrl_db=cbar.tolist(),
                b0=float(p["b0"]),
                b1=float(p["b1"]),
                static_rel=0.0,
            ),
            mic_response=dict(
                kind="pair_eq",
                pair_eq=dict(freqs_hz=EQ_GRID_HZ.tolist(), db=rig.eq_all.tolist()),
            ),
            wind=dict(kind="per_recording", shape=wind_shape_spec()),
        ),
        recordings=recordings,
        objective=dict(
            total_nats=record["objective"],
            whittle_nats=float(sum(record["whittle_per_recording"])),
            n_cells=model.n_cells,
        ),
        optimiser={k: v for k, v in record.items() if k != "whittle_per_recording"},
        priors=_priors_block(model),
        centres=dict(
            fhat_db=np.asarray(rig.centres["fhat_db"]).tolist(),
            c4_aero_db=np.asarray(rig.centres["c4_aero"]).tolist(),
            c4_motor_db=np.asarray(rig.centres["c4_motor"]).tolist(),
            floor_mean_db=rig.centres["floor_mean_db"],
            sigma_b_db=rig.centres["sigma_b_db"],
            wind_db=[np.asarray(w).tolist() for w in rig.centres["wind_db"]],
        ),
        lag_support=lag_support_check(model),
        git=git,
    )
