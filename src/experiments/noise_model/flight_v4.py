"""Noise model v4, procedure B: the free-flight rig fit of Michael's drone (MD2).

Authority: ``docs/explainers/noise-model-v4.qmd`` § Procedure B, § Fitting,
§ The model, § The priors. The fit keeps the nv3r4 flight procedure
(``docs/experiments/noise-model-v3.md`` § Round 4; mode ``flight_v3`` with
``--priors v4``) and changes only what v4 changes.

KEPT FROM nv3r4, REUSED NOT COPIED. The pool is one :class:`.model.SupportBatch`
from :func:`.model.flight_batch` (16 kHz, 2048/512 periodic Hann, frames thinned
by 4, composite exposure weights, the channels divided by the measured array
response :class:`.model.ChannelGains` ``band="transfer"``); Adam minibatches are
:func:`.model.batch_slice`, the L-BFGS subset is
:func:`.model.stratified_frames`, the Whittle term is :func:`.model.whittle_risk`
summed over :func:`.model.frame_chunks`; the floor is the nv3r4
:class:`.spectrum.FloorBasis` (14-point spline, measured ``sigma_B``, ``mu`` a
measured constant) through :func:`.spectrum.floor_frames_from_autocov`; the
L-BFGS is :class:`.lbfgs.LBFGS` (strong Wolfe); no wander latents.

WHY A MODULE OF ITS OWN. v4 adds blocks the v2/v3 parameter set
(:class:`.spectrum.V2Params`) and its Pyro site builder cannot express: two
line families, the hierarchical rotor profiles, the two-process pedestal in
place of the Lorentzian ``gamma_rk``, the floor with ``b(f)`` and the
rotor-microphone geometry term. Threading them through ``model.sample_params``
would touch every v2/v3 mode; the forward model here is ~150 lines over the
shared kernels, and the MAP objective is plain torch, as in :mod:`.bench_v4`.

THE FORWARD MODEL (mic ``m``, frame ``n``)::

    E I_m = T(f) [ sum_r g_rm L_r + F_1 + s F_s ],   g_rm = 10^{Delta G_rm / 10}

* ``L_r``: rotor ``r``'s aerodynamic lines at every ``k`` (power
  ``2 10^{p^a_rk/10}``, envelope ``s_r(t)^{alpha^a/2}``, lag law shaft x
  amplitude-OU x phase-OU) plus its motor lines at ``k = 3 p n`` (``p^m_rn``,
  ``s^{alpha^m/2}``, shaft only), summed in the lag domain through each order's
  unit-atom autocorrelation exactly as :func:`.spectrum._line_sum_unit_autocorr`;
* ``F_1``: ONE floor for the rig (decision 2026-10-21, payload
  ``floor.floor_rule = "one_floor_mean_speed"``), driven by the rotor-MEAN
  speed ``sbar(t) = mean_r s_r(t)``: the window kernel of the atom
  ``w(t) sbar(t)^{b0/2}`` on the floor autocovariance, times the frame's
  ``sbar_n^{b1 log2(max(f, 30)/2 kHz)}`` (``sbar_n`` the window's geometric
  mean of ``sbar``); no geometry weight (``Delta G`` is on the lines only).
  A sum of four rotor floors let ``b`` select a rotor (b0 ~ 34 put 97 % of the
  cruise floor on one rotor); ``F_s`` is the same kernel with a unit atom
  (the static share ``s``);
* ``T(f)``: the render chain transfer.

The pedestal is HELD: the amplitude OU from MD2's own block-wander measurement,
capped at :func:`.v4_priors.amp_cap`; the phase OU at the prior medians. The
shaft has ``lam`` pinned at :data:`.v4_priors.FLIGHT_LAM` (``b_lam = 0``) and
the speed law ``sigma_nu = sigma_ref s^{a_sigma}`` per rotor and frame (``s``
the rotor's frame speed over ``f_ref``, the window-power-weighted geometric
mean), ``sigma_ref ~`` :data:`.v4_priors.FLIGHT_SIGMA_NU_LN`, ``a_sigma ~``
:data:`.v4_priors.FLIGHT_A_SIGMA`: nv3's per-regime fits gave standby 0.35
rad/s against cruise 3.8, so a flat ``sigma_nu`` fits neither regime.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from data_processing.noise_model.ou import integrated_ou_increment_var
from data_processing.noise_model.v4 import FLOOR_RULE_ONE
from data_processing.sources.michaels import ROTOR_ORDER, get_geometry
from experiments.stochastic_fit.revised_phase import (
    MODEL_FLOOR,
    SPEED_FLOOR_RPS,
    floor_geometry,
)

from . import model as MD
from . import spectrum as SP
from . import v4_priors as V4
from .lbfgs import LBFGS

__all__ = [
    "FlightData",
    "FlightMeasured",
    "FlightV4Model",
    "OptimV4",
    "FLIGHT_V4_MODE",
    "MD2_BLADES",
    "MD2_POLE_PAIRS",
    "STATIC_SHARE_LN",
    "fit_flight",
    "flight_pedestal",
    "geometry_db",
    "joint_k_max",
    "measure",
    "payload",
    "prepare",
    "reduce_payloads",
]

FLIGHT_V4_MODE = "flight_v4"
#: DJI Matrice 100: two blades, 14-pole (seven pole-pair) motors.
MD2_BLADES = 2
MD2_POLE_PAIRS = 7
#: The static share ``s`` of the flight floor (spec tbl-priors, nv3r4).
STATIC_SHARE_LN = (math.log(2.5e-3), 1.0)
#: ``sigma_B`` is held at least this (nv3r4 ``PriorsV3.floor_shape_sd_min_db``).
FLOOR_SHAPE_SD_MIN_DB = 1.0
#: ``b(f)`` reads the frequency no lower than the band edge (contract).
B_F_LO_HZ = SP.BAND_F_MIN
B_F_REF_HZ = V4.B_F_REF_HZ
#: The half-normal's median as a multiple of its scale.
HALFNORMAL_MEDIAN = 0.6744897501960817
#: Log-space coordinates are held inside +-LOG_BOX (nv3r4 ``fit.LOG_BOX``).
LOG_BOX = 60.0
#: Convergence tolerances (spec § The optimiser, step 5).
TOL_NATS_PER_CELL = 1e-4
MOVE_TOL = dict(levels_db=0.1, exponents=0.02, floor_ctrl_db=0.1, log_sigma_nu=0.05)
DEFAULT_WANDER = "results/noise_v3/wander/michaels.json"


def _t(v: Any, ref: Tensor) -> Tensor:
    return torch.as_tensor(np.asarray(v, dtype=np.float64), dtype=torch.float64, device=ref.device)


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


# ── constants of the fit (spec step B1) ─────────────────────────────────────


def geometry_db(mics: list[int] | None = None) -> np.ndarray:
    """``(R, M)`` ``Delta G_rm = G_rm - mean_r G_rm``, ``G = -20 log10 d_rm``.

    Rotor order is :data:`data_processing.sources.michaels.ROTOR_ORDER`, which
    is the label's (the first four ``Motor:Speed:*`` CSV columns)."""
    mic_pos, rotor_pos = get_geometry()
    d = np.linalg.norm(rotor_pos[:, None, :] - mic_pos[None, :, :], axis=-1)  # (R, 8)
    g = -20.0 * np.log10(d)
    dg = g - g.mean(axis=0, keepdims=True)
    return dg if mics is None else dg[:, np.asarray(mics, dtype=np.int64)]


def _by_order(k: np.ndarray, block: dict[str, Any] | None, key: str, pooled: float) -> np.ndarray:
    """A wander value per order from its ``k_edges`` groups (pooled where absent)."""
    if not block:
        return np.full(k.shape, float(pooled))
    edges = np.asarray(block["k_edges"], dtype=np.int64)
    vals = np.asarray(block[key], dtype=np.float64)
    idx = np.clip(np.searchsorted(edges, k, side="right") - 1, 0, vals.size - 1)
    return vals[idx]


def flight_pedestal(k_max: int, *, wander: dict[str, Any]) -> dict[str, np.ndarray]:
    """The HELD pedestal per order ``1..k_max`` (spec § Procedure B).

    Amplitude: ``sigma2 = (sigma_v ln10 / 20)^2`` capped at
    :func:`.v4_priors.amp_cap`, ``gamma = 1 / (2 pi tau_v)`` from the pool's own
    block wander (per order group where measured). Phase: the prior medians,
    ``sigma2 = exp(PHASE_SIGMA2_LN[0])`` and ``gamma`` solving
    :func:`.v4_priors.pedestal_width` at the half-normal median of ``W``."""
    k = np.arange(1, k_max + 1, dtype=np.int64)
    sig_db = _by_order(k, wander.get("sigma_v_db_by_order"), "sigma_db", wander["sigma_v_db"])
    tau_s = _by_order(k, wander.get("tau_v_s_by_order"), "tau_s", wander["tau_v_s"])
    s2a_raw = (sig_db * math.log(10.0) / 20.0) ** 2
    s2a = np.minimum(s2a_raw, V4.amp_cap(k))
    ga = 1.0 / (2.0 * math.pi * tau_s)
    s2p = np.full(k.shape, math.exp(V4.PHASE_SIGMA2_LN[0]))
    w_med = V4.width_scale(k) * HALFNORMAL_MEDIAN
    gp = w_med * (-np.expm1(-s2p)) / s2p
    return dict(
        sigma2_amp=s2a,
        gamma_amp_hz=ga,
        sigma2_phase=s2p,
        gamma_phase_hz=gp,
        sigma2_amp_uncapped=s2a_raw,
    )


def joint_k_max(batch: MD.SupportBatch, *, k_cap: int) -> int:
    """The order count of a pool that joins regimes: the highest order below
    the Nyquist in ANY window (each window's own fastest carrier), capped.

    :func:`.model.flight_batch` reads the pool's fastest carrier, which on a
    standby + cruise pool would drop every standby order above ~81 (the MD2
    motor line k = 84 among them). Orders above a fast window's Nyquist fall
    outside the analysis bins there and contribute nothing but skirts."""
    assert batch.rate_work is not None and batch.frame_window is not None
    fw = np.asarray(batch.frame_window)
    rw = batch.rate_work.detach()
    peak_frame = rw.amax(dim=(0, 2)).cpu().numpy()  # (N,)
    k_w = [
        SP.k_max_for_carrier(float(peak_frame[fw == w].max()), int(batch.grid.sr))
        for w in np.unique(fw)
    ]
    k = min(int(k_cap), int(max(k_w)))
    top = float(peak_frame.max()) * k
    grid = batch.grid
    assert isinstance(grid, SP.FlightGrid)
    if top >= 0.5 * float(grid.sr_work):
        raise ValueError(
            f"order {k} of the fastest carrier sits at {top:.0f} Hz, over the work Nyquist "
            f"{0.5 * grid.sr_work:.0f} Hz: lower --k-cap or raise the work rate"
        )
    return k


@dataclass
class FlightData:
    """The pool and every constant the fit holds (spec step B1)."""

    batch: MD.SupportBatch
    k_max: int
    motor_orders: np.ndarray
    geometry_db: np.ndarray  # (R, M)
    pedestal: dict[str, np.ndarray]
    wander_path: str
    mics: list[int]
    blades: int = MD2_BLADES
    pole_pairs: int = MD2_POLE_PAIRS


def prepare(
    batch: MD.SupportBatch,
    *,
    k_cap: int,
    mics: list[int] | None = None,
    wander_path: str = DEFAULT_WANDER,
    blades: int = MD2_BLADES,
    pole_pairs: int = MD2_POLE_PAIRS,
) -> FlightData:
    """The flight pool with the joint order count, the geometry and the pedestal."""
    if batch.n_rotors != len(ROTOR_ORDER):
        raise ValueError(f"MD2 has {len(ROTOR_ORDER)} rotors, the labels carry {batch.n_rotors}")
    k = joint_k_max(batch, k_cap=k_cap)
    used = list(range(8)) if mics is None else [int(m) for m in mics]
    if len(used) != batch.n_mics:
        raise ValueError(f"{len(used)} mic indices for a {batch.n_mics}-mic batch")
    wander = json.loads(Path(wander_path).read_text())
    step = 3 * int(pole_pairs)
    return FlightData(
        batch=replace(batch, k_max=k),
        k_max=k,
        motor_orders=np.arange(step, k + 1, step, dtype=np.int64),
        geometry_db=geometry_db(used),
        pedestal=flight_pedestal(k, wander=wander),
        wander_path=str(wander_path),
        mics=used,
        blades=int(blades),
        pole_pairs=int(pole_pairs),
    )


# ── the kernel pieces ───────────────────────────────────────────────────────


def _phase_speed(grid: SP.FlightGrid, rate: Tensor) -> tuple[Tensor, Tensor]:
    """``(phase, speed)`` of a chunk's work-grid carrier: the window-centred
    phase integral (:func:`.spectrum._line_blocks`) and ``rate / f_ref``."""
    with torch.no_grad():
        n_work = int(rate.shape[-1])
        phase = (2.0 * math.pi / float(grid.sr_work)) * torch.cumsum(rate, dim=-1)
        phase = phase - phase[..., n_work // 2 : n_work // 2 + 1]
        speed = rate.clamp_min(SPEED_FLOOR_RPS) / SP.AMP_RPS_REF
    return phase, speed


def _unit_autocorr(
    phase: Tensor, env: Tensor, k: Tensor, n: int, log_half: Tensor | None
) -> tuple[Tensor, Tensor | None]:
    """``(R, k, n_c, n)`` positive-lag autocorrelation of each order's unit
    atom ``env e^{i k phi}`` and, with ``log_half``, its derivative in the
    exponent (:func:`.spectrum._line_sum_unit_autocorr`). No gradient."""
    with torch.no_grad():
        kph = torch.remainder(k[None, :, None, None] * phase[:, None], 2.0 * math.pi)
        unit = torch.polar(env[:, None].expand_as(kph), kph)
        del kph
        spec = torch.fft.fft(unit, n=2 * n, dim=-1)
        g = torch.fft.ihfft(spec.real**2 + spec.imag**2, dim=-1)[..., :n]
        dg = None
        if log_half is not None:
            d_spec = torch.fft.fft(unit * log_half[:, None], n=2 * n, dim=-1)
            cross = 2.0 * (d_spec.real * spec.real + d_spec.imag * spec.imag)
            del d_spec
            dg = torch.fft.ihfft(cross, dim=-1)[..., :n]
    return g, dg


def _family_lag_sum(
    grid: SP.FlightGrid,
    phase: Tensor,
    speed: Tensor,
    orders: Tensor,
    power: Tensor,
    sigma2: Tensor,
    v_unit: Tensor,
    ped: Tensor | None,
    alpha: Tensor,
    chunk: int,
) -> Tensor:
    """``(R, n_c, n)`` lag-domain sum of one line family: ``sum_k power_rk
    rho_rkn(tau) G_rkn(tau)`` with ``G`` the unit atom ``w s^{alpha/2} e^{i k phi}``'s
    autocorrelation, ``alpha``'s gradient through the tangent ``dG/d alpha``.
    ``rho`` is built per order chunk: the shaft factor of each rotor and frame
    (:func:`_shaft_frames`, ``sigma2`` ``(R, n_c)``) times the held pedestal
    ``ped`` ``(K, n)`` (``None``: the motor family, shaft only)."""
    n = int(grid.n_fft_work)
    fitted = bool(alpha.requires_grad)
    with torch.no_grad():
        env = torch.sqrt(speed ** alpha.detach()) * grid.window_work
        log_half = 0.5 * torch.log(speed) if fitted else None
    r, n_c = int(speed.shape[0]), int(speed.shape[1])
    lag = torch.zeros(r, n_c, n, dtype=torch.complex128, device=speed.device)
    tangent = torch.zeros_like(lag) if fitted else None
    for c0 in range(0, int(orders.numel()), max(1, int(chunk))):
        c1 = min(c0 + max(1, int(chunk)), int(orders.numel()))
        g, dg = _unit_autocorr(phase, env, orders[c0:c1], n, log_half)
        rho = _shaft_frames(orders[c0:c1], sigma2, v_unit)
        if ped is not None:
            rho = rho * ped[c0:c1, None, :]
        weight = power[:, c0:c1, None, None] * rho
        lag = lag + (g * weight).sum(dim=1)
        if tangent is not None and dg is not None:
            with torch.no_grad():
                tangent += (dg * weight).sum(dim=1)
    if tangent is not None:
        lag = lag + (alpha - alpha.detach()) * tangent
    return lag


def _lag_to_bins(grid: SP.FlightGrid, lag: Tensor) -> Tensor:
    """``(..., F)`` analysis-bin periodogram of a Hermitian lag sum
    (:func:`.spectrum._line_sum_unit_autocorr`'s fold and ``hfft``), with
    ``grid_power_factor`` applied."""
    n = int(grid.n_fft_work)
    half = n // 2
    folded = torch.cat(
        (lag[..., :1], lag[..., 1 : half + 1] + lag.flip(-1)[..., :half].conj()), dim=-1
    )
    p_z = torch.fft.hfft(folded, n=n, dim=-1)
    pos = torch.arange(0, grid.n_fft // 2 + 1, device=p_z.device)
    neg = torch.remainder(n - pos, n)
    out = 0.25 * (p_z.index_select(-1, pos) + p_z.index_select(-1, neg))
    return out / float(grid.window_work_sumsq) * grid.grid_power_factor


def _atom_autocorr(atom: Tensor, n: int) -> Tensor:
    """``(..., 2 n)`` complex autocorrelation of a real window atom
    (:func:`.spectrum.floor_unit_autocorr`'s transform)."""
    spec = torch.fft.fft(atom.to(torch.complex128), n=2 * n, dim=-1)
    return torch.fft.ifft(spec.real**2 + spec.imag**2, dim=-1)


def _log_sbar(grid: SP.FlightGrid, speed: Tensor) -> Tensor:
    """``(R, n_c)`` window-power-weighted mean of ``log s`` over each frame."""
    w2 = grid.window_work**2
    return (torch.log(speed) * w2).sum(dim=-1) / w2.sum()


def _v_unit(grid: SP.FlightGrid, lam: float) -> Tensor:
    """``(n,)`` the shaft structure function ``V_theta`` on the work lags at
    ``sigma_nu = 1`` (``V_theta`` is linear in ``sigma_nu^2``); ``lam`` pinned."""
    return integrated_ou_increment_var(grid.tau_s_work, lam=float(lam), sigma=1.0)


def _shaft_frames(k: Tensor, sigma2: Tensor, v_unit: Tensor) -> Tensor:
    """``(R, K, n_c, n)`` the shaft factor ``exp(-k^2 sigma2_rn V_1(tau) / 2)``."""
    return torch.exp(-0.5 * k[None, :, None, None] ** 2 * sigma2[:, None, :, None] * v_unit)


def _sigma2_frames(
    grid: SP.FlightGrid, speed: Tensor, log_sigma_ref: Tensor, a_sigma: Tensor
) -> Tensor:
    """``(R, n_c)`` ``sigma_nu^2`` of each rotor and frame from the shaft law
    ``sigma_nu = sigma_ref sbar_rn^{a_sigma}`` (``sbar_rn`` the frame's speed / f_ref)."""
    with torch.no_grad():
        log_sbar = _log_sbar(grid, speed)
    return torch.exp(2.0 * (log_sigma_ref + a_sigma * log_sbar))


def pedestal_lag(grid: SP.FlightGrid, ped: dict[str, np.ndarray]) -> Tensor:
    """``(K, n)`` ``exp[s2A (e^{-2 pi gA tau} - 1) + s2P (e^{-2 pi gP tau} - 1)]``."""
    ref = grid.window_work
    tau = grid.tau_s_work[None, :]
    s2a, ga, s2p, gp = (
        _t(ped[key], ref)[:, None]
        for key in ("sigma2_amp", "gamma_amp_hz", "sigma2_phase", "gamma_phase_hz")
    )
    return torch.exp(
        s2a * torch.expm1(-2.0 * math.pi * ga * tau) + s2p * torch.expm1(-2.0 * math.pi * gp * tau)
    )


def _floor_one(
    grid: SP.FlightGrid,
    speed: Tensor,
    c: Tensor,
    b0: Tensor,
    b1: Tensor,
    l_f: Tensor,
) -> Tensor:
    """``(n_c, F)`` the rig's ONE floor at unit level: the kernel of the atom
    ``w sbar(t)^{b0/2}`` (``sbar`` the rotor-mean speed ``mean_r s_r(t)``) on
    the floor autocovariance ``c``, times the frame's ``sbar_n^{b1 l(f)}``."""
    mean_speed = speed.mean(dim=0)  # (n_c, n_work)
    atom = grid.window_work * torch.sqrt(mean_speed**b0)
    g1 = _atom_autocorr(atom, int(grid.n_fft_work))
    fr = SP.floor_frames_from_autocov(grid, g1, c)
    return fr * torch.exp(b1 * _log_sbar(grid, mean_speed)[..., None] * l_f)


# ── measurement and initialisation (spec step B1, § Initialisation) ────────


@dataclass
class FlightMeasured:
    """What the pool measured, the prior centres it sets, and the start."""

    floor_mean_db: float
    floor_shape_sd_db: float
    floor_ctrl_db: np.ndarray  # (14,) about mu
    floor_line_db: np.ndarray  # (R, K) F-hat per line, line units
    floor_order_db: np.ndarray  # (K,) F-hat_k = mean over rotors
    init: dict[str, np.ndarray]
    record: dict[str, Any]


def _floor_c(grid: SP.FlightGrid, mu: Any, z: Any, sd: Any, ref: Tensor) -> Tensor:
    fp = SP.FloorParams(
        mean_db=_t(mu, ref) if not isinstance(mu, Tensor) else mu,
        shape_z=_t(z, ref) if not isinstance(z, Tensor) else z,
        tilt_db_oct=torch.zeros((), dtype=torch.float64, device=ref.device),
        mic_floor_db=torch.zeros(1, dtype=torch.float64, device=ref.device),
        shape_sd_db=_t(sd, ref) if not isinstance(sd, Tensor) else sd,
    )
    return grid.floor.autocovariance(fp)


def _ridge_z(grid: SP.FlightGrid, sigma_b: float, ctrl_rel: np.ndarray) -> np.ndarray:
    chol = grid.floor.shape_chol.detach().cpu().numpy()
    a = sigma_b * chol
    return np.linalg.solve(a.T @ a + np.eye(a.shape[1]), a.T @ np.asarray(ctrl_rel))


def _frame_slices(n: int, step: int | None) -> list[slice]:
    s = n if step is None or int(step) <= 0 else int(step)
    return [slice(i, min(i + s, n)) for i in range(0, n, s)]


def measure(data: FlightData, *, chunk_orders: int = 32) -> FlightMeasured:
    """Spec § Initialisation steps 1-3 and step B1's ``sigma_B`` and ``F-hat``.

    1. Floor: per frame and mic, the minimum of the periodogram in each run of
       bins between neighbouring lines of all rotors (lines at ``k sbar_rn``,
       +-1 bin excluded), times the run length (the minimum of ``m`` iid
       exponential cells has mean ``1/m`` of theirs), divided by the model's
       unit floor at the run centre times ``sbar_n^{b_init} + s_init`` (``sbar_n``
       the frame's rotor-mean speed, the one-floor rule);
       averaged per bin over every periodogram; the median per spline cell
       gives the control values. ``mu`` is their mean (a measured constant, as
       in nv3r4), ``sigma_B`` their rms about it (at least 1 dB), ``z`` the
       ridge solution.
    2. ``F-hat_rk``: the mean over in-band frames of the floor at the line's
       bin over the line's unit kernel there (power 0 dB at f_ref, ``alpha^a``
       at its prior centre, the held pedestal, the shaft law at its prior centres,
       mic-mean geometry). Aerodynamic start: ``sum (I - floor) / sum unit``
       over the frames, or the prior centre where it is not positive.
    3. Motor start at ``k = 3 p n``: the aerodynamic level is the mean of
       those at ``k +- B``; the motor line takes the rest of the measured
       power, or its prior centre.
    """
    batch = data.batch
    grid = batch.grid
    assert isinstance(grid, SP.FlightGrid) and batch.rate_work is not None
    ref = grid.window_work
    dev = ref.device
    r_n, n_frames, kk = batch.n_rotors, int(batch.power.shape[1]), data.k_max
    f = np.asarray(grid.freqs_hz, dtype=np.float64)
    df = float(f[1] - f[0])
    band = batch.band.detach().cpu().numpy()
    f_top = float(f[band].max())
    n_bins = f.size
    gain = 10.0 ** (data.geometry_db / 10.0)  # (R, M)
    gbar = gain.mean(axis=1)  # (R,)
    l_f_np = np.log2(np.maximum(f, B_F_LO_HZ) / B_F_REF_HZ)
    b_init = V4.B0[0] + V4.B1[0] * l_f_np
    s_init = math.exp(STATIC_SHARE_LN[0])
    alpha_a0, alpha_m0 = V4.ALPHA_AERO[0], V4.ALPHA_MOTOR[0]
    sigma0 = math.exp(V4.FLIGHT_SIGMA_NU_LN[0])
    log_sig_t = torch.tensor(V4.FLIGHT_SIGMA_NU_LN[0], dtype=torch.float64, device=dev)
    a_sig_t = torch.tensor(V4.FLIGHT_A_SIGMA[0], dtype=torch.float64, device=dev)
    k_a = torch.arange(1, kk + 1, dtype=torch.float64, device=dev)
    k_m = _t(data.motor_orders, ref)
    n_work = int(grid.n_fft_work)
    g1_static = _atom_autocorr(grid.window_work, n_work)

    with torch.no_grad():
        u0 = SP.floor_frames_from_autocov(
            grid, g1_static, _floor_c(grid, 0.0, np.zeros(14), 1.0, ref)
        )
        u0 = (u0 * grid.transfer_power).cpu().numpy()  # (F,)
        log_sbar = np.empty((r_n, n_frames))
        log_sbar_mean = np.empty(n_frames)
        for sl in _frame_slices(n_frames, batch.chunk_frames):
            _ph, sp = _phase_speed(grid, batch.rate_work[:, sl])
            log_sbar[:, sl] = _log_sbar(grid, sp).cpu().numpy()
            log_sbar_mean[sl] = _log_sbar(grid, sp.mean(dim=0)).cpu().numpy()
    sbar = np.exp(log_sbar)  # (R, N)
    sbar_mean = np.exp(log_sbar_mean)  # (N,)
    obs = batch.power.detach().cpu().numpy()  # (M, N, F)
    orders = np.arange(1, kk + 1, dtype=np.float64)

    # ── step 1: the floor from the gap minima ──
    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for n in range(n_frames):
        lf = np.outer(sbar[:, n] * SP.AMP_RPS_REF, orders).ravel()
        lf = lf[lf <= f_top + df]
        excl = np.zeros(n_bins, dtype=bool)
        idx = np.rint(lf / df).astype(np.int64)
        for d in (-1, 0, 1):
            excl[np.clip(idx + d, 0, n_bins - 1)] = True
        allowed = np.flatnonzero(band & ~excl)
        if allowed.size == 0:
            continue
        starts = np.flatnonzero(np.diff(allowed, prepend=-10) != 1)
        lengths = np.diff(np.append(starts, allowed.size))
        minima = np.minimum.reduceat(obs[:, n, allowed], starts, axis=1)  # (M, runs)
        centre = allowed[starts + lengths // 2]
        factor = sbar_mean[n] ** b_init[centre] + s_init  # (runs,)
        ratio = minima * lengths[None, :] / (u0[centre] * factor)[None, :]
        np.add.at(sums, centre, ratio.sum(axis=0))
        np.add.at(counts, centre, float(ratio.shape[0]))
    have = counts > 0
    ratio_db = 10.0 * np.log10(np.maximum(sums[have] / counts[have], 1e-300))
    ctrl_hz = np.asarray(grid.floor.ctrl_hz)
    nearest = np.argmax(floor_geometry(f[have], ctrl_hz)[0], axis=1)
    ctrl = np.full(ctrl_hz.size, np.nan)
    for j in range(ctrl_hz.size):
        cell = nearest == j
        if np.any(cell):
            ctrl[j] = float(np.median(ratio_db[cell]))
    ok = np.isfinite(ctrl)
    if not np.any(ok):
        raise ValueError("no floor bin between the lines in any frame: cannot initialise the floor")
    ctrl = np.interp(np.arange(ctrl.size), np.flatnonzero(ok), ctrl[ok])
    mu = float(np.mean(ctrl))
    rel = ctrl - mu
    sigma_b = max(float(np.sqrt(np.mean(rel**2))), FLOOR_SHAPE_SD_MIN_DB)
    z0 = _ridge_z(grid, sigma_b, rel)

    # ── steps 2-3: per frame, the floor and each line's unit kernel at its bin ──
    ped = pedestal_lag(grid, data.pedestal)
    line_ok = np.zeros((r_n, kk, n_frames), dtype=bool)
    jbin = np.zeros((r_n, kk, n_frames), dtype=np.int64)
    for r in range(r_n):
        fl = orders[:, None] * sbar[r][None, :] * SP.AMP_RPS_REF  # (K, N)
        jbin[r] = np.clip(np.rint(fl / df).astype(np.int64), 0, n_bins - 1)
        line_ok[r] = (fl >= SP.BAND_F_MIN) & (fl <= f_top)
    unit_a = np.zeros((r_n, kk, n_frames))
    floor_at = np.zeros((r_n, kk, n_frames))
    obs_at = np.zeros((r_n, kk, n_frames))
    mo_idx = data.motor_orders - 1
    unit_m = np.zeros((r_n, mo_idx.size, n_frames))
    obs_mean = obs.mean(axis=0)  # (N, F)
    with torch.no_grad():
        c_init = _floor_c(grid, mu, z0, sigma_b, ref)
        b0_t = _t(V4.B0[0], ref)
        b1_t = _t(V4.B1[0], ref)
        l_f = _t(l_f_np, ref)
        v_unit = _v_unit(grid, V4.FLIGHT_LAM)
        fs = SP.floor_frames_from_autocov(grid, g1_static, c_init)
        for sl in _frame_slices(n_frames, batch.chunk_frames):
            phase, speed = _phase_speed(grid, batch.rate_work[:, sl])
            fr = _floor_one(grid, speed, c_init, b0_t, b1_t, l_f)
            fl_mean = ((fr + s_init * fs) * grid.transfer_power).cpu().numpy()  # (n_c, F)
            frames = np.arange(sl.start, sl.stop)
            sigma2 = _sigma2_frames(grid, speed, log_sig_t, a_sig_t)
            for fam, kt, alpha, fam_ped, sink in (
                ("a", k_a, alpha_a0, ped, unit_a),
                ("m", k_m, alpha_m0, None, unit_m),
            ):
                if kt.numel() == 0:
                    continue
                env = torch.sqrt(speed ** float(alpha)) * grid.window_work
                for c0 in range(0, int(kt.numel()), chunk_orders):
                    c1 = min(c0 + chunk_orders, int(kt.numel()))
                    g, _ = _unit_autocorr(phase, env, kt[c0:c1], n_work, None)
                    rho = _shaft_frames(kt[c0:c1], sigma2, v_unit)
                    if fam_ped is not None:
                        rho = rho * fam_ped[c0:c1, None, :]
                    spec = _lag_to_bins(grid, 2.0 * g * rho)
                    spec = (spec * grid.transfer_power).cpu().numpy()  # (R, kc, n_c, F)
                    kidx = (kt[c0:c1].cpu().numpy() - 1).astype(np.int64)
                    for r in range(r_n):
                        jj = jbin[r][np.ix_(kidx, frames)]  # (kc, n_c)
                        sink[r, c0:c1, sl] = (
                            np.take_along_axis(spec[r], jj[..., None], axis=-1)[..., 0] * gbar[r]
                        )
                        if fam == "a":
                            floor_at[r, c0:c1, sl] = np.take_along_axis(fl_mean, jj.T, axis=1).T
                            obs_at[r, c0:c1, sl] = np.take_along_axis(obs_mean[sl], jj.T, axis=1).T

    valid = line_ok & (unit_a > 0.0)
    cnt = valid.sum(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        fhat_lin = np.where(valid, floor_at / np.where(valid, unit_a, 1.0), 0.0).sum(axis=2) / cnt
        fhat = 10.0 * np.log10(fhat_lin)
    for r in range(r_n):
        okk = np.isfinite(fhat[r])
        if np.any(okk):
            fhat[r] = np.interp(orders, orders[okk], fhat[r][okk])
    if not np.all(np.isfinite(fhat)):
        fill = float(np.nanmean(np.where(np.isfinite(fhat), fhat, np.nan)))
        fhat = np.where(np.isfinite(fhat), fhat, fill)
    fhat_k = fhat.mean(axis=0)
    odd = (np.arange(1, kk + 1) % int(data.blades) != 0).astype(np.float64)
    centre_a = fhat_k - V4.C4_OFFSET_DB - V4.C4_ODD_PENALTY_DB * odd
    centre_m = fhat_k[mo_idx] - V4.C4_OFFSET_DB

    excess = np.where(valid, obs_at - floor_at, 0.0).sum(axis=2)
    den_a = np.where(valid, unit_a, 0.0).sum(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_a = np.where(
            (excess > 0.0) & (den_a > 0.0),
            10.0 * np.log10(np.maximum(excess, 1e-300) / np.where(den_a > 0, den_a, 1.0)),
            np.broadcast_to(centre_a, (r_n, kk)),
        )
    p_m = np.broadcast_to(centre_m, (r_n, mo_idx.size)).copy()
    vm = valid[:, mo_idx]
    den_m = np.where(vm, unit_m, 0.0).sum(axis=2)
    for i, k in enumerate(data.motor_orders):
        lo, hi = k - 1 - data.blades, k - 1 + data.blades
        nb = [j for j in (lo, hi) if 0 <= j < kk]
        aero = p_a[:, nb].mean(axis=1) if nb else centre_a[k - 1] * np.ones(r_n)
        rest = excess[:, k - 1] - 10.0 ** (aero / 10.0) * den_a[:, k - 1]
        p_a[:, k - 1] = aero
        with np.errstate(divide="ignore", invalid="ignore"):
            p_m[:, i] = np.where(
                (rest > 0.0) & (den_m[:, i] > 0.0),
                10.0
                * np.log10(np.maximum(rest, 1e-300) / np.where(den_m[:, i] > 0, den_m[:, i], 1.0)),
                centre_m[i],
            )

    def hier(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pbar = p.mean(axis=0)
        c = (p - pbar[None, :]).mean(axis=1)
        return pbar, c - c.mean()

    pbar_a, c_a = hier(p_a)
    pbar_m, c_m = hier(p_m) if p_m.shape[1] else (np.zeros(0), np.zeros(r_n))
    init = dict(
        p_a=p_a,
        pbar_a=pbar_a,
        c_a=c_a[:-1],
        p_m=p_m,
        pbar_m=pbar_m,
        c_m=c_m[:-1],
        alpha_a=np.asarray(alpha_a0),
        alpha_m=np.asarray(alpha_m0),
        # b(f) >= 0 over the band: fitted at the band edges (v4_priors)
        **dict(zip(("u_blo", "u_bhi"), map(np.asarray, V4.floor_coords(V4.B0[0], V4.B1[0])))),
        z=z0,
        log_s=np.asarray(STATIC_SHARE_LN[0]),
        log_sigma_nu=np.asarray(V4.FLIGHT_SIGMA_NU_LN[0]),
        a_sigma=np.asarray(V4.FLIGHT_A_SIGMA[0]),
    )
    record = dict(
        floor_rule=(
            "per frame and mic the minimum in each run of bins between neighbouring lines of all "
            "rotors (+-1 bin excluded) x run length, over the unit floor x (sbar_mean^b_init "
            "+ s_init); mean per bin, median per spline cell; mu = mean, sigma_B = rms about it"
        ),
        floor_bins_read=int(have.sum()),
        floor_ctrl_hz=ctrl_hz.tolist(),
        floor_ctrl_db=ctrl.tolist(),
        floor_mean_db=mu,
        floor_shape_sd_db=sigma_b,
        floor_line_db=fhat.tolist(),
        floor_order_db=fhat_k.tolist(),
        line_frames_in_band=cnt.tolist(),
        init_rule="spec § Initialisation steps 1-5 (no earlier fit read)",
        init_alpha_aero=alpha_a0,
        init_sigma_nu_ref=sigma0,
        init_a_sigma=V4.FLIGHT_A_SIGMA[0],
        sbar_range_rev_s=[float(sbar.min() * SP.AMP_RPS_REF), float(sbar.max() * SP.AMP_RPS_REF)],
    )
    return FlightMeasured(
        floor_mean_db=mu,
        floor_shape_sd_db=sigma_b,
        floor_ctrl_db=rel,
        floor_line_db=fhat,
        floor_order_db=fhat_k,
        init=init,
        record=record,
    )


# ── the model ───────────────────────────────────────────────────────────────


class FlightV4Model:
    """The v4 flight forward model and MAP objective of one MD2 pool."""

    def __init__(
        self,
        data: FlightData,
        measured: FlightMeasured,
        *,
        kappa: float = 0.0,
        harmonic_chunk: int = 32,
        init: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.data = data
        self.meas = measured
        self.kappa = float(kappa)
        self.harmonic_chunk = int(harmonic_chunk)
        grid = data.batch.grid
        assert isinstance(grid, SP.FlightGrid)
        self.grid = grid
        ref = grid.window_work
        self.ref = ref
        kk = data.k_max
        self.k_a = torch.arange(1, kk + 1, dtype=torch.float64, device=ref.device)
        self.k_m = _t(data.motor_orders, ref)
        self.ped = pedestal_lag(grid, data.pedestal)
        self.gain = _t(10.0 ** (data.geometry_db.T / 10.0), ref)  # (M, R)
        self.l_f = _t(np.log2(np.maximum(grid.freqs_hz, B_F_LO_HZ) / B_F_REF_HZ), ref)
        self.g1_static = _atom_autocorr(grid.window_work, int(grid.n_fft_work))
        self.v_unit = _v_unit(grid, V4.FLIGHT_LAM)
        self.mu = _t(measured.floor_mean_db, ref)
        self.sigma_b = _t(measured.floor_shape_sd_db, ref)
        odd = (np.arange(1, kk + 1) % int(data.blades) != 0).astype(np.float64)
        fk = measured.floor_order_db
        self.centre_a = _t(fk - V4.C4_OFFSET_DB - V4.C4_ODD_PENALTY_DB * odd, ref)
        self.centre_m = _t(fk[data.motor_orders - 1] - V4.C4_OFFSET_DB, ref)
        start = measured.init if init is None else init
        self.params: dict[str, Tensor] = {
            key: torch.tensor(np.asarray(start[key], dtype=np.float64), device=ref.device)
            .to(torch.float64)
            .requires_grad_(True)
            for key in (
                "p_a",
                "pbar_a",
                "c_a",
                "p_m",
                "pbar_m",
                "c_m",
                "alpha_a",
                "alpha_m",
                "u_blo",
                "u_bhi",
                "z",
                "log_s",
                "log_sigma_nu",
                "a_sigma",
            )
        }
        self.n_cells = int(data.batch.n_cells)

    # ── pieces ──
    @staticmethod
    def _full_c(c_free: Tensor) -> Tensor:
        """The rotor offsets with their sum pinned at zero."""
        return torch.cat((c_free, -c_free.sum().reshape(1)))

    def values(self) -> dict[str, Tensor]:
        p = self.params
        b0, b1 = V4.floor_exponents(p["u_blo"], p["u_bhi"])
        return dict(
            p,
            c_a_full=self._full_c(p["c_a"]),
            c_m_full=self._full_c(p["c_m"]),
            s=torch.exp(p["log_s"]),
            sigma_nu=torch.exp(p["log_sigma_nu"]),
            b0=b0,
            b1=b1,
        )

    def ctrl_db(self, z: Tensor | None = None) -> Tensor:
        """``(14,)`` the floor's control values ``mu + sigma_B (L z)_j`` (dB)."""
        zz = self.params["z"] if z is None else z
        return self.mu + self.grid.floor.shape_db(zz, self.sigma_b)

    def floor_c(self) -> Tensor:
        return _floor_c(self.grid, self.mu, self.params["z"], self.sigma_b, self.ref)

    def components(self, rate_work: Tensor) -> dict[str, Tensor]:
        """A chunk's modelled parts before the transfer: ``lines_r`` per rotor
        ``(R, n_c, F)``, the one ``floor`` ``(n_c, F)`` and ``floor_s`` ``(F,)``."""
        v = self.values()
        grid = self.grid
        rate = rate_work.to(dtype=self.ref.dtype, device=self.ref.device)
        phase, speed = _phase_speed(grid, rate)
        sigma2 = _sigma2_frames(grid, speed, v["log_sigma_nu"], v["a_sigma"])
        lag = _family_lag_sum(
            grid,
            phase,
            speed,
            self.k_a,
            2.0 * 10.0 ** (v["p_a"] / 10.0),
            sigma2,
            self.v_unit,
            self.ped,
            v["alpha_a"],
            self.harmonic_chunk,
        )
        if self.k_m.numel():
            lag = lag + _family_lag_sum(
                grid,
                phase,
                speed,
                self.k_m,
                2.0 * 10.0 ** (v["p_m"] / 10.0),
                sigma2,
                self.v_unit,
                None,
                v["alpha_m"],
                self.harmonic_chunk,
            )
        c = self.floor_c()
        return dict(
            lines_r=_lag_to_bins(grid, lag),
            floor=_floor_one(grid, speed, c, v["b0"], v["b1"], self.l_f),
            floor_s=v["s"] * SP.floor_frames_from_autocov(grid, self.g1_static, c),
        )

    def forward(self, batch: MD.SupportBatch) -> Tensor:
        """``(M, n_c, F)`` expected periodogram of ``batch``'s frames."""
        assert batch.rate_work is not None
        parts = self.components(batch.rate_work)
        lines = torch.einsum("mr,rnf->mnf", self.gain, parts["lines_r"])
        observed = (lines + parts["floor"] + parts["floor_s"]) * self.grid.transfer_power
        return observed.clamp_min(MODEL_FLOOR)

    def log_prior(self) -> tuple[Tensor, dict[str, float]]:
        """``log p`` of every fitted value plus the floor push ``kappa sum_j cbar_j``."""
        v = self.values()
        nd = torch.distributions.Normal
        ln = torch.distributions.LogNormal
        ref = self.ref

        def n_(loc: Any, sd: float, x: Tensor) -> Tensor:
            return (
                nd(torch.as_tensor(loc, dtype=torch.float64, device=ref.device), sd)
                .log_prob(x)
                .sum()
            )

        parts = dict(
            profile_aero=n_(self.centre_a, V4.C4_SD_DB, v["pbar_a"])
            + n_(0.0, V4.PROFILE_ROTOR_SD_DB, v["c_a_full"])
            + n_(v["pbar_a"][None, :] + v["c_a_full"][:, None], V4.PROFILE_LINE_SD_DB, v["p_a"]),
            profile_motor=n_(self.centre_m, V4.C4_SD_DB, v["pbar_m"])
            + n_(0.0, V4.PROFILE_ROTOR_SD_DB, v["c_m_full"])
            + n_(v["pbar_m"][None, :] + v["c_m_full"][:, None], V4.PROFILE_LINE_SD_DB, v["p_m"]),
            alpha=n_(V4.ALPHA_AERO[0], V4.ALPHA_AERO[1], v["alpha_a"])
            + n_(V4.ALPHA_MOTOR[0], V4.ALPHA_MOTOR[1], v["alpha_m"]),
            floor_exp=n_(V4.B0[0], V4.B0[1], v["b0"]) + n_(V4.B1[0], V4.B1[1], v["b1"]),
            floor_z=n_(0.0, 1.0, v["z"]),
            static=ln(_t(STATIC_SHARE_LN[0], ref), STATIC_SHARE_LN[1]).log_prob(v["s"]),
            sigma_nu=ln(_t(V4.FLIGHT_SIGMA_NU_LN[0], ref), V4.FLIGHT_SIGMA_NU_LN[1]).log_prob(
                v["sigma_nu"]
            )
            + n_(V4.FLIGHT_A_SIGMA[0], V4.FLIGHT_A_SIGMA[1], v["a_sigma"]),
            floor_push=self.kappa * self.ctrl_db().sum(),
        )
        total = torch.stack([t.reshape(()) for t in parts.values()]).sum()
        return total, {k: float(t.detach()) for k, t in parts.items()}

    def objective(self, batch: MD.SupportBatch, *, grad: bool) -> Tensor:
        """``Whittle - log p`` on ``batch``, the Whittle term summed over its
        frame chunks with the gradient accumulated chunk by chunk."""
        with torch.set_grad_enabled(grad):
            lp, _ = self.log_prior()
            nlp = -lp
            if grad:
                nlp.backward()
            total = nlp.detach()
            for sub in MD.frame_chunks(batch):
                w = MD.whittle_risk(sub, self.forward(sub))
                if grad:
                    w.backward()
                total = total + w.detach()
        return total

    def whittle(self, batch: MD.SupportBatch) -> float:
        with torch.no_grad():
            return float(
                sum(float(MD.whittle_risk(s, self.forward(s))) for s in MD.frame_chunks(batch))
            )

    def box(self) -> None:
        with torch.no_grad():
            for key in ("log_s", "log_sigma_nu"):
                self.params[key].clamp_(-LOG_BOX, LOG_BOX)

    def snapshot(self) -> dict[str, np.ndarray]:
        """The groups the convergence test reads, as numbers."""
        with torch.no_grad():
            v = self.values()
            return dict(
                levels_db=np.concatenate(
                    [v["p_a"].cpu().numpy().ravel(), v["p_m"].cpu().numpy().ravel()]
                ),
                exponents=np.array(
                    [float(v[k]) for k in ("alpha_a", "alpha_m", "b0", "b1", "a_sigma")]
                ),
                floor_ctrl_db=self.ctrl_db().cpu().numpy(),
                log_sigma_nu=np.array([float(v["log_sigma_nu"])]),
                log_s=np.array([float(v["log_s"])]),
            )


def movement(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, float]:
    """Per-group parameter movement between two snapshots (spec step 5)."""
    return dict(
        levels_db=float(np.median(np.abs(b["levels_db"] - a["levels_db"]))),
        exponents=float(np.max(np.abs(b["exponents"] - a["exponents"]))),
        floor_ctrl_db=float(np.max(np.abs(b["floor_ctrl_db"] - a["floor_ctrl_db"]))),
        log_sigma_nu=float(np.max(np.abs(b["log_sigma_nu"] - a["log_sigma_nu"]))),
        log_s=float(np.max(np.abs(b["log_s"] - a["log_s"]))),
    )


# ── the optimiser (spec § The optimiser; nv3r4 schedule) ───────────────────


@dataclass(frozen=True)
class OptimV4:
    """The flight schedule. Defaults: spec § The optimiser on the nv3r4 front."""

    adam_steps: int = 1500
    adam_lr: float = 0.02
    adam_batch: int = 8
    lbfgs_iters: int = 600
    lbfgs_frames: int = 64
    polish_iters: int = 150
    lbfgs_history: int = 10
    tol_nats_per_cell: float = TOL_NATS_PER_CELL
    seed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return dict(
            adam_steps=self.adam_steps,
            adam_lr=self.adam_lr,
            adam_batch=self.adam_batch,
            lbfgs_iters=self.lbfgs_iters,
            lbfgs_frames=self.lbfgs_frames,
            lbfgs_frames_rule="model.stratified_frames over the windows",
            polish_iters=self.polish_iters,
            restart_pass_iters="half the pass it restarts (fresh L-BFGS memory)",
            lbfgs_history=self.lbfgs_history,
            lbfgs_line_search="strong_wolfe",
            tol_nats_per_cell=self.tol_nats_per_cell,
            movement_tolerances=dict(MOVE_TOL),
            seed=self.seed,
            evidence_freeze=(
                "not run: the flight fit holds the phase pedestal at its prior medians "
                "(spec § Procedure B), so there is nothing to freeze"
            ),
        )


def restart_init(measured: FlightMeasured, seed: int) -> dict[str, np.ndarray]:
    """Restart ``seed``'s start: 0 takes the prior centres of the shaft law (the
    pedestal is held); 1, 2, ... draw ``sigma_ref`` and ``a_sigma`` from their
    priors."""
    init = {k: np.array(v, copy=True) for k, v in measured.init.items()}
    if int(seed) > 0:
        rng = np.random.default_rng(int(seed))
        init["log_sigma_nu"] = np.asarray(
            rng.normal(V4.FLIGHT_SIGMA_NU_LN[0], V4.FLIGHT_SIGMA_NU_LN[1])
        )
        init["a_sigma"] = np.asarray(rng.normal(V4.FLIGHT_A_SIGMA[0], V4.FLIGHT_A_SIGMA[1]))
    return init


def init_from_payload(pay: dict[str, Any]) -> dict[str, np.ndarray]:
    """The raw parameters of a ``noise-v4-fit/3`` flight payload of the same
    pool: the start of a continuation (warm-start) fit."""
    pr, h = pay["params"], pay["hierarchy"]
    sh, fl = pr["shaft"], pr["floor"]

    def a(v: Any) -> np.ndarray:
        return np.asarray(v, dtype=np.float64)

    return dict(
        p_a=a(pr["profile_aero_db"]),
        pbar_a=a(h["pbar_aero_db"]),
        c_a=a(h["c_aero_db"])[:-1],
        p_m=a(pr["profile_motor_db"]),
        pbar_m=a(h["pbar_motor_db"]),
        c_m=a(h["c_motor_db"])[:-1],
        alpha_a=a(pr["alpha_aero"]),
        alpha_m=a(pr["alpha_motor"]),
        **dict(zip(("u_blo", "u_bhi"), map(np.asarray, V4.floor_coords(fl["b0"], fl["b1"])))),
        z=a(fl["shape_z"]),
        log_s=np.log(a(fl["static_rel"])),
        log_sigma_nu=np.log(a(sh["sigma_nu_rad_s"])),
        a_sigma=a(sh["a_sigma"]),
    )


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _lbfgs_pass(
    model: FlightV4Model, batch: MD.SupportBatch, iters: int, history: int
) -> dict[str, Any]:
    """One fresh-memory strong-Wolfe L-BFGS pass on ``batch``."""
    params = list(model.params.values())
    opt = LBFGS(
        params, max_iter=int(iters), history_size=int(history), line_search_fn="strong_wolfe"
    )
    evals = 0

    def closure() -> Tensor:
        nonlocal evals
        model.box()
        opt.zero_grad(set_to_none=False)
        evals += 1
        return model.objective(batch, grad=True)

    dev = model.ref.device
    t0 = time.time()
    with torch.no_grad():
        before = float(model.objective(batch, grad=False))
    opt.step(closure)
    model.box()
    with torch.no_grad():
        after = float(model.objective(batch, grad=False))
    _sync(dev)
    wall = time.time() - t0
    return dict(
        loss_before=before,
        loss_after=after,
        iters=int(opt.n_iter),
        evals=int(evals),
        wall_s=wall,
        s_per_eval=wall / max(1, evals + 2),
        trace=[float(x) for x in opt.trace],
        frames=int(batch.power.shape[1]),
        cells=int(batch.n_cells),
    )


def fit_flight(
    data: FlightData,
    measured: FlightMeasured,
    *,
    kappa: float = 0.0,
    optim: OptimV4 = OptimV4(),
    progress: int = 0,
    start: dict[str, Any] | None = None,
) -> tuple[FlightV4Model, dict[str, Any]]:
    """MAP-fit one restart (spec steps B2, § The optimiser).

    Adam on ``adam_batch``-frame minibatches; L-BFGS on the stratified
    ``lbfgs_frames`` subset and a fresh-memory restart pass of half its
    budget; the all-frames polish and its fresh-memory restart pass, which is
    the convergence test: gain per cell under ``tol_nats_per_cell`` AND every
    group's movement under :data:`MOVE_TOL`. ``start`` (a payload of the same
    pool) replaces the restart's initialisation (a continuation); with
    ``lbfgs_iters = 0`` the subset stage is skipped and the polish runs
    ``polish_iters``."""
    torch.manual_seed(int(optim.seed))
    batch = data.batch
    dev = batch.power.device
    init = (
        init_from_payload(start) if start is not None else restart_init(measured, int(optim.seed))
    )
    model = FlightV4Model(data, measured, kappa=kappa, init=init)
    rng = np.random.default_rng(int(optim.seed))
    n_frames = int(batch.power.shape[1])
    t0 = time.time()

    with torch.no_grad():
        start_loss = float(model.objective(batch, grad=False))
    adam_losses: list[float] = []
    if int(optim.adam_steps) > 0:
        params = list(model.params.values())
        adam = torch.optim.Adam(params, lr=float(optim.adam_lr))
        for step in range(int(optim.adam_steps)):
            if int(optim.adam_batch) < n_frames:
                idx = np.sort(rng.choice(n_frames, size=int(optim.adam_batch), replace=False))
                sub = MD.batch_slice(batch, idx)
            else:
                sub = batch
            adam.zero_grad(set_to_none=False)
            loss = float(model.objective(sub, grad=True))
            adam.step()
            model.box()
            adam_losses.append(loss)
            if progress and step % progress == 0:
                print(f"  adam {step:5d}  loss {loss:.6g}", flush=True)
    _sync(dev)
    adam_s = time.time() - t0
    with torch.no_grad():
        after_adam = float(model.objective(batch, grad=False))

    stages: dict[str, Any] = {}
    if int(optim.lbfgs_frames) < n_frames and int(optim.lbfgs_iters) > 0:
        assert batch.frame_window is not None
        idx = np.unique(MD.stratified_frames(batch.frame_window, int(optim.lbfgs_frames)))
        subset = MD.batch_slice(batch, idx)
        stages["subset"] = _lbfgs_pass(model, subset, optim.lbfgs_iters, optim.lbfgs_history)
        stages["subset_restart"] = _lbfgs_pass(
            model, subset, max(1, optim.lbfgs_iters // 2), optim.lbfgs_history
        )
        if progress:
            print(f"  lbfgs subset {stages['subset_restart']['loss_after']:.6g}", flush=True)
    polish_iters = (
        int(optim.polish_iters) if stages or int(optim.lbfgs_iters) == 0 else int(optim.lbfgs_iters)
    )
    stages["polish"] = _lbfgs_pass(model, batch, polish_iters, optim.lbfgs_history)
    before_restart = model.snapshot()
    stages["polish_restart"] = _lbfgs_pass(
        model, batch, max(1, polish_iters // 2), optim.lbfgs_history
    )
    moved = movement(before_restart, model.snapshot())
    gain = max(0.0, stages["polish"]["loss_after"] - stages["polish_restart"]["loss_after"])
    gain_cell = gain / max(1, model.n_cells)
    gain_ok = bool(gain_cell < float(optim.tol_nats_per_cell))
    move_ok = {k: bool(moved[k] < tol) for k, tol in MOVE_TOL.items()}
    converged = gain_ok and all(move_ok.values())
    if progress:
        print(
            f"  polish {stages['polish_restart']['loss_after']:.6g}  gain/cell {gain_cell:.3g}  "
            f"moved {moved}  converged {converged}",
            flush=True,
        )

    with torch.no_grad():
        lp, lp_parts = model.log_prior()
        whittle = model.whittle(batch)
    total = whittle - float(lp)
    record = dict(
        objective=dict(
            total_nats=total,
            whittle_nats=whittle,
            neg_log_prior_nats=-float(lp),
            log_prior_parts=lp_parts,
            n_cells=model.n_cells,
            total_per_cell=total / max(1, model.n_cells),
            start_total_nats=start_loss,
            exposure_scale=batch.exposure_scale,
        ),
        optimiser=dict(
            **optim.as_dict(),
            started_from_payload=start is not None,
            device=str(dev),
            chunk_frames=batch.chunk_frames,
            adam_first_loss=adam_losses[0] if adam_losses else None,
            adam_last_loss=adam_losses[-1] if adam_losses else None,
            adam_loss_head=adam_losses[:5],
            adam_loss_tail=adam_losses[-5:],
            adam_wall_s=adam_s,
            adam_s_per_step=adam_s / max(1, len(adam_losses)),
            total_after_adam=after_adam,
            stages=stages,
            convergence=dict(
                test="the all-frames polish's fresh-memory restart pass",
                gain_nats=gain,
                gain_per_cell=gain_cell,
                gain_ok=gain_ok,
                movement=moved,
                movement_ok=move_ok,
                converged=bool(converged),
            ),
            wall_s=time.time() - t0,
        ),
    )
    return model, record


# ── export ──────────────────────────────────────────────────────────────────


def _np(v: Tensor) -> np.ndarray:
    return v.detach().cpu().numpy().astype(np.float64)


def payload(
    model: FlightV4Model,
    record: dict[str, Any],
    *,
    channel_gains: MD.ChannelGains | None,
    git: str | None = None,
) -> dict[str, Any]:
    """The ``noise-v4-fit/3`` flight payload of one restart (contract)."""
    data = model.data
    batch = data.batch
    grid = model.grid
    v = model.values()
    band_f = grid.freqs_hz[np.asarray(batch.band.cpu())]
    sigma_nu = float(v["sigma_nu"])
    ped = data.pedestal
    meas = model.meas
    array_response = None if channel_gains is None else channel_gains.array_response()
    return dict(
        schema=V4.FIT_SCHEMA,
        kind="flight",
        rig="michaels",
        mode=FLIGHT_V4_MODE,
        n_rotors=batch.n_rotors,
        n_mics=batch.n_mics,
        mics=list(data.mics),
        rotor_order=list(ROTOR_ORDER),
        blades=data.blades,
        pole_pairs=data.pole_pairs,
        f_ref_rev_s=float(SP.AMP_RPS_REF),
        sr=int(grid.sr),
        band_hz=[float(band_f.min()), float(band_f.max())],
        front_end=dict(sr_work=int(grid.sr_work)),
        params=dict(
            profile_aero_db=_np(v["p_a"]).tolist(),
            motor_orders=data.motor_orders.tolist(),
            profile_motor_db=_np(v["p_m"]).tolist(),
            alpha_aero=float(v["alpha_a"]),
            alpha_motor=float(v["alpha_m"]),
            pedestal=dict(
                sigma2_amp=np.asarray(ped["sigma2_amp"]).tolist(),
                gamma_amp_hz=np.asarray(ped["gamma_amp_hz"]).tolist(),
                sigma2_phase=np.asarray(ped["sigma2_phase"]).tolist(),
                gamma_phase_hz=np.asarray(ped["gamma_phase_hz"]).tolist(),
                held=True,
                source=dict(
                    amplitude=(
                        f"block wander {data.wander_path}: sigma2 = (sigma_v ln10/20)^2 capped at "
                        "v4_priors.amp_cap, gamma = 1/(2 pi tau_v), per order group"
                    ),
                    phase="prior medians: sigma2 = exp(PHASE_SIGMA2_LN[0]); W = c_W x 0.6745",
                    sigma2_amp_uncapped=np.asarray(ped["sigma2_amp_uncapped"]).tolist(),
                ),
            ),
            shaft=dict(
                eps=sigma_nu / (2.0 * math.pi * float(SP.AMP_RPS_REF)),
                a_sigma=float(v["a_sigma"]),
                lam_ref_per_s=float(V4.FLIGHT_LAM),
                b_lam=0.0,
                f_ref_rev_s=float(SP.AMP_RPS_REF),
                pinned=["lam_ref_per_s", "b_lam"],
                sigma_nu_rad_s=sigma_nu,
                sigma_nu_note="sigma_ref at f_ref; per rotor and frame sigma_ref s^a_sigma",
            ),
            floor=dict(
                floor_rule=FLOOR_RULE_ONE,
                mean_db=float(meas.floor_mean_db),
                shape_z=_np(v["z"]).tolist(),
                shape_sd_db=float(meas.floor_shape_sd_db),
                ctrl_hz=np.asarray(grid.floor.ctrl_hz).tolist(),
                ctrl_db=_np(model.ctrl_db()).tolist(),
                b0=float(v["b0"]),
                b1=float(v["b1"]),
                b_f_lo_hz=B_F_LO_HZ,
                b_f_ref_hz=B_F_REF_HZ,
                static_rel=float(v["s"]),
                basis="experiments.noise_model.spectrum FloorBasis (tilt = 0), sr_work grid",
                sr_work=int(grid.sr_work),
            ),
            mic_response=dict(
                kind="array_geometry",
                array_response=array_response,
                geometry_db=np.asarray(data.geometry_db).tolist(),
                geometry_rule="Delta G_rm = G_rm - mean_r G_rm, G = -20 log10 d_rm "
                "(data_processing.sources.michaels.get_geometry)",
            ),
            wind=None,
        ),
        hierarchy=dict(
            pbar_aero_db=_np(v["pbar_a"]).tolist(),
            c_aero_db=_np(v["c_a_full"]).tolist(),
            pbar_motor_db=_np(v["pbar_m"]).tolist(),
            c_motor_db=_np(v["c_m_full"]).tolist(),
        ),
        objective=record["objective"],
        optimiser=record["optimiser"],
        priors=priors_record(model),
        measured=meas.record,
        pool=dict(
            members=list(batch.members),
            k_max=data.k_max,
            batch=dict(batch.diagnostics),
        ),
        git=git if git is not None else _git_sha(),
    )


def priors_record(model: FlightV4Model) -> dict[str, Any]:
    """Every prior value the fit used (contract ``priors``)."""
    return dict(
        profile_aero_centre=dict(
            rule="N(F-hat_k - C4_OFFSET - C4_ODD_PENALTY [k % B != 0], C4_SD)",
            offset_db=V4.C4_OFFSET_DB,
            odd_penalty_db=V4.C4_ODD_PENALTY_DB,
            sd_db=V4.C4_SD_DB,
            centre_db=_np(model.centre_a).tolist(),
        ),
        profile_motor_centre=dict(
            rule="N(F-hat_{3pn} - C4_OFFSET, C4_SD)", centre_db=_np(model.centre_m).tolist()
        ),
        profile_line_sd_db=V4.PROFILE_LINE_SD_DB,
        profile_rotor_sd_db=V4.PROFILE_ROTOR_SD_DB,
        rotor_offset_rule="sum over rotors pinned at 0",
        alpha_aero=list(V4.ALPHA_AERO),
        alpha_motor=list(V4.ALPHA_MOTOR),
        b0=list(V4.B0),
        b1=list(V4.B1),
        floor_z="N(0, I), control values mu + sigma_B (L z)",
        floor_shape_sd_min_db=FLOOR_SHAPE_SD_MIN_DB,
        static_rel_lognormal=list(STATIC_SHARE_LN),
        sigma_nu_lognormal=list(V4.FLIGHT_SIGMA_NU_LN),
        sigma_nu_at="f_ref",
        a_sigma_normal=list(V4.FLIGHT_A_SIGMA),
        lam_pinned_per_s=V4.FLIGHT_LAM,
        kappa=model.kappa,
        floor_push="+kappa sum_j cbar_j on the log prior (dB)",
        amp_groups=[list(g) for g in V4.AMP_GROUPS],
        amp_log_sd=V4.AMP_LOG_SD,
        phase_sigma2_ln=list(V4.PHASE_SIGMA2_LN),
        c_w=V4.C_W,
        w_max_hz=V4.W_MAX_HZ,
        phase_sigma2_max=V4.PHASE_SIGMA2_MAX,
        c_w_note="v4_priors.C_W (step A0, even orders, every order)",
    )


# ── restarts ────────────────────────────────────────────────────────────────


def _spread(values: list[Any]) -> Any:
    a = np.asarray(values, dtype=np.float64)
    if a.ndim == 1:
        return float(a.max() - a.min())
    rng = a.max(axis=0) - a.min(axis=0)
    return dict(median=float(np.median(rng)), max=float(np.max(rng)))


def reduce_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """The lowest-objective restart, with the spread of every group over the
    restarts (spec step 6)."""
    if not payloads:
        raise ValueError("no restart payloads")
    best = min(payloads, key=lambda d: float(d["objective"]["total_nats"]))
    pr = [d["params"] for d in payloads]
    spread = dict(
        total_nats=_spread([d["objective"]["total_nats"] for d in payloads]),
        profile_aero_db=_spread([p["profile_aero_db"] for p in pr]),
        profile_motor_db=_spread([p["profile_motor_db"] for p in pr]),
        alpha_aero=_spread([p["alpha_aero"] for p in pr]),
        alpha_motor=_spread([p["alpha_motor"] for p in pr]),
        b0=_spread([p["floor"]["b0"] for p in pr]),
        b1=_spread([p["floor"]["b1"] for p in pr]),
        static_rel=_spread([p["floor"]["static_rel"] for p in pr]),
        floor_ctrl_db=_spread([p["floor"]["ctrl_db"] for p in pr]),
        sigma_nu_rad_s=_spread([p["shaft"]["sigma_nu_rad_s"] for p in pr]),
        a_sigma=_spread([p["shaft"]["a_sigma"] for p in pr]),
    )
    out = dict(best)
    out["restarts"] = dict(
        n=len(payloads),
        seeds=[d["optimiser"]["seed"] for d in payloads],
        total_nats=[d["objective"]["total_nats"] for d in payloads],
        converged=[d["optimiser"]["convergence"]["converged"] for d in payloads],
        selected_seed=best["optimiser"]["seed"],
        spread=spread,
    )
    return out
