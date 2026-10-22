"""The priors of noise model v4 (``docs/explainers/noise-model-v4.qmd`` § The priors).

One module for the bench rig fit (:mod:`.bench_v4`) and the flight rig fit, so
the two procedures read the same numbers. Everything here is a constant or a
pure function of the order ``k``.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "ALPHA_AERO",
    "ALPHA_MOTOR",
    "ALPHA_MOTOR_MAX",
    "AMP_GROUPS",
    "AMP_LOG_SD",
    "B0",
    "B1",
    "C4_ODD_PENALTY_DB",
    "C4_OFFSET_DB",
    "C4_SD_DB",
    "C_W",
    "EPS_HALF_NORMAL",
    "A_SIGMA",
    "B_LAM",
    "FIT_SCHEMA",
    "FLIGHT_A_SIGMA",
    "FLIGHT_LAM",
    "FLIGHT_SIGMA_NU_LN",
    "GAMMA_AMP_LOG_SD",
    "LAM_REF_LN",
    "PHASE_SIGMA2_MAX",
    "B_EDGES_HZ",
    "B_F_REF_HZ",
    "floor_coords",
    "floor_exponents",
    "PHASE_SIGMA2_LN",
    "PROFILE_LINE_SD_DB",
    "PROFILE_ROTOR_SD_DB",
    "W_MAX_HZ",
    "amp_cap",
    "amp_log_median",
    "gamma_amp_log_median",
    "pedestal_width",
    "width_scale",
]

FIT_SCHEMA = "noise-v4-fit/3"

#: Level prior relative to the measured floor (nv3r4 change C4).
C4_OFFSET_DB = 12.0
C4_ODD_PENALTY_DB = 15.0
C4_SD_DB = 10.0
#: The four rotor profiles: per-line spread around the rig centre, plus one
#: level offset per rotor (static-rig-profiles.md § Generic drone prior).
PROFILE_LINE_SD_DB = 3.5
PROFILE_ROTOR_SD_DB = 2.5

#: Speed exponents (line power ~ s^alpha; floor ~ s^{b0 + b1 log2(f/2 kHz)}).
ALPHA_AERO = (5.5, 1.0)
ALPHA_MOTOR = (0.0, 2.0)
#: Hard bound |alpha_motor| <= ALPHA_MOTOR_MAX (the flight fit's truncation of
#: the N(0, 2) prior at 2 sd). The bench measures the motor-line exponents at
#: -2.1..+1.9 (k = 84, four motors; static-rig speed_laws.json). A per-regime
#: flight pool spans 1.4-1.9x in speed, so its Whittle term can push alpha_motor
#: far out (MD2 cruise-only r5: -40.8, standby-only: +17.5), and the regime
#: composition then extrapolates the motor lines by 10^10 into the ramp.
ALPHA_MOTOR_MAX = 4.0
B0 = (3.0, 1.0)
B1 = (0.5, 0.25)
#: The floor exponent ``b(f) = b0 + b1 log2(f / B_F_REF_HZ)`` is truncated to
#: ``b >= 0`` over the likelihood band (``spectrum.BAND_F_MIN`` / ``BAND_F_MAX``):
#: the broadband floor does not get quieter as the rotor speeds up at any
#: frequency. ``b`` is linear in ``log f``, so the two band edges carry the
#: constraint; the fit's coordinates are ``u = softplus^-1 b`` at the edges.
B_F_REF_HZ = 2000.0
B_EDGES_HZ = (30.0, 7900.0)

#: Bench shaft law: sigma_nu = 2 pi f_ref eps s^a, lam = lam_ref s^b.
EPS_HALF_NORMAL = 0.005
A_SIGMA = (1.0, 0.5)
B_LAM = (1.0, 0.5)
LAM_REF_LN = (math.log(12.0), 0.5)

#: Flight shaft (nv3r4 A1, A2): lam pinned at the label's band edge, sigma_nu
#: LogNormal from the telemetry residuals of six drones; a = b = 0.
FLIGHT_LAM = 2.0 * math.pi * 16.0
FLIGHT_SIGMA_NU_LN = (1.68, 0.66)
#: Flight shaft speed exponent ``sigma_nu = sigma_ref s^a``: wider than the bench's
#: N(1, 0.5) because nv3 per-regime fits gave 0.35 vs 3.8 rad/s over ~2.3x (a ~ 3).
FLIGHT_A_SIGMA = (1.0, 2.0)

#: Amplitude pedestal: (k_lo, k_hi, median, cap) of the per-recording
#: log-amplitude variance over the 20 DREGON bench recordings
#: (results/static_rig/single_rotor/amplitude_jitter.json). The last group is
#: measured on k = 41-80 and extended to every higher order.
AMP_GROUPS = (
    (1, 3, 0.001, 0.075),
    (4, 8, 0.11, 0.29),
    (9, 20, 0.14, 0.24),
    (21, 40, 0.10, 0.20),
    (41, 10**9, 0.02, 0.12),
)
AMP_LOG_SD = 0.5
#: sd of log gamma_A around its median (``gamma_amp_log_median``).
GAMMA_AMP_LOG_SD = 0.5

#: Phase pedestal variance: wide; the bench data set it. Hard cap: the A0
#: p97.5 of s2P (2.81, results/noise_v4/phase_width/summary.json) rounded up,
#: so the pedestal holds at most 1 - e^-3 = 95 % of a line's power.
PHASE_SIGMA2_LN = (0.0, 1.5)
PHASE_SIGMA2_MAX = 3.0

#: Half-normal scale (Hz) of the phase-pedestal mean half-width W, every order:
#: p97.5 of the measured W on the blade-pass (even) orders / 2.24 (step A0, 20
#: bench recordings, n = 387). The odd orders measured wider (c_W 30.7 Hz,
#: n = 94), but a scale that wide let the round-4 bench fits trade odd-order
#: pedestals for floor at ~6-10 nats, so the even scale holds for all orders.
#: The fit on the resolved lags only (>= 33-61 ms) gives c_W 18.4 Hz: a width
#: much above 1/(2 pi 33 ms) ~ 5 Hz is decorrelated at every resolved lag and
#: is set there by its tail, so it is not used.
C_W = 6.94
#: Hard upper bound on W: half the slowest bench line spacing (48.4 rev/s). A
#: wider pedestal reaches its neighbours' lines and becomes floor.
W_MAX_HZ = 24.0


def _group(k: np.ndarray, col: int) -> np.ndarray:
    k = np.asarray(k, dtype=np.int64)
    out = np.empty(k.shape, dtype=np.float64)
    for lo, hi, med, cap in AMP_GROUPS:
        sel = (k >= lo) & (k <= hi)
        out[sel] = (lo, hi, med, cap)[col]
    return out


def amp_log_median(k: np.ndarray) -> np.ndarray:
    """``log`` of the amplitude-variance prior median per order."""
    return np.log(_group(k, 2))


def amp_cap(k: np.ndarray) -> np.ndarray:
    """The amplitude-variance cap per order (the group's largest p90)."""
    return _group(k, 3)


def gamma_amp_log_median(k: np.ndarray) -> np.ndarray:
    """``log`` of the amplitude rate prior median (Hz): 0.4 for k <= 8, 1.0 above."""
    k = np.asarray(k, dtype=np.float64)
    return np.where(k <= 8, math.log(0.4), math.log(1.0))


def width_scale(k: np.ndarray) -> np.ndarray:
    """Half-normal scale (Hz) of the phase-pedestal width of order ``k``."""
    return np.full(np.shape(k), C_W, dtype=np.float64)


def pedestal_width(gamma, sigma2):
    """Power-weighted mean half-width ``gamma sigma2 / (1 - e^{-sigma2})`` of a
    bounded OU pedestal; works on numpy arrays and torch tensors alike."""
    return gamma * sigma2 / (-(-sigma2).expm1() if hasattr(sigma2, "expm1") else -np.expm1(-sigma2))


def _edge_logs() -> tuple[float, float]:
    lo, hi = B_EDGES_HZ
    return math.log2(lo / B_F_REF_HZ), math.log2(hi / B_F_REF_HZ)


def floor_exponents(u_lo, u_hi):
    """``(b0, b1)`` from the edge coordinates: ``b = softplus(u) >= 0`` at both
    band edges, the line through them in ``log2(f / B_F_REF_HZ)``; numpy arrays
    or torch tensors alike."""
    if hasattr(u_lo, "clamp_min"):
        b_lo = (-u_lo.abs()).exp().log1p() + u_lo.clamp_min(0.0)
        b_hi = (-u_hi.abs()).exp().log1p() + u_hi.clamp_min(0.0)
    else:
        b_lo, b_hi = np.logaddexp(0.0, u_lo), np.logaddexp(0.0, u_hi)
    l_lo, l_hi = _edge_logs()
    b1 = (b_hi - b_lo) / (l_hi - l_lo)
    return b_lo - b1 * l_lo, b1


def floor_coords(b0: float, b1: float, *, b_min: float = 0.01) -> tuple[float, float]:
    """Inverse of :func:`floor_exponents`; an edge exponent under ``b_min``
    (an untruncated start, e.g. the prior centre) starts at ``b_min``."""
    l_lo, l_hi = _edge_logs()
    out = []
    for b in (b0 + b1 * l_lo, b0 + b1 * l_hi):
        b = max(float(b), b_min)
        out.append(b + math.log(-math.expm1(-b)))  # softplus^-1, overflow-safe
    return out[0], out[1]
