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
    "AMP_GROUPS",
    "AMP_LOG_SD",
    "B0",
    "B1",
    "C4_ODD_PENALTY_DB",
    "C4_OFFSET_DB",
    "C4_SD_DB",
    "C_W_EVEN",
    "C_W_ODD",
    "EPS_HALF_NORMAL",
    "A_SIGMA",
    "B_LAM",
    "FIT_SCHEMA",
    "FLIGHT_A_SIGMA",
    "FLIGHT_LAM",
    "FLIGHT_SIGMA_NU_LN",
    "GAMMA_AMP_LOG_SD",
    "LAM_REF_LN",
    "PHASE_SIGMA2_LN",
    "PROFILE_LINE_SD_DB",
    "PROFILE_ROTOR_SD_DB",
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
B0 = (3.0, 1.0)
B1 = (0.5, 0.25)

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

#: Phase pedestal variance: wide; the bench data set it.
PHASE_SIGMA2_LN = (0.0, 1.5)

#: Half-normal scales (Hz) of the phase-pedestal mean half-width W, blade-pass
#: orders and the others: p97.5 of the measured W per parity / 2.24 (step A0,
#: 20 bench recordings, results/noise_v4/phase_width/summary.json; even n = 387,
#: odd n = 94). The fit on the resolved lags only (>= 33-61 ms) gives
#: c_W 18.4 / 163 Hz: a width much above 1/(2 pi 33 ms) ~ 5 Hz is decorrelated
#: at every resolved lag and is set there by its tail, so it is not used.
C_W_EVEN = 6.94
C_W_ODD = 30.68


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


def width_scale(k: np.ndarray, blades: int) -> np.ndarray:
    """Half-normal scale (Hz) of the phase-pedestal width of order ``k``."""
    k = np.asarray(k, dtype=np.int64)
    return np.where(k % int(blades) == 0, C_W_EVEN, C_W_ODD).astype(np.float64)


def pedestal_width(gamma, sigma2):
    """Power-weighted mean half-width ``gamma sigma2 / (1 - e^{-sigma2})`` of a
    bounded OU pedestal; works on numpy arrays and torch tensors alike."""
    return gamma * sigma2 / (-(-sigma2).expm1() if hasattr(sigma2, "expm1") else -np.expm1(-sigma2))
