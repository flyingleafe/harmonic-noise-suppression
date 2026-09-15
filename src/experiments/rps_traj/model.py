"""The NEW generative rotor-speed model and its exact-likelihood fit.

The model (round 4).  Rotor speeds in rev/s, mixer order
``[RFront, LFront, LBack, RBack]``::

    w(t) = mu + delta_flight + M @ R(theta) @ v(t) + e(t)
    v_i(t) = OU(tau_slow_i, sigma_slow_i) + CAR2(f0_i, zeta_i, sigma_osc_i)
    e_j(t) = OU(tau_e, sigma_e),  independent across rotors

* ``M`` = :data:`tracking.rotors.MIXER`, columns ``[common, roll, pitch, yaw]``.
* ``R(theta)`` rotates the (common, yaw) plane; the exploration found the two
  loud modes to be a collective/yaw mixture.  ``theta`` is reported canonically
  in ``(-pi/4, pi/4]`` (:meth:`Params.canonical`).
* ``CAR2`` is the white-noise-driven damped oscillator
  ``x'' + 2 zeta w0 x' + w0^2 x = xi``, ``w0 = 2 pi f0``, ``sigma_osc`` its
  stationary std.  It is flat below ``f0`` and ``f^-4`` above, where a sum of
  Lorentzians is capped at ``f^-2`` — the measured mode spectra are flat below
  ~0.3 Hz and then fall at -23 to -61 dB/decade, which is why the second OU of
  round 2 was replaced (round 2's michaels fit put 7.8x too much variance below
  0.2 Hz and its ACF decayed with a 2.5 s time constant against a real 0.25 s;
  round 3's matched every mode to 11 % in 0.2-1 Hz).  With ``zeta`` free it is
  also the optional per-rig resonance: ``zeta ~ 1`` a Matern-3/2-like shoulder,
  ``zeta < 1`` the MikroKopter's 3 Hz attitude-loop peak, ``zeta > 1``
  overdamped (an M100's featureless power law).
* ``e(t)``: the MEASUREMENT process, one OU per rotor with shared
  ``(tau_e, sigma_e)``, independent across rotors.  Round 4 replaced round 3's
  white ``sigma_w`` because the artefacts it has to absorb are COLOURED and
  incoherent — DREGON's 45 Hz sample-and-hold, michaels' hold-resampling and
  interpolation residue, per-rotor ESC jitter — and a white term cannot absorb
  them, so in round 3 the shaft modes did it instead and DREGON's rotor
  cross-correlation came out at 0.46 against a real 0.17.  It is labelled
  "measurement" so a training sampler can drop it and keep the shaft.
* ``delta_flight = 1 c + r``, one draw per flight, with ``c ~ N(0, s_c^2)``
  COMMON to the four rotors and ``r ~ N(0, diag(s_r^2))`` per rotor.  The frozen
  ``rotor_var`` pools all airborne samples about the *pooled* mean, so the
  between-flight level differences are part of what is scored; they are partly
  common (neurobem's gentle and aggressive flights differ by ~41 rev/s on all
  four rotors at once) and partly per-rotor (michaels' rotor 0 carries 2.5x its
  neighbours' variance because its own trim moves 17 rev/s), and a model with
  only the per-rotor part manufactures rotor spreads no real flight has — which
  is what cost round 4 a third of neurobem's samples to the airborne rule.

THE LIKELIHOOD IS NOW EXACT.  Rounds 1-3 used a Whittle (frequency-domain)
approximation, which weights every Rayleigh bin of a linear grid equally and so
spent 88 % of its evidence on the top two octaves; it also had to be told which
band to trust, and with only the periodogram to look at it kept parking
unresolvable slow power.  Round 4 evaluates the exact Gaussian likelihood of
the state-space model with a STEADY-STATE Kalman filter:

* every airborne segment is cut into :data:`BLOCK_S`-second blocks that overlap
  :data:`BURN_S` seconds on the left, and the innovations of those burn-in
  samples are excluded from the NLL, so every scored innovation comes from a
  filter that has already forgotten its initial condition;
* all blocks of one shape are batched into one ``(B, T, 4)`` tensor and driven
  through one batched recursion ``x_{t+1} = (Phi - K H) x_t + K y_t``, with
  ``e_t = y_t - H x_t``;
* ``K`` and the innovation covariance ``S`` come from the discrete Riccati
  fixed point (:func:`steady_state_gain`), solved by iterating the recursion in
  torch so the gain is differentiable in the parameters;
* ``NLL = 0.5 sum_t [log det S + e_t' S^-1 e_t]`` over the scored samples.

``y`` is each block demeaned per rotor, which is the simpler of the two ways to
remove the unknown per-block level (the other being a diffuse state).  It makes
the likelihood that of a rank-4 projection of the block rather than the block
itself, and it is why ``mu`` and ``s`` are ESTIMATED from the airborne means
rather than fitted: they do not enter the likelihood at all.

Both components and the measurement process are defined by their EXACT discrete
state space on the fit grid — ``Phi = expm(A dt)``, ``Q = P - Phi P Phi^T`` with
``P`` the continuous stationary covariance, exact because the process is
stationary — and the spectrum in :meth:`Params.spectral_matrix` is computed from
the same ``(Phi, Q, H)``, so the sampler, the likelihood and the spectrum cannot
drift apart.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.signal import butter, lfilter, lfiltic, sosfiltfilt
from scipy.signal.windows import hann

from experiments.rps_traj.data import (
    ANTIALIAS_ORDER,
    ERODE_S,
    RATE_HZ,
    Flight,
    airborne_segments,
)
from experiments.rps_traj.flight import FlightPhaseRanges, wrap_airborne
from experiments.rps_traj.stats import Sampler
from tracking.rotors import MIXER, NUM_ROTORS

#: Likelihood block length (s) and the burn-in each block overlaps its
#: predecessor by.  A block's first :data:`BURN_S` seconds are filtered but NOT
#: scored, so every scored innovation comes from a converged filter; the first
#: block of a segment starts at the segment start and has no burn-in (its own
#: initial condition is the stationary prior, which is exact).
BLOCK_S = 30.0
BURN_S = 5.0

#: A block must contribute at least this many seconds of SCORED samples.
MIN_SCORED_S = 5.0

#: Blocks are padded up to a multiple of this many samples (never beyond a full
#: block) before batching.  Grouping by EXACT length gives one group per segment
#: — real segments differ by a sample or two — which cost neurobem 18 s per
#: likelihood evaluation in 138 single-block groups; padding everything to the
#: longest block instead wastes the recursion's steps on rigs whose segments are
#: short, and the per-step cost grows with the batch, so quantising splits the
#: difference.  Measured on michaels: 155 ms per evaluation at 256, 271 ms at
#: 512, 293 ms at 1024.
PAD_QUANTUM = 256

#: Per-rig likelihood sample rate (Hz).  This replaced the round-3 fit BAND: a
#: band told the Whittle fit which frequencies to believe, whereas an exact
#: time-domain likelihood is a statement about a sampled series, so the honest
#: knob is the rate at which the telemetry is believable.  michaels is logged at
#: 29.41 Hz, so its 100 Hz grid is interpolation above ~12 Hz and its spectra
#: dive two decades at the 25-30 Hz interpolation null; fitting it at 25 Hz says
#: exactly that and nothing more.  Every other rig is genuine ESC feedback at
#: >= 100 Hz native.
FIT_RATE_HZ: dict[str, float] = {"michaels": 25.0, "default": 100.0}

#: MAP priors: (median, sd) of a log-normal on tau_slow / the sigmas /
#: (tau_e, sigma_e).
TAU_SLOW_PRIOR = (2.0, 1.5)
SIGMA_PRIOR = (1.0, 3.0)
TAU_E_PRIOR = (0.05, 2.0)
SIGMA_E_PRIOR = (0.1, 3.0)

#: HARD parameter ranges, imposed by parametrisation rather than by the
#: optimiser's box.  ``f0`` spans the resolvable range; ``zeta`` reaches 3 so a
#: mode can be firmly overdamped and 0.2 so it can be as sharp as the
#: MikroKopter's Q ~ 6 attitude loop; ``tau_slow`` is bounded BELOW by the
#: oscillator's corner time ``tau_c = 1/(2 pi f0)``, which orders the two
#: components without a label-switching gauge, and ABOVE by 10 s, the longest
#: lag the frozen ACF scores.  Slower variation is ``delta_flight``'s job, and
#: that term is estimated rather than fitted, so the cap removes a double count
#: rather than capability.
F0_MIN_HZ, F0_MAX_HZ = 0.05, 20.0
ZETA_MIN, ZETA_MAX = 0.2, 3.0
TAU_SLOW_MAX_S = 10.0

#: HARD range on the measurement process' time constant (s).  It models
#: sample-and-hold, quantisation and per-rotor ESC jitter, whose correlation
#: times are milliseconds to tens of milliseconds (michaels' median hold is
#: 68 ms, DREGON's 19 ms); a second would be shaft behaviour, not measurement.
#: The bound is also what keeps the filter well posed: the measurement process
#: is the only term with no observation-noise floor under it, so its
#: contribution to the innovation covariance is its own one-step innovation
#: variance ``sigma_e^2 (1 - exp(-2 dt/tau_e))``, which vanishes as
#: ``tau_e -> inf`` and leaves ``S = H P H'`` singular — the optimiser found
#: exactly that on the first try.
TAU_E_MIN_S, TAU_E_MAX_S = 1e-3, 1.0

#: Numerical box for the unconstrained optimiser vector (safety, not modelling).
_BOUNDS = {
    "theta": (-np.pi, np.pi),
    "u_slow": (-12.0, 12.0),
    "v_f0": (-12.0, 12.0),
    "v_zeta": (-12.0, 12.0),
    "log_sigma": (np.log(1e-5), np.log(1e4)),
    "v_tau_e": (-12.0, 12.0),
    "log_sigma_e": (np.log(1e-3), np.log(1e4)),
}

#: Floor on the per-flight offset std (rev/s) and the flight count below which
#: the offset term is switched off entirely.
S_FLOOR = 1e-3
S_MIN_FLIGHTS = 3

#: Floor under ``sigma_e``.  The measurement process is the ONLY term that makes
#: the innovation covariance ``S = H P H^T`` full rank (there is no separate
#: observation-noise matrix: the measurement process lives in the state), so it
#: must stay positive.
SIGMA_E_FLOOR = 1e-6

#: Riccati fixed point: relative tolerance and iteration cap.  The tolerance is
#: RELATIVE because the rigs' variances span 0.4 to 3000 (rev/s)^2, so an
#: absolute 1e-9 would be unreachable on neurobem and trivial on michaels.
RICCATI_TOL = 1e-9
RICCATI_MAX_ITER = 2000

#: Number of parameters, and the optimiser vector's length.
N_PARAMS = 32
N_FIT_PARAMS = 23

#: State dimension: 4 slow OUs, 4 oscillator pairs, 4 measurement OUs.
N_STATES = 16

_TWO_PI = 2.0 * np.pi


def fit_rate_hz(rig: str) -> float:
    """The likelihood sample rate of ``rig`` — exact id, then prefix, then
    default."""
    if rig in FIT_RATE_HZ:
        return FIT_RATE_HZ[rig]
    for key, rate in FIT_RATE_HZ.items():
        if key != "default" and rig.startswith(key):
            return rate
    return FIT_RATE_HZ["default"]


def rotation(theta: float) -> np.ndarray:
    """``R(theta)``: rotation by ``theta`` in the (common, yaw) plane."""
    c, s = float(np.cos(theta)), float(np.sin(theta))
    r = np.eye(NUM_ROTORS)
    r[0, 0], r[0, 3] = c, -s
    r[3, 0], r[3, 3] = s, c
    return r


def decimate(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Anti-alias and decimate ``(..., n)`` to ``fs_out`` by an integer factor.

    The same shape of filter the campaign's common grid already uses — a
    zero-phase Butterworth of order :data:`data.ANTIALIAS_ORDER` — with the
    corner at ``0.4 * fs_out``, then every ``fs_in / fs_out``-th sample.
    """
    factor = int(round(float(fs_in) / float(fs_out)))
    if factor <= 1:
        return np.ascontiguousarray(x)
    sos = butter(ANTIALIAS_ORDER, 0.4 * float(fs_out), btype="low", fs=float(fs_in), output="sos")
    return np.ascontiguousarray(sosfiltfilt(sos, x, axis=-1)[..., ::factor])


# ─── bounded transforms ───────────────────────────────────────────────────────


def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _logit(p: np.ndarray | float, eps: float = 1e-9) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(q / (1.0 - q))


def f0_from_v(v: np.ndarray | float) -> np.ndarray:
    """Oscillator centre frequency in ``(F0_MIN_HZ, F0_MAX_HZ)``."""
    return F0_MIN_HZ + (F0_MAX_HZ - F0_MIN_HZ) * _sigmoid(v)


def v_from_f0(f0: np.ndarray | float) -> np.ndarray:
    return _logit((np.asarray(f0, dtype=np.float64) - F0_MIN_HZ) / (F0_MAX_HZ - F0_MIN_HZ))


def zeta_from_v(v: np.ndarray | float) -> np.ndarray:
    """Damping ratio in ``(ZETA_MIN, ZETA_MAX)``; above 1 is overdamped."""
    return ZETA_MIN + (ZETA_MAX - ZETA_MIN) * _sigmoid(v)


def v_from_zeta(zeta: np.ndarray | float) -> np.ndarray:
    return _logit((np.asarray(zeta, dtype=np.float64) - ZETA_MIN) / (ZETA_MAX - ZETA_MIN))


def tau_e_from_v(v: np.ndarray | float) -> float:
    """Measurement time constant in ``(TAU_E_MIN_S, TAU_E_MAX_S)``."""
    return float(TAU_E_MIN_S + (TAU_E_MAX_S - TAU_E_MIN_S) * _sigmoid(v))


def v_from_tau_e(tau_e: float) -> float:
    return float(_logit((float(tau_e) - TAU_E_MIN_S) / (TAU_E_MAX_S - TAU_E_MIN_S)))


def corner_tau_s(f0: np.ndarray | float) -> np.ndarray:
    """``tau_c = 1 / (2 pi f0)`` — the oscillator's corner time."""
    return 1.0 / (_TWO_PI * np.asarray(f0, dtype=np.float64))


def tau_slow_from_u(u: np.ndarray | float, f0: np.ndarray | float) -> np.ndarray:
    """``tau_c + (TAU_SLOW_MAX_S - tau_c) * sigmoid(u)`` — always the SLOWER of
    the two components, and never slower than :data:`TAU_SLOW_MAX_S`."""
    tau_c = corner_tau_s(f0)
    return tau_c + (TAU_SLOW_MAX_S - tau_c) * _sigmoid(u)


def u_from_tau_slow(tau: np.ndarray | float, f0: np.ndarray | float) -> np.ndarray:
    tau_c = corner_tau_s(f0)
    return _logit((np.asarray(tau, dtype=np.float64) - tau_c) / (TAU_SLOW_MAX_S - tau_c))


# ─── the components, as exact discrete state spaces ───────────────────────────


def ou_state_space(
    sigma: float, tau: float, fs: float = RATE_HZ
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(Phi, Q, P)`` of an OU on the ``fs`` grid; 1 state, ``H = [1]``."""
    a = float(np.exp(-1.0 / (float(fs) * float(tau))))
    p = np.array([[float(sigma) ** 2]])
    phi = np.array([[a]])
    return phi, p - phi @ p @ phi.T, p


def car2_continuous(f0: float, zeta: float) -> np.ndarray:
    """``A`` of ``x'' + 2 zeta w0 x' + w0^2 x = xi`` on ``X = [x, x']``."""
    w0 = _TWO_PI * float(f0)
    return np.array([[0.0, 1.0], [-w0 * w0, -2.0 * float(zeta) * w0]])


def car2_state_space(
    sigma: float, f0: float, zeta: float, fs: float = RATE_HZ
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(Phi, Q, P)`` of a CAR2 on the ``fs`` grid; 2 states, ``H = [1, 0]``.

    ``P`` is the continuous stationary covariance, diagonal for this system:
    ``diag(sigma^2, (w0 sigma)^2)``, from ``Var(x) = q/(4 zeta w0^3)`` and
    ``Var(x') = q/(4 zeta w0)`` with ``q = 4 zeta w0^3 sigma^2``.
    ``Phi = expm(A dt)``; ``Q = P - Phi P Phi^T`` is exact because the process
    is stationary, so no Van Loan block exponential is needed.
    """
    w0 = _TWO_PI * float(f0)
    phi = expm(car2_continuous(f0, zeta) / float(fs))
    p = np.diag([float(sigma) ** 2, (w0 * float(sigma)) ** 2])
    return phi, p - phi @ p @ phi.T, p


def _psd_from_1state(phi: float, q: float, cosw: np.ndarray, fs: float) -> np.ndarray:
    return (2.0 / fs) * q / (1.0 - 2.0 * phi * cosw + phi * phi)


def _psd_from_2state(
    phi: np.ndarray, q: np.ndarray, cosw: np.ndarray, sinw: np.ndarray, fs: float
) -> np.ndarray:
    """One-sided grid PSD of a 2-state space with ``H = [1, 0]``.

    ``H (zI - Phi)^-1 = [z - Phi11, Phi01] / det``, so the quadratic form is
    real elementwise arithmetic in ``cos w`` and ``sin w``.
    """
    tr = phi[0, 0] + phi[1, 1]
    det = phi[0, 0] * phi[1, 1] - phi[0, 1] * phi[1, 0]
    cos2w = 2.0 * cosw * cosw - 1.0
    sin2w = 2.0 * sinw * cosw
    den = (cos2w - tr * cosw + det) ** 2 + (sin2w - tr * sinw) ** 2
    num = (
        q[0, 0] * (1.0 - 2.0 * phi[1, 1] * cosw + phi[1, 1] ** 2)
        + 2.0 * q[0, 1] * phi[0, 1] * (cosw - phi[1, 1])
        + q[1, 1] * phi[0, 1] ** 2
    )
    return (2.0 / fs) * num / den


def state_space_psd(
    f: np.ndarray, phi: np.ndarray, q: np.ndarray, fs: float = RATE_HZ
) -> np.ndarray:
    """``S(f)`` of a state space with ``H = e_0``, straight from the definition:
    ``(2/fs) H (e^{iw} I - Phi)^-1 Q (e^{-iw} I - Phi^T)^-1 H^T``."""
    f = np.atleast_1d(np.asarray(f, dtype=np.float64))
    phi = np.atleast_2d(phi)
    eye = np.eye(phi.shape[0])
    out = np.empty(f.size)
    for k, freq in enumerate(f):
        z = np.exp(1j * _TWO_PI * freq / float(fs))
        g = np.linalg.solve(z * eye - phi, eye)[0]
        out[k] = (2.0 / float(fs)) * float(np.real(g @ q @ np.conj(g)))
    return out


def ou_psd_grid(f: np.ndarray, sigma: float, tau: float, fs: float = RATE_HZ) -> np.ndarray:
    """One-sided PSD of the SAMPLED OU on an ``fs`` grid (exact)."""
    phi, q, _ = ou_state_space(sigma, tau, fs)
    cosw = np.cos(_TWO_PI * np.asarray(f, dtype=np.float64) / float(fs))
    return _psd_from_1state(float(phi[0, 0]), float(q[0, 0]), cosw, float(fs))


def car2_psd_grid(
    f: np.ndarray, sigma: float, f0: float, zeta: float, fs: float = RATE_HZ
) -> np.ndarray:
    """One-sided PSD of the SAMPLED CAR2 on an ``fs`` grid (exact).

    Flat below ``f0``, ``f^-4`` above, with a peak when ``zeta < 1/sqrt(2)``.
    """
    phi, q, _ = car2_state_space(sigma, f0, zeta, fs)
    w = _TWO_PI * np.asarray(f, dtype=np.float64) / float(fs)
    return _psd_from_2state(phi, q, np.cos(w), np.sin(w), float(fs))


# ─── parameters ───────────────────────────────────────────────────────────────


@dataclass
class Params:
    """The 32 parameters of the new model.

    ``mu`` (4) rev/s in mixer rotor order; ``theta`` rad; per mode
    ``tau_slow`` (s, in ``(tau_c, TAU_SLOW_MAX_S)``), ``sigma_slow`` (rev/s),
    ``f0`` (Hz), ``zeta``, ``sigma_osc`` (rev/s); the measurement process'
    ``tau_e`` (s) and ``sigma_e`` (rev/s), shared across rotors; the per-flight
    offset ``delta_flight = 1 c + r`` with ``s_c`` rev/s (std of the common
    level ``c``) and ``s_r`` (4) rev/s (std of the per-rotor part ``r``).  Every
    ``sigma`` is a STATIONARY std, so a mode's variance is
    ``sigma_slow^2 + sigma_osc^2`` and a rotor's measurement variance is
    ``sigma_e^2``.
    """

    mu: np.ndarray
    theta: float
    tau_slow: np.ndarray
    sigma_slow: np.ndarray
    f0: np.ndarray
    zeta: np.ndarray
    sigma_osc: np.ndarray
    tau_e: float
    sigma_e: float
    s_c: float
    s_r: np.ndarray

    def __post_init__(self) -> None:
        for name in ("mu", "tau_slow", "sigma_slow", "f0", "zeta", "sigma_osc", "s_r"):
            setattr(
                self,
                name,
                np.asarray(getattr(self, name), dtype=np.float64).reshape(NUM_ROTORS),
            )
        self.theta = float(self.theta)
        self.tau_e = float(self.tau_e)
        self.sigma_e = float(self.sigma_e)
        self.s_c = float(self.s_c)

    # ── derived ──
    @property
    def corner_tau(self) -> np.ndarray:
        """``1 / (2 pi f0)`` — the oscillator corner time, below ``tau_slow``."""
        return corner_tau_s(self.f0)

    @property
    def mode_var(self) -> np.ndarray:
        """Stationary variance of each ``v_i`` (before the rotation)."""
        return self.sigma_slow**2 + self.sigma_osc**2

    @property
    def mixing(self) -> np.ndarray:
        """``A = M R(theta)``, mapping ``v`` onto rotor speeds."""
        return MIXER @ rotation(self.theta)

    def rotor_var(self, *, include_offset: bool = True) -> np.ndarray:
        """Analytic per-rotor variance of stationary airborne samples.

        With ``include_offset`` (the default) this is what the frozen
        ``rotor_var`` statistic measures: it pools every airborne sample of
        every flight about the POOLED mean, so the per-flight offset shows up as
        variance.  A SINGLE sampled flight carries one offset draw, hence one
        constant level, and its own sample variance converges on
        ``include_offset=False`` instead.
        """
        a = self.mixing
        out = (a * a) @ self.mode_var + self.sigma_e**2
        if include_offset:
            out = out + self.s_c**2 + self.s_r**2
        return out

    def component_state_spaces(
        self, fs: float = RATE_HZ
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """``[(Phi, Q, P)] * 8``: the slow OU then the oscillator of each mode."""
        out = [
            ou_state_space(float(self.sigma_slow[i]), float(self.tau_slow[i]), fs)
            for i in range(NUM_ROTORS)
        ]
        out += [
            car2_state_space(float(self.sigma_osc[i]), float(self.f0[i]), float(self.zeta[i]), fs)
            for i in range(NUM_ROTORS)
        ]
        return out

    def component_psd(self, f: np.ndarray, fs: float = RATE_HZ) -> np.ndarray:
        """``(4, F)`` PSD of each ``v_i`` = its OU plus its oscillator."""
        f = np.atleast_1d(np.asarray(f, dtype=np.float64))
        return np.stack(
            [
                ou_psd_grid(f, float(self.sigma_slow[i]), float(self.tau_slow[i]), fs)
                + car2_psd_grid(
                    f, float(self.sigma_osc[i]), float(self.f0[i]), float(self.zeta[i]), fs
                )
                for i in range(NUM_ROTORS)
            ],
            axis=0,
        )

    def measurement_psd(self, f: np.ndarray, fs: float = RATE_HZ) -> np.ndarray:
        """``(F,)`` PSD of one rotor's measurement process."""
        return ou_psd_grid(f, self.sigma_e, self.tau_e, fs)

    def spectral_matrix(self, f: np.ndarray, fs: float = RATE_HZ) -> np.ndarray:
        """``(F, 4, 4)`` one-sided rotor cross-spectrum at frequencies ``f``.

        ``A diag(S_v) A^T + S_e I``: the measurement process is independent
        across rotors, so it only adds to the diagonal.  The per-flight offset
        is a DC term and does not appear (blocks are demeaned).
        """
        modes = self.component_psd(f, fs).T  # (F, 4)
        a = self.mixing
        out = np.einsum("ji,fi,ki->fjk", a, modes, a)
        out += np.eye(NUM_ROTORS) * self.measurement_psd(f, fs)[:, None, None]
        return out

    def mode_acf(self, lag_samples: np.ndarray, fs: float = RATE_HZ) -> np.ndarray:
        """``(4, L)`` autocorrelation of the MODE series at integer lags.

        ``m = M^T w / 4 = M^T (mu + delta) / 4 + R v + M^T e / 4``, so mode
        ``i``'s covariance is ``(R diag(C_v(lag)) R^T)_ii`` plus
        ``C_e(lag) / 4`` (``M^T M / 16 = I / 4``).  Component covariances come
        from the same state spaces as everything else: ``C(k) = H Phi^k P H^T``.
        """
        lags = np.atleast_1d(np.asarray(lag_samples, dtype=np.int64))
        every = np.concatenate([[0], lags])
        spaces = self.component_state_spaces(fs)
        cov = np.empty((NUM_ROTORS, every.size))
        for i in range(NUM_ROTORS):
            for j, k in enumerate(every):
                total = 0.0
                for phi, _, p in (spaces[i], spaces[i + NUM_ROTORS]):
                    total += float((np.linalg.matrix_power(phi, int(k)) @ p)[0, 0])
                cov[i, j] = total
        a_e = float(np.exp(-1.0 / (float(fs) * self.tau_e)))
        meas = self.sigma_e**2 * a_e ** every.astype(np.float64) / 4.0
        r = rotation(self.theta)
        mode_cov = np.einsum("ij,jl->il", r * r, cov) + meas[None, :]
        return mode_cov[:, 1:] / mode_cov[:, 0][:, None]

    def mode_var_in_window(self, n_samples: int, fs: float = RATE_HZ) -> np.ndarray:
        """``(4,)`` expected per-MODE variance of an ``n_samples`` window after
        its mean is removed — what a per-segment-demeaned statistic can see.

        ``E[var] = C(0) - Var(mean) = C(0) - n^-2 [n C(0) + 2 sum_k (n-k) C(k)]``,
        exact on the grid for any component.
        """
        n = int(n_samples)
        if n <= 1:
            return np.zeros(NUM_ROTORS)
        lags = np.arange(1, n)
        acf = self.mode_acf(lags, fs)
        r = rotation(self.theta)
        zero = (r * r) @ self.mode_var + self.sigma_e**2 / 4.0
        weights = (n - lags) / float(n * n)
        mean_var = zero * (1.0 / n + 2.0 * (acf * weights).sum(axis=1))
        return np.maximum(zero - mean_var, 0.0)

    # ── gauge ──
    def canonical(self) -> Params:
        """Equivalent parameters with ``theta`` in ``(-pi/4, pi/4]``.

        ``theta -> theta - pi/2`` swaps the roles of mode 0 and mode 3 (one of
        the two rotated columns changes sign, which a zero-mean Gaussian cannot
        see).  All FIVE per-mode parameters move together, which keeps
        ``tau_slow > tau_c`` intact.  ``s`` and the measurement process are
        untouched: both are per rotor, outside the rotation.
        """
        k = int(np.floor((self.theta + np.pi / 4.0) / (np.pi / 2.0)))
        order = [3, 1, 2, 0] if k % 2 else [0, 1, 2, 3]
        return Params(
            mu=self.mu,
            theta=self.theta - k * (np.pi / 2.0),
            tau_slow=self.tau_slow[order],
            sigma_slow=self.sigma_slow[order],
            f0=self.f0[order],
            zeta=self.zeta[order],
            sigma_osc=self.sigma_osc[order],
            tau_e=self.tau_e,
            sigma_e=self.sigma_e,
            s_c=self.s_c,
            s_r=self.s_r,
        )

    # ── (de)serialisation ──
    def to_vector(self) -> np.ndarray:
        """The 31-dim unconstrained vector ``[log mu, theta, u_slow,
        log sigma_slow, v_f0, v_zeta, log sigma_osc, v_tau_e, log sigma_e,
        log s]``.

        ``u_slow``, ``v_f0`` and ``v_zeta`` are the bounded transforms' logits,
        so ANY vector maps back to parameters inside their ranges.  ``s`` is
        floored at :data:`S_FLOOR`, so a rig fitted with the offset term
        switched off (``s = 0``) round-trips to ``s = 1e-3`` rev/s.
        """
        return np.concatenate(
            [
                np.log(self.mu),
                [self.theta],
                u_from_tau_slow(self.tau_slow, self.f0),
                np.log(self.sigma_slow),
                v_from_f0(self.f0),
                v_from_zeta(self.zeta),
                np.log(self.sigma_osc),
                [
                    v_from_tau_e(self.tau_e),
                    np.log(max(self.sigma_e, SIGMA_E_FLOOR)),
                    np.log(max(self.s_c, S_FLOOR)),
                ],
                np.log(np.maximum(self.s_r, S_FLOOR)),
            ]
        )

    @classmethod
    def from_vector(cls, v: np.ndarray) -> Params:
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        if v.size != N_PARAMS:
            raise ValueError(f"expected a {N_PARAMS}-dim parameter vector, got {v.size}")
        f0 = f0_from_v(v[13:17])
        return cls(
            mu=np.exp(v[0:4]),
            theta=float(v[4]),
            tau_slow=tau_slow_from_u(v[5:9], f0),
            sigma_slow=np.exp(v[9:13]),
            f0=f0,
            zeta=zeta_from_v(v[17:21]),
            sigma_osc=np.exp(v[21:25]),
            tau_e=tau_e_from_v(v[25]),
            sigma_e=float(np.exp(v[26])),
            s_c=float(np.exp(v[27])),
            s_r=np.exp(v[28:32]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mu": self.mu.tolist(),
            "theta": self.theta,
            "tau_slow": self.tau_slow.tolist(),
            "sigma_slow": self.sigma_slow.tolist(),
            "f0": self.f0.tolist(),
            "zeta": self.zeta.tolist(),
            "sigma_osc": self.sigma_osc.tolist(),
            "corner_tau": self.corner_tau.tolist(),
            "tau_e": self.tau_e,
            "sigma_e": self.sigma_e,
            "s_c": self.s_c,
            "s_r": self.s_r.tolist(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, allow_nan=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Params:
        return cls(
            mu=payload["mu"],
            theta=payload["theta"],
            tau_slow=payload["tau_slow"],
            sigma_slow=payload["sigma_slow"],
            f0=payload["f0"],
            zeta=payload["zeta"],
            sigma_osc=payload["sigma_osc"],
            tau_e=payload["tau_e"],
            sigma_e=payload["sigma_e"],
            s_c=payload["s_c"],
            s_r=payload["s_r"],
        )

    @classmethod
    def from_json(cls, text: str | bytes) -> Params:
        return cls.from_dict(json.loads(text))

    # ── sampling ──
    def _component_paths(self, n: int, rng: np.random.Generator, fs: float) -> np.ndarray:
        """``(4, n)`` realisation of ``v``: each mode's OU plus its oscillator."""
        out = np.empty((NUM_ROTORS, n))
        for i in range(NUM_ROTORS):
            out[i] = _ou_path(float(self.sigma_slow[i]), float(self.tau_slow[i]), n, rng, fs)
        for i in range(NUM_ROTORS):
            phi, q, p = car2_state_space(
                float(self.sigma_osc[i]), float(self.f0[i]), float(self.zeta[i]), fs
            )
            out[i] += _car2_path(phi, q, p, n, rng)
        return out

    def sample_airborne(
        self,
        n_samples: int,
        rng: np.random.Generator,
        fs: float = RATE_HZ,
        *,
        offset: np.ndarray | None = None,
        clip: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """``(4, n)`` stationary airborne rotor speeds, one flight offset draw."""
        if int(n_samples) <= 0:
            return np.zeros((NUM_ROTORS, 0))
        return self._draw(int(n_samples), rng, fs, offset, clip)[0]

    def _draw(
        self,
        n_samples: int,
        rng: np.random.Generator,
        fs: float,
        offset: np.ndarray | None,
        clip: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(w, delta_flight)``; ``offset`` overrides the drawn offset.

        The offset is drawn from ``rng`` even when overridden, so every other
        random stream is byte-identical to the plain sampler.
        """
        n = int(n_samples)
        v = self._component_paths(n, rng, float(fs))
        drawn = float(rng.standard_normal()) * self.s_c + (
            rng.standard_normal(NUM_ROTORS) * self.s_r
        )
        delta = drawn if offset is None else np.asarray(offset, dtype=np.float64)
        out = (self.mu + delta)[:, None] + self.mixing @ v
        for j in range(NUM_ROTORS):
            out[j] += _ou_path(self.sigma_e, self.tau_e, n, rng, float(fs))
        if clip is not None:
            np.clip(out, clip[0], clip[1], out=out)
        return out, delta

    def sampler(
        self,
        fs: float = RATE_HZ,
        *,
        antithetic_offsets: bool = False,
        clip: tuple[float, float] | None = None,
    ) -> Sampler:
        """The frozen sampler protocol: ``sampler(n, rng) -> (4, n)`` rev/s.

        With ``antithetic_offsets`` the closure pairs consecutive calls: the
        second flight of each pair reuses the NEGATED offset of the first.  That
        is Monte-Carlo variance reduction, not a different model — each flight's
        offset keeps its marginal law, so ``rotor_var``,
        ``acf`` and ``xcorr`` are unchanged in distribution — but the offsets
        over an EVEN number of sampled flights cancel exactly, so the pooled
        model mean is ``mu`` instead of carrying an ``s / sqrt(N)`` error (worth
        26.8 rev/s on neurobem at ``n_rep = 5``, which no affordable ``n_rep``
        removes, while the baseline has no offset term and pays nothing).  The
        cost is that the sampler becomes stateful across calls — still
        deterministic in call order, so ``stats_from_samples`` stays
        reproducible — which is why it is off by default and never used by
        :meth:`NewFit.full_flight`.

        ``clip`` is the rig's ESC floor and ceiling; see :class:`NewFit`.
        """
        if not antithetic_offsets:

            def _sample(n_samples: int, rng: np.random.Generator) -> np.ndarray:
                return self.sample_airborne(n_samples, rng, fs=fs, clip=clip)

            return _sample

        state: dict[str, Any] = {"calls": 0, "last": None}

        def _antithetic(n_samples: int, rng: np.random.Generator) -> np.ndarray:
            if int(n_samples) <= 0:
                return np.zeros((NUM_ROTORS, 0))
            paired = state["calls"] % 2 == 1
            override = -state["last"] if paired and state["last"] is not None else None
            out, delta = self._draw(int(n_samples), rng, fs, override, clip)
            state["calls"] += 1
            state["last"] = None if paired else delta
            return out

        return _antithetic


def _psd_matrix_sqrt(m: np.ndarray) -> np.ndarray:
    """Symmetric square root of a small PSD matrix, robust to rank deficiency.

    ``Q`` of a CAR2 is driven through one channel, so at small ``dt`` it is
    numerically rank-1 and Cholesky fails; an eigen-decomposition with the
    eigenvalues clipped at zero does not.
    """
    vals, vecs = np.linalg.eigh(np.asarray(m, dtype=np.float64))
    return vecs @ np.diag(np.sqrt(np.maximum(vals, 0.0))) @ vecs.T


def _ou_path(sigma: float, tau: float, n: int, rng: np.random.Generator, fs: float) -> np.ndarray:
    """``(n,)`` stationary OU realisation from its exact discrete recursion."""
    a = float(np.exp(-1.0 / (float(fs) * float(tau))))
    x0 = float(rng.standard_normal()) * float(sigma)
    drive = rng.standard_normal(n) * (float(sigma) * np.sqrt(max(1.0 - a * a, 0.0)))
    return lfilter([1.0], [1.0, -a], drive, zi=[a * x0])[0]


def _car2_path(
    phi: np.ndarray, q: np.ndarray, p: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """``(n,)`` stationary realisation of ``H x`` for a 2-state space.

    ``y_k = tr(Phi) y_{k-1} - det(Phi) y_{k-2} + u_k`` with
    ``u_k = eta0_{k-1} - Phi11 eta0_{k-2} + Phi01 eta1_{k-2}`` (Cayley-Hamilton
    on the 2-state recursion), so the path is one ``lfilter`` pass rather than a
    Python loop over samples.
    """
    x0 = _psd_matrix_sqrt(p) @ rng.standard_normal(2)
    eta = _psd_matrix_sqrt(q) @ rng.standard_normal((2, n))
    tr = float(phi[0, 0] + phi[1, 1])
    det = float(phi[0, 0] * phi[1, 1] - phi[0, 1] * phi[1, 0])

    y = np.empty(n)
    y[0] = x0[0]
    if n == 1:
        return y
    y[1] = float(phi[0, 0]) * x0[0] + float(phi[0, 1]) * x0[1] + eta[0, 0]
    if n == 2:
        return y
    u = eta[0, 1:-1] - float(phi[1, 1]) * eta[0, :-2] + float(phi[0, 1]) * eta[1, :-2]
    zi = lfiltic([1.0], [1.0, -tr, det], y=[y[1], y[0]])
    y[2:] = lfilter([1.0], [1.0, -tr, det], u, zi=zi)[0]
    return y


# ─── likelihood blocks ────────────────────────────────────────────────────────


@dataclass
class _Batch:
    """Every likelihood block sharing a burn-in length, as ONE ``(B, T, 4)``
    tensor.

    Blocks are zero-PADDED to the group's longest length and ``mask`` is 1
    exactly on the samples that count: past the burn-in and inside the block's
    real length.  Padding is what makes this one recursion instead of many —
    real airborne segments differ in length by a sample or two, so grouping by
    exact length gave neurobem 138 groups of one block each and cost 18 s per
    likelihood evaluation against 0.3 s batched.  The recursion runs over the
    padded tail (driven by zeros, so it just decays) and the mask keeps it out
    of the NLL.
    """

    y: torch.Tensor  # (B, T, 4), demeaned per block per rotor, zero-padded
    mask: torch.Tensor  # (B, T) 1.0 on scored samples
    burn: int

    @property
    def n_blocks(self) -> int:
        return int(self.y.shape[0])

    @property
    def n_scored(self) -> int:
        return int(self.mask.sum().item())


def likelihood_batches(
    flights: Sequence[Flight], fit_rate: float, fs: float = RATE_HZ
) -> list[_Batch]:
    """Every airborne segment cut into overlapping, demeaned blocks.

    Segments come from the frozen airborne rule on the 100 Hz grid, are then
    decimated to ``fit_rate`` (:func:`decimate`) and cut into
    :data:`BLOCK_S`-second blocks whose starts advance by
    ``BLOCK_S - BURN_S``; the first block of a segment has no burn-in and every
    later one discards its first :data:`BURN_S` seconds from the NLL.  A block
    must contribute at least :data:`MIN_SCORED_S` of scored samples.  Blocks are
    grouped by burn-in only and padded to a common length, so a rig is at most
    two batched recursions.
    """
    block = int(round(BLOCK_S * fit_rate))
    burn = int(round(BURN_S * fit_rate))
    hop = block - burn
    min_scored = int(round(MIN_SCORED_S * fit_rate))

    groups: dict[tuple[int, int], list[np.ndarray]] = {}
    for flight in flights:
        rps = np.asarray(flight.rps, dtype=np.float64)
        for sl in airborne_segments(rps, fs):
            seg = rps[:, sl]
            if not np.isfinite(seg).all():
                continue
            seg = decimate(seg, fs, fit_rate)
            n = seg.shape[1]
            start = 0
            while start < n:
                stop = min(start + block, n)
                this_burn = 0 if start == 0 else burn
                if stop - start - this_burn < min_scored:
                    break
                chunk = seg[:, start:stop]
                bucket = min(-(-chunk.shape[1] // PAD_QUANTUM) * PAD_QUANTUM, block)
                groups.setdefault((this_burn, bucket), []).append(
                    chunk - chunk.mean(axis=1, keepdims=True)
                )
                if stop == n:
                    break
                start += hop

    batches: list[_Batch] = []
    for (burn_n, width), blocks in sorted(groups.items()):
        y = np.zeros((len(blocks), NUM_ROTORS, width))
        mask = np.zeros((len(blocks), width))
        for i, b in enumerate(blocks):
            y[i, :, : b.shape[1]] = b
            mask[i, burn_n : b.shape[1]] = 1.0
        batches.append(
            _Batch(
                y=torch.as_tensor(y, dtype=torch.float64).transpose(1, 2),
                mask=torch.as_tensor(mask, dtype=torch.float64),
                burn=burn_n,
            )
        )
    if not batches:
        raise ValueError("no usable likelihood blocks — every segment was too short or NaN")
    return batches


# ─── the exact likelihood ─────────────────────────────────────────────────────


def _unpack(vec: torch.Tensor) -> dict[str, torch.Tensor]:
    """The 23-dim optimiser vector → parameters, all inside their ranges."""
    f0 = F0_MIN_HZ + (F0_MAX_HZ - F0_MIN_HZ) * torch.sigmoid(vec[9:13])
    tau_c = 1.0 / (_TWO_PI * f0)
    return {
        "theta": vec[0],
        "tau_slow": tau_c + (TAU_SLOW_MAX_S - tau_c) * torch.sigmoid(vec[1:5]),
        "sigma_slow": torch.exp(vec[5:9]),
        "f0": f0,
        "zeta": ZETA_MIN + (ZETA_MAX - ZETA_MIN) * torch.sigmoid(vec[13:17]),
        "sigma_osc": torch.exp(vec[17:21]),
        "tau_e": TAU_E_MIN_S + (TAU_E_MAX_S - TAU_E_MIN_S) * torch.sigmoid(vec[21]),
        "sigma_e": torch.exp(vec[22]),
    }


def _state_space_torch(
    p: dict[str, torch.Tensor], dt: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(Phi, Q, P_stationary, H)`` of the full 16-state model at step ``dt``.

    State order: the 4 slow OUs, then the 4 oscillator pairs, then the 4
    per-rotor measurement OUs.  ``Phi`` and ``Q`` are block diagonal because
    every component is independent; all the coupling is in ``H``.
    """
    fs = 1.0 / float(dt)

    a_slow = torch.exp(-float(dt) / p["tau_slow"])  # (4,)
    p_slow = p["sigma_slow"] ** 2
    q_slow = p_slow * (1.0 - a_slow * a_slow)

    a_e = torch.exp(torch.as_tensor(-float(dt)) / p["tau_e"])
    p_e = p["sigma_e"] ** 2
    q_e = p_e * (1.0 - a_e * a_e)

    w0 = _TWO_PI * p["f0"]
    zero, one = torch.zeros_like(w0), torch.ones_like(w0)
    a_osc = torch.stack(
        [
            torch.stack([zero, one], dim=-1),
            torch.stack([-w0 * w0, -2.0 * p["zeta"] * w0], dim=-1),
        ],
        dim=-2,
    ) * float(dt)
    phi_osc = torch.linalg.matrix_exp(a_osc)  # (4, 2, 2)
    p_osc = torch.diag_embed(torch.stack([p["sigma_osc"] ** 2, (w0 * p["sigma_osc"]) ** 2], dim=-1))
    q_osc = p_osc - phi_osc @ p_osc @ phi_osc.transpose(-1, -2)

    def _blocks(scalars: torch.Tensor, mats: torch.Tensor, meas: torch.Tensor) -> torch.Tensor:
        return torch.block_diag(
            *[scalars[i].reshape(1, 1) for i in range(NUM_ROTORS)],
            *[mats[i] for i in range(NUM_ROTORS)],
            *[meas.reshape(1, 1) for _ in range(NUM_ROTORS)],
        )

    phi = _blocks(a_slow, phi_osc, a_e)
    q = _blocks(q_slow, q_osc, q_e)
    p_stat = _blocks(p_slow, p_osc, p_e)

    cos, sin = torch.cos(p["theta"]), torch.sin(p["theta"])
    mixer = torch.as_tensor(MIXER, dtype=phi.dtype)
    rot = torch.eye(NUM_ROTORS, dtype=phi.dtype).clone()
    rot = torch.stack(
        [
            torch.stack([cos, rot[0, 1], rot[0, 2], -sin]),
            rot[1],
            rot[2],
            torch.stack([sin, rot[3, 1], rot[3, 2], cos]),
        ]
    )
    mixing = mixer @ rot  # (4, 4) = A
    h_osc = torch.stack([mixing, torch.zeros_like(mixing)], dim=-1).reshape(NUM_ROTORS, 8)
    h = torch.cat([mixing, h_osc, torch.eye(NUM_ROTORS, dtype=phi.dtype)], dim=1)
    del fs
    return phi, q, p_stat, h


def steady_state_gain(
    phi: torch.Tensor, q: torch.Tensor, p_stat: torch.Tensor, h: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(K, S)`` of the steady-state one-step predictor.

    Iterates the discrete Riccati recursion
    ``P <- Phi (P - P H' S^-1 H P) Phi' + Q`` from the stationary covariance
    until ``||P_{k+1} - P_k||_F <= RICCATI_TOL * ||P_k||_F`` (at most
    :data:`RICCATI_MAX_ITER` steps), then returns
    ``S = H P H'`` and ``K = Phi P H' S^-1``.  The iteration runs inside the
    autograd graph, so the gain is differentiable in the parameters — which is
    the whole reason for iterating rather than calling a solver.

    There is no separate observation-noise matrix: the measurement process is
    part of the state, so ``S = H P H'`` and ``sigma_e > 0`` is what keeps it
    full rank.
    """
    p = p_stat
    for _ in range(RICCATI_MAX_ITER):
        s = h @ p @ h.T
        g = p @ h.T
        p_next = phi @ (p - g @ torch.linalg.solve(s, g.T)) @ phi.T + q
        p_next = 0.5 * (p_next + p_next.T)
        if torch.linalg.norm(p_next - p) <= RICCATI_TOL * torch.linalg.norm(p):
            p = p_next
            break
        p = p_next
    s = h @ p @ h.T
    # solve, not inv: K' solves S K' = (Phi P H')'
    return torch.linalg.solve(s, (phi @ p @ h.T).T).T, s


class _Predictor(torch.autograd.Function):
    """``x_{t+1} = F x_t + d_t``, ``p_t = H x_t``, with a HAND-WRITTEN adjoint.

    Autograd's own backward through this loop is what makes the exact
    likelihood expensive: it builds a few nodes per time step, and on
    neurobem's 27 760 sequential steps the backward cost 18.2 s of a 19.4 s
    gradient — 94 % of it, at ~160 us of pure graph-traversal overhead per node
    against microseconds of arithmetic.  The adjoint below is three lines of
    algebra and runs as ONE ``addmm`` per step, with the whole parameter
    gradient contracted in three ``einsum``s outside the loop:

        a_t = Gp_t H + a_{t+1} F,        a_T = 0
        dL/dd_t = a_{t+1}
        dL/dF   = sum_t a_{t+1}' x_t
        dL/dH   = sum_t Gp_t' x_t

    Everything is in ROW convention (states are rows of a ``(B, 16)`` block) and
    the time axis is leading so that ``x[t]`` is a contiguous slice and
    ``addmm`` can write straight into it.  ``F`` and ``d`` are passed in rather
    than ``Phi``, ``K``, ``H`` so that ordinary autograd still chains
    ``F = Phi - K H`` and ``d = y K'`` back to the parameters.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, drive: torch.Tensor, f: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        n_t, n_b = int(drive.shape[0]), int(drive.shape[1])
        f_t = f.T.contiguous()
        x = drive.new_zeros((n_t, n_b, N_STATES))
        for t in range(n_t - 1):
            torch.addmm(drive[t], x[t], f_t, out=x[t + 1])
        ctx.save_for_backward(x, f, h)
        return x @ h.T.contiguous()

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_p: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, f, h = ctx.saved_tensors
        n_t = int(x.shape[0])
        gph = (grad_p @ h).contiguous()  # (T, B, 16) = rows of H' Gp
        a = torch.empty_like(x)
        a[n_t - 1] = gph[n_t - 1]
        for t in range(n_t - 2, -1, -1):
            torch.addmm(gph[t], a[t + 1], f, out=a[t])
        grad_drive = torch.zeros_like(x)
        grad_drive[:-1] = a[1:]
        grad_f = torch.einsum("tbk,tbl->kl", a[1:], x[:-1])
        grad_h = torch.einsum("tbk,tbl->kl", grad_p, x)
        return grad_drive, grad_f, grad_h


def _batch_nll(
    batch: _Batch, phi: torch.Tensor, k: torch.Tensor, h: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """``0.5 sum [log det S + e' S^-1 e]`` over one batch's scored samples.

    The recursion is ``x_{t+1} = (Phi - K H) x_t + K y_t`` with ``x_0 = 0`` (the
    stationary prior mean), so the driving term ``K y`` is one batched matmul
    outside the loop and the loop itself is :class:`_Predictor`.  The
    innovations are assembled for every sample at once and weighted by the
    batch's mask, which drops both the burn-in and the padding.
    """
    y = batch.y  # (B, T, 4)
    f = phi - k @ h
    drive = (y @ k.T).transpose(0, 1).contiguous()  # (T, B, 16)
    pred = cast(torch.Tensor, _Predictor.apply(drive, f, h))  # (T, B, 4)
    e = y - pred.transpose(0, 1)

    chol = torch.linalg.cholesky(s)
    flat = (e * batch.mask[..., None]).reshape(-1, NUM_ROTORS).T
    quad = torch.sum(flat * torch.cholesky_solve(flat, chol))
    logdet = 2.0 * torch.sum(torch.log(torch.diagonal(chol)))
    return 0.5 * (batch.n_scored * logdet + quad)


def _nll_batches(vec: torch.Tensor, batches: Sequence[_Batch], fit_rate: float) -> torch.Tensor:
    """The exact Gaussian NLL of every block, up to a constant."""
    p = _unpack(vec)
    phi, q, p_stat, h = _state_space_torch(p, 1.0 / float(fit_rate))
    k, s = steady_state_gain(phi, q, p_stat, h)
    total = vec.new_zeros(())
    for batch in batches:
        total = total + _batch_nll(batch, phi, k, h, s)
    return total


def _log_normal_penalty(x: torch.Tensor, median: float, sd: float) -> torch.Tensor:
    z = (torch.log(x) - float(np.log(median))) / float(sd)
    return 0.5 * torch.sum(z * z)


def _neg_log_post(vec: torch.Tensor, batches: Sequence[_Batch], fit_rate: float) -> torch.Tensor:
    p = _unpack(vec)
    penalty = (
        _log_normal_penalty(p["tau_slow"], *TAU_SLOW_PRIOR)
        + _log_normal_penalty(p["sigma_slow"], *SIGMA_PRIOR)
        + _log_normal_penalty(p["sigma_osc"], *SIGMA_PRIOR)
        + _log_normal_penalty(p["tau_e"].reshape(1), *TAU_E_PRIOR)
        + _log_normal_penalty(p["sigma_e"].reshape(1), *SIGMA_E_PRIOR)
    )
    return _nll_batches(vec, batches, fit_rate) + penalty


def exact_nll(params: Params, batches: Sequence[_Batch], fit_rate: float) -> float:
    """The exact Gaussian NLL of ``params`` on ``batches`` (no priors)."""
    return float(_nll_batches(_fit_vector(params), batches, fit_rate).item())


def _fit_vector(params: Params) -> torch.Tensor:
    """The 23-dim vector the optimiser works on (``mu`` and ``s`` are fixed)."""
    return torch.as_tensor(
        np.concatenate(
            [
                [params.theta],
                u_from_tau_slow(params.tau_slow, params.f0),
                np.log(params.sigma_slow),
                v_from_f0(params.f0),
                v_from_zeta(params.zeta),
                np.log(params.sigma_osc),
                [v_from_tau_e(params.tau_e), np.log(max(params.sigma_e, SIGMA_E_FLOOR))],
            ]
        ),
        dtype=torch.float64,
    )


def _params_from_fit_vector(vec: np.ndarray, mu: np.ndarray, s_c: float, s_r: np.ndarray) -> Params:
    v = np.asarray(vec, dtype=np.float64)
    f0 = f0_from_v(v[9:13])
    return Params(
        mu=mu,
        theta=float(v[0]),
        tau_slow=tau_slow_from_u(v[1:5], f0),
        sigma_slow=np.exp(v[5:9]),
        f0=f0,
        zeta=zeta_from_v(v[13:17]),
        sigma_osc=np.exp(v[17:21]),
        tau_e=tau_e_from_v(v[21]),
        sigma_e=float(np.exp(v[22])),
        s_c=s_c,
        s_r=s_r,
    )


def _fit_bounds() -> list[tuple[float, float]]:
    b = _BOUNDS
    return (
        [b["theta"]]
        + [b["u_slow"]] * 4
        + [b["log_sigma"]] * 4
        + [b["v_f0"]] * 4
        + [b["v_zeta"]] * 4
        + [b["log_sigma"]] * 4
        + [b["v_tau_e"], b["log_sigma_e"]]
    )


# ─── the fit ──────────────────────────────────────────────────────────────────


@dataclass
class NewFit:
    """A fitted new-model rig: parameters plus the fit's provenance.

    ``rps_min``/``rps_max`` are the rig's real airborne extremes over all its
    flights, and the sampler clamps to them: a real ESC has a floor and a
    ceiling, and a Gaussian model does not know that.
    """

    rig: str
    params: Params
    idle_rps: np.ndarray
    rps_min: float = 0.0
    rps_max: float = float("inf")
    fs: float = RATE_HZ
    fit_rate_hz: float = RATE_HZ
    nll: float = float("nan")
    n_iter: int = 0
    wall_s: float = 0.0
    n_blocks: int = 0
    n_scored: int = 0
    n_flights: int = 0
    n_restarts: int = 0

    def __post_init__(self) -> None:
        self.idle_rps = np.asarray(self.idle_rps, dtype=np.float64).reshape(NUM_ROTORS)

    @property
    def clip(self) -> tuple[float, float]:
        return (float(self.rps_min), float(self.rps_max))

    def sampler(self, fs: float = RATE_HZ, *, antithetic_offsets: bool = False) -> Sampler:
        """The frozen sampler protocol (contract 3), clamped to the rig's ESC
        floor and ceiling."""
        return self.params.sampler(fs=fs, antithetic_offsets=antithetic_offsets, clip=self.clip)

    def full_flight(
        self,
        duration_s: float,
        fs: float = RATE_HZ,
        rng: np.random.Generator | int | None = None,
        phases: FlightPhaseRanges | None = None,
    ) -> np.ndarray:
        """``(4, n)`` whole flight: ground, spin-up, idle, take-off, the
        stationary airborne process, landing, spin-down, ground."""
        generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

        def airborne(n: int, rng_: np.random.Generator, fs: float = fs) -> np.ndarray:
            return self.params.sample_airborne(n, rng_, fs=fs, clip=self.clip)

        return wrap_airborne(airborne, duration_s, fs, generator, idle=self.idle_rps, phases=phases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "new",
            "rig": self.rig,
            "params": self.params.to_dict(),
            "idle_rps": self.idle_rps.tolist(),
            "rps_min": self.rps_min,
            "rps_max": self.rps_max,
            "fs": self.fs,
            "fit_rate_hz": self.fit_rate_hz,
            "nll": self.nll,
            "n_iter": self.n_iter,
            "wall_s": self.wall_s,
            "n_blocks": self.n_blocks,
            "n_scored": self.n_scored,
            "n_flights": self.n_flights,
            "n_restarts": self.n_restarts,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, allow_nan=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NewFit:
        return cls(
            rig=payload["rig"],
            params=Params.from_dict(payload["params"]),
            idle_rps=payload["idle_rps"],
            rps_min=float(payload.get("rps_min", 0.0)),
            rps_max=float(payload.get("rps_max", float("inf"))),
            fs=float(payload.get("fs", RATE_HZ)),
            fit_rate_hz=float(payload.get("fit_rate_hz", RATE_HZ)),
            nll=float(payload.get("nll", float("nan"))),
            n_iter=int(payload.get("n_iter", 0)),
            wall_s=float(payload.get("wall_s", 0.0)),
            n_blocks=int(payload.get("n_blocks", 0)),
            n_scored=int(payload.get("n_scored", 0)),
            n_flights=int(payload.get("n_flights", 0)),
            n_restarts=int(payload.get("n_restarts", 0)),
        )

    @classmethod
    def from_json(cls, text: str | bytes) -> NewFit:
        return cls.from_dict(json.loads(text))


def _airborne_stack(flight: Flight, fs: float) -> np.ndarray | None:
    segs = [flight.rps[:, sl] for sl in airborne_segments(flight.rps, fs)]
    return np.concatenate(segs, axis=1) if segs else None


def _nanmean_rotors(x: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x)
    count = finite.sum(axis=1)
    total = np.where(finite, x, 0.0).sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def pooled_means(flights: Sequence[Flight], fs: float) -> tuple[np.ndarray, float, np.ndarray]:
    """``(mu, s_c, s_r)``: the pooled airborne rotor mean and the per-flight
    offset's COMMON and PER-ROTOR standard deviations.

    ``delta_flight = 1 c + r`` with ``c ~ N(0, s_c^2)`` shared by all four
    rotors and ``r ~ N(0, diag(s_r^2))``, moment-estimated from the population
    covariance ``C`` of the per-flight airborne rotor means: ``s_c^2`` is the
    mean of ``C``'s six off-diagonal entries (floored at 0, since a negative
    average cross-covariance is not a common mode) and ``s_r^2 = max(diag(C) -
    s_c^2, 0)``.  With fewer than :data:`S_MIN_FLIGHTS` flights both are 0,
    because the offset cannot be told apart from the pooled mean.

    WHY THE SPLIT (round 5).  Round 4 estimated a per-rotor offset only, and on
    neurobem it came out at ~41 rev/s on ALL FOUR rotors — that is one common
    level difference between its gentle and aggressive flights, not four
    independent ones.  Drawing it independently per rotor manufactures rotor
    SPREADS no real flight has, and the frozen airborne rule then throws away
    31 % of the samples (retention 0.69) from the low side and biases every
    mean statistic.  Splitting the covariance into a common part and a residual
    reproduces the same ``rotor_var`` while keeping a sampled flight's four
    rotors as close together as the real ones.

    The population (not sample) covariance is the right estimator: the frozen
    ``rotor_var`` pools every airborne sample about the POOLED mean, so it sees
    exactly the population variance of the flight means, and the model
    reproduces it without a Bessel correction.
    """
    per_flight = [x for x in (_airborne_stack(f, fs) for f in flights) if x is not None]
    if not per_flight:
        raise ValueError("no airborne samples in the given flights")
    pooled = np.concatenate(per_flight, axis=1)
    mu = _nanmean_rotors(pooled)
    if len(per_flight) < S_MIN_FLIGHTS:
        return mu, 0.0, np.zeros(NUM_ROTORS)
    offsets = np.stack([_nanmean_rotors(x) - mu for x in per_flight], axis=0)
    cov = (offsets.T @ offsets) / float(offsets.shape[0])
    off_diag = cov[~np.eye(NUM_ROTORS, dtype=bool)]
    var_c = max(float(off_diag.mean()), 0.0)
    var_r = np.maximum(np.diag(cov) - var_c, 0.0)
    return mu, float(np.sqrt(var_c)), np.sqrt(var_r)


def airborne_extremes(flights: Sequence[Flight], fs: float) -> tuple[float, float]:
    """``(min, max)`` rotor speed over every airborne sample of every flight."""
    lo, hi = np.inf, -np.inf
    for flight in flights:
        x = _airborne_stack(flight, fs)
        if x is None:
            continue
        finite = x[np.isfinite(x)]
        if finite.size:
            lo = min(lo, float(finite.min()))
            hi = max(hi, float(finite.max()))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("no finite airborne samples")
    return lo, hi


def _start_params(
    flights: Sequence[Flight], fs: float, mu: np.ndarray, s_c: float, s_r: np.ndarray
) -> Params:
    """The data-driven start: mode variances split 70/30 slow/oscillator,
    ``tau_slow`` 2 s, ``f0`` 1 Hz, ``zeta`` 1 (critically damped, no resonance
    assumed), ``theta`` 0, and a small fast measurement process."""
    blocks = []
    for flight in flights:
        x = _airborne_stack(flight, fs)
        if x is None:
            continue
        x = x[:, np.isfinite(x).all(axis=0)]
        if x.size:
            blocks.append(x - x.mean(axis=1, keepdims=True))
    centred = np.concatenate(blocks, axis=1) if blocks else np.zeros((NUM_ROTORS, 1))
    mode_var = np.maximum(((MIXER.T @ centred) / NUM_ROTORS).var(axis=1), 1e-6)
    rms = float(np.sqrt(np.mean(centred.var(axis=1))))
    return Params(
        mu=mu,
        theta=0.0,
        tau_slow=np.full(NUM_ROTORS, 2.0),
        sigma_slow=np.sqrt(0.7 * mode_var),
        f0=np.full(NUM_ROTORS, 1.0),
        zeta=np.full(NUM_ROTORS, 1.0),
        sigma_osc=np.sqrt(0.3 * mode_var),
        tau_e=0.05,
        sigma_e=max(0.05 * rms, 1e-3),
        s_c=s_c,
        s_r=s_r,
    )


def _jitter(start: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One random restart around the start vector."""
    v = start.copy()
    v[0] = rng.uniform(-np.pi / 4.0, np.pi / 4.0)  # theta
    v[1:5] += rng.normal(0.0, 0.7, 4)  # u_slow
    v[5:9] += rng.normal(0.0, 0.4, 4)  # log sigma_slow
    v[9:13] += rng.normal(0.0, 1.2, 4)  # v_f0
    v[13:17] += rng.normal(0.0, 1.0, 4)  # v_zeta
    v[17:21] += rng.normal(0.0, 0.4, 4)  # log sigma_osc
    v[21:23] += rng.normal(0.0, 0.5, 2)  # log tau_e, log sigma_e
    return v


def fit_rig(
    flights: Sequence[Flight],
    rig: str,
    *,
    seed: int = 0,
    n_restarts: int = 4,
    fit_rate: float | None = None,
    idle_rps: np.ndarray | None = None,
    start: Params | None = None,
    max_iter: int | None = None,
) -> NewFit:
    """Exact-likelihood MAP fit of the new model to one rig's real flights.

    ``mu``, the per-flight offset std ``s`` and the ESC extremes come from the
    airborne samples; everything else is a MAP fit of the batched steady-state
    Kalman likelihood at the rig's :func:`fit_rate_hz`, with ``n_restarts``
    L-BFGS-B runs.  ``start`` warm-starts restart 0 (the previous round's fit);
    the remaining restarts are jittered around it.
    """
    flights = list(flights)
    if not flights:
        raise ValueError("no flights to fit")
    fs = float(flights[0].fs)
    if any(abs(float(f.fs) - fs) > 1e-9 for f in flights):
        raise ValueError("flights disagree on the sample rate")
    rate = fit_rate_hz(rig) if fit_rate is None else float(fit_rate)

    mu, s_c, s_r = pooled_means(flights, fs)
    rps_min, rps_max = airborne_extremes(flights, fs)
    batches = likelihood_batches(flights, rate, fs)
    start_params = start or _start_params(flights, fs, mu, s_c, s_r)
    start_vec = _fit_vector(start_params).numpy()
    bounds = _fit_bounds()
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        vec = torch.as_tensor(x, dtype=torch.float64).requires_grad_(True)
        value = _neg_log_post(vec, batches, rate)
        (grad,) = torch.autograd.grad(value, vec)
        return float(value.item()), grad.numpy()

    options = {"maxiter": int(max_iter)} if max_iter else None
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    best: tuple[float, np.ndarray, int] | None = None
    for restart in range(max(int(n_restarts), 1)):
        x0 = np.clip(start_vec if restart == 0 else _jitter(start_vec, rng), lo, hi)
        res = minimize(objective, x0, jac=True, method="L-BFGS-B", bounds=bounds, options=options)
        if np.isfinite(res.fun) and (best is None or res.fun < best[0]):
            best = (float(res.fun), np.asarray(res.x, dtype=np.float64), int(res.nit))
    wall = time.perf_counter() - t0
    if best is None:
        raise RuntimeError(f"every restart of the {rig!r} fit failed")

    params = _params_from_fit_vector(best[1], mu, s_c, s_r).canonical()
    if idle_rps is None:
        idle_rps = _idle_level(flights, mu)
    return NewFit(
        rig=rig,
        params=params,
        idle_rps=idle_rps,
        rps_min=rps_min,
        rps_max=rps_max,
        fs=fs,
        fit_rate_hz=rate,
        nll=exact_nll(params, batches, rate),
        n_iter=best[2],
        wall_s=wall,
        n_blocks=int(sum(b.n_blocks for b in batches)),
        n_scored=int(sum(b.n_scored for b in batches)),
        n_flights=len(flights),
        n_restarts=max(int(n_restarts), 1),
    )


def _idle_level(flights: Sequence[Flight], mu: np.ndarray) -> np.ndarray:
    """Per-rotor warm-up idle level (rev/s).

    ``data.ground_level(flight) -> float | None`` is the scalar common-mode
    standby level of one flight; the median over the flights that have one is
    spread over the rotors by the rig's own trim ratio ``mu / mean(mu)``
    (Michael's four motors visibly idle at different speeds, at about the same
    relative spread as in cruise).  Rigs whose recordings start already airborne
    return ``None`` everywhere — then fall back to 0.35 x the mean hover level.
    """
    trim = mu / float(np.mean(mu))
    levels: list[float] = []
    try:
        from experiments.rps_traj.data import ground_level  # noqa: PLC0415
    except ImportError:
        ground_level = None  # type: ignore[assignment]
    if ground_level is not None:
        for flight in flights:
            level = ground_level(flight)
            if level is not None and np.isfinite(level) and float(level) > 0.0:
                levels.append(float(level))
    common = float(np.median(levels)) if levels else 0.35 * float(np.mean(mu))
    return common * trim


def cross_periodogram(block: np.ndarray, fs: float = RATE_HZ) -> tuple[np.ndarray, np.ndarray]:
    """``(f, I)`` with ``I`` ``(F, 4, 4)``, Hann-tapered and taper-corrected.

    Not part of the likelihood any more, but it is what pins the PSD
    convention: ``sum_f I(f) df`` reproduces the tapered block variance, so
    ``I`` and :meth:`Params.spectral_matrix` are directly comparable, and the
    diagnostic script and one test use it to check that the spectrum really is
    the spectrum of what the sampler draws.
    """
    block = np.asarray(block, dtype=np.float64)
    length = block.shape[1]
    w = hann(length, sym=False)
    centred = block - block.mean(axis=1, keepdims=True)
    spec = np.fft.rfft(centred * w, axis=-1)
    scale = 2.0 / (fs * float(np.sum(w**2)))
    return np.fft.rfftfreq(length, d=1.0 / fs), scale * np.einsum(
        "jf,kf->fjk", spec, np.conj(spec)
    ).real


#: Erosion the frozen airborne rule removes from both ends of a sample, so a
#: runner asking for a real segment length must ask for this much more.
SAMPLE_PAD_S = 2.0 * ERODE_S


__all__ = [
    "BLOCK_S",
    "BURN_S",
    "F0_MAX_HZ",
    "F0_MIN_HZ",
    "FIT_RATE_HZ",
    "MIN_SCORED_S",
    "PAD_QUANTUM",
    "N_FIT_PARAMS",
    "N_PARAMS",
    "N_STATES",
    "SAMPLE_PAD_S",
    "TAU_E_MAX_S",
    "TAU_E_MIN_S",
    "TAU_SLOW_MAX_S",
    "ZETA_MAX",
    "ZETA_MIN",
    "NewFit",
    "Params",
    "airborne_extremes",
    "car2_psd_grid",
    "car2_state_space",
    "corner_tau_s",
    "cross_periodogram",
    "decimate",
    "exact_nll",
    "f0_from_v",
    "fit_rate_hz",
    "fit_rig",
    "likelihood_batches",
    "ou_psd_grid",
    "ou_state_space",
    "pooled_means",
    "rotation",
    "state_space_psd",
    "steady_state_gain",
    "tau_e_from_v",
    "tau_slow_from_u",
    "u_from_tau_slow",
    "v_from_f0",
    "v_from_tau_e",
    "v_from_zeta",
    "zeta_from_v",
]
