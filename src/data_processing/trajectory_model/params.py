"""The rotor-speed trajectory model's parameters, state spaces and sampler.

Rotor speeds in rev/s, mixer order ``[RFront, LFront, LBack, RBack]``::

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
  "measurement" so a training sampler can drop it (``measurement_noise:
  false``) and keep the shaft.
* ``delta_flight = 1 c + r``, one draw per flight, with ``c ~ N(0, s_c^2)``
  COMMON to the four rotors and ``r ~ N(0, diag(s_r^2))`` per rotor.  The frozen
  ``rotor_var`` pools every airborne sample about the *pooled* mean, so the
  between-flight level differences are part of what is scored; they are partly
  common (neurobem's gentle and aggressive flights differ by ~41 rev/s on all
  four rotors at once) and partly per-rotor (michaels' rotor 0 carries 2.5x its
  neighbours' variance because its own trim moves 17 rev/s), and a model with
  only the per-rotor part manufactures rotor spreads no real flight has.

Every component is defined by its EXACT discrete state space on the sampling
grid — ``Phi = expm(A dt)``, ``Q = P - Phi P Phi^T`` with ``P`` the continuous
stationary covariance, exact because the process is stationary — and the
spectrum in :meth:`Params.spectral_matrix` is computed from the same
``(Phi, Q, H)``, so the sampler and the spectrum cannot drift apart.

This module is the SAMPLING half of the model: parameters, discretisation,
the state-space recurrences and the spectra they imply.  The exact-likelihood
MAP fit that produced the shipped parameters (torch Kalman filter, priors,
optimiser) lives in :mod:`experiments.rps_traj.model`, which imports from
here; the campaign write-up is ``docs/experiments/rps-trajectory-model.md``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import expm
from scipy.signal import lfilter, lfiltic

from tracking.rotors import MIXER, NUM_ROTORS

#: The grid the shipped fits are stated on, and every default ``fs`` here (Hz).
#: Also the campaign's common analysis rate
#: (:data:`experiments.rps_traj.data.RATE_HZ`, which reads it from here).
RATE_HZ = 100.0

#: HARD parameter ranges, imposed by parametrisation rather than by the
#: optimiser's box.  ``f0`` spans the resolvable range; ``zeta`` reaches 3 so a
#: mode can be firmly overdamped and 0.2 so it can be as sharp as a Q ~ 6
#: attitude loop; ``tau_slow`` is bounded BELOW by the oscillator's corner time
#: ``tau_c = 1/(2 pi f0)``, which orders the two components without a
#: label-switching gauge, and ABOVE by 10 s, the longest lag the frozen ACF
#: scores.
F0_MIN_HZ, F0_MAX_HZ = 0.05, 20.0
ZETA_MIN, ZETA_MAX = 0.2, 3.0
TAU_SLOW_MAX_S = 10.0

#: HARD range on the measurement process' time constant (s).  It models
#: sample-and-hold, quantisation and per-rotor ESC jitter, whose correlation
#: times are milliseconds to tens of milliseconds (michaels' median hold is
#: 68 ms, DREGON's 19 ms); a second would be shaft behaviour, not measurement.
TAU_E_MIN_S, TAU_E_MAX_S = 1e-3, 1.0

#: Floor on the per-flight offset std (rev/s), so a rig fitted with the offset
#: term switched off round-trips through the log coordinates.
S_FLOOR = 1e-3

#: Floor under ``sigma_e``.  The measurement process is the ONLY term that makes
#: the likelihood's innovation covariance full rank, so it must stay positive.
SIGMA_E_FLOOR = 1e-6

#: Number of parameters.
N_PARAMS = 32

#: State dimension: 4 slow OUs, 4 oscillator pairs, 4 measurement OUs.
N_STATES = 16

_TWO_PI = 2.0 * np.pi

#: A trajectory sampler: ``sampler(n_samples, rng) -> (n_rotors, n_samples)``
#: rev/s.  All randomness MUST come from ``rng`` — that is what makes the
#: frozen judge (:func:`experiments.rps_traj.stats.stats_from_samples`) and the
#: training streams reproducible.
Sampler = Callable[[int, np.random.Generator], np.ndarray]


def rotation(theta: float) -> np.ndarray:
    """``R(theta)``: rotation by ``theta`` in the (common, yaw) plane."""
    c, s = float(np.cos(theta)), float(np.sin(theta))
    r = np.eye(NUM_ROTORS)
    r[0, 0], r[0, 3] = c, -s
    r[3, 0], r[3, 3] = s, c
    return r


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
    """The 32 parameters of the trajectory model.

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
        """``(4, n)`` stationary airborne rotor speeds, one flight offset draw.

        ``offset`` overrides the per-flight offset ``delta_flight`` (rev/s per
        rotor); it is drawn from ``rng`` either way, so every other random
        stream is byte-identical to the plain sampler.
        """
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
        """``(w, delta_flight)``; ``offset`` overrides the drawn offset."""
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
        offset keeps its marginal law, so ``rotor_var``, ``acf`` and ``xcorr``
        are unchanged in distribution — but the offsets over an EVEN number of
        sampled flights cancel exactly, so the pooled model mean is ``mu``
        instead of carrying an ``s / sqrt(N)`` error.  The cost is that the
        sampler becomes stateful across calls — still deterministic in call
        order — which is why it is off by default.

        ``clip`` is the rig's ESC floor and ceiling; see
        :class:`~data_processing.trajectory_model.sampler.NewFit`.
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


__all__ = [
    "F0_MAX_HZ",
    "F0_MIN_HZ",
    "N_PARAMS",
    "N_STATES",
    "RATE_HZ",
    "SIGMA_E_FLOOR",
    "S_FLOOR",
    "TAU_E_MAX_S",
    "TAU_E_MIN_S",
    "TAU_SLOW_MAX_S",
    "ZETA_MAX",
    "ZETA_MIN",
    "Params",
    "Sampler",
    "car2_continuous",
    "car2_psd_grid",
    "car2_state_space",
    "corner_tau_s",
    "f0_from_v",
    "ou_psd_grid",
    "ou_state_space",
    "rotation",
    "state_space_psd",
    "tau_e_from_v",
    "tau_slow_from_u",
    "u_from_tau_slow",
    "v_from_f0",
    "v_from_tau_e",
    "v_from_zeta",
    "zeta_from_v",
]
