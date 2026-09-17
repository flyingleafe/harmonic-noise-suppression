"""Stage-2 synthetic study: when are R near-identical rotors separable?

The v2 rig fit of ``motor_allMotors_70`` has to reproduce four per-rotor fits
whose carriers sit 0.83-3.0 Hz apart (min spacing 0.83 Hz). At order ``k`` the
lines of two rotors sit ``k delta`` apart while every line carries an intrinsic
width set by the SAME dynamics the fit is measuring, so "can the rig fit see
four profiles at all" is a question about the model, not about the optimiser.
This module answers it on synthetic data where the per-rotor profile is KNOWN.

SIGNAL MODEL (the campaign model of ``docs/explainers/noise-model-v2-plan.qmd``,
"Total phase", with the per-mic response made explicit)::

    x_m(t) = sum_r sum_k Re[ c_rkm exp(i 2 pi k phi_r(t)) psi_rk(t) ] + floor_m(t)

* ``phi_r(t) = int_0^t f_r`` in TURNS is the KNOWN per-rotor track. On the
  bench the campaign fixes a constant ``f_r``; the truth carries a slow
  stationary OU drift about that constant whose std matches the measured
  per-rotor carrier residual (0.23-0.28 Hz on ``allMotors_70``). The drift is
  known to the phase-aware estimator (it is a track) and NOT to the
  periodogram reader (which registers on the constant) -- that asymmetry is one
  of the two levers under test.
* ``psi_rk(t) = exp(i [k theta_r(t) + eps_rk(t)])`` is the v2 decoherence:
  one integrated-OU shaft error per rotor plus an independent per-order phase
  OU with parity-selected ``(sigma_eps, lam_eps)``. Both are drawn with the
  same exact transitions :mod:`.render` uses (``simulate_state``,
  ``_ar1_causal``), so ``E[psi_rk(t + tau) conj(psi_rk(t))] = R_rk(tau)`` is
  exactly :func:`.lag.r_tau` -- the estimators below are therefore calibrated
  against the law the Pyro fit assumes, not against a re-derivation of it.
* ``c_rkm = sqrt(2 P_rk) g_rm exp(-i 2 pi k f_r eta_rm) exp(i beta_rk)``:
  ``P_rk`` is the per-rotor order power (the estimand, ``profile_db``),
  ``g_rm`` a per-rotor-per-mic gain and ``eta_rm`` a per-rotor-per-mic delay,
  so ONE steering vector per rotor serves every order up to the frequency
  dependence ``exp(-i 2 pi k f_r eta_rm)``. ``beta_rk`` is the per-(rotor,
  order) source phase, uniform and CONSTANT in time: it is not identifiable
  from a periodogram (it is C4's ``alpha`` promoted to the source side), which
  is why a steering vector is only ever recoverable up to a per-(r, k) phase.
* the floor is the frozen coloured floor of :func:`revised_phase.
  floor_power_spectrum` scaled to a requested in-band comb-to-floor ratio.

THE ESTIMAND is the mic-averaged per-rotor order power
``P_rk = (1/M) sum_m |c_rkm|^2 / 2`` -- exactly the ``profile_db`` row of a
``noise-v2-fit/1`` payload, with the mic gains mean-pinned as the fit pins
them. All three estimators return ``(R, K)`` power in the same units.

ESTIMATORS

``E1`` (:func:`estimate_e1`, current practice) is the full-length periodogram
read at ``k f_r`` with the campaign's constant carriers: Voronoi cells between
neighbouring lines, local-median floor subtraction, Parseval conversion.

``E2`` (:func:`estimate_e2`, phase-aware) demodulates by the MEAN track,
low-passes to the beat band and solves a linear least squares against the
known per-rotor beat bases over short windows. ``multi=False`` solves each mic
separately; ``multi=True`` imposes one steering vector per rotor, estimated
from the orders the criterion calls resolvable, and solves all mics jointly --
the spatial-separation lever.

THE CRITERION (:func:`resolvability_score`) is ``k delta min(tau_c(k), T)``:
the number of beat cycles of the pair observable inside the shorter of the
joint coherence time and the record. ``>= 1`` is the hypothesis' threshold and
is the same statement as "linewidth below spacing" for the periodogram, so it
should bound E1 and E2-single alike; only E2-multi has a lever that does not
go through it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from experiments.stochastic_fit.phase_kernel import hann_window
from experiments.stochastic_fit.revised_phase import (
    _ar1_causal,
    floor_geometry,
    floor_power_spectrum,
    simulate_state,
)

from . import spectrum as SP
from .lag import parity_select, r_tau

__all__ = [
    "FloorSpec",
    "MultiRotorScene",
    "RotorSpec",
    "SceneConfig",
    "coefficient_ratio",
    "coherence_time_s",
    "estimate_e1",
    "estimate_e2",
    "estimate_steering",
    "joint_coherence_time_s",
    "line_half_width_hz",
    "make_config",
    "per_rotor_scores",
    "profile_error_db",
    "resolvability_score",
    "rotor_from_fit",
    "scale_carriers",
    "simulate_scene",
    "window_coherent_fraction",
]

#: Speed of sound (m/s) used to turn an array aperture into a delay bound.
SPEED_OF_SOUND = 343.0

#: ``exp(-1)``: the coherence-time threshold. One nat of the lag law.
COHERENCE_THRESHOLD = math.exp(-1.0)


# ── specifications ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RotorSpec:
    """One rotor: its constant carrier, its profile and its v2 dynamics.

    ``profile_db`` is a ``(K,)`` row of per-order line powers in dB, i.e. one
    row of a fit's ``params.profile.profile_db``. ``sigma_eps`` and
    ``lam_eps`` are ``(even, odd)`` pairs exactly as :func:`.lag.r_tau` takes
    them.
    """

    name: str
    carrier_rev_s: float
    profile_db: np.ndarray
    sigma_nu: float
    lam: float
    sigma_eps: tuple[float, float]
    lam_eps: tuple[float, float]
    p: float = SP.P_ORDER_EXPONENT

    @property
    def n_orders(self) -> int:
        return int(np.asarray(self.profile_db).shape[-1])


@dataclass(frozen=True)
class FloorSpec:
    """The coloured floor's shape. Its LEVEL is set by the comb-to-floor ratio."""

    shape_z: np.ndarray
    tilt_db_oct: float
    mean_db: float = 0.0


@dataclass(frozen=True)
class SceneConfig:
    """Everything a synthetic multi-rotor scene fixes before any estimator."""

    rotors: tuple[RotorSpec, ...]
    n_mics: int = 8
    sr: int = SP.FLIGHT_SR
    duration_s: float = 24.0
    #: stationary std (Hz) of the per-rotor track drift about its constant.
    track_drift_std_hz: float = 0.24
    #: OU rate (1/s) of that drift. NOT measured by the campaign; see
    #: ``results/noise_v2/multirotor_synth/findings.md``.
    track_drift_rate_hz: float = 1.0
    #: in-band total comb power over total floor power, dB.
    comb_to_floor_db: float = 7.0
    #: per-mic line-gain spread (dB std) of the steering magnitudes.
    steering_gain_std_db: float = 2.0
    #: array aperture (m); per-rotor-per-mic delays are uniform on +-aperture/c.
    array_aperture_m: float = 0.17
    #: all rotors share ONE steering vector -- kills the spatial lever.
    identical_steering: bool = False
    floor: FloorSpec | None = None
    k_cap: int | None = None
    seed: int = 0

    @property
    def n_rotors(self) -> int:
        return len(self.rotors)

    @property
    def n_samples(self) -> int:
        return int(round(self.duration_s * self.sr))

    @property
    def carriers_rev_s(self) -> np.ndarray:
        return np.array([r.carrier_rev_s for r in self.rotors], dtype=np.float64)

    @property
    def min_spacing_hz(self) -> float:
        f = np.sort(self.carriers_rev_s)
        return float(np.min(np.diff(f))) if f.size > 1 else float("inf")


@dataclass
class MultiRotorScene:
    """A realised scene: the audio, the KNOWN tracks and the planted truth."""

    cfg: SceneConfig
    audio: np.ndarray  # (M, N) float64, absolute units
    track_rev_s: np.ndarray  # (R, N) instantaneous known speed
    track_turns: np.ndarray  # (R, N) phi_r in TURNS, phi_r(0) = 0
    power_true: np.ndarray  # (R, K) planted mic-averaged order power
    steering: np.ndarray  # (M, R, K) complex c_rkm / sqrt(2 P_rk)
    delays_s: np.ndarray  # (M, R)
    k_max: int
    floor_psd: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def sr(self) -> int:
        return self.cfg.sr

    @property
    def mean_track_turns(self) -> np.ndarray:
        return self.track_turns.mean(axis=0)


def rotor_from_fit(fit: dict[str, Any], *, name: str, rotor: int = 0) -> RotorSpec:
    """A :class:`RotorSpec` read verbatim out of a ``noise-v2-fit/1`` payload."""
    p = fit["params"]
    prof = np.asarray(p["profile"]["profile_db"], dtype=np.float64)
    return RotorSpec(
        name=name,
        carrier_rev_s=float(np.atleast_1d(p["carrier_rev_s"])[rotor]),
        profile_db=prof[rotor].copy(),
        sigma_nu=float(p["sigma_nu"]),
        lam=float(p["lam"]),
        sigma_eps=(float(p["sigma_eps_even"]), float(p["sigma_eps_odd"])),
        lam_eps=(float(p["lam_eps_even"]), float(p["lam_eps_odd"])),
        p=float(p.get("p", SP.P_ORDER_EXPONENT)),
    )


# ── the criterion ───────────────────────────────────────────────────────────


def _r_of(spec: RotorSpec, tau: np.ndarray, k: float) -> np.ndarray:
    return np.asarray(
        r_tau(
            tau,
            float(k),
            sigma_nu=spec.sigma_nu,
            lam=spec.lam,
            sigma_eps=list(spec.sigma_eps),
            lam_eps=list(spec.lam_eps),
            p=spec.p,
        ),
        dtype=np.float64,
    )


def _lag_grid(cap_s: float) -> np.ndarray:
    return np.concatenate([[0.0], np.geomspace(1e-6, max(float(cap_s), 1e-5), 4000)])


def _first_crossing(tau: np.ndarray, rho: np.ndarray, level: float) -> float:
    below = np.nonzero(rho < level)[0]
    if below.size == 0:
        return float(tau[-1])
    j = int(below[0])
    if j == 0:
        return 0.0
    t0, t1, r0, r1 = tau[j - 1], tau[j], rho[j - 1], rho[j]
    if r0 == r1:
        return float(t1)
    return float(t0 + (r0 - level) * (t1 - t0) / (r0 - r1))


def coherence_time_s(spec: RotorSpec, k: float, *, cap_s: float = 1e3) -> float:
    """Lag at which ``R_rk`` first falls to ``exp(-1)``, capped at ``cap_s``."""
    tau = _lag_grid(cap_s)
    return min(_first_crossing(tau, _r_of(spec, tau, k), COHERENCE_THRESHOLD), float(cap_s))


def joint_coherence_time_s(a: RotorSpec, b: RotorSpec, k: float, *, cap_s: float = 1e3) -> float:
    """``exp(-1)`` time of ``R_ak R_bk``.

    The beat of a PAIR is observable only while BOTH lines stay coherent: the
    cross term the least squares uses carries the product of the two residual
    phasors, so the product's decay -- not either factor's -- is the limit.
    """
    tau = _lag_grid(cap_s)
    rho = _r_of(a, tau, k) * _r_of(b, tau, k)
    return min(_first_crossing(tau, rho, COHERENCE_THRESHOLD), float(cap_s))


def resolvability_score(
    a: RotorSpec, b: RotorSpec, k: float, *, duration_s: float, delta_hz: float | None = None
) -> float:
    """``k delta min(tau_c(k), T)``: beat cycles of the pair actually observed.

    ``>= 1`` is the hypothesis' resolvability threshold. ``delta_hz`` defaults
    to the two rotors' carrier spacing. Because the shaft term is ballistic on
    the scale of one coherence time (``lam tau << 1``), ``tau_c(k) ~ C / k``
    and the score is nearly INDEPENDENT of ``k`` -- a prediction the sweep
    checks directly.
    """
    d = (
        abs(float(a.carrier_rev_s) - float(b.carrier_rev_s))
        if delta_hz is None
        else float(delta_hz)
    )
    tau_c = joint_coherence_time_s(a, b, k, cap_s=max(10.0 * float(duration_s), 1.0))
    return float(k) * d * min(tau_c, float(duration_s))


def per_rotor_scores(
    rotors: tuple[RotorSpec, ...], orders: np.ndarray, *, duration_s: float
) -> np.ndarray:
    """``(R, K)`` worst-neighbour score: the rotor is only as separable as its
    hardest neighbour makes it."""
    out = np.full((len(rotors), len(orders)), np.inf)
    for i, a in enumerate(rotors):
        for j, b in enumerate(rotors):
            if i == j:
                continue
            for q, k in enumerate(orders):
                out[i, q] = min(
                    out[i, q], resolvability_score(a, b, float(k), duration_s=duration_s)
                )
    return out


def window_coherent_fraction(spec: RotorSpec, k: float, window_s: float) -> float:
    """``E| (1/T_w) int_0^{T_w} psi_rk |^2`` under the lag law.

    ``= (2 / T_w) int_0^{T_w} (1 - tau / T_w) R_rk(tau) d tau``, the Bartlett
    weighting of the autocorrelation. This is the factor by which a least
    squares that treats ``psi`` as CONSTANT over the window attenuates the
    power it recovers, so :func:`estimate_e2` divides by it.
    """
    n = 2048
    tau = np.linspace(0.0, float(window_s), n)
    rho = _r_of(spec, tau, k)
    w = 1.0 - tau / max(float(window_s), 1e-12)
    return float(2.0 * np.trapezoid(w * rho, tau) / max(float(window_s), 1e-12))


# ── the generator ───────────────────────────────────────────────────────────


def _floor_psd(cfg: SceneConfig, n: int) -> np.ndarray:
    spec = cfg.floor
    freqs = np.fft.rfftfreq(n, d=1.0 / cfg.sr)
    if spec is None:
        return np.ones(freqs.size)
    shape_mat, tilt_oct = floor_geometry(freqs, SP.floor_ctrl_hz(cfg.sr))
    return np.asarray(
        floor_power_spectrum(
            shape_mat,
            tilt_oct,
            mean_db=float(spec.mean_db),
            ctrl_db=SP.floor_shape_db(spec.shape_z, sr=cfg.sr),
            tilt_db_oct=float(spec.tilt_db_oct),
            rate_factor=1.0,
        ),
        dtype=np.float64,
    )


def simulate_scene(cfg: SceneConfig) -> MultiRotorScene:
    """Draw one realisation of the scene described by ``cfg``.

    The comb is synthesised directly at ``cfg.sr`` over orders whose line sits
    below the Nyquist (:func:`.spectrum.k_max_for_carrier`), so nothing
    aliases and no decimator is needed: unlike :func:`.render.render_noise`
    this generator is a study instrument, and its output has to be the model
    without a front end in the way.
    """
    n = cfg.n_samples
    sr = float(cfg.sr)
    dt = 1.0 / sr
    r_n = cfg.n_rotors
    m_n = cfg.n_mics
    ss = np.random.SeedSequence(int(cfg.seed))
    rng_track, rng_state, rng_eps, rng_beta, rng_steer, rng_floor = (
        np.random.default_rng(s) for s in ss.spawn(6)
    )

    # ── the known track: constant + stationary OU drift ──
    drift = np.zeros((r_n, n))
    if cfg.track_drift_std_hz > 0.0 and cfg.track_drift_rate_hz > 0.0:
        e = math.exp(-float(cfg.track_drift_rate_hz) * dt)
        z = rng_track.standard_normal((r_n, n))
        sd = float(cfg.track_drift_std_hz)
        drive = z * (sd * math.sqrt(max(1.0 - e * e, 0.0)))
        drive[:, 0] = sd * z[:, 0]
        drift = _ar1_causal(drive, e)
    track = cfg.carriers_rev_s[:, None] + drift  # (R, N) rev/s
    # phi in TURNS, trapezoid-free cumulative sum on the sample grid: the
    # estimator uses the SAME phi, so any quadrature convention is shared and
    # cannot create a mismatch between truth and basis.
    phi = np.cumsum(track, axis=1) * dt  # (R, N)

    k_max = min(
        min(r.n_orders for r in cfg.rotors),
        SP.k_max_for_carrier(track.max(axis=1), cfg.sr, k_cap=cfg.k_cap),
    )

    # ── steering: one vector per rotor, gains + delays ──
    d_max = float(cfg.array_aperture_m) / SPEED_OF_SOUND
    gain_db = rng_steer.normal(0.0, float(cfg.steering_gain_std_db), size=(m_n, r_n))
    delays = rng_steer.uniform(-d_max, d_max, size=(m_n, r_n))
    if cfg.identical_steering:
        gain_db = np.repeat(gain_db[:, :1], r_n, axis=1)
        delays = np.repeat(delays[:, :1], r_n, axis=1)
    gain = 10.0 ** ((gain_db - gain_db.mean(axis=0, keepdims=True)) / 20.0)
    gain = gain / np.sqrt((gain**2).mean(axis=0, keepdims=True))  # mean |g|^2 = 1 per rotor
    beta = rng_beta.uniform(0.0, 2.0 * np.pi, size=(r_n, k_max))

    power_true = np.stack([10.0 ** (r.profile_db[:k_max] / 10.0) for r in cfg.rotors])  # (R, K)
    k_vec = np.arange(1, k_max + 1, dtype=np.float64)
    steer = (
        gain[:, :, None]
        * np.exp(
            -2j
            * np.pi
            * k_vec[None, None, :]
            * cfg.carriers_rev_s[None, :, None]
            * delays[:, :, None]
        )
        * np.exp(1j * beta)[None]
    )  # (M, R, K)

    # ── the comb ──
    audio = np.zeros((m_n, n))
    two_pi = 2.0 * np.pi
    for r_i, spec in enumerate(cfg.rotors):
        innov = rng_state.standard_normal((n, 2))
        theta, _nu = simulate_state(innov, lam=spec.lam, sigma=spec.sigma_nu, dt=dt)
        for k in range(1, k_max + 1):
            s_e = float(parity_select(np.asarray(spec.sigma_eps), float(k))) * float(k) ** (
                0.5 * spec.p
            )
            l_e = float(parity_select(np.asarray(spec.lam_eps), float(k)))
            e = math.exp(-l_e * dt)
            z = rng_eps.standard_normal(n)
            drive = z * (s_e * math.sqrt(max(1.0 - e * e, 0.0)))
            drive[0] = s_e * z[0]
            eps = _ar1_causal(drive, e)
            # the angle is reduced modulo one turn BEFORE the trig, exactly as
            # spectrum._comb_autocovariance does: k phi reaches 4e7 turns.
            ang = two_pi * (np.remainder(k * phi[r_i], 1.0)) + k * theta + eps
            amp = math.sqrt(2.0 * float(power_true[r_i, k - 1]))
            ec, es = amp * np.cos(ang), amp * np.sin(ang)
            for m in range(m_n):
                c = steer[m, r_i, k - 1]
                audio[m] += c.real * ec - c.imag * es

    # ── the floor, scaled to the requested in-band comb-to-floor ratio ──
    psd = _floor_psd(cfg, n)
    freqs = np.fft.rfftfreq(n, d=1.0 / cfg.sr)
    in_band = SP.bench_band(freqs, cfg.sr)
    comb_power = float(power_true.sum())
    # E[var] of ``irfft(rfft(white) sqrt(psd))`` restricted to the band is
    # ``(2 / N) sum_band psd`` (Parseval with E|rfft(white)|^2 = N), so the
    # level is set analytically -- no calibration draw, no seed dependence.
    unit_band = float(2.0 * psd[in_band].sum() / n)
    scale = math.sqrt(comb_power / max(unit_band, 1e-300) * 10.0 ** (-cfg.comb_to_floor_db / 10.0))
    for m in range(m_n):
        white = rng_floor.standard_normal(n)
        audio[m] += scale * np.fft.irfft(np.fft.rfft(white) * np.sqrt(psd), n=n)

    return MultiRotorScene(
        cfg=cfg,
        audio=audio,
        track_rev_s=track,
        track_turns=phi,
        power_true=power_true,
        steering=steer,
        delays_s=delays,
        k_max=int(k_max),
        floor_psd=psd * scale**2,
        diagnostics=dict(
            k_max=int(k_max),
            min_spacing_hz=cfg.min_spacing_hz,
            comb_power=comb_power,
            floor_band_power=float(unit_band * scale**2),
            comb_to_floor_db=float(cfg.comb_to_floor_db),
            track_drift_std_hz=float(drift.std(axis=1).mean()),
            delay_bound_s=d_max,
            steering_gain_db=gain_db.tolist(),
        ),
    )


# ── E1: the full-length periodogram ─────────────────────────────────────────


def _cell_edges(lines_hz: np.ndarray) -> np.ndarray:
    """``(L, 2)`` Voronoi cell edges of ``sorted(lines)``.

    The outer edges mirror the end gaps, so the first and last line keep a
    cell of the same width as their only neighbour gives them.
    """
    f = np.asarray(lines_hz, dtype=np.float64)
    if f.size == 1:
        return np.array([[f[0] - 0.5, f[0] + 0.5]])
    mids = 0.5 * (f[:-1] + f[1:])
    lo = np.concatenate([[f[0] - (mids[0] - f[0])], mids])
    hi = np.concatenate([mids, [f[-1] + (f[-1] - mids[-1])]])
    return np.stack([lo, hi], axis=1)


def line_half_width_hz(spec: RotorSpec, k: float, *, width_sigmas: float = 3.5) -> float:
    """Half-width (Hz) of one line's core, from its own coherence time.

    On the scale of a coherence time the shaft term is ballistic
    (``lam tau_c << 1`` for every fitted rotor), so ``R_rk`` is Gaussian with
    ``sigma_f = 1 / (sqrt(2) 2 pi tau_c)`` and ``width_sigmas`` standard
    deviations hold the core. Deriving the width from ``tau_c`` rather than
    from ``k sigma_nu`` keeps the per-order pedestal in it: a rotor whose
    ``sigma_eps`` dominates has a short ``tau_c`` and gets a wide band, which
    is what the pedestal needs.
    """
    tau_c = coherence_time_s(spec, k, cap_s=100.0)
    return float(width_sigmas / (math.sqrt(2.0) * 2.0 * math.pi * max(tau_c, 1e-9)))


def _smooth_floor(p_mean: np.ndarray, block: int = 2048) -> np.ndarray:
    """A per-bin floor estimate: block medians, linearly interpolated.

    A median over 2048 bins (85 Hz at 24 s) spans more than one order spacing
    and the comb occupies a small fraction of those bins, so the median is the
    floor and not the comb. Cheap, monotone in nothing, and it never has to be
    told where the lines are.
    """
    n_b = max(p_mean.size // block, 2)
    idx = np.linspace(0, p_mean.size, n_b + 1).astype(np.int64)
    centres = 0.5 * (idx[:-1] + idx[1:])
    meds = np.array([np.median(p_mean[a:b]) for a, b in zip(idx[:-1], idx[1:], strict=True)])
    return np.interp(np.arange(p_mean.size), centres, meds)


def estimate_e1(
    scene: MultiRotorScene,
    orders: np.ndarray,
    *,
    width_sigmas: float = 3.5,
    merge_factor: float = 1.0,
) -> dict[str, Any]:
    """Current practice: read the line powers off the full-length periodogram.

    One periodogram per mic, mic-averaged, ``P = |rfft(w x)|^2 / sum w^2`` with
    the frozen periodic Hann of :func:`phase_kernel.hann_window`. Every comb
    line of every rotor is placed at ``k f_r`` using the CONSTANT carriers (the
    campaign's bench convention -- and the reason a drifting track hurts this
    reader), the frequency axis is cut into GLOBAL Voronoi cells between
    neighbouring lines, and the floor of :func:`_smooth_floor` is subtracted.
    Conversion is Parseval's: a line of power ``p`` puts ``N p / 2`` into the
    one-sided periodogram whatever the window, so

        P_hat = (2 / N) sum_{j in cell} (P_j - floor_j).

    ATTRIBUTION. Lines closer than :func:`line_half_width_hz` blur together,
    so they are clustered: the cluster's TOTAL is read from the union of its
    cells (which is unbiased however blurred it is) and split between its
    members in proportion to their own cells' mass. Two consequences, both of
    them the real failure mode: an unresolved pair converges on the equal
    split ``G / 2`` regardless of its true ratio, and a cluster total is never
    lost to a cell boundary. Reading each cell in isolation instead would mix
    that attribution error with a truncation error and hide which one bites.
    """
    cfg = scene.cfg
    n = scene.audio.shape[1]
    w = hann_window(n)
    power = (np.abs(np.fft.rfft(scene.audio * w[None, :], axis=1)) ** 2) / float((w**2).sum())
    p_mean = power.mean(axis=0)
    floor = _smooth_floor(p_mean)
    excess = p_mean - floor
    df = float(cfg.sr) / n

    k_all = np.arange(1, scene.k_max + 1, dtype=np.int64)
    rot_idx = np.repeat(np.arange(cfg.n_rotors), k_all.size)
    ord_idx = np.tile(k_all, cfg.n_rotors)
    freq = cfg.carriers_rev_s[rot_idx] * ord_idx
    half = np.array(
        [
            line_half_width_hz(cfg.rotors[r], float(k), width_sigmas=width_sigmas)
            for r, k in zip(rot_idx, ord_idx, strict=True)
        ]
    )
    srt = np.argsort(freq)
    freq, rot_idx, ord_idx, half = freq[srt], rot_idx[srt], ord_idx[srt], half[srt]
    edges = _cell_edges(freq)

    j0 = np.clip(np.ceil(edges[:, 0] / df - 1e-9).astype(np.int64), 0, p_mean.size - 1)
    j1 = np.clip(np.floor(edges[:, 1] / df + 1e-9).astype(np.int64), 0, p_mean.size - 1)
    near = np.clip(np.rint(freq / df).astype(np.int64), 0, p_mean.size - 1)
    j0 = np.where(j1 >= j0, j0, near)
    j1 = np.where(j1 >= j0, j1, near)
    mass = np.array([float(excess[a : b + 1].sum()) for a, b in zip(j0, j1, strict=True)])
    cell_bins = (j1 - j0 + 1).astype(np.int64)

    gap = np.diff(freq)
    joined = gap < merge_factor * np.maximum(half[:-1], half[1:])
    cluster = np.concatenate([[0], np.cumsum(~joined)])
    n_cl = int(cluster.max()) + 1
    total = np.zeros(n_cl)
    pos_mass = np.zeros(n_cl)
    size = np.zeros(n_cl, dtype=np.int64)
    np.add.at(total, cluster, mass)
    np.add.at(pos_mass, cluster, np.maximum(mass, 0.0))
    np.add.at(size, cluster, 1)
    share = np.where(
        pos_mass[cluster] > 0.0,
        np.maximum(mass, 0.0) / np.where(pos_mass[cluster] > 0.0, pos_mass[cluster], 1.0),
        1.0 / size[cluster],
    )
    est = np.maximum(2.0 * total[cluster] * share / n, 0.0)

    order_list = np.asarray(orders, dtype=np.int64)
    out = np.zeros((cfg.n_rotors, order_list.size))
    bins = np.zeros((cfg.n_rotors, order_list.size), dtype=np.int64)
    csize = np.zeros((cfg.n_rotors, order_list.size), dtype=np.int64)
    pos = {(int(r), int(k)): i for i, (r, k) in enumerate(zip(rot_idx, ord_idx, strict=True))}
    for r in range(cfg.n_rotors):
        for q, k in enumerate(order_list):
            i = pos[(r, int(k))]
            out[r, q] = est[i]
            bins[r, q] = cell_bins[i]
            csize[r, q] = size[cluster[i]]
    return dict(
        power=out,
        orders=order_list,
        bin_hz=df,
        cell_bins=bins,
        cluster_size=csize,
        n_clusters=n_cl,
        estimator="E1",
    )


# ── E2: demodulate, low-pass, linear least squares ──────────────────────────


def _demodulate(
    audio: np.ndarray, phi_mean: np.ndarray, k: int, sr: int, bandwidth_hz: float
) -> tuple[np.ndarray, float]:
    """``(M, n_d)`` band-limited analytic baseband of order ``k``, and its rate.

    ``y_m = x_m exp(-i 2 pi k phi_mean)`` puts the order's group at baseband
    and its negative-frequency image near ``-2 k f``; keeping only
    ``|f| <= bandwidth`` deletes the image and every other order. The
    resampling is spectral (select the retained bins of the length-``N``
    complex FFT and inverse transform the short vector), which is exact for a
    signal already confined to the band and costs one FFT pair per mic.
    """
    n = audio.shape[1]
    y = audio * np.exp(-2j * np.pi * float(k) * np.remainder(phi_mean, 1.0))[None, :]
    half = int(min(max(math.ceil(bandwidth_hz * n / sr), 4), (n - 1) // 2))
    yf = np.fft.fft(y, axis=1)
    sel = np.concatenate([yf[:, : half + 1], yf[:, n - half :]], axis=1)
    n_d = sel.shape[1]
    return np.fft.ifft(sel, axis=1) * (n_d / n), float(n_d) / (n / float(sr))


def _basis(
    phi: np.ndarray, phi_mean: np.ndarray, k: int, n_d: int, sr: int, duration_s: float
) -> np.ndarray:
    """``(n_d, R)`` beat bases ``exp(i 2 pi k (phi_r - phi_mean))`` on the
    decimated grid. The phase difference is smooth and tiny (``k delta T`` turns
    at most), so it is read off the full-rate track by linear interpolation."""
    n = phi.shape[1]
    t_full = np.arange(n) / float(sr)
    t_d = np.arange(n_d) * (duration_s / n_d)
    diff = phi - phi_mean[None, :]
    cols = [np.interp(t_d, t_full, float(k) * diff[r]) for r in range(phi.shape[0])]
    return np.exp(2j * np.pi * np.stack(cols, axis=1))


def _window_length(k: int, delta_hz: float, sr_d: float, n_d: int, n_rotors: int) -> int:
    """Samples per LS window: two beat cycles of the closest pair, at least
    ``R + 2`` samples, at most the whole record."""
    want = 2.0 / max(float(k) * float(delta_hz), 1e-9)
    return int(min(max(int(round(want * sr_d)), n_rotors + 2), n_d))


def _solve_windows(
    y: np.ndarray, basis: np.ndarray, n_w: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Windowed complex LS of ``y`` (``n_d,``) on ``basis`` (``n_d, R``).

    Returns the per-window coefficients ``(W, R)``, the per-window noise
    variance of each coefficient (``sigma^2 diag (A^H A)^-1``, the exact
    white-noise covariance diagonal of the LS estimate) and the median
    condition number of the normal matrix. The noise term is SUBTRACTED from
    ``|c|^2`` by the caller: an LS coefficient is a complex Gaussian whose
    squared modulus is biased up by exactly its own variance, and at low
    line-to-floor ratio that bias is the whole answer.
    """
    n_d = y.size
    n_win = n_d // n_w
    if n_win == 0:
        n_win, n_w = 1, n_d
    r_n = basis.shape[1]
    yy = y[: n_win * n_w].reshape(n_win, n_w)
    aa = basis[: n_win * n_w].reshape(n_win, n_w, r_n)
    gram = np.einsum("wtr,wts->wrs", aa.conj(), aa)
    rhs = np.einsum("wtr,wt->wr", aa.conj(), yy)
    eye = np.eye(r_n)[None]
    # a Tikhonov floor at 1e-12 of the trace keeps a singular window solvable;
    # it is 120 dB below the signal and never binds on a conditioned window.
    reg = 1e-12 * np.trace(gram, axis1=1, axis2=2).real[:, None, None] / r_n
    coef = np.linalg.solve(gram + reg * eye, rhs[:, :, None])[:, :, 0]
    resid = yy - np.einsum("wtr,wr->wt", aa, coef)
    dof = max(n_w - r_n, 1)
    sig2 = (np.abs(resid) ** 2).sum(axis=1) / dof  # (W,)
    inv_diag = np.linalg.inv(gram + reg * eye)[:, np.arange(r_n), np.arange(r_n)].real
    cond = float(np.median(np.linalg.cond(gram + reg * eye)))
    return coef, sig2[:, None] * inv_diag, cond


def coefficient_ratio(coef_windows: np.ndarray) -> np.ndarray:
    """``(M, R)`` mic-relative steering ratio from per-window LS coefficients.

    ``coef_windows`` is ``(M, W, R)``. The decoherence ``psi_rk`` is SHARED
    across microphones (both the shaft and the per-order term are source-side
    in the v2 model), so it cancels in the mic ratio window by window while a
    whole-record solve would average it to nothing. The estimator is the LS
    ratio ``sum_w c_m conj(c_0) / sum_w |c_0|^2``, which weights the windows
    by the reference mic's own power.
    """
    ref = coef_windows[0]  # (W, R)
    num = (coef_windows * np.conj(ref)[None]).sum(axis=1)  # (M, R)
    den = (np.abs(ref) ** 2).sum(axis=0)[None]  # (1, R)
    return num / np.where(np.abs(den) > 0.0, den, 1.0)


def estimate_steering(
    ratio: np.ndarray, orders: np.ndarray, carriers_rev_s: np.ndarray, *, delay_bound_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-rotor steering from the RESOLVABLE orders' mic-relative ratios.

    ``ratio`` is ``(M, R, K_s)``, :func:`coefficient_ratio` stacked over the
    orders ``orders``. The model is ``c_rkm = a_rk g_rm exp(-i 2 pi k f_r
    eta_rm)`` with ``a_rk`` an unknown per-(rotor, order) amplitude and phase,
    so only the mic-RELATIVE steering is identifiable -- which is all the
    joint solve needs. The gain is the median modulus over the orders; the
    delay difference is found by a matched filter over ``eta`` (no phase
    unwrapping: the orders need not be consecutive, and a grid search over a
    bounded physical delay is both robust and cheap).

    Returns ``(gain (M, R), delay (M, R))`` with mic 0 as the reference.
    """
    gain = np.median(np.abs(ratio), axis=2)  # (M, R)
    k = np.asarray(orders, dtype=np.float64)
    grid = np.linspace(-2.0 * delay_bound_s, 2.0 * delay_bound_s, 4001)
    phase = ratio / np.maximum(np.abs(ratio), 1e-300)
    delay = np.zeros(gain.shape)
    for r in range(gain.shape[1]):
        kern = np.exp(2j * np.pi * np.outer(grid, k * float(carriers_rev_s[r])))  # (G, K_s)
        sc = np.abs(phase[:, r, :] @ kern.T)  # (M, G)
        delay[:, r] = grid[np.argmax(sc, axis=1)]
    return gain, delay


def estimate_e2(
    scene: MultiRotorScene,
    orders: np.ndarray,
    *,
    multi: bool = False,
    steering_orders: np.ndarray | None = None,
    bandwidth_margin_hz: float = 4.0,
    correct_decoherence: bool = True,
) -> dict[str, Any]:
    """Phase-aware per-rotor order power.

    For each order ``k``: demodulate every mic by the MEAN track phase, keep
    the beat band, build the ``(n_d, R)`` design matrix
    ``A[t, r] = exp(i 2 pi k (phi_r(t) - phi_bar(t)))`` and solve
    ``min_c || y - A c ||^2`` over windows of two beat cycles.

    THE DESIGN MATRIX. ``y_m(t)`` is the analytic baseband, so the real comb's
    ``Re[c exp(i Theta)]`` contributes ``c / 2`` to it; the recovered power is
    therefore ``P = |2 c_LS|^2 / 2 = 2 |c_LS|^2``, debiased by the LS noise
    variance and divided by :func:`window_coherent_fraction` -- the Bartlett
    average of the lag law over the window, which is what a constant-``psi``
    design matrix loses. That correction uses the KNOWN dynamics; in the rig
    fit the dynamics are sampled jointly, so it is a parameter of the
    likelihood rather than a calibration.

    ``multi=True`` imposes ONE steering vector per rotor: the columns become
    ``s_rm(k) A[t, r]`` stacked over mics, so two rotors whose beat bases are
    nearly collinear in time are still separated if their steering vectors
    differ. The steering is estimated by :func:`estimate_steering` from
    ``steering_orders`` (default: the orders given, restricted to those whose
    per-rotor criterion score is largest).
    """
    cfg = scene.cfg
    order_list = np.asarray(orders, dtype=np.int64)
    phi_mean = scene.mean_track_turns
    delta = cfg.min_spacing_hz
    spread = float(np.max(np.abs(cfg.carriers_rev_s - cfg.carriers_rev_s.mean())))
    drift = float(cfg.track_drift_std_hz)
    r_n, m_n = cfg.n_rotors, cfg.n_mics

    def one_order(k: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        bw = float(k) * (spread + 4.0 * drift) + bandwidth_margin_hz
        y, sr_d = _demodulate(scene.audio, phi_mean, k, cfg.sr, bw)
        n_d = y.shape[1]
        basis = _basis(scene.track_turns, phi_mean, k, n_d, cfg.sr, cfg.duration_s)
        n_w = _window_length(k, delta, sr_d, n_d, r_n)
        power = np.zeros((m_n, r_n))
        coef_w: list[np.ndarray] = []
        conds = []
        for m in range(m_n):
            c_w, var_w, cond = _solve_windows(y[m], basis, n_w)
            power[m] = np.maximum((np.abs(c_w) ** 2 - var_w).mean(axis=0), 0.0)
            coef_w.append(c_w)
            conds.append(cond)
        ratio = coefficient_ratio(np.stack(coef_w))  # (M, R)
        return (
            power,
            ratio,
            dict(
                bandwidth_hz=bw,
                sr_d=sr_d,
                n_d=n_d,
                window_s=n_w / sr_d,
                cond=float(np.median(conds)),
            ),
        )

    per_order: dict[int, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for k in order_list:
        per_order[int(k)] = one_order(int(k))

    gamma = np.array(
        [
            [
                window_coherent_fraction(cfg.rotors[r], float(k), per_order[int(k)][2]["window_s"])
                if correct_decoherence
                else 1.0
                for k in order_list
            ]
            for r in range(r_n)
        ]
    )
    single = np.stack([per_order[int(k)][0].mean(axis=0) for k in order_list], axis=1)  # (R, K)
    single = 2.0 * single / np.maximum(gamma, 1e-6)
    diag: dict[str, Any] = dict(
        orders=order_list,
        window_s=np.array([per_order[int(k)][2]["window_s"] for k in order_list]),
        bandwidth_hz=np.array([per_order[int(k)][2]["bandwidth_hz"] for k in order_list]),
        cond=np.array([per_order[int(k)][2]["cond"] for k in order_list]),
        gamma=gamma,
    )
    if not multi:
        return dict(power=single, estimator="E2-single", **diag)

    if steering_orders is None:
        score = per_rotor_scores(cfg.rotors, order_list, duration_s=cfg.duration_s).min(axis=0)
        snr = single.sum(axis=0)
        rank = np.lexsort((-snr, -(score >= 1.0).astype(float)))
        steering_orders = order_list[np.sort(rank[: min(16, order_list.size)])]
    steering_orders = np.asarray(steering_orders, dtype=np.int64)
    missing = [int(k) for k in steering_orders if int(k) not in per_order]
    for k in missing:
        per_order[k] = one_order(k)
    coef_s = np.stack([per_order[int(k)][1] for k in steering_orders], axis=2)  # (M, R, K_s)
    gain, delay = estimate_steering(
        coef_s,
        steering_orders,
        cfg.carriers_rev_s,
        delay_bound_s=float(scene.diagnostics["delay_bound_s"]),
    )
    gain = gain / np.sqrt((gain**2).mean(axis=0, keepdims=True))  # mean |s|^2 = 1 per rotor

    multi_power = np.zeros((r_n, order_list.size))
    conds = np.zeros(order_list.size)
    for q, k in enumerate(order_list):
        k = int(k)
        _pw, _cf, info = per_order[k]
        y, sr_d = _demodulate(scene.audio, phi_mean, k, cfg.sr, info["bandwidth_hz"])
        n_d = y.shape[1]
        basis = _basis(scene.track_turns, phi_mean, k, n_d, cfg.sr, cfg.duration_s)
        s = gain * np.exp(-2j * np.pi * float(k) * cfg.carriers_rev_s[None, :] * delay)  # (M, R)
        # stack the mics along time: column r becomes s_rm A[t, r], so the
        # Gram matrix accumulates sum_m conj(s_rm) s_sm <A_r, A_s> -- two
        # rotors with different steering decorrelate even at zero beat.
        big_y = y.reshape(-1)
        big_a = (s[:, None, :] * basis[None, :, :]).reshape(-1, r_n)
        n_w = _window_length(k, delta, sr_d, n_d, r_n)
        c_w, var_w, cond = _solve_windows_stacked(big_y, big_a, n_w, m_n, n_d)
        multi_power[:, q] = 2.0 * np.maximum((np.abs(c_w) ** 2 - var_w).mean(axis=0), 0.0)
        conds[q] = cond
    multi_power = multi_power / np.maximum(gamma, 1e-6)
    diag["cond_multi"] = conds
    return dict(
        power=multi_power,
        estimator="E2-multi",
        steering_orders=steering_orders,
        steering_gain=gain,
        steering_delay=delay,
        **diag,
    )


def _solve_windows_stacked(
    y: np.ndarray, basis: np.ndarray, n_w: int, n_mics: int, n_d: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """:func:`_solve_windows` on a mic-stacked design: one window spans the
    SAME time slice of every mic, so the rows are gathered per window rather
    than reshaped."""
    n_win = max(n_d // n_w, 1)
    n_w = min(n_w, n_d)
    r_n = basis.shape[1]
    idx = (np.arange(n_mics)[:, None] * n_d)[:, :, None] + (
        np.arange(n_win)[:, None] * n_w + np.arange(n_w)[None, :]
    )[None]  # (M, W, n_w)
    rows = idx.transpose(1, 0, 2).reshape(n_win, n_mics * n_w)
    yy = y[rows]
    aa = basis[rows]
    gram = np.einsum("wtr,wts->wrs", aa.conj(), aa)
    rhs = np.einsum("wtr,wt->wr", aa.conj(), yy)
    eye = np.eye(r_n)[None]
    reg = 1e-12 * np.trace(gram, axis1=1, axis2=2).real[:, None, None] / r_n
    coef = np.linalg.solve(gram + reg * eye, rhs[:, :, None])[:, :, 0]
    resid = yy - np.einsum("wtr,wr->wt", aa, coef)
    sig2 = (np.abs(resid) ** 2).sum(axis=1) / max(n_mics * n_w - r_n, 1)
    inv_diag = np.linalg.inv(gram + reg * eye)[:, np.arange(r_n), np.arange(r_n)].real
    cond = float(np.median(np.linalg.cond(gram + reg * eye)))
    return coef, sig2[:, None] * inv_diag, cond


# ── scoring ─────────────────────────────────────────────────────────────────


def profile_error_db(
    estimate: np.ndarray, truth: np.ndarray, *, floor_db: float = -200.0
) -> np.ndarray:
    """``10 log10(estimate / truth)``, the signed dB error of a power estimate.

    A collapsed estimate (exactly zero after floor subtraction) maps to
    ``floor_db`` rather than ``-inf`` so a sweep table stays finite; the
    accompanying JSON records how many cells hit it.
    """
    est = np.asarray(estimate, dtype=np.float64)
    tru = np.asarray(truth, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 10.0 * np.log10(np.where(est > 0.0, est, np.nan) / tru)
    return np.where(np.isfinite(out), out, floor_db)


def scale_carriers(base_rev_s: np.ndarray, delta_hz: float) -> np.ndarray:
    """The real four-rotor carrier pattern rescaled to a minimum spacing.

    The offsets from the mean are multiplied by ``delta / delta_base``, so the
    3.02 : 1.08 : 0.83 Hz gap PATTERN of ``allMotors_70`` is preserved and only
    its scale moves. Preserving the pattern rather than equalising the gaps
    keeps one pair at ``delta`` and the others further out, which is the real
    rig's geometry.
    """
    f = np.sort(np.asarray(base_rev_s, dtype=np.float64))
    base_delta = float(np.min(np.diff(f)))
    return f.mean() + (f - f.mean()) * (float(delta_hz) / base_delta)


def make_config(
    rotors: tuple[RotorSpec, ...], *, delta_hz: float, base_carriers: np.ndarray, **kw: Any
) -> SceneConfig:
    """A :class:`SceneConfig` whose carriers are :func:`scale_carriers`'s."""
    car = scale_carriers(base_carriers, delta_hz)
    return SceneConfig(
        rotors=tuple(replace(r, carrier_rev_s=float(c)) for r, c in zip(rotors, car, strict=True)),
        **kw,
    )
