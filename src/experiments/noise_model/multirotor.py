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
* ``u_rk(t) = exp(i [k theta_r(t) + psi_rk(t)])`` is the v2 decoherence:
  one integrated-OU shaft error per rotor (``simulate_state``, the exact
  transition :mod:`.render` uses) plus Model R3's independent per-line Wiener
  phase of half-width ``gamma_rk`` Hz, drawn as the same random walk the
  renderer draws, so ``E[u_rk(t + tau) conj(u_rk(t))] = R_rk(tau)`` is
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

from . import model as MD
from . import spectrum as SP
from .lag import r_tau

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
    "interfering_lines",
    "joint_coherence_time_s",
    "line_half_width_hz",
    "line_snr_db",
    "make_config",
    "neighbour_contrast_db",
    "offset_spread_ratio",
    "per_rotor_scores",
    "profile_error_db",
    "record_resolvability_score",
    "refine_offsets",
    "refined_tracks",
    "resolvability_score",
    "rotor_from_fit",
    "scale_carriers",
    "simulate_scene",
    "window_coherent_fraction",
    "within_record_spec",
]

#: Speed of sound (m/s) used to turn an array aperture into a delay bound.
SPEED_OF_SOUND = 343.0


#: Fewest orders the steering delay matched filter needs to break its own
#: aliasing period ``1 / (k f)``.
MIN_STEERING_ORDERS = 4
#: ``exp(-1)``: the coherence-time threshold. One nat of the lag law.
COHERENCE_THRESHOLD = math.exp(-1.0)


# ── specifications ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RotorSpec:
    """One rotor: its constant carrier, its profile and its v2 dynamics.

    ``profile_db`` is a ``(K,)`` row of per-order line powers in dB, i.e. one
    row of a fit's ``params.profile.profile_db``, and ``gamma_hz`` is the
    matching ``(K,)`` row of Model R3 per-line Lorentzian half-widths in Hz,
    exactly as :func:`.lag.r_tau` takes them.
    """

    name: str
    carrier_rev_s: float
    profile_db: np.ndarray
    sigma_nu: float
    lam: float
    gamma_hz: np.ndarray

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
    #: (R,) realised mean shaft speed error of THIS record, rev/s. Truth for
    #: :func:`refine_offsets`; not visible to any estimator.
    shaft_offset_rev_s: np.ndarray
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
    """A :class:`RotorSpec` read verbatim out of a fit payload.

    Reads ``noise-v2-fit/2`` and, through :func:`.model.gamma_from_params`,
    the ``/1`` payloads of R1 and R2 as well.
    """
    p = fit["params"]
    prof = np.asarray(p["profile"]["profile_db"], dtype=np.float64)
    return RotorSpec(
        name=name,
        carrier_rev_s=float(np.atleast_1d(p["carrier_rev_s"])[rotor]),
        profile_db=prof[rotor].copy(),
        sigma_nu=float(p["sigma_nu"]),
        lam=float(p["lam"]),
        gamma_hz=MD.gamma_from_params(p)[rotor].copy(),
    )


# ── the criterion ───────────────────────────────────────────────────────────


def gamma_at(spec: RotorSpec, k: float) -> float:
    """The width of order ``k``, clamped to the spec's widest fitted order.

    The resolvability criterion probes orders a spec need not carry (a rotor
    is specified up to its own Nyquist, the criterion asks about any line).
    R3 has no law across ``k`` to extrapolate with, so the honest stand-in is
    the widest order the spec does carry, and it is the conservative one: it
    over-states the width, i.e. under-states resolvability.
    """
    g = np.atleast_1d(np.asarray(spec.gamma_hz, dtype=np.float64))
    return float(g[min(max(int(k) - 1, 0), int(g.size) - 1)])


def _r_of(spec: RotorSpec, tau: np.ndarray, k: float) -> np.ndarray:
    return np.asarray(
        r_tau(
            tau,
            float(k),
            sigma_nu=spec.sigma_nu,
            lam=spec.lam,
            gamma_hz=gamma_at(spec, k),
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


def within_record_spec(spec: RotorSpec, duration_s: float) -> RotorSpec:
    """``spec`` with the shaft scale replaced by its WITHIN-RECORD part.

    The lag law's ``sigma_nu`` is the STATIONARY spread of the shaft speed
    error, i.e. an ensemble statement. Every DREGON single-motor fit came back
    with ``lam`` between 0.002 and 0.16 s^-1, so over one 24 s support
    ``lam T <= 3.8`` and the shaft error is not ergodic inside the record: a
    single realisation is ``nu(0) T`` -- a constant FREQUENCY OFFSET -- plus a
    small Brownian wander. The campaign fits a constant carrier per rotor
    (the support index's window-refined carrier, frozen), so that
    offset is ESTIMATED, not suffered, and what is left to decohere the line
    inside the record is the wander about the record's own mean speed:

        sigma_eff^2 = sigma_nu^2 min(1, 2 lam T / 3)

    (the variance of a Brownian ``nu`` about its own time average over ``T``,
    with diffusion ``2 lam sigma_nu^2``, saturating at the stationary value
    once ``lam T >= 1.5``). For ``Motor4_70`` -- ``sigma_nu = 1.94`` rad/s at
    ``lam = 0.0027`` s^-1 -- this is a factor 4.8 narrower, and it is the
    difference between "unresolvable" and "resolvable" at 0.83 Hz.
    """
    factor = math.sqrt(min(1.0, 2.0 * float(spec.lam) * float(duration_s) / 3.0))
    return replace(spec, sigma_nu=float(spec.sigma_nu) * factor)


def _maybe_within(spec: RotorSpec, duration_s: float | None) -> RotorSpec:
    return spec if duration_s is None else within_record_spec(spec, duration_s)


def coherence_time_s(
    spec: RotorSpec, k: float, *, cap_s: float = 1e3, within_record_s: float | None = None
) -> float:
    """Lag at which ``R_rk`` first falls to ``exp(-1)``, capped at ``cap_s``.

    ``within_record_s`` applies :func:`within_record_spec` first, which is
    what a carrier-refined estimator actually faces.
    """
    tau = _lag_grid(cap_s)
    s = _maybe_within(spec, within_record_s)
    return min(_first_crossing(tau, _r_of(s, tau, k), COHERENCE_THRESHOLD), float(cap_s))


def joint_coherence_time_s(
    a: RotorSpec,
    b: RotorSpec,
    k: float,
    *,
    cap_s: float = 1e3,
    within_record_s: float | None = None,
) -> float:
    """``exp(-1)`` time of ``R_ak R_bk``.

    The beat of a PAIR is observable only while BOTH lines stay coherent: the
    cross term the least squares uses carries the product of the two residual
    phasors, so the product's decay -- not either factor's -- is the limit.
    """
    tau = _lag_grid(cap_s)
    rho = _r_of(_maybe_within(a, within_record_s), tau, k) * _r_of(
        _maybe_within(b, within_record_s), tau, k
    )
    return min(_first_crossing(tau, rho, COHERENCE_THRESHOLD), float(cap_s))


def resolvability_score(
    a: RotorSpec,
    b: RotorSpec,
    k: float,
    *,
    duration_s: float,
    delta_hz: float | None = None,
    within_record: bool = True,
) -> float:
    """``k delta min(tau_c(k), T)``: beat cycles of the pair actually observed.

    ``>= 1`` is the hypothesis' resolvability threshold. ``delta_hz`` defaults
    to the two rotors' carrier spacing. Because the shaft term is ballistic on
    the scale of one coherence time (``lam tau << 1``), ``tau_c(k) ~ C / k``
    and the score is nearly INDEPENDENT of ``k`` -- a prediction the sweep
    checks directly.

    ``within_record`` (the default) scores the CARRIER-REFINED problem via
    :func:`within_record_spec`; ``False`` scores the raw ensemble linewidth,
    which is what a reader that trusts the survey carrier faces.
    """
    d = (
        abs(float(a.carrier_rev_s) - float(b.carrier_rev_s))
        if delta_hz is None
        else float(delta_hz)
    )
    tau_c = joint_coherence_time_s(
        a,
        b,
        k,
        cap_s=max(10.0 * float(duration_s), 1.0),
        within_record_s=float(duration_s) if within_record else None,
    )
    return float(k) * d * min(tau_c, float(duration_s))


def offset_spread_ratio(rotors: tuple[RotorSpec, ...], *, delta_hz: float) -> float:
    """``max_r sigma_nu_r / (2 pi delta)``: can carrier refinement even ORDER
    the rotors?

    The realised per-record speed offset of rotor ``r`` is drawn from
    ``N(0, (sigma_nu / 2 pi)^2)`` in rev/s. Once that spread approaches the
    spacing the refined carriers can cross, and no estimator can say which
    recovered profile belongs to which rotor: the labels, not the profiles,
    become unidentifiable. Reported alongside the score because it is a
    DIFFERENT failure.
    """
    worst = max(float(r.sigma_nu) for r in rotors) / (2.0 * math.pi)
    return worst / max(float(delta_hz), 1e-12)


def per_rotor_scores(
    rotors: tuple[RotorSpec, ...],
    orders: np.ndarray,
    *,
    duration_s: float,
    within_record: bool = True,
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
                    out[i, q],
                    resolvability_score(
                        a, b, float(k), duration_s=duration_s, within_record=within_record
                    ),
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
    shaft_offset = np.zeros(r_n)
    for r_i, spec in enumerate(cfg.rotors):
        innov = rng_state.standard_normal((n, 2))
        theta, _nu = simulate_state(innov, lam=spec.lam, sigma=spec.sigma_nu, dt=dt)
        # the REALISED mean speed error of this record, rev/s: the quantity a
        # bench fit's refined constant carrier absorbs.
        shaft_offset[r_i] = float(theta[-1] - theta[0]) / (two_pi * (n - 1) * dt)
        for k in range(1, k_max + 1):
            # Model R3's per-line Wiener phase, the SAME draw render.py makes:
            # increments N(0, 4 pi gamma dt), so the rendered line carries the
            # Lorentzian of half-width gamma_rk the lag law assumes.
            step = math.sqrt(4.0 * math.pi * max(float(np.asarray(spec.gamma_hz)[k - 1]), 0.0) * dt)
            psi = np.cumsum(rng_eps.standard_normal(n) * step)
            # the angle is reduced modulo one turn BEFORE the trig, exactly as
            # spectrum._comb_autocovariance does: k phi reaches 4e7 turns.
            ang = two_pi * (np.remainder(k * phi[r_i], 1.0)) + k * theta + psi
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
        shaft_offset_rev_s=shaft_offset,
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


def line_half_width_hz(
    spec: RotorSpec, k: float, *, width_sigmas: float = 3.5, within_record_s: float | None = None
) -> float:
    """Half-width (Hz) of one line's core, from its own coherence time.

    On the scale of a coherence time the shaft term is ballistic
    (``lam tau_c << 1`` for every fitted rotor), so ``R_rk`` is Gaussian with
    ``sigma_f = 1 / (sqrt(2) 2 pi tau_c)`` and ``width_sigmas`` standard
    deviations hold the core. Deriving the width from ``tau_c`` rather than
    from ``k sigma_nu`` keeps the per-line width in it: a rotor whose
    ``gamma_rk`` dominates has a short ``tau_c`` and gets a wide band, which
    is what a broad line needs.

    ``within_record_s`` gives the width a CARRIER-REFINED reader sees
    (:func:`within_record_spec`); without it the width is the ensemble one,
    which for a small-``lam`` rotor is mostly an offset the refinement removes.
    """
    tau_c = coherence_time_s(spec, k, cap_s=100.0, within_record_s=within_record_s)
    return float(width_sigmas / (math.sqrt(2.0) * 2.0 * math.pi * max(tau_c, 1e-9)))


def refine_offsets(
    scene: MultiRotorScene,
    orders: np.ndarray,
    *,
    bound_rev_s: float = 0.5,
    margin_hz: float = 2.0,
    max_band_fraction: float = 0.4,
) -> dict[str, Any]:
    """Per-rotor residual speed offset (rev/s), the campaign's carrier refit.

    Inside one support the shaft error is a near-constant frequency offset
    (:func:`within_record_spec`), and every v2 bench fit refines and then FITS
    its carriers, so an estimator that reads the survey carrier is measuring
    an artefact, not a linewidth. This is that refinement: demodulate by the
    rotor's OWN order-``k`` phase, low-pass, and read where the residual line
    actually sits. The offset is magnified by ``k``, so a moderate order
    resolves it far better than order 1.

    TWO BOUNDS FIGHT HERE, and both are reported.

    * The search window must not reach the rotor's NEIGHBOUR, so it is
      ``+- min(bound, half the nearest carrier spacing)`` (``room``). A
      realised offset larger than that is the label-identifiability failure
      :func:`offset_spread_ratio` predicts -- flagged ``clipped``, never
      silently absorbed.
    * The analysis band must not reach the neighbouring ORDER, so
      ``k * room + margin <= max_band_fraction * f``, which caps the usable
      order at ``k_max_usable``. Orders above it are skipped; if none
      survives, ``n_orders_used`` is 0 and the offset is left at zero.

    The score is the campaign's own: ``sum_k log P_k(k delta)`` over the
    surviving orders, with ``P_k`` the baseband periodogram after
    demodulating by ``k phi_r`` -- :func:`spectrum.refine_bench_carrier`'s
    harmonic log-power sum, moved into the demodulated domain so the known
    track drift is already out of the way. A harmonic sum is what makes this
    robust: a single order whose line is under the floor contributes noise to
    the sum instead of deciding it.
    """
    cfg = scene.cfg
    carriers = cfg.carriers_rev_s
    r_n = cfg.n_rotors
    band_cap = float(max_band_fraction) * float(carriers.min())
    room = np.array(
        [
            min(
                float(bound_rev_s),
                0.5
                * min(
                    (abs(carriers[r] - carriers[j]) for j in range(r_n) if j != r),
                    default=float(bound_rev_s),
                ),
            )
            for r in range(r_n)
        ]
    )
    k_use = np.maximum(((band_cap - margin_hz) / np.maximum(room, 1e-9)).astype(np.int64), 0)
    grid = np.linspace(-1.0, 1.0, 2001)  # in units of room[r]
    best = np.zeros(r_n)
    clipped = np.zeros(r_n, dtype=bool)
    n_used = np.zeros(r_n, dtype=np.int64)
    score_gain = np.zeros(r_n)
    for r in range(r_n):
        score = np.zeros(grid.size)
        used = 0
        for k_ in np.asarray(orders, dtype=np.int64):
            k = int(k_)
            if k > int(k_use[r]) or k > scene.k_max:
                continue
            bw = float(k) * room[r] + margin_hz
            y, sr_d = _demodulate(scene.audio, float(k) * scene.track_turns[r], cfg.sr, bw)
            n_d = y.shape[1]
            w = hann_window(n_d)
            p = (np.abs(np.fft.fft(y * w[None, :], axis=1)) ** 2).sum(axis=0)
            f = np.fft.fftshift(np.fft.fftfreq(n_d, d=1.0 / sr_d))
            p = np.fft.fftshift(p)
            lp = np.log(np.maximum(p, 1e-300))
            score = score + np.interp(grid * (float(k) * room[r]), f, lp)
            used += 1
        if used == 0:
            continue
        j = int(np.argmax(score))
        best[r] = grid[j] * room[r]
        n_used[r] = used
        score_gain[r] = float((score[j] - np.median(score)) / used)
        clipped[r] = abs(grid[j]) > 0.9
    return dict(
        offsets_rev_s=best,
        score_gain=score_gain,
        n_orders_used=n_used,
        clipped=clipped,
        room_rev_s=room,
        k_max_usable=k_use,
        bound_rev_s=float(bound_rev_s),
    )


def refined_tracks(
    scene: MultiRotorScene, offsets_rev_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(carriers, phi)`` with a constant per-rotor speed offset folded in.

    ``phi_r(t) + delta f_r t`` in turns: the estimator's track, refined the way
    the bench fit refines its constants.
    """
    off = np.asarray(offsets_rev_s, dtype=np.float64).reshape(-1)
    t = np.arange(scene.audio.shape[1]) / float(scene.sr)
    return scene.cfg.carriers_rev_s + off, scene.track_turns + off[:, None] * t[None, :]


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
    offsets_rev_s: np.ndarray | None = None,
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
    carriers = cfg.carriers_rev_s + (
        0.0 if offsets_rev_s is None else np.asarray(offsets_rev_s, dtype=np.float64).reshape(-1)
    )
    freq = carriers[rot_idx] * ord_idx
    half = np.array(
        [
            line_half_width_hz(
                cfg.rotors[r],
                float(k),
                width_sigmas=width_sigmas,
                within_record_s=cfg.duration_s,
            )
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
    audio: np.ndarray, phase_turns: np.ndarray, sr: int, bandwidth_hz: float
) -> tuple[np.ndarray, float]:
    """``(M, n_d)`` band-limited analytic baseband about ``phase_turns``.

    ``y_m = x_m exp(-i 2 pi phase)`` puts the chosen line at baseband and its
    negative-frequency image near ``-2 k f``; keeping only ``|f| <=
    bandwidth`` deletes the image and every line further away than the
    bandwidth. The resampling is spectral (select the retained bins of the
    length-``N`` complex FFT and inverse transform the short vector), which is
    exact for a signal already confined to the band and costs one FFT pair per
    mic.
    """
    n = audio.shape[1]
    y = audio * np.exp(-2j * np.pi * np.remainder(phase_turns, 1.0))[None, :]
    half = int(min(max(math.ceil(bandwidth_hz * n / sr), 4), (n - 1) // 2))
    yf = np.fft.fft(y, axis=1)
    sel = np.concatenate([yf[:, : half + 1], yf[:, n - half :]], axis=1)
    n_d = sel.shape[1]
    return np.fft.ifft(sel, axis=1) * (n_d / n), float(n_d) / (n / float(sr))


def _resample_phase(phase_turns: np.ndarray, n_d: int, sr: int, duration_s: float) -> np.ndarray:
    """``phase_turns`` read on the decimated grid by linear interpolation.

    The phase difference of two comb lines inside one analysis band advances
    by at most ``bandwidth`` turns per second and is smooth on the 16 kHz
    grid, so linear interpolation costs ``O((2 pi f / sr)^2)`` rad -- below
    1e-6 rad for every band this module opens.
    """
    n = phase_turns.size
    t_full = np.arange(n) / float(sr)
    t_d = np.arange(n_d) * (duration_s / n_d)
    return np.interp(t_d, t_full, phase_turns)


def interfering_lines(
    carriers_rev_s: np.ndarray,
    rotor: int,
    order: int,
    *,
    k_max: int,
    bandwidth_hz: float,
    guard: float = 0.9,
) -> list[tuple[int, int, float]]:
    """``[(rotor, order, offset_hz)]``: every comb line inside the band.

    THE REASON THIS EXISTS. "The R lines of order ``k``" is only a group while
    ``k delta`` stays below the carrier itself. At ``delta = 0.83`` Hz and
    ``f = 67.7`` rev/s the order-104 lines of the four rotors are spread over
    260 Hz while consecutive ORDERS are 68 Hz apart, so the neighbours of a
    high-order line are other orders of other rotors, not its own order's
    partners. An estimator that demodulates the mean track and solves against
    four order-``k`` bases is then solving the wrong problem: it has three
    wrong columns and misses the lines that actually overlap.

    So the design matrix is built from the band, not from the order: every
    ``(s, k')`` whose line ``k' f_s`` lies within ``guard * bandwidth`` of the
    target ``k f_r``, the target itself first (offset exactly 0).
    """
    f = np.asarray(carriers_rev_s, dtype=np.float64)
    f0 = float(order) * float(f[rotor])
    lim = float(guard) * float(bandwidth_hz)
    out = [(int(rotor), int(order), 0.0)]
    for s in range(f.size):
        lo = max(1, int(math.floor((f0 - lim) / f[s])))
        hi = min(int(k_max), int(math.ceil((f0 + lim) / f[s])))
        for kp in range(lo, hi + 1):
            if s == rotor and kp == order:
                continue
            off = kp * float(f[s]) - f0
            if abs(off) <= lim:
                out.append((int(s), int(kp), float(off)))
    return out


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
    ratio: np.ndarray,
    orders: np.ndarray,
    carriers_rev_s: np.ndarray,
    *,
    delay_bound_s: float,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-rotor steering from the RESOLVABLE orders' mic-relative ratios.

    ``ratio`` is ``(M, R, K_s)``, :func:`coefficient_ratio` stacked over the
    orders ``orders``. The model is ``c_rkm = a_rk g_rm exp(-i 2 pi k f_r
    eta_rm)`` with ``a_rk`` an unknown per-(rotor, order) amplitude and phase,
    so only the mic-RELATIVE steering is identifiable -- which is all the
    joint solve needs. The delay difference is found by a matched filter over
    ``eta`` (no phase unwrapping: the orders need not be consecutive, and a
    grid search over a bounded physical delay is both robust and cheap).

    ``weights`` is ``(R, K_s)``, normally the per-rotor estimated line power.
    It is not a refinement either: the fitted profiles have 30-40 dB nulls,
    so the same high order that pins one rotor's delay is under the floor for
    another. Weighting by power per ROTOR is what keeps a null from voting.

    Returns ``(gain (M, R), delay (M, R))`` with mic 0 as the reference.
    """
    k = np.asarray(orders, dtype=np.float64)
    w = (
        np.ones((ratio.shape[1], k.size))
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-300)
    gain = np.einsum("mrk,rk->mr", np.abs(ratio), w)
    grid = np.linspace(-2.0 * delay_bound_s, 2.0 * delay_bound_s, 4001)
    phase = ratio / np.maximum(np.abs(ratio), 1e-300)
    delay = np.zeros(gain.shape)
    for r in range(gain.shape[1]):
        kern = np.exp(2j * np.pi * np.outer(grid, k * float(carriers_rev_s[r])))  # (G, K_s)
        sc = np.abs((phase[:, r, :] * w[r][None, :]) @ kern.T)  # (M, G)
        delay[:, r] = grid[np.argmax(sc, axis=1)]
    return gain, delay


def _extrapolate_zero_window(
    p_long: np.ndarray, p_short: np.ndarray, t_long: float, t_short: float
) -> np.ndarray:
    """Gaussian extrapolation of a windowed power estimate to zero window.

    A constant-``psi`` design matrix attenuates the recovered power by the
    Bartlett average of the residual-phase autocorrelation, which for every
    process in the v2 law is ``1 - a T_w^2 + O(T_w^4)`` at small ``T_w``. Two
    window lengths therefore identify ``a`` and the zero-window limit:

        log P(0) = (T_l^2 log P_s - T_s^2 log P_l) / (T_l^2 - T_s^2).

    Doing it this way rather than dividing by the lag law's
    :func:`window_coherent_fraction` means the estimator needs NO knowledge of
    the dynamics -- which matters here, because inside one support the shaft
    term's ensemble width is dominated by a constant offset (see
    :func:`within_record_spec`) and the ensemble correction would over-correct
    by several dB on a small-``lam`` rotor. The correction is clamped to 10 dB:
    beyond that the two solves are noise and the raw value is the honest one.
    """
    tl2, ts2 = float(t_long) ** 2, float(t_short) ** 2
    ok = (p_long > 0.0) & (p_short > 0.0) & (tl2 > ts2 * 1.05)
    with np.errstate(divide="ignore", invalid="ignore"):
        log0 = (tl2 * np.log(p_short) - ts2 * np.log(p_long)) / (tl2 - ts2)
        gain = np.clip(log0 - np.log(np.where(p_long > 0.0, p_long, 1.0)), 0.0, math.log(10.0))
    return np.where(ok, p_long * np.exp(gain), p_long)


def estimate_e2(
    scene: MultiRotorScene,
    orders: np.ndarray,
    *,
    multi: bool = False,
    offsets_rev_s: np.ndarray | None = None,
    extrapolate: bool = True,
    steering: tuple[np.ndarray, np.ndarray] | None = None,
    bandwidth_margin_hz: float = 4.0,
    max_band_fraction: float = 0.4,
    n_beat_cycles: float = 2.0,
    window_cap_factor: float = 1.0,
    min_windows: int = 8,
    steering_score_min: float = 3.0,
    n_steering_orders: int = 16,
) -> dict[str, Any]:
    """Phase-aware per-rotor order power.

    ONE SOLVE PER TARGET LINE ``(r, k)``. The audio is demodulated by that
    line's own known phase ``k phi_r(t)`` and low-passed to a band that holds
    the line's core and whatever else is near it; the design matrix has one
    column per comb line inside the band (:func:`interfering_lines`), each
    column being the KNOWN relative phase ``exp(i 2 pi (k' phi_s - k
    phi_r))``; the complex amplitudes are solved by least squares over windows
    long enough for the nearest interferer's beat to be seen
    (``n_beat_cycles`` of it) and the target column's power is read off.

    Demodulating per line rather than per order is not a refinement, it is
    required: past ``k ~ f / delta`` the order-``k`` lines of the rotors are
    spread wider than the spacing between ORDERS, so an order group is not a
    group any more (see :func:`interfering_lines`).

    THE UNITS. ``y_m(t)`` is the analytic baseband, so the real comb's
    ``Re[c exp(i Theta)]`` contributes ``c / 2`` and the recovered power is
    ``P = |2 c_LS|^2 / 2 = 2 |c_LS|^2``. Two corrections, neither using a
    fitted dynamics parameter:

    * the LS noise variance ``sigma^2 [ (A^H A)^-1 ]_rr`` is SUBTRACTED from
      ``|c_LS|^2`` -- a complex LS coefficient's squared modulus is biased up
      by exactly its own variance, and near the floor that bias is the whole
      number;
    * the window's coherent loss is removed by
      :func:`_extrapolate_zero_window` from a second solve at half the window.

    ``offsets_rev_s`` (normally :func:`refine_offsets`' output) folds a
    constant per-rotor speed offset into the track before demodulating, which
    is what the bench fit's refined-and-fitted carrier does.

    ``multi=True`` imposes ONE steering vector per rotor: column ``(s, k')``
    becomes ``g_sm exp(-i 2 pi k' f_s eta_sm)`` times the same time basis,
    stacked over mics, so two lines whose time bases are nearly collinear are
    still separated when their rotors' steering vectors differ. The steering
    is taken from ``steering`` (a ``(gain, delay)`` pair, normally the
    ``E2-single`` pass' own :func:`estimate_steering` output) or estimated
    inside this call from the per-line mic ratios.
    """
    cfg = scene.cfg
    order_list = np.asarray(orders, dtype=np.int64)
    carriers, track = (
        (cfg.carriers_rev_s, scene.track_turns)
        if offsets_rev_s is None
        else refined_tracks(scene, offsets_rev_s)
    )
    r_n, m_n = cfg.n_rotors, cfg.n_mics
    k_max = scene.k_max
    band_cap = float(max_band_fraction) * float(carriers.min())
    delta_min = float(np.min(np.diff(np.sort(carriers)))) if carriers.size > 1 else float("inf")
    steer_orders = np.zeros(0, dtype=np.int64)

    def core_bw(r: int, k: int) -> float:
        """Band that holds the target line: its within-record width, the part
        of the planted track drift the demodulation cannot remove (none, the
        track is known) and a fixed margin."""
        half = line_half_width_hz(
            cfg.rotors[r], float(k), width_sigmas=4.0, within_record_s=cfg.duration_s
        )
        return 2.0 * half + bandwidth_margin_hz

    def band_for(r: int, k: int) -> float:
        spread = float(np.max(np.abs(carriers - carriers[r]))) * float(k)
        return float(min(max(core_bw(r, k), min(spread, band_cap)), band_cap))

    def solve_line(
        r: int, k: int, gain_del: tuple[np.ndarray, np.ndarray] | None
    ) -> dict[str, Any]:
        """One target line. ``gain_del`` present -> the mic-stacked joint solve
        with the rank-one steering constraint; absent -> per-mic solves whose
        powers are averaged and whose mic ratio is returned."""
        bw = band_for(r, k)
        lines = interfering_lines(carriers, r, k, k_max=k_max, bandwidth_hz=bw, guard=0.9)
        y, sr_d = _demodulate(scene.audio, float(k) * track[r], cfg.sr, bw)
        n_d = y.shape[1]
        cols = [
            _resample_phase(kp * track[s] - float(k) * track[r], n_d, cfg.sr, cfg.duration_s)
            for s, kp, _off in lines
        ]
        basis = np.exp(2j * np.pi * np.stack(cols, axis=1))  # (n_d, L)
        near = min((abs(o) for _s, _k, o in lines[1:]), default=float("inf"))
        # TWO constraints on the window, and where they conflict the cap wins.
        # Long enough: the nearest interferer's beat must turn ``n_beat_cycles``
        # times inside it, or the two columns are collinear and the split is
        # arbitrary. Short enough: beyond one coherence time the constant-psi
        # column stops matching the line and the recovered power falls like
        # ``tau_c / T_w`` -- a loss no small-``T_w`` expansion can undo. Taking
        # the cap keeps the estimate nearly unbiased and lets the leakage show,
        # which is the failure the criterion is about.
        tau_c = coherence_time_s(
            cfg.rotors[r], float(k), cap_s=cfg.duration_s, within_record_s=cfg.duration_s
        )
        want = (
            float(n_beat_cycles) / near
            if math.isfinite(near)
            else cfg.duration_s / float(min_windows)
        )
        want = min(want, float(window_cap_factor) * tau_c)
        n_w = int(min(max(int(round(want * sr_d)), len(lines) + 2), n_d))
        n_h = max(n_w // 2, len(lines) + 2)
        info: dict[str, Any] = dict(
            n_columns=len(lines),
            window_s=n_w / sr_d,
            window_short_s=n_h / sr_d,
            bandwidth_hz=bw,
            nearest_line_hz=near,
            beat_cycles=near * n_w / sr_d if math.isfinite(near) else np.inf,
            coherence_time_s=tau_c,
            ratio=np.zeros(m_n, dtype=complex),
        )
        if gain_del is not None:
            g, dly = gain_del
            s_vec = np.stack(
                [
                    g[:, s] * np.exp(-2j * np.pi * kp * carriers[s] * dly[:, s])
                    for s, kp, _o in lines
                ],
                axis=1,
            )  # (M, L)
            big_y = y.reshape(-1)
            big_a = (s_vec[:, None, :] * basis[None, :, :]).reshape(-1, len(lines))
            c_w, var_w, cond = _solve_windows_stacked(big_y, big_a, n_w, m_n, n_d)
            info["power"] = 2.0 * max(float((np.abs(c_w[:, 0]) ** 2 - var_w[:, 0]).mean()), 0.0)
            if n_h < n_w:
                c_h, var_h, _c = _solve_windows_stacked(big_y, big_a, n_h, m_n, n_d)
                info["power_short"] = 2.0 * max(
                    float((np.abs(c_h[:, 0]) ** 2 - var_h[:, 0]).mean()), 0.0
                )
            else:
                info["power_short"] = info["power"]
            info["cond"] = cond
            return info
        pw = np.zeros(m_n)
        pw_s = np.zeros(m_n)
        coef_w = []
        cs = []
        for m in range(m_n):
            c_w, var_w, cond = _solve_windows(y[m], basis, n_w)
            pw[m] = max(float((np.abs(c_w[:, 0]) ** 2 - var_w[:, 0]).mean()), 0.0)
            coef_w.append(c_w[:, :1])
            cs.append(cond)
            if n_h < n_w:
                c_h, var_h, _c = _solve_windows(y[m], basis, n_h)
                pw_s[m] = max(float((np.abs(c_h[:, 0]) ** 2 - var_h[:, 0]).mean()), 0.0)
            else:
                pw_s[m] = pw[m]
        info["power"] = 2.0 * float(pw.mean())
        info["power_short"] = 2.0 * float(pw_s.mean())
        info["ratio"] = coefficient_ratio(np.stack(coef_w))[:, 0]
        info["cond"] = float(np.median(cs))
        return info

    def run(
        order_set: np.ndarray, gain_del: tuple[np.ndarray, np.ndarray] | None
    ) -> dict[str, Any]:
        shape = (r_n, order_set.size)
        acc: dict[str, Any] = dict(
            power=np.zeros(shape),
            power_short=np.zeros(shape),
            ratio=np.zeros((m_n, *shape), dtype=complex),
            n_columns=np.zeros(shape, dtype=np.int64),
            cond=np.zeros(shape),
            window_s=np.zeros(shape),
            window_short_s=np.zeros(shape),
            bandwidth_hz=np.zeros(shape),
            nearest_line_hz=np.zeros(shape),
            beat_cycles=np.zeros(shape),
            coherence_time_s=np.zeros(shape),
        )
        for r in range(r_n):
            for q, k_ in enumerate(order_set):
                one = solve_line(r, int(k_), gain_del)
                acc["ratio"][:, r, q] = one.pop("ratio")
                for key, val in one.items():
                    acc[key][r, q] = val
        return acc

    single = run(order_list, None)
    gain_del = steering
    if multi and gain_del is None:
        # The steering vectors are read off a CONSECUTIVE block of orders that
        # are already separable (record score above ``steering_score_min``) and
        # then imposed on the rest. Consecutive matters: the delay matched
        # filter's alias period is ``1 / (Delta k f)``, so a sparse ladder of
        # high orders aliases inside the array's own delay range while a
        # consecutive block puts the first alias at ``1 / f`` = 15 ms, far
        # outside it.
        k0 = max(
            int(math.ceil(float(steering_score_min) / max(delta_min * cfg.duration_s, 1e-9))), 1
        )
        block = np.arange(k0, min(k0 + int(n_steering_orders), k_max + 1), dtype=np.int64)
        if block.size < MIN_STEERING_ORDERS:
            block = np.arange(max(k_max - int(n_steering_orders) + 1, 1), k_max + 1, dtype=np.int64)
        steer = run(block, None)
        gain, delay = estimate_steering(
            steer["ratio"],
            block,
            carriers,
            delay_bound_s=float(scene.diagnostics["delay_bound_s"]),
            weights=steer["power"],
        )
        gain = gain / np.sqrt((gain**2).mean(axis=0, keepdims=True))
        gain_del = (gain, delay)
        steer_orders = block
    acc = run(order_list, gain_del) if multi else single
    power = acc["power"]
    power_short = acc["power_short"]
    ratios = single["ratio"]
    n_cols = acc["n_columns"]
    conds = acc["cond"]
    win_s = acc["window_s"]
    win_short_s = acc["window_short_s"]
    bws = acc["bandwidth_hz"]
    nearest = acc["nearest_line_hz"]

    out = power.copy()
    if extrapolate:
        for r in range(r_n):
            for q in range(order_list.size):
                out[r, q] = float(
                    _extrapolate_zero_window(
                        np.array([power[r, q]]),
                        np.array([power_short[r, q]]),
                        win_s[r, q],
                        win_short_s[r, q],
                    )[0]
                )
    res: dict[str, Any] = dict(
        power=out,
        power_raw=power,
        estimator="E2-multi" if multi else "E2-single",
        orders=order_list,
        n_columns=n_cols,
        cond=conds,
        window_s=win_s,
        window_short_s=win_short_s,
        bandwidth_hz=bws,
        nearest_line_hz=nearest,
        beat_cycles=acc["beat_cycles"],
        coherence_time_s=acc["coherence_time_s"],
        offsets_rev_s=np.zeros(r_n) if offsets_rev_s is None else np.asarray(offsets_rev_s),
        steering_orders=steer_orders,
    )
    if gain_del is not None:
        res["steering_gain"], res["steering_delay"] = gain_del
    if not multi:
        res["mic_ratio"] = ratios
    return res


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


def record_resolvability_score(k: float, delta_hz: float, duration_s: float) -> float:
    """``k delta T``: beat cycles of a pair inside the whole RECORD.

    The identifiability bound of a least squares against KNOWN bases, as
    opposed to :func:`resolvability_score`'s coherence bound. The Gram matrix
    of two beat bases over ``T`` is conditioned as soon as their relative
    phase turns once, and that is a property of the tracks alone -- no
    dynamics parameter enters. Where the two scores disagree, the sweep says
    which one the data obeys.
    """
    return float(k) * float(delta_hz) * float(duration_s)


def line_snr_db(scene: MultiRotorScene, orders: np.ndarray, bandwidth_hz: np.ndarray) -> np.ndarray:
    """``(R, K)`` planted line power over the floor power in its own band.

    The floor's in-band variance is ``(2 / N) sum_{|f - k f_r| <= bw} psd``,
    the same Parseval bookkeeping :func:`simulate_scene` uses to set the
    level. A cell whose SNR is negative is not a statement about
    resolvability at all -- the line is not in the recording -- so the sweep
    gates its verdict on this number instead of reporting an error that only
    measures the floor.
    """
    n = scene.audio.shape[1]
    freqs = np.fft.rfftfreq(n, d=1.0 / scene.sr)
    order_list = np.asarray(orders, dtype=np.int64)
    out = np.zeros((scene.cfg.n_rotors, order_list.size))
    for r in range(scene.cfg.n_rotors):
        for q, k in enumerate(order_list):
            f0 = float(scene.cfg.carriers_rev_s[r]) * float(k)
            sel = np.abs(freqs - f0) <= float(bandwidth_hz[r, q])
            fp = 2.0 * float(scene.floor_psd[sel].sum()) / n
            out[r, q] = 10.0 * math.log10(
                max(float(scene.power_true[r, int(k) - 1]), 1e-300) / max(fp, 1e-300)
            )
    return out


def neighbour_contrast_db(
    scene: MultiRotorScene, orders: np.ndarray, bandwidth_hz: np.ndarray
) -> np.ndarray:
    """``(R, K)`` dB by which the loudest OTHER line in the band beats the
    target.

    A separation error only shows up as a profile error in proportion to the
    contrast: leaking 1% of a neighbour 30 dB louder is a 10 dB error, leaking
    1% of an equal neighbour is 0.04 dB. Reported next to the criterion score
    so the sweep's failures can be read as ``contrast + leakage(score)``
    rather than as an unexplained scatter.
    """
    order_list = np.asarray(orders, dtype=np.int64)
    carriers = scene.cfg.carriers_rev_s
    out = np.full((scene.cfg.n_rotors, order_list.size), -np.inf)
    for r in range(scene.cfg.n_rotors):
        for q, k in enumerate(order_list):
            lines = interfering_lines(
                carriers, r, int(k), k_max=scene.k_max, bandwidth_hz=float(bandwidth_hz[r, q])
            )
            tgt = max(float(scene.power_true[r, int(k) - 1]), 1e-300)
            others = [float(scene.power_true[s, kp - 1]) for s, kp, _o in lines[1:]]
            if others:
                out[r, q] = 10.0 * math.log10(max(max(others), 1e-300) / tgt)
    return out
