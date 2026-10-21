"""Reading a ``noise-v4-fit/3`` payload: the laws of noise model v4, in numpy.

``docs/explainers/noise-model-v4.qmd`` § The model is the specification; this
module is what a RENDERER (and a check that compares a render against the
model) needs out of the payload, so that :mod:`.render` and the tests read one
definition of each law:

* :meth:`V4Params.line_power` — the MEAN POWER of every aerodynamic line
  ``10^{p^a_rk/10} s_r^{alpha_a}`` and every motor line
  ``10^{p^m_rn/10} s_r^{alpha_m}`` at the relative speed ``s = f / f_ref``;
* :class:`Pedestal` — the per-order log-amplitude OU ``(sigma2_amp,
  gamma_amp_hz)`` and phase OU ``(sigma2_phase, gamma_phase_hz)``;
* :class:`ShaftLaw` — ``sigma_nu(s) = 2 pi f_ref eps s^a_sigma`` (rad/s) and
  ``lam(s) = lam_ref s^b_lam`` (1/s), shared by the rotors;
* :class:`FloorLaw` — ONE rotor's floor ``F(f)`` at ``f_ref`` (the nv3r4
  spline: ``mean_db + sigma_B (L z)`` interpolated in log-frequency on the
  payload's 14 ``ctrl_hz``, no tilt, in the analysis periodogram units of
  :class:`experiments.noise_model.spectrum.FloorBasis`: white noise of
  variance ``v`` reads ``v``) and its speed exponent
  ``b(f) = b0 + b1 log2(max(f, band_lo) / 2 kHz)``, combined over the rotors
  by the payload's ``floor_rule`` (:data:`FLOOR_RULES`: the bench's per-rotor
  sum, or MD2 flight's one floor on the rotor-mean speed);
* :class:`MicResponse` — ``E_rm(f)`` in dB, split into the part that differs
  between the rotors (:meth:`~MicResponse.rotor_db`: the bench pair EQ, or the
  flight geometric term ``Delta G_rm``) and the part shared by them
  (:meth:`~MicResponse.array_db`: the flight array response ``A_m(f)``, zero on
  the bench);
* :func:`line_lag_law` — eq-lag; :func:`expected_periodogram` — the
  Hann-window expected periodogram of a CONSTANT-speed render.

LINE POWER CONVENTION. ``p`` is the line's MEAN power and the lag law is one at
``tau = 0``, so the complex envelope of an aerodynamic line must have
``E|env|^2 = 2 * 10^{p/10} s^alpha``: with ``a`` an OU of variance
``sigma2_amp`` that is ``exp(a - sigma2_amp)``, whose square has mean one and
whose lag correlation is ``exp(sigma2_amp (rho - 1))``. (``exp(a -
sigma2_amp / 2)`` keeps the mean AMPLITUDE instead and adds ``e^{sigma2_amp}``
to the power: +1.26 dB at the 0.29 cap.)
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from data_processing.noise_model import FIT_SCHEMA_V4
from data_processing.noise_model.constants import SPEED_FLOOR_RPS
from data_processing.noise_model.floor import floor_geometry, floor_power_spectrum, interp_matrix
from data_processing.noise_model.ou import integrated_ou_increment_var
from data_processing.noise_model.params import array_response_db
from data_processing.noise_model.spectrum import floor_shape_chol
from data_processing.noise_model.v3 import wind_shape

__all__ = [
    "FLOOR_EXP_REF_HZ",
    "FLOOR_RULE_ONE",
    "FLOOR_RULE_SUM",
    "FLOOR_RULES",
    "WORK_RATE_HZ",
    "FloorLaw",
    "MicResponse",
    "Pedestal",
    "ShaftLaw",
    "V4Params",
    "expected_periodogram",
    "hann_window",
    "line_lag_law",
]

#: The render work rate of a v4 payload that records none: the 32 kHz the GPU
#: flight fits run at, whose chain transfer is 64 kHz's to 0.002 dB over the band
#: (docs/experiments/noise-model-v3.md § "GPU round") at half the line cost.
WORK_RATE_HZ = 32000

#: The pivot of the floor's speed exponent ``b(f) = b0 + b1 log2(f / 2 kHz)``.
FLOOR_EXP_REF_HZ = 2000.0

#: How the rotors make the floor (payload ``params.floor.floor_rule``).
#: ``per_rotor_sum`` (the bench, and any payload without the key): one floor
#: per rotor, ``sum_r 10^{E_rm/10} F(f) s_r^{b(f)}``. ``one_floor_mean_speed``
#: (MD2 flight, decision 2026-10-21): ONE rig floor ``F(f) sbar^{b(f)}`` driven
#: by the rotor-mean speed ``sbar = mean_r s_r``, with the array response and
#: no per-rotor ``E_rm``. Both add the static share ``F(f) s``.
FLOOR_RULE_SUM = "per_rotor_sum"
FLOOR_RULE_ONE = "one_floor_mean_speed"
FLOOR_RULES = (FLOOR_RULE_SUM, FLOOR_RULE_ONE)

TWO_PI = 2.0 * math.pi


def _arr(x: Any, ndim: int, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != ndim or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite {ndim}-d array, got shape {a.shape}")
    return a


@dataclass(frozen=True)
class Pedestal:
    """The per-order line pedestal, shared by the rotors and the mics, ``(K,)`` each."""

    sigma2_amp: np.ndarray
    gamma_amp_hz: np.ndarray
    sigma2_phase: np.ndarray
    gamma_phase_hz: np.ndarray

    @classmethod
    def from_mapping(cls, d: Mapping[str, Any]) -> Pedestal:
        arrays = [_arr(d[k], 1, f"pedestal.{k}") for k in _PEDESTAL_KEYS]
        if len({a.size for a in arrays}) != 1:
            raise ValueError("pedestal arrays must all have K entries")
        if any(np.any(a < 0.0) for a in arrays):
            raise ValueError("pedestal variances and rates must be non-negative")
        return cls(*arrays)


_PEDESTAL_KEYS = ("sigma2_amp", "gamma_amp_hz", "sigma2_phase", "gamma_phase_hz")


@dataclass(frozen=True)
class ShaftLaw:
    """eq-shaft-law: the shaft OU's scale and rate as functions of the speed."""

    eps: float
    a_sigma: float
    lam_ref_per_s: float
    b_lam: float
    f_ref_rev_s: float

    @classmethod
    def from_mapping(cls, d: Mapping[str, Any], f_ref: float) -> ShaftLaw:
        return cls(
            eps=float(d["eps"]),
            a_sigma=float(d["a_sigma"]),
            lam_ref_per_s=float(d["lam_ref_per_s"]),
            b_lam=float(d["b_lam"]),
            f_ref_rev_s=float(d.get("f_ref_rev_s", f_ref)),
        )

    def sigma_nu(self, s: Any) -> np.ndarray:
        """Stationary sd of the speed error at relative speed ``s``, rad/s."""
        s = np.maximum(np.asarray(s, dtype=np.float64), SPEED_FLOOR_RPS / self.f_ref_rev_s)
        return TWO_PI * self.f_ref_rev_s * self.eps * s**self.a_sigma

    def lam(self, s: Any) -> np.ndarray:
        """Rate of the speed error at relative speed ``s``, 1/s."""
        s = np.maximum(np.asarray(s, dtype=np.float64), SPEED_FLOOR_RPS / self.f_ref_rev_s)
        return self.lam_ref_per_s * s**self.b_lam


@dataclass(frozen=True)
class FloorLaw:
    """eq-floor without the mics: the floor ``F(f)`` at ``f_ref`` and its exponent
    ``b(f)``; ``rule`` (:data:`FLOOR_RULES`) says whether ``F`` is ONE rotor's
    floor summed over the rotors or the rig's one floor on the mean speed."""

    mean_db: float
    shape_z: np.ndarray
    shape_sd_db: float
    ctrl_hz: np.ndarray
    b0: float
    b1: float
    static_rel: float
    band_lo_hz: float
    rule: str = FLOOR_RULE_SUM

    @classmethod
    def from_mapping(cls, d: Mapping[str, Any], band_lo_hz: float) -> FloorLaw:
        ctrl = _arr(d["ctrl_hz"], 1, "floor.ctrl_hz")
        z = _arr(d["shape_z"], 1, "floor.shape_z")
        if z.size != ctrl.size or np.any(np.diff(ctrl) <= 0.0) or ctrl[0] <= 0.0:
            raise ValueError("floor.ctrl_hz must be increasing, positive, and as long as shape_z")
        rule = str(d.get("floor_rule") or FLOOR_RULE_SUM)
        if rule not in FLOOR_RULES:
            raise ValueError(f"floor.floor_rule must be one of {FLOOR_RULES}, got {rule!r}")
        return cls(
            mean_db=float(d["mean_db"]),
            shape_z=z,
            shape_sd_db=float(d["shape_sd_db"]),
            ctrl_hz=ctrl,
            b0=float(d["b0"]),
            b1=float(d["b1"]),
            static_rel=float(d.get("static_rel") or 0.0),
            band_lo_hz=float(band_lo_hz),
            rule=rule,
        )

    def ctrl_db(self) -> np.ndarray:
        """The 14 control values relative to ``mean_db``: ``sigma_B (L z)``."""
        return self.shape_sd_db * (floor_shape_chol(self.ctrl_hz) @ self.shape_z)

    def psd(self, freqs_hz: Any) -> np.ndarray:
        """``F(f)``: ONE rotor's floor at ``f_ref`` on ``freqs_hz``, periodogram units."""
        shape, tilt = floor_geometry(freqs_hz, self.ctrl_hz)
        return np.asarray(
            floor_power_spectrum(
                shape,
                tilt,
                mean_db=self.mean_db,
                ctrl_db=self.ctrl_db(),
                tilt_db_oct=0.0,
                rate_factor=1.0,
            ),
            dtype=np.float64,
        )

    def exponent(self, freqs_hz: Any) -> np.ndarray:
        """``b(f) = b0 + b1 log2(max(f, band_lo) / 2 kHz)``."""
        f = np.maximum(np.asarray(freqs_hz, dtype=np.float64), self.band_lo_hz)
        return self.b0 + self.b1 * np.log2(f / FLOOR_EXP_REF_HZ)

    def rotor_psd(self, freqs_hz: Any, s: Any) -> np.ndarray:
        """``(R, F)`` each rotor's floor ``F(f) s_r^{b(f)}`` at relative speeds ``s (R,)``."""
        s = np.maximum(np.atleast_1d(np.asarray(s, dtype=np.float64)), SPEED_FLOOR_RPS)
        b = self.exponent(freqs_hz)
        return self.psd(freqs_hz)[None, :] * np.exp(np.log(s)[:, None] * b[None, :])

    def mean_speed_psd(self, freqs_hz: Any, s: Any) -> np.ndarray:
        """``(F,)`` the one-floor rule's ``F(f) sbar^{b(f)}``, ``sbar = mean_r s_r``
        at relative speeds ``s (R,)``."""
        s = np.atleast_1d(np.asarray(s, dtype=np.float64))
        sbar = max(float(s.mean()), SPEED_FLOOR_RPS)
        return self.psd(freqs_hz) * np.exp(math.log(sbar) * self.exponent(freqs_hz))


@dataclass(frozen=True)
class MicResponse:
    """``E_rm(f)`` (dB) of either kind: ``pair_eq`` (bench) or ``array_geometry`` (flight)."""

    kind: str
    n_rotors: int
    n_mics: int
    pair_freqs_hz: np.ndarray | None = None
    pair_db: np.ndarray | None = None  # (R, M, Fq)
    array_response: Mapping[str, Any] | None = None
    geometry_db: np.ndarray | None = None  # (R, M)

    @classmethod
    def from_mapping(cls, d: Mapping[str, Any], n_rotors: int, n_mics: int) -> MicResponse:
        kind = str(d["kind"])
        if kind == "pair_eq":
            pe = d["pair_eq"]
            fq = _arr(pe["freqs_hz"], 1, "mic_response.pair_eq.freqs_hz")
            db = _arr(pe["db"], 3, "mic_response.pair_eq.db")
            if db.shape != (n_rotors, n_mics, fq.size) or np.any(np.diff(fq) <= 0.0):
                raise ValueError(
                    f"pair_eq.db must be (R, M, Fq) = {(n_rotors, n_mics, fq.size)} on increasing "
                    f"freqs_hz, got {db.shape}"
                )
            return cls(kind, n_rotors, n_mics, pair_freqs_hz=fq, pair_db=db)
        if kind == "array_geometry":
            geo = _arr(d["geometry_db"], 2, "mic_response.geometry_db")
            if geo.shape != (n_rotors, n_mics):
                raise ValueError(
                    f"geometry_db must be (R, M) = {(n_rotors, n_mics)}, got {geo.shape}"
                )
            response = d["array_response"]
            if np.atleast_2d(np.asarray(response["gain_db"])).shape[0] != n_mics:
                raise ValueError(f"array_response must carry {n_mics} channels")
            return cls(kind, n_rotors, n_mics, array_response=response, geometry_db=geo)
        raise ValueError(f"mic_response.kind must be 'pair_eq' or 'array_geometry', got {kind!r}")

    def rotor_db(self, freqs_hz: Any) -> np.ndarray:
        """``(R, M, F)`` the ROTOR-SPECIFIC part of ``E_rm(f)``: the pair EQ
        interpolated linearly in ``f`` and held flat outside its grid (bench),
        or ``Delta G_rm`` at every frequency (flight)."""
        f = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
        R, M = self.n_rotors, self.n_mics
        if self.kind == "pair_eq":
            assert self.pair_freqs_hz is not None and self.pair_db is not None
            out = interp_matrix(f, self.pair_freqs_hz) @ self.pair_db.reshape(R * M, -1).T
            return out.T.reshape(R, M, f.size)
        assert self.geometry_db is not None
        return np.broadcast_to(self.geometry_db[:, :, None], (R, M, f.size)).copy()

    def array_db(self, freqs_hz: Any) -> np.ndarray:
        """``(M, F)`` the part shared by the rotors: the array response
        ``A_m(f)`` (flight; the data were divided by it), zero on the bench."""
        f = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
        if self.array_response is None:
            return np.zeros((self.n_mics, f.size))
        return array_response_db(dict(self.array_response), f)

    def db(self, freqs_hz: Any) -> np.ndarray:
        """``(R, M, F)`` the whole ``E_rm(f)``."""
        return self.rotor_db(freqs_hz) + self.array_db(freqs_hz)[None, :, :]


@dataclass(frozen=True)
class V4Params:
    """Everything the renderer reads out of a ``noise-v4-fit/3`` payload."""

    kind: str
    n_rotors: int
    n_mics: int
    blades: int
    pole_pairs: int
    f_ref_rev_s: float
    sr: int
    band_hz: tuple[float, float]
    profile_aero_db: np.ndarray  # (R, K)
    motor_orders: np.ndarray  # (n_m,) int
    profile_motor_db: np.ndarray  # (R, n_m)
    alpha_aero: float
    alpha_motor: float
    pedestal: Pedestal
    shaft: ShaftLaw
    floor: FloorLaw
    mic: MicResponse
    #: ``(M,)`` the flight static per-mic wind block (DREGON, v3's ``wind_db``)
    #: or ``None``. A bench payload's wind is per RECORDING and is not a rig term.
    wind_db: np.ndarray | None

    @classmethod
    def from_payload(cls, fit: Mapping[str, Any]) -> V4Params:
        if fit.get("schema") != FIT_SCHEMA_V4:
            raise ValueError(f"expected schema {FIT_SCHEMA_V4!r}, got {fit.get('schema')!r}")
        p = fit["params"]
        R, M = int(fit["n_rotors"]), int(fit["n_mics"])
        f_ref = float(fit.get("f_ref_rev_s", 80.0))
        band = tuple(float(x) for x in fit["band_hz"])
        aero = _arr(p["profile_aero_db"], 2, "profile_aero_db")
        orders_in, motor_in = p.get("motor_orders"), p.get("profile_motor_db")
        orders = np.asarray([] if orders_in is None else orders_in, dtype=np.int64).reshape(-1)
        motor = np.asarray(np.zeros((R, 0)) if motor_in is None else motor_in, dtype=np.float64)
        if motor.size != R * orders.size:
            raise ValueError(
                f"profile_motor_db must be (R, n_m) = {(R, orders.size)}, got {motor.shape}"
            )
        motor = motor.reshape(R, orders.size)
        ped = Pedestal.from_mapping(p["pedestal"])
        if aero.shape[0] != R or ped.sigma2_amp.size != aero.shape[1]:
            raise ValueError(
                f"profile_aero_db {aero.shape} and pedestal (K={ped.sigma2_amp.size}) "
                f"disagree with n_rotors={R}"
            )
        if np.any(orders < 1):
            raise ValueError(f"motor_orders must be positive orders, got {orders.tolist()}")
        wind_db = None
        wind = p.get("wind")
        if fit.get("kind") == "flight" and wind is not None:
            wind_db = _arr(wind["wind_db"], 1, "wind.wind_db")
            if wind_db.size != M:
                raise ValueError(f"wind.wind_db must carry {M} levels, got {wind_db.size}")
        return cls(
            kind=str(fit.get("kind")),
            n_rotors=R,
            n_mics=M,
            blades=int(fit["blades"]),
            pole_pairs=int(fit["pole_pairs"]),
            f_ref_rev_s=f_ref,
            sr=int(fit.get("sr", 16000)),
            band_hz=(band[0], band[1]),
            profile_aero_db=aero,
            motor_orders=orders,
            profile_motor_db=motor,
            alpha_aero=float(p["alpha_aero"]),
            alpha_motor=float(p["alpha_motor"]),
            pedestal=ped,
            shaft=ShaftLaw.from_mapping(p["shaft"], f_ref),
            floor=FloorLaw.from_mapping(p["floor"], band[0]),
            mic=MicResponse.from_mapping(p["mic_response"], R, M),
            wind_db=wind_db,
        )

    @property
    def k_max(self) -> int:
        """The profile width ``K``: the highest aerodynamic order of the payload."""
        return int(self.profile_aero_db.shape[1])

    def speed(self, rps_rev_s: Any) -> np.ndarray:
        """``s = max(f, SPEED_FLOOR_RPS) / f_ref``, same shape as ``rps_rev_s``."""
        return (
            np.maximum(np.asarray(rps_rev_s, dtype=np.float64), SPEED_FLOOR_RPS) / self.f_ref_rev_s
        )

    def line_power(self, s: Any) -> tuple[np.ndarray, np.ndarray]:
        """Mean line powers at relative speeds ``s``, ``(R,)`` or ``(R, T)``:
        aerodynamic ``(R, K[, T])`` and motor ``(R, n_m[, T])``, linear."""
        s = np.maximum(np.asarray(s, dtype=np.float64), SPEED_FLOOR_RPS / self.f_ref_rev_s)
        if s.shape[0] != self.n_rotors or s.ndim > 2:
            raise ValueError(
                f"speeds must be (R,) or (R, T) with R = {self.n_rotors}, got {s.shape}"
            )
        ls = np.log(s)[:, None, ...]
        ext = (slice(None), slice(None)) + (None,) * (s.ndim - 1)
        aero = 10.0 ** (self.profile_aero_db[ext] / 10.0) * np.exp(self.alpha_aero * ls)
        motor = 10.0 ** (self.profile_motor_db[ext] / 10.0) * np.exp(self.alpha_motor * ls)
        return aero, motor


def line_lag_law(
    tau_s: Any,
    k: Any,
    *,
    sigma_nu: float,
    lam: float,
    sigma2_amp: Any = 0.0,
    gamma_amp_hz: Any = 0.0,
    sigma2_phase: Any = 0.0,
    gamma_phase_hz: Any = 0.0,
) -> np.ndarray:
    """eq-lag: ``exp(-k^2 V_theta / 2) exp(s2A (e^{-2 pi gA |tau|} - 1)) exp(s2P (e^{-2 pi gP |tau|} - 1))``.

    Broadcast by the ordinary rules with the lag the LAST axis (``k`` and the
    pedestal arrays carry an explicit trailing axis). A motor line is the
    default: the shaft factor alone.
    """
    tau = np.abs(np.asarray(tau_s, dtype=np.float64))
    k = np.asarray(k, dtype=np.float64)
    expo = -0.5 * k**2 * integrated_ou_increment_var(tau, lam=lam, sigma=sigma_nu)
    expo = expo + np.asarray(sigma2_amp) * np.expm1(-TWO_PI * np.asarray(gamma_amp_hz) * tau)
    expo = expo + np.asarray(sigma2_phase) * np.expm1(-TWO_PI * np.asarray(gamma_phase_hz) * tau)
    return np.exp(expo)


def hann_window(n_fft: int) -> np.ndarray:
    """The PERIODIC Hann window of the v4 front ends (``np.hanning(n + 1)[:n]``)."""
    return np.hanning(int(n_fft) + 1)[: int(n_fft)]


def expected_periodogram(
    fit: Mapping[str, Any],
    rps_rev_s: Any,
    *,
    n_fft: int = 2048,
    sr: int | None = None,
    floor_oversample: int = 16,
) -> np.ndarray:
    """``(M, n_fft // 2 + 1)`` expected periodic-Hann periodogram of a render of
    ``fit`` at CONSTANT rotor speeds ``rps_rev_s (R,)``, per microphone.

    The periodogram is ``|sum_t w_t x_t e^{-2 pi i f t}|^2 / sum_t w_t^2``
    (white noise of variance ``v`` reads ``v``). For every process ``c_m(tau)``
    is its autocovariance, and the expectation is the DTFT of
    ``(w * w)(tau) c_m(tau) / sum w^2`` at the bins, exactly (no line-shape
    approximation):

    * each aerodynamic line ``(r, k)`` with ``k f_r`` under the Nyquist:
      ``10^{E_rm(k f_r)/10} P^a_rk(s_r) R_rk(tau) cos(2 pi k f_r tau)`` with
      the full eq-lag; each motor line the same with the shaft factor only;
    * the floor ``F(f) [sum_r 10^{E_rm(f)/10} s_r^{b(f)} + 10^{A_m(f)/10} s]``
      (rule ``per_rotor_sum``) or ``F(f) 10^{A_m(f)/10} [sbar^{b(f)} + s]``,
      ``sbar = mean_r s_r`` (rule ``one_floor_mean_speed``), ``s`` the static
      share, and, in flight, the static per-mic wind
      ``10^{w_m/10} wind_shape(f) 10^{A_m(f)/10}``, transformed from a PSD
      grid ``floor_oversample`` times finer than the window.

    The render chain's anti-alias/decimator transfer is NOT applied: it is one
    within 0.01 dB below 7.6 kHz (the fit's forward model in
    :mod:`experiments.noise_model.spectrum` carries it over the whole band).
    """
    par = V4Params.from_payload(fit)
    sr = par.sr if sr is None else int(sr)
    n = int(n_fft)
    rps = np.atleast_1d(np.asarray(rps_rev_s, dtype=np.float64))
    if rps.shape != (par.n_rotors,):
        raise ValueError(f"rps_rev_s must be (R,) = ({par.n_rotors},), got {rps.shape}")
    f_r = np.maximum(rps, SPEED_FLOOR_RPS)
    s = par.speed(f_r)
    tau = np.arange(n) / float(sr)
    w = hann_window(n)
    ww = np.correlate(w, w, mode="full")[n - 1 :]  # (w * w)(tau), tau = 0 .. n - 1
    p_aero, p_motor = par.line_power(s)
    ped = par.pedestal
    c = np.zeros((par.n_mics, n))
    for r in range(par.n_rotors):
        sig, lam = float(par.shaft.sigma_nu(s[r])), float(par.shaft.lam(s[r]))
        k = np.arange(1, par.k_max + 1, dtype=np.float64)
        keep = k * f_r[r] < 0.5 * sr
        k = k[keep]
        lag = line_lag_law(
            tau[None, :],
            k[:, None],
            sigma_nu=sig,
            lam=lam,
            sigma2_amp=ped.sigma2_amp[keep][:, None],
            gamma_amp_hz=ped.gamma_amp_hz[keep][:, None],
            sigma2_phase=ped.sigma2_phase[keep][:, None],
            gamma_phase_hz=ped.gamma_phase_hz[keep][:, None],
        )
        gain = 10.0 ** (par.mic.db(k * f_r[r])[r] / 10.0)  # (M, K')
        c += (gain * p_aero[r, keep][None, :]) @ (lag * np.cos(TWO_PI * f_r[r] * k[:, None] * tau))
        km = par.motor_orders.astype(np.float64)
        mk = km * f_r[r] < 0.5 * sr
        if np.any(mk):
            km = km[mk]
            lag_m = line_lag_law(tau[None, :], km[:, None], sigma_nu=sig, lam=lam)
            gain_m = 10.0 ** (par.mic.db(km * f_r[r])[r] / 10.0)
            c += (gain_m * p_motor[r, mk][None, :]) @ (
                lag_m * np.cos(TWO_PI * f_r[r] * km[:, None] * tau)
            )
    n_psd = int(floor_oversample) * n
    f_psd = np.fft.rfftfreq(n_psd, d=1.0 / float(sr))
    array_lin = 10.0 ** (par.mic.array_db(f_psd) / 10.0)  # (M, F)
    if par.floor.rule == FLOOR_RULE_ONE:
        psd = par.floor.mean_speed_psd(f_psd, s)[None, :] * array_lin
    else:
        rotor = par.floor.rotor_psd(f_psd, s)  # (R, F)
        psd = np.einsum("rmf,rf->mf", 10.0 ** (par.mic.db(f_psd) / 10.0), rotor)
    psd += par.floor.psd(f_psd)[None, :] * par.floor.static_rel * array_lin
    if par.wind_db is not None:
        psd += 10.0 ** (par.wind_db[:, None] / 10.0) * wind_shape(f_psd)[None, :] * array_lin
    c += np.fft.irfft(psd, n=n_psd, axis=1)[:, :n]
    q = c * (ww / float(np.sum(w**2)))[None, :]
    # the even sequence q(|tau|), |tau| < n, read at f_j = j sr / n off one
    # length-2n rFFT at its even bins
    ext = np.concatenate([q, np.zeros((q.shape[0], 1)), q[:, :0:-1]], axis=1)
    return np.fft.rfft(ext, axis=1).real[:, ::2]
