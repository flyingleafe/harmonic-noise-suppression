"""The stochastic family's spectrum as a differentiable PyTorch model.

Everything the renderer randomizes is a parameter here, in the renderer's own
coordinates (decibels, Hz, rev/s), so a fitted set can be read against
:class:`data_processing.stochastic_rotor_noise.StochasticRanges` directly:

* ``profile_db (R, K)`` — static line levels (the renderer's ``harm_mean_db +
  profile_db``), free per line;
* ``h (R, K, Tk)`` — the log-power drift on a knot grid, parametrized by its
  whitened GP coordinates ``z`` so the prior is ``|z|^2 / 2``;
* ``gamma0_r, slope_r`` — the width law ``gamma_rk = gamma0 + slope k`` (or
  a free width per line, ``free_gamma``, to test the law itself);
* the floor: level, a 14-knot shape in log frequency, tilt, slow level and
  tilt drifts (whitened GP knots), an optional per-microphone offset
  (``mic_floor``; the renderer's floor is common to all microphones);
* ``mic_gain_db (M, R)`` line gains per microphone, mean over mics pinned at 0;
* the speed law exponents (fixed 2.5 unless ``fit_speed_law``), the static
  floor share, and an optional smooth per-rotor carrier offset in rev/s
  (``rps_offset``) with a tight prior for imperfect references.

Lines are bin-integrated (the Lorentzian's arctan antiderivative, the
renderer's ``line_bin_integrate`` route, or a Gaussian of equal HWHM under
``line_shape="gauss"``) and the whole spectrum is convolved with the periodic
Hann window's power response ``[1/6, 2/3, 1/6]`` — the analysis window's
smearing of a stationary spectrum on the bin grid — so the model predicts the
periodogram's expectation, not the PSD.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

FLOOR_SHAPE_F_MIN = 30.0
FLOOR_TILT_REF_HZ = 500.0
FLOOR_SHAPE_N_CTRL = 14
AMP_RPS_REF = 80.0
HANN_POWER_KERNEL = (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)
GAMMA_FLOOR_HZ = 0.05


@dataclass
class Spec:
    """Everything that shapes the model but is not fitted."""

    freqs: np.ndarray  # (F,) Hz
    times: np.ndarray  # (N,) s
    rps: np.ndarray  # (R, N) rev/s at the frames
    n_mics: int
    n_harm: int
    f_min: float = FLOOR_SHAPE_F_MIN
    f_max: float | None = None  # fit band ceiling (None: Nyquist)
    knot_dt_s: float = 0.1  # GP knot spacing of the line drift and floor drift
    gp_std_db: float = 3.0
    gp_tau_s: float = 1.5
    floor_gp_std_db: float = 2.0
    floor_gp_tau_s: float = 3.0
    floor_tilt_gp_std: float = 0.5
    floor_tilt_gp_tau_s: float = 6.0
    floor_shape_std_db: float = 5.0
    floor_shape_oct: float = 1.5
    line_shape: str = "lorentz"  # lorentz | gauss
    free_gamma: bool = False
    mic_floor: bool = False
    fit_speed_law: bool = False
    rps_offset: bool = False
    rps_offset_std: float = 0.3  # rev/s prior std of the carrier correction
    rps_offset_dt_s: float = 0.5
    amp_rps_exponent: float = 2.5
    window_kernel: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


def se_cholesky(n: int, dt: float, tau: float, jitter: float = 1e-6) -> np.ndarray:
    t = np.arange(n) * dt
    k = np.exp(-0.5 * ((t[:, None] - t[None, :]) / max(tau, 1e-9)) ** 2)
    return np.linalg.cholesky(k + jitter * np.eye(n))


def interp_matrix(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Linear interpolation ``x <- knots`` as a dense ``(len(x), len(knots))``
    matrix; outside the knot range it clamps to the end knots."""
    x = np.asarray(x, dtype=np.float64)
    a = np.zeros((x.size, knots.size))
    idx = np.clip(np.searchsorted(knots, x, side="right") - 1, 0, knots.size - 2)
    w = (x - knots[idx]) / (knots[idx + 1] - knots[idx])
    w = np.clip(w, 0.0, 1.0)
    a[np.arange(x.size), idx] = 1.0 - w
    a[np.arange(x.size), idx + 1] = w
    return a


def _knots(t_end: float, dt: float) -> np.ndarray:
    n = max(int(math.ceil(t_end / dt)) + 1, 2)
    return np.linspace(0.0, t_end, n)


class CombSpectrum(nn.Module):
    """``M[m, t, f]`` — the expected periodogram of one clip."""

    def __init__(
        self, spec: Spec, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        # buffers (declared for the type checker; registered below)
        self.freqs: Tensor
        self.rps: Tensor
        self.k: Tensor
        self.band: Tensor
        self.shape_interp: Tensor
        self.shape_chol: Tensor
        self.tilt_oct: Tensor
        self.t_interp: Tensor
        self.h_chol: Tensor
        self.b_chol: Tensor
        self.tilt_chol: Tensor
        self.o_interp: Tensor
        self.kernel: Tensor
        self.spec = spec
        R, N = spec.rps.shape
        F = spec.freqs.size
        K, M = spec.n_harm, spec.n_mics
        self.R, self.N, self.F, self.K, self.M = R, N, F, K, M
        dev, dt = torch.device(device), dtype
        self._dev, self._dtype = dev, dt
        freqs = np.asarray(spec.freqs, dtype=np.float64)
        self.df = float(freqs[1] - freqs[0])
        self.register_buffer("freqs", torch.as_tensor(freqs, dtype=dt, device=dev))
        self.register_buffer("rps", torch.as_tensor(spec.rps, dtype=dt, device=dev))
        self.register_buffer("k", torch.arange(1, K + 1, dtype=dt, device=dev))
        f_max = spec.f_max or float(freqs[-1])
        band = (freqs >= spec.f_min) & (freqs <= f_max)
        self.register_buffer("band", torch.as_tensor(band, device=dev))
        t_rel = np.asarray(spec.times) - float(spec.times[0])
        t_end = float(t_rel[-1])

        # Floor shape: knots in octaves above f_min, linear interpolation.
        ctrl_hz = np.geomspace(spec.f_min, float(freqs[-1]), FLOOR_SHAPE_N_CTRL)
        ctrl_oct = np.log2(ctrl_hz / ctrl_hz[0])
        f_oct = np.log2(np.maximum(freqs, spec.f_min) / ctrl_hz[0])
        self.register_buffer(
            "shape_interp", torch.as_tensor(interp_matrix(f_oct, ctrl_oct), dtype=dt, device=dev)
        )
        self.register_buffer(
            "shape_chol",
            torch.as_tensor(
                se_cholesky(
                    FLOOR_SHAPE_N_CTRL, float(ctrl_oct[1] - ctrl_oct[0]), spec.floor_shape_oct
                ),
                dtype=dt,
                device=dev,
            ),
        )
        self.ctrl_hz = ctrl_hz
        self.register_buffer(
            "tilt_oct",
            torch.as_tensor(
                np.log2(np.maximum(freqs, spec.f_min) / FLOOR_TILT_REF_HZ), dtype=dt, device=dev
            ),
        )

        # Time knots for the drifts (lines and floor share the grid).
        knots = _knots(t_end, spec.knot_dt_s)
        self.knots = knots
        self.register_buffer(
            "t_interp", torch.as_tensor(interp_matrix(t_rel, knots), dtype=dt, device=dev)
        )
        kdt = float(knots[1] - knots[0])
        self.register_buffer(
            "h_chol",
            torch.as_tensor(se_cholesky(knots.size, kdt, spec.gp_tau_s), dtype=dt, device=dev),
        )
        self.register_buffer(
            "b_chol",
            torch.as_tensor(
                se_cholesky(knots.size, kdt, spec.floor_gp_tau_s), dtype=dt, device=dev
            ),
        )
        self.register_buffer(
            "tilt_chol",
            torch.as_tensor(
                se_cholesky(knots.size, kdt, spec.floor_tilt_gp_tau_s), dtype=dt, device=dev
            ),
        )
        Tk = knots.size

        # Carrier offsets: a coarse random-walk-ish grid, tight prior.
        oknots = _knots(t_end, spec.rps_offset_dt_s)
        self.register_buffer(
            "o_interp", torch.as_tensor(interp_matrix(t_rel, oknots), dtype=dt, device=dev)
        )

        z = lambda *s: nn.Parameter(torch.zeros(*s, dtype=dt, device=dev))  # noqa: E731
        self.floor_mean_db = z(1)
        self.floor_shape_z = z(FLOOR_SHAPE_N_CTRL)
        self.floor_tilt_db_oct = z(1)
        self.floor_level_z = z(Tk)
        self.floor_tilt_z = z(Tk)
        self.mic_floor_db = z(M)
        self.profile_db = z(R, K)
        self.h_z = z(R, K, Tk)
        self.gamma0_raw = nn.Parameter(
            torch.full((R,), 2.0, dtype=dt, device=dev)
        )  # softplus -> Hz
        self.slope_raw = nn.Parameter(torch.full((R,), 0.3, dtype=dt, device=dev))
        self.log_gamma_free = z(R, K)  # log Hz, used when free_gamma
        self.mic_gain_db = z(M, R)
        self.amp_exp = nn.Parameter(torch.full((1,), spec.amp_rps_exponent, dtype=dt, device=dev))
        self.floor_exp = nn.Parameter(torch.full((1,), spec.amp_rps_exponent, dtype=dt, device=dev))
        self.floor_static_raw = nn.Parameter(
            torch.full((1,), -6.0, dtype=dt, device=dev)
        )  # softplus -> share
        self.rps_offset_knots = z(R, oknots.size)
        self.active_k = K  # harmonic ladder: lines with k > active_k are off

        kernel = torch.tensor(HANN_POWER_KERNEL, dtype=dt, device=dev).view(1, 1, 3)
        self.register_buffer("kernel", kernel)

    # ── pieces ────────────────────────────────────────────────────────────

    @property
    def gamma(self) -> Tensor:
        """``(R, K)`` half widths in Hz."""
        if self.spec.free_gamma:
            return torch.exp(self.log_gamma_free).clamp_min(GAMMA_FLOOR_HZ)
        g0 = torch.nn.functional.softplus(self.gamma0_raw)
        sl = torch.nn.functional.softplus(self.slope_raw)
        return (g0[:, None] + sl[:, None] * self.k[None, :]).clamp_min(GAMMA_FLOOR_HZ)

    @property
    def h_db(self) -> Tensor:
        """``(R, K, Tk)`` line drift in dB at the knots."""
        return self.spec.gp_std_db * torch.einsum("ij,rkj->rki", self.h_chol, self.h_z)

    def carrier(self) -> Tensor:
        """``(R, N)`` rev/s actually used for the lines."""
        r = self.rps
        if self.spec.rps_offset:
            r = r + torch.einsum("nj,rj->rn", self.o_interp, self.rps_offset_knots)
        return r.clamp_min(0.0)

    def floor(self) -> Tensor:
        """``(M, N, F)`` broadband floor (per mic only through ``mic_floor``)."""
        s = self.spec
        shape = self.shape_interp @ (
            s.floor_shape_std_db * (self.shape_chol @ self.floor_shape_z)
        )  # (F,)
        level_t = self.t_interp @ (s.floor_gp_std_db * (self.b_chol @ self.floor_level_z))  # (N,)
        tilt_t = self.t_interp @ (
            s.floor_tilt_gp_std * (self.tilt_chol @ self.floor_tilt_z)
        )  # (N,)
        db = (
            self.floor_mean_db
            + shape[None, :]
            + level_t[:, None]
            + (self.floor_tilt_db_oct + tilt_t)[:, None] * self.tilt_oct[None, :]
        )
        speed = self.rps.clamp_min(0.0) / AMP_RPS_REF
        fexp = (
            self.floor_exp
            if s.fit_speed_law
            else torch.full_like(self.floor_exp, s.amp_rps_exponent)
        )
        static = (
            torch.nn.functional.softplus(self.floor_static_raw)
            if s.fit_speed_law
            else torch.zeros_like(self.floor_static_raw)
        )
        gain = (speed**fexp).mean(dim=0) + static  # (N,)
        floor = 10.0 ** (db / 10.0) * gain[:, None]  # (N, F)
        if s.mic_floor:
            return floor[None] * 10.0 ** (self.mic_floor_db[:, None, None] / 10.0)
        return floor[None].expand(self.M, -1, -1)

    def line_power(self) -> Tensor:
        """``(R, K, N)`` line powers (unit-area weights) incl. the speed law."""
        s = self.spec
        h = torch.einsum("nj,rkj->rkn", self.t_interp, self.h_db)
        db = self.profile_db[:, :, None] + h
        speed = self.rps.clamp_min(0.0) / AMP_RPS_REF
        aexp = (
            self.amp_exp if s.fit_speed_law else torch.full_like(self.amp_exp, s.amp_rps_exponent)
        )
        power = 10.0 ** (db / 10.0) * (speed**aexp)[:, None, :]
        if self.active_k < self.K:
            power = power * (self.k <= self.active_k).to(power.dtype)[None, :, None]
        return power

    def _line_density(self, d: Tensor, gamma: Tensor) -> Tensor:
        """Bin-integrated unit-area line density at signed offsets ``d`` (Hz)."""
        half = 0.5 * self.df
        if self.spec.line_shape == "gauss":
            sigma = gamma / math.sqrt(2.0 * math.log(2.0))
            s2 = sigma * math.sqrt(2.0)
            return 0.5 * (torch.erf((d + half) / s2) - torch.erf((d - half) / s2)) / self.df
        return (torch.atan((d + half) / gamma) - torch.atan((d - half) / gamma)) / (
            math.pi * self.df
        )

    def lines(self, k_chunk: int = 32) -> Tensor:
        """``(R, N, F)`` the rotors' line spectra (before microphone gains)."""
        power = self.line_power()  # (R, K, N)
        gamma = self.gamma  # (R, K)
        carrier = self.carrier()  # (R, N)
        out = torch.zeros(self.R, self.N, self.F, dtype=self._dtype, device=self._dev)
        k_top = min(self.K, self.active_k)
        for k0 in range(0, k_top, k_chunk):
            k1 = min(k0 + k_chunk, k_top)
            centres = self.k[k0:k1][None, :, None] * carrier[:, None, :]  # (R, k, N)
            d = self.freqs[None, None, None, :] - centres[..., None]  # (R, k, N, F)
            dens = self._line_density(d, gamma[:, k0:k1, None, None])
            out = out + torch.einsum("rkn,rknf->rnf", power[:, k0:k1], dens)
        return out

    def forward(self) -> Tensor:
        """``(M, N, F)`` expected periodogram."""
        gains = 10.0 ** (
            (self.mic_gain_db - self.mic_gain_db.mean(dim=0, keepdim=True)) / 10.0
        )  # (M, R)
        spectrum = self.floor() + torch.einsum("mr,rnf->mnf", gains, self.lines())
        if self.spec.window_kernel:
            x = spectrum.reshape(-1, 1, self.F)
            x = torch.nn.functional.conv1d(
                torch.nn.functional.pad(x, (1, 1), mode="replicate"), self.kernel
            )
            spectrum = x.reshape(self.M, self.N, self.F)
        return spectrum

    # ── objective ─────────────────────────────────────────────────────────

    def prior(self) -> Tensor:
        """``-log p(latents)`` up to constants: whitened GP knots + carrier prior."""
        s = self.spec
        p = 0.5 * (
            self.h_z.square().sum()
            + self.floor_level_z.square().sum()
            + self.floor_tilt_z.square().sum()
            + self.floor_shape_z.square().sum()
        )
        if s.rps_offset:
            p = p + 0.5 * self.rps_offset_knots.square().sum() / s.rps_offset_std**2
        if s.free_gamma:
            # smoothness of log gamma along k (second differences), weak
            d2 = (
                self.log_gamma_free[:, 2:]
                - 2 * self.log_gamma_free[:, 1:-1]
                + self.log_gamma_free[:, :-2]
            )
            p = p + 0.5 * d2.square().sum() / 0.3**2
        return p

    def whittle(self, power: Tensor, model: Tensor | None = None) -> Tensor:
        """Summed Whittle NLL over the fit band: ``sum I/M + log M``."""
        m = self.forward() if model is None else model
        m = m.clamp_min(1e-12)
        cell = power / m + torch.log(m)
        return cell[..., self.band].sum()

    def n_cells(self) -> int:
        return int(self.M * self.N * int(self.band.sum().item()))

    # ── export ────────────────────────────────────────────────────────────

    def export(self) -> dict[str, Any]:
        with torch.no_grad():
            g = torch.nn.functional.softplus
            out = dict(
                profile_db=self.profile_db.cpu().numpy().copy(),
                h_db=self.h_db.cpu().numpy().copy(),
                knots_s=self.knots.copy(),
                gamma=self.gamma.cpu().numpy().copy(),
                gamma0=g(self.gamma0_raw).cpu().numpy().copy(),
                gamma_slope=g(self.slope_raw).cpu().numpy().copy(),
                floor_mean_db=float(self.floor_mean_db.item()),
                floor_shape_db=(
                    self.spec.floor_shape_std_db * (self.shape_chol @ self.floor_shape_z)
                )
                .cpu()
                .numpy()
                .copy(),
                floor_ctrl_hz=self.ctrl_hz.copy(),
                floor_tilt_db_oct=float(self.floor_tilt_db_oct.item()),
                floor_level_db=(self.spec.floor_gp_std_db * (self.b_chol @ self.floor_level_z))
                .cpu()
                .numpy()
                .copy(),
                floor_tilt_gp=(self.spec.floor_tilt_gp_std * (self.tilt_chol @ self.floor_tilt_z))
                .cpu()
                .numpy()
                .copy(),
                mic_floor_db=self.mic_floor_db.cpu().numpy().copy(),
                mic_gain_db=(self.mic_gain_db - self.mic_gain_db.mean(dim=0, keepdim=True))
                .cpu()
                .numpy()
                .copy(),
                amp_exp=float(self.amp_exp.item()),
                floor_exp=float(self.floor_exp.item()),
                floor_static_rel=float(g(self.floor_static_raw).item()),
                rps_offset=(torch.einsum("nj,rj->rn", self.o_interp, self.rps_offset_knots))
                .cpu()
                .numpy()
                .copy(),
                carrier=self.carrier().cpu().numpy().copy(),
            )
        return out

    def parameter_groups(self, stage: str) -> list[nn.Parameter]:
        """Which parameters a stage moves."""
        s = self.spec
        floor = [
            self.floor_mean_db,
            self.floor_shape_z,
            self.floor_tilt_db_oct,
            self.floor_level_z,
            self.floor_tilt_z,
        ]
        if s.mic_floor:
            floor.append(self.mic_floor_db)
        if s.fit_speed_law:
            floor += [self.floor_exp, self.floor_static_raw]
        if stage == "floor":
            return floor
        lines = [self.profile_db, self.h_z, self.mic_gain_db]
        lines += [self.log_gamma_free] if s.free_gamma else [self.gamma0_raw, self.slope_raw]
        if s.fit_speed_law:
            lines.append(self.amp_exp)
        if s.rps_offset:
            lines.append(self.rps_offset_knots)
        return floor + lines
