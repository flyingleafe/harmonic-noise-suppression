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

Lines are sampled at the bin centres with the renderer's 0.6-bin width floor
(its default route; ``line_bin_integrate`` switches to the exact arctan bin
integral, ``line_shape="gauss"`` to a Gaussian of equal HWHM) and the whole
spectrum is convolved with the periodic Hann window's power response
``[1/6, 2/3, 1/6]`` — the analysis window's smearing of a stationary spectrum
on the bin grid — so the model predicts the periodogram's expectation, not
the PSD.
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
#: The renderer floors every half width at 0.6 bins (``GAMMA_MIN_BINS``) so a
#: sub-bin line keeps its power; the family's *realized* spectra carry that
#: floor, and so does the model unless ``gamma_min_bins`` is set to 0.
GAMMA_MIN_BINS = 0.6
#: The renderer renders each line over +-5 half widths and divides by the
#: fraction of a Lorentzian inside that support (``stochastic_rotor_noise``).
LORENTZ_SUPPORT_HWHM = 5.0
LORENTZ_TRUNC_NORM = float(2.0 / math.pi * math.atan(LORENTZ_SUPPORT_HWHM))


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
    #: width law exponent: ``gamma = gamma0 + slope * k**width_power``. 1 is the
    #: quasi-static shaft regime (a frozen speed offset smears every line by
    #: ``k * sigma``), 2 the diffusive one (an OU shaft with correlation time
    #: short against the analysis window gives a Lorentzian of HWHM
    #: ``2 pi tau k^2 sigma^2``). ``fit_width_power`` makes the exponent a
    #: fitted scalar, so the regime is read off the likelihood instead of
    #: assumed; the OU shaft interpolates between the two as tau crosses the
    #: window length, so a non-integer value is meaningful.
    width_power: float = 1.0
    fit_width_power: bool = False
    width_power_log_std: float = 0.5
    mic_floor: bool = False
    fit_speed_law: bool = False
    rps_offset: bool = False
    rps_offset_std: float = 0.3  # rev/s prior std of the carrier correction
    rps_offset_dt_s: float = 0.5
    amp_rps_exponent: float = 2.5
    window_kernel: bool = True
    gamma_min_bins: float = GAMMA_MIN_BINS
    #: The renderer samples each line's density at the bin centres
    #: (``line_bin_integrate=False`` in every training stream); ``True`` uses
    #: the exact bin integral instead (the renderer's optional route).
    line_bin_integrate: bool = False
    #: drift kernel of the line levels and the per-mic modulation: ``se`` or ``ou``
    gp_kernel: str = "se"
    #: per-mic multiplicative modulation of the low band (DREGON flow noise):
    #: ``umod_std_db`` > 0 switches it on; ``umod_lines`` lets it act on the
    #: lines at that microphone as well as on the floor
    umod_std_db: float = 0.0
    umod_tau_s: float = 0.25
    umod_corner_hz: float = 500.0
    umod_lines: bool = True
    #: one per-mic gain on everything (Michael's rig) — fitted only in a rig fit
    gain_all: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def se_cholesky(n: int, dt: float, tau: float, jitter: float = 1e-6) -> np.ndarray:
    t = np.arange(n) * dt
    k = np.exp(-0.5 * ((t[:, None] - t[None, :]) / max(tau, 1e-9)) ** 2)
    return np.linalg.cholesky(k + jitter * np.eye(n))


def ou_cholesky(n: int, dt: float, tau: float, jitter: float = 1e-6) -> np.ndarray:
    """Matérn-1/2 (Ornstein-Uhlenbeck) kernel: ``exp(-|dt|/tau)``."""
    t = np.arange(n) * dt
    k = np.exp(-np.abs(t[:, None] - t[None, :]) / max(tau, 1e-9))
    return np.linalg.cholesky(k + jitter * np.eye(n))


def drift_cholesky(kind: str, n: int, dt: float, tau: float) -> np.ndarray:
    return ou_cholesky(n, dt, tau) if kind == "ou" else se_cholesky(n, dt, tau)


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
        self.u_chol: Tensor
        self.u_band: Tensor
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
            torch.as_tensor(
                drift_cholesky(spec.gp_kernel, knots.size, kdt, spec.gp_tau_s), dtype=dt, device=dev
            ),
        )
        self.register_buffer(
            "u_chol",
            torch.as_tensor(
                drift_cholesky(spec.gp_kernel, knots.size, kdt, spec.umod_tau_s),
                dtype=dt,
                device=dev,
            ),
        )
        # soft low-band selector of the per-mic modulation (one octave transition)
        self.register_buffer(
            "u_band",
            torch.sigmoid(
                -torch.as_tensor(
                    np.log2(np.maximum(freqs, 1.0) / spec.umod_corner_hz), dtype=dt, device=dev
                )
                * 4.0
            ),
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
        self.log_width_power = z(1)  # log ratio to spec.width_power
        self.mic_gain_db = z(M, R)
        self.amp_exp = nn.Parameter(torch.full((1,), spec.amp_rps_exponent, dtype=dt, device=dev))
        self.floor_exp = nn.Parameter(torch.full((1,), spec.amp_rps_exponent, dtype=dt, device=dev))
        self.floor_static_raw = nn.Parameter(
            torch.full((1,), -6.0, dtype=dt, device=dev)
        )  # softplus -> share
        self.rps_offset_knots = z(R, oknots.size)
        self.u_z = z(M, Tk)  # per-mic low-band modulation knots (whitened)
        self.active_k = K  # harmonic ladder: lines with k > active_k are off

        kernel = torch.tensor(HANN_POWER_KERNEL, dtype=dt, device=dev).view(1, 1, 3)
        self.register_buffer("kernel", kernel)

    # ── parameter hooks (a clip inside a rig fit substitutes composites) ──

    def _profile_db(self) -> Tensor:
        return self.profile_db  # (R, K)

    def _floor_shape_z(self) -> Tensor:
        return self.floor_shape_z  # (n_ctrl,)

    def _floor_tilt_db_oct(self) -> Tensor:
        return self.floor_tilt_db_oct  # (1,)

    def _mic_gain_db(self) -> Tensor:
        return self.mic_gain_db  # (M, R)

    def _mic_floor_db(self) -> Tensor:
        return self.mic_floor_db  # (M,)

    def _gain_all_db(self) -> Tensor | None:
        return None  # (M,) dB on floor and lines alike

    def _gamma_raw(self) -> tuple[Tensor, Tensor]:
        return self.gamma0_raw, self.slope_raw  # (R,), (R,)

    def _width_power(self) -> Tensor:
        p = torch.as_tensor(
            self.spec.width_power,
            dtype=self.log_width_power.dtype,
            device=self.log_width_power.device,
        )
        return p * torch.exp(self.log_width_power[0]) if self.spec.fit_width_power else p

    def _amp_exp(self) -> Tensor:
        return self.amp_exp

    def _floor_exp(self) -> tuple[Tensor, Tensor]:
        return self.floor_exp, self.floor_static_raw

    def _umod_db(self) -> Tensor | None:
        """``(M, N)`` per-mic low-band modulation in dB, or None."""
        if self.spec.umod_std_db <= 0:
            return None
        return self.spec.umod_std_db * (self.t_interp @ (self.u_chol @ self.u_z.T)).T

    # ── pieces ────────────────────────────────────────────────────────────

    @property
    def gamma(self) -> Tensor:
        """``(R, K)`` half widths in Hz."""
        floor = max(GAMMA_FLOOR_HZ, self.spec.gamma_min_bins * self.df)
        if self.spec.free_gamma:
            return torch.exp(self.log_gamma_free).clamp_min(floor)
        g0_raw, sl_raw = self._gamma_raw()
        g0 = torch.nn.functional.softplus(g0_raw)
        sl = torch.nn.functional.softplus(sl_raw)
        return (g0[:, None] + sl[:, None] * self.k[None, :] ** self._width_power()).clamp_min(floor)

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
            s.floor_shape_std_db * (self.shape_chol @ self._floor_shape_z())
        )  # (F,)
        level_t = self.t_interp @ (s.floor_gp_std_db * (self.b_chol @ self.floor_level_z))  # (N,)
        tilt_t = self.t_interp @ (
            s.floor_tilt_gp_std * (self.tilt_chol @ self.floor_tilt_z)
        )  # (N,)
        db = (
            self.floor_mean_db
            + shape[None, :]
            + level_t[:, None]
            + (self._floor_tilt_db_oct() + tilt_t)[:, None] * self.tilt_oct[None, :]
        )
        speed = self.rps.clamp_min(0.0) / AMP_RPS_REF
        floor_exp, static_raw = self._floor_exp()
        fexp = floor_exp if s.fit_speed_law else torch.full_like(floor_exp, s.amp_rps_exponent)
        static = (
            torch.nn.functional.softplus(static_raw)
            if s.fit_speed_law
            else torch.zeros_like(static_raw)
        )
        gain = (speed**fexp).mean(dim=0) + static  # (N,)
        floor = 10.0 ** (db / 10.0) * gain[:, None]  # (N, F)
        if s.mic_floor:
            return floor[None] * 10.0 ** (self._mic_floor_db()[:, None, None] / 10.0)
        return floor[None].expand(self.M, -1, -1)

    def line_power(self) -> Tensor:
        """``(R, K, N)`` line powers (unit-area weights) incl. the speed law."""
        s = self.spec
        h = torch.einsum("nj,rkj->rkn", self.t_interp, self.h_db)
        db = self._profile_db()[:, :, None] + h
        speed = self.rps.clamp_min(0.0) / AMP_RPS_REF
        amp_exp = self._amp_exp()
        aexp = amp_exp if s.fit_speed_law else torch.full_like(amp_exp, s.amp_rps_exponent)
        power = 10.0 ** (db / 10.0) * (speed**aexp)[:, None, :]
        if self.active_k < self.K:
            power = power * (self.k <= self.active_k).to(power.dtype)[None, :, None]
        return power

    def _line_density(self, d: Tensor, gamma: Tensor) -> Tensor:
        """Unit-area line density at signed offsets ``d`` (Hz): the renderer's
        point sample at the bin centre, or the exact bin integral.

        ``line_shape``: ``lorentz`` (full skirts), ``lorentz_trunc`` (the
        renderer's realized line: support cut at ``LORENTZ_SUPPORT_HWHM`` half
        widths and the kept 87.4 % renormalized to unit area — the family's
        clips carry no skirts beyond 5 gamma) or ``gauss`` (equal HWHM).
        """
        if self.spec.line_shape == "gauss":
            sigma = gamma / math.sqrt(2.0 * math.log(2.0))
            if not self.spec.line_bin_integrate:
                return torch.exp(-0.5 * (d / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))
            half = 0.5 * self.df
            s2 = sigma * math.sqrt(2.0)
            return 0.5 * (torch.erf((d + half) / s2) - torch.erf((d - half) / s2)) / self.df
        if not self.spec.line_bin_integrate:
            dens = gamma / (math.pi * (d * d + gamma * gamma))
        else:
            half = 0.5 * self.df
            dens = (torch.atan((d + half) / gamma) - torch.atan((d - half) / gamma)) / (
                math.pi * self.df
            )
        if self.spec.line_shape == "lorentz_trunc":
            keep = (d.abs() <= LORENTZ_SUPPORT_HWHM * gamma).to(dens.dtype)
            return dens * keep / LORENTZ_TRUNC_NORM
        if self.spec.line_shape == "lorentz_bucket":
            # the renderer's REALIZED support: ceil(5 gamma / df) bins rounded up
            # to a power of two, on either side of the centre bin, still divided
            # by the fixed 87.4 % norm — so a realized line carries slightly
            # more than unit area (``build_psd``, bucket rendering)
            half_w = torch.ceil(LORENTZ_SUPPORT_HWHM * gamma.detach() / self.df).clamp_min(1.0)
            bucket = torch.exp2(torch.ceil(torch.log2(half_w)))
            keep = (d.abs() <= bucket * self.df + 0.5 * self.df).to(dens.dtype)
            return dens * keep / LORENTZ_TRUNC_NORM
        return dens

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
        mg = self._mic_gain_db()
        gains = 10.0 ** ((mg - mg.mean(dim=0, keepdim=True)) / 10.0)  # (M, R)
        floor = self.floor()
        lines = torch.einsum("mr,rnf->mnf", gains, self.lines())
        u = self._umod_db()
        if u is not None:
            mod = (
                1.0 + (10.0 ** (u / 10.0) - 1.0)[:, :, None] * self.u_band[None, None, :]
            )  # (M, N, F)
            floor = floor * mod
            if self.spec.umod_lines:
                lines = lines * mod
        spectrum = floor + lines
        g_all = self._gain_all_db()
        if g_all is not None:
            spectrum = spectrum * 10.0 ** ((g_all - g_all.mean()) / 10.0)[:, None, None]
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
        if s.umod_std_db > 0:
            p = p + 0.5 * self.u_z.square().sum()
        if s.free_gamma:
            # smoothness of log gamma along k (second differences), weak
            d2 = (
                self.log_gamma_free[:, 2:]
                - 2 * self.log_gamma_free[:, 1:-1]
                + self.log_gamma_free[:, :-2]
            )
            p = p + 0.5 * d2.square().sum() / 0.3**2
        if s.fit_width_power:
            p = p + 0.5 * self.log_width_power.square().sum() / s.width_power_log_std**2
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
            g0_raw, sl_raw = self._gamma_raw()
            fe, fs_raw = self._floor_exp()
            mg = self._mic_gain_db()
            # every entry goes through the parameter hooks, so a clip inside a
            # rig fit exports the composite values ``forward`` used, not its
            # own unused initializations
            out = dict(
                profile_db=self._profile_db().cpu().numpy().copy(),
                h_db=self.h_db.cpu().numpy().copy(),
                knots_s=self.knots.copy(),
                gamma=self.gamma.cpu().numpy().copy(),
                gamma0=g(g0_raw).cpu().numpy().copy(),
                gamma_slope=g(sl_raw).cpu().numpy().copy(),
                width_power=float(self._width_power().item()),
                floor_mean_db=float(self.floor_mean_db.item()),
                floor_shape_db=(
                    self.spec.floor_shape_std_db * (self.shape_chol @ self._floor_shape_z())
                )
                .cpu()
                .numpy()
                .copy(),
                floor_ctrl_hz=self.ctrl_hz.copy(),
                floor_tilt_db_oct=float(self._floor_tilt_db_oct().item()),
                floor_level_db=(self.spec.floor_gp_std_db * (self.b_chol @ self.floor_level_z))
                .cpu()
                .numpy()
                .copy(),
                floor_tilt_gp=(self.spec.floor_tilt_gp_std * (self.tilt_chol @ self.floor_tilt_z))
                .cpu()
                .numpy()
                .copy(),
                mic_floor_db=self._mic_floor_db().cpu().numpy().copy(),
                mic_gain_db=(mg - mg.mean(dim=0, keepdim=True)).cpu().numpy().copy(),
                amp_exp=float(self._amp_exp().item()),
                floor_exp=float(fe.item()),
                floor_static_rel=float(g(fs_raw).item()),
                rps_offset=(torch.einsum("nj,rj->rn", self.o_interp, self.rps_offset_knots))
                .cpu()
                .numpy()
                .copy(),
                carrier=self.carrier().cpu().numpy().copy(),
            )
            u = self._umod_db()
            if u is not None:
                out["umod_db"] = u.cpu().numpy().copy()
            ga = self._gain_all_db()
            if ga is not None:
                out["gain_all_db"] = (ga - ga.mean()).cpu().numpy().copy()
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
        if s.fit_width_power:
            lines.append(self.log_width_power)
        if s.fit_speed_law:
            lines.append(self.amp_exp)
        if s.rps_offset:
            lines.append(self.rps_offset_knots)
        if s.umod_std_db > 0:
            floor.append(self.u_z)
        return floor + lines


def make_spec(
    pg: Any, *, n_mics: int, f_max: float | None, k_cap: int, variant: dict[str, Any]
) -> Spec:
    """The :class:`Spec` one clip is fitted under.

    The harmonic ladder is sized by the SLOWEST rotor speed in the clip, so
    every order the model carries stays inside the fit band on every frame
    (an order that leaves the band carries no likelihood term and falls to its
    prior — the mechanism behind DREGON's flat profile hold above order 91).
    """
    live = pg.rps[pg.rps > 5.0]
    slowest = float(live.min()) if live.size else 20.0
    nyq = f_max or float(pg.freqs[-1])
    n_harm = int(min(np.floor(nyq / slowest), k_cap))
    return Spec(
        freqs=pg.freqs,
        times=pg.times,
        rps=pg.rps,
        n_mics=n_mics,
        n_harm=max(n_harm, 1),
        f_max=f_max,
        **variant,
    )
