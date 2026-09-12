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
import torch.utils.checkpoint
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

#: Resolution (in bins) and reach of the tabulated window power response used
#: for a COHERENT line. A Hann window's response is below -60 dB past 8 bins.
NEEDLE_U_STEP = 1.0 / 64.0
NEEDLE_U_MAX = 12.0


def hann_power_response(u_step: float = NEEDLE_U_STEP, u_max: float = NEEDLE_U_MAX) -> np.ndarray:
    """``|W(u)|^2`` of the Hann window, ``u`` in bins, unit area over ``u``.

    Tabulated from the window itself rather than from a closed form, so the
    nulls and side lobes are exact. A coherent tone's periodogram is exactly
    this shape, scaled by the tone's power.
    """
    n = 4096
    over = int(round(1.0 / u_step))
    w = np.hanning(n)
    spec = np.abs(np.fft.rfft(w, n=n * over)) ** 2
    u = np.arange(spec.size) / over
    keep = u <= u_max
    tab = spec[keep]
    # unit area over the full (two-sided) u axis: the table is one-sided
    area = (tab[0] + 2.0 * tab[1:].sum()) * u_step
    return tab / area


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
    #: Pin the pedestal width to be order-INDEPENDENT (``gamma_k = gamma0``).
    #: Shaft-angle jitter must scale with ``k``; a modulation with a fixed
    #: bandwidth in hertz does not. Measured on the bench: once ``w_k`` has
    #: fallen and the pedestal is what is visible, its equivalent width is flat
    #: at 4.4-5.6 Hz from k = 24 to k = 64, where a ``k``-linear law fitted the
    #: same clip at 7.5-9.9 Hz. So the pedestal is a fixed-bandwidth process,
    #: not shaft jitter.
    const_gamma: bool = False
    fit_width_power: bool = False
    width_power_log_std: float = 0.5
    mic_floor: bool = False
    fit_speed_law: bool = False
    rps_offset: bool = False
    rps_offset_std: float = 0.3  # rev/s prior std of the carrier correction
    rps_offset_dt_s: float = 0.5
    amp_rps_exponent: float = 2.5
    window_kernel: bool = True
    #: Add the width a line acquires from the rate changing ACROSS one analysis
    #: window, computed from the carrier with no free parameter. Without it the
    #: fitted width absorbs the sweep, which on a ramp is several hertz at
    #: modest orders and is what made a comb of pure tones come back 12-70 Hz
    #: wide. See :meth:`CombSpectrum.gamma_frames`.
    chirp_width: bool = False
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
    #: Score the likelihood with the COHERENCE the width law already implies,
    #: instead of assuming every line is Rayleigh.
    #:
    #: The plain Whittle term ``I / M + log M`` is the exponential density of a
    #: periodogram bin, i.e. it assumes a circular-Gaussian (Rayleigh) line. On
    #: DREGON's bench the per-frame band power of order 2 has a coefficient of
    #: variation of 0.06 -- a deterministic tone -- while order 8 and above sit
    #: at 1.1. The split is NOT a free parameter: for a shaft whose phase is a
    #: Wiener process, a line of half width ``gamma_k`` observed over a window
    #: of length ``T = 1 / df`` has coherent power fraction
    #:
    #:     ``x = 2 pi gamma_k T``,
    #:     ``w = (1 - exp(-x))^2 / (2 (x - 1 + exp(-x)))``
    #:
    #: which is 1 for a line narrow against the window and ``1 / (2x) =
    #: N_c / (2N)`` for a line broad against it (the Schawlow-Townes/laser
    #: linewidth regime). So the same ``gamma0 + slope * k**width_power`` that
    #: sets the MEAN spectrum also sets the bin-power DISTRIBUTION, and the two
    #: must agree -- which is what discriminates the quasi-static (k) shaft
    #: from the diffusive (k^2) one. ``w = 0`` recovers the Whittle term.
    coherence_from_width: bool = False
    #: The competing hypothesis, one fitted parameter: the coherent share is a
    #: property of the EXCITATION rather than of the shaft, ``w_k =
    #: exp(-(k / k_half) ** 2)`` -- deterministic blade passage at low order,
    #: turbulent (amplitude-random) excitation above it. Needed because phase
    #: modulation CONSERVES band power, so a wandering shaft cannot produce the
    #: bench's per-frame band-power CV of 1.09 at k = 8 (measured in a +-6 Hz
    #: band, where a Wiener-phase tone of the fitted width reads 0.00). The two
    #: variants are compared by held-out NLL, not assumed.
    fit_coherence: bool = False
    #: Give the coherent needle the window's own power response
    #: ``|W(f - f0)|^2`` instead of a near-zero Lorentzian smeared by the 3-tap
    #: ``window_kernel``. A coherent tone's periodogram IS the window transform,
    #: nulls included, so this is the only shape it can have. With this on the
    #: needle is not smeared again, because its shape already is the smearing.
    needle_window_shape: bool = False
    coherence_k_half: float = 5.0
    coherence_log_std: float = 0.7
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
        self.needle_tab: Tensor
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
        self.register_buffer(
            "needle_tab", torch.as_tensor(hann_power_response(), dtype=dt, device=dev)
        )
        f_max = spec.f_max or float(freqs[-1])
        band = (freqs >= spec.f_min) & (freqs <= f_max)
        self.register_buffer("band", torch.as_tensor(band, device=dev))
        t_rel = np.asarray(spec.times) - float(spec.times[0])
        t_end = float(t_rel[-1])
        self.t_step = float(t_rel[1] - t_rel[0]) if t_rel.size > 1 else 1.0

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
        self.log_k_half = nn.Parameter(
            torch.full((1,), math.log(max(spec.coherence_k_half, 1e-3)), dtype=dt, device=dev)
        )
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
        if self.spec.const_gamma:
            return g0[:, None].expand(-1, self.K).clamp_min(floor)
        sl = torch.nn.functional.softplus(sl_raw)
        return (g0[:, None] + sl[:, None] * self.k[None, :] ** self._width_power()).clamp_min(floor)

    def gamma_frames(self) -> Tensor:
        """``(R, K, N)`` half width per frame: intrinsic plus the known chirp.

        A line is not stationary inside an analysis window. The rotor changes
        speed, so order ``k`` sweeps ``k * ds`` hertz across the window, and a
        linear sweep of total width ``W`` has a half width ``W / 2``. With a
        window of ``1 / df`` seconds that is

            gamma_chirp = 0.5 * k * |ds/dt| / df

        which at order 8 on a 10 rev/s per second ramp is 5 Hz — the same size
        as the widths the fit was returning for a comb of PURE TONES. Section 12
        of the explainer proves rate drift and line width are exactly degenerate
        inside one window, so the only way to keep the fitted width meaningful
        is to put the part that is KNOWN from the labels in as a covariate with
        no free parameter, leaving the fitted value intrinsic.
        """
        gamma = self.gamma
        if not self.spec.chirp_width:
            return gamma[:, :, None].expand(self.R, self.K, self.N)
        rate = self.carrier()  # (R, N) — the carrier the lines actually ride
        if self.N < 2:
            return gamma[:, :, None].expand(self.R, self.K, self.N)
        slope = torch.zeros_like(rate)
        dt = self.t_step
        slope[:, 1:-1] = (rate[:, 2:] - rate[:, :-2]) / (2.0 * dt)
        slope[:, 0] = (rate[:, 1] - rate[:, 0]) / dt
        slope[:, -1] = (rate[:, -1] - rate[:, -2]) / dt
        sweep = 0.5 * self.k[None, :, None] * slope.abs()[:, None, :] / self.df
        return gamma[:, :, None] + sweep

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

    def _needle_density(self, d: Tensor) -> Tensor:
        """Unit-area density of a COHERENT line at signed offsets ``d`` (Hz).

        This is the analysis window's power response ``|W(u)|^2`` with ``u`` in
        bins, tabulated once from the window itself (so the nulls and the side
        lobes are exact) and read by linear interpolation. Divided by ``df`` it
        is a density per hertz, like :meth:`_line_density`.
        """
        tab = self.needle_tab
        step = NEEDLE_U_STEP
        idx = (d.abs() / self.df) / step
        i0 = idx.floor().clamp(0.0, float(tab.numel() - 2))
        frac = (idx - i0).clamp(0.0, 1.0)
        i0l = i0.to(torch.long)
        lo = tab[i0l]
        hi = tab[i0l + 1]
        inside = (idx <= float(tab.numel() - 2)).to(d.dtype)
        return (lo + (hi - lo) * frac) * inside / self.df

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

    def coherent_fraction(self) -> Tensor | None:
        """``(R, K, N)`` coherent share of each line's power, or ``None``.

        Two variants, selected by held-out NLL (:attr:`Spec.fit_coherence`,
        :attr:`Spec.coherence_from_width`).

        ``coherence_from_width`` derives it with no parameter:
        ``w = (1 - e^-x)^2 / (2 (x - 1 + e^-x))`` with
        ``x = 2 pi gamma_k T`` and ``T = 1 / df`` the analysis window, the
        exact coherent fraction of ``|int_0^T e^{i psi}|^2 / T`` for a Wiener
        shaft phase whose autocorrelation decays as ``e^{-2 pi gamma |tau|}``.
        See :attr:`Spec.coherence_from_width`.
        """
        s = self.spec
        if s.fit_coherence:
            k_half = torch.exp(self.log_k_half[0]).clamp(0.3, 1e3)
            w = torch.exp(-((self.k / k_half) ** 2))
            return w.clamp(1e-6, 1.0 - 1e-6)[None, :, None].expand(self.R, self.K, self.N)
        if not s.coherence_from_width:
            return None
        x = (2.0 * math.pi * self.gamma_frames() / self.df).clamp(1e-6, 1e6)
        em = torch.expm1(-x)  # = e^-x - 1, accurate as x -> 0
        w = em.square() / (2.0 * (x + em))
        return w.clamp(1e-6, 1.0 - 1e-6)

    def lines(
        self,
        k_chunk: int = 32,
        weight: Tensor | None = None,
        gamma: Tensor | None = None,
        shape: str = "line",
    ) -> Tensor:
        """``(R, N, F)`` the rotors' line spectra (before microphone gains).

        ``weight`` (``(R, K, N)``) scales each line's power and ``gamma``
        overrides its half width, which is how the coherent needle and the
        incoherent pedestal are built from one line model. ``shape="needle"``
        reads the window response instead of the line law.
        """
        power = self.line_power()  # (R, K, N)
        if weight is not None:
            power = power * weight
        gamma = self.gamma_frames() if gamma is None else gamma  # (R, K, N)
        needle = shape == "needle"
        carrier = self.carrier()  # (R, N)
        out = torch.zeros(self.R, self.N, self.F, dtype=self._dtype, device=self._dev)
        k_top = min(self.K, self.active_k)

        def chunk(p_c: Tensor, g_c: Tensor, k_c: Tensor) -> Tensor:
            centres = k_c[None, :, None] * carrier[:, None, :]  # (R, k, N)
            d = self.freqs[None, None, None, :] - centres[..., None]  # (R, k, N, F)
            dens = self._needle_density(d) if needle else self._line_density(d, g_c[..., None])
            return torch.einsum("rkn,rknf->rnf", p_c, dens)

        for k0 in range(0, k_top, k_chunk):
            k1 = min(k0 + k_chunk, k_top)
            args = (power[:, k0:k1], gamma[:, k0:k1, :], self.k[k0:k1])
            # Each chunk holds several (R, k, N, F) intermediates: the offsets,
            # the two arctangents of the bin integral, the support mask. Keeping
            # them all for the backward pass over 200 orders and five clips
            # exhausted 30 GB and the fit was killed with no traceback.
            # Recomputing them in the backward pass costs about a third more
            # time and removes the peak.
            if torch.is_grad_enabled():
                piece: Tensor = torch.utils.checkpoint.checkpoint(chunk, *args, use_reentrant=False)  # type: ignore[assignment]
            else:
                piece = chunk(*args)
            out = out + piece
        return out

    def needle_gamma(self) -> Tensor:
        """``(R, K, N)`` half width of the COHERENT component: none of its own.

        A coherent tone's periodogram is the window transform, so its width in
        this model comes from ``window_kernel``, not from the line law. Measured
        on the bench: a pure tone reads an equivalent width of 1.49 Hz in a
        16384-point Hann window (1.5 bins), and order 1 of a real clip reads
        1.50 Hz.
        """
        return torch.full_like(self.gamma_frames(), 0.01 * self.df)

    def line_total(self) -> Tensor:
        """``(R, N, F)`` the line spectra, one or two components.

        With a coherence split each order is a mixture of a window-limited
        needle (share ``w_k``) and a pedestal of the fitted width (share
        ``1 - w_k``). That mixture is what produced the bench's apparent width
        law: the measured equivalent width goes 1.50 Hz at k = 1 to 5.3 Hz at
        k >= 24 with no change of pedestal width, purely because ``w_k`` falls.
        Two independent statistics -- the equivalent width and the centre-bin
        share -- return the same ``w_k`` to within 0.005 over k = 1..16, so the
        split lives in the MEAN spectrum and the ordinary Whittle likelihood
        identifies it.
        """
        w = self.coherent_fraction()
        if w is None:
            return self.lines()
        return self.needle(w) + self.lines(weight=1.0 - w)

    def needle(self, w: Tensor) -> Tensor:
        """``(R, N, F)`` the coherent component of every line.

        With :attr:`Spec.needle_window_shape` the component is built only on the
        bins the window response actually reaches (``NEEDLE_U_MAX`` either side),
        and scattered into place. Evaluating it over the whole band instead cost
        969 ms per gradient step against 189 ms for a one-component model; this
        form is exact to floating point and removes that cost.
        """
        if not self.spec.needle_window_shape:
            return self.lines(weight=w, gamma=self.needle_gamma())
        power = self.line_power() * w  # (R, K, N)
        if self.active_k < self.K:
            power = power * (self.k <= self.active_k).to(power.dtype)[None, :, None]
        centres = self.k[None, :, None] * self.carrier()[:, None, :]  # (R, K, N)
        f0 = self.freqs[0]
        near = int(math.ceil(NEEDLE_U_MAX)) + 1
        offs = torch.arange(-near, near + 1, device=self._dev, dtype=self._dtype)
        base = torch.round((centres - f0) / self.df)  # (R, K, N)
        bins = base[..., None] + offs  # (R, K, N, B)
        d = f0 + bins * self.df - centres[..., None]
        contrib = power[..., None] * self._needle_density(d)
        inside = ((bins >= 0) & (bins <= self.F - 1)).to(self._dtype)
        # (R, K, N, B) -> (R, N, K*B) so one scatter per (rotor, frame) row
        vals = (contrib * inside).permute(0, 2, 1, 3).reshape(self.R * self.N, -1)
        idx = (
            bins.clamp(0, self.F - 1)
            .permute(0, 2, 1, 3)
            .reshape(self.R * self.N, -1)
            .to(torch.long)
        )
        out = torch.zeros(self.R * self.N, self.F, dtype=self._dtype, device=self._dev)
        out = out.scatter_add(1, idx, vals)
        return out.reshape(self.R, self.N, self.F)

    def parts(self) -> tuple[Tensor | None, Tensor]:
        """``(coherent, stochastic)`` expected periodograms.

        The coherent part is the deterministic phasor power; the stochastic part
        is the incoherent line share plus the floor. ``coherent`` is ``None``
        when the split is off, and then ``stochastic`` is :meth:`forward`.
        """
        w = self.coherent_fraction()
        if w is None:
            return None, self.forward()
        mg = self._mic_gain_db()
        gains = 10.0 ** ((mg - mg.mean(dim=0, keepdim=True)) / 10.0)
        coh = self._post(
            torch.zeros_like(self.floor()) + torch.einsum("mr,rnf->mnf", gains, self.needle(w)),
            smear=not self.spec.needle_window_shape,
        )
        inc = self._post(
            self.floor() + torch.einsum("mr,rnf->mnf", gains, self.lines(weight=1.0 - w))
        )
        return coh, inc

    def _post(self, spectrum: Tensor, smear: bool = True) -> Tensor:
        """The per-mic gain and the window smearing a spectrum share carries.

        ``smear=False`` is for a component whose shape already IS the window
        response (the coherent needle), which must not be smeared twice.
        """
        g_all = self._gain_all_db()
        if g_all is not None:
            spectrum = spectrum * 10.0 ** ((g_all - g_all.mean()) / 10.0)[:, None, None]
        if smear and self.spec.window_kernel:
            x = spectrum.reshape(-1, 1, self.F)
            x = torch.nn.functional.conv1d(
                torch.nn.functional.pad(x, (1, 1), mode="replicate"), self.kernel
            )
            spectrum = x.reshape(self.M, self.N, self.F)
        return spectrum

    def forward(self) -> Tensor:
        """``(M, N, F)`` expected periodogram."""
        if self.spec.needle_window_shape and self.coherent_fraction() is not None:
            coh, inc = self.parts()
            assert coh is not None
            return coh + inc
        mg = self._mic_gain_db()
        gains = 10.0 ** ((mg - mg.mean(dim=0, keepdim=True)) / 10.0)  # (M, R)
        floor = self.floor()
        lines = torch.einsum("mr,rnf->mnf", gains, self.line_total())
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
        if s.fit_coherence:
            ref = math.log(max(s.coherence_k_half, 1e-3))
            p = p + 0.5 * ((self.log_k_half[0] - ref) / s.coherence_log_std) ** 2
        return p

    def whittle(self, power: Tensor, model: Tensor | None = None) -> Tensor:
        """Summed Whittle NLL over the fit band: ``sum I/M + log M``.

        This is the exponential density of a periodogram bin, i.e. it treats
        every line as circular Gaussian. Coherence is NOT scored here: a
        coherent tone's periodogram shape is the window transform
        ``|W(f - f0)|^2``, with spectral nulls, not the smooth Lorentzian this
        model carries, so assigning coherent power bin-by-bin puts power where
        the realized tone has a null and the term diverges. Measured: on a clip
        whose low orders were rendered as pure tones, a bin-level Rice term
        preferred zero coherence by 3268 nats over the planted value, on the
        line cells alone. Coherence is scored by
        :meth:`band_coherence_nll` instead, which integrates the band and is
        therefore blind to line shape and sub-bin centre error.
        """
        m = self.forward() if model is None else model
        m = m.clamp_min(1e-12)
        cell = power / m + torch.log(m)
        return cell[..., self.band].sum()

    def line_bands(self, half_bins: float = 3.0, k_chunk: int = 16) -> Tensor:
        """``(R, K, N, F)`` per-line band membership, as a float mask."""
        carrier = self.carrier()  # (R, N)
        out = torch.zeros(self.R, self.K, self.N, self.F, dtype=self._dtype, device=self._dev)
        width = half_bins * self.df
        for k0 in range(0, self.K, k_chunk):
            k1 = min(k0 + k_chunk, self.K)
            centres = self.k[k0:k1][None, :, None] * carrier[:, None, :]  # (R, k, N)
            d = (self.freqs[None, None, None, :] - centres[..., None]).abs()
            out[:, k0:k1] = ((d <= width) & self.band[None, None, None, :]).to(self._dtype)
        return out

    def band_coherence_nll(
        self,
        power: Tensor,
        *,
        nu: float,
        half_bins: float = 3.0,
        n_terms: int = 48,
    ) -> Tensor:
        """NLL of the per-line BAND powers under the coherent/stochastic split.

        A band of ``2 * half_bins + 1`` bins around each line centre holds a
        deterministic share ``C`` and a stochastic share, so its power is
        non-central chi-square with ``2 nu`` degrees of freedom and
        noncentrality ``2 C / sigma^2``, evaluated by its Poisson mixture

            ``p(x) = sum_j Poisson(j; lambda/2) chi2(x; 2 nu + 2 j)``

        ``nu`` is the band's effective degrees of freedom, which is a property
        of the ANALYSIS (window, overlap), not of the fit, and is measured on
        line-free bands by :func:`measure_band_dof`.

        Integrating the band is what makes this well specified where the
        bin-level term is not: the statistic is insensitive to the line's shape
        and to sub-bin centre error, while still separating a tone
        (band power nearly deterministic) from a Rayleigh line (band power
        exponential) -- the bench's measured CV of 0.06 at k = 2 against 1.09
        at k = 8.
        """
        if self.coherent_fraction() is None:
            raise ValueError("band_coherence_nll needs a coherence variant enabled")
        mask = self.line_bands(half_bins)  # (R, K, N, F)
        obs = torch.einsum("mnf,rknf->mrkn", power, mask)
        coh, inc = self.parts()
        assert coh is not None
        c = torch.einsum("mnf,rknf->mrkn", coh, mask)
        n_tot = torch.einsum("mnf,rknf->mrkn", inc, mask)
        live = (mask.sum(dim=-1) > 0).to(self._dtype)[None]  # (1, R, K, N)
        cell = noncentral_band_nll(obs, c, n_tot, nu=nu, n_terms=n_terms)
        return (cell * live).sum()

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
                coherence_k_half=(
                    float(torch.exp(self.log_k_half[0]).item()) if self.spec.fit_coherence else 0.0
                ),
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
        if s.fit_coherence:
            lines.append(self.log_k_half)
        if s.fit_speed_law:
            lines.append(self.amp_exp)
        if s.rps_offset:
            lines.append(self.rps_offset_knots)
        if s.umod_std_db > 0:
            floor.append(self.u_z)
        return floor + lines


#: The forward model every fit uses unless a ladder rung overrides it.
#:
#: WHY EACH ENTRY. `rps_offset` because telemetry is not the shaft, with a prior
#: of 1 rev/s to cover both rigs' measured robust scales (0.41 on Michael's
#: crops, 1.38 on DREGON's) — a 0.3 prior fought an offset three times its size
#: and the mismatch was bought with line width. `line_bin_integrate` because a
#: line narrower than a bin cannot be point-sampled at bin centres without
#: aliasing, and the exact bin integral costs two arctangents. `gamma_min_bins`
#: at 0.01 instead of 0.6 because the 0.6-bin floor DOUBLE-COUNTED the window:
#: `window_kernel` already convolves the model with the Hann power response, so
#: a floor of 0.6 bins (4.7 Hz at n_fft 2048) forbade the model from
#: representing anything narrower than the window while the kernel was already
#: supplying that broadening. Measured consequence: a comb of pure tones came
#: back with 4.69 Hz widths — exactly the floor — while the bench measures real
#: lines at 0.14 Hz at order 4 and 1.87 at order 32, so every flight width fit
#: was censored 3 to 30 times too wide. 0.01 bins is 0.078 Hz at n_fft 2048,
#: under the narrowest line the bench resolves (0.14 Hz at order 4), because a
#: floor above anything we intend to represent is a censor, not a safeguard. `chirp_width` because the rest of the
#: excess is the rate sweeping across the window, which is known from the
#: labels and must not be paid for with a free parameter.
BASE_VARIANT: dict[str, Any] = dict(
    rps_offset=True,
    rps_offset_std=1.0,
    line_shape="gauss",
    line_bin_integrate=True,
    gamma_min_bins=0.01,
    chirp_width=True,
)


def noncentral_band_nll(
    obs: Tensor,
    coherent: Tensor,
    stochastic: Tensor,
    *,
    nu: float,
    n_terms: int = 48,
) -> Tensor:
    """Per-band NLL of a band power holding deterministic and random shares.

    A band whose content is a deterministic phasor of power ``C`` plus circular
    complex Gaussian noise of total power ``N`` spread over ``nu`` independent
    modes has band power ``B`` distributed as a scaled non-central chi-square:
    with ``sigma^2 = N / (2 nu)`` (the per-real-component variance), ``B /
    sigma^2 ~ chi2_{2 nu}(lambda)`` at ``lambda = C / sigma^2``, so
    ``E[B] = N + C`` and the spread collapses as ``C`` takes over. Evaluated by
    the Poisson mixture ``p(x) = sum_j Pois(j; lambda/2) chi2(x; 2 nu + 2 j)``,
    which is exact, differentiable, and needs no Bessel function of fractional
    order.

    ``C = 0`` gives a Gamma(``nu``) band power, the Whittle/Rayleigh case.
    """
    sigma2 = (stochastic / (2.0 * nu)).clamp_min(1e-30)
    x = (obs / sigma2).clamp_min(1e-30)
    lam = (coherent / sigma2).clamp_min(1e-30)
    j = torch.arange(n_terms, dtype=obs.dtype, device=obs.device)
    jj = j.view(-1, *([1] * x.dim()))
    half_lam = 0.5 * lam
    df2 = nu + jj  # half the degrees of freedom of the j-th term
    terms = (
        -half_lam
        + jj * torch.log(half_lam)
        - torch.lgamma(j + 1.0).view(-1, *([1] * x.dim()))
        + (df2 - 1.0) * torch.log(x)
        - 0.5 * x
        - df2 * math.log(2.0)
        - torch.lgamma(df2)
    )
    return -(torch.logsumexp(terms, dim=0) - torch.log(sigma2))


def measure_band_dof(
    power: np.ndarray,
    freqs: np.ndarray,
    centres: np.ndarray,
    *,
    half_bins: float = 3.0,
    df: float | None = None,
) -> float:
    """Effective degrees of freedom of a line band, measured on LINE-FREE bands.

    The band power of pure noise over ``2 * half_bins + 1`` bins would have
    ``nu`` equal to the bin count if the bins were independent, but a Hann
    window with 50 % overlap correlates neighbours, so the real number is
    smaller. It is a property of the analysis, not of the fit, so it is
    measured here rather than assumed: bands are placed HALFWAY between
    consecutive line centres, where no line sits, and ``nu = mean^2 / var`` of
    their per-frame power.

    Args:
        power: ``(N, F)`` per-frame periodogram of one channel.
        freqs: ``(F,)`` bin frequencies.
        centres: ``(K,)`` line centre frequencies (to be avoided).
        half_bins: half width of the band, in bins.
        df: bin spacing; taken from ``freqs`` when omitted.
    """
    step = float(freqs[1] - freqs[0]) if df is None else df
    gaps = 0.5 * (centres[:-1] + centres[1:])
    nus: list[float] = []
    for g in gaps:
        sel = np.abs(freqs - g) <= half_bins * step
        if sel.sum() < 2 * half_bins:
            continue
        b = power[:, sel].sum(axis=1)
        if b.mean() <= 0 or b.var() <= 0:
            continue
        nus.append(float(b.mean() ** 2 / b.var()))
    if not nus:
        raise ValueError("no line-free band available to measure the dof")
    return float(np.median(nus))


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
