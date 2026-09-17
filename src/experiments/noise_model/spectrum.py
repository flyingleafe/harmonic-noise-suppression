"""The v2 expected periodogram ``M``, in both of the campaign's two windows.

TWO MODES, one parameter set, one lag law (:func:`.lag.r_tau`):

* **BENCH** (:func:`bench_model`) — a stationary support: constant carriers, ONE
  periodic-Hann window over the whole stationary segment, no STFT, no moving
  atom. The whole-segment periodogram is the sufficient statistic of a
  stationary record, so there is nothing to gain from a moving frame and a lot
  to gain in resolution (0.033 Hz over 30 s).
* **FLIGHT** (:func:`flight_model`) — the moving label carrier through the
  front-end STFT (NFFT 2048, hop 512 at 16 kHz), which is
  :meth:`revised_phase._RevisedModel.frame_model`'s math with the v2 lag law
  substituted at the kernel call. The shared finite-window kernel
  :func:`phase_kernel.expected_periodogram_from_atoms` is IMPORTED, never
  reimplemented, so a v2 flight prediction and a C4 one differ only where the
  model differs — which is what the differential test in
  ``tests/experiments/test_noise_model_core.py`` pins to 1e-6.

WHY BENCH IS NOT A KERNEL CALL. With a constant carrier the complex atom is
``a_t = w_t A e^{i omega t}``, so its finite-window autocorrelation is
``g(tau) = A^2 e^{i omega tau} (w * w)(tau)`` EXACTLY, and the kernel's
``P_Z = DFT[g R]`` collapses to a shift of one order-independent transform.
Summing the whole comb's autocovariance in the LAG domain and transforming ONCE
per rotor therefore computes the same number as ``R x K`` kernel calls, at
``O((R + 1) N log N)`` instead of ``O(R K N log N)``:

    c_r(tau) = sum_k P_rk R_rk(tau) cos(2 pi k f_r tau)      (per rotor)
    M_m(f)   = [ sum_{|tau| < N} (w * w)(tau) c_m(tau) e^{-2 pi i f tau} ]
               / sum_t w_t^2,        c_m = sum_r G_mr c_r + G^floor_m c_floor

and since ``(w * w)`` and every ``c`` are real and EVEN in ``tau``, the DTFT is
real and the analysis bins ``f_j = j Fs / N`` are read off one rFFT of length
``2 N`` at its even bins — the same ``L = 2 n_fft`` anti-aliasing of the lag
wrap the kernel uses, for the same reason. The equality to the kernel is a
test, not a claim (``test_bench_matches_finite_window_kernel``).

WHAT MAKES IT AFFORDABLE. Only the SHAFT factor of the lag law decays without
bound, and it decays as ``exp(-k^2 V_theta(tau)/2)``: at the prior centre order
110 of a 70 rev/s rotor is below ``e^-40`` by 216 ms, i.e. 0.7 % of a 30 s
segment. :func:`.lag.order_lag_support_s` gives that lag per order and
:func:`order_groups` buckets the orders into power-of-two lag blocks, so the
comb's autocovariance costs ``sum_k min(N, sr tau_k)`` element-ops instead of
``K N``. Nothing is approximated beyond a factor below ``e^-40``; the retained
support carries the exact gradient.

THE TRANSFER, applied EXACTLY ONCE, in both modes, on the sum of lines and
floor: :func:`stage2.render_transfer_power` of the render anti-alias plus the
decimator. It is in the BENCH model too, although a bench recording never went
through the renderer, because the thing the campaign must be able to do is
RENDER a fitted bench rig (:func:`.render.render_noise` synthesises at 64 kHz
and decimates) and read back the ``M`` that was fitted. The smooth 5 dB it
costs at 7900 Hz is absorbed by the per-order profile and the floor shape,
which are free; a mode-dependent transfer would instead put a 5 dB error
between the fit and the render.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from experiments.stochastic_fit.model import (
    AMP_RPS_REF,
    FLOOR_SHAPE_F_MIN,
    FLOOR_SHAPE_N_CTRL,
    se_cholesky,
)
from experiments.stochastic_fit.phase_kernel import (
    expected_periodogram_from_atoms,
    hann_window,
)
from experiments.stochastic_fit.revised_phase import (
    FLOOR_PSD_OVERSAMPLE,
    FLOOR_SHAPE_OCT,
    FLOOR_SHAPE_STD_DB,
    MODEL_FLOOR,
    SPEED_FLOOR_RPS,
    floor_geometry,
    floor_power_spectrum,
)
from experiments.stochastic_fit.stage2 import render_transfer_power

from .lag import P_ORDER_EXPONENT, order_lag_support_s, r_tau

__all__ = [
    "BAND_F_MIN",
    "BAND_F_MAX",
    "BenchGrid",
    "FlightGrid",
    "FloorBasis",
    "FloorParams",
    "V2Params",
    "bench_band",
    "bench_grid",
    "bench_model",
    "flight_grid",
    "flight_model",
    "flight_rate_work",
    "k_max_for_carrier",
    "order_groups",
    "refine_bench_carrier",
]

#: The observation band. ``BAND_F_MAX`` is the frozen evaluator edge; a bench
#: support sampled below 17.6 kHz is additionally capped at ``0.45 sr`` so the
#: decimator's own rolloff never enters the likelihood.
BAND_F_MIN = 30.0
BAND_F_MAX = 7900.0

#: The flight front end of the v2 campaign (``docs/explainers/
#: noise-model-v2-plan.qmd``, "Where and how the rig is fitted"). NOT C4's
#: 16384/1024: a long window smears a moving line by ``k |df/dt| T``.
FLIGHT_N_FFT = 2048
FLIGHT_HOP = 512
FLIGHT_SR = 16000
SAMPLE_RATE_WORK = 64000

#: Lags of the floor's autocovariance the bench model reads. The floor's
#: sharpest structure is the 30 Hz low-end shape control (~33 ms), so 1.024 s
#: is thirty correlation times; :func:`bench_grid` reports the retained tail so
#: the truncation is measured rather than trusted.
BENCH_FLOOR_LAG = 1 << 14

#: ``exp(-nats)`` below which an order's shaft factor is dropped from the bench
#: autocovariance. 40 nats is 4e-18, under float64's grip on a line that is at
#: most 80 dB over the floor.
LAG_SUPPORT_NATS = 40.0


# ── parameters ──────────────────────────────────────────────────────────────


@dataclass
class FloorParams:
    """The coloured floor, C4's parameterisation unchanged.

    ``shape_z`` is the standard-normal GP coordinate of
    :meth:`revised_phase._RevisedModel.floor_shape_db`; ``exp`` and
    ``static_rel`` are the flight speed law of the floor's envelope and are
    ignored on the bench, where the carrier is constant and the envelope is a
    constant absorbed by ``mean_db``.
    """

    mean_db: Any
    shape_z: Any
    tilt_db_oct: Any
    mic_floor_db: Any
    exp: Any = 0.0
    static_rel: Any = 0.0


@dataclass
class V2Params:
    """Every fitted quantity of one support's forward model.

    ``sigma_eps``/``lam_eps`` are ``(even, odd)`` pairs (see :mod:`.lag`);
    ``carrier_rev_s`` is ``(R,)`` on the bench (a fitted constant per rotor)
    and unused in flight, where the label supplies the carrier;
    ``mic_line_gain_db`` is ``(M, R)`` and ``gain_all_db`` is ``(M,)``, both
    mean-pinned exactly as C4 pins them, so neither can absorb the overall
    level.
    """

    sigma_nu: Any
    lam: Any
    sigma_eps: Any
    lam_eps: Any
    profile_db: Any
    floor: FloorParams
    mic_line_gain_db: Any
    gain_all_db: Any
    carrier_rev_s: Any = None
    amp_exp: Any = 0.0
    p: float = P_ORDER_EXPONENT


# ── geometry ────────────────────────────────────────────────────────────────


def _as_t(v: Any, ref: Tensor) -> Tensor:
    if isinstance(v, Tensor):
        return v.to(dtype=ref.dtype, device=ref.device)
    return torch.as_tensor(v, dtype=ref.dtype, device=ref.device)


def _mean_pinned_db(db: Tensor) -> Tensor:
    """``10^((db - mean_over_mics(db)) / 10)``: C4's mic-line gain, pinned."""
    return 10.0 ** ((db - db.mean(dim=0, keepdim=True)) / 10.0)


@dataclass(frozen=True)
class FloorBasis:
    """Pure geometry of the coloured floor on one grid: no fitted number."""

    ctrl_hz: np.ndarray
    shape_psd: Tensor
    tilt_oct_psd: Tensor
    psd_len: int
    lag_len: int
    rate_factor: float
    shape_chol: Tensor

    def shape_db(self, shape_z: Any) -> Tensor:
        z = _as_t(shape_z, self.shape_chol)
        return FLOOR_SHAPE_STD_DB * (self.shape_chol @ z)

    def psd(self, floor: FloorParams) -> Tensor:
        """The floor power spectrum on this basis' PSD grid, inside the graph."""
        return floor_power_spectrum(
            self.shape_psd,
            self.tilt_oct_psd,
            mean_db=_as_t(floor.mean_db, self.shape_psd),
            ctrl_db=self.shape_db(floor.shape_z),
            tilt_db_oct=_as_t(floor.tilt_db_oct, self.shape_psd),
            rate_factor=self.rate_factor,
        )

    def autocovariance(self, floor: FloorParams) -> Tensor:
        """``c(tau)``, ``tau = 0 .. lag_len - 1``, of the REAL floor process.

        ``irfft`` of a one-sided expected periodogram is the autocovariance in
        the very units the periodogram is in (white noise of variance ``s^2``
        reads ``s^2``), which is why the bench model multiplies it straight
        into the window autocorrelation with no factor of two: the two appears
        only in the complex-atom convention the kernel wants
        (:meth:`revised_phase._RevisedModel.floor_covariance`).
        """
        psd = self.psd(floor)
        c = torch.fft.irfft(psd.to(torch.complex128), n=self.psd_len).real
        return c[: self.lag_len]


def _floor_basis(
    *, sr_grid: int, lag_len: int, ctrl_max_hz: float, rate_factor: float, device: Any
) -> FloorBasis:
    ctrl_hz = np.geomspace(FLOOR_SHAPE_F_MIN, float(ctrl_max_hz), FLOOR_SHAPE_N_CTRL)
    ctrl_oct = np.log2(ctrl_hz / ctrl_hz[0])
    psd_len = int(FLOOR_PSD_OVERSAMPLE) * int(lag_len)
    shape, tilt = floor_geometry(np.fft.rfftfreq(psd_len, d=1.0 / float(sr_grid)), ctrl_hz)
    return FloorBasis(
        ctrl_hz=ctrl_hz,
        shape_psd=torch.as_tensor(shape, dtype=torch.float64, device=device),
        tilt_oct_psd=torch.as_tensor(tilt, dtype=torch.float64, device=device),
        psd_len=psd_len,
        lag_len=int(lag_len),
        rate_factor=float(rate_factor),
        shape_chol=torch.as_tensor(
            se_cholesky(FLOOR_SHAPE_N_CTRL, float(ctrl_oct[1] - ctrl_oct[0]), FLOOR_SHAPE_OCT),
            dtype=torch.float64,
            device=device,
        ),
    )


def floor_ctrl_hz(sr: int) -> np.ndarray:
    """The floor's shape control points: C4's geometric ladder to the Nyquist.

    ONE definition, read by :func:`bench_grid`, :func:`flight_grid` AND
    :func:`.render.render_noise`, so a fitted ``floor_shape_z`` means the same
    curve in the fit and in the render.
    """
    return np.geomspace(FLOOR_SHAPE_F_MIN, 0.5 * float(sr), FLOOR_SHAPE_N_CTRL)


def floor_shape_chol(ctrl_hz: np.ndarray) -> np.ndarray:
    """The squared-exponential Cholesky of the shape GP on ``ctrl_hz``."""
    oct_ = np.log2(np.asarray(ctrl_hz, dtype=np.float64) / float(ctrl_hz[0]))
    return se_cholesky(FLOOR_SHAPE_N_CTRL, float(oct_[1] - oct_[0]), FLOOR_SHAPE_OCT)


def floor_shape_db(shape_z: np.ndarray, *, sr: int) -> np.ndarray:
    """``FLOOR_SHAPE_STD_DB * (chol @ z)``: the dB control values from the GP
    coordinate, the numpy twin of :meth:`FloorBasis.shape_db`."""
    chol = floor_shape_chol(floor_ctrl_hz(sr))
    return FLOOR_SHAPE_STD_DB * (chol @ np.asarray(shape_z, dtype=np.float64))


def bench_band(freqs_hz: np.ndarray, sr: int) -> np.ndarray:
    """``30 Hz .. min(7900, 0.45 sr)``, the bench observation band."""
    f = np.asarray(freqs_hz, dtype=np.float64)
    return (f >= BAND_F_MIN) & (f <= min(BAND_F_MAX, 0.45 * float(sr)))


@dataclass(frozen=True)
class BenchGrid:
    """Everything a stationary whole-segment support fixes before any fit."""

    sr: int
    n: int
    freqs_hz: np.ndarray
    band: np.ndarray
    win_acf: Tensor
    window_sumsq: float
    tau_s: Tensor
    transfer_power: Tensor
    floor: FloorBasis
    diagnostics: dict[str, Any] = field(default_factory=dict)


def bench_grid(
    *,
    n: int,
    sr: int,
    device: Any = "cpu",
    floor_lag: int | None = None,
    apply_transfer: bool = True,
) -> BenchGrid:
    """Geometry of one stationary segment of ``n`` samples at ``sr``.

    ``n`` is BOTH the window length and the FFT length: the support's
    periodogram is one periodic-Hann frame over the whole segment, so
    ``freqs_hz = rfftfreq(n, 1/sr)``.
    """
    n = int(n)
    if n < 16:
        raise ValueError(f"a bench segment of {n} samples is not a support")
    w = hann_window(n)
    # sum_t w_t w_{t - tau} for tau = 0 .. n - 1, by the same 2 n transform the
    # kernel uses: a length-n autocorrelation wrapped into n bins would alias
    # lag n - 1 onto lag 1.
    spec = np.fft.rfft(w, n=2 * n)
    acf = np.fft.irfft(spec.real**2 + spec.imag**2, n=2 * n)[:n]
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
    transfer = (
        np.asarray(
            render_transfer_power(
                freqs, sample_rate_work=SAMPLE_RATE_WORK, sample_rate_out=int(sr)
            ),
            dtype=np.float64,
        )
        if apply_transfer
        else np.ones_like(freqs)
    )
    lag_len = min(n, int(BENCH_FLOOR_LAG if floor_lag is None else floor_lag))
    basis = _floor_basis(
        sr_grid=sr,
        lag_len=lag_len,
        ctrl_max_hz=float(freqs[-1]),
        # the floor parameters are at the ANALYSIS convention and the bench
        # model works directly on the analysis grid, so the work/analysis
        # pre-compensation of ``floor_power_spectrum`` and the fit's
        # ``grid_power_factor`` both collapse to one.
        rate_factor=1.0,
        device=device,
    )
    return BenchGrid(
        sr=int(sr),
        n=n,
        freqs_hz=freqs,
        band=bench_band(freqs, sr),
        win_acf=torch.as_tensor(acf, dtype=torch.float64, device=device),
        window_sumsq=float(np.sum(w**2)),
        tau_s=torch.arange(n, dtype=torch.float64, device=device) / float(sr),
        transfer_power=torch.as_tensor(transfer, dtype=torch.float64, device=device),
        floor=basis,
        diagnostics=dict(
            n_samples=n,
            duration_s=n / float(sr),
            bin_hz=float(sr) / n,
            floor_lag_len=lag_len,
            floor_lag_s=lag_len / float(sr),
            n_band_bins=int(bench_band(freqs, sr).sum()),
            transfer_applied=bool(apply_transfer),
        ),
    )


@dataclass(frozen=True)
class FlightGrid:
    """C4's moving-frame geometry at the v2 flight front end (2048 / 512)."""

    sr: int
    n_fft: int
    hop: int
    sr_work: int
    n_fft_work: int
    oversample: int
    freqs_hz: np.ndarray
    band: np.ndarray
    window_work: Tensor
    window_work_sumsq: float
    tau_s_work: Tensor
    win_idx_work: Tensor
    transfer_power: Tensor
    grid_power_factor: float
    floor: FloorBasis
    diagnostics: dict[str, Any] = field(default_factory=dict)


def flight_grid(
    *,
    sr: int = FLIGHT_SR,
    n_fft: int = FLIGHT_N_FFT,
    hop: int = FLIGHT_HOP,
    sr_work: int = SAMPLE_RATE_WORK,
    device: Any = "cpu",
    apply_transfer: bool = True,
) -> FlightGrid:
    """Geometry of the moving-carrier STFT model, C4's construction verbatim."""
    sr, n_fft, sr_work = int(sr), int(n_fft), int(sr_work)
    if sr_work % sr:
        raise ValueError(f"sample_rate_work {sr_work} must be an integer multiple of sr {sr}")
    n_fft_work = n_fft * sr_work // sr
    w = hann_window(n_fft_work)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    transfer = (
        np.asarray(
            render_transfer_power(freqs, sample_rate_work=sr_work, sample_rate_out=sr),
            dtype=np.float64,
        )
        if apply_transfer
        else np.ones_like(freqs)
    )
    return FlightGrid(
        sr=sr,
        n_fft=n_fft,
        hop=int(hop),
        sr_work=sr_work,
        n_fft_work=n_fft_work,
        oversample=sr_work // sr,
        freqs_hz=freqs,
        band=(freqs >= BAND_F_MIN) & (freqs <= BAND_F_MAX),
        window_work=torch.as_tensor(w, dtype=torch.float64, device=device),
        window_work_sumsq=float(np.sum(w**2)),
        tau_s_work=torch.arange(n_fft_work, dtype=torch.float64, device=device) / float(sr_work),
        win_idx_work=torch.arange(n_fft_work, dtype=torch.int64, device=device),
        transfer_power=torch.as_tensor(transfer, dtype=torch.float64, device=device),
        grid_power_factor=float(sr) / float(sr_work),
        floor=_floor_basis(
            sr_grid=sr_work,
            lag_len=n_fft_work,
            ctrl_max_hz=float(freqs[-1]),
            rate_factor=float(sr_work) / float(sr),
            device=device,
        ),
        diagnostics=dict(sr=sr, n_fft=n_fft, hop=int(hop), sr_work=sr_work, n_fft_work=n_fft_work),
    )


# ── orders ──────────────────────────────────────────────────────────────────


def k_max_for_carrier(carrier_rev_s: Any, sr: int, *, k_cap: int | None = None) -> int:
    """Highest order whose line sits strictly below the analysis Nyquist.

    Orders above it were removed from the real recording by its own front end
    (a decimator's anti-alias, never an alias), and they contribute to the
    observed band only through line skirts far below the floor — so they are
    not modelled. ``k_cap`` clamps the count to the profile's width.
    """
    f = float(np.max(np.asarray(carrier_rev_s, dtype=np.float64)))
    if f <= 0.0:
        raise ValueError(f"carrier must be positive, got {f}")
    k = int(math.floor(0.5 * float(sr) / f - 1e-9))
    return max(1, k if k_cap is None else min(k, int(k_cap)))


def order_groups(
    k_max: int,
    *,
    sigma_nu: float,
    lam: float,
    sr: int,
    n: int,
    threshold_nats: float = LAG_SUPPORT_NATS,
    min_len: int = 256,
) -> list[tuple[int, np.ndarray]]:
    """``[(lag_len, orders)]``: orders bucketed by a fixed shaft-lag geometry.

    The support is a POWER OF TWO so a handful of buckets covers every order.
    Callers choose ``sigma_nu`` and ``lam`` before they build the Pyro graph;
    the lag law evaluated on that grid remains fully differentiable in the
    sampled dynamics parameters.
    """
    k = np.arange(1, int(k_max) + 1, dtype=np.float64)
    tau = np.atleast_1d(
        order_lag_support_s(
            k, sigma_nu=sigma_nu, lam=lam, threshold_nats=threshold_nats, tau_cap_s=n / float(sr)
        )
    )
    want = np.ceil(tau * float(sr)).astype(np.int64) + 1
    # the next power of two at or above ``want``, floored at ``min_len`` and
    # capped at the segment: one bucket per exponent, so a 114-order comb needs
    # about ten vectorised blocks instead of 114 of them
    exp2 = np.where(want > 1, np.ceil(np.log2(np.maximum(want, 1))).astype(np.int64), 0)
    lengths = np.minimum(int(n), np.maximum(int(min_len), (1 << exp2).astype(np.int64)))
    return [
        (int(length), np.nonzero(lengths == length)[0] + 1)
        for length in sorted(set(int(v) for v in lengths))
    ]


# ── bench ───────────────────────────────────────────────────────────────────


def _comb_autocovariance(
    grid: BenchGrid,
    *,
    params: V2Params,
    carrier: Tensor,
    k_max: int,
    groups: list[tuple[int, np.ndarray]],
) -> Tensor:
    """``(R, n)`` per-rotor autocovariance of the comb, lag ``0 .. n - 1``."""
    profile = _as_t(params.profile_db, grid.win_acf)
    n_rotors = int(profile.shape[0])
    if int(profile.shape[1]) < k_max:
        raise ValueError(f"profile has {int(profile.shape[1])} orders, need {k_max}")
    out = torch.zeros(n_rotors, grid.n, dtype=grid.win_acf.dtype, device=grid.win_acf.device)
    two_pi = 2.0 * math.pi
    for length, orders in groups:
        k = torch.as_tensor(orders, dtype=out.dtype, device=out.device)  # (Kg,)
        tau = grid.tau_s[:length]
        rho = r_tau(
            tau[None, :],
            k[:, None],
            sigma_nu=params.sigma_nu,
            lam=params.lam,
            sigma_eps=params.sigma_eps,
            lam_eps=params.lam_eps,
            p=params.p,
        )  # (Kg, length)
        power = 10.0 ** (profile[:, orders - 1] / 10.0)  # (R, Kg)
        # the line's angle is k f_r tau: reduced modulo one turn BEFORE the
        # cosine, in float64, so 8 kHz x 30 s (1.5e6 rad) costs 1e-10 rad
        # instead of eating the mantissa.
        ang = torch.remainder(k[None, :, None] * carrier[:, None, None] * tau[None, None, :], 1.0)
        block = (power[:, :, None] * rho[None] * torch.cos(two_pi * ang)).sum(dim=1)  # (R, length)
        if length == grid.n:
            out = out + block
        else:
            out = out + torch.nn.functional.pad(block, (0, grid.n - length))
    return out


def _even_bin_dtft(q: Tensor, n: int) -> Tensor:
    """``sum_{|tau| < n} q(|tau|) e^{-2 pi i j tau / n}`` for ``j = 0 .. n // 2``.

    ``q`` is real and the implied two-sided sequence is even, so the transform
    is real: ``2 Re DFT_{2n}[q][2j] - q(0)``. The length ``2 n`` is the SAME
    anti-aliasing of the lag wrap :mod:`phase_kernel` performs, and bin ``2j``
    of the ``2 n`` grid is exactly bin ``j`` of the ``n`` grid — an exact
    subsample, never an interpolation.
    """
    spec = torch.fft.rfft(q, n=2 * n, dim=-1).real  # (..., n + 1)
    return 2.0 * spec[..., 0 : n + 1 : 2] - q[..., 0:1]


def bench_model(
    grid: BenchGrid,
    params: V2Params,
    *,
    k_max: int,
    groups: list[tuple[int, np.ndarray]] | None = None,
) -> Tensor:
    """``(M, 1, F)`` expected periodogram of one stationary bench support.

    ``F = n // 2 + 1`` and the units are ``data.periodogram``'s
    (``|rfft(w x)|^2 / sum w^2``), i.e. the support's own — there is no frame
    axis to average and no normalisation to undo.

    ``k_max`` is the prefit carrier geometry's Nyquist cap.  A pruned
    ``groups`` grid is also fixed before fitting; without one this function
    computes the exact full-lag transform.  Neither is inferred from a live
    tensor parameter, which would insert a discontinuous NumPy decision into
    the differentiable model.
    """
    carrier = _as_t(params.carrier_rev_s, grid.win_acf).reshape(-1)
    n_rotors = int(carrier.shape[0])
    profile = _as_t(params.profile_db, grid.win_acf)
    if int(profile.shape[0]) != n_rotors:
        raise ValueError(f"{int(profile.shape[0])} profiles for {n_rotors} carriers")
    kk = int(k_max)
    if not 1 <= kk <= int(profile.shape[1]):
        raise ValueError(f"k_max {kk} is outside the profile's 1..{int(profile.shape[1])} orders")
    if groups is None:
        groups = [(grid.n, np.arange(1, kk + 1, dtype=np.int64))]
    c_comb = _comb_autocovariance(grid, params=params, carrier=carrier, k_max=kk, groups=groups)
    m_lines = _even_bin_dtft(grid.win_acf[None, :] * c_comb, grid.n) / grid.window_sumsq  # (R, F)

    c_floor = grid.floor.autocovariance(params.floor)
    q_floor = grid.win_acf[: grid.floor.lag_len] * c_floor
    if grid.floor.lag_len < grid.n:
        q_floor = torch.nn.functional.pad(q_floor, (0, grid.n - grid.floor.lag_len))
    m_floor = _even_bin_dtft(q_floor, grid.n) / grid.window_sumsq  # (F,)

    line_gain = _mean_pinned_db(_as_t(params.mic_line_gain_db, grid.win_acf))  # (M, R)
    floor_gain = 10.0 ** (_as_t(params.floor.mic_floor_db, grid.win_acf) / 10.0)  # (M,)
    all_gain = _mean_pinned_db(_as_t(params.gain_all_db, grid.win_acf).reshape(-1, 1)).reshape(-1)
    observed = (
        torch.einsum("mr,rf->mf", line_gain, m_lines) + floor_gain[:, None] * m_floor[None, :]
    )
    observed = observed * grid.transfer_power[None, :] * all_gain[:, None]
    return observed.clamp_min(MODEL_FLOOR)[:, None, :]


def refine_bench_carrier(
    power: np.ndarray,
    freqs_hz: np.ndarray,
    f0_rev_s: float,
    *,
    sr: int,
    half_width_rev_s: float = 1.0,
    n_grid: int = 4001,
    k_min: int = 4,
    exclude: np.ndarray | None = None,
) -> tuple[float, dict[str, Any]]:
    """Refine ONE constant bench carrier by the harmonic log-power sum.

    The survey/manifest speed is good to about 1 rev/s, which at order 110 is
    110 Hz — 3300 bins of a 30 s periodogram. A Whittle fit started there sees
    no line at all, so the carrier is refined on the support's own periodogram
    BEFORE the fit: score ``sum_k log P(k f)`` on a dense grid of ``f``, then
    one parabolic step. The FITTED carrier still carries the approved
    ``N(survey, 0.5^2)`` prior; this only supplies its initial value.

    ``exclude`` masks bins already claimed by another rotor of a multi-rotor
    point, so the rotors of ``motor_allMotors_70`` cannot collapse onto one
    speed.
    """
    p = np.asarray(power, dtype=np.float64)
    p = p.reshape(-1, p.shape[-1]).mean(axis=0) if p.ndim > 1 else p
    f = np.asarray(freqs_hz, dtype=np.float64)
    lp = np.log(np.maximum(p, 1e-24))
    if exclude is not None:
        lp = np.where(np.asarray(exclude, dtype=bool), np.median(lp), lp)
    df = float(f[1] - f[0])
    f_top = min(BAND_F_MAX, 0.45 * float(sr))

    def score(cand: np.ndarray) -> np.ndarray:
        s = np.zeros_like(cand)
        for k in range(int(k_min), int(math.floor(f_top / max(float(cand.max()), 1e-9))) + 1):
            idx = np.rint(k * cand / df).astype(np.int64)
            ok = (idx > 0) & (idx < lp.size)
            s = s + np.where(ok, lp[np.clip(idx, 0, lp.size - 1)], 0.0)
        return s

    grid = np.linspace(
        max(f0_rev_s - half_width_rev_s, 1e-3), f0_rev_s + half_width_rev_s, int(n_grid)
    )
    sc = score(grid)
    i = int(np.argmax(sc))
    coarse = float(grid[i])
    # a second, fine pass: the harmonic sum is multi-modal on the bin scale, so
    # one parabolic step on the coarse grid is not enough
    step = float(grid[1] - grid[0])
    fine = np.linspace(coarse - step, coarse + step, 401)
    sf = score(fine)
    best = float(fine[int(np.argmax(sf))])
    return best, dict(
        survey_rev_s=float(f0_rev_s),
        coarse_rev_s=coarse,
        refined_rev_s=best,
        shift_rev_s=best - float(f0_rev_s),
        grid_step_rev_s=step,
        fine_step_rev_s=float(fine[1] - fine[0]),
        harmonic_score=float(np.max(sf)),
        k_min=int(k_min),
    )


# ── flight ──────────────────────────────────────────────────────────────────


def flight_rate_work(
    grid: FlightGrid, carrier_rev_s: np.ndarray, frame_starts: np.ndarray
) -> Tensor:
    """``(R, n_c, n_fft_work)`` work-grid carrier samples of the chunk's frames.

    ``carrier_rev_s`` is the label on the AUDIO grid (``(R, n_samples)`` at
    ``grid.sr``); it is resampled onto the declared work grid with the
    renderer's own edge-held ``np.interp`` convention and then windowed at the
    integer frame starts, exactly as
    :meth:`revised_phase._RevisedModel.frame_model` does.
    """
    raw = np.atleast_2d(np.asarray(carrier_rev_s, dtype=np.float64))
    t_src = np.arange(raw.shape[1]) / float(grid.sr)
    t_work = np.arange(raw.shape[1] * grid.oversample) / float(grid.sr_work)
    work = np.stack([np.interp(t_work, t_src, r) for r in raw])
    w = torch.as_tensor(work, dtype=torch.float64, device=grid.window_work.device)
    starts = torch.as_tensor(
        np.asarray(frame_starts, dtype=np.int64) * grid.oversample,
        dtype=torch.int64,
        device=grid.window_work.device,
    )
    idx = starts[:, None] + grid.win_idx_work[None, :]
    if int(idx.max()) >= w.shape[1]:
        raise ValueError(
            f"frame window reaches work sample {int(idx.max())} of {int(w.shape[1])}: the label is "
            "shorter than the last frame"
        )
    return w[:, idx]


def _floor_frames(grid: FlightGrid, rate: Tensor, floor: FloorParams) -> Tensor:
    """``(M, n_c, F)`` broadband floor of a chunk, C4's ``_floor_frames``.

    The floor goes through the SAME finite-window kernel as the lines, with the
    real atom ``w(t) A_floor(t)`` and ``r_tau = 2 R_floor(tau)``, so it carries
    the window response on a coloured spectrum and the within-window variation
    of its envelope exactly.
    """
    ref = grid.window_work
    gain_t = ((rate.clamp_min(SPEED_FLOOR_RPS) / AMP_RPS_REF) ** _as_t(floor.exp, ref)).mean(
        dim=0
    ) + _as_t(floor.static_rel, ref)
    c = grid.floor.autocovariance(floor)
    c0 = c[0].clamp_min(MODEL_FLOOR)
    atom = ref[None, :] * torch.sqrt(c0 * gain_t)
    shape_work = expected_periodogram_from_atoms(
        atom,
        2.0 * c / c0,
        n_fft=grid.n_fft_work,
        window_sumsq=grid.window_work_sumsq,
    )
    f_head = shape_work[..., : grid.n_fft // 2 + 1] * grid.grid_power_factor
    return f_head[None] * 10.0 ** (_as_t(floor.mic_floor_db, ref)[:, None, None] / 10.0)


def flight_model(
    grid: FlightGrid,
    params: V2Params,
    *,
    rate_work: Tensor,
    k_max: int | None = None,
    harmonic_chunk: int | None = 32,
) -> Tensor:
    """``(M, n_c, F)`` expected periodogram of a moving-carrier chunk.

    The math is :meth:`revised_phase._RevisedModel.frame_model`'s with ONE
    substitution: the prior lag law at the kernel call is :func:`.lag.r_tau`
    instead of :func:`revised_phase.prior_r_tau`. Everything else — the local
    within-window carrier integral, the per-window phase centring, the line
    amplitude, the speed law, the shared kernel, the floor through the same
    kernel, the transfer applied once — is C4's, and the differential test
    pins the two to 1e-6 where the laws coincide (``sigma_eps = 0``).
    """
    ref = grid.window_work
    rate = rate_work.to(dtype=ref.dtype, device=ref.device)
    n_rotors, n_c, n_work = rate.shape
    profile = _as_t(params.profile_db, ref)
    kk = int(profile.shape[1]) if k_max is None else int(k_max)
    if kk > int(profile.shape[1]):
        raise ValueError(f"k_max {kk} exceeds the profile's {int(profile.shape[1])} orders")

    phase = (2.0 * math.pi / float(grid.sr_work)) * torch.cumsum(rate, dim=-1)
    phase = phase - phase[..., n_work // 2 : n_work // 2 + 1]
    prof_amp = torch.sqrt(2.0 * 10.0 ** (profile[:, :kk] / 10.0))  # (R, K)
    speed_amp = torch.sqrt(
        (rate.clamp_min(SPEED_FLOOR_RPS) / AMP_RPS_REF) ** _as_t(params.amp_exp, ref)
    )
    env = speed_amp * ref  # (R, n_c, n_work)

    shapes = torch.zeros(n_rotors, n_c, grid.n_fft // 2 + 1, dtype=ref.dtype, device=ref.device)
    step = kk if harmonic_chunk is None else max(1, int(harmonic_chunk))
    for k0 in range(0, kk, step):
        k1 = min(k0 + step, kk)
        k = torch.arange(k0 + 1, k1 + 1, dtype=ref.dtype, device=ref.device)
        kph = torch.remainder(k[None, :, None, None] * phase[:, None], 2.0 * math.pi)
        atoms = torch.polar(prof_amp[:, k0:k1, None, None] * env[:, None], kph)
        rho = r_tau(
            grid.tau_s_work[None, :],
            k[:, None],
            sigma_nu=params.sigma_nu,
            lam=params.lam,
            sigma_eps=params.sigma_eps,
            lam_eps=params.lam_eps,
            p=params.p,
        )[None, :, None, :]
        block = expected_periodogram_from_atoms(
            atoms, rho, n_fft=grid.n_fft_work, window_sumsq=grid.window_work_sumsq
        )
        shapes = shapes + block[..., : grid.n_fft // 2 + 1].sum(dim=1)
    shapes = shapes * grid.grid_power_factor

    line_gain = _mean_pinned_db(_as_t(params.mic_line_gain_db, ref))
    all_gain = _mean_pinned_db(_as_t(params.gain_all_db, ref).reshape(-1, 1)).reshape(-1)
    lines = torch.einsum("mr,rnf->mnf", line_gain, shapes)
    floor = _floor_frames(grid, rate, params.floor)
    observed = (floor + lines) * grid.transfer_power[None, None, :]
    return (observed * all_gain[:, None, None]).clamp_min(MODEL_FLOOR)
