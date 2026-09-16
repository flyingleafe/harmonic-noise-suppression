"""The EXACT-likelihood MAP fit of the rotor-speed trajectory model.

The MODEL — its 32 parameters, their exact discrete state spaces, the sampler,
the whole-flight envelope and the rig posterior's draw — lives in
:mod:`data_processing.trajectory_model`, because the training streams draw from
it (``rps.kind: fitted_traj``).  This module is the campaign half: the exact
likelihood of one rig's real telemetry, the MAP priors and the optimiser that
together produce a :class:`~data_processing.trajectory_model.NewFit`.

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

The likelihood's state space is built in torch (:func:`_state_space_torch`)
from the same exact discretisation the model package samples from — ``Phi =
expm(A dt)``, ``Q = P - Phi P Phi^T`` — so the two cannot drift apart.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import hann

from data_processing.trajectory_model import (
    F0_MAX_HZ,
    F0_MIN_HZ,
    N_STATES,
    SIGMA_E_FLOOR,
    TAU_E_MAX_S,
    TAU_E_MIN_S,
    TAU_SLOW_MAX_S,
    ZETA_MAX,
    ZETA_MIN,
    NewFit,
    Params,
    f0_from_v,
    tau_e_from_v,
    tau_slow_from_u,
    u_from_tau_slow,
    v_from_f0,
    v_from_tau_e,
    v_from_zeta,
    zeta_from_v,
)
from experiments.rps_traj.data import (
    ANTIALIAS_ORDER,
    ERODE_S,
    RATE_HZ,
    Flight,
    airborne_segments,
)
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

#: The flight count below which the per-flight offset term is switched off
#: entirely: with fewer flights the offset cannot be told from the pooled mean.
S_MIN_FLIGHTS = 3

#: Riccati fixed point: relative tolerance and iteration cap.  The tolerance is
#: RELATIVE because the rigs' variances span 0.4 to 3000 (rev/s)^2, so an
#: absolute 1e-9 would be unreachable on neurobem and trivial on michaels.
RICCATI_TOL = 1e-9
RICCATI_MAX_ITER = 2000

#: The optimiser vector's length.
N_FIT_PARAMS = 23

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
    "FIT_RATE_HZ",
    "MIN_SCORED_S",
    "N_FIT_PARAMS",
    "PAD_QUANTUM",
    "SAMPLE_PAD_S",
    "airborne_extremes",
    "cross_periodogram",
    "decimate",
    "exact_nll",
    "fit_rate_hz",
    "fit_rig",
    "likelihood_batches",
    "pooled_means",
    "steady_state_gain",
]
