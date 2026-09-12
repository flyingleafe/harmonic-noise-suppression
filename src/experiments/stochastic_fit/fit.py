"""Fit one clip: initialization, the harmonic ladder, references.

The fit is a MAP under the Whittle likelihood with the GP knots' whitened
priors; nothing else is regularized. Stages:

1. ``floor`` — lines off. Its optimum is also the *floor-only reference*: the
   likelihood a model with no comb at all reaches, i.e. the bottom of the scale.
2. ``ladder`` — lines on up to order 16, then 48, then all, so the carrier
   corrections (when fitted) and the widths lock on the well-resolved low
   orders before the crowded high orders pull on them.
3. ``polish`` — everything, Adam then L-BFGS.

The top of the scale is the leave-one-out periodogram smoother: the mean of the
four time neighbours of a cell (same bin, same microphone) predicts that cell.
For a locally stationary process that is an unbiased estimate of the cell's
expectation, so its Whittle score is what a *correct* model would reach, up to
the analytic bias of dividing by a four-sample mean (``n/(n-1)`` on the ratio,
``psi(n) - log n`` on the log), which is subtracted. On a clip whose lines move
by more than a bin over two hops (a fast ramp) the neighbours are not
replicates and the reference is optimistic; ``loo_motion_bins`` reports how far
the fastest in-band line moves so such clips can be read with that caveat.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from scipy.special import digamma

from .data import Periodogram
from .model import AMP_RPS_REF, CombSpectrum, Spec

#: Neighbour frame offsets of the leave-one-out smoother. At a quarter-window
#: hop the ADJACENT frame's periodogram is correlated with the cell's (power
#: correlation ~0.43 for Hann), which makes a +-1 smoother optimistic by a
#: constant ~0.25 nats/cell (measured on renderer controls). Two hops apart
#: the window overlap is half and the power correlation ~0.03, so the
#: replicates are taken at +-2 and +-4 hops: four near-independent samples
#: over a +-128 ms span.
LOO_OFFSETS = (-4, -2, 2, 4)
LOO_HALF = 4  # frames excluded at each clip edge


def loo_offsets_for(n_frames: int) -> tuple[int, ...]:
    """The widest neighbourhood a clip of ``n_frames`` can afford.

    The leave-one-out reference predicts a frame from its neighbours, so it
    cannot score frames within ``max|offset|`` of an edge. With NON-overlapping
    analysis frames a 9 s bench span holds eight frames, and the fixed
    four-frame neighbourhood left nothing to score: every ``excess_over_loo``
    came back ``nan``. The neighbourhood now shrinks to keep at least two
    scorable frames.
    """
    half = max(2, min(LOO_HALF, (n_frames - 2) // 2))
    return tuple(s for s in LOO_OFFSETS if abs(s) <= half)


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(x))


def loo_reference(
    power: np.ndarray, band: np.ndarray, offsets: tuple[int, ...] = LOO_OFFSETS
) -> tuple[float, np.ndarray]:
    """Bias-corrected leave-one-out Whittle score per cell and the smoother.

    Returns ``(nll_per_cell, m_hat (M, N, F))``; frames within ``max|offset|``
    of the clip edges are excluded from the score (one-sided neighbourhoods).
    """
    m, n, f = power.shape
    half = max(abs(s) for s in offsets)
    acc = np.zeros_like(power, dtype=np.float64)
    for s in offsets:
        acc[:, max(0, -s) : n - max(0, s)] += power[:, max(0, s) : n - max(0, -s)]
    m_hat = acc / len(offsets)
    inner = slice(half, n - half)
    p = power[:, inner][..., band].astype(np.float64)
    mh = np.maximum(m_hat[:, inner][..., band], 1e-30)
    nll = float(np.mean(p / mh + np.log(mh)))
    n_nb = len(offsets)
    bias = n_nb / (n_nb - 1.0) - 1.0 + (float(digamma(n_nb)) - math.log(n_nb))
    return nll - bias, m_hat


def initialize(model: CombSpectrum, power: torch.Tensor) -> None:
    """Crude but safe starting point: floor from a low percentile, profiles from
    the periodogram at the line centres, widths at the family's midpoint."""
    s = model.spec
    with torch.no_grad():
        db = 10.0 * torch.log10(power.clamp_min(1e-20))  # (M, N, F)
        floor_db = (
            torch.quantile(db.reshape(-1, model.F), 0.2, dim=0) + 6.5
        )  # exp. 20th pct = 0.223 M
        band = model.band
        model.floor_mean_db.fill_(float(floor_db[band].median()))
        model.floor_shape_z.zero_()
        model.floor_tilt_db_oct.zero_()
        model.floor_level_z.zero_()
        model.floor_tilt_z.zero_()
        model.mic_floor_db.zero_()
        model.mic_gain_db.zero_()
        model.h_z.zero_()
        model.rps_offset_knots.zero_()
        model.gamma0_raw.fill_(_inv_softplus(2.0))
        model.slope_raw.fill_(_inv_softplus(0.3))
        gamma = model.gamma  # (R, K)
        model.log_gamma_free.copy_(torch.log(gamma))
        # line level: median periodogram at the centre bin, over mics and frames
        carrier = model.carrier()  # (R, N)
        speed = (model.rps.clamp_min(0.0) / AMP_RPS_REF) ** s.amp_rps_exponent  # (R, N)
        for r in range(model.R):
            centres = model.k[:, None] * carrier[r][None, :]  # (K, N)
            idx = torch.clamp(torch.round(centres / model.df).long(), 0, model.F - 1)  # (K, N)
            at = db[:, torch.arange(model.N, device=db.device)[None, :], idx]  # (M, K, N)
            level = at.median(dim=0).values.median(dim=1).values  # (K,)
            fl = floor_db[idx].median(dim=1).values  # (K,)
            excess = 10.0 ** (level / 10.0) - 10.0 ** (fl / 10.0)
            excess = excess.clamp_min(10.0 ** ((fl - 10.0) / 10.0))
            peak_dens = (
                torch.minimum(1.0 / (math.pi * gamma[r]), torch.full_like(gamma[r], 1.0 / model.df))
                * 2.0
                / 3.0
            )
            amp = speed[r].mean().clamp_min(1e-6)
            model.profile_db[r] = 10.0 * torch.log10(excess / peak_dens / amp)


def _run_adam(
    model: CombSpectrum, power: torch.Tensor, params: list, iters: int, lr: float
) -> float:
    opt = torch.optim.Adam(params, lr=lr)
    last = float("nan")
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        loss = model.whittle(power) + model.prior()
        loss.backward()
        opt.step()
        last = float(loss.item())
    return last


def _run_lbfgs(model: CombSpectrum, power: torch.Tensor, params: list, iters: int) -> float:
    opt = torch.optim.LBFGS(
        params,
        lr=1.0,
        max_iter=iters,
        history_size=20,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
    )

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = model.whittle(power) + model.prior()
        loss.backward()
        return loss

    return float(opt.step(closure).item())


def fit_clip(
    pg: Periodogram,
    spec: Spec,
    *,
    device: str = "cpu",
    ladder: tuple[int, ...] = (16, 48),
    iters: tuple[int, int, int, int] = (
        150,
        150,
        300,
        60,
    ),  # floor adam, ladder adam per rung, final adam, lbfgs
    lr: float = 0.1,
    log: Any = print,
) -> dict[str, Any]:
    """Fit ``spec`` to the periodogram; returns scores, parameters and the
    fitted spectrum. Scores are Whittle NLL per band cell in nats, with the
    periodogram scaled to unit mean (a constant shift, the same for every
    model and reference of the clip)."""
    t0 = time.time()
    scale = float(np.mean(pg.power))
    power_np = pg.power.astype(np.float32) / scale
    model = CombSpectrum(spec, device=device)
    power = torch.as_tensor(power_np, device=device)
    band_np = model.band.cpu().numpy()
    n_cells = model.n_cells()
    initialize(model, power)

    # stage 1: floor only
    model.active_k = 0
    floor_params = model.parameter_groups("floor")
    _run_adam(model, power, floor_params, iters[0], lr)
    _run_lbfgs(model, power, floor_params, 30)
    with torch.no_grad():
        nll_floor = float(model.whittle(power).item()) / n_cells
    log(f"  floor-only: {nll_floor:.4f} nats/cell  ({time.time() - t0:.0f}s)")

    # stage 2: harmonic ladder
    all_params = model.parameter_groups("all")
    for rung in (*ladder, model.K):
        model.active_k = int(min(rung, model.K))
        loss = _run_adam(model, power, all_params, iters[1] if rung < model.K else iters[2], lr)
        log(f"  k<={model.active_k}: {loss / n_cells:.4f}  ({time.time() - t0:.0f}s)")
    model.active_k = model.K
    _run_lbfgs(model, power, all_params, iters[3])
    with torch.no_grad():
        spectrum = model.forward()
        nll_fit = float(model.whittle(power, spectrum).item()) / n_cells
        prior = float(model.prior().item())
    log(f"  fit: {nll_fit:.4f} nats/cell, prior {prior:.1f}  ({time.time() - t0:.0f}s)")

    nll_loo, m_hat = loo_reference(power_np, band_np)
    inner = slice(LOO_HALF, model.N - LOO_HALF)
    with torch.no_grad():
        cell = (power / spectrum.clamp_min(1e-12) + torch.log(spectrum.clamp_min(1e-12)))[:, inner][
            ..., model.band
        ]
        nll_fit_inner = float(cell.mean().item())
    # how far the fastest in-band line moves over the LOO neighbourhood, in bins
    carrier = model.carrier().detach().cpu().numpy()
    k_top = np.minimum(
        model.K, np.floor((spec.f_max or pg.freqs[-1]) / np.maximum(carrier.max(axis=1), 1e-3))
    )
    motion = np.abs(np.diff(carrier, axis=1)).max(axis=1) * k_top * LOO_HALF / pg.df
    explained = (nll_floor - nll_fit) / max(nll_floor - nll_loo, 1e-9)
    out = dict(
        spec=asdict(spec) | dict(freqs=None, times=None, rps=None),
        scores=dict(
            nll_floor_only=nll_floor,
            nll_fit=nll_fit,
            nll_fit_inner=nll_fit_inner,
            nll_loo=nll_loo,
            excess_over_loo=nll_fit_inner - nll_loo,
            explained_fraction=explained,
            prior=prior,
            n_cells=n_cells,
            power_scale=scale,
            loo_motion_bins=float(motion.max()),
            seconds=time.time() - t0,
        ),
        params=model.export(),
        spectrum=(spectrum.cpu().numpy() * scale).astype(np.float32),
        loo_smoother=(m_hat * scale).astype(np.float32),
    )
    return out
