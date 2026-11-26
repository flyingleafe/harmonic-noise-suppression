"""StoRM predictor-corrector sampler — port of ``sgmse/sampling`` of
https://github.com/sp-uhh/storm (commit 257e9636, MIT), conditioned path only.

Differs from :mod:`models.sgmse.sampling` (the newer sgmse code) in one place that
matters numerically: StoRM's reverse-diffusion predictor discretises with
``dt = 1 / N`` (``SDE.discretize``), not with the spacing of the
``linspace(T, eps, N)`` time grid. Per step the annealed-Langevin corrector runs
first, then the predictor; the final one-step-denoised mean is returned.
RNG draws happen in the upstream order (prior, then per step corrector noise
and predictor noise), so a fixed seed reproduces upstream samples.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from models.sgmse.sde import OUVESDE

ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def pc_sample(
    sde: OUVESDE,
    score_fn: ScoreFn,
    y: torch.Tensor,
    *,
    N: int,
    eps: float,
    corrector: str = "ald",
    snr: float = 0.5,
    corrector_steps: int = 1,
) -> torch.Tensor:
    """Run the reverse SDE from ``y + sigma(T) z`` down to ``eps`` and return x_0.

    ``y`` is the steady-state mean of the OUVE process (StoRM: the predictor
    output D(y)), complex ``(B, 1, F, T)``; ``score_fn(x, t)`` closes over the
    score network and its conditioning. ``corrector`` is ``"ald"`` (one or more
    annealed-Langevin steps with step size ``2 (snr * std(t))^2``) or ``"none"``.
    """
    if corrector not in ("ald", "none"):
        raise ValueError(f"corrector must be 'ald' or 'none', got {corrector!r}")
    batch = y.shape[0]
    xt = sde.prior_sampling(y.shape, y)
    timesteps = torch.linspace(sde.T, eps, N, device=y.device)
    dt = 1.0 / N  # SDE.discretize uses 1/N regardless of eps
    sqrt_dt = torch.sqrt(torch.tensor(dt, device=y.device))
    xt_mean = xt
    for i in range(N):
        vec_t = torch.ones(batch, device=y.device) * timesteps[i]
        if corrector == "ald":
            step_size = ((snr * sde._std(vec_t)) ** 2 * 2)[:, None, None, None]
            for _ in range(corrector_steps):
                grad = score_fn(xt, vec_t)
                noise = torch.randn_like(xt)
                xt = xt + step_size * grad + noise * torch.sqrt(step_size * 2)
        # reverse-diffusion predictor (ReverseDiffusionPredictor + RSDE.discretize)
        drift, diffusion = sde.sde(xt, y, vec_t)
        g = (diffusion * sqrt_dt)[:, None, None, None]
        rev_f = drift * dt - g**2 * score_fn(xt, vec_t)
        z = torch.randn_like(xt)
        xt_mean = xt - rev_f
        xt = xt_mean + g * z
    return xt_mean
