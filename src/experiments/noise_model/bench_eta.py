"""The likelihood weight ``eta`` of the v4 bench fit by block jackknife
(``docs/explainers/noise-model-v4.qmd`` § Rebalancing; round-5 step 3).

The Whittle likelihood treats every (microphone, frequency) cell as independent.
The cells are not: the eight microphones hear one source and a line's pedestal
and the analysis window spread it over neighbouring bins. Dependence and model
error both make the per-cell scores co-vary, so the curvature the fit uses
(``H``) understates the true variance of the score (``J``). The sandwich
comparison measures this:

* ``g_b`` the Whittle score (gradient of the negative log-likelihood) of block
  ``b``: one recording's cells in one frequency band, all microphones;
* ``J = B/(B-1) sum_b (g_b - mean g)(g_b - mean g)^T`` the block covariance;
* ``H`` the Hessian of the Whittle term, ``P`` the prior's, ``H_post = H + P``;
* ``p_eff = tr(H_post^-1 H)`` the likelihood-identified dimension and
  ``eta = p_eff / tr(H_post^-1 J)``: 1 for independent, correctly modelled
  cells, below 1 when the likelihood is overconfident.

On a planted twin (data rendered from the model) ``eta`` measures the
dependence alone; the gap to the real-data ``eta`` is model error. Growing the
blocks until ``eta`` stops moving is the block-size check.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor

from experiments.noise_model.bench_v4 import RigModel

__all__ = ["block_scores", "eta_report", "hessians", "free_params"]


def free_params(model: RigModel) -> list[tuple[str, Tensor]]:
    """The fitted parameters in a fixed order (pinned ones excluded)."""
    return [(n, t) for n, t in model.params.items() if n not in model.pinned]


def _flat(grads: tuple[Tensor | None, ...], params: list[Tensor]) -> Tensor:
    return torch.cat(
        [(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads, params)]
    )


def block_scores(model: RigModel, band_hz: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """``(B, p)`` per-block Whittle scores and the block index: per recording,
    contiguous ``band_hz`` frequency bands over every microphone."""
    params = [t for _, t in free_params(model)]
    rows: list[np.ndarray] = []
    index: list[dict[str, Any]] = []
    for j, fz in enumerate(model.frozen):
        f = fz.grid.freqs_hz[np.asarray(fz.band.cpu())]
        edges = np.arange(f[0], f[-1] + band_hz, band_hz)
        which = np.clip(np.searchsorted(edges, f, side="right") - 1, 0, edges.size - 2)
        m = model.expected(j)[:, fz.band]
        cell = torch.log(m) + fz.power[:, fz.band] / m  # (M, Fb)
        per_band = cell.sum(dim=0)
        for b in np.unique(which):
            sel = torch.as_tensor(np.flatnonzero(which == b), device=model.device)
            g = torch.autograd.grad(
                per_band[sel].sum(), params, retain_graph=True, allow_unused=True
            )
            rows.append(_flat(g, params).cpu().numpy())
            index.append(
                dict(recording=j, band=int(b), f_lo=float(edges[b]), n_bins=int(sel.numel()))
            )
        del m, cell, per_band
    return np.stack(rows), index


def _hessian_rows(f: Tensor, params: list[Tensor], p: int) -> np.ndarray:
    """``(p, p)`` the Hessian of the scalar ``f``: one row per second backward."""
    g = _flat(torch.autograd.grad(f, params, create_graph=True, allow_unused=True), params)
    out = np.zeros((p, p))
    for i in range(p):
        if g[i].requires_grad:
            row = torch.autograd.grad(g[i], params, retain_graph=True, allow_unused=True)
            out[i] = _flat(row, params).cpu().numpy()
    return out


def hessians(model: RigModel, *, log: Any = print) -> tuple[np.ndarray, np.ndarray]:
    """``(H, P)``: the Hessians of the Whittle term and of the negative log
    prior in the free parameters (:func:`free_params` order), one recording's
    Whittle term at a time so one recording's double graph is alive."""
    params = [t for _, t in free_params(model)]
    p = int(sum(t.numel() for t in params))
    P = _hessian_rows(model.neg_log_prior(), params, p)
    H = np.zeros((p, p))
    t0 = time.perf_counter()
    for j in range(len(model.frozen)):
        H += _hessian_rows(model.whittle(j), params, p)
        log(
            f"[eta] Hessian: recording {j + 1}/{len(model.frozen)} ({time.perf_counter() - t0:.0f} s)"
        )
    return 0.5 * (H + H.T), 0.5 * (P + P.T)


def eta_report(
    model: RigModel,
    *,
    band_hz: float = 200.0,
    coarsen: tuple[int, ...] = (1, 2, 4, 8),
    log: Any = print,
) -> dict[str, Any]:
    """``eta`` at ``band_hz`` blocks and at blocks ``coarsen`` times wider (the
    block-size check), plus one block per recording."""
    names = [(n, t.numel()) for n, t in free_params(model)]
    t0 = time.perf_counter()
    G, index = block_scores(model, band_hz)
    log(f"[eta] {G.shape[0]} block scores ({time.perf_counter() - t0:.0f} s)")
    H, P = hessians(model, log=log)
    Hp = H + P
    # the posterior curvature is positive definite at the MAP; a tiny ridge
    # guards round-off in directions the prior barely holds
    w, V = np.linalg.eigh(Hp)
    floor = max(float(w.max()) * 1e-12, 1e-12)
    inv = (V / np.maximum(w, floor)) @ V.T
    p_eff = float(np.trace(inv @ H))
    rec = np.asarray([b["recording"] for b in index])
    band = np.asarray([b["band"] for b in index])

    def eta_of(groups: np.ndarray) -> dict[str, Any]:
        keys = np.unique(groups)
        S = np.stack([G[groups == k].sum(axis=0) for k in keys])
        B = S.shape[0]
        C = S - S.mean(axis=0, keepdims=True)
        J = (B / max(B - 1, 1)) * (C.T @ C)
        tr = float(np.trace(inv @ J))
        return dict(n_blocks=int(B), tr_Hinv_J=tr, eta=p_eff / tr if tr > 0 else math.nan)

    by_size = {}
    for c in coarsen:
        by_size[f"{band_hz * c:g}Hz"] = eta_of(rec * 100000 + band // c)
    by_size["recording"] = eta_of(rec)
    return dict(
        band_hz=band_hz,
        n_params=int(sum(s for _, s in names)),
        params=[dict(name=n, size=s) for n, s in names],
        p_eff=p_eff,
        hessian_min_eig=float(w.min()),
        by_block=by_size,
        wall_s=time.perf_counter() - t0,
    )
