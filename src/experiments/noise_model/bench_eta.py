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

``eta`` needs the fit at a true minimum: ``H_post`` without materially
negative eigenvalues. A converged fit can stop on a flat saddle (a near-
invisible line's level or pedestal); :func:`saddle_escape` walks down each
negative direction on the MAP objective, re-runs the convergence loop and
re-checks the Hessian.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from experiments.noise_model.bench_v4 import RigModel

__all__ = [
    "block_scores",
    "curvature",
    "eta_report",
    "free_params",
    "hessians",
    "line_search",
    "saddle_escape",
]


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


def curvature(H: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """``(w, V, tol)``: the eigenpairs of ``H + P`` (ascending) and the
    tolerance of the rule: ``w < -tol`` is materially negative curvature,
    ``|w| <= tol`` a flat direction."""
    w, V = np.linalg.eigh(H + P)
    return w, V, float(np.abs(w).max()) * 1e-10


def eta_report(
    model: RigModel,
    *,
    band_hz: float = 200.0,
    coarsen: tuple[int, ...] = (1, 2, 4, 8),
    log: Any = print,
    save: Path | None = None,
    hp: tuple[np.ndarray, np.ndarray] | None = None,
    descent_free: bool = False,
) -> dict[str, Any]:
    """``eta`` at ``band_hz`` blocks and at blocks ``coarsen`` times wider (the
    block-size check), plus one block per recording; ``save`` writes ``H``,
    ``P`` and the block scores (``.npz``) for re-analysis; ``hp`` is ``(H, P)``
    already computed at the model's parameters (:func:`hessians`).
    ``descent_free``: :func:`saddle_escape` found no negative direction that
    lowers the objective at these parameters (:data:`MINIMUM_RULE`), so eta is
    reported with them left out like flat directions."""
    names = [(n, t.numel()) for n, t in free_params(model)]
    t0 = time.perf_counter()
    G, index = block_scores(model, band_hz)
    log(f"[eta] {G.shape[0]} block scores ({time.perf_counter() - t0:.0f} s)")
    H, P = hessians(model, log=log) if hp is None else hp
    # at a MAP the posterior curvature is positive definite. Directions of
    # non-positive curvature are counted and left out (the inverse lives on the
    # positive subspace); eta is not reported if any is materially negative,
    # unless the escape found that none of them lowers the objective
    w, V, tol = curvature(H, P)
    keep = w > tol
    n_neg = int((w < -tol).sum())
    inv = (V[:, keep] / w[keep]) @ V[:, keep].T
    p_eff = float(np.trace(inv @ H))
    at_minimum = n_neg == 0 or descent_free
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save,
            H=H,
            P=P,
            G=G,
            recording=np.asarray([b["recording"] for b in index]),
            band=np.asarray([b["band"] for b in index]),
            names=np.asarray([n for n, _ in names]),
            sizes=np.asarray([s for _, s in names]),
        )
    rec = np.asarray([b["recording"] for b in index])
    band = np.asarray([b["band"] for b in index])

    def eta_of(groups: np.ndarray) -> dict[str, Any]:
        keys = np.unique(groups)
        S = np.stack([G[groups == k].sum(axis=0) for k in keys])
        B = S.shape[0]
        C = S - S.mean(axis=0, keepdims=True)
        J = (B / max(B - 1, 1)) * (C.T @ C)
        tr = float(np.trace(inv @ J))
        eta = p_eff / tr if tr > 0 and at_minimum else math.nan
        return dict(n_blocks=int(B), tr_Hinv_J=tr, eta=eta)

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
        n_negative_curvature=n_neg,
        negative_without_descent=bool(descent_free and n_neg > 0),
        minimum_rule=MINIMUM_RULE,
        at_minimum=at_minimum,
        n_dropped_directions=int((~keep).sum()),
        by_block=by_size,
        wall_s=time.perf_counter() - t0,
    )


#: the trial steps of the line search along a negative-curvature direction
#: (unit eigenvector, free coordinates), tried in both signs
ESCAPE_STEPS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
#: When a payload counts as a minimum (``at_minimum``). ``curvature``'s tolerance
#: (1e-10 of the largest eigenvalue) also flags curvature of a fraction of a nat
#: per unit step on coordinates of near-invisible orders: on the DREGON bench fit
#: (c6) the last one has eigenvalue -0.275, a quadratic gain of 0.034 nats at the
#: smallest step, and no trial step lowers the objective at all.
MINIMUM_RULE = (
    "no materially negative curvature, or none of those directions lowers the MAP "
    "objective at any trial step (ESCAPE_STEPS, both signs); those directions are "
    "then left out of H^-1 like flat ones"
)


def get_flat(model: RigModel) -> Tensor:
    """The flat free-parameter vector (:func:`free_params` order), detached."""
    return torch.cat([t.detach().reshape(-1) for _, t in free_params(model)]).clone()


@torch.no_grad()
def set_flat(model: RigModel, x: Tensor) -> None:
    """Write the flat free-parameter vector ``x`` into the model in place."""
    o = 0
    for _, t in free_params(model):
        n = t.numel()
        t.copy_(x[o : o + n].view_as(t))
        o += n


def _coordinate(model: RigModel, v: np.ndarray) -> dict[str, Any]:
    """The free coordinate carrying most of the unit vector ``v``."""
    i = int(np.argmax(v**2))
    o = 0
    for name, t in free_params(model):
        if i < o + t.numel():
            return dict(param=name, index=i - o, mass=float(v[i] ** 2))
        o += t.numel()
    raise IndexError(i)


def line_search(
    model: RigModel, v: np.ndarray, f0: float, steps: tuple[float, ...] = ESCAPE_STEPS
) -> dict[str, Any]:
    """Move the model to the best of ``x + s t v`` (``s = +-1``, ``t`` in
    ``steps``) on the MAP objective (:meth:`RigModel.objective`), or leave it
    at ``x`` if none is below ``f0`` (its objective there). Each sign walks
    out until the objective stops falling."""
    x0 = get_flat(model)
    d = torch.as_tensor(v, dtype=x0.dtype, device=x0.device)
    best_s, best_f, n_evals = 0.0, f0, 0
    for sign in (1.0, -1.0):
        prev = f0
        for t in steps:
            set_flat(model, x0 + (sign * t) * d)
            f, _ = model.objective()
            n_evals += 1
            if f < best_f:
                best_s, best_f = sign * t, f
            if not f < prev:  # stopped falling (or not finite)
                break
            prev = f
    set_flat(model, x0 + best_s * d)
    return dict(step=best_s, gain=f0 - best_f, objective=best_f, n_evals=n_evals)


def saddle_escape(
    model: RigModel,
    converge: Callable[[RigModel, float | None], dict[str, Any]],
    *,
    max_rounds: int = 4,
    steps: tuple[float, ...] = ESCAPE_STEPS,
    deadline: float | None = None,
    hp: tuple[np.ndarray, np.ndarray] | None = None,
    hessian_s: float | None = None,
    log: Any = print,
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray] | None]:
    """Leave the saddle a converged fit can stop on, in place on ``model``.

    A round: the Hessian ``H + P`` (:func:`hessians`); each materially
    negative direction (:func:`curvature`'s rule), most negative first, by
    :func:`line_search` on the MAP objective; then ``converge(model,
    budget_s)`` (the fit's convergence loop, returning its summary) so the
    other coordinates re-balance; the Hessian again. Stops at a minimum (no
    materially negative curvature), after ``max_rounds`` rounds, when no
    negative direction lowers the objective at any trial step (``stuck``: a
    minimum by :data:`MINIMUM_RULE`), or when the next Hessian would end past
    ``deadline`` (a ``time.perf_counter()`` instant).

    ``hp`` is ``(H, P)`` already computed at the model's parameters (a previous
    run's last Hessian) and ``hessian_s`` its wall time; the first round then
    starts at the line search.

    Returns the record and the last ``(H, P)`` if it is at the model's final
    parameters (``None`` if the deadline stopped the run before it)."""
    t_all = time.perf_counter()

    def left() -> float:
        return math.inf if deadline is None else deadline - time.perf_counter()

    if hp is None:
        t0 = time.perf_counter()
        hp = hessians(model, log=log)
        hessian_s = time.perf_counter() - t0
    elif hessian_s is None:
        raise ValueError("a precomputed Hessian needs its wall time (hessian_s)")
    hess_s = float(hessian_s)
    rounds: list[dict[str, Any]] = []
    check: dict[str, Any] = {}
    while True:
        assert hp is not None
        w, V, tol = curvature(*hp)
        neg = np.flatnonzero(w < -tol)
        check = dict(n_negative=int(neg.size), min_eig=float(w.min()), tol=tol, hessian_s=hess_s)
        log(f"[escape] round {len(rounds)}: {neg.size} negative, min eig {w.min():.4g}")
        if neg.size == 0:
            stop = "at_minimum"
            break
        if len(rounds) >= max_rounds:
            stop = "max_rounds"
            break
        f_start, _ = model.objective()
        f, directions = f_start, []
        for i in neg:  # ascending: most negative first
            if left() < 1.25 * hess_s:
                break
            ls = line_search(model, V[:, i], f, steps)
            f = ls["objective"]
            directions.append(dict(eig=float(w[i]), **_coordinate(model, V[:, i]), **ls))
        n_moved = sum(d["step"] != 0.0 for d in directions)
        log(
            f"[escape] {n_moved}/{neg.size} directions moved, "
            f"gain {f_start - f:.4g} nats ({len(directions)} searched)"
        )
        if n_moved == 0:
            rounds.append(dict(check, objective_start=f_start, directions=directions))
            # every negative direction searched, none lowers the objective at any
            # trial step: a minimum to the objective's resolution; fewer searched
            # means the deadline cut the round
            stop = "stuck" if len(directions) == neg.size else "deadline"
            break
        budget = None if deadline is None else left() - 1.25 * hess_s
        conv = converge(model, budget)
        rounds.append(
            dict(
                check,
                objective_start=f_start,
                objective_escaped=f,
                directions=directions,
                convergence=conv,
            )
        )
        if left() < 1.15 * hess_s:
            hp, stop = None, "deadline"
            break
        t0 = time.perf_counter()
        hp = hessians(model, log=log)
        hess_s = time.perf_counter() - t0
    final = check if hp is not None else dict(n_negative=None, min_eig=None)
    record = dict(
        rounds=rounds,
        max_rounds=int(max_rounds),
        steps=list(steps),
        stop=stop,
        n_negative=final["n_negative"],
        min_eig=final["min_eig"],
        at_minimum=None if hp is None else (final["n_negative"] == 0 or stop == "stuck"),
        minimum_rule=MINIMUM_RULE,
        hessian_s=hess_s,
        wall_s=time.perf_counter() - t_all,
    )
    return record, hp
