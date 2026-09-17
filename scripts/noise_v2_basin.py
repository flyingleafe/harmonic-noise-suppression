#!/usr/bin/env python
"""Basin/ridge diagnosis of the v2 objective in the six dynamics parameters.

    # the Michael's FLY125 cruise pool: both 2-D ridge maps and the profiles,
    # on the SAME deterministic 64-frame set the L-BFGS polish used
    python scripts/noise_v2_basin.py scan \
        --fit results/noise_v2/rounds/round1/fits/michaels_fly125_cruise__flight.json \
        --frames 64 --points 15 --threads 8
    # the converged bench control, every window (one frame, cheap)
    python scripts/noise_v2_basin.py scan \
        --fit results/noise_v2/rounds/round1/fits/bench_dregon_Motor1_80__bench.json \
        --what profiles --points 21
    # the three-panel figure over whatever landed in the output directory
    python scripts/noise_v2_basin.py plot --dir results/noise_v2/rounds/round1/basin \
        --name michaels_fly125_cruise__flight

WHY. Every block but one is held at the FITTED values and only the dynamics
block moves, so what the grid shows is the objective's own geometry in
``(sigma_nu, lam, sigma_eps, lam_eps)`` — not a re-fit.

THE RIDGE COORDINATES, from :func:`experiments.noise_model.lag.r_tau` alone.
The shaft factor of order ``k`` is ``exp(-k^2 V(tau) / 2)`` with

    V(tau) = 2 sigma_nu^2 / lam^2 * (lam|tau| - 1 + exp(-lam|tau|)) ,

so in ``x = lam|tau|``:

* ``x >> 1`` (rate decorrelates inside the window):
  ``V -> 2 (sigma_nu^2 / lam) |tau| - 2 sigma_nu^2 / lam^2``, i.e. Brownian
  phase with diffusion ``D = sigma_nu^2 / lam`` [rad^2/s] and a
  LAG-INDEPENDENT offset ``2 D / lam``. Hold ``D`` fixed and the whole
  lag dependence is fixed; the only residual ``lam`` dependence is the
  coherence gain ``exp(+k^2 D / lam)``, which vanishes as ``lam`` grows.
  ``D_shaft = sigma_nu^2 / lam`` is therefore THE ridge coordinate, and
  ``sigma_nu = sqrt(D lam)`` is how the grid walks along it.
* ``x << 1`` (rate frozen across the window): ``V -> sigma_nu^2 tau^2``,
  Gaussian line of sd ``k sigma_nu / 2pi`` Hz, ``lam`` absent. Here
  ``sigma_nu`` alone is identified.
* The crossover order is ``k* = sqrt(2) lam / sigma_nu``: orders above it sit
  in the frozen branch, orders below it in the diffusive branch.

The per-order factor of order ``k`` is ``exp(-sigma_eps^2 k^p (1 -
exp(-lam_eps|tau|)))``, so

* ``lam_eps |tau| >> 1``: it saturates at ``exp(-sigma_eps^2 k^p)`` — a
  lag-independent coherent fraction. ``A = sigma_eps^2`` is the ridge
  coordinate and ``lam_eps`` is unidentified above ``1 / tau_max``.
* ``lam_eps |tau| << 1``: it becomes ``exp(-sigma_eps^2 lam_eps k^p |tau|)``,
  a per-order diffusion; only the PRODUCT ``D_eps = sigma_eps^2 lam_eps`` is
  identified.

Both grids walk the natural ridge coordinate (``D_shaft``, ``A``) against the
rate, so a ridge shows up as a row of equal objective and the small-rate
diagonal ridge shows up as ``A lam_eps = const``.

Output: one ``noise-v2-basin/1`` JSON per grid, objective in nats per observed
cell of the EVALUATED batch (the same denominator
``experiments.noise_model.fit`` divides its restart gain by, so the 1e-4
convergence tolerance is directly comparable), plus a three-panel PNG.

``PYTHONPATH=src``. The support ``.npz`` caches are gitignored, so a remote job
must run ``scripts/noise_v2_supports.py build`` first, in the same job.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SCHEMA = "noise-v2-basin/1"
OUT_DIR = "results/noise_v2/rounds/round1/basin"
#: same profile width the fit driver uses
K_CAP = 130
#: the six dynamics parameters, in the order the findings table quotes them
DYN_KEYS = ("sigma_nu", "lam", "sigma_eps_even", "sigma_eps_odd", "lam_eps_even", "lam_eps_odd")


# ── the batch the fit saw ───────────────────────────────────────────────────


def _spec_text(name: str) -> str:
    """The spec whose cache stem is ``name`` (the fit JSON records names)."""
    from experiments.noise_model import supports as SU

    for set_name in SU.SUPPORT_SETS:
        for spec in SU.support_set(set_name):
            if spec.name == name:
                return spec.text
    raise SystemExit(f"no support spec has cache name {name!r}")


def _support(name: str) -> Any:
    from experiments.noise_model import supports as SU

    cached = SU.load_cached(name)
    return cached if cached is not None else SU.load_support(_spec_text(name))


def rebuild(fit: dict[str, Any], *, frames: int | None, max_frames: int) -> tuple[Any, Any, dict]:
    """``(batch, params, meta)``: the fit's own batch, optionally frame-sliced.

    The slice is the L-BFGS polish's deterministic frame set
    (``np.linspace`` over the pooled frames), so a per-cell number here and the
    fit's ``lbfgs_*`` per-cell numbers share both frames and denominator.
    """
    import torch

    from experiments.noise_model import model as MD
    from experiments.noise_model import supports as SU

    mode = str(fit["mode"])
    if mode == "bench":
        sup = _support(str(fit["support"]))
        batch = MD.bench_batch(
            name=sup.name,
            power=np.asarray(sup.power, dtype=np.float64),
            sr=int(sup.sr),
            carrier_mean=np.asarray(sup.carrier_rev_s, dtype=np.float64).mean(axis=1),
            k_cap=K_CAP,
        )
    else:
        members = []
        for name in list(fit["supports"]):
            s = SU.load_cached(name) or SU.load_support(_spec_text(name))
            members.append(
                (
                    s.name,
                    np.asarray(s.power, dtype=np.float64),
                    np.asarray(s.carrier_rev_s_audio, dtype=np.float64),
                    np.asarray(s.frame_starts, dtype=np.int64),
                )
            )
        fe = fit["front_end"]
        batch = MD.flight_batch(
            name=str(fit["support"]),
            members=members,
            sr=int(fe["sr"]),
            n_fft=int(fe["n_fft"]),
            hop=int(fe["hop"]),
            k_cap=K_CAP,
            frame_stride=int(fit["diagnostics"]["batch"]["frame_stride"]),
            max_frames=int(max_frames),
        )
    if int(batch.k_max) != int(fit["k_max"]):
        raise SystemExit(f"rebuilt k_max {batch.k_max} != fit's {fit['k_max']}")
    full_cells = int(batch.n_cells)
    if full_cells != int(fit["objective"]["n_cells"]):
        raise SystemExit(f"rebuilt {full_cells} cells != fit's {fit['objective']['n_cells']}")

    n_frames = int(batch.power.shape[1])
    sliced = False
    if frames is not None and batch.mode == "flight" and 0 < int(frames) < n_frames:
        idx = np.unique(np.linspace(0, n_frames - 1, int(frames)).round().astype(np.int64))
        batch = MD.batch_slice(batch, idx)
        sliced = True
    params = MD.params_from_dict(fit["params"])
    torch.set_grad_enabled(False)
    meta = dict(
        support=str(fit["support"]),
        mode=mode,
        supports=list(fit["supports"]),
        n_frames_pooled=n_frames,
        n_frames_evaluated=int(batch.power.shape[1]),
        frame_subset="lbfgs polish linspace" if sliced else "all pooled frames",
        n_cells_evaluated=int(batch.n_cells),
        n_cells_pooled=full_cells,
        k_max=int(batch.k_max),
        n_mics=int(batch.n_mics),
        n_rotors=int(batch.n_rotors),
        front_end=dict(fit["front_end"]),
        dynamics_free=bool("dynamics" in MD.free_blocks(mode)),
    )
    return batch, params, meta


class Objective:
    """The fit's objective at an arbitrary dynamics point, everything else held.

    ``risk`` is the Whittle composite risk :func:`model.whittle_risk` — the fit's
    likelihood term verbatim. ``map`` adds the dynamics block's own log-prior
    (the only prior term that moves here; every other block contributes an
    additive constant, so MAP DIFFERENCES over the grid are exact).
    """

    def __init__(self, batch: Any, params: Any, *, dynamics_free: bool) -> None:
        import torch

        from experiments.noise_model import model as MD

        self._t = lambda v: torch.as_tensor(np.asarray(v, dtype=np.float64), dtype=torch.float64)
        self._MD = MD
        self._torch = torch
        self.batch = batch
        self.params = params
        self.priors = MD.PRIORS
        self.dynamics_free = bool(dynamics_free)
        self.n_cells = int(batch.n_cells)
        self.n_evals = 0
        self.eval_s = 0.0

    def _logprior(self, sigma_nu: float, lam: float, sigma_eps: Any, lam_eps: Any) -> float:
        if not self.dynamics_free:
            return 0.0
        d = self._torch.distributions.LogNormal
        out = 0.0
        for prior, value in (
            (self.priors.log_sigma_nu, sigma_nu),
            (self.priors.log_lam, lam),
            (self.priors.log_sigma_eps, sigma_eps),
            (self.priors.log_lam_eps, lam_eps),
        ):
            dd = d(self._t(prior[0]), float(prior[1]))
            out += float(dd.log_prob(self._t(value)).sum())
        return out

    def at(self, *, sigma_nu: float, lam: float, sigma_eps: Any, lam_eps: Any) -> dict[str, float]:
        t0 = time.time()
        pr = replace(
            self.params,
            sigma_nu=self._t(sigma_nu),
            lam=self._t(lam),
            sigma_eps=self._t(sigma_eps),
            lam_eps=self._t(lam_eps),
        )
        m = self._MD.forward(self.batch, pr)
        risk = float(self._MD.whittle_risk(self.batch, m))
        lp = self._logprior(sigma_nu, lam, sigma_eps, lam_eps)
        self.n_evals += 1
        self.eval_s += time.time() - t0
        return dict(
            risk=risk,
            risk_per_cell=risk / self.n_cells,
            map=risk - lp,
            map_per_cell=(risk - lp) / self.n_cells,
            log_prior_dynamics=lp,
        )

    def fitted_point(self) -> dict[str, float]:
        p = self._MD.params_to_dict(self.params)
        return {k: float(p[k]) for k in DYN_KEYS}


def _point(fitted: dict[str, float], **over: float) -> dict[str, Any]:
    """The four ``r_tau`` arguments at the fitted point with overrides applied."""
    v = dict(fitted, **over)
    return dict(
        sigma_nu=v["sigma_nu"],
        lam=v["lam"],
        sigma_eps=[v["sigma_eps_even"], v["sigma_eps_odd"]],
        lam_eps=[v["lam_eps_even"], v["lam_eps_odd"]],
    )


def _decade_axis(centre: float, *, span: float, n: int) -> np.ndarray:
    return float(centre) * 10.0 ** np.linspace(-float(span), float(span), int(n))


def _ridge_report(axis_ridge: np.ndarray, axis_rate: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    """Flat-vs-curved summary of one map: ``z`` is ``(n_ridge, n_rate)`` per-cell.

    ALONG the ridge means at fixed ridge coordinate, varying the rate; ACROSS
    means at fixed rate, varying the ridge coordinate.
    """
    finite = np.isfinite(z)
    j_best, i_best = np.unravel_index(np.nanargmin(np.where(finite, z, np.inf)), z.shape)
    along = z[j_best, :]
    across = z[:, i_best]
    # the rate band that stays within the convergence tolerance of the best
    # point at the best ridge coordinate: this is the UNIDENTIFIED band
    tol = 1e-4
    near = np.isfinite(along) & (along - np.nanmin(along) < tol)
    band = [float(axis_rate[near].min()), float(axis_rate[near].max())] if near.any() else None
    profiled = np.nanmin(np.where(finite, z, np.inf), axis=0)
    ridge_arg = np.nanargmin(np.where(finite, z, np.inf), axis=0)
    profiled_bands = {}
    for level in (1e-4, 1e-2, 5e-2):
        ok = profiled - np.nanmin(profiled) < level
        profiled_bands[f"within_{level:g}"] = (
            [float(axis_rate[ok].min()), float(axis_rate[ok].max()), int(ok.sum())]
            if ok.any()
            else None
        )
    return dict(
        argmin_ridge=float(axis_ridge[j_best]),
        argmin_rate=float(axis_rate[i_best]),
        min_per_cell=float(z[j_best, i_best]),
        along_ridge_range_per_cell=float(np.nanmax(along) - np.nanmin(along)),
        along_ridge_range_per_cell_rate_ge_1=float(
            np.nanmax(along[axis_rate >= 1.0]) - np.nanmin(along[axis_rate >= 1.0])
        ),
        across_ridge_range_per_cell=float(np.nanmax(across) - np.nanmin(across)),
        along_ridge_per_cell=[float(v) for v in along],
        across_ridge_per_cell=[float(v) for v in across],
        tol_nats_per_cell=tol,
        rate_band_within_tol=band,
        n_rate_points_within_tol=int(near.sum()),
        # PROFILED over the ridge coordinate: the best objective attainable at
        # each rate. This, not the row through the best ridge value, is the
        # identifiability statement about the rate — the ridge of the shaft
        # term is not horizontal in (D, lam), it tracks constant sigma_nu.
        profiled_over_ridge_per_cell=[float(v) for v in profiled],
        profiled_argmin_rate=float(axis_rate[int(np.nanargmin(profiled))]),
        profiled_range_per_cell=float(np.nanmax(profiled) - np.nanmin(profiled)),
        profiled_range_per_cell_rate_ge_1=float(
            np.nanmax(profiled[axis_rate >= 1.0]) - np.nanmin(profiled[axis_rate >= 1.0])
        ),
        profiled_ridge_argmin=[float(axis_ridge[j]) for j in ridge_arg],
        profiled_rate_band=profiled_bands,
    )


def _rate_axis(lo: float, hi: float, *, n: int, fitted: float) -> np.ndarray:
    """``geomspace(lo, hi, n)``, with the FITTED rate inserted if it falls
    outside the mandated span (a grid whose star is off the map says nothing
    about the point the fit actually reached)."""
    axis = np.geomspace(float(lo), float(hi), int(n))
    if not (float(lo) <= float(fitted) <= float(hi)):
        axis = np.unique(np.concatenate([axis, [float(fitted)]]))
    return axis


# ── the three scans ─────────────────────────────────────────────────────────


def shaft_grid(
    obj: Objective, *, points: int, span: float, rate_lo: float, rate_hi: float, log: bool = True
) -> dict[str, Any]:
    """Objective on ``(D_shaft = sigma_nu^2 / lam, lam)``, ``sigma_nu`` derived."""
    fitted = obj.fitted_point()
    d_fit = fitted["sigma_nu"] ** 2 / fitted["lam"]
    d_axis = _decade_axis(d_fit, span=span, n=points)
    lam_axis = _rate_axis(rate_lo, rate_hi, n=points, fitted=fitted["lam"])
    risk = np.full((d_axis.size, lam_axis.size), np.nan)
    mp = np.full_like(risk, np.nan)
    sig = np.full_like(risk, np.nan)
    for j, d in enumerate(d_axis):
        for i, lam in enumerate(lam_axis):
            s = math.sqrt(float(d) * float(lam))
            out = obj.at(**_point(fitted, sigma_nu=s, lam=float(lam)))
            risk[j, i], mp[j, i], sig[j, i] = out["risk_per_cell"], out["map_per_cell"], s
        if log:
            print(f"  shaft row {j + 1}/{d_axis.size} D={d:.6g} {obj.eval_s:.0f}s", flush=True)
    return dict(
        term="shaft",
        ridge_coordinate="D_shaft = sigma_nu^2 / lam  [rad^2/s]",
        rate_parameter="lam [1/s]",
        ridge_axis=[float(v) for v in d_axis],
        rate_axis=[float(v) for v in lam_axis],
        sigma_nu_grid=[[float(v) for v in row] for row in sig],
        risk_per_cell=[[float(v) for v in row] for row in risk],
        map_per_cell=[[float(v) for v in row] for row in mp],
        fitted_ridge=float(d_fit),
        fitted_rate=float(fitted["lam"]),
        crossover_order_at_fit=float(math.sqrt(2.0) * fitted["lam"] / fitted["sigma_nu"]),
        verdict=dict(
            risk=_ridge_report(d_axis, lam_axis, risk), map=_ridge_report(d_axis, lam_axis, mp)
        ),
    )


def order_grid(
    obj: Objective,
    *,
    parity: str,
    points: int,
    span: float,
    rate_lo: float,
    rate_hi: float,
    log: bool = True,
) -> dict[str, Any]:
    """Objective on ``(A = sigma_eps^2, lam_eps)`` of ONE parity."""
    fitted = obj.fitted_point()
    s_key, l_key = f"sigma_eps_{parity}", f"lam_eps_{parity}"
    a_fit = fitted[s_key] ** 2
    a_axis = _decade_axis(a_fit, span=span, n=points)
    rate_axis = _rate_axis(rate_lo, rate_hi, n=points, fitted=fitted[l_key])
    risk = np.full((a_axis.size, rate_axis.size), np.nan)
    mp = np.full_like(risk, np.nan)
    for j, a in enumerate(a_axis):
        for i, rate in enumerate(rate_axis):
            over = {s_key: math.sqrt(float(a)), l_key: float(rate)}
            out = obj.at(**_point(fitted, **over))
            risk[j, i], mp[j, i] = out["risk_per_cell"], out["map_per_cell"]
        if log:
            print(
                f"  order-{parity} row {j + 1}/{a_axis.size} A={a:.6g} {obj.eval_s:.0f}s",
                flush=True,
            )
    return dict(
        term=f"order_{parity}",
        ridge_coordinate=f"A_{parity} = sigma_eps_{parity}^2  [rad^2 at k=1]",
        rate_parameter=f"lam_eps_{parity} [1/s]",
        ridge_axis=[float(v) for v in a_axis],
        rate_axis=[float(v) for v in rate_axis],
        risk_per_cell=[[float(v) for v in row] for row in risk],
        map_per_cell=[[float(v) for v in row] for row in mp],
        fitted_ridge=float(a_fit),
        fitted_rate=float(fitted[l_key]),
        small_rate_product_at_fit=float(a_fit * fitted[l_key]),
        verdict=dict(
            risk=_ridge_report(a_axis, rate_axis, risk), map=_ridge_report(a_axis, rate_axis, mp)
        ),
    )


def profiles(obj: Objective, *, points: int, span: float, log: bool = True) -> dict[str, Any]:
    """1-D profile of the objective along each of the six dynamics parameters."""
    fitted = obj.fitted_point()
    out: dict[str, Any] = {}
    for key in DYN_KEYS:
        axis = _decade_axis(fitted[key], span=span, n=points)
        risk, mp = [], []
        for v in axis:
            r = obj.at(**_point(fitted, **{key: float(v)}))
            risk.append(r["risk_per_cell"])
            mp.append(r["map_per_cell"])
        risk_a, mp_a = np.asarray(risk), np.asarray(mp)
        j = int(np.nanargmin(mp_a))
        out[key] = dict(
            axis=[float(v) for v in axis],
            risk_per_cell=[float(v) for v in risk_a],
            map_per_cell=[float(v) for v in mp_a],
            fitted=float(fitted[key]),
            argmin=float(axis[j]),
            argmin_decades_from_fit=float(np.log10(axis[j] / fitted[key])),
            risk_range_per_cell=float(np.nanmax(risk_a) - np.nanmin(risk_a)),
            map_range_per_cell=float(np.nanmax(mp_a) - np.nanmin(mp_a)),
            # how wide a plateau around the profile minimum stays inside the
            # fit's own convergence tolerance: a parameter the data does not
            # constrain has a plateau many decades wide
            decades_within_tol=float(_plateau_decades(axis, mp_a, tol=1e-4)),
            decades_within_1_nat=float(_plateau_decades(axis, mp_a, tol=1.0)),
        )
        if log:
            print(
                f"  profile {key}: range {out[key]['map_range_per_cell']:.4g} nats/cell", flush=True
            )
    return dict(term="profiles", span_decades=float(span), points=int(points), profiles=out)


def slices(
    obj: Objective,
    *,
    sweep: str,
    at: dict[str, list[float]],
    points: int,
    span: float,
    log: bool = True,
) -> dict[str, Any]:
    """``sweep``'s 1-D profile at each HELD value of another parameter.

    What a 2-D ``(ridge, rate)`` grid cannot answer when its ridge axis is
    only two decades wide: at a rate far from the fit the identified ridge
    coordinate can leave the grid, and the per-rate minimum is then an upper
    bound. Sweeping the ridge coordinate itself at a NAMED rate profiles it
    honestly, which is what comparing two candidate pins needs.
    """
    fitted = obj.fitted_point()
    axis = _decade_axis(fitted[sweep], span=span, n=points)
    out = []
    for key, values in at.items():
        for held in values:
            row_risk, row_map = [], []
            for v in axis:
                r = obj.at(**_point(fitted, **{sweep: float(v), key: float(held)}))
                row_risk.append(r["risk_per_cell"])
                row_map.append(r["map_per_cell"])
            m = np.asarray(row_map)
            j = int(np.nanargmin(m))
            out.append(
                dict(
                    held_parameter=key,
                    held_value=float(held),
                    swept_parameter=sweep,
                    axis=[float(v) for v in axis],
                    risk_per_cell=[float(v) for v in row_risk],
                    map_per_cell=[float(v) for v in row_map],
                    argmin=float(axis[j]),
                    min_per_cell=float(m[j]),
                    at_grid_edge=bool(j in (0, m.size - 1)),
                )
            )
            if log:
                print(
                    f"  slice {key}={held:g}: min {m[j]:.6f} nats/cell at "
                    f"{sweep}={axis[j]:.6g}{' (EDGE)' if out[-1]['at_grid_edge'] else ''}",
                    flush=True,
                )
    best = min(s["min_per_cell"] for s in out)
    for s in out:
        s["excess_over_best_per_cell"] = float(s["min_per_cell"] - best)
    return dict(
        term=f"slices_{sweep}",
        span_decades=float(span),
        points=int(points),
        best_min_per_cell=float(best),
        slices=out,
    )


def _plateau_decades(axis: np.ndarray, z: np.ndarray, *, tol: float) -> float:
    """Width in decades of the CONNECTED region around ``argmin`` within ``tol``."""
    z = np.asarray(z, dtype=np.float64)
    ok = np.isfinite(z) & (z - np.nanmin(z) < float(tol))
    if not ok.any():
        return 0.0
    j = int(np.nanargmin(np.where(np.isfinite(z), z, np.inf)))
    lo = j
    while lo - 1 >= 0 and ok[lo - 1]:
        lo -= 1
    hi = j
    while hi + 1 < ok.size and ok[hi + 1]:
        hi += 1
    return float(np.log10(axis[hi] / axis[lo]))


# ── the figure ──────────────────────────────────────────────────────────────


def _panel_map(ax: Any, grid: dict[str, Any], *, field: str, title: str) -> Any:
    import matplotlib.patheffects as pe
    import numpy as np
    from matplotlib.colors import LogNorm

    z = np.asarray(grid[f"{field}_per_cell"], dtype=np.float64)
    x = np.asarray(grid["rate_axis"], dtype=np.float64)
    y = np.asarray(grid["ridge_axis"], dtype=np.float64)
    excess = z - np.nanmin(z)
    floor = 1e-6
    im = ax.pcolormesh(
        x,
        y,
        np.maximum(excess, floor),
        norm=LogNorm(vmin=floor, vmax=max(np.nanmax(excess), 10.0 * floor)),
        cmap="inferno",
        shading="nearest",
    )
    cs = ax.contour(
        x,
        y,
        np.maximum(excess, floor),
        levels=[1e-4, 1e-2, 1.0],
        colors="deepskyblue",
        linewidths=0.9,
    )
    ax.clabel(cs, fmt=lambda v: f"{v:g}", fontsize=7)
    # the ridge trace: the best ridge coordinate at each rate. Outlined,
    # because over the bright end of `inferno` a plain white dashed line
    # disappears.
    j = np.nanargmin(np.where(np.isfinite(z), z, np.inf), axis=0)
    ax.plot(
        x,
        y[j],
        color="white",
        lw=1.6,
        ls="--",
        label="min over ridge coord",
        path_effects=[pe.Stroke(linewidth=3.0, foreground="black"), pe.Normal()],
    )
    ax.plot(
        [grid["fitted_rate"]],
        [grid["fitted_ridge"]],
        marker="*",
        ms=14,
        color="lime",
        mec="black",
        label="fitted point",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(grid["rate_parameter"])
    ax.set_ylabel(grid["ridge_coordinate"])
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.85)
    return im


def plot(paths: dict[str, Path], out: Path, *, field: str = "map", title: str = "") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), dpi=160)
    for ax, key, name in (
        (axes[0], "grid_shaft", "shaft term: excess objective over grid min"),
        (axes[1], "grid_order_odd", "odd per-order term: excess over grid min"),
    ):
        if key in paths and paths[key].exists():
            grid = json.loads(paths[key].read_text())
            im = _panel_map(ax, grid, field=field, title=f"{name}\n[{field} nats/cell]")
            fig.colorbar(im, ax=ax, label="excess nats/cell")
        else:
            ax.text(0.5, 0.5, f"{key}: not run", ha="center", va="center")
            ax.set_axis_off()

    ax = axes[2]
    if "profiles" in paths and paths["profiles"].exists():
        prof = json.loads(paths["profiles"].read_text())["profiles"]
        for key, style in zip(DYN_KEYS, ("-", "-", "--", "--", ":", ":"), strict=True):
            p = prof[key]
            axis = np.asarray(p["axis"], dtype=np.float64)
            z = np.asarray(p[f"{field}_per_cell"], dtype=np.float64)
            ax.plot(
                np.log10(axis / p["fitted"]),
                np.maximum(z - np.nanmin(z), 1e-7),
                style,
                lw=1.4,
                label=f"{key} (fit {p['fitted']:.4g})",
            )
        ax.axhline(1e-4, color="grey", lw=0.8, ls="-.")
        ax.text(
            0.02,
            1.3e-4,
            "L-BFGS convergence tol 1e-4",
            fontsize=7,
            color="grey",
            transform=ax.get_yaxis_transform(),
        )
        ax.set_yscale("log")
        ax.set_xlabel("decades from the fitted value")
        ax.set_ylabel("excess nats/cell over the profile minimum")
        ax.set_title(f"1-D profiles, all other parameters held\n[{field} nats/cell]", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    else:
        ax.text(0.5, 0.5, "profiles: not run", ha="center", va="center")
        ax.set_axis_off()

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ── CLI ─────────────────────────────────────────────────────────────────────


def _write(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"# wrote {path}", flush=True)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan")
    s.add_argument("--fit", required=True, help="a noise-v2-fit/1 JSON")
    s.add_argument("--out", default=OUT_DIR)
    s.add_argument(
        "--what",
        nargs="*",
        default=["shaft", "order-odd", "profiles"],
        choices=["shaft", "order-odd", "order-even", "profiles", "slices"],
    )
    s.add_argument("--points", type=int, default=15, help="grid points per axis")
    s.add_argument("--profile-points", type=int, default=21)
    s.add_argument("--span", type=float, default=2.0, help="decades either side of the fit")
    s.add_argument("--rate-lo", type=float, default=0.1)
    s.add_argument("--rate-hi", type=float, default=1000.0)
    s.add_argument(
        "--frames",
        type=int,
        default=0,
        help="evaluate on this many pooled FLIGHT frames (the polish set); 0 = all",
    )
    s.add_argument("--max-frames", type=int, default=256)
    s.add_argument("--threads", type=int, default=1)
    s.add_argument("--plot", action="store_true", help="also write the PNG")
    s.add_argument(
        "--sweep",
        default=None,
        help="with --what slices: the parameter to sweep, e.g. sigma_nu",
    )
    s.add_argument(
        "--at",
        default=None,
        metavar="NAME=V1,V2,...",
        help="with --what slices: the parameter HELD at each of these values, "
        "e.g. --at lam=0.5,5.5,268.27",
    )

    p = sub.add_parser("plot")
    p.add_argument("--dir", default=OUT_DIR)
    p.add_argument("--name", required=True, help="fit stem, e.g. michaels_fly125_cruise__flight")
    p.add_argument("--field", default="map", choices=["map", "risk"])
    p.add_argument("--out", default=None)

    r = sub.add_parser(
        "reverdict", help="recompute a grid JSON's verdict block in place (no supports needed)"
    )
    r.add_argument("--grid", nargs="+", required=True, help="grid JSON(s) to update")

    args = ap.parse_args(argv)

    if args.cmd == "plot":
        d = Path(args.dir)
        paths = {
            k: d / f"{args.name}__{k}.json"
            for k in ("grid_shaft", "grid_order_odd", "grid_order_even", "profiles")
        }
        out = Path(args.out) if args.out else d / f"{args.name}__basin.png"
        plot(paths, out, field=args.field, title=f"v2 objective geometry — {args.name}")
        print(f"# wrote {out}", flush=True)
        return 0

    if args.cmd == "reverdict":
        for path in args.grid:
            q = Path(path)
            grid = json.loads(q.read_text())
            ridge = np.asarray(grid["ridge_axis"], dtype=np.float64)
            rate = np.asarray(grid["rate_axis"], dtype=np.float64)
            grid["verdict"] = dict(
                risk=_ridge_report(
                    ridge, rate, np.asarray(grid["risk_per_cell"], dtype=np.float64)
                ),
                map=_ridge_report(ridge, rate, np.asarray(grid["map_per_cell"], dtype=np.float64)),
            )
            _write(grid, q)
        return 0

    import torch

    torch.set_num_threads(max(1, int(args.threads)))
    fit_path = Path(args.fit)
    fit = json.loads(fit_path.read_text())
    stem = fit_path.stem
    out_dir = Path(args.out)

    t0 = time.time()
    batch, params, meta = rebuild(
        fit, frames=(int(args.frames) or None), max_frames=int(args.max_frames)
    )
    obj = Objective(batch, params, dynamics_free=meta["dynamics_free"])
    base = obj.at(**_point(obj.fitted_point()))
    print(
        f"# batch {meta['n_frames_evaluated']} frames, {obj.n_cells} cells, "
        f"build {time.time() - t0:.0f}s, fitted risk/cell {base['risk_per_cell']:.6f}, "
        f"one eval {obj.eval_s:.2f}s",
        flush=True,
    )

    common = dict(
        schema=SCHEMA,
        fit=str(fit_path),
        fit_git=fit.get("git"),
        fit_converged=bool(fit["optimiser"].get("converged", False)),
        fit_restart_gain_per_cell=float(
            fit["optimiser"].get("lbfgs_restart_gain_per_cell", float("nan"))
        ),
        batch=meta,
        fitted_dynamics=obj.fitted_point(),
        fitted_objective=base,
        span_decades=float(args.span),
        rate_range=[float(args.rate_lo), float(args.rate_hi)],
    )
    written: dict[str, Path] = {}
    for what in args.what:
        t1 = time.time()
        if what == "shaft":
            payload = shaft_grid(
                obj,
                points=int(args.points),
                span=float(args.span),
                rate_lo=float(args.rate_lo),
                rate_hi=float(args.rate_hi),
            )
            key = "grid_shaft"
        elif what in ("order-odd", "order-even"):
            parity = what.split("-")[1]
            payload = order_grid(
                obj,
                parity=parity,
                points=int(args.points),
                span=float(args.span),
                rate_lo=float(args.rate_lo),
                rate_hi=float(args.rate_hi),
            )
            key = f"grid_order_{parity}"
        elif what == "slices":
            if not (args.sweep and args.at):
                raise SystemExit("--what slices needs --sweep NAME and --at NAME=V1,V2,...")
            key_at, _, values = str(args.at).partition("=")
            payload = slices(
                obj,
                sweep=str(args.sweep),
                at={key_at: [float(v) for v in values.split(",")]},
                points=int(args.points),
                span=float(args.span),
            )
            key = f"slices_{args.sweep}_at_{key_at}"
        else:
            payload = profiles(obj, points=int(args.profile_points), span=float(args.span))
            key = "profiles"
        written[key] = _write(
            dict(common, **payload, wall_s=time.time() - t1, n_evals=obj.n_evals),
            out_dir / f"{stem}__{key}.json",
        )

    print(f"# {obj.n_evals} objective evaluations, {obj.eval_s:.0f}s in the objective", flush=True)
    if args.plot:
        paths = {
            k: out_dir / f"{stem}__{k}.json"
            for k in ("grid_shaft", "grid_order_odd", "grid_order_even", "profiles")
        }
        plot(paths, out_dir / f"{stem}__basin.png", title=f"v2 objective geometry — {stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
