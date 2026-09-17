"""MAP fit of the v2 model and the ``noise-v2-fit/1`` JSON it writes.

ONE objective and ONE optimiser pair, per the round plan: an ``AutoDelta``
guide over the free blocks of :mod:`.model`, driven by ``pyro.optim.Adam``
(lr 0.02, 1500 steps by default) and then polished with ``torch.optim.LBFGS``
under a strong-Wolfe line search. Pyro 1.9.1 ships no ``PyroLBFGS``, so the
polish is torch's L-BFGS on the guide's own unconstrained parameters against
``Trace_ELBO().differentiable_loss`` — the same objective the Adam phase
minimised, to the last term. WHICH phase converged, how long each took and
both objectives are recorded in the JSON's ``optimiser`` block; nothing is
tuned against a result.

WHY BOTH. Adam gets the levels and the carrier into the right basin from a
cold, weakly-informed start; L-BFGS is what actually resolves a comb's
per-order profile, where the curvature across 130 orders spans decades. A fit
that only ran Adam is marked ``converged: false`` unless the polish agrees.

INITIALISATION is not a detail on this model. ``AutoDelta``'s default
``init_to_median`` would start every order of the profile at the prior mean
(-45 dB) and the floor at -70 dB, tens of dB from a real support, and a Whittle
objective whose model is 40 dB under the data has a gradient dominated by
``I / M``. :func:`initial_values` therefore seeds

* the floor level from the band median of the observed 20th-percentile dB
  curve plus C4's +6.5 dB exponential-percentile correction,
* each order of the profile from the observed peak excess over that floor,
  measured against a UNIT-profile forward pass so the seed is in the model's
  own units (window response, lag law and transfer included),
* the per-microphone broadband gain from each microphone's band-mean level,
* the bench carrier from :func:`.spectrum.refine_bench_carrier` — the survey
  speed is good to ~1 rev/s, which is 3300 bins of a 30 s periodogram at order
  110, while the FITTED carrier keeps the approved ``N(survey, 0.5^2)`` prior.

The dynamics start at their prior medians, which is the point of having
measured them.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDelta, init_to_value
from pyro.optim.optim import PyroOptim
from torch import Tensor

from experiments.stochastic_fit.model import FLOOR_SHAPE_N_CTRL
from experiments.stochastic_fit.revised_phase import (
    FLOOR_INIT_DB_CORRECTION,
    FLOOR_INIT_QUANTILE,
    _git_head,
)

from . import FIT_SCHEMA
from . import model as MD
from . import spectrum as SP
from .lag import saturated_coherence

__all__ = [
    "FitOutcome",
    "OptimSpec",
    "fit_support",
    "initial_values",
    "write_fit",
]


@dataclass(frozen=True)
class OptimSpec:
    """The recorded optimiser schedule. Defaults are the round-plan ones."""

    adam_steps: int = 1500
    adam_lr: float = 0.02
    #: frames per Adam step for a FLIGHT batch (``None`` = every frame). One
    #: Michael's cruise pool is several hundred 2048-point frames on a 64 kHz
    #: work grid; the objective is a sum over frames, so a minibatch gradient is
    #: an unbiased estimate of the full one and the L-BFGS polish is what runs
    #: on the deterministic frame set.
    adam_batch: int | None = 8
    lbfgs_iters: int = 200
    lbfgs_frames: int | None = 64
    lbfgs_history: int = 10
    seed: int = 0
    #: objective gain per observed cell an L-BFGS RESTART may still find and
    #: the fit still count as converged (nats/cell)
    tol_nats_per_cell: float = 1e-4

    def as_dict(self) -> dict[str, Any]:
        return dict(
            adam_steps=self.adam_steps,
            adam_lr=self.adam_lr,
            adam_batch=self.adam_batch,
            lbfgs_iters=self.lbfgs_iters,
            lbfgs_frames=self.lbfgs_frames,
            lbfgs_history=self.lbfgs_history,
            lbfgs_line_search="strong_wolfe",
            seed=self.seed,
            tol_nats_per_cell=self.tol_nats_per_cell,
            guide="AutoDelta",
            temperature=1.0,
            note="pyro 1.9.1 has no PyroLBFGS; the polish is torch.optim.LBFGS on the "
            "AutoDelta guide's unconstrained parameters against the same Trace_ELBO loss",
        )


@dataclass
class FitOutcome:
    """Everything :func:`write_fit` needs, plus the fitted parameters."""

    params: SP.V2Params
    objective: dict[str, Any]
    optimiser: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def converged(self) -> bool:
        return bool(self.optimiser.get("converged", False))


# ── initialisation ──────────────────────────────────────────────────────────


def _observed_db(batch: MD.SupportBatch) -> tuple[np.ndarray, np.ndarray]:
    """``(frame/mic-mean power, 20th-percentile floor curve in dB)``."""
    p = batch.power.detach().cpu().numpy()
    mean = p.mean(axis=(0, 1))
    db = 10.0 * np.log10(np.maximum(p.reshape(-1, p.shape[-1]), 1e-30))
    floor_db = np.quantile(db, FLOOR_INIT_QUANTILE, axis=0) + FLOOR_INIT_DB_CORRECTION
    return mean, floor_db


def initial_values(
    batch: MD.SupportBatch,
    *,
    mode: str,
    priors: MD.Priors = MD.PRIORS,
    frozen: dict[str, Any] | None = None,
) -> dict[str, Tensor]:
    """Site-name -> initial value for the free blocks of ``mode``."""
    free = MD.free_blocks(mode)
    band = batch.band.detach().cpu().numpy()
    obs_mean, floor_db = _observed_db(batch)
    t = lambda v: torch.as_tensor(np.asarray(v, dtype=np.float64), dtype=torch.float64)  # noqa: E731
    out: dict[str, Tensor] = {}

    if "dynamics" in free:
        out["sigma_nu"] = t(math.exp(priors.log_sigma_nu[0]))
        out["lam"] = t(math.exp(priors.log_lam[0]))
        out["sigma_eps"] = t([math.exp(priors.log_sigma_eps[0])] * 2)
        out["lam_eps"] = t([math.exp(priors.log_lam_eps[0])] * 2)

    carrier = None
    if batch.mode == "bench":
        carrier = batch.carrier_init if batch.carrier_init is not None else batch.carrier_mean
        assert carrier is not None
        if "carrier" in free:
            out["carrier_rev_s"] = carrier.detach().clone()

    # TWO probe forward passes in the model's OWN units — window response, lag
    # law, floor colour and transfer all included — so the seeds below are
    # offsets in dB against the model and not against a hand-derived formula:
    # ``quiet`` is the floor alone (every order at -300 dB) and ``unit`` adds a
    # comb whose every line carries unit power.
    seed_params = _seed_params(batch, mode=mode, priors=priors, frozen=frozen, carrier=carrier)
    with torch.no_grad():
        quiet = MD.forward(batch, _with_profile(seed_params, -300.0))
        unit = MD.forward(batch, _with_profile(seed_params, 0.0))
        lines_unit = (unit - quiet).clamp_min(1e-30).mean(dim=(0, 1)).cpu().numpy()
        floor_unit = quiet.mean(dim=(0, 1)).cpu().numpy()

    floor_offset_db = 0.0

    if "floor" in free:
        floor_offset_db = float(
            np.median(floor_db[band] - 10.0 * np.log10(np.maximum(floor_unit[band], 1e-30)))
        )
        out["floor_mean_db"] = t(float(np.asarray(seed_params.floor.mean_db)) + floor_offset_db)
        out["floor_shape_z"] = torch.zeros(FLOOR_SHAPE_N_CTRL, dtype=torch.float64)
        out["floor_tilt_db_oct"] = t(0.0)
        out["mic_floor_db"] = torch.zeros(batch.n_mics, dtype=torch.float64)
        if batch.mode == "flight":
            out["floor_exp"] = t(priors.floor_exp[0])
            out["floor_static_rel"] = t(math.exp(priors.log_floor_static[0]))

    if "profile" in free:
        floor_lin = np.maximum(floor_unit, 1e-30) * 10.0 ** (floor_offset_db / 10.0)
        excess = np.maximum(obs_mean - floor_lin, 1e-12)
        prof = np.full((batch.n_rotors, batch.k_max), priors.profile_db[0], dtype=np.float64)
        if carrier is not None:
            lo_r = hi_r = np.atleast_1d(np.asarray(carrier.cpu(), dtype=np.float64))
        else:
            assert batch.rate_work is not None
            rw = batch.rate_work.detach()
            lo_r = rw.amin(dim=(1, 2)).cpu().numpy()
            hi_r = rw.amax(dim=(1, 2)).cpu().numpy()
        df = float(batch.grid.freqs_hz[1] - batch.grid.freqs_hz[0])
        f_top = float(batch.grid.freqs_hz[band].max())
        for r in range(int(lo_r.size)):
            for k in range(1, batch.k_max + 1):
                f_lo, f_hi = k * float(lo_r[r]), k * float(hi_r[r])
                if f_hi < SP.BAND_F_MIN or f_lo > f_top:
                    continue
                # the window spans where the line went over the batch, plus the
                # window's own main lobe: a moving carrier smears the
                # frame-mean peak across k * (f_max - f_min)
                j0 = max(0, int(math.floor(f_lo / df)) - 3)
                j1 = min(excess.size, int(math.ceil(f_hi / df)) + 4)
                num = float(np.max(excess[j0:j1]))
                den = float(np.max(lines_unit[j0:j1]))
                if den > 0.0 and num > 0.0:
                    prof[r, k - 1] = float(np.clip(10.0 * math.log10(num / den), -120.0, 40.0))
        out["profile_db"] = t(prof)
        if batch.mode == "flight":
            out["amp_exp"] = t(priors.amp_exp[0])

    if "mic" in free:
        p = batch.power.detach().cpu().numpy()
        per_mic_db = 10.0 * np.log10(np.maximum(p[:, :, band].mean(axis=(1, 2)), 1e-30))
        out["gain_all_db"] = t(per_mic_db - per_mic_db.mean())
        out["mic_line_gain_db"] = torch.zeros(batch.n_mics, batch.n_rotors, dtype=torch.float64)
    return out


def _seed_params(
    batch: MD.SupportBatch,
    *,
    mode: str,
    priors: MD.Priors,
    frozen: dict[str, Any] | None,
    carrier: Tensor | None,
) -> SP.V2Params:
    """A concrete parameter set at the priors' centres (no Pyro site involved).

    Used ONLY by :func:`initial_values` for its two probe forward passes; a
    frozen block uses its frozen value so a ``--floor-only`` seed is measured
    against the real comb rather than against a prior-mean one.
    """
    free = MD.free_blocks(mode)
    fz = dict(frozen or {})
    t = lambda v: torch.as_tensor(np.asarray(v, dtype=np.float64), dtype=torch.float64)  # noqa: E731
    r, m, k = batch.n_rotors, batch.n_mics, batch.k_max
    flight = batch.mode == "flight"

    def pick(block: str, key: str, default: Any, shape: tuple[int, ...] | None = None) -> Tensor:
        v = t(default) if block in free or key not in fz else t(fz[key])
        if shape is not None and tuple(v.shape) != shape:
            if v.numel() == 1:
                v = v.reshape(()).expand(shape).clone()
            elif v.ndim == 2 and int(v.shape[1]) >= shape[1]:
                v = v[: shape[0], : shape[1]].clone()
        return v

    zero = torch.zeros((), dtype=torch.float64)
    return SP.V2Params(
        sigma_nu=pick("dynamics", "sigma_nu", math.exp(priors.log_sigma_nu[0])),
        lam=pick("dynamics", "lam", math.exp(priors.log_lam[0])),
        sigma_eps=pick("dynamics", "sigma_eps", [math.exp(priors.log_sigma_eps[0])] * 2, (2,)),
        lam_eps=pick("dynamics", "lam_eps", [math.exp(priors.log_lam_eps[0])] * 2, (2,)),
        profile_db=pick("profile", "profile_db", np.zeros((r, k)), (r, k)),
        floor=SP.FloorParams(
            mean_db=pick("floor", "floor_mean_db", 0.0),
            shape_z=pick(
                "floor", "floor_shape_z", np.zeros(FLOOR_SHAPE_N_CTRL), (FLOOR_SHAPE_N_CTRL,)
            ),
            tilt_db_oct=pick("floor", "floor_tilt_db_oct", 0.0),
            mic_floor_db=pick("floor", "mic_floor_db", np.zeros(m), (m,)),
            exp=pick("floor", "floor_exp", priors.floor_exp[0]) if flight else zero,
            static_rel=(
                pick("floor", "floor_static_rel", math.exp(priors.log_floor_static[0]))
                if flight
                else zero
            ),
        ),
        mic_line_gain_db=pick("mic", "mic_line_gain_db", np.zeros((m, r)), (m, r)),
        gain_all_db=pick("mic", "gain_all_db", np.zeros(m), (m,)),
        carrier_rev_s=carrier,
        amp_exp=pick("profile", "amp_exp", priors.amp_exp[0]) if flight else zero,
        p=priors.p,
    )


def _with_profile(params: SP.V2Params, db: float) -> SP.V2Params:
    import dataclasses

    prof = torch.full_like(torch.as_tensor(params.profile_db, dtype=torch.float64), float(db))
    return dataclasses.replace(params, profile_db=prof)


# ── the fit ─────────────────────────────────────────────────────────────────


def _grad_norm(params: Sequence[Tensor]) -> float:
    return float(math.sqrt(sum(float((p.grad**2).sum()) for p in params if p.grad is not None)))


def fit_support(
    batch: MD.SupportBatch,
    *,
    mode: str,
    priors: MD.Priors = MD.PRIORS,
    frozen: dict[str, Any] | None = None,
    optim: OptimSpec = OptimSpec(),
    forward_kw: dict[str, Any] | None = None,
    progress: int = 0,
) -> FitOutcome:
    """MAP-fit one support (or one pooled set of flight windows)."""
    torch.manual_seed(int(optim.seed))
    pyro.set_rng_seed(int(optim.seed))
    pyro.clear_param_store()

    full = batch
    init = initial_values(batch, mode=mode, priors=priors, frozen=frozen)

    def model_for(b: MD.SupportBatch) -> Any:
        def fn() -> Any:
            return MD.support_model(
                b, mode=mode, priors=priors, frozen=frozen, temperature=1.0, forward_kw=forward_kw
            )

        return fn

    guide = AutoDelta(model_for(full), init_loc_fn=init_to_value(values=init))
    elbo = Trace_ELBO()

    rng = np.random.default_rng(int(optim.seed))
    n_frames = int(full.power.shape[1])
    use_batches = (
        full.mode == "flight" and optim.adam_batch is not None and optim.adam_batch < n_frames
    )

    t0 = time.time()
    adam_optim = PyroOptim(torch.optim.Adam, {"lr": float(optim.adam_lr)})
    adam = SVI(model_for(full), guide, adam_optim, loss=elbo)
    adam_losses: list[float] = []
    loss: Any
    for step in range(int(optim.adam_steps)):
        if use_batches:
            assert optim.adam_batch is not None
            idx = rng.choice(n_frames, size=int(optim.adam_batch), replace=False)
            sub = MD.batch_slice(full, idx)
            # Preserve Adam's moments across frame minibatches.  Recreating a
            # Pyro optimiser here silently converts 1,500 Adam steps into 1,500
            # independent first steps.
            loss = SVI(model_for(sub), guide, adam_optim, loss=elbo).step()
        else:
            loss = adam.step()
        adam_losses.append(float(loss))
        if progress and step % progress == 0:
            print(f"  adam {step:5d}  loss {loss:.6g}", flush=True)
    adam_s = time.time() - t0
    adam_final = float(adam_losses[-1]) if adam_losses else float("nan")

    # the polish: a DETERMINISTIC frame set (a strong-Wolfe line search on a
    # resampled objective is not a line search)
    polish = full
    if full.mode == "flight" and optim.lbfgs_frames is not None and optim.lbfgs_frames < n_frames:
        idx = np.linspace(0, n_frames - 1, int(optim.lbfgs_frames)).round().astype(np.int64)
        polish = MD.batch_slice(full, np.unique(idx))
    loss_fn = model_for(polish)
    params = [p for p in guide.parameters() if p.requires_grad]
    elbo_of = lambda: elbo.differentiable_loss(loss_fn, guide)  # noqa: E731

    def run_lbfgs(max_iter: int) -> tuple[float, int]:
        """One L-BFGS pass; returns its final loss and its evaluation count."""
        opt = torch.optim.LBFGS(
            params,
            max_iter=int(max_iter),
            history_size=int(optim.lbfgs_history),
            line_search_fn="strong_wolfe",
        )
        count = 0

        def closure() -> Tensor:
            nonlocal count
            opt.zero_grad(set_to_none=False)
            loss = elbo_of()
            loss.backward()
            count += 1
            return loss

        opt.step(closure)
        with torch.no_grad():
            return float(elbo_of()), count

    t1 = time.time()
    with torch.no_grad():
        before = float(elbo_of())
    first_pass, evals = run_lbfgs(int(optim.lbfgs_iters))
    # THE convergence test, not a threshold on a trace: L-BFGS is RESTARTED
    # with a fresh Hessian approximation and half the budget. If a restart
    # cannot find more than ``tol_nats_per_cell`` per observed cell, the first
    # pass had converged; if it can, it had not, and the fit says so.
    restart, restart_evals = run_lbfgs(max(1, int(optim.lbfgs_iters) // 2))
    lbfgs_s = time.time() - t1
    for p in params:
        p.grad = None
    elbo_of().backward()
    gnorm = _grad_norm(params)
    gain_per_cell = max(0.0, first_pass - restart) / max(1, polish.n_cells)
    converged = bool(np.isfinite(restart) and gain_per_cell < float(optim.tol_nats_per_cell))
    final_loss = min(first_pass, restart)

    med = guide.median()
    fitted = MD.sample_params_from_values(full, mode=mode, priors=priors, frozen=frozen, values=med)
    with torch.no_grad():
        m_model = MD.forward(full, fitted, **(forward_kw or {}))
        objective = MD.objective_breakdown(full, m_model)
    return FitOutcome(
        params=fitted,
        objective=objective,
        optimiser=dict(
            **optim.as_dict(),
            adam_final_loss=adam_final,
            adam_first_loss=float(adam_losses[0]) if adam_losses else float("nan"),
            adam_wall_s=adam_s,
            lbfgs_loss_before=before,
            lbfgs_loss_after=final_loss,
            lbfgs_loss_first_pass=float(first_pass),
            lbfgs_loss_after_restart=float(restart),
            lbfgs_restart_gain_per_cell=float(gain_per_cell),
            lbfgs_evals=int(evals),
            lbfgs_restart_evals=int(restart_evals),
            lbfgs_wall_s=lbfgs_s,
            lbfgs_frames_used=int(polish.power.shape[1]),
            lbfgs_cells=int(polish.n_cells),
            grad_norm=gnorm,
            converged=converged,
            which_converged="lbfgs" if converged else "none",
            wall_s=adam_s + lbfgs_s,
        ),
        diagnostics=dict(
            batch=dict(full.diagnostics),
            n_cells=full.n_cells,
            adam_loss_head=adam_losses[:5],
            adam_loss_tail=adam_losses[-5:],
            init_floor_mean_db=float(init["floor_mean_db"]) if "floor_mean_db" in init else None,
        ),
    )


# ── the record ──────────────────────────────────────────────────────────────


def write_fit(
    path: str | Path,
    *,
    support: str,
    kind: str,
    mode: str,
    outcome: FitOutcome,
    batch: MD.SupportBatch,
    priors: MD.Priors = MD.PRIORS,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the ``noise-v2-fit/1`` JSON and return its path."""
    p = MD.params_to_dict(outcome.params)
    k = np.arange(1, int(np.asarray(p["profile"]["profile_db"]).shape[1]) + 1)
    payload: dict[str, Any] = dict(
        schema=FIT_SCHEMA,
        support=support,
        supports=list(batch.members) if batch.members else [support],
        kind=kind,
        mode=mode,
        n_rotors=batch.n_rotors,
        n_mics=batch.n_mics,
        k_max=batch.k_max,
        sr=int(batch.grid.sr),
        front_end=dict(batch.grid.diagnostics),
        params=p,
        objective=outcome.objective,
        optimiser=outcome.optimiser,
        priors=priors.as_dict(),
        diagnostics=dict(
            outcome.diagnostics,
            saturated_coherence=dict(
                k=[1, 10, 20, 40],
                value=[
                    float(
                        saturated_coherence(
                            kk, sigma_eps=[p["sigma_eps_even"], p["sigma_eps_odd"]], p=p["p"]
                        )
                    )
                    for kk in (1, 10, 20, 40)
                ],
            ),
            profile_orders=k.tolist(),
            # ``floor_mean_db`` and the MEAN of ``mic_floor_db`` are a ridge in
            # C4's parameterisation (the model reads their sum); their SUM is
            # the identified broadband floor level and is what the findings
            # quote. Kept unpinned so the flight forward model stays bit-equal
            # to C4's ``_floor_frames``.
            floor_level_db=float(p["floor"]["floor_mean_db"] + np.mean(p["floor"]["mic_floor_db"])),
        ),
        git=_git_head(),
    )
    if extra:
        payload.update(extra)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    return out
