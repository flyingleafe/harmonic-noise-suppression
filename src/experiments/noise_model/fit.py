"""MAP fit of the v2 model and the ``noise-v2-fit/2`` JSON it writes.

ONE objective and ONE optimiser pair, per the round plan: an ``AutoDelta``
guide over the free blocks of :mod:`.model`, driven by ``pyro.optim.Adam``
(lr 0.02, 1500 steps by default) and then polished with torch's L-BFGS
under a strong-Wolfe line search (:class:`.lbfgs.LBFGS`: torch's step plus an
optional relative objective-change stop, ``OptimSpec.lbfgs_rtol``, off in
the v2 modes). Pyro 1.9.1 ships no ``PyroLBFGS``, so the polish runs on the
guide's own unconstrained parameters against
``Trace_ELBO().differentiable_loss`` — the same objective the Adam phase
minimised, to the last term. WHICH phase converged, how long each took and
both objectives are recorded in the JSON's ``optimiser`` block; nothing is
tuned against a result.

WHY BOTH. Adam gets the levels and the carrier into the right basin from a
cold, weakly-informed start; L-BFGS is what actually resolves a comb's
per-order profile, where the curvature across 130 orders spans decades. A fit
that only ran Adam is marked ``converged: false`` unless the polish agrees.

MEASUREMENT AND INITIALISATION are not a detail on this model, and in R3 they
are the same pass (:func:`seeds`): two priors are centred on what the support
shows, so the measurement is carried on the batch (:class:`.model.Measured`)
and the guide is initialised from the very same numbers. ``AutoDelta``'s
default ``init_to_median`` would start every order of the profile at a prior
mean tens of dB from a real support, and a Whittle objective whose model is
40 dB under the data has a gradient dominated by ``I / M``. What one pass
measures:

* the floor level, from the band median of the observed 20th-percentile dB
  curve plus C4's +6.5 dB exponential-percentile correction — the centre of
  the ``floor_mean_db`` prior as well as its initialisation,
* each order's line power and its SNR over that floor, measured against a
  UNIT-profile forward pass so the seed is in the model's own units (window
  response, lag law and transfer included); the SNR picks which of the
  profile prior's two regimes the line is in,
* each visible line's -3 dB half width, floored at the window's resolution
  and clipped to the width prior's central 95 %, as the ``gamma_rk``
  initialisation (the prior itself is the physical law, never the
  measurement),
* the per-microphone broadband gain from each microphone's band-mean level,
* NOT the bench carrier: it is FROZEN at the support index's window-refined
  value (bench rule rev 2), a constant of the model with no site and no
  ``N(survey, 0.5^2)`` prior, so the pass only READS it for its probe passes.

``sigma_nu`` and ``lam`` start at their prior medians, which is the point of
having measured them. Every fit records the low-order ``gamma_rk`` check
(:func:`gamma_low_order_check`) and which speed laws the pool's span pinned.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDelta, init_to_value
from pyro.optim.optim import PyroOptim
from torch import Tensor
from torch.distributions import constraints

from data_processing.noise_model.constants import FLOOR_TILT_REF_HZ
from data_processing.noise_model.floor import floor_geometry
from data_processing.noise_model.v3 import WIND_CORNER_HZ
from experiments.stochastic_fit.model import FLOOR_SHAPE_N_CTRL
from experiments.stochastic_fit.revised_phase import (
    FLOOR_INIT_DB_CORRECTION,
    FLOOR_INIT_QUANTILE,
    _git_head,
)

from . import FIT_SCHEMA, FIT_SCHEMA_V3
from . import model as MD
from . import spectrum as SP
from .lbfgs import LBFGS

__all__ = [
    "FitOutcome",
    "OptimSpec",
    "OptimSpecV3",
    "ProfileInit",
    "Seeds",
    "V2Fit",
    "chunked_objective",
    "fit_support",
    "fit_v3",
    "fit_latents",
    "gamma_low_order_check",
    "initial_values",
    "load_channel_gains",
    "load_profile_init",
    "measure_batch",
    "seeds",
    "span_pin_record",
    "warm_start_v3",
    "write_fit",
]


@dataclass(frozen=True)
class ProfileInit:
    """A per-line ``profile_db`` centre measured OUTSIDE the likelihood.

    ``profile_db`` is ``(R, K)`` in the model's own units and ``sigma_db`` the
    matching per-line prior width. **NaN means "leave the model alone"**: a NaN
    centre keeps the initialiser's measured value, a NaN width keeps the R3
    two-regime prior, so a file can never pin an order its source never
    measured. Both are re-shaped to the batch's ``(R, K)`` by
    :meth:`aligned` — padded with NaN when the source is narrower, cut when it
    is wider.

    Written by ``scripts/noise_v2_fourmotor.py estimate`` from the multi-rotor
    estimator of :mod:`.multirotor`, whose per-order error budget is the sd.
    """

    profile_db: np.ndarray
    sigma_db: np.ndarray | None = None
    carrier_offset_rev_s: np.ndarray | None = None
    source: str = ""

    def aligned(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """``(centre, sd)`` on the batch's ``(R, K)`` grid, NaN-padded."""
        r_n, k_n = int(shape[0]), int(shape[1])
        db = np.atleast_2d(np.asarray(self.profile_db, dtype=np.float64))
        if int(db.shape[0]) != r_n:
            raise ValueError(f"profile init has {int(db.shape[0])} rotors, the batch has {r_n}")
        sd = (
            np.full_like(db, np.nan)
            if self.sigma_db is None
            else np.atleast_2d(np.asarray(self.sigma_db, dtype=np.float64))
        )
        if sd.shape != db.shape:
            raise ValueError(f"sigma_db {sd.shape} does not match profile_db {db.shape}")
        out_db = np.full((r_n, k_n), np.nan)
        out_sd = np.full((r_n, k_n), np.nan)
        m = min(k_n, int(db.shape[1]))
        out_db[:, :m] = db[:, :m]
        out_sd[:, :m] = sd[:, :m]
        return out_db, out_sd


def load_profile_init(path: Any) -> ProfileInit:
    """Read a ``.npz`` (or ``.json``) ``profile_db`` / ``sigma_db`` payload."""
    p = Path(path)
    if p.suffix == ".json":
        d = json.loads(p.read_text())
    else:
        with np.load(p, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
    if "profile_db" not in d:
        raise ValueError(f"{p}: no profile_db")
    off = d.get("carrier_offset_rev_s")
    return ProfileInit(
        profile_db=np.asarray(d["profile_db"], dtype=np.float64),
        sigma_db=None if d.get("sigma_db") is None else np.asarray(d["sigma_db"], dtype=np.float64),
        carrier_offset_rev_s=None if off is None else np.asarray(off, dtype=np.float64).reshape(-1),
        source=str(p),
    )


#: ``flight_v3``'s :attr:`OptimSpec.lbfgs_rtol` (the CLI's ``--lbfgs-rtol``
#: default there): the relative change that IS ``tol_nats_per_cell`` = 1e-4
#: nats per cell per iteration on the v3 pools, whose objective sits at
#: 6..9 nats per cell in magnitude (smoke pool -6.2, DREGON -7.2, Michael's
#: standby -9.3 on the CPU round 1): 1e-5 x 6..9 = 0.6..0.9e-4 nats per cell.
V3_LBFGS_RTOL = 1e-5

#: ``flight_v3``'s rig L-BFGS frame subset (the CLI's ``--lbfgs-frames``
#: default there), stratified over the windows; the fit ends with one
#: all-frames polish (:class:`OptimSpecV3`).
V3_LBFGS_FRAMES = 64

#: ``flight_v3``'s kernel work rate (the CLI's ``--work-rate`` default there),
#: with the render chain's transfer at that rate (which a render of the fit
#: goes through: ``render_noise`` defaults to the fit's ``front_end.sr_work``).
#: The lowest rate whose expected periodogram matches the 64 kHz kernel to
#: < 0.05 dB over 30-7900 Hz on the smoke windows (16 kHz misses by 0.07 dB,
#: aliased top-order skirts and shaft tails), at half the cost per evaluation;
#: its transfer is 64 kHz's to 0.002 dB (docs/experiments/noise-model-v3.md
#: § "GPU round").
V3_WORK_RATE = 32000


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
    #: draw the ``lbfgs_frames`` subset per window (:func:`.model.stratified_frames`)
    #: instead of evenly over the pooled frame sequence (the v2 modes' rule)
    lbfgs_stratified: bool = False
    lbfgs_history: int = 10
    #: stop an L-BFGS pass once one iteration moves the objective by less than
    #: this fraction of it (scipy's ``ftol``, :class:`.lbfgs.LBFGS`);
    #: ``lbfgs_iters`` stays the cap. ``None`` (the v2 modes' default) leaves
    #: torch's own tests alone, whose absolute 1e-9 lets a pass run to
    #: ``lbfgs_iters``. ``flight_v3``'s default is :data:`V3_LBFGS_RTOL`.
    lbfgs_rtol: float | None = None
    seed: int = 0
    #: objective gain per observed cell an L-BFGS RESTART may still find and
    #: the fit still count as converged (nats/cell)
    tol_nats_per_cell: float = 1e-4
    #: log-space sd of the MULTI-START perturbation of the free dynamics
    #: initialisation. A bench fit is otherwise deterministic — the guide is
    #: initialised at a data-driven point, Adam sees every cell and L-BFGS is
    #: deterministic — so a bare change of ``seed`` reproduces the same fit to
    #: the last digit and says nothing about the landscape. With a jitter, the
    #: restarts are genuine multi-starts and their spread is the evidence about
    #: how well the MAP problem is posed.
    init_jitter: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return dict(
            adam_steps=self.adam_steps,
            adam_lr=self.adam_lr,
            adam_batch=self.adam_batch,
            lbfgs_iters=self.lbfgs_iters,
            lbfgs_frames=self.lbfgs_frames,
            lbfgs_stratified=self.lbfgs_stratified,
            lbfgs_history=self.lbfgs_history,
            lbfgs_rtol=self.lbfgs_rtol,
            lbfgs_line_search="strong_wolfe",
            seed=self.seed,
            tol_nats_per_cell=self.tol_nats_per_cell,
            init_jitter=self.init_jitter,
            guide="AutoDelta",
            temperature=1.0,
            note="pyro 1.9.1 has no PyroLBFGS; the polish is torch.optim.LBFGS on the "
            "AutoDelta guide's unconstrained parameters against the same Trace_ELBO loss",
        )


@dataclass(frozen=True)
class OptimSpecV3:
    """The v3 alternation's schedule (explainer §3.4).

    Round 0 is step (i): the rig on the pool with every latent at zero, under
    ``rig`` — the v2 schedule, restarts and jitter included. Each of up to
    ``rounds`` rounds is step (ii), every window's latents alone with the rig
    fixed (``latent_lbfgs_iters`` of strong-Wolfe L-BFGS on the pooled
    latents, from the previous round's tracks, the rig's line spectra stored
    in ``latent_dtype``), then step (iii), the rig again with the latents
    fixed, warm-started (``refit_adam_steps`` Adam steps, then ``rig``'s
    L-BFGS polish). The alternation stops when the Whittle term moves by less
    than ``tol_nats_per_cell`` per observed cell between rounds.

    When ``rig.lbfgs_frames`` is a SUBSET of the pool's frames (``flight_v3``'s
    CLI default: 64, stratified over the windows), every rig step above runs its
    L-BFGS on that subset, and the fit ends with ONE all-frames L-BFGS polish of
    the rig at the final latents (``optimiser.polish``). The latent step always
    sees every frame.
    """

    rig: OptimSpec = field(default_factory=OptimSpec)
    rounds: int = 3
    refit_adam_steps: int = 0
    latent_lbfgs_iters: int = 100
    latent_lbfgs_history: int = 10
    latent_dtype: str = "float64"
    tol_nats_per_cell: float = 1e-4
    #: after each latent step (and once after the last rig step) move the
    #: static part of the latents into the rig by :func:`static_ridge_step`,
    #: the exact prior minimum along the Whittle-invariant directions
    ridge_step: bool = False

    def as_dict(self) -> dict[str, Any]:
        return dict(
            rig=self.rig.as_dict(),
            rounds=self.rounds,
            refit_adam_steps=self.refit_adam_steps,
            latent_lbfgs_iters=self.latent_lbfgs_iters,
            latent_lbfgs_history=self.latent_lbfgs_history,
            latent_line_search="strong_wolfe",
            latent_dtype=self.latent_dtype,
            tol_nats_per_cell=self.tol_nats_per_cell,
            ridge_step=self.ridge_step,
            scheme="(i) rig, latents at zero; then per round (ii) each window's latents, rig "
            "fixed, [ridge step,] (iii) rig, latents fixed; stop when the Whittle term and the "
            "total objective move < tol per cell; "
            "a rig L-BFGS on a frame subset ends in one all-frames rig L-BFGS polish",
        )


@dataclass
class FitOutcome:
    """Everything :func:`write_fit` needs, plus the fitted parameters."""

    params: SP.V2Params
    objective: dict[str, Any]
    optimiser: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    #: the fitted ``comb_gain_db`` when the mode frees it (``flight_floor_only``
    #: does), ``None`` otherwise. It is NOT a field of
    #: :class:`~experiments.noise_model.spectrum.V2Params`: the model folds it
    #: into ``profile_db``, so it has to be carried here to be recorded.
    comb_gain_db: float | None = None
    #: the fitted per-order ``low_order_gain_db`` when the mode frees it
    #: (``flight_floor_lowk``), ``None`` otherwise. Folded into ``profile_db``
    #: by the model exactly as ``comb_gain_db`` is, so it too is carried here
    #: to be recorded rather than re-applied.
    low_order_gain_db: list[float] | None = None
    #: the guide's fitted site values (``AutoDelta.median()``, detached): what
    #: a warm-started refit of the same model starts from
    sites: dict[str, Tensor] = field(default_factory=dict)
    #: v3 only: the ``latents`` block of the record (per window, the MAP block
    #: tracks) and the :class:`.model.WindowLatents` they were read from
    latents: dict[str, Any] | None = None
    window_latents: dict[int, MD.WindowLatents] | None = None

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


@dataclass(frozen=True)
class Seeds:
    """What ONE pass over the support's data yields: the prior centres the R3
    model needs (:class:`.model.Measured`) and the guide's initial values.

    They come from the same probe forward passes, so a site's initialisation
    and the prior it is initialised under cannot drift apart.
    """

    measured: MD.Measured
    init: dict[str, Tensor]


def measure_batch(
    batch: MD.SupportBatch,
    *,
    mode: str,
    priors: MD.Priors = MD.PRIORS,
    frozen: dict[str, Any] | None = None,
    pin: dict[str, Any] | None = None,
    low_orders: int | None = None,
) -> MD.Measured:
    """The data-driven prior centres of one batch (see :func:`seeds`)."""
    return seeds(
        batch, mode=mode, priors=priors, frozen=frozen, pin=pin, low_orders=low_orders
    ).measured


def initial_values(
    batch: MD.SupportBatch,
    *,
    mode: str,
    priors: MD.Priors = MD.PRIORS,
    frozen: dict[str, Any] | None = None,
    pin: dict[str, Any] | None = None,
    low_orders: int | None = None,
) -> dict[str, Tensor]:
    """Site-name -> initial value for the free sites of ``mode``.

    A pinned, frozen or span-pinned site gets NO entry: it has no Pyro site at
    all, and a stale key would be ignored by ``init_to_value`` — a pin that
    silently does not pin.
    """
    return seeds(
        batch, mode=mode, priors=priors, frozen=frozen, pin=pin, low_orders=low_orders
    ).init


def seeds(
    batch: MD.SupportBatch,
    *,
    mode: str,
    priors: MD.Priors = MD.PRIORS,
    frozen: dict[str, Any] | None = None,
    pin: dict[str, Any] | None = None,
    low_orders: int | None = None,
    profile_init: ProfileInit | None = None,
) -> Seeds:
    """Measure the support and seed every free site from the measurement.

    ``profile_init`` overrides the measured profile per line where it carries a
    finite number: the initialisation everywhere it does, and the prior's
    centre and width wherever it also carries a finite sd. See
    :class:`ProfileInit`.
    """
    if mode == MD.V3_MODE:
        return _seeds_v3(batch, priors=priors, pin=pin, profile_init=profile_init)
    free = MD.free_blocks(mode)
    fz = dict(frozen or {})
    span_pinned = MD.span_pinned_sites(batch, priors=priors)
    band = batch.band.detach().cpu().numpy()
    obs_mean, floor_db = _observed_db(batch)
    t = lambda v: torch.as_tensor(np.asarray(v, dtype=np.float64), dtype=torch.float64)  # noqa: E731
    out: dict[str, Tensor] = {}
    flight = batch.mode == "flight"

    carrier = None
    if batch.mode == "bench":
        # the frozen carrier: a constant of the model, so no init entry
        carrier = batch.carrier_mean
        assert carrier is not None

    if "dynamics" in free:
        out["sigma_nu"] = t(math.exp(priors.sigma_nu_prior(flight)[0]))
        if not flight and not MD.is_pinned(pin, "lam"):
            # in flight ``lam`` is a CONSTANT (the bench value or the approved
            # default), so it has no site to initialise
            out["lam"] = t(math.exp(priors.log_lam[0]))

    # Probe forward passes in the model's OWN units — window response, lag
    # law, floor colour and transfer all included — so the seeds below are
    # offsets in dB against the model and not against a hand-derived formula:
    # ``quiet`` is the floor alone (every order at -300 dB) and ``unit`` adds a
    # comb whose every line carries unit power.
    seed_params = _seed_params(
        batch, mode=mode, priors=priors, frozen=frozen, pin=pin, carrier=carrier
    )
    with torch.no_grad():
        quiet = MD.forward(batch, _with_profile(seed_params, -300.0))
        unit = MD.forward(batch, _with_profile(seed_params, 0.0))
        lines_unit = (unit - quiet).clamp_min(1e-30).mean(dim=(0, 1)).cpu().numpy()
        floor_unit = quiet.mean(dim=(0, 1)).cpu().numpy()
        # a THIRD pass when the transplanted comb's own level is free: the comb
        # at its FROZEN profile, so the seed below measures how far that comb
        # sits from the observed line excess instead of guessing
        lines_frozen = (
            (MD.forward(batch, seed_params) - quiet).clamp_min(1e-30).mean(dim=(0, 1)).cpu().numpy()
            if "comb_gain" in free
            else None
        )

    floor_offset_db = float(
        np.median(floor_db[band] - 10.0 * np.log10(np.maximum(floor_unit[band], 1e-30)))
    )
    floor_mean_db = float(np.asarray(seed_params.floor.mean_db)) + floor_offset_db
    if "floor" in free:
        out["floor_mean_db"] = t(floor_mean_db)
        out["floor_shape_z"] = torch.zeros(FLOOR_SHAPE_N_CTRL, dtype=torch.float64)
        out["floor_tilt_db_oct"] = t(0.0)
        out["mic_floor_db"] = torch.zeros(batch.n_mics, dtype=torch.float64)
        if flight and "floor_exp" not in span_pinned:
            out["floor_exp"] = t(math.exp(priors.log_floor_exp[0]))
        if flight and "floor_static_rel" not in span_pinned:
            out["floor_static_rel"] = t(math.exp(priors.log_floor_static[0]))
    else:
        floor_mean_db = float(np.asarray(seed_params.floor.mean_db))

    # ── what every line of the comb measures ────────────────────────────────
    floor_lin = np.maximum(floor_unit, 1e-30) * 10.0 ** (floor_offset_db / 10.0)
    excess = np.maximum(obs_mean - floor_lin, 1e-12)
    df = float(batch.grid.freqs_hz[1] - batch.grid.freqs_hz[0])
    shape = (batch.n_rotors, batch.k_max)
    orders = np.broadcast_to(np.arange(1, batch.k_max + 1, dtype=np.float64), shape)
    gamma_median = float(priors.gamma_per_order_hz) * orders
    # a line the data never shows sits at the prior's two centres: the
    # below-floor profile regime and the physical width law
    prof = np.full(shape, floor_mean_db + float(priors.profile_below_offset_db), dtype=np.float64)
    snr = np.full(shape, -99.0, dtype=np.float64)
    gamma = gamma_median.copy()
    windows = _line_windows(batch, band=band, carrier=carrier)
    for r, k, j0, j1 in windows:
        num = float(np.max(excess[j0:j1]))
        den = float(np.max(lines_unit[j0:j1]))
        if den <= 0.0 or num <= 0.0:
            continue
        prof[r, k - 1] = float(np.clip(10.0 * math.log10(num / den), -120.0, 40.0))
        jpk = int(j0 + np.argmax(excess[j0:j1]))
        snr[r, k - 1] = 10.0 * math.log10(num / max(float(floor_lin[jpk]), 1e-30))
        if snr[r, k - 1] >= priors.line_visible_snr_db:
            # the line's own -3 dB half width, floored at the window's
            # resolution and held inside the prior's central 95 %: a width
            # measured on a noisy peak must seed the fit, not steer it
            gamma[r, k - 1] = float(
                np.clip(
                    max(_half_width_hz(excess, jpk, j0, j1, df), batch.resolution_hz),
                    gamma_median[r, k - 1] * math.exp(-2.0 * priors.gamma_log_sd),
                    gamma_median[r, k - 1] * math.exp(2.0 * priors.gamma_log_sd),
                )
            )
    ext_db: np.ndarray | None = None
    ext_sd: np.ndarray | None = None
    if profile_init is not None:
        ext_db, ext_sd = profile_init.aligned(shape)
    measured = MD.Measured(
        floor_mean_db=floor_mean_db,
        profile_db=prof,
        line_snr_db=snr,
        gamma_hz=gamma,
        resolution_hz=batch.resolution_hz,
        # the prior moves only where a WIDTH was supplied: an order the study
        # could not measure keeps the model's own two-regime centre even though
        # its initialisation comes from the estimator
        profile_prior_db=(
            None
            if ext_db is None or ext_sd is None
            else np.where(np.isfinite(ext_sd), ext_db, np.nan)
        ),
        profile_prior_sd_db=ext_sd,
    )

    if "dynamics" in free and "gamma_hz" not in fz:
        out["gamma_hz"] = t(gamma)
    if "profile" in free:
        out["profile_db"] = t(
            prof if ext_db is None else np.where(np.isfinite(ext_db), ext_db, prof)
        )
        if flight and "amp_exp" not in span_pinned:
            out["amp_exp"] = t(priors.amp_exp[0])

    if "comb_gain" in free:
        assert lines_frozen is not None
        shifts: list[float] = []
        per_order: dict[int, list[float]] = {}
        for _r, kk_line, j0, j1 in windows:
            num = float(np.max(excess[j0:j1]))
            den = float(np.max(lines_frozen[j0:j1]))
            if den > 0.0 and num > 0.0:
                shifts.append(10.0 * math.log10(num / den))
                per_order.setdefault(int(kk_line), []).append(shifts[-1])
        # the MEDIAN over the lines: one order whose window catches a tonal the
        # frozen comb never had must not set the level of the whole comb. Held
        # inside the prior's 3 sd, which is where a seed stops being a seed.
        wide = 3.0 * priors.comb_gain_db[1]
        comb_seed = float(np.clip(np.median(shifts), -wide, wide)) if shifts else 0.0
        out["comb_gain_db"] = t(comb_seed)
        if "low_order_gain" in free:
            # what each low order wants ON TOP of that scalar: its own measured
            # shift (median over the rotors, which share the site) minus the
            # shared one, so the seed of the whole block is zero where the
            # comb's low orders already sit at the right level
            n_low = min(int(low_orders or priors.low_orders), batch.k_max)
            low = np.zeros(n_low, dtype=np.float64)
            for i in range(n_low):
                vals = per_order.get(i + 1)
                if vals:
                    low[i] = float(np.clip(float(np.median(vals)) - comb_seed, -wide, wide))
            out["low_order_gain_db"] = t(low)

    if "mic" in free:
        p = batch.power.detach().cpu().numpy()
        per_mic_db = 10.0 * np.log10(np.maximum(p[:, :, band].mean(axis=(1, 2)), 1e-30))
        out["gain_all_db"] = t(per_mic_db - per_mic_db.mean())
        out["mic_line_gain_db"] = torch.zeros(batch.n_mics, batch.n_rotors, dtype=torch.float64)
    return Seeds(measured=measured, init=out)


#: Median of a unit half-normal, ``Phi^-1(3/4)``: where v3 starts a
#: half-normal site (its mode, zero, is a boundary no log-space start can reach).
HALFNORMAL_MEDIAN = 0.6744897501960817

#: The v3 floor-shape measurement reads only bins where the start comb
#: predicts less than this share of the observed pooled power (see
#: :func:`_seeds_v3`): the rest are the comb's, not the floor's.
V3_FLOOR_COMB_SHARE = 0.5

#: How far under its prior centre (the pooled line + floor level) a v3 line
#: whose measured excess is at or below the floor STARTS: sunk, as the
#: explainer's §2.3 argues it will end up, rather than parked by a rule.
V3_SINK_INIT_DB = 20.0


def _seeds_v3(
    batch: MD.SupportBatch,
    *,
    priors: MD.Priors,
    pin: dict[str, Any] | None,
    profile_init: ProfileInit | None,
) -> Seeds:
    """The v3 measurement (explainer §2.3-§2.4) and the guide's start.

    One pass, probe forward passes in the model's own units as in v2:

    * ``mu`` (``floor_mean_db``): the band median of the observed floor curve
      against the model's unit floor — v2's floor level, now a CONSTANT;
    * the profile prior's centre ``p_hat``: the pooled observed level at each
      line's peak — line PLUS floor, an upper bound — against the unit comb,
      for EVERY line the model carries; its start is the v2 excess estimate,
      held inside ``[p_hat - V3_SINK_INIT_DB, p_hat]``;
    * the floor SHAPE: the pooled mean with that start comb subtracted through
      the model, against the unit floor, read per control point as the median
      over the bins nearest it, about ``mu`` — the control values ``c_j`` —
      and ``sigma_B`` = their RMS (the real window's floor-shape spread, held
      at least :attr:`PriorsV3.floor_shape_sd_min_db`); ``z`` starts at the
      ridge solution of ``sigma_B L z = c`` (``L`` is too ill-conditioned for
      a plain inverse);
    * the dynamics start at their half-normal medians (no measurement: in
      flight the measured widths are resolution-limited).
    """
    if not isinstance(priors, MD.PriorsV3):
        raise TypeError(f"mode {MD.V3_MODE!r} needs PriorsV3, got {type(priors).__name__}")
    if batch.mode != "flight":
        raise ValueError(f"mode {MD.V3_MODE!r} is a flight mode; batch {batch.name!r} is not")
    probe_batch = replace(batch, latents=None)
    grid = batch.grid
    assert isinstance(grid, SP.FlightGrid)
    span_pinned = MD.span_pinned_sites(batch, priors=priors)
    band = batch.band.detach().cpu().numpy()
    obs_mean, floor_db = _observed_db(batch)
    t = lambda v: torch.as_tensor(np.asarray(v, dtype=np.float64), dtype=torch.float64)  # noqa: E731
    r, m, k = batch.n_rotors, batch.n_mics, batch.k_max
    orders = np.broadcast_to(np.arange(1, k + 1, dtype=np.float64), (r, k))
    gamma0 = priors.gamma_scale(orders).detach().cpu().numpy() * HALFNORMAL_MEDIAN
    sigma0 = float(priors.sigma_nu_scale) * HALFNORMAL_MEDIAN
    med = priors.speed_law_medians()
    zero = torch.zeros((), dtype=torch.float64)
    probe = SP.V2Params(
        sigma_nu=MD.pin_applied(pin, "sigma_nu", sigma0),
        lam=MD.pin_applied(pin, "lam", MD.flight_lam(priors, None)),
        gamma_hz=t(gamma0),
        profile_db=t(np.zeros((r, k))),
        floor=SP.FloorParams(
            mean_db=zero,
            shape_z=torch.zeros(FLOOR_SHAPE_N_CTRL, dtype=torch.float64),
            tilt_db_oct=zero,
            mic_floor_db=torch.zeros(m, dtype=torch.float64),
            exp=t(med["floor_exp"]),
            static_rel=t(med["floor_static_rel"]),
            shape_sd_db=t(1.0),
        ),
        mic_line_gain_db=torch.zeros(m, r, dtype=torch.float64),
        gain_all_db=torch.zeros(m, dtype=torch.float64),
        carrier_rev_s=None,
        amp_exp=t(med["amp_exp"]),
    )
    with torch.no_grad():
        quiet = MD.forward(probe_batch, _with_profile(probe, -300.0))
        unit = MD.forward(probe_batch, _with_profile(probe, 0.0))
        lines_unit = (unit - quiet).clamp_min(1e-30).mean(dim=(0, 1)).cpu().numpy()
        floor_unit = quiet.mean(dim=(0, 1)).cpu().numpy()

    # ── the floor level mu: v2's measurement, now a constant ────────────────
    resid = floor_db[band] - 10.0 * np.log10(np.maximum(floor_unit[band], 1e-30))
    mu = float(np.median(resid))

    # ── the comb: every line's pooled level and its start ──────────────────
    floor_lin = np.maximum(floor_unit, 1e-30) * 10.0 ** (mu / 10.0)
    excess = np.maximum(obs_mean - floor_lin, 1e-12)
    p_hat = np.full((r, k), mu, dtype=np.float64)
    prof = np.full((r, k), mu - V3_SINK_INIT_DB, dtype=np.float64)
    snr = np.full((r, k), -99.0, dtype=np.float64)
    for rr, kk, j0, j1 in _line_windows(batch, band=band, carrier=None, keep_all=True):
        den = float(np.max(lines_unit[j0:j1]))
        if den <= 0.0:
            continue
        total = float(np.max(obs_mean[j0:j1]))
        p_hat[rr, kk - 1] = float(np.clip(10.0 * math.log10(max(total, 1e-30) / den), -120.0, 40.0))
        num = float(np.max(excess[j0:j1]))
        x_db = 10.0 * math.log10(num / den)
        prof[rr, kk - 1] = float(
            np.clip(x_db, p_hat[rr, kk - 1] - V3_SINK_INIT_DB, p_hat[rr, kk - 1])
        )
        jpk = int(j0 + np.argmax(excess[j0:j1]))
        snr[rr, kk - 1] = 10.0 * math.log10(num / max(float(floor_lin[jpk]), 1e-30))

    # ── the floor SHAPE: control values and sigma_B ────────────────────────
    # read only where the FLOOR dominates: the pooled mean with the comb at
    # its start levels subtracted through the model (window response and
    # sidelobes included), on the bins where that comb predicts under
    # V3_FLOOR_COMB_SHARE of what was observed. A strong line lifts every bin
    # of a narrow control cell, even under a 20th-percentile curve, and a
    # subtraction AT the line leaves its own sampling noise, tens of dB over
    # the floor. Each control point reads the median over the kept bins it is
    # nearest to (its largest hat weight); a point with none is interpolated.
    with torch.no_grad():
        lines_x = MD.forward(probe_batch, replace(probe, profile_db=t(prof))) - quiet
        lines_x = lines_x.clamp_min(0.0).mean(dim=(0, 1)).cpu().numpy()
    keep = lines_x[band] < V3_FLOOR_COMB_SHARE * obs_mean[band]
    clean = np.maximum(obs_mean[band] - lines_x[band], 1e-30)
    shape_resid = 10.0 * np.log10(clean / np.maximum(floor_unit[band], 1e-30))
    nearest = np.argmax(floor_geometry(grid.freqs_hz[band], grid.floor.ctrl_hz)[0], axis=1)
    ctrl = np.full(FLOOR_SHAPE_N_CTRL, np.nan)
    for j in range(FLOOR_SHAPE_N_CTRL):
        cell = keep & (nearest == j)
        if np.any(cell):
            ctrl[j] = float(np.median(shape_resid[cell])) - mu
    have = np.isfinite(ctrl)
    ctrl = (
        np.interp(np.arange(ctrl.size), np.flatnonzero(have), ctrl[have])
        if np.any(have)
        else np.zeros(ctrl.size)
    )
    sigma_b = max(float(np.sqrt(np.mean(ctrl**2))), float(priors.floor_shape_sd_min_db))
    z0 = _floor_z(grid, sigma_b, ctrl)

    wind_db = _wind_centres(batch) if priors.wind else None

    ext_db: np.ndarray | None = None
    ext_sd: np.ndarray | None = None
    if profile_init is not None:
        ext_db, ext_sd = profile_init.aligned((r, k))
    measured = MD.Measured(
        floor_mean_db=mu,
        profile_db=p_hat,
        line_snr_db=snr,
        gamma_hz=gamma0,
        resolution_hz=batch.resolution_hz,
        profile_prior_db=(
            None
            if ext_db is None or ext_sd is None
            else np.where(np.isfinite(ext_sd), ext_db, np.nan)
        ),
        profile_prior_sd_db=ext_sd,
        floor_shape_sd_db=sigma_b,
        floor_ctrl_db=ctrl,
        wind_db=wind_db,
    )

    out: dict[str, Tensor] = {}
    if not MD.is_pinned(pin, "sigma_nu"):
        out["sigma_nu"] = t(sigma0)
    out["gamma_hz"] = t(gamma0)
    out["profile_db"] = t(prof if ext_db is None else np.where(np.isfinite(ext_db), ext_db, prof))
    if "amp_exp" not in span_pinned:
        out["amp_exp"] = t(priors.amp_exp[0])
    out["floor_shape_z"] = t(z0)
    if "floor_exp" not in span_pinned:
        out["floor_exp"] = t(med["floor_exp"])
    if "floor_static_rel" not in span_pinned:
        out["floor_static_rel"] = t(med["floor_static_rel"])
    if wind_db is not None:
        out["wind_db"] = t(wind_db)
    return Seeds(measured=measured, init=out)


def _floor_z(grid: SP.FlightGrid, sigma_b: float, ctrl_db: np.ndarray) -> np.ndarray:
    """The ``floor_shape_z`` whose spline ``sigma_B (L z)`` reads ``ctrl_db`` at
    the control points: the ridge solution of ``sigma_B L z = c`` (``L`` is
    too ill-conditioned for a plain inverse; the ridge is the ``N(0, I)``
    prior's own)."""
    chol = grid.floor.shape_chol.detach().cpu().numpy()
    return np.linalg.solve(
        sigma_b**2 * (chol.T @ chol) + np.eye(chol.shape[1]),
        sigma_b * (chol.T @ np.asarray(ctrl_db, dtype=np.float64)),
    )


#: A warm-started site is clipped into its v3 prior's central ``1 - 2 q``
#: (``q .. 1 - q`` quantiles): a v2 value the v3 prior calls impossible — a
#: v2 width parked at 10^5 x its bench law — is a start the v3 fit would first
#: have to climb out of.
WARM_START_PRIOR_Q = 1e-3

#: Adam steps a warm-started v3 fit runs before its L-BFGS polish (the CLI's
#: ``--adam-steps`` default with ``--init-from``): the start is a MAP of the
#: same data, not a measurement.
WARM_START_ADAM_STEPS = 300


@dataclass(frozen=True)
class V2Fit:
    """A v2 fit of the same pool, the warm start of a v3 fit (``--init-from``)."""

    path: str
    payload: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> V2Fit:
        d = json.loads(Path(path).read_text())
        if d.get("schema") not in (FIT_SCHEMA, "noise-v2-fit/1"):
            raise ValueError(f"{path}: not a v2 fit (schema {d.get('schema')!r})")
        if d.get("kind") != "flight":
            raise ValueError(f"{path}: a v3 warm start needs a FLIGHT fit, got {d.get('kind')!r}")
        return cls(path=str(path), payload=d)


def _prior_box(
    batch: MD.SupportBatch, *, priors: MD.PriorsV3, pin: dict[str, Any] | None, q: float
) -> dict[str, tuple[Tensor, Tensor]]:
    """``{site: (lo, hi)}``: every v3 rig site's prior ``q`` and ``1 - q``
    quantiles, read off the very distributions the model samples."""
    boxes: dict[str, tuple[Tensor, Tensor]] = {}

    def collect(name: str, d: Any) -> Tensor:
        base = d
        while isinstance(base, torch.distributions.Independent):
            base = base.base_dist
        at = lambda p: torch.full(  # noqa: E731
            tuple(base.batch_shape), p, dtype=torch.float64, device=batch.power.device
        )
        boxes[name] = (base.icdf(at(q)), base.icdf(at(1.0 - q)))
        return boxes[name][0]

    MD.sample_params(batch, mode=MD.V3_MODE, priors=priors, pin=pin, site=collect)
    return boxes


def warm_start_v3(
    v2: V2Fit,
    *,
    batch: MD.SupportBatch,
    start: Seeds,
    priors: MD.PriorsV3,
    pin: dict[str, Any] | None = None,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """The v3 guide's start from a v2 fit of the same pool, and its record.

    ``start`` is the v3 measurement of ``batch`` (the prior centres stay
    v3's own); its ``init`` is replaced site by site:

    * ``sigma_nu``, ``gamma_hz`` (the first ``K`` orders; orders the v2 fit
      does not carry keep the measured start), ``profile_db`` likewise, and
      the speed laws v3 does not pin — straight from the v2 ``params``;
    * ``floor_shape_z`` RE-EXPRESSED: v3's floor is the spline alone about
      the measured ``mu`` with the measured ``sigma_B``, so the v2 floor's
      microphone-mean dB curve at the control points (``floor_mean_db +
      mean(mic_floor_db) + 6 (L z) + tilt log2(f_j / f_ref)``) minus ``mu``
      is solved for ``z`` exactly as the measurement's own start is — and
      KEPT only if the objective there beats the measured floor's under the
      same warm comb: a v2 floor can sit tens of dB under the data where its
      over-wide lines (v2's widths reach 10^5 x the bench law) made a
      broadband pedestal the v3 width prior clips away;
    * ``lam`` is not a v3 site (pinned in flight) and is only recorded;
    * every site clipped into its v3 prior's :data:`WARM_START_PRIOR_Q`
      central box (:func:`_prior_box`).
    """
    p = v2.payload["params"]
    r, k = batch.n_rotors, batch.k_max
    grid = batch.grid
    assert isinstance(grid, SP.FlightGrid) and start.measured.floor_shape_sd_db is not None
    t = lambda v: torch.as_tensor(np.asarray(v, dtype=np.float64), dtype=torch.float64)  # noqa: E731

    def per_line(v2_rk: np.ndarray, site: str) -> Tensor:
        got = np.atleast_2d(np.asarray(v2_rk, dtype=np.float64))
        if got.shape[0] != r:
            raise ValueError(f"{v2.path}: {site} has {got.shape[0]} rotors, the pool {r}")
        out = start.init[site].detach().cpu().numpy().copy()
        kk = min(k, int(got.shape[1]))
        out[:, :kk] = got[:, :kk]
        return t(out)

    v2_sites: dict[str, Tensor] = dict(
        sigma_nu=t(p["sigma_nu"]),
        gamma_hz=per_line(MD.gamma_from_params(p), "gamma_hz"),
        profile_db=per_line(p["profile"]["profile_db"], "profile_db"),
        amp_exp=t(p["profile"]["amp_exp"]),
        floor_exp=t(p["floor"]["floor_exp"]),
        floor_static_rel=t(p["floor"]["floor_static_rel"]),
    )
    fl = p["floor"]
    ctrl_hz = np.asarray(grid.floor.ctrl_hz, dtype=np.float64)
    v2_ctrl = (
        float(fl["floor_mean_db"])
        + float(np.mean(np.asarray(fl["mic_floor_db"], dtype=np.float64)))
        + grid.floor.shape_db(t(fl["floor_shape_z"])).detach().cpu().numpy()
        + float(fl["floor_tilt_db_oct"]) * np.log2(ctrl_hz / FLOOR_TILT_REF_HZ)
    )
    sigma_b = float(start.measured.floor_shape_sd_db)
    v2_sites["floor_shape_z"] = t(
        _floor_z(grid, sigma_b, v2_ctrl - float(start.measured.floor_mean_db))
    )

    boxes = _prior_box(batch, priors=priors, pin=pin, q=WARM_START_PRIOR_Q)
    init = dict(start.init)
    clipped: dict[str, int] = {}
    for name, value in v2_sites.items():
        if name not in init:  # pinned (a span pin or --pin): no site to start
            continue
        lo, hi = (b.detach().cpu() for b in boxes[name])
        init[name] = torch.maximum(torch.minimum(value, hi), lo)
        clipped[name] = int((init[name] != value).sum())
    at_v2_floor = _objective_at(batch, init, priors=priors, pin=pin)
    at_measured_floor = _objective_at(
        batch, dict(init, floor_shape_z=start.init["floor_shape_z"]), priors=priors, pin=pin
    )
    floor_from = "v2 re-expressed" if at_v2_floor <= at_measured_floor else "measured"
    if floor_from == "measured":
        init["floor_shape_z"] = start.init["floor_shape_z"]
    record = dict(
        source=v2.path,
        schema=v2.payload.get("schema"),
        mode=v2.payload.get("mode"),
        git=v2.payload.get("git"),
        sites=sorted(n for n in clipped if n != "floor_shape_z" or floor_from != "measured"),
        clipped=clipped,
        prior_box_quantiles=[WARM_START_PRIOR_Q, 1.0 - WARM_START_PRIOR_Q],
        floor_start=floor_from,
        objective_at_v2_floor=at_v2_floor,
        objective_at_measured_floor=at_measured_floor,
        objective_at_measured_start=_objective_at(batch, start.init, priors=priors, pin=pin),
        v2_floor_ctrl_db=np.round(v2_ctrl - float(start.measured.floor_mean_db), 3).tolist(),
        not_v3_sites=dict(
            lam=p.get("lam"), note="lam is pinned in v3 flight; v2 mic gains are data-normalised"
        ),
    )
    return init, record


def _objective_at(
    batch: MD.SupportBatch,
    values: dict[str, Tensor],
    *,
    priors: MD.PriorsV3,
    pin: dict[str, Any] | None,
) -> float:
    """``Whittle - log p(sites)`` of ``batch`` at site ``values`` (no gradient)."""
    dev = batch.power.device
    vals = {k: v.to(dev) for k, v in values.items()}
    kw: dict[str, Any] = dict(mode=MD.V3_MODE, priors=priors, pin=pin, values=vals)
    with torch.no_grad():
        nlp = -MD.log_prior_tensor(batch, **kw)
        m_model = MD.forward(batch, MD.sample_params_from_values(batch, **kw))
        return float(nlp + MD.whittle_risk(batch, m_model))


#: How far under the quietest microphone's low-band level a v3 wind prior
#: centre may sink: the quietest capsule's wind is not measurable against the
#: shared floor, so it starts (and is centred) this far under it.
V3_WIND_SINK_DB = 20.0


def _wind_centres(batch: MD.SupportBatch) -> np.ndarray:
    """``(M,)`` prior centres of the per-mic wind levels ``wind_db`` (dB).

    Each mic's low band level ``l_m`` is the median, over the analysis bins
    from 30 Hz to the wind shape's flat-part corner (100 Hz), of its OWN
    20th-percentile floor curve (+ the C4 exponential-percentile correction)
    — on the channel-NORMALISED data, so what is left between mics is the
    per-capsule excess the rank test found below 500 Hz. The shared floor
    cannot sit above the quietest mic, so a mic's measured wind is its POWER
    excess over that one: ``10 log10(10^{l_m/10} - 10^{l_min/10})``, and no
    lower than :data:`V3_WIND_SINK_DB` under ``l_min``.
    """
    p = batch.power.detach().cpu().numpy()
    f = np.asarray(batch.grid.freqs_hz, dtype=np.float64)
    low = (f >= SP.BAND_F_MIN) & (f <= WIND_CORNER_HZ)
    if not np.any(low):
        raise ValueError("the analysis grid has no bin between 30 Hz and the wind corner")
    db = 10.0 * np.log10(np.maximum(p[:, :, low], 1e-30))
    per_mic = np.quantile(db, FLOOR_INIT_QUANTILE, axis=1) + FLOOR_INIT_DB_CORRECTION  # (M, F_low)
    level = np.median(per_mic, axis=1)
    ref = float(level.min())
    excess = 10.0 ** (level / 10.0) - 10.0 ** (ref / 10.0)
    return np.maximum(10.0 * np.log10(np.maximum(excess, 1e-300)), ref - V3_WIND_SINK_DB)


def load_channel_gains(
    path: str | Path, *, rig: str, mics: Sequence[int] | None = None
) -> MD.ChannelGains:
    """The per-channel gains v3 normalises ``rig``'s data by (explainer §2.5).

    Read from the rank-test record (``mic-gain-rank/1``,
    ``results/noise_v2/mic_gains/mic_gains.json``): the rig's rank-one FLOOR
    gains on the 1/3-octave bands at and above 500 Hz
    (``rigs.<rig>.wind.excess_db_above_500hz``) — where comb and floor share
    one gain per channel on both rigs (r = 0.990 DREGON, 0.996 Michael's) and
    below which DREGON's per-capsule wind excess lives. ``mics`` selects
    channels (a smoke fit on a subset); the selected gains are re-centred on
    their own mean, so the normalisation moves no absolute level.
    """
    p = Path(path)
    d = json.loads(p.read_text())
    if d.get("schema") != "mic-gain-rank/1":
        raise ValueError(f"{p}: expected schema 'mic-gain-rank/1', got {d.get('schema')!r}")
    rigs = d.get("rigs") or {}
    if rig not in rigs:
        raise KeyError(f"{p}: no rig {rig!r} (have {sorted(rigs)})")
    g = np.asarray(rigs[rig]["wind"]["excess_db_above_500hz"], dtype=np.float64)
    if mics is not None:
        g = g[np.asarray(list(mics), dtype=np.int64)]
    return MD.ChannelGains(
        gains_db=g - g.mean(),
        source=str(p),
        rig=str(rig),
        rule="rank-one floor gain per channel on the 1/3-octave bands >= 500 Hz "
        "(rigs.<rig>.wind.excess_db_above_500hz), re-centred on the channels used; "
        "each channel's periodogram is divided by 10^(g/10)",
    )


def _half_width_hz(excess: np.ndarray, jpk: int, j0: int, j1: int, df: float) -> float:
    """The measured HALF width at half maximum of the peak at ``jpk``, in Hz.

    Walked outwards over the observed line excess rather than fitted: the
    quantity is an INITIALISATION for ``gamma_rk``, and a three-parameter
    Lorentzian fit per line on 130 orders would cost more than the fit it
    seeds. The walk stops at the line window's own edges, so a width can never
    run into the neighbouring order.
    """
    half = 0.5 * float(excess[jpk])
    lo = jpk
    while lo > j0 and float(excess[lo - 1]) >= half:
        lo -= 1
    hi = jpk
    while hi < j1 - 1 and float(excess[hi + 1]) >= half:
        hi += 1
    return 0.5 * float(hi - lo + 1) * float(df)


def _line_windows(
    batch: MD.SupportBatch, *, band: np.ndarray, carrier: Tensor | None, keep_all: bool = False
) -> list[tuple[int, int, int, int]]:
    """``(rotor, order, j0, j1)`` per comb line: the bins that line covers.

    The window spans where the line went over the batch, plus the analysis
    window's own main lobe: a moving carrier smears the frame-mean peak across
    ``k * (f_max - f_min)``. Lines wholly outside the fitted band are dropped
    unless ``keep_all`` (v3 centres a prior on EVERY line it models).
    """
    if carrier is not None:
        lo_r = hi_r = np.atleast_1d(np.asarray(carrier.cpu(), dtype=np.float64))
    else:
        assert batch.rate_work is not None
        rw = batch.rate_work.detach()
        lo_r = rw.amin(dim=(1, 2)).cpu().numpy()
        hi_r = rw.amax(dim=(1, 2)).cpu().numpy()
    df = float(batch.grid.freqs_hz[1] - batch.grid.freqs_hz[0])
    f_top = float(batch.grid.freqs_hz[band].max())
    n_bins = int(np.asarray(batch.grid.freqs_hz).size)
    out: list[tuple[int, int, int, int]] = []
    for r in range(int(lo_r.size)):
        for k in range(1, batch.k_max + 1):
            f_lo, f_hi = k * float(lo_r[r]), k * float(hi_r[r])
            if not keep_all and (f_hi < SP.BAND_F_MIN or f_lo > f_top):
                continue
            j0 = min(max(0, int(math.floor(f_lo / df)) - 3), n_bins - 1)
            j1 = max(min(n_bins, int(math.ceil(f_hi / df)) + 4), j0 + 1)
            out.append((r, k, j0, j1))
    return out


def _seed_params(
    batch: MD.SupportBatch,
    *,
    mode: str,
    priors: MD.Priors,
    frozen: dict[str, Any] | None,
    carrier: Tensor | None,
    pin: dict[str, Any] | None = None,
) -> SP.V2Params:
    """A concrete parameter set at the priors' centres (no Pyro site involved).

    Used ONLY by :func:`initial_values` for its probe forward passes; a
    frozen block uses its frozen value so a ``--floor-only`` seed is measured
    against the real comb rather than against a prior-mean one, and a PINNED
    dynamics coordinate uses its pinned value so the profile/floor seeds are
    calibrated at the dynamics the fit will actually run at.
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
    gamma_centre = float(priors.gamma_per_order_hz) * np.broadcast_to(
        np.arange(1, k + 1, dtype=np.float64), (r, k)
    )
    return SP.V2Params(
        sigma_nu=(
            MD.pin_applied(pin, "sigma_nu", math.exp(priors.sigma_nu_prior(flight)[0]))
            if "dynamics" in free
            else pick("dynamics", "sigma_nu", math.exp(priors.sigma_nu_prior(flight)[0]))
        ),
        lam=(
            # in flight the rate is a CONSTANT: the frozen bench value if a
            # mapping carries one, otherwise the approved pin
            MD.pin_applied(
                pin,
                "lam",
                MD.flight_lam(priors, fz) if flight else math.exp(priors.log_lam[0]),
            )
            if "dynamics" in free
            else pick("dynamics", "lam", math.exp(priors.log_lam[0]))
        ),
        gamma_hz=pick("dynamics", "gamma_hz", gamma_centre, (r, k)),
        profile_db=pick("profile", "profile_db", np.zeros((r, k)), (r, k)),
        floor=SP.FloorParams(
            mean_db=pick("floor", "floor_mean_db", 0.0),
            shape_z=pick(
                "floor", "floor_shape_z", np.zeros(FLOOR_SHAPE_N_CTRL), (FLOOR_SHAPE_N_CTRL,)
            ),
            tilt_db_oct=pick("floor", "floor_tilt_db_oct", 0.0),
            mic_floor_db=pick("floor", "mic_floor_db", np.zeros(m), (m,)),
            exp=pick("floor", "floor_exp", math.exp(priors.log_floor_exp[0])) if flight else zero,
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
    )


def _with_profile(params: SP.V2Params, db: float) -> SP.V2Params:
    import dataclasses

    prof = torch.full_like(torch.as_tensor(params.profile_db, dtype=torch.float64), float(db))
    return dataclasses.replace(params, profile_db=prof)


# ── the fit ─────────────────────────────────────────────────────────────────


def _grad_norm(params: Sequence[Tensor]) -> float:
    return float(math.sqrt(sum(float((p.grad**2).sum()) for p in params if p.grad is not None)))


def _sync(device: torch.device) -> None:
    """Wait for the device's queued work, so a wall clock reads it."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_mb(device: torch.device, *, reset: bool = False) -> float | None:
    """The CUDA allocator's peak since the last reset, in MiB (``None`` off CUDA)."""
    if device.type != "cuda":
        return None
    peak = torch.cuda.max_memory_allocated(device) / 2.0**20
    if reset:
        torch.cuda.reset_peak_memory_stats(device)
    return float(peak)


#: Half-width of the box a POSITIVE site's unconstrained coordinate is held in
#: during the L-BFGS polish. A strong-Wolfe probe extrapolates before it
#: brackets, and one such probe on ``bench_dregon_Motor3_70`` pushed
#: ``log sigma_nu`` far enough negative that ``exp`` underflowed to exactly
#: 0.0 — outside the site's support, which raises inside ``log_prob`` and kills
#: the fit. e^±60 spans 9e-27 to 1e26, so the box cannot bind on any plausible
#: optimum; it only stops a runaway probe, and the polish records whether any
#: coordinate ended up against it.
LOG_BOX = 60.0


def _log_space_params(guide: AutoDelta) -> list[Tensor]:
    """The guide tensors that live in log space (their site is positive).

    ``AutoDelta`` names each parameter ``<site>_unconstrained`` and transforms
    it with the site's own bijector, so a site supported on ``(0, inf)`` — the
    four dynamics sites — is the exponential of its parameter, and only those
    can underflow out of their support.
    """
    trace = guide.prototype_trace
    if trace is None:
        raise RuntimeError("the guide must have run before its log-space sites are known")
    out: list[Tensor] = []
    for name, p in guide.named_parameters():
        node = trace.nodes.get(name.removesuffix("_unconstrained"))
        support = getattr(node.get("fn"), "support", None) if node is not None else None
        base = getattr(support, "base_constraint", support)
        if isinstance(base, constraints.greater_than) and float(base.lower_bound) == 0.0:
            out.append(p)
    return out


def _profile_init_record(
    profile_init: ProfileInit | None, measured: MD.Measured
) -> dict[str, Any] | None:
    """What the fit JSON says about an external profile: where it came from and
    how many lines it actually moved."""
    if profile_init is None:
        return None
    loc, sd = measured.profile_prior_db, measured.profile_prior_sd_db
    off = profile_init.carrier_offset_rev_s
    return dict(
        source=profile_init.source,
        n_centres=0 if loc is None else int(np.isfinite(loc).sum()),
        n_widths=0 if sd is None else int(np.isfinite(sd).sum()),
        carrier_offset_rev_s=None if off is None else [float(v) for v in off],
    )


def chunked_objective(
    batch: MD.SupportBatch,
    guide: AutoDelta,
    *,
    mode: str,
    priors: MD.Priors = MD.PRIORS,
    frozen: dict[str, Any] | None = None,
    pin: dict[str, Any] | None = None,
    low_orders: int | None = None,
    forward_kw: dict[str, Any] | None = None,
    grad: bool,
) -> Tensor:
    """``Whittle - log p(sites)`` of ``batch`` at ``guide``'s point — the
    ``Trace_ELBO`` loss an ``AutoDelta`` guide minimises — with the Whittle
    term summed over :func:`.model.frame_chunks`.

    With ``grad`` each chunk's gradient is accumulated (into the site values,
    then once through the guide's transforms into its parameters' ``.grad``)
    and the chunk's graph released before the next one is built: every frame
    counts in every evaluation while the kernel's autograd memory is one
    chunk's. Returns the objective, detached.
    """
    site_kw: dict[str, Any] = dict(
        mode=mode, priors=priors, frozen=frozen, pin=pin, low_orders=low_orders
    )
    with torch.set_grad_enabled(grad):
        values = guide()
        leaves = {k: v.detach().requires_grad_(grad) for k, v in values.items()}
        nlp = -MD.log_prior_tensor(batch, values=leaves, **site_kw)
        if grad:
            nlp.backward()
        total = nlp.detach()
        for sub in MD.frame_chunks(batch):
            params = MD.sample_params_from_values(sub, values=leaves, **site_kw)
            whittle = MD.whittle_risk(sub, MD.forward(sub, params, **(forward_kw or {})))
            if grad:
                whittle.backward()
            total = total + whittle.detach()
        if grad:
            names = [k for k, v in leaves.items() if v.grad is not None]
            torch.autograd.backward([values[k] for k in names], [leaves[k].grad for k in names])
    return total


def fit_support(
    batch: MD.SupportBatch,
    *,
    mode: str,
    priors: MD.Priors = MD.PRIORS,
    frozen: dict[str, Any] | None = None,
    pin: dict[str, Any] | None = None,
    low_orders: int | None = None,
    optim: OptimSpec = OptimSpec(),
    forward_kw: dict[str, Any] | None = None,
    profile_init: ProfileInit | None = None,
    progress: int = 0,
    start: Seeds | None = None,
) -> FitOutcome:
    """MAP-fit one support (or one pooled set of flight windows).

    The support is MEASURED first (:func:`seeds`): the R3 profile and floor
    priors are centred on the data, so the measurement is attached to the
    batch the model is built on and travels with every minibatch of it.

    ``pin`` (in :func:`model.dynamics_pin`'s spelling) holds a named dynamics
    scalar FIXED while the rest of the block is fitted. It is a constant of
    the model, not a parameter under a tight prior: the guide allocates
    nothing for it, the log-prior counts nothing for it, and the recorded
    ``params`` carry the pinned value.

    ``profile_init`` enters the profile's INITIALISATION and, where it carries
    a width, its prior centre and sd (:class:`ProfileInit`).

    ``start`` replaces the measurement pass: its ``measured`` is the prior
    centres (so a refit keeps the priors of the fit it continues) and its
    ``init`` the guide's start (a warm start). The v3 alternation
    (:func:`fit_v3`) refits the rig this way with the latents held on the
    batch; ``None`` measures ``batch`` as always.
    """
    torch.manual_seed(int(optim.seed))
    pyro.set_rng_seed(int(optim.seed))
    pyro.clear_param_store()

    measured = (
        start
        if start is not None
        else seeds(
            batch,
            mode=mode,
            priors=priors,
            frozen=frozen,
            pin=pin,
            low_orders=low_orders,
            profile_init=profile_init,
        )
    )
    full = replace(batch, measured=measured.measured)
    dev = full.power.device
    init = {k: v.to(dev) for k, v in measured.init.items()}
    if optim.init_jitter > 0.0:
        # a log-normal multi-start on the dynamics block only: the profile,
        # floor and carrier initialisations are read off the data and a random
        # start for them would test the initialiser, not the landscape
        jrng = np.random.default_rng(int(optim.seed))
        for key in ("sigma_nu", "lam", "gamma_hz"):
            if key in init:
                v = init[key]
                shift = jrng.normal(0.0, float(optim.init_jitter), size=tuple(v.shape))
                init[key] = v * torch.as_tensor(np.exp(shift), dtype=torch.float64, device=dev)

    def model_for(b: MD.SupportBatch) -> Any:
        def fn() -> Any:
            return MD.support_model(
                b,
                mode=mode,
                priors=priors,
                frozen=frozen,
                pin=pin,
                low_orders=low_orders,
                temperature=1.0,
                forward_kw=forward_kw,
            )

        return fn

    # the frame-CHUNKED objective (``batch.chunk_frames``, flight): the ELBO's
    # MAP objective with its Whittle term accumulated chunk by chunk
    # (:func:`chunked_objective`)
    chunked = full.rate_work is not None and full.chunk_frames is not None

    # CLONE into the guide. ``AutoDelta`` keeps the very tensors ``init_to_value``
    # hands it as its own (unconstrained) parameters and then updates them IN
    # PLACE, so passing ``init`` itself would make the recorded
    # ``diagnostics.init_*`` read the FITTED value and report every fit as one
    # that never left its initialisation. A chunked fit builds the guide's
    # prototype on one chunk: the sites do not depend on the frame count.
    guide = AutoDelta(
        model_for(MD.frame_chunks(full)[0] if chunked else full),
        init_loc_fn=init_to_value(values={k: v.detach().clone() for k, v in init.items()}),
    )
    elbo = Trace_ELBO()

    def chunked_loss(b: MD.SupportBatch, *, grad: bool) -> Tensor:
        return chunked_objective(
            b,
            guide,
            mode=mode,
            priors=priors,
            frozen=frozen,
            pin=pin,
            low_orders=low_orders,
            forward_kw=forward_kw,
            grad=grad,
        )

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
    adam_chunked: torch.optim.Adam | None = None
    if chunked and int(optim.adam_steps) > 0:
        with torch.no_grad():
            guide()
        adam_chunked = torch.optim.Adam(
            [p for p in guide.parameters() if p.requires_grad], lr=float(optim.adam_lr)
        )
    for step in range(int(optim.adam_steps)):
        if use_batches:
            assert optim.adam_batch is not None
            idx = rng.choice(n_frames, size=int(optim.adam_batch), replace=False)
            sub = MD.batch_slice(full, idx)
            if adam_chunked is not None:
                adam_chunked.zero_grad(set_to_none=False)
                loss = float(chunked_loss(sub, grad=True))
                adam_chunked.step()
            else:
                # Preserve Adam's moments across frame minibatches.  Recreating a
                # Pyro optimiser here silently converts 1,500 Adam steps into 1,500
                # independent first steps.
                loss = SVI(model_for(sub), guide, adam_optim, loss=elbo).step()
        elif adam_chunked is not None:
            adam_chunked.zero_grad(set_to_none=False)
            loss = float(chunked_loss(full, grad=True))
            adam_chunked.step()
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
        if optim.lbfgs_stratified and full.frame_window is not None:
            idx = MD.stratified_frames(full.frame_window, int(optim.lbfgs_frames))
        else:
            idx = np.linspace(0, n_frames - 1, int(optim.lbfgs_frames)).round().astype(np.int64)
        polish = MD.batch_slice(full, np.unique(idx))
    loss_fn = model_for(polish)

    def elbo_of() -> Tensor:
        if chunked:
            return chunked_loss(polish, grad=False)
        return elbo.differentiable_loss(loss_fn, guide)

    if not adam_losses:
        # a warm-started refit may skip Adam: the guide must still have run
        # once (on the polish set) before it has any parameter to polish
        elbo_of().detach()
    params = [p for p in guide.parameters() if p.requires_grad]
    boxed = _log_space_params(guide)

    def box() -> None:
        """Hold every log-space coordinate inside ``LOG_BOX``."""
        with torch.no_grad():
            for p in boxed:
                p.clamp_(-LOG_BOX, LOG_BOX)

    def loss_and_grad() -> Tensor:
        """The polish objective, its gradient accumulated into ``params``."""
        if chunked:
            return chunked_loss(polish, grad=True)
        loss = elbo_of()
        loss.backward()
        return loss

    traces: list[list[float]] = []

    def run_lbfgs(max_iter: int) -> tuple[float, int, int]:
        """One L-BFGS pass; returns its final loss, its evaluation count and
        the iterations it ran (``max_iter`` unless a tolerance stopped it).
        Its per-iteration objective goes to ``traces``."""
        opt = LBFGS(
            params,
            rtol=optim.lbfgs_rtol,
            max_iter=int(max_iter),
            history_size=int(optim.lbfgs_history),
            line_search_fn="strong_wolfe",
        )
        count = 0

        def closure() -> Tensor:
            nonlocal count
            # the probe point, not just the accepted step: strong Wolfe
            # extrapolates first and it is the EXTRAPOLATION that underflows
            box()
            opt.zero_grad(set_to_none=False)
            loss = loss_and_grad()
            count += 1
            return loss

        opt.step(closure)
        traces.append(list(opt.trace))
        # L-BFGS leaves the parameters at its own accepted point, which the
        # closure never saw
        box()
        with torch.no_grad():
            return float(elbo_of()), count, opt.n_iter

    t1 = time.time()
    with torch.no_grad():
        before = float(elbo_of())
    first_pass, evals, iters = run_lbfgs(int(optim.lbfgs_iters))
    # THE convergence test, not a threshold on a trace: L-BFGS is RESTARTED
    # with a fresh Hessian approximation and half the budget. If a restart
    # cannot find more than ``tol_nats_per_cell`` per observed cell, the first
    # pass had converged; if it can, it had not, and the fit says so.
    restart, restart_evals, restart_iters = run_lbfgs(max(1, int(optim.lbfgs_iters) // 2))
    lbfgs_s = time.time() - t1
    for p in params:
        p.grad = None
    loss_and_grad()
    gnorm = _grad_norm(params)
    gain_per_cell = max(0.0, first_pass - restart) / max(1, polish.n_cells)
    converged = bool(np.isfinite(restart) and gain_per_cell < float(optim.tol_nats_per_cell))
    final_loss = min(first_pass, restart)

    med = guide.median()
    fitted = MD.sample_params_from_values(
        full, mode=mode, priors=priors, frozen=frozen, pin=pin, low_orders=low_orders, values=med
    )
    with torch.no_grad():
        m_model = MD.forward(full, fitted, **(forward_kw or {}))
        objective = MD.objective_breakdown(full, m_model)
    return FitOutcome(
        params=fitted,
        sites={name: v.detach().clone() for name, v in med.items()},
        comb_gain_db=(float(med["comb_gain_db"]) if "comb_gain_db" in med else None),
        low_order_gain_db=(
            np.asarray(med["low_order_gain_db"].detach().cpu(), dtype=np.float64).tolist()
            if "low_order_gain_db" in med
            else None
        ),
        objective=objective,
        optimiser=dict(
            **optim.as_dict(),
            pinned_dynamics=(dict(pin) if pin else None),
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
            lbfgs_iters_used=int(iters),
            lbfgs_restart_iters_used=int(restart_iters),
            lbfgs_trace_first_pass=traces[0],
            lbfgs_trace_restart=traces[1],
            log_box_hits=int(sum(int((p.detach().abs() >= LOG_BOX - 1e-9).sum()) for p in boxed)),
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
            init_comb_gain_db=float(init["comb_gain_db"]) if "comb_gain_db" in init else None,
            init_gamma_hz=(
                # ``init`` lives on the fit's device: bring it home before numpy
                np.asarray(init["gamma_hz"].detach().cpu(), dtype=np.float64).tolist()
                if "gamma_hz" in init
                else None
            ),
            gamma_low_order_check=gamma_low_order_check(fitted.gamma_hz, batch=full),
            span_pins=span_pin_record(full, priors=priors),
            profile_init=_profile_init_record(profile_init, measured.measured),
            measured=_measured_record(measured.measured, priors=priors),
        ),
    )


def _measured_record(measured: MD.Measured, *, priors: MD.Priors) -> dict[str, Any]:
    """The ``diagnostics.measured`` block: what the prior centres were read off."""
    if isinstance(priors, MD.PriorsV3):
        return dict(
            floor_mean_db=measured.floor_mean_db,
            floor_shape_sd_db=measured.floor_shape_sd_db,
            floor_ctrl_db=(
                None
                if measured.floor_ctrl_db is None
                else np.asarray(measured.floor_ctrl_db, dtype=np.float64).tolist()
            ),
            wind_db=(
                None
                if measured.wind_db is None
                else np.asarray(measured.wind_db, dtype=np.float64).tolist()
            ),
            resolution_hz=measured.resolution_hz,
            n_lines=int(measured.line_snr_db.size),
            profile_centre="pooled observed level at k f_r (line + floor), every line",
            # the (R, K) centres themselves: the prior predictive and the
            # parameter view draw / plot the profile against them
            profile_centre_db=np.asarray(measured.profile_db, dtype=np.float64).tolist(),
        )
    return dict(
        floor_mean_db=measured.floor_mean_db,
        resolution_hz=measured.resolution_hz,
        n_lines_visible=int((measured.line_snr_db >= priors.line_visible_snr_db).sum()),
        n_lines=int(measured.line_snr_db.size),
        line_visible_snr_db=priors.line_visible_snr_db,
    )


# ── v3: the alternation ─────────────────────────────────────────────────────


def fit_latents(
    batch: MD.SupportBatch,
    params: SP.V2Params,
    *,
    wander: MD.Wander,
    init: dict[int, MD.WindowLatents],
    iters: int = 100,
    history: int = 10,
    line_dtype: torch.dtype = torch.float64,
) -> tuple[dict[int, MD.WindowLatents], dict[str, Any]]:
    """Step (ii): EVERY window's block latents with the rig held FIXED.

    ``batch`` is the pool and ``params`` the rig as constants; ONE
    ``AutoDelta`` over the pooled ``wander_*`` sites of
    :func:`.model.latent_model` (a ``windows`` plate, one ``(W, ..., B)`` site
    per latent), started at ``init`` and polished by ONE strong-Wolfe L-BFGS on
    the summed objective — the windows are independent given the rig, so this
    is every window's own MAP at once, with no Python loop over windows. With
    ``sigma, tau`` fixed it is the correctly-shrunk MAP of the explainer's
    §3.3a — a Kalman smoother of the block amplitudes, no variance to cheat
    with. The rig's line and floor spectra are computed ONCE
    (:func:`.model.latent_cache`, lines stored in ``line_dtype``): an
    evaluation only moves their block multipliers.
    """
    t0 = time.time()
    windows = tuple(sorted(init))
    n_blocks = tuple(int(batch.window_blocks[w]) for w in windows)
    values = MD.stack_latents(init, windows, max(n_blocks))
    rec: dict[str, Any] = dict(
        windows=len(windows), n_blocks=list(n_blocks), line_dtype=str(line_dtype)
    )
    if not values:
        return init, dict(rec, tracks=[], note="no active wander track")
    pyro.clear_param_store()
    cache = MD.latent_cache(batch, params, line_dtype=line_dtype)
    if cache.windows != windows:
        raise ValueError(f"latents for windows {windows}, batch has {cache.windows}")
    _sync(batch.power.device)
    cache_s = time.time() - t0

    def model() -> Any:
        return MD.latent_model(batch, cache, wander=wander)

    guide = AutoDelta(model, init_loc_fn=init_to_value(values=values))
    elbo = Trace_ELBO()
    before = float(elbo.differentiable_loss(model, guide).detach())
    opt_params = [p for p in guide.parameters() if p.requires_grad]
    opt = torch.optim.LBFGS(
        opt_params, max_iter=int(iters), history_size=int(history), line_search_fn="strong_wolfe"
    )
    count = 0

    def closure() -> Tensor:
        nonlocal count
        opt.zero_grad(set_to_none=False)
        loss = elbo.differentiable_loss(model, guide)
        loss.backward()
        count += 1
        return loss

    t1 = time.time()
    opt.step(closure)
    lbfgs_s = time.time() - t1
    with torch.no_grad():
        after = float(elbo.differentiable_loss(model, guide))
    out = MD.unstack_latents(guide.median(), windows, n_blocks)
    return out, dict(
        rec,
        tracks=sorted(k.removeprefix("wander_") for k in values),
        loss_before=before,
        loss_after=after,
        evals=count,
        cache_s=cache_s,
        s_per_eval=lbfgs_s / max(1, count),
        wall_s=time.time() - t0,
    )


def _priors_nats(
    batch: MD.SupportBatch,
    sites: dict[str, Tensor],
    latents: dict[int, MD.WindowLatents],
    *,
    priors: MD.PriorsV3,
    pin: dict[str, Any] | None,
) -> float:
    """Rig ``-log prior`` at ``sites`` plus the OU ``-log prior`` of ``latents``."""
    assert priors.wander is not None
    rig = -MD.log_prior(
        replace(batch, latents=latents), mode=MD.V3_MODE, values=sites, priors=priors, pin=pin
    )
    return rig + MD.ou_prior_nats(latents, priors.wander)


def _ou_shift_coeffs(
    x: np.ndarray, sigma: np.ndarray, rho: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(g, h)`` per track of the stationary OU quadratic under a constant
    shift: ``Q(x - c) = Q(x) - c g + c^2 h / 2`` (last axis = blocks).

    ``Q`` is quadratic, so three evaluations give ``g`` and ``h`` exactly;
    a track with ``sigma`` 0 (pinned) gives zeros."""

    def q(y: np.ndarray) -> np.ndarray:
        s = np.broadcast_to(sigma, y.shape[:-1])
        r = np.broadcast_to(rho, y.shape[:-1])
        innov = y[..., 1:] - r[..., None] * y[..., :-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            out = 0.5 * (y[..., 0] ** 2 / s**2 + (innov**2).sum(axis=-1) / (s**2 * (1.0 - r**2)))
        return np.where(s > 0.0, out, 0.0)

    q0, qp, qm = q(x), q(x - 1.0), q(x + 1.0)
    return 0.5 * (qm - qp), qp + qm - 2.0 * q0


def static_ridge_step(
    sites: dict[str, Tensor],
    latents: dict[int, MD.WindowLatents],
    *,
    wander: MD.Wander,
    measured: MD.Measured,
    priors: MD.PriorsV3,
    shape_chol: Tensor,
    families: Sequence[str] = ("d", "v", "u", "uj"),
) -> tuple[dict[str, Tensor], dict[int, MD.WindowLatents], dict[str, Any]]:
    """The EXACT minimum of the priors along the Whittle-invariant ridge.

    A constant shift ``c`` of one latent track over every window and block,
    with ``-c`` put into the rig parameter the track multiplies, leaves every
    block's expected periodogram unchanged: ``profile_db[r, k] + d_r + v_rk``
    for the lines, ``sigma_B L z + u + u_j`` at the floor's control points
    (the latents' floor basis is the spline's own interpolation, whose rows
    sum to one). The Whittle term cannot see these moves, so neither step of
    the alternation makes them: the latent step only feels the OU prior's
    weak pull on a static offset, and the rig step with the latents fixed has
    no gradient along them at all. This step makes them in closed form. Every
    prior involved is Gaussian — the profile ``N(p_hat, sd)``, the floor
    ``z ~ N(0, I)`` and each stationary OU (quadratic in a constant shift) —
    so the minimum is one small linear solve per rotor for the lines
    (``d_r`` shifts rotor ``r``'s whole profile, ``v_rk`` one line) and one
    ``(J + 1)`` solve for the floor (``u`` shifts every control value,
    ``u_j`` one). Tracks the wander pins (``sigma`` 0) never move. No
    parameter is added; ``families`` restricts the move.

    Returns the moved sites, the moved latents and a record of the shifts and
    of each prior's change (nats).
    """
    windows = sorted(latents)
    fam = set(families)

    def arr(t: Tensor | None) -> np.ndarray | None:
        return None if t is None else t.detach().cpu().numpy().astype(np.float64)

    prof = arr(sites["profile_db"])
    z = arr(sites["floor_shape_z"])
    assert prof is not None and z is not None
    n_r, n_k = prof.shape
    loc, scale = measured.profile_prior(priors)
    p_hat = loc.detach().cpu().numpy()[:n_r, :n_k]
    s2 = scale.detach().cpu().numpy()[:n_r, :n_k] ** 2
    lat = {w: {n: arr(t) for n, t in latents[w].tracks().items()} for w in windows}

    def coeffs(name: str, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        """``(g, h)`` summed over the windows, per track of ``name``."""
        g, h = np.zeros(shape), np.zeros(shape)
        if name not in fam or not wander.active(name):
            return g, h
        sig: Any = wander.track_sigma(name, n_k)
        rho: Any = wander.track_rho(name, n_k)
        if name == "v":
            sig = np.broadcast_to(np.asarray(sig, dtype=np.float64), (n_k,))[None, :]
            rho = np.broadcast_to(np.asarray(rho, dtype=np.float64), (n_k,))[None, :]
        for w in windows:
            x = lat[w].get(name)
            if x is None:
                continue
            if name == "v":
                x = x[:, :n_k]
            gw, hw = _ou_shift_coeffs(
                x, np.asarray(sig, dtype=np.float64), np.asarray(rho, dtype=np.float64)
            )
            g, h = g + gw, h + hw
        return g, h

    # ── lines: per rotor, delta_r (d) and eps_rk (v) ───────────────────────
    g_d, h_d = coeffs("d", (n_r,))
    g_v, h_v = coeffs("v", (n_r, n_k))
    live_v = h_v > 0.0
    e = prof - p_hat
    den = 1.0 / s2 + h_v
    a = np.where(live_v, (g_v - e / s2) / den, 0.0)
    b = np.where(live_v, (1.0 / s2) / den, 0.0)
    delta = np.zeros(n_r)
    if np.any(h_d > 0.0):
        num = g_d - ((e + a) / s2).sum(axis=1)
        cur = ((1.0 - b) / s2).sum(axis=1) + h_d
        delta = np.where(h_d > 0.0, num / cur, 0.0)
    eps = np.where(live_v, a - b * delta[:, None], 0.0)
    shift_line = delta[:, None] + eps

    # ── floor: w = (delta_u, eps_j); ctrl shift delta_u 1 + eps ───────────
    sb = float(measured.floor_shape_sd_db or 1.0)
    chol = shape_chol.detach().cpu().numpy().astype(np.float64)
    n_j = chol.shape[0]
    g_u, h_u = coeffs("u", ())
    g_j, h_j = coeffs("uj", (n_j,))
    use = np.concatenate([[float(h_u) > 0.0], h_j > 0.0])
    w_shift = np.zeros(n_j + 1)
    if np.any(use):
        amat = np.linalg.solve(sb * chol, np.eye(n_j))  # (sigma_B L)^-1
        m_full = np.concatenate([amat @ np.ones(n_j)[:, None], amat], axis=1)[:, use]
        g_full = np.concatenate([[float(g_u)], g_j])[use]
        h_full = np.concatenate([[float(h_u)], h_j])[use]
        lhs = m_full.T @ m_full + np.diag(h_full)
        w_shift[use] = np.linalg.solve(lhs, g_full - m_full.T @ z)
        z_new = z + m_full @ w_shift[use]
    else:
        z_new = z.copy()
    d_u, eps_j = float(w_shift[0]), w_shift[1:]

    # ── apply ──────────────────────────────────────────────────────────────
    dev = sites["profile_db"].device

    def t(x: np.ndarray) -> Tensor:
        return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float64, device=dev)

    new_sites = dict(sites)
    new_sites["profile_db"] = t(prof + shift_line)
    new_sites["floor_shape_z"] = t(z_new)
    new_lat: dict[int, MD.WindowLatents] = {}
    for w in windows:
        old = latents[w]
        new_lat[w] = MD.WindowLatents(
            d=None if old.d is None else old.d - t(delta)[:, None],
            v=None
            if old.v is None
            else torch.cat([old.v[:, :n_k] - t(eps)[..., None], old.v[:, n_k:]], dim=1),
            u=None if old.u is None else old.u - d_u,
            uj=None if old.uj is None else old.uj - t(eps_j)[:, None],
        )
    prof_nats = float((((prof + shift_line - p_hat) ** 2 - e**2) / (2.0 * s2)).sum())
    z_nats = 0.5 * float(z_new @ z_new - z @ z)
    ou_before = MD.ou_prior_nats(latents, wander)
    ou_after = MD.ou_prior_nats(new_lat, wander)
    record = dict(
        families=sorted(fam),
        shift_d_db=delta.tolist(),
        shift_v_rms_db=float(np.sqrt((eps[live_v] ** 2).mean())) if live_v.any() else 0.0,
        shift_u_db=d_u,
        shift_uj_db=eps_j.tolist(),
        profile_prior_change_nats=prof_nats,
        z_prior_change_nats=z_nats,
        ou_prior_change_nats=ou_after - ou_before,
        total_prior_change_nats=prof_nats + z_nats + ou_after - ou_before,
        z_norm=[float(np.linalg.norm(z)), float(np.linalg.norm(z_new))],
    )
    return new_sites, new_lat, record


def _rig_timing(opt: dict[str, Any]) -> dict[str, float | None]:
    """Per-step costs of one rig fit, from its optimiser record: seconds per
    Adam step and per L-BFGS evaluation (forward + gradient, every chunk), and
    the L-BFGS iterations the first pass and the restart ran."""
    steps = int(opt.get("adam_steps") or 0)
    evals = int(opt.get("lbfgs_evals") or 0) + int(opt.get("lbfgs_restart_evals") or 0)
    return dict(
        adam_s_per_step=(float(opt["adam_wall_s"]) / steps) if steps else None,
        rig_lbfgs_evals=evals,
        rig_lbfgs_iters=opt.get("lbfgs_iters_used"),
        rig_lbfgs_restart_iters=opt.get("lbfgs_restart_iters_used"),
        rig_s_per_eval=(float(opt["lbfgs_wall_s"]) / evals) if evals else None,
    )


def fit_v3(
    batch: MD.SupportBatch,
    *,
    priors: MD.PriorsV3,
    pin: dict[str, Any] | None = None,
    optim: OptimSpecV3 = OptimSpecV3(),
    profile_init: ProfileInit | None = None,
    progress: int = 0,
    init_from: V2Fit | None = None,
) -> FitOutcome:
    """MAP-fit noise model v3 on a flight pool (explainer §3.4).

    The pool is MEASURED once, latent-free (:func:`seeds`: the prior centres,
    ``mu``, ``sigma_B``, the wind centres), its frames are assigned to blocks
    of the MEASURED ``priors.wander.block_s``, and the fit alternates:

    (i)   the rig with every latent at zero — :func:`fit_support` exactly as a
          v2 fit runs, restarts, jitter and pins included;
    (ii)  every window's latents with the rig fixed (:func:`fit_latents`: the
          windows are independent given the rig, so this is ONE batched
          problem over a ``windows`` plate);
    (iii) the rig with the latents fixed, warm-started from (i)/(iii);

    repeating (ii)-(iii) until the Whittle term and the total objective move by
    less than ``optim.tol_nats_per_cell`` per cell, or ``optim.rounds``
    rounds. The wander's ``(sigma, tau)`` never move, so this converges to the
    joint MAP over rig and latents without the variance cheat of §3.3a.
    ``optim.ridge_step`` inserts :func:`static_ridge_step` between (ii) and
    (iii), and once after the last rig step: neither step can move a static
    offset between the latents and the rig (the Whittle term is flat along
    it), so without it the static part stays wherever round 1 put it.
    A rig L-BFGS on a frame subset (``optim.rig.lbfgs_frames``) is followed,
    after the last round, by ONE all-frames polish of the rig at the final
    latents; what it gains over the subset optimum is ``optimiser.polish``.

    ``init_from`` starts step (i) at a v2 fit of the same pool
    (:func:`warm_start_v3`; recorded in ``diagnostics.init_from``) instead
    of the measured start; the priors are the measured ones either way.

    The returned outcome's ``params`` are the RIG only; ``latents`` is the
    record's per-window block tracks (the §3.5 check) and the objective carries
    the total ``Whittle + rig prior + OU prior``, which is also what
    ``optimiser.lbfgs_loss_after`` reports so restarts reduce on it.
    """
    if priors.wander is None:
        raise ValueError("flight_v3 needs the measured wander (PriorsV3.wander, --wander)")
    wander = priors.wander
    mode = MD.V3_MODE
    t0 = time.time()
    dev = batch.power.device
    _peak_mb(dev, reset=True)
    blocked = MD.with_blocks(replace(batch, latents=None), wander.block_s)
    start = seeds(blocked, mode=mode, priors=priors, pin=pin, profile_init=profile_init)
    measured_batch = replace(blocked, measured=start.measured)
    n_cells = max(1, blocked.n_cells)
    warm: dict[str, Any] | None = None
    if init_from is not None:
        # the guide starts at the v2 MAP of the same pool (clipped into the v3
        # priors); the priors stay the ones v3 measured
        warm_init, warm = warm_start_v3(
            init_from, batch=measured_batch, start=start, priors=priors, pin=pin
        )
        start = Seeds(measured=start.measured, init=warm_init)

    # every rig evaluation sums the lines through their unit-atom
    # autocorrelation (spectrum._line_sum_unit_autocorr: no autograd through
    # the atoms; a fitted amp_exp through the autocorrelation's tangent)
    rig_kw: dict[str, Any] = dict(unit_autocorr=True)

    rig = fit_support(
        blocked,
        mode=mode,
        priors=priors,
        pin=pin,
        optim=optim.rig,
        forward_kw=rig_kw,
        profile_init=profile_init,
        progress=progress,
        start=start,
    )
    first_rig = rig
    history: list[dict[str, Any]] = [
        dict(
            round=0,
            step="(i) rig, latents at zero",
            whittle_nats=float(rig.objective["whittle_nats"]),
            rig_converged=rig.converged,
            **_rig_timing(rig.optimiser),
            peak_mem_mb=_peak_mb(dev, reset=True),
            wall_s=time.time() - t0,
        )
    ]
    if progress:
        print(f"  v3 round 0: whittle {history[-1]['whittle_nats']:.6g}", flush=True)
    latents: dict[int, MD.WindowLatents] = {
        w: MD.zero_latents(
            wander, n_rotors=blocked.n_rotors, k_max=blocked.k_max, n_blocks=nb, device=dev
        )
        for w, nb in enumerate(blocked.window_blocks)
        if nb > 0
    }
    latent_fit: dict[str, Any] | None = None
    prev = float(rig.objective["whittle_nats"])
    prev_total = prev + _priors_nats(measured_batch, rig.sites, latents, priors=priors, pin=pin)
    alternation_converged = int(optim.rounds) == 0
    refit = replace(optim.rig, adam_steps=int(optim.refit_adam_steps), init_jitter=0.0)
    shape_chol = blocked.grid.floor.shape_chol
    for rnd in range(1, int(optim.rounds) + 1):
        t_round = time.time()
        fixed = MD.detach_params(rig.params)
        latents, latent_fit = fit_latents(
            measured_batch,
            fixed,
            wander=wander,
            init=latents,
            iters=optim.latent_lbfgs_iters,
            history=optim.latent_lbfgs_history,
            line_dtype=getattr(torch, optim.latent_dtype),
        )
        t_latent = time.time() - t_round
        latent_peak = _peak_mb(dev, reset=True)
        rig_start = dict(rig.sites)
        ridge: dict[str, Any] | None = None
        if optim.ridge_step:
            rig_start, latents, ridge = static_ridge_step(
                rig_start,
                latents,
                wander=wander,
                measured=start.measured,
                priors=priors,
                shape_chol=shape_chol,
            )
        rig = fit_support(
            replace(blocked, latents=latents),
            mode=mode,
            priors=priors,
            pin=pin,
            optim=refit,
            forward_kw=rig_kw,
            profile_init=profile_init,
            progress=progress,
            start=Seeds(measured=start.measured, init=rig_start),
        )
        now = float(rig.objective["whittle_nats"])
        now_total = now + _priors_nats(measured_batch, rig.sites, latents, priors=priors, pin=pin)
        # the move of the Whittle term AND of the total objective: a ridge
        # step moves the priors at a fixed Whittle term
        move = max(abs(prev - now), abs(prev_total - now_total)) / n_cells
        history.append(
            dict(
                round=rnd,
                step="(ii) latents, rig fixed; (iii) rig, latents fixed",
                whittle_nats=now,
                total_nats=now_total,
                whittle_move_per_cell=abs(prev - now) / n_cells,
                move_per_cell=move,
                ridge=ridge,
                rig_converged=rig.converged,
                latent_wall_s=t_latent,
                latent_evals=latent_fit.get("evals"),
                latent_s_per_eval=latent_fit.get("s_per_eval"),
                latent_cache_s=latent_fit.get("cache_s"),
                latent_peak_mem_mb=latent_peak,
                **_rig_timing(rig.optimiser),
                peak_mem_mb=_peak_mb(dev, reset=True),
                wall_s=time.time() - t_round,
            )
        )
        if progress:
            print(f"  v3 round {rnd}: whittle {now:.6g}  move/cell {move:.3g}", flush=True)
        prev, prev_total = now, now_total
        if move < float(optim.tol_nats_per_cell):
            alternation_converged = True
            break

    # every rig step above ran its L-BFGS on a stratified frame SUBSET
    # (``optim.rig.lbfgs_frames``): ONE all-frames L-BFGS polish of the rig at
    # the final latents, warm-started from the subset optimum, and what it
    # still gains on the full pool recorded
    polish: dict[str, Any] | None = None
    n_frames = int(blocked.power.shape[1])
    if optim.rig.lbfgs_frames is not None and int(optim.rig.lbfgs_frames) < n_frames:
        t_polish = time.time()
        subset_rig = rig
        rig = fit_support(
            replace(blocked, latents=latents),
            mode=mode,
            priors=priors,
            pin=pin,
            optim=replace(refit, adam_steps=0, lbfgs_frames=None),
            forward_kw=rig_kw,
            profile_init=profile_init,
            progress=progress,
            start=Seeds(measured=start.measured, init=dict(rig.sites)),
        )
        at_subset = float(rig.optimiser["lbfgs_loss_before"])
        polished = float(rig.optimiser["lbfgs_loss_after"])
        polish = dict(
            frames=n_frames,
            subset_frames=int(subset_rig.optimiser["lbfgs_frames_used"]),
            loss_note="Whittle - rig log prior on the FULL pool at the final latents",
            loss_at_subset_optimum=at_subset,
            loss_polished=polished,
            gain_nats=at_subset - polished,
            gain_per_cell=(at_subset - polished) / n_cells,
            whittle_at_subset_optimum=float(subset_rig.objective["whittle_nats"]),
            whittle_polished=float(rig.objective["whittle_nats"]),
            rig_converged=rig.converged,
            **_rig_timing(rig.optimiser),
            peak_mem_mb=_peak_mb(dev, reset=True),
            wall_s=time.time() - t_polish,
        )
        if progress:
            print(
                f"  v3 all-frames polish: whittle {polish['whittle_polished']:.6g}  "
                f"gain/cell {polish['gain_per_cell']:.3g}",
                flush=True,
            )

    final_ridge: dict[str, Any] | None = None
    if optim.ridge_step:
        # the last rig step leaves a static part in the latents it could not
        # see: fold it out exactly, so the recorded rig carries it (a render
        # draws zero-mean latents)
        sites_f, latents, final_ridge = static_ridge_step(
            dict(rig.sites),
            latents,
            wander=wander,
            measured=start.measured,
            priors=priors,
            shape_chol=shape_chol,
        )
        params_f = MD.sample_params_from_values(
            measured_batch, mode=mode, priors=priors, pin=pin, values=sites_f
        )
        with torch.no_grad():
            objective_f = MD.objective_breakdown(
                replace(blocked, latents=latents),
                MD.forward(replace(blocked, latents=latents), params_f, **rig_kw),
            )
        rig = replace(rig, params=params_f, sites=sites_f, objective=objective_f)

    final = replace(measured_batch, latents=latents)
    whittle = float(rig.objective["whittle_nats"])
    rig_nlp = -MD.log_prior(final, mode=mode, values=rig.sites, priors=priors, pin=pin)
    ou_nlp = MD.ou_prior_nats(latents, wander)
    total = whittle + rig_nlp + ou_nlp
    converged = bool(alternation_converged and rig.converged)
    optimiser = dict(
        **optim.as_dict(),
        seed=int(optim.rig.seed),
        init_jitter=float(optim.rig.init_jitter),
        pinned_dynamics=(dict(pin) if pin else None),
        rounds_run=len(history) - 1,
        history=history,
        alternation_converged=bool(alternation_converged),
        rig_converged=bool(rig.converged),
        converged=converged,
        which_converged=(
            "alternation+lbfgs" if converged else ("lbfgs" if rig.converged else "none")
        ),
        lbfgs_loss_after=total,
        loss_note="lbfgs_loss_after is the TOTAL objective Whittle + rig -log prior + OU "
        "-log prior on the full pool at the final rig and latents (what restarts reduce on)",
        first_rig=first_rig.optimiser,
        last_rig=rig.optimiser,
        polish=polish,
        final_ridge=final_ridge,
        device=str(dev),
        chunk_frames=batch.chunk_frames,
        line_kernel="unit_autocorr",
        peak_mem_mb=max(
            (
                float(v)
                for row in (*history, polish or {})
                for k in ("peak_mem_mb", "latent_peak_mem_mb")
                if (v := row.get(k)) is not None
            ),
            default=None,
        ),
        wall_s=time.time() - t0,
    )
    objective = dict(
        rig.objective,
        rig_neg_log_prior_nats=rig_nlp,
        ou_neg_log_prior_nats=ou_nlp,
        total_nats=total,
        whittle_latents_zero_nats=float(first_rig.objective["whittle_nats"]),
    )
    diagnostics = dict(
        rig.diagnostics,
        blocks=dict(
            block_s=wander.block_s,
            window_blocks=list(blocked.window_blocks),
            n_windows=len(blocked.window_blocks),
        ),
        init_from=warm,
    )
    return FitOutcome(
        params=rig.params,
        objective=objective,
        optimiser=optimiser,
        diagnostics=diagnostics,
        sites=rig.sites,
        latents=_latents_record(blocked, latents, wander, latent_fit),
        window_latents=latents,
    )


def _latents_record(
    batch: MD.SupportBatch,
    latents: dict[int, MD.WindowLatents],
    wander: MD.Wander,
    fit: dict[str, Any] | None,
) -> dict[str, Any]:
    """The ``latents`` block of a v3 record: every window's MAP block tracks
    (dB, rounded to 1e-3) and, per track, the pooled spread and lag-1
    correlation of the fitted tracks against the measured ``sigma``/``rho`` —
    the explainer's §3.5 post-fit check reads these."""
    assert batch.frame_window is not None and batch.frame_block is not None

    def arr(x: Tensor | None) -> Any:
        return None if x is None else np.round(x.detach().cpu().numpy(), 3).tolist()

    windows = []
    for w, lat in sorted(latents.items()):
        n_b = int(batch.window_blocks[w])
        counts = np.bincount(batch.frame_block[batch.frame_window == w], minlength=n_b)
        windows.append(
            dict(
                window=int(w),
                name=batch.members[w] if w < len(batch.members) else str(w),
                n_blocks=n_b,
                frames_per_block=counts.tolist(),
                d=arr(lat.d),
                v=arr(lat.v),
                u=arr(lat.u),
                uj=arr(lat.uj),
            )
        )
    summary: dict[str, Any] = {}
    for name in ("d", "v", "u", "uj"):
        if not wander.active(name):
            continue
        xs = [
            x.detach().cpu().numpy()
            for lat in latents.values()
            if (x := lat.tracks().get(name)) is not None
        ]
        if not xs:
            continue
        flat = np.concatenate([x.reshape(-1) for x in xs])
        num = sum(float((x[..., 1:] * x[..., :-1]).sum()) for x in xs)
        den = sum(float((x[..., :-1] ** 2).sum()) for x in xs)
        summary[name] = dict(
            prior_sd_db=wander.sigma(name),
            prior_rho=wander.rho(name),
            fitted_sd_db=float(np.sqrt(np.mean(flat**2))),
            fitted_lag1=(num / den) if den > 0.0 else None,
            n_values=int(flat.size),
        )
    return dict(
        block_s=wander.block_s,
        units="dB",
        tracks=[n for n in ("d", "v", "u", "uj") if wander.active(n)],
        shapes="d (R, B), v (R, K, B), u (B,), uj (J, B) per window",
        note="per-window MAP nuisance; NOT rendered (render_noise draws fresh OU tracks)",
        fit=fit,
        summary=summary,
        windows=windows,
    )


# ── the post-fit checks ─────────────────────────────────────────────────────

#: A fitted low-order width may sit this many times over the window's
#: resolution floor and still count as "at the floor". Well inside one log-sd
#: of the width prior (e^1 = 2.7 at k = 1 against a 0.01 Hz median on a 30 s
#: bench window), and far under the x 16 a k^2 ramp from k = 1 to k = 4 would
#: produce — which is the failure this check exists to name.
GAMMA_FLOOR_FACTOR = 3.0

#: A k = 4 to k = 1 width ratio at or over this is the shaft-absorbing ramp.
#: Half of the k^2 value (16), so a ramp has to be unambiguous to be called.
GAMMA_RAMP_RATIO = 8.0

#: The orders the check reads. "Low orders" of the R3 identification section.
GAMMA_CHECK_K = 4


def gamma_low_order_check(gamma_hz: Any, *, batch: MD.SupportBatch) -> dict[str, Any]:
    """The R3 low-order ``gamma_rk`` check (explainer, "Identification").

    The one degeneracy R3 can have is the shaft being absorbed by the free
    widths: in the Brownian limit the shaft's own factor is Lorentzian too,
    ``prop k^2 sigma_nu^2 / lam``, so a ``gamma_rk`` ramping as ``k^2`` would
    fit the same line shape with no shaft at all. PASS is the fitted width at
    ``k <= 4`` sitting AT OR UNDER the window's resolution floor; FAIL is a
    ``k^2`` ramp above it, and then ``lam`` takes the long-lag pin instead of
    the bench value. Measured here after every fit rather than assumed away.

    The floor test is the DECISIVE one and the ramp ratio is only read where
    it fails. Widths under the resolution are all the same statement — "this
    line is unresolved, the shaft alone sets its shape" — and the objective is
    flat among them, so their RATIO is noise: the first R3 bench fit put
    ``gamma_1 = 2.5e-4`` and ``gamma_4 = 1.5e-2`` Hz on a 12 s window whose
    resolution is 4.2e-2 Hz, a ratio of 59 between two numbers that are both
    a quarter of a bin wide and a shaft (``sigma_nu = 0.40``) that plainly was
    NOT absorbed. A ramp that absorbs the shaft has to be visible first.
    """
    g = np.atleast_2d(np.asarray(_np(gamma_hz), dtype=np.float64))
    kk = min(int(GAMMA_CHECK_K), int(g.shape[1]))
    low = g[:, :kk]
    res = float(batch.resolution_hz)
    over = low / max(res, 1e-30)
    ramp = float(np.max(low[:, kk - 1] / np.maximum(low[:, 0], 1e-30))) if kk >= 2 else 1.0
    at_floor = bool(np.all(over <= GAMMA_FLOOR_FACTOR))
    is_ramp = bool(ramp >= GAMMA_RAMP_RATIO)
    verdict = "pass" if at_floor else ("fail_k2_ramp" if is_ramp else "fail_above_floor")
    return dict(
        verdict=verdict,
        passed=verdict == "pass",
        k_checked=list(range(1, kk + 1)),
        gamma_hz=low.tolist(),
        resolution_hz=res,
        gamma_over_resolution=over.tolist(),
        max_over_resolution=float(over.max()),
        floor_factor=GAMMA_FLOOR_FACTOR,
        k4_over_k1=ramp,
        ramp_ratio_flagged_at=GAMMA_RAMP_RATIO,
        rule=(
            "pass = every gamma_rk at k <= 4 within floor_factor of 1 / (2 T); above the "
            "floor, fail_k2_ramp = gamma_4 / gamma_1 >= ramp threshold (the shaft absorbed), "
            "otherwise fail_above_floor"
        ),
    )


def span_pin_record(batch: MD.SupportBatch, *, priors: MD.Priors = MD.PRIORS) -> dict[str, Any]:
    """Which speed laws this pool's carrier span pinned, and at what value."""
    pinned = MD.span_pinned_sites(batch, priors=priors)
    medians = priors.speed_law_medians()
    return dict(
        speed_span=float(batch.speed_span),
        threshold=float(priors.speed_span_pin),
        pinned=list(pinned),
        values={name: medians[name] for name in pinned},
        rule="a pool spanning less than the threshold in max/min carrier cannot identify the "
        "speed laws, so they are constants at their prior medians (no site, no prior term)",
    )


def _np(v: Any) -> np.ndarray:
    return np.asarray(v.detach().cpu() if isinstance(v, Tensor) else v, dtype=np.float64)


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
    """Write the ``noise-v2-fit/2`` JSON (``noise-v3-fit/1`` for mode
    ``flight_v3``, :func:`_write_fit_v3`) and return its path."""
    if mode == MD.V3_MODE:
        return _write_fit_v3(
            path,
            support=support,
            kind=kind,
            outcome=outcome,
            batch=batch,
            priors=priors,
            extra=extra,
        )
    p = MD.params_to_dict(outcome.params)
    if outcome.comb_gain_db is not None:
        # the level the transplanted comb was re-levelled BY. ``profile_db``
        # above already carries it (the model folds it in), so a reader must
        # NEVER apply it again: it is recorded so the shift is auditable
        # against the bench profile the fit froze.
        p["profile"]["comb_gain_db"] = outcome.comb_gain_db
    if outcome.low_order_gain_db is not None:
        # the per-order dB the frozen comb's low orders were re-levelled BY,
        # on top of comb_gain_db. ``profile_db`` above already carries both.
        p["profile"]["low_order_gain_db"] = outcome.low_order_gain_db
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
            profile_orders=k.tolist(),
            gamma_hz_at=dict(
                k=[kk for kk in (1, 2, 4, 8, 16, 32) if kk <= int(k.size)],
                value=[
                    np.asarray(p["gamma_hz"], dtype=np.float64)[:, kk - 1].tolist()
                    for kk in (1, 2, 4, 8, 16, 32)
                    if kk <= int(k.size)
                ],
            ),
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
    if outcome.comb_gain_db is not None:
        payload["frozen_from"] = dict(
            payload.get("frozen_from") or {},
            comb_gain_db=outcome.comb_gain_db,
            comb_gain_rule="one free scalar re-levels the whole frozen comb; "
            "params.profile.profile_db ALREADY includes it and must not be shifted again",
            comb_gain_prior=list(priors.comb_gain_db),
        )
    if outcome.low_order_gain_db is not None:
        payload["frozen_from"] = dict(
            payload.get("frozen_from") or {},
            low_order_gain_db=outcome.low_order_gain_db,
            low_orders=len(outcome.low_order_gain_db),
            low_order_gain_rule="one gain PER ORDER below low_orders, shared by the rotors, "
            "on top of comb_gain_db; params.profile.profile_db ALREADY includes both and "
            "must not be shifted again. Above low_orders the frozen comb keeps its shape",
            low_order_gain_prior=list(priors.low_order_gain_db),
        )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    return out


def _write_fit_v3(
    path: str | Path,
    *,
    support: str,
    kind: str,
    outcome: FitOutcome,
    batch: MD.SupportBatch,
    priors: MD.Priors,
    extra: dict[str, Any] | None,
) -> Path:
    """The ``noise-v3-fit/1`` record: rig ``params``, measured ``priors``
    (wander, ``sigma_B``, channel gains, wind shape) and the per-window
    ``latents`` block."""
    if not isinstance(priors, MD.PriorsV3) or priors.wander is None:
        raise TypeError("a noise-v3-fit/1 record needs PriorsV3 carrying the measured wander")
    p = MD.params_to_dict_v3(outcome.params, wander=priors.wander)
    meas = outcome.diagnostics.get("measured") or {}
    pri = priors.as_dict()
    pri["floor_shape_z"] = dict(
        pri["floor_shape_z"],
        sigma_B_db=meas.get("floor_shape_sd_db"),
        mu_db=meas.get("floor_mean_db"),
        measured_ctrl_db=meas.get("floor_ctrl_db"),
    )
    if priors.wind:
        pri["wind"] = dict(pri["wind"] or {}, measured_centres_db=meas.get("wind_db"))
    pri["channel_gains"] = batch.diagnostics.get("channel_gains")
    k = np.arange(1, int(np.asarray(p["profile"]["profile_db"]).shape[1]) + 1)
    gamma = np.asarray(p["gamma_hz"], dtype=np.float64)
    ladder = [kk for kk in (1, 2, 4, 8, 16, 32) if kk <= int(k.size)]
    payload: dict[str, Any] = dict(
        schema=FIT_SCHEMA_V3,
        support=support,
        supports=list(batch.members) if batch.members else [support],
        kind=kind,
        mode=MD.V3_MODE,
        n_rotors=batch.n_rotors,
        n_mics=batch.n_mics,
        k_max=batch.k_max,
        sr=int(batch.grid.sr),
        front_end=dict(batch.grid.diagnostics),
        params=p,
        objective=outcome.objective,
        optimiser=outcome.optimiser,
        priors=pri,
        latents=outcome.latents,
        diagnostics=dict(
            outcome.diagnostics,
            profile_orders=k.tolist(),
            gamma_hz_at=dict(k=ladder, value=[gamma[:, kk - 1].tolist() for kk in ladder]),
            floor_level_db=float(p["floor"]["floor_mean_db"]),
        ),
        git=_git_head(),
    )
    if extra:
        payload.update(extra)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    return out
