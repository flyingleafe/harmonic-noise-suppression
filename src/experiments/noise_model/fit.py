"""MAP fit of the v2 model and the ``noise-v2-fit/2`` JSON it writes.

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

from experiments.stochastic_fit.model import FLOOR_SHAPE_N_CTRL
from experiments.stochastic_fit.revised_phase import (
    FLOOR_INIT_DB_CORRECTION,
    FLOOR_INIT_QUANTILE,
    _git_head,
)

from . import FIT_SCHEMA
from . import model as MD
from . import spectrum as SP

__all__ = [
    "FitOutcome",
    "OptimSpec",
    "ProfileInit",
    "Seeds",
    "fit_support",
    "gamma_low_order_check",
    "initial_values",
    "load_profile_init",
    "measure_batch",
    "seeds",
    "span_pin_record",
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
            lbfgs_history=self.lbfgs_history,
            lbfgs_line_search="strong_wolfe",
            seed=self.seed,
            tol_nats_per_cell=self.tol_nats_per_cell,
            init_jitter=self.init_jitter,
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
    batch: MD.SupportBatch, *, band: np.ndarray, carrier: Tensor | None
) -> list[tuple[int, int, int, int]]:
    """``(rotor, order, j0, j1)`` per comb line: the bins that line covers.

    The window spans where the line went over the batch, plus the analysis
    window's own main lobe: a moving carrier smears the frame-mean peak across
    ``k * (f_max - f_min)``. Lines wholly outside the fitted band are dropped.
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
            if f_hi < SP.BAND_F_MIN or f_lo > f_top:
                continue
            j0 = max(0, int(math.floor(f_lo / df)) - 3)
            j1 = min(n_bins, int(math.ceil(f_hi / df)) + 4)
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
    """
    torch.manual_seed(int(optim.seed))
    pyro.set_rng_seed(int(optim.seed))
    pyro.clear_param_store()

    measured = seeds(
        batch,
        mode=mode,
        priors=priors,
        frozen=frozen,
        pin=pin,
        low_orders=low_orders,
        profile_init=profile_init,
    )
    full = replace(batch, measured=measured.measured)
    init = dict(measured.init)
    if optim.init_jitter > 0.0:
        # a log-normal multi-start on the dynamics block only: the profile,
        # floor and carrier initialisations are read off the data and a random
        # start for them would test the initialiser, not the landscape
        jrng = np.random.default_rng(int(optim.seed))
        for key in ("sigma_nu", "lam", "gamma_hz"):
            if key in init:
                v = init[key]
                shift = jrng.normal(0.0, float(optim.init_jitter), size=tuple(v.shape))
                init[key] = v * torch.as_tensor(np.exp(shift), dtype=torch.float64)

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

    # CLONE into the guide. ``AutoDelta`` keeps the very tensors ``init_to_value``
    # hands it as its own (unconstrained) parameters and then updates them IN
    # PLACE, so passing ``init`` itself would make the recorded
    # ``diagnostics.init_*`` read the FITTED value and report every fit as one
    # that never left its initialisation.
    guide = AutoDelta(
        model_for(full),
        init_loc_fn=init_to_value(values={k: v.detach().clone() for k, v in init.items()}),
    )
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
    boxed = _log_space_params(guide)
    elbo_of = lambda: elbo.differentiable_loss(loss_fn, guide)  # noqa: E731

    def box() -> None:
        """Hold every log-space coordinate inside ``LOG_BOX``."""
        with torch.no_grad():
            for p in boxed:
                p.clamp_(-LOG_BOX, LOG_BOX)

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
            # the probe point, not just the accepted step: strong Wolfe
            # extrapolates first and it is the EXTRAPOLATION that underflows
            box()
            opt.zero_grad(set_to_none=False)
            loss = elbo_of()
            loss.backward()
            count += 1
            return loss

        opt.step(closure)
        # L-BFGS leaves the parameters at its own accepted point, which the
        # closure never saw
        box()
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
    fitted = MD.sample_params_from_values(
        full, mode=mode, priors=priors, frozen=frozen, pin=pin, low_orders=low_orders, values=med
    )
    with torch.no_grad():
        m_model = MD.forward(full, fitted, **(forward_kw or {}))
        objective = MD.objective_breakdown(full, m_model)
    return FitOutcome(
        params=fitted,
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
                np.asarray(init["gamma_hz"], dtype=np.float64).tolist()
                if "gamma_hz" in init
                else None
            ),
            gamma_low_order_check=gamma_low_order_check(fitted.gamma_hz, batch=full),
            span_pins=span_pin_record(full, priors=priors),
            profile_init=_profile_init_record(profile_init, measured.measured),
            measured=dict(
                floor_mean_db=measured.measured.floor_mean_db,
                resolution_hz=measured.measured.resolution_hz,
                n_lines_visible=int(
                    (measured.measured.line_snr_db >= priors.line_visible_snr_db).sum()
                ),
                n_lines=int(measured.measured.line_snr_db.size),
                line_visible_snr_db=priors.line_visible_snr_db,
            ),
        ),
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
    """Write the ``noise-v2-fit/2`` JSON and return its path."""
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
