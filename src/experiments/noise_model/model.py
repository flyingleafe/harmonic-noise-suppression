"""The v2 Pyro model: measured priors, one Whittle factor, one objective.

WHAT PYRO IS HERE FOR. C2/C3 put ``log sigma, log D ~ N(0, 2^2)`` on the
dynamics, which is a statement that nothing is known
(``revised_phase.py:481-487``). Every number this campaign measured is a prior
here instead (``docs/explainers/noise-model-v2-plan.qmd``, "Model R3"):

    sigma_nu ~ LN(0.3, 0.7)   bench;  LN(0.3, 0.5) in flight
    lam      ~ LN(2.0, 1.0)   bench only; PINNED in flight
    gamma_rk ~ LN(0.01 k, 1.0) per rotor and order
    p_rk     ~ N(measured, 10) where the line is visible, N(mu_F - 15, 8) below
    mu_F     ~ N(measured band median, 10)

``gamma_rk``'s prior is the BENCH DECOHERENCE LAW, not the window resolution:
the measured per-order phase structure function ``V_eps ~ 0.09 k^1.1
tau^0.43`` rad^2 (``results/noise_v2/decoherence/findings.md``) is an
effective diffusion of about ``0.1 k`` rad^2/s, i.e. a half-width of about
``0.01 k`` Hz. It is deliberately tight and physically anchored so that it
cannot absorb the shaft: the shaft's own core is Gaussian of width
``~ k sigma_nu / (2 pi)`` Hz, some five times above this median at the bench
shaft scale, while one log-sd of 1.0 still lets a genuinely broad line reach
``x 7`` at 2 sd. The R1 per-order OU ``(sigma_eps, lam_eps)`` pair, its
exponent ``p`` and the never-fitted path term are GONE: the flight window
identified only the product ``sigma_eps^2 lam_eps`` and the bench found no
``k^p`` law joining the orders (``rounds/round1/basin/findings.md``,
``results/noise_v2/decoherence/findings.md``).

Positivity is carried by ``LogNormal`` under ``constraints.positive`` — no
softplus, so an ``AutoDelta`` MAP is the mode of the real posterior in the
constrained space and not of a reparameterised surrogate. The profile and the
floor level are centred on what the initialiser MEASURED on the support
(:class:`Measured`, filled by :func:`.fit.measure_batch` and carried on the
batch), because a prior mean 40 dB from the data is what parked R1's invisible
orders in a 400 Hz pedestal over the floor. The floor shape keeps C4's only
real nuisance prior (the standard-normal GP coordinate of
:meth:`revised_phase._RevisedModel.floor_shape_db`) and the microphone levels
keep proper but wide Gaussians.

THE SPEED-SPAN PIN. ``a`` (``amp_exp``), ``b`` (``floor_exp``) and ``s``
(``floor_static_rel``) are speed laws: a pool that barely changes speed cannot
see them, and on R2's cruise-only DREGON pool (span 1.14x) they ran to 38.8
and 17 — four sd of their priors — and put the standby floor 37 dB over the
real clip. When a pool's carrier span ``max/min`` is under
:attr:`Priors.speed_span_pin` they are therefore CONSTANTS at their prior
medians, with no site and no prior term (:func:`span_pinned_sites`), and the
fit JSON records which ones were pinned and on what span.

THE LIKELIHOOD is the campaign's composite risk, unchanged and imported:

    pyro.factor("whittle", -sum_i w_i sum_{m, f in band} [I / M + log M] / T)

with ``w_i = 1`` on the bench (one frame, one window, nothing to weight) and
:func:`revised_phase.composite_weights` exposure weights in flight. ``T = 1``
for the MAP and is RECORDED; the frozen composite temperatures of C4
(5.192017220082491 DREGON, 15.214066879865468 Michael's) rescale a POSTERIOR,
not a mode, so they do not belong in a MAP and are not silently applied.

The two-band split at 300 Hz (the approved floor band edge) is reported in the
objective, never fitted separately: the comb band is decisive for the
likelihood gate and the floor band is what the DREGON floor-only fit moves.

MODE ``flight_v3`` is noise model v3 (``docs/explainers/noise-model-v3-wander.qmd``,
:class:`PriorsV3`): the same flight forward model with light-tailed LINEAR
priors on the dynamics (``gamma_rk / (0.01 k) ~ HalfNormal(3)``,
``sigma_nu ~ HalfNormal(0.6)``), ONE profile prior for every order
(``N(pooled level at k f_r, 10)``, no visibility switch), the floor as the
spline alone (``c_j = mu + sigma_B (L z)_j`` with ``mu`` and ``sigma_B``
MEASURED, no mean/tilt sites), no microphone sites (the channels are
normalised in the data), an optional per-mic static wind level and per-window
block-wander latents with fixed OU priors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from typing import Any

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from torch import Tensor
from torch.distributions import constraints

# MOVED to data_processing.noise_model.params: the v2 renderer reads a fit's
# line widths and data_processing may not import experiments. Re-exported here
# unchanged, so `MD.gamma_from_params` keeps working.
from data_processing.noise_model.params import gamma_from_params
from data_processing.noise_model.v3 import Wander, wind_shape_spec
from experiments.stochastic_fit.model import FLOOR_SHAPE_N_CTRL
from experiments.stochastic_fit.revised_phase import composite_risk, composite_weights

from . import spectrum as SP
from .spectrum import BenchGrid, FlightGrid, FloorParams, V2Params

__all__ = [
    "BAND_SPLIT_HZ",
    "DYN_SITES",
    "PRIORS",
    "PRIORS_V3",
    "SPEED_LAW_SITES",
    "V3_MODE",
    "ChannelGains",
    "LatentCache",
    "Measured",
    "OUChain",
    "Priors",
    "PriorsV3",
    "SupportBatch",
    "Wander",
    "WindowLatents",
    "batch_slice",
    "bench_batch",
    "block_params",
    "detach_params",
    "dynamics_pin",
    "flight_batch",
    "flight_lam",
    "forward",
    "forward_v3",
    "free_blocks",
    "frozen_from_params",
    "gamma_from_params",
    "is_pinned",
    "latent_cache",
    "latent_model_cached",
    "log_prior",
    "objective_breakdown",
    "ou_log_density",
    "ou_prior_nats",
    "params_from_dict",
    "params_to_dict",
    "params_to_dict_v3",
    "pin_applied",
    "sample_params",
    "sample_window_latents",
    "sample_params_from_values",
    "span_pinned_sites",
    "support_model",
    "whittle_risk",
    "window_batch",
    "with_blocks",
    "zero_latents",
]

#: The approved floor/comb band edge. Below it the wind-dominated floor sets the
#: risk; above it the comb does, and the comb band is the decisive one.
BAND_SPLIT_HZ = 300.0

#: Every fitted block. ``--floor-only`` frees ``{"floor", "mic", "comb_gain"}``
#: and freezes the rest from a bench fit; a full flight fit frees everything but
#: ``"carrier"`` (the label IS the carrier in flight). ``"comb_gain"`` is not a
#: block of the export: it is the ONE scalar a frozen comb still needs (see
#: :func:`free_blocks`) and it is folded into ``profile_db``.
BLOCKS = ("dynamics", "profile", "floor", "mic", "carrier")

#: The dynamics SITES that a ``--pin`` can hold fixed, with their width. R3's
#: third dynamics site, ``gamma_hz``, is a whole ``(R, K)`` block and is
#: pinned by FREEZING it (``flight_floor_only``), never coordinate by
#: coordinate.
DYN_SITES: dict[str, int] = dict(sigma_nu=1, lam=1)

#: The three speed-law sites the span rule pins.
SPEED_LAW_SITES = ("amp_exp", "floor_exp", "floor_static_rel")


@dataclass(frozen=True)
class Priors:
    """The R3 priors (``noise-model-v2-plan.qmd``, "Parameters, with their
    priors"). ``(mean, sd)`` pairs; log-parameters in log space."""

    #: bench: 0.16-1.0 measured over the R2 bench fits
    log_sigma_nu: tuple[float, float] = (math.log(0.3), 0.7)
    #: flight: the same centre, tighter — with ``gamma_rk`` carrying the line
    #: width, ``sigma_nu`` should sit at the telemetry residual, and a fit that
    #: still wants > 2 rad/s is a model error to report, not a prior to loosen
    log_sigma_nu_flight: tuple[float, float] = (math.log(0.3), 0.5)
    #: bench only; 128 ms of flight lag cannot see it (R1 basin: 0.03
    #: nats/cell over three decades), so in flight it is PINNED
    log_lam: tuple[float, float] = (math.log(2.0), 1.0)
    #: ``gamma_rk ~ LN(gamma_per_order_hz * k, gamma_log_sd)``: the bench
    #: decoherence law ``V_eps ~ 0.09 k^1.1 tau^0.43`` rad^2 is an effective
    #: diffusion of ~ ``0.1 k`` rad^2/s, i.e. a HWHM of ~ ``0.01 k`` Hz. Tight
    #: and physical so it cannot absorb the shaft (whose core is Gaussian at
    #: ~ ``k sigma_nu / 2 pi`` Hz, about five times wider at the bench shaft
    #: scale); one log-sd still allows x 7 at 2 sd for a genuinely broad line.
    gamma_per_order_hz: float = 0.01
    gamma_log_sd: float = 1.0
    #: the pinned rate of the flight shaft when no bench fit supplies one
    #: (Michael's); the DREGON frozen-comb fit takes the bench value instead
    flight_lam: float = 0.5
    #: the profile's TWO regimes. A line whose measured excess over the floor
    #: reaches ``line_visible_snr_db`` is centred on that measurement; one
    #: below it is centred ``profile_below_offset_db`` under the floor level,
    #: because R1's invisible orders parked at a flat N(-45, 40) built a
    #: 400 Hz pedestal over the floor (``round1/bench_diag/findings.md``)
    line_visible_snr_db: float = 2.0
    profile_db_sd: float = 10.0
    profile_below_offset_db: float = -15.0
    profile_below_sd: float = 8.0
    #: centred on the measured band median; the old sd 40 had no reason
    floor_mean_db_sd: float = 10.0
    floor_tilt_db_oct: tuple[float, float] = (0.0, 5.0)
    #: the pooled flight speed exponent; 3.9 on the 4.7x-span pool
    amp_exp: tuple[float, float] = (2.0, 1.0)
    #: Rig-to-rig level offset of a TRANSPLANTED comb, in dB, and the only
    #: absolute comb scale a frozen-comb flight fit has left: ``render_noise``
    #: mean-centres ``mic_line_gain_db`` over mics per rotor and
    #: ``mic_gains_db`` over mics, so ``profile_db`` alone carries the comb's
    #: level and it arrives frozen at the BENCH rig's. Zero-mean (no offset is
    #: the null) and wide: the measured bench → DREGON-flight swing is 31.7 dB
    #: (``results/noise_v2/rounds/round2/render_dregon/findings.md``), which
    #: this prior must not fight.
    comb_gain_db: tuple[float, float] = (0.0, 20.0)
    #: Per-ORDER dB gain of the frozen comb's low orders in
    #: ``flight_floor_lowk``, shared across rotors (at 2048-point frames the
    #: DREGON rotors are not resolvable at k <= 8). As wide and zero-mean as
    #: ``comb_gain_db``, and for the same reason: the measured rig-to-rig
    #: swing is tens of dB and the prior must not fight it.
    low_order_gain_db: tuple[float, float] = (0.0, 20.0)
    #: default width of that block
    low_orders: int = 8
    #: LOG space: the floor's speed exponent is POSITIVE by construction. A
    #: rotor floor cannot get louder as the rotors slow (R2: -10.19 on a
    #: near-constant pool put the standby floor +37.0 dB over the real clip,
    #: ``round2/render_regime/findings.md``), and R3 tightens it because the
    #: span pin below, not the prior, is what protects a short-span pool.
    log_floor_exp: tuple[float, float] = (math.log(2.0), 0.5)
    log_floor_static: tuple[float, float] = (math.log(2.5e-3), 1.0)
    mic_line_gain_db: float = 6.0
    mic_floor_db: float = 6.0
    gain_all_db: float = 6.0
    #: a pool whose carrier span ``max / min`` is under this cannot identify
    #: the three speed laws, so they are held at their prior medians
    speed_span_pin: float = 1.5

    def sigma_nu_prior(self, mode_is_flight: bool) -> tuple[float, float]:
        return self.log_sigma_nu_flight if mode_is_flight else self.log_sigma_nu

    def gamma_loc(self, k: np.ndarray | Tensor) -> Tensor:
        """``log(gamma_per_order_hz * k)``: the prior's log-median per order."""
        kk = torch.as_tensor(np.array(k, dtype=np.float64), dtype=torch.float64)
        return torch.log(float(self.gamma_per_order_hz) * kk)

    def speed_law_medians(self) -> dict[str, float]:
        return dict(
            amp_exp=float(self.amp_exp[0]),
            floor_exp=math.exp(self.log_floor_exp[0]),
            floor_static_rel=math.exp(self.log_floor_static[0]),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "log_sigma_nu_bench": list(self.log_sigma_nu),
            "log_sigma_nu_flight": list(self.log_sigma_nu_flight),
            "log_lam": list(self.log_lam),
            "gamma_hz": {
                "family": "LogNormal",
                "median_hz": f"{self.gamma_per_order_hz:g} * k",
                "log_sd": self.gamma_log_sd,
                "source": "bench decoherence law V_eps ~ 0.09 k^1.1 tau^0.43 rad^2",
            },
            "flight_lam_pin": self.flight_lam,
            "profile_db": {
                "visible": ["measured", self.profile_db_sd],
                "below": [
                    f"floor_mean_db {self.profile_below_offset_db:+g}",
                    self.profile_below_sd,
                ],
                "line_visible_snr_db": self.line_visible_snr_db,
            },
            "floor_mean_db": ["measured band median", self.floor_mean_db_sd],
            "floor_tilt_db_oct": list(self.floor_tilt_db_oct),
            "amp_exp": list(self.amp_exp),
            "comb_gain_db": list(self.comb_gain_db),
            "low_order_gain_db": list(self.low_order_gain_db),
            "low_orders_default": self.low_orders,
            "log_floor_exp": list(self.log_floor_exp),
            "log_floor_static": list(self.log_floor_static),
            "mic_line_gain_db_sd": self.mic_line_gain_db,
            "mic_floor_db_sd": self.mic_floor_db,
            "gain_all_db_sd": self.gain_all_db,
            "speed_span_pin": self.speed_span_pin,
        }


PRIORS = Priors()

#: The v3 fit mode (``docs/explainers/noise-model-v3-wander.qmd``).
V3_MODE = "flight_v3"


@dataclass(frozen=True)
class PriorsV3(Priors):
    """The v3 priors (explainer §2.2-§2.6), for mode :data:`V3_MODE`.

    A SUBCLASS so every shared law — the speed laws ``amp_exp``,
    ``log_floor_exp``, ``log_floor_static``, their span pin and the pinned
    flight ``lam`` — is the very object v2 reads; the fields v3 replaces
    (the LogNormal dynamics, the two-regime profile, the floor mean/tilt, the
    mic sds) are simply not read in v3 mode and not recorded by
    :meth:`as_dict`.

    * ``gamma_rk / (gamma_per_order_hz k) ~ HalfNormal(gamma_c)`` and
      ``sigma_nu ~ HalfNormal(sigma_nu_scale)``, LINEAR and untruncated: a
      width of ``5 gamma_0 k`` costs 1.4 nats, 13 Hz at ``k = 1`` ~ 1e5;
      ``sigma_nu_scale`` is twice the v2 flight LogNormal median (0.3 rad/s);
    * ``profile_db ~ N(pooled level at k f_r, profile_db_sd)`` for EVERY line;
    * the floor control values ``mu + sigma_B (L z)_j``, ``z ~ N(0, I)``, with
      ``mu`` and ``sigma_B`` measured (:attr:`Measured.floor_shape_sd_db`,
      held at least ``floor_shape_sd_min_db``);
    * ``wind_db[m] ~ N(measured low-band excess, wind_db_sd)`` when ``wind``;
    * the block wander's MEASURED hyperparameters ``wander`` (fixed; the
      record they were read from, provenance included, in ``wander_record``).
    """

    gamma_c: float = 3.0
    sigma_nu_scale: float = 0.6
    floor_shape_sd_min_db: float = 1.0
    wind: bool = False
    wind_db_sd: float = 6.0
    wander: Wander | None = None
    wander_record: dict[str, Any] | None = field(default=None, compare=False, hash=False)

    def gamma_scale(self, k: np.ndarray | Tensor) -> Tensor:
        """``gamma_c * gamma_per_order_hz * k``: the half-normal scale per order."""
        kk = torch.as_tensor(np.array(k, dtype=np.float64), dtype=torch.float64)
        return float(self.gamma_c) * float(self.gamma_per_order_hz) * kk

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": "v3",
            "sigma_nu": {
                "family": "HalfNormal",
                "scale_rad_s": self.sigma_nu_scale,
                "source": "2 x the v2 flight LogNormal median (0.3 rad/s)",
            },
            "gamma_hz": {
                "family": "HalfNormal",
                "scale_hz": f"{self.gamma_c:g} * {self.gamma_per_order_hz:g} * k",
                "gamma_c": self.gamma_c,
                "gamma0_hz": self.gamma_per_order_hz,
                "source": "bench decoherence law V_eps ~ 0.09 k^1.1 tau^0.43 rad^2",
            },
            "flight_lam_pin": self.flight_lam,
            "profile_db": {
                "family": "Normal",
                "loc": "measured pooled level at k f_r (line + floor)",
                "sd": self.profile_db_sd,
                "rule": "every order; no visibility switch",
            },
            "floor_shape_z": {
                "family": "Normal(0, I)",
                "control_values_db": "mu + sigma_B (L z)_j",
                "sigma_B_min_db": self.floor_shape_sd_min_db,
            },
            "amp_exp": list(self.amp_exp),
            "log_floor_exp": list(self.log_floor_exp),
            "log_floor_static": list(self.log_floor_static),
            "speed_span_pin": self.speed_span_pin,
            "mic": "none: channels normalised in the data",
            "wind": (
                {
                    "family": "Normal",
                    "loc": "measured per-mic low-band excess",
                    "sd": self.wind_db_sd,
                    "shape": wind_shape_spec(),
                }
                if self.wind
                else None
            ),
            "wander": (
                self.wander_record
                if self.wander_record is not None
                else (self.wander.as_params() if self.wander is not None else None)
            ),
        }


PRIORS_V3 = PriorsV3()


@dataclass(frozen=True)
class Measured:
    """What the initialiser MEASURED on one batch, in the model's own units.

    Filled by :func:`.fit.measure_batch` and carried on the batch, because two
    R3 priors are data-driven: the profile's centre (and which of its two
    regimes a line is in) and the floor level's. ``gamma_hz`` is the measured
    -3 dB half width per line, the ``gamma_hz`` INITIALISATION — the prior
    itself is the physical law of :class:`Priors`.
    """

    floor_mean_db: float
    profile_db: np.ndarray
    line_snr_db: np.ndarray
    gamma_hz: np.ndarray
    resolution_hz: float
    #: ``(R, K)`` per-line prior centre from OUTSIDE the batch, NaN where the
    #: two-regime rule above keeps its say, and the matching per-line sd. The
    #: four-motor rig fit enters the multi-rotor estimator's profile here
    #: (``scripts/noise_v2_fourmotor.py``): it is a point estimate with a
    #: measured per-order error budget, so it belongs in the prior's centre and
    #: width, never in the likelihood.
    profile_prior_db: np.ndarray | None = None
    profile_prior_sd_db: np.ndarray | None = None
    #: v3 only: the MEASURED floor-shape spread ``sigma_B`` (dB) that scales
    #: the spline's control values, the measured control values themselves
    #: (dB about ``floor_mean_db``) and the per-mic wind prior centres.
    floor_shape_sd_db: float | None = None
    floor_ctrl_db: np.ndarray | None = None
    wind_db: np.ndarray | None = None

    def profile_prior(self, priors: Priors) -> tuple[Tensor, Tensor]:
        """``(loc, scale)`` of the profile prior, per line: v2's two regimes,
        or v3's ONE ``N(measured pooled level, profile_db_sd)`` for every line."""
        if isinstance(priors, PriorsV3):
            loc = np.asarray(self.profile_db, dtype=np.float64)
            scale = np.full(loc.shape, float(priors.profile_db_sd))
        else:
            snr = np.asarray(self.line_snr_db, dtype=np.float64)
            visible = snr >= float(priors.line_visible_snr_db)
            loc = np.where(
                visible,
                np.asarray(self.profile_db, dtype=np.float64),
                float(self.floor_mean_db) + float(priors.profile_below_offset_db),
            )
            scale = np.where(visible, float(priors.profile_db_sd), float(priors.profile_below_sd))
        if self.profile_prior_db is not None:
            ext = np.asarray(self.profile_prior_db, dtype=np.float64)
            loc = np.where(np.isfinite(ext), ext, loc)
        if self.profile_prior_sd_db is not None:
            sd = np.asarray(self.profile_prior_sd_db, dtype=np.float64)
            scale = np.where(np.isfinite(sd), sd, scale)
        return (
            torch.as_tensor(loc, dtype=torch.float64),
            torch.as_tensor(scale, dtype=torch.float64),
        )


def free_blocks(mode: str) -> tuple[str, ...]:
    """Which blocks a mode fits. Anything else must arrive frozen."""
    if mode == "bench":
        # no "carrier": the bench carrier is FROZEN at the support index's
        # window-refined value (rule rev 2), so there is no carrier site and no
        # N(survey, 0.5^2) prior to fight
        return ("dynamics", "profile", "floor", "mic")
    if mode == "flight":
        return ("dynamics", "profile", "floor", "mic")
    if mode == "flight_profile":
        # The MIRROR of ``flight_floor_lowk``: there the bench rig arrives
        # whole and the recording sets only a level; here the recording sets
        # the whole per-order comb and the BENCH sets the dynamics. In DREGON
        # free flight neither the raw nor the refined rotor labels follow the
        # harmonics closely, so a free ``sigma_nu``/``gamma_hz`` absorbs the
        # LABEL's error as shaft wander and line width (R4's free-profile fit
        # on the legacy render landed at a 13 Hz half width at k=1). The
        # dynamics are a property of the ROTOR, measured on a bench where the
        # carrier is known to ~0.003 rev/s, so they are transplanted from the
        # bench fits exactly as the frozen comb of ``flight_floor_lowk`` is
        # (``noise_v2_fit.mean_comb``) and only the profile, the floor and the
        # mic gains move.
        return ("profile", "floor", "mic")
    if mode in ("flight_floor_only", "flight_floor_lowk"):
        # the comb arrives FROZEN from a bench rig, so its absolute level is
        # that rig's, and nothing downstream can re-level it: render_noise
        # mean-centres mic_line_gain_db over mics per rotor and mic_gains_db
        # over mics, which leaves profile_db as the ONLY absolute comb scale.
        # One shared scalar, no more — the comb's SHAPE stays frozen.
        #
        # ``flight_floor_lowk`` frees ONE more thing, and only below
        # ``low_orders``: a per-order gain shared by the rotors. The
        # transplanted bench comb keeps its shape above that order; below it
        # the flight recording sets the level, because the registration study
        # (``results/noise_v2/rounds/round3/registration/findings.md``) finds
        # no resolvable comb above k ~ 8 in the DREGON score windows while
        # k = 1-7 carry the merged low-order power HPPNet tracks. A single
        # scalar spread over 80+ orders that are simply ABSENT from the
        # recording drags those low orders ~20 dB too quiet (R2: the seed
        # wanted +8.5 dB, the one-scalar fit landed at -3.2 dB).
        return (
            ("floor", "mic", "comb_gain")
            if mode == "flight_floor_only"
            else ("floor", "mic", "comb_gain", "low_order_gain")
        )
    # the ATTRIBUTION mode of the four-motor validation: a transfer gap that it
    # closes is a gap in that block alone
    if mode == "bench_dynamics_only":
        return ("dynamics",)
    if mode == V3_MODE:
        # v3: no "mic" block (channels normalised in the data) and nothing
        # frozen. "wind" is sampled only when PriorsV3.wind; "latents" are the
        # per-window block-wander tracks, fitted by the alternation of
        # fit.fit_v3 rather than inside one guide with the rig
        return ("dynamics", "profile", "floor", "wind", "latents")
    raise ValueError(f"unknown mode {mode!r}")


def span_pinned_sites(batch: SupportBatch, *, priors: Priors = PRIORS) -> tuple[str, ...]:
    """The speed-law sites this batch's carrier span cannot identify.

    A bench support has ONE speed and never carries these sites at all; a
    flight pool spanning less than :attr:`Priors.speed_span_pin` in ``max/min``
    carrier gets them as constants at their prior medians.
    """
    if batch.mode != "flight":
        return ()
    return () if batch.speed_span >= float(priors.speed_span_pin) else SPEED_LAW_SITES


# ── the observation ─────────────────────────────────────────────────────────


@dataclass
class WindowLatents:
    """The block-wander latents of ONE window, in dB (explainer §2.3, §2.4).

    ``d`` is ``(R, B)`` (per rotor, shared by its orders), ``v`` ``(R, K, B)``
    (per line), ``u`` ``(B,)`` (floor level) and ``uj`` ``(J, B)`` (per floor
    control point). A track whose measured ``sigma`` is zero is ``None`` — no
    site, no prior term, nothing added.
    """

    d: Tensor | None = None
    v: Tensor | None = None
    u: Tensor | None = None
    uj: Tensor | None = None

    def tracks(self) -> dict[str, Tensor]:
        """``{name: tensor}`` of the tracks this window carries."""
        pairs = (("d", self.d), ("v", self.v), ("u", self.u), ("uj", self.uj))
        return {name: t for name, t in pairs if t is not None}


@dataclass(frozen=True)
class ChannelGains:
    """The per-microphone gains v3 NORMALISES the data by (explainer §2.5).

    ``gains_db`` is ``(M,)``, one per channel of the batch, relative to the
    channels' mean; :meth:`normalise` divides each channel's periodogram by
    ``10^{g_m / 10}``. Read by :func:`.fit.load_channel_gains` from the
    rank-test record (``results/noise_v2/mic_gains/mic_gains.json``).
    """

    gains_db: np.ndarray
    source: str = ""
    rig: str = ""
    rule: str = ""

    def normalise(self, power: np.ndarray) -> np.ndarray:
        """``(M, ...)`` periodogram with channel ``m`` divided by ``10^{g_m/10}``."""
        p = np.asarray(power, dtype=np.float64)
        g = np.asarray(self.gains_db, dtype=np.float64).reshape(-1)
        if g.size != int(p.shape[0]):
            raise ValueError(f"{g.size} channel gains for a {int(p.shape[0])}-mic periodogram")
        return p / (10.0 ** (g / 10.0)).reshape((-1,) + (1,) * (p.ndim - 1))

    def as_dict(self) -> dict[str, Any]:
        return dict(
            gains_db=np.asarray(self.gains_db, dtype=np.float64).tolist(),
            source=self.source,
            rig=self.rig,
            rule=self.rule,
        )


@dataclass
class SupportBatch:
    """One support (or one pool of flight windows) as the objective sees it.

    ``power`` is ``(M, N, F)`` in the support's own periodogram units,
    ``weights`` is ``(N,)`` and ``band`` a boolean ``(F,)`` mask. ``rate_work``
    is the flight chunk's work-grid carrier and is ``None`` on the bench, where
    the carrier is a fitted constant instead.
    """

    mode: str
    name: str
    grid: BenchGrid | FlightGrid
    power: Tensor
    weights: Tensor
    band: Tensor
    band_lo: Tensor
    band_hi: Tensor
    n_rotors: int
    n_mics: int
    k_max: int
    rate_work: Tensor | None = None
    carrier_mean: Tensor | None = None
    carrier_init: Tensor | None = None
    #: Fixed before fitting from the approved prior centre.  This is numerical
    #: integration geometry, not a function of a sampled dynamics parameter;
    #: letting a tensor parameter choose its own truncated lag grid would
    #: detach a piece of the spectral model from the MAP gradient.
    bench_order_groups: list[tuple[int, np.ndarray]] | None = None
    exposure_scale: float = 1.0
    #: ``max / min`` carrier over the pool: the span the speed-law pin reads.
    #: One on the bench, where there is one speed and no speed law at all.
    speed_span: float = 1.0
    #: What the initialiser measured on this batch, in the model's own units.
    #: Two R3 priors are centred on it, so a model built on a batch that was
    #: never measured is an error rather than a fit under R1's flat priors.
    measured: Measured | None = None
    members: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)
    #: FLIGHT only: ``(N,)`` member (window) index of every frame and its
    #: centre time in seconds from the window's first sample. v2 never reads
    #: them; v3 groups frames into blocks with them.
    frame_window: np.ndarray | None = None
    frame_time_s: np.ndarray | None = None
    #: v3 only (:func:`with_blocks`): ``(N,)`` block of every frame inside its
    #: window, each window's block count, and the block length.
    frame_block: np.ndarray | None = None
    window_blocks: tuple[int, ...] = ()
    block_s: float | None = None
    #: v3 only: the per-window block-wander latents :func:`forward` applies —
    #: CONSTANTS of a rig step. ``None`` means all zero, i.e. plain
    #: :func:`.spectrum.flight_model`.
    latents: dict[int, WindowLatents] | None = None

    @property
    def n_cells(self) -> int:
        return int(self.power.shape[0]) * int(self.power.shape[1]) * int(self.band.sum())

    @property
    def window_s(self) -> float:
        """The analysis window's length in seconds: the support's own on the
        bench (one whole-segment periodogram), the STFT frame's in flight."""
        g = self.grid
        n = int(g.n) if isinstance(g, BenchGrid) else int(g.n_fft)
        return n / float(g.sr)

    @property
    def resolution_hz(self) -> float:
        """``1 / (2 T)``: the narrowest half-width this window can resolve, and
        the floor the low-order ``gamma_rk`` check measures against."""
        return 0.5 / self.window_s


def _band_masks(freqs: np.ndarray, band: np.ndarray, device: Any) -> tuple[Tensor, Tensor, Tensor]:
    lo = band & (freqs < BAND_SPLIT_HZ)
    hi = band & (freqs >= BAND_SPLIT_HZ)
    t = lambda m: torch.as_tensor(m, dtype=torch.bool, device=device)  # noqa: E731
    return t(band), t(lo), t(hi)


def bench_batch(
    *,
    name: str,
    power: np.ndarray,
    sr: int,
    carrier_mean: np.ndarray,
    k_cap: int,
    n_samples: int | None = None,
    device: Any = "cpu",
    apply_transfer: bool = True,
    floor_lag: int | None = None,
) -> SupportBatch:
    """Build the bench objective of one stationary support.

    ``power`` is ``(M, 1, F)`` — the support's whole-segment periodogram.
    ``n_samples`` is the segment length; without it the length is recovered as
    ``2 (F - 1)``, which is ``n - 1`` for an ODD segment and stretches the
    model's bin grid against the data's by ``1/n`` (``bench_dregon_Motor1_70``:
    116682 against 116683, i.e. 0.45 bins at 7.2 kHz). Pass the support's own
    ``n_fft``.

    ``carrier_mean`` is the support's FROZEN carrier — the index's
    window-refined value — and the bench model has no carrier parameter: the
    in-fit ``refine_bench_carrier`` re-refinement and the ``N(survey, 0.5^2)``
    nuisance prior are both gone (rule rev 2). The re-refinement searched
    +-1 rev/s around a survey value while the index carrier is a demodulation
    measurement good to ~0.003 rev/s, and on a support with no comb it let the
    MAP carrier walk 3.8 prior sigmas onto a fixed-frequency interferer
    (``results/noise_v2/rounds/round1/bench_diag/findings.md``).
    """
    p = np.asarray(power, dtype=np.float64)
    if p.ndim != 3 or p.shape[1] != 1:
        raise ValueError(f"a bench support carries ONE frame; got power {p.shape}")
    n = 2 * (int(p.shape[2]) - 1) if n_samples is None else int(n_samples)
    if n // 2 + 1 != int(p.shape[2]):
        raise ValueError(f"{n} samples give {n // 2 + 1} bins, not the {int(p.shape[2])} supplied")
    grid = SP.bench_grid(
        n=n, sr=sr, device=device, apply_transfer=apply_transfer, floor_lag=floor_lag
    )
    band, lo, hi = _band_masks(grid.freqs_hz, grid.band, device)
    carrier = np.atleast_1d(np.asarray(carrier_mean, dtype=np.float64))
    k_max = SP.k_max_for_carrier(carrier, sr, k_cap=k_cap)
    # The high-order lag integration grid must be fixed BEFORE the Pyro graph
    # is built.  It uses the R3 prior centres and a 40-nat tail; r_tau itself
    # retains every sampled dynamics parameter in the graph. This avoids a
    # parameter-dependent Python/NumPy truncation in the likelihood while
    # retaining the whole-window result for low orders. Both terms of the law
    # only shorten a line's support, so the prior centre is the conservative
    # (longest) grid: a fitted width larger than the centre decays faster.
    lag_sigma = math.exp(PRIORS.log_sigma_nu[0])
    lag_lam = math.exp(PRIORS.log_lam[0])
    lag_gamma = PRIORS.gamma_per_order_hz * np.arange(1, k_max + 1, dtype=np.float64)
    groups = SP.order_groups(k_max, sigma_nu=lag_sigma, lam=lag_lam, gamma_hz=lag_gamma, sr=sr, n=n)
    return SupportBatch(
        mode="bench",
        name=name,
        grid=grid,
        power=torch.as_tensor(p, dtype=torch.float64, device=device),
        # ONE frame, ONE window, nothing duplicated: the exposure bookkeeping
        # composite_weights exists for has nothing to do here.
        weights=torch.ones(1, dtype=torch.float64, device=device),
        band=band,
        band_lo=lo,
        band_hi=hi,
        n_rotors=int(carrier.size),
        n_mics=int(p.shape[0]),
        k_max=k_max,
        carrier_mean=torch.as_tensor(carrier, dtype=torch.float64, device=device),
        carrier_init=torch.as_tensor(carrier, dtype=torch.float64, device=device),
        bench_order_groups=groups,
        diagnostics=dict(
            grid=dict(grid.diagnostics),
            carrier_rev_s=carrier.tolist(),
            carrier_source="frozen: the support index's window-refined carrier",
            n_samples_source=("support n_fft" if n_samples is not None else "2 (F - 1)"),
            lag_integration=dict(
                source="R3 dynamics-prior centres (fixed before MAP)",
                sigma_nu=lag_sigma,
                lam=lag_lam,
                gamma_hz_per_order=PRIORS.gamma_per_order_hz,
                tail_nats=SP.LAG_SUPPORT_NATS,
                orders=len(groups),
            ),
        ),
    )


def flight_batch(
    *,
    name: str,
    members: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    sr: int,
    n_fft: int,
    hop: int,
    k_cap: int,
    device: Any = "cpu",
    frame_stride: int = 1,
    max_frames: int | None = None,
    apply_transfer: bool = True,
    sr_work: int = SP.SAMPLE_RATE_WORK,
    channel_gains: ChannelGains | None = None,
) -> SupportBatch:
    """Pool flight windows into one objective.

    ``members`` is ``[(support_name, power (M, N, F), carrier_audio (R, T),
    frame_starts (N,))]``. Frames are thinned by ``frame_stride`` — at hop 512
    and NFFT 2048 a stride of 4 selects DISJOINT windows, which is the most
    informative subsample per unit cost and removes the overlap correlation the
    composite weights otherwise carry — and the retained exposure weights are
    scaled back up by the thinning factor so the likelihood keeps the support's
    true exposure against the priors.

    ``channel_gains`` (v3) NORMALISES THE DATA: mic ``m``'s periodogram is
    divided by ``10^{g_m / 10}`` before anything reads it, so the model needs
    no microphone parameter (explainer §2.5). The gains applied and their
    source are recorded in ``diagnostics["channel_gains"]``.
    """
    grid = SP.flight_grid(
        sr=sr, n_fft=n_fft, hop=hop, sr_work=sr_work, device=device, apply_transfer=apply_transfer
    )
    powers: list[np.ndarray] = []
    rates: list[Tensor] = []
    keys: list[tuple[str, int]] = []
    n_total = 0
    carriers: list[np.ndarray] = []
    frame_window: list[np.ndarray] = []
    frame_time: list[np.ndarray] = []
    for w_idx, (mname, power, carrier, starts) in enumerate(members):
        p = np.asarray(power, dtype=np.float64)
        if channel_gains is not None:
            p = channel_gains.normalise(p)
        st = np.asarray(starts, dtype=np.int64)
        n_total += int(st.size)
        sel = np.arange(0, st.size, max(1, int(frame_stride)))
        if max_frames is not None:
            per = max(1, int(max_frames) // max(1, len(members)))
            sel = sel[:per]
        powers.append(p[:, sel, :])
        rates.append(SP.flight_rate_work(grid, carrier, st[sel]))
        keys.extend((mname, int(s)) for s in st[sel])
        carriers.append(np.asarray(carrier, dtype=np.float64))
        frame_window.append(np.full(sel.size, w_idx, dtype=np.int64))
        frame_time.append((st[sel] + 0.5 * float(n_fft)) / float(sr))
    power_t = torch.as_tensor(np.concatenate(powers, axis=1), dtype=torch.float64, device=device)
    rate_t = torch.cat(rates, dim=1)
    n_sel = int(power_t.shape[1])
    scale = float(n_total) / float(n_sel)
    w = composite_weights(keys, hop=hop, n_fft=n_fft) * scale
    band, lo, hi = _band_masks(grid.freqs_hz, grid.band, device)
    c_all = np.concatenate(carriers, axis=1)
    # the span the speed-law pin reads: ONE number per pool, the widest ratio
    # of carriers any rotor of it reaches
    c_lo = float(np.min(c_all[c_all > 0.0])) if np.any(c_all > 0.0) else 0.0
    span = float(c_all.max() / c_lo) if c_lo > 0.0 else float("inf")
    return SupportBatch(
        mode="flight",
        name=name,
        grid=grid,
        power=power_t,
        weights=torch.as_tensor(w, dtype=torch.float64, device=device),
        band=band,
        band_lo=lo,
        band_hi=hi,
        n_rotors=int(c_all.shape[0]),
        n_mics=int(power_t.shape[0]),
        k_max=SP.k_max_for_carrier(c_all.max(axis=1), sr, k_cap=k_cap),
        rate_work=rate_t,
        members=tuple(m[0] for m in members),
        exposure_scale=scale,
        speed_span=span,
        diagnostics=dict(
            grid=dict(grid.diagnostics),
            n_frames_total=n_total,
            n_frames_used=n_sel,
            frame_stride=int(frame_stride),
            exposure_scale=scale,
            carrier_min_rev_s=float(c_all.min()),
            carrier_max_rev_s=float(c_all.max()),
            speed_span=span,
            **({} if channel_gains is None else {"channel_gains": channel_gains.as_dict()}),
        ),
        frame_window=np.concatenate(frame_window),
        frame_time_s=np.concatenate(frame_time),
    )


def batch_slice(batch: SupportBatch, idx: np.ndarray) -> SupportBatch:
    """A frame minibatch of a FLIGHT batch, weights rescaled to the full set.

    Adam runs on minibatches because one Michael's cruise pool is ~500 frames
    of 2048-point STFT at a 64 kHz work grid; the L-BFGS polish then runs on a
    fixed frame set so its objective is deterministic (a line search on a
    resampled objective is not a line search).
    """
    if batch.rate_work is None:
        raise ValueError("batch_slice is for flight batches")
    i = torch.as_tensor(np.asarray(idx, dtype=np.int64), device=batch.power.device)
    scale = float(batch.weights.sum() / batch.weights.index_select(0, i).sum())
    return replace(
        batch,
        power=batch.power.index_select(1, i),
        weights=batch.weights.index_select(0, i) * scale,
        rate_work=batch.rate_work.index_select(1, i),
        **_frame_fields(batch, np.asarray(idx, dtype=np.int64)),
    )


def _frame_fields(batch: SupportBatch, idx: np.ndarray) -> dict[str, Any]:
    """The per-frame bookkeeping arrays of ``batch`` at frames ``idx``."""
    return {
        key: (None if getattr(batch, key) is None else getattr(batch, key)[idx])
        for key in ("frame_window", "frame_time_s", "frame_block")
    }


def window_batch(batch: SupportBatch, window: int) -> SupportBatch:
    """The frames of ONE member window, weights NOT rescaled.

    What the per-window latent step of the v3 alternation fits against: a
    window's latents must see exactly that window's exposure against their OU
    prior, not the pool's (which :func:`batch_slice`'s rescaling would give).
    """
    if batch.rate_work is None or batch.frame_window is None:
        raise ValueError("window_batch is for flight batches with frame bookkeeping")
    idx = np.flatnonzero(batch.frame_window == int(window))
    if idx.size == 0:
        raise ValueError(f"window {window} has no frame in batch {batch.name!r}")
    i = torch.as_tensor(idx, device=batch.power.device)
    return replace(
        batch,
        power=batch.power.index_select(1, i),
        weights=batch.weights.index_select(0, i),
        rate_work=batch.rate_work.index_select(1, i),
        latents=None,
        **_frame_fields(batch, idx),
    )


def with_blocks(batch: SupportBatch, block_s: float) -> SupportBatch:
    """``batch`` with every frame assigned to its window's block of ``block_s``.

    Block ``b`` of a window holds the frames whose CENTRE lies in
    ``[b block_s, (b + 1) block_s)`` from the window's first sample; a window
    has ``1 + max b`` blocks over the frames the batch kept (a block the frame
    thinning left empty is a pure-prior block of the OU chain).
    """
    if batch.frame_window is None or batch.frame_time_s is None:
        raise ValueError("with_blocks needs a flight batch with frame bookkeeping")
    if not (math.isfinite(block_s) and block_s > 0.0):
        raise ValueError(f"block_s must be finite and positive, got {block_s!r}")
    fb = np.floor(np.asarray(batch.frame_time_s) / float(block_s)).astype(np.int64)
    n_win = len(batch.members) if batch.members else int(batch.frame_window.max()) + 1
    counts = tuple(
        int(fb[batch.frame_window == w].max()) + 1 if np.any(batch.frame_window == w) else 0
        for w in range(n_win)
    )
    return replace(batch, frame_block=fb, window_blocks=counts, block_s=float(block_s))


# ── parameters ──────────────────────────────────────────────────────────────


SiteFn = Any


def _pyro_site(name: str, d: Any) -> Tensor:
    return pyro.sample(name, d)


def _normal(site: SiteFn, name: str, loc: Any, scale: float, shape: tuple[int, ...] = ()) -> Tensor:
    d = dist.Normal(torch.as_tensor(loc, dtype=torch.float64), float(scale))
    if shape:
        d = d.expand(shape).to_event(len(shape))
    return site(name, d)


def _lognormal(
    site: SiteFn, name: str, prior: tuple[float, float], shape: tuple[int, ...] = ()
) -> Tensor:
    d = dist.LogNormal(torch.tensor(prior[0], dtype=torch.float64), float(prior[1]))
    if shape:
        d = d.expand(shape).to_event(len(shape))
    return site(name, d)


# ── pinning a dynamics scalar ───────────────────────────────────────────────


def dynamics_pin(spec: dict[str, Any] | None) -> dict[str, float] | None:
    """``{"lam": 0.5}`` validated against the pinnable dynamics sites.

    R3 has exactly two scalar dynamics sites (:data:`DYN_SITES`), and a pin is
    a CONSTANT of the model rather than a tightly-prior'd parameter: the guide
    allocates nothing for it and the log-prior counts nothing for it. The
    per-line ``gamma_hz`` block is not pinned coordinate by coordinate — a
    mode that must hold the comb fixed freezes the whole block instead.
    """
    if not spec:
        return None
    out: dict[str, float] = {}
    for key, value in spec.items():
        if str(key) not in DYN_SITES:
            raise ValueError(f"cannot pin {key!r}; pin one of {sorted(DYN_SITES)}")
        if value is None:
            continue
        out[str(key)] = float(value)
    return out or None


def is_pinned(pin: dict[str, Any] | None, name: str) -> bool:
    """Whether dynamics site ``name`` is a constant rather than a site."""
    return bool(pin) and pin is not None and pin.get(name) is not None


def flight_lam(priors: Priors, frozen: dict[str, Any] | None) -> float:
    """The PINNED shaft rate of a flight fit.

    128 ms of lag does not identify ``lam`` (R1 basin: 0.03 nats/cell over
    three decades), so flight never samples it: a frozen mapping's bench value
    where one exists — the DREGON frozen-comb fit — and
    :attr:`Priors.flight_lam` otherwise, which is Michael's.
    """
    fz = frozen or {}
    return float(fz["lam"]) if "lam" in fz else float(priors.flight_lam)


def pin_applied(pin: dict[str, Any] | None, name: str, centre: float) -> Tensor:
    """A CONCRETE value of site ``name``: the pin if pinned, ``centre``
    otherwise. This is what a probe forward pass must use, so the seeds it
    calibrates are measured at the dynamics the fit will actually run at."""
    value = float(pin[name]) if pin is not None and is_pinned(pin, name) else float(centre)
    return torch.as_tensor(value, dtype=torch.float64)


def _dyn_site(
    site: SiteFn, name: str, prior: tuple[float, float], *, pin: dict[str, Any] | None
) -> Tensor:
    """Dynamics site ``name``, or its pinned constant."""
    if is_pinned(pin, name):
        assert pin is not None
        return torch.as_tensor(float(pin[name]), dtype=torch.float64)
    return _lognormal(site, name, prior)


def sample_params_from_values(
    batch: SupportBatch,
    *,
    mode: str,
    values: dict[str, Tensor],
    priors: Priors = PRIORS,
    frozen: dict[str, Any] | None = None,
    pin: dict[str, Any] | None = None,
    low_orders: int | None = None,
) -> V2Params:
    """Assemble :class:`.spectrum.V2Params` from a guide's ``median()`` dict.

    The SAME construction the model samples through, with the site draw
    replaced by a lookup: a MAP parameter set and the model's own parameter
    set cannot disagree about shapes or which block is frozen.
    """

    def lookup(name: str, _d: Any) -> Tensor:
        if name not in values:
            raise KeyError(f"guide has no site {name!r} (sites: {sorted(values)})")
        return torch.as_tensor(values[name], dtype=torch.float64)

    return sample_params(
        batch,
        mode=mode,
        priors=priors,
        frozen=frozen,
        pin=pin,
        low_orders=low_orders,
        site=lookup,
    )


def sample_params(
    batch: SupportBatch,
    *,
    mode: str,
    priors: Priors = PRIORS,
    frozen: dict[str, Any] | None = None,
    pin: dict[str, Any] | None = None,
    low_orders: int | None = None,
    site: SiteFn = _pyro_site,
) -> V2Params:
    """Sample (or read frozen) every parameter of one support's forward model.

    A frozen block creates NO Pyro site, so ``AutoDelta`` never allocates a
    guide parameter for it: ``--floor-only`` really holds the comb's SHAPE,
    its per-line widths and the shaft fixed rather than fitting them under a
    tight prior — its only comb freedom is the single ``comb_gain_db`` scalar
    of the ``"comb_gain"`` block, which re-levels the transplanted comb as a
    whole. Three further sites are CONSTANTS rather than parameters where the
    data cannot see them: ``lam`` in flight (128 ms of lag does not identify
    it: the frozen mapping's bench value, or :attr:`Priors.flight_lam`), and
    the three speed laws on a pool that barely changes speed
    (:func:`span_pinned_sites`). ``pin`` does the same for a named dynamics
    scalar of an otherwise free block.
    """
    if mode == V3_MODE:
        return _sample_params_v3(batch, priors=priors, frozen=frozen, pin=pin, site=site)
    free = free_blocks(mode)
    fz = dict(frozen or {})
    r, m, k = batch.n_rotors, batch.n_mics, batch.k_max
    flight = batch.mode == "flight"
    span_pinned = span_pinned_sites(batch, priors=priors)
    medians = priors.speed_law_medians()

    def take(block: str, key: str, shape: tuple[int, ...] | None = None) -> Tensor:
        v = torch.as_tensor(np.asarray(fz[key], dtype=np.float64), dtype=torch.float64)
        if shape is not None and tuple(v.shape) != shape:
            if v.numel() == 1:
                v = v.reshape(()).expand(shape).clone()
            elif key in ("profile_db", "gamma_hz") and v.ndim == 2 and int(v.shape[1]) >= shape[1]:
                v = v[: shape[0], : shape[1]].clone()
            elif key in ("profile_db", "gamma_hz") and v.ndim == 2 and int(v.shape[0]) == 1:
                v = v[:, : shape[1]].expand(shape).clone()
            else:
                raise ValueError(f"frozen {key} has shape {tuple(v.shape)}, need {shape}")
        return v.to(device=batch.power.device)

    def speed_law(name: str, draw: Any) -> Tensor:
        """A speed law: its site, or its prior median where the span pins it."""
        if not flight:
            return torch.zeros((), dtype=torch.float64)
        if name in span_pinned:
            return torch.as_tensor(medians[name], dtype=torch.float64)
        return draw()

    if "dynamics" in free:
        sigma_nu = _dyn_site(site, "sigma_nu", priors.sigma_nu_prior(flight), pin=pin)
        if flight:
            # PINNED in flight: the bench value a frozen mapping carries, or
            # the approved default. 128 ms of lag cannot see the shaft rate
            # (R1 basin: 0.03 nats/cell over three decades of lam).
            lam = pin_applied(pin, "lam", flight_lam(priors, fz))
        else:
            lam = _dyn_site(site, "lam", priors.log_lam, pin=pin)
        if "gamma_hz" in fz:
            gamma_hz = take("dynamics", "gamma_hz", (r, k))
        else:
            orders = np.broadcast_to(np.arange(1, k + 1, dtype=np.float64), (r, k))
            gamma_hz = site(
                "gamma_hz",
                dist.LogNormal(priors.gamma_loc(orders), float(priors.gamma_log_sd)).to_event(2),
            )
    else:
        sigma_nu = take("dynamics", "sigma_nu")
        lam = take("dynamics", "lam")
        gamma_hz = take("dynamics", "gamma_hz", (r, k))

    zero = torch.zeros((), dtype=torch.float64)
    if "profile" in free:
        if batch.measured is None:
            raise ValueError(
                f"batch {batch.name!r} was never measured: the R3 profile and floor priors are "
                "centred on the data (fit.measure_batch, carried on SupportBatch.measured)"
            )
        loc, scale = batch.measured.profile_prior(priors)
        profile_db = site("profile_db", dist.Normal(loc[:r, :k], scale[:r, :k]).to_event(2))
        amp_exp = speed_law("amp_exp", lambda: _normal(site, "amp_exp", *priors.amp_exp))
    else:
        profile_db = take("profile", "profile_db", (r, k))
        amp_exp = take("profile", "amp_exp") if flight else zero

    if "comb_gain" in free:
        # ONE shared scalar, folded into profile_db right here, so write_fit,
        # params_to_dict, render_noise and expected_periodogram need no change
        # at all: the comb they read is already at its fitted level
        profile_db = profile_db + _normal(site, "comb_gain_db", *priors.comb_gain_db)

    if "low_order_gain" in free:
        # ...and, below ``low_orders``, one gain PER ORDER shared by the
        # rotors, on top of that scalar and folded in the same way. Above the
        # cut the transplanted comb keeps its bench shape exactly.
        kk = min(int(low_orders if low_orders is not None else priors.low_orders), k)
        if kk < 1:
            raise ValueError(f"flight_floor_lowk needs low_orders >= 1, got {low_orders!r}")
        low = _normal(site, "low_order_gain_db", 0.0, priors.low_order_gain_db[1], (kk,))
        pad = torch.zeros(k, dtype=torch.float64, device=profile_db.device)
        profile_db = profile_db + torch.cat([low, pad[kk:]])[None, :]

    if "floor" in free:
        if batch.measured is None:
            raise ValueError(f"batch {batch.name!r} was never measured (see fit.measure_batch)")
        floor = FloorParams(
            mean_db=_normal(
                site,
                "floor_mean_db",
                float(batch.measured.floor_mean_db),
                priors.floor_mean_db_sd,
            ),
            shape_z=_normal(site, "floor_shape_z", 0.0, 1.0, (FLOOR_SHAPE_N_CTRL,)),
            tilt_db_oct=_normal(site, "floor_tilt_db_oct", *priors.floor_tilt_db_oct),
            mic_floor_db=_normal(site, "mic_floor_db", 0.0, priors.mic_floor_db, (m,)),
            exp=speed_law("floor_exp", lambda: _lognormal(site, "floor_exp", priors.log_floor_exp)),
            static_rel=speed_law(
                "floor_static_rel",
                lambda: _lognormal(site, "floor_static_rel", priors.log_floor_static),
            ),
        )
    else:
        floor = FloorParams(
            mean_db=take("floor", "floor_mean_db"),
            shape_z=take("floor", "floor_shape_z", (FLOOR_SHAPE_N_CTRL,)),
            tilt_db_oct=take("floor", "floor_tilt_db_oct"),
            mic_floor_db=take("floor", "mic_floor_db", (m,)),
            exp=take("floor", "floor_exp") if flight else zero,
            static_rel=take("floor", "floor_static_rel") if flight else zero,
        )

    if "mic" in free:
        mic_line = _normal(site, "mic_line_gain_db", 0.0, priors.mic_line_gain_db, (m, r))
        gain_all = _normal(site, "gain_all_db", 0.0, priors.gain_all_db, (m,))
    else:
        mic_line = take("mic", "mic_line_gain_db", (m, r))
        gain_all = take("mic", "gain_all_db", (m,))

    carrier = None
    if batch.mode == "bench":
        if batch.carrier_mean is None:
            raise ValueError("a bench batch needs carrier_mean (its frozen carrier)")
        # a CONSTANT of the model, not a site: the support index's
        # window-refined carrier is a demodulation measurement good to
        # ~0.003 rev/s and the fit has nothing to add to it (rule rev 2). A
        # frozen mapping may still override it, which is what the four-motor
        # attribution modes read.
        carrier = (
            take("carrier", "carrier_rev_s", (r,))
            if "carrier_rev_s" in fz
            else torch.as_tensor(batch.carrier_mean, dtype=torch.float64)
        )

    return V2Params(
        sigma_nu=sigma_nu,
        lam=lam,
        gamma_hz=gamma_hz,
        profile_db=profile_db,
        floor=floor,
        mic_line_gain_db=mic_line,
        gain_all_db=gain_all,
        carrier_rev_s=carrier,
        amp_exp=amp_exp,
    )


def _sample_params_v3(
    batch: SupportBatch,
    *,
    priors: Priors,
    frozen: dict[str, Any] | None,
    pin: dict[str, Any] | None,
    site: SiteFn,
) -> V2Params:
    """Every RIG parameter of the v3 flight model (mode :data:`V3_MODE`).

    Differences from v2, each one a line of the explainer's §2.6 table:
    ``sigma_nu`` and ``gamma_hz`` are half-normal in linear units, the profile
    prior is ``N(measured, profile_db_sd)`` for every line, the floor has NO
    ``floor_mean_db``/``floor_tilt_db_oct`` site (``mu`` and ``sigma_B`` are
    measured constants, ``floor_shape_z`` the only shape site), and there is no
    microphone site at all — the three v2 mic blocks are zeros, which the
    mean-pinned forward model reads as unit gains. ``wind_db`` is a site only
    when :attr:`PriorsV3.wind`. The block latents are NOT sampled here: they
    are per window (:func:`sample_window_latents`, :func:`latent_model_cached`).
    """
    if not isinstance(priors, PriorsV3):
        raise TypeError(f"mode {V3_MODE!r} needs PriorsV3, got {type(priors).__name__}")
    if frozen:
        raise ValueError(f"mode {V3_MODE!r} freezes nothing; got frozen keys {sorted(frozen)}")
    if batch.mode != "flight":
        raise ValueError(f"mode {V3_MODE!r} is a flight mode; batch {batch.name!r} is {batch.mode}")
    meas = batch.measured
    if meas is None or meas.floor_shape_sd_db is None:
        raise ValueError(
            f"batch {batch.name!r} carries no v3 measurement (fit.measure_batch with mode "
            f"{V3_MODE!r}): the profile centre, mu and sigma_B are measured"
        )
    r, m, k = batch.n_rotors, batch.n_mics, batch.k_max
    span_pinned = span_pinned_sites(batch, priors=priors)
    medians = priors.speed_law_medians()
    zero = torch.zeros((), dtype=torch.float64)

    def speed_law(name: str, draw: Any) -> Tensor:
        if name in span_pinned:
            return torch.as_tensor(medians[name], dtype=torch.float64)
        return draw()

    if is_pinned(pin, "sigma_nu"):
        assert pin is not None
        sigma_nu = torch.as_tensor(float(pin["sigma_nu"]), dtype=torch.float64)
    else:
        sigma_nu = site(
            "sigma_nu",
            dist.HalfNormal(torch.tensor(float(priors.sigma_nu_scale), dtype=torch.float64)),
        )
    lam = pin_applied(pin, "lam", flight_lam(priors, None))
    orders = np.broadcast_to(np.arange(1, k + 1, dtype=np.float64), (r, k))
    gamma_hz = site("gamma_hz", dist.HalfNormal(priors.gamma_scale(orders)).to_event(2))

    loc, scale = meas.profile_prior(priors)
    profile_db = site("profile_db", dist.Normal(loc[:r, :k], scale[:r, :k]).to_event(2))
    amp_exp = speed_law("amp_exp", lambda: _normal(site, "amp_exp", *priors.amp_exp))

    floor = FloorParams(
        mean_db=torch.as_tensor(float(meas.floor_mean_db), dtype=torch.float64),
        shape_z=_normal(site, "floor_shape_z", 0.0, 1.0, (FLOOR_SHAPE_N_CTRL,)),
        tilt_db_oct=zero,
        mic_floor_db=torch.zeros(m, dtype=torch.float64),
        exp=speed_law("floor_exp", lambda: _lognormal(site, "floor_exp", priors.log_floor_exp)),
        static_rel=speed_law(
            "floor_static_rel",
            lambda: _lognormal(site, "floor_static_rel", priors.log_floor_static),
        ),
        shape_sd_db=torch.as_tensor(float(meas.floor_shape_sd_db), dtype=torch.float64),
    )

    wind_db = None
    if priors.wind:
        if meas.wind_db is None:
            raise ValueError(f"batch {batch.name!r}: --wind needs the measured wind centres")
        wind_db = _normal(
            site, "wind_db", np.asarray(meas.wind_db, dtype=np.float64), priors.wind_db_sd, (m,)
        )

    return V2Params(
        sigma_nu=sigma_nu,
        lam=lam,
        gamma_hz=gamma_hz,
        profile_db=profile_db,
        floor=floor,
        mic_line_gain_db=torch.zeros(m, r, dtype=torch.float64),
        gain_all_db=torch.zeros(m, dtype=torch.float64),
        carrier_rev_s=None,
        amp_exp=amp_exp,
        wind_db=wind_db,
    )


def log_prior(
    batch: SupportBatch,
    *,
    mode: str,
    values: dict[str, Tensor],
    priors: Priors = PRIORS,
    pin: dict[str, Any] | None = None,
) -> float:
    """``sum log p(site)`` of the rig sites at ``values`` (a guide's medians).

    The SAME construction the model samples through, with each draw replaced
    by a lookup that also accumulates the site's log density — what the v3
    record adds to the Whittle term to report a total objective."""
    acc: list[Tensor] = []

    def lookup(name: str, d: Any) -> Tensor:
        v = torch.as_tensor(values[name], dtype=torch.float64)
        acc.append(d.log_prob(v).sum())
        return v

    sample_params(batch, mode=mode, priors=priors, pin=pin, site=lookup)
    return float(sum(float(a) for a in acc))


def detach_params(params: V2Params) -> V2Params:
    """``params`` with every tensor detached: the rig as CONSTANTS."""

    def d(v: Any) -> Any:
        return v.detach() if isinstance(v, Tensor) else v

    fl = params.floor
    return V2Params(
        sigma_nu=d(params.sigma_nu),
        lam=d(params.lam),
        gamma_hz=d(params.gamma_hz),
        profile_db=d(params.profile_db),
        floor=FloorParams(**{f.name: d(getattr(fl, f.name)) for f in fields(FloorParams)}),
        mic_line_gain_db=d(params.mic_line_gain_db),
        gain_all_db=d(params.gain_all_db),
        carrier_rev_s=d(params.carrier_rev_s),
        amp_exp=d(params.amp_exp),
        wind_db=d(params.wind_db),
    )


# ── v3: the block wander ────────────────────────────────────────────────────


def ou_log_density(x: Tensor, sigma: float, rho: float) -> Tensor:
    """``log N(x; 0, sigma^2 rho^|i - j|)`` along the LAST axis (explainer §3.3).

    The stationary OU sampled at the block rate, as its Markov factorisation:

        -1/2 [x_1^2 / s^2 + sum_{b>=2} (x_b - rho x_{b-1})^2 / (s^2 (1 - rho^2))]
        - B/2 log(2 pi s^2) - (B - 1)/2 log(1 - rho^2)

    — the quadratic form plus the log-determinant, which is constant here
    because ``sigma`` and ``rho`` are MEASURED and fixed. One value per track
    (every leading index).
    """
    s2 = float(sigma) ** 2
    one = 1.0 - float(rho) ** 2
    n = int(x.shape[-1])
    quad = x[..., 0] ** 2 / s2
    if n > 1:
        quad = quad + ((x[..., 1:] - float(rho) * x[..., :-1]) ** 2).sum(dim=-1) / (s2 * one)
    logdet = n * math.log(2.0 * math.pi * s2) + (n - 1) * math.log(one)
    return -0.5 * (quad + logdet)


class OUChain(dist.TorchDistribution):
    """A stationary OU track over ``n_blocks`` blocks as ONE Pyro event.

    ``log_prob`` is :func:`ou_log_density`; ``sample`` the exact recursion
    (the renderer's :func:`data_processing.noise_model.v3.ou_blocks`). What a
    per-window latent site of the v3 model is drawn from.
    """

    arg_constraints: dict[str, Any] = {}  # noqa: RUF012

    @property
    def support(self) -> Any:
        return constraints.real_vector

    def __init__(
        self, sigma: float, rho: float, n_blocks: int, batch_shape: tuple[int, ...] = ()
    ) -> None:
        if not (float(sigma) > 0.0 and 0.0 <= float(rho) < 1.0 and int(n_blocks) >= 1):
            raise ValueError(
                f"OUChain needs sigma > 0, 0 <= rho < 1, n >= 1: {sigma}, {rho}, {n_blocks}"
            )
        self.sigma, self.rho, self.n_blocks = float(sigma), float(rho), int(n_blocks)
        super().__init__(torch.Size(batch_shape), torch.Size([self.n_blocks]), validate_args=False)

    def expand(self, batch_shape: Any, _instance: Any = None) -> OUChain:
        return OUChain(self.sigma, self.rho, self.n_blocks, tuple(torch.Size(batch_shape)))

    def sample(self, sample_shape: Any = torch.Size()) -> Tensor:
        shape = torch.Size(sample_shape) + self.batch_shape + self.event_shape
        e = torch.randn(shape, dtype=torch.float64)
        out = torch.empty_like(e)
        out[..., 0] = self.sigma * e[..., 0]
        innov = math.sqrt(1.0 - self.rho**2) * self.sigma
        for b in range(1, self.n_blocks):
            out[..., b] = self.rho * out[..., b - 1] + innov * e[..., b]
        return out

    def log_prob(self, value: Tensor) -> Tensor:
        return ou_log_density(value, self.sigma, self.rho)


def _latent_shapes(n_rotors: int, k_max: int) -> dict[str, tuple[int, ...]]:
    """The leading (per-track) shape of each latent, before the block axis."""
    return dict(d=(n_rotors,), v=(n_rotors, k_max), u=(), uj=(FLOOR_SHAPE_N_CTRL,))


def sample_window_latents(
    site: SiteFn, wander: Wander, *, n_rotors: int, k_max: int, n_blocks: int
) -> WindowLatents:
    """ONE window's block latents as sites ``wander_{d,v,u,uj}`` under their
    OU priors; a track whose measured ``sigma`` is zero has no site."""
    out: dict[str, Tensor | None] = {}
    for name, shape in _latent_shapes(n_rotors, k_max).items():
        if not wander.active(name):
            out[name] = None
            continue
        d: Any = OUChain(wander.sigma(name), wander.rho(name), n_blocks)
        if shape:
            d = d.expand(shape).to_event(len(shape))
        out[name] = site(f"wander_{name}", d)
    return WindowLatents(**out)


def zero_latents(wander: Wander, *, n_rotors: int, k_max: int, n_blocks: int) -> WindowLatents:
    """All-zero latents of one window, on the tracks ``wander`` carries."""
    return WindowLatents(
        **{
            name: (
                torch.zeros(shape + (n_blocks,), dtype=torch.float64)
                if wander.active(name)
                else None
            )
            for name, shape in _latent_shapes(n_rotors, k_max).items()
        }
    )


def block_params(params: V2Params, latents: WindowLatents | None, block: int) -> V2Params:
    """The rig with ONE block's latents applied — the explainer's §3.2.

    Line ``(r, k)``'s power is multiplied by ``10^{(d_r + v_rk)/10}`` (added to
    ``profile_db``) and the floor's by ``10^{(u + sum_j B_j(f) u_j)/10}`` (``u``
    added to its level, ``u_j`` to its control values). Nothing else moves;
    with every latent at zero this IS ``params`` to the last bit.
    """
    if latents is None:
        return params
    b = int(block)
    line: Tensor | None = None
    if latents.d is not None:
        line = latents.d[:, b][:, None]
    if latents.v is not None:
        line = latents.v[:, :, b] if line is None else line + latents.v[:, :, b]
    profile = params.profile_db
    if line is not None:
        profile = torch.as_tensor(profile, dtype=torch.float64) + line
    floor = params.floor
    if latents.u is not None or latents.uj is not None:
        floor = replace(
            floor,
            mean_db=(
                torch.as_tensor(floor.mean_db, dtype=torch.float64) + latents.u[b]
                if latents.u is not None
                else floor.mean_db
            ),
            ctrl_offset_db=latents.uj[:, b] if latents.uj is not None else floor.ctrl_offset_db,
        )
    return replace(params, profile_db=profile, floor=floor)


def forward_v3(
    batch: SupportBatch, params: V2Params, latents: dict[int, WindowLatents], **kw: Any
) -> Tensor:
    """``(M, N, F)`` expected periodogram with per-block latents (§3.2).

    The frames are grouped by (window, block) and each group goes through
    :func:`.spectrum.flight_model` UNCHANGED with :func:`block_params` —
    the v2 forward model with two per-block multipliers, the line-shape
    kernel untouched. A window with no entry in ``latents`` is at zero.
    """
    assert isinstance(batch.grid, FlightGrid) and batch.rate_work is not None
    if batch.frame_window is None or batch.frame_block is None:
        raise ValueError(f"batch {batch.name!r} has no block assignment (model.with_blocks)")
    fw = np.asarray(batch.frame_window, dtype=np.int64)
    fb = np.asarray(batch.frame_block, dtype=np.int64)
    key = fw * (int(fb.max()) + 1) + fb
    order = np.argsort(key, kind="stable")
    cuts = np.flatnonzero(np.diff(key[order])) + 1
    dev = batch.rate_work.device
    pieces: list[Tensor] = []
    for seg in np.split(order, cuts):
        p_g = block_params(params, latents.get(int(fw[seg[0]])), int(fb[seg[0]]))
        idx = torch.as_tensor(seg, dtype=torch.int64, device=dev)
        pieces.append(
            SP.flight_model(
                batch.grid,
                p_g,
                rate_work=batch.rate_work.index_select(1, idx),
                k_max=batch.k_max,
                **kw,
            )
        )
    out = torch.cat(pieces, dim=1)
    if np.array_equal(order, np.arange(order.size)):
        return out
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    return out.index_select(1, torch.as_tensor(inv, dtype=torch.int64, device=dev))


@dataclass
class LatentCache:
    """Step (ii)'s forward model with the rig FIXED, precomputed once per rig.

    With the rig a constant a block latent moves two multipliers and nothing
    else (:func:`block_params`): line ``(r, k)`` of a frame in block ``b`` is
    ``10^{(d_r(b) + v_rk(b))/10}`` times its rig spectrum and the floor's dB
    curve moves by ``u(b) + S u_j(b)``. ``flight`` holds the rig's per-line
    spectra and floor atoms (:func:`.spectrum.flight_cache`), so an evaluation
    is elementwise work, one batched matrix product and the floor's FFTs
    (:func:`.spectrum.flight_model_cached`) instead of :func:`forward_v3`'s
    ``R K N`` line kernels.

    ``windows`` are the batch windows the cache covers, each with its
    ``n_blocks``; a latent track is laid out ``(W, ..., B)`` with ``B`` the
    largest count (a shorter window's tail blocks touch no frame) and
    ``frame_track`` ``(N,)`` holds ``w B + b`` of every frame.
    """

    grid: FlightGrid
    flight: SP.FlightCache
    windows: tuple[int, ...]
    n_blocks: tuple[int, ...]
    frame_track: Tensor

    @property
    def n_block_max(self) -> int:
        return max(self.n_blocks)

    def expected(
        self,
        d: Tensor | None = None,
        v: Tensor | None = None,
        u: Tensor | None = None,
        uj: Tensor | None = None,
    ) -> Tensor:
        """``(M, N, F)`` at latents ``d (W, R, B)``, ``v (W, R, K, B)``,
        ``u (W, B)``, ``uj (W, J, B)`` (``None``: zero)."""
        n_tracks = len(self.windows) * self.n_block_max
        ft = self.frame_track
        line: Tensor | None = None
        if d is not None:
            line = d.permute(1, 0, 2).reshape(d.shape[1], n_tracks).index_select(1, ft)[..., None]
        if v is not None:
            per = v.permute(1, 2, 0, 3).reshape(v.shape[1], v.shape[2], n_tracks)
            vv = per.index_select(2, ft).permute(0, 2, 1)
            line = vv if line is None else line + vv
        floor_db: Tensor | None = None
        if u is not None:
            floor_db = u.reshape(n_tracks)[:, None]
        if uj is not None:
            add = uj.permute(0, 2, 1).reshape(n_tracks, -1) @ self.grid.floor.shape_psd.T
            floor_db = add if floor_db is None else floor_db + add
        return SP.flight_model_cached(
            self.grid,
            self.flight,
            line_db=line,
            floor_db=floor_db,
            floor_group=None if floor_db is None else ft,
        )


def latent_cache(
    batch: SupportBatch,
    params: V2Params,
    *,
    chunk_frames: int | None = None,
    line_dtype: torch.dtype = torch.float64,
) -> LatentCache:
    """The :class:`LatentCache` of the rig ``params`` on ``batch``'s frames."""
    assert isinstance(batch.grid, FlightGrid) and batch.rate_work is not None
    if batch.frame_window is None or batch.frame_block is None:
        raise ValueError(f"batch {batch.name!r} has no block assignment (model.with_blocks)")
    fw = np.asarray(batch.frame_window, dtype=np.int64)
    fb = np.asarray(batch.frame_block, dtype=np.int64)
    windows = tuple(int(w) for w in np.unique(fw))
    n_blocks = tuple(int(batch.window_blocks[w]) for w in windows)
    local = np.searchsorted(np.asarray(windows, dtype=np.int64), fw)
    track = torch.as_tensor(
        local * max(n_blocks) + fb, dtype=torch.int64, device=batch.power.device
    )
    flight = SP.flight_cache(
        batch.grid,
        params,
        rate_work=batch.rate_work,
        k_max=batch.k_max,
        chunk_frames=chunk_frames,
        line_dtype=line_dtype,
    )
    return LatentCache(
        grid=batch.grid, flight=flight, windows=windows, n_blocks=n_blocks, frame_track=track
    )


def latent_model_cached(
    batch: SupportBatch, cache: LatentCache, *, wander: Wander
) -> WindowLatents:
    """The Pyro model of ONE window's latents with the rig held FIXED.

    Step (ii) of the explainer's §3.4 alternation: the windows are
    conditionally independent given the rig — what a per-window plate
    asserts — so each is its own small problem over ``wander_*`` sites, fitted
    against exactly that window's frames and exposure (:func:`window_batch`),
    through the window's :class:`LatentCache`.
    """
    if len(cache.windows) != 1:
        raise ValueError(f"one window per cache here, got {len(cache.windows)}")
    lat = sample_window_latents(
        _pyro_site,
        wander,
        n_rotors=batch.n_rotors,
        k_max=batch.k_max,
        n_blocks=cache.n_blocks[0],
    )

    def one(x: Tensor | None) -> Tensor | None:
        return None if x is None else x[None]

    m_model = cache.expected(one(lat.d), one(lat.v), one(lat.u), one(lat.uj))
    pyro.factor("whittle", -whittle_risk(batch, m_model))
    return lat


def ou_prior_nats(latents: dict[int, WindowLatents], wander: Wander) -> float:
    """``-sum log p(latents)`` over every window and track (the OU priors)."""
    total = 0.0
    for lat in latents.values():
        for name, x in lat.tracks().items():
            total -= float(ou_log_density(x.detach(), wander.sigma(name), wander.rho(name)).sum())
    return total


def forward(batch: SupportBatch, params: V2Params, **kw: Any) -> Tensor:
    """``(M, N, F)`` expected periodogram of the batch under ``params`` (and,
    for v3, the batch's own block latents held as constants)."""
    if batch.mode == "bench":
        assert isinstance(batch.grid, BenchGrid)
        return SP.bench_model(
            batch.grid, params, k_max=batch.k_max, groups=batch.bench_order_groups, **kw
        )
    assert isinstance(batch.grid, FlightGrid) and batch.rate_work is not None
    if batch.latents is not None:
        return forward_v3(batch, params, batch.latents, **kw)
    return SP.flight_model(batch.grid, params, rate_work=batch.rate_work, k_max=batch.k_max, **kw)


def whittle_risk(batch: SupportBatch, m_model: Tensor) -> Tensor:
    """``sum_i w_i sum_{m, f in band} [I / M + log M]`` over the whole band."""
    return composite_risk(batch.power, m_model, batch.weights, band=batch.band)


def objective_breakdown(batch: SupportBatch, m_model: Tensor) -> dict[str, Any]:
    """The objective block of the fit JSON: total and the 300 Hz split."""
    with torch.no_grad():
        total = float(whittle_risk(batch, m_model))
        lo = float(composite_risk(batch.power, m_model, batch.weights, band=batch.band_lo))
        hi = float(composite_risk(batch.power, m_model, batch.weights, band=batch.band_hi))
    return dict(
        whittle_nats=total,
        n_cells=batch.n_cells,
        per_band=dict(floor=lo, comb=hi),
        band_split_hz=BAND_SPLIT_HZ,
        band_hz=[SP.BAND_F_MIN, float(batch.grid.freqs_hz[np.asarray(batch.band.cpu())].max())],
        n_cells_per_band=dict(
            floor=int(batch.power.shape[0]) * int(batch.power.shape[1]) * int(batch.band_lo.sum()),
            comb=int(batch.power.shape[0]) * int(batch.power.shape[1]) * int(batch.band_hi.sum()),
        ),
        exposure_scale=batch.exposure_scale,
    )


def support_model(
    batch: SupportBatch,
    *,
    mode: str,
    priors: Priors = PRIORS,
    frozen: dict[str, Any] | None = None,
    pin: dict[str, Any] | None = None,
    low_orders: int | None = None,
    temperature: float = 1.0,
    forward_kw: dict[str, Any] | None = None,
) -> V2Params:
    """The Pyro model of one support: priors, forward model, one Whittle factor."""
    if not (math.isfinite(temperature) and temperature > 0.0):
        raise ValueError(f"temperature must be finite and positive, got {temperature!r}")
    params = sample_params(
        batch, mode=mode, priors=priors, frozen=frozen, pin=pin, low_orders=low_orders
    )
    m_model = forward(batch, params, **(forward_kw or {}))
    pyro.factor("whittle", -whittle_risk(batch, m_model) / float(temperature))
    return params


# ── export / import ─────────────────────────────────────────────────────────


def params_to_dict(params: V2Params) -> dict[str, Any]:
    """The ``params`` block of the ``noise-v2-fit/2`` JSON.

    ``gamma_hz`` is the ``(R, K)`` block of per-line Lorentzian half-widths
    that replaced R1's four ``*_eps`` scalars and the fixed exponent ``p``.
    """
    f = lambda v: float(np.asarray(v.detach().cpu() if isinstance(v, Tensor) else v))  # noqa: E731
    a = lambda v: np.asarray(  # noqa: E731
        v.detach().cpu() if isinstance(v, Tensor) else v, dtype=np.float64
    ).tolist()
    prof = np.atleast_2d(np.asarray(a(params.profile_db), dtype=np.float64))
    gamma = np.asarray(a(params.gamma_hz), dtype=np.float64)
    gamma = np.broadcast_to(np.atleast_2d(gamma), prof.shape) if gamma.size else gamma
    return dict(
        sigma_nu=f(params.sigma_nu),
        lam=f(params.lam),
        gamma_hz=gamma.tolist(),
        carrier_rev_s=(a(params.carrier_rev_s) if params.carrier_rev_s is not None else None),
        profile=dict(
            profile_db=prof.tolist(),
            amp_exp=f(params.amp_exp),
            mic_line_gain_db=a(params.mic_line_gain_db),
        ),
        floor=dict(
            floor_mean_db=f(params.floor.mean_db),
            floor_shape_z=a(params.floor.shape_z),
            floor_tilt_db_oct=f(params.floor.tilt_db_oct),
            floor_exp=f(params.floor.exp),
            floor_static_rel=f(params.floor.static_rel),
            mic_floor_db=a(params.floor.mic_floor_db),
        ),
        mic_gains_db=a(params.gain_all_db),
    )


def params_to_dict_v3(params: V2Params, *, wander: Wander | None) -> dict[str, Any]:
    """The ``params`` block of a ``noise-v3-fit/1`` JSON: the RIG only.

    No microphone block (the channels were normalised in the data); the floor
    as ``mu`` (``floor_mean_db``, measured), the GP coordinate ``z`` and the
    measured ``sigma_B`` (``floor_shape_sd_db``) with no tilt; the per-mic wind
    level with its fixed shape when fitted; and the MEASURED wander
    hyperparameters the renderer draws fresh block latents from. The fitted
    per-window latents are nuisance and live in the record's ``latents``
    block, not here.
    """
    f = lambda v: float(np.asarray(v.detach().cpu() if isinstance(v, Tensor) else v))  # noqa: E731
    a = lambda v: np.asarray(  # noqa: E731
        v.detach().cpu() if isinstance(v, Tensor) else v, dtype=np.float64
    ).tolist()
    if params.floor.shape_sd_db is None:
        raise ValueError("a v3 parameter set carries the measured sigma_B (floor.shape_sd_db)")
    prof = np.atleast_2d(np.asarray(a(params.profile_db), dtype=np.float64))
    gamma = np.broadcast_to(
        np.atleast_2d(np.asarray(a(params.gamma_hz), dtype=np.float64)), prof.shape
    )
    return dict(
        sigma_nu=f(params.sigma_nu),
        lam=f(params.lam),
        gamma_hz=gamma.tolist(),
        carrier_rev_s=None,
        profile=dict(profile_db=prof.tolist(), amp_exp=f(params.amp_exp)),
        floor=dict(
            floor_mean_db=f(params.floor.mean_db),
            floor_shape_z=a(params.floor.shape_z),
            floor_shape_sd_db=f(params.floor.shape_sd_db),
            floor_exp=f(params.floor.exp),
            floor_static_rel=f(params.floor.static_rel),
        ),
        wind=(
            None
            if params.wind_db is None
            else dict(wind_db=a(params.wind_db), shape=wind_shape_spec())
        ),
        wander=None if wander is None else wander.as_params(),
    )


def params_from_dict(
    d: dict[str, Any], *, device: Any = "cpu", n_mics: int | None = None
) -> V2Params:
    """Inverse of :func:`params_to_dict` (what the renderer and a frozen-comb
    flight fit read), for a ``/2`` payload or a mapped ``/1`` one
    (:func:`gamma_from_params`) — and of :func:`params_to_dict_v3`, whose
    missing microphone block reads as zeros over ``n_mics`` microphones (the
    wind block's count when it has one, else ``n_mics``, else one)."""

    def t(v: Any) -> Tensor:
        return torch.as_tensor(np.asarray(v, dtype=np.float64), dtype=torch.float64, device=device)

    prof, floor = d["profile"], d["floor"]
    if "floor_shape_sd_db" in floor:
        wind = d.get("wind")
        n_m = len(wind["wind_db"]) if wind is not None else int(n_mics or 1)
        n_r = int(np.atleast_2d(np.asarray(prof["profile_db"])).shape[0])
        return V2Params(
            sigma_nu=t(d["sigma_nu"]),
            lam=t(d["lam"]),
            gamma_hz=t(gamma_from_params(d)),
            profile_db=t(prof["profile_db"]),
            floor=FloorParams(
                mean_db=t(floor["floor_mean_db"]),
                shape_z=t(floor["floor_shape_z"]),
                tilt_db_oct=t(0.0),
                mic_floor_db=t(np.zeros(n_m)),
                exp=t(floor["floor_exp"]),
                static_rel=t(floor["floor_static_rel"]),
                shape_sd_db=t(floor["floor_shape_sd_db"]),
            ),
            mic_line_gain_db=t(np.zeros((n_m, n_r))),
            gain_all_db=t(np.zeros(n_m)),
            carrier_rev_s=None,
            amp_exp=t(prof["amp_exp"]),
            wind_db=None if wind is None else t(wind["wind_db"]),
        )
    return V2Params(
        sigma_nu=t(d["sigma_nu"]),
        lam=t(d["lam"]),
        gamma_hz=t(gamma_from_params(d)),
        profile_db=t(prof["profile_db"]),
        floor=FloorParams(
            mean_db=t(floor["floor_mean_db"]),
            shape_z=t(floor["floor_shape_z"]),
            tilt_db_oct=t(floor["floor_tilt_db_oct"]),
            mic_floor_db=t(floor["mic_floor_db"]),
            exp=t(floor["floor_exp"]),
            static_rel=t(floor["floor_static_rel"]),
        ),
        mic_line_gain_db=t(prof["mic_line_gain_db"]),
        gain_all_db=t(d["mic_gains_db"]),
        carrier_rev_s=None if d.get("carrier_rev_s") is None else t(d["carrier_rev_s"]),
        amp_exp=t(prof["amp_exp"]),
    )


def frozen_from_params(d: dict[str, Any]) -> dict[str, Any]:
    """The ``frozen`` mapping :func:`sample_params` reads, from a fit's params.

    Reads a ``/1`` payload as well, through :func:`gamma_from_params`.
    """
    return dict(
        sigma_nu=d["sigma_nu"],
        lam=d["lam"],
        gamma_hz=gamma_from_params(d).tolist(),
        profile_db=d["profile"]["profile_db"],
        amp_exp=d["profile"]["amp_exp"],
        mic_line_gain_db=d["profile"]["mic_line_gain_db"],
        floor_mean_db=d["floor"]["floor_mean_db"],
        floor_shape_z=d["floor"]["floor_shape_z"],
        floor_tilt_db_oct=d["floor"]["floor_tilt_db_oct"],
        floor_exp=d["floor"]["floor_exp"],
        floor_static_rel=d["floor"]["floor_static_rel"],
        mic_floor_db=d["floor"]["mic_floor_db"],
        gain_all_db=d["mic_gains_db"],
        carrier_rev_s=d.get("carrier_rev_s"),
        # already folded into profile_db above (the model adds it there), so a
        # reader must NEVER apply it again; carried so the shift stays
        # auditable against the bench comb the fit froze
        low_order_gain_db=d["profile"].get("low_order_gain_db"),
    )
