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
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from torch import Tensor

from experiments.stochastic_fit.model import FLOOR_SHAPE_N_CTRL
from experiments.stochastic_fit.revised_phase import composite_risk, composite_weights

from . import spectrum as SP
from .spectrum import BenchGrid, FlightGrid, FloorParams, V2Params

__all__ = [
    "BAND_SPLIT_HZ",
    "DYN_SITES",
    "PRIORS",
    "SPEED_LAW_SITES",
    "Measured",
    "Priors",
    "SupportBatch",
    "batch_slice",
    "bench_batch",
    "dynamics_pin",
    "flight_batch",
    "flight_lam",
    "forward",
    "free_blocks",
    "frozen_from_params",
    "gamma_from_params",
    "is_pinned",
    "objective_breakdown",
    "params_from_dict",
    "params_to_dict",
    "pin_applied",
    "sample_params",
    "sample_params_from_values",
    "span_pinned_sites",
    "support_model",
    "whittle_risk",
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
            "log_floor_exp": list(self.log_floor_exp),
            "log_floor_static": list(self.log_floor_static),
            "mic_line_gain_db_sd": self.mic_line_gain_db,
            "mic_floor_db_sd": self.mic_floor_db,
            "gain_all_db_sd": self.gain_all_db,
            "speed_span_pin": self.speed_span_pin,
        }


PRIORS = Priors()


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

    def profile_prior(self, priors: Priors) -> tuple[Tensor, Tensor]:
        """``(loc, scale)`` of the two-regime profile prior, per line."""
        snr = np.asarray(self.line_snr_db, dtype=np.float64)
        visible = snr >= float(priors.line_visible_snr_db)
        loc = np.where(
            visible,
            np.asarray(self.profile_db, dtype=np.float64),
            float(self.floor_mean_db) + float(priors.profile_below_offset_db),
        )
        scale = np.where(visible, float(priors.profile_db_sd), float(priors.profile_below_sd))
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
    if mode == "flight_floor_only":
        # the comb arrives FROZEN from a bench rig, so its absolute level is
        # that rig's, and nothing downstream can re-level it: render_noise
        # mean-centres mic_line_gain_db over mics per rotor and mic_gains_db
        # over mics, which leaves profile_db as the ONLY absolute comb scale.
        # One shared scalar, no more — the comb's SHAPE stays frozen.
        return ("floor", "mic", "comb_gain")
    # the ATTRIBUTION mode of the four-motor validation: a transfer gap that it
    # closes is a gap in that block alone
    if mode == "bench_dynamics_only":
        return ("dynamics",)
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
) -> SupportBatch:
    """Pool flight windows into one objective.

    ``members`` is ``[(support_name, power (M, N, F), carrier_audio (R, T),
    frame_starts (N,))]``. Frames are thinned by ``frame_stride`` — at hop 512
    and NFFT 2048 a stride of 4 selects DISJOINT windows, which is the most
    informative subsample per unit cost and removes the overlap correlation the
    composite weights otherwise carry — and the retained exposure weights are
    scaled back up by the thinning factor so the likelihood keeps the support's
    true exposure against the priors.
    """
    grid = SP.flight_grid(
        sr=sr, n_fft=n_fft, hop=hop, sr_work=sr_work, device=device, apply_transfer=apply_transfer
    )
    powers: list[np.ndarray] = []
    rates: list[Tensor] = []
    keys: list[tuple[str, int]] = []
    n_total = 0
    carriers: list[np.ndarray] = []
    for mname, power, carrier, starts in members:
        p = np.asarray(power, dtype=np.float64)
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
        ),
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
    )


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
) -> V2Params:
    """Assemble :class:`.spectrum.V2Params` from a guide's ``median()`` dict.

    The SAME construction the model samples through, with the site draw
    replaced by a lookup: a MAP parameter set and the model's own parameter set
    cannot disagree about shapes, parities or which block is frozen.
    """

    def lookup(name: str, _d: Any) -> Tensor:
        if name not in values:
            raise KeyError(f"guide has no site {name!r} (sites: {sorted(values)})")
        return torch.as_tensor(values[name], dtype=torch.float64)

    return sample_params(batch, mode=mode, priors=priors, frozen=frozen, pin=pin, site=lookup)


def sample_params(
    batch: SupportBatch,
    *,
    mode: str,
    priors: Priors = PRIORS,
    frozen: dict[str, Any] | None = None,
    pin: dict[str, Any] | None = None,
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


def forward(batch: SupportBatch, params: V2Params, **kw: Any) -> Tensor:
    """``(M, N, F)`` expected periodogram of the batch under ``params``."""
    if batch.mode == "bench":
        assert isinstance(batch.grid, BenchGrid)
        return SP.bench_model(
            batch.grid, params, k_max=batch.k_max, groups=batch.bench_order_groups, **kw
        )
    assert isinstance(batch.grid, FlightGrid) and batch.rate_work is not None
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
    temperature: float = 1.0,
    forward_kw: dict[str, Any] | None = None,
) -> V2Params:
    """The Pyro model of one support: priors, forward model, one Whittle factor."""
    if not (math.isfinite(temperature) and temperature > 0.0):
        raise ValueError(f"temperature must be finite and positive, got {temperature!r}")
    params = sample_params(batch, mode=mode, priors=priors, frozen=frozen, pin=pin)
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


def gamma_from_params(d: dict[str, Any]) -> np.ndarray:
    """``(R, K)`` widths of a ``/2`` payload, or a ``/1`` payload MAPPED.

    A ``noise-v2-fit/1`` fit carried a per-order OU instead: phase variance
    ``sigma_eps^2 k^p`` relaxing at ``lam_eps``, both by the PARITY of ``k``.
    At lags short against ``1 / lam_eps`` — the regime every fitted rate of R1
    and R2 sat in on its own window — its exponent is
    ``sigma_eps^2 k^p lam_eps |tau|``, which is R3's ``2 pi gamma_rk |tau|``
    with

        gamma_rk = sigma_eps(parity of k)^2 k^p lam_eps(parity of k) / (2 pi).

    That is the equivalence used here, so an old fit renders as the same line
    shape it was fitted with wherever its per-order term was diffusive; where
    it had saturated, the mapped Lorentzian is WIDER than the old pedestal was
    (the saturated case is the one R3 removed for having no evidence).
    """
    if "gamma_hz" in d:
        return np.atleast_2d(np.asarray(d["gamma_hz"], dtype=np.float64))
    prof = np.atleast_2d(np.asarray(d["profile"]["profile_db"], dtype=np.float64))
    k = np.arange(1, prof.shape[1] + 1, dtype=np.float64)
    even = np.remainder(k, 2.0) == 0.0
    se = np.where(even, float(d["sigma_eps_even"]), float(d["sigma_eps_odd"]))
    le = np.where(even, float(d["lam_eps_even"]), float(d["lam_eps_odd"]))
    p = float(d.get("p", 1.0))
    gamma = se**2 * k**p * le / (2.0 * math.pi)
    return np.broadcast_to(gamma[None, :], prof.shape).copy()


def params_from_dict(d: dict[str, Any], *, device: Any = "cpu") -> V2Params:
    """Inverse of :func:`params_to_dict` (what the renderer and a frozen-comb
    flight fit read), for a ``/2`` payload or a mapped ``/1`` one
    (:func:`gamma_from_params`)."""

    def t(v: Any) -> Tensor:
        return torch.as_tensor(np.asarray(v, dtype=np.float64), dtype=torch.float64, device=device)

    prof, floor = d["profile"], d["floor"]
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
    )
