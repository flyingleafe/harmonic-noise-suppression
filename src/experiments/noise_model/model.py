"""The v2 Pyro model: measured priors, one Whittle factor, one objective.

WHAT PYRO IS HERE FOR. C2/C3 put ``log sigma, log D ~ N(0, 2^2)`` on the
dynamics, which is a statement that nothing is known
(``revised_phase.py:481-487``). Every number this campaign measured — the bench
shaft scale, the telemetry rate, the per-order decoherence — is a prior here
instead (``docs/explainers/noise-model-v2-plan.qmd``, "The priors"):

    log sigma_nu ~ N(ln 0.45, 0.7^2)      shared-term bench median, 0.11-1.8
    log lam      ~ N(ln 5.5,  0.9^2)      bench median, covers telemetry 0.67-14.4
    log sigma_eps~ N(ln 0.3,  0.6^2)      per PARITY of the order
    log lam_eps  ~ N(ln 2,    1.0^2)      per PARITY of the order
    p = 1 fixed;  the per-microphone PATH term is named and NOT fitted in R1.

Positivity is carried by ``LogNormal`` under ``constraints.positive`` — no
softplus, so an ``AutoDelta`` MAP is the mode of the real posterior in the
constrained space and not of a reparameterised surrogate. The profile, the
floor and the microphone terms keep C4's weak treatment: the floor shape is the
standard-normal GP coordinate of
:meth:`revised_phase._RevisedModel.floor_shape_db` (C4's ONLY real prior on the
nuisance block, ``revised_phase.py:2429-2432``) and the levels get proper but
wide Gaussians, because a Pyro site needs a distribution and an improper flat
prior is not one. Bench carriers are per-support NUISANCE parameters with the
approved ``N(survey, 0.5 rev/s ^2)``.

THE LIKELIHOOD is the campaign's composite risk, unchanged and imported:

    pyro.factor("whittle", -sum_i w_i sum_{m, f in band} [I / M + log M] / T)

with ``w_i = 1`` on the bench (one frame, one window, nothing to weight) and
:func:`revised_phase.composite_weights` exposure weights in flight. ``T = 1``
for the R1 MAP and is RECORDED; the frozen composite temperatures of C4
(5.192017220082491 DREGON, 15.214066879865468 Michael's) rescale a POSTERIOR,
not a mode, so they do not belong in an R1 MAP and are not silently applied.

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
    "Priors",
    "SupportBatch",
    "batch_slice",
    "bench_batch",
    "dynamics_pin",
    "flight_batch",
    "forward",
    "free_blocks",
    "frozen_from_params",
    "objective_breakdown",
    "params_from_dict",
    "params_to_dict",
    "pin_applied",
    "pin_free_mask",
    "sample_params",
    "sample_params_from_values",
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

#: The four dynamics SITES. ``sigma_eps`` and ``lam_eps`` are one ``(2,)``
#: site each — ``(even, odd)``, selected by :func:`.lag.parity_select` — not
#: two scalars, which is why a pin of one parity is a PARTIAL pin of a site.
DYN_SITES: dict[str, int] = dict(sigma_nu=1, lam=1, sigma_eps=2, lam_eps=2)


@dataclass(frozen=True)
class Priors:
    """The approved priors. ``(mean, sd)`` pairs; log-parameters in log space."""

    log_sigma_nu: tuple[float, float] = (math.log(0.45), 0.7)
    log_lam: tuple[float, float] = (math.log(5.5), 0.9)
    log_sigma_eps: tuple[float, float] = (math.log(0.3), 0.6)
    log_lam_eps: tuple[float, float] = (math.log(2.0), 1.0)
    #: weak, proper stand-ins for C4's improper flat nuisance block
    profile_db: tuple[float, float] = (-45.0, 40.0)
    floor_mean_db: tuple[float, float] = (-70.0, 40.0)
    floor_tilt_db_oct: tuple[float, float] = (0.0, 10.0)
    amp_exp: tuple[float, float] = (2.0, 2.0)
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
    #: rotor floor cannot get louder as the rotors slow, and on a pool that
    #: barely varies in speed a Normal prior let it go to -10.19, which put the
    #: standby floor +37.0 dB over the real clip
    #: (``results/noise_v2/rounds/round2/render_regime/findings.md``). The rest
    #: of the project already carries this invariant as a hard clamp on an
    #: export (``rig_sampler.check_sample``, ``_rebase_divergent_floor``); here
    #: it is the site's own support instead of a repair after the fact.
    log_floor_exp: tuple[float, float] = (math.log(2.0), 0.7)
    log_floor_static: tuple[float, float] = (math.log(2.5e-3), 2.0)
    mic_line_gain_db: float = 6.0
    mic_floor_db: float = 6.0
    gain_all_db: float = 6.0
    #: the approved bench carrier nuisance prior, in rev/s
    carrier_sd_rev_s: float = 0.5
    #: ``p`` is fixed, not sampled (approved decision 1 of 2026-09-17)
    p: float = SP.P_ORDER_EXPONENT

    def as_dict(self) -> dict[str, Any]:
        return {
            "log_sigma_nu": list(self.log_sigma_nu),
            "log_lam": list(self.log_lam),
            "log_sigma_eps_per_parity": list(self.log_sigma_eps),
            "log_lam_eps_per_parity": list(self.log_lam_eps),
            "profile_db": list(self.profile_db),
            "floor_mean_db": list(self.floor_mean_db),
            "floor_tilt_db_oct": list(self.floor_tilt_db_oct),
            "amp_exp": list(self.amp_exp),
            "comb_gain_db": list(self.comb_gain_db),
            "log_floor_exp": list(self.log_floor_exp),
            "log_floor_static": list(self.log_floor_static),
            "mic_line_gain_db_sd": self.mic_line_gain_db,
            "mic_floor_db_sd": self.mic_floor_db,
            "gain_all_db_sd": self.gain_all_db,
            "carrier_sd_rev_s": self.carrier_sd_rev_s,
            "p_fixed": self.p,
            "path_term": "named, not fitted in R1",
        }


PRIORS = Priors()


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
    members: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return int(self.power.shape[0]) * int(self.power.shape[1]) * int(self.band.sum())


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
    # is built.  It uses the approved shaft-prior centre and a 40-nat tail;
    # r_tau itself retains every sampled dynamics parameter in the graph.
    # This avoids a parameter-dependent Python/NumPy truncation in the
    # likelihood while retaining the whole-window result for low orders.
    lag_sigma = math.exp(PRIORS.log_sigma_nu[0])
    lag_lam = math.exp(PRIORS.log_lam[0])
    groups = SP.order_groups(k_max, sigma_nu=lag_sigma, lam=lag_lam, sr=sr, n=n)
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
                source="approved dynamics-prior centre (fixed before MAP)",
                sigma_nu=lag_sigma,
                lam=lag_lam,
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
        diagnostics=dict(
            grid=dict(grid.diagnostics),
            n_frames_total=n_total,
            n_frames_used=n_sel,
            frame_stride=int(frame_stride),
            exposure_scale=scale,
            carrier_min_rev_s=float(c_all.min()),
            carrier_max_rev_s=float(c_all.max()),
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


# ── pinning individual dynamics coordinates ─────────────────────────────────


def dynamics_pin(spec: dict[str, Any] | None) -> dict[str, Any] | None:
    """``{"lam": 200.0, "lam_eps_odd": 75.0}`` in the SITE spelling the model
    reads: ``{"lam": 200.0, "lam_eps": [None, 75.0]}``.

    Accepted keys are the six parameter names the fit JSON quotes
    (``sigma_nu``, ``lam``, ``sigma_eps_even/odd``, ``lam_eps_even/odd``) and
    the bare two-vector site names, which pin BOTH parities. ``None`` is the
    free marker, so a half-pinned site round-trips through JSON.
    """
    if not spec:
        return None
    out: dict[str, Any] = {}
    for key, value in spec.items():
        if key in DYN_SITES:
            out[key] = value
            continue
        site, _, parity = str(key).rpartition("_")
        if site not in DYN_SITES or DYN_SITES[site] != 2 or parity not in ("even", "odd"):
            raise ValueError(
                f"cannot pin {key!r}; pin one of {sorted(DYN_SITES)} or "
                "<site>_even / <site>_odd for the two-vector sites"
            )
        entries = list(out.get(site, [None, None]))
        entries[0 if parity == "even" else 1] = value
        out[site] = entries
    return out


def _pin_entries(value: Any, n: int) -> list[Any]:
    """``value`` as ``n`` per-coordinate entries, ``None`` where still free."""
    if value is None:
        return [None] * n
    entries = [value] * n if np.ndim(value) == 0 else list(value)
    if len(entries) != n:
        raise ValueError(f"a pin of a {n}-coordinate site needs {n} entries, got {value!r}")
    return entries


def pin_free_mask(pin: dict[str, Any] | None, name: str) -> np.ndarray:
    """Which coordinates of dynamics site ``name`` are still SAMPLED."""
    n = DYN_SITES[name]
    if pin is None or name not in pin:
        return np.ones(n, dtype=bool)
    return np.array([v is None for v in _pin_entries(pin[name], n)], dtype=bool)


def pin_applied(pin: dict[str, Any] | None, name: str, centre: float) -> Tensor:
    """A CONCRETE value of site ``name``: the pin where pinned, ``centre``
    elsewhere. This is what a probe forward pass must use, so the seeds it
    calibrates are measured at the dynamics the fit will actually run at."""
    n = DYN_SITES[name]
    entries = _pin_entries(None if pin is None else pin.get(name), n)
    vals = [float(centre) if v is None else float(v) for v in entries]
    out = torch.as_tensor(vals, dtype=torch.float64)
    return out.reshape(()) if n == 1 else out


def _dyn_site(
    site: SiteFn, name: str, prior: tuple[float, float], *, pin: dict[str, Any] | None
) -> Tensor:
    """Dynamics site ``name``, minus whatever ``pin`` holds fixed.

    A partially pinned two-vector site samples ONE site of exactly its free
    width, so ``AutoDelta`` allocates no parameter for a pinned coordinate and
    the log-prior counts no pinned coordinate either — a pin is a constant in
    the model, not a tightly-prior'd parameter.
    """
    n = DYN_SITES[name]
    mask = pin_free_mask(pin, name)
    if mask.all():
        return _lognormal(site, name, prior, () if n == 1 else (n,))
    entries = _pin_entries(pin[name] if pin else None, n)
    if not mask.any():
        out = torch.as_tensor([float(v) for v in entries], dtype=torch.float64)
        return out.reshape(()) if n == 1 else out
    drawn = _lognormal(site, name, prior, (int(mask.sum()),))
    parts: list[Tensor] = []
    j = 0
    for v in entries:
        if v is None:
            parts.append(drawn[j])
            j += 1
        else:
            parts.append(torch.as_tensor(float(v), dtype=torch.float64))
    return torch.stack(parts)


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
    guide parameter for it: ``--floor-only`` really holds the comb's SHAPE and
    dynamics fixed rather than fitting them under a tight prior — its only comb
    freedom is the single ``comb_gain_db`` scalar of the ``"comb_gain"`` block,
    which re-levels the transplanted comb as a whole. ``pin`` does the same for
    INDIVIDUAL dynamics coordinates of an otherwise free dynamics block
    (:func:`dynamics_pin`, :func:`_dyn_site`) — the identified-ridge
    reparameterisation R1 uses when a rate is not identifiable from the data.
    """
    free = free_blocks(mode)
    fz = dict(frozen or {})
    r, m, k = batch.n_rotors, batch.n_mics, batch.k_max
    flight = batch.mode == "flight"

    def take(block: str, key: str, shape: tuple[int, ...] | None = None) -> Tensor:
        v = torch.as_tensor(np.asarray(fz[key], dtype=np.float64), dtype=torch.float64)
        if shape is not None and tuple(v.shape) != shape:
            if v.numel() == 1:
                v = v.reshape(()).expand(shape).clone()
            elif key == "profile_db" and v.ndim == 2 and int(v.shape[1]) >= shape[1]:
                v = v[: shape[0], : shape[1]].clone()
            else:
                raise ValueError(f"frozen {key} has shape {tuple(v.shape)}, need {shape}")
        return v.to(device=batch.power.device)

    if "dynamics" in free:
        sigma_nu = _dyn_site(site, "sigma_nu", priors.log_sigma_nu, pin=pin)
        lam = _dyn_site(site, "lam", priors.log_lam, pin=pin)
        sigma_eps = _dyn_site(site, "sigma_eps", priors.log_sigma_eps, pin=pin)
        lam_eps = _dyn_site(site, "lam_eps", priors.log_lam_eps, pin=pin)
    else:
        sigma_nu = take("dynamics", "sigma_nu")
        lam = take("dynamics", "lam")
        sigma_eps = take("dynamics", "sigma_eps", (2,))
        lam_eps = take("dynamics", "lam_eps", (2,))

    zero = torch.zeros((), dtype=torch.float64)
    if "profile" in free:
        profile_db = _normal(site, "profile_db", priors.profile_db[0], priors.profile_db[1], (r, k))
        amp_exp = _normal(site, "amp_exp", *priors.amp_exp) if flight else zero
    else:
        profile_db = take("profile", "profile_db", (r, k))
        amp_exp = take("profile", "amp_exp") if flight else zero

    if "comb_gain" in free:
        # ONE shared scalar, folded into profile_db right here, so write_fit,
        # params_to_dict, render_noise and expected_periodogram need no change
        # at all: the comb they read is already at its fitted level
        profile_db = profile_db + _normal(site, "comb_gain_db", *priors.comb_gain_db)

    if "floor" in free:
        floor = FloorParams(
            mean_db=_normal(site, "floor_mean_db", *priors.floor_mean_db),
            shape_z=_normal(site, "floor_shape_z", 0.0, 1.0, (FLOOR_SHAPE_N_CTRL,)),
            tilt_db_oct=_normal(site, "floor_tilt_db_oct", *priors.floor_tilt_db_oct),
            mic_floor_db=_normal(site, "mic_floor_db", 0.0, priors.mic_floor_db, (m,)),
            exp=_lognormal(site, "floor_exp", priors.log_floor_exp) if flight else zero,
            static_rel=(
                _lognormal(site, "floor_static_rel", priors.log_floor_static) if flight else zero
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
        sigma_eps=sigma_eps,
        lam_eps=lam_eps,
        profile_db=profile_db,
        floor=floor,
        mic_line_gain_db=mic_line,
        gain_all_db=gain_all,
        carrier_rev_s=carrier,
        amp_exp=amp_exp,
        p=priors.p,
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
    """The ``params`` block of the ``noise-v2-fit/1`` JSON."""
    f = lambda v: float(np.asarray(v.detach().cpu() if isinstance(v, Tensor) else v))  # noqa: E731
    a = lambda v: np.asarray(  # noqa: E731
        v.detach().cpu() if isinstance(v, Tensor) else v, dtype=np.float64
    ).tolist()
    se = np.atleast_1d(
        np.asarray(
            params.sigma_eps.detach().cpu()
            if isinstance(params.sigma_eps, Tensor)
            else params.sigma_eps,
            dtype=np.float64,
        )
    )
    le = np.atleast_1d(
        np.asarray(
            params.lam_eps.detach().cpu() if isinstance(params.lam_eps, Tensor) else params.lam_eps,
            dtype=np.float64,
        )
    )
    se = np.broadcast_to(se, (2,)) if se.size == 1 else se
    le = np.broadcast_to(le, (2,)) if le.size == 1 else le
    return dict(
        sigma_nu=f(params.sigma_nu),
        lam=f(params.lam),
        sigma_eps_even=float(se[0]),
        sigma_eps_odd=float(se[1]),
        lam_eps_even=float(le[0]),
        lam_eps_odd=float(le[1]),
        p=float(params.p),
        carrier_rev_s=(a(params.carrier_rev_s) if params.carrier_rev_s is not None else None),
        profile=dict(
            profile_db=a(params.profile_db),
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


def params_from_dict(d: dict[str, Any], *, device: Any = "cpu") -> V2Params:
    """Inverse of :func:`params_to_dict` (what the renderer and a frozen-comb
    flight fit read)."""

    def t(v: Any) -> Tensor:
        return torch.as_tensor(np.asarray(v, dtype=np.float64), dtype=torch.float64, device=device)

    prof, floor = d["profile"], d["floor"]
    return V2Params(
        sigma_nu=t(d["sigma_nu"]),
        lam=t(d["lam"]),
        sigma_eps=t([d["sigma_eps_even"], d["sigma_eps_odd"]]),
        lam_eps=t([d["lam_eps_even"], d["lam_eps_odd"]]),
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
        p=float(d.get("p", SP.P_ORDER_EXPONENT)),
    )


def frozen_from_params(d: dict[str, Any]) -> dict[str, Any]:
    """The ``frozen`` mapping :func:`sample_params` reads, from a fit's params."""
    return dict(
        sigma_nu=d["sigma_nu"],
        lam=d["lam"],
        sigma_eps=[d["sigma_eps_even"], d["sigma_eps_odd"]],
        lam_eps=[d["lam_eps_even"], d["lam_eps_odd"]],
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
