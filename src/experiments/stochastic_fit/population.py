"""Empirical-Bayes population fit for the stochastic rig spectrum.

The structural rig fit in :mod:`.rig` is a joint MAP estimator.  This module
keeps its shared rig parameters but integrates the clip random effects with a
reparameterized diagonal Gaussian posterior.  The renderer samples those same
priors; optimized clip values are never reinterpreted as a sampling range.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.func import functional_call

from .model import CombSpectrum, Spec
from .rig import (
    RigClip,
    RigParams,
    RigSpec,
    _init_rig_from_clips,
    _prepare,
    _scores,
    _stages,
)

_LOG_2PI = math.log(2.0 * math.pi)
_STANDARD_NORMAL_NAMES = (
    "floor_level_z",
    "floor_tilt_z",
    "h_z",
    "profile_z",
    "u_z",
)
_POINT_NUISANCE_NAMES = ("floor_mean_db",)


def _positive_raw(value: float, floor: float) -> float:
    x = max(float(value) - floor, 1e-6)
    return math.log(math.expm1(x))


def _profile_reference_db(model: CombSpectrum) -> Tensor:
    """Rig mean at order 2 (order 1 for a one-order model), invariant to clip level."""
    rig = getattr(model, "rig", None)
    if not isinstance(rig, RigParams):
        raise TypeError("population effects require ClipInRig")
    index = min(1, rig.profile_db.numel() - 1)
    reference = rig.profile_db[index]
    if rig.rig.rotor_delta:
        reference = reference + rig.delta_db[:, index].mean()
    return reference


@dataclass(frozen=True)
class PopulationFitSpec:
    """Numerical choices for the variational marginal objective."""

    mc_samples: int = 2
    iw_samples: int = 16
    init_posterior_std: float = 0.1
    min_population_std_db: float = 0.1
    min_log_scale: float = -8.0
    max_log_scale: float = 3.0


class PopulationHyperParams(nn.Module):
    """Scale-invariant populations for the line/floor ratio and rotor contrast."""

    def __init__(
        self,
        *,
        line_floor_mean_db: float,
        line_floor_std_db: float,
        rotor_contrast_std_db: float,
        min_std_db: float,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.line_floor_mean_db = nn.Parameter(
            torch.tensor(float(line_floor_mean_db), device=device, dtype=dtype)
        )
        self.line_floor_std_raw = nn.Parameter(
            torch.tensor(
                _positive_raw(line_floor_std_db, min_std_db),
                device=device,
                dtype=dtype,
            )
        )
        self.rotor_contrast_std_raw = nn.Parameter(
            torch.tensor(
                _positive_raw(rotor_contrast_std_db, min_std_db),
                device=device,
                dtype=dtype,
            )
        )
        self.min_std_db = float(min_std_db)

    @classmethod
    def from_clips(cls, rcs: list[RigClip], *, min_std_db: float) -> PopulationHyperParams:
        floors = torch.cat([rc.model.floor_mean_db.detach().reshape(-1) for rc in rcs])
        levels = torch.stack([rc.model.level_db.detach() for rc in rcs])
        line_floor = levels.mean(dim=1) + _profile_reference_db(rcs[0].model) - floors
        contrasts = levels - levels.mean(dim=1, keepdim=True)
        return cls(
            line_floor_mean_db=float(line_floor.mean().item()),
            line_floor_std_db=float(line_floor.std(unbiased=False).clamp_min(1.0).item()),
            rotor_contrast_std_db=float(contrasts.square().mean().sqrt().clamp_min(1.0).item()),
            min_std_db=min_std_db,
            device=floors.device,
            dtype=floors.dtype,
        )

    @property
    def line_floor_std_db(self) -> Tensor:
        return torch.nn.functional.softplus(self.line_floor_std_raw) + self.min_std_db

    @property
    def rotor_contrast_std_db(self) -> Tensor:
        return torch.nn.functional.softplus(self.rotor_contrast_std_raw) + self.min_std_db

    def prior(self) -> Tensor:
        """Weak proper hyperprior; random-effect normalizers identify scales."""
        return 0.5 * (
            (self.line_floor_mean_db / 50.0).square()
            + (self.line_floor_std_db / 20.0).square()
            + (self.rotor_contrast_std_db / 20.0).square()
        )

    def export(self) -> dict[str, float]:
        return {
            "line_floor_mean_db": float(self.line_floor_mean_db.detach().cpu()),
            "line_floor_std_db": float(self.line_floor_std_db.detach().cpu()),
            "rotor_contrast_std_db": float(self.rotor_contrast_std_db.detach().cpu()),
        }


class VariationalClip(nn.Module):
    """A diagonal Gaussian posterior over one clip's renderer random effects."""

    def __init__(self, model: CombSpectrum, *, init_std: float):
        super().__init__()
        names = ["level_db", *_STANDARD_NORMAL_NAMES]
        if model.spec.rps_offset:
            names.append("rps_offset_knots")
        self.loc = nn.ParameterDict()
        self.log_scale = nn.ParameterDict()
        for name in names:
            value = getattr(model, name, None)
            if not isinstance(value, Tensor) or value.numel() == 0:
                continue
            self.loc[name] = nn.Parameter(value.detach().clone())
            self.log_scale[name] = nn.Parameter(torch.full_like(value, math.log(float(init_std))))
        if "level_db" not in self.loc:
            raise ValueError("population fit requires tied profiles with clip rotor levels")
        active = {id(parameter) for parameter in model.parameter_groups("all")}
        local_active = {
            name
            for name, parameter in model.named_parameters(recurse=False)
            if id(parameter) in active
        }
        missing = local_active - set(self.loc) - set(_POINT_NUISANCE_NAMES)
        if missing:
            raise ValueError(
                "population fit has no prior for active clip parameter(s): "
                + ", ".join(sorted(missing))
            )

    def draw_eps(self, generator: torch.Generator) -> dict[str, Tensor]:
        return {
            name: torch.randn(
                value.shape,
                dtype=value.dtype,
                device=value.device,
                generator=generator,
            )
            for name, value in self.loc.items()
        }

    def rsample(self, eps: dict[str, Tensor], sign: float = 1.0) -> dict[str, Tensor]:
        return {
            name: value + sign * eps[name] * self.log_scale[name].exp()
            for name, value in self.loc.items()
        }

    def entropy(self) -> Tensor:
        out = next(iter(self.log_scale.values())).new_zeros(())
        for log_scale in self.log_scale.values():
            out = out + (log_scale + 0.5 * (1.0 + _LOG_2PI)).sum()
        return out

    def log_prob(self, sample: dict[str, Tensor]) -> Tensor:
        out = next(iter(self.loc.values())).new_zeros(())
        for name, loc in self.loc.items():
            log_scale = self.log_scale[name]
            z = (sample[name] - loc) * torch.exp(-log_scale)
            out = out - 0.5 * (z.square() + _LOG_2PI).sum() - log_scale.sum()
        return out

    def mean_replacements(self) -> dict[str, Tensor]:
        return dict(self.loc.items())

    def copy_mean_to(self, model: CombSpectrum) -> None:
        with torch.no_grad():
            for name, value in self.loc.items():
                target = getattr(model, name)
                target.copy_(value)

    def clamp_scales_(self, lo: float, hi: float) -> None:
        with torch.no_grad():
            for value in self.log_scale.values():
                value.clamp_(lo, hi)

    def export(self) -> dict[str, dict[str, np.ndarray]]:
        return {
            "loc": {name: value.detach().cpu().numpy().copy() for name, value in self.loc.items()},
            "std": {
                name: value.detach().exp().cpu().numpy().copy()
                for name, value in self.log_scale.items()
            },
        }


def random_effect_nll(
    sample: dict[str, Tensor], model: CombSpectrum, hyper: PopulationHyperParams
) -> Tensor:
    """Normalized ``-log p(eta | theta)`` for scale-invariant clip effects."""
    out = next(iter(sample.values())).new_zeros(())
    for name, value in sample.items():
        if name == "level_db":
            mean_level = value.mean()
            ratio = mean_level + _profile_reference_db(model) - model.floor_mean_db.squeeze()
            ratio_z = (ratio - hyper.line_floor_mean_db) / hyper.line_floor_std_db
            out = out + 0.5 * (ratio_z.square() + _LOG_2PI)
            out = out + torch.log(hyper.line_floor_std_db)
            if value.numel() > 1:
                contrast = value - mean_level
                contrast_z = contrast / hyper.rotor_contrast_std_db
                out = out + 0.5 * contrast_z.square().sum()
                out = out + (value.numel() - 1) * (
                    torch.log(hyper.rotor_contrast_std_db) + 0.5 * _LOG_2PI
                )
            continue
        if name == "rps_offset_knots":
            std = value.new_tensor(model.spec.rps_offset_std)
            z = value / std
        else:
            std = value.new_tensor(1.0)
            z = value
        out = out + 0.5 * (z.square() + _LOG_2PI).sum()
        out = out + value.numel() * torch.log(std)
    return out


def _sample_signs(n: int) -> tuple[float, ...]:
    if n < 1:
        raise ValueError("mc_samples must be positive")
    signs: list[float] = []
    while len(signs) < n:
        signs.extend((1.0, -1.0))
    return tuple(signs[:n])


def population_objective(
    rig: RigParams,
    rcs: list[RigClip],
    posteriors: list[VariationalClip],
    hyper: PopulationHyperParams,
    *,
    mc_samples: int,
    generator: torch.Generator,
    backward: bool,
    include_global_prior: bool = True,
) -> tuple[Tensor, float]:
    """Negative ELBO, accumulating one clip graph at a time."""
    total_value = 0.0
    detached_total = hyper.line_floor_mean_db.new_zeros(())
    signs = _sample_signs(mc_samples)
    for rc, posterior in zip(rcs, posteriors, strict=True):
        eps_by_pair: list[dict[str, Tensor]] = []
        terms: list[Tensor] = []
        for draw, sign in enumerate(signs):
            pair = draw // 2
            if pair == len(eps_by_pair):
                eps_by_pair.append(posterior.draw_eps(generator))
            sample = posterior.rsample(eps_by_pair[pair], sign)
            spectrum = functional_call(rc.model, sample, ())
            terms.append(
                rc.model.whittle(rc.power, spectrum) + random_effect_nll(sample, rc.model, hyper)
            )
        clip_loss = torch.stack(terms).mean() - posterior.entropy()
        if backward:
            clip_loss.backward()
        else:
            detached_total = detached_total + clip_loss
        total_value += float(clip_loss.detach())

    if include_global_prior:
        global_prior = rig.prior() + hyper.prior()
        if backward:
            global_prior.backward()
        else:
            detached_total = detached_total + global_prior
        total_value += float(global_prior.detach())
    return detached_total, total_value


def _unique(parameters: list[nn.Parameter]) -> list[nn.Parameter]:
    seen: set[int] = set()
    out: list[nn.Parameter] = []
    for parameter in parameters:
        if id(parameter) not in seen:
            seen.add(id(parameter))
            out.append(parameter)
    return out


def _point_parameters(rc: RigClip, posterior: VariationalClip) -> list[nn.Parameter]:
    active = {id(parameter) for parameter in rc.model.parameter_groups("all")}
    out: list[nn.Parameter] = []
    for name, parameter in rc.model.named_parameters(recurse=False):
        if name in posterior.loc:
            parameter.requires_grad_(False)
        elif name in _POINT_NUISANCE_NAMES and id(parameter) in active:
            parameter.requires_grad_(True)
            out.append(parameter)
    return out


def fit_population(
    clips: list[tuple[str, str, Any, Spec]],
    rig_spec: RigSpec,
    *,
    fit_spec: PopulationFitSpec = PopulationFitSpec(),
    device: str = "cpu",
    ladder: tuple[int, ...] = (16, 48),
    map_iters: tuple[int, int, int] = (60, 60, 120),
    vi_iters: int = 200,
    map_lr: float = 0.1,
    vi_lr: float = 0.03,
    seed: int = 0,
    log: Any = print,
) -> dict[str, Any]:
    """Fit shared rig parameters by an ELBO over renderer clip effects."""
    if not rig_spec.tie_profile:
        raise ValueError("population fit requires tie_profile=True")
    t0 = time.time()
    spec0 = clips[0][3]
    n_rotors = clips[0][2].rps.shape[0]
    rig = RigParams(spec0, rig_spec, n_rotors, device)
    rcs = _prepare(clips, rig, device)
    _init_rig_from_clips(rig, rcs)
    _stages(
        rig,
        rcs,
        rig_free=True,
        ladder=ladder,
        iters=map_iters,
        lr=map_lr,
        log=log,
        t0=t0,
    )

    hyper = PopulationHyperParams.from_clips(rcs, min_std_db=fit_spec.min_population_std_db)
    posteriors = [VariationalClip(rc.model, init_std=fit_spec.init_posterior_std) for rc in rcs]
    point_parameters = [
        parameter
        for rc, posterior in zip(rcs, posteriors, strict=True)
        for parameter in _point_parameters(rc, posterior)
    ]
    parameters = list(rig.parameters_for("all")) + list(hyper.parameters())
    parameters += point_parameters
    parameters += [parameter for posterior in posteriors for parameter in posterior.parameters()]
    opt = torch.optim.Adam(_unique(parameters), lr=vi_lr)
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(seed)
    last = float("nan")
    for step in range(vi_iters):
        opt.zero_grad(set_to_none=True)
        _, last = population_objective(
            rig,
            rcs,
            posteriors,
            hyper,
            mc_samples=fit_spec.mc_samples,
            generator=generator,
            backward=True,
        )
        opt.step()
        for posterior in posteriors:
            posterior.clamp_scales_(fit_spec.min_log_scale, fit_spec.max_log_scale)
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == vi_iters:
            log(f"  population {step + 1}/{vi_iters}: {-last:.1f} ELBO  ({time.time() - t0:.0f}s)")

    for rc, posterior in zip(rcs, posteriors, strict=True):
        posterior.copy_mean_to(rc.model)
    per_clip = {
        rc.clip_id: {
            "group": rc.group,
            "scores": _scores(rc),
            "params": rc.model.export(),
            "posterior": posterior.export(),
        }
        for rc, posterior in zip(rcs, posteriors, strict=True)
    }
    n_cells = sum(rc.model.n_cells() for rc in rcs)
    return {
        "rig_spec": asdict(rig_spec),
        "fit_spec": asdict(fit_spec),
        "spec": asdict(spec0) | {"freqs": None, "times": None, "rps": None},
        "rig": rig.export(),
        "population": hyper.export(),
        "rig_state": {name: value.detach().cpu() for name, value in rig.state_dict().items()},
        "population_state": {
            name: value.detach().cpu() for name, value in hyper.state_dict().items()
        },
        "clips": per_clip,
        "elbo": -last,
        "elbo_per_cell": -last / n_cells,
        "seconds": time.time() - t0,
    }


def importance_log_evidence(
    rc: RigClip,
    posterior: VariationalClip,
    hyper: PopulationHyperParams,
    *,
    samples: int,
    generator: torch.Generator,
) -> tuple[float, float]:
    """Importance estimate of clip log evidence and effective sample size."""
    if samples < 1:
        raise ValueError("importance samples must be positive")
    weights: list[Tensor] = []
    eps_by_pair: list[dict[str, Tensor]] = []
    with torch.no_grad():
        for draw, sign in enumerate(_sample_signs(samples)):
            pair = draw // 2
            if pair == len(eps_by_pair):
                eps_by_pair.append(posterior.draw_eps(generator))
            sample = posterior.rsample(eps_by_pair[pair], sign)
            spectrum = functional_call(rc.model, sample, ())
            log_joint = -rc.model.whittle(rc.power, spectrum) - random_effect_nll(
                sample, rc.model, hyper
            )
            weights.append(log_joint - posterior.log_prob(sample))
        w = torch.stack(weights).to(torch.float64)
        log_sum = torch.logsumexp(w, dim=0)
        log_evidence = log_sum - math.log(samples)
        log_ess = 2.0 * log_sum - torch.logsumexp(2.0 * w, dim=0)
    return float(log_evidence), float(torch.exp(log_ess).clamp_max(samples))


def _init_heldout_profiles(rig: RigParams, rcs: list[RigClip]) -> None:
    with torch.no_grad():
        base = rig.profile_db[None, :]
        if rig.rig.rotor_delta:
            base = base + rig.delta_db
        basis = rig.profile_basis()
        for rc in rcs:
            residual = rc.model.profile_db - base
            rc.model.level_db.copy_(residual.median(dim=1).values)
            if rc.model.profile_z is not None:
                shape = residual - rc.model.level_db[:, None]
                # The N(0, I) factor prior makes this a ridge-MAP initialization.
                # Plain least squares explodes when an evidence-rejected mode has
                # almost zero loading, which poisoned held-out optimization.
                gram = basis @ basis.T + torch.eye(
                    basis.shape[0], dtype=basis.dtype, device=basis.device
                )
                solution = torch.linalg.solve(gram, basis @ shape.T).T
                rc.model.profile_z.copy_(solution.clamp(-6.0, 6.0))


def fit_population_heldout(
    fitted: dict[str, Any],
    clips: list[tuple[str, str, Any, Spec]],
    *,
    device: str = "cpu",
    ladder: tuple[int, ...] = (16, 48),
    map_iters: tuple[int, int, int] = (60, 60, 120),
    vi_iters: int = 150,
    map_lr: float = 0.1,
    vi_lr: float = 0.03,
    seed: int = 1,
    log: Any = print,
) -> dict[str, Any]:
    """Freeze a population fit and marginally score unseen recordings."""
    t0 = time.time()
    rig_spec = RigSpec(**fitted["rig_spec"])
    fit_spec = PopulationFitSpec(**fitted["fit_spec"])
    spec0 = clips[0][3]
    n_rotors = clips[0][2].rps.shape[0]
    rig = RigParams(spec0, rig_spec, n_rotors, device)
    rig.load_state_dict({name: value.to(device) for name, value in fitted["rig_state"].items()})
    population = fitted["population"]
    hyper = PopulationHyperParams(
        line_floor_mean_db=population["line_floor_mean_db"],
        line_floor_std_db=population["line_floor_std_db"],
        rotor_contrast_std_db=population["rotor_contrast_std_db"],
        min_std_db=fit_spec.min_population_std_db,
        device=device,
    )
    hyper.load_state_dict(
        {name: value.to(device) for name, value in fitted["population_state"].items()}
    )
    for parameter in (*rig.parameters(), *hyper.parameters()):
        parameter.requires_grad_(False)

    rcs = _prepare(clips, rig, device)
    _init_heldout_profiles(rig, rcs)
    _stages(
        rig,
        rcs,
        rig_free=False,
        ladder=ladder,
        iters=map_iters,
        lr=map_lr,
        log=log,
        t0=t0,
    )
    posteriors = [VariationalClip(rc.model, init_std=fit_spec.init_posterior_std) for rc in rcs]
    point_parameters = [
        parameter
        for rc, posterior in zip(rcs, posteriors, strict=True)
        for parameter in _point_parameters(rc, posterior)
    ]
    opt = torch.optim.Adam(
        point_parameters
        + [parameter for posterior in posteriors for parameter in posterior.parameters()],
        lr=vi_lr,
    )
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(seed)
    last = float("nan")
    for step in range(vi_iters):
        opt.zero_grad(set_to_none=True)
        _, last = population_objective(
            rig,
            rcs,
            posteriors,
            hyper,
            mc_samples=fit_spec.mc_samples,
            generator=generator,
            backward=True,
            include_global_prior=False,
        )
        opt.step()
        for posterior in posteriors:
            posterior.clamp_scales_(fit_spec.min_log_scale, fit_spec.max_log_scale)
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == vi_iters:
            log(
                f"  held-out population {step + 1}/{vi_iters}: "
                f"{-last:.1f} ELBO  ({time.time() - t0:.0f}s)"
            )

    evidence_generator = torch.Generator(device=torch.device(device))
    evidence_generator.manual_seed(seed + 1)
    per_clip: dict[str, dict[str, Any]] = {}
    for rc, posterior in zip(rcs, posteriors, strict=True):
        log_evidence, ess = importance_log_evidence(
            rc,
            posterior,
            hyper,
            samples=fit_spec.iw_samples,
            generator=evidence_generator,
        )
        posterior.copy_mean_to(rc.model)
        per_clip[rc.clip_id] = {
            "group": rc.group,
            "scores": _scores(rc),
            "params": rc.model.export(),
            "posterior": posterior.export(),
            "iw_log_evidence": log_evidence,
            "iw_nll_per_cell": -log_evidence / rc.model.n_cells(),
            "iw_ess": ess,
        }
    return {
        "clips": per_clip,
        "elbo": -last,
        "elbo_per_cell": -last / sum(rc.model.n_cells() for rc in rcs),
        "seconds": time.time() - t0,
    }


__all__ = [
    "PopulationFitSpec",
    "PopulationHyperParams",
    "VariationalClip",
    "fit_population",
    "fit_population_heldout",
    "importance_log_evidence",
    "population_objective",
    "random_effect_nll",
]
