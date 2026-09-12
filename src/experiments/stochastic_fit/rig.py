"""One model per rig: rig-level parameters tied across clips, per-rotor
deviations, per-clip nuisances — Bretthorst's joint analysis of multiple
measurements (§ 7.5) with the Whittle likelihood.

Hierarchy (``docs/hierarchical-rig-model-plan.md`` § 3):

- rig: floor shape and tilt, the harmonic profile ``A_k``, the width law
  (``gamma0``, ``slope``), the microphone pattern (line gains, floor gains,
  gain on everything), the speed law;
- rotor: profile deviation ``delta_rk ~ N(0, delta_std^2)``, width scale ``w_r``;
- clip: floor level and its drifts, per-rotor level, line drifts, carrier
  correction, the per-mic low-band modulation.

``RigSpec`` flags switch each tie on; with every flag off ``fit_rig`` is the
independent fit of every clip in one process (M0), so the ladder M1..M6 is
one code path. Held-out scoring freezes the rig and rotor parameters and
refits only the clip nuisances (``fit_heldout``).
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .data import Periodogram
from .fit import _inv_softplus, initialize, loo_offsets_for, loo_reference
from .model import FLOOR_SHAPE_N_CTRL, CombSpectrum, Spec


@dataclass
class RigSpec:
    """Which parameters are tied at rig level, and the rotor-level priors."""

    tie_floor_shape: bool = True
    tie_tilt: bool = True
    tie_profile: bool = True
    tie_width: bool = True
    tie_mic_gain: bool = True  # the (mic, rotor) line-gain pattern
    tie_mic_floor: bool = True  # per-mic floor gains (only if spec.mic_floor)
    tie_speed_law: bool = True  # only if spec.fit_speed_law
    rotor_delta: bool = False  # per-rotor profile deviation delta_rk
    delta_std_db: float = 2.5  # its prior std (bench: ~5 dB^2 -> 2.2 dB)
    rotor_width: bool = False  # per-rotor width scale w_r, prior std 0.2 in log
    width_scale_std: float = 0.2
    level_prior_db: float = 10.0  # per-clip per-rotor level, weak prior
    profile_rank: int = 0  # low-rank correlated clip/rotor profile population
    profile_mode_smooth_db: float = 3.0  # second-difference prior on each loading
    profile_mode_scale_prior_db: float = 5.0
    extra: dict[str, Any] = field(default_factory=dict)


class RigParams(nn.Module):
    """Rig- and rotor-level parameters shared by every clip of a rig."""

    def __init__(self, spec: Spec, rig: RigSpec, n_rotors: int, device, dtype=torch.float32):
        super().__init__()
        R, K, M = n_rotors, spec.n_harm, spec.n_mics
        z = lambda *s: nn.Parameter(torch.zeros(*s, dtype=dtype, device=device))  # noqa: E731
        self.floor_shape_z = z(FLOOR_SHAPE_N_CTRL)
        self.floor_tilt_db_oct = z(1)
        self.profile_db = z(K)  # A_k
        self.delta_db = z(R, K)  # delta_rk
        self.profile_basis_db: nn.Parameter | None
        self.profile_mode_std_raw: nn.Parameter | None
        if rig.profile_rank > 0:
            self.profile_basis_db = z(rig.profile_rank, K)
            self.profile_mode_std_raw = nn.Parameter(
                torch.full(
                    (rig.profile_rank,),
                    _inv_softplus(2.0),
                    dtype=dtype,
                    device=device,
                )
            )
        else:
            self.register_parameter("profile_basis_db", None)
            self.register_parameter("profile_mode_std_raw", None)
        self.gamma0_raw = nn.Parameter(
            torch.full((1,), _inv_softplus(2.0), dtype=dtype, device=device)
        )
        self.slope_raw = nn.Parameter(
            torch.full((1,), _inv_softplus(0.3), dtype=dtype, device=device)
        )
        self.log_width_scale = z(R)  # w_r
        self.log_width_power = z(1)  # log ratio to spec.width_power, tied with the law
        self.mic_gain_db = z(M, R)
        self.mic_floor_db = z(M)
        self.gain_all_db = z(M)
        self.amp_exp = nn.Parameter(
            torch.full((1,), spec.amp_rps_exponent, dtype=dtype, device=device)
        )
        self.floor_exp = nn.Parameter(
            torch.full((1,), spec.amp_rps_exponent, dtype=dtype, device=device)
        )
        self.floor_static_raw = nn.Parameter(torch.full((1,), -6.0, dtype=dtype, device=device))
        self.spec, self.rig, self.R = spec, rig, R

    def prior(self) -> Tensor:
        r = self.rig
        p = torch.zeros((), device=self.floor_shape_z.device)
        if r.tie_floor_shape:
            p = p + 0.5 * self.floor_shape_z.square().sum()
        if r.rotor_delta:
            p = p + 0.5 * self.delta_db.square().sum() / r.delta_std_db**2
        if r.rotor_width:
            p = p + 0.5 * self.log_width_scale.square().sum() / r.width_scale_std**2
        if self.spec.fit_width_power and r.tie_width:
            p = p + 0.5 * self.log_width_power.square().sum() / self.spec.width_power_log_std**2
        if self.profile_basis_db is not None:
            basis = self.profile_basis()
            if basis.shape[1] > 2:
                d2 = basis[:, 2:] - 2.0 * basis[:, 1:-1] + basis[:, :-2]
                p = p + 0.5 * d2.square().sum() / r.profile_mode_smooth_db**2
            assert self.profile_mode_std_raw is not None
            mode_std = torch.nn.functional.softplus(self.profile_mode_std_raw)
            p = p + 0.5 * (mode_std / r.profile_mode_scale_prior_db).square().sum()
        return p

    def profile_basis(self) -> Tensor:
        """Centred unit-RMS directions times explicit mode standard deviations."""
        if self.profile_basis_db is None:
            return self.profile_db.new_zeros((0, self.profile_db.numel()))
        assert self.profile_mode_std_raw is not None
        centred = self.profile_basis_db - self.profile_basis_db.mean(dim=1, keepdim=True)
        rms = centred.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        scale = torch.nn.functional.softplus(self.profile_mode_std_raw)[:, None]
        return centred / rms * scale

    def parameters_for(self, stage: str) -> list[nn.Parameter]:
        s, r = self.spec, self.rig
        floor: list[nn.Parameter] = []
        if r.tie_floor_shape:
            floor.append(self.floor_shape_z)
        if r.tie_tilt:
            floor.append(self.floor_tilt_db_oct)
        if s.mic_floor and r.tie_mic_floor:
            floor.append(self.mic_floor_db)
        if s.gain_all:
            floor.append(self.gain_all_db)
        if s.fit_speed_law and r.tie_speed_law:
            floor += [self.floor_exp, self.floor_static_raw]
        if stage == "floor":
            return floor
        lines: list[nn.Parameter] = []
        if r.tie_profile:
            lines.append(self.profile_db)
        if r.rotor_delta:
            lines.append(self.delta_db)
        if self.profile_basis_db is not None:
            assert self.profile_mode_std_raw is not None
            lines.extend((self.profile_basis_db, self.profile_mode_std_raw))
        if r.tie_width:
            lines += [self.gamma0_raw, self.slope_raw]
            if s.fit_width_power:
                lines.append(self.log_width_power)
        if r.rotor_width:
            lines.append(self.log_width_scale)
        if r.tie_mic_gain:
            lines.append(self.mic_gain_db)
        if s.fit_speed_law and r.tie_speed_law:
            lines.append(self.amp_exp)
        return floor + lines

    def export(self) -> dict[str, Any]:
        g = torch.nn.functional.softplus
        with torch.no_grad():
            return dict(
                # whitened knots -> dB curve needs the shape kernel's Cholesky,
                # which lives on the clip models; export the knots and let
                # ``ClipInRig.export`` (through ``_floor_shape_z``) give the curve
                floor_shape_z=self.floor_shape_z.cpu().numpy().copy(),
                floor_tilt_db_oct=float(self.floor_tilt_db_oct.item()),
                profile_db=self.profile_db.cpu().numpy().copy(),
                profile_basis_db=(
                    self.profile_basis().cpu().numpy().copy()
                    if self.profile_basis_db is not None
                    else None
                ),
                profile_mode_std_db=(
                    torch.nn.functional.softplus(self.profile_mode_std_raw).cpu().numpy().copy()
                    if self.profile_mode_std_raw is not None
                    else None
                ),
                delta_db=self.delta_db.cpu().numpy().copy(),
                gamma0=float(g(self.gamma0_raw).item()),
                gamma_slope=float(g(self.slope_raw).item()),
                width_power=float(
                    self.spec.width_power
                    * (
                        torch.exp(self.log_width_power[0]).item()
                        if self.spec.fit_width_power
                        else 1.0
                    )
                ),
                width_scale=torch.exp(self.log_width_scale).cpu().numpy().copy(),
                mic_gain_db=(self.mic_gain_db - self.mic_gain_db.mean(dim=0, keepdim=True))
                .cpu()
                .numpy()
                .copy(),
                mic_floor_db=self.mic_floor_db.cpu().numpy().copy(),
                gain_all_db=(self.gain_all_db - self.gain_all_db.mean()).cpu().numpy().copy(),
                amp_exp=float(self.amp_exp.item()),
                floor_exp=float(self.floor_exp.item()),
                floor_static_rel=float(g(self.floor_static_raw).item()),
            )


class ClipInRig(CombSpectrum):
    """A clip's spectrum whose tied parameters come from ``RigParams``."""

    def __init__(self, spec: Spec, rig: RigParams, device, dtype=torch.float32):
        super().__init__(spec, device=device, dtype=dtype)
        self.rig = rig
        self.level_db = nn.Parameter(torch.zeros(self.R, dtype=dtype, device=device))  # l_cr
        self.profile_z: nn.Parameter | None
        if rig.profile_basis_db is not None:
            self.profile_z = nn.Parameter(
                torch.zeros(self.R, rig.rig.profile_rank, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("profile_z", None)

    # the rig's floor-shape prior is paid once, by the rig, when tied
    def _floor_shape_z(self) -> Tensor:
        return self.rig.floor_shape_z if self.rig.rig.tie_floor_shape else self.floor_shape_z

    def _floor_tilt_db_oct(self) -> Tensor:
        return self.rig.floor_tilt_db_oct if self.rig.rig.tie_tilt else self.floor_tilt_db_oct

    def _profile_db(self) -> Tensor:
        r = self.rig.rig
        if not r.tie_profile:
            return self.profile_db
        prof = self.rig.profile_db[None, :] + self.level_db[:, None]
        if r.rotor_delta:
            prof = prof + self.rig.delta_db
        if self.profile_z is not None:
            prof = prof + self.profile_z @ self.rig.profile_basis()
        return prof

    def _gamma_raw(self) -> tuple[Tensor, Tensor]:
        r = self.rig.rig
        if not r.tie_width:
            return self.gamma0_raw, self.slope_raw
        scale = (
            torch.exp(self.rig.log_width_scale)
            if r.rotor_width
            else torch.ones(self.R, device=self._dev)
        )
        # softplus is applied by the caller; scale the widths through the raw values' softplus inverse
        g0 = torch.nn.functional.softplus(self.rig.gamma0_raw) * scale
        sl = torch.nn.functional.softplus(self.rig.slope_raw) * scale
        return _inv_softplus_t(g0), _inv_softplus_t(sl)

    def _width_power(self) -> Tensor:
        if not (self.rig.rig.tie_width and self.spec.fit_width_power):
            return super()._width_power()
        p = torch.as_tensor(self.spec.width_power, dtype=self._dtype, device=self._dev)
        return p * torch.exp(self.rig.log_width_power[0])

    def _mic_gain_db(self) -> Tensor:
        return self.rig.mic_gain_db if self.rig.rig.tie_mic_gain else self.mic_gain_db

    def _mic_floor_db(self) -> Tensor:
        return self.rig.mic_floor_db if self.rig.rig.tie_mic_floor else self.mic_floor_db

    def _gain_all_db(self) -> Tensor | None:
        return self.rig.gain_all_db if self.spec.gain_all else None

    def _amp_exp(self) -> Tensor:
        return self.rig.amp_exp if self.rig.rig.tie_speed_law else self.amp_exp

    def _floor_exp(self) -> tuple[Tensor, Tensor]:
        if self.rig.rig.tie_speed_law:
            return self.rig.floor_exp, self.rig.floor_static_raw
        return self.floor_exp, self.floor_static_raw

    def prior(self) -> Tensor:
        p = super().prior()
        r = self.rig.rig
        if r.tie_floor_shape:  # paid by the rig
            p = p - 0.5 * self.floor_shape_z.square().sum()
        if r.tie_profile:
            p = p + 0.5 * self.level_db.square().sum() / r.level_prior_db**2
        if self.profile_z is not None:
            p = p + 0.5 * self.profile_z.square().sum()
        return p

    def parameter_groups(self, stage: str) -> list[nn.Parameter]:
        """Only the clip's own (untied) parameters."""
        s, r = self.spec, self.rig.rig
        floor = [self.floor_mean_db, self.floor_level_z, self.floor_tilt_z]
        if not r.tie_floor_shape:
            floor.append(self.floor_shape_z)
        if not r.tie_tilt:
            floor.append(self.floor_tilt_db_oct)
        if s.mic_floor and not r.tie_mic_floor:
            floor.append(self.mic_floor_db)
        if s.fit_speed_law and not r.tie_speed_law:
            floor += [self.floor_exp, self.floor_static_raw]
        if s.umod_std_db > 0:
            floor.append(self.u_z)
        if stage == "floor":
            return floor
        lines = [self.h_z]
        lines.append(self.level_db if r.tie_profile else self.profile_db)
        if self.profile_z is not None:
            lines.append(self.profile_z)
        if not r.tie_width:
            lines += [self.log_gamma_free] if s.free_gamma else [self.gamma0_raw, self.slope_raw]
        if not r.tie_mic_gain:
            lines.append(self.mic_gain_db)
        if s.fit_speed_law and not r.tie_speed_law:
            lines.append(self.amp_exp)
        if s.fit_coherence:
            # a clip property: a rotor's lines may decohere at a different
            # order at a different speed
            lines.append(self.log_k_half)
        if s.rps_offset:
            lines.append(self.rps_offset_knots)
        return floor + lines


def _inv_softplus_t(x: Tensor) -> Tensor:
    return x + torch.log(-torch.expm1(-x))


# ── fitting ───────────────────────────────────────────────────────────────


@dataclass
class RigClip:
    clip_id: str
    group: str
    pg: Periodogram
    model: ClipInRig
    power: Tensor
    scale: float


def _prepare(
    clips: list[tuple[str, str, Periodogram, Spec]], rig: RigParams, device
) -> list[RigClip]:
    out = []
    for clip_id, group, pg, spec in clips:
        scale = float(np.mean(pg.power))
        power = torch.as_tensor(pg.power.astype(np.float32) / scale, device=device)
        model = ClipInRig(spec, rig, device=device)
        initialize(model, power)
        out.append(RigClip(clip_id, group, pg, model, power, scale))
    return out


def _init_rig_from_clips(rig: RigParams, rcs: list[RigClip]) -> None:
    """Rig values at the medians of the clips' crude initializations; per-clip
    levels carry the rest; deviations start at zero."""
    with torch.no_grad():
        prof = torch.stack([rc.model.profile_db for rc in rcs])  # (C, R, K)
        a = prof.median(dim=0).values.median(dim=0).values  # (K,)
        rig.profile_db.copy_(a)
        for rc in rcs:
            rc.model.level_db.copy_((rc.model.profile_db - a[None, :]).median(dim=1).values)
        if rig.profile_basis_db is not None:
            levels = torch.stack([rc.model.level_db for rc in rcs])
            residual = prof - a[None, None, :] - levels[:, :, None]
            flat = residual.reshape(-1, residual.shape[-1])
            flat = flat - flat.mean(dim=1, keepdim=True)
            u, singular, vh = torch.linalg.svd(flat, full_matrices=False)
            q = min(rig.rig.profile_rank, u.shape[1])
            sample_scale = float(max(flat.shape[0] - 1, 1)) ** 0.5
            desired = singular[:q, None] * vh[:q] / sample_scale
            rig.profile_basis_db.zero_()
            rig.profile_basis_db[:q].copy_(desired)
            assert rig.profile_mode_std_raw is not None
            desired_std = desired.square().mean(dim=1).sqrt().clamp_min(1e-3)
            rig.profile_mode_std_raw[:q].copy_(_inv_softplus_t(desired_std))
            scores = u[:, :q] * sample_scale
            for c, rc in enumerate(rcs):
                assert rc.model.profile_z is not None
                rc.model.profile_z.zero_()
                rc.model.profile_z[:, :q].copy_(scores[c * rig.R : (c + 1) * rig.R])
        rig.floor_tilt_db_oct.copy_(
            torch.stack([rc.model.floor_tilt_db_oct for rc in rcs]).median(dim=0).values
        )
        rig.floor_shape_z.zero_()
        rig.delta_db.zero_()
        rig.log_width_scale.zero_()
        rig.mic_gain_db.zero_()
        rig.mic_floor_db.zero_()
        rig.gain_all_db.zero_()


def _objective(rig: RigParams, rcs: list[RigClip]) -> tuple[Tensor, float]:
    """Sum over clips of Whittle + clip prior, plus the rig prior. Backward is
    done per clip so only one clip's graph is alive at a time."""
    total = 0.0
    for rc in rcs:
        loss = rc.model.whittle(rc.power) + rc.model.prior()
        loss.backward()
        total += float(loss.item())
    rp = rig.prior()
    if rp.requires_grad:
        rp.backward()
    return rp, total + float(rp.item())


def _adam(
    rig: RigParams,
    rcs: list[RigClip],
    params: list[nn.Parameter],
    iters: int,
    lr: float,
    retries: int = 3,
) -> float:
    if not params:
        return float("nan")
    initial = [parameter.detach().clone() for parameter in params]

    def recover(reason: str) -> float:
        with torch.no_grad():
            for parameter, value in zip(params, initial, strict=True):
                parameter.copy_(value)
        if retries == 0:
            raise FloatingPointError(
                f"rig optimizer {reason} remained non-finite at learning rate {lr:g}"
            )
        next_lr = lr / 3.0
        warnings.warn(
            f"rig optimizer {reason} became non-finite at lr={lr:g}; "
            f"retrying the stage from its initial state at lr={next_lr:g}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _adam(
            rig,
            rcs,
            params,
            iters,
            next_lr,
            retries=retries - 1,
        )

    opt = torch.optim.Adam(params, lr=lr)
    last = float("nan")
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        _, last = _objective(rig, rcs)
        if not math.isfinite(last):
            return recover("loss")
        if any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for parameter in params
        ):
            return recover("gradient")
        opt.step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in params):
            return recover("parameter")
    return last


def _scores(rc: RigClip) -> dict[str, float]:
    model = rc.model
    band_np = model.band.cpu().numpy()
    n_cells = model.n_cells()
    with torch.no_grad():
        spectrum = model.forward()
        # the model's OWN likelihood, so a coherence variant is scored by the
        # Rice-power density it actually assumes; ``nll_fit_inner`` below stays
        # on the exponential form because its LOO reference is exponential too
        nll_fit = float(model.whittle(rc.power).item()) / n_cells
        offsets = loo_offsets_for(model.N)
        half = max(abs(s) for s in offsets)
        inner = slice(half, model.N - half)
        cell = (rc.power / spectrum.clamp_min(1e-12) + torch.log(spectrum.clamp_min(1e-12)))[
            :, inner
        ][..., model.band]
        nll_inner = float(cell.mean().item())
        # per order band (k<=8 excluded from ladder verdicts: interference)
        prior = float(model.prior().item())
    nll_loo, _ = loo_reference(rc.power.cpu().numpy(), band_np, offsets=offsets)
    return dict(
        nll_fit=nll_fit,
        nll_fit_inner=nll_inner,
        nll_loo=nll_loo,
        excess_over_loo=nll_inner - nll_loo,
        prior=prior,
        n_cells=n_cells,
        power_scale=rc.scale,
    )


def _stages(
    rig: RigParams,
    rcs: list[RigClip],
    *,
    rig_free: bool,
    ladder: tuple[int, ...],
    iters: tuple[int, int, int],
    lr: float,
    log: Any,
    t0: float,
) -> None:
    K = rcs[0].model.K
    for rc in rcs:
        rc.model.active_k = 0
    params = [p for rc in rcs for p in rc.model.parameter_groups("floor")]
    if rig_free:
        params += rig.parameters_for("floor")
    loss = _adam(rig, rcs, params, iters[0], lr)
    log(f"  floor-only: {loss:.1f}  ({time.time() - t0:.0f}s)")
    params = [p for rc in rcs for p in rc.model.parameter_groups("all")]
    if rig_free:
        params += rig.parameters_for("all")
    for rung in sorted({*(min(r, K) for r in ladder), K}):
        for rc in rcs:
            rc.model.active_k = int(min(rung, K))
        loss = _adam(rig, rcs, params, iters[1] if rung < K else iters[2], lr)
        log(f"  k<={min(rung, K)}: {loss:.1f}  ({time.time() - t0:.0f}s)")


def fit_rig(
    clips: list[tuple[str, str, Periodogram, Spec]],
    rig_spec: RigSpec,
    *,
    device: str = "cpu",
    ladder: tuple[int, ...] = (16, 48),
    iters: tuple[int, int, int] = (100, 100, 200),
    lr: float = 0.1,
    log: Any = print,
) -> dict[str, Any]:
    """Joint MAP over all ``clips`` (``(clip_id, group, periodogram, spec)``)
    with the rig-level parameters tied per ``rig_spec``."""
    t0 = time.time()
    spec0 = clips[0][3]
    R = clips[0][2].rps.shape[0]
    rig = RigParams(spec0, rig_spec, R, device)
    rcs = _prepare(clips, rig, device)
    _init_rig_from_clips(rig, rcs)
    _stages(rig, rcs, rig_free=True, ladder=ladder, iters=iters, lr=lr, log=log, t0=t0)
    per_clip: dict[str, dict[str, Any]] = {
        rc.clip_id: dict(group=rc.group, scores=_scores(rc), params=rc.model.export()) for rc in rcs
    }
    ex: list[float] = [float(v["scores"]["excess_over_loo"]) for v in per_clip.values()]
    log(
        f"  train excess median {np.median(ex):+.3f} mean {np.mean(ex):+.3f}  ({time.time() - t0:.0f}s)"
    )
    return dict(
        rig_spec=asdict(rig_spec),
        spec=asdict(spec0) | dict(freqs=None, times=None, rps=None),
        rig=rig.export(),
        rig_state={k: v.detach().cpu() for k, v in rig.state_dict().items()},
        clips=per_clip,
        seconds=time.time() - t0,
    )


def fit_heldout(
    fitted: dict[str, Any],
    clips: list[tuple[str, str, Periodogram, Spec]],
    *,
    device: str = "cpu",
    ladder: tuple[int, ...] = (16, 48),
    iters: tuple[int, int, int] = (100, 100, 200),
    lr: float = 0.1,
    log: Any = print,
) -> dict[str, Any]:
    """Freeze the rig's rig- and rotor-level parameters, fit only the clip
    nuisances of ``clips``; the predictive score of the rig model."""
    t0 = time.time()
    rig_spec = RigSpec(**fitted["rig_spec"])
    spec0 = clips[0][3]
    R = clips[0][2].rps.shape[0]
    rig = RigParams(spec0, rig_spec, R, device)
    rig.load_state_dict({k: v.to(device) for k, v in fitted["rig_state"].items()})
    for p in rig.parameters():
        p.requires_grad_(False)
    rcs = _prepare(clips, rig, device)
    with torch.no_grad():
        for rc in rcs:
            rc.model.level_db.copy_(
                (rc.model.profile_db - rig.profile_db[None, :]).median(dim=1).values
            )
    _stages(rig, rcs, rig_free=False, ladder=ladder, iters=iters, lr=lr, log=log, t0=t0)
    per_clip: dict[str, dict[str, Any]] = {
        rc.clip_id: dict(group=rc.group, scores=_scores(rc), params=rc.model.export()) for rc in rcs
    }
    ex: list[float] = [float(v["scores"]["excess_over_loo"]) for v in per_clip.values()]
    log(
        f"  held-out excess median {np.median(ex):+.3f} mean {np.mean(ex):+.3f}  ({time.time() - t0:.0f}s)"
    )
    return dict(clips=per_clip, seconds=time.time() - t0)


__all__ = ["RigSpec", "RigParams", "ClipInRig", "fit_rig", "fit_heldout"]
