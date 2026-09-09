"""Planted-rig control for the rig fit.

A rig with known rig-, rotor- and clip-level parameters generates the
expected spectrum of every clip on *real* carrier trajectories; periodogram
cells are drawn as independent exponentials around it (the Whittle model
exactly, so the leave-one-out reference is unbiased and a correct fit sits
at excess 0). ``fit_rig`` then has to give the planted rig back.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .data import Periodogram
from .fit import _inv_softplus
from .model import Spec
from .population import PopulationFitSpec, fit_population, fit_population_heldout
from .rig import ClipInRig, RigParams, RigSpec, fit_heldout, fit_rig


def plant_and_draw(
    pgs: list[Periodogram],
    spec: Spec,
    rig_spec: RigSpec,
    *,
    seed: int = 0,
    delta_std_db: float = 2.2,
    mic_gain_spread_db: float = 4.0,
) -> tuple[list[Periodogram], dict[str, Any]]:
    """Returns periodograms whose ``power`` is drawn from a planted rig, and
    the planted parameters."""
    rng = np.random.default_rng(seed)
    R = pgs[0].rps.shape[0]
    rig = RigParams(spec, rig_spec, R, "cpu")
    K, M = spec.n_harm, spec.n_mics
    k = np.arange(1, K + 1)
    profile = (
        2.0 - 12.0 * np.log10(k) + rng.normal(0, 2.0, K)
    )  # roll-off + jitter; floor visible at high k
    delta = rng.normal(0, delta_std_db, (R, K))
    mic_gain = rng.normal(0, mic_gain_spread_db / 2, (M, R))
    shape_z = rng.normal(0, 1.0, rig.floor_shape_z.numel())
    with torch.no_grad():
        rig.profile_db.copy_(torch.as_tensor(profile, dtype=torch.float32))
        rig.delta_db.copy_(torch.as_tensor(delta, dtype=torch.float32))
        rig.floor_tilt_db_oct.fill_(-6.0)
        rig.floor_shape_z.copy_(torch.as_tensor(shape_z, dtype=torch.float32))
        rig.gamma0_raw.fill_(_inv_softplus(2.0))
        rig.slope_raw.fill_(_inv_softplus(0.5))
        rig.log_width_scale.copy_(torch.as_tensor(rng.normal(0, 0.15, R), dtype=torch.float32))
        rig.mic_gain_db.copy_(torch.as_tensor(mic_gain, dtype=torch.float32))
    planted: dict[str, Any] = dict(rig=rig.export(), clips={})
    out = []
    for i, pg in enumerate(pgs):
        s = Spec(**{**spec.__dict__, "freqs": pg.freqs, "times": pg.times, "rps": pg.rps})
        m = ClipInRig(s, rig, "cpu")
        with torch.no_grad():
            m.floor_mean_db.fill_(-25.0 + rng.normal(0, 2.0))
            m.level_db.copy_(torch.as_tensor(rng.normal(0, 2.0, R), dtype=torch.float32))
            m.h_z.normal_()
            m.floor_level_z.normal_()
            if s.umod_std_db > 0:
                m.u_z.normal_()
            spectrum = m.forward().numpy()
        power = rng.exponential(1.0, spectrum.shape) * spectrum
        planted["clips"][i] = m.export()
        out.append(
            Periodogram(power.astype(np.float32), pg.freqs, pg.times, pg.rps, pg.n_fft, pg.hop)
        )
    return out, planted


def plant_population_and_draw(
    pgs: list[Periodogram],
    spec: Spec,
    rig_spec: RigSpec,
    *,
    seed: int = 0,
    floor_mean_db: float = -25.0,
    floor_std_db: float = 2.5,
    level_std_db: float = 3.0,
) -> tuple[list[Periodogram], dict[str, Any]]:
    """Draw a multi-clip population with known profile covariance."""
    rng = np.random.default_rng(seed)
    n_rotors = pgs[0].rps.shape[0]
    rig = RigParams(spec, rig_spec, n_rotors, "cpu")
    k = np.arange(1, spec.n_harm + 1, dtype=np.float32)
    profile = 3.0 - 14.0 * np.log10(k)
    delta = rng.normal(0.0, 1.5, (n_rotors, spec.n_harm))
    delta -= delta.mean(axis=0, keepdims=True)
    rank = rig_spec.profile_rank
    modes = np.zeros((rank, spec.n_harm), dtype=np.float32)
    if rank:
        logk = np.log2(k)
        candidates = [
            -(logk - logk.mean()),
            np.exp(-0.5 * ((logk - 1.5) / 0.65) ** 2),
            np.cos(np.pi * k),
        ]
        scales = (4.0, 3.0, 2.0)
        for q in range(rank):
            mode = candidates[q % len(candidates)].astype(np.float32)
            mode -= mode.mean()
            mode /= max(float(np.sqrt(np.mean(mode**2))), 1e-6)
            modes[q] = scales[q % len(scales)] * mode
    with torch.no_grad():
        rig.profile_db.copy_(torch.as_tensor(profile))
        rig.delta_db.copy_(torch.as_tensor(delta))
        if rig.profile_basis_db is not None:
            rig.profile_basis_db.copy_(torch.as_tensor(modes))
            assert rig.profile_mode_std_raw is not None
            mode_std = torch.as_tensor(
                np.sqrt(np.mean(modes**2, axis=1)), dtype=torch.float32
            ).clamp_min(1e-3)
            rig.profile_mode_std_raw.copy_(torch.log(torch.expm1(mode_std)))
        rig.floor_tilt_db_oct.fill_(-6.0)
        rig.gamma0_raw.fill_(_inv_softplus(2.0))
        rig.slope_raw.fill_(_inv_softplus(0.5))

    planted: dict[str, Any] = {
        "rig": rig.export(),
        "population": {
            "line_floor_mean_db": float(profile[min(1, spec.n_harm - 1)] - floor_mean_db),
            "line_floor_std_db": float(np.sqrt(floor_std_db**2 + level_std_db**2 / n_rotors)),
            "rotor_contrast_std_db": level_std_db,
        },
        "clips": {},
    }
    out: list[Periodogram] = []
    for i, pg in enumerate(pgs):
        clip_spec = Spec(**{**spec.__dict__, "freqs": pg.freqs, "times": pg.times, "rps": pg.rps})
        model = ClipInRig(clip_spec, rig, "cpu")
        with torch.no_grad():
            model.floor_mean_db.fill_(floor_mean_db + rng.normal(0.0, floor_std_db))
            model.level_db.copy_(
                torch.as_tensor(rng.normal(0.0, level_std_db, n_rotors), dtype=torch.float32)
            )
            if model.profile_z is not None:
                model.profile_z.copy_(
                    torch.as_tensor(
                        rng.normal(size=tuple(model.profile_z.shape)), dtype=torch.float32
                    )
                )
            model.h_z.normal_()
            model.floor_level_z.normal_()
            model.floor_tilt_z.normal_()
            if clip_spec.rps_offset:
                model.rps_offset_knots.normal_(std=clip_spec.rps_offset_std)
            if clip_spec.umod_std_db > 0:
                model.u_z.normal_()
            spectrum = model.forward().numpy()
        power = rng.exponential(1.0, spectrum.shape) * spectrum
        planted["clips"][i] = model.export()
        out.append(
            Periodogram(
                power.astype(np.float32),
                pg.freqs,
                pg.times,
                pg.rps,
                pg.n_fft,
                pg.hop,
            )
        )
    return out, planted


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.corrcoef(a - a.mean(), b - b.mean())[0, 1])


def planted_rig_control(
    pgs: list[Periodogram],
    spec: Spec,
    rig_spec: RigSpec,
    *,
    n_train: int,
    seed: int = 0,
    device: str = "cpu",
    iters: tuple[int, int, int] = (60, 60, 120),
    log: Any = print,
) -> dict[str, Any]:
    """Plant, fit on the first ``n_train`` clips, score the rest held-out,
    and compare the recovered rig with the planted one."""
    drawn, planted = plant_and_draw(pgs, spec, rig_spec, seed=seed)

    def mk(i: int) -> tuple[str, str, Periodogram, Spec]:
        d = drawn[i]
        return (
            f"ctl_{i}",
            "control",
            d,
            Spec(**{**spec.__dict__, "freqs": d.freqs, "times": d.times, "rps": d.rps}),
        )

    train = [mk(i) for i in range(n_train)]
    test = [mk(i) for i in range(n_train, len(drawn))]
    fitted = fit_rig(train, rig_spec, device=device, iters=iters, log=log)
    held = (
        fit_heldout(fitted, test, device=device, iters=iters, log=log) if test else dict(clips={})
    )
    pr, fr = planted["rig"], fitted["rig"]
    k0 = 1  # order 1 is inside the floor's low band on real carriers; skip it
    res = dict(
        train_excess=float(
            np.median([c["scores"]["excess_over_loo"] for c in fitted["clips"].values()])
        ),
        heldout_excess=float(
            np.median([c["scores"]["excess_over_loo"] for c in held["clips"].values()])
        )
        if held["clips"]
        else float("nan"),
        profile_corr=_corr(pr["profile_db"][k0:], fr["profile_db"][k0:]),
        profile_rms_db=float(np.std((fr["profile_db"] - pr["profile_db"])[k0:])),
        delta_corr=_corr(pr["delta_db"][:, k0:], fr["delta_db"][:, k0:]),
        delta_rms_db=float(np.std((fr["delta_db"] - pr["delta_db"])[:, k0:])),
        delta_planted_std=float(np.std(pr["delta_db"])),
        width_scale_corr=_corr(np.log(pr["width_scale"]), np.log(fr["width_scale"])),
        gamma0=(pr["gamma0"], fr["gamma0"]),
        gamma_slope=(pr["gamma_slope"], fr["gamma_slope"]),
        mic_gain_corr=_corr(pr["mic_gain_db"], fr["mic_gain_db"]),
        mic_gain_rms_db=float(np.std(fr["mic_gain_db"] - pr["mic_gain_db"])),
        tilt=(pr["floor_tilt_db_oct"], fr["floor_tilt_db_oct"]),
        shape_corr=_corr(pr["floor_shape_db"], fr["floor_shape_db"]),
    )
    if spec.umod_std_db > 0:
        pu = np.concatenate([planted["clips"][i]["umod_db"].ravel() for i in range(n_train)])
        fu = np.concatenate(
            [fitted["clips"][f"ctl_{i}"]["params"]["umod_db"].ravel() for i in range(n_train)]
        )
        res["umod_corr"] = _corr(pu, fu)
        res["umod_std"] = (float(pu.std()), float(fu.std()))
    return res


def planted_population_control(
    pgs: list[Periodogram],
    spec: Spec,
    rig_spec: RigSpec,
    *,
    n_train: int,
    fit_spec: PopulationFitSpec = PopulationFitSpec(),
    seed: int = 0,
    device: str = "cpu",
    map_iters: tuple[int, int, int] = (30, 30, 60),
    vi_iters: int = 100,
    log: Any = print,
) -> dict[str, Any]:
    """Fit a planted population and score fresh clips without refitting globals."""
    drawn, planted = plant_population_and_draw(pgs, spec, rig_spec, seed=seed)

    def mk(i: int) -> tuple[str, str, Periodogram, Spec]:
        pg = drawn[i]
        clip_spec = Spec(**{**spec.__dict__, "freqs": pg.freqs, "times": pg.times, "rps": pg.rps})
        return f"population_ctl_{i}", "control", pg, clip_spec

    fitted = fit_population(
        [mk(i) for i in range(n_train)],
        rig_spec,
        fit_spec=fit_spec,
        device=device,
        map_iters=map_iters,
        vi_iters=vi_iters,
        seed=seed,
        log=log,
    )
    heldout = fit_population_heldout(
        fitted,
        [mk(i) for i in range(n_train, len(drawn))],
        device=device,
        map_iters=map_iters,
        vi_iters=vi_iters,
        seed=seed + 100,
        log=log,
    )
    planted_basis = np.asarray(planted["rig"]["profile_basis_db"])
    fitted_basis = np.asarray(fitted["rig"]["profile_basis_db"])
    planted_cov = planted_basis.T @ planted_basis
    fitted_cov = fitted_basis.T @ fitted_basis
    covariance_cosine = float(
        np.sum(planted_cov * fitted_cov)
        / max(np.linalg.norm(planted_cov) * np.linalg.norm(fitted_cov), 1e-12)
    )
    return {
        "planted_population": planted["population"],
        "fitted_population": fitted["population"],
        "profile_covariance_cosine": covariance_cosine,
        "train_excess": float(
            np.median([clip["scores"]["excess_over_loo"] for clip in fitted["clips"].values()])
        ),
        "heldout_excess": float(
            np.median([clip["scores"]["excess_over_loo"] for clip in heldout["clips"].values()])
        ),
        "heldout_iw_nll": float(
            np.median([clip["iw_nll_per_cell"] for clip in heldout["clips"].values()])
        ),
        "heldout_iw_ess": float(np.median([clip["iw_ess"] for clip in heldout["clips"].values()])),
    }
