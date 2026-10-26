"""Synthetic VALIDATION SECTORS for the wide-prior (prior_v4) training runs.

A sector is a small set of rigs a model is scored on beside the real panel
(``docs/experiments/rp-synthetic.md`` § "Validation sectors"; Dmitrii,
2026-10-25/26). The rigs are drawn the way the training bank's are, from a
base seed the bank does not use, and pass the same identifiability gate on the
same trajectory pool (``scripts/validation_sectors_v4_build.py``):

* ``uniform``: gated :func:`.prior_v4.sample_prior_v4` draws, unselected: the
  only synthetic view used for stopping and checkpoint selection;
* three AXIS sectors, one tail of one coordinate of a larger pool of gated
  draws each: ``weak_comb`` (lowest v1 ``k2_over_floor_db``), ``strong_motor``
  (highest v1 ``motor_boost_db``) and ``wide_pedestal`` (highest
  :func:`pedestal_width_hz`; the phase pedestal is drawn per order, so no one
  law coordinate carries a rig's pedestal width);
* two NEIGHBOURHOODS of real-drone fits (:data:`FITS`): the DREGON bench fit and
  the MD2 standby/cruise pair, each perturbed by :func:`perturb_fit` at the
  strength that puts the median :func:`mel_distance_db` to the fit at
  :data:`NEIGHBOURHOOD_FRAC` of the distance between the two fits.

DISTANCE (Dmitrii: "perceptual"): :func:`log_mel` is the mean-removed dB of a
64-band mel filterbank (30 Hz - 7.9 kHz) on the mic-mean
``v4.expected_periodogram`` (2048-point Hann at 16 kHz, no wind) at three
reference speeds, the rotors at ``speed + (-3, -1, +1, +3)`` rev/s; a
standby/cruise pair is power-blended at each speed by
:func:`.render.regime_blend_weight`. :func:`mel_distance_db` is the RMS of the
difference over bands and speeds. The mean removal makes it blind to level,
which the stream normalises per window.

PRIOR-SHAPED PERTURBATION (Dmitrii, 2026-10-26: :func:`perturb_fit`): the
fit's own parameters moved by the wide prior's spreads
(:class:`.drone_prior.PriorSpec`, :data:`.prior_v4.PHASE_GROUPS`) times one
strength ``s``, ONE draw shared by the regimes of a pair:

* comb, relative to the floor: the law's coordinates ``k2`` (all orders), the
  decay slope (times ``log10(k / 2)``), the blade-pass boost, the odd-order
  penalty (linear in ``log10 k`` through ``k = 3`` and ``30``), the motor boost
  (motor lines); a level per rotor and a scatter per (rotor, line);
* speed laws: ``alpha_aero`` (sd 1.5), ``alpha_motor`` (sd 2, kept within the
  prior's bound), the floor exponent ``b0`` (sd 1.5, the log-normal law's
  linear spread at its median, floored at 0) and the static share (log sd 1);
* floor shape: the 14 spline weights ``shape_z`` (sd 0.7, the prior's);
* pedestal per order: amplitude ``sigma2`` (log sd 1.0 / 0.7 / 0.3 for
  ``k <= 8`` / ``<= 30`` / above, capped at the larger of the bank cap and the
  fit's own value), phase ``sigma2`` and half-width ``W`` (the PHASE_GROUPS
  log sds, the v4 caps), the phase rate re-derived from them.

The fit's shaft, mic response, motor orders and floor level are kept; its wind
is dropped (a bench fit's is per recording; the stream adds ``wind_sc`` per
clip as for the bank). The floor level is not moved because the comb moves
against it and the stream normalises the level.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from data_processing.noise_model import v4 as V4
from data_processing.noise_model.render import regime_blend_weight

from . import v4_priors as VP
from .drone_prior import PriorSpec
from .prior_v4 import PHASE_GROUPS, amp_cap, phase_gamma_hz

__all__ = [
    "FITS",
    "NEIGHBOURHOOD_FRAC",
    "OFFSETS",
    "SPEEDS",
    "fit_regimes",
    "log_mel",
    "mel_distance_db",
    "pedestal_width_hz",
    "perturb_fit",
]

SR = 16000
N_FFT = 2048
#: Reference speeds (rev/s) and the rotors' offsets about each.
SPEEDS = (40.0, 60.0, 80.0)
OFFSETS = (-3.0, -1.0, 1.0, 3.0)
N_MELS = 64
MEL_BAND_HZ = (30.0, 7900.0)
#: The neighbourhoods' median distance to their fit, as a fraction of the
#: distance between the two fits.
NEIGHBOURHOOD_FRAC = 1.0 / 3.0

#: The fits the neighbourhoods are drawn around: ``name -> (regimes, traj_rig)``.
#: DREGON: the bench fit of the explainer's "DREGON bench fit only" row (bench
#: lines, floor and pair EQ; no flight stage, no wind). MD2: the per-regime
#: flight pair.
FITS: dict[str, tuple[dict[str, str], str]] = {
    "dregon": (
        {"single": "results/noise_v4/bench/restarts/dregon__kappa0__r5__s2__c7.json"},
        "dregon",
    ),
    "md2": (
        {
            "standby": "results/noise_v4/flight/restarts/"
            "md2_fly125_standby_fpb__flight_v4__s0__r6c1.json",
            "cruise": "results/noise_v4/flight/restarts/"
            "md2_fly125_cruise_fpb__flight_v4__s0__r6c3.json",
        },
        "michaels",
    ),
}


def _calm(fit: Mapping[str, Any]) -> dict[str, Any]:
    return dict(fit, params=dict(fit["params"], wind=None))


def fit_regimes(name: str, root: str | Path = ".") -> dict[str, dict[str, Any]]:
    """The regimes of fit ``name`` of :data:`FITS`, read from ``root``, wind dropped."""
    paths, _ = FITS[name]
    return {k: _calm(json.loads((Path(root) / p).read_text())) for k, p in paths.items()}


def _mel_fb() -> np.ndarray:
    f = np.fft.rfftfreq(N_FFT, 1.0 / SR)

    def mel(x: Any) -> Any:
        return 2595.0 * np.log10(1.0 + np.asarray(x) / 700.0)

    edges = 700.0 * (
        10.0 ** (np.linspace(mel(MEL_BAND_HZ[0]), mel(MEL_BAND_HZ[1]), N_MELS + 2) / 2595.0) - 1.0
    )
    fb = np.zeros((N_MELS, f.size))
    for i in range(N_MELS):
        lo, c, hi = edges[i : i + 3]
        fb[i] = np.clip(np.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)), 0.0, None)
    return fb / np.maximum(fb.sum(axis=1, keepdims=True), 1e-12)


_FB = _mel_fb()


def log_mel(regimes: Mapping[str, Mapping[str, Any]] | Mapping[str, Any]) -> np.ndarray:
    """``(len(SPEEDS), N_MELS)`` mean-removed log-mel (dB) of one payload or a
    ``standby``/``cruise`` pair (module docstring), wind excluded."""
    regs = {"single": regimes} if "params" in regimes else dict(regimes)
    offs = np.asarray(OFFSETS)
    rows = []
    for s in SPEEDS:
        rps = s + offs
        if "single" in regs:
            weights = {"single": 1.0}
        else:
            w = float(regime_blend_weight(np.repeat(rps[:, None], 1600, 1), sr=SR).mean())
            weights = {"standby": 1.0 - w, "cruise": w}
        e = sum(
            wt * V4.expected_periodogram(_calm(regs[k]), rps, n_fft=N_FFT, sr=SR).mean(axis=0)
            for k, wt in weights.items()
            if wt > 0.0
        )
        rows.append(10.0 * np.log10(np.maximum(_FB @ e, 1e-30)))
    x = np.asarray(rows)
    return x - x.mean()


def mel_distance_db(a: np.ndarray, b: np.ndarray) -> float:
    """RMS difference (dB) of two :func:`log_mel` arrays."""
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def pedestal_width_hz(fit: Mapping[str, Any], rps: float = 80.0) -> float:
    """A rig's line width: the phase-pedestal half-width ``(1 - e^{-s2}) W``
    of each aerodynamic order under the Nyquist at ``rps``, weighted by the
    order's rotor-mean power at ``f_ref``."""
    par = V4.V4Params.from_payload(fit)
    ped = fit["params"]["pedestal"]
    s2 = np.asarray(ped["sigma2_phase"], dtype=np.float64)
    w = np.asarray(ped["width_phase_hz"], dtype=np.float64)
    k = np.arange(1, par.k_max + 1)
    keep = k * rps < 0.5 * SR
    p = (10.0 ** (par.profile_aero_db / 10.0)).mean(axis=0)[keep]
    return float(np.sum(p * -np.expm1(-s2[keep]) * w[keep]) / np.sum(p))


def _group_sd(k: np.ndarray, col: int) -> np.ndarray:
    out = np.empty(k.shape, dtype=np.float64)
    for row in PHASE_GROUPS:
        out[(k >= row[0]) & (k <= row[1])] = row[col]
    return out


def perturb_fit(
    regimes: Mapping[str, Mapping[str, Any]],
    rng: np.random.Generator,
    strength: float,
    spec: PriorSpec = PriorSpec(),
) -> dict[str, dict[str, Any]]:
    """One prior-shaped perturbation of ``regimes`` (``{"single": fit}`` or a
    ``standby``/``cruise`` pair of ``noise-v4-fit/3`` payloads) at ``strength``
    (module docstring): ONE draw, applied to every regime on its own orders.
    Returns new payloads, wind dropped, each with ``perturbation`` provenance."""
    s = float(strength)
    pars = {name: V4.V4Params.from_payload(fit) for name, fit in regimes.items()}
    R = {p.n_rotors for p in pars.values()}
    blades = {p.blades for p in pars.values()}
    if len(R) != 1 or len(blades) != 1:
        raise ValueError("a regime pair must share its rotors and blade count")
    (n_rot,), (n_bl,) = R, blades
    K = max(p.k_max for p in pars.values())
    k = np.arange(1, K + 1, dtype=np.float64)
    motor_all = sorted({int(o) for p in pars.values() for o in p.motor_orders})

    # --- the comb, relative to the floor (dB) ---
    d_k2 = rng.normal(0.0, spec.k2_over_floor_db[1])
    d_slope = rng.normal(0.0, spec.slope_db_dec[1])
    d_bpf = rng.normal(0.0, spec.bpf_boost_db[1])
    d_pen3 = rng.normal(0.0, spec.odd_pen_k3_db[1])
    d_pen30 = rng.normal(0.0, spec.odd_pen_k30_db[1])
    d_motor = rng.normal(0.0, spec.motor_boost_db[1])
    law = d_k2 - d_slope * np.maximum(np.log10(k / 2.0), 0.0)
    law = np.where(k == n_bl, law + d_bpf, law)
    odd = (k % n_bl) != 0
    law = np.where(odd, law - (d_pen3 + (d_pen30 - d_pen3) * np.log10(k / 3.0)), law)
    level = rng.normal(0.0, spec.rotor_level_sd_db, (n_rot, 1))
    d_aero = law[None, :] + level + rng.normal(0.0, spec.rotor_line_sd_db, (n_rot, K))
    d_mot = {
        o: d_motor + level[:, 0] + rng.normal(0.0, spec.rotor_line_sd_db, n_rot) for o in motor_all
    }
    # --- speed laws, floor shape and static share ---
    d_alpha_a = rng.normal(0.0, spec.amp_exp[1])
    d_alpha_m = rng.normal(0.0, VP.ALPHA_MOTOR[1])
    d_b0 = rng.normal(0.0, np.exp(spec.log_floor_exp[0]) * spec.log_floor_exp[1])
    d_static = rng.normal(0.0, spec.log_floor_static_rel[1])
    d_z = rng.normal(0.0, 0.7, 14)
    # --- pedestal per order ---
    sd_amp = np.where(
        k <= 8,
        spec.log_am_s2_low[1],
        np.where(k <= 30, spec.log_am_s2_mid[1], 0.3),
    )
    z_amp = rng.standard_normal(K) * sd_amp
    z_s2p = rng.standard_normal(K) * _group_sd(k.astype(np.int64), 3)
    z_w = rng.standard_normal(K) * _group_sd(k.astype(np.int64), 5)

    out: dict[str, dict[str, Any]] = {}
    for name, fit in regimes.items():
        par = pars[name]
        Kr = par.k_max
        q = copy.deepcopy(dict(fit))
        p = q["params"]
        p["wind"] = None
        p.pop("hierarchy", None)
        p["profile_aero_db"] = (par.profile_aero_db + s * d_aero[:, :Kr]).tolist()
        if par.motor_orders.size:
            dm = np.stack([d_mot[int(o)] for o in par.motor_orders], axis=1)  # (R, n_m)
            p["profile_motor_db"] = (par.profile_motor_db + s * dm).tolist()
        p["alpha_aero"] = float(par.alpha_aero + s * d_alpha_a)
        p["alpha_motor"] = float(
            np.clip(par.alpha_motor + s * d_alpha_m, -VP.ALPHA_MOTOR_MAX, VP.ALPHA_MOTOR_MAX)
        )
        fl = dict(p["floor"])
        fl["shape_z"] = (np.asarray(fl["shape_z"], dtype=np.float64) + s * d_z).tolist()
        fl["b0"] = float(max(float(fl["b0"]) + s * d_b0, 0.0))
        fl["static_rel"] = float(float(fl.get("static_rel") or 0.0) * np.exp(s * d_static))
        fl.pop("ctrl_db", None)  # derived from shape_z; stale once it moves
        p["floor"] = fl
        ped = dict(p["pedestal"])
        kr = np.arange(1, Kr + 1)
        s2a = np.asarray(ped["sigma2_amp"], dtype=np.float64)
        s2a = np.minimum(s2a * np.exp(s * z_amp[:Kr]), np.maximum(amp_cap(kr), s2a))
        s2p = np.asarray(ped["sigma2_phase"], dtype=np.float64)
        s2p = np.minimum(s2p * np.exp(s * z_s2p[:Kr]), VP.PHASE_SIGMA2_MAX)
        w = np.asarray(ped["width_phase_hz"], dtype=np.float64)
        w = np.minimum(w * np.exp(s * z_w[:Kr]), VP.W_MAX_HZ)
        ped.update(
            sigma2_amp=s2a.tolist(),
            sigma2_phase=s2p.tolist(),
            width_phase_hz=w.tolist(),
            gamma_phase_hz=phase_gamma_hz(s2p, w).tolist(),
        )
        p["pedestal"] = ped
        q["perturbation"] = {
            "rule": "sectors_v4.perturb_fit (prior-shaped, one draw per rig)",
            "strength": s,
            "drawn": dict(
                k2_db=float(d_k2),
                slope_db_dec=float(d_slope),
                bpf_db=float(d_bpf),
                odd_pen_k3_db=float(d_pen3),
                odd_pen_k30_db=float(d_pen30),
                motor_db=float(d_motor),
                alpha_aero=float(d_alpha_a),
                alpha_motor=float(d_alpha_m),
                b0=float(d_b0),
                log_static=float(d_static),
            ),
        }
        V4.V4Params.from_payload(q)  # the result must still be a valid payload
        out[name] = q
    return out
