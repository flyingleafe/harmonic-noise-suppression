"""The prior_v1 rig bank in noise model v4: one rig of ``noise-prior-bank/1``
(``noise-v3-fit/1``, :mod:`.drone_prior`) as a ``noise-v4-fit/3`` FLIGHT
payload (:mod:`data_processing.noise_model.v4`), the rules decided by Dmitrii
on 2026-10-24 (:data:`RULES`, version :data:`RULES_VERSION`).

A v1 rig's line ``(r, k)`` is a COHERENT core of mean power
``C = 10^{profile_db/10}`` at ``s = 1`` times a unit-mean-amplitude log-OU
envelope of variance ``s2_rk``, so its TOTAL power is ``T = C e^{s2_rk}`` and
its lag law ``T exp(s2_rk (rho(tau) - 1))``, ``rho = e^{-2 pi gamma_k |tau|}``.
v4 keeps the total per rotor and line and the lag law per order:

* the motor boost ``b_n = motor_boost_db 0.8^(n-1)`` that v1 put on the
  orders ``k = 3 p n`` becomes a v4 motor line (coherent, no pedestal) of
  power ``C (1 - 10^{-b_n/10})``, leaving the aerodynamic line its
  unboosted total ``T 10^{-b_n/10}`` (no motor line when ``b_n <= 0``);
* the pedestal is shared by the rotors in v4: ``s2_k = mean_r s2_rk``, split
  into an amplitude part capped at the v4 cap of the order's group
  (:data:`.v4_priors.AMP_GROUPS`) and a phase part with the SAME rate, which
  keeps ``exp(s2_k (rho - 1))`` exactly;
* the shaft (``sigma_nu``, ``lam``) is v1's flat law; the floor is v1's
  spline, level and exponent under the one-floor rule (``(mean_r s_r)^b``
  where v1 had ``mean_r s_r^b``: equal at equal speeds);
* the mics are unit capsules (``array_geometry`` with no array response and
  zero geometry); v1's per-clip line scatter ``mic_dev_sd_db`` is the render
  option ``line_scatter_db`` of the ``noise_v2`` pool, not a payload term;
* ``alpha_motor`` (v1 had no motor law) and the wind (v1 drew it per clip)
  are drawn ONCE per rig from the rig's own stream ``SeedSequence([seed,
  i])``: the wind from v1's ``wind_sc`` law, as an ``sc`` block on the v1 gust
  law with levels over the rig's 20-100 Hz floor at ``sbar = 1``.

:func:`convert_prior_v1` is pure and deterministic in ``(bank rig, seed,
shape)``; the payload is assembled by
:func:`data_processing.noise_model.prior_bank_v4.rig_payload`, the same
function that reads a rig back out of a ``noise-prior-bank-v4/1`` file.
Builder: ``scripts/prior_bank_v4_build.py``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from data_processing.noise_model import v4 as V4
from data_processing.noise_model.constants import AMP_RPS_REF
from data_processing.noise_model.prior_bank import PriorBank
from data_processing.noise_model.prior_bank_v4 import SILENT_LINE_DB, rig_payload
from data_processing.noise_model.spectrum import FLIGHT_SR, floor_ctrl_hz

from . import v4_priors as VP

__all__ = [
    "BAND_HZ",
    "GUST_LAW",
    "RULES",
    "RULES_VERSION",
    "SEED",
    "SOURCE_BANK",
    "convert_prior_v1",
    "floor_band_db",
    "prior_v4_const",
    "wind_shape_block",
]

RULES_VERSION = "prior_v4/1"
#: The v1 bank this conversion is defined on (``conf/online_mix/noise_prior_v1_5050.yaml``).
SOURCE_BANK = (
    "dload:prior-v1-banks@edbe77d9b3444460edd24f76d189dd2aef624a8bf13f37f87fcb979bed3515db"
    "/prior_v1_n16384.npz"
)
#: The base seed of the per-rig draws (``alpha_motor`` and the wind).
SEED = 20261024
#: The MD2 flight payload's (``results/noise_v4/flight/md2_fly125__flight_v4.json``).
BAND_HZ = (31.25, 7898.4375)
#: v1 drew its wind per 4 s render (the policy's ``render_duration_s``) with
#: ``gustiness`` knots: 3 knots per 4 s is a 0.75 Hz knot rate.
V1_RENDER_S = 4.0
#: v1's ``_sc_wind``: a capsule ``excess_db <= -10`` over its floor gets no wind.
WIND_SILENT_EXCESS_DB = -10.0
#: The SC gust law ``(speed_offset_ms, speed_scale_ms, gust_rate_hz)``: SC's
#: defaults and v1's knot rate.
GUST_LAW = (0.0, 2.0, 3.0 / V1_RENDER_S)

RULES: dict[str, str] = {
    "version": RULES_VERSION,
    "decided": "Dmitrii, 2026-10-24",
    "meta": "kind flight, rig prior_v1, blades and pole_pairs v1's, f_ref 80 rev/s, sr 16000, "
    "band_hz the MD2 flight payload's, front_end sr_work 32000",
    "lines": "per (r, k): C = 10^{profile_db/10}, T = C e^{s2_rk}; at k = 3 p n <= K with "
    "b = motor_boost_db 0.8^(n-1): b > 0 -> motor C (1 - 10^{-b/10}), aero T 10^{-b/10}; "
    "b <= 0 -> motor silent (-300 dB), aero T; elsewhere aero T",
    "pedestal": "s2_k = mean_r s2_rk; sigma2_amp = min(s2_k, amp cap of the AMP_GROUPS group), "
    "sigma2_phase = s2_k - sigma2_amp; gamma_amp_hz = gamma_phase_hz = am_gamma_hz",
    "speed_laws": "alpha_aero = amp_exp; alpha_motor ~ N(0, 2) truncated to |alpha| <= 4, "
    "per-rig stream",
    "shaft": "eps = sigma_nu / (2 pi 80), a_sigma = 0, lam_ref_per_s = lam, b_lam = 0",
    "floor": "one_floor_mean_speed; mean_db, shape_z, shape_sd_db v1's on floor_ctrl_hz(16000); "
    "b0 = floor_exp, b1 = 0, static_rel = floor_static_rel",
    "mics": "array_geometry, array_response None, geometry_db zeros: unit capsules; v1's "
    "mic_dev_sd_db is the render option line_scatter_db",
    "wind": "per rig, per-rig stream: none with prob 1 - p_wind; else u ~ U(0, 1), per mic "
    "excess = max_over_floor_db u - Exp(shield_mean_db) dB, <= -10 -> WIND_OFF_DB, else "
    "level_db = F20_100 + excess, F20_100 the rig floor's 20-100 Hz mean at sbar = 1 with the "
    "static share; kind sc, SC law offset 0, scale 2, gust_rate_hz gustiness / 4 s, one "
    "sc_shape for the bank (sr 16000, 4 s clips)",
}


def wind_shape_block() -> dict[str, Any]:
    """The bank's one ``sc`` shape: :func:`.dregon_wind_v4.sc_shape` at
    :data:`GUST_LAW`, 16 kHz, on 4 s clips (v1's render, 3 knots each)."""
    from .dregon_wind_v4 import sc_shape  # noqa: PLC0415 (heavy module, builder-only)

    return sc_shape(*GUST_LAW, sr=FLIGHT_SR, clip_s=V1_RENDER_S)


def prior_v4_const(
    bank: PriorBank, *, shape: dict[str, Any], seed: int, source_bank: str = SOURCE_BANK
) -> dict[str, Any]:
    """The part every converted rig of ``bank`` shares
    (:func:`data_processing.noise_model.prior_bank_v4.bank_const`)."""
    c = bank.const
    R, M = int(c["n_rotors"]), int(c["n_mics"])
    return {
        "kind": "flight",
        "rig": "prior_v1",
        "n_rotors": R,
        "n_mics": M,
        "blades": int(c["spec"]["blades"]),
        "f_ref_rev_s": float(AMP_RPS_REF),
        "sr": FLIGHT_SR,
        "band_hz": list(BAND_HZ),
        "front_end": {"sr_work": V4.WORK_RATE_HZ},
        "shaft": {"a_sigma": 0.0, "b_lam": 0.0},
        "floor": {
            "floor_rule": V4.FLOOR_RULE_ONE,
            "ctrl_hz": floor_ctrl_hz(FLIGHT_SR).tolist(),
            "b1": 0.0,
        },
        "mic_response": {
            "kind": "array_geometry",
            "array_response": None,
            "geometry_db": np.zeros((R, M)).tolist(),
        },
        "wind": {
            "kind": V4.WIND_SC,
            "speed_offset_ms": GUST_LAW[0],
            "speed_scale_ms": GUST_LAW[1],
            "gust_rate_hz": GUST_LAW[2],
            "shape": shape,
        },
        "provenance": {"source_bank": source_bank, "rules": RULES_VERSION, "seed": int(seed)},
    }


def floor_band_db(fit: dict[str, Any], freqs_hz: Any) -> float:
    """``10 log10`` of the floor PSD of ``fit`` at ``sbar = 1``, static share
    included, averaged over the ``freqs_hz`` inside :data:`.v4.WIND_REF_BAND_HZ`
    (the ``sc`` level's band): the v4 payload's periodogram units."""
    par = V4.V4Params.from_payload(fit)
    f = np.asarray(freqs_hz, dtype=np.float64)
    lo, hi = V4.WIND_REF_BAND_HZ
    f = f[(f >= lo) & (f <= hi)]
    floor = par.floor.mean_speed_psd(f, np.ones(par.n_rotors))
    floor = floor + par.floor.psd(f) * par.floor.static_rel
    return float(10.0 * np.log10(np.mean(floor)))


def _alpha_motor(rng: np.random.Generator) -> float:
    """``N(ALPHA_MOTOR)`` truncated to ``|alpha| <= ALPHA_MOTOR_MAX`` (rejection)."""
    mu, sd = VP.ALPHA_MOTOR
    while True:
        a = float(rng.normal(mu, sd))
        if abs(a) <= VP.ALPHA_MOTOR_MAX:
            return a


def convert_prior_v1(
    bank: PriorBank,
    i: int,
    *,
    shape: dict[str, Any],
    seed: int,
    source_bank: str = SOURCE_BANK,
) -> dict[str, Any]:
    """Rig ``i`` of the v1 ``bank`` as a ``noise-v4-fit/3`` flight payload
    (module docstring; :data:`RULES`). ``shape`` is :func:`wind_shape_block`'s
    (shared, not copied); ``seed`` the base of the rig's own draws."""
    i = int(i)
    pl = bank.payload(i)
    p, d = pl["params"], pl["_prior"]["drawn"]
    spec = pl["_prior"]["spec"]
    const = prior_v4_const(bank, shape=shape, seed=seed, source_bank=source_bank)
    R, M = const["n_rotors"], const["n_mics"]
    rng_alpha, rng_wind = (
        np.random.default_rng(s) for s in np.random.SeedSequence([int(seed), i]).spawn(2)
    )

    # lines: v1 core C and total T; the motor boost becomes a coherent motor line
    prof = np.asarray(p["profile"]["profile_db"], dtype=np.float64)
    s2 = np.asarray(p["am"]["sigma2"], dtype=np.float64)
    if prof.shape != (R, s2.shape[1]) or s2.shape[0] != R:
        raise ValueError(f"rig {i}: profile {prof.shape} and am.sigma2 {s2.shape} are not (R, K)")
    K = prof.shape[1]
    core = 10.0 ** (prof / 10.0)
    total = core * np.exp(s2)
    pole_pairs, period = int(d["pole_pairs"]), int(d["motor_period"])
    if period != 3 * pole_pairs:
        raise ValueError(f"rig {i}: motor period {period} is not 3 p for p = {pole_pairs}")
    n = np.arange(1, K // period + 1)
    rows = period * n - 1
    boost = float(d["motor_boost_db"]) * float(spec["motor_decay_per_multiple"]) ** (n - 1)
    live = boost > 0.0
    keep = np.where(live, 10.0 ** (-np.maximum(boost, 0.0) / 10.0), 1.0)  # (n_m,)
    aero = total.copy()
    aero[:, rows] = total[:, rows] * keep[None, :]
    motor_db = np.full((R, n.size), SILENT_LINE_DB)
    if np.any(live):
        motor = core[:, rows[live]] * (1.0 - keep[live])[None, :]
        motor_db[:, live] = 10.0 * np.log10(motor)

    # pedestal: shared by the rotors, split at the v4 amplitude cap, one rate
    s2_k = s2.mean(axis=0)
    s2_amp = np.minimum(s2_k, VP.amp_cap(np.arange(1, K + 1)))
    gamma = np.asarray(p["am"]["gamma_hz"], dtype=np.float64)[0]

    rig: dict[str, Any] = {
        "profile_aero_db": 10.0 * np.log10(aero),
        "profile_motor_db": motor_db,
        "pole_pairs": pole_pairs,
        "alpha_aero": float(p["profile"]["amp_exp"]),
        "alpha_motor": _alpha_motor(rng_alpha),
        "sigma2_amp": s2_amp,
        "gamma_amp_hz": gamma,
        "sigma2_phase": s2_k - s2_amp,
        "gamma_phase_hz": gamma,
        "eps": float(p["sigma_nu"]) / (2.0 * math.pi * float(AMP_RPS_REF)),
        "lam_ref_per_s": float(p["lam"]),
        "floor_mean_db": float(p["floor"]["floor_mean_db"]),
        "floor_shape_z": np.asarray(p["floor"]["floor_shape_z"], dtype=np.float64),
        "floor_shape_sd_db": float(p["floor"]["floor_shape_sd_db"]),
        "floor_b0": float(p["floor"]["floor_exp"]),
        "floor_static_rel": float(p["floor"]["floor_static_rel"]),
        "wind_level_db": None,
        "source_index": i,
        "v1_seed": int(pl["_prior"].get("seed", -1)),
    }

    # wind, once per rig from v1's per-clip law, over this rig's 20-100 Hz floor
    w = p["wind_sc"] or {}
    if rng_wind.uniform() < float(w.get("p_wind", 0.7)):
        u = float(rng_wind.uniform())
        excess = float(w.get("max_over_floor_db", 33.0)) * u - rng_wind.exponential(
            float(w.get("shield_mean_db", 8.0)), M
        )
        ref = floor_band_db(rig_payload(const, rig), shape["freqs_hz"])
        rig["wind_level_db"] = np.where(
            excess <= WIND_SILENT_EXCESS_DB, V4.WIND_OFF_DB, ref + excess
        )
    return rig_payload(const, rig)
