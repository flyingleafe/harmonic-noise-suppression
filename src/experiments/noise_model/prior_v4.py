"""The wide prior in noise model v4: a drone drawn from the prior_v1 laws
(:func:`.drone_prior.sample_prior`, a ``noise-v3-fit/1`` payload) and mapped
to a ``noise-v4-fit/3`` FLIGHT payload (:mod:`data_processing.noise_model.v4`)
by the rules decided by Dmitrii on 2026-10-24 and revised on 2026-10-25
(:data:`RULES`, version :data:`RULES_VERSION`), with a phase pedestal drawn
per order (:func:`with_phase_pedestal`). :func:`sample_prior_v4` is one draw;
``scripts/prior_bank_v4_build.py`` gates the draws and writes the bank.

A v1 rig's line ``(r, k)`` is a COHERENT core of mean power
``C = 10^{profile_db/10}`` at ``s = 1`` times a unit-mean-amplitude log-OU
envelope of variance ``s2_rk``, so its TOTAL power is ``T = C e^{s2_rk}`` and
its lag law ``T exp(s2_rk (rho(tau) - 1))``, ``rho = e^{-2 pi gamma_k |tau|}``.
The map (:func:`v4_from_v1`) keeps the total per rotor and line:

* the motor boost ``b_n = motor_boost_db 0.8^(n-1)`` that v1 put on the
  orders ``k = 3 p n`` becomes a v4 motor line (coherent, no pedestal) of
  power ``C (1 - 10^{-b_n/10})``, leaving the aerodynamic line its
  unboosted total ``T 10^{-b_n/10}`` (no motor line when ``b_n <= 0``);
* the amplitude pedestal is shared by the rotors in v4: ``s2_k = mean_r
  s2_rk`` capped at the bank cap of the order's group (:data:`AMP_CAPS`,
  above the fits' :data:`.v4_priors.AMP_GROUPS`), at v1's rate;
* the phase pedestal is drawn per order around the bench's direct
  measurement (:data:`PHASE_GROUPS`, :func:`draw_phase_pedestal`), with no
  floor guard; :func:`v4_from_v1` alone puts v1's excess over the cap there
  at the same rate (the exact lag law of the v1 rig), which the draw replaces;
* the shaft (``sigma_nu``, ``lam``) is v1's flat law; the floor is v1's
  spline, level and exponent under the one-floor rule (``(mean_r s_r)^b``
  where v1 had ``mean_r s_r^b``: equal at equal speeds);
* the mics are unit capsules (``array_geometry`` with no array response and
  zero geometry); v1's per-clip line scatter ``mic_dev_sd_db`` is the render
  option ``line_scatter_db`` of the ``noise_v2`` pool, not a payload term;
* ``alpha_motor`` (v1 had no motor law) is drawn per rig;
* the rig carries NO wind (Dmitrii, 2026-10-25: ``params.wind`` None): v1
  drew its wind per clip, and that per-clip ``wind_sc`` law is the render
  option ``wind_sc`` of the ``noise_v2`` pool, a render-time augmentation.

A draw is reproducible from ``(seed, worker, attempt)`` alone
(:func:`sample_prior_v4`); the payload is assembled by
:func:`data_processing.noise_model.prior_bank_v4.rig_payload`, the same
function that reads a rig back out of a ``noise-prior-bank-v4/2`` file.
:func:`convert_prior_v1` maps one rig of the v1 bank the same way, for
comparison with what the prior_v1 arms trained on.
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
    "AMP_CAPS",
    "BAND_HZ",
    "GUARD_DB",
    "GUARD_RPS",
    "PHASE_GROUPS",
    "PRIOR",
    "RULES",
    "RULES_VERSION",
    "SEED",
    "SOURCE_BANK",
    "amp_cap",
    "convert_prior_v1",
    "draw_phase_pedestal",
    "guard_phase_pedestal",
    "pedestal_skirt_density",
    "pedestal_skirt_margin_db",
    "phase_gamma_hz",
    "prior_v4_const",
    "sample_prior_v4",
    "v4_from_v1",
    "with_phase_pedestal",
]

RULES_VERSION = "prior_v4/4"
#: What a bank rig is a draw of (its ``_prior_v4.prior``).
PRIOR = "drone_prior.sample_prior (PriorSpec defaults) -> prior_v4 RULES -> phase pedestal"
#: The v1 bank the prior_v1 arms trained on (``conf/online_mix/noise_prior_v1_5050.yaml``):
#: :func:`convert_prior_v1` maps its rigs for comparison.
SOURCE_BANK = (
    "dload:prior-v1-banks@edbe77d9b3444460edd24f76d189dd2aef624a8bf13f37f87fcb979bed3515db"
    "/prior_v1_n16384.npz"
)
#: The base seed of the bank's draws (:func:`sample_prior_v4`) and of the
#: converted rigs' ``alpha_motor``.
SEED = 20261024
#: The MD2 flight payload's (``results/noise_v4/flight/md2_fly125__flight_v4.json``).
BAND_HZ = (31.25, 7898.4375)
#: The bank's amplitude-pedestal cap per order group ``(k_lo, k_hi, cap)``
#: (Dmitrii, 2026-10-25): the larger of the fits' cap
#: (:data:`.v4_priors.AMP_GROUPS`, from the 0.25 s-frame
#: ``amplitude_jitter.json``, blind to flicker above ~2 Hz) and the p90 of
#: the demodulated bench amplitude variance of step A0
#: (``results/noise_v4/phase_width/raw/*.json``, every non-motor order with an
#: amplitude fit: 0.052 / 0.613 / 0.228 / 0.205 / 0.226), to 2 decimals. The
#: fits keep their own caps.
AMP_CAPS = (
    (1, 3, 0.075),
    (4, 8, 0.61),
    (9, 20, 0.24),
    (21, 40, 0.21),
    (41, 10**9, 0.23),
)

RULES: dict[str, str] = {
    "version": RULES_VERSION,
    "decided": "Dmitrii, 2026-10-24; amplitude caps AMP_CAPS, no wind in the payload, the "
    "phase pedestal without floor guard, fresh draws: 2026-10-25",
    "draw": "per draw SeedSequence([seed, worker, attempt]).spawn(3): v1 laws "
    "(drone_prior.sample_prior, PriorSpec defaults), alpha_motor, phase pedestal",
    "meta": "kind flight, rig wide_prior, blades and pole_pairs v1's, f_ref 80 rev/s, "
    "sr 16000, band_hz the MD2 flight payload's, front_end sr_work 32000",
    "lines": "per (r, k): C = 10^{profile_db/10}, T = C e^{s2_rk}; at k = 3 p n <= K with "
    "b = motor_boost_db 0.8^(n-1): b > 0 -> motor C (1 - 10^{-b/10}), aero T 10^{-b/10}; "
    "b <= 0 -> motor silent (-300 dB), aero T; elsewhere aero T",
    "pedestal": "s2_k = mean_r s2_rk; sigma2_amp = min(s2_k, AMP_CAPS cap of the order's group), "
    "gamma_amp_hz = am_gamma_hz",
    "phase_pedestal": "per order sigma2_phase, W log-normal about the PHASE_GROUPS medians (A0 "
    "bench), capped at v4_priors.PHASE_SIGMA2_MAX / W_MAX_HZ, gamma_phase_hz = W (1 - e^-s2) / "
    "s2; no floor guard",
    "speed_laws": "alpha_aero = amp_exp; alpha_motor ~ N(0, 2) truncated to |alpha| <= 4",
    "shaft": "eps = sigma_nu / (2 pi 80), a_sigma = 0, lam_ref_per_s = lam, b_lam = 0",
    "floor": "one_floor_mean_speed; mean_db, shape_z, shape_sd_db v1's on floor_ctrl_hz(16000); "
    "b0 = floor_exp, b1 = 0, static_rel = floor_static_rel",
    "mics": "array_geometry, array_response None, geometry_db zeros: unit capsules; v1's "
    "mic_dev_sd_db is the render option line_scatter_db",
    "wind": "none in the payload (params.wind None); v1's per-clip wind_sc law is the noise_v2 "
    "pool's render option wind_sc (render-time augmentation)",
}


def prior_v4_const(
    n_rotors: int, n_mics: int, blades: int, *, seed: int, rig: str, prior: str
) -> dict[str, Any]:
    """The part every rig of a bank shares
    (:func:`data_processing.noise_model.prior_bank_v4.bank_const`)."""
    R, M = int(n_rotors), int(n_mics)
    return {
        "kind": "flight",
        "rig": str(rig),
        "n_rotors": R,
        "n_mics": M,
        "blades": int(blades),
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
        "provenance": {"prior": str(prior), "rules": RULES_VERSION, "seed": int(seed)},
    }


def amp_cap(k: Any) -> np.ndarray:
    """The bank's amplitude-variance cap of each order ``k`` (:data:`AMP_CAPS`)."""
    k = np.asarray(k, dtype=np.int64)
    out = np.empty(k.shape, dtype=np.float64)
    for lo, hi, cap in AMP_CAPS:
        out[(k >= lo) & (k <= hi)] = cap
    return out


def _alpha_motor(rng: np.random.Generator) -> float:
    """``N(ALPHA_MOTOR)`` truncated to ``|alpha| <= ALPHA_MOTOR_MAX`` (rejection)."""
    mu, sd = VP.ALPHA_MOTOR
    while True:
        a = float(rng.normal(mu, sd))
        if abs(a) <= VP.ALPHA_MOTOR_MAX:
            return a


def v4_from_v1(
    pl: dict[str, Any],
    rng_alpha: np.random.Generator,
    *,
    seed: int,
    draw: tuple[int, int],
    rig_name: str = "wide_prior",
    prior: str = PRIOR,
) -> dict[str, Any]:
    """The v1 payload ``pl`` (:func:`.drone_prior.sample_prior`'s) as a
    ``noise-v4-fit/3`` flight payload (module docstring; :data:`RULES`), its
    phase part v1's excess over the amplitude cap; ``alpha_motor`` from
    ``rng_alpha``. ``seed``, ``draw = (worker, attempt)``, ``rig_name`` and
    ``prior`` are provenance."""
    p, d = pl["params"], pl["_prior"]["drawn"]
    spec = pl["_prior"]["spec"]
    const = prior_v4_const(
        int(pl["n_rotors"]),
        int(pl["n_mics"]),
        int(spec["blades"]),
        seed=seed,
        rig=rig_name,
        prior=prior,
    )
    R = const["n_rotors"]

    # lines: v1 core C and total T; the motor boost becomes a coherent motor line
    prof = np.asarray(p["profile"]["profile_db"], dtype=np.float64)
    s2 = np.asarray(p["am"]["sigma2"], dtype=np.float64)
    if prof.shape != (R, s2.shape[1]) or s2.shape[0] != R:
        raise ValueError(f"draw {draw}: profile {prof.shape} and am.sigma2 {s2.shape} not (R, K)")
    K = prof.shape[1]
    core = 10.0 ** (prof / 10.0)
    total = core * np.exp(s2)
    pole_pairs, period = int(d["pole_pairs"]), int(d["motor_period"])
    if period != 3 * pole_pairs:
        raise ValueError(f"draw {draw}: motor period {period} is not 3 p for p = {pole_pairs}")
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

    # pedestal: shared by the rotors, split at the bank amplitude cap, one rate
    s2_k = s2.mean(axis=0)
    s2_amp = np.minimum(s2_k, amp_cap(np.arange(1, K + 1)))
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
        "draw_worker": int(draw[0]),
        "draw_attempt": int(draw[1]),
    }
    return rig_payload(const, rig)


def convert_prior_v1(bank: PriorBank, i: int, *, seed: int) -> dict[str, Any]:
    """Rig ``i`` of the v1 ``bank`` (what the prior_v1 arms trained on) mapped
    by :func:`v4_from_v1`, ``alpha_motor`` from ``SeedSequence([seed,
    i]).spawn(1)[0]``; its draw is ``(-1, i)`` and its rig ``prior_v1``. For
    comparison only: the bank is :func:`sample_prior_v4`'s draws."""
    i = int(i)
    rng_alpha = np.random.default_rng(np.random.SeedSequence([int(seed), i]).spawn(1)[0])
    return v4_from_v1(
        bank.payload(i), rng_alpha, seed=seed, draw=(-1, i), rig_name="prior_v1", prior=SOURCE_BANK
    )


# -- the phase pedestal (Dmitrii, 2026-10-25: adopted, no floor guard) --------
#
# v1 has no fast per-line phase noise (its pedestal is the bench AMPLITUDE
# flicker read from 0.25 s frames; the phase went to one quiet shaft), so a
# mapped rig's lines are needles. The phase pedestal of every order is drawn
# around the bench's direct measurement; the floor guard below is kept for
# comparison (prior_rigs_lab.PED_GUARD) and is not part of the prior.

#: The bench's directly measured phase pedestal per order group (step A0,
#: ``results/noise_v4/phase_width/raw/*.json``: 20 DREGON single-rotor
#: recordings, carrier-tracked demodulation with the shaft removed; the
#: blade-pass orders that were detected and not at a bound, motor orders out;
#: k 1-3 pools both parities, n = 12, and borrows the width of k 4-8, its two
#: even widths being unresolved): ``(k_lo, k_hi, sigma2 median, sd of log
#: sigma2, W median (Hz), sd of log W)``. The odd orders measured wider, but
#: an odd width law lets the bench fits trade pedestal for floor
#: (:data:`.v4_priors.C_W`), so the blade-pass law holds for every order.
PHASE_GROUPS = (
    (1, 3, 0.040, 0.40, 0.79, 1.68),
    (4, 8, 0.198, 0.97, 0.79, 1.68),
    (9, 20, 0.956, 0.99, 1.61, 0.74),
    (21, 40, 1.696, 0.40, 4.87, 0.53),
    (41, 10**9, 1.665, 0.79, 8.38, 0.83),
)
#: The floor guard: with every rotor at :data:`GUARD_RPS`, the pedestal
#: skirts (amplitude and phase) at each midpoint ``(k + 1/2) f`` between two
#: lines stay :data:`GUARD_DB` under the broadband PSD there (no wind) - the
#: DREGON flight fit's own margin on the orders its bench data identify
#: (k <= 60, at most -3.3 dB). A violating midpoint shrinks the phase
#: ``sigma2`` of its two lines by :data:`GUARD_SHRINK` per step, widths kept,
#: down to :data:`GUARD_PHASE_MIN` (below it the skirt is the amplitude
#: pedestal's, v1's measured flicker, which the guard does not touch).
GUARD_RPS = float(AMP_RPS_REF)
GUARD_DB = 3.0
GUARD_SHRINK = 0.8
GUARD_PHASE_MIN = 1e-3
GUARD_MAX_STEPS = 80
#: Terms of the Lorentzian series of ``exp(s2 (rho - 1))``; ``3^n / n!`` is
#: below 1e-12 by n = 30.
_SKIRT_TERMS = 40


def _phase_group(k: np.ndarray, col: int) -> np.ndarray:
    out = np.empty(k.shape, dtype=np.float64)
    for row in PHASE_GROUPS:
        out[(k >= row[0]) & (k <= row[1])] = row[col]
    return out


def draw_phase_pedestal(rng: np.random.Generator, n_orders: int) -> tuple[np.ndarray, np.ndarray]:
    """``(sigma2, W_hz)`` per order ``k = 1..n_orders``: log-normal around the
    :data:`PHASE_GROUPS` medians, independent per order, capped at
    :data:`.v4_priors.PHASE_SIGMA2_MAX` and :data:`.v4_priors.W_MAX_HZ`."""
    k = np.arange(1, int(n_orders) + 1)
    z = rng.standard_normal((2, k.size))
    s2 = _phase_group(k, 2) * np.exp(_phase_group(k, 3) * z[0])
    w = _phase_group(k, 4) * np.exp(_phase_group(k, 5) * z[1])
    return np.minimum(s2, VP.PHASE_SIGMA2_MAX), np.minimum(w, VP.W_MAX_HZ)


def phase_gamma_hz(sigma2: np.ndarray, width_hz: np.ndarray) -> np.ndarray:
    """The OU rate ``gamma = W (1 - e^-s2) / s2`` of a pedestal of mean
    half-width ``W`` (:func:`.v4_priors.pedestal_width` inverted; ``W`` as
    ``s2 -> 0``)."""
    s2 = np.asarray(sigma2, dtype=np.float64)
    frac = np.where(s2 > 1e-12, -np.expm1(-s2) / np.maximum(s2, 1e-300), 1.0)
    return np.asarray(width_hz, dtype=np.float64) * frac


def pedestal_skirt_density(
    s2a: Any, ga: Any, s2p: Any, gp: Any, delta: Any, *, tol: float = 1e-14
) -> np.ndarray:
    """One-sided density (1/Hz) of the pedestal skirt of a unit-power line at
    offsets ``delta``: the spectrum of ``exp(s2a (rho_a - 1) + s2p (rho_p -
    1))`` without its core, ``e^-(s2a + s2p) sum_{(m, n) != (0, 0)} s2a^m / m!
    s2p^n / n! L_{m ga + n gp}(delta)``, ``L`` the unit Lorentzian (Lorentzians
    convolve by adding widths). The four pedestal arrays broadcast together
    and against ``delta``; terms whose Poisson weight is below ``tol`` on
    every line (past the mode) are dropped."""
    s2a, ga, s2p, gp = np.broadcast_arrays(
        *(np.asarray(x, dtype=np.float64) for x in (s2a, ga, s2p, gp))
    )
    delta = np.asarray(delta, dtype=np.float64)
    out = np.zeros(np.broadcast_shapes(s2a.shape, delta.shape))
    top_a, top_p = float(s2a.max(initial=0.0)), float(s2p.max(initial=0.0))
    wa = np.exp(-s2a)
    for m in range(_SKIRT_TERMS + 1):
        if m > 0:
            wa = wa * s2a / m
        if m > top_a and not np.any(wa > tol):
            break
        w = wa * np.exp(-s2p)
        for n in range(_SKIRT_TERMS + 1):
            if n > 0:
                w = w * s2p / n
            if n > top_p and not np.any(w > tol):
                break
            if m == 0 and n == 0:
                continue
            g = np.maximum(m * ga + n * gp, 1e-9)
            out += w * (g / np.pi) / (g**2 + delta**2)
    return out


def pedestal_skirt_margin_db(
    fit: dict[str, Any], sigma2_phase: np.ndarray, width_hz: np.ndarray
) -> np.ndarray:
    """``(K - 1,)``: at :data:`GUARD_RPS` on every rotor, ``10 log10`` of the
    pedestal skirts (``fit``'s amplitude pedestal and the given phase one) of
    ALL aerodynamic lines at the midpoint ``(k + 1/2) f`` over the broadband
    PSD there (no wind), mic-mean; ``-inf`` past the Nyquist. Line powers
    ``P_rk`` are variances (one-sided density ``P L``), the broadband ``F``
    reads ``2 F / sr`` (:func:`.v4.broadband_psd`). Against the exact
    :func:`.v4.expected_periodogram` (rig 13504, wind off) it reads 1.6 dB low
    in the median (0.8-3.1 dB, p5-p95): the guard is that much lenient."""
    par = V4.V4Params.from_payload(fit)
    K = par.k_max
    s = par.speed(np.full(par.n_rotors, GUARD_RPS))
    k = np.arange(1, K + 1, dtype=np.float64)
    mid = (k[:-1] + 0.5) * GUARD_RPS
    inside = mid < 0.5 * par.sr
    p_aero, _ = par.line_power(s)  # (R, K)
    gain = 10.0 ** (par.mic.db(k * GUARD_RPS) / 10.0)  # (R, M, K)
    power = np.einsum("rmk,rk->k", gain, p_aero) / par.n_mics  # (K,) mic-mean, rotors summed
    ped = par.pedestal
    s2p = np.asarray(sigma2_phase, dtype=np.float64)
    gp = phase_gamma_hz(s2p, width_hz)
    delta = mid[:, None] - k[None, :] * GUARD_RPS  # (K-1, K)
    density = pedestal_skirt_density(
        ped.sigma2_amp[None, :], ped.gamma_amp_hz[None, :], s2p[None, :], gp[None, :], delta
    )
    skirt = density @ power  # (K-1,)
    floor = 2.0 * V4.broadband_psd(par, mid, s, wind=False).mean(axis=0) / float(par.sr)
    out = np.full(K - 1, -np.inf)
    ok = inside & (skirt > 0.0)
    out[ok] = 10.0 * np.log10(skirt[ok] / floor[ok])
    return out


def guard_phase_pedestal(
    fit: dict[str, Any], sigma2: np.ndarray, width_hz: np.ndarray
) -> tuple[np.ndarray, int]:
    """``(sigma2, n_shrunk)``: the phase ``sigma2`` with the lines around every
    midpoint whose skirts break :data:`GUARD_DB` shrunk until none does (module
    constants); a midpoint whose two lines are already below
    :data:`GUARD_PHASE_MIN` is the amplitude pedestal's alone and left as it
    is. ``n_shrunk`` counts the orders that moved."""
    s2 = np.array(sigma2, dtype=np.float64)
    moved = np.zeros(s2.size, dtype=bool)
    for _ in range(GUARD_MAX_STEPS):
        live = (s2[:-1] >= GUARD_PHASE_MIN) | (s2[1:] >= GUARD_PHASE_MIN)
        bad = (pedestal_skirt_margin_db(fit, s2, width_hz) > -GUARD_DB) & live
        if not bad.any():
            return s2, int(moved.sum())
        hit = np.zeros(s2.size, dtype=bool)
        hit[:-1] |= bad
        hit[1:] |= bad
        hit &= s2 >= GUARD_PHASE_MIN
        s2[hit] *= GUARD_SHRINK
        moved |= hit
    raise RuntimeError(f"phase guard did not settle in {GUARD_MAX_STEPS} steps")


def with_phase_pedestal(
    fit: dict[str, Any], rng: np.random.Generator, *, guard: bool = False
) -> dict[str, Any]:
    """A copy of the mapped payload ``fit`` with its phase pedestal drawn:
    :func:`draw_phase_pedestal`, then (``guard``, not part of the prior)
    :func:`guard_phase_pedestal`; the amplitude pedestal, shaft and everything
    else are ``fit``'s. ``out["phase_pedestal"]`` records the rule and the
    guarded order count (``None`` without the guard)."""
    K = V4.V4Params.from_payload(fit).k_max
    s2, w = draw_phase_pedestal(rng, K)
    n_shrunk = None
    if guard:
        s2, n_shrunk = guard_phase_pedestal(fit, s2, w)
    out = dict(fit, params=dict(fit["params"]))
    out["params"]["pedestal"] = dict(
        fit["params"]["pedestal"],
        sigma2_phase=s2.tolist(),
        gamma_phase_hz=phase_gamma_hz(s2, w).tolist(),
        width_phase_hz=w.tolist(),
    )
    out["phase_pedestal"] = {
        "rule": "PHASE_GROUPS log-normal per order, capped; "
        + (f"floor guard {GUARD_DB:g} dB at {GUARD_RPS:g} rev/s" if guard else "no floor guard"),
        "guard_shrunk": n_shrunk,
    }
    return out


def sample_prior_v4(seed: int, worker: int, attempt: int) -> dict[str, Any]:
    """Draw ``attempt`` of ``worker`` from the wide prior: the v1 laws
    (:func:`.drone_prior.sample_prior`, :class:`.drone_prior.PriorSpec`
    defaults), :func:`v4_from_v1`, :func:`with_phase_pedestal` (no guard), on
    the three children of ``SeedSequence([seed, worker, attempt])``. Ungated:
    ``scripts/prior_bank_v4_build.py`` keeps the draws that pass the
    identifiability gate."""
    from .drone_prior import sample_prior  # noqa: PLC0415 (heavy module, sampler-only)

    ss = np.random.SeedSequence([int(seed), int(worker), int(attempt)]).spawn(3)
    rng_v1, rng_alpha, rng_phase = (np.random.default_rng(s) for s in ss)
    v1 = sample_prior(rng_v1)
    out = with_phase_pedestal(
        v4_from_v1(v1, rng_alpha, seed=seed, draw=(worker, attempt)), rng_phase
    )
    # the v1 laws' scalar draws, for the builder's report (not kept in the bank)
    out["_prior_v1_drawn"] = {
        k: v for k, v in v1["_prior"]["drawn"].items() if isinstance(v, (int, float))
    }
    return out
