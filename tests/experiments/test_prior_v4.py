"""The prior_v1 rig bank in noise model v4 (:mod:`experiments.noise_model.prior_v4`).

Three v1 prior draws go through the real v1 bank format (float16 tables, as the
16 384-rig bank) and are converted. At the reference speed the v4 line powers
are the v1 totals with the motor boost split off as a coherent motor line; the
pedestal is the rotor mean of v1's, split at the v4 amplitude cap with one
rate; the shaft is v1's at every speed. With the rotors sharing their
pedestal (where rule 3 is exact) the v4 expected spectrum at 80 rev/s is the
v1 rig's (:mod:`.identifiability`'s line shapes and floor, the v1 reference)
away from the motor orders. The per-rig draws follow their laws and are
deterministic in the seed.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import pytest
import torch

from data_processing.noise_model import v4 as V4
from data_processing.noise_model.prior_bank import PriorBank, write_prior_bank
from data_processing.noise_model.prior_bank_v4 import SILENT_LINE_DB
from experiments.noise_model import identifiability as ID
from experiments.noise_model import v4_priors as VP
from experiments.noise_model.drone_prior import sample_prior
from experiments.noise_model.prior_v4 import SEED, convert_prior_v1, floor_band_db

SR, N_FFT, RPS = 16000, 2048, 80.0
#: A flat stand-in for the bank's measured SC shape: the conversion only
#: passes it through and reads its frequency grid for the 20-100 Hz level.
SHAPE: dict[str, Any] = {
    "freqs_hz": np.fft.rfftfreq(4096, 1.0 / SR).tolist(),
    "psd": np.ones(2049).tolist(),
    "ref_band_hz": [20.0, 100.0],
    "unit_psd_ref": 1.0,
}


@pytest.fixture(scope="module")
def bank(tmp_path_factory) -> PriorBank:
    """Three v1 draws (motor boost 8.8, 5.6 and 0.2 dB; periods 21, 24, 18),
    the last given a NEGATIVE drawn boost: its motor orders keep no motor line."""
    rng = np.random.default_rng(0)
    rigs = [sample_prior(rng) for _ in range(3)]
    for i, r in enumerate(rigs):
        r["_prior"]["seed"] = 100 + i
    path = write_prior_bank(tmp_path_factory.mktemp("v1") / "b.npz", rigs, {"note": "test"})
    b = PriorBank(path)
    b.arrays["motor_boost_db"][2] = -2.0
    return b


def _shared(bank: PriorBank) -> PriorBank:
    """``bank`` with every rig's AM variance replaced by its rotor mean."""
    out = bank.with_offset(0.0)
    out.arrays = dict(bank.arrays)
    s2 = bank.arrays["am_sigma2"].astype(np.float64)
    out.arrays["am_sigma2"] = np.repeat(s2.mean(axis=1, keepdims=True), s2.shape[1], 1).astype(
        np.float16
    )
    return out


@pytest.mark.parametrize("i", [0, 1, 2])
def test_line_powers_pedestal_and_shaft_follow_the_rules(bank: PriorBank, i: int) -> None:
    v1 = bank.payload(i)
    pl = convert_prior_v1(bank, i, shape=SHAPE, seed=SEED)
    par = V4.V4Params.from_payload(pl)
    p, d = v1["params"], v1["_prior"]["drawn"]
    core = 10.0 ** (np.asarray(p["profile"]["profile_db"]) / 10.0)  # (R, K)
    s2 = np.asarray(p["am"]["sigma2"])
    total = core * np.exp(s2)
    R, K = core.shape
    k = np.arange(1, K + 1)

    period = int(d["motor_period"])
    assert par.motor_orders.tolist() == list(range(period, K + 1, period))
    n = par.motor_orders // period
    b = float(d["motor_boost_db"]) * 0.8 ** (n - 1)
    keep = np.where(b > 0.0, 10.0 ** (-b / 10.0), 1.0)  # (n_m,)
    rows = par.motor_orders - 1

    aero, motor = par.line_power(np.ones(R))
    # at s = 1: v1's total everywhere but the motor orders, where the motor
    # line's coherent share leaves the aerodynamic line with the unboosted total
    off = np.setdiff1d(k - 1, rows)
    np.testing.assert_allclose(aero[:, off], total[:, off], rtol=1e-12)
    pedestal_share = (total[:, rows] - core[:, rows]) * (1.0 - keep)
    np.testing.assert_allclose(aero[:, rows] + motor, total[:, rows] - pedestal_share, rtol=1e-12)
    # the motor line carries no pedestal: it is the boost's share of the core
    s2_k = s2.mean(axis=0)
    coherent = motor + aero[:, rows] * np.exp(-s2_k[rows])
    want = core[:, rows] * (1.0 - keep + keep * np.exp(s2[:, rows] - s2_k[rows]))
    np.testing.assert_allclose(coherent, want, rtol=1e-12)
    if float(d["motor_boost_db"]) <= 0.0:
        assert np.all(par.profile_motor_db == SILENT_LINE_DB)

    ped = par.pedestal
    assert np.all(ped.sigma2_amp <= VP.amp_cap(k) + 1e-15)
    np.testing.assert_allclose(ped.sigma2_amp + ped.sigma2_phase, s2_k, rtol=1e-12)
    np.testing.assert_allclose(ped.gamma_amp_hz, np.asarray(p["am"]["gamma_hz"])[0])
    np.testing.assert_array_equal(ped.gamma_phase_hz, ped.gamma_amp_hz)
    assert par.alpha_aero == float(p["profile"]["amp_exp"])

    s = np.array([0.1, 0.5, 1.0, 1.4, 3.0])
    np.testing.assert_allclose(par.shaft.sigma_nu(s), float(p["sigma_nu"]), rtol=1e-12)
    np.testing.assert_allclose(par.shaft.lam(s), float(p["lam"]), rtol=1e-12)
    assert par.floor.rule == V4.FLOOR_RULE_ONE and par.floor.b1 == 0.0
    assert par.floor.b0 == float(p["floor"]["floor_exp"])
    assert np.all(par.mic.db(np.array([100.0, 4000.0])) == 0.0)


def _v1_expected(fit: dict[str, Any]) -> np.ndarray:
    """The v1 rig's mic-mean expected periodogram at constant :data:`RPS`
    (:mod:`.identifiability`'s line shapes, AM envelope included, and floor on
    the 32 kHz work grid), with the render chain's transfer divided out."""
    pl = ID._payload(fit, n_fft=N_FFT, hop=N_FFT // 4, sr=SR)
    sh = ID.line_shapes(pl, step_hz=0.1)
    R = sh.shape.shape[0]
    rps = np.full((R, 2 * N_FFT), RPS)
    with torch.no_grad():
        out = ID._floor_obs(pl, rps, np.array([0])).numpy()[0]
    kr = ID._k_full(fit, rps, SR)
    nu = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    k = np.arange(1, kr + 1)
    flat = sh.shape[:, :kr].reshape(R * kr, -1)
    amp = sh.level[:, :kr] * sh.speed_law(np.full((R, 1), RPS))
    obs = pl.line_obs.numpy()
    sel = nu < 7800.0
    for r in range(R):
        q = np.broadcast_to(r * kr + np.arange(kr), (int(sel.sum()), kr))
        h = ID._shape_at(flat, q, nu[sel][:, None] - k * RPS, sh.step_hz)
        h += ID._shape_at(flat, q, nu[sel][:, None] + k * RPS, sh.step_hz)
        out[sel] += (h * amp[r]).sum(axis=1) * obs[r, sel]
    return out / np.asarray(pl.grid.transfer_power)


def _v4_expected(fit: dict[str, Any]) -> np.ndarray:
    """``v4.expected_periodogram`` at constant :data:`RPS`, mic mean, with the
    lines' expectation taken on a 32 kHz grid (twice the window, same
    duration; half the reading) so that, as on v1's work grid and in the
    render, their Lorentzian pedestal tails do not fold at 8 kHz."""
    R = int(fit["n_rotors"])
    p = fit["params"]
    orders = (np.arange(1, len(p["profile_aero_db"][0]) + 1), np.asarray(p["motor_orders"]))
    lines, floor = copy.deepcopy(fit), copy.deepcopy(fit)
    lines["params"]["floor"]["mean_db"] = -300.0
    for key, k in zip(("profile_aero_db", "profile_motor_db"), orders, strict=True):
        # v1's lines stop under 8 kHz; the 32 kHz grid would take the rest
        lines["params"][key] = np.where(k * RPS < SR / 2, p[key], -300.0).tolist()
        floor["params"][key] = np.full(np.shape(p[key]), -300.0).tolist()
    e_lines = V4.expected_periodogram(lines, np.full(R, RPS), n_fft=2 * N_FFT, sr=2 * SR)
    e_floor = V4.expected_periodogram(floor, np.full(R, RPS), n_fft=N_FFT, sr=SR)
    return 0.5 * e_lines.mean(axis=0)[: N_FFT // 2 + 1] + e_floor.mean(axis=0)


@pytest.mark.parametrize("i", [0, 2])
def test_expected_spectrum_at_80_rev_s_is_v1s_away_from_motor_orders(bank: PriorBank, i: int):
    shared = _shared(bank)
    v1 = shared.payload(i)
    pl = convert_prior_v1(shared, i, shape=SHAPE, seed=SEED)
    pl["params"]["wind"] = None
    err = 10.0 * np.log10(_v4_expected(pl) / _v1_expected(v1))
    nu = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    orders = np.asarray(pl["params"]["motor_orders"])
    if np.any(np.asarray(pl["params"]["profile_motor_db"]) > SILENT_LINE_DB):
        # the motor lines lost v1's pedestal share: excluded within two orders
        near = np.any(np.abs(nu[:, None] - orders * RPS) < 2.0 * RPS, axis=1)
    else:
        near = np.zeros(nu.size, bool)  # no motor line: equal everywhere
    band = (nu >= 40.0) & (nu < 7800.0) & ~near
    worst = np.abs(err[band])
    assert worst.max() < 0.1, (worst.max(), nu[band][np.argmax(worst)])


def test_per_rig_draws_follow_their_laws_and_the_seed(bank: PriorBank) -> None:
    a = convert_prior_v1(bank, 1, shape=SHAPE, seed=SEED)
    assert a == convert_prior_v1(bank, 1, shape=SHAPE, seed=SEED)
    draws = [convert_prior_v1(bank, 1, shape=SHAPE, seed=s) for s in range(400)]
    alpha = np.array([d["params"]["alpha_motor"] for d in draws])
    assert np.abs(alpha).max() <= VP.ALPHA_MOTOR_MAX
    # N(0, 2) truncated at 2 sd: sd 1.76
    assert abs(alpha.std() - 1.759) < 0.15 and abs(alpha.mean()) < 0.25
    windy = [d for d in draws if d["params"]["wind"] is not None]
    assert abs(len(windy) / len(draws) - 0.7) < 0.07  # v1's p_wind
    ref = floor_band_db(a, SHAPE["freqs_hz"])
    levels = np.concatenate([np.asarray(d["params"]["wind"]["level_db"]) for d in windy])
    on = levels > V4.WIND_OFF_DB
    assert np.all(levels[~on] == V4.WIND_OFF_DB)
    # live capsules: 33 u - Exp(8) dB over the floor's 20-100 Hz, above -10
    assert np.all((levels[on] > ref - 10.0) & (levels[on] <= ref + 33.0))
    assert 0.8 < on.mean() < 1.0 and math.isclose(np.median(levels[on]) - ref, 11.0, abs_tol=2.0)
