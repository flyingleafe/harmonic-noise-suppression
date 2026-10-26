"""The prior_v1 rig bank in noise model v4 (:mod:`experiments.noise_model.prior_v4`).

Three v1 prior draws go through the real v1 bank format (float16 tables, as the
16 384-rig bank) and are converted. At the reference speed the v4 line powers
are the v1 totals with the motor boost split off as a coherent motor line; the
pedestal is the rotor mean of v1's, split at the v4 amplitude cap with one
rate; the shaft is v1's at every speed. With the rotors sharing their
pedestal (where rule 3 is exact) the v4 expected spectrum at 80 rev/s is the
v1 rig's (:mod:`.identifiability`'s line shapes and floor, the v1 reference)
away from the motor orders. The per-rig draws follow their laws and are
deterministic in the seed; a bank rig drawn from the wide prior is given back
by the draw it records.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
import torch

from data_processing.noise_model import v4 as V4
from data_processing.noise_model.prior_bank import PriorBank, write_prior_bank
from data_processing.noise_model.prior_bank_v4 import (
    SILENT_LINE_DB,
    PriorBankV4,
    write_prior_bank_v4,
)
from data_processing.noise_model.render import regime_blend_weight
from data_processing.noise_model.v3 import wind_shape_spec
from experiments.noise_model import identifiability as ID
from experiments.noise_model import v4_priors as VP
from experiments.noise_model.drone_prior import sample_prior
from experiments.noise_model.prior_v4 import (
    SEED,
    amp_cap,
    convert_prior_v1,
    sample_prior_v4,
    with_phase_pedestal,
)

SR, N_FFT, RPS = 16000, 2048, 80.0


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
    pl = convert_prior_v1(bank, i, seed=SEED)
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
    assert np.all(ped.sigma2_amp <= amp_cap(k) + 1e-15)
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
    pl = convert_prior_v1(shared, i, seed=SEED)
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
    a = convert_prior_v1(bank, 1, seed=SEED)
    assert a == convert_prior_v1(bank, 1, seed=SEED)
    draws = [convert_prior_v1(bank, 1, seed=s) for s in range(400)]
    assert all(d["params"]["wind"] is None for d in draws)  # wind is the render's wind_sc
    alpha = np.array([d["params"]["alpha_motor"] for d in draws])
    assert np.abs(alpha).max() <= VP.ALPHA_MOTOR_MAX
    # N(0, 2) truncated at 2 sd: sd 1.76
    assert abs(alpha.std() - 1.759) < 0.15 and abs(alpha.mean()) < 0.25


def _only(fit: dict[str, Any], r: int, k: int) -> dict[str, Any]:
    """``fit`` with harmonic ``(r, k)`` alone (none when ``k`` is past its
    profile): no other line, floor or wind."""
    par = V4.V4Params.from_payload(fit)
    out = copy.deepcopy(fit)
    p = out["params"]
    aero, motor = (
        np.asarray(p[key], dtype=np.float64) for key in ("profile_aero_db", "profile_motor_db")
    )
    keep_aero = np.full_like(aero, SILENT_LINE_DB)
    if k <= aero.shape[1]:
        keep_aero[r, k - 1] = aero[r, k - 1]
    keep_motor = np.full_like(motor, SILENT_LINE_DB)
    j = np.flatnonzero(par.motor_orders == k)
    keep_motor[r, j] = motor[r, j]
    p["profile_aero_db"], p["profile_motor_db"] = keep_aero.tolist(), keep_motor.tolist()
    p["floor"] = dict(p["floor"], mean_db=-10000.0, static_rel=0.0)
    p["wind"] = None
    return out


def _narrow(fit: dict[str, Any], k_max: int) -> dict[str, Any]:
    """``fit`` cut to the orders ``1 .. k_max`` (aero lines, pedestal, motor orders)."""
    out = copy.deepcopy(fit)
    p = out["params"]
    p["profile_aero_db"] = [row[:k_max] for row in p["profile_aero_db"]]
    p["pedestal"] = {
        key: (v[:k_max] if isinstance(v, list) else v) for key, v in p["pedestal"].items()
    }
    keep = [j for j, o in enumerate(p["motor_orders"]) if o <= k_max]
    p["motor_orders"] = [p["motor_orders"][j] for j in keep]
    p["profile_motor_db"] = [[row[j] for j in keep] for row in p["profile_motor_db"]]
    return out


@pytest.mark.parametrize("case", ["flat_shaft", "speed_shaft", "regime_pair"])
def test_the_v4_gate_splits_the_exact_spectrum_into_one_line_and_the_rest(
    bank: PriorBank, case: str
) -> None:
    """The identifiability gate's v4 contrasts of a converted rig with the phase
    pedestal and motor lines (the bank's gate input) are the exact
    ``v4.expected_periodogram`` of the rig WITHOUT wind at constant speed,
    split into one harmonic (aerodynamic + motor line) and everything else, up
    to the shape table: the gate ignores a payload's wind (given here a loud
    static one). The same holds for a shaft law that is not flat in speed (a
    fit's: each rotor's shaft at its own speed) and for a standby/cruise pair of
    different profile widths, blended at the frame's regime weight (an order
    the narrower regime lacks is the wider one's line alone)."""
    rng = np.random.default_rng(0)
    fit = with_phase_pedestal(convert_prior_v1(bank, 0, seed=SEED), rng, guard=False)
    loud = float(fit["params"]["floor"]["mean_db"]) + 20.0
    fit["params"]["wind"] = {
        "kind": V4.WIND_STATIC,
        "wind_db": [loud] * int(fit["n_mics"]),
        "shape": wind_shape_spec(),
    }
    par = V4.V4Params.from_payload(fit)
    assert par.wind is not None and par.motor_orders.tolist()[:2] == [21, 42]
    if case != "flat_shaft":
        # the DREGON bench fit's shaft law (a_sigma 2.11, b_lam 4.44) at 4x its scale,
        # so the shaft visibly widens the high orders and differs between the rotors
        fit["params"]["shaft"] = dict(
            fit["params"]["shaft"], eps=0.006, a_sigma=2.11, lam_ref_per_s=52.5, b_lam=4.44
        )
    speeds = np.array([55.0, 64.0, 81.0, 98.0])
    track = np.repeat(speeds[:, None], N_FFT, 1)
    weights: dict[str, tuple[float, dict[str, Any]]] = {"single": (1.0, fit)}
    if case == "regime_pair":
        other = with_phase_pedestal(convert_prior_v1(bank, 1, seed=SEED), rng, guard=False)
        w = float(regime_blend_weight(track, sr=SR)[N_FFT // 2])
        assert 0.1 < w < 0.9
        weights = {"standby": (1.0 - w, _narrow(other, 85)), "cruise": (w, fit)}
    pool = ID.frame_pool([track], n_frames=1, seed=0)
    gated = weights["single"][1] if "single" in weights else {n: f for n, (_, f) in weights.items()}
    ((fast, _),) = ID.line_contrasts(gated, pool)

    def expected(f: dict[str, Any]) -> np.ndarray:
        return V4.expected_periodogram(f, speeds, n_fft=N_FFT, sr=SR).mean(axis=0)

    total = np.sum(
        [wt * expected(dict(f, params=dict(f["params"], wind=None))) for wt, f in weights.values()],
        axis=0,
    )
    err = []
    for r, k in [(1, k) for k in (1, 2, 3, 8, 13, 21, 42, 60, 100)] + [(3, 2), (3, 21), (3, 50)]:
        b = int(np.rint(k * speeds[r] / (SR / N_FFT)))
        line = sum(wt * expected(_only(f, r, k))[b] for wt, f in weights.values())
        exact = 10.0 * np.log10(total[b] / (total[b] - line))
        err.append(abs(exact - fast[r, 0, k - 1]))
    assert max(err) < 0.05, err


def test_a_bank_rig_is_redrawn_from_the_draw_it_records(tmp_path) -> None:
    """``sample_prior_v4(seed, worker, attempt)`` is a pure function of its
    three integers, and a rig read back from a bank names the draw that gives
    it again (up to the bank's float16), phase pedestal included."""
    drawn = [sample_prior_v4(SEED, 3, a) for a in (0, 1)]
    back = PriorBankV4(write_prior_bank_v4(tmp_path / "b.npz", drawn, {"note": "test"})).payload(1)
    prov = back["_prior_v4"]
    assert (prov["draw_worker"], prov["draw_attempt"]) == (3, 1) and back["params"]["wind"] is None
    again = sample_prior_v4(prov["seed"], prov["draw_worker"], prov["draw_attempt"])["params"]
    np.testing.assert_allclose(
        back["params"]["profile_aero_db"], again["profile_aero_db"], atol=0.04
    )
    for key in ("sigma2_amp", "sigma2_phase", "gamma_phase_hz"):
        np.testing.assert_allclose(
            back["params"]["pedestal"][key], again["pedestal"][key], rtol=1e-3
        )
    assert not np.allclose(drawn[0]["params"]["profile_aero_db"], again["profile_aero_db"])
