"""The v4 renderer reads back as the v4 model (``docs/explainers/noise-model-v4.qmd``).

One 20 s constant-speed render of a two-rotor, two-mic bench-kind payload with
every v4 term live — an amplitude AND a phase pedestal, motor lines at
``k = 3 p n``, a frequency-dependent floor exponent (``b1 != 0``) and a
frequency-dependent pair EQ — must match :func:`v4.expected_periodogram` in
octave bands and in a band above the last line where only the two rotors'
floors are, and the log-amplitude flicker of a high order must be the
payload's ``sigma2_amp``. A renderer that drew the pedestal as amplitude, kept
the mean amplitude instead of the mean power, averaged instead of summed the
rotor floors, or dropped the floor's ``b1`` fails one of the three. The same
holds under the flight rule ``one_floor_mean_speed`` (ONE floor on the
rotor-mean speed, no pair EQ on it).

The aerodynamic lines at the motor orders are silent: an aerodynamic and a
motor line of one order share the carrier and keep their start phases for the
whole clip, so one clip carries their cross term ``2 sqrt(P_a P_m)
e^{-sigma2_phase/2} cos(phase difference)``, which only the expectation over
clips removes.

A flight payload with the STATIC per-mic wind (``static_per_mic``) renders and
reads exactly as it did before the typed wind block and the ``sc`` kind: a
golden of its render samples and expected periodogram. An ``sc`` one renders
exactly as before the ``line_scatter_db`` option and :data:`.v4.WIND_OFF_DB`
existed, at the option's default and at an explicit zero; a capsule at the
off level draws no wind. ``line_scatter_db > 0`` moves every (mic, line) gain
of a render by its own static ``N(0, sd^2)`` dB, the shaft, envelopes and
phases being the same draw. ``wind_sc`` adds v1's per-clip SC wind over each
capsule's rendered floor.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.signal import firwin, oaconvolve

from data_processing.noise_model import FIT_SCHEMA_V4
from data_processing.noise_model.render import render_noise, render_v4_wind
from data_processing.noise_model.spectrum import floor_ctrl_hz
from data_processing.noise_model.v4 import (
    WIND_OFF_DB,
    V4Params,
    broadband_periodogram,
    expected_periodogram,
    hann_window,
    rotor_line_periodograms,
)

SR = 16000
K = 90
#: Both rotors below f_ref, so the floor's s^{b(f)} depends on b1.
RPS = (72.0, 64.0)
MOTOR_ORDERS = (21, 42)
#: No line above 3.3 kHz: 4.5-6 kHz is the two rotors' floors alone, but for
#: the pedestals' Lorentzian tails from 1.2 kHz away (2 % of it, -0.08 dB):
#: the renderer's 1 kHz envelope grid folds them back within 500 Hz of a line.
LINE_TOP_HZ = 3300.0
FLOOR_BAND_HZ = (4500.0, 6000.0)
#: The flicker is read on order 40 of each rotor, lifted 12 dB so the floor
#: inside the demodulation band does not inflate its log-amplitude variance.
#: Each rotor is silent within 150 Hz of the OTHER rotor's order-40 line, so the
#: nearest lines to either are its own neighbours, 64-72 Hz away.
K_FLICKER = 40
FLICKER_LIFT_DB = 12.0
QUIET_HALF_HZ = 150.0
SIGMA2_AMP_HI = 0.2


def _payload(floor_rule: str | None = None) -> dict[str, Any]:
    k = np.arange(1, K + 1)
    base = -20.0 - 18.0 * np.log10(k) - 6.0 * (k % 2)
    base[np.asarray(MOTOR_ORDERS) - 1] = -150.0
    profile = np.stack([base, base - 2.0])
    profile[:, K_FLICKER - 1] += FLICKER_LIFT_DB
    for r, other in ((0, 1), (1, 0)):
        quiet = np.abs(k * RPS[r] - K_FLICKER * RPS[other]) < QUIET_HALF_HZ
        profile[r, quiet | (k * RPS[r] > LINE_TOP_HZ)] = -150.0
    rng = np.random.default_rng(0)
    eq_f = np.array([100.0, 500.0, 1000.0, 2000.0, 4000.0, 7000.0])
    eq = rng.normal(0.0, 1.5, size=(2, 2, eq_f.size))
    eq -= eq.mean(axis=1, keepdims=True)
    return {
        "schema": FIT_SCHEMA_V4,
        "kind": "bench",
        "rig": "test",
        "n_rotors": 2,
        "n_mics": 2,
        "blades": 2,
        "pole_pairs": 7,
        "f_ref_rev_s": 80.0,
        "sr": SR,
        "band_hz": [30.0, 7200.0],
        "params": {
            "profile_aero_db": profile.tolist(),
            "motor_orders": list(MOTOR_ORDERS),
            "profile_motor_db": [[-42.0, -50.0], [-44.0, -52.0]],
            "alpha_aero": 5.5,
            "alpha_motor": 0.0,
            "pedestal": {
                "sigma2_amp": np.where(k <= 8, 0.02, SIGMA2_AMP_HI).tolist(),
                "gamma_amp_hz": np.full(K, 3.0).tolist(),
                "sigma2_phase": np.where(k <= 8, 0.05, 1.0).tolist(),
                "gamma_phase_hz": np.full(K, 1.0).tolist(),
            },
            "shaft": {
                "eps": 0.001,
                "a_sigma": 1.0,
                "lam_ref_per_s": 12.0,
                "b_lam": 1.0,
                "f_ref_rev_s": 80.0,
                "pinned": [],
            },
            "floor": {
                "mean_db": -45.0,
                "shape_z": rng.normal(0.0, 1.0, size=14).tolist(),
                "shape_sd_db": 3.0,
                "ctrl_hz": floor_ctrl_hz(SR).tolist(),
                "b0": 3.0,
                "b1": 0.5,
                "static_rel": 0.0,
                **({} if floor_rule is None else {"floor_rule": floor_rule}),
            },
            "mic_response": {
                "kind": "pair_eq",
                "pair_eq": {"freqs_hz": eq_f.tolist(), "db": eq.tolist()},
            },
            "wind": None,
        },
    }


def _mean_periodogram(x: np.ndarray, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    w = hann_window(n_fft)
    idx = np.arange((x.shape[-1] - n_fft) // hop + 1)[:, None] * hop + np.arange(n_fft)
    spec = np.abs(np.fft.rfft(x[:, idx] * w, axis=-1)) ** 2 / np.sum(w**2)
    return spec.mean(axis=1)


@pytest.mark.parametrize("floor_rule", [None, "one_floor_mean_speed"])
def test_v4_render_matches_its_expected_periodogram_and_amplitude_pedestal(
    floor_rule: str | None,
):
    fit = _payload(floor_rule)
    rps = np.repeat(np.asarray(RPS)[:, None], 20 * SR, axis=1)
    x = render_noise(fit, rps, sr=SR, n_mics=2, seed=7)

    got = _mean_periodogram(x)
    want = expected_periodogram(fit, np.asarray(RPS), n_fft=2048, sr=SR)
    f = np.fft.rfftfreq(2048, 1.0 / SR)
    for lo in (100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0):
        band = (f >= lo) & (f < min(2.0 * lo, 6000.0))
        err_db = 10.0 * np.log10(got[:, band].mean(axis=1) / want[:, band].mean(axis=1))
        assert np.all(np.abs(err_db) < 0.5), (lo, err_db)
    # the floors alone: ~190 bins x 620 frames, so a tighter bound (dropping
    # b1 moves this band 0.45 dB, averaging the rotor floors 2.9 dB)
    band = (f >= FLOOR_BAND_HZ[0]) & (f < FLOOR_BAND_HZ[1])
    err_db = 10.0 * np.log10(got[:, band].mean(axis=1) / want[:, band].mean(axis=1))
    assert np.all(np.abs(err_db) < 0.25), err_db

    # the log-amplitude of order 40 of each rotor (two independent amplitude
    # OUs), demodulated with a +-35 Hz low-pass (the nearest lines are 64-72 Hz
    # away; the 3 Hz Lorentzian of the OU keeps 95 % of its variance inside)
    # and read every 10 ms
    t = np.arange(x.shape[1]) / SR
    lp = firwin(4001, 35.0, fs=SR)
    var = []
    for f_r in RPS:
        z = oaconvolve(x[0] * np.exp(-2j * np.pi * K_FLICKER * f_r * t), lp, mode="same")
        var.append(np.var(np.log(np.abs(z[SR : -SR : SR // 100]))))
    assert abs(np.mean(var) / SIGMA2_AMP_HI - 1.0) < 0.25, var


def _static_wind_flight_payload() -> dict[str, Any]:
    """A small two-rotor, two-mic FLIGHT payload with v3's static per-mic wind."""
    k = np.arange(1, 13)
    base = -20.0 - 18.0 * np.log10(k) - 6.0 * (k % 2)
    rng = np.random.default_rng(0)
    eq_f = np.array([100.0, 500.0, 1000.0, 2000.0, 4000.0, 7000.0])
    eq = rng.normal(0.0, 1.5, size=(2, 2, eq_f.size))
    eq -= eq.mean(axis=1, keepdims=True)
    fit = _payload()
    fit["kind"] = "flight"
    fit["params"].update(
        profile_aero_db=np.stack([base, base - 2.0]).tolist(),
        motor_orders=[21],
        profile_motor_db=[[-42.0], [-44.0]],
        pedestal={
            "sigma2_amp": np.full(12, 0.02).tolist(),
            "gamma_amp_hz": np.full(12, 3.0).tolist(),
            "sigma2_phase": np.full(12, 0.05).tolist(),
            "gamma_phase_hz": np.full(12, 1.0).tolist(),
        },
        mic_response={"kind": "pair_eq", "pair_eq": {"freqs_hz": eq_f.tolist(), "db": eq.tolist()}},
        wind={"kind": "static_per_mic", "wind_db": [-20.0, -26.0]},
    )
    fit["params"]["floor"].update(shape_z=rng.normal(0.0, 1.0, size=14).tolist(), static_rel=0.01)
    return fit


#: ``render_noise(_static_wind_flight_payload(), 0.5 s at (72, 64) rev/s, seed=3)[:, ::997]``
#: and ``expected_periodogram(., (72, 64), n_fft=256)[:, 1:40:6]`` from the renderer as it was
#: BEFORE the ``sc`` wind kind existed (the static wind dominates the samples).
STATIC_WIND_GOLDEN = np.array(
    [
        [-0.01624978686105636, -0.09185363978496798, -0.0299637979298215, -0.08379446059770775,
         -0.0889859086130272, -0.16372149182278684, -0.05468641711615371, -0.12944735763036969,
         -0.0537935919666867],
        [-0.1184129194458974, -0.29015012575494303, -0.17121666345328493, -0.19350894392100268,
         -0.1576983348959862, -0.13094573719700184, -0.01270186104491509, -0.0808193581612045,
         0.04625597100499262],
    ]
)  # fmt: skip
STATIC_WIND_GOLDEN_E = np.array(
    [
        [1.4467517651937845e-01, 2.0187828052768184e-02, 4.9552153331956636e-03,
         2.7547154313054372e-05, 2.8475387962512960e-03, 2.5468257181419279e-05,
         2.6420186436482474e-05],
        [2.3778386460169457e-01, 2.9826164569598627e-02, 4.6614661790552025e-03,
         2.8064373681837363e-05, 1.7912476509030374e-03, 2.6330191031333808e-05,
         2.7036750966798380e-05],
    ]
)  # fmt: skip


def test_static_per_mic_wind_renders_as_before() -> None:
    """A ``static_per_mic`` flight payload (the flight stage's wind kind, and
    every payload written before the ``sc`` kind) renders and reads as before."""
    fit = _static_wind_flight_payload()
    rps = np.repeat(np.asarray(RPS)[:, None], SR // 2, axis=1)
    x = render_noise(fit, rps, sr=SR, n_mics=2, seed=3)
    np.testing.assert_allclose(x[:, ::997], STATIC_WIND_GOLDEN, rtol=1e-12, atol=1e-15)
    e = expected_periodogram(fit, np.asarray(RPS), n_fft=256, sr=SR)
    np.testing.assert_allclose(e[:, 1:40:6], STATIC_WIND_GOLDEN_E, rtol=1e-12)


def test_rotor_lines_and_broadband_add_up_to_the_expected_periodogram() -> None:
    """Each rotor's lines alone plus the one broadband part (floor, static share
    and wind) are the expected periodogram, bin for bin: nothing is dropped or
    counted twice when a spectrum is split per rotor."""
    fit = _static_wind_flight_payload()
    par = V4Params.from_payload(fit)
    lines = rotor_line_periodograms(par, np.asarray(RPS), n_fft=512, sr=SR)
    assert lines.shape == (par.n_rotors, par.n_mics, 257)
    broad = broadband_periodogram(par, np.asarray(RPS), n_fft=512, sr=SR)
    np.testing.assert_allclose(
        lines.sum(axis=0) + broad,
        expected_periodogram(fit, np.asarray(RPS), n_fft=512, sr=SR),
        rtol=1e-10,
    )


def _sc_wind_flight_payload(level_db: list[float]) -> dict[str, Any]:
    """The static-wind flight payload with an ``sc`` wind on a flat shape."""
    fit = _static_wind_flight_payload()
    fit["params"]["wind"] = {
        "kind": "sc",
        "level_db": list(level_db),
        "speed_offset_ms": 0.0,
        "speed_scale_ms": 2.0,
        "gust_rate_hz": 0.75,
        "shape": {"freqs_hz": [0.0, 8000.0], "psd": [1.0, 1.0], "unit_psd_ref": 1.0},
    }
    return fit


#: ``render_noise(_sc_wind_flight_payload([-30, -36]), 0.5 s at (72, 64) rev/s,
#: seed=3)[:, ::997]`` from the renderer as it was BEFORE ``line_scatter_db`` and
#: ``WIND_OFF_DB`` existed.
SC_WIND_GOLDEN = np.array(
    [
        [-0.003863947118281391, -0.11101616471591923, -0.05442338062546269, -0.06943848402120668,
         -0.09744564199817274, -0.14721465575408388, -0.03293016420782411, -0.11388083415470701,
         -0.045317625274431075],
        [-0.12918205163155005, -0.2970618244909733, -0.15558180313799638, -0.19613154302765187,
         -0.139198091557284, -0.1334575683597426, -0.022272399884021295, -0.09327992732458082,
         0.04594166902594644],
    ]
)  # fmt: skip


def test_sc_wind_render_is_unchanged_without_line_scatter() -> None:
    fit = _sc_wind_flight_payload([-30.0, -36.0])
    rps = np.repeat(np.asarray(RPS)[:, None], SR // 2, axis=1)
    x = render_noise(fit, rps, sr=SR, n_mics=2, seed=3)
    np.testing.assert_allclose(x[:, ::997], SC_WIND_GOLDEN, rtol=1e-12, atol=1e-15)
    x0 = render_noise(fit, rps, sr=SR, n_mics=2, seed=3, line_scatter_db=0.0)
    np.testing.assert_array_equal(x0, x)


def test_a_capsule_at_the_off_level_carries_no_wind() -> None:
    wind = render_v4_wind(_sc_wind_flight_payload([WIND_OFF_DB, -30.0]), SR // 2, sr=SR, n_mics=2)
    assert np.all(wind[0] == 0.0) and np.sqrt(np.mean(wind[1] ** 2)) > 1e-3


def test_line_scatter_moves_each_line_gain_by_its_own_static_draw() -> None:
    """One rotor of pure lines (no pedestal, no shaft noise, no floor): the
    same seed with and without the scatter draws the same carriers, so the
    Hann-weighted projection of a mic on a line's frequency changes by that
    (mic, line)'s gain alone. 4 mics x 60 lines (58 aerodynamic, 2 motor)."""
    fit = _payload()
    k_max, orders = 60, np.asarray(MOTOR_ORDERS)
    base = -20.0 - 12.0 * np.log10(np.arange(1, k_max + 1))
    base[orders - 1] = -150.0  # the motor lines alone at their orders
    eq = np.zeros((1, 4, 2))
    fit.update(n_rotors=1, n_mics=4)
    fit["params"].update(
        profile_aero_db=[base.tolist()],
        profile_motor_db=[[-30.0, -36.0]],
        pedestal={
            key: np.zeros(k_max).tolist()
            for key in ("sigma2_amp", "gamma_amp_hz", "sigma2_phase", "gamma_phase_hz")
        },
        mic_response={
            "kind": "pair_eq",
            "pair_eq": {"freqs_hz": [100.0, 7000.0], "db": eq.tolist()},
        },
    )
    fit["params"]["shaft"]["eps"] = 0.0
    fit["params"]["floor"]["mean_db"] = -200.0
    f_r, n = 72.0, 2 * SR
    rps = np.full((1, n), f_r)
    plain = render_noise(fit, rps, sr=SR, n_mics=4, seed=11)
    scat = render_noise(fit, rps, sr=SR, n_mics=4, seed=11, line_scatter_db=3.0)
    t, w = np.arange(n) / SR, hann_window(n)
    k = np.arange(1, k_max + 1)  # at 21 and 42 the motor line (the aero one is silent)
    probe = w * np.exp(-2j * np.pi * f_r * k[:, None] * t)  # (L, n)
    delta = 20.0 * np.log10(np.abs((scat @ probe.T) / (plain @ probe.T)))  # (M, L)
    assert delta.shape == (4, 60)
    assert abs(delta.std() / 3.0 - 1.0) < 0.12 and abs(delta.mean()) < 0.5
    # per line, not per mic or rotor: every capsule's lines scatter about it,
    # and the same line moves differently on every capsule
    assert np.all(np.abs(np.std(delta, axis=1) / 3.0 - 1.0) < 0.27)
    assert np.all(np.std(delta, axis=0) > 0.0)


def test_line_scatter_is_refused_for_a_v3_payload() -> None:
    fit = {"schema": "noise-v3-fit/1", "params": {}}
    with pytest.raises(ValueError, match="line_scatter_db"):
        render_noise(fit, np.full((1, SR // 4), 80.0), sr=SR, n_mics=1, line_scatter_db=1.0)


def test_wind_sc_draws_v1s_per_clip_wind_over_each_capsules_rendered_floor() -> None:
    """``wind_sc`` on a windless flight payload adds v1's SC wind. With
    ``p_wind`` 1 and a vanishing shield, every capsule's wind is ``30 u`` dB
    over its OWN rendered 20-100 Hz floor (the realisation, not the
    expectation), one ``u`` per clip: the capsules agree, two clips differ. The
    lines are silent, so a render without the option is the floor alone and
    the wind is the difference (independent streams, linear after the sum). A
    payload with its own wind refuses the option."""
    fit = _static_wind_flight_payload()
    fit["params"].update(
        wind=None,
        profile_aero_db=np.full((2, 12), -300.0).tolist(),
        profile_motor_db=[[-300.0], [-300.0]],
    )
    law = {"p_wind": 1.0, "max_over_floor_db": 30.0, "shield_mean_db": 1e-9, "gustiness": 3}
    rps = np.repeat(np.asarray(RPS)[:, None], SR, axis=1)
    f = np.fft.rfftfreq(SR, 1.0 / SR)
    band = (f >= 20.0) & (f <= 100.0)

    def band_power(x: np.ndarray) -> np.ndarray:
        return np.mean(np.abs(np.fft.rfft(x, axis=1)[:, band]) ** 2, axis=1)

    excess = []
    for seed in (3, 4):
        floor = render_noise(fit, rps, sr=SR, n_mics=2, seed=seed)
        wind = render_noise(fit, rps, sr=SR, n_mics=2, seed=seed, wind_sc=law) - floor
        excess.append(10.0 * np.log10(band_power(wind) / band_power(floor)))
    db = np.asarray(excess)  # (clip, mic)
    assert np.ptp(db, axis=1).max() < 0.05, db
    assert abs(db[0, 0] - db[1, 0]) > 0.5 and np.all((db > -10.0) & (db < 30.05)), db
    with pytest.raises(ValueError, match="wind_sc"):
        render_noise(_static_wind_flight_payload(), rps, sr=SR, n_mics=2, wind_sc=law)
