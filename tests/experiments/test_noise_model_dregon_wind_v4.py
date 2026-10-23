"""The v4 ``sc`` wind (``experiments.noise_model.dregon_wind_v4``, the
renderer's ``sc`` kind).

A payload with the SC gust wind renders through ``render_noise`` with every
capsule's mean wind PSD at ``10^{level/10} shape.psd`` (the stated level and
shape, per mic) and ``expected_periodogram`` reads the same; and the gust law
moves the stage's gust instrument the right way: a larger wind-speed scale
gives a larger floor-envelope sd. CPU-small (2-8 s clips, few seeds).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from data_processing.noise_model import FIT_SCHEMA_V4
from data_processing.noise_model.render import render_noise, render_v4_wind
from data_processing.noise_model.spectrum import floor_ctrl_hz
from data_processing.noise_model.v4 import expected_periodogram
from experiments.noise_model import dregon_wind_v4 as DW

SR = 16000
LEVEL_DB = [-20.0, -26.0]


def _payload(wind: dict[str, Any]) -> dict[str, Any]:
    """A one-rotor flight payload whose lines and floor are silent (``-1e4`` dB:
    exactly zero power), with the wind block ``wind``."""
    m = len(wind["level_db"])
    return {
        "schema": FIT_SCHEMA_V4,
        "kind": "flight",
        "rig": "test",
        "n_rotors": 1,
        "n_mics": m,
        "blades": 2,
        "pole_pairs": 7,
        "f_ref_rev_s": 80.0,
        "sr": SR,
        "band_hz": [30.0, 7200.0],
        "params": {
            "profile_aero_db": [[-1.0e4] * 4],
            "motor_orders": [],
            "profile_motor_db": [[]],
            "alpha_aero": 0.0,
            "alpha_motor": 0.0,
            "pedestal": {
                k: [0.0] * 4
                for k in ("sigma2_amp", "gamma_amp_hz", "sigma2_phase", "gamma_phase_hz")
            },
            "shaft": {"eps": 0.001, "a_sigma": 0.0, "lam_ref_per_s": 10.0, "b_lam": 0.0},
            "floor": {
                "mean_db": -1.0e4,
                "shape_z": [0.0] * 14,
                "shape_sd_db": 0.0,
                "ctrl_hz": floor_ctrl_hz(SR).tolist(),
                "b0": 0.0,
                "b1": 0.0,
                "static_rel": 0.0,
            },
            "mic_response": {
                "kind": "pair_eq",
                "pair_eq": {"freqs_hz": [100.0, 1000.0], "db": [[[0.0, 0.0]] * m]},
            },
            "wind": wind,
        },
    }


def test_sc_payload_renders_its_stated_level_and_shape() -> None:
    clip_s, n_fft = 2.0, 2048
    shape = DW.sc_shape(2.0, 1.0, 1.0, sr=SR, clip_s=clip_s, n_mc=24, seed=1, n_fft=n_fft)
    fit = _payload(DW.sc_block(LEVEL_DB, 2.0, 1.0, 1.0, shape))
    rps = np.full((1, int(clip_s * SR)), 80.0)
    got = np.mean(
        [
            DW.mean_periodogram(render_noise(fit, rps, sr=SR, n_mics=2, seed=s), n_fft)
            for s in range(24)
        ],
        axis=0,
    )
    f = np.asarray(shape["freqs_hz"])
    want = 10.0 ** (np.asarray(LEVEL_DB)[:, None] / 10.0) * np.asarray(shape["psd"])[None, :]
    e = expected_periodogram(fit, [80.0], n_fft=n_fft, sr=SR)
    for lo in (20.0, 40.0, 80.0, 160.0, 320.0):
        band = (f >= lo) & (f < min(2.0 * lo, 500.0))
        err_db = 10.0 * np.log10(got[:, band].sum(axis=1) / want[:, band].sum(axis=1))
        assert np.all(np.abs(err_db) < 1.0), (lo, err_db)
        # the model's expectation is the stated wind, window-smoothed
        e_db = 10.0 * np.log10(e[:, band].sum(axis=1) / want[:, band].sum(axis=1))
        assert np.all(np.abs(e_db) < 0.3), (lo, e_db)


def test_larger_speed_scale_gives_a_gustier_floor_envelope() -> None:
    clip_s, n_mics = 8.0, 4
    n = int(clip_s * SR)
    floor = np.random.default_rng(0).standard_normal((n_mics, n)) * np.sqrt(1e-3)
    rps = np.zeros((1, n))  # no rotor lines: every 50-500 Hz bin is a floor cell
    std = {}
    for scale in (0.25, 3.0):
        shape = DW.sc_shape(2.0, scale, 1.0, sr=SR, clip_s=clip_s, n_mc=4, seed=1)
        fit = _payload(DW.sc_block([-10.0] * n_mics, 2.0, scale, 1.0, shape))
        x = floor + render_v4_wind(fit, n, sr=SR, n_mics=n_mics, seed=4)
        std[scale] = float(np.mean(DW.wind_statistics(x, rps, SR)["mic"]["std"]))
    # the same draws but for the knots' scale: 1.9 vs 4.2 dB at these seeds
    assert std[3.0] > std[0.25] + 1.0, std
