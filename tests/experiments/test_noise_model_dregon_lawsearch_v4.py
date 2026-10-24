"""The law search's fast path (``experiments.noise_model.dregon_lawsearch_v4``):
a candidate composed from ONE render's parts is ``render_noise`` of the edited
payload at the same seed. Gains, floor level and floor exponent compose
exactly (float32 rounding); a moved line exponent is applied on the envelope
knots after the anti-alias filter, a second-order approximation the bound
below holds. If the renderer changes its streams or its floor, the search
scores something other than what it exports, and this fails. CPU-small.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from data_processing.noise_model import FIT_SCHEMA_V4
from data_processing.noise_model.render import render_noise
from data_processing.noise_model.spectrum import floor_ctrl_hz
from experiments.noise_model import dregon_lawsearch_v4 as LS

SR = 16000
N = int(1.5 * SR)
SEED = 11


def _payload(alpha_aero: float, alpha_motor: float) -> dict[str, Any]:
    """A two-rotor, two-mic flight payload: lines, motor lines, a per-rotor
    floor with a static share and a static wind."""
    k = np.arange(1, 25)
    base = -20.0 - 18.0 * np.log10(k) - 6.0 * (k % 2)
    rng = np.random.default_rng(0)
    eq_f = np.array([100.0, 500.0, 1000.0, 2000.0, 4000.0, 7000.0])
    eq = rng.normal(0.0, 1.5, size=(2, 2, eq_f.size))
    return {
        "schema": FIT_SCHEMA_V4,
        "kind": "flight",
        "rig": "test",
        "n_rotors": 2,
        "n_mics": 2,
        "blades": 2,
        "pole_pairs": 7,
        "f_ref_rev_s": 80.0,
        "sr": SR,
        "band_hz": [30.0, 7200.0],
        "params": {
            "profile_aero_db": np.stack([base, base - 2.0]).tolist(),
            "motor_orders": [21],
            "profile_motor_db": [[-42.0], [-44.0]],
            "alpha_aero": alpha_aero,
            "alpha_motor": alpha_motor,
            "pedestal": {
                "sigma2_amp": np.full(k.size, 0.1).tolist(),
                "gamma_amp_hz": np.full(k.size, 5.0).tolist(),
                "sigma2_phase": np.full(k.size, 0.5).tolist(),
                "gamma_phase_hz": np.full(k.size, 5.0).tolist(),
            },
            "shaft": {"eps": 0.001, "a_sigma": 1.0, "lam_ref_per_s": 12.0, "b_lam": 1.0},
            "floor": {
                "mean_db": -45.0,
                "shape_z": rng.normal(0.0, 1.0, size=14).tolist(),
                "shape_sd_db": 3.0,
                "ctrl_hz": floor_ctrl_hz(SR).tolist(),
                "b0": 4.0,
                "b1": 0.8,
                "static_rel": 0.3,
            },
            "mic_response": {
                "kind": "pair_eq",
                "pair_eq": {"freqs_hz": eq_f.tolist(), "db": eq.tolist()},
            },
            "wind": {"kind": "static_per_mic", "wind_db": [-20.0, -26.0]},
        },
    }


def _rps() -> np.ndarray:
    """Two rotors through ~0.85-1.1 of f_ref, as in flight."""
    t = np.arange(N) / SR
    return np.stack(
        [70.0 + 15.0 * t / t[-1] + 3.0 * np.sin(2.0 * np.pi * 0.7 * t), 88.0 - 14.0 * t / t[-1]]
    )


def _composed_and_direct(fit: dict[str, Any], point: dict[str, float]) -> np.ndarray:
    rps = _rps()
    parts = LS.render_parts(fit, rps, sr=SR, n_mics=2, seed=SEED)
    comp = LS.components(
        dict(
            parts,
            name="test",
            recording="test",
            start_s=0.0,
            dur_s=N / SR,
            sr=SR,
            seed=SEED,
            rps=rps,
            real=np.zeros((2, N), dtype=np.float32),
        ),
        fit,
    )
    composed = LS.compose(comp, point)
    direct = render_noise(LS.edit_payload(fit, point), rps, sr=SR, n_mics=2, seed=SEED)
    return (composed - direct) / np.sqrt(np.mean(direct**2))


def test_gains_and_the_floor_law_compose_exactly() -> None:
    fit = _payload(0.0, 0.0)
    err = _composed_and_direct(
        fit,
        dict(
            g_a=5.0,
            g_a1=-4.0,
            g_r1=2.0,
            g_r2=-3.0,
            g_r3=0.0,
            g_r4=1.5,
            alpha_a=0.0,
            g_m=-7.0,
            alpha_m=0.0,
            g_f=3.0,
            b0=9.0,
        ),
    )
    assert np.abs(err).max() < 1e-5


def test_moved_line_exponents_compose_to_a_small_error() -> None:
    fit = _payload(4.0, 0.5)
    err = _composed_and_direct(
        fit,
        dict(
            g_a=-2.0,
            g_a1=0.0,
            g_r1=0.0,
            g_r2=0.0,
            g_r3=0.0,
            g_r4=0.0,
            alpha_a=12.0,
            g_m=4.0,
            alpha_m=-3.0,
            g_f=-6.0,
            b0=2.5,
        ),
    )
    # measured: rms 3.4e-5, max 4.0e-4 of the direct render's rms
    assert np.sqrt(np.mean(err**2)) < 3e-4
    assert np.abs(err).max() < 4e-3
