"""The v4 ``sc`` wind (``experiments.noise_model.dregon_wind_v4``, the
renderer's ``sc`` kind) and the battery gust search.

A payload with the SC gust wind renders through ``render_noise`` with every
capsule's mean wind PSD at ``10^{level/10} shape.psd`` (the stated level and
shape, per mic) and ``expected_periodogram`` reads the same; and the gust law
moves the stage's gust instrument the right way: a larger wind-speed scale
gives a larger floor-envelope sd. The battery search's floor split is the
renderer's (a floor ``mean_db`` step is an exact amplitude step of the floor
share), a battery point at an sc source's own law recovers the source from
data that are the source's render at the grid seeds (objective ~0, the gust law
round-trips into the exported payload), and the source's law is always a grid
point. CPU-small (2-8 s clips, few seeds).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from data_processing.noise_model import FIT_SCHEMA_V4
from data_processing.noise_model.render import render_noise, render_v4_wind
from data_processing.noise_model.spectrum import floor_ctrl_hz
from data_processing.noise_model.v4 import expected_periodogram
from experiments.noise_model import dregon_flight_v4 as DF
from experiments.noise_model import dregon_wind_v4 as DW

SR = 16000
LEVEL_DB = [-20.0, -26.0]


def _payload(wind: dict[str, Any], floor_db: float = -1.0e4) -> dict[str, Any]:
    """A one-rotor flight payload whose lines are silent (``-1e4`` dB: exactly
    zero power), with a flat floor of ``floor_db`` (silent by default) and the
    wind block ``wind``."""
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
                "mean_db": floor_db,
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


def test_floor_mean_step_is_an_amplitude_step_of_the_split_floor() -> None:
    # the battery search renders the source once, split into lines and floor,
    # and moves the floor mean by scaling the floor share
    shape = DW.sc_shape(2.0, 1.0, 1.0, sr=SR, clip_s=2.0, n_mc=2, seed=1)
    fit = DW.with_wind(_payload(DW.sc_block(LEVEL_DB, 2.0, 1.0, 1.0, shape), -40.0), None)
    fit["params"]["profile_aero_db"] = [[-10.0, -14.0, -18.0, -22.0]]
    rps = np.linspace(70.0, 90.0, 2 * SR)[None, :]
    full = render_noise(fit, rps, sr=SR, n_mics=2, seed=3)
    lines = render_noise(DF.silent_floor(fit), rps, sr=SR, n_mics=2, seed=3)
    fit["params"]["floor"]["mean_db"] = -40.0 + 7.0
    moved = render_noise(fit, rps, sr=SR, n_mics=2, seed=3)
    want = lines + 10.0 ** (7.0 / 20.0) * (full - lines)
    assert np.abs(moved - want).max() < 1e-9 * np.abs(moved).max()


def _battery_context(stage: dict[str, Any], clip_s: float = 2.0) -> DW.BatteryContext:
    """A one-segment, one-seed battery context whose 'real' segment is the
    source's own render at the grid's seeds and whose pool periodograms are the
    source's expectation (so the inner fit's MAP is the source's values)."""
    n = int(clip_s * SR)
    rps = np.full((1, n), 80.0)
    nowind = DW.with_wind(stage, None)
    full = render_noise(nowind, rps, sr=SR, n_mics=2, seed=DW.base_seed(0, 0))
    lines = render_noise(DF.silent_floor(nowind), rps, sr=SR, n_mics=2, seed=DW.base_seed(0, 0))
    real = full + render_v4_wind(stage, n, sr=SR, n_mics=2, seed=DW.wind_seed(0, 0))
    pair = [render_noise(stage, rps, sr=SR, n_mics=2, seed=DW.floor_seed(0, q)) for q in (0, 1)]
    floors = DW.seed_floors([dict(segment=0, pair=0, terms=DW.battery_terms(*pair)[0])])
    n_fft, speeds = 512, (70.0, 80.0, 90.0)
    band = np.flatnonzero(np.fft.rfftfreq(n_fft, 1.0 / SR) >= 30.0)

    def expect(fit: dict[str, Any]) -> np.ndarray:
        return np.stack(
            [expected_periodogram(fit, [s], n_fft=n_fft, sr=SR)[:, band] for s in speeds], axis=1
        )

    frozen = expect(nowind)
    level = DW.LevelData(
        power=expect(stage),
        frozen=frozen,
        weights=np.ones(len(speeds)),
        band=band,
        centre_db=DW.source_level_db(stage),
        n_fft=n_fft,
        sr=SR,
        floor=frozen - expect(DF.silent_floor(nowind)),
        floor_centre_db=float(stage["params"]["floor"]["mean_db"]),
    )
    return DW.BatteryContext(
        stage=stage,
        names=["seg0"],
        rps=rps[None],
        real=real[None],
        sr=SR,
        n_mics=2,
        seeds=1,
        n_mc=4,
        shape_seed=1,
        level=level,
        floors=floors["floor"],
        weights=dict(DW.BATTERY_WEIGHTS),
        base=lambda i, k: (lines, full - lines),
    )


def test_battery_point_at_the_sc_source_law_recovers_the_source() -> None:
    shape = DW.sc_shape(2.0, 0.5, 2.0, sr=SR, clip_s=2.0, n_mc=4, seed=1)
    stage = _payload(DW.sc_block(LEVEL_DB, 2.0, 0.5, 2.0, shape), -45.0)
    ctx = _battery_context(stage)
    row = DW.battery_point(ctx, *DW.source_law(stage), source=True)
    # the inner fit lands on the source, so the sim is the 'real' render itself
    assert np.allclose(row["level_db"], LEVEL_DB, atol=0.01), row["level_db"]
    assert abs(row["floor_mean_db"] + 45.0) < 0.01, row["floor_mean_db"]
    assert row["objective"] < 1e-3, row["objective_terms"]["by_term"]
    # ... and the exported payload carries the source's gust law and shape
    wind = DW.gust_payload(stage, row)["params"]["wind"]
    for key in ("speed_offset_ms", "speed_scale_ms", "gust_rate_hz", "shape"):
        assert wind[key] == stage["params"]["wind"][key], key
    # another law cannot reproduce the realisation
    other = DW.battery_point(ctx, 2.0, 3.0, 0.5)
    assert other["objective"] > 1.0, other["objective"]


def test_the_sc_source_law_is_always_a_grid_point() -> None:
    shape = DW.sc_shape(2.0, 0.5, 2.0, sr=SR, clip_s=2.0, n_mc=1, seed=1)
    stage = _payload(DW.sc_block(LEVEL_DB, 2.0, 0.5, 2.0, shape))
    inside = DW.grid_laws((0.0, 2.0), (0.5, 1.0), (2.0,), stage)
    assert [x["source"] for x in inside] == [False, False, True, False]
    missed = DW.grid_laws((0.0, 4.0), (1.0,), (1.0,), stage)
    assert len(missed) == 3 and missed[-1]["source"] and missed[-1]["law"] == (2.0, 0.5, 2.0)
    static = DW.with_wind(stage, dict(kind="static_per_mic", wind_db=LEVEL_DB))
    assert not any(x["source"] for x in DW.grid_laws((0.0, 4.0), (1.0,), (1.0,), static))
