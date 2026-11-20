"""Behavioural regressions for the EONE-RPS curve bank (generator paper control)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.generator_paper.cases import FlightClip
from experiments.generator_paper.eone_baseline import (
    EoneFit,
    EoneSettings,
    fit_eone,
    noise_band_layout,
    render_eone,
)

SR = 16000


def _bank(settings: EoneSettings, n_mics: int = 1, n_rotors: int = 1) -> np.ndarray:
    return np.zeros((n_mics, n_rotors, settings.n_components, settings.n_knots), dtype=np.float32)


def _render(
    settings: EoneSettings, curves: np.ndarray, rps: np.ndarray, seed: int = 0
) -> np.ndarray:
    fit = EoneFit(settings=settings, curves=curves, report={})
    return render_eone(fit, rps, sr=SR, rng=np.random.default_rng(seed)).astype(np.float64)


def _amplitude_spectrum(y: np.ndarray) -> np.ndarray:
    return 2.0 * np.abs(np.fft.rfft(y)) / y.size


def test_harmonic_lands_on_its_order_and_nothing_folds_past_nyquist():
    cfg = EoneSettings(n_orders=81, n_bands=4, n_knots=5, noise_loop=4096)
    curves = _bank(cfg)
    curves[0, 0, 4] = 0.5  # shaft order 5 -> 500 Hz
    curves[0, 0, 79] = 1.0  # order 80 -> 8000 Hz = Nyquist: gated
    curves[0, 0, 80] = 1.0  # order 81 -> 8100 Hz, would alias to 7900 Hz: gated
    y = _render(cfg, curves, np.full((1, SR), 100.0))[0]
    spec = _amplitude_spectrum(y)
    assert spec[500] == pytest.approx(0.5, rel=1e-3)
    rest = np.delete(spec, 500)
    assert np.sum(rest**2) < 1e-9 * spec[500] ** 2

    # Inside the 7900-8000 Hz raised-cosine transition the gate is 0.5 at 7950 Hz.
    curves = _bank(cfg)
    curves[0, 0, 79] = 1.0
    y = _render(cfg, curves, np.full((1, SR), 7950.0 / 80.0))[0]
    spec = _amplitude_spectrum(y)
    assert spec[7950] == pytest.approx(0.5, rel=1e-3)
    assert np.sum(np.delete(spec, 7950) ** 2) < 1e-9 * spec[7950] ** 2


def test_gain_interpolates_between_the_two_nearest_knots_only():
    cfg = EoneSettings(n_orders=4, n_bands=2, n_knots=5, noise_loop=4096)  # knots every 2500 RPM
    rps = np.full((1, SR), 5625.0 / 60.0)  # 2.25 knot spacings: 0.75 * knot2 + 0.25 * knot3
    curves = _bank(cfg)
    curves[0, 0, 0] = [7.0, 9.0, 2.0, 4.0, 11.0]
    y = _render(cfg, curves, rps)[0]
    base = 2.0 * math.pi * rps[0, 0] * np.arange(1, SR + 1) / SR
    coef, *_ = np.linalg.lstsq(np.stack([np.cos(base), np.sin(base)], axis=1), y, rcond=None)
    assert math.hypot(*coef) == pytest.approx(0.75 * 2.0 + 0.25 * 4.0, rel=1e-4)

    far = curves.copy()
    far[0, 0, 0, [0, 1, 4]] = [0.1, 50.0, 3.0]
    assert np.array_equal(_render(cfg, far, rps)[0], y)
    near = curves.copy()
    near[0, 0, 0, 3] = 5.0
    assert not np.allclose(_render(cfg, near, rps)[0], y)


def test_gain_ramps_linearly_between_frame_centres():
    cfg = EoneSettings(n_orders=4, n_bands=2, n_knots=5, hop=64, noise_loop=4096)
    n, step = 10 * cfg.hop, 4 * cfg.hop
    rps = np.where(np.arange(n) < step, 2500.0 / 60.0, 7500.0 / 60.0)[None, :]
    curves = _bank(cfg)
    curves[0, 0, 0, 1] = 1.0  # knot at 2500 RPM
    curves[0, 0, 0, 3] = 3.0  # knot at 7500 RPM
    seed = 3
    y = _render(cfg, curves, rps, seed=seed)[0]
    # The same draw with unit gain exposes the carrier without pinning RNG order.
    unit = _bank(cfg)
    unit[0, 0, 0] = 1.0
    carrier = _render(cfg, unit, rps, seed=seed)[0]
    centres = np.arange(n // cfg.hop) * cfg.hop + (cfg.hop - 1) / 2.0
    frame_gain = np.where(np.arange(n // cfg.hop) < step // cfg.hop, 1.0, 3.0)
    expected = np.interp(np.arange(n), centres, frame_gain)
    ok = np.abs(carrier) > 0.3
    np.testing.assert_allclose(y[ok] / carrier[ok], expected[ok], atol=1e-4)


def test_noise_band_is_a_unit_rms_loop_confined_to_its_erb_support():
    cfg = EoneSettings(n_orders=2, n_bands=4, n_knots=5, noise_loop=4096)
    loop, band = cfg.noise_loop, 2
    curves = _bank(cfg)
    curves[0, 0, cfg.n_orders + band] = 1.0
    rps = np.full((1, 3 * loop), 50.0)
    y = _render(cfg, curves, rps, seed=7)[0]
    np.testing.assert_allclose(y[loop : 2 * loop], y[:loop], atol=1e-5)
    np.testing.assert_allclose(y[2 * loop :], y[:loop], atol=1e-5)
    assert math.sqrt(np.mean(y[:loop] ** 2)) == pytest.approx(1.0, rel=1e-4)
    assert abs(np.mean(y[:loop])) < 1e-5

    # The level of the first period does not depend on the requested duration.
    short = _render(cfg, curves, rps[:, :loop], seed=7)[0]
    np.testing.assert_allclose(short, y[:loop], atol=1e-6)

    power = np.abs(np.fft.rfft(y[:loop])) ** 2
    freqs = np.fft.rfftfreq(loop, 1.0 / SR)
    lay = noise_band_layout(cfg)
    bin_hz = SR / loop
    outside = (freqs < lay["lower_hz"][band] - bin_hz) | (freqs > lay["upper_hz"][band] + bin_hz)
    assert outside.any()
    assert power[outside].sum() < 1e-9 * power.sum()


_SMOKE = EoneSettings(
    piece_s=0.25,
    n_orders=8,
    n_bands=4,
    n_knots=5,
    hop=64,
    noise_loop=4096,
    epochs=6,
    max_lr=0.05,
    mrstft_max_fft=1024,
    mrstft_min_fft=64,
    harmonic_n_fft=4096,
    harmonic_win=1024,
    harmonic_hop=64,
)


def _synthetic_calibration() -> list[FlightClip]:
    truth = _bank(_SMOKE, n_mics=2)
    truth[0, 0, 1] = 0.3  # shaft order 2
    truth[1, 0, 1] = 0.15
    truth[:, 0, _SMOKE.n_orders + 1] = 0.02
    gen = EoneFit(settings=_SMOKE, curves=truth, report={})
    clips = []
    for i, (a, b) in enumerate([(60.0, 80.0), (80.0, 60.0)]):
        rps = np.linspace(a, b, SR // 2)[None, :]
        audio = render_eone(gen, rps, sr=SR, rng=np.random.default_rng(100 + i)).astype(np.float64)
        clips.append(
            FlightClip(
                key=f"md2:SYN@{0.5 * i:.3f}+0.5",
                rig="md2",
                recording_id="SYN",
                start_s=0.5 * i,
                sr=SR,
                audio=audio,
                rps=rps,
                native_sr=SR,
                native_audio=audio,
                regime="cruise",
                provenance={"split": "calibration"},
            )
        )
    return clips


def test_tiny_fit_learns_the_dominant_order_and_reloads_identically(tmp_path):
    clips = _synthetic_calibration()
    path = tmp_path / "eone.pt"
    fit = fit_eone(clips, seed=0, checkpoint=path, settings=_SMOKE, device="cpu")
    rep = fit.report
    assert rep["production_settings"] is False
    assert rep["calibration_splits"] == ["calibration"]
    assert rep["training"]["completed_epochs"] == _SMOKE.epochs
    assert rep["training"]["best_eval_objective"] < rep["training"]["initial_eval_objective"]
    # 60-80 rps = 3600-4800 RPM, mostly weighting the 5000-RPM knot (index 2).
    assert fit.harmonic_curves[0, 0, 1, 2] > fit.harmonic_curves[0, 0, 6, 2]

    again = EoneFit.load(path)
    assert np.array_equal(again.curves, fit.curves)
    rps = clips[0].rps
    a = render_eone(fit, rps, sr=SR, rng=np.random.default_rng(5))
    b = render_eone(again, rps, sr=SR, rng=np.random.default_rng(5))
    assert a.dtype == np.float32 and a.shape == clips[0].audio.shape
    assert np.array_equal(a, b)


def test_fit_refuses_evaluation_windows(tmp_path):
    clips = _synthetic_calibration()
    leaked = [FlightClip(**{**c.__dict__, "provenance": {"split": "evaluation"}}) for c in clips]
    with pytest.raises(ValueError, match="calibration windows only"):
        fit_eone(leaked, seed=0, checkpoint=tmp_path / "x.pt", settings=_SMOKE, device="cpu")
    assert not (tmp_path / "x.pt").exists()
