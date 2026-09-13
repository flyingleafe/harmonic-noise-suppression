"""The render's timebase, its LEVEL units, and the observation's units.

Three things are pinned here:

* **Timebase.** ``stage2.render_from_export`` takes a 16 kHz rotor track,
  resamples it onto the native render grid, and returns 16 kHz audio of the
  SAME length as the track it was given. The rps16k -> native conversion is
  existing, ruled correct upstream, and is NOT modified by these tests — they
  exist so that a future change to it shows up as a failure here instead of as
  a silent half-length synthetic clip.
* **Level units.** The declared conversion from stored fit power to renderer
  units (``revised_eval.to_renderer_units`` plus ``normalize_rms=None``) is
  validated by PLANTING known levels and measuring the rendered, decimated
  clip: a pure tone and the broadband floor, separately, because the two level
  effects are separable. ``normalize_rms`` is a common gain on the whole
  waveform (it moves tone and floor together and leaves their ratio intact);
  the ``sr/native`` term is the periodogram-vs-PSD factor.
* **Observation units.** The frozen spectral observation is
  ``|STFT|^2 / sum(w^2)``, so white noise of variance ``s^2`` has mean
  periodogram ``s^2`` and a pure tone of amplitude ``A`` carries ``A^2 / 2``
  summed over its leakage — including when its frequency sits between two
  bins. The composite score's ``I / M`` ratio is only meaningful if that holds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from experiments.stochastic_fit import revised_eval as RE
from experiments.stochastic_fit import stage2 as S2
from experiments.stochastic_fit.data import Clip

SR = 16000


def _export() -> dict[str, Any]:
    """A minimal four-rotor export, enough for ``params_from_export``."""
    return dict(
        profile_db=np.full((4, 40), -20.0).tolist(),
        floor_mean_db=-40.0,
        floor_shape_db=[0.0, 0.0],
        floor_ctrl_hz=[30.0, 8000.0],
        gamma0=[1.0] * 4,
        gamma_slope=[0.5] * 4,
        mic_floor_db=[0.0, 0.0],
        mic_gain_db=[[0.0] * 4, [0.0] * 4],
        amp_exp=2.5,
        floor_exp=2.5,
        floor_static_rel=0.1,
        coherence_k_half=2.0,
        floor_tilt_db_oct=0.0,
    )


def test_the_rotor_track_is_resampled_onto_the_native_grid_and_the_length_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The renderer must be driven at the NATIVE rate for the same duration, and
    the decimated result must carry the duration it was asked for."""
    from data_processing import stochastic_rotor_noise as srn

    seen: dict[str, Any] = {}

    def fake_synthesize(params: Any, rps: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, dict]:
        seen["sample_rate"] = int(params.sample_rate)
        seen["rps_shape"] = tuple(np.asarray(rps).shape)
        n = int(np.asarray(rps).shape[-1])
        rng = np.random.default_rng(0)
        return rng.standard_normal((kwargs.get("n_mics", 2), n)) * 0.01, {}

    monkeypatch.setattr(srn, "synthesize", fake_synthesize)

    seconds = 4.0
    n16 = int(SR * seconds)
    rps16 = np.tile(np.linspace(80.0, 82.0, n16), (4, 1))
    audio = S2.render_from_export(_export(), rps16, n_mics=2, seed=0)

    # driven natively, for the same wall-clock duration as the 16 kHz track
    assert seen["sample_rate"] == S2.clips.NATIVE_SR
    assert seen["rps_shape"] == (4, int(round(n16 / SR * S2.clips.NATIVE_SR)))
    # and decimated back to the duration that was asked for (one sample of slack)
    assert abs(audio.shape[-1] - n16) <= 1
    assert audio.shape[0] == 2


def test_a_native_rate_track_is_not_silently_reinterpreted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The contract is a 16 kHz track. Handing it a native-rate one asks for a
    clip 2.76x too long — pinned so the convention cannot drift unnoticed."""
    from data_processing import stochastic_rotor_noise as srn

    seen: dict[str, Any] = {}

    def fake_synthesize(params: Any, rps: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, dict]:
        seen["n"] = int(np.asarray(rps).shape[-1])
        return np.zeros((kwargs.get("n_mics", 1), seen["n"])), {}

    monkeypatch.setattr(srn, "synthesize", fake_synthesize)
    native = np.tile(np.full(int(S2.clips.NATIVE_SR * 1.0), 80.0), (4, 1))
    S2.render_from_export(_export(), native, n_mics=1, seed=0)
    assert seen["n"] == int(round(S2.clips.NATIVE_SR / SR * S2.clips.NATIVE_SR))


def test_white_noise_reads_its_own_variance_on_the_observation_grid() -> None:
    rng = np.random.default_rng(3)
    sigma = 0.3
    x = (rng.standard_normal((1, 3 * RE.OBS_N_FFT)) * sigma).astype(np.float32)
    pg = RE.window_periodogram(Clip("w", "synthetic", x, np.full((1, x.shape[-1]), 80.0), SR))
    mean_power = float(np.asarray(pg.power, dtype=np.float64).mean())
    assert mean_power == pytest.approx(sigma**2, rel=0.05)


@pytest.mark.parametrize("offset_bins", [0.0, 0.5, 0.31])
def test_a_tone_carries_half_its_squared_amplitude_at_any_fractional_bin(
    offset_bins: float,
) -> None:
    """Leakage moves power between bins; the summed in-band power does not."""
    n = 2 * RE.OBS_N_FFT
    df = SR / RE.OBS_N_FFT
    hz = 1000.0 + offset_bins * df
    amp = 0.5
    t = np.arange(n) / SR
    x = (amp * np.sin(2.0 * np.pi * hz * t))[None, :].astype(np.float32)
    pg = RE.window_periodogram(Clip("t", "synthetic", x, np.full((1, n), 80.0), SR))
    power = np.asarray(pg.power, dtype=np.float64)
    near = np.abs(pg.freqs - hz) <= 20 * df
    # mean over frames, summed over the tone's leakage: the one-sided periodogram
    # holds half the mean power, so 2 * sum / n_fft is the tone's own A^2 / 2
    per_bin = power[0].mean(axis=0)
    total = 2.0 * float(per_bin[near].sum()) / RE.OBS_N_FFT
    assert total == pytest.approx(amp**2 / 2.0, rel=0.05)


# ── the declared level conversion, planted and measured ─────────────────────

PLANT_FLOOR_DB, PLANT_PROFILE_DB, PLANT_ORDERS = -40.0, -10.0, 40


def _planted(offset_db: float = 0.0) -> dict[str, Any]:
    """One rotor, 40 equal orders, a flat floor, no speed law, coherent lines.

    ``coherence_k_half=0`` puts the whole line power through the tone bank, and
    the zero exponents switch the speed law off, so the fitted numbers ARE the
    levels the rendered clip must come back with.
    """
    return dict(
        profile_db=[[PLANT_PROFILE_DB + offset_db] * PLANT_ORDERS],
        floor_mean_db=PLANT_FLOOR_DB + offset_db,
        floor_shape_db=[0.0] * 14,
        floor_ctrl_hz=np.geomspace(30.0, 8000.0, 14).tolist(),
        floor_tilt_db_oct=0.0,
        gamma0=[1.0],
        gamma_slope=[0.0],
        mic_floor_db=[0.0],
        mic_gain_db=[[0.0]],
        amp_exp=0.0,
        floor_exp=0.0,
        floor_static_rel=0.0,
        coherence_k_half=0.0,
    )


def _measure(offset_db: float, *, normalize_rms: float | None, seconds: float = 3.0) -> dict:
    n = int(SR * seconds)
    rps = np.full((1, n), 80.0)
    audio = S2.render_from_export(
        _planted(offset_db), rps, n_mics=1, seed=5, normalize_rms=normalize_rms
    )
    clip = Clip("planted", "synthetic", audio.astype(np.float32), rps[:, : audio.shape[-1]], SR)
    pg = RE.window_periodogram(clip)
    power = np.asarray(pg.power, dtype=np.float64)[0].mean(axis=0)
    f, df = pg.freqs, pg.df
    floor = float(power[(f > 4000.0) & (f < 6000.0)].mean())  # above the 40th order (3200 Hz)
    out = dict(floor_db=10.0 * np.log10(floor / 10.0 ** (PLANT_FLOOR_DB / 10.0)))
    for k in (5, 10, 20):
        centre = 80.0 * k
        near = np.abs(f - centre) <= 20 * df
        side = (np.abs(f - centre) > 20 * df) & (np.abs(f - centre) <= 35 * df)
        integral = float((power[near] - power[side].mean()).sum()) * df
        out[f"line_{k}_db"] = 10.0 * np.log10(integral / 10.0 ** (PLANT_PROFILE_DB / 10.0))
    return out


def test_the_declared_conversion_lands_a_planted_tone_and_floor_on_their_fitted_levels() -> None:
    """As stored, a native render reads 4.4 dB low in BOTH the tone integral and
    the floor; with the declared term applied it reads the fitted levels."""
    stored = _measure(0.0, normalize_rms=None)
    # The helper uses render_from_export's historical 44.1 kHz work grid default.
    historical_factor = RE.HISTORICAL_RATE_FACTOR_DB
    assert stored["floor_db"] == pytest.approx(-historical_factor, abs=0.3)
    for k in (5, 10, 20):
        assert stored[f"line_{k}_db"] == pytest.approx(-historical_factor, abs=0.3)

    converted = _measure(historical_factor, normalize_rms=None)
    assert abs(converted["floor_db"]) <= 0.35
    for k in (5, 10, 20):
        assert abs(converted[f"line_{k}_db"]) <= 0.35


def test_normalize_rms_is_a_common_gain_that_leaves_the_line_to_floor_ratio_alone() -> None:
    """The two level effects are separable: ``normalize_rms`` moves the tone and
    the floor by the SAME (clip-dependent) amount, so only the absolute level is
    destroyed — which is why the evaluator switches it off instead of
    compensating for it."""
    physical = _measure(0.0, normalize_rms=None)
    normalized = _measure(0.0, normalize_rms=0.1)
    gains = [normalized["floor_db"] - physical["floor_db"]] + [
        normalized[f"line_{k}_db"] - physical[f"line_{k}_db"] for k in (5, 10, 20)
    ]
    assert max(gains) - min(gains) < 0.35  # one common gain, not a tone-vs-floor effect
    assert abs(gains[0]) > 3.0  # and it really does move the absolute level
