"""Consumer-visible units, silence handling and rotor-clock accuracy."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.generator_paper.baselines import _pulse_events
from experiments.generator_paper.metrics import evaluate_pair


def test_power_spectral_distance_uses_amplitude_gain_in_decibels():
    real = 0.02 * np.random.default_rng(12).standard_normal((1, 2 * 16000))
    rps = np.full((1, real.shape[-1]), 80.0)
    result = evaluate_pair(real, 2.0 * real, rps)
    assert result["summary"]["lsd_db"] == pytest.approx(20.0 * np.log10(2.0), abs=1e-5)


def test_silent_candidate_has_no_fictitious_perceptual_match():
    real = 0.02 * np.random.default_rng(13).standard_normal((1, 2 * 16000))
    rps = np.full((1, real.shape[-1]), 80.0)
    result = evaluate_pair(real, np.zeros_like(real), rps)
    assert result["summary"]["lsd_db"] > 40.0
    key = "nominal80__aures_sharpness_abs_diff_acum"
    assert key not in result["summary"]
    assert key in result["details"]["undefined"]


def test_out_of_domain_playback_preserves_spectral_error_without_partial_sharpness_credit():
    real = 0.02 * np.random.default_rng(14).standard_normal((2, 2 * 16000))
    generated = real.copy()
    generated[1] *= 1e6
    rps = np.full((1, real.shape[-1]), 80.0)
    result = evaluate_pair(real, generated, rps)
    assert result["summary"]["lsd_db"] == pytest.approx(60.0, abs=1e-4)
    key = "nominal80__aures_sharpness_abs_diff_acum"
    assert key not in result["summary"] and key in result["details"]["undefined"]
    psycho = result["details"]["nominal80_psychoacoustics"]
    assert psycho["zwicker_domain_valid_per_mic"] == [True, False]
    assert psycho["mics_with_defined_sharpness"] == 0


def test_fractional_pulse_clock_does_not_accumulate_period_rounding_error():
    sr, rps = 48000, 83.123
    audio = _pulse_events(
        np.full(2 * sr, 2.0 * rps),
        np.zeros(2 * sr),
        sr=sr,
        phase_carry=True,
        initial_phase=0.0,
        pulse_width=1,
    )
    onsets = np.flatnonzero(audio)
    measured_rps = sr / (2.0 * np.diff(onsets).mean())
    assert measured_rps == pytest.approx(rps, abs=0.002)
