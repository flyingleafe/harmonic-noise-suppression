"""The lag-coherence statistic against signals whose answer is known."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.stochastic_fit.phase_stats import (
    HANN_OVERLAP,
    lag_coherence,
    lorentzian_lag_prediction,
)

SR, N_FFT, HOP = 16000, 2048, 512
LAGS = (1, 2, 3, 4, 6, 8, 12, 16)


def _tracks_at(freq_hz: float, n_frames: int) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(np.arange(n_frames), np.full(n_frames, freq_hz))]


def _frames(n_samples: int) -> int:
    return 1 + (n_samples - N_FFT) // HOP


def test_white_noise_follows_the_window_overlap():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, SR * 30))
    coh = lag_coherence(x, _tracks_at(1000.0, _frames(x.shape[1])), LAGS)
    expect = np.array([HANN_OVERLAP[min(lag, 4)] for lag in LAGS])
    np.testing.assert_allclose(coh.coherence[:4], expect[:4], atol=0.03)
    assert np.all(coh.coherence[4:] < 0.04)
    # the null is white noise through the same window: it carries the overlap too
    np.testing.assert_allclose(coh.null[:4], expect[:4], atol=0.03)
    assert np.all(coh.null[4:] < 0.04)


@pytest.mark.parametrize("gamma_hz", [1.0, 4.0])
def test_wiener_phase_tone_decays_like_a_lorentzian(gamma_hz: float):
    """A tone with Wiener phase (diffusion q = 4 pi gamma) has a Lorentzian
    line of half width gamma; its lag coherence is the window overlap times
    exp(-2 pi gamma tau) — no plateau."""
    rng = np.random.default_rng(1)
    n = SR * 40
    t = np.arange(n) / SR
    q = 4 * np.pi * gamma_hz
    phase = np.cumsum(rng.standard_normal((4, n)) * np.sqrt(q / SR), axis=1)
    x = np.cos(2 * np.pi * 1000.0 * t + phase) + 0.01 * rng.standard_normal((4, n))
    coh = lag_coherence(x, _tracks_at(1000.0, _frames(n)), LAGS)
    pred = lorentzian_lag_prediction(gamma_hz, np.asarray(LAGS))
    np.testing.assert_allclose(coh.coherence, pred, atol=0.05)
    assert coh.coherence[-1] < 0.1  # no plateau


@pytest.mark.parametrize("share", [0.3, 0.7])
def test_tone_in_noise_plateaus_at_its_coherent_share(share: float):
    """Coherent tone + independent narrowband noise at the same bin: beyond the
    window overlap the coherence sits at the tone's share of the bin power."""
    rng = np.random.default_rng(2)
    n = SR * 40
    t = np.arange(n) / SR
    f0 = 1000.0
    df = SR / N_FFT
    # the noise: white noise filtered to ±2 bins around f0 so its power sits in the tone's bin
    white = rng.standard_normal((4, n))
    spec = np.fft.rfft(white, axis=1)
    f = np.fft.rfftfreq(n, 1 / SR)
    spec[:, np.abs(f - f0) > 2 * df] = 0
    noise = np.fft.irfft(spec, n=n, axis=1)
    noise /= noise.std()
    tone = np.sqrt(2.0) * np.cos(2 * np.pi * f0 * t)  # unit power
    x = np.sqrt(share) * tone[None, :] + np.sqrt(1 - share) * noise
    coh = lag_coherence(x, _tracks_at(f0, _frames(n)), LAGS)
    plateau = coh.coherence[4:]  # lags 6..16: overlap is zero there
    # the bin sees the tone's full power but only the part of the noise inside
    # one bin's response; the plateau is therefore >= share, and flat
    assert plateau.min() > share - 0.05
    assert plateau.max() - plateau.min() < 0.08
    if share > 0.5:
        assert plateau.mean() > 0.6


def test_a_chirp_crossing_bins_reads_as_a_coherent_tone():
    """A clean chirp sweeping through many FFT bins is perfectly coherent; the
    per-frame centre evaluation must not see bin-boundary phase flips."""
    rng = np.random.default_rng(3)
    n = SR * 20
    t = np.arange(n) / SR
    f_inst = 800.0 + 60.0 * t  # 60 Hz/s: ~8 bins over the clip
    phase = 2 * np.pi * np.cumsum(f_inst) / SR
    x = np.cos(phase)[None, :] + 0.05 * rng.standard_normal((2, n))
    n_frames = _frames(n)
    centres = f_inst[np.arange(n_frames) * HOP + N_FFT // 2]
    coh = lag_coherence(x, [(np.arange(n_frames), centres)], LAGS)
    assert np.all(coh.coherence > 0.97)


def test_many_short_runs_of_noise_read_as_their_null():
    """Pooling many short runs must not manufacture coherence: white noise
    chopped into 20-frame runs reads at (not above) the phase-scrambled null
    beyond the window overlap, and that null is well above zero."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, SR * 60))
    n_frames = _frames(x.shape[1])
    frames = np.arange(n_frames)
    keep = (frames % 21) != 20  # runs of 20 frames
    tracks = [(frames[keep], np.full(int(keep.sum()), 1000.0))]
    coh = lag_coherence(x, tracks, LAGS)
    beyond = (np.array(LAGS) >= 6) & (np.array(LAGS) <= 12)
    assert np.all(coh.coherence[beyond] < coh.null[beyond] + 0.06)
    assert np.all(coh.null[beyond] > 0.1)
