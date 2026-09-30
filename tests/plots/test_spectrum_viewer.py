"""Numerics of plots.spectrum_viewer: the amplitude convention and display decimation."""

from __future__ import annotations

import numpy as np
import pytest

from plots.spectrum_viewer import (
    _pool_columns,
    frame_spectrum,
    mean_spectrum,
    overall_spectrum,
    peak_decimate,
    stft_of,
)

SR = 16000
N_FFT = 2048


def _tone(amplitude: float, seconds: float = 4.0) -> np.ndarray:
    # Bin-centred for both the STFT (SR / N_FFT) and the whole-signal FFT (1 / seconds).
    f0 = SR / N_FFT * 40
    t = np.arange(int(SR * seconds)) / SR
    return amplitude * np.sin(2 * np.pi * f0 * t)


def test_a_bin_centred_tone_reads_its_amplitude_in_every_mode() -> None:
    x = _tone(0.3, seconds=8.0)
    s = stft_of(x, SR, n_fft=N_FFT, hop=512)
    _, overall = overall_spectrum(x, SR)
    assert overall.max() == pytest.approx(0.3, rel=1e-3)
    assert frame_spectrum(s, 4.0).max() == pytest.approx(0.3, rel=1e-3)
    # The centred first/last frames see reflect padding; over 8 s the RMS stays within 1 %.
    assert mean_spectrum(s).max() == pytest.approx(0.3, rel=1e-2)


def test_peak_decimation_keeps_a_one_bin_line() -> None:
    freqs = np.linspace(0, 8000, 100_001)
    values = np.full_like(freqs, 1e-3)
    values[61_234] = 1.0
    f, v = peak_decimate(freqs, values, 0.0, 8000.0, 500)
    assert f.size <= 500
    assert v.max() == 1.0 and f[np.argmax(v)] == freqs[61_234]
    f_log, v_log = peak_decimate(freqs, values, 10.0, 8000.0, 300, log=True)
    assert f_log.size <= 300 and v_log.max() == 1.0 and f_log.min() >= 10.0


def test_column_pooling_keeps_the_trailing_columns() -> None:
    power = np.ones((3, 103), dtype=np.float32)
    power[:, -1] = 7.0
    times = np.arange(103, dtype=np.float64)
    pooled, pooled_t = _pool_columns(power, times, 10)
    # k = 11: ten groups, the last one is columns 99..102 (4 columns, one of them the 7).
    assert pooled.shape[1] == 10
    assert pooled_t[-1] == pytest.approx(100.5)
    assert pooled[0, -1] == pytest.approx((3 * 1.0 + 7.0) / 4)
