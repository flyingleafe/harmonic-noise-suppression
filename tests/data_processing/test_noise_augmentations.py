"""Tests for the pre-mix noise augmentations (``spectral_recolor``,
``random_reverb``) and their Frame contract: audio-only, labels untouched."""

from __future__ import annotations

import numpy as np
import pytest
import tdseries as td

from data_processing.noise_augmentations import (
    _random_reverb,
    _spectral_recolor,
    maybe_apply_noise_augmentation,
)

SR = 16000
T = 16000


def _comb_audio(f0: float, n_harm: int = 20, channels: int = 2, amp: float = 0.05) -> np.ndarray:
    t = np.arange(T, dtype=np.float64) / SR
    x = sum(amp / k * np.sin(2 * np.pi * k * f0 * t + 0.3 * k) for k in range(1, n_harm + 1))
    return np.tile(np.asarray(x, dtype=np.float32), (channels, 1))


def _mag(audio_ch: np.ndarray) -> np.ndarray:
    """Hann-windowed |rfft| of one channel — 1 Hz bins over the 1 s chunk."""
    return np.abs(np.fft.rfft(audio_ch * np.hanning(audio_ch.shape[-1])))


def _band_energy(mag: np.ndarray, f_center: float, half_hz: float) -> float:
    freqs = np.fft.rfftfreq(T, 1.0 / SR)
    band = (freqs >= f_center - half_hz) & (freqs <= f_center + half_hz)
    return float(np.sum(mag[band] ** 2))


def _peak_bin(mag: np.ndarray, f_center: float, half_hz: float = 10.0) -> int:
    freqs = np.fft.rfftfreq(T, 1.0 / SR)
    idx = np.flatnonzero((freqs >= f_center - half_hz) & (freqs <= f_center + half_hz))
    return int(idx[np.argmax(mag[idx])])


def _db(x: float) -> float:
    return 10.0 * np.log10(x + 1e-30)


def test_spectral_recolor_changes_magnitude_not_peaks():
    audio = _comb_audio(60.0)
    out = _spectral_recolor(audio, {}, np.random.default_rng(1), sample_rate=SR)
    mag_in, mag_out = _mag(audio[0]), _mag(out[0])
    ratios_db = []
    for k in range(1, 11):
        assert _peak_bin(mag_out, 60.0 * k) == _peak_bin(mag_in, 60.0 * k), k
        ratios_db.append(
            _db(_band_energy(mag_out, 60.0 * k, 3.0)) - _db(_band_energy(mag_in, 60.0 * k, 3.0))
        )
    # Magnitudes moved (some tooth recolored by > 2 dB), and not uniformly.
    assert max(abs(r) for r in ratios_db) > 2.0
    assert max(ratios_db) - min(ratios_db) > 1.0


def test_random_reverb_preserves_rms_and_peaks():
    audio = _comb_audio(60.0)
    out = _random_reverb(audio, {}, np.random.default_rng(2), sample_rate=SR)
    rms_in = np.sqrt(np.mean(audio**2))
    rms_out = np.sqrt(np.mean(out**2))
    assert abs(20.0 * np.log10(rms_out / rms_in)) < 1.0  # renormalized
    mag_in, mag_out = _mag(audio[0]), _mag(out[0])
    for k in (1, 2, 3, 5, 8):
        assert abs(_peak_bin(mag_out, 60.0 * k) - _peak_bin(mag_in, 60.0 * k)) <= 1, k


@pytest.mark.parametrize("drr_db", [3.0, 9.0, 15.0])
def test_random_reverb_honours_drr_and_rt60(drr_db):
    # An impulse in every channel reads the RIR back: direct tap / tail energy
    # is the configured DRR, the tail spans the configured RT60, and the tails
    # are independent per channel.
    impulse = np.zeros((3, T), dtype=np.float32)
    impulse[:, 0] = 1.0
    params = {"drr_db_low": drr_db, "drr_db_high": drr_db, "rt60_low": 0.25, "rt60_high": 0.25}
    out = _random_reverb(impulse, params, np.random.default_rng(5), sample_rate=SR).astype(
        np.float64
    )
    n_tail = int(round(0.25 * SR))
    for ch in range(3):
        measured = 10.0 * np.log10(out[ch, 0] ** 2 / np.sum(out[ch, 1:] ** 2))
        assert abs(measured - drr_db) < 1e-3, (ch, measured)
        # The tail is round(RT60 * sr) samples: its last 10 % still carries
        # energy (-54..-60 dB), past it only FFT round-off remains.
        total = np.sum(out[ch, 1:] ** 2)
        assert np.sum(out[ch, int(0.9 * n_tail) : n_tail + 1] ** 2) > 1e-7 * total
        assert np.max(np.abs(out[ch, n_tail + 1 :])) < 1e-9 * abs(out[ch, 0])
    assert not np.allclose(out[0, 1:], out[1, 1:])


def test_random_reverb_drr_draw_spans_the_configured_range():
    impulse = np.zeros((1, T), dtype=np.float32)
    impulse[:, 0] = 1.0
    params = {"drr_db_low": 3.0, "drr_db_high": 15.0, "rt60_low": 0.1, "rt60_high": 0.3}
    measured = []
    for seed in range(64):
        out = _random_reverb(impulse, params, np.random.default_rng(seed), sample_rate=SR)
        x = out[0].astype(np.float64)
        measured.append(10.0 * np.log10(x[0] ** 2 / np.sum(x[1:] ** 2)))
    assert 3.0 - 1e-3 <= min(measured) < 5.0 and 13.0 < max(measured) <= 15.0 + 1e-3


def _noise_frame(t_start: float = 3.25) -> td.Frame:
    audio = td.uniform(_comb_audio(60.0), SR, dims=("mic", "time"), t_start=t_start)
    t_m = t_start + np.linspace(0.0, 0.99, 50)
    rps = td.events(t_m, np.full((4, 50), 60.0), dims=("rotor", "time"), t_start=t_start)
    return td.Frame({"audio": audio, "rps": rps})


@pytest.mark.parametrize("name", ["spectral_recolor", "random_reverb"])
def test_a_fire_replaces_only_the_audio(name):
    frame = _noise_frame()
    spec = {"probability": 1.0, "choices": [{name: {}}]}
    out = maybe_apply_noise_augmentation(frame, spec, np.random.default_rng(7), sample_rate=SR)

    # Labels bit-identical (values and stamps).
    assert out["rps"].tindex == frame["rps"].tindex
    np.testing.assert_array_equal(np.asarray(out["rps"].data), np.asarray(frame["rps"].data))
    assert out["audio"].tindex == frame["audio"].tindex  # same length, rate, timeline
    assert not np.array_equal(np.asarray(out["audio"].data), np.asarray(frame["audio"].data))


def test_no_fire_returns_the_frame_and_removed_names_raise():
    frame = _noise_frame()
    rng = np.random.default_rng(8)
    assert maybe_apply_noise_augmentation(frame, None, rng, sample_rate=SR) is frame
    spec = {"probability": 0.0, "choices": ["spectral_recolor"]}
    assert maybe_apply_noise_augmentation(frame, spec, rng, sample_rate=SR) is frame
    with pytest.raises(ValueError, match="removed in the rp cutover"):
        maybe_apply_noise_augmentation(
            frame, {"probability": 1.0, "choices": ["spec_mask"]}, rng, sample_rate=SR
        )
