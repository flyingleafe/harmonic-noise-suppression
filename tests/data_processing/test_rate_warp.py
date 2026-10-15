"""Tests for the native-rate time-varying rate warp (``data_processing.rate_warp``).

The label transform is checked against the AUDIO it is paired with: a source
tone whose instantaneous frequency is ``K * r(s)`` must come out at
``K * r_tilde(t)`` for the emitted label ``r_tilde`` — a wrong source-time
map ``tau`` or a missing ``alpha0`` factor breaks that by hundreds of Hz.
"""

from __future__ import annotations

import numpy as np
import pytest
import tdseries as td

from data_processing.rate_warp import (
    RateWarpParams,
    apply_rate_warp,
    sample_rate_warp_params,
    source_window_s,
)

FS = 44_100  # native noise rate (DREGON, MD2)
SR = 16_000  # model rate
K = 40  # harmonic carried by the synthetic tone: f(t) = K * r(t)
PROTOCOL = {
    "alpha_low": 0.5,
    "alpha_high": 1.5,
    "dev_const": 0.08,
    "dev_sine": 0.04,
    "f_low": 0.1,
    "f_high": 1.0,
}


def _frame(audio: np.ndarray, rps_fn, duration_s: float, sr: int = FS) -> td.Frame:
    """A native-rate noise Frame: ``(C, N)`` audio + an ``rps`` event track."""
    t_m = np.linspace(0.0, duration_s - 0.005, 400)
    values = np.stack([rps_fn(t_m) + 2.0 * r for r in range(4)])
    return td.Frame(
        {
            "audio": td.uniform(audio.astype(np.float32), sr, dims=("mic", "time"), t_start=0.0),
            "rps": td.events(t_m, values, dims=("rotor", "time"), t_start=0.0, t_end=duration_s),
        }
    )


def _chirp_frame(duration_s: float = 4.0) -> td.Frame:
    """Two channels (ch1 = -0.5 * ch0) of a tone at K * r(s), r(s) = 40 + 6 s."""

    def r(s):
        return 40.0 + 6.0 * s

    s = np.arange(int(duration_s * FS)) / FS
    phase = 2.0 * np.pi * K * (40.0 * s + 3.0 * s**2)  # integral of K * r
    tone = 0.1 * np.sin(phase)
    return _frame(np.stack([tone, -0.5 * tone]), r, duration_s)


def _peak_track(x: np.ndarray, n_fft: int = 2048, hop: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Parabolic-interpolated peak frequency (Hz) per Hann frame, and frame centres (s)."""
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * np.hanning(n_fft)
    mag = np.abs(np.fft.rfft(frames, axis=-1))
    rows = np.arange(mag.shape[0])
    k = np.clip(np.argmax(mag, axis=-1), 1, mag.shape[1] - 2)
    a, b, c = (np.log(mag[rows, k + d] + 1e-12) for d in (-1, 0, 1))
    delta = 0.5 * (a - c) / (a - 2.0 * b + c)
    return (k + delta) * SR / n_fft, (rows * hop + n_fft / 2) / SR


def _band_power(x: np.ndarray, lo: float, hi: float) -> float:
    spec = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(x.shape[-1], 1.0 / SR)
    return float(spec[..., (freqs >= lo) & (freqs < hi)].mean())


@pytest.mark.parametrize(
    "params",
    [
        RateWarpParams(alpha0=0.6, c=1.0, a=0.0, f=0.5, phi=0.0),
        RateWarpParams(alpha0=1.4, c=1.0, a=0.0, f=0.5, phi=0.0),
        RateWarpParams(alpha0=0.7, c=1.06, a=0.04, f=0.9, phi=1.3),
        None,  # the un-warped native path: decimation only
    ],
)
def test_warped_audio_frequency_matches_k_times_the_emitted_label(params):
    out = apply_rate_warp(_chirp_frame(), params, target_len=2 * SR, sample_rate=SR)

    audio = np.asarray(out["audio"].data)
    assert audio.shape == (2, 2 * SR) and audio.dtype == np.float32
    # One resampling map for every channel: the channel relation survives.
    np.testing.assert_allclose(audio[1], -0.5 * audio[0], atol=1e-6)

    label = np.asarray(out["rps"].data, dtype=np.float64)  # (4, L) @ 100 Hz from t = 0
    t_label = np.arange(label.shape[1]) / 100.0
    peak_hz, t_frames = _peak_track(audio[0])
    expected_hz = K * np.interp(t_frames, t_label, label[0])
    # The tone sweeps; a 128 ms frame reads its centre to well under 1 Hz
    # (a missing alpha0 or alpha(t) factor is off by >= 50 Hz).
    np.testing.assert_allclose(peak_hz, expected_hz, rtol=0.0, atol=2.0)
    # Every rotor carries the same transform (source offsets 0/2/4/6 rev/s).
    alpha = (params or RateWarpParams(1.0, 1.0, 0.0, 1.0, 0.0)).alpha(t_label)
    for r in range(1, 4):
        np.testing.assert_allclose(label[r] - label[0], 2.0 * r * alpha, atol=1e-4)


def test_constant_alpha0_scales_constant_labels_exactly():
    base = np.array([40.0, 50.0, 60.0, 70.0])
    noise = np.random.default_rng(0).standard_normal((1, 3 * FS))
    frame = _frame(noise, lambda t: np.full_like(t, base[0]), 3.0)
    for alpha0 in (0.5, 0.83, 1.5):
        params = RateWarpParams(alpha0=alpha0, c=1.0, a=0.0, f=0.3, phi=0.0)
        label = np.asarray(
            apply_rate_warp(frame, params, target_len=SR, sample_rate=SR)["rps"].data
        )
        expected = alpha0 * (base[0] + 2.0 * np.arange(4))
        np.testing.assert_allclose(
            label, np.tile(expected[:, None], (1, label.shape[1])), atol=1e-4
        )


def test_alpha_below_one_leaves_no_silent_band_below_nyquist():
    # White native noise warped by 0.5: the model band 0-8 kHz comes from the
    # source's 0-16 kHz, so 6-8 kHz must be as full as 4-6 kHz (a 16 kHz-domain
    # resampling would leave [4, 8] kHz empty).
    noise = np.random.default_rng(1).standard_normal((2, 6 * FS))
    frame = _frame(noise, lambda t: np.full_like(t, 60.0), 6.0)
    params = RateWarpParams(alpha0=0.5, c=1.0, a=0.0, f=0.3, phi=0.0)
    audio = np.asarray(
        apply_rate_warp(frame, params, target_len=2 * SR, sample_rate=SR)["audio"].data
    )

    ratio = _band_power(audio, 6000.0, 8000.0) / _band_power(audio, 4000.0, 6000.0)
    assert ratio > 0.7, ratio  # soxr's own passband edge (~7.5 kHz) is the only roll-off
    assert _band_power(audio, 6000.0, 7500.0) / _band_power(audio, 4000.0, 6000.0) > 0.9


def test_stream_window_covers_every_draw_and_a_short_window_raises():
    duration_s = 2.0
    window_s = source_window_s(PROTOCOL, duration_s)
    noise = np.random.default_rng(2).standard_normal((1, int(window_s * FS)))
    frame = _frame(noise, lambda t: np.full_like(t, 60.0), window_s)
    rng = np.random.default_rng(3)
    draws = [sample_rate_warp_params(PROTOCOL, rng) for _ in range(200)]
    # The extreme corner: largest alpha0 and c, full sine depth, any phase.
    draws += [
        RateWarpParams(alpha0=1.5, c=1.08, a=0.04, f=f, phi=phi)
        for f in (0.1, 1.0)
        for phi in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    ]
    for params in draws:
        out = apply_rate_warp(frame, params, target_len=int(duration_s * SR), sample_rate=SR)
        assert out["audio"].dim_size("time") == int(duration_s * SR)

    n_short = int(0.9 * 1.5 * 1.12 * duration_s * FS)
    short = _frame(noise[:, :n_short], lambda t: np.full_like(t, 60.0), 3.0)
    with pytest.raises(ValueError, match="beyond"):
        apply_rate_warp(
            short,
            RateWarpParams(alpha0=1.5, c=1.08, a=0.04, f=0.1, phi=0.0),
            target_len=int(duration_s * SR),
            sample_rate=SR,
        )


def test_model_rate_input_is_rejected():
    frame = _frame(np.zeros((1, 2 * SR)), lambda t: np.full_like(t, 60.0), 2.0, sr=SR)
    with pytest.raises(ValueError, match="native-rate"):
        apply_rate_warp(frame, None, target_len=SR, sample_rate=SR)


def test_parameter_draws_stay_in_range_and_follow_the_spec():
    rng = np.random.default_rng(4)
    t = np.linspace(0.0, 2.0, 400)
    for _ in range(500):
        p = sample_rate_warp_params(PROTOCOL, rng)
        assert 0.5 <= p.alpha0 <= 1.5 and 0.92 <= p.c <= 1.08 and 0.0 <= p.a <= 0.04
        assert 0.1 <= p.f <= 1.0 and 0.0 <= p.phi < 2.0 * np.pi
        alpha = p.alpha(t)
        assert np.all(alpha >= 0.5 * 0.88 - 1e-12) and np.all(alpha <= 1.5 * 1.12 + 1e-12)
        # tau is the integral of alpha (closed form vs cumulative trapezoid).
        tau_num = np.concatenate([[0.0], np.cumsum(0.5 * (alpha[1:] + alpha[:-1]) * np.diff(t))])
        np.testing.assert_allclose(p.tau(t), tau_num, atol=1e-5)
    with pytest.raises(ValueError, match="missing"):
        sample_rate_warp_params({"alpha_low": 0.5}, rng)
