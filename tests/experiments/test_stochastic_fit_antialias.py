"""The render's anti-alias low-pass: what it must do to a tone and a transient.

The render-only filter (`stage2.antialias`) is applied ONCE by centred
overlap-add FFT convolution. These tests defend the four properties that
application was changed to get:

* the fit passband (up to 7900 Hz) survives within its stated ripple;
* the output Nyquist (8000 Hz and above) is attenuated by the stated 100 dB;
* the realized response is ``H``, not ``|H|^2`` — a single application;
* a transient stays where it was (zero phase, centred on the group delay) and
  short inputs are handled deterministically instead of raising.

The real clips' decimator (`clips.decimate`) is deliberately not touched by any
of this and is not exercised here.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.stochastic_fit.stage2 import (
    AA_PASS_HZ,
    AA_RIPPLE_DB,
    AA_STOP_DB,
    AA_STOP_HZ,
    antialias,
    antialias_filter,
)

NATIVE_SR = 44100.0


def _tone_gain_db(hz: float, *, sr: float = NATIVE_SR, seconds: float = 2.0) -> float:
    """Steady-state gain of the actual application, transients excluded."""
    n = int(sr * seconds)
    t = np.arange(n) / sr
    x = np.sin(2.0 * np.pi * hz * t)[None, :]
    y = antialias(x, sr)
    guard = int(sr * 0.25)
    inner = slice(guard, n - guard)
    rms_in = float(np.sqrt(np.mean(x[0, inner] ** 2)))
    rms_out = float(np.sqrt(np.mean(y[0, inner] ** 2)))
    return 20.0 * np.log10(max(rms_out, 1e-300) / rms_in)


def test_the_filter_is_designed_from_its_specification() -> None:
    f = antialias_filter(NATIVE_SR)
    assert f.taps.size % 2 == 1  # type-I linear phase
    assert f.delay == (f.taps.size - 1) // 2
    assert np.allclose(f.taps, f.taps[::-1])  # symmetric: exactly zero phase
    assert f.spec()["pass_hz"] == AA_PASS_HZ
    assert f.spec()["stop_hz"] == AA_STOP_HZ
    assert f.spec()["stop_db"] == AA_STOP_DB
    assert f.spec()["ripple_db"] == AA_RIPPLE_DB


@pytest.mark.parametrize("hz", [100.0, 1000.0, 4000.0, 7000.0, 7800.0, 7900.0])
def test_the_fit_passband_survives_within_its_ripple(hz: float) -> None:
    """7900 Hz is the fit's own band edge: the render must not eat into it."""
    assert abs(_tone_gain_db(hz)) <= AA_RIPPLE_DB


@pytest.mark.parametrize("hz", [8000.0, 8500.0, 9000.0, 12000.0, 20000.0])
def test_everything_at_and_above_the_output_nyquist_is_stopped(hz: float) -> None:
    assert _tone_gain_db(hz) <= -AA_STOP_DB


def test_one_application_not_a_squared_response() -> None:
    """At the cutoff a single pass reads -6 dB; a doubled application reads -12.

    This is the concrete difference from the previous ``filtfilt`` route, whose
    realized response was the square of the designed one.
    """
    cutoff = (AA_PASS_HZ + AA_STOP_HZ) / 2.0
    single = _tone_gain_db(cutoff)
    assert single == pytest.approx(-6.02, abs=0.15)
    n = int(NATIVE_SR * 2.0)
    t = np.arange(n) / NATIVE_SR
    x = np.sin(2.0 * np.pi * cutoff * t)[None, :]
    twice = antialias(antialias(x, NATIVE_SR), NATIVE_SR)
    guard = int(NATIVE_SR * 0.25)
    inner = slice(guard, n - guard)
    doubled = 20.0 * np.log10(
        float(np.sqrt(np.mean(twice[0, inner] ** 2))) / float(np.sqrt(np.mean(x[0, inner] ** 2)))
    )
    assert doubled == pytest.approx(2.0 * single, abs=0.2)


def test_a_transient_stays_where_it_was() -> None:
    """Zero phase: the centred slice is aligned sample-for-sample, so a click
    does not move and its response is symmetric about its own position."""
    n, at = 8001, 4000
    x = np.zeros((1, n))
    x[0, at] = 1.0
    y = antialias(x, NATIVE_SR)
    assert y.shape == x.shape
    assert int(np.argmax(np.abs(y[0]))) == at
    left = y[0, at - 500 : at]
    right = y[0, at + 1 : at + 501][::-1]
    assert np.allclose(left, right, atol=1e-15)


def test_a_transient_keeps_its_in_band_energy() -> None:
    """A band-limited click must survive: the filter removes only what is above
    the stop edge."""
    n = 16384
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(n)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1.0 / NATIVE_SR)
    spec[freqs > AA_PASS_HZ] = 0.0  # in-band only
    x = np.fft.irfft(spec, n)[None, :]
    y = antialias(x, NATIVE_SR)
    guard = 4000  # the filter's own edge transient at both ends
    kept = float(np.sum(y[0, guard:-guard] ** 2) / np.sum(x[0, guard:-guard] ** 2))
    assert kept == pytest.approx(1.0, abs=0.01)


def test_an_out_of_band_tone_cannot_fold_into_the_top_band() -> None:
    """The reason the render filters itself: a 9 kHz line must not appear in the
    decimated 7.5-8 kHz band."""
    n = int(NATIVE_SR * 1.0)
    t = np.arange(n) / NATIVE_SR
    x = np.sin(2.0 * np.pi * 9000.0 * t)[None, :]
    y = antialias(x, NATIVE_SR)
    band = lambda z, lo, hi: float(  # noqa: E731
        np.mean(
            np.abs(np.fft.rfft(z[0] * np.hanning(z.shape[-1])))[
                (np.fft.rfftfreq(z.shape[-1], 1.0 / NATIVE_SR) >= lo)
                & (np.fft.rfftfreq(z.shape[-1], 1.0 / NATIVE_SR) < hi)
            ]
            ** 2
        )
    )
    before, after = band(x, 8500.0, 9500.0), band(y, 8500.0, 9500.0)
    assert 10.0 * np.log10(after / before) <= -AA_STOP_DB


def test_short_inputs_are_deterministic_instead_of_an_error() -> None:
    """``filtfilt`` refuses anything shorter than ``3 * (n_taps - 1)``; the
    centred overlap-add route is defined for every length, including inputs
    shorter than the kernel."""
    f = antialias_filter(NATIVE_SR)
    assert f.taps.size > 64  # the kernel really is longer than these inputs
    for n in (1, 2, 64, 1000):
        x = np.ones((2, n))
        y = antialias(x, NATIVE_SR)
        assert y.shape == (2, n)
        assert np.all(np.isfinite(y))
        assert np.array_equal(y, antialias(x, NATIVE_SR))
    from scipy.signal import filtfilt

    with pytest.raises(ValueError):
        filtfilt(f.taps, [1.0], np.ones((2, 64)), axis=-1)


def test_multichannel_and_single_channel_agree() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 5000))
    multi = antialias(x, NATIVE_SR)
    for ch in range(3):
        assert np.allclose(multi[ch], antialias(x[ch], NATIVE_SR), atol=1e-12)


def test_an_impossible_specification_is_refused() -> None:
    with pytest.raises(ValueError, match="not inside the Nyquist band"):
        antialias_filter(8000.0)  # stop edge at 8000 Hz is the Nyquist itself
    with pytest.raises(ValueError, match="ripple_db must be positive"):
        antialias_filter(NATIVE_SR, 7900.0, 8000.0, 100.0, 0.0)


# ── the declared render transfer (AA x resample_poly) ───────────────────────


def test_the_declared_transfer_matches_the_measured_render_chain() -> None:
    """``render_transfer_power`` is the multiplier a candidate's expected
    spectrum carries, so it must equal what a tone actually loses going through
    ``antialias`` and the UNCHANGED ``clips.decimate``."""
    from experiments.stochastic_fit import clips as C
    from experiments.stochastic_fit.data import Clip
    from experiments.stochastic_fit.stage2 import render_transfer_power

    sr_work, sr_out = 64000, 16000
    freqs = np.fft.rfftfreq(16384, 1.0 / sr_out)
    transfer = render_transfer_power(freqs, sample_rate_work=sr_work, sample_rate_out=sr_out)
    assert transfer.shape == freqs.shape

    def measured_db(hz: float, seconds: float = 1.0) -> float:
        n = int(sr_work * seconds)
        t = np.arange(n) / sr_work
        x = np.sin(2.0 * np.pi * hz * t)[None, :]
        y = antialias(x, sr_work)
        clip = Clip("t", "synthetic", y.astype(np.float32), np.zeros((1, n)), sr_work)
        out = np.asarray(C.decimate(clip, sr_out).audio, dtype=np.float64)
        guard = int(sr_out * 0.2)
        return 20.0 * np.log10(np.sqrt(2.0 * np.mean(out[0, guard:-guard] ** 2)))

    for hz in (1000.0, 4000.0, 7000.0, 7500.0, 7900.0):
        i = int(np.argmin(np.abs(freqs - hz)))
        assert measured_db(hz) == pytest.approx(10.0 * np.log10(transfer[i]), abs=0.05)
        # explicit observation-accuracy gate: in-band stationary tone <= 0.1 dB
        assert abs(measured_db(hz) - 10.0 * np.log10(transfer[i])) <= 0.1


def test_moving_chirp_accuracy_against_the_known_transfer() -> None:
    """A chirp crossing the 7.9/8 kHz band edge is predicted by the known
    transfer with <= 2% normalized L1 error on interior-window support.

    The source-side work-grid periodogram is scaled by ``n_fft_analysis / n_fft_work``
    (= ``sr_analysis / sr_work``) so its units match the candidate kernel's
    analysis-grid output, then multiplied by the transfer once.
    """
    from experiments.stochastic_fit import clips as C
    from experiments.stochastic_fit.data import Clip, periodogram
    from experiments.stochastic_fit.stage2 import render_transfer_power

    sr_work, sr_out = 64000, 16000
    n_fft_out, hop_out = 16384, 1024
    n_fft_work, hop_work = 4 * n_fft_out, 4 * hop_out
    seconds = 1.6
    n = int(sr_work * seconds)
    t = np.arange(n) / sr_work
    f0, f1 = 7500.0, 8500.0
    phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) / seconds * t**2)
    x = np.sin(phase)[None, :]

    y = antialias(x, sr_work)
    clip = Clip("chirp", "synthetic", y.astype(np.float32), np.zeros((1, n)), sr_work)
    out = np.asarray(C.decimate(clip, sr_out).audio, dtype=np.float64)

    pg_out = periodogram(
        Clip("chirp_out", "synthetic", out.astype(np.float32), np.zeros((1, out.shape[1])), sr_out),
        n_fft=n_fft_out,
        hop=hop_out,
    )
    pg_work = periodogram(
        Clip("chirp_work", "synthetic", x.astype(np.float32), np.zeros((1, n)), sr_work),
        n_fft=n_fft_work,
        hop=hop_work,
    )

    freqs_work = np.asarray(pg_work.freqs, dtype=np.float64)
    transfer = render_transfer_power(freqs_work, sample_rate_work=sr_work, sample_rate_out=sr_out)
    # scale to analysis-grid units (candidate-kernel convention)
    scale = float(n_fft_out) / float(n_fft_work)
    expected = scale * np.asarray(pg_work.power, dtype=np.float64) * transfer[None, None, :]
    expected = expected[..., : n_fft_out // 2 + 1]

    n_frames = min(expected.shape[1], pg_out.power.shape[1])
    expected = expected[:, :n_frames, :]
    actual = np.asarray(pg_out.power, dtype=np.float64)[:, :n_frames, :]
    interior = slice(2, -2)
    expected = expected[:, interior, :]
    actual = actual[:, interior, :]

    band = (pg_out.freqs >= 30.0) & (pg_out.freqs <= 7900.0)
    num = float(np.abs(actual[..., band] - expected[..., band]).sum())
    den = float(np.abs(expected[..., band]).sum())
    l1 = num / den if den > 0 else float("inf")
    assert l1 <= 0.02, f"chirp L1 error {l1:.3%} exceeds 2% gate; worst-case actual-vs-expected"


def test_the_in_band_loss_is_the_decimator_not_the_antialias() -> None:
    """Attribution, because the two filters are physically different: the AA is
    flat through 7900 Hz, and the ~4.9 dB there is ``resample_poly``'s own
    rolloff."""
    from scipy.signal import freqz

    from experiments.stochastic_fit.stage2 import render_transfer_power

    sr_work, sr_out = 64000, 16000
    aa = antialias_filter(float(sr_work))
    w = 2.0 * np.pi * 7900.0 / sr_work
    aa_db = 20.0 * np.log10(abs(freqz(aa.taps, worN=[w])[1][0]))
    total_db = 10.0 * np.log10(
        render_transfer_power(np.array([7900.0]), sample_rate_work=sr_work, sample_rate_out=sr_out)[
            0
        ]
    )
    assert abs(aa_db) <= AA_RIPPLE_DB  # the AA costs nothing at the band edge
    assert total_db == pytest.approx(-4.9, abs=0.2)  # the decimator costs it all
