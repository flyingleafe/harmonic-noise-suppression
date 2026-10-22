"""Tests for the analytic static-comb rotor-spectral noise model (E8)."""

from __future__ import annotations

import numpy as np
import tdseries as td

from data_processing.online_mixing import build_noise_stream
from data_processing.rotor_spectral_model import (
    DiverseCombRanges,
    ProfileRanges,
    StaticCombNoisePool,
    _comb_waveform,
    render_diverse_comb,
    sample_diverse_rotor,
    sample_profile,
)


def test_profile_enforces_min_fraction_above_floor():
    rng = np.random.default_rng(0)
    for _ in range(50):
        prof = sample_profile(
            rng,
            ProfileRanges(),
            n_harmonics=100,
            ref_rps=80.0,
            sample_rate=16000,
            min_harm_above_floor=0.30,
        )
        assert prof.frac_above_floor >= 0.30 - 1e-9
        assert prof.a_k.shape == (100,)
        assert abs(prof.a_k[0] - 1.0) < 1e-6  # fundamental normalised to 1


def test_comb_amplitudes_are_static_in_time():
    # A constant-RPS comb must not drift in per-harmonic magnitude over time —
    # this is the whole point (no amplitude->RPS shortcut).
    rng = np.random.default_rng(1)
    sr = 16000
    prof = sample_profile(
        rng,
        ProfileRanges(),
        n_harmonics=100,
        ref_rps=90.0,
        sample_rate=sr,
        min_harm_above_floor=0.30,
    )
    w = _comb_waveform(np.full(sr, 90.0), prof.a_k, sr, rng)
    win = np.hanning(1024)
    cols = np.array([np.abs(np.fft.rfft(w[i : i + 1024] * win)) for i in range(0, sr - 1024, 512)])
    cov = cols.std(0) / (cols.mean(0) + 1e-9)
    assert float(np.median(cov)) < 0.1  # magnitudes essentially constant across frames


def test_static_comb_pool_yields_wellformed_timeframe():
    pool = StaticCombNoisePool(
        sample_rate=16000, duration_s=0.5, n_harmonics=64, n_mics=8, n_rotors=4, seed=3
    )
    tf = pool.sample_timeframe(np.random.default_rng(0), 0.5)
    assert isinstance(tf, td.Frame)
    audio = tf["audio"]
    assert audio.data.shape == (8, int(round(0.5 * 16000)))
    assert np.isfinite(np.asarray(audio.data)).all()
    rps = np.asarray(tf["rps"].data)
    assert rps.shape[0] == 4
    assert rps.min() >= 20.0 and rps.max() <= 200.0


def test_build_noise_stream_dispatches_static_comb():
    stream, ceiling = build_noise_stream(
        {"kind": "static_comb", "n_harmonics": 32, "n_mics": 8, "n_rotors": 4},
        sample_rate=16000,
        window_s=0.5,
        seed=0,
    )

    frame = next(iter(stream))
    assert frame["audio"].data.shape == (8, int(round(0.5 * 16000)))
    assert ceiling == 8


# ── RPS-amplitude scaling + full-flight windowing (physical plausibility) ──────


def test_full_flight_windows_reach_low_and_zero_rps():
    """full_flight windowing visits warm-up/ground; intermittent stays at cruise."""
    ff = StaticCombNoisePool(
        sample_rate=16000,
        duration_s=1.0,
        n_harmonics=48,
        n_mics=2,
        rps_kind="full_flight",
        flight_reuse=64,
        seed=0,
    )
    rng = np.random.default_rng(1)
    means = np.array([ff.render(rng, 1.0)[1].mean() for _ in range(150)])
    assert means.min() < 20.0  # some warm-up/ground windows
    assert means.max() > 60.0  # some cruise windows

    cruise = StaticCombNoisePool(
        sample_rate=16000, duration_s=1.0, n_harmonics=48, n_mics=2, seed=0
    )
    rng2 = np.random.default_rng(1)
    cmeans = np.array([cruise.render(rng2, 1.0)[1].mean() for _ in range(40)])
    assert cmeans.min() > 40.0  # intermittent never leaves the hover regime


def test_zero_rps_segment_is_silent():
    """Within a window straddling the ground (rps=0) and rotors-on (rps>0), the
    zero-RPS part is ~silent (amplitude ~ rps^p, so rps=0 => no sound)."""
    ff = StaticCombNoisePool(
        sample_rate=16000,
        duration_s=3.0,
        n_harmonics=48,
        n_mics=2,
        rps_kind="full_flight",
        flight_reuse=2,
        seed=0,
    )
    rng = np.random.default_rng(0)
    for _ in range(300):
        audio, rps, _ = ff.render(rng, 3.0)
        mono = audio.mean(0)
        mrps = rps.mean(0)
        zero = mrps < 1.0
        on = mrps > 25.0  # rotors on (idle/ramp/cruise)
        if zero.sum() > 300 and on.sum() > 300:
            rms_zero = float(np.sqrt(np.mean(mono[zero] ** 2)))
            rms_on = float(np.sqrt(np.mean(mono[on] ** 2)))
            assert rms_zero < 0.05 * rms_on  # ground part is essentially silent
            return
    raise AssertionError("no window straddling zero-RPS and rotors-on was drawn")


def test_amp_scaling_config_wired():
    cfg = {
        "kind": "static_comb",
        "n_harmonics": 32,
        "n_mics": 2,
        "amp_rps_exponent": 3.0,
        "amp_rps_ref": 70.0,
        "rps": {"kind": "full_flight", "flight_reuse": 8},
    }
    pool = StaticCombNoisePool.from_config(cfg, duration_s=1.0, sample_rate=16000)
    assert pool.amp_rps_exponent == 3.0 and pool.amp_rps_ref == 70.0
    assert pool.rps_kind == "full_flight" and pool.flight_reuse == 8
    # dispatch through build_noise_stream still renders static_comb chunks.
    stream, _ = build_noise_stream(cfg, sample_rate=16000, window_s=1.0, seed=0)

    frame = next(iter(stream))
    assert frame["audio"].data.shape[-1] == 16000


# ── The diverse static comb ───────────────────────────────────────────────────


def test_diverse_comb_floor_buries_the_drawn_share_as_heard_in_the_audio():
    """Measured from the rendered audio — line amplitudes by projection, the
    floor from the residual — the white floor buries at most the drawn share
    of every rotor's lines at every microphone, and that share on the most
    buried one, where "buried" is a line under the floor in a 2048-point Hann
    spectrogram (a^2/2 < 3 sigma^2 / 2048). Rotor and microphone gains and the
    speed law all sit between the drawn profile and what is heard, so a slip in
    any of them moves the floor off the drawn share."""
    sr, dur, nfft, n_harm = 16000, 2.0, 2048, 60
    speeds = (50.0, 61.0)  # integer-Hz lines; lcm 3050 Hz lies above both combs' overlap
    ranges = DiverseCombRanges(drop_prob=(0.2, 0.2), rotor_gain_db=(-4.0, 4.0))
    rng = np.random.default_rng(11)
    n = int(sr * dur)
    rps = np.stack([np.full(n, s) for s in speeds])
    t = np.arange(n) / sr
    for subsumed in (0.0, 0.4, 0.85):
        profiles = [sample_diverse_rotor(rng, ranges, n_harmonics=n_harm) for _ in speeds]
        mic_gains = 10.0 ** (rng.uniform(-4.0, 0.0, size=(3, len(speeds))) / 20.0)
        audio, _, _ = render_diverse_comb(
            rps,
            profiles,
            mic_gains,
            subsumed,
            rng,
            sample_rate=sr,
            amp_rps_exponent=2.5,
            amp_rps_ref=80.0,
            band_taper_frac=0.0,
            floor_ref_nfft=nfft,
        )
        for x in audio:
            residual = x.copy()
            amps = []
            for s, prof in zip(speeds, profiles, strict=True):
                k = np.arange(1, n_harm + 1)[prof.kept]
                basis = np.exp(-2j * np.pi * np.outer(k * s, t))
                coef = 2.0 * (basis @ x) / n
                amps.append(np.abs(coef))
                residual -= np.real(coef[:, None] * np.conj(basis)).sum(axis=0)
            sigma2 = float(np.var(residual))
            buried = [float(np.mean(a**2 / 2 < 3.0 * sigma2 / nfft)) for a in amps]
            n_lines = min(a.size for a in amps)
            assert max(buried) <= subsumed + 2.0 / n_lines
            assert max(buried) >= subsumed - 2.0 / n_lines
