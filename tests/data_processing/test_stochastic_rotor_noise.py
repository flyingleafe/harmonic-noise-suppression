"""Tests for the stochastic rotor-noise model."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from data_processing import stochastic_rotor_noise as srn

SR = 16000


def _params(seed: int = 0, **overrides):
    params = srn.sample_params(
        np.random.default_rng(seed), n_rotors=4, n_harmonics=40, sample_rate=SR
    )
    return params.with_(**overrides) if overrides else params


def _cruise(n_seconds: float = 2.0, speed: float = 80.0) -> np.ndarray:
    n = int(n_seconds * SR)
    return np.tile(np.array([[speed]]), (4, n)) + np.array([0.0, 1.5, -2.0, 0.7])[:, None]


def _power_spectrum(x: np.ndarray, n_fft: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    hop = n_fft // 4
    window = np.hanning(n_fft + 1)[:n_fft]
    n_frames = (x.size - n_fft) // hop
    frames = np.stack([x[i * hop : i * hop + n_fft] * window for i in range(n_frames)])
    power = (np.abs(np.fft.rfft(frames, axis=-1)) ** 2).mean(axis=0)
    return power, np.fft.rfftfreq(n_fft, 1.0 / SR)


def test_gp_statistics_match_the_requested_kernel():
    rng = np.random.default_rng(0)
    series = srn.sample_gp(rng, 400, 256, dt=0.05, tau=1.0, std=3.0)
    assert series.shape == (400, 256)
    assert np.std(series) == pytest.approx(3.0, rel=0.1)
    # A process with a one-second correlation time and a 50 ms step is strongly
    # correlated between adjacent samples and weakly correlated 40 steps out.
    near = np.corrcoef(series[:, :-1].ravel(), series[:, 1:].ravel())[0, 1]
    far = np.corrcoef(series[:, :-40].ravel(), series[:, 40:].ravel())[0, 1]
    assert near > 0.99
    assert abs(far) < 0.3


def test_zero_std_gives_a_constant_process():
    out = srn.sample_gp(np.random.default_rng(0), 3, 64, dt=0.1, tau=1.0, std=0.0)
    assert np.all(out == 0.0)


def test_realized_spectrum_follows_the_model():
    params = _params(1)
    audio, diag = srn.synthesize(params, _cruise(), rng=np.random.default_rng(2), n_mics=1)
    power, freqs = _power_spectrum(audio[0].astype(np.float64))
    model = 10.0 ** (srn.model_psd_db(diag, 0).mean(axis=0) / 10.0)
    band = (freqs > 60.0) & (freqs < 7000.0)
    realized_db = 10.0 * np.log10(power[band] / power[band].max())
    model_db = 10.0 * np.log10(model[band] / model[band].max())
    assert np.corrcoef(realized_db, model_db)[0, 1] > 0.95


def test_the_comb_sits_at_multiples_of_the_rotor_speed():
    # One rotor, a clean profile, and a floor far below it: the spectrum's peaks
    # must land on multiples of the speed.
    params = _params(3, floor_mean_db=-45.0).with_(n_rotors=1)
    params = params.with_(profile_db=params.profile_db[:1], gamma0=params.gamma0[:1])
    params = params.with_(gamma_slope=params.gamma_slope[:1] * 0.0 + 0.1)
    speed = 70.0
    rps = np.full((1, 2 * SR), speed)
    audio, _ = srn.synthesize(params, rps, rng=np.random.default_rng(4), n_mics=1)
    power, freqs = _power_spectrum(audio[0].astype(np.float64))
    band = (freqs > 50.0) & (freqs < 3000.0)
    peak_freqs = freqs[band][power[band] > np.percentile(power[band], 99.0)]
    residual = np.abs(peak_freqs / speed - np.round(peak_freqs / speed))
    assert np.median(residual) < 0.12


def test_amplitudes_carry_no_speed_information():
    # The same parameters under two different constant speeds must give the
    # same harmonic levels: the model's amplitudes are drawn independently of
    # the trajectory, which is what keeps the amplitude shortcut closed.
    params = _params(5, amp_rps_exponent=0.0)
    freqs = np.fft.rfftfreq(2048, 1.0 / SR)
    slow = srn.build_psd(
        params, np.full((4, 60), 60.0), freqs, dt=0.032, rng=np.random.default_rng(6)
    )
    fast = srn.build_psd(
        params, np.full((4, 60), 95.0), freqs, dt=0.032, rng=np.random.default_rng(6)
    )
    assert slow["lines"].sum() == pytest.approx(fast["lines"].sum(), rel=0.05)
    assert np.allclose(slow["floor"], fast["floor"])


def test_stopped_rotors_are_silent():
    params = _params(7)
    rps = np.zeros((4, SR))
    audio, _ = srn.synthesize(
        params, rps, rng=np.random.default_rng(8), n_mics=1, normalize_rms=None
    )
    assert float(np.abs(audio).max()) < 1e-6


def test_line_power_scales_with_the_level_knob():
    params = _params(9)
    freqs = np.fft.rfftfreq(2048, 1.0 / SR)
    rps = np.full((4, 40), 80.0)
    base = srn.build_psd(params, rps, freqs, dt=0.032, rng=np.random.default_rng(10))
    louder = srn.build_psd(
        params.with_(harm_mean_db=params.harm_mean_db + 10.0),
        rps,
        freqs,
        dt=0.032,
        rng=np.random.default_rng(10),
    )
    assert louder["lines"].sum() / base["lines"].sum() == pytest.approx(10.0, rel=0.02)


def test_floor_shape_is_smooth():
    params = _params(11)
    freqs = np.linspace(30.0, 8000.0, 4000)
    shape = srn.floor_shape_db(params, freqs)
    # No step in the interpolated curve: the largest jump between neighbouring
    # points stays far below the curve's own range.
    assert np.abs(np.diff(shape)).max() < 0.1 * (shape.max() - shape.min())


def test_pool_emits_a_usable_frame():
    pool = srn.StochasticNoisePool(
        sample_rate=SR, duration_s=1.0, n_harmonics=30, n_mics=2, n_rotors=4
    )
    frame = pool.sample_timeframe(np.random.default_rng(12), 1.0)
    audio = np.asarray(frame["audio"].data)
    rps = np.asarray(frame["rps"].data)
    assert audio.shape == (2, SR)
    assert rps.shape[0] == 4
    assert np.isfinite(audio).all()
    assert float(np.abs(audio).max()) > 0.0


def test_pool_is_seed_reproducible():
    pool = srn.StochasticNoisePool(sample_rate=SR, duration_s=0.5, n_harmonics=20, n_mics=1)
    a = pool.sample_timeframe(np.random.default_rng(13), 0.5)
    b = pool.sample_timeframe(np.random.default_rng(13), 0.5)
    assert np.array_equal(np.asarray(a["audio"].data), np.asarray(b["audio"].data))


def test_pool_draws_a_fresh_parameter_set_per_window():
    pool = srn.StochasticNoisePool(sample_rate=SR, duration_s=0.5, n_harmonics=20, n_mics=1)
    rng = np.random.default_rng(14)
    first = pool.render(rng, 0.5)[2]
    second = pool.render(rng, 0.5)[2]
    assert not np.allclose(first.profile_db, second.profile_db)


def test_full_flight_windows_move_along_one_flight():
    pool = srn.StochasticNoisePool(
        sample_rate=SR, duration_s=1.0, n_harmonics=20, n_mics=1, rps_kind="full_flight"
    )
    rng = np.random.default_rng(15)
    windows = [pool.sample_rps(rng, 1.0) for _ in range(12)]
    # One flight is cached and windowed, so the windows differ and the flight
    # itself runs from the ground to cruise.
    assert len({round(float(w.mean()), 3) for w in windows}) > 6
    assert pool._flight is not None
    assert float(pool._flight.rps.min()) < 5.0
    assert float(pool._flight.rps.max()) > 60.0


def test_registered_as_an_online_mix_engine():
    from data_processing.online_mixing import _build_engine

    engine = _build_engine(
        {"kind": "stochastic", "n_harmonics": 20, "n_mics": 1}, window_s=0.5, sample_rate=SR
    )
    assert isinstance(engine, srn.StochasticNoisePool)
    assert engine.n_harmonics == 20


def test_rps_scale_range_moves_the_whole_trajectory():
    slow = srn.StochasticNoisePool(
        sample_rate=SR, duration_s=0.5, n_harmonics=20, n_mics=1, rps_scale_range=(0.5, 0.5)
    )
    plain = srn.StochasticNoisePool(sample_rate=SR, duration_s=0.5, n_harmonics=20, n_mics=1)
    a = slow.sample_rps(np.random.default_rng(20), 0.5)
    b = plain.sample_rps(np.random.default_rng(20), 0.5)
    # The same draw, halved: a stopped rotor stays stopped and every other
    # speed halves, so the comb moves with the label.
    assert np.allclose(a, 0.5 * b)


def test_rps_scale_range_spreads_the_speed_prior():
    pool = srn.StochasticNoisePool(
        sample_rate=SR, duration_s=0.5, n_harmonics=20, n_mics=1, rps_scale_range=(0.6, 1.5)
    )
    rng = np.random.default_rng(21)
    means = np.array([float(pool.sample_rps(rng, 0.5).mean()) for _ in range(24)])
    assert means.max() / max(means.min(), 1e-6) > 1.8


def test_normalize_rms_range_spreads_the_output_level():
    pool = srn.StochasticNoisePool(
        sample_rate=SR,
        duration_s=0.5,
        n_harmonics=20,
        n_mics=1,
        normalize_rms=(0.005, 0.2),
    )
    rng = np.random.default_rng(22)
    levels = []
    for _ in range(16):
        audio = pool.render(rng, 0.5)[0]
        levels.append(float(np.sqrt(np.mean(np.square(audio)))))
    levels = np.array(levels)
    assert levels.min() >= 0.004
    assert levels.max() <= 0.21
    assert levels.max() / levels.min() > 5.0


def test_scalar_normalize_rms_is_exact():
    pool = srn.StochasticNoisePool(
        sample_rate=SR, duration_s=0.5, n_harmonics=20, n_mics=1, normalize_rms=0.04
    )
    audio = pool.render(np.random.default_rng(23), 0.5)[0]
    assert float(np.sqrt(np.mean(np.square(audio)))) == pytest.approx(0.04, rel=1e-4)


@pytest.mark.parametrize("speed", [95.0, 130.0, 220.0])
def test_harmonics_above_nyquist_do_not_escape_the_grid(speed):
    # k * rps runs far past Nyquist at the top of the speed range. Those lines
    # carry no power, and they must not carry an index either: an out-of-range
    # scatter used to make bincount return a longer array than the accumulator.
    params = _params(30, harm_mean_db=0.0)
    rps = np.full((4, SR), speed)
    audio, _ = srn.synthesize(params, rps, rng=np.random.default_rng(31), n_mics=1)
    assert np.isfinite(audio).all()
    assert float(np.abs(audio).max()) > 0.0


def test_high_speed_pool_windows_render():
    pool = srn.StochasticNoisePool(
        sample_rate=SR,
        duration_s=1.0,
        n_harmonics=80,
        n_mics=2,
        rps_kind="full_flight",
        rps_scale_range=(0.4, 1.5),
        aggressiveness=(0.8, 2.5),
    )
    rng = np.random.default_rng(32)
    for _ in range(8):
        frame = pool.sample_timeframe(rng, 1.0)
        assert np.isfinite(np.asarray(frame["audio"].data)).all()


def test_flight_level_mode_keeps_the_silence_cue():
    # A real recorder holds one gain for a whole flight, so a stopped-rotor
    # window is quiet and a cruise window is loud. Normalizing every window
    # instead throws that away, and the silence cue with it.
    common: dict[str, Any] = dict(
        sample_rate=SR, duration_s=1.0, n_harmonics=30, n_mics=1, rps_kind="full_flight"
    )
    levels: dict[str, list[tuple[float, float]]] = {}
    for mode in ("window", "flight"):
        pool = srn.StochasticNoisePool(**common, level_mode=mode)
        rng = np.random.default_rng(40)
        rows = []
        for _ in range(40):
            audio, rps, _, _ = pool.render(rng, 1.0)
            rows.append((float(rps.mean()), float(np.sqrt(np.mean(np.square(audio))))))
        levels[mode] = rows

    def band(rows, lo, hi):
        vals = [r for speed, r in rows if lo <= speed < hi]
        return float(np.mean(vals)) if vals else float("nan")

    slow_w, fast_w = band(levels["window"], 8, 60), band(levels["window"], 60, 1e9)
    slow_f, fast_f = band(levels["flight"], 8, 60), band(levels["flight"], 60, 1e9)
    assert np.isfinite([slow_w, fast_w, slow_f, fast_f]).all()
    assert slow_w == pytest.approx(fast_w, rel=0.05)  # window mode: level says nothing
    assert fast_f / slow_f > 3.0  # flight mode: level says a great deal


def _line_flicker_db(audio: np.ndarray, speed: float, harmonics=(3, 8, 20)) -> float:
    """Mean frame-to-frame level standard deviation of the named harmonics, in dB."""
    power, freqs = _power_spectrum_frames(audio)
    df = float(freqs[1] - freqs[0])
    out = []
    for k in harmonics:
        track = power[:, int(round(k * speed / df))]
        out.append(float(np.std(10.0 * np.log10(track / track.mean()))))
    return float(np.mean(out))


def _power_spectrum_frames(x: np.ndarray, n_fft: int = 2048):
    hop = n_fft // 4
    window = np.hanning(n_fft + 1)[:n_fft]
    n_frames = (x.size - n_fft) // hop
    frames = np.stack([x[i * hop : i * hop + n_fft] * window for i in range(n_frames)])
    return np.abs(np.fft.rfft(frames, axis=-1)) ** 2, np.fft.rfftfreq(n_fft, 1.0 / SR)


def test_coherent_lines_are_steadier_only_while_they_stay_coherent():
    # Filtered noise and a phase-wandering tone share a power spectrum and
    # differ in their magnitude statistics — but only while the tone stays
    # coherent across an analysis frame. A line whose half width is tens of
    # hertz decoheres inside 128 ms and flickers either way, which is why the
    # difference is at low harmonics and vanishes at high ones.
    #
    # Measured on `free-flight_nosource_room1`, a REAL harmonic flickers 5.14,
    # 4.02, 4.26 and 4.11 dB at k = 3, 8, 20, 40. The stochastic mode gives 3.60
    # to 3.21 and the coherent mode 0.79 to 3.66, so the stochastic mode is the
    # one that matches a real recording and stays the default.
    params = _params(2, harm_gp_std_db=0.0, floor_mean_db=-60.0)
    params = params.with_(
        n_rotors=1,
        profile_db=params.profile_db[:1],
        gamma0=np.array([1.0]),
        gamma_slope=np.array([0.6]),
    )
    rps = np.full((1, 6 * SR), 80.0)
    flicker = {}
    for mode in ("stochastic", "coherent"):
        audio, _ = srn.synthesize(
            params, rps, rng=np.random.default_rng(3), n_mics=1, line_mode=mode
        )
        flicker[mode] = _line_flicker_db(audio[0].astype(np.float64), 80.0, harmonics=(3, 5))
    assert flicker["stochastic"] > 2.5
    assert flicker["coherent"] < 0.6 * flicker["stochastic"]


def test_the_two_line_modes_share_a_spectrum():
    params = _params(5)
    rps = np.tile(np.array([[80.0], [82.0], [78.0], [81.0]]), (1, 4 * SR))
    spectra = {}
    for mode in ("stochastic", "coherent"):
        audio, _ = srn.synthesize(
            params, rps, rng=np.random.default_rng(6), n_mics=1, line_mode=mode
        )
        power, freqs = _power_spectrum_frames(audio[0].astype(np.float64))
        spectra[mode] = 10.0 * np.log10(power.mean(axis=0) + 1e-30)
    band = (freqs > 60.0) & (freqs < 7000.0)
    assert np.corrcoef(spectra["stochastic"][band], spectra["coherent"][band])[0, 1] > 0.98


def test_coherent_mode_runs_through_the_pool():
    pool = srn.StochasticNoisePool(
        sample_rate=SR, duration_s=1.0, n_harmonics=40, n_mics=2, line_mode="coherent"
    )
    frame = pool.sample_timeframe(np.random.default_rng(50), 1.0)
    audio = np.asarray(frame["audio"].data)
    assert audio.shape == (2, SR)
    assert np.isfinite(audio).all()
    assert float(np.abs(audio).max()) > 0.0


def test_n_harmonics_range_draws_the_comb_length_per_clip():
    """``n_harmonics_range`` makes the comb partially observed; absent, nothing moves."""
    lo, hi = 10, 30
    drawn = [
        srn.sample_params(
            np.random.default_rng(s),
            n_rotors=4,
            n_harmonics=80,
            n_harmonics_range=(lo, hi),
            sample_rate=SR,
        ).n_harmonics
        for s in range(24)
    ]
    assert all(lo <= k <= hi for k in drawn)
    assert len(set(drawn)) > 1  # a fresh length per clip, not one fixed value
    # The profile is sized by the drawn length, so the comb really is shorter.
    params = srn.sample_params(
        np.random.default_rng(0),
        n_rotors=4,
        n_harmonics=80,
        n_harmonics_range=(lo, hi),
        sample_rate=SR,
    )
    assert params.profile_db.shape == (4, params.n_harmonics)
    # Without the key the value is exactly `n_harmonics`.
    assert (
        srn.sample_params(
            np.random.default_rng(0), n_rotors=4, n_harmonics=80, sample_rate=SR
        ).n_harmonics
        == 80
    )


def test_fitted_population_sets_profile_and_scale_invariant_line_floor_ratio():
    mean = (2.0, 6.0, -1.0, -4.0, -7.0)
    rotor = ((1.0, -1.0, 0.5, 0.0, -0.5), (-1.0, 1.0, -0.5, 0.0, 0.5))
    ranges = srn.StochasticRanges(
        profile_mean_db=mean,
        profile_basis_db=(),
        profile_rotor_db=rotor,
        line_floor_mean_db=18.0,
        line_floor_std_db=0.0,
        rotor_contrast_std_db=0.0,
        # These incompatible legacy controls must be ignored by the fitted path.
        rolloff_p=(9.0, 9.0),
        harm_jitter_db=(30.0, 30.0),
        floor_shape_std_db=(0.0, 0.0),
        floor_tilt_db_oct=(0.0, 0.0),
    )
    params = srn.sample_params(
        np.random.default_rng(3),
        ranges,
        n_rotors=2,
        n_harmonics=5,
        sample_rate=SR,
    )

    expected = np.asarray(mean)[None, :] + np.asarray(rotor)
    np.testing.assert_allclose(params.profile_db, expected)
    assert params.profile_db[:, 1].mean() - params.floor_mean_db == pytest.approx(18.0)


def test_fitted_population_draws_the_learned_cross_order_covariance():
    mode = np.array((-4.0, -2.0, 0.0, 2.0, 4.0))
    ranges = srn.StochasticRanges(
        profile_mean_db=(0.0,) * 5,
        profile_basis_db=(tuple(mode),),
        profile_rotor_db=((0.0,) * 5,),
        line_floor_mean_db=18.0,
        line_floor_std_db=0.0,
        rotor_contrast_std_db=0.0,
        floor_shape_std_db=(0.0, 0.0),
        floor_tilt_db_oct=(0.0, 0.0),
    )
    profiles = np.stack(
        [
            srn.sample_params(
                np.random.default_rng(seed),
                ranges,
                n_rotors=1,
                n_harmonics=5,
                sample_rate=SR,
            ).profile_db[0]
            for seed in range(512)
        ]
    )
    empirical = np.cov(profiles, rowvar=False)
    expected = np.outer(mode, mode)
    cosine = np.sum(empirical * expected) / (np.linalg.norm(empirical) * np.linalg.norm(expected))
    assert cosine > 0.999


def test_visibility_hurdle_uses_a_finite_attenuated_component() -> None:
    ranges = srn.StochasticRanges(
        profile_mean_db=(0.0,) * 5,
        profile_basis_db=(),
        profile_rotor_db=((0.0,) * 5,),
        line_floor_mean_db=18.0,
        line_floor_std_db=0.0,
        rotor_contrast_std_db=0.0,
        visibility_probability=(1.0, 0.0, 1.0, 0.0, 1.0),
        visibility_attenuation_db=(8.0, 10.0, 12.0, 14.0, 16.0),
        visibility_attenuation_std_db=0.0,
        floor_shape_std_db=(0.0, 0.0),
        floor_tilt_db_oct=(0.0, 0.0),
    )
    params = srn.sample_params(
        np.random.default_rng(19),
        ranges,
        n_rotors=1,
        n_harmonics=5,
        sample_rate=SR,
    )

    np.testing.assert_allclose(params.profile_db[0], [0.0, -10.0, 0.0, -14.0, 0.0])
    assert params.floor_mean_db == pytest.approx(-18.0)
    assert np.min(params.profile_db) > -120.0


def test_visibility_uses_a_substream_and_changes_only_the_profile() -> None:
    plain_ranges = srn.StochasticRanges(
        profile_mean_db=(0.0,) * 5,
        profile_basis_db=(),
        profile_rotor_db=((0.0,) * 5,),
        line_floor_mean_db=18.0,
        line_floor_std_db=0.0,
        rotor_contrast_std_db=0.0,
        floor_shape_std_db=(0.0, 0.0),
        floor_tilt_db_oct=(0.0, 0.0),
    )
    visible_ranges = replace(
        plain_ranges,
        visibility_probability=(1.0,) * 5,
        visibility_attenuation_db=(10.0,) * 5,
    )
    plain_rng = np.random.default_rng(31)
    visible_rng = np.random.default_rng(31)
    plain = srn.sample_params(
        plain_rng,
        plain_ranges,
        n_rotors=1,
        n_harmonics=5,
        sample_rate=SR,
    )
    visible = srn.sample_params(
        visible_rng,
        visible_ranges,
        n_rotors=1,
        n_harmonics=5,
        sample_rate=SR,
    )

    np.testing.assert_allclose(visible.profile_db, plain.profile_db)
    np.testing.assert_allclose(visible.floor_ctrl_db, plain.floor_ctrl_db)
    np.testing.assert_allclose(visible.gamma0, plain.gamma0)
    assert visible.floor_mean_db == plain.floor_mean_db
    assert visible_rng.random() == plain_rng.random()


def test_n_harmonics_range_reaches_the_pool_and_shortens_the_comb():
    pool = srn.StochasticNoisePool(
        sample_rate=SR, duration_s=1.0, n_harmonics_range=(8, 16), n_mics=1, n_rotors=4
    )
    _, _, params, _ = pool.render(np.random.default_rng(7), 1.0)
    assert 8 <= params.n_harmonics <= 16
    plain = srn.StochasticNoisePool(sample_rate=SR, duration_s=1.0, n_mics=1, n_rotors=4)
    _, _, plain_params, _ = plain.render(np.random.default_rng(7), 1.0)
    assert plain_params.n_harmonics > 16  # the default comb still fills the band


def test_fm_mode_per_mic_floor_gain_leaves_the_lines_alone():
    # A static per-mic floor gain must move the floor and only the floor: the
    # line-to-floor ratio realized at each microphone shifts by exactly the
    # mic's floor gain, and the lines themselves stay at one level across mics.
    rng = np.random.default_rng(3)
    ranges = srn.StochasticRanges(
        rolloff_p=(0.5, 0.5),
        harm_jitter_db=(0.0, 0.0),
        blade_counts=(1,),
        rotor_similarity=(1.0, 1.0),
        gamma0_hz=(0.5, 0.5),
        gamma_slope_hz=(0.2, 0.2),
        floor_shape_std_db=(0.0, 0.0),
        floor_tilt_db_oct=(-3.0, -3.0),
        floor_rel_db=(-20.0, -20.0),
        harm_gp_std_db=(0.0, 0.0),
        floor_gp_std_db=(0.0, 0.0),
        floor_tilt_gp_std=(0.0, 0.0),
        mic_floor_std_db=(6.0, 6.0),
    )
    params = srn.sample_params(rng, ranges, n_rotors=1, n_harmonics=20, sample_rate=SR)
    rps = np.full((1, SR * 4), 100.0)
    audio, diag = srn.synthesize(
        params, rps, rng=rng, n_mics=6, mic_gain_db=(0.0, 0.0), line_mode="fm", normalize_rms=None
    )
    power, freqs = _power_spectrum_frames(audio[0])
    df = float(freqs[1] - freqs[0])
    line_bins = np.zeros(freqs.size, bool)
    for k in range(1, 21):
        i = int(round(k * 100.0 / df))
        line_bins[i - 2 : i + 3] = True
    floor_bins = ~line_bins & (freqs > 300) & (freqs < 3000)
    lines, floors = [], []
    for m in range(6):
        pm = _power_spectrum_frames(audio[m])[0].mean(axis=0)
        floor_bin = float(np.median(pm[floor_bins]))
        lines.append(10 * np.log10(max(pm[line_bins].sum() - floor_bin * line_bins.sum(), 1e-20)))
        floors.append(10 * np.log10(floor_bin))
    lines, floors = np.array(lines), np.array(floors)
    assert np.ptp(floors) > 6.0  # the floor gains were drawn (std 6 dB) and applied
    assert np.ptp(lines) < 2.5  # the lines did not follow them (the leak gives >= 6 dB)


def test_fitted_width_and_microphone_parameters_reach_the_waveform() -> None:
    gain_db = ((0.0, -6.0), (3.0, 1.0))
    floor_db = (-2.0, 4.0)
    all_db = (1.5, -1.5)
    ranges = srn.StochasticRanges(
        fixed_gamma0_hz=(1.0, 2.0),
        fixed_gamma_slope_hz=(0.2, 0.4),
        fixed_shaft_jitter_rps=(0.3, 0.7),
        fixed_mic_gain_db=gain_db,
        fixed_mic_floor_db=floor_db,
        fixed_mic_gain_all_db=all_db,
        floor_shape_std_db=(0.0, 0.0),
        floor_tilt_db_oct=(0.0, 0.0),
        harm_gp_std_db=(0.0, 0.0),
        floor_gp_std_db=(0.0, 0.0),
        floor_tilt_gp_std=(0.0, 0.0),
    )
    params = srn.sample_params(
        np.random.default_rng(4),
        ranges,
        n_rotors=2,
        n_harmonics=12,
        sample_rate=SR,
    )
    rps = np.stack((np.full(SR // 2, 70.0), np.full(SR // 2, 83.0)))
    _, diagnostics = srn.synthesize(
        params,
        rps,
        rng=np.random.default_rng(5),
        n_mics=2,
        mic_gain_db=(-30.0, -20.0),
        line_mode="fm",
        normalize_rms=None,
    )

    np.testing.assert_allclose(params.gamma0, [1.0, 2.0])
    np.testing.assert_allclose(params.gamma_slope, [0.2, 0.4])
    np.testing.assert_allclose(params.shaft_jitter_rps, [0.3, 0.7])
    np.testing.assert_allclose(diagnostics["mic_gains"], 10.0 ** (np.asarray(gain_db) / 10.0))
    np.testing.assert_allclose(
        diagnostics["mic_floor_gains"],
        10.0 ** (np.asarray(floor_db) / 10.0),
    )
    np.testing.assert_allclose(diagnostics["mic_gain_all_db"], all_db)


def test_fm_harmonics_have_static_microphone_phase() -> None:
    ranges = srn.StochasticRanges(
        gamma0_hz=(0.5, 0.5),
        gamma_slope_hz=(0.1, 0.1),
        fixed_shaft_jitter_rps=(0.0,),
        phase_diffusion_hz_per_order=(0.0, 0.0),
        floor_shape_std_db=(0.0, 0.0),
        floor_tilt_db_oct=(0.0, 0.0),
        floor_rel_db=(-80.0, -80.0),
        min_lines_above_floor=0.0,
        harm_gp_std_db=(0.0, 0.0),
        floor_gp_std_db=(0.0, 0.0),
        floor_tilt_gp_std=(0.0, 0.0),
    )
    params = srn.sample_params(
        np.random.default_rng(18),
        ranges,
        n_rotors=1,
        n_harmonics=12,
        sample_rate=SR,
    )
    audio, _ = srn.synthesize(
        params,
        np.full((1, 2 * SR), 73.0),
        rng=np.random.default_rng(19),
        n_mics=2,
        mic_gain_db=(0.0, 0.0),
        line_mode="fm",
        normalize_rms=None,
    )

    # Propagation contributes a static phase at each microphone. Equal gains
    # therefore preserve the line powers without cloning the line waveform.
    assert abs(float(np.corrcoef(audio)[0, 1])) < 0.9
    p0, freqs = _power_spectrum(audio[0])
    p1, _ = _power_spectrum(audio[1])
    line_bins = np.asarray([np.argmin(np.abs(freqs - 73.0 * k)) for k in range(1, 13)])
    np.testing.assert_allclose(
        10.0 * np.log10(np.maximum(p0[line_bins], 1e-20)),
        10.0 * np.log10(np.maximum(p1[line_bins], 1e-20)),
        atol=1.0,
    )


def test_shaft_offset_moves_the_comb_off_the_label() -> None:
    """The comb rides the shaft; the label the caller passed stays the label.

    Real telemetry differs from the shaft by a static per-rotor offset, so a
    line at order ``k`` sits ``k`` times that offset away from the frequency the
    label predicts. A generator that ignores this teaches a regressor an
    accuracy no real recording supports.
    """
    label_rps = 70.0
    offset_std = 2.0
    ranges = srn.StochasticRanges(
        gamma0_hz=(0.5, 0.5),
        gamma_slope_hz=(0.05, 0.05),
        fixed_shaft_jitter_rps=(0.0,),
        phase_diffusion_hz_per_order=(0.0, 0.0),
        floor_shape_std_db=(0.0, 0.0),
        floor_tilt_db_oct=(0.0, 0.0),
        floor_rel_db=(-80.0, -80.0),
        min_lines_above_floor=0.0,
        harm_gp_std_db=(0.0, 0.0),
        floor_gp_std_db=(0.0, 0.0),
        floor_tilt_gp_std=(0.0, 0.0),
        shaft_offset_rps=(offset_std, offset_std),
    )
    params = srn.sample_params(
        np.random.default_rng(31),
        ranges,
        n_rotors=1,
        n_harmonics=20,
        sample_rate=SR,
    )
    offset = float(np.asarray(params.shaft_offset_rps).ravel()[0])
    assert offset != 0.0
    audio, _ = srn.synthesize(
        params,
        np.full((1, 2 * SR), label_rps),
        rng=np.random.default_rng(32),
        n_mics=1,
        mic_gain_db=(0.0, 0.0),
        line_mode="fm",
        normalize_rms=None,
    )
    power, freqs = _power_spectrum(audio[0].astype(np.float64))
    for order in (2, 5, 10):
        centre = order * label_rps
        window = np.flatnonzero(np.abs(freqs - centre) < 0.4 * label_rps)
        peak = float(freqs[window][np.argmax(power[window])])
        assert abs(peak - centre - order * offset) < 1.5 * (freqs[1] - freqs[0])

    # A stopped rotor stays stopped — the offset is an error about a running
    # shaft's speed, not a phantom rotation. The recording chain's static floor
    # keeps the clip audible, so a phantom comb would show as a tonal peak.
    stopped, _ = srn.synthesize(
        params.with_(floor_static_rel=0.2),
        np.zeros((1, SR)),
        rng=np.random.default_rng(33),
        n_mics=1,
        mic_gain_db=(0.0, 0.0),
        line_mode="fm",
        normalize_rms=None,
    )
    stopped_power, stopped_freqs = _power_spectrum(stopped[0].astype(np.float64))
    band = (stopped_freqs > 20.0) & (stopped_freqs < 4000.0)
    assert float(stopped_power[band].max() / np.median(stopped_power[band])) < 50.0


def test_shaft_jitter_log_std_varies_the_whole_clip_together() -> None:
    """One width draw per clip, shared by its rotors.

    A real flight's shaft wanders more in one segment than another, and the
    per-clip fits measure that as a common mode across the four rotors. The
    spread must therefore scale every rotor's jitter by the SAME factor — a
    per-rotor draw would be the estimation noise the measurement subtracts.
    """
    fixed = (0.2, 0.4, 0.6, 0.8)
    ranges = srn.StochasticRanges(
        fixed_shaft_jitter_rps=fixed,
        shaft_jitter_log_std=(0.5, 0.5),
    )
    factors = []
    for seed in range(6):
        params = srn.sample_params(
            np.random.default_rng(seed),
            ranges,
            n_rotors=4,
            n_harmonics=8,
            sample_rate=SR,
        )
        drawn = np.asarray(params.shaft_jitter_rps, dtype=np.float64)
        ratio = drawn / np.asarray(fixed)
        np.testing.assert_allclose(ratio, ratio[0], rtol=1e-9)
        factors.append(float(ratio[0]))
    # a real spread, and centred on the fitted width (lognormal median 1)
    assert np.std(np.log(factors), ddof=1) > 0.1
    assert min(factors) < 1.0 < max(factors)


def test_render_reuse_amortises_synthesis_without_freezing_the_noise() -> None:
    """One render per ``render_reuse`` draws, and the pool stays diverse.

    Rendering a clip for every training sample starves the GPU; a real-noise
    baseline reuses a finite corpus, so reusing rendered clips is the same
    regime. What must hold is that the render RATE drops and the drawn clips do
    not collapse onto one waveform.
    """

    def make(reuse: int) -> srn.StochasticNoisePool:
        return srn.StochasticNoisePool(
            sample_rate=SR,
            duration_s=1.0,
            n_harmonics=24,
            n_mics=2,
            line_mode="fm",
            ranges={"floor_rel_db": (-20.0, -20.0), "min_lines_above_floor": 0.0},
            render_reuse=reuse,
            render_pool=4,
        )

    def counting(pool: srn.StochasticNoisePool) -> list[int]:
        calls = [0]
        inner = pool.render

        def counted(*args: object, **kwargs: object) -> object:
            calls[0] += 1
            return inner(*args, **kwargs)  # type: ignore[arg-type]

        pool.render = counted  # type: ignore[assignment]
        return calls

    reuse = 8
    pooled = make(reuse)
    calls = counting(pooled)
    rng = np.random.default_rng(4)
    drawn = [np.asarray(pooled.sample_timeframe(rng, 1.0)["audio"].data) for _ in range(32)]

    assert calls[0] == 32 // reuse, f"expected {32 // reuse} renders, got {calls[0]}"
    distinct = {round(float(np.std(x, dtype=np.float64)), 9) for x in drawn}
    assert len(distinct) >= 3, "the pool collapsed onto too few clips"

    eager = make(1)
    eager_calls = counting(eager)
    for _ in range(5):
        eager.sample_timeframe(rng, 1.0)
    assert eager_calls[0] == 5, "the default must still render every sample"


def test_per_rotor_floor_guard_lifts_the_quietest_rotor_above_the_floor():
    """A pooled coverage fraction can bury a whole rotor; the per-rotor form
    cannot.

    One rotor is put 30 dB under the other three. With only the pooled guard
    the loud rotors satisfy it on their own and the quiet one keeps no visible
    line at all — a clip whose fourth label has no evidence anywhere in the
    spectrum. The per-rotor guard lowers the floor until every rotor keeps its
    share.
    """
    params = _params(3)
    profile = params.profile_db.copy()
    profile[3] -= 30.0
    quiet = replace(params, profile_db=profile)

    def coverage(level: float) -> np.ndarray:
        zero = replace(quiet, floor_mean_db=0.0)
        return np.array(
            [
                float(np.mean(peaks > level + srn.floor_shape_db(zero, centers)))
                for peaks, centers in srn.line_peaks_by_rotor(quiet)
            ]
        )

    pooled = srn.calibrate_floor(quiet, -2.0, min_lines_above_floor=0.30)
    guarded = srn.calibrate_floor(
        quiet, -2.0, min_lines_above_floor=0.30, min_lines_above_floor_per_rotor=0.30
    )
    assert coverage(pooled).mean() >= 0.30  # the pooled guard is satisfied ...
    assert coverage(pooled).min() < 0.05  # ... with one rotor essentially invisible
    assert guarded < pooled  # the floor had to come down
    assert coverage(guarded).min() >= 0.30  # every rotor keeps a trackable comb
    # A zero threshold is exactly the old behaviour.
    assert srn.calibrate_floor(
        quiet, -2.0, min_lines_above_floor=0.30, min_lines_above_floor_per_rotor=0.0
    ) == pytest.approx(pooled)


def test_per_rotor_floor_threshold_reaches_the_renderer_from_its_range():
    """``StochasticRanges`` carries the threshold into ``sample_params``."""
    profile = None
    for seed in range(6):
        loose = srn.sample_params(
            np.random.default_rng(seed), n_rotors=4, n_harmonics=64, sample_rate=SR
        )
        tight = srn.sample_params(
            np.random.default_rng(seed),
            n_rotors=4,
            n_harmonics=64,
            sample_rate=SR,
            ranges=srn.StochasticRanges(min_lines_above_floor_per_rotor=0.95),
        )
        assert tight.floor_mean_db <= loose.floor_mean_db
        if tight.floor_mean_db < loose.floor_mean_db:
            profile = tight
    assert profile is not None, "no draw needed the per-rotor guard"
    zero = replace(profile, floor_mean_db=0.0)
    per_rotor = [
        float(np.mean(peaks > profile.floor_mean_db + srn.floor_shape_db(zero, centers)))
        for peaks, centers in srn.line_peaks_by_rotor(profile)
    ]
    assert min(per_rotor) >= 0.95
