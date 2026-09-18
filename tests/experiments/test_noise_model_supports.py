"""The v2 supports' two load-bearing claims: absolute level and the bench rule.

Everything here runs on planted audio — no dataset, no network — because the
two things that can silently poison every round-1 number are a periodogram in
the wrong units and a stationarity rule that accepts a drifting carrier.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.noise_model import supports as S


def _tone(sr: float, seconds: float, freq: float, amp: float = 0.25) -> np.ndarray:
    t = np.arange(int(round(seconds * sr))) / float(sr)
    return amp * np.cos(2.0 * np.pi * freq * t)


def _peak(support: S.Support, freq: float) -> float:
    j = int(np.argmin(np.abs(support.freqs_hz - freq)))
    return float(support.power[0, 0, j])


class TestAbsoluteLevel:
    """``power`` is ``|rfft(x*w)|^2/sum(w^2)``, one-sided, at 16 kHz."""

    def test_planted_sinusoid_reads_the_frozen_convention(self):
        # A bin-centred tone reads A^2 * n_fft / 6 in this convention: the Hann
        # coherent gain (sum w)^2/sum(w^2) = 2N/3 times A^2/4.
        sr, seconds, freq, amp = 16000, 4.0, 1000.0, 0.25
        sup = S.bench_support(
            "planted", _tone(sr, seconds, freq, amp), sr, [50.0], segment=(0.0, seconds)
        )
        n_fft = int(round(seconds * sr))
        assert sup.sr == S.SR
        assert sup.n_fft == n_fft and sup.power.shape == (1, 1, n_fft // 2 + 1)
        assert _peak(sup, freq) == pytest.approx(amp**2 * n_fft / 6.0, rel=1e-3)

    def test_native_rate_and_decimated_agree_in_the_shared_band(self):
        # The same physical segment at 44.1 kHz: the bench route decimates it to
        # 16 kHz, so its level must match the native-rate periodogram scaled by
        # 16000/44100 (both the tone and the broadband floor carry that factor,
        # because a fixed DURATION makes n_fft proportional to the rate).
        seconds, freq, amp = 4.0, 1000.0, 0.25
        native_sr = 44100
        rng = np.random.default_rng(0)
        noise = 0.01 * rng.standard_normal(int(round(seconds * native_sr)))
        audio = _tone(native_sr, seconds, freq, amp) + noise

        sup = S.bench_support("planted", audio, native_sr, [50.0], segment=(0.0, seconds))

        n_native = audio.size
        window = np.hanning(n_native + 1)[:n_native]
        spec = np.fft.rfft(audio * window)
        native_power = (spec.real**2 + spec.imag**2) / float(np.sum(window**2))
        native_power *= S.SR / native_sr
        native_freqs = np.fft.rfftfreq(n_native, 1.0 / native_sr)

        peak_native = float(native_power[int(np.argmin(np.abs(native_freqs - freq)))])
        assert _peak(sup, freq) == pytest.approx(peak_native, rel=5e-3)

        # ... and the broadband floor, averaged over a decade inside both
        # passbands, carries the same factor.
        band = (native_freqs >= 500.0) & (native_freqs <= 5000.0)
        band &= np.abs(native_freqs - freq) > 5.0
        mine = (sup.freqs_hz >= 500.0) & (sup.freqs_hz <= 5000.0)
        mine &= np.abs(sup.freqs_hz - freq) > 5.0
        assert float(np.mean(sup.power[0, 0, mine])) == pytest.approx(
            float(np.mean(native_power[band])), rel=0.05
        )

    def test_bench_support_is_one_frame_and_flight_geometry_is_reconstructible(self):
        sr, seconds = 16000, 4.0
        sup = S.bench_support(
            "planted", _tone(sr, 6.0, 1000.0), sr, [50.0, 60.0], segment=(1.0, 1.0 + seconds)
        )
        assert sup.n_frames == 1 and sup.kind == "bench"
        assert sup.n_rotors == 2
        assert sup.carrier_rev_s.shape == (2, 1)
        assert sup.carrier_rev_s[:, 0] == pytest.approx([50.0, 60.0])
        assert sup.carrier_rev_s_audio.shape == (2, int(round(seconds * sr)))
        assert sup.frame_centres_s == pytest.approx((sup.frame_starts + sup.n_fft / 2) / sup.sr)
        assert sup.segment == (1.0, 5.0)

    def test_a_segment_outside_the_recording_is_refused(self):
        with pytest.raises(ValueError, match="not inside"):
            S.bench_support("planted", _tone(16000, 2.0, 900.0), 16000, [50.0], segment=(3.0, 5.0))


def _bench_audio(
    sr: float, speed_rev_s: np.ndarray, order: int, *, seed: int = 0, noise: float = 0.02
) -> np.ndarray:
    """One harmonic of a given speed TRACK, plus a little broadband."""
    phase = 2.0 * np.pi * order * (np.cumsum(speed_rev_s) - 0.5 * speed_rev_s[0]) / float(sr)
    rng = np.random.default_rng(seed)
    return np.cos(phase) + noise * rng.standard_normal(speed_rev_s.size)


class TestLineMargin:
    def test_the_margin_of_a_jittered_line_is_length_invariant(self):
        """The margin is a BAND-power ratio, so it must not fall as the window
        grows. A peak-bin margin does: block-averaging a longer record averages
        the line's wandering position, which cost six round-2 DREGON windows
        (12-32 s) the 3 dB rule that a 4 s window passed."""
        sr, order, base = 16000.0, 70, 68.0
        rng = np.random.default_rng(11)
        n = int(30 * sr)
        # a STATIONARY jitter of the measured size (~0.25 Hz at k = 70, so
        # 0.0036 rev/s): smoothed white noise, not a random walk, so the line's
        # width is the same in a 4 s and a 30 s window and only the statistic
        # is under test
        w = int(0.1 * sr)
        raw = rng.standard_normal(n + w)
        smooth = np.convolve(raw, np.ones(w) / w, mode="valid")[:n]
        jitter = 0.25 / order * smooth / np.std(smooth)
        audio = _bench_audio(sr, base + jitter, order, noise=0.3)
        short = S.line_margins(audio[None, : int(4 * sr)], sr, base)[order]
        long = S.line_margins(audio[None, :], sr, base)[order]
        assert abs(short - long) < 1.0
        assert min(short, long) > S.BENCH_LINE_MARGIN_DB

    def test_a_pure_noise_order_scores_near_zero_at_any_length(self):
        """No length-dependent bias on noise either. The margin locates the
        line by its largest bin in a +-4 Hz search before integrating, so pure
        noise scores a few dB rather than exactly 0 and the brightest of 21
        orders can reach ~5 dB; the TYPICAL order must stay under the rule."""
        sr, base = 16000.0, 68.0
        noise = np.random.default_rng(12).standard_normal(int(30 * sr))[None, :]
        medians = []
        for seconds in (4, 30):
            m = S.line_margins(noise[:, : int(seconds * sr)], sr, base)
            medians.append(float(np.median(list(m.values()))))
            assert medians[-1] < S.BENCH_LINE_MARGIN_DB
        assert abs(medians[0] - medians[1]) < 1.5


class TestBenchStationarityRule:
    """The +-1 Hz rule finds the steady span and refuses a drifting one."""

    def test_finds_the_steady_span_of_a_tone_that_starts_drifting(self):
        sr, order, base = 16000.0, 70, 68.0
        t = np.arange(int(14 * sr)) / sr
        speed = np.where(t < 8.0, base, base + 0.5 * (t - 8.0))
        rule = S.stationary_segment(_bench_audio(sr, speed, order), sr, [base])

        assert rule["passed"] is True
        assert rule["orders"] == [order]
        # the steady span is [0, 8) and BENCH_EDGE_S trims the first 2 s
        assert rule["start_s"] == pytest.approx(S.BENCH_EDGE_S, abs=0.5)
        # the 2 s residual average sees the ramp about a second late now that
        # the wide-band test (rule rev 1) no longer clips the span early
        assert 6.3 <= rule["end_s"] <= 8.7
        assert rule["duration_s"] >= S.BENCH_MIN_SEGMENT_S
        # the carrier is recovered from the survey speed it was handed
        assert rule["carrier_rev_s"][0] == pytest.approx(base, abs=0.01)
        assert rule["residual_std_hz"][0] < 0.3
        assert rule["residual_max_abs_hz"] <= S.BENCH_RESIDUAL_TOL_HZ

    def test_a_carrier_that_drifts_throughout_fails(self):
        sr, order, base = 16000.0, 70, 68.0
        t = np.arange(int(14 * sr)) / sr
        rule = S.stationary_segment(_bench_audio(sr, base + 0.5 * t, order), sr, [base])
        # rule rev 2 catches it on the half-window DRIFT, not on the span: a
        # carrier that leaves the 1.5 Hz demodulation band makes the narrow
        # residual read filtered noise, i.e. perfect steadiness
        assert rule["drift_ok"] is False
        assert rule["carrier_drift_hz"][0] > S.BENCH_RESIDUAL_TOL_HZ
        # and the recording is still given a defensible window
        assert rule["duration_s"] >= S.BENCH_MIN_SEGMENT_S

    def test_a_quantised_survey_speed_is_refined_back_onto_the_line(self):
        # The manifest quantises speeds to ~0.01 rev/s, which at order 70 is
        # 0.7 Hz — most of the tolerance. The rule must refine it away.
        sr, order, true_speed = 16000.0, 70, 68.037
        speed = np.full(int(12 * sr), true_speed)
        rule = S.stationary_segment(_bench_audio(sr, speed, order), sr, [68.0])
        assert rule["carrier_rev_s"][0] == pytest.approx(true_speed, abs=0.005)
        assert abs(rule["carrier_shift_rev_s"][0]) > 0.02
        assert rule["passed"] is True
        assert abs(rule["residual_mean_hz"][0]) < 0.3

    def test_every_rotor_must_be_steady(self):
        # Two rotors, the second drifting: the accepted span is the intersection,
        # so a span that is steady for rotor 0 alone is not enough.
        sr, order = 16000.0, 70
        t = np.arange(int(14 * sr)) / sr
        steady = np.full(t.size, 68.0)
        drifting = 55.0 + 0.2 * np.sin(2.0 * np.pi * t / 14.0)
        # Use a distinct actual order for the second rotor.  Its slow excursion
        # cannot average away in the 2 s residual smoother.
        audio = _bench_audio(sr, steady, order) + _bench_audio(sr, drifting, 63, seed=1)
        rule = S.stationary_segment(audio, sr, [68.0, 55.0])
        # rev 2 gates on the margin and the span; the second rotor's excursion
        # is reported as drift (it is slow speed wander, which the model's
        # shaft dynamics absorb, not a reason to refuse the window)
        # the second rotor fails one of the two per-rotor tests: its residual
        # excursion inside the window, or its half-window carrier drift
        assert rule["longest_inside_s"] < S.BENCH_MIN_SEGMENT_S or rule["drift_ok"] is False

    def test_a_window_with_no_motor_is_refused_even_though_silence_is_steady(self):
        """Rule rev 2's level gate. Silence has a perfectly stationary
        filtered-noise residual, so rev 1 preferred it: 12 of the 21 DREGON
        bench windows landed on the post-spin-down tail, 10-33 dB down."""
        sr, order, base = 16000.0, 70, 68.0
        n = int(9 * sr)
        speed = np.full(n, base)
        motor = _bench_audio(sr, speed, order, noise=0.02)
        rng = np.random.default_rng(3)
        quiet = 0.02 * rng.standard_normal(int(11 * sr))
        rule = S.stationary_segment(np.concatenate([motor, quiet])[None, :], sr, [base])

        # the accepted window is the motor's, not the longer quiet tail
        assert rule["start_s"] >= S.BENCH_EDGE_S - 1e-6
        assert rule["end_s"] <= 9.5
        assert rule["level_deficit_db"] < S.BENCH_LEVEL_TOL_DB
        # and the window is certified by the margin measured INSIDE it
        assert rule["line_margin_db"][0] >= S.BENCH_LINE_MARGIN_DB
        assert rule["passed"] is True

    def test_the_carrier_and_margin_are_measured_on_the_selected_window(self):
        """Rule rev 2 refines the carrier on the window, not the recording: a
        recording that is two thirds silence diluted rev 1's residual mean to
        ~zero and left the carrier at the survey value."""
        sr, order, true_speed = 16000.0, 70, 68.03
        motor = _bench_audio(sr, np.full(int(7 * sr), true_speed), order)
        quiet = 0.02 * np.random.default_rng(5).standard_normal(int(14 * sr))
        rule = S.stationary_segment(np.concatenate([motor, quiet])[None, :], sr, [68.0])

        assert rule["carrier_rev_s"][0] == pytest.approx(true_speed, abs=0.01)
        # the whole-recording pass is recorded beside it and is the worse
        # estimate: two thirds of the record has no line to demodulate
        assert abs(rule["carrier_recording_rev_s"][0] - true_speed) >= abs(
            rule["carrier_rev_s"][0] - true_speed
        )
        assert rule["line_margin_db"][0] > rule["line_margin_recording_db"][0]


class TestSpecsAndCache:
    def test_spec_text_round_trips(self):
        specs = [
            S.bench_dregon_motor("Motor3", 90),
            S.bench_dregon_motor("allMotors", 70),
            S.bench_point("DREGON-bench__motor_Motor1_70"),
            S.flight_michaels("FLY125", 16.0, 8.0),
            S.flight_michaels("FLY124", 27.68, 8.0),
            S.flight_dregon("hovering_nosource_room2", 1511903905.3944898, 4.0),
        ]
        for spec in specs:
            assert S.parse_spec(spec.text) == spec
        # the frozen cohort's label keys are not interchangeable
        assert S.flight_michaels("FLY124", 8.0, 8.0).args["rps_key"] == "rps"
        assert S.flight_michaels("FLY125", 16.0, 8.0).args["rps_key"] == "rps_refined"
        assert S.flight_dregon("x", 0.0, 4.0).args["rps_key"] == "motors_command"
        with pytest.raises(ValueError):
            S.parse_spec("bench_dregon_motor:Motor1")

    def test_the_round_one_sets_are_the_frozen_ones(self):
        sizes = {name: len(S.support_set(name)) for name in S.SUPPORT_SETS}
        assert sizes == {
            "dregon-bench": 21,
            "bench-points": 135,
            "michaels-cruise": 13,
            "dregon-floor": 10,
        }
        # the floor segments never touch the five frozen scoring windows
        floor = S.support_set("dregon-floor")
        scored = {
            (s.args["recording"], s.args["start_s"], s.args["dur_s"])
            for s in floor
            if s.args["dur_s"] == S.DREGON_SCORED_DUR_S
        }
        assert len(scored) == 5
        for rec, start in S.DREGON_SCORED:
            assert (rec, start, S.DREGON_SCORED_DUR_S) in scored
        for spec in floor:
            if spec.args["dur_s"] != S.DREGON_FLOOR_DUR_S:
                continue
            same = [s for s, t in S.DREGON_SCORED if s == spec.args["recording"]]
            assert same, spec.name
            scored_start = next(t for s, t in S.DREGON_SCORED if s == spec.args["recording"])
            assert spec.args["start_s"] >= scored_start + S.DREGON_SCORED_DUR_S
        # FLY125's eight cruise windows are disjoint
        starts = sorted(
            s.args["start_s"]
            for s in S.support_set("michaels-cruise")
            if s.args["recording"] == "FLY125"
        )
        assert len(starts) == 8
        assert all(b - a >= 8.0 for a, b in zip(starts, starts[1:]))

    def test_npz_cache_round_trips(self, tmp_path):
        sr, seconds = 16000, 2.0
        sup = S.bench_support(
            "cached",
            _tone(sr, 6.0, 1000.0),
            sr,
            [61.5],
            segment=(3.0, 3.0 + seconds),
            meta=dict(recording_id="planted", stationary_pass=True),
        )
        path = S.save_support(sup, out_dir=tmp_path)
        assert path.exists()
        back = S.load_cached("cached", cache_dir=tmp_path)
        assert back is not None
        assert back.kind == sup.kind and back.name == sup.name
        assert back.sr == sup.sr and back.n_fft == sup.n_fft and back.hop == sup.hop
        assert back.segment == sup.segment
        assert back.meta["recording_id"] == "planted"
        assert back.power == pytest.approx(sup.power.astype(np.float32), rel=0, abs=0)
        assert back.freqs_hz == pytest.approx(sup.freqs_hz)
        assert back.carrier_rev_s == pytest.approx(sup.carrier_rev_s)
        assert back.frame_starts.tolist() == sup.frame_starts.tolist()
        assert S.load_cached("absent", cache_dir=tmp_path) is None

    def test_a_multi_frame_bench_support_is_rejected(self):
        sup = S.bench_support("ok", _tone(16000, 1.0, 900.0), 16000, [50.0], segment=(0.0, 1.0))
        with pytest.raises(ValueError, match="ONE frame"):
            S.Support(
                kind="bench",
                name="broken",
                sr=sup.sr,
                n_mics=sup.n_mics,
                freqs_hz=sup.freqs_hz,
                power=np.repeat(sup.power, 2, axis=1),
                carrier_rev_s=np.repeat(sup.carrier_rev_s, 2, axis=1),
                carrier_rev_s_audio=sup.carrier_rev_s_audio,
                frame_starts=np.array([0, 1]),
                frame_centres_s=np.array([0.0, 1.0]),
                n_fft=sup.n_fft,
                hop=sup.hop,
                segment=sup.segment,
            )
