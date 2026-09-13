"""The contracts of the revised-phase baseline instrument.

Every test here defends a property a plausible bug would break, and each one
was written against a decision that is frozen upstream:

* duplicate observation windows must SPLIT exposure, never double the evidence;
* the held-out support guard must REPORT a shortfall, never pad, repeat or
  cycle, and must not hand back a window that overlaps a fit support;
* export pairing is by recording IDENTITY, with no positional or modulo route;
* per-clip nuisance is averaged in LINEAR POWER with each clip's own
  ``power_scale`` folded in first;
* the gate arithmetic, at and around its thresholds, including Michael's
  ratio-of-means (not mean-of-ratios) form and the refusal to produce an
  interval from a single cluster;
* the scoring reference may never be a refined/posterior track;
* both timebase conventions — DREGON's absolute epoch clock and Michael's
  relative one — must work through the same code.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.stochastic_fit import accept_stats as stats
from experiments.stochastic_fit import revised_eval as RE

SR = 16000.0


# ── exposure and the composite score ────────────────────────────────────────


def _frames(
    recording: str,
    start: float,
    values: list[float],
    *,
    hop: int = RE.OBS_HOP,
    n_fft: int = RE.OBS_N_FFT,
    sr: int = RE.SR,
) -> RE.FrameScore:
    """Frames on the real cadence: centres ``hop`` samples apart from ``start``."""
    times = start + np.arange(len(values)) * (hop / sr)
    return RE.FrameScore(
        window=RE.Window(recording, start, len(values) * hop / sr),
        frame_times=times,
        frame_nll=np.asarray(values, dtype=np.float64),
        n_cells_per_frame=10,
        n_fft=n_fft,
        hop=hop,
        sr=sr,
    )


def test_duplicating_one_ANALYSIS_FRAME_does_not_change_the_risk() -> None:
    """The invariant at the granularity that matters: a single duplicated frame
    entry, not merely a duplicated clip. Exposure is keyed on
    (recording, absolute centre, n_fft), so the two copies split one unit."""
    base = _frames("rec", 100.0, [1.0, 2.0, 3.0, 4.0])
    single = RE.composite_score([base])
    # the SECOND frame of the same clip, handed in again on its own
    dup = RE.FrameScore(
        window=base.window,
        frame_times=base.frame_times[1:2],
        frame_nll=base.frame_nll[1:2],
        n_cells_per_frame=base.n_cells_per_frame,
    )
    doubled = RE.composite_score([base, dup])
    assert doubled["score"] == pytest.approx(single["score"], rel=0, abs=1e-12)
    assert doubled["unique_seconds"] == pytest.approx(single["unique_seconds"], abs=1e-12)
    assert doubled["n_frames"] == single["n_frames"] + 1  # the copy is recorded, not silently lost


def test_duplicating_a_whole_clip_also_changes_nothing() -> None:
    one = _frames("rec", 100.0, [1.0, 2.0, 3.0, 4.0])
    two = _frames("rec", 100.0, [1.0, 2.0, 3.0, 4.0])
    assert RE.composite_score([one, two])["score"] == pytest.approx(
        RE.composite_score([one])["score"], abs=1e-12
    )


def test_the_exposure_factor_is_hop_over_window() -> None:
    item = _frames("rec", 0.0, [1.0] * 5)
    weights = RE.exposure_weights([item])[0]
    assert weights == pytest.approx(np.full(5, RE.OBS_HOP / RE.OBS_N_FFT))
    out = RE.composite_score([item])
    assert out["unique_seconds"] == pytest.approx(5 * RE.OBS_HOP / RE.SR)
    assert out["weighted_nats"] == pytest.approx(5 * RE.OBS_HOP / RE.OBS_N_FFT)


def test_hop_density_leaves_the_risk_per_unique_second_about_constant() -> None:
    """Sixteenth-window hop against eighth-window hop over the same 1 s of
    material: twice the frames, each carrying half the exposure, so the risk
    per unique second must NOT scale with the frame count."""
    per_frame = 7.5  # the same in-band cell sum at either density
    dense = _frames("rec", 0.0, [per_frame] * (RE.SR // RE.OBS_HOP), hop=RE.OBS_HOP)
    sparse = _frames("rec", 0.0, [per_frame] * (RE.SR // (2 * RE.OBS_HOP)), hop=2 * RE.OBS_HOP)
    a, b = RE.composite_score([dense])["score"], RE.composite_score([sparse])["score"]
    assert a == pytest.approx(b, rel=1e-9)
    assert a == pytest.approx(per_frame * RE.SR / RE.OBS_N_FFT, rel=1e-9)


def test_genuinely_different_frames_are_never_merged() -> None:
    """One hop apart is a different observation; the same centre in a different
    recording is too."""
    a = _frames("rec", 0.0, [1.0, 1.0])
    shifted = _frames("rec", RE.OBS_HOP / RE.SR, [1.0, 1.0])  # overlaps a by one frame
    out = RE.composite_score([a, shifted])
    assert out["unique_seconds"] == pytest.approx(3 * RE.OBS_HOP / RE.SR)
    assert out["weighted_nats"] == pytest.approx(3 * RE.OBS_HOP / RE.OBS_N_FFT)
    other = _frames("other", 0.0, [1.0, 1.0])
    two_rigs = RE.composite_score([a, other])
    assert set(two_rigs["per_recording"]) == {"rec", "other"}
    assert two_rigs["unique_seconds"] == pytest.approx(4 * RE.OBS_HOP / RE.SR)


def test_marginal_frame_nll_is_the_whittle_cell_sum_on_the_band_only() -> None:
    power = np.full((2, 3, 5), 4.0)
    model = np.full((2, 3, 5), 2.0)
    band = np.array([False, True, True, True, False])
    nll, cells = RE.marginal_frame_nll(power, model, band)
    per_cell = 4.0 / 2.0 + np.log(2.0)
    assert cells == 2 * 3
    assert nll == pytest.approx(np.full(3, per_cell * 6), abs=1e-12)


def test_observation_band_is_the_fixed_30_to_7900_window() -> None:
    freqs = np.array([0.0, 29.0, 30.0, 4000.0, 7900.0, 7901.0, 8000.0])
    assert RE.observation_band(freqs).tolist() == [False, False, True, True, True, False, False]


# ── the support guard ───────────────────────────────────────────────────────


@dataclass
class _Rec:
    """The minimum of ``clips.Recording`` that ``windows`` reads."""

    recording_id: str
    t_start: float
    seconds: float
    rps: float = 80.0

    @property
    def duration_s(self) -> float:
        return self.seconds

    @property
    def coverage(self) -> tuple[float, float]:
        return self.t_start, self.t_start + self.seconds

    def rps_at(self, times: np.ndarray) -> np.ndarray:
        return np.full((4, len(times)), self.rps)


def test_held_out_windows_exclude_the_fit_supports() -> None:
    rec = _Rec("free-flight_nosource_room2", 1512727400.0, 40.0)
    fit = [RE.Window(rec.recording_id, 1512727416.0, 16.0, role="calibration")]
    got, report = RE.resolve_evaluation_windows(
        rec, regime="cruise", calibration=fit, min_seconds=8.0, max_windows=2, min_rps=65.0
    )
    assert report.sufficient
    assert [w.start_s for w in got] == [1512727400.0, 1512727408.0]
    assert all(not w.overlaps(fit[0]) for w in got)
    assert len(report.dropped_overlapping) == 2


def test_a_recording_that_cannot_supply_the_support_is_reported_not_padded() -> None:
    """20 s of audio, 16 s of it fitted: one 8 s window is impossible."""
    rec = _Rec("hovering_nosource_room2", 1511903900.0, 20.0)
    fit = [RE.Window(rec.recording_id, 1511903900.0, 16.0, role="calibration")]
    got, report = RE.resolve_evaluation_windows(
        rec, regime="cruise", calibration=fit, min_seconds=8.0, max_windows=1, min_rps=65.0
    )
    assert got == []
    assert not report.sufficient
    assert "reported, not padded" in report.note
    assert report.as_dict()["disjoint_seconds"] == 0.0


def test_out_of_regime_material_is_not_offered_as_support() -> None:
    rec = _Rec("standby_only", 0.0, 40.0, rps=35.0)
    got, report = RE.resolve_evaluation_windows(
        rec, regime="cruise", calibration=[], min_seconds=8.0, max_windows=1, min_rps=65.0
    )
    assert got == [] and report.n_in_regime == 0 and not report.sufficient


def test_explicit_windows_and_candidate_leakage_share_guard_boundary() -> None:
    fit = [RE.Window("FLY125", 16.0, 16.0, role="calibration")]
    adjacent = RE.Window("FLY125", 32.0, 4.0)
    exact_boundary = RE.Window("FLY125", 33.024, 4.0)
    kept, reports = RE.check_explicit_windows(
        [adjacent, exact_boundary], regime="cruise", calibration=fit, guard_seconds=1.024
    )
    assert [w.start_s for w in kept] == [33.024]
    assert reports[0].dropped_overlapping == ("FLY125@32.000000+4.000000",)
    assert RE.training_leakage(fit, [adjacent], guard_seconds=1.024)["clean"] is False
    assert RE.training_leakage(fit, [exact_boundary], guard_seconds=1.024)["clean"] is True


def test_both_timebase_conventions_go_through_the_same_arithmetic() -> None:
    """DREGON's absolute epoch clock and Michael's relative one."""
    absolute = _Rec("dregon", 1512727417.205, 24.0)
    relative = _Rec("FLY125", 0.0, 24.0)
    for rec in (absolute, relative):
        got, report = RE.resolve_evaluation_windows(
            rec, regime="cruise", calibration=[], min_seconds=8.0, max_windows=3, min_rps=65.0
        )
        assert report.sufficient and len(got) == 3
        assert got[0].start_s == pytest.approx(rec.t_start)
        assert got[-1].end_s == pytest.approx(rec.t_start + 24.0)
    assert RE.union_seconds(
        [RE.Window("dregon", 1512727417.205, 16.0), RE.Window("dregon", 1512727425.205, 16.0)]
    ) == pytest.approx(24.0)


# ── export identity pairing and nuisance aggregation ────────────────────────


def _export(path: Path, recordings: list[str], starts: list[float], *, data: bool) -> Path:
    clips = {}
    for i, rid in enumerate(recordings):
        clips[f"{rid.lower()}_cruise_0{i}"] = dict(
            group="g",
            scores=dict(power_scale=0.01 if i == 0 else 0.04),
            params=dict(
                profile_db=[[0.0, -3.0], [0.0, -3.0]],
                floor_mean_db=-10.0 if i == 0 else -4.0,
                floor_shape_db=[0.0, 0.0],
                floor_ctrl_hz=[30.0, 8000.0],
                gamma0=[1.0, 1.0],
                gamma_slope=[0.5, 0.5],
                mic_floor_db=[0.0, 0.0],
                mic_gain_db=[[0.0, 0.0], [0.0, 0.0]],
                amp_exp=2.5,
                floor_exp=2.5,
                floor_static_rel=0.1,
                coherence_k_half=2.0,
                width_power=1.0,
                floor_tilt_db_oct=-1.0,
                h_db=[[[0.0], [0.0]], [[0.0], [0.0]]],
            ),
        )
    summary: dict = dict(clips=clips, spec=dict(f_max=7900.0), regime="cruise")
    if data:
        summary["data"] = dict(
            recordings=recordings,
            clips=list(clips),
            starts_s=starts,
            seconds=16.0,
            dataset="DREGON-frames",
            rps_key="rps_refined",
        )
    path.write_text(json.dumps(summary))
    return path


def test_pairing_is_by_recording_identity_and_never_by_position(tmp_path: Path) -> None:
    p = _export(
        tmp_path / "e.json",
        ["free-flight_nosource_room2", "hovering_nosource_room2"],
        [1512727417.205, 1511903913.394],
        data=True,
    )
    b = RE.read_export(p, family="refined")
    assert b.clips_of("hovering_nosource_room2") == (
        ("hovering_nosource_room2_cruise_01", 1511903913.394),
    )
    # a recording the export does not name gets NOTHING — no clip i % n
    assert b.clips_of("updown_nosource_room2") == ()
    cov = RE.family_coverage(
        {"refined": b}, ["free-flight_nosource_room2", "updown_nosource_room2"]
    )
    assert not cov["refined"].eligible
    assert cov["refined"].missing == ("updown_nosource_room2",)


def test_a_legacy_export_without_provenance_is_refused(tmp_path: Path) -> None:
    p = _export(tmp_path / "legacy.json", ["FLY125"], [16.0], data=False)
    with pytest.raises(ValueError, match="declared provenance"):
        RE.read_export(p, family="raw")
    b = RE.read_export(
        p,
        family="raw",
        declared=dict(recordings=["FLY125"], starts_s=[16.0], seconds=16.0, rps_key="rps"),
    )
    assert b.declared_provenance and b.calibration_windows()[0].key == "FLY125@16.000000+16.000000"


def test_declared_provenance_must_describe_the_export(tmp_path: Path) -> None:
    p = _export(tmp_path / "legacy.json", ["a", "b"], [0.0, 16.0], data=False)
    with pytest.raises(ValueError, match="provenance does not describe"):
        RE.read_export(
            p, family="raw", declared=dict(recordings=["a"], starts_s=[0.0], seconds=16.0)
        )


def test_ineligible_family_cannot_win_the_global_choice(tmp_path: Path) -> None:
    full = RE.read_export(
        _export(tmp_path / "full.json", ["a", "b"], [0.0, 16.0], data=True), family="refined"
    )
    partial = RE.read_export(_export(tmp_path / "part.json", ["a"], [0.0], data=True), family="raw")
    cov = RE.family_coverage({"refined": full, "raw": partial}, ["a", "b"])
    # the ineligible family has the better (lower) score and must still lose
    chosen = RE.select_family({"refined": -10.0, "raw": -99.0}, cov)
    assert chosen["chosen"] == "refined"
    # ... unless coverage is explicitly not required (Michael's extrapolation)
    assert (
        RE.select_family({"refined": -10.0, "raw": -99.0}, cov, require_coverage=False)["chosen"]
        == "raw"
    )


def test_nuisance_is_averaged_in_linear_power_with_power_scale_folded_in(tmp_path: Path) -> None:
    b = RE.read_export(
        _export(tmp_path / "e.json", ["a", "b"], [0.0, 16.0], data=True), family="refined"
    )
    mp = RE.aggregate_nuisance(b, list(b.clip_ids), label="agg")
    # clip 0: floor -10 dB at power_scale 0.01 (-20 dB) -> -30 dB physical
    # clip 1: floor  -4 dB at power_scale 0.04 (-13.979 dB) -> -17.979 dB physical
    physical = [-10.0 + 10 * np.log10(0.01), -4.0 + 10 * np.log10(0.04)]
    expected = 10.0 * np.log10(np.mean([10 ** (v / 10) for v in physical]))
    assert mp.params["floor_mean_db"] == pytest.approx(expected, abs=1e-9)
    # a dB mean would have given -23.99, which the linear-power mean must beat
    assert mp.params["floor_mean_db"] > float(np.mean(physical)) + 3.0
    assert mp.params["power_scale"] == 1.0
    assert mp.source["power_scale_folded_db"] == pytest.approx([-20.0, -13.979400086720376])
    # clip-local latents are dropped, and that is recorded
    assert "h_db" in mp.source["dropped_clip_local_latents"]
    assert "h_db" not in mp.params


# ── the scoring reference ───────────────────────────────────────────────────


def test_a_refined_track_is_refused_as_the_scoring_reference() -> None:
    for key in ("rps_refined", "auto"):
        with pytest.raises(ValueError, match="not raw telemetry"):
            RE.assert_raw_reference(key)
    assert RE.assert_raw_reference("motors_command") == "motors_command"
    assert RE.assert_raw_reference("rps") == "rps"
    with pytest.raises(ValueError, match="unknown scoring reference"):
        RE.assert_raw_reference("rps_posterior")


# ── clustered intervals ─────────────────────────────────────────────────────


def test_one_cluster_yields_no_interval_at_all() -> None:
    single = RE.cluster_interval([0.4])
    assert single.mean == 0.4
    assert single.lower is None and single.upper is None
    assert "insufficient_clusters" in single.note


def test_the_reported_bounds_are_the_conservative_ones() -> None:
    ci = RE.cluster_interval([0.1, 0.2, 0.3, 0.4, 0.5], alpha=0.05, n_boot=2000, seed=1)
    assert ci.lower == min(ci.t_lower, ci.boot_lower)
    assert ci.upper == max(ci.t_upper, ci.boot_upper)
    assert ci.lower < ci.mean < ci.upper
    assert RE.ADAPTIVE_SELECTION_CAVEAT in ci.as_dict()["caveat"]


# ── gate arithmetic ────────────────────────────────────────────────────────


def _triple(real: float, base: float, cand: float) -> dict[str, float]:
    return dict(real=real, baseline=base, candidate=cand)


COHORT3 = ("a", "b", "c")


def test_dregon_gate_target_is_the_70_percent_gap_point() -> None:
    """E_new <= E_real + 0.7 * (E_best_synth - E_real), and the interval must
    exclude no improvement."""
    passing = RE.dregon_pit_gate(
        {
            "a": _triple(1.0, 2.0, 1.60),
            "b": _triple(1.0, 2.0, 1.62),
            "c": _triple(1.0, 2.0, 1.61),
        },
        gap_fraction=0.70,
        alpha=0.05,
        required_recordings=COHORT3,
    )
    assert passing.detail["clusters"][0]["target"] == pytest.approx(1.70)
    assert passing.passed
    # one hair above the point target fails, with the same clean interval
    failing = RE.dregon_pit_gate(
        {
            "a": _triple(1.0, 2.0, 1.70),
            "b": _triple(1.0, 2.0, 1.71),
            "c": _triple(1.0, 2.0, 1.70),
        },
        gap_fraction=0.70,
        alpha=0.05,
        required_recordings=COHORT3,
    )
    assert not failing.passed
    assert failing.detail["checks"]["point_target"] is False
    assert failing.detail["checks"]["improvement_excludes_zero"] is True


def test_dregon_gate_fails_on_a_partial_cohort_however_good_the_numbers() -> None:
    """The frozen cohort is all five room2 recordings. Two excellent clusters
    used to pass on `>= 2`; they must now fail on completeness."""
    five = (
        "free-flight_nosource_room2",
        "hovering_nosource_room2",
        "updown_nosource_room2",
        "rectangle_nosource_room2",
        "spinning_nosource_room2",
    )
    gate = RE.dregon_pit_gate(
        {five[0]: _triple(1.0, 2.0, 1.1), five[1]: _triple(1.0, 2.0, 1.1)},
        gap_fraction=0.70,
        alpha=0.05,
        required_recordings=five,
    )
    assert gate.detail["checks"]["point_target"] is True
    assert gate.detail["checks"]["improvement_excludes_zero"] is True
    assert gate.detail["checks"]["cohort_complete"] is False
    assert gate.detail["missing_recordings"] == list(five[2:])
    assert not gate.passed


def test_cohort_completeness_gate_fails_on_an_insufficient_support() -> None:
    complete = RE.cohort_completeness_gate(
        required=["a", "b"],
        measured=["a", "b"],
        support_reports=[
            dict(recording="a", sufficient=True),
            dict(recording="b", sufficient=True),
        ],
        name="x",
    )
    assert complete.passed
    short = RE.cohort_completeness_gate(
        required=["a", "b"],
        measured=["a", "b"],
        support_reports=[
            dict(recording="a", sufficient=True),
            dict(recording="b", regime="ramp", sufficient=False, note="0 of 2 windows"),
        ],
        name="x",
    )
    assert short.detail["checks"]["every_support_sufficient"] is False
    assert not short.passed
    assert not RE.cohort_completeness_gate(
        required=["a", "b"], measured=["a"], support_reports=[], name="x"
    ).passed


def test_dregon_gate_fails_when_the_improvement_interval_touches_zero() -> None:
    """The point target is met on the means, but one cluster got WORSE, so the
    one-sided interval no longer excludes 'no improvement'."""
    gate = RE.dregon_pit_gate(
        {"a": _triple(1.0, 2.0, 1.2), "b": _triple(1.0, 2.0, 2.1)},
        gap_fraction=0.70,
        alpha=0.05,
        required_recordings=("a", "b"),
    )
    assert gate.detail["mean_candidate"] <= gate.detail["mean_target"]
    assert gate.detail["checks"]["point_target"] is True
    assert gate.detail["checks"]["improvement_excludes_zero"] is False
    assert not gate.passed


def test_dregon_gate_fails_on_a_single_cluster() -> None:
    gate = RE.dregon_pit_gate(
        {"a": _triple(1.0, 2.0, 1.0)},
        gap_fraction=0.70,
        alpha=0.05,
        required_recordings=("a",),
    )
    assert gate.detail["checks"]["enough_clusters"] is False
    assert not gate.passed


def test_dregon_gate_fails_when_the_calibrated_gap_is_indistinguishable() -> None:
    gate = RE.dregon_pit_gate(
        {"a": _triple(2.0, 1.9, 1.0), "b": _triple(2.0, 2.1, 1.0), "c": _triple(2.0, 2.0, 1.0)},
        gap_fraction=0.70,
        alpha=0.05,
        required_recordings=COHORT3,
    )
    assert gate.detail["checks"]["gap_positive"] is False
    assert not gate.passed


def test_composite_gate_needs_a_negative_mean_and_a_negative_upper_bound() -> None:
    base = {"a": -100.0, "b": -100.0, "c": -100.0}
    clear = RE.composite_delta_gate(base, {"a": -110.0, "b": -111.0, "c": -109.0}, alpha=0.05)
    assert clear.passed and clear.detail["mean_delta"] < 0
    noisy = RE.composite_delta_gate(base, {"a": -140.0, "b": -60.0, "c": -110.0}, alpha=0.05)
    assert noisy.detail["checks"]["mean_delta_negative"] is True
    assert noisy.detail["checks"]["upper_bound_negative"] is False
    assert not noisy.passed


def test_composite_gate_refuses_an_unpaired_comparison() -> None:
    gate = RE.composite_delta_gate(
        {"a": -100.0, "b": -100.0}, {"a": -110.0, "c": -110.0}, alpha=0.05
    )
    assert gate.detail["checks"]["fully_paired"] is False
    assert gate.detail["unpaired_recordings"] == dict(baseline_only=["b"], candidate_only=["c"])
    assert not gate.passed


def test_ltas_gate_is_baseline_calibrated_at_its_tolerance_boundary() -> None:
    base = {"a": 2.0, "b": 3.0}
    assert RE.ltas_gate(base, {"a": 2.4, "b": 3.4}, tolerance_db=0.4).passed
    assert not RE.ltas_gate(base, {"a": 2.5, "b": 3.6}, tolerance_db=0.4).passed
    zero = RE.ltas_gate(base, {"a": 2.0, "b": 3.0}, tolerance_db=0.0)
    assert zero.passed and zero.detail["tolerance_db"] == 0.0


def test_michaels_gate_is_a_ratio_of_equally_weighted_maes() -> None:
    """The frozen arithmetic: mean the MAEs, then divide. A mean of per-regime
    ratios would let a tiny standby MAE dominate — this is that counterexample."""
    blocks = {
        "standby": [dict(key="s1", baseline=0.05, candidate=0.10)],
        "ramp": [dict(key="r1", baseline=2.00, candidate=2.00)],
        "cruise": [dict(key="c1", baseline=1.00, candidate=1.00)],
    }
    gate = RE.michaels_ratio_gate(blocks, ratio_max=1.05, alpha=0.05)
    assert gate.detail["rig_baseline_mae"] == pytest.approx((0.05 + 2.0 + 1.0) / 3)
    assert gate.detail["rig_candidate_mae"] == pytest.approx((0.10 + 2.0 + 1.0) / 3)
    assert gate.detail["aggregate_ratio"] == pytest.approx(3.10 / 3.05)
    assert gate.passed  # 1.016 <= 1.05
    # the rejected mean-of-ratios form would have been (2.0 + 1.0 + 1.0)/3 = 1.33
    assert gate.detail["per_regime"]["standby"]["ratio"] == pytest.approx(2.0)


def test_michaels_gate_boundary_and_missing_regime() -> None:
    at_limit = {
        "standby": [dict(key="s1", baseline=1.0, candidate=1.05)],
        "ramp": [dict(key="r1", baseline=1.0, candidate=1.05)],
        "cruise": [dict(key="c1", baseline=1.0, candidate=1.05)],
    }
    assert RE.michaels_ratio_gate(at_limit, ratio_max=1.05, alpha=0.05).passed
    over = {k: [dict(v[0], candidate=1.06)] for k, v in at_limit.items()}
    assert not RE.michaels_ratio_gate(over, ratio_max=1.05, alpha=0.05).passed
    missing = RE.michaels_ratio_gate(
        {"standby": at_limit["standby"], "cruise": at_limit["cruise"], "ramp": []},
        ratio_max=1.05,
        alpha=0.05,
    )
    assert missing.detail["checks"]["all_regimes_present"] is False
    assert not missing.passed


# ── absolute LTAS ───────────────────────────────────────────────────────────


def test_absolute_ltas_sees_a_level_change_that_the_shape_form_hides() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(32000)
    quiet = x * 0.5
    dev = RE.ltas_deviation_db(x[None, :], quiet[None, :])
    assert dev["level_offset_db"] == pytest.approx(20 * np.log10(0.5), abs=0.05)
    assert dev["mean_abs_db"] == pytest.approx(abs(20 * np.log10(0.5)), abs=0.05)
    # the secondary shape-only diagnostic is blind to it, which is why it is secondary
    assert dev["shape_only_mean_abs_db"] < 1e-6


# ── the conditional-on-FLY124 gates and missing tolerances ──────────────────


def test_conditional_block_gate_needs_mean_and_upper_bound_within_tolerance() -> None:
    base = {"w1": -100.0, "w2": -100.0, "w3": -100.0}
    clean = RE.conditional_non_regression_gate(
        base,
        {"w1": -101.0, "w2": -100.5, "w3": -101.5},
        tolerance=0.0,
        alpha=0.05,
        name="m_nll",
        quantity="composite",
        recording="FLY124",
    )
    assert clean.passed
    assert "conditional on FLY124" in clean.detail["conditioning"]
    noisy = RE.conditional_non_regression_gate(
        base,
        {"w1": -140.0, "w2": -60.0, "w3": -101.0},
        tolerance=0.0,
        alpha=0.05,
        name="m_nll",
        quantity="composite",
        recording="FLY124",
    )
    assert noisy.detail["checks"]["mean_within_tolerance"] is True
    assert noisy.detail["checks"]["upper_bound_within_tolerance"] is False
    assert not noisy.passed


def test_a_missing_tolerance_fails_the_gate_instead_of_waving_it_through() -> None:
    for gate in (
        RE.ltas_gate({"a": 1.0, "b": 1.0}, {"a": 0.1, "b": 0.1}, tolerance_db=None),
        RE.composite_delta_gate(
            {"a": -1.0, "b": -1.0}, {"a": -9.0, "b": -9.0}, alpha=0.05, tolerance=None
        ),
        RE.conditional_non_regression_gate(
            {"w": 1.0},
            {"w": 0.0},
            tolerance=None,
            alpha=0.05,
            name="m",
            quantity="q",
            recording="FLY124",
        ),
    ):
        assert not gate.passed
        assert gate.detail["checks"] == dict(tolerance_available=False)
        assert "never defaulted" in gate.detail["rule"]


# ── the pre-registered regime supports ──────────────────────────────────────


def _support(mask: list[bool], *, regime: str = "ramp", start: float = 10.0) -> RE.RegimeSupport:
    return RE.RegimeSupport(
        window=RE.Window("FLY124", start, len(mask) / RE.SR, regime=regime),
        regime=regime,
        min_rps=45.0,
        max_rps=65.0,
        sample_mask=np.asarray(mask, dtype=bool),
        sr=RE.SR,
    )


def test_regime_support_reads_the_band_from_the_raw_telemetry() -> None:
    n = 4 * RE.SR
    ref = np.tile(np.linspace(30.0, 90.0, n), (4, 1))  # a standby -> cruise sweep
    sup = RE.regime_support(
        RE.Window("FLY124", 0.0, 4.0, regime="ramp"),
        ref,
        regime="ramp",
        min_rps=45.0,
        max_rps=65.0,
    )
    inside = (ref[0] >= 45.0) & (ref[0] <= 65.0)
    assert sup.sample_mask.tolist() == inside.tolist()
    assert sup.scored_seconds == pytest.approx(inside.sum() / RE.SR)
    assert len(sup.intervals()) == 1
    a, b = sup.intervals()[0]
    assert b - a == pytest.approx(sup.scored_seconds, abs=1.0 / RE.SR)
    # one rotor out of band takes the sample out of the support
    ref2 = ref.copy()
    ref2[2] = 90.0
    assert (
        RE.regime_support(
            RE.Window("FLY124", 0.0, 4.0), ref2, regime="ramp", min_rps=45.0, max_rps=65.0
        ).n_scored
        == 0
    )


def test_output_frames_follow_the_trackers_own_alignment() -> None:
    centres = RE.output_frame_centres(4, 4 * RE.SR)
    assert centres == pytest.approx([0.5, 1.5, 2.5, 3.5])
    sup = _support([True] * (2 * RE.SR) + [False] * (2 * RE.SR), start=0.0)
    keep = RE.frames_in_support(4, sup, n_samples=4 * RE.SR)
    assert keep.tolist() == [True, True, False, False]
    assert RE.frames_in_support(4, None, n_samples=4 * RE.SR).tolist() == [True] * 4


def _fake_score(bad_from: int) -> Any:
    """A scorer whose error jumps after frame ``bad_from`` — so masking shows."""

    def score(fm: Any, metric: Any, audio: np.ndarray, rps: np.ndarray, sr: int, mic: int):
        n_frames = 8
        truth = np.full((1, n_frames), 50.0)
        pred = truth.copy()
        pred[:, bad_from:] += 10.0
        return pred, truth, float(np.abs(pred - truth).mean())

    return score


def test_pit_mae_scores_only_the_declared_support() -> None:
    """Half the window is in-regime and perfectly tracked, half is out of it and
    badly tracked: the whole-window average is 5, the ramp MAE is 0."""
    n = 8 * RE.SR
    audio = np.zeros((2, n))
    ref = np.full((1, n), 50.0)
    sup = _support([True] * (n // 2) + [False] * (n - n // 2), start=0.0)
    scored = RE.pit_mae(
        _fake_score(4),
        None,
        None,
        audio,
        ref,
        mics=[0, 1],
        expected_samples=n,
        support=sup,
    )
    assert scored["mae"] == pytest.approx(0.0)
    assert scored["n_scored_frames"] == 4 and scored["scored_fraction"] == pytest.approx(0.5)
    assert scored["support"]["regime"] == "ramp"
    assert len(scored["scored_frame_centres_s"]) == 4
    whole = RE.pit_mae(
        _fake_score(4), None, None, audio, ref, mics=[0, 1], expected_samples=n, support=None
    )
    assert whole["mae"] == pytest.approx(5.0)


def test_pit_mae_refuses_a_short_or_narrow_arm() -> None:
    n = 8 * RE.SR
    ref = np.full((1, n), 50.0)
    with pytest.raises(ValueError, match="different timeline"):
        RE.pit_mae(
            _fake_score(8),
            None,
            None,
            np.zeros((2, n - 1000)),
            ref,
            mics=[0, 1],
            expected_samples=n,
        )
    with pytest.raises(ValueError, match="fewer microphones"):
        RE.pit_mae(
            _fake_score(8), None, None, np.zeros((1, n)), ref, mics=[0, 1], expected_samples=n
        )
    # a stated tolerance is honoured, and only up to its own size
    ok = RE.pit_mae(
        _fake_score(8),
        None,
        None,
        np.zeros((2, n - 1)),
        ref,
        mics=[0, 1],
        expected_samples=n,
        sample_tolerance=1,
    )
    assert ok["n_samples"] == n - 1


def test_pit_mae_refuses_a_support_that_selects_no_frame() -> None:
    n = 8 * RE.SR
    empty = _support([False] * n, start=0.0)
    with pytest.raises(ValueError, match="selects no output frame"):
        RE.pit_mae(
            _fake_score(8),
            None,
            None,
            np.zeros((2, n)),
            np.full((1, n), 50.0),
            mics=[0, 1],
            expected_samples=n,
            support=empty,
        )


# ── model-family dispatch, provenance and leakage ───────────────────────────


def _c2_diagnostics(*, valid: bool = True) -> dict[str, Any]:
    return {
        "shared_phase_evidence": "not_identified_by_marginal_score",
        "marginal_fit": {"valid": valid, "finite_gradients": True, "final_loss": 1.0},
        "carrier_fit": {"valid": valid, "finite_gradients": True, "final_loss": 1.0},
    }


def _revised_export(path: Path, *, clips: list[dict[str, Any]], digest: str = "abc") -> Path:
    path.write_text(
        json.dumps(
            dict(
                schema_version=1,
                model_family="shared_shaft_ou",
                fit_method="marginal_then_carrier",
                lambda_source="fixed_reference",
                shared_phase_evidence="not_identified_by_marginal_score",
                rig_id="michaels",
                parameters=dict(lam=6.0, sigma=1.0, d_scalar=1.0, profile_db=[[0.0, -3.0], [0.0, -3.0]]),
                training_provenance=dict(manifest_sha256=digest, clips=clips),
                diagnostics=_c2_diagnostics(),
            )
        )
    )
    return path


def test_model_family_is_the_dispatch_test_and_each_door_refuses_the_other(
    tmp_path: Path,
) -> None:
    legacy = _export(tmp_path / "legacy.json", ["a"], [0.0], data=True)
    revised = _revised_export(
        tmp_path / "revised.json",
        clips=[dict(recording="FLY125", start_s=16.0, seconds=16.0, regime="cruise")],
    )
    assert RE.export_family(json.loads(legacy.read_text())) == "legacy"
    assert RE.export_family(json.loads(revised.read_text())) == "shared_shaft_ou"
    with pytest.raises(ValueError, match="revised export"):
        RE.read_export(revised, family="candidate")
    with pytest.raises(ValueError, match="legacy descriptive export"):
        RE.read_candidate_export(legacy)
    cand = RE.read_candidate_export(revised)
    assert cand.n_rotors == 2 and cand.n_orders == 2
    assert [w.key for w in cand.training_windows()] == ["FLY125@16.000000+16.000000"]
    assert cand.provenance()["fit_manifest_sha256"] == "abc"
    assert cand.digest["sha256"] and cand.digest["bytes"] > 0


def test_composite_temperature_is_j_over_h_and_fails_on_bad_calibration() -> None:
    """T = J/H on unnormalized weights; J <= 0 fails."""
    rec = RE.composite_temperature(
        [1.0, 2.0, 3.0], exposure_h=10.0, master_seed=0, clips=["a"], rig="r"
    )
    assert rec["ok"] is True
    assert rec["temperature"] == pytest.approx(np.var([1.0, 2.0, 3.0], ddof=1) / 10.0, rel=1e-9)
    assert rec["application"] == "the fit divides: L / T"
    bad = RE.composite_temperature(
        [1.0, 1.0, 1.0], exposure_h=10.0, master_seed=0, clips=["a"], rig="r"
    )
    assert bad["ok"] is False
    assert bad["temperature"] is None
    one = RE.composite_temperature([1.0], exposure_h=10.0, master_seed=0, clips=["a"], rig="r")
    assert one["ok"] is False


def test_unnormalized_gain_total_weights_draws_like_the_composite() -> None:
    """Gain terms carry the same a_i = hop/n_fft duplicate-split weights."""
    items = [
        RE.FrameScore(
            RE.Window("r", 0.0, 1.0, regime="c"), np.array([0.5]), np.array([0.0]), 8, 16, 1024
        )
    ]
    draws = [[np.array([2.0])]]
    out = RE.unnormalized_gain_total(items, draws)
    assert out["exposure_h"] == pytest.approx(1024.0 / 16.0 * 8)
    assert out["u_by_draw"] == pytest.approx([2.0 * 1024.0 / 16.0])


def test_revised_export_must_declare_valid_c2_contract(tmp_path: Path) -> None:
    """C2 is accepted by its marginal-then-carrier contract, not a fake C1
    dynamics-identification flag."""
    p = _revised_export(
        tmp_path / "bad_contract.json",
        clips=[dict(recording="FLY125", start_s=16.0, seconds=16.0, regime="cruise")],
    )
    data = json.loads(p.read_text())
    data["diagnostics"]["identified"] = False
    p.write_text(json.dumps(data))
    assert RE.read_candidate_export(p).fit_contract["fit_method"] == "marginal_then_carrier"

    for field, value, pattern in (
        ("fit_method", "moment_then_map", "fit_method=.*marginal_then_carrier"),
        ("lambda_source", "estimated", "lambda_source='fixed_reference'"),
        ("shared_phase_evidence", "identified", "must not claim causal sharing"),
    ):
        data = json.loads(_revised_export(p, clips=[dict(recording="FLY125", start_s=16.0, seconds=16.0, regime="cruise")]).read_text())
        data[field] = value
        p.write_text(json.dumps(data))
        with pytest.raises(ValueError, match=pattern):
            RE.read_candidate_export(p)
    data = json.loads(p.read_text())
    data["fit_method"] = "marginal_then_carrier"
    data["lambda_source"] = "fixed_reference"
    data["shared_phase_evidence"] = "not_identified_by_marginal_score"
    data["diagnostics"]["marginal_fit"]["valid"] = False
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="diagnostics.marginal_fit.valid"):
        RE.read_candidate_export(p)


def test_training_leakage_catches_an_overlapping_fit_support() -> None:
    train = [RE.Window("FLY124", 100.0, 16.0, role="candidate_training")]
    scored_clean = [RE.Window("FLY124", 130.0, 8.0)]
    scored_dirty = [RE.Window("FLY124", 110.0, 8.0)]
    assert RE.training_leakage(train, scored_clean)["clean"]
    dirty = RE.training_leakage(train, scored_dirty)
    assert not dirty["clean"] and dirty["overlaps"][0]["overlap_s"] == pytest.approx(6.0)
    # the analysis/filter guard widens the training span, so a near miss counts
    near = [RE.Window("FLY124", 116.5, 8.0)]
    assert RE.training_leakage(train, near)["clean"]
    assert not RE.training_leakage(train, near, guard_seconds=1.024)["clean"]
    # a different recording never collides
    assert RE.training_leakage(train, [RE.Window("FLY125", 100.0, 16.0)])["clean"]


# ── the declared render-unit conversion ─────────────────────────────────────


def test_to_renderer_units_applies_the_rate_factor_to_levels_only(tmp_path: Path) -> None:
    b = RE.read_export(_export(tmp_path / "e.json", ["a"], [0.0], data=True), family="refined")
    mp = RE.aggregate_nuisance(b, list(b.clip_ids), label="agg")
    phys = RE.to_renderer_units(mp)
    off = RE.RENDER_RATE_FACTOR_DB
    assert off == pytest.approx(10 * np.log10(64000 / 16000), abs=1e-9)
    assert phys.params["floor_mean_db"] == pytest.approx(mp.params["floor_mean_db"] + off)
    assert np.asarray(phys.params["profile_db"]) == pytest.approx(
        np.asarray(mp.params["profile_db"]) + off
    )
    # relative patterns and non-power fields are untouched
    assert phys.params["mic_floor_db"] == mp.params["mic_floor_db"]
    assert phys.params["gamma0"] == mp.params["gamma0"]
    assert phys.source["render_units"]["normalize_rms"].startswith("None")
    assert phys.source["render_units"]["applied_to"] == ["profile_db", "floor_mean_db"]


# ── the candidate dispatch, against the REAL candidate API ──────────────────


def _runner() -> Any:
    """``scripts/stochastic_fit_revised_eval.py``, imported by path.

    Registered in ``sys.modules`` first, because its ``@dataclass`` bodies are
    resolved through the module name.
    """
    import importlib.util
    import sys

    name = "_revised_eval_runner"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[2] / "scripts" / "stochastic_fit_revised_eval.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _minimal_revised_export(path: Path, *, n_fft: int, hop: int) -> Path:
    """A hand-written export of the schema the candidate's fit emits.

    Deliberately NOT imported from the candidate's own test module: this is the
    contract the evaluator dispatches on, so it is written out here and handed
    to the real ``revised_phase`` functions.
    """
    from experiments.stochastic_fit import revised_phase as RP

    profile = [-6.0, -12.0, -18.0, -24.0]
    path.write_text(
        json.dumps(
            dict(
                schema_version=RP.SCHEMA_VERSION,
                model_family=RP.MODEL_FAMILY,
                fit_method="marginal_then_carrier",
                lambda_source="fixed_reference",
                shared_phase_evidence="not_identified_by_marginal_score",
                rig_id="test_rig",
                parameters=dict(
                    lam=6.0,
                    sigma=6.0,
                    d_scalar=1.0,
                    delay_s={"0": 0.0},
                    bias_hz={},
                    bias_mean_hz=[0.5],
                    bias_prior_std_hz=0.5,
                    profile_db=[profile],
                    amp_exp=0.0,
                    floor_mean_db=-60.0,
                    floor_shape_db=[0.0] * 14,
                    floor_ctrl_hz=np.geomspace(30.0, 8000.0, 14).tolist(),
                    floor_tilt_db_oct=0.0,
                    floor_exp=0.0,
                    floor_static_rel=0.0,
                    mic_gain_db=[[0.0], [0.0]],
                    gain_all_db=[0.0, 0.0],
                    mic_floor_db=[0.0, 0.0],
                    k_cap=len(profile),
                    n_mics=2,
                    n_rotors=1,
                ),
                training_provenance=dict(
                    manifest_path=None,
                    manifest_sha256="fit-manifest-digest",
                    clips=[
                        dict(
                            clip_id="fly125_cruise_00",
                            dataset="michaels-frames",
                            recording="FLY125",
                            start_s=16.0,
                            seconds=16.0,
                            channels=[0, 1],
                            rps_key="rps",
                            regime="cruise",
                        )
                    ],
                    front_end=dict(
                        n_fft=n_fft,
                        hop=hop,
                        sr=RE.SR,
                        band_hz=[30.0, 7900.0],
                        sample_rate_work=4 * RE.SR,
                    ),
                    model_grid_hz=4 * RE.SR,
                    analysis_grid_hz=RE.SR,
                    state_rate_hz=1000.0,
                    optimizer=dict(iters=1, lr=0.05, frame_chunk=2, atom_dtype="float64"),
                    seed=0,
                    composite_temperature=1.0,
                    priors=dict(bias_std_hz=0.5, log_d_mean=0.0, log_d_std=2.0),
                ),
                diagnostics=_c2_diagnostics(),
            )
        )
    )
    return path


def test_the_candidate_arm_dispatches_to_the_real_revised_api(tmp_path: Path) -> None:
    """A ``model_family`` export must reach ``render_revised`` /
    ``predict_spectrum`` — audio, the diagnostic ``physical_rps`` and a spectrum
    on the frozen observation grid — and never the legacy renderer."""
    from experiments.stochastic_fit.data import Clip

    runner = _runner()
    n_fft, hop = 2048, 512  # a small front end keeps the test cheap
    export = _minimal_revised_export(tmp_path / "candidate.json", n_fft=n_fft, hop=hop)
    spec = dict(kind="revised_export", export=str(export), label="candidate 1")
    model = runner.arm_model("candidate", spec, regime="cruise", recording="FLY124")
    assert model.kind == "revised"
    assert model.source()["model_family"] == "shared_shaft_ou"
    assert "predict_spectrum" in model.source()["observation_law"]

    n = 3 * n_fft
    rps = np.full((1, n), 300.0)
    audio, physical, diag = model.render(rps, n_mics=2, seed=7)
    assert audio.shape == (2, n)
    assert physical is not None and physical.shape == (1, n)
    assert diag["eps_in_physical_rps"] is False  # the diagnostic excludes eps

    # and the spectrum comes back on the FROZEN observation grid, through the
    # same ArmModel the runner uses
    long_rps = np.full((1, RE.OBS_N_FFT + RE.OBS_HOP), 300.0)
    long_audio, _, _ = model.render(long_rps, n_mics=2, seed=7)
    clip = Clip("c", "synthetic", long_audio.astype(np.float32), long_rps, RE.SR)
    m = model.spectrum(clip, n_mics=2)
    expected = np.asarray(RE.window_periodogram(clip).power).shape
    assert m is not None and m.shape == (2, expected[1], expected[2])
    assert np.isfinite(m).all() and float(m.max()) > 0.0


def test_the_candidate_leakage_guard_pins_the_fit_manifest(tmp_path: Path) -> None:
    runner = _runner()
    export = _minimal_revised_export(tmp_path / "candidate.json", n_fft=2048, hop=512)
    scored_clean = [RE.Window("FLY124", 40.0, 8.0, regime="cruise")]
    spec = dict(
        kind="revised_export", export=str(export), fit_manifest_sha256="fit-manifest-digest"
    )
    report = runner.candidate_leakage_guard(spec, scored_clean, cohort_name="michaels_fly124")
    assert report["clean"] and report["guard_seconds"] == pytest.approx(RE.OBS_N_FFT / RE.SR)
    # a wrong or missing fit-manifest digest is a hard stop
    with pytest.raises(SystemExit, match="fit_manifest_sha256"):
        runner.candidate_leakage_guard(
            dict(kind="revised_export", export=str(export)), scored_clean, cohort_name="c"
        )
    with pytest.raises(SystemExit, match="was fitted under manifest"):
        runner.candidate_leakage_guard(
            dict(kind="revised_export", export=str(export), fit_manifest_sha256="other"),
            scored_clean,
            cohort_name="c",
        )
    with pytest.raises(SystemExit, match="belongs in observation"):
        runner.candidate_leakage_guard(
            spec | {"training_guard_seconds": 0.0}, scored_clean, cohort_name="c"
        )
    # and a scored window that the candidate trained on is refused
    with pytest.raises(SystemExit, match="intersect scored held-out supports"):
        runner.candidate_leakage_guard(
            spec, [RE.Window("FLY125", 20.0, 8.0, regime="cruise")], cohort_name="c"
        )


# ── LTAS window geometry: one complete window, exact-N, 0.99 s ramp ─────────


def test_absolute_ltas_bands_accepts_exactly_one_window() -> None:
    """8192 samples must yield one Welch window; the old ``range(0, size-n, ...)``
    gave ``range(0, 0, ...)`` and raised."""
    x = np.random.default_rng(101).standard_normal(8192)
    bands = RE.absolute_ltas_bands(x)
    assert bands.shape == (len(stats.BANDS),)
    assert np.isfinite(bands).all()


def test_absolute_ltas_bands_fits_0_99s_ramp() -> None:
    """0.99 s at 16 kHz = 15840 samples. Two complete 8192 windows fit
    (starts 0 and 4096), so the LTAS is available."""
    n = int(0.99 * RE.SR)
    assert n == 15840
    x = np.random.default_rng(102).standard_normal(n)
    bands = RE.absolute_ltas_bands(x)
    assert bands.shape == (len(stats.BANDS),)


def test_ltas_deviation_derives_relative_from_absolute_geometry() -> None:
    """A pure level change is invisible to the shape-only diagnostic because it
    is derived from the same absolute geometry, not recomputed."""
    rng = np.random.default_rng(103)
    x = rng.standard_normal(32000)
    quiet = x * 0.5
    dev = RE.ltas_deviation_db(x[None, :], quiet[None, :])
    assert dev["shape_only_mean_abs_db"] < 1e-6


def test_scored_ltas_allows_0_99s_ramp_and_refuses_shorter() -> None:
    """The gate needs one complete 8192-sample Welch window, not two."""
    runner = _runner()
    ok_n = int(0.99 * RE.SR)
    real = np.random.default_rng(104).standard_normal((1, ok_n))
    arm = real.copy()
    support = RE.RegimeSupport(
        window=RE.Window("FLY124", 0.0, ok_n / RE.SR, regime="ramp"),
        regime="ramp",
        min_rps=45.0,
        max_rps=65.0,
        sample_mask=np.ones(ok_n, dtype=bool),
        sr=RE.SR,
    )
    out = runner.scored_ltas(real, arm, support)
    assert out is not None
    assert out["scored_span_samples"] == [0, ok_n]

    short = 8191
    support_short = RE.RegimeSupport(
        window=RE.Window("FLY124", 0.0, short / RE.SR, regime="ramp"),
        regime="ramp",
        min_rps=45.0,
        max_rps=65.0,
        sample_mask=np.ones(short, dtype=bool),
        sr=RE.SR,
    )
    assert runner.scored_ltas(real[:, :short], arm[:, :short], support_short) is None


# ── protocol fingerprint: candidate/output are excluded, design is hashed ───


def _base_manifest(tmp_path: Path) -> dict[str, Any]:
    return {
        "schema": "revised-phase-baseline/1",
        "status": "PROPOSED",
        "notes": ["a note"],
        "scorer": {"experiment": "hppnet_l2_r2_s0", "ckpt": "best"},
        "observation": {"sr": 16000, "n_fft": 16384, "hop": 1024, "f_min": 30.0, "f_max": 7900.0, "training_guard_seconds": 1.024},
        "gates": {
            "alpha": 0.05,
            "bootstrap_seed": 0,
            "dregon_gap_fraction": 0.7,
            "michaels_ratio_max": 1.05,
            "composite_tolerance": 0.0,
        },
        "null_variation": {"seeds": [2001, 2002, 2003, 2004], "metrics": ["pit_mae", "ltas_abs_db"]},
        "cohorts": [],
        "arms": {"real": {"kind": "real"}, "baseline": {"kind": "export_render"}},
        "candidate_arm_template": {"kind": "revised_export"},
        "out_dir": "results/revised_phase/baseline_v1",
    }


def test_protocol_fingerprint_excludes_run_inputs_only(tmp_path: Path) -> None:
    runner = _runner()
    man = _base_manifest(tmp_path)
    man["_path"] = str(tmp_path / "manifest.json")
    man["_digest"] = hashlib.sha256(json.dumps(man).encode()).hexdigest()
    base = runner.protocol_fingerprint(man)

    # Adding a candidate, changing output path, or mutating authoring notes
    # must NOT change the protocol fingerprint.
    man["arms"]["candidate"] = {
        "kind": "revised_export",
        "export": str(tmp_path / "candidate.json"),
        "fit_manifest_sha256": "abc",
    }
    man["out_dir"] = "results/revised_phase/round_1"
    man["notes"] = ["a different note"]
    assert runner.protocol_fingerprint(man) == base

    # But a guard, gate threshold, seed, or cohort change DOES change it.
    man["observation"]["training_guard_seconds"] = 0.0
    assert runner.protocol_fingerprint(man) != base
    man["observation"]["training_guard_seconds"] = 1.024
    man["gates"]["dregon_gap_fraction"] = 0.8
    assert runner.protocol_fingerprint(man) != base


def test_protocol_fingerprint_adopts_old_calibration_manifest(tmp_path: Path) -> None:
    """A calibration recorded before protocol fingerprints can be adopted by
    verifying its original manifest digest and deriving the protocol hash."""
    import json as _json

    runner = _runner()
    man = _base_manifest(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(_json.dumps(man))
    man["_path"] = str(path)
    man["_digest"] = hashlib.sha256(path.read_bytes()).hexdigest()

    # Simulate an old calibration record with no protocol_sha256.
    cal = {
        "manifest": {"path": str(path), "sha256": man["_digest"]},
        "frozen_inputs": {"manifest": {"path": str(path), "sha256": man["_digest"]}, "scorer": {"sha256": "same"}, "exports": {}, "datasets": {}},
    }
    sha = runner.verify_protocol_fingerprint(man, cal)
    assert sha == runner.protocol_fingerprint(man)


# ── per-rig candidate mapping and rig_id validation ─────────────────────────


def _set_rig_id(path: Path, rig_id: str) -> None:
    data = json.loads(path.read_text())
    data["rig_id"] = rig_id
    path.write_text(json.dumps(data))


def _set_fit_manifest_sha(path: Path, sha: str) -> None:
    data = json.loads(path.read_text())
    data["training_provenance"]["manifest_sha256"] = sha
    path.write_text(json.dumps(data))


def test_per_rig_candidate_selects_export_by_cohort_rig(tmp_path: Path) -> None:
    runner = _runner()
    dregon = _minimal_revised_export(tmp_path / "dregon.json", n_fft=2048, hop=512)
    michaels = _minimal_revised_export(tmp_path / "michaels.json", n_fft=2048, hop=512)
    _set_rig_id(dregon, "dregon")
    _set_rig_id(michaels, "michaels")
    spec = {
        "kind": "revised_export",
        "per_rig": {
            "dregon": {"export": str(dregon), "fit_manifest_sha256": "fit-d"},
            "michaels": {"export": str(michaels), "fit_manifest_sha256": "fit-m"},
        },
    }
    d = runner.arm_model("cand", spec, regime="cruise", recording="x", rig="dregon")
    assert d.candidate is not None and d.candidate.rig_id == "dregon"
    m = runner.arm_model("cand", spec, regime="cruise", recording="x", rig="michaels")
    assert m.candidate is not None and m.candidate.rig_id == "michaels"
    with pytest.raises(SystemExit, match="no candidate export declared"):
        runner.arm_model("cand", spec, regime="cruise", recording="x", rig="other")


def test_per_rig_candidate_validates_export_rig_id_against_cohort(tmp_path: Path) -> None:
    runner = _runner()
    export = _minimal_revised_export(tmp_path / "michaels.json", n_fft=2048, hop=512)
    _set_rig_id(export, "michaels")
    spec = {"kind": "revised_export", "export": str(export), "fit_manifest_sha256": "fit-m"}
    # single-export form also validates when a rig is supplied
    with pytest.raises(SystemExit, match="rig_id"):
        runner.arm_model("cand", spec, regime="cruise", recording="x", rig="dregon")
    # and the same via per_rig
    spec_per_rig = {
        "kind": "revised_export",
        "per_rig": {"dregon": {"export": str(export), "fit_manifest_sha256": "fit-m"}},
    }
    with pytest.raises(SystemExit, match="rig_id"):
        runner.arm_model("cand", spec_per_rig, regime="cruise", recording="x", rig="dregon")


def test_per_rig_candidate_leakage_guard_pins_fit_manifest(tmp_path: Path) -> None:
    runner = _runner()
    dregon = _minimal_revised_export(tmp_path / "dregon.json", n_fft=2048, hop=512)
    _set_rig_id(dregon, "dregon")
    _set_fit_manifest_sha(dregon, "fit-d")
    spec = {
        "kind": "revised_export",
        "per_rig": {
            "dregon": {"export": str(dregon), "fit_manifest_sha256": "fit-d"},
        },
    }
    scored = [RE.Window("free-flight_nosource_room2", 0.0, 8.0, regime="cruise")]
    report = runner.candidate_leakage_guard(spec, scored, cohort_name="dregon_room2_cruise", rig="dregon")
    assert report["clean"]
    spec_override = {
        "kind": "revised_export",
        "per_rig": {
            "dregon": {
                "export": str(dregon),
                "fit_manifest_sha256": "fit-d",
                "training_guard_seconds": 0.0,
            },
        },
    }
    with pytest.raises(SystemExit, match="belongs in observation"):
        runner.candidate_leakage_guard(
            spec_override, scored, cohort_name="dregon_room2_cruise", rig="dregon"
        )
    assert report["candidate"]["fit_manifest_sha256"] == "fit-d"
    # wrong rig: a michaels entry pointing to the dregon export must fail rig_id validation
    spec_mismatch = {
        "kind": "revised_export",
        "per_rig": {
            "michaels": {"export": str(dregon), "fit_manifest_sha256": "fit-d"},
        },
    }
    with pytest.raises(SystemExit, match="rig_id"):
        runner.candidate_leakage_guard(spec_mismatch, scored, cohort_name="michaels_fly124", rig="michaels")
