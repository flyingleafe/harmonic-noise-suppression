"""Behaviour of ``training.stopping.MultiMetricController``: dual-best
bookkeeping per score and the saturation stop on the control score."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from training.stopping import MultiMetricController


def _cfg(**overrides):
    values = {
        "smoothing_window": 5,
        "min_relative_improvement": 0.01,
        "min_rounds": 40,
        "final_patience": 30,
        "control": "real",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _stop_round(controller: MultiMetricController, series) -> int | None:
    """1-based round of the first stop verdict while feeding ``series``."""
    for round_index, scores in enumerate(series, start=1):
        if controller.step(scores).stop_reason is not None:
            return round_index
    return None


def test_flat_control_stops_at_min_rounds_not_after_patience_alone():
    """The first full window (round 5) sets the best; 30 stale rounds are
    reached at round 35, but nothing stops before min_rounds = 40."""
    controller = MultiMetricController(_cfg(), ("real",))
    verdicts = [controller.step({"real": 3.0}) for _ in range(39)]
    assert all(v.stop_reason is None for v in verdicts)
    assert controller.stale_rounds == 34
    assert controller.step({"real": 3.0}).stop_reason == "saturation"


def test_stop_comes_exactly_final_patience_rounds_after_the_last_median_improvement():
    controller = MultiMetricController(_cfg(min_rounds=0), ("real",))
    series = [{"real": 10.0 * 0.97**i} for i in range(60)] + [{"real": 10.0 * 0.97**59}] * 100
    last_improvement = 0
    for round_index, scores in enumerate(series, start=1):
        verdict = controller.step(scores)
        if "real" in verdict.improved_median:
            last_improvement = round_index
        if verdict.stop_reason is not None:
            break
    assert last_improvement >= 60  # improving rounds never count as stale
    assert round_index == last_improvement + 30


def test_only_the_control_score_keeps_a_run_alive():
    """A primary that keeps improving earns checkpoints, not extra rounds."""
    controller = MultiMetricController(_cfg(), ("real", "md2_ramp"))
    series = [{"real": 3.0, "md2_ramp": 8.0 * 0.95**i} for i in range(60)]

    assert _stop_round(controller, series) == 40
    # md2_ramp's median still improved at the stopping round (median of i = 35..39).
    assert controller.best_smoothed["md2_ramp"] == pytest.approx(8.0 * 0.95**37)


def test_raw_best_tracks_single_round_dips_the_median_ignores():
    controller = MultiMetricController(_cfg(smoothing_window=3, min_rounds=0), ("real",))
    first = controller.step({"real": 5.0})
    assert first.improved_raw == ("real",) and first.improved_median == ()  # window filling
    controller.step({"real": 5.0})
    full = controller.step({"real": 5.0})
    assert full.improved_median == ("real",) and full.improved_raw == ()
    dip = controller.step({"real": 4.0})
    assert dip.improved_raw == ("real",)  # raw best: 4.0 < 5.0
    assert dip.improved_median == ()  # median [5, 5, 4] = 5 does not move
    assert controller.best_raw == {"real": 4.0}
    assert controller.best_smoothed == {"real": 5.0}


def test_median_improvement_is_relative_and_scale_free():
    controller = MultiMetricController(_cfg(smoothing_window=1, min_rounds=0), ("real", "md2"))
    controller.step({"real": 100.0, "md2": 1.0})
    small = controller.step({"real": 99.5, "md2": 0.995})  # 0.5 %: raw best, not meaningful
    assert small.improved_median == () and set(small.improved_raw) == {"real", "md2"}
    meaningful = controller.step({"real": 98.0, "md2": 0.98})  # 2 % against the 100 / 1.0 best
    assert set(meaningful.improved_median) == {"real", "md2"}


def test_resumed_controller_reaches_the_same_verdicts():
    series = [{"real": 5.0 - 0.1 * (i % 7), "md2": 3.0 + 0.05 * i} for i in range(70)]
    full = MultiMetricController(_cfg(min_rounds=10, final_patience=8), ("real", "md2"))
    expected = [full.step(scores) for scores in series]

    first = MultiMetricController(_cfg(min_rounds=10, final_patience=8), ("real", "md2"))
    head = [first.step(scores) for scores in series[:23]]
    resumed = MultiMetricController(_cfg(min_rounds=10, final_patience=8), ("real", "md2"))
    resumed.load_state_dict(first.state_dict())
    tail = [resumed.step(scores) for scores in series[23:]]

    assert head + tail == expected


def test_control_must_be_a_tracked_score_and_scores_must_be_finite():
    with pytest.raises(ValueError, match="control"):
        MultiMetricController(_cfg(control="real"), ("dregon", "md2"))
    controller = MultiMetricController(_cfg(), ("real",))
    with pytest.raises(ValueError, match="non-finite"):
        controller.step({"real": float("nan")})
