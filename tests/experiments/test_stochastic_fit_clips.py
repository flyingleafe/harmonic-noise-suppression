"""Clip selection against a label that does not cover the whole audio, and the
one-rig guard on a pooled fit.

DREGON's telemetry starts 2.5-6.5 s after its audio and stops before it ends,
with the last sample still in cruise (room2: +5.223 s in, 2.873 s short, last
sample 69-76 rev/s). ``np.interp`` holds those endpoint speeds flat, so an
uncovered window used to read as steady cruise and pass the regime test with a
carrier nothing measured. These are the properties that forbid it, plus the
refusal to tie one rig's parameters to another rig's clips.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.stochastic_fit.campaign import assert_one_rig
from experiments.stochastic_fit.clips import Recording, windows

SR = 16000.0
#: audio 0..40 s, label 5..30 s: late start AND early stop, the DREGON shape.
AUDIO_S, LABEL_START_S, LABEL_END_S = 40.0, 5.0, 30.0


def _recording(t_start: float = 0.0) -> Recording:
    """Four rotors held at 80 rev/s over the label's span only."""
    n = int(AUDIO_S * SR)
    rps_t = np.arange(LABEL_START_S, LABEL_END_S, 0.032) + t_start
    return Recording(
        dataset="test-frames",
        version=None,
        recording_id="planted",
        group="dregon_room2",
        audio=np.zeros((8, n), dtype=np.float32),
        sr=SR,
        t_start=t_start,
        rps=np.full((4, rps_t.size), 80.0),
        rps_t=rps_t,
        rps_key="rps_refined",
        meta={},
    )


def test_coverage_is_audio_intersected_with_the_label() -> None:
    rec = _recording(t_start=1000.0)
    lo, hi = rec.coverage
    assert (lo - rec.t_start, hi - rec.t_start) == pytest.approx(
        (LABEL_START_S, LABEL_END_S - 0.032), abs=0.04
    )


def test_rps_at_refuses_to_invent_a_carrier_past_the_label() -> None:
    rec = _recording()
    inside = np.linspace(LABEL_START_S + 1.0, LABEL_END_S - 1.0, 100)
    assert rec.rps_at(inside).shape == (4, 100)
    for outside in (
        np.linspace(0.0, 8.0, 100),  # before the label starts
        np.linspace(LABEL_END_S - 1.0, AUDIO_S, 100),  # past its last stamp
    ):
        with pytest.raises(ValueError, match="leaves the rps_refined span"):
            rec.rps_at(outside)


def test_windows_drop_the_uncovered_ones_and_keep_the_audio_anchored_grid() -> None:
    """The grid is anchored to the AUDIO start, so offsets do not move.

    A shifted grid would silently renumber every clip of every earlier fit;
    what an uncovered span may do is remove a window, never move one.
    """
    rec = _recording(t_start=1000.0)
    got = [round(s - rec.t_start, 3) for s, _ in windows(rec, seconds=8.0, max_clips=99)]
    # 8 s windows on the audio grid: 0, 8, 16, 24, 32 — only 8 and 16 sit
    # inside [5, 30); 24+8 = 32 runs past the label's end.
    assert got == [8.0, 16.0]

    # the regime test must not be what rejects them: every covered window here
    # is a steady 80 rev/s, so a cruise floor keeps exactly the same two
    assert [round(s - rec.t_start, 3) for s, _ in windows(rec, seconds=8.0, min_rps=65.0)] == got
    # and a standby band (20-45) keeps none of them
    assert windows(rec, seconds=8.0, min_rps=20.0, max_rps=45.0) == []


def test_a_recording_with_no_label_covers_its_whole_audio() -> None:
    """The bench publishes no rotor track; its span is the audio's."""
    rec = _recording()
    bench = type(rec)(**{**rec.__dict__, "rps": None, "rps_t": None, "rps_key": None})
    assert bench.coverage == (0.0, AUDIO_S)
    assert len(windows(bench, seconds=8.0, max_clips=99)) == 5


# ── the one-rig guard on a pooled fit ────────────────────────────────────────


class _Clip:
    def __init__(self, channels: int, rotors: int) -> None:
        self.audio = np.zeros((channels, 10))
        self.rps = np.zeros((rotors, 10))


def _row(group: str, channels: int = 8, rotors: int = 4) -> tuple[str, str, object, None]:
    return (f"{group}_00", group, _Clip(channels, rotors), None)


def test_one_rig_may_pool_several_recordings_of_that_rig() -> None:
    """Two rooms are the same airframe; two flights are the same rig."""
    assert_one_rig([_row("dregon_room2"), _row("dregon_room1")]) is None
    assert_one_rig([_row("fly125"), _row("fly124")]) is None


def test_pooling_two_rigs_is_refused() -> None:
    """`fit_rig` ties profile, widths, floor shape and mic gains across every
    clip from the FIRST clip's spec, so a mixed set would tie DREGON's rig
    parameters to Michael's clips instead of fitting two rigs."""
    with pytest.raises(ValueError, match="cannot pool"):
        assert_one_rig([_row("dregon_room2"), _row("fly125")])


def test_pooling_two_geometries_is_refused() -> None:
    with pytest.raises(ValueError, match="disagree on"):
        assert_one_rig([_row("dregon_room2"), _row("dregon_room2", channels=4)])
