"""Boundaries of the four-regime frame rule (metrics.regimes)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from metrics.regimes import REGIMES, frame_regimes

FPS = 31.25
ZERO, STANDBY, RAMP, CRUISE = (REGIMES.index(name) for name in REGIMES)


def _labels(track: np.ndarray, rotors: np.ndarray | None = None) -> np.ndarray:
    """Regime codes of one clip whose rotors all follow ``track`` (or ``rotors``)."""
    target = np.repeat(track[None], 4, axis=0) if rotors is None else rotors
    return frame_regimes(torch.as_tensor(target)[None], FPS)[0].numpy()


def test_codes_index_the_regime_names():
    assert (ZERO, STANDBY, RAMP, CRUISE) == (0, 1, 2, 3)
    assert _labels(np.zeros(40)).dtype == np.int8


@pytest.mark.parametrize(
    ("level", "expected"),
    [(0.0, ZERO), (0.99, ZERO), (1.0, STANDBY), (44.9, STANDBY), (45.0, CRUISE)],
)
def test_level_thresholds_on_a_steady_clip(level, expected):
    assert set(_labels(np.full(60, level))) == {expected}


def test_zero_needs_every_rotor_below_one():
    rotors = np.zeros((4, 60))
    rotors[2] = 2.0  # one rotor idling: mean 0.5 < 1, but max >= 1
    assert set(_labels(np.zeros(60), rotors)) == {STANDBY}


def test_ramp_threshold_is_twenty_rev_per_s2_and_beats_level():
    """A linear spin-up is labelled ramp at >= 20 rev/s^2 regardless of level,
    and its interior is the slope exactly (smoothing preserves a line)."""
    t = np.arange(200) / FPS
    interior = slice(20, 180)
    assert set(_labels(60.0 + 20.05 * t)[interior]) == {RAMP}
    assert set(_labels(60.0 + 19.95 * t)[interior]) == {CRUISE}
    assert set(_labels(5.0 + 20.05 * t)[interior]) == {RAMP}  # standby levels: still ramp


def test_derivative_uses_quarter_second_central_difference_after_smoothing():
    """A step from 0 to 80 rev/s marks ramp only within ~(0.25 s + half the
    0.288 s boxcar) of the step; far frames keep their level regime."""
    track = np.where(np.arange(200) < 100, 0.0, 80.0)
    labels = _labels(track)
    ramp = np.flatnonzero(labels == RAMP)
    assert ramp.min() >= 100 - 8 - 4 and ramp.max() <= 99 + 8 + 4
    assert set(labels[:80]) == {ZERO}
    assert set(labels[120:]) == {CRUISE}


def test_regimes_scale_with_frame_rate():
    """The same physical slope is a ramp on any frame grid."""
    for fps in (31.25, 62.5):
        t = np.arange(int(6 * fps)) / fps
        target = torch.as_tensor(np.repeat((60.0 + 30.0 * t)[None], 4, axis=0))[None]
        labels = frame_regimes(target, fps)[0].numpy()
        assert set(labels[int(fps) : -int(fps)]) == {RAMP}
