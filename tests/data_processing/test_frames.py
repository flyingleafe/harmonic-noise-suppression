"""Frame conventions: which published entry carries the rotor speed.

The preference order is a *data* decision with two consumers — the noise-pool
adapter (:func:`data_processing.frames.adapt_recording_frame`) and the plotting
alias table (:data:`data_processing.canonical.ENTRY_ALIASES`) — so it is pinned
here once.

The same adapter also decides WHICH of the two published label tracks a
training pool is fitted to (raw telemetry vs the regime-gated ``rps_refined``),
which is a whole experiment's independent variable — so the selection and its
refusal to fall back are pinned here too.
"""

from __future__ import annotations

import numpy as np
import pytest
import tdseries as td

from data_processing.canonical import ENTRY_ALIASES
from data_processing.frames import PUBLISHED_RPS_KEYS, adapt_recording_frame, get_meta

SR = 16000
N_MOTOR = 32


def _audio() -> td.Series:
    rng = np.random.default_rng(0)
    return td.uniform(
        (0.05 * rng.standard_normal((2, SR))).astype(np.float32),
        SR,
        dims=("mic", "time"),
        t_start=0.0,
    )


def _motors(base: float) -> td.Series:
    values = np.tile(np.arange(4, dtype=np.float32)[:, None] + base, (1, N_MOTOR))
    return td.events(
        np.linspace(0.0, 0.99, N_MOTOR), values, dims=("rotor", "time"), t_start=0.0, t_end=1.0
    )


def test_published_rps_key_preference_order():
    """``rps`` > ``motors_measured`` > ``motors_command``.

    ``motors_measured`` is the real tachometer and the track the beat-VK
    protocol pins as ground truth; ``motors_command`` is only what the flight
    controller asked for and is the fallback for the recordings (most of
    DREGON) that log no measured track.
    """
    assert PUBLISHED_RPS_KEYS == ("rps", "motors_measured", "motors_command")
    # The plotting aliases are the same order minus the canonical name itself,
    # so a frame plots under the track the pipeline reads.
    assert ENTRY_ALIASES["rps"][:2] == ("motors_measured", "motors_command")


@pytest.mark.parametrize(
    ("entries", "expected_base"),
    [
        ({"rps": 70.0, "motors_measured": 55.0, "motors_command": 60.0}, 70.0),
        ({"motors_measured": 55.0, "motors_command": 60.0}, 55.0),
        ({"motors_command": 60.0}, 60.0),
    ],
)
def test_adapt_recording_frame_picks_the_preferred_track(entries, expected_base):
    frame = td.Frame({"audio": _audio(), **{k: _motors(v) for k, v in entries.items()}})

    adapted = adapt_recording_frame(frame, sample_rate=SR)

    assert adapted is not None
    assert set(adapted.keys()) == {"audio", "rps", "meta"}
    np.testing.assert_array_equal(
        np.asarray(adapted["rps"].data), np.asarray(_motors(expected_base).data)
    )


def test_adapt_recording_frame_returns_none_without_a_rotor_track():
    assert adapt_recording_frame(td.Frame({"audio": _audio()}), sample_rate=SR) is None


def _rich_frame() -> td.Frame:
    """A published michaels-style frame: raw ``rps`` and a DIFFERENT refined track."""
    return td.Frame(
        {
            "audio": _audio(),
            "rps": _motors(70.0),
            "rps_refined": _motors(70.5),
            "meta": td.Frame({"recording_id": "FLY125"}),
        }
    )


@pytest.mark.parametrize(
    ("rps_key", "expected_base", "expected_variant"),
    [(None, 70.0, "published"), ("rps_refined", 70.5, "refined")],
)
def test_adapt_recording_frame_selects_the_requested_label_track(
    rps_key, expected_base, expected_variant
):
    """The default stays on the raw telemetry; ``rps_refined`` is opt-in.

    Both tracks are defined on every labelled published recording, so the
    default MUST keep reading the raw one — otherwise every existing config
    would silently change the labels it trains on.
    """
    adapted = adapt_recording_frame(_rich_frame(), sample_rate=SR, rps_key=rps_key)

    assert adapted is not None
    np.testing.assert_array_equal(
        np.asarray(adapted["rps"].data), np.asarray(_motors(expected_base).data)
    )
    # The provenance a trained run reports its labels by.
    assert get_meta(adapted, "label_variant") == expected_variant


def test_adapt_recording_frame_refuses_to_fall_back_to_another_label():
    """A labelled recording missing the demanded track is an error.

    Silently falling through to the raw telemetry would put two label
    definitions in one pool and make a refined-vs-raw comparison meaningless.
    """
    frame = td.Frame({"audio": _audio(), "motors_command": _motors(60.0)})

    with pytest.raises(KeyError, match="rps_refined"):
        adapt_recording_frame(frame, sample_rate=SR, rps_key="rps_refined")


def test_adapt_recording_frame_still_skips_an_unlabelled_recording():
    """An explicit key does not turn the no-rotor-track filter into a crash.

    DREGON's bench runs and the clean-source recordings carry no rotor track at
    all; they are not candidates for either label and are skipped, exactly as
    on the default path.
    """
    frame = td.Frame({"audio": _audio()})

    assert adapt_recording_frame(frame, sample_rate=SR, rps_key="rps_refined") is None
