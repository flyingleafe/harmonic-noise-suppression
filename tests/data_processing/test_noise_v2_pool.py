"""Tests for the fitted rotor-noise source (``kind: noise_v2``)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from data_processing.frames import get_meta
from data_processing.noise_v2_pool import PRESET_BANK_FORMAT, NoiseV2Pool

SR = 16000

DREGON_FIT = "results/noise_v2/rounds/round5/fits/dregon_room2_floor__flight_profile.json"
MICHAELS_CRUISE = "results/noise_v2/rounds/round3/fits/michaels_fly125_cruise__flight.json"
MICHAELS_STANDBY = "results/noise_v2/rounds/round3/fits/michaels_fly125_standby__flight.json"

pytestmark = pytest.mark.skipif(
    not Path(DREGON_FIT).is_file() or not Path(MICHAELS_CRUISE).is_file(),
    reason="the round-3/round-5 v2 fits are not in this checkout",
)


def _cfg(**overrides: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = dict(
        kind="noise_v2",
        n_mics=2,
        n_rotors=4,
        fits=[{"name": "dregon", "cruise": DREGON_FIT}],
        rps={"kind": "full_flight", "flight_fs": 200, "flight_reuse": 64},
        seed=1,
    )
    cfg.update(overrides)
    return cfg


def _pool(duration_s: float = 2.0, **overrides: Any) -> NoiseV2Pool:
    return NoiseV2Pool.from_config(_cfg(**overrides), duration_s=duration_s, sample_rate=SR)


def _band_power(audio: np.ndarray, lo: float, hi: float, n_fft: int = 2048) -> float:
    window = np.hanning(n_fft + 1)[:n_fft]
    spectrum = np.abs(np.fft.rfft(np.asarray(audio, dtype=np.float64)[0, :n_fft] * window)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / SR)
    return float(spectrum[(freqs >= lo) & (freqs <= hi)].sum())


# ── the policy is validated where it is loaded ──────────────────────────────


def test_a_fit_file_that_is_not_there_is_refused_when_the_pool_is_built():
    with pytest.raises(ValueError, match="no such fit file"):
        _pool(fits=[{"name": "ghost", "cruise": "results/noise_v2/nope__flight.json"}])


def test_a_payload_whose_schema_the_renderer_cannot_read_is_refused(tmp_path):
    fit = json.loads(Path(DREGON_FIT).read_text())
    fit["schema"] = "noise-v2-fit/99"
    path = tmp_path / "future.json"
    path.write_text(json.dumps(fit))
    with pytest.raises(ValueError, match="noise-v2-fit/99"):
        _pool(fits=[{"name": "future", "cruise": str(path)}])


def test_a_fit_with_fewer_microphones_than_the_policy_renders_is_refused(tmp_path):
    fit = json.loads(Path(DREGON_FIT).read_text())
    fit["params"]["mic_gains_db"] = list(np.asarray(fit["params"]["mic_gains_db"])[:2])
    path = tmp_path / "two_mic.json"
    path.write_text(json.dumps(fit))
    with pytest.raises(ValueError, match="microphones in mic_gains_db"):
        _pool(n_mics=8, fits=[{"name": "two_mic", "cruise": str(path)}])


def test_a_bank_without_the_format_tag_is_refused(tmp_path):
    path = tmp_path / "bank.json"
    path.write_text(json.dumps({"entries": []}))
    with pytest.raises(ValueError, match=PRESET_BANK_FORMAT):
        _pool(fits=None, preset_bank=str(path))


def test_the_policy_names_exactly_one_source_of_entries(tmp_path):
    with pytest.raises(ValueError, match="exactly one of 'fits' or 'preset_bank'"):
        _pool(preset_bank=str(tmp_path / "bank.json"))


def test_a_trajectory_that_is_not_a_whole_flight_is_refused():
    with pytest.raises(ValueError, match="rps.kind"):
        _pool(rps={"kind": "synthetic_intermittent"})


# ── what a window carries ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fits",
    [
        pytest.param([{"name": "dregon", "cruise": DREGON_FIT}], id="single_regime"),
        pytest.param(
            [{"name": "michaels", "cruise": MICHAELS_CRUISE, "standby": MICHAELS_STANDBY}],
            id="regime_pair",
        ),
    ],
)
def test_a_two_second_window_renders_audio_beside_its_rotor_speeds(fits):
    frame = _pool(duration_s=2.0, fits=fits).sample_timeframe(np.random.default_rng(4), 2.0)
    audio = np.asarray(frame["audio"].data)
    rps = np.asarray(frame["rps"].data)
    assert audio.shape == (2, 2 * SR)
    assert rps.shape == (4, 2 * SR)
    assert np.isfinite(audio).all()
    assert np.isfinite(rps).all()
    assert float(np.abs(audio).max()) > 0.0
    # absolute by default: the fitted level, not a normalised one
    assert 1e-4 < float(np.sqrt(np.mean(np.square(audio.astype(np.float64))))) < 1.0


def test_a_bank_entry_renders_the_same_window_as_the_fit_file_it_inlines(tmp_path):
    bank = tmp_path / "bank.json"
    bank.write_text(
        json.dumps(
            {
                "format": PRESET_BANK_FORMAT,
                "entries": [
                    {
                        "name": "dregon",
                        "cruise": json.loads(Path(DREGON_FIT).read_text()),
                        "standby": None,
                        "traj_rig": "dregon",
                        "provenance": {"source": DREGON_FIT},
                    }
                ],
                "provenance": {"builder": "test"},
            }
        )
    )
    from_file = _pool(duration_s=0.25).render(np.random.default_rng(9), 0.25)[0]
    from_bank = _pool(duration_s=0.25, fits=None, preset_bank=str(bank)).render(
        np.random.default_rng(9), 0.25
    )[0]
    assert np.array_equal(from_file, from_bank)


def test_normalising_the_window_overrides_the_fitted_absolute_level():
    audio, _, _ = _pool(duration_s=0.5, normalize_rms=0.2).render(np.random.default_rng(6), 0.5)
    assert float(np.sqrt(np.mean(np.square(audio.astype(np.float64))))) == pytest.approx(
        0.2, rel=1e-3
    )


# ── the knobs the arms turn ─────────────────────────────────────────────────


def test_a_six_db_comb_offset_raises_the_comb_band_by_six_db():
    """The offset moves the lines and leaves the floor, so a band the comb
    dominates — Michael's cruise comb sits ~19 dB over its floor there —
    follows the offset."""
    fits = [{"name": "michaels", "cruise": MICHAELS_CRUISE}]
    flat, _, _ = _pool(duration_s=1.0, n_mics=1, fits=fits).render(np.random.default_rng(7), 1.0)
    lifted, rps, _ = _pool(duration_s=1.0, n_mics=1, fits=fits, comb_offset_db=6.0).render(
        np.random.default_rng(7), 1.0
    )
    assert rps.min() > 45.0, "this check needs a cruise window, where the comb dominates"
    delta_db = 10.0 * np.log10(
        _band_power(lifted, 300.0, 4000.0) / _band_power(flat, 300.0, 4000.0)
    )
    assert delta_db == pytest.approx(6.0, abs=0.5)


def test_the_loaded_fit_is_never_mutated_by_the_comb_offset():
    before = json.loads(Path(DREGON_FIT).read_text())["params"]["profile"]["profile_db"]
    _pool(duration_s=0.25, comb_offset_db=9.0)
    after = json.loads(Path(DREGON_FIT).read_text())["params"]["profile"]["profile_db"]
    assert before == after


def test_entries_are_drawn_uniformly_over_the_whole_set():
    pool = _pool(
        duration_s=0.05,
        fits=[
            {"name": "dregon", "cruise": DREGON_FIT},
            {"name": "michaels", "cruise": MICHAELS_CRUISE},
            {"name": "michaels_pair", "cruise": MICHAELS_CRUISE, "standby": MICHAELS_STANDBY},
        ],
    )
    rng = np.random.default_rng(2)
    counts = Counter(pool.render(rng, 0.05)[2].name for _ in range(150))
    assert set(counts) == {"dregon", "michaels", "michaels_pair"}
    # 150 draws over 3 entries: mean 50, sd 5.8, so +-4 sd is [27, 73]
    assert all(27 <= n <= 73 for n in counts.values()), counts


def test_render_reuse_renders_once_every_reuse_windows_and_repeats_in_between():
    pool = _pool(duration_s=0.05, render_reuse=8, render_pool=2)
    rng = np.random.default_rng(3)
    drawn = [pool._pooled_render(rng, 0.05)[0] for _ in range(16)]
    distinct = {a.tobytes() for a in drawn}
    assert len(distinct) == 2, "16 draws at reuse 8 must come from 2 rendered clips"


# ── registration ────────────────────────────────────────────────────────────


def test_the_kind_is_reachable_through_the_online_mixing_engine_registry():
    from data_processing.online_mixing import _build_engine

    engine = _build_engine(_cfg(), window_s=0.25, sample_rate=SR)
    assert isinstance(engine, NoiseV2Pool)
    frame = engine.sample_timeframe(np.random.default_rng(5), 0.25)
    assert np.asarray(frame["audio"].data).shape == (2, int(0.25 * SR))


def test_each_bank_entry_is_flown_on_its_own_trajectory_rig(tmp_path):
    """A mixed bank must not cross one rig's comb with another rig's flight:
    the entry is drawn first and its `traj_rig` picks the fitted trajectory
    source, which every window reports in `meta.noise_v2_traj_rig`."""
    bank = tmp_path / "mixed.json"
    bank.write_text(
        json.dumps(
            {
                "format": PRESET_BANK_FORMAT,
                "entries": [
                    {
                        "name": "dregon_room2_floor",
                        "cruise": json.loads(Path(DREGON_FIT).read_text()),
                        "standby": None,
                        "traj_rig": "dregon",
                    },
                    {
                        "name": "michaels_fly125",
                        "cruise": json.loads(Path(MICHAELS_CRUISE).read_text()),
                        "standby": None,
                        "traj_rig": "michaels",
                    },
                ],
            }
        )
    )
    pool = _pool(
        duration_s=0.05,
        n_mics=1,
        fits=None,
        preset_bank=str(bank),
        render_reuse=1,
        rps={
            "kind": "fitted_traj",
            "fits": "dload:rps-traj-fits",
            "rigs": {"michaels": 1, "dregon": 1},
            "flight_fs": 200,
            "flight_reuse": 4,
        },
    )
    expected = {e.name: e.traj_rig for e in pool.entries}
    assert set(pool._traj) == {None, "dregon", "michaels"}
    rng = np.random.default_rng(11)
    seen = Counter()
    for _ in range(64):
        frame = pool.sample_timeframe(rng, 0.05)
        name = get_meta(frame, "noise_v2_entry")
        rig = get_meta(frame, "noise_v2_traj_rig")
        assert rig == expected[name], (name, rig)
        seen[name] += 1
    assert set(seen) == set(expected), "both entries must be exercised"


# ── the window the pool reports ─────────────────────────────────────────────


#: ``sha256`` of ``sample_rps(default_rng(7), 0.5).tobytes()`` for the pool in
#: :func:`test_flight_window_lift_left_the_draw_order_alone`, pinned at
#: f91b01b0^ — the value the PRE-lift code produced, read by running that pool
#: from ``git show f91b01b0^:src/data_processing/...``. The lift moved the
#: ``rps_scale`` multiply inside ``window_flight``; the draws and their order
#: had to stay put, because a stream windowing a different slice of a different
#: flight is a silently different dataset. A deliberate change to the flight
#: generator moves this too: pin it again, from the commit that changed it.
PRE_LIFT_FLIGHT_RPS_SHA256 = "29348b083dea9b2d22558917ce70ab9b88956e98da44425d2d73896b9bc2777e"


def test_flight_window_lift_left_the_draw_order_alone():
    """One seed, one short window, one checksum — taken from the pre-lift code.

    ``sample_rps`` draws the speed scale, then (when the cache is cold) a whole
    flight, then the window's start. Reordering those keeps every range and
    shape assertion green while handing the stream a different trajectory, so
    the guard has to be the bytes.
    """
    pool = _pool(duration_s=0.5, rps_scale_range=[0.45, 1.2])
    rps = pool.sample_rps(np.random.default_rng(7), 0.5)
    assert rps.shape == (4, 8000) and rps.dtype == np.float64
    assert hashlib.sha256(rps.tobytes()).hexdigest() == PRE_LIFT_FLIGHT_RPS_SHA256


def test_last_window_is_the_draw_that_made_the_rps():
    """``last_window`` reports one window's placement without redrawing it.

    ``notebooks/noise_lab.trajectory("v2_stream")`` shows a window of the arm's
    own stream beside the numbers that placed it, so those numbers must
    describe THE returned track: the same slice of the entry's own cached
    flight, the same multiply, and that flight's own hover, UNSCALED.
    """
    pool = _pool(duration_s=0.5, rps_scale_range=[0.45, 1.2])
    rps = pool.sample_rps(np.random.default_rng(44), 0.5)
    window = pool.last_window
    assert window is not None
    assert window.rps is rps

    cache = pool._flights[None]  # the policy's own trajectory, no entry given
    t_win = window.start_s + np.arange(rps.shape[1]) / SR
    slice_at_start = np.stack(
        [np.interp(t_win, cache.t_low, cache.rps[r]) for r in range(cache.rps.shape[0])]
    )
    assert np.array_equal(rps, window.rps_scale * slice_at_start)
    assert window.hover == pytest.approx(float(cache.hover))


def test_last_window_scale_and_start_stay_inside_what_the_policy_allows():
    """Over many draws: the scale spans its log-uniform range two-sidedly, and
    every start leaves a whole window inside the cached flight."""
    pool = _pool(duration_s=0.5, rps_scale_range=[0.45, 1.2])
    rng = np.random.default_rng(45)
    scales, starts = [], []
    for _ in range(64):
        pool.sample_rps(rng, 0.5)
        window = pool.last_window
        cache = pool._flights[None]
        assert window is not None
        assert 0.0 <= window.start_s <= float(cache.t_low[-1]) - 0.5
        scales.append(window.rps_scale)
        starts.append(window.start_s)
    scales = np.array(scales)
    assert scales.min() >= 0.45 and scales.max() <= 1.2
    # Log-uniform: the geometric midpoint splits the draws, not the arithmetic
    # one. Only the two-sidedness is asserted — a constant or a one-sided scale
    # is the failure this catches.
    below = int(np.sum(scales < np.sqrt(0.45 * 1.2)))
    assert 16 <= below <= 48, scales
    assert len(set(starts)) > 32
