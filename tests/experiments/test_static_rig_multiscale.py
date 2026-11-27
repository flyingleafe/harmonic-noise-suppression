from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.static_rig.multiscale import (
    _fill,
    _Lines,
    _owned_cells,
    _paths,
    _Point,
    _weak,
    track,
)

_FS = 8000


def _tones(rates, *, slopes=None, seconds=8.0, blades=2, seed=12):
    rng = np.random.default_rng(seed)
    t = np.arange(round(seconds * _FS)) / _FS
    audio = rng.normal(0, 0.010, (2, t.size))
    if slopes is None:
        slopes = [0.0] * len(rates)
    for rate, slope in zip(rates, slopes, strict=True):
        angle = 2 * np.pi * (rate * t + 0.5 * slope * t**2)
        for k in blades * np.arange(1, 10):
            if k * (rate + abs(slope) * seconds) >= 0.44 * _FS:
                continue
            gain = rng.uniform(0.07, 0.14, (2, 1)) * (k / blades) ** -0.8
            phi = rng.uniform(-np.pi, np.pi, (2, 1))
            audio += gain * np.cos(k * angle + phi)
    return audio.astype(np.float32)


@pytest.mark.parametrize(
    "rates", [(71.36,), (57.21, 83.14), (52.42, 74.19, 99.61), (53.41, 59.83, 72.95, 96.27)]
)
def test_full_span_points_are_anonymous_and_have_unique_support(rates):
    r = track(_tones(rates), _FS, expected=len(rates), blades=2)
    json.dumps(r, allow_nan=False)
    np.testing.assert_allclose(r["rates_rps"], np.broadcast_to(rates, (8, len(rates))), atol=0.085)
    assert all(all(n is not None and n > 0 for n in row) for row in r["uncertainty_rps"])
    assert all(len(set(row)) == len(rates) for row in r["track_id"])
    assert all(all(v == "measured" for v in row) for row in r["provenance"])
    assert r["diagnostics"]["active_spans"] == [[0.0, 8.0]]
    assert r["diagnostics"]["steady_spans"]


def test_full_fft_rescues_two_close_steady_rates_without_local_scale_support():
    rng = np.random.default_rng(4)
    t = np.arange(24 * _FS) / _FS
    x = rng.normal(0, 0.018, (2, len(t)))
    truth = (73.13, 73.31)
    for rate, amplitudes in zip(
        truth, ([0.22, 0.13, 0.09, 0.09, 0.06], [0.06, 0.045, 0.012, 0.028, 0.027]), strict=True
    ):
        for order, amp in zip((2, 4, 6, 8, 12), amplitudes, strict=True):
            phase = rng.uniform(-np.pi, np.pi, (2, 1))
            x += amp * np.cos(2 * np.pi * order * rate * t + phase)
    r = track(x.astype(np.float32), _FS, expected=2, blades=2)
    (steady,) = r["diagnostics"]["steady_spans"]
    np.testing.assert_allclose(steady["rates_rps"], truth, atol=0.015)
    assert (steady["start_s"], steady["stop_s"]) == (0.0, 24.0)
    rescued = [
        row[1]
        for row, prov, support in zip(
            r["rates_rps"], r["provenance"], r["order_support"], strict=True
        )
        if prov[1] == "track_supported" and len(support[1]) >= 2
    ]
    assert 0 < len(rescued) == steady["local_rescues"] < len(t) / _FS
    np.testing.assert_allclose(rescued, truth[1], atol=0.015)


def test_clocks_drift_silence_and_identity_reset():
    x = np.concatenate(
        [
            _tones([72.5], slopes=[0.25]),
            np.zeros((2, 4 * _FS), dtype=np.float32),
            _tones([82.5]),
        ],
        axis=1,
    )
    r = track(x, _FS, expected=1, blades=2)
    truth = [72.5 + 0.25 * t for t in r["times_s"][:8]]
    np.testing.assert_allclose(np.ravel(r["rates_rps"][1:7]), truth[1:7], atol=0.18)
    assert r["provenance"][8:12] == [["inactive"]] * 4
    assert r["rates_rps"][8:12] == [[None]] * 4
    assert r["diagnostics"]["active_spans"] == [[0.0, 8.0], [12.0, 20.0]]
    np.testing.assert_allclose(np.ravel(r["rates_rps"][12:]), 82.5, atol=0.08)
    assert r["track_id"][7] != r["track_id"][12]


def test_coincident_sources_do_not_create_four_rotors_or_an_octave():
    x = _tones([63.4, 63.4], seconds=7, blades=3)
    r = track(x, _FS, expected=4, blades=3)
    for row in r["rates_rps"]:
        np.testing.assert_allclose([v for v in row if v is not None], [63.4], atol=0.08)
    assert all(v.count("unresolved") == 3 for v in r["provenance"])


@pytest.mark.parametrize("blade_train", [False, True])
def test_primitive_shaft_carriers_do_not_anchor_a_half_rate(blade_train):
    rng = np.random.default_rng(314)
    truth = 62.8
    t = np.arange(8 * _FS) / _FS
    x = rng.normal(0, 0.02, (2, len(t)))
    orders = range(1, 13) if blade_train else range(1, 14, 2)
    for k in orders:
        amp = (0.35 * (1 if k % 2 == 0 else 0.40) if blade_train else 0.18) / k**0.65
        x += amp * np.cos(2 * np.pi * k * truth * t + rng.uniform(-np.pi, np.pi, (2, 1)))
    r = track(x.astype(np.float32), _FS, expected=4, blades=2)
    for row in r["rates_rps"]:
        np.testing.assert_allclose([v for v in row if v is not None], [truth], atol=0.08)


def test_anchors_cannot_supply_truth_across_missing_audio_or_unknown_scale():
    anchors = [{"time_s": 1.5, "rates_rps": [71.0, 142.0]}, {"time_s": 4.5, "rates_rps": [71.0]}]
    x = np.zeros((2, 4 * _FS), dtype=np.float32)
    r = track(x, _FS, expected=4, blades=2, anchors=anchors)
    assert r["rates_rps"] == [[None] * 4] * 4
    assert r["provenance"] == [["inactive"] * 4] * 4
    assert r["diagnostics"]["anchor_supports"][0]["matches"] == []
    r = track(_tones([71.0]), _FS, expected=1, blades=None, anchors=anchors)
    assert all(row == [None] for row in r["rates_rps"])


def test_fractional_native_end_and_single_short_frame():
    x = _tones([62.1])[:, :27_593]
    r = track(x, 22_049, expected=1, blades=2)
    assert r["times_s"][-1] < len(x[0]) / 22_049
    assert r["diagnostics"]["duration_s"] == len(x[0]) / 22_049
    r = track(x[:, :57], 22_049, expected=1, blades=2)
    assert r["times_s"] == [57 / 22_049 / 2]
    assert r["rates_rps"] == [[None]]


@pytest.mark.parametrize(("gap", "fillable"), [(2, True), (4, True), (7, False)])
def test_explicit_pcm_gaps_are_not_silence_and_only_bounded_gaps_fill(gap, fillable):
    x = np.repeat(_tones([65.21], slopes=[0.12], seconds=20), 3, axis=1)
    sr = 3 * _FS
    x[:, 8 * sr : (8 + gap) * sr] = np.nan
    r = track(x, sr, expected=1, blades=2)
    assert r["diagnostics"]["pcm_unavailable_spans"] == [[8.0, float(8 + gap)]]
    assert r["provenance"][8 : 8 + gap] == [["interpolated" if fillable else "unresolved"]] * gap
    assert r["order_support"][8 : 8 + gap] == [[[]]] * gap
    if fillable:
        truth = 65.21 + 0.12 * np.asarray(r["times_s"][8 : 8 + gap])
        np.testing.assert_allclose(np.ravel(r["rates_rps"][8 : 8 + gap]), truth, atol=0.08)
        assert all(row[0] >= 0.42 for row in r["uncertainty_rps"][8 : 8 + gap])
        assert r["track_id"][7] == r["track_id"][8 + gap]
    else:
        assert r["rates_rps"][8 : 8 + gap] == [[None]] * gap
    x[:, 8 * sr : (8 + gap) * sr] = 0
    r = track(x, sr, expected=1, blades=2)
    assert r["provenance"][8 : 8 + gap] == [["inactive"]] * gap
    assert r["rates_rps"][8 : 8 + gap] == [[None]] * gap


def test_unavailable_endpoints_never_extrapolate():
    x = _tones([71.0], seconds=12)
    x[:, : 2 * _FS] = np.nan
    x[:, -2 * _FS :] = np.nan
    r = track(x, _FS, expected=1, blades=2)
    assert r["provenance"][:2] == r["provenance"][-2:] == [["unresolved"]] * 2
    assert r["rates_rps"][:2] == r["rates_rps"][-2:] == [[None]] * 2


def _point(rate):
    return _Point(rate, 0.025, 12.0, (2, 4, 6), (0, 1, 2))


def test_global_links_bridge_only_an_unambiguous_bounded_gap():
    a, b, c, d = _point(50), _point(70.0), _point(50.6), _point(70.2)
    frames = [[a, b], [_point(70.07)], [_point(70.13)], [c, d]]
    times = np.arange(0.5, 4.5)
    chains = _paths(times, frames)
    chain = [(0, a), (3, c)]
    assert chain in chains
    empty = _Lines(1.7, 3500.0, np.empty(0), np.empty(0), np.empty(0))
    bridge = list(_fill(times, [empty] * 4, frames, chain, np.ones(4)))
    assert [(k, v.provenance) for k, v in bridge] == [(1, "interpolated"), (2, "interpolated")]
    np.testing.assert_allclose([p.rate for _, p in bridge], [50.2, 50.4])
    assert all(p.error > a.error for _, p in bridge)
    assert not list(_fill(times, [empty] * 4, frames, chain, [1, 0.1, 0.1, 1]))
    assert not list(_fill(times, [empty] * 4, [[a], [], [], [c]], chain, np.ones(4)))
    ambiguous = [[a, _point(50.5)], [_point(50.25)]]
    assert all(len(path) == 1 for path in _paths(times[:2], ambiguous))
    opposite_scale = [[a], [_point(100.4)], [_point(100.8)], [c]]
    assert not list(_fill(times, [empty] * 4, opposite_scale, chain, np.ones(4)))


def test_track_support_requires_actual_other_unassigned_local_orders():
    p = _Lines(1.7, 3500.0, np.array([100.0, 250.0]), np.array([18.0, 16.0]), np.ones(2))
    found = _weak(p, 50.0, [], 0.12)
    assert found is not None
    assert found.rate == pytest.approx(50.0)
    assert (found.provenance, found.orders) == ("track_supported", (2, 5))
    occupied = _Point(100.0, 0.02, 12.0, (1,), (0,))
    assert _weak(p, 50.0, [occupied], 0.12) is None


def test_one_comb_owns_symmetric_modulation_and_teeth_beyond_search_orders():
    carriers = np.array([130.0, 260.0, 390.0, 50 * 65.0])
    modulation = np.r_[carriers[:3] - 1.8, carriers[:3] + 1.8]
    twin = 260.0 + 3.1
    hz = np.sort(np.r_[carriers, modulation, twin])
    core = np.isin(hz, carriers[:3])
    levels = np.where(np.isin(hz, carriers), 30.0, 16.0)
    p = _Lines(10.0, 3500.0, hz, levels, np.full(len(hz), 0.14))
    rotor = _Point(65.0, 0.008, 20.0, (2, 4, 6), tuple(np.flatnonzero(core).tolist()))
    np.testing.assert_array_equal(_owned_cells(p, rotor), hz != twin)


def test_integer_and_rational_subsets_of_one_am_carrier_do_not_split():
    rng = np.random.default_rng(132)
    fs = 12_000
    t = np.arange(12 * fs) / fs
    speed = 62.7
    x = rng.normal(0, 0.015, (2, len(t)))
    # Orders divisible by 2 or 3 make nested apparent combs, but those
    # components explain no spectra beyond the primitive continuous carrier.
    for k in (*range(1, 15), 18, 20, 23, 25, 29, 34, 50, 52, 54):
        gain = (0.5 + 0.9 * (k % 2 == 0) + 0.4 * (k % 3 == 0)) / k**0.65
        am = 1.0 + 0.45 * np.cos(2 * np.pi * 2.2 * t + k / 7)
        phase = rng.uniform(-np.pi, np.pi, (2, 1))
        x += gain * am * np.cos(2 * np.pi * k * speed * t + phase)
    r = track(x.astype(np.float32), fs, expected=4, blades=2)
    for row in r["rates_rps"]:
        np.testing.assert_allclose([s for s in row if s is not None], [speed], atol=0.08)
