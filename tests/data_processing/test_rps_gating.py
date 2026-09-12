"""The refinement gate: standby is never refined, and the merge is smooth."""

from __future__ import annotations

import numpy as np

from data_processing.rps_gating import GatePolicy, gate_refinement, regime_weight


def _profile(standby_s: float, ramp_s: float, cruise_s: float, dt: float = 0.032):
    """Telemetry that idles at 35, ramps to 80, and cruises, plus a fake refinement."""
    n_s, n_r, n_c = (int(round(x / dt)) for x in (standby_s, ramp_s, cruise_s))
    lo = np.concatenate([np.full(n_s, 35.0), np.linspace(35.0, 80.0, n_r), np.full(n_c, 80.0)])
    tel = np.repeat(lo[None, :], 4, axis=0)
    ft = np.arange(tel.shape[-1]) * dt
    return ft, tel


def test_standby_is_exactly_telemetry():
    ft, tel = _profile(3.0, 1.0, 6.0)
    refined = tel - 0.7  # a uniform correction the refiner might report
    labels, w = gate_refinement(ft, tel, refined)
    standby = np.nanmin(tel, axis=0) < GatePolicy().standby_max_rps
    assert np.array_equal(labels[:, standby], tel[:, standby])
    assert np.all(w[standby] == 0.0)


def test_settled_cruise_is_exactly_refined():
    ft, tel = _profile(3.0, 1.0, 6.0)
    refined = tel - 0.7
    labels, w = gate_refinement(ft, tel, refined)
    full = w >= 1.0
    assert full.any()
    assert np.array_equal(labels[:, full], refined[:, full])


def test_merge_is_continuous_and_monotone():
    ft, tel = _profile(3.0, 2.0, 6.0)
    _, w = gate_refinement(ft, tel, tel - 0.7)
    dw = np.diff(w)
    # no jump: the rate is smooth here, so the weight must be too
    assert dw.max() < 0.1
    # and it never falls back while the rate is rising
    lo = np.nanmin(tel, axis=0)
    rising = np.diff(lo) > 0
    assert np.all(dw[rising] >= -1e-12)


def test_recording_that_starts_in_cruise_is_refined_immediately():
    """The settle window damps a ramp transient; it must not blank a mid-flight start."""
    ft = np.arange(20) * 0.032
    tel = np.full((4, 20), 80.0)
    labels, w = gate_refinement(ft, tel, tel - 1.0)
    assert np.all(w == 1.0)
    assert np.allclose(labels, tel - 1.0)


def test_all_standby_recording_is_untouched():
    ft = np.arange(50) * 0.032
    tel = np.full((4, 50), 35.0)
    labels, w = gate_refinement(ft, tel, tel + 2.0)
    assert np.all(w == 0.0)
    assert np.array_equal(labels, tel)


def test_regime_read_from_the_slowest_rotor():
    """One rotor still idling means the rig is not cruising."""
    ft = np.arange(30) * 0.032
    tel = np.stack([np.full(30, 80.0)] * 3 + [np.full(30, 30.0)])
    assert np.all(regime_weight(ft, tel) == 0.0)
