"""Contracts of the stage-2 multi-rotor separability instrument.

Five tests, each pinning something the design study's conclusions rest on:

1. the generator's planted line power is what a periodogram reads back when
   the rotors are far apart (the absolute-level contract of :func:`estimate_e1`
   -- a Parseval conversion with no window constant in it);
2. the phase-aware solve is EXACT on a noiseless two-rotor beat with
   ``delta T_w = 2`` (the identifiability contract of :func:`estimate_e2`: at
   two observed beat cycles the design matrix is conditioned and the answer is
   the planted amplitude, not an approximation of it);
3. the same at ``delta T_w`` well under one, where the solve MUST lose the
   split -- the criterion's other side;
4. the criterion function's threshold behaviour in ``delta`` and its
   near-independence of ``k``;
5. the within-record shaft correction, which is what makes the criterion
   usable on the fitted ``lam`` values.

CPU-small: every scene here is two or three rotors over a couple of seconds.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.noise_model import multirotor as MR

SR = 16000


def _spec(name: str, carrier: float, *, n_orders: int = 6, db: float = -30.0, **kw) -> MR.RotorSpec:
    profile = np.full(n_orders, db)
    profile[1::2] -= 6.0  # a little order structure, so a wrong split shows
    return MR.RotorSpec(
        name=name,
        carrier_rev_s=carrier,
        profile_db=profile,
        sigma_nu=kw.pop("sigma_nu", 1e-9),
        lam=kw.pop("lam", 1.0),
        gamma_hz=np.asarray(kw.pop("gamma_hz", np.zeros(n_orders)), dtype=np.float64),
    )


def test_e1_reads_back_the_planted_line_power_when_rotors_are_far_apart():
    """Two well-separated rotors: the periodogram read is the planted power.

    This is the generator/estimator level contract. Carriers are chosen
    INCOMMENSURATE (60 and 83.37 rev/s) because a rational ratio puts two
    harmonics in the same bin, which is a real collision and not what this
    test is about.
    """
    cfg = MR.SceneConfig(
        rotors=(_spec("a", 60.0), _spec("b", 83.37)),
        n_mics=2,
        duration_s=4.0,
        track_drift_std_hz=0.0,
        comb_to_floor_db=30.0,
        k_cap=6,
        seed=1,
    )
    scene = MR.simulate_scene(cfg)
    orders = np.arange(1, 7)
    err = MR.profile_error_db(MR.estimate_e1(scene, orders)["power"], scene.power_true)
    assert np.abs(err).max() < 0.5, err


def test_e2_is_exact_on_a_noiseless_beat_at_two_cycles_per_window():
    """``delta T_w = 2``: the phase-aware solve returns the planted amplitudes.

    ``delta = 0.5`` Hz over a 4 s record gives two beat cycles in the
    whole-record window at order 1, which is the window
    :func:`multirotor.estimate_e2` picks. With no floor and no decoherence the
    least squares is an exact linear inverse, so the tolerance is numerical,
    not statistical -- and both the single-mic and the steering-constrained
    joint solve must meet it.
    """
    cfg = MR.SceneConfig(
        rotors=(_spec("a", 67.75), _spec("b", 68.25)),
        n_mics=4,
        duration_s=4.0,
        track_drift_std_hz=0.0,
        comb_to_floor_db=250.0,
        k_cap=6,
        seed=3,
    )
    scene = MR.simulate_scene(cfg)
    orders = np.arange(1, 7)
    single = MR.estimate_e2(scene, orders)
    assert single["window_s"][0, 0] * 0.5 == pytest.approx(2.0, abs=0.05)
    assert np.abs(MR.profile_error_db(single["power"], scene.power_true)).max() < 1e-3
    joint = MR.estimate_e2(scene, orders, multi=True)
    assert np.abs(MR.profile_error_db(joint["power"], scene.power_true)).max() < 1e-2


def test_e2_loses_the_split_when_the_beat_is_not_observed():
    """Under one observed beat cycle the split is not identifiable.

    Two rotors 3 mHz apart: at order 1 a 4 s record holds 0.012 of a beat
    cycle, the two columns of the design matrix are collinear and the normal
    matrix' condition number is in the thousands. The planted profiles differ
    by 12 dB and the quiet rotor's line is still 10 dB over the floor of its
    own band, so what breaks here is the SEPARATION and not the SNR: the quiet
    rotor comes back several dB high because the ill-conditioned inverse hands
    it part of its neighbour. If this passed, the criterion this module
    reports would be meaningless.
    """
    cfg = MR.SceneConfig(
        rotors=(_spec("quiet", 68.0, db=-42.0), _spec("loud", 68.003, db=-30.0)),
        n_mics=4,
        duration_s=4.0,
        track_drift_std_hz=0.0,
        comb_to_floor_db=0.0,
        k_cap=2,
        seed=3,
    )
    scene = MR.simulate_scene(cfg)
    orders = np.array([1, 2])
    res = MR.estimate_e2(scene, orders)
    assert res["beat_cycles"].max() < 0.1
    assert res["cond"].min() > 1e3
    assert MR.line_snr_db(scene, orders, res["bandwidth_hz"])[0, 0] > 10.0
    err = MR.profile_error_db(res["power"], scene.power_true[:, :2])
    assert err[0, 0] > 3.0, err


def test_resolvability_score_is_linear_in_delta_and_flat_in_order():
    """The criterion crosses 1 in ``delta`` and barely moves in ``k``.

    ``tau_c(k)`` of the fitted shaft is ballistic (``lam tau_c << 1``), so it
    falls as ``1 / k`` and ``k delta tau_c`` is order-INDEPENDENT. That is the
    structural prediction the sweep tests, so it is pinned here: an order of
    magnitude in ``k`` may move the score by less than 30 %, while an order of
    magnitude in ``delta`` must move it by ten.
    """
    a = _spec("a", 68.0, sigma_nu=0.185, lam=0.157)
    b = _spec("b", 68.83, sigma_nu=0.180, lam=0.002)
    scores = {k: MR.resolvability_score(a, b, k, duration_s=24.0) for k in (1, 4, 16, 64)}
    lo, hi = min(scores.values()), max(scores.values())
    assert hi / lo < 1.3, scores
    tenth = MR.resolvability_score(a, b, 8, duration_s=24.0, delta_hz=0.083)
    full = MR.resolvability_score(a, b, 8, duration_s=24.0, delta_hz=0.83)
    assert full / tenth == pytest.approx(10.0, rel=0.05)
    assert full > 1.0 > tenth


def test_within_record_shaft_scale_shrinks_only_the_slow_rotors():
    """``sigma_eff = sigma_nu sqrt(min(1, 2 lam T / 3))``, and it matters.

    ``Motor4_70``'s fitted ``lam = 0.0027`` s^-1 means its shaft error is a
    frozen frequency offset over a 24 s support, not a linewidth; a rotor at
    ``lam T > 1.5`` is ergodic inside the record and must be left alone. Both
    branches are checked, and so is the consequence: the 0.83 Hz spacing is
    under the criterion on the ensemble width and over it on the within-record
    width.
    """
    slow = _spec("slow", 69.5, sigma_nu=1.936, lam=0.0027)
    fast = _spec("fast", 68.67, sigma_nu=0.166, lam=5.5)
    factor = math.sqrt(2.0 * 0.0027 * 24.0 / 3.0)
    assert factor == pytest.approx(0.20785, rel=1e-4)
    assert MR.within_record_spec(slow, 24.0).sigma_nu == pytest.approx(1.936 * factor, rel=1e-9)
    assert MR.within_record_spec(fast, 24.0).sigma_nu == pytest.approx(0.166)
    ensemble = MR.resolvability_score(slow, fast, 8, duration_s=24.0, within_record=False)
    within = MR.resolvability_score(slow, fast, 8, duration_s=24.0)
    assert ensemble < 1.0 < within, (ensemble, within)
