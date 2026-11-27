from __future__ import annotations

import numpy as np
import pytest

from tracking.steady_tachometry import corroborate, estimate, merge_estimates

_FS = 12_000


def _audio(
    rates, *, duration=8.0, drift=None, orders=(2, 4, 6, 8, 10, 14, 18, 24, 32, 42), noise=0.018
):
    rng = np.random.default_rng(23143)
    t = np.arange(round(duration * _FS)) / _FS
    out = rng.normal(0, noise, len(t))
    drift = drift or [0] * len(rates)
    for s, d in zip(rates, drift, strict=True):
        phase = 2 * np.pi * (s * t + 0.5 * d * t**2)
        for k in orders:
            if k * (s + abs(d) * duration) < _FS * 0.45:
                out += (
                    0.18
                    / (k / min(orders)) ** 0.75
                    * np.cos(k * phase + rng.uniform(-np.pi, np.pi))
                )
    return out.astype(np.float32)


def _facts(n, **updates):
    return {"blade_count": "2", "n_active": n, "relative_motion": "fixed", **updates}


def _check(result, rates, tol=0.06):
    assert result["windows"], result
    for w in result["windows"]:
        np.testing.assert_allclose(w["rates_rev_s"], rates, atol=tol)
        assert all(len(orders) >= 3 for orders in w["order_support"])
        assert all(0 < s < 0.30 for s in w["uncertainty_rev_s"])


@pytest.mark.parametrize(
    "rates",
    [
        [77.13],
        [57.211, 57.353],
        [47.2, 54.1, 63.3],
        [74.2, 78.7, 91.1, 99.43],
        [38.2, 49.7, 62.4, 79.3, 88.7, 119.1],
    ],
)
def test_distinct_rotors(rates):
    r = estimate(_audio(rates), _FS, _facts(len(rates)), t_start=21.0)
    assert r["status"] == "resolved", r
    assert r["identity"] == "anonymous"
    assert r["windows"][0]["start_s"] == 21.0
    _check(r, rates)


@pytest.mark.parametrize("delta", [0.0, 0.0005])
def test_indistinguishable_twins_are_not_filled(delta):
    r = estimate(_audio([55.14, 55.14 + delta]), _FS, _facts(2))
    assert r["status"] == "partial"
    _check(r, [55.14])


def test_drift_is_local_not_an_average():
    x = _audio([73.2, 94.1], drift=[0.20, -0.17])
    r = estimate(x, _FS, _facts(2), t_start=5)
    assert len(r["windows"]) >= 2, r
    for w in r["windows"]:
        t = w["center_s"] - 5
        np.testing.assert_allclose(w["rates_rev_s"], [73.2 + 0.20 * t, 94.1 - 0.17 * t], atol=0.14)


def test_rejects_noise_silence_and_short_spectrum():
    for x in (_audio([]), np.zeros(8 * _FS)):
        r = estimate(x, _FS, _facts(4))
        assert r["status"] == "unresolved"
        assert all(not w["accepted"] and not w["rates_rev_s"] for w in r["windows"])
    assert estimate(_audio([75.2], duration=0.5), _FS, _facts(1))["status"] == "too_short"


def test_does_not_confuse_apparent_periodicities_with_shaft_rates():
    x = _audio([72.5])
    unknown = estimate(x, _FS, _facts(1, blade_count="unknown"))
    moving = estimate(x, _FS, _facts(1, relative_motion="moving"))
    assert unknown["reason"] == "unknown_shaft_order"
    assert moving["reason"] == "uncorrected_doppler"
    for r, diag in ((unknown, "apparent_periodicities_hz"), (moving, "apparent_rates_rev_s")):
        assert r["status"] == "unresolved"
        assert all(not w["rates_rev_s"] and w["quality"][diag] for w in r["windows"])
        assert r["regime_evidence"]["accepted_s"] == 0


def test_generic_blade_count_and_documented_motor_orders():
    x = _audio([63.9], orders=(3, 6, 9, 12, 15, 21, 30))
    _check(estimate(x, _FS, _facts(1, blade_count="3")), [63.9])
    m = _audio([63.9], orders=(7, 14, 21, 28, 35, 42))
    _check(
        estimate(m, _FS, _facts(1, blade_count="0", propulsion="motor_only", shaft_orders=[7])),
        [63.9],
    )


def test_keeps_separate_motor_on_plateaus_and_timestamps():
    a, b = _audio([62.03], duration=4.0), _audio([79.0], duration=4.0)
    silence = np.zeros(4 * _FS, dtype=np.float32)
    r = estimate(np.concatenate((a, silence, b)), _FS, _facts(1), t_start=30.25)
    assert r["regime_evidence"]["suggested"] == "mixed", r
    good = [w for w in r["windows"] if w["accepted"]]
    assert len(good) == 2, r
    assert good[0]["start_s"] == 30.25
    assert good[0]["end_s"] <= 34.25
    assert good[1]["start_s"] >= 38.25
    assert sum(w["end_s"] - w["start_s"] for w in r["windows"] if not w["accepted"]) == 4.0
    p1 = estimate(a, _FS, _facts(1), t_start=30.25)
    p2 = estimate(b, _FS, _facts(1), t_start=38.25)
    merged = merge_estimates([p2, p1])
    assert merged["windows"] == p1["windows"] + p2["windows"]
    assert merged["regime_evidence"]["accepted_s"] == 8.0
    with pytest.raises(ValueError, match="overlap"):
        merge_estimates([p1, p1])


def test_dominant_shaft_harmonics_not_a_synthetic_bpf_at_one_third_speed():
    rng = np.random.default_rng(373)
    t = np.arange(8 * _FS) / _FS
    p = 2 * np.pi * 133.4 * t
    x = rng.normal(0, 0.03, t.size)
    for k, amp in [(1, 0.45), (2, 0.017), (3, 0.011), (4, 0.008), (5, 0.005), (9, 0.004)]:
        x += amp * np.sin(k * p + rng.uniform(-np.pi, np.pi))
    _check(estimate(x, _FS, _facts(1, blade_count="3")), [133.4])


def test_realistic_varying_voice_pitch_does_not_form_a_steady_rotor():
    rng = np.random.default_rng(555)
    t = np.arange(8 * _FS) / _FS
    f0 = 137 + 24 * np.sin(2 * np.pi * 0.83 * t) + 11 * np.sin(2 * np.pi * 2.88 * t)
    phase = 2 * np.pi * np.cumsum(f0) / _FS
    x = rng.normal(0, 0.007, t.size)
    for k in range(1, 10):
        formant = np.exp(-((k * f0 - (450 + 350 * np.sin(2 * np.pi * 0.6 * t))) ** 2) / 50_000)
        x += 0.13 / np.sqrt(k) * formant * np.cos(k * phase)
    r = estimate(x, _FS, _facts(4))
    assert r["status"] == "unresolved", r
    assert all(not w["accepted"] for w in r["windows"])


def test_substantial_slow_ramp_is_not_constant_and_partial_is_unknown():
    long = estimate(_audio([70.0], duration=24.0, drift=[0.31]), _FS, _facts(1))
    assert long["regime_evidence"]["suggested"] != "constant", long
    partial = estimate(_audio([77.2]), _FS, _facts(4))
    assert partial["regime_evidence"]["suggested"] == "unknown"


def test_antialiasing_does_not_extend_a_bounded_views_recording_clock():
    fs = 50_003
    n = 2 * fs + 1  # not divisible by the integer decimation factor
    t = np.arange(2 * n) / fs
    phase = 2 * np.pi * 83.57 * t
    x = np.sum([0.30 / k * np.sin(k * phase) for k in (2, 4, 6, 8, 12, 22, 32)], axis=0)
    start = 30.25
    a = estimate(x[:n], fs, _facts(1), t_start=start)
    b = estimate(x[n:], fs, _facts(1), t_start=start + n / fs)
    r = merge_estimates([a, b])
    _check(r, [83.57])
    assert r["windows"][0]["end_s"] == r["windows"][1]["start_s"]
    assert r["windows"][-1]["end_s"] <= start + 2 * n / fs


def test_documented_high_motor_orders_expand_band_without_a_speed_prior():
    fs = 48_000
    shaft = 132.31
    t = np.arange(4 * fs) / fs
    p = 2 * np.pi * shaft * t
    x = np.sum([0.13 * np.cos(k * p + k / 100) for k in (14, 28, 56, 84, 98)], axis=0)
    r = estimate(x, fs, _facts(1, blade_count="0", propulsion="motor_only", shaft_orders=[84, 98]))
    _check(r, [shaft])
    assert all({84, 98}.issubset(w["order_support"][0]) for w in r["windows"])


def test_nominal_speed_limits_scale_but_never_supplies_a_rate():
    x = _audio([65.01])
    facts = _facts(
        1, nominal_rps=70, speed_regime="constant", speed_supplied="yes", supplied_kind="nominal"
    )
    r = estimate(x, _FS, facts)
    _check(r, [65.01])
    assert r["nominal_scale_prior_rev_s"] == [70.0]
    assert r["nominal_scale_prior_fraction"] == 0.15
    assert estimate(x, _FS, {**facts, "speed_supplied": "no"})["nominal_scale_prior_rev_s"] == []
    assert not any(w["accepted"] for w in estimate(_audio([]), _FS, facts)["windows"])


def test_native_feedback_veto_never_supplies_or_overwrites_a_rate():
    facts = _facts(2)
    raw = estimate(_audio([63.153, 90.77]), _FS, facts, candidate_only=True)
    assert raw["publication_state"] == "private_candidates"
    ref = {
        "basis": "native_shaft_feedback",
        "entry": "observations/shafts",
        "source": "two independent encoders",
        "coverage": 1.0,
        "rates_rev_s": [63.45, 110.0],
        "ranges_rev_s": [[63.3, 63.6], [109.9, 110.1]],
    }
    validated = corroborate(raw, facts, references=[ref] * len(raw["windows"]))
    assert validated["status"] == "partial"
    assert validated["publication_state"] == "corroborated"
    _check(validated, [63.153], tol=0.01)
    assert all(
        w["quality"]["acceptance_basis"] == "sensor_corroborated" for w in validated["windows"]
    )
    assert all(w["quality"]["apparent_rates_rev_s"] for w in validated["windows"])
    assert validated["acceptance_basis_counts"] == {"sensor_corroborated": len(raw["windows"])}
    nominal = _facts(
        2, nominal_rps=150, speed_supplied="yes", supplied_kind="nominal", speed_regime="constant"
    )
    assert estimate(_audio([63.153, 90.77]), _FS, nominal, candidate_only=True) == raw


def test_release_gate_fails_closed_on_legacy_private_results():
    facts = _facts(1)
    raw = estimate(_audio([63.153]), _FS, facts, candidate_only=True)
    raw.pop("publication_state")
    for w in raw["windows"]:
        w["quality"].pop("_band_checks", None)
    checked = corroborate(raw, facts)
    assert checked["status"] == "unresolved"
    assert all(
        not w["rates_rev_s"] and w["quality"]["apparent_rates_rev_s"] for w in checked["windows"]
    )
