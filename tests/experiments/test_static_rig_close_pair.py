"""Planted phase checks; no measurements from the pilot calibration/test panel."""

from __future__ import annotations

import json

import numpy as np

from experiments.static_rig.close_pair import _bounded_envelope, fit_pair, predict_pair

FS = 2400
N = 4 * FS
ORDERS = np.arange(1, 8)


def _scene(rates, drifts, *, weak=0.75, extra_s=0):
    rng = np.random.default_rng(18463)
    t = (np.arange(N + extra_s * FS) - 0.5 * (N - 1)) / FS
    x = np.zeros((2, t.size))
    for i, (r, d) in enumerate(zip(rates, drifts)):
        phase = 2.0 * np.pi * ORDERS[:, None] * (r * t + 0.5 * d * t**2)
        gain = weak**i * np.exp(1j * rng.uniform(-np.pi, np.pi, (2, len(ORDERS))))
        gain /= ORDERS**0.23
        x += (gain @ np.exp(1j * phase)).real
    return x + 0.035 * rng.standard_normal(x.shape)


def test_one_seed_recovers_drifting_pair_without_using_final_samples() -> None:
    truth = [79.58, 80.53]
    drift = [-0.105, 0.115]
    raw = _scene(truth, drift, extra_s=2)
    fit = fit_pair(raw[:, :N], FS, 80.21)
    json.dumps(fit, allow_nan=False)
    assert fit["eligible"], fit["reason"]
    np.testing.assert_allclose(fit["rates"], truth, atol=0.065, rtol=0.0)
    np.testing.assert_allclose(fit["drift_rps_per_s"], drift, atol=0.085, rtol=0.0)
    assert fit["score"] > 0.5
    lo, hi = fit["separation_interval_rps"]
    assert 0.0 < lo <= fit["separation_rps"] <= hi < 1.15
    assert fit["separation_profile"][0][0] == 0.0
    assert {"unseen_both", "unseen_orders", "unseen_channels"} <= fit["diagnostics"][
        "heldout"
    ].keys()

    altered = raw[:, :N].copy()
    cut = fit["diagnostics"]["block_samples"][2]
    altered[:, cut:] *= -1
    changed = fit_pair(altered, FS, 80.21)
    for key in ("rates", "drift_rps_per_s", "null_rate_rps", "separation_profile"):
        assert changed[key] == fit[key]
    a = changed["diagnostics"]["heldout"]["unseen_both"]["pair_sse"]
    b = fit["diagnostics"]["heldout"]["unseen_both"]["pair_sse"]
    assert a > 10.0 * b

    checked = predict_pair(raw[:, N:], FS, fit, center_offset_s=3.0)
    assert checked["reason"] == "ok"
    assert checked["score"] > 0.4
    assert (
        predict_pair(raw[:, N:], FS, fit, center_offset_s=2.0)["reason"]
        == "prediction_block_overlaps_fit"
    )


def test_order_scaled_single_sideband_competes_with_a_split() -> None:
    # The second comb here is exactly one complex envelope sideband of the
    # first, with a different sideband phase allowed at every mic and order.
    x = _scene([80.0, 80.39], [0.0, 0.0], weak=0.32)
    fit = fit_pair(x, FS, [79.97, 80.41])
    assert fit["null_am_rps"] is not None
    assert abs(fit["null_am_rps"] - 0.39) < 0.06
    assert fit["score"] < 0.15
    rejected = fit_pair(x, FS, 80.0, other_rates=(80.0,))
    assert not rejected["eligible"] and rejected["rates"] == []
    assert rejected["reason"] == "no_clean_train_or_heldout_orders"


def test_bounded_regression_preserves_feasible_subcycle_sideband() -> None:
    t = np.linspace(-0.52, 0.52, 50)
    phase = 2.0 * np.pi * 0.23 * t
    features = np.stack((np.cos(phase), np.sin(phase)), axis=-1)[None, :, :]
    y = np.column_stack((1.0 + 0.32 * np.exp(1j * phase), 0.1 + np.exp(1j * phase)))[None]
    gain = _bounded_envelope(features, y, 0.82)[0]
    np.testing.assert_allclose(gain[:, 0], [1.0, 0.32, 0.32j], atol=1e-11)
    assert np.linalg.norm(gain[1:, 1]) <= 0.82 * abs(gain[0, 1]) + 1e-10
    pred = gain[0, 1] + features[0] @ gain[1:, 1]
    assert np.linalg.norm(pred - y[0, :, 1]) < np.linalg.norm(y[0, :, 1] - y[0, :, 1].mean())
