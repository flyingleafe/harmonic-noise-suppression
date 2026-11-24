"""Helpers of ``scripts/michaels_calib/november_vk_clock.py`` (no raw data needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "michaels_calib"))

import november_vk_clock as V  # noqa: E402

NOM = V.NOMINAL


def _ticks(times_s: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(times_s) * NOM).astype(np.int64)


def test_covered_mask_equals_every_offset_brute_force() -> None:
    native = np.r_[np.arange(0.0, 3.0, 0.06), np.arange(3.5, 6.0, 0.06)]  # one 0.5 s gap
    ticks = _ticks(native)
    t = np.arange(10.0, 17.0, 1e-3)
    o_lo, o_hi = 10.2, 10.6
    mask = V.covered_mask(t, ticks, o_lo, o_hi)
    brute = np.ones_like(mask)
    for o in np.linspace(o_lo, o_hi, 41):
        _, ok = V.trajectory(t, ticks, np.ones((len(ticks), 4)), o, NOM, 1.0)
        brute &= ok
    # interval arithmetic and sampled offsets agree except within a step of an edge
    assert np.mean(mask != brute) < 0.01
    assert not mask[(t > 10.6 + 3.0 - 0.06) & (t < 10.2 + 3.5)].any()


def test_trajectory_never_bridges_a_gap_or_holds_beyond_the_log() -> None:
    native = np.r_[0.0, 0.06, 0.12, 0.6, 0.66]
    rps = np.tile(np.array([10.0, 20.0, 30.0, 40.0, 50.0])[:, None], (1, 4))
    t = np.array([-0.1, 0.03, 0.3, 0.63, 0.8])
    r, ok = V.trajectory(t, _ticks(native), rps, 0.0, NOM, 2.0)
    np.testing.assert_array_equal(ok, [False, True, False, True, False])
    np.testing.assert_allclose(r[0], [0.0, 30.0, 0.0, 90.0, 0.0])


def test_map_line_and_local_offset_round_trip() -> None:
    a, b = 29.09, -1.2e-5
    A, f = V.map_from_line(a, b)
    for centre in (60.0, 500.0):
        o = V.local_offset(A, f, centre)
        native = centre - o
        assert o == pytest.approx(a + b * native, abs=1e-9)


def test_fit_line_recovers_exact_map() -> None:
    A, f = 1.5, 84_000_700.0
    centres = np.array([50.0, 200.0, 350.0, 500.0])
    offset = np.array([V.local_offset(A, f, c) for c in centres])
    line = V.fit_line(centres - offset, offset)
    assert line["A_s"] == pytest.approx(A, abs=1e-6)
    assert line["rate_hz"] == pytest.approx(f, rel=1e-9)
    assert line["jackknife_se_A_ms"] < 1e-3


def test_analyse_curve_flags() -> None:
    xs = np.linspace(-0.05, 0.05, 51)
    good = V.analyse_curve(xs, 0.5 + 40 * (xs - 0.011) ** 2, "fine")
    assert good["usable"] and good["best"] == pytest.approx(0.011, abs=1e-4)
    assert good["sharpness_ms"] > 0
    assert V.analyse_curve(xs, 0.5 + 40 * (xs + 0.08) ** 2, "fine")["edge"]
    assert V.analyse_curve(xs, 0.5 + 1e-4 * xs**2, "fine")["flat"]
    xc = np.linspace(-1, 1, 101)
    two = (
        0.9 - 0.3 * np.exp(-(((xc - 0.5) / 0.03) ** 2)) - 0.25 * np.exp(-(((xc + 0.5) / 0.03) ** 2))
    )
    assert V.analyse_curve(xc, two, "coarse")["multimodal"]


@pytest.mark.parametrize(
    ("kw", "verdict"),
    [
        ({"well_conditioned": False}, "indecisive_data"),
        ({"max_map_diff_ms": 1.0, "scatter_ms": 2.0}, "comb_validated"),
        (
            {"pooled": {"vk": 0.5, "comb": 0.6}, "lag_rms_ms": {"vk": 1.0, "comb": 4.0}},
            "vk_preferred",
        ),
        (
            {"pooled": {"vk": 0.7, "comb": 0.6}, "lag_rms_ms": {"vk": 5.0, "comb": 4.0}},
            "comb_preferred",
        ),
        (
            {"pooled": {"vk": 0.5, "comb": 0.6}, "lag_rms_ms": {"vk": 5.0, "comb": 4.0}},
            "mixed_evidence",
        ),
    ],
)
def test_decision_rule(kw: dict, verdict: str) -> None:
    args = {
        "well_conditioned": True,
        "reasons": [],
        "pooled": {"vk": 0.5, "comb": 0.6},
        "lag_rms_ms": {"vk": 1.0, "comb": 4.0},
        "max_map_diff_ms": 10.0,
        "scatter_ms": 2.0,
    } | kw
    d = V.decide(**args)
    assert d["verdict"] == verdict
    assert d["selected_map"] == ("vk" if verdict == "vk_preferred" else "comb")


def test_vk_residual_is_minimal_at_the_true_offset() -> None:
    """Synthetic blade-pass comb on a ramp: the VK referee recovers a known clock offset."""
    model = V.BLADE
    fs, dur, true_o = model.fs, 4.0, 0.137
    native = np.arange(-1.0, dur + 1.0, 0.06)
    shaft = 40.0 + 6.0 * native[:, None] + np.array([0.0, 3.0, 6.0, 9.0])  # rev/s, 6 rev/s^2
    ticks = _ticks(native)
    t = true_o + np.arange(int(dur * fs)) / fs
    r, _ = V.trajectory(t, ticks, shaft, true_o, NOM, 1.0)
    phase = 2 * np.pi * np.cumsum(r, axis=-1) / fs
    rng = np.random.default_rng(0)
    x = sum(np.cos(2 * k * phase[i] + rng.uniform(0, 6)) / k for i in range(4) for k in range(1, 8))
    x = np.stack([x, x]) + 0.3 * rng.standard_normal((2, len(t)))
    mask = (t > t[0] + V.EDGE_S) & (t < t[-1] - V.EDGE_S)
    ctx = V.Context(
        x, t, mask, float(np.sum(x[:, mask] ** 2)), ticks, shaft, V.vk_config(model, 1.0), model, {}
    )
    nums = {d: V.residual_energy(ctx, true_o + d, NOM, 1.0)[0] for d in (-0.03, 0.0, 0.03)}
    assert min(nums, key=nums.__getitem__) == 0.0
