"""The frozen rotor-speed statistics must recover what they claim to measure."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.rps_traj.data import RATE_HZ, Flight, airborne_segments, to_common_grid
from experiments.rps_traj.stats import (
    LAGS_S,
    TrajStats,
    compute_stats,
    discrepancy,
    passes,
    stats_from_samples,
)


def _ou(n: int, tau_s: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """One AR(1) path with continuous-time ACF exp(-lag / tau) and std sigma."""
    a = float(np.exp(-1.0 / (tau_s * RATE_HZ)))
    noise = rng.normal(0.0, sigma * np.sqrt(1.0 - a * a), size=n)
    x = np.empty(n)
    x[0] = rng.normal(0.0, sigma)
    for i in range(1, n):
        x[i] = a * x[i - 1] + noise[i]
    return x


def _ou_quad(
    n: int, rng: np.random.Generator, *, tau_s: float, sigma: float, rho: float, level: float
) -> np.ndarray:
    """(4, n) OU-like rotor speeds: common factor + independent per-rotor part,
    giving pairwise correlation ``rho`` and per-rotor std ``sigma``."""
    common = _ou(n, tau_s, sigma, rng)
    out = np.empty((4, n))
    for r in range(4):
        out[r] = level + np.sqrt(rho) * common + np.sqrt(1.0 - rho) * _ou(n, tau_s, sigma, rng)
    return out


def test_compute_stats_recovers_ou_acf_and_xcorr() -> None:
    tau_s, sigma, rho, level = 1.5, 4.0, 0.6, 90.0
    rng = np.random.default_rng(7)
    flights = [
        Flight(
            rig="synthetic",
            flight=f"f{i}",
            fs=RATE_HZ,
            t0=0.0,
            rps=_ou_quad(int(400 * RATE_HZ), rng, tau_s=tau_s, sigma=sigma, rho=rho, level=level),
            source="test",
        )
        for i in range(3)
    ]
    stats = compute_stats(flights)

    assert stats.overall_mean == pytest.approx(level, abs=0.5)
    assert np.allclose(stats.rotor_mean, level, atol=0.6)
    assert np.allclose(stats.rotor_var, sigma**2, rtol=0.15)

    # Only lags well inside the 400 s segments: the frozen ACF is biased-
    # normalised, so long lags carry a known (1 - k/N) taper on top of the
    # process ACF and are not meant to match exp(-lag/tau) tightly.
    short = LAGS_S <= 5.0
    expected = np.exp(-LAGS_S[short] / tau_s)
    got = stats.acf[:, short]
    assert np.allclose(got, expected[None, :], atol=0.08)

    off = ~np.eye(4, dtype=bool)
    assert np.allclose(stats.xcorr[off], rho, atol=0.05)
    assert np.allclose(np.diag(stats.xcorr), 1.0, atol=1e-12)


def test_nan_gaps_do_not_bias_the_statistics() -> None:
    """A telemetry dropout must remove samples, not shift means or the ACF."""
    rng = np.random.default_rng(11)
    clean = _ou_quad(int(300 * RATE_HZ), rng, tau_s=1.0, sigma=3.0, rho=0.5, level=120.0)
    holed = clean.copy()
    holed[1, 5000:5300] = np.nan  # a 3 s dropout on one rotor
    make = lambda rps: [  # noqa: E731
        Flight(rig="s", flight="f", fs=RATE_HZ, t0=0.0, rps=rps, source="test")
    ]
    a = compute_stats(make(clean))
    b = compute_stats(make(holed))
    assert b.rotor_mean[1] == pytest.approx(a.rotor_mean[1], abs=0.5)
    assert b.rotor_var[1] == pytest.approx(a.rotor_var[1], rel=0.1)
    assert np.allclose(b.acf[1, LAGS_S <= 2.0], a.acf[1, LAGS_S <= 2.0], atol=0.05)
    assert np.isfinite(b.xcorr).all()


def test_airborne_segments_keeps_cruise_eroded_by_one_second() -> None:
    fs = RATE_HZ
    ground = np.full(int(10 * fs), 5.0)
    ramp = np.linspace(5.0, 100.0, int(2 * fs))
    cruise = np.full(int(40 * fs), 100.0)
    land = np.linspace(100.0, 5.0, int(2 * fs))
    after = np.full(int(6 * fs), 5.0)
    track = np.concatenate([ground, ramp, cruise, land, after])
    rps = np.tile(track, (4, 1))

    segments = airborne_segments(rps, fs)
    assert len(segments) == 1
    seg = segments[0]
    # thr = 0.5 * p90 = 50 rev/s: the ramp crosses it halfway, the landing
    # ramp likewise, and the rule then erodes 1 s off both ends.
    ramp_cross = int(10 * fs) + int(np.searchsorted(ramp, 50.0, side="right"))
    fall_cross = int(12 * fs) + int(40 * fs) + int(np.sum(land > 50.0))
    assert seg.start == pytest.approx(ramp_cross + int(fs), abs=1)
    assert seg.stop == pytest.approx(fall_cross - int(fs), abs=1)
    # Above-threshold run (~42.1 s: cruise + the above-50 halves of both
    # ramps) minus 1 s of erosion at each end.
    assert (seg.stop - seg.start) / fs == pytest.approx(
        (fall_cross - ramp_cross) / fs - 2.0, abs=0.02
    )
    assert np.all(rps[:, seg] > 50.0)


def test_airborne_segments_drops_short_hops_and_grounded_rotors() -> None:
    fs = RATE_HZ
    # A 4 s hop survives no erosion; and a rotor that never spins up kills the
    # whole run (the rule needs EVERY rotor above threshold).
    hop = np.concatenate([np.full(int(3 * fs), 5.0), np.full(int(4 * fs), 100.0)])
    assert airborne_segments(np.tile(hop, (4, 1)), fs) == []

    long_flight = np.concatenate([np.full(int(3 * fs), 5.0), np.full(int(40 * fs), 100.0)])
    rps = np.tile(long_flight, (4, 1))
    assert len(airborne_segments(rps, fs)) == 1
    rps[2] = 5.0
    assert airborne_segments(rps, fs) == []


def test_passes_rule() -> None:
    base = {
        "overall_mean": 1.0,
        "rotor_mean": 1.0,
        "rotor_var": 1.0,
        "acf": 1.0,
        "xcorr": 1.0,
    }
    all_better = {k: 0.5 for k in base}
    verdict, ok = passes(all_better, base)
    assert verdict and all(ok.values())

    # Three strict improvements, two exact ties: accepted.
    three_strict = dict(base, overall_mean=0.9, rotor_mean=0.9, rotor_var=0.9)
    verdict, ok = passes(three_strict, base)
    assert verdict and all(ok.values())

    # Only two strict improvements (three families tie): rejected.
    two_strict = dict(base, overall_mean=0.9, rotor_mean=0.9)
    verdict, ok = passes(two_strict, base)
    assert not verdict and all(ok.values())

    # One family regresses, however much the others improve: rejected.
    worse_one = {k: 0.1 for k in base} | {"xcorr": 1.0 + 1e-6}
    verdict, ok = passes(worse_one, base)
    assert not verdict
    assert ok == {
        "overall_mean": True,
        "rotor_mean": True,
        "rotor_var": True,
        "acf": True,
        "xcorr": False,
    }


def test_discrepancy_is_zero_against_itself_and_grows_with_error() -> None:
    rng = np.random.default_rng(3)
    real = compute_stats(
        [
            Flight(
                rig="s",
                flight="f",
                fs=RATE_HZ,
                t0=0.0,
                rps=_ou_quad(int(120 * RATE_HZ), rng, tau_s=1.0, sigma=3.0, rho=0.4, level=90.0),
                source="test",
            )
        ]
    )
    zero = discrepancy(real, real)
    assert set(zero) == {"overall_mean", "rotor_mean", "rotor_var", "acf", "xcorr"}
    assert all(v == pytest.approx(0.0, abs=1e-12) for v in zero.values())

    shifted = TrajStats(
        overall_mean=real.overall_mean + 2.0,
        rotor_mean=np.asarray(real.rotor_mean) + 2.0,
        rotor_var=np.asarray(real.rotor_var) * np.e,
        acf=np.asarray(real.acf),
        xcorr=np.asarray(real.xcorr),
    )
    d = discrepancy(shifted, real)
    assert d["overall_mean"] == pytest.approx(2.0)
    assert d["rotor_mean"] == pytest.approx(2.0)
    assert d["rotor_var"] == pytest.approx(1.0, abs=1e-9)
    assert d["acf"] == pytest.approx(0.0, abs=1e-12)


def test_stats_from_samples_matches_the_same_process_and_is_deterministic() -> None:
    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        return _ou_quad(n, rng, tau_s=1.0, sigma=3.0, rho=0.5, level=120.0)

    durations = [40.0, 60.0]
    a = stats_from_samples(sampler, durations, n_rep=3, seed=0)
    b = stats_from_samples(sampler, durations, n_rep=3, seed=0)
    assert a.to_json() == b.to_json()
    assert TrajStats.from_json(a.to_json()).to_json() == a.to_json()

    real = compute_stats(
        [
            Flight(
                rig="s",
                flight=f"f{i}",
                fs=RATE_HZ,
                t0=0.0,
                rps=sampler(int(d * RATE_HZ), np.random.default_rng(100 + i)),
                source="test",
            )
            for i, d in enumerate(durations)
        ]
    )
    d = discrepancy(a, real)
    assert d["overall_mean"] < 1.0
    assert d["rotor_var"] < 0.3
    assert d["acf"] < 0.1
    assert d["xcorr"] < 0.1


def test_to_common_grid_decimates_without_aliasing_and_keeps_gaps() -> None:
    native = 400.0
    t = np.arange(int(40 * native)) / native
    # 120 Hz ripple would fold to 40 Hz at 100 Hz without the anti-alias
    # filter; the 2 Hz component must survive it.
    x = 100.0 + 5.0 * np.sin(2 * np.pi * 2.0 * t) + 5.0 * np.sin(2 * np.pi * 120.0 * t)
    grid = to_common_grid(t, np.tile(x, (2, 1)), native)
    assert grid.shape == (2, int(np.floor(t[-1] * RATE_HZ)) + 1)
    expected = 100.0 + 5.0 * np.sin(2 * np.pi * 2.0 * np.arange(grid.shape[1]) / RATE_HZ)
    inner = slice(int(2 * RATE_HZ), -int(2 * RATE_HZ))
    assert np.allclose(grid[0, inner], expected[inner], atol=0.4)

    holed = np.tile(x, (2, 1))
    holed[0, int(10 * native) : int(12 * native)] = np.nan
    gapped = to_common_grid(t, holed, native)
    hole = gapped[0, int(10.2 * RATE_HZ) : int(11.8 * RATE_HZ)]
    assert np.isnan(hole).all()
    assert not np.isnan(gapped[1]).any()
