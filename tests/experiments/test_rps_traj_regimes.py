"""The flight-regime modules must mean what they say.

Phases are read off the speed level by a fixed rule, the whole-flight layer
replays what it measured, and the cruise switch is fitted through a filter:
each can be silently wrong while looking plausible.  So the phase rule is run
on a flight whose phase changes are known by construction, the layer on
flights built from a known program, and the filter against an independent
Kalman recursion and against the level law that generated the data.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.stats import kurtosis

from data_processing.trajectory_model import Params
from experiments.rps_traj import regime_campaign as rc
from experiments.rps_traj.cruise_api import burn_samples, cut_blocks
from experiments.rps_traj.data import RATE_HZ, Flight
from experiments.rps_traj.phases import (
    DREGON_AUDIO_STANDBY_S,
    DREGON_IDLE_RPS,
    PHASES,
    block_mean,
    cruise_groups,
    folds,
    landmarks,
    phase_labels,
    readouts,
    settled_cruise,
    with_recovered_standby,
)
from experiments.rps_traj.switching import (
    SwitchingCruise,
    fit_levels,
    levels_from_vector,
    state_space,
    stationary,
    total_logpdf,
    vector_from_levels,
)
from experiments.rps_traj.transients import fit_phase_layer, replay_error
from tracking.rotors import MIXER

FS = RATE_HZ


def _params() -> Params:
    return Params(
        mu=np.array([86.0, 72.0, 82.0, 74.0]),
        theta=0.1,
        tau_slow=np.array([2.0, 2.8, 4.0, 3.2]),
        sigma_slow=np.array([0.2, 0.5, 0.5, 0.2]),
        f0=np.array([1.4, 3.0, 2.6, 0.8]),
        zeta=np.array([1.1, 0.7, 0.8, 1.6]),
        sigma_osc=np.array([1.3, 0.4, 0.45, 1.2]),
        tau_e=0.1,
        sigma_e=0.05,
        s_c=2.0,
        s_r=np.array([1.0, 0.5, 0.8, 0.3]),
    )


def _s(index: int | None, fs: float = FS) -> float:
    """A landmark in seconds; a missing one fails the test."""
    assert index is not None
    return index / fs


# ─── a flight whose phases are known ──────────────────────────────────────────

#: Phase changes of the programmed flight (s): motors on, start-up settled,
#: take-off onset, arrival at hover, landing onset, motors off.
T_ON, T_SETTLED, T_TAKEOFF, T_HOVER, T_LAND, T_OFF = 1.0, 3.0, 9.0, 10.0, 40.0, 43.0
STANDBY = np.array([30.0, 34.0, 40.0, 36.0])
HOVER = np.array([88.0, 72.0, 82.0, 74.0])


def _programmed(
    rng: np.random.Generator,
    *,
    fan: np.ndarray | None = None,
    hold_s: float = 0.0,
    jitter: float = 0.3,
) -> np.ndarray:
    """``(4, n)`` flight at 100 Hz: zero, a start-up in which every rotor
    spikes to 1.4 x the MEAN standby level in unison, standby, a linear ramp,
    hover with small jitter, a landing that drops to half, fans the rotors out
    by ``fan`` (roll, pitch, yaw) x hover level and spins down, zero."""
    t = np.arange(int((T_OFF + 2.0 + hold_s) * FS)) / FS
    sb, h = STANDBY.mean(), HOVER.mean()
    to, hov, land, off = T_TAKEOFF + hold_s, T_HOVER + hold_s, T_LAND + hold_s, T_OFF + hold_s
    spike = sb * (1.0 + 0.4 * np.exp(-0.5 * ((t - (T_ON + 0.8)) / 0.12) ** 2))
    rise = np.clip((t - T_ON) / 0.4, 0.0, 1.0)
    settle = np.clip((t - (T_ON + 1.2)) / (T_SETTLED - T_ON - 1.2), 0.0, 1.0)
    x = np.zeros((4, t.size))
    ground = (t >= T_ON) & (t < to)
    x[:, ground] = (rise * spike)[None, ground] + settle[None, ground] * (STANDBY - sb)[:, None]
    ramp = (t >= to) & (t < hov)
    frac = (t[ramp] - to) / (hov - to)
    x[:, ramp] = STANDBY[:, None] + frac[None, :] * (HOVER - STANDBY)[:, None]
    air = (t >= hov) & (t < land)
    x[:, air] = HOVER[:, None] + jitter * rng.standard_normal((4, int(air.sum())))
    down = (t >= land) & (t < off)
    u = (t[down] - land) / (off - land)
    level = np.where(u < 0.1, 1.0 - 5.0 * u, 0.5 * (1.0 - (u - 0.1) / 0.9))
    x[:, down] = HOVER[:, None] * level[None, :]
    if fan is not None:
        course = np.sin(np.pi * np.clip(u, 0.0, 1.0)) ** 2
        x[:, down] += h * (MIXER[:, 1:] @ fan)[:, None] * course[None, :]
    return np.clip(x, 0.0, None)


def _flight(rps: np.ndarray, name: str = "f", source: str = "test:rps") -> Flight:
    return Flight(rig="test", flight=name, fs=FS, t0=0.0, rps=rps, source=source)


def test_landmarks_find_the_programmed_phase_changes() -> None:
    rps = _programmed(np.random.default_rng(0))
    marks = landmarks(rps, FS)
    assert marks is not None and marks.whole
    assert _s(marks.motors_on) == pytest.approx(T_ON, abs=0.05)
    assert marks.standby is not None and marks.standby_level is not None
    assert marks.hover_in is not None
    # standby starts where the ROTOR MEAN is back within 8 % of its plateau: the
    # 0.4 spike (centre 0.8 s after motors-on, width 0.12 s) has decayed to 0.08
    spike_over = T_ON + 0.8 + 0.12 * np.sqrt(2.0 * np.log(0.4 / 0.08))
    assert _s(marks.standby[0]) == pytest.approx(spike_over, abs=0.1)
    assert _s(marks.takeoff) == pytest.approx(T_TAKEOFF, abs=0.15)
    # arrival = 90 % of the step, 0.9 s up a 1 s ramp
    assert _s(marks.arrival) == pytest.approx(T_TAKEOFF + 0.9, abs=0.08)
    # the landing mark is where the rotor mean passes 75 % of the level left: 0.15 s in
    assert _s(marks.landing) == pytest.approx(T_LAND + 0.15, abs=0.08)
    assert _s(marks.motors_off) == pytest.approx(T_OFF, abs=0.6)
    np.testing.assert_allclose(marks.standby_level, STANDBY, atol=0.3)
    np.testing.assert_allclose(marks.hover_in, HOVER, atol=0.3)


def test_settled_cruise_and_labels_keep_the_transients_out() -> None:
    rps = _programmed(np.random.default_rng(1))
    marks = landmarks(rps, FS)
    assert marks is not None
    (seg,) = settled_cruise(rps, FS)
    assert seg.start / FS >= T_HOVER + 2.5 and seg.stop / FS <= T_LAND - 2.0
    # the frozen rule alone would have kept part of the ramp
    assert np.ptp(rps[:, seg].mean(axis=0)) < 3.0
    lab = phase_labels(marks)
    changes = lab[np.flatnonzero(np.diff(lab)) + 1]
    assert [PHASES[i] for i in changes] == [
        "startup",
        "standby",
        "takeoff",
        "cruise",
        "landing",
        "zero",
    ]
    assert lab[0] == 0 and lab[int((T_TAKEOFF + 0.5) * FS)] == PHASES.index("takeoff")


def test_readouts_measure_the_programmed_flight() -> None:
    out = readouts(_programmed(np.random.default_rng(2), fan=np.array([0.2, -0.1, 0.0])), FS)
    assert out["valid"] == 1.0
    assert out["startup_peak_rel"] == pytest.approx(1.4, abs=0.05)
    assert out["standby_rel"] == pytest.approx(STANDBY.mean() / HOVER.mean(), abs=0.01)
    assert out["standby_spread_rel"] == pytest.approx(np.ptp(STANDBY) / HOVER.mean(), abs=0.01)
    assert out["takeoff_rise_s"] == pytest.approx(0.9, abs=0.1)
    assert out["airborne_s"] == pytest.approx(T_LAND + 0.15 - T_TAKEOFF - 0.9, abs=0.2)
    # a fan-out of 0.2 (roll) and -0.1 (pitch) spreads the rotors by up to 0.6 of hover
    assert 0.45 < out["landing_spread_rel"] < 0.8
    flat = readouts(_programmed(np.random.default_rng(2)), FS)
    assert flat["landing_spread_rel"] < 0.3


def test_recovered_standby_is_max_of_idle_and_command() -> None:
    name, (start, _) = next(iter(DREGON_AUDIO_STANDBY_S.items()))
    command = np.zeros((4, 600))
    command[:, 400:] = np.linspace(0.0, 60.0, 200)[None, :]
    patched = with_recovered_standby(_flight(command, name, "DREGON-frames:motors_command"))
    i = int(round(start * FS))
    assert (patched.rps[:, :i] == 0.0).all()
    assert (patched.rps[:, i:400] == DREGON_IDLE_RPS).all()
    np.testing.assert_array_equal(patched.rps[:, 500:], command[:, 500:])
    measured = _flight(command, name, "DREGON-frames:motors_measured")
    assert with_recovered_standby(measured) is measured


def test_block_mean_and_blocks() -> None:
    x = np.arange(4 * 23, dtype=float).reshape(4, 23)
    y = block_mean(x, 100.0, 25.0)
    assert y.shape == (4, 5)
    np.testing.assert_allclose(y[:, 0], x[:, :4].mean(axis=1))
    blocks = cut_blocks(np.random.default_rng(0).standard_normal((4, 1000)) + 5.0, 12.5)
    assert [b.shape[1] for b in blocks] == [375, 375, 250]
    assert all(abs(b.mean(axis=1)).max() < 1e-12 for b in blocks)
    assert cut_blocks(np.full((4, 40), np.nan), 12.5) == []


def test_folds_hold_every_group_out_once_and_spread_strata() -> None:
    names = [f"g{i}" for i in range(10)]
    strata = ["a"] * 5 + ["b"] * 5
    held = folds(names, strata)
    assert sorted(g for h in held for g in h) == sorted(names)
    assert len(held) == 5 and all(len(h) == 2 for h in held)
    assert all({strata[names.index(g)] for g in h} == {"a", "b"} for h in held)
    assert held == folds(list(reversed(names)), list(reversed(strata)))
    assert [len(h) for h in folds(["x", "y", "z"])] == [1, 1, 1]


def test_single_flight_is_split_in_time() -> None:
    rps = _programmed(np.random.default_rng(3))
    groups = cruise_groups([_flight(rps, "only")], 25.0)
    assert sorted(groups) == ["only#a", "only#b"]
    a, b = groups["only#a"][0], groups["only#b"][0]
    assert abs(a.shape[1] - b.shape[1]) <= 1 and a.shape[1] > 100


# ─── the whole-flight layer ───────────────────────────────────────────────────


def _layer_flights(fans: list[np.ndarray]) -> list[Flight]:
    rng = np.random.default_rng(5)
    return [
        _flight(_programmed(rng, fan=fan, hold_s=hold), f"f{i}")
        for i, (fan, hold) in enumerate(zip(fans, (0.0, 2.0, 5.0, 1.0)))
    ]


def test_layer_recovers_the_program() -> None:
    fans = [
        np.array([0.20, -0.10, 0.0]),
        np.array([0.10, -0.20, 0.0]),
        np.array([0.0, -0.15, 0.05]),
    ]
    layer = fit_phase_layer(_layer_flights(fans))
    assert layer.counts == {"takeoffs": 3, "landings": 3}
    assert layer.startup_common.max() == pytest.approx(1.4, abs=0.05)
    # every rotor spikes in unison: the per-rotor pattern is not handed in yet at the spike
    spike = int(0.8 * FS)
    assert abs(layer.startup_split[spike]) < 0.15 and layer.startup_split[-1] == pytest.approx(1.0)
    np.testing.assert_allclose(layer.standby_level, STANDBY, atol=0.3)
    step = HOVER.mean() - STANDBY.mean()
    assert layer.takeoff_rate == pytest.approx(step / 1.0, rel=0.1)
    assert layer.landing_amp_mean is not None
    np.testing.assert_allclose(layer.landing_amp_mean, np.mean(fans, axis=0), atol=0.03)
    assert np.exp(layer.pre_takeoff_log_s[0]) == pytest.approx(
        np.exp(np.mean(np.log([8.0, 10.0, 13.0]))), rel=0.05
    )


def test_layer_samples_whole_flights_like_its_training_flights() -> None:
    fans = [
        np.array([0.20, -0.10, 0.0]),
        np.array([0.10, -0.20, 0.0]),
        np.array([0.0, -0.15, 0.05]),
    ]
    layer = fit_phase_layer(_layer_flights(fans))
    params = _params()
    params.mu = HOVER.copy()
    params.s_c, params.s_r = 0.0, np.zeros(4)
    cruise = SwitchingCruise(name="gauss", rate_hz=25.0, params=params)
    rng = np.random.default_rng(6)
    for _ in range(3):
        draw = layer.sample(cruise, rng, airborne_s=20.0)
        out = readouts(draw["rps"], layer.fs)
        assert out["valid"] == 1.0
        assert out["startup_peak_rel"] == pytest.approx(1.4, abs=0.06)
        assert out["standby_rel"] == pytest.approx(STANDBY.mean() / HOVER.mean(), abs=0.03)
        assert out["takeoff_rise_s"] == pytest.approx(0.9, abs=0.15)
        assert out["airborne_s"] == pytest.approx(20.0, abs=0.5)
        assert (draw["rps"] >= 0.0).all() and (draw["rps"][:, :50] == 0.0).all()


def test_replay_error_separates_unison_spike_from_rescaling() -> None:
    fans = [
        np.array([0.20, -0.10, 0.0]),
        np.array([0.10, -0.20, 0.0]),
        np.array([0.0, -0.15, 0.05]),
    ]
    flights = _layer_flights([*fans, np.array([0.15, -0.12, 0.02])])
    layer = fit_phase_layer(flights[:3])
    err = replay_error(layer, flights[3])
    assert err["startup_common"] < 0.5 and err["startup_split"] < 0.5
    # rescaling one template per rotor misses the unison spike by (level - mean) * 0.4
    assert err["startup_split_prop"] > 4.0 * err["startup_split"]
    assert err["landing_split"] < 0.3 * err["landing_split_prop"]


# ─── the switching cruise model ───────────────────────────────────────────────


def _kalman_logpdf(params: Params, rate: float, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """An independent dense Kalman filter: log predictive and last filtered mean."""
    phi, qs, qm, p0, h = state_space(params, rate)
    x, p = np.zeros(phi.shape[0]), p0.copy()
    out = []
    for t in range(y.shape[1]):
        xp, pp = phi @ x, phi @ p @ phi.T + qs + qm
        e, s = y[:, t] - h @ xp, h @ pp @ h.T
        out.append(
            -0.5 * (4 * np.log(2 * np.pi) + np.linalg.slogdet(s)[1] + e @ np.linalg.solve(s, e))
        )
        gain = pp @ h.T @ np.linalg.inv(s)
        x, p = xp + gain @ e, pp - gain @ s @ gain.T
    return np.array(out), x


TWO_LEVELS = (np.array([[-0.5, 0.5], [2.5, -2.5]]), np.array([1.0, 9.0]))


def _two_level(params: Params, rate: float) -> SwitchingCruise:
    gen, k2 = TWO_LEVELS
    return SwitchingCruise(
        name="switch2",
        rate_hz=rate,
        params=params,
        gen=gen,
        kappa2=k2 / float(stationary(gen) @ k2),
    )


def test_one_level_filter_is_the_kalman_filter() -> None:
    params, rate = _params(), 12.5
    model = SwitchingCruise(name="gauss", rate_hz=rate, params=params)
    y = model.sample(300, np.random.default_rng(0))
    ref, _ = _kalman_logpdf(params, rate, y)
    np.testing.assert_allclose(model.logpdf(y, 0), ref, atol=1e-9)
    same = SwitchingCruise(
        name="x", rate_hz=rate, params=params, gen=TWO_LEVELS[0], kappa2=np.ones(2)
    )
    np.testing.assert_allclose(same.logpdf(y, 0), ref, atol=1e-9)
    np.testing.assert_allclose(model.logpdf(y, 40), ref[40:], atol=1e-9)


def test_level_vector_round_trip_and_normalisation() -> None:
    for k in (2, 3):
        rng = np.random.default_rng(k)
        theta = rng.normal(size=k * (k - 1) + k - 1)
        gen, k2 = levels_from_vector(theta, k)
        assert np.allclose(gen.sum(axis=1), 0.0) and (np.diff(k2) > 0).all()
        assert float(stationary(gen) @ k2) == pytest.approx(1.0)
        assert np.allclose(stationary(gen) @ expm(gen * 0.3), stationary(gen))
        np.testing.assert_allclose(vector_from_levels(gen, k2), theta, atol=1e-9)


def test_switching_keeps_second_moments_and_adds_tails() -> None:
    params, rate = _params(), 25.0
    rng = np.random.default_rng(1)
    gauss = SwitchingCruise(name="gauss", rate_hz=rate, params=params)
    two = _two_level(params, rate)
    n = 150_000
    yg, (yt, path) = gauss.sample(n, rng), two.sample(n, rng, return_levels=True)
    target = params.rotor_var(include_offset=False)
    np.testing.assert_allclose(yg.var(axis=1), target, rtol=0.08)
    np.testing.assert_allclose(yt.var(axis=1), target, rtol=0.08)
    lag = int(0.2 * rate)
    assert abs(kurtosis(yg[:, lag:] - yg[:, :-lag], axis=1)).max() < 0.15
    assert kurtosis(yt[:, lag:] - yt[:, :-lag], axis=1).min() > 0.5
    # the loud level is visited as often and as long as the generator says
    assert path.mean() == pytest.approx(two.occupancy[1], abs=0.02)
    runs = np.diff(np.flatnonzero(np.diff(np.r_[0, path, 0])))[::2]
    assert runs.mean() / rate == pytest.approx(two.dwell_s[1], rel=0.15)
    # leaving the measurement process out removes exactly its variance
    shaft = gauss.sample(n, rng, shaft_only=True)
    np.testing.assert_allclose(shaft.var(axis=1), target - params.sigma_e**2, rtol=0.08)


def test_likelihood_prefers_the_level_law_that_made_the_data() -> None:
    params, rate = _params(), 12.5
    truth = _two_level(params, rate)
    rng = np.random.default_rng(2)
    blocks = [b for _ in range(8) for b in cut_blocks(truth.sample(375, rng), rate)]
    burn = burn_samples(rate)
    at_truth = total_logpdf(truth, blocks, burn)
    gauss = SwitchingCruise(name="gauss", rate_hz=rate, params=params)
    slow = SwitchingCruise(
        name="x", rate_hz=rate, params=params, gen=TWO_LEVELS[0] / 20.0, kappa2=truth.kappa2
    )
    flat = SwitchingCruise(
        name="x", rate_hz=rate, params=params, gen=TWO_LEVELS[0], kappa2=np.array([0.9, 1.5])
    )
    flat.kappa2 = flat.kappa2 / float(flat.occupancy @ flat.kappa2)
    n = sum(b.shape[1] - burn for b in blocks)
    assert (at_truth - total_logpdf(gauss, blocks, burn)) / n > 0.2
    assert at_truth > total_logpdf(slow, blocks, burn)
    assert at_truth > total_logpdf(flat, blocks, burn)


@pytest.mark.slow
def test_fit_levels_recovers_the_level_law() -> None:
    params, rate = _params(), 12.5
    truth = _two_level(params, rate)
    rng = np.random.default_rng(3)
    blocks = [b for _ in range(40) for b in cut_blocks(truth.sample(375, rng), rate)]
    fitted = fit_levels(params, blocks, rate, 2)
    assert np.sqrt(fitted.kappa2[1] / fitted.kappa2[0]) == pytest.approx(3.0, rel=0.2)
    assert fitted.occupancy[1] == pytest.approx(truth.occupancy[1], abs=0.05)
    assert fitted.dwell_s[1] == pytest.approx(truth.dwell_s[1], rel=0.35)
    assert fitted.n_params == 26


def test_forecast_continues_from_the_filtered_state() -> None:
    params, rate = _params(), 12.5
    model = SwitchingCruise(name="gauss", rate_hz=rate, params=params)
    rng = np.random.default_rng(4)
    history = model.sample(200, rng)
    _, x = _kalman_logpdf(params, rate, history)
    phi, _, _, _, h = state_space(params, rate)
    paths = model.forecast(history, 3, 4000, rng)
    assert paths.shape == (4000, 4, 3)
    sd = paths.std(axis=0) / np.sqrt(4000)
    for lead in range(3):
        mean = h @ np.linalg.matrix_power(phi, lead + 1) @ x
        assert (np.abs(paths[:, :, lead].mean(axis=0) - mean) < 5.0 * sd[:, lead]).all()
    two = _two_level(params, rate)
    assert two.forecast(history, 5, 7, rng).shape == (7, 4, 5)
    back = SwitchingCruise.from_dict(two.to_dict())
    np.testing.assert_allclose(back.logpdf(history, 12), two.logpdf(history, 12), atol=1e-9)


# ─── the campaign's scoring and aggregation ───────────────────────────────────


def test_first_pass_cuts_after_the_first_landing() -> None:
    one = _programmed(np.random.default_rng(7))[:, ::4]
    twice = np.concatenate([one, one], axis=1)
    cut = rc.first_pass(twice, 25.0, float(HOVER.mean()))
    assert one.shape[1] - 60 < cut.shape[1] < one.shape[1] + 30
    airborne = one[:, : int(30 * 25.0)]
    assert rc.first_pass(airborne, 25.0, float(HOVER.mean())).shape == airborne.shape


def test_score_flights_counts_a_diverged_flight_as_invalid() -> None:
    rng = np.random.default_rng(8)
    held = [block_mean(_programmed(rng), FS, 25.0)]
    good = [block_mean(_programmed(rng, hold_s=h), FS, 25.0) for h in (0.0, 1.0, 2.0)]
    bad = good[0].copy()
    bad[:, 500:] = 1e9
    out = rc.score_flights([*good, bad], held, 25.0)
    assert out["valid_rate"] == pytest.approx(0.75) and out["diverged_rate"] == pytest.approx(0.25)
    assert out["readouts"]["startup_peak_rel"]["crps"] < 0.02
    assert out["readouts"]["standby_s"]["n_gen"] == 3 and out["speed_w1"] < 0.05
    assert out["inc_q999_gen_0.2"] < 10.0


def test_aggregate_means_ranks_and_parameter_sets() -> None:
    def scores(logpdf: float, energy: float, w1: float) -> dict:
        horizons = {
            h: {"energy": energy, "crps": energy, "energy_clim": 2.0, "crps_clim": 2.0}
            for h in ("0.2", "1", "2")
        }
        return {"logpdf": logpdf, "forecast": {"horizons": horizons}, "sample": {"inc_w1_0.2": w1}}

    def unit(rig: str, fold: str, family: str, arms: dict) -> dict:
        body = {
            name: {
                "n_params": 3,
                "fit_s": 1.0,
                "params": {"name": name},
                **({"scores": s} if s else {}),
            }
            for name, s in arms.items()
        }
        return {
            "kind": "cruise",
            "rig": rig,
            "fold": fold,
            "family": family,
            "rate_hz": 25.0,
            "arms": body,
        }

    rows = [
        unit(
            "michaels",
            "0",
            "ours",
            {"gauss": scores(-4.0, 1.0, 0.2), "switch2": scores(-3.0, 0.8, 0.1)},
        ),
        unit(
            "michaels",
            "1",
            "ours",
            {"gauss": scores(-5.0, 1.2, 0.4), "switch2": scores(-3.5, 1.0, 0.1)},
        ),
        unit("michaels", "0", "var", {"var": scores(-3.8, 0.5, 0.3)}),
        unit("michaels", "1", "var", {"var": scores(-4.4, 0.7, 0.3)}),
        unit("michaels", "all", "ours", {"gauss": None, "switch2": None}),
        {
            "kind": "flight",
            "rig": "michaels",
            "fold": "0",
            "arm": "ours",
            "replay_error": {},
            "scores": {
                "valid_rate": 1.0,
                "readouts": {"standby_s": {"crps": 2.0}},
                "speed_w1": 0.1,
                "inc_w1_0.2": 0.01,
                "inc_w1_1": 0.02,
                "inc_q999_gen_0.2": 0.3,
                "inc_q999_real_0.2": 0.4,
            },
        },
    ]
    summary = rc.aggregate(rows)
    arms = summary["cruise"]["michaels"]
    assert arms["gauss"]["mean"]["logpdf"] == {
        "mean": -4.5,
        "sd": pytest.approx(np.sqrt(0.5)),
        "n": 2,
    }
    assert arms["switch2"]["mean"]["es_skill_1"]["mean"] == pytest.approx(1.0 - 0.9 / 2.0)
    assert summary["rank"]["logpdf"] == {"switch2": 1.0, "var": 2.0, "gauss": 3.0}
    assert summary["rank"]["es_skill_0.2"]["var"] == 1.0
    assert (
        summary["rank"]["inc_w1_0.2"] == {"switch2": 1.0, "gauss": 2.5, "var": 2.5}
        or summary["rank"]["inc_w1_0.2"]["switch2"] == 1.0
    )
    assert sorted(summary["params"]["michaels"]["cruise"]) == ["gauss", "switch2"]
    flight = summary["flight"]["michaels"]["ours"]["mean"]
    assert flight["crps_standby_s"]["mean"] == 2.0 and flight["inc_q999_ratio_0.2"][
        "mean"
    ] == pytest.approx(0.75)
    text = rc.markdown(summary)
    assert "| switch2 | 3 | -3.250 | 1.00 |" in text and "whole flights, michaels" in text
