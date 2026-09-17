"""The frozen noise-model-v2 round gates, at their boundaries.

Synthetic inputs only: every test plants a measurement that sits exactly on a
frozen threshold, or one step past it, and asserts the verdict flips.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.noise_model import gates as G
from experiments.stochastic_fit import revised_eval as RE

SR = RE.SR


def dregon(value: float | dict[str, float]) -> dict[str, float]:
    if isinstance(value, dict):
        return {s.key: float(value[s.recording]) for s in G.DREGON_CRUISE_SUPPORTS}
    return {s.key: float(value) for s in G.DREGON_CRUISE_SUPPORTS}


def michaels(per_regime: dict[str, float]) -> dict[str, float]:
    return {s.key: float(per_regime[s.regime]) for s in G.MICHAELS_SUPPORTS}


def passing_michaels() -> dict[str, float]:
    """Regime MAEs whose equal-regime mean is well inside the frozen bound."""
    return michaels({"standby": 0.3, "ramp": 4.0, "cruise": 0.7})


# ── the DREGON cruise PIT gate ──────────────────────────────────────────────


def test_dregon_point_target_is_the_frozen_30_percent_closure() -> None:
    at = G.hppnet_gate(dregon(G.DREGON_PIT_TARGET) | passing_michaels(), {})
    over = G.hppnet_gate(dregon(G.DREGON_PIT_TARGET + 1e-9) | passing_michaels(), {})
    assert at["dregon_cruise"]["pass"] and at["pass"]
    assert not over["dregon_cruise"]["pass"] and not over["pass"]
    assert over["dregon_cruise"]["checks"]["point_target"] is False
    # the target is real + 0.7 * (baseline - real) on the frozen v2 scalars
    assert (
        pytest.approx(
            G.DREGON_REAL_PIT_MAE
            + G.DREGON_GAP_FRACTION * (G.DREGON_BASELINE_PIT_MAE - G.DREGON_REAL_PIT_MAE)
        )
        == G.DREGON_PIT_TARGET
    )


def test_dregon_one_sided_interval_can_fail_a_passing_mean() -> None:
    """Recording-level spread is decisive, not just the point mean."""
    spread = {
        "free-flight_nosource_room2": 1.0,
        "hovering_nosource_room2": 1.2,
        "updown_nosource_room2": 1.5,
        "rectangle_nosource_room2": 2.4,
        "spinning_nosource_room2": 2.6,
    }
    out = G.hppnet_gate(dregon(spread) | passing_michaels(), {})["dregon_cruise"]
    assert out["mean_candidate_rev_s"] < G.DREGON_PIT_TARGET
    assert out["candidate_interval"]["upper"] > G.DREGON_PIT_TARGET
    assert out["checks"]["point_target"] is True
    assert out["checks"]["interval_upper_within_target"] is False
    assert out["pass"] is False


def test_dregon_partial_cohort_fails_before_any_bootstrap() -> None:
    values = dregon(1.0)
    dropped = G.DREGON_CRUISE_SUPPORTS[2].key
    values.pop(dropped)
    out = G.hppnet_gate(values | passing_michaels(), {})["dregon_cruise"]
    assert out["missing_supports"] == [dropped]
    assert out["checks"]["cohort_complete"] is False
    assert out["pass"] is False


def test_dregon_real_arm_is_a_reproduction_check_not_a_gate_input() -> None:
    real = dregon(G.DREGON_REAL_PIT_MAE)
    out = G.hppnet_gate(dregon(1.0) | passing_michaels(), real)["dregon_cruise"]
    assert out["real_arm"]["mean_rev_s"] == pytest.approx(G.DREGON_REAL_PIT_MAE)
    assert out["real_arm"]["relative"] == pytest.approx(0.0)
    # a real arm far from the frozen value does not change the verdict
    off = G.hppnet_gate(dregon(1.0) | passing_michaels(), dregon(5.0))["dregon_cruise"]
    assert off["pass"] is True
    assert off["real_arm"]["relative"] > 3.0


def test_dregon_paired_improvement_interval_only_with_baselines() -> None:
    without = G.hppnet_gate(dregon(1.0) | passing_michaels(), {})["dregon_cruise"]
    assert without["paired_improvement_interval"] is None
    with_base = G.hppnet_gate(
        dregon(1.0) | passing_michaels(), {}, baseline_by_support=dregon(2.0)
    )["dregon_cruise"]
    assert with_base["paired_improvement_interval"]["mean"] == pytest.approx(1.0)


# ── Michael's FLY124 ratio gate ─────────────────────────────────────────────


def test_michaels_ratio_boundary_is_1_05_of_the_frozen_regime_mean() -> None:
    at = G.MICHAELS_BASELINE_REGIME_MEAN * G.MICHAELS_RATIO_MAX
    ok = G.hppnet_gate(dregon(1.0) | michaels({r: at for r in ("standby", "ramp", "cruise")}), {})
    bad = G.hppnet_gate(
        dregon(1.0) | michaels({r: at * (1 + 1e-9) for r in ("standby", "ramp", "cruise")}), {}
    )
    assert ok["michaels_fly124"]["aggregate_ratio"] == pytest.approx(G.MICHAELS_RATIO_MAX)
    assert ok["michaels_fly124"]["pass"] and ok["pass"]
    assert bad["michaels_fly124"]["pass"] is False and bad["pass"] is False
    assert at == pytest.approx(G.MICHAELS_PIT_BOUND)


def test_michaels_is_a_ratio_of_means_not_a_mean_of_ratios() -> None:
    """A near-zero standby MAE must not buy a broken ramp."""
    values = michaels({"standby": 1e-6, "ramp": 9.5, "cruise": 0.5})
    out = G.hppnet_gate(dregon(1.0) | values, {})["michaels_fly124"]
    mean_of_ratios = float(
        np.mean([out["per_regime"][r]["ratio"] for r in ("standby", "ramp", "cruise")])
    )
    assert out["aggregate_ratio"] == pytest.approx((1e-6 + 9.5 + 0.5) / 3 / 3.026661398168452)
    assert mean_of_ratios < 1.0 < out["aggregate_ratio"]
    assert out["pass"] is False


def test_michaels_regime_mae_averages_over_that_regimes_blocks() -> None:
    values = {s.key: 1.0 for s in G.MICHAELS_SUPPORTS}
    values[G.MICHAELS_SUPPORTS[3].key] = 3.0  # one of the two cruise blocks
    out = G.hppnet_gate(dregon(1.0) | values, {})["michaels_fly124"]
    assert out["per_regime"]["cruise"]["n_blocks"] == 2
    assert out["per_regime"]["cruise"]["candidate_mae"] == pytest.approx(2.0)
    assert out["per_regime"]["ramp"]["n_blocks"] == 1
    assert out["rig_candidate_mae"] == pytest.approx((1.0 + 1.0 + 2.0) / 3)


def test_michaels_partial_cohort_fails() -> None:
    values = passing_michaels()
    values.pop(G.MICHAELS_SUPPORTS[2].key)  # the single ramp block
    out = G.hppnet_gate(dregon(1.0) | values, {})["michaels_fly124"]
    assert out["checks"]["cohort_complete"] is False
    assert out["pass"] is False


# ── the spectrogram proxy gate ──────────────────────────────────────────────


def audio(rng: np.random.Generator, *, seconds: float = 2.0, gain_db: float = 0.0) -> np.ndarray:
    n = int(seconds * SR)
    x = rng.standard_normal((2, n))
    return x * float(10.0 ** (gain_db / 20.0))


def proxy_inputs(gain_db: float, *, groups: tuple[str, ...] = ("dregon", "michaels")) -> tuple:
    rng = np.random.default_rng(7)
    real: dict[str, np.ndarray] = {}
    synth: dict[str, np.ndarray] = {}
    for s in (*G.DREGON_CRUISE_SUPPORTS, *G.MICHAELS_CRUISE_SUPPORTS):
        if s.rig not in groups:
            continue
        x = audio(rng)
        real[s.key] = x
        synth[s.key] = x * float(10.0 ** (gain_db / 20.0))
    return real, synth


def test_proxy_gate_boundary_is_the_closure_070_threshold() -> None:
    """A pure level offset just inside the gate passes; one step past it fails."""
    for group, gate_db in G.PROXY_LTAS_GATE_DB.items():
        rig = group.split("_")[0]
        real, synth = proxy_inputs(gate_db - 1e-6, groups=(rig,))
        at = G.proxy_gate(real, synth)["groups"][group]
        assert at["mean_ltas_abs_db"] == pytest.approx(gate_db, abs=1e-5)
        assert at["pass"] is True
        assert at["margin_db"] > 0.0
        real, synth = proxy_inputs(gate_db + 1e-3, groups=(rig,))
        over = G.proxy_gate(real, synth)["groups"][group]
        assert over["mean_ltas_abs_db"] > gate_db
        assert over["pass"] is False


def test_proxy_gate_thresholds_match_the_approved_closure() -> None:
    for group, ref in G.PROXY_REFERENCE.items():
        assert G.PROXY_LTAS_GATE_DB[group] == pytest.approx(
            ref["c3_db"] - 0.70 * (ref["c3_db"] - ref["oracle_db"])
        )
    assert G.PROXY_LTAS_GATE_DB["dregon_cruise"] == pytest.approx(1.979, abs=5e-4)
    assert G.PROXY_LTAS_GATE_DB["michaels_cruise"] == pytest.approx(1.220, abs=5e-4)


def test_proxy_gate_fails_on_a_missing_support_and_reports_spread() -> None:
    real, synth = proxy_inputs(0.5)
    dropped = G.DREGON_CRUISE_SUPPORTS[0].key
    real.pop(dropped)
    synth.pop(dropped)
    out = G.proxy_gate(real, synth)
    assert out["groups"]["dregon_cruise"]["missing_supports"] == [dropped]
    assert out["groups"]["dregon_cruise"]["pass"] is False
    assert out["pass"] is False
    assert out["groups"]["michaels_cruise"]["spread_ltas_abs_db"] == pytest.approx(0.0, abs=1e-9)


def test_proxy_gate_reports_injected_mr_ltas_without_deciding_on_it() -> None:
    real, synth = proxy_inputs(0.1)
    out = G.proxy_gate(real, synth, mr_ltas_fn=lambda _r, _a: 99.0)
    for group in G.PROXY_LTAS_GATE_DB:
        assert out["groups"][group]["mean_mr_ltas"] == pytest.approx(99.0)
        assert out["groups"][group]["pass"] is True
    assert out["pass"] is True
    bare = G.proxy_gate(real, synth)
    assert bare["groups"]["dregon_cruise"]["mean_mr_ltas"] is None


# ── the likelihood gate ─────────────────────────────────────────────────────


def cell(
    support: G.ScoredSupport,
    *,
    model_comb: float,
    model_floor: float,
    oracle: float = 1.0,
    power: float = 1.0,
    n_frames: int = 6,
) -> G.SpectralCell:
    freqs = np.fft.rfftfreq(G.LIKELIHOOD_N_FFT, 1.0 / SR)
    shape = (2, n_frames, freqs.size)
    comb = freqs >= G.BAND_EDGE_HZ
    m = np.where(comb, model_comb, model_floor)
    times = np.arange(n_frames) * (G.LIKELIHOOD_HOP / SR)
    return G.SpectralCell(
        support=support,
        freqs_hz=freqs,
        frame_times_s=times,
        power=np.full(shape, power),
        model=np.broadcast_to(m, shape).copy(),
        oracle=np.full(shape, oracle),
    )


def cruise_cells(**kw: float) -> list[G.SpectralCell]:
    return [cell(s, **kw) for s in G.MICHAELS_CRUISE_SUPPORTS]  # type: ignore[arg-type]


def test_likelihood_gate_is_decided_on_the_comb_band_only() -> None:
    """Comb better than the oracle passes even when the floor band is worse."""
    cells = cruise_cells(model_comb=1.0, model_floor=8.0, oracle=1.3, power=1.0)
    out = G.likelihood_gate(cells)
    assert out["per_band"]["comb"]["margin_nats_per_s"] < 0.0
    assert out["per_band"]["floor"]["margin_nats_per_s"] > 0.0
    assert out["decisive_band"] == "comb"
    assert out["pass"] is True
    assert out["frozen_margin_nats_per_s"] == out["per_band"]["comb"]["margin_nats_per_s"]
    assert out["per_band"]["comb"]["band_hz"] == [300.0, RE.OBS_F_MAX]
    assert out["per_band"]["floor"]["band_hz"] == [RE.OBS_F_MIN, 300.0]


def test_likelihood_gate_requires_strictly_below_the_oracle() -> None:
    equal = G.likelihood_gate(cruise_cells(model_comb=1.0, model_floor=1.0, oracle=1.0))
    assert equal["per_band"]["comb"]["margin_nats_per_s"] == pytest.approx(0.0)
    assert equal["pass"] is False
    worse = G.likelihood_gate(cruise_cells(model_comb=2.0, model_floor=1.0, oracle=1.0))
    assert worse["per_band"]["comb"]["margin_nats_per_s"] > 0.0
    assert worse["pass"] is False


def test_likelihood_gate_pools_only_the_two_frozen_cruise_supports() -> None:
    cells = cruise_cells(model_comb=1.0, model_floor=1.0, oracle=1.3)
    one = G.likelihood_gate(cells[:1])
    assert one["missing_supports"] == [G.MICHAELS_CRUISE_SUPPORTS[1].key]
    assert one["pass"] is False
    extra = [*cells, cell(G.MICHAELS_SUPPORTS[0], model_comb=1.0, model_floor=1.0, oracle=1.3)]
    out = G.likelihood_gate(extra)
    assert out["unexpected_supports"] == [G.MICHAELS_SUPPORTS[0].key]
    assert out["pass"] is False


def test_likelihood_gate_score_is_the_frozen_composite_risk() -> None:
    """The pooled per-band risk equals revised_eval.composite_score on the same cells."""
    cells = cruise_cells(model_comb=1.0, model_floor=1.0, oracle=1.3)
    out = G.likelihood_gate(cells)
    items = []
    for c in cells:
        band = RE.observation_band(c.freqs_hz) & (c.freqs_hz >= G.BAND_EDGE_HZ)
        nll, n_cells = RE.marginal_frame_nll(c.power, c.model, band)
        items.append(
            RE.FrameScore(
                c.support.window,
                c.support.start_s + c.frame_times_s,
                nll,
                n_cells,
                n_fft=G.LIKELIHOOD_N_FFT,
                hop=G.LIKELIHOOD_HOP,
            )
        )
    assert out["per_band"]["comb"]["model_nats_per_s"] == pytest.approx(
        RE.composite_score(items)["score"]
    )


def test_likelihood_gate_cross_checks_the_recorded_oracle() -> None:
    cells = cruise_cells(model_comb=1.0, model_floor=1.0, oracle=1.3)
    rows = [
        dict(
            arm="oracle_np",
            n_fft=G.LIKELIHOOD_N_FFT,
            hop=G.LIKELIHOOD_HOP,
            support=c.support.key,
            available=True,
            weighted_nats=-100.0,
            unique_seconds=4.0,
        )
        for c in cells
    ]
    out = G.likelihood_gate(cells, oracle_json=dict(rows=rows, schema="x"))
    rep = out["oracle"]["reproduction"]
    assert rep["available"] is True
    assert rep["recorded_pooled_nats_per_s"] == pytest.approx(-200.0 / 8.0)
    # a setting that is not in the file is reported as unavailable, never guessed
    thin = G.likelihood_gate(cells, oracle_json=dict(rows=rows[:1]))
    assert thin["oracle"]["reproduction"]["available"] is False
