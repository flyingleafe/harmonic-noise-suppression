"""The v2 rig sampler's contracts: the pin, the guards, the path, the bank.

Every test here pins something a CONSUMER observes — a bank entry's speed laws,
a guard's refusal, the path's endpoints, a bank's byte-for-byte
reproducibility — rather than the sampler's internals. The expensive parts (the
expected-periodogram probe) are shared through one module-scoped fixture, and
every draw count is the smallest that can still fail.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from data_processing.noise_v2_pool import NoiseV2Pool, load_preset_bank
from experiments.noise_model import rig_sampler as RS


@pytest.fixture(scope="module")
def probe() -> RS.ModelProbe:
    return RS.ModelProbe()


@pytest.fixture(scope="module")
def anchors() -> dict:
    return RS.pinned_anchors()


@pytest.fixture(scope="module")
def structure() -> dict:
    return RS.load_fit(RS.STRUCTURE_PATH)


@pytest.fixture(scope="module")
def dregon_ref(probe: RS.ModelProbe, anchors: dict, structure: dict) -> RS.Reference:
    tol = RS.default_tolerances(structure)["dregon"]
    return RS.reference_of(anchors["dregon"], probe, ltas_tol_db=tol, name="dregon")


# ── the speed-law pin ───────────────────────────────────────────────────────


def test_pin_sets_the_short_span_contract_and_records_what_it_replaced() -> None:
    raw = RS.load_fit(RS.ANCHORS["dregon"]["cruise"])
    before = float(raw["params"]["floor"]["floor_exp"])
    pinned = RS.pin_speed_laws(raw)

    assert pinned["params"]["profile"]["amp_exp"] == RS.SPAN_PIN["amp_exp"]
    assert pinned["params"]["floor"]["floor_exp"] == RS.SPAN_PIN["floor_exp"]
    assert pinned["params"]["floor"]["floor_static_rel"] == RS.SPAN_PIN["floor_static_rel"]
    assert pinned["params"]["span_pin_record"]["floor_exp"] == pytest.approx(before)
    # the anchor on disk is untouched: two arms reading the same file cannot
    # leak one's pin into the other
    assert float(raw["params"]["floor"]["floor_exp"]) == pytest.approx(before)


def test_pin_leaves_the_cruise_spectrum_at_the_reference_speed(
    probe: RS.ModelProbe,
) -> None:
    """The pin is applied where every speed factor is 1, so it moves nothing at
    80 rev/s — that is the whole reason the reference speed is the pin point."""
    raw = RS.load_fit(RS.ANCHORS["michaels"]["cruise"])
    _, before, _ = probe.bands(raw, RS.PROBE_CRUISE_RPS)
    _, after, _ = probe.bands(RS.pin_speed_laws(raw), RS.PROBE_CRUISE_RPS)
    assert np.allclose(before, after, atol=1e-9)


# ── the guards ──────────────────────────────────────────────────────────────


def test_guard_refuses_a_rising_harmonic_profile(
    probe: RS.ModelProbe, anchors: dict, dregon_ref: RS.Reference
) -> None:
    bad = RS.strip_payload(anchors["dregon"])
    prof = np.asarray(bad["params"]["profile"]["profile_db"], dtype=np.float64)
    x = RS.order_axis(prof.shape[1])
    # flip every rotor's trend: same level, rising with order
    parts = RS.decompose_block(prof)
    bad["params"]["profile"]["profile_db"] = np.stack(
        [q.gain - q.slope * (x - x.mean()) + q.resid for q in parts]
    ).tolist()

    guards = RS.check_sample(bad, dregon_ref, probe)
    assert "trend_falls" in guards["failed"]
    assert max(guards["trend_drop_db"]) < RS.TREND_MARGIN_DB


def test_guard_refuses_a_line_width_excursion_past_the_cap(
    probe: RS.ModelProbe, anchors: dict, dregon_ref: RS.Reference
) -> None:
    bad = RS.strip_payload(anchors["dregon"])
    factor = 2.0 * RS.GAMMA_EXCURSION_CAP
    bad["params"]["gamma_hz"] = (
        factor * np.asarray(bad["params"]["gamma_hz"], dtype=np.float64)
    ).tolist()

    guards = RS.check_sample(bad, dregon_ref, probe)
    assert "gamma_excursion" in guards["failed"]
    assert guards["gamma_ratio"][0] == pytest.approx(factor, rel=1e-6)


def test_guard_refuses_a_level_far_outside_the_real_window_envelope(
    probe: RS.ModelProbe, anchors: dict, dregon_ref: RS.Reference
) -> None:
    bad = RS.strip_payload(anchors["dregon"])
    shift = 4.0 * dregon_ref.ltas_tol_db
    prof = np.asarray(bad["params"]["profile"]["profile_db"], dtype=np.float64)
    bad["params"]["profile"]["profile_db"] = (prof + shift).tolist()
    bad["params"]["floor"]["floor_mean_db"] = float(bad["params"]["floor"]["floor_mean_db"] + shift)

    guards = RS.check_sample(bad, dregon_ref, probe)
    assert "ltas" in guards["failed"]
    assert guards["ltas_rms_db"] > dregon_ref.ltas_tol_db


def test_guard_refuses_a_negative_speed_exponent(
    probe: RS.ModelProbe, anchors: dict, dregon_ref: RS.Reference
) -> None:
    bad = RS.strip_payload(anchors["dregon"])
    bad["params"]["floor"]["floor_exp"] = -3.0
    assert "speed_law" in RS.check_sample(bad, dregon_ref, probe)["failed"]


def test_the_anchor_itself_passes_every_guard(
    probe: RS.ModelProbe, anchors: dict, dregon_ref: RS.Reference
) -> None:
    guards = RS.check_sample(RS.strip_payload(anchors["dregon"]), dregon_ref, probe)
    assert guards["ok"], guards["failed"]
    assert guards["ltas_rms_db"] == pytest.approx(0.0, abs=1e-9)


# ── the draw ────────────────────────────────────────────────────────────────


def test_a_draw_is_a_function_of_its_substream_only(anchors: dict) -> None:
    a, _ = RS.draw_fit(anchors["dregon"], np.random.default_rng([7, 3]), 2.0)
    b, _ = RS.draw_fit(anchors["dregon"], np.random.default_rng([7, 3]), 2.0)
    c, _ = RS.draw_fit(anchors["dregon"], np.random.default_rng([7, 4]), 2.0)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert json.dumps(a, sort_keys=True) != json.dumps(c, sort_keys=True)


def test_a_draw_keeps_the_pinned_speed_laws(anchors: dict) -> None:
    drawn, _ = RS.draw_fit(anchors["michaels"], np.random.default_rng([1, 1]), 4.0)
    assert drawn["params"]["profile"]["amp_exp"] == RS.SPAN_PIN["amp_exp"]
    assert drawn["params"]["floor"]["floor_exp"] == RS.SPAN_PIN["floor_exp"]
    assert drawn["params"]["floor"]["floor_static_rel"] == RS.SPAN_PIN["floor_static_rel"]


def test_the_level_coordinate_moves_comb_and_floor_together(anchors: dict) -> None:
    """A level draw is a RIG level: it must not change the comb-to-floor ratio,
    which is the one thing the DREGON fit is known to be uncertain about."""
    anchor = anchors["dregon"]
    drawn, info = RS.draw_fit(anchor, np.random.default_rng([2, 2]), 0.0, level_db=3.0)
    prof_before = np.asarray(anchor["params"]["profile"]["profile_db"], dtype=np.float64)
    prof_after = np.asarray(drawn["params"]["profile"]["profile_db"], dtype=np.float64)
    assert info["level_db"] == pytest.approx(3.0)
    assert np.allclose(prof_after - prof_before, 3.0, atol=1e-4)
    assert drawn["params"]["floor"]["floor_mean_db"] - anchor["params"]["floor"][
        "floor_mean_db"
    ] == pytest.approx(3.0, abs=1e-6)


def test_zero_strength_reproduces_the_anchor(anchors: dict) -> None:
    drawn, _ = RS.draw_fit(anchors["michaels"], np.random.default_rng([3, 3]), 0.0)
    for got, want in (
        (
            drawn["params"]["profile"]["profile_db"],
            anchors["michaels"]["params"]["profile"]["profile_db"],
        ),
        (drawn["params"]["gamma_hz"], anchors["michaels"]["params"]["gamma_hz"]),
    ):
        assert np.allclose(np.asarray(got, float), np.asarray(want, float), atol=1e-4)


# ── the path ────────────────────────────────────────────────────────────────


def test_path_endpoints_reproduce_the_anchors_on_the_shared_order_range(
    anchors: dict,
) -> None:
    a, b = anchors["dregon"], anchors["michaels"]
    at0 = RS.interpolate_fits(a, b, 0.0)
    at1 = RS.interpolate_fits(a, b, 1.0)
    k = RS.PATH_K_MAX
    for point, anchor in ((at0, a), (at1, b)):
        for path in (
            ("profile", "profile_db"),
            ("gamma_hz",),
        ):
            got = np.asarray(_dig(point["params"], path), dtype=np.float64)[:, :k]
            want = np.asarray(_dig(anchor["params"], path), dtype=np.float64)[:, :k]
            assert np.allclose(got, want, rtol=1e-5, atol=1e-5)
        assert point["params"]["sigma_nu"] == pytest.approx(anchor["params"]["sigma_nu"], rel=1e-9)


def test_path_drops_the_unshared_orders_and_says_so(anchors: dict) -> None:
    point = RS.interpolate_fits(anchors["dregon"], anchors["michaels"], 0.5)
    assert np.asarray(point["params"]["profile"]["profile_db"]).shape[1] == RS.PATH_K_MAX
    assert np.asarray(point["params"]["gamma_hz"]).shape[1] == RS.PATH_K_MAX
    assert point["_path"]["orders_dropped"] == {"a": 88 - RS.PATH_K_MAX, "b": 0}
    assert point["k_max"] == RS.PATH_K_MAX


def test_path_keeps_a_falling_trend_at_every_mixing_coordinate(anchors: dict) -> None:
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        point = RS.interpolate_fits(anchors["dregon"], anchors["michaels"], t)
        drops = [q.drop for q in RS.decompose_block(point["params"]["profile"]["profile_db"])]
        assert min(drops) >= RS.TREND_MARGIN_DB, (t, drops)


def _dig(params: dict, path: tuple[str, ...]):
    out = params
    for key in path:
        out = out[key]
    return out


# ── the bank ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def tiny_spec(structure: dict) -> RS.BankSpec:
    return RS.BankSpec(
        preset="easy",
        n=2,
        seed=RS.__dict__.get("SEED", 20260921),
        strength=2.0,
        ltas_tol_db=RS.default_tolerances(structure),
        widths=RS.WIDTHS.as_dict(),
    )


def test_bank_bytes_are_reproducible_and_the_digest_covers_the_code(
    tiny_spec: RS.BankSpec, tmp_path
) -> None:
    first, stats_a = RS.build_entries(tiny_spec, workers=1)
    second, stats_b = RS.build_entries(tiny_spec, workers=2)
    payload_a = RS.bank_payload(first, tiny_spec, stats_a)
    payload_b = RS.bank_payload(second, tiny_spec, stats_b)
    text_a = json.dumps(payload_a, separators=(",", ":"))
    text_b = json.dumps(payload_b, separators=(",", ":"))
    assert text_a == text_b, "a bank must not depend on the worker count or the wall clock"

    # the skip digest is content-addressed over the sampler's own source, so a
    # code change invalidates an existing bank
    assert "src/experiments/noise_model/rig_sampler.py" in RS.code_digest()
    moved = RS.BankSpec(
        preset=tiny_spec.preset,
        n=tiny_spec.n,
        seed=tiny_spec.seed,
        strength=tiny_spec.strength + 0.5,
        ltas_tol_db=tiny_spec.ltas_tol_db,
        widths=tiny_spec.widths,
    )
    assert moved.digest != tiny_spec.digest


def test_bank_entries_stream_through_the_pool(tiny_spec: RS.BankSpec, tmp_path) -> None:
    entries, stats = RS.build_entries(tiny_spec, workers=1)
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(RS.bank_payload(entries, tiny_spec, stats)))

    loaded = load_preset_bank(path)
    assert [e.traj_rig for e in loaded] == ["dregon", "michaels"]
    assert loaded[0].standby is None and loaded[1].standby is not None

    pool = NoiseV2Pool(
        sample_rate=16000,
        duration_s=0.5,
        n_mics=8,
        n_rotors=4,
        entries=loaded,
        rps_kind="full_flight",
        seed=0,
    )
    audio, rps, entry = pool.render(np.random.default_rng(0), 0.5)
    assert audio.shape == (8, 8000)
    assert rps.shape[0] == 4
    assert np.all(np.isfinite(audio)) and float(np.abs(audio).max()) > 0.0
    assert entry.name.startswith(("dregon_", "michaels_"))


def test_hard_entries_carry_no_traj_rig_and_a_recorded_mixing_coordinate(
    structure: dict,
) -> None:
    spec = RS.BankSpec(
        preset="hard",
        n=4,
        seed=20260921,
        strength=2.0,
        ltas_tol_db=RS.default_tolerances(structure),
        widths=RS.WIDTHS.as_dict(),
    )
    entries, stats = RS.build_entries(spec, workers=2)
    assert all(e["traj_rig"] is None for e in entries)
    assert all(0.0 <= e["provenance"]["t"] <= 1.0 for e in entries)
    assert all(
        np.asarray(e["cruise"]["params"]["profile"]["profile_db"]).shape[1] == RS.PATH_K_MAX
        for e in entries
    )
    # the standby SLOT is a policy, not an interpolation: an entry either
    # carries Michael's standby payload or none at all
    for entry in entries:
        assert entry["standby"] is None or "params" in entry["standby"]
    assert "ks_uniform" in stats


def test_ks_uniform_matches_scipy_and_its_threshold_catches_a_skewed_sample() -> None:
    """Differential: the realised-``t`` statistic the bank records is the
    textbook one, pinned against ``scipy.stats.kstest``."""
    from scipy import stats as sps

    for seed in range(4):
        sample = np.random.default_rng(seed).uniform(size=512)
        got = RS.ks_uniform(sample)
        assert got["ks"] == pytest.approx(float(sps.kstest(sample, "uniform").statistic))
    skewed = np.random.default_rng(0).uniform(size=512) ** 2
    assert not RS.ks_uniform(skewed)["passes"]
    assert RS.ks_uniform(np.random.default_rng(2).uniform(size=512))["passes"]


def test_coverage_counts_a_bracketed_band_and_refuses_an_unbracketed_one() -> None:
    centres = np.array([100.0, 400.0, 1000.0])
    cloud = np.array([[0.0, -2.0, 5.0], [0.0, 2.0, 7.0]])
    real = np.array([[0.0, 0.0, 6.0], [0.0, 9.0, 6.0]])
    cov = RS.coverage(cloud, real, centres)
    # above 300 Hz there are two bands per clip: clip 0 is inside both, clip 1
    # is outside the 400 Hz one
    assert cov["above_300hz"] == pytest.approx(0.75)
    assert cov["above_300hz_worst_clip"] == pytest.approx(0.5)
