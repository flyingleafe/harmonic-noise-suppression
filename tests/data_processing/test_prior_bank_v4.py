"""The compact v4 prior bank: round trip, format dispatch, and the pool reading it."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from data_processing.noise_model.prior_bank import PriorBank, write_prior_bank
from data_processing.noise_model.prior_bank_v4 import PriorBankV4, write_prior_bank_v4
from data_processing.noise_model.v4 import WIND_OFF_DB
from data_processing.noise_v2_pool import NoiseV2Pool, load_preset_bank
from experiments.noise_model.drone_prior import sample_prior
from experiments.noise_model.prior_v4 import SEED, convert_prior_v1

SHAPE: dict[str, Any] = {
    "freqs_hz": np.fft.rfftfreq(4096, 1.0 / 16000).tolist(),
    "psd": np.ones(2049).tolist(),
    "ref_band_hz": [20.0, 100.0],
    "unit_psd_ref": 1.0,
}
RPS = {"kind": "fitted_traj", "fits": "dload:rps-traj-fits", "rigs": {"posterior": 1.0}}


@pytest.fixture(scope="module")
def banks(tmp_path_factory) -> tuple[Any, list[dict[str, Any]], Any]:
    """A v1 bank of three draws and its v4 conversion: rig 0 without wind,
    rig 1 with one capsule off, rig 2 as drawn."""
    tmp = tmp_path_factory.mktemp("banks")
    rng = np.random.default_rng(0)
    v1 = write_prior_bank(tmp / "v1.npz", [sample_prior(rng) for _ in range(3)], {"note": "v1"})
    payloads = [convert_prior_v1(PriorBank(v1), i, shape=SHAPE, seed=SEED) for i in range(3)]
    payloads[0]["params"]["wind"] = None
    wind = payloads[1]["params"]["wind"] or copy.deepcopy(payloads[2]["params"]["wind"])
    assert wind is not None, "the fixture needs one windy draw"
    wind["level_db"] = [WIND_OFF_DB, *[-30.0 - m for m in range(7)]]
    payloads[1]["params"]["wind"] = wind
    v4 = write_prior_bank_v4(tmp / "v4.npz", payloads, {"note": "v4"})
    return v1, payloads, v4


def test_round_trip_keeps_every_payload_within_float16(banks) -> None:
    _, payloads, path = banks
    b = PriorBankV4(path)
    assert len(b) == 3 and b.header == {"note": "v4"}
    for i, pl in enumerate(payloads):
        q = b.payload(i)
        p, pq = pl["params"], q["params"]
        top = [k for k in pl if k != "params"]
        assert {k: q[k] for k in top} == {k: pl[k] for k in top}  # meta + _prior_v4
        assert pq["motor_orders"] == p["motor_orders"]
        np.testing.assert_allclose(pq["profile_aero_db"], p["profile_aero_db"], atol=0.07)
        np.testing.assert_allclose(pq["profile_motor_db"], p["profile_motor_db"], atol=0.07)
        for key in p["pedestal"]:
            np.testing.assert_allclose(
                pq["pedestal"][key], p["pedestal"][key], rtol=1e-3, atol=1e-7
            )
        for block in ("shaft", "floor"):
            assert pq[block].keys() == p[block].keys()
            for key, v in p[block].items():
                if isinstance(v, str):
                    assert pq[block][key] == v
                else:
                    np.testing.assert_allclose(pq[block][key], v, rtol=1e-6)
        assert pq["mic_response"] == p["mic_response"]
        assert pq["alpha_aero"] == pytest.approx(p["alpha_aero"], rel=1e-6)
        assert pq["alpha_motor"] == pytest.approx(p["alpha_motor"], rel=1e-6)
        if p["wind"] is None:
            assert pq["wind"] is None
            continue
        assert {k: v for k, v in pq["wind"].items() if k != "level_db"} == {
            k: v for k, v in p["wind"].items() if k != "level_db"
        }
        np.testing.assert_allclose(pq["wind"]["level_db"], p["wind"]["level_db"], atol=0.05)
    assert b.payload(1)["params"]["wind"]["level_db"][0] == WIND_OFF_DB


def test_a_rig_off_the_shared_part_is_refused(banks, tmp_path) -> None:
    _, payloads, _ = banks
    odd = copy.deepcopy(payloads[2])
    odd["band_hz"] = [20.0, 7900.0]
    with pytest.raises(ValueError, match="rig 1"):
        write_prior_bank_v4(tmp_path / "x.npz", [payloads[2], odd], {})


def test_the_bank_is_read_on_its_format_and_the_pool_renders_from_it(banks) -> None:
    v1, _, v4 = banks
    assert isinstance(load_preset_bank(v1), PriorBank)
    entries = load_preset_bank(v4)
    assert isinstance(entries, PriorBankV4) and len(entries) == 3
    assert entries[-1].name == "prior_v4_00002" and entries[1].traj_rig is None
    cfg = dict(
        preset_bank=str(v4),
        n_mics=2,
        n_rotors=4,
        comb_offset_db=6.0,
        line_scatter_db=3.4,
        rps=RPS,
    )
    pool = NoiseV2Pool.from_config(cfg, duration_s=0.5, sample_rate=16000)
    assert isinstance(pool.entries, PriorBankV4)
    raw = PriorBankV4(v4).payload(2)["params"]
    off = pool.entries[2].cruise["params"]
    for key in ("profile_aero_db", "profile_motor_db"):
        np.testing.assert_allclose(np.subtract(off[key], raw[key]), 6.0, atol=1e-9)
    frame = pool.sample_timeframe(np.random.default_rng(1), 0.5)
    audio = np.asarray(frame["audio"].data)
    assert audio.shape == (2, 8000) and np.all(np.isfinite(audio)) and np.std(audio) > 0.0
    assert dict(frame["meta"].items())["noise_v2_entry"].startswith("prior_v4_")

    # a v3 rig has no such option: refused when the pool is built
    with pytest.raises(ValueError, match="line_scatter_db"):
        NoiseV2Pool.from_config(dict(cfg, preset_bank=str(v1)), duration_s=0.5, sample_rate=16000)


@pytest.mark.parametrize(
    ("rule", "moving"),
    # rotor speeds (2, 0, 0, 0) f_ref: one floor on sbar = 0.5, or the rotor mean of s^b0
    [("one_floor_mean_speed", lambda b0: 0.5**b0), ("per_rotor_sum", lambda b0: 2.0**b0 / 4.0)],
)
def test_flight_level_puts_the_v4_floor_envelope_back(banks, rule, moving) -> None:
    _, _, v4 = banks
    pool = NoiseV2Pool(
        entries=PriorBankV4(v4), n_mics=2, n_rotors=4, normalize_rms=0.1, level_mode="flight"
    )
    entry = pool.entries[0]
    entry.cruise["params"]["floor"]["floor_rule"] = rule
    floor = entry.cruise["params"]["floor"]
    rps = np.zeros((4, 100))
    rps[0] = 2.0 * entry.cruise["f_ref_rev_s"]
    out = pool._apply_level(np.full((2, 100), 3.0), rps, entry, 0.1)
    want = 0.1 * np.sqrt(moving(floor["b0"]) + floor["static_rel"])
    np.testing.assert_allclose(out, want, rtol=1e-12)
