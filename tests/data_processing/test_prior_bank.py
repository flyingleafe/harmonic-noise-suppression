"""The compact prior bank: round trip, laziness, and the pool reading it."""

from __future__ import annotations

import numpy as np
import pytest

from data_processing.noise_model.prior_bank import PriorBank, write_prior_bank
from data_processing.noise_v2_pool import NoiseV2Pool, load_preset_bank
from experiments.noise_model.drone_prior import sample_prior


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    rng = np.random.default_rng(0)
    rigs = [sample_prior(rng) for _ in range(3)]
    for i, r in enumerate(rigs):
        r["_prior"]["gate"] = dict(frames_ok=[5, 4, 6, 7], eligible=[6, 5, 6, 7], threshold_db=3.0)
        r["_prior"]["seed"] = 100 + i
    path = write_prior_bank(tmp_path_factory.mktemp("bank") / "b.npz", rigs, {"note": "test"})
    return rigs, path


def test_round_trip_keeps_every_payload_within_float16(bank):
    rigs, path = bank
    b = PriorBank(path)
    assert len(b) == 3 and b.header == {"note": "test"}
    for i, r in enumerate(rigs):
        q = b.payload(i)
        p, pq = r["params"], q["params"]
        assert q["schema"] == r["schema"] and q["front_end"] == r["front_end"]
        assert np.allclose(pq["profile"]["profile_db"], p["profile"]["profile_db"], atol=0.07)
        assert np.allclose(pq["am"]["sigma2"], p["am"]["sigma2"], rtol=2e-3)
        assert np.allclose(pq["am"]["gamma_hz"], p["am"]["gamma_hz"], rtol=2e-3)
        assert np.allclose(pq["floor"]["floor_shape_z"], p["floor"]["floor_shape_z"], rtol=1e-6)
        assert pq["sigma_nu"] == pytest.approx(p["sigma_nu"], rel=1e-6)
        assert pq["wander"] == p["wander"] and pq["wind_sc"] == p["wind_sc"]
        assert q["_prior"]["gate"]["frames_ok"] == [5, 4, 6, 7] and q["_prior"]["seed"] == 100 + i
        assert q["_prior"]["drawn"]["floor_template"] == r["_prior"]["drawn"]["floor_template"]


def test_the_sequence_is_lazy_and_the_pool_renders_from_it(bank):
    _, path = bank
    entries = load_preset_bank(path)
    assert isinstance(entries, PriorBank) and len(entries) == 3
    assert entries[1].name == "prior_00001" and entries[-1].name == "prior_00002"
    assert entries[1].traj_rig is None and entries[1].standby is None
    cfg = dict(
        preset_bank=str(path),
        n_mics=2,
        n_rotors=4,
        comb_offset_db=6.0,
        rps=dict(kind="fitted_traj", fits="dload:rps-traj-fits", rigs={"posterior": 1.0}),
    )
    pool = NoiseV2Pool.from_config(cfg, duration_s=0.5, sample_rate=16000)
    assert isinstance(pool.entries, PriorBank)
    off = np.asarray(pool.entries[0].cruise["params"]["profile"]["profile_db"])
    raw = np.asarray(PriorBank(path).payload(0)["params"]["profile"]["profile_db"])
    assert np.allclose(off - raw, 6.0, atol=1e-5)
    frame = pool.sample_timeframe(np.random.default_rng(1), 0.5)
    assert np.asarray(frame["audio"].data).shape == (2, 8000)
    assert dict(frame["meta"].items())["noise_v2_entry"].startswith("prior_")
