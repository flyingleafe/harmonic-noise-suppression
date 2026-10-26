"""The neighbourhood bank (:mod:`data_processing.noise_model.nbhd_bank_v4`).

A bank stores each base fit once and, per draw, only the fields a perturbation
moves. A draw read back must be the written payload exactly, single-regime and
standby/cruise alike, with its base's trajectory rig; the pool's comb offset
moves the aerodynamic and motor tables and nothing else; a draw that moved a
field outside the patch cannot be stored, since the bank could not give it back.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from data_processing.noise_model.nbhd_bank_v4 import NbhdBankV4, write_nbhd_bank_v4
from data_processing.noise_v2_pool import load_preset_bank

ROOT = Path(__file__).resolve().parents[2]
FLIGHT = ROOT / "results/noise_v4/flight/restarts"
STANDBY = FLIGHT / "md2_fly125_standby_fpb__flight_v4__s0__r6c1.json"
CRUISE = FLIGHT / "md2_fly125_cruise_fpb__flight_v4__s0__r6c3.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _perturbed(fit: dict[str, Any], shift: float) -> dict[str, Any]:
    """A stand-in perturbation: every patch field moved, the derived fields dropped."""
    q = copy.deepcopy(fit)
    p = q["params"]
    p.pop("hierarchy", None)
    p["profile_aero_db"] = (np.asarray(p["profile_aero_db"]) + shift).tolist()
    p["profile_motor_db"] = (np.asarray(p["profile_motor_db"]) - shift).tolist()
    p["alpha_aero"] = float(p["alpha_aero"]) + 0.1 * shift
    p["floor"] = dict(p["floor"], b0=float(p["floor"]["b0"]) + 0.01 * shift)
    p["floor"]["shape_z"] = (np.asarray(p["floor"]["shape_z"]) + shift).tolist()
    p["floor"].pop("ctrl_db", None)
    ped = dict(p["pedestal"])
    ped["width_phase_hz"] = (np.asarray(ped["width_phase_hz"]) * (1.0 + 0.1 * shift)).tolist()
    p["pedestal"] = ped
    q["perturbation"] = {"rule": "test", "shift": shift}
    return q


@pytest.fixture(scope="module")
def bank(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[dict[str, Any]]]:
    standby, cruise = _load(STANDBY), _load(CRUISE)
    bases = [
        {"name": "solo", "traj_rig": "dregon", "regimes": {"single": cruise}},
        {"name": "pair", "traj_rig": "michaels", "regimes": {"standby": standby, "cruise": cruise}},
    ]
    draws = []
    for i, shift in enumerate((1.0, -2.0, 0.5)):
        draws.append(
            {
                "base": 1,
                "name": f"pair_{i}",
                "regimes": {
                    "standby": _perturbed(standby, shift),
                    "cruise": _perturbed(cruise, shift),
                },
            }
        )
        draws.append(
            {"base": 0, "name": f"solo_{i}", "regimes": {"single": _perturbed(cruise, shift)}}
        )
    path = write_nbhd_bank_v4(
        tmp_path_factory.mktemp("nb") / "b.npz", bases, draws, {"note": "test"}
    )
    return path, draws


def test_a_draw_reads_back_as_written(bank: tuple[Path, list[dict[str, Any]]]) -> None:
    path, draws = bank
    b = load_preset_bank(path)
    assert isinstance(b, NbhdBankV4) and len(b) == len(draws)
    assert b.traj_rigs == ("dregon", "michaels")
    for i, d in enumerate(draws):
        e = b[i]
        assert e.name == d["name"]
        assert e.traj_rig == ("michaels" if d["base"] == 1 else "dregon")
        want = d["regimes"]
        got = (
            {"single": e.cruise}
            if e.standby is None
            else {"standby": e.standby, "cruise": e.cruise}
        )
        assert set(got) == set(want)
        for regime, payload in want.items():
            assert json.dumps(got[regime], sort_keys=True) == json.dumps(payload, sort_keys=True)
    assert [e.name for e in b.probe()] == ["pair_0", "solo_0"]


def test_the_comb_offset_moves_the_line_tables_only(
    bank: tuple[Path, list[dict[str, Any]]],
) -> None:
    path, _ = bank
    plain, lifted = NbhdBankV4(path)[0].cruise, NbhdBankV4(path).with_offset(6.0)[0].cruise
    for key in ("profile_aero_db", "profile_motor_db"):
        np.testing.assert_allclose(
            np.asarray(lifted["params"][key]) - np.asarray(plain["params"][key]), 6.0
        )
    for key in ("floor", "pedestal", "shaft", "alpha_aero"):
        assert lifted["params"][key] == plain["params"][key]


def test_a_draw_that_moves_a_field_outside_the_patch_is_refused(tmp_path: Path) -> None:
    cruise = _load(CRUISE)
    bad = _perturbed(cruise, 1.0)
    bad["params"]["shaft"] = dict(bad["params"]["shaft"], eps=0.5)
    with pytest.raises(ValueError, match="shaft.eps"):
        write_nbhd_bank_v4(
            tmp_path / "b.npz",
            [{"name": "solo", "traj_rig": None, "regimes": {"single": cruise}}],
            [{"base": 0, "name": "bad", "regimes": {"single": bad}}],
            {},
        )
