"""The round-score runner's two bars, at their boundaries.

The frozen gate functions live in ``experiments.noise_model.gates`` and are
tested next door; what is tested here is the runner's own reporting contract:
every HPPNet number is read against the LEGACY PARITY bar (the previous best,
which decides the round) and against the frozen 0.70-gap DREGON STRETCH target,
and an incomplete cohort is never a pass on either.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from experiments.noise_model import gates as G

RUNNER = Path(__file__).resolve().parents[2] / "scripts" / "noise_v2_round_score.py"


def _runner() -> Any:
    spec = importlib.util.spec_from_file_location("noise_v2_round_score", RUNNER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # the module's dataclasses resolve through it
    spec.loader.exec_module(mod)
    return mod


RS = _runner()


def dregon(value: float) -> dict[str, float]:
    return {s.key: float(value) for s in G.DREGON_CRUISE_SUPPORTS}


def michaels(value: float) -> dict[str, float]:
    return {s.key: float(value) for s in G.MICHAELS_SUPPORTS}


def bars_of(pred: dict[str, float]) -> dict[str, Any]:
    return RS.bars(G.hppnet_gate(pred, {}))


def test_parity_admits_a_candidate_the_stretch_target_rejects() -> None:
    between = (G.DREGON_PIT_TARGET + G.DREGON_BASELINE_PIT_MAE) / 2
    b = bars_of(dregon(between) | michaels(3.0))
    assert b["dregon"]["parity"]["within"] is True
    assert b["dregon"]["stretch"]["within"] is False
    assert b["parity_pass"] is True
    assert b["stretch_pass"] is False


def test_both_bars_need_michaels_parity_too() -> None:
    over = G.MICHAELS_PIT_BOUND + 1e-6
    b = bars_of(dregon(G.DREGON_PIT_TARGET) | michaels(over))
    assert b["dregon"]["stretch"]["within"] is True
    assert b["michaels"]["parity"]["within"] is False
    assert b["parity_pass"] is False
    assert b["stretch_pass"] is False


def test_stretch_pass_implies_parity_pass() -> None:
    b = bars_of(dregon(G.DREGON_PIT_TARGET) | michaels(3.0))
    assert b["stretch_pass"] is True
    assert b["parity_pass"] is True


@pytest.mark.parametrize("bar", ["parity", "stretch"])
def test_an_incomplete_dregon_cohort_is_never_a_pass(bar: str) -> None:
    pred = dregon(0.5) | michaels(3.0)
    del pred[G.DREGON_CRUISE_SUPPORTS[0].key]
    b = bars_of(pred)
    assert b["dregon"]["cohort_complete"] is False
    assert b["dregon"][bar]["within"] is None
    assert b["dregon"]["mean_rev_s"] is None
    assert b["parity_pass"] is False and b["stretch_pass"] is False


def test_michaels_has_no_stretch_bar_and_says_so() -> None:
    b = bars_of(dregon(G.DREGON_PIT_TARGET) | michaels(3.0))
    assert b["michaels"]["stretch"]["bar_rev_s"] is None
    assert b["michaels"]["stretch"]["within"] is None
    assert "parity bar" in b["michaels"]["stretch"]["note"]


def test_bars_without_a_probe_report_the_reason_and_fail() -> None:
    b = RS.bars(dict(name="hppnet_pit_mae", unavailable="no probe run", **{"pass": False}))
    assert b["unavailable"] == "no probe run"
    assert b["dregon"] is None and b["michaels"] is None
    assert b["parity_pass"] is False and b["stretch_pass"] is False


def _fit_json(path: Path, schema: str, support: str) -> None:
    path.write_text(json.dumps(dict(schema=schema, support=support, params={})))


def test_read_fits_takes_both_fit_schemas_and_prefers_the_current_one(tmp_path: Path) -> None:
    # one support fitted under both parameterisations, in both filename orders:
    # the round is scored on the CURRENT schema whichever way the glob sorts
    _fit_json(tmp_path / "a_michaels_v2.json", "noise-v2-fit/2", "michaels_fly125_all")
    _fit_json(tmp_path / "z_michaels_v1.json", "noise-v2-fit/1", "michaels_fly125_all")
    _fit_json(tmp_path / "a_dregon_v1.json", "noise-v2-fit/1", "dregon_room2_floor")
    _fit_json(tmp_path / "z_dregon_v2.json", "noise-v2-fit/2", "dregon_room2_floor")
    _fit_json(tmp_path / "cruise.json", "noise-v2-fit/1", "michaels_fly125_cruise")
    _fit_json(tmp_path / "alien.json", "some-other-record/1", "michaels_fly125_all")

    fits = RS.read_fits(tmp_path)

    assert set(fits) == {"michaels_fly125_all", "michaels_fly125_cruise", "dregon_room2_floor"}
    assert fits["michaels_fly125_all"]["_path"].endswith("a_michaels_v2.json")
    assert fits["dregon_room2_floor"]["_path"].endswith("z_dregon_v2.json")
    assert fits["michaels_fly125_cruise"]["schema"] == "noise-v2-fit/1"


def test_read_fits_rejects_a_directory_with_no_readable_fit(tmp_path: Path) -> None:
    _fit_json(tmp_path / "alien.json", "some-other-record/1", "michaels_fly125_all")
    with pytest.raises(SystemExit):
        RS.read_fits(tmp_path)
