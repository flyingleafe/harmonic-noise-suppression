"""Rigs drawn from a fit's recorded prior (:mod:`experiments.noise_model.rig_prior`)."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from experiments.noise_model import rig_prior as RP

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/noise_v3/fits_r4/michaels_fly125_cruise__flight_v3.json"


@dataclass
class _Verdict:
    passed: bool
    reason: str


@pytest.fixture(scope="module")
def base() -> dict:
    return json.loads(BASE.read_text())


def test_a_draw_is_its_seed_and_leaves_the_base_fit_alone(base):
    before = copy.deepcopy(base)
    a = RP.draw_rig(base, np.random.default_rng([3, 1]))
    b = RP.draw_rig(base, np.random.default_rng([3, 1]))
    c = RP.draw_rig(base, np.random.default_rng([3, 2]))
    assert a == b and a != c
    assert base == before
    # measured quantities are the base fit's
    assert a["params"]["lam"] == base["params"]["lam"]
    assert a["diagnostics"]["measured"] == base["diagnostics"]["measured"]


def test_a_speed_law_that_falls_with_speed_is_redrawn_not_clipped(base):
    leaning = copy.deepcopy(base)
    leaning["priors"]["amp_exp"] = [-1.0, 1.0]  # ~84 % of raw draws negative
    assert "amp_exp" not in (leaning["diagnostics"]["span_pins"].get("pinned") or [])
    draws = [
        RP.draw_rig(leaning, np.random.default_rng([5, i]))["params"]["profile"]["amp_exp"]
        for i in range(40)
    ]
    assert min(draws) > 0.0
    assert len(set(draws)) == 40  # a draw, not a clip to one value


def test_the_sampler_returns_n_rigs_or_raises_with_its_draws(base):
    passing = lambda fit: _Verdict(True, "ok")  # noqa: E731
    accepted, rejected = RP.sample_rigs(base, n=3, seed=0, accept=passing, max_attempts=5)
    assert [d.attempt for d in accepted] == [0, 1, 2] and rejected == []
    failing = lambda fit: _Verdict(False, "rotor 2 passed 1 of 20 (needs 18)")  # noqa: E731
    with pytest.raises(RP.SamplingExhausted) as info:
        RP.sample_rigs(base, n=2, seed=0, accept=failing, max_attempts=4)
    assert len(info.value.rejected) == 4 and info.value.accepted == []
    assert "rotor 2 passed 1 of 20 x4" in str(info.value)
