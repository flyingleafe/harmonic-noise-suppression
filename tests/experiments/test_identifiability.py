"""The per-rotor identifiability gate (:mod:`experiments.noise_model.identifiability`)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.noise_model import identifiability as ID

ROOT = Path(__file__).resolve().parents[2]
CRUISE = ROOT / "results/noise_v3/fits_r4/michaels_fly125_cruise__flight_v3.json"
N_SAMPLES = ID.SP.FLIGHT_N_FFT + 11 * ID.SP.FLIGHT_HOP  # twelve frames


@pytest.fixture(scope="module")
def cruise() -> dict[str, Any]:
    return json.loads(CRUISE.read_text())


def _moved(fit: dict[str, Any], db: float) -> dict[str, Any]:
    out = copy.deepcopy(fit)
    prof = np.asarray(out["params"]["profile"]["profile_db"], dtype=np.float64)
    out["params"]["profile"]["profile_db"] = (prof + db).tolist()
    return out


def _pool(speeds: list[float], *, ramp: float = 0.0) -> ID.FramePool:
    t = np.arange(N_SAMPLES) / ID.SP.FLIGHT_SR
    rps = np.asarray(speeds, dtype=np.float64)[:, None] + ramp * t[None, :]
    return ID.frame_pool([rps], n_frames=12, seed=0)


SEPARATED = [70.0, 76.0, 83.0, 90.0]


def test_a_loud_comb_passes_and_a_sunk_comb_fails(cruise):
    pool = _pool(SEPARATED)
    loud = ID.gate(_moved(cruise, 30.0), pool)
    assert loud.passed, loud.reason
    assert loud.frames_ok.tolist() == [12, 12, 12, 12]
    sunk = ID.gate(_moved(cruise, -100.0), pool)
    assert not sunk.passed
    assert sunk.reason.startswith("rotor 1 failed")


def test_the_gate_never_passes_vacuously(cruise):
    loud = _moved(cruise, 30.0)
    together = ID.gate(loud, _pool([80.0, 80.0, 80.0, 80.0]))
    assert not together.passed and together.reason.startswith("untestable")
    assert together.close.tolist() == [12, 12, 12, 12]
    stopped = ID.gate(loud, _pool([0.0, 0.0, 0.0, 0.0]))
    assert not stopped.passed and stopped.reason.startswith("untestable")
    assert stopped.stopped.tolist() == [12, 12, 12, 12]


def test_rotors_closer_than_the_separation_are_excused_not_failed(cruise):
    pool = _pool([80.0, 80.8, 72.0, 88.0])
    loud = _moved(cruise, 30.0)
    strict = ID.gate(loud, pool)
    assert not strict.passed and strict.reason.startswith("untestable: rotor 1")
    lax = ID.gate(loud, pool, ID.GateConfig(min_eligible_frames=0))
    assert lax.passed, lax.reason
    assert lax.eligible.tolist() == [0, 0, 12, 12]
    assert lax.frames_ok.tolist() == [0, 0, 12, 12]


def test_progressive_orders_count_what_one_full_pass_counts(cruise):
    """Pass 1's partial background is only trusted below the margin, so the
    two-pass count equals the all-orders count, on a ramp (chirped frames) and
    at the fit's own levels, where some rotors pass and some do not."""
    pool = _pool([55.0, 64.0, 81.0, 97.0], ramp=6.0)
    counts = {}
    for first_k in (8, 24, 10_000):
        cfg = ID.GateConfig(first_k=first_k, pass_frac=0.0, min_eligible_frames=0)
        counts[first_k] = ID.gate(cruise, pool, cfg).frames_ok.tolist()
    assert counts[8] == counts[24] == counts[10_000]
