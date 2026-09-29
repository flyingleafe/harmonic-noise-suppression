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
    assert sunk.reason.startswith("rotor 1 passed 0 of 12")


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


def _exact_contrasts(fit, pool: ID.FramePool) -> list[np.ndarray]:
    out = []
    for w, st in zip(pool.windows, pool.starts, strict=True):
        dec = ID.decompose(fit, w, st)
        n_r, n, kk, _ = dec.lines.shape
        hz = dec.centre_rps[:, :, None] * np.arange(1, kk + 1)
        b = np.clip(np.rint(hz * pool.n_fft / pool.sr).astype(int), 0, pool.n_fft // 2)
        ri, ni = np.meshgrid(np.arange(n_r), np.arange(n), indexing="ij")
        line = dec.lines[ri[..., None], ni[..., None], np.arange(kk), b]
        s = dec.power[ni[..., None], b]
        out.append(10.0 * np.log10(s / np.maximum(s - line, 1e-300)))
    return out


def test_fast_contrasts_are_the_exact_model_at_constant_speed(cruise):
    """At constant speed the fast scheme is the forward model's own line kernel
    summed over every line, so it must reproduce the exact per-line contrast
    (:func:`decompose`) up to the shape table's interpolation."""
    pool = _pool([55.0, 64.0, 81.0, 118.0])  # 118 rev/s: the order cap bites below 81
    cfg = ID.GateConfig()
    (fast, hz), exact = ID.line_contrasts(cruise, pool, cfg)[0], _exact_contrasts(cruise, pool)[0]
    kk = min(fast.shape[2], exact.shape[2])
    band = (hz[:, :, :kk] >= cfg.f_min_hz) & (hz[:, :, :kk] <= cfg.f_max_hz)
    f, e = fast[:, :, :kk][band], exact[:, :, :kk][band]
    relevant = (f > 0.5) | (e > 0.5)
    assert relevant.sum() > 100
    assert np.max(np.abs(f - e)[relevant]) < 0.2


def test_the_exact_reference_is_the_render_expectation_with_the_array_response():
    """:func:`decompose`'s spectrum is the mic mean of
    :func:`render.expected_periodogram_regimes`, array response included."""
    from experiments.noise_model import render as RD

    def load(name: str) -> dict:
        return json.loads((ROOT / f"results/noise_v3/fits_r4/{name}__flight_v3.json").read_text())

    pair = {"standby": load("michaels_fly125_standby"), "cruise": load("michaels_fly125_cruise")}
    assert pair["cruise"]["params"].get("array_response") is not None
    t = np.arange(N_SAMPLES) / ID.SP.FLIGHT_SR
    rps = np.array([40.0, 50.0, 58.0, 70.0])[:, None] + 8.0 * t[None, :]  # through the blend
    starts = np.arange(1 + (N_SAMPLES - ID.SP.FLIGHT_N_FFT) // ID.SP.FLIGHT_HOP) * ID.SP.FLIGHT_HOP
    ref = RD.expected_periodogram_regimes(pair, rps, n_mics=8).mean(axis=0)
    got = ID.decompose(pair, rps, starts).power
    assert np.max(np.abs(10.0 * np.log10(got / ref))) < 1e-9
