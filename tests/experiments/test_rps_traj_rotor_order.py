"""Rotor order and the ground/idle level — the two per-rig conventions every
campaign statistic silently depends on.

A wrong rotor permutation does not crash anything: it just relabels the mixer
columns, so the collective/yaw split and the diagonal-vs-adjacent correlation
contrast (the model's fit targets) come out wrong on that rig alone. These
tests pin the permutation table's arithmetic and the inference rule's one
guarantee, offline and without touching dload.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.rps_traj.data import (
    DIAGONAL_PAIRS,
    NUM_ROTORS,
    RIG_SOURCES,
    RIGS,
    ROTOR_TO_MIXER,
    Flight,
    ground_level,
    infer_rotor_to_mixer,
)


def _flight(rps: np.ndarray, *, rig: str = "test", fs: float = 100.0) -> Flight:
    return Flight(rig=rig, flight="f", fs=fs, t0=0.0, rps=rps, source="test:rps")


def test_every_campaign_rig_is_loadable_and_its_permutation_is_a_permutation():
    for rig in RIGS:
        assert rig in RIG_SOURCES, f"{rig} has no frames dataset"
    for rig, perm in ROTOR_TO_MIXER.items():
        assert sorted(perm) == list(range(NUM_ROTORS)), f"{rig}: {perm} is not a permutation"
    # blackbird_quad is the one rig with an undocumented layout — it must stay
    # out of the table so load_rig infers it instead of trusting a guess.
    assert "blackbird_quad" not in ROTOR_TO_MIXER


def _coupled(pairing: tuple[tuple[int, int], tuple[int, int]], seed: int = 0) -> np.ndarray:
    """(4, T) rev/s whose only strong coupling is within ``pairing``."""
    rng = np.random.default_rng(seed)
    n = 20_000
    out = np.empty((NUM_ROTORS, n))
    for pair in pairing:
        shared = rng.standard_normal(n)
        for rotor in pair:
            out[rotor] = 90.0 + 5.0 * shared + 1.0 * rng.standard_normal(n)
    return out


def test_infer_rotor_to_mixer_recovers_the_diagonal_pairing():
    for pairing in (((0, 2), (1, 3)), ((0, 1), (2, 3)), ((0, 3), (1, 2))):
        perm = infer_rotor_to_mixer(_coupled(pairing))
        found = {frozenset(perm[i] for i in slots) for slots in DIAGONAL_PAIRS}
        assert found == {frozenset(pair) for pair in pairing}, f"{pairing} -> {perm}"


def test_infer_rotor_to_mixer_keeps_the_published_order_without_evidence():
    """Uncoupled rotors give every pairing the same score up to sampling
    noise — with no diagonal signature to read, the publisher's order must
    stand rather than being shuffled by noise."""
    rng = np.random.default_rng(1)
    assert infer_rotor_to_mixer(90.0 + rng.standard_normal((NUM_ROTORS, 20_000))) == (0, 1, 2, 3)


def test_infer_rotor_to_mixer_rejects_a_non_quad():
    with pytest.raises(ValueError):
        infer_rotor_to_mixer(np.zeros((3, 100)))


def _ground_then_flight(idle: float, cruise: float, fs: float = 100.0) -> np.ndarray:
    idle_n, ramp_n, cruise_n = int(20 * fs), int(3 * fs), int(60 * fs)
    ramp = np.linspace(idle, cruise, ramp_n)
    track = np.concatenate([np.full(idle_n, idle), ramp, np.full(cruise_n, cruise)])
    return np.tile(track, (NUM_ROTORS, 1))


def test_ground_level_reads_the_idle_plateau_not_the_takeoff_ramp():
    level = ground_level(_flight(_ground_then_flight(10.0, 90.0)))
    assert level == pytest.approx(10.0, abs=0.5)


def test_ground_level_is_none_when_the_flight_starts_airborne():
    """NeuroBEM segments, PI-TCN bags: no ground part at all. The airborne
    rule's 1 s erosion margin must not be mistaken for one."""
    rng = np.random.default_rng(2)
    cruise = 200.0 + rng.standard_normal((NUM_ROTORS, 6000))
    assert ground_level(_flight(cruise)) is None


def test_ground_level_is_none_when_the_motors_never_idle():
    """An instantaneous takeoff from motors-off: nothing pre-airborne clears
    IDLE_MIN_RPS, so there is no idle level to report."""
    fs = 100.0
    off = np.zeros((NUM_ROTORS, int(20 * fs)))
    cruise = np.full((NUM_ROTORS, int(60 * fs)), 90.0)
    assert ground_level(_flight(np.concatenate([off, cruise], axis=1), fs=fs)) is None
