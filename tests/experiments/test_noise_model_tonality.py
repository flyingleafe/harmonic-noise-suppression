"""The demodulated prominence (``tonality.demod_prominence``): what it reads.

A comb that follows its carrier must stand far over the floor; white noise must
read zero; the same comb read on a wrong carrier must lose it (which is why a
recording's number, read on noisy labels, is a lower bound); a stopped rotor
has no order to read.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.noise_model import tonality as TN

SR = 16000
SECONDS = 8.0


def _track(n_rotors: int) -> np.ndarray:
    """``(R, T)`` rev/s: each rotor sweeps +-6 rev/s around its own speed."""
    t = np.arange(int(SECONDS * SR)) / SR
    base = 80.0 + np.array([0.0, -6.3, 3.1, -8.7])[:n_rotors, None]
    phase = np.arange(n_rotors)[:, None]
    return base + 6.0 * np.sin(2.0 * np.pi * 0.3 * t[None, :] + phase)


def _comb(track: np.ndarray, rng: np.random.Generator, *, n_orders: int = 30) -> np.ndarray:
    """``(T,)`` unit-amplitude orders ``1 .. n_orders`` of ONE rotor, on ``track``."""
    phase = np.cumsum(track) / SR
    k = np.arange(1, n_orders + 1, dtype=np.float64)[:, None]
    alpha = rng.uniform(0.0, 2.0 * np.pi, size=(n_orders, 1))
    return np.cos(2.0 * np.pi * k * phase[None, :] + alpha).sum(axis=0)


@pytest.fixture(scope="module")
def comb_clip() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    track = _track(1)
    audio = _comb(track[0], rng)[None, :] + 1e-2 * rng.standard_normal((2, track.shape[1]))
    return audio, track


def test_comb_on_its_carrier_stands_far_over_the_floor(comb_clip) -> None:
    audio, track = comb_clip
    res = TN.demod_prominence(audio, track, SR, k_max=12)
    assert res.prom_db.shape == (1, 12)
    assert float(res.prom_db.min()) > 40.0
    summary = res.summary()
    assert summary["all"]["frac_gt10"] == 1.0
    assert summary["k1-4"]["n"] == 4 and summary["k9-16"]["n"] == 4


def test_comb_on_a_wrong_carrier_loses_its_prominence(comb_clip) -> None:
    """A 1 % speed error puts order k at 0.008 k f off DC: the line leaves the
    centre bin, so the number falls from tens of dB to around zero."""
    audio, track = comb_clip
    res = TN.demod_prominence(audio, track * 1.01, SR, k_max=12)
    assert float(np.median(res.prom_db[:, 3:])) < 3.0


def test_white_noise_reads_zero_and_a_stopped_rotor_is_skipped() -> None:
    rng = np.random.default_rng(1)
    track = _track(4)
    track[3] = 0.0  # a rotor that never turns
    noise = rng.standard_normal((4, track.shape[1]))
    res = TN.demod_prominence(noise, track, SR)
    live = res.prom_db[:3]
    assert np.isfinite(live).all()
    assert abs(float(np.median(live))) < 1.0
    assert float(np.abs(live).max()) < 4.0
    assert np.isnan(res.prom_db[3]).all()
    summary = res.summary()
    assert summary["all"]["n"] == 3 * TN.DEMOD_K_MAX
    assert summary["all"]["frac_gt6"] == 0.0
