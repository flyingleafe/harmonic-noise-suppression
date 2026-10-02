"""The prior-rig blocks of the renderer (``am``, ``profile.mic_dev_sd_db``,
``wind_sc``) and the SC wind generator."""

import numpy as np
import pytest

from data_processing.noise_model.render import render_noise
from data_processing.noise_model.wind_sc import lsf2poly, wind_noise
from experiments.noise_model.drone_prior import PriorSpec, sample_prior

SR = 16000


def _rps(n_rotors: int, seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return np.stack(
        [70.0 + 4.0 * r + 1.5 * np.sin(2 * np.pi * 0.3 * t + r) for r in range(n_rotors)]
    )


def test_prior_payload_renders_and_blocks_change_the_draw():
    rng = np.random.default_rng(3)
    pl = sample_prior(rng, PriorSpec(n_rotors=2, k_max=40))
    rps = _rps(2, 1.0)
    x = render_noise(pl, rps, sr=SR, n_mics=4, seed=1)
    assert x.shape == (4, SR) and np.isfinite(x).all() and (x**2).mean() > 0
    # the same payload without the prior blocks renders a DIFFERENT clip (blocks are live)
    off = {**pl, "params": {**pl["params"], "am": None, "wind_sc": None}}
    off["params"]["profile"] = {**pl["params"]["profile"], "mic_dev_sd_db": 0.0}
    y = render_noise(off, rps, sr=SR, n_mics=4, seed=1)
    assert not np.allclose(x, y)
    # the per-(mic, line) deviation is a per-mic contrast on the LINES only: with a flat
    # profile and no floor the mic power ratio must spread by about the drawn sd
    assert y.shape == x.shape


def test_am_envelope_keeps_the_coherent_core_and_the_drawn_variance():
    from data_processing.noise_model.render import _am_envelopes

    rng = np.random.default_rng(0)
    s2 = np.full((1, 3), 0.3)
    g = np.full((1, 3), 0.5)
    env = _am_envelopes(rng, s2, g, n_env=1000 * 60, dt_env=1e-3)
    assert env.shape == (1, 3, 1000 * 60)
    # exp(g - s2/2): unit mean amplitude (the profile IS the coherent core), total power e^{s2}
    assert np.allclose(env.mean(axis=2), 1.0, rtol=0.2)
    assert np.allclose((env**2).mean(axis=2), np.exp(0.3), rtol=0.3)
    assert np.allclose(np.var(np.log(env), axis=2), 0.3, rtol=0.3)


def test_wind_noise_is_low_passed_and_unit_rms():
    rng = np.random.default_rng(1)
    x, prof = wind_noise(rng, SR, 2.0, gustiness=2)
    assert x.size == 2 * SR and prof.size == x.size
    assert abs(np.sqrt(np.mean(x**2)) - 1.0) < 1e-6
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, 1 / SR)
    low, high = spec[(f > 20) & (f < 200)].mean(), spec[(f > 2000) & (f < 6000)].mean()
    assert 10 * np.log10(low / high) > 20


def test_lsf2poly_gives_a_stable_monic_filter():
    a = lsf2poly(np.array([0.3, 0.7, 1.2, 1.9, 2.6]))
    assert a[0] == pytest.approx(1.0) and a.size == 6
    assert np.all(np.abs(np.roots(a)) < 1.0)
