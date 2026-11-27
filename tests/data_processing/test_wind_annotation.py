"""Synthetic behaviour of the PCM wind/downwash assessment (auxiliary to the real
uni-cpu validation; no recordings are read here)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from data_processing.noise_model.wind_sc import wind_noise
from data_processing.recording_annotations import wind as W

SR = 16000


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(spec.size, dtype=float)
    f[0] = 1.0
    x = np.fft.irfft(spec / np.sqrt(f), n)
    return x / x.std()


def _array(n: int, rng: np.random.Generator, n_ch: int = 4) -> np.ndarray:
    """A compact array in an acoustic pink field (coherent) with -30 dB sensor noise."""
    common = _pink(n, rng)
    return np.stack([common + 0.03 * rng.standard_normal(n) for _ in range(n_ch)])


POS = [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.05, 0.05, 0.0]]


def test_stationary_noise_is_not_significant_and_null_matches_constants():
    rng = np.random.default_rng(0)
    res = W.assess((0.01 * rng.standard_normal((4, SR * 20))).astype(np.float32), SR, {})
    assert res["wind"] == "not_significant"
    rho = [c["rho"] for c in res["channels"]]
    gust = [c["gust_db"] for c in res["channels"]]
    assert max(abs(r) for r in rho) < 3 * W.NULL_RHO_SD
    assert abs(np.median(gust) - W.NULL_GUST_DB) < 0.4


def test_short_simulated_and_low_rate_inputs():
    assert W.assess(np.zeros(SR // 2, np.float32), SR, {})["reason"] == "too_short"
    assert (
        W.assess(np.zeros(SR * 2, np.float32), SR, {"environment": "simulated"})["wind"]
        == "not_applicable"
    )
    assert W.assess(np.zeros(4000, np.float32), 2000, {})["reason"] == "sample_rate_too_low"
    assert W.too_short(0.5, "ground")["wind"] == "unknown"


def test_incoherent_wind_on_one_capsule_of_a_coherent_array():
    rng = np.random.default_rng(1)
    n = SR * 20
    x = _array(n, rng)
    w, _ = wind_noise(np.random.default_rng(2), SR, 20.0, speed_offset_ms=3.0)
    x[2] += 30.0 * w[:n]
    res = W.assess((0.002 * x).astype(np.float32), SR, {"mic_positions": POS})
    assert res["wind"] in ("significant", "mixed")
    assert res["windy_channels"] == [2]
    assert res["spatial"] == "acoustic_reference"
    calm = [c for c in res["channels"] if c["channel"] != 2]
    assert all(c["wind"] == "not_significant" for c in calm)


def test_incoherent_self_noise_and_mono_buffeting_are_not_certified():
    rng = np.random.default_rng(8)
    n = SR * 20
    # Compact array whose LF is independent stationary self-noise: incoherent but not wind.
    sos = butter(2, 150.0, btype="lowpass", fs=SR, output="sos")
    selfnoise = np.stack(
        [sosfilt(sos, rng.standard_normal(n)) + 0.01 * rng.standard_normal(n) for _ in range(4)]
    )
    res = W.assess((0.01 * selfnoise).astype(np.float32), SR, {"mic_positions": POS})
    assert res["wind"] not in ("significant", "mixed")
    # Mono buffeting: counted as buffeting-like, never certified without spatial evidence.
    w, _ = wind_noise(np.random.default_rng(9), SR, 20.0, speed_offset_ms=3.0)
    mono = 0.05 * w[:n] + 0.0005 * rng.standard_normal(n)
    res = W.assess(mono[None].astype(np.float32), SR, {})
    assert res["wind"] == "unknown"


def test_periodic_beating_rotor_lines_and_dc_are_not_wind():
    rng = np.random.default_rng(3)
    n = SR * 20
    t = np.arange(n) / SR
    tones = sum(
        np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28)) / k
        for f0 in (95.0, 101.0, 108.0, 115.0)
        for k in range(1, 20)
    )
    am = 1.0 + 0.5 * np.sin(2 * np.pi * 2.0 * t)  # strong 2 Hz level modulation of the lines
    x = _array(n, rng) * 0.3 + (am * tones)[None] + 0.4  # plus a DC bias
    res = W.assess((0.01 * x).astype(np.float32), SR, {"mic_positions": POS})
    assert res["wind"] == "not_significant"


def test_zero_padding_is_excluded_from_coverage():
    rng = np.random.default_rng(4)
    x = (0.01 * rng.standard_normal((2, SR * 10))).astype(np.float32)
    x[:, : SR * 3] = 0.0
    x[1, -SR * 2 :] = np.nan
    res = W.assess(x, SR, {})
    cov = {c["channel"]: c["covered_s"] for c in res["channels"]}
    assert cov[0] < 7.5 and cov[1] < cov[0]
    assert res["wind"] == "not_significant"


def test_high_passed_wind_floor_is_not_called_clean():
    rng = np.random.default_rng(5)
    n = SR * 20
    sos = butter(8, 300.0, btype="highpass", fs=SR, output="sos")
    w, _ = wind_noise(np.random.default_rng(6), SR, 20.0, speed_offset_ms=3.0)
    x = np.asarray(sosfilt(sos, 0.05 * w[:n] + 0.001 * rng.standard_normal(n)), dtype=np.float32)
    res = W.assess(x[np.newaxis], SR, {})
    assert res["wind"] in ("unknown", "significant", "mixed")


def test_merge_and_combine_precedence():
    rng = np.random.default_rng(7)
    a = W.assess((0.01 * rng.standard_normal((2, SR * 6))).astype(np.float32), SR, {}, role="audio")
    b = W.assess((0.01 * rng.standard_normal((2, SR * 4))).astype(np.float32), SR, {}, role="audio")
    m = W.merge([a, b])
    assert m["coverage_s"] == pytest.approx(a["coverage_s"] + b["coverage_s"], abs=0.01)
    assert m["n_views"] == 2 and m["wind"] == "not_significant"
    pcm = W.combine({"audio": m})
    assert pcm["wind"] == "not_significant" and pcm["role_wind"] == {"audio": "not_significant"}
    direct = W.combine(
        {"audio": m}, {"wind": "significant", "evidence": "annotator: gusts audible"}
    )
    assert direct["wind"] == "significant" and "PCM disagrees" in direct["evidence"]
    partial = W.combine({"audio": m, "ground": W.too_short(0.5, "ground")})
    assert partial["wind"] == "unknown"
    focused = W.combine({"audio": m, "ground": W.too_short(0.5, "ground")}, wind_role="audio")
    assert focused["wind"] == "not_significant"
