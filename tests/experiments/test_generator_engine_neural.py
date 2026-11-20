"""Opt-in round trips against the real pinned PTR checkout, without dataset access."""

import os
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from experiments.generator_paper.cases import FlightClip
from experiments.generator_paper.engine_neural import (
    UPSTREAM_COMMIT,
    EngineSettings,
    fit_engine,
    load_engine,
)


@pytest.mark.slow
@pytest.mark.parametrize("method", ["ptr", "hpn"])
def test_engine_native_roundtrip(tmp_path, method):
    root = os.environ.get("HNS_PTR_UPSTREAM")
    if not root:
        pytest.skip("set HNS_PTR_UPSTREAM to the pinned official checkout")
    pytest.importorskip("torchlpc")
    sr = 16000
    n = 16640  # Minimum 128-aligned length that exercises all native loss terms.
    t = np.arange(n) / sr
    rps = np.array([75 + k * 2.1 + 4 * np.sin((k + 1) * t) for k in range(4)])
    audio = (0.05 * np.sin(2 * np.pi * np.cumsum(rps, axis=-1) / sr).sum(axis=0))[None]
    clip = FlightClip(
        key="calibration-smoke",
        rig="md2",
        recording_id="synthetic",
        start_s=0.0,
        sr=sr,
        audio=audio,
        rps=rps,
        native_sr=sr,
        native_audio=audio,
        regime="cruise",
        provenance={"split": "calibration", "label_key": "rps_refined"},
    )
    checkpoint = tmp_path / "expert.json"
    report = fit_engine(
        [clip],
        checkpoint,
        method=method,
        device="cpu",
        upstream=Path(root),
        settings=EngineSettings(epochs=1, batch_size=1, chunk_size=n),
    )
    assert report["status"] == "completed"
    assert not report["production_settings"]
    assert report["upstream"]["commit"] == UPSTREAM_COMMIT
    assert report["n_mics"] == 1 and report["n_rotors"] == 4
    assert report["calibration_keys"] == [clip.key]
    assert report["optimizer_steps_per_mic"] == 1
    step = report["mic_fits"][0]["steps"][0]
    assert step["total_loss"] == pytest.approx((step["stft_loss"] + step["harmonic_loss"]) / 2)
    assert np.isfinite(step["gradient_norm"])
    assert len(set(report["mic_fits"][0]["source_initialization_seeds"])) == 4

    moved = tmp_path / "moved"
    moved.mkdir()
    weights = checkpoint.with_name(report["weights"]["file"])
    assert sha256(weights.read_bytes()).hexdigest() == report["weights"]["sha256"]
    weights.rename(moved / weights.name)
    checkpoint = checkpoint.rename(moved / checkpoint.name)
    expert = load_engine(checkpoint, device="cpu")
    assert expert.report == report

    for length in (4097, 1, 4111):
        track = rps[:, :length]
        first = expert.render(track, sr=sr, rng=np.random.default_rng(17))
        second = expert.render(track, sr=sr, rng=np.random.default_rng(17))
        assert first.shape == (1, length)
        assert np.isfinite(first).all()
        np.testing.assert_array_equal(first, second)
    different = expert.render(track, sr=sr, rng=np.random.default_rng(18))
    assert not np.array_equal(second, different)
    with pytest.raises(ValueError, match="positive RPS"):
        expert.render(np.zeros_like(track), sr=sr, rng=np.random.default_rng(17))
