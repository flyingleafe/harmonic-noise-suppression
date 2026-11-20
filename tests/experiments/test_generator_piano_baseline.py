"""DDSP-Piano input contract and optional native checkpoint round trip."""

import os
from pathlib import Path
from shutil import copyfile

import numpy as np
import pytest

from experiments.generator_paper.cases import FlightClip
from experiments.generator_paper.piano_baseline import (
    PianoSettings,
    _conditioning,
    fit_piano,
    load_piano,
)


def test_continuous_midi_and_fixed_onset():
    rps = np.arange(4, dtype=np.float32)[:, None] + np.linspace(37, 53, 129)
    conditioning = _conditioning(rps)

    assert conditioning.shape == (3, 4, 2)
    np.testing.assert_allclose(
        440 * np.exp2((conditioning[..., 0] - 69) / 12), rps[:, ::64].T, rtol=1e-6
    )
    np.testing.assert_array_equal(conditioning[0, :, 1], np.full(4, 64 / 127, dtype=np.float32))
    assert not np.any(conditioning[1:, :, 1])
    with pytest.raises(ValueError, match="strictly positive"):
        _conditioning(np.zeros((4, 64)))


@pytest.mark.slow
def test_native_fit_and_movable_render(tmp_path):
    upstream = os.getenv("HNS_PIANO_UPSTREAM")
    if not upstream:
        pytest.skip("set HNS_PIANO_UPSTREAM and install native DDSP/TensorFlow runtime")

    sr = 16_000
    time = np.arange(3 * sr) / sr
    tracks = np.arange(4)[:, None] + np.linspace(39, 54, len(time))
    audio = np.sin(2 * np.pi * time * 45)[None, :] * np.array([[0.05], [0.1]])
    clip = FlightClip(
        key="synthetic-training-example",
        rig="synthetic",
        recording_id="synthetic",
        start_s=0.0,
        sr=sr,
        audio=audio,
        rps=tracks,
        native_sr=sr,
        native_audio=audio,
        regime="standby",
        provenance={"split": "calibration"},
    )
    checkpoint = tmp_path / "piano.json"
    report = fit_piano(
        [clip],
        checkpoint,
        device="cpu",
        upstream=Path(upstream),
        settings=PianoSettings(epochs=(1, 1, 1), steps_per_epoch=1, batch_sizes=(1, 2, 1)),
    )
    assert report["status"] == "completed"
    assert report["total_updates"] == 3
    assert not report["production_settings"]
    assert [phase["kind"] for phase in report["phases"]] == [
        "amplitude_noise",
        "frequency_tuning",
        "amplitude_noise",
    ]
    assert [epoch["steps"][0]["update"] for epoch in report["history"]] == [1, 2, 3]

    relocated = tmp_path / "relocated"
    relocated.mkdir()
    for name in (checkpoint.name, *report["artifacts"]):
        copyfile(tmp_path / name, relocated / name)
        (tmp_path / name).unlink()
    model = load_piano(relocated / checkpoint.name, upstream=Path(upstream))

    short = tracks[:, :193]
    draw = model.render(short, sr=sr, rng=np.random.default_rng(5))
    long = np.linspace(tracks[:, 0], tracks[:, -1], 100_337).T
    assert model.render(long, sr=sr, rng=np.random.default_rng(5)).shape == (2, 100_337)
    assert draw.shape == (2, 193)
    assert np.isfinite(draw).all()
    np.testing.assert_array_equal(draw, model.render(short, sr=sr, rng=np.random.default_rng(5)))
    assert not np.array_equal(draw, model.render(short, sr=sr, rng=np.random.default_rng(6)))
