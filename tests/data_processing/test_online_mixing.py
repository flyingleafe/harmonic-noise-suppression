"""Tests for the online-mixing pipeline compiler (``build_online_mix_pipeline``).

The policy-resolution unit tests are unchanged from the class-based mixer (the
policy semantics are preserved verbatim); the behavior tests exercise the
compiled dload pipeline over synthetic local-repo datasets (see
``tests/data_processing/conftest.py`` for the shared fixtures).
"""

from __future__ import annotations

import itertools
import pickle

import numpy as np
import pytest
import tdseries as td
from omegaconf import OmegaConf

from data_processing.frame_datasets import FixedSynthFrameDataset
from data_processing.frames import get_meta
from data_processing.online_mixing import (
    _max_window_s,
    _resolve_policy,
    _resolve_probability,
    _resolve_scalar,
    build_online_mix_pipeline,
    make_rng,
)
from data_processing.rate_warp import EDGE_S, SOURCE_MARGIN_S

from .conftest import SPEECH_DATASET, SR  # noqa: F401  (constants; fixtures auto-discovered)


def _rps_cfg(**over):
    cfg = {
        "sample_rate": SR,
        "duration_s": 1.0,
        "base_seed": 123,
        "sources": {
            "noise": [
                {
                    "kind": "frames",
                    "dataset": "MICHAELS-FRAMES-TEST",
                    "min_motor_rps": 0.0,
                }
            ],
            "speech": [{"dataset": SPEECH_DATASET, "subpath": "LibriSpeech/train-clean-100"}],
        },
        "policy": {"source_prob": 1.0, "snr_db": -10.0, "speech_per_channel": "shared"},
    }
    cfg.update(over)
    return cfg


def test_fixed_synthetic_validation_reuses_complete_disk_cache(tmp_path):
    cache = tmp_path / "fixed.pkl"
    frame = td.Frame({"feature": td.wrap(np.ones(2), dims=("feature",))})
    with cache.open("wb") as handle:
        pickle.dump([frame], handle)

    dataset = FixedSynthFrameDataset(
        path=tmp_path / "policy-does-not-need-to-exist.yaml",
        n=1,
        cache_path=cache,
    )

    np.testing.assert_array_equal(dataset[0]["feature"].data, np.ones(2))


# ─── policy resolution ────────────────────────────────────────────────────────


def test_resolve_policy_uses_sample_id_stages():
    policy = {
        "stages": [
            {"until": 50_000, "source_prob": 1.0},
            {"until": None, "source_prob": 1.0, "augmentations": {"probability": 0.5}},
        ]
    }

    assert "augmentations" not in _resolve_policy(policy, 49_999)
    assert _resolve_policy(policy, 50_000)["augmentations"]["probability"] == 0.5


def test_resolve_probability_ramp_interpolation():
    ramp = {"ramp": {"from": 25_000, "until": 125_000, "start": 0.0, "end": 0.7}}

    assert _resolve_probability(0.5, 0) == 0.5
    assert _resolve_probability(1, 0) == 1.0
    # Before/at the window start: clamped to `start` but FLOORED at 1e-9 — a
    # ramp never resolves to exactly 0.0 (the fire draw must stay consumed).
    assert _resolve_probability(ramp, 0) == 1e-9
    assert _resolve_probability(ramp, 25_000) == 1e-9
    # Inside: linear interpolation.
    assert abs(_resolve_probability(ramp, 75_000) - 0.35) < 1e-12
    assert abs(_resolve_probability(ramp, 100_000) - 0.525) < 1e-12
    # At/after the window end: clamped to `end`.
    assert _resolve_probability(ramp, 125_000) == 0.7
    assert _resolve_probability(ramp, 10_000_000) == 0.7


def test_resolve_scalar_linear_and_cosine_ramps():
    def ramp(shape):
        return {
            "ramp": {"from": 160_000, "until": 560_000, "start": -30.0, "end": 5.0, "shape": shape}
        }

    assert _resolve_scalar(-12.5, 0) == -12.5
    for shape in ("linear", "cosine"):
        # Clipped to start before/at `from`, to end at/after `until`; a severity
        # ramp has no floor (it may cross zero and be negative).
        assert _resolve_scalar(ramp(shape), 0) == -30.0
        assert _resolve_scalar(ramp(shape), 160_000) == -30.0
        assert _resolve_scalar(ramp(shape), 560_000) == 5.0
        assert _resolve_scalar(ramp(shape), 1_600_000) == 5.0
        # Both shapes pass the midpoint at the mean of start and end.
        assert abs(_resolve_scalar(ramp(shape), 360_000) - (-12.5)) < 1e-12
    # Linear is linear; cosine is slow at the ends: (1 - cos(pi t)) / 2 at t = 1/4.
    assert abs(_resolve_scalar(ramp("linear"), 260_000) - (-30.0 + 35.0 * 0.25)) < 1e-12
    quarter = -30.0 + 35.0 * (1.0 - np.cos(np.pi / 4)) / 2
    assert abs(_resolve_scalar(ramp("cosine"), 260_000) - quarter) < 1e-12
    # A decreasing ramp (DRR lower bound 15 -> 3 dB) mirrors it.
    drr = {"ramp": {"from": 80_000, "until": 320_000, "start": 15.0, "end": 3.0, "shape": "cosine"}}
    assert abs(_resolve_scalar(drr, 200_000) - 9.0) < 1e-12
    # `shape` defaults to linear (the historical probability ramp).
    assert _resolve_scalar({"ramp": {"from": 0, "until": 10, "start": 0.0, "end": 1.0}}, 5) == 0.5


@pytest.mark.parametrize(
    "bad",
    [
        {"ramp": {"from": 0, "until": 10, "start": 0.0, "end": 1.0, "shape": "exp"}},
        {"ramp": {"from": 10, "until": 10, "start": 0.0, "end": 1.0}},
        {"ramp": {"from": 0, "until": 10, "start": 0.0}},
        {"ramp": {"from": 0, "until": 10, "start": 0.0, "end": 1.0, "speed": 2}},
    ],
)
def test_malformed_ramps_are_rejected(bad):
    with pytest.raises(ValueError, match="ramp"):
        _resolve_scalar(bad, 5)


def test_resolve_policy_materializes_every_ramp_without_mutation():
    p_ramp = {"ramp": {"from": 100, "until": 200, "start": 0.0, "end": 1.0}}
    snr_hi = {"ramp": {"from": 100, "until": 200, "start": -30.0, "end": 5.0, "shape": "cosine"}}
    drr_lo = {"ramp": {"from": 100, "until": 200, "start": 15.0, "end": 3.0}}
    alpha_hi = {"ramp": {"from": 100, "until": 200, "start": 1.0, "end": 1.5}}
    recolor = {"probability": 1.0, "choices": [{"spectral_recolor": {}}]}
    stage = {
        "until": None,
        "snr_db": {"uniform": {"low": -30.0, "high": snr_hi}},
        "augmentations": {"probability": p_ramp, "choices": [{"random_gain": {}}]},
        "noise_augmentations": [
            recolor,
            {"probability": 0.5, "choices": [{"random_reverb": {"drr_db_low": drr_lo}}]},
        ],
        "noise_rate_warp": {"probability": 1.0, "alpha_low": 0.5, "alpha_high": alpha_hi},
    }
    policy = {"stages": [stage]}

    resolved = _resolve_policy(policy, 150)

    assert resolved["augmentations"]["probability"] == 0.5
    assert resolved["snr_db"]["uniform"]["low"] == -30.0
    assert resolved["snr_db"]["uniform"]["high"] == pytest.approx(-12.5)
    reverb = resolved["noise_augmentations"][1]["choices"][0]["random_reverb"]
    assert reverb == {"drr_db_low": 9.0}
    assert resolved["noise_rate_warp"]["alpha_high"] == 1.25
    # The source policy is never mutated (it is shared across samples/workers).
    assert stage["snr_db"]["uniform"]["high"] is snr_hi
    assert stage["augmentations"]["probability"] is p_ramp
    assert stage["noise_augmentations"][1]["choices"][0]["random_reverb"]["drr_db_low"] is drr_lo
    assert stage["noise_rate_warp"]["alpha_high"] is alpha_hi
    # Ramp-free sub-trees are reused as-is.
    assert resolved["noise_augmentations"][0] is recolor


def test_window_covers_the_worst_rate_over_the_ramp_and_ignores_inactive_warps():
    warp = {
        "probability": 1.0,
        "alpha_low": 0.5,
        "alpha_high": {"ramp": {"from": 0, "until": 10, "start": 1.5, "end": 1.2}},
        "dev_const": 0.08,
        "dev_sine": 0.04,
        "f_low": 0.1,
        "f_high": 1.0,
    }
    expected = 1.5 * 1.12 * (2.0 + 2 * EDGE_S) + SOURCE_MARGIN_S
    assert abs(_max_window_s({"stages": [{"noise_rate_warp": warp}]}, 2.0) - expected) < 1e-12
    assert (
        _max_window_s({"stages": [{"noise_rate_warp": {**warp, "probability": 0.0}}]}, 2.0) == 2.0
    )
    assert _max_window_s({"source_prob": 1.0}, 2.0) == 2.0


# ─── compiled-pipeline behavior ────────────────────────────────────────────────


def _take(cfg, n, start=0):
    pipe = build_online_mix_pipeline({**cfg, "start_sample_id": start})
    return list(itertools.islice(iter(pipe), n))


def test_pipeline_sample_shapes_determinism_and_ids(michaels_frames_dataset, speech_dataset):
    cfg = _rps_cfg()
    run_a = _take(cfg, 4)
    run_b = _take(cfg, 4)

    for fa, fb in zip(run_a, run_b):
        np.testing.assert_array_equal(
            np.asarray(fa["mixture"].data), np.asarray(fb["mixture"].data)
        )
        np.testing.assert_array_equal(np.asarray(fa["rps"].data), np.asarray(fb["rps"].data))

    first = run_a[0]
    assert first["mixture"].shape == (2, SR)  # the 2-mic synthetic recording
    assert first["rps"].shape == (4, SR // 512 + 1)
    assert [get_meta(f, "sample_id") for f in run_a] == [0, 1, 2, 3]
    # different samples -> different content
    assert not np.array_equal(
        np.asarray(run_a[0]["mixture"].data), np.asarray(run_a[1]["mixture"].data)
    )


def test_pipeline_start_sample_id_offsets_the_id_stream(michaels_frames_dataset, speech_dataset):
    frames = _take(_rps_cfg(), 3, start=41)
    assert [get_meta(f, "sample_id") for f in frames] == [41, 42, 43]


def test_pipeline_speech_excludes_held_out_speakers(michaels_frames_dataset, speech_dataset):
    # The speech fixture keys carry speaker ids 19/103/200; excluding 200 must
    # leave the stream functional (it simply never draws 200/*).
    cfg = _rps_cfg()
    cfg["sources"]["speech"] = [
        {
            "dataset": SPEECH_DATASET,
            "subpath": "LibriSpeech/train-clean-100",
            "exclude": ["200"],
        }
    ]
    frames = _take(cfg, 3)
    assert all(f["mixture"].shape[-1] == SR for f in frames)


def test_online_mix_frame_dataset_dataloader_batches(michaels_frames_dataset, speech_dataset):
    from torch.utils.data import DataLoader

    from data_processing.collate import frame_collate
    from data_processing.frame_datasets import OnlineMixFrameDataset

    ds = OnlineMixFrameDataset.from_config(_rps_cfg())
    loader = DataLoader(ds, batch_size=4, num_workers=2, collate_fn=frame_collate)
    batch = next(iter(loader))
    assert batch["mixture"].data.shape == (4, 2, SR)
    assert batch["rps"].data.shape == (4, 4, SR // 512 + 1)


def test_noise_augmentation_blocks_change_audio_never_labels(
    michaels_frames_dataset, speech_dataset
):
    # Three policies sharing every seed: the chunk at a stream position is
    # identical; only the fire decisions of the two sequential blocks differ.
    def cfg_with(p_recolor: float, p_reverb: float):
        cfg = _rps_cfg(base_seed=7)
        cfg["policy"] = {
            "source_prob": 1.0,
            "snr_db": -10.0,
            "speech_per_channel": "shared",
            "noise_augmentations": [
                {"probability": p_recolor, "choices": [{"spectral_recolor": {}}]},
                {"probability": p_reverb, "choices": [{"random_reverb": {}}]},
            ],
        }
        return cfg

    pos = 5
    both = _take(cfg_with(1.0, 1.0), 1, start=pos)[0]
    first = _take(cfg_with(1.0, 1e-9), 1, start=pos)[0]
    neither = _take(cfg_with(1e-9, 1e-9), 1, start=pos)[0]

    audio = [np.asarray(f["mixture"].data) for f in (both, first, neither)]
    assert not np.array_equal(audio[0], audio[1])  # block 2 fires on top of block 1
    assert not np.array_equal(audio[1], audio[2])
    for f in (both, first):  # label-preserving: bit-identical targets
        np.testing.assert_array_equal(np.asarray(f["rps"].data), np.asarray(neither["rps"].data))


def _warp_cfg(alpha_low=0.85, alpha_high=1.15, silence_weight=None):
    cfg = _rps_cfg(base_seed=11)
    if silence_weight is not None:
        cfg["sources"]["noise"].append(
            {"kind": "silence", "weight": silence_weight, "n_channels": 2}
        )
    cfg["policy"] = {
        "source_prob": 0.0,
        "noise_rate_warp": {
            "probability": 1.0,
            "alpha_low": alpha_low,
            "alpha_high": alpha_high,
            "dev_const": 0.08,
            "dev_sine": 0.04,
            "f_low": 0.1,
            "f_high": 1.0,
        },
    }
    return cfg


def test_rate_warp_labels_are_alpha_times_source_and_deterministic_per_id(
    michaels_frames_dataset, michaels_rps_values, speech_dataset
):
    from data_processing.online_mixing import _maybe_sample_rate_warp

    cfg = _warp_cfg(silence_weight=1.0)
    run_a = _take(cfg, 8)
    run_b = _take(cfg, 8)

    base = michaels_rps_values[:, 0].astype(np.float64)  # constant 60..63 rev/s
    kinds = set()
    for fa, fb in zip(run_a, run_b):
        mix, rps = np.asarray(fa["mixture"].data), np.asarray(fa["rps"].data)
        np.testing.assert_array_equal(mix, np.asarray(fb["mixture"].data))
        np.testing.assert_array_equal(rps, np.asarray(fb["rps"].data))
        assert mix.shape == (2, SR) and rps.shape == (4, SR // 512 + 1)
        if not rps.any():
            kinds.add("silence")  # the model-rate floor: zero labels, never warped
            continue
        kinds.add("warped")
        gid = int(get_meta(fa, "sample_id"))
        policy = _resolve_policy(cfg["policy"], gid)
        warp = _maybe_sample_rate_warp(policy["noise_rate_warp"], make_rng(11, gid))
        assert warp is not None
        alpha = warp.alpha(np.arange(rps.shape[1]) * 512 / SR)
        np.testing.assert_allclose(rps, alpha[None, :] * base[:, None], rtol=0, atol=2e-3)
    assert kinds == {"silence", "warped"}


def test_ramped_bounds_consume_no_rng(michaels_frames_dataset, speech_dataset):
    def cfg_with(high):
        cfg = _rps_cfg(base_seed=3)
        cfg["policy"] = {
            "source_prob": 1.0,
            "speech_per_channel": "shared",
            "snr_db": {"uniform": {"low": -20.0, "high": high}},
            "augmentations": {"probability": 0.5, "choices": [{"random_gain": {}}]},
        }
        return cfg

    flat_ramp = {"ramp": {"from": 0, "until": 1000, "start": -5.0, "end": -5.0}}
    for fa, fb in zip(_take(cfg_with(-5.0), 4), _take(cfg_with(flat_ramp), 4)):
        np.testing.assert_array_equal(
            np.asarray(fa["mixture"].data), np.asarray(fb["mixture"].data)
        )


@pytest.mark.parametrize(
    ("stage", "match"),
    [
        ({"noise_time_warp": {"probability": 0.5}}, "removed in the rp cutover"),
        (
            {"noise_augmentations": {"probability": 1.0, "choices": [{"freq_scale": {}}]}},
            "removed in the rp cutover",
        ),
        (
            {"noise_augmentations": [{"probability": 0.7, "choices": ["tooth_dropout"]}]},
            "removed in the rp cutover",
        ),
        ({"augmentations": {"probability": 0.5, "choices": ["random_polarity"]}}, "removed"),
        (
            {"noise_augmentations": {"choices": [{"random_reverb": {"drr_low_db": 3.0}}]}},
            "unknown parameters",
        ),
        ({"noise_rate_warp": {"probability": 1.0, "alpha_low": 0.5}}, "missing"),
        (
            {"snr_db": {"uniform": {"low": -30.0, "high": {"ramp": {"from": 5, "until": 1}}}}},
            "ramp",
        ),
    ],
)
def test_pre_rp_and_malformed_policies_fail_at_build(stage, match):
    cfg = {
        "sources": {"noise": [{"kind": "frames", "dataset": "NEVER-LOADED"}]},
        "policy": {"stages": [{"until": None, **stage}]},
    }
    with pytest.raises(ValueError, match=match):
        build_online_mix_pipeline(cfg)


def test_rate_warp_needs_native_headroom(michaels_frames_dataset, speech_dataset):
    # alpha_low 0.5 reaches local rates of 0.44: a 32 kHz recording would leave
    # 7-8 kHz empty, so the pool refuses it (16 kHz / 0.44 = 36.4 kHz needed).
    with pytest.raises(ValueError, match="native audio above"):
        build_online_mix_pipeline(_warp_cfg(alpha_low=0.5))
    # Engines render at the model rate; only the silence floor is admitted.
    cfg = _warp_cfg()
    cfg["sources"]["noise"] = [{"kind": "static_comb"}]
    with pytest.raises(ValueError, match="native-rate noise"):
        build_online_mix_pipeline(cfg)


def test_v4_michaels_online_config_uses_librispeech_dataset_form():
    cfg = OmegaConf.load("conf/online_mix/online_mix_v4_michaels_train_no_room1.yaml")
    speech = cfg.sources.speech[0]

    assert speech.dataset == "librispeech"
    assert speech.subpath == "LibriSpeech/train-clean-100"
    assert "root" not in speech and "cache" not in speech
