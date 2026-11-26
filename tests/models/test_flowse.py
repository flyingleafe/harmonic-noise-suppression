"""Tests for the FlowSE flow-matching SE baseline (``conf/model/flowse.yaml``).

Built through ``training.config.instantiate_model`` exactly as a Hydra run
builds it. Covers both legs of the generative protocol (``forward(mix,
target)`` -> CFM loss, ``forward(mix)`` -> Euler sampler), the codec call path,
the parameter count, and that the EMA weights travel through ``state_dict``
and are the ones evaluation uses. Signals stay at or below 1 s and the
sampler at one or two steps so the file runs on CPU.

The full 65.6M-parameter network is used for every inference test. The
training-mechanics tests (backward pass, Adam steps + EMA + state_dict round
trip) run by default on the same config with a narrow backbone (``nf=16``,
identical code path) and on the full network only under ``-m slow``, since a
full-size backward pass with Adam state needs more than 1 GB of RAM.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from models.flowse import FlowSEModel
from tasks.codecs import build_codec
from training.config import instantiate_model

CONF = Path(__file__).resolve().parents[2] / "conf" / "model" / "flowse.yaml"

# NCSN++ in the BBED/CRP/SGMSE+ configuration. The FlowSE paper prints no
# count; the SGMSE+ paper quotes 65.6M for this network, and the official
# flowmse code (seongq/flowmse @ f6b479d1) builds exactly this many. The total
# includes the 128 frozen weights of the Gaussian Fourier time projection.
PARAMS = 65_590_822
TRAINABLE = 65_590_694

BACKBONES = [
    pytest.param({"nf": 16}, id="nf16"),
    pytest.param({}, id="full", marks=pytest.mark.slow),
]


def _load(num_steps: int = 1, **backbone: object) -> FlowSEModel:
    cfg = OmegaConf.load(CONF)
    cfg.params.num_steps = num_steps
    for key, value in backbone.items():
        cfg.params.backbone[key] = value
    model = instantiate_model(cfg)
    assert isinstance(model, FlowSEModel)
    return model


@pytest.fixture(scope="module")
def eval_model() -> FlowSEModel:
    torch.manual_seed(0)
    return _load(num_steps=1).eval()


@pytest.mark.parametrize("length", [4001, 16000])
def test_forward_shape_and_finite(eval_model: FlowSEModel, length: int) -> None:
    x = 0.1 * torch.randn(2, 1, length)
    with torch.no_grad():
        out = eval_model(x)
    assert out.shape == (2, 1, length), out.shape
    assert torch.isfinite(out).all()


def test_codec_roundtrip(eval_model: FlowSEModel) -> None:
    codec = build_codec(
        "speech_enhancement", n_channels=None, use_rps=False, predict_rps=False, sr=[16000, 1]
    )
    inputs = {"mixture": 0.1 * torch.randn(1, 16000)}
    with torch.no_grad():
        out = codec.call_model(eval_model, inputs)
    out = out.squeeze(1)
    assert out.shape == (1, 16000), out.shape
    assert torch.isfinite(out).all()


def test_param_count(eval_model: FlowSEModel) -> None:
    n_params = sum(p.numel() for p in eval_model.parameters())
    assert n_params == PARAMS, n_params
    n_trainable = sum(p.numel() for p in eval_model.parameters() if p.requires_grad)
    assert n_trainable == TRAINABLE, n_trainable
    # the EMA copy is buffers, not parameters
    n_ema = sum(t.numel() for t in eval_model.ema.shadow_tensors())
    assert n_ema == PARAMS


def test_sampler_is_deterministic_given_generator(eval_model: FlowSEModel) -> None:
    mix = 0.1 * torch.randn(1, 1, 4000)
    a = eval_model.enhance(mix, num_steps=2, generator=torch.Generator().manual_seed(3))
    b = eval_model.enhance(mix, num_steps=2, generator=torch.Generator().manual_seed(3))
    c = eval_model.enhance(mix, num_steps=2, generator=torch.Generator().manual_seed(4))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


@pytest.mark.parametrize("backbone", BACKBONES)
def test_loss_backward_finite(backbone: dict) -> None:
    torch.manual_seed(0)
    model = _load(**backbone).train()
    mix = 0.1 * torch.randn(1, 1, 4000)
    target = 0.5 * mix + 0.01 * torch.randn(1, 1, 4000)
    loss = model(mix, target)
    assert loss.ndim == 0 and torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    grads = [p.grad for p in trainable if p.grad is not None]
    assert len(grads) == len(trainable)
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("backbone", BACKBONES)
def test_state_dict_carries_ema(backbone: dict) -> None:
    torch.manual_seed(0)
    model = _load(num_steps=2, **backbone).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mix = 0.1 * torch.randn(1, 1, 4000)
    target = 0.5 * mix
    for _ in range(2):
        loss = model(mix, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    del opt, loss

    model.eval()  # applies the EMA update of the last step
    assert model.ema.num_updates == 2
    ema_gap = max(
        (s - p).abs().max().item()
        for s, p in zip(model.ema.shadow_tensors(), model.dnn.parameters(), strict=True)
    )
    assert ema_gap > 0.0

    state = model.state_dict()
    assert any(k.startswith("ema.module.") for k in state)
    assert "ema._extra_state" in state
    buf = io.BytesIO()
    torch.save(state, buf)
    del state
    buf.seek(0)

    gen = torch.Generator()
    out = model.enhance(mix, generator=gen.manual_seed(5))
    model.train()
    out_raw = model.enhance(mix, generator=gen.manual_seed(5))
    assert not torch.equal(out, out_raw), "eval must sample with the EMA weights"
    del model

    torch.manual_seed(1)  # a different random init for the fresh instance
    fresh = _load(num_steps=2, **backbone)
    fresh.load_state_dict(torch.load(buf, weights_only=True))
    fresh.eval()
    assert fresh.ema.num_updates == 2
    out_fresh = fresh.enhance(mix, generator=gen.manual_seed(5))
    assert torch.equal(out, out_fresh)
