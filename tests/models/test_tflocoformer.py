"""Tests for the TF-Locoformer (Medium) SE port and its speech_enhancement wiring.

Builds the model from ``conf/model/tflocoformer.yaml`` through the exact
``training.config.instantiate_model`` dispatch a real Hydra run uses. Short
signals so everything runs on CPU in seconds. Numerical parity against the
upstream module is a scratch-script check (upstream code is not vendored);
the ``slow`` test pins the published DNS checkpoint's strict load.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from models.tflocoformer import RotaryEmbedding
from tasks.codecs import build_codec
from training.config import instantiate_model

ROOT = Path(__file__).resolve().parents[2]
CONF = ROOT / "conf" / "model" / "tflocoformer.yaml"
DNS_CKPT = ROOT / ".cache" / "pretrained" / "tflocoformer" / "dns_ins20_valid.loss.ave_5best.pth"

# Upstream TFLocoformerSeparator(num_spk=1, medium DNS recipe): sum of parameters().
UPSTREAM_PARAMS = 14_973_346
ROPE_FREQ_PARAMS = 32  # two shared 16-entry RoPE tables, requires_grad=False


def _load_model() -> torch.nn.Module:
    torch.manual_seed(0)
    return instantiate_model(OmegaConf.load(CONF))


@pytest.fixture(scope="module")
def model() -> torch.nn.Module:
    return _load_model().eval()


@pytest.mark.parametrize("n_samples", [4000, 12345])
def test_forward_shape_and_finite(model: torch.nn.Module, n_samples: int) -> None:
    x = torch.randn(2, 1, n_samples)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, n_samples), out.shape
    assert torch.isfinite(out).all()


def test_codec_roundtrip(model: torch.nn.Module) -> None:
    codec = build_codec(
        "speech_enhancement", n_channels=None, use_rps=False, predict_rps=False, sr=[16000, 1]
    )
    inputs = {"mixture": torch.randn(2, 16000)}
    with torch.no_grad():
        out = codec.call_model(model, inputs)
    out = out.squeeze(1)
    assert out.shape == (2, 16000), out.shape
    assert torch.isfinite(out).all()


def test_param_count_matches_upstream_medium(model: torch.nn.Module) -> None:
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == UPSTREAM_PARAMS, n_params
    assert n_trainable == UPSTREAM_PARAMS - ROPE_FREQ_PARAMS, n_trainable
    assert round(n_params / 1e6, 1) == 15.0  # paper, Table 1 (M)


def test_scale_equivariant(model: torch.nn.Module) -> None:
    """Std normalisation in and out makes the estimate scale with the input."""
    x = torch.randn(1, 1, 4000)
    with torch.no_grad():
        a = model(x)
        b = model(1e-3 * x)
    torch.testing.assert_close(b, 1e-3 * a, rtol=1e-4, atol=1e-9)


def test_rope_matches_reference_formula() -> None:
    """Interleaved-pair RoPE with theta 10000 (rotary_embedding_torch defaults)."""
    rope = RotaryEmbedding(32)
    t = torch.randn(2, 4, 7, 32)
    pos = torch.arange(7.0)[:, None]
    inv = 1.0 / 10000 ** (torch.arange(0, 32, 2).float() / 32)
    ang = pos * inv  # (7, 16)
    x1, x2 = t[..., 0::2], t[..., 1::2]
    expected = torch.empty_like(t)
    expected[..., 0::2] = x1 * ang.cos() - x2 * ang.sin()
    expected[..., 1::2] = x2 * ang.cos() + x1 * ang.sin()
    torch.testing.assert_close(rope(t), expected)


def test_backward_finite_grads() -> None:
    model = _load_model().train()
    x = torch.randn(2, 1, 4000)
    target = torch.randn(2, 1, 4000)
    loss = (model(x) - target).abs().mean()
    loss.backward()
    params = [p for p in model.parameters() if p.requires_grad]
    grads = [p.grad for p in params if p.grad is not None]
    assert len(grads) == len(params)
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_espnet_xavier_init() -> None:
    model = _load_model()
    sd = model.state_dict()
    assert all(v.abs().sum() == 0 for k, v in sd.items() if k.endswith(".bias") and v.dim() == 1)
    assert torch.all(sd["separator.conv.1.weight"] == 1)  # GroupNorm reset
    assert all(torch.all(v == 1) for k, v in sd.items() if k.endswith("gamma"))


def test_state_dict_roundtrip(model: torch.nn.Module) -> None:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    torch.manual_seed(1)
    fresh = instantiate_model(OmegaConf.load(CONF)).eval()
    fresh.load_state_dict(torch.load(buf, weights_only=True), strict=True)
    x = torch.randn(1, 1, 8000)
    with torch.no_grad():
        torch.testing.assert_close(fresh(x), model(x), rtol=0, atol=0)


@pytest.mark.slow
def test_dns_checkpoint_loads_strict() -> None:
    """The published DNS-2020 checkpoint (upstream repo, 60 MB) loads unchanged."""
    if not DNS_CKPT.exists():
        pytest.skip(f"missing {DNS_CKPT}")
    cfg = OmegaConf.load(CONF)
    cfg.params.checkpoint = str(DNS_CKPT)
    model = instantiate_model(cfg).eval()
    x = torch.randn(1, 1, 16000)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 16000)
    assert torch.isfinite(out).all()
