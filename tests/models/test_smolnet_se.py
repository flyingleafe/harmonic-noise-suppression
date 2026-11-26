"""Tests for the SMoLnet / SMoLnet-T speech-enhancement ports (``models.smolnet_se``).

Both models are built from ``conf/model/smolnet{,_t}.yaml`` through
``training.config.instantiate_model``. No upstream code exists, so the
parity checks are the parameter counts against the papers, a hand-written
Eqs. (27)-(37) reference for one spectral combination transformer, and the
SMoLnet conv body against the in-repo ``SMoLnetRPSBackbone`` configured as
the 2019 paper's layers 1-13. CPU-fast: signals of at most 1 s.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from models.smolnet_se import (
    SMoLnet,
    SMoLnetT,
    SpectralCombinationTransformer,
    sinusoidal_positional_encoding,
)
from tasks.codecs import build_codec
from training.config import instantiate_model

CONF_DIR = Path(__file__).resolve().parents[2] / "conf" / "model"
CONFS = ["smolnet", "smolnet_t"]


def _load(name: str, **overrides) -> torch.nn.Module:
    cfg = OmegaConf.load(CONF_DIR / f"{name}.yaml")
    for key, value in overrides.items():
        cfg.params[key] = value
    return instantiate_model(cfg)


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("name", CONFS)
@pytest.mark.parametrize("length", [4000, 16001])
def test_forward_shape_and_finite(name: str, length: int) -> None:
    torch.manual_seed(0)
    model = _load(name).eval()
    x = torch.randn(2, 1, length)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, length), out.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", CONFS)
def test_codec_roundtrip(name: str) -> None:
    model = _load(name).eval()
    codec = build_codec(
        "speech_enhancement", n_channels=None, use_rps=False, predict_rps=False, sr=[16000, 1]
    )
    with torch.no_grad():
        out = codec.call_model(model, {"mixture": torch.randn(2, 16000)})
    out = out.squeeze(1)
    assert out.shape == (2, 16000), out.shape
    assert torch.isfinite(out).all()


def test_smolnet_param_count_matches_papers() -> None:
    # 2019 Table V: 224k (BatchNorm). 2024 Table I: 249k (LayerNorm over F).
    n_bn = _n_params(_load("smolnet"))
    n_ln = _n_params(_load("smolnet", norm="layer"))
    assert n_bn == 224_194
    assert n_ln == 249_180
    assert abs(n_bn - 224_000) / 224_000 < 0.01
    assert abs(n_ln - 249_000) / 249_000 < 0.01


@pytest.mark.parametrize(
    ("m", "expected", "expected_11", "published"),
    [
        (1, 182_230, 196_632, 192_000),
        (2, 232_214, 246_616, 254_000),
        (3, 282_198, 296_600, 317_000),
    ],
)
def test_smolnet_t_param_count(m: int, expected: int, expected_11: int, published: int) -> None:
    # Equations as written (the default) fall 5-11 % short of the published
    # sizes; the gap and the readings tried are in the module docstring.
    n = _n_params(_load("smolnet_t", n_transformers=m))
    assert n == expected
    assert n < published
    # Reading 3: L = log2(F) + 1 = 11 CNN layers.
    assert _n_params(_load("smolnet_t", n_transformers=m, n_cnn_layers=11)) == expected_11


@pytest.mark.parametrize("name", CONFS)
def test_backward_finite_gradients(name: str) -> None:
    torch.manual_seed(0)
    model = _load(name).train()
    x = torch.randn(2, 1, 8000)
    target = torch.randn(2, 1, 8000)
    loss = (model(x) - target).pow(2).mean()
    loss.backward()
    for pname, p in model.named_parameters():
        assert p.grad is not None, pname
        assert torch.isfinite(p.grad).all(), pname
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.parametrize("name", CONFS)
def test_state_dict_roundtrip(name: str) -> None:
    torch.manual_seed(0)
    model = _load(name).train()
    with torch.no_grad():
        model(torch.randn(2, 1, 8000))  # move BatchNorm running stats off init
    fresh = _load(name)
    fresh.load_state_dict(model.state_dict())
    model.eval()
    fresh.eval()
    x = torch.randn(1, 1, 12000)
    with torch.no_grad():
        assert torch.equal(model(x), fresh(x))


@pytest.mark.parametrize("name", CONFS)
def test_scale_equivariance(name: str) -> None:
    torch.manual_seed(0)
    model = _load(name).eval()
    x = torch.randn(1, 1, 8000)
    with torch.no_grad():
        torch.testing.assert_close(model(7.0 * x), 7.0 * model(x), rtol=1e-4, atol=1e-5)


def test_smolnet_context_blocks_are_independent() -> None:
    torch.manual_seed(0)
    model = _load("smolnet").eval()
    assert isinstance(model, SMoLnet) and model.context_frames == 9
    feats = torch.randn(1, 2, 20, model.n_freqs)  # blocks [0, 9), [9, 18), [18, 20)
    perturbed = feats.clone()
    perturbed[:, :, 9:18] += torch.randn_like(perturbed[:, :, 9:18])
    with torch.no_grad():
        a, b = model.map(feats), model.map(perturbed)
    assert a.shape == feats.shape
    assert torch.equal(a[:, :, :9], b[:, :, :9])
    assert torch.equal(a[:, :, 18:], b[:, :, 18:])
    assert not torch.allclose(a[:, :, 9:18], b[:, :, 9:18])
    # Within one block, the block-wise path equals the fully convolutional one.
    model.context_frames = None
    with torch.no_grad():
        full = model.map(feats[:, :, :9])
    torch.testing.assert_close(a[:, :, :9], full, rtol=1e-5, atol=1e-5)


def test_smolnet_body_matches_rps_backbone() -> None:
    """Layers 1-13 equal ``SMoLnetRPSBackbone`` set to Table I (layout (B,C,F,T))."""
    from models.rps_predictor import SMoLnetRPSBackbone

    torch.manual_seed(0)
    model = _load("smolnet").eval()
    assert isinstance(model, SMoLnet)
    ref = SMoLnetRPSBackbone(
        input_ch=2, inner_channels=64, dilated_layers=10, total_layers=13, max_dilation=512
    ).eval()
    with torch.no_grad():  # non-trivial BatchNorm statistics
        for bn in (m for m in model.body.modules() if isinstance(m, torch.nn.BatchNorm2d)):
            assert bn.running_mean is not None and bn.running_var is not None
            bn.running_mean.copy_(torch.randn(64) * 0.1)
            bn.running_var.copy_(torch.rand(64) + 0.5)
    mapped = {}
    for key, value in model.body.state_dict().items():
        idx, rest = key.split(".", 1)
        if int(idx) < 10:  # dilated layers: main = Sequential(conv, bn, relu)
            rest = rest.replace("conv.", "main.0.").replace("norm.", "main.1.")
        if value.dim() == 4:  # conv kernel (out, in, kT, kF) -> (out, in, kF, kT)
            value = value.transpose(2, 3)
        mapped[f"net.{idx}.{rest}"] = value
    assert set(mapped) == set(ref.state_dict())
    ref.load_state_dict(mapped)
    x = torch.randn(1, 2, 9, model.n_freqs)  # (B, 2, T, F)
    with torch.no_grad():
        ours = model.body(x)
        theirs = ref(x.transpose(2, 3)).transpose(2, 3)
    torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=1e-5)


def test_spectral_combination_transformer_matches_equations() -> None:
    """One SCT block against a direct transcription of Eqs. (27)-(37)."""
    torch.manual_seed(0)
    model = _load("smolnet_t").eval()
    assert isinstance(model, SMoLnetT)
    block = model.transformers[0]
    assert isinstance(block, SpectralCombinationTransformer)
    c, h = 64, block.mha.num_heads
    d = torch.randn(5, 11, c)
    pe = sinusoidal_positional_encoding(11, c)
    with torch.no_grad():
        got = block(d, pe)
        dp = d + pe  # (27)
        wq, wk, wv = block.mha.in_proj_weight.chunk(3)
        bq, bk, bv = block.mha.in_proj_bias.chunk(3)
        q, k, v = dp @ wq.T + bq, dp @ wk.T + bk, dp @ wv.T + bv  # (31)-(33)
        heads = []
        for i in range(h):
            s = slice(i * c // h, (i + 1) * c // h)
            att = torch.softmax(q[..., s] @ k[..., s].transpose(1, 2) / math.sqrt(c // h), -1)
            heads.append(att @ v[..., s])  # (30)
        a = torch.cat(heads, -1) @ block.mha.out_proj.weight.T + block.mha.out_proj.bias  # (29)
        d1 = F.layer_norm(a + dp, (c,), block.ln1.weight, block.ln1.bias)  # (34)
        d2 = F.relu(d1 @ block.ffn1.weight.T + block.ffn1.bias)  # (35)
        d3 = F.relu(d2 @ block.ffn2.weight.T + block.ffn2.bias)  # (36)
        want = F.layer_norm(d1 + d3, (c,), block.ln2.weight, block.ln2.bias)  # (37)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_positional_encoding_is_vaswani() -> None:
    pe = sinusoidal_positional_encoding(7, 8)
    t, i = 5, 3
    assert pe[t, 2 * i].item() == pytest.approx(math.sin(t / 10000 ** (2 * i / 8)), abs=1e-6)
    assert pe[t, 2 * i + 1].item() == pytest.approx(math.cos(t / 10000 ** (2 * i / 8)), abs=1e-6)
