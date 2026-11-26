"""DroneAudioNet port (``models.droneaudionet``).

Builds the model from ``conf/model/droneaudionet.yaml`` through
``training.config.instantiate_model``. The fast tests use a reduced-width U-Net
(same topology, STFT, FiLM conditioning and 6-channel residual head) and random
weights; the full-size forward/backward and the two published checkpoints
(``.cache/pretrained/droneaudionet/*.ckpt``, skipped when absent) are ``slow``.
Numerical parity against the upstream module was checked in a scratch script
(see the module docstring).
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
import torch
from omegaconf import OmegaConf

from models.droneaudionet.model import _QUERY_EMBEDDING_FILE, DroneAudioNet
from tasks.codecs import build_codec
from training.config import instantiate_model

_ROOT = Path(__file__).resolve().parents[2]
_CONF = _ROOT / "conf" / "model" / "droneaudionet.yaml"
_CKPT_DIR = _ROOT / ".cache" / "pretrained" / "droneaudionet"
_DAN_CKPT = _CKPT_DIR / "droneaudionet.ckpt"
_BASE_CKPT = _CKPT_DIR / "audiosep_base_4M_steps.ckpt"

_FULL_PARAMS = 26_448_042  # paper App. A.2: "26.4M" trainable (the ResUNet)
_SMALL = {"channels": [4, 4, 8, 8, 8, 8, 8]}


def _build(**overrides) -> DroneAudioNet:
    cfg = OmegaConf.load(_CONF)
    for k, v in overrides.items():
        cfg.params[k] = v
    model = instantiate_model(cfg)
    assert isinstance(model, DroneAudioNet)
    return model


def _perturb_batchnorm(model: torch.nn.Module) -> None:
    # Fresh BN/zero-bias init makes the masks near-constant; perturb so outputs differ.
    for name, p in model.named_parameters():
        if name.startswith("base.") and "bn" in name:
            torch.nn.init.normal_(p, mean=1.0 if name.endswith("weight") else 0.0, std=0.1)


@pytest.mark.parametrize("n_samples", [4000, 13_337])
def test_forward_shape_and_finite(n_samples: int) -> None:
    torch.manual_seed(0)
    model = _build(**_SMALL).eval()
    with torch.no_grad():
        out = model(0.1 * torch.randn(2, 1, n_samples))
    assert out.shape == (2, 1, n_samples), out.shape
    assert torch.isfinite(out).all()


def test_codec_roundtrip() -> None:
    model = _build(**_SMALL).eval()
    codec = build_codec(
        "speech_enhancement", n_channels=None, use_rps=False, predict_rps=False, sr=[16000, 1]
    )
    with torch.no_grad():
        out = codec.call_model(model, {"mixture": 0.1 * torch.randn(2, 16000)})
    out = out.squeeze()
    assert out.shape == (2, 16000), out.shape
    assert torch.isfinite(out).all()


def test_param_count_matches_paper() -> None:
    model = _build()
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == n_train == _FULL_PARAMS, (n_params, n_train)
    assert round(n_params / 1e6, 1) == 26.4


def test_query_embedding_is_fixed_buffer() -> None:
    model = _build(**_SMALL)
    stored = torch.load(_QUERY_EMBEDDING_FILE, map_location="cpu", weights_only=True)
    assert stored.shape == (512,)
    assert torch.isclose(stored.norm(), torch.tensor(1.0), atol=1e-4)
    buffers = dict(model.named_buffers())
    assert torch.equal(buffers["query_embedding"], stored.reshape(1, -1))
    assert all(p is not buffers["query_embedding"] for p in model.parameters())
    assert "query_embedding" not in model.state_dict()  # constant, not a learned weight


def test_import_needs_no_clap_stack() -> None:
    banned = ("transformers", "laion_clap", "torchlibrosa", "models.CLAP")
    code = (
        "import sys, models.droneaudionet as m; m.build_droneaudionet(channels=[4,4,8,8,8,8,8]);"
        f"bad = [b for b in {banned!r} if any(k == b or k.startswith(b + '.') for k in sys.modules)];"
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_backward_finite_gradients() -> None:
    torch.manual_seed(0)
    model = _build(**_SMALL).train()
    x = 0.1 * torch.randn(2, 1, 8000)
    clean = 0.1 * torch.randn(2, 1, 8000)
    loss = (model(x) - clean).abs().mean()
    loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert grads, "no gradients reached the parameters"
    assert all(torch.isfinite(g).all() for g in grads.values())
    for name in (
        "base.alpha",
        "base.beta",
        "base.after_conv.weight",
        "film.encoder_block1->conv_block1->beta1.weight",
    ):
        assert name in grads, name


def test_state_dict_roundtrip() -> None:
    torch.manual_seed(0)
    model = _build(**_SMALL)
    _perturb_batchnorm(model)
    model.eval()
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    torch.manual_seed(1)
    fresh = _build(**_SMALL).eval()
    fresh.load_state_dict(torch.load(buf, weights_only=True), strict=True)
    x = 0.1 * torch.randn(1, 1, 8000)
    with torch.no_grad():
        assert torch.equal(model(x), fresh(x))


def test_audiosep_head_warm_start() -> None:
    """A 3-channel AudioSep-style state dict loads with the paper's 6-channel warm start."""
    torch.manual_seed(0)
    src = _build(**_SMALL)
    upstream = {
        f"ss_model.{k}": v.clone()
        for k, v in src.state_dict().items()
        if k.startswith(("base.", "film."))
    }
    upstream = {k: v for k, v in upstream.items() if not k.endswith(("base.alpha", "base.beta"))}
    upstream["ss_model.base.after_conv.weight"] = upstream["ss_model.base.after_conv.weight"][:3]
    upstream["ss_model.base.after_conv.bias"] = torch.tensor([0.1, 0.2, 0.3])
    upstream["ss_model.base.stft.conv_real.weight"] = torch.zeros(1)  # frozen DFT kernels: dropped
    upstream["query_encoder.model.text_branch.x"] = torch.zeros(1)  # CLAP: dropped
    dst = _build(**_SMALL)
    dst.load_upstream_state_dict(upstream)
    w, b = dst.base.after_conv.weight, dst.base.after_conv.bias
    assert b is not None
    assert torch.equal(w[:3], upstream["ss_model.base.after_conv.weight"])
    assert torch.equal(b, torch.tensor([0.1, 0.2, 0.3, 0.0, 0.0, 0.0]))
    assert w[3:].abs().sum() > 0
    assert dst.base.alpha.item() == dst.base.beta.item() == 1.0
    assert torch.equal(
        _conv1_weight(dst.base, "encoder_block3"),
        _conv1_weight(src.base, "encoder_block3"),
    )


def _conv1_weight(base: torch.nn.Module, block: str) -> torch.Tensor:
    return cast(torch.nn.Conv2d, base.get_submodule(f"{block}.conv_block1.conv1")).weight


@pytest.mark.slow
def test_full_size_forward_backward() -> None:
    torch.manual_seed(0)
    model = _build().train()
    x = 0.1 * torch.randn(1, 1, 8000)
    out = model(x)
    assert out.shape == (1, 1, 8000)
    out.abs().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.slow
@pytest.mark.skipif(not _DAN_CKPT.exists(), reason="droneaudionet.ckpt not downloaded")
def test_published_droneaudionet_checkpoint_loads() -> None:
    model = _build(checkpoint=str(_DAN_CKPT)).eval()
    # Learned scalars reported in the paper (Table 2 / App.): alpha = 1.0196, beta = 0.9743.
    assert abs(model.base.alpha.item() - 1.0196) < 1e-4
    assert abs(model.base.beta.item() - 0.9743) < 1e-4
    torch.manual_seed(0)
    x = 0.1 * torch.randn(1, 1, 16000)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.slow
@pytest.mark.skipif(not _BASE_CKPT.exists(), reason="audiosep_base_4M_steps.ckpt not downloaded")
def test_audiosep_base_checkpoint_warm_starts() -> None:
    model = _build(checkpoint=str(_BASE_CKPT)).eval()
    assert model.base.alpha.item() == model.base.beta.item() == 1.0
    bias = model.base.after_conv.bias
    assert bias is not None
    assert torch.equal(bias[3:], torch.zeros(3))
    torch.manual_seed(0)
    x = 0.1 * torch.randn(1, 1, 16000)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
