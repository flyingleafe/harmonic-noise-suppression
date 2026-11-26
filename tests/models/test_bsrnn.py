"""Tests for the URGENT 2026 BSRNN port (``models.bsrnn``) at 16 kHz.

Builds the model from ``conf/model/bsrnn.yaml`` through
``training.config.instantiate_model`` (the Hydra dispatch) and exercises the
``SpeechEnhancementCodec`` path. The slow tests load the published checkpoint
and compare the port against the upstream URGENT/ESPnet module; the latter
needs a scratch clone (``BSRNN_UPSTREAM``, default ``/tmp/bsrnn_upstream``)
holding ``baseline_code/``, the ESPnet ``v.202412`` sources under
``espnet_src/`` and import stubs (``torch_complex``, ``typeguard``,
``espnet.nets``) under ``stubs/``, and skips when it is absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from models.bsrnn import (
    BSRNN,
    BSRNNSE,
    SUBBANDS_16K,
    SUBBANDS_48K,
    build_bsrnn,
    convert_urgent_state_dict,
    load_urgent_checkpoint,
)
from tasks.codecs import build_codec
from training.config import instantiate_model

ROOT = Path(__file__).resolve().parents[2]
CONF = ROOT / "conf" / "model" / "bsrnn.yaml"
CKPT = ROOT / ".cache" / "pretrained" / "bsrnn" / "bsrnn.ckpt"
UPSTREAM = Path(os.environ.get("BSRNN_UPSTREAM", "/tmp/bsrnn_upstream"))


def _load_model() -> torch.nn.Module:
    return instantiate_model(OmegaConf.load(CONF))


@pytest.mark.parametrize("n", [4000, 7777])
def test_forward_shape_and_finite(n: int) -> None:
    model = _load_model().eval()
    x = 0.1 * torch.randn(2, 1, n)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, n), out.shape
    assert torch.isfinite(out).all()


def test_codec_roundtrip() -> None:
    model = _load_model().eval()
    codec = build_codec(
        "speech_enhancement",
        n_channels=None,
        use_rps=False,
        predict_rps=False,
        sr=[16000, 1],
    )
    inputs = {"mixture": 0.1 * torch.randn(2, 16000)}
    with torch.no_grad():
        out = codec.call_model(model, inputs)
    out = out.squeeze(1)
    assert out.shape == (2, 16000), out.shape
    assert torch.isfinite(out).all()


def test_param_count_matches_upstream() -> None:
    model = _load_model()
    n_params = sum(p.numel() for p in model.parameters())
    n_groupnorm = sum(
        p.numel()
        for mod in model.modules()
        if isinstance(mod, torch.nn.GroupNorm)
        for p in mod.parameters()
    )
    # Upstream's 16 kHz 27-band module, counted exactly.
    assert n_params == 33_628_912
    # BSRNN_baseline.yaml: "Parameters: 32.0456657409668 M" at 16 kHz, in units
    # of 2**20 and without GroupNorm affines.
    assert n_params - n_groupnorm == round(32.0456657409668 * 2**20) == 33_602_316


def test_backward_finite_grads() -> None:
    model = _load_model().train()
    x = 0.1 * torch.randn(1, 1, 4000)
    model(x).pow(2).mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_state_dict_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    src = _load_model().eval()
    torch.save(src.state_dict(), tmp_path / "bsrnn.pt")
    torch.manual_seed(1)
    dst = _load_model().eval()
    dst.load_state_dict(torch.load(tmp_path / "bsrnn.pt", weights_only=True), strict=True)
    x = 0.1 * torch.randn(1, 1, 4000)
    with torch.no_grad():
        torch.testing.assert_close(dst(x), src(x), rtol=0, atol=0)


def test_backbone_standalone() -> None:
    net = BSRNN(subbands=SUBBANDS_16K, num_channel=8, num_layer=1).eval()
    spec = torch.randn(2, 5, 161, 2)
    with torch.no_grad():
        out = net(spec)
    assert out.shape == (2, 1, 5, 161, 2)
    with pytest.raises(ValueError):
        net(torch.randn(2, 5, 141, 2))  # last band would hold no real bin
    with pytest.raises(ValueError):
        net(torch.randn(2, 5, 182, 2))


@pytest.mark.slow
@pytest.mark.skipif(not CKPT.exists(), reason="URGENT bsrnn.ckpt not downloaded")
def test_pretrained_checkpoint_loads() -> None:
    cfg = OmegaConf.load(CONF)
    cfg.params.checkpoint = str(CKPT)
    model = instantiate_model(cfg).eval()
    assert isinstance(model, BSRNNSE)
    t = torch.arange(8000) / 16000
    x = (0.3 * torch.sin(2 * torch.pi * 440 * t) + 0.05 * torch.randn(8000)).view(1, 1, -1)
    with torch.no_grad():
        out = model(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def _assert_parity(port: torch.nn.Module, up: torch.nn.Module, x: torch.Tensor) -> None:
    n = x.shape[-1]
    with torch.no_grad():
        ref, _ = up(x, torch.full((x.shape[0],), n), 16000)
        got = port(x.unsqueeze(1)).squeeze(1)
    assert got.shape == ref.shape
    # Same ops in the same order: agreement is expected to fp32 rounding.
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5 * ref.abs().max().item())


@pytest.mark.slow
@pytest.mark.skipif(
    not (UPSTREAM / "baseline_code" / "models" / "bsrnn.py").exists(),
    reason="URGENT upstream scratch clone absent",
)
def test_upstream_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    for sub in ("stubs", "espnet_src", "."):
        monkeypatch.syspath_prepend(str(UPSTREAM / sub))
    from baseline_code.models.bsrnn import BSRNN_SE

    torch.manual_seed(0)
    up = BSRNN_SE(num_channel=196, num_layer=6).eval()
    port = BSRNNSE(num_channel=196, num_layer=6).eval()
    port.bsrnn.load_state_dict(
        convert_urgent_state_dict(up.state_dict(), prefix="bsrnn.bsrnn."), strict=True
    )
    for n in (4000, 7777):
        _assert_parity(port, up, 0.1 * torch.randn(2, n))

    if not CKPT.exists():
        return
    del port
    state = load_urgent_checkpoint(str(CKPT), n_bands=len(SUBBANDS_48K))
    up.bsrnn.bsrnn.load_state_dict(state, strict=True)
    del state
    port = build_bsrnn(checkpoint=str(CKPT)).eval()
    t = torch.arange(7777) / 16000
    x = torch.stack(
        [
            0.3 * torch.sin(2 * torch.pi * (200 + 1500 * t) * t) + 0.05 * torch.randn(7777),
            0.1 * torch.randn(7777),
        ]
    )
    _assert_parity(port, up, x)
