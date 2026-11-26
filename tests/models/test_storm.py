"""Tests for the StoRM port (``models.storm``) and its speech_enhancement wiring.

Builds the model from ``conf/model/storm.yaml`` through the exact
``training.config.instantiate_model`` dispatch a real Hydra run uses. The
parameter count is checked on the published size (2 x NCSN++M, 55.1M
parameters, no forward pass). Forward, codec, gradient and checkpoint tests
shrink the backbone width to ``nf=16`` (same code path) so they stay CPU-fast
and under 1 GB; the published-size inference smoke test is ``slow`` (about
1.8 GB peak, a minute on 4 CPU threads).
Upstream parity (sp-uhh/storm @257e9636) is a scratch script, not a test: the
upstream clone needs pytorch_lightning/torch_ema stubs this repository does not
carry.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from omegaconf import OmegaConf

from models.storm.model import StoRMModel
from tasks.codecs import build_codec
from training.config import instantiate_model

CONF = Path(__file__).resolve().parents[2] / "conf" / "model" / "storm.yaml"

# Measured on the published WSJ0+Chime3 checkpoint (epoch=148) and on this port;
# the paper (Sec. IV-B) quotes 27.8M for NCSN++M.
N_PARAMS_PREDICTOR = 27_342_350
N_PARAMS_SCORE = 27_776_038


def _load(small: bool = False, **params: Any) -> StoRMModel:
    cfg = OmegaConf.load(CONF)
    if small:
        cfg.params.backbone.nf = 16
    for key, value in params.items():
        cfg.params[key] = value
    model = instantiate_model(cfg)
    assert isinstance(model, StoRMModel)
    return model


def _num_updates(model: StoRMModel) -> int:
    return int(cast(torch.Tensor, model.ema.num_updates))


def test_param_count_matches_published() -> None:
    model = _load()
    n_pred = sum(p.numel() for p in model.denoiser_net.parameters())
    n_score = sum(p.numel() for p in model.score_net.parameters())
    assert n_pred == N_PARAMS_PREDICTOR, n_pred
    assert n_score == N_PARAMS_SCORE, n_score
    assert abs(n_score - 27.8e6) / 27.8e6 < 0.01
    assert sum(p.numel() for p in model.parameters()) == n_pred + n_score


def test_forward_shape_and_finite_two_lengths() -> None:
    torch.manual_seed(0)
    model = _load(small=True, sampler_steps=2).eval()
    for batch, length in ((2, 4000), (1, 7001)):
        with torch.no_grad():
            out = model(0.1 * torch.randn(batch, 1, length))
        assert out.shape == (batch, 1, length), out.shape
        assert torch.isfinite(out).all()


def test_codec_roundtrip() -> None:
    torch.manual_seed(0)
    model = _load(small=True, sampler_steps=1).eval()
    codec = build_codec(
        "speech_enhancement",
        n_channels=None,
        use_rps=False,
        predict_rps=False,
        sr=[16000, 1],
    )
    inputs = {"mixture": 0.1 * torch.randn(1, 8000)}
    with torch.no_grad():
        out = codec.call_model(model, inputs)
    out = out.squeeze()
    assert out.shape == (8000,), out.shape
    assert torch.isfinite(out).all()


@pytest.mark.slow
def test_published_size_inference() -> None:
    torch.manual_seed(0)
    model = _load(sampler_steps=1).eval()
    with torch.no_grad():
        out = model(0.1 * torch.randn(1, 1, 4000))
    assert out.shape == (1, 1, 4000), out.shape
    assert torch.isfinite(out).all()


def test_backward_finite_grads_and_ema_update() -> None:
    torch.manual_seed(0)
    model = _load(small=True).train()
    mix = 0.1 * torch.randn(1, 1, 4000)
    target = 0.05 * torch.randn(1, 1, 4000)
    loss = model(mix, target)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    for net in (model.denoiser_net, model.score_net):
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        assert grads, "no gradient reached a network"
        assert all(torch.isfinite(g).all() for g in grads)
    assert _num_updates(model) == 1

    # one optimizer step, then the next training forward pulls the EMA toward it
    torch.optim.Adam(model.parameters(), lr=1e-3).step()
    model(mix, target)
    assert _num_updates(model) == 2
    name, param = next((n, p) for n, p in model.named_parameters() if p.requires_grad)
    shadow = model.ema.shadow(name)
    assert not torch.equal(shadow, param.detach())

    # no EMA update without gradients (validation loss)
    with torch.no_grad():
        model(mix, target)
    assert _num_updates(model) == 2


def test_state_dict_roundtrip_keeps_ema() -> None:
    torch.manual_seed(0)
    model = _load(small=True, sampler_steps=2)
    # make the EMA weights differ from the live weights
    with torch.no_grad():
        for name, _ in model.named_parameters():
            shadow = model.ema.shadow(name)
            shadow.add_(0.01 * torch.randn_like(shadow))
    sd = model.state_dict()
    assert any(k.startswith("ema.") for k in sd)
    assert "ema.num_updates" in sd
    n_param_keys = sum(1 for _ in model.named_parameters())
    assert sum(1 for k in sd if k.startswith("ema.") and k != "ema.num_updates") == n_param_keys

    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)
    fresh = _load(small=True, sampler_steps=2)
    fresh.load_state_dict(torch.load(buf, weights_only=True), strict=True)

    mix = 0.1 * torch.randn(1, 1, 4000)
    model.eval()
    fresh.eval()
    with torch.no_grad():
        torch.manual_seed(123)
        out_a = model(mix)
        torch.manual_seed(123)
        out_b = fresh(mix)
        fresh.use_ema = False
        torch.manual_seed(123)
        out_raw = fresh(mix)
    assert torch.equal(out_a, out_b)
    # eval runs on the EMA weights, not on the live ones
    assert not torch.allclose(out_b, out_raw)
