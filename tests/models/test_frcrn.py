"""FRCRN backbone + frequency-adapter port (``models.frcrn``).

Builds both models from their conf yaml through ``training.config.instantiate_model``
with random weights (``checkpoint: null``), so the fast tests need no download.
The ``slow`` test loads the published ModelScope checkpoint from
``.cache/pretrained/frcrn/pytorch_model.bin`` (skipped when absent). Numerical
parity of the backbone with the ModelScope source was checked in a scratch
script (result in ``models/frcrn/backbone.py``'s docstring).
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch
from omegaconf import OmegaConf

from models.frcrn import ComplexFreqAdapter, FRCRNEnhancer
from models.frcrn.backbone import Encoder
from tasks.codecs import build_codec
from training.config import instantiate_model

_ROOT = Path(__file__).resolve().parents[2]
_ADAPTER_CONF = _ROOT / "conf" / "model" / "adapter_frcrn.yaml"
_FRCRN_CONF = _ROOT / "conf" / "model" / "frcrn.yaml"
_CKPT = _ROOT / ".cache" / "pretrained" / "frcrn" / "pytorch_model.bin"

# Released FRCRN: 13,141,844 learnable + 821,760 fixed (STFT/iSTFT kernels) tensors.
_BACKBONE_PARAMS = 13_963_604
_ADAPTER_PARAMS = 66_616  # 7 encoder-block adapters of the first U-Net
_ADAPTER2_PARAMS = 133_232  # adapt_unet2=True: both U-Nets
_ATTENTION_PARAMS = 220_480  # every SE layer of both U-Nets


def _build(conf: Path, **overrides) -> FRCRNEnhancer:
    cfg = OmegaConf.load(conf)
    for k, v in overrides.items():
        cfg.params[k] = v
    model = instantiate_model(cfg)
    assert isinstance(model, FRCRNEnhancer)
    return model


def _adapters(model: FRCRNEnhancer) -> torch.nn.ModuleList:
    assert model.adapters is not None
    return model.adapters


def _running_mean(model: FRCRNEnhancer) -> torch.Tensor:
    running_mean = cast(Encoder, model.frcrn.unet.encoder0).bn.bn_re.running_mean
    assert running_mean is not None
    return running_mean


def _randomise_adapters(model: FRCRNEnhancer) -> None:
    for p in _adapters(model).parameters():
        torch.nn.init.normal_(p, std=0.05)


@pytest.mark.parametrize("n_samples", [4000, 13_337])
def test_forward_shape_and_finite(n_samples: int) -> None:
    torch.manual_seed(0)
    model = _build(_ADAPTER_CONF).eval()
    _randomise_adapters(model)
    with torch.no_grad():
        out = model(0.1 * torch.randn(2, 1, n_samples))
    assert out.shape == (2, n_samples), out.shape
    assert torch.isfinite(out).all()


def test_codec_roundtrip() -> None:
    model = _build(_ADAPTER_CONF).eval()
    codec = build_codec(
        "speech_enhancement", n_channels=None, use_rps=False, predict_rps=False, sr=[16000, 1]
    )
    with torch.no_grad():
        out = codec.call_model(model, {"mixture": 0.1 * torch.randn(2, 16000)})
    assert out.squeeze().shape == (2, 16000), out.shape
    assert torch.isfinite(out).all()


def test_param_counts() -> None:
    model = _build(_ADAPTER_CONF)
    n_adapter = sum(p.numel() for p in _adapters(model).parameters())
    n_backbone = sum(p.numel() for p in model.frcrn.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_adapter == _ADAPTER_PARAMS
    assert n_backbone == _BACKBONE_PARAMS
    # Paper Table 2: "0.3 M" trainable (adapters + attention blocks).
    assert n_train == _ADAPTER_PARAMS + _ATTENTION_PARAMS
    plain = _build(_FRCRN_CONF)
    assert sum(p.numel() for p in plain.parameters()) == _BACKBONE_PARAMS
    both = _build(_ADAPTER_CONF, adapt_unet2=True)
    assert sum(p.numel() for p in _adapters(both).parameters()) == _ADAPTER2_PARAMS


def test_frozen_backbone_stays_in_eval_mode() -> None:
    model = _build(_ADAPTER_CONF).train()
    assert model.training and _adapters(model).training
    assert not any(m.training for m in model.frcrn.modules())
    assert all(p.requires_grad for p in _adapters(model).parameters())
    trainable = {n for n, p in model.frcrn.named_parameters() if p.requires_grad}
    assert trainable and all(".se_layer" in n for n in trainable)
    running = _running_mean(model).clone()
    model(0.1 * torch.randn(1, 1, 4000))
    assert torch.equal(running, _running_mean(model))


def test_zero_up_adapters_are_identity() -> None:
    torch.manual_seed(0)
    adapted = _build(_ADAPTER_CONF).eval()
    deltas = []
    for adapter in cast(torch.nn.ModuleList, _adapters(adapted)[0]):
        adapter.register_forward_hook(lambda m, i, o: deltas.append(o - i[0]))
    plain = _build(_FRCRN_CONF).eval()
    plain.frcrn.load_state_dict(adapted.frcrn.state_dict())
    x = 0.1 * torch.randn(2, 1, 8000)
    with torch.no_grad():
        y, ref = adapted(x), plain(x)
    # Exact: every adapter adds exactly zero at the zero-up initialisation.
    assert len(deltas) == 7 and all(torch.count_nonzero(d) == 0 for d in deltas)
    # Whole model: equal up to fp32 rounding (measured 4.4e-5 of max|y| with the checkpoint).
    torch.testing.assert_close(y, ref, rtol=5e-4, atol=5e-4 * ref.abs().max().item())


def test_backward_finite_grads_only_on_trainable() -> None:
    torch.manual_seed(0)
    model = _build(_ADAPTER_CONF, train_attention=False).train()
    assert not any(p.requires_grad for p in model.frcrn.parameters())
    assert not any(m.training for m in model.frcrn.modules())
    assert all(p.requires_grad for p in _adapters(model).parameters())
    out = model(0.1 * torch.randn(1, 1, 4000))
    out.pow(2).mean().backward()
    ups = [
        cast(ComplexFreqAdapter, m).cell_r.up.weight.grad
        for unet in _adapters(model)
        for m in cast(torch.nn.ModuleList, unet)
    ]
    assert all(g is not None and torch.isfinite(g).all() for g in ups)
    # zero_up init is not a dead point
    assert any(g is not None and g.abs().sum() > 0 for g in ups)
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all()
    assert all(p.grad is None for p in model.frcrn.parameters() if not p.requires_grad)


def test_state_dict_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _build(_ADAPTER_CONF).eval()
    _randomise_adapters(model)
    fresh = _build(_ADAPTER_CONF).eval()
    state_path = tmp_path / "frcrn.pt"
    torch.save(model.state_dict(), state_path)
    fresh.load_state_dict(torch.load(state_path, weights_only=True), strict=True)
    x = 0.1 * torch.randn(1, 1, 8000)
    with torch.no_grad():
        assert torch.equal(model(x), fresh(x))


@pytest.mark.slow
@pytest.mark.skipif(not _CKPT.exists(), reason="published FRCRN checkpoint not downloaded")
def test_published_checkpoint_loads_strict() -> None:
    torch.set_flush_denormal(True)  # the published weights hit CPU denormals
    torch.manual_seed(0)
    x = 0.1 * torch.randn(1, 1, 8000)
    model = _build(_ADAPTER_CONF, checkpoint=str(_CKPT)).eval()
    with torch.no_grad():
        y = model(x)
    del model
    plain = _build(_FRCRN_CONF, checkpoint=str(_CKPT)).eval()
    with torch.no_grad():
        ref = plain(x)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, ref, rtol=5e-4, atol=5e-4 * ref.abs().max().item())
