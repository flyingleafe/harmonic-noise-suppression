"""Tests for the DroFiT SE baseline (``src/models/drofit.py``).

Builds the model from ``conf/model/drofit.yaml`` through the exact
``training.config.instantiate_model`` dispatch a Hydra run uses. CPU-fast:
signals of at most 1 s, a 168k-parameter model.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from tasks.codecs import build_codec
from training.config import instantiate_model

CONF = Path(__file__).resolve().parents[2] / "conf" / "model" / "drofit.yaml"
PAPER_PARAMS = 168_000  # Interspeech 2026, Table 3: 0.168 M (base, Lite and Quant)
HOP = 512


def _load_model(**overrides) -> torch.nn.Module:
    cfg = OmegaConf.load(CONF)
    for key, value in overrides.items():
        cfg.params[key] = value
    return instantiate_model(cfg)


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("n_samples", [4000, 16000])
def test_forward_shape_and_finite(n_samples: int) -> None:
    torch.manual_seed(0)
    model = _load_model().eval()
    x = torch.randn(2, 1, n_samples)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, n_samples), out.shape
    assert torch.isfinite(out).all()


def test_codec_roundtrip() -> None:
    torch.manual_seed(0)
    model = _load_model().eval()
    codec = build_codec(
        "speech_enhancement",
        n_channels=None,
        use_rps=False,
        predict_rps=False,
        sr=[16000, 1],
    )
    inputs = {"mixture": torch.randn(2, 16000)}
    with torch.no_grad():
        out = codec.call_model(model, inputs)
    out = out.squeeze()
    assert out.shape == (2, 16000), out.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("attention", ["softmax", "linear"])
def test_param_count_matches_paper(attention: str) -> None:
    """Base and Lite share one weight set: 0.168 M in Table 3 for both."""
    n_params = _n_params(_load_model(attention=attention))
    assert n_params == 167_884, n_params
    assert abs(n_params - PAPER_PARAMS) / PAPER_PARAMS < 0.02


def test_backward_finite_grads() -> None:
    torch.manual_seed(0)
    model = _load_model().train()
    x = torch.randn(2, 1, 8000)
    target = torch.randn(2, 1, 8000)
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()
    params = [p for p in model.parameters() if p.requires_grad]
    grads = [p.grad for p in params if p.grad is not None]
    assert len(grads) == len(params), "a parameter received no gradient"
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_lite_forward() -> None:
    torch.manual_seed(0)
    model = _load_model(attention="linear").eval()
    x = torch.randn(1, 1, 8000)
    with torch.no_grad():
        out = model(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_state_dict_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _load_model()
    # A few train steps so BatchNorm running stats and weights leave their init.
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(2):
        x = torch.randn(2, 1, 8000)
        loss = model(x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    path = tmp_path / "drofit.pt"
    torch.save(model.state_dict(), path)

    torch.manual_seed(1)
    fresh = _load_model()
    fresh.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    fresh.eval()
    x = torch.randn(1, 1, 8000)
    with torch.no_grad():
        assert torch.equal(model(x), fresh(x))


def test_receptive_field_matches_table2() -> None:
    """Table 2: DroFiT (L=1, M=2) has a 288 ms receptive field = 9 frames of
    32 ms with 128 ms = 4 frames look-ahead. A sample at index ``s`` enters STFT
    frames within one hop of it, so with the 4-frame symmetric temporal context
    and the 1024-sample synthesis window, output samples farther than
    (4 + 2) hops from ``s`` must not move."""
    torch.manual_seed(0)
    model = _load_model().eval()
    x = torch.randn(1, 1, 16000)
    s = 8000
    x2 = x.clone()
    x2[..., s] += 1.0
    with torch.no_grad():
        diff = (model(x) - model(x2)).abs()[0, 0]
    reach = (4 + 2) * HOP
    assert diff[: s - reach].max() < 1e-6, diff[: s - reach].max()
    assert diff[s + reach :].max() < 1e-6, diff[s + reach :].max()
    # ... while the Post-TCN context still spreads the change past the frames
    # holding the perturbed sample (> 2 hops away on both sides).
    assert diff[: s - 2 * HOP].max() > 1e-7
    assert diff[s + 2 * HOP :].max() > 1e-7
