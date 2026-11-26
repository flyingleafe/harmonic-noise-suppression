"""Tests for the official-release MP-SENet port (``models.mpsenet_official``).

Builds the generator from ``conf/model/mpsenet_official.yaml`` through
``training.config.instantiate_model`` (the Hydra path) and covers the SE
contract, the parameter counts of upstream's release, the losses, gradient
flow, state-dict round trips, and the batch independence of the default
transformer layout. Slow tests (``-m slow``) pin upstream's output for both
released checkpoints (skipped when the git-ignored files are not present under
``.cache/pretrained/mpsenet_official/``) and, when the upstream clone exists,
run upstream itself in a subprocess on the same weights and inputs and compare
the generator, the discriminator and every loss term.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from models.mpsenet_official import MetricDiscriminator, MPSENetOfficial
from models.mpsenet_official.losses import (
    LossWeights,
    anti_wrapping_function,
    discriminator_loss,
    metric_loss,
    mpsenet_losses,
    phase_losses,
)
from tasks.codecs import build_codec
from training.config import instantiate_model

ROOT = Path(__file__).resolve().parents[2]
CONF = ROOT / "conf" / "model" / "mpsenet_official.yaml"
CKPT_DIR = ROOT / ".cache" / "pretrained" / "mpsenet_official"
# Scratch clone of https://github.com/yxlu-0102/MP-SENet @ 89932cf (MIT).
UPSTREAM = Path(os.environ.get("MPSENET_UPSTREAM", "/tmp/mpsenet_official_upstream"))


def _load_model() -> MPSENetOfficial:
    torch.manual_seed(0)
    model = instantiate_model(OmegaConf.load(CONF))
    assert isinstance(model, MPSENetOfficial)
    return model


def _speechish(n: int, seed: int = 0, batch: int = 2) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n) / 16000
    tone = 0.3 * torch.sin(2 * math.pi * 220 * t) * (1 + torch.sin(2 * math.pi * 3 * t))
    return tone + 0.05 * torch.randn(batch, n, generator=g)


@pytest.mark.parametrize("n", [4000, 16000 + 37])
def test_forward_shape_and_finite(n: int) -> None:
    model = _load_model().eval()
    x = _speechish(n).unsqueeze(1)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, n), out.shape
    assert torch.isfinite(out).all()


def test_codec_roundtrip() -> None:
    model = _load_model().eval()
    codec = build_codec(
        "speech_enhancement", n_channels=None, use_rps=False, predict_rps=False, sr=[16000, 1]
    )
    inputs = {"mixture": _speechish(16000)}
    with torch.no_grad():
        out = codec.call_model(model, inputs)
    out = out.squeeze()
    assert out.shape == (2, 16000), out.shape
    assert torch.isfinite(out).all()


def test_param_counts_match_upstream_release() -> None:
    model = _load_model()
    assert sum(p.numel() for p in model.parameters()) == 2_263_372
    assert sum(p.numel() for p in MetricDiscriminator().parameters()) == 181_650


def test_per_utterance_layout_is_batch_independent() -> None:
    """Default layout: each utterance's output equals its single-utterance output;
    the upstream layout couples the utterances of a batch (upstream behaviour)."""
    torch.manual_seed(0)
    pu = MPSENetOfficial(transformer_layout="per_utterance").eval()
    up = MPSENetOfficial(transformer_layout="upstream").eval()
    up.load_state_dict(pu.state_dict())
    x = _speechish(8000).unsqueeze(1)
    with torch.no_grad():
        both = pu(x)
        alone = torch.cat([pu(x[:1]), pu(x[1:])])
        up_alone = up(x[:1])
        up_both = up(x)
    # Measured against upstream: per-utterance vs upstream layout differ by <= 8e-7
    # (fp32 matmul batching); upstream's B = 2 coupling moved outputs by 3.9e-2.
    torch.testing.assert_close(both, alone, rtol=1e-5, atol=5e-6)
    torch.testing.assert_close(up_alone, alone[:1], rtol=1e-5, atol=5e-6)
    assert (up_both[:1] - up_alone).abs().max() > 1e-3


def test_backward_finite_gradients() -> None:
    model = _load_model().train()
    disc = MetricDiscriminator().train()
    noisy = _speechish(8000, seed=1)
    clean = _speechish(8000, seed=2)
    res = mpsenet_losses(model, disc, noisy, clean, pesq_target=torch.tensor([0.4, 0.6]))
    assert torch.isfinite(res.generator) and torch.isfinite(res.discriminator)
    disc.zero_grad()
    res.discriminator.backward(retain_graph=True)
    assert all(p.grad is None for p in model.parameters())
    d_grads = [p.grad for p in disc.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in d_grads)
    disc.zero_grad()
    res.generator.backward()
    g_grads = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in g_grads)
    assert any(g is not None and g.abs().sum() > 0 for g in g_grads)


def test_pesq_target_none_drops_fake_term() -> None:
    model = _load_model().eval()
    disc = MetricDiscriminator().eval()
    with torch.no_grad():
        res = mpsenet_losses(
            model, disc, _speechish(4000, 1), _speechish(4000, 2), pesq_target=None
        )
        assert res.batch is not None
        metric_r = disc(res.batch.clean_mag, res.batch.clean_mag)
    torch.testing.assert_close(res.discriminator, ((metric_r.flatten() - 1) ** 2).mean())


def test_loss_terms_definitions() -> None:
    g = torch.Generator().manual_seed(3)
    x = (torch.rand(1000, generator=g) * 2 - 1) * 20
    aw = anti_wrapping_function(x)
    assert (aw >= 0).all() and (aw <= math.pi + 1e-5).all()
    torch.testing.assert_close(anti_wrapping_function(x + 2 * math.pi), aw, atol=1e-4, rtol=0)
    pha = (torch.rand(2, 201, 20, generator=g) * 2 - 1) * math.pi
    for term in phase_losses(pha, pha + 2 * math.pi):
        assert term.abs() < 1e-5
    assert metric_loss(torch.ones(3, 1)) == 0
    assert (
        discriminator_loss(torch.ones(2, 1), torch.full((2, 1), 0.5), torch.tensor([0.5, 0.5])) == 0
    )
    assert LossWeights() == LossWeights(0.9, 0.3, 0.1, 0.1, 0.05, 0.2)


def test_state_dict_roundtrip() -> None:
    model = _load_model().eval()
    disc = MetricDiscriminator().train()
    m = torch.rand(2, 201, 40, generator=torch.Generator().manual_seed(4))
    disc(m, m)  # one power iteration moves the spectral-norm u/v buffers
    fresh = MPSENetOfficial().eval()
    fresh.load_state_dict(model.state_dict())
    fresh_d = MetricDiscriminator()
    fresh_d.load_state_dict(disc.state_dict())
    disc.eval()
    fresh_d.eval()
    x = _speechish(8000, 5).unsqueeze(1)
    with torch.no_grad():
        torch.testing.assert_close(fresh(x), model(x), rtol=0, atol=0)
        torch.testing.assert_close(fresh_d(m, m * 0.5), disc(m, m * 0.5), rtol=0, atol=0)


# Upstream (yxlu-0102/MP-SENet @ 89932cf, inference.py: RMS norm, STFT, MPNet,
# iSTFT, de-norm; B = 1) run with each released checkpoint on
# ``_speechish(7950, seed=9, batch=1)``: (sum, sum |y|, y[1000], y[4321]) of its
# 7900-sample output, measured in fp32 on CPU (torch 2.7.0). The port's full output
# differed from upstream's by at most 3.6e-7 (vb) / 4.8e-7 (dns) absolute and these
# statistics by at most 1.7e-6 relative, so rtol 1e-5 / atol 1e-6 leaves ~5x margin.
_UPSTREAM_REF = {
    "g_best_vb": (3.3091682923009103, 596.742293646304, -0.153391495347023, -0.0034962573554366827),
    "g_best_dns": (2.737537311966662, 651.2300491033147, -0.2008991688489914, 0.004829993937164545),
}


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(_UPSTREAM_REF))
def test_published_checkpoint_matches_upstream(name: str) -> None:
    ckpt = CKPT_DIR / name
    if not ckpt.exists():
        pytest.skip(f"published checkpoint {ckpt} not present")
    model = MPSENetOfficial(checkpoint=str(ckpt)).eval()
    x = _speechish(7950, seed=9, batch=1)
    with torch.no_grad():
        y = model(x.unsqueeze(1))[0, 0, :7900].double()
    got = torch.stack([y.sum(), y.abs().sum(), y[1000], y[4321]])
    want = torch.tensor(_UPSTREAM_REF[name], dtype=torch.float64)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


# Runs inside the upstream clone (its ``models`` is a namespace package that would
# clash with ours, hence a subprocess). Transcribes upstream inference.py and the
# loss part of train.py (fixed PESQ target instead of joblib-parallel PESQ).
_UPSTREAM_SCRIPT = r"""
import json, sys, types
up, io_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, up)
pkg = types.ModuleType("models")
pkg.__path__ = [up + "/models"]
sys.modules["models"] = pkg
import torch
import torch.nn.functional as F
from dataset import mag_pha_istft, mag_pha_stft
from env import AttrDict
from models.discriminator import MetricDiscriminator
from models.model import MPNet, phase_losses

h = AttrDict(json.load(open(up + "/config.json")))
io = torch.load(io_path)
g = MPNet(h)
g.load_state_dict(io["g_sd"])
d = MetricDiscriminator()
d.load_state_dict(io["d_sd"])
stft = lambda y: mag_pha_stft(y, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
istft = lambda m, p: mag_pha_istft(m, p, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

def infer(wav):
    norm = torch.sqrt(wav.shape[-1] / torch.sum(wav ** 2.0, dim=-1, keepdim=True))
    amp, pha, _ = stft(wav * norm)
    a, p, _ = g(amp, pha)
    return istft(a, p) / norm

out = {}
g.eval()
with torch.no_grad():
    x = io["x"]
    out["y"] = torch.cat([infer(x[i : i + 1]) for i in range(x.shape[0])])
    out["y_batch"] = infer(x)
    d.train()
    torch.manual_seed(5)
    out["d_train"] = d(io["ma"], io["mb"])
    d.eval()
    out["d_eval"] = d(io["ma"], io["mb"])
    d.load_state_dict(io["d_sd"])
    clean, noisy = io["clean"], io["noisy"]
    norm = torch.sqrt(noisy.shape[-1] / torch.sum(noisy ** 2.0, dim=-1, keepdim=True))
    clean_audio, noisy_audio = clean * norm, noisy * norm
    g.train()
    d.train()
    torch.manual_seed(11)
    one = torch.ones(clean.shape[0])
    clean_mag, clean_pha, clean_com = stft(clean_audio)
    noisy_mag, noisy_pha, _ = stft(noisy_audio)
    mag_g, pha_g, com_g = g(noisy_mag, noisy_pha)
    audio_g = istft(mag_g, pha_g)
    mag_g_hat, _, com_g_hat = stft(audio_g)
    metric_r = d(clean_mag, clean_mag)
    metric_g = d(clean_mag, mag_g_hat.detach())
    disc = F.mse_loss(one, metric_r.flatten()) + F.mse_loss(io["pesq_target"], metric_g.flatten())
    ip, gd, iaf = phase_losses(clean_pha, pha_g)
    out["losses"] = {
        "discriminator": disc,
        "magnitude": F.mse_loss(clean_mag, mag_g),
        "phase_ip": ip,
        "phase_gd": gd,
        "phase_iaf": iaf,
        "complex": F.mse_loss(clean_com, com_g) * 2,
        "consistency": F.mse_loss(com_g, com_g_hat) * 2,
        "time": F.l1_loss(clean_audio, audio_g),
        "metric": F.mse_loss(d(clean_mag, mag_g_hat).flatten(), one),
    }
    t = out["losses"]
    out["losses"]["generator"] = (
        t["magnitude"] * 0.9 + (ip + gd + iaf) * 0.3 + t["complex"] * 0.1
        + t["consistency"] * 0.1 + t["metric"] * 0.05 + t["time"] * 0.2
    )
torch.save(out, io_path + ".out")
"""


@pytest.mark.slow
def test_upstream_clone_parity(tmp_path: Path) -> None:
    """Same weights, same inputs, same dropout seed: port vs upstream code.

    Measured (scratch parity run, fp32 CPU): generator outputs agree to <= 1.1e-6
    absolute (outputs ~0.5), the train-mode discriminator bit-exactly, every loss
    term to <= 3.6e-6 relative; tolerances below leave ~10x margin.
    """
    if not (UPSTREAM / "models" / "model.py").exists():
        pytest.skip(f"upstream clone {UPSTREAM} not present")
    torch.manual_seed(0)
    pu = MPSENetOfficial(transformer_layout="per_utterance").eval()
    up = MPSENetOfficial(transformer_layout="upstream").eval()
    up.load_state_dict(pu.state_dict())
    disc = MetricDiscriminator()
    g = torch.Generator().manual_seed(2)
    io = {
        "g_sd": pu.generator.state_dict(),
        "d_sd": {k: v.clone() for k, v in disc.state_dict().items()},  # u/v move in train()
        "x": _speechish(8000, seed=7),
        "ma": torch.rand(2, 201, 41, generator=g),
        "mb": torch.rand(2, 201, 41, generator=g),
        "clean": _speechish(4000, seed=8),
        "noisy": _speechish(4000, seed=8) + 0.2 * torch.randn(2, 4000, generator=g),
        "pesq_target": torch.tensor([0.3, 0.45]),
    }
    io_path = tmp_path / "io.pt"
    torch.save(io, io_path)
    script = tmp_path / "upstream_ref.py"
    script.write_text(_UPSTREAM_SCRIPT)
    subprocess.run(
        [sys.executable, str(script), str(UPSTREAM), str(io_path)],
        check=True,
        capture_output=True,
        timeout=900,
    )
    ref = torch.load(str(io_path) + ".out")

    with torch.no_grad():
        n = ref["y"].shape[-1]
        torch.testing.assert_close(
            pu(io["x"].unsqueeze(1))[:, 0, :n], ref["y"], rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            up(io["x"].unsqueeze(1))[:, 0, :n], ref["y_batch"], rtol=1e-5, atol=1e-5
        )
        disc.train()
        torch.manual_seed(5)
        torch.testing.assert_close(disc(io["ma"], io["mb"]), ref["d_train"], rtol=1e-6, atol=1e-7)
        disc.eval()
        torch.testing.assert_close(disc(io["ma"], io["mb"]), ref["d_eval"], rtol=1e-6, atol=1e-7)
        disc.load_state_dict(io["d_sd"])
        up.train()
        disc.train()
        torch.manual_seed(11)
        res = mpsenet_losses(up, disc, io["noisy"], io["clean"], pesq_target=io["pesq_target"])
    torch.testing.assert_close(
        res.discriminator, ref["losses"]["discriminator"], rtol=3e-5, atol=1e-7
    )
    torch.testing.assert_close(res.generator, ref["losses"]["generator"], rtol=3e-5, atol=1e-7)
    for k in [
        "magnitude",
        "phase_ip",
        "phase_gd",
        "phase_iaf",
        "complex",
        "consistency",
        "time",
        "metric",
    ]:
        torch.testing.assert_close(res.terms[k], ref["losses"][k], rtol=3e-5, atol=1e-7, msg=k)
