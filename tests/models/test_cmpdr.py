"""Tests for the cMPDR + DNN port (``models.cmpdr``) and its configs.

Builds every model through ``training.config.instantiate_model`` on
``conf/model/cmpdr_{crn,ulcnet,filter}.yaml`` (the path a Hydra run uses).
CPU-fast: signals of at most 1 s.

The ``slow`` tests compare against upstream and skip cleanly when the
reference material is absent:

* ``CMVDR_UPSTREAM`` (default ``/tmp/cmvdr_upstream``): a clone of
  github.com/Screeen/cmvdr @fe4386cda30b; the filter is compared with
  ``ExperimentManager.run_cmvdr_inference_file`` (blind, packaged preset).
* ``CMPDR_TFREF`` (default ``/tmp/cmpdr_scratch/tfref.npz``): random Keras
  weights + outputs of the upstream CRNN / ULCNet (github.com/narrietal/cMPDR_DNN
  @753c04cbe986), dumped by a TensorFlow 2.14 run; the torch networks load
  the same weights and must reproduce the outputs.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from models.cmpdr.model import CMPDRDNN
from models.cmpdr.networks import CRNN, ULCNet
from tasks.codecs import build_codec
from training.config import instantiate_model

CONF = Path(__file__).resolve().parents[2] / "conf" / "model"
FS = 16000


def _load(name: str) -> CMPDRDNN:
    model = instantiate_model(OmegaConf.load(CONF / f"{name}.yaml"))
    assert isinstance(model, CMPDRDNN)
    return model


def _mixture(
    n: int, f0: float = 97.3, snr_db: float = -10.0, seed: int = 0, batch: int = 2
) -> torch.Tensor:
    """Speech-like harmonic target plus stationary harmonic noise, (B, n)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    rows = []
    for b in range(batch):
        fs0 = rng.uniform(120, 220)
        env = np.exp(-0.5 * ((t - t[-1] / 2) / (t[-1] / 5)) ** 2)
        s = sum(
            env * np.sin(2 * np.pi * h * fs0 * t * (1 + 0.05 * np.sin(2 * np.pi * 3 * t))) / h
            for h in range(1, 10)
        )
        v = sum(
            np.cos(2 * np.pi * p * (f0 + 7 * b) * t + rng.uniform(0, 6)) / p**0.5
            for p in range(1, 11)
        )
        v = v + 0.01 * rng.standard_normal(n)
        v *= np.sqrt(np.sum(s**2) / np.sum(v**2) / 10 ** (snr_db / 10))
        rows.append((s + v) / np.max(np.abs(s + v)) * 0.5)
    return torch.from_numpy(np.stack(rows).astype(np.float32))


@pytest.mark.parametrize("name", ["cmpdr_crn", "cmpdr_ulcnet", "cmpdr_filter"])
@pytest.mark.parametrize("n", [8000, 16000])
def test_forward_shape_and_finite(name: str, n: int) -> None:
    model = _load(name).eval()
    x = _mixture(n)[:, None]
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, n), out.shape
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", ["cmpdr_crn", "cmpdr_ulcnet"])
def test_codec_roundtrip(name: str) -> None:
    model = _load(name).eval()
    codec = build_codec(
        "speech_enhancement", n_channels=None, use_rps=False, predict_rps=False, sr=[16000, 1]
    )
    inputs = {"mixture": torch.randn(2, 16000) * 0.1}
    with torch.no_grad():
        out = codec.call_model(model, inputs)
    out = out.squeeze()
    assert out.shape == (2, 16000), out.shape
    assert torch.isfinite(out).all()


def test_param_counts() -> None:
    crn = _load("cmpdr_crn")
    n_crn = sum(p.numel() for p in crn.parameters())
    n_bn = sum(b.numel() for k, b in crn.named_buffers() if "running_" in k)
    # Released Keras code: 181,945 (count_params, BN statistics included). The
    # paper's Table I says 0.46M; the released architecture does not reach it.
    assert n_crn == 181_913
    assert n_crn + n_bn == 181_945
    n_ulc = sum(p.numel() for p in _load("cmpdr_ulcnet").parameters())
    assert n_ulc == 685_147  # Keras count_params; paper Table I: 0.68M
    assert abs(n_ulc / 1e6 - 0.68) < 0.01
    assert sum(p.numel() for p in _load("cmpdr_filter").parameters()) == 0


@pytest.mark.parametrize("name", ["cmpdr_crn", "cmpdr_ulcnet"])
def test_backward_finite(name: str) -> None:
    torch.manual_seed(0)
    model = _load(name).train()
    x = _mixture(8000)[:, None]
    target = torch.randn(2, 1, 8000) * 0.1
    loss = (model(x) - target).abs().mean()
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads if g is not None) > 0


@pytest.mark.parametrize("name", ["cmpdr_crn", "cmpdr_ulcnet"])
def test_state_dict_roundtrip(name: str) -> None:
    torch.manual_seed(1)
    a = _load(name)
    with torch.no_grad():  # non-trivial BN statistics
        for k, buf in a.named_buffers():
            if "running_var" in k:
                buf.uniform_(0.5, 1.5)
            elif "running_mean" in k:
                buf.normal_()
    a.eval()
    buf = io.BytesIO()
    torch.save(a.state_dict(), buf)
    buf.seek(0)
    torch.manual_seed(2)
    b = _load(name)
    b.load_state_dict(torch.load(buf))
    b.eval()
    x = _mixture(8000)[:, None]
    with torch.no_grad():
        assert torch.equal(a(x), b(x))


def test_frontend_batch_independent() -> None:
    fe = _load("cmpdr_filter").eval()
    x = torch.cat([_mixture(16000, seed=3, batch=1), _mixture(16000, f0=133.1, seed=4, batch=1)])
    with torch.no_grad():
        joint = fe(x[:, None])
        single = torch.cat([fe(x[i : i + 1, None]) for i in range(2)])
    torch.testing.assert_close(joint, single, rtol=0, atol=1e-6)


def test_frontend_suppresses_stationary_harmonics() -> None:
    """The filter's purpose: a stationary harmonic interferer is cancelled while
    the target passes (distortionless). Error power drops well below the noise."""
    n = 16000
    t = np.arange(n) / FS
    rng = np.random.default_rng(0)
    env = np.exp(-0.5 * ((t - 0.5) / 0.2) ** 2)
    s = sum(env * np.sin(2 * np.pi * h * 171.0 * t) / h for h in range(1, 8))
    v = sum(np.cos(2 * np.pi * p * 97.3 * t + rng.uniform(0, 6)) / p**0.5 for p in range(1, 11))
    v = v * np.sqrt(np.sum(s**2) / np.sum(v**2) * 10)  # -10 dB
    fe = _load("cmpdr_filter").eval()
    with torch.no_grad():
        y = fe(torch.from_numpy((s + v)[None, None]).float())[0, 0].double().numpy()
    gain_db = 10 * np.log10(np.sum((y - s) ** 2) / np.sum(v**2))
    assert gain_db < -6.0, gain_db


def test_filter_passthrough_without_peaks() -> None:
    """Silence: no periodogram peak, no cyclic bin -> STFT/iSTFT identity."""
    fe = _load("cmpdr_filter").eval()
    x = torch.zeros(1, 1, 8000)
    with torch.no_grad():
        y = fe(x)
    assert torch.equal(y, x)


# ------------------------------------------------------------------ slow parity
CMVDR = Path(os.environ.get("CMVDR_UPSTREAM", "/tmp/cmvdr_upstream"))
TFREF = Path(os.environ.get("CMPDR_TFREF", "/tmp/cmpdr_scratch/tfref.npz"))


@pytest.mark.slow
@pytest.mark.skipif(not (CMVDR / "cmvdr").is_dir(), reason="cmvdr upstream clone absent")
def test_frontend_matches_upstream_cmvdr() -> None:
    import warnings

    sys.path.insert(0, str(CMVDR))
    warnings.filterwarnings("ignore")
    try:
        from cmvdr.util import config
        from cmvdr.util import globs as gs

        gs.rng, _ = gs.compute_rng(seed_is_random=False, rnd_seed_=0, verbose=False)  # as the CLI
        from cmvdr import experiment_manager
    except ImportError as exc:  # missing optional upstream dependency
        pytest.skip(f"cmvdr not importable: {exc}")
    cfg = config.assign_default_values(config.load_configuration_outer("inference"))
    cfg["beamforming"]["methods"] = ["cmvdr_blind"]
    cfg["harmonics_est"]["source_signal_name"] = "noisy"
    x = _mixture(16000, snr_db=-10.0, seed=5)
    ref = np.stack(
        [
            experiment_manager.ExperimentManager.run_cmvdr_inference_file(
                np.asarray(r, dtype=np.float64), cfg
            )["cmvdr_blind"]
            for r in x.numpy()
        ]
    )
    fe = _load("cmpdr_filter").frontend
    assert fe is not None
    y = fe(x.double()).numpy()
    assert np.abs(y - ref).max() <= 1e-9 * np.abs(ref).max()


def _keras_into_torch(d: np.lib.npyio.NpzFile, crnn: CRNN, ulcnet: ULCNet) -> None:
    def w(net: str, layer: str, i: int) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(d[f"{net}/{layer}/{i}"]))

    def reorder(k: torch.Tensor, u: int) -> torch.Tensor:  # keras [z, r, h] -> torch [r, z, n]
        return torch.cat([k[..., u : 2 * u], k[..., :u], k[..., 2 * u :]], dim=-1)

    def gru(m: torch.nn.GRU, net: str, layer: str, base: int = 0, suffix: str = "") -> None:
        u = m.hidden_size
        b = w(net, layer, base + 2)
        getattr(m, f"weight_ih_l0{suffix}").copy_(reorder(w(net, layer, base), u).T)
        getattr(m, f"weight_hh_l0{suffix}").copy_(reorder(w(net, layer, base + 1), u).T)
        getattr(m, f"bias_ih_l0{suffix}").copy_(reorder(b[0], u))
        getattr(m, f"bias_hh_l0{suffix}").copy_(reorder(b[1], u))

    def conv(m: torch.nn.Conv2d, net: str, layer: str) -> None:
        m.weight.copy_(w(net, layer, 0).permute(3, 2, 0, 1))

    def dense(m: torch.nn.Linear, net: str, layer: str) -> None:
        m.weight.copy_(w(net, layer, 0).T)
        m.bias.copy_(w(net, layer, 1))

    lay = iter(str(s) for s in d["crnn_layers"])
    for c, bn in [(crnn.conv1, crnn.bn1), (crnn.conv2, crnn.bn2), (crnn.conv3, crnn.bn3)]:
        conv(c, "crnn", next(lay))
        name = next(lay)
        for i, t in enumerate([bn.weight, bn.bias, bn.running_mean, bn.running_var]):
            assert t is not None  # Keras-style BN tracks running stats
            t.copy_(w("crnn", name, i))
    gru(crnn.gru, "crnn", next(lay))
    dense(crnn.dense, "crnn", next(lay))

    lay = iter(str(s) for s in d["ulcnet_layers"])
    for s in [ulcnet.sep1, ulcnet.sep2, ulcnet.sep3, ulcnet.sep4]:
        name = next(lay)
        s.depthwise.weight.copy_(w("ulcnet", name, 0).permute(2, 3, 0, 1))
        s.pointwise.weight.copy_(w("ulcnet", name, 1).permute(3, 2, 0, 1))
    name = next(lay)
    gru(ulcnet.freq_gru, "ulcnet", name)
    gru(ulcnet.freq_gru, "ulcnet", name, 3, "_reverse")
    conv(ulcnet.pointwise, "ulcnet", next(lay))
    # model.layers lists the sub-band GRUs by graph depth (gru_1, gru_3, gru_2,
    # gru_4); their name suffix is the creation order sub1 x2, then sub2 x2.
    grus = sorted((next(lay) for _ in range(4)), key=lambda s: int(s.rsplit("_", 1)[1]))
    for g, name in zip(
        [ulcnet.sub1_gru1, ulcnet.sub1_gru2, ulcnet.sub2_gru1, ulcnet.sub2_gru2], grus
    ):
        gru(g, "ulcnet", name)
    dense(ulcnet.fc1, "ulcnet", next(lay))
    dense(ulcnet.fc2, "ulcnet", next(lay))
    for c in [ulcnet.cnn1, ulcnet.cnn2, ulcnet.mask_conv]:
        conv(c, "ulcnet", next(lay))


@pytest.mark.slow
@pytest.mark.skipif(not TFREF.is_file(), reason="Keras reference dump absent")
def test_networks_match_keras() -> None:
    d = np.load(TFREF)
    crnn = _load("cmpdr_crn").network
    ulcnet = _load("cmpdr_ulcnet").network
    assert isinstance(crnn, CRNN) and isinstance(ulcnet, ULCNet)
    crnn.eval()
    ulcnet.eval()
    with torch.no_grad():
        _keras_into_torch(d, crnn, ulcnet)
        x = torch.from_numpy(d["x"])
        y = crnn(x).numpy()
        s = ulcnet.spectrum(x).numpy()
    ref_y, ref_s = d["crnn_y"], d["ulcnet_y"]
    assert np.abs(y - ref_y).max() <= 1e-5 * np.abs(ref_y).max()
    # ULCNet compresses Re/Im of the float32 STFT with |x|^0.3, whose slope is
    # unbounded at 0: FFT rounding in near-zero components (TF vs torch) moves
    # a few compressed values by up to ~30 %. The network itself agrees to
    # ~6e-7 (magnitude-mask stage); the output differs above 1e-5 of the
    # maximum in ~0.01 % of the bins only.
    err = np.abs(s - ref_s) / np.abs(ref_s).max()
    assert np.mean(err > 1e-5) < 1e-3
    assert err.max() < 1e-3
