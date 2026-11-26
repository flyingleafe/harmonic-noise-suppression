"""Tests for the DPT-FSNet SE baseline (``models.dpt_fsnet``).

Built from ``conf/model/dpt_fsnet.yaml`` through ``training.config.instantiate_model``
(the dispatch a real Hydra run uses). CPU-fast: signals of at most 1 s.

The ``slow`` parity test compares against the upstream code
(github.com/dangf15/DPT-FSNet @ 10cbda3, no licence, so it is loaded at run
time from a scratch clone at ``/tmp/dpt_fsnet_upstream`` and never copied into
the repository); it skips when the clone is absent.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest
import tdseries as td
import torch
from omegaconf import OmegaConf

from models.dpt_fsnet import DPTFSNet
from tasks.codecs import build_codec
from training.config import instantiate_model

CONF = Path(__file__).resolve().parents[2] / "conf" / "model" / "dpt_fsnet.yaml"
UPSTREAM = Path("/tmp/dpt_fsnet_upstream")
PAPER_PARAMS = 879_460  # paper Table 1: 0.88 M


def _load_model() -> DPTFSNet:
    model = instantiate_model(OmegaConf.load(CONF))
    assert isinstance(model, DPTFSNet)
    return model


@pytest.mark.parametrize("n_samples", [4000, 16037])
def test_forward_shape_and_finite(n_samples: int) -> None:
    torch.manual_seed(0)
    model = _load_model().eval()
    x = torch.randn(2, 1, n_samples)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, n_samples), out.shape
    assert torch.isfinite(out).all()


def test_codec_roundtrip() -> None:
    model = _load_model().eval()
    codec = build_codec(
        "speech_enhancement", n_channels=None, use_rps=False, predict_rps=False, sr=(16000, 1)
    )
    batch = td.Frame(
        {"mixture": td.uniform(torch.randn(2, 8000), (16000, 1), dims=("batch", "time"))}
    )
    with torch.no_grad():
        outputs = codec.call_model(model, codec.to_inputs(batch))
    frame = codec.to_frame(outputs, batch)
    enhanced = frame["enhanced"]
    assert enhanced.dims == ("batch", "time")
    assert tuple(enhanced.data.shape) == (2, 8000)
    assert torch.isfinite(enhanced.data).all()


def test_param_count_matches_paper() -> None:
    model = _load_model()
    assert sum(p.numel() for p in model.parameters()) == PAPER_PARAMS


def test_stft_perfect_reconstruction() -> None:
    model = _load_model()
    stft = model.stft
    x = torch.randn(2, 8000, dtype=torch.float64)
    stft = stft.double()
    y = stft.inverse(stft(x))
    torch.testing.assert_close(y, x, rtol=0, atol=1e-6)


def test_backward_finite_grads() -> None:
    torch.manual_seed(0)
    model = _load_model().train()
    x = torch.randn(2, 1, 4000)
    target = torch.randn(2, 1, 4000)
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)
    assert any(g.abs().sum() > 0 for g in grads if g is not None)


def test_state_dict_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _load_model().eval()
    path = tmp_path / "dpt_fsnet.pt"
    torch.save(model.state_dict(), path)
    torch.manual_seed(1)
    fresh = _load_model().eval()
    fresh.load_state_dict(torch.load(path, weights_only=True))
    x = torch.randn(1, 1, 6000)
    with torch.no_grad():
        torch.testing.assert_close(fresh(x), model(x), rtol=0, atol=0)


# --- parity against the upstream code --------------------------------------------

_DENSE_SUB = {"conv": "conv", "norm": "norm", "prelu": "act"}
_TRANS_SUB = {
    "self_attn": "attn",
    "gru": "gru",
    "linear2": "ffn_out",
    "norm1": "attn_norm",
    "norm2": "ffn_norm",
}
_PATH_SUB = {"row": "intra", "col": "inter"}
_KEY_RULES: list[tuple[str, object]] = [
    (r"^inp_conv\.", "encoder.in_conv."),
    (r"^inp_norm\.", "encoder.in_norm."),
    (r"^inp_prelu\.", "encoder.in_act."),
    (
        r"^(enc|dec)_dense1\.(conv|norm|prelu)(\d+)\.",
        lambda m: (
            f"{'encoder' if m[1] == 'enc' else 'decoder'}.dense.layers."
            f"{int(m[3]) - 1}.{_DENSE_SUB[m[2]]}."
        ),
    ),
    (r"^dual_transformer\.input\.0\.", "dptpm.in_conv."),
    (r"^dual_transformer\.input\.1\.", "dptpm.in_act."),
    (
        r"^dual_transformer\.(row|col)_trans\.(\d+)\.(self_attn|gru|linear2|norm1|norm2)\.",
        lambda m: f"dptpm.blocks.{m[2]}.{_PATH_SUB[m[1]]}.{_TRANS_SUB[m[3]]}.",
    ),
    (
        r"^dual_transformer\.(row|col)_norm\.(\d+)\.",
        lambda m: f"dptpm.blocks.{m[2]}.{_PATH_SUB[m[1]]}_norm.",
    ),
    (r"^dual_transformer\.output\.0\.", "dptpm.out_act."),
    (r"^dual_transformer\.output\.1\.", "dptpm.out_conv."),
    (r"^output1\.0\.", "dptpm.gate_tanh."),
    (r"^output2\.0\.", "dptpm.gate_sigmoid."),
    (r"^out_conv\.", "decoder.mask_conv."),
]


def upstream_to_port_key(key: str) -> str | None:
    """Map an upstream ``Net`` state_dict key to this port; ``None`` for the
    fixed STFT buffers."""
    if key.startswith(("stft.", "istft.")):
        return None
    for pattern, repl in _KEY_RULES:
        new, n = re.subn(pattern, repl, key)  # type: ignore[arg-type]
        if n:
            return new
    raise KeyError(f"unmapped upstream key {key!r}")


def _load_upstream(n_fft: int, hop: int, win: int) -> torch.nn.Module:
    """Build upstream ``Net`` on CPU with the complex STFT the paper specifies.

    Run-time patches (none alter the network): a stub for the broken
    ``utils`` module (show_params/show_model), ``'hanning'`` → ``'hann'`` for
    current scipy, the hard-coded ``.to(device='cuda')`` → the input's device,
    and the STFT front-end returning ``(real, imag)`` instead of upstream's
    ``(magnitude, phase)`` (see the port's docstring, "Deviations").
    """
    import scipy.signal

    sys.path.insert(0, str(UPSTREAM))
    try:
        stub = types.ModuleType("utils")
        stub.show_params = stub.show_model = lambda *_a, **_k: None  # type: ignore[attr-defined]
        sys.modules["utils"] = stub
        import conv_stft

        conv_stft.get_window = lambda name, n, fftbins=True: scipy.signal.get_window(
            "hann" if name == "hanning" else name, n, fftbins=fftbins
        )
        src = (UPSTREAM / "model.py").read_text().replace(".to(device='cuda')", ".to(real.device)")
        mod = types.ModuleType("dpt_fsnet_upstream_model")
        exec(compile(src, str(UPSTREAM / "model.py"), "exec"), mod.__dict__)
    finally:
        sys.path.remove(str(UPSTREAM))
        sys.modules.pop("utils", None)

    class ComplexSTFT(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = conv_stft.ConvSTFT(win, hop, n_fft, "hanning", "complex")

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            spec = self.inner(x)
            return spec[:, : n_fft // 2 + 1], spec[:, n_fft // 2 + 1 :]

    net = mod.Net(L=n_fft)
    net.stft = ComplexSTFT()
    net.istft = conv_stft.ConviSTFT(win, hop, n_fft, "hanning", "complex")
    return net.eval()


@pytest.mark.slow
@pytest.mark.skipif(not UPSTREAM.is_dir(), reason="upstream clone absent")
@pytest.mark.parametrize(("n_fft", "hop", "win"), [(512, 100, 400), (512, 256, 512)])
def test_parity_with_upstream(n_fft: int, hop: int, win: int) -> None:
    torch.manual_seed(0)
    port = DPTFSNet(n_fft=n_fft, hop_length=hop, win_length=win).eval()
    upstream = _load_upstream(n_fft, hop, win)
    port_sd = port.state_dict()
    mapped = {}
    for key, value in upstream.state_dict().items():
        new = upstream_to_port_key(key)
        mapped[key] = value if new is None else port_sd.pop(new)
    assert not port_sd, f"port keys with no upstream source: {sorted(port_sd)}"
    upstream.load_state_dict(mapped, strict=True)

    n_samples = 16000 - 16000 % hop
    x = 0.1 * torch.randn(2, 1, n_samples)
    with torch.no_grad():
        ref = upstream(x)
        out = port(x)
    assert out.shape == ref.shape == (2, 1, n_samples)
    diff = (out - ref).abs().max().item()
    scale = ref.abs().max().item()
    print(
        f"\nDPT-FSNet parity n_fft={n_fft} hop={hop} win={win}: max|diff|={diff:.3e} max|ref|={scale:.3e}"
    )
    assert diff <= 1e-4 * scale
