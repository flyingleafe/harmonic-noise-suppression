import math

import pytest
import tdseries as td
import torch
from torch import nn

from losses.marcoux_grondin import MarcouxGrondinLoss
from models.frontends import build_frontend
from models.marcoux_grondin import MarcouxGrondinRPS4

RATE = (16000, 512)
ROTOR_DIMS = ("batch", "rotor", "time")


def _frames(rates, logits, telemetry):
    grid = {"time": td.GridIndex.create(RATE, rates.shape[-1], t_start=0)}
    pred = td.Frame(
        {
            "rps_pred": td.Series(rates, ROTOR_DIMS, grid),
            "activity_logits": td.Series(logits, ("batch", "time"), grid),
        }
    )
    return pred, td.Frame({"rps": td.Series(telemetry, ROTOR_DIMS, grid)})


def test_cepstrum_places_periodic_energy_at_the_period():
    frontend = build_frontend("mel_cepstrum")
    # Band-rich pulse train: 250 Hz gives a 64-sample period at 16 kHz.
    audio = torch.zeros(1, 4096)
    audio[:, ::64] = 1.0
    features = frontend(audio)
    center = features[0, :, :, 4]
    assert center[0].min() >= 0
    assert center[1, 1:].argmax().item() + 1 == 64
    assert torch.isfinite(frontend(torch.zeros_like(audio))).all()


def test_loss_uses_per_frame_fastest_two_and_bpf_units():
    telemetry = torch.tensor([[[10.0, 70, 0], [30, 10, 0], [60, 30, 0], [70, 60, 0]]])
    rates = torch.tensor([[[60.0, 59, 0], [70, 70, 0]]])
    logits = torch.tensor([[1.0, -3, -2]])
    pred, target = _frames(rates, logits, telemetry)

    loss = MarcouxGrondinLoss(RATE, num_blades=2)(pred, target)
    bce = sum(math.log1p(math.exp(x)) for x in (-1, 3, -2)) / 3
    assert loss.item() == pytest.approx(4 / 6 + bce)


def test_pit_loss_matches_tracks_and_keeps_partly_stopped_frames_active():
    # Frame 1 has one rotor stopped: still active, or a gated model could not
    # output the three turning rotors.
    telemetry = torch.tensor([[[60.0, 62, 0], [70, 0, 0], [50, 55, 0], [80, 81, 0]]])
    rates = telemetry[:, [2, 0, 3, 1]]
    logits = torch.tensor([[2.0, 2, -2]])
    pred, target = _frames(rates, logits, telemetry)

    loss = MarcouxGrondinLoss(RATE, num_blades=2, matching="pit")(pred, target)
    assert loss.item() == pytest.approx(math.log1p(math.exp(-2)))


class _FixedLogits(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.logits[..., None]


def test_gated_variant_zeroes_speeds_exactly_where_activity_is_off(monkeypatch):
    torch.manual_seed(0)
    model = MarcouxGrondinRPS4().eval()
    logits = torch.tensor([[1.0, -1, 2, 0, -3, 4, -5, 6, -0.5]])
    monkeypatch.setattr(model, "activity", _FixedLogits(logits))
    audio = torch.randn(1, 8 * 512)
    with torch.no_grad():
        gated, _ = model(audio, return_activity=True)
        model.gate_by_activity = False
        free = model(audio)

    on = (logits > 0)[:, None].expand_as(gated)
    assert gated.shape == (1, 4, 9)
    assert torch.equal(gated[~on], torch.zeros_like(gated[~on]))
    assert torch.equal(gated[on], free[on])
    assert free[~on].abs().sum() > 0
