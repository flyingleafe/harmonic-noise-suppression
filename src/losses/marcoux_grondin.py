"""BPF MSE plus activity BCE (Marcoux–Grondin §II-A, Eq. 2)."""

import tdseries as td
import torch
import torch.nn.functional as F

from framespec import FrameSpec, SeriesSpec
from losses._common import get_tensor, rps_series_spec
from losses.pit import pit_mse_loss


class MarcouxGrondinLoss:
    """Equal-weight BPF MSE and activity BCE against rotor telemetry.

    ``matching="rank"`` (paper): the targets are the per-frame fastest
    telemetry speeds, ascending, one per output row. ``matching="pit"``
    (adaptation): one rotor assignment per clip over unsorted tracks.

    The activity target is 1 where any rotor turns (speed > 0): frames with
    some rotors stopped stay active, so an activity-gated model can still
    output the turning ones. Errors are in BPF Hz, keeping the paper's
    balance between the two terms.
    """

    def __init__(self, rate: tuple[int, int], num_blades: int, matching: str = "rank") -> None:
        if matching not in ("rank", "pit"):
            raise ValueError(f"matching must be 'rank' or 'pit', got {matching!r}")
        self.num_blades = num_blades
        self.matching = matching
        rps = rps_series_spec(rate)
        self.requires_pred = FrameSpec(
            {
                "rps_pred": rps,
                "activity_logits": SeriesSpec(dims=("batch", "time"), time="grid", rate=rate),
            }
        )
        self.requires_target = FrameSpec({"rps": rps})

    def __call__(self, pred: td.Frame, target: td.Frame) -> torch.Tensor:
        rates = get_tensor(pred, "rps_pred")
        logits = get_tensor(pred, "activity_logits")
        telemetry = get_tensor(target, "rps")
        activity = (telemetry.amax(dim=1) > 0).to(logits.dtype)
        if self.matching == "rank":
            telemetry = telemetry.sort(dim=1).values[:, -rates.shape[1] :]
        if rates.shape != telemetry.shape:
            raise ValueError(f"RPS shape mismatch: {rates.shape} vs {telemetry.shape}")
        est, ref = rates * self.num_blades, telemetry * self.num_blades
        if self.matching == "pit":
            bpf_mse = pit_mse_loss(est, ref)
            assert isinstance(bpf_mse, torch.Tensor)
        else:
            bpf_mse = (est - ref).square().mean()
        return bpf_mse + F.binary_cross_entropy_with_logits(logits, activity)
