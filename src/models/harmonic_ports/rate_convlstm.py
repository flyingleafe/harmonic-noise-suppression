"""ConvLSTM over time whose gates are 1-D convolutions along the salience axis.

The replacement for HPPNet's `FreqGroupLSTM` (variant A of
``docs/pyramid-harmonic-frontend-design.md`` § 7). `FreqGroupLSTM` runs one
recurrence per frequency bin and cannot move state across bins; here the
hidden-to-gate map is a ``k``-tap convolution along the axis, so the state at
bin ``g`` is written from bins ``g ± k // 2`` of the previous frame and can
follow a rotor through a ramp. The half-width is sized from the raw telemetry's
largest sustained per-frame speed change on the model's own axis (rate bins
for `HPPNetPyramid`, CQT log bins for `HPPNetOrig`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn

__all__ = ["RateConvLSTM"]


def _cell_eager(
    gx: torch.Tensor, gh: torch.Tensor, c: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One LSTM cell update: ``gx + gh`` are the four stacked gate pre-activations."""
    gi, gf, go, gu = (gx + gh).chunk(4, dim=1)
    c = torch.sigmoid(gf) * c + torch.sigmoid(gi) * torch.tanh(gu)
    h = torch.sigmoid(go) * torch.tanh(c)
    return h, c


# The cell is ~10 pointwise kernels per step and the recurrence runs it 2T
# times with nothing to overlap, so launch latency, not work, sets its cost;
# fusing them is the whole optimisation. ``script``: TorchScript's pointwise
# fuser; ``compile``: Inductor (Triton), compiled once per shape.
_CELLS: dict[str, object] = {
    "eager": _cell_eager,
    "script": torch.jit.script(_cell_eager),
}


def _cell_compiled(
    gx: torch.Tensor, gh: torch.Tensor, c: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    fn = _CELLS.get("_compiled")
    if fn is None:
        fn = torch.compile(_cell_eager, dynamic=False)
        _CELLS["_compiled"] = fn
    return cast(Callable[..., tuple[torch.Tensor, torch.Tensor]], fn)(gx, gh, c)


_CELLS["compile"] = _cell_compiled


class RateConvLSTM(nn.Module):
    """ConvLSTM over time with gate convolutions along the rate axis.

    ``(B, C_in, T, G) -> (B, D * hidden, T, G)``, ``D = 2`` if bidirectional.
    The input contribution to the gates is a 1x1 convolution (the trunk has
    already supplied the rate context); the recurrent contribution is a
    ``k``-tap 1-D convolution of the previous hidden map, so state at rate bin
    ``g`` is updated from bins ``g ± k // 2`` of the previous frame and that is
    the state's whole reach per frame. The forget-gate bias starts at 1.
    ``cell`` selects the pointwise-cell implementation: ``"script"`` (fused,
    default), ``"compile"`` (Inductor) or ``"eager"``.
    """

    def __init__(
        self,
        c_in: int,
        hidden: int,
        kernel_size: int,
        bidirectional: bool = True,
        separable: bool = False,
        cell: str = "script",
    ) -> None:
        super().__init__()
        if cell not in ("eager", "script", "compile"):
            raise ValueError(f"cell must be 'eager', 'script' or 'compile', got {cell!r}")
        self.cell = cell

        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.hidden, self.kernel_size = int(hidden), int(kernel_size)
        self.n_dir = 2 if bidirectional else 1
        pad = self.kernel_size // 2
        self.x_conv = nn.Conv2d(int(c_in), self.n_dir * 4 * self.hidden, kernel_size=1)

        def recurrent() -> nn.Module:
            if not separable:
                return nn.Conv1d(
                    self.hidden, 4 * self.hidden, self.kernel_size, padding=pad, bias=False
                )
            # Rank-factorised: a per-channel k-tap displacement kernel (where
            # the state moves along the axis), then a 1x1 channel mix.
            return nn.Sequential(
                nn.Conv1d(
                    self.hidden,
                    self.hidden,
                    self.kernel_size,
                    padding=pad,
                    groups=self.hidden,
                    bias=False,
                ),
                nn.Conv1d(self.hidden, 4 * self.hidden, 1, bias=False),
            )

        self.h_conv = nn.ModuleList([recurrent() for _ in range(self.n_dir)])
        with torch.no_grad():
            bias = cast(torch.Tensor, self.x_conv.bias).view(self.n_dir, 4, self.hidden)
            bias[:, 1].fill_(1.0)

    def _run(self, gx: torch.Tensor, h_conv: nn.Module, reverse: bool) -> torch.Tensor:
        b, _, t, g = gx.shape
        h = gx.new_zeros(b, self.hidden, g)
        c = gx.new_zeros(b, self.hidden, g)
        outs: list[torch.Tensor] = [h] * t
        steps = range(t - 1, -1, -1) if reverse else range(t)
        cell = cast(Callable[..., tuple[torch.Tensor, torch.Tensor]], _CELLS[self.cell])
        # One unbind, not ``gx[:, :, s]`` per step: each select/slice backward
        # materialises a full-size zero copy of ``gx`` (85 % of the step on a T4).
        frames = gx.unbind(2)
        for s in steps:
            h, c = cell(frames[s], h_conv(h), c)
            outs[s] = h
        return torch.stack(outs, dim=2)  # (B, hidden, T, G)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = self.x_conv(x).chunk(self.n_dir, dim=1)
        outs = [
            self._run(gx, conv, reverse=bool(d))
            for d, (gx, conv) in enumerate(zip(gates, self.h_conv, strict=True))
        ]
        return torch.cat(outs, dim=1)
