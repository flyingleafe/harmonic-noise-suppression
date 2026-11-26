"""MP-SENet metric discriminator (CMGAN-style, PESQ surrogate) — official release port.

Upstream: https://github.com/yxlu-0102/MP-SENet ``models/discriminator.py`` and
``utils.py`` (``LearnableSigmoid1d``), commit
``89932cfe90d1dacb8e170e4a331d762462c21792``, MIT licence, Copyright (c) 2023
Yexin Lu. State-dict keys are identical to upstream's ``MetricDiscriminator``
(``nn.utils.spectral_norm`` hooks: ``weight_orig`` / ``weight_u`` /
``weight_v``), 181,650 parameters.

Input: reference and test compressed magnitudes ``(B, F, T)`` (upstream feeds
the clean and the re-analysed enhanced magnitude, both ``|STFT| ** 0.3``).
Output: ``(B, 1)`` in ``[0, 1]``, the predicted PESQ normalised as
``(PESQ - 1) / 3.5``. Like upstream, training mode applies dropout 0.3 and one
spectral-norm power iteration per call.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["LearnableSigmoid1d", "MetricDiscriminator", "build_metric_discriminator"]


class LearnableSigmoid1d(nn.Module):
    """``beta * sigmoid(slope * x)`` with a learnable ``(in_features,)`` slope."""

    def __init__(self, in_features: int, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.beta * torch.sigmoid(self.slope * x)


class MetricDiscriminator(nn.Module):
    def __init__(self, dim: int = 16, in_channel: int = 2) -> None:
        super().__init__()
        sn = nn.utils.spectral_norm
        self.layers = nn.Sequential(
            sn(nn.Conv2d(in_channel, dim, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim),
            sn(nn.Conv2d(dim, dim * 2, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 2, affine=True),
            nn.PReLU(dim * 2),
            sn(nn.Conv2d(dim * 2, dim * 4, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 4, affine=True),
            nn.PReLU(dim * 4),
            sn(nn.Conv2d(dim * 4, dim * 8, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 8, affine=True),
            nn.PReLU(dim * 8),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            sn(nn.Linear(dim * 8, dim * 4)),
            nn.Dropout(0.3),
            nn.PReLU(dim * 4),
            sn(nn.Linear(dim * 4, 1)),
            LearnableSigmoid1d(1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Reference ``x`` and test ``y`` magnitudes ``(B, F, T)`` -> ``(B, 1)``."""
        return self.layers(torch.stack((x, y), dim=1))


def build_metric_discriminator(dim: int = 16, in_channel: int = 2) -> MetricDiscriminator:
    return MetricDiscriminator(dim=dim, in_channel=in_channel)
