"""Validation metrics for the per-rotor salience layers.

Two numbers, and the split between them is deliberate.

``bce`` mirrors :class:`losses.LayerPITSalienceBCELoss` exactly (same grid
spec, ``sigma_rev_s`` and ``focus``), so the training objective can be read on
the validation set.

``rps_mae`` is the same object in rev/s: each layer's peak, refined by the
three-point log-parabolic fit in rev/s coordinates, PIT-aligned to the target.
It reads the layers the way `models.salience_crf` proves is exact — the log of a
Gaussian is a parabola, so three bins locate the vertex with no residue on any
grid; on a log grid an argmax at bin 0 (the off state) reads 0.0, on the linear
grid bin 0 is the 0 rev/s rate bin — and it is NOT the deployed
decoder, which is a CRF best path per layer
(`models.harmonic_ports.layer_readout`). The difference is temporal robustness
only: on a correct layer the two agree, and the CRF additionally survives a
frame whose own peak is wrong. The CRF is not used here because metrics run per
sample on CPU.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import permutations
from typing import Any

import numpy as np
import tdseries as td
import torch
import torch.nn.functional as F

from framespec import FrameSpec, SeriesSpec
from losses.salience_layers import layer_pit_bce, resolve_focus
from metrics._common import get_array
from models.salience_crf import gaussian_layer_target, grid_from_spec, refine_peak

__all__ = ["LayerPITSalienceBCEMetric", "LayerPeakRPSMetric", "peak_readout"]


def peak_readout(scores: torch.Tensor, grid: np.ndarray, *, has_off: bool) -> torch.Tensor:
    """``(B, R, G, T)`` LOG-domain layers -> ``(B, R, T)`` rev/s.

    Argmax along the rate axis, then `models.salience_crf.refine_peak` (the
    three-point parabola in rev/s; with an off state bin 0 reads 0.0). On a
    Gaussian layer this is exact; it is the readout without the CRF's temporal
    model.
    """
    s = scores.double()
    b, r, n_g, n_t = s.shape
    flat = s.reshape(b * r, n_g, n_t)
    g = torch.as_tensor(np.asarray(grid), dtype=torch.float64, device=scores.device)
    return refine_peak(flat, flat.argmax(dim=1), g, has_off=has_off).reshape(b, r, n_t)


def _pit_mae(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    """Best-assignment mean absolute error over ``(R, T)`` pairs, rev/s."""
    r = pred.shape[0]
    cost = (pred[:, None] - tgt[None, :]).abs().mean(dim=-1)  # (R_pred, R_tgt)
    best = min(float(sum(cost[i, p[i]] for i in range(r))) for p in permutations(range(r)))
    return best / r


class _LayerGridMetric:
    """Shared grid/spec plumbing for the two layer metrics."""

    name = "bce"

    def __init__(
        self,
        *,
        grid: Mapping[str, Any],
        n_layers: int = 4,
        sigma_rev_s: float = 1.0,
        focus: float | str = 0.0,
        rate: tuple[int, int] | None = None,
        pred_key: str = "salience",
        target_key: str = "rps",
    ) -> None:
        self._freqs, self.has_off = grid_from_spec(grid)
        self.n_layers = int(n_layers)
        self.sigma_rev_s = float(sigma_rev_s)
        self.focus = resolve_focus(focus, self._freqs, self.sigma_rev_s, has_off=self.has_off)
        self.pred_key = pred_key
        self.target_key = target_key
        self.requires_pred = FrameSpec(
            {pred_key: SeriesSpec(dims=("batch", "freq", "time"), time="grid", rate=rate)}
        )
        self.requires_target = FrameSpec(
            {target_key: SeriesSpec(dims=("batch", "rotor", "time"), time="grid", rate=None)}
        )

    def _unpack(self, pred: td.Frame, target: td.Frame) -> tuple[torch.Tensor, torch.Tensor]:
        """One sample -> ``(1, R, G, T)`` layers and ``(1, R, T)`` rev/s on that grid."""
        logits = torch.as_tensor(get_array(pred, self.pred_key)).unsqueeze(0)  # (1, R*G, T)
        rps = torch.as_tensor(get_array(target, self.target_key)).float().unsqueeze(0)
        b, fg, n_t = logits.shape
        r, n_g = self.n_layers, len(self._freqs)
        if fg != r * n_g:
            raise ValueError(f"model emits {fg} bins; {r} layers x {n_g} rate bins configured")
        layers = logits.reshape(b, r, n_g, n_t)
        rps_grid = F.interpolate(rps, size=n_t, mode="linear", align_corners=False)
        return layers, rps_grid


class LayerPITSalienceBCEMetric(_LayerGridMetric):
    """Per-sample twin of :class:`losses.LayerPITSalienceBCELoss`."""

    name = "bce"

    def __call__(self, pred: td.Frame, target: td.Frame) -> float:
        layers, rps_grid = self._unpack(pred, target)
        tgt = gaussian_layer_target(
            rps_grid, self._freqs, sigma_rev_s=self.sigma_rev_s, has_off=self.has_off
        )
        return float(layer_pit_bce(layers, tgt, focus=self.focus, off_bin=self.has_off).item())


class LayerPeakRPSMetric(_LayerGridMetric):
    """PIT mean absolute error in rev/s, read off the layers by peak + parabola."""

    name = "rps_mae"

    def __call__(self, pred: td.Frame, target: td.Frame) -> float:
        layers, rps_grid = self._unpack(pred, target)
        speeds = peak_readout(F.logsigmoid(layers), self._freqs, has_off=self.has_off)
        return _pit_mae(speeds[0], rps_grid[0].double())
