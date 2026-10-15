"""Permutation-invariant BCE over PER-ROTOR salience layers.

WHY A SECOND SALIENCE LOSS. :class:`losses.SalienceRPSBCELoss` trains one shared
map with a triangular kernel, and `models.salience_crf` measured what that pair
costs: encoding real training telemetry and decoding it again returns the
trajectory 8.24 rev/s away on average, with 39-45% of frames more than half a bin
off. The cause is representational, not statistical — DREGON's rotor pairs sit
0.13-0.86 rev/s apart, INSIDE one 0.5 rev/s bin, so a shared map has fewer peaks
than it has rotors. One layer per rotor removes the merge entirely, and a
Gaussian kernel in rev/s makes the sub-bin readout exact on any grid, because
the log of a Gaussian is a parabola globally and three bins locate its vertex.

Per-rotor layers cost one thing: the layers have no canonical order, so the loss
must be permutation invariant. This is ordinary PIT — all ``R!`` assignments per
CLIP (24 at four rotors), the best one kept — the same object
:mod:`losses.pit` applies to direct RPS regression, moved onto the map.

``pos_weight`` IS 1 AND MUST STAY 1. With a soft target ``p``, weighted BCE is
minimized at ``sigmoid(z) = w p / (w p + 1 - p)``, not at ``p``. At the
``auto`` weight this grid would otherwise take (about 119 for a single Gaussian
layer) that saturates every bin within 2.2 sigma of the peak to ``sigmoid(z) ~
1``, which FLATTENS the parabola the decoder fits — the imbalance correction and
the exact readout are directly opposed. Unweighted BCE against a soft target is
minimized at ``sigmoid(z) = p`` exactly, so ``log sigmoid(z) = -d^2 / 2 sigma^2``
and the readout stays exact.

THE IMBALANCE IS HANDLED BY ``focus`` INSTEAD, and the difference is where the
weight is applied. ``pos_weight`` reweights ONE SIDE of a bin's cross-entropy
against the other, which moves that bin's minimum. ``focus`` multiplies a bin's
WHOLE cross-entropy by ``1 + focus * target``, and a positive constant does not
move a minimum: every bin still bottoms out at ``sigmoid(z) = target``, so the
readout stays exact while the peak gets the emphasis. ``focus="auto"`` takes
``G / m`` with ``m`` the expected Gaussian mass of one layer on its grid
(`models.salience_crf.layer_focus`: ``sqrt(2 pi) sigma / step`` bins on a linear
grid, averaged over the represented rotor rates on a log grid, whose bins widen
with the rate), which splits the total weight about evenly between the peak and
the rest.

THE GRID IS A SPEC, NOT THREE NUMBERS. ``grid: {kind: linear, fmin, fmax,
bins}`` or ``{kind: log_off, fmin, bins_per_octave, n_bins}``
(`models.salience_crf.grid_from_spec`); the kind decides whether bin 0 is an
OFF state (``log_off``) or the 0 rev/s rate bin (``linear``), and that flag
(``has_off``) drives the target, the focus and the readout. The model must
emit ``n_layers * len(grid)`` bins, checked on every call.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import permutations
from typing import Any

import tdseries as td
import torch
import torch.nn.functional as F

from framespec import FrameSpec, SeriesSpec
from losses._common import get_tensor
from models.salience_crf import gaussian_layer_target, grid_from_spec, layer_focus

__all__ = ["layer_pit_bce", "LayerPITSalienceBCELoss", "resolve_focus"]


def layer_pit_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float | None = None,
    focus: float = 0.0,
) -> torch.Tensor:
    """Permutation-invariant BCE between two stacks of layers.

    Args:
        logits: ``(B, R, G, T)`` raw per-rotor salience logits.
        target: ``(B, R, G, T)`` soft per-rotor target in ``[0, 1]``.
        pos_weight: positive-class weight; ``None`` (the default) means 1, and
            see the module docstring for why anything else breaks the readout.
        focus: PER-BIN emphasis, weight ``1 + focus * target`` on bins
            ``1..``. This is the class-imbalance handle that ``pos_weight``
            cannot be here. It scales each bin's WHOLE cross-entropy by a
            positive constant, and a positive constant does not move a
            minimum — every bin's optimum stays ``sigmoid(z) = target``, so
            the readout stays exact. ``focus = G / m``, ``m`` the layer's
            expected Gaussian mass in bins (`models.salience_crf.layer_focus`),
            puts about equal total weight on the peak and on everything else.

            BIN 0 IS NEVER FOCUS-WEIGHTED, on any grid. Its positive is a
            CLASS present on every stopped frame (silence arm + ground spans,
            ~20 % of training), not a sparse peak among ``G`` competitors: on
            a log grid it is the off state; on the linear grid it is the
            0 rev/s bin, which a log trunk reaches only through its pooled
            off logit (bins 1-27 are unreachable) and which on a 0-floored
            grid is the edge the stopped class sits on. Weighting that class
            ~65:1 made the cheapest head say "0" everywhere above a correct
            map — the HPPNet-L2 collapses of 2026-10-15 (off-state pilots 1-2:
            off p 0.95 on cruise frames; linear-convention pilot 4: the same
            drift through bin 0, `real_cruise` 1.7 → 22 by step 4.5k). The
            one run that held was the one with bin 0 at weight 1. Bin 0 gets
            plain BCE; the rate bins keep the emphasis.

    Returns:
        Scalar: the mean over the batch of the best assignment's mean BCE.
    """
    b, r, _g, _t = logits.shape
    if target.shape != logits.shape:
        raise ValueError(
            f"shape mismatch: logits {tuple(logits.shape)} target {tuple(target.shape)}"
        )
    pw = None
    if pos_weight is not None:
        pw = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)

    # cost[b, i, j] = mean BCE of predicted layer i against target layer j.
    # Built one TARGET layer at a time: the full outer product would hold
    # B*R*R*G*T floats at once, and validation clips are eight times longer
    # than training clips.
    cols = []
    for j in range(r):
        tgt = target[:, j : j + 1].expand(-1, r, -1, -1).to(logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, tgt, pos_weight=pw, reduction="none")
        if focus:
            weight = 1.0 + float(focus) * tgt
            weight[:, :, :1] = 1.0
            bce = bce * weight
        cols.append(bce.mean(dim=(2, 3)))  # (B, R)
    cost = torch.stack(cols, dim=2)  # (B, R, R)

    idx = torch.arange(r, device=logits.device)
    scores = torch.stack(
        [
            cost[:, idx, torch.as_tensor(p, device=logits.device)].sum(dim=1)
            for p in permutations(range(r))
        ],
        dim=1,
    )  # (B, R!)
    return (scores.min(dim=1).values / r).mean()


class LayerPITSalienceBCELoss:
    """Frame adapter: derives the per-rotor Gaussian target from ``target["rps"]``.

    Like :class:`losses.SalienceRPSBCELoss` it owns the grid, because no dataset
    emits a ``salience`` entry. It differs in three ways, all of them the reason
    it exists:

    1. The target is :func:`models.salience_crf.gaussian_layer_target` — one
       GAUSSIAN layer per rotor, ``sigma_rev_s`` wide in rev/s on any grid —
       not one triangular-kernel map shared by every rotor.
    2. A stopped rotor is bin 0 (the off state) alone, not a dark column. The
       old target drops any rotor under 0.1 rev/s, which encodes "stopped" as
       absence of evidence and is what forces the decoder to carry a threshold.
    3. The loss is permutation invariant over the layers.

    The prediction arrives in the codec's ``(batch, freq, time)`` wire format
    with the rotor layers stacked along the frequency axis, so ``n_layers *
    len(grid)`` must equal the model's output width.

    Args:
        grid: the output grid spec (module docstring) — the model's own.
        n_layers: per-rotor layers stacked in the prediction.
        sigma_rev_s: Gaussian width in rev/s, shared by every grid.
        pos_weight: must stay ``None`` (module docstring).
        focus: per-bin emphasis ``1 + focus * target``; ``"auto"`` is
            `models.salience_crf.layer_focus` on ``grid``.
    """

    def __init__(
        self,
        *,
        grid: Mapping[str, Any],
        n_layers: int = 4,
        sigma_rev_s: float = 1.0,
        pos_weight: float | None = None,
        focus: float | str = 0.0,
        rate: tuple[int, int] | None = None,
        pred_key: str = "salience",
        target_key: str = "rps",
    ) -> None:
        self._freqs, self.has_off = grid_from_spec(grid)
        self.n_layers = int(n_layers)
        self.sigma_rev_s = float(sigma_rev_s)
        self.pos_weight = pos_weight
        self.focus = resolve_focus(focus, self._freqs, self.sigma_rev_s, has_off=self.has_off)
        self.pred_key = pred_key
        self.target_key = target_key
        self.requires_pred = FrameSpec(
            {pred_key: SeriesSpec(dims=("batch", "freq", "time"), time="grid", rate=rate)}
        )
        self.requires_target = FrameSpec(
            {target_key: SeriesSpec(dims=("batch", "rotor", "time"), time="grid", rate=None)}
        )

    def layers_and_target(
        self, logits: torch.Tensor, rps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, R*G, T)``, ``(B, R, T_any)`` -> two ``(B, R, G, T)`` stacks."""
        b, fg, n_grid_t = logits.shape
        r = self.n_layers
        if fg != r * len(self._freqs):
            raise ValueError(
                f"model emits {fg} output bins; {r} layers x {len(self._freqs)} "
                "rate bins were configured"
            )
        layers = logits.reshape(b, r, len(self._freqs), n_grid_t)
        rps_grid = F.interpolate(rps.float(), size=n_grid_t, mode="linear", align_corners=False)
        tgt = gaussian_layer_target(
            rps_grid, self._freqs, sigma_rev_s=self.sigma_rev_s, has_off=self.has_off
        )
        return layers, tgt

    def __call__(self, pred: td.Frame, target: td.Frame) -> torch.Tensor:
        logits = get_tensor(pred, self.pred_key)
        rps = get_tensor(target, self.target_key).float()
        layers, tgt = self.layers_and_target(logits, rps)
        return layer_pit_bce(layers, tgt, pos_weight=self.pos_weight, focus=self.focus)


def resolve_focus(focus: float | str, grid: Any, sigma_rev_s: float, *, has_off: bool) -> float:
    """A config's ``focus``: a number, or ``"auto"`` = `layer_focus` on ``grid``."""
    if isinstance(focus, str):
        if focus != "auto":
            raise ValueError(f"focus must be a number or 'auto', got {focus!r}")
        return layer_focus(grid, sigma_rev_s, has_off=has_off)
    return float(focus)
