"""Shared-map rotor tracking, batched on the salience tensor's device.

This is the L0/L1 readout of the salience ladder — the published multi-pitch
recipe: threshold the map, take its per-frame peaks, and continue each rotor
track frame to frame by a minimum-cost assignment between the live tracks and
the current peaks, with a jump cap and a hold on rejection. It used to run as
one SciPy ``linear_sum_assignment`` per frame per sample on the CPU; that made
a 1,320-sample validation panel a matter of minutes, so it could not be the
training-time monitor. This module keeps the rule and removes the loop over
samples: every frame is one batched step over ``(B, ...)`` tensors, and the
only host round-trip is one ``max`` over peak counts.

THE ASSIGNMENT IS EXACT. Track and peak positions live on one line and the
cost is ``|f_track - f_peak|``, so a minimum-cost injective assignment can
always be chosen order-preserving (the L1 Monge property: for ``a <= b`` and
``c <= d``, ``|a-c| + |b-d| <= |a-d| + |b-c|``). :func:`monotone_match` finds
that assignment by a prefix-minimum dynamic programme of ``rows`` ``cummin``
steps, where every track may also stay unmatched at a fixed large price — so
``min(tracks, peaks)`` pairs are matched, as the rectangular Hungarian step
did, and then by least movement. The jump cap is applied AFTER the assignment,
as the reference did: a matched track that would move more than
``max_jump_bins`` holds its value and the peak stays consumed.

WHY THE RULE IS KEPT VERBATIM, TIE-BREAK ASIDE. With L1 costs the assignment
problem has SYSTEMATIC ties — two tracks below two peaks cost the same crossed
or uncrossed — and the reference returned whichever optimum SciPy found. That
cannot be reproduced; this returns the monotone optimum. Alternatives were
measured against the generating trajectories of synthetic maps with dropouts
(``drop_p`` 0 / 0.05 / 0.15 / 0.3): putting the cap INTO the cost (match only
within the cap, then most pairs, then least movement) scores 3-4 % worse than
the reference, because the reference's tie choices happen to drag stale
duplicate tracks onto live peaks; letting stale tracks jump uncapped scores
2-5x worse. The verbatim rule with the monotone tie-break is within +-1 % of
the reference at every dropout rate, and identical to it on every frame whose
assignment is unique (``tests/models/test_salience_tracker.py``).

Peak detection, for the record. A column is BINARY when every active value is
exactly one (a ground-truth map): every active bin is then a peak. Otherwise
the peaks are the interior local maxima above the threshold; adjacent local
maxima (which can only be exact ties, e.g. a saturated ``sigmoid`` plateau) are
thinned to every other bin from the lower end, which is what the reference's
sort-then-suppress did with a stable tie order.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["shared_map_peaks", "monotone_match", "track_shared_map"]

#: Price of leaving a track unmatched. Far above any grid distance in rev/s,
#: so ``min(tracks, peaks)`` pairs are matched before movement is minimised.
HOLD_COST = 1e9


def shared_map_peaks(salience: torch.Tensor, threshold: float) -> torch.Tensor:
    """``(B, G, T)`` map -> ``(B, G, T)`` bool: the bins that are peaks per frame."""
    active = salience > threshold
    binary = (~active | (salience == 1.0)).all(dim=1, keepdim=True)

    local = torch.zeros_like(active)
    inner = salience[:, 1:-1]
    local[:, 1:-1] = active[:, 1:-1] & (inner >= salience[:, :-2]) & (inner >= salience[:, 2:])

    # Thin runs of adjacent local maxima to every other bin, from the low end.
    g = torch.arange(salience.shape[1], device=salience.device)[None, :, None]
    run_start = local.clone()
    run_start[:, 1:] &= ~local[:, :-1]
    start_idx = torch.where(run_start, g, torch.full_like(g, -1)).cummax(dim=1).values
    thinned = local & (((g - start_idx) % 2) == 0)

    return torch.where(binary, active, thinned)


def monotone_match(cost: torch.Tensor, hold: float = HOLD_COST) -> torch.Tensor:
    """Order-preserving minimum-cost injective assignment of rows to columns.

    ``cost`` is ``(B, n, m)`` with ``inf`` for forbidden pairs. Row ``i`` may
    take a column strictly after the column of the previous matched row, or
    stay unmatched at price ``hold``. Returns ``(B, n)`` column indices, ``-1``
    for unmatched rows.
    """
    b, n, m = cost.shape
    inf = torch.full((b, 1), float("inf"), device=cost.device, dtype=cost.dtype)
    # ``before[j]``: best cost of the rows so far using columns strictly before
    # ``j`` (``j = 0 .. m``). No rows yet costs nothing wherever they start.
    before = cost.new_zeros((b, m + 1))
    args, takes = [], []
    for i in range(n):
        value, arg = (before[:, :m] + cost[:, i]).cummin(dim=1)
        matched = torch.cat((inf, value), dim=1)
        held = before + hold
        take = matched < held
        before = torch.where(take, matched, held)
        args.append(arg)
        takes.append(take)

    cols = cost.new_full((b, n), -1, dtype=torch.long)
    pointer = torch.full((b,), m, device=cost.device, dtype=torch.long)
    for i in range(n - 1, -1, -1):
        take = takes[i].gather(1, pointer[:, None]).squeeze(1)
        chosen = args[i].gather(1, (pointer - 1).clamp_min(0)[:, None]).squeeze(1)
        cols[:, i] = torch.where(take, chosen, cols[:, i])
        pointer = torch.where(take, chosen, pointer)
    return cols


def track_shared_map(
    salience: torch.Tensor,
    freqs: np.ndarray | torch.Tensor,
    num_rotors: int,
    *,
    threshold: float,
    max_jump_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(B, G, T)`` salience on grid ``freqs`` -> ``(B, R, T)`` rev/s, ``(B, T)`` merges.

    A frame with no peak decodes to 0.0 for every rotor (silence == zero rotor
    speed) without touching track identity. The first lit frame seeds the
    tracks with its peaks in ascending order, duplicating the lowest peak when
    there are fewer peaks than rotors. Every later lit frame matches the live
    tracks to the peaks by :func:`monotone_match`; a match that would move a
    track by more than ``max_jump_bins`` grid bins is rejected and the track
    holds its value, as does any track left unmatched.
    ``merges`` marks lit frames with fewer peaks than rotors.
    """
    b, n_g, n_t = salience.shape
    device = salience.device
    k = int(num_rotors)
    grid = torch.as_tensor(np.asarray(freqs), dtype=torch.float64, device=device)
    grid_pad = torch.cat((grid, grid.new_full((1,), float("inf"))))

    kept = shared_map_peaks(salience, threshold)
    counts = kept.sum(dim=1)  # (B, T)
    rps = torch.zeros((b, k, n_t), dtype=torch.float64, device=device)
    merges = torch.zeros((b, n_t), dtype=torch.bool, device=device)
    width = int(counts.max())
    if width == 0:
        return rps.float(), merges
    width = max(width, k)

    g = torch.arange(n_g, device=device)[None, :, None]
    bins_all = torch.where(kept, g, torch.full_like(g, n_g)).sort(dim=1).values[:, :width]
    bins_all = bins_all.permute(2, 0, 1)  # (T, B, width), padded with n_g
    counts = counts.transpose(0, 1)  # (T, B)

    rotor = torch.arange(k, device=device)[None, :]
    cur = torch.zeros((b, k), dtype=torch.long, device=device)
    alive = torch.zeros((b,), dtype=torch.bool, device=device)
    for t in range(n_t):
        p_bins = bins_all[t]
        p_freq = grid_pad[p_bins]
        count = counts[t]
        lit = count > 0
        seed = lit & ~alive
        step = lit & alive

        seeded = torch.where(rotor < count[:, None], p_bins[:, :k], p_bins[:, :1])

        sorted_bins, order = cur.sort(dim=1, stable=True)
        cost = (grid[sorted_bins][:, :, None] - p_freq[:, None, :]).abs()
        cols = monotone_match(cost)
        new_bins = p_bins.gather(1, cols.clamp_min(0))
        accept = (cols >= 0) & ((new_bins - sorted_bins).abs() <= max_jump_bins)
        moved = torch.where(accept, new_bins, sorted_bins)
        stepped = torch.empty_like(cur).scatter_(1, order, moved)

        cur = torch.where(seed[:, None], seeded, torch.where(step[:, None], stepped, cur))
        rps[:, :, t] = torch.where(lit[:, None], grid[cur], 0.0)
        merges[:, t] = step & (count < k)
        alive = alive | seed

    return rps.float(), merges
