"""The batched shared-map tracker against the CPU/SciPy tracker it replaced.

The oracle below is the deleted ``models.multif0.utils._extract_peaks_per_frame``
+ ``_hungarian_tracking`` pair, kept verbatim (minus the merge bookkeeping the
decoder discarded). The two agree exactly on every frame whose assignment is
unique; where the L1 assignment has several optima the oracle's answer is
SciPy's implementation-defined choice and the tracker returns the monotone
one, so the maps below are built to keep assignments unique.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from models.multif0.utils import cqt_freq_grid, linear_freq_grid
from models.salience_tracker import monotone_match, shared_map_peaks, track_shared_map


def _oracle_peaks(salience: np.ndarray, freqs: np.ndarray, threshold: float) -> list:
    n_bins, n_t = salience.shape
    peaks = []
    for t in range(n_t):
        col = salience[:, t]
        active = col > threshold
        if not active.any():
            peaks.append(np.array([], dtype=np.float64))
            continue
        vals = col[active]
        if np.all((vals == 0.0) | (vals == 1.0)):
            peaks.append(freqs[active])
            continue
        local_max = np.zeros(n_bins, dtype=bool)
        for b in range(1, n_bins - 1):
            if col[b] > threshold and col[b] >= col[b - 1] and col[b] >= col[b + 1]:
                local_max[b] = True
        lm_bins = np.where(local_max)[0]
        order = np.argsort(-col[lm_bins], kind="stable")
        keep = np.ones(len(lm_bins), dtype=bool)
        for i in range(len(order)):
            if not keep[order[i]]:
                continue
            for j in range(i + 1, len(order)):
                if abs(lm_bins[order[j]] - lm_bins[order[i]]) < 2:
                    keep[order[j]] = False
        peaks.append(freqs[lm_bins[keep]])
    return peaks


def _oracle_track(peaks_per_frame: list, k: int, freqs: np.ndarray, max_jump: int) -> np.ndarray:
    n_t = len(peaks_per_frame)
    rps = np.full((k, n_t), np.nan)
    current = np.full(k, np.nan)
    for t in range(n_t):
        p = np.asarray(peaks_per_frame[t], dtype=np.float64)
        if len(p) == 0:
            rps[:, t] = 0.0
            continue
        if t == 0 or np.all(np.isnan(current)):
            order = np.argsort(p)
            n_use = min(k, len(p))
            for i in range(n_use):
                current[i] = p[order[i]]
            for i in range(n_use, k):
                current[i] = p[order[0]]
            rps[:, t] = current
            continue
        cost = np.abs(current[:, None] - p[None, :])
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            new_bin = np.abs(freqs - p[c]).argmin()
            prev_bin = np.abs(freqs - current[r]).argmin()
            if abs(new_bin - prev_bin) > max_jump:
                rps[r, t] = current[r]
                continue
            current[r] = p[c]
            rps[r, t] = p[c]
        for r in range(k):
            if np.isnan(rps[r, t]):
                rps[r, t] = current[r]
    return np.nan_to_num(rps, nan=0.0)


def _oracle(salience: torch.Tensor, freqs: np.ndarray, threshold: float, max_jump: int):
    out = [
        _oracle_track(_oracle_peaks(row.numpy(), freqs, threshold), 4, freqs, max_jump)
        for row in salience
    ]
    return torch.from_numpy(np.stack(out)).float()


def _drone_map(freqs: np.ndarray, seed: int, n_t: int = 80, dropout: float = 0.0):
    """Four Gaussian bumps drifting slowly, distinct spacings so every
    assignment is unique; optional per-rotor dropouts and dark frames."""
    rng = np.random.default_rng(seed)
    n_g = len(freqs)
    sal = np.zeros((n_g, n_t), dtype=np.float32)
    f0 = np.array([0.15, 0.32, 0.55, 0.81]) * (freqs[-1] - freqs[0]) + freqs[0]
    traj = f0[:, None] + np.cumsum(rng.normal(0.0, 0.08, (4, n_t)), axis=1)
    for t in range(n_t):
        if 30 <= t < 34:
            continue  # dark frames
        for r in range(4):
            if dropout and rng.random() < dropout:
                continue
            centre = np.abs(freqs - traj[r, t]).argmin()
            bump = np.exp(-0.5 * ((np.arange(n_g) - centre) / 1.5) ** 2) * rng.uniform(0.6, 0.95)
            sal[:, t] = np.maximum(sal[:, t], bump)
    return torch.from_numpy(sal), traj


LINEAR = linear_freq_grid(20.0, 130.0, 720)
CQT = cqt_freq_grid(fmin=32.7, n_octaves=6, over_sample=5)


@pytest.mark.parametrize("freqs,max_jump", [(LINEAR, 10), (CQT, 3)])
def test_peaks_match_oracle_on_soft_binary_and_saturated_columns(freqs, max_jump):
    sal, _ = _drone_map(freqs, seed=0)
    sal = sal.clone()
    sal[:, 5] = (sal[:, 5] > 0.5).float()  # a binary column: every active bin is a peak
    sal[:, 6] = torch.clamp(sal[:, 6] * 4, max=1.0)  # saturated plateaus among soft values
    kept = shared_map_peaks(sal[None], 0.3)[0]
    oracle = _oracle_peaks(sal.numpy(), freqs, 0.3)
    for t in range(sal.shape[1]):
        assert np.array_equal(freqs[kept[:, t].numpy()], np.sort(oracle[t])), t


def test_monotone_match_returns_the_hungarian_optimum_when_unique():
    rng = np.random.default_rng(3)
    for _ in range(200):
        n, m = int(rng.integers(1, 5)), int(rng.integers(1, 7))
        tracks = np.sort(rng.uniform(0, 100, n))
        peaks = np.sort(rng.uniform(0, 100, m))
        cost = np.abs(tracks[:, None] - peaks[None, :])
        rows, cols = linear_sum_assignment(cost)
        ours = monotone_match(torch.from_numpy(cost)[None])[0].numpy()
        matched = ours >= 0
        assert matched.sum() == min(n, m)
        # Same total cost as the Hungarian optimum (assignment itself may differ on ties).
        assert cost[np.arange(n)[matched], ours[matched]].sum() == pytest.approx(
            cost[rows, cols].sum()
        )


@pytest.mark.parametrize("freqs,max_jump", [(LINEAR, 10), (CQT, 3)])
def test_tracker_equals_oracle_on_unique_assignments(freqs, max_jump):
    sal = torch.stack([_drone_map(freqs, seed=s)[0] for s in range(4)])
    ours, merges = track_shared_map(sal, freqs, 4, threshold=0.3, max_jump_bins=max_jump)
    want = _oracle(sal, freqs, 0.3, max_jump)
    assert torch.equal(ours, want)
    assert torch.all(ours[:, :, 30:34] == 0.0)
    assert not merges.any()


def test_dropped_rotor_holds_and_the_rest_follow_their_peaks():
    sal, _ = _drone_map(LINEAR, seed=7)
    kept = shared_map_peaks(sal[None], 0.3)[0]
    bins_9 = torch.nonzero(kept[:, 9]).flatten()
    bins_10 = torch.nonzero(kept[:, 10]).flatten()
    assert len(bins_9) == len(bins_10) == 4
    # Erase the second rotor's bump at frame 10 only.
    sal = sal.clone()
    lo, hi = int(bins_10[1]) - 6, int(bins_10[1]) + 7
    sal[lo:hi, 10] = 0.0
    ours, merges = track_shared_map(sal[None], LINEAR, 4, threshold=0.3, max_jump_bins=10)
    want = LINEAR[bins_10.numpy()].copy()
    want[1] = LINEAR[int(bins_9[1])]  # held from the previous frame
    assert np.array_equal(np.sort(ours[0, :, 10].numpy()), np.sort(want).astype(np.float32))
    assert bool(merges[0, 10]) and not bool(merges[0, 9])


def test_batched_result_is_independent_of_batch_composition():
    a, _ = _drone_map(LINEAR, seed=1, dropout=0.2)
    b, _ = _drone_map(LINEAR, seed=2, dropout=0.2)
    single_a = track_shared_map(a[None], LINEAR, 4, threshold=0.3, max_jump_bins=10)[0]
    both = track_shared_map(torch.stack([a, b]), LINEAR, 4, threshold=0.3, max_jump_bins=10)[0]
    assert torch.equal(single_a[0], both[0])
