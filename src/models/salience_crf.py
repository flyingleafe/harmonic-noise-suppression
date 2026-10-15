"""Per-rotor salience layers with a CRF readout — a LOSSLESS encode/decode pair.

THE GATE THIS MODULE EXISTS TO PASS. Encoding real training trajectories into a
salience map and decoding them back must be an isomorphism: whatever a model
loses, the representation itself must lose nothing. Measured on 40 clips of real
training telemetry (DREGON ``in_flight_noise`` minus the held-out room1, plus
FLY125; speeds 0.00-91.58 rev/s, per-frame |d rps| mean 0.718, p99 3.008, max
15.301), on a 0-150 rev/s grid at 0.5 rev/s:

    encode/decode pair                              max err     mean err
    triangular kernel, SHARED map, Hungarian        83.53       8.24
    Gaussian layers, CRF + log-parabolic            2.22e-16    2.22e-16

The old pair loses 8.24 rev/s on a PERFECT target, and 39-45% of frames land
more than 0.5 rev/s away. That is an oracle floor: no model reading that
representation can score better, however well it predicts. Three causes, all
fixed here:

1. ONE MAP CANNOT HOLD FOUR ROTORS. DREGON's rotor pairs sit 0.13-0.86 rev/s
   apart, inside a single 0.5 rev/s bin, so a shared map has fewer peaks than
   rotors. One layer per rotor removes the failure completely.
2. THE KERNEL MUST BE GAUSSIAN IN REV/S. ``log`` of a Gaussian is a parabola
   *globally*, so a three-point parabolic fit through ANY three bins — uniform
   or log-spaced, the fit is done in rev/s coordinates — recovers the peak
   exactly. The width ``sigma_rev_s`` is one number in rev/s shared by every
   grid, so a log grid and the linear grid carry the same target.
3. ZERO IS A RATE ON THE LINEAR GRID AND A STATE ON A LOG GRID. The linear
   0-160 grid has bin 0 at 0.0 rev/s like any other rate bin: a stopped rotor
   is a Gaussian centred at 0 (the convention of every pre-2026-10 per-rotor
   run), nothing is special-cased, which is what keeps the linear grid simple
   and stable (Dmitrii, 2026-10-15). A log grid has no zero rate of its own,
   so its models emit ``G_log + 1`` bins (`log_off_grid`) whose bin 0 is the
   OFF state: a stopped rotor (below `OFF_RPS`) puts its whole mass there and
   a running rotor puts nothing there. Which convention applies is the grid
   kind (``has_off``: ``log_off`` -> True, ``linear`` -> False), threaded
   explicitly through the target, the focus, the transition cost and the
   readout; a value test cannot tell the two apart (both start at 0.0).

THE CHAIN. `crf_decode_layers` runs one best path per layer under a DENSE
transition cost in rev/s (`transition_penalty`): free up to
``max_step_rev_s`` per frame (`MAX_STEP_REV_S`, sized from the telemetry's
23.8 rev/s per-frame extreme), quadratic past it, forbidden past four times it —
on any grid, because the cost reads the rates, not the bin offsets. On a log
grid transitions into and out of the off bin cost nothing, so a 0 -> 60 rev/s
start in one frame is decodable; on the linear grid the same start is a 60
rev/s jump under the ordinary cost (quadratic past 25, forbidden past 100).
On the 164-bin grid the dense ``G x G`` step is also cheaper than the
old ``(2 * span + 1) x G`` band (span 102 at 25 rev/s / 0.977).

`band_for_rev_s` is the older uniform-grid band without an off state; only
`losses.LayerPITCRFLoss` still uses it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from models import comb_crf
from models.multif0.utils import linear_freq_grid

__all__ = [
    "RP_LINEAR_STEP",
    "RP_LINEAR_GRID",
    "OFF_RPS",
    "MAX_STEP_REV_S",
    "log_off_grid",
    "grid_from_spec",
    "gaussian_layer_target",
    "expected_layer_mass",
    "layer_focus",
    "transition_penalty",
    "refine_peak",
    "crf_decode_layers",
    "band_for_rev_s",
]

#: The protocol's output step: level 0 of the HRSPNet pyramid, 16000 / 16384.
RP_LINEAR_STEP = 0.9765625
#: The output grid of every per-rotor model (``docs/experiments/
#: paper-real-protocol.md``): ``g * 0.9765625`` rev/s, ``g = 0..163``. Bin 0 is
#: the 0 rev/s RATE bin (no off state). The real data peaks at 100.8 rev/s and
#: ``rate_warp`` at alpha 1.5 reaches 151, so 159.2 covers both.
RP_LINEAR_GRID = linear_freq_grid(0.0, 163 * RP_LINEAR_STEP, 164)

#: On a grid WITH an off state, a rotor below this is STOPPED: its target is
#: bin 0 alone. Half the linear step, so a stopped rotor and the first rate bin
#: never share a label. Linear grids never read it.
OFF_RPS = 0.5

#: The per-frame change the transition cost makes FREE, in rev/s. 40 clips of
#: real training telemetry on the 32 ms frame grid move 0.476 rev/s per frame
#: on average, 2.65 at the 99th percentile and 23.8 at the extreme; at 16 the
#: perfect-target round trip loses 0.203 rev/s on its worst frame, at 25 it
#: loses 1.4e-14.
MAX_STEP_REV_S = 25.0

_GRID_KEYS = {
    "linear": {"fmin", "fmax", "bins"},
    "log_off": {"fmin", "bins_per_octave", "n_bins"},
}


def log_off_grid(fmin: float, bins_per_octave: int, n_bins: int) -> np.ndarray:
    """``[0.0, fmin * 2**(b / bins_per_octave) for b < n_bins]`` — ``n_bins + 1`` entries.

    A model's own log grid plus the off state at bin 0.
    """
    b = np.arange(int(n_bins), dtype=np.float64)
    return np.concatenate([[0.0], float(fmin) * 2.0 ** (b / int(bins_per_octave))])


def grid_from_spec(spec: Mapping[str, Any]) -> tuple[np.ndarray, bool]:
    """A per-rotor output grid (rev/s) and its ``has_off`` flag from a config spec.

    ``{kind: linear, fmin, fmax, bins}`` -> a plain rate grid, ``has_off``
    False (bin 0 is whatever ``fmin`` says, 0.0 rev/s on the protocol grid).
    ``{kind: log_off, fmin, bins_per_octave, n_bins}`` -> ``n_bins`` log bins
    behind an off state at bin 0, ``has_off`` True. Unknown or missing keys
    raise, so a typo cannot silently build a different grid than the model's.
    """
    params = {str(k): v for k, v in dict(spec).items()}
    kind = params.pop("kind", None)
    if kind not in _GRID_KEYS:
        raise ValueError(f"grid kind must be one of {sorted(_GRID_KEYS)}, got {kind!r}")
    if set(params) != _GRID_KEYS[kind]:
        raise ValueError(
            f"grid {kind!r} takes keys {sorted(_GRID_KEYS[kind])}, got {sorted(params)}"
        )
    if kind == "linear":
        grid = linear_freq_grid(float(params["fmin"]), float(params["fmax"]), int(params["bins"]))
        has_off = False
    else:
        grid = log_off_grid(
            float(params["fmin"]), int(params["bins_per_octave"]), int(params["n_bins"])
        )
        has_off = True
    _check_grid(grid, has_off)
    return grid, has_off


def _check_grid(grid: np.ndarray, has_off: bool) -> None:
    g = np.asarray(grid, dtype=np.float64)
    if g.ndim != 1 or g.size < 4:
        raise ValueError(f"a per-rotor grid needs >= 4 bins, got shape {g.shape}")
    if has_off and g[0] != 0.0:
        raise ValueError(f"a grid with an off state has it at bin 0 (0.0 rev/s); got {g[0]}")
    if not np.all(np.diff(g) > 0):
        raise ValueError("a per-rotor grid must be strictly ascending")


def gaussian_layer_target(
    rps: torch.Tensor, grid: Any, sigma_rev_s: float = 1.0, *, has_off: bool
) -> torch.Tensor:
    """``(B, R, T)`` rev/s -> ``(B, R, G, T)``: one Gaussian layer per rotor.

    Without an off state (linear grids) every rotor, stopped or not, is
    ``exp(-(f_g - r)^2 / 2 sigma^2)`` over all bins — a stopped rotor is a
    peak at 0 rev/s. With one (log grids) a running rotor puts that Gaussian
    on bins ``g >= 1`` and 0 on bin 0; a stopped rotor (``r < OFF_RPS``) is 1
    on bin 0 and 0 everywhere else.
    """
    g_np = np.asarray(grid, dtype=np.float64)
    _check_grid(g_np, has_off)
    if not has_off:
        g = torch.as_tensor(g_np, dtype=rps.dtype, device=rps.device)
        d = (g.view(1, 1, -1, 1) - rps.unsqueeze(2)) / float(sigma_rev_s)
        return torch.exp(-0.5 * d * d)
    g = torch.as_tensor(g_np[1:], dtype=rps.dtype, device=rps.device)
    stopped = (rps < OFF_RPS).unsqueeze(2)  # (B, R, 1, T)
    d = (g.view(1, 1, -1, 1) - rps.unsqueeze(2)) / float(sigma_rev_s)
    on = torch.exp(-0.5 * d * d) * (~stopped).to(rps.dtype)
    return torch.cat([stopped.to(rps.dtype), on], dim=2)


def expected_layer_mass(grid: Any, sigma_rev_s: float = 1.0, *, has_off: bool) -> float:
    """Mean target mass of one running rotor on ``grid``, in bins.

    Averaged over the rotor rates the grid represents: the `RP_LINEAR_GRID`
    rates inside the grid's rate span (bins ``1..`` with an off state, every
    bin without). On a linear grid this is ``sqrt(2 pi) sigma / step``; on a
    log grid it falls with the rate as the bins widen, hence the average.
    """
    g = np.asarray(grid, dtype=np.float64)
    _check_grid(g, has_off)
    rate_bins = g[1:] if has_off else g
    rates = RP_LINEAR_GRID[(rate_bins[0] <= RP_LINEAR_GRID) & (g[-1] >= RP_LINEAR_GRID)]
    d = (rate_bins[:, None] - rates[None, :]) / float(sigma_rev_s)
    return float(np.exp(-0.5 * d * d).sum(axis=0).mean())


def layer_focus(grid: Any, sigma_rev_s: float = 1.0, *, has_off: bool) -> float:
    """``G / expected_layer_mass``: the focus that splits the BCE weight evenly
    between a layer's Gaussian mass and the rest of its ``G`` bins."""
    return float(len(np.asarray(grid))) / expected_layer_mass(grid, sigma_rev_s, has_off=has_off)


def transition_penalty(
    grid: Any,
    max_step_rev_s: float = MAX_STEP_REV_S,
    stiff: float = 40.0,
    *,
    has_off: bool,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """``(G, G)`` cost of moving from bin ``j`` (row) to bin ``i`` (column).

    In rev/s, on any grid: 0 up to ``max_step_rev_s``, ``stiff * (excess /
    max_step_rev_s)^2`` past it, forbidden (a huge cost) past four times it —
    the hinge of `comb_crf.band_penalty`, with the truncation at the same
    multiple. With an off state, row 0 and column 0 are zero: the off state is
    entered and left for free. Without one, bin 0 is a rate like any other.
    """
    g = np.asarray(grid, dtype=np.float64)
    _check_grid(g, has_off)
    m = max(float(max_step_rev_s), 1e-9)
    d = np.abs(g[:, None] - g[None, :])
    pen = float(stiff) * (np.clip(d - m, 0.0, None) / m) ** 2
    pen[d > 4.0 * m] = np.inf
    if has_off:
        pen[0, :] = 0.0
        pen[:, 0] = 0.0
    big = min(1e30, float(torch.finfo(dtype).max) / 4.0)
    return torch.as_tensor(np.minimum(pen, big), dtype=dtype, device=device)


def _viterbi_dense(scores: torch.Tensor, pen: torch.Tensor) -> torch.Tensor:
    """Best path: ``(N, G, T)`` scores, ``(G, G)`` cost -> ``(N, T)`` bin indices."""
    n, n_g, n_t = scores.shape
    best = scores[:, :, 0]
    back = torch.zeros((n_t, n, n_g), dtype=torch.int32, device=scores.device)
    for t in range(1, n_t):
        val, src = (best.unsqueeze(2) - pen.unsqueeze(0)).max(dim=1)
        back[t] = src.to(torch.int32)
        best = val + scores[:, :, t]
    path = torch.zeros((n, n_t), dtype=torch.long, device=scores.device)
    path[:, -1] = best.argmax(dim=1)
    for t in range(n_t - 1, 0, -1):
        path[:, t - 1] = back[t].gather(1, path[:, t : t + 1]).squeeze(1).long()
    return path


def refine_peak(
    scores: torch.Tensor, idx: torch.Tensor, grid: torch.Tensor, *, has_off: bool
) -> torch.Tensor:
    """Sub-bin rate at ``idx`` by a three-point parabola in rev/s coordinates.

    ``scores`` is ``(N, G, T)`` LOG-domain, ``idx`` ``(N, T)`` bin indices,
    ``grid`` ``(G,)`` rev/s. With an off state at bin 0 the triplet always lies
    in the rate bins ``1..G-1`` (centre clamped to ``2..G-2``), so it never
    straddles the off state, and ``idx == 0`` reads exactly 0.0. Without one
    the triplet runs over all bins (centre clamped to ``1..G-2``). A flat
    triplet keeps the bin's own rate; the vertex is clamped to the triplet's
    span. On a Gaussian in rev/s the vertex is exact on any grid.
    """
    n_g = grid.numel()
    i0 = idx.clamp(2 if has_off else 1, n_g - 2)
    x0, x1, x2 = grid[i0 - 1], grid[i0], grid[i0 + 1]
    y0 = scores.gather(1, (i0 - 1).unsqueeze(1)).squeeze(1)
    y1 = scores.gather(1, i0.unsqueeze(1)).squeeze(1)
    y2 = scores.gather(1, (i0 + 1).unsqueeze(1)).squeeze(1)
    a, c = x1 - x0, x1 - x2
    num = a * a * (y1 - y2) - c * c * (y1 - y0)
    den = a * (y1 - y2) - c * (y1 - y0)
    # A fixed tolerance underflows to zero in float32; exact zero is the case.
    curved = den != 0
    vertex = x1 - 0.5 * num / torch.where(curved, den, torch.ones_like(den))
    vertex = torch.minimum(torch.maximum(vertex, x0), x2)
    rate = torch.where(curved, vertex, grid[idx])
    if has_off:
        rate = torch.where(idx == 0, torch.zeros_like(rate), rate)
    return rate


def crf_decode_layers(
    layers: torch.Tensor,
    grid: Any,
    *,
    has_off: bool,
    max_step_rev_s: float = MAX_STEP_REV_S,
    stiff: float = 40.0,
    logits: bool = False,
    subgrid: bool = True,
) -> torch.Tensor:
    """``(B, R, G, T)`` per-rotor layers -> ``(B, R, T)`` rev/s.

    One best path per layer under `transition_penalty`, so there is no
    assignment step and no threshold: a stopped rotor is the path sitting at
    bin 0 (the off state on a log grid, the 0 rev/s rate bin on the linear
    grid). ``logits=True`` treats the input as already log-domain; otherwise
    it is a probability-like map and is logged here. ``subgrid`` refines each
    frame with `refine_peak`.
    """
    g_np = np.asarray(grid, dtype=np.float64)
    _check_grid(g_np, has_off)
    s = layers if logits else torch.log(layers.clamp_min(torch.finfo(layers.dtype).tiny))
    b, r, n_g, n_t = s.shape
    if n_g != g_np.size:
        raise ValueError(f"layers carry {n_g} bins, the grid has {g_np.size}")
    pen = transition_penalty(
        g_np, max_step_rev_s, stiff, has_off=has_off, dtype=s.dtype, device=s.device
    )
    flat = s.reshape(b * r, n_g, n_t)
    path = _viterbi_dense(flat, pen)
    g = torch.as_tensor(g_np, dtype=s.dtype, device=s.device)
    rate = refine_peak(flat, path, g, has_off=has_off) if subgrid else g[path]
    return rate.reshape(b, r, n_t)


def band_for_rev_s(
    max_step_rev_s: float, grid_step: float, stiff: float = 40.0, dtype=torch.float32
):
    """Uniform-grid band ``(span, pen)`` that ADMITS a per-frame change of
    ``max_step_rev_s`` — `comb_crf`'s Toeplitz form, with no off state.

    Only `losses.LayerPITCRFLoss` reads it; the decoder uses the dense,
    off-aware `transition_penalty`.
    """
    return comb_crf.band_penalty(max(max_step_rev_s / grid_step, 1e-9), stiff, dtype=dtype)
