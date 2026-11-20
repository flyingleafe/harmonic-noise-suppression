"""The one contract every cruise model of the flight-regime campaign obeys.

The campaign (``docs/experiments/rps-flight-regimes.md``) compares the
structured rotor-speed model with literature baselines on HELD-OUT flights.
Every arm, ours or a baseline, is scored by the same code, so every arm
implements :class:`CruiseModel` over the same data unit, the BLOCK:

* an array ``(4, n)`` of rotor speeds in rev/s, mixer rotor order
  ``[RFront, LFront, LBack, RBack]``, on the arm's own sample rate
  ``rate_hz``;
* one contiguous piece of settled cruise, at most :data:`BLOCK_S` seconds,
  DEMEANED PER ROTOR (the level is a per-flight offset that every arm draws
  from the same law, so it is not part of the comparison);
* its first :func:`burn_samples` samples only condition the predictive
  density and are never scored, so an autoregression of any order below
  that has its lags, and a state-space filter has forgotten its prior.

:func:`cut_blocks` is the only place a block is made.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

#: Rotors of every campaign rig.
NUM_ROTORS = 4

#: Longest block (s), the shortest one worth scoring (s), and the leading part
#: of every block that conditions but is not scored (s).
BLOCK_S = 30.0
MIN_BLOCK_S = 5.0
BURN_S = 1.0


def burn_samples(rate_hz: float) -> int:
    """Samples of a block that condition the predictive density unscored."""
    return int(round(BURN_S * float(rate_hz)))


def cut_blocks(
    segment: np.ndarray,
    rate_hz: float,
    *,
    block_s: float = BLOCK_S,
    min_s: float = MIN_BLOCK_S,
) -> list[np.ndarray]:
    """Cut one settled-cruise segment ``(4, n)`` into consecutive blocks.

    Blocks are ``block_s`` long, do not overlap and are demeaned per rotor; a
    tail shorter than ``min_s`` is dropped, as is a segment with a non-finite
    sample (a telemetry gap has no predictive density).
    """
    seg = np.asarray(segment, dtype=np.float64)
    if seg.ndim != 2 or seg.shape[0] != NUM_ROTORS:
        raise ValueError(f"segment must be (4, n), got {seg.shape}")
    if not np.isfinite(seg).all():
        return []
    size, least = int(round(block_s * rate_hz)), int(round(min_s * rate_hz))
    out = []
    for start in range(0, seg.shape[1], size):
        block = seg[:, start : start + size]
        if block.shape[1] >= least:
            out.append(np.ascontiguousarray(block - block.mean(axis=1, keepdims=True)))
    return out


@runtime_checkable
class CruiseModel(Protocol):
    """A fitted generative model of the cruise fluctuation, at ``rate_hz``.

    ``name`` is the arm id, ``n_params`` the number of free parameters the fit
    estimated from data (reported next to every score).
    """

    name: str
    rate_hz: float
    n_params: int

    def logpdf(self, block: np.ndarray, burn: int) -> np.ndarray:
        """``(n - burn,)`` natural-log one-step-ahead predictive densities
        ``log p(y_t | y_0 .. y_{t-1})`` for ``t = burn .. n-1`` of a block
        ``(4, n)``. The first ``burn`` samples are filtered, not scored."""
        ...

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``(4, n)`` stationary zero-mean realisation; all randomness from
        ``rng``."""
        ...

    def forecast(
        self, history: np.ndarray, horizon: int, n_paths: int, rng: np.random.Generator
    ) -> np.ndarray:
        """``(n_paths, 4, horizon)`` draws of the ``horizon`` samples that
        follow ``history`` ``(4, h)``, given all of it."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable parameters (the per-drone parameter set)."""
        ...


__all__ = [
    "BLOCK_S",
    "BURN_S",
    "MIN_BLOCK_S",
    "NUM_ROTORS",
    "CruiseModel",
    "burn_samples",
    "cut_blocks",
]
