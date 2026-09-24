"""The v2 front-end geometry the RENDERER reads.

MOVED from :mod:`experiments.noise_model.spectrum`, which imports every name
back and keeps the torch forward model (:func:`bench_model`,
:func:`flight_model` and their grids) upstairs: a training stream renders, it
never evaluates an expected periodogram, and dragging the Pyro-adjacent forward
model into ``data_processing`` would buy nothing.

ONE definition of the floor's control ladder and of the order cap, read by the
fit's grids AND by :func:`.render.render_noise`, so a fitted ``floor_shape_z``
means the same curve in the fit and in the render.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from data_processing.noise_model.constants import (
    FLOOR_SHAPE_F_MIN,
    FLOOR_SHAPE_N_CTRL,
    FLOOR_SHAPE_OCT,
    FLOOR_SHAPE_STD_DB,
)
from data_processing.noise_model.floor import se_cholesky

__all__ = [
    "FLIGHT_HOP",
    "FLIGHT_N_FFT",
    "FLIGHT_SR",
    "SAMPLE_RATE_WORK",
    "floor_ctrl_hz",
    "floor_shape_chol",
    "floor_shape_db",
    "k_max_for_carrier",
]

#: The flight front end of the v2 campaign (``docs/explainers/
#: noise-model-v2-plan.qmd``, "Where and how the rig is fitted"). NOT C4's
#: 16384/1024: a long window smears a moving line by ``k |df/dt| T``.
FLIGHT_N_FFT = 2048
FLIGHT_HOP = 512
FLIGHT_SR = 16000
SAMPLE_RATE_WORK = 64000


def floor_ctrl_hz(sr: int) -> np.ndarray:
    """The floor's shape control points: C4's geometric ladder to the Nyquist.

    ONE definition, read by :func:`bench_grid`, :func:`flight_grid` AND
    :func:`.render.render_noise`, so a fitted ``floor_shape_z`` means the same
    curve in the fit and in the render.
    """
    return np.geomspace(FLOOR_SHAPE_F_MIN, 0.5 * float(sr), FLOOR_SHAPE_N_CTRL)


def floor_shape_chol(ctrl_hz: np.ndarray) -> np.ndarray:
    """The squared-exponential Cholesky of the shape GP on ``ctrl_hz``."""
    oct_ = np.log2(np.asarray(ctrl_hz, dtype=np.float64) / float(ctrl_hz[0]))
    return se_cholesky(FLOOR_SHAPE_N_CTRL, float(oct_[1] - oct_[0]), FLOOR_SHAPE_OCT)


def floor_shape_db(
    shape_z: np.ndarray, *, sr: int, scale_db: float = FLOOR_SHAPE_STD_DB
) -> np.ndarray:
    """``scale_db * (chol @ z)``: the dB control values from the GP coordinate,
    the numpy twin of :meth:`FloorBasis.shape_db`. v2 scales by the fixed
    ``FLOOR_SHAPE_STD_DB``; v3 by its MEASURED ``sigma_B`` (``floor_shape_sd_db``)."""
    chol = floor_shape_chol(floor_ctrl_hz(sr))
    return float(scale_db) * (chol @ np.asarray(shape_z, dtype=np.float64))


def k_max_for_carrier(carrier_rev_s: Any, sr: int, *, k_cap: int | None = None) -> int:
    """Highest order whose line sits strictly below the analysis Nyquist.

    Orders above it were removed from the real recording by its own front end
    (a decimator's anti-alias, never an alias), and they contribute to the
    observed band only through line skirts far below the floor — so they are
    not modelled. ``k_cap`` clamps the count to the profile's width.
    """
    f = float(np.max(np.asarray(carrier_rev_s, dtype=np.float64)))
    if f <= 0.0:
        raise ValueError(f"carrier must be positive, got {f}")
    k = int(math.floor(0.5 * float(sr) / f - 1e-9))
    return max(1, k if k_cap is None else min(k, int(k_cap)))
