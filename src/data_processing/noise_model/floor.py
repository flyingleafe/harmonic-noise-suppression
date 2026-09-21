"""The coloured broadband floor: its geometry and its power spectrum.

MOVED verbatim from :mod:`experiments.stochastic_fit.model`
(:func:`interp_matrix`, :func:`se_cholesky`) and
:mod:`experiments.stochastic_fit.revised_phase` (:func:`floor_geometry`,
:func:`floor_power_spectrum`), both of which import them back.

:func:`floor_power_spectrum` is THE one source of truth for the floor: the
fit's graph builds its covariance from it and the renderer shapes its white
noise with its square root, so the fitted floor and the synthesised floor
cannot drift apart. That is exactly why it is here rather than duplicated —
``data_processing`` may not import ``experiments``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from data_processing.noise_model.constants import FLOOR_SHAPE_F_MIN, FLOOR_TILT_REF_HZ

__all__ = ["floor_geometry", "floor_power_spectrum", "interp_matrix", "se_cholesky"]


def se_cholesky(n: int, dt: float, tau: float, jitter: float = 1e-6) -> np.ndarray:
    t = np.arange(n) * dt
    k = np.exp(-0.5 * ((t[:, None] - t[None, :]) / max(tau, 1e-9)) ** 2)
    return np.linalg.cholesky(k + jitter * np.eye(n))


def interp_matrix(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Linear interpolation ``x <- knots`` as a dense ``(len(x), len(knots))``
    matrix; outside the knot range it clamps to the end knots."""
    x = np.asarray(x, dtype=np.float64)
    a = np.zeros((x.size, knots.size))
    idx = np.clip(np.searchsorted(knots, x, side="right") - 1, 0, knots.size - 2)
    w = (x - knots[idx]) / (knots[idx + 1] - knots[idx])
    w = np.clip(w, 0.0, 1.0)
    a[np.arange(x.size), idx] = 1.0 - w
    a[np.arange(x.size), idx + 1] = w
    return a


def floor_geometry(freqs_hz: Any, ctrl_hz: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(shape_matrix, tilt_oct)`` of the floor at ``freqs_hz``.

    Pure geometry — no fitted parameter — so a caller builds it once per grid
    and multiplies the fitted control points into it every step.
    :func:`model.interp_matrix` clamps to the end knots, which is what carries
    the shape past the last control point: the work grid runs to the work
    Nyquist, the control points only to the analysis Nyquist.
    """
    freqs = np.asarray(freqs_hz, dtype=np.float64)
    ctrl = np.asarray(ctrl_hz, dtype=np.float64)
    ctrl_oct = np.log2(ctrl / ctrl[0])
    f_oct = np.log2(np.maximum(freqs, FLOOR_SHAPE_F_MIN) / ctrl[0])
    tilt_oct = np.log2(np.maximum(freqs, FLOOR_SHAPE_F_MIN) / FLOOR_TILT_REF_HZ)
    return interp_matrix(f_oct, ctrl_oct), tilt_oct


def floor_power_spectrum(
    shape_matrix: Any,
    tilt_oct: Any,
    *,
    mean_db: Any,
    ctrl_db: Any,
    tilt_db_oct: Any,
    rate_factor: float,
) -> Any:
    """The floor's power spectrum on the grid ``shape_matrix``/``tilt_oct``
    describe, in WORK-grid periodogram units.

    THE one source of truth: :meth:`_RevisedModel.floor_psd` reads it to build
    ``R_floor`` inside the fit's graph and :func:`render_revised` shapes its
    white noise with its square root, so the fitted floor and the synthesized
    floor cannot drift apart.

    ``rate_factor = sample_rate_work / sample_rate`` because the parameters are
    at the ANALYSIS convention: noise shaped on the work grid reads
    ``sample_rate / sample_rate_work`` times lower once decimated, which is the
    very factor the fit's ``grid_power_factor`` divides back out.

    numpy in, numpy out; torch in, torch out and differentiable — the body is
    one matrix product, one broadcast and one power.
    """
    db = mean_db + shape_matrix @ ctrl_db + tilt_db_oct * tilt_oct
    return float(rate_factor) * 10.0 ** (db / 10.0)
