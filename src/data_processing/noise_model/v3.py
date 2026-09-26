"""The v3 additions a RENDERER needs: the block wander and the wind shape.

Noise model v3 (``docs/explainers/noise-model-v3-wander.qmd``) keeps the v2
forward model and adds two things that a data stream must be able to draw,
so they live here rather than in :mod:`experiments.noise_model` (which the fit
reads them back from):

* **the slow amplitude wander** (§1.4, §2.3, §2.4) — per block of ``block_s``
  seconds, dB latents that follow a stationary Ornstein-Uhlenbeck process
  across blocks: ``d_r(b)`` per rotor and ``v_rk(b)`` per line on the comb,
  ``u(b)`` for the floor level and ``u_j(b)`` per floor control point. The
  ``(sigma, tau)`` of each are MEASURED on real windows and fixed
  (:class:`Wander`, read from ``results/noise_v3/wander/<rig>.json``); the fit
  estimates the per-window tracks as nuisance and the renderer draws FRESH
  ones per clip (:func:`ou_blocks`);
* **the static per-microphone wind term** ``W_m(f)`` of DREGON (§2.4, §2.5) —
  one free level per mic times the fixed shape :func:`wind_shape`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from data_processing.noise_model.floor import se_cholesky

__all__ = [
    "WIND_CORNER_HZ",
    "WIND_CUT_HZ",
    "WIND_TAPER_HZ",
    "Wander",
    "ou_blocks",
    "wind_shape",
    "wind_shape_spec",
]

#: The wind shape's corner: flat below, ``-6 dB/oct`` (``f^-2`` in power) above.
WIND_CORNER_HZ = 100.0
#: Where the raised-cosine taper to zero starts...
WIND_TAPER_HZ = 400.0
#: ...and where the shape is exactly zero from.
WIND_CUT_HZ = 500.0


def wind_shape(freqs_hz: Any) -> np.ndarray:
    """The FIXED power shape of the per-mic wind term, ``s(f)`` in ``[0, 1]``.

    Exactly::

        s(f) = 1                                          f <= 100 Hz
        s(f) = (100 / f)^2                          100 < f <= 400 Hz
        s(f) = (100 / f)^2 * (1 + cos(pi (f - 400) / 100)) / 2   400 < f < 500 Hz
        s(f) = 0                                          f >= 500 Hz

    i.e. a flat low end, a ``-6.02 dB/oct`` roll-off above 100 Hz (``-12 dB``
    at 400 Hz) and a raised-cosine taper that reaches exactly zero at 500 Hz,
    so the term is identically zero above the band the DREGON rank test found
    the per-capsule excess in (``results/noise_v2/mic_gains/findings.md``:
    4.9 dB spread below 500 Hz, 1.4 dB above). The taper, not a step, keeps
    the window response of the term inside a few bins of 500 Hz.
    """
    f = np.abs(np.asarray(freqs_hz, dtype=np.float64))
    roll = (WIND_CORNER_HZ / np.maximum(f, WIND_CORNER_HZ)) ** 2
    x = np.clip((f - WIND_TAPER_HZ) / (WIND_CUT_HZ - WIND_TAPER_HZ), 0.0, 1.0)
    taper = 0.5 * (1.0 + np.cos(np.pi * x))
    return np.where(f >= WIND_CUT_HZ, 0.0, roll * taper)


def wind_shape_spec() -> dict[str, Any]:
    """The shape of :func:`wind_shape`, as the fit JSON records it."""
    return dict(
        corner_hz=WIND_CORNER_HZ,
        slope_db_oct=-20.0 * math.log10(2.0),
        taper_hz=[WIND_TAPER_HZ, WIND_CUT_HZ],
        taper="raised cosine to exactly zero",
        zero_above_hz=WIND_CUT_HZ,
        level="wind_db[m] is 10 log10 of W_m(f) on the flat part (f <= corner), in the "
        "model's periodogram units; W_m is static (no speed law, no channel gain) and is "
        "added to the floor",
    )


@dataclass(frozen=True)
class Wander:
    """The MEASURED block-wander hyperparameters of one rig (explainer §3.3a).

    Every ``sigma`` is the STATIONARY sd of its OU in dB and every ``tau`` its
    correlation time in seconds; ``block_s`` is the block length the tracks
    were measured on and the model is evaluated on. ``sigma_uj_db`` is the
    per-control-point colour residual of the floor on top of the band-common
    level ``u``; a record that does not carry it has no colour track.

    The per-line track ``v`` may be measured PER ORDER GROUP (record schema
    ``noise-v3-wander/2``): ``sigma_v_db_by_order = {k_edges, sigma_db}`` —
    ``sigma_db[i]`` holds for orders ``k_edges[i] <= k < k_edges[i + 1]`` —
    and optionally ``tau_v_s_by_order = {k_edges, tau_s}`` likewise. Kept as
    ``v_sigma_edges``/``v_sigma_db`` and ``v_tau_edges``/``v_tau_s``;
    :meth:`track_sigma`/:meth:`track_rho` expand them per order (an order past
    the last edge takes the last group). Without them the scalar
    ``sigma_v_db``/``tau_v_s`` holds for every line.

    ``uj_corr_oct`` > 0 correlates the colour tracks ACROSS the control
    points: every block's ``u_j`` vector is ``A y`` with ``y`` independent OU
    tracks of ``(sigma_uj_db, tau_uj_s)`` and ``A`` the Cholesky factor of the
    squared-exponential correlation over the control points' octave ladder of
    length ``uj_corr_oct`` octaves (:meth:`uj_mix`, the rig floor spline's
    own kernel at its own length): the colour may wander only as smoothly as
    the floor it colours. 0 (the default, and every record without the key)
    keeps the tracks independent.
    """

    sigma_d_db: float
    tau_d_s: float
    sigma_v_db: float
    tau_v_s: float
    sigma_u_db: float
    tau_u_s: float
    block_s: float
    sigma_uj_db: float = 0.0
    tau_uj_s: float = 1.0
    v_sigma_edges: tuple[int, ...] = ()
    v_sigma_db: tuple[float, ...] = ()
    v_tau_edges: tuple[int, ...] = ()
    v_tau_s: tuple[float, ...] = ()
    uj_corr_oct: float = 0.0

    #: ``(name, sigma field, tau field)`` of the four latent tracks.
    TRACKS = (
        ("d", "sigma_d_db", "tau_d_s"),
        ("v", "sigma_v_db", "tau_v_s"),
        ("u", "sigma_u_db", "tau_u_s"),
        ("uj", "sigma_uj_db", "tau_uj_s"),
    )

    def __post_init__(self) -> None:
        if not (math.isfinite(self.block_s) and self.block_s > 0.0):
            raise ValueError(f"block_s must be finite and positive, got {self.block_s!r}")
        for name, s_key, t_key in self.TRACKS:
            s, t = float(getattr(self, s_key)), float(getattr(self, t_key))
            if not (math.isfinite(s) and s >= 0.0):
                raise ValueError(f"{s_key} must be finite and >= 0, got {s!r}")
            if s > 0.0 and not (math.isfinite(t) and t > 0.0):
                raise ValueError(f"track {name!r} has sigma {s} but tau {t!r}")
        if self.v_sigma_db and max(self.v_sigma_db) > 0.0 and not self.v_tau_s:
            t = float(self.tau_v_s)
            if not (math.isfinite(t) and t > 0.0):
                raise ValueError(f"sigma_v_db_by_order is positive but tau_v_s is {t!r}")
        for what, edges, vals, positive in (
            ("sigma_v_db_by_order", self.v_sigma_edges, self.v_sigma_db, False),
            ("tau_v_s_by_order", self.v_tau_edges, self.v_tau_s, True),
        ):
            if not edges and not vals:
                continue
            e = np.asarray(edges, dtype=np.int64)
            v = np.asarray(vals, dtype=np.float64)
            if not (
                e.size >= 2
                and v.size == e.size - 1
                and int(e[0]) >= 1
                and bool(np.all(np.diff(e) > 0))
                and bool(np.all(np.isfinite(v)))
                and bool(np.all(v > 0.0) if positive else np.all(v >= 0.0))
            ):
                raise ValueError(
                    f"{what}: need increasing k_edges >= 1 and one "
                    f"{'positive' if positive else 'non-negative'} value per group; "
                    f"got k_edges {list(edges)}, values {list(vals)}"
                )

    @classmethod
    def from_mapping(cls, d: Mapping[str, Any]) -> Wander:
        """From a ``results/noise_v3/wander/<rig>.json`` record (extra keys ignored)."""

        def num(key: str, default: float | None = None) -> float:
            v = d.get(key, default)
            if v is None:
                raise KeyError(f"wander record has no {key!r}")
            return float(v)

        def by_order(key: str, field: str) -> tuple[tuple[int, ...], tuple[float, ...]]:
            blk = d.get(key)
            if not blk:
                return (), ()
            return (
                tuple(int(k) for k in blk["k_edges"]),
                tuple(float(v) for v in blk[field]),
            )

        v_sigma_edges, v_sigma_db = by_order("sigma_v_db_by_order", "sigma_db")
        v_tau_edges, v_tau_s = by_order("tau_v_s_by_order", "tau_s")
        return cls(
            sigma_d_db=num("sigma_d_db"),
            tau_d_s=num("tau_d_s"),
            sigma_v_db=num("sigma_v_db"),
            tau_v_s=num("tau_v_s"),
            sigma_u_db=num("sigma_u_db"),
            tau_u_s=num("tau_u_s"),
            block_s=num("block_s"),
            sigma_uj_db=num("sigma_uj_db", 0.0),
            tau_uj_s=num("tau_uj_s", 1.0),
            v_sigma_edges=v_sigma_edges,
            v_sigma_db=v_sigma_db,
            v_tau_edges=v_tau_edges,
            v_tau_s=v_tau_s,
            uj_corr_oct=num("uj_corr_oct", 0.0),
        )

    def sigma(self, track: str) -> float:
        """The track's scalar (for ``v``: pooled over the orders) sd in dB."""
        return float(getattr(self, dict((n, s) for n, s, _ in self.TRACKS)[track]))

    def rho(self, track: str) -> float:
        """``exp(-block_s / tau)``: the block-to-block correlation of ``track``."""
        tau = float(getattr(self, dict((n, t) for n, _, t in self.TRACKS)[track]))
        return math.exp(-float(self.block_s) / tau)

    def track_sigma(self, track: str, k_max: int) -> float | np.ndarray:
        """The OU sd of ``track`` per TRACK: ``(k_max,)`` for ``v`` measured
        per order (orders ``1 .. k_max``, broadcast over the rotors), else the
        scalar :meth:`sigma`."""
        if track == "v" and self.v_sigma_db:
            return _per_order(self.v_sigma_edges, self.v_sigma_db, k_max)
        return self.sigma(track)

    def track_rho(self, track: str, k_max: int) -> float | np.ndarray:
        """:meth:`rho` per track, as :meth:`track_sigma`."""
        if track == "v" and self.v_tau_s:
            tau = _per_order(self.v_tau_edges, self.v_tau_s, k_max)
            return np.exp(-float(self.block_s) / tau)
        return self.rho(track)

    def active(self, track: str) -> bool:
        """Whether ``track`` exists at all: some line of it has a positive sd."""
        if track == "v" and self.v_sigma_db:
            return max(self.v_sigma_db) > 0.0
        return self.sigma(track) > 0.0

    def as_params(self) -> dict[str, Any]:
        """The ``params.wander`` block of a ``noise-v3-fit/1`` payload."""
        out: dict[str, Any] = dict(
            sigma_d_db=self.sigma_d_db,
            tau_d_s=self.tau_d_s,
            sigma_v_db=self.sigma_v_db,
            tau_v_s=self.tau_v_s,
            sigma_u_db=self.sigma_u_db,
            tau_u_s=self.tau_u_s,
            sigma_uj_db=self.sigma_uj_db,
            tau_uj_s=self.tau_uj_s,
            block_s=self.block_s,
        )
        if self.v_sigma_db:
            out["sigma_v_db_by_order"] = dict(
                k_edges=list(self.v_sigma_edges), sigma_db=list(self.v_sigma_db)
            )
        if self.v_tau_s:
            out["tau_v_s_by_order"] = dict(k_edges=list(self.v_tau_edges), tau_s=list(self.v_tau_s))
        if self.uj_corr_oct > 0.0:
            out["uj_corr_oct"] = self.uj_corr_oct
        return out

    def uj_mix(self, ctrl_hz: Any) -> np.ndarray | None:
        """``(J, J)`` lower-triangular ``A``: a block's colour vector is
        ``A y`` with ``y`` the independent OU draws (``None``: independent
        tracks, ``uj_corr_oct`` 0). The squared-exponential correlation of
        :func:`.floor.se_cholesky` on ``log2(ctrl_hz)``, the rig floor
        spline's construction."""
        if self.uj_corr_oct <= 0.0:
            return None
        oct_ = np.log2(np.asarray(ctrl_hz, dtype=np.float64) / float(np.asarray(ctrl_hz)[0]))
        return se_cholesky(int(oct_.size), float(oct_[1] - oct_[0]), float(self.uj_corr_oct))


def _per_order(edges: tuple[int, ...], values: tuple[float, ...], k_max: int) -> np.ndarray:
    """``(k_max,)``: ``values[i]`` at orders ``edges[i] <= k < edges[i + 1]``,
    the first group below ``edges[0]`` and the last one from its edge up."""
    k = np.arange(1, int(k_max) + 1)
    group = np.clip(np.searchsorted(np.asarray(edges), k, side="right") - 1, 0, len(values) - 1)
    return np.asarray(values, dtype=np.float64)[group]


def ou_blocks(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    n_blocks: int,
    *,
    sigma: float | np.ndarray,
    rho: float | np.ndarray,
) -> np.ndarray:
    """``shape + (n_blocks,)`` STATIONARY OU tracks sampled at the block rate.

    ``x_1 ~ N(0, sigma^2)``, ``x_b = rho x_{b-1} + sqrt(1 - rho^2) sigma e_b`` —
    the exact discretisation of the OU whose prior the fit evaluates
    (``experiments.noise_model.model.ou_log_density``). ``sigma`` and ``rho``
    are scalars or per track (broadcast against ``shape``, e.g. ``(K,)`` per
    order of ``(R, K)``); a track of ``sigma = 0`` is zero, and an all-zero
    ``sigma`` draws nothing.
    """
    out = np.zeros(tuple(shape) + (int(n_blocks),), dtype=np.float64)
    s = np.broadcast_to(np.asarray(sigma, dtype=np.float64), tuple(shape))
    r = np.broadcast_to(np.asarray(rho, dtype=np.float64), tuple(shape))
    if not np.any(s > 0.0) or int(n_blocks) < 1:
        return out
    e = rng.standard_normal(out.shape)
    out[..., 0] = s * e[..., 0]
    innov = np.sqrt(np.maximum(1.0 - r**2, 0.0)) * s
    for b in range(1, int(n_blocks)):
        out[..., b] = r * out[..., b - 1] + innov * e[..., b]
    return out
