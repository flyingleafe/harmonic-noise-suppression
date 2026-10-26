"""A rotor-permutation-invariant spectral distance between v4 noise rigs.

Decided with Dmitrii, 2026-10-26 (``docs/experiments/rp-synthetic.md`` §
"Prior coverage map"). A rig is one ``noise-v4-fit/3`` payload or a
``standby``/``cruise`` pair. It is read at CONSTANT speeds: every rotor at each
reference speed of :data:`SPEEDS` (10-150 rev/s), the expected periodogram on a
:data:`N_FFT`-point Hann window at :data:`SR` (3.9 Hz bins), mic-mean, the bins
inside :data:`BAND_HZ`, no wind. The periodogram splits into each rotor's lines
(:func:`~data_processing.noise_model.v4.rotor_line_periodograms`) and ONE
broadband part (:func:`~data_processing.noise_model.v4.broadband_periodogram`);
a pair is power-blended per speed with the renderer's regime weight.

* HARMONIC feature of rotor ``r``: ``10 log10(1 + L_r / B)`` per bin and speed,
  the rotor's lines over the broadband; 0 dB where no line shows, so a buried
  or missing order needs no clip.
* BROADBAND feature: ``10 log10 B`` with the rig's mean removed (gain-free).

Between two rigs, the HARMONIC distance is the PIT distance of their rotors: the
RMS over (speed, bin) of two rotors' feature difference, averaged over the best
of the ``R!`` rotor matchings (one matching for every speed). Relabelling the
same rotors therefore costs 0. The BROADBAND distance is the RMS difference of the
broadband features. The TOTAL is their sum, all in dB. The PIT distance is a
metric but not Euclidean, so a 2-D map of it is METRIC MDS (SMACOF from the
classical solution) and reports Kruskal's stress-1.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Any

import numpy as np

from data_processing.noise_model import v4 as V4
from data_processing.noise_model.render import regime_blend_weight

#: Reference speeds (rev/s): the trajectory model's span up to its 150 rev/s cap.
SPEEDS = np.arange(10.0, 151.0, 10.0)
N_FFT = 4096
SR = 16000
BAND_HZ = (30.0, 7900.0)
_F = np.fft.rfftfreq(N_FFT, 1.0 / SR)
#: The periodogram bins the distance reads.
KEEP = (BAND_HZ[0] <= _F) & (BAND_HZ[1] >= _F)


def regimes_of(rig: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """``{"single": payload}`` or ``{"standby": ..., "cruise": ...}`` from a
    payload, a ``{"single": payload}``, a regime pair, or a ``noise_v2`` bank
    entry (``standby`` None)."""
    if "params" in rig:
        return {"single": rig}
    if set(rig) == {"single"}:
        return {"single": rig["single"]}
    regs = {k: rig[k] for k in ("standby", "cruise") if rig.get(k) is not None}
    if set(regs) == {"cruise"}:
        return {"single": regs["cruise"]}
    if set(regs) != {"standby", "cruise"}:
        raise ValueError(f"expected a payload or a standby/cruise pair, got keys {sorted(rig)}")
    return regs


def features(rig: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """``(harmonic (R, S, F), broadband (S, F))`` float32 features of one rig
    (module docstring): ``S = len(SPEEDS)``, ``F = KEEP.sum()``."""
    pars = {k: V4.V4Params.from_payload(v) for k, v in regimes_of(rig).items()}
    n_rot = next(iter(pars.values())).n_rotors
    harm = np.empty((n_rot, SPEEDS.size, int(KEEP.sum())), dtype=np.float32)
    broad = np.empty((SPEEDS.size, int(KEEP.sum())), dtype=np.float64)
    for i, s in enumerate(SPEEDS):
        rps = np.full(n_rot, float(s))
        if "single" in pars:
            weights = {"single": 1.0}
        else:
            w = float(regime_blend_weight(np.repeat(rps[:, None], 1600, 1), sr=SR).mean())
            weights = {"standby": 1.0 - w, "cruise": w}
        lines = np.zeros((n_rot, N_FFT // 2 + 1))
        floor = np.zeros(N_FFT // 2 + 1)
        for name, wt in weights.items():
            if wt <= 0.0:
                continue
            par = pars[name]
            lines += wt * V4.rotor_line_periodograms(par, rps, n_fft=N_FFT, sr=SR).mean(axis=1)
            floor += wt * V4.broadband_periodogram(par, rps, n_fft=N_FFT, sr=SR, wind=False).mean(
                axis=0
            )
        lines, floor = lines[:, KEEP], floor[KEEP]
        harm[:, i] = 10.0 * np.log10(1.0 + lines / floor[None, :])
        broad[i] = 10.0 * np.log10(floor)
    return harm, (broad - broad.mean()).astype(np.float32)


def _rms_distances(x: np.ndarray) -> np.ndarray:
    """``(n, n)`` RMS distance between the rows of ``x (n, d)``, by the Gram
    matrix of the centred rows (centring removes most of float32's cancellation)."""
    x = x - x.mean(axis=0, keepdims=True)
    nrm = np.einsum("ij,ij->i", x, x)
    d2 = nrm[:, None] + nrm[None, :] - 2.0 * (x @ x.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2 / x.shape[1])


def pairwise(harm: np.ndarray, broad: np.ndarray) -> dict[str, np.ndarray]:
    """``{"harmonic", "broadband", "total"}`` ``(n, n)`` distances (dB) between
    ``n`` rigs, from stacked :func:`features`: ``harm (n, R, S, F)``,
    ``broad (n, S, F)``."""
    n, n_rot = harm.shape[:2]
    rotor = _rms_distances(harm.reshape(n * n_rot, -1)).reshape(n, n_rot, n, n_rot)
    dh = np.full((n, n), np.inf, dtype=np.float32)
    for perm in itertools.permutations(range(n_rot)):
        acc = sum(rotor[:, r, :, p] for r, p in enumerate(perm))
        np.minimum(dh, np.asarray(acc) / n_rot, out=dh)
    db = _rms_distances(broad.reshape(n, -1)).astype(np.float32)
    for d in (dh, db):
        np.fill_diagonal(d, 0.0)
        d[:] = 0.5 * (d + d.T)
    return {"harmonic": dh, "broadband": db, "total": dh + db}


def classical_mds(d: np.ndarray, k: int = 2) -> np.ndarray:
    """``(n, k)`` Torgerson embedding of the distance matrix ``d``."""
    d2 = np.asarray(d, dtype=np.float64) ** 2
    b = -0.5 * (d2 - d2.mean(axis=0) - d2.mean(axis=1)[:, None] + d2.mean())
    w, v = np.linalg.eigh(b)
    top = np.argsort(w)[::-1][:k]
    return v[:, top] * np.sqrt(np.maximum(w[top], 0.0))


def stress1(d: np.ndarray, xy: np.ndarray) -> float:
    """Kruskal's stress-1 of the embedding ``xy`` of ``d``."""
    e = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(len(d), 1)
    return float(np.sqrt(((e[iu] - d[iu]) ** 2).sum() / (np.asarray(d[iu], np.float64) ** 2).sum()))


def metric_mds(
    d: np.ndarray, *, max_iter: int = 1000, tol: float = 1e-7
) -> tuple[np.ndarray, float]:
    """``(xy (n, 2), stress-1)``: SMACOF (Guttman transform, unit weights) from
    the classical solution, until the raw stress improves by less than ``tol``
    of itself."""
    d64 = np.asarray(d, dtype=np.float64)
    n = len(d64)
    xy = classical_mds(d64)
    prev = np.inf
    diag = np.diag_indices(n)
    for _ in range(max_iter):
        e = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
        raw = float(((e - d64) ** 2).sum())
        if prev - raw < tol * prev:
            break
        prev = raw
        b = -np.divide(d64, e, out=np.zeros_like(d64), where=e > 0.0)
        b[diag] = 0.0
        b[diag] = -b.sum(axis=1)
        xy = b @ xy / n
    return xy, stress1(d64, xy)


__all__ = [
    "BAND_HZ",
    "KEEP",
    "N_FFT",
    "SPEEDS",
    "SR",
    "classical_mds",
    "features",
    "metric_mds",
    "pairwise",
    "regimes_of",
    "stress1",
]
