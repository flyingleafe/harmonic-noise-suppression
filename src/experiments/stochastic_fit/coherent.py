"""Coherent share of a harmonic order: Bretthorst's general linear model on
the refined carriers.

For order ``k`` and a segment of ``T`` seconds, the model functions are
``cos(k phi_r(t))``, ``sin(k phi_r(t))`` for every rotor ``r`` (``phi_r`` the
integrated refined rate), fitted *jointly* by least squares — the Gram matrix
carries the near-collinearity of four rotors whose lines sit inside one
analysis bin (Bretthorst § 6.3). The explained energy over the band energy
around the lines is the segment's coherent share; the same functions shifted
by half an order (``(k + 1/2) phi``) give the null level of a fit to nothing.
Share versus ``T`` is the tone's decoherence curve: a phase-locked tone stays
at its share, a random-phase line falls to the null as ``T`` grows past its
coherence time.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .data import Clip


def _phase(rps: np.ndarray, sr: int) -> np.ndarray:
    return 2.0 * np.pi * np.cumsum(rps, axis=1) / sr  # (R, n)


def _band(x: np.ndarray, lo: float, hi: float, sr: int) -> np.ndarray:
    lo, hi = max(lo, 5.0), min(hi, sr / 2 - 5.0)
    if hi <= lo:
        return np.zeros_like(x)
    sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def coherent_share(
    clip: Clip,
    orders: tuple[int, ...] = tuple(range(1, 13)),
    seg_s: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
    guard_hz: float = 6.0,
) -> dict[str, np.ndarray]:
    """``share[k, T]`` and ``null[k, T]`` (medians over mics and segments)."""
    sr, x, rps = clip.sr, clip.audio, clip.rps
    M, n = x.shape
    phi = _phase(rps, sr)
    share = np.full((len(orders), len(seg_s)), np.nan)
    null = np.full_like(share, np.nan)
    for ik, k in enumerate(orders):
        for io, T in enumerate(seg_s):
            L = int(T * sr)
            vals, nulls = [], []
            for s in range(0, n - L + 1, L):
                sl = slice(s, s + L)
                r = rps[:, sl]
                if r.min() < 5:
                    continue
                centres = k * r.mean(axis=1)
                lo, hi = centres.min() - guard_hz, centres.max() + guard_hz
                xb = _band(x[:, sl], lo, hi, sr)  # (M, L)
                for kk, store in ((k, vals), (k + 0.5, nulls)):
                    ph = kk * phi[:, sl]  # (R, L)
                    X = np.concatenate([np.cos(ph), np.sin(ph)], axis=0).T  # (L, 2R)
                    G = X.T @ X
                    try:
                        coef = np.linalg.solve(
                            G + 1e-9 * np.trace(G) / G.shape[0] * np.eye(G.shape[0]), X.T @ xb.T
                        )
                    except np.linalg.LinAlgError:
                        continue
                    explained = np.einsum("lm,lm->m", X @ coef, X @ coef)
                    total = np.einsum("ml,ml->m", xb, xb)
                    store.append(np.median(explained / np.maximum(total, 1e-20)))
            if vals:
                share[ik, io] = float(np.median(vals))
                null[ik, io] = float(np.median(nulls)) if nulls else np.nan
    return dict(orders=np.asarray(orders), seg_s=np.asarray(seg_s), share=share, null=null)
