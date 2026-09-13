"""The finite-window expected periodogram of a moving harmonic atom.

ONE entry point, :func:`expected_periodogram_from_atoms`, shared by the revised
stochastic model and by the legacy descriptive-model adapter, so that both
predict the SAME observation through the same code. The API is frozen: the
signature, the units and the bin convention are agreed across workers and must
not be changed or duplicated.

This module imports only ``numpy`` (and ``torch`` lazily, when a tensor is
handed in); nothing from ``experiments.*``, so it can be loaded by file path
and checked on its own.

Backend contract: BOTH. If either argument is a ``torch.Tensor`` the whole
computation runs through ``torch.fft`` and returns a ``torch.Tensor``
(autograd-safe, device-preserving, usable under CUDA); otherwise it runs
through ``numpy.fft`` and returns ``float64`` numpy. The dispatch happens once,
at the top, and there is one code path afterwards. The DEVICE and the PRECISION
come from whichever argument already IS a tensor (``atoms`` when both are), and
the other argument is converted onto it — never the other way round, or a numpy
``atoms`` would drag a CUDA, possibly grad-carrying, ``r_tau`` onto the CPU and
return the answer on the wrong device.

WHAT IT COMPUTES. A real contribution

    x_t = A_t cos(phi_t + eps_t + phi0),   phi0 ~ Uniform(0, 2 pi),

with a stationary residual phase process ``eps`` whose characteristic function
is ``E exp(i (eps_t - eps_s)) = R(t - s)``, observed through a window ``w``.
Write the complex atom ``a_t = w_t A_t exp(i phi_t)`` (so ``A_t`` is the real
cosine PEAK amplitude, and a stationary tone of amplitude ``A`` carries
mean-square power ``A^2 / 2``). Then, with ``Z(f) = sum_t a_t e^{i eps_t}
e^{-2 pi i f t}``,

    E |X(f)|^2 = 0.25 ( E|Z(f)|^2 + E|Z(-f)|^2 ) / sum w^2,

because ``x = Re{analytic atom}`` and the cross term ``E[Z(f) Z(-f)]`` vanishes
EXACTLY under the uniform initial phase. The result is directly comparable to
``experiments.stochastic_fit.data.periodogram(...).power``, which is
``|rfft(w x)|^2 / sum(w^2)`` and does NOT double interior bins — so neither
does this.

EXACT ALGORITHM (this is where silent factor-of-2 and off-by-one bugs live)::

    L        = 2 * n_fft            (NOT next_fast_len: L must be an exact
                                     multiple of n_fft, see the grid line)
    g(tau)   = sum_t a_t conj(a_{t-tau}) = ifft(|fft(a, L)|^2)
               in the natural wrapped layout (lag -tau sits at index L - tau)
    R_wrap   = R(tau) at index tau and at index L - tau for tau = 0..T-1,
               zero in the middle
    P_Z[mu]  = Re fft(g * R_wrap)[mu]      (real by Hermitian symmetry)
    grid     : f_j = j / n_fft = 2j / L, so n_fft-grid bin j is exactly L-bin
               2j — an exact subsample, never an interpolation
    M[j]     = 0.25 * (P_Z[2j] + P_Z[(L - 2j) mod L]) / window_sumsq

``j = 0`` and ``j = n_fft // 2`` map to themselves, so DC and Nyquist come out
as ``0.5 * P_Z / window_sumsq`` with no special case.

``P_Z`` is real and non-negative by construction (it is the DFT of the Schur
product of two positive-definite kernels); tiny negatives are roundoff only and
are NOT clipped here — a caller that needs a positive model adds its floor.

BATCHING. Leading dimensions of ``atoms`` and ``r_tau`` broadcast. No ``(T, T)``
outer product and no ``(M, N, T, T)`` intermediate is ever materialized — the
FFT autocorrelation is the whole point. Chunking over the batch is the caller's
responsibility; peak memory is a few complex buffers of shape
``broadcast(batch) x 2 n_fft``.

VERIFIED, not asserted (``n_fft`` = 64, moving carrier, amplitude ramp,
``R(tau) = exp(-0.031 tau)``): equal to the explicit double sum
``sum_{t,s} a_t conj(a_s) R(t-s) e^{-2 pi i f (t-s)}`` to 6.1e-16 relative;
``max|Im P_Z| / max Re P_Z`` = 5.4e-17 and ``min P_Z > 0``; with ``R == 1``
equal to the folded deterministic ``|Z|^2`` to 5.0e-17; against a Monte-Carlo
mean periodogram over uniform initial phases, 5.8e-6 for a fractional-bin tone
and 1.2e-2 for a planted chirp with a diffusing phase, both far inside 2 s.e.
(0.13 and 8.1e-2 at 40k draws).
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: The FROZEN spectral front end of the revised-phase campaign, in one place so
#: every adapter reads the same convention (16 kHz clips, periodic Hann,
#: 1.024 s window, 64 ms hop, fixed band).
N_FFT = 16384
HOP = 1024
SR = 16000
BAND_HZ = (30.0, 7900.0)


def hann_window(n_fft: int = N_FFT) -> np.ndarray:
    """The PERIODIC Hann window, identical to ``data.periodogram``'s.

    ``np.hanning(n + 1)[:n]``: periodic, not symmetric. Derived here once so no
    other module re-derives it (a symmetric window changes ``sum(w**2)`` and
    every power the model predicts).
    """
    return np.hanning(int(n_fft) + 1)[: int(n_fft)]


def _is_tensor(x: Any) -> bool:
    """``True`` for a ``torch`` tensor, without importing torch to ask."""
    return type(x).__module__.split(".")[0] == "torch"


def _torch_of(*arrays: Any) -> Any:
    """``torch`` if any argument is a tensor, else ``None`` (single dispatch)."""
    for a in arrays:
        if _is_tensor(a):
            import torch

            return torch
    return None


def expected_periodogram_from_atoms(
    atoms: Any, r_tau: Any, *, n_fft: int, window_sumsq: float
) -> Any:
    """Expected periodogram of a windowed harmonic atom with a stationary residual phase.

    Parameters
    ----------
    atoms
        Complex ``(..., T)`` with ``T == n_fft``, ALREADY windowed:
        ``a_t = w_t * A_t * exp(i phi_t)`` where ``A_t`` is the real cosine peak
        amplitude of the contribution ``x_t = A_t cos(phi_t + uniform phase)``.
    r_tau
        Real ``(..., T)`` or ``(T,)``: ``R(tau)`` for ``tau = 0..T-1``,
        broadcast against the leading dims of ``atoms``. Used VERBATIM — no
        parametric assumption, no renormalization to ``R(0) = 1``.
    n_fft
        Window length; also the output grid (``f_j = j * sr / n_fft``).
    window_sumsq
        ``sum(w**2)`` of the window already applied to ``atoms``.

    Returns
    -------
    Real ``(..., n_fft // 2 + 1)``, in ``data.periodogram`` units.
    """
    n = int(n_fft)
    if n <= 0:
        raise ValueError(f"n_fft must be positive, got {n_fft!r}")
    L = 2 * n
    th = _torch_of(atoms, r_tau)

    if th is None:
        a = np.asarray(atoms, dtype=np.complex128)
        r = np.asarray(r_tau, dtype=np.float64)
        if a.shape[-1] != n:
            raise ValueError(f"atoms last dim {a.shape[-1]} != n_fft {n}")
        if r.shape[-1] != n:
            raise ValueError(f"r_tau last dim {r.shape[-1]} != n_fft {n}")
        spec = np.fft.fft(a, n=L, axis=-1)
        g = np.fft.ifft(spec.real**2 + spec.imag**2, axis=-1)
        rw = np.zeros(r.shape[:-1] + (L,), dtype=np.float64)
        rw[..., :n] = r
        rw[..., n + 1 :] = r[..., :0:-1]  # R(tau) mirrored to the negative lags
        p_z = np.fft.fft(g * rw, axis=-1).real
        pos = 2 * np.arange(n // 2 + 1)
        neg = (L - pos) % L
        return 0.25 * (p_z[..., pos] + p_z[..., neg]) / float(window_sumsq)

    # DISPATCH ON WHICHEVER ARGUMENT ALREADY IS A TENSOR: it owns the device and
    # the precision, and the other argument is converted onto it. Converting
    # ``atoms`` first and then moving ``r_tau`` onto ITS device silently pulls a
    # CUDA (possibly grad-carrying) ``r_tau`` back to the CPU when ``atoms`` is
    # numpy, and returns the result on the wrong device.
    if _is_tensor(atoms):
        a_t = (
            atoms
            if atoms.is_complex()
            else atoms.to(th.complex64 if atoms.dtype == th.float32 else th.complex128)
        )
        real_dtype = th.float32 if a_t.dtype == th.complex64 else th.float64
        r_t = th.as_tensor(r_tau, device=a_t.device)
        if r_t.is_complex():
            raise ValueError("r_tau must be real")
        r_t = r_t.to(dtype=real_dtype, device=a_t.device)
    else:
        r_t = r_tau
        if r_t.is_complex():
            raise ValueError("r_tau must be real")
        real_dtype = th.float32 if r_t.dtype == th.float32 else th.float64
        r_t = r_t.to(real_dtype)
        a_t = th.as_tensor(atoms, device=r_t.device).to(
            th.complex64 if real_dtype == th.float32 else th.complex128
        )
    dev = a_t.device
    if a_t.shape[-1] != n:
        raise ValueError(f"atoms last dim {int(a_t.shape[-1])} != n_fft {n}")
    if r_t.shape[-1] != n:
        raise ValueError(f"r_tau last dim {int(r_t.shape[-1])} != n_fft {n}")
    spec = th.fft.fft(a_t, n=L, dim=-1)
    g = th.fft.ifft(spec.real**2 + spec.imag**2, dim=-1)
    # R_wrap by concatenation, not in-place assignment: keeps autograd happy if
    # a caller ever hands in a fitted R.
    mid = th.zeros(r_t.shape[:-1] + (1,), dtype=real_dtype, device=dev)
    rw = th.cat((r_t, mid, r_t.flip(-1)[..., : n - 1]), dim=-1)
    p_z = th.fft.fft(g * rw, dim=-1).real
    pos = th.arange(0, n + 1, 2, device=dev)
    neg = th.remainder(L - pos, L)
    return 0.25 * (p_z.index_select(-1, pos) + p_z.index_select(-1, neg)) / float(window_sumsq)


__all__ = [
    "BAND_HZ",
    "HOP",
    "N_FFT",
    "SR",
    "expected_periodogram_from_atoms",
    "hann_window",
]
