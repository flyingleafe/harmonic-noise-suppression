"""The render chain's anti-alias low-pass and its decimator.

MOVED verbatim from :mod:`experiments.stochastic_fit.stage2` (the anti-alias
filter and its specification) and :mod:`experiments.stochastic_fit.clips` (the
``resample_poly`` decimation, split out of :func:`clips.decimate` so a caller
with plain audio does not need a :class:`~experiments.stochastic_fit.data.Clip`);
both import them back.

This pair IS the observation chain the v2 fit's ``render_transfer_power``
describes, so the renderer must keep going through exactly these two filters:
the fitted ``M`` carries their transfer. One definition, here, because
``data_processing`` may not import ``experiments``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import gcd

import numpy as np

__all__ = [
    "AA_PASS_HZ",
    "AA_RIPPLE_DB",
    "AA_STOP_DB",
    "AA_STOP_HZ",
    "AntialiasFilter",
    "antialias",
    "antialias_filter",
    "decimate_audio",
]


#: The render's own anti-alias low-pass, specified rather than inherited.
#: Passband edge (the fit's own band edge, F_MAX), stopband edge (the output
#: Nyquist), stopband attenuation and the passband ripple the fit band must
#: keep; the filter length follows from them through ``kaiserord``, so the
#: transition is as narrow as the specification requires instead of as narrow
#: as a fixed length allows.
AA_PASS_HZ, AA_STOP_HZ, AA_STOP_DB, AA_RIPPLE_DB = 7900.0, 8000.0, 100.0, 0.01


def _ripple_attenuation_db(ripple_db: float) -> float:
    """Stopband attenuation a Kaiser design needs for a passband ripple.

    A Kaiser window's deviation ``delta`` is the SAME in both bands, so a
    passband tolerance of ``ripple_db`` (peak-to-nominal, dB) is the
    attenuation ``-20 log10(10**(ripple_db/20) - 1)``. Stating it separately
    is what makes the passband a specification rather than a by-product of the
    stopband number.
    """
    delta = 10.0 ** (float(ripple_db) / 20.0) - 1.0
    if delta <= 0.0:
        raise ValueError(f"ripple_db must be positive, got {ripple_db}")
    return float(-20.0 * np.log10(delta))


@dataclass(frozen=True)
class AntialiasFilter:
    """The render's low-pass, as a designed object rather than a side effect."""

    taps: np.ndarray  # (L,) odd length, symmetric: type-I linear phase
    sample_rate: float
    pass_hz: float
    stop_hz: float
    stop_db: float
    ripple_db: float
    beta: float

    @property
    def delay(self) -> int:
        """Group delay in samples — exactly the centre tap of a type-I FIR."""
        return (self.taps.size - 1) // 2

    def spec(self) -> dict[str, float]:
        return dict(
            sample_rate=float(self.sample_rate),
            pass_hz=float(self.pass_hz),
            stop_hz=float(self.stop_hz),
            stop_db=float(self.stop_db),
            ripple_db=float(self.ripple_db),
            beta=float(self.beta),
            n_taps=int(self.taps.size),
            delay=int(self.delay),
        )


@lru_cache(maxsize=8)
def antialias_filter(
    sample_rate: float,
    pass_hz: float = AA_PASS_HZ,
    stop_hz: float = AA_STOP_HZ,
    stop_db: float = AA_STOP_DB,
    ripple_db: float = AA_RIPPLE_DB,
) -> AntialiasFilter:
    """Design the render low-pass from its specification (cached per spec).

    ``kaiserord`` sizes the filter from the transition width and the stricter
    of the two band specifications, so both the stopband attenuation AND the
    passband ripple are honoured by construction.
    """
    from scipy.signal import firwin, kaiserord

    nyquist = float(sample_rate) / 2.0
    width = (float(stop_hz) - float(pass_hz)) / nyquist
    if not 0.0 < float(pass_hz) < float(stop_hz) <= nyquist or not 0.0 < width < 1.0:
        raise ValueError(
            f"transition [{pass_hz}, {stop_hz}] Hz is not inside the Nyquist band "
            f"of {sample_rate} Hz (Nyquist {nyquist} Hz)"
        )
    attenuation = max(float(stop_db), _ripple_attenuation_db(ripple_db))
    n_taps, beta = kaiserord(attenuation, width)
    n_taps = int(n_taps) | 1  # firwin wants an odd length for a type-I linear phase
    taps = firwin(n_taps, (pass_hz + stop_hz) / 2.0, window=("kaiser", beta), fs=sample_rate)
    return AntialiasFilter(
        taps=np.asarray(taps, dtype=np.float64),
        sample_rate=float(sample_rate),
        pass_hz=float(pass_hz),
        stop_hz=float(stop_hz),
        stop_db=float(stop_db),
        ripple_db=float(ripple_db),
        beta=float(beta),
    )


def antialias(
    x: np.ndarray,
    sample_rate: float,
    *,
    pass_hz: float = AA_PASS_HZ,
    stop_hz: float = AA_STOP_HZ,
    stop_db: float = AA_STOP_DB,
    ripple_db: float = AA_RIPPLE_DB,
) -> np.ndarray:
    """Low-pass ``x`` so that nothing above ``stop_hz`` survives decimation.

    A rendered comb carries lines above the OUTPUT Nyquist; the shared
    decimator's own filter is a fixed-length Kaiser whose stop band is about
    40-50 dB, and raising its beta widens the transition instead of deepening
    it at constant length. So the render removes its own out-of-band energy
    first, with a filter whose passband, stopband, attenuation and ripple are
    stated (:func:`antialias_filter`).

    ONE application, by overlap-add FFT convolution, centred on the type-I
    filter's own group delay. Three consequences, each a property the previous
    ``filtfilt`` route did not have:

    * the realized response is ``H``, not ``|H|^2``, so the designed 0.01 dB
      passband ripple and 100 dB stop band are the ones the render gets
      instead of their squares;
    * the cost is one FFT convolution, not two direct passes over 2827 taps;
    * the edges are a deterministic zero-extension (a linear convolution
      centred on the delay), defined for EVERY input length. ``filtfilt``
      refuses any input shorter than ``3 * (n_taps - 1)`` — 8478 samples,
      0.19 s at 44.1 kHz — and its odd extension makes short-input behaviour a
      function of the length.

    Zero phase comes from the symmetric odd-length kernel: the centred slice is
    aligned sample-for-sample with the input, so a rendered clip's rotor track
    still describes its own audio.
    """
    from scipy.signal import oaconvolve

    filt = antialias_filter(
        float(sample_rate), float(pass_hz), float(stop_hz), float(stop_db), float(ripple_db)
    )
    a = np.asarray(x, dtype=np.float64)
    n = a.shape[-1]
    kernel = filt.taps.reshape((1,) * (a.ndim - 1) + (filt.taps.size,))
    full = oaconvolve(a, kernel, mode="full", axes=-1)
    return np.ascontiguousarray(full[..., filt.delay : filt.delay + n])


def decimate_audio(audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Resample ``(..., T)`` audio to ``sr_to`` with a polyphase anti-alias filter.

    The audio half of :func:`experiments.stochastic_fit.clips.decimate`, which
    calls it: ``scipy.signal.resample_poly``'s Kaiser-windowed FIR, float64 in,
    float32 out, and an identity at equal rates (the rate the caller asked for
    is already the rate it has, so nothing is filtered and nothing is cast).

    This is the path REAL clips take, and it is deliberately unchanged: a
    render that carries energy above the output Nyquist must remove it itself
    (:func:`antialias`) before calling this, not move the band edge for every
    recording in the project.
    """
    if int(sr_from) == int(sr_to):
        return np.asarray(audio)
    import scipy.signal as sps

    g = gcd(int(sr_from), int(sr_to))
    up, down = int(sr_to) // g, int(sr_from) // g
    out = sps.resample_poly(np.asarray(audio, dtype=np.float64), up, down, axis=-1)
    return np.ascontiguousarray(out.astype(np.float32))
