"""Plain complex demodulation: ``demodulate(x, carrier, band)``.

Multiply real audio by ``exp(-i * phase)`` of a carrier and keep only
``(-band, +band)`` Hz of the result. Numpy + scipy, one function, no
decimation, no stride bookkeeping — the tool for looking at one harmonic
of one rotor by hand. The heavy batched version (comb recursion, brickwall
band select + decimation, torch) is :func:`tracking.dsp.demod`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal


def carrier_phase(freq_hz: np.ndarray, sr: float, *, order: float = 1.0) -> np.ndarray:
    """Instantaneous phase ``2 pi k * integral f(t) dt`` (rad) of a frequency track on the audio grid.

    ``freq_hz`` is ``(T,)`` at the audio rate ``sr``; trapezoid integration.
    A rotor-speed track in rev/s is a frequency in Hz, so pass it directly and
    put the harmonic index in ``order``.
    """
    f = np.asarray(freq_hz, dtype=np.float64)
    integral = (np.cumsum(f) - 0.5 * (f + f[0])) / float(sr)
    return 2.0 * np.pi * float(order) * integral


def demodulate(
    x: np.ndarray,
    carrier: np.ndarray,
    band: float,
    sr: float,
    *,
    order: float = 1.0,
    filt_order: int = 6,
) -> np.ndarray:
    """Baseband of ``x`` around ``carrier``, band-limited to ``(-band, +band)`` Hz.

    ``x``: ``(T,)`` or ``(C, T)`` real audio at ``sr``.
    ``carrier``: a frequency track ``(T,)`` in Hz on the audio grid — a
    rotor-speed track in rev/s, with the harmonic index in ``order``. Build
    the phase with :func:`carrier_phase` if you need it separately.
    ``band``: one-sided bandwidth in Hz; zero-phase Butterworth low-pass
    (``sosfiltfilt``) on the complex baseband, so the result stays on the
    audio grid with no delay.

    Returns complex128 of the same shape as ``x``. Its magnitude is the
    harmonic's envelope and its unwrapped angle the phase residual relative
    to the carrier (the label error times ``order`` plus the rotor's own jitter).
    """
    x = np.asarray(x, dtype=np.float64)
    phase = carrier_phase(carrier, sr, order=order)
    if phase.shape[-1] != x.shape[-1]:
        raise ValueError(f"carrier length {phase.shape[-1]} != audio length {x.shape[-1]}")
    if not 0.0 < band < 0.5 * sr:
        raise ValueError(f"band={band} Hz must lie in (0, sr/2)")
    z = np.asarray(x * np.exp(-1j * phase), dtype=np.complex128)
    sos = signal.butter(filt_order, band, btype="low", fs=float(sr), output="sos")
    return signal.sosfiltfilt(sos, np.real(z), axis=-1) + 1j * signal.sosfiltfilt(
        sos, np.imag(z), axis=-1
    )


def residual_frequency(z: np.ndarray, sr: float, *, smooth_s: float = 0.05) -> np.ndarray:
    """Instantaneous frequency offset (Hz) of a demodulated baseband, from its unwrapped phase.

    Positive means the harmonic sits ABOVE the carrier. ``smooth_s`` is the
    moving-average length applied to the phase derivative (0 disables it).
    Divide by the harmonic order to get the rotor-rate error in rev/s.
    """
    z = np.asarray(z)
    dphi = np.diff(np.unwrap(np.angle(z), axis=-1), axis=-1) * float(sr) / (2.0 * np.pi)
    if smooth_s > 0:
        n = max(1, int(round(smooth_s * sr)))
        kern = np.ones(n) / n
        dphi = np.apply_along_axis(lambda v: np.convolve(v, kern, mode="same"), -1, dphi)
    return np.concatenate([dphi[..., :1], dphi], axis=-1)


def zoom_spectrogram(
    z: np.ndarray,
    sr: float,
    band: float,
    *,
    win_s: float = 1.0,
    hop_s: float | None = None,
    n_freq: int = 256,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STFT of a demodulated baseband on the frequency grid ``(-band, +band)`` Hz.

    ``z``: complex baseband from :func:`demodulate` (``(T,)`` or ``(C, T)``)
    at the audio rate ``sr``. The signal is first decimated to ``fs_z =
    2 * band * 1.25`` (the low-pass already limits it to ``+-band``, so
    nothing is lost), then a two-sided STFT with a ``win_s``-second
    window and ``hop_s`` hop (default ``win_s / 4``) is zero-padded so the
    grid has ``n_freq`` bins across ``(-band, +band)``.

    Returns ``(t_s, f_hz, S)`` with ``S`` complex ``(..., n_freq, n_frames)``;
    ``f_hz`` ascending from ``-band`` to ``+band`` (negative = below the
    carrier). ``|S|**2`` is the zoomed power spectrogram; ``np.angle(S)``
    the phase per bin. Bin spacing is ``2 * band / n_freq``; resolution is
    still set by ``win_s`` (periodic Hann: ``1.44 / win_s`` Hz half-power).
    """
    z = np.asarray(z, dtype=np.complex128)
    fs_z = 2.5 * float(band)
    q = max(1, int(np.floor(float(sr) / fs_z)))
    fs_z = float(sr) / q
    zd = z[..., ::q]
    nper = max(8, int(round(win_s * fs_z)))
    hop = max(1, int(round((hop_s if hop_s is not None else win_s / 4) * fs_z)))
    nfft = max(nper, int(np.ceil(n_freq * fs_z / (2.0 * band))))
    # the scipy stubs type `boundary` as str; None (no edge padding) is the documented runtime form
    stft_kw: dict[str, Any] = dict(boundary=None, padded=False)
    f, t, S = signal.stft(
        zd,
        fs=fs_z,
        window=window,
        nperseg=nper,
        noverlap=nper - hop,
        nfft=nfft,
        return_onesided=False,
        **stft_kw,
    )
    order = np.argsort(f)
    f, S = f[order], S[..., order, :]
    keep = np.abs(f) <= band
    return np.asarray(t), np.asarray(f[keep]), np.asarray(S[..., keep, :])
