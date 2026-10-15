"""Time-varying rate warp of a native-rate noise chunk and its rotor labels.

A rotating-noise recording with rotor-speed trajectory ``r(t)`` is replayed at
a time-varying rate ``alpha(t)``. That scales every frequency by ``alpha`` and
turns the recording into a valid recording of the same rig with the rotor
trajectory

    r_tilde(t) = alpha(t) * r(tau(t)),     tau(t) = integral_0^t alpha(u) du,

where ``tau`` is the source-time map (monotone since ``alpha > 0``). At
constant ``alpha`` this is speed perturbation; the room's RIR is the one part
that does not scale like a real rotor at ``alpha * r`` would. Parametrisation
(``docs/experiments/paper-real-protocol.md`` § "Augmentations"):

    alpha(t) = alpha0 * (c + a * sin(2*pi*f*t + phi))

with ``alpha0 ~ U(alpha_low, alpha_high)``, ``c ~ U(1 - dev_const, 1 + dev_const)``,
``a ~ U(0, dev_sine)``, ``f ~ U(f_low, f_high)`` Hz, ``phi ~ U(0, 2*pi)``, drawn
in that order from the per-sample RNG. ``tau`` has the closed form

    tau(t) = alpha0 * (c*t + (a / (2*pi*f)) * (cos(phi) - cos(2*pi*f*t + phi))).

Signal path: ONE resampling of the native-rate window (44.1 kHz for DREGON and
MD2) at the positions ``tau(t)`` — a Kaiser-windowed sinc interpolator (32
taps, images above ~0.57 fs suppressed by ~70 dB) — then the decimation to the
model rate with soxr HQ, last. A native rate ``fs`` covers any local
``alpha >= sample_rate / fs`` (0.36 at 44.1 -> 16 kHz), so ``alpha < 1`` never
leaves a silent band below the model's Nyquist. The warped native signal spans
``[-EDGE_S, T + EDGE_S]`` and the decimated output is cropped to ``[0, T]``, so
the decimator's edge transient never reaches the chunk.

Output convention (shared with the un-warped native path, ``params=None``):
a fresh Frame, timeline at 0, with ``audio`` exactly ``target_len`` samples at
the model rate and a clean ``rps`` label track on a uniform
``LABEL_RATE_HZ`` grid covering ``[0, T]`` (``online_mixing.
interpolate_rps_to_stft_grid`` re-grids it to the STFT frames). A window too
short for the drawn warp raises — the stream window is sized for the worst
case by :func:`source_window_s`, so this never pads.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import soxr
import tdseries as td
from numba import njit

from data_processing.mixing import resolve_motor_tracks
from data_processing.sources.dregon import clean_command_spikes

#: Uniform grid rate (Hz) of the emitted ``rps`` label track.
LABEL_RATE_HZ = 100.0

#: Pre/post-roll (seconds) warped beyond the chunk on each side and cropped
#: after decimation: soxr's HQ edge transient is ~4.5 ms at 44.1 -> 16 kHz.
EDGE_S = 0.02

#: Extra source seconds on top of the analytic worst case (interpolator taps,
#: rounding of the window cut).
SOURCE_MARGIN_S = 0.05

#: The parameters a ``noise_rate_warp`` block must set (besides ``probability``).
PARAM_KEYS = ("alpha_low", "alpha_high", "dev_const", "dev_sine", "f_low", "f_high")

# Kaiser-windowed sinc interpolation kernel, cutoff at the source Nyquist:
# half-width KERNEL_HALF taps, tabulated at KERNEL_PHASES points per sample.
KERNEL_HALF = 16
KERNEL_PHASES = 512
KERNEL_BETA = 7.0
_KERNEL_X = np.arange(2 * KERNEL_HALF * KERNEL_PHASES + 1) / KERNEL_PHASES - KERNEL_HALF
_KERNEL_TABLE = np.sinc(_KERNEL_X) * np.kaiser(_KERNEL_X.size, KERNEL_BETA)


@dataclass(frozen=True)
class RateWarpParams:
    """One drawn ``alpha(t) = alpha0 * (c + a*sin(2*pi*f*t + phi))`` warp."""

    alpha0: float
    c: float
    a: float
    f: float
    phi: float

    def alpha(self, t: np.ndarray) -> np.ndarray:
        """Playback rate at output time ``t`` (seconds)."""
        return self.alpha0 * (self.c + self.a * np.sin(2.0 * np.pi * self.f * t + self.phi))

    def tau(self, t: np.ndarray) -> np.ndarray:
        """Source time ``tau(t) = integral_0^t alpha`` (seconds) at output time ``t``."""
        w = 2.0 * np.pi * self.f
        return self.alpha0 * (
            self.c * t + (self.a / w) * (np.cos(self.phi) - np.cos(w * t + self.phi))
        )

    @property
    def peak_rate(self) -> float:
        """Upper bound of ``alpha(t)`` over all ``t``."""
        return self.alpha0 * (self.c + self.a)


def _params(spec: Mapping[str, Any]) -> dict[str, float]:
    missing = [k for k in PARAM_KEYS if k not in spec]
    if missing:
        raise ValueError(f"noise_rate_warp is missing {missing} (required: {list(PARAM_KEYS)})")
    p = {k: float(spec[k]) for k in PARAM_KEYS}
    if not 0.0 < p["alpha_low"] <= p["alpha_high"]:
        raise ValueError(f"noise_rate_warp needs 0 < alpha_low <= alpha_high, got {p}")
    if p["dev_const"] < 0.0 or p["dev_sine"] < 0.0 or p["dev_const"] + p["dev_sine"] >= 1.0:
        raise ValueError(f"noise_rate_warp needs dev_const, dev_sine >= 0 and sum < 1, got {p}")
    if not 0.0 < p["f_low"] <= p["f_high"]:
        raise ValueError(f"noise_rate_warp needs 0 < f_low <= f_high, got {p}")
    return p


def sample_rate_warp_params(spec: Mapping[str, Any], rng: np.random.Generator) -> RateWarpParams:
    """Draw ``(alpha0, c, a, f, phi)`` — in that fixed order — from ``rng``.

    ``spec`` holds resolved floats (ramps materialised by the caller). The
    single fire-decision draw is the caller's.
    """
    p = _params(spec)
    alpha0 = float(rng.uniform(p["alpha_low"], p["alpha_high"]))
    c = float(rng.uniform(1.0 - p["dev_const"], 1.0 + p["dev_const"]))
    a = float(rng.uniform(0.0, p["dev_sine"]))
    f = float(rng.uniform(p["f_low"], p["f_high"]))
    phi = float(rng.uniform(0.0, 2.0 * np.pi))
    return RateWarpParams(alpha0=alpha0, c=c, a=a, f=f, phi=phi)


def worst_case_rate(spec: Mapping[str, Any]) -> float:
    """Largest local rate a ``spec`` can draw: ``alpha_high * (1 + dev_const + dev_sine)``."""
    p = _params(spec)
    return p["alpha_high"] * (1.0 + p["dev_const"] + p["dev_sine"])


def lowest_rate(spec: Mapping[str, Any]) -> float:
    """Smallest local rate a ``spec`` can draw: ``alpha_low * (1 - dev_const - dev_sine)``."""
    p = _params(spec)
    return p["alpha_low"] * (1.0 - p["dev_const"] - p["dev_sine"])


def source_window_s(spec: Mapping[str, Any], duration_s: float) -> float:
    """Source seconds a ``duration_s`` output chunk may read under ``spec``.

    Every warp drawn from ``spec`` (and the un-warped native path) fits in a
    window this long, so :func:`apply_rate_warp` never runs out of source.
    """
    return worst_case_rate(spec) * (float(duration_s) + 2.0 * EDGE_S) + SOURCE_MARGIN_S


@njit(cache=True, fastmath=True)
def _sinc_interp(data_t: np.ndarray, pos: np.ndarray, table: np.ndarray, half: int, phases: int):
    """``out[j, c] = sum_n data_t[n, c] * h(n - pos[j])`` over the 2*half taps
    around ``pos[j]`` (``h`` tabulated, linearly interpolated between phases).
    Time-major ``(N, C)`` so the channel loop is contiguous. The caller
    guarantees every tap index is inside ``data_t``."""
    n_ch = data_t.shape[1]
    m = pos.shape[0]
    last = table.shape[0] - 1
    out = np.empty((m, n_ch), dtype=np.float32)
    acc = np.zeros(n_ch, dtype=np.float32)
    for j in range(m):
        p = pos[j]
        n0 = int(np.floor(p))
        frac = p - n0
        acc[:] = 0.0
        for k in range(1 - half, half + 1):
            x = (k - frac + half) * phases
            i = int(x)
            if i >= last:
                h = table[last]
            else:
                w = x - i
                h = table[i] * (1.0 - w) + table[i + 1] * w
            hf = np.float32(h)
            row = data_t[n0 + k]
            for ch in range(n_ch):
                acc[ch] += hf * row[ch]
        out[j] = acc
    return out


def _label_track(
    frame: td.Frame, t_src: np.ndarray, gain: np.ndarray, audio_t_start: float
) -> np.ndarray:
    """``gain * r(t_src)`` per rotor; ``t_src`` in seconds since the audio start."""
    _, rps_key, needs_clean = resolve_motor_tracks(frame)
    motor = cast(td.Series, frame[rps_key])
    if motor.data is None or motor.dim_size("time") == 0:
        return np.zeros((4, t_src.size), dtype=np.float32)
    vals = np.asarray(motor.data, dtype=np.float64)
    if needs_clean:
        vals = clean_command_spikes(vals)
    ti = motor.tindex
    if isinstance(ti, td.StampIndex):
        times = np.asarray(ti.abs_stamps, dtype=np.float64)
    else:
        times = np.asarray(cast(td.GridIndex, ti).sample_times(), dtype=np.float64)
    times = times - audio_t_start
    out = np.empty((vals.shape[0], t_src.size), dtype=np.float32)
    for r in range(vals.shape[0]):
        out[r] = (gain * np.interp(t_src, times, vals[r])).astype(np.float32)
    return out


def apply_rate_warp(
    frame: td.Frame,
    params: RateWarpParams | None,
    *,
    target_len: int,
    sample_rate: int,
    label_rate_hz: float = LABEL_RATE_HZ,
) -> td.Frame:
    """Warp a native-rate noise window by ``params`` and decimate it to ``sample_rate``.

    ``params=None`` is the un-warped native path (identity map, decimation
    only), so a pool kept at its native rate yields chunks in one convention
    whether the warp fired or not. Raises when the window audio is not above
    the model rate or is too short for the warp (never pads).
    """
    audio = cast(td.Series, frame["audio"])
    sr_in = float(cast(td.GridIndex, audio.tindex).sr)
    if not sr_in.is_integer() or sr_in <= sample_rate:
        raise ValueError(
            f"rate warp needs native-rate audio above the model rate {sample_rate}, got {sr_in}"
        )
    fs = int(sr_in)
    data = np.asarray(audio.data, dtype=np.float32)
    is_mono = data.ndim == 1
    if is_mono:
        data = data[None, :]
    n_src = data.shape[-1]

    warp = params if params is not None else RateWarpParams(1.0, 1.0, 0.0, 1.0, 0.0)
    edge_out = int(round(EDGE_S * sample_rate))
    edge_in = edge_out * fs / float(sample_rate)
    n_native = int(np.ceil((target_len + 2 * edge_out) * fs / float(sample_rate)))
    # Native output grid on the chunk's time axis: t = 0 is output sample edge_out.
    t_native = np.arange(n_native, dtype=np.float64) / fs - edge_out / float(sample_rate)
    # Source offset: tau(-EDGE_S) >= -peak * EDGE_S keeps the first taps in range.
    s0 = float(np.ceil(warp.peak_rate * edge_in) + KERNEL_HALF + 1)
    pos = s0 + warp.tau(t_native) * fs
    if int(np.floor(pos[-1])) + KERNEL_HALF >= n_src:
        raise ValueError(
            f"rate warp (peak rate {warp.peak_rate:.3f}) reads source sample "
            f"{pos[-1]:.0f} + {KERNEL_HALF} taps beyond the {n_src}-sample window "
            "(the stream window is undersized for this policy)"
        )
    offset = s0 - edge_in
    if params is None and offset.is_integer():
        start = int(offset)
        native = np.ascontiguousarray(data[:, start : start + n_native].T)  # (time, ch)
    else:
        lo = int(np.floor(pos[0])) - KERNEL_HALF + 1
        hi = int(np.floor(pos[-1])) + KERNEL_HALF + 1
        native = _sinc_interp(
            np.ascontiguousarray(data[:, lo:hi].T),
            pos - lo,
            _KERNEL_TABLE,
            KERNEL_HALF,
            KERNEL_PHASES,
        )

    decimated = soxr.resample(native, fs, int(sample_rate), quality="HQ")  # (time, ch)
    out = np.ascontiguousarray(decimated[edge_out : edge_out + target_len].T, dtype=np.float32)
    if out.shape[-1] != target_len:
        raise ValueError(f"rate warp produced {out.shape[-1]} samples, expected {target_len}")

    n_label = int(np.ceil(target_len / float(sample_rate) * label_rate_hz)) + 1
    t_label = np.arange(n_label, dtype=np.float64) / float(label_rate_hz)
    label = _label_track(
        frame, s0 / fs + warp.tau(t_label), warp.alpha(t_label), float(audio.t_start)
    )

    entries: dict[str, Any] = {
        "audio": td.uniform(
            out[0] if is_mono else out, int(sample_rate), dims=audio.dims, t_start=0.0
        ),
        "rps": td.uniform(label, int(round(label_rate_hz)), dims=("rotor", "time"), t_start=0.0),
    }
    if "meta" in frame:
        entries["meta"] = frame["meta"]
    return td.Frame(entries)
