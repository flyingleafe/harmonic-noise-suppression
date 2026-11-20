"""Target-calibrated source-stage adaptations of Heutschi (2020) and Yang (2025).

Heutschi: native-rate recorded reference, quadratic Lagrange resampling, and
third-octave EQ linear in RPM (10.1051/aacus/2020023, Eqs. 2 and 5).
Yang: pulse events, shared piecewise frequency micromodulation, white noise,
1024-sample STFT EQ and a second-order 2400-Hz low-pass, at the official 48 kHz
source rate (10.23919/EUSIPCO63237.2025.11226465). Upstream revision:
da80f4a22c35b9361ae8db879888764264d99d0f, DroneSynthesis2DOA.

Both omit far-field propagation and fit spectral parameters on calibration
recordings only. Actual telemetry replaces synthetic flight/wind trajectories.
Yang additionally receives separate per-rotor telemetry (the upstream wrapper
shares one macro trajectory); no duplicate wind-driven speed noise is added.
The event-timing variant retains integer-period quantization; the phase-carry
variant removes that implementation artifact and is reported separately.
Neither adaptation is represented as the original authors' deployed system.

The pulse/micromodulation implementation is adapted from X. Yang's MIT source.
Copyright (c) 2025 SOUNDS-RESEARCH
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from scipy import signal

from data_processing.noise_model.resample import antialias, decimate_audio
from experiments.generator_paper.cases import FlightClip

UPSTREAM_SHA = "da80f4a22c35b9361ae8db879888764264d99d0f"
YANG_SOURCE_SR = 48000
EQ_CENTRES = np.array(
    [
        100,
        125,
        160,
        200,
        250,
        315,
        400,
        500,
        630,
        800,
        1000,
        1250,
        1600,
        2000,
        2500,
        3150,
        4000,
        5000,
        6300,
        8000,
    ],
    dtype=np.float64,
)
_EPS = 1e-24
_PREVIEW_S = 4.0


@dataclass(frozen=True)
class HeutschiFit:
    reference: np.ndarray
    native_sr: int
    reference_rps: float
    slope_db_per_rpm: np.ndarray
    report: dict[str, Any]


@dataclass(frozen=True)
class YangFit:
    reference_rps: float
    offset_db: np.ndarray
    slope_db_per_rpm: np.ndarray
    phase_carry: bool
    modulation_range_hz: float
    white_ratio: float
    report: dict[str, Any]


def _psd(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    n = min(int(round(0.256 * sr)), audio.shape[-1])
    f, p = signal.welch(
        np.atleast_2d(audio),
        fs=sr,
        window="hann",
        nperseg=n,
        noverlap=3 * n // 4,
        detrend=False,  # pyright: ignore[reportArgumentType]
        scaling="density",
        axis=-1,
    )
    return f, p


def _bands(f: np.ndarray, p: np.ndarray) -> np.ndarray:
    power = []
    for centre in EQ_CENTRES:
        keep = (f >= centre / 2 ** (1 / 6)) & (f < centre * 2 ** (1 / 6))
        if not keep.any():
            raise ValueError(f"no PSD support for the {centre:g}-Hz EQ band")
        power.append(p[..., keep].sum(axis=-1) * (f[1] - f[0]))
    return np.stack(power, axis=-1)


def _segments(clips: list[FlightClip]):
    for clip in clips:
        n = int(round(_PREVIEW_S * clip.sr))
        for start in range(0, clip.audio.shape[-1] - n + 1, n):
            yield clip, start, clip.audio[:, start : start + n], clip.rps[:, start : start + n]


def _equalize(
    audio: np.ndarray,
    rps: np.ndarray,
    sr: int,
    reference_rps: float,
    offset_db: np.ndarray,
    slope_db_per_rpm: np.ndarray,
) -> np.ndarray:
    f, t, spec = signal.stft(
        np.atleast_2d(audio),
        fs=sr,
        window="hann",
        nperseg=1024,
        noverlap=512,
        boundary="zeros",
        padded=True,
        axis=-1,
    )
    rpm_delta = 60.0 * (
        np.interp(t, np.arange(rps.shape[-1]) / sr, rps.mean(axis=0)) - reference_rps
    )
    offsets = np.stack([np.interp(f, EQ_CENTRES, row) for row in offset_db])
    slopes = np.stack([np.interp(f, EQ_CENTRES, row) for row in slope_db_per_rpm])
    gain = 10.0 ** ((offsets[..., None] + slopes[..., None] * rpm_delta) / 20.0)
    shaped = spec * gain
    _, out = signal.istft(shaped, fs=sr, nperseg=1024, noverlap=512, time_axis=-1, freq_axis=-2)
    return out[:, : audio.shape[-1]]


def _warp_reference(
    reference: np.ndarray,
    native_sr: int,
    reference_rps: float,
    rps: np.ndarray,
    sr: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample on a safe work grid before the common observation decimator.

    The work rate prevents high native-reference frequencies from aliasing into
    the output passband. A finite reference must cover the requested source
    interval; we never synthesize padding, silently loop it or drop high bands.
    """
    ratio = rps.mean(axis=0) / reference_rps
    if not np.isfinite(ratio).all() or np.any(ratio <= 0):
        raise ValueError("Heutschi resampling requires finite positive rotor speeds")
    oversample = max(1, int(np.ceil(float(ratio.max()) / 2 + (sr / 2) / native_sr)))
    work_sr = native_sr * oversample
    n_work = int(round(rps.shape[-1] * work_sr / sr))
    t = np.arange(n_work) / work_sr
    ratio_work = np.interp(t, np.arange(ratio.size) / sr, ratio)
    positions = np.cumsum(ratio_work) * (native_sr / work_sr)
    positions -= positions[0]
    available = reference.shape[-1] - 3 - float(positions[-1])
    if available <= 1:
        raise ValueError(
            "native reference is too short for this trajectory; use a longer calibration "
            "reference or a shorter evaluation window, not looped/zero-filled audio"
        )
    positions += rng.uniform(1.0, available)
    index = np.floor(positions).astype(np.int64)
    frac = positions - index
    out = reference[:, index] * (1.0 - frac * frac)
    out += reference[:, index - 1] * (0.5 * frac * (frac - 1.0))
    out += reference[:, index + 1] * (0.5 * frac * (frac + 1.0))
    return decimate_audio(antialias(out, work_sr), work_sr, sr)[:, : rps.shape[-1]]


def fit_heutschi(clips: list[FlightClip], *, seed: int = 0) -> HeutschiFit:
    if not clips:
        raise ValueError("Heutschi calibration requires real calibration windows")
    mean_speeds = np.array([c.rps.mean() for c in clips])
    centre = float(np.median(mean_speeds))
    reference_clip = min(
        clips, key=lambda c: (abs(float(c.rps.mean()) - centre), float(c.rps.mean(axis=0).std()))
    )
    reference = np.array(reference_clip.native_audio, dtype=np.float64, copy=True)
    reference_rps = float(reference_clip.rps.mean())
    x, errors = [], []
    rng = np.random.default_rng(seed)
    for clip, _, real, rps in _segments(clips):
        raw = _warp_reference(reference, reference_clip.native_sr, reference_rps, rps, clip.sr, rng)
        f, pr = _psd(real, clip.sr)
        _, pg = _psd(raw, clip.sr)
        errors.append(10.0 * np.log10((_bands(f, pr) + _EPS) / (_bands(f, pg) + _EPS)))
        x.append(60.0 * (float(rps.mean()) - reference_rps))
    if not x:
        raise ValueError("calibration windows must contain complete 4-s emission segments")
    xx = np.asarray(x)
    yy = np.asarray(errors)
    # Through-origin regression is Eq. 2: the reference supplies the intercept.
    denominator = float(xx @ xx)
    slopes = (
        np.einsum("n,nmb->mb", xx, yy) / denominator if denominator > 0 else np.zeros_like(yy[0])
    )
    report = {
        "method": "Heutschi 2020 source-stage adaptation",
        "doi": "10.1051/aacus/2020023",
        "reference_clip": reference_clip.key,
        "native_sr": reference_clip.native_sr,
        "reference_rps": reference_rps,
        "eq_centres_hz": EQ_CENTRES.tolist(),
        "slope_db_per_rpm": slopes.tolist(),
        "slope_identified": bool(denominator > 0),
        "calibration_segments": len(x),
        "calibration_clip_keys": [c.key for c in clips],
        "input_control": "mean of supplied per-rotor telemetry; no synthetic wind trajectory",
        "propagation": "omitted: source-fixed onboard microphones",
        "reference_looping": False,
        "seed": seed,
    }
    return HeutschiFit(reference, reference_clip.native_sr, reference_rps, slopes, report)


def render_heutschi(
    fit: HeutschiFit, rps: np.ndarray, *, sr: int, rng: np.random.Generator
) -> np.ndarray:
    raw = _warp_reference(fit.reference, fit.native_sr, fit.reference_rps, rps, sr, rng)
    return _equalize(
        raw, rps, sr, fit.reference_rps, np.zeros_like(fit.slope_db_per_rpm), fit.slope_db_per_rpm
    ).astype(np.float32)


def _micro_modulation(
    n: int, sr: int, peak_to_peak_hz: float, rng: np.random.Generator
) -> np.ndarray:
    out = np.zeros(n)
    if peak_to_peak_hz == 0:
        return out
    start = 0
    while start + 0.05 * sr < n:
        count = int(rng.uniform(0.02, 0.05) * sr)
        out[start : start + count] = rng.uniform(-peak_to_peak_hz / 2, peak_to_peak_hz / 2)
        start += count
    return out


def _pulse_events(
    base_frequency: np.ndarray,
    modulation: np.ndarray,
    *,
    sr: int,
    phase_carry: bool,
    initial_phase: float,
    pulse_width: int = 3,
) -> np.ndarray:
    """Published rectangular pulse events; optional fractional-period carry."""
    out = np.zeros(base_frequency.size)
    time = float(int(initial_phase * sr / float(base_frequency[0])))
    events = []
    while time < base_frequency.size:
        index = int(time)
        frequency = float(base_frequency[index] + modulation[index])
        if not 0 < frequency < sr:
            raise ValueError(f"pulse frequency {frequency} Hz is outside (0, {sr})")
        period = sr / frequency
        time += period if phase_carry else int(period)
        events.append(int(time))
    # Upstream getOneRotorSignal deliberately drops its last two pulse events.
    for event in events[:-2]:
        out[event : event + pulse_width] = 1.0
    return out


def _yang_components(
    rps: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    *,
    phase_carry: bool,
    modulation_range_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(round(rps.shape[-1] * YANG_SOURCE_SR / sr))
    t = np.arange(n) / YANG_SOURCE_SR
    native_rps = np.stack([np.interp(t, np.arange(row.size) / sr, row) for row in rps])
    modulation = _micro_modulation(n, YANG_SOURCE_SR, modulation_range_hz, rng)
    pulse = np.zeros(n)
    for row in native_rps:
        pulse += _pulse_events(
            2.0 * row,
            modulation,
            sr=YANG_SOURCE_SR,
            phase_carry=phase_carry,
            initial_phase=float(rng.uniform()),
        )
    return pulse, rng.standard_normal(n), native_rps


def _yang_observe(audio: np.ndarray, sr: int) -> np.ndarray:
    lowpass = signal.butter(2, 2400.0, fs=YANG_SOURCE_SR, output="sos")
    # Without initial conditions, sosfilt returns only the filtered samples.
    filtered = cast(np.ndarray, signal.sosfilt(lowpass, audio, axis=-1))
    return decimate_audio(antialias(filtered, YANG_SOURCE_SR), YANG_SOURCE_SR, sr)


def fit_yang(clips: list[FlightClip], *, seed: int = 0, phase_carry: bool = False) -> YangFit:
    if not clips:
        raise ValueError("Yang calibration requires real calibration windows")
    reference_rps = float(np.median([c.rps.mean() for c in clips]))
    candidates = []
    fitted = []
    # These are calibration candidates, all retained in the report; never test-selected.
    for modulation_range in (0.0, 2.0, 5.0, 10.0, 20.0, 40.0):
        rng = np.random.default_rng(seed)
        spectral = []
        x = []
        for clip, _, real, rps in _segments(clips):
            pulse, white, _ = _yang_components(
                rps,
                clip.sr,
                rng,
                phase_carry=phase_carry,
                modulation_range_hz=modulation_range,
            )
            f, pr = _psd(real, clip.sr)
            _, pp = _psd(_yang_observe(pulse, clip.sr), clip.sr)
            _, pn = _psd(_yang_observe(white, clip.sr), clip.sr)
            spectral.append((f, pr, pp, pn))
            x.append(60.0 * (float(rps.mean()) - reference_rps))
        if not spectral:
            raise ValueError("calibration windows must contain complete 4-s emission segments")
        xx = np.column_stack([np.ones(len(x)), np.asarray(x) / 1000.0])
        for white_ratio in (0.002, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
            y = np.asarray(
                [
                    10.0
                    * np.log10(
                        (_bands(f, pr) + _EPS) / (_bands(f, pp + white_ratio**2 * pn) + _EPS)
                    )
                    for f, pr, pp, pn in spectral
                ]
            )
            coefficients, _, rank, _ = np.linalg.lstsq(xx, y.reshape(len(x), -1), rcond=None)
            intercept = coefficients[0].reshape(y.shape[1:])
            slopes = coefficients[1].reshape(y.shape[1:]) / 1000.0
            losses = []
            for delta, (f, pr, pp, pn) in zip(x, spectral, strict=True):
                gain = np.stack(
                    [
                        np.interp(f, EQ_CENTRES, b + s * delta)
                        for b, s in zip(intercept, slopes, strict=True)
                    ]
                )
                predicted = (pp + white_ratio**2 * pn) * 10.0 ** (gain / 10.0)
                keep = (f >= 125.0) & (f <= 7200.0)
                floor = max(float(pr[:, keep].mean()) * 1e-10, _EPS)
                error = 10.0 * np.log10((predicted[:, keep] + floor) / (pr[:, keep] + floor))
                losses.append(float(np.sqrt(np.mean(error**2, axis=-1)).mean()))
            score = float(np.mean(losses))
            candidates.append(
                {
                    "modulation_range_hz": modulation_range,
                    "white_ratio": white_ratio,
                    "calibration_lsd_db": score,
                    "eq_design_rank": int(rank),
                }
            )
            fitted.append((score, intercept, slopes, modulation_range, white_ratio))
    _, intercept, slopes, modulation_range, white_ratio = min(fitted, key=lambda row: row[0])
    report = {
        "method": "Yang 2025 source-stage adaptation",
        "doi": "10.23919/EUSIPCO63237.2025.11226465",
        "upstream": "https://github.com/SOUNDS-RESEARCH/DroneSynthesis2DOA",
        "upstream_sha": UPSTREAM_SHA,
        "license": "MIT",
        "source_sr": YANG_SOURCE_SR,
        "pulse_shape": "rectangular",
        "pulse_width_source_samples": 3,
        "blades": 2,
        "phase_carry": phase_carry,
        "modulation_range_hz": modulation_range,
        "white_ratio": white_ratio,
        "relative_wind_speed_jitter": 0.0,
        "input_control": "separate observed per-rotor RPS; replaces shared synthetic flight trajectory",
        "lowpass_hz": 2400.0,
        "lowpass_order": 2,
        "reference_rps": reference_rps,
        "offset_db": intercept.tolist(),
        "slope_db_per_rpm": slopes.tolist(),
        "eq_centres_hz": EQ_CENTRES.tolist(),
        "calibration_candidates": candidates,
        "selection": "minimum calibration Welch-LSD; no FLY124 observations",
        "calibration_segments": len(spectral),
        "calibration_clip_keys": [c.key for c in clips],
        "seed": seed,
    }
    return YangFit(
        reference_rps, intercept, slopes, phase_carry, modulation_range, white_ratio, report
    )


def render_yang(fit: YangFit, rps: np.ndarray, *, sr: int, rng: np.random.Generator) -> np.ndarray:
    pulse, white, native_rps = _yang_components(
        rps,
        sr,
        rng,
        phase_carry=fit.phase_carry,
        modulation_range_hz=fit.modulation_range_hz,
    )
    raw = pulse + fit.white_ratio * white
    shaped = _equalize(
        raw,
        native_rps,
        YANG_SOURCE_SR,
        fit.reference_rps,
        fit.offset_db,
        fit.slope_db_per_rpm,
    )
    return _yang_observe(shaped, sr)[:, : rps.shape[-1]].astype(np.float32)
