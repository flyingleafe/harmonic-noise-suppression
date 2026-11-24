"""Reproduce November DAT → WAV clock maps on disjoint moving-speed windows.

Run from the project: PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
nice -n 19 taskset -c 0 python scripts/michaels_calib/november_motor_clock.py

Times below are frozen window centres on the WAV. Coarse offsets come from
motor on/off transitions, not file lengths. The nuisance RPM gain is selected
on fit windows only; it is never applied to the delivered rotor speeds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import ndimage, signal

from data_processing.sources import raw_root
from data_processing.sources.motor_dat import ROTORS, read_motor_dat

# File, coarse WAV-minus-controller offset, interior fit centres, held-out centres.
WINDOWS = {
    "10_Nov/FLY013.dat": (
        28.6,
        [104.1, 209.5, 284.0, 380.0, 516.0],
        [54.5, 177.0, 249.0, 325.4, 454.0, 583.0],
    ),
    "14_Nov/FLY018.dat": (-54.0, [112.88, 442.54, 770.16], [21.72, 315.16, 592.62, 696.22, 922.68]),
    "15_Nov/FLY022.dat": (-274.0, [135.34, 494.14, 735.0], [35.52, 283.88, 331.48, 608.0, 873.82]),
}
NOMINAL = 84_000_000  # reference scale, not assumed to equal the WAV sample clock
GAIN = np.arange(0.995, 1.016, 0.001)


def scan(wav, centre, seed, ticks_s, speeds, bare):
    """Whitened log STFT at native-RPM-predicted orders, vs frequency gain and lag."""
    orders = np.array([84, 98]) if bare else np.arange(2, 15, 2)
    sr, nwin, hop, nfft = (48_000, 3840, 960, 8192) if bare else (6000, 960, 120, 4096)
    half = 7.5 if bare else 5.7
    extent = 100 if bare else 40
    shifts = np.arange(-extent, extent + 1)
    with sf.SoundFile(wav) as audio:
        assert audio.samplerate == 48_000
        start = round((centre - half) * audio.samplerate)
        audio.seek(start)
        x = audio.read(round(2 * half * audio.samplerate), dtype="float32", always_2d=True)[:, 0]
    if sr != 48_000:
        x = signal.resample_poly(x, 1, 48_000 // sr)
    _, st, magnitude = signal.spectrogram(
        x,
        sr,
        window="hann",
        nperseg=nwin,
        noverlap=nwin - hop,
        nfft=nfft,
        scaling="spectrum",
        mode="magnitude",
    )
    spec = np.log(np.maximum(magnitude.T, 1e-12))
    spec -= ndimage.gaussian_filter1d(spec, 6 if bare else 12, axis=1)
    np.clip(spec, -2.5, 4.0, out=spec)
    stride = 4 if bare else 2
    idx = np.arange(extent + 3, len(st) - extent - 3, stride)
    t = start / 48_000 + st[idx] - seed
    # Score only times bounded by neighbouring actual packets, not across holes.
    j = np.searchsorted(ticks_s, t)
    valid = (j > 0) & (j < len(ticks_s))
    j = np.clip(j, 1, len(ticks_s) - 1)
    valid &= (ticks_s[j] - ticks_s[j - 1]) <= 0.20
    rpm = np.stack([np.interp(t, ticks_s, v) for v in speeds.T], axis=1)
    weight = np.clip(abs(np.gradient(rpm, stride * hop / sr, axis=0)), 0.5, 30)
    weight *= (rpm > 20) & valid[:, None]
    if not np.any(weight):
        raise ValueError(f"no moving motor support at {centre}")
    density = weight.sum() * len(orders)
    result = np.empty((len(GAIN), len(shifts)))
    for n, gain in enumerate(GAIN):
        bins = rpm[:, :, None] * orders * gain * (nfft / sr)
        lower = np.clip(bins.astype(int), 0, spec.shape[1] - 2)
        frac = bins - lower
        for k, lag in enumerate(shifts):
            row = (idx + lag)[:, None, None]
            intensity = spec[row, lower] * (1 - frac) + spec[row, lower + 1] * frac
            result[n, k] = (intensity * weight[:, :, None]).sum() / density
    return shifts * (hop / sr) + seed, result


def measure(root, relative, seed, fit, holdout):
    dat = root / relative
    (wav,) = dat.parent.glob("*.wav")
    motor = read_motor_dat(dat).sort_values("tick")
    ticks_s = motor.tick.to_numpy() / NOMINAL
    speeds = motor[list(ROTORS)].to_numpy() / 60
    centres = np.array(fit + holdout)
    curves = []
    for centre in centres:
        offset_grid, curve = scan(wav, centre, seed, ticks_s, speeds, dat.stem == "FLY013")
        curves.append(curve)
    curves = np.stack(curves)
    # One frequency scale per recording. Do not choose a best gain separately per lag.
    gain = curves[: len(fit)].mean(axis=0).max(axis=1).argmax()
    offsets = []
    for curve in curves[:, gain, :]:
        i = curve.argmax()
        if i == 0 or i == len(curve) - 1:
            raise ValueError(f"{dat.stem}: lag peak outside scan")
        a, b, c = curve[i - 1 : i + 2]
        offsets.append(
            offset_grid[i] + 0.5 * (a - c) / (a - 2 * b + c) * (offset_grid[1] - offset_grid[0])
        )
    native = centres - seed
    slope, intercept = np.polyfit(native[: len(fit)], offsets[: len(fit)], 1)
    rate = NOMINAL / (1 + slope)
    residual_ms = 1000 * (np.array(offsets) - (intercept + slope * native))
    rows = pd.DataFrame(
        {
            "recording": dat.stem,
            "role": ["fit"] * len(fit) + ["holdout"] * len(holdout),
            "wav_window_s": centres,
            "native_center_tick": np.rint(native * NOMINAL).astype("int64"),
            "wav_minus_tick84_s": offsets,
            "residual_ms": residual_ms,
        }
    )
    print(
        f"{dat.stem}: A={intercept:.6f}, f={rate:.0f} Hz, held-out RMS="
        f"{np.mean(residual_ms[len(fit) :] ** 2) ** 0.5:.2f} ms; RPM gain for scoring={GAIN[gain]:.3f}"
    )
    return rows


if __name__ == "__main__":
    root = raw_root("new_motor_data")
    rows = pd.concat([measure(root, name, *cfg) for name, cfg in WINDOWS.items()])
    output = Path("results/new_motor_data_alignment/native_clock_windows.csv")
    output.parent.mkdir(exist_ok=True, parents=True)
    rows.to_csv(output, index=False)
