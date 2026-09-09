"""Per-rotor harmonic profiles from the DREGON single-motor bench recordings.

``Motor{1-4}_{50,60,70,80,90}.wav`` each spin one rotor at a fixed setpoint
in front of the 8-mic array. Every line level is read per microphone from a
Welch periodogram on a comb refined against the audio (the filename setpoint
is 1-4 % above the acoustic rate, and the four motors differ by up to 2.8 %
at the same setpoint), and the per-mic level is removed before rotors are
compared, so what remains is each rotor's *timbre*: its relative harmonic
profile. Supersedes the 2026-07-18 report's nearest-mic, nominal-70-Hz,
twelve-harmonic figure.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import welch

BENCH_DIR = (
    Path.home()
    / ".cache/dload/materialized/DREGON/db39bcf762d0/DREGON_individual_motors_recordings"
)
MOTORS = ("Motor1", "Motor2", "Motor3", "Motor4")
SPEEDS = (50, 60, 70, 80, 90)
#: Acoustic rates from ``docs/experiments/dregon-telemetry-forensics.md``; Motor2_60 was unstable there.
ACOUSTIC_RPS = {
    ("Motor1", 50): 49.03,
    ("Motor1", 60): 58.78,
    ("Motor1", 70): 68.49,
    ("Motor1", 80): 78.22,
    ("Motor1", 90): 87.98,
    ("Motor2", 50): 48.46,
    ("Motor2", 60): None,
    ("Motor2", 70): 67.71,
    ("Motor2", 80): 77.31,
    ("Motor2", 90): 86.92,
    ("Motor3", 50): 49.15,
    ("Motor3", 60): 59.05,
    ("Motor3", 70): 68.71,
    ("Motor3", 80): 78.66,
    ("Motor3", 90): 88.45,
    ("Motor4", 50): 49.79,
    ("Motor4", 60): 59.81,
    ("Motor4", 70): 69.50,
    ("Motor4", 80): 79.38,
    ("Motor4", 90): 89.40,
}
NFFT = 2**16  # 0.67 Hz at 44.1 kHz


def spectrum_db(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    f, p = welch(x, sr, window="hann", nperseg=NFFT, noverlap=NFFT // 2, axis=0, scaling="density")
    return f, 10.0 * np.log10(np.maximum(p, 1e-30))


def refine_rate(f: np.ndarray, pdb_mean: np.ndarray, f0: float, k: int = 25) -> float:
    """Comb-sum maximization over +-3 % of ``f0`` in 0.005 rev/s steps."""
    df = f[1] - f[0]
    grid = np.arange(f0 * 0.97, f0 * 1.03, 0.005)
    scores = [pdb_mean[np.round(np.arange(1, k + 1) * g / df).astype(int)].sum() for g in grid]
    return float(grid[int(np.argmax(scores))])


def line_levels(
    f: np.ndarray, pdb: np.ndarray, f0: float, kmax: int = 60
) -> tuple[np.ndarray, np.ndarray]:
    """Per mic and order: peak dB within +-1.5 Hz of ``k f0`` and the local floor
    (median over +-f0/2 excluding +-4 Hz around the line). ``(M, K)`` each."""
    df = f[1] - f[0]
    lv = np.full((pdb.shape[1], kmax), np.nan)
    fl = np.full_like(lv, np.nan)
    for k in range(1, kmax + 1):
        c = k * f0
        a, b = int((c - 1.5) / df), int((c + 1.5) / df) + 1
        if b >= f.size:
            break
        lv[:, k - 1] = pdb[a:b].max(axis=0)
        w0, w1 = int((c - f0 / 2) / df), int((c + f0 / 2) / df)
        m = np.ones(w1 - w0, bool)
        m[int((c - 4) / df) - w0 : int((c + 4) / df) - w0] = False
        fl[:, k - 1] = np.median(pdb[w0:w1][m], axis=0)
    return lv, fl


def bench_profiles(
    start_s: float = 2.0, dur_s: float = 20.0, kmax: int = 60
) -> dict[tuple[str, int], dict]:
    """``{(motor, setpoint): {f0, lv (M,K), fl (M,K)}}`` for every bench file."""
    out = {}
    for p in sorted(BENCH_DIR.glob("Motor*_*.wav")):
        m = re.match(r"(Motor\d)_(\d+)", p.stem)
        assert m
        mot, sp = m.group(1), int(m.group(2))
        x, sr = sf.read(p, dtype="float32")
        x = x[int(start_s * sr) : int((start_s + dur_s) * sr)]
        f, pdb = spectrum_db(x, sr)
        f0 = ACOUSTIC_RPS[(mot, sp)]
        if f0 is None:
            f0 = refine_rate(f, pdb.mean(axis=1), sp * 0.97)
        f0 = refine_rate(f, pdb.mean(axis=1), f0)
        lv, fl = line_levels(f, pdb, f0, kmax)
        out[(mot, sp)] = dict(f0=f0, lv=lv, fl=fl)
    return out


def centred(lv: np.ndarray, kmax: int = 48) -> np.ndarray:
    """Line levels minus each mic's median over the first ``kmax`` orders."""
    x = lv[:, :kmax]
    return x - np.nanmedian(x, axis=1, keepdims=True)
