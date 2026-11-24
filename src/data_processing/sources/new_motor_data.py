"""Audio-clock calibration of the three November 2026 DJI M100 / ZOOM sessions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from data_processing.sources.motor_dat import read_motor_dat

# WAV time at native tick zero (s), native ticks per WAV second. The three
# recording sessions do not share an offset. See docs/data-catalog.md.
_AUDIO_CLOCK = {
    "FLY013": (29.091134, 84_000_904.0),
    "FLY018": (-53.894454, 84_000_510.0),
    "FLY022": (-273.865007, 84_001_237.0),
}


def audio_seconds(flight: str, ticks: ArrayLike) -> NDArray[np.float64]:
    """Map unwrapped native ticks to seconds from the paired WAV's first sample.

    Negative values and times beyond EOF are real, non-overlapping telemetry.
    For CSV rows unwrap the full ``Clock:Tick#`` column with :func:`datcon_ticks`
    first; the resulting stamp dates the exported row, not its held sensors.
    """
    offset, hz = _AUDIO_CLOCK[flight]
    return offset + np.asarray(ticks, dtype=np.float64) / hz


def read_aligned_motor(path: str | Path) -> pd.DataFrame:
    """Read a November DAT as native RPM samples with a calibrated ``audio_s`` column.

    Keeps all packets, including those outside the WAV. Does not resample,
    invent telemetry for silent lead-ins/tails, or alter measured speeds.
    """
    path = Path(path)
    motor = read_motor_dat(path)
    motor.insert(0, "audio_s", audio_seconds(path.stem, motor["tick"]))
    return motor


def datcon_ticks(ticks: ArrayLike) -> NDArray[np.int64]:
    """Recover native ticks from a *complete, ordered* DatCon ``Clock:Tick#`` column.

    FLY013's exporter sometimes increments the high word both before and at
    the actual 32-bit rollover. Only the stored low word is authoritative.
    Unwrap before slicing: the full exports start in epoch zero and have no
    unobserved interval as long as a whole wrap (about 51 s).
    """
    low = np.asarray(ticks).astype(np.uint32, casting="unsafe", copy=False)
    if low.ndim != 1:
        raise ValueError("expected the one-dimensional Clock:Tick# column")
    result = np.empty(low.size, dtype=np.int64)
    if result.size:
        result[0] = low[0]
        np.cumsum(np.diff(low), dtype=np.int64, out=result[1:])
        result[1:] += result[0]
    return result
