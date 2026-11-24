"""Audio-clock calibration of the three November 2026 DJI M100 / ZOOM sessions."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from data_processing.sources.motor_dat import DatTelemetry, read_motor_dat

# WAV time at native tick zero (s), native ticks per WAV second. The three
# recording sessions do not share an offset. See docs/data-catalog.md.
AUDIO_CLOCK = {
    "FLY013": (29.091134, 84_000_904.0),
    "FLY018": (-53.894454, 84_000_510.0),
    "FLY022": (-273.865007, 84_001_237.0),
}


def audio_seconds(flight: str, ticks: ArrayLike) -> NDArray[np.float64]:
    """Map unwrapped native ticks to seconds from the paired WAV's first sample.

    Negative values and times beyond EOF are real, non-overlapping telemetry.
    For CSV rows unwrap the full ``Clock:Tick#`` column with ``motor_dat.datcon_ticks``
    first; the resulting stamp dates the exported row, not its held sensors.
    """
    offset, hz = AUDIO_CLOCK[flight]
    return offset + np.asarray(ticks, dtype=np.float64) / hz


def read_aligned_motor(path: str | Path) -> DatTelemetry:
    """Open all November DAT groups with ``audio_s`` on each native packet clock.

    Keeps samples outside the WAV. No track is resampled or forward-filled.
    The optional DatCon table is aligned on its export row clock instead.
    """
    path = Path(path)
    if path.stem not in AUDIO_CLOCK:
        raise KeyError(f"no WAV clock calibration for {path.stem!r}")
    return read_motor_dat(path, audio_clock=partial(audio_seconds, path.stem))
