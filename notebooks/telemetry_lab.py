"""Helpers for ``telemetry_lab.ipynb``: native-rate rotor-speed telemetry.

Load every flight of one telemetry rig at its OWN logging rate, cut the
airborne segments by the campaign's frozen rule, and give back plain numpy
arrays plus a few one-line statistics (ACF, Welch PSD, structure function).
Nothing here fits a model; the notebook is for looking at the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal

#: Rig name -> (frames dataset, rps entry names, rig filter, recording filter).
RIGS: dict[str, dict[str, Any]] = {
    "neurobem": {"dataset": "NeuroBEM-frames", "keys": ("rps",), "rig": "neurobem_quad"},
    "blackbird": {"dataset": "Blackbird-frames", "keys": ("rps",), "rig": "blackbird_quad"},
    "vid": {"dataset": "VID-frames", "keys": ("rps",), "rig": "vid_m100"},
    "nanobench": {"dataset": "NanoBench-frames", "keys": ("rps",), "rig": "nanobench_cf21b"},
    "pitcn": {"dataset": "PITCN-frames", "keys": ("rps",), "rig": "pitcn_quad"},
    "dregon_measured": {
        "dataset": "DREGON-frames",
        "keys": ("motors_measured",),
        "select": "room1",
    },
    "dregon_command": {"dataset": "DREGON-frames", "keys": ("motors_command",), "select": "room1"},
    "michaels": {"dataset": "michaels-frames", "keys": ("rps",)},
}

#: The rotor-speed label rate the training stream works at (16 kHz / 512).
LABEL_RATE_HZ = 31.25


@dataclass
class Flight:
    rig: str
    name: str
    fs: float  # uniform grid rate, Hz
    rps: np.ndarray  # (4, T) rev/s on the uniform grid
    raw_t: np.ndarray  # (N,) logged timestamps, s (uniform grid if the source is uniform)
    raw_rps: np.ndarray  # (4, N) as logged

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.rps.shape[-1]) / self.fs

    @property
    def duration_s(self) -> float:
        return self.rps.shape[-1] / self.fs

    def airborne(self, min_s: float = 5.0) -> list[slice]:
        """Airborne segments (campaign rule) of at least ``min_s`` seconds."""
        from experiments.rps_traj.data import airborne_segments

        return [
            s for s in airborne_segments(self.rps, self.fs) if (s.stop - s.start) / self.fs >= min_s
        ]


def load(rig: str, limit: int | None = None) -> list[Flight]:
    """Every flight of ``rig`` at its native rate (see ``RIGS``)."""
    from data_processing.frames import meta_dict
    from data_processing.streams import iter_published_frames

    spec = RIGS[rig]
    out: list[Flight] = []
    for frame in iter_published_frames(spec["dataset"]):
        meta = meta_dict(frame)
        rid = str(meta.get("recording_id"))
        if spec.get("select") and spec["select"] not in rid:
            continue
        group = meta.get("system")
        try:
            sysd = {k: group[k] for k in group} if group is not None else {}
        except TypeError:
            sysd = {}
        want = spec.get("rig")
        if want is not None and sysd.get("rig") not in (None, want):
            continue
        key = next((k for k in spec["keys"] if k in frame), None)
        if key is None:
            continue
        series = frame[key]
        values = np.asarray(series.data, dtype=np.float64)
        values = values[None, :] if values.ndim == 1 else values
        if values.shape[-1] < 64 or values.shape[0] != 4:
            continue
        idx = series.indexes["time"]
        sr = getattr(idx, "sr", None)
        if sr:
            fs = float(sr)
            grid = values
            t = np.arange(values.shape[-1]) / fs
        else:
            t = np.asarray(idx.timestamps, dtype=np.float64)
            fs = 1.0 / float(np.median(np.diff(t)))
            n = int(np.floor((t[-1] - t[0]) * fs)) + 1
            tg = t[0] + np.arange(n) / fs
            grid = np.stack([np.interp(tg, t, row) for row in values])
        out.append(Flight(rig=rig, name=rid, fs=fs, rps=grid, raw_t=t, raw_rps=values))
        if limit is not None and len(out) >= limit:
            break
    return sorted(out, key=lambda f: f.name)


# ── one-line statistics ──────────────────────────────────────────────────────


def highpass(
    x: np.ndarray, fs: float, f_hp: float = LABEL_RATE_HZ / 2, order: int = 4
) -> np.ndarray:
    """Zero-phase Butterworth high-pass along the last axis (what a label at ``LABEL_RATE_HZ`` cannot carry)."""
    if f_hp >= 0.45 * fs:
        raise ValueError(f"f_hp={f_hp} Hz is at or above 0.45*fs={0.45 * fs:.1f} Hz")
    sos = signal.butter(order, f_hp, btype="high", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1)


def label_residual(x: np.ndarray, fs: float, label_fs: float = LABEL_RATE_HZ) -> np.ndarray:
    """``x`` minus ``x`` resampled to the label rate and back (linear both ways)."""
    n = x.shape[-1]
    t = np.arange(n) / fs
    tl = np.arange(0.0, t[-1], 1.0 / label_fs)
    down = np.stack([np.interp(tl, t, row) for row in np.atleast_2d(x)])
    back = np.stack([np.interp(t, tl, row) for row in down])
    return x - (back[0] if x.ndim == 1 else back)


def acf(x: np.ndarray, fs: float, max_lag_s: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Normalised autocorrelation of a 1-D series at lags 0..``max_lag_s``; returns (lag_s, rho)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    m = int(max_lag_s * fs)
    full = signal.correlate(x, x, mode="full", method="fft")[x.size - 1 : x.size + m]
    rho = full / full[0]
    return np.arange(m + 1) / fs, rho


def welch(x: np.ndarray, fs: float, nperseg_s: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD (Hann, 50 % overlap) of a 1-D series; returns (f_hz, P)."""
    nper = min(int(nperseg_s * fs), x.size)
    return signal.welch(x, fs=fs, nperseg=nper, window="hann")


def loglog_slope(f: np.ndarray, P: np.ndarray, band: tuple[float, float]) -> float:
    """Least-squares log-log slope of ``P(f)`` inside ``band`` (0 = white, -2 = OU tail)."""
    m = (f >= band[0]) & (f <= band[1]) & (P > 0)
    if m.sum() < 4:
        return float("nan")
    return float(np.polyfit(np.log(f[m]), np.log(P[m]), 1)[0])


def structure_function(theta: np.ndarray, fs: float, lags_s: np.ndarray) -> np.ndarray:
    """``Var[theta(t+tau) - theta(t)]`` at the given lags (rad^2 if ``theta`` is in rad)."""
    out = np.full(lags_s.size, np.nan)
    for i, tau in enumerate(lags_s):
        k = int(round(tau * fs))
        if 0 < k < theta.size:
            d = theta[k:] - theta[:-k]
            out[i] = float(np.var(d))
    return out


def phase(nu_rad_s: np.ndarray, fs: float) -> np.ndarray:
    """Shaft-angle error ``theta = integral(nu) dt`` (rad) by cumulative sum."""
    return np.cumsum(nu_rad_s, axis=-1) / fs


def white_null(n: int, fs: float, f_hp: float = LABEL_RATE_HZ / 2, seed: int = 0) -> np.ndarray:
    """White noise of length ``n`` through the same high-pass: the ACF shape the filter alone produces."""
    return highpass(np.random.default_rng(seed).standard_normal(n), fs, f_hp)
