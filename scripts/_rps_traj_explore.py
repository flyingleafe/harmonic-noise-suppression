"""What does a real rotor-speed trajectory actually look like, second-order?

Campaign ``rps-trajectory-model`` needs a generative model of per-rotor speed
(rev/s) trajectories.  Before choosing a model class we measure the empirical
second-order structure of every rig whose per-rotor telemetry we already hold:

* Michael's DJI Matrice 100 (``recording_with_motor_speed`` FLY124/FLY125 and
  the held-out ``new-drone-noises`` FLY103/FLY108) — ESC feedback logged by
  DatCon at 29.41 Hz, calibrated to rev/s by
  :func:`data_processing.sources.michaels.read_motor_speeds`.
* DREGON (MikroKopter quad) — ``motor.measured`` at ~929 Hz effective in the
  ``room1`` free-flight set, ``motor.command`` only in ``room2``.

Everything is computed on a single FROZEN pipeline so that later model
discrepancies are measured against numbers produced here:

1. native samples -> uniform 100 Hz grid by linear interpolation.  DREGON is
   first put on a uniform 1 kHz grid and zero-phase Butterworth low-passed at
   40 Hz (order 4, ``filtfilt``) so the 10x decimation does not alias.
2. airborne segmentation, FROZEN RULE: ``m(t)`` = mean over rotors,
   ``thr = 0.5 * percentile(m, 90)``, airborne = *all* rotors above ``thr``,
   then erode 1.0 s at both ends of every run and keep runs >= 5 s.
3. per airborne run, per rotor demeaned: Welch PSD (10 s Hann, 50 % overlap,
   (rev/s)^2/Hz), the full 4x4 cross-spectral matrix, the six magnitude-squared
   coherences, the zero-lag correlation matrix, the eigen-decomposition of the
   4x4 covariance (with the eigenvectors projected on the four
   :data:`tracking.rotors.MIXER` control modes), ACFs on a log lag grid
   0.02-10 s, and the kurtosis of the 0.01 s and 0.5 s increments.
4. per rig: duration-weighted averages of all of the above.

All rotor axes are permuted into the MIXER row order
``[RFront, LFront, LBack, RBack]`` at load time -- Michael's CSV columns are
already in that order (:data:`~data_processing.sources.michaels.ROTOR_ORDER`);
DREGON's motor rows follow ``coordinates.mat['rotorsPos']``, which is
``[LBack, RBack, RFront, LFront]``, so DREGON rows are reindexed by
``(2, 3, 0, 1)``.

Two telemetry-channel audits sit alongside, because they turned out to gate
every spectral claim: :func:`channel_fidelity` /
:func:`dregon_channel_compare` (DREGON ``motor.measured`` is a ~45 Hz
sample-and-hold quantised to ~0.3 rev/s, so its >5 Hz spectrum is its own
staircase) and :func:`michaels_quantisation` (1 RPM steps, a ~17 Hz
asynchronous ESC update resampled into a 29.41 Hz log, and no anti-alias
roll-off).  :func:`same_drone_check` answers whether FLY103/108 are the same
airframe as FLY124/125.

Outputs (``results/rps_traj/explore/``): ``summary.json`` (every number, per
flight and per rig), ``findings.md`` (the read), and per rig
``*_psd.png``, ``*_coherence.png``, ``*_eigen.png``, ``*_acf.png``,
``*_timeseries.png`` plus ``michaels_quant.png`` and
``dregon_channel_fidelity.png``.

Raw trees needed (pull with ``dload pull <name>`` if missing):
``recording_with_motor_speed``, ``new-drone-noises``, ``DREGON``.

Usage:
    python scripts/_rps_traj_explore.py
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import scipy.signal as sg
import scipy.stats as st

from data_processing.sources import michaels as MI
from tracking.rotors import MIXER, MODE_NAMES

# ─── Frozen analysis constants ────────────────────────────────────────────────

FS = 100.0  # analysis grid (Hz)
DREGON_WORK_FS = 1000.0  # uniform grid DREGON is filtered on
DREGON_LP_HZ = 40.0  # zero-phase Butterworth cutoff before decimation
DREGON_LP_ORDER = 4
NPERSEG_S = 10.0  # Welch segment
ERODE_S = 1.0  # airborne-run erosion at each end
MIN_RUN_S = 5.0  # shortest airborne run kept
THR_PCTL = 90.0  # airborne threshold percentile of the rotor mean
THR_FRAC = 0.5  # airborne threshold = THR_FRAC * that percentile
BANDS = ((0.0, 0.5), (0.5, 5.0), (5.0, 15.0), (15.0, 50.0))
BAND_NAMES = ("lt0.5", "0.5-5", "5-15", "gt15")
ACF_LAGS_S = np.logspace(np.log10(0.02), np.log10(10.0), 30)
INCR_LAGS_S = (0.01, 0.5)

#: Rotor axis order used everywhere below (MIXER rows).
ROTORS = ("RFront", "LFront", "LBack", "RBack")
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
PAIR_NAMES = tuple(f"{ROTORS[i]}-{ROTORS[j]}" for i, j in PAIRS)
#: diagonal (same rotation sense) pairs, by index into PAIRS
DIAG_PAIRS = (1, 4)  # RFront-LBack, LFront-RBack

#: DREGON motor rows -> MIXER row order, from ``coordinates.mat['rotorsPos']``
#: (+x forward, +y left): rows are [LBack, RBack, RFront, LFront].
DREGON_TO_MIXER = (2, 3, 0, 1)
DREGON_NATIVE_ORDER = ("LBack", "RBack", "RFront", "LFront")

MICHAELS_FILES = {
    "FLY124": "data/recording_with_motor_speed/recording_1/FLY124.csv",
    "FLY125": "data/recording_with_motor_speed/recording_2/FLY125.csv",
    "FLY103": "data/new-drone-noises/FLY103.csv",
    "FLY108": "data/new-drone-noises/FLY108.csv",
}
DREGON_ROOT = Path("data/DREGON")
OUT_DIR = Path("results/rps_traj/explore")

MODE_UNIT = MIXER / np.linalg.norm(MIXER, axis=0, keepdims=True)  # (4, 4) unit columns

# ─── Records ──────────────────────────────────────────────────────────────────


@dataclass
class Flight:
    """One flight's native per-rotor telemetry, rows in :data:`ROTORS` order."""

    flight: str
    rig: str
    signal: str  # "measured" | "command"
    t: np.ndarray  # (M,) seconds from 0
    rps: np.ndarray  # (4, M) rev/s
    native_rate: float
    lowpass: bool
    source_file: str
    extra: dict[str, Any] = field(default_factory=dict)


def _native_rate(t: np.ndarray) -> float:
    return float(1.0 / np.median(np.diff(t)))


def _trim_all_nan(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    good = np.where(np.isfinite(x).any(axis=0))[0]
    sl = slice(int(good[0]), int(good[-1]) + 1)
    return t[sl], x[:, sl]


def load_michaels() -> Iterator[Flight]:
    for fid, rel in MICHAELS_FILES.items():
        path = Path(rel)
        t, rps = MI.read_motor_speeds(path)  # (M,), (4, M) rev/s, MIXER order
        raw_rpm = np.asarray(rps / MI.rps_scale_for(path) * 60.0)
        attrs = _michaels_attrs(path)
        t, rps = _trim_all_nan(t, rps)
        yield Flight(
            flight=fid,
            rig="michaels",
            signal="measured",
            t=t - t[0],
            rps=np.ascontiguousarray(rps, dtype=np.float64),
            native_rate=_native_rate(t),
            lowpass=False,
            source_file=rel,
            extra={
                "rps_scale": MI.rps_scale_for(path),
                "attrs": attrs,
                "raw_rpm_is_integer": bool(
                    np.all(
                        np.isclose(
                            raw_rpm[np.isfinite(raw_rpm)], np.round(raw_rpm[np.isfinite(raw_rpm)])
                        )
                    )
                ),
                "raw_rpm_unique": int(np.unique(raw_rpm[np.isfinite(raw_rpm)]).size),
                "rotor_order_native": list(MI.ROTOR_ORDER),
            },
        )


def _michaels_attrs(path: Path) -> dict[str, str]:
    # pandas' usecols overloads are narrower than its runtime contract.
    csv = pd.read_csv(path, low_memory=False, usecols=cast(Any, ["Attribute|Value"]), nrows=500)
    out: dict[str, str] = {}
    for v in csv["Attribute|Value"].dropna().astype(str):
        if "|" in v:
            k, _, val = v.partition("|")
            out.setdefault(k.strip(), val.strip())
    return out


def _dregon_mats() -> list[Path]:
    # ``data/DREGON/nosource/`` holds byte-duplicates of the per-recording
    # directories; take only the per-recording trees.
    return sorted(DREGON_ROOT.glob("DREGON_*_room*/*_motors.mat"))


def load_dregon() -> Iterator[Flight]:
    for path in _dregon_mats():
        rec = path.parent.name
        room = "room1" if rec.endswith("room1") else "room2"
        mat = scipy.io.loadmat(str(path))["motor"]
        ts = mat["timestamps"][0, 0].flatten().astype(np.float64)
        ts = ts - ts[0]
        names = mat.dtype.names
        for signal in ("measured", "command"):
            if signal not in names:
                continue
            vals = np.asarray(mat[signal][0, 0], dtype=np.float64).T  # (4, M)
            vals = vals[list(DREGON_TO_MIXER)]
            t, rps = _trim_all_nan(ts, vals)
            yield Flight(
                flight=rec.replace("DREGON_", ""),
                rig=f"dregon_{room}_{signal}",
                signal=signal,
                t=t - t[0],
                rps=np.ascontiguousarray(rps),
                native_rate=_native_rate(t),
                lowpass=True,
                source_file=str(path),
                extra={
                    "effective_rate_hz": float(len(t) / (t[-1] - t[0])),
                    "dt_max_ms": float(np.diff(t).max() * 1e3),
                    "rotor_order_native": list(DREGON_NATIVE_ORDER),
                    "trajectory": rec.split("_")[1],
                    "source_cond": rec.split("_")[2],
                },
            )


# ─── Frozen pipeline: resample + segment ──────────────────────────────────────


def _interp_rows(t: np.ndarray, x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    out = np.empty((x.shape[0], grid.size), dtype=np.float64)
    for i, row in enumerate(x):
        ok = np.isfinite(row)
        out[i] = np.interp(grid, t[ok], row[ok])
    return out


def to_grid(fl: Flight) -> tuple[np.ndarray, dict[str, Any]]:
    """Native samples -> uniform :data:`FS` grid (low-passed first if asked)."""
    t, x = fl.t, fl.rps
    if fl.lowpass:
        work = np.arange(0.0, t[-1], 1.0 / DREGON_WORK_FS)
        y = _interp_rows(t, x, work)
        b, a = cast(
            tuple[np.ndarray, np.ndarray],
            sg.butter(DREGON_LP_ORDER, DREGON_LP_HZ / (DREGON_WORK_FS / 2), btype="low"),
        )
        y = np.asarray(sg.filtfilt(b, a, y, axis=-1))
        t, x = work, y
    grid = np.arange(0.0, t[-1], 1.0 / FS)
    out = _interp_rows(t, x, grid)
    # how much of the grid sits inside a native gap wider than 1.5 native steps
    native_dt = 1.0 / fl.native_rate
    gaps = np.diff(fl.t)
    wide = gaps > 1.5 * native_dt
    gap_s = float(gaps[wide].sum()) if wide.any() else 0.0
    info = {
        "grid_samples": int(out.shape[1]),
        "grid_duration_s": float(out.shape[1] / FS),
        "native_samples": int(fl.rps.shape[1]),
        "native_rate_hz": fl.native_rate,
        "native_nan_frac": float(np.mean(~np.isfinite(fl.rps))),
        "interp_gap_frac": gap_s / float(fl.t[-1] - fl.t[0]),
        "max_native_gap_s": float(gaps.max()),
    }
    return out, info


def airborne_runs(x: np.ndarray) -> tuple[list[slice], float]:
    """FROZEN airborne rule.  Returns ``(runs, threshold)``."""
    m = x.mean(axis=0)
    thr = THR_FRAC * float(np.percentile(m, THR_PCTL))
    mask = np.all(x > thr, axis=0)
    erode = int(round(ERODE_S * FS))
    min_len = int(round(MIN_RUN_S * FS))
    runs: list[slice] = []
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    for lo, hi in zip(idx[0::2], idx[1::2]):
        lo, hi = int(lo) + erode, int(hi) - erode
        if hi - lo >= min_len:
            runs.append(slice(lo, hi))
    return runs, thr


# ─── Per-run second-order statistics ──────────────────────────────────────────


def _welch_matrix(x: np.ndarray, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    """``(freq, S)`` with ``S`` the (4, 4, F) cross-spectral matrix, density."""
    kw: dict[str, Any] = {
        "fs": FS,
        "window": "hann",
        "nperseg": nperseg,
        "noverlap": nperseg // 2,
        "detrend": "constant",
    }
    n = x.shape[0]
    f = np.asarray(sg.csd(x[0], x[0], **kw)[0])
    S = np.empty((n, n, f.size), dtype=np.complex128)
    for i in range(n):
        for j in range(i, n):
            s = np.asarray(sg.csd(x[i], x[j], **kw)[1])
            S[i, j] = s
            if i != j:
                S[j, i] = np.conj(s)
    return f, S


def _acf(x: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """(4, L) normalised autocorrelation of demeaned rows at sample ``lags``."""
    out = np.empty((x.shape[0], lags.size))
    var = x.var(axis=1)
    for k, L in enumerate(lags):
        out[:, k] = np.einsum("ij,ij->i", x[:, : x.shape[1] - L], x[:, L:]) / (
            (x.shape[1] - L) * var
        )
    return out


def _band_fracs(f: np.ndarray, psd: np.ndarray) -> dict[str, np.ndarray]:
    total = np.trapezoid(psd, f, axis=-1)
    out = {}
    for name, (lo, hi) in zip(BAND_NAMES, BANDS):
        sel = (f >= lo) & (f < hi)
        out[name] = np.trapezoid(psd[..., sel], f[sel], axis=-1) / total
    return out


def _hump(f: np.ndarray, psd: np.ndarray, fmax: float) -> dict[str, Any]:
    """Resonant-hump read of one PSD curve against a robust broadband baseline.

    Baseline: cubic polynomial in ``(log10 f, dB)`` refit four times with
    positive residuals downweighted, so it follows the broadband trend and not
    the peaks.  A "hump" is then a local maximum of the residual inside
    ``[0.3, fmax/1.2]`` Hz (the margin keeps the polynomial's end behaviour out
    of the answer).  ``prominence_db`` is the residual height; the width is the
    -3 dB residual crossing either side, and ``broad`` says that crossing never
    happened inside the band, i.e. it is a shoulder, not a resonance.
    """
    sel = (f >= 0.2) & (f <= fmax) & (psd > 0)
    fs_, db = f[sel], 10 * np.log10(psd[sel])
    lf = np.log10(fs_)
    w = np.ones_like(db)
    coef = np.polyfit(lf, db, 3, w=w)
    for _ in range(4):
        res = db - np.polyval(coef, lf)
        s = float(np.sqrt(np.average(res**2, weights=w)))
        w = np.where(res > 0, 1.0 / (1.0 + (res / max(s, 1e-9)) ** 2), 1.0)
        coef = np.polyfit(lf, db, 3, w=w)
    res = db - np.polyval(coef, lf)
    ok = np.flatnonzero((fs_ >= 0.3) & (fs_ <= fmax / 1.2))
    peaks = [i for i in ok[1:-1] if res[i] >= res[i - 1] and res[i] >= res[i + 1]]
    peaks.sort(key=lambda i: -res[i])
    picked: list[int] = []
    for i in peaks:
        if all(abs(lf[i] - lf[j]) > 0.15 for j in picked):
            picked.append(i)
        if len(picked) == 2:
            break

    def describe(k: int) -> dict[str, Any]:
        half = res[k] - 3.0
        lo, hi = k, k
        while lo > 0 and res[lo] > half:
            lo -= 1
        while hi < res.size - 1 and res[hi] > half:
            hi += 1
        broad = bool(res[lo] > half or res[hi] > half)
        return {
            "peak_hz": float(fs_[k]),
            "prominence_db": float(res[k]),
            "width_lo_hz": float(fs_[lo]),
            "width_hi_hz": float(fs_[hi]),
            "q": None if broad else float(fs_[k] / max(fs_[hi] - fs_[lo], 1e-9)),
            "broad": broad,
            "is_resonance": bool(res[k] >= 3.0 and not broad),
        }

    out = (
        describe(picked[0])
        if picked
        else {
            "peak_hz": None,
            "prominence_db": 0.0,
            "width_lo_hz": None,
            "width_hi_hz": None,
            "q": None,
            "broad": True,
            "is_resonance": False,
        }
    )
    out["secondary"] = describe(picked[1]) if len(picked) > 1 else None
    out["loglog_slope_db_per_decade"] = float(np.polyfit(lf, db, 1)[0])
    out["search_band_hz"] = [0.3, fmax / 1.2]
    return out


def analyse_run(x: np.ndarray, fmax_hump: float) -> dict[str, Any]:
    """Every per-run number.  ``x`` is (4, N) rev/s on the :data:`FS` grid."""
    n = x.shape[1]
    nperseg = min(int(round(NPERSEG_S * FS)), n)
    mean = x.mean(axis=1)
    xd = x - mean[:, None]
    f, S = _welch_matrix(xd, nperseg)
    psd = np.real(np.einsum("iif->if", S))
    coh = np.empty((len(PAIRS), f.size))
    coy = np.empty((len(PAIRS), f.size), dtype=np.complex128)
    for k, (i, j) in enumerate(PAIRS):
        coh[k] = np.abs(S[i, j]) ** 2 / np.maximum(psd[i] * psd[j], 1e-300)
        # complex coherency: keeps the cross-spectral PHASE, which the MSC and
        # the PSDs together cannot reconstruct
        coy[k] = S[i, j] / np.maximum(np.sqrt(psd[i] * psd[j]), 1e-300)

    cov = np.cov(xd)
    corr = np.corrcoef(xd)
    ev, evec = np.linalg.eigh(cov)
    order = np.argsort(ev)[::-1]
    ev, evec = ev[order], evec[:, order]
    # sign convention: largest-|component| positive, so plots are comparable
    evec = evec * np.sign(evec[np.argmax(np.abs(evec), axis=0), np.arange(4)])[None, :]
    proj = np.abs(evec.T @ MODE_UNIT)  # (eig, mode) cosines

    modes = (MIXER.T @ xd) / 4.0
    mode_var = modes.var(axis=1)
    eig_ts = evec.T @ xd
    f2, S_eig = _welch_matrix(eig_ts, nperseg)
    psd_eig = np.real(np.einsum("iif->if", S_eig))
    _, S_mode = _welch_matrix(modes, nperseg)
    psd_mode = np.real(np.einsum("iif->if", S_mode))

    lags = np.unique(np.maximum(1, np.round(ACF_LAGS_S * FS).astype(int)))
    lags = lags[lags < n // 2]
    acf = _acf(xd, lags)

    kurt = {}
    for lag_s in INCR_LAGS_S:
        L = max(1, int(round(lag_s * FS)))
        d = x[:, L:] - x[:, :-L]
        kurt[f"{lag_s}s"] = {
            "kurtosis_excess": st.kurtosis(d, axis=1, fisher=True, bias=False),
            "std": d.std(axis=1),
        }

    bands = _band_fracs(f, psd)
    return {
        "n_samples": n,
        "duration_s": n / FS,
        "nperseg": nperseg,
        "freq": f,
        "psd": psd,
        "coherence": coh,
        "csd_abs": np.abs(S),
        "coy_re": coy.real,
        "coy_im": coy.imag,
        "corr": corr,
        "cov": cov,
        "eigvals": ev,
        "eigvecs": evec,
        "eig_mode_proj": proj,
        "eig_frac": ev / ev.sum(),
        "mode_var": mode_var,
        "mode_var_frac": mode_var / mode_var.sum(),
        "psd_eig": psd_eig,
        "psd_mode": psd_mode,
        "acf_lags_s": lags / FS,
        "acf": acf,
        "increments": kurt,
        "mean": mean,
        "std": x.std(axis=1),
        "band_fracs": bands,
        "band_fracs_mode": _band_fracs(f, psd_mode),
        "hump_rotor_mean": _hump(f, psd.mean(axis=0), fmax_hump),
    }


# ─── Pooling ──────────────────────────────────────────────────────────────────


def _wavg(values: list[np.ndarray], weights: list[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return np.tensordot(w, np.stack(values), axes=(0, 0))


def pool(runs: list[dict[str, Any]], fmax_hump: float) -> dict[str, Any]:
    """Duration-weighted rig aggregate over per-run results."""
    w = [r["duration_s"] for r in runs]
    ref = max(runs, key=lambda r: r["freq"].size)
    f = ref["freq"]
    lags = ref["acf_lags_s"]

    def on_f(key: str, log: bool = True) -> list[np.ndarray]:
        """Per-run arrays on the reference frequency grid.

        Runs shorter than ``NPERSEG_S`` get a coarser Welch grid; those are
        interpolated onto the reference grid, log-log for positive spectra and
        linearly in log-frequency for signed quantities (the coherency parts).
        """
        out = []
        for r in runs:
            if r["freq"].size == f.size:
                out.append(r[key])
                continue
            v = np.atleast_2d(r[key])
            lf, lfr = np.log(np.maximum(f, 1e-6)), np.log(np.maximum(r["freq"], 1e-6))
            if log:
                rows = [np.exp(np.interp(lf, lfr, np.log(np.maximum(row, 1e-300)))) for row in v]
            else:
                rows = [np.interp(lf, lfr, row) for row in v]
            out.append(np.stack(rows))
        return out

    psd = _wavg(on_f("psd"), w)
    coh = _wavg(on_f("coherence"), w)
    acf = _wavg(
        [np.stack([np.interp(lags, r["acf_lags_s"], row) for row in r["acf"]]) for r in runs], w
    )
    out = {
        "n_runs": len(runs),
        "total_airborne_s": float(sum(w)),
        "freq": f,
        "psd": psd,
        "coherence": coh,
        "coherency_re": _wavg(on_f("coy_re", log=False), w),
        "coherency_im": _wavg(on_f("coy_im", log=False), w),
        "psd_eig": _wavg(on_f("psd_eig"), w),
        "psd_mode": _wavg(on_f("psd_mode"), w),
        "acf_lags_s": lags,
        "acf": acf,
        "corr": _wavg([r["corr"] for r in runs], w),
        "cov": _wavg([r["cov"] for r in runs], w),
        "eigvals": _wavg([r["eigvals"] for r in runs], w),
        "eig_frac": _wavg([r["eig_frac"] for r in runs], w),
        "eig_mode_proj": _wavg([r["eig_mode_proj"] for r in runs], w),
        "mode_var": _wavg([r["mode_var"] for r in runs], w),
        "mode_var_frac": _wavg([r["mode_var_frac"] for r in runs], w),
        "mean": _wavg([r["mean"] for r in runs], w),
        "std": _wavg([r["std"] for r in runs], w),
        "band_fracs": {k: _wavg([r["band_fracs"][k] for r in runs], w) for k in BAND_NAMES},
        "band_fracs_mode": {
            k: _wavg([r["band_fracs_mode"][k] for r in runs], w) for k in BAND_NAMES
        },
        "increments": {
            k: {
                "kurtosis_excess": _wavg([r["increments"][k]["kurtosis_excess"] for r in runs], w),
                "std": _wavg([r["increments"][k]["std"] for r in runs], w),
            }
            for k in runs[0]["increments"]
        },
        "hump_rotor_mean": _hump(f, psd.mean(axis=0), fmax_hump),
        "hump_per_rotor": [_hump(f, p, fmax_hump) for p in psd],
    }
    out["band_fracs_rotor_mean"] = {k: float(v.mean()) for k, v in out["band_fracs"].items()}
    out["coherence_bands"] = {
        name: {
            bn: float(np.mean(coh[k][(f >= lo) & (f < hi)]))
            for bn, (lo, hi) in zip(BAND_NAMES[:3], BANDS[:3])
        }
        for k, name in enumerate(PAIR_NAMES)
    }
    out["coherency_phase_deg"] = np.degrees(np.arctan2(out["coherency_im"], out["coherency_re"]))
    out["coherency_phase_bands_deg"] = {
        name: {
            bn: float(
                np.degrees(
                    np.arctan2(
                        np.mean(out["coherency_im"][k][(f >= lo) & (f < hi)]),
                        np.mean(out["coherency_re"][k][(f >= lo) & (f < hi)]),
                    )
                )
            )
            for bn, (lo, hi) in zip(BAND_NAMES[:3], BANDS[:3])
        }
        for k, name in enumerate(PAIR_NAMES)
    }
    return out


# ─── Michael's-specific quantisation / aliasing check ─────────────────────────


def michaels_quantisation(flights: list[Flight]) -> dict[str, Any]:
    """Is the 29.41 Hz DatCon telemetry itself quantised / aliased?

    Run on the NATIVE grid (100 Hz interpolation would hide everything above
    14.7 Hz): uniform 29.41 Hz resample of each airborne run, Welch PSD, and
    the 1 RPM quantisation floor ``q^2/12`` spread over 0-Nyquist.
    """
    q = 1.0 / 60.0  # 1 RPM in rev/s
    out: dict[str, Any] = {
        "quant_step_rps": q,
        "quant_var_rps2": q**2 / 12,
        "per_flight": {},
        "psd_curves": {},
    }
    for fl in flights:
        rate = fl.native_rate
        nyq = rate / 2
        grid, _ = to_grid(fl)
        runs, _thr = airborne_runs(grid)
        # native-rate uniform grid, airborne windows mapped back by time
        t_nat = np.arange(0.0, fl.t[-1], 1.0 / rate)
        x_nat = _interp_rows(fl.t, fl.rps, t_nat)
        psds, weights = [], []
        f_ref = None
        for r in runs:
            lo, hi = int(r.start / FS * rate), int(r.stop / FS * rate)
            seg = x_nat[:, lo:hi]
            nper = min(int(round(NPERSEG_S * rate)), seg.shape[1])
            f, p = sg.welch(
                seg,
                fs=rate,
                window="hann",
                nperseg=nper,
                noverlap=nper // 2,
                detrend="constant",
                axis=-1,
            )
            if f_ref is None or f.size > f_ref.size:
                f_ref = f
            psds.append(p)
            weights.append(seg.shape[1] / rate)
        if f_ref is None:
            continue
        f = np.asarray(f_ref)
        psd = _wavg(
            [p for p in psds if p.shape[-1] == f.size],
            [w for p, w in zip(psds, weights) if p.shape[-1] == f.size],
        )
        rot = psd.mean(axis=0)

        def band(lo: float, hi: float, rot: np.ndarray = rot, f: np.ndarray = f) -> float:
            return float(np.mean(rot[(f >= lo) & (f < hi)]))

        # Value / increment statistics MUST come off the RAW rows: resampling
        # onto a uniform grid interpolates across held samples and destroys
        # exactly the repeat structure we are testing for.
        keep = np.isfinite(fl.rps).all(axis=0)
        inside = np.zeros(fl.t.size, dtype=bool)
        for r in runs:
            inside |= (fl.t >= r.start / FS) & (fl.t < r.stop / FS)
        x_raw = fl.rps[:, keep & inside]
        rpm_raw = np.round(x_raw / fl.extra["rps_scale"] * 60.0).astype(np.int64)
        d_air = np.diff(rpm_raw, axis=1).ravel().astype(np.float64)
        rpm = rpm_raw.ravel().astype(np.float64)
        lsb = rpm_raw.ravel() % 2
        changed = d_air[d_air != 0]
        # ALIAS TEST: a band-limited (anti-alias filtered) log must roll off
        # below the broadband trend as it approaches Nyquist.  Fit the trend on
        # 2-8 Hz and ask by how many dB the 12-Nyquist band exceeds it.
        fit = (f >= 2.0) & (f <= 8.0)
        trend = np.polyfit(np.log10(f[fit]), 10 * np.log10(rot[fit]), 1)
        near = (f >= 12.0) & (f <= nyq + 1e-9)
        excess = float(np.mean(10 * np.log10(rot[near]) - np.polyval(trend, np.log10(f[near]))))
        upper = (f >= 8.0) & (f <= nyq + 1e-9)
        slope_hi = float(np.polyfit(np.log10(f[upper]), 10 * np.log10(rot[upper]), 1)[0])
        out["per_flight"][fl.flight] = {
            "native_rate_hz": rate,
            "nyquist_hz": nyq,
            "psd_5_10hz": band(5.0, 10.0),
            "psd_12_nyq": band(12.0, nyq + 1e-9),
            "psd_near_nyq_over_5_10": band(12.0, nyq + 1e-9) / max(band(5.0, 10.0), 1e-300),
            "quant_floor_psd": (q**2 / 12) / nyq,
            "psd_near_nyq_over_quant_floor": band(12.0, nyq + 1e-9) / ((q**2 / 12) / nyq),
            "slope_2_8hz_db_per_dec": float(trend[0]),
            "slope_8_nyq_db_per_dec": slope_hi,
            "alias_excess_db_12_nyq": excess,
            "raw_rpm_is_integer": fl.extra["raw_rpm_is_integer"],
            "raw_rpm_unique_values": fl.extra["raw_rpm_unique"],
            "airborne_rpm_unique_values": int(np.unique(np.round(rpm)).size),
            "even_rpm_fraction": float(np.mean(lsb == 0)),
            "zero_increment_fraction": float(np.mean(d_air == 0)),
            "median_abs_increment_rpm": float(np.median(np.abs(d_air))),
            "median_abs_increment_when_changed_rpm": float(np.median(np.abs(changed))),
            "increment_rpm_iqr": float(np.subtract(*np.percentile(d_air, [75, 25]))),
            "increment_rpm_std": float(d_air.std()),
            "increment_kurtosis_native": float(st.kurtosis(d_air, fisher=True, bias=False)),
        }
        edges = np.arange(-250.5, 251.5, 1.0)  # integer-RPM bins, +/- 6 sigma
        counts, _ = np.histogram(d_air, bins=edges, density=True)
        out["psd_curves"][fl.flight] = {
            "freq": f,
            "psd_rotor_mean": rot,
            "incr_hist_centres_rpm": 0.5 * (edges[1:] + edges[:-1]),
            "incr_hist_density": counts,
        }
    return out


# ─── Telemetry-channel fidelity (native grid, no resampling) ──────────────────


def channel_fidelity(fl: Flight, windows: list[tuple[float, float]]) -> dict[str, Any]:
    """How good is this telemetry *channel*, before any physics is read off it.

    Measured on the NATIVE samples inside the airborne ``windows`` (on the
    ground the value is a constant, which would fake a sample-and-hold): how
    often the value actually changes (a held channel reports the log rate, not
    its update rate), the effective update rate, the hold length, and the
    amplitude quantisation step at the operating point.  DREGON's
    ``motor.measured`` fails badly here and that invalidates its
    high-frequency spectrum, so this is not optional bookkeeping.
    """
    keep = np.isfinite(fl.rps).all(axis=0)
    if windows:
        inside = np.zeros(fl.t.size, dtype=bool)
        for t0, t1 in windows:
            inside |= (fl.t >= t0) & (fl.t < t1)
        keep &= inside
    x = fl.rps[:, keep]
    t = fl.t[keep]
    dur = float(t[-1] - t[0])
    d = np.diff(x, axis=1)
    changes = np.count_nonzero(d[0] != 0)
    hold = np.diff(np.flatnonzero(d[0] != 0))
    vals = np.unique(x[0])
    step = float(np.median(np.diff(vals))) if vals.size > 2 else float("nan")
    return {
        "native_log_rate_hz": fl.native_rate,
        "airborne_samples": int(x.shape[1]),
        "zero_change_frac": float(np.mean(d == 0)),
        "update_rate_hz": changes / dur,
        "median_hold_ms": float(np.median(hold) / fl.native_rate * 1e3) if hold.size else 0.0,
        "unique_values_rotor0": int(vals.size),
        "median_value_step_rps": step,
        "rel_resolution_pct": 100.0 * step / float(np.median(x[0])),
        "usable_bandwidth_hz": min(fl.native_rate, changes / dur) / 2.0,
    }


def dregon_channel_compare() -> dict[str, Any]:
    """``motor.measured`` vs ``motor.command`` on the room1 flights, natively.

    Both channels are put on a common 1 kHz grid (no low-pass) so the residual
    ``measured - command`` can be read out to 470 Hz.  If that residual is flat
    and sits above the measured PSD's own high-frequency content, the measured
    channel's HF "dynamics" are its own staircase, not the shaft's.
    """
    out: dict[str, Any] = {"per_flight": {}, "curves": {}}
    for path in _dregon_mats():
        mat = scipy.io.loadmat(str(path))["motor"]
        if "measured" not in mat.dtype.names:
            continue
        ts = mat["timestamps"][0, 0].flatten().astype(np.float64)
        ts = ts - ts[0]
        me = np.asarray(mat["measured"][0, 0], dtype=np.float64).T[list(DREGON_TO_MIXER)]
        cm = np.asarray(mat["command"][0, 0], dtype=np.float64).T[list(DREGON_TO_MIXER)]
        grid = np.arange(0.0, ts[-1], 1.0 / DREGON_WORK_FS)
        M, C = _interp_rows(ts, me, grid), _interp_rows(ts, cm, grid)
        m = M.mean(axis=0)
        air = np.flatnonzero(np.all(THR_FRAC * np.percentile(m, THR_PCTL) < M, axis=0))
        seg = slice(int(air[0] + DREGON_WORK_FS), int(air[-1] - DREGON_WORK_FS))
        nper = int(10 * DREGON_WORK_FS)
        curves = {}
        for name, v in (("measured", M), ("command", C), ("residual", M - C)):
            f, p = sg.welch(
                v[:, seg],
                fs=DREGON_WORK_FS,
                window="hann",
                nperseg=nper,
                noverlap=nper // 2,
                detrend="constant",
                axis=-1,
            )
            curves[name] = p.mean(axis=0)
        f_ = f

        def band(
            c: str,
            lo: float,
            hi: float,
            curves: dict[str, np.ndarray] = curves,
            f_: np.ndarray = f_,
        ) -> float:
            keep = (f_ >= lo) & (f_ < hi)
            return float(np.trapezoid(curves[c][keep], f_[keep]))

        rec = path.parent.name.replace("DREGON_", "")
        out["per_flight"][rec] = {
            "mean_abs_diff_rps": float(np.abs(M[:, seg] - C[:, seg]).mean()),
            "mean_offset_rps": float((M[:, seg] - C[:, seg]).mean()),
            "var_ratio_cmd_over_meas": {
                bn: band("command", lo, hi) / max(band("measured", lo, hi), 1e-300)
                for bn, (lo, hi) in zip(BAND_NAMES, BANDS)
            },
            "residual_frac_of_measured": {
                bn: band("residual", lo, hi) / max(band("measured", lo, hi), 1e-300)
                for bn, (lo, hi) in zip(BAND_NAMES, BANDS)
            },
            "residual_psd_0.5_10hz": float(np.mean(curves["residual"][(f_ >= 0.5) & (f_ < 10.0)])),
            "residual_var_rps2": band("residual", 0.0, 470.0),
            "measured_psd_50_200hz": float(
                np.mean(curves["measured"][(f_ >= 50.0) & (f_ < 200.0)])
            ),
        }
        out["curves"][rec] = {"freq": f_, **curves}
    return out


def same_drone_check(
    per_flight: dict[str, dict[str, Any]], flights: list[Flight]
) -> dict[str, Any]:
    """Do FLY103/108 look like the same airframe as FLY124/125?

    The raw trim vector mixes two very different things, so it is also read in
    the MIXER basis: the YAW component of the trim is the airframe's own
    rotor/torque imbalance (a property of the vehicle), while roll and pitch
    trim are CG/payload placement (a property of the day).  Two logs of the
    same drone must agree on yaw trim and may disagree on roll/pitch.
    """
    ids = list(per_flight)
    mean = {k: np.asarray(per_flight[k]["rig_pooled_mean"]) for k in ids}
    std = {k: np.asarray(per_flight[k]["rig_pooled_std"]) for k in ids}
    trim = {k: mean[k] - mean[k].mean() for k in ids}
    modes = {k: (MIXER.T @ trim[k]) / 4.0 for k in ids}  # common(=0), roll, pitch, yaw
    attrs = {fl.flight: fl.extra["attrs"] for fl in flights}
    pairs = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            pairs[f"{a}|{b}"] = {
                "trim_l2_rps": float(np.linalg.norm(trim[a] - trim[b])),
                "trim_max_abs_rps": float(np.max(np.abs(trim[a] - trim[b]))),
                "trim_corr": float(np.corrcoef(trim[a], trim[b])[0, 1]),
                "yaw_trim_diff_rps": float(modes[a][3] - modes[b][3]),
                "rollpitch_trim_diff_rps": float(np.linalg.norm(modes[a][1:3] - modes[b][1:3])),
                "std_ratio": (std[a] / std[b]).tolist(),
                "std_l2_rps": float(np.linalg.norm(std[a] - std[b])),
                "mean_l2_rps": float(np.linalg.norm(mean[a] - mean[b])),
                "same_serial": attrs[a].get("mcID(SN)") == attrs[b].get("mcID(SN)"),
            }
    within_new = pairs["FLY103|FLY108"]["trim_l2_rps"]
    within_old = pairs["FLY124|FLY125"]["trim_l2_rps"]
    cross = [pairs[k]["trim_l2_rps"] for k in pairs if k not in ("FLY103|FLY108", "FLY124|FLY125")]
    yaws = np.array([modes[k][3] for k in ids])
    return {
        "attrs": attrs,
        "rotor_means": {k: mean[k].tolist() for k in ids},
        "trim_rps": {k: trim[k].tolist() for k in ids},
        "trim_modes_rps": {k: {n: float(v) for n, v in zip(MODE_NAMES, modes[k])} for k in ids},
        "std_rps": {k: std[k].tolist() for k in ids},
        "pairs": pairs,
        "trim_l2_within_new": within_new,
        "trim_l2_within_old": within_old,
        "trim_l2_cross_mean": float(np.mean(cross)),
        "yaw_trim_rps": {k: float(modes[k][3]) for k in ids},
        "yaw_trim_spread_rps": float(yaws.max() - yaws.min()),
        "yaw_trim_mean_rps": float(yaws.mean()),
        "max_rollpitch_trim_diff_rps": max(p["rollpitch_trim_diff_rps"] for p in pairs.values()),
        "serials_all_equal": len({a.get("mcID(SN)") for a in attrs.values()}) == 1,
        "serial": attrs[ids[0]].get("mcID(SN)"),
        "actypes": sorted({a.get("ACType", "?") for a in attrs.values()}),
        "datetimes": {k: attrs[k].get("dateTime") for k in ids},
        "verdict": (
            "same physical vehicle"
            if len({a.get("mcID(SN)") for a in attrs.values()}) == 1
            and float(yaws.max() - yaws.min()) < 2.0
            else "inconclusive"
        ),
    }


# ─── Plots ────────────────────────────────────────────────────────────────────

ROTOR_COLORS = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd")


def _finish(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def plot_psd(rig: str, agg: dict, overlay: dict | None, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    f = agg["freq"]
    for i, name in enumerate(ROTORS):
        ax.loglog(
            f,
            agg["psd"][i],
            color=ROTOR_COLORS[i],
            lw=1.2,
            label=f"{name} measured" if overlay else name,
        )
    if overlay is not None:
        for i, name in enumerate(ROTORS):
            ax.loglog(
                overlay["freq"],
                overlay["psd"][i],
                color=ROTOR_COLORS[i],
                lw=1.0,
                ls="--",
                alpha=0.75,
                label=f"{name} command",
            )
    for h, style in ((agg["hump_rotor_mean"], "k"), (agg["hump_rotor_mean"]["secondary"], "0.4")):
        if h is None or h["peak_hz"] is None:
            continue
        ax.axvline(h["peak_hz"], color=style, lw=0.8, ls=":")
        tag = "resonance" if h["is_resonance"] else "shoulder"
        ax.annotate(
            f"{tag} {h['peak_hz']:.2f} Hz\n+{h['prominence_db']:.1f} dB",
            xy=(h["peak_hz"], ax.get_ylim()[1]),
            xytext=(4, -38),
            textcoords="offset points",
            fontsize=7,
            color=style,
        )
    for _lo, hi in BANDS[:-1]:
        ax.axvline(hi, color="0.8", lw=0.6, zorder=0)
    if rig == "michaels":
        ax.axvline(14.7, color="crimson", lw=0.8, ls="-.", label="native Nyquist 14.7 Hz")
    ax.set_xlim(0.05, 50)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel(r"PSD $(\mathrm{rev/s})^2/\mathrm{Hz}$")
    ax.set_title(
        f"{rig}: per-rotor RPS PSD (duration-weighted, {agg['total_airborne_s']:.0f} s airborne)"
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    return _finish(fig, out / f"{rig}_psd.png")


def plot_coherence(rig: str, agg: dict, overlay: dict | None, out: Path) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.6), sharex=True, sharey=True)
    f = agg["freq"]
    for k, ax in enumerate(axes.ravel()):
        ax.semilogx(
            f,
            agg["coherence"][k],
            lw=1.1,
            color="#1f77b4",
            label="MSC measured" if overlay else "MSC",
        )
        if overlay is not None:
            ax.semilogx(
                overlay["freq"],
                overlay["coherence"][k],
                lw=1.0,
                ls="--",
                color="#ff7f0e",
                label="MSC command",
            )
        ph = ax.twinx()
        ph.semilogx(
            f, agg["coherency_phase_deg"][k], lw=0.5, color="0.55", alpha=0.7, label="phase"
        )
        ph.set_ylim(-180, 180)
        ph.set_yticks([-180, 0, 180])
        ph.tick_params(labelsize=6)
        if k % 3 != 2:
            ph.set_yticklabels([])
        diag = " (diagonal)" if k in DIAG_PAIRS else ""
        ax.set_title(f"{PAIR_NAMES[k]}{diag}", fontsize=9)
        ax.set_xlim(0.05, 50)
        ax.set_ylim(0, 1)
        ax.set_zorder(ph.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.grid(True, which="both", alpha=0.25)
        if k == 0:
            ax.legend(fontsize=6, loc="upper right")
    for ax in axes[1]:
        ax.set_xlabel("frequency (Hz)")
    for ax in axes[:, 0]:
        ax.set_ylabel("MS coherence")
    fig.suptitle(
        f"{rig}: rotor-pair magnitude-squared coherence (left axis) and "
        "cross-spectral phase in degrees (right axis, grey)",
        fontsize=10,
    )
    return _finish(fig, out / f"{rig}_coherence.png")


def plot_eigen(rig: str, agg: dict, out: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    f = agg["freq"]
    for i in range(4):
        axes[0].loglog(
            f, agg["psd_eig"][i], lw=1.2, label=f"e{i + 1} ({agg['eig_frac'][i] * 100:.1f} % var)"
        )
        axes[1].loglog(
            f,
            agg["psd_mode"][i],
            lw=1.2,
            label=f"{MODE_NAMES[i]} ({agg['mode_var_frac'][i] * 100:.1f} % var)",
        )
    for ax, ttl in zip(axes, ("covariance eigenvectors", "MIXER control modes")):
        ax.set_xlim(0.05, 50)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel(r"PSD $(\mathrm{rev/s})^2/\mathrm{Hz}$")
        ax.set_title(ttl, fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
    txt = "|e_i . mode| cosines\n      " + "  ".join(f"{m[:4]:>5s}" for m in MODE_NAMES)
    for i in range(4):
        txt += f"\ne{i + 1}:  " + "  ".join(f"{v:5.2f}" for v in agg["eig_mode_proj"][i])
    axes[0].text(
        0.02, 0.03, txt, transform=axes[0].transAxes, fontsize=6.5, family="monospace", va="bottom"
    )
    fig.suptitle(f"{rig}: eigen- and control-mode spectra", fontsize=11)
    return _finish(fig, out / f"{rig}_eigen.png")


def plot_acf(rig: str, agg: dict, overlay: dict | None, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for i, name in enumerate(ROTORS):
        ax.semilogx(
            agg["acf_lags_s"],
            agg["acf"][i],
            "o-",
            ms=3,
            color=ROTOR_COLORS[i],
            lw=1.1,
            label=f"{name} measured" if overlay else name,
        )
    if overlay is not None:
        for i, name in enumerate(ROTORS):
            ax.semilogx(
                overlay["acf_lags_s"],
                overlay["acf"][i],
                "--",
                color=ROTOR_COLORS[i],
                lw=0.9,
                alpha=0.8,
                label=f"{name} command",
            )
    ax.axhline(0, color="k", lw=0.7)
    ax.axhline(1 / np.e, color="0.6", lw=0.7, ls=":")
    ax.set_xlabel("lag (s)")
    ax.set_ylabel("autocorrelation")
    ax.set_title(f"{rig}: per-rotor RPS autocorrelation (demeaned per airborne run)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    return _finish(fig, out / f"{rig}_acf.png")


def plot_timeseries(
    rig: str, fl: Flight, grid: np.ndarray, runs: list[slice], thr: float, out: Path
) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 6.0))
    t = np.arange(grid.shape[1]) / FS
    for i, name in enumerate(ROTORS):
        axes[0].plot(t, grid[i], color=ROTOR_COLORS[i], lw=0.5, label=name)
    axes[0].axhline(thr, color="k", ls=":", lw=0.9, label=f"airborne thr {thr:.1f}")
    for r in runs:
        axes[0].axvspan(r.start / FS, r.stop / FS, color="0.85", zorder=0)
    axes[0].set_title(f"{rig} / {fl.flight}: full log (grey = airborne runs)", fontsize=10)
    axes[0].legend(fontsize=7, ncol=3)
    longest = max(runs, key=lambda r: r.stop - r.start)
    mid = (longest.start + longest.stop) // 2
    z = slice(max(longest.start, mid - int(10 * FS)), min(longest.stop, mid + int(10 * FS)))
    for i, name in enumerate(ROTORS):
        axes[1].plot(t[z], grid[i, z], color=ROTOR_COLORS[i], lw=0.9, label=name)
    axes[1].set_title(f"{rig} / {fl.flight}: 20 s inside the longest airborne run", fontsize=10)
    for ax in axes:
        ax.set_xlabel("time (s)")
        ax.set_ylabel("rev/s")
        ax.grid(alpha=0.25)
    return _finish(fig, out / f"{rig}_timeseries.png")


def plot_michaels_quant(q: dict, out: Path) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    for fid, cur in q["psd_curves"].items():
        axes[0].loglog(cur["freq"], cur["psd_rotor_mean"], lw=1.0, label=fid)
    nyq = list(q["per_flight"].values())[0]["nyquist_hz"]
    floor = list(q["per_flight"].values())[0]["quant_floor_psd"]
    fl0 = list(q["per_flight"])[0]
    p0 = q["per_flight"][fl0]
    c0 = q["psd_curves"][fl0]
    anchor = float(np.interp(2.0, np.asarray(c0["freq"]), np.asarray(c0["psd_rotor_mean"])))
    fx = np.array([2.0, nyq])
    axes[0].loglog(
        fx,
        anchor * (fx / 2.0) ** (p0["slope_2_8hz_db_per_dec"] / 10.0),
        color="k",
        ls=":",
        lw=1.1,
        label=f"{fl0} 2-8 Hz trend extrapolated",
    )
    axes[0].axhline(floor, color="k", ls="--", lw=0.9, label="1 RPM quantisation floor")
    axes[0].axvline(nyq, color="crimson", ls="-.", lw=0.9, label=f"Nyquist {nyq:.2f} Hz")
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel(r"PSD $(\mathrm{rev/s})^2/\mathrm{Hz}$")
    axes[0].set_title("native-rate (29.41 Hz) rotor-mean PSD", fontsize=10)
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=7)

    for fid, cur in q["psd_curves"].items():
        axes[1].semilogy(cur["incr_hist_centres_rpm"], cur["incr_hist_density"], lw=0.9, label=fid)
    s = q["per_flight"][fl0]["increment_rpm_std"]
    xs = np.asarray(q["psd_curves"][fl0]["incr_hist_centres_rpm"])
    axes[1].semilogy(
        xs,
        np.exp(-0.5 * (xs / s) ** 2) / (s * np.sqrt(2 * np.pi)),
        "k--",
        lw=0.9,
        label=f"Gaussian, std {s:.1f} RPM",
    )
    axes[1].set_ylim(1e-6, 1)
    axes[1].set_xlabel("consecutive-sample increment (RPM)")
    axes[1].set_ylabel("density")
    axes[1].set_title("increments: integer-quantised and heavy-tailed", fontsize=10)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7)

    ks = list(q["per_flight"])
    x = np.arange(len(ks))
    axes[2].bar(
        x - 0.2,
        [q["per_flight"][k]["zero_increment_fraction"] for k in ks],
        0.4,
        label=r"P(|$\Delta$| < 0.5 RPM) between samples",
    )
    axes[2].bar(
        x + 0.2,
        [q["per_flight"][k]["even_rpm_fraction"] for k in ks],
        0.4,
        label="fraction of even RPM values",
    )
    axes[2].axhline(0.5, color="k", lw=0.7, ls=":")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(ks)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("quantisation fingerprints (airborne, native grid)", fontsize=10)
    axes[2].legend(fontsize=7)
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("michaels: is the 29.41 Hz DatCon telemetry quantised / aliased?", fontsize=11)
    return _finish(fig, out / "michaels_quant.png")


def plot_dregon_channel(cmp_: dict, out: Path) -> str:
    """Why DREGON's ``measured`` cannot be believed above ~10 Hz."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    rec = sorted(cmp_["curves"])[0]
    cur = cmp_["curves"][rec]
    f = cur["freq"]
    for name, color in (("measured", "#1f77b4"), ("command", "#ff7f0e"), ("residual", "0.35")):
        axes[0].loglog(f, cur[name], lw=1.0, color=color, label=f"{name} (rotor mean)")
    axes[0].axvline(DREGON_LP_HZ, color="0.7", ls=":", lw=0.8)
    axes[0].set_xlim(0.1, 470)
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel(r"PSD $(\mathrm{rev/s})^2/\mathrm{Hz}$")
    axes[0].set_title(f"{rec}: native 1 kHz grid, no low-pass", fontsize=9)
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=7)

    path = DREGON_ROOT / f"DREGON_{rec}" / f"DREGON_{rec}_motors.mat"
    mat = scipy.io.loadmat(str(path))["motor"]
    ts = mat["timestamps"][0, 0].flatten().astype(np.float64)
    ts -= ts[0]
    me = np.asarray(mat["measured"][0, 0], dtype=np.float64).T[list(DREGON_TO_MIXER)]
    cm = np.asarray(mat["command"][0, 0], dtype=np.float64).T[list(DREGON_TO_MIXER)]
    mid = ts.size // 2
    sl = slice(mid, mid + int(1.0 / np.median(np.diff(ts))))
    axes[1].step(ts[sl], me[0, sl], where="post", lw=1.1, color="#1f77b4", label="measured")
    axes[1].plot(ts[sl], cm[0, sl], lw=0.9, color="#ff7f0e", label="command")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("rev/s")
    axes[1].set_title("1 s of rotor RFront: measured is a staircase", fontsize=9)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    upd = np.count_nonzero(np.diff(me[0]) != 0) / (ts[-1] - ts[0])
    stp = np.median(np.diff(np.unique(me[0][me[0] > 10.0])))
    fig.suptitle(
        f"dregon: motor.measured is a ~{upd:.0f} Hz sample-and-hold, quantised to ~{stp:.2f} rev/s",
        fontsize=11,
    )
    return _finish(fig, out / "dregon_channel_fidelity.png")


# ─── JSON ─────────────────────────────────────────────────────────────────────

#: Heavy / redundant arrays kept out of ``summary.json``: the full 4x4 |CSD|
#: (recoverable from ``psd`` + ``coherence``), the raw coherency parts (kept as
#: ``coherency_phase_deg``), the eigenvectors and the covariance (kept as
#: ``corr`` + ``std`` + ``eigvals``).
_DROP = {"csd_abs", "eigvecs", "cov", "coherency_re", "coherency_im"}


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items() if k not in _DROP}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else float(f"{v:.6g}")
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


# ─── Findings ─────────────────────────────────────────────────────────────────


def _fmt_bands(agg: dict) -> str:
    return " / ".join(f"{agg['band_fracs_rotor_mean'][b] * 100:.1f}%" for b in BAND_NAMES)


def _hump_str(h: dict) -> str:
    if h["peak_hz"] is None:
        return "no local maximum in band"
    kind = "RESONANCE" if h["is_resonance"] else "broad shoulder"
    s = f"{kind} at {h['peak_hz']:.2f} Hz, +{h['prominence_db']:.1f} dB"
    if not h["broad"]:
        s += f", -3 dB width {h['width_lo_hz']:.2f}-{h['width_hi_hz']:.2f} Hz (Q ~ {h['q']:.1f})"
    else:
        s += " (no -3 dB crossing inside the band)"
    sec = h.get("secondary")
    if sec and sec["peak_hz"] is not None:
        s += f"; second peak {sec['peak_hz']:.2f} Hz +{sec['prominence_db']:.1f} dB"
    return s


def _tau_1e(agg: dict) -> float:
    """Lag where the rotor-mean ACF first drops below 1/e (linear in log-lag)."""
    lags, a = agg["acf_lags_s"], agg["acf"].mean(axis=0)
    below = np.flatnonzero(a < 1 / np.e)
    if below.size == 0:
        return float("nan")
    k = int(below[0])
    if k == 0:
        return float(lags[0])
    x0, x1, y0, y1 = np.log(lags[k - 1]), np.log(lags[k]), a[k - 1], a[k]
    return float(np.exp(x0 + (1 / np.e - y0) * (x1 - x0) / (y1 - y0)))


def write_findings(
    rigs: dict[str, dict],
    per_flight: dict,
    quant: dict,
    same: dict,
    fidelity: dict,
    chan: dict,
    path: Path,
) -> None:
    mi, d1m, d1c, d2c = (
        rigs["michaels"],
        rigs["dregon_room1_measured"],
        rigs["dregon_room1_command"],
        rigs["dregon_room2_command"],
    )
    primary = (
        ("michaels", mi),
        ("dregon_room1_measured", d1m),
        ("dregon_room1_command", d1c),
        ("dregon_room2_command", d2c),
    )
    L: list[str] = []
    A = L.append
    A("# RPS trajectory exploration — what the telemetry actually says")
    A("")
    A(
        f"Frozen pipeline: native -> {FS:.0f} Hz linear interp (DREGON low-passed zero-phase "
        f"Butterworth {DREGON_LP_ORDER}/{DREGON_LP_HZ:.0f} Hz on a {DREGON_WORK_FS:.0f} Hz grid "
        f"first); airborne = all rotors > {THR_FRAC} x p{THR_PCTL:.0f}(rotor mean), eroded "
        f"{ERODE_S:.0f} s/end, runs >= {MIN_RUN_S:.0f} s; Welch {NPERSEG_S:.0f} s Hann 50 %, "
        "demeaned per rotor per run; duration-weighted pooling per rig. Rotor axis order is the "
        "MIXER order [RFront, LFront, LBack, RBack] (DREGON motor rows reindexed (2,3,0,1) from "
        "`coordinates.mat['rotorsPos']`)."
    )
    A("")
    A(
        f"Corpus: michaels {mi['n_runs']} runs / {mi['total_airborne_s']:.0f} s airborne "
        "(FLY124, FLY125, FLY103, FLY108; full flight logs, not the audio windows); "
        f"dregon room1 measured {d1m['n_runs']} runs / {d1m['total_airborne_s']:.0f} s; "
        f"dregon room1 command {d1c['n_runs']} runs / {d1c['total_airborne_s']:.0f} s (same flights); "
        f"dregon room2 command {d2c['n_runs']} runs / {d2c['total_airborne_s']:.0f} s. "
        "Every flight yields exactly one airborne run under the frozen rule."
    )
    A("")
    A("## Findings")
    A("")

    # -- 1-3: channel fidelity first; it gates every spectral claim ------------
    fm = fidelity["michaels/FLY124"]
    fd = fidelity["dregon_room1_measured/free-flight_nosource_room1"]
    fc = fidelity["dregon_room1_command/free-flight_nosource_room1"]
    A(
        f"- **DREGON `motor.measured` is a ~{fd['update_rate_hz']:.0f} Hz sample-and-hold, not a "
        f"{fd['native_log_rate_hz']:.0f} Hz measurement.** {fd['zero_change_frac'] * 100:.1f} % of "
        f"consecutive native samples are bit-identical, the value changes "
        f"{fd['update_rate_hz']:.0f} times/s (median hold {fd['median_hold_ms']:.0f} ms) and takes "
        f"only {fd['unique_values_rotor0']} distinct values per flight, spaced "
        f"{fd['median_value_step_rps']:.2f} rev/s ({fd['rel_resolution_pct']:.2f} % of the operating "
        f"point). Usable bandwidth is ~{fd['usable_bandwidth_hz']:.0f} Hz, not 464 Hz. "
        f"`motor.command` in the same file updates at {fc['update_rate_hz']:.0f} Hz with a "
        f"{fc['median_value_step_rps']:.2e} rev/s step."
    )
    cf = chan["per_flight"][sorted(chan["per_flight"])[0]]
    A(
        f"- **So DREGON's measured high-frequency 'dynamics' are its own staircase.** On the same "
        f"room1 flights the `measured - command` residual is essentially flat at "
        f"{cf['residual_psd_0.5_10hz']:.2e} (rev/s)^2/Hz over 0.5-10 Hz and accounts for "
        + ", ".join(f"{b} {cf['residual_frac_of_measured'][b] * 100:.0f} %" for b in BAND_NAMES)
        + " of the measured band variance (bands <0.5 / 0.5-5 / 5-15 / >15 Hz; over 100 % means "
        "measured and command are incoherent there, so the difference carries both). The measured "
        f"PSD collapses to {cf['measured_psd_50_200hz']:.1e} (rev/s)^2/Hz above 50 Hz, i.e. right "
        f"where a {fd['update_rate_hz']:.0f} Hz staircase must die. Treat DREGON measured as "
        "trustworthy below ~5 Hz only (`dregon_channel_fidelity.png`)."
    )
    A(
        f"- **Michael's telemetry has the opposite defect: fine in amplitude, coarse in time.** "
        f"Every logged value is an integer RPM (1 RPM = {quant['quant_step_rps'] * 1000:.1f} mrev/s, "
        f"{100 * quant['quant_step_rps'] / 80:.2f} % of the operating point — 20x finer than "
        f"DREGON's {fd['median_value_step_rps']:.2f} rev/s), but the log rate is only "
        f"{fm['native_log_rate_hz']:.2f} Hz (Nyquist "
        f"{quant['per_flight']['FLY124']['nyquist_hz']:.2f} Hz) and it is partly held too: "
        + ", ".join(
            f"{k} P(delta=0)={v['zero_increment_fraction']:.2f}"
            for k, v in quant["per_flight"].items()
        )
        + f", median hold {fm['median_hold_ms']:.0f} ms, effective update "
        + "/".join(
            f"{fidelity[f'michaels/{k}']['update_rate_hz']:.0f}" for k in quant["per_flight"]
        )
        + f" Hz. Conditional on changing, the median jump is "
        f"{quant['per_flight']['FLY124']['median_abs_increment_when_changed_rpm']:.0f} RPM against "
        f"a {quant['per_flight']['FLY124']['median_abs_increment_rpm']:.0f} RPM unconditional "
        "median — a bimodal 'hold or jump' pattern, i.e. an asynchronous ESC update resampled "
        f"into the log. Usable bandwidth ~{fm['usable_bandwidth_hz']:.0f} Hz, not 14.7 Hz."
    )
    A("")

    # -- 4-6: resonance -------------------------------------------------------
    A(
        "- **Is there a resonant hump, and where?** (robust cubic baseline in log-f; `RESONANCE` "
        "= at least 3 dB prominence with both -3 dB crossings inside the band):"
    )
    for rig, agg in primary:
        A(
            f"  - `{rig}`: {_hump_str(agg['hump_rotor_mean'])}; per-rotor primary peaks "
            f"{[None if h['peak_hz'] is None else round(h['peak_hz'], 2) for h in agg['hump_per_rotor']]} Hz "
            f"with prominences "
            f"{[round(h['prominence_db'], 1) for h in agg['hump_per_rotor']]} dB; broadband log-log "
            f"slope {agg['hump_rotor_mean']['loglog_slope_db_per_decade']:.1f} dB/decade."
        )
    A(
        f"- **Read of the above: the only clean resonance in the corpus is in the DREGON *command* "
        f"channel** — room2 command peaks at {d2c['hump_rotor_mean']['peak_hz']:.1f} Hz "
        f"(+{d2c['hump_rotor_mean']['prominence_db']:.1f} dB, "
        f"Q ~ {d2c['hump_rotor_mean']['q']:.1f}) with a second at "
        f"{d2c['hump_rotor_mean']['secondary']['peak_hz']:.1f} Hz, and room1 command at "
        f"{d1c['hump_rotor_mean']['peak_hz']:.1f} Hz "
        f"(+{d1c['hump_rotor_mean']['prominence_db']:.1f} dB) — the MikroKopter attitude loop. "
        f"Michael's ESC feedback shows NO hump worth the name (best "
        f"+{mi['hump_rotor_mean']['prominence_db']:.1f} dB, no -3 dB crossing): a DJI M100's rotor "
        "speed is a nearly featureless power law. `dregon_room1_measured`'s "
        f"{d1m['hump_rotor_mean']['secondary']['peak_hz']:.0f} Hz secondary bump is the staircase "
        "of the previous bullets, not a mode (it is absent from `command` on the same flights). "
        "A model must therefore make the resonance a per-rig parameter that is allowed to vanish, "
        "not a fixed feature."
    )
    A("")

    # -- 7: band table --------------------------------------------------------
    A(
        "- **Variance by band** (fraction of the 0-50 Hz PSD integral, rotor-mean; columns "
        "<0.5 / 0.5-5 / 5-15 / >15 Hz):"
    )
    for rig, agg in primary:
        A(f"  - `{rig}`: {_fmt_bands(agg)}")
    A(
        f"  - Michael's two right-hand columns lie above its native Nyquist "
        f"({quant['per_flight']['FLY124']['nyquist_hz']:.2f} Hz) and are interpolation residue; "
        "DREGON measured's two right-hand columns are the staircase. The only columns comparable "
        "across rigs are <0.5 Hz and 0.5-5 Hz, and there all four series agree within "
        f"{100 * max(abs(a['band_fracs_rotor_mean']['lt0.5'] - b['band_fracs_rotor_mean']['lt0.5']) for _, a in primary for _, b in primary):.0f} "
        "pp: **~40-50 % of the variance is below 0.5 Hz and ~36-40 % in 0.5-5 Hz**."
    )
    A("")

    # -- 8-10: modes ----------------------------------------------------------
    A("- **Which control mode dominates, and how spread are the eigenvalues?**")
    for rig, agg in primary:
        dom = int(np.argmax(agg["mode_var_frac"]))
        fr, ev = agg["mode_var_frac"], agg["eig_frac"]
        A(
            f"  - `{rig}`: {MODE_NAMES[dom]} dominates "
            f"({', '.join(f'{m}={v * 100:.1f}%' for m, v in zip(MODE_NAMES, fr))} of rotor "
            f"variance); covariance eigenvalue fractions "
            f"{', '.join(f'{v * 100:.1f}%' for v in ev)}, spread "
            f"lambda1/lambda4 = {agg['eigvals'][0] / max(agg['eigvals'][3], 1e-12):.1f}x; e1 "
            f"projects on the modes with cosines "
            f"{', '.join(f'{m}={c:.2f}' for m, c in zip(MODE_NAMES, agg['eig_mode_proj'][0]))}."
        )
    A(
        "- **Read: the collective/yaw pair carries almost everything, roll and pitch almost "
        "nothing.** collective+yaw = "
        + ", ".join(
            f"{rig.replace('dregon_', '')} {(agg['mode_var_frac'][0] + agg['mode_var_frac'][3]) * 100:.0f} %"
            for rig, agg in primary
        )
        + ". The eigenbasis is NOT the mixer basis: the leading eigenvector is a "
        "collective/yaw mixture (cosines above), because a quad trims yaw with the same two "
        "diagonal rotors it uses for thrust. A 4-rotor model can be written as 2 loud + 2 quiet "
        "modes, but the 2 loud ones must be allowed to rotate within the collective-yaw plane."
    )
    A(
        "- **The four modes do NOT share a spectral shape — each needs its own.** Fraction of "
        "each mode's OWN variance below 0.5 Hz (common / roll / pitch / yaw): "
        + "; ".join(
            f"{rig.replace('dregon_', '')} "
            + "/".join(f"{v * 100:.0f}%" for v in agg["band_fracs_mode"]["lt0.5"])
            for rig, agg in primary
        )
        + ". On michaels the yaw mode is the reddest (it owns the sub-0.4 Hz band and then falls "
        "away) while the collective mode dominates from ~0.7 Hz up; roll and pitch sit ~10 dB "
        "below collective across the whole band (`*_eigen.png`, right panel). A model that scales "
        "one shared spectrum per mode cannot fit this; the mode spectra need independent "
        "time constants."
    )
    A("")

    # -- 11-12: coupling ------------------------------------------------------
    def coh_mean(agg: dict, band: str, pairs=PAIR_NAMES) -> float:
        return float(np.mean([agg["coherence_bands"][p][band] for p in pairs]))

    diag = tuple(PAIR_NAMES[i] for i in DIAG_PAIRS)
    A(
        "- **Rotor coupling is real but band- and channel-dependent** — mean magnitude-squared "
        "coherence over the six pairs, <0.5 Hz -> 0.5-5 Hz -> 5-15 Hz: "
        + "; ".join(
            f"{rig.replace('dregon_', '')} {coh_mean(a, 'lt0.5'):.2f} -> "
            f"{coh_mean(a, '0.5-5'):.2f} -> {coh_mean(a, '5-15'):.2f}"
            for rig, a in primary
        )
        + ". Michael's and room1 lose coherence with frequency (the HF there is per-rotor "
        "measurement noise), whereas room2 command GAINS it "
        f"({coh_mean(d2c, 'lt0.5'):.2f} -> {coh_mean(d2c, '5-15'):.2f}) because its 3 Hz / 6.7 Hz "
        "attitude-loop resonance is common-mode across all four rotors. So 'independent above "
        "5 Hz' is a property of the measurement chain, not of the vehicle; what is robust is that "
        "every series is coupled below 1 Hz, so independent per-rotor noise is wrong there."
    )
    A(
        "- **The coupling is diagonal-pair coupling** — zero-lag correlation, "
        "diagonal pairs (same rotation sense: "
        + ", ".join(diag)
        + ") vs adjacent: "
        + "; ".join(
            f"{rig.replace('dregon_', '')} {np.mean([a['corr'][0, 2], a['corr'][1, 3]]):.2f} vs "
            f"{np.mean([a['corr'][0, 1], a['corr'][0, 3], a['corr'][1, 2], a['corr'][2, 3]]):.2f}"
            for rig, a in primary
        )
        + f" (mean off-diagonal {', '.join(f'{np.mean(a["corr"][np.triu_indices(4, 1)]):.2f}' for _, a in primary)}). "
        "That is the signature of the yaw degree of freedom, and it means the 4x4 covariance needs "
        "exactly two free off-diagonals, not one. The cross-spectral PHASE says the coupling is "
        "instantaneous, not a lag: below 0.5 Hz the six coherency phases are "
        + ", ".join(
            f"{rig.replace('dregon_', '')} "
            f"[{', '.join(f'{a["coherency_phase_bands_deg"][p]["lt0.5"]:.0f}' for p in PAIR_NAMES)}]"
            for rig, a in primary
        )
        + " degrees (pair order "
        + ", ".join(PAIR_NAMES)
        + "). The diagonal pairs sit at ~0 deg "
        "everywhere; the adjacent pairs sit near +/-170 deg on michaels and room2 command "
        "(anti-phase = the yaw mode) but near 0 deg on room1, where the collective mode dominates. "
        "Either way the phase is 0 or 180 deg and never intermediate, so the coupling is "
        "instantaneous: a static 4x4 mixing matrix over independent mode processes suffices and no "
        "inter-rotor lag is needed."
    )
    A("")

    # -- 13: ACF --------------------------------------------------------------
    A(
        "- **The ACF has two time scales, so no single exponential (AR(1)) can fit it** — "
        "rotor-mean autocorrelation on the frozen log lag grid:"
    )
    for rig, agg in primary:
        lags, a = agg["acf_lags_s"], agg["acf"].mean(axis=0)
        A(
            f"  - `{rig}`: {a[0]:.3f} at {lags[0]:.2f} s, "
            f"{np.interp(0.1, lags, a):.2f} at 0.1 s, {np.interp(1.0, lags, a):.2f} at 1 s, "
            f"{np.interp(10.0, lags, a):.3f} at 10 s; 1/e crossing {_tau_1e(agg):.2f} s."
        )
    A("")

    # -- 14-15: increments ----------------------------------------------------
    A(
        "- **Increments are heavy-tailed at both lags** — excess kurtosis of the 0.01 s and 0.5 s "
        "differences (Gaussian = 0):"
    )
    for rig, agg in primary:
        k1 = agg["increments"]["0.01s"]["kurtosis_excess"]
        k2 = agg["increments"]["0.5s"]["kurtosis_excess"]
        A(
            f"  - `{rig}`: {np.mean(k1):.1f} at 0.01 s (per rotor {np.round(k1, 1).tolist()}), "
            f"{np.mean(k2):.1f} at 0.5 s (per rotor {np.round(k2, 1).tolist()})."
        )
    kn = [quant["per_flight"][k]["increment_kurtosis_native"] for k in quant["per_flight"]]
    A(
        f"- **Read: heavy tails survive every channel caveat.** For michaels the 0.01 s figure is "
        "really the interpolated native 0.034 s increment, and the raw-sample kurtosis "
        f"({np.round(kn, 1).tolist()} for FLY124/125/103/108) is inflated by the hold spike at "
        "delta = 0, so the cleanest michaels number is the 0.5 s lag: excess kurtosis "
        f"{np.mean(mi['increments']['0.5s']['kurtosis_excess']):.1f}, still far from Gaussian. "
        f"DREGON is unambiguous at both lags ({np.mean(d1m['increments']['0.5s']['kurtosis_excess']):.0f} "
        f"and {np.mean(d2c['increments']['0.01s']['kurtosis_excess']):.0f}). Every series needs a "
        "heavy-tailed (or stochastic-volatility) innovation; Gaussian increments are excluded at "
        "every lag we can measure."
    )
    A("")

    # -- 16-17: command vs measured ------------------------------------------
    A(
        f"- **DREGON command vs measured on the same room1 flights** — they agree where both are "
        f"believable and diverge where only one is: airborne means agree to "
        f"{np.max(np.abs(d1m['mean'] - d1c['mean'])):.2f} rev/s, band fractions measured "
        f"{_fmt_bands(d1m)} vs command {_fmt_bands(d1c)}, and the command/measured variance ratio "
        "per band is "
        + ", ".join(f"{b} {cf['var_ratio_cmd_over_meas'][b]:.2f}" for b in BAND_NAMES)
        + ". Below 0.5 Hz the two channels are the same signal (ratio "
        f"{cf['var_ratio_cmd_over_meas']['lt0.5']:.2f}); above 5 Hz the measured channel has "
        f"{1 / max(cf['var_ratio_cmd_over_meas']['5-15'], 1e-9):.0f}x more power, which the "
        "residual analysis attributes to the staircase, not the shaft."
    )
    A(
        "- **Consequence for fitting: use command below 5 Hz and do not claim anything above it.** "
        "Neither DREGON channel measures real >5 Hz shaft jitter (command is a setpoint, measured "
        "is a staircase) and Michael's band above ~5 Hz is contaminated by folding and "
        "hold-resampling (see the aliasing bullet below). "
        "The campaign's discrepancy statistics should therefore be weighted to 0.02-5 Hz, where "
        "all four series are self-consistent."
    )
    A("")

    # -- 18: aliasing ---------------------------------------------------------
    q = quant["per_flight"]
    A(
        f"- **Michael's 29.41 Hz telemetry is quantised AND aliased.** Quantisation: 1 RPM steps, "
        f"q^2/12 = {quant['quant_var_rps2']:.3g} (rev/s)^2 white, i.e. "
        f"{q['FLY124']['quant_floor_psd']:.2e} (rev/s)^2/Hz — but the measured 12-14.7 Hz band sits "
        + "/".join(f"{q[k]['psd_near_nyq_over_quant_floor']:.0f}" for k in q)
        + "x above that floor, so the near-Nyquist content is NOT quantisation noise. Aliasing: "
        "extrapolating the 2-8 Hz power-law trend to 12-14.7 Hz, the observed band is "
        + ", ".join(f"{k} {q[k]['alias_excess_db_12_nyq']:+.1f} dB" for k in q)
        + f" off the trend (slopes {', '.join(f'{q[k]["slope_2_8hz_db_per_dec"]:.0f}' for k in q)} "
        f"dB/dec on 2-8 Hz vs {', '.join(f'{q[k]["slope_8_nyq_db_per_dec"]:.0f}' for k in q)} "
        "dB/dec on 8-14.7 Hz). A properly anti-alias-filtered log must roll off steeply into "
        "Nyquist; these instead continue the trend and two of four flatten. Two mechanisms feed "
        "that tail and both are measurement-chain artifacts: folding of genuine >14.7 Hz shaft "
        f"motion, and the hold-and-resample of the ~{fm['update_rate_hz']:.0f} Hz asynchronous ESC "
        "update into the 29.41 Hz log (the bimodal increments of bullet 3). Either way the band "
        "above ~5 Hz is not physical spectrum. See `michaels_quant.png`, left panel: the observed "
        "PSD tracks the extrapolated 2-8 Hz trend all the way to the Nyquist line."
    )
    A(
        "  - Nor is that tail quantisation noise: the 1 RPM floor is "
        f"{q['FLY124']['quant_floor_psd']:.2e} (rev/s)^2/Hz and the even-RPM fraction is "
        + "/".join(f"{q[k]['even_rpm_fraction']:.2f}" for k in q)
        + " (0.5 = the LSB is fully exercised, so the effective step really is 1 RPM, not 2). The "
        "increment distribution is strongly non-Gaussian in the same panel: an exponential-ish "
        f"core plus tails out to +/-250 RPM against a matched-std Gaussian of "
        f"{q['FLY124']['increment_rpm_std']:.0f} RPM."
    )
    A("")

    # -- 19-20: same drone ----------------------------------------------------
    A(
        f"- **FLY103/FLY108 are the same physical drone as FLY124/FLY125 — proven, not inferred.** "
        f"All four DatCon headers carry the same flight-controller serial "
        f"`mcID(SN)|{same['serial']}`, the same `ACType|{'/'.join(same['actypes'])}`, the same "
        f"`mcVer|{same['attrs']['FLY124'].get('mcVer')}` and the same firmware date "
        f"(`{same['attrs']['FLY124'].get('Firmware Date')}`). Log dates: "
        + ", ".join(f"{k}={v}" for k, v in same["datetimes"].items() if v)
        + "."
    )
    A(
        "- **The rotor-speed statistics back that up once the trim is read in the mixer basis.** "
        "Rotor means (rev/s) "
        + "; ".join(f"{k} {np.round(v, 1).tolist()}" for k, v in same["rotor_means"].items())
        + "; trim = mean - overall mean "
        + "; ".join(f"{k} {np.round(v, 2).tolist()}" for k, v in same["trim_rps"].items())
        + f". Raw trim L2: FLY103-FLY108 {same['trim_l2_within_new']:.2f}, FLY124-FLY125 "
        f"{same['trim_l2_within_old']:.2f}, cross-pair mean {same['trim_l2_cross_mean']:.2f} rev/s "
        "— which looks like a mismatch until it is decomposed: the **yaw** trim (the airframe's own "
        "torque imbalance) is "
        + ", ".join(f"{k} {v:.2f}" for k, v in same["yaw_trim_rps"].items())
        + f" rev/s, a spread of only {same['yaw_trim_spread_rps']:.2f} rev/s about "
        f"{same['yaw_trim_mean_rps']:.2f}, while roll/pitch trim moves by up to "
        f"{same['max_rollpitch_trim_diff_rps']:.2f} rev/s — i.e. same vehicle, different CG/payload "
        f"and different operating point (overall mean "
        + ", ".join(f"{k} {np.mean(v):.1f}" for k, v in same["rotor_means"].items())
        + f" rev/s). Per-rotor std agrees to {max(p['std_l2_rps'] for p in same['pairs'].values()):.2f} "
        f"rev/s L2 across all six pairs. Verdict: **{same['verdict']}**."
    )
    A("")

    # -- closing --------------------------------------------------------------
    lows = [a["band_fracs_rotor_mean"]["lt0.5"] for _, a in primary]
    A(
        "- **What the model must reproduce** (targets for the frozen discrepancy statistics): "
        f"(i) {min(lows) * 100:.0f}-{max(lows) * 100:.0f} % of variance below 0.5 Hz with a "
        "power-law shoulder, not a flat spectrum; (ii) a collective+yaw-dominated 4x4 covariance "
        f"with lambda1/lambda4 ~ {np.mean([a['eigvals'][0] / a['eigvals'][3] for _, a in primary]):.0f} "
        "and diagonal-pair correlation well above adjacent-pair; (iii) a two-time-scale ACF "
        f"(1/e at {_tau_1e(mi):.1f}-{_tau_1e(d2c):.1f} s with a 1-10 s tail); (iv) heavy-tailed "
        "increments at every lag; (v) an OPTIONAL per-rig resonance (0 dB for the M100, "
        f"+{d2c['hump_rotor_mean']['prominence_db']:.0f} dB at "
        f"{d2c['hump_rotor_mean']['peak_hz']:.1f} Hz for the MikroKopter). A Gaussian per-rotor "
        "AR(1) reproduces none of (i)-(v)."
    )
    A("")
    path.write_text("\n".join(L) + "\n")


# ─── Driver ───────────────────────────────────────────────────────────────────


def check_raw_trees() -> None:
    """Fail early and name the ``dload pull`` that fixes it."""
    missing = {
        name: paths
        for name, paths in (
            ("recording_with_motor_speed", [MICHAELS_FILES["FLY124"], MICHAELS_FILES["FLY125"]]),
            ("new-drone-noises", [MICHAELS_FILES["FLY103"], MICHAELS_FILES["FLY108"]]),
            ("DREGON", [str(DREGON_ROOT)]),
        )
        if not all(Path(p).exists() for p in paths)
    }
    if missing:
        raise SystemExit(
            "missing raw tree(s): "
            + "; ".join(f"{k} ({', '.join(v)})" for k, v in missing.items())
            + "\nrun: "
            + " && ".join(f"dload pull {k}" for k in missing)
        )
    if not _dregon_mats():
        raise SystemExit(
            f"no *_motors.mat under {DREGON_ROOT}/DREGON_*_room*/ — run: dload pull DREGON"
        )


def main() -> None:
    check_raw_trees()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    flights = list(load_michaels()) + list(load_dregon())
    mich = [f for f in flights if f.rig == "michaels"]

    per_flight: dict[str, dict[str, Any]] = {}
    rig_runs: dict[str, list[dict]] = {}
    rig_example: dict[str, tuple[Flight, np.ndarray, list[slice], float]] = {}
    windows: dict[str, list[tuple[float, float]]] = {}

    for fl in flights:
        grid, info = to_grid(fl)
        runs, thr = airborne_runs(grid)
        fmax = 12.0 if fl.rig == "michaels" else 30.0
        res = [analyse_run(grid[:, r], fmax) for r in runs]
        key = f"{fl.rig}/{fl.flight}"
        windows[key] = [(r.start / FS, r.stop / FS) for r in runs]
        if not res:
            per_flight[key] = {"grid": info, "airborne_threshold_rps": thr, "n_runs": 0}
            continue
        pooled = pool(res, fmax)
        per_flight[key] = {
            "rig": fl.rig,
            "flight": fl.flight,
            "signal": fl.signal,
            "source_file": fl.source_file,
            "grid": info,
            "airborne_threshold_rps": thr,
            "runs": [
                {
                    "t_start_s": r.start / FS,
                    "duration_s": (r.stop - r.start) / FS,
                    "nperseg": a["nperseg"],
                    "mean": a["mean"],
                    "std": a["std"],
                    "band_fracs": a["band_fracs"],
                    "corr": a["corr"],
                    "eigvals": a["eigvals"],
                    "eig_frac": a["eig_frac"],
                    "eig_mode_proj": a["eig_mode_proj"],
                    "mode_var_frac": a["mode_var_frac"],
                    "increments": a["increments"],
                    "hump_rotor_mean": a["hump_rotor_mean"],
                }
                for r, a in zip(runs, res)
            ],
            "rig_pooled_mean": pooled["mean"],
            "rig_pooled_std": pooled["std"],
            "n_runs": len(res),
            "total_airborne_s": pooled["total_airborne_s"],
            "band_fracs_rotor_mean": pooled["band_fracs_rotor_mean"],
            "hump_rotor_mean": pooled["hump_rotor_mean"],
            "corr": pooled["corr"],
            "eig_frac": pooled["eig_frac"],
            "mode_var_frac": pooled["mode_var_frac"],
            "coherence_bands": pooled["coherence_bands"],
            "acf_lags_s": pooled["acf_lags_s"],
            "acf": pooled["acf"],
            "increments": pooled["increments"],
        }
        rig_runs.setdefault(fl.rig, []).extend(res)
        if fl.rig not in rig_example:
            rig_example[fl.rig] = (fl, grid, runs, thr)
        print(
            f"  {key:44s} {len(res)} run(s) {pooled['total_airborne_s']:7.1f} s "
            f"mean={np.round(pooled['mean'], 1)} std={np.round(pooled['std'], 2)}"
        )

    rigs = {r: pool(v, 12.0 if r == "michaels" else 30.0) for r, v in rig_runs.items()}
    fidelity = {
        f"{fl.rig}/{fl.flight}": channel_fidelity(fl, windows[f"{fl.rig}/{fl.flight}"])
        for fl in flights
    }
    quant = michaels_quantisation(mich)
    chan = dregon_channel_compare()
    same = same_drone_check(
        {k.split("/")[1]: v for k, v in per_flight.items() if v.get("rig") == "michaels"}, mich
    )

    pngs = []
    for rig in ("michaels", "dregon_room1_measured", "dregon_room2_command"):
        agg = rigs[rig]
        ov = rigs["dregon_room1_command"] if rig == "dregon_room1_measured" else None
        pngs += [
            plot_psd(rig, agg, ov, OUT_DIR),
            plot_coherence(rig, agg, ov, OUT_DIR),
            plot_eigen(rig, agg, OUT_DIR),
            plot_acf(rig, agg, ov, OUT_DIR),
        ]
        ex_rig = "dregon_room1_measured" if rig == "dregon_room1_measured" else rig
        fl, grid, runs, thr = rig_example[ex_rig]
        pngs.append(plot_timeseries(rig, fl, grid, runs, thr, OUT_DIR))
    pngs.append(plot_eigen("dregon_room1_command", rigs["dregon_room1_command"], OUT_DIR))
    pngs.append(plot_michaels_quant(quant, OUT_DIR))
    pngs.append(plot_dregon_channel(chan, OUT_DIR))

    summary = {
        "config": {
            "fs_hz": FS,
            "nperseg_s": NPERSEG_S,
            "erode_s": ERODE_S,
            "min_run_s": MIN_RUN_S,
            "thr_pctl": THR_PCTL,
            "thr_frac": THR_FRAC,
            "bands_hz": BANDS,
            "band_names": BAND_NAMES,
            "rotor_order": ROTORS,
            "pair_names": PAIR_NAMES,
            "mode_names": MODE_NAMES,
            "dregon_lp_hz": DREGON_LP_HZ,
            "dregon_to_mixer_perm": DREGON_TO_MIXER,
            "dregon_native_order": DREGON_NATIVE_ORDER,
            "acf_lags_s_nominal": ACF_LAGS_S,
            "incr_lags_s": INCR_LAGS_S,
        },
        "per_flight": per_flight,
        "per_rig": rigs,
        "channel_fidelity": fidelity,
        "michaels_quantisation": quant,
        "dregon_channel_compare": chan,
        "michaels_same_drone": same,
        "pngs": pngs,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(jsonable(summary), indent=1))
    write_findings(rigs, per_flight, quant, same, fidelity, chan, OUT_DIR / "findings.md")
    print(f"\nwrote {OUT_DIR}/summary.json, findings.md and {len(pngs)} PNGs")
    for p in pngs:
        print("   ", p)


if __name__ == "__main__":
    main()
