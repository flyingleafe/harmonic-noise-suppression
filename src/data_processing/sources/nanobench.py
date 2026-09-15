"""NanoBench source — nano-quadrotor system-identification benchmark telemetry.

Nonlinear System Identification for a Nano-drone Benchmark (IDSIA / USI-SUPSI)
https://github.com/idsia-robotics/nanodrone-sysid-benchmark
Busetto, Cereda, Forgione, Maroni, Piga, Palossi, *Control Engineering
Practice* 172:106871 (2026), doi:10.1016/j.conengprac.2026.106871.

Real flights of a **Crazyflie 2.1 Brushless** nano-quadrotor in a motion-capture
room, published as 15 CSVs (4 ``square`` + 4 ``random`` + 4 ``chirp`` training
runs, 3 ``melon`` test runs), one per flight, resampled to a uniform 100 Hz
grid. Each row holds the mocap/onboard state (position, quaternion, velocity,
body rates, body acceleration) and the four ESC-reported motor speeds
``m1_rads``..``m4_rads``.

:func:`build` keeps only the rotor speeds: one Frame per CSV with ``rps``
(rev/s, ``("rotor", "time")``) and ``meta``. No audio.

Units (evidence, see ``extra.units_note`` of every frame)
---------------------------------------------------------
The CSV columns are *mechanical* rad/s. The publisher's own pipeline builds
them from the Crazyflie's bidirectional-DSHOT eRPM telemetry
(``processing/csv_to_processed.ipynb`` on the repo's ``dev`` branch)::

    # Electrical RPM -> mechanical RPM -> rad/s
    ERPM_TO_RADS = 2 * np.pi / (6 * 60)   # 6 electrical pole pairs

i.e. eRPM / 6 pole pairs -> mechanical RPM -> rad/s (the raw CF log field is
eRPM/100 with 65535 as the "no reading" sentinel; ``utils/topic_utils.py``
``extract_motor_erpm`` rescales by 100 and NaNs the sentinel). The magnitude
confirms it independently: hover sits near 1700 rad/s = 271 rev/s = 16.2 kRPM,
and the publisher's physics model (``models/models.py``,
``PhysQuadModel(Kt=3.72e-08)``, ``T = Kt * sum(omega**2)``, docstring "u_mot:
(B,4) in rad/s") then gives a hover thrust of 4 * 3.72e-8 * 1700**2 = 0.43 N =
43.8 g, matching the 45 g vehicle mass the same pipeline uses
(``processing/csv_to_processed.ipynb``: ``m = 0.045``). Reading the column as
mechanical RPM (28 rev/s) or as eRPM (4.7 rev/s) is off by 1-2 orders of
magnitude from anything that can hover. So ``rps = m*_rads / (2*pi)``.

Rotor order / layout
--------------------
The README does not state the motor numbering, but the publisher's physics
model does, through its mixing matrix (``models/models.py``): roll groups
``{3,4}`` against ``{1,2}``, pitch groups ``{2,3}`` against ``{1,4}`` and yaw
groups ``{1,3}`` against ``{2,4}`` — exactly the Crazyflie firmware's
quadrotor power distribution (``m1: -r+p``, ``m2: -r-p``, ``m3: +r-p``,
``m4: +r+p``, yaw ``+,-,+,-``), i.e. the stock Crazyflie X-configuration
numbering M1 front-right, M2 back-right, M3 back-left, M4 front-left.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tdseries as td

from data_processing.downloaders import http_fetch
from data_processing.sources._common import meta_frame, safe_key

RIG = "nanobench_cf21b"
VEHICLE = "Crazyflie 2.1 Brushless nano-quadrotor"
NUM_ROTORS = 4
NATIVE_RATE_HZ = 100.0
#: Crazyflie X-configuration numbering (see the module docstring for evidence).
ROTOR_LAYOUT = ["front_right", "back_right", "back_left", "front_left"]
MOTOR_COLUMNS = ("m1_rads", "m2_rads", "m3_rads", "m4_rads")
VELOCITY_COLUMNS = ("vx", "vy", "vz")
RADS_TO_REV_PER_S = 1.0 / (2.0 * np.pi)
#: Relative jitter of the sample period above which we fall back to td.events.
UNIFORM_JITTER_TOL = 0.05

REPO = "idsia-robotics/nanodrone-sysid-benchmark"
BRANCH = "main"
#: Git-LFS media endpoint: serves the resolved LFS object, no auth, no git.
LFS_MEDIA_URL = "https://media.githubusercontent.com/media/{repo}/{branch}/{path}"

#: The published flights, as ``<split>/<file>.csv`` relative to the raw root
#: (the repo path is ``data/<split>/<file>.csv``).
CSV_FILES: tuple[str, ...] = (
    "test/melon_20251017_run1.csv",
    "test/melon_20251017_run2.csv",
    "test/melon_20251017_run3.csv",
    "train/chirp_20251017_run1.csv",
    "train/chirp_20251017_run2.csv",
    "train/chirp_20251017_run3.csv",
    "train/chirp_20251017_run4.csv",
    "train/random_20251017_run1.csv",
    "train/random_20251017_run2.csv",
    "train/random_20251017_run3.csv",
    "train/random_20251017_run4.csv",
    "train/square_20251017_run1.csv",
    "train/square_20251017_run2.csv",
    "train/square_20251017_run3.csv",
    "train/square_20251017_run4.csv",
)

_LFS_POINTER_MAGIC = b"version https://git-lfs"


# ─── Raw download (Git-LFS media URLs; no git/LFS client needed) ──────────────


def _download_csv(rel_path: str, dest_file: Path) -> Path:
    if dest_file.exists() and dest_file.stat().st_size > len(_LFS_POINTER_MAGIC):
        return dest_file
    url = LFS_MEDIA_URL.format(repo=REPO, branch=BRANCH, path=f"data/{rel_path}")
    print(f"Downloading {rel_path}...")
    http_fetch(url, dest_file)
    with open(dest_file, "rb") as f:
        head = f.read(len(_LFS_POINTER_MAGIC))
    if head == _LFS_POINTER_MAGIC:
        dest_file.unlink()
        raise RuntimeError(
            f"{url} returned a Git-LFS pointer, not the CSV bytes "
            "(media.githubusercontent.com did not resolve the LFS object)"
        )
    return dest_file


def download_nanobench(dest: Path) -> Path:
    """Fetch the 15 flight CSVs into ``dest`` (idempotent, ~20 MB total)."""
    root = Path(dest)
    root.mkdir(parents=True, exist_ok=True)
    for rel_path in CSV_FILES:
        _download_csv(rel_path, root / rel_path)
    return root


# ─── Discovery + frame building ───────────────────────────────────────────────


def discover_flights(raw_dir: Path) -> list[Path]:
    """Every flight CSV in the raw tree, in deterministic (sorted) order."""
    raw_dir = Path(raw_dir)
    return sorted(
        (p for p in raw_dir.rglob("*.csv") if p.is_file()),
        key=lambda p: p.relative_to(raw_dir).as_posix(),
    )


def _rps_series(rps: np.ndarray, times: np.ndarray) -> td.Series:
    """``(4, T)`` rev/s on its own clock.

    The publisher resamples every run onto an exact ``np.arange(t0, t1, 0.01)``
    grid (``processing/bag_to_csv.py`` on the ``dev`` branch), so a regular
    100 Hz grid becomes a ``td.uniform`` series at the declared rate (the
    reconstructed period is only float-exact to ~1e-13 and tdseries rejects a
    non-integral float rate). Anything else keeps its true timestamps.
    """
    if times.size < 2:
        return td.uniform(rps, NATIVE_RATE_HZ, dims=("rotor", "time"))
    dt = np.diff(times)
    period = float(np.median(dt))
    jitter = float(np.max(np.abs(dt - period))) / period if period > 0 else np.inf
    off_rate = abs(1.0 / period - NATIVE_RATE_HZ) / NATIVE_RATE_HZ if period > 0 else np.inf
    if jitter < UNIFORM_JITTER_TOL and off_rate < UNIFORM_JITTER_TOL:
        return td.uniform(rps, NATIVE_RATE_HZ, dims=("rotor", "time"))
    return td.events(times, rps.T, dims=("rotor", "time"))


def build_frame(csv_path: Path, raw_dir: Path) -> td.Frame:
    """One NanoBench flight CSV -> ``rps`` (rev/s) + ``meta`` Frame."""
    rel_path = csv_path.relative_to(raw_dir).as_posix()
    flight = csv_path.stem
    trajectory = flight.split("_", 1)[0]

    frame_csv = pd.read_csv(
        csv_path, usecols=lambda c: c in ("t", *VELOCITY_COLUMNS, *MOTOR_COLUMNS)
    )
    missing = [c for c in ("t", *MOTOR_COLUMNS) if c not in frame_csv.columns]
    if missing:
        raise ValueError(f"{rel_path}: missing column(s) {missing}")

    rps = frame_csv.loc[:, list(MOTOR_COLUMNS)].to_numpy(dtype=np.float64).T * RADS_TO_REV_PER_S
    times = frame_csv["t"].to_numpy(dtype=np.float64)

    # Drop leading/trailing samples with no reading on any rotor; interior NaNs
    # (DSHOT dropouts, the 65535 sentinel upstream) are kept as NaN.
    valid = np.flatnonzero(np.any(np.isfinite(rps), axis=0))
    if valid.size == 0:
        raise ValueError(f"{rel_path}: no finite motor samples")
    lo, hi = int(valid[0]), int(valid[-1]) + 1
    rps = np.ascontiguousarray(rps[:, lo:hi])
    times = times[lo:hi] - times[lo]

    speeds = (
        np.linalg.norm(
            frame_csv.loc[:, list(VELOCITY_COLUMNS)].to_numpy(dtype=np.float64)[lo:hi], axis=1
        )
        if all(c in frame_csv.columns for c in VELOCITY_COLUMNS)
        else None
    )
    max_speed = (
        None if speeds is None or not np.any(np.isfinite(speeds)) else float(np.nanmax(speeds))
    )

    rps_series = _rps_series(rps, times)
    meta = meta_frame(
        recording_id=flight,
        dataset="NanoBench",
        system={
            "rig": RIG,
            "vehicle": VEHICLE,
            "n_rotors": NUM_ROTORS,
            "rotor_layout": ROTOR_LAYOUT,
            "speed_source": "esc_feedback",
            "native_rate_hz": NATIVE_RATE_HZ,
        },
        operating={
            "trajectory": trajectory,
            "split": csv_path.parent.name,
            "duration_s": float(times[-1] - times[0]),
            "max_speed_mps": max_speed,
            "environment": "indoor",
        },
        extra={
            "units_note": (
                "CSV columns m1_rads..m4_rads are mechanical rad/s; rps = rad/s / (2*pi). "
                "Publisher pipeline (dev branch processing/csv_to_processed.ipynb): "
                "ERPM_TO_RADS = 2*pi/(6*60), i.e. bidirectional-DSHOT eRPM / 6 pole pairs "
                "-> mechanical RPM -> rad/s (12-pole motors). Cross-check: hover ~1700 rad/s "
                "= 271 rev/s = 16.2 kRPM, and models.py PhysQuadModel(Kt=3.72e-08) with "
                "T = Kt*sum(omega^2) gives 0.43 N = 43.8 g hover thrust vs the pipeline's "
                "m = 0.045 kg; RPM or eRPM readings (28 / 4.7 rev/s) cannot hover."
            ),
            "layout_note": (
                "rotor_layout is the stock Crazyflie X-config numbering, inferred from the "
                "publisher's mixing matrix in models/models.py (roll {3,4}|{1,2}, "
                "pitch {2,3}|{1,4}, yaw {1,3}|{2,4}) matching the Crazyflie firmware's "
                "power_distribution_quadrotor; the README does not state it."
            ),
            "source_file": rel_path,
        },
    )
    return td.Frame({"rps": rps_series, "meta": meta})


def build(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """Yield ``(key, frame)`` for every flight CSV (one flight in RAM at a time)."""
    raw_dir = Path(raw_dir)
    for csv_path in discover_flights(raw_dir):
        yield safe_key(f"{RIG}__{csv_path.stem}"), build_frame(csv_path, raw_dir)


# ─── Registry provenance ──────────────────────────────────────────────────────

PROVENANCE: dict[str, Any] = {
    "source_url": "https://github.com/idsia-robotics/nanodrone-sysid-benchmark",
    "license": (
        "no licence file in the repository and no licence declared on GitHub "
        "(as of 2026-09-15); published as the companion data of the paper, used "
        "here for research with attribution"
    ),
    "citation": (
        "R. Busetto, E. Cereda, M. Forgione, G. Maroni, D. Piga, D. Palossi, "
        "'Nonlinear system identification for a nano-drone benchmark', Control "
        "Engineering Practice 172:106871 (2026), "
        "doi:10.1016/j.conengprac.2026.106871 (arXiv:2512.14450)."
    ),
    "description": (
        "Crazyflie 2.1 Brushless nano-quadrotor rotor-speed telemetry: 15 mocap-room "
        "flights (square/random/chirp train, melon test) as rps Frames in rev/s at "
        "100 Hz, converted from the published mechanical-rad/s columns "
        "(bidirectional-DSHOT eRPM / 6 pole pairs upstream). Telemetry only, no audio."
    ),
    "sample_rate": NATIVE_RATE_HZ,
    "channels": NUM_ROTORS,
}
