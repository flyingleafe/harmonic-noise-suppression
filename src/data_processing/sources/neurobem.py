"""NeuroBEM source — 400 Hz per-rotor speed telemetry of agile quadrotor flights.

NeuroBEM: Hybrid Aerodynamic Quadrotor Model
L. Bauersfeld*, E. Kaufmann*, P. Foehn, S. Sun, D. Scaramuzza, RSS 2021.
Project page: https://rpg.ifi.uzh.ch/NeuroBEM.html
Schema / README: https://rpg.ifi.uzh.ch/neuro_bem/Readme.html
Download index: https://download.ifi.uzh.ch/rpg/NeuroBEM/

1h15min of flights of one custom-built quadrotor (0.772 kg) in a large hangar
with a Vicon system, at speeds up to 65 km/h. The raw tree this module needs is
just ``processed_data.zip`` (627 MB zipped, 1.4 GB of csv) plus the 23 kB
``Flights.txt`` flight index; the published ``raw_data/`` (Vicon rosbags +
betaflight blackbox logs), ``pdf/``, ``predictions/`` and ``code/`` folders are
*not* fetched — they carry no extra rotor-speed information.

``processed_data/merged_<flight>_seg_<k>.csv`` holds one continuous, Vicon-
dropout-free segment of flight ``<flight>``, sampled at exactly 400 Hz, with
the 29 columns documented in the README. This builder keeps only the rotor
speeds: ``mot 1..mot 4`` [rad/s], converted to rev/s, in the README's own
rotor order (mot 1 back-right, mot 2 front-right, mot 3 back-left, mot 4
front-left). One :class:`td.Frame` per csv segment, ``rps`` + ``meta`` only
(this dataset has no audio).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import numpy as np
import tdseries as td

from data_processing.downloaders import extract_zip, http_fetch
from data_processing.sources._common import meta_frame, safe_key

# ─── Raw fetch ────────────────────────────────────────────────────────────────

BASE_URL = "https://download.ifi.uzh.ch/rpg/NeuroBEM"
PROCESSED_DATA_URL = f"{BASE_URL}/processed_data.zip"
FLIGHTS_URL = f"{BASE_URL}/Flights.txt"
#: Size of ``processed_data.zip`` as published (Last-Modified 2021-06-14);
#: ``http_fetch`` skips an already-complete download by size match.
PROCESSED_DATA_BYTES = 657316788
#: Byte size of ``Flights.txt`` as published.
FLIGHTS_BYTES = 7043
EXTRACTED_MARKER = ".extracted"

# ─── Dataset facts (README) ───────────────────────────────────────────────────

RIG = "neurobem_quad"
N_ROTORS = 4
#: Native logging rate of ``processed_data/`` (README: "At a rate of 400 Hz").
RATE_HZ = 400.0
#: README column order of ``mot 1..mot 4``.
ROTOR_LAYOUT = ["back_right", "front_right", "back_left", "front_left"]
TIME_COLUMN = "t"
RPS_COLUMNS = ("mot 1", "mot 2", "mot 3", "mot 4")
#: Tolerated sampling jitter (fraction of the median period) for ``td.uniform``.
MAX_RATE_JITTER = 0.05

_CSV_RE = re.compile(r"^merged_(?P<flight>[\d-]+)_seg_(?P<seg>\d+)\.csv$")
_FLIGHT_LINE_RE = re.compile(
    r"""^\s*dataset\s*=\s*"(?P<flight>[^"]+)"\s*;\s*%?\s*(?P<note>.*?)\s*$"""
)
_VEL_RE = re.compile(r"vel\s*=\s*([0-9]*\.?[0-9]+)")


def download_neurobem(dest: Path) -> Path:
    """Fetch ``processed_data/`` + ``Flights.txt`` into *dest* (idempotent).

    Only these two artifacts: the published ``raw_data/`` (Vicon rosbags +
    betaflight logs), ``pdf/``, ``predictions/`` and ``code/`` folders carry no
    extra rotor-speed information. The zip is deleted once extracted — the csv
    tree is 1.4 GB and keeping the 627 MB zip beside it only costs disk; a
    re-run re-downloads it only when the extraction marker is missing.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    http_fetch(FLIGHTS_URL, dest / "Flights.txt", expected_size=FLIGHTS_BYTES)

    marker = dest / EXTRACTED_MARKER
    if not marker.exists():
        zip_path = dest / "processed_data.zip"
        http_fetch(PROCESSED_DATA_URL, zip_path, expected_size=PROCESSED_DATA_BYTES)
        got = zip_path.stat().st_size
        if got != PROCESSED_DATA_BYTES:
            zip_path.unlink()
            raise OSError(f"{PROCESSED_DATA_URL}: got {got} B, want {PROCESSED_DATA_BYTES} B")
        print(f"Extracting {zip_path.name}...")
        extract_zip(zip_path, dest)
        zip_path.unlink()
        marker.write_text("processed_data.zip\n", encoding="utf-8")

    processed = dest / "processed_data"
    if not processed.is_dir():
        raise FileNotFoundError(f"missing {processed} after extraction")
    return dest


# ─── Flight index ─────────────────────────────────────────────────────────────


def parse_flights_txt(path: Path) -> dict[str, str]:
    """``{flight id: trajectory note}`` from the published ``Flights.txt``."""
    notes: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = _FLIGHT_LINE_RE.match(line)
        if m:
            notes[m.group("flight")] = m.group("note").strip().rstrip(",").strip()
    return notes


def _listed_speed(note: str | None) -> float | None:
    """The ``vel=`` reference speed [m/s] the publisher lists for a flight."""
    if not note:
        return None
    m = _VEL_RE.search(note)
    return float(m.group(1)) if m else None


# ─── Segment loading ──────────────────────────────────────────────────────────


def _iter_segments(raw_dir: Path) -> Iterator[tuple[str, int, Path]]:
    """``(flight, segment, csv path)`` for every processed segment, sorted."""
    processed = Path(raw_dir) / "processed_data"
    found: list[tuple[str, int, Path]] = []
    for path in processed.iterdir():
        m = _CSV_RE.match(path.name)  # skips the stray .swp the publisher shipped
        if path.is_file() and m:
            found.append((m.group("flight"), int(m.group("seg")), path))
    yield from sorted(found, key=lambda item: (item[0], item[1]))


def _read_segment(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``((4, T) rev/s, (T,) seconds-from-first-sample)`` from one segment csv."""
    import pandas as pd

    # pandas' usecols overloads are narrower than its runtime contract.
    table = pd.read_csv(path, usecols=cast(Any, [TIME_COLUMN, *RPS_COLUMNS]), dtype=np.float64)
    t = table[TIME_COLUMN].to_numpy(dtype=np.float64)
    rps = table[list(RPS_COLUMNS)].to_numpy(dtype=np.float64).T / (2.0 * np.pi)

    keep = ~np.all(np.isnan(rps), axis=0)
    if not keep.any():
        return rps[:, :0], t[:0]
    first, last = int(np.argmax(keep)), int(len(keep) - np.argmax(keep[::-1]))
    rps, t = np.ascontiguousarray(rps[:, first:last]), t[first:last]
    return rps, t - t[0]


def _rps_series(rps: np.ndarray, t: np.ndarray) -> tuple[td.Series, float]:
    """``rps`` as a Series on its native clock, plus the rate it is stored at.

    ``processed_data`` clocks are exact rational periods (1/400 s everywhere
    except flight 2021-02-18-16-43-54, which is logged at 1/164 s), but the
    printed decimals only recover them to ~1e-13; ``td.uniform`` refuses a
    non-integral float rate, so the measured period is snapped to the nearest
    simple fraction and the snap is verified against every sample interval.
    """
    if len(t) < 2:
        return td.uniform(rps, RATE_HZ, dims=("rotor", "time")), RATE_HZ
    dt = np.diff(t)
    period = float(np.median(dt))
    jitter = float(np.max(np.abs(dt - period))) / period
    if jitter <= MAX_RATE_JITTER:
        snapped = Fraction(period).limit_denominator(1_000_000)
        rate = Fraction(snapped.denominator, snapped.numerator)
        if abs(float(snapped) - period) <= 1e-6 * period:
            return td.uniform(rps, rate, dims=("rotor", "time")), float(rate)
    return td.events(t, rps, dims=("rotor", "time")), 1.0 / period


# ─── Builder (raw tree -> tdframe-v1 telemetry Frames) ───────────────────────


def build(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """Yield ``(key, frame)`` with ``rps`` [rev/s] + ``meta`` per flight segment."""
    raw_dir = Path(raw_dir)
    flights_txt = raw_dir / "Flights.txt"
    if flights_txt.is_file():
        notes = parse_flights_txt(flights_txt)
    else:
        notes = {}
        print(f"Warning: {flights_txt} missing — no trajectory/max-speed labels")

    for flight, segment, path in _iter_segments(raw_dir):
        rps, t = _read_segment(path)
        if rps.shape[1] == 0:
            print(f"Warning: skipping all-NaN segment {path.name}")
            continue
        series, rate = _rps_series(rps, t)
        note = notes.get(flight)
        recording_id = f"{flight}_seg_{segment}"
        meta = meta_frame(
            recording_id=recording_id,
            dataset="NeuroBEM",
            system={
                "rig": RIG,
                "vehicle": "custom-built agile quadrotor, 0.772 kg, betaflight inner loop",
                "n_rotors": N_ROTORS,
                "rotor_layout": list(ROTOR_LAYOUT),
                "speed_source": "esc_feedback",
                "native_rate_hz": float(rate),
            },
            operating={
                "trajectory": note,
                "duration_s": float(t[-1] - t[0]),
                "max_speed_mps": _listed_speed(note),
                "environment": "indoor",
            },
            extra={
                "units_note": "processed_data 'mot 1..mot 4' [rad/s] / (2*pi)",
                "source_file": str(path.relative_to(raw_dir)),
            },
        )
        yield safe_key(f"{RIG}__{recording_id}"), td.Frame({"rps": series, "meta": meta})


# ─── Registry provenance ──────────────────────────────────────────────────────

PROVENANCE: dict[str, Any] = {
    "source_url": "https://rpg.ifi.uzh.ch/NeuroBEM.html",
    "download_url": PROCESSED_DATA_URL,
    "license": (
        "not stated — the project page carries only '(c) 2021 Robotics and Perception "
        "Group, University of Zurich, Switzerland' and asks that the RSS 2021 paper be "
        "cited; no dataset licence or terms-of-use document is published"
    ),
    "citation": (
        "L. Bauersfeld, E. Kaufmann, P. Foehn, S. Sun, D. Scaramuzza, "
        "NeuroBEM: Hybrid Aerodynamic Quadrotor Model, Robotics: Science and Systems, 2021."
    ),
    "description": (
        "Per-rotor speed telemetry of one custom-built 0.772 kg quadrotor flown "
        "agilely (up to 65 km/h) in a Vicon-equipped hangar: 1h15min over 95 flights, "
        "published as 247 dropout-free 400 Hz segments in processed_data/. Frames "
        "carry rps (rev/s, from the README's 'mot 1..mot 4' [rad/s] columns divided "
        "by 2*pi) and meta only; no audio. Rotor order is the README's: mot 1 "
        "back-right, mot 2 front-right, mot 3 back-left, mot 4 front-left. "
        "speed_source='esc_feedback': the README lists 'individual motor speeds' among "
        "the measured signals and the raw tree provides betaflight blackbox logs as "
        "their origin, i.e. ESC RPM telemetry (the publisher already converted to "
        "rad/s, so no pole-pair factor is applied here). operating.trajectory is the "
        "Flights.txt comment verbatim and operating.max_speed_mps is the 'vel=' "
        "reference speed it lists (nominal set-point, not a measured maximum; absent "
        "for the cpc/TWR and random-points flights)."
    ),
    "sample_rate": RATE_HZ,
    "channels": N_ROTORS,
}
