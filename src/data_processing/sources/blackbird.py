"""Blackbird source — per-rotor speed telemetry of the MIT AERA Blackbird quad.

The Blackbird Dataset: A large-scale dataset for UAV perception in aggressive
flight (Antonini, Guerra, Murali, Sayre-McCord, Karaman; ISER 2018 / IJRR 2020)
https://github.com/mit-aera/Blackbird-Dataset · https://arxiv.org/abs/1810.01987

One custom-built 0.915 kg quadrotor (DJI Snail propulsion, Xsens MTi-3 IMU,
Jetson TX2), 163 unique indoor flights over 17 trajectories at 0.5–7.0 m/s in
an 11 × 11 × 5.5 m motion-capture room. Rotor speed comes from *custom optical
motor encoders* ("custom made optical motor encoders for accurate motor speed
measurements", § 2; verified against an external tachometer, § 4) logged at
~190 Hz — hence ``speed_source="tachometer"``, not ESC feedback. The published
``blackbird/MotorRPM`` message is ``Header header; time[] sample_stamp;
float32[] rpm`` with one ``rpm`` entry per rotor in *mechanical* RPM (no pole
pairs involved): the paper's thrust coefficient ``CT = 2.27e-8 N/rpm²`` puts
hover at ``sqrt(0.915 * 9.81 / 4 / 2.27e-8) ≈ 9.9 krpm ≈ 166 rev/s``, which is
what the telemetry shows.

Smallest artefact holding the rotor speeds is the publisher's per-topic CSV
export ``<flight>/csv/blackbird_slash_rotor_rpm.csv`` (~8 MB/flight, produced
by ``logConversionUtilities/bagToCsv.py``), an order of magnitude smaller than
the flight's ``rosbag.bag`` (IMU + pose + PWM + TF + RPM). :func:`build` reads
either: the CSV export if present, else the bag (lazy ``rosbags`` import, the
``blackbird/MotorRPM`` definition taken from the bag's own connection header).

ACCESS NOTE (checked 2026-09-15) — the canonical distribution is OFFLINE:

- ``blackbird-dataset.mit.edu`` does not resolve (NXDOMAIN). Dead since ~2024;
  the maintainers said in issue #28 (2024-04-10) that they were migrating off
  AWS, and never published a new location (issues #28/#30/#31/#32/#34 open).
- The legacy S3 buckets behind it (``ijrr-blackbird-dataset``,
  ``blackbird-dataset-static-index``, both referenced from the repo's
  ``websiteUtils/redirect_rules.xml`` / ``updateStaticIndex.sh``) answer
  ``AccessDenied`` for every key and for listing.
- The Academic Torrents copy (eb542a231dbeb2125e4ec88ddd18841a867c2656,
  4.79 TB) contains 560 ``.tar`` image archives + 187 ``.mp4`` and *no*
  telemetry at all (bags/CSVs were never in it — see repo issue #1), with no
  seeders; the Internet Archive mirror of that item holds only the torrent.
- The Wayback Machine archived only ``.../videos/`` directory listings.
- No mirror on HuggingFace, Zenodo, Kaggle, OpenDataLab or Dataverse.

Third-party repositories that vendored a Blackbird flight are therefore the
only live source of real rotor telemetry; :data:`FLIGHT_MIRRORS` pins the ones
found. :func:`download_blackbird` still tries the canonical URL for every
flight in :data:`FLIGHTS`, so the fetcher reproduces the full intended subset
the day the publisher comes back.
"""

from __future__ import annotations

import ssl
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import numpy as np
import tdseries as td

from data_processing.sources._common import meta_frame, safe_key

DATASET_NAME = "Blackbird"
#: One physical vehicle for the whole dataset (the "Blackbird" quadrotor).
RIG_ID = "blackbird_quad"
NUM_ROTORS = 4
#: Publisher's nominal motor-encoder logging rate (paper § 3).
NOMINAL_RATE_HZ = 190.0
#: Relative path of the per-topic rotor-speed CSV export inside a flight dir.
RPM_CSV_RELPATH = "csv/blackbird_slash_rotor_rpm.csv"
CANONICAL_BASE = "http://blackbird-dataset.mit.edu/BlackbirdDatasetData"
#: Regular sampling is claimed only when peak jitter stays below this fraction
#: of the median period; Blackbird's logger drops samples, so it lands on
#: :func:`tdseries.events`.
UNIFORM_JITTER_TOL = 0.05

#: Fallback ``blackbird/MotorRPM`` definition (bags normally carry their own).
MOTOR_RPM_MSGDEF = """Header header
time[] sample_stamp
float32[] rpm
"""

# ─── Flight subset ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FlightSpec:
    """One flight of the chosen subset (``<trajectory>/<yawMode>/maxSpeed<V>``)."""

    flight: str
    #: Size of ``csv/blackbird_slash_rotor_rpm.csv`` in bytes, where a copy
    #: could actually be measured (the canonical host is down, see module doc).
    rpm_csv_bytes: int | None = None

    @property
    def trajectory(self) -> str:
        return self.flight.split("/")[0]

    @property
    def yaw_mode(self) -> str:
        return self.flight.split("/")[1]

    @property
    def max_speed_mps(self) -> float:
        return parse_max_speed(self.flight.split("/")[2])


def parse_max_speed(token: str) -> float:
    """``"maxSpeed5p0"`` → ``5.0``."""
    return float(token.removeprefix("maxSpeed").replace("p", "."))


#: The subset this source ingests: every trajectory of the published catalogue
#: (paper Table 3 / repo README) at two speeds — a slow and a fast one — plus
#: yawConstant twins of the fast flights, so the rotor-speed model sees both
#: yaw regimes. 45 flights × ~8 MB of rotor-RPM CSV ≈ 0.36 GB (cap: 6 GB); the
#: flights' ``rosbag.bag`` files are never fetched.
FLIGHTS: tuple[FlightSpec, ...] = (
    FlightSpec("ampersand/yawForward/maxSpeed1p0"),
    FlightSpec("ampersand/yawForward/maxSpeed2p0"),
    FlightSpec("ampersand/yawConstant/maxSpeed3p0"),
    FlightSpec("bentDice/yawForward/maxSpeed1p0"),
    FlightSpec("bentDice/yawForward/maxSpeed3p0"),
    FlightSpec("bentDice/yawConstant/maxSpeed4p0"),
    FlightSpec("clover/yawForward/maxSpeed1p0"),
    FlightSpec("clover/yawForward/maxSpeed5p0", rpm_csv_bytes=8_017_574),
    FlightSpec("clover/yawConstant/maxSpeed6p0"),
    FlightSpec("dice/yawForward/maxSpeed1p0"),
    FlightSpec("dice/yawForward/maxSpeed3p0"),
    FlightSpec("dice/yawConstant/maxSpeed4p0"),
    FlightSpec("halfMoon/yawForward/maxSpeed1p0"),
    FlightSpec("halfMoon/yawForward/maxSpeed4p0"),
    FlightSpec("halfMoon/yawConstant/maxSpeed4p0"),
    FlightSpec("mouse/yawForward/maxSpeed0p5"),
    FlightSpec("mouse/yawForward/maxSpeed7p0"),
    FlightSpec("mouse/yawConstant/maxSpeed7p0"),
    FlightSpec("oval/yawForward/maxSpeed1p0"),
    FlightSpec("oval/yawForward/maxSpeed4p0"),
    FlightSpec("oval/yawConstant/maxSpeed4p0"),
    FlightSpec("patrick/yawForward/maxSpeed0p5"),
    FlightSpec("patrick/yawForward/maxSpeed4p0"),
    FlightSpec("patrick/yawConstant/maxSpeed5p0"),
    FlightSpec("picasso/yawForward/maxSpeed0p5"),
    FlightSpec("picasso/yawForward/maxSpeed5p0"),
    FlightSpec("picasso/yawConstant/maxSpeed6p0"),
    FlightSpec("sid/yawForward/maxSpeed0p5"),
    FlightSpec("sid/yawForward/maxSpeed4p0"),
    FlightSpec("sid/yawConstant/maxSpeed7p0"),
    FlightSpec("sphinx/yawForward/maxSpeed1p0"),
    FlightSpec("sphinx/yawForward/maxSpeed4p0"),
    FlightSpec("sphinx/yawConstant/maxSpeed4p0"),
    FlightSpec("star/yawForward/maxSpeed0p5"),
    FlightSpec("star/yawForward/maxSpeed5p0"),
    FlightSpec("star/yawConstant/maxSpeed5p0"),
    FlightSpec("thrice/yawForward/maxSpeed0p5"),
    FlightSpec("thrice/yawForward/maxSpeed6p0"),
    FlightSpec("thrice/yawConstant/maxSpeed7p0"),
    FlightSpec("tiltedThrice/yawForward/maxSpeed0p5"),
    FlightSpec("tiltedThrice/yawForward/maxSpeed6p0"),
    FlightSpec("tiltedThrice/yawConstant/maxSpeed7p0"),
    FlightSpec("winter/yawForward/maxSpeed0p5"),
    FlightSpec("winter/yawForward/maxSpeed4p0"),
    FlightSpec("winter/yawConstant/maxSpeed5p0"),
)

#: Live third-party copies of the publisher's own rotor-RPM CSV export, byte
#: for byte (same ``bagToCsv.py`` column layout). The only way to get real
#: Blackbird rotor telemetry while the canonical host is down.
FLIGHT_MIRRORS: dict[str, str] = {
    "clover/yawForward/maxSpeed5p0": (
        "https://raw.githubusercontent.com/fbanelli/learned-acceleration-estimator/"
        "main/cloverData/blackbird_slash_rotor_rpm.csv"
    ),
}


# ─── Raw download ─────────────────────────────────────────────────────────────


def _ssl_context() -> ssl.SSLContext:
    """Verified TLS context — OpenSSL's default CA file is absent on NixOS."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _download_file(url: str, dest: Path, *, timeout: float = 60.0) -> bool:
    """Fetch ``url`` → ``dest`` atomically. ``False`` on any HTTP/URL error."""
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    context = _ssl_context() if url.startswith("https:") else None
    try:
        with urllib.request.urlopen(url, timeout=timeout, context=context) as resp:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
        tmp.rename(dest)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
        tmp.unlink(missing_ok=True)
        return False


def download_blackbird(dest: Path) -> Path:
    """Fetch the rotor-RPM CSV export of every flight in :data:`FLIGHTS`.

    Each flight is tried on its mirror first (:data:`FLIGHT_MIRRORS`), then on
    the canonical host. The raw tree mirrors the publisher's layout:
    ``<dest>/<trajectory>/<yawMode>/maxSpeed<V>/csv/blackbird_slash_rotor_rpm.csv``.
    Raises only when *nothing* could be fetched (see the module's access note).
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    have: list[str] = []
    missing: list[str] = []
    for spec in FLIGHTS:
        out = dest / spec.flight / RPM_CSV_RELPATH
        if out.exists() and out.stat().st_size > 0:
            have.append(spec.flight)
            continue
        urls = [
            url
            for url in (
                FLIGHT_MIRRORS.get(spec.flight),
                f"{CANONICAL_BASE}/{spec.flight}/{RPM_CSV_RELPATH}",
            )
            if url
        ]
        if any(_download_file(url, out) for url in urls):
            have.append(spec.flight)
        else:
            missing.append(spec.flight)
    if not have:
        raise RuntimeError(
            "Blackbird: no flight could be fetched. The canonical host "
            f"({CANONICAL_BASE}) has been offline since ~2024 (NXDOMAIN) and the "
            "legacy S3 buckets answer AccessDenied; see this module's docstring "
            "for the full access record."
        )
    if missing:
        print(
            f"Blackbird: fetched {len(have)}/{len(FLIGHTS)} flights; "
            f"{len(missing)} unavailable (canonical host offline): {missing[0]}, ..."
        )
    return dest


# ─── Rotor-speed readers ──────────────────────────────────────────────────────

#: Column layout of ``bagToCsv.py``'s flattened ``blackbird/MotorRPM`` export:
#: ``rosbagTimestamp, header, seq, stamp, secs, nsecs, frame_id, sample_stamp,
#: (secs, nsecs) × 4 sample stamps, rpm1..rpm4``.
_CSV_HEADER_PREFIX = (
    "rosbagTimestamp",
    "header",
    "seq",
    "stamp",
    "secs",
    "nsecs",
    "frame_id",
    "sample_stamp",
)
_CSV_N_COLS = 20
_CSV_STAMP_COLS = (4, 5)  # header.stamp secs, nsecs
_CSV_RPM_COLS = (16, 17, 18, 19)


def _read_rpm_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(rpm (4, T) float64, t (T,) seconds)`` from the publisher's CSV export.

    Times come from ``header.stamp``, the message clock of the 190 Hz encoder
    log. The per-rotor ``sample_stamp`` array of the message is NOT usable as a
    time base: in the real exports it runs on a different, frequently stalled
    clock (observed: non-monotonic, up to 62 s spread inside one message and a
    drifting 0–65 s offset from the header stamp).
    """
    import pandas as pd

    header = pd.read_csv(path, nrows=0).columns.tolist()
    if len(header) != _CSV_N_COLS or tuple(header[: len(_CSV_HEADER_PREFIX)]) != _CSV_HEADER_PREFIX:
        raise ValueError(f"{path}: unexpected rotor_rpm CSV layout: {header}")
    cols = [*_CSV_STAMP_COLS, *_CSV_RPM_COLS]
    # pandas' usecols overloads are narrower than its runtime contract.
    df = pd.read_csv(path, usecols=cast(Any, cols), header=0)
    secs = df.iloc[:, 0].to_numpy(dtype=np.int64)
    nsecs = df.iloc[:, 1].to_numpy(dtype=np.int64)
    t_ns = secs * 1_000_000_000 + nsecs
    t = (t_ns - t_ns[0]).astype(np.float64) / 1e9
    rpm = np.ascontiguousarray(df.iloc[:, 2:].to_numpy(dtype=np.float64).T)
    return rpm, t


def _split_msgdefs(msgdef: str, msgtype: str) -> list[tuple[str, str]]:
    """Split a ROS1 connection message definition into ``(msgtype, text)``.

    Bag connections carry the full dependency chain concatenated behind
    ``====``/``MSG: pkg/Name`` separators; ``rosbags`` parses one at a time.
    """
    out: list[tuple[str, str]] = []
    current_type = msgtype
    lines: list[str] = []
    for line in msgdef.splitlines():
        if line.startswith("====="):
            out.append((current_type, "\n".join(lines)))
            lines = []
            current_type = ""
        elif not current_type and line.startswith("MSG:"):
            name = line.removeprefix("MSG:").strip()
            pkg, _, short = name.partition("/")
            current_type = f"{pkg}/msg/{short}"
        else:
            lines.append(line)
    if current_type:
        out.append((current_type, "\n".join(lines)))
    return [(name, text) for name, text in out if name and text.strip()]


def _read_rpm_bag(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(rpm (4, T) float64, t (T,) seconds)`` from a flight's ``rosbag.bag``.

    ``rosbags`` is imported lazily so that every other builder (and any box
    without it installed) keeps working.
    """
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)
    with Reader(path) as reader:
        conns = [c for c in reader.connections if c.msgtype.rsplit("/", 1)[-1] == "MotorRPM"]
        if not conns:
            raise ValueError(f"{path}: no blackbird/MotorRPM connection")
        for conn in conns:
            # ``conn.msgdef`` is a ``MessageDefinition`` (rosbags >= 0.10) or
            # plain text on older releases; empty on hand-rolled bags.
            raw_def = getattr(conn.msgdef, "data", conn.msgdef) or MOTOR_RPM_MSGDEF
            for name, text in _split_msgdefs(str(raw_def), conn.msgtype):
                try:
                    typestore.register(get_types_from_msg(text, name))
                except (KeyError, ValueError):  # already known (std_msgs/Header, ...)
                    continue
        times: list[int] = []
        rows: list[np.ndarray] = []
        for conn, _bag_ns, raw in reader.messages(connections=conns):
            msg = cast(Any, typestore.deserialize_ros1(raw, conn.msgtype))
            stamp = msg.header.stamp
            times.append(int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec))
            rows.append(np.asarray(msg.rpm, dtype=np.float64))
    if not rows:
        raise ValueError(f"{path}: empty blackbird/MotorRPM stream")
    t_ns = np.asarray(times, dtype=np.int64)
    t = (t_ns - t_ns[0]).astype(np.float64) / 1e9
    return np.ascontiguousarray(np.stack(rows, axis=1)), t


# ─── Frame assembly ───────────────────────────────────────────────────────────


def _trim_and_order(rps: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by time, drop leading/trailing all-NaN columns, re-zero the clock."""
    order = np.argsort(t, kind="stable")
    rps, t = rps[:, order], t[order]
    valid = np.flatnonzero(~np.isnan(rps).all(axis=0))
    if valid.size == 0:
        return rps[:, :0], t[:0]
    lo, hi = int(valid[0]), int(valid[-1]) + 1
    return np.ascontiguousarray(rps[:, lo:hi]), t[lo:hi] - t[lo]


def _rps_series(rps: np.ndarray, t: np.ndarray) -> tuple[td.Series, float]:
    """``(series, measured rate Hz)`` — uniform when the clock really is."""
    dt = np.diff(t)
    period = float(np.median(dt))
    if period <= 0.0:
        raise ValueError("non-increasing rotor-speed timestamps")
    rate = 1.0 / period
    jitter = float(np.max(np.abs(dt - period))) / period
    if jitter < UNIFORM_JITTER_TOL:
        # td.uniform stores the rate exactly: a measured float needs a ratio.
        exact = Fraction(rate).limit_denominator(1_000_000)
        return td.uniform(rps, exact, dims=("rotor", "time")), rate
    return td.events(t, rps, dims=("rotor", "time")), rate


def _iter_flight_files(raw_dir: Path) -> Iterator[tuple[str, Path]]:
    """``(flight id, artefact)`` per flight, deterministic, CSV export first."""
    flights: dict[str, Path] = {}
    for csv_path in sorted(raw_dir.rglob("blackbird_slash_rotor_rpm.csv")):
        flight = csv_path.parent.parent.relative_to(raw_dir).as_posix()
        flights[flight] = csv_path
    for bag_path in sorted(raw_dir.rglob("*.bag")):
        flight = bag_path.parent.relative_to(raw_dir).as_posix()
        flights.setdefault(flight, bag_path)
    yield from sorted(flights.items())


def build(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """Stream one ``rps`` Frame per Blackbird flight (rev/s, ~190 Hz).

    Reads the publisher's rotor-RPM CSV export where present, else the flight's
    rosbag; one flight is held in memory at a time.
    """
    raw_dir = Path(raw_dir)
    for flight, path in _iter_flight_files(raw_dir):
        rpm, t = _read_rpm_csv(path) if path.suffix == ".csv" else _read_rpm_bag(path)
        if rpm.shape[0] != NUM_ROTORS:
            raise ValueError(f"{path}: expected {NUM_ROTORS} rotors, got {rpm.shape[0]}")
        rps, t = _trim_and_order(rpm / 60.0, t)
        if t.size < 2:
            continue
        series, rate = _rps_series(rps, t)
        parts = flight.split("/")
        meta = meta_frame(
            recording_id=flight,
            dataset=DATASET_NAME,
            system={
                "rig": RIG_ID,
                "vehicle": "MIT Blackbird custom quadrotor (0.915 kg, DJI Snail propulsion)",
                "n_rotors": NUM_ROTORS,
                # Motor order in the MotorRPM array is undocumented (repo issue
                # #26 asked; never answered) — no layout claimed.
                "rotor_layout": None,
                "speed_source": "tachometer",
                "native_rate_hz": rate,
            },
            operating={
                "trajectory": flight,
                "duration_s": float(t[-1] - t[0]),
                "max_speed_mps": parse_max_speed(parts[-1]) if len(parts) >= 3 else None,
                "environment": "indoor",
            },
            extra={
                "units_note": (
                    "rev/s = MotorRPM.rpm / 60; rpm is mechanical shaft RPM from the "
                    "vehicle's optical motor encoders (no pole-pair conversion). "
                    "Cross-check: CT = 2.27e-8 N/rpm^2 puts hover at ~166 rev/s."
                ),
                "source_file": path.relative_to(raw_dir).as_posix(),
                "time_base": "MotorRPM header.stamp (per-rotor sample_stamp is unusable)",
            },
        )
        yield safe_key(f"{RIG_ID}__{flight}"), td.Frame({"rps": series, "meta": meta})


PROVENANCE = {
    "source_url": "https://github.com/mit-aera/Blackbird-Dataset",
    "license": "MIT (tooling); dataset released for research use",
    "citation": (
        "Antonini, Guerra, Murali, Sayre-McCord, Karaman. The Blackbird UAV dataset. "
        "IJRR 2020, doi:10.1177/0278364920908331 (ISER 2018: arXiv:1810.01987)."
    ),
    "collection_method": (
        "163 indoor flights of one custom quadrotor tracked by 24 OptiTrack cameras in an "
        "11 x 11 x 5.5 m motion-capture room; 17 periodic minimum-snap trajectories flown at "
        "0.5-7.0 m/s under an INDI controller, logged over LCM and republished as ROS bags."
    ),
    "equipment": (
        "Custom optical motor encoders (~190 Hz, tachometer-verified), Xsens MTi-3 IMU (100 Hz), "
        "DJI Snail propulsion, Jetson TX2."
    ),
    "observation_type": "onboard_telemetry",
    "sample_rate": NOMINAL_RATE_HZ,
    "channels": "4 rotor speeds (mechanical RPM -> rev/s)",
    "rig": RIG_ID,
    "raw_host_status": (
        "Canonical host blackbird-dataset.mit.edu is NXDOMAIN (offline since ~2024, repo issues "
        "#28/#30/#31/#32/#34); legacy S3 buckets ijrr-blackbird-dataset and "
        "blackbird-dataset-static-index return AccessDenied; the 4.79 TB Academic Torrents copy "
        "holds only 560 image .tar + 187 .mp4 (no telemetry, no seeders); Wayback archived only "
        "directory listings. Checked 2026-09-15."
    ),
    "description": (
        "Rotor-speed (rev/s) trajectories of the Blackbird quadrotor, one Frame per flight, "
        "read from the publisher's blackbird/MotorRPM CSV export (or rosbag). Flights are named "
        "<trajectory>/<yawMode>/maxSpeed<V>. Telemetry only; the dataset's imagery is not fetched."
    ),
}
