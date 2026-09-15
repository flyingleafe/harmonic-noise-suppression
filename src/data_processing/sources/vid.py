"""VID source — rotor-speed telemetry from the ZJU FAST-Lab VID dataset.

VID: Visual-Inertial-Dynamical dataset (a full-state UAV dataset with
per-rotor speed feedback)
https://github.com/ZJU-FAST-Lab/VID-Dataset  (arXiv:2103.11152)

Platform: a DJI Matrice-100 airframe re-motored with four DJI RoboMaster
M3508 brushless motors driven by C620 ESCs.  Each ESC reports *measured*
rotor speed in RPM over CAN at ~1 kHz; the onboard MCU stamps every sample
with a hardware clock (``hwts``) synchronised to the flight controller's PPS
and republishes it as ``/m100withm3508/m3508_m<k>`` (``uavmotor/m3508``:
``Header header; time hwts; uint8 id; int16 current; int16 rpm``).  The
dataset's own ``rpmconvert`` tool turns those four topics into
``/synced/allrpm`` by taking a median over 11 samples and copies the ``rpm``
field unscaled, which fixes the unit: ``rev/s = rpm / 60``.

Raw tree layout (dload dataset ``VID``)::

    <raw>/<sequence>.motors.npz

The published sequences are single ROS1 bags of 5.5-17 GB that bundle the
D435 depth/infra image streams with the telemetry, and the publisher offers
no telemetry-only variant.  :func:`download_vid` therefore *streams* each bag
over HTTP (QNAP share link, Range-resumable) through a sequential rosbag1
reader and writes only the motor records to disk (~5 MB per sequence instead
of ~10 GB).  Camera, IMU, RTK, Vicon and force-sensor records are skipped as
they fly past.

:func:`build` turns each distilled sequence into one ``tdframe-v1`` telemetry
Frame with ``rps`` (measured, rev/s, ``("rotor", "time")`` on rotor 1's
hardware event clock), ``rps_command`` (``target_rpm``, rev/s, on its own
~100 Hz clock) and ``meta``.  No audio.  The ESC current counts stay in the
raw ``.npz`` — no consumer asks for them yet.
"""

from __future__ import annotations

import base64
import bz2
import json
import re
import struct
import time
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import tdseries as td
from scipy.ndimage import median_filter

from data_processing.sources._common import meta_frame, safe_key

# ─── Publisher's QNAP share link ──────────────────────────────────────────────

SHARE_CGI = "http://zjufast.kmras.com:9110/share.cgi"
SHARE_SSID = "e08267c50fbe4e4bb9cdfe40f9fdb712"
SHARE_PASSWORD = "viddataset@2021"  # published alongside the link in the README
#: The share's password token: QNAP's ``ezEncode`` is plain base64.
SHARE_EP = base64.b64encode(SHARE_PASSWORD.encode()).decode()

# ─── Telemetry layout ─────────────────────────────────────────────────────────

NUM_ROTORS = 4
NATIVE_RATE_HZ = 1000.0  # C620 CAN feedback rate (measured: ~890-990 Hz)
RPM_TO_RPS = 1.0 / 60.0

#: ``/m100withm3508/m3508_m1`` in the flight bags, ``m35085_m1`` in the
#: motor-test bags (publisher typo), ``/synced/m3508_m1`` after their
#: time-sync tool has been run.
MOTOR_TOPIC_RE = re.compile(r"/m3508\d*_m([1-4])$")
TARGET_TOPIC_RE = re.compile(r"/target_rpm$")

#: Max tolerated offset between the four per-motor event clocks (one native
#: period).  Above this the four topics cannot be put on a shared clock.
STAMP_TOL_S = 1.0e-3
#: A rotor sample farther than this from the m1 stamp it is mapped onto is a
#: telemetry dropout and stays NaN instead of being interpolated across.
GAP_TOL_S = 2.0e-3
#: A sample whose deviation from a 5-sample median exceeds this is a corrupt
#: CAN frame, not rotor dynamics (measured: real deviations peak at 4.4 rev/s,
#: glitches at 100-500 rev/s), and becomes NaN.
GLITCH_TOL_RPS = 30.0

M3508_STRUCT = struct.Struct("<IIII")  # seq, stamp.sec, stamp.nsec, frame_id len
M3508_TAIL = struct.Struct("<IIBhh")  # hwts.sec, hwts.nsec, id, current, rpm
MOTORDATA_TAIL = struct.Struct("<II4h")  # hwts.sec, hwts.nsec, int16[4]

# ─── Published sequences (README table + share listing) ───────────────────────
# ``size_bytes`` is the published bag size; it is streamed, never stored.

SEQUENCES: dict[str, dict[str, Any]] = {
    "outdoor_rect_fast": {
        "folder": "/outdoor",
        "filename": "outdoor_rect_fast_3547.2g_113.21s.bag",
        "size_bytes": 8418332479,
        "environment": "outdoor",
        "trajectory": "rectangle",
        "feature": "fast, with yaw",
        "mass_g": 3547.2,
        "duration_s": 113.21,
    },
    "outdoor_rect_slow": {
        "folder": "/outdoor",
        "filename": "outdoor_rect_slow_3547.2g_175.38s.bag",
        "size_bytes": 13041260347,
        "environment": "outdoor",
        "trajectory": "rectangle",
        "feature": "slow, with yaw",
        "mass_g": 3547.2,
        "duration_s": 175.38,
    },
    "outdoor_round_yaw": {
        "folder": "/outdoor",
        "filename": "outdoor_round_yaw_3541.7g_147.34s.bag",
        "size_bytes": 10955075684,
        "environment": "outdoor",
        "trajectory": "round",
        "feature": "with yaw",
        "mass_g": 3541.7,
        "duration_s": 147.34,
    },
    "outdoor_round_noyaw": {
        "folder": "/outdoor",
        "filename": "outdoor_round_noyaw_3541.7g_105.85s.bag",
        "size_bytes": 7871668004,
        "environment": "outdoor",
        "trajectory": "round",
        "feature": "without yaw",
        "mass_g": 3541.7,
        "duration_s": 105.85,
    },
    "outdoor_8_yaw": {
        "folder": "/outdoor",
        "filename": "outdoor_8_yaw_3541.7g_184.29s.bag",
        "size_bytes": 13701990272,
        "environment": "outdoor",
        "trajectory": "8-character",
        "feature": "with yaw",
        "mass_g": 3541.7,
        "duration_s": 184.29,
    },
    "outdoor_8_noyaw": {
        "folder": "/outdoor",
        "filename": "outdoor_8_noyaw_3541.4g_243.73s.bag",
        "size_bytes": 18187068621,
        "environment": "outdoor",
        "trajectory": "8-character",
        "feature": "without yaw",
        "mass_g": 3541.4,
        "duration_s": 243.73,
    },
    "night_rect_fast": {
        "folder": "/night",
        "filename": "night_rect_fast_3460.0g_133.58.bag",
        "size_bytes": 9932002490,
        "environment": "outdoor",
        "trajectory": "rectangle",
        "feature": "fast, with yaw",
        "mass_g": 3460.0,
        "duration_s": 133.58,
        "lighting": "night",
    },
    "night_rect_slow": {
        "folder": "/night",
        "filename": "night_rect_slow_3541.6g_179.48s.bag",
        "size_bytes": 13345619265,
        "environment": "outdoor",
        "trajectory": "rectangle",
        "feature": "slow, with yaw",
        "mass_g": 3541.6,
        "duration_s": 179.48,
        "lighting": "night",
    },
    "night_round_noyaw": {
        "folder": "/night",
        "filename": "night_round_no_yaw_3541.6g_82.07s.bag",
        "size_bytes": 6102636804,
        "environment": "outdoor",
        "trajectory": "round",
        "feature": "without yaw",
        "mass_g": 3541.6,
        "duration_s": 82.07,
        "lighting": "night",
    },
    "night_8_noyaw": {
        "folder": "/night",
        "filename": "night_8_noyaw_3547.7g_121.87s.bag",
        "size_bytes": 9062515073,
        "environment": "outdoor",
        "trajectory": "8-character",
        "feature": "without yaw",
        "mass_g": 3547.7,
        "duration_s": 121.87,
        "lighting": "night",
    },
    "indoor_loadless_hovor": {
        "folder": "/indoor",
        "filename": "indoor_loadless_hovor_3096.1g_79.04s.bag",
        "size_bytes": 5879866888,
        "environment": "indoor",
        "trajectory": "hovor",
        "feature": "loadless",
        "mass_g": 3096.1,
        "duration_s": 79.04,
    },
    "indoor_loadless_round": {
        "folder": "/indoor",
        "filename": "indoor_loadless_round_3096.1g_117.89s.bag",
        "size_bytes": 8770541296,
        "environment": "indoor",
        "trajectory": "round",
        "feature": "loadless",
        "mass_g": 3096.1,
        "duration_s": 117.89,
    },
    "indoor_loadless_8": {
        "folder": "/indoor",
        "filename": "indoor_loadless_8_3096.1g_109.17s.bag",
        "size_bytes": 8121789983,
        "environment": "indoor",
        "trajectory": "8-character",
        "feature": "loadless",
        "mass_g": 3096.1,
        "duration_s": 109.17,
    },
    "indoor_loaded_hovor": {
        "folder": "/indoor",
        "filename": "indoor_loaded_hovor_3103.2g_load269.0g_80.11s.bag",
        "size_bytes": 5960893648,
        "environment": "indoor",
        "trajectory": "hovor",
        "feature": "loaded",
        "mass_g": 3103.2,
        "payload_g": 269.0,
        "duration_s": 80.11,
    },
    "indoor_loaded_round": {
        "folder": "/indoor",
        "filename": "indoor_loaded_round_3103.2g_load269.0g_107.15s.bag",
        "size_bytes": 7973146411,
        "environment": "indoor",
        "trajectory": "round",
        "feature": "loaded",
        "mass_g": 3103.2,
        "payload_g": 269.0,
        "duration_s": 107.15,
    },
    "indoor_loaded_8": {
        "folder": "/indoor",
        "filename": "indoor_loaded_8_3103.2g_load269.0g_138.41s.bag",
        "size_bytes": 10299551893,
        "environment": "indoor",
        "trajectory": "8-character",
        "feature": "loaded",
        "mass_g": 3103.2,
        "payload_g": 269.0,
        "duration_s": 138.41,
    },
    "indoor_sensor_fast": {
        "folder": "/indoor",
        "filename": "indoor_sensor_fast_3101.5g_126.53s.bag",
        "size_bytes": 9423125175,
        "environment": "indoor",
        "trajectory": "random",
        "feature": "rope pulled (force sensor)",
        "mass_g": 3101.5,
        "duration_s": 126.53,
    },
    "indoor_sensor_slow": {
        "folder": "/indoor",
        "filename": "indoor_sensor_slow_3101.5g_155.38s.bag",
        "size_bytes": 11569450847,
        "environment": "indoor",
        "trajectory": "random",
        "feature": "rope pulled (force sensor)",
        "mass_g": 3101.5,
        "duration_s": 155.38,
    },
    "motortest_channel_1": {
        "folder": "/motortest",
        "filename": "channel_1.bag",
        "size_bytes": 5699189,
        "environment": None,
        "trajectory": "motor bench test",
        "feature": "single motor on a force sensor",
        "duration_s": 45.04,
        "bench": True,
    },
    "motortest_channel_2": {
        "folder": "/motortest",
        "filename": "channel_2.bag",
        "size_bytes": 5597992,
        "environment": None,
        "trajectory": "motor bench test",
        "feature": "single motor on a force sensor",
        "duration_s": 45.08,
        "bench": True,
    },
    "motortest_channel_3": {
        "folder": "/motortest",
        "filename": "channel_3.bag",
        "size_bytes": 5444610,
        "environment": None,
        "trajectory": "motor bench test",
        "feature": "single motor on a force sensor",
        "duration_s": 43.00,
        "bench": True,
    },
    "motortest_channel_4": {
        "folder": "/motortest",
        "filename": "channel_4.bag",
        "size_bytes": 5504758,
        "environment": None,
        "trajectory": "motor bench test",
        "feature": "single motor on a force sensor",
        "duration_s": 43.84,
        "bench": True,
    },
}

#: Sequences fetched by default: four flights covering indoor hover, outdoor
#: figure-8, flight with payload and rope-pulled random flight, plus the four
#: (tiny) single-motor bench runs.
DEFAULT_SEQUENCES: tuple[str, ...] = (
    "indoor_loadless_hovor",
    "indoor_loaded_8",
    "indoor_sensor_fast",
    "outdoor_8_yaw",
    "motortest_channel_1",
    "motortest_channel_2",
    "motortest_channel_3",
    "motortest_channel_4",
)

FLIGHT_RIG = "vid_m100"
BENCH_RIG = "vid_m3508_bench"


def share_url(folder: str, filename: str) -> str:
    """Direct download URL for one file in the publisher's QNAP share."""
    query = urllib.parse.urlencode(
        {
            "ssid": SHARE_SSID,
            "openfolder": "forcedownload",
            "ep": SHARE_EP,
            "fid": SHARE_SSID,
            "path": folder,
            "filename": filename,
        }
    )
    return f"{SHARE_CGI}?{query}"


# ─── Minimal rosbag1 reader (sequential, stream-friendly) ─────────────────────
# rosbags cannot read from a socket: it needs a seekable file, and these bags
# are 5-17 GB of mostly camera data that we refuse to store.  The v2.0 record
# format is simple enough to walk forwards: <hlen><header><dlen><data>.

OP_MSG = b"\x02"
OP_BAG_HEADER = b"\x03"
OP_CHUNK = b"\x05"
OP_CHUNK_INFO = b"\x06"
OP_CONNECTION = b"\x07"


def _parse_record_header(buf: bytes | memoryview) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    i, n = 0, len(buf)
    while i < n:
        (length,) = struct.unpack_from("<I", buf, i)
        i += 4
        key, _, value = bytes(buf[i : i + length]).partition(b"=")
        out[key.decode()] = value
        i += length
    return out


class _LocalBag:
    """Random/sequential access to a bag on disk."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.name = self.path.name
        self.size = self.path.stat().st_size

    def read_range(self, start: int, length: int) -> bytes:
        with self.path.open("rb") as fh:
            fh.seek(start)
            return fh.read(length)

    def reader(self, start: int, stop: int) -> _LocalReader:
        return _LocalReader(self.path, start, stop)


class _LocalReader:
    def __init__(self, path: Path, start: int, stop: int) -> None:
        self._fh = path.open("rb")
        self._fh.seek(start)
        self.pos = start
        self.stop = stop

    def read_exact(self, n: int) -> bytes:
        buf = self._fh.read(n)
        if len(buf) != n:
            raise EOFError(f"truncated bag at {self.pos} (wanted {n}, got {len(buf)})")
        self.pos += n
        return buf

    def skip(self, n: int) -> None:
        self._fh.seek(n, 1)
        self.pos += n

    def close(self) -> None:
        self._fh.close()


class _HttpBag:
    """Range-resumable HTTP access to a remote bag."""

    def __init__(self, url: str, *, name: str, timeout: float = 60.0, retries: int = 8) -> None:
        self.url = url
        self.name = name
        self.timeout = timeout
        self.retries = retries
        self.size = self._probe_size()

    def _open(self, start: int, stop: int | None = None):
        end = "" if stop is None else str(stop - 1)
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        return urllib.request.urlopen(req, timeout=self.timeout)

    def _probe_size(self) -> int:
        with self._open(0, 1) as resp:
            content_range = resp.headers.get("Content-Range", "")
            resp.read()
        total = content_range.rpartition("/")[2]
        if not total.isdigit():
            raise OSError(f"{self.name}: server did not report a size ({content_range!r})")
        return int(total)

    def read_range(self, start: int, length: int) -> bytes:
        last = "?"
        for attempt in range(self.retries):
            try:
                with self._open(start, start + length) as resp:
                    buf = resp.read()
                if len(buf) == length:
                    return buf
                last = f"short read ({len(buf)} of {length})"
            except OSError as exc:  # transient network/server hiccup
                last = repr(exc)
            time.sleep(min(2.0**attempt, 30.0))
        raise OSError(f"{self.name}: range {start}+{length} failed: {last}")

    def reader(self, start: int, stop: int) -> _HttpReader:
        return _HttpReader(self, start, stop)


class _HttpReader:
    """Forward-only reader over an HTTP range, reopening on connection loss."""

    def __init__(self, bag: _HttpBag, start: int, stop: int) -> None:
        self._bag = bag
        self.pos = start
        self.stop = stop
        self._resp: Any = None

    def _reopen(self) -> None:
        self.close()
        self._resp = self._bag._open(self.pos, self.stop)

    def read_exact(self, n: int) -> bytes:
        parts: list[bytes] = []
        need = n
        fails = 0
        while need:
            try:
                if self._resp is None:
                    self._reopen()
                block = self._resp.read(need)
            except OSError:
                block = b""
                self._resp = None
            if not block:
                self._resp = None
                fails += 1
                if fails > self._bag.retries:
                    raise OSError(f"{self._bag.name}: stream stalled at byte {self.pos}")
                time.sleep(min(2.0**fails, 30.0))
                continue
            fails = 0
            parts.append(block)
            need -= len(block)
            self.pos += len(block)
        return parts[0] if len(parts) == 1 else b"".join(parts)

    def skip(self, n: int) -> None:
        while n:
            step = min(n, 1 << 20)
            self.read_exact(step)
            n -= step

    def close(self) -> None:
        if self._resp is not None:
            try:
                self._resp.close()
            finally:
                self._resp = None


def _bag_directory(bag: _LocalBag | _HttpBag) -> dict[str, Any]:
    """``{first_chunk, index_pos, connections, counts}`` from head + index."""
    head = bag.read_range(0, 8192)
    if not head.startswith(b"#ROSBAG V2.0"):
        raise ValueError(f"{bag.name}: not a rosbag1 v2.0 file")
    off = head.index(b"\n") + 1
    (hlen,) = struct.unpack_from("<I", head, off)
    header = _parse_record_header(head[off + 4 : off + 4 + hlen])
    if header["op"] != OP_BAG_HEADER:
        raise ValueError(f"{bag.name}: first record is not a bag header")
    (dlen,) = struct.unpack_from("<I", head, off + 4 + hlen)
    index_pos = struct.unpack("<q", header["index_pos"])[0]
    if index_pos <= 0:
        raise ValueError(f"{bag.name}: unindexed bag (index_pos=0); reindex it first")

    tail = bag.read_range(index_pos, bag.size - index_pos)
    connections: dict[int, tuple[str, str]] = {}
    counts: dict[int, int] = {}
    i, n = 0, len(tail)
    while i + 8 <= n:
        (hlen_t,) = struct.unpack_from("<I", tail, i)
        rec = _parse_record_header(tail[i + 4 : i + 4 + hlen_t])
        i += 4 + hlen_t
        (dlen_t,) = struct.unpack_from("<I", tail, i)
        i += 4
        data = tail[i : i + dlen_t]
        i += dlen_t
        if rec["op"] == OP_CONNECTION:
            conn = struct.unpack("<i", rec["conn"])[0]
            fields = _parse_record_header(data)
            connections[conn] = (rec["topic"].decode(), fields["type"].decode())
        elif rec["op"] == OP_CHUNK_INFO:
            count = struct.unpack("<i", rec["count"])[0]
            for k in range(count):
                conn, cnt = struct.unpack_from("<ii", data, k * 8)
                counts[conn] = counts.get(conn, 0) + cnt
    return {
        "first_chunk": off + 8 + hlen + dlen,
        "index_pos": index_pos,
        "connections": connections,
        "counts": counts,
    }


def _decompress(compression: bytes, data: bytes) -> bytes:
    if compression == b"none":
        return data
    if compression == b"bz2":
        return bz2.decompress(data)
    raise NotImplementedError(f"unsupported rosbag chunk compression {compression!r}")


def _unpack_m3508(payload: bytes | memoryview) -> tuple[int, int, int, int]:
    """``(stamp_ns, hwts_ns, current, rpm)`` from a ``uavmotor/m3508`` message."""
    _seq, sec, nsec, flen = M3508_STRUCT.unpack_from(payload, 0)
    hsec, hnsec, _mid, current, rpm = M3508_TAIL.unpack_from(payload, 16 + flen)
    return sec * 1_000_000_000 + nsec, hsec * 1_000_000_000 + hnsec, current, rpm


def _unpack_motordata(payload: bytes | memoryview) -> tuple[int, int, tuple[int, ...]]:
    """``(stamp_ns, hwts_ns, int16[4])`` from a ``uavmotor/motordata`` message."""
    _seq, sec, nsec, flen = M3508_STRUCT.unpack_from(payload, 0)
    hsec, hnsec, *values = MOTORDATA_TAIL.unpack_from(payload, 16 + flen)
    return sec * 1_000_000_000 + nsec, hsec * 1_000_000_000 + hnsec, tuple(values)


def distill_motor_topics(bag: _LocalBag | _HttpBag, *, progress: bool = False) -> dict[str, Any]:
    """Walk a bag once and return only its rotor-speed records as arrays.

    Chunks are read in file order; every non-motor message (depth/infra
    images, IMU, RTK, Vicon, force sensor) is stepped over without being
    decoded, so peak memory is one chunk (~1.3 MB) plus the motor arrays.
    """
    directory = _bag_directory(bag)
    conns = directory["connections"]
    motor_conn: dict[int, int] = {}
    target_conn: set[int] = set()
    for conn, (topic, msgtype) in conns.items():
        match = MOTOR_TOPIC_RE.search(topic)
        if match and msgtype.endswith("m3508"):
            motor_conn[conn] = int(match.group(1))
        elif TARGET_TOPIC_RE.search(topic) and msgtype.endswith("motordata"):
            target_conn.add(conn)
    if not motor_conn:
        raise ValueError(f"{bag.name}: no m3508_m* topics (found {sorted(conns.values())})")
    wanted = set(motor_conn) | target_conn

    motors: dict[int, list[tuple[int, int, int, int]]] = {r: [] for r in motor_conn.values()}
    target: list[tuple[int, int, tuple[int, ...]]] = []

    reader = bag.reader(directory["first_chunk"], directory["index_pos"])
    started = time.monotonic()
    try:
        while reader.pos < directory["index_pos"]:
            (hlen,) = struct.unpack("<I", reader.read_exact(4))
            header = _parse_record_header(reader.read_exact(hlen))
            (dlen,) = struct.unpack("<I", reader.read_exact(4))
            if header["op"] != OP_CHUNK:
                reader.skip(dlen)
                continue
            chunk = _decompress(header["compression"], reader.read_exact(dlen))
            view = memoryview(chunk)
            i, n = 0, len(chunk)
            while i + 8 <= n:
                (ihlen,) = struct.unpack_from("<I", view, i)
                irec = _parse_record_header(view[i + 4 : i + 4 + ihlen])
                i += 4 + ihlen
                (idlen,) = struct.unpack_from("<I", view, i)
                i += 4
                if irec["op"] == OP_MSG:
                    conn = struct.unpack("<i", irec["conn"])[0]
                    if conn in wanted:
                        payload = view[i : i + idlen]
                        if conn in target_conn:
                            target.append(_unpack_motordata(payload))
                        else:
                            motors[motor_conn[conn]].append(_unpack_m3508(payload))
                i += idlen
            if progress and reader.pos % (1 << 28) < (1 << 21):
                done = reader.pos - directory["first_chunk"]
                span = directory["index_pos"] - directory["first_chunk"]
                rate = done / max(time.monotonic() - started, 1e-9) / 2**20
                print(f"  {bag.name}: {100 * done / span:5.1f}%  {rate:5.1f} MB/s", flush=True)
    finally:
        reader.close()

    out: dict[str, Any] = {}
    for rotor, rows in sorted(motors.items()):
        arr = np.array(rows, dtype=np.int64)  # (T, 4): stamp, hwts, current, rpm
        out[f"m{rotor}_stamp_ns"] = arr[:, 0]
        out[f"m{rotor}_hwts_ns"] = arr[:, 1]
        out[f"m{rotor}_current"] = arr[:, 2].astype(np.int16)
        out[f"m{rotor}_rpm"] = arr[:, 3].astype(np.int16)
    if target:
        out["target_stamp_ns"] = np.array([r[0] for r in target], dtype=np.int64)
        out["target_hwts_ns"] = np.array([r[1] for r in target], dtype=np.int64)
        out["target_rpm"] = np.array([r[2] for r in target], dtype=np.int16)
    out["meta"] = json.dumps(
        {
            "source_file": bag.name,
            "source_bytes": bag.size,
            "topics": {conns[c][0]: directory["counts"].get(c, 0) for c in sorted(wanted)},
            "rotors": sorted(motors),
        }
    )
    return out


# ─── Raw fetch (streaming distillation) ───────────────────────────────────────


def download_vid(
    dest: Path, sequences: Iterable[str] = DEFAULT_SEQUENCES, *, progress: bool = True
) -> Path:
    """Stream the selected VID bags and write their motor records to ``dest``.

    Idempotent: a sequence whose ``<name>.motors.npz`` already exists is
    skipped.  Only ~5 MB per sequence lands on disk; the 5-17 GB of camera
    data in each bag is discarded as it streams.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in sequences:
        if name not in SEQUENCES:
            raise KeyError(f"unknown VID sequence {name!r}")
        out_path = dest / f"{name}.motors.npz"
        if out_path.exists():
            continue
        spec = SEQUENCES[name]
        bag = _HttpBag(share_url(spec["folder"], spec["filename"]), name=str(spec["filename"]))
        if progress:
            print(f"streaming {name} ({bag.size / 2**30:.2f} GiB)", flush=True)
        arrays = distill_motor_topics(bag, progress=progress)
        tmp = out_path.with_name(out_path.name + ".part")
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        tmp.replace(out_path)
    return dest


# ─── Frame building ───────────────────────────────────────────────────────────


def _trim_nan_edges(values: np.ndarray, stamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop leading/trailing all-NaN columns; keep interior NaNs."""
    good = ~np.all(np.isnan(values), axis=0)
    if not good.any():
        raise ValueError("all rotor-speed samples are NaN")
    first, last = int(np.argmax(good)), len(good) - int(np.argmax(good[::-1]))
    return values[:, first:last], stamps[first:last]


def _hwts_clock(hwts_ns: np.ndarray, *, label: str) -> tuple[np.ndarray, np.ndarray]:
    """``(seconds, keep_mask)`` for a topic's hardware stamps.

    The onboard MCU builds ``hwts`` from a PPS-locked seconds counter plus a
    free-running timer, and the pairing occasionally tears:

    - roughly once per 30 000 samples the timer wraps one sample before the
      seconds field increments, so a single stamp sits exactly ~1 s in the
      past — that one is repairable, and repaired;
    - much more rarely a stamp is simply garbage (sub-microsecond junk, ~1.3 s
      into the future) — that sample cannot be placed on the timeline, so it
      is dropped via ``keep_mask``.

    A clock that is still non-monotone afterwards is a layout problem, not a
    glitch this function understands, and raises.
    """
    stamps = hwts_ns.astype(np.int64).copy()
    (dips,) = np.nonzero(np.diff(stamps) < 0)
    for i in dips:
        j = i + 1
        rolled = stamps[j] + 1_000_000_000
        if rolled > stamps[i] and (j + 1 >= stamps.size or rolled <= stamps[j + 1]):
            stamps[j] = rolled
    keep = np.ones(stamps.size, dtype=bool)
    (descents,) = np.nonzero(np.diff(stamps) < 0)
    for i in descents:
        # Blame whichever of the pair disagrees with the surrounding clock.
        before = stamps[i - 1] if i else np.int64(-1)
        keep[i if before <= stamps[i + 1] else i + 1] = False
    if descents.size and np.any(np.diff(stamps[keep]) < 0):
        raise ValueError(
            f"{label}: hardware clock still non-monotone after repairing "
            f"{descents.size} discontinuities"
        )
    return stamps.astype(np.float64) / 1e9, keep


def _rotor_series(
    data: dict[str, np.ndarray], rotor: int, *, label: str
) -> tuple[np.ndarray, np.ndarray, int]:
    """``(t_seconds, rev/s, n_glitches)`` of one rotor on its hardware clock.

    A handful of CAN feedback frames per flight arrive corrupt and read as a
    single-sample impulse of 100-500 rev/s.  A rotor cannot change by more
    than a few rev/s in one 1 ms sample (the measured worst real deviation
    from a 5-sample median is 4.4 rev/s, over bench step responses), so
    samples beyond :data:`GLITCH_TOL_RPS` are unknowable and become NaN.  The
    untouched counts stay in the raw tree's npz.
    """
    stamps, keep = _hwts_clock(data[f"m{rotor}_hwts_ns"], label=f"{label} m{rotor}")
    values = data[f"m{rotor}_rpm"].astype(np.float64) * RPM_TO_RPS
    stamps, values = stamps[keep], values[keep]
    glitches = np.abs(values - median_filter(values, size=5, mode="nearest")) > GLITCH_TOL_RPS
    values[glitches] = np.nan
    return stamps, values, int(glitches.sum())


def _merge_rotors(
    data: dict[str, np.ndarray], rotors: list[int], *, label: str
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """``(rps (n_rotors, T), t_seconds, worst_clock_offset_s, n_glitches)``.

    Each ESC is polled independently, so the four topics carry their own
    stamps.  They agree to well under one native period, so rotors 2..N are
    linearly interpolated onto rotor 1's clock; a sample whose nearest source
    stamp is farther than :data:`GAP_TOL_S` away (a telemetry dropout) stays
    NaN rather than being bridged, and NaN glitches propagate into their own
    immediate neighbourhood rather than being smoothed over.
    """
    base, first_row, glitches = _rotor_series(data, rotors[0], label=label)
    rows = np.empty((len(rotors), base.size), dtype=np.float64)
    rows[0] = first_row
    worst = 0.0
    for k, rotor in enumerate(rotors[1:], start=1):
        stamps, values, n_glitch = _rotor_series(data, rotor, label=label)
        glitches += n_glitch
        idx = np.clip(np.searchsorted(stamps, base), 1, stamps.size - 1)
        nearest = np.abs(np.stack([stamps[idx] - base, stamps[idx - 1] - base])).min(axis=0)
        typical = float(np.median(nearest))
        worst = max(worst, typical)
        if typical > STAMP_TOL_S:
            raise ValueError(
                f"{label}: rotor {rotor} clock sits {1e3 * typical:.2f} ms off rotor "
                f"{rotors[0]} (tolerance {1e3 * STAMP_TOL_S:.2f} ms); the topics cannot "
                "share a clock"
            )
        row = np.interp(base, stamps, values)
        row[nearest > GAP_TOL_S] = np.nan
        rows[k] = row
    rps, stamps_s = _trim_nan_edges(rows, base)
    return rps, stamps_s - stamps_s[0], worst, glitches


def _frame_from_arrays(name: str, data: dict[str, np.ndarray], dataset: str) -> td.Frame:
    spec = SEQUENCES[name]
    meta_json = json.loads(str(data["meta"]))
    rotors = [int(r) for r in meta_json["rotors"]]
    rps, stamps, clock_offset, glitches = _merge_rotors(data, rotors, label=name)
    duration = float(stamps[-1] - stamps[0])
    bench = bool(spec.get("bench"))

    entries: dict[str, Any] = {"rps": td.events(stamps, rps, dims=("rotor", "time"))}
    if "target_rpm" in data:
        target_t, keep = _hwts_clock(data["target_hwts_ns"], label=f"{name} target_rpm")
        # ``int16[4]`` is ordered m1..m4; keep the rotors this recording has.
        commanded = data["target_rpm"][:, [r - 1 for r in rotors]].astype(np.float64)
        entries["rps_command"] = td.events(
            target_t[keep] - target_t[keep][0],
            commanded[keep].T * RPM_TO_RPS,
            dims=("rotor", "time"),
        )
    entries["meta"] = meta_frame(
        recording_id=name,
        dataset=dataset,
        system={
            "rig": BENCH_RIG if bench else FLIGHT_RIG,
            "vehicle": (
                "DJI RoboMaster M3508 motor + C620 ESC on a thrust/force-sensor bench"
                if bench
                else "DJI M100-derived, M3508 motors"
            ),
            "n_rotors": len(rotors),
            "rotor_layout": None,  # publisher does not state rotor positions
            "speed_source": "esc_feedback",
            "native_rate_hz": NATIVE_RATE_HZ,
        },
        operating={
            "trajectory": spec["trajectory"],
            "duration_s": duration,
            "max_speed_mps": None,
            "environment": spec["environment"],
        },
        extra={
            "units_note": (
                "rev/s = C620 ESC CAN feedback 'rpm' field / 60 (the publisher's "
                "rpmconvert tool copies this field unscaled into /synced/allrpm); "
                "timestamps are the MCU hardware clock (hwts), rotors 2..N "
                "interpolated onto rotor 1's stamps"
            ),
            "source_file": f"{spec['folder'].lstrip('/')}/{spec['filename']}",
            "feature": spec["feature"],
            "mass_g": spec.get("mass_g"),
            "payload_g": spec.get("payload_g"),
            "lighting": spec.get("lighting"),
            "published_duration_s": spec["duration_s"],
            "rotor_clock_offset_s": clock_offset,
            "glitch_samples": glitches,
        },
    )
    return td.Frame(entries)


def build(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """Yield ``(key, frame)`` for every distilled sequence in the raw tree.

    One flight is held in memory at a time; order follows the sorted file
    names, so the output is deterministic.
    """
    raw_dir = Path(raw_dir)
    paths = sorted(raw_dir.rglob("*.motors.npz"))
    if not paths:
        raise FileNotFoundError(f"no *.motors.npz under {raw_dir} - run download_vid first")
    for path in paths:
        name = path.name[: -len(".motors.npz")]
        if name not in SEQUENCES:
            raise KeyError(f"{path}: unknown VID sequence {name!r}")
        with np.load(path, allow_pickle=False) as handle:
            data = {k: handle[k] for k in handle.files}
        frame = _frame_from_arrays(name, data, dataset="VID")
        rig = BENCH_RIG if SEQUENCES[name].get("bench") else FLIGHT_RIG
        yield safe_key(f"{rig}__{name}"), frame


# ─── Registry provenance ──────────────────────────────────────────────────────
# (entry assembled in sources/__init__.py; raw fetch is the custom
# download_vid - the published bags bundle 5-17 GB of camera streams, so the
# motor topics are extracted while streaming and never stored)

PROVENANCE: dict[str, Any] = {
    "source_url": "https://github.com/ZJU-FAST-Lab/VID-Dataset",
    "license": "AGPL-3.0 (tools); dataset released for research use",
    "citation": (
        "Ding et al., VID-Fusion: Robust Visual-Inertial-Dynamics Odometry for "
        "Accurate External Force Estimation, ICRA 2021 (arXiv:2103.11152)."
    ),
    "description": (
        "VID per-rotor speed telemetry as td.Frames: measured rev/s from the four "
        "M3508/C620 ESC CAN feedback topics (~1 kHz, hardware-stamped) on their "
        "native event clock, plus commanded rev/s (target_rpm) and ESC current, "
        "for four flights (indoor hover, indoor loaded 8-character, indoor "
        "rope-pulled random, outdoor 8-character with yaw) and four single-motor "
        "bench runs. Camera/IMU/RTK/Vicon records are skipped during fetch."
    ),
    "native_rate_hz": NATIVE_RATE_HZ,
    "n_rotors": NUM_ROTORS,
}
