"""PI-TCN source — 100 Hz per-rotor ESC RPM telemetry of 68 agile quad flights.

Physics-Inspired Temporal Learning of Quadrotor Dynamics for Accurate Model
Predictive Trajectory Tracking
A. Saviolo, G. Li, G. Loianno, IEEE RA-L 7(4):10256-10263, 2022.
Paper: https://arxiv.org/abs/2206.03305 (doi 10.1109/LRA.2022.3192609)
Code / dataset README: https://github.com/arplaboratory/PI-TCN

68 trajectories (61.4 min) of one ARPL quadrotor (namespace ``dragonfly17``)
flown indoors in a 10x6x4 m Vicon room at NYU: straight-line accelerations,
circles, ovals, parabolas and lemniscates, each flown for several axis
combinations and aggressiveness settings. Each trajectory is one ROS1 bag with
the full onboard topic set; this builder reads **only** ``*/motor_rpm``.

Raw-source note — the dataset links printed in the PI-TCN README (Google Drive
file ids ``1b1PFSBlKTdrlTIurYNpTJWWEx1KIJzuR`` for the bags and
``1s7nSqATpCS849csSdkHNL0-VLZwdNzg4`` for the pdf illustrations) are **dead**:
both return HTTP 404 "Page not found" from every Drive endpoint (checked
2026-09-15), i.e. the files were removed rather than quota-limited. The same 68
bags are redistributed inside the ``data.zip`` of the authors' follow-up repo
https://github.com/arplaboratory/long-horizon-dynamics (Drive file id
:data:`DATA_ZIP_FILE_ID`, 2.04 GB, live), under ``data/pi_tcn/rosbags/``, and
that is what :func:`download_pitcn` fetches. Only those 68 bag members are
extracted; the zip's ``data/neurobem/`` half and the authors' derived csv splits
are skipped (NeuroBEM has its own registry entry, fed from the UZH release).

Rotor speeds come from ``/dragonfly17/motor_rpm``
(``quadrotor_msgs/MotorSpeed``: ``Header header; float64[4] rpm;
quadrotor_msgs/AuxCommand aux``) at a rigid 100 Hz, and are **ESC feedback**,
not the controller's command — see :data:`PROVENANCE` for the four pieces of
evidence. The field is named ``rpm`` and is RPM, so ``rps = rpm / 60``. One
:class:`td.Frame` per bag, ``rps`` + ``meta`` only (no audio).
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import tdseries as td

from data_processing.sources._common import meta_frame, safe_key

# ─── Raw fetch ────────────────────────────────────────────────────────────────

#: Google Drive file id of ``data.zip`` in arplaboratory/long-horizon-dynamics.
DATA_ZIP_FILE_ID = "1BB-r63qgiqB5uJ5xVbTcfR-6j9rXCFCA"
#: Size of that ``data.zip`` as published (checked 2026-09-15).
DATA_ZIP_BYTES = 2043979141
DATA_ZIP_NAME = "data.zip"
#: Zip prefix holding the 68 original PI-TCN bags.
ZIP_BAG_PREFIX = "data/pi_tcn/rosbags/"
#: Directory the bags are extracted into, relative to the raw root.
BAGS_DIRNAME = "rosbags"
EXTRACTED_MARKER = ".extracted"
_DRIVE_URL = "https://drive.usercontent.google.com/download"
_HIDDEN_INPUT_RE = re.compile(r'<input\s+type="hidden"\s+name="([^"]+)"\s+value="([^"]*)"')
_CHUNK = 1 << 22

# ─── Dataset facts ────────────────────────────────────────────────────────────

RIG = "pitcn_quad"
N_ROTORS = 4
#: Native telemetry rate: the ``motor_rpm`` header clock fits 100.0000 Hz on
#: every bag (least-squares rate within 2.5e-4 Hz, no drift). See ``_rps_series``.
RATE_HZ = 100.0
#: Topic suffix carrying the rotor speeds (namespace is ``/dragonfly17``).
MOTOR_TOPIC_SUFFIX = "/motor_rpm"
#: Rotor order of the ``rpm[4]`` field, derived from the nominal torque map in
#: the publisher's ``process_data.py`` (see ``PROVENANCE['rotor_layout_note']``).
ROTOR_LAYOUT = ["front_left", "back_left", "back_right", "front_right"]
#: Accept ``td.uniform`` while the stamp scatter stays this small a fraction of
#: the period (std of the inter-arrival times over the median period).
MAX_RATE_JITTER = 0.05
#: ...and while no stamp is further than this many periods off the fitted grid.
MAX_GRID_DRIFT_PERIODS = 3.0
#: ``gt_*`` bags: the five longest recordings (106-226 s), undocumented prefix.
_GT_PREFIX = "gt_"


def _requests() -> Any:
    import requests  # deferred: transitively present, not a hard import-time dep

    return requests


def _fetch_drive_file(file_id: str, dest_path: Path, *, expected_size: int | None) -> Path:
    """Stream a public Drive *file* to ``dest_path``, passing the scan warning.

    ``gdown`` cannot fetch this file (it fails on its own public test file in
    this environment), so the two-step flow is done directly: the first GET
    answers with the "Virus scan warning" html whose hidden form fields
    (``confirm``, ``uuid``, ...) the second GET replays.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if (
        dest_path.exists()
        and expected_size is not None
        and dest_path.stat().st_size == expected_size
    ):
        return dest_path
    requests = _requests()
    with requests.Session() as session:
        params: dict[str, str] = {"id": file_id, "export": "download"}
        resp = session.get(_DRIVE_URL, params=params, stream=True, timeout=60)
        resp.raise_for_status()
        if "text/html" in resp.headers.get("content-type", ""):
            fields = dict(_HIDDEN_INPUT_RE.findall(resp.text))
            resp.close()
            if "confirm" not in fields:
                raise RuntimeError(
                    f"Google Drive refused file {file_id} without a confirm form; "
                    "the file may have been removed or rate-limited"
                )
            resp = session.get(_DRIVE_URL, params=fields, stream=True, timeout=60)
            resp.raise_for_status()
        with resp:
            tmp = dest_path.with_suffix(dest_path.suffix + ".part")
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=_CHUNK):
                    if chunk:
                        fh.write(chunk)
            tmp.replace(dest_path)
    return dest_path


def download_pitcn(dest: Path) -> Path:
    """Fetch the 68 PI-TCN bags into ``dest/rosbags/`` (idempotent, ~3.0 GB).

    Downloads the 2.04 GB redistribution zip, extracts only its
    ``data/pi_tcn/rosbags/*.bag`` members, then deletes the zip again so the
    raw tree stays at the size of the bags themselves.
    """
    dest = Path(dest)
    bags_dir = dest / BAGS_DIRNAME
    if (bags_dir / EXTRACTED_MARKER).exists():
        return dest
    zip_path = _fetch_drive_file(
        DATA_ZIP_FILE_ID, dest / DATA_ZIP_NAME, expected_size=DATA_ZIP_BYTES
    )
    bags_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            info
            for info in zf.infolist()
            if info.filename.startswith(ZIP_BAG_PREFIX)
            and not info.is_dir()
            and info.filename.endswith(".bag")
        ]
        if not members:
            raise RuntimeError(f"{zip_path} holds no {ZIP_BAG_PREFIX}*.bag members")
        for info in members:
            target = bags_dir / Path(info.filename).name
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            with zf.open(info) as src, open(target, "wb") as fh:
                while chunk := src.read(_CHUNK):
                    fh.write(chunk)
    (bags_dir / EXTRACTED_MARKER).write_text(f"{len(members)} bags from {DATA_ZIP_FILE_ID}\n")
    zip_path.unlink(missing_ok=True)
    return dest


# ─── Bag reading ──────────────────────────────────────────────────────────────


def _iter_bags(raw_dir: Path) -> Iterator[Path]:
    """Every PI-TCN bag in the raw tree, in deterministic (sorted) order."""
    raw_dir = Path(raw_dir)
    bags_dir = raw_dir / BAGS_DIRNAME
    root = bags_dir if bags_dir.is_dir() else raw_dir
    yield from sorted(path for path in root.rglob("*.bag") if path.is_file())


def _read_bag(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """``((4, T) rev/s, (T,) seconds-from-first-sample, topic)`` from one bag.

    ``rosbags`` is imported here, not at module scope: it is only needed to
    ingest this one dataset, and the rest of the package must import without it.
    """
    from rosbags.highlevel import AnyReader  # lazy: optional ingest-only dep

    stamps: list[float] = []
    speeds: list[np.ndarray] = []
    with AnyReader([Path(path)]) as reader:
        connections = [c for c in reader.connections if c.topic.endswith(MOTOR_TOPIC_SUFFIX)]
        if not connections:
            return np.empty((N_ROTORS, 0)), np.empty(0), ""
        topic = sorted({c.topic for c in connections})[0]
        for connection, _, rawdata in reader.messages(connections=connections):
            msg = cast(Any, reader.deserialize(rawdata, connection.msgtype))
            stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            speeds.append(np.asarray(msg.rpm, dtype=np.float64))

    rps = np.asarray(speeds, dtype=np.float64).T / 60.0  # (n_rotors, T), RPM -> rev/s
    t = np.asarray(stamps, dtype=np.float64)

    keep = ~np.all(np.isnan(rps), axis=0)
    if not keep.any():
        return rps[:, :0], t[:0], topic
    first, last = int(np.argmax(keep)), int(len(keep) - np.argmax(keep[::-1]))
    rps = np.ascontiguousarray(rps[:, first:last])
    t = t[first:last]
    return rps, t - t[0], topic


def _rps_series(rps: np.ndarray, t: np.ndarray) -> tuple[td.Series, float, float]:
    """``(series, rate, stamp jitter)`` for one flight's rotor speeds.

    The ``motor_rpm`` stamps are ROS receive times, so they carry a few ms of
    scheduling scatter, but the underlying telemetry clock is rigid: on every
    published bag the least-squares rate is 100.0000 Hz and no stamp sits more
    than 1.4 periods off the grid anchored at the first sample, over flights of
    up to 22589 samples. Both of those are re-checked here, and a bag that
    failed either would be stored on its own (event) clock instead.
    """
    if len(t) < 2:
        return td.uniform(rps, RATE_HZ, dims=("rotor", "time")), RATE_HZ, 0.0
    dt = np.diff(t)
    period = float(np.median(dt))
    jitter = float(np.std(dt)) / period
    grid = t[0] + np.arange(len(t)) / RATE_HZ
    drift = float(np.max(np.abs(t - grid))) * RATE_HZ
    if jitter <= MAX_RATE_JITTER and drift <= MAX_GRID_DRIFT_PERIODS:
        return td.uniform(rps, RATE_HZ, dims=("rotor", "time")), RATE_HZ, jitter
    return td.events(t, rps, dims=("rotor", "time")), 1.0 / period, jitter


def _trajectory_parts(stem: str) -> tuple[str, str | None]:
    """``(family, param)`` of a bag name.

    The publisher's names are ``[gt_]<family>[_<numbers>]``, where the family
    may itself be numeric (``8`` is the figure-eight): ``circlez_2_5`` ->
    ``("circlez", "2_5")``, ``8_4_5`` -> ``("8", "4_5")``, ``gt_v_5`` ->
    ``("gt_v", "5")``, ``gt_goto`` -> ``("gt_goto", None)``.
    """
    prefix, rest = "", stem
    if stem.startswith(_GT_PREFIX):
        prefix, rest = _GT_PREFIX, stem[len(_GT_PREFIX) :]
    head, _, tail = rest.partition("_")
    return f"{prefix}{head}", tail or None


# ─── Builder (raw tree -> tdframe-v1 telemetry Frames) ───────────────────────


def build(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """Yield ``(key, frame)`` with ``rps`` [rev/s] + ``meta`` per flight bag."""
    raw_dir = Path(raw_dir)
    for path in _iter_bags(raw_dir):
        flight = path.stem
        rps, t, topic = _read_bag(path)
        if rps.shape[1] == 0:
            print(f"Warning: skipping {path.name} — no {MOTOR_TOPIC_SUFFIX} samples")
            continue
        series, rate, jitter = _rps_series(rps, t)
        family, param = _trajectory_parts(flight)
        meta = meta_frame(
            recording_id=flight,
            dataset="PI-TCN",
            system={
                "rig": RIG,
                "vehicle": (
                    "ARPL DragonFly17 quadrotor, 0.25 kg, 0.076 m arm "
                    "(mass/arm/inertia from the publisher's process_data.py)"
                ),
                "n_rotors": int(rps.shape[0]),
                "rotor_layout": list(ROTOR_LAYOUT),
                "speed_source": "esc_feedback",
                "native_rate_hz": float(rate),
            },
            operating={
                "trajectory": flight,
                "duration_s": float(t[-1] - t[0]),
                "max_speed_mps": None,  # no speed is documented per trajectory
                "environment": "indoor",
            },
            extra={
                "units_note": (
                    f"{topic or MOTOR_TOPIC_SUFFIX} quadrotor_msgs/MotorSpeed "
                    "'float64[4] rpm' [RPM] / 60"
                ),
                "source_file": str(path.relative_to(raw_dir)),
                "trajectory_family": family,
                "trajectory_param": param,
                "stamp_jitter": float(jitter),
            },
        )
        yield safe_key(f"{RIG}__{flight}"), td.Frame({"rps": series, "meta": meta})


# ─── Registry provenance ──────────────────────────────────────────────────────

PROVENANCE: dict[str, Any] = {
    "source_url": "https://github.com/arplaboratory/PI-TCN",
    "download_url": f"https://drive.google.com/file/d/{DATA_ZIP_FILE_ID}/view",
    "doi": "10.1109/LRA.2022.3192609",
    "license": (
        "no dataset licence is published; the PI-TCN code repository that releases "
        "the bags is GPL-3.0 and asks that the RA-L 2022 paper be cited, and the "
        "long-horizon-dynamics repository the zip is fetched from carries no licence "
        "file at all"
    ),
    "citation": (
        "A. Saviolo, G. Li, G. Loianno, Physics-Inspired Temporal Learning of "
        "Quadrotor Dynamics for Accurate Model Predictive Trajectory Tracking, "
        "IEEE Robotics and Automation Letters, 7(4):10256-10263, 2022."
    ),
    "description": (
        "Per-rotor speed telemetry of one ARPL quadrotor (ROS namespace "
        "'dragonfly17', 0.25 kg, 0.076 m arm) flown indoors in a 10x6x4 m Vicon room "
        "at NYU: 68 trajectory bags, 61.4 min, 100 Hz. Frames carry rps (rev/s) and "
        "meta only; no audio. operating.trajectory is the bag name verbatim (the "
        "publisher's only label); extra.trajectory_family/param split it, e.g. "
        "'circlez_2_5' -> family 'circlez', param '2_5'. The families are 8, 8z, "
        "circle, circlez, line, linez, line8z, oval, ovalz, v, vz, vT, w, wz plus "
        "five long 'gt_*' recordings (106-226 s); the trailing numeric parameter is "
        "undocumented but is inversely related to aggressiveness (flight duration "
        "grows and mean rotor speed falls as it grows, e.g. circle_2 is 17.6 s at "
        "225.8 rev/s mean and circle_10 is 64.8 s at 198.4 rev/s), so it is NOT "
        "recorded as max_speed_mps, which stays None."
    ),
    "units_note": (
        "rps = rpm / 60, from the 'float64[4] rpm' field of "
        "quadrotor_msgs/MotorSpeed on /dragonfly17/motor_rpm. RPM is confirmed four "
        "ways: (1) the message field is named 'rpm' "
        "(arplaboratory/arpl_msgs quadrotor_msgs/msg/MotorSpeed.msg) and the "
        "publisher's process_data.py reads the columns as 'rpm_0..rpm_3'; (2) the "
        "paper's Fig. 4 axes give the thrust constant k_f in 'g rpm^-2'; (3) with "
        "process_data.py's k_f = 4.379e-9 and m = 0.25 kg, hover needs "
        "sqrt(mg/4/k_f) = 11832 RPM = 197 rev/s, and the measured per-flight means "
        "are 185-230 rev/s (min 87, max 290 rev/s over all 68 bags); (4) the "
        "authors' follow-up loader scales these columns by 0.001 to get O(10) network "
        "inputs, i.e. the raw values are O(10^4)."
    ),
    "speed_source_note": (
        "speed_source='esc_feedback'. The README states the onboard motor speeds are "
        "recorded and that process_data.py takes 'rotor speeds from ESC'. Three "
        "further checks rule out the controller command: (a) the AuxCommand 'aux' "
        "field of every motor_rpm message is entirely zero/False while the same "
        "field on the actual command topic /dragonfly17/so3_cmd is populated "
        "(enable_motors=True, current_yaw != 0), so the two topics have different "
        "producers; (b) every rpm value in every bag is a multiple of 4 RPM, i.e. a "
        "decoded fixed-point telemetry field, which a float command would not be; "
        "(c) k_f * sum(rpm^2) correlates only 0.77 with the commanded thrust "
        "|so3_cmd.force| (median relative error 8%, p95 43%) instead of matching it "
        "deterministically. No pole-pair factor applies: the field is already RPM."
    ),
    "rotor_layout_note": (
        "The publisher never states rotor positions. The layout recorded here "
        "(rpm[0] front-left, rpm[1] back-left, rpm[2] back-right, rpm[3] "
        "front-right) is derived from the nominal torque map in process_data.py, "
        "tau = [l(f0+f1-f2-f3), l(-f0+f1+f2-f3), (k/k_f)(f0-f1+f2-f3)], read in an "
        "x-forward/y-left/z-up body frame (the same file's vdot uses z-up gravity). "
        "The yaw row's pairing of {0,2} against {1,3} confirms the two diagonals but "
        "not which diagonal is which, so treat the front/back and left/right names "
        "as derived, not published."
    ),
    "access_note": (
        "The two Google Drive links in the PI-TCN README (file ids "
        "1b1PFSBlKTdrlTIurYNpTJWWEx1KIJzuR for the bags, "
        "1s7nSqATpCS849csSdkHNL0-VLZwdNzg4 for the pdf illustrations) both return "
        "HTTP 404 'Page not found' from /file/d/<id>/view, /uc?id=<id> and "
        "drive.usercontent.google.com/download (checked 2026-09-15), so the files are "
        "gone rather than quota-limited; a known-public control file still returns "
        "200 through the same path. The bags are therefore fetched from the "
        f"data.zip (id {DATA_ZIP_FILE_ID}, {DATA_ZIP_BYTES} B) of the authors' "
        "follow-up repository arplaboratory/long-horizon-dynamics, whose "
        "data/pi_tcn/rosbags/ holds all 68 bags with their original names and the "
        "full onboard topic set (68 trajectories and 61.4 min match the README's 68 "
        "trajectories / 58'03''); only those members are extracted."
    ),
    "sample_rate": RATE_HZ,
    "channels": N_ROTORS,
}
