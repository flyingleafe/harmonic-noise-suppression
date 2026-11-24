"""November 2026 M100 recordings (``new_motor_data``) as lossless ``tdframe-v1`` Frames.

One Frame per recording (FLY013/FLY018/FLY022), keyed by the DAT stem:

- ``audio`` — the session's stereo ZOOM WAV, ``(mic, time)`` float32 at the
  file rate, ``t_start`` 0. Integer PCM divided by full scale (soundfile), so
  PCM_16/PCM_24 values are exact; other encodings are refused.
- ``rps`` — ``(rotor, time)`` float64 rev/s: EVERY native ``Motor`` packet
  (record 10090, 152 B) speed in RPM / 60, rotors in :data:`ROTORS` order.
  Packets before/after the WAV, late packets and duplicate ticks are kept.
- ``dat`` — nested Frame, one sub-Frame per native group of
  :func:`~data_processing.sources.new_motor_data.read_aligned_motor` (the
  ``DatCon`` CSV export is not copied; it stays in the raw parent). Rows are
  events on the calibrated clock ``audio_s = A + tick / f``
  (:data:`~data_processing.sources.new_motor_data.AUDIO_CLOCK`), rounded to
  tdseries ns and ordered by ``(tick, offset)`` (``StampIndex`` must be
  sorted; ``offset`` restores file order). Each row is ONE structured-dtype
  record, so a group has a single stamp vector:

  * decoded groups (builtin and/or firmware layout): entry ``rows`` (or
    ``rows_<width>`` per payload width when a group has several) with
    exactly the columns and dtypes of ``read_aligned_motor(dat)[group]``
    except ``audio_s``: ``tick`` int64 (unwrapped), ``tick_raw`` uint32,
    ``offset`` int64 (packet byte offset), ``in_order`` bool, the packed
    fields, engineered columns (``utc`` datetime64[s] with NaT, ...);
  * text (``EventLog``, ``RecDefs``, ...) and undocumented ``record_<type>``
    groups: ``rows`` with ``tick``/``tick_raw``/``offset``/``in_order``/
    ``payload_start``/``payload_length`` into the invariant uint8 ``payload``
    stream (XOR-decoded payload bytes, file order, NUL padding included).

- ``dat_bytes`` — the whole original DAT file, invariant uint8: headers,
  CRCs, XOR-encoded payloads, invalid ranges and padding byte for byte
  (:func:`restore_dat`).
- ``meta`` — files (relpath, size, sha256), WAV encoding, clock calibration,
  DAT container statistics and the per-group field layout (``dat_layout``).

:func:`group_to_table` turns a group back into the exact
``read_aligned_motor`` table (row order, dtypes, NaN payloads, messages,
opaque bytes, ``audio_s``). No array is object-typed, so samples load with
``allow_pickle=False``; ``meta`` is finite JSON.

Memory: FLY018 (272 MB DAT, 952 s stereo) peaks at 3.1 GB RSS after
``streams.frame_to_sample`` (1.03 GB of sample fields); build one at a time.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import tdseries as td

from data_processing.frames import audio_series
from data_processing.sources._common import meta_frame
from data_processing.sources._dat_records import BUILTIN, ROTORS
from data_processing.sources.motor_dat import (
    _CRC_LEN,
    _HEADER_LEN,
    _TEXT_GROUPS,
    DAT_MAGIC,
    DATCON_CLOCK_HZ,
    DatTelemetry,
    _payloads,
)
from data_processing.sources.new_motor_data import AUDIO_CLOCK, audio_seconds, read_aligned_motor

DATASET = "new_motor_data"
MOTOR_GROUP = "Motor"
#: Integer WAV encodings that float32 / full scale represents exactly.
EXACT_PCM = frozenset({"PCM_S8", "PCM_U8", "PCM_16", "PCM_24"})
_BYTE_ROW = np.dtype(
    [
        ("tick", "<i8"),
        ("tick_raw", "<u4"),
        ("offset", "<i8"),
        ("in_order", "?"),
        ("payload_start", "<i8"),
        ("payload_length", "u1"),
    ]
)


def build(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """Yield ``(recording_id, frame)`` for every calibrated DAT in the raw tree."""
    raw_dir = Path(raw_dir)
    dats = sorted(
        (p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".dat"),
        key=lambda p: p.stem,
    )
    if not dats:
        raise FileNotFoundError(f"no .dat logs under {raw_dir}")
    stems = [p.stem for p in dats]
    if len(set(stems)) != len(stems):
        raise ValueError(f"duplicate recording ids under {raw_dir}: {stems}")
    for dat in dats:
        yield dat.stem, build_frame(raw_dir, dat)


def build_frame(raw_dir: Path, dat: Path) -> td.Frame:
    """One recording: native audio + rps + every DAT group + the DAT bytes."""
    raw_dir, dat = Path(raw_dir), Path(dat)
    flight = dat.stem
    if flight not in AUDIO_CLOCK:
        raise KeyError(f"no WAV clock calibration for {flight!r}")
    offset_s, ticks_per_second = AUDIO_CLOCK[flight]
    wavs = sorted(p for p in dat.parent.iterdir() if p.is_file() and p.suffix.lower() == ".wav")
    if len(wavs) != 1:
        raise ValueError(f"{dat.parent}: expected one WAV next to {dat.name}, found {wavs}")
    audio, audio_meta = _read_wav(wavs[0])

    raw = np.memmap(dat, mode="r", dtype=np.uint8)
    log = read_aligned_motor(dat)
    groups, layout = _groups(log, flight)
    if MOTOR_GROUP not in groups or "rows" not in groups[MOTOR_GROUP]:
        raise ValueError(f"{dat}: no single-layout {MOTOR_GROUP!r} group")
    motor = groups[MOTOR_GROUP]["rows"]
    rpm = np.stack([np.asarray(motor.data[r], dtype=np.float64) for r in ROTORS])
    rps = td.Series(rpm / 60.0, ("rotor", "time"), {"time": motor.tindex})

    magic = int(raw[:4096].tobytes().find(DAT_MAGIC))
    packets = sum(info["packets"] for info in layout.values())
    packet_bytes = sum(
        k["packets"] * (k["width"] + _HEADER_LEN + _CRC_LEN)
        for info in layout.values()
        for k in info["keys"]
    )
    header_bytes = magic + len(DAT_MAGIC)
    extra: dict[str, Any] = {
        "session": dat.parent.name,
        "files": _files(raw_dir, dat, wavs[0], raw),
        "audio": audio_meta,
        "clock": {
            "offset_s": float(offset_s),
            "ticks_per_second": float(ticks_per_second),
            "formula": "audio_s = offset_s + tick / ticks_per_second",
            "source": "data_processing.sources.new_motor_data.AUDIO_CLOCK",
            "tick_wrap": 2**32,
            "first_valid_tick": int(log._origin),
            "datcon_nominal_hz": float(DATCON_CLOCK_HZ),
            "stamps": "tdseries ns = round(audio_s * 1e9); dat rows sorted by (tick, offset); "
            "group_to_table restores file order",
        },
        "rps": {
            "source": "dat Motor (record 10090, 152 B) speed, RPM / 60, every packet",
            "unit": "rev/s",
            "rotor_order": list(ROTORS),
            "n_packets": int(rpm.shape[1]),
        },
        "dat": {
            "n_bytes": int(raw.size),
            "magic_offset": magic,
            "header_bytes": header_bytes,
            "n_packets": int(packets),
            "packet_bytes": int(packet_bytes),
            "bytes_outside_packets": int(raw.size - header_bytes - packet_bytes),
            "n_groups": len(groups),
            "excluded": "DatCon CSV export (sibling .csv; raw parent)",
        },
        "dat_layout": layout,
        "provenance": {
            "builder": "data_processing.sources.november_frames.build",
            "telemetry": "read_aligned_motor native groups, lossless; rows on the calibrated "
            "WAV clock; group_to_table reproduces read_aligned_motor(dat)[group]",
        },
    }
    json.dumps(extra, allow_nan=False)  # finite, plain JSON only
    meta = meta_frame(
        flight,
        DATASET,
        system={"platform": "DJI Matrice 100", "n_rotors": len(ROTORS)},
        observation={
            "recorder": "ZOOM H1n",
            "n_channels": int(audio.shape[0]),
            "sample_rate": int(audio_meta["sample_rate"]),
        },
        extra=extra,
    )
    return td.Frame(
        {
            "audio": audio_series(audio, int(audio_meta["sample_rate"])),
            "rps": rps,
            "dat": td.Frame(groups),
            "dat_bytes": td.wrap(raw, dims=(None,)),
            "meta": meta,
        }
    )


def group_to_table(frame: td.Frame, name: str) -> pd.DataFrame:
    """``read_aligned_motor(dat)[name]`` rebuilt from a built Frame, exactly.

    Same columns, dtypes and row (file) order; ``audio_s`` is recomputed
    from ``meta.clock``; text groups decode ``message`` and undocumented
    groups return ``payload`` bytes as the reader does.
    """
    meta = frame["meta"]
    clock, info = meta["clock"], meta["dat_layout"][name]
    group = frame["dat"][name]
    if info["decode"] in ("text", "raw"):
        rows = _file_order(np.asarray(group["rows"].data))
        stream = np.asarray(group["payload"].data)
        chunks = [
            stream[s : s + n].tobytes()
            for s, n in zip(rows["payload_start"], rows["payload_length"], strict=True)
        ]
        cols: dict[str, Any] = {k: rows[k] for k in ("tick", "tick_raw", "offset", "in_order")}
        if info["decode"] == "text":
            cols["message"] = np.array(
                [c.rstrip(b"\x00").decode("utf-8", "backslashreplace") for c in chunks],
                dtype=object,
            )
        else:
            cols["payload"] = np.array(chunks, dtype=object)
        df = pd.DataFrame(cols)
    else:
        parts = [
            _columns(_file_order(np.asarray(group[_rows_entry(info, k)].data)))
            for k in info["keys"]
        ]
        if len(parts) == 1:
            df = parts[0]
        else:
            df = pd.concat(parts, ignore_index=True)
            df.sort_values("offset", inplace=True, ignore_index=True)
    ticks = np.asarray(df["tick"].to_numpy(), dtype=np.float64)
    df.insert(0, "audio_s", clock["offset_s"] + ticks / clock["ticks_per_second"])
    return df


def restore_dat(frame: td.Frame, path: str | Path) -> Path:
    """Write the original DAT file from ``dat_bytes``; verify it against ``meta``."""
    path = Path(path)
    data = np.asarray(frame["dat_bytes"].data, dtype=np.uint8)
    want = frame["meta"]["files"]["dat"]["sha256"]
    got = hashlib.sha256(memoryview(data)).hexdigest()
    if got != want:
        raise ValueError(f"dat_bytes sha256 {got} != meta {want}")
    data.tofile(path)
    return path


def _columns(rows: np.ndarray) -> pd.DataFrame:
    """Structured rows -> DataFrame, one column per field, dtypes unchanged."""
    return pd.DataFrame({name: rows[name] for name in rows.dtype.names or ()})


def _rows_entry(info: dict[str, Any], key: dict[str, Any]) -> str:
    return "rows" if len(info["keys"]) == 1 else f"rows_{key['width']}"


def _file_order(rows: np.ndarray) -> np.ndarray:
    return rows[np.argsort(rows["offset"], kind="stable")]


def _events(flight: str, rows: np.ndarray) -> td.Series:
    """Rows as events on the calibrated clock, sorted by ``(tick, offset)``."""
    rows = rows[np.lexsort((rows["offset"], rows["tick"]))]
    return td.events(audio_seconds(flight, rows["tick"]), rows, dims=("time",))


def _records(df: pd.DataFrame) -> np.ndarray:
    """A reader table as one packed structured array, dtypes unchanged."""
    fields = []
    for col in df.columns:
        dtype = df[col].dtype
        if not isinstance(dtype, np.dtype) or dtype.hasobject:
            raise TypeError(f"column {col!r} has non-numpy dtype {dtype}")
        fields.append((str(col), dtype))
    out = np.empty(len(df), np.dtype(fields))
    for col in df.columns:
        out[str(col)] = df[col].to_numpy()
    return out


def _fields(dtype: np.dtype) -> list[list[Any]]:
    """``[name, dtype, payload byte offset]`` per packed field."""
    return [[n, f.str, int(o)] for n, (f, o, *_) in (dtype.fields or {}).items()]


def _groups(log: DatTelemetry, flight: str) -> tuple[dict[str, td.Frame], dict[str, Any]]:
    groups: dict[str, td.Frame] = {}
    layout: dict[str, Any] = {}
    for name, keys in log._groups.items():
        if not keys:
            continue  # the DatCon CSV export
        if "/" in name or "#" in name:
            raise ValueError(f"group name {name!r} is not a tdframe entry name")
        decoded = [k in BUILTIN or k in log._schemas for k in keys]
        if any(decoded) != all(decoded):
            raise ValueError(f"group {name!r} mixes decoded and raw layouts: {keys}")
        info: dict[str, Any] = {"record_type": int(keys[0][0]), "keys": []}
        for rtype, width in keys:
            key_info: dict[str, Any] = {
                "width": int(width),
                "packets": len(log._index[(rtype, width)]),
            }
            if (rtype, width) in BUILTIN:
                key_info["builtin"] = _fields(BUILTIN[(rtype, width)][1])
            if (rtype, width) in log._schemas:
                key_info["firmware"] = _fields(log._schemas[(rtype, width)][1])
            info["keys"].append(key_info)
        info["packets"] = sum(k["packets"] for k in info["keys"])
        if all(decoded):
            sources = {s for k in info["keys"] for s in ("builtin", "firmware") if s in k}
            info["decode"] = "+".join(sorted(sources))
            entries = {
                _rows_entry(info, k): _events(flight, _records(log._native(key)))
                for key, k in zip(keys, info["keys"], strict=True)
            }
        else:
            info["decode"] = "text" if keys[0][0] in _TEXT_GROUPS else "raw"
            entries = _byte_entries(log, keys, flight)
        groups[name] = td.Frame(entries)
        layout[name] = info
    return groups, layout


def _byte_entries(log: DatTelemetry, keys: list[tuple[int, int]], flight: str) -> dict[str, Any]:
    """Undecoded groups: header rows + one uint8 payload stream in file order."""
    if len(keys) == 1:
        key = keys[0]
        index = log._index[key]
        offsets = np.abs(index[:, 0])
        widths = key[1]
        starts = np.arange(len(index), dtype=np.int64) * widths
        stream = _payloads(log._buf, index, widths).ravel()
    else:
        index = np.concatenate([log._index[k] for k in keys])
        widths = np.concatenate([np.full(len(log._index[k]), k[1], np.uint8) for k in keys])
        offsets = np.abs(index[:, 0])
        order = np.argsort(offsets, kind="stable")
        starts = np.empty(len(index), np.int64)
        starts[order] = np.cumsum(widths[order], dtype=np.int64) - widths[order]
        stream = np.empty(int(widths.sum()), np.uint8)
        first = 0
        for key in keys:
            n = len(log._index[key])
            payload = _payloads(log._buf, log._index[key], key[1])
            stream[starts[first : first + n, None] + np.arange(key[1])] = payload
            first += n
    rows = np.empty(len(index), _BYTE_ROW)
    rows["tick"] = index[:, 1]
    rows["tick_raw"] = index[:, 1].astype(np.uint32)
    rows["offset"] = offsets
    rows["in_order"] = index[:, 0] > 0
    rows["payload_start"] = starts
    rows["payload_length"] = widths
    return {"rows": _events(flight, rows), "payload": td.wrap(stream, dims=(None,))}


def _read_wav(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    info = sf.info(str(path))
    if info.subtype not in EXACT_PCM:
        raise ValueError(f"{path}: {info.subtype} is not exactly representable as float32")
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = np.ascontiguousarray(data.T)
    del data
    return audio, {
        "file": path.name,
        "format": str(info.format),
        "subtype": str(info.subtype),
        "sample_rate": int(sr),
        "channels": int(audio.shape[0]),
        "frames": int(audio.shape[1]),
        "duration_s": audio.shape[1] / float(sr),
        "encoding": "float32 = integer PCM / full scale (exact)",
    }


def _sha256(path: Path) -> str:
    """Hash an original sidecar without materializing it."""
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _file_info(raw_dir: Path, path: Path, digest: str | None = None) -> dict[str, Any]:
    return {
        "name": path.name,
        "relpath": path.relative_to(raw_dir).as_posix(),
        "size": path.stat().st_size,
        "sha256": digest or _sha256(path),
    }


def _files(raw_dir: Path, dat: Path, wav: Path, raw: np.ndarray) -> dict[str, Any]:
    """Every file of the session directory with its sha256, by role."""
    files: dict[str, Any] = {
        "dat": _file_info(raw_dir, dat, hashlib.sha256(memoryview(raw)).hexdigest()),
        "wav": _file_info(raw_dir, wav),
    }
    others = []
    for p in sorted(dat.parent.iterdir()):
        if not p.is_file() or p in (dat, wav):
            continue
        if p.suffix.lower() == ".csv" and p.stem == dat.stem:
            files["csv"] = _file_info(raw_dir, p)
        else:
            others.append(_file_info(raw_dir, p))
    files["other"] = others
    return files
