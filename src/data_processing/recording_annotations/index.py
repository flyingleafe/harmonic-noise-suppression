"""One SQLite database per annotated parent ``dataset@version``: schema, writer, reader.

The database is the whole annotation of one exact parent dload version: the ten
condition facets per sample key, their evidence, role maps, the parent shard /
byte span of every sample (so audio can be fetched for exactly the selected
keys), and the acoustic shaft-rate windows of the tachometry pass. Common
content is normalized: identical classifier records share one ``profile`` row
and JSON payloads (evidence, extras, summaries) live once in ``blob``, so a
318k-clip corpus with a handful of profiles stays a few MB.

Packaging into the ``drone-recording-annotations`` dload release lives in
:mod:`.storage`; the public API is re-exported by the package ``__init__``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import threading
import zlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1

#: The ten facets, in display order; column names of ``profile``.
FACET_NAMES: tuple[str, ...] = (
    "rig_mount",
    "speed_regime",
    "propulsion",
    "rotor_activity",
    "relative_motion",
    "speed_supplied",
    "sound_content",
    "environment",
    "blade_count",
    "wind",
)
UNKNOWN = "unknown"
NOT_APPLICABLE = "not_applicable"
#: Filter value matching any facet value other than unknown / not_applicable.
KNOWN = "@known"
#: Largest page :meth:`Index.select` returns.
MAX_PAGE = 10_000
TACH_STATUSES = ("resolved", "partial", "unresolved", "too_short", "not_applicable")
#: ``tach_status`` filter value for rows without a stored estimate.
NO_TACH = "none"
#: facet_counts label (and filter value) of an absent nullable value.
NULL_LABEL = "null"

#: Record keys with their own profile column; everything else of the record
#: except ``evidence`` goes to the profile's ``extra`` blob.
_PROFILE_SCALARS = ("n_rotors", "n_active", "provenance_kind", "supplied_kind", "tach_candidate")
_ROLE_KINDS = {"role_motion": "motion", "role_wind": "wind"}
#: Filters decided by the profile alone (counted via ``v_profile`` without a row scan).
_PROFILE_FILTERS = frozenset({*FACET_NAMES, *_PROFILE_SCALARS, *_ROLE_KINDS})
#: Window-array keys of :meth:`Index.tach_arrays`.
_WINDOW_ARRAYS = ("times", "accepted", "rates", "uncertainty", "order_support", "notes")
#: ``order_support`` byte meaning "absent" in the packed window arrays.
_NO_SUPPORT = 255
_ROW_COLUMNS = (
    "key",
    *FACET_NAMES,
    "n_rotors",
    "n_active",
    "provenance_kind",
    "supplied_kind",
    "tach_candidate",
    "has_audio",
    "duration_s",
    "sample_rate",
    "n_channels",
    "tach_status",
    "tach_n_windows",
)
_ORDERS = {
    "key": "v.key",
    "duration_s": "v.duration_s, v.key",
    "-duration_s": "v.duration_s DESC, v.key",
}

_DDL = f"""
CREATE TABLE release_meta(name TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE parent_shard(shard INTEGER PRIMARY KEY, digest TEXT NOT NULL,
    size INTEGER NOT NULL, num_samples INTEGER NOT NULL);
CREATE TABLE blob(bid INTEGER PRIMARY KEY, sha1 TEXT UNIQUE NOT NULL, json TEXT NOT NULL);
CREATE TABLE profile(pid INTEGER PRIMARY KEY, sha1 TEXT UNIQUE NOT NULL,
    {", ".join(f"{f} TEXT NOT NULL" for f in FACET_NAMES)},
    n_rotors INTEGER, n_active INTEGER, provenance_kind TEXT, supplied_kind TEXT,
    tach_candidate INTEGER, evidence_bid INTEGER NOT NULL, extra_bid INTEGER NOT NULL);
CREATE TABLE profile_role(pid INTEGER NOT NULL, kind TEXT NOT NULL, role TEXT NOT NULL,
    value TEXT NOT NULL, PRIMARY KEY(pid, kind, role)) WITHOUT ROWID;
CREATE TABLE recording(rid INTEGER PRIMARY KEY, key TEXT UNIQUE NOT NULL, pid INTEGER NOT NULL,
    has_audio INTEGER, duration_s REAL, sample_rate REAL, n_channels INTEGER,
    shard INTEGER, span_offset INTEGER, span_length INTEGER, fields TEXT,
    wind_bid INTEGER, measure_bid INTEGER);
CREATE TABLE tach(rid INTEGER PRIMARY KEY, status TEXT NOT NULL, reason TEXT, identity TEXT,
    n_expected INTEGER, role TEXT, audio_t_start_s REAL, n_windows INTEGER NOT NULL,
    n_accepted INTEGER NOT NULL, rate_min REAL, rate_max REAL, summary_bid INTEGER,
    n_slots INTEGER NOT NULL, win_time BLOB, win_accepted BLOB, win_rates BLOB, win_unc BLOB,
    win_support BLOB, win_notes BLOB);
CREATE TABLE profile_n(pid INTEGER PRIMARY KEY, n INTEGER NOT NULL);
{"".join(f"CREATE INDEX profile_{f} ON profile({f});" for f in FACET_NAMES)}
CREATE INDEX recording_pid ON recording(pid);
CREATE INDEX tach_status ON tach(status);
CREATE VIEW v_profile AS SELECT p.pid AS pid,
    {", ".join(f"p.{f} AS {f}" for f in FACET_NAMES)},
    p.n_rotors AS n_rotors, p.n_active AS n_active, p.provenance_kind AS provenance_kind,
    p.supplied_kind AS supplied_kind, p.tach_candidate AS tach_candidate, pn.n AS n
    FROM profile p JOIN profile_n pn ON pn.pid = p.pid;
CREATE VIEW v_row AS SELECT r.rid AS rid, r.pid AS pid, r.key AS key,
    {", ".join(f"p.{f} AS {f}" for f in FACET_NAMES)},
    p.n_rotors AS n_rotors, p.n_active AS n_active, p.provenance_kind AS provenance_kind,
    p.supplied_kind AS supplied_kind, p.tach_candidate AS tach_candidate,
    r.has_audio AS has_audio, r.duration_s AS duration_s, r.sample_rate AS sample_rate,
    r.n_channels AS n_channels, COALESCE(t.status, '{NO_TACH}') AS tach_status,
    COALESCE(t.n_windows, 0) AS tach_n_windows, t.identity AS tach_identity
    FROM recording r JOIN profile p ON p.pid = r.pid LEFT JOIN tach t ON t.rid = r.rid;
"""


class NotAnnotatedError(LookupError):
    """No annotation database exists for the exact parent ``dataset@version``."""


@dataclass(frozen=True, slots=True)
class SampleLocation:
    """Where one sample's bytes sit in its parent shard.

    ``offset``/``length``: absolute byte span of the sample payload in the
    shard file (pack fields are written contiguously); ``fields``:
    ``(name, offset relative to the span start, length)`` in pack order.
    """

    shard: int
    offset: int
    length: int
    fields: tuple[tuple[str, int, int], ...]


def location_from_entry(shard: int, entry: Mapping[str, Sequence[int]]) -> SampleLocation:
    """:class:`SampleLocation` from one dload pack-footer entry ``{field: [offset, length]}``."""
    if not entry:
        return SampleLocation(int(shard), 0, 0, ())
    start = min(int(off) for off, _ in entry.values())
    end = max(int(off) + int(n) for off, n in entry.values())
    fields = tuple(
        (str(name), int(off) - start, int(n))
        for name, (off, n) in sorted(entry.items(), key=lambda kv: int(kv[1][0]))
    )
    return SampleLocation(int(shard), start, end - start, fields)


# ─── JSON helpers ──────────────────────────────────────────────────────────────


def _plain(value: Any) -> Any:
    """Recursively convert to JSON-native values; NaN/inf become None."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    item = getattr(value, "item", None)  # numpy scalar
    if callable(item) and getattr(value, "ndim", None) == 0:
        return _plain(item())
    tolist = getattr(value, "tolist", None)  # numpy array
    if callable(tolist):
        return _plain(tolist())
    raise TypeError(f"annotation value of type {type(value).__name__} is not JSON-serializable")


def _dumps(value: Any) -> str:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _f32(value: Any) -> float | None:
    """A stored float32 as the shortest decimal that round-trips it (NaN -> None)."""
    v = np.float32(value)
    return None if not np.isfinite(v) else float(str(v))


def _int_or_none(value: Any) -> int | None:
    return None if value is None else int(value)


# ─── Writer ────────────────────────────────────────────────────────────────────


def _facet_vocabulary() -> dict[str, list[str]]:
    from data_processing.recording_annotations import schema

    facets = {str(k): [str(v) for v in vals] for k, vals in schema.FACETS.items()}
    if tuple(facets) != FACET_NAMES:
        raise RuntimeError(
            f"schema.FACETS order {tuple(facets)} differs from the index columns {FACET_NAMES}"
        )
    return facets


def _valid_value(facet: str, value: str, vocab: Mapping[str, Sequence[str]]) -> bool:
    """Reader-side check against the vocabulary stored in the database (canonical
    decimal blade counts are open-ended); the writer uses ``schema.valid_value``."""
    if value in vocab[facet]:
        return True
    return (
        facet == "blade_count"
        and value.isascii()
        and value.isdecimal()
        and str(int(value)) == value
    )


def _schema_valid(facet: str, value: Any) -> bool:
    from data_processing.recording_annotations import schema

    return isinstance(value, str) and schema.valid_value(facet, value)


def _audio_channels(info: Mapping[str, Any]) -> int | None:
    if info.get("has_audio") is False:
        return None
    if info.get("n_channels") is not None:
        return int(info["n_channels"])
    series = info.get("series") or {}
    for name in ("audio", "mixture"):
        desc = series.get(name)
        if isinstance(desc, Mapping) and desc.get("shape") is not None:
            shape = list(desc["shape"])
            return int(shape[0]) if len(shape) == 2 else 1
    return None


class IndexWriter:
    """Build the annotation database of one exact parent ``dataset@version``.

    ``parent_shards`` is the parent manifest's ``shards`` (``ShardInfo`` or
    ``(digest, size, num_samples)``) — stored so readers fetch sample bytes
    without the parent manifest and so a version mix-up is detectable. The
    file appears at ``path`` only on a successful :meth:`close`.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        dataset: str,
        version: str,
        layout: str | None,
        parent_shards: Iterable[Any] = (),
        parent_meta: Mapping[str, Any] | None = None,
        source_commit: str | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        if len(str(version)) != 64:
            raise ValueError(
                f"parent version must be the full 64-hex manifest version, got {version!r}"
            )
        self.path = Path(path)
        self.dataset = str(dataset)
        self.version = str(version)
        self._vocab = _facet_vocabulary()
        self._tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.building")
        self._tmp.unlink(missing_ok=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self._tmp)
        self._closed = False
        # 16 KiB pages: a 40-window track row (~2 KB) would waste half of a 4 KiB page.
        self._db.executescript(
            "PRAGMA page_size=16384; PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;" + _DDL
        )
        self._blob_ids: dict[str, int] = {}
        self._profile_ids: dict[str, int] = {}
        self._record_memo: dict[int, tuple[Any, int]] = {}
        self._keys: set[str] = set()
        self._n_tach = self._n_windows = 0
        for i, shard in enumerate(parent_shards):
            digest, size, n = (
                (shard.digest, shard.size, shard.num_samples)
                if hasattr(shard, "digest")
                else tuple(shard)
            )
            self._db.execute(
                "INSERT INTO parent_shard VALUES (?,?,?,?)", (i, str(digest), int(size), int(n))
            )
        self._meta: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "dataset": self.dataset,
            "version": self.version,
            "layout": layout,
            "parent_meta": _plain(dict(parent_meta or {})),
            "source_commit": source_commit,
            "provenance": _plain(dict(provenance or {})),
            "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "facets": self._vocab,
        }

    def __enter__(self) -> IndexWriter:
        return self

    def __exit__(self, exc_type: Any, *exc: Any) -> None:
        if exc_type is not None:
            self.abort()
        elif not self._closed:
            self.close()

    # -- internals -------------------------------------------------------

    def _insert(self, sql: str, params: Sequence[Any]) -> int:
        """Run one INSERT and return its rowid (every table here has one)."""
        rowid = self._db.execute(sql, tuple(params)).lastrowid
        if rowid is None:
            raise RuntimeError(f"sqlite returned no rowid for {sql[:40]!r}")
        return rowid

    def _blob(self, value: Any) -> int | None:
        if value is None:
            return None
        text = _dumps(value)
        digest = _sha1(text)
        bid = self._blob_ids.get(digest)
        if bid is None:
            bid = self._insert("INSERT INTO blob(sha1, json) VALUES (?,?)", (digest, text))
            self._blob_ids[digest] = bid
        return bid

    def _validate(self, key: str, record: Mapping[str, Any]) -> None:
        for facet in FACET_NAMES:
            value = record.get(facet)
            if not _schema_valid(facet, value):
                raise ValueError(f"{self.dataset}/{key}: invalid {facet}={value!r}")
        evidence = record.get("evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError(f"{self.dataset}/{key}: record has no evidence mapping")
        for kind in _ROLE_KINDS:
            roles = record.get(kind)
            if roles is None:
                continue
            facet = "relative_motion" if kind == "role_motion" else "wind"
            for role, value in roles.items():
                if not _schema_valid(facet, value):
                    raise ValueError(f"{self.dataset}/{key}: invalid {kind}[{role!r}]={value!r}")
        for name in ("n_rotors", "n_active"):
            value = record.get(name)
            if value is not None and (isinstance(value, bool) or int(value) != value or value < 0):
                raise ValueError(f"{self.dataset}/{key}: invalid {name}={value!r}")

    def _profile(self, key: str, record: Mapping[str, Any]) -> int:
        memo = self._record_memo.get(id(record))
        if memo is not None and memo[0] is record:
            return memo[1]
        self._validate(key, record)
        text = _dumps(record)
        digest = _sha1(text)
        pid = self._profile_ids.get(digest)
        if pid is None:
            extra = {
                k: v
                for k, v in record.items()
                if k not in FACET_NAMES and k not in _PROFILE_SCALARS and k != "evidence"
            }
            tach_candidate = record.get("tach_candidate")
            pid = self._insert(
                f"INSERT INTO profile(sha1, {', '.join(FACET_NAMES)}, n_rotors, n_active, "
                "provenance_kind, supplied_kind, tach_candidate, evidence_bid, extra_bid) "
                f"VALUES ({', '.join('?' * (len(FACET_NAMES) + 8))})",
                (
                    digest,
                    *(record[f] for f in FACET_NAMES),
                    _int_or_none(record.get("n_rotors")),
                    _int_or_none(record.get("n_active")),
                    record.get("provenance_kind"),
                    record.get("supplied_kind"),
                    None if tach_candidate is None else int(bool(tach_candidate)),
                    self._blob(record["evidence"]),
                    self._blob(extra),
                ),
            )
            for kind, short in _ROLE_KINDS.items():
                for role, value in (record.get(kind) or {}).items():
                    self._db.execute(
                        "INSERT INTO profile_role VALUES (?,?,?,?)", (pid, short, str(role), value)
                    )
            self._profile_ids[digest] = pid
        self._record_memo[id(record)] = (record, pid)
        return pid

    def _pack_tach(
        self, key: str, tach: Mapping[str, Any], t0: float, role: str
    ) -> tuple[Any, ...]:
        """Validate one estimate and pack its ``tach`` row (sans rid); windows become compact little-endian arrays
        (read back by :meth:`Index.tach_arrays`): float32 times relative to ``t0``, uint8
        accepted, float32 ``(W, n_slots)`` rates/uncertainties (NaN = absent),
        uint8 order support (255 = absent), zlib JSON for per-window notes."""
        where = f"{self.dataset}/{key}"
        status = tach.get("status")
        if status not in TACH_STATUSES:
            raise ValueError(f"{where}: tach status {status!r} not in {TACH_STATUSES}")
        windows = list(tach.get("windows") or [])
        n_expected = _int_or_none(tach.get("n_expected"))
        parsed = []
        prev_center = -math.inf
        for w, win in enumerate(windows):
            rates = [_float_or_none(r) for r in win.get("rates_rev_s") or []]
            unc = [_float_or_none(u) for u in win.get("uncertainty_rev_s") or []]
            if len(unc) != len(rates):
                raise ValueError(f"{where}: window {w} rates/uncertainty length mismatch")
            slots = win.get("slots")
            slots = list(range(len(rates))) if slots is None else [int(s) for s in slots]
            if (
                len(slots) != len(rates)
                or len(set(slots)) != len(slots)
                or any(s < 0 for s in slots)
            ):
                raise ValueError(f"{where}: window {w} slots {slots} do not index its rates")
            if n_expected is not None and any(s >= n_expected for s in slots):
                raise ValueError(f"{where}: window {w} slot beyond n_expected={n_expected}")
            center = _float_or_none(win.get("center_s"))
            if center is None:
                raise ValueError(f"{where}: window {w} has no finite center_s")
            if center <= prev_center:
                raise ValueError(f"{where}: windows are not strictly chronological")
            prev_center = center
            support = win.get("order_support")
            parsed.append((win, rates, unc, slots, center, support))
        n_slots = max([n_expected or 0, *(1 + max(p[3], default=-1) for p in parsed)])
        rotor_ids = tach.get("rotor_ids")
        if rotor_ids is not None:
            if tach.get("identity") != "physical":
                raise ValueError(
                    f"{where}: rotor_ids need identity='physical' (ranks are not motors)"
                )
            if len(rotor_ids) != n_slots or len(set(map(str, rotor_ids))) != n_slots:
                raise ValueError(
                    f"{where}: rotor_ids {rotor_ids} must name the {n_slots} slots uniquely"
                )
        n_w = len(parsed)
        times = np.full((n_w, 3), np.nan, np.float32)
        accepted = np.zeros(n_w, np.uint8)
        rates_a = np.full((n_w, n_slots), np.nan, np.float32)
        unc_a = np.full((n_w, n_slots), np.nan, np.float32)
        support_a = np.full((n_w, n_slots), _NO_SUPPORT, np.uint8)
        notes: list[dict[str, Any]] = []
        for w, (win, rates, unc, slots, center, _) in enumerate(parsed):
            start, end = _float_or_none(win.get("start_s")), _float_or_none(win.get("end_s"))
            times[w] = [
                np.nan if start is None else start - t0,
                np.nan if end is None else end - t0,
                center - t0,
            ]
            accepted[w] = bool(win.get("accepted", any(r is not None for r in rates)))
            for s, r, u in zip(slots, rates, unc, strict=True):
                if r is not None:
                    rates_a[w, s] = r
                    unc_a[w, s] = np.nan if u is None else u
            notes.append({k: win[k] for k in ("reason", "quality") if win.get(k) is not None})
        # Order support: uint8 per slot when every window gives small ints parallel
        # to its rates; any other shape is kept verbatim in the notes.
        supports = [(p[3], p[5]) for p in parsed]
        int_support = all(
            s is None
            or (
                isinstance(s, (list, tuple))
                and len(s) == len(slots)
                and all(
                    isinstance(v, int) and not isinstance(v, bool) and 0 <= v < _NO_SUPPORT
                    for v in s
                )
            )
            for slots, s in supports
        )
        for w, (slots, s) in enumerate(supports):
            if s is None:
                continue
            if int_support:
                for slot, v in zip(slots, s, strict=True):
                    support_a[w, slot] = v
            else:
                notes[w]["order_support"] = s
        ok = accepted.astype(bool)[:, None] & np.isfinite(rates_a)
        summary = {
            k: v
            for k, v in tach.items()
            if k not in ("status", "reason", "identity", "n_expected", "windows", "role")
        }
        has_support = int_support and bool((support_a != _NO_SUPPORT).any())
        return (
            status,
            tach.get("reason"),
            tach.get("identity"),
            n_expected,
            str(tach.get("role") or role),
            float(t0),
            n_w,
            int(accepted.sum()),
            _f32(rates_a[ok].min()) if ok.any() else None,
            _f32(rates_a[ok].max()) if ok.any() else None,
            self._blob(summary or None),
            n_slots,
            times.astype("<f4").tobytes(),
            accepted.tobytes(),
            rates_a.astype("<f4").tobytes(),
            unc_a.astype("<f4").tobytes(),
            support_a.tobytes() if has_support else None,
            zlib.compress(_dumps(notes).encode(), 9) if any(notes) else None,
        )

    # -- public ------------------------------------------------------------

    def add(
        self,
        key: str,
        record: Mapping[str, Any],
        *,
        info: Mapping[str, Any],
        location: SampleLocation | None = None,
        tach: Mapping[str, Any] | None = None,
        audio_t_start_s: float = 0.0,
        wind: Mapping[str, Any] | None = None,
        measure: Mapping[str, Any] | None = None,
    ) -> None:
        """Store one parent sample. ``record`` = validated classifier output
        (shared instances across keys are cheap); ``tach`` = the
        ``steady_tachometry.estimate`` dict computed with
        ``t_start=audio_t_start_s``; ``wind``/``measure`` = provenance dicts."""
        if self._closed:
            raise ValueError("writer is closed")
        key = str(key)
        if key.startswith("_") or key in self._keys:
            raise ValueError(f"{self.dataset}: bookkeeping or duplicate key {key!r}")
        # Validate everything before the first row of this key is written, so a
        # rejected sample leaves no partial rows (profiles/blobs are content-keyed).
        pid = self._profile(key, record)
        packed = (
            None
            if tach is None
            else self._pack_tach(
                key, tach, float(audio_t_start_s), str(record.get("tach_role") or "audio")
            )
        )
        loc = location
        rid = self._insert(
            "INSERT INTO recording(key, pid, has_audio, duration_s, sample_rate, n_channels, shard, "
            "span_offset, span_length, fields, wind_bid, measure_bid) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                key,
                pid,
                None if info.get("has_audio") is None else int(bool(info["has_audio"])),
                _float_or_none(info.get("duration_s")),
                _float_or_none(info.get("sample_rate")),
                _audio_channels(info),
                None if loc is None else int(loc.shard),
                None if loc is None else int(loc.offset),
                None if loc is None else int(loc.length),
                None if loc is None else _dumps([list(f) for f in loc.fields]),
                self._blob(wind),
                self._blob(measure),
            ),
        )
        if packed is not None:
            self._db.execute(f"INSERT INTO tach VALUES ({', '.join('?' * 19)})", (rid, *packed))
            self._n_tach += 1
            self._n_windows += packed[6]
        self._keys.add(key)

    def abort(self) -> None:
        if not self._closed:
            self._db.close()
            self._closed = True
        self._tmp.unlink(missing_ok=True)

    def close(self, expected_keys: Iterable[str] | None = None) -> Path:
        """Check completeness against ``expected_keys`` (the parent's data
        keys), finalize and atomically place the compacted database at
        ``path``."""
        if self._closed:
            raise ValueError("writer is closed")
        if expected_keys is not None:
            expected = {str(k) for k in expected_keys if not str(k).startswith("_")}
            missing, extra = sorted(expected - self._keys), sorted(self._keys - expected)
            if missing or extra:
                self.abort()
                raise ValueError(
                    f"{self.dataset}@{self.version[:12]}: {len(missing)} parent keys unannotated "
                    f"(e.g. {missing[:3]}), {len(extra)} keys not in the parent (e.g. {extra[:3]})"
                )
        self._meta.update(
            n_rows=len(self._keys),
            n_profiles=len(self._profile_ids),
            n_tach=self._n_tach,
            n_tach_windows=self._n_windows,
        )
        self._db.executemany(
            "INSERT INTO release_meta VALUES (?,?)", [(k, _dumps(v)) for k, v in self._meta.items()]
        )
        self._db.execute("INSERT INTO profile_n SELECT pid, COUNT(*) FROM recording GROUP BY pid")
        self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._db.commit()
        final_tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.vacuum")
        final_tmp.unlink(missing_ok=True)
        self._db.execute("VACUUM INTO ?", (str(final_tmp),))
        self._db.close()
        self._closed = True
        self._tmp.unlink(missing_ok=True)
        os.replace(final_tmp, self.path)
        return self.path


# ─── Reader ────────────────────────────────────────────────────────────────────


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple, set, frozenset)) else [value]


class Index:
    """Read-only view of one annotation database (thread-safe)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._db = sqlite3.connect(
            f"{self.path.resolve().as_uri()}?mode=ro&immutable=1", uri=True, check_same_thread=False
        )
        self._lock = threading.Lock()
        (user_version,) = self._db.execute("PRAGMA user_version").fetchone()
        if user_version != SCHEMA_VERSION:
            self._db.close()
            raise ValueError(
                f"{self.path}: annotation schema {user_version}, reader expects {SCHEMA_VERSION}"
            )
        self._meta = {
            k: json.loads(v) for k, v in self._db.execute("SELECT name, value FROM release_meta")
        }
        self.dataset: str = self._meta["dataset"]
        self.version: str = self._meta["version"]
        self.layout: str = self._meta["layout"]
        self.n_rows: int = int(self._meta["n_rows"])
        self.facets: dict[str, list[str]] = self._meta["facets"]

    def __repr__(self) -> str:
        return f"Index({self.dataset}@{self.version[:12]}, {self.n_rows} rows)"

    def close(self) -> None:
        self._db.close()

    def meta(self) -> dict[str, Any]:
        return dict(self._meta)

    def _query(self, sql: str, params: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
        with self._lock:
            return self._db.execute(sql, tuple(params)).fetchall()

    # -- filters ---------------------------------------------------------

    def _facet_clause(self, facet: str, values: Any) -> tuple[str, list[Any]]:
        vals = [str(v) for v in _as_list(values)]
        known = KNOWN in vals
        plain = [v for v in vals if v != KNOWN]
        for v in plain:
            if not _valid_value(facet, v, self.facets):
                raise ValueError(f"{facet}={v!r} is not in the vocabulary {self.facets[facet]}")
        parts: list[str] = []
        params: list[Any] = []
        if plain:
            parts.append(f"v.{facet} IN ({', '.join('?' * len(plain))})")
            params += plain
        if known:
            parts.append(f"v.{facet} NOT IN (?, ?)")
            params += [UNKNOWN, NOT_APPLICABLE]
        if not parts:
            return "0", []
        return "(" + " OR ".join(parts) + ")", params

    def _where(
        self, filters: Mapping[str, Any] | None, exclude: str | None = None
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        for name, value in (filters or {}).items():
            if name == exclude:
                continue
            if name in FACET_NAMES:
                clause, p = self._facet_clause(name, value)
            elif name in (
                "n_rotors",
                "n_active",
                "provenance_kind",
                "supplied_kind",
                "tach_identity",
            ):
                # None or "null" (the facet_counts label of an absent value) match IS NULL.
                vals = _as_list(value)
                null = any(v is None or v == NULL_LABEL for v in vals)
                cast = int if name in ("n_rotors", "n_active") else str
                p = [cast(v) for v in vals if v is not None and v != NULL_LABEL]
                sub = [f"v.{name} IN ({', '.join('?' * len(p))})"] if p else []
                if null:
                    sub.append(f"v.{name} IS NULL")
                clause = ("(" + " OR ".join(sub) + ")") if sub else "0"
            elif name == "tach_status":
                vals = [str(v) for v in _as_list(value)]
                bad = [v for v in vals if v not in (*TACH_STATUSES, NO_TACH)]
                if bad:
                    raise ValueError(f"tach_status {bad} not in {(*TACH_STATUSES, NO_TACH)}")
                clause, p = (
                    f"v.tach_status IN ({', '.join('?' * len(vals))})" if vals else "0",
                    vals,
                )
            elif name in ("tach_candidate", "has_audio"):
                clause, p = f"v.{name} = ?", [int(bool(value))]
            elif name in _ROLE_KINDS:
                facet = "relative_motion" if name == "role_motion" else "wind"
                sub, p = [], []
                for role, vals in dict(value).items():
                    vals = [str(v) for v in _as_list(vals)]
                    for v in vals:
                        if not _valid_value(facet, v, self.facets):
                            raise ValueError(f"{name}[{role!r}]={v!r} is not a {facet} value")
                    sub.append(
                        "EXISTS (SELECT 1 FROM profile_role pr WHERE pr.pid = v.pid AND pr.kind = ? "
                        f"AND pr.role = ? AND pr.value IN ({', '.join('?' * len(vals))}))"
                    )
                    p += [_ROLE_KINDS[name], str(role), *vals]
                clause = " AND ".join(sub) if sub else "1"
            elif name == "keys":
                vals = [str(v) for v in _as_list(value)]
                clause, p = f"v.key IN ({', '.join('?' * len(vals))})" if vals else "0", vals
            elif name == "key_prefix":
                clause, p = "substr(v.key, 1, ?) = ?", [len(str(value)), str(value)]
            else:
                raise ValueError(f"unknown annotation filter {name!r}")
            clauses.append(clause)
            params += p
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    # -- queries ---------------------------------------------------------

    def _source(
        self, filters: Mapping[str, Any] | None, exclude: str | None = None
    ) -> tuple[str, str]:
        """``(view, count expression)``: profile-level filters count over the
        few profiles (``v_profile.n``) instead of scanning every recording."""
        names = {n for n in (filters or {}) if n != exclude}
        if names <= _PROFILE_FILTERS:
            return "v_profile", "SUM(v.n)"
        return "v_row", "COUNT(*)"

    def count(self, filters: Mapping[str, Any] | None = None) -> int:
        where, params = self._where(filters)
        view, n = self._source(filters)
        return int(self._query(f"SELECT COALESCE({n}, 0) FROM {view} v{where}", params)[0][0])

    def facet_counts(self, filters: Mapping[str, Any] | None = None) -> dict[str, dict[str, int]]:
        """``{group: {value: n}}`` for the ten facets, ``tach_status`` (incl.
        ``none``), ``provenance_kind``, ``supplied_kind`` (``null`` = absent)
        and the role maps ``role_motion``/``role_wind`` as ``{"<role>=<value>": n}``.
        Each group is counted under every filter except its own (multiselect
        semantics)."""
        filters = dict(filters or {})
        prof = {k: v for k, v in filters.items() if k in _PROFILE_FILTERS}
        row = {k: v for k, v in filters.items() if k not in _PROFILE_FILTERS}
        # Row-level filters (tach status, has_audio, keys) reduce to per-profile
        # counts in ONE scan; every profile-level group is then a cheap join.
        if row:
            where, params = self._where(row)
            sql = f"WITH m AS MATERIALIZED (SELECT v.pid AS pid, COUNT(*) AS n FROM v_row v{where} GROUP BY v.pid) "
            src, n = "m JOIN v_profile v ON v.pid = m.pid", "SUM(m.n)"
        else:
            sql, params, src, n = "", [], "v_profile v", "SUM(v.n)"
        parts: list[str] = []
        groups = (*FACET_NAMES, "provenance_kind", "supplied_kind")
        for group in groups:
            where, p = self._where(prof, exclude=group)
            parts.append(f"SELECT '{group}', v.{group}, {n} FROM {src}{where} GROUP BY v.{group}")
            params += p
        for group, kind in _ROLE_KINDS.items():
            where, p = self._where(prof, exclude=group)
            parts.append(
                f"SELECT '{group}', pr.role || '=' || pr.value, {n} FROM {src} JOIN profile_role pr "
                f"ON pr.pid = v.pid AND pr.kind = '{kind}'{where} GROUP BY pr.role, pr.value"
            )
            params += p
        out: dict[str, dict[str, int]] = {g: {} for g in (*groups, "tach_status", *_ROLE_KINDS)}
        for group, value, count in self._query(sql + " UNION ALL ".join(parts), params):
            out[group][NULL_LABEL if value is None else str(value)] = int(count)
        where, p = self._where(filters, exclude="tach_status")
        for value, count in self._query(
            f"SELECT v.tach_status, COUNT(*) FROM v_row v{where} GROUP BY 1", p
        ):
            out["tach_status"][str(value)] = int(count)
        return {g: dict(sorted(vals.items())) for g, vals in out.items()}

    def select(
        self,
        filters: Mapping[str, Any] | None = None,
        *,
        limit: int = 100,
        offset: int = 0,
        order: str = "key",
    ) -> list[dict[str, Any]]:
        if not 0 < int(limit) <= MAX_PAGE:
            raise ValueError(f"limit must be in 1..{MAX_PAGE}, got {limit}")
        if order not in _ORDERS:
            raise ValueError(f"order must be one of {sorted(_ORDERS)}")
        where, params = self._where(filters)
        rows = self._query(
            f"SELECT {', '.join('v.' + c for c in _ROW_COLUMNS)} FROM v_row v{where} "
            f"ORDER BY {_ORDERS[order]} LIMIT ? OFFSET ?",
            [*params, int(limit), int(offset)],
        )
        out = []
        for row in rows:
            d = dict(zip(_ROW_COLUMNS, row, strict=True))
            for flag in ("tach_candidate", "has_audio"):
                if d[flag] is not None:
                    d[flag] = bool(d[flag])
            out.append(d)
        return out

    def keys(self, filters: Mapping[str, Any] | None = None) -> Iterator[str]:
        """Every matching key in key order, streamed from one cursor in pages
        of :data:`MAX_PAGE` (the database is immutable, so the order is stable)."""
        where, params = self._where(filters)
        with self._lock:
            cursor = self._db.execute(
                f"SELECT v.key FROM v_row v{where} ORDER BY v.key", tuple(params)
            )
        try:
            while True:
                with self._lock:
                    page = cursor.fetchmany(MAX_PAGE)
                if not page:
                    return
                for (key,) in page:
                    yield key
        finally:
            cursor.close()

    def _blob_json(self, bid: int | None) -> Any:
        if bid is None:
            return None
        return json.loads(self._query("SELECT json FROM blob WHERE bid = ?", [bid])[0][0])

    def locate(self, key: str) -> tuple[SampleLocation | None, dict[str, Any]]:
        """Parent byte location and stored info of one key (KeyError if absent)."""
        rows = self._query(
            "SELECT shard, span_offset, span_length, fields, has_audio, duration_s, sample_rate, "
            "n_channels FROM recording WHERE key = ?",
            [str(key)],
        )
        if not rows:
            raise KeyError(f"{self.dataset}@{self.version[:12]}: no annotated key {key!r}")
        shard, off, length, fields, has_audio, duration, sr, nch = rows[0]
        loc = (
            None
            if shard is None
            else SampleLocation(
                int(shard), int(off), int(length), tuple(tuple(f) for f in json.loads(fields))
            )
        )
        info = {
            "has_audio": None if has_audio is None else bool(has_audio),
            "duration_s": duration,
            "sample_rate": sr,
            "n_channels": nch,
            "layout": self.layout,
        }
        return loc, info

    def parent_shards(self) -> list[tuple[str, int, int]]:
        """``(digest, size, num_samples)`` of the parent manifest's shards, in order."""
        return [
            (d, int(s), int(n))
            for d, s, n in self._query(
                "SELECT digest, size, num_samples FROM parent_shard ORDER BY shard"
            )
        ]

    def tach_arrays(self, key: str) -> dict[str, Any] | None:
        """Stored estimate of ``key`` with windows as arrays: ``times`` ``(W, 3)``
        float32 start/end/centre relative to ``audio_t_start_s`` (NaN = absent),
        ``accepted`` ``(W,)`` bool, ``rates``/``uncertainty`` ``(W, n_slots)``
        float32 (NaN = unresolved), ``order_support`` ``(W, n_slots)`` uint8
        (255 = absent) or None, ``notes`` per-window dicts (reason/quality)."""
        rows = self._query(
            "SELECT t.status, t.reason, t.identity, t.n_expected, t.role, t.audio_t_start_s, "
            "t.n_windows, t.n_accepted, t.rate_min, t.rate_max, t.summary_bid, t.n_slots, t.win_time, "
            "t.win_accepted, t.win_rates, t.win_unc, t.win_support, t.win_notes "
            "FROM tach t JOIN recording r ON r.rid = t.rid WHERE r.key = ?",
            [str(key)],
        )
        if not rows:
            return None
        (status, reason, identity, n_exp, role, t0, n_w, n_acc, rmin, rmax, sbid, n_slots,
         wt, wa, wr, wu, ws, wn) = rows[0]  # fmt: skip
        shape = (n_w, n_slots)
        return {
            "status": status,
            "reason": reason,
            "identity": identity,
            "n_expected": n_exp,
            "role": role,
            "audio_t_start_s": t0,
            "n_windows": n_w,
            "n_accepted": n_acc,
            "rate_min_rev_s": rmin,
            "rate_max_rev_s": rmax,
            "summary": self._blob_json(sbid),
            "times": np.frombuffer(wt or b"", "<f4").reshape(n_w, 3),
            "accepted": np.frombuffer(wa or b"", np.uint8).astype(bool),
            "rates": np.frombuffer(wr or b"", "<f4").reshape(shape),
            "uncertainty": np.frombuffer(wu or b"", "<f4").reshape(shape),
            "order_support": None if ws is None else np.frombuffer(ws, np.uint8).reshape(shape),
            "notes": json.loads(zlib.decompress(wn))
            if wn is not None
            else [{} for _ in range(n_w)],
        }

    def tach(self, key: str) -> dict[str, Any] | None:
        """Stored estimate of ``key`` with ``windows`` as a list of dicts in the
        estimator's shape (values are the stored float32, shortest repr)."""
        arr = self.tach_arrays(key)
        if arr is None:
            return None
        t0 = float(arr["audio_t_start_s"] or 0.0)
        windows = []
        for w in range(arr["n_windows"]):
            start, end, center = (_f32(v) for v in arr["times"][w])
            rates = arr["rates"][w]
            slots = [int(s) for s in np.flatnonzero(np.isfinite(rates))]
            note = arr["notes"][w]
            support = arr["order_support"]
            windows.append(
                {
                    "start_s": start,
                    "end_s": end,
                    "center_s": center,
                    "center_abs_s": round(t0 + float(arr["times"][w, 2]), 6),
                    "accepted": bool(arr["accepted"][w]),
                    "slots": slots,
                    "rates_rev_s": [_f32(rates[s]) for s in slots],
                    "uncertainty_rev_s": [_f32(arr["uncertainty"][w, s]) for s in slots],
                    "order_support": (
                        [
                            int(support[w, s]) if support[w, s] != _NO_SUPPORT else None
                            for s in slots
                        ]
                        if support is not None
                        else note.get("order_support")
                    ),
                    "quality": note.get("quality"),
                    "reason": note.get("reason"),
                }
            )
        out = {k: v for k, v in arr.items() if k not in _WINDOW_ARRAYS}
        out["windows"] = windows
        return out

    def record(self, key: str, *, windows: bool = True) -> dict[str, Any] | None:
        """The full stored annotation of ``key`` (None if the key is absent);
        ``windows=False`` omits ``tach.windows`` (use :meth:`tach_arrays`)."""
        rows = self._query(
            f"SELECT p.{', p.'.join(FACET_NAMES)}, p.n_rotors, p.n_active, p.provenance_kind, "
            "p.supplied_kind, p.tach_candidate, p.evidence_bid, p.extra_bid, r.wind_bid, r.measure_bid "
            "FROM recording r JOIN profile p ON p.pid = r.pid WHERE r.key = ?",
            [str(key)],
        )
        if not rows:
            return None
        row = rows[0]
        n = len(FACET_NAMES)
        out: dict[str, Any] = {"dataset": self.dataset, "version": self.version, "key": str(key)}
        out.update(zip(FACET_NAMES, row[:n], strict=True))
        n_rotors, n_active, prov, supplied, cand, ebid, xbid, wbid, mbid = row[n:]
        out.update(
            evidence=self._blob_json(ebid),
            n_rotors=n_rotors,
            n_active=n_active,
            provenance_kind=prov,
            supplied_kind=supplied,
            tach_candidate=None if cand is None else bool(cand),
        )
        out.update(self._blob_json(xbid) or {})
        _, info = self.locate(key)
        out["info"] = info
        out["wind_detail"] = self._blob_json(wbid)
        out["measure"] = self._blob_json(mbid)
        tach = self.tach(key) if windows else self.tach_arrays(key)
        if tach is not None and not windows:
            tach = {k: v for k, v in tach.items() if k not in _WINDOW_ARRAYS}
        out["tach"] = tach
        return out

    def locations(
        self, keys: Sequence[str]
    ) -> dict[str, tuple[SampleLocation | None, dict[str, Any]]]:
        return {k: self.locate(k) for k in keys}


def open_index(path: str | Path) -> Index:
    """Open one annotation database read-only."""
    return Index(path)
