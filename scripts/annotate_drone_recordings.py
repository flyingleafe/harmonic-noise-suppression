#!/usr/bin/env python
"""Production campaign for the `drone-recording-annotations` companion release.

Stages (each restartable; unit JSONs under ``<run>/<stage>/raw/`` via
``utils.gridrun``; heavy stages run as omnirun ``uni-cpu`` jobs):

  freeze   exact parent versions of ``rules.DATASETS`` (dload.lock pin, else
           refs/latest stamped now), manifest copies, per-shard source (local
           stage/cache copy sha256-verified, else R2 object present)
  scan     per (dataset, shard): pack footer + ``_frame``/npy/WAV headers by
           ranged reads, ``rules.classify`` per key (rules pass, no PCM)
  measure  per bounded group of selected recordings: steady tachometry on the
           ``tach_role`` entry and the PCM wind pass on ``wind_role``, bounded
           time views of finite samples only, ultrasonic takes decimated
  build    per parent: final records (unknown wind / speed regime revised only
           on supported waveform evidence, re-normalized) -> ``IndexWriter``
           SQLite at ``<run>/staging/<dataset>@<version>.sqlite``
  report   coverage, rules-pass vs post-waveform facet counts, statuses,
           refusal reasons, throughput -> ``<run>/report/``
  show     QA preview of records/measurements for chosen keys
  publish  (authorized only) new companion dataset + direct readback receipt

Never writes parent datasets, never copies waveforms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from utils.gridrun import Unit, add_gridrun_args, gridrun_from_args, unit_path

RELEASE_NAME = "drone-recording-annotations"
#: Bump on any behavioural change of the runner's selection/merging policy.
RUNNER_RECIPE = "annotate-drone-recordings/3"
#: Bytes of float32 PCM one tachometry view may hold.
VIEW_BYTES_CAP = 512 << 20
#: Audio seconds (x channels) one measure unit may analyse.
UNIT_CHANNEL_SECONDS = 4 * 3600.0
_STATUS_RANK = {"resolved": 0, "partial": 1, "unresolved": 2, "too_short": 3, "not_applicable": 4}
_NO_PCM_PROVENANCE = frozenset({"log_only"})
#: Remote shards up to this size holding many samples are fetched in one GET.
PREFETCH_MAX_BYTES = 512 << 20
#: ...and only when samples are small on average (headers would otherwise be most of the cost).
PREFETCH_MAX_SAMPLE_BYTES = 2 << 20


class CampaignError(RuntimeError):
    """A unit cannot complete (missing prerequisite, completeness violation); gridrun records it."""


_TACH_FACT_KEYS = (
    "speed_regime",
    "rotor_activity",
    "n_rotors",
    "n_active",
    "blade_count",
    "shaft_orders",
    "relative_motion",
    "relative_motion_basis",
    "propulsion",
    "sound_content",
    "nominal_rps",
    "speed_supplied",
    "supplied_kind",
    "evidence",
)


# ─── small helpers ─────────────────────────────────────────────────────────


def _jdump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True, default=_jsonable))
    os.replace(tmp, path)


def _jsonable(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    raise TypeError(f"not JSON-serializable: {type(value).__name__}")


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_jsonable)


def _sha256_file(path: Path, chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _safe(name: str) -> str:
    return name.replace("/", "_").replace(":", "_").replace(" ", "_")


def _run_paths(run: Path) -> dict[str, Path]:
    return {
        "freeze": run / "freeze.json",
        "manifests": run / "manifests",
        "extract": run / "extract",
        "classify": run / "classify",
        "measure": run / "measure",
        "staging": run / "staging",
        "report": run / "report",
        "publish": run / "publish",
    }


def _load_freeze(run: Path) -> dict[str, Any]:
    path = _run_paths(run)["freeze"]
    if not path.exists():
        raise CampaignError(f"{path} missing: run `freeze` first")
    return json.loads(path.read_text())


def _manifest(run: Path, dataset: str, version: str) -> Any:
    from dload.manifest import Manifest

    text = (_run_paths(run)["manifests"] / f"{dataset}@{version}.json").read_text()
    m = Manifest.from_json(text)
    if m.version != version:
        raise CampaignError(
            f"frozen manifest copy {dataset}@{version[:12]} does not hash to its version"
        )
    return m


def _opener(local_roots: Iterable[str]) -> Any:
    from data_processing.recording_annotations.scan import RemoteRanges, ShardOpener
    from data_processing.streams import open_repository

    return ShardOpener(local_roots, RemoteRanges.from_repository(open_repository()))


def _selected(freeze: Mapping[str, Any], datasets: list[str] | None) -> list[str]:
    names = sorted(freeze["datasets"])
    if not datasets:
        return names
    unknown = sorted(set(datasets) - set(names))
    if unknown:
        raise SystemExit(f"not in the freeze: {unknown}")
    return [n for n in names if n in set(datasets)]


# ─── freeze ────────────────────────────────────────────────────────────────


def cmd_freeze(args: argparse.Namespace) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from data_processing.recording_annotations import rules
    from data_processing.recording_annotations.scan import RemoteRanges, ShardOpener
    from data_processing.streams import open_repository

    run = Path(args.run)
    paths = _run_paths(run)
    if paths["freeze"].exists():
        raise SystemExit(f"{paths['freeze']} exists: a run freezes once (start a new --run)")
    repo = open_repository()
    scope = sorted(args.datasets or rules.DATASETS)
    outside = sorted(set(scope) - set(rules.PARENT_VERSIONS))
    if outside:
        raise SystemExit(f"outside the frozen recording inventory (parents.json): {outside}")
    opener = ShardOpener(args.local_root, RemoteRanges.from_repository(repo))
    remote_shards: set[str] | None = None
    out: dict[str, Any] = {}
    missing: dict[str, str] = {}
    for name in scope:
        version, how = rules.PARENT_VERSIONS[name], "parents.json"
        try:
            manifest = repo.manifest(name, version)
            if manifest.version != version:
                raise ValueError(f"manifest hashes to {manifest.version}, not {version}")
        except Exception as exc:  # noqa: BLE001 — a missing parent is reported, never skipped silently
            missing[name] = f"{type(exc).__name__}: {exc}"
            continue
        text = manifest.to_json()
        (paths["manifests"]).mkdir(parents=True, exist_ok=True)
        (paths["manifests"] / f"{name}@{version}.json").write_text(text)
        meta = manifest.meta if isinstance(manifest.meta, dict) else {}
        shards = []
        for i, s in enumerate(manifest.shards):
            local = opener.locate(s.digest, s.size)
            shards.append(
                {
                    "shard": i,
                    "digest": s.digest,
                    "size": s.size,
                    "num_samples": s.num_samples,
                    "local": str(local) if local else None,
                }
            )
        out[name] = {
            "version": version,
            "resolved_from": how,
            "created": manifest.created,
            "layout": meta.get("layout"),
            "samples": manifest.num_samples,
            "bytes": manifest.total_bytes,
            "meta": meta,
            "shards": shards,
        }
    # local copies: sha256 must equal the content address; otherwise read remotely
    local = [(n, s) for n, d in out.items() for s in d["shards"] if s["local"]]
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        digests = list(pool.map(lambda ns: _sha256_file(Path(ns[1]["local"])), local))
    bad_local = []
    for (name, s), got in zip(local, digests, strict=True):
        s["local_sha256_ok"] = got == s["digest"]
        if not s["local_sha256_ok"]:
            bad_local.append(f"{name}#{s['shard']}")
            s["local"] = None
    # every shard not served locally must exist on R2 (one listing)
    if any(not s["local"] for d in out.values() for s in d["shards"]):
        remote_shards = {k.rsplit("/", 1)[-1] for k in repo.remote.list("shards/")}
    absent = []
    for name, d in out.items():
        for s in d["shards"]:
            s["remote_present"] = None if remote_shards is None else s["digest"] in remote_shards
            if not s["local"] and not s["remote_present"]:
                absent.append(f"{name}#{s['shard']}:{s['digest'][:12]}")
    freeze = {
        "runner_recipe": RUNNER_RECIPE,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_commit": args.source_commit,
        "local_roots": list(args.local_root),
        "scope": scope,
        "datasets": out,
        "missing_parents": missing,
        "bad_local_copies": bad_local,
        "absent_shards": absent,
    }
    _jdump(paths["freeze"], freeze)
    n_local = sum(1 for d in out.values() for s in d["shards"] if s["local"])
    n_all = sum(len(d["shards"]) for d in out.values())
    print(
        json.dumps(
            {
                "datasets": len(out),
                "shards": n_all,
                "local_verified": n_local,
                "missing_parents": missing,
                "bad_local": bad_local,
                "absent": absent[:20],
            },
            indent=1,
        )
    )
    _collect_context(run, out, opener)
    if absent or missing:
        raise SystemExit(
            f"{len(missing)} parents missing, {len(absent)} shards neither local nor on R2"
        )


#: Per-sample bookkeeping fields (sample-dir-v1 provenance) collected at freeze.
_CONTEXT_FIELD = "metadata"
_CONTEXT_MAX = 64 << 10


def _collect_context(run: Path, frozen: Mapping[str, Any], opener: Any) -> None:
    """Store the ``_meta`` bookkeeping blob and small per-sample ``metadata`` files of
    ``sample-dir-v1`` parents (provenance lives there, not in the data samples)."""
    for name, d in sorted(frozen.items()):
        if d["layout"] != "sample-dir-v1" and "meta_sample" not in d["meta"]:
            continue
        blobs: dict[str, bytes] = {}
        for s in d["shards"]:
            shard = opener.open(s["digest"], s["size"])
            keys, entries = shard.index()
            for key, entry in zip(keys, entries, strict=True):
                span = entry.get(_CONTEXT_FIELD)
                if span is None:
                    continue
                if key == "_meta" or (not key.startswith("_") and span[1] <= _CONTEXT_MAX):
                    blobs[key] = shard.read(int(span[0]), int(span[1]))
            shard.close()
        root = run / "context" / _safe(name)
        root.mkdir(parents=True, exist_ok=True)
        index = {}
        for i, (key, blob) in enumerate(sorted(blobs.items())):
            fn = f"{i:06d}.bin"
            (root / fn).write_bytes(blob)
            index[key] = {
                "file": fn,
                "sha256": hashlib.sha256(blob).hexdigest(),
                "bytes": len(blob),
            }
        _jdump(root / "index.json", index)
        print(
            f"context {name}: {len(index)} blobs ({'with' if '_meta' in index else 'NO'} _meta)",
            flush=True,
        )


_CONTEXT_ROWS: dict[str, dict[str, dict[str, Any]]] = {}


def _context_rows(run: Path, dataset: str) -> dict[str, dict[str, Any]]:
    """``derived.summary_rows`` over the frozen bookkeeping blobs (once per process)."""
    if dataset not in _CONTEXT_ROWS:
        root = run / "context" / _safe(dataset)
        rows: dict[str, dict[str, Any]] = {}
        if (root / "index.json").exists():
            from data_processing.recording_annotations import derived

            index = json.loads((root / "index.json").read_text())
            if dataset in derived.DATASETS:
                blobs = {k: (root / v["file"]).read_bytes() for k, v in index.items()}
                rows = derived.summary_rows(dataset, blobs)
        _CONTEXT_ROWS[dataset] = rows
    return _CONTEXT_ROWS[dataset]


# ─── scan (rules pass) ─────────────────────────────────────────────────────


def _extract_path(run: Path, dataset: str, shard: int) -> Path:
    return run / "extract" / "data" / _safe(dataset) / f"{shard:05d}.json.gz"


def _extract_worker(unit: Unit) -> dict[str, Any]:
    """One shard: footer + descriptor/header ranged reads -> per-key meta/info/location (no PCM)."""
    import gzip

    from data_processing.recording_annotations.index import location_from_entry
    from data_processing.recording_annotations.scan import scan_sample

    p = unit.params
    run = Path(p["run"])
    freeze = _load_freeze(run)
    d = freeze["datasets"][p["dataset"]]
    s = d["shards"][p["shard"]]
    t0 = time.time()
    opener = _opener([str(Path(s["local"]).parents[2])] if s["local"] else [])
    shard = opener.open(s["digest"], s["size"])
    if (
        shard.source == "remote"
        and s["size"] <= PREFETCH_MAX_BYTES
        and s["num_samples"] >= 16
        and s["size"] <= PREFETCH_MAX_SAMPLE_BYTES * s["num_samples"]
    ):
        shard.prefetch()  # dense small clips: one GET instead of per-sample header reads
    keys, entries = shard.index()
    metas: list[Any] = []
    infos: list[Any] = []
    memo_m: dict[str, int] = {}
    memo_i: dict[str, int] = {}
    rows, errors, bookkeeping = [], [], []
    media: Counter = Counter()
    durations: Counter = Counter()
    for key, entry in zip(keys, entries, strict=True):
        if key.startswith("_"):
            bookkeeping.append({"key": key, "fields": sorted(entry)})
            continue
        loc = location_from_entry(p["shard"], entry)
        try:
            sc = scan_sample(
                shard,
                p["shard"],
                key,
                entry,
                dataset=p["dataset"],
                version=d["version"],
                layout=d["layout"] or "",
                fields_map=_fields_map(d),
            )
        except Exception as exc:  # noqa: BLE001 — per-key failure is recorded, never dropped
            errors.append({"key": key, "error": f"{type(exc).__name__}: {exc}"})
            continue
        media.update(_media_kinds(sc))
        dur = sc.info.get("duration_s")
        durations[
            "none" if dur is None else "<1.5s" if dur < 1.5 else "<16s" if dur < 16 else ">=16s"
        ] += 1
        mi = memo_m.setdefault(_canon(sc.meta), len(metas))
        if mi == len(metas):
            metas.append(sc.meta)
        ii = memo_i.setdefault(_canon(sc.info), len(infos))
        if ii == len(infos):
            infos.append(sc.info)
        rows.append(
            [key, mi, ii, [loc.shard, loc.offset, loc.length, [list(f) for f in loc.fields]]]
        )
    shard.close()
    out = _extract_path(run, p["dataset"], p["shard"])
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = gzip.compress(
        _canon({"rows": rows, "metas": metas, "infos": infos}).encode(), 6, mtime=0
    )
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_bytes(blob)
    os.replace(tmp, out)
    remote = opener.remote
    return {
        "dataset": p["dataset"],
        "version": d["version"],
        "shard": p["shard"],
        "digest": s["digest"],
        "source": shard.source,
        "n_pack_keys": len(keys),
        "n_rows": len(rows),
        "bookkeeping": bookkeeping,
        "errors": errors,
        "media": dict(media),
        "durations": dict(durations),
        "n_distinct_meta": len(metas),
        "n_distinct_info": len(infos),
        "data": {"path": str(out), "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()},
        "io": {
            "requests": remote.n_requests if remote else 0,
            "bytes": remote.n_bytes if remote else 0,
        },
        "wall_s": time.time() - t0,
        "max_rss_mb": _rss_mb(),
        "code": _code_stamp(),
    }


def _classify_worker(unit: Unit) -> dict[str, Any]:
    """Rules pass from the extracted metadata of one shard (pure; rerun freely)."""
    import gzip

    from data_processing.recording_annotations import rules

    p = unit.params
    run = Path(p["run"])
    freeze = _load_freeze(run)
    d = freeze["datasets"][p["dataset"]]
    ex = json.loads(
        unit_path(run / "extract", f"x__{_safe(p['dataset'])}__{p['shard']:05d}").read_text()
    )
    blob = _extract_path(run, p["dataset"], p["shard"]).read_bytes()
    if hashlib.sha256(blob).hexdigest() != ex["data"]["sha256"]:
        raise CampaignError(f"{p['dataset']}#{p['shard']}: extract data does not match its receipt")
    data = json.loads(gzip.decompress(blob))
    t0 = time.time()
    context = _context_rows(run, p["dataset"])
    dataset_meta = d["meta"]
    for info in data["infos"]:
        info["dataset_meta"] = dataset_meta  # one shared object, as the classifier contract allows
    records: list[dict[str, Any]] = []
    memo: dict[str, int] = {}
    rows, errors = [], list(ex["errors"])
    for key, mi, ii, loc in data["rows"]:
        meta, info = data["metas"][mi], data["infos"][ii]
        extra = context.get(key)
        try:
            record = rules.classify(p["dataset"], key, {**meta, **extra} if extra else meta, info)
        except Exception as exc:  # noqa: BLE001 — per-key failure is recorded, never dropped
            errors.append({"key": key, "error": f"{type(exc).__name__}: {exc}"})
            continue
        rtext = _canon(record)
        ri = memo.setdefault(rtext, len(records))
        if ri == len(records):
            records.append(record)
        rows.append([key, ri, ii, loc])
    infos = [{k: v for k, v in info.items() if k != "dataset_meta"} for info in data["infos"]]
    return {
        "dataset": p["dataset"],
        "version": d["version"],
        "shard": p["shard"],
        "digest": ex["digest"],
        "source": ex["source"],
        "n_pack_keys": ex["n_pack_keys"],
        "bookkeeping": ex["bookkeeping"],
        "rows": rows,
        "records": records,
        "infos": infos,
        "errors": errors,
        "media": ex["media"],
        "io": ex["io"],
        "wall_s": time.time() - t0,
        "extract_wall_s": ex["wall_s"],
        "max_rss_mb": _rss_mb(),
        "code": _code_stamp(),
        "extract_code": ex["code"],
    }


def _media_kinds(sc: Any) -> list[str]:
    """How every field of one sample was understood (audit of unhandled media)."""
    kinds = []
    used = set()
    for ref in sc.series.values():
        used.add(ref.field)
        used.update(f for f in sc.fields if f.startswith(f"{ref.field}#"))
        if ref.is_audio:
            kinds.append(f"audio:{ref.container}")
        elif ref.index in ("grid", "stamps", "spans"):
            kinds.append(f"track:{ref.index}")
        else:
            kinds.append("array")
    for f in sc.fields:
        if f in used or f == "_frame" or f.startswith("meta/"):
            continue
        if f in sc.errors:
            kinds.append(f"UNREADABLE:{f}")
        elif f in ("meta", "metadata") or f.endswith("json"):
            kinds.append("json")
        else:
            kinds.append(f"UNHANDLED:{f}")
    return kinds


def _code_stamp() -> str:
    """Content address of the code this process runs (set by the job wrapper)."""
    return os.environ.get("HNS_CONDITIONS_SNAPSHOT", "unstamped-working-tree")


def _shard_units(
    prefix: str, run: Path, freeze: Mapping[str, Any], datasets: list[str], max_shards: int | None
) -> list[Unit]:
    units = []
    for n in datasets:
        for s in freeze["datasets"][n]["shards"][: max_shards if max_shards is not None else None]:
            units.append(
                Unit(
                    f"{prefix}__{_safe(n)}__{s['shard']:05d}",
                    {"run": str(run), "dataset": n, "shard": s["shard"]},
                )
            )
    return units


def cmd_extract(args: argparse.Namespace) -> None:
    run = Path(args.run)
    freeze = _load_freeze(run)
    units = _shard_units("x", run, freeze, _selected(freeze, args.datasets), args.max_shards)
    res = gridrun_from_args(
        args, units, _extract_worker, run / "extract", summarize=_extract_summary
    )
    raise SystemExit(1 if res.n_failed else 0)


def cmd_classify(args: argparse.Namespace) -> None:
    run = Path(args.run)
    freeze = _load_freeze(run)
    units = _shard_units("c", run, freeze, _selected(freeze, args.datasets), args.max_shards)
    res = gridrun_from_args(
        args, units, _classify_worker, _run_paths(run)["classify"], summarize=_classify_summary
    )
    raise SystemExit(1 if res.n_failed else 0)


def _extract_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per: dict[str, Any] = defaultdict(lambda: Counter())
    for r in rows:
        c = per[r["dataset"]]
        c["shards"] += 1
        c["keys"] += r["n_rows"]
        c["bookkeeping"] += len(r["bookkeeping"])
        c["errors"] += len(r["errors"])
        c["local_shards"] += r["source"] == "local"
        c["bytes_remote"] += r["io"]["bytes"]
        c["requests"] += r["io"]["requests"]
        c["data_bytes"] += r["data"]["bytes"]
        c["wall_s"] += r["wall_s"]
        c.update({f"media:{k}": v for k, v in r["media"].items()})
        c.update({f"dur:{k}": v for k, v in r["durations"].items()})
    return {k: dict(v) for k, v in sorted(per.items())}


def _classify_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per: dict[str, Any] = defaultdict(lambda: Counter())
    for r in rows:
        c = per[r["dataset"]]
        c["shards"] += 1
        c["keys"] += len(r["rows"])
        c["errors"] += len(r["errors"])
        c["distinct_records"] += len(r["records"])
    return {k: dict(v) for k, v in sorted(per.items())}


def _iter_scan(run: Path, dataset: str, freeze: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    """Classified shard units of one parent (rules-pass records + per-key info/location)."""
    for s in freeze["datasets"][dataset]["shards"]:
        path = unit_path(_run_paths(run)["classify"], f"c__{_safe(dataset)}__{s['shard']:05d}")
        if not path.exists():
            raise CampaignError(f"classify unit missing: {path} (rerun `extract`/`classify`)")
        yield json.loads(path.read_text())


# ─── measurement plan ──────────────────────────────────────────────────────


def _wind_min_s() -> float:
    from data_processing.recording_annotations import wind

    return float(wind.MIN_VIEW_S)


def _tach_min_s() -> float:
    # Campaign threshold; metadata planning and wind must not import the tach estimator.
    return 1.5


def _role(record: Mapping[str, Any], info: Mapping[str, Any], kind: str) -> tuple[str | None, str]:
    """Audio entry a measurement reads, and how it was chosen (explicit role, else a
    role documented as rigidly fixed to the rotors, else ``audio``, else the sole entry)."""
    explicit = record.get(f"{kind}_role")
    audio = info.get("audio_entries") or []
    if explicit:
        return (
            (explicit, "record")
            if explicit in audio
            else (None, f"{kind}_role {explicit!r} is not an audio entry")
        )
    motion = record.get("role_motion") or {}
    basis = record.get("role_motion_basis") or {}
    fixed = sorted(
        (r for r in audio if motion.get(r) == "fixed"),
        key=lambda r: (basis.get(r) == "nominal_hover", r != "audio", r),
    )
    if fixed:  # a rigidly attached role beats a nominally hovering one
        return fixed[0], f"role_motion fixed ({basis.get(fixed[0], 'basis unstated')})"
    if "audio" in audio:
        return "audio", "default"
    if len(audio) == 1:
        return audio[0], "sole audio entry"
    # file-per-stem mixtures: the drone stem, else the observed mixture (speech/target may
    # mask lines there; recorded as such); clean vocals/target/speech stems never
    if "noise" in audio:
        return "noise", "mixture's drone noise stem"
    if "mixture" in audio:
        return "mixture", "observed mixture (drone stem unpublished; speech/target may mask lines)"
    return None, f"no {kind}_role and no drone/mixture stem among {audio}"


def _reason_class(reason: Any) -> str:
    """Refusal reason with numbers masked, so tallies group by cause."""
    return re.sub(r"\d+(\.\d+)?(e-?\d+)?", "#", str(reason or "unstated"))[:120]


def _parent_ref(record: Mapping[str, Any]) -> str:
    parent = record.get("parent")
    if isinstance(parent, Mapping):
        return (
            "/".join(str(parent[k]) for k in ("dataset", "key") if parent.get(k))
            or "parent unnamed"
        )
    return str(parent) if parent else "parent unnamed"


def _binding(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """The exact canonical observation an alias/copy row shares, if the rules bound one."""
    parent = record.get("parent")
    # a binding still asking for verification (``verify``, e.g. pcm_equal) is not trusted
    # for projection: such rows are measured on their own samples
    if (
        isinstance(parent, Mapping)
        and parent.get("binding") in ("exact", "crop")
        and parent.get("dataset")
        and parent.get("key")
        and not parent.get("verify")
    ):
        return dict(parent)
    return None


def _no_audio_reason(info: Mapping[str, Any]) -> str:
    return f"no decodable audio Series (fields: {', '.join(info.get('fields') or []) or 'none'})"


def _short_refusal(
    record: Mapping[str, Any], info: Mapping[str, Any], tach_min_s: float
) -> dict[str, Any] | None:
    """An explicit ``too_short`` (audio never read) for powered drone audio that the
    rules exclude only for its length: the source's ``tach_skip`` verbatim, else a
    rotor-on drone row (not no_drone/off/none/alias) whose audio is under the minimum."""
    skip = record.get("tach_skip")
    if isinstance(skip, Mapping) and skip.get("status") == "too_short":
        return {
            "run": False,
            "status": "too_short",
            "reason": str(skip.get("reason")),
            "source": "rules tach_skip",
        }
    if (
        not info.get("has_audio")
        or record.get("provenance_kind") in ("alias", "log_only", "non_drone")
        or record.get("sound_content") in ("no_drone", "unknown", "not_applicable")
        or record.get("rotor_activity") in ("off", "unknown", "not_applicable")
        or record.get("propulsion") == "none"
        or record.get("speed_regime") == "off"
    ):
        return None
    role, _how = _role(record, info, "tach")
    dur = ((info.get("series") or {}).get(role) or {}).get("duration_s") if role else None
    if dur is not None and dur < tach_min_s:
        return {
            "run": False,
            "status": "too_short",
            "role": role,
            "reason": f"{role} lasts {dur} s < {tach_min_s} s minimum view; audio not read",
            "source": "runner length check",
        }
    return None


def plan_row(
    record: Mapping[str, Any],
    info: Mapping[str, Any],
    *,
    tach_min_s: float,
    wind_min_s: float,
    parent_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """What the waveform pass does for one recording, with an explicit reason either way.

    A row bound exactly (or as a known crop) to a canonical observation that is
    itself measured is *linked* (the parent's result is projected at build);
    any other eligible row, alias or not, is measured on its own clock.
    """
    has_audio = bool(info.get("has_audio"))
    prov = record.get("provenance_kind")
    rotor_on = record.get("rotor_activity") not in ("off",) and record.get("speed_regime") != "off"
    bound = _binding(record)
    out: dict[str, Any] = {}

    want_tach = bool(record.get("tach_candidate")) or (
        record.get("speed_regime") == "constant" and rotor_on
    )
    if not want_tach:
        out["tach"] = _short_refusal(record, info, tach_min_s) or {
            "run": False,
            "reason": "not a tachometry candidate (rules pass)",
        }
    elif (
        bound
        and parent_plan
        and parent_plan["tach"].get("run")
        and parent_plan["tach"].get("role") == (bound.get("role") or "audio")
    ):
        # linked only when the canonical row actually measured the bound role
        out["tach"] = {
            "run": False,
            "link": bound,
            "reason": f"linked to the exact canonical observation {_parent_ref(record)}",
        }
    elif not has_audio or prov in _NO_PCM_PROVENANCE:
        out["tach"] = {"run": False, "status": "not_applicable", "reason": _no_audio_reason(info)}
    else:
        role, how = _role(record, info, "tach")
        dur = (info.get("series", {}).get(role, {}) or {}).get("duration_s") if role else None
        if role is None:
            out["tach"] = {"run": False, "status": "not_applicable", "reason": how}
        elif dur is None or dur < tach_min_s:
            out["tach"] = {
                "run": False,
                "status": "too_short",
                "role": role,
                "reason": f"{role} lasts {dur} s < {tach_min_s} s minimum view; audio not read",
            }
        else:
            out["tach"] = {"run": True, "role": role, "role_from": how, "duration_s": dur}

    out["wind"] = _plan_wind(
        record,
        info,
        prov=prov,
        has_audio=has_audio,
        wind_min_s=wind_min_s,
        bound=bound,
        parent_plan=parent_plan,
    )
    return out


def _plan_wind(
    record: Mapping[str, Any],
    info: Mapping[str, Any],
    *,
    prov: Any,
    has_audio: bool,
    wind_min_s: float,
    bound: Mapping[str, Any] | None = None,
    parent_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Every distinct physical audio role is assessed on its own; a row bound to a
    measured canonical observation links that observation's role instead."""
    if record.get("wind") != "unknown":
        return {
            "run": False,
            "reason": f"wind={record.get('wind')} set by the rules pass",
            "roles": {},
        }
    if (
        bound and parent_plan and bound.get("binding") == "exact"
    ):  # crops are assessed on their own PCM
        prole = bound.get("role") or "audio"
        if (parent_plan["wind"].get("roles", {}).get(prole) or {}).get("run"):
            return {
                "run": False,
                "roles": {},
                "link": bound,
                "reason": f"linked to the exact canonical observation {_parent_ref(record)} role {prole}",
            }
    if not has_audio or prov in _NO_PCM_PROVENANCE:
        return {"run": False, "reason": _no_audio_reason(info), "roles": {}}
    audio = list(info.get("audio_entries") or [])
    if record.get("wind_roles"):  # explicit list from the rules
        wanted, basis = list(record["wind_roles"]), "rules wind_roles"
    elif record.get("sound_content") in ("digital_mix", "both"):
        # a mixture's clean stems (vocals/target/speech/reference) say nothing about wind:
        # the declared wind_role, else the noise stem, else the mixture / sole audio as heard
        if record.get("wind_role"):
            wanted, basis = [record["wind_role"]], "digital mix: declared wind_role"
        elif "noise" in audio:
            wanted, basis = ["noise"], "digital mix: noise stem (no declared wind_role)"
        elif carrier := next((r for r in ("mixture", "audio") if r in audio), None):
            wanted = [carrier]
            basis = f"digital mix: wind as heard in {carrier}; source attribution not established"
        else:
            wanted, basis = (
                [],
                f"digital mix without a noise stem or mixture ({', '.join(audio) or 'no audio'})",
            )
    else:  # recordings: every physical audio role (wind_role is only the preferred one)
        wanted, basis = audio, "all physical audio roles"
    roles: dict[str, dict[str, Any]] = {}
    for role in wanted:
        if role not in audio:
            roles[role] = {"run": False, "reason": f"wind role {role!r} is not an audio entry"}
            continue
        dur = info["series"][role].get("duration_s")
        if dur is None or dur < wind_min_s:
            roles[role] = {
                "run": False,
                "too_short": True,
                "duration_s": dur,
                "reason": f"{role} lasts {dur} s < {wind_min_s} s minimum view; audio not read",
            }
        else:
            roles[role] = {"run": True, "duration_s": dur}
    return {
        "run": any(t["run"] for t in roles.values()),
        "roles": roles,
        "basis": basis,
        "reason": f"per-role PCM wind pass ({basis})" if roles else basis,
    }


def _channel_seconds(
    info: Mapping[str, Any], plan: Mapping[str, Any], tasks: tuple[str, ...] = ("tach", "wind")
) -> float:
    """Audio channel-seconds the selected tasks of one plan will read."""
    total = 0.0
    work = [(plan["tach"].get("role"), plan["tach"])] if "tach" in tasks else []
    if "wind" in tasks:
        work += list(plan["wind"].get("roles", {}).items())
    for role, task in work:
        if task.get("run"):
            s = info["series"][role]
            total += float(s.get("duration_s") or 0.0) * max(1, int(s.get("n_channels") or 1))
    return total


def _fields_map(d: Mapping[str, Any]) -> dict[str, str]:
    """Manifest stem -> filename map of file-per-field layouts (sample-dir-v1)."""
    fields = d["meta"].get("fields") if isinstance(d.get("meta"), Mapping) else None
    return dict(fields) if isinstance(fields, Mapping) else {}


def _measured_regime() -> Any:
    from data_processing.recording_annotations import measured_regime

    return measured_regime


def _regime_datasets() -> frozenset[str]:
    """Parents with authoritative source speed tracks (read in the waveform stage)."""
    return frozenset(_measured_regime().DATASETS)


class _Planner:
    """``plan_row`` with the exact parent's own plan looked up for bound rows
    (classify units of a parent loaded once per process)."""

    def __init__(self, run: Path, freeze: Mapping[str, Any]) -> None:
        self.run, self.freeze = run, freeze
        self.tmin, self.wmin = _tach_min_s(), _wind_min_s()
        self._entries: dict[str, dict[str, tuple[dict, dict]]] = {}

    def entry(self, dataset: str, key: str) -> tuple[dict, dict] | None:
        if dataset not in self.freeze["datasets"]:
            return None
        if dataset not in self._entries:
            self._entries[dataset] = {
                k: (su["records"][ri], su["infos"][ii])
                for su in _iter_scan(self.run, dataset, self.freeze)
                for k, ri, ii, _ in su["rows"]
            }
        return self._entries[dataset].get(key)

    def __call__(self, record: Mapping[str, Any], info: Mapping[str, Any]) -> dict[str, Any]:
        bound = _binding(record)
        parent_plan = None
        if bound:
            ent = self.entry(bound["dataset"], bound["key"])
            if ent is not None:
                parent_plan = plan_row(ent[0], ent[1], tach_min_s=self.tmin, wind_min_s=self.wmin)
        plan = plan_row(
            record, info, tach_min_s=self.tmin, wind_min_s=self.wmin, parent_plan=parent_plan
        )
        if bound and parent_plan is None:
            plan["link_missing"] = (
                f"bound parent {_parent_ref(record)} is not a row of this campaign"
            )
        return plan


ALL_TASKS = ("regime", "tach", "wind")


def _units_measure(
    run: Path, freeze: Mapping[str, Any], datasets: list[str], tasks: tuple[str, ...] = ALL_TASKS
) -> list[Unit]:
    """Bounded measure units for the selected ``tasks``; unit ids carry the task set, so
    a wind-only pass and a later tach/regime pass are disjoint, restartable partitions."""
    planner = _Planner(run, freeze)
    prefix = "m" if tuple(sorted(tasks)) == ALL_TASKS else "m-" + "-".join(sorted(tasks))
    units: list[Unit] = []
    for name in datasets:
        regime = "regime" in tasks and name in _regime_datasets()
        for su in _iter_scan(run, name, freeze):
            group: list[str] = []
            budget = 0.0
            g = 0

            def flush(group: list[str], g: int, shard: int = su["shard"], name: str = name) -> None:
                units.append(
                    Unit(
                        f"{prefix}__{_safe(name)}__{shard:05d}__{g:04d}",
                        {
                            "run": str(run),
                            "dataset": name,
                            "shard": shard,
                            "keys": group,
                            "tasks": list(tasks),
                        },
                    )
                )

            for key, ri, ii, _loc in su["rows"]:
                plan = planner(su["records"][ri], su["infos"][ii])
                cost = _channel_seconds(su["infos"][ii], plan, tasks) + (60.0 if regime else 0.0)
                if cost <= 0:
                    continue
                if group and budget + cost > UNIT_CHANNEL_SECONDS:
                    flush(group, g)
                    group, budget, g = [], 0.0, g + 1
                group.append(key)
                budget += cost
            if group:
                flush(group, g)
    return units


# ─── measurement (waveform pass) ───────────────────────────────────────────


def _spans(record: Mapping[str, Any], ref: Any, role: str) -> list[tuple[float, float]]:
    """Valid Frame-clock spans of one role (``role_valid_spans_s[role]``, else the
    record-level ``valid_spans_s`` when it belongs to this role, else the whole Series)."""
    t0, t1 = float(ref.t_start_s), float(ref.t_end_s)
    spans = _for_role(record, role, "valid_spans_s", ("valid_spans_s", "tach_role"))
    if not spans:
        return [(t0, t1)]
    out = []
    for a, b in spans:
        a, b = max(t0, float(a)), min(t1, float(b))
        if b > a:
            out.append((a, b))
    return out


def _views(spans: list[tuple[float, float]], view_s: float) -> list[tuple[float, float]]:
    """Equal-length consecutive views (no tiny tails) covering each span."""
    out = []
    for a, b in spans:
        n = max(1, math.ceil((b - a) / view_s - 1e-9))
        step = (b - a) / n
        out += [(a + i * step, a + (i + 1) * step) for i in range(n)]
    return out


def _mic_positions(shard: Any, sc: Any, role: str, channels: list[int] | None) -> Any:
    import io

    import numpy as np

    ref = sc.series.get("mic_pos")
    if role != "audio" or ref is None or ref.container != "npy" or ref.length > (64 << 10):
        return None
    arr = np.load(io.BytesIO(shard.read(ref.offset, ref.length)), allow_pickle=False)
    n = sc.series[role].n_channels
    if arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] != 3:
        return None
    return (arr if channels is None else arr[channels]).tolist()


def _measure_tach(
    shard: Any, sc: Any, record: Mapping[str, Any], task: Mapping[str, Any], view_s: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    from data_processing.recording_annotations.scan import finite_runs, read_view
    from tracking import steady_tachometry as st

    role = task["role"]
    ref = sc.series[role]
    channels = _for_role(record, role, "tach_channels", ("tach_channels", "tach_role"))
    nch = len(channels) if channels else ref.n_channels
    facts = _tach_facts(record, role)
    sr = float(ref.rate_hz)
    cap = VIEW_BYTES_CAP / (4.0 * nch * sr)
    st_min = float(getattr(st, "MIN_VIEW_S", 1.5))
    view = max(st_min, min(view_s, cap))
    parts, seconds, n_runs = [], 0.0, 0
    spans = _spans(record, ref, role)
    views = _views(spans, view)
    for a, b in views:  # native-rate finite views; the estimator anti-aliases and tiles itself
        x, t_view = read_view(shard, ref, a, b, channels)
        for i0, i1 in finite_runs(x, int(math.ceil(st_min * sr))):
            # private candidates with their independent spectral evidence; never published
            parts.append(
                st.estimate(x[:, i0:i1], sr, facts, t_start=t_view + i0 / sr, candidate_only=True)
            )
            seconds += (i1 - i0) / sr
            n_runs += 1
        del x
    if parts:
        tach = st.merge_estimates(parts) if len(parts) > 1 else parts[0]
    else:
        tach = {
            "status": "too_short",
            "reason": f"no finite contiguous run >= {st_min} s in {role}",
            "identity": None,
            "n_expected": record.get("n_active"),
            "windows": [],
        }
    meas = {
        "role": role,
        "role_from": task["role_from"],
        "channels": channels,
        "rate_hz": sr,
        "spans_s": spans,
        "view_s": view,
        "n_views": len(views),
        "n_finite_runs": n_runs,
        "seconds_analysed": seconds,
        "audio_t_start_s": ref.t_start_s,
        "recipe": getattr(st, "RECIPE", None),
    }
    return tach, meas


def _for_role(record: Mapping[str, Any], role: str, name: str, *fallbacks: tuple[str, str]) -> Any:
    """Role-specific value: ``role_<name>[role]``, else a record-level value whose
    owning role (``fallbacks`` = (value key, role key)) is this role."""
    per = record.get(f"role_{name}") or {}
    if role in per:
        return per[role]
    for key, role_key in fallbacks:
        if record.get(key) is not None and role == (record.get(role_key) or "audio"):
            return record[key]
    return None


def _measure_wind(
    shard: Any, sc: Any, record: Mapping[str, Any], role: str, task: Mapping[str, Any]
) -> dict[str, Any]:
    from data_processing.recording_annotations import wind
    from data_processing.recording_annotations.scan import finite_runs, read_view

    ref = sc.series[role]
    channels = _for_role(
        record,
        role,
        "wind_channels",
        ("wind_channels", "wind_role"),
        ("tach_channels", "tach_role"),
    )
    cap = int(getattr(wind, "MAX_CHANNELS", 16))
    if (
        channels is None and ref.n_channels > cap
    ):  # wind.assess would subsample evenly anyway; bound IO
        channels = sorted({round(i * (ref.n_channels - 1) / (cap - 1)) for i in range(cap)})
    # the view is already the channel subset; documented exposure stays out of the PCM test
    facts = {k: v for k, v in record.items() if k not in ("wind_channels", "wind_exposed_channels")}
    exposed = (record.get("wind_exposed_channels") or {}).get(role)
    rm = (record.get("role_motion") or {}).get(role)
    if rm:
        facts["relative_motion"] = rm
    mp = _mic_positions(shard, sc, role, channels)
    if mp is not None:
        facts["mic_positions"] = mp
    spans = _spans(record, ref, role)
    total = sum(b - a for a, b in spans)
    if total <= float(wind.MAX_VIEW_S) and len(spans) == 1:
        windows = spans
    else:  # MAX_VIEWS evenly spaced PREFERRED_VIEW_S views over the valid spans
        pref = float(wind.PREFERRED_VIEW_S)
        n = int(wind.MAX_VIEWS)
        windows = []
        usable = [(a, b) for a, b in spans if b - a >= float(wind.MIN_VIEW_S)]
        length = sum(b - a for a, b in usable)
        for i in range(n):
            target = (i + 0.5) * length / n
            for a, b in usable:
                if target <= b - a:
                    c = a + target
                    w0, w1 = max(a, c - pref / 2), min(b, c + pref / 2)
                    if not windows or w0 >= windows[-1][1]:
                        windows.append((w0, w1))
                    break
                target -= b - a
    results, seconds = [], 0.0
    sr = float(ref.rate_hz)
    min_n = int(math.ceil(float(wind.MIN_VIEW_S) * sr))
    for a, b in windows:  # native rate: wind.assess decimates internally
        x, _t = read_view(shard, ref, a, b, channels)
        for i0, i1 in finite_runs(x, min_n):
            results.append(wind.assess(x[:, i0:i1], sr, facts, role=role))
            seconds += (i1 - i0) / sr
        del x
    if not results:
        res = wind.too_short(seconds, role)
    elif len(results) == 1:
        res = results[0]
    else:
        res = wind.merge(results, role=role)
    res = dict(res)
    if channels is not None:  # assess saw the view's rows 0..k-1: report the physical channel ids
        phys = [int(c) for c in channels]
        res["channels"] = [
            {**c, "channel": phys[int(c["channel"])], "view_index": int(c["channel"])}
            for c in res.get("channels") or []
        ]
        for key in ("windy_channels", "calm_channels"):
            if key in res:
                res[key] = [phys[int(i)] for i in res[key]]
    res["measure"] = {
        "role": role,
        "channels": channels,
        "views_s": windows,
        "seconds_analysed": seconds,
        "rate_hz": sr,
        "duration_s": task.get("duration_s"),
        "recipe": getattr(wind, "RECIPE", None),
        "documented_exposed_channels": exposed,
    }
    return res


def _measure_worker(unit: Unit) -> dict[str, Any]:
    from data_processing.recording_annotations.scan import scan_shard

    p = unit.params
    run = Path(p["run"])
    freeze = _load_freeze(run)
    d = freeze["datasets"][p["dataset"]]
    s = d["shards"][p["shard"]]
    su = json.loads(
        unit_path(
            _run_paths(run)["classify"], f"c__{_safe(p['dataset'])}__{p['shard']:05d}"
        ).read_text()
    )
    by_key = {k: (su["records"][ri], su["infos"][ii]) for k, ri, ii, _ in su["rows"]}
    opener = _opener([str(Path(s["local"]).parents[2])] if s["local"] else [])
    shard = opener.open(s["digest"], s["size"])
    planner = _Planner(run, freeze)
    regime_mod = _measured_regime() if p["dataset"] in _regime_datasets() else None
    tasks = set(p.get("tasks") or ALL_TASKS)
    out = []
    t_unit = time.time()
    for sc in scan_shard(
        shard,
        p["shard"],
        dataset=p["dataset"],
        version=d["version"],
        layout=d["layout"] or "",
        fields_map=_fields_map(d),
        keys=p["keys"],
    ):
        record, info = by_key[sc.key]
        plan = planner(record, info)
        row: dict[str, Any] = {"key": sc.key, "plan": plan}
        t0 = time.time()
        b0 = opener.remote.n_bytes if opener.remote else 0
        try:
            if (
                regime_mod is not None and "regime" in tasks
            ):  # source telemetry; never fed to the fit
                row["native_regime"] = regime_mod.from_scan(shard, sc, p["dataset"])
            if plan["tach"].get("run") and "tach" in tasks:
                cand, row["tach_measure"] = _measure_tach(
                    shard, sc, record, plan["tach"], p["view_s"]
                )
                row["tach_candidates"] = cand  # private: only corroborated windows are published
                if (
                    regime_mod is not None
                    and plan["tach"]["role"] == "audio"
                    and cand.get("windows")
                ):
                    # compact native-sensor summaries per fit window, for post-fit corroboration only
                    row["tach_references"] = regime_mod.reference_windows(
                        shard, sc, p["dataset"], cand["windows"]
                    )
            if "wind" in tasks:
                winds = {
                    role: _measure_wind(shard, sc, record, role, task)
                    for role, task in plan["wind"]["roles"].items()
                    if task.get("run")
                }
                if winds:
                    row["wind_roles"] = winds
        except Exception as exc:  # noqa: BLE001 — per-recording failure is reported with its traceback
            import traceback

            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()[-4000:]
        row["wall_s"] = time.time() - t0
        row["bytes_remote"] = (opener.remote.n_bytes if opener.remote else 0) - b0
        out.append(row)
    shard.close()
    return {
        "dataset": p["dataset"],
        "version": d["version"],
        "shard": p["shard"],
        "source": shard.source,
        "rows": out,
        "wall_s": time.time() - t_unit,
        "max_rss_mb": _rss_mb(),
        "code": _code_stamp(),
        "io": {
            "requests": opener.remote.n_requests if opener.remote else 0,
            "bytes": opener.remote.n_bytes if opener.remote else 0,
        },
    }


def cmd_measure(args: argparse.Namespace) -> None:
    run = Path(args.run)
    freeze = _load_freeze(run)
    tasks = tuple(sorted(set(args.tasks)))
    units = _units_measure(run, freeze, _selected(freeze, args.datasets), tasks)
    if args.keys:
        wanted = set(args.keys)
        units = [
            Unit(
                f"{u.uid}__preview",
                {**u.params, "keys": [k for k in u.params["keys"] if k in wanted]},
            )
            for u in units
            if wanted & set(u.params["keys"])
        ]
    units = [Unit(u.uid, {**u.params, "view_s": args.view_s}) for u in units]
    if args.shard_of is not None:
        k, n = args.shard_of
        units = [u for i, u in enumerate(units) if i % n == k]
    print(
        json.dumps({"units": len(units), "keys": sum(len(u.params["keys"]) for u in units)}),
        flush=True,
    )
    # concurrent partitions share raw/ (disjoint unit ids) but never one summary file
    res = gridrun_from_args(
        args,
        units,
        _measure_worker,
        _run_paths(run)["measure"],
        summarize=_measure_summary,
        summary_name=f"summary.{_safe(args.partition)}.json",
    )
    raise SystemExit(1 if res.n_failed else 0)


def _measure_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per: dict[str, Any] = defaultdict(lambda: Counter())
    for r in rows:
        c = per[r["dataset"]]
        c["units"] += 1
        c["wall_s"] += r["wall_s"]
        c["bytes_remote"] += r["io"]["bytes"]
        for row in r["rows"]:
            c["keys"] += 1
            c["errors"] += "error" in row
            if "tach_candidates" in row:
                c[f"private_tach_{row['tach_candidates'].get('status')}"] += 1
            for role, res in (row.get("wind_roles") or {}).items():
                c[f"wind[{role}]_{res.get('wind')}"] += 1
    return {k: dict(v) for k, v in sorted(per.items())}


def _measure_rows(
    run: Path, dataset: str, candidate_snapshot: str | None = None
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Measured rows by key; a sealed build reads private tach only from its approved snapshot."""
    out: dict[str, dict[str, Any]] = {}
    codes: set[str] = set()
    raw = _run_paths(run)["measure"] / "raw"
    for path in sorted(raw.glob(f"m*__{_safe(dataset)}__*.json")):  # every task partition
        if path.stem.endswith("__preview"):
            continue
        unit = json.loads(path.read_text())
        if unit["dataset"] != dataset:
            continue
        stamp = unit.get("code", "unstamped")
        unit_is_old_tach = (
            candidate_snapshot is not None
            and not stamp.startswith(candidate_snapshot + " ")
            and path.name.startswith(("m-regime-tach__", "m-tach__", "m__"))
        )
        if unit_is_old_tach:
            continue
        codes.add(stamp)
        for row in unit["rows"]:
            out.setdefault(row["key"], {}).update(row)
    return out, codes


# ─── final records ─────────────────────────────────────────────────────────


def _tach_facts(record: Mapping[str, Any], role: str | None) -> dict[str, Any]:
    """Facts the estimator sees for one role (the role's own geometry overrides the summary)."""
    facts = {k: record.get(k) for k in _TACH_FACT_KEYS}
    rm = (record.get("role_motion") or {}).get(role) if role else None
    if rm:
        facts["relative_motion"] = rm
        facts["relative_motion_basis"] = (record.get("role_motion_basis") or {}).get(role)
    return facts


def _published(row: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    """Row with its publishable ``tach``: the private candidates gated by
    ``steady_tachometry.corroborate`` (PCM-free; independent spectral evidence stored with
    the candidates plus native sensor windows where allowed). Candidates never publish."""
    out = dict(row)
    cand = out.pop("tach_candidates", None)
    references = out.pop("tach_references", None)
    if cand is not None:
        if not cand["windows"] and cand["status"] == "too_short":
            # No finite minimum-duration span, hence no estimates to corroborate.
            out["tach"] = cand
        else:
            from tracking import steady_tachometry as st

            if cand["windows"] and cand.get("publication_state") != "private_candidates":
                raise CampaignError(f"{row['key']}: missing private candidate evidence")
            role = (row.get("tach_measure") or {}).get("role")
            out["tach"] = st.corroborate(cand, _tach_facts(record, role), references=references)
            if out["tach"].get("publication_state") != "corroborated":
                raise CampaignError(f"{row['key']}: candidate gate did not run")
    elif (existing := out.get("tach") or {}).get("publication_state") != "corroborated" and any(
        w.get("rates_rev_s") for w in existing.get("windows", [])
    ):
        raise CampaignError(f"{row.get('key')}: ungated acoustic rates")
    return out


def _project(
    link: Mapping[str, Any],
    prow: Mapping[str, Any],
    plan: Mapping[str, Any],
    record: Mapping[str, Any],
    info: Mapping[str, Any],
) -> dict[str, Any]:
    """The bound canonical observation's tachometry on this row's own clock.

    Parent windows are absolute (parent Frame clock); ``offset_s`` is this row's
    t=0 relative to the parent's analysed *audio start* (may be negative for a
    late recorder group). Child time = (abs - parent_audio_t0) - offset + child_t0.
    ``crop`` keeps only windows wholly inside [0, duration_s] (the true overlap) and
    recomputes status/regime from what survives; rates are copied verbatim. Wind is
    never projected for a crop (the row is assessed on its own PCM)."""
    pmeas = prow.get("tach_measure") or {}
    p_t0 = float(pmeas.get("audio_t_start_s") or 0.0)
    crop = link["binding"] == "crop"
    off = float(link.get("offset_s") or 0.0) if crop else 0.0
    dur = link.get("duration_s") if crop else None
    audio = info.get("audio_entries") or []
    c_role = record.get("tach_role") or (
        "audio" if "audio" in audio else (audio[0] if audio else None)
    )
    c_t0 = float(((info.get("series") or {}).get(c_role) or {}).get("t_start_s") or 0.0)
    out: dict[str, Any] = {}
    if plan["tach"].get("link"):
        if "tach" not in prow or pmeas.get("role") != (link.get("role") or "audio"):
            raise CampaignError(
                f"{link['dataset']}/{link['key']}: bound tach role was not measured"
            )
        parent = prow["tach"]
        windows = []
        for w in parent.get("windows") or []:
            sr = (w["start_s"] - p_t0) - off
            er = (w["end_s"] - p_t0) - off
            if crop and (sr < -1e-6 or (dur is not None and er > float(dur) + 1e-6)):
                continue
            s, e = c_t0 + max(0.0, sr), c_t0 + er
            windows.append({**w, "start_s": s, "end_s": e, "center_s": s + (e - s) / 2})
        tach = {**parent, "windows": windows, "projected_from": dict(link)}
        if crop:
            accepted = [w for w in windows if w.get("accepted", bool(w.get("rates_rev_s")))]
            accepted_s = sum(w["end_s"] - w["start_s"] for w in accepted)
            regime = dict(parent.get("regime_evidence") or {})
            regime.update(
                suggested="unknown",
                accepted_s=accepted_s,
                analyzed_s=float(dur) if dur is not None else None,
                note="crop of a longer canonical analysis; regime not re-derived",
            )
            tach["regime_evidence"] = regime
            if not accepted:
                tach.update(
                    status="unresolved",
                    reason="no accepted canonical window lies inside the bound crop",
                )
            elif parent["status"] != "resolved" or dur is None or accepted_s + 1e-3 < float(dur):
                tach.update(
                    status="partial", reason="canonical windows cover only part of the bound crop"
                )
        out["tach"] = tach
        out["tach_measure"] = {
            **pmeas,
            "role": c_role,
            "projected_from": dict(link),
            "audio_t_start_s": c_t0,
            "parent_audio_t_start_s": p_t0,
            "projection_chain": [dict(link), *pmeas.get("projection_chain", [])],
        }
    prole = link.get("role") or "audio"
    if plan["wind"].get("link"):
        if prole not in (prow.get("wind_roles") or {}):
            raise CampaignError(
                f"{link['dataset']}/{link['key']}: bound wind role was not measured"
            )
        audio = info.get("audio_entries") or []
        role = record.get("wind_role") or (
            "audio" if "audio" in audio else (audio[0] if audio else prole)
        )
        parent_res = prow["wind_roles"][prole]
        chans = link.get("channels")
        if chans is None:
            out["wind_roles"] = {role: {**parent_res, "projected_from": dict(link)}}
        else:  # the bound mic(s) only, from wind.assess's own per-channel verdicts
            picked = [c for c in parent_res.get("channels") or [] if c.get("channel") in set(chans)]
            verdicts = {c.get("wind") for c in picked}
            verdict = (verdicts.pop() if len(verdicts) == 1 else "mixed") if picked else "unknown"
            out["wind_roles"] = {
                role: {
                    "wind": verdict,
                    "channels": picked,
                    "projected_from": dict(link),
                    "reason": (
                        f"per-channel verdicts of canonical channels {chans}"
                        if picked
                        else f"canonical result has no per-channel verdict for channels {chans}"
                    ),
                    "recipe": parent_res.get("recipe"),
                }
            }
    return out


def _single_rotor_identity(tach: dict[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    """A run with exactly one documented active rotor (``active_rotor_ids``) whose every
    accepted window resolved exactly one rate: that rate is the named physical rotor
    (slot 0, ``n_expected`` 1). Multi-rotor rates stay anonymous, never rank-named."""
    ids = record.get("active_rotor_ids")
    if not (isinstance(ids, list) and len(ids) == 1 and record.get("n_active") in (1, None)):
        return tach
    rated = [w for w in tach.get("windows") or [] if w.get("rates_rev_s")]
    if not rated or any(len(w["rates_rev_s"]) != 1 for w in rated):
        return tach
    return {
        **tach,
        "identity": "physical",
        "rotor_ids": [str(ids[0])],
        "n_expected": 1,
        "windows": [{**w, "slots": [0]} if w.get("rates_rev_s") else w for w in tach["windows"]],
    }


def _strong_acoustic_change(tach: Mapping[str, Any]) -> dict[str, Any] | None:
    """Robust full-rig rate change, not a comparison of named rotor identities.

    Require ≥90% complete windows over analysed active audio and support from
    at least two different windows at each end of the change. Restrict to
    corroborated, unprojected rates; idle alone cannot satisfy the rate range.
    """
    if (
        tach.get("publication_state") != "corroborated"
        or tach.get("projected_from")
        or not (expected := tach.get("n_expected"))
    ):
        return None
    windows = [
        w
        for w in tach.get("windows") or []
        if w.get("accepted")
        and len(w["rates_rev_s"]) == expected
        and (w.get("quality") or {}).get("acceptance_basis")
    ]
    if len(windows) < 4:
        return None
    regime = tach.get("regime_evidence") or {}
    active_s = max(0.0, float(regime.get("analyzed_s") or 0) - float(regime.get("quiet_s") or 0))
    complete_s = sum(w["end_s"] - w["start_s"] for w in windows)
    if not active_s or complete_s < 0.9 * active_s:
        return None

    import numpy as np

    ranks = np.array([sorted(w["rates_rev_s"]) for w in windows])
    q10, median, q90 = np.quantile(ranks, [0.1, 0.5, 0.9], axis=0)
    ordered = np.sort(ranks, axis=0)
    spread = np.minimum(q90, ordered[-2]) - np.maximum(q10, ordered[1])
    threshold = np.maximum(2.0, 0.05 * median)
    if not np.any(spread > threshold):
        return None
    return {
        "suggested": "mixed" if regime.get("suggested") == "mixed" else "variable",
        "active_coverage": min(1.0, complete_s / active_s),
        "complete_windows": len(windows),
        "robust_range_rev_s": spread.tolist(),
        "threshold_rev_s": threshold.tolist(),
    }


def finalize(
    record: Mapping[str, Any],
    info: Mapping[str, Any],
    plan: Mapping[str, Any],
    mrow: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None, dict[str, Any], list[str]]:
    """Fold waveform-stage evidence into one record: ``(record, tach, wind, measure, qa_flags)``.

    An explicitly constant source protocol stays constant: the facet describes
    the operating condition, not the precision of its RPM. Native and acoustic
    evidence retain any contrary observations separately. Strong telemetry
    may revise other protocol labels; acoustics can fill unknowns. Wind: each
    assessed audio role is combined by ``wind.combine`` with the direct source facet (direct
    evidence wins; unassessed roles stay unknown). Other disagreements are
    reported as QA flags, never applied.
    """
    from data_processing.recording_annotations import schema

    rec = dict(record)
    evidence = dict(rec.get("evidence") or {})
    flags: list[str] = []
    measure: dict[str, Any] = {"runner_recipe": RUNNER_RECIPE, "plan": plan}
    tach = wind_res = None
    if mrow and mrow.get("error"):
        measure["error"] = mrow["error"]
        flags.append("measure_error")

    native = (mrow or {}).get("native_regime")
    if native:
        measure["native_regime"] = native
        sugg = native.get("suggested") or native.get("speed_regime")
        old = rec.get("speed_regime")
        if sugg and sugg != "unknown" and sugg != old:
            note = f"Source telemetry ({native.get('source')}; {native.get('basis')}): {native.get('evidence')}"
            if old == "unknown":
                rec["speed_regime"] = sugg
                evidence["speed_regime"] = note
            elif old == "constant":
                flags.append(f"speed_regime_protocol_vs_telemetry:constant!={sugg}")
            elif native.get("strong"):
                rec["speed_regime_protocol"] = old
                rec["speed_regime_protocol_evidence"] = evidence.get("speed_regime")
                rec["speed_regime"] = sugg
                evidence["speed_regime"] = note
                flags.append(f"speed_regime_protocol_overridden:{old}->{sugg}")
            else:
                flags.append(f"speed_regime_protocol_vs_weak_telemetry:{old}!={sugg}")

    t_plan = plan.get("tach", {})
    if mrow and "tach" in mrow:
        tach = mrow["tach"]
        measure["tach"] = mrow.get("tach_measure")
        regime = tach.get("regime_evidence") or {}
        sugg = regime.get("suggested")
        if sugg and sugg != "unknown" and tach.get("status") in ("resolved", "partial"):
            if rec.get("speed_regime") == "unknown":
                rec["speed_regime"] = sugg
                evidence["speed_regime"] = (
                    f"Acoustic steady tachometry ({(measure.get('tach') or {}).get('recipe')}, status {tach['status']}): "
                    f"accepted {regime.get('accepted_s', 0):.1f} s of {regime.get('analyzed_s', 0):.1f} s analysed, "
                    f"max within-window drift {regime.get('max_within_window_drift_rev_s', 0):.2f} rev/s"
                )
            elif rec.get("speed_regime") != sugg:
                rec["speed_regime_acoustic_disagreement"] = {
                    "kept": rec.get("speed_regime"),
                    "suggested": sugg,
                    "reason": regime.get("reason"),
                }
                flags.append(f"speed_regime_vs_acoustic:{rec.get('speed_regime')}!={sugg}")
        if rec.get("speed_regime") == "constant" and not (native or {}).get("strong"):
            change = _strong_acoustic_change(tach)
            if change and record.get("speed_regime") == "constant":
                rec["speed_regime_acoustic_disagreement"] = {"kept": "constant", **change}
            elif change:
                rec["speed_regime_protocol"] = "constant"
                rec["speed_regime_protocol_evidence"] = evidence.get("speed_regime")
                rec["speed_regime"] = change["suggested"]
                rec["speed_regime_acoustic_disagreement"] = change
                evidence["speed_regime"] = (
                    f"Corroborated acoustic rates: {change['complete_windows']} complete windows, "
                    f"{change['active_coverage']:.0%} of analysed active time, robust rate range "
                    f"{max(change['robust_range_rev_s']):.2f} rev/s despite the constant protocol label"
                )
                flags.append(f"speed_regime_acoustic_override:constant->{change['suggested']}")
    elif t_plan.get("status"):
        tach = {
            "status": t_plan["status"],
            "reason": t_plan["reason"],
            "identity": None,
            "n_expected": rec.get("n_active"),
            "windows": [],
        }

    w_roles = dict(plan.get("wind", {}).get("roles", {}))
    for role in (mrow or {}).get("wind_roles") or {}:  # projected from an exactly bound parent
        w_roles.setdefault(role, {"run": False, "projected": True})
    if w_roles and rec.get("wind") == "unknown":
        from data_processing.recording_annotations import wind

        per_role: dict[str, Any] = {}
        measured_wind = (mrow or {}).get("wind_roles") or {}
        for role, task in w_roles.items():
            if role in measured_wind:
                per_role[role] = measured_wind[role]
            elif task.get("too_short"):
                per_role[role] = wind.too_short(task.get("duration_s") or 0.0, role)
            else:
                per_role[role] = {"wind": "unknown", "reason": task.get("reason") or "not assessed"}
        direct = {
            "wind": rec.get("wind"),
            "evidence": evidence.get("wind"),
            "role_wind": rec.get("role_wind"),
        }
        # On real multirole recordings a preference locates the microphone, not the verdict.
        primary = rec.get("wind_role") if len(per_role) == 1 else None
        comb = wind.combine(per_role, direct=direct, wind_role=primary)
        if rec.get("wind") == "unknown" and comb.get("wind") not in (None, "unknown"):
            rec["wind"] = comb["wind"]
            evidence["wind"] = comb.get("evidence") or "PCM wind pass"
        if comb.get("role_wind"):
            rec["role_wind"] = {**(rec.get("role_wind") or {}), **comb["role_wind"]}
        wind_res = {"roles": per_role, "combined": comb}

    role = t_plan.get("role")  # the entry actually analysed, when the runner chose it
    if role and not rec.get("tach_role"):
        rec["tach_role"] = role
    rec["evidence"] = evidence
    rec = schema.normalize(rec, has_audio=bool(info.get("has_audio")))
    if rec.get("tach_candidate") and tach is None:
        tach = {
            "status": "not_applicable",
            "reason": t_plan.get("reason") or "no waveform estimate",
            "identity": None,
            "n_expected": rec.get("n_active"),
            "windows": [],
        }
    if tach is not None:
        tach = _single_rotor_identity(tach, rec)
    return rec, tach, wind_res, measure, flags


# ─── build ─────────────────────────────────────────────────────────────────


def _build_worker(unit: Unit) -> dict[str, Any]:
    from data_processing.recording_annotations.index import IndexWriter, SampleLocation

    p = unit.params
    run = Path(p["run"])
    freeze = _load_freeze(run)
    name = p["dataset"]
    d = freeze["datasets"][name]
    manifest = _manifest(run, name, d["version"])
    candidate_snapshot = p.get("candidate_snapshot")
    measured, measure_codes = _measure_rows(run, name, candidate_snapshot)
    units = list(_iter_scan(run, name, freeze))
    provenance = {
        "freeze": freeze["source_commit"],
        "extract": sorted({u.get("extract_code", "unstamped") for u in units}),
        "classify": sorted({u.get("code", "unstamped") for u in units}),
        "candidate_snapshot": candidate_snapshot,
        "measure": sorted(measure_codes),
        "build": _code_stamp(),
        "runner": RUNNER_RECIPE,
    }
    del units
    planner = _Planner(run, freeze)
    regime_names = _regime_datasets()
    parent_cache = {name: measured}
    sealed_links: dict[tuple[str, str], dict[str, Any]] = {}

    def resolve_parent(
        dataset: str, key: str, visited: frozenset[tuple[str, str]]
    ) -> dict[str, Any]:
        """Seal and compose each validated link on its own parent/child clock."""
        address = (dataset, key)
        if address in visited:
            raise CampaignError(f"{dataset}/{key}: cyclic observation binding")
        if address in sealed_links:
            return sealed_links[address]
        entry = planner.entry(dataset, key)
        if entry is None:
            raise CampaignError(f"{dataset}/{key}: linked parent has no classification")
        if dataset not in parent_cache:
            parent_cache[dataset] = _measure_rows(run, dataset, candidate_snapshot)[0]
        row = _published(parent_cache[dataset].get(key, {}), entry[0])
        parent_plan = planner(*entry)
        ancestor = parent_plan["tach"].get("link") or parent_plan["wind"].get("link")
        if ancestor is not None:
            upstream = resolve_parent(ancestor["dataset"], ancestor["key"], visited | {address})
            row.update(_project(ancestor, upstream, parent_plan, *entry))
        sealed_links[address] = row
        return row

    out_path = _run_paths(run)["staging"] / f"{name}@{d['version']}.sqlite"
    expected: list[str] = []
    rules_counts: dict[str, Counter] = defaultdict(Counter)
    final_counts: dict[str, Counter] = defaultdict(Counter)
    by_group: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    rules_by_group: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    tach_basis: Counter = Counter()
    tach_acceptance: Counter = Counter()
    tach_status: Counter = Counter()
    wind_src: Counter = Counter()
    refusals: Counter = Counter()
    flags: Counter = Counter()
    flag_examples: dict[str, list[str]] = defaultdict(list)
    scan_errors: list[dict[str, Any]] = []
    from data_processing.recording_annotations.schema import FACETS

    t0 = time.time()
    n_pack = 0
    with IndexWriter(
        out_path,
        dataset=name,
        version=d["version"],
        layout=d["layout"],
        parent_shards=manifest.shards,
        parent_meta=d["meta"],
        source_commit=freeze["source_commit"],
        provenance=provenance,
    ) as w:
        for su in _iter_scan(run, name, freeze):
            n_pack += su["n_pack_keys"]
            if su["n_pack_keys"] != len(su["bookkeeping"]) + len(su["rows"]) + len(su["errors"]):
                raise CampaignError(f"{name}#{su['shard']}: pack keys not all accounted for")
            scan_errors += su["errors"]
            final_memo: dict[tuple[int, int], tuple] = {}
            for key, ri, ii, loc in su["rows"]:
                record, info = su["records"][ri], su["infos"][ii]
                plan = planner(record, info)
                regime_parent = name in regime_names
                mrow = measured.get(key)
                results = mrow or {}
                missing = [
                    f"wind[{role}]"
                    for role, task in plan["wind"].get("roles", {}).items()
                    if task.get("run") and role not in (results.get("wind_roles") or {})
                ]
                if plan["tach"].get("run") and "tach_candidates" not in results:
                    missing.append("tach")
                if regime_parent and "native_regime" not in results:
                    missing.append("regime")
                if missing:
                    raise CampaignError(
                        f"{name}/{key}: missing planned measurements {missing}; "
                        f"{results.get('error') or 'run the missing tasks'}"
                    )
                link = plan["tach"].get("link") or plan["wind"].get("link")
                if link is not None:
                    prow = resolve_parent(link["dataset"], link["key"], frozenset({(name, key)}))
                    mrow = {**(mrow or {}), **_project(link, prow, plan, record, info)}
                if mrow is not None:
                    mrow = _published(mrow, record)
                if mrow is None:  # metadata-only rows share one final record object per profile
                    memo_key = (ri, ii)
                    if memo_key not in final_memo:
                        final_memo[memo_key] = finalize(record, info, plan, None)
                    rec, tach, wres, meas, fl = final_memo[memo_key]
                else:
                    rec, tach, wres, meas, fl = finalize(record, info, plan, mrow)
                group = str(rec.get("provenance_kind") or "unstated")
                for f in FACETS:
                    rules_counts[f][record[f]] += 1
                    final_counts[f][rec[f]] += 1
                    by_group[group][f][rec[f]] += 1
                    rules_by_group[group][f][record[f]] += 1
                if tach:
                    for win in tach.get("windows") or []:
                        if win.get("accepted") and win.get("rates_rev_s"):
                            basis = (win.get("quality") or {}).get("acceptance_basis")
                            if not basis or tach.get("publication_state") != "corroborated":
                                raise CampaignError(
                                    f"{name}/{key}: accepted window without corroboration"
                                )
                            tach_basis[str(basis)] += 1
                            tach_acceptance[f"windows:{len(win['rates_rev_s'])}_resolved"] += 1
                    accepted = [
                        w
                        for w in tach.get("windows") or []
                        if w.get("accepted") and w.get("rates_rev_s")
                    ]
                    if accepted:
                        origin = "linked" if tach.get("projected_from") else "direct"
                        tach_acceptance[f"{origin}:recordings_with_rates"] += 1
                        if any(len(win["rates_rev_s"]) == 4 for win in accepted):
                            tach_acceptance[
                                f"{origin}:recordings_with_four_simultaneous_rates"
                            ] += 1
                        for rotor_id in tach.get("rotor_ids") or []:
                            tach_acceptance[f"{origin}:recordings_with_named_rotor:{rotor_id}"] += 1
                for f in fl:
                    flags[f.split(":")[0]] += 1
                    if len(flag_examples[f.split(":")[0]]) < 10:
                        flag_examples[f.split(":")[0]].append(f"{key} {f}")
                origin = (
                    "linked"
                    if tach and tach.get("projected_from")
                    else "measured"
                    if plan["tach"].get("run")
                    else "not_run"
                )
                tach_status[f"{origin}:{tach.get('status') if tach else 'none'}"] += 1
                if not plan["tach"].get("run"):
                    refusals[f"tach: {_reason_class(plan['tach'].get('reason'))}"] += 1
                for role, task in plan["wind"].get("roles", {}).items():
                    if not task.get("run"):
                        refusals[f"wind[{role}]: {_reason_class(task.get('reason'))}"] += 1
                if not plan["wind"].get("roles"):
                    refusals[f"wind: {_reason_class(plan['wind'].get('reason'))}"] += 1
                comb = (wres or {}).get("combined") or {}
                wind_src[
                    f"{'pcm' if (mrow or {}).get('wind_roles') else 'no_pcm'}->{comb.get('wind', rec['wind'])}"
                ] += 1
                t_start = (meas.get("tach") or {}).get("audio_t_start_s") or 0.0
                location = SampleLocation(loc[0], loc[1], loc[2], tuple(tuple(f) for f in loc[3]))
                w.add(
                    key,
                    rec,
                    info=info,
                    location=location,
                    tach=tach,
                    audio_t_start_s=float(t_start),
                    wind=wres,
                    measure=meas,
                )
                expected.append(key)
        if n_pack != manifest.num_samples:
            raise CampaignError(
                f"{name}: scanned {n_pack} pack keys, manifest declares {manifest.num_samples}"
            )
        if scan_errors:
            raise CampaignError(
                f"{name}: {len(scan_errors)} keys failed the rules pass, e.g. {scan_errors[:3]}"
            )
        w.close(expected_keys=expected)
    return {
        "dataset": name,
        "version": d["version"],
        "rows": len(expected),
        "path": str(out_path),
        "db_bytes": out_path.stat().st_size,
        "db_sha256": _sha256_file(out_path),
        "provenance": provenance,
        "rules_pass": {f: dict(c) for f, c in rules_counts.items()},
        "post_waveform": {f: dict(c) for f, c in final_counts.items()},
        "rules_pass_by_provenance": {
            g: {f: dict(c) for f, c in fs.items()} for g, fs in rules_by_group.items()
        },
        "tach_accepted_windows_by_basis": dict(tach_basis),
        "tach_acceptance": dict(tach_acceptance),
        "post_waveform_by_provenance": {
            g: {f: dict(c) for f, c in fs.items()} for g, fs in by_group.items()
        },
        "tach_status": dict(tach_status),
        "wind_source": dict(wind_src),
        "refusals": dict(refusals),
        "qa_flags": dict(flags),
        "qa_flag_examples": dict(flag_examples),
        "wall_s": time.time() - t0,
        "max_rss_mb": _rss_mb(),
    }


def cmd_build(args: argparse.Namespace) -> None:
    run = Path(args.run)
    freeze = _load_freeze(run)
    units = [
        Unit(
            f"build__{_safe(n)}",
            {
                "run": str(run),
                "dataset": n,
                "candidate_snapshot": args.candidate_snapshot,
            },
        )
        for n in _selected(freeze, args.datasets)
    ]
    res = gridrun_from_args(
        args,
        units,
        _build_worker,
        run / "build",
        summarize=lambda rows: {r["dataset"]: r["rows"] for r in rows},
    )
    raise SystemExit(1 if res.n_failed else 0)


# ─── report ────────────────────────────────────────────────────────────────


def _pct_known(counts: Mapping[str, int]) -> float:
    total = sum(counts.values())
    known = sum(n for v, n in counts.items() if v not in ("unknown", "not_applicable"))
    return round(100.0 * known / total, 1) if total else 0.0


def cmd_report(args: argparse.Namespace) -> None:
    run = Path(args.run)
    freeze = _load_freeze(run)
    paths = _run_paths(run)
    builds = {
        r["dataset"]: r
        for r in (json.loads(p.read_text()) for p in sorted((run / "build" / "raw").glob("*.json")))
    }
    scans = [json.loads(p.read_text()) for p in sorted((paths["extract"] / "raw").glob("*.json"))]
    meas = [json.loads(p.read_text()) for p in sorted((paths["measure"] / "raw").glob("*.json"))]
    errs = sorted(
        str(p.name)
        for stage in ("extract", "classify", "measure")
        for p in (paths[stage] / "raw").glob("*.err")
    )
    unhandled: dict[str, Counter] = defaultdict(Counter)
    for s in scans:
        unhandled[s["dataset"]].update(
            {k: v for k, v in s["media"].items() if k.startswith(("UNHANDLED", "UNREADABLE"))}
        )
    from data_processing.recording_annotations.schema import FACETS

    groups: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    datasets = {}
    for name in sorted(freeze["datasets"]):
        b = builds.get(name)
        datasets[name] = {
            "version": freeze["datasets"][name]["version"],
            "resolved_from": freeze["datasets"][name]["resolved_from"],
            "samples_manifest": freeze["datasets"][name]["samples"],
            "rows_indexed": b["rows"] if b else None,
            "pct_known_rules": {f: _pct_known(b["rules_pass"][f]) for f in FACETS} if b else None,
            "pct_known_final": {f: _pct_known(b["post_waveform"][f]) for f in FACETS}
            if b
            else None,
            "tach_status": b["tach_status"] if b else None,
            "tach_acceptance": b.get("tach_acceptance") if b else None,
            "tach_accepted_windows_by_basis": b.get("tach_accepted_windows_by_basis")
            if b
            else None,
            "refusals": b["refusals"] if b else None,
            "qa_flags": b["qa_flags"] if b else None,
            "db": {"bytes": b["db_bytes"], "sha256": b["db_sha256"]} if b else None,
        }
        if b:
            for f in FACETS:
                groups["all"][f].update(b["post_waveform"][f])
                groups["all_rules"][f].update(b["rules_pass"][f])
            for field, phase in (
                ("rules_pass_by_provenance", "rules"),
                ("post_waveform_by_provenance", "final"),
            ):
                for group, facets in b.get(field, {}).items():
                    for f, counts in facets.items():
                        groups[f"{group}:{phase}"][f].update(counts)
    tp = {
        "extract_units": len(scans),
        "extract_wall_s": round(sum(s["wall_s"] for s in scans), 1),
        "extract_remote_bytes": sum(s["io"]["bytes"] for s in scans),
        "extract_remote_requests": sum(s["io"]["requests"] for s in scans),
        "extract_data_bytes": sum(s["data"]["bytes"] for s in scans),
        "extract_max_rss_mb": max((s["max_rss_mb"] for s in scans), default=0),
        "measure_units": len(meas),
        "measure_wall_s": round(sum(m["wall_s"] for m in meas), 1),
        "measure_remote_bytes": sum(m["io"]["bytes"] for m in meas),
        "measure_max_rss_mb": max((m["max_rss_mb"] for m in meas), default=0),
        "audio_seconds_tach": round(
            sum(
                (r.get("tach_measure") or {}).get("seconds_analysed", 0.0)
                for m in meas
                for r in m["rows"]
            ),
            1,
        ),
        "audio_seconds_wind": round(
            sum(
                (res.get("measure") or {}).get("seconds_analysed", 0.0)
                for m in meas
                for r in m["rows"]
                for res in (r.get("wind_roles") or {}).values()
            ),
            1,
        ),
    }
    errors = [
        {"dataset": m["dataset"], "key": r["key"], "error": r["error"]}
        for m in meas
        for r in m["rows"]
        if "error" in r
    ]
    report = {
        "runner_recipe": RUNNER_RECIPE,
        "source_commit": freeze["source_commit"],
        "frozen_at": freeze["frozen_at"],
        "missing_parents": freeze["missing_parents"],
        "absent_shards": freeze["absent_shards"],
        "bad_local_copies": freeze["bad_local_copies"],
        "unit_errors": errs,
        "measure_errors": errors[:200],
        "n_measure_errors": len(errors),
        "unhandled_media": {k: dict(v) for k, v in sorted(unhandled.items()) if v},
        "datasets": datasets,
        "facets_all": {g: {f: dict(c) for f, c in fs.items()} for g, fs in groups.items()},
        "pct_known_by_group": {
            g: {f: _pct_known(c) for f, c in fs.items()} for g, fs in groups.items()
        },
        "throughput": tp,
    }
    _jdump(paths["report"] / "report.json", report)
    lines = [
        f"# drone-recording-annotations campaign report ({freeze['frozen_at']})",
        "",
        f"source `{freeze['source_commit']}`, runner `{RUNNER_RECIPE}`",
        "",
        "| dataset | version | manifest | indexed | tach statuses | % known (rules → final) |",
        "|---|---|---:|---:|---|---|",
    ]
    for name, d in datasets.items():
        pk = (
            ", ".join(f"{f}:{d['pct_known_rules'][f]}→{d['pct_known_final'][f]}" for f in FACETS)
            if d["pct_known_final"]
            else "NOT BUILT"
        )
        lines.append(
            f"| {name} | {d['version'][:12]} ({d['resolved_from']}) | {d['samples_manifest']} | "
            f"{d['rows_indexed']} | {d['tach_status']} | {pk} |"
        )
    lines += ["", "## Throughput", "```", json.dumps(tp, indent=1), "```"]
    (paths["report"] / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ─── plan (coverage QA + partitions, metadata only) ────────────────────────


def cmd_plan(args: argparse.Namespace) -> None:
    """Per parent: what the waveform pass will do for every row, from classify units only.

    ``constant_on`` = rows the rules call constant with rotors on: each must be direct,
    linked, too_short or carry an explicit unscoreable reason. ``wind_only`` lists parents
    with wind work and no direct tachometry (safe to measure before the tach gate)."""
    run = Path(args.run)
    freeze = _load_freeze(run)
    planner = _Planner(run, freeze)
    per: dict[str, Any] = {}
    for name in _selected(freeze, args.datasets):
        c: Counter = Counter()
        for su in _iter_scan(run, name, freeze):
            for _key, ri, ii, _loc in su["rows"]:
                rec, info = su["records"][ri], su["infos"][ii]
                plan = planner(rec, info)
                t = plan["tach"]
                kind = (
                    "direct"
                    if t.get("run")
                    else "linked"
                    if t.get("link")
                    else t.get("status") or "not_candidate"
                )
                c[f"tach:{kind}"] += 1
                if rec.get("speed_regime") == "constant" and rec.get("rotor_activity") != "off":
                    c["constant_on"] += 1
                    c[f"constant_on:{kind}"] += 1
                    if kind == "not_applicable":
                        c[f"constant_on:not_applicable:{_reason_class(t.get('reason'))}"] += 1
                if t.get("run"):
                    c[f"tach_role:{t['role']} ({t.get('role_from')})"] += 1
                for role, task in plan["wind"].get("roles", {}).items():
                    c[
                        f"wind[{role}]:{'run' if task.get('run') else 'too_short' if task.get('too_short') else 'skip'}"
                    ] += 1
                if plan["wind"].get("link"):
                    c["wind:linked"] += 1
                if plan.get("link_missing"):
                    c["link_missing"] += 1
        per[name] = dict(sorted(c.items()))
    wind_only = sorted(
        n
        for n, c in per.items()
        if not c.get("tach:direct")
        and not c.get("tach:linked")
        and any(k.startswith("wind[") and k.endswith(":run") for k in c)
    )
    out = {
        "runner_recipe": RUNNER_RECIPE,
        "code": _code_stamp(),
        "datasets": per,
        "wind_only": wind_only,
        "constant_on_total": sum(c.get("constant_on", 0) for c in per.values()),
    }
    _jdump(run / "plan" / f"plan.{_safe(args.label)}.json", out)
    print(
        json.dumps(
            {"constant_on_total": out["constant_on_total"], "wind_only": wind_only}, indent=1
        )
    )
    for name, c in per.items():
        keep = {
            k: v
            for k, v in c.items()
            if k.startswith(("constant_on", "tach:direct", "tach:linked"))
        }
        if keep:
            print(name, json.dumps(keep))


# ─── show (QA preview) ─────────────────────────────────────────────────────


def cmd_show(args: argparse.Namespace) -> None:
    run = Path(args.run)
    freeze = _load_freeze(run)
    planner = _Planner(run, freeze)
    name = args.dataset
    measured = _measure_rows(run, name)[0] if (_run_paths(run)["measure"] / "raw").exists() else {}
    n = 0
    seen_profiles: set[str] = set()
    for su in _iter_scan(run, name, freeze):
        for key, ri, ii, _loc in su["rows"]:
            if args.keys and key not in args.keys:
                continue
            rec = su["records"][ri]
            if args.distinct and _canon(rec) in seen_profiles:
                continue
            seen_profiles.add(_canon(rec))
            info = su["infos"][ii]
            plan = planner(rec, info)
            final = (
                finalize(rec, info, plan, measured.get(key))
                if (key in measured or not (plan["tach"].get("run") or plan["wind"].get("run")))
                else None
            )
            print(
                json.dumps(
                    {
                        "key": key,
                        "record": rec,
                        "info": {
                            k: info[k]
                            for k in ("duration_s", "sample_rate", "audio_entries", "layout")
                        },
                        "plan": plan,
                        "final": None
                        if final is None
                        else {
                            "record": final[0],
                            "tach": final[1],
                            "wind": final[2],
                            "qa": final[4],
                        },
                    },
                    indent=1,
                    sort_keys=True,
                    default=_jsonable,
                )[: args.max_chars]
            )
            n += 1
            if n >= args.limit:
                return


# ─── publish (authorized only) ─────────────────────────────────────────────


def cmd_publish(args: argparse.Namespace) -> None:
    import tempfile

    from dload import Repository
    from dload.cache import ShardCache

    from data_processing import recording_annotations as ra
    from data_processing.streams import open_repository

    if args.confirm != RELEASE_NAME:
        raise SystemExit(
            f"refusing: pass --confirm {RELEASE_NAME} once Main authorizes the release"
        )
    run = Path(args.run)
    staging = _run_paths(run)["staging"]
    freeze = _load_freeze(run)
    expected = {f"{name}@{d['version']}" for name, d in freeze["datasets"].items()}
    want = {p.stem: _sha256_file(p) for p in sorted(staging.glob("*@*.sqlite"))}
    if set(want) != expected or freeze["missing_parents"] or freeze["absent_shards"]:
        raise CampaignError("refusing to publish an incomplete companion release")
    for name in freeze["datasets"]:
        receipt = json.loads(unit_path(run / "build", f"build__{_safe(name)}").read_text())
        key = f"{name}@{freeze['datasets'][name]['version']}"
        if receipt["db_sha256"] != want[key]:
            raise CampaignError(f"staged DB {key} differs from its build receipt")
    fingerprint = hashlib.sha256(
        _canon({"runner": RUNNER_RECIPE, "parents": want}).encode()
    ).hexdigest()
    repo = open_repository()
    before = (
        [m.version for m in repo.versions(RELEASE_NAME)]
        if RELEASE_NAME in repo.list_datasets()
        else []
    )
    manifest = ra.publish(
        staging,
        repo=repo,
        recipe=f"python scripts/annotate_drone_recordings.py publish --run {run} --confirm {RELEASE_NAME}",
        provenance={
            "runner": RUNNER_RECIPE,
            "fingerprint": fingerprint,
            "freeze_source": freeze["source_commit"],
            "frozen_at": freeze["frozen_at"],
            "publish_code": _code_stamp(),
        },
    )
    # Fetch every new companion DB through a fresh cache, not through the upload cache.
    got = {}
    with tempfile.TemporaryDirectory(prefix="conditions-readback-") as cache:
        cold_repo = Repository(repo.remote, ShardCache(Path(cache), None))
        rel = ra.open_release(version=manifest.version, repo=cold_repo)
        for parent in rel.parents():
            idx = rel.index(parent.dataset, version=parent.version)
            got[f"{parent.dataset}@{parent.version}"] = {
                "n_rows": idx.n_rows,
                "db_bytes": parent.db_bytes,
            }
        rel.close()
    meta_parents = {f"{p['dataset']}@{p['version']}": p for p in manifest.meta["parents"]}
    mismatches = [k for k, sha in want.items() if meta_parents.get(k, {}).get("db_sha256") != sha]
    receipt = {
        "release": RELEASE_NAME,
        "fingerprint": fingerprint,
        "version": manifest.version,
        "previous_versions": before,
        "staged": want,
        "readback": got,
        "sha_mismatches": mismatches,
        "published_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _jdump(_run_paths(run)["publish"] / "receipt.json", receipt)
    print(json.dumps(receipt, indent=1))
    if mismatches:
        raise SystemExit(f"readback sha256 mismatch: {mismatches}")


# ─── CLI ───────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser, *, jobs: bool = True) -> None:
        p.add_argument("--run", required=True, help="campaign directory (GPFS on uni-cpu)")
        p.add_argument("--datasets", nargs="*", help="restrict to these frozen datasets")
        if jobs:
            add_gridrun_args(p, jobs=4)

    p = sub.add_parser("freeze")
    p.add_argument("--run", required=True)
    p.add_argument("--datasets", nargs="*", help="override rules.DATASETS (testing only)")
    p.add_argument(
        "--local-root",
        action="append",
        default=[],
        help="dir holding shards/<xx>/<digest> copies (stage-remote, dload cache)",
    )
    p.add_argument("--source-commit", required=True, help="code snapshot sha256 + git HEAD")
    p.add_argument("--jobs", type=int, default=8, help="parallel sha256 of local copies")
    p.set_defaults(fn=cmd_freeze)

    p = sub.add_parser("extract", help="ranged metadata reads per shard (no PCM, no rules)")
    common(p)
    p.add_argument("--max-shards", type=int, help="QA preview: first K shards per dataset")
    p.set_defaults(fn=cmd_extract)

    p = sub.add_parser(
        "classify", help="rules pass over extracted metadata (rerun with --no-resume)"
    )
    common(p)
    p.add_argument("--max-shards", type=int, help="QA preview: first K shards per dataset")
    p.set_defaults(fn=cmd_classify)

    p = sub.add_parser("measure")
    common(p)
    p.add_argument("--keys", nargs="*", help="restrict to these sample keys")
    p.add_argument(
        "--view-s", type=float, default=120.0, help="max tachometry view (s), memory-capped"
    )
    p.add_argument(
        "--shard-of", nargs=2, type=int, metavar=("K", "N"), help="this job runs units i%%N==K"
    )
    p.add_argument(
        "--partition",
        default="all",
        help="label of this job's slice; its summary goes to measure/summary.<label>.json",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        choices=list(ALL_TASKS),
        default=list(ALL_TASKS),
        help="waveform-stage tasks of this partition (e.g. wind alone before the tach gate)",
    )
    p.set_defaults(fn=cmd_measure)

    p = sub.add_parser("build")
    common(p)
    p.add_argument(
        "--candidate-snapshot",
        help="only use private candidates from this validated code archive SHA256",
    )
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("report")
    p.add_argument("--run", required=True)
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser(
        "plan", help="metadata-only coverage QA of the waveform plan + wind-only partition"
    )
    p.add_argument("--run", required=True)
    p.add_argument("--datasets", nargs="*")
    p.add_argument("--label", default="all")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("show")
    p.add_argument("--run", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--keys", nargs="*")
    p.add_argument("--distinct", action="store_true", help="one row per distinct rules record")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--max-chars", type=int, default=6000)
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("publish")
    p.add_argument("--run", required=True)
    p.add_argument("--confirm", default="", help=f"must equal {RELEASE_NAME}")
    p.set_defaults(fn=cmd_publish)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    main()
