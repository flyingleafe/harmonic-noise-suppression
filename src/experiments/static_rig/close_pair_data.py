"""Frozen-source staging for the close-rotor pilot; never writes parent objects."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from dload.remote import LocalRemote

from data_processing.recording_annotations.scan import (
    RemoteRanges,
    ShardBytes,
    read_view,
    scan_sample,
)
from data_processing.recording_annotations.storage import open_release
from data_processing.streams import _decode_index, open_repository

RELEASE = "497ea2c5828a6cac19dc23ceabe3e1c4340725265ad99c06bc0218108797cd10"
TABLE_SHA256 = "303927e49df22b93a66124e92bad6d3bcc7765a890edf35b844e2ec078d51cc8"
DAS = "DroneAudioSet"
DREGON = "DREGON-frames"
AGH = "SPCUP19-frames"
LEUVEN = "KULeuven-UAV-egonoise-frames"
AUDIO_SECONDS = 8.0


def panel(table: Path) -> list[dict]:
    """The OS-independent split groups cover whole configurations / source takes."""
    if hashlib.sha256(table.read_bytes()).hexdigest() != TABLE_SHA256:
        raise ValueError("unexpected frozen 608-recording table digest")
    rows = []
    with table.open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            ds, recording = row["dataset"], row["recording"]
            keys = json.loads(row["source_key_status_json"])
            expected = int(row["expected_active_rotors"])
            if ds == DAS and recording.startswith("drone-only-recordings__"):
                arm = "four"
                group = recording.rsplit("__", 1)[0]
                split = (
                    "development"
                    if ("drone1-only" in group) == ("throttle-low" in group)
                    else "test"
                )
            elif ds in (LEUVEN, "AVQ") or (ds == DREGON and recording == "motor_allMotors_70"):
                arm, group, split = "four", recording, "test"
            elif expected == 1 and (
                ds == DREGON or (ds == AGH and "__single_rotors__" in recording)
            ):
                arm, group = "single", recording
                # All 50/70/90% DREGON takes and alternating AGH takes are held out.
                is_test = (
                    recording.endswith(("_50", "_70", "_90"))
                    if ds == DREGON
                    else int(recording.rsplit("__", 1)[1]) % 2 == 1
                )
                split = "test" if is_test else "development"
            else:
                continue
            for key, status in keys.items():
                rows.append(
                    dict(
                        dataset=ds,
                        group=group,
                        recording=recording,
                        key=key,
                        expected=expected,
                        arm=arm,
                        split=split,
                        release_status=status,
                        recording_release_max=int(row["max_simultaneous_annotated"]),
                    )
                )
    for key in ("S1_seq4", "S2_seq3", "S2_seq4"):
        rows.append(
            dict(
                dataset="AVQ",
                group=key,
                recording=key,
                key=key,
                expected=0,
                arm="speech",
                split="test",
            )
        )
    for key in ("FLY124", "FLY125"):
        rows.append(
            dict(
                dataset="michaels-frames",
                group=key,
                recording=key,
                key=key,
                expected=4,
                arm="esc",
                split="test",
            )
        )
    groups = {(r["dataset"], r["recording"]) for r in rows if r["arm"] == "four"}
    if len(groups) != 66 or sum(r["arm"] == "single" for r in rows) != 28:
        raise ValueError(f"wrong frozen panel: {len(groups)} four-rotor groups")
    return sorted(rows, key=lambda r: (r["dataset"], r["recording"], r["key"]))


@contextmanager
def _waveform(repo, ranges, index, shard_info, location, key: str, role: str, *, with_esc: bool):
    digest, size, _ = shard_info
    local = (
        repo.remote.root / "shards" / digest[:2] / digest
        if isinstance(repo.remote, LocalRemote)
        else None
    )
    fields = {f: (location.offset + off, n) for f, off, n in location.fields}
    if "_frame" in fields:
        # Scan only the roles actually used. Irregular ancillary sensor
        # series may have large stamp arrays unrelated to this pilot.
        keep = {"_frame", role}
        if with_esc:
            keep.update(("motor_speed", "rps"))
            keep.update(f for f in fields if f.startswith(("motor_speed#", "rps#")))
        fields = {f: v for f, v in fields.items() if f in keep}
    shard = ShardBytes(digest, size, path=local, remote=ranges)
    try:
        sample = scan_sample(
            shard,
            location.shard,
            key,
            fields,
            dataset=index.dataset,
            version=index.version,
            layout=index.layout,
            dataset_meta=index.meta().get("parent_meta"),
        )
        refs = sample.audio_refs()
        ref = refs.get(role)
        if ref is None and len(refs) == 1:
            (ref,) = refs.values()
        if (
            ref is None
            or ref.t_start_s is None
            or ref.rate_hz is None
            or ref.n_time is None
            or ref.container not in ("npy", "soundfile")
        ):
            raise ValueError(f"{index.dataset}/{key}: missing bounded native audio role {role}")
        yield shard, ref, fields
    finally:
        shard.close()


def _telemetry(shard, fields) -> dict[str, np.ndarray]:
    """MD2 ESC speed on native aligned times; never use the refined labels."""
    entries = json.loads(shard.read(*fields["_frame"]))["frame"]["entries"]
    out = {}
    for key in ("motor_speed", "rps"):
        dim, desc = next(
            (d, i) for d, i in entries[key]["indexes"].items() if i["type"] == "stamps"
        )
        stamp_field = f"{key}#{dim}.stamps"
        index = _decode_index(desc, key, dim, {stamp_field: shard.read(*fields[stamp_field])})
        # The stored ticks are relative to the stamp index's start. The
        # public timestamps property places them on the *absolute* Frame clock.
        out[f"{key}_time"] = np.asarray(index.timestamps, dtype=np.float64)
        out[key] = np.load(io.BytesIO(shard.read(*fields[key])), allow_pickle=False)
    return out


def _save_esc(path: Path, telemetry: dict[str, np.ndarray]) -> None:
    np.savez(
        path,
        allow_pickle=False,
        motor_speed_time=telemetry["motor_speed_time"],
        motor_speed=telemetry["motor_speed"],
        rps_time=telemetry["rps_time"],
        rps=telemetry["rps"],
    )


def _bounds(center: float, start: float, stop: float) -> tuple[float, float]:
    length = min(AUDIO_SECONDS, stop - start)
    # Make the short frozen interval coincide with the untouched suffix
    # where possible; earlier audio is strictly analysis context.
    lo = float(np.clip(center - 0.9 * length, start, stop - length))
    return lo, lo + length


def selected(record: dict, row: dict, duration: float) -> tuple[list[dict], dict]:
    """Freeze sampling intervals; adjacent active windows are analysis context."""
    tach = record.get("tach") or {}
    attempted = [
        w
        for w in tach.get("windows", ())
        if w.get("start_s") is not None and w.get("end_s") is not None
    ]
    windows = [w for w in attempted if w["end_s"] - w["start_s"] >= 1.5]
    accepted = [w for w in windows if w["accepted"] and w["rates_rev_s"]]
    regime = (tach.get("summary") or {}).get("regime_evidence") or {}
    coverage = {
        key: regime.get(key)
        for key in ("analyzed_s", "accepted_s", "full_rig_accepted_s", "quiet_s", "transition_s")
    }
    active = coverage["analyzed_s"]
    if active is not None:
        active = max(0.0, active - (coverage["quiet_s"] or 0.0) - (coverage["transition_s"] or 0.0))
    coverage["active_s"] = active
    choices = []
    if row["arm"] == "esc":
        # Fixed audio coordinates, not selected using the telemetry series.
        choices = [
            (name, t, None)
            for name, t in (("early", 25), ("mid", 45), ("late", 65))
            if 4 <= t <= duration - 4
        ]
    elif row["arm"] == "speech":
        choices = [(f"off_{i}", duration * q, None) for i, q in enumerate((0.2, 0.5, 0.8))]
    elif row["arm"] == "single":
        w = min(
            accepted, key=lambda w: (abs(w["center_s"] - duration / 2), w["center_s"]), default=None
        )
        choices = [("single", duration / 2 if w is None else w["center_s"], w)]
    else:
        partial = [w for w in accepted if len(w["rates_rev_s"]) < 4]
        full = [w for w in accepted if len(w["rates_rev_s"]) == 4]
        for name, pool in (("dispute", partial), ("full_anchor", full)):
            if not pool:
                continue
            w = min(
                pool,
                key=lambda w: (
                    -len(w["rates_rev_s"]),
                    abs(w["center_s"] - duration / 2),
                    w["center_s"],
                ),
            )
            choices.append((name, w["center_s"], w))
        if not choices:
            opaque = [
                w
                for w in windows
                if not any(term in str(w.get("reason")) for term in ("silent", "off", "invalid"))
            ]
            w = min(opaque, key=lambda w: abs(w["center_s"] - duration / 2), default=None)
            choices = [("unresolved", duration / 2 if w is None else w["center_s"], w)]
    exclusions = [
        w
        for w in attempted
        if not w["accepted"]
        and any(
            term in str(w.get("reason")) for term in ("silent", "off", "invalid_pcm", "dropout")
        )
    ]
    out = []
    for name, center, w in choices:
        if any(bad["start_s"] <= center < bad["end_s"] for bad in exclusions):
            continue
        left = max((bad["end_s"] for bad in exclusions if bad["end_s"] <= center), default=0.0)
        right = min(
            (bad["start_s"] for bad in exclusions if bad["start_s"] >= center), default=duration
        )
        if right <= left:
            continue
        lo, hi = _bounds(center, max(0.0, left), min(duration, right))
        local = [
            {
                "start_s": v["start_s"],
                "end_s": v["end_s"],
                "count": len(v["rates_rev_s"]) if v["accepted"] else 0,
            }
            for v in attempted
            if v["start_s"] < hi and v["end_s"] > lo
        ]
        out.append(
            {
                "kind": name,
                "requested": (lo, hi),
                "release_window": w,
                "release_local_counts": local,
            }
        )
    return out, coverage


def stage(table: Path, output: Path, part: int, parts: int) -> None:
    if not 0 <= part < parts:
        raise ValueError("invalid shard")
    repo = open_repository()
    ranges = None if isinstance(repo.remote, LocalRemote) else RemoteRanges.from_repository(repo)
    release = open_release(version=RELEASE, repo=repo)
    outputs = output / f"stage-{part:02d}"
    outputs.mkdir(parents=True, exist_ok=True)
    all_rows = panel(table)
    indexes, shards, rows = {}, {}, []
    checked = set()
    for row in all_rows:
        # Give one process all views of a DAS grouped segment.
        tag = hashlib.sha256((row["dataset"] + ":" + row["recording"]).encode()).digest()
        if int.from_bytes(tag[:4], "big") % parts != part:
            continue
        ds, key = row["dataset"], row["key"]
        if ds not in indexes:
            (parent,) = (p for p in release.parents() if p.dataset == ds)
            indexes[ds] = release.index(ds, parent.version)
            shards[ds] = indexes[ds].parent_shards()
        index = indexes[ds]
        loc, info = index.locate(key)
        record = index.record(key)
        if loc is None or not info["has_audio"]:
            raise ValueError(f"{ds}/{key}: frozen parent has no waveform")
        if (version := index.version) not in checked:
            assert repo.manifest(ds, version).version == version
            checked.add(version)
        tach = record.get("tach") or {}
        if "release_status" in row and tach.get("status") != row["release_status"]:
            raise ValueError(f"{ds}/{key}: tabulated and stored status differ")
        role = str(tach.get("role") or "audio")
        esc = row["arm"] == "esc"
        first_bytes = ranges.n_bytes if ranges else 0
        first_requests = ranges.n_requests if ranges else 0
        with _waveform(
            repo, ranges, index, shards[ds][loc.shard], loc, key, role, with_esc=esc
        ) as (shard, ref, fields):
            fs, t0, count = ref.rate_hz, ref.t_start_s, ref.n_time
            assert fs is not None and t0 is not None and count is not None
            if (
                tach.get("audio_t_start_s") is not None
                and abs(tach["audio_t_start_s"] - t0) > 1 / fs
            ):
                raise ValueError(f"{ds}/{key}: annotation/waveform clock mismatch")
            requested, coverage = selected(record, row, count / fs)
            raw_blades = record.get("blade_count")
            blades = (
                int(raw_blades) if str(raw_blades).isdecimal() and int(raw_blades) > 0 else None
            )
            blade_count_evidence = (
                "release"
                if blades
                else "documented MD2 two blades"
                if esc
                else "configured hypothesis, unverified in release"
            )
            row.update(
                parent_version=version,
                shard_digest=shard.digest,
                role=role,
                sample_rate=fs,
                audio_start_s=t0,
                duration_s=count / fs,
                active_seconds=coverage["active_s"],
                release_coverage=coverage,
                blades=blades or (3 if ds == AGH else 2),
                blade_count_evidence=blade_count_evidence,
                sound_content=record.get("sound_content"),
                windows=[],
            )
            uid = hashlib.sha256((ds + ":" + key).encode()).hexdigest()[:16]
            for win in requested:
                start, stop = win.pop("requested")
                pcm, actual = read_view(shard, ref, t0 + start, t0 + stop)
                lo = round((actual - t0) * fs)
                hi = lo + pcm.shape[-1]
                path = outputs / f"{uid}-{win['kind']}.npy"
                np.save(path, pcm, allow_pickle=False)
                row["windows"].append(
                    {
                        **win,
                        "file": str(path),
                        "samples": [lo, hi],
                        "time_s": [actual, actual + pcm.shape[-1] / fs],
                        "channels": pcm.shape[0],
                        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                    }
                )
            if esc:
                path = outputs / f"{uid}-esc.npz"
                _save_esc(path, _telemetry(shard, fields))
                row.update(esc_file=str(path), esc_stamps_are_absolute=True)
        row.update(
            range_bytes=(ranges.n_bytes - first_bytes if ranges else 0),
            range_requests=(ranges.n_requests - first_requests if ranges else 0),
        )
        rows.append(row)
        print(
            f"STAGED {ds} {key} {len(row['windows'])} windows fs={fs:g} range={row['range_bytes']} bytes",
            flush=True,
        )
    receipt = outputs / "manifest.json"
    receipt.write_text(
        json.dumps(
            dict(release=RELEASE, table_sha256=TABLE_SHA256, part=part, parts=parts, rows=rows),
            allow_nan=False,
        )
    )
    print(
        f"RECEIPT {receipt}: {len(rows)} source keys; {sum(r['range_bytes'] for r in rows)} range bytes",
        flush=True,
    )


def verify_staged(data: Path, output: Path) -> None:
    """Bit-check representative native PCM and recover exact ESC stamp origins.

    Open the same immutable byte spans used for a previously staged panel.
    Never replace the old PCM files or mutate the staging receipt.
    """
    rows = []
    for path in sorted(data.glob("stage-*/manifest.json")):
        receipt = json.loads(path.read_text())
        if receipt["release"] != RELEASE:
            raise ValueError(f"unexpected release in {path}")
        rows.extend(receipt["rows"])
    repo = open_repository()
    ranges = None if isinstance(repo.remote, LocalRemote) else RemoteRanges.from_repository(repo)
    release = open_release(version=RELEASE, repo=repo)
    output.mkdir(parents=True, exist_ok=True)
    seen = set()
    checks, esc_files = [], {}
    for row in sorted(rows, key=lambda r: (r["dataset"], r["key"])):
        ds, key, role = row["dataset"], row["key"], row["role"]
        esc = row["arm"] == "esc"
        if not row["windows"] or (not esc and (ds, role) in seen):
            continue
        idx = release.index(ds, row["parent_version"])
        loc, _ = idx.locate(key)
        if loc is None:
            raise ValueError(f"{ds}/{key}: frozen byte location is unavailable")
        with _waveform(
            repo, ranges, idx, idx.parent_shards()[loc.shard], loc, key, role, with_esc=esc
        ) as (shard, ref, fields):
            if (
                shard.digest != row["shard_digest"]
                or ref.rate_hz != row["sample_rate"]
                or ref.t_start_s != row["audio_start_s"]
            ):
                raise ValueError(f"source identity / native clock mismatch for {ds}/{key}")
            for win in row["windows"] if esc else row["windows"][:1]:
                pcm, actual = read_view(shard, ref, *win["time_s"])
                digest = hashlib.sha256(pcm).hexdigest()
                if digest != win["pcm_sha256"] or actual != win["time_s"][0]:
                    raise ValueError(f"replay mismatch for {ds}/{key}/{win['kind']}")
                checks.append(
                    dict(
                        dataset=ds,
                        key=key,
                        kind=win["kind"],
                        time_s=win["time_s"],
                        pcm_sha256=digest,
                    )
                )
            if esc:
                fixed = _telemetry(shard, fields)
                path = output / f"{key}-esc.npz"
                _save_esc(path, fixed)
                esc_files[key] = str(path)
                with np.load(row["esc_file"], allow_pickle=False) as previous:
                    for telemetry_role in ("motor_speed", "rps"):
                        t = f"{telemetry_role}_time"
                        if not np.array_equal(
                            fixed[telemetry_role], previous[telemetry_role], equal_nan=True
                        ):
                            raise ValueError(f"ESC value mismatch for {key}/{telemetry_role}")
                        print(
                            f"ESC {key}/{telemetry_role} first={fixed[t][0]:.6f} last={fixed[t][-1]:.6f} "
                            f"absolute minus previous={np.median(fixed[t] - previous[t]):.6f} s",
                            flush=True,
                        )
        seen.add((ds, role))
    if len(esc_files) != 2 or not checks:
        raise ValueError("expected representative frozen PCM and both MD2 recordings")
    (output / "replay.json").write_text(
        json.dumps(dict(release=RELEASE, checks=checks, esc_files=esc_files))
    )
    print(f"REPLAY {len(checks)} bit-identical PCM spans: {output / 'replay.json'}", flush=True)
