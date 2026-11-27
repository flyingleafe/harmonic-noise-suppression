#!/usr/bin/env python3
"""Experimental native-clock rotor tracks on the frozen original-recording cohort.

Plan once from the immutable annotation companion; run independently resumable
parts against exact pack ranges; merge only after all 1,728 source rows exist.
Neither the dload parent nor the companion is ever written. All track/range
seconds are relative to the selected native role's explicit audio start.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import resource
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

RELEASE = "497ea2c5828a6cac19dc23ceabe3e1c4340725265ad99c06bc0218108797cd10"
COHORT_SHA256 = "303927e49df22b93a66124e92bad6d3bcc7765a890edf35b844e2ec078d51cc8"
SHAPE = (608, 1728)
FILTER = {"speed_regime": "constant", "has_audio": True, "provenance_kind": "recording"}
LEVELS = {
    "direct": ("measured",),
    "supported": ("measured", "track_supported"),
    "filled": ("measured", "track_supported", "interpolated"),
}
TOKENS = {*LEVELS["filled"], "unresolved", "inactive"}
ANCHOR_RULE = "497ea same-role accepted centres; hold out floor(relative_time_s/12)%5==2; blind if no anchor-bearing held block"
HOLDOUT_TOLERANCE_RPS = 0.65
SCHEMA = "multiscale-corpus/1"
# Documented acquisition facts, not labels from the annotation companion.
FIGURE_BLADES = {
    (
        "SPCUP19-frames",
        "a81ee75821c6900446ca615d5bc927dc2258deba952b38503e323d5f9ce667bf",
        "Maverick__4",
    ): {
        "url": "https://dregon.inria.fr/SPCup2019_egonoise/SPCUP19_Maverick_documentation.pdf",
        "locator": "Acquisition report p. 4, Fig. 2 top photograph; two folding blades per hub",
    },
    (
        "zenodo_drone_noises",
        "a5d0ed89df141780502eef5084bb722ec2d38b4418ff7d0a573dd26f2eea28c6",
        "raw/noises-test-drones/n121",
    ): {
        "url": "https://doi.org/10.1109/JSEN.2018.2825879",
        "locator": "Wang & Cavallaro, Fig. 1(a,c), AS 3DR IRIS schematic and photographed assembly; two blades per hub",
    },
}


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_line(obj):
    return json.dumps(obj, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json_line(obj))
    tmp.replace(path)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with (
        tmp.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz,
        io.TextIOWrapper(gz, encoding="utf-8") as text,
    ):
        for row in rows:
            text.write(json_line(row))
    tmp.replace(path)


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        yield from map(json.loads, handle)


def union_seconds(spans):
    total, end = 0.0, -math.inf
    for a, b in sorted((float(a), float(b)) for a, b in spans):
        if math.isfinite(a) and math.isfinite(b):
            total += max(0.0, b - max(a, end))
            end = max(end, b)
    return total


def baseline_for(tach, status, expected):
    if tach is None or tach["status"] != status or tach["n_expected"] != expected:
        raise ValueError(f"baseline disagrees with frozen cohort: {status}, {expected}: {tach!r}")
    times = tach["times"]
    rates = tach["rates"]
    n = np.isfinite(rates).sum(axis=1) * tach["accepted"]
    return {
        "status": status,
        "role": tach["role"],
        "audio_t_start_s": tach["audio_t_start_s"],
        "assessed_s": union_seconds(times[:, :2]),
        "any_s": union_seconds(times[n > 0, :2]),
        "all_s": union_seconds(times[n == expected, :2]),
        "max_simultaneous": int(n.max(initial=0)),
        "accepted_points": [
            {"time_s": float(t), "rates_rps": [float(v) for v in row[np.isfinite(row)]]}
            for t, row in zip(times[n > 0, 2], rates[n > 0], strict=True)
            if math.isfinite(t)
        ],
    }


def source_channels(record, role):
    per_role = record.get("role_tach_channels") or {}
    if role in per_role:
        return per_role[role]
    if role == (record.get("tach_role") or "audio"):
        return record.get("tach_channels")
    return None


def build_plan(args):
    from data_processing import recording_annotations as ra
    from data_processing.streams import open_repository

    if sha256(args.cohort) != COHORT_SHA256:
        raise ValueError("the original group-to-source TSV does not match its reviewed SHA-256")
    with args.cohort.open(newline="") as handle:
        groups = list(csv.DictReader(handle, delimiter="\t"))
    repo = open_repository()
    rel = ra.open_release(version=RELEASE, repo=repo)
    plans = defaultdict(list)
    for group in groups:
        plans[group["dataset"]].append(group)
    rows, group_rows, parents = [], [], {}
    for dataset in sorted(plans, key=lambda ds: (ds != "AVQ", ds)):
        (parent,) = [p for p in rel.parents() if p.dataset == dataset]
        idx = rel.index(dataset, parent.version)
        manifest = repo.manifest(dataset, parent.version)
        shards = idx.parent_shards()
        if not all(
            (d, sz) == (sh.digest, sh.size)
            for (d, sz, _), sh in zip(shards, manifest.shards, strict=True)
        ):
            raise ValueError(f"packed source manifest mismatch: {dataset}@{parent.version}")
        fields_map = (manifest.meta or {}).get("fields") or {}
        parents[dataset] = {
            "version": parent.version,
            "annotation_db_sha256": parent.db_sha256,
            "fields_map": fields_map,
        }
        all_keys = set()
        for group in plans[dataset]:
            keys = json.loads(group["source_key_status_json"])
            expected = int(group["expected_active_rotors"])
            recording = group["recording"]
            group_id = f"{dataset}/{recording}"
            if len(keys) != (3 if dataset == "DroneAudioSet" else 1) or all_keys.intersection(keys):
                raise ValueError(f"ambiguous group membership: {group_id}")
            group_rows.append(
                {
                    "dataset": dataset,
                    "recording": recording,
                    "group_id": group_id,
                    "expected": expected,
                    "keys": sorted(keys),
                    "baseline_whole": bool(int(group["recording_status_resolved"])),
                    "baseline_ever": bool(int(group["all_rotors_in_one_window"])),
                    "baseline_max_simultaneous": int(group["max_simultaneous_annotated"]),
                }
            )
            all_keys.update(keys)
            for key, status in sorted(keys.items()):
                loc, info = idx.locate(key)
                record = idx.record(key, windows=False)
                tach = idx.tach_arrays(key)
                if record is None or loc is None or not info["has_audio"]:
                    raise ValueError(f"no exact native PCM location: {dataset}/{key}")
                baseline = baseline_for(tach, status, expected)
                digest, size, _ = shards[loc.shard]
                role = baseline["role"] or record.get("tach_role")
                blade_text = str(record.get("blade_count"))
                blades = int(blade_text) if blade_text.isdigit() and int(blade_text) > 0 else None
                blade_basis = record.get("blade_count_basis")
                evidence = None
                if args.figure_blades and blade_text == "unknown":
                    evidence = FIGURE_BLADES.get((dataset, parent.version, key))
                    if evidence is not None:
                        blades, blade_basis = 2, "observed_acquisition_figure"
                measure = (record.get("measure") or {}).get("tach") or {}
                nchan = info.get("n_channels") or 1
                weight = max(1.0, float(info["duration_s"] or 1.0)) * (1 + math.log2(nchan))
                rows.append(
                    {
                        "dataset": dataset,
                        "parent_version": parent.version,
                        "key": key,
                        "group_id": group_id,
                        "recording": recording,
                        "expected": expected,
                        "blades": blades,
                        "blade_count_basis": blade_basis,
                        "blade_count_evidence": evidence,
                        "role": role,
                        "channels": source_channels(record, role),
                        "source_view_label": (record.get("source_detail") or {}).get("mic"),
                        "anchor_binding_allowed": (
                            (record.get("parent") or {}).get("binding") in (None, "none")
                        ),
                        "layout": info["layout"],
                        "shard": loc.shard,
                        "shard_sha256": digest,
                        "shard_bytes": size,
                        "fields": {name: [loc.offset + off, n] for name, off, n in loc.fields},
                        "prior_duration_s": measure.get("seconds_analysed")
                        or info.get("duration_s"),
                        "weight": weight,
                        "baseline": baseline,
                    }
                )
        if all_keys != set(idx.keys(FILTER)):
            raise ValueError(f"annotated original key set differs from frozen cohort: {dataset}")
    if (len(group_rows), len(rows)) != SHAPE or Counter(g["expected"] for g in group_rows) != {
        4: 580,
        1: 28,
    }:
        raise ValueError("wrong frozen group/source counts")
    # Group all longer work across independent workers (LPT scheduling), while
    # the AVQ shard can start first and every source belongs to exactly one job.
    parts = [[] for _ in range(args.parts + 1)]
    parts[0] = [r for r in rows if r["dataset"] == "AVQ"]
    figure_rows = [r for r in rows if r["blade_count_evidence"] is not None]
    if args.figure_blades and len(figure_rows) != len(FIGURE_BLADES):
        raise ValueError(
            "exact figure-derived blade-setting sources not found as unknown in the companion"
        )
    load = [0.0] * args.parts
    other_rows = [r for r in rows if r["dataset"] != "AVQ" and r["blade_count_evidence"] is None]
    for row in sorted(other_rows, key=lambda r: -r["weight"]):
        i = min(range(args.parts), key=load.__getitem__)
        parts[i + 1].append(row)
        load[i] += row["weight"]
    if figure_rows:
        parts.append(figure_rows)
    files = []
    for i, part in enumerate(parts):
        name = f"part-{i:03d}.jsonl.gz"
        file = args.plan / name
        part.sort(key=lambda r: (r["dataset"], r["group_id"], r["key"]))
        write_rows(file, part)
        files.append({"name": name, "rows": len(part), "sha256": sha256(file)})
    write_json(
        args.plan / "plan.json",
        {
            "schema": SCHEMA,
            "release": RELEASE,
            "cohort_sha256": COHORT_SHA256,
            "figure_blade_settings": [
                {
                    k: r[k]
                    for k in (
                        "dataset",
                        "parent_version",
                        "key",
                        "blades",
                        "blade_count_basis",
                        "blade_count_evidence",
                    )
                }
                for r in figure_rows
            ],
            "parents": parents,
            "groups": group_rows,
            "sources": len(rows),
            "parts": files,
        },
    )
    print(f"plan {len(group_rows)} groups / {len(rows)} sources in {len(parts)} parts", flush=True)


def open_plan(path):
    plan = json.loads((path / "plan.json").read_text())
    if (plan["schema"], plan["release"], plan["cohort_sha256"]) != (SCHEMA, RELEASE, COHORT_SHA256):
        raise ValueError("plan does not use the frozen cohort")
    return plan


def plan_part(path, desc):
    file = path / desc["name"]
    if sha256(file) != desc["sha256"]:
        raise ValueError(f"damaged plan part: {file}")
    return read_rows(file)


def native_pcm(shard, ref, channels):
    from data_processing.recording_annotations.scan import read_view

    n = int(ref.n_time)
    fs, t0 = float(ref.rate_hz), float(ref.t_start_s)
    # Bound transient range decoding, including interleaved array channels not
    # selected in the role. This is IO chunking only; the tracker gets one full
    # continuous native-rate record with no clipping, gaps, or reset boundaries.
    chunk = max(1, (32 << 20) // (8 * ref.n_channels))
    if n <= chunk:
        return read_view(shard, ref, t0, t0 + n / fs, channels)[0]
    pcm = np.empty((len(channels), n), dtype=np.float32)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        x, actual = read_view(shard, ref, t0 + lo / fs, t0 + hi / fs, channels)
        if x.shape != (len(channels), hi - lo) or abs((actual - t0) * fs - lo) > 0.25:
            raise ValueError(f"native packed clock mismatch at sample {lo}: {x.shape}, {actual}")
        pcm[:, lo:hi] = x
    return pcm


def select_anchors(row, role, t0, fs, duration, blind):
    baseline = row["baseline"]
    points = baseline["accepted_points"]
    old_start = baseline["audio_t_start_s"]
    if (
        not row["anchor_binding_allowed"]
        or role != baseline["role"]
        or old_start is None
        or abs(old_start - t0) > 0.5 / fs
    ):
        points, reason = [], "no verified same-source role/clock binding"
    else:
        points = [p for p in points if 0 <= p["time_s"] < duration]
        reason = None
    held = [p for p in points if math.floor(p["time_s"] / 12) % 5 == 2]
    if blind or not held:
        used, withheld = [], points
        mode = "blind_requested" if blind else "blind_no_anchor_bearing_holdout"
    else:
        used = [p for p in points if math.floor(p["time_s"] / 12) % 5 != 2]
        withheld = held
        mode = "same_source_acoustic_prior" if used else "blind_no_training_anchor"
    return {
        "mode": mode,
        "selector": ANCHOR_RULE,
        "release": RELEASE,
        "reason": reason,
        "points": used,
        "eligible_count": len(points),
        "withheld_count": len(withheld),
        "used_count": len(used),
        "accepted_baseline_count": len(baseline["accepted_points"]),
    }, withheld


def _grid_cells(times, duration):
    if not len(times):
        return np.empty(0), np.empty(0)
    # Midpoints assign each estimate to a disjoint time cell, but a missing
    # section in an irregular grid must not borrow coverage from distant points.
    hop = min(1.25, float(np.median(np.diff(times)))) if len(times) > 1 else 1.0
    mid = (times[:-1] + times[1:]) * 0.5
    lo = np.maximum(np.r_[0.0, mid], times - 0.5 * hop)
    hi = np.minimum(np.r_[mid, duration], times + 0.5 * hop)
    if times[0] <= hop:
        lo[0] = 0.0
    if duration - times[-1] <= hop:
        hi[-1] = duration
    return lo, hi


def longest_span(bounds):
    longest, left, right = 0.0, 0.0, -math.inf
    for a, b in bounds:
        if a > right + 1e-8:
            left = a
        right = b
        longest = max(longest, b - left)
    return float(longest)


def coverage(track, expected, duration):
    times = np.asarray(track["times_s"], dtype=float)
    rates = np.asarray(track["rates_rps"], dtype=float)
    prov = np.asarray(track["provenance"], dtype=str)
    if (
        times.ndim != 1
        or np.any(~np.isfinite(times))
        or np.any(np.diff(times) <= 0)
        or np.any((times < 0) | (times > duration + 1e-5))
    ):
        raise ValueError("invalid relative track clock")
    if not len(times) or rates.shape != (len(times), expected) or prov.shape != rates.shape:
        raise ValueError(
            "track must supply a nonempty time grid and one rate/provenance column per slot"
        )
    if (
        set(prov.ravel()) - TOKENS
        or np.any(np.isfinite(rates) != np.isin(prov, LEVELS["filled"]))
        or np.any(rates[np.isfinite(rates)] <= 0)
    ):
        raise ValueError("inconsistent track rates and provenance")
    spans = track["diagnostics"]["active_spans"]
    unavailable = track["diagnostics"].get("pcm_unavailable_spans", [])
    if any(not (0 <= a < b <= duration + 1e-5) for a, b in spans):
        raise ValueError(f"invalid relative active spans: {spans!r}")
    lo, hi = _grid_cells(times, duration)
    w = np.zeros(len(times))
    for a, b in spans:
        w += np.maximum(0, np.minimum(hi, b) - np.maximum(lo, a))
    active_s = union_seconds(spans)
    if abs(active_s - sum(b - a for a, b in spans)) > 1e-5:
        raise ValueError("tracker returned overlapping active intervals")
    out = {
        "active_s": active_s,
        "active_slot_s": active_s * expected,
        "active_spans": len(spans),
        "longest_active_s": longest_span(sorted(spans)),
        "grid_active_s": float(w.sum()),
        "unrepresented_active_s": max(0.0, active_s - float(w.sum())),
        "pcm_unavailable_s": union_seconds(unavailable),
        "interpolated_native_slot_s": float((hi - lo) @ (prov == "interpolated").sum(axis=1)),
        "native_s": duration,
        "provenance_active_slot_s": {p: float(w @ (prov == p).sum(axis=1)) for p in sorted(TOKENS)},
    }
    for level, labels in LEVELS.items():
        count = np.isin(prov, labels).sum(axis=1)
        all_s = float(w[count == expected].sum())
        complete_cells = count == expected
        complete_spans = [
            [float(left), float(right)]
            for a, b in sorted(spans)
            for left, right in zip(
                np.maximum(lo[complete_cells], a), np.minimum(hi[complete_cells], b), strict=True
            )
            if right > left
        ]
        out[level] = {
            "any_s": float(w[count > 0].sum()),
            "all_s": all_s,
            "slot_s": float(w @ count),
            "ever": bool(np.any((count == expected) & (w > 0))),
            "whole": bool(active_s > 0 and active_s - all_s <= 1e-5),
            "longest_complete_s": longest_span(complete_spans),
            "max_simultaneous": int(count[w > 0].max(initial=0)),
        }
    return out


def compare_heldout(track, withheld, duration):
    from scipy.optimize import linear_sum_assignment

    times = np.asarray(track["times_s"], dtype=float)
    lo, hi = _grid_cells(times, duration)
    rates = np.asarray(track["rates_rps"], dtype=float)
    prov = np.asarray(track["provenance"], dtype=str)
    comparison = []
    for anchor in withheld:
        t = anchor["time_s"]
        i = int(np.searchsorted(hi, t, side="right"))
        obs = np.asarray(anchor["rates_rps"], dtype=float)
        bands = {}
        for level, labels in LEVELS.items():
            est = rates[i, np.isin(prov[i], labels)] if i < len(lo) and lo[i] <= t else []
            cost = abs(obs[:, None] - np.asarray(est)[None, :])
            # Maximize pairs within tolerance before minimizing their error.
            penalty = (len(obs) + 1) * HOLDOUT_TOLERANCE_RPS
            a, b = linear_sum_assignment(np.where(cost <= HOLDOUT_TOLERANCE_RPS, cost, penalty))
            errors = cost[a, b]
            bands[level] = {
                "n_prediction": len(est),
                "matched_errors_rps": errors[errors <= HOLDOUT_TOLERANCE_RPS].tolist(),
            }
        comparison.append({"time_s": t, "n_prior_rates": len(obs), "levels": bands})
    return comparison


def fit_row(row, parent, opener, model, model_sha, blind):
    from data_processing.recording_annotations.scan import scan_sample

    ts = time.monotonic()
    shard = opener.open(row["shard_sha256"], row["shard_bytes"])
    try:
        scan = scan_sample(
            shard,
            row["shard"],
            row["key"],
            row["fields"],
            dataset=row["dataset"],
            version=row["parent_version"],
            layout=row["layout"],
            fields_map=parent["fields_map"],
        )
        refs = scan.audio_refs()
        role = row["role"]
        if role not in refs:
            if role is not None or len(refs) != 1:
                raise ValueError(f"no verified intended role {role!r}: {list(refs)}; {scan.errors}")
            role = next(iter(refs))
        ref = refs[role]
        n, fs, t0 = ref.n_time, ref.rate_hz, ref.t_start_s
        if n is None or fs is None or t0 is None:
            raise ValueError(f"missing native clock for {row['dataset']}/{row['key']}:{role}")
        documented = row["channels"]
        channels = list(map(int, documented)) if documented else list(range(ref.n_channels))
        x = native_pcm(shard, ref, channels)
    finally:
        shard.close()
    if x.shape != (len(channels), n):
        raise ValueError(f"incomplete native PCM: {x.shape}, expected {n} samples")
    io_s = time.monotonic() - ts
    duration = n / fs
    anchors, withheld = select_anchors(row, role, t0, fs, duration, blind)
    track = model.track(
        x, fs, expected=row["expected"], blades=row["blades"], anchors=anchors["points"] or None
    )
    clean = {
        k: row[k]
        for k in (
            "dataset",
            "parent_version",
            "key",
            "group_id",
            "recording",
            "expected",
            "blades",
            "blade_count_basis",
            "baseline",
        )
    }
    clean.update(
        {
            "blade_count_evidence": row.get("blade_count_evidence"),
            "schema": SCHEMA,
            "review_status": "experimental",
            "role": role,
            "audio_t_start_s": t0,
            "duration_s": duration,
            "sample_rate_hz": fs,
            "n_samples": n,
            "source_view": {
                "role": role,
                "channels": channels,
                "view_label": row["source_view_label"],
                "selection": "documented native tach channels"
                if documented
                else "one native audio role",
            },
            "parent_shard_sha256": row["shard_sha256"],
            "estimator_sha256": model_sha,
            "anchors": anchors,
            "track": track,
            "coverage": coverage(track, row["expected"], duration),
            "heldout": compare_heldout(track, withheld, duration),
        }
    )
    return clean, {
        "io_s": io_s,
        "fit_and_metrics_s": time.monotonic() - ts - io_s,
        "native_pcm_bytes": int(x.nbytes),
        "packed_read": shard.source,
    }


def track_part(args):
    from data_processing.recording_annotations.scan import RemoteRanges, ShardOpener
    from data_processing.streams import open_repository
    from experiments.static_rig import multiscale

    plan = open_plan(args.plan)
    desc = plan["parts"][args.part]
    runner_sha = sha256(__file__)
    model_sha = sha256(multiscale.__file__)
    if model_sha != args.estimator_sha256:
        raise ValueError(
            f"unexpected estimator digest {model_sha} (expected {args.estimator_sha256})"
        )
    remote = RemoteRanges.from_repository(open_repository())
    opener = ShardOpener(args.local_root, remote=remote)
    target = args.out / "tracks" / desc["name"]
    scratch = args.scratch / desc["name"].removesuffix(".jsonl.gz")
    scratch.mkdir(parents=True, exist_ok=True)
    receipts = []
    result_files = []
    for i, row in enumerate(plan_part(args.plan, desc)):
        temp = scratch / f"{i:04d}.json"
        expected_identity = (
            row["dataset"],
            row["parent_version"],
            row["key"],
            model_sha,
            runner_sha,
            desc["sha256"],
            args.blind,
        )
        if temp.exists():
            saved = json.loads(temp.read_text())
            if tuple(saved["identity"]) != expected_identity:
                raise ValueError(f"foreign checkpoint at {temp}")
        else:
            result, metrics = fit_row(
                row, plan["parents"][row["dataset"]], opener, multiscale, model_sha, args.blind
            )
            saved = {"identity": expected_identity, "metrics": metrics, "result": result}
            write_json(temp, saved)
            if i % 5 == 0:
                print(
                    f"{desc['name']} {i + 1}/{desc['rows']} {row['dataset']}/{row['key']} "
                    f"{result['duration_s']:.2f} native s {metrics}",
                    flush=True,
                )
        receipts.append(saved["metrics"])
        result_files.append(temp)
    if len(result_files) != desc["rows"]:
        raise ValueError(f"incomplete planned part: {desc}")
    write_rows(target, (json.loads(p.read_text())["result"] for p in result_files))
    write_json(
        args.out / "receipts" / desc["name"].replace(".jsonl.gz", ".json"),
        {
            "schema": SCHEMA,
            "part": args.part,
            "rows": len(result_files),
            "blind_requested": args.blind,
            "plan_part_sha256": desc["sha256"],
            "estimator_sha256": model_sha,
            "sha256": sha256(target),
            "runner_sha256": runner_sha,
            "job_id": os.getenv("SLURM_JOB_ID"),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "max_native_pcm_bytes": max(r["native_pcm_bytes"] for r in receipts),
            "total_io_s": sum(r["io_s"] for r in receipts),
            "total_fit_and_metrics_s": sum(r["fit_and_metrics_s"] for r in receipts),
            "new_remote_range_gets": remote.n_requests,
            "new_remote_bytes": remote.n_bytes,
            "packed_read_counts": Counter(r["packed_read"] for r in receipts),
        },
    )
    print(f"finished {target} ({len(result_files)} source rows, {sha256(target)})", flush=True)


def group_view(rows):
    # Select once per segment without mixing clocks, even for selected slices.
    # Prioritize supported full simultaneity/time (not just the number of slots
    # detected in different intervals), then filled; minimize uncertainty by
    # preferring direct time at equal coverage. Stable exact-key tie break.
    def rank(r):
        c = r["coverage"]
        d, s, f = (c[x] for x in LEVELS)
        return (
            s["whole"],
            s["all_s"],
            s["max_simultaneous"],
            s["slot_s"],
            f["whole"],
            f["all_s"],
            f["slot_s"],
            d["all_s"],
            d["slot_s"],
            c["active_s"],
        )

    return max(sorted(rows, key=lambda r: r["key"]), key=rank)


def group_status(c):
    if not c["active_s"]:
        return "activity_unresolved"
    if c["filled"]["whole"]:
        return "complete"
    if c["filled"]["any_s"] > 0:
        return "partial"
    return "unresolved"


def baseline_change(groups, level, endpoint):
    pairs = [(r[f"baseline_{endpoint}"], r["coverage"][level][endpoint]) for r in groups]
    return {
        "improved": pairs.count((False, True)),
        "regressed": pairs.count((True, False)),
        "unchanged_positive": pairs.count((True, True)),
        "unchanged_negative": pairs.count((False, False)),
    }


def rollup(groups):
    return {
        "groups": len(groups),
        "sources": sum(len(r["source_views"]) for r in groups),
        "baseline_whole": sum(r["baseline_whole"] for r in groups),
        "baseline_ever": sum(r["baseline_ever"] for r in groups),
        "filled_status": Counter(r["filled_status"] for r in groups),
        "baseline_selected_view_s": {
            fld: sum(r["baseline_on_selected_view"][fld] for r in groups)
            for fld in ("assessed_s", "any_s", "all_s")
        },
        "activity_unresolved_groups": sum(r["coverage"]["active_s"] == 0 for r in groups),
        "active_s": sum(r["coverage"]["active_s"] for r in groups),
        "active_slot_s": sum(r["coverage"]["active_slot_s"] for r in groups),
        "native_s": sum(r["coverage"]["native_s"] for r in groups),
        "pcm_unavailable_s": sum(r["coverage"]["pcm_unavailable_s"] for r in groups),
        "provenance_active_slot_s": {
            p: sum(r["coverage"]["provenance_active_slot_s"][p] for r in groups)
            for p in sorted(TOKENS)
        },
        "interpolated_native_slot_s": sum(
            r["coverage"]["interpolated_native_slot_s"] for r in groups
        ),
        "levels": {
            level: {
                fld: sum(r["coverage"][level][fld] for r in groups)
                for fld in ("any_s", "all_s", "slot_s", "whole", "ever")
            }
            for level in LEVELS
        },
        "change_from_frozen_group_status": {
            level: {
                endpoint: baseline_change(groups, level, endpoint) for endpoint in ("whole", "ever")
            }
            for level in LEVELS
        },
    }


def aggregate_holdout(rows):
    errors = defaultdict(list)
    npred = Counter()
    total = 0
    for r in rows:
        for case in r["heldout"]:
            total += case["n_prior_rates"]
            for level, value in case["levels"].items():
                errors[level].extend(value["matched_errors_rps"])
                npred[level] += value["n_prediction"]
    return {
        "n_prior_rates": total,
        "match_tolerance_rps": HOLDOUT_TOLERANCE_RPS,
        "note": "Agreement independent of supplied acoustic labels in these blocks; not independent tachometry.",
        "levels": {
            level: {
                "n_prediction": npred[level],
                "n_matched": len(errors[level]),
                "mean_matched_abs_error_rps": float(np.mean(errors[level]))
                if errors[level]
                else None,
            }
            for level in LEVELS
        },
    }


def recorded_runner_sha(out: Path, plan: Path) -> str:
    """Use the verified archived runner for a re-summary of a sealed run."""
    receipt = out / "inputs.sha256"
    if not receipt.exists():
        # An in-flight run reduces its own outputs before sealing the sources.
        return sha256(__file__)
    expected = {}
    for line in receipt.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name] = digest
    source = out / "source-overlay.tgz"
    if sha256(source) != expected["source-overlay.tgz"]:
        raise ValueError(f"source overlay does not match the input receipt: {source}")
    if sha256(plan / "plan.json") != expected["plan.json"]:
        raise ValueError(f"plan does not match the input receipt: {plan}")
    with tarfile.open(source, "r:gz") as archive:
        runner = archive.extractfile("./scripts/multiscale_corpus.py")
        if runner is None:
            raise ValueError(f"source overlay does not contain the corpus runner: {source}")
        with runner:
            return hashlib.sha256(runner.read()).hexdigest()


def summarize(args):
    plan = open_plan(args.plan)
    runner_sha = recorded_runner_sha(args.out, args.plan)
    by_group = defaultdict(list)
    seen = set()
    parts = []
    source_index = []
    for desc in plan["parts"]:
        p = args.out / "tracks" / desc["name"]
        receipt = args.out / "receipts" / desc["name"].replace(".jsonl.gz", ".json")
        r = json.loads(receipt.read_text())
        if (
            r["estimator_sha256"],
            r["runner_sha256"],
            r["plan_part_sha256"],
            r["sha256"],
            r["blind_requested"],
        ) != (args.estimator_sha256, runner_sha, desc["sha256"], sha256(p), args.blind):
            raise ValueError(f"output lineage mismatch: {p}")
        parts.append(r)
        for ordinal, (track, planned) in enumerate(
            zip(read_rows(p), plan_part(args.plan, desc), strict=True)
        ):
            k = tuple(track[f] for f in ("dataset", "parent_version", "key", "group_id"))
            if k in seen or any(
                track[f] != planned[f] for f in ("dataset", "parent_version", "key", "group_id")
            ):
                raise ValueError(f"duplicated/unexpected track: {k}")
            seen.add(k)
            source_index.append(
                {
                    **{
                        f: track[f]
                        for f in ("dataset", "parent_version", "key", "group_id", "role")
                    },
                    "part": f"tracks/{desc['name']}",
                    "zero_based_row": ordinal,
                }
            )
            # The complete arrays already live in the track shards. Keep only
            # the per-row metrics and held-out comparisons in the reducer.
            # Derive the current summary metrics from the sealed track, even
            # when this run predates fields added to the coverage rollup.
            track["coverage"] = coverage(track["track"], track["expected"], track["duration_s"])
            del track["track"]
            del track["baseline"]["accepted_points"]
            del track["anchors"]["points"]
            by_group[track["group_id"]].append(track)
    if (len(by_group), len(seen)) != SHAPE or set(by_group) != {
        g["group_id"] for g in plan["groups"]
    }:
        raise ValueError(f"corpus incomplete: {len(by_group)} group(s), {len(seen)} source(s)")
    groups = []
    for group in sorted(plan["groups"], key=lambda g: g["group_id"]):
        rows = by_group[group["group_id"]]
        if sorted(r["key"] for r in rows) != group["keys"]:
            raise ValueError(f"incomplete views: {group['group_id']}")
        best = group_view(rows)
        groups.append(
            {
                **group,
                "coverage_source_key": best["key"],
                "coverage_role": best["role"],
                "coverage": best["coverage"],
                "baseline_on_selected_view": best["baseline"],
                "filled_status": group_status(best["coverage"]),
                "source_views": [
                    {
                        k: r[k]
                        for k in ("key", "role", "source_view", "coverage", "baseline", "anchors")
                    }
                    for r in sorted(rows, key=lambda p: p["key"])
                ],
            }
        )
    grouped = defaultdict(list)
    for g in groups:
        grouped[g["dataset"]].append(g)
    summary = {
        "schema": SCHEMA,
        "review_status": "experimental",
        "release": RELEASE,
        "cohort_sha256": COHORT_SHA256,
        "estimator_sha256": args.estimator_sha256,
        "runner_sha256": runner_sha,
        "parents": plan["parents"],
        "figure_blade_settings": plan.get("figure_blade_settings", []),
        "anchor_rule": "Blind; all eligible frozen same-source points withheld."
        if args.blind
        else ANCHOR_RULE,
        "baseline_comparison": "Frozen group whole/ever vs one new selected view; baseline seconds separately from that exact same view.",
        "coverage_definition": {
            "source_clock": "one independently analyzed native role per source, one selected source per group; no view fusion",
            "denominator": "eligible acoustic active spans (not proof of motor activity), times expected slots; no inferred reduction in rotor count",
            "whole": "positive eligible active duration with every active cell covered at the stated level",
            "ever": "all expected rates simultaneous in at least one positive-duration eligible cell",
            "levels": LEVELS,
            "zero_active": "reported separately; not complete",
            "longest_complete_s": "longest contiguous interval wholly complete at the stated level; never bridges inactive time",
        },
        "all": rollup(groups),
        "by_expected": {str(e): rollup([g for g in groups if g["expected"] == e]) for e in (1, 4)},
        "by_dataset": {
            ds: {
                **rollup(g),
                "heldout": aggregate_holdout(r for group in g for r in by_group[group["group_id"]]),
            }
            for ds, g in grouped.items()
        },
        "heldout": aggregate_holdout(r for rows in by_group.values() for r in rows),
        "source_anchor_modes": Counter(
            r["anchors"]["mode"] for rows in by_group.values() for r in rows
        ),
        "total_native_source_s": sum(r["duration_s"] for rows in by_group.values() for r in rows),
        "parts": parts,
    }
    with (args.out / "sources.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(source_index[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(sorted(source_index, key=lambda r: (r["dataset"], r["key"])))
    write_rows(args.out / "groups.jsonl.gz", groups)
    write_json(args.out / "summary.json", summary)
    columns = [
        "dataset",
        "recording",
        "group_id",
        "expected",
        "coverage_source_key",
        "active_s",
        "filled_status",
        "baseline_whole",
        "baseline_ever",
        "source_keys_json",
        *(
            f"{level}_{end}"
            for level in LEVELS
            for end in ("whole", "ever", "has_any", "max_simultaneous", "all_s")
        ),
    ]
    for tag in ("groups", "partial", "unresolved", "activity_unresolved", "complete"):
        with (args.out / f"{tag}.tsv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for g in groups:
                if tag != "groups" and tag != g["filled_status"]:
                    continue
                c = g["coverage"]
                writer.writerow(
                    {
                        **{k: g[k] for k in columns[:5]},
                        "active_s": c["active_s"],
                        **{k: g[k] for k in columns[6:9]},
                        "source_keys_json": json.dumps(g["keys"], separators=(",", ":")),
                        **{
                            f"{level}_{end}": c[level][end]
                            for level in LEVELS
                            for end in ("whole", "ever", "max_simultaneous", "all_s")
                        },
                        **{f"{level}_has_any": c[level]["any_s"] > 0 for level in LEVELS},
                    }
                )
    print(
        json_line(
            {
                "all": summary["all"],
                "by_expected": summary["by_expected"],
                "source_anchor_modes": summary["source_anchor_modes"],
            }
        ),
        flush=True,
    )


def manifest(args):
    paths = [
        p
        for p in sorted(args.out.rglob("*"))
        if p.is_file() and not p.name.startswith(".") and p.name != "SHA256SUMS"
    ]
    (args.out / "SHA256SUMS").write_text(
        "".join(f"{sha256(p)}  {p.relative_to(args.out)}\n" for p in paths)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    cmd = p.add_subparsers(dest="command", required=True)
    plan = cmd.add_parser("plan")
    plan.add_argument("--cohort", type=Path, required=True)
    plan.add_argument("--plan", type=Path, required=True)
    plan.add_argument(
        "--parts", type=int, required=True, help="parts other than the first AVQ-only part"
    )
    plan.add_argument(
        "--figure-blades",
        action="store_true",
        help="apply two exact-version documented acquisition-figure counts, isolated in the last part",
    )
    part = cmd.add_parser("track")
    part.add_argument("--plan", type=Path, required=True)
    part.add_argument("--out", type=Path, required=True)
    part.add_argument("--part", type=int, required=True)
    part.add_argument("--scratch", type=Path, required=True)
    part.add_argument("--blind", action="store_true")
    part.add_argument("--local-root", type=Path, action="append", default=[])
    part.add_argument("--estimator-sha256", required=True)
    merge = cmd.add_parser("summarize")
    merge.add_argument("--plan", type=Path, required=True)
    merge.add_argument("--out", type=Path, required=True)
    merge.add_argument("--estimator-sha256", required=True)
    merge.add_argument("--blind", action="store_true", help="require all parts to have run blind")
    sums = cmd.add_parser("manifest")
    sums.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    {"plan": build_plan, "track": track_part, "summarize": summarize, "manifest": manifest}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
