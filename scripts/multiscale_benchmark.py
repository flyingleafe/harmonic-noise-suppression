#!/usr/bin/env python3
"""Evaluate whole-source multiscale tracks without changing any annotation.

`freeze` reads the October AVQ linked windows and the prior short-window QC
receipt. Run `run` on uni-cpu on the resulting reference file. The model does
not receive the historical labels during blind comparisons. Times are seconds
from the beginning of the exact input role, not Frame timestamps or seconds
from the October archive's already-trimmed active span.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
TOL = 0.65
PROVENANCE = ("measured", "track_supported", "interpolated", "unresolved", "inactive")
EVIDENCE = ("measured", "track_supported")
MODES = {
    "measured": ("measured",),
    "supported": EVIDENCE,
    "including_interpolation": PROVENANCE[:3],
}
FAMILIES = {
    "S1_seq1": ((51, 54.5), (54.5, 56.5), (56.5, 62), (77, 80)),
    "S1_seq2": ((79, 82.4), (81.6, 85), (85, 88.5), (97, 98.5)),
    "S1_seq3": ((88, 90), (92.5, 94), (95, 97), (109.5, 112.5)),
}
SPEECH = {"S1_seq4": (18, 50), "S2_seq3": (18, 50), "S2_seq4": (18, 50)}
QC_SHA = "dbf50ce4b7a68a747b507a2180bf3b31f20bfd7769bafe4d704cc465ba10e0e1"
AVQ_PIN = "a37840cace1ef87cabcb6c1a3432c67ca942db12611bcca8bd9a782c88622caf"

VK_METADATA = {
    "S1_seq3": (
        "static-rig-tracks-c693d2",
        "1457ff904029aa17134adc941695cfc86456ef4d855142dfbb41da6b07ece2c5",
    ),
    "S2_seq1": (
        "static-rig-combined-3a33f0",
        "82fe7f9f666fdfdfc7bd574ee8ac0fac3b184a584c4880896cb84716f298e152",
    ),
}


def archived_vk(paths, summaries):
    out = {}
    for path in paths:
        row = json.loads(path.read_text())
        key = row["key"]
        job, sha = VK_METADATA[key]
        assert digest(path) == sha and key not in out
        ref = summaries[key]
        t = np.asarray(row["frame_times"]) + float(row["span_s"][0])
        y = np.asarray(
            [r["track"] for r, ok in zip(row["rotors"], ref["valid"], strict=True) if ok]
        ).T
        assert len(y) == len(t) and np.allclose(y.mean(axis=0), ref["means"], atol=0.0007, rtol=0)
        out[key] = dict(**ref, job=job, sha256=sha, times_s=t.tolist(), rates_rps=y.tolist())
    if out.keys() != VK_METADATA.keys():
        raise ValueError(
            "provide both October VK per-frame artifacts, one for each declared source"
        )
    return out


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def js(x):
    x = np.asarray(x)
    return np.where(np.isfinite(x), x, np.asarray(None, dtype=object)).tolist()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def historical_families(d, bands):
    """One archived line per family, then the historical leave-one-out gate.

    Maximize the number of assignable families, preferring each band's first
    (lowest) detection as avq_amp did. Break overlapping-band conflicts by
    distance to band midpoint. Exact duplicated archived rates are one line.
    Do not re-impute after rejection, or treat historical interp as evidence.
    """
    windows = np.asarray([[w["t0"], w["t1"]] for w in d["linked"]]) + d["span_s"][0]
    t = windows.mean(axis=1)
    chosen = np.full((len(t), 4), np.nan)
    for i, leaf in enumerate(d["linked"]):
        rates = np.unique(leaf["speeds"])
        # Four dummy columns allow unassigned families; a valid line beats any dummy.
        cost = np.full((4, len(rates) + 4), 1000.0)
        cost[:, len(rates) :] = 100.0
        for j, (lo, hi) in enumerate(bands):
            ids = np.flatnonzero((lo <= rates) & (rates <= hi))
            cost[j, ids] = np.arange(len(ids)) + 0.1 * abs(rates[ids] - (lo + hi) / 2) / (hi - lo)
        a, b = linear_sum_assignment(cost)
        for j, k in zip(a, b, strict=True):
            if cost[j, k] < 100:
                chosen[i, j] = rates[k]
    found = np.isfinite(chosen)
    for j in range(4):
        ix = np.flatnonzero(found[:, j])
        for left, middle, right in zip(ix[:-2], ix[1:-1], ix[2:], strict=True):
            pred = np.interp(t[middle], t[[left, right]], chosen[[left, right], j])
            found[middle, j] = abs(chosen[middle, j] - pred) < 1.0
    reviewed = np.where(found, chosen, np.nan)
    old_interp = np.full_like(reviewed, np.nan)
    for j in range(4):
        use = found[:, j]
        if use.any():
            old_interp[:, j] = np.interp(t, t[use], reviewed[use, j])
    return windows, t, chosen, found, reviewed, old_interp


def freeze(args):
    store = args.archives
    summary_path = store / "report" / "report.json"
    joint_paths = {k: store / "joint" / f"AVQ__{k}.json" for k in FAMILIES}
    dregon_path = store / "single_rotor" / "dregon_singles.json"
    vk = json.loads(summary_path.read_text())["speeds"]["rows"]
    dregon = json.loads(dregon_path.read_text())["recordings"]
    if digest(args.qc) != QC_SHA:
        raise ValueError("QC receipt is not the frozen steady-tach-releaseq-883d79 SHA-256")
    qc = [
        r
        for line in args.qc.read_text().splitlines()
        if (r := json.loads(line))["key"].startswith("AVQ/")
    ]
    baselines = {}
    guard = dict(S1_seq1=(28, 8, 1), S1_seq2=(32, 23, 20), S1_seq3=(10, 2, 2))
    for key, path in joint_paths.items():
        d = json.loads(path.read_text())
        windows, t, chosen, found, reviewed, old_interp = historical_families(d, FAMILIES[key])
        old_n = old_raw = old_gated = 0
        rows = []
        for r in qc:
            _, rec, leaf_id = r["key"].split("/")
            if rec != key:
                continue
            ix = int(leaf_id)
            old_gated += r["all_matched"]
            old_raw += r["ungated"]["all_matched"]
            for raw, gated in zip(r["ungated"]["details"], r["details"], strict=True):
                assert raw["center"] == gated["center"]
                rows.append(
                    dict(
                        time_s=raw["center"],
                        duration_s=raw["duration"],
                        leaf=ix,
                        ungated=raw["s"],
                        gated=gated["s"],
                        reviewed=js(reviewed[ix]),
                    )
                )
                old_n += 1
                assert windows[ix, 0] <= raw["center"] <= windows[ix, 1], (key, ix, raw)
        if (old_n, old_raw, old_gated) != guard[key]:
            raise ValueError(
                f"{key}: unexpected historical QC counts {(old_n, old_raw, old_gated)}"
            )
        rng = np.random.default_rng(71219 + list(FAMILIES).index(key))
        eligible = np.flatnonzero(found.all(axis=1))
        eligible = eligible[(eligible > 0) & (eligible < len(windows) - 1)]
        ix = int(rng.choice(eligible))
        baselines[key] = dict(
            span_s=d["span_s"],
            windows_s=windows.tolist(),
            center_s=t.tolist(),
            assigned_before_smooth=js(chosen),
            found=found.tolist(),
            reviewed=js(reviewed),
            historical_interp_descriptive=js(old_interp),
            blackout_s=[windows[ix, 0], windows[ix, 1]],
            blackout_leaf=ix,
            archived_qc_vs_original=guard[key],
            short_qc=rows,
        )
    row = baselines["S1_seq1"]["reviewed"][2]
    assert 55.0187 < row[1] < 55.0189
    write_json(
        args.out,
        dict(
            tolerance_rps=TOL,
            avq_pin=AVQ_PIN,
            avq_families_commit="3edac1b9",
            vk_summary_commit="6920ecc0",
            archived_qc_job="steady-tach-releaseq-883d79",
            sha256={
                p.name: digest(p)
                for p in [*joint_paths.values(), args.qc, summary_path, dregon_path]
            },
            avq=baselines,
            avq_vk=archived_vk(args.vk_track, {r["key"]: r for r in vk if r["key"] in VK_METADATA}),
            single_rates={
                **{k: v["s_bar_rev_s"] for k, v in dregon.items()},
                **{r["key"]: r["means"][0] for r in vk if "__single_rotors__" in r["key"]},
            },
            single_spans={k: v["span_s"] for k, v in dregon.items()},
            protocol=dict(
                pit="maximum number within tolerance, then minimum total error; one-to-one",
                grid="fixed 0.5+n s, nearest output knot <=0.55 s, never score-impute",
                blackout="mask PCM with NaN (all channels), exclude all anchors inside; no truth at probe",
                overlap="TP / (truth_count + emitted_count - TP); include refusals and extra rates",
                full="exact emitted cardinality and every reference rate matched; omit unknown truth",
                synthetic_rates="analytically integrated phase via cumulative shaft rate",
                controls="expected=4 even on speech/singles/ties/clones; no source setting as speed",
            ),
        ),
    )


def pit(pred, truth):
    """Maximum tolerance matches; among those, minimize total absolute error."""
    if len(pred) and len(truth):
        error = np.abs(np.asarray(pred)[:, None] - np.asarray(truth))
        weight = np.minimum(error, 1e5) / 1e6
        a, b = linear_sum_assignment((error > TOL).astype(float) + weight)
        err = error[a, b]
        return int((err <= TOL).sum()), err.tolist()
    return 0, []


def metrics(rows):
    """Rows are (prediction, reference, seconds_weight); reference None means unknown."""
    s = Counter()
    counts = Counter()
    errors = []
    for pred, ref, weight in rows:
        n = len(pred)
        s["scored_units"] += 1
        s["scored_weight"] += weight
        s["emitted_weight"] += weight * n
        counts[n] += weight
        if ref is None:
            continue
        nr = len(ref)
        tp, err = pit(pred, ref)
        errors.extend(err)
        s["labeled_weight"] += weight
        s["reference_weight"] += weight * nr
        s["emitted_on_labeled_weight"] += weight * n
        s["matched_weight"] += weight * tp
        s["fully_matching_weight"] += weight * (n == nr == tp)
        s["extra_weight"] += weight * max(0, n - nr)
        s["any_extra_weight"] += weight * (n > nr)
        s["wrong_or_extra_weight"] += weight * (n - tp)

    def ratio(n, d):
        return n / d if d else None

    return dict(
        **s,
        n_by_emitted=dict(counts),
        recall=ratio(s["matched_weight"], s["reference_weight"]),
        precision=ratio(s["matched_weight"], s["emitted_on_labeled_weight"]),
        overlap=ratio(
            s["matched_weight"],
            s["emitted_on_labeled_weight"] + s["reference_weight"] - s["matched_weight"],
        ),
        full_match=ratio(s["fully_matching_weight"], s["labeled_weight"]),
        conditional_abs_error=dict(
            n=len(errors),
            mae=float(np.mean(errors)) if errors else None,
            p95=float(np.quantile(errors, 0.95)) if errors else None,
            max=float(max(errors)) if errors else None,
        ),
    )


def as_track(x, fs, expected, blades, anchors=None):
    from experiments.static_rig.multiscale import track

    started = time.perf_counter()
    out = track(x, fs, expected=expected, blades=blades, anchors=anchors)
    check_track(out, x.shape[1] / fs, expected)
    out["elapsed_s"] = time.perf_counter() - started
    return out


def check_track(out, duration, expected):
    rt = np.asarray(out["times_s"], dtype=float)
    p = np.asarray(out["provenance"])
    y = np.asarray(out["rates_rps"], dtype=float)
    shape = (len(rt), expected)
    assert p.shape == y.shape == shape or len(rt) == len(y) == len(p) == 0
    assert not len(rt) or (
        np.isfinite(rt).all() and (np.diff(rt) > 0).all() and 0 <= rt[0] <= rt[-1] <= duration
    )
    assert np.isin(p, PROVENANCE).all()
    assert (np.isfinite(y) == np.isin(p, PROVENANCE[:3])).all()
    assert (y[np.isfinite(y)] > 0).all()


def select_at(out, times):
    rt = np.asarray(out["times_s"], dtype=float)
    if not len(rt):
        return [[] for _ in times], [[] for _ in times], [None] * len(times)
    y, p = np.asarray(out["rates_rps"], dtype=float), np.asarray(out["provenance"])
    ix = np.searchsorted(rt, times).clip(0, len(rt) - 1)
    prev = (ix - 1).clip(0)
    ix = np.where(abs(rt[prev] - times) <= abs(rt[ix] - times), prev, ix)
    use = abs(rt[ix] - times) <= 0.55
    return (
        [js(y[i]) if g else [] for i, g in zip(ix, use, strict=True)],
        [p[i].tolist() if g else [] for i, g in zip(ix, use, strict=True)],
        [float(rt[i]) if g else None for i, g in zip(ix, use, strict=True)],
    )


def compare(out, times, target, weights=None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y, p, actual = select_at(out, np.asarray(times))
    if weights is None:
        weights = np.ones(len(times))
    summaries = {}
    for mode, keep in MODES.items():
        rows = [
            (np.asarray([v for v, q in zip(ys, ps, strict=True) if q in keep]), ref, w)
            for ys, ps, ref, w in zip(y, p, target, weights, strict=True)
        ]
        summaries[mode] = metrics(rows)
    return summaries, [
        dict(time_s=float(t), output_knot_s=at, prediction=ys, provenance=ps, reference=ref)
        for t, at, ys, ps, ref in zip(times, actual, y, p, target, strict=True)
    ]


def grid(duration):
    return np.arange(0.5, duration, 1.0)


def qcs(out, historical):
    rows = historical["short_qc"]
    times = np.asarray([r["time_s"] for r in rows])
    refs = [r["reviewed"] if all(v is not None for v in r["reviewed"]) else None for r in rows]
    p, detail = compare(out, times, refs)
    return dict(
        proposed=p,
        old_raw=metrics((r["ungated"], ref, 1) for r, ref in zip(rows, refs, strict=True)),
        old_gated=metrics((r["gated"], ref, 1) for r, ref in zip(rows, refs, strict=True)),
        exact_centers=[{**r, **d} for r, d in zip(rows, detail, strict=True)],
    )


def avq_report(out, frozen, duration, historical, vk):
    times = grid(duration)
    unc = [None] * len(times)
    res: dict[str, Any] = dict(full_run_coverage=compare(out, times, unc)[0])
    if historical:
        t = historical["center_s"]
        # Partial references are valid for recall, not claims of false extras.
        target = [[v for v in row if v is not None] for row in historical["reviewed"]]
        sel = [i for i, row in enumerate(target) if len(row) == 4]
        res["archive_leaf_centers_all_four"] = compare(
            out, [t[i] for i in sel], [target[i] for i in sel]
        )
        _, found_probe = compare(out, t, [r or None for r in target])
        res["detected_archive_families_by_leaf"] = found_probe
        res["detected_archive_family_recall"] = {
            mode: dict(
                matched=sum(
                    pit(
                        [
                            v
                            for v, p in zip(row["prediction"], row["provenance"], strict=True)
                            if p in keep
                        ],
                        row["reference"],
                    )[0]
                    for row in found_probe
                    if row["reference"]
                ),
                available=sum(map(len, target)),
            )
            for mode, keep in MODES.items()
        }
        res["archived_short_windows"] = qcs(out, historical)
    if vk:
        # October VK summary is a second acoustic reading, not physical truth.
        # These are full-record slot means, not replicated per-frame truth.
        y, p = np.asarray(out["rates_rps"], dtype=float), np.asarray(out["provenance"])
        means = (
            []
            if not len(y)
            else [
                float(y[valid, j].mean()) if valid.any() else None
                for j in range(y.shape[1])
                if (valid := np.isin(p[:, j], EVIDENCE)).any()
            ]
        )
        old = [v for v, ok in zip(vk["means"], vk["valid"], strict=True) if ok]
        slot_coverage = [] if not len(y) else np.isin(p, EVIDENCE).sum(axis=0).tolist()
        res["october_vk_summary"] = dict(
            archived=vk,
            supported_slot_means=means,
            supported_slot_counts=slot_coverage,
            pit_abs_errors_rps=pit(means, old)[1],
        )
        if "times_s" in vk:
            res["october_vk_pit_descriptive"] = compare(out, vk["times_s"], vk["rates_rps"])
    if frozen == "S1_seq1":
        truth = historical["reviewed"][2]
        spot = [18.0, 18.5, 19.0, 19.5, 20.0]
        res["formerly_gated_55_019"] = dict(
            comparison=compare(out, spot, [truth] * len(spot)),
            matching_family_index=1,
            archived_leaf_s=historical["windows_s"][2],
            release_candidate_check=None,
        )
    return res


def blackout(x, fs, gap, anchors):
    a, b = (round(value * fs) for value in gap)
    assert 0 < a < b < x.shape[1]
    y = x.copy()
    y[:, a:b] = np.nan
    return y, [k for k in anchors if not a / fs <= k["time_s"] < b / fs]


class NativePanel:
    """Read full exact-source audio using the already-replayed pilot's range loader."""

    def __init__(self, frozen, table):
        from dload.remote import LocalRemote

        from data_processing.recording_annotations.scan import RemoteRanges
        from data_processing.recording_annotations.storage import open_release
        from data_processing.streams import open_repository
        from experiments.static_rig.close_pair_data import RELEASE, panel

        self.repo = repo = open_repository()
        self.ranges = (
            None if isinstance(repo.remote, LocalRemote) else RemoteRanges.from_repository(repo)
        )
        release = open_release(version=RELEASE, repo=repo)
        self.rows = panel(table)
        self.parents = {p.dataset: p.version for p in release.parents()}
        self.indexes, self.shards = {}, {}
        if self.parents["AVQ"] != frozen["avq_pin"]:
            raise ValueError("October AVQ input pin differs from the comparison release")
        self.release = release

    def load(self, dataset, key, *, span=None, esc=False):
        from data_processing.recording_annotations.scan import read_view
        from experiments.static_rig.close_pair_data import _telemetry, _waveform

        if dataset not in self.indexes:
            index = self.indexes[dataset] = self.release.index(dataset, self.parents[dataset])
            self.shards[dataset] = index.parent_shards()
        index = self.indexes[dataset]
        loc, info = index.locate(key)
        if loc is None or not info["has_audio"]:
            raise ValueError(f"{dataset}/{key}: frozen parent has no waveform")
        record = index.record(key)
        tach = record.get("tach") or {}
        role = str(tach.get("role") or "audio")
        with _waveform(
            self.repo,
            self.ranges,
            index,
            self.shards[dataset][loc.shard],
            loc,
            key,
            role,
            with_esc=esc,
        ) as (shard, ref, fields):
            t0, fs, n = ref.t_start_s, ref.rate_hz, ref.n_time
            if t0 is None or fs is None or n is None:
                raise ValueError(
                    f"{dataset}/{key}: waveform lacks a native time start, rate or length"
                )
            if (
                tach.get("audio_t_start_s") is not None
                and abs(tach["audio_t_start_s"] - t0) > 1 / fs
            ):
                raise ValueError(f"{dataset}/{key}: companion and sample clocks disagree")
            a, b = (0, n) if span is None else (round(span[0] * fs), min(n, round(span[1] * fs)))
            samples = np.empty((ref.n_channels, b - a), dtype=np.float32)
            block = round(12 * fs)
            for lo in range(a, b, block):
                hi = min(b, lo + block)
                x, actual = read_view(shard, ref, t0 + lo / fs, t0 + hi / fs)
                assert (
                    x.shape == (ref.n_channels, hi - lo)
                    and abs(actual - (t0 + lo / fs)) < 0.01 / fs
                )
                samples[:, lo - a : hi - a] = x
            lineage = dict(
                dataset=dataset,
                key=key,
                role=role,
                parent_version=index.version,
                shard_digest=shard.digest,
                audio_frame_start_s=t0,
                input_frame_start_s=t0 + a / fs,
                source_samples=[a, b],
                sample_rate_hz=fs,
                channels=ref.n_channels,
                anchor_binding_allowed=(record.get("parent") or {}).get("binding")
                in (None, "none"),
                pcm_sha256=hashlib.sha256(samples).hexdigest(),
            )
            telem = _telemetry(shard, fields) if esc else None
        return samples, fs, lineage, tach, telem


def gap_report(out, gap, times, ref):
    selected = [(t, r) for t, r in zip(times, ref, strict=True) if gap[0] <= t < gap[1]]
    if selected:
        t, r = map(list, zip(*selected, strict=True))
    else:
        t, r = [], []
    scores, rows = compare(out, t, r)
    interpolation = metrics(
        (
            [
                v
                for v, q in zip(row["prediction"], row["provenance"], strict=True)
                if q == "interpolated"
            ],
            row["reference"],
            1,
        )
        for row in rows
    )
    return dict(blackout_s=gap, scores=scores, interpolation_only=interpolation, probes=rows)


def save_case(outdir, case, result):
    write_json(outdir / f"{case}.json", result)
    print("CASE " + case, flush=True)


def released_spot(tach):
    # Released estimate is a third acoustic comparison on the same clock.
    # The old ad-hoc QC analysis used independently chosen window boundaries.
    candidates = []
    for w in tach.get("windows", []):
        start, end = w.get("start_s"), w.get("end_s")
        if (
            start is not None
            and end is not None
            and abs(start - 18.0) < 0.26
            and abs(end - 20.0) < 0.26
        ):
            candidates.append(w)
    return dict(search_s=[18.0, 20.0], windows=candidates)


def release_reference(out, tach):
    windows = tach.get("windows", [])
    times, reference = [], []
    for w in windows:
        if w["center_s"] is None:
            continue
        v = sorted({r for r in w["rates_rev_s"] if r is not None})
        times.append(w["center_s"])
        reference.append(v if w["accepted"] and len(v) == 4 else None)
    return dict(
        same_source_release_is_acoustic_only=True,
        attempted_centers=len(times),
        complete_reference_centers=sum(r is not None for r in reference),
        comparison=compare(out, times, reference),
    )


def scale_comparison(blind, anchors, times, refs, label):
    """Compare a fixed post-fit transformation, never using test references as priors."""
    from experiments.static_rig.scale_tracks import calibrate

    # With no admissible prior the output must be bit-identical to blind;
    # do not even invoke the post-fit duplicate-removal path.
    scaled = calibrate(blind, anchors, written_for=label) if anchors else blind
    decisions = scaled["diagnostics"].get("scale_calibration", {})
    return dict(
        anchors=anchors,
        identity_no_prior=not anchors,
        anchor_source=label,
        track_decisions=dict(Counter(r["basis"] for r in decisions.get("tracks", []))),
        coincident_cells_removed=decisions.get("coincident_cells_removed", 0),
        blind=compare(blind, times, refs),
        conditional=compare(scaled, times, refs),
        calibrated_track=scaled if anchors else None,
    )


def companion_split(tach, lineage, duration):
    """Exact old source/role clocks only; 12-s blocks 2 modulo 5 are held out."""
    if (
        not lineage["anchor_binding_allowed"]
        or lineage["role"] != tach.get("role")
        or tach.get("audio_t_start_s") != lineage["input_frame_start_s"]
    ):
        return [], []
    anchors, held = [], []
    for w in tach["windows"]:
        t = w["center_s"]
        if t is None or not 0 <= t < duration or not w["accepted"]:
            continue
        rates = sorted({r for r in w["rates_rev_s"] if r is not None})
        if not rates:
            continue
        if int(t // 12) % 5 == 2:
            held.append(dict(time_s=t, rates_rps=rates))
        else:
            # Even complete old windows supply at most one witnessed shaft to
            # calibration; there is no per-time complete-set teacher.
            anchors.append(dict(time_s=t, rates_rps=[rates[len(anchors) % len(rates)]]))
    return (anchors if held else []), held


def sparse_conditioning(times, refs, *, first, step, exclude_s, gap=None):
    anchors = []
    withheld = np.ones(len(times), dtype=bool)
    for rot, i in enumerate(range(first, len(times), step)):
        t, r = float(times[i]), refs[i]
        if not r or (gap is not None and gap[0] <= t < gap[1]):
            continue
        anchors.append(dict(time_s=t, rates_rps=[r[rot % len(r)]]))
        withheld &= abs(times - t) > exclude_s
    return anchors, withheld


def reused_avq(directory, lineage):
    """Reuse only exact, blind, full-role PCM readback from this model snapshot."""
    from experiments.static_rig import multiscale

    key = lineage["key"]
    path = directory / f"{list(FAMILIES).index(key):04d}.json"
    row = json.loads(path.read_text())["result"]
    checks = dict(
        dataset="AVQ",
        key=key,
        parent_version=lineage["parent_version"],
        role=lineage["role"],
        parent_shard_sha256=lineage["shard_digest"],
        audio_t_start_s=lineage["input_frame_start_s"],
        sample_rate_hz=lineage["sample_rate_hz"],
        n_samples=lineage["source_samples"][1],
        estimator_sha256=digest(Path(multiscale.__file__)),
        expected=4,
        blades=2,
    )
    if (
        any(row[k] != v for k, v in checks.items())
        or lineage["source_samples"][0] != 0
        or row["source_view"]["channels"] != list(range(lineage["channels"]))
        or row["anchors"]["mode"] != "blind_requested"
        or row["anchors"]["points"]
    ):
        raise ValueError(f"{key}: refusing blind replay with a mismatched source/view/model")
    check_track(row["track"], row["duration_s"], expected=4)
    lineage["reused_blind_receipt_sha256"] = digest(path)
    return row["track"]


def avq_group(group, frozen, panel, outdir, reuse_blind=None):
    keys = tuple(FAMILIES) if group == "avq_dev" else ("S2_seq1", "S2_seq2", "S2_seq5", "S2_seq6")
    avq = frozen["avq"]
    for key in keys:
        x, fs, lineage, tach, _ = panel.load("AVQ", key)
        duration = x.shape[1] / fs
        historical = avq.get(key)
        blind = as_track(x, fs, 4, 2) if reuse_blind is None else reused_avq(reuse_blind, lineage)
        summary = avq_report(blind, key, duration, historical, frozen["avq_vk"].get(key))
        if key == "S1_seq1":
            summary["formerly_gated_55_019"]["release_candidate_check"] = released_spot(tach)
        if not historical:
            summary["frozen_release_complete_windows"] = release_reference(blind, tach)
        result: dict[str, Any] = dict(lineage=lineage, blind=blind, summary=summary)
        if not historical:
            anchors, held = companion_split(tach, lineage, duration)
            result["scale_calibration"] = scale_comparison(
                blind,
                anchors,
                [p["time_s"] for p in held],
                [p["rates_rps"] if len(p["rates_rps"]) == 4 else None for p in held],
                "497ea train-block single acoustic point on same waveform role",
            )
        if historical:
            observed = [
                (t, [r for r in s if r is not None])
                for t, s in zip(historical["center_s"], historical["reviewed"], strict=True)
            ]
            anchors = [
                dict(time_s=t, rates_rps=[r[i % len(r)]]) for i, (t, r) in enumerate(observed) if r
            ]
            gap = historical["blackout_s"]
            masked, outside = blackout(x, fs, gap, anchors)
            target = [s if all(r is not None for r in s) else None for s in historical["reviewed"]]
            blocked_blind = as_track(masked, fs, 4, 2)
            blocked_seeded = as_track(masked, fs, 4, 2, outside) if outside else blocked_blind
            result["historical_blackout"] = dict(
                scored_only_against_found_all_four=True,
                withheld_anchor_count=len(anchors) - len(outside),
                supplied_anchors=outside,
                blind=blocked_blind,
                historical_seeded=blocked_seeded,
                blind_score=gap_report(blocked_blind, gap, historical["center_s"], target),
                seeded_score=gap_report(blocked_seeded, gap, historical["center_s"], target),
            )
        save_case(outdir, key, result)


def count_control(out, duration, n):
    times = grid(duration)
    rates, provenance, actual = select_at(out, times)
    report: dict[str, Any] = {}
    for mode, keep in MODES.items():
        counts = [sum(p in keep for p in row) for row in provenance]
        report[mode] = dict(
            grid_seconds=len(times),
            any_output_seconds=int(np.sum(np.asarray(counts) > 0)),
            over_count_seconds=int(np.sum(np.asarray(counts) > n)),
            four_output_seconds=int(np.sum(np.asarray(counts) >= 4)),
            max_count=max(counts, default=0),
            false_extra_rate_seconds=sum(max(0, k - n) for k in counts),
            emitted_rate_seconds=sum(counts),
        )
    report["checkpoints"] = [
        dict(
            time_s=float(times[i]),
            output_knot_s=actual[i],
            rates_rps=rates[i],
            provenance=provenance[i],
        )
        for i in sorted({0, len(times) // 2, len(times) - 1})
        if len(times)
    ]
    return report


def real_nulls(group, frozen, panel, outdir):
    cases: list[tuple[str, str, int, int, tuple[int, int] | None]]
    if group == "speech":
        cases = [("AVQ", key, 2, 0, span) for key, span in SPEECH.items()]
    else:
        dataset = "DREGON-frames" if group in ("dregon_dev", "dregon_test") else "SPCUP19-frames"
        split = "development" if group.endswith("_dev") else "test"
        blades = 2 if dataset == "DREGON-frames" else 3
        cases = [
            (dataset, r["key"], blades, 1, None)
            for r in panel.rows
            if r["arm"] == "single" and r["dataset"] == dataset and r["split"] == split
        ]
    for ds, key, blades, n, span in cases:
        x, fs, lineage, tach, _ = panel.load(ds, key, span=span)
        duration = x.shape[1] / fs
        forced = as_track(x, fs, 4, blades)
        summary: dict[str, Any] = dict(forced_expected_4=count_control(forced, duration, n))
        result: dict[str, Any] = dict(lineage=lineage, forced_expected_4=forced, summary=summary)
        if n:
            desired = as_track(x, fs, 1, blades)
            result["matched_expected_1"] = desired
            summary["matched_expected_1"] = count_control(desired, duration, n)
            t = grid(duration)
            lo, hi = frozen["single_spans"].get(key, [0, duration])
            ref = [[frozen["single_rates"][key]] if lo <= value < hi else None for value in t]
            summary["forced_archived_acoustic_rate"] = compare(forced, t, ref)[0]
            summary["single_archived_acoustic_rate"] = compare(desired, t, ref)[0]
            if group.endswith("_test"):
                anchors, _ = companion_split(tach, lineage, duration)
                mask = (
                    np.array([int(value // 12) % 5 == 2 for value in t], dtype=bool)
                    if anchors
                    else np.ones(len(t), dtype=bool)
                )
                result["scale_calibration"] = scale_comparison(
                    forced,
                    anchors,
                    t[mask],
                    [r for r, ok in zip(ref, mask, strict=True) if ok],
                    "497ea train-block isolated single acoustic rate on exact source clock",
                )
        else:
            t = grid(duration)
            summary["forced_speech_zero_track"] = compare(forced, t, [[] for _ in t])[0]
        save_case(outdir, key, result)


def md2_truth(times, telem):
    """Use the published Frame alignment, not refined acoustic RPS or stretched ticks."""
    t = telem["motor_speed_time"]
    rates = telem["motor_speed"][:4].T / 60.0
    assert len(t) == len(rates) and rates.shape[1] == 4 and (np.diff(t) >= 0).all()
    out = []
    for value in times:
        i = np.searchsorted(t, value)
        if not (0 < i < len(t)) or t[i] - t[i - 1] > 0.20:
            out.append(None)
            continue
        r = rates[i - 1] + (rates[i] - rates[i - 1]) * ((value - t[i - 1]) / (t[i] - t[i - 1]))
        out.append(r.tolist() if np.isfinite(r).all() and np.min(r) > 20 else None)
    return out


def md2_group(panel, outdir):
    for key in ("FLY124", "FLY125"):
        x, fs, lineage, _, telem = panel.load("michaels-frames", key, esc=True)
        if telem is None:
            raise ValueError(f"{key}: native ESC telemetry is missing")
        t0 = lineage["input_frame_start_s"]
        t = grid(x.shape[1] / fs)
        truth = md2_truth(t + t0, telem)
        # One native ESC speed at each sparse training time; tests exclude
        # their +/- 2 s neighborhoods in the blind, seeded and scale-only arms.
        anchors, choice = sparse_conditioning(t, truth, first=6, step=15, exclude_s=2.0)
        blind = as_track(x, fs, 4, 2)
        seeded = as_track(x, fs, 4, 2, anchors) if anchors else blind
        scale = scale_comparison(
            blind,
            anchors,
            t[choice],
            [r for r, keep in zip(truth, choice, strict=True) if keep],
            "single raw MD2 ESC RPM/60 on published Frame train time",
        )
        withheld = [r for r, keep in zip(truth, choice, strict=True) if keep]
        if not any(r is not None for r in withheld):
            raise ValueError(f"{key}: no native ESC/audio time overlap")
        save_case(
            outdir,
            key,
            dict(
                lineage=lineage,
                anchor_source="same-record raw ESC RPM/60 on published Frame clock",
                anchor_exclusion_s=2.0,
                anchors=anchors,
                blind=blind,
                seeded=seeded,
                reference_full=truth,
                reference_grid_s=t.tolist(),
                independent_blind_full=compare(blind, t, truth)[0],
                withheld_blind=compare(blind, t[choice], withheld),
                withheld_seeded=compare(seeded, t[choice], withheld),
                scale_calibration=scale,
            ),
        )


def real_mix(frozen, panel, outdir):
    # No mic/clock fusion: additive synthetic mixtures of separately recorded
    # mono channel-0 sources on one new generated sample clock. Each source uses
    # its own native uncontaminated full-record acoustic estimate as reference.
    for throttle, arm in ((70, "close"), (None, "spread")):
        sources, rate, x = [], [], []
        fs0 = None
        for m, su in enumerate((50, 60, 80, 90), 1):
            key = f"motor_Motor{m}_{throttle or su}"
            a, b = frozen["single_spans"][key]
            margin = min(3.0, (b - a) / 4)
            wave, fs, lineage, _, _ = panel.load(
                "DREGON-frames", key, span=(a + margin, b - margin)
            )
            if fs0 is not None and fs != fs0:
                raise ValueError("real mixture would combine native rates without resampling")
            fs0 = fs
            sources.append(dict(**lineage, channel=0))
            x.append(wave[0].copy())
            rate.append(frozen["single_rates"][key])
        n = min(map(len, x))
        y = np.zeros((1, n), dtype=np.float32)
        for signal in x:
            wave = signal[(len(signal) - n) // 2 :][:n]
            y[0] += wave / np.sqrt(np.mean(wave**2, dtype=np.float64))
        assert fs0 is not None
        out = as_track(y, fs0, 4, 2)
        t = grid(n / fs0)
        save_case(
            outdir,
            arm,
            dict(
                sources=sources,
                rates_are_isolated_acoustic_estimates=True,
                mixed_pcm_sha256=hashlib.sha256(y).hexdigest(),
                blind=out,
                rates_rps=rate,
                scores=compare(out, t, [rate] * len(t)),
            ),
        )


def analytic_case(kind, seed):
    rng = np.random.default_rng(seed)
    fs, duration, channels = 12000, 32, 3
    t = np.arange(fs * duration) / fs
    constant = 64 + rng.uniform(-3, 3)
    other = np.asarray([82.3, 103.5]) + rng.uniform(-2, 2, 2)
    rates = None
    order = tuple(range(1, 13))
    levels = [1.0] * 4
    if kind in ("tie", "scale_clone", "odd", "even_only"):
        physical = [constant] * 4 if kind == "tie" else [constant]
        rates = np.broadcast_to(np.asarray(physical)[None, :], (len(t), len(physical)))
        if kind == "odd":
            order = tuple(range(1, 14, 2))
        if kind == "even_only":
            order = tuple(range(2, 20, 2))
    elif kind in ("static_separated", "static_close025", "static_close065", "static_close150"):
        delta = {
            "static_separated": 5.0,
            "static_close025": 0.25,
            "static_close065": 0.65,
            "static_close150": 1.5,
        }[kind]
        rates = np.broadcast_to(
            np.asarray([constant, constant + delta, *other])[None, :], (len(t), 4)
        )
        if kind != "static_separated":
            levels = [1.0, 0.44, 0.83, 1.0]
    elif kind in ("drift", "drift_gap2", "drift_gap4", "drift_gap7"):
        z = (t - duration / 2) / duration
        rates = np.column_stack(
            (
                constant - 1.3 + (3.2 * z + 0.25 * np.sin(2 * np.pi * z)),
                constant + 1.3 - (2.4 * z + 0.4 * np.sin(2 * np.pi * z)),
                other[0] + 4.0 * z - 0.55 * np.cos(2 * np.pi * z),
                other[1] - 3.0 * z + 0.75 * np.sin(3 * np.pi * z),
            )
        )
    else:
        raise ValueError(kind)
    x = np.zeros((channels, len(t)))
    # Orders with a shaft fundamental plus favoured even/BPF orders; mic
    # phases and levels vary but share the generating time base.
    for j, r in enumerate(rates.T):
        phi = (2 * np.pi / fs) * (np.cumsum(r) - r[0])
        for k in order:
            amps = rng.uniform(0.55, 1.45, size=(channels, 1)) * (0.45 if k % 2 else 1.0) / k**0.65
            phase = rng.uniform(0, 2 * np.pi, size=(channels, 1))
            am = 1 + 0.22 * np.cos(2 * np.pi * rng.uniform(0.22, 0.8) * t + phase)
            x += levels[j] * amps * am * np.sin(k * phi + phase)
            if kind == "scale_clone" and k >= 4:
                x += 0.7 * amps * am * np.sin(2 * k * phi - phase)
    x += rng.normal(0, 0.12, size=x.shape)
    return x.astype(np.float32), fs, rates, rng


def synthetic_group(group, outdir):
    dev = group == "synth_dev"
    kinds = (
        "static_separated",
        "static_close025",
        "static_close065",
        "static_close150",
        "drift",
        "drift_gap2",
        "drift_gap4",
        "drift_gap7",
        "tie",
        "scale_clone",
        "odd",
        "even_only",
    )
    # Disjoint generation across development and sealed cases. Sample the gap
    # without looking at sound, fractions of active time, or model confidence.
    for index, kind in enumerate(kinds):
        for rep in range(1 if dev else 3):
            seed = (40789 if dev else 91547) + 19 * index + rep
            x, fs, rates, rng = analytic_case(kind, seed)
            t = grid(x.shape[1] / fs)
            ref = [np.unique(rates[round(u * fs)]).tolist() for u in t]
            arm = f"{kind}-{rep}"
            gap = None
            if "_gap" in kind:
                start = int(rng.integers(8, 18))
                gap = [start, start + int(kind[-1])]
                x, _ = blackout(x, fs, gap, [])
            anchors, held = sparse_conditioning(t, ref, first=2, step=4, exclude_s=1.0, gap=gap)
            blind = as_track(x, fs, 4, 2)
            sampled = compare(blind, t, ref)
            result: dict[str, Any] = dict(
                seed=seed,
                case=kind,
                blind=blind,
                blind_full_scores=sampled,
                null_count=count_control(blind, x.shape[1] / fs, 1)
                if kind in ("tie", "scale_clone", "odd", "even_only")
                else None,
            )
            result["scale_calibration"] = scale_comparison(
                blind,
                anchors,
                t[held],
                [r for r, ok in zip(ref, held, strict=True) if ok],
                "one planted shaft per sparse train time outside any blackout",
            )
            if kind.startswith("static_close"):
                gap_rps = ref[0][1] - ref[0][0]
                exact = min(0.20, 0.30 * gap_rps)
                result["close_pair_strict"] = {
                    "tolerance_rps": exact,
                    "supported_pair_seconds": sum(
                        abs(u - row["reference"][0]) <= exact
                        and abs(v - row["reference"][1]) <= exact
                        for row in sampled[1]
                        for kept in [
                            sorted(
                                s
                                for s, p in zip(row["prediction"], row["provenance"], strict=True)
                                if p in EVIDENCE
                            )
                        ]
                        if len(kept) >= 2
                        for u, v in [
                            min(
                                itertools.combinations(kept, 2),
                                key=lambda pair: sum(
                                    abs(x - y)
                                    for x, y in zip(pair, row["reference"][:2], strict=True)
                                ),
                            )
                        ]
                    ),
                }
            if gap is not None:
                seeded = as_track(x, fs, 4, 2, anchors)
                result.update(
                    supplied_anchors=anchors,
                    seeded=seeded,
                    seeded_withheld_score=compare(
                        seeded, t[held], [r for r, ok in zip(ref, held, strict=True) if ok]
                    )[0],
                    blind_gap=gap_report(blind, gap, t, ref),
                    seeded_gap=gap_report(seeded, gap, t, ref),
                )
            save_case(outdir, arm, result)


GROUPS = (
    "avq_dev",
    "avq_sealed",
    "speech",
    "dregon_dev",
    "dregon_test",
    "agh_dev",
    "agh_test",
    "md2",
    "mixture",
    "synth_dev",
    "synth_sealed",
)


def run(args):
    if args.reuse_blind is not None and args.group != "avq_dev":
        raise ValueError(
            "--reuse-blind is valid only for the same-snapshot S1 development readback"
        )
    out = args.out / args.group
    out.mkdir(parents=True, exist_ok=True)
    if args.group.startswith("synth_"):
        synthetic_group(args.group, out)
    else:
        frozen = json.loads(args.reference.read_text())
        panel = NativePanel(frozen, args.table)
        if args.group.startswith("avq_"):
            avq_group(args.group, frozen, panel, out, args.reuse_blind)
        elif args.group in ("speech", "dregon_dev", "dregon_test", "agh_dev", "agh_test"):
            real_nulls(args.group, frozen, panel, out)
        elif args.group == "md2":
            md2_group(panel, out)
        elif args.group == "mixture":
            real_mix(frozen, panel, out)
        else:
            raise ValueError(args.group)


def summarize(args):
    """Compact per-case receipts and a flat, clock-corrected AVQ comparison table."""
    summary, avq_cases = {}, []
    metrics_to_keep = (
        "scored_units",
        "labeled_weight",
        "reference_weight",
        "emitted_weight",
        "n_by_emitted",
        "matched_weight",
        "fully_matching_weight",
        "any_extra_weight",
        "recall",
        "precision",
        "overlap",
        "full_match",
        "conditional_abs_error",
    )

    def compact(scores):
        return {key: {k: m[k] for k in metrics_to_keep if k in m} for key, m in scores.items()}

    for arm in GROUPS:
        cases = {}
        for path in sorted((args.receipts / arm).glob("*.json")):
            r = json.loads(path.read_text())
            case = path.stem
            m: dict[str, Any] = {}
            if arm.startswith("avq_"):
                s = r["summary"]
                m["full_run_coverage"] = compact(s["full_run_coverage"])
                if leaf := s.get("detected_archive_families_by_leaf"):
                    avq_cases.extend(dict(case=case, **row) for row in leaf)
                if leaf := s.get("archive_leaf_centers_all_four"):
                    m["archive_leaf_centers_all_four"] = compact(leaf[0])
                if short := s.get("archived_short_windows"):
                    m["short_window_baselines"] = compact(
                        {k: short[k] for k in ("old_raw", "old_gated")}
                    )
                    m["short_window_proposed"] = compact(short["proposed"])
                if vk := s.get("october_vk_pit_descriptive"):
                    m["october_vk_pit_descriptive"] = compact(vk[0])
                if old := s.get("frozen_release_complete_windows"):
                    m["frozen_release_complete_windows"] = compact(old["comparison"][0])
                if gap := r.get("historical_blackout"):
                    m["historical_blackout"] = {
                        key: gap[key]["interpolation_only"]
                        for key in ("blind_score", "seeded_score")
                    }
                if spot := s.get("formerly_gated_55_019"):
                    m["spot_55_019"] = spot
            elif arm.startswith("synth_"):
                m["full"] = compact(r["blind_full_scores"][0])
                if r["null_count"]:
                    m["null_count"] = r["null_count"]
                if "close_pair_strict" in r:
                    m["close_pair_strict"] = r["close_pair_strict"]
                if "blind_gap" in r:
                    m["gap"] = {
                        k: {
                            "scores": compact(r[k]["scores"]),
                            "interpolation": r[k]["interpolation_only"],
                        }
                        for k in ("blind_gap", "seeded_gap")
                    }
            elif arm == "md2":
                for key in ("independent_blind_full", "withheld_blind", "withheld_seeded"):
                    m[key] = compact(r[key] if key == "independent_blind_full" else r[key][0])
                m["number_of_anchors"] = len(r["anchors"])
            elif arm == "mixture":
                m["independent_isolated_acoustic_rate_comparison"] = compact(r["scores"][0])
            else:
                m = r["summary"]
            if scale := r.get("scale_calibration"):
                m["scale_calibration"] = {
                    k: scale[k]
                    for k in (
                        "anchor_source",
                        "identity_no_prior",
                        "track_decisions",
                        "coincident_cells_removed",
                    )
                }
                m["scale_calibration"]["anchor_count"] = len(scale["anchors"])
                m["scale_calibration"].update(
                    blind=compact(scale["blind"][0]), conditional=compact(scale["conditional"][0])
                )
            cases[case] = m
        if cases:
            summary[arm] = cases
    if not summary:
        raise ValueError("no benchmark receipts found")
    write_json(args.receipts / "summary.json", summary)
    if avq_cases:
        fields = ("case", "time_s", "output_knot_s", "reference", "prediction", "provenance")
        with (args.receipts / "avq_leaf_agreement.tsv").open("w", newline="") as f:
            out = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
            out.writeheader()
            for row in avq_cases:
                out.writerow(
                    {
                        k: json.dumps(v) if isinstance(v, (list, type(None))) else v
                        for k, v in row.items()
                    }
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmd = parser.add_subparsers(dest="mode", required=True)
    f = cmd.add_parser("freeze")
    f.add_argument("--archives", type=Path, default=ROOT / "results/static_rig")
    f.add_argument("--qc", type=Path, required=True)
    f.add_argument("--vk-track", type=Path, action="append", required=True)
    f.add_argument("--out", type=Path, required=True)
    f.set_defaults(action=freeze)
    r = cmd.add_parser("run")
    r.add_argument("--reference", type=Path, required=True)
    r.add_argument("--table", type=Path, required=True)
    r.add_argument("--group", choices=GROUPS, required=True)
    r.add_argument(
        "--reuse-blind", type=Path, help="same-snapshot blind development receipt directory"
    )
    r.add_argument("--out", type=Path, required=True)
    r.set_defaults(action=run)
    s = cmd.add_parser("summarize")
    s.add_argument("--receipts", type=Path, required=True)
    s.set_defaults(action=summarize)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
