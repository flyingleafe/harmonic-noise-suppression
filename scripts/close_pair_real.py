#!/usr/bin/env python3
"""Score the frozen close-pair real panel; writes private receipts, not annotations."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import resample_poly

from experiments.static_rig.close_pair_data import DAS, DREGON, LEUVEN, RELEASE, TABLE_SHA256
from tracking.steady_tachometry import estimate

# Public prefix families provide the surrounding nuisance combs. The private
# fallback only supplies a disputed centre; it cannot certify another shaft.
TRAIN_FRAC = 0.48
MATCH_RPS = 0.35
PAIR_MODE_MAX_SPLIT_RPS = 1.5


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    tmp.replace(path)


def best_window(res: dict, duration: float) -> dict:
    windows = [w for w in res["windows"] if w["accepted"] and w["rates_rev_s"]]
    return min(
        windows,
        key=lambda w: (-len(w["rates_rev_s"]), abs(w["center_s"] - duration / 2)),
        default={},
    )


def facts_for(blades: int, count: int, content: str | None) -> dict:
    # For rotor-off speech the *identity* is treated blind. Do not give the
    # detector source facts that trivially turn off the negative control.
    return dict(
        blade_count=str(blades),
        n_active=count or None,
        n_rotors=count or None,
        relative_motion="fixed",
        sound_content=content or "unknown",
    )


def public_prefix(x: np.ndarray, fs: float, facts: dict) -> dict:
    result = estimate(x, fs, facts)
    return best_window(result, x.shape[-1] / fs)


def seeds_for(x: np.ndarray, fs: float, facts: dict) -> tuple[list[float], list[float], str, dict]:
    primary = {} if x.shape[-1] / fs < 1.5 else public_prefix(x, fs, facts)
    public = primary.get("rates_rev_s", [])
    if public:
        return public, public, "steady_tachometry_public_prefix", primary
    if x.shape[-1] / fs >= 1.5:
        prepared = best_window(estimate(x, fs, facts, candidate_only=True), x.shape[-1] / fs)
        if apparent := prepared.get("rates_rev_s"):
            return public, apparent, "steady_tachometry_private_prefix", primary
    # Same unconfirmed prefix proposal rule as the synthetic null. It cannot
    # supply a retained family or independently qualify a rate.
    from tracking import steady_tachometry

    combs, _ = steady_tachometry._select(
        steady_tachometry._peaks(x, fs), steady_tachometry._physics(facts), np.empty(0)
    )
    if combs:
        return public, [float(c.rate) for c in combs], "steady_tachometry_unchecked_prefix", primary
    # Even a single unresolved peak may be a doublet. Joint-spectrum fallback
    # proposes one centre only using the fit block, without asserting a shaft.
    from experiments.static_rig.joint_speeds import JointParams, WindowEvidence

    p = JointParams(
        blades=int(facts["blade_count"]),
        lo=14.0,
        hi=240.0,
        step=0.016,
        k_max=20,
        f_max=min(3000.0, 0.42 * fs),
        n_peaks=12,
        n_modes=1,
        pad=2,
    )
    candidates, strength = WindowEvidence(x, round(fs), p).peaks()
    if not len(candidates):
        return public, [], "no_prefix_seed", primary
    return public, [float(candidates[np.argmax(strength)])], "joint_speeds_private_prefix", primary


def disputes_for(public: list[float], seeds: list[float]) -> list[dict]:
    pairs = [
        dict(seed=[a, b], other_rates=[v for v in public if v not in (a, b)], replacement=True)
        for a, b in zip(public, public[1:])
        if 0.0 < b - a <= 4.25
    ]
    return [
        dict(seed=[r], other_rates=[v for v in public if v != r], replacement=False) for r in seeds
    ] + pairs


def phase_eligible(result: dict, seed: list[float], other: list[float]) -> bool:
    rates = result["rates"]
    center = float(np.mean(seed))
    radius = 2.8 if len(seed) == 1 else max(1.7, 0.5 * (seed[1] - seed[0]) + 1.05)
    return (
        result["eligible"]
        and len(rates) == 2
        and 0.0 < rates[1] - rates[0] <= 5.0
        and max(abs(v - center) for v in rates) <= radius
        and all(abs(rate - nuisance) > 0.12 for rate in rates for nuisance in other)
    )


def pair_mode_baseline(x: np.ndarray, fs: float, seeds: list[float], others: list[float]) -> dict:
    from tracking.phase_increment_tracker import pi_kalman_refine

    lo, hi = (seeds[0] - 0.3, seeds[0] + 0.3) if len(seeds) == 1 else seeds
    if (
        not 5 < lo < hi
        or hi - lo >= PAIR_MODE_MAX_SPLIT_RPS
        or any(lo - 0.2 < o < hi + 0.2 for o in others)
    ):
        return {"reason": "two_tone_not_applicable", "rates": [], "locked": 0}
    fs_int = round(fs)
    if abs(fs - fs_int) > 1e-6:
        return {"reason": "existing_tracker_requires_integer_clock", "rates": [], "locked": 0}
    ft = np.arange(0, x.shape[-1] / fs, 0.02)
    sorted_rates = sorted([lo, hi, *others])
    refined, diagnostics = pi_kalman_refine(
        x,
        np.broadcast_to(np.asarray(sorted_rates)[:, None], (len(sorted_rates), len(ft))),
        ft,
        fs_int,
        n_iter=1,
        fs_env=125,
        k_max=16,
        k_caps=(16,),
        f_max=min(0.44 * fs, 4000.0),
        joint_win_s=0.5,
        pair_mode="joint",
        threads=1,
    )
    slots = [sorted_rates.index(lo), sorted_rates.index(hi)]
    locks = [d for pair_it in diagnostics.get("pairs", ()) for d in pair_it if d["pair"] == slots]
    n = sum(d.get("n_windows_locked", 0) for d in locks)
    orders = sorted({k for d in locks if d.get("n_windows_locked", 0) for k in d.get("ks", ())})
    # The tracker reports tracks even if no pair observations were admitted;
    # never read those tracks as measured two-tone evidence.
    return dict(
        rates=sorted(map(float, refined[slots, len(ft) // 2])) if n else [],
        locked=n,
        candidate_orders=orders,
        reason="two_tone_observations" if n else "no_locked_pair_observations",
    )


def physical_views(row: dict, x: np.ndarray) -> list[tuple[str, list[int]]]:
    ds = row["dataset"]
    if ds == LEUVEN:
        # Ch17 is externally clocked; it is excluded even if broad events align.
        # Fit same-clock subsets explicitly; the shared estimator caps each
        # fit at eight channels, so never silently shadow the near mics.
        return [
            ("onboard_joint", [0, 3, 6, 9, 12, 13, 14, 15]),
            ("main_array_01_08", list(range(8))),
            ("main_array_09_12", list(range(8, 12))),
            ("under_rotor", list(range(12, 16))),
        ]
    return [("native_array", list(range(x.shape[0])))]


def frame_bounds(start: float, n: int, fs: float) -> dict:
    # Rules correspond to the shared estimator's three block lengths.
    a, b = int(TRAIN_FRAC * n), int(0.74 * n)
    return dict(
        discovery_s=[start, start + a / fs],
        gain_calibration_s=[start + a / fs, start + b / fs],
        internal_test_s=[start + b / fs, start + n / fs],
        fit_center_s=start + 0.5 * n / fs,
    )


def run_window(row: dict, win: dict, view: str, channels: list[int], audio: np.ndarray) -> dict:
    from experiments.static_rig.close_pair import fit_pair, predict_pair

    x = audio[channels]
    fs = row["sample_rate"]
    start = win["time_s"][0]
    n = x.shape[-1]
    external = min(round(3.2 * fs), round(0.4 * n)) if n >= 4 * fs else 0
    n_fit = n - external
    fit_audio = x[:, :n_fit]
    fit = frame_bounds(start, n_fit, fs)
    external_bounds = (
        [start + n_fit / fs, start + (n_fit + external // 2) / fs, start + n / fs]
        if external
        else None
    )
    release = win.get("release_window") or {}
    output = {
        k: row[k]
        for k in (
            "dataset",
            "key",
            "recording",
            "group",
            "arm",
            "split",
            "expected",
            "sample_rate",
            "blades",
        )
    }
    output.update(
        {
            k: row[k]
            for k in (
                "parent_version",
                "shard_digest",
                "role",
                "active_seconds",
                "release_coverage",
                "blade_count_evidence",
                "release_status",
                "recording_release_max",
                "sound_content",
            )
            if k in row
        }
    )
    t0 = row.get("audio_start_s", 0.0)
    local_counts = [
        {**v, "start_s": v["start_s"] + t0, "end_s": v["end_s"] + t0}
        for v in win.get("release_local_counts", ())
    ]
    output.update(
        window_kind=win["kind"],
        window_samples=win["samples"],
        window_s=win["time_s"],
        pcm_sha256=win.get("pcm_sha256"),
        fit_blocks=fit,
        external_blocks_s=external_bounds,
        channel_view=view,
        channels=channels,
        frozen_rates=release.get("rates_rev_s", []),
        frozen_rate_uncertainty=release.get("uncertainty_rev_s", []),
        frozen_window_s=[release[k] + t0 for k in ("start_s", "end_s")] if release else None,
        release_local_counts=local_counts,
        clock_policy=("no_cross_device_fusion" if row["dataset"] == DAS else "native_source_grid"),
        candidates=[],
    )
    if n_fit < 2 * fs:
        output.update(
            reason="not_enough_contiguous_pcm_for_split_test",
            seed_origin="none",
            public_prefix_rates=[],
        )
        return output
    prefix = fit_audio[:, : int(TRAIN_FRAC * n_fit)]
    speech = row["arm"] == "speech"
    facts = facts_for(
        row["blades"],
        row["expected"] if row["arm"] == "four" else 0,
        None if speech else row.get("sound_content"),
    )
    if speech:
        output["speech_label_withheld"] = True
    public, seeds, origin, prefix_window = seeds_for(prefix, fs, facts)
    output.update(
        seed_origin=origin,
        public_prefix_rates=public,
        public_prefix_uncertainty=prefix_window.get("uncertainty_rev_s", []),
        all_seeds=seeds,
    )
    candidates = output["candidates"]
    for dispute in disputes_for(public, seeds):
        seed, other = dispute["seed"], dispute["other_rates"]
        result = fit_pair(fit_audio, fs, seed, other_rates=other, blades=row["blades"])
        baseline = pair_mode_baseline(prefix, fs, seed, other)
        candidates.append(
            {
                **dispute,
                "phase": result,
                "campaign_eligible": phase_eligible(result, seed, other),
                "pair_mode_joint": baseline,
            }
        )
    eligible = [
        (c["phase"]["score"], i)
        for i, c in enumerate(candidates)
        if c["campaign_eligible"] and c["phase"]["score"] is not None
    ]
    output["winner"] = max(eligible, key=lambda p: (p[0], -p[1]))[1] if eligible else None
    scored = [c["phase"]["score"] for c in candidates if c["phase"]["score"] is not None]
    output["raw_max_score"] = max(scored, default=None)
    output["reason"] = "candidate_evaluated" if candidates else "no_prefix_seed"
    if external and output["winner"] is not None:
        chosen = candidates[output["winner"]]
        # This was not used to choose the candidate, rate, spatial channels,
        # orders, or acceptance threshold. Map the same native clock.
        pred = predict_pair(x[:, n_fit:], fs, chosen["phase"], center_offset_s=n / (2 * fs))
        chosen["external_prediction"] = pred
        other_check = public_prefix(x[:, n_fit + external // 2 :], fs, facts)
        chosen["external_public_rates"] = other_check.get("rates_rev_s", [])
        chosen["external_public_uncertainty"] = other_check.get("uncertainty_rev_s", [])
    return output


def load_stage(root: Path) -> list[dict]:
    manifests = [json.loads(p.read_text()) for p in sorted(root.glob("stage-*/manifest.json"))]
    if not manifests or any(m["release"] != RELEASE for m in manifests):
        raise ValueError("no consistent frozen staging receipts")
    if set(m["part"] for m in manifests) != set(range(manifests[0]["parts"])) or any(
        m["table_sha256"] != TABLE_SHA256 or m["parts"] != manifests[0]["parts"] for m in manifests
    ):
        raise ValueError("incomplete or inconsistent staging receipts")
    return [r for m in manifests for r in m["rows"]]


def _load(win: dict) -> np.ndarray:
    x = np.load(win["file"], allow_pickle=False)
    if hashlib.sha256(x).hexdigest() != win["pcm_sha256"]:
        raise ValueError(f"PCM receipt mismatch: {win['file']}")
    return x


def match(observed, known) -> dict:
    if len(observed) == 0 or len(known) == 0:
        return {"abs_error": [], "matches": [], "mae": None}
    distance = np.abs(np.asarray(observed)[:, None] - np.asarray(known)[None, :])
    a, b = linear_sum_assignment(distance)
    errors = distance[a, b]
    return dict(
        abs_error=errors.tolist(),
        matches=np.stack((a, b), axis=1).tolist(),
        mae=float(np.mean(errors)),
    )


def esc_compare(result: dict, esc) -> dict:
    t = result["fit_blocks"]["fit_center_s"]
    before, after = (
        result["fit_blocks"]["discovery_s"][0],
        result["fit_blocks"]["internal_test_s"][1],
    )
    raw_time, raw = esc["motor_speed_time"], esc["motor_speed"][:4] / 60.0
    aligned_time, aligned = esc["rps_time"], esc["rps"][:4]
    references = {}
    for name, stamps, values in (
        ("raw_ESC_RPM_div_60", raw_time, raw),
        ("upstream_global_scale_sensitivity", aligned_time, aligned),
    ):
        center, span = [], []
        for channel in values:
            ok = np.isfinite(channel)
            times, rates = stamps[ok], channel[ok]
            if not len(times) or t < times[0] or t > times[-1]:
                continue
            center.append(float(np.interp(t, times, rates)))
            live = rates[(times >= before) & (times < after)]
            span.append(np.percentile(live, [5, 95]).tolist() if len(live) else None)
        raw_order = sorted(range(len(center)), key=center.__getitem__)
        rates = [center[i] for i in raw_order]
        refs = dict(
            center_rates=rates,
            p05_p95=[span[i] for i in raw_order],
            matched_public_prefix=match(result["public_prefix_rates"], rates),
        )
        if (index := result.get("winner")) is not None:
            row = result["candidates"][index]
            refs["matched_phase_and_retained_prefix"] = match(
                sorted([*row["phase"]["rates"], *row["other_rates"]]), rates
            )
        references[name] = refs
    return references


def score_panel(rows: list[dict], output: Path, part: int, parts: int) -> None:
    results = []
    for row in sorted(rows, key=lambda r: (r["dataset"], r["key"])):
        uid = f"{row['dataset']}:{row['key']}"
        if int.from_bytes(hashlib.sha256(uid.encode()).digest()[:4], "big") % parts != part:
            continue
        if not row["windows"]:
            results.append(
                dict(
                    dataset=row["dataset"],
                    key=row["key"],
                    recording=row["recording"],
                    group=row["group"],
                    arm=row["arm"],
                    split=row["split"],
                    expected=row["expected"],
                    reason="no_prespecified_active_block",
                    window_kind="none",
                    candidates=[],
                    public_prefix_rates=[],
                    winner=None,
                )
            )
        for win in row["windows"]:
            audio = _load(win)
            for name, channels in physical_views(row, audio):
                result = run_window(row, win, name, channels, audio)
                if row["arm"] == "esc" and row.get("esc_stamps_are_absolute"):
                    # Do not report a comparison on unverified relative ticks.
                    with np.load(row["esc_file"], allow_pickle=False) as esc:
                        result["blind_esc_comparison"] = esc_compare(result, esc)
                results.append(result)
                print(
                    f"SCORED {uid} {win['kind']} {name} calls={len(result['candidates'])} max={result.get('raw_max_score')}",
                    flush=True,
                )
    write_json(output / f"score-{part:02d}.json", results)


def isolate(row: dict) -> tuple[dict, np.ndarray] | None:
    if not row["windows"]:
        return None
    (win,) = row["windows"]
    x, fs = _load(win), row["sample_rate"]
    if x.shape[0] < 2 or x.shape[-1] < 2.8 * fs:
        return None
    f = facts_for(row["blades"], 1, "ego_noise_only")
    h = public_prefix(x, fs, f)
    if len(h.get("rates_rev_s", [])) != 1:
        return None
    # Isolated source rates are separate-reference acoustic readings, not
    # tachometer truth. Include time variation within the source as uncertainty.
    chunks = []
    for piece in np.array_split(x, 3, axis=-1):
        reading = public_prefix(piece, fs, f)
        if reading:
            chunks.extend(reading["rates_rev_s"])
    (rate,) = h["rates_rev_s"]
    u = max(h["uncertainty_rev_s"][0], max((abs(value - rate) for value in chunks), default=0.0))
    ref = {
        k: row[k]
        for k in (
            "dataset",
            "key",
            "recording",
            "parent_version",
            "shard_digest",
            "role",
            "sample_rate",
            "blades",
            "split",
        )
    }
    ref.update(
        samples=win["samples"],
        time_s=win["time_s"],
        pcm_sha256=win["pcm_sha256"],
        isolated_rate_rps=rate,
        isolated_rate_uncertainty_rps=u,
    )
    return ref, x


def make_mixtures(rows: list[dict], output: Path) -> None:
    isolates = defaultdict(list)
    exclusions = []
    for row in sorted(rows, key=lambda r: (r["dataset"], r["key"])):
        if (row["arm"], row["split"]) != ("single", "test"):
            continue
        data = isolate(row)
        if data is None:
            exclusions.append(
                dict(
                    dataset=row["dataset"],
                    key=row["key"],
                    reason="no_multichannel_locally_identified_reference_with_enough_pcm",
                )
            )
            continue
        lineage, x = data
        isolates[row["dataset"], row["sample_rate"], row["blades"], x.shape[0]].append((lineage, x))
    results, pairs = [], []
    for (dataset, fs, blades, n_channels), sources in sorted(isolates.items()):
        # DREGON: pair equal-throttle takes from different isolated motors.
        # AGH: only same published mic count, setup and native recording rate.
        if dataset == DREGON:
            sources.sort(key=lambda src: (src[0]["key"].rsplit("_", 1)[1], src[0]["key"]))
        while sources:
            a, xa = sources.pop(0)
            eligible = [
                i
                for i, (b, _) in enumerate(sources)
                if dataset != DREGON
                or (
                    a["key"].rsplit("_", 1)[1] == b["key"].rsplit("_", 1)[1]
                    and a["key"].rsplit("_", 1)[0] != b["key"].rsplit("_", 1)[0]
                )
            ]
            if not eligible:
                exclusions.append(
                    dict(dataset=dataset, key=a["key"], reason="no_disjoint_same_setup_partner")
                )
                continue
            idx = min(
                eligible,
                key=lambda i: abs(a["isolated_rate_rps"] - sources[i][0]["isolated_rate_rps"]),
            )
            b, xb = sources.pop(idx)
            pair_id = f"{dataset}:{a['key']}+{b['key']}"
            pairs.append(pair_id)
            center = 0.5 * (a["isolated_rate_rps"] + b["isolated_rate_rps"])
            # Repeated scenarios use the same lineage; independent unit is
            # the disjoint source pair, not every separation/level scenario.
            for delta, weak_db in itertools.product((0.0, 0.20, 0.5, 1.0, 2.0, 4.0), (0.0, -12.0)):
                imposed, waveforms, lineage = [], [], []
                for src, pcm, target in ((a, xa, center - delta / 2), (b, xb, center + delta / 2)):
                    frac = Fraction(src["isolated_rate_rps"] / target).limit_denominator(1024)
                    y = resample_poly(pcm, frac.numerator, frac.denominator, axis=-1)
                    ratio = frac.denominator / frac.numerator
                    imposed.append(src["isolated_rate_rps"] * ratio)
                    waveforms.append(y)
                    lineage.append(
                        {
                            **src,
                            "resample_up": frac.numerator,
                            "resample_down": frac.denominator,
                            "imposed_rate_uncertainty_rps": src["isolated_rate_uncertainty_rps"]
                            * ratio,
                        }
                    )
                n = min(*(p.shape[-1] - 2 * round(0.15 * fs) for p in waveforms), round(8 * fs))
                if n < 2 * fs:
                    raise ValueError(f"mixture too short: {pair_id}")
                xmix = np.zeros((n_channels, n), dtype=np.float32)
                for i, y in enumerate(waveforms):
                    lo = (y.shape[-1] - n) // 2
                    y = y[:, lo : lo + n]
                    gain = 10 ** ((weak_db if i else 0.0) / 20) / np.sqrt(
                        np.mean(y**2, dtype=np.float64)
                    )
                    xmix += (gain * y).astype(np.float32)
                    lineage[i].update(
                        mix_output_begin_sample=lo,
                        mix_output_end_sample=lo + n,
                        mix_gain=float(gain),
                    )
                key = f"{pair_id}/{delta:g}rps/{weak_db:g}dB"
                row = dict(
                    dataset=dataset,
                    key=key,
                    recording=key,
                    group=pair_id,
                    split="test",
                    arm="real_mix",
                    expected=2,
                    sample_rate=fs,
                    blades=blades,
                )
                win = dict(
                    kind="mixture",
                    samples=[0, n],
                    time_s=[0.0, n / fs],
                    pcm_sha256=hashlib.sha256(xmix).hexdigest(),
                )
                result = run_window(row, win, "native_array", list(range(n_channels)), xmix)
                result.update(
                    real_mixture=dict(
                        pair_id=pair_id,
                        separation_target_rps=delta,
                        weak_db=weak_db,
                        mapped_rates=imposed,
                        source_lineage=lineage,
                    )
                )
                results.append(result)
                print(
                    f"MIXTURE {key} calls={len(result['candidates'])} max={result.get('raw_max_score')}",
                    flush=True,
                )
    write_json(
        output / "mixtures.json", dict(results=results, source_pairs=pairs, exclusions=exclusions)
    )


def cutoff_pass(score: float | None, rule: dict) -> bool:
    if score is None or rule["reject_all"]:
        return False
    return rule["accept_all_scored"] or score > rule["score_gt"]


def _nuisance_support(r: dict, chosen: dict, win: dict) -> tuple[float, float] | None:
    """Verify both retained families on exactly the untouched test suffix."""
    _, lo, hi = r["external_blocks_s"]
    fs = r["sample_rate"]
    a, b = [round((t - r["window_s"][0]) * fs) for t in (lo, hi)]
    pcm = _load(win)[r["channels"], a:b]
    w = public_prefix(pcm, fs, facts_for(r["blades"], 4, r.get("sound_content")))
    if not w:
        return None
    # A nuisance read assigned to the disputed pair's own extrapolated rates
    # cannot count as another shaft. Check the whole nuisance support span.
    anchor = np.asarray(chosen["phase"]["rates"])
    drift = np.asarray(chosen["phase"]["drift_rps_per_s"])
    t = np.asarray([w["start_s"] + lo, w["end_s"] + lo])
    displacement = anchor[:, None] + drift[:, None] * (t - r["fit_blocks"]["fit_center_s"])[None, :]
    public = []
    for hz in w["rates_rev_s"]:
        delta = displacement - hz
        if (np.abs(delta) > 0.12).all() and (delta[:, 0] * delta[:, 1] > 0.0).all():
            public.append(hz)
    matched = match(chosen["other_rates"], public)["abs_error"]
    if len(matched) != 2 or max(matched) > MATCH_RPS:
        return None
    return max(lo, t[0]), min(hi, t[1])


def summarize(data: Path, output: Path, threshold: Path, replay_path: Path | None) -> None:
    staged = load_stage(data)
    rule = json.loads(threshold.read_text())["thresholds"]["phase"]
    replay = json.loads(replay_path.read_text()) if replay_path is not None else {}
    if replay and replay["release"] != RELEASE:
        raise ValueError("native-PCM replay refers to a different release")
    absolute_esc = {
        row["key"]: row["esc_file"] for row in staged if row.get("esc_stamps_are_absolute")
    }
    absolute_esc.update(replay.get("esc_files", {}))
    if any(row["arm"] == "esc" and row["key"] not in absolute_esc for row in staged):
        raise ValueError("trusted absolute ESC stamp replay required")
    by_window = {(r["dataset"], r["key"], w["kind"]): w for r in staged for w in r["windows"]}
    staged_by_key = {(r["dataset"], r["key"]): r for r in staged}
    mixtures = json.loads((output / "mixtures.json").read_text())
    merged = [
        item
        for file in sorted(output.glob("score-*.json"))
        for item in json.loads(file.read_text())
    ]
    if {(r["dataset"], r["key"]) for r in merged} != {(r["dataset"], r["key"]) for r in staged}:
        raise ValueError("scoring receipts do not cover exactly the staged source keys")
    # The first private run still sent the frozen release's no_drone field to
    # the public peak finder. Reuse the exact independent PCM, now with the
    # label withheld from every method. This is fixed by source schema, not scores.
    for i, r in enumerate(merged):
        if (
            r["arm"] == "speech"
            and r["window_kind"] != "none"
            and not r.get("speech_label_withheld")
        ):
            ds, key, kind = r["dataset"], r["key"], r["window_kind"]
            win = by_window[ds, key, kind]
            merged[i] = run_window(
                staged_by_key[ds, key], win, r["channel_view"], r["channels"], _load(win)
            )
    rows = [*merged, *mixtures["results"]]
    totals, reasons, gains, maxima = {}, Counter(), [], defaultdict(list)
    qualified_predictions = []
    mix_strata = defaultdict(list)
    strata = defaultdict(list)
    for r in rows:
        r["cutoff_accepts"] = False
        r["independent_gain"] = False
        r["complete_gain_s"] = 0.0
        if r["arm"] == "esc":
            with np.load(absolute_esc[r["key"]], allow_pickle=False) as esc:
                r["blind_esc_comparison"] = esc_compare(r, esc)
        if (winner := r.get("winner")) is not None:
            candidate = r["candidates"][winner]
            p = candidate["phase"]
            r["cutoff_accepts"] = candidate["campaign_eligible"] and cutoff_pass(p["score"], rule)
            external = candidate.get("external_prediction") or {}
            external_score = external.get("score")
            r["independent_gain"] = (
                r["cutoff_accepts"] and external_score is not None and external_score > 0.0
            )
            r["external_score"] = external_score
            if r["cutoff_accepts"]:
                qualified_predictions.append(
                    dict(
                        dataset=r["dataset"],
                        key=r["key"],
                        window_s=r["window_s"],
                        fit_blocks=r["fit_blocks"],
                        external_blocks_s=r["external_blocks_s"],
                        arm=r["arm"],
                        view=r["channel_view"],
                        score=p["score"],
                        external_score=external_score,
                        external_reason=external.get("reason"),
                    )
                )
            rates = sorted(p["rates"] + candidate["other_rates"])
            r["resulting_unordered_rates"] = rates
            if r["arm"] == "real_mix":
                truth = r["real_mixture"]
                comparison = match(p["rates"], truth["mapped_rates"])
                u = [s["imposed_rate_uncertainty_rps"] for s in truth["source_lineage"]]
                truth_sep = abs(truth["mapped_rates"][1] - truth["mapped_rates"][0])
                s_int = p.get("separation_interval_rps")
                r["mixture_comparison"] = {
                    **comparison,
                    "within_reference_tolerance": len(comparison["abs_error"]) == 2
                    and all(
                        err <= max(MATCH_RPS, u[known])
                        for err, (_, known) in zip(
                            comparison["abs_error"], comparison["matches"], strict=True
                        )
                    ),
                    "separation_interval_contains_imposed": None
                    if s_int is None or None in s_int
                    else s_int[0] <= truth_sep <= s_int[1],
                    "separation_interval_overlaps_reference_tolerance": None
                    if s_int is None or None in s_int
                    else s_int[0] <= truth_sep + sum(u) and s_int[1] >= max(0, truth_sep - sum(u)),
                }
            if (
                r["arm"] == "four"
                and r["independent_gain"]
                and not candidate["replacement"]
                and len(candidate["other_rates"]) == 2
                and len(rates) == 4
                and len(r["frozen_rates"]) < 4
                and r.get("blade_count_evidence") == "release"
                and (
                    support := _nuisance_support(
                        r, candidate, by_window[r["dataset"], r["key"], r["window_kind"]]
                    )
                )
                is not None
            ):
                lo, hi = support
                # Count only the time concurrently supported by all inputs,
                # never intervals already resolved or unseen by the release.
                intervals = [
                    [max(lo, v["start_s"]), min(hi, v["end_s"])]
                    for v in r["release_local_counts"]
                    if v["count"] < 4 and v["start_s"] < hi and v["end_s"] > lo
                ]
                if intervals:
                    r["complete_gain_s"] = sum(b - a for a, b in intervals)
                    r["complete_gain_spans_s"] = intervals
                    gains.append(
                        {
                            k: r[k]
                            for k in (
                                "dataset",
                                "recording",
                                "key",
                                "window_s",
                                "external_blocks_s",
                                "channel_view",
                                "resulting_unordered_rates",
                                "complete_gain_s",
                                "complete_gain_spans_s",
                            )
                        }
                    )
        for c in r["candidates"]:
            reasons[c["phase"]["reason"]] += 1
        if r["arm"] == "real_mix":
            truth = r["real_mixture"]
            mix_strata[(truth["separation_target_rps"], truth["weak_db"])].append(r)
            r["mixture_public_comparison"] = match(r["public_prefix_rates"], truth["mapped_rates"])
            baseline_attempts = [c["pair_mode_joint"] for c in r["candidates"]]
            r["mixture_existing_two_tone_comparisons"] = [
                match(a["rates"], truth["mapped_rates"]) for a in baseline_attempts
            ]
            most_locked = max(baseline_attempts, key=lambda a: a["locked"], default=None)
            r["mixture_existing_two_tone_top"] = match(
                most_locked["rates"] if most_locked else [], truth["mapped_rates"]
            )
        strata[(r["arm"], r["dataset"], r["split"])].append(r)
        maxima[r["arm"], r["dataset"], r["split"], r["group"]].append(r)
    for (arm, ds, split), subset in sorted(strata.items()):
        key = f"{arm}/{ds}/{split}"
        totals[key] = dict(
            windows=len(subset),
            source_keys=len({r["key"] for r in subset}),
            grouped_takes=len({r["recording"] for r in subset}),
            eligible=sum(r.get("winner") is not None for r in subset),
            accepted=sum(r["cutoff_accepts"] for r in subset),
            heldout_gain=sum(r["independent_gain"] for r in subset),
            heldout_nonpositive=sum(
                r["cutoff_accepts"]
                and r.get("external_score") is not None
                and r["external_score"] <= 0.0
                for r in subset
            ),
            heldout_unavailable=sum(
                r["cutoff_accepts"] and r.get("external_score") is None for r in subset
            ),
            full_anchor_windows=sum(r["window_kind"] == "full_anchor" for r in subset),
            three_release_windows=sum(len(r.get("frozen_rates", ())) == 3 for r in subset),
            calls=sum(len(r["candidates"]) for r in subset),
        )
    for (arm, ds, split, group), subset in sorted(maxima.items()):
        if arm not in ("single", "speech"):
            continue
        scores = [
            r["candidates"][r["winner"]]["phase"]["score"]
            for r in subset
            if r.get("winner") is not None
        ]
        raw_scores = [r["raw_max_score"] for r in subset if r.get("raw_max_score") is not None]
        totals[f"{arm}/{ds}/{split}"].setdefault("null_search_max_by_take", {})[group] = dict(
            max_eligible=max(scores, default=None),
            max_raw=max(raw_scores, default=None),
            false_split=any(r["cutoff_accepts"] for r in subset),
        )
    # Complete gain is measured only on independent local test spans. Views of
    # one DAS segment cannot be summed; select one confirmed view conservatively.
    completeness = defaultdict(list)
    baseline_by_key = {}
    for r in staged:
        if r["arm"] != "four":
            continue
        key = f"{r['dataset']}/{r['key']}"
        baseline_by_key[key] = r["release_coverage"]
        completeness[r["dataset"], r["recording"]].append((key, r["recording_release_max"]))
    per_key = defaultdict(list)
    for gain in gains:
        per_key[f"{gain['dataset']}/{gain['key']}"].append(gain["complete_gain_s"])
    gain_fraction_per_key = {
        key: max(per_key[key], default=0.0) / baseline["active_s"] if baseline["active_s"] else None
        for key, baseline in baseline_by_key.items()
    }
    by_group = []
    for (dataset, recording), source in sorted(completeness.items()):
        key, baseline_count = max(
            source,
            key=lambda pair: (
                max(per_key[pair[0]], default=0.0),
                baseline_by_key[pair[0]]["active_s"] or -1.0,
                pair[0],
            ),
        )
        by_group.append(
            dict(
                dataset=dataset,
                recording=recording,
                reported_clock=key,
                frozen_max=baseline_count,
                gained_s=max(per_key[key], default=0.0),
                active_s=baseline_by_key[key]["active_s"],
            )
        )
    active_total = sum(g["active_s"] or 0.0 for g in by_group)
    gain_total = sum(g["gained_s"] for g in by_group)
    mix_report = []
    for (delta, weak_db), group in sorted(mix_strata.items()):
        old = [r["mixture_public_comparison"]["abs_error"] for r in group]
        new = [r.get("mixture_comparison", {}) for r in group]
        existing = [r["mixture_existing_two_tone_top"]["abs_error"] for r in group]
        mix_report.append(
            dict(
                separation_target_rps=delta,
                weak_db=weak_db,
                trials=len(group),
                source_pairs=len({r["group"] for r in group}),
                baseline_matched=sum(len(err) == 2 and max(err) <= MATCH_RPS for err in old),
                existing_two_tone_matched=sum(
                    len(err) == 2 and max(err) <= MATCH_RPS for err in existing
                ),
                phase_passes=sum(r["cutoff_accepts"] for r in group),
                phase_match_with_reference_uncertainty=sum(
                    r["cutoff_accepts"] and p.get("within_reference_tolerance", False)
                    for r, p in zip(group, new, strict=True)
                ),
                phase_match_with_fixed_tolerance=sum(
                    r["cutoff_accepts"]
                    and len(p.get("abs_error", ())) == 2
                    and max(p["abs_error"]) <= MATCH_RPS
                    for r, p in zip(group, new, strict=True)
                ),
                independent_predictions=sum(r["independent_gain"] for r in group),
                interval_contains_imposed=sum(
                    p.get("separation_interval_contains_imposed") is True for p in new
                ),
                interval_evaluable=sum(
                    p.get("separation_interval_contains_imposed") is not None for p in new
                ),
            )
        )
    report = dict(
        release=RELEASE,
        threshold=rule,
        threshold_sha256=hashlib.sha256(threshold.read_bytes()).hexdigest(),
        strata=totals,
        phase_reasons=dict(reasons),
        qualified_local_gain_spans=gains,
        threshold_passed_candidates=qualified_predictions,
        additional_four_of_four_grouped_takes=sum(
            g["gained_s"] > 0.0 and g["frozen_max"] < 4 for g in by_group
        ),
        complete_seconds_gained_conservative=gain_total,
        known_assessed_active_seconds_grouped=active_total,
        groups_with_known_active_seconds=sum(g["active_s"] is not None for g in by_group),
        complete_gain_fraction_of_known_assessed_active_seconds=gain_total / active_total
        if active_total
        else None,
        complete_gain_and_coverage_by_group=by_group,
        baseline_active_and_accepted_seconds_by_source_key=baseline_by_key,
        complete_gain_fraction_by_source_key=gain_fraction_per_key,
        mixture_source_pairs=mixtures["source_pairs"],
        mixture_exclusions=mixtures["exclusions"],
        mixture_strata=mix_report,
        DAS_clock_policy="array-only; no cross-recorder time or phase correspondence claimed",
        esc_clock_policy="raw independent ESC values on upstream aligned, absolute stamps; compare calibrated series separately, not blind fitter inputs",
        esc_comparisons=[
            dict(
                key=r["key"],
                window_s=r["window_s"],
                passes_threshold=r["cutoff_accepts"],
                references=r["blind_esc_comparison"],
            )
            for r in rows
            if r["arm"] == "esc"
        ],
        native_pcm_replay=replay.get("checks"),
        range_bytes=sum(r["range_bytes"] for r in staged),
        range_requests=sum(r["range_requests"] for r in staged),
    )
    write_json(output / "report.json", report)
    write_json(output / "all-windows.json", rows)
    omit = (
        "baseline_active_and_accepted_seconds_by_source_key",
        "complete_gain_fraction_by_source_key",
        "mixture_exclusions",
        "threshold_passed_candidates",
        "complete_gain_and_coverage_by_group",
    )
    print(json.dumps({k: v for k, v in report.items() if k not in omit}, indent=2), flush=True)


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("command", choices=("stage", "verify-staged", "score", "mix", "report"))
    cli.add_argument("--data", type=Path, required=True)
    cli.add_argument("--output", type=Path)
    cli.add_argument("--table", type=Path, help="frozen condition TSV, required for stage")
    cli.add_argument("--part", type=int, default=0)
    cli.add_argument("--parts", type=int, default=1)
    cli.add_argument("--threshold", type=Path)
    cli.add_argument("--native-replay", type=Path)
    args = cli.parse_args()
    if args.command != "stage" and args.output is None:
        cli.error(f"{args.command} requires --output")
    if args.command == "stage":
        if args.table is None:
            cli.error("stage requires the frozen constant-recording table")
        from experiments.static_rig.close_pair_data import stage

        stage(args.table, args.data, args.part, args.parts)
    elif args.command == "verify-staged":
        from experiments.static_rig.close_pair_data import verify_staged

        verify_staged(args.data, args.output)
    elif args.command == "report":
        if args.threshold is None:
            cli.error("report requires the frozen controls threshold JSON")
        summarize(args.data, args.output, args.threshold, args.native_replay)
    else:
        rows = load_stage(args.data)
        if args.command == "mix":
            make_mixtures(rows, args.output)
        else:
            if not 0 <= args.part < args.parts:
                cli.error("invalid shard index")
            score_panel(rows, args.output, args.part, args.parts)


if __name__ == "__main__":
    main()
