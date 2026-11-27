"""Re-read the October AVQ track families on native audio (assisted, never blind).

The old family detections define rate neighborhoods, not the output values.
Only local uncontested spectral lines can remeasure the rate. Search and
confirmation use different order sets and microphones. Bracketed interpolation
has an explicit state. Produce a separate source product, not tachometer truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
from scipy.fft import next_fast_len, rfft
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from data_processing import recording_annotations as ra
from data_processing.streams import open_repository

BASE = "497ea2c5828a6cac19dc23ceabe3e1c4340725265ad99c06bc0218108797cd10"
TRAIN = (1, 2, 5, 8, 12, 18, 30, 42)
CHECK = (3, 4, 6, 10, 14, 16, 20, 24, 28, 36, 40)


def peaks(x, fs):
    n = x.shape[1]
    nfft = cast(int, next_fast_len(2 * n))
    hz = fs / nfft
    spectrum = cast(np.ndarray, rfft(x * np.hanning(n).astype(np.float32), nfft, axis=-1))
    power = abs(spectrum[:, : int(6500 / hz) + 1]) ** 2
    groups = []
    for v in (power[::2], power[1::2]):
        # Individual channel normalization stops near-field gain from deciding which mic votes.
        db = 10 * np.log10(np.maximum(v, np.finfo(np.float32).tiny))
        z = np.median(
            db - median_filter(db, size=(1, max(5, int(20 / hz) | 1)), mode="nearest"), axis=0
        )
        pk = find_peaks(z, height=6.0, prominence=4.0)[0]
        pk = pk[(pk > 1) & (pk < len(z) - 2)]
        delta = np.clip(
            0.5 * (z[pk - 1] - z[pk + 1]) / np.minimum(-1e-8, z[pk - 1] - 2 * z[pk] + z[pk + 1]),
            -0.5,
            0.5,
        )
        groups.append(((pk + delta) * hz, z[pk]))
    return groups


def read_family(spectra, seeds, i, resolution):
    seed = seeds[i]
    rivals = np.delete(seeds, i)
    rivals = rivals[np.isfinite(rivals)]
    # Multiple teeth, not just the maximum at a seeded harmonic. Each rival's
    # closest order is masked inclusively; reduce the search radius near crossings.
    radius = min(1.0, min(abs(rivals - seed), default=4.0) * 0.4)
    estimates = []
    supports = []
    for (fr, db), orders in zip(spectra, (TRAIN, CHECK), strict=True):
        obs = []
        for k in orders:
            lo, hi = fr.searchsorted([k * (seed - radius), k * (seed + radius)])
            candidates = []
            for h, level in zip(fr[lo:hi], db[lo:hi], strict=True):
                if np.any(
                    np.abs(h - np.maximum(1, np.rint(h / rivals)) * rivals)
                    < max(1.5 * resolution, 0.10 * k)
                ):
                    continue
                candidates.append((float(h / k), float(level), k))
            obs.extend(candidates)
        if not obs:
            return None
        f, p, o = np.array(obs).T
        tol = np.maximum(0.045, 1.15 * resolution / o)
        score = np.zeros(len(f))
        for j in range(len(f)):
            keep = abs(f - f[j]) <= tol
            score[j] = sum(max(p[keep & (o == order)]) for order in set(o[keep])) - 3 * abs(
                f[j] - seed
            )
        best = f[score.argmax()]
        keep = abs(f - best) <= tol
        sel = np.array(
            [
                np.flatnonzero(keep & (o == k))[np.argmin(abs(f[keep & (o == k)] - best))]
                for k in sorted(set(o[keep]))
            ]
        )
        estimates.append(float(np.median(f[sel])))
        supports.append(o[sel].astype(int).tolist())
    if abs(estimates[0] - estimates[1]) > 0.22:
        return None
    if sum(map(len, supports)) < 3:
        return None
    return (
        float(np.mean(estimates)),
        max(
            0.05,
            abs(estimates[0] - estimates[1]),
            0.5 * resolution / max(supports[0] + supports[1]),
        ),
        sorted(supports[0] + supports[1]),
    )


def run_one(x, fs, t_anchor, reviewed, span, *, leaveout=False):
    ts = np.arange(0.5, x.shape[1] / fs, 1.0)
    count = len(ts)
    rates = np.full((count, 4), np.nan)
    err = rates.copy()
    tags = np.full(rates.shape, "unresolved", dtype=object)
    orders = [[[] for _ in range(4)] for _ in ts]
    seeds = rates.copy()
    rng = [0.0] * 4
    for i in range(4):
        ix = np.flatnonzero(np.isfinite(reviewed[:, i]))
        ix = ix[~np.isin(ix, ix[1::2])] if leaveout else ix
        if not len(ix):
            continue
        tt = t_anchor[ix]
        vv = reviewed[ix, i]
        seeds[:, i] = np.interp(ts, tt, vv)
        # Never silently extend a prior beyond its published measurement span.
        left = np.searchsorted(tt, ts) - 1
        right = left + 1
        hull = (ts >= span[0]) & (ts <= span[1])
        near = np.min(abs(ts[:, None] - tt), axis=1)
        inside = (left >= 0) & (right < len(tt))
        short = np.zeros(count, dtype=bool)
        short[inside] = (tt[right[inside]] - tt[left[inside]]) <= 32
        seeds[~(hull & (short | (near <= 5))), i] = np.nan
        rng[i] = float(np.ptp(vv))
    for j, tc in enumerate(ts):
        if not np.isfinite(seeds[j]).any():
            continue
        for width in (4.0, 1.5):
            start = max(0, int(round((tc - width / 2) * fs)))
            stop = min(x.shape[1], int(round((tc + width / 2) * fs)))
            if stop - start < 0.7 * fs:
                continue
            sp = peaks(x[:, start:stop], fs)
            for i in np.flatnonzero(np.isfinite(seeds[j]) & ~np.isfinite(rates[j])):
                a = read_family(sp, seeds[j], i, fs / (stop - start))
                if a:
                    rates[j, i], err[j, i], orders[j][i] = a
                    tags[j, i] = "measured" if len(orders[j][i]) >= 4 else "track_supported"
    # Reject local swaps/collisions; never count the same audible rate twice.
    for j in range(count):
        for i in range(4):
            for h in range(i + 1, 4):
                if abs(rates[j, i] - rates[j, h]) < max(0.10, err[j, i], err[j, h]):
                    bad = h if len(orders[j][i]) >= len(orders[j][h]) else i
                    rates[j, bad] = err[j, bad] = np.nan
                    tags[j, bad] = "unresolved"
                    orders[j][bad] = []
    for i in range(4):
        ii = np.flatnonzero(np.isfinite(rates[:, i]))
        for left, right in zip(ii, ii[1:]):
            if (
                right - left <= 1
                or right - left > 16
                or abs(rates[left, i] - rates[right, i]) > 1.2
                or not np.isfinite(seeds[left : right + 1, i]).all()
            ):
                continue
            for k in range(left + 1, right):
                v = rates[left, i] + (rates[right, i] - rates[left, i]) * (k - left) / (
                    right - left
                )
                if np.any(abs(np.delete(rates[k], i) - v) < 0.18):
                    continue
                rates[k, i] = v
                err[k, i] = max(
                    err[left, i], err[right, i], 0.15 + 0.035 * min(k - left, right - k)
                )
                tags[k, i] = "interpolated"
    missing = np.array(None, dtype=object)
    return {
        "times_s": ts.tolist(),
        "rates_rps": np.where(np.isfinite(rates), rates, missing).tolist(),
        "uncertainty_rps": np.where(np.isfinite(err), err, missing).tolist(),
        "track_id": np.where(np.isfinite(rates), np.arange(4)[None, :], missing).tolist(),
        "provenance": tags.tolist(),
        "order_support": orders,
        "diagnostics": {
            "assisted_by": "october_family_neighborhoods",
            "reference_span_s": span,
            "withheld_alternating_family_observations": leaveout,
            "anchor_family_ranges_rps": rng,
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--guide", choices=["joint", "vk"], default="joint")
    args = p.parse_args()
    f = json.loads(args.reference.read_text())
    r = ra.open_release(version=BASE, repo=open_repository())
    idx = r.index("AVQ", f["avq_pin"])
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    modelhash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cases = f["avq"] if args.guide == "joint" else f["avq_vk"]
    for k, old in sorted(cases.items()):
        frame = next(r.frames(idx, [k]))[1]
        a = frame["audio"]
        x = np.asarray(a.data)
        fs = float(a.tindex.sr)
        if args.guide == "joint":
            t = np.array(old["center_s"])
            y = np.array(old["reviewed"], dtype=float)
            span = old["span_s"]
        else:
            if not all(old["valid"]):
                raise ValueError(f"{k}: old VK trajectories failed the historical acceptance gate")
            times = np.array(old["times_s"])
            ix = np.r_[np.arange(0, len(times) - 1, 16), len(times) - 1]
            t = times[ix]
            y = np.array(old["rates_rps"], dtype=float)[ix]
            span = [float(t[0]), float(t[-1])]
        summary[k] = {}
        for how, withhold in [("assisted", False), ("alternating_holdout", True)]:
            tr = run_one(x, fs, t, y, span, leaveout=withhold)
            pred = np.array(tr["rates_rps"], dtype=float)
            times = np.array(tr["times_s"])
            n = np.isfinite(pred).sum(1)
            tags = np.array(tr["provenance"])
            report = {
                "samples": len(n),
                "full_direct": int(np.all(tags == "measured", axis=1).sum()),
                "full_evidence": int(
                    np.all(np.isin(tags, ["measured", "track_supported"]), axis=1).sum()
                ),
                "full_with_interp": int((n == 4).sum()),
                "finite_by_track": np.isfinite(pred).sum(0).tolist(),
            }
            if withhold:
                errors = []
                counts = 0
                for j in range(4):
                    ix = np.flatnonzero(np.isfinite(y[:, j]))[1::2]
                    counts += len(ix)
                    for h in ix:
                        z = np.argmin(abs(times - t[h]))
                        e = abs(pred[z, j] - y[h, j])
                        if np.isfinite(e):
                            errors.append(float(e))
                report["withheld_family_refs"] = {
                    "n": counts,
                    "covered": len(errors),
                    "within_0.65": sum(e <= 0.65 for e in errors),
                    "max_error_rps": max(errors, default=None),
                }
            row = {
                "dataset": "AVQ",
                "key": k,
                "recording": k,
                "parent_version": idx.version,
                "role": "audio",
                "audio_t_start_s": float(a.t_start),
                "duration_s": x.shape[1] / fs,
                "sample_rate_hz": fs,
                "expected": 4,
                "estimator_sha256": modelhash,
                "review_status": "historical_reference_assisted_not_blind",
                "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                "track": tr,
            }
            row["track"]["diagnostics"]["guide"] = (
                "reviewed_joint_leaf_families"
                if args.guide == "joint"
                else "accepted_VK_tracks_subsampled_not_per_frame_observations"
            )
            (args.out / f"{k}-{how}.json").write_text(
                json.dumps(row, allow_nan=False, separators=(",", ":"))
            )
            summary[k][how] = report
        print(k, summary[k], flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
