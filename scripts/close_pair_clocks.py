"""Check shared broadband timing in the 56 drone-only DroneAudioSet groups.

A failed check is NOT evidence of a separate physical event. Do not fuse phase
across devices on the strength of filenames, matching duration, or tonal beats.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import numpy as np
from close_pair_inventory import DAS
from scipy.fft import irfft, next_fast_len

from data_processing import recording_annotations as ra
from data_processing.recording_annotations.scan import (
    RemoteRanges,
    ShardBytes,
    read_view,
    scan_sample,
)
from data_processing.streams import open_repository


def lag(x, y, fs):
    """GCC-PHAT on disjoint broadband bands, suppressing the strongest tonal bins."""
    n = cast(int, next_fast_len(2 * len(x)))
    w = np.hanning(len(x))
    xx, yy = np.fft.rfft(x * w, n), np.fft.rfft(y * w, n)
    z = xx.conj() * yy
    freq = np.fft.rfftfreq(n, 1 / fs)
    limit = int(0.45 * fs)
    delays = []
    for lo, hi in ((400, 1700), (2100, 5400)):
        keep = (freq >= lo) & (freq < hi)
        amp = abs(z)
        keep &= (amp > 0) & (amp < np.quantile(amp[keep], 0.85))
        weight = np.zeros_like(z)
        weight[keep] = z[keep] / amp[keep]
        cc = irfft(weight, n)
        cc = abs(np.r_[cc[-limit:], cc[: limit + 1]])
        j = int(np.argmax(cc))
        v = cc[j]
        cc[max(0, j - 80) : j + 81] = 0
        ratio = float(v / max(cc.max(), 1e-12))
        delays.append((j - limit, ratio))
    return delays


def assess(sections, fs, pair):
    measures = [lag(x[pair[0]], x[pair[1]], fs) for _, x in sections]
    times = np.array([t for t, _ in sections])
    # Endpoints and middle calibrate an affine sample clock independently per band.
    # Odd-index observations test it; no optimizing offsets on those observations.
    yy = np.array(measures)[:, :, 0]
    residuals, slopes = [], []
    for band in yy.T:
        slope, intercept = np.polyfit(times[::2], band[::2], 1)
        residuals.extend(np.abs(band[1::2] - (slope * times[1::2] + intercept)))
        slopes.append(float(slope * 1e6 / fs))
    unique = bool(np.all(np.array(measures)[:, :, 1] >= 1.5))
    spread = float(np.ptp(yy[:, 0] - yy[:, 1]))
    return {
        "max_holdout_err_samples": float(max(residuals)),
        "slopes_ppm": slopes,
        "all_peaks_unique": unique,
        "interband_lag_range_samples": spread,
        "screen_pass": bool(unique and max(residuals) <= 0.5 and spread <= 2),
        "band_lag_samples_and_peak_ratio": measures,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    repo = open_repository()
    rel = ra.open_release(version=args.release, repo=repo)
    idx = rel.index("DroneAudioSet")
    shards = idx.parent_shards()
    groups = defaultdict(dict)
    for key in idx.keys({"sound_content": "drone_only", "provenance_kind": "recording"}):
        m = DAS.fullmatch(key)
        assert m is not None
        groups[f"{m[1]}__{m[3]}"][m[2]] = (key, idx.locate(key))
    assert len(groups) == 56 and all(len(g) == 3 for g in groups.values())

    def run(item):
        scene, views = item
        remote = RemoteRanges.from_repository(repo)
        refs = {}
        for name, (key, (loc, info)) in views.items():
            digest, size, _ = shards[loc.shard]
            shard = ShardBytes(digest, size, path=None, remote=remote)
            fields = {k: (loc.offset + off, n) for k, off, n in loc.fields}
            sc = scan_sample(
                shard,
                loc.shard,
                key,
                fields,
                dataset=idx.dataset,
                version=idx.version,
                layout=info["layout"],
            )
            refs[name] = shard, sc.series["audio"]
        (fs,) = {ref.rate_hz for _, ref in refs.values()}
        duration = min(ref.n_time / fs for _, ref in refs.values())
        sections = []
        for t in np.linspace(1, duration - 1, 5):
            data = {}
            for name, (shard, ref) in refs.items():
                chans = [0, 1] if name.startswith("mic3") else [0]
                a, _ = read_view(
                    shard, ref, ref.t_start_s + t - 0.875, ref.t_start_s + t + 0.875, chans
                )
                data[name] = a[0]
                if len(chans) == 2:
                    data["up_ch1"] = a[1]
            sections.append((float(t), data))
        pairs = list(itertools.combinations(sorted(views), 2)) + [("mic3_8array-up", "up_ch1")]
        return {
            "scene": scene,
            "times_s": [t for t, _ in sections],
            "bytes": remote.n_bytes,
            "requests": remote.n_requests,
            "pairs": {"/".join(pair): assess(sections, fs, pair) for pair in pairs},
        }

    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(run, sorted(groups.items())))
    summary = {k: sum(row["pairs"][k]["screen_pass"] for row in rows) for k in rows[0]["pairs"]}
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(
        json.dumps({"release": rel.version, "pass_counts": summary, "rows": rows}, indent=1)
    )
    print(
        "groups", len(rows), "pass", summary, "bytes", sum(row["bytes"] for row in rows), flush=True
    )


if __name__ == "__main__":
    main()
