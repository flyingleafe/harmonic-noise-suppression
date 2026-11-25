"""Prepare measured stereo and four native rotor tracks for November per-recording fits.

Windows are frozen on the calibrated WAV clock before fitting. FLY013's two listening
examples lie in its first motor run; calibration uses later runs only. The two flights
use settled, in-air intervals. This is within-recording prediction, not an unseen-rig test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from data_processing.noise_model.resample import decimate_audio
from data_processing.sources.motor_dat import ROTORS
from data_processing.streams import iter_published_shard

DATASET = "new_motor_data-frames"
VERSION = "e766b6a9306b6687ebcc8e3a08fcb54a225b488eba962e6a9cbac15174ab4e0b"
SR = 16_000
AIR_FIT = list(range(80, 781, 100))
WINDOWS = {
    "FLY013": ([202, 246, 270, 281, 350, 438, 478, 550], [74, 137]),
    "FLY018": (AIR_FIT, [319, 621]),
    "FLY022": (AIR_FIT, [310, 592]),
}


def prepare(frame, rid: str, out: Path) -> None:
    clip_dir = out / "inputs" / rid
    dest = out / rid
    clip_dir.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    native_sr = int(frame["audio"].tindex.sr)
    if native_sr != 48000 or frame["audio"].t_start != 0 or frame["audio"].shape[0] != 2:
        raise ValueError(f"unexpected ZOOM audio format for {rid}")
    audio = np.asarray(frame["audio"].data)
    rps = np.asarray(frame["rps"].data)
    t = np.asarray(frame["rps"].timestamps)
    state = frame["dat"]["aircraft_condition"]["rows"]
    fit, held = WINDOWS[rid]
    meta = {
        "dataset": DATASET,
        "version": VERSION,
        "recording": rid,
        "regime": "bare-motor ground running" if rid == "FLY013" else "airborne cruise",
        "native_sr": native_sr,
        "model_sr": SR,
        "rotor_order": list(ROTORS),
        "fit_duration_s": 4,
        "heldout_duration_s": 8,
        "fit_start_s": fit,
        "heldout_start_s": held,
        "windows": [],
    }
    for a in fit:
        for b in held:
            if not (a + 4 + 2 <= b or b + 8 + 2 <= a):
                raise ValueError(f"fit overlaps held-out support or its guard: {a}, {b}")
    for kind, starts, sec in (("fit", fit, 4), ("holdout", held, 8)):
        waveforms, carriers = [], []
        for idx, a in enumerate(starts, 1):
            i, j = a * native_sr, (a + sec) * native_sr
            # 60 ms of surrounding WAV on either side prevents the resampler's edge
            # padding from leaking into the displayed reference (polyphase filter is shorter).
            pad = 3 * 960
            sample = decimate_audio(audio[:, i - pad : j + pad], native_sr, SR)[:, 960:-960]
            if sample.shape != (2, sec * SR):
                raise ValueError(f"{rid}: recording does not cover {a}:{a + sec}")
            trace = np.arange(sec * SR) / SR + a
            lo = np.searchsorted(t, trace[0], side="right") - 1
            hi = np.searchsorted(t, trace[-1], side="left") + 1
            if lo < 0 or hi > len(t) or np.diff(t[lo:hi]).max() > 0.20:
                raise ValueError(f"{rid}: missing motor data in {a}:{a + sec}")
            speeds = np.stack([np.interp(trace, t[lo:hi], row[lo:hi]) for row in rps]).astype(
                np.float32
            )
            fs = np.unique(
                state.data["flight_state"][(state.timestamps >= a) & (state.timestamps < a + sec)]
            )
            if fs.tolist() != [1 if rid == "FLY013" else 2] or speeds.min() <= 0:
                raise ValueError(f"{rid}: not in requested running state at {a}:{a + sec} ({fs})")
            waveforms.append(sample)
            carriers.append(speeds)
            meta["windows"].append(
                {
                    "split": kind,
                    "start_s": a,
                    "end_s": a + sec,
                    "flight_state": int(fs[0]),
                    "min_rps": float(speeds.min()),
                    "max_rps": float(speeds.max()),
                    "mean_rotor_rps": speeds.mean(axis=1).tolist(),
                    "max_packet_gap_s": float(np.diff(t[lo:hi]).max()),
                }
            )
            if kind == "holdout":
                # Float32 stores every PCM16 input exactly. The report's A/B uses the
                # 16 kHz reference; this separate original makes the lost band audible.
                sf.write(
                    dest / f"native_reference_{idx}.wav",
                    audio[:, i:j].T,
                    native_sr,
                    subtype="FLOAT",
                )
        np.savez(
            clip_dir / f"{kind}.npz",
            audio=np.stack(waveforms),
            rps=np.stack(carriers),
            sr=np.int64(SR),
            start_s=np.array(starts, dtype=float),
        )
    (clip_dir / "selection.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(rid, meta["regime"], f"{len(fit)} calibration + {len(held)} held-out clips", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/november_cruise_generator"))
    parser.add_argument(
        "--recording", choices=WINDOWS, help="prepare just this flight's pinned shard"
    )
    args = parser.parse_args()
    root = args.out
    wanted = {args.recording} if args.recording else set(WINDOWS)
    seen = set()
    for shard, key in enumerate(sorted(WINDOWS)):
        if key not in wanted:
            continue
        for rid, frame in iter_published_shard(DATASET, shard, VERSION):
            if rid != key:
                raise ValueError(f"expected pinned shard {shard} to contain {key}, found {rid}")
            prepare(frame, rid, root)
            seen.add(rid)
            del frame
    if seen != wanted:
        raise ValueError(f"missing recordings: {wanted - seen}")
