"""Audit a published November frames snapshot against its pinned raw parent, one flight at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from data_processing.derivations import PARENTS
from data_processing.sources.motor_dat import ROTORS
from data_processing.sources.new_motor_data import read_aligned_motor
from data_processing.sources.november_frames import group_to_table
from data_processing.streams import iter_published_shard, open_repository, resolve_source

NAME = "new_motor_data-frames"


def verify(version: str) -> dict:
    repo = open_repository()
    manifest = repo.manifest(NAME, version)
    raw = resolve_source(PARENTS["new_motor_data"])
    files = {p.stem: p for p in raw.glob("*/*.dat")}
    if manifest.meta.get("layout") != "tdframe-v1" or manifest.num_samples != len(files):
        raise ValueError("manifest layout or recording count differs from raw parent")
    results = {}
    for i in range(len(manifest.shards)):
        for key, frame in iter_published_shard(NAME, i, manifest.version, repo=repo):
            if key in results or key not in files:
                raise ValueError(f"duplicate or unknown recording: {key}")
            dat = files[key]
            with dat.open("rb") as f:
                digest = hashlib.file_digest(f, "sha256").hexdigest()
            original = np.asarray(frame["dat_bytes"].data)
            assert original.dtype == np.uint8 and original.shape == (dat.stat().st_size,)
            assert hashlib.sha256(original).hexdigest() == digest, f"{key}: DAT bytes differ"
            log = read_aligned_motor(dat)
            groups = set(log) - {"DatCon"}
            assert set(frame["dat"]) == groups, f"{key}: native groups differ"
            packets = 0
            for name in sorted(groups):
                ref = log[name]
                restored = group_to_table(frame, name)
                pd.testing.assert_frame_equal(restored, ref, check_exact=True, obj=f"{key}/{name}")
                packets += len(ref)
                if name == "Motor":
                    ref = ref.sort_values(["tick", "offset"])
                    np.testing.assert_array_equal(
                        frame["rps"].data, ref[list(ROTORS)].to_numpy().T / 60.0
                    )
                    np.testing.assert_allclose(
                        frame["rps"].timestamps, ref["audio_s"], atol=1e-9, rtol=0
                    )
                del ref, restored
            (wav,) = dat.parent.glob("*.wav")
            audio = frame["audio"]
            with sf.SoundFile(wav) as f:
                assert audio.shape == (f.channels, f.frames) and audio.tindex.sr == f.samplerate
                assert audio.t_start == 0
                pos = 0
                for block in f.blocks(blocksize=262144, dtype="float32", always_2d=True):
                    np.testing.assert_array_equal(audio.data[:, pos : pos + len(block)], block.T)
                    pos += len(block)
            results[key] = {
                "native_groups": len(groups),
                "packets": packets,
                "audio_samples": pos,
                "dat_bytes": original.size,
                "dat_sha256": digest,
            }
            print(key, results[key], flush=True)
            del frame, log, original, audio
    if set(results) != set(files):
        raise ValueError("recording set differs from raw parent")
    return {
        "dataset": NAME,
        "version": manifest.version,
        "parent": PARENTS["new_motor_data"],
        "flights": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="published full version or unambiguous prefix")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evidence = verify(args.version)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2) + "\n")
