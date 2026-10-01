"""Static rig profiles: rotor speeds + per-harmonic, per-mic line powers of every
drone-noise-only stationary recording we hold (campaign doc
``docs/experiments/static-rig-profiles.md``).

Census (drone fixed, rotors on, no other source):

- ``DREGON-frames`` ``motor_Motor{1-4}_{50..90}`` (1 rotor) + ``motor_allMotors_70``;
- ``SPCUP19-frames`` recordings annotated ``bench`` with ``external_source: none``
  (AGH 4-rotor array takes + single-rotor takes, KU_Leuven, Maverick 4);
- ``AVQ`` sequences annotated ``bench`` + ``ego_noise_only``;
- DroneAudioSet ``drone-only`` (HuggingFace parquet; the pinned dload copy is
  not key-indexed and 88 GiB).

Two stages, both meant for ``uni-cpu`` (the audio is large)::

    python scripts/static_rig.py fetch --work /tmp/static_rig   # audio -> npy + units.json
    python scripts/static_rig.py run --work /tmp/static_rig --out results/static_rig/speeds --jobs 6

``run`` writes one ``raw/<uid>.json`` per recording (gridrun) and a compact
``spec/<uid>.npz`` (channel-mean prominence up to 4 kHz, float16) for figures.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from utils.gridrun import Unit, run_grid  # noqa: E402

DAS_URL = (
    "https://huggingface.co/datasets/ahlab-drone-project/DroneAudioSet/"
    "resolve/main/drone-only/train_%03d-00000-of-00001.parquet"
)
DAS_SHARDS = 28
#: DREGON-frames shards holding the motor bench (``plots.explore`` index).
DREGON_SHARDS = range(10, 17)
#: AVQ shards holding S1_seq1..S2_seq1 (recipe 2 pin).
AVQ_SHARDS = range(0, 4)


def _uid(dataset: str, key: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", f"{dataset}__{key}")


def _meta_get(frame: Any, path: str, default: Any = None) -> Any:
    node = frame["meta"]
    for part in path.split("."):
        if part not in node:
            return default
        node = node[part]
    return node


def _audio_ct(frame: Any) -> tuple[Any, int]:
    import numpy as np

    audio = frame["audio"]
    data = np.asarray(audio.data, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    return data, int(round(float(audio.tindex.sr)))


def _save(work: Path, uid: str, x: Any, fs: int, meta: dict[str, Any], units: list) -> None:
    import numpy as np

    path = work / "audio" / f"{uid}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(x, dtype=np.float32))
    units.append({"uid": uid, "audio": str(path), "fs": fs, "n_ch": int(x.shape[0]), **meta})
    print(f"  {uid}: {x.shape[0]} ch, {x.shape[1] / fs:.1f} s @ {fs}", flush=True)


def fetch_dregon(work: Path, units: list) -> None:
    from data_processing.streams import iter_published_shard

    for shard in DREGON_SHARDS:
        for key, frame in iter_published_shard("DREGON-frames", shard):
            if not key.startswith("motor_"):
                continue
            x, fs = _audio_ct(frame)
            m = re.match(r"motor_(Motor(\d)|allMotors)_(\d+)", key)
            assert m, key
            single = m.group(2) is not None
            meta = {
                "dataset": "DREGON-frames",
                "key": key,
                "rig": "dregon",
                "n_rotors": 1 if single else 4,
                "motor": int(m.group(2)) if single else None,
                "throttle": f"{m.group(3)} %",
            }
            _save(work, _uid("DREGON-frames", key), x, fs, meta, units)


def fetch_spcup(work: Path, units: list) -> None:
    from data_processing.sources.spcup19 import load_annotations
    from data_processing.streams import iter_published_frames

    wanted = {}
    for team, doc in load_annotations().items():
        for rec in doc["recordings"]:
            if (
                rec["condition"] == "bench"
                and rec["external_source"] == "none"
                and rec["contains_rotor_noise"]
            ):
                wanted[rec["key"]] = (team, rec)
    for frame in iter_published_frames("SPCUP19-frames"):
        key = str(_meta_get(frame, "recording_id", ""))
        if key not in wanted:
            continue
        team, rec = wanted.pop(key)
        x, fs = _audio_ct(frame)
        meta = {
            "dataset": "SPCUP19-frames",
            "key": key,
            "rig": f"spcup_{team.lower()}",
            "n_rotors": int(rec["n_active_rotors"]),
            "throttle": rec.get("throttle_or_speed"),
            "geometry": rec.get("geometry"),
        }
        _save(work, _uid("SPCUP19-frames", key), x, fs, meta, units)
    if wanted:
        raise SystemExit(f"SPCUP19-frames is missing {sorted(wanted)}")


def fetch_avq(work: Path, units: list) -> None:
    from data_processing.sources.avq import load_annotations
    from data_processing.streams import iter_published_shard

    by_key = load_annotations()["by_key"]
    wanted = {
        k
        for k, r in by_key.items()
        if r["condition"] == "bench" and r["content"] == "ego_noise_only"
    }
    for shard in AVQ_SHARDS:
        for key, frame in iter_published_shard("AVQ", shard):
            if key not in wanted:
                continue
            wanted.discard(key)
            x, fs = _audio_ct(frame)
            meta = {
                "dataset": "AVQ",
                "key": key,
                "rig": "avq",
                "n_rotors": int(by_key[key]["n_active_rotors"]),
                "throttle": by_key[key]["throttle_or_speed"],
            }
            _save(work, _uid("AVQ", key), x, fs, meta, units)
    if wanted:
        raise SystemExit(f"AVQ shards {list(AVQ_SHARDS)} miss {sorted(wanted)}")


def fetch_daset(work: Path, units: list) -> None:
    import pyarrow.parquet as pq
    import requests

    from data_processing.sources.droneaudio import _arrow_list_scalar_to_ct

    raw = work / "daset_parquet"
    raw.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    for i in range(1, DAS_SHARDS + 1):
        path = raw / f"train_{i:03d}.parquet"
        if path.exists():
            continue
        with session.get(DAS_URL % i, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            tmp = path.with_suffix(".part")
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(1 << 22):
                    fh.write(chunk)
            tmp.rename(path)
    for path in sorted(raw.glob("*.parquet")):
        pf = pq.ParquetFile(str(path))
        for batch in pf.iter_batches(batch_size=2):
            col = batch.column("audio")
            arrays = col.field("array")
            srs = col.field("sampling_rate").to_pylist()
            fps = batch.column("file_path").to_pylist()
            for i in range(batch.num_rows):
                fp = str(fps[i])
                low = fp.lower()
                drone = "drone2" if "drone2" in low else "drone1"
                mic = (
                    "M_down"
                    if "8array-down" in low
                    else ("M_up" if "8array-up" in low else "M_center")
                )
                meta = {
                    "dataset": "DroneAudioSet",
                    "key": fp,
                    "rig": f"daset_{drone}",
                    "n_rotors": 4,
                    "throttle": "high" if "throttle-high" in low else "low",
                    "mic_dist": "25cm" if "25cm" in low else "50cm",
                    "mic": mic,
                    "take": Path(fp).stem.split("-")[-1],
                }
                x = _arrow_list_scalar_to_ct(arrays[i])
                _save(work, _uid("DroneAudioSet", fp), x, int(srs[i]), meta, units)


FETCHERS = {"dregon": fetch_dregon, "spcup": fetch_spcup, "avq": fetch_avq, "daset": fetch_daset}


# ─── per-recording worker ────────────────────────────────────────────────────


def _jsonable(v: Any) -> Any:
    import numpy as np

    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    if isinstance(v, float) and v != v:
        return None
    return v


def worker(unit: Unit) -> dict[str, Any]:
    import numpy as np

    from experiments.static_rig import spectra as S

    prm = dict(unit.params)
    p = S.Params()
    x = np.load(prm["audio"], mmap_mode="r")
    fs = int(prm["fs"])
    a, b = S.motor_on_span(np.asarray(x), fs, p)
    xs = np.asarray(x[:, a:b], dtype=np.float64)
    sp = S.spectrum(xs, fs, p)
    n_rot = int(prm["n_rotors"])
    modes = S.rotor_speeds(sp, n_rot, p)
    comb = modes.pop("comb")
    speeds = [r["speed"] for r in modes["rotors"]]
    halves = []
    mid = xs.shape[1] // 2
    for part in (xs[:, :mid], xs[:, mid:]):
        mh = S.speed_modes(S.spectrum(part, fs, p), comb["f_bar"], n_rot, p)
        hs = [r["speed"] for r in mh["rotors"]]
        halves.append([min(hs, key=lambda h, s=s: abs(h - s)) if hs else None for s in speeds])
    lines = S.line_powers(sp, speeds, p)
    spec_dir = Path(prm["out"]) / "spec"
    spec_dir.mkdir(parents=True, exist_ok=True)
    keep = sp.f <= 4000.0
    np.savez_compressed(
        spec_dir / f"{unit.uid}.npz",
        f0=sp.f[0],
        df=sp.df,
        prom_mean=sp.prom_mean[keep].astype(np.float16),
    )
    return {
        **{k: v for k, v in prm.items() if k not in ("audio", "out")},
        "span_s": [a / fs, b / fs],
        "df": sp.df,
        "comb": comb,
        "modes": _jsonable(modes),
        "halves": halves,
        "lines": [
            {
                "speed": ln["speed"],
                "power_db": _jsonable(ln["power_db"]),
                "snr_db": _jsonable(ln["snr_db"]),
                "merged": ln["merged"],
            }
            for ln in lines
        ],
        "params": p.__dict__,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One line per recording: rig, rotor count, distinct speeds found."""
    return {
        "n": len(rows),
        "recordings": {
            f"{r['dataset']}:{r['key']}": {
                "rig": r["rig"],
                "n_rotors": r["n_rotors"],
                "f_bar": round(r["comb"]["f_bar"], 3),
                "speeds": [round(x["speed"], 3) for x in r["modes"]["rotors"]],
                "halves": r["halves"],
            }
            for r in rows
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--work", type=Path, required=True)
    f.add_argument("--sources", default=",".join(FETCHERS))
    r = sub.add_parser("run")
    r.add_argument("--work", type=Path, required=True)
    r.add_argument("--out", type=Path, default=_ROOT / "results/static_rig/speeds")
    r.add_argument("--jobs", type=int, default=6)
    r.add_argument("--only", default=None, help="regex over uid")
    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        units: list[dict[str, Any]] = []
        for name in args.sources.split(","):
            print(f"fetch {name}", flush=True)
            FETCHERS[name](args.work, units)
        path = args.work / "units.json"
        old = json.loads(path.read_text()) if path.exists() else []
        merged = {u["uid"]: u for u in old + units}
        path.write_text(json.dumps(list(merged.values()), indent=1))
        print(f"{len(merged)} units in {path}")
        return 0
    specs = json.loads((args.work / "units.json").read_text())
    pat = re.compile(args.only) if args.only else None
    units_ = [
        Unit(uid=u["uid"], params={**u, "out": str(args.out)})
        for u in specs
        if pat is None or pat.search(u["uid"])
    ]
    res = run_grid(units_, worker, args.out, jobs=args.jobs, summarize=_summary)
    return res.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
