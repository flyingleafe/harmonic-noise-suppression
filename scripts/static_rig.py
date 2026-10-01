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
``track`` follows each rotor over time (multichannel VK). ``combined`` reads the
total comb of recordings whose rotors are not all resolved, in the band
``results/static_rig/combined/bands.json`` written by
``scripts/static_rig_report.py bands``.
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


def _match(full: list[float], half: list[float]) -> list[float | None]:
    """One-to-one (Hungarian) match of a half-record's speeds to the full read."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    out: list[float | None] = [None] * len(full)
    if full and half:
        cost = np.abs(np.subtract.outer(np.asarray(full), np.asarray(half)))
        for i, j in zip(*linear_sum_assignment(cost), strict=True):
            out[int(i)] = float(half[j])
    return out


#: AGH single-rotor shaft rates measured in AGH.yaml (by take index).
_AGH_SINGLE = [159.5, 132.6, 113.7, 97.6, 77.7, 97.6, 97.6, 97.6]
#: DroneAudioSet cell medians of noise-v2-bench-points (rev/s).
_DAS_CELL = {
    ("daset_drone1", "low"): 84.96,
    ("daset_drone1", "high"): 118.68,
    ("daset_drone2", "low"): 105.83,
    ("daset_drone2", "high"): 126.03,
}


def rate_range(u: dict[str, Any]) -> tuple[float, float]:
    """Seeder rate range per recording: narrow (no octave inside) where the
    rate is known from a law or an earlier reading, wide where it is not."""
    rig, key = u["rig"], u["key"]
    if rig == "dregon":
        if u["n_rotors"] == 4:
            return 58.0, 80.0
        law = 0.975 * float(str(u["throttle"]).split()[0]) + 0.37
        return 0.85 * law, 1.15 * law
    if rig == "spcup_agh":
        if "single_rotors" in key:
            s = _AGH_SINGLE[int(key.rsplit("__", 1)[1])]
            return 0.85 * s, 1.15 * s
        # 3-blade props (AGH.yaml, report Fig. 2): the shaft band, so blade-pass
        # combs (3x) are not tracked as rotors
        return 40.0, 100.0
    if rig == "spcup_ku_leuven":
        return 80.0, 150.0
    if rig == "spcup_maverick":
        return 28.0, 50.0
    if rig == "avq":
        return 65.0, 120.0
    s = _DAS_CELL[(rig, u["throttle"])]
    return 0.75 * s, 1.3 * s


def worker_track(unit: Unit) -> dict[str, Any]:
    import numpy as np

    from experiments.static_rig import spectra as S
    from experiments.static_rig import tracks as T

    prm = dict(unit.params)
    x = np.load(prm["audio"], mmap_mode="r")
    fs = int(prm["fs"])
    a, b = S.motor_on_span(np.asarray(x), fs, S.Params())
    cap = int(float(prm["max_s"]) * fs)
    if b - a > cap:
        mid = (a + b) // 2
        a, b = mid - cap // 2, mid + cap // 2
    lo, hi = rate_range(prm)
    out = T.track(np.asarray(x[:, a:b]), fs, int(prm["n_rotors"]), lo, hi, T.TrackParams())
    return {
        **{k: v for k, v in prm.items() if k not in ("audio", "out")},
        "span_s": [a / fs, b / fs],
        "rate_range": [lo, hi],
        **out,
    }


def _load_span(path: str, fs: int, max_s: float):
    import numpy as np

    from experiments.static_rig import spectra as S

    x = np.load(path, mmap_mode="r")
    a, b = S.motor_on_span(np.asarray(x), fs, S.Params())
    cap = int(max_s * fs)
    if b - a > cap:
        mid = (a + b) // 2
        a, b = mid - cap // 2, mid + cap // 2
    return np.asarray(x[:, a:b], dtype=np.float64), [a / fs, b / fs]


def worker_track_group(unit: Unit) -> dict[str, Any]:
    """One DroneAudioSet physical take: the up / down rings and the centre mic
    were recorded simultaneously. Seed + refine on the primary ring (up, else
    down), then refine that trajectory on each other member (its own clock:
    the centre mic runs on an independent device) and profile every member."""
    import numpy as np

    from experiments.static_rig import tracks as T

    prm = dict(unit.params)
    tp = T.TrackParams()
    members = prm["members"]
    order = [m for m in ("M_up", "M_down", "M_center") if m in members]
    lo, hi = rate_range(prm)
    out: dict[str, Any] = {k: v for k, v in prm.items() if k not in ("members", "out")}
    out.update(rate_range=[lo, hi], primary=order[0], members={})
    ref_t = ref_r = None
    for m in order:
        info = members[m]
        x, span = _load_span(info["audio"], int(info["fs"]), float(prm["max_s"]))
        x = T.resample(x, int(info["fs"]), tp.fs)[:8]
        if ref_r is None:
            seeds, ft, ch = T.seed(x, int(prm["n_rotors"]), lo, hi, tp)
            res = T.refine_and_profile(x, seeds, ft, tp)
            res.update(seed_channel=ch, seeds=seeds.tolist())
            ref_t = np.asarray(res["frame_times"])
            ref_r = np.array([r["track"] for r in res["rotors"]])
        else:
            assert ref_t is not None
            ft = T.frame_grid(x.shape[1], tp)
            r0 = np.stack([np.interp(ft, ref_t, row) for row in ref_r])
            res = T.refine_and_profile(x, r0, ft, tp)
        res.update(key=info["key"], span_s=span, n_ch=int(x.shape[0]))
        out["members"][m] = res
    return out


def _das_groups(specs: list[dict[str, Any]], out: str, max_s: float) -> list[Unit]:
    groups: dict[str, dict[str, Any]] = {}
    for u in specs:
        if u["dataset"] != "DroneAudioSet":
            continue
        gid = _uid("DAS", f"{u['rig']}_{u['throttle']}_{u['mic_dist']}_{u['take']}")
        g = groups.setdefault(
            gid,
            {
                "dataset": "DroneAudioSet",
                "key": gid,
                "rig": u["rig"],
                "throttle": u["throttle"],
                "mic_dist": u["mic_dist"],
                "take": u["take"],
                "n_rotors": u["n_rotors"],
                "members": {},
                "out": out,
                "max_s": max_s,
            },
        )
        g["members"][u["mic"]] = {"audio": u["audio"], "fs": u["fs"], "key": u["key"]}
    return [Unit(uid=gid, params=g) for gid, g in sorted(groups.items())]


def _summary_track(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def one(r: dict[str, Any]) -> dict[str, Any]:
        return {
            "n_distinct": r["n_distinct"],
            "means": [round(x["mean"], 3) for x in r["rotors"]],
            "stds": [round(x["std"], 3) for x in r["rotors"]],
            "residual": round(r["residual_ratios"][-1], 3),
        }

    return {
        "n": len(rows),
        "recordings": {
            f"{r['dataset']}:{r['key']}": (
                {m: one(v) for m, v in r["members"].items()} if "members" in r else one(r)
            )
            for r in rows
        },
    }


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
        halves.append(_match(speeds, hs))
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


def worker_combined(unit: Unit) -> dict[str, Any]:
    """Total comb power of one recording whose rotors are not all resolved:
    full motor-on span, every rotor in the band ``[s_lo, s_hi]`` (from
    ``static_rig_report.py bands``)."""
    import numpy as np

    from experiments.static_rig import spectra as S

    prm = dict(unit.params)
    p = S.Params()
    x = np.load(prm["audio"], mmap_mode="r")
    fs = int(prm["fs"])
    a, b = S.motor_on_span(np.asarray(x), fs, p)
    sp = S.spectrum(np.asarray(x[:, a:b], dtype=np.float64), fs, p)
    band = prm["band"]
    res = S.comb_band_powers(sp, float(band["s_lo"]), float(band["s_hi"]), p)
    return {
        **{k: v for k, v in prm.items() if k not in ("audio", "out")},
        "span_s": [a / fs, b / fs],
        "df": sp.df,
        "orders": res["orders"],
        "power_db": _jsonable(res["power_db"]),
        "snr_db": _jsonable(res["snr_db"]),
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
    t = sub.add_parser("track")
    t.add_argument("--work", type=Path, required=True)
    t.add_argument("--out", type=Path, default=_ROOT / "results/static_rig/tracks")
    t.add_argument("--jobs", type=int, default=6)
    t.add_argument("--only", default=None, help="regex over uid")
    t.add_argument("--max-s", type=float, default=160.0)
    c = sub.add_parser("combined")
    c.add_argument("--work", type=Path, required=True)
    c.add_argument("--out", type=Path, default=_ROOT / "results/static_rig/combined")
    c.add_argument("--bands", type=Path, default=_ROOT / "results/static_rig/combined/bands.json")
    c.add_argument("--jobs", type=int, default=6)
    c.add_argument("--only", default=None, help="regex over uid")
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
    if args.cmd == "combined":
        bands = json.loads(args.bands.read_text())
        cu = [
            Unit(uid=u.uid, params={**u.params, "band": bands[u.uid]})
            for u in units_
            if u.uid in bands
        ]
        return run_grid(cu, worker_combined, args.out, jobs=args.jobs).exit_code
    if args.cmd == "track":
        singles = [
            Unit(uid=u.uid, params={**u.params, "max_s": args.max_s})
            for u in units_
            if u.params["dataset"] != "DroneAudioSet"
        ]
        das = [
            g
            for g in _das_groups(specs, str(args.out), args.max_s)
            if pat is None or pat.search(g.uid)
        ]
        ok = run_grid(singles, worker_track, args.out, jobs=args.jobs, summarize=_summary_track)
        res = run_grid(
            das,
            worker_track_group,
            args.out,
            jobs=args.jobs,
            summarize=_summary_track,
            summary_name="summary_das.json",
        )
        return ok.exit_code or res.exit_code
    res = run_grid(units_, worker, args.out, jobs=args.jobs, summarize=_summary)
    return res.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
