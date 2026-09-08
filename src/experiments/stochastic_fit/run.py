"""Job driver.

``prepare`` (CPU, once): assemble the clip bundle — the 27 refined training
crops plus the ``nosource`` validation clips of DREGON room1 / FLY124 with
their telemetry refined the same way — and upload it to R2 under
``artifacts/stochastic-fit/clips/``. Synthetic control clips (renderer draws
on real trajectories) are made here too, so the GPU side never imports the
renderer or the tracker and stays inside a slim source snapshot.

``fit`` (GPU): pull the bundle, fit the requested model variants to each
clip, write ``results/stochastic_fit/<variant>/<clip>.npz`` (scores, fitted
parameters, fitted spectrum, LOO smoother, the periodogram) — everything the
offline diagnostics need — and a ``summary.json``. Restartable: existing
outputs are skipped.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import boto3
import numpy as np

from .data import Clip, periodogram
from .model import Spec

BUCKET = "ml-data"
PREFIX = "artifacts/stochastic-fit/clips"
RESULTS = Path("results/stochastic_fit")

VARIANTS: dict[str, dict[str, Any]] = {
    # the renderer's family, as is
    "family": {},
    # + a smooth per-rotor carrier correction (imperfect references)
    "family_rps": dict(rps_offset=True),
    # structural alternatives, each one change from family_rps
    "free_gamma": dict(rps_offset=True, free_gamma=True),
    "gauss": dict(rps_offset=True, line_shape="gauss"),
    "mic_floor": dict(rps_offset=True, mic_floor=True),
    "speed_law": dict(rps_offset=True, fit_speed_law=True),
    # everything at once: the most the family's *shape* can be stretched
    "extended": dict(rps_offset=True, free_gamma=True, mic_floor=True, fit_speed_law=True),
}


def r2_client():
    from dotenv import load_dotenv

    load_dotenv(Path(".env")) if Path(".env").exists() else None
    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def clip_to_bytes(clip: Clip) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        audio=clip.audio.astype(np.float32),
        rps=clip.rps.astype(np.float64),
        rps_original=(clip.rps if clip.rps_original is None else clip.rps_original).astype(
            np.float64
        ),
        meta=np.array(
            json.dumps(dict(clip.meta, clip_id=clip.clip_id, group=clip.group, sample_rate=clip.sr))
        ),
    )
    return buf.getvalue()


def clip_from_bytes(data: bytes) -> Clip:
    with np.load(io.BytesIO(data), allow_pickle=True) as z:
        meta = json.loads(str(z["meta"]))
        return Clip(
            meta["clip_id"],
            meta["group"],
            np.asarray(z["audio"]),
            np.asarray(z["rps"]),
            int(meta["sample_rate"]),
            np.asarray(z["rps_original"]),
            meta,
        )


# ── prepare ───────────────────────────────────────────────────────────────


def prepare(args: argparse.Namespace) -> None:
    from . import data

    client = r2_client()
    manifest: list[dict[str, Any]] = []

    def put(clip: Clip) -> None:
        key = f"{PREFIX}/{clip.clip_id}.npz"
        client.put_object(Bucket=BUCKET, Key=key, Body=clip_to_bytes(clip))
        manifest.append(
            dict(
                clip_id=clip.clip_id,
                group=clip.group,
                key=key,
                duration_s=clip.duration_s,
                rps_median=np.median(clip.rps, axis=1).round(2).tolist(),
            )
        )
        print(f"  uploaded {key}", flush=True)

    if args.refs_dir:
        data.REFS_DIR = Path(args.refs_dir)
        for path in data.ref_clips():
            put(data.load_ref_clip(path))

    if args.valid:
        if args.valid_dir:
            data.VALID_DIR = Path(args.valid_dir)
        else:
            from data_processing.streams import ensure_local

            data.VALID_DIR = Path(ensure_local("DREGON-LM-V4-michaels-valid-full"))
        for sample_id, _rec in data.valid_nosource_ids():
            clip = data.load_valid_clip(sample_id)
            moving = clip.rps.max(axis=1) > 15.0
            if moving.any():
                t0 = time.time()
                try:
                    clip = data.refine_rps(clip)
                    print(f"  refined {clip.clip_id} in {time.time() - t0:.0f}s", flush=True)
                except Exception as exc:  # a stopped/ramp clip the tracker cannot hold
                    print(
                        f"  refinement failed on {clip.clip_id}: {exc!r}; keeping telemetry",
                        flush=True,
                    )
                    clip.meta["refinement"] = dict(error=repr(exc))
            put(clip)

    if args.synthetic:
        # renderer draws on real trajectories: planted-parameter controls
        rng = np.random.default_rng(0)
        sources = [m for m in manifest if m["group"] in ("fly125", "dregon_room2")]
        for i in range(args.synthetic):
            src = sources[int(rng.integers(len(sources)))]
            obj = client.get_object(Bucket=BUCKET, Key=src["key"])["Body"].read()
            real = clip_from_bytes(obj)
            clip, diag = data.synthetic_clip(1000 + i, real.rps)
            p = diag["params"]
            clip.meta.update(
                source_clip=src["clip_id"],
                planted=dict(
                    profile_db=p.profile_db.tolist(),
                    gamma0=p.gamma0.tolist(),
                    gamma_slope=p.gamma_slope.tolist(),
                    floor_ctrl_hz=p.floor_ctrl_hz.tolist(),
                    floor_ctrl_db=p.floor_ctrl_db.tolist(),
                    floor_tilt_db_oct=p.floor_tilt_db_oct,
                    harm_mean_db=p.harm_mean_db,
                    floor_mean_db=p.floor_mean_db,
                    harm_gp_std_db=p.harm_gp_std_db,
                    harm_gp_tau_s=p.harm_gp_tau_s,
                    harm_coherence=p.harm_coherence,
                    floor_gp_std_db=p.floor_gp_std_db,
                    floor_gp_tau_s=p.floor_gp_tau_s,
                    mic_gains_db=(10 * np.log10(diag["mic_gains"])).tolist(),
                    n_harmonics=p.n_harmonics,
                ),
            )
            put(clip)

    client.put_object(
        Bucket=BUCKET, Key=f"{PREFIX}/manifest.json", Body=json.dumps(manifest, indent=1).encode()
    )
    print(f"manifest: {len(manifest)} clips")


# ── fit ───────────────────────────────────────────────────────────────────


def make_spec(pg, *, n_mics: int, f_max: float | None, k_cap: int, variant: dict[str, Any]) -> Spec:
    live = pg.rps[pg.rps > 5.0]
    slowest = float(live.min()) if live.size else 20.0
    nyq = f_max or float(pg.freqs[-1])
    n_harm = int(min(np.floor(nyq / slowest), k_cap))
    return Spec(
        freqs=pg.freqs,
        times=pg.times,
        rps=pg.rps,
        n_mics=n_mics,
        n_harm=max(n_harm, 1),
        f_max=f_max,
        **variant,
    )


def fit(args: argparse.Namespace) -> None:
    import torch

    from .fit import fit_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}", flush=True)
    client = r2_client()
    manifest = json.loads(
        client.get_object(Bucket=BUCKET, Key=f"{PREFIX}/manifest.json")["Body"].read()
    )
    if args.groups:
        manifest = [m for m in manifest if m["group"] in args.groups]
    if args.clips:
        manifest = [m for m in manifest if m["clip_id"] in args.clips]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        manifest = manifest[i::n]
    variants = {k: VARIANTS[k] for k in args.variants}
    print(f"{len(manifest)} clips x {list(variants)}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS / f"summary_{args.tag}.json"
    rows: list[dict[str, Any]] = (
        json.loads(summary_path.read_text()) if summary_path.exists() else []
    )
    for entry in manifest:
        clip = clip_from_bytes(client.get_object(Bucket=BUCKET, Key=entry["key"])["Body"].read())
        pg = periodogram(clip)
        for name, variant in variants.items():
            out = RESULTS / name / f"{clip.clip_id}.npz"
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            spec = make_spec(
                pg, n_mics=clip.audio.shape[0], f_max=args.f_max, k_cap=args.k_cap, variant=variant
            )
            print(f"{clip.clip_id} [{name}] K={spec.n_harm} N={pg.times.size}", flush=True)
            try:
                res = fit_clip(
                    pg,
                    spec,
                    device=device,
                    iters=tuple(args.iters),
                    log=lambda s: print(s, flush=True),
                )
            except Exception as exc:
                print(f"  FAILED: {exc!r}", flush=True)
                (out.with_suffix(".err")).write_text(repr(exc))
                continue
            np.savez_compressed(
                out,
                power=pg.power,
                freqs=pg.freqs,
                times=pg.times,
                rps=pg.rps,
                spectrum=res["spectrum"],
                loo_smoother=res["loo_smoother"],
                scores=np.array(json.dumps(res["scores"])),
                params=np.array(
                    json.dumps(
                        {
                            k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in res["params"].items()
                        }
                    )
                ),
                spec=np.array(json.dumps({k: v for k, v in res["spec"].items() if k != "extra"})),
                meta=np.array(
                    json.dumps(
                        dict(clip.meta, clip_id=clip.clip_id, group=clip.group, variant=name)
                    )
                ),
            )
            rows.append(
                dict(
                    clip_id=clip.clip_id,
                    group=clip.group,
                    variant=name,
                    n_harm=spec.n_harm,
                    **res["scores"],
                )
            )
            summary_path.write_text(json.dumps(rows, indent=1))
            print(
                f"  -> {out}  explained={res['scores']['explained_fraction']:.3f} excess={res['scores']['excess_over_loo']:.4f}",
                flush=True,
            )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument(
        "--refs-dir", default=None, help="directory of refined-reference npz files (laptop only)"
    )
    p.add_argument(
        "--valid", action="store_true", help="include and refine the validation nosource clips"
    )
    p.add_argument("--valid-dir", default=None)
    p.add_argument(
        "--synthetic", type=int, default=0, help="number of renderer control clips to add"
    )
    p.set_defaults(func=prepare)
    f = sub.add_parser("fit")
    f.add_argument("--variants", nargs="+", default=["family"], choices=list(VARIANTS))
    f.add_argument("--groups", nargs="*", default=None)
    f.add_argument("--clips", nargs="*", default=None)
    f.add_argument("--shard", default=None, help="i/n: take every n-th clip starting at i")
    f.add_argument("--f-max", type=float, default=None)
    f.add_argument("--k-cap", type=int, default=300)
    f.add_argument("--iters", nargs=4, type=int, default=[150, 150, 300, 60])
    f.add_argument("--tag", default="run")
    f.set_defaults(func=fit)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
