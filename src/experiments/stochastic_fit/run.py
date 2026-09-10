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
import contextlib
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
from .model import Spec, make_spec

BUCKET = "ml-data"
PREFIX = "artifacts/stochastic-fit/clips"
RESULTS = Path("results/stochastic_fit")

VARIANTS: dict[str, dict[str, Any]] = {
    # the renderer's family, as is
    "family": {},
    # + a smooth per-rotor carrier correction (imperfect references)
    "family_rps": dict(rps_offset=True),
    # the renderer's REALIZED line: skirts cut at +-5 gamma and renormalized
    "family_trunc": dict(line_shape="lorentz_trunc"),
    "family_trunc_rps": dict(rps_offset=True, line_shape="lorentz_trunc"),
    # the renderer's realized support (power-of-two bucket >= 5 gamma, fixed norm)
    "family_bucket": dict(line_shape="lorentz_bucket"),
    "family_bucket_rps": dict(rps_offset=True, line_shape="lorentz_bucket"),
    # structural alternatives, each one change from family_rps
    "free_gamma": dict(rps_offset=True, free_gamma=True),
    "gauss": dict(rps_offset=True, line_shape="gauss"),
    "mic_floor": dict(rps_offset=True, mic_floor=True),
    "speed_law": dict(rps_offset=True, fit_speed_law=True),
    # sub-bin lines: no 0.6-bin width floor, exact bin integral — tests the
    # peaked low-order residual (real low harmonics sharper than the renderer can make)
    # discretization alone: exact bin integral, original 0.6-bin width floor;
    # `sharp - integrated` isolates sub-bin width, `integrated - family_rps` the sampling
    "integrated": dict(rps_offset=True, line_bin_integrate=True),
    "sharp": dict(rps_offset=True, gamma_min_bins=0.0, line_bin_integrate=True),
    # weak drift prior (the family's top of range): tests the slow residual structure
    "drift6": dict(rps_offset=True, gp_std_db=6.0),
    # the Gaussian line shape removes ~0.24 of the ~0.30 nats/cell excess on
    # the 27 crops (2026-09-08); the rest is probed on top of it
    "gauss_sharp": dict(
        rps_offset=True, line_shape="gauss", gamma_min_bins=0.0, line_bin_integrate=True
    ),
    "gauss_mic": dict(rps_offset=True, line_shape="gauss", mic_floor=True),
    "gauss_drift6": dict(rps_offset=True, line_shape="gauss", gp_std_db=6.0),
    "gauss_free": dict(rps_offset=True, line_shape="gauss", free_gamma=True),
    # the width law's regime: k^1 is a frozen speed offset (quasi-static shaft),
    # k^2 an OU shaft that decorrelates inside the analysis window (diffusive).
    # ``gauss_wp`` fits the exponent, so the regime is read off the likelihood.
    "gauss_wp": dict(rps_offset=True, line_shape="gauss", fit_width_power=True),
    "gauss_wp2": dict(rps_offset=True, line_shape="gauss", width_power=2.0),
    "gauss_all": dict(
        rps_offset=True,
        line_shape="gauss",
        free_gamma=True,
        mic_floor=True,
        gamma_min_bins=0.0,
        line_bin_integrate=True,
        gp_std_db=6.0,
    ),
    # everything at once: the most the family's *shape* can be stretched
    "extended": dict(
        rps_offset=True,
        free_gamma=True,
        mic_floor=True,
        fit_speed_law=True,
        gamma_min_bins=0.0,
        line_bin_integrate=True,
        gp_std_db=6.0,
    ),
}


def r2_client():
    env = Path(".env")
    if env.exists():  # the job ships .env; parse it without depending on python-dotenv
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(
                    key.strip().removeprefix("export "), value.strip().strip("'\"")
                )
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
    # incremental: the refs, the validation clips and the controls are uploaded in separate runs
    with contextlib.suppress(client.exceptions.NoSuchKey):
        manifest = json.loads(
            client.get_object(Bucket=BUCKET, Key=f"{PREFIX}/manifest.json")["Body"].read()
        )

    def put(clip: Clip) -> None:
        key = f"{PREFIX}/{clip.clip_id}.npz"
        client.put_object(Bucket=BUCKET, Key=key, Body=clip_to_bytes(clip))
        manifest[:] = [m for m in manifest if m["clip_id"] != clip.clip_id]
        manifest.append(
            dict(
                clip_id=clip.clip_id,
                group=clip.group,
                key=key,
                duration_s=clip.duration_s,
                rps_median=np.median(clip.rps, axis=1).round(2).tolist(),
                refinement=clip.meta.get("refinement"),
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


# ── preset renders (the calibration gate's clips) ─────────────────────────


def prepare_presets(args: argparse.Namespace) -> None:
    """Render the rig presets of an online-mix policy on the REAL clips'
    trajectories and add them to the manifest as ``syn_<preset>`` groups, so the
    rig fit can score the renderer's own output the way it scores recordings
    (gate 1 of ``docs/hierarchical-rig-model-plan.md`` § 7)."""
    import yaml

    from data_processing import stochastic_rotor_noise as srn

    from .data import Clip

    client = r2_client()
    manifest = json.loads(
        client.get_object(Bucket=BUCKET, Key=f"{PREFIX}/manifest.json")["Body"].read()
    )
    pol = yaml.safe_load(Path(args.policy).read_text())
    presets = [src for src in pol["sources"]["noise"] if src.get("kind") == "stochastic"]
    names = args.names or [f"preset{i}" for i in range(len(presets))]
    rng = np.random.default_rng(args.seed)
    for name, src, groups in zip(names, presets, args.groups_per_preset):
        ranges = srn.StochasticRanges.from_dict(src.get("ranges"))
        sources = [m for m in manifest if m["group"] in groups.split(",")]
        n_done = 0
        for entry in sources:
            if n_done >= args.per_preset:
                break
            clip = clip_from_bytes(
                client.get_object(Bucket=BUCKET, Key=entry["key"])["Body"].read()
            )
            if clip.rps.max() < 5:
                continue
            rps = clip.rps.astype(np.float64)
            hover = float(np.median(rps[rps > 5]))
            n_harm = int(np.clip(np.ceil(clip.sr / 2.0 / max(hover, 1.0)), 40, 200))
            params = srn.sample_params(
                rng, ranges, n_rotors=rps.shape[0], n_harmonics=n_harm, sample_rate=clip.sr
            )
            audio, _ = srn.synthesize(
                params,
                rps,
                rng=rng,
                n_mics=clip.audio.shape[0],
                mic_gain_db=tuple(src.get("mic_gain_db", (0.0, 0.0))),
                line_mode=str(src.get("line_mode", "stochastic")),
            )
            cid = f"syn_{name}_{clip.clip_id}"
            syn = Clip(
                cid,
                f"syn_{name}",
                audio.astype(np.float32),
                clip.rps,
                clip.sr,
                clip.rps.copy(),
                dict(source=clip.clip_id, preset=name),
            )
            key = f"{PREFIX}/{cid}.npz"
            client.put_object(Bucket=BUCKET, Key=key, Body=clip_to_bytes(syn))
            manifest = [m for m in manifest if m["clip_id"] != cid]
            manifest.append(
                dict(
                    clip_id=cid,
                    group=f"syn_{name}",
                    key=key,
                    duration_s=audio.shape[1] / clip.sr,
                    rps_median=np.median(rps, axis=1).round(2).tolist(),
                    refinement=None,
                )
            )
            n_done += 1
            print(f"rendered {cid}", flush=True)
    client.put_object(
        Bucket=BUCKET, Key=f"{PREFIX}/manifest.json", Body=json.dumps(manifest, indent=1).encode()
    )
    print(f"manifest: {len(manifest)} clips")


# ── fit ───────────────────────────────────────────────────────────────────


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
    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    summary_path = results / f"summary_{args.tag}.json"
    rows: list[dict[str, Any]] = (
        json.loads(summary_path.read_text()) if summary_path.exists() else []
    )
    for entry in manifest:
        clip = clip_from_bytes(client.get_object(Bucket=BUCKET, Key=entry["key"])["Body"].read())
        pg = periodogram(clip)
        for name, variant in variants.items():
            out = results / name / f"{clip.clip_id}.npz"
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            extra = {} if args.gp_std is None else dict(gp_std_db=float(args.gp_std))
            spec = make_spec(
                pg,
                n_mics=clip.audio.shape[0],
                f_max=args.f_max,
                k_cap=args.k_cap,
                variant=variant | extra,
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
            finite = np.isfinite(res["scores"]["nll_fit"]) and np.all(np.isfinite(res["spectrum"]))
            if not finite:
                print("  FAILED: non-finite fit", flush=True)
                (out.with_suffix(".err")).write_text("non-finite fit")
                continue
            # slim: the periodogram and the LOO smoother are recomputed from the R2
            # clip by the offline diagnostics; the fitted spectrum travels as
            # float16 decibels (0.01 dB resolution). Full-size files (2.5 GB per
            # 50-clip job) broke Kaggle's output collection.
            np.savez_compressed(
                out,
                freqs=pg.freqs,
                times=pg.times,
                rps=pg.rps,
                spectrum_db=(10.0 * np.log10(np.maximum(res["spectrum"], 1e-30))).astype(
                    np.float16
                ),
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


# ── rig fit (the hierarchical ladder) ─────────────────────────────────────

#: Ladder steps of ``docs/hierarchical-rig-model-plan.md`` § 4: each is a
#: (Spec overrides, RigSpec overrides) pair; every step is Gaussian lines with
#: a carrier correction, the two established facts.
BASE_VARIANT: dict[str, Any] = dict(rps_offset=True, line_shape="gauss")
LADDER: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    # independent fits in the rig code path (the M0 reference)
    "M0": (
        {},
        dict(
            tie_floor_shape=False,
            tie_tilt=False,
            tie_profile=False,
            tie_width=False,
            tie_mic_gain=False,
        ),
    ),
    "M1": ({}, {}),  # every rig-level parameter tied
    "M2": ({}, dict(rotor_delta=True)),
    "M3": ({}, dict(rotor_delta=True, rotor_width=True)),
    # per-mic floor gains + per-mic low-band modulation (DREGON's flow noise)
    "M4": (
        dict(mic_floor=True, umod_std_db=3.5, umod_tau_s=0.25),
        dict(rotor_delta=True, rotor_width=True),
    ),
    # one per-mic gain on everything (Michael's rig) instead of separate floor gains
    "M4g": (
        dict(gain_all=True, umod_std_db=3.5, umod_tau_s=0.25),
        dict(rotor_delta=True, rotor_width=True),
    ),
    # faster, rougher amplitude process
    "M5": (
        dict(mic_floor=True, umod_std_db=3.5, umod_tau_s=0.25, gp_kernel="ou", gp_tau_s=0.5),
        dict(rotor_delta=True, rotor_width=True),
    ),
    "M5g": (
        dict(gain_all=True, umod_std_db=3.5, umod_tau_s=0.25, gp_kernel="ou", gp_tau_s=0.5),
        dict(rotor_delta=True, rotor_width=True),
    ),
    # + the speed laws fitted and tied at rig level. Per-clip fits cannot
    # identify an exponent (a 4 s crop spans too little speed: per-clip values
    # scatter from -0.2 to 6.4), but a rig fit pools every training clip, which
    # collectively cover the whole flight envelope. This is the only route to a
    # measured speed law for a rig with no published decomposition.
    "M5s": (
        dict(
            mic_floor=True,
            umod_std_db=3.5,
            umod_tau_s=0.25,
            gp_kernel="ou",
            gp_tau_s=0.5,
            fit_speed_law=True,
        ),
        dict(rotor_delta=True, rotor_width=True),
    ),
    "M5gs": (
        dict(
            gain_all=True,
            umod_std_db=3.5,
            umod_tau_s=0.25,
            gp_kernel="ou",
            gp_tau_s=0.5,
            fit_speed_law=True,
        ),
        dict(rotor_delta=True, rotor_width=True),
    ),
}


def rigfit(args: argparse.Namespace) -> None:
    import torch

    from .rig import RigSpec, fit_heldout, fit_rig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}", flush=True)
    client = r2_client()
    manifest = json.loads(
        client.get_object(Bucket=BUCKET, Key=f"{PREFIX}/manifest.json")["Body"].read()
    )
    if args.clips:
        manifest = [m for m in manifest if m["clip_id"] in args.clips]

    def load(groups: list[str]) -> list[tuple[str, str, Any, Any]]:
        out = []
        for entry in manifest:
            if entry["group"] not in groups:
                continue
            clip = clip_from_bytes(
                client.get_object(Bucket=BUCKET, Key=entry["key"])["Body"].read()
            )
            if clip.rps.max() < 5:
                continue
            out.append((clip.clip_id, entry["group"], clip, periodogram(clip)))
        return out

    train_raw = load(args.train_groups)
    test_raw = load(args.test_groups) if args.test_groups else []
    print(
        f"train {len(train_raw)} clips {args.train_groups}; test {len(test_raw)} {args.test_groups}",
        flush=True,
    )
    results = Path(args.results_dir) / args.tag
    results.mkdir(parents=True, exist_ok=True)
    for name in args.configs:
        spec_over, rig_over = LADDER[name]
        out_json = results / f"{name}.json"
        if out_json.exists():
            print(f"{name}: exists, skipping", flush=True)
            continue

        def clips_for(raw, spec_over=spec_over):
            return [
                (
                    cid,
                    g,
                    pg,
                    make_spec(
                        pg,
                        n_mics=clip.audio.shape[0],
                        f_max=args.f_max,
                        k_cap=args.k_cap,
                        variant=BASE_VARIANT | spec_over,
                    ),
                )
                for cid, g, clip, pg in raw
            ]

        train = clips_for(train_raw)
        # one K for the rig: the smallest over clips (the profile is shared)
        k_rig = min(c[3].n_harm for c in train + (clips_for(test_raw) if test_raw else []))
        train = [(a, b, c, Spec(**{**d.__dict__, "n_harm": k_rig})) for a, b, c, d in train]
        print(f"== {name}: K={k_rig}, spec {spec_over}, rig {rig_over}", flush=True)
        t0 = time.time()
        fitted = fit_rig(
            train,
            RigSpec(**rig_over),
            device=device,
            iters=tuple(args.iters),
            log=lambda m: print(m, flush=True),
        )
        held = None
        if test_raw:
            test = [
                (a, b, c, Spec(**{**d.__dict__, "n_harm": k_rig}))
                for a, b, c, d in clips_for(test_raw)
            ]
            held = fit_heldout(
                fitted,
                test,
                device=device,
                iters=tuple(args.iters),
                log=lambda m: print(m, flush=True),
            )

        def slim(clips: dict[str, Any]) -> dict[str, Any]:
            return {
                cid: dict(
                    group=v["group"],
                    scores=v["scores"],
                    params={
                        k: (x.tolist() if isinstance(x, np.ndarray) else x)
                        for k, x in v["params"].items()
                        if k not in ("carrier", "h_db", "umod_db", "rps_offset")
                    },
                )
                for cid, v in clips.items()
            }

        summary = dict(
            config=name,
            spec=fitted["spec"],
            rig_spec=fitted["rig_spec"],
            k_rig=k_rig,
            rig={
                k: (x.tolist() if isinstance(x, np.ndarray) else x)
                for k, x in fitted["rig"].items()
            },
            train=slim(fitted["clips"]),
            heldout=slim(held["clips"]) if held else None,
            seconds=time.time() - t0,
        )
        out_json.write_text(json.dumps(summary))
        torch.save(fitted["rig_state"], results / f"{name}_rig_state.pt")
        arrays: dict[str, np.ndarray] = {
            f"{cid}__{k}": np.asarray(x)
            for cid, v in fitted["clips"].items()
            for k, x in v["params"].items()
            if isinstance(x, np.ndarray)
        }
        with open(results / f"{name}_clip_params.npz", "wb") as fh:
            np.savez_compressed(fh, **arrays)  # type: ignore[arg-type]
        ex_tr = [v["scores"]["excess_over_loo"] for v in fitted["clips"].values()]
        line = f"{name}: train excess median {np.median(ex_tr):+.3f}"
        if held:
            ex_te = [v["scores"]["excess_over_loo"] for v in held["clips"].values()]
            line += f"; held-out median {np.median(ex_te):+.3f}"
        print(line + f"  ({time.time() - t0:.0f}s)", flush=True)


def population_fit(args: argparse.Namespace) -> None:
    """Fit and held-out-score the P0--P3 marginal population ladder."""
    import torch

    from .population import PopulationFitSpec, fit_population, fit_population_heldout
    from .rig import RigSpec

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}", flush=True)
    client = r2_client()
    manifest = json.loads(
        client.get_object(Bucket=BUCKET, Key=f"{PREFIX}/manifest.json")["Body"].read()
    )
    if args.clips:
        manifest = [entry for entry in manifest if entry["clip_id"] in args.clips]

    def load(groups: list[str]) -> list[tuple[str, str, Any, Any]]:
        loaded = []
        for entry in manifest:
            if entry["group"] not in groups:
                continue
            clip = clip_from_bytes(
                client.get_object(Bucket=BUCKET, Key=entry["key"])["Body"].read()
            )
            if clip.rps.max() < 5:
                continue
            loaded.append((clip.clip_id, entry["group"], clip, periodogram(clip)))
        return loaded

    raw_train = load(args.train_groups)
    raw_test = load(args.test_groups) if args.test_groups else []
    print(
        f"train {len(raw_train)} clips {args.train_groups}; "
        f"test {len(raw_test)} {args.test_groups}",
        flush=True,
    )
    spec_over, base_rig_over = LADDER[args.rig_config]

    def make_clips(raw: list[tuple[str, str, Any, Any]]) -> list[tuple[str, str, Any, Spec]]:
        return [
            (
                clip_id,
                group,
                pg,
                make_spec(
                    pg,
                    n_mics=clip.audio.shape[0],
                    f_max=args.f_max,
                    k_cap=args.k_cap,
                    variant=BASE_VARIANT | spec_over,
                ),
            )
            for clip_id, group, clip, pg in raw
        ]

    untrimmed = make_clips(raw_train) + make_clips(raw_test)
    k_rig = min(item[3].n_harm for item in untrimmed)

    def trim(raw: list[tuple[str, str, Any, Any]]) -> list[tuple[str, str, Any, Spec]]:
        return [
            (clip_id, group, pg, Spec(**{**spec.__dict__, "n_harm": k_rig}))
            for clip_id, group, pg, spec in make_clips(raw)
        ]

    train = trim(raw_train)
    test = trim(raw_test)
    results = Path(args.results_dir) / args.tag
    results.mkdir(parents=True, exist_ok=True)
    fit_spec = PopulationFitSpec(
        mc_samples=args.mc_samples,
        iw_samples=args.iw_samples,
        init_posterior_std=args.init_posterior_std,
    )

    def array_dict(values: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in values.items()
        }

    def slim(clips: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for clip_id, value in clips.items():
            row = {
                "group": value["group"],
                "scores": value["scores"],
                "params": {
                    key: item.tolist() if isinstance(item, np.ndarray) else item
                    for key, item in value["params"].items()
                    if key not in ("carrier", "h_db", "umod_db", "rps_offset")
                },
                "posterior": {
                    part: {
                        key: item.tolist()
                        for key, item in values.items()
                        if key in ("level_db", "profile_z")
                    }
                    for part, values in value["posterior"].items()
                },
            }
            for key in ("iw_log_evidence", "iw_nll_per_cell", "iw_ess"):
                if key in value:
                    row[key] = value[key]
            out[clip_id] = row
        return out

    for rank in args.ranks:
        name = f"P{rank}"
        output = results / f"{name}.json"
        if output.exists():
            print(f"{name}: exists, skipping", flush=True)
            continue
        rig_spec = RigSpec(**(base_rig_over | {"profile_rank": rank}))
        print(
            f"== {name}: K={k_rig}, base={args.rig_config}, rank={rank}",
            flush=True,
        )
        started = time.time()
        fitted = fit_population(
            train,
            rig_spec,
            fit_spec=fit_spec,
            device=device,
            map_iters=tuple(args.map_iters),
            vi_iters=args.vi_iters,
            map_lr=args.map_lr,
            vi_lr=args.vi_lr,
            seed=args.seed + rank,
            log=lambda message: print(message, flush=True),
        )
        heldout = (
            fit_population_heldout(
                fitted,
                test,
                device=device,
                map_iters=tuple(args.map_iters),
                vi_iters=args.heldout_vi_iters,
                map_lr=args.map_lr,
                vi_lr=args.vi_lr,
                seed=args.seed + 100 + rank,
                log=lambda message: print(message, flush=True),
            )
            if test
            else None
        )
        summary = {
            "config": name,
            "base_config": args.rig_config,
            "spec": fitted["spec"],
            "rig_spec": fitted["rig_spec"],
            "fit_spec": fitted["fit_spec"],
            "k_rig": k_rig,
            "rig": array_dict(fitted["rig"]),
            "population": fitted["population"],
            "train_elbo_per_cell": fitted["elbo_per_cell"],
            "heldout_elbo_per_cell": (heldout["elbo_per_cell"] if heldout is not None else None),
            "train": slim(fitted["clips"]),
            "heldout": slim(heldout["clips"]) if heldout is not None else None,
            "seconds": time.time() - started,
        }
        output.write_text(json.dumps(summary))
        torch.save(
            {
                "rig_spec": fitted["rig_spec"],
                "fit_spec": fitted["fit_spec"],
                "rig_state": fitted["rig_state"],
                "population_state": fitted["population_state"],
                "rig": fitted["rig"],
                "population": fitted["population"],
            },
            results / f"{name}_state.pt",
        )
        heldout_nll = (
            np.median([value["iw_nll_per_cell"] for value in heldout["clips"].values()])
            if heldout is not None
            else float("nan")
        )
        print(
            f"{name}: train ELBO/cell {fitted['elbo_per_cell']:+.4f}; "
            f"held-out IW NLL/cell {heldout_nll:+.4f} "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )


def population_latent_diagnostic(args: argparse.Namespace) -> None:
    """Compare prior draws with fitted latents; explicitly not a realism gate."""
    from .predictive import (
        fitted_profile_draws,
        latent_topology_diagnostic,
        synthetic_profile_draws,
    )

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text())
    fitted_train = fitted_profile_draws(summary, "train")
    fitted_test = fitted_profile_draws(summary, "heldout")
    synthetic_train = synthetic_profile_draws(
        summary, fitted_train.profile_db.shape[0], seed=args.seed
    )
    synthetic_test = synthetic_profile_draws(
        summary, fitted_test.profile_db.shape[0], seed=args.seed + 1
    )
    result = latent_topology_diagnostic(
        fitted_train,
        fitted_test,
        synthetic_train,
        synthetic_test,
        seed=args.seed + 2,
        n_bootstrap=args.bootstrap,
    )
    output = (
        Path(args.output)
        if args.output is not None
        else summary_path.with_suffix(".latent-topology.json")
    )
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def population_raw_gate(args: argparse.Namespace) -> None:
    """Compare held-out real and posterior-predictive waveforms without refitting."""
    import yaml

    from .raw_predictive import (
        extract_raw_topology,
        raw_topology_gate,
        render_matched_population,
    )

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text())
    policy = yaml.safe_load(Path(args.policy).read_text())
    source = policy["sources"]["noise"][args.source_index]
    if source.get("kind") != "stochastic":
        raise ValueError("source-index must select a stochastic noise source")
    if args.classifier and any(
        not group.startswith("fly") for group in (*args.train_groups, *args.test_groups)
    ):
        raise ValueError(
            "the waveform classifier is reserved for Michael's recordings; "
            "DREGON gusts make it a trivial domain detector"
        )

    client = r2_client()
    manifest = json.loads(
        client.get_object(Bucket=BUCKET, Key=f"{PREFIX}/manifest.json")["Body"].read()
    )

    def load(groups: list[str]) -> list[Clip]:
        clips: list[Clip] = []
        for entry in manifest:
            if entry["group"] not in groups:
                continue
            clip = clip_from_bytes(
                client.get_object(Bucket=BUCKET, Key=entry["key"])["Body"].read()
            )
            if (
                entry["group"].startswith("dregon")
                and clip.rps.max() >= 5
                and clip.meta.get("refinement") is None
            ):
                raise ValueError(f"{clip.clip_id}: DREGON waveform gates require refined RPS")
            if clip.rps.max() >= 5:
                clips.append(clip)
        return clips

    train_real = load(args.train_groups)
    test_real = load(args.test_groups)
    observation_augmentations: list[dict[str, Any]] = []
    if args.observation in ("recolor", "both"):
        observation_augmentations.append(
            {
                "probability": 1.0,
                "choices": [
                    {
                        "spectral_recolor": {
                            "gain_db": 8.0,
                            "n_anchors": 10,
                            "f_low": 30.0,
                            "f_high": 8000.0,
                        }
                    }
                ],
            }
        )
    if args.observation in ("reverb", "both"):
        observation_augmentations.append(
            {
                "probability": 1.0,
                "choices": [
                    {
                        "random_reverb": {
                            "n_rirs": 200,
                            "rt60_low": 0.1,
                            "rt60_high": 0.8,
                            "drr_low_db": 3.0,
                            "drr_high_db": 15.0,
                        }
                    }
                ],
            }
        )
    train_synthetic = render_matched_population(
        summary,
        train_real,
        base_ranges=source.get("ranges", {}),
        line_mode=source.get("line_mode", "stochastic"),
        mic_gain_db=tuple(source.get("mic_gain_db", (-12.0, 0.0))),
        seed=args.seed,
        observation_augmentations=observation_augmentations,
        draws_per_clip=args.draws_per_clip,
    )
    test_synthetic = render_matched_population(
        summary,
        test_real,
        base_ranges=source.get("ranges", {}),
        line_mode=source.get("line_mode", "stochastic"),
        mic_gain_db=tuple(source.get("mic_gain_db", (-12.0, 0.0))),
        seed=args.seed + 10_000,
        observation_augmentations=observation_augmentations,
        draws_per_clip=args.draws_per_clip,
    )
    raw_train_real = extract_raw_topology(train_real, k_max=args.k_max)
    raw_test_real = extract_raw_topology(test_real, k_max=args.k_max)
    raw_train_synthetic = extract_raw_topology(train_synthetic, k_max=args.k_max)
    raw_test_synthetic = extract_raw_topology(test_synthetic, k_max=args.k_max)
    result = raw_topology_gate(
        raw_train_real,
        raw_test_real,
        raw_train_synthetic,
        raw_test_synthetic,
        seed=args.seed + 20_000,
        n_bootstrap=args.bootstrap,
        run_classifier=args.classifier,
    )
    result.update(
        summary=str(summary_path),
        policy=args.policy,
        source_index=args.source_index,
        observation=args.observation,
        classifier=args.classifier,
        draws_per_clip=args.draws_per_clip,
        train_groups=args.train_groups,
        test_groups=args.test_groups,
    )
    output = (
        Path(args.output)
        if args.output is not None
        else summary_path.with_suffix(".raw-topology.json")
    )
    output.write_text(json.dumps(result, indent=2))
    with open(output.with_suffix(".npz"), "wb") as handle:
        np.savez_compressed(
            handle,
            train_real=raw_train_real.margin_db,
            test_real=raw_test_real.margin_db,
            train_synthetic=raw_train_synthetic.margin_db,
            test_synthetic=raw_test_synthetic.margin_db,
            train_real_observations=raw_train_real.observations,
            test_real_observations=raw_test_real.observations,
            train_synthetic_observations=raw_train_synthetic.observations,
            test_synthetic_observations=raw_test_synthetic.observations,
        )
    if args.compact:
        keys = (
            "scope",
            "status",
            "classifier_enabled",
            "classifier_auc",
            "classifier_auc_95",
            "predictive_90_coverage",
            "median_curve_iqr_rmse",
        )
        print(json.dumps({key: result.get(key) for key in keys}), flush=True)
    else:
        print(json.dumps(result, indent=2), flush=True)


def population_calibrate(args: argparse.Namespace) -> None:
    """Fit the training-only Gaussian synthetic likelihood and update a summary."""
    from .topology_calibration import (
        apply_topology_calibration,
        fit_topology_calibration,
    )

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text())
    with np.load(args.raw_npz) as arrays:
        calibration = fit_topology_calibration(
            np.asarray(arrays["train_real"]),
            np.asarray(arrays["train_synthetic"]),
            reference_order=args.reference_order,
        )
    corrected = apply_topology_calibration(summary, calibration)
    output = (
        Path(args.output)
        if args.output is not None
        else summary_path.with_suffix(".calibrated.json")
    )
    output.write_text(json.dumps(corrected))
    print(json.dumps(calibration.export(), indent=2), flush=True)


def population_visibility(args: argparse.Namespace) -> None:
    """Fit the PV hurdle arm from training-recording waveform margins."""
    from .visibility import apply_visibility_model, fit_visibility_model

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text())
    with np.load(args.raw_npz) as arrays:
        model = fit_visibility_model(
            np.asarray(arrays["train_real"]),
            np.asarray(arrays["train_synthetic"]),
            threshold_db=args.threshold_db,
        )
    visible = apply_visibility_model(summary, model)
    output = (
        Path(args.output)
        if args.output is not None
        else summary_path.with_suffix(".visibility.json")
    )
    output.write_text(json.dumps(visible))
    print(json.dumps(model.export(), indent=2), flush=True)


def population_decomp_profile(args: argparse.Namespace) -> None:
    """Fit the Michael profile hierarchy from VK amplitude envelopes."""
    from data_processing.frames import meta_dict
    from data_processing.streams import iter_published_frames

    from .decomp_population import apply_decomp_profile, fit_decomp_profile

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text())
    frames = [
        frame
        for frame in iter_published_frames(args.dataset)
        if str(meta_dict(frame).get("recording_id")) == args.recording
    ]
    if len(frames) != 1:
        raise ValueError(
            f"{args.dataset}: expected one {args.recording!r} frame, got {len(frames)}"
        )
    model = fit_decomp_profile(
        frames[0],
        k_max=args.k_max,
        chunk_s=args.chunk_s,
        min_rps=args.min_rps,
        valid_fraction=args.valid_fraction,
        max_rank=args.max_rank,
    )
    fitted = apply_decomp_profile(summary, model)
    output = (
        Path(args.output)
        if args.output is not None
        else summary_path.with_suffix(".decomp-profile.json")
    )
    output.write_text(json.dumps(fitted))
    print(json.dumps(model.export(), indent=2), flush=True)


def population_decomp_dynamics(args: argparse.Namespace) -> None:
    """Fit the amplitude process and both speed laws from one decomposition."""
    from dataclasses import asdict

    from data_processing.frames import meta_dict
    from data_processing.streams import iter_published_frames

    from .decomp_dynamics import apply_decomp_dynamics, fit_decomp_dynamics

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text())
    frames = [
        frame
        for frame in iter_published_frames(args.dataset)
        if str(meta_dict(frame).get("recording_id")) == args.recording
    ]
    if len(frames) != 1:
        raise ValueError(
            f"{args.dataset}: expected one {args.recording!r} frame, got {len(frames)}"
        )
    dynamics = fit_decomp_dynamics(
        frames[0],
        recording_id=args.recording,
        k_max=args.k_max,
        min_rps=args.min_rps,
        k_min=args.k_min,
        segment_s=args.segment_s,
        max_components=args.max_components,
    )
    fitted = apply_decomp_dynamics(summary, dynamics)
    output = (
        Path(args.output)
        if args.output is not None
        else summary_path.with_suffix(".decomp-dynamics.json")
    )
    output.write_text(json.dumps(fitted))
    printable = asdict(dynamics)
    printable.pop("residual_std_db")
    print(json.dumps(printable, indent=2), flush=True)


def population_bench(args: argparse.Namespace) -> None:
    """Fit DREGON's single-motor bench recordings and merge what they identify.

    The bench is stationary and single-rotor, so it measures three things the
    4 s flight crops cannot: the per-order profile beyond the flight fit's
    measured support, the line-width law with no tracking error in it, and
    both speed exponents (five setpoints per rotor). Levels and floor shape
    are NOT merged — a bench motor has no inflow and a different room.

    After merging, the render calibration must be re-run: it is a per-order
    correction fitted against a matched render, and changing the widths by a
    factor of three invalidates it.
    """
    from .bench import apply_bench, bench_summary, fit_bench

    fit = fit_bench()
    if args.bench_output:
        Path(args.bench_output).write_text(json.dumps(bench_summary(fit), indent=1))
    print(
        json.dumps({"width_law": fit.width_law, "speed_law": fit.speed_law}, indent=2), flush=True
    )
    if args.summary:
        summary = json.loads(Path(args.summary).read_text())
        merged = apply_bench(summary, fit)
        Path(args.output).write_text(json.dumps(merged))
        print(
            f"merged into {args.output}: "
            f"{merged.get('bench_profile_orders_replaced')} orders replaced, "
            f"amp_exp {merged['rig']['amp_exp']:.3f}, floor_exp {merged['rig']['floor_exp']:.3f}, "
            f"gamma_slope {merged['rig']['gamma_slope']:.4f} Hz/order",
            flush=True,
        )
    else:
        Path(args.output).write_text(json.dumps(bench_summary(fit), indent=1))


def population_controls(args: argparse.Namespace) -> None:
    """Write the two control streams from the accepted fitted policy.

    Everything outside ``sources.noise`` is copied verbatim, so a control run
    differs from the run under test in the noise family and nothing else.
    """
    import yaml

    from .control_streams import build_diverse_policy, build_static_policy

    fitted = yaml.safe_load(Path(args.policy).read_text())
    for path, build, label in (
        (args.static_output, build_static_policy, "C1 static combs"),
        (args.diverse_output, build_diverse_policy, "C2 diverse parameters"),
    ):
        if not path:
            continue
        policy = build(fitted)
        header = (
            f"# {label} — GENERATED by `experiments.stochastic_fit.run popcontrols`\n"
            f"# from {args.policy}. Do not hand-edit: change\n"
            f"# src/experiments/stochastic_fit/controls.py, which documents why every\n"
            f"# range is where it is, and regenerate.\n"
        )
        Path(path).write_text(header + yaml.safe_dump(policy, sort_keys=False, width=100))
        arms = [f"{a.get('kind')}@{a.get('weight')}" for a in policy["sources"]["noise"]]
        print(f"{path}: {label} — arms {', '.join(arms)}", flush=True)


def population_width_spread(args: argparse.Namespace) -> None:
    """Measure the flight-to-flight line-width spread from training-clip fits."""
    from dataclasses import asdict

    from .carrier_error import apply_width_population, fit_width_population

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text())
    width = fit_width_population(
        args.fit_dir,
        clip_prefixes=tuple(args.clip_prefix),
        variant=args.variant,
        seed=args.seed,
    )
    fitted = apply_width_population(summary, width)
    output = (
        Path(args.output) if args.output is not None else summary_path.with_suffix(".width.json")
    )
    output.write_text(json.dumps(fitted))
    printable = asdict(width)
    printable.pop("clips")
    print(json.dumps(printable, indent=2), flush=True)


def population_carrier_error(args: argparse.Namespace) -> None:
    """Measure the static shaft-minus-label offset from training-clip fits."""
    from dataclasses import asdict

    from .carrier_error import apply_carrier_error, fit_carrier_error

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text())
    carrier = fit_carrier_error(
        args.fit_dir,
        clip_prefixes=tuple(args.clip_prefix),
        variant=args.variant,
        seed=args.seed,
    )
    fitted = apply_carrier_error(summary, carrier)
    output = (
        Path(args.output) if args.output is not None else summary_path.with_suffix(".carrier.json")
    )
    output.write_text(json.dumps(fitted))
    printable = asdict(carrier)
    printable.pop("per_clip_rotor_offsets")
    print(json.dumps(printable, indent=2), flush=True)


def _assert_export_matches_gate(
    source: dict[str, Any], summary: dict[str, Any], index: int
) -> None:
    """Fail unless the exported source renders what the gate rendered.

    The gate builds its parameters as ``population_ranges`` plus three rig
    scalars applied through ``params.with_``; the training stream builds them
    by reading the policy through ``StochasticNoisePool.from_dict``. Nothing
    guaranteed the two agreed, and for a while they did not: the floor speed
    exponent and the static floor share were dropped on the way out, so the
    trained-on stream had a speed-invariant comb prominence and a static floor
    the fit had measured to be absent. This check closes that gap by exercising
    the real loader path.
    """
    import numpy as np

    from data_processing.stochastic_rotor_noise import StochasticNoisePool

    pool = StochasticNoisePool.from_config(source, duration_s=2.0, sample_rate=16000)
    rig = summary["rig"]
    checks = {
        "amp_rps_exponent": (pool.amp_rps_exponent, float(rig["amp_exp"])),
        "amp_rps_exponent_floor": (pool.amp_rps_exponent_floor, float(rig["floor_exp"])),
        "floor_static_rel": (
            float(np.mean(pool.ranges.floor_static_rel)),
            float(rig["floor_static_rel"]),
        ),
    }
    wrong = {
        name: (loaded, wanted)
        for name, (loaded, wanted) in checks.items()
        if loaded is None or abs(float(loaded) - wanted) > 1e-9
    }
    if wrong:
        raise ValueError(
            f"source {index}: the exported policy does not reproduce the fitted rig — "
            + ", ".join(f"{k}: loader {v[0]!r} vs fit {v[1]!r}" for k, v in wrong.items())
        )


def population_export_policy(args: argparse.Namespace) -> None:
    """Write a stream policy whose stochastic sources carry the fitted presets.

    The mapping is ``raw_predictive.population_ranges`` — the very function the
    posterior-predictive gate renders through — so an exported policy and a
    passed gate cannot drift apart.
    """
    import yaml

    from .raw_predictive import population_ranges

    policy = yaml.safe_load(Path(args.policy).read_text())
    sources = policy["sources"]["noise"]
    for assignment in args.preset:
        index_text, _, summary_path = assignment.partition("=")
        if not summary_path:
            raise ValueError(f"--preset wants <source-index>=<summary.json>, got {assignment!r}")
        index = int(index_text)
        source = sources[index]
        if source.get("kind") != "stochastic":
            raise ValueError(f"source {index} is {source.get('kind')!r}, not stochastic")
        summary = json.loads(Path(summary_path).read_text())
        rig = summary["rig"]
        ranges = population_ranges(summary, source.get("ranges", {}))
        # THE THREE RIG SCALARS. ``render_matched_population`` — the gate's
        # renderer — applies all three through ``params.with_``; a policy that
        # carries only the line exponent silently keeps the hand-written floor
        # law and static share, which makes the comb's prominence
        # speed-invariant (the floor then follows the SAME exponent as the
        # lines) and adds a static floor the fit says is not there.
        ranges["floor_static_rel"] = [
            float(rig["floor_static_rel"]),
            float(rig["floor_static_rel"]),
        ]
        source["ranges"] = ranges
        source["amp_rps_exponent"] = float(rig["amp_exp"])
        source["amp_rps_exponent_floor"] = float(rig["floor_exp"])
        # ``n_harmonics`` stays the policy's: it caps the comb the renderer
        # sizes per clip (and may be scaled up further by
        # ``n_harmonics_range``), while the fitted profile only has to be long
        # enough to cover it.
        if len(summary["rig"]["profile_db"]) < int(source.get("n_harmonics", 0)):
            raise ValueError(
                f"source {index}: fitted profile has {len(summary['rig']['profile_db'])} "
                f"harmonics, policy asks for {source['n_harmonics']}"
            )
        source["fitted_from"] = str(summary_path)
        _assert_export_matches_gate(source, summary, index)
    output = Path(args.output)
    output.write_text(
        "# GENERATED by experiments.stochastic_fit.run popexport -- do not hand-edit.\n"
        f"# base policy: {args.policy}\n"
        + "".join(f"# preset: {item}\n" for item in args.preset)
        + yaml.safe_dump(policy, sort_keys=False, width=100)
    )
    print(f"wrote {output}", flush=True)


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
    pp = sub.add_parser("prepare-presets")
    pp.add_argument("--policy", required=True)
    pp.add_argument("--names", nargs="*", default=None)
    pp.add_argument(
        "--groups-per-preset",
        nargs="+",
        required=True,
        help="comma-joined real groups whose trajectories each preset renders on",
    )
    pp.add_argument("--per-preset", type=int, default=12)
    pp.add_argument("--seed", type=int, default=0)
    pp.set_defaults(func=prepare_presets)
    f = sub.add_parser("fit")
    f.add_argument("--variants", nargs="+", default=["family"], choices=list(VARIANTS))
    f.add_argument("--groups", nargs="*", default=None)
    f.add_argument("--clips", nargs="*", default=None)
    f.add_argument("--shard", default=None, help="i/n: take every n-th clip starting at i")
    f.add_argument("--f-max", type=float, default=None)
    f.add_argument("--k-cap", type=int, default=300)
    f.add_argument("--iters", nargs=4, type=int, default=[150, 150, 300, 60])
    f.add_argument("--tag", default="run")
    f.add_argument("--results-dir", default=str(RESULTS))
    f.add_argument(
        "--gp-std",
        type=float,
        default=None,
        help="line-drift GP prior std in dB (default: the family midpoint, 3)",
    )
    f.set_defaults(func=fit)
    r = sub.add_parser("rigfit")
    r.add_argument("--configs", nargs="+", default=["M0", "M1", "M2"], choices=list(LADDER))
    r.add_argument("--train-groups", nargs="+", required=True)
    r.add_argument("--test-groups", nargs="*", default=None)
    r.add_argument("--clips", nargs="*", default=None)
    r.add_argument("--f-max", type=float, default=None)
    r.add_argument("--k-cap", type=int, default=300)
    r.add_argument("--iters", nargs=3, type=int, default=[100, 100, 200])
    r.add_argument("--tag", default="rig")
    r.add_argument("--results-dir", default=str(RESULTS) + "_rig")
    r.set_defaults(func=rigfit)
    pf = sub.add_parser("popfit")
    pf.add_argument("--rig-config", choices=["M5", "M5g", "M5s", "M5gs"], required=True)
    pf.add_argument("--ranks", nargs="+", type=int, default=[0, 1, 2, 3])
    pf.add_argument("--train-groups", nargs="+", required=True)
    pf.add_argument("--test-groups", nargs="*", default=None)
    pf.add_argument("--clips", nargs="*", default=None)
    pf.add_argument("--f-max", type=float, default=None)
    pf.add_argument("--k-cap", type=int, default=128)
    pf.add_argument("--map-iters", nargs=3, type=int, default=[60, 60, 120])
    pf.add_argument("--vi-iters", type=int, default=200)
    pf.add_argument("--heldout-vi-iters", type=int, default=150)
    pf.add_argument("--map-lr", type=float, default=0.1)
    pf.add_argument("--vi-lr", type=float, default=0.03)
    pf.add_argument("--mc-samples", type=int, default=2)
    pf.add_argument("--iw-samples", type=int, default=16)
    pf.add_argument("--init-posterior-std", type=float, default=0.1)
    pf.add_argument("--seed", type=int, default=0)
    pf.add_argument("--tag", default="population")
    pf.add_argument("--results-dir", default=str(RESULTS) + "_population")
    pf.set_defaults(func=population_fit)
    pd = sub.add_parser("poplatent")
    pd.add_argument("--summary", required=True)
    pd.add_argument("--output", default=None)
    pd.add_argument("--seed", type=int, default=0)
    pd.add_argument("--bootstrap", type=int, default=1000)
    pd.set_defaults(func=population_latent_diagnostic)
    pg = sub.add_parser("poprawgate")
    pg.add_argument("--summary", required=True)
    pg.add_argument("--policy", required=True)
    pg.add_argument("--source-index", type=int, required=True)
    pg.add_argument("--train-groups", nargs="+", required=True)
    pg.add_argument("--test-groups", nargs="+", required=True)
    pg.add_argument("--k-max", type=int, default=64)
    pg.add_argument(
        "--observation",
        choices=["none", "recolor", "reverb", "both"],
        default="none",
    )
    pg.add_argument(
        "--classifier",
        action="store_true",
        help="enable the two-sample classifier (Michael's recordings only)",
    )
    pg.add_argument("--draws-per-clip", type=int, default=4)
    pg.add_argument("--compact", action="store_true")
    pg.add_argument("--output", default=None)
    pg.add_argument("--seed", type=int, default=0)
    pg.add_argument("--bootstrap", type=int, default=1000)
    pg.set_defaults(func=population_raw_gate)
    pc = sub.add_parser("popcalibrate")
    pc.add_argument("--summary", required=True)
    pc.add_argument("--raw-npz", required=True)
    pc.add_argument("--reference-order", type=int, default=2)
    pc.add_argument("--output", default=None)
    pc.set_defaults(func=population_calibrate)
    pv = sub.add_parser("popvisibility")
    pv.add_argument("--summary", required=True)
    pv.add_argument("--raw-npz", required=True)
    pv.add_argument("--threshold-db", type=float, default=6.0)
    pv.add_argument("--output", default=None)
    pv.set_defaults(func=population_visibility)
    dp = sub.add_parser("popdecomp")
    dp.add_argument("--summary", required=True)
    dp.add_argument("--dataset", default="decomp-frames-v2")
    dp.add_argument("--recording", default="FLY125")
    dp.add_argument("--k-max", type=int, default=64)
    dp.add_argument("--chunk-s", type=float, default=2.0)
    dp.add_argument("--min-rps", type=float, default=30.0)
    dp.add_argument("--valid-fraction", type=float, default=0.1)
    dp.add_argument("--max-rank", type=int, default=8)
    dp.add_argument("--output", default=None)
    dp.set_defaults(func=population_decomp_profile)
    dy = sub.add_parser("popdyn")
    dy.add_argument("--summary", required=True)
    dy.add_argument("--dataset", default="decomp-frames-v2")
    dy.add_argument("--recording", required=True)
    dy.add_argument("--k-max", type=int, default=64)
    dy.add_argument("--k-min", type=int, default=4)
    dy.add_argument("--min-rps", type=float, default=30.0)
    dy.add_argument("--segment-s", type=float, default=40.96)
    dy.add_argument("--max-components", type=int, default=3)
    dy.add_argument("--output")
    dy.set_defaults(func=population_decomp_dynamics)
    pe = sub.add_parser("popexport")
    pe.add_argument("--policy", required=True)
    pe.add_argument("--preset", action="append", required=True)
    pe.add_argument("--output", required=True)
    pe.set_defaults(func=population_export_policy)
    pk = sub.add_parser("popcarrier")
    pk.add_argument("--summary", required=True)
    pk.add_argument("--fit-dir", required=True)
    pk.add_argument("--variant", default="gauss")
    pk.add_argument("--clip-prefix", action="append", required=True)
    pk.add_argument("--seed", type=int, default=0)
    pk.add_argument("--output")
    pk.set_defaults(func=population_carrier_error)
    pw = sub.add_parser("popwidth")
    pw.add_argument("--summary", required=True)
    pw.add_argument("--fit-dir", required=True)
    pw.add_argument("--variant", default="gauss")
    pw.add_argument("--clip-prefix", action="append", required=True)
    pw.add_argument("--seed", type=int, default=0)
    pw.add_argument("--output")
    pw.set_defaults(func=population_width_spread)
    pb = sub.add_parser("popbench")
    pb.add_argument(
        "--summary", default=None, help="rig summary to merge the bench measurements into"
    )
    pb.add_argument("--output", required=True)
    pb.add_argument("--bench-output", default=None, help="also write the raw bench fit here")
    pb.set_defaults(func=population_bench)
    pc = sub.add_parser("popcontrols")
    pc.add_argument(
        "--policy", required=True, help="the accepted fitted policy to take the chassis from"
    )
    pc.add_argument("--static-output", default=None, help="write C1 (static combs) here")
    pc.add_argument("--diverse-output", default=None, help="write C2 (diverse parameters) here")
    pc.set_defaults(func=population_controls)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
