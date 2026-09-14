"""Multi-resolution DIAGNOSTIC rescore: PREVIOUS (coherent+Lorentzian, S2) vs
REVISED (C3) on CRUISE material of both rigs, at three analysis windows.

The question: does the two families' relative spectral fit depend on the
analysis window length? The frozen protocol's 16384-point window (1.024 s)
integrates rotor-speed WANDER into apparent line BROADENING, so a family that
only matches the long-window line width should lose ground as the window
shortens. Three settings, all at the frozen overlap ratio 16 so the exposure
weights ``a_i = hop / n_fft`` and the reported per-unique-second scale stay
comparable:

    (n_fft, hop) = (16384, 1024) | (4096, 256) | (2048, 128)

Everything else is the frozen protocol, reused verbatim from the evaluator:

* observation  -- ``revised_eval.window_periodogram`` (periodic Hann, no pad)
* band         -- ``revised_eval.observation_band`` (30-7900 Hz)
* previous M   -- ``revised_eval.read_export`` + ``aggregate_nuisance``
                  + ``predicted_m`` (the HISTORICAL-FORWARD baseline adapter,
                  carrying ``revised_eval.HISTORICAL_FORWARD_LABEL``)
* revised M    -- ``revised_phase.predict_spectrum(..., mode="prior")``
* risk         -- ``revised_eval.marginal_frame_nll`` + ``FrameScore``
                  + ``composite_score`` + ``unnormalized_weighted_total``

THIS IS A DIAGNOSTIC, NOT A GATE. It changes no campaign artefact, no manifest
and no frozen protocol. Two deliberate departures from the C3 scorecard, stated
wherever a number from here is read:

1. the scored supports are the A/B-comparison cruise supports (which for
   Michael's FLY125 and for DREGON room 2 are the PREVIOUS model's own fit
   supports), not the scorecard's held-out cohort windows -- so the previous
   arm is scored IN SAMPLE here and is flattered relative to the scorecard;
2. no baseline-variability calibration, no bootstrap, no temperature.

Run (CPU only):

    PYTHONPATH="$PWD/src:$PWD/scripts" python scripts/_multires_rescore.py
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.stochastic_fit import revised_eval as RE
from experiments.stochastic_fit import revised_phase as RP

OUT = Path("results/multires_rescore")

#: The three analysis settings. Overlap ratio n_fft/hop == 16 throughout.
SETTINGS: tuple[tuple[int, int], ...] = ((16384, 1024), (4096, 256), (2048, 128))

#: Revised candidate: the C3 exports (best existing revised fit).
C3_EXPORT = {
    "michaels": Path(
        ".worktrees/revised-phase-preflight/omnirun-outputs/"
        "revised-phase-c3-fits-ad2c57/results/revised_phase/c3/michaels.json"
    ),
    "dregon": Path(
        ".worktrees/revised-phase-preflight/omnirun-outputs/"
        "revised-phase-c3-fits-ad2c57/results/revised_phase/c3/dregon.json"
    ),
}

#: Previous model: the stage-2 coherent+Lorentzian exports, with the DECLARED
#: provenance the campaign manifests use for the legacy file that has no
#: ``data`` block (docs/revised-phase-baseline-manifest-v2.json).
PREV_EXPORT = {
    "michaels": dict(
        path=Path("results/S2/cruise_8clip.json"),
        family="raw",
        regime="cruise",
        declared=dict(
            recordings=["FLY125"],
            starts_s=[16.0, 32.0, 48.0, 64.0, 96.0, 112.0, 128.0, 144.0],
            seconds=16.0,
            dataset="michaels-frames",
            version=None,
            rps_key="rps",
        ),
    ),
    "dregon": dict(
        path=Path("results/S2/dregon_room2_cruise_refined.json"),
        family="refined",
        regime="cruise",
        declared=None,  # this export carries its own `data` block
    ),
}

#: RAW telemetry per rig -- never the default `rps` for DREGON.
RAW_KEY = {"michaels": "rps", "dregon": "motors_command"}
DATASET = {"michaels": "michaels-frames", "dregon": "DREGON-frames"}

#: The scored cruise supports. DREGON: the two room-2 recordings of the
#: existing A/B comparison, windows taken verbatim from
#: docs/explainers/revised-compare/revcmp_dregon.json. Michael: the FLY125
#: cruise windows [32,48), [64,80), [96,112) s.
#: ``seconds`` is the DECLARED support; ``--support`` truncates it (cost).
FULL_CLIPS: tuple[dict[str, Any], ...] = (
    dict(rig="michaels", recording="FLY125", start_s=32.0, seconds=16.0, rank=0),
    dict(rig="michaels", recording="FLY125", start_s=64.0, seconds=16.0, rank=1),
    dict(rig="michaels", recording="FLY125", start_s=96.0, seconds=16.0, rank=2),
    dict(
        rig="dregon",
        recording="free-flight_nosource_room2",
        start_s=1512727417.2050455,
        seconds=8.0,
        rank=0,
    ),
    dict(
        rig="dregon",
        recording="hovering_nosource_room2",
        start_s=1511903913.3944898,
        seconds=8.0,
        rank=1,
    ),
)

RIGS = ("michaels", "dregon")


# ── provenance ──────────────────────────────────────────────────────────────


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception as exc:  # pragma: no cover - provenance must never abort a run
        return f"unavailable: {exc}"


def dataset_provenance(dataset: str) -> dict[str, Any]:
    """The dataset version actually opened, via the evaluator's own resolver."""
    try:
        from stochastic_fit_revised_eval import resolved_dataset_version

        return resolved_dataset_version(dataset, None)
    except Exception as exc:
        return dict(dataset=dataset, declared_version=None, resolved_version=f"unavailable: {exc}")


# ── models ──────────────────────────────────────────────────────────────────


def prev_model_params(rig: str, recording: str) -> RE.ModelParams:
    """The previous model's rig parameter set for ONE recording.

    Exactly the evaluator's ``match: identity`` baseline route: this
    recording's own fitted clips, aggregated in linear power with each clip's
    ``power_scale`` folded in, clip-local latents dropped
    (``revised_eval.aggregate_nuisance``).
    """
    cfg = PREV_EXPORT[rig]
    bundle = RE.read_export(
        cfg["path"], family=cfg["family"], regime=cfg["regime"], declared=cfg["declared"]
    )
    pairs = bundle.clips_of(recording)
    if not pairs:
        raise SystemExit(
            f"{cfg['path']}: does not name {recording!r} (it names {list(bundle.recordings)})"
        )
    ids = [cid for cid, _ in pairs]
    mp = RE.aggregate_nuisance(bundle, ids, label=f"{cfg['family']}:{cfg['regime']}:identity")
    mp.source.update(for_recording=recording, baseline_route="identity")
    return mp


# ── one (clip, setting, model) row ──────────────────────────────────────────


def score_clip(
    window: RE.Window,
    clip: Any,
    *,
    n_fft: int,
    hop: int,
    model: str,
    prev_mp: RE.ModelParams | None,
    candidate: RE.CandidateExport | None,
) -> tuple[RE.FrameScore, dict[str, Any]]:
    """One clip's frame terms under one model at one analysis setting."""
    pg = RE.window_periodogram(clip, n_fft=int(n_fft), hop=int(hop))
    band = RE.observation_band(pg.freqs)
    mics = int(clip.audio.shape[0])
    power = np.asarray(pg.power, dtype=np.float64)[:mics]
    t0 = time.time()
    if model == "previous":
        assert prev_mp is not None
        m = RE.predicted_m(prev_mp, pg, n_mics=mics)
    elif model == "revised":
        assert candidate is not None
        m = np.asarray(
            RP.predict_spectrum(
                candidate.summary, clip, n_fft=int(n_fft), hop=int(hop), mode="prior"
            ),
            dtype=np.float64,
        )
        if m.shape[0] < mics:
            raise SystemExit(
                f"predict_spectrum returned {m.shape[0]} mics, the frozen set needs {mics}"
            )
        m = m[:mics]
    else:  # pragma: no cover
        raise SystemExit(f"unknown model {model!r}")
    seconds_model = time.time() - t0
    if m.shape != power.shape:
        raise SystemExit(f"model {model}: predicted {m.shape} against observed {power.shape}")
    nll, cells = RE.marginal_frame_nll(power, m, band)
    fs = RE.FrameScore(
        window, RE.frame_times_on_clock(window, pg), nll, cells, int(n_fft), int(hop)
    )
    detail = dict(
        n_mics=mics,
        n_band_bins=int(band.sum()),
        df_hz=float(pg.df),
        model_seconds=float(seconds_model),
    )
    return fs, detail


def row_from_items(items: list[RE.FrameScore]) -> dict[str, Any]:
    """Risk of a set of frame scores, in BOTH conventions the evaluator uses."""
    comp = RE.composite_score(items)
    unn = RE.unnormalized_weighted_total(items)
    return dict(
        risk_unnormalised=float(comp["weighted_nats"]),
        risk_per_unique_second=float(comp["score"]),
        unique_seconds=float(comp["unique_seconds"]),
        exposure_h=float(unn["exposure_h"]),
        sum_weights=float(unn["exposure_h"]) / float(next(iter(comp["cells_per_frame"]))),
        risk_per_exposure_cell=float(comp["weighted_nats"]) / float(unn["exposure_h"]),
        risk_per_band_cell_second=(
            None if comp["score_per_band_cell"] is None else float(comp["score_per_band_cell"])
        ),
        n_frames=int(comp["n_frames"]),
        cells_per_frame=[int(v) for v in comp["cells_per_frame"]],
        hop_window_factor=[float(v) for v in comp["exposure"]["hop_window_factor"]],
    )


# ── the grid ────────────────────────────────────────────────────────────────


def run(clips_spec: list[dict[str, Any]], settings: list[tuple[int, int]]) -> dict[str, Any]:
    for p in C3_EXPORT.values():
        RE.read_candidate_export(p)  # refuses legacy / C1 / malformed before any work
    candidates = {rig: RE.read_candidate_export(C3_EXPORT[rig]) for rig in RIGS}

    prev_mps: dict[tuple[str, str], RE.ModelParams] = {}
    loaded: dict[str, tuple[RE.Window, Any]] = {}
    for spec in clips_spec:
        rig, rec = spec["rig"], spec["recording"]
        if (rig, rec) not in prev_mps:
            prev_mps[(rig, rec)] = prev_model_params(rig, rec)
        window = RE.Window(rec, float(spec["start_s"]), float(spec["seconds"]), regime="cruise")
        key = RE.assert_raw_reference(RAW_KEY[rig])
        clip = RE.load_window(
            window, dataset=DATASET[rig], version=None, channels=None, rps_key=key
        )
        loaded[window.key] = (window, clip)
        spec["window_key"] = window.key
        spec["clip_id"] = clip.clip_id
        spec["n_mics"] = int(clip.audio.shape[0])
        spec["n_samples"] = int(clip.audio.shape[1])
        spec["mean_rps"] = [round(float(v), 3) for v in np.asarray(clip.rps).mean(axis=1)]

    per_clip: list[dict[str, Any]] = []
    items: dict[tuple[str, int, int, str], list[RE.FrameScore]] = {}
    t_start = time.time()
    for n_fft, hop in settings:
        for spec in clips_spec:
            window, clip = loaded[spec["window_key"]]
            for model in ("previous", "revised"):
                fs, detail = score_clip(
                    window,
                    clip,
                    n_fft=n_fft,
                    hop=hop,
                    model=model,
                    prev_mp=prev_mps[(spec["rig"], spec["recording"])],
                    candidate=candidates[spec["rig"]],
                )
                items.setdefault((spec["rig"], n_fft, hop, model), []).append(fs)
                row = dict(
                    rig=spec["rig"],
                    recording=spec["recording"],
                    clip=spec["window_key"],
                    clip_id=spec["clip_id"],
                    start_s=float(spec["start_s"]),
                    seconds=float(spec["seconds"]),
                    n_fft=int(n_fft),
                    hop=int(hop),
                    overlap_ratio=int(n_fft // hop),
                    model=model,
                    telemetry_key=RAW_KEY[spec["rig"]],
                    **detail,
                    **row_from_items([fs]),
                )
                per_clip.append(row)
                print(
                    f"[{n_fft:>5}/{hop:>4}] {spec['window_key']:<46} {model:<8} "
                    f"risk/s {row['risk_per_unique_second']:>14.2f}  "
                    f"unnorm {row['risk_unnormalised']:>16.2f}  "
                    f"H {row['exposure_h']:>12.1f}  frames {row['n_frames']:>4}  "
                    f"({detail['model_seconds']:.1f} s)",
                    flush=True,
                )
    wall = time.time() - t_start

    # ── per-rig pooled risk and the paired previous-minus-revised difference ──
    pooled: list[dict[str, Any]] = []
    for rig in RIGS:
        for n_fft, hop in settings:
            have = {
                model: items.get((rig, n_fft, hop, model), []) for model in ("previous", "revised")
            }
            if not have["previous"] or not have["revised"]:
                continue
            prev_row = row_from_items(have["previous"])
            rev_row = row_from_items(have["revised"])
            clip_deltas = [
                dict(
                    clip=p["clip"],
                    previous=p["risk_per_unique_second"],
                    revised=r["risk_per_unique_second"],
                    delta_previous_minus_revised=p["risk_per_unique_second"]
                    - r["risk_per_unique_second"],
                )
                for p, r in zip(
                    [
                        row
                        for row in per_clip
                        if row["rig"] == rig
                        and row["n_fft"] == n_fft
                        and row["model"] == "previous"
                    ],
                    [
                        row
                        for row in per_clip
                        if row["rig"] == rig and row["n_fft"] == n_fft and row["model"] == "revised"
                    ],
                    strict=True,
                )
            ]
            pooled.append(
                dict(
                    rig=rig,
                    n_fft=int(n_fft),
                    hop=int(hop),
                    window_ms=1000.0 * n_fft / float(RE.SR),
                    overlap_ratio=int(n_fft // hop),
                    previous=prev_row,
                    revised=rev_row,
                    delta_per_unique_second=prev_row["risk_per_unique_second"]
                    - rev_row["risk_per_unique_second"],
                    delta_unnormalised=prev_row["risk_unnormalised"] - rev_row["risk_unnormalised"],
                    per_clip_delta=clip_deltas,
                )
            )

    provenance = dict(
        git_head=git_head(),
        band_hz=[float(RE.OBS_F_MIN), float(RE.OBS_F_MAX)],
        sample_rate=int(RE.SR),
        window_shape="periodic Hann (np.hanning(n_fft+1)[:n_fft]), frames fully inside the clip",
        settings=[dict(n_fft=int(n), hop=int(h), overlap_ratio=int(n // h)) for n, h in settings],
        telemetry_keys=dict(RAW_KEY),
        datasets={rig: dataset_provenance(DATASET[rig]) for rig in RIGS},
        mics={rig: sorted({s["n_mics"] for s in clips_spec if s["rig"] == rig}) for rig in RIGS},
        revised_exports={
            rig: dict(RE.artifact_digest(C3_EXPORT[rig]), model_family=candidates[rig].model_family)
            for rig in RIGS
        },
        previous_exports={
            rig: dict(
                RE.artifact_digest(PREV_EXPORT[rig]["path"]),
                family=PREV_EXPORT[rig]["family"],
                declared_provenance=PREV_EXPORT[rig]["declared"] is not None,
            )
            for rig in RIGS
        },
        previous_observation_law=RE.HISTORICAL_FORWARD_LABEL,
        revised_observation_law=(
            "revised_phase.predict_spectrum(mode='prior'): the exact moving-window kernel on the "
            "raw telemetry plus the learned bias law; never the stored training MAP path"
        ),
        reused_functions=[
            "revised_eval.window_periodogram",
            "revised_eval.observation_band",
            "revised_eval.load_window",
            "revised_eval.read_export",
            "revised_eval.aggregate_nuisance",
            "revised_eval.predicted_m",
            "revised_phase.predict_spectrum",
            "revised_eval.marginal_frame_nll",
            "revised_eval.FrameScore",
            "revised_eval.exposure_weights (via composite_score)",
            "revised_eval.composite_score",
            "revised_eval.unnormalized_weighted_total",
        ],
        import_provenance=RE.import_provenance(),
        caveats=[
            "DIAGNOSTIC ONLY: not a gate, not a campaign artefact, no protocol change.",
            "The scored supports are the A/B-comparison cruise supports, which ARE the previous "
            "model's own fit supports for both rigs -- the previous arm is scored IN SAMPLE and "
            "is therefore flattered relative to the C3 scorecard's held-out windows.",
            "No baseline-variability calibration, no bootstrap interval, no composite temperature.",
            RE.ADAPTIVE_SELECTION_CAVEAT,
        ],
        wall_seconds=float(wall),
    )
    return dict(
        schema="multires-rescore-diagnostic/1",
        provenance=provenance,
        clips=clips_spec,
        per_clip=per_clip,
        pooled=pooled,
    )


# ── the table ───────────────────────────────────────────────────────────────


def table_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Multi-resolution diagnostic rescore: previous vs revised (C3), cruise")
    lines.append("")
    lines.append(
        "Composite spectral risk, nats per second of unique material (the evaluator's own "
        "`composite_score` convention: lower is better). `delta = previous - revised`, so "
        "**positive means the revised model fits better**. Diagnostic only: these supports are "
        "the previous model's own fit supports, so the previous arm is scored in sample."
    )
    lines.append("")
    lines.append(
        "| rig | window | n_fft / hop | previous | revised | delta (prev - rev) | "
        "delta / |prev| | clips | frames | unique s |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload["pooled"]:
        p, r = row["previous"], row["revised"]
        rel = row["delta_per_unique_second"] / max(abs(p["risk_per_unique_second"]), 1e-12)
        lines.append(
            f"| {row['rig']} | {row['window_ms']:.0f} ms | {row['n_fft']} / {row['hop']} | "
            f"{p['risk_per_unique_second']:,.0f} | {r['risk_per_unique_second']:,.0f} | "
            f"{row['delta_per_unique_second']:+,.0f} | {rel:+.4%} | "
            f"{len(row['per_clip_delta'])} | {p['n_frames']} | {p['unique_seconds']:.3f} |"
        )
    lines.append("")
    lines.append("## Unnormalised risk (sum_i a_i * (I/M + log M), a_i = hop/n_fft)")
    lines.append("")
    lines.append("| rig | window | previous | revised | delta | exposure H |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in payload["pooled"]:
        p, r = row["previous"], row["revised"]
        lines.append(
            f"| {row['rig']} | {row['window_ms']:.0f} ms | {p['risk_unnormalised']:,.0f} | "
            f"{r['risk_unnormalised']:,.0f} | {row['delta_unnormalised']:+,.0f} | "
            f"{p['exposure_h']:,.0f} |"
        )
    lines.append("")
    lines.append("## Per-clip spread of the paired difference (previous - revised, nats/s)")
    lines.append("")
    clip_keys: list[str] = []
    for row in payload["pooled"]:
        for d in row["per_clip_delta"]:
            if d["clip"] not in clip_keys:
                clip_keys.append(d["clip"])
    lines.append("| rig | window | " + " | ".join(clip_keys) + " |")
    lines.append("|---|---|" + "---:|" * len(clip_keys))
    for row in payload["pooled"]:
        by = {d["clip"]: d["delta_previous_minus_revised"] for d in row["per_clip_delta"]}
        cells = [f"{by[k]:+,.0f}" if k in by else "" for k in clip_keys]
        lines.append(f"| {row['rig']} | {row['window_ms']:.0f} ms | " + " | ".join(cells) + " |")
    lines.append("")
    prov = payload["provenance"]
    lines.append(
        f"Band {prov['band_hz'][0]:.0f}-{prov['band_hz'][1]:.0f} Hz, {prov['sample_rate']} Hz, "
        f"overlap ratio 16 at every setting, git `{prov['git_head'][:12]}`."
    )
    lines.append("")
    for c in prov["caveats"]:
        lines.append(f"- {c}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--support",
        type=float,
        default=None,
        help="truncate every declared support to this many seconds (cost control)",
    )
    ap.add_argument(
        "--clips-per-rig",
        type=int,
        default=None,
        help="keep only the first N declared clips of each rig (cost control)",
    )
    ap.add_argument(
        "--settings",
        default=",".join(f"{n}:{h}" for n, h in SETTINGS),
        help="comma-separated n_fft:hop list",
    )
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument(
        "--probe",
        action="store_true",
        help="time ONE clip at ONE setting for both arms and print the grid projection",
    )
    args = ap.parse_args()

    settings = [
        (int(a), int(b)) for a, b in (s.split(":") for s in args.settings.split(",") if s.strip())
    ]
    for n_fft, hop in settings:
        if n_fft // hop != 16 or n_fft % hop:
            raise SystemExit(f"setting {n_fft}:{hop} breaks the frozen overlap ratio of 16")

    specs = [dict(s) for s in FULL_CLIPS]
    if args.clips_per_rig is not None:
        specs = [s for s in specs if int(s["rank"]) < int(args.clips_per_rig)]
    if args.support is not None:
        for s in specs:
            s["declared_seconds"] = float(s["seconds"])
            s["seconds"] = float(min(float(args.support), float(s["seconds"])))
            s["truncated"] = s["seconds"] < s["declared_seconds"]

    if args.probe:
        probe = [s for s in specs if s["rig"] == "dregon"][:1]
        t0 = time.time()
        run(probe, settings[:1])
        dt = time.time() - t0
        audio_probe = float(probe[0]["seconds"])
        audio_total = sum(float(s["seconds"]) for s in specs) * len(settings)
        rate = dt / max(audio_probe, 1e-9)
        print(
            f"\nPROBE: 1 clip ({audio_probe:g} s audio) x 1 setting, both arms, {dt:.1f} s wall "
            f"=> {rate:.1f} s CPU per audio-second per setting\n"
            f"PROJECTION for the requested grid ({len(specs)} clips, {len(settings)} settings, "
            f"{audio_total:g} audio-seconds of work): {rate * audio_total / 60.0:.1f} min",
            flush=True,
        )
        return

    payload = run(specs, settings)
    payload["provenance"]["cost_control"] = dict(
        measured_rate_s_cpu_per_audio_second_per_setting=49.9,
        full_grid_audio_seconds=sum(float(s["seconds"]) for s in FULL_CLIPS) * len(settings),
        full_grid_projection_minutes=round(
            49.9 * sum(float(s["seconds"]) for s in FULL_CLIPS) * len(settings) / 60.0, 1
        ),
        clips_per_rig=args.clips_per_rig,
        support_seconds=args.support,
        run_audio_seconds=sum(float(s["seconds"]) for s in specs) * len(settings),
        rule=(
            "revised_phase.predict_spectrum on CPU costs ~50 s per audio-second per setting "
            "(measured: 2 s DREGON clip, both arms, 16384/1024 -> 99.8 s wall, of which 97.6 s "
            "is predict_spectrum and 1.3 s is predicted_m). The full declared grid (5 clips, "
            "64 audio-seconds, 3 settings = 192 audio-seconds of work) projects to ~160 min, and "
            "even the prescribed minimum cut of 2 clips per rig at declared support lengths "
            "(2x16 s + 2x8 s = 48 audio-seconds, 144 of work) projects to ~120 min. To land "
            "inside the ~20 min budget the run keeps 2 clips per rig (dropping Michael's "
            "[96,112) s window) AND truncates each support to its first 2.0 s, which is "
            "4 clips x 2 s x 3 settings = 24 audio-seconds of work. Nothing else was cut: both "
            "rigs, both models, all three settings and the full 30-7900 Hz band survive."
        ),
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    RE.write_json(out / "scores.json", payload)
    md = table_md(payload)
    (out / "table.md").write_text(md)
    digest = hashlib.sha256((out / "scores.json").read_bytes()).hexdigest()
    print()
    print(md)
    print(f"wrote {out / 'scores.json'} (sha256 {digest[:12]})")
    print(f"wrote {out / 'table.md'}")
    print(f"total wall {payload['provenance']['wall_seconds']:.1f} s")


if __name__ == "__main__":
    main()
