"""Noise-model-v2 side measurement: the PREVIOUS-generation rotor-noise model's
Whittle (mean-prediction composite) risk on a SHORT front-end STFT.

One question, no study: the legacy descriptive model — the coherent needle plus
Lorentzian pedestal family of ``experiments.stochastic_fit.model``
(``Spec.fit_coherence`` / ``Spec.needle_window_shape``) — was always scored at
the frozen NFFT 16384 / hop 1024 front end. Does a SHORT analysis window change
how far it sits from a non-parametric floor?

The objective is the FROZEN one and is not re-implemented here: the expected
periodogram comes from ``revised_eval.predicted_m`` (the historical-forward
baseline adapter) and the risk from ``revised_eval.marginal_frame_nll`` +
``FrameScore`` + ``composite_score`` — positive exposure-weighted
``sum_i a_i (I_i/M_i + log M_i)`` on 30-7900 Hz, per unique observed second,
with the ``hop/n_fft`` exposure factor and the duplicate-frame rule. ONLY the
``(n_fft, hop)`` pair changes between rows.

Two supports, each with the export the campaign actually fitted for it:

* **DREGON room2 cruise** — the five frozen 4 s scoring windows of
  ``docs/revised-phase-baseline-manifest-v2.json``, scored with the
  **BENCH-fitted** export ``results/S1/bayes_rig.json`` (Stage-1 rig fit over
  the 20 single-motor bench cells ``Motor{1-4}_{50..90}``). A bench rig fit has
  ``R = 1``, so its tied profile is ONE shape: it is tiled across the flight's
  four rotors, which is what "one rig file gives four rotors the same shape"
  means in ``scripts/_dregon_transfer.py``. The bench is single-channel
  (``channel 7``, ``mic_gain_db = [[0]]``, ``spec.mic_floor = False``), so it
  supplies no per-microphone pattern — a stated limitation of the transfer, not
  a modelling choice. Its absolute level and its room floor are the bench's.
* **Michael's FLY124** — the five held-out windows the frozen evaluator
  resolved, scored with the **FLIGHT-fitted** export
  ``results/S2/cruise_8clip_refined.json`` (FLY125, 8 x 16 s, ``rps_refined``),
  the manifest's ``refined``/``cruise`` family for the ``michaels_fly124``
  cohort. FLY124 is named by no export, so this is the manifest's declared
  extrapolation (``require_identity_coverage: false``) and the same aggregate
  is used on the standby and ramp supports too.

The floor is the study's own oracle (``results/noise_v2/criteria/findings.md``
section 2 / 3.1): ``M`` is the Welch MEAN PERIODOGRAM of a legal DISJOINT real
segment of the SAME recording at the SAME NFFT, its start chosen on a 0.25 s
grid to minimise the per-rotor mean-speed mismatch, because an unmatched
segment does not read as a floor. The achieved mismatch travels with every row.

Supports, guards, the oracle rule and the loader are imported from
``scripts/noise_v2_likelihood_window.py`` rather than restated, so the two
scripts cannot drift.

Idempotent: writes ``results/noise_v2/short_whittle/short_whittle.json`` and
``findings.md``; re-running recomputes the same numbers and overwrites both.
``--reuse`` keeps every (rig, arm, setting) block already in the JSON and only
computes the missing ones.

    python scripts/noise_v2_short_whittle.py
    python scripts/noise_v2_short_whittle.py --settings 2048:512 --supports-per-rig 1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.stochastic_fit import revised_eval as RE
from experiments.stochastic_fit.data import periodogram

TOPIC = "short_whittle"
OUT_DEFAULT = Path("results/noise_v2") / TOPIC

#: ``(n_fft, hop)`` at 16 kHz, periodic Hann. The first three are the SHORT
#: front ends under test (25 % hop); the last is the frozen legacy reference.
SETTINGS_DEFAULT: tuple[tuple[int, int], ...] = (
    (2048, 512),
    (1024, 256),
    (4096, 1024),
    (16384, 1024),
)

#: The legacy export per rig, with the route it is read through.
EXPORTS: dict[str, dict[str, Any]] = {
    "dregon": dict(
        path="results/S1/bayes_rig.json",
        label="bench-fitted (S1 rig fit, 20 single-motor bench cells)",
        family="bench",
        regime="bench",
        route="aggregate",
        tile_profile=True,
        declared=dict(
            recordings=["Motor1", "Motor2", "Motor3", "Motor4"],
            starts_s=[0.0] * 20,
            seconds=9.0,
            dataset="DREGON-bench",
            version=None,
            rps_key=None,
            note=(
                "DECLARED: legacy export with no data block. The bench cells are whole "
                "single-motor recordings; starts_s/seconds are the nominal steady-span cap "
                "(stage1_bayes.BENCH_SECONDS = 9.0, bench_span is data-dependent per cell) and "
                "are provenance only — no held-out guard is involved, the bench material shares "
                "no recording with any flight support."
            ),
        ),
    ),
    "michaels": dict(
        path="results/S2/cruise_8clip_refined.json",
        label="flight-fitted (S2 FLY125 cruise, 8 x 16 s, rps_refined)",
        family="refined",
        regime="cruise",
        route="aggregate",
        tile_profile=False,
        declared=None,
    ),
}

ARM_MODEL = "legacy_model"
ARM_ORACLE = "oracle_np"


def criteria_module() -> Any:
    """``scripts/noise_v2_likelihood_window.py`` as a module: one frozen cohort."""
    path = Path(__file__).resolve().parent / "noise_v2_likelihood_window.py"
    spec = importlib.util.spec_from_file_location("noise_v2_likelihood_window", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LW = criteria_module()


def scored_supports(rig: str, limit: int | None) -> list[Any]:
    """The frozen held-out supports of one rig, in the study's own order."""
    out = []
    if rig == "dregon":
        for rec, start, dur in LW.DREGON_SCORED:
            out.append(LW.Support(rig, rec, start, dur, "cruise", "scored", "frozen manifest v2"))
    else:
        for rec, start, dur, regime in LW.MICHAELS_SCORED:
            out.append(
                LW.Support(rig, rec, start, dur, regime, "scored", "frozen evaluator resolution")
            )
    return out if limit is None else out[: int(limit)]


def legacy_params(rig: str, n_rotors: int) -> RE.ModelParams:
    """The rig's legacy export on the evaluator's own nuisance route.

    ``read_export`` + ``aggregate_nuisance``: per-clip ``power_scale`` folded
    in, levels and patterns averaged in linear power, clip-local latents
    (drift draws, carrier correction, per-clip level) dropped. The bench rig
    fit carries one rotor, so its profile is tiled onto the flight's rotors;
    every other field is already rig-level.
    """
    cfg = EXPORTS[rig]
    bundle = RE.read_export(
        cfg["path"], family=cfg["family"], regime=cfg["regime"], declared=cfg["declared"]
    )
    mp = RE.aggregate_nuisance(
        bundle, list(bundle.clip_ids), label=f"{cfg['family']}:{cfg['regime']}", extrapolated=True
    )
    params = dict(mp.params)
    profile = np.atleast_2d(np.asarray(params["profile_db"], dtype=np.float64))
    tiled = bool(cfg["tile_profile"]) and profile.shape[0] < int(n_rotors)
    if tiled:
        params["profile_db"] = np.repeat(profile[:1], int(n_rotors), axis=0).tolist()
    source = dict(mp.source)
    source.update(
        export=str(cfg["path"]),
        export_label=cfg["label"],
        baseline_route=cfg["route"],
        profile_tiled_to_rotors=int(n_rotors) if tiled else None,
        profile_rotors_in_export=int(profile.shape[0]),
    )
    return RE.ModelParams(params=params, spec=mp.spec, source=source)


def level_offset_db(power: np.ndarray, model: np.ndarray, band: np.ndarray) -> float:
    """Median in-band dB offset of the real periodogram over the model.

    Reported next to every model row because a composite risk in absolute units
    mixes a level error with a shape error, and a reader must be able to tell
    which one is talking.
    """
    real = np.median(np.asarray(power, dtype=np.float64)[..., band].mean(axis=1))
    pred = np.median(np.asarray(model, dtype=np.float64)[..., band].mean(axis=1))
    return float(10.0 * np.log10(max(real, 1e-300) / max(pred, 1e-300)))


def ratio_diagnostics(power: np.ndarray, model: np.ndarray, band: np.ndarray) -> dict[str, float]:
    """Where the ``I/M`` part of the risk comes from: the bulk or a thin tail.

    A mean-prediction risk is unbounded above in ``I/M``, so a model whose
    lines sit a few bins off can be convicted by a handful of cells. The median
    ratio and the share of the total ``I/M`` carried by the worst 0.1 % of
    in-band cells say which of the two is being measured.
    """
    ratio = (
        np.asarray(power, dtype=np.float64)[..., band]
        / np.maximum(np.asarray(model, dtype=np.float64)[..., band], 1e-300)
    ).ravel()
    k = max(int(ratio.size * 0.001), 1)
    top = np.partition(ratio, ratio.size - k)[ratio.size - k :]
    total = float(ratio.sum())
    return dict(
        im_median=float(np.median(ratio)),
        im_mean=float(ratio.mean()),
        im_top_0p1pct_share=float(top.sum() / max(total, 1e-300)),
    )


def run(
    *,
    rigs: list[str],
    settings: list[tuple[int, int]],
    supports_per_rig: int | None,
    n_mics: int,
    cached: dict[str, Any] | None,
    f_min: float = RE.OBS_F_MIN,
) -> dict[str, Any]:
    payload: dict[str, Any] = dict(
        schema=f"noise-v2-{TOPIC}/1",
        band_hz=[max(float(LW.BAND_HZ[0]), float(f_min)), float(LW.BAND_HZ[1])],
        objective=(
            "frozen composite spectral risk: revised_eval.marginal_frame_nll + FrameScore + "
            "composite_score, positive exposure-weighted sum a_i (I_i/M_i + log M_i) on "
            "30-7900 Hz, per unique observed second, hop/n_fft exposure factor and the "
            "duplicate-frame rule; only (n_fft, hop) varies"
        ),
        arms={
            ARM_MODEL: (
                "legacy descriptive model (coherent needle + Lorentzian pedestal) through "
                f"revised_eval.predicted_m ({RE.HISTORICAL_FORWARD_LABEL})"
            ),
            ARM_ORACLE: (
                "Welch mean periodogram of a speed-matched legal disjoint real segment of the "
                "same recording at the same NFFT (criteria findings section 3.1)"
            ),
        },
        exports={r: dict(EXPORTS[r]) for r in rigs},
        settings=[dict(n_fft=int(n), hop=int(h)) for n, h in settings],
        supports=[],
        rows=[],
        pooled=[],
        timing=[],
    )
    old_rows = list((cached or {}).get("rows") or [])
    old_pooled = list((cached or {}).get("pooled") or [])
    old_timing = list((cached or {}).get("timing") or [])
    have = {(p["rig"], p["arm"], p["n_fft"], p["hop"]) for p in old_pooled}
    for rig, cfg in ((cached or {}).get("exports") or {}).items():
        if rig in payload["exports"] and cfg.get("nuisance_source"):
            payload["exports"][rig]["nuisance_source"] = cfg["nuisance_source"]
    cached_supports = {
        (s["rig"], s["scored"]["key"]): s for s in ((cached or {}).get("supports") or [])
    }

    for rig in rigs:
        supports = scored_supports(rig, supports_per_rig)
        todo = [
            (n, h)
            for n, h in settings
            if not {(rig, ARM_MODEL, n, h), (rig, ARM_ORACLE, n, h)} <= have
        ]
        for n_fft, hop in settings:
            if (n_fft, hop) in todo:
                continue
            for p in old_pooled:
                if (p["rig"], p["n_fft"], p["hop"]) == (rig, n_fft, hop):
                    payload["pooled"].append(p)
            for r in old_rows:
                if (r["rig"], r["n_fft"], r["hop"]) == (rig, n_fft, hop):
                    payload["rows"].append(r)
            for t in old_timing:
                if (t["rig"], t["n_fft"], t["hop"]) == (rig, n_fft, hop):
                    payload["timing"].append(t)
            print(f"[{rig} {n_fft}/{hop}] reused from cache")
        if not todo:
            continue

        clips: dict[str, Any] = {}
        for scored in supports:
            rec = LW.C.load_recording(LW.DATASET[rig], scored.recording, None, LW.RAW_KEY[rig])
            in_regime = scored.regime != "ramp"
            found = LW.disjoint_segment(rec, rig, scored, min_seconds=2.048, in_regime=in_regime)
            ref, ref_detail = (None, None) if found is None else found
            clip = LW.load(scored, rps_key=LW.RAW_KEY[rig], channels=None)
            ref_clip = None if ref is None else LW.load(ref, rps_key=LW.RAW_KEY[rig], channels=None)
            clips[scored.window.key] = (scored, clip, ref_clip)
            payload["supports"].append(
                dict(
                    rig=rig,
                    scored=scored.as_dict(),
                    oracle_reference=None if ref is None else ref.as_dict(),
                    oracle_match=ref_detail,
                    n_mics=int(min(int(n_mics), int(clip.audio.shape[0]))),
                    n_rotors=int(clip.rps.shape[0]),
                )
            )
        rotors = max(int(c.rps.shape[0]) for _, c, _ in clips.values())
        mp = legacy_params(rig, rotors)
        payload["exports"][rig]["nuisance_source"] = dict(mp.source)

        for n_fft, hop in todo:
            wall0 = time.time()
            model_seconds = 0.0
            items: dict[str, list[RE.FrameScore]] = {ARM_MODEL: [], ARM_ORACLE: []}
            for scored, clip, ref_clip in clips.values():
                mics = min(int(n_mics), int(clip.audio.shape[0]))
                pg = periodogram(clip, n_fft=n_fft, hop=hop)
                band = RE.observation_band(pg.freqs, f_min=max(RE.OBS_F_MIN, f_min))
                power = np.asarray(pg.power, dtype=np.float64)[:mics]
                t0 = time.time()
                m_model = RE.predicted_m(mp, pg, n_mics=mics)
                dt = time.time() - t0
                model_seconds += dt
                models: dict[str, np.ndarray | None] = {ARM_MODEL: m_model}
                models[ARM_ORACLE] = (
                    None
                    if ref_clip is None
                    else np.broadcast_to(
                        LW.oracle_mean_periodogram(ref_clip, n_fft=n_fft, hop=hop, n_mics=mics),
                        power.shape,
                    )
                )
                for arm, model in models.items():
                    if model is None:
                        payload["rows"].append(
                            dict(
                                rig=rig,
                                regime=scored.regime,
                                support=scored.window.key,
                                arm=arm,
                                n_fft=int(n_fft),
                                hop=int(hop),
                                available=False,
                                reason="no legal disjoint real segment of this recording",
                            )
                        )
                        continue
                    nll, cells = RE.marginal_frame_nll(power, model, band)
                    fs = RE.FrameScore(
                        scored.window,
                        RE.frame_times_on_clock(scored.window, pg),
                        nll,
                        cells,
                        int(n_fft),
                        int(hop),
                    )
                    items[arm].append(fs)
                    one = RE.composite_score([fs])
                    payload["rows"].append(
                        dict(
                            rig=rig,
                            regime=scored.regime,
                            support=scored.window.key,
                            arm=arm,
                            n_fft=int(n_fft),
                            hop=int(hop),
                            available=True,
                            n_mics=int(mics),
                            n_frames=int(one["n_frames"]),
                            band_bins=int(band.sum()),
                            score=float(one["score"]),
                            score_per_band_cell=float(one["score_per_band_cell"]),
                            unique_seconds=float(one["unique_seconds"]),
                            weighted_nats=float(one["weighted_nats"]),
                            level_offset_db=(
                                level_offset_db(power, model, band) if arm == ARM_MODEL else None
                            ),
                            model_seconds=float(dt) if arm == ARM_MODEL else 0.0,
                            band_cells_per_unique_second=float(
                                int(mics) * int(band.sum()) * LW.SR / float(n_fft)
                            ),
                            **(ratio_diagnostics(power, model, band) if arm == ARM_MODEL else {}),
                        )
                    )
            wall = time.time() - wall0
            scores: dict[str, float | None] = {}
            for arm, fs_list in items.items():
                if not fs_list:
                    scores[arm] = None
                    continue
                comp = RE.composite_score(fs_list)
                scores[arm] = float(comp["score"])
                payload["pooled"].append(
                    dict(
                        rig=rig,
                        arm=arm,
                        n_fft=int(n_fft),
                        hop=int(hop),
                        pooled_score=float(comp["score"]),
                        pooled_score_per_band_cell=float(comp["score_per_band_cell"]),
                        unique_seconds=float(comp["unique_seconds"]),
                        n_frames=int(comp["n_frames"]),
                        n_supports=len(fs_list),
                    )
                )
            audio_seconds = sum(float(c.duration_s) for _, c, _ in clips.values())
            payload["timing"].append(
                dict(
                    rig=rig,
                    n_fft=int(n_fft),
                    hop=int(hop),
                    model_seconds=float(model_seconds),
                    wall_seconds=float(wall),
                    audio_seconds=float(audio_seconds),
                    model_seconds_per_audio_second=float(model_seconds / max(audio_seconds, 1e-9)),
                )
            )
            mdl, orc = scores.get(ARM_MODEL), scores.get(ARM_ORACLE)
            diff = None if (mdl is None or orc is None) else mdl - orc
            print(
                f"[{rig} {n_fft}/{hop}] model={mdl if mdl is None else round(mdl, 3)} "
                f"oracle={orc if orc is None else round(orc, 3)} "
                f"diff={diff if diff is None else round(diff, 3)} "
                f"model {model_seconds:.1f} s, wall {wall:.1f} s"
            )
    seen = {(s["rig"], s["scored"]["key"]) for s in payload["supports"]}
    scored_keys = {(r["rig"], r["support"]) for r in payload["rows"]}
    for key, entry in cached_supports.items():
        if key not in seen and key in scored_keys:
            payload["supports"].append(entry)
    payload["supports"].sort(key=lambda s: (s["rig"], s["scored"]["key"]))
    payload["rows"].sort(key=lambda r: (r["rig"], r["n_fft"], r["arm"], r["support"]))
    payload["pooled"].sort(key=lambda p: (p["rig"], p["n_fft"], p["arm"]))
    payload["timing"].sort(key=lambda t: (t["rig"], t["n_fft"]))
    return payload


def table(payload: dict[str, Any], settings: list[tuple[int, int]]) -> list[str]:
    """The one table this script exists for."""
    pooled = {(p["rig"], p["arm"], p["n_fft"], p["hop"]): p for p in payload["pooled"]}
    timing = {(t["rig"], t["n_fft"], t["hop"]): t for t in payload["timing"]}
    lines = [
        "| rig | NFFT | hop | legacy model (nats/s) | oracle (nats/s) | model − oracle | "
        "model wall (s) | setting wall (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rig in sorted({p["rig"] for p in payload["pooled"]}):
        for n_fft, hop in settings:
            m = pooled.get((rig, ARM_MODEL, n_fft, hop))
            o = pooled.get((rig, ARM_ORACLE, n_fft, hop))
            t = timing.get((rig, n_fft, hop))
            if m is None and o is None:
                continue
            ms = "—" if m is None else f"{m['pooled_score']:,.1f}"
            orc = "—" if o is None else f"{o['pooled_score']:,.1f}"
            gap = None if (m is None or o is None) else m["pooled_score"] - o["pooled_score"]
            ds = "—" if gap is None else f"{gap:,.1f}"
            mt = "—" if t is None else f"{t['model_seconds']:.1f}"
            wt = "—" if t is None else f"{t['wall_seconds']:.1f}"
            legacy = " (legacy ref)" if (n_fft, hop) == (16384, 1024) else ""
            lines.append(f"| {rig}{legacy} | {n_fft} | {hop} | {ms} | {orc} | {ds} | {mt} | {wt} |")
    return lines


def findings(payload: dict[str, Any], settings: list[tuple[int, int]]) -> str:
    rows = [r for r in payload["rows"] if r.get("available")]
    pooled = {(p["rig"], p["arm"], p["n_fft"], p["hop"]): p for p in payload["pooled"]}
    out: list[str] = []
    out.append("# Noise model v2 — the legacy model's Whittle risk on a SHORT front end")
    out.append("")
    out.append(
        "Author: `ShortWhittle`. Every number is produced by "
        "`scripts/noise_v2_short_whittle.py` (→ `short_whittle.json`) in this directory. "
        "One measurement, not a study."
    )
    out.append("")
    out.append(
        "The objective is the FROZEN composite spectral risk, taken from the evaluator itself "
        "(`revised_eval.predicted_m` → `marginal_frame_nll` → `FrameScore` → "
        "`composite_score`): positive exposure-weighted `Σ a_i (I_i/M_i + log M_i)` on "
        "30–7900 Hz, per unique observed second, with the `hop/n_fft` exposure factor and the "
        "duplicate-frame rule. **Only `(NFFT, hop)` changes between rows**, so the column is "
        "comparable down its own length. Lower is better; the numbers are ABSOLUTE-level risks "
        "in nats per unique second, so they are large and signed — read the `model − oracle` "
        "difference, not the magnitude."
    )
    out.append("")
    out.append("## The table")
    out.append("")
    out.extend(table(payload, settings))
    out.append("")
    out.append(
        "Oracle = the study's own non-parametric floor: `M` is the Welch mean periodogram of a "
        "speed-matched legal disjoint real segment of the SAME recording at the SAME NFFT "
        "(`results/noise_v2/criteria/findings.md` § 3.1). The oracle carries no model and no "
        "carrier, and it is the same material at every setting."
    )
    out.append("")
    cells = sorted({int(round(r["band_cells_per_unique_second"])) for r in rows})
    out.append(
        "Why the four rows are comparable at all: the exposure weights sum to `sr/n_fft` per "
        "second and a frame carries `mics × band_bins` cells, so the CELL BUDGET per unique "
        f"second is `mics × band_bins × sr/n_fft` = {cells} — the same to 0.1 % at every "
        "setting. A short front end changes the resolution, not how much material is scored."
    )
    out.append("")
    out.append("## What it says")
    out.append("")
    for rig in sorted({p["rig"] for p in payload["pooled"]}):
        diffs: dict[tuple[int, int], float] = {}
        for n_fft, hop in settings:
            m = pooled.get((rig, ARM_MODEL, n_fft, hop))
            o = pooled.get((rig, ARM_ORACLE, n_fft, hop))
            if m is not None and o is not None:
                diffs[(n_fft, hop)] = m["pooled_score"] - o["pooled_score"]
        if not diffs:
            continue
        ref = diffs.get((16384, 1024))
        best = min(diffs, key=lambda k: diffs[k])
        listed = ", ".join(f"{n}/{h} {diffs[(n, h)]:,.0f}" for n, h in diffs)
        line = f"* **{rig}** — gap to the floor in nats/s: {listed}."
        if ref is not None and ref != 0.0:
            line += (
                f" The best setting is {best[0]}/{best[1]} at {diffs[best]:,.0f}, "
                f"a factor {ref / diffs[best]:.2f} of the legacy 16384/1024 reference "
                f"({ref:,.0f})."
            )
        out.append(line)
        tails = {
            (r["n_fft"], r["hop"]): r
            for r in rows
            if r["rig"] == rig and r["arm"] == ARM_MODEL and r.get("im_top_0p1pct_share")
        }
        shares = {
            k: np.median(
                [
                    r["im_top_0p1pct_share"]
                    for r in rows
                    if (r["rig"], r["arm"], r["n_fft"], r["hop"]) == (rig, ARM_MODEL, k[0], k[1])
                ]
            )
            for k in tails
        }
        offsets = {
            k: np.median(
                [
                    r["level_offset_db"]
                    for r in rows
                    if (r["rig"], r["arm"], r["n_fft"], r["hop"]) == (rig, ARM_MODEL, k[0], k[1])
                ]
            )
            for k in tails
        }
        out.append(
            "  Median in-band level offset (real over model): "
            + ", ".join(f"{n}/{h} {offsets[(n, h)]:+.2f} dB" for n, h in offsets)
            + ". Share of `Σ I/M` carried by the worst 0.1 % of in-band cells (median over "
            "supports): " + ", ".join(f"{n}/{h} {shares[(n, h)]:.1%}" for n, h in shares) + "."
        )
    regimes = sorted({(r["rig"], r["regime"]) for r in rows})
    if len({rig for rig, _ in regimes}) < len(regimes):
        out.append("")
        out.append(
            "The per-regime split, because one export is used on every support of its rig and "
            "the frames of the five supports are disjoint (so the numerator and the unique "
            "seconds simply add):"
        )
        out.append("")
        out.append("| rig | regime | " + " | ".join(f"{n}/{h}" for n, h in settings) + " |")
        out.append("|---|---|" + "---:|" * len(settings))
        for rig, regime in regimes:
            cols = []
            for n_fft, hop in settings:
                sel = [
                    r
                    for r in rows
                    if (r["rig"], r["regime"], r["n_fft"], r["hop"]) == (rig, regime, n_fft, hop)
                ]
                num = {
                    arm: sum(r["weighted_nats"] for r in sel if r["arm"] == arm)
                    for arm in (ARM_MODEL, ARM_ORACLE)
                }
                den = {
                    arm: sum(r["unique_seconds"] for r in sel if r["arm"] == arm)
                    for arm in (ARM_MODEL, ARM_ORACLE)
                }
                if min(den.values()) <= 0.0:
                    cols.append("—")
                    continue
                gap = num[ARM_MODEL] / den[ARM_MODEL] - num[ARM_ORACLE] / den[ARM_ORACLE]
                cols.append(f"{gap:,.0f}")
            out.append(f"| {rig} | {regime} | " + " | ".join(cols) + " |")
        out.append("")
        out.append("(model − oracle gap in nats/s, pooled over that regime's supports.)")
    out.append("")
    out.append("## Exports used, and what they can and cannot supply")
    out.append("")
    for rig, cfg in payload["exports"].items():
        src = dict(cfg.get("nuisance_source") or {})
        out.append(
            f"* **{rig}** — `{cfg['path']}`, {cfg['label']}; route `{cfg['route']}` over "
            f"{src.get('n_clips', '?')} fitted clips "
            f"(`aggregate_nuisance`, linear-power mean, `power_scale` folded per clip, "
            f"clip-local latents dropped: {src.get('dropped_clip_local_latents')})."
        )
        if src.get("profile_tiled_to_rotors"):
            out.append(
                f"  The bench rig fit carries {src['profile_rotors_in_export']} rotor, so its "
                f"tied profile is tiled onto the flight's {src['profile_tiled_to_rotors']} "
                "rotors (one shape at one level per rotor, as in `scripts/_dregon_transfer.py`); "
                "the bench is single-channel, so no per-microphone pattern is transferred and "
                "the absolute level and the room floor are the bench's."
            )
        if src.get("extrapolated"):
            out.append(
                "  The scored recording is named by no clip of this export: this is the "
                "manifest's declared extrapolation, not an identity match."
            )
    out.append("")
    out.append("## Run time")
    out.append("")
    out.append("| rig | NFFT | hop | audio (s) | `predicted_m` (s) | s per audio-s | wall (s) |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for t in payload["timing"]:
        out.append(
            f"| {t['rig']} | {t['n_fft']} | {t['hop']} | {t['audio_seconds']:.0f} | "
            f"{t['model_seconds']:.1f} | {t['model_seconds_per_audio_second']:.2f} | "
            f"{t['wall_seconds']:.1f} |"
        )
    out.append("")
    out.append(
        "The criteria study measured ~20 s of `predict_spectrum` (the CANDIDATE path) per "
        "audio-second at NFFT 16384. The legacy `predicted_m` path measured here is two to "
        "three orders of magnitude cheaper, so the whole grid runs on the laptop in minutes and "
        "no cluster submission is needed — which is just as well, since both exports are "
        "gitignored and `omnirun` ships the git revision only."
    )
    out.append("")
    out.append("## Per-support detail")
    out.append("")
    out.append(
        "| rig | support | regime | NFFT | model (nats/s) | oracle (nats/s) | model − oracle | "
        "model level offset (dB) | speed mismatch (rev/s) |"
    )
    out.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    mismatch = {
        (s["rig"], s["scored"]["key"]): (
            None if not s.get("oracle_match") else s["oracle_match"]["mean_rps_mismatch_rps"]
        )
        for s in payload["supports"]
    }
    by_key: dict[tuple[str, str, int, str], dict[str, Any]] = {
        (r["rig"], r["support"], r["n_fft"], r["arm"]): r for r in rows
    }
    for rig, support, n_fft in sorted({(r["rig"], r["support"], r["n_fft"]) for r in rows}):
        m = by_key.get((rig, support, n_fft, ARM_MODEL))
        o = by_key.get((rig, support, n_fft, ARM_ORACLE))
        if m is None:
            continue
        mm = mismatch.get((rig, support))
        orc = "—" if o is None else f"{o['score']:,.1f}"
        diff = "—" if o is None else f"{m['score'] - o['score']:,.1f}"
        mms = "—" if mm is None else f"{mm:.3f}"
        out.append(
            f"| {rig} | {support} | {m['regime']} | {n_fft} | {m['score']:,.1f} | {orc} | "
            f"{diff} | {m['level_offset_db']:+.2f} | {mms} |"
        )
    out.append("")
    out.append("## Provenance")
    out.append("")
    prov = payload["provenance"]
    out.append(
        f"* git HEAD `{prov['git_head']}`, {prov['timestamp']}, wall {prov['wall_seconds']:.1f} s"
    )
    out.append(f"* band {payload['band_hz'][0]}–{payload['band_hz'][1]} Hz, 16 kHz periodic Hann")
    out.append(f"* {RE.HISTORICAL_FORWARD_LABEL}")
    out.append("")
    for (rig, arm, n_fft, hop), p in sorted(pooled.items()):
        out.append(
            f"* pooled `{rig}`/`{arm}` at {n_fft}/{hop}: {p['pooled_score']:,.4f} nats/s, "
            f"{p['pooled_score_per_band_cell']:.4f} nats/band-cell, "
            f"{p['unique_seconds']:.3f} unique s over {p['n_supports']} supports, "
            f"{p['n_frames']} frames"
        )
    out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="legacy rotor-noise model: composite Whittle risk on a short STFT front end"
    )
    ap.add_argument("--rigs", default="dregon michaels", help="dregon and/or michaels")
    ap.add_argument(
        "--settings",
        default=" ".join(f"{n}:{h}" for n, h in SETTINGS_DEFAULT),
        help="`n_fft:hop` pairs, space or comma separated",
    )
    ap.add_argument("--supports-per-rig", type=int, default=None)
    ap.add_argument("--mics", type=int, default=8)
    ap.add_argument(
        "--f-min",
        type=float,
        default=RE.OBS_F_MIN,
        help="lower band edge, Hz (default: the frozen 30 Hz)",
    )
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument(
        "--reuse",
        action="store_true",
        help="keep (rig, setting) blocks already present in the output JSON",
    )
    args = ap.parse_args()

    rigs = [r for r in str(args.rigs).replace(",", " ").split() if r]
    unknown = [r for r in rigs if r not in EXPORTS]
    if unknown:
        raise SystemExit(f"unknown rigs {unknown}; known: {sorted(EXPORTS)}")
    settings: list[tuple[int, int]] = []
    for token in str(args.settings).replace(",", " ").split():
        n_fft, _, hop = token.partition(":")
        settings.append((int(n_fft), int(hop or int(n_fft) // 4)))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "short_whittle.json"
    cached = json.loads(out_json.read_text()) if (args.reuse and out_json.exists()) else None

    t0 = time.time()
    payload = run(
        rigs=rigs,
        settings=settings,
        supports_per_rig=args.supports_per_rig,
        n_mics=int(args.mics),
        cached=cached,
        f_min=float(args.f_min),
    )
    payload["provenance"] = dict(
        git_head=LW.git_head(),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        wall_seconds=float(time.time() - t0),
        command=" ".join(sys.argv),
        rigs=rigs,
        mics=int(args.mics),
        supports_per_rig=args.supports_per_rig,
        reused=bool(cached),
        caveat=RE.ADAPTIVE_SELECTION_CAVEAT,
    )
    out_json.write_text(json.dumps(payload, indent=1) + "\n")
    (out_dir / "findings.md").write_text(findings(payload, settings))
    print("\n".join(table(payload, settings)))
    print(f"wrote {out_json} and {out_dir / 'findings.md'}")


if __name__ == "__main__":
    main()
