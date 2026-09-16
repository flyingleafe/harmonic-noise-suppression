#!/usr/bin/env python3
"""Assemble the *noise model v2* corpus survey: one verdict per corpus.

Reads the per-recording readings of :mod:`scripts.noise_v2_bench_speed`
(``bench_speeds.json``) and joins them to a frozen corpus registry — what each
corpus is, where it lives, what kind of recording it holds, how many rigs, what
telemetry exists, where a rotor speed can come from, its licence — and writes:

* ``<out>/survey.json``      — the whole survey, machine-readable.
* ``<out>/survey_table.md``  — the corpus table plus the per-usable-recording
                               table (one row per fit point).
* ``<out>/survey_speeds.png``— estimated speed against throttle/condition, per
                               corpus (also copied to ``--fig-dir``).

The corpus registry is data, not code: every row carries its evidence string,
so a verdict can be traced without re-reading the campaign docs. Corpora with
no audio in the repo (literature names) carry the access verdict only.

Run:
  PYTHONPATH=src python scripts/noise_v2_corpus_survey.py
  PYTHONPATH=src python scripts/noise_v2_corpus_survey.py --speeds results/noise_v2/survey/bench_speeds_all.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if (_HERE.parent / "src").is_dir() else Path.cwd().resolve()
sys.path.insert(0, str(_ROOT / "src"))

#: One entry per surveyed corpus. ``families`` links the corpus to the
#: ``family`` field of the ``bench_speeds.json`` rows (empty = no audio pass).
CORPORA: list[dict[str, Any]] = [
    {
        "corpus": "DREGON single-motor bench",
        "families": ["dregon_bench"],
        "location": "dload:DREGON-frames (recordings motor_Motor{1-4}_{50..90}, motor_allMotors_70)",
        "type": "static single-motor bench; motor clamped, one throttle per recording",
        "rigs": "1 airframe (MikroKopter), 4 individually driven motors + 1 all-motors run",
        "telemetry": "none logged; the THROTTLE SETPOINT is in the recording name",
        "speed_source": "estimated (Welch harmonic sum + odd-harmonic octave check, "
        "tol from the two-half spread) — cross-checked against the validated "
        "throttle law rate = 0.975*throttle + 0.37 rev/s",
        "format": "8 ch, 44.1 kHz, 25-45 s files with 11-35 s of motor-on",
        "licence": "research use (INRIA DREGON)",
        "verdict": "USABLE",
        "reason": "one throttle per recording, one rotor per recording, and a "
        "non-acoustic reference (the throttle law) to score the estimator against",
    },
    {
        "corpus": "SPCUP19 AGH single rotors",
        "families": ["spcup_single"],
        "location": "dload:SPCUP19-egonoise (AGH/ego-noise/single rotors/0..7)",
        "type": "static single-rotor takes, one mic",
        "rigs": "1 (AGH quadrotor, model unspecified)",
        "telemetry": "none",
        "speed_source": "estimated (same estimator); cross-checked against the "
        "accepted blind-campaign readings (blind-corpus-annotation.md)",
        "format": "1 ch, 44.1 kHz, 16.4-17.6 s",
        "licence": "free for personal, educational and academic use only",
        "verdict": "USABLE",
        "reason": "single-source static takes; the blind campaign already accepted "
        "all 8 under the one-source rule, so the readings are independently checkable",
    },
    {
        "corpus": "SPCUP19 static / hover (quadrotor)",
        "families": ["spcup_static"],
        "location": "dload:SPCUP19-egonoise (AGH static clean/corrupted, "
        "Diagonal_Unloading static, Idea_ssu stationary, Shout_COOEE StaticSubmission1/2)",
        "type": "four-rotor static / hover windows on 4 team rigs",
        "rigs": "4 (AGH quadrotor, DJI Phantom 4 PRO, DJI Phantom 4 GL300C, Intel Aero RTF)",
        "telemetry": "none",
        "speed_source": "none accepted (estimator margins below the 3 dB rule)",
        "format": "1-8 ch, 44.1/48 kHz, 11.5-85.9 s",
        "licence": "free for personal, educational and academic use only",
        "verdict": "REJECTED",
        "reason": "four combs inside a few rev/s make every rival comb score almost "
        "as well as the accepted one, so no window clears the 3 dB margin — the same "
        "geometry that flattened the blind campaign's ridge clearance (no SPCUP static "
        "window cleared its off-comb null there either)",
    },
    {
        "corpus": "SPCUP19 ChuMS propeller rig",
        "families": ["chums_bench"],
        "location": "dregon.inria.fr SPCUP19_ChuMS_data.zip -> UAV_rotor_recordings.mat "
        "(TestResults.Test(1..9))",
        "type": "STATIC PROPELLER RIG: 1, 2 or 3 propellers running, 3 repeats each, "
        "8 calibrated mics on a 1 m arc (MicPositions in the .mat)",
        "rigs": "1 rig (Skylark M4-680 / Dotterel) in 3 propeller-count conditions",
        "telemetry": "none; the .mat records the propeller COUNT and per-mic OASPL, no RPM",
        "speed_source": "estimated (same estimator)",
        "format": "8 ch, 44.1 kHz, 73-75 s, calibrated in Pa",
        "licence": "free for personal, educational and academic use only",
        "verdict": "PARTLY USABLE",
        "reason": "a genuine bench rig that the published SPCUP19-egonoise frames "
        "expose only as 216 unlabelled arrays (Freq/SPL/RawTruncatedCalibrated per "
        "mic); the recordings are stationary but the rig runs several propellers at "
        "uncontrolled speeds, so only the runs that clear the 3 dB margin are kept",
    },
    {
        "corpus": "DroneAudioSet (drone-only)",
        "families": ["daset"],
        "location": "HuggingFace ahlab-drone-project/DroneAudioSet, subset "
        "drone-only/ (28 parquet shards, 3.1 GiB, 168 recordings); published as "
        "dload:DroneAudioSet (88.4 GiB, all subsets)",
        "type": "rig-mounted static: the quadcopter is bolted to an aluminium frame "
        "at 1.5 m and run at a fixed throttle (the paper's hover emulation)",
        "rigs": "2 (DJI F450 'D_large', 450 mm wheelbase, 9.4x5.0 props; DJI F330 "
        "'D_small', 330 mm, 8x4.5 props) x 2 throttles (low/high) x mic distance 25/50 cm",
        "telemetry": "none; throttle is a two-level label. The paper states spectral "
        "lines 168/235 Hz (D_large low/high) and 156/259 Hz (D_small low/high)",
        "speed_source": "estimated (same estimator); cross-checked against the "
        "paper's stated lines via the measured blade-pass frequency 2*f",
        "format": "8 ch (M_up / M_down arrays) or 1 ch (M_center Soundskrit), 16 kHz, 30-152 s",
        "licence": "MIT",
        "verdict": "PARTLY USABLE",
        "reason": "the largest static multichannel drone corpus and the only one with "
        "two airframes; the recordings are stationary but the four rotors are not "
        "near-equal on every mic, so the reading is accepted per recording",
    },
    {
        "corpus": "AVQ constant-throttle ego-noise",
        "families": ["avq_bench"],
        "location": "dload:AVQ (sequences S1_seq1/S1_seq2/S1_seq3/S2_seq1; the same "
        "recordings are in dload:AVQ-egonoise at channel 0, 16 kHz)",
        "type": "BENCH/CONSTANT-THROTTLE EGO-NOISE: the four sequences the AVQ spec "
        "table marks Type = EO with Drone = constant — S1 seq1 50 % (120 s), S1 seq2 "
        "100 % (120 s), S1 seq3 150 % (40 s), S2 seq1 100 % (210 s) — read in "
        "consecutive 30 s windows. The other eight sequences are excluded by type: "
        "S2_seq2 is EO at DYNAMIC throttle, S2_seq5/seq6 are constant 100 % but "
        "speech+ego-noise MIXTURES, and S1_seq5/S2_seq3/seq4/seq7/seq8 are speech "
        "with the drone muted or dynamic",
        "rigs": "1 quadrotor (onboard 8-mic circular array + camera, QMUL)",
        "telemetry": "none logged; the THROTTLE SETTING per sequence is in the spec "
        "table (50/100/150 %), which is what makes these four sequences bench class",
        "speed_source": "estimated (same estimator, multi-rotor mode)",
        "format": "8 ch, 44.1 kHz; 40-210 s per sequence -> 30 s windows",
        "licence": "free for academic/research use (courtesy of Lin Wang, QMUL)",
        "verdict": "USABLE",
        "reason": "the first survey pass refused the whole corpus as 'free flight' "
        "without reading the spec table. The table's Drone column is explicit: four "
        "sequences hold ego-noise only at a CONSTANT throttle setting, which is the "
        "bench-class definition used here (stationary window, one speed per rotor). "
        "The blind-campaign octave warning (median fvk_ratio_double 1.044) was "
        "measured on the FLIGHT sequences, not on these",
    },
    {
        "corpus": "drone_audio (Al-Emadi IWCMC 2019)",
        "families": ["drone_audio"],
        "location": "data/drone_audio, dload:drone_audio "
        "(Binary_Drone_Audio/yes_drone, 1332 clips; Multiclass bebop_1/membo_1)",
        "type": "indoor propeller recordings of a Parrot Bebop and a Parrot Mambo, "
        "cut into 1 s clips; the sibling unknown/ class is ESC-50 + white noise + silence",
        "rigs": "2 (Bebop, Mambo)",
        "telemetry": "none",
        "speed_source": "none accepted",
        "format": "1 ch, 16 kHz, 0.65-1.02 s per clip",
        "licence": "none stated (citation request only, IWCMC 2019 paper)",
        "verdict": "REJECTED",
        "reason": "the clips are 1.02 s, an eighth of the 8 s stationary window the "
        "tolerance rule needs, and there is no take-level grouping that would let "
        "clips be re-joined; the sampled clips also fail the margin rule outright",
    },
    {
        "corpus": "zenodo_drone_noises",
        "families": ["zenodo"],
        "location": "data/zenodo_drone_noises (all_drone_noises.zip -> "
        "noises-train-drones/n116..n120, noises-test-drones/n121..n122)",
        "type": "7 unlabelled drone-noise clips; no rig, session or setup recorded "
        "anywhere in the zip (no README, no metadata file)",
        "rigs": "unknown (one unknown rig per file at best)",
        "telemetry": "none",
        "speed_source": "none accepted for a rig; per-file estimates reported",
        "format": "1 ch, 8 kHz (n121/n122) or 44.1 kHz, 40-215 s",
        "licence": "none stated",
        "verdict": "REJECTED",
        "reason": "no rig identity, so a per-rig noise parameter cannot be attached to "
        "the point even where the estimator reads a speed; the recordings also drift "
        "(flight, not bench) and mostly fail the margin rule",
    },
    {
        "corpus": "KAIST-rotating-acoustic (control)",
        "families": ["kaist"],
        "location": "dload:KAIST-rotating-acoustic",
        "type": "industrial rotating-machine testbed, mono bench, dataset-stated 3010 RPM",
        "rigs": "1 (not a drone)",
        "telemetry": "the stated nominal 3010 RPM = 50.167 rev/s",
        "speed_source": "paper-stated nominal; the estimator is scored against it",
        "format": "1 ch, 51.2 kHz, 60 s, 5 recordings (fault + severity per file)",
        "licence": "CC BY 4.0",
        "verdict": "CONTROL ONLY",
        "reason": "kept as the out-of-domain control the blind campaign used: bearing "
        "fault lines are real, strong and NOT octaves of the shaft, so this corpus "
        "measures whether the gate refuses what it cannot read. Not a drone fit point",
    },
    {
        "corpus": "DronePrint (Kolamunna et al. 2021)",
        "families": [],
        "location": "OSF repository via github.com/DronePrint/DronePrint (not in this repo)",
        "type": "far-field FREE FLIGHT: 5 drone classes recorded with a RODE NTG4 "
        "shotgun mic at ~20 m altitude within a 50 m radius, plus YouTube-scraped clips",
        "rigs": "5 recorded (Bebop 2, Mavic Pro, Phantom 4 Pro, Spark, Matrice 100) "
        "+ 15 online classes",
        "telemetry": "none",
        "speed_source": "none",
        "format": "1 ch, 44.1 kHz",
        "licence": "open (OSF), attribution",
        "verdict": "REJECTED (not downloaded)",
        "reason": "free flight at 20 m with a directional ground mic: the comb "
        "decoheres, the speed is unknown and the manoeuvre is unconstrained — class G "
        "of the blind-corpus plan (mono far field, refused by default)",
    },
    {
        "corpus": "MAVD",
        "families": [],
        "location": "two unrelated datasets answer to this name: MAVD (Mandarin "
        "audio-visual with depth, github.com/SpringHuo/MAVD) and MAVD-Traffic "
        "(Montevideo audio-visual traffic)",
        "type": "neither is drone ego-noise: speech+depth corpus / street-traffic corpus",
        "rigs": "none",
        "telemetry": "n/a",
        "speed_source": "n/a",
        "format": "n/a",
        "licence": "MAVD: access by e-mail request to the authors",
        "verdict": "REJECTED (not a drone-noise corpus)",
        "reason": "the name does not resolve to a rotor-noise dataset; the speech "
        "variant is also gated behind an e-mail request, which the survey rules out",
    },
    {
        "corpus": "DroneNoise Database (Salford)",
        "families": [],
        "location": "salford.figshare.com/articles/dataset/DroneNoise_Database/22133411 "
        "(figshare article 22133411, 175 files, 742 MB)",
        "type": "field OVERFLIGHT campaign (Edzell, Scotland, 2022-08-17): sUAS flying "
        "over a ground microphone array, 3 events per configuration",
        "rigs": "several sUAS types (file stems Ed_3p / Ed_Fp / Ed_M3 / Ed_Yn)",
        "telemetry": "none published with the audio (no RPM channel in the file set)",
        "speed_source": "none",
        "format": "ground mics M1..M?, ~45-68 s WAV per event",
        "licence": "CC BY 4.0",
        "verdict": "REJECTED (metadata only, not downloaded)",
        "reason": "overflight recordings are Doppler-shifted and non-stationary by "
        "construction, and no rotor speed is published, so no fit point can be built "
        "from them",
    },
    {
        "corpus": "ESC-50",
        "families": [],
        "location": "github.com/karolpiczak/ESC-50 (already inside "
        "data/drone_audio as the unknown/ negatives)",
        "type": "2000 five-second environmental clips, 50 classes; no drone class "
        "(the closest are helicopter and chainsaw)",
        "rigs": "none",
        "telemetry": "none",
        "speed_source": "none",
        "format": "1 ch, 44.1 kHz, 5 s",
        "licence": "CC BY-NC 3.0",
        "verdict": "REJECTED",
        "reason": "not a drone corpus at all; it is the negative-class pool of "
        "drone_audio and carries no rotating-source speed",
    },
]


def load_speeds(paths: list[Path]) -> dict[str, Any]:
    """Merge one or more ``bench_speeds*.json`` payloads (later files win per id).

    The corpus pass runs remotely for the dload/HuggingFace corpora and locally
    for the two whose raw trees live only in ``data/``, so the survey reads
    several reading files and joins them on ``corpus/id``.
    """
    merged: dict[str, Any] = {}
    rows: dict[tuple[str, str], dict] = {}
    checks: dict[str, Any] = {}
    for path in paths:
        if not Path(path).exists():
            raise SystemExit(f"{path} not found — run scripts/noise_v2_bench_speed.py first")
        payload = json.loads(Path(path).read_text())
        merged = {**merged, **{k: v for k, v in payload.items() if k != "rows"}}
        checks.update(payload.get("cross_checks") or {})
        for row in payload["rows"]:
            rows[(row["corpus"], row["id"])] = row
    merged["rows"] = [rows[k] for k in sorted(rows)]
    merged["cross_checks"] = checks
    merged["sources"] = [str(p) for p in paths]
    return merged


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, list):
        return ", ".join(f"{float(v):.2f}" for v in value)
    return str(value)


def corpus_stats(rows: list[dict], families: list[str]) -> dict[str, Any]:
    mine = [r for r in rows if r["family"] in families]
    usable = [r for r in mine if r["usable"]]
    rigs = sorted({r["rig"] for r in usable})
    return {
        "n_recordings": len(mine),
        "n_usable": len(usable),
        "usable_rigs": rigs,
        "rig_condition_points": sorted(
            {
                f"{r['rig']}|{r.get('throttle') or r.get('throttle_level') or r['condition']}"
                for r in usable
            }
        ),
        "speed_range_rev_s": (
            [
                round(min(r["rate_rev_s"] for r in usable), 2),
                round(max(r["rate_rev_s"] for r in usable), 2),
            ]
            if usable
            else None
        ),
        "median_tolerance_rev_s": (
            round(float(sorted(r["speed_tolerance_rev_s"] for r in usable)[len(usable) // 2]), 3)
            if usable
            else None
        ),
        "n_resolved_by_rig": {
            rig: {
                "n_usable": len([r for r in usable if r["rig"] == rig]),
                "n_rotors": max(int(r.get("n_rotors", 1)) for r in usable if r["rig"] == rig),
                "histogram": {
                    str(n): len([r for r in usable if r["rig"] == rig and r["n_resolved"] == n])
                    for n in sorted({r["n_resolved"] for r in usable if r["rig"] == rig})
                },
                "n_multiplicity_unresolved": len(
                    [r for r in usable if r["rig"] == rig and r["multiplicity_unresolved"]]
                ),
                "max_spread_rev_s": round(
                    max(float(r["spread_rev_s"] or 0.0) for r in usable if r["rig"] == rig), 3
                ),
            }
            for rig in rigs
        },
        "reject_reasons": _reject_histogram(mine),
    }


def _reject_histogram(rows: list[dict]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for r in rows:
        if r["usable"]:
            continue
        for part in str(r["reason"]).split(";"):
            key = part.strip().split(" ")[0:3]
            label = " ".join(key)
            hist[label] = hist.get(label, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: -kv[1]))


def survey_table(entries: list[dict], rows: list[dict]) -> str:
    head = (
        "| Corpus | Location | Type | Rigs | Telemetry | Speed source | fs / ch / duration "
        "| Licence | Verdict | Reason |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = []
    for e in entries:
        st = e["stats"]
        verdict = e["verdict"]
        if st["n_recordings"]:
            verdict = f"{verdict} ({st['n_usable']}/{st['n_recordings']} recordings)"
        body.append(
            "| "
            + " | ".join(
                x.replace("\n", " ")
                for x in (
                    e["corpus"],
                    e["location"],
                    e["type"],
                    e["rigs"],
                    e["telemetry"],
                    e["speed_source"],
                    e["format"],
                    e["licence"],
                    verdict,
                    e["reason"],
                )
            )
            + " |"
        )
    out = ["## Corpora", "", head + "\n".join(body), ""]

    usable = [r for r in rows if r["usable"]]
    out += [
        "## Fit points (one row per usable recording)",
        "",
        "| Corpus | Recording | Rig | Condition | f_i rev/s (one per resolved rotor) "
        "| f̄ rev/s | Tol rev/s | Spread rev/s | n_res / n_rotors | Octave | Margin dB "
        "| Margin family-only dB | Half Δ rev/s | Window s | ch | fs |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(usable, key=lambda r: (r["corpus"], r["id"])):
        out.append(
            "| "
            + " | ".join(
                (
                    r["corpus"],
                    r["id"],
                    r["rig"],
                    str(r["condition"]),
                    _fmt(r["speed_rev_s"]),
                    _fmt(r["rate_rev_s"]),
                    _fmt(r["speed_tolerance_rev_s"]),
                    _fmt(r["spread_rev_s"]),
                    f"{r['n_resolved']} / {r['n_rotors']}"
                    + (" (unresolved)" if r["multiplicity_unresolved"] else ""),
                    r["octave_verdict"],
                    _fmt(r["margin_db"]),
                    _fmt(r["margin_family_only_db"]),
                    _fmt(r["half_delta_rev_s"]),
                    _fmt(r["window_s"]),
                    str(r["channels"]),
                    str(r["fs"]),
                )
            )
            + " |"
        )
    out.append("")

    rejected = [r for r in rows if not r["usable"]]
    out += [
        "## Rejected recordings",
        "",
        "| Corpus | Recording | Reading rev/s | Margin dB | Reason |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(rejected, key=lambda r: (r["corpus"], r["id"])):
        out.append(
            f"| {r['corpus']} | {r['id']} | {_fmt(r['rate_rev_s'])} | "
            f"{_fmt(r['margin_db'])} | {r['reason']} |"
        )
    return "\n".join(out) + "\n"


def draw_speeds(rows: list[dict], out_dir: Path, fig_dir: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["corpus"], []).append(r)
    order = sorted(groups)
    fig, axes = plt.subplots(
        1, len(order), figsize=(2.5 * len(order) + 1.5, 4.4), dpi=150, sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, corpus in zip(axes, order, strict=False):
        rs = groups[corpus]
        labels = sorted({_condition_key(r) for r in rs})
        pos = {k: i for i, k in enumerate(labels)}
        rng = np.random.default_rng(7)
        for r in rs:
            x = pos[_condition_key(r)] + float(rng.uniform(-0.22, 0.22))
            style = dict(marker="o", s=14) if r["usable"] else dict(marker="x", s=16)
            for value in r["speed_rev_s"]:
                ax.scatter(
                    x,
                    float(value),
                    color="#1f77b4" if r["usable"] else "#d62728",
                    alpha=0.8,
                    linewidths=0.8,
                    **style,
                )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=6.5)
        ax.set_title(corpus, fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("estimated shaft speed (rev/s)")
    fig.suptitle(
        "Estimated rotor speed per corpus and condition (o usable, x rejected)", fontsize=10
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    name = "survey_speeds.png"
    fig.savefig(out_dir / name)
    fig.savefig(fig_dir / name)
    plt.close(fig)
    return name


def draw_multirotor_split(rows: list[dict], out_dir: Path, fig_dir: Path) -> str | None:
    """One panel per multi-rotor rig: the high-order band the split is read in.

    The example per rig is the reading with the most resolved rotors (ties
    broken by margin), so the panel shows what the rule actually saw: the band
    ``k·f_line ± 6 %``, its local median, the ``+6 dB`` peak threshold, and a
    marker at every resolved rotor line ``k·f_i/scale``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best: dict[str, dict] = {}
    for r in rows:
        ex = r.get("split_example")
        if not ex or r.get("mode") != "multi_rotor":
            continue
        key = str(r["rig"])
        cur = best.get(key)
        rank = (int(r["n_resolved"]), float(r["margin_db"]))
        if cur is None or rank > (int(cur["n_resolved"]), float(cur["margin_db"])):
            best[key] = r
    if not best:
        return None
    rigs = sorted(best)
    ncol = min(3, len(rigs))
    nrow = int(np.ceil(len(rigs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.0 * nrow), dpi=150, squeeze=False)
    for ax, rig in zip(axes.ravel(), rigs, strict=False):
        r = best[rig]
        ex = r["split_example"]
        f_hz = np.asarray(ex["f_hz"], dtype=float)
        p_db = np.asarray(ex["p_db"], dtype=float)
        k, scale = int(ex["order"]), float(ex["scale"])
        ax.plot(f_hz, p_db, lw=0.8, color="#333333")
        ax.axhline(ex["median_db"], color="#1f77b4", ls="--", lw=0.9, label="band median")
        ax.axhline(ex["threshold_db"], color="#2ca02c", ls=":", lw=0.9, label="median + 6 dB")
        for i, speed in enumerate(r["speed_rev_s"]):
            ax.axvline(
                float(speed) * k / max(scale, 1e-9),
                color="#d62728",
                lw=1.0,
                alpha=0.85,
                label="resolved rotor" if i == 0 else None,
            )
        ax.set_title(
            f"{rig}\n{r['id']}: order {k} of {ex['comb_rev_s']:.2f} rev/s, "
            f"n_res {r['n_resolved']}/{r['n_rotors']}"
            + (" (unresolved)" if r["multiplicity_unresolved"] else ""),
            fontsize=7.5,
        )
        ax.set_xlabel("frequency (Hz)", fontsize=8)
        ax.set_ylabel("power (dB)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6, loc="upper right")
    for ax in axes.ravel()[len(rigs) :]:
        ax.axis("off")
    fig.suptitle(
        "Per-rotor split: the high-order band, its 6 dB peak threshold and the resolved "
        "rotor lines",
        fontsize=10,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    name = "survey_multirotor_split.png"
    fig.savefig(out_dir / name)
    fig.savefig(fig_dir / name)
    plt.close(fig)
    return name


def _condition_key(row: dict) -> str:
    if row.get("throttle") is not None:
        return f"{int(row['throttle'])}%"
    if row.get("throttle_level"):
        return f"{row.get('drone', '')} {row['throttle_level']}"
    cond = str(row.get("condition", ""))
    return cond[:22]


#: Which reading source each corpus's audio comes from in the ingestion
#: derivation (``derivations.generate_noise_v2_bench_points``).
_SOURCE_OF_FAMILY = {
    "dregon_bench": "DREGON-frames",
    "spcup_single": "SPCUP19-egonoise",
    "spcup_static": "SPCUP19-egonoise",
    "daset": "DroneAudioSet-drone-only",
    "chums_bench": "SPCUP19-ChuMS-bench",
    "avq_bench": "AVQ",
}

#: Longest audio slice published per fit point (s).
PUBLISH_MAX_S = 30.0


def _cell(row: dict) -> str:
    """The (rig, condition) cell a reading belongs to — its repeatability group."""
    return f"{row['rig']}|{_condition_key(row)}"


def cell_octave_outliers(rows: list[dict]) -> dict[str, str]:
    """Usable readings that disagree with their own cell by a factor near 2.

    A (rig, condition) cell is the same rotor at the same setpoint recorded
    more than once, so a reading twice or half its cell median is an octave
    failure of the estimator on that recording, not a different speed. Those
    readings are dropped from the fit points (they are not re-scored here) and
    listed with their factor.
    """
    groups: dict[str, list[dict]] = {}
    for r in rows:
        if r["usable"]:
            groups.setdefault(_cell(r), []).append(r)
    out: dict[str, str] = {}
    for cell, rs in groups.items():
        if len(rs) < 3:
            continue
        med = float(np.median([r["rate_rev_s"] for r in rs]))
        for r in rs:
            ratio = r["rate_rev_s"] / med if med > 0 else 1.0
            if abs(ratio - 2.0) <= 0.12 or abs(ratio - 0.5) <= 0.06:
                out[r["id"]] = (
                    f"factor {ratio:.2f} of its {cell} cell median {med:.2f} rev/s "
                    "(octave failure on this recording)"
                )
    return out


def fit_points(rows: list[dict], outliers: dict[str, str]) -> list[dict]:
    """The manifest the ingestion derivation consumes: one entry per fit point."""
    points = []
    for r in sorted(rows, key=lambda r: (r["corpus"], r["id"])):
        if not r["usable"] or r["id"] in outliers:
            continue
        source = _SOURCE_OF_FAMILY.get(r["family"])
        if source is None:
            continue
        # ``offset_s`` is the reader's own cut inside the source recording (the
        # AVQ sequences are read in consecutive 30 s windows), so the published
        # slice is addressed relative to the FULL parent recording.
        offset = float(r.get("offset_s") or 0.0)
        centre = float(r["window_start_s"]) + 0.5 * float(r["window_s"])
        length = min(PUBLISH_MAX_S, max(float(r["active_s"]), float(r["window_s"])))
        start = offset + max(0.0, min(centre - 0.5 * length, float(r["duration_s"]) - length))
        points.append(
            {
                "key": f"{r['corpus'].replace('/', '-')}__{r['id']}",
                "corpus": r["corpus"],
                "source": source,
                "source_id": r.get("raw_path") or r.get("sequence") or r["id"],
                "rig": r["rig"],
                "rig_model": r.get("rig_model"),
                "condition": r["condition"],
                "throttle": r.get("throttle"),
                "throttle_level": r.get("throttle_level"),
                "n_rotors": int(r["n_rotors"]),
                "n_resolved": int(r["n_resolved"]),
                "multiplicity_unresolved": bool(r["multiplicity_unresolved"]),
                "octave_verdict": r["octave_verdict"],
                "speed_rev_s": [round(float(v), 4) for v in r["speed_rev_s"]],
                "speed_tolerance_rev_s": round(float(r["speed_tolerance_rev_s"]), 4),
                "spread_rev_s": round(float(r["spread_rev_s"] or 0.0), 4),
                "speed_source": "estimated(welch-harmonic-sum + odd-harmonic octave check, "
                "scripts/noise_v2_bench_speed.py)",
                "publish_start_s": round(start, 3),
                "publish_s": round(length, 3),
                "margin_db": r["margin_db"],
                "half_delta_rev_s": r["half_delta_rev_s"],
                "margin_family_only_db": r["margin_family_only_db"],
            }
        )
    return points


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--speeds",
        type=Path,
        nargs="+",
        default=[_ROOT / "results/noise_v2/survey/bench_speeds.json"],
        help="one or more bench_speeds JSONs written by scripts/noise_v2_bench_speed.py",
    )
    ap.add_argument("--out", type=Path, default=_ROOT / "results/noise_v2/survey")
    ap.add_argument("--fig-dir", type=Path, default=_ROOT / "docs/explainers/noise-model-v2-plan")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="keep only N reading rows (smoke run)")
    ap.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="also write the ingestion manifest (e.g. "
        "src/data_processing/noise_v2_bench_points.json)",
    )
    args = ap.parse_args(argv)

    speeds = load_speeds(args.speeds)
    rows = speeds["rows"]
    if args.limit:
        rows = rows[: args.limit]

    entries = []
    for entry in CORPORA:
        entries.append({**entry, "stats": corpus_stats(rows, entry["families"])})

    figures: list[str] = []
    if not args.no_figures:
        figures.append(draw_speeds(rows, args.out, args.fig_dir))
        split_fig = draw_multirotor_split(rows, args.out, args.fig_dir)
        if split_fig is not None:
            figures.append(split_fig)

    usable = [r for r in rows if r["usable"]]
    outliers = cell_octave_outliers(rows)
    points = fit_points(rows, outliers)
    rig_points = sorted({f"{r['rig']}|{_condition_key(r)}" for r in usable})
    payload = {
        "speeds_source": [str(p) for p in args.speeds],
        "tolerance_rule": speeds.get("tolerance_rule"),
        "estimator_config": speeds.get("config"),
        "cross_checks": speeds.get("cross_checks"),
        "n_recordings": len(rows),
        "n_usable": len(usable),
        "n_fit_points": len(points),
        "n_rig_condition_points": len(rig_points),
        "rig_condition_points": rig_points,
        "cell_octave_outliers": outliers,
        "publish_max_s": PUBLISH_MAX_S,
        "corpora": entries,
        "figures": figures,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "survey.json").write_text(json.dumps(payload, indent=2, default=str))
    (args.out / "survey_table.md").write_text(survey_table(entries, rows))
    if args.manifest_out is not None:
        manifest = {
            "produced_by": "scripts/noise_v2_corpus_survey.py --manifest-out",
            "speeds_source": [Path(p).name for p in args.speeds],
            "tolerance_rule": speeds.get("tolerance_rule"),
            "publish_max_s": PUBLISH_MAX_S,
            "cell_octave_outliers": outliers,
            "points": points,
        }
        Path(args.manifest_out).write_text(json.dumps(manifest, indent=2, sort_keys=False))
        print(f"wrote {args.manifest_out}: {len(points)} fit points")
    print(
        f"wrote {args.out / 'survey.json'} and {args.out / 'survey_table.md'}: "
        f"{len(rows)} recordings, {len(usable)} usable, {len(points)} fit points, "
        f"{len(rig_points)} distinct rig/condition points"
    )
    for e in entries:
        st = e["stats"]
        print(
            f"  {e['corpus']:42s} {e['verdict']:28s} {st['n_usable']}/{st['n_recordings']} usable"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
