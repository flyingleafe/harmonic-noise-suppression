"""Do the trained RPS predictors read DREGON-LIKE SYNTHETIC audio of their own family?

The legacy ``rig_*`` arms beat every noise-model-v2 arm on REAL DREGON cruise,
and this asks the narrow question that separates the two explanations: is the
legacy advantage visible already on synthetic audio built from the legacy
DREGON rig, or is it something the synthetic families cannot show? So every
model here is scored on five conditions of DREGON-like clips rendered by
``notebooks/noise_lab.py`` — the same objects the arms' pools render from —
with no real audio anywhere in the loop.

Five conditions are defined; the DEFAULT run scores the three of
:data:`DEFAULT_CONDITIONS` on :data:`DEFAULT_MODELS`, at :data:`N_CLIPS` clips
each (seeds ``0..N_CLIPS-1``, 8 s, 1 mic, ``level=("window", 0.1)``), which is
what a CPU run costs. The other two conditions and the HPPNet models are one
``--models`` / ``--conditions`` flag away.

``legacy_bank_dregon``            a legacy ``rig_easy`` bank entry from the
                                  bank's DREGON half, on the ``rig_easy_5050``
                                  arm's own trajectory stream.
``legacy_fit_dregon``             the stage-2 DREGON anchor the bank was drawn
                                  around (``dregon_cruise_refined``, donor
                                  dynamics), same trajectories.
``legacy_bank_dregon_fittedtraj`` the same bank entries flown on the FITTED
                                  DREGON trajectory model instead — real-like
                                  DREGON speeds, without the arm's log-uniform
                                  ``rps_scale`` rescale.
``v2_bank_dregon``                a noise-v2 ``easy`` bank entry with
                                  ``traj_rig == "dregon"``, on the
                                  ``noise_v2_easy_5050`` arm's own stream (that
                                  stream draws its OWN entry for the
                                  trajectory, so a window's ``traj_rig`` is
                                  recorded per clip and need not be DREGON).
``v2_fit_dregon``                 the winning v2 DREGON fit on the fitted
                                  DREGON trajectory.

WHICH HALF OF A BANK IS DREGON. ``rig_easy_n2048.json``'s
``provenance.anchors`` lists two anchors of 1024 entries each, michaels
(``results/S2/cruise_8clip.json``) first and DREGON
(``results/S2/dregon_room2_cruise_refined.json``) second, and the entries carry
no anchor field of their own. The halves are told apart by what the builder's
exponent bound does to them: DREGON's fitted ``floor_exp`` is -3.711 and is
clipped to 0, so 75.7 % of entries ``[1024:2048]`` have
``amp_rps_exponent_floor == 0`` against 10.2 % of ``[0:1024]``, and the DREGON
anchor's 109-harmonic ladder appears in the second half against michaels' 111
in the first. Hence :data:`LEGACY_DREGON_RANGE`. The v2 bank states it
outright: every entry carries ``traj_rig``, and ``[0:1024]`` is DREGON.

Scoring is the benchmark's own: ``rps_bench.pit_mae(align_rps_to_gt(pred, gt),
gt)`` on the model's DEPLOYED readout, i.e. what ``rps_bench.overlay`` puts in
``meta.mae`` and what ``scripts/_regime_decomp.py`` measured the real numbers
on. The rotor peak-to-peak spread (``_regime_decomp.rotor_spread``) is carried
beside it so a cell says whether a model collapsed the four rotors onto one
speed.

``findings.md`` is regenerated from ``scores.json``; a ``## Verdict`` section
already in the file is preserved verbatim, so re-running refreshes the tables
without touching the authored reading of them.

    python scripts/_legacy_self_check.py
    python scripts/_legacy_self_check.py --clips 2 --models rig_easy_scv2_unified
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import tdseries as td

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "notebooks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import noise_lab as nl  # noqa: E402
from _regime_decomp import rotor_spread  # noqa: E402

SR = 16000
HOP = 512
DURATION_S = 8.0
N_CLIPS = 6
LEVEL = ("window", 0.1)
OUT_DIR = REPO_ROOT / "results/legacy_self_check"
DECOMP_DIR = REPO_ROOT / "results/regime_decomp"

#: Entry range of each bank's DREGON half (see the module docstring).
LEGACY_DREGON_RANGE = (1024, 2048)
V2_DREGON_RANGE = (0, 1024)


def _spread(lo: int, hi: int, n: int) -> list[int]:
    """``n`` entries evenly spread over ``[lo, hi)`` — a fixed, stated draw."""
    step = (hi - lo) // n
    return [lo + i * step for i in range(n)]


#: Heading of the reference column: the model's REAL DREGON cruise MAE.
REF_COL = "real DREGON cruise"


#: label -> (experiment, checkpoint). ``best_real_overall`` is the checkpoint
#: the transfer campaign selects on; ``hppnet_l2_r2_s0`` is a real-data run and
#: has only ``best``.
MODELS: dict[str, tuple[str, str]] = {
    "rig_easy_scv2_unified": ("rig_easy_scv2_unified", "best_real_overall"),
    "rig_hard_scv2_unified": ("rig_hard_scv2_unified", "best_real_overall"),
    "rig_easy_hppnet_l2_unified": ("rig_easy_hppnet_l2_unified", "best_real_overall"),
    "nv2_easy_scv2": ("nv2_easy_scv2", "best_real_overall"),
    "nv2_hard_scv2": ("nv2_hard_scv2", "best_real_overall"),
    "nv2_easy_hppnet_l2": ("nv2_easy_hppnet_l2", "best_real_overall"),
    "nv2_hard_hppnet_l2": ("nv2_hard_hppnet_l2", "best_real_overall"),
    "real_r4_scv2_unified": ("real_r4_scv2_unified", "best_real_overall"),
    "hppnet_l2_r2_s0": ("hppnet_l2_r2_s0", "best"),
}

#: The models the committed run scores: the scv2 family on one architecture, so
#: the legacy / v2 / real-trained comparison is not confounded by the head. The
#: HPPNet entries above stay reachable through ``--models``.
DEFAULT_MODELS = (
    "rig_easy_scv2_unified",
    "rig_hard_scv2_unified",
    "nv2_easy_scv2",
    "nv2_hard_scv2",
    "real_r4_scv2_unified",
)

#: The conditions the committed run scores: each family's bank on its own arm's
#: trajectory stream, plus the legacy DREGON anchor the bank was drawn around.
#: The two fitted-trajectory conditions stay reachable through ``--conditions``.
DEFAULT_CONDITIONS = ("legacy_bank_dregon", "legacy_fit_dregon", "v2_bank_dregon")

TrajFn = Callable[[int], Any]
SourceFn = Callable[[int, int], Any]


def _legacy_stream(seed: int) -> Any:
    return nl.trajectory("legacy_stream", seed=seed, duration_s=DURATION_S, policy="rig_easy_5050")


def _v2_stream(seed: int) -> Any:
    return nl.trajectory("v2_stream", seed=seed, duration_s=DURATION_S, policy="noise_v2_easy_5050")


def _fitted_dregon(seed: int) -> Any:
    return nl.trajectory("fitted", rig="dregon", duration_s=DURATION_S, seed=seed)


#: name -> (source per clip index, trajectory per seed, one-line description).
CONDITIONS: dict[str, tuple[SourceFn, TrajFn, str]] = {
    "legacy_bank_dregon": (
        lambda i, n: nl.LegacyBank("easy", _spread(*LEGACY_DREGON_RANGE, n)[i]),
        _legacy_stream,
        "legacy rig_easy bank, DREGON half, on the rig_easy_5050 stream",
    ),
    "legacy_fit_dregon": (
        lambda _i, _n: nl.LegacyFit("dregon_cruise_refined"),
        _legacy_stream,
        "stage-2 DREGON anchor (donor dynamics) on the rig_easy_5050 stream",
    ),
    "legacy_bank_dregon_fittedtraj": (
        lambda i, n: nl.LegacyBank("easy", _spread(*LEGACY_DREGON_RANGE, n)[i]),
        _fitted_dregon,
        "legacy rig_easy bank, DREGON half, on the fitted DREGON trajectory",
    ),
    "v2_bank_dregon": (
        lambda i, n: nl.V2Bank("easy", _spread(*V2_DREGON_RANGE, n)[i]),
        _v2_stream,
        "noise-v2 easy bank, DREGON entries, on the noise_v2_easy_5050 stream",
    ),
    "v2_fit_dregon": (
        lambda _i, _n: nl.V2Fit("dregon"),
        _fitted_dregon,
        "winning v2 DREGON fit on the fitted DREGON trajectory",
    ),
}

#: Trajectory-meta fields worth keeping per clip (absent keys are skipped).
TRAJ_META_KEYS = (
    "traj_kind",
    "traj_rig",
    "traj_rig_drawn",
    "policy",
    "rps_scale",
    "rps_scale_drawn",
    "rps_scale_range",
    "window_start_s",
    "flight_hover_rev_s",
    "stream_entry",
    "hover_rev_s",
    "amp_rps_ref",
)


def repo_head() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def real_dregon_cruise() -> dict[str, float]:
    """``experiment -> by_rig.dregon.regimes.cruise.mae`` over every decomposition."""
    out: dict[str, float] = {}
    for path in sorted(DECOMP_DIR.glob("*.json")):
        payload = json.loads(path.read_text())
        rows = payload["rows"] if isinstance(payload, dict) else payload
        for row in rows:
            cell = row.get("by_rig", {}).get("dregon", {}).get("regimes", {}).get("cruise", {})
            if "mae" in cell:
                out[str(row["experiment"])] = float(cell["mae"])
    return out


def build_frame(rendered: td.Frame) -> td.Frame:
    """A bench frame from a noise_lab render: mono ``mixture`` + ``rps`` label.

    Built exactly as ``online_mixing._render_rps_sample`` builds a training
    frame — ``n_frames = T // hop + 1``, the carrier interpolated at the STFT
    frame times of the 16 kHz track the audio was rendered FROM.
    """
    from data_processing.frames import audio_series, rps_series

    audio = np.asarray(rendered["audio"].data, dtype=np.float32)
    n_frames = audio.shape[-1] // HOP + 1
    times = np.arange(n_frames) * HOP / SR
    labels = np.asarray(rendered["rps_render"].interpolate(times), dtype=np.float32)
    return td.Frame(
        {
            "mixture": audio_series(audio[:1], SR),
            "rps": rps_series(labels, sample_rate=SR, hop_length=HOP),
            "meta": td.Frame({"sample_id": 0, "task": "rps_prediction"}),
        }
    )


def clip_record(name: str, clip: int, seed: int, rendered: td.Frame, gt: np.ndarray) -> dict:
    """What this clip IS: source, trajectory, the draw, and what the label does."""
    meta = dict(rendered["meta"].items())
    spread = rotor_spread(gt.astype(np.float64))
    record: dict[str, Any] = {
        "condition": name,
        "clip": clip,
        "seed": seed,
        "source": str(meta["source"]),
        "generation": str(meta["generation"]),
        "entry": str(meta["entry"]),
        "level_gain": float(meta["level_gain"]),
        "rms": float(meta["rms"]),
        "label_rev_s_mean": float(gt.mean()),
        "label_rev_s_min": float(gt.min()),
        "label_rev_s_max": float(gt.max()),
        "label_spread_mean": float(spread.mean()),
    }
    for key in TRAJ_META_KEYS:
        if key in meta and meta[key] is not None:
            value = meta[key]
            record[key] = list(value) if isinstance(value, list | tuple) else value
    return record


def score(
    conditions: list[str], models: list[tuple[str, str, str]], n_clips: int, device: str
) -> dict:
    """Every (model, condition, clip) cell, plus the clip records."""
    import experiments.rps_bench as rb

    clips: dict[str, list[dict]] = {}
    scores: dict[str, dict[str, list[dict]]] = {}
    for name in conditions:
        source_fn, traj_fn, _doc = CONDITIONS[name]
        clips[name] = []
        scores[name] = {label: [] for label, _e, _c in models}
        for clip in range(n_clips):
            t0 = time.time()
            traj = traj_fn(clip)
            rendered = nl.render(source_fn(clip, n_clips), traj, seed=clip, n_mics=1, level=LEVEL)
            frame = build_frame(rendered)
            gt = np.asarray(frame["rps"].data, dtype=np.float64)
            clips[name].append(clip_record(name, clip, clip, rendered, gt))
            for label, exp, ckpt in models:
                pred = rb._live_pred(exp, frame, device, ckpt, "deployed")
                aligned = rb.align_rps_to_gt(pred, gt)
                scores[name][label].append(
                    {
                        "clip": clip,
                        "mae": float(rb.pit_mae(aligned, gt)),
                        "pred_spread_mean": float(rotor_spread(aligned).mean()),
                        "label_spread_mean": float(rotor_spread(gt).mean()),
                    }
                )
            print(f"  {name} clip {clip:2d}  {time.time() - t0:5.1f}s", flush=True)
    return {"clips": clips, "scores": scores}


def quartiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    q1, med, q3 = (float(v) for v in np.percentile(arr, [25, 50, 75]))
    return {"median": med, "q1": q1, "q3": q3, "mean": float(arr.mean()), "n": int(arr.size)}


def summarise(scores: dict[str, dict[str, list[dict]]]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for name, per_model in scores.items():
        for label, cells in per_model.items():
            row = out.setdefault(label, {})
            row[name] = {
                "mae": quartiles([c["mae"] for c in cells]),
                "pred_spread": quartiles([c["pred_spread_mean"] for c in cells]),
                "label_spread": quartiles([c["label_spread_mean"] for c in cells]),
            }
    return out


def _table(header: str, rows: list[str], conditions: list[str], reference: str | None) -> str:
    cols = ["model", *conditions] + ([reference] if reference else [])
    out = [header, "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out.extend(rows)
    return "\n".join(out) + "\n"


def findings_tables(payload: dict) -> str:
    conditions = list(payload["conditions"])
    summary = payload["summary"]
    reference = payload["real_dregon_cruise_mae"]
    labels = list(payload["models"])

    median_rows, iqr_rows, spread_rows = [], [], []
    for label in labels:
        ref = reference.get(label)
        ref_txt = "n/a" if ref is None else f"{ref:.2f}"
        cells = [summary[label][c]["mae"] for c in conditions]
        median_rows.append(
            "| " + " | ".join([label, *[f"{c['median']:.2f}" for c in cells], ref_txt]) + " |"
        )
        iqr_rows.append(
            "| "
            + " | ".join([label, *[f"{c['q1']:.2f}-{c['q3']:.2f}" for c in cells], ref_txt])
            + " |"
        )
        pred = [summary[label][c]["pred_spread"]["median"] for c in conditions]
        spread_rows.append("| " + " | ".join([label, *[f"{v:.2f}" for v in pred]]) + " |")

    label_spread = [
        f"{summary[labels[0]][c]['label_spread']['median']:.2f} rev/s ({c})" for c in conditions
    ]
    parts = [
        "# Legacy self-check: trained predictors on DREGON-like synthetic clips",
        "",
        f"Generated by `scripts/_legacy_self_check.py` at `{payload['git_head'][:12]}`; "
        f"{payload['n_clips']} clips per condition, {DURATION_S:.0f} s, 1 mic, "
        f"level `{LEVEL[0]} {LEVEL[1]}`, deployed readout, CPU. "
        "Numbers are PIT MAE in rev/s; the reference column is each model's REAL "
        "DREGON cruise MAE from `results/regime_decomp/` "
        "(`by_rig.dregon.regimes.cruise.mae`).",
        "",
        "Conditions:",
        "",
        *[f"- `{name}` — {CONDITIONS[name][2]}" for name in conditions],
        "",
        _table("## Median PIT MAE (rev/s) over the clips", median_rows, conditions, REF_COL),
        "",
        _table("## Interquartile range (q1-q3)", iqr_rows, conditions, REF_COL),
        "",
        _table(
            "## Median frame-mean predicted rotor peak-to-peak spread (rev/s)",
            spread_rows,
            conditions,
            None,
        ),
        "",
        "The labels' own median spread per condition: " + ", ".join(label_spread) + ".",
        "",
    ]
    return "\n".join(parts)


def write_findings(path: Path, payload: dict) -> None:
    """Rewrite the tables, keeping any authored ``## Verdict`` section."""
    verdict = ""
    if path.is_file():
        text = path.read_text()
        head = text.find("## Verdict")
        if head >= 0:
            verdict = "\n" + text[head:].rstrip() + "\n"
    path.write_text(findings_tables(payload) + verdict)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", type=int, default=N_CLIPS)
    ap.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    import experiments.rps_bench as rb

    warnings.filterwarnings("ignore", category=RuntimeWarning, module="noise_lab")
    specs = rb.model_specs({label: MODELS[label] for label in args.models})
    reference = real_dregon_cruise()

    t0 = time.time()
    result = score(args.conditions, specs, int(args.clips), str(args.device))
    payload = {
        "git_head": repo_head(),
        "generated_s": round(time.time() - t0, 1),
        "n_clips": int(args.clips),
        "duration_s": DURATION_S,
        "n_mics": 1,
        "level": list(LEVEL),
        "readout": "deployed",
        "rate": [SR, HOP],
        "legacy_dregon_range": list(LEGACY_DREGON_RANGE),
        "v2_dregon_range": list(V2_DREGON_RANGE),
        "legacy_bank_indices": _spread(*LEGACY_DREGON_RANGE, int(args.clips)),
        "v2_bank_indices": _spread(*V2_DREGON_RANGE, int(args.clips)),
        "models": {label: {"experiment": e, "ckpt": c} for label, e, c in specs},
        "conditions": {name: CONDITIONS[name][2] for name in args.conditions},
        "real_dregon_cruise_mae": {label: reference.get(exp) for label, exp, _c in specs},
        **result,
        "summary": summarise(result["scores"]),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "scores.json").write_text(json.dumps(payload, indent=1) + "\n")
    write_findings(args.out / "findings.md", payload)
    print(findings_tables(payload))


if __name__ == "__main__":
    main()
