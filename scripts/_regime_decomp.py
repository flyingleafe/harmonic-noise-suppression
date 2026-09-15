"""Four-regime decomposition of the real-validation score of a checkpoint.

``scripts/valid_regime_eval.py`` splits the frozen real split three ways
(``zero`` / ``low`` / ``flight``) on the target LEVEL alone. That lumps a
spin-up together with a hover at the same mean speed, and those are the two
things the transfer arms are most likely to differ on: a model can track a
steady 80 rev/s comb and still lose the rotor completely while the speed is
moving. This module splits four ways, taking the SLEW first:

    ramp     |d(mean target)/dt| >= 20 rev/s^2      (a transition, any level)
    zero     not ramp, max target < 1 rev/s         (every rotor stopped)
    standby  not ramp, mean target < 45 rev/s       (idling / warm-up)
    cruise   not ramp, mean target >= 45 rev/s      (steady flight)

The derivative is a central difference over +-0.25 s of the rotor-mean target
speed, boxcar-smoothed over 0.288 s first (the real labels are piecewise-held,
so an unsmoothed difference of the raw track is a train of spikes).

THE THRESHOLDS ARE STATED, NOT FITTED. Nothing here was tuned on a model's
error. They were chosen from the label statistics alone, and the split they
produce is comfortable: on this split the measured |slew| is ~0.07 rev/s^2 at
the median of standby-level frames and ~0.8 at the median of cruise-level
frames, while the frames above 20 reach ~28 at their p90 — so 20 rev/s^2 sits
in an empty gap, and moving it by a factor of two either way relabels almost
nothing. Every boundary is a function of the TARGET only, so the labelling is
identical for every model and the cells are directly comparable.

Matching is per-frame Hungarian, pooled over channels and clips, imported from
``valid_regime_eval`` — the convention every rotor-speed number in this project
uses.

    python scripts/_regime_decomp.py
    python scripts/_regime_decomp.py --exp rig_easy_scv2_unified --ckpt best
    python scripts/_regime_decomp.py --channels 1 --limit 4      # smoke run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from valid_regime_eval import RIGS, VALID, clip_rigs, pit_abs_error  # noqa: E402

#: Regime order used by every table, JSON key and figure in this decomposition.
REGIMES = ("zero", "standby", "ramp", "cruise")

#: The two finished scv2 transfer arms plus the real-data-trained reference the
#: whole comparison is against — all three are part of the deliverable table.
DEFAULT_EXPERIMENTS = (
    "rig_easy_scv2_unified",
    "rig_hard_scv2_unified",
    "real_r4_scv2_unified",
)

#: The BEST-ROUND value of ``val/real_overall`` each run ever logged, read off
#: the wandb history, with ``(round, total rounds)``.
#:
#: This is a DIFFERENT OBJECT from the checkpoint scored below, and confusing
#: the two puts a wrong number in a talk. ``training.loop`` aliases
#: ``best_<score>.ckpt`` when the MEDIAN-SMOOTHED score improves, so the file on
#: R2 is the last round whose smoothed score improved — not the round with the
#: best raw score. Every per-regime number in this file belongs to the SELECTED
#: file (the one inference actually ran on); these are quoted beside it so the
#: gap is visible rather than discovered later.
HISTORY_BEST = {
    "rig_easy_scv2_unified": {"raw": 5.717, "round": 29, "rounds": 94},
    "rig_hard_scv2_unified": {"raw": 5.370, "round": 102, "rounds": 137},
    "real_r4_scv2_unified": {"raw": 2.991, "round": 48, "rounds": 70},
}

FPS = 16000 / 512  # the STFT frame rate the rps target lives on: 31.25 Hz

#: Every number that defines the split, recorded verbatim in the JSON output.
THRESHOLDS = {
    "ramp_slew_rev_s2": 20.0,
    "zero_max_rev_s": 1.0,
    "standby_mean_rev_s": 45.0,
    "derivative_halfwidth_s": 0.25,
    "smoothing_width_s": 9 / FPS,
    "frame_rate_hz": FPS,
}

_SMOOTH_W = 9  # 0.288 s boxcar, odd so it is centred
_DERIV_HALF = int(round(0.25 * FPS))  # 8 frames -> a +-0.256 s central difference


def _smooth(x: np.ndarray, width: int = _SMOOTH_W) -> np.ndarray:
    """Centred boxcar moving average with edge padding, length preserved."""
    if width < 3:
        return x
    pad = width // 2
    return np.convolve(np.pad(x, pad, mode="edge"), np.ones(width) / width, mode="valid")


def mean_speed_slew(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(mean_speed, |d mean_speed/dt|)`` from the ``(R, F)`` target speeds.

    Both are ``(F,)``. The derivative is a central difference over
    ``+-_DERIV_HALF`` frames of the smoothed rotor mean, with the indices
    clamped at the clip edges (so the first and last quarter-second see a
    one-sided difference over a shorter base — still the right units, just
    noisier; no clip in the split starts or ends mid-slew).
    """
    mean = target.mean(axis=0)
    smoothed = _smooth(mean)
    idx = np.arange(mean.size)
    hi = np.minimum(idx + _DERIV_HALF, mean.size - 1)
    lo = np.maximum(idx - _DERIV_HALF, 0)
    span = (hi - lo) / FPS
    slew = np.divide(smoothed[hi] - smoothed[lo], span, out=np.zeros_like(mean), where=span > 0)
    return mean, np.abs(slew)


def frame_regimes4(target: np.ndarray) -> np.ndarray:
    """``(F,)`` four-regime label per frame, from the ``(R, F)`` target speeds.

    Slew first, then level — a spin-up through 45 rev/s is a ``ramp``, not half
    a ``standby`` and half a ``cruise``.
    """
    mean, slew = mean_speed_slew(target)
    ramp = slew >= THRESHOLDS["ramp_slew_rev_s2"]
    labels = np.full(target.shape[1], "cruise", dtype=object)
    labels[~ramp & (mean < THRESHOLDS["standby_mean_rev_s"])] = "standby"
    labels[~ramp & (target.max(axis=0) < THRESHOLDS["zero_max_rev_s"])] = "zero"
    labels[ramp] = "ramp"
    return labels


def pit_align(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """``pred`` rows permuted onto ``target``'s rows, independently per frame.

    THE METRIC'S ALIGNMENT, AND ONLY THE METRIC'S. The scoring convention
    matches per frame, so ``|pit_align(pred, target) - target|`` is exactly the
    error :func:`valid_regime_eval.pit_abs_error` pools, kept in target-row
    order.

    Nothing draws through this. Two rotors of a real quadrotor sit within a
    rev/s of each other most of the time, so the per-frame assignment flips at
    near-zero cost difference and a line drawn through it hops between target
    rows while the model's output has not changed — one flip every ~5 frames as
    measured on this split. The figures plot the model's four output series and
    the four target series as they are, with no permutation anywhere.
    """
    from scipy.optimize import linear_sum_assignment

    out = np.empty_like(target, dtype=np.float64)
    for i in range(target.shape[1]):
        cost = np.abs(pred[:, None, i] - target[None, :, i])
        rows, cols = linear_sum_assignment(cost)
        out[cols, i] = pred[rows, i]
    return out


# ─── Provenance ───────────────────────────────────────────────────────────────


def split_version() -> str:
    """The pinned content hash of the materialized frozen split."""
    from data_processing.streams import ensure_local

    return Path(ensure_local(VALID.removeprefix("dload:"))).name


def checkpoint_uri(experiment: str, ckpt: str) -> str:
    """The ``r2://`` (or local) reference ``zoo.load`` will resolve for this arm."""
    from hydra import compose, initialize_config_dir

    from training.config import register_configs
    from zoo.frame_model import _checkpoint_ref

    register_configs()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=[f"experiment={experiment}"])
    return _checkpoint_ref(experiment, ckpt, cfg)


def checkpoint_digest(uri: str) -> str:
    """``sha256:<hex>`` of the resolved checkpoint file."""
    from utils.checkpoints import resolve_checkpoint_uri

    local = Path(resolve_checkpoint_uri(uri, REPO_ROOT / ".cache" / "r2_checkpoints"))
    h = hashlib.sha256()
    with local.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return f"sha256:{h.hexdigest()}"


def recorded_training_score(experiment: str, score: str = "real_overall") -> dict:
    """What the TRAINING RUN itself logged for the checkpoint it saved.

    ``training.loop`` writes ``checkpoints/best_checkpoints.json`` next to the
    checkpoints, recording for each selection score the validation round the
    alias points at and both the RAW and the MEDIAN-SMOOTHED metric of that
    round. It matters here because ``best_<score>.ckpt`` is aliased when the
    SMOOTHED value improves — so the file on R2 is NOT the round with the best
    raw score, and quoting a run's best raw ``val/real_overall`` as if it were
    this checkpoint's score is wrong. Carrying the recorded row into the output
    makes the comparison checkable instead of a matter of memory.
    """
    from utils.checkpoints import resolve_checkpoint_uri

    uri = f"r2://ml-data/artifacts/{experiment}/checkpoints/best_checkpoints.json"
    try:
        local = resolve_checkpoint_uri(uri, REPO_ROOT / ".cache" / "r2_checkpoints")
        row = json.loads(Path(local).read_text()).get(score, {})
    except Exception as exc:  # noqa: BLE001 — provenance must never break the eval
        return {"error": repr(exc)}
    return {
        "source": uri,
        "score": score,
        "checkpoint": row.get("checkpoint"),
        "validation_round": row.get("validation_round"),
        "optimizer_step": row.get("optimizer_step"),
        "raw": row.get("raw"),
        "smoothed": row.get("smoothed"),
    }


def git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 — provenance must never break the eval
        return "unknown"


# ─── The decomposition ────────────────────────────────────────────────────────


def _cell(values: list[np.ndarray], n_total: int) -> dict:
    """One table cell: PIT MAE, the frames behind it, and its share of the split."""
    if not values:
        return {"mae": None, "frames": 0, "share": 0.0}
    pooled = np.concatenate(values)
    return {
        "mae": float(pooled.mean()),
        "frames": int(pooled.size),
        "share": float(pooled.size / n_total) if n_total else 0.0,
    }


def predict_clip(model, frame, salience: bool, threshold: float = 0.3) -> np.ndarray:
    """``(R, F)`` predicted speeds for one frame, whichever head the model has."""
    if salience:
        from valid_regime_eval import salience_rps_pred

        return salience_rps_pred(model.model, frame, threshold)
    return np.asarray(model(frame)["rps_pred"].data, dtype=np.float64)


def decompose(
    experiment: str,
    ckpt: str,
    channels: int = 8,
    limit: int | None = None,
    device: str = "cpu",
) -> dict:
    """Score one checkpoint on the frozen real split, split four ways.

    Returns the whole row: overall PIT MAE, a cell per regime, a cell per
    (rig, regime), and a per-clip breakdown (used to choose the clips the
    prediction figures show — including the ones the model handles worst).
    """
    import torch

    import zoo
    from data_processing.frame_datasets import DregonLMFrameDataset
    from metrics.rps import batched_pit_mae

    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = zoo.load(experiment, ckpt=ckpt, device=device)
    salience = bool(getattr(getattr(model, "model", None), "outputs_salience", False))
    rigs = clip_rigs()

    everything: list[np.ndarray] = []
    by_regime: dict[str, list[np.ndarray]] = {r: [] for r in REGIMES}
    by_rig: dict[str, dict[str, list[np.ndarray]]] = {rig: {r: [] for r in REGIMES} for rig in RIGS}
    per_clip: dict[int, dict] = {}
    # The metric training actually reports as ``val/real_overall``: ONE
    # permutation per clip (``metrics.rps.batched_pit_mae`` minimises the
    # time-averaged cost), not one per frame. It is always >= the per-frame
    # number, and computing both here is what lets the decomposition be checked
    # against the run's own logged score instead of asserted to match it.
    clip_level: list[float] = []

    for channel in range(channels):
        dataset = DregonLMFrameDataset(
            data_dir=VALID, n_fft=2048, hop_length=512, sample_rate=16000, channel=channel
        )
        n = len(dataset) if limit is None else min(limit, len(dataset))
        for i in range(n):
            frame = dataset[i]
            target = np.asarray(frame["rps"].data, dtype=np.float64)
            pred = predict_clip(model, frame, salience)
            width = min(pred.shape[1], target.shape[1])
            target, pred = target[:, :width], pred[:, :width]
            err = pit_abs_error(pred, target)
            labels = frame_regimes4(target)
            rig = rigs[i] if i < len(rigs) else "dregon"

            everything.append(err.ravel())
            clip = per_clip.setdefault(
                i, {"clip": i, "rig": rig, "errors": [], "regime_errors": {r: [] for r in REGIMES}}
            )
            clip["errors"].append(err.ravel())
            for regime in REGIMES:
                mask = labels == regime
                if not mask.any():
                    continue
                vals = err[:, mask].ravel()
                by_regime[regime].append(vals)
                by_rig[rig][regime].append(vals)
                clip["regime_errors"][regime].append(vals)
            clip_level.append(
                float(
                    batched_pit_mae(
                        torch.as_tensor(pred[None], dtype=torch.float32),
                        torch.as_tensor(target[None], dtype=torch.float32),
                    )[0]
                )
            )

    n_total = int(sum(v.size for v in everything))
    row = {
        "experiment": experiment,
        "ckpt": ckpt,
        "channels": channels,
        "salience": salience,
        "overall": {
            "mae": float(np.concatenate(everything).mean()),
            "frames": n_total,
            "share": 1.0,
        },
        "clip_level_pit_mae": float(np.mean(clip_level)),
        "regimes": {r: _cell(by_regime[r], n_total) for r in REGIMES},
        "by_rig": {},
        "per_clip": [],
    }
    for rig in RIGS:
        pooled = [v for vals in by_rig[rig].values() for v in vals]
        n_rig = int(sum(v.size for v in pooled))
        row["by_rig"][rig] = {
            "overall": _cell(pooled, n_rig),
            "regimes": {r: _cell(by_rig[rig][r], n_rig) for r in REGIMES},
        }
    for i in sorted(per_clip):
        clip = per_clip[i]
        n_clip = int(sum(v.size for v in clip["errors"]))
        row["per_clip"].append(
            {
                "clip": i,
                "rig": clip["rig"],
                "overall": _cell(clip["errors"], n_clip),
                "regimes": {r: _cell(clip["regime_errors"][r], n_clip) for r in REGIMES},
            }
        )
    return row


# ─── Output ───────────────────────────────────────────────────────────────────


def _fmt(cell: dict) -> str:
    """A cell's MAE, or ``--`` when no frame of the split lands in it."""
    if cell["mae"] is None:
        return "    --"
    return f"{cell['mae']:6.2f}"


#: A cell is too thin to read as a result below this many DISTINCT clip frames.
#: Rotor-frames overcount by 4 rotors x 8 mics, and eight mic views of the same
#: 32 ms of flight are not eight independent observations — 200 frames is 6.4 s.
THIN_FRAMES = 200


def table(rows: list[dict]) -> str:
    out: list[str] = []
    head = f"{'experiment':26s} {'rig':9s} {'overall':>8s}" + "".join(f" {r:>8s}" for r in REGIMES)
    out.append(head)
    out.append("-" * len(head))
    for row in rows:
        out.append(
            f"{row['experiment']:26s} {'both':9s} {_fmt(row['overall']):>8s}"
            + "".join(f" {_fmt(row['regimes'][r]):>8s}" for r in REGIMES)
        )
    out.append("")
    for rig in RIGS:
        for row in rows:
            block = row["by_rig"][rig]
            out.append(
                f"{row['experiment']:26s} {rig:9s} {_fmt(block['overall']):>8s}"
                + "".join(f" {_fmt(block['regimes'][r]):>8s}" for r in REGIMES)
            )
        out.append("")
    ref = rows[0]
    per_frame = max(1, int(ref["channels"]) * 4)  # rotor-frames per distinct clip frame
    out.append(
        f"{'clip frames / share':26s} {'both':9s} {ref['overall']['frames'] // per_frame:8d}"
        + "".join(f" {ref['regimes'][r]['frames'] // per_frame:8d}" for r in REGIMES)
    )
    out.append(
        f"{'':26s} {'':9s} {'100.0%':>8s}"
        + "".join(f" {100 * ref['regimes'][r]['share']:7.1f}%" for r in REGIMES)
    )
    for rig in RIGS:
        block = ref["by_rig"][rig]
        out.append(
            f"{'':26s} {rig:9s} {block['overall']['frames'] // per_frame:8d}"
            + "".join(f" {block['regimes'][r]['frames'] // per_frame:8d}" for r in REGIMES)
        )
        out.append(
            f"{'':26s} {'':9s} {'100.0%':>8s}"
            + "".join(f" {100 * block['regimes'][r]['share']:7.1f}%" for r in REGIMES)
        )
    thin = [
        f"{rig}/{r} ({ref['by_rig'][rig]['regimes'][r]['frames'] // per_frame} clip frames, "
        f"{ref['by_rig'][rig]['regimes'][r]['frames'] // per_frame / FPS:.1f} s)"
        for rig in RIGS
        for r in REGIMES
        if 0 < ref["by_rig"][rig]["regimes"][r]["frames"] // per_frame < THIN_FRAMES
    ]
    if thin:
        out.append("")
        out.append(
            f"TOO THIN TO READ AS A RESULT (< {THIN_FRAMES} distinct clip frames): "
            + ", ".join(thin)
        )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", nargs="+", default=list(DEFAULT_EXPERIMENTS))
    ap.add_argument(
        "--ckpt", default="best_real_overall", help="checkpoint selector for every --exp"
    )
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="clips per channel (smoke runs)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/regime_decomp/scv2_regimes.json")
    args = ap.parse_args()

    rows: list[dict] = []
    for experiment in args.exp:
        uri = checkpoint_uri(experiment, args.ckpt)
        row = decompose(experiment, args.ckpt, args.channels, args.limit, args.device)
        rec = recorded_training_score(experiment)
        row["provenance"] = {
            "experiment": experiment,
            "checkpoint": args.ckpt,
            "checkpoint_uri": uri,
            "checkpoint_digest": checkpoint_digest(uri),
            "split": VALID,
            "split_version": split_version(),
            "thresholds": THRESHOLDS,
            "git_head": git_head(),
            "scored_object": (
                "Every per-regime and per-clip number in this row belongs to the "
                "SELECTED CHECKPOINT — the file at checkpoint_uri, aliased by "
                "training.loop when the MEDIAN-SMOOTHED val/real_overall improved. It is "
                "NOT the best-round checkpoint of the run's history: "
                f"selected = round {rec.get('validation_round')} at raw "
                f"val/real_overall {rec.get('raw')}, while the best round the history "
                f"ever logged was {HISTORY_BEST.get(experiment, {}).get('round')} at "
                f"{HISTORY_BEST.get(experiment, {}).get('raw')}. Inference ran on the "
                "selected file only."
            ),
            "selected_checkpoint": rec,
            "best_round_history": HISTORY_BEST.get(experiment),
        }
        rows.append(row)
        print(
            f"scored {experiment} @ {args.ckpt}: overall PIT MAE {row['overall']['mae']:.3f}",
            flush=True,
        )

    print()
    print(table(rows))

    print()
    print("WHICH OBJECT IS THIS?  val/real_overall, two different things")
    head = (
        f"{'experiment':26s} {'best round (history)':>22s} {'selected ckpt (median)':>24s} "
        f"{'this file: clip-PIT':>20s} {'per-frame PIT':>14s}"
    )
    print(head)
    print("-" * len(head))
    for row in rows:
        rec = row["provenance"]["selected_checkpoint"]
        hist = row["provenance"]["best_round_history"] or {}
        left = f"{hist['raw']:.3f} @ r{hist['round']}/{hist['rounds']}" if hist else "unknown"
        raw = rec.get("raw")
        right = f"{raw:.3f} @ r{rec.get('validation_round')}" if raw is not None else "unknown"
        print(
            f"{row['experiment']:26s} {left:>22s} {right:>24s} "
            f"{row['clip_level_pit_mae']:20.3f} {row['overall']['mae']:14.3f}"
        )
    print(
        "  Columns 2 and 3 are the SAME metric on DIFFERENT checkpoints: col 2 is the\n"
        "  best round the run ever logged, col 3 is the round whose file is on R2 as\n"
        "  best_real_overall.ckpt (aliased on the median-smoothed score, hence later\n"
        "  and worse). Col 4 recomputes training's own clip-level metric on that file\n"
        "  here and must agree with col 3. Col 5 is the per-frame Hungarian PIT MAE the\n"
        "  regime table decomposes; it is <= col 4 by construction, since per-frame\n"
        "  matching takes the minimum in every frame. EVERY per-regime number above\n"
        "  belongs to the selected checkpoint (col 3/4), not to the best round (col 2)."
    )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"regimes": list(REGIMES), "rows": rows}, indent=1))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
