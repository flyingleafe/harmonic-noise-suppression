"""Figure payloads for the INTERACTIVE version of the 24-clip model matrix.

`scripts/_model_matrix.py` owns the matrix: it renders the clips, runs the four
checkpoints, caches every prediction under `results/model_matrix/preds/` and
writes `results/model_matrix/matrix.json`. It also draws 24 static PNGs.

This script draws nothing and computes no prediction. It re-reads those caches
and writes, per clip, the small payload a Plotly figure needs:

  * the frame time axis (seconds) shared by both panels;
  * the four TARGET tracks;
  * each model's FOUR RAW output series, in the matrix's own draw order;
  * the four-regime run-length spans (`_regime_decomp.frame_regimes4`);
  * the per-clip PIT MAE of each model, copied from `matrix.json`;
  * a mic-0 spectrogram rendered ONCE to a PNG, with the dB limits and the
    time/frequency extent it must be stretched to inside the upper panel.

Pixels stay pixels: the spectrogram travels as a PNG that the page base64-embeds
into `layout.images`, never as ~250k heatmap data points per clip.

Every prediction is read from the cache and CHECKED against the checkpoint and
audio digests the cache was written with. A missing or stale entry is a hard
error naming the refill command — this script never invents a track.

    PYTHONPATH="$PWD/src:$PWD/scripts" python scripts/_model_matrix_interactive.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _model_matrix as mm  # noqa: E402
import _regime_decomp as rd  # noqa: E402

#: Where the page reads the payloads from. One JSON and one PNG per clip, plus
#: an `index.json` naming the clip order, the models and their colours.
OUT_DIR = REPO_ROOT / "docs/explainers/model-matrix/interactive"

#: Speeds and times are written at this many decimals. The metric is quoted to
#: two, the axis spans tens of rev/s over eight seconds, and 0.01 rev/s is three
#: orders of magnitude below anything the figure can resolve — this is what
#: keeps twenty tracks per clip from dominating the page.
SPEED_DECIMALS = 2
TIME_DECIMALS = 4

#: Frequency rows kept in the spectrogram PNG. The 2048-point STFT gives 1025
#: rows over 0-8 kHz; the upper panel is ~260 px tall on a 1440 px page, so the
#: full grid is ~4x more rows than can ever be shown. Every row is kept only if
#: this is None.
SPEC_ROWS = 512


def refill_command() -> str:
    return 'PYTHONPATH="$PWD/src:$PWD/scripts" python scripts/_model_matrix.py'


def load_pred(tag: str, clip: mm.Clip, ckpt_digest: str) -> np.ndarray:
    """One model's cached raw output for one clip, digest-checked.

    Raises with the refill command if the cache is absent or was written for a
    different checkpoint or different audio.
    """
    cache = mm.PRED_DIR / f"{tag}__{clip.key}.npz"
    if not cache.is_file():
        raise SystemExit(
            f"MISSING prediction cache {cache.relative_to(REPO_ROOT)} — "
            f"refill it with:\n    {refill_command()}"
        )
    blob = np.load(cache, allow_pickle=False)
    if str(blob["ckpt"]) != ckpt_digest or str(blob["clip"]) != clip.digest:
        raise SystemExit(
            f"STALE prediction cache {cache.relative_to(REPO_ROOT)}: written for "
            f"ckpt {str(blob['ckpt'])[:23]}… / audio {str(blob['clip'])[:16]}…, matrix.json "
            f"expects {ckpt_digest[:23]}… / {clip.digest[:16]}… — refill it with:\n"
            f"    {refill_command()}"
        )
    pred = np.asarray(blob["pred"], dtype=np.float64)
    if not np.isfinite(pred).all():
        raise SystemExit(
            f"non-finite values in {cache.relative_to(REPO_ROOT)} — the figure would carry NaN"
        )
    return pred


def spectrogram_png(pr: Any, audio: np.ndarray, path: Path) -> dict[str, Any]:
    """Write the mic-0 spectrogram as a bare PNG and describe how to place it.

    Same transform, colour map and percentile dB limits as the static figure
    (`prepare_regime.draw`), so the interactive upper panel is the same picture.
    """
    import matplotlib.pyplot as plt

    spec = pr.spectrogram_db(audio)  # (1025, N) dB, row 0 = DC
    lo, hi = (float(v) for v in np.percentile(spec, [5.0, 99.8]))
    rows = spec.shape[0]
    if SPEC_ROWS is not None and rows > SPEC_ROWS:
        # Average dB in equal frequency bins: one row per displayed pixel, so
        # the harmonic combs stay legible and no row is silently dropped.
        edges = np.linspace(0, rows, SPEC_ROWS + 1).astype(int)
        spec = np.stack([spec[a:b].mean(axis=0) for a, b in zip(edges[:-1], edges[1:])])
    plt.imsave(path, spec, cmap="magma", vmin=lo, vmax=hi, origin="lower", format="png")
    return {
        "file": path.name,
        "rows": int(spec.shape[0]),
        "cols": int(spec.shape[1]),
        "db_range": [round(lo, 2), round(hi, 2)],
        "freq_khz": [0.0, mm.SAMPLE_RATE / 2000.0],
        "bytes": path.stat().st_size,
    }


def payload(pr: Any, clip: mm.Clip, row: dict[str, Any], preds: dict[str, np.ndarray]) -> dict:
    """Everything one interactive figure needs, and nothing else."""
    width = int(row["frames_scored"])
    target = clip.target[:, :width]
    labels = pr.frame_regimes4(target)
    duration = float(clip.audio.size / mm.SAMPLE_RATE)
    t = np.arange(width) / rd.FPS

    def series(a: np.ndarray) -> list[list[float]]:
        return np.round(a[:, :width], SPEED_DECIMALS).tolist()

    spec = spectrogram_png(pr, clip.audio, OUT_DIR / f"{clip.key}.spec.png")
    stack = np.concatenate([target.ravel()] + [p[:, :width].ravel() for p in preds.values()])
    return {
        "key": clip.key,
        "category": clip.category,
        "rig": clip.rig,
        "label": clip.label,
        "recording": clip.recording,
        "why": clip.why,
        "frames": width,
        "duration_s": round(duration, 4),
        "fps": rd.FPS,
        "time_s": np.round(t, TIME_DECIMALS).tolist(),
        "target": series(target),
        "preds": {tag: series(preds[tag]) for tag in mm.DRAW_ORDER},
        "pit_mae": {tag: float(row["pit_mae"][tag]) for tag in row["pit_mae"]},
        "regimes": [
            {
                "regime": regime,
                "t0": round(a / rd.FPS, TIME_DECIMALS),
                "t1": round(min(b, width) / rd.FPS, TIME_DECIMALS),
                "frames": int(min(b, width) - a),
            }
            for regime, a, b in pr.segments(labels)
        ],
        "speed_range": [round(float(stack.min()), 2), round(float(stack.max()), 2)],
        "visits_zero": bool((labels == "zero").any()),
        "spectrogram": spec,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--n-traj",
        type=int,
        default=None,
        help="stream trajectories; default: whatever matrix.json recorded",
    )
    args = ap.parse_args()

    matrix_path = mm.RESULTS / "matrix.json"
    if not matrix_path.is_file():
        raise SystemExit(
            f"no {matrix_path.relative_to(REPO_ROOT)} — build it with:\n    {refill_command()}"
        )
    matrix = json.loads(matrix_path.read_text())
    rows = {c["key"]: c for c in matrix["clips"]}
    digests = {tag: str(v["digest"]) for tag, v in matrix["models"].items()}
    recorded_traj = int(matrix["clip_set"]["stream_trajectories"])
    n_traj = args.n_traj if args.n_traj is not None else recorded_traj

    wall = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    colours = {tag: colour for _e, _c, tag, colour in mm.MODELS}
    pr = mm.load_prepare_regime([(tag, colours[tag]) for tag in mm.DRAW_ORDER])

    print("rebuilding the clip set from the render cache", flush=True)
    clips, _shared = mm.build_clips(n_traj, rerender=False)
    missing = sorted(set(rows) - {c.key for c in clips})
    if missing:
        raise SystemExit(
            f"matrix.json has clips this build does not: {missing} — rebuild with:\n"
            f"    {refill_command()}"
        )

    entries: list[dict[str, Any]] = []
    for clip in clips:
        row = rows.get(clip.key)
        if row is None:
            raise SystemExit(
                f"clip {clip.key} is not in matrix.json — rebuild with:\n    {refill_command()}"
            )
        if row["audio_sha256"] != clip.digest:
            raise SystemExit(
                f"clip {clip.key} audio digest {clip.digest[:16]}… does not match matrix.json's "
                f"{row['audio_sha256'][:16]}… — the render cache and the scores disagree; "
                f"rebuild with:\n    {refill_command()}"
            )
        preds = {tag: load_pred(tag, clip, digests[tag]) for tag in mm.DRAW_ORDER}
        width = min([clip.target.shape[1]] + [p.shape[1] for p in preds.values()])
        if width != int(row["frames_scored"]):
            raise SystemExit(
                f"clip {clip.key}: cached predictions give {width} scorable frames, matrix.json "
                f"recorded {row['frames_scored']} — rebuild with:\n    {refill_command()}"
            )
        entry = payload(pr, clip, row, preds)
        out = OUT_DIR / f"{clip.key}.json"
        out.write_text(json.dumps(entry, separators=(",", ":"), allow_nan=False))
        entries.append(
            {
                "key": clip.key,
                "category": clip.category,
                "json": out.name,
                "spectrogram": entry["spectrogram"]["file"],
                "frames": entry["frames"],
                "json_bytes": out.stat().st_size,
                "spectrogram_bytes": entry["spectrogram"]["bytes"],
            }
        )
        print(
            f"  {clip.key:<34} {entry['frames']:>4} frames  "
            f"json {out.stat().st_size / 1e3:6.1f} kB  "
            f"png {entry['spectrogram']['bytes'] / 1e3:6.1f} kB  "
            f"({entry['spectrogram']['rows']}x{entry['spectrogram']['cols']})",
            flush=True,
        )

    index = {
        "generated_by": "scripts/_model_matrix_interactive.py",
        "reads": {
            "matrix": str(matrix_path.relative_to(REPO_ROOT)),
            "predictions": str(mm.PRED_DIR.relative_to(REPO_ROOT)),
            "renders": str(mm.RENDER_SPLIT.relative_to(REPO_ROOT)),
        },
        "git_head": matrix["git_head"],
        "metric": matrix["metric"],
        "tracks": "model tracks are RAW model output: no matching, no permutation",
        "models": {
            tag: {"colour": colours[tag], **matrix["models"][tag]}
            for _e, _c, tag, _col in mm.MODELS
        },
        "draw_order": list(mm.DRAW_ORDER),
        "regime_colour": dict(pr.REGIME_COLOUR),
        "regimes": list(rd.REGIMES),
        "spectrogram": {
            "transform": "prepare_regime.spectrogram_db: |rfft| of a 2048-point Hann frame, "
            f"hop {mm.HOP}, in dB",
            "colourmap": "magma",
            "db_limits": "per-clip 5th and 99.8th percentile, as in the static figure",
            "freq_rows": SPEC_ROWS,
            "channel": "mic 0",
        },
        "speed_decimals": SPEED_DECIMALS,
        "clips": entries,
    }
    (OUT_DIR / "index.json").write_text(json.dumps(index, indent=1, default=mm._jsonable))
    total = sum(e["json_bytes"] + e["spectrogram_bytes"] for e in entries)
    print(
        f"\nwrote {len(entries)} clips to {OUT_DIR.relative_to(REPO_ROOT)} "
        f"({total / 1e6:.2f} MB on disk) in {time.time() - wall:.0f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
