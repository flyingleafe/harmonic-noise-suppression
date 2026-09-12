"""Score the two best-on-Michael's-cruise checkpoints across every real clip set.

Selected from ``results/paper_regime_matrix/{ladder,blocks}.csv`` on the
``fly124_cruise`` column (frame-weighted PIT MAE, Michael's HELD-OUT cruise):

* ``hppnet_l2_r2_s0``     — 0.768 rev/s, best overall, task ``salience_rps``;
* ``r2hb_gru_nomix_wu``   — 1.027 rev/s, best REGRESSOR, task ``rps_prediction``.

Both were trained on ``conf/online_mix/hb_silence_dload.yaml``: DREGON-frames
``in_flight_noise`` minus ``free-flight_nosource_room1``, plus Michael's FLY125,
plus a zero-labelled silence arm. So FLY125 and DREGON room2 are TRAINING
recordings; FLY124 and DREGON room1 free-flight are held out.

Every clip is scored on all 8 microphones separately, with the project's own
readouts — ``metrics.salience_layers.LayerPeakRPSMetric`` for the salience model
(per-rotor layer peak + three-point log-parabolic vertex, PIT-aligned) and
``metrics.rps_mae_frame`` for the regressor. Campaign-temporary; delete with the
campaign.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tdseries as td
import torch
import torch.nn.functional as F

import zoo
from experiments.stochastic_fit import data as SD
from metrics.salience_layers import LayerPeakRPSMetric, peak_readout

OUT = Path("results/bestmodel_eval")
MODELS = {
    "hppnet_l2_r2_s0": {"matrix_fly124_cruise": 0.768, "kind": "salience"},
    "r2hb_gru_nomix_wu": {"matrix_fly124_cruise": 1.027, "kind": "regressor"},
}
#: clip sets: label -> (held_out?, [(clip_id, loader)])
FLY124_CRUISE = ["sample_00026", "sample_00030", "sample_00033"]
DREGON_ROOM1 = ["sample_00016", "sample_00018", "sample_00021"]
FLY125_CRUISE = ["FLY125_02", "FLY125_04", "FLY125_07"]
FLY125_IDLE = ["FLY125_00", "FLY125_01"]
DREGON_ROOM2 = [
    "free-flight_nosource_room2_00",
    "hovering_nosource_room2_00",
    "updown_nosource_room2_01",
]
#: one figure per (model, clip) for these
FIGURE_CLIPS = FLY124_CRUISE + DREGON_ROOM1 + FLY125_CRUISE + FLY125_IDLE + DREGON_ROOM2


def _ref(clip_id: str) -> SD.Clip:
    return SD.load_ref_clip(next(p for p in SD.ref_clips() if p.stem == clip_id))


CLIP_SETS: list[tuple[str, bool, list[str], Callable[[str], SD.Clip]]] = [
    ("Michael's FLY124 cruise", True, FLY124_CRUISE, SD.load_valid_clip),
    ("DREGON room1 free-flight", True, DREGON_ROOM1, SD.load_valid_clip),
    ("Michael's FLY125 cruise", False, FLY125_CRUISE, _ref),
    ("Michael's FLY125 idle", False, FLY125_IDLE, _ref),
    ("DREGON room2 flight", False, DREGON_ROOM2, _ref),
]


def frame_of(audio_ch: np.ndarray, sr: int) -> td.Frame:
    return td.Frame(
        {
            "mixture": td.Series(
                audio_ch.astype(np.float32),
                dims=("time",),
                indexes={"time": td.GridIndex(sr_num=int(sr), size=int(audio_ch.size))},
            )
        }
    )


def rps_frame(rps: np.ndarray, sr: int) -> td.Frame:
    return td.Frame(
        {
            "rps": td.Series(
                rps.astype(np.float32),
                dims=("rotor", "time"),
                indexes={"time": td.GridIndex(sr_num=int(sr), size=int(rps.shape[1]))},
            )
        }
    )


def predict(
    fm: Any, kind: str, metric: LayerPeakRPSMetric, clip: SD.Clip, mic: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """``(pred (R,T), truth-on-pred-grid (R,T), mae)`` for one microphone."""
    out = fm(frame_of(clip.audio[mic], clip.sr))
    tgt = rps_frame(clip.rps, clip.sr)
    if kind == "salience":
        layers, rps_grid = metric._unpack(out, tgt)
        pred = peak_readout(F.logsigmoid(layers), metric._freqs)[0].numpy()
        truth = rps_grid[0].numpy()
    else:
        entry = next(k for k in list(out.keys()) if k != "meta")
        pred = np.asarray(out[entry].data, dtype=np.float64)
        if pred.ndim == 3:
            pred = pred[0]
        truth = F.interpolate(
            torch.as_tensor(clip.rps[None], dtype=torch.float32),
            size=pred.shape[-1],
            mode="linear",
            align_corners=False,
        )[0].numpy()
    # PIT assignment on rows of pred, MAE-optimal (4! = 24)
    from itertools import permutations

    perm = min(
        permutations(range(pred.shape[0])),
        key=lambda p: float(np.abs(pred[list(p)] - truth).mean()),
    )
    pred = pred[list(perm)]
    return pred, truth, float(np.abs(pred - truth).mean())


def figure(
    model: str,
    clip: SD.Clip,
    set_label: str,
    held: bool,
    pred: np.ndarray,
    truth: np.ndarray,
    mae: float,
    per_mic: list[float],
    matrix: float,
) -> Path:
    x = clip.audio[0].astype(np.float64)
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(13, 8.5), gridspec_kw=dict(height_ratios=[1.2, 1])
    )
    n, hop = 2048, 512
    w = np.hanning(n + 1)[:n]
    fr = np.stack([x[s : s + n] * w for s in range(0, x.size - n, hop)])
    S = 20 * np.log10(np.abs(np.fft.rfft(fr, axis=-1)).T + 1e-8)
    v = np.percentile(S, 99.5)
    ax0.imshow(
        S,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, x.size / clip.sr, 0, clip.sr / 2000],
        vmin=v - 70,
        vmax=v,
    )
    ax0.set_ylabel("kHz")
    ax0.set_title(
        f"{clip.clip_id} — {set_label} ({'HELD OUT' if held else 'IN TRAINING'}), mic 0\n"
        f"{model} — matrix FLY124 cruise {matrix:.3f} rev/s; this clip mic 0 {mae:.2f}, "
        f"8-mic mean {np.mean(per_mic):.2f} rev/s",
        fontsize=11,
    )
    t_pred = np.arange(pred.shape[1]) / pred.shape[1] * (x.size / clip.sr)
    t_true = np.arange(clip.rps.shape[1]) / clip.sr
    cols = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    for r in range(pred.shape[0]):
        ax1.plot(t_true, clip.rps[r], color=cols[r % 4], lw=2.8, alpha=0.33)
        ax1.plot(t_pred, pred[r], color=cols[r % 4], lw=1.0, marker=".", ms=3)
    ax1.plot([], [], color="k", lw=2.8, alpha=0.33, label="ground truth (telemetry)")
    ax1.plot([], [], color="k", lw=1.0, marker=".", ms=3, label="prediction (PIT-matched)")
    ax1.set_xlabel("s")
    ax1.set_ylabel("rev/s")
    ax1.grid(alpha=0.25)
    ax1.set_xlim(0, x.size / clip.sr)
    ax1.legend(fontsize=9, loc="lower right")
    per_rotor = np.abs(pred - truth).mean(1)
    ax1.set_title(
        "per-rotor MAE "
        + ", ".join(f"{q:.2f}" for q in per_rotor)
        + " rev/s | per-mic MAE "
        + ", ".join(f"{q:.2f}" for q in per_mic),
        fontsize=9,
    )
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{model}__{clip.clip_id}.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metric = LayerPeakRPSMetric()
    results: dict[str, Any] = {}
    for model, info in MODELS.items():
        fm = zoo.load(model, ckpt="best")
        rows = []
        for set_label, held, clips, loader in CLIP_SETS:
            for cid in clips:
                clip = loader(cid)
                pred0, truth0, mae0 = predict(fm, info["kind"], metric, clip, 0)
                per_mic = [mae0]
                for mic in range(1, clip.audio.shape[0]):
                    per_mic.append(predict(fm, info["kind"], metric, clip, mic)[2])
                row = {
                    "clip": clip.clip_id,
                    "set": set_label,
                    "held_out": held,
                    "true_rps_mean": clip.rps.mean(1).round(2).tolist(),
                    "per_mic_mae": [round(q, 4) for q in per_mic],
                    "mic0_mae": round(per_mic[0], 4),
                    "mean_mae": round(float(np.mean(per_mic)), 4),
                    "best_mic": int(np.argmin(per_mic)),
                    "worst_mic": int(np.argmax(per_mic)),
                    "per_rotor_mae_mic0": np.abs(pred0 - truth0).mean(1).round(3).tolist(),
                }
                if cid in FIGURE_CLIPS:
                    row["figure"] = figure(
                        model,
                        clip,
                        set_label,
                        held,
                        pred0,
                        truth0,
                        per_mic[0],
                        per_mic,
                        info["matrix_fly124_cruise"],
                    ).name
                rows.append(row)
                print(
                    f"{model:20s} {clip.clip_id:32s} {set_label:26s} "
                    f"mic0 {per_mic[0]:6.3f}  8-mic {np.mean(per_mic):6.3f}  "
                    f"[{min(per_mic):.2f}–{max(per_mic):.2f}]",
                    flush=True,
                )
        results[model] = {"info": info, "rows": rows}
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
