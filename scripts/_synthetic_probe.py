"""Does the best real-trained RPS model also work on SYNTHETIC rotor noise?

The discriminative test the identifiability work implies: ``hppnet_l2_r2_s0``
reaches 0.99 rev/s on Michael's HELD-OUT cruise, so Michael's four rotors carry
enough information to be tracked. If a synthetic stream is close to Michael's
data, the SAME frozen checkpoint should score comparably on it.

The test is only informative if it can FAIL. So the counterexample comes first:
draw clips from the deliberately DIVERSE control policy
(``conf/online_mix/ctrl_diverse.yaml`` — wide parametric arm plus two
measurement-anchored neighbourhood arms with inflated residual), which nobody
claims looks like Michael's rig. If the checkpoint tracks those just as well,
the probe measures "is there a comb at all", not "is this Michael's comb", and
it is worthless as a fidelity test.

Every policy is rendered NOISE-ONLY (``source_prob: 0.0``, silence arm dropped)
because the real reference clips are ``nosource``. Scored with the project's own
readout, all 8 microphones, same as ``scripts/_bestmodel_eval.py``.
Campaign-temporary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import copy

import matplotlib.pyplot as plt
import numpy as np
import tdseries as td
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf

import zoo
from data_processing.online_mixing import build_online_mix_pipeline
from metrics.salience_layers import LayerPeakRPSMetric, peak_readout

OUT = Path("results/synthetic_probe")
MODEL = "hppnet_l2_r2_s0"
POLICIES = {
    "ctrl_diverse": "conf/online_mix/ctrl_diverse.yaml",
    "rig_fitted_5050": "conf/online_mix/rig_fitted_5050.yaml",
}
N_CLIPS = 12
#: Michael's cruise band, for selecting comparable synthetic clips.
CRUISE_LO, CRUISE_HI = 65.0, 100.0


def noise_only(path: str) -> Any:
    """The policy with speech disabled, silence dropped, and reuse switched off.

    ``render_reuse``/``flight_reuse`` exist to amortise rendering during
    training — they hand the same parameter draw and the same flight to dozens
    of consecutive samples. For a probe that is a trap: eight consecutive
    samples would be three distinct draws. Both are forced to 1 so every clip
    is an independent draw from the policy's population.
    """
    cfg = copy.deepcopy(yaml.safe_load(Path(path).read_text()))
    cfg["sources"]["noise"] = [a for a in cfg["sources"]["noise"] if a.get("kind") != "silence"]
    cfg["sources"].pop("speech", None)
    for arm in cfg["sources"]["noise"]:
        arm["render_reuse"] = 1
        if isinstance(arm.get("rps"), dict):
            arm["rps"]["flight_reuse"] = 1
    for stage in cfg.get("policy", {}).get("stages", []):
        stage["source_prob"] = 0.0
    cfg["task"] = "rps_prediction"
    return OmegaConf.create(cfg)


def sample_clips(path: str, n: int) -> list[tuple[np.ndarray, np.ndarray, dict]]:
    """``n`` noise-only samples as ``(audio (M,T), rps (R,T), meta)``."""
    pipe = build_online_mix_pipeline(noise_only(path))
    out = []
    for frame in pipe:
        audio = np.asarray(frame["mixture"].data, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[None]
        rps = np.asarray(frame["rps"].data, dtype=np.float64)
        meta = dict(frame["meta"]) if "meta" in list(frame.keys()) else {}
        out.append(
            (audio, rps, {k: v for k, v in meta.items() if isinstance(v, (int, float, str, bool))})
        )
        if len(out) >= n:
            break
    return out


def score(
    fm: Any, metric: LayerPeakRPSMetric, audio: np.ndarray, rps: np.ndarray, sr: int, mic: int
) -> tuple[np.ndarray, np.ndarray, float]:
    x = audio[mic].astype(np.float32)
    inp = td.Frame(
        {
            "mixture": td.Series(
                x, dims=("time",), indexes={"time": td.GridIndex(sr_num=sr, size=x.size)}
            )
        }
    )
    # The rps label lives on the STFT grid; interpolate onto the model's frames.
    tgt_rps = rps if rps.shape[-1] > 8 else rps
    tgt = td.Frame(
        {
            "rps": td.Series(
                tgt_rps.astype(np.float32),
                dims=("rotor", "time"),
                indexes={"time": td.GridIndex(sr_num=sr, size=tgt_rps.shape[-1])},
            )
        }
    )
    layers, rps_grid = metric._unpack(fm(inp), tgt)
    pred = peak_readout(F.logsigmoid(layers), metric._freqs)[0].numpy()
    truth = rps_grid[0].numpy()
    from itertools import permutations

    perm = min(
        permutations(range(pred.shape[0])),
        key=lambda p: float(np.abs(pred[list(p)] - truth).mean()),
    )
    pred = pred[list(perm)]
    return pred, truth, float(np.abs(pred - truth).mean())


def figure(
    tag: str,
    i: int,
    audio: np.ndarray,
    rps: np.ndarray,
    sr: int,
    pred: np.ndarray,
    truth: np.ndarray,
    per_mic: list[float],
) -> str:
    x = audio[0].astype(np.float64)
    dur = x.size / sr
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
        extent=[0, dur, 0, sr / 2000],
        vmin=v - 70,
        vmax=v,
    )
    ax0.set_ylabel("kHz")
    ax0.set_title(
        f"{tag} — synthetic clip {i}, mic 0 | {MODEL} mic0 MAE "
        f"{per_mic[0]:.2f}, 8-mic {np.mean(per_mic):.2f} rev/s",
        fontsize=11,
    )
    t_pred = np.linspace(0, dur, pred.shape[1])
    t_true = np.linspace(0, dur, rps.shape[-1])
    cols = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    for r in range(pred.shape[0]):
        ax1.plot(t_true, rps[r], color=cols[r % 4], lw=2.8, alpha=0.33)
        ax1.plot(t_pred, pred[r], color=cols[r % 4], lw=1.0, marker=".", ms=3)
    ax1.plot([], [], color="k", lw=2.8, alpha=0.33, label="synthetic ground truth")
    ax1.plot([], [], color="k", lw=1.0, marker=".", ms=3, label="prediction (PIT-matched)")
    ax1.set_xlabel("s")
    ax1.set_ylabel("rev/s")
    ax1.grid(alpha=0.25)
    ax1.set_xlim(0, dur)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.set_title(
        "per-rotor MAE "
        + ", ".join(f"{q:.2f}" for q in np.abs(pred - truth).mean(1))
        + " rev/s | per-mic "
        + ", ".join(f"{q:.2f}" for q in per_mic),
        fontsize=9,
    )
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"{tag}__clip{i:02d}.png"
    fig.savefig(OUT / name, dpi=100)
    plt.close(fig)
    return name


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fm = zoo.load(MODEL, ckpt="best")
    metric = LayerPeakRPSMetric()
    results: dict[str, Any] = {}
    for tag, path in POLICIES.items():
        clips = sample_clips(path, N_CLIPS)
        rows = []
        for i, (audio, rps, meta) in enumerate(clips):
            sr = 16000
            per_mic, pred0, truth0 = [], None, None
            for mic in range(audio.shape[0]):
                pred, truth, mae = score(fm, metric, audio, rps, sr, mic)
                per_mic.append(mae)
                if mic == 0:
                    pred0, truth0 = pred, truth
            assert pred0 is not None and truth0 is not None
            active = rps[rps > 5.0]
            row = {
                "clip": i,
                "policy": tag,
                "mean_rps": round(float(active.mean()) if active.size else 0.0, 2),
                "rps_per_rotor": rps.mean(-1).round(2).tolist(),
                "per_mic_mae": [round(q, 4) for q in per_mic],
                "mic0_mae": round(per_mic[0], 4),
                "mean_mae": round(float(np.mean(per_mic)), 4),
                "per_rotor_mae_mic0": np.abs(pred0 - truth0).mean(1).round(3).tolist(),
                "cruise_band": bool(active.size and CRUISE_LO <= active.mean() <= CRUISE_HI),
                "figure": figure(tag, i, audio, rps, sr, pred0, truth0, per_mic),
                "meta": meta,
            }
            rows.append(row)
            print(
                f"{tag:18s} clip {i:2d}  mean rps {row['mean_rps']:6.1f}  "
                f"mic0 {per_mic[0]:6.2f}  8-mic {np.mean(per_mic):6.2f}  "
                f"[{min(per_mic):.2f}-{max(per_mic):.2f}]"
                f"{'  CRUISE' if row['cruise_band'] else ''}",
                flush=True,
            )
        results[tag] = rows
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
