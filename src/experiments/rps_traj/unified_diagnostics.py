"""Secondary whole-flight diagnostics, independent of the frozen phase ranking."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from experiments.rps_traj.scores import energy_score
from experiments.rps_traj.unified_response import UnifiedResponseModel
from tracking.rotors import MIXER


def _yaw(waves: np.ndarray) -> np.ndarray:
    # Yaw-like diagonal speed contrast, not measured aircraft yaw.
    yaw = np.einsum("r,nrt->nt", MIXER[:, 3] / 4, waves)
    return uniform_filter1d(yaw - yaw.mean(axis=1, keepdims=True), size=3, axis=1, mode="nearest")


def _excursions(yaw: np.ndarray, threshold: float, fs: float) -> np.ndarray:
    """Event rate and excess opposite-sign neighbours/s over an order-shuffled null."""
    out = np.empty((len(yaw), 2))
    for i, row in enumerate(yaw):
        sign = (row > threshold).astype("int8") - (row < -threshold).astype("int8")
        starts = np.flatnonzero((sign != 0) & (sign != np.r_[np.int8(0), sign[:-1]]))
        n = len(starts)
        excess = 0.0
        if n > 1:
            s = sign[starts]
            near = np.diff(starts) <= fs  # successor within one second
            pos = np.count_nonzero(s > 0)
            p_reverse = 2 * pos * (n - pos) / (n * (n - 1))
            excess = np.count_nonzero(near & (s[:-1] != s[1:])) - near.sum() * p_reverse
        out[i] = (n, excess)
    return out * (fs / yaw.shape[1])


def diagnose(out: Path, targets: list[dict]) -> dict:
    """Compare both inference objectives and a sequence readout. No fitting here."""
    from experiments.rps_traj.schedule_eval import PHASES
    from experiments.rps_traj.schedule_score import score_phase
    from experiments.rps_traj.unified_experiment import _metric

    manifest = json.loads((out / "pack/manifest.json").read_text())
    rows = []
    for t in targets:
        rig, fold, name = t["rig"], t["fold"], t["flight"]
        key = f"{rig}__{fold}"
        hover = json.loads((out / "pack" / f"{key}.json").read_text())["train_hover_rps"]
        metric = _metric(manifest, rig)
        with np.load(out / "pack" / f"{key}.eval.npz") as ev:
            observation = ev[f"{name}__x"]
            phases = ev[f"{name}__phase"]
            refs = ev[f"{name}__cruise__refs"]
            truth = ev[f"{name}__cruise__obs"]
        for arm in ("hierarchical", "flat"):
            fit = json.loads((out / "fits" / f"{key}__{arm}.json").read_text())
            model = UnifiedResponseModel.from_dict(fit["params"])
            with np.load(out / "draws" / f"{key}__{arm}.npz") as z:
                cruise = z[f"{name}__cruise__draws"]
                free = z[f"{name}__paths"]
            shaft = {}
            with (
                np.load(out / "pack" / f"{key}.eval.npz") as ev,
                np.load(out / "draws" / f"{key}__{arm}.npz") as z,
            ):
                for phase in PHASES:
                    k = f"{name}__{phase}"
                    shaft[phase] = float(
                        np.mean(
                            [
                                score_phase(phase, obs, donor, np.maximum(g, 0), hover, metric)[
                                    "score"
                                ]
                                for obs, donor, g in zip(
                                    ev[k + "__obs"], ev[k + "__refs"], z[k + "__draws"], strict=True
                                )
                            ]
                        )
                    )
            vals = []
            for obs, donors, gen in zip(truth, refs, cruise, strict=True):
                threshold = max(0.05, float(np.median(_yaw(donors).std(axis=1))))
                stats = _excursions(_yaw(np.concatenate((obs[None], gen))), threshold, 25.0)
                vals.append([energy_score(stats[1:], stats[0]), *stats[0], *stats[1:].mean(axis=0)])
            est = np.mean(vals, axis=0)
            h = np.asarray(fit["history"])
            if not np.isfinite(h).all() or np.any(
                np.diff(h) < -1e-8 * np.maximum(1.0, np.abs(h[:-1]))
            ):
                raise ValueError(f"non-monotone training objective: {key} {arm}")
            # Rates include tiny negative samples at rest; additionally record substantial negatives.
            neg = free.min(axis=1)
            rows.append(
                {
                    **t,
                    "arm": arm,
                    "heldout_mode_nats_per_step": model.loglik(observation, phases)
                    / (len(phases) - 2),
                    "stop": model.fit_info["stop_reason"],
                    "steps": len(h) - 1,
                    "final_gain_per_step": float(h[-1] - h[-2]) / model.fit_info["n_valid"]
                    if len(h) > 1
                    else None,
                    "negative_fraction": float((neg < 0).mean()),
                    "below_minus_one": float((neg < -1.0).mean()),
                    "minimum_rps": float(neg.min()),
                    "maximum_rps": float(free.max()),
                    "nonnegative_readout_scores": shaft,
                    "order_score": float(est[0]),
                    "real_event_rate": float(est[1]),
                    "real_excess_reversal_rate": float(est[2]),
                    "sample_event_rate": float(est[3]),
                    "sample_excess_reversal_rate": float(est[4]),
                }
            )
    balanced = {}
    for arm in ("hierarchical", "flat"):
        arm_rows = [r for r in rows if r["arm"] == arm]
        balanced[arm] = {
            metric: float(
                np.mean(
                    np.asarray(
                        [
                            np.mean([r[metric] for r in arm_rows if r["rig"] == rig])
                            for rig in sorted({r["rig"] for r in arm_rows})
                        ]
                    )
                )
            )
            for metric in (
                "heldout_mode_nats_per_step",
                "order_score",
                "negative_fraction",
                "below_minus_one",
                "real_event_rate",
                "real_excess_reversal_rate",
                "sample_event_rate",
                "sample_excess_reversal_rate",
            )
        }
    return {"per_flight": rows, "rig_balanced": balanced}
