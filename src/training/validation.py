"""Full-panel RPS validation: one inference pass, per-frame errors, meta views.

A validation plan concatenates each finite dataset once and reads every
sample's ``frame["meta"]`` (and its target, for the regime check) ONCE at
build time. Named views are predicates over that meta (``where``) optionally
restricted to one frame regime of :mod:`metrics.regimes` (``regime``); the
per-round pass only computes ``(N, T)`` per-frame errors under each clip's
optimal rotor assignment (:func:`metrics.rps.batched_pit_frame_errors`) and
the target's regime labels, then reduces every view on the host:

* score = frame-weighted mean error over the view's frames;
* clip statistics = mean, population std and count of the per-clip means,
  a clip being every item sharing ``(dataset, meta["clip_index"])`` (the 8
  flattened channels of one recording window).

Aggregates are weighted sums of views (e.g. ``real_macro = ½ dregon + ½ md2``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sized
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import tdseries as td
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import ConcatDataset, Dataset

from data_processing.frames import meta_dict
from losses._common import get_tensor
from metrics.regimes import REGIMES, frame_regimes
from metrics.rps import batched_pit_frame_errors
from training.config import build_dataset
from training.stopping import control_scores

__all__ = [
    "CLIP_KEY",
    "ResolvedView",
    "ValidationPlan",
    "ValidationSamples",
    "RPSReadout",
    "build_validation_plan",
    "rps_readout_for",
    "validate_rps",
]

#: Meta key grouping flattened items into clips for the clip-level statistics.
CLIP_KEY = "clip_index"
_VIEW_KEYS = {"where", "regime"}


@dataclass(frozen=True)
class ResolvedView:
    name: str
    samples: np.ndarray  # (N,) bool over the concatenated panel
    regime: int | None  # metrics.regimes code, or None for every frame


@dataclass(frozen=True)
class ValidationPlan:
    dataset: ConcatDataset
    size: int
    views: tuple[ResolvedView, ...]
    aggregates: dict[str, dict[str, float]]
    primary: tuple[str, ...]
    control: tuple[str, ...]
    clip_ids: np.ndarray  # (N,) dense clip id, unique per (dataset, clip_index)
    clip_index: np.ndarray  # (N,) meta clip_index
    channel: np.ndarray  # (N,) meta channel, -1 when absent
    frame_rate: float  # frames/s of every target rps series


@dataclass(frozen=True)
class ValidationSamples:
    """Per-sample output of one validation round (host arrays)."""

    pit_mae: np.ndarray  # (N,) float32, clip score = frame_err.mean(-1)
    frame_err: np.ndarray  # (N, T) float32
    frame_regime: np.ndarray  # (N, T) int8, metrics.regimes codes
    clip_index: np.ndarray  # (N,)
    channel: np.ndarray  # (N,)

    def arrays(self) -> dict[str, np.ndarray]:
        """The per-sample arrays written to ``val_samples/round_<N>.npz``."""
        return {
            "pit_mae": self.pit_mae,
            "frame_err": self.frame_err,
            "frame_regime": self.frame_regime,
            "clip_index": self.clip_index,
            "channel": self.channel,
        }


def _plain(value: Any) -> Any:
    if isinstance(value, DictConfig | ListConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _target_rate(frame: td.Frame) -> float:
    entry = frame["rps"]
    index = entry.tindex if isinstance(entry, td.Series) and entry.has_time else None
    if not isinstance(index, td.GridIndex):
        raise TypeError("validation targets must be rps Series on a uniform time grid")
    return index.sr_num / index.sr_den


def build_validation_plan(cfg: Any) -> ValidationPlan:
    datasets_cfg = dict(_plain(cfg.datasets) or {})
    if not datasets_cfg:
        raise ValueError("validation.datasets must contain at least one finite dataset")
    views_cfg = {str(name): dict(raw) for name, raw in dict(_plain(cfg.views) or {}).items()}
    for name, view in views_cfg.items():
        unknown = set(view) - _VIEW_KEYS
        if unknown:
            raise ValueError(
                f"validation view {name!r} has unknown keys {sorted(unknown)}; views are "
                f"meta predicates {{where: {{key: value | [values]}}, regime: <name>}}"
            )
        regime = view.get("regime")
        if regime is not None and regime not in REGIMES:
            raise ValueError(f"validation view {name!r}: regime {regime!r} not in {REGIMES}")
    meta_keys = {CLIP_KEY} | {key for view in views_cfg.values() for key in view.get("where", {})}

    datasets: list[Dataset] = []
    metas: list[dict[str, Any]] = []
    regimes: list[torch.Tensor] = []
    clip_keys: list[tuple[str, int]] = []
    frame_rate: float | None = None
    for dataset_name, spec in datasets_cfg.items():
        dataset = build_dataset(spec)
        if not isinstance(dataset, Dataset):
            raise TypeError(f"validation dataset {dataset_name!r} must be finite/map-style")
        datasets.append(dataset)
        for index in range(len(cast(Sized, dataset))):
            frame = cast(td.Frame, dataset[index])
            meta = meta_dict(frame)
            missing = meta_keys - set(meta)
            if missing:
                raise ValueError(
                    f"validation dataset {dataset_name!r} sample {index} lacks meta "
                    f"{sorted(missing)}"
                )
            rate = _target_rate(frame)
            if frame_rate is None:
                frame_rate = rate
            elif not math.isclose(rate, frame_rate):
                raise ValueError(
                    f"validation targets mix frame rates ({frame_rate} and {rate} frames/s)"
                )
            metas.append(meta)
            regimes.append(frame_regimes(get_tensor(frame, "rps")[None], rate)[0])
            clip_keys.append((str(dataset_name), int(meta[CLIP_KEY])))
    assert frame_rate is not None
    size = len(metas)

    views: list[ResolvedView] = []
    for name, view in views_cfg.items():
        samples = np.ones(size, dtype=bool)
        for key, wanted in dict(view.get("where") or {}).items():
            allowed = list(wanted) if isinstance(wanted, list | tuple) else [wanted]
            samples &= np.array([meta[key] in allowed for meta in metas], dtype=bool)
        regime = REGIMES.index(view["regime"]) if view.get("regime") is not None else None
        has_frames = samples.any() and (
            regime is None
            or any(bool((regimes[i] == regime).any()) for i in np.flatnonzero(samples))
        )
        if not has_frames:
            raise ValueError(f"validation view {name!r} selects no frames of the panel")
        views.append(ResolvedView(name=name, samples=samples, regime=regime))

    aggregates = {
        str(name): {str(part): float(weight) for part, weight in dict(weights).items()}
        for name, weights in dict(_plain(cfg.aggregates) or {}).items()
    }
    primary = tuple(str(name) for name in (_plain(cfg.primary) or ()))
    known = {view.name for view in views}
    for aggregate, weights in aggregates.items():
        missing = set(weights) - known
        if missing:
            raise ValueError(
                f"validation aggregate {aggregate!r} has unknown inputs {sorted(missing)}"
            )
        if not weights or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6):
            raise ValueError(f"validation aggregate {aggregate!r} weights must sum to one")
        known.add(aggregate)
    missing_primary = set(primary) - known
    if missing_primary:
        raise ValueError(f"validation.primary contains unknown scores {sorted(missing_primary)}")
    control = control_scores(_plain(cfg.control))
    unknown_control = [name for name in control if name not in primary]
    if unknown_control:
        raise ValueError(f"validation.control {unknown_control} must be primary scores")

    dense = {key: i for i, key in enumerate(dict.fromkeys(clip_keys))}
    return ValidationPlan(
        dataset=ConcatDataset(datasets),
        size=size,
        views=tuple(views),
        aggregates=aggregates,
        primary=primary,
        control=control,
        clip_ids=np.array([dense[key] for key in clip_keys], dtype=np.int64),
        clip_index=np.array([key[1] for key in clip_keys], dtype=np.int64),
        channel=np.array([int(meta.get("channel", -1)) for meta in metas], dtype=np.int64),
        frame_rate=float(frame_rate),
    )


RPSReadout = Callable[[torch.nn.Module, td.Frame, dict[str, torch.Tensor]], torch.Tensor]
"""``(model, pred_frame, inputs) -> (B, R, T)`` rev/s on the STFT frame grid."""


def _direct_readout(
    model: torch.nn.Module, pred_frame: td.Frame, inputs: dict[str, torch.Tensor]
) -> torch.Tensor:
    return get_tensor(pred_frame, "rps_pred")


def _salience_readout(
    model: torch.nn.Module, pred_frame: td.Frame, inputs: dict[str, torch.Tensor]
) -> torch.Tensor:
    # The model's deployed decoder (``decode_logits``), on the logits' device.
    return model.decode_logits(  # type: ignore[operator]
        get_tensor(pred_frame, "salience"), int(inputs["mixture"].shape[-1])
    )


_READOUTS: dict[str, RPSReadout] = {
    "rps_prediction": _direct_readout,
    "salience_rps": _salience_readout,
}


def rps_readout_for(task_name: str) -> RPSReadout:
    """The readout that turns a task's prediction Frame into ``(B, R, T)`` rev/s."""
    try:
        return _READOUTS[task_name]
    except KeyError:
        raise ValueError(
            f"multi-validation supports tasks {sorted(_READOUTS)}, got {task_name!r}"
        ) from None


def _reduce_views(
    plan: ValidationPlan, frame_err: np.ndarray, frame_regime: np.ndarray
) -> dict[str, float]:
    scores: dict[str, float] = {}
    errors = frame_err.astype(np.float64)
    for view in plan.views:
        mask = np.broadcast_to(view.samples[:, None], frame_err.shape)
        if view.regime is not None:
            mask = mask & (frame_regime == view.regime)
        frames = mask.sum(axis=1)
        if not frames.any():
            raise RuntimeError(f"validation view {view.name!r} selected no frames this round")
        scores[view.name] = float(np.where(mask, errors, 0.0).sum() / frames.sum())
    for name, weights in plan.aggregates.items():
        scores[name] = sum(scores[part] * weight for part, weight in weights.items())
    return scores


def validate_rps(
    *,
    model: torch.nn.Module,
    codec: Any,
    loss_fn: Any,
    valid_loader: Any,
    plan: ValidationPlan,
    device: torch.device,
    amp: bool,
    amp_dtype: torch.dtype | None,
    readout: RPSReadout = _direct_readout,
) -> tuple[dict[str, float], float, ValidationSamples]:
    """One batched inference pass; returns ``(scores, mean loss, per-sample arrays)``.

    Everything stays on ``device`` until one host transfer at the end.
    """
    model.eval()
    errors: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    sample_count = 0
    kwargs: dict[str, Any] = (
        {"dtype": amp_dtype} if amp_dtype is not None and device.type == "cuda" else {}
    )

    with torch.inference_mode():
        for batch in valid_loader:
            batch = batch.map_data(lambda value: value.to(device, non_blocking=True))
            inputs = codec.to_inputs(batch)
            with torch.autocast(device_type=device.type, enabled=amp, **kwargs):
                outputs = codec.call_model(model, inputs)
                pred_frame = codec.to_frame(outputs, batch)
                loss = loss_fn(pred_frame, batch)
            pred = readout(model, pred_frame, inputs)
            target = get_tensor(batch, "rps").float()
            errors.append(batched_pit_frame_errors(pred, target))
            n_pred, n_target = int(pred.shape[-1]), int(target.shape[-1])
            if n_target != n_pred:
                target = F.interpolate(target, size=n_pred, mode="linear", align_corners=False)
            labels.append(frame_regimes(target, plan.frame_rate * n_pred / n_target))
            batch_size = int(target.shape[0])
            loss_sum += loss.detach().to(torch.float64) * batch_size
            sample_count += batch_size

    if sample_count != plan.size:
        raise RuntimeError(
            f"validation loader yielded {sample_count} samples, expected {plan.size}"
        )
    if len({int(error.shape[-1]) for error in errors}) != 1:
        raise RuntimeError("validation predictions must share one frame count across the panel")
    frame_err = torch.cat(errors).cpu().numpy().astype(np.float32)
    frame_regime = torch.cat(labels).cpu().numpy()
    loss_value = float(loss_sum.cpu()) / max(sample_count, 1)
    scores = _reduce_views(plan, frame_err, frame_regime)
    samples = ValidationSamples(
        pit_mae=frame_err.mean(axis=1, dtype=np.float64).astype(np.float32),
        frame_err=frame_err,
        frame_regime=frame_regime,
        clip_index=plan.clip_index,
        channel=plan.channel,
    )
    return scores, loss_value, samples
