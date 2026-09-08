"""GPU-resident, multi-subset validation for RPS regressors.

A validation plan concatenates each unique finite dataset exactly once. Named
views are cheap index masks over the resulting per-sample PIT-MAE tensor, so
nested real views (R1/R2/R3, source/no-source) never repeat model inference.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from collections.abc import Callable, Mapping, Sized
from dataclasses import dataclass
from typing import Any, cast

import tdseries as td
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, Dataset

from losses._common import get_tensor
from metrics.rps import batched_pit_mae
from training.config import build_dataset

MULTI_VALIDATION_STATE_KEY = "multi_validation"


@dataclass(frozen=True)
class ResolvedView:
    name: str
    dataset: str
    dataset_offset: int
    start: int
    stop: int
    channels: tuple[int, ...]
    channels_per_clip: int | None


@dataclass(frozen=True)
class ValidationPlan:
    dataset: ConcatDataset
    size: int
    views: tuple[ResolvedView, ...]
    aggregates: dict[str, dict[str, float]]
    primary: tuple[str, ...]


def _plain(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def build_validation_plan(cfg: Any) -> ValidationPlan:
    datasets_cfg = dict(_plain(cfg.datasets) or {})
    if not datasets_cfg:
        raise ValueError("validation.datasets must contain at least one finite dataset")

    datasets: list[Dataset] = []
    offsets: dict[str, tuple[int, int]] = {}
    offset = 0
    for name, spec in datasets_cfg.items():
        dataset = build_dataset(spec)
        if not isinstance(dataset, Dataset):
            raise TypeError(f"validation dataset {name!r} must be finite/map-style")
        size = len(cast(Sized, dataset))
        offsets[str(name)] = (offset, size)
        datasets.append(dataset)
        offset += size

    views: list[ResolvedView] = []
    for name, raw in dict(_plain(cfg.views) or {}).items():
        view = dict(raw)
        dataset_name = str(view["dataset"])
        if dataset_name not in offsets:
            raise ValueError(
                f"validation view {name!r} references unknown dataset {dataset_name!r}"
            )
        dataset_offset, dataset_size = offsets[dataset_name]
        start = int(view.get("start", 0))
        stop = int(view.get("stop", dataset_size))
        if not 0 <= start < stop <= dataset_size:
            raise ValueError(
                f"validation view {name!r} range [{start}, {stop}) is outside "
                f"dataset {dataset_name!r} of length {dataset_size}"
            )
        channels = tuple(int(x) for x in (view.get("channels") or ()))
        channels_per_clip = view.get("channels_per_clip")
        channels_per_clip = int(channels_per_clip) if channels_per_clip is not None else None
        if channels and (channels_per_clip is None or channels_per_clip <= 0):
            raise ValueError(
                f"validation view {name!r} selects channels but has no channels_per_clip"
            )
        if channels_per_clip is not None and any(
            channel < 0 or channel >= channels_per_clip for channel in channels
        ):
            raise ValueError(f"validation view {name!r} has an invalid channel index")
        views.append(
            ResolvedView(
                name=str(name),
                dataset=dataset_name,
                dataset_offset=dataset_offset,
                start=dataset_offset + start,
                stop=dataset_offset + stop,
                channels=channels,
                channels_per_clip=channels_per_clip,
            )
        )

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
    if str(cfg.control) not in known:
        raise ValueError(f"validation.control references unknown score {cfg.control!r}")

    return ValidationPlan(
        dataset=ConcatDataset(datasets),
        size=offset,
        views=tuple(views),
        aggregates=aggregates,
        primary=primary,
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
    # The model's deployed decoder (``SalienceRPSPredictor.decode_logits``):
    # shared-map tracking or per-layer CRF, on the logits' device.
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
) -> tuple[dict[str, float], float]:
    """Run one batched inference pass and reduce all RPS scores on the GPU."""
    model.eval()
    score_sums = torch.zeros(len(plan.views), device=device, dtype=torch.float64)
    score_counts = torch.zeros(len(plan.views), device=device, dtype=torch.int64)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    sample_count = 0
    offset = 0
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
            target = get_tensor(batch, "rps")
            per_sample = batched_pit_mae(pred, target).to(torch.float64)
            batch_size = int(per_sample.shape[0])
            indices = torch.arange(offset, offset + batch_size, device=device)

            for view_idx, view in enumerate(plan.views):
                mask = (indices >= view.start) & (indices < view.stop)
                if view.channels:
                    assert view.channels_per_clip is not None
                    local = indices - view.dataset_offset
                    channel_mask = torch.zeros_like(mask)
                    for channel in view.channels:
                        channel_mask |= local.remainder(view.channels_per_clip) == channel
                    mask &= channel_mask
                score_sums[view_idx] += per_sample.masked_select(mask).sum()
                score_counts[view_idx] += mask.sum()

            loss_sum += loss.detach().to(torch.float64) * batch_size
            sample_count += batch_size
            offset += batch_size

    if offset != plan.size:
        raise RuntimeError(f"validation loader yielded {offset} samples, expected {plan.size}")
    packed = torch.cat((score_sums, score_counts.to(torch.float64), loss_sum[None])).cpu()
    count = len(plan.views)
    sums = packed[:count]
    counts = packed[count : 2 * count]
    scores = {
        view.name: float(sums[i] / counts[i]) for i, view in enumerate(plan.views) if counts[i] > 0
    }
    for name, weights in plan.aggregates.items():
        scores[name] = sum(scores[part] * weight for part, weight in weights.items())
    return scores, float(packed[-1] / max(sample_count, 1))


__all__ = [
    "MULTI_VALIDATION_STATE_KEY",
    "ResolvedView",
    "ValidationPlan",
    "RPSReadout",
    "build_validation_plan",
    "rps_readout_for",
    "validate_rps",
    "MultiValidationVerdict",
    "MultiMetricController",
]


@dataclass(frozen=True)
class MultiValidationVerdict:
    smoothed: dict[str, float]
    improved: tuple[str, ...]
    lr_reduced: bool
    stop_reason: str | None


class MultiMetricController:
    """Any-subset meaningful-progress LR and saturation controller."""

    def __init__(self, cfg: Any, names: tuple[str, ...]) -> None:
        self.window = int(cfg.smoothing_window)
        self.min_log_improvement = math.log1p(float(cfg.min_relative_improvement))
        self.min_rounds = int(cfg.min_rounds)
        self.lr_patience = int(cfg.lr_patience)
        self.lr_factor = float(cfg.lr_factor)
        self.min_lr_reductions = int(cfg.min_lr_reductions)
        self.final_patience = int(cfg.final_patience)
        self.names = names
        self.windows = {name: deque(maxlen=self.window) for name in names}
        self.best_smoothed: dict[str, float] = {}
        self.stale_rounds = 0
        self.lr_reductions = 0
        self.rounds_completed = 0
        self.stop_reason: str | None = None

    def _improved(self, value: float, best: float) -> bool:
        if value < 0.0 or best < 0.0:
            raise ValueError("log-scale validation monitors must be non-negative")
        if best == 0.0:
            return False
        if value == 0.0:
            return True
        return math.log(best) - math.log(value) >= self.min_log_improvement

    def step(
        self,
        scores: Mapping[str, float],
        optimizer: torch.optim.Optimizer,
    ) -> MultiValidationVerdict:
        self.rounds_completed += 1
        values = {name: float(scores[name]) for name in self.names}
        if not all(math.isfinite(value) for value in values.values()):
            self.stop_reason = "nonfinite"
            return MultiValidationVerdict({}, (), False, self.stop_reason)

        for name, value in values.items():
            self.windows[name].append(value)
        smoothed = {name: statistics.median(window) for name, window in self.windows.items()}
        if any(len(window) < self.window for window in self.windows.values()):
            return MultiValidationVerdict(smoothed, (), False, None)

        improved: list[str] = []
        for name, value in smoothed.items():
            best = self.best_smoothed.get(name)
            if best is None or self._improved(value, best):
                self.best_smoothed[name] = value
                improved.append(name)

        reduced = False
        if improved:
            self.stale_rounds = 0
        else:
            self.stale_rounds += 1
            if self.lr_reductions < self.min_lr_reductions:
                if self.stale_rounds >= self.lr_patience:
                    for group in optimizer.param_groups:
                        group["lr"] *= self.lr_factor
                    self.lr_reductions += 1
                    self.stale_rounds = 0
                    reduced = True
            elif (
                self.rounds_completed >= self.min_rounds
                and self.stale_rounds >= self.final_patience
            ):
                self.stop_reason = "saturation"

        return MultiValidationVerdict(
            smoothed=smoothed,
            improved=tuple(improved),
            lr_reduced=reduced,
            stop_reason=self.stop_reason,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "windows": {name: list(window) for name, window in self.windows.items()},
            "best_smoothed": dict(self.best_smoothed),
            "stale_rounds": self.stale_rounds,
            "lr_reductions": self.lr_reductions,
            "stop_reason": self.stop_reason,
            "rounds_completed": self.rounds_completed,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if tuple(state["names"]) != self.names:
            raise ValueError("saved multi-validation monitor names differ from current config")
        self.windows = {
            name: deque((float(value) for value in state["windows"][name]), maxlen=self.window)
            for name in self.names
        }
        self.best_smoothed = {
            str(name): float(value) for name, value in state["best_smoothed"].items()
        }
        self.stale_rounds = int(state["stale_rounds"])
        self.lr_reductions = int(state["lr_reductions"])
        self.rounds_completed = int(state.get("rounds_completed", 0))
        self.stop_reason = state.get("stop_reason")
