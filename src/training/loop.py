"""One generic training loop (docs/refactor-unified-framework.md § "train.py
/ eval.py"; schedule, panel and checkpoints: docs/experiments/paper-real-protocol.md).

Covers map-style *or* iterable (online-mixing) datasets, AMP autocast +
``GradScaler``, gradient accumulation, grad-norm clipping, an optimizer
factory, a step-based LR schedule (linear warm-up then cosine, stepped once per
OPTIMIZER step, a pure function of the step count — validation never touches
it), atomic checkpointing and wandb logging (run name = experiment name, git
commit hash, dirty-tree guard, run-id file for the job runner).

Two validation paths:

* ``validation.enabled`` (RPS tasks): the full panel of
  :mod:`training.validation` every ``every_optimizer_steps`` optimizer steps,
  up to ``max_optimizer_steps``; :class:`~training.stopping.MultiMetricController`
  picks ``best_<score>_median.ckpt`` / ``best_<score>_raw.ckpt`` per primary
  score and stops on the control score's saturation. Every round appends to
  ``validation_history.jsonl`` and writes ``val_samples/round_<N>.npz``.
* otherwise the per-sample :class:`~metrics.suite.MetricSuite` over
  ``data.valid`` once per epoch, logged only (no selection, no early stop).

``last.ckpt`` is written every round; a non-finite round stops the run before
anything for it is written.

Everything Frame-shaped funnels through the task's :class:`~tasks.codecs.Codec`
(``to_inputs`` / ``call_model`` / ``to_frame``) and
``data_processing.collate`` (``frame_collate`` for batching,
``slice_sample`` for the per-sample view :class:`~metrics.suite.MetricSuite`
needs) — the loop itself never inspects task-specific tensor shapes.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import time
from collections.abc import Iterable, Iterator
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import tdseries as td
import torch
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm

import wandb
from data_processing.collate import batch_size as frame_batch_size
from data_processing.collate import frame_collate, slice_sample
from tasks.codecs import Codec
from tasks.task import Task
from training.artifacts import ArtifactStore, resolve_checkpoint_uri
from training.config import (
    build_dataset,
    build_losses,
    build_metrics,
    build_task_and_codec,
    instantiate_model,
)
from training.lora import maybe_apply_lora
from training.stopping import STATE_KEY, MultiMetricController, MultiValidationVerdict
from training.validation import (
    ValidationSamples,
    build_validation_plan,
    rps_readout_for,
    validate_rps,
)

__all__ = [
    "run_training",
    "get_optimizer",
    "warmup_cosine_factor",
    "build_lr_scheduler",
    "git_commit_hash",
    "is_git_dirty",
]

logger = logging.getLogger(__name__)


# ─── Optimizer factory + LR schedule ──────────────────────────────────────────


def get_optimizer(
    model: torch.nn.Module,
    *,
    name: str,
    lr: float,
    weight_decay: float = 0.0,
    extra_params: dict[str, Any] | None = None,
) -> torch.optim.Optimizer:
    """Build an optimizer by name — adam/adamw/radam/rmsprop/sgd/prodigy/adamw8bit.

    ``weight_decay`` is a named kwarg so every optimizer gets it uniformly;
    ``extra_params`` are the optimizer-specific passthrough kwargs.
    """
    params = dict(extra_params or {})
    params.setdefault("weight_decay", weight_decay)
    trainable = model.parameters()
    if name == "adam":
        return torch.optim.Adam(trainable, lr=lr, **params)
    if name == "adamw":
        return torch.optim.AdamW(trainable, lr=lr, **params)
    if name == "radam":
        return torch.optim.RAdam(trainable, lr=lr, **params)
    if name == "rmsprop":
        return torch.optim.RMSprop(trainable, lr=lr, **params)
    if name == "sgd":
        return torch.optim.SGD(trainable, lr=lr, **params)
    if name == "prodigy":
        from prodigyopt import Prodigy

        return Prodigy(trainable, lr=lr, **params)
    if name == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(trainable, lr=lr, **params)
    raise ValueError(f"unknown optimizer {name!r}")


def warmup_cosine_factor(
    step: int, *, warmup_steps: int, total_steps: int, min_lr_frac: float
) -> float:
    """LR multiplier after ``step`` completed optimizer steps.

    Linear 0 -> 1 over ``warmup_steps``, then cosine 1 -> ``min_lr_frac`` at
    ``total_steps``, constant after.
    """
    if step < warmup_steps:
        return step / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_lr_frac + (1.0 - min_lr_frac) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_lr_scheduler(optimizer: torch.optim.Optimizer, optim_cfg: Any) -> LambdaLR:
    """The ``optim`` group's warm-up + cosine schedule; step it once per optimizer step."""
    warmup_steps = int(optim_cfg.warmup_steps)
    total_steps = int(optim_cfg.total_steps)
    min_lr_frac = float(optim_cfg.min_lr_frac)
    if not 0 <= warmup_steps < total_steps:
        raise ValueError(
            f"optim needs 0 <= warmup_steps < total_steps, got {warmup_steps}, {total_steps}"
        )
    if not 0.0 <= min_lr_frac <= 1.0:
        raise ValueError(f"optim.min_lr_frac must be in [0, 1], got {min_lr_frac}")
    factor = partial(
        warmup_cosine_factor,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_frac=min_lr_frac,
    )
    return LambdaLR(optimizer, factor)


# ─── Git state (dirty-tree guard + commit hash for wandb) ────────────────────


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def is_git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
        return bool(out.strip())
    except Exception:
        return False


# ─── Small helpers ─────────────────────────────────────────────────────────────


def _to_device(frame: td.Frame, device: torch.device) -> td.Frame:
    return frame.map_data(lambda tensor: tensor.to(device, non_blocking=True))


def _make_loader(
    dataset: Any,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    persistent_workers: bool = False,
) -> DataLoader:
    iterable = isinstance(dataset, IterableDataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and not iterable),
        num_workers=num_workers,
        collate_fn=frame_collate,
        persistent_workers=(num_workers > 0 and (iterable or persistent_workers)),
        pin_memory=torch.cuda.is_available(),
    )


def _take(it: Iterator[td.Frame], n: int) -> Iterable[td.Frame]:
    """``next(it)`` ``n`` times — a plain function (not inline ``next()`` in a
    generator expression) so a preceding ``assert it is not None`` narrows
    the type at the call site."""
    for _ in range(n):
        yield next(it)


# ─── Core epoch steps ──────────────────────────────────────────────────────────


def _forward(
    codec: Codec,
    model: torch.nn.Module,
    batch: td.Frame,
    *,
    device: torch.device,
    amp: bool,
    amp_dtype: torch.dtype | None = None,
) -> td.Frame:
    inputs = codec.to_inputs(batch)
    # dtype is only forwarded on cuda: cpu autocast supports bfloat16 only, so
    # the cpu path keeps torch's per-device default.
    kwargs: dict[str, Any] = (
        {"dtype": amp_dtype} if (amp_dtype is not None and device.type == "cuda") else {}
    )
    with torch.autocast(device_type=device.type, enabled=amp, **kwargs):
        outputs = codec.call_model(model, inputs)
    return codec.to_frame(outputs, batch)


def _train_one_epoch(
    *,
    model: torch.nn.Module,
    codec: Codec,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: GradScaler,
    batches: Iterable[td.Frame],
    n_batches: int | None,
    device: torch.device,
    amp: bool,
    amp_dtype: torch.dtype | None = None,
    grad_clip: float | None,
    grad_accum_steps: int,
    epoch: int,
) -> tuple[float, int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    count = 0
    optimizer_steps = 0
    pbar = tqdm(batches, total=n_batches, desc=f"train e{epoch}", leave=False)
    for i, batch in enumerate(pbar):
        batch = _to_device(batch, device)
        pred_frame = _forward(codec, model, batch, device=device, amp=amp, amp_dtype=amp_dtype)
        loss = loss_fn(pred_frame, batch)
        # Disabled scalers (CPU/bf16) need an explicit guard. Under fp16,
        # GradScaler already rejects non-finite gradients at the optimizer
        # boundary; avoiding a Python branch on a CUDA scalar removes one
        # device synchronization from every ordinary training batch.
        if (not scaler.is_enabled() or grad_accum_steps > 1) and not torch.isfinite(loss.detach()):
            logger.warning("epoch %d step %d: non-finite loss %s — batch dropped", epoch, i, loss)
            optimizer.zero_grad(set_to_none=True)
            continue
        accepted = True
        scaler.scale(loss / grad_accum_steps).backward()

        is_last = n_batches is not None and i == n_batches - 1
        if (i + 1) % grad_accum_steps == 0 or is_last:
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            accepted = not scaler.is_enabled() or scaler.get_scale() >= scale_before
            if accepted:
                # Only steps the optimizer actually took advance the schedule,
                # so scheduler.last_epoch == completed optimizer steps.
                scheduler.step()
                optimizer_steps += 1
            optimizer.zero_grad(set_to_none=True)

        if accepted:
            total_loss += loss.detach().to(torch.float64)
            count += 1
    return float(total_loss.item() / max(count, 1)), optimizer_steps


def _validate(
    *,
    model: torch.nn.Module,
    codec: Codec,
    loss_fn: torch.nn.Module,
    valid_loader: DataLoader,
    metric_suite: Any,
    device: torch.device,
    amp: bool,
    amp_dtype: torch.dtype | None = None,
) -> tuple[dict[str, float], float]:
    """Stream legacy per-sample metrics without retaining validation Frames."""
    model.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    sample_count = 0
    with torch.inference_mode():
        for batch in valid_loader:
            batch = _to_device(batch, device)
            pred_frame = _forward(
                codec,
                model,
                batch,
                device=device,
                amp=amp,
                amp_dtype=amp_dtype,
            )
            loss = loss_fn(pred_frame, batch)
            size = frame_batch_size(batch)
            loss_sum += loss.detach().to(torch.float64) * size
            sample_count += size
            pred_cpu = pred_frame.map_data(lambda tensor: tensor.detach().cpu())
            batch_cpu = batch.map_data(lambda tensor: tensor.detach().cpu())
            pairs = ((slice_sample(pred_cpu, i), slice_sample(batch_cpu, i)) for i in range(size))
            for row in metric_suite.evaluate(pairs).rows:
                for name in metric_suite.metrics:
                    value = float(row[name])
                    if math.isfinite(value):
                        totals[name] = totals.get(name, 0.0) + value
                        counts[name] = counts.get(name, 0) + 1
    metrics = {name: totals[name] / counts[name] for name in totals}
    return metrics, float(loss_sum.item() / max(sample_count, 1))


# ─── Checkpointing ──────────────────────────────────────────────────────────────


def _save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(model.state_dict(), tmp)
    tmp.replace(path)


def _replace_link_or_copy(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def _save_validation_samples(path: Path, samples: ValidationSamples) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    arrays: dict[str, Any] = samples.arrays()
    with tmp.open("wb") as handle:  # a file handle: savez must not append ".npz"
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


# ``train_state.pt`` is deliberately SEPARATE from ``last.ckpt`` rather than a
# richer dict inside it: every existing consumer (eval.py, _warm_start,
# scripts/se_eval.py, the R2 artifact store) reads the .ckpt files as
# bare state_dicts, so widening them would ripple. This file holds only the
# bookkeeping needed to continue an interrupted run.
TRAIN_STATE_NAME = "train_state.pt"


def _save_train_state(
    path: Path,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: GradScaler,
    epoch: int,
    optimizer_steps: int,
    controller: MultiMetricController | None = None,
) -> None:
    """Persist everything needed to continue training after ``epoch``."""
    tmp = path.with_suffix(".pt.tmp")
    state: dict[str, Any] = {
        "optimizer": optimizer.state_dict(),
        "optimizer_steps": optimizer_steps,
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "next_epoch": epoch + 1,
    }
    if controller is not None:
        state[STATE_KEY] = controller.state_dict()
    torch.save(state, tmp)
    tmp.replace(path)  # atomic: a job killed mid-save leaves the old state intact


def _fetch_train_state(
    store: ArtifactStore, state_path: Path, weights_path: Path
) -> tuple[Path | None, Path | None]:
    """Pull ``train_state.pt`` and ``last.ckpt`` from R2, INDEPENDENTLY.

    Each element is ``None`` when that file is not in the store. They are
    resolved separately because "no weights at all" (a genuine first launch)
    and "weights but no bookkeeping" (a partially uploaded run) need opposite
    responses, and a combined result cannot tell the caller which it has.

    Never raises: an unreachable store must degrade to "nothing found" rather
    than kill the run, the same defensive contract the upload side has.
    """
    root = f"r2://{store.bucket}/{store.prefix}/{store.experiment_name}/checkpoints"

    def one(name: str) -> Path | None:
        try:
            local = Path(resolve_checkpoint_uri(f"{root}/{name}"))
        except Exception:
            return None
        return local if local.exists() else None

    return one(state_path.name), one(weights_path.name)


def _load_train_state(
    run_dir: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: GradScaler,
    device: torch.device,
    store: ArtifactStore | None = None,
    controller: MultiMetricController | None = None,
) -> tuple[int, int]:
    """Restore an interrupted run in place; return ``(start_epoch, optimizer_steps)``.

    Returns ``(0, 0)`` when there is nothing to resume from, so a first
    launch with ``resume=true`` (the safe default for preemptible queues) just
    starts normally.
    """
    state_path = run_dir / TRAIN_STATE_NAME
    weights_path = run_dir / "last.ckpt"
    state_local = state_path if state_path.exists() else None
    weights_local = weights_path if weights_path.exists() else None
    if (state_local is None or weights_local is None) and store is not None:
        # THE RUN DIR IS GONE, WHICH IS THE NORMAL CASE ON A PREEMPTIBLE BOX.
        # Kaggle, Colab and a reclaimed vast instance all hand back an empty
        # filesystem, so "resume" there means "resume from R2 or not at all".
        # Measured: `salv2_hf0_comb_mix` was killed at the 12 h session cap on
        # epoch 101 of ~105 with only the checkpoints uploaded -- the
        # optimizer, the LR schedule, the epoch counter and the stopping
        # counters were all on the dead VM.
        remote_state, remote_weights = _fetch_train_state(store, state_path, weights_path)
        state_local = state_local or remote_state
        weights_local = weights_local or remote_weights
    if state_local is None and weights_local is None:
        return 0, 0  # a genuine first launch: nothing to resume from
    if state_local is None or weights_local is None:
        # WEIGHTS WITHOUT BOOKKEEPING (or the reverse) MUST NOT START FRESH.
        # A fresh controller treats its first round as a new best, so the
        # next round OVERWRITES every best checkpoint with a worse model and
        # resets the stopping counters -- silently. Measured on
        # `m3mixv2_scv2`: its best sat at epoch 8 (val/mse 48.09) and the best
        # checkpoint left on R2 scored 85.4, its epoch-77 value. Refusing is
        # strictly better than destroying a run's best weights.
        missing = TRAIN_STATE_NAME if state_local is None else weights_path.name
        present = weights_path.name if state_local is None else TRAIN_STATE_NAME
        raise RuntimeError(
            f"cannot resume {run_dir.name}: found {present} but no {missing}. "
            f"Starting fresh here would overwrite the best checkpoints with a worse "
            f"model and reset the stopping counters. Either restore {missing}, or run "
            f"under a new experiment_name, or set `resume=false` to start deliberately."
        )
    state = torch.load(state_local, map_location=device, weights_only=False)
    if "best_metric" in state or (controller is not None and STATE_KEY not in state):
        raise RuntimeError(
            f"cannot resume {run_dir.name}: its train_state.pt was written by the retired "
            f"plateau-LR protocol or without the validation controller; start a new experiment"
        )
    model.load_state_dict(torch.load(weights_local, map_location=device, weights_only=False))
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    if controller is not None:
        controller.load_state_dict(state[STATE_KEY])
    start_epoch = int(state["next_epoch"])
    optimizer_steps = int(state["optimizer_steps"])
    logger.info(
        "resuming %s at round %d, optimizer step %d (lr %.3g)",
        run_dir.name,
        start_epoch,
        optimizer_steps,
        optimizer.param_groups[0]["lr"],
    )
    return start_epoch, optimizer_steps


def _checkpoint_slug(name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in name).strip("_")


def _warm_start(
    model: torch.nn.Module,
    checkpoint: str,
    device: torch.device,
    *,
    artifacts_cfg: Any,
) -> None:
    """Initialise a curriculum stage from explicit weights or a score's best.

    ``best:<score>@<experiment>`` resolves that run's median-selected R2
    checkpoint ``best_<score>_median.ckpt``, e.g. ``best:real@rp_hrspnet_all``.
    A curriculum always starts fresh optimizer/scheduler state; same-run
    continuation remains ``resume``.
    """
    if checkpoint.startswith("best:"):
        try:
            score, experiment = checkpoint.removeprefix("best:").rsplit("@", 1)
        except ValueError as error:
            raise ValueError("best-checkpoint syntax is best:<score>@<experiment>") from error
        checkpoint = (
            f"r2://{artifacts_cfg.bucket}/{artifacts_cfg.prefix}/{experiment}/"
            f"checkpoints/best_{_checkpoint_slug(score)}_median.ckpt"
        )
    path = resolve_checkpoint_uri(checkpoint)
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    result = model.load_state_dict(state, strict=False)
    missing, unexpected = result.missing_keys, result.unexpected_keys
    print(
        f"[warm-start] loaded weights from {checkpoint} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )


def _upload_checkpoint_and_record(
    *, store: ArtifactStore, upload_enabled: bool, path: Path, run: Any, summary_key: str
) -> None:
    """Upload ``path`` to R2 and, if it succeeded, write its ``r2://...`` URI
    into the wandb run summary under ``summary_key``. No-op when
    ``upload_enabled`` is false; never raises (``ArtifactStore`` already
    swallows its own upload failures)."""
    if not upload_enabled:
        return
    uri = store.upload_checkpoint(path)
    if uri is not None and run is not None:
        run.summary[summary_key] = uri


def _link_best_checkpoints(
    *,
    verdict: MultiValidationVerdict,
    scores: dict[str, float],
    run_dir: Path,
    last_ckpt_path: Path,
    checkpoint_index: dict[str, Any],
    epoch: int,
    optimizer_steps: int,
    store: ArtifactStore,
    upload: bool,
    run: Any,
) -> bool:
    """Hard-link ``best_<score>_{median,raw}.ckpt`` to ``last.ckpt`` for every
    improved score and record them in ``checkpoint_index``; True if any did."""
    improved = False
    for kind, names in (("median", verdict.improved_median), ("raw", verdict.improved_raw)):
        for score in names:
            slug = _checkpoint_slug(score)
            best_path = run_dir / f"best_{slug}_{kind}.ckpt"
            _replace_link_or_copy(last_ckpt_path, best_path)
            # Server-side copy of the just-uploaded last.ckpt: no re-upload.
            uri = store.alias_checkpoint(last_ckpt_path.name, best_path.name) if upload else None
            if uri is not None and run is not None:
                run.summary[f"r2/best_{slug}_{kind}"] = uri
            checkpoint_index[f"{score}/{kind}"] = {
                "checkpoint": best_path.name,
                "validation_round": epoch,
                "optimizer_step": optimizer_steps,
                "raw": scores[score],
                "median": verdict.smoothed[score],
            }
            improved = True
    return improved


# ─── Main entry point ──────────────────────────────────────────────────────────


def run_training(cfg: Any, *, artifact_store: ArtifactStore | None = None) -> dict[str, float]:
    """Train per ``cfg`` (a composed ``training.config.RootConfig``).

    Sets up the run dir + wandb, builds every component, then runs validation
    rounds until the step budget, the round cap (``epochs``) or the
    controller's saturation stop. Returns the final round, the optimizer-step
    count and (full-panel runs) the control score's best median and raw values.

    ``artifact_store`` is dependency-injected for tests (an
    :class:`~training.artifacts.ArtifactStore` built around a fake
    filesystem); when omitted, a real store is built from ``cfg.artifacts``
    (no-op unless R2 credentials are present — see ``training.artifacts``).
    """
    if is_git_dirty() and not cfg.allow_dirty:
        raise RuntimeError(
            "git working tree is dirty; commit your changes or pass allow_dirty=true"
        )
    commit = git_commit_hash()

    run_dir = Path(cfg.results_root) / cfg.experiment_name
    if run_dir.exists() and not cfg.resume and any(run_dir.iterdir()):
        raise FileExistsError(
            f"{run_dir} already exists and is non-empty; pass resume=true to continue into it"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    task: Task
    task, codec = build_task_and_codec(cfg.model)
    model = instantiate_model(cfg.model)
    model = maybe_apply_lora(model, cfg.lora)
    model = model.to(device)
    store = (
        artifact_store
        if artifact_store is not None
        else ArtifactStore(
            experiment_name=cfg.experiment_name,
            bucket=cfg.artifacts.bucket,
            prefix=cfg.artifacts.prefix,
            enabled=cfg.artifacts.enabled,
        )
    )
    upload = bool(cfg.artifacts.upload_checkpoints)
    if getattr(cfg, "checkpoint", None):
        _warm_start(model, str(cfg.checkpoint), device, artifacts_cfg=cfg.artifacts)
    loss_fn = build_losses(cfg.loss).to(device)
    metric_suite = build_metrics(cfg.metrics)
    validation_cfg = getattr(cfg, "validation", None)
    multi_cfg: Any = (
        validation_cfg
        if validation_cfg is not None and getattr(validation_cfg, "enabled", False)
        else None
    )
    rps_readout = rps_readout_for(task.name) if multi_cfg is not None else None

    optimizer = get_optimizer(
        model,
        name=cfg.optim.optimizer,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        extra_params=dict(cfg.optim.optimizer_params or {}),
    )
    scheduler = build_lr_scheduler(optimizer, cfg.optim)

    train_ds = build_dataset(cfg.data.train)
    batch_size = cfg.data.batch_size or cfg.batch_size
    num_workers = cfg.data.num_workers if cfg.data.num_workers is not None else cfg.num_workers
    train_loader = _make_loader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    validation_plan = build_validation_plan(multi_cfg) if multi_cfg is not None else None
    valid_ds = (
        validation_plan.dataset if validation_plan is not None else build_dataset(cfg.data.valid)
    )
    valid_batch_size = (
        int(multi_cfg.batch_size)
        if multi_cfg is not None and multi_cfg.batch_size is not None
        else int(getattr(cfg.data, "valid_batch_size", None) or batch_size)
    )
    valid_num_workers = (
        int(multi_cfg.num_workers)
        if multi_cfg is not None and multi_cfg.num_workers is not None
        else num_workers
    )
    valid_loader = _make_loader(
        valid_ds,
        batch_size=valid_batch_size,
        num_workers=valid_num_workers,
        shuffle=False,
        persistent_workers=True,
    )
    controller = (
        MultiMetricController(multi_cfg, validation_plan.primary)
        if validation_plan is not None
        else None
    )

    train_iterable = isinstance(train_ds, IterableDataset)
    train_iter: Iterator[td.Frame] | None = iter(train_loader) if train_iterable else None
    batches_per_epoch: int | None = None
    if train_iterable:
        if multi_cfg is not None:
            batches_per_epoch = int(multi_cfg.every_optimizer_steps) * max(
                1, int(cfg.grad_accum_steps)
            )
        elif cfg.samples_per_validation is not None:
            batches_per_epoch = math.ceil(cfg.samples_per_validation / batch_size)
        else:
            raise ValueError(
                "an iterable training dataset needs validation.every_optimizer_steps "
                "or samples_per_validation"
            )

    amp_dtype = (
        torch.bfloat16 if getattr(cfg, "amp_dtype", "float16") == "bfloat16" else torch.float16
    )
    # bfloat16 keeps fp32's exponent range — loss scaling is unnecessary, and a
    # disabled GradScaler makes scale/unscale/step exact no-op passthroughs.
    scaler = GradScaler(
        device.type,
        enabled=(cfg.amp and device.type == "cuda" and amp_dtype is torch.float16),
    )

    wandb_mode = (
        cfg.logging.mode if cfg.logging.mode else (None if cfg.logging.enabled else "disabled")
    )
    # Explicit identity lets a continuation keep its original W&B history while
    # writing checkpoints under a separate experiment name. A missing explicit
    # ID must fail, not silently create another run.
    run_id_file = run_dir / "wandb_run_id.txt"
    prior_run_id = run_id_file.read_text().strip() if cfg.resume and run_id_file.exists() else ""
    prior_run_id = cfg.logging.resume_id or prior_run_id
    run = wandb.init(
        entity=cfg.logging.entity,
        project=cfg.logging.project,
        name=cfg.logging.name or cfg.experiment_name,
        mode=wandb_mode,
        tags=[task.name, *list(cfg.logging.tags or [])],
        dir=str(run_dir),
        config=(
            {"continuation/git_commit": commit, "continuation/experiment_name": cfg.experiment_name}
            if cfg.logging.resume_id
            else {"git_commit": commit, "experiment_name": cfg.experiment_name}
        ),
        id=prior_run_id or None,
        resume="must" if cfg.logging.resume_id else ("allow" if prior_run_id else None),
    )
    if run is not None and getattr(run, "id", None):
        (run_dir / "wandb_run_id.txt").write_text(run.id)

    start_epoch, optimizer_steps = (
        _load_train_state(
            run_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            store=store if upload else None,
            controller=controller,
        )
        if cfg.resume
        else (0, 0)
    )

    def summary(final_epoch: int) -> dict[str, float]:
        result = {"final_epoch": float(final_epoch), "optimizer_steps": float(optimizer_steps)}
        if controller is not None:
            control = controller.control
            if control in controller.best_smoothed:
                result[f"best_{control}_median"] = controller.best_smoothed[control]
            if control in controller.best_raw:
                result[f"best_{control}_raw"] = controller.best_raw[control]
        return result

    epoch = start_epoch
    max_optimizer_steps = int(multi_cfg.max_optimizer_steps) if multi_cfg is not None else None
    done_reason: str | None = None
    if max_optimizer_steps is not None and optimizer_steps >= max_optimizer_steps:
        done_reason = "censored at optimizer-step budget"
    elif controller is not None and controller.stop_reason:
        done_reason = controller.stop_reason
    elif start_epoch >= cfg.epochs:
        done_reason = "round budget"
    if done_reason:
        logger.info(
            "%s not training at validation round %d (%s)",
            cfg.experiment_name,
            start_epoch,
            done_reason,
        )
        wandb.finish()
        return summary(start_epoch)

    history_path = run_dir / "validation_history.jsonl"
    checkpoint_index_path = run_dir / "best_checkpoints.json"
    checkpoint_index = (
        json.loads(checkpoint_index_path.read_text()) if checkpoint_index_path.exists() else {}
    )
    grad_accum = max(1, int(cfg.grad_accum_steps))
    stop_reason: str | None = None
    for epoch in range(start_epoch, cfg.epochs):
        target_updates: int | None = None
        if train_iterable:
            assert train_iter is not None and batches_per_epoch is not None
            if multi_cfg is not None:
                target_updates = batches_per_epoch // grad_accum
                if max_optimizer_steps is not None:
                    target_updates = min(target_updates, max_optimizer_steps - optimizer_steps)
                n_batches = target_updates * grad_accum
            else:
                n_batches = batches_per_epoch
            batches = _take(train_iter, n_batches)
        else:
            batches = train_loader
            n_batches = len(train_loader)

        train_once = partial(
            _train_one_epoch,
            model=model,
            codec=codec,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            amp=cfg.amp,
            amp_dtype=amp_dtype,
            grad_clip=cfg.grad_clip,
            grad_accum_steps=grad_accum,
            epoch=epoch,
        )

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_started_unix = time.time()
        train_started = time.perf_counter()
        train_loss, completed_steps = train_once(batches=batches, n_batches=n_batches)
        if target_updates is not None:
            assert train_iter is not None  # step-cadence rounds only exist for streams
            weighted_loss = train_loss * completed_steps
            while completed_steps < target_updates:
                missing = target_updates - completed_steps
                extra_loss, extra_steps = train_once(
                    batches=_take(train_iter, missing * grad_accum),
                    n_batches=missing * grad_accum,
                )
                if extra_steps == 0:
                    raise RuntimeError(
                        "GradScaler skipped every retry batch; cannot reach validation cadence"
                    )
                weighted_loss += extra_loss * extra_steps
                completed_steps += extra_steps
            train_loss = weighted_loss / completed_steps
        train_finished_unix = time.time()
        train_seconds = time.perf_counter() - train_started
        optimizer_steps += completed_steps

        validation_started_unix = time.time()
        validation_started = time.perf_counter()
        samples: ValidationSamples | None = None
        if validation_plan is not None:
            assert rps_readout is not None
            val_metrics, val_loss, samples = validate_rps(
                model=model,
                codec=codec,
                loss_fn=loss_fn,
                valid_loader=valid_loader,
                plan=validation_plan,
                device=device,
                amp=cfg.amp,
                amp_dtype=amp_dtype,
                readout=rps_readout,
            )
        else:
            val_metrics, val_loss = _validate(
                model=model,
                codec=codec,
                loss_fn=loss_fn,
                valid_loader=valid_loader,
                metric_suite=metric_suite,
                device=device,
                amp=cfg.amp,
                amp_dtype=amp_dtype,
            )
        validation_finished_unix = time.time()
        validation_seconds = time.perf_counter() - validation_started
        peak_allocated_gb = (
            torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
        )
        peak_reserved_gb = (
            torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.0
        )

        round_values = {
            "train/loss": train_loss,
            "val/loss": val_loss,
            **{f"val/{key}": value for key, value in val_metrics.items()},
        }
        nonfinite = sorted(name for name, value in round_values.items() if not math.isfinite(value))
        if nonfinite:
            failure_payload = {
                "epoch": epoch,
                "validation_round": epoch,
                "optimizer_step": optimizer_steps,
                **round_values,
            }
            logger.error(
                "non-finite round at optimizer step %d (%s); stopping before any checkpoint",
                optimizer_steps,
                nonfinite,
            )
            with history_path.open("a") as history:
                history.write(
                    json.dumps({**failure_payload, "validation/nonfinite": nonfinite}) + "\n"
                )
            wandb.log(failure_payload)
            stop_reason = "nonfinite"
            break

        log_payload: dict[str, Any] = {
            "epoch": epoch,
            "validation_round": epoch,
            "optimizer_step": optimizer_steps,
            "lr": optimizer.param_groups[0]["lr"],
            "perf/train_s": train_seconds,
            "perf/validation_s": validation_seconds,
            "perf/train_started_unix": train_started_unix,
            "perf/train_finished_unix": train_finished_unix,
            "perf/validation_started_unix": validation_started_unix,
            "perf/validation_finished_unix": validation_finished_unix,
            "perf/peak_allocated_gb": peak_allocated_gb,
            "perf/peak_reserved_gb": peak_reserved_gb,
            **round_values,
        }
        verdict: MultiValidationVerdict | None = None
        history_extra: dict[str, Any] = {}
        if controller is not None:
            assert samples is not None
            verdict = controller.step(val_metrics)
            stop_reason = verdict.stop_reason
            log_payload.update(
                {f"val/{name}_median": value for name, value in verdict.smoothed.items()}
            )
            for name, stats in samples.clip_stats.items():
                log_payload[f"val/{name}_clip_mean"] = stats.mean
                log_payload[f"val/{name}_clip_std"] = stats.std
                log_payload[f"val/{name}_clips"] = stats.count
            log_payload["validation/stale_rounds"] = controller.stale_rounds
            log_payload["validation/rounds_completed"] = controller.rounds_completed
            history_extra = {
                "validation/improved_median": list(verdict.improved_median),
                "validation/improved_raw": list(verdict.improved_raw),
                "validation/stop_reason": verdict.stop_reason,
            }
        with history_path.open("a") as history:
            history.write(json.dumps({**log_payload, **history_extra}, sort_keys=True) + "\n")
        wandb.log(log_payload)
        if upload:
            store.upload_file(history_path, "checkpoints/validation_history.jsonl")

        last_ckpt_path = run_dir / "last.ckpt"
        _save_checkpoint(model, last_ckpt_path)
        _upload_checkpoint_and_record(
            store=store,
            upload_enabled=upload,
            path=last_ckpt_path,
            run=run,
            summary_key="r2/last_checkpoint",
        )

        if verdict is not None and samples is not None:
            samples_path = run_dir / "val_samples" / f"round_{epoch:04d}.npz"
            _save_validation_samples(samples_path, samples)
            improved = _link_best_checkpoints(
                verdict=verdict,
                scores=val_metrics,
                run_dir=run_dir,
                last_ckpt_path=last_ckpt_path,
                checkpoint_index=checkpoint_index,
                epoch=epoch,
                optimizer_steps=optimizer_steps,
                store=store,
                upload=upload,
                run=run,
            )
            if improved:
                checkpoint_index_path.write_text(
                    json.dumps(checkpoint_index, indent=2, sort_keys=True) + "\n"
                )
                if upload:
                    store.upload_file(checkpoint_index_path, "checkpoints/best_checkpoints.json")
                    store.upload_file(samples_path, f"val_samples/{samples_path.name}")

        if cfg.checkpoint_every and (epoch + 1) % cfg.checkpoint_every == 0:
            periodic_ckpt_path = run_dir / f"step{optimizer_steps}.ckpt"
            _replace_link_or_copy(last_ckpt_path, periodic_ckpt_path)
            uri = (
                store.alias_checkpoint(last_ckpt_path.name, periodic_ckpt_path.name)
                if upload
                else None
            )
            if uri is not None and run is not None:
                run.summary[f"r2/checkpoint_step{optimizer_steps}"] = uri

        train_state_path = run_dir / TRAIN_STATE_NAME
        _save_train_state(
            train_state_path,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            optimizer_steps=optimizer_steps,
            controller=controller,
        )
        _upload_checkpoint_and_record(
            store=store,
            upload_enabled=upload,
            path=train_state_path,
            run=run,
            summary_key="r2/train_state",
        )

        if stop_reason is None and (
            max_optimizer_steps is not None and optimizer_steps >= max_optimizer_steps
        ):
            stop_reason = "censored at optimizer-step budget"
        if stop_reason:
            break

    if stop_reason:
        logger.info(
            "%s stopped after validation round %d at optimizer step %d: %s",
            cfg.experiment_name,
            epoch,
            optimizer_steps,
            stop_reason,
        )
    if run is not None and stop_reason is not None:
        run.summary["stop_reason"] = stop_reason
    wandb.finish()
    return summary(epoch)
