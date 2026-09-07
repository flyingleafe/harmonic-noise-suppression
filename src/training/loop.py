"""One generic training loop (docs/refactor-unified-framework.md § "train.py
/ eval.py").

Covers the union of behaviors of the three old trainers (``train.py``,
``train_rps_predictor.py``, ``train_noise_gen.py``): map-style *or* iterable
(online-mixing) datasets, AMP autocast + ``GradScaler``, gradient
accumulation, grad-norm clipping, an optimizer factory ported from
``train.py::get_optimizer``, ``ReduceLROnPlateau`` + early stopping on a
configurable monitor metric (raw ``patience`` by default, or the opt-in
median-smoothed policy in ``training.stopping`` when
``early_stopping.enabled``), checkpointing (``best.ckpt`` best-monitor +
``last.ckpt`` latest-epoch + optional periodic
``ep{N}_{monitor}_{value:.4f}.ckpt``), and wandb logging (run name =
experiment name, git commit hash, dirty-tree guard, run-id file for the
job runner to pick up — mirrors ``train.py::wandb_init``).

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
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import tdseries as td
import torch
from torch.amp.grad_scaler import GradScaler
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
from training.stopping import STATE_KEY, StoppingController
from training.validation import (
    MULTI_VALIDATION_STATE_KEY,
    MultiMetricController,
    build_validation_plan,
    validate_rps,
)

__all__ = ["run_training", "get_optimizer", "git_commit_hash", "is_git_dirty"]

logger = logging.getLogger(__name__)


# ─── Optimizer factory (ported from train.py::get_optimizer) ─────────────────


def get_optimizer(
    model: torch.nn.Module,
    *,
    name: str,
    lr: float,
    weight_decay: float = 0.0,
    extra_params: dict[str, Any] | None = None,
) -> torch.optim.Optimizer:
    """Build an optimizer by name — adam/adamw/radam/rmsprop/sgd/prodigy/adamw8bit.

    ``weight_decay`` is a named kwarg so every optimizer gets it uniformly
    (the original ``train.py`` version relied on each optimizer accepting it
    via ``**optim_params``); ``extra_params`` are the optimizer-specific
    passthrough kwargs (``config.optimizer`` in the original).
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


def _better(mode: str):
    if mode == "min":
        return lambda new, best: new < best
    if mode == "max":
        return lambda new, best: new > best
    raise ValueError(f"optim.monitor_mode must be 'min' or 'max', got {mode!r}")


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
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: GradScaler,
    epoch: int,
    best_metric: float | None,
    no_improve: int,
    stopping: StoppingController | None = None,
    multi_validation: MultiMetricController | None = None,
    optimizer_steps: int = 0,
) -> None:
    """Persist everything needed to continue training after ``epoch``."""
    tmp = path.with_suffix(".pt.tmp")
    state: dict[str, Any] = {
        "optimizer": optimizer.state_dict(),
        "optimizer_steps": optimizer_steps,
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "next_epoch": epoch + 1,
        "best_metric": best_metric,
        "no_improve": no_improve,
    }
    if multi_validation is not None:
        state[MULTI_VALIDATION_STATE_KEY] = multi_validation.state_dict()
    if stopping is not None:
        state[STATE_KEY] = stopping.state_dict()
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
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: GradScaler,
    device: torch.device,
    store: ArtifactStore | None = None,
    stopping: StoppingController | None = None,
    multi_validation: MultiMetricController | None = None,
    progress: dict[str, int] | None = None,
) -> tuple[int, float | None, int]:
    """Restore an interrupted run in place; return ``(start_epoch, best, no_improve)``.

    Returns ``(0, None, 0)`` when there is nothing to resume from, so a first
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
        # epoch 101 of ~105, and only `last.ckpt`/`best.ckpt` had been uploaded
        # -- the optimizer, the plateau scheduler (its LR had decayed 1e-3 ->
        # 8e-6 over seven reductions), the epoch counter and the early-stop
        # counter were all on the dead VM.
        remote_state, remote_weights = _fetch_train_state(store, state_path, weights_path)
        state_local = state_local or remote_state
        weights_local = weights_local or remote_weights
    if state_local is None and weights_local is None:
        return 0, None, 0  # a genuine first launch: nothing to resume from
    if state_local is None or weights_local is None:
        # WEIGHTS WITHOUT BOOKKEEPING (or the reverse) MUST NOT START FRESH.
        # `best_metric` would come back None, and the epoch loop treats a None
        # best as "improved", so the very next epoch OVERWRITES `best.ckpt`
        # with a worse model and resets `no_improve` -- losing both the best
        # checkpoint and the early-stop counter, silently. Measured on
        # `m3mixv2_scv2`: its best sat at epoch 8 (val/mse 48.09) and the
        # `best.ckpt` left on R2 scores 85.4, its epoch-77 value, while the run
        # continued 69 epochs past a patience of 20. Refusing is strictly
        # better than destroying a run's best weights.
        missing = TRAIN_STATE_NAME if state_local is None else weights_path.name
        present = weights_path.name if state_local is None else TRAIN_STATE_NAME
        raise RuntimeError(
            f"cannot resume {run_dir.name}: found {present} but no {missing}. "
            f"Starting fresh here would overwrite best.ckpt with a worse model "
            f"and reset early stopping. Either restore {missing}, or run under a "
            f"new experiment_name, or set `resume=false` to start deliberately."
        )
    state_path, weights_path = state_local, weights_local
    state = torch.load(state_path, map_location=device, weights_only=False)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
    optimizer.load_state_dict(state["optimizer"])
    if multi_validation is not None:
        if MULTI_VALIDATION_STATE_KEY not in state:
            raise RuntimeError(
                "cannot resume a multi-validation run from state written by "
                "the single-monitor protocol; start a new experiment"
            )
        multi_validation.load_state_dict(state[MULTI_VALIDATION_STATE_KEY])
    if progress is not None:
        progress["optimizer_steps"] = int(state.get("optimizer_steps", 0))
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    if stopping is not None and STATE_KEY in state:
        # Absent on a freshly prepared continuation bundle: fresh controller.
        stopping.load_state_dict(state[STATE_KEY])
    start_epoch = int(state["next_epoch"])
    logger.info(
        "resuming %s at epoch %d (best=%s, no_improve=%d)",
        run_dir.name,
        start_epoch,
        state["best_metric"],
        state["no_improve"],
    )
    return start_epoch, state["best_metric"], int(state["no_improve"])


def _checkpoint_slug(name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in name).strip("_")


def _warm_start(
    model: torch.nn.Module,
    checkpoint: str,
    device: torch.device,
    *,
    artifacts_cfg: Any,
) -> None:
    """Initialise a curriculum stage from explicit weights or a subset best.

    ``best:<score>@<experiment>`` resolves the independently selected R2
    checkpoint, e.g. ``best:real_r3@m3cur_scv2_s1``. A curriculum always starts
    fresh optimizer/scheduler state; same-run continuation remains ``resume``.
    """
    if checkpoint.startswith("best:"):
        try:
            score, experiment = checkpoint.removeprefix("best:").rsplit("@", 1)
        except ValueError as error:
            raise ValueError("subset checkpoint syntax is best:<score>@<experiment>") from error
        checkpoint = (
            f"r2://{artifacts_cfg.bucket}/{artifacts_cfg.prefix}/{experiment}/"
            f"checkpoints/best_{_checkpoint_slug(score)}.ckpt"
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
    """Upload ``path`` (best/periodic checkpoint) to R2 and, if it succeeded,
    write its ``r2://...`` URI into the wandb run summary under
    ``summary_key`` (e.g. ``"r2/best_checkpoint"``). No-op when
    ``upload_enabled`` is false; never raises (``ArtifactStore`` already
    swallows its own upload failures)."""
    if not upload_enabled:
        return
    uri = store.upload_checkpoint(path)
    if uri is not None and run is not None:
        run.summary[summary_key] = uri


# ─── Main entry point ──────────────────────────────────────────────────────────


def run_training(cfg: Any, *, artifact_store: ArtifactStore | None = None) -> dict[str, float]:
    """Train per ``cfg`` (a composed ``training.config.RootConfig``).

    Sets up the run dir + wandb, builds every component, then runs
    epochs/iterable-stream-chunks with early stopping. Returns the best
    monitored metric value and the epoch it stopped at.

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
    if getattr(cfg, "checkpoint", None):
        _warm_start(model, str(cfg.checkpoint), device, artifacts_cfg=cfg.artifacts)
    loss_fn = build_losses(cfg.loss).to(device)
    metric_suite = build_metrics(cfg.metrics)
    validation_cfg = getattr(cfg, "validation", None)
    multi_validation_enabled = bool(
        validation_cfg is not None and getattr(validation_cfg, "enabled", False)
    )
    multi_cfg: Any = validation_cfg if multi_validation_enabled else None
    if multi_validation_enabled and task.name != "rps_prediction":
        raise ValueError("multi-validation currently requires task=rps_prediction")
    if multi_validation_enabled and getattr(cfg.early_stopping, "enabled", False):
        raise ValueError("validation.enabled and early_stopping.enabled are mutually exclusive")

    optimizer = get_optimizer(
        model,
        name=cfg.optim.optimizer,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        extra_params=dict(cfg.optim.optimizer_params or {}),
    )
    es_cfg = getattr(cfg, "early_stopping", None)
    stopping = (
        StoppingController(es_cfg, mode=cfg.optim.monitor_mode)
        if es_cfg is not None and es_cfg.enabled
        else None
    )
    scheduler_kwargs: dict[str, Any] = dict(
        mode=cfg.optim.monitor_mode,
        patience=cfg.optim.patience,
        factor=cfg.optim.factor,
        cooldown=getattr(cfg.optim, "cooldown", 0),
    )
    if stopping is not None:
        # Absolute threshold: the controller re-derives it every epoch from
        # the best smoothed value (see training.stopping).
        scheduler_kwargs.update(threshold=stopping.min_delta(None), threshold_mode="abs")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)
    better = _better(cfg.optim.monitor_mode)

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
        else batch_size
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
    multi_validation = (
        MultiMetricController(multi_cfg, validation_plan.primary)
        if validation_plan is not None and multi_cfg is not None
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

    monitor = str(multi_cfg.control) if multi_cfg is not None else cfg.optim.monitor
    progress: dict[str, int] = {}
    start_epoch, best_metric, no_improve = (
        _load_train_state(
            run_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            store=store if cfg.artifacts.upload_checkpoints else None,
            stopping=stopping,
            multi_validation=multi_validation,
            progress=progress,
        )
        if cfg.resume
        else (0, None, 0)
    )
    optimizer_steps = int(progress.get("optimizer_steps", 0))
    if stopping is not None and cfg.resume:
        for key in ("patience", "factor", "cooldown"):
            setattr(scheduler, key, scheduler_kwargs[key])
        scheduler.threshold_mode = "abs"

    epoch = start_epoch
    max_optimizer_steps = int(multi_cfg.max_optimizer_steps) if multi_cfg is not None else None
    done_reason: str | None = None
    if max_optimizer_steps is not None and optimizer_steps >= max_optimizer_steps:
        done_reason = "optimizer-step budget"
    elif multi_validation is not None and multi_validation.stop_reason:
        done_reason = multi_validation.stop_reason
    elif best_metric is not None and start_epoch >= cfg.epochs:
        done_reason = "epoch budget"
    elif (
        multi_validation is None
        and stopping is None
        and best_metric is not None
        and no_improve >= cfg.patience
    ):
        done_reason = "early stopping"
    elif stopping is not None and stopping.stop_reason:
        done_reason = stopping.stop_reason
    elif stopping is not None and stopping.deadline_reached():
        done_reason = "deadline"
    if done_reason:
        logger.info(
            "%s not training at validation round %d (%s)",
            cfg.experiment_name,
            start_epoch,
            done_reason,
        )
        wandb.finish()
        best = best_metric if best_metric is not None else math.nan
        return {
            f"best_{monitor}": best,
            "final_epoch": float(start_epoch),
            "optimizer_steps": float(optimizer_steps),
        }

    checkpoint_index_path = run_dir / "best_checkpoints.json"
    checkpoint_index = (
        json.loads(checkpoint_index_path.read_text()) if checkpoint_index_path.exists() else {}
    )
    stop_reason: str | None = None
    for epoch in range(start_epoch, cfg.epochs):
        if train_iterable:
            assert train_iter is not None and batches_per_epoch is not None
            n_batches = batches_per_epoch
            if max_optimizer_steps is not None:
                remaining_updates = max_optimizer_steps - optimizer_steps
                if remaining_updates <= 0:
                    stop_reason = "optimizer-step budget"
                    break
                n_batches = min(
                    n_batches,
                    remaining_updates * max(1, int(cfg.grad_accum_steps)),
                )
            batches = _take(train_iter, n_batches)
        else:
            batches = train_loader
            n_batches = len(train_loader)

        train_loss, completed_steps = _train_one_epoch(
            model=model,
            codec=codec,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            batches=batches,
            n_batches=n_batches,
            device=device,
            amp=cfg.amp,
            amp_dtype=amp_dtype,
            grad_clip=cfg.grad_clip,
            grad_accum_steps=max(1, cfg.grad_accum_steps),
            epoch=epoch,
        )
        optimizer_steps += completed_steps

        if validation_plan is not None:
            val_metrics, val_loss = validate_rps(
                model=model,
                codec=codec,
                loss_fn=loss_fn,
                valid_loader=valid_loader,
                plan=validation_plan,
                device=device,
                amp=cfg.amp,
                amp_dtype=amp_dtype,
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

        if multi_validation is not None:
            multi_verdict = multi_validation.step(val_metrics, optimizer)
            stop_reason = multi_verdict.stop_reason
            metric_value = val_metrics[monitor]
            smoothed_log = {
                f"val/{name}_median": value for name, value in multi_verdict.smoothed.items()
            }
            smoothed_log.update(
                {
                    f"val/{name}_log_median": math.log(value)
                    for name, value in multi_verdict.smoothed.items()
                    if value > 0.0
                }
            )
            stable_improvements = multi_verdict.improved
            if monitor in multi_validation.best_smoothed:
                best_metric = multi_validation.best_smoothed[monitor]
            no_improve = multi_validation.stale_rounds
        else:
            if monitor == "loss":
                metric_value = train_loss
            elif monitor == "val_loss":
                metric_value = val_loss
            else:
                metric_value = val_metrics[monitor]
            stable_improvements = ()
            if stopping is not None:
                verdict = stopping.step(
                    epoch=epoch,
                    train_loss=train_loss,
                    raw_metric=metric_value,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
                stop_reason = verdict.stop_reason
                smoothed_log = {f"val/{monitor}_median": verdict.smoothed}
            else:
                scheduler.step(metric_value)
                smoothed_log = {}

        if stop_reason == "nonfinite":
            wandb.log(
                {
                    "epoch": epoch,
                    "validation_round": epoch,
                    "optimizer_step": optimizer_steps,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                }
            )
            break

        lr = optimizer.param_groups[0]["lr"]
        log_payload = {
            "epoch": epoch,
            "validation_round": epoch,
            "optimizer_step": optimizer_steps,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "lr": lr,
            **{f"val/{key}": value for key, value in val_metrics.items()},
            **{
                f"val/{key}_log": math.log(value)
                for key, value in val_metrics.items()
                if multi_validation is not None and value > 0.0
            },
            **smoothed_log,
        }
        with (run_dir / "validation_history.jsonl").open("a") as history:
            history.write(json.dumps(log_payload, sort_keys=True) + "\n")
        wandb.log(log_payload)
        if cfg.artifacts.upload_checkpoints:
            store.upload_file(
                run_dir / "validation_history.jsonl",
                "checkpoints/validation_history.jsonl",
            )

        last_ckpt_path = run_dir / "last.ckpt"
        _save_checkpoint(model, last_ckpt_path)
        _upload_checkpoint_and_record(
            store=store,
            upload_enabled=cfg.artifacts.upload_checkpoints,
            path=last_ckpt_path,
            run=run,
            summary_key="r2/last_checkpoint",
        )

        if multi_validation is not None:
            for score_name in stable_improvements:
                slug = _checkpoint_slug(score_name)
                best_path = run_dir / f"best_{slug}.ckpt"
                _replace_link_or_copy(last_ckpt_path, best_path)
                uri = (
                    store.alias_checkpoint(last_ckpt_path.name, best_path.name)
                    if cfg.artifacts.upload_checkpoints
                    else None
                )
                if uri is not None and run is not None:
                    run.summary[f"r2/best_{slug}"] = uri
                checkpoint_index[score_name] = {
                    "checkpoint": best_path.name,
                    "validation_round": epoch,
                    "optimizer_step": optimizer_steps,
                    "raw": val_metrics[score_name],
                    "smoothed": multi_verdict.smoothed[score_name],
                    "scores": val_metrics,
                }
                if score_name == monitor:
                    legacy_best_path = run_dir / "best.ckpt"
                    _replace_link_or_copy(last_ckpt_path, legacy_best_path)
                    uri = (
                        store.alias_checkpoint(last_ckpt_path.name, legacy_best_path.name)
                        if cfg.artifacts.upload_checkpoints
                        else None
                    )
                    if uri is not None and run is not None:
                        run.summary["r2/best_checkpoint"] = uri
            if stable_improvements:
                checkpoint_index_path.write_text(
                    json.dumps(checkpoint_index, indent=2, sort_keys=True) + "\n"
                )
                if cfg.artifacts.upload_checkpoints:
                    store.upload_file(
                        checkpoint_index_path,
                        "checkpoints/best_checkpoints.json",
                    )
        else:
            improved = best_metric is None or better(metric_value, best_metric)
            if improved:
                best_metric = metric_value
                no_improve = 0
                best_ckpt_path = run_dir / "best.ckpt"
                _replace_link_or_copy(last_ckpt_path, best_ckpt_path)
                uri = (
                    store.alias_checkpoint(last_ckpt_path.name, best_ckpt_path.name)
                    if cfg.artifacts.upload_checkpoints
                    else None
                )
                if uri is not None and run is not None:
                    run.summary["r2/best_checkpoint"] = uri
            else:
                no_improve += 1

        if cfg.checkpoint_every and (epoch + 1) % cfg.checkpoint_every == 0:
            periodic_ckpt_path = run_dir / f"ep{epoch}_{monitor}_{metric_value:.4f}.ckpt"
            _replace_link_or_copy(last_ckpt_path, periodic_ckpt_path)
            uri = (
                store.alias_checkpoint(last_ckpt_path.name, periodic_ckpt_path.name)
                if cfg.artifacts.upload_checkpoints
                else None
            )
            if uri is not None and run is not None:
                run.summary[f"r2/checkpoint_ep{epoch}"] = uri

        train_state_path = run_dir / TRAIN_STATE_NAME
        _save_train_state(
            train_state_path,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_metric=best_metric,
            no_improve=no_improve,
            stopping=stopping,
            multi_validation=multi_validation,
            optimizer_steps=optimizer_steps,
        )
        _upload_checkpoint_and_record(
            store=store,
            upload_enabled=cfg.artifacts.upload_checkpoints,
            path=train_state_path,
            run=run,
            summary_key="r2/train_state",
        )

        if multi_validation is None and stopping is None and no_improve >= cfg.patience:
            stop_reason = "early stopping"
        elif stopping is not None and stop_reason is None and stopping.deadline_reached():
            stop_reason = "deadline"
        elif max_optimizer_steps is not None and optimizer_steps >= max_optimizer_steps:
            stop_reason = "optimizer-step budget"
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
    wandb.finish()
    best = best_metric if best_metric is not None else math.nan
    return {
        f"best_{monitor}": best,
        "final_epoch": float(epoch),
        "optimizer_steps": float(optimizer_steps),
    }
