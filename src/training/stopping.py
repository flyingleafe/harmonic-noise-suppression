"""Noise-tolerant stopping policy — opt-in via ``early_stopping.enabled``.

The historical loop stops after ``RootConfig.patience`` raw epochs without a
new best, which a noisy validation metric trips long before a model has
saturated. :class:`StoppingController` replaces that decision (never the
``best.ckpt`` selection, which stays on the RAW monitor value) with:

* **Smoothing** — every check uses the median of the last ``median_window``
  raw monitor values (a partial window at the start). The same smoothed
  value drives ``ReduceLROnPlateau``, whose ``threshold`` is re-derived each
  epoch so scheduler and controller share one notion of *meaningful*
  improvement: ``max(min_delta_abs, min_delta_rel * |best_smoothed|)``.
* **Saturation stop** — all four gates at once: ``epoch + 1 >= min_epochs``
  (absolute, counting the original run's epochs on a continuation),
  ``patience`` consecutive checks without meaningful improvement,
  ``min_lr_reductions`` ACTUAL reductions observed on the optimizer, and
  ``lr_grace_epochs`` checks since the latest one.
* **Non-finite stop** — a NaN/inf train loss or monitor ends the run before
  anything for that epoch is checkpointed, so ``last.ckpt``/``train_state``
  keep the last good weights.
* **Divergence guard** — a *finite* but sustained worsening: the raw train
  loss AND the smoothed monitor are both worse than their best-so-far by
  ``diverge_rel`` (relative; the monitor side floored at ``min_delta_abs``)
  for ``diverge_epochs`` consecutive epochs. A single spike resets the
  streak. The first trigger reduces the LR once (``scheduler.factor``) as a
  recovery attempt; a second full streak stops the run.
* **Deadline** — ``deadline_unix`` is checked by the loop between epochs
  (before the first and after each saved one), never mid-epoch.

Every counter lives in :meth:`StoppingController.state_dict` and is persisted
under ``train_state["early_stopping"]`` next to the optimizer/scaler, so a
preempted run continues the policy instead of restarting it. A bundle without
that key (a freshly prepared continuation) starts a clean controller.
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from dataclasses import dataclass
from typing import Any

import torch

__all__ = ["StoppingController", "Verdict", "STATE_KEY"]

logger = logging.getLogger(__name__)

STATE_KEY = "early_stopping"


@dataclass(frozen=True)
class Verdict:
    smoothed: float
    lr_reduced: bool
    stop_reason: str | None  # "nonfinite" | "divergence" | "saturation" | None


class StoppingController:
    def __init__(self, cfg: Any, *, mode: str) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"monitor_mode must be 'min' or 'max', got {mode!r}")
        self.cfg = cfg
        self.mode = mode
        self.window: list[float] = []
        self.best_smoothed: float | None = None
        self.checks_without_improvement = 0
        self.lr_reductions = 0
        self.last_reduction_epoch: int | None = None
        self.best_train_loss: float | None = None
        self.diverge_streak = 0
        self.recovery_attempts = 0
        self.stop_reason: str | None = None

    # ── persistence ────────────────────────────────────────────────────────

    _FIELDS = (
        "window",
        "best_smoothed",
        "checks_without_improvement",
        "lr_reductions",
        "last_reduction_epoch",
        "best_train_loss",
        "diverge_streak",
        "recovery_attempts",
        "stop_reason",
    )

    def state_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self._FIELDS}
        d["window"] = list(self.window)
        return d

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for k in self._FIELDS:
            setattr(self, k, state[k])
        self.window = list(self.window)

    # ── comparisons ────────────────────────────────────────────────────────

    def min_delta(self, best: float | None) -> float:
        """Meaningful-improvement margin around ``best``: ``max(abs, rel*|best|)``."""
        if best is None or not math.isfinite(best):
            return float(self.cfg.min_delta_abs)
        return max(float(self.cfg.min_delta_abs), float(self.cfg.min_delta_rel) * abs(best))

    def _better(self, value: float, ref: float, margin: float) -> bool:
        return value < ref - margin if self.mode == "min" else value > ref + margin

    def _worse(self, value: float, ref: float, margin: float) -> bool:
        return value > ref + margin if self.mode == "min" else value < ref - margin

    def deadline_reached(self, now: float | None = None) -> bool:
        deadline = self.cfg.deadline_unix
        return deadline is not None and (time.time() if now is None else now) >= float(deadline)

    # ── one check per epoch ────────────────────────────────────────────────

    def step(
        self,
        *,
        epoch: int,
        train_loss: float,
        raw_metric: float,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    ) -> Verdict:
        """Fold this epoch's results in, step the scheduler on the smoothed
        monitor, and decide whether to stop. On ``"nonfinite"`` nothing is
        recorded: the caller must not checkpoint that epoch."""
        if not (math.isfinite(train_loss) and math.isfinite(raw_metric)):
            logger.error(
                "epoch %d: non-finite train_loss=%r / monitor=%r; stopping without "
                "saving this epoch (last good weights and train state kept)",
                epoch,
                train_loss,
                raw_metric,
            )
            return Verdict(smoothed=math.nan, lr_reduced=False, stop_reason="nonfinite")

        self.window.append(float(raw_metric))
        del self.window[: -int(self.cfg.median_window)]
        smoothed = statistics.median(self.window)

        # Scheduler and controller agree on what "improved" means.
        scheduler.threshold = self.min_delta(scheduler.best)
        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step(smoothed)
        reduced = optimizer.param_groups[0]["lr"] < lr_before

        if self.best_smoothed is None or self._better(
            smoothed, self.best_smoothed, self.min_delta(self.best_smoothed)
        ):
            self.best_smoothed = smoothed
            self.checks_without_improvement = 0
        else:
            self.checks_without_improvement += 1

        # Divergence: both signals must be clearly worse than their best.
        if self.best_train_loss is None or train_loss < self.best_train_loss:
            self.best_train_loss = float(train_loss)
        rel = float(self.cfg.diverge_rel)
        train_worse = train_loss - self.best_train_loss > rel * abs(self.best_train_loss)
        diverging = train_worse and self._worse(
            smoothed,
            self.best_smoothed,
            max(float(self.cfg.min_delta_abs), rel * abs(self.best_smoothed)),
        )
        self.diverge_streak = self.diverge_streak + 1 if diverging else 0
        if self.diverge_streak >= int(self.cfg.diverge_epochs):
            self.diverge_streak = 0
            if self.recovery_attempts == 0:
                self.recovery_attempts = 1
                if not reduced:
                    for group in optimizer.param_groups:
                        group["lr"] *= scheduler.factor
                    scheduler.num_bad_epochs = 0
                    scheduler.cooldown_counter = scheduler.cooldown
                    reduced = True
                logger.warning(
                    "epoch %d: sustained divergence (%d epochs); one recovery LR reduction to %.3g",
                    epoch,
                    self.cfg.diverge_epochs,
                    optimizer.param_groups[0]["lr"],
                )
            else:
                self.stop_reason = "divergence"

        if reduced:
            self.lr_reductions += 1
            self.last_reduction_epoch = epoch
        since_reduction = (
            math.inf if self.last_reduction_epoch is None else epoch - self.last_reduction_epoch
        )
        if self.stop_reason is None and (
            epoch + 1 >= int(self.cfg.min_epochs)
            and self.checks_without_improvement >= int(self.cfg.patience)
            and self.lr_reductions >= int(self.cfg.min_lr_reductions)
            and since_reduction >= int(self.cfg.lr_grace_epochs)
        ):
            self.stop_reason = "saturation"

        logger.info(
            "epoch %d: monitor=%.4g median=%.4g best=%.4g no_improve=%d/%d "
            "lr_reductions=%d/%d since_reduction=%s streak=%d%s",
            epoch,
            raw_metric,
            smoothed,
            self.best_smoothed,
            self.checks_without_improvement,
            self.cfg.patience,
            self.lr_reductions,
            self.cfg.min_lr_reductions,
            since_reduction,
            self.diverge_streak,
            f" -> STOP ({self.stop_reason})" if self.stop_reason else "",
        )
        return Verdict(smoothed=smoothed, lr_reduced=reduced, stop_reason=self.stop_reason)
