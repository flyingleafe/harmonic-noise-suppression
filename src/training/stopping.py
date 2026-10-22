"""Checkpoint selection and the saturation stop for full-panel validation.

:class:`MultiMetricController` folds one validation round's scores in and
reports, per tracked score, whether its RAW round value set a new best and
whether its median over the last ``smoothing_window`` rounds improved by at
least ``min_relative_improvement`` (relative, i.e. on a log scale, so every
score is judged scale-free). The loop links ``best_<score>_raw.ckpt`` /
``best_<score>_median.ckpt`` to ``last.ckpt`` on those verdicts.

The stop is decided on the CONTROL scores alone (``validation.control``, a
list: the protocol's ``[real_macro]``, or ``[real_macro, static_comb]`` for a
synthetic-trained arm): ``stale_rounds`` counts full-window rounds since the
median of ANY control score last improved, and the run stops
(``"saturation"``) once ``rounds_completed >= min_rounds`` and
``stale_rounds >= final_patience``. So a run that still improves on one
control keeps training while it gets worse on another. The other primaries
only select checkpoints — a sub-panel cell drifting down by 1 % cannot keep a
converged run alive. The controller never touches the optimizer: the LR is a
pure function of the step count (``training.loop``).

A non-finite round never reaches the controller (it raises on one): the loop
stops the run (``"nonfinite"``) before anything for that round is
checkpointed. All state rides in ``train_state.pt`` under :data:`STATE_KEY`,
so a resumed run continues the same windows and counters.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["STATE_KEY", "MultiValidationVerdict", "MultiMetricController", "control_scores"]

STATE_KEY = "multi_validation"


@dataclass(frozen=True)
class MultiValidationVerdict:
    smoothed: dict[str, float]
    improved_median: tuple[str, ...]
    improved_raw: tuple[str, ...]
    stop_reason: str | None  # "saturation" | None


def control_scores(control: Any) -> tuple[str, ...]:
    """``validation.control`` as a tuple of score names (a non-empty list)."""
    if isinstance(control, str) or not control:
        raise ValueError(f"validation.control must be a non-empty list of scores, got {control!r}")
    return tuple(str(name) for name in control)


class MultiMetricController:
    """Dual-best bookkeeping per score; saturation stop on the control scores."""

    def __init__(self, cfg: Any, names: tuple[str, ...]) -> None:
        self.window = int(cfg.smoothing_window)
        self.min_log_improvement = math.log1p(float(cfg.min_relative_improvement))
        self.min_rounds = int(cfg.min_rounds)
        self.final_patience = int(cfg.final_patience)
        self.control = control_scores(cfg.control)
        missing = [name for name in self.control if name not in names]
        if missing:
            raise ValueError(
                f"validation.control {missing} must be among the primary scores {names}"
            )
        self.names = names
        self.windows: dict[str, deque[float]] = {name: deque(maxlen=self.window) for name in names}
        self.best_smoothed: dict[str, float] = {}
        self.best_raw: dict[str, float] = {}
        self.stale_rounds = 0
        self.rounds_completed = 0
        self.stop_reason: str | None = None

    def _improved(self, value: float, best: float) -> bool:
        if value < 0.0 or best < 0.0:
            raise ValueError("log-scale validation scores must be non-negative")
        if best == 0.0:
            return False
        if value == 0.0:
            return True
        return math.log(best) - math.log(value) >= self.min_log_improvement

    def step(self, scores: Mapping[str, float]) -> MultiValidationVerdict:
        self.rounds_completed += 1
        values = {name: float(scores[name]) for name in self.names}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"non-finite validation scores reached the controller: {values}")

        improved_raw: list[str] = []
        for name, value in values.items():
            best = self.best_raw.get(name)
            if best is None or value < best:
                self.best_raw[name] = value
                improved_raw.append(name)
            self.windows[name].append(value)
        smoothed = {name: statistics.median(window) for name, window in self.windows.items()}
        if any(len(window) < self.window for window in self.windows.values()):
            return MultiValidationVerdict(smoothed, (), tuple(improved_raw), None)

        improved_median: list[str] = []
        for name, value in smoothed.items():
            best = self.best_smoothed.get(name)
            if best is None or self._improved(value, best):
                self.best_smoothed[name] = value
                improved_median.append(name)

        if any(name in improved_median for name in self.control):
            self.stale_rounds = 0
        else:
            self.stale_rounds += 1
        if self.rounds_completed >= self.min_rounds and self.stale_rounds >= self.final_patience:
            self.stop_reason = "saturation"
        return MultiValidationVerdict(
            smoothed=smoothed,
            improved_median=tuple(improved_median),
            improved_raw=tuple(improved_raw),
            stop_reason=self.stop_reason,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "control": list(self.control),
            "windows": {name: list(window) for name, window in self.windows.items()},
            "best_smoothed": dict(self.best_smoothed),
            "best_raw": dict(self.best_raw),
            "stale_rounds": self.stale_rounds,
            "rounds_completed": self.rounds_completed,
            "stop_reason": self.stop_reason,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        saved_control = state["control"]
        # a run written before ``control`` became a list stored one name
        if isinstance(saved_control, str):
            saved_control = [saved_control]
        if tuple(state["names"]) != self.names or tuple(saved_control) != self.control:
            raise ValueError("saved validation scores/control differ from the current config")
        self.windows = {
            name: deque((float(value) for value in state["windows"][name]), maxlen=self.window)
            for name in self.names
        }
        self.best_smoothed = {str(k): float(v) for k, v in state["best_smoothed"].items()}
        self.best_raw = {str(k): float(v) for k, v in state["best_raw"].items()}
        self.stale_rounds = int(state["stale_rounds"])
        self.rounds_completed = int(state["rounds_completed"])
        self.stop_reason = state["stop_reason"]
