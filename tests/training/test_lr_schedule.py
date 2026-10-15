"""The step-based warm-up + cosine LR schedule of ``training.loop``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from training.loop import build_lr_scheduler, warmup_cosine_factor

PROTOCOL = {"warmup_steps": 5_000, "total_steps": 100_000, "min_lr_frac": 0.01}


@pytest.mark.parametrize(
    ("step", "factor"),
    [
        (0, 0.0),  # warm-up starts from zero
        (2_500, 0.5),  # linear half-way
        (5_000, 1.0),  # max LR at the end of warm-up
        (52_500, 0.505),  # cosine mid-point: (1 + 0.01) / 2
        (100_000, 0.01),  # floor at the step budget
        (150_000, 0.01),  # constant after it
    ],
)
def test_protocol_schedule_values(step, factor):
    assert warmup_cosine_factor(step, **PROTOCOL) == pytest.approx(factor)


def test_cosine_is_monotone_between_warmup_and_budget():
    values = [warmup_cosine_factor(step, **PROTOCOL) for step in range(5_000, 100_001, 500)]
    assert all(a >= b for a, b in zip(values, values[1:], strict=False))


def test_scheduler_lr_is_a_function_of_optimizer_steps_and_survives_a_state_round_trip():
    cfg = SimpleNamespace(warmup_steps=10, total_steps=100, min_lr_frac=0.01)

    def fresh():
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=3e-4)
        return optimizer, build_lr_scheduler(optimizer, cfg)

    optimizer, scheduler = fresh()
    assert optimizer.param_groups[0]["lr"] == 0.0
    for _ in range(40):
        optimizer.step()
        scheduler.step()
    expected = 3e-4 * warmup_cosine_factor(40, warmup_steps=10, total_steps=100, min_lr_frac=0.01)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected)

    resumed_optimizer, resumed = fresh()
    resumed_optimizer.load_state_dict(optimizer.state_dict())
    resumed.load_state_dict(scheduler.state_dict())
    for _ in range(5):
        optimizer.step()
        scheduler.step()
        resumed_optimizer.step()
        resumed.step()
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(optimizer.param_groups[0]["lr"])


def test_schedule_rejects_inconsistent_bounds():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    with pytest.raises(ValueError, match="warmup_steps"):
        build_lr_scheduler(
            optimizer, SimpleNamespace(warmup_steps=100, total_steps=100, min_lr_frac=0.01)
        )
