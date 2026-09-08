"""Behavioral regressions for ``training.stopping.StoppingController``.

Each test drives the controller with a scripted metric sequence through a
real ``ReduceLROnPlateau`` — the same wiring ``training.loop`` uses.
"""

from __future__ import annotations

import math

import pytest
import torch

from training.config import EarlyStoppingConfig
from training.stopping import StoppingController


def _rig(es: EarlyStoppingConfig, *, sched_patience: int, cooldown: int = 0, lr: float = 1e-3):
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        patience=sched_patience,
        factor=0.5,
        cooldown=cooldown,
        threshold=es.min_delta_abs,
        threshold_mode="abs",
    )
    return StoppingController(es, mode="min"), opt, sched


def _drive(ctrl, opt, sched, values, *, start=0, train_loss=None):
    """Feed ``values`` as successive epochs (halting on a stop verdict, as the
    loop does); return the list of verdicts."""
    out = []
    for i, v in enumerate(values):
        tl = train_loss[i] if train_loss is not None else 1.0
        out.append(
            ctrl.step(epoch=start + i, train_loss=tl, raw_metric=v, optimizer=opt, scheduler=sched)
        )
        if out[-1].stop_reason:
            break
    return out


def test_single_spike_neither_resets_patience_nor_counts_as_improvement():
    es = EarlyStoppingConfig(enabled=True, median_window=5, min_epochs=0, patience=100)
    ctrl, opt, sched = _rig(es, sched_patience=1000)

    _drive(ctrl, opt, sched, [10.0] * 8)
    stalled = ctrl.checks_without_improvement
    assert stalled == 7  # first value set the best, the rest did not move it

    # One spuriously GOOD epoch: the 5-median stays at 10, so it is not a new
    # best and the stall counter keeps climbing instead of resetting.
    _drive(ctrl, opt, sched, [5.0], start=8)
    assert ctrl.best_smoothed == 10.0
    assert ctrl.checks_without_improvement == stalled + 1

    # One spuriously BAD epoch: no divergence streak starts either.
    _drive(ctrl, opt, sched, [50.0], start=9)
    assert ctrl.diverge_streak == 0
    assert ctrl.stop_reason is None


def test_saturation_waits_for_min_epochs_then_lr_reductions_then_grace():
    es = EarlyStoppingConfig(
        enabled=True,
        median_window=1,
        min_epochs=12,
        patience=3,
        min_lr_reductions=2,
        lr_grace_epochs=4,
    )
    ctrl, opt, sched = _rig(es, sched_patience=4, cooldown=0)
    verdicts = _drive(ctrl, opt, sched, [10.0] * 40)
    stop_at = len(verdicts) - 1
    assert verdicts[stop_at].stop_reason == "saturation"

    # Patience (3) was exhausted by epoch 3, but the epoch floor is 12 ...
    assert all(v.stop_reason is None for v in verdicts[:12])
    # ... and the scheduler needed patience=4 bad epochs per reduction, so the
    # second reduction landed at epoch 10 and the 4-epoch grace ends at 14.
    assert [v.lr_reduced for v in verdicts].index(True) == 5
    assert ctrl.lr_reductions == 2
    assert ctrl.last_reduction_epoch == 10
    assert stop_at == 14
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3 * 0.25)


def test_serialized_state_continues_exactly_like_an_uninterrupted_run():
    es = EarlyStoppingConfig(
        enabled=True,
        median_window=3,
        min_epochs=6,
        patience=4,
        min_lr_reductions=1,
        lr_grace_epochs=2,
    )
    values = [10.0, 9.0, 8.5, 8.4, 8.4, 8.6, 8.3, 8.5, 8.4, 8.4, 8.4, 8.4, 8.4, 8.4, 8.4]

    ref_ctrl, ref_opt, ref_sched = _rig(es, sched_patience=2)
    ref = _drive(ref_ctrl, ref_opt, ref_sched, values)

    a_ctrl, a_opt, a_sched = _rig(es, sched_patience=2)
    first = _drive(a_ctrl, a_opt, a_sched, values[:6])
    saved = {
        "early_stopping": a_ctrl.state_dict(),
        "scheduler": a_sched.state_dict(),
        "optimizer": a_opt.state_dict(),
    }

    b_ctrl, b_opt, b_sched = _rig(es, sched_patience=2)
    b_ctrl.load_state_dict(saved["early_stopping"])
    b_sched.load_state_dict(saved["scheduler"])
    b_opt.load_state_dict(saved["optimizer"])
    second = _drive(b_ctrl, b_opt, b_sched, values[6:], start=6)

    assert first + second == ref
    assert b_ctrl.state_dict() == ref_ctrl.state_dict()
    assert any(v.stop_reason == "saturation" for v in second)


def test_fresh_controller_has_no_history_and_absent_state_is_not_loaded():
    es = EarlyStoppingConfig(enabled=True)
    ctrl = StoppingController(es, mode="min")
    state = ctrl.state_dict()
    assert state["window"] == [] and state["lr_reductions"] == 0
    assert state["last_reduction_epoch"] is None and state["stop_reason"] is None


def test_sustained_divergence_gets_one_lr_recovery_then_stops():
    es = EarlyStoppingConfig(
        enabled=True, median_window=1, min_epochs=0, diverge_rel=0.25, diverge_epochs=3
    )
    ctrl, opt, sched = _rig(es, sched_patience=1000)
    _drive(ctrl, opt, sched, [1.0, 1.0], train_loss=[1.0, 1.0])  # establish bests

    # A single bad epoch on BOTH signals is a spike, not divergence.
    _drive(ctrl, opt, sched, [2.0], start=2, train_loss=[2.0])
    _drive(ctrl, opt, sched, [1.05], start=3, train_loss=[1.05])
    assert ctrl.diverge_streak == 0 and ctrl.recovery_attempts == 0

    # Train loss worsening alone (val fine) never counts.
    _drive(ctrl, opt, sched, [1.0] * 3, start=4, train_loss=[2.0] * 3)
    assert ctrl.diverge_streak == 0

    # Three consecutive epochs worse on both -> ONE recovery LR reduction.
    v = _drive(ctrl, opt, sched, [2.0] * 3, start=7, train_loss=[2.0] * 3)
    assert [x.lr_reduced for x in v] == [False, False, True]
    assert opt.param_groups[0]["lr"] == pytest.approx(5e-4)
    assert ctrl.recovery_attempts == 1 and ctrl.lr_reductions == 1
    assert ctrl.stop_reason is None

    # Persisting after the recovery -> stop.
    v = _drive(ctrl, opt, sched, [2.0] * 3, start=10, train_loss=[2.0] * 3)
    assert [x.stop_reason for x in v] == [None, None, "divergence"]
    assert opt.param_groups[0]["lr"] == pytest.approx(5e-4)  # no second reduction


def test_nonfinite_epoch_stops_and_leaves_state_untouched():
    es = EarlyStoppingConfig(enabled=True, median_window=3, min_epochs=0)
    ctrl, opt, sched = _rig(es, sched_patience=1000)
    _drive(ctrl, opt, sched, [3.0, 2.0])
    before = ctrl.state_dict()
    v = ctrl.step(epoch=2, train_loss=math.nan, raw_metric=1.0, optimizer=opt, scheduler=sched)
    assert v.stop_reason == "nonfinite"
    v = ctrl.step(epoch=2, train_loss=1.0, raw_metric=math.inf, optimizer=opt, scheduler=sched)
    assert v.stop_reason == "nonfinite"
    assert ctrl.state_dict() == before  # not recorded: resume re-runs this epoch


def test_deadline_is_a_wall_clock_gate():
    ctrl = StoppingController(EarlyStoppingConfig(enabled=True, deadline_unix=1000.0), mode="min")
    assert not ctrl.deadline_reached(now=999.0)
    assert ctrl.deadline_reached(now=1000.0)
    assert not StoppingController(EarlyStoppingConfig(enabled=True), mode="min").deadline_reached()
