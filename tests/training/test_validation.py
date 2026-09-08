from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data_processing.collate import frame_collate
from losses import PITMSELoss
from tasks.codecs import RPSPredictionCodec
from tests.training._fixtures import TinyRPSModel
from training.validation import (
    MultiMetricController,
    build_validation_plan,
    validate_rps,
)


def _dataset(seed: int, size: int) -> dict:
    return {
        "_target_": "tests.training._fixtures.TinyRPSFrameDataset",
        "params": {
            "n_samples": size,
            "duration_s": 0.5,
            "sample_rate": 16000,
            "seed": seed,
        },
        "iterable": False,
    }


def test_validation_plan_reuses_datasets_for_overlapping_views():
    cfg = OmegaConf.create(
        {
            "datasets": {"real": _dataset(1, 8), "synthetic": _dataset(2, 4)},
            "views": {
                "r1": {
                    "dataset": "real",
                    "start": 0,
                    "stop": 8,
                    "channels": [0],
                    "channels_per_clip": 2,
                },
                "r2": {"dataset": "real", "start": 0, "stop": 8},
                "synthetic": {"dataset": "synthetic"},
            },
            "aggregates": {
                "overall": {"r2": 0.5, "synthetic": 0.5},
            },
            "primary": ["r1", "r2", "synthetic", "overall"],
            "control": "overall",
        }
    )
    plan = build_validation_plan(cfg)

    assert plan.size == 12
    assert len(plan.dataset.datasets) == 2
    views = {view.name: view for view in plan.views}
    assert (views["r1"].start, views["r1"].stop, views["r1"].channels) == (0, 8, (0,))
    assert (views["synthetic"].start, views["synthetic"].stop) == (8, 12)


def test_gpu_style_validation_reduces_overlapping_views_in_one_pass():
    cfg = OmegaConf.create(
        {
            "datasets": {"real": _dataset(1, 8), "synthetic": _dataset(2, 4)},
            "views": {
                "r1": {
                    "dataset": "real",
                    "start": 0,
                    "stop": 8,
                    "channels": [0],
                    "channels_per_clip": 2,
                },
                "r2": {"dataset": "real"},
                "synthetic": {"dataset": "synthetic"},
            },
            "aggregates": {"overall": {"r2": 0.5, "synthetic": 0.5}},
            "primary": ["r1", "r2", "synthetic", "overall"],
            "control": "overall",
        }
    )
    plan = build_validation_plan(cfg)
    loader = DataLoader(plan.dataset, batch_size=5, collate_fn=frame_collate)
    scores, loss = validate_rps(
        model=TinyRPSModel(),
        codec=RPSPredictionCodec(frame_rate=(125, 4)),
        loss_fn=PITMSELoss(rate=(125, 4)),
        valid_loader=loader,
        plan=plan,
        device=torch.device("cpu"),
        amp=False,
        amp_dtype=None,
    )

    assert set(scores) == {"r1", "r2", "synthetic", "overall"}
    assert scores["overall"] == pytest.approx(0.5 * scores["r2"] + 0.5 * scores["synthetic"])
    assert all(torch.isfinite(torch.tensor(value)) for value in [*scores.values(), loss])


def test_any_subset_progress_delays_lr_reduction_and_all_subset_stagnation_stops():
    cfg = SimpleNamespace(
        smoothing_window=1,
        min_rounds=0,
        min_relative_improvement=0.01,
        lr_patience=2,
        lr_factor=0.5,
        min_lr_reductions=2,
        final_patience=2,
    )
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    controller = MultiMetricController(cfg, ("real", "synthetic"))

    assert controller.step({"real": 5.0, "synthetic": 5.0}, optimizer).improved
    assert not controller.step({"real": 5.0, "synthetic": 4.9}, optimizer).lr_reduced
    assert not controller.step({"real": 5.0, "synthetic": 4.9}, optimizer).lr_reduced
    assert controller.step({"real": 5.0, "synthetic": 4.9}, optimizer).lr_reduced
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
    controller.step({"real": 5.0, "synthetic": 4.9}, optimizer)
    assert controller.step({"real": 5.0, "synthetic": 4.9}, optimizer).lr_reduced
    controller.step({"real": 5.0, "synthetic": 4.9}, optimizer)
    verdict = controller.step({"real": 5.0, "synthetic": 4.9}, optimizer)
    assert verdict.stop_reason == "saturation"


def test_meaningful_improvement_is_scale_invariant_in_log_space():
    cfg = SimpleNamespace(
        smoothing_window=1,
        min_rounds=0,
        min_relative_improvement=0.01,
        lr_patience=10,
        lr_factor=0.5,
        min_lr_reductions=1,
        final_patience=10,
    )
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    controller = MultiMetricController(cfg, ("large", "small"))
    controller.step({"large": 100.0, "small": 1.0}, optimizer)

    subpercent = controller.step({"large": 99.5, "small": 0.995}, optimizer)
    meaningful = controller.step({"large": 98.9, "small": 0.989}, optimizer)

    assert subpercent.improved == ()
    assert set(meaningful.improved) == {"large", "small"}


def test_revised_policy_stops_after_exactly_thirty_six_stagnant_rounds():
    cfg = SimpleNamespace(
        smoothing_window=1,
        min_rounds=30,
        min_relative_improvement=0.01,
        lr_patience=8,
        lr_factor=0.5,
        min_lr_reductions=3,
        final_patience=12,
    )
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1e-3)
    controller = MultiMetricController(cfg, ("score",))
    controller.step({"score": 1.0}, optimizer)

    verdicts = [controller.step({"score": 1.0}, optimizer) for _ in range(36)]

    assert [i + 1 for i, verdict in enumerate(verdicts) if verdict.lr_reduced] == [8, 16, 24]
    assert all(verdict.stop_reason is None for verdict in verdicts[:-1])
    assert verdicts[-1].stop_reason == "saturation"
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.25e-4)


def test_controller_state_resume_matches_uninterrupted_decisions():
    cfg = SimpleNamespace(
        smoothing_window=3,
        min_rounds=0,
        min_relative_improvement=0.01,
        lr_patience=2,
        lr_factor=0.5,
        min_lr_reductions=1,
        final_patience=2,
    )
    values = [5.0, 4.8, 4.7, 4.7, 4.7, 4.7]

    def rig() -> tuple[MultiMetricController, torch.optim.Optimizer]:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        return MultiMetricController(cfg, ("score",)), torch.optim.SGD([parameter], lr=1.0)

    full, full_opt = rig()
    full_verdicts = [full.step({"score": value}, full_opt) for value in values]

    first, first_opt = rig()
    for value in values[:3]:
        first.step({"score": value}, first_opt)
    resumed, resumed_opt = rig()
    resumed.load_state_dict(first.state_dict())
    resumed_opt.param_groups[0]["lr"] = first_opt.param_groups[0]["lr"]
    resumed_verdicts = [resumed.step({"score": value}, resumed_opt) for value in values[3:]]

    assert resumed_verdicts == full_verdicts[3:]
    assert resumed_opt.param_groups[0]["lr"] == full_opt.param_groups[0]["lr"]
