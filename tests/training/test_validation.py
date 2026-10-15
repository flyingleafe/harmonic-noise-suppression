"""Full-panel validation: meta-predicate views, frame-weighted scores, clip
statistics and the per-sample arrays (``training.validation``)."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import tdseries as td
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data_processing.collate import frame_collate
from losses._common import get_tensor
from metrics.regimes import REGIMES, frame_regimes
from tasks.codecs import SalienceRPSCodec
from training.validation import build_validation_plan, rps_readout_for, validate_rps

FPS = 16000 / 512
PANEL_CLIPS = [
    {"dataset": "dregon", "source_type": "nosource", "profile": "cruise"},
    {"dataset": "dregon", "source_type": "loudspeaker", "profile": "zero_then_cruise"},
    {"dataset": "md2", "source_type": "nosource", "profile": "ramp"},
    {"dataset": "md2", "source_type": "mixed", "profile": "standby"},
]


def _panel_cfg(views: dict, *, clips=None, duration_s: float = 2.0, **extra):
    return OmegaConf.create(
        {
            "datasets": {
                "real": {
                    "_target_": "tests.training._fixtures.TinyPanelFrameDataset",
                    "params": {
                        "clips": clips or PANEL_CLIPS,
                        "n_channels": 2,
                        "duration_s": duration_s,
                    },
                    "iterable": False,
                }
            },
            "views": views,
            "aggregates": extra.pop("aggregates", {}),
            "primary": extra.pop("primary", list(views)),
            "control": extra.pop("control", next(iter(views))),
        }
    )


class _TableModel(torch.nn.Module):
    """Salience stand-in whose deployed decoder returns rows of a fixed RPS
    table in loader order, so expected errors are known independently."""

    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        self.table = table
        self.cursor = 0

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        return torch.zeros(mixture.shape[0], 6, 10)

    def decode_logits(self, logits: torch.Tensor, n_samples: int) -> torch.Tensor:
        out = self.table[self.cursor : self.cursor + logits.shape[0]]
        self.cursor += logits.shape[0]
        return out


def _run(plan, table: torch.Tensor):
    return validate_rps(
        model=_TableModel(table),
        codec=SalienceRPSCodec(frame_rate=(125, 4)),
        loss_fn=lambda pred_frame, batch: torch.zeros(()),
        valid_loader=DataLoader(plan.dataset, batch_size=3, collate_fn=frame_collate),
        plan=plan,
        device=torch.device("cpu"),
        amp=False,
        amp_dtype=None,
        readout=rps_readout_for("salience_rps"),
    )


def _targets(plan) -> torch.Tensor:
    return torch.stack(
        [get_tensor(cast(td.Frame, plan.dataset[i]), "rps") for i in range(plan.size)]
    )


def test_views_select_items_by_meta_predicates():
    plan = build_validation_plan(
        _panel_cfg(
            {
                "dregon": {"where": {"dataset": "dregon"}},
                "md2_source": {"where": {"dataset": "md2", "source_type": "mixed"}},
                "source": {"where": {"source_type": ["loudspeaker", "mixed"]}},
                "dregon_zero": {"where": {"dataset": "dregon"}, "regime": "zero"},
            }
        )
    )
    views = {view.name: view for view in plan.views}

    assert plan.size == 8
    assert views["dregon"].samples.tolist() == [True] * 4 + [False] * 4
    assert views["md2_source"].samples.tolist() == [False] * 6 + [True] * 2
    assert views["source"].samples.tolist() == [False, False, True, True, False, False, True, True]
    assert views["dregon_zero"].regime == REGIMES.index("zero")
    assert plan.clip_index.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert plan.channel.tolist() == [0, 1] * 4
    assert plan.frame_rate == pytest.approx(FPS)


@pytest.mark.parametrize(
    ("views", "message"),
    [
        ({"real": {"dataset": "real", "start": 0, "stop": 8}}, "unknown keys"),
        ({"md2_zero": {"where": {"dataset": "md2"}, "regime": "zero"}}, "selects no frames"),
        ({"x": {"where": {"dataset": "nope"}}}, "selects no frames"),
        ({"x": {"where": {"drone": "dregon"}}}, "lacks meta"),
        ({"x": {"where": {"dataset": "md2"}, "regime": "hover"}}, "not in"),
    ],
)
def test_plan_rejects_index_ranges_empty_cells_and_unknown_meta(views, message):
    with pytest.raises(ValueError, match=message):
        build_validation_plan(_panel_cfg(views))


def test_plan_requires_the_control_among_primaries():
    with pytest.raises(ValueError, match="control"):
        build_validation_plan(
            _panel_cfg(
                {"dregon": {"where": {"dataset": "dregon"}}},
                primary=["dregon"],
                control="real_macro",
            )
        )


def test_scores_are_frame_weighted_over_every_view_shape():
    views = {
        "dregon": {"where": {"dataset": "dregon"}},
        "md2": {"where": {"dataset": "md2"}},
        "dregon_zero": {"where": {"dataset": "dregon"}, "regime": "zero"},
        "dregon_cruise": {"where": {"dataset": "dregon"}, "regime": "cruise"},
        "md2_ramp": {"where": {"dataset": "md2"}, "regime": "ramp"},
        "real": {"where": {}},
        "real_cruise": {"where": {}, "regime": "cruise"},
    }
    plan = build_validation_plan(
        _panel_cfg(views, aggregates={"real_macro": {"dregon": 0.5, "md2": 0.5}})
    )
    targets = _targets(plan)
    n_frames = targets.shape[-1]
    # A per-item, per-frame offset common to all rotors, smaller than every
    # rotor gap, so the identity assignment is optimal and the error IS it.
    offsets = (
        0.005 * (torch.arange(plan.size)[:, None] + 1) * (1 + torch.arange(n_frames) / n_frames)
    )
    table = targets + offsets[:, None, :]
    scores, loss, samples = _run(plan, table)

    # Identity-assignment error of the realised float32 table, per frame.
    expected_err = (table - targets).abs().mean(dim=1).numpy()
    regimes = frame_regimes(targets, FPS).numpy()
    np.testing.assert_allclose(samples.frame_err, expected_err, rtol=1e-5)
    np.testing.assert_array_equal(samples.frame_regime, regimes)
    np.testing.assert_allclose(samples.pit_mae, expected_err.mean(axis=1), rtol=1e-5)
    assert samples.frame_regime.dtype == np.int8 and loss == 0.0

    for view in plan.views:
        mask = np.broadcast_to(view.samples[:, None], regimes.shape)
        if view.regime is not None:
            mask = mask & (regimes == view.regime)
        assert scores[view.name] == pytest.approx(expected_err[mask].mean(), rel=1e-5)
    assert plan.views[5].samples.all()  # an empty `where` pools the whole panel
    assert scores["real"] == pytest.approx(expected_err.mean(), rel=1e-5)
    assert scores["real_macro"] == pytest.approx(0.5 * scores["dregon"] + 0.5 * scores["md2"])
    # Frame weighting differs from averaging per-item means on a regime cell.
    zero_mask = (regimes == REGIMES.index("zero")) & plan.views[2].samples[:, None]
    assert scores["dregon_zero"] != pytest.approx(samples.pit_mae[zero_mask.any(axis=1)].mean())


def test_regimes_on_a_coarser_prediction_grid_keep_physical_slopes():
    """Predictions on half the target's frame rate: the target is resampled
    onto the prediction grid and labelled at THAT grid's rate — a 25 rev/s^2
    spin-up stays a ramp (a mislabelled rate would halve the slope to 12.5)."""
    plan = build_validation_plan(
        _panel_cfg(
            {"md2": {"where": {"dataset": "md2"}}},
            clips=[{"dataset": "md2", "source_type": "nosource", "profile": "ramp"}],
            duration_s=4.0,
        )
    )
    targets = _targets(plan)
    coarse = F.interpolate(targets, size=targets.shape[-1] // 2, mode="linear", align_corners=False)
    scores, _, samples = _run(plan, coarse)

    assert samples.frame_err.shape == (2, coarse.shape[-1])
    interior = samples.frame_regime[:, 10:-10]
    assert (interior == REGIMES.index("ramp")).all()
    assert scores["md2"] == pytest.approx(float(samples.frame_err.mean()), rel=1e-5)


def test_readout_dispatch_rejects_unsupported_tasks():
    with pytest.raises(ValueError, match="multi-validation supports tasks"):
        rps_readout_for("speech_enhancement")
