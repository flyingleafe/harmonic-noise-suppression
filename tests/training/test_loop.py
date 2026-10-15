"""Loop smoke tests: tiny synthetic datasets + a 2-layer model on the
rps_prediction task, CPU, wandb mocked out entirely.

``wandb.init`` reassigns ``wandb.log``/``wandb.run`` internally when a real
run starts, which silently defeats a plain ``monkeypatch.setattr(wandb,
"log", ...)`` done *before* ``run_training`` calls ``wandb.init`` — so this
mocks the ``wandb`` name binding inside ``training.loop`` itself, replacing
it with a tiny recording stub for the duration of each test.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import training.loop as loop_module
from tests.training.conftest import make_tiny_config
from tests.training.test_artifacts import FakeS3Client
from training.artifacts import ArtifactStore
from training.loop import build_lr_scheduler, run_training

pytestmark = pytest.mark.slow

PANEL_CLIPS = [
    {"dataset": "dregon", "source_type": "nosource", "profile": "cruise"},
    {"dataset": "md2", "source_type": "mixed", "profile": "zero_then_cruise"},
]
PRIMARY = ["dregon", "md2", "md2_cruise", "real_macro"]


class _FakeRun:
    id = "fake-run-id"

    def __init__(self) -> None:
        self.summary: dict = {}


class _FakeWandb:
    def __init__(self) -> None:
        self.logged: list[dict] = []
        self.last_run: _FakeRun | None = None

    def init(self, *args, **kwargs):
        run = _FakeRun()
        self.last_run = run
        return run

    def log(self, data, *args, **kwargs):
        self.logged.append(dict(data))

    def finish(self, *args, **kwargs):
        pass


def _panel_validation(**overrides):
    values = {
        "enabled": True,
        "every_optimizer_steps": 2,
        "max_optimizer_steps": 4,
        "batch_size": 2,
        "num_workers": 0,
        "datasets": {
            "real": {
                "_target_": "tests.training._fixtures.TinyPanelFrameDataset",
                "params": {"clips": PANEL_CLIPS, "n_channels": 2, "duration_s": 2.0},
                "iterable": False,
            }
        },
        "views": {
            "dregon": {"where": {"dataset": "dregon"}},
            "md2": {"where": {"dataset": "md2"}},
            "md2_cruise": {"where": {"dataset": "md2"}, "regime": "cruise"},
        },
        "aggregates": {"real_macro": {"dregon": 0.5, "md2": 0.5}},
        "primary": PRIMARY,
        "control": "real_macro",
        "smoothing_window": 1,
        "min_rounds": 0,
        "min_relative_improvement": 0.01,
        "final_patience": 50,
    }
    values.update(overrides)
    return OmegaConf.create(values)


def _panel_config(tmp_path, name: str, *, epochs: int = 2, **validation):
    cfg = make_tiny_config(
        results_root=str(tmp_path), experiment_name=name, epochs=epochs, n_train=4, batch_size=2
    )
    cfg.validation = _panel_validation(**validation)
    return cfg


def test_run_training_writes_last_checkpoint_and_logs_finite_losses(tmp_path, monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(loop_module, "wandb", fake_wandb)

    cfg = make_tiny_config(results_root=str(tmp_path), experiment_name="tiny_loop", epochs=2)
    result = run_training(cfg)

    run_dir = tmp_path / "tiny_loop"
    assert (run_dir / "last.ckpt").is_file()
    assert not list(run_dir.glob("best*.ckpt"))  # the legacy path selects nothing
    rows = [row for row in fake_wandb.logged if "train/loss" in row]
    assert len(rows) == 2
    assert all(math.isfinite(row["train/loss"]) and math.isfinite(row["val/loss"]) for row in rows)
    assert result["final_epoch"] == 1.0


def test_run_training_refuses_to_overwrite_a_nonempty_run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_module, "wandb", _FakeWandb())

    cfg = make_tiny_config(results_root=str(tmp_path), experiment_name="tiny_loop2", epochs=1)
    run_training(cfg)
    with pytest.raises(FileExistsError):
        run_training(cfg)


def test_run_training_uploads_last_checkpoint_to_injected_artifact_store(tmp_path, monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(loop_module, "wandb", fake_wandb)
    client = FakeS3Client()
    name = "tiny_loop_artifacts"
    store = ArtifactStore(experiment_name=name, client=client, enabled=True)

    cfg = make_tiny_config(
        results_root=str(tmp_path), experiment_name=name, epochs=2, artifacts_enabled=True
    )
    run_training(cfg, artifact_store=store)

    key = f"ml-data/artifacts/{name}/checkpoints/last.ckpt"
    assert key in client.objects
    assert fake_wandb.last_run is not None
    assert fake_wandb.last_run.summary.get("r2/last_checkpoint") == f"r2://{key}"


def test_panel_run_writes_dual_bests_history_and_per_sample_arrays(tmp_path, monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(loop_module, "wandb", fake_wandb)
    result = run_training(_panel_config(tmp_path, "tiny_panel", epochs=1, max_optimizer_steps=2))

    run_dir = tmp_path / "tiny_panel"
    index = json.loads((run_dir / "best_checkpoints.json").read_text())
    assert set(index) == {f"{score}/{kind}" for score in PRIMARY for kind in ("median", "raw")}
    entry = index["real_macro/median"]
    assert set(entry) == {"checkpoint", "validation_round", "optimizer_step", "raw", "median"}
    assert (entry["checkpoint"], entry["validation_round"], entry["optimizer_step"]) == (
        "best_real_macro_median.ckpt",
        0,
        2,
    )
    # Coincident bests are hard links to that round's last.ckpt: one inode.
    assert os.path.samefile(run_dir / "best_real_macro_median.ckpt", run_dir / "last.ckpt")
    assert os.path.samefile(
        run_dir / "best_real_macro_raw.ckpt", run_dir / "best_md2_cruise_raw.ckpt"
    )
    assert not (run_dir / "best.ckpt").exists()
    assert result["optimizer_steps"] == 2.0
    assert result["best_real_macro_raw"] == pytest.approx(entry["raw"])

    with np.load(run_dir / "val_samples" / "round_0000.npz") as arrays:
        assert set(arrays.files) == {
            "pit_mae",
            "frame_err",
            "frame_regime",
            "clip_index",
            "channel",
        }
        assert arrays["frame_err"].shape == (4, 63)
        assert arrays["frame_regime"].dtype == np.int8
        assert arrays["clip_index"].tolist() == [0, 0, 1, 1]
        assert arrays["channel"].tolist() == [0, 1, 0, 1]
        np.testing.assert_allclose(arrays["pit_mae"], arrays["frame_err"].mean(axis=1), rtol=1e-6)

    history = [
        json.loads(line) for line in (run_dir / "validation_history.jsonl").read_text().splitlines()
    ]
    row = history[0]
    assert row["optimizer_step"] == 2 and row["validation_round"] == 0 and "lr" in row
    assert row["val/real_macro"] == pytest.approx(0.5 * row["val/dregon"] + 0.5 * row["val/md2"])
    assert not [key for key in row if key.endswith(("_clips", "_clip_mean", "_clip_std"))]
    assert set(row["validation/improved_raw"]) == set(PRIMARY)
    assert not [key for key in row if key.endswith(("_log", "_log_median"))]


def test_median_and_raw_bests_follow_their_own_rounds(tmp_path, monkeypatch):
    """Scripted control series 5.0, 4.0, 4.5, 4.4 with a 2-round median:
    raw best at round 1 (4.0); medians 4.5 (r1), 4.25 (r2), 4.45 (r3), so the
    median best is round 2. Each best links to ITS round's weights, and the
    round's per-sample arrays are uploaded only when some best moved."""
    monkeypatch.setattr(loop_module, "wandb", _FakeWandb())
    series = iter([5.0, 4.0, 4.5, 4.4])
    real_validate = loop_module.validate_rps

    def scripted(**kwargs):
        scores, loss, samples = real_validate(**kwargs)
        value = next(series)
        return {name: value for name in scores}, loss, samples

    monkeypatch.setattr(loop_module, "validate_rps", scripted)
    client = FakeS3Client()
    store = ArtifactStore(experiment_name="tiny_dual", client=client, enabled=True)
    cfg = _panel_config(
        tmp_path, "tiny_dual", epochs=4, smoothing_window=2, max_optimizer_steps=100
    )
    cfg.artifacts.enabled = True
    cfg.checkpoint_every = 1  # step<N>.ckpt per round: the per-round weights to compare to
    run_training(cfg, artifact_store=store)

    run_dir = tmp_path / "tiny_dual"
    index = json.loads((run_dir / "best_checkpoints.json").read_text())
    assert (
        index["real_macro/raw"]["validation_round"] == 1 and index["real_macro/raw"]["raw"] == 4.0
    )
    assert index["real_macro/median"]["validation_round"] == 2
    assert index["real_macro/median"]["median"] == pytest.approx(4.25)
    assert os.path.samefile(run_dir / "best_real_macro_raw.ckpt", run_dir / "step4.ckpt")
    assert os.path.samefile(run_dir / "best_real_macro_median.ckpt", run_dir / "step6.ckpt")
    uploaded = {key.rsplit("/", 1)[-1] for key in client.objects if "/val_samples/" in key}
    assert uploaded == {"round_0000.npz", "round_0001.npz", "round_0002.npz"}
    assert len(list((run_dir / "val_samples").glob("round_*.npz"))) == 4
    prefix = "ml-data/artifacts/tiny_dual/checkpoints"
    assert (
        client.objects[f"{prefix}/best_real_macro_median.ckpt"]
        == (run_dir / "step6.ckpt").read_bytes()
    )


def test_iterable_panel_round_is_exact_optimizer_step_interval(tmp_path, monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(loop_module, "wandb", fake_wandb)
    real_train = loop_module._train_one_epoch
    batch_counts: list[int] = []

    def one_scaler_skip(**kwargs):
        batch_counts.append(kwargs["n_batches"])
        loss, completed = real_train(**kwargs)
        return loss, completed - 1 if len(batch_counts) == 1 else completed

    monkeypatch.setattr(loop_module, "_train_one_epoch", one_scaler_skip)
    cfg = _panel_config(
        tmp_path, "tiny_step_cadence", epochs=3, every_optimizer_steps=3, max_optimizer_steps=6
    )
    cfg.data.train._target_ = "tests.training._fixtures.TinyRPSIterableDataset"
    cfg.data.train.iterable = True

    result = run_training(cfg)
    steps = [row["optimizer_step"] for row in fake_wandb.logged if "optimizer_step" in row]

    assert steps == [3, 6]  # the step budget censors before the round cap
    assert result["optimizer_steps"] == 6
    assert batch_counts == [3, 1, 3]


def test_saturated_control_stops_the_run_and_a_resume_trains_nothing(tmp_path, monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(loop_module, "wandb", fake_wandb)
    real_validate = loop_module.validate_rps

    def flat(**kwargs):
        scores, loss, samples = real_validate(**kwargs)
        return {name: 3.0 for name in scores}, loss, samples

    monkeypatch.setattr(loop_module, "validate_rps", flat)

    def cfg_for(epochs: int):
        return _panel_config(
            tmp_path, "tiny_saturated", epochs=epochs, final_patience=2, max_optimizer_steps=100
        )

    run_training(cfg_for(10))
    rounds = [row["validation_round"] for row in fake_wandb.logged if "validation_round" in row]
    assert rounds == [0, 1, 2]  # best at round 0, two stale rounds, stop
    assert fake_wandb.last_run is not None
    assert fake_wandb.last_run.summary["stop_reason"] == "saturation"

    fake_wandb.logged.clear()
    cfg = cfg_for(10)
    cfg.resume = True
    run_training(cfg)
    assert [row for row in fake_wandb.logged if "epoch" in row] == []


def test_resume_continues_rounds_and_the_lr_schedule(tmp_path, monkeypatch):
    """A relaunch with resume=true picks up where the last one stopped — the
    round counter AND the cosine: the LR sequence equals an uninterrupted run's."""
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(loop_module, "wandb", fake_wandb)

    def cfg_for(name: str, epochs: int):
        return make_tiny_config(
            results_root=str(tmp_path),
            experiment_name=name,
            epochs=epochs,
            warmup_steps=4,
            total_steps=12,
        )

    run_training(cfg_for("tiny_straight", 4))
    straight = [row["lr"] for row in fake_wandb.logged if "lr" in row]
    fake_wandb.logged.clear()

    run_training(cfg_for("tiny_resume", 2))
    cfg = cfg_for("tiny_resume", 4)
    cfg.resume = True
    run_training(cfg)
    rows = [row for row in fake_wandb.logged if "lr" in row]

    assert [row["epoch"] for row in rows] == [0, 1, 2, 3]  # epochs 2, 3 only after resume
    assert [row["lr"] for row in rows] == pytest.approx(straight)
    assert len(set(straight)) == 4  # the schedule actually moved


def test_resume_on_a_fresh_run_dir_starts_from_zero(tmp_path, monkeypatch):
    """resume=true is the safe default to submit with, so it must be a no-op
    when there is nothing to resume from."""
    monkeypatch.setattr(loop_module, "wandb", (fake_wandb := _FakeWandb()))

    cfg = make_tiny_config(results_root=str(tmp_path), experiment_name="tiny_fresh", epochs=2)
    cfg.resume = True
    run_training(cfg)

    assert [row["epoch"] for row in fake_wandb.logged if "epoch" in row] == [0, 1]


def test_resume_refuses_state_from_the_retired_plateau_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_module, "wandb", _FakeWandb())
    cfg = make_tiny_config(results_root=str(tmp_path), experiment_name="tiny_retired", epochs=1)
    run_training(cfg)
    state_path = tmp_path / "tiny_retired" / "train_state.pt"
    state = torch.load(state_path, weights_only=False)
    torch.save({**state, "best_metric": 1.0, "no_improve": 0}, state_path)

    cfg = make_tiny_config(results_root=str(tmp_path), experiment_name="tiny_retired", epochs=2)
    cfg.resume = True
    with pytest.raises(RuntimeError, match="retired"):
        run_training(cfg)


# ── resuming a preemptible box from R2 ──────────────────────────────────────


def _bookkeeping(model: torch.nn.Module):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = build_lr_scheduler(
        optimizer, OmegaConf.create({"warmup_steps": 0, "total_steps": 10, "min_lr_frac": 0.01})
    )
    return optimizer, scheduler


def test_fetch_train_state_returns_none_when_r2_is_unreachable(tmp_path, monkeypatch):
    """A run that cannot reach the store starts fresh instead of dying.

    Same defensive contract as the upload side: `ArtifactStore` swallows its own
    failures, so the resume path must too.
    """

    def boom(_uri):
        raise RuntimeError("no credentials")

    monkeypatch.setattr(loop_module, "resolve_checkpoint_uri", boom)
    store = loop_module.ArtifactStore(experiment_name="nope")
    got = loop_module._fetch_train_state(store, tmp_path / "train_state.pt", tmp_path / "last.ckpt")
    assert got == (None, None)


def test_fetch_train_state_uses_the_experiment_checkpoints_prefix(tmp_path, monkeypatch):
    """The two files must be looked up under <prefix>/<experiment>/checkpoints/."""
    asked: list[str] = []

    def fake_resolve(uri):
        asked.append(uri)
        p = tmp_path / uri.rsplit("/", 1)[-1]
        p.write_bytes(b"x")
        return str(p)

    monkeypatch.setattr(loop_module, "resolve_checkpoint_uri", fake_resolve)
    store = loop_module.ArtifactStore(experiment_name="expt")
    got = loop_module._fetch_train_state(store, tmp_path / "train_state.pt", tmp_path / "last.ckpt")
    assert all(g is not None for g in got)
    assert asked == [
        f"r2://{store.bucket}/{store.prefix}/expt/checkpoints/train_state.pt",
        f"r2://{store.bucket}/{store.prefix}/expt/checkpoints/last.ckpt",
    ]


def test_load_train_state_refuses_weights_without_bookkeeping(tmp_path):
    """`last.ckpt` but no `train_state.pt` must RAISE, not start fresh.

    The silent fresh start is what destroyed `m3mixv2_scv2`: a fresh selector
    reads its first round as "improved" and the next round overwrote the best
    checkpoint (its epoch-8 model, val/mse 48.09) with the current one (85.4).
    Losing a run's best weights is worse than refusing to start.
    """
    model = torch.nn.Linear(2, 2)
    torch.save(model.state_dict(), tmp_path / "last.ckpt")  # weights, no state
    optimizer, scheduler = _bookkeeping(model)
    with pytest.raises(RuntimeError, match="no train_state.pt"):
        loop_module._load_train_state(
            tmp_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=loop_module.GradScaler(enabled=False),
            device=torch.device("cpu"),
            store=None,
        )


def test_load_train_state_starts_fresh_when_nothing_exists(tmp_path):
    """An empty run dir is a genuine first launch and must still start at 0,
    without network access when no store is given."""
    model = torch.nn.Linear(2, 2)
    optimizer, scheduler = _bookkeeping(model)
    got = loop_module._load_train_state(
        tmp_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=loop_module.GradScaler(enabled=False),
        device=torch.device("cpu"),
        store=None,
    )
    assert got == (0, 0)


def test_nonfinite_train_loss_stops_before_touching_the_last_good_state(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_module, "wandb", _FakeWandb())
    real_train = loop_module._train_one_epoch

    def poisoned(**kw):
        loss, steps = real_train(**kw)
        return (math.nan, steps) if kw["epoch"] == 1 else (loss, steps)

    monkeypatch.setattr(loop_module, "_train_one_epoch", poisoned)
    cfg = make_tiny_config(results_root=str(tmp_path), experiment_name="tiny_nan", epochs=4)
    result = run_training(cfg)

    run_dir = tmp_path / "tiny_nan"
    state = torch.load(run_dir / "train_state.pt", weights_only=False)
    assert state["next_epoch"] == 1  # epoch 1 was never checkpointed
    assert result["final_epoch"] == 1.0
    weights = torch.load(run_dir / "last.ckpt", weights_only=False)
    assert all(torch.isfinite(t).all() for t in weights.values())
