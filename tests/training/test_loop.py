"""Loop smoke test: tiny synthetic map-style dataset + 2-layer model on the
rps_prediction task, 2 epochs CPU, wandb mocked out entirely.

``wandb.init`` reassigns ``wandb.log``/``wandb.run`` internally when a real
run starts, which silently defeats a plain ``monkeypatch.setattr(wandb,
"log", ...)`` done *before* ``run_training`` calls ``wandb.init`` — so this
mocks the ``wandb`` name binding inside ``training.loop`` itself, replacing
it with a tiny recording stub for the duration of each test.
"""

from __future__ import annotations

import json
import math

import pytest
import torch
from omegaconf import OmegaConf

import training.loop as loop_module
from tests.training.conftest import make_tiny_config
from tests.training.test_artifacts import FakeS3Client
from training.artifacts import ArtifactStore
from training.loop import run_training

pytestmark = pytest.mark.slow


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


def test_run_training_writes_checkpoints_and_produces_finite_loss(tmp_path, monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(loop_module, "wandb", fake_wandb)

    cfg = make_tiny_config(
        results_root=str(tmp_path),
        experiment_name="tiny_loop",
        epochs=2,
        n_train=6,
        n_valid=4,
        batch_size=2,
    )
    result = run_training(cfg)

    run_dir = tmp_path / "tiny_loop"
    assert (run_dir / "best.ckpt").is_file()

    train_losses = [row["train/loss"] for row in fake_wandb.logged if "train/loss" in row]
    assert len(train_losses) == 2  # one per epoch
    assert all(math.isfinite(v) for v in train_losses)

    # The training loss is also evaluated on validation data and logged as
    # val/loss (additive; early stopping still watches `monitor`).
    val_losses = [row["val/loss"] for row in fake_wandb.logged if "val/loss" in row]
    assert len(val_losses) == 2  # one per epoch
    assert all(math.isfinite(v) for v in val_losses)

    assert "best_mse" in result
    assert math.isfinite(result["best_mse"])
    assert math.isfinite(result["final_epoch"])


def test_run_training_refuses_to_overwrite_a_nonempty_run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_module, "wandb", _FakeWandb())

    cfg = make_tiny_config(results_root=str(tmp_path), experiment_name="tiny_loop2", epochs=1)
    run_training(cfg)

    try:
        run_training(cfg)
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected FileExistsError on a second run without resume=true")


def test_run_training_uploads_best_checkpoint_to_injected_artifact_store(tmp_path, monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(loop_module, "wandb", fake_wandb)

    client = FakeS3Client()
    experiment_name = "tiny_loop_artifacts"
    store = ArtifactStore(experiment_name=experiment_name, client=client, enabled=True)

    cfg = make_tiny_config(
        results_root=str(tmp_path),
        experiment_name=experiment_name,
        epochs=2,
        n_train=6,
        n_valid=4,
        batch_size=2,
        artifacts_enabled=True,
    )
    run_training(cfg, artifact_store=store)

    ckpt_key = f"ml-data/artifacts/{experiment_name}/checkpoints/best.ckpt"
    assert ckpt_key in client.objects, (
        f"expected checkpoint at {ckpt_key}, got keys: {list(client.objects)}"
    )

    assert fake_wandb.last_run is not None
    recorded_uri = fake_wandb.last_run.summary.get("r2/best_checkpoint")
    assert recorded_uri == f"r2://{ckpt_key}"


def test_multi_validation_writes_independent_stable_best_checkpoints(tmp_path, monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(loop_module, "wandb", fake_wandb)
    cfg = make_tiny_config(
        results_root=str(tmp_path),
        experiment_name="tiny_multi_validation",
        epochs=1,
        n_train=4,
        batch_size=2,
    )
    valid_spec = {
        "_target_": "tests.training._fixtures.TinyRPSFrameDataset",
        "params": {
            "n_samples": 4,
            "duration_s": 0.5,
            "sample_rate": 16000,
            "seed": 7,
        },
        "iterable": False,
    }
    cfg.validation = OmegaConf.create(
        {
            "enabled": True,
            "every_optimizer_steps": 2,
            "max_optimizer_steps": 2,
            "batch_size": 2,
            "num_workers": 0,
            "datasets": {"real": valid_spec},
            "views": {
                "first_half": {"dataset": "real", "start": 0, "stop": 2},
                "all": {"dataset": "real"},
            },
            "aggregates": {"overall": {"all": 1.0}},
            "primary": ["first_half", "all", "overall"],
            "control": "overall",
            "smoothing_window": 1,
            "min_relative_improvement": 0.01,
            "lr_patience": 2,
            "lr_factor": 0.5,
            "min_lr_reductions": 1,
            "final_patience": 2,
        }
    )

    result = run_training(cfg)
    run_dir = tmp_path / "tiny_multi_validation"
    index = json.loads((run_dir / "best_checkpoints.json").read_text())

    assert result["optimizer_steps"] == 2
    assert set(index) == {"first_half", "all", "overall"}
    assert (run_dir / "best_first_half.ckpt").is_file()
    assert (run_dir / "best_all.ckpt").is_file()
    assert (run_dir / "best_overall.ckpt").is_file()
    assert (run_dir / "best.ckpt").is_file()
    scalar_row = next(row for row in fake_wandb.logged if "optimizer_step" in row)
    assert scalar_row["optimizer_step"] == 2
    assert scalar_row["validation_round"] == 0


def test_iterable_multi_validation_epoch_is_exact_optimizer_step_interval(tmp_path, monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(loop_module, "wandb", fake_wandb)
    cfg = make_tiny_config(
        results_root=str(tmp_path),
        experiment_name="tiny_step_cadence",
        epochs=2,
        n_train=4,
        batch_size=2,
    )
    cfg.data.train._target_ = "tests.training._fixtures.TinyRPSIterableDataset"
    cfg.data.train.iterable = True
    cfg.validation = OmegaConf.create(
        {
            "enabled": True,
            "every_optimizer_steps": 3,
            "max_optimizer_steps": 6,
            "batch_size": 2,
            "num_workers": 0,
            "datasets": {"valid": cfg.data.valid},
            "views": {"all": {"dataset": "valid"}},
            "aggregates": {},
            "primary": ["all"],
            "control": "all",
            "smoothing_window": 1,
            "min_relative_improvement": 0.01,
            "lr_patience": 10,
            "lr_factor": 0.5,
            "min_lr_reductions": 4,
            "final_patience": 20,
        }
    )

    result = run_training(cfg)
    steps = [row["optimizer_step"] for row in fake_wandb.logged if "optimizer_step" in row]

    assert steps == [3, 6]
    assert result["optimizer_steps"] == 6


def test_resume_continues_from_the_saved_epoch_instead_of_restarting(tmp_path, monkeypatch):
    """A relaunch with resume=true must pick up where the last one stopped.

    This is what makes preemptible/short-time-limit queues usable: before, an
    interrupted run restarted from scratch (`resume` only unlocked the run dir),
    so a 1 h wall-clock limit capped total training at 1 h no matter how many
    times it was relaunched.
    """
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(loop_module, "wandb", fake_wandb)

    def cfg_for(epochs: int):
        return make_tiny_config(
            results_root=str(tmp_path),
            experiment_name="tiny_resume",
            epochs=epochs,
            n_train=6,
            n_valid=4,
        )

    run_training(cfg_for(2))
    run_dir = tmp_path / "tiny_resume"
    assert (run_dir / "train_state.pt").is_file()

    first_epochs = [row["epoch"] for row in fake_wandb.logged if "epoch" in row]
    assert first_epochs == [0, 1]

    fake_wandb.logged.clear()
    cfg = cfg_for(4)
    cfg.resume = True
    run_training(cfg)

    # Epochs 2 and 3 only — not 0..3 again.
    assert [row["epoch"] for row in fake_wandb.logged if "epoch" in row] == [2, 3]


def test_resume_on_a_fresh_run_dir_starts_from_zero(tmp_path, monkeypatch):
    """resume=true is the safe default to submit with, so it must be a no-op
    when there is nothing to resume from."""
    monkeypatch.setattr(loop_module, "wandb", (fake_wandb := _FakeWandb()))

    cfg = make_tiny_config(
        results_root=str(tmp_path), experiment_name="tiny_fresh", epochs=2, n_train=6, n_valid=4
    )
    cfg.resume = True
    run_training(cfg)

    assert [row["epoch"] for row in fake_wandb.logged if "epoch" in row] == [0, 1]


def test_resume_after_early_stop_exits_without_training(tmp_path, monkeypatch):
    """A chain of short segments must stop by itself once the run has converged.

    Early stopping is checked at the END of an epoch, so a resumed already-done
    run would otherwise train one full epoch before re-discovering it was done —
    once per queued segment, indefinitely.
    """
    monkeypatch.setattr(loop_module, "wandb", (fake_wandb := _FakeWandb()))

    def cfg_for(epochs: int):
        cfg = make_tiny_config(
            results_root=str(tmp_path),
            experiment_name="tiny_done",
            epochs=epochs,
            n_train=6,
            n_valid=4,
        )
        cfg.patience = 1  # so the tiny run early-stops within its epoch budget
        return cfg

    run_training(cfg_for(3))
    # The tiny model improves every epoch, so drive it to the converged state
    # directly rather than contriving a plateau.
    state_path = tmp_path / "tiny_done" / "train_state.pt"
    state = torch.load(state_path, weights_only=False)
    state["no_improve"] = 1
    torch.save(state, state_path)

    fake_wandb.logged.clear()
    cfg = cfg_for(10)
    cfg.resume = True
    result = run_training(cfg)

    assert [row for row in fake_wandb.logged if "epoch" in row] == []  # no epoch ran
    assert math.isfinite(result["best_mse"])


# ── resuming a preemptible box from R2 ──────────────────────────────────────


def test_fetch_train_state_returns_none_when_r2_is_unreachable(tmp_path, monkeypatch):
    """A run that cannot reach the store starts fresh instead of dying.

    Same defensive contract as the upload side: `ArtifactStore` swallows its own
    failures, so the resume path must too.
    """
    from training import loop as loop_mod

    def boom(_uri):
        raise RuntimeError("no credentials")

    monkeypatch.setattr(loop_mod, "resolve_checkpoint_uri", boom)
    store = loop_mod.ArtifactStore(experiment_name="nope")
    got = loop_mod._fetch_train_state(store, tmp_path / "train_state.pt", tmp_path / "last.ckpt")
    assert got == (None, None)


def test_fetch_train_state_uses_the_experiment_checkpoints_prefix(tmp_path, monkeypatch):
    """The two files must be looked up under <prefix>/<experiment>/checkpoints/."""
    from training import loop as loop_mod

    asked: list[str] = []

    def fake_resolve(uri):
        asked.append(uri)
        p = tmp_path / uri.rsplit("/", 1)[-1]
        p.write_bytes(b"x")
        return str(p)

    monkeypatch.setattr(loop_mod, "resolve_checkpoint_uri", fake_resolve)
    store = loop_mod.ArtifactStore(experiment_name="expt")
    got = loop_mod._fetch_train_state(store, tmp_path / "train_state.pt", tmp_path / "last.ckpt")
    assert all(g is not None for g in got)
    assert asked == [
        f"r2://{store.bucket}/{store.prefix}/expt/checkpoints/train_state.pt",
        f"r2://{store.bucket}/{store.prefix}/expt/checkpoints/last.ckpt",
    ]


def test_load_train_state_without_a_store_is_local_only(tmp_path):
    """No store (uploads disabled) must not attempt any network access."""
    import torch

    from training import loop as loop_mod

    model = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
    scaler = loop_mod.GradScaler(enabled=False)
    got = loop_mod._load_train_state(
        tmp_path,
        model=model,
        optimizer=opt,
        scheduler=sched,
        scaler=scaler,
        device=torch.device("cpu"),
        store=None,
    )
    assert got == (0, None, 0)


def test_load_train_state_refuses_weights_without_bookkeeping(tmp_path):
    """`last.ckpt` but no `train_state.pt` must RAISE, not start fresh.

    The silent fresh start is what destroyed `m3mixv2_scv2`: `best_metric` came
    back None, the epoch loop reads a None best as "improved", and the next
    epoch overwrote `best.ckpt` (its epoch-8 model, val/mse 48.09) with the
    current one (85.4) while resetting `no_improve`, so the run also sailed 69
    epochs past a patience of 20. Losing a run's best weights is worse than
    refusing to start.
    """
    import pytest
    import torch

    from training import loop as loop_mod

    model = torch.nn.Linear(2, 2)
    torch.save(model.state_dict(), tmp_path / "last.ckpt")  # weights, no state
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
    with pytest.raises(RuntimeError, match="no train_state.pt"):
        loop_mod._load_train_state(
            tmp_path,
            model=model,
            optimizer=opt,
            scheduler=sched,
            scaler=loop_mod.GradScaler(enabled=False),
            device=torch.device("cpu"),
            store=None,
        )


def test_load_train_state_starts_fresh_when_nothing_exists(tmp_path):
    """An empty run dir is a genuine first launch and must still start at 0.

    The guard above must not break `resume=true` as the default on preemptible
    queues, which is the setting every cluster job uses.
    """
    import torch

    from training import loop as loop_mod

    model = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
    got = loop_mod._load_train_state(
        tmp_path,
        model=model,
        optimizer=opt,
        scheduler=sched,
        scaler=loop_mod.GradScaler(enabled=False),
        device=torch.device("cpu"),
        store=None,
    )
    assert got == (0, None, 0)


# ── opt-in robust stopping policy (training.stopping) ───────────────────────


def test_robust_policy_persists_its_counters_and_bypasses_raw_patience(tmp_path, monkeypatch):
    """With ``early_stopping.enabled`` the raw ``patience`` guard is inert
    (min_epochs/LR gates decide instead) and the controller's history rides
    along in train_state.pt, so a relaunch continues the SAME window/counters."""
    monkeypatch.setattr(loop_module, "wandb", (fake_wandb := _FakeWandb()))

    def cfg_for(epochs: int):
        cfg = make_tiny_config(
            results_root=str(tmp_path),
            experiment_name="tiny_robust",
            epochs=epochs,
            n_train=6,
            n_valid=4,
            early_stopping={"enabled": True, "median_window": 3, "min_epochs": 100},
        )
        cfg.patience = 1  # would stop the raw policy after one non-improving epoch
        return cfg

    run_training(cfg_for(3))
    state = torch.load(tmp_path / "tiny_robust" / "train_state.pt", weights_only=False)
    es = state["early_stopping"]
    assert len(es["window"]) == 3 and es["stop_reason"] is None
    logged = [row for row in fake_wandb.logged if "epoch" in row]
    assert [row["epoch"] for row in logged] == [0, 1, 2]
    assert all(math.isfinite(row["val/mse_median"]) for row in logged)

    # Force the raw counter past cfg.patience: a robust-policy resume ignores it.
    state["no_improve"] = 5
    torch.save(state, tmp_path / "tiny_robust" / "train_state.pt")
    fake_wandb.logged.clear()
    cfg = cfg_for(5)
    cfg.resume = True
    run_training(cfg)
    assert [row["epoch"] for row in fake_wandb.logged if "epoch" in row] == [3, 4]
    state = torch.load(tmp_path / "tiny_robust" / "train_state.pt", weights_only=False)
    assert len(state["early_stopping"]["window"]) == 3  # rolled, not restarted


def test_nonfinite_train_loss_stops_before_touching_the_last_good_state(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_module, "wandb", _FakeWandb())
    real_train = loop_module._train_one_epoch

    def poisoned(**kw):
        loss, steps = real_train(**kw)
        return (math.nan, steps) if kw["epoch"] == 1 else (loss, steps)

    monkeypatch.setattr(loop_module, "_train_one_epoch", poisoned)
    cfg = make_tiny_config(
        results_root=str(tmp_path),
        experiment_name="tiny_nan",
        epochs=4,
        n_train=6,
        n_valid=4,
        early_stopping={"enabled": True},
    )
    result = run_training(cfg)

    run_dir = tmp_path / "tiny_nan"
    state = torch.load(run_dir / "train_state.pt", weights_only=False)
    assert state["next_epoch"] == 1  # epoch 1 was never checkpointed
    assert result["final_epoch"] == 1.0
    assert math.isfinite(result["best_mse"])
    weights = torch.load(run_dir / "last.ckpt", weights_only=False)
    assert all(torch.isfinite(t).all() for t in weights.values())


def test_deadline_before_the_first_epoch_trains_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_module, "wandb", (fake_wandb := _FakeWandb()))
    cfg = make_tiny_config(
        results_root=str(tmp_path),
        experiment_name="tiny_deadline",
        epochs=2,
        n_train=6,
        n_valid=4,
        early_stopping={"enabled": True, "deadline_unix": 1.0},  # long past
    )
    result = run_training(cfg)
    assert [row for row in fake_wandb.logged if "epoch" in row] == []
    assert result["final_epoch"] == 0.0
