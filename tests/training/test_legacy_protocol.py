"""``train.py``/``eval.py`` refuse plateau-protocol experiments with a named reason."""

from __future__ import annotations

from pathlib import Path

from training.config import legacy_protocol_error


def _conf(tmp_path: Path, files: dict[str, str]) -> Path:
    exp = tmp_path / "experiment"
    exp.mkdir()
    for name, body in files.items():
        (exp / f"{name}.yaml").write_text(body)
    return tmp_path


def test_clean_experiment_is_runnable(tmp_path: Path) -> None:
    conf = _conf(tmp_path, {"rp_ok": "experiment_name: rp_ok\nepochs: 200\noptim:\n  lr: 0.001\n"})
    assert legacy_protocol_error(["experiment=rp_ok"], conf) is None
    assert legacy_protocol_error(["model=foo"], conf) is None  # no experiment override at all


def test_removed_keys_and_retired_groups_are_named(tmp_path: Path) -> None:
    conf = _conf(
        tmp_path,
        {
            "old": (
                "defaults:\n  - override /validation: rps_unified\n  - _self_\n"
                "patience: 20\noptim:\n  monitor: rps_mae\n  monitor_mode: min\n"
            )
        },
    )
    message = legacy_protocol_error(["experiment=old"], conf)
    assert message is not None
    for token in ("legacy experiment", "patience", "optim.monitor", "validation=rps_unified"):
        assert token in message


def test_inherited_legacy_key_is_found_through_the_parent_chain(tmp_path: Path) -> None:
    conf = _conf(
        tmp_path,
        {
            "parent": "optim:\n  factor: 0.5\n",
            "child": "defaults:\n  - parent\n  - _self_\nexperiment_name: child\n",
        },
    )
    message = legacy_protocol_error(["+experiment=child"], conf)
    assert message is not None and "optim.factor" in message
