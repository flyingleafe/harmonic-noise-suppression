#!/usr/bin/env python
"""Score trained checkpoints on the rp protocol's real panel (`conf/validation/rp_real.yaml`).

Loads each experiment through `zoo.load` (schema-free compose: its frozen model
config, its own loss), builds the panel plan once, runs the training loop's own
`validate_rps` and prints the panel views per checkpoint. For legacy runs this
is the only way to put their checkpoints next to the rp_* grid; note that their
training streams were NOT held out from the panel's recordings, so their
numbers are a floor, not a fair row (docs/experiments/paper-real-protocol.md).

    python scripts/rp_panel_eval.py real_r4_hppnet_pyrlstm_unified:best_real_overall \\
        rp_pyrsep_in_all:best_real_macro_median --out results/rp_panel_eval/legacy.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from tap_ablation_regimes import ablate  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import zoo  # noqa: E402
from data_processing.collate import frame_collate  # noqa: E402
from training.config import build_losses, register_configs  # noqa: E402
from training.validation import build_validation_plan, rps_readout_for, validate_rps  # noqa: E402

VIEWS = (
    "real_macro",
    "real",
    "dregon",
    "md2",
    "real_nosource",
    "real_source",
    "real_zero",
    "real_standby",
    "real_ramp",
    "real_cruise",
)


def _panel(validation: str, via: str):
    """The panel config, composed through experiment ``via`` (the root config's
    mandatory groups need a full experiment behind them)."""
    register_configs()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "conf"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[f"experiment={via}", f"validation={validation}", "~schema"],
        )
    return cfg.validation


def _loss_for(name: str):
    register_configs()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "conf"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[f"experiment={name}", "~schema", "validation=disabled"],
        )
    return build_losses(cfg.loss)


def _windowed(readout, fm, seconds: float, sr: int = 16000):
    """A readout that re-runs the model on ``seconds``-long windows of the clip
    and concatenates the decoded rates: the panel's 8 s clips scored at the
    2 s training-chunk horizon, which is the length-generalisation probe for
    recurrent heads (a stateless head gives the same answer either way)."""
    w = int(seconds * sr)

    def read(model, pred_frame, inputs):
        audio = inputs["mixture"]
        parts = []
        for s in range(0, audio.shape[-1], w):
            sub = dict(inputs)
            sub["mixture"] = audio[..., s : s + w]
            out = fm.codec.call_model(model, sub)
            parts.append(readout(model, fm.codec.to_frame(out, pred_frame), sub))
        return torch.cat(parts, dim=-1)

    return read


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "targets", nargs="+", help="<experiment>[:<checkpoint>], default checkpoint 'best'"
    )
    p.add_argument("--validation", default="rp_real")
    p.add_argument("--via", default="rp_scv2_all", help="experiment the panel is composed through")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out", type=Path, help="JSON file to write {target: scores}")
    p.add_argument(
        "--drop-k",
        nargs="*",
        type=int,
        default=[],
        help="also score each target with its harmonic taps of order >= K zeroed "
        "(scripts/tap_ablation_regimes.py's knock-out; pyramid and CQT families)",
    )
    p.add_argument(
        "--window",
        type=float,
        default=None,
        help="run inference in windows of this many seconds and concatenate "
        "(length-generalisation probe for recurrent heads; 2 = the training chunk)",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panel_cfg = _panel(args.validation, args.via)
    plan = build_validation_plan(panel_cfg)
    loader = DataLoader(
        plan.dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=frame_collate,
    )
    results: dict[str, dict[str, float]] = {}
    header = f"{'target':52s} " + " ".join(f"{v[:10]:>10s}" for v in VIEWS)
    print(header, flush=True)
    for target in args.targets:
        name, _, ckpt = target.partition(":")
        fm = zoo.load(name, ckpt or "best", device=device)
        loss_fn = _loss_for(name).to(device)
        full_state = {k: v.clone() for k, v in fm.model.state_dict().items()}
        for k_from in (None, *args.drop_k):
            label = target if k_from is None else f"{target}/drop_k{k_from}"
            if args.window:
                label += f"/win{args.window:g}s"
            state = full_state if k_from is None else ablate(full_state, k_from)[0]
            fm.model.load_state_dict(state)
            readout = rps_readout_for(fm.task.name)
            if args.window:
                readout = _windowed(readout, fm, args.window)
            scores, loss, _ = validate_rps(
                model=fm.model,
                codec=fm.codec,
                loss_fn=loss_fn,
                valid_loader=loader,
                plan=plan,
                device=device,
                amp=device.type == "cuda",
                amp_dtype=torch.float16,
                readout=readout,
            )
            scores["loss"] = loss
            results[label] = scores
            print(f"{label:52s} " + " ".join(f"{scores[v]:10.2f}" for v in VIEWS), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
