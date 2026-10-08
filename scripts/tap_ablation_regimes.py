"""Four-regime decomposition of a trained salience arm with its harmonic taps
zeroed above an order ``K`` — no retraining, the same frozen real split and
table as ``scripts/_regime_decomp.py``.

Pyramid arms (`hppnet_pyramid`): ``conv_3.weight[k - 1]`` is the tap row of
harmonic ``k``; rows ``k >= K`` are zeroed, sub-harmonic rows and the bias
kept. CQT arms (`HPPNetOrig`): branch ``k`` of ``HarmonicDilatedConv`` is
``trunk.conv_3.convs[k - 2]`` with a ``(1, 3)`` kernel whose columns read
``r/k``, ``r``, ``k*r``; only the ``k*r`` column of branches ``k >= K`` is
zeroed (centre and sub-harmonic columns kept), so the two families lose the
same information.

The ablated checkpoints are written to ``results/<experiment>/<variant>.ckpt``
(where ``zoo.load`` looks first) from the arm's ``best_real_overall`` file,
re-fetched from R2 so a still-training arm is scored on its current best.

    python scripts/tap_ablation_regimes.py --exp real_r4_hppnet_pyrk84_unified \\
        real_r4_hppnet_l2k84nolstm_unified --drop 61 31 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _regime_decomp import (  # noqa: E402
    REGIMES,
    checkpoint_uri,
    decompose,
    recorded_training_score,
    table,
)

from utils.checkpoints import resolve_checkpoint_uri  # noqa: E402

CACHE = REPO_ROOT / ".cache" / "r2_checkpoints"


def fetch_best(experiment: str) -> tuple[Path, dict]:
    """Fresh ``best_real_overall.ckpt`` of ``experiment`` and its state dict."""
    uri = checkpoint_uri(experiment, "best_real_overall")
    if uri.startswith("r2://"):
        cached = CACHE / uri.removeprefix("r2://").replace("/", "__")
        if cached.exists():
            cached.unlink()
    local = Path(resolve_checkpoint_uri(uri, CACHE))
    state = torch.load(local, map_location="cpu", weights_only=True)
    return local, state


def ablate(state: dict, k_from: int) -> tuple[dict, str]:
    """Zero the harmonic taps of orders ``>= k_from``; returns (state, family)."""
    sd = state.get("state_dict", state)
    sd = {k: v.clone() for k, v in sd.items()}
    if "conv_3.weight" in sd:  # pyramid: (J, O, C), rows 0..k_max-1 are k = 1..k_max
        w = sd["conv_3.weight"]
        k_max = w.shape[0] - 7
        for j in range(k_from - 1, k_max):
            w[j].zero_()
        family = "pyramid"
    else:  # CQT: trunk.conv_3.convs.<i>.weight (O, C, 1, 3), branch i is k = i + 2
        keys = [k for k in sd if ".conv_3.convs." in k and k.endswith(".weight")]
        if not keys:
            raise KeyError("neither conv_3.weight nor *.conv_3.convs.*.weight in the state dict")
        for key in keys:
            i = int(key.split(".convs.")[1].split(".")[0])
            if i + 2 >= k_from:
                sd[key][:, :, 0, 2].zero_()
        family = "cqt"
    out = dict(state)
    if "state_dict" in state:
        out["state_dict"] = sd
    else:
        out = sd
    return out, family


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--drop", nargs="+", type=int, default=[61, 31, 16])
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/regime_decomp/tap_ablation.json")
    args = ap.parse_args()

    rows = []
    for experiment in args.exp:
        local, state = fetch_best(experiment)
        rec = recorded_training_score(experiment)
        print(
            f"{experiment}: best_real_overall = round {rec.get('validation_round')} raw {rec.get('raw')} smoothed {rec.get('smoothed')}",
            flush=True,
        )
        variants = [("best_real_overall", None)] + [(f"drop_k{k}", k) for k in args.drop]
        for name, k_from in variants:
            if k_from is not None:
                abl, _family = ablate(state, k_from)
                dest = REPO_ROOT / "results" / experiment / f"{name}.ckpt"
                dest.parent.mkdir(parents=True, exist_ok=True)
                torch.save(abl, dest)
            row = decompose(experiment, name, args.channels, args.limit, args.device)
            row["experiment"] = (
                f"{experiment.replace('real_r4_hppnet_', '').replace('_unified', '')}/{name}"
            )
            row["provenance"] = {
                "source_checkpoint": str(local),
                "selected_checkpoint": rec,
                "k_from": k_from,
            }
            rows.append(row)
            print(
                f"  {row['experiment']:32s} overall per-frame PIT {row['overall']['mae']:.3f}  clip-PIT {row['clip_level_pit_mae']:.3f}",
                flush=True,
            )
    print()
    print(table(rows))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"regimes": list(REGIMES), "rows": rows}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
