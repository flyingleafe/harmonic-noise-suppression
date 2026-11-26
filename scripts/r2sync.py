#!/usr/bin/env python3
"""Push/pull a ``results/`` subtree to/from its R2 mirror ``r2://ml-data/artifacts/<relpath>``.

The explicit result channel of remote jobs — omnirun collects no outputs
(``omnirun.toml``; docs/data-and-artifacts.md § "Job results")::

    python scripts/r2sync.py push results/<name>[/<sub>] ...  # end of a job command
    python scripts/r2sync.py pull <name>[/<sub>] ...          # laptop -> results/<name>[/<sub>]
    python scripts/r2sync.py pull <exp> --weights             # checkpoints too

``train.py`` and ``eval.py`` publish their run records themselves; this is for
everything else. Weights (``*.ckpt``/``*.pt``/...) and ``wandb/`` dirs are never
pushed; pull skips weights unless ``--weights`` and files whose size+mtime match.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Import this tree's src/, not whatever the shared .venv's editable install points at.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.artifacts import mirror_store, publish_results  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--results-root", default="results", help="local results root")
    sub = ap.add_subparsers(dest="cmd", required=True)
    push = sub.add_parser("push", help="upload files/dirs below the results root")
    push.add_argument("paths", nargs="+")
    pull = sub.add_parser("pull", help="download relpaths below the results root")
    pull.add_argument("relpaths", nargs="+", help="e.g. rp_scv2_all or se_eval/<file>.csv")
    pull.add_argument("--weights", action="store_true", help="also fetch checkpoints")
    args = ap.parse_args()

    root = Path(args.results_root)
    if args.cmd == "push":
        failed = False
        for path in args.paths:
            uris = publish_results(path, results_root=root)
            print(f"push {path}: {len(uris)} file(s)", flush=True)
            failed |= not uris
        return int(failed)
    for relpath in args.relpaths:
        relpath = relpath.removeprefix(f"{root.as_posix().rstrip('/')}/")
        store, subkey = mirror_store(relpath)
        try:
            got = store.download_tree(subkey, root / relpath, weights=args.weights)
        except (FileNotFoundError, RuntimeError) as exc:
            raise SystemExit(f"pull {relpath}: {exc}") from None
        print(f"pull {relpath}: {len(got)} file(s) updated -> {root / relpath}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
