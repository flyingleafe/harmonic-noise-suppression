#!/usr/bin/env python3
"""Publish the fitted rotor-speed trajectory model as the ``rps-traj-fits`` tree.

Copies the campaign's shipped per-rig fits and the GLOBAL fit (the rig
posterior) into a small raw dload tree the training streams read
(``rps.kind: fitted_traj``, see
:mod:`data_processing.trajectory_model.source`)::

    data/rps-traj-fits/
      manifest.json      what this tree is, and where it came from
      fits/<rig>.json    one NewFit per rig, byte-for-byte as fitted
      posterior.json     the 32-dim diagonal Gaussian over all rigs
      README.md          what each file is

The fit JSONs are copied VERBATIM: they are the campaign's artefacts, and a
stream that reads a reformatted copy is no longer reading the fit.  The script
is idempotent — running it twice writes the same bytes, including
``manifest.json``'s ``created`` — so re-publishing an unchanged tree does not
churn a new dload version.

Then commit and pin the tree::

    python scripts/rps_traj_publish_fits.py --results results/rps_traj \\
        --out data/rps-traj-fits
    dload commit rps-traj-fits --from data/rps-traj-fits
    dload pin rps-traj-fits && git add dload.lock
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if (_HERE.parent / "src").is_dir() else Path.cwd().resolve()
sys.path.insert(0, str(_ROOT / "src"))

from data_processing.trajectory_model import N_PARAMS, RATE_HZ  # noqa: E402

#: The model form the shipped parameters describe: round-4 dynamics (an OU plus
#: a CAR(2) per control mode, a per-rotor measurement OU) with round-5's split
#: per-flight offset (a common level plus a per-rotor part).
MODEL_VERSION = "new-32p-r4dyn-r5offset"
ROUND = "4/5"

#: What a consumer must know about the rate: the fits are STATED on the
#: campaign's 100 Hz analysis grid, but every component is discretised exactly
#: at whatever rate the sampler is asked for, so a stream may draw at 200 Hz (or
#: any other rate) without changing the process it draws from.
SAMPLER_RATE_NOTE = (
    f"Parameters are continuous-time; the fits were scored on the {RATE_HZ:.0f} Hz "
    "analysis grid (michaels at 25 Hz, see fit_rate_hz in each fit). The sampler "
    "discretises exactly at the requested rate, so sampling at 200 Hz is the same "
    "process, not an interpolation of a 100 Hz one."
)

README = """# rps-traj-fits — the fitted rotor-speed trajectory model

This tree is the output of the trajectory-model campaign
(`docs/experiments/rps-trajectory-model.md`). The training streams read it
through `rps.kind: fitted_traj`
(`src/data_processing/trajectory_model/source.py`).

## Files

- `manifest.json` — the campaign round, the git commit that published the tree,
  the rigs it covers, the model version and a note on sample rates.
- `fits/<rig>.json` — one fitted rig, exactly as the fit wrote it: the 32
  parameters (`params`), the warm-up idle level (`idle_rps`), the rig's real
  airborne speed extremes (`rps_min`, `rps_max`, which the sampler clamps to)
  and the fit's own provenance (`nll`, `n_iter`, `n_flights`, ...).
- `posterior.json` — the GLOBAL fit: a diagonal Gaussian over 32 scale-free
  rig coordinates, fitted to all the rigs above. A draw from it is a FRESH
  drone, not one of the rigs. In a policy it is the reserved rig name
  `posterior`.

## What a rig is

Rotor speeds in rev/s, in mixer rotor order `[RFront, LFront, LBack, RBack]`.
Each rig's airborne process is `mu + delta_flight + M R(theta) v(t) + e(t)`,
with one OU plus one damped oscillator per control mode, one measurement OU per
rotor, and one level offset per flight. The model, the sampler and the exact
discretisation live in `src/data_processing/trajectory_model/`.

## Rounds

The dynamics come from round 4 (the exact Kalman likelihood); the per-flight
offset was re-estimated in round 5 and split into a common part (`s_c`) and a
per-rotor part (`s_r`). `mu`, `s_c` and `s_r` are estimated from the airborne
means, not fitted by the likelihood, which never sees a block's level.
"""


def git_commit(root: Path) -> str:
    """The publishing checkout's HEAD, with ``-dirty`` when it has changes."""
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{head}-dirty" if dirty else head


def build(results: Path, out: Path) -> dict[str, Any]:
    """Write the tree and return the manifest payload."""
    fit_dir = results / "fits" / "new"
    fit_paths = sorted(fit_dir.glob("*.json"))
    if not fit_paths:
        raise SystemExit(f"no fits under {fit_dir}")
    posterior_path = results / "posterior.json"
    if not posterior_path.is_file():
        raise SystemExit(f"no global fit at {posterior_path}")

    rigs = [path.stem for path in fit_paths]
    for path in fit_paths:
        named = json.loads(path.read_text(encoding="utf-8")).get("rig")
        if named != path.stem:
            raise SystemExit(f"{path} is the fit of rig {named!r}, not {path.stem!r}")
    posterior = json.loads(posterior_path.read_text(encoding="utf-8"))
    post_rigs = sorted(str(rig) for rig in posterior.get("rigs", ()))
    if post_rigs != sorted(rigs):
        raise SystemExit(
            "the global fit covers different rigs than the fits directory:\n"
            f"  {posterior_path.name}: {post_rigs}\n"
            f"  {fit_dir}: {sorted(rigs)}\n"
            "re-run scripts/rps_traj_posterior.py so the two agree before publishing"
        )

    payload: dict[str, Any] = {
        "round": ROUND,
        "git_commit": git_commit(_ROOT),
        "rigs": sorted(rigs),
        "created": "",
        "model_version": MODEL_VERSION,
        "sampler_rate_note": SAMPLER_RATE_NOTE,
    }
    # Idempotent: an unchanged tree keeps its original timestamp, so
    # re-publishing does not push a new dload version for a new date.
    manifest_path = out / "manifest.json"
    created = datetime.now(UTC).replace(microsecond=0).isoformat()
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if {k: v for k, v in previous.items() if k != "created"} == {
            k: v for k, v in payload.items() if k != "created"
        }:
            created = str(previous.get("created", created))
    payload["created"] = created

    (out / "fits").mkdir(parents=True, exist_ok=True)
    for path in fit_paths:
        (out / "fits" / path.name).write_bytes(path.read_bytes())
    (out / "posterior.json").write_bytes(posterior_path.read_bytes())
    (out / "README.md").write_text(README, encoding="utf-8")
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--results", type=Path, default=Path("results/rps_traj"))
    ap.add_argument("--out", type=Path, default=Path("data/rps-traj-fits"))
    args = ap.parse_args()

    payload = build(Path(args.results), Path(args.out))
    print(f"{args.out}: round {payload['round']}, model {payload['model_version']}")
    print(f"  {len(payload['rigs'])} rigs: {', '.join(payload['rigs'])}")
    print(f"  {N_PARAMS} parameters per rig; global fit over all of them")
    print(f"  created {payload['created']}, git {payload['git_commit']}")
    print(f"\nnext: dload commit rps-traj-fits --from {args.out} && dload pin rps-traj-fits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
