"""Print the campaign's round tables as markdown, straight from the artefacts.

``docs/experiments/rps-trajectory-model.md`` quotes one table per round plus a
corpus table.  Nothing in that document may be typed by hand, so this script is
the single place the tables are produced: re-run it after a new round lands and
paste its output over the corresponding table.

Sources (read-only):

* ``results/rps_traj/rounds/round<N>.json`` — one entry per rig with the five
  frozen discrepancy families of the fitted candidate (``new``) and of the
  fitted baseline (``base``), plus the ``verdict`` of
  :func:`experiments.rps_traj.stats.passes`.  The strict count and the
  regression flags are RECOMPUTED here with that same frozen rule, so a table
  cannot disagree with the judge.
* ``results/rps_traj/stats/real/<rig>.flights.json`` — the per-rig corpus
  (flights, recorded and airborne seconds, NaN fraction).
* ``results/rps_traj/stats/real/<rig>.json`` — the real statistics the rounds
  are scored against (used for the corpus table's mean and variance columns).
* ``results/rps_traj/posterior.json`` — the rig posterior's dimension and rigs.

Usage:
    python scripts/_rps_traj_rounds_table.py            # corpus + every round
    python scripts/_rps_traj_rounds_table.py --rounds 3 # one round only
    python scripts/_rps_traj_rounds_table.py --no-corpus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from experiments.rps_traj.stats import FAMILIES, MIN_STRICT_FAMILIES, passes  # noqa: E402

ROUNDS_DIR = REPO / "results/rps_traj/rounds"
REAL_DIR = REPO / "results/rps_traj/stats/real"
POSTERIOR = REPO / "results/rps_traj/posterior.json"

#: Column header per frozen family, and the unit each discrepancy carries.
FAMILY_HEADERS = {
    "overall_mean": "overall_mean (rev/s)",
    "rotor_mean": "rotor_mean (rev/s)",
    "rotor_var": "rotor_var (log)",
    "acf": "acf",
    "xcorr": "xcorr",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _cell(base: float, new: float) -> str:
    """``base -> new``, with the direction of the change marked."""
    mark = "" if new <= base else " !"
    return f"{base:.4f} -> {new:.4f}{mark}"


def round_table(payload: dict) -> str:
    """The per-rig markdown table of one round JSON."""
    head = ["rig"] + [FAMILY_HEADERS[f] for f in FAMILIES] + ["strict", "verdict"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for rig in payload["rigs"]:
        base, new = rig["base"], rig["new"]
        verdict = passes(new, base)[0]
        strict = sum(1 for f in FAMILIES if new[f] < base[f])
        if verdict != bool(rig["verdict"]):
            raise SystemExit(
                f"{rig['rig']}: stored verdict {rig['verdict']} != frozen rule {verdict}"
            )
        cells = [rig["rig"]] + [_cell(base[f], new[f]) for f in FAMILIES]
        cells += [f"{strict}/5", "**PASS**" if verdict else "FAIL"]
        lines.append("| " + " | ".join(cells) + " |")
    n_pass = sum(1 for rig in payload["rigs"] if rig["verdict"])
    lines.append("")
    lines.append(
        f"{n_pass}/{len(payload['rigs'])} PASS "
        f"(no family may regress and >= {MIN_STRICT_FAMILIES} must strictly improve; "
        f"`!` marks a regression). Estimator: `n_rep` {payload['n_rep']}, "
        f"seed {payload['seed']}, {payload['rate_hz']:.0f} Hz grid."
    )
    return "\n".join(lines)


def corpus_table() -> str:
    """Flights, seconds and real first/second moments per rig."""
    head = [
        "rig",
        "flights",
        "recorded (s)",
        "airborne (s)",
        "segments",
        "overall mean (rev/s)",
        "rotor var ((rev/s)^2)",
        "NaN",
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    total = {"flights": 0, "rec": 0.0, "air": 0.0, "seg": 0}
    for path in sorted(REAL_DIR.glob("*.flights.json")):
        book = _load(path)
        stats = _load(path.with_name(path.name.replace(".flights", "")))
        flights = book["flights"]
        rec = sum(f["duration_s"] for f in flights)
        air = sum(f["airborne_s"] for f in flights)
        seg = sum(len(f["segments"]) for f in flights)
        nan = max(f["nan_fraction"] for f in flights)
        var = " ".join(f"{v:.0f}" if v >= 100 else f"{v:.1f}" for v in stats["rotor_var"])
        lines.append(
            "| "
            + " | ".join(
                [
                    book["rig"],
                    str(book["n_flights"]),
                    f"{rec:.0f}",
                    f"{air:.0f}",
                    str(seg),
                    f"{stats['overall_mean']:.1f}",
                    var,
                    "0" if nan == 0 else f"{nan:.5f}",
                ]
            )
            + " |"
        )
        total["flights"] += book["n_flights"]
        total["rec"] += rec
        total["air"] += air
        total["seg"] += seg
    lines.append(
        f"| **total** | **{total['flights']}** | **{total['rec']:.0f}** | "
        f"**{total['air']:.0f}** | **{total['seg']}** | | | |"
    )
    return "\n".join(lines)


def posterior_line() -> str:
    post = _load(POSTERIOR)
    return (
        f"`{POSTERIOR.relative_to(REPO)}`: {post['kind']}, {post['dim']} coordinates, "
        f"fitted on {len(post['rigs'])} rigs ({', '.join(sorted(post['rigs']))})."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--rounds",
        type=int,
        nargs="*",
        help="round numbers to print; default every round JSON present",
    )
    ap.add_argument("--no-corpus", action="store_true", help="skip the corpus table")
    args = ap.parse_args()

    if not args.no_corpus:
        print("## Corpus\n")
        print(corpus_table())
        print()
        print(posterior_line())
        print()

    available = sorted(int(p.stem.removeprefix("round")) for p in ROUNDS_DIR.glob("round*.json"))
    wanted = args.rounds if args.rounds else available
    for n in wanted:
        path = ROUNDS_DIR / f"round{n}.json"
        if not path.exists():
            raise SystemExit(f"{path} does not exist (present: {available})")
        print(f"## Round {n}\n")
        print(round_table(_load(path)))
        print()


if __name__ == "__main__":
    main()
