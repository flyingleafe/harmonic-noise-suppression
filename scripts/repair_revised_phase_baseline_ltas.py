"""Targeted repair of missing/affected absolute-LTAS observations.

Reads an existing calibration produced before the one-window LTAS fix,
re-renders the baseline arm only for windows whose scored support is long
enough for one 8192-sample Welch window but was refused by the old
``2 * LTAS_N`` gate, and rewrites ``calibration.json`` with the repaired
LTAS values and updated null-variation tolerances. Composite temperature
(B16) and all non-LTAS measurements are preserved.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

RUNNER_PATH = Path(__file__).resolve().parent / "stochastic_fit_revised_eval.py"
RUNNER_NAME = "_revised_eval_runner_for_repair"
spec = importlib.util.spec_from_file_location(RUNNER_NAME, RUNNER_PATH)
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
sys.modules[RUNNER_NAME] = runner
spec.loader.exec_module(runner)

RE = runner.RE


def die(message: str) -> None:
    raise SystemExit(f"error: {message}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _row_window(row: dict[str, Any]) -> RE.Window:
    w = row["window"]
    return RE.Window(
        str(w["recording"]),
        float(w["start_s"]),
        float(w["duration_s"]),
        regime=str(w["regime"]),
        role=str(w.get("role", "evaluation")),
    )


def _update_baseline_ltas(
    row: dict[str, Any],
    *,
    cohort: dict[str, Any],
    regime_spec: dict[str, Any],
    baseline_model: Any,
) -> dict[str, Any] | None:
    """Compute the baseline LTAS for one measurement row, leaving PIT untouched."""
    window = _row_window(row)
    dataset = str(cohort["dataset"])
    key = RE.assert_raw_reference(str(cohort["scoring_rps_key"]))
    clip = RE.load_window(
        window,
        dataset=dataset,
        version=cohort.get("version"),
        channels=cohort.get("channels"),
        rps_key=key,
    )
    real_all = np.asarray(clip.audio, dtype=np.float64)
    n_mics = int(row["mics"][-1]) + 1 if row["mics"] else real_all.shape[0]
    real = real_all[:n_mics]
    reference = np.asarray(clip.rps, dtype=np.float64)
    support = RE.regime_support(
        window,
        reference,
        regime=str(regime_spec["regime"]),
        min_rps=regime_spec.get("min_rps"),
        max_rps=regime_spec.get("max_rps"),
        sr=clip.sr,
    )
    audio, _, _ = baseline_model.render(reference, n_mics=n_mics, seed=int(row["seed"]))
    audio = np.asarray(audio, dtype=np.float64)[:n_mics]
    return runner.scored_ltas(real, audio, support)


def repair_calibration(man: dict[str, Any], cal: dict[str, Any]) -> dict[str, Any]:
    """Return a calibration record with affected baseline LTAS values repaired."""
    repaired = copy.deepcopy(cal)

    affected_rows: list[tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Any]]] = []

    for cohort in man["cohorts"]:
        name = str(cohort["name"])
        bundles_by_regime = runner.all_families(cohort)
        for regime_spec in cohort["regimes"]:
            regime = str(regime_spec["regime"])
            rres = dict(repaired["cohorts"][name]["regimes"][regime])
            frozen_family = str(rres.get("frozen_family"))
            for row in rres.get("measurements", []):
                baseline = dict(row.get("arms", {})).get("baseline") or {}
                if baseline.get("ltas") is not None:
                    continue
                if baseline.get("ltas_unavailable") is None:
                    continue
                # Recompute baseline LTAS for this row.
                mp = runner.baseline_params(
                    cohort,
                    regime,
                    row["window"]["recording"],
                    family=frozen_family,
                    bundles_by_regime=bundles_by_regime,
                )
                baseline_model = runner.ArmModel(
                    "baseline",
                    "legacy",
                    label=mp.source["label"],
                    legacy=mp,
                    spec={"kind": "export_render"},
                )
                new_ltas = _update_baseline_ltas(
                    row,
                    cohort=cohort,
                    regime_spec=regime_spec,
                    baseline_model=baseline_model,
                )
                row["arms"]["baseline"]["ltas"] = new_ltas
                if new_ltas is not None:
                    row["arms"]["baseline"].pop("ltas_unavailable", None)
                affected_rows.append((name, regime, row, cohort, regime_spec))

    if not affected_rows:
        print("no affected baseline LTAS rows found; nothing to repair")

    # Re-derive the calibration summary from the updated measurements.
    repaired["calibration"] = runner.summarize_prepare(repaired, man)

    # Add the protocol fingerprint and repair provenance.
    protocol_sha = runner.protocol_fingerprint(man)
    repaired.setdefault("manifest", dict(cal.get("manifest") or {}))
    repaired["manifest"]["protocol_sha256"] = protocol_sha
    repaired["repair_provenance"] = dict(
        original_manifest_sha256=cal["manifest"]["sha256"],
        original_calibration_sha256=hashlib.sha256(
            json.dumps(cal, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        n_affected_rows=len(affected_rows),
        affected=[
            dict(cohort=c, regime=r, window=row["window"]["key"])
            for c, r, row, _, _ in affected_rows
        ],
        repair_note=(
            "Re-rendered baseline audio only for rows where the old 2*LTAS_N gate dropped the "
            "absolute LTAS. Recomputed null-variation LTAS tolerances from the updated rows. "
            "Composite temperature (B16) and non-LTAS measurements are preserved."
        ),
    )
    return repaired


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="repair affected baseline LTAS observations")
    ap.add_argument("--manifest", required=True, type=Path, help="frozen manifest path")
    ap.add_argument(
        "--calibration",
        required=True,
        type=Path,
        help="existing calibration.json to repair in place",
    )
    args = ap.parse_args(argv)

    man = runner.load_manifest(args.manifest)
    cal = load_json(args.calibration)

    # Verify the calibration's original manifest is intact.
    cal_manifest = dict(cal.get("manifest") or {})
    orig_path = Path(cal_manifest.get("path", args.manifest))
    orig_sha = cal_manifest.get("sha256")
    if orig_sha is None:
        die("calibration has no recorded manifest SHA")
    if not orig_path.is_file():
        die(f"recorded manifest {orig_path} not found")
    raw = orig_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != orig_sha:
        die("recorded manifest digest does not match the file on disk")

    # Verify frozen inputs and protocol fingerprint before mutating.
    tracker = runner.Tracker(dict(man["scorer"]))
    now = runner.frozen_inputs(man, tracker.record)
    runner.verify_frozen_inputs(now, dict(cal.get("frozen_inputs") or {}))
    runner.verify_protocol_fingerprint(man, cal)

    original_cal_sha = hashlib.sha256(json.dumps(cal, sort_keys=True).encode("utf-8")).hexdigest()

    repaired = repair_calibration(man, cal)

    # Write back to the same path; the caller is responsible for versioning.
    RE.write_json(args.calibration, repaired)

    new_cal_sha = hashlib.sha256(json.dumps(repaired, sort_keys=True).encode("utf-8")).hexdigest()

    report = dict(
        mode="repair_ltas",
        manifest=dict(
            path=man["_path"],
            sha256=man["_digest"],
            protocol_sha256=runner.protocol_fingerprint(man),
        ),
        calibration_path=str(args.calibration),
        original_calibration_sha256=original_cal_sha,
        repaired_calibration_sha256=new_cal_sha,
        repair_provenance=repaired["repair_provenance"],
    )
    RE.write_json(args.calibration.with_name("repair_ltas_report.json"), report)
    print(json.dumps(RE.json_ready(report), indent=1))
    print(f"wrote repaired {args.calibration}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
