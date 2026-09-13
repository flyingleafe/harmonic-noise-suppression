#!/usr/bin/env python3
"""Compose the revised-phase v2 calibration from v1 plus repaired DREGON windows.

The repair is metadata-only for the split: C2 candidate fits are not refit and
unchanged held-out measurements are copied from the v1 calibration. Only the
three DREGON windows invalidated by the v1 missing guard are remeasured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import stochastic_fit_revised_eval as E

RE = E.RE
AFFECTED = {
    "updown_nosource_room2": 1511903578.348311,
    "rectangle_nosource_room2": 1511905725.952559,
    "spinning_nosource_room2": 1511905200.978012,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _windows_by_recording(windows: list[Any]) -> dict[str, Any]:
    return {w.recording: w for w in windows}


def _copy_json(obj: Any) -> Any:
    return json.loads(json.dumps(obj))


def _stage_family_exports(source: dict[str, Any]) -> list[dict[str, Any]]:
    """Copy v1-pinned family exports from sibling omnirun worktrees when absent."""
    staged: list[dict[str, Any]] = []
    exports = dict(dict(source.get("frozen_inputs") or {}).get("exports") or {})
    siblings = list(Path.cwd().parent.glob("*/results/S2"))
    for rec in exports.values():
        path = Path(str(rec.get("path", "")))
        if path.name == "." or path.is_file():
            continue
        expected = str(rec.get("sha256") or "")
        for root in siblings:
            candidate = root / path.name
            if candidate.is_file() and (not expected or _sha256(candidate) == expected):
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(candidate, path)
                staged.append(dict(path=str(path), source=str(candidate), sha256=_sha256(path)))
                break
    missing = [
        str(rec.get("path"))
        for rec in exports.values()
        if rec.get("path") and not Path(str(rec["path"])).is_file()
    ]
    if missing:
        raise SystemExit(f"missing v1-pinned family exports after sibling staging: {missing}")
    return staged


def _replace_measurements(
    old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]], affected: set[str]
) -> tuple[list[dict[str, Any]], int]:
    kept = [r for r in old_rows if str(r["window"]["recording"]) not in affected]
    removed = len(old_rows) - len(kept)
    order = {"free-flight_nosource_room2": 0, "hovering_nosource_room2": 1, **{r: i + 2 for i, r in enumerate(AFFECTED)}}
    out = kept + new_rows
    out.sort(key=lambda r: (order.get(str(r["window"]["recording"]), 999), float(r["seed"])))
    return out, removed


def repair(args: argparse.Namespace) -> None:
    source_path = Path(args.source_calibration)
    if not source_path.is_file() and args.source_r2_uri:
        E._download_s3_uri(str(args.source_r2_uri), source_path)
    source_sha = _sha256(source_path)
    if args.expected_source_sha and source_sha != args.expected_source_sha:
        raise SystemExit(
            f"{source_path}: sha256 {source_sha} != expected {args.expected_source_sha}"
        )
    source = json.loads(source_path.read_text())
    man = E.load_manifest(Path(args.manifest))
    staged_family_exports = _stage_family_exports(source)
    guard_seconds = E.training_guard_seconds(man)
    if guard_seconds != 1.024:
        raise SystemExit(f"v2 repair is frozen to guard 1.024, got {guard_seconds}")

    tracker = E.Tracker(dict(man["scorer"]))
    record = _copy_json(source)
    record["manifest"] = dict(
        path=man["_path"], sha256=man["_digest"], protocol_sha256=E.protocol_fingerprint(man)
    )
    record["import_provenance"] = RE.import_provenance(expected_root=Path.cwd())
    record["scorer"] = tracker.record
    record["frozen_inputs"] = E.frozen_inputs(man, tracker.record)
    record["gates"] = man["gates"]
    record["null_variation_rule"] = dict(man["null_variation"])

    dregon = next(c for c in man["cohorts"] if c["name"] == "dregon_room2_cruise")
    regime_spec = next(r for r in dregon["regimes"] if r["regime"] == "cruise")
    bundles_by_regime = E.all_families(dregon)
    bundles = bundles_by_regime["cruise"]
    calibration = [w for b in bundles.values() for w in b.calibration_windows()]
    windows, reports = E.resolve_windows(
        dregon, regime_spec, calibration, guard_seconds=guard_seconds
    )
    got = _windows_by_recording(windows)
    missing = sorted(set(AFFECTED) - set(got))
    if missing:
        raise SystemExit(f"v2 explicit windows missing after guarded validation: {missing}")
    for rid, start in AFFECTED.items():
        if got[rid].start_s != start or got[rid].duration_s != 4.0:
            raise SystemExit(f"{rid}: resolved {got[rid].as_dict()}, expected start {start} +4s")

    affected_windows = [got[rid] for rid in AFFECTED]
    seeds = [int(s) for s in man["null_variation"]["seeds"]]
    family = str(
        source["cohorts"]["dregon_room2_cruise"]["regimes"]["cruise"].get("frozen_family")
        or "refined"
    )
    baseline_spec = dict(E.require(dict(man["arms"]), "baseline", "arms"))

    def model_for(arm: str, spec: dict[str, Any], w: Any) -> Any:
        if arm == "real":
            return E.ArmModel("real", "real", label="real audio", spec=spec)
        mp = E.baseline_params(
            dregon, "cruise", w.recording, family=family, bundles_by_regime=bundles_by_regime
        )
        return E.ArmModel(arm, "legacy", label=mp.source["label"], legacy=mp, spec=spec)

    new_rows = E.measure_arm_set(
        dregon,
        regime_spec,
        affected_windows,
        tracker=tracker,
        arms={"real": dict(kind="real"), "baseline": baseline_spec},
        model_for=model_for,
        seeds=seeds,
        n_mics=args.mics,
        tag="dregon_room2_cruise:v2_repair",
    )

    dres = record["cohorts"]["dregon_room2_cruise"]["regimes"]["cruise"]
    old_rows = list(dres["measurements"])
    dres["measurements"], removed = _replace_measurements(old_rows, new_rows, set(AFFECTED))
    dres["support_reports"] = reports
    dres["windows"] = [w.as_dict() for w in windows]
    record["cohorts"]["dregon_room2_cruise"]["plan"] = E.cohort_plan(
        dregon, guard_seconds=guard_seconds
    )
    record["calibration"] = E.summarize_prepare(record, man)
    record["repair_provenance"] = dict(
        protocol="revised-phase guarded split repair v2",
        source_calibration=dict(path=str(source_path), sha256=source_sha),
        source_manifest=source.get("manifest"),
        prior_repair_provenance=source.get("repair_provenance"),
        requirements=(
            "metadata-only repair before candidate outcomes; guard=1.024 in hashed observation; "
            "explicit DREGON windows; recompute only affected three DREGON real/baseline rows "
            "for seeds 2001..2004; preserve two DREGON rows, all Michael rows, family selection, "
            "and B16 composite temperatures"
        ),
        training_guard_seconds=guard_seconds,
        affected_recordings=sorted(AFFECTED),
        removed_measurement_rows=removed,
        added_measurement_rows=len(new_rows),
        staged_family_exports=staged_family_exports,
        preserved_dregon_recordings=["free-flight_nosource_room2", "hovering_nosource_room2"],
        preserved_michaels=True,
        composite_temperature={
            name: record["cohorts"][name].get("composite_temperature")
            for name in sorted(record["cohorts"])
        },
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "calibration.json"
    RE.write_json(path, record)
    print(json.dumps(RE.json_ready(record["calibration"]["cohorts"]["dregon_room2_cruise"]["cruise"]), indent=1))
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--source-calibration", required=True, type=Path)
    ap.add_argument("--source-r2-uri", default="")
    ap.add_argument("--expected-source-sha", default="")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mics", type=int, default=None)
    args = ap.parse_args()
    repair(args)

if __name__ == "__main__":
    main()
