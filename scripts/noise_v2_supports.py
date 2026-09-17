"""Build the noise-model-v2 round-1 SUPPORTS: the windows every fit and gate reads.

No measurement, no fit — this script only resolves
:mod:`experiments.noise_model.supports` specs against the published datasets,
writes one ``.npz`` per support, and records what it chose. Everything a later
reader needs to trust a number (which segment, which carrier, which label key,
whether the bench stationarity rule passed) travels in ``index.json``.

Four sets:

``dregon-bench``
    The 20 DREGON single-motor cells ``Motor{1-4}_{50,60,70,80,90}`` plus the
    quad validation recording ``allMotors_70``. These publish NO rotor track,
    so the carrier comes from the corpus survey manifest and the stationary
    span is measured from the audio by the approved +-1 Hz residual rule
    (:func:`supports.stationary_segment`).
``bench-points``
    All 135 published ``noise-v2-bench-points`` frames, each already stationary
    by the manifest's own tolerance rule, taken whole.
``michaels-cruise``
    FLY125's eight cruise windows (the first 8 s of each 16 s window of
    ``results/S2/cruise_8clip_refined.json``, so the eight are disjoint and the
    material is the legacy export's) plus the five frozen FLY124 evaluation
    supports.
``dregon-floor``
    The five frozen room-2 scoring windows (4 s, all mics) and, for the floor
    fit, five DISJOINT 8 s segments of the SAME recordings — starting after the
    scored window plus a 2 s guard, so no floor parameter is fitted on scored
    material.

    python scripts/noise_v2_supports.py build --set dregon-bench
    python scripts/noise_v2_supports.py build --set bench-points --out results/noise_v2/rounds/round1/supports
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.noise_model import supports as SUP

TOPIC = "supports"
OUT_DEFAULT = SUP.CACHE_DIR


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception as exc:  # pragma: no cover - provenance never aborts a run
        return f"unavailable: {exc}"


def build(set_name: str, *, out_dir: Path, limit: int | None = None) -> dict[str, Any]:
    """Resolve and cache every support of ``set_name``; return the index payload."""
    specs = SUP.support_set(set_name)
    if limit is not None:
        specs = specs[: int(limit)]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    t0 = time.time()
    for support in SUP.iter_supports(specs):
        path = SUP.save_support(support, out_dir=out_dir)
        row = SUP.index_row(support)
        row["npz"] = path.name
        row["npz_bytes"] = int(path.stat().st_size)
        rows.append(row)
        print(
            f"  {row['name']}: {row['n_mics']}x{row['n_frames']}x{row['n_bins']} "
            f"{row['duration_s']:.3f} s carriers {row['carriers_rev_s']} "
            f"pass={row['stationary_pass']}",
            flush=True,
        )
    missing = sorted({s.name for s in specs} - {r["name"] for r in rows})
    for name in missing:
        failures.append(dict(name=name, error="not resolved by iter_supports"))
    return dict(
        schema="noise-v2-supports/1",
        set=set_name,
        n_specs=len(specs),
        n_built=len(rows),
        supports=rows,
        failures=failures,
        rule=dict(
            sr=SUP.SR,
            obs_n_fft=SUP.OBS_N_FFT,
            obs_hop=SUP.OBS_HOP,
            bench_order_range=list(SUP.BENCH_ORDER_RANGE),
            bench_demod_band_hz=SUP.BENCH_DEMOD_BAND_HZ,
            bench_residual_smooth_s=SUP.BENCH_RESIDUAL_SMOOTH_S,
            bench_residual_tol_hz=SUP.BENCH_RESIDUAL_TOL_HZ,
            bench_min_segment_s=SUP.BENCH_MIN_SEGMENT_S,
            bench_edge_s=SUP.BENCH_EDGE_S,
        ),
        provenance=dict(
            git_head=git_head(),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            wall_seconds=round(time.time() - t0, 3),
            command=" ".join(sys.argv),
        ),
    )


def merge_index(out_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """One ``index.json`` for every set, keyed by set name.

    The four sets run as four jobs, so the index is merged, never overwritten:
    a set that was not rebuilt keeps the block it had.
    """
    path = out_dir / "index.json"
    index: dict[str, Any] = dict(schema="noise-v2-supports-index/1", sets={})
    if path.exists():
        try:
            prior = json.loads(path.read_text())
            if isinstance(prior.get("sets"), dict):
                index["sets"] = prior["sets"]
        except json.JSONDecodeError:
            pass
    index["sets"][payload["set"]] = payload
    index["n_supports"] = sum(int(b.get("n_built", 0)) for b in index["sets"].values())
    index["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    path.write_text(json.dumps(index, indent=1) + "\n")
    return index


def _bench_table(rows: list[dict[str, Any]]) -> list[str]:
    out = [
        "| support | pass | segment (s) | dur (s) | longest +-1 Hz (s) | order | survey (rev/s) "
        "| carrier (rev/s) | shift | residual std (Hz) | mics | bins |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        seg = f"{r['segment'][0]:.2f}-{r['segment'][1]:.2f}"
        survey = ", ".join(f"{v:.4f}" for v in (r.get("survey_rev_s") or []))
        carrier = ", ".join(f"{v:.4f}" for v in r["carriers_rev_s"])
        shift = ", ".join(f"{v:+.4f}" for v in (r.get("carrier_shift_rev_s") or []))
        std = ", ".join(f"{v:.3f}" for v in (r.get("residual_std_hz") or []))
        longest = r.get("longest_inside_s")
        out.append(
            f"| `{r['name']}` | {'PASS' if r['stationary_pass'] else 'FAIL'} | {seg} | "
            f"{r['duration_s']:.3f} | {longest:.2f} | {r.get('orders')} | {survey} | {carrier} | "
            f"{shift} | {std} | {r['n_mics']} | {r['n_bins']} |"
        )
    return out


def _flight_table(rows: list[dict[str, Any]]) -> list[str]:
    out = [
        "| support | recording | start (s) | dur (s) | label | mics | frames | rotors "
        "| carrier mean (rev/s) | carrier span (rev/s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        mean = ", ".join(f"{v:.2f}" for v in r["carriers_rev_s"])
        span = (
            f"{min(r['carrier_min_rev_s']):.2f}-{max(r['carrier_max_rev_s']):.2f}"
            if r["carrier_min_rev_s"]
            else "-"
        )
        out.append(
            f"| `{r['name']}` | {r['recording_id']} | {r['segment'][0]:.3f} | "
            f"{r['duration_s']:.3f} | `{r['rps_key']}` | {r['n_mics']} | {r['n_frames']} | "
            f"{r['n_rotors']} | {mean} | {span} |"
        )
    return out


def _point_table(rows: list[dict[str, Any]]) -> list[str]:
    out = [
        "| support | rig | dur (s) | mics | bins | rotors | carrier (rev/s) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        carrier = ", ".join(f"{v:.3f}" for v in r["carriers_rev_s"])
        out.append(
            f"| `{r['name']}` | {r.get('rig')} | {r['duration_s']:.3f} | {r['n_mics']} | "
            f"{r['n_bins']} | {r['n_rotors']} | {carrier} |"
        )
    return out


def findings(index: dict[str, Any]) -> str:
    """One findings file over every set present in the index."""
    out: list[str] = [
        "# Noise model v2, round 1 — supports",
        "",
        "Every support is one observation window at 16 kHz in the frozen periodogram",
        "convention `|rfft(x*w)|^2 / sum(w^2)` (periodic Hann, one-sided,",
        "`revised_eval.window_periodogram`): bench = ONE frame over the whole segment,",
        "flight = NFFT 2048 / hop 512. Numbers below are read from `index.json`, which is",
        "written by `scripts/noise_v2_supports.py build`.",
        "",
    ]
    sets = index.get("sets", {})
    out.append(f"Sets present: {', '.join(sorted(sets)) or 'none'}. ")
    out.append(f"Total supports cached: {index.get('n_supports', 0)}.")
    out.append("")

    for name in sorted(sets):
        block = sets[name]
        rows = block["supports"]
        out.append(f"## {name}")
        out.append("")
        out.append(
            f"{block['n_built']} of {block['n_specs']} specs built "
            f"({block['provenance']['wall_seconds']:.1f} s, git "
            f"`{block['provenance']['git_head'][:12]}`)."
        )
        if block["failures"]:
            out.append("")
            if not rows:
                out.append(
                    f"FAILURES: all {len(block['failures'])} requested supports are unavailable: "
                    f"{block['failures'][0]['error']}. Individual support names are in `index.json`."
                )
            else:
                out.append(
                    "FAILURES: "
                    + "; ".join(f"`{f['name']}` {f['error']}" for f in block["failures"])
                )
        out.append("")

        if not rows:
            out.append(
                "No support material was resolved. The requested count is recorded above; "
                "segments, carrier estimates, and residual statistics are unavailable."
            )
            out.append("")
            continue

        if name == "dregon-bench":
            n_pass = sum(1 for r in rows if r["stationary_pass"])
            lengths = sorted(r["longest_inside_s"] for r in rows)
            passed = [r["duration_s"] for r in rows if r["stationary_pass"]]
            out.append(
                f"The +-1 Hz rule (order {SUP.BENCH_ORDER_RANGE[0]}-{SUP.BENCH_ORDER_RANGE[1]}, "
                f"residual averaged over {SUP.BENCH_RESIDUAL_SMOOTH_S:g} s, minimum "
                f"{SUP.BENCH_MIN_SEGMENT_S:g} s): **{n_pass} of {len(rows)} recordings pass**."
            )
            out.append("")
            if passed:
                out.append(
                    f"Passing segments run {min(passed):.2f}-{max(passed):.2f} s "
                    f"(median {sorted(passed)[len(passed) // 2]:.2f} s, "
                    f"total {sum(passed):.1f} s of stationary bench material)."
                )
            fails = [r for r in rows if not r["stationary_pass"]]
            if fails:
                out.append("")
                out.append(
                    "Failing recordings (kept with the most stationary "
                    f"{SUP.BENCH_MIN_SEGMENT_S:g} s window, flagged `FAIL`): "
                    + "; ".join(
                        f"`{r['name']}` longest {r['longest_inside_s']:.2f} s, "
                        f"worst residual {r['residual_max_abs_hz']:.2f} Hz"
                        for r in fails
                    )
                )
            out.append("")
            out.append(
                "Longest +-1 Hz spans over all 21 recordings: "
                f"min {lengths[0]:.2f} s, median {lengths[len(lengths) // 2]:.2f} s, "
                f"max {lengths[-1]:.2f} s."
            )
            shifts = [
                abs(v) for r in rows for v in (r.get("carrier_shift_rev_s") or []) if v is not None
            ]
            if shifts:
                out.append("")
                out.append(
                    "Carrier refinement (survey speed -> demodulated line): "
                    f"|shift| up to {max(shifts):.4f} rev/s, median "
                    f"{sorted(shifts)[len(shifts) // 2]:.4f} rev/s. At order 70 a 0.01 rev/s "
                    "error already displaces the line by 0.7 Hz, most of the tolerance, which "
                    "is why the carrier is refined before the rule is applied."
                )
            out.append("")
            out.extend(_bench_table(rows))
        elif name == "bench-points":
            mics = sorted({r["n_mics"] for r in rows})
            durs = sorted(r["duration_s"] for r in rows)
            rigs = sorted({str(r.get("rig")) for r in rows})
            out.append(
                f"{len(rows)} published points, {len(rigs)} rigs, channel counts {mics}, "
                f"durations {durs[0]:.1f}-{durs[-1]:.1f} s "
                f"(median {durs[len(durs) // 2]:.1f} s). Each is taken whole: the publishing "
                "derivation already cut it to its stationary span (<= 30 s)."
            )
            out.append("")
            out.extend(_point_table(rows))
        else:
            out.append("")
            out.extend(_flight_table(rows))
        out.append("")

    out.append("## Bytes")
    out.append("")
    for name in sorted(sets):
        total = sum(int(r.get("npz_bytes", 0)) for r in sets[name]["supports"])
        out.append(f"* `{name}`: {total / 1e6:.1f} MB of `.npz` (power as float32)")
    out.append("")
    out.append(
        "The `.npz` caches are NOT committed (`results/**` is gitignored and they are "
        "hundreds of MB); `index.json` and this file are. Any consumer regenerates a cache "
        "with `python scripts/noise_v2_supports.py build --set <set>`."
    )
    out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="build the noise-model-v2 round-1 supports")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="resolve a set and cache its supports")
    b.add_argument("--set", dest="set_name", required=True, choices=sorted(SUP.SUPPORT_SETS))
    b.add_argument("--out", type=Path, default=OUT_DEFAULT)
    b.add_argument("--limit", type=int, default=None, help="first N specs only (smoke runs)")

    s = sub.add_parser("list", help="print a set's spec strings")
    s.add_argument("--set", dest="set_name", required=True, choices=sorted(SUP.SUPPORT_SETS))

    args = ap.parse_args()
    if args.cmd == "list":
        for spec in SUP.support_set(args.set_name):
            print(spec.text)
        return

    out_dir = Path(args.out)
    print(f"building set {args.set_name} into {out_dir}", flush=True)
    payload = build(args.set_name, out_dir=out_dir, limit=args.limit)
    index = merge_index(out_dir, payload)
    (out_dir / "findings.md").write_text(findings(index))
    print(
        f"built {payload['n_built']}/{payload['n_specs']} supports; "
        f"wrote {out_dir / 'index.json'} and {out_dir / 'findings.md'}"
    )
    if payload["failures"]:
        raise SystemExit(f"unresolved supports: {payload['failures']}")


if __name__ == "__main__":
    main()
