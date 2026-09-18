"""Build the noise-model-v2 round-1 SUPPORTS: the windows every fit and gate reads.

No measurement, no fit — this script only resolves
:mod:`experiments.noise_model.supports` specs against the published datasets,
writes one ``.npz`` per support, and records what it chose. Everything a later
reader needs to trust a number (which segment, which carrier, which label key,
whether the bench stationarity rule passed) travels in ``index.json``.

Five sets:

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
``michaels-all``
    The R2 flight pool: FLY125's ONE standby window, its ramp CONTEXT window
    and the same eight cruise windows, all 8 s, FLY125 only. R1 fitted cruise
    alone (carriers 68.2-97.9 rev/s, a 1.44x span) and its speed-law exponents
    were unidentified there; this set spans 36-98 rev/s so they are fitted
    rather than extrapolated. See :func:`supports.set_michaels_all` for how
    each regime's windows are chosen and why the recording offers only one of
    each outside cruise.
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
import os
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


REPO_ROOT = Path(__file__).resolve().parents[1]


def require_credentials() -> None:
    """Fail with the fix instead of a bare ``botocore.NoCredentialsError``.

    Every support reads a published frames dataset over ``dload``, which takes
    its R2 credentials from the AWS environment chain. A remote job that has
    ``.env`` in its worktree but never sourced it dies several gigabytes into a
    stream with ``Unable to locate credentials`` and no hint of the cause, so
    the check runs before the first byte. ``.env`` is consulted the same way
    :mod:`data_processing.streams` consults it (shell-provided wins).
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is a project dependency
        pass
    else:
        load_dotenv(REPO_ROOT / ".env", override=False)
    from utils.checkpoints import R2_ENV_VARS

    missing = [name for name in R2_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise SystemExit(
            f"R2 credentials missing ({', '.join(missing)}) — "
            "`set -a; source .env; set +a` before running, or export "
            "R2_ACCOUNT_ID / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY"
        )


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


def _r1_carriers() -> dict[str, list[float]]:
    """The round-1 index's carriers, for the shift column (empty if absent)."""
    path = Path("results/noise_v2/rounds/round1/supports/index.json")
    if not path.exists():
        return {}
    idx = json.loads(path.read_text())
    out: dict[str, list[float]] = {}
    for s in idx.get("sets", {}).values():
        for r in s.get("supports", []):
            out[str(r["name"])] = [float(v) for v in (r.get("carriers_rev_s") or [])]
    return out


def _bench_table(rows: list[dict[str, Any]]) -> list[str]:
    r1 = _r1_carriers()
    out = [
        "| support | pass | segment (s) | dur (s) | longest +-1 Hz (s) | level deficit (dB) "
        "| in-window margin (dB) | order | survey (rev/s) | carrier (rev/s) | shift vs survey "
        "| shift vs R1 | residual std (Hz) | mics | bins |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    fmt = lambda vs, spec: ", ".join(format(v, spec) for v in vs)  # noqa: E731
    for r in rows:
        seg = f"{r['segment'][0]:.2f}-{r['segment'][1]:.2f}"
        carriers = [float(v) for v in r["carriers_rev_s"]]
        old = r1.get(str(r["name"]))
        vs_r1 = (
            fmt([c - o for c, o in zip(carriers, old)], "+.4f")
            if old and len(old) == len(carriers)
            else "-"
        )
        deficit = r.get("level_deficit_db")
        margin = r.get("line_margin_db")
        longest = r.get("longest_inside_s")
        out.append(
            f"| `{r['name']}` | {'PASS' if r['stationary_pass'] else 'FAIL'} | {seg} | "
            f"{r['duration_s']:.3f} | {longest:.2f} | "
            f"{'-' if deficit is None else format(deficit, '.2f')} | "
            f"{'-' if margin is None else fmt(margin, '.2f')} | {r.get('orders')} | "
            f"{fmt([float(v) for v in (r.get('survey_rev_s') or [])], '.4f')} | "
            f"{fmt(carriers, '.4f')} | "
            f"{fmt([float(v) for v in (r.get('carrier_shift_rev_s') or [])], '+.4f')} | {vs_r1} | "
            f"{fmt([float(v) for v in (r.get('residual_std_hz') or [])], '.3f')} | "
            f"{r['n_mics']} | {r['n_bins']} |"
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
        "# Noise model v2 — supports",
        "",
        "Every support is one observation window at 16 kHz in the frozen periodogram",
        "convention `|rfft(x*w)|^2 / sum(w^2)` (periodic Hann, one-sided,",
        "`revised_eval.window_periodogram`): bench = ONE frame over the whole segment,",
        "flight = NFFT 2048 / hop 512. Numbers below are read from `index.json`, which is",
        "written by `scripts/noise_v2_supports.py build`.",
        "",
        "BENCH STATIONARITY RULE REVISION 2 (2026-09-17). A candidate sample must now also",
        "lie where the band-limited (30 Hz - 7.9 kHz) median level over a sliding",
        "`BENCH_MIN_SEGMENT_S` window is within `BENCH_LEVEL_TOL_DB` = 6 dB of the",
        "recording's loudest such window, the accepted window is certified by the line",
        "margin measured INSIDE it (not over the recording), the carrier is refined on the",
        "window and FROZEN (the fit has no carrier parameter), and revision 1's wide-band",
        "residual test is gone. Revision 1 scored frequency-residual stationarity alone,",
        "which silence satisfies perfectly, and put 12 of the 21 DREGON bench windows",
        "10-33 dB below the loudest window of their own recording — see",
        "`results/noise_v2/rounds/round1/bench_diag/findings.md`. `level_deficit_db`,",
        "`line_margin_db` and `carrier_recording_rev_s` in `index.json` are the new",
        "per-recording evidence.",
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
                f"The rev-2 rule (order {SUP.BENCH_ORDER_RANGE[0]}-{SUP.BENCH_ORDER_RANGE[1]}, "
                f"residual +-{SUP.BENCH_RESIDUAL_TOL_HZ:g} Hz averaged over "
                f"{SUP.BENCH_RESIDUAL_SMOOTH_S:g} s, level gate within "
                f"{SUP.BENCH_LEVEL_TOL_DB:g} dB, in-window margin >= "
                f"{SUP.BENCH_LINE_MARGIN_DB:g} dB, minimum {SUP.BENCH_MIN_SEGMENT_S:g} s): "
                f"**{n_pass} of {len(rows)} recordings pass**."
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
                    "Failing recordings (window kept and cached, flagged `FAIL`; under rule "
                    "rev 2 a bench recording fails on the LINE MARGIN measured inside its own "
                    f"window, against {SUP.BENCH_LINE_MARGIN_DB:g} dB): "
                    + "; ".join(
                        f"`{r['name']}` margin "
                        + ", ".join(f"{v:.2f}" for v in (r.get("line_margin_db") or []))
                        + " dB (recording "
                        + ", ".join(f"{v:.2f}" for v in (r.get("line_margin_recording_db") or []))
                        + f" dB), window {r['duration_s']:.2f} s, level deficit "
                        f"{r['level_deficit_db']:.2f} dB"
                        for r in fails
                    )
                )
            deficits = [
                r["level_deficit_db"] for r in rows if r.get("level_deficit_db") is not None
            ]
            margins = [v for r in rows for v in (r.get("line_margin_db") or [])]
            if deficits and margins:
                out.append("")
                out.append(
                    f"Level gate (rev 2): every window sits {min(deficits):.2f}-"
                    f"{max(deficits):.2f} dB below the loudest {SUP.BENCH_MIN_SEGMENT_S:g} s "
                    "window of its own recording, against 0.2-32.8 dB under rev 1 "
                    "(`results/noise_v2/rounds/round1/bench_diag/census.json`). In-window line "
                    f"margins run {min(margins):.2f}-{max(margins):.2f} dB."
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
        "hundreds of MB); `index.json` and this file are. Every row carries its `spec`, "
        "so a consumer rebuilds one support with `supports.load_support(<spec>)` and a "
        "whole set with"
    )
    out.append("")
    out.append("```bash")
    out.append("set -a; . ./.env; set +a          # R2 credentials for the dload stream")
    out.append("PYTHONPATH=src python scripts/noise_v2_supports.py build --set <set>")
    out.append("```")
    out.append("")
    out.append(
        "Sourcing the credentials is not optional: a remote job's worktree is a bare git "
        "checkout with NO `.env` in it (omnirun ships the secrets to `$JOB_DIR/.env` and "
        "sources them in its own bootstrap), so a build whose environment carries no R2 "
        "keys now stops on the credential check instead of streaming into a botocore "
        "`NoCredentialsError`."
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

    require_credentials()
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
