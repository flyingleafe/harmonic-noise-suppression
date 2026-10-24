#!/usr/bin/env python3
"""Noise model v4: a battery of real-vs-synthetic similarity measures for the
DREGON flight arms — REPORT ONLY, nothing is optimised::

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/noise_v4_similarity_report.py \\
        --out results/noise_v4/similarity/

The measures (:mod:`experiments.noise_model.similarity`): line/floor
log-spectral distance per octave band, order-tracked line levels, frame band
level distributions and modulation spectra, spatial level spread and
coherence, Itakura-Saito divergence (linear-bin and octave-balanced) and the
spectro-temporal Fréchet / co-modulation measures, next to the old mic-0 band LTAS
(``revised_eval.ltas_deviation_db``) for reference. Two window sets, scored
separately: the five official 4 s DREGON cruise scoring windows
(``gates.DREGON_CRUISE_SUPPORTS``) and the five 8 s tuning segments
(``dregon_flight_v4.pool_specs()``). Every arm is rendered on the window's real
``motors_command`` trajectory (lines aligned with the data), render seed 2001,
all eight mics. Each arm's SEED FLOOR (``<arm> seed floor``) is the same
payload on the same trajectory, seed 2001 as "real" against seed 2002: the
realisation noise of every measure at that window length and trajectory.
Outputs ``report.json`` (every per-window, per-band, per-mic number) and
``report.md`` (one row per arm, its seed floor under it).

``--arm NAME=PATH`` (repeatable) adds a payload arm after the built-in ones
(:data:`ARMS`; a built-in name is replaced); ``--arm NAME`` names a built-in.
``--only`` restricts the report to the ``--arm`` arms, in their order (each with
its seed-floor row), e.g. to gate a new payload against nv3r4::

    python scripts/noise_v4_similarity_report.py --out results/noise_v4/similarity/gate/ \\
        --only --arm nv3r4 --arm v4_gust=results/noise_v4/flight/wind/<stem>__gust.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SEED, FLOOR_SEED = 2001, 2002
N_MICS = 8
CWIND = "results/noise_v4/flight/wind/dregon_room2__flight_v4__dregon__kappa0__r5__s2__scwind.json"
#: Payload arms; a ``(base, point)`` pair is ``dregon_lawsearch_v4.edit_payload(base, point)``.
ARMS: dict[str, Any] = {
    "nv3r4": "results/noise_v3/fits_r4/dregon_room2_floor__flight_v3.json",
    "v4_stage_static": "results/noise_v4/flight/dregon_room2__flight_v4__dregon__kappa0__r5__s2.json",
    "v4_cwind": CWIND,
    # law-search stage-2 p13: best PIT with LTAS <= nv3r4 on the tuning segments
    "v4_laws_A": (
        CWIND,
        dict(
            g_a=2.2059561743269582,
            alpha_a=0.22937750636516413,
            g_m=-1.9331727082069055,
            alpha_m=-0.09835463762283325,
            g_f=-5.189213604329466,
            b0=4.95384576990166,
        ),
    ),
    # best PIT overall
    "v4_laws_B": (
        CWIND,
        dict(
            g_a=5.963658365992765,
            alpha_a=0.24529653462262896,
            g_m=-9.230984067528118,
            alpha_m=0.5191844180226326,
            g_f=-3.968099142258164,
            b0=6.167430491520295,
        ),
    ),
}
FLOOR_SUFFIX = " seed floor"
SETS = ("scoring", "tuning")


def arm_rows(arms: dict[str, Any]) -> tuple[str, ...]:
    """The report's rows: each arm, then its seed floor."""
    return tuple(row for arm in arms for row in (arm, f"{arm}{FLOOR_SUFFIX}"))


def select_arms(specs: list[str], only: bool) -> dict[str, Any]:
    """The arms of a run: :data:`ARMS` (none with ``only``) then each ``--arm``
    spec, ``NAME=PATH`` (a payload; a built-in name is replaced in place) or
    ``NAME`` (a built-in)."""
    arms: dict[str, Any] = {} if only else dict(ARMS)
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not name or (sep and not path):
            raise SystemExit(f"--arm {spec!r}: expected NAME=PATH or a built-in NAME")
        if sep:
            if not Path(path).is_file():
                raise SystemExit(f"--arm {spec!r}: no payload at {path}")
            arms[name] = path
        elif name in ARMS:
            arms[name] = ARMS[name]
        else:
            raise SystemExit(f"--arm {name!r} is not a built-in arm ({', '.join(ARMS)})")
    if not arms:
        raise SystemExit("--only needs at least one --arm")
    return arms


def _payloads(arms: dict[str, Any]) -> dict[str, dict[str, Any]]:
    from experiments.noise_model import dregon_lawsearch_v4 as LS

    out = {}
    for name, spec in arms.items():
        if isinstance(spec, tuple):
            out[name] = LS.edit_payload(json.loads(Path(spec[0]).read_text()), spec[1])
        else:
            out[name] = json.loads(Path(spec).read_text())
    return out


def _windows(which: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """``(name, real (8, T), label (R, T))`` of one window set."""
    from experiments.noise_model import dregon_flight_v4 as DF
    from experiments.noise_model import gates as GT
    from experiments.noise_model import supports as SU
    from experiments.stochastic_fit import revised_eval as RE

    out = []
    if which == "scoring":
        for sup in GT.DREGON_CRUISE_SUPPORTS:
            clip = RE.load_window(
                sup.window,
                dataset=GT.DATASET["dregon"],
                version=None,
                channels=None,
                rps_key=GT.RAW_RPS_KEY["dregon"],
            )
            out.append((sup.recording, clip.audio, clip.rps))
    else:
        for spec in DF.pool_specs():
            clip = SU.flight_clip(spec)
            out.append((str(spec.args["recording"]), clip.audio, clip.rps))
    return [
        (
            n,
            np.asarray(a, dtype=np.float64)[:N_MICS],
            np.atleast_2d(np.asarray(r, dtype=np.float64)),
        )
        for n, a, r in out
    ]


def _compare(real: np.ndarray, arm: np.ndarray, rps: np.ndarray) -> dict[str, Any]:
    from experiments.noise_model import similarity as SIM
    from experiments.stochastic_fit import revised_eval as RE

    res: dict[str, Any] = SIM.battery(real, arm, rps)
    res["ltas"] = RE.ltas_deviation_db(real, arm)
    res["ltas"]["summary"] = dict(mean_abs_db=res["ltas"]["mean_abs_db"])
    return res


def _clean(obj: Any) -> Any:
    """JSON-ready, non-finite floats as ``null``."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _mean(xs: Any, axis: int = 0) -> Any:
    """NaN-aware mean along ``axis`` (all-NaN slices stay NaN, silently)."""
    a = np.asarray([np.asarray(x, dtype=np.float64) for x in xs]) if isinstance(xs, list) else xs
    fin = np.isfinite(a)
    m = np.where(fin, a, 0.0).sum(axis=axis) / np.where(
        fin.any(axis=axis), fin.sum(axis=axis), np.nan
    )
    return float(m) if np.ndim(m) == 0 else m


def _aggregate(per_window: list[dict[str, Any]], rows_order: tuple[str, ...]) -> dict[str, Any]:
    """Per arm: the window mean of every summary scalar, and of the per-band
    rows of the tables (mic-averaged first)."""
    out: dict[str, Any] = {}
    for arm in rows_order:
        rows = [w["arms"][arm] for w in per_window]
        summ = {
            f"{meas}.{k}": _mean([r[meas]["summary"][k] for r in rows])
            for meas in rows[0]
            for k in rows[0][meas]["summary"]
        }
        bands = {
            key: _mean([_mean(r["line_floor_lsd"][key]) for r in rows])
            for key in ("line_abs_db", "line_signed_db", "floor_abs_db", "floor_signed_db")
        }
        bands |= {
            "smear_fraction": _mean([r["line_floor_lsd"]["smear_fraction"] for r in rows]),
            "w1_db": _mean([r["level_distribution"]["w1_db"].mean(axis=0) for r in rows]),
            "msc_diff": _mean([r["spatial"]["msc_diff"] for r in rows]),
            "level_sd_diff_db": _mean([r["spatial"]["level_sd_diff_db"] for r in rows]),
            "is_octave": _mean([_mean(r["itakura_saito"]["is_octave_real_arm"]) for r in rows]),
            "is_octave_symmetric": _mean(
                [_mean(r["itakura_saito"]["is_octave_symmetric"]) for r in rows]
            ),
        }
        out[arm] = dict(summary=summ, bands=bands)
    return out


def _fmt(v: Any, nd: int = 2) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def _table(head: list[str], rows: list[list[str]]) -> list[str]:
    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"

    return [
        line(head),
        "|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|",
        *(line(r) for r in rows),
    ]


def _markdown(report: dict[str, Any]) -> str:
    from experiments.noise_model import similarity as SIM

    groups = [f"k{lo}-{hi}" for lo, hi in SIM.ORDER_GROUPS]
    order = tuple(report["arm_order"])
    md = [
        "# Noise model v4 — DREGON similarity battery",
        "",
        f"git `{report['git']}`, render seeds {SEED} (arm) and {FLOOR_SEED} (seed floor), "
        f"label `motors_command`, all {N_MICS} mics; wall {report['wall_s']:.0f} s. Real is the "
        "reference; signed numbers are arm − real. `<arm> seed floor` = the arm's seed-"
        f"{SEED} render as reference against its seed-{FLOOR_SEED} render on the same "
        "trajectory: the distance a perfect generator would have from data that were one "
        "more draw from it. Every number is the mean over the five windows of the set (and "
        "over mics/bands/orders as stated).",
        "",
    ]
    for set_name in SETS:
        agg = report["sets"][set_name]["aggregate"]
        dur = report["sets"][set_name]["window_seconds"]
        md += [
            f"## {set_name} windows ({len(report['sets'][set_name]['windows'])} × {dur:g} s)",
            "",
        ]

        def s(arm: str, key: str, nd: int = 2, agg: dict[str, Any] = agg) -> str:
            return _fmt(agg[arm]["summary"][key], nd)

        md += ["### Spectral: old LTAS, line/floor LSD (dB), Itakura-Saito", ""]
        md += _table(
            [
                "arm",
                "LTAS mic0 |Δ| (old)",
                "line |Δ|",
                "line Δ",
                "floor |Δ|",
                "floor Δ",
                "IS (linear bins, long-term PSD)",
                "IS sym (linear)",
                "IS octave-balanced",
                "IS sym octave-balanced",
            ],
            [
                [
                    a,
                    s(a, "ltas.mean_abs_db"),
                    s(a, "line_floor_lsd.line_abs_db"),
                    s(a, "line_floor_lsd.line_signed_db"),
                    s(a, "line_floor_lsd.floor_abs_db"),
                    s(a, "line_floor_lsd.floor_signed_db"),
                    s(a, "itakura_saito.is_div", 3),
                    s(a, "itakura_saito.is_symmetric", 3),
                    s(a, "itakura_saito.is_octave", 3),
                    s(a, "itakura_saito.is_octave_symmetric", 3),
                ]
                for a in order
            ],
        )
        md += [
            "",
            "### Spectro-temporal: 128 ms / 16 ms frames, third-octave log levels "
            f"({len(SIM.THIRD_OCTAVE_EDGES) - 1} bands, 31.5 Hz-7.9 kHz), mic mean",
            "",
        ]
        md += _table(
            [
                "arm",
                "Fréchet (dB²)",
                "FD mean term",
                "FD cov term",
                "co-mod ‖ΔR‖_F / n",
                "off-diag r (ref)",
                "off-diag r (arm)",
                "band sd Δ",
                "band sd |Δ|",
            ],
            [
                [
                    a,
                    s(a, "spectro_temporal.frechet_db2", 1),
                    s(a, "spectro_temporal.frechet_mean_term_db2", 1),
                    s(a, "spectro_temporal.frechet_cov_term_db2", 1),
                    s(a, "spectro_temporal.comod_frobenius", 3),
                    s(a, "spectro_temporal.real_offdiag_corr", 3),
                    s(a, "spectro_temporal.arm_offdiag_corr", 3),
                    s(a, "spectro_temporal.sd_signed_diff_db"),
                    s(a, "spectro_temporal.sd_abs_diff_db"),
                ]
                for a in order
            ],
        )
        md += [
            "",
            "### Order-tracked lines: |Δ| (dB) per order group, absolute peak and prominence",
            "",
        ]
        md += _table(
            [
                "arm",
                *[f"peak {g}" for g in groups],
                "peak all",
                *[f"prom {g}" for g in groups],
                "prom all",
            ],
            [
                [
                    a,
                    *[s(a, f"order_level_error.peak_abs_db_{g}") for g in groups],
                    s(a, "order_level_error.peak_abs_db"),
                    *[s(a, f"order_level_error.prominence_abs_db_{g}") for g in groups],
                    s(a, "order_level_error.prominence_abs_db"),
                ]
                for a in order
            ],
        )
        md += ["", "### Time structure and space", ""]
        md += _table(
            [
                "arm",
                "W1 (dB)",
                "W1 centred",
                "|Δsd|",
                "Δsd",
                "|Δ(p90−p10)|",
                "mod norm |Δ| (dB)",
                "mod abs |Δ| (dB)",
                "mic-spread |Δsd|",
                "MSC |Δ|",
                "MSC Δ",
            ],
            [
                [
                    a,
                    s(a, "level_distribution.w1_db"),
                    s(a, "level_distribution.w1_centered_db"),
                    s(a, "level_distribution.sd_abs_diff_db"),
                    s(a, "level_distribution.sd_signed_diff_db"),
                    s(a, "level_distribution.spread_abs_diff_db"),
                    s(a, "level_distribution.mod_norm_abs_db"),
                    s(a, "level_distribution.mod_abs_abs_db"),
                    s(a, "spatial.level_sd_abs_diff_db"),
                    s(a, "spatial.msc_abs_diff", 3),
                    s(a, "spatial.msc_signed_diff", 3),
                ]
                for a in order
            ],
        )
        for key, title, nd in (
            ("line_abs_db", "Line bins |ΔdB| per octave band (mic mean)", 2),
            ("line_signed_db", "Line bins ΔdB per octave band (mic mean, arm − real)", 2),
            ("floor_abs_db", "Floor cells |ΔdB| per octave band (mic mean)", 2),
            ("floor_signed_db", "Floor cells ΔdB per octave band (mic mean, arm − real)", 2),
            ("w1_db", "Frame band-level W1 (dB) per octave band (mic mean)", 2),
            ("msc_diff", "Mean pairwise MSC, arm − real, per octave band", 3),
            ("is_octave", "IS(real‖arm) per octave band, long-term PSD (mic mean)", 3),
        ):
            md += ["", f"### {title}", ""]
            md += _table(
                ["arm", *SIM.BAND_NAMES],
                [[a, *[_fmt(v, nd) for v in agg[a]["bands"][key]]] for a in order],
            )
        smear = agg[order[0]]["bands"]["smear_fraction"]
        md += [
            "",
            "Line-bin smearing on the arms' (= real) label (fraction of frames whose label "
            "moves the bin's order by > 0.05, mean over line bins), per band: "
            + ", ".join(f"{b} {_fmt(v)}" for b, v in zip(SIM.BAND_NAMES, smear, strict=True)),
            "",
        ]
    md += ["## Notes", "", *[f"- {n}" for n in report["notes"]], ""]
    return "\n".join(md)


NOTES = [
    "Lowest octave (31.25-62.5 Hz) holds no integer-order line at cruise (k = 1 is at 75-90 Hz), "
    "so its line column is empty (–); its floor column is the only sub-100 Hz reading.",
    "Line bins use each window's mean label (±0.05 order, 3.9 Hz bins). The label drifts: the "
    "smear fraction passes 0.5 above ~500 Hz and reaches 0.95 in 4-8 kHz, and because the "
    "frame-wise floor cells (`floor_cells`, ±5 Hz guard) follow the moving lines, nearly every "
    "bin above ~1 kHz is a floor cell in ≥ 25 % of frames — the line and floor sets overlap "
    "there (`line_bins`/`floor_bins` per band in the JSON), so the line/floor split is "
    "trustworthy below ~500 Hz only.",
    "Floor PSD: per bin the mean power over the frames where it is a floor cell; a bin enters "
    "only if it is a floor cell in ≥ 25 % of the frames (same cells on both sides: one label).",
    "Order levels: per 4096-point frame and rotor, the peak within ±0.08 order of k·f_r (frame-mean "
    "label) and its prominence over the median 0.3-0.5 orders away, frame-averaged in dB, per mic; "
    "orders up to 100 whose side window stays below 7.9 kHz. `accept_stats.order_profile` was not "
    "reused: it reads one constant rate per clip (single rotor, no drift).",
    "Modulation spectrum: the band envelope at the 62.5 Hz frame rate, Welch with ≤ 128-frame "
    "(2 s) segments zero-padded to 256, octave bins 0.5-1 … 16-31.25 Hz. `mod abs` is the "
    "contract's linear-power envelope with its mean removed (a level offset of x dB reads as 2x "
    "dB); `mod norm` divides by the mean first (modulation depth only) and is the level-free one.",
    "Coherence: Welch's MSC estimator (identical to `scipy.signal.coherence` with a 4096-point "
    "Hann) computed from one shared STFT at 75 % overlap; it is biased up by ≈ 1/n_frames.",
    "Model-vs-real comparisons render each arm on the REAL window's `motors_command` "
    "trajectory, so the arm's lines sit where the data's do. The seed floor renders the same "
    "payload on the same trajectory with a second seed, so it measures realisation noise "
    "only; it is a floor for the arm's own variability (an arm that is less variable than the "
    "data has a lower floor than the data would). The earlier half-split floor (real first half "
    "vs second half) was dropped: the halves follow different speed trajectories, so their "
    "lines sit at different frequencies, and they also differ by level drift.",
    "LTAS (old) is `revised_eval.ltas_deviation_db` (mic 0, 8192-point Welch, 7 bands 100-7900 Hz).",
    "Itakura-Saito: both forms read the LONG-TERM (whole-window) fine PSDs, so they are "
    "time-blind. The linear-bin mean weights every 3.9 Hz bin equally, so 2-8 kHz holds most "
    "of it; the octave-balanced form averages the eight per-octave means (per-octave values in "
    "report.json, `bands.is_octave`).",
    "Spectro-temporal (time-aware): per mic, 128 ms periodic-Hann frames at a 16 ms hop, "
    "log power summed in 24 third-octave bands (31.5 Hz-7.9 kHz). Fréchet distance between "
    "Gaussians fitted to the frame level vectors (dB², `scipy.linalg.sqrtm`, real part, ridge "
    "1e-6 dB²), split into the mean term (long-term spectrum) and the covariance term (how "
    "band levels vary and co-vary over time). Co-modulation: the 24×24 correlation matrix of "
    "frame levels; ‖R_arm − R_ref‖_F / 24 and each side's mean off-diagonal correlation "
    "(broadband bursts raise it). In seed-floor rows 'ref' is the seed-2001 render.",
]


def main(argv: list[str] | None = None) -> int:
    import subprocess

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="results/noise_v4/similarity/")
    ap.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME[=PATH]",
        help="add a payload arm NAME=PATH (a built-in NAME is replaced) or name a built-in arm; "
        "repeatable",
    )
    ap.add_argument(
        "--only",
        action="store_true",
        help="report only the --arm arms (each with its seed floor row); use another --out",
    )
    args = ap.parse_args(argv)
    arms = select_arms(list(args.arm), bool(args.only))
    from data_processing.noise_model.render import render_noise

    t_start = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payloads = _payloads(arms)
    report: dict[str, Any] = dict(
        schema="noise-v4-similarity/1",
        git=subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        seed=SEED,
        arms={
            k: (v if isinstance(v, str) else dict(base=v[0], point=v[1])) for k, v in arms.items()
        },
        arm_order=list(arm_rows(arms)),
        sets={},
        notes=NOTES,
    )
    for set_name in SETS:
        per_window = []
        windows = _windows(set_name)
        for name, real, rps in windows:
            t0 = time.time()
            n = real.shape[1]
            rows: dict[str, Any] = {}
            for arm, pay in payloads.items():
                x, y = (
                    np.asarray(render_noise(pay, rps, sr=16000, n_mics=N_MICS, seed=s))[:, :n]
                    for s in (SEED, FLOOR_SEED)
                )
                rows[arm] = _compare(real, x, rps)
                rows[f"{arm}{FLOOR_SUFFIX}"] = _compare(x, y, rps)
            per_window.append(
                dict(
                    window=name,
                    seconds=n / 16000.0,
                    label_mean_rps=rps.mean(axis=1),
                    label_sd_rps=rps.std(axis=1),
                    arms=rows,
                )
            )
            print(
                f"{set_name} {name}: {time.time() - t0:.1f}s  "
                + "  ".join(
                    f"{a} LTAS {rows[a]['ltas']['mean_abs_db']:.2f}"
                    f"/{rows[a + FLOOR_SUFFIX]['ltas']['mean_abs_db']:.2f}"
                    for a in arms
                ),
                flush=True,
            )
        report["sets"][set_name] = dict(
            windows=[w["window"] for w in per_window],
            window_seconds=per_window[0]["seconds"],
            aggregate=_aggregate(per_window, arm_rows(arms)),
            per_window=per_window,
        )
    report["wall_s"] = time.time() - t_start
    clean = _clean(report)
    (out / "report.json").write_text(json.dumps(clean, indent=1))
    (out / "report.md").write_text(_markdown(clean))
    print(f"wrote {out}/report.json, report.md; wall {report['wall_s']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
