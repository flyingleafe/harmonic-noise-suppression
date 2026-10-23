#!/usr/bin/env python3
"""Noise model v4: a battery of real-vs-synthetic similarity measures for the
DREGON flight arms — REPORT ONLY, nothing is optimised::

    OMP_NUM_THREADS=2 PYTHONPATH=src:scripts python scripts/noise_v4_similarity_report.py \\
        --out results/noise_v4/similarity/

The measures (:mod:`experiments.noise_model.similarity`): line/floor
log-spectral distance per octave band, order-tracked line levels, frame band
level distributions and modulation spectra, spatial level spread and
coherence, Itakura-Saito divergence, next to the old mic-0 band LTAS
(``revised_eval.ltas_deviation_db``) for reference. Two window sets, scored
separately: the five official 4 s DREGON cruise scoring windows
(``gates.DREGON_CRUISE_SUPPORTS``) and the five 8 s tuning segments
(``dregon_flight_v4.pool_specs()``). Every arm is rendered on the window's
``motors_command`` label, render seed 2001, all eight mics. ``real_self`` is
the noise floor: the real window's first half against its second half (each
side on its own label). Outputs ``report.json`` (every per-window, per-band,
per-mic number) and ``report.md`` (one row per arm).
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

SEED = 2001
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
ARM_ORDER = ("real_self", *ARMS)
SETS = ("scoring", "tuning")


def _payloads() -> dict[str, dict[str, Any]]:
    from experiments.noise_model import dregon_lawsearch_v4 as LS

    out = {}
    for name, spec in ARMS.items():
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


def _compare(
    real: np.ndarray, arm: np.ndarray, rps: np.ndarray, arm_rps: np.ndarray | None = None
) -> dict[str, Any]:
    from experiments.noise_model import similarity as SIM
    from experiments.stochastic_fit import revised_eval as RE

    res: dict[str, Any] = SIM.battery(real, arm, rps, arm_labels=arm_rps)
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


def _aggregate(per_window: list[dict[str, Any]]) -> dict[str, Any]:
    """Per arm: the window mean of every summary scalar, and of the per-band
    rows of the tables (mic-averaged first)."""
    out: dict[str, Any] = {}
    for arm in ARM_ORDER:
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
    md = [
        "# Noise model v4 — DREGON similarity battery",
        "",
        f"git `{report['git']}`, render seed {SEED}, label `motors_command`, all {N_MICS} mics; "
        f"wall {report['wall_s']:.0f} s. Real is the reference; signed numbers are arm − real. "
        "`real_self` = real first half vs real second half of each window (the noise floor a "
        "perfect generator would roughly reach; see Notes). Every number is the mean over the "
        "five windows of the set (and over mics/bands/orders as stated).",
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
                "IS(real‖arm)",
                "IS sym",
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
                ]
                for a in ARM_ORDER
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
                for a in ARM_ORDER
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
                for a in ARM_ORDER
            ],
        )
        for key, title, nd in (
            ("line_abs_db", "Line bins |ΔdB| per octave band (mic mean)", 2),
            ("line_signed_db", "Line bins ΔdB per octave band (mic mean, arm − real)", 2),
            ("floor_abs_db", "Floor cells |ΔdB| per octave band (mic mean)", 2),
            ("floor_signed_db", "Floor cells ΔdB per octave band (mic mean, arm − real)", 2),
            ("w1_db", "Frame band-level W1 (dB) per octave band (mic mean)", 2),
            ("msc_diff", "Mean pairwise MSC, arm − real, per octave band", 3),
        ):
            md += ["", f"### {title}", ""]
            md += _table(
                ["arm", *SIM.BAND_NAMES],
                [[a, *[_fmt(v, nd) for v in agg[a]["bands"][key]]] for a in ARM_ORDER],
            )
        smear = agg["v4_cwind"]["bands"]["smear_fraction"]
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
    "only if it is a floor cell in ≥ 25 % of frames on both sides.",
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
    "real_self compares halves (2 s on the scoring set, 4 s on the tuning set) while the arms "
    "compare full windows (4 s / 8 s): the estimation noise of every measure is larger for the "
    "halves (fewer frames: 28 vs 61 fine frames on the scoring set; MSC bias higher), and the "
    "halves carry different labels (the per-window label drift is in the JSON), so real_self is "
    "a pessimistic floor. No measure needed a window-length change beyond the modulation "
    "spectrum's segment length (2 s → one segment on a 2 s half).",
    "The real flight audio is not stationary within a window: the real_self LTAS error is "
    "4.8-5.7 dB on hovering, rectangle and spinning (mic 0 of hovering's halves differs by 3-7 dB "
    "per band; rectangle's label ramps 65 → 90 rev/s in its first 0.5 s), so real_self's "
    "level-sensitive columns (LTAS, line/floor Δ, W1, IS) "
    "measure that drift as well as estimation noise; its level-free columns (W1 centred, "
    "mod norm, MSC, prominence) are less affected and are the better floors.",
    "LTAS (old) is `revised_eval.ltas_deviation_db` (mic 0, 8192-point Welch, 7 bands 100-7900 Hz).",
]


def main(argv: list[str] | None = None) -> int:
    import subprocess

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="results/noise_v4/similarity/")
    args = ap.parse_args(argv)
    from data_processing.noise_model.render import render_noise

    t_start = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payloads = _payloads()
    report: dict[str, Any] = dict(
        schema="noise-v4-similarity/1",
        git=subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        seed=SEED,
        arms={
            k: (v if isinstance(v, str) else dict(base=v[0], point=v[1])) for k, v in ARMS.items()
        },
        sets={},
        notes=NOTES,
    )
    for set_name in SETS:
        per_window = []
        windows = _windows(set_name)
        for name, real, rps in windows:
            t0 = time.time()
            n = real.shape[1]
            h = n // 2
            rows: dict[str, Any] = {
                "real_self": _compare(
                    real[:, :h], real[:, h : 2 * h], rps[:, :h], rps[:, h : 2 * h]
                )
            }
            for arm, pay in payloads.items():
                x = np.asarray(render_noise(pay, rps, sr=16000, n_mics=N_MICS, seed=SEED))[:, :n]
                rows[arm] = _compare(real, x, rps)
            per_window.append(
                dict(
                    window=name,
                    seconds=n / 16000.0,
                    label_mean_rps=rps.mean(axis=1),
                    half_label_shift_rps=rps[:, h : 2 * h].mean(axis=1) - rps[:, :h].mean(axis=1),
                    arms=rows,
                )
            )
            print(
                f"{set_name} {name}: {time.time() - t0:.1f}s  "
                + "  ".join(f"{a} LTAS {rows[a]['ltas']['mean_abs_db']:.2f}" for a in ARM_ORDER),
                flush=True,
            )
        report["sets"][set_name] = dict(
            windows=[w["window"] for w in per_window],
            window_seconds=per_window[0]["seconds"],
            aggregate=_aggregate(per_window),
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
