#!/usr/bin/env python
"""Noise model v3: the checks of ``docs/explainers/noise-model-v3-wander.qmd`` §3.5.

Render and measure only: nothing here fits a likelihood, so every subcommand
is laptop-safe. Reads the reduced v3 fits (``results/noise_v3/fits``), the v2
fits they replace, and the measured wander records; writes
``results/noise_v3/checks/<check>/<check>.json`` plus figures.

    PYTHONPATH=src python scripts/noise_v3_checks.py prior      # (a) prior predictive
    PYTHONPATH=src python scripts/noise_v3_checks.py heldout    # (b) held-out line-power spread
    PYTHONPATH=src python scripts/noise_v3_checks.py rendered   # (c) rendered-audio tonality
    PYTHONPATH=src python scripts/noise_v3_checks.py latents    # (e) fitted latents vs measured
    PYTHONPATH=src python scripts/noise_v3_checks.py summary    # the section 3.5 table

``--keys dregon`` (or ``michaels_cruise,michaels_standby``) checks the fits
named; a later run with other keys keeps the earlier entries of each record.

(c)'s expectation half is ``scripts/noise_v2_tonality_audit.py --fit`` and (d)
is ``scripts/noise_v2_round_score.py --fit``; ``summary`` reads their records
(``results/noise_v3/checks/tonality/fits.json``, ``results/noise_v3/checks/
parity/arm_*.json``) next to this script's.

The three fits are keyed ``dregon`` (room-2 floor pool), ``michaels_cruise``
and ``michaels_standby`` (FLY125, fitted separately as R3 did). Renders are
driven by the REAL label of the window they are compared with
(:func:`data_processing.noise_model.render.render_noise`, eight mics, the
frozen render seeds) and measured by exactly the code that measured the
real windows: the wander estimator of ``scripts/noise_v3_measure_wander.py``
(:mod:`experiments.noise_model.wander`) and R4's ``line_width_db3``
(``scripts/noise_v2_widen_dregon.py``). A v3 render is normalised per channel
(the fit saw channel-normalised data); every statistic here is a within-mic
difference or a mic mean of dB, which a per-channel gain does not move.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from data_processing.noise_model import spectrum as DSP  # noqa: E402
from data_processing.noise_model.render import render_noise  # noqa: E402
from experiments.noise_model import supports as SUP  # noqa: E402
from experiments.noise_model import tonality as TN  # noqa: E402
from experiments.noise_model import wander as W  # noqa: E402

OUT = Path("results/noise_v3/checks")
FITS_DIR = Path("results/noise_v3/fits")
WANDER_DIR = Path("results/noise_v3/wander")
SCHEMA = "noise-v3-checks/1"

#: The v3 fits, by key: the pooled fit's name, its rig and its regime.
FITS: dict[str, dict[str, str]] = {
    "dregon": {"name": "dregon_room2_floor", "rig": "dregon", "regime": "cruise"},
    "michaels_cruise": {"name": "michaels_fly125_cruise", "rig": "michaels", "regime": "cruise"},
    "michaels_standby": {
        "name": "michaels_fly125_standby",
        "rig": "michaels",
        "regime": "standby",
    },
}
#: The fits of each rig (the held-out windows of a rig render from all of them).
RIG_KEYS = {rig: [k for k, v in FITS.items() if v["rig"] == rig] for rig in ("dregon", "michaels")}
#: The v2 fit each v3 fit replaces: R5's DREGON flight_profile (as fitted,
#: no calibration pin) and R3's per-regime Michael's pair.
V2_FITS = {
    "dregon": "results/noise_v2/rounds/round5/fits/dregon_room2_floor__flight_profile.json",
    "michaels_cruise": "results/noise_v2/rounds/round3/fits/michaels_fly125_cruise__flight.json",
    "michaels_standby": "results/noise_v2/rounds/round3/fits/michaels_fly125_standby__flight.json",
}
#: The line set each rig's wander was measured on (``<rig>.json`` ``line_set``).
LINE_SET = {"dregon": "resolvable", "michaels": "dominant"}

BLOCK_S = 0.5
RENDER_SEEDS = (2001, 2002, 2003, 2004)
N_MICS = 8
#: (b): the orders whose per-block prominence is histogrammed ("mid" comb).
MID_ORDERS = (8, 24)
#: Block prominence (cells over local floor) at which the LINE power equals
#: the local floor power: 10 log10(2). Below it the line is "under its floor".
UNDER_FLOOR_DB = float(10.0 * np.log10(2.0))
#: A line is PRESENT in a window when its window prominence reaches this
#: (``wander.PROMINENCE_DB``, the resolvable rule's block bar).
PRESENT_DB = W.PROMINENCE_DB
HIST_EDGES_DB = np.arange(-4.0, 40.5, 1.0)
N_BOOT = 1000
BOOT_SEED = 20260925
#: (a): draws from the prior per fit, the trajectory length, the width bar.
N_PRIOR_DRAWS = 8
PRIOR_TRAJ_S = 4.0
WIDTH_BAR_GAMMA0 = 5.0
#: (c): orders of the R4 width estimator.
WIDTH_ORDERS = tuple(range(1, 9))
LADDER_ORDERS = (1, 2, 4, 8, 12, 16, 24, 32, 48)
NAN = float("nan")


def _module(name: str) -> Any:
    """A sibling script as a module."""
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


WM = _module("noise_v3_measure_wander")
WIDEN = _module("noise_v2_widen_dregon")


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _clean(obj: Any) -> Any:
    """JSON-safe: numpy to Python, non-finite floats to None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(payload), indent=1) + "\n")
    return path


def write_merged(path: Path, payload: dict[str, Any], *fields: str) -> Path:
    """``write_json``, keeping the ``fields`` entries an earlier run of the same
    check wrote for OTHER fits (``--keys``: each family is checked as it lands)."""
    if path.is_file():
        old = json.loads(path.read_text())
        if old.get("check") == payload.get("check"):
            for field in fields:
                payload[field] = {**(old.get(field) or {}), **payload[field]}
    return write_json(path, payload)


def load_fit(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def v3_path(key: str, fits_dir: Path) -> Path:
    return Path(fits_dir) / f"{FITS[key]['name']}__flight_v3.json"


def fits_for(keys: Sequence[str], fits_dir: Path) -> dict[str, dict[str, Any]]:
    """``{key: {"v3": fit, "v2": fit, "v3_path": ..., "v2_path": ...}}``."""
    out = {}
    for key in keys:
        p3 = v3_path(key, fits_dir)
        if not p3.is_file():
            raise SystemExit(f"{p3}: no reduced v3 fit (run noise_v2_fit.py reduce first)")
        out[key] = dict(
            v3=load_fit(p3), v2=load_fit(V2_FITS[key]), v3_path=str(p3), v2_path=V2_FITS[key]
        )
    return out


def real_audio(spec: SUP.SupportSpec) -> tuple[np.ndarray, np.ndarray]:
    """``(audio (M, T), label (R, T))`` of a real flight window at 16 kHz: the
    loader :func:`supports._flight_support` reads (a Support keeps only the
    periodogram)."""
    from experiments.stochastic_fit import clips as C

    a = spec.args
    clip = C.load_clip(
        a["dataset"],
        a["recording"],
        a["start_s"],
        a["dur_s"],
        version=None,
        channels=None,
        rps_key=a["rps_key"],
        clip_id=spec.name,
    )
    clip = C.decimate(clip, SUP.SR)
    return np.asarray(clip.audio, dtype=np.float64), np.asarray(clip.rps, dtype=np.float64)


def render_on(fit: dict[str, Any], sup: SUP.Support, seed: int, name: str) -> SUP.Support:
    """``fit`` rendered on ``sup``'s own label, as a support on the same front end."""
    label = np.asarray(sup.carrier_rev_s_audio, dtype=np.float64)
    audio = render_noise(fit, label, sr=int(sup.sr), n_mics=N_MICS, seed=int(seed))
    return SUP.synthetic_support(
        name,
        audio,
        label,
        sr=int(sup.sr),
        segment=sup.segment,
        meta=dict(group="render", recording_id=name, source=sup.name, seed=int(seed)),
    )


def render_audio(fit: dict[str, Any], label: np.ndarray, sr: int, seed: int) -> np.ndarray:
    return render_noise(fit, label, sr=int(sr), n_mics=N_MICS, seed=int(seed))


def _q(v: Any, qs: Sequence[float] = (0.05, 0.5, 0.95)) -> list[float]:
    a = np.asarray(v, dtype=np.float64)
    a = a[np.isfinite(a)]
    return [float(np.quantile(a, q)) if a.size else NAN for q in qs]


# ---------------------------------------------------------------------------
# (a) prior predictive
# ---------------------------------------------------------------------------


def prior_traj_spec(key: str) -> SUP.SupportSpec:
    """The fixed 4 s trajectory each fit's prior is rendered on: a held-out
    window of the fit's own regime (DREGON's first frozen score window,
    FLY124 cruise @40 s, FLY124 standby @8 s; refined labels on Michael's)."""
    if key == "dregon":
        rec, start = SUP.DREGON_SCORED[0]
        return SUP.flight_dregon(rec, start, PRIOR_TRAJ_S)
    start = 40.0 if key == "michaels_cruise" else 8.0
    return SUP.flight_michaels("FLY124", start, PRIOR_TRAJ_S, rps_key="rps_refined")


def prior_draw(fit: dict[str, Any], rng: np.random.Generator) -> tuple[dict[str, Any], dict]:
    """One draw of every RIG site from the v3 prior the fit recorded.

    The laws are the model's (``model._sample_params_v3``): ``sigma_nu ~
    HalfNormal``, ``gamma_rk ~ HalfNormal(c gamma_0 k)``, ``profile_db ~
    N(measured centre, sd)``, ``z ~ N(0, I)`` on the floor spline at the
    measured ``mu``/``sigma_B``, the speed laws at their pins or priors, the
    wind at ``N(measured, sd)``. The wander hyperparameters are the fit's
    (measured, fixed); the render draws fresh OU tracks from them.
    """
    pr, p = fit["priors"], fit["params"]
    meas = fit["diagnostics"]["measured"]
    pins = fit["diagnostics"]["span_pins"]
    prof = np.asarray(p["profile"]["profile_db"], dtype=np.float64)
    r_, k_ = prof.shape
    ks = np.arange(1, k_ + 1, dtype=np.float64)
    g0, gc = float(pr["gamma_hz"]["gamma0_hz"]), float(pr["gamma_hz"]["gamma_c"])
    gamma = np.abs(rng.normal(0.0, gc * g0 * np.broadcast_to(ks, (r_, k_))))
    sigma_nu = abs(float(rng.normal(0.0, float(pr["sigma_nu"]["scale_rad_s"]))))
    centre = np.asarray(meas["profile_centre_db"], dtype=np.float64)[:r_, :k_]
    profile = rng.normal(centre, float(pr["profile_db"]["sd"]))
    pinned = set(pins.get("pinned") or [])

    def law(name: str, draw: Any) -> float:
        return float(pins["values"][name]) if name in pinned else float(draw())

    amp_exp = law("amp_exp", lambda: rng.normal(*pr["amp_exp"]))
    floor_exp = law("floor_exp", lambda: np.exp(rng.normal(*pr["log_floor_exp"])))
    static = law("floor_static_rel", lambda: np.exp(rng.normal(*pr["log_floor_static"])))
    z = rng.normal(size=len(p["floor"]["floor_shape_z"]))
    out = copy.deepcopy(fit)
    q = out["params"]
    q["sigma_nu"] = sigma_nu
    q["gamma_hz"] = gamma.tolist()
    q["profile"]["profile_db"] = profile.tolist()
    q["profile"]["amp_exp"] = amp_exp
    q["floor"]["floor_shape_z"] = z.tolist()
    q["floor"]["floor_exp"] = floor_exp
    q["floor"]["floor_static_rel"] = static
    if q.get("wind") is not None:
        q["wind"]["wind_db"] = rng.normal(
            np.asarray(meas["wind_db"], dtype=np.float64), float(pr["wind"]["sd"])
        ).tolist()
    ratio = gamma / (g0 * ks[None, :])
    shape = DSP.floor_shape_db(
        z, sr=int(fit["sr"]), scale_db=float(p["floor"]["floor_shape_sd_db"])
    )
    return out, dict(
        sigma_nu=sigma_nu,
        gamma_ratio_max=float(ratio.max()),
        gamma_ratio_q95=float(np.quantile(ratio, 0.95)),
        gamma_frac_over_bar=float((ratio > WIDTH_BAR_GAMMA0).mean()),
        gamma_hz_max=float(gamma.max()),
        gamma_hz_max_k1_8=float(gamma[:, :8].max()),
        floor_shape_sd_db=float(np.std(shape)),
        floor_shape_maxabs_db=float(np.max(np.abs(shape))),
        amp_exp=amp_exp,
        floor_exp=floor_exp,
        floor_static_rel=static,
    )


def floor_shape_of(sup: SUP.Support) -> dict[str, float]:
    """The floor's SHAPE on a window: comb-masked control-band levels
    (``wander.measure_floor``), mic mean of dB, time mean over blocks, minus
    their mean over bands; and the block-to-block swing of the band-mean level."""
    blocks = W.frame_blocks(sup.frame_centres_s, BLOCK_S)
    lb = W.measure_lines(sup.power, sup.freqs_hz, sup.carrier_rev_s, blocks, block_s=BLOCK_S)
    fb = W.measure_floor(
        sup.power,
        sup.freqs_hz,
        sup.carrier_rev_s,
        blocks,
        W.control_band_edges(),
        strong=lb.window_prominence_db >= W.STRONG_LINE_DB,
    )
    y = fb.micmean_db()  # (J, B)
    fin = np.isfinite(y)
    ok = fin.any(axis=1)  # bands with at least one surviving block
    band = np.where(fin, y, 0.0)[ok].sum(axis=1) / fin[ok].sum(axis=1)
    dev = band - band.mean()
    # the block level over the bands present in EVERY block (no composition drift)
    every = fin.all(axis=1)
    level = y[every].mean(axis=0) if every.any() else np.full(y.shape[1], NAN)
    return dict(
        shape_sd_db=float(np.std(dev)),
        shape_maxabs_db=float(np.max(np.abs(dev))),
        shape_span_db=float(dev.max() - dev.min()),
        level_block_sd_db=float(np.nanstd(level, ddof=1)),
        n_bands=int(ok.sum()),
    )


def widths(audio: np.ndarray, label: np.ndarray, sr: int) -> list[float]:
    """R4's -3 dB order-tracked width (Hz) at ``WIDTH_ORDERS``; NaN = no peak."""
    out = []
    for k in WIDTH_ORDERS:
        w = WIDEN.line_width_db3(audio, label, int(k), sr=int(sr))["width_hz"]
        out.append(NAN if w is None else float(w))
    return out


def run_prior(fits_dir: Path, out_dir: Path, keys: Sequence[str]) -> dict[str, Any]:
    fits = fits_for(keys, fits_dir)
    res: dict[str, Any] = {}
    for key, f in fits.items():
        fit = f["v3"]
        spec = prior_traj_spec(key)
        sup = SUP.load_support(spec)
        label = np.asarray(sup.carrier_rev_s_audio, dtype=np.float64)
        sigma_b = float(fit["params"]["floor"]["floor_shape_sd_db"])
        real = dict(floor=floor_shape_of(sup), widths_hz=widths(real_audio(spec)[0], label, sup.sr))
        rows = []
        rng = np.random.default_rng([BOOT_SEED, list(FITS).index(key)])
        for i in range(N_PRIOR_DRAWS):
            t0 = time.time()
            draw, stats = prior_draw(fit, rng)
            audio = render_audio(draw, label, sup.sr, seed=RENDER_SEEDS[0] + i)
            rsup = SUP.synthetic_support(
                f"prior_{key}_{i}", audio, label, sr=sup.sr, segment=sup.segment, meta={}
            )
            rows.append(
                dict(
                    draw=i,
                    params=stats,
                    render_floor=floor_shape_of(rsup),
                    render_widths_hz=widths(audio, label, sup.sr),
                    wall_s=time.time() - t0,
                )
            )
            print(
                f"  prior {key} draw {i}: max gamma/(g0 k) {stats['gamma_ratio_max']:.2f}, "
                f"shape sd {rows[-1]['render_floor']['shape_sd_db']:.2f} dB "
                f"(sigma_B {sigma_b:.2f})",
                flush=True,
            )
        ratio_max = max(r["params"]["gamma_ratio_max"] for r in rows)
        rw = np.array([r["render_widths_hz"] for r in rows], dtype=np.float64)
        res[key] = dict(
            fit=f["v3_path"],
            trajectory=spec.text,
            sigma_b_db=sigma_b,
            wander=fit["params"]["wander"],
            real=real,
            draws=rows,
            summary=dict(
                gamma_ratio_max=ratio_max,
                gamma_frac_over_bar=float(
                    np.mean([r["params"]["gamma_frac_over_bar"] for r in rows])
                ),
                gamma_hz_max=max(r["params"]["gamma_hz_max"] for r in rows),
                gamma_hz_max_k1_8=max(r["params"]["gamma_hz_max_k1_8"] for r in rows),
                sigma_nu_max=max(r["params"]["sigma_nu"] for r in rows),
                param_shape_sd_over_sigma_b_max=max(r["params"]["floor_shape_sd_db"] for r in rows)
                / sigma_b,
                param_shape_maxabs_over_sigma_b_max=max(
                    r["params"]["floor_shape_maxabs_db"] for r in rows
                )
                / sigma_b,
                render_shape_sd_db=_q(
                    [r["render_floor"]["shape_sd_db"] for r in rows], (0, 0.5, 1)
                ),
                render_shape_span_db=_q(
                    [r["render_floor"]["shape_span_db"] for r in rows], (0, 0.5, 1)
                ),
                render_level_block_sd_db=_q(
                    [r["render_floor"]["level_block_sd_db"] for r in rows], (0, 0.5, 1)
                ),
                render_width_hz_max_per_k=np.nanmax(rw, axis=0).tolist(),
                render_width_hz_median_per_k=np.nanmedian(rw, axis=0).tolist(),
                real_width_hz_per_k=real["widths_hz"],
                resolution_hz=float(sup.sr) / WIDEN.WIDE_N,
            ),
        )
    payload = dict(
        schema=SCHEMA,
        check="prior",
        git=git_head(),
        n_draws=N_PRIOR_DRAWS,
        traj_s=PRIOR_TRAJ_S,
        width_bar_gamma0=WIDTH_BAR_GAMMA0,
        note="profile_db drawn about diagnostics.measured.profile_centre_db (the fit's own "
        "prior centre); widths read at 8192-point R4 resolution (1.95 Hz), so the render "
        "widths bound a pathology (the v2 13 Hz half width at k=1), not a 0.03 k Hz line",
        fits=res,
    )
    write_merged(out_dir / "prior" / "prior.json", payload, "fits")
    return payload


# ---------------------------------------------------------------------------
# (b) held-out line-power spread and line intermittency
# ---------------------------------------------------------------------------


def heldout_specs(rig: str) -> list[tuple[SUP.SupportSpec, str]]:
    """``(spec, fit key)``: DREGON's five frozen room-2 score windows (4 s, not
    in the pool); FLY124's standby and cruise windows (8 s, refined label; the
    ramp window is outside both regime fits and is not rendered)."""
    if rig == "dregon":
        return [
            (SUP.flight_dregon(r, s, SUP.DREGON_SCORED_DUR_S), "dregon")
            for r, s in SUP.DREGON_SCORED
        ]
    return [
        (
            SUP.flight_michaels("FLY124", s, SUP.MICHAELS_FLIGHT_DUR_S, rps_key="rps_refined"),
            f"michaels_{regime}",
        )
        for s, regime in SUP.MICHAELS_FROZEN
        if regime != "ramp"
    ]


def measure(sup: SUP.Support, spec_text: str, tag: str, regime: str) -> tuple[Any, W.LineBlocks]:
    """``(WindowData, full LineBlocks)`` at ``BLOCK_S``: the wander script's
    per-window record (orders pruned to the resolvable ones) and every order's
    block prominence (for the intermittency statistics)."""
    wd = WM.measure_support(sup, spec_text, tag, regime, (BLOCK_S,))
    blocks = W.frame_blocks(sup.frame_centres_s, BLOCK_S)
    full = W.measure_lines(sup.power, sup.freqs_hz, sup.carrier_rev_s, blocks, block_s=BLOCK_S)
    return wd, full


def line_cells(full: W.LineBlocks) -> list[dict[str, Any]]:
    """Every mid-order (rotor, order) line of one window: its valid block
    prominences and its window prominence."""
    out = []
    sel = np.nonzero((full.orders >= MID_ORDERS[0]) & (full.orders <= MID_ORDERS[1]))[0]
    for r in range(full.prominence_db.shape[0]):
        for i in sel:
            ok = full.valid[r, i] & np.isfinite(full.prominence_db[r, i])
            if ok.sum() < 2:
                continue
            out.append(
                dict(
                    rotor=int(r),
                    order=int(full.orders[i]),
                    prom=full.prominence_db[r, i][ok],
                    window_prom=float(full.window_prominence_db[r, i]),
                )
            )
    return out


def intermittency(lines: Sequence[dict[str, Any]]) -> dict[str, float]:
    """The line statistics of a set of windows' mid-order lines.

    * ``frac_under`` -- share of all (line, block) cells whose line power is
      under its local floor (block prominence < 10 log10 2 = 3.01 dB);
    * ``frac_under_present`` -- the same among lines PRESENT in their window
      (window prominence >= 6 dB): how often a visible line disappears;
    * ``frac_intermittent`` -- share of all lines with at least one block
      >= 6 dB AND one block under the floor: a line that appears and
      disappears inside one window;
    * ``block_sd_present_db`` -- the median over present lines of the sd of
      their block prominence.
    """
    if not lines:
        return dict(n_lines=0)
    allp = np.concatenate([ln["prom"] for ln in lines])
    pres = [ln for ln in lines if ln["window_prom"] >= PRESENT_DB]
    presp = np.concatenate([ln["prom"] for ln in pres]) if pres else np.array([])
    inter = [
        (ln["prom"].max() >= PRESENT_DB) and (ln["prom"].min() < UNDER_FLOOR_DB) for ln in lines
    ]
    return dict(
        n_lines=len(lines),
        n_cells=int(allp.size),
        n_present=len(pres),
        frac_under=float(np.mean(allp < UNDER_FLOOR_DB)),
        frac_under_present=float(np.mean(presp < UNDER_FLOOR_DB)) if presp.size else NAN,
        frac_intermittent=float(np.mean(inter)),
        block_sd_present_db=(
            float(np.median([np.std(ln["prom"], ddof=1) for ln in pres])) if pres else NAN
        ),
        prom_median_db=float(np.median(allp)),
    )


def boot_intermittency(
    groups: Sequence[Sequence[dict[str, Any]]], rng: np.random.Generator, n_boot: int
) -> dict[str, list[float]]:
    """Window bootstrap (``groups`` = one list of lines per window, or per
    window with all its render seeds): [q05, q50, q95] of every statistic."""
    keys = ("frac_under", "frac_under_present", "frac_intermittent", "block_sd_present_db")
    draws: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, len(groups), size=len(groups))
        st = intermittency([ln for i in idx for ln in groups[int(i)]])
        for k in keys:
            draws[k].append(st.get(k, NAN))
    return {k: _q(v) for k, v in draws.items()}


def wander_numbers(wins: Sequence[Any], which: str, n_boot: int) -> dict[str, Any]:
    """The wander script's estimator on a window set: point numbers (rig
    centring, tau at lag 1) and the window bootstrap [q05, q50, q95]."""
    s = WM.analyse_setting(wins, BLOCK_S, which=which, lags=WM.LAGS_1)
    bt = WM.bootstrap(wins, which, BLOCK_S, n_boot=n_boot, seed=BOOT_SEED)
    return dict(
        numbers=s["numbers"],
        n_line_tracks=s["n_line_tracks"],
        n_floor_tracks=s["n_floor_tracks"],
        bootstrap=bt["lag1"],
        zero_share=bt["lag1"]["zero_share"],
    )


def prom_hist(lines: Sequence[dict[str, Any]]) -> list[int]:
    if not lines:
        return [0] * (HIST_EDGES_DB.size - 1)
    allp = np.clip(
        np.concatenate([ln["prom"] for ln in lines]), HIST_EDGES_DB[0], HIST_EDGES_DB[-1]
    )
    return np.histogram(allp, bins=HIST_EDGES_DB)[0].tolist()


def run_heldout(
    fits_dir: Path, out_dir: Path, keys: Sequence[str], *, n_boot_wander: int
) -> dict[str, Any]:
    fits = fits_for(keys, fits_dir)
    rigs: dict[str, Any] = {}
    # a rig is checked once every fit its held-out windows render from is in ``keys``
    for rig in [r for r in ("dregon", "michaels") if all(k in fits for k in RIG_KEYS[r])]:
        which = LINE_SET[rig]
        specs = heldout_specs(rig)
        arms: dict[str, dict[str, Any]] = {
            a: dict(wins=[], groups=[], windows=[]) for a in ("real", "v3", "v2")
        }
        for wi, (spec, key) in enumerate(specs):
            t0 = time.time()
            sup = SUP.load_support(spec)
            regime = FITS[key]["regime"]
            wd, full = measure(sup, spec.text, "real", regime)
            arms["real"]["wins"].append(wd)
            arms["real"]["groups"].append(line_cells(full))
            arms["real"]["windows"].append(dict(spec=spec.text, fit=key, regime=regime))
            for arm in ("v3", "v2"):
                group: list[dict[str, Any]] = []
                for seed in RENDER_SEEDS:
                    name = f"{arm}_{sup.name}_s{seed}"
                    rs = render_on(fits[key][arm], sup, seed, name)
                    rwd, rfull = measure(rs, f"render:{name}", arm, regime)
                    arms[arm]["wins"].append(rwd)
                    group += line_cells(rfull)
                arms[arm]["groups"].append(group)
                arms[arm]["windows"].append(dict(spec=spec.text, fit=key, seeds=list(RENDER_SEEDS)))
            print(f"  heldout {rig} window {wi}: {sup.name} ({time.time() - t0:.0f} s)", flush=True)
        rng = np.random.default_rng(BOOT_SEED)
        block: dict[str, Any] = dict(line_set=which)
        for arm, a in arms.items():
            lines = [ln for g in a["groups"] for ln in g]
            block[arm] = dict(
                windows=a["windows"],
                n_windows=len(a["wins"]),
                wander=wander_numbers(a["wins"], which, n_boot_wander),
                intermittency=intermittency(lines),
                intermittency_boot=boot_intermittency(a["groups"], rng, N_BOOT),
                prominence_hist=prom_hist(lines),
            )
        rigs[rig] = block
    payload = dict(
        schema=SCHEMA,
        check="heldout",
        git=git_head(),
        block_s=BLOCK_S,
        render_seeds=list(RENDER_SEEDS),
        mid_orders=list(MID_ORDERS),
        under_floor_db=UNDER_FLOOR_DB,
        present_db=PRESENT_DB,
        hist_edges_db=HIST_EDGES_DB.tolist(),
        fits={k: dict(v3=v["v3_path"], v2=v["v2_path"]) for k, v in fits.items()},
        wander_contract={
            rig: json.loads((WANDER_DIR / f"{rig}.json").read_text())
            for rig in ("dregon", "michaels")
        },
        note="real = held-out windows (DREGON frozen room-2 score windows, motors_command label; "
        "FLY124 standby + cruise, rps_refined label); v3/v2 = the fit rendered on each window's "
        "own label at the four frozen render seeds, measured by the same estimator. Wander "
        "numbers: scripts/noise_v3_measure_wander.py's rig-centred moments (tau at lag 1), "
        "window bootstrap; intermittency: every (rotor, order 8-24, block), window bootstrap "
        "with a window's render seeds kept together",
        rigs=rigs,
    )
    write_merged(out_dir / "heldout" / "heldout.json", payload, "rigs", "fits")
    figure_heldout(payload, out_dir / "heldout")
    return payload


def figure_heldout(payload: dict[str, Any], out_dir: Path) -> Path:
    edges = np.asarray(payload["hist_edges_db"], dtype=np.float64)
    centres = 0.5 * (edges[1:] + edges[:-1])
    rigs = list(payload["rigs"])
    fig, axes = plt.subplots(1, len(rigs), figsize=(7.0 * len(rigs), 5.6), squeeze=False)
    style = {"real": ("k", "-", 2.0), "v3": ("C3", "-", 1.6), "v2": ("C0", "--", 1.4)}
    for ax, rig in zip(axes[0], rigs):
        hi = edges[1]
        for arm, (c, ls, lw) in style.items():
            h = np.asarray(payload["rigs"][rig][arm]["prominence_hist"], dtype=np.float64)
            st = payload["rigs"][rig][arm]["intermittency"]
            drop = st.get("frac_under_present")
            lab = (
                f"{arm}: under floor {100 * (st.get('frac_under') or NAN):.1f} %, "
                "present-line dropouts "
                + ("n/a (no present line)" if drop is None else f"{100 * drop:.1f} %")
            )
            ax.step(centres, h / max(h.sum(), 1.0), where="mid", color=c, ls=ls, lw=lw, label=lab)
            if h.any():
                hi = max(hi, float(edges[int(np.nonzero(h)[0][-1]) + 1]))
        uf, pr = float(payload["under_floor_db"]), float(payload["present_db"])
        ax.axvline(uf, color="0.5", lw=1.0, ls=":", label=f"line power = floor ({uf:.2f} dB)")
        ax.axvline(pr, color="0.5", lw=1.0, ls="-.", label=f"present bar ({pr:g} dB)")
        ax.set_xlim(edges[0], hi + 1.0)
        ax.set_title(
            f"{rig}: block prominence, orders {payload['mid_orders'][0]}-{payload['mid_orders'][1]}"
        )
        ax.set_xlabel("block prominence (cells over local floor), dB")
        ax.set_ylabel("share of (line, block) cells")
        ax.grid(alpha=0.3)
        # below the axes: inside, the five entries cover the histogram's peak
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
    fig.tight_layout()
    path = out_dir / "prominence_hist.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# (c) rendered-audio tonality: R4 widths and the prominence ladder
# ---------------------------------------------------------------------------


def pattern_specs() -> dict[str, SUP.SupportSpec]:
    """Support name -> spec over every window the tonality patterns come from."""
    from experiments.noise_model import rig_sampler as RS

    return {s.name: s for group in RS._real_window_specs().values() for s in group}


def ladder(full: W.LineBlocks) -> dict[str, Any]:
    """Rotor-median window prominence at ``LADDER_ORDERS`` and the per-rotor
    count / highest order at >= 6 dB (the rendered twin of the audit's
    ``count_ge6`` / ``highest_ge6``)."""
    wp = np.where(np.isfinite(full.window_prominence_db), full.window_prominence_db, -np.inf)
    ks = full.orders
    at = {
        int(k): float(np.median(wp[:, int(np.nonzero(ks == k)[0][0])]))
        for k in LADDER_ORDERS
        if k in ks
    }
    hits = wp >= PRESENT_DB
    return dict(
        prom_at_k_db=at,
        count_ge6=float(hits.sum(axis=1).mean()),
        highest_ge6=float(np.mean([ks[h].max() if h.any() else 0 for h in hits])),
        curve_db=np.median(wp, axis=0).tolist(),
        orders=ks.tolist(),
    )


def run_rendered(fits_dir: Path, out_dir: Path, keys: Sequence[str]) -> dict[str, Any]:
    fits = fits_for(keys, fits_dir)
    specs = pattern_specs()
    patterns = TN.select_patterns()
    res: dict[str, Any] = {}
    for key, f in fits.items():
        rig, regime = FITS[key]["rig"], FITS[key]["regime"]
        pats = [p for p in patterns if p.rig == rig and p.regime == regime]
        rows = []
        for pat in pats:
            spec = specs[pat.support]
            sup = SUP.load_support(spec)
            label = np.asarray(sup.carrier_rev_s_audio, dtype=np.float64)
            audio_real = real_audio(spec)[0]
            blocks = W.frame_blocks(sup.frame_centres_s, BLOCK_S)
            real_full = W.measure_lines(
                sup.power, sup.freqs_hz, sup.carrier_rev_s, blocks, block_s=BLOCK_S
            )
            row: dict[str, Any] = dict(
                pattern=pat.name,
                support=spec.text,
                duration_s=float(sup.duration_s),
                real=dict(widths_hz=widths(audio_real, label, sup.sr), ladder=ladder(real_full)),
            )
            for arm in ("v3", "v2"):
                clips = []
                for seed in RENDER_SEEDS:
                    t0 = time.time()
                    audio = render_audio(f[arm], label, sup.sr, seed)
                    rs = SUP.synthetic_support(
                        f"{arm}_{pat.name}_{seed}",
                        audio,
                        label,
                        sr=sup.sr,
                        segment=sup.segment,
                        meta={},
                    )
                    rfull = W.measure_lines(
                        rs.power, rs.freqs_hz, rs.carrier_rev_s, blocks, block_s=BLOCK_S
                    )
                    clips.append(
                        dict(
                            seed=seed, widths_hz=widths(audio, label, sup.sr), ladder=ladder(rfull)
                        )
                    )
                    print(
                        f"  rendered {key} {pat.name} {arm} s{seed} ({time.time() - t0:.0f} s)",
                        flush=True,
                    )
                row[arm] = clips
            rows.append(row)
        res[key] = dict(fit_v3=f["v3_path"], fit_v2=f["v2_path"], patterns=rows)
    payload = dict(
        schema=SCHEMA,
        check="rendered_tonality",
        git=git_head(),
        width_orders=list(WIDTH_ORDERS),
        ladder_orders=list(LADDER_ORDERS),
        resolution_hz=16000.0 / WIDEN.WIDE_N,
        render_seeds=list(RENDER_SEEDS),
        note="each fit rendered on the REAL label of the two tonality-audit pattern windows "
        "of its regime (TN.select_patterns: narrowest / widest rotor spread) at four seeds = 8 "
        "clips per fit; widths = R4 line_width_db3 (8192/1024, order-tracked, mic and rotor "
        "mean); ladder = wander.measure_lines window prominence (mic-summed 3-bin line cells "
        "over the local q25 floor), rotor median",
        fits=res,
    )
    write_merged(out_dir / "tonality" / "rendered.json", payload, "fits")
    return payload


# ---------------------------------------------------------------------------
# (e) the fitted latents against the measured wander
# ---------------------------------------------------------------------------


def pool_specs(fit: dict[str, Any]) -> dict[str, SUP.SupportSpec]:
    """The fit's pool windows by support name (the named set the name came from)."""
    names = set(fit["supports"])
    out = {}
    for set_name in ("dregon-floor", "michaels-cruise", "michaels-standby"):
        for s in SUP.support_set(set_name):
            if s.name in names:
                out[s.name] = s
    return out


def _lag1(arrs: Sequence[np.ndarray]) -> float:
    """Lag-1 autocorrelation of ``(..., B)`` tracks, products pooled over every
    track and window, about ZERO, the latents' prior mean (centring a 8-16
    block track on its own mean would bias the lag-1 product low by the very
    ``tau`` it is compared with)."""
    num = den = 0.0
    for x in arrs:
        a = np.asarray(x, dtype=np.float64)
        num += float((a[..., 1:] * a[..., :-1]).sum())
        den += float((a[..., :-1] ** 2).sum() + (a[..., 1:] ** 2).sum()) / 2.0
    return num / den if den > 0 else NAN


def run_latents(fits_dir: Path, out_dir: Path, keys: Sequence[str]) -> dict[str, Any]:
    detail = json.loads((WANDER_DIR / "wander_detail.json").read_text())
    res: dict[str, Any] = {}
    for key in keys:
        cfg = FITS[key]
        path = v3_path(key, fits_dir)
        fit = load_fit(path)
        lat = fit["latents"]
        wander = fit["params"]["wander"]
        bs = float(wander["block_s"])
        noise = detail["rigs"][cfg["rig"]]["noise"][f"{bs:g}"]
        s2_line = float(noise["line_s2_measured_median_db2"])
        s2_floor = float(noise["floor_s2_measured_median_db2"])
        specs = pool_specs(fit)
        tracks: dict[str, list[np.ndarray]] = {"d": [], "v": [], "v_present": [], "u": [], "uj": []}
        for w in lat["windows"]:
            for name in ("d", "u", "uj"):
                if w.get(name) is not None:
                    tracks[name].append(np.atleast_2d(np.asarray(w[name], dtype=np.float64)))
            if w.get("v") is None:
                continue
            v = np.asarray(w["v"], dtype=np.float64)  # (R, K, B)
            tracks["v"].append(v.reshape(-1, v.shape[-1]))
            spec = specs.get(w["name"])
            if spec is None:
                continue
            sup = SUP.load_support(spec)
            blocks = W.frame_blocks(sup.frame_centres_s, bs)
            lb = W.measure_lines(sup.power, sup.freqs_hz, sup.carrier_rev_s, blocks, block_s=bs)
            present = lb.window_prominence_db >= PRESENT_DB  # (R, K_measured)
            kk = min(present.shape[1], v.shape[1])
            sel = present[:, :kk]
            if sel.any():
                tracks["v_present"].append(v[:, :kk][sel])
        rho = {
            n: float(np.exp(-bs / float(wander[f"tau_{n}_s"]))) if wander.get(f"tau_{n}_s") else NAN
            for n in ("d", "v", "u", "uj")
        }
        fam: dict[str, Any] = {}
        for name, arrs in tracks.items():
            base = "v" if name == "v_present" else name
            sigma = wander.get(f"sigma_{base}_db")
            if not arrs or sigma is None:
                fam[name] = dict(n_values=0, measured_sigma_db=sigma)
                continue
            # each window's tracks keep their own block axis; pooled moments
            n_val = sum(a.size for a in arrs)
            sd2 = float(sum(float((a**2).sum()) for a in arrs) / max(n_val, 1))
            s2 = s2_floor if base in ("u", "uj") else s2_line
            sig2 = float(sigma) ** 2
            fam[name] = dict(
                n_values=int(sum(a.size for a in arrs)),
                fitted_sd_db=float(np.sqrt(sd2)),
                fitted_sd_plus_noise_db=float(np.sqrt(sd2 + s2)),
                block_noise_s2_db2=s2,
                measured_sigma_db=float(sigma),
                shrunk_expectation_db=float(np.sqrt(sig2 * sig2 / (sig2 + s2)))
                if sig2 > 0
                else 0.0,
                fitted_lag1=_lag1(arrs),
                measured_rho=rho[base],
            )
        res[key] = dict(
            fit=str(path),
            block_s=bs,
            wander=wander,
            summary_recorded=lat.get("summary"),
            families=fam,
        )
    payload = dict(
        schema=SCHEMA,
        check="latents",
        git=git_head(),
        note="fitted_sd = rms of the fitted block latents (pooled over windows); +noise = sqrt("
        "fitted var + the measured median block noise at block_s, "
        "wander_detail.json rigs.<rig>.noise); shrunk_expectation = sqrt(sigma^4 / (sigma^2 + "
        "s^2)), the spread a correctly-shrunk MAP latent has in the explainer's toy (section "
        "3.3a); v_present = the per-line residual on the lines PRESENT in the pool window "
        "(window prominence >= 6 dB, wander.LineBlocks.window_prominence_db at block_s)",
        fits=res,
    )
    write_merged(out_dir / "latents" / "latents.json", payload, "fits")
    return payload


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


def _f(v: Any, fmt: str = "{:.2f}") -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    return fmt.format(x) if np.isfinite(x) else "—"


def _ci(q: Sequence[Any] | None, fmt: str = "{:.2f}") -> str:
    if not q:
        return "—"
    return f"{_f(q[1], fmt)} [{_f(q[0], fmt)}, {_f(q[2], fmt)}]"


def _pct(q: Sequence[Any] | None) -> str:
    if not q:
        return "—"
    p = [None if v is None else 100.0 * float(v) for v in q]
    return f"{_f(p[1], '{:.1f}')} [{_f(p[0], '{:.1f}')}, {_f(p[2], '{:.1f}')}] %"


def summary_md(out_dir: Path) -> str:
    lines: list[str] = ["# Noise model v3 — the section 3.5 checks", ""]
    lines.append(
        "`scripts/noise_v3_checks.py` (render and measure only). Every number below is read "
        "from the JSON named in its section."
    )
    lines.append("")
    p = out_dir / "prior" / "prior.json"
    if p.is_file():
        d = json.loads(p.read_text())
        lines += [
            "## (a) Prior predictive (`prior/prior.json`)",
            "",
            f"{d['n_draws']} draws of every rig site from each fit's recorded v3 prior, rendered "
            f"on a fixed {d['traj_s']:g} s held-out trajectory (8 mics).",
            "",
            "| fit | max γ/(0.01k) | share of lines > 5 γ0k | max γ Hz (k ≤ 8) | max σ_ν rad/s | "
            "drawn shape sd / σ_B (max) | drawn max abs / σ_B (max) | σ_B dB | render floor "
            "shape sd dB (min / med / max) | real floor shape sd dB | render width Hz k=1..8 (max) "
            "| real width Hz k=1..8 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|",
        ]
        for key, r in d["fits"].items():
            s = r["summary"]
            lines.append(
                f"| {key} | {_f(s['gamma_ratio_max'])} | {_f(100 * s['gamma_frac_over_bar'], '{:.1f}')} % "
                f"| {_f(s['gamma_hz_max_k1_8'], '{:.3f}')} | {_f(s['sigma_nu_max'])} "
                f"| {_f(s['param_shape_sd_over_sigma_b_max'])} "
                f"| {_f(s['param_shape_maxabs_over_sigma_b_max'])} | {_f(r['sigma_b_db'])} "
                f"| {' / '.join(_f(v) for v in s['render_shape_sd_db'])} "
                f"| {_f(r['real']['floor']['shape_sd_db'])} "
                f"| {', '.join(_f(v, '{:.1f}') for v in s['render_width_hz_max_per_k'])} "
                f"| {', '.join(_f(v, '{:.1f}') for v in s['real_width_hz_per_k'])} |"
            )
        lines.append("")
    p = out_dir / "heldout" / "heldout.json"
    if p.is_file():
        d = json.loads(p.read_text())
        lines += [
            "## (b) Held-out line-power spread and intermittency (`heldout/heldout.json`)",
            "",
            "Wander estimator of `noise_v3_measure_wander.py` (rig-centred, block "
            f"{d['block_s']:g} s, τ at lag 1), window bootstrap median [5 %, 95 %]:",
            "",
            "| rig | arm | windows | line tracks | σ_total dB | σ_d dB | σ_v dB | σ_u dB |",
            "|---|---|---:|---:|---|---|---|---|",
        ]
        for rig, b in d["rigs"].items():
            for arm in ("real", "v3", "v2"):
                w = b[arm]["wander"]
                bt = w["bootstrap"]
                # no line track passed the estimator's rule: the line spreads are
                # unidentified (the estimator reports 0), not zero
                line = (
                    {
                        k: f"{_f(w['numbers'][k])} ({_ci(bt[k])})"
                        for k in ("sigma_total_db", "sigma_d_db", "sigma_v_db")
                    }
                    if w["n_line_tracks"]
                    else dict.fromkeys(
                        ("sigma_total_db", "sigma_d_db", "sigma_v_db"), "— (no track)"
                    )
                )
                lines.append(
                    f"| {rig} | {arm} | {b[arm]['n_windows']} | {w['n_line_tracks']} "
                    f"| {line['sigma_total_db']} | {line['sigma_d_db']} | {line['sigma_v_db']} "
                    f"| {_f(w['numbers']['sigma_u_db'])} ({_ci(bt['sigma_u_db'])}) |"
                )
        lines += [
            "",
            f"Mid-order lines (k {d['mid_orders'][0]}-{d['mid_orders'][1]}, every rotor), per "
            f"block: UNDER the floor = line power below the local floor (prominence < "
            f"{d['under_floor_db']:.2f} dB); PRESENT = window prominence >= {d['present_db']:g} dB. "
            "Window bootstrap median [5 %, 95 %]:",
            "",
            "| rig | arm | lines | cells under floor | present-line blocks under floor | "
            "lines that appear and disappear | block sd of a present line dB |",
            "|---|---|---:|---|---|---|---|",
        ]
        for rig, b in d["rigs"].items():
            for arm in ("real", "v3", "v2"):
                st, bt = b[arm]["intermittency"], b[arm]["intermittency_boot"]
                lines.append(
                    f"| {rig} | {arm} | {st.get('n_lines', 0)} ({st.get('n_present', 0)} present) "
                    f"| {_pct(bt['frac_under'])} | {_pct(bt['frac_under_present'])} "
                    f"| {_pct(bt['frac_intermittent'])} | {_ci(bt['block_sd_present_db'])} |"
                )
        lines += ["", "Figure: `heldout/prominence_hist.png`.", ""]
    p = out_dir / "tonality" / "rendered.json"
    if p.is_file():
        d = json.loads(p.read_text())
        lines += [
            "## (c) Tonality on rendered audio (`tonality/rendered.json`)",
            "",
            "R4 `line_width_db3` (-3 dB width, Hz, 8192-point, resolution "
            f"{d['resolution_hz']:.2f} Hz) at k = 1..8 and the order-6-dB counts on the window "
            "prominence ladder; renders: median over the pattern windows x 4 seeds.",
            "",
            "| fit | arm | width k=1..8 Hz | orders >= 6 dB / rotor | highest >= 6 dB | prom k=1,2,4,8,16 dB |",
            "|---|---|---|---:|---:|---|",
        ]
        for key, r in d["fits"].items():
            for arm in ("real", "v3", "v2"):
                ws, cs, hs, pk = [], [], [], []
                for row in r["patterns"]:
                    items = [row["real"]] if arm == "real" else row[arm]
                    for it in items:
                        ws.append(it["widths_hz"])
                        cs.append(it["ladder"]["count_ge6"])
                        hs.append(it["ladder"]["highest_ge6"])
                        pk.append(
                            [
                                it["ladder"]["prom_at_k_db"].get(str(k), NAN)
                                for k in (1, 2, 4, 8, 16)
                            ]
                        )
                wmed = np.nanmedian(np.asarray(ws, dtype=np.float64), axis=0)
                pmed = np.nanmedian(np.asarray(pk, dtype=np.float64), axis=0)
                lines.append(
                    f"| {key} | {arm} | {', '.join(_f(v, '{:.1f}') for v in wmed)} "
                    f"| {_f(np.median(cs), '{:.1f}')} | {_f(np.median(hs), '{:.0f}')} "
                    f"| {', '.join(_f(v, '{:.1f}') for v in pmed)} |"
                )
        lines.append("")
    p = out_dir / "tonality" / "fits.json"
    if p.is_file():
        d = json.loads(p.read_text())
        lines += [
            "## (c) Tonality on the expectation (`tonality/fits.json`, `noise_v2_tonality_audit.py --fit`)",
            "",
            "| pattern | payload | prom k=1,2,4,8,16 dB (mic median) | orders >= 6 dB / rotor | highest >= 6 dB | trend crosses floor at k | pedestal median dB |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
        # per pattern: the v3 fit of its rig, then the v2 anchor of its regime
        for pat in d["patterns"]:
            rig, name = pat["rig"], pat["name"]
            anchor = rig + ("_standby" if pat["regime"] == "standby" else "")
            for label, st in (
                (f"v3 {rig}", d["rows"].get(rig, {}).get(name)),
                (f"v2 anchor {anchor}", d["references"]["rows"].get(anchor, {}).get(name)),
            ):
                if not isinstance(st, dict) or "k_max" not in st:
                    continue
                pk = [st.get(f"prom_k{k}", [NAN, NAN, NAN])[1] for k in (1, 2, 4, 8, 16)]
                ped = (
                    st["pedestal"]["median_db"]
                    if isinstance(st["pedestal"], dict)
                    else st["pedestal"]
                )
                lines.append(
                    f"| {name} | {label} | {', '.join(_f(v, '{:.1f}') for v in pk)} "
                    f"| {_f(st['count_ge6'][0], '{:.1f}')} | {_f(st['highest_ge6'][0], '{:.0f}')} "
                    f"| {_f(st['trend_cross_order'][0], '{:.0f}')} | {_f(ped, '{:.1f}')} |"
                )
        lines.append("")
    parity = sorted((out_dir / "parity").glob("arm_*.json"))
    if parity:
        lines += [
            "## (d) Parity gate (`parity/arm_*.json`, `noise_v2_round_score.py --fit`)",
            "",
            "HPPNet PIT MAE on renders of the frozen supports (four frozen render seeds, 8 mics) "
            "against the legacy PARITY bar and DREGON's frozen STRETCH target; proxy "
            "`ltas_abs_db` against its closure-0.7 gate.",
            "",
            "| arm | rig | PIT MAE rev/s (mean) | 95 % upper | Michael's ratio | parity bar | "
            "parity | stretch bar | stretch | proxy dB | proxy gate dB |",
            "|---|---|---:|---:|---:|---:|---|---:|---|---:|---:|",
        ]
        for path in parity:
            a = json.loads(path.read_text())
            groups = a["gates"]["proxy"]["groups"]
            for rig, b in a["bars"].items():
                if not isinstance(b, dict) or not b.get("cohort_complete"):
                    continue
                grp = groups.get(f"{rig}_cruise") or {}
                st = b.get("stretch") or {}
                lines.append(
                    f"| `{path.name}` | {rig} | {_f(b.get('mean_rev_s'), '{:.6f}')} "
                    f"| {_f(b.get('interval_upper_rev_s'), '{:.6f}')} | {_f(b.get('ratio'), '{:.4f}')} "
                    f"| {_f(b['parity']['bar_rev_s'], '{:.6f}')} "
                    f"| {'PASS' if b['parity']['within'] else 'FAIL'} "
                    f"| {_f(st.get('bar_rev_s'), '{:.6f}')} "
                    f"| {'—' if st.get('within') is None else ('PASS' if st['within'] else 'FAIL')} "
                    f"| {_f(grp.get('mean_ltas_abs_db'), '{:.4f}')} | {_f(grp.get('gate_db'), '{:.4f}')} |"
                )
        lines.append("")
    p = out_dir / "latents" / "latents.json"
    if p.is_file():
        d = json.loads(p.read_text())
        lines += [
            "## (e) Fitted latents against the measured wander (`latents/latents.json`)",
            "",
            "| fit | family | values | fitted sd dB | sqrt(fitted var + s²) dB | measured σ dB | "
            "shrunk expectation dB | fitted lag-1 | measured ρ |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for key, r in d["fits"].items():
            for name, f in r["families"].items():
                if not f.get("n_values"):
                    continue
                lines.append(
                    f"| {key} | {name} | {f['n_values']} | {_f(f['fitted_sd_db'])} "
                    f"| {_f(f['fitted_sd_plus_noise_db'])} | {_f(f['measured_sigma_db'])} "
                    f"| {_f(f['shrunk_expectation_db'])} | {_f(f['fitted_lag1'])} "
                    f"| {_f(f['measured_rho'])} |"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=("prior", "heldout", "rendered", "latents", "summary"))
    ap.add_argument("--fits", type=Path, default=FITS_DIR, help="directory of the reduced v3 fits")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument(
        "--wander-boot", type=int, default=WM.N_BOOT, help="(b): wander window-bootstrap draws"
    )
    ap.add_argument(
        "--keys",
        default=",".join(FITS),
        help=f"comma list of the fits to check (default all: {','.join(FITS)}); a rerun with "
        "other keys keeps the earlier fits' entries",
    )
    args = ap.parse_args(argv)
    keys = [k.strip() for k in str(args.keys).split(",") if k.strip()]
    unknown = sorted(set(keys) - set(FITS))
    if unknown:
        raise SystemExit(f"--keys: unknown fit(s) {unknown}; known {list(FITS)}")
    out_dir = Path(args.out)
    t0 = time.time()
    if args.cmd == "prior":
        run_prior(args.fits, out_dir, keys)
    elif args.cmd == "heldout":
        run_heldout(args.fits, out_dir, keys, n_boot_wander=int(args.wander_boot))
    elif args.cmd == "rendered":
        run_rendered(args.fits, out_dir, keys)
    elif args.cmd == "latents":
        run_latents(args.fits, out_dir, keys)
    text = summary_md(out_dir)
    (out_dir / "findings.md").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "findings.md").write_text(text)
    print(f"# {args.cmd} done in {time.time() - t0:.0f} s; wrote {out_dir / 'findings.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
