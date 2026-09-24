"""Why do DREGON's rotor lines appear and disappear? A laptop diagnostic.

The wander measurement (``scripts/noise_v3_measure_wander.py``,
``results/noise_v3/wander/findings.md``) read the line-level sigma off
RESOLVABLE tracks only -- block prominence >= 6 dB in >= 80 % of a window's
0.5 s blocks -- which leaves out exactly the DREGON lines that come and go.
This script characterises those left out, on the same windows
(``_mic_gain_rank.rig_specs(extra=True)``), the same front end and the same
block line measurement (:func:`experiments.noise_model.wander.measure_lines`).

A (window, rotor, k) TRACK is valid in a block when its line stays in the
band and on the grid; a track valid in >= 80 % of blocks is INTERMITTENT when
its prominence is >= 6 dB in 20-80 % of its valid blocks, RESOLVABLE at >= 80 %,
ALWAYS-UNDER below 20 %. An ABSENT block is one under 6 dB. Three tests, each
on DREGON and on Michael's rig as the control, and each also run as a
COUNTERFACTUAL -- the mechanism removed, the intermittent tracks re-measured:
how many absent blocks return, present ones go, tracks stay intermittent.

(a) LABEL DISPLACEMENT. The carrier the tracks follow is DREGON's commanded
    speed (``motors_command``), not a measured one. Re-estimates of where the
    line really is, the prominence recomputed there: (a1) per block, the
    rotor's carrier offset (``measure_lines`` at ``f_r + dc`` on a grid) that
    maximises the mean prominence of the rotor's OTHER line orders
    (leave-one-out, no selection on the track itself); (a2) per track and
    block, a peak search +-3 bins around ``k f_r`` (``measure_lines`` on the
    periodogram shifted by ``s`` bins) -- a best-of-seven, so an upper bound,
    with the recoveries that land on ANOTHER rotor's line counted apart and
    the always-under tracks as the null; (a3) the label smoothed (window
    mean, 1 s moving mean: a motor's inertia low-passes the command).
(b) DISTRIBUTION SHAPE. The pooled block prominence against a Gaussian in dB,
    forward-modelled to the observable: each track's line-to-floor ratio an
    OU of the resolvable lines' ``(sigma_total, tau_total)``
    (``results/noise_v3/wander/wander_detail.json``) plus the rig's measured
    block noise, optionally minus the track's MEASURED local floor
    deviation, plus the empirical cell ratio of line-free blocks, read as
    prominence, the level matched to the track's median; and the same with
    the sigma fitted to the observed scatter. Scatter, kurtosis, 1- vs
    2-Gaussian BIC, the tail (blocks > 6 dB under the track mean, under
    3 dB), lag-1 autocorrelation, against 200 draws; on the intermittent
    tracks, on the near-threshold ones (median 4-8 dB, chosen blind to the
    scatter) and on the resolvable ones (the calibration).
(c) FLOOR MASKING. Pearson r of a track's block prominence against the floor
    under it -- its own local floor (the prominence's denominator), the
    ``measure_floor`` control band that holds the line, and the level over
    bands (comb-masked, speed law removed) -- per track and pooled within
    tracks; the share of absent blocks whose floor sits > +1 sigma_u over its
    mean against the base rate; and the counterfactual: the line's absolute
    level held, the floor moved to its mean.

Usage
-----

    PYTHONPATH=src python scripts/_dregon_intermittency.py
    PYTHONPATH=src python scripts/_dregon_intermittency.py --limit 3 --n-sim 5   # smoke
    PYTHONPATH=src python scripts/_dregon_intermittency.py --cache /tmp/m.pkl    # re-analyse
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pickle
import subprocess
import sys
import warnings
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter1d

from experiments.noise_model import supports as SUP
from experiments.noise_model import wander as W


def _module(name: str) -> Any:
    """A sibling script as a module (the wander measurement's loader and laws)."""
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


MW = _module("noise_v3_measure_wander")

OUT_DIR = Path("results/noise_v3/intermittency")
WANDER_DIR = Path("results/noise_v3/wander")
SCHEMA = "noise-v3-intermittency/1"
RIGS = ("dregon", "michaels")
BLOCK_S = MW.CHOSEN_BLOCK_S
PROM_DB = W.PROMINENCE_DB
#: Intermittent: prominent in [LO, HI) of the valid blocks; resolvable at >= HI.
INTER_LO = 0.2
INTER_HI = W.PROMINENT_BLOCK_FRAC
#: A track is classified only if it is valid (in band, on the grid) in this
#: share of the window's blocks; a line leaving the band is not intermittency.
MIN_VALID_FRAC = 0.8
#: (a2) the per-track peak search, in bins either side of the label bin.
SHIFT_BINS = 3
#: (a2) a recovered peak is ANOTHER rotor's when that rotor's nearest order
#: sits within this many bins of the peak bin and nearer than the own label.
NEIGHBOUR_BINS = 1.0
#: (a1) the comb carrier re-estimate: offsets from the label, rev/s, clipped
#: per rotor to half the gap to the nearest other rotor (a comb offset by a
#: whole rotor gap would lock onto that rotor's comb).
COMB_STEP_REV_S = 0.05
COMB_MAX_REV_S = 1.0
#: (a1) the other line orders a comb estimate reads: an order this high
#: moves by >= 0.3 bins per 0.5 rev/s, lower ones barely see the offset.
COMB_MIN_ORDER = 5
#: (a3) smoothed labels: the rotor's window-mean carrier, and a centred 1 s
#: moving mean of the frame carriers.
SMOOTH_LABELS = ("window_mean", "moving_1s")
#: (b) the Gaussian forward model.
N_SIM = 200
SIM_SEED = 20260924
#: (b) bisection steps of the fitted-sigma Gaussian.
FIT_STEPS = 14
#: (b) the track sets: the intermittent ones; every classified track whose
#: MEDIAN block prominence is in this band (dB), a set chosen blind to the
#: scatter the 20-80 % rule selects on; and the resolvable ones.
SHAPE_SETS = ("intermittent", "near_threshold", "resolvable")
NEAR_THRESHOLD_DB = (4.0, 8.0)
#: (b) a class with fewer tracks than this is not shape-tested.
MIN_SHAPE_TRACKS = 5
#: (b) forward-model draws pooled into each model histogram of the figure.
HIST_DRAWS = 10
#: (c) the floor measures: the prominence's own local floor, the control band
#: that holds the line (the model's u + u_j there), the level over bands (u).
FLOOR_KEYS = ("local", "uj", "u")
#: A track or block counts as line-free for the floor-ratio null when its
#: window-median prominence is under this (always-under tracks only).
NULL_MEDIAN_DB = 1.0
TAIL_DB = 6.0
GONE_DB = 3.0
N_WORKERS = 4
#: (c) window bootstrap of the pooled within-track correlation.
N_BOOT = 200
NAN = float("nan")


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ---------------------------------------------------------------------------
# measurement (one window)
# ---------------------------------------------------------------------------


@dataclass
class Win:
    """One window at ``BLOCK_S``. Axes: ``s`` shift, ``g`` comb offset, ``r``
    rotor, ``k`` order (``orders[k]``), ``b`` block, ``j`` floor band."""

    name: str
    tag: str
    regime: str
    recording: str
    orders: np.ndarray  # (K,)
    df: float
    carrier: np.ndarray  # (R, B) block-mean label carrier, rev/s
    valid: np.ndarray  # (R, K, B)
    prom: np.ndarray  # (R, K, B) at the label, NaN where invalid
    prom_shift: np.ndarray  # (S, R, K, B) periodogram shifted by shifts[s] bins
    shifts: np.ndarray  # (S,)
    comb_prom: np.ndarray  # (G, R, K, B) float32 at label + comb_grid[g], NaN out of reach
    comb_grid: np.ndarray  # (G,)
    comb_reach: np.ndarray  # (R,) the grid's reach per rotor, rev/s
    prom_smooth: np.ndarray  # (L, R, K, B) at the smoothed labels SMOOTH_LABELS
    own_share: np.ndarray  # (R, K)
    s2_line: np.ndarray  # (R, K, B) measured mic-mean block noise, dB^2
    u_band: np.ndarray  # (J, B) floor per control band (mic mean, dB), speed law removed
    local_floor: np.ndarray  # (R, K, B) the prominence's own denominator, dB


def shifted(power: np.ndarray, s: int) -> np.ndarray:
    """``out[..., f] = power[..., f + s]`` (edge bins repeated)."""
    if s == 0:
        return power
    a = abs(s)
    pad = np.pad(power, ((0, 0), (0, 0), (a, a)), mode="edge")
    return pad[..., a + s : a + s + power.shape[-1]]


def local_floor_db(
    pw: np.ndarray,
    freqs: np.ndarray,
    car: np.ndarray,
    blocks: Sequence[np.ndarray],
    orders: np.ndarray,
) -> np.ndarray:
    """``(R, K, B)`` the floor right under each line: the mic-summed local floor
    :func:`wander.measure_lines` divides by (the ``LOCAL_FLOOR_QUANTILE`` of the
    cells ``LOCAL_FLOOR_GAP_BINS``..``LOCAL_FLOOR_HALF_BINS`` either side of the
    label bin over the block's frames, over ``-ln(1 - q)``), in dB."""
    df = float(freqs[1] - freqs[0])
    w_hi = W.LOCAL_FLOOR_HALF_BINS
    side = np.arange(W.LOCAL_FLOOR_GAP_BINS, w_hi + 1)
    offs = np.concatenate([-side[::-1], side])
    q_scale = -math.log1p(-W.LOCAL_FLOOR_QUANTILE)
    m_, t_, f_ = pw.shape
    t = np.arange(t_)[None, :, None]
    out = np.full((car.shape[0], orders.size, len(blocks)), np.nan)
    for r in range(car.shape[0]):
        b0 = np.rint(orders[:, None] * car[r][None, :] / df).astype(np.int64)
        b0 = np.clip(b0, w_hi, f_ - 1 - w_hi)  # (K, T)
        cells = pw[:, t, b0[:, :, None] + offs]  # (M, K, T, S)
        for b, fidx in enumerate(blocks):
            flat = cells[:, :, fidx, :].reshape(m_, orders.size, -1)
            q = np.quantile(flat, W.LOCAL_FLOOR_QUANTILE, axis=-1).sum(axis=0) / q_scale
            out[r, :, b] = 10.0 * np.log10(np.maximum(q, 1e-300))
    return out


def measure_window(spec_text: str, tag: str, regime: str) -> Win:
    sup = SUP.load_support(SUP.parse_spec(spec_text))
    pw = np.asarray(sup.power, dtype=np.float64)
    freqs = np.asarray(sup.freqs_hz, dtype=np.float64)
    car = np.asarray(sup.carrier_rev_s, dtype=np.float64)
    blocks = W.frame_blocks(sup.frame_centres_s, BLOCK_S)
    lb = W.measure_lines(pw, freqs, car, blocks, block_s=BLOCK_S)
    orders = lb.orders
    ks = [int(k) for k in orders]
    df = float(freqs[1] - freqs[0])
    prom = np.where(lb.valid, lb.prominence_db, np.nan)

    shifts = np.arange(-SHIFT_BINS, SHIFT_BINS + 1)
    prom_shift = np.stack(
        [
            np.where(
                lb.valid,
                W.measure_lines(
                    shifted(pw, int(s)), freqs, car, blocks, block_s=BLOCK_S, orders=ks
                ).prominence_db,
                np.nan,
            )
            for s in shifts
        ]
    )

    # (a1) per-rotor comb carrier: each rotor's grid stops half-way to its nearest neighbour
    cbar = car.mean(axis=1)
    r_ = cbar.size
    gap = np.array([np.min(np.abs(cbar[r] - np.delete(cbar, r))) for r in range(r_)])
    reach = np.minimum(COMB_MAX_REV_S, 0.5 * gap)
    n_half = int(round(COMB_MAX_REV_S / COMB_STEP_REV_S))
    grid = np.arange(-n_half, n_half + 1) * COMB_STEP_REV_S
    g_prom = np.full((grid.size,) + prom.shape, np.nan, dtype=np.float32)
    for gi, dc in enumerate(grid):
        use = np.abs(dc) <= reach + 1e-9
        if not use.any():
            continue
        lg = W.measure_lines(pw, freqs, car + dc, blocks, block_s=BLOCK_S, orders=ks)
        g_prom[gi][use] = np.where(lb.valid & lg.valid, lg.prominence_db, np.nan)[use]

    # (a3) the label smoothed: a motor's inertia low-passes the commanded speed
    fps = W.FLIGHT_SR / W.FLIGHT_HOP
    smooth = {
        "window_mean": np.broadcast_to(cbar[:, None], car.shape),
        "moving_1s": uniform_filter1d(car, size=int(round(fps)), axis=1, mode="nearest"),
    }
    prom_smooth = np.stack(
        [
            np.where(
                lb.valid,
                W.measure_lines(
                    pw,
                    freqs,
                    np.ascontiguousarray(smooth[n]),
                    blocks,
                    block_s=BLOCK_S,
                    orders=ks,
                ).prominence_db,
                np.nan,
            )
            for n in SMOOTH_LABELS
        ]
    )

    # the floor: the wander measurement's comb-masked control bands
    strong = lb.window_prominence_db >= W.STRONG_LINE_DB
    fb = W.measure_floor(pw, freqs, car, blocks, W.control_band_edges(), strong=strong)
    u_band = fb.micmean_db() - MW.floor_speed_db(lb)[None, :]
    return Win(
        name=sup.name,
        tag=tag,
        regime=regime,
        recording=str(sup.meta.get("recording_id", sup.name)),
        orders=orders,
        df=df,
        carrier=lb.carrier_rev_s,
        valid=lb.valid,
        prom=prom,
        prom_shift=prom_shift,
        shifts=shifts,
        comb_prom=g_prom,
        comb_grid=grid,
        comb_reach=reach,
        prom_smooth=prom_smooth,
        own_share=lb.own_share,
        s2_line=lb.s2_emp_micmean,
        u_band=u_band,
        local_floor=np.where(lb.valid, local_floor_db(pw, freqs, car, blocks, orders), np.nan),
    )


def _measure(row: tuple[str, str, str]) -> Win:
    return measure_window(*row)


def measure_rig(rows: Sequence[tuple[Any, str, str]], workers: int) -> list[Win]:
    args = [(spec.text, tag, regime) for spec, tag, regime in rows]
    if workers <= 1:
        return [_measure(a) for a in args]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_measure, args))


# ---------------------------------------------------------------------------
# the block table
# ---------------------------------------------------------------------------

CLASSES = ("resolvable", "intermittent", "under", "off_grid")
RES, INTER, UNDER, OFF = range(4)


def classify(w: Win) -> np.ndarray:
    """``(R, K)`` index into :data:`CLASSES`."""
    nval = w.valid.sum(axis=-1)
    pres = (np.nan_to_num(w.prom, nan=-np.inf) >= PROM_DB).sum(axis=-1)
    frac = pres / np.maximum(nval, 1)
    out = np.full(nval.shape, OFF)
    ok = nval >= MIN_VALID_FRAC * w.prom.shape[-1]
    out[ok & (frac >= INTER_HI)] = RES
    out[ok & (frac >= INTER_LO) & (frac < INTER_HI)] = INTER
    out[ok & (frac < INTER_LO)] = UNDER
    return out


def _argmax_small_first(x: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """argmax over axis 0 (NaN = -inf); ties go to the smallest ``|offset|``."""
    order = np.argsort(np.abs(offsets), kind="stable")
    filled = np.where(np.isfinite(x), x, -np.inf)
    return order[np.argmax(filled[order], axis=0)]


def nearest_other_rotor(w: Win, pos: np.ndarray) -> np.ndarray:
    """``(R, K, B)`` distance in bins from ``pos`` (fractional bins, ``(R, K, B)``)
    to the nearest order of any OTHER rotor at the block-mean labels."""
    out = np.full(pos.shape, np.inf)
    for r2 in range(w.carrier.shape[0]):
        c2 = w.carrier[r2][None, None, :] / w.df
        k2 = np.maximum(1.0, np.rint(pos / c2))
        d = np.abs(pos - k2 * c2)
        d[r2] = np.inf
        out = np.minimum(out, d)
    return out


def loo_comb(w: Win, cls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(a1) ``(P, dc)`` ``(R, K, B)``: a track's prominence at its rotor's carrier
    re-estimated per block from the rotor's OTHER line orders (resolvable or
    intermittent, order >= COMB_MIN_ORDER): the offset maximising their mean
    prominence. NaN where the rotor has no other line order."""
    g = w.comb_prom.astype(np.float64)
    p_out = np.full(w.prom.shape, np.nan)
    dc_out = np.full(w.prom.shape, np.nan)
    lines = ((cls == RES) | (cls == INTER)) & (w.orders[None, :] >= COMB_MIN_ORDER)
    cols = np.arange(w.prom.shape[-1])
    for r in range(cls.shape[0]):
        idx = np.nonzero(lines[r])[0]
        for k in np.nonzero(cls[r] <= UNDER)[0]:
            others = idx[idx != k]
            if others.size == 0:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # a block no other order sees
                score = np.nanmean(g[:, r, others, :], axis=1)  # (G, B)
            best = _argmax_small_first(score, w.comb_grid)
            seen = np.isfinite(score).any(axis=0)
            p_out[r, k] = np.where(seen, g[best, r, k, cols], np.nan)
            dc_out[r, k] = np.where(seen, w.comb_grid[best], np.nan)
    return p_out, dc_out


def window_rows(w: Win, wi: int, tid0: int) -> dict[str, np.ndarray]:
    """One row per valid block of every classified track of ``w``."""
    cls = classify(w)
    r_, k_, b_ = w.prom.shape
    rr, kk, bb = np.meshgrid(np.arange(r_), np.arange(k_), np.arange(b_), indexing="ij")
    keep = (cls[..., None] <= UNDER) & w.valid
    best = _argmax_small_first(w.prom_shift, w.shifts)
    s_star = w.shifts[best]
    p_max = np.take_along_axis(w.prom_shift, best[None], axis=0)[0]
    x_own = w.orders[None, :, None] * w.carrier[:, None, :] / w.df
    peak = np.rint(x_own) + s_star
    d_own = np.abs(peak - x_own)
    d_nb = nearest_other_rotor(w, peak)
    p_loo, dc_loo = loo_comb(w, cls)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a band the comb covers
        ub_c = w.u_band - np.nanmean(w.u_band, axis=1, keepdims=True)
        # the floor LEVEL: every band's deviation from its window mean, averaged
        # over the bands measured in at least half the blocks
        dense = np.isfinite(w.u_band).mean(axis=1) >= 0.5
        u_c = np.nanmean(ub_c[dense], axis=0)
    edges = W.control_band_edges()
    f_line = w.orders[None, :] * w.carrier.mean(axis=1)[:, None]
    band = np.clip(np.searchsorted(edges, f_line, side="right") - 1, 0, edges.size - 2)
    uj = ub_c[band]  # (R, K, B): the band that holds the line, NaN where not measured
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # an off-grid track
        lf = w.local_floor - np.nanmean(w.local_floor, axis=-1, keepdims=True)
    dev = w.carrier - w.carrier.mean(axis=1, keepdims=True)
    label_dev = w.orders[None, :, None] * dev[:, None, :] / w.df
    full = (r_, k_, b_)

    def pick(a: np.ndarray) -> np.ndarray:
        return np.broadcast_to(a, full)[keep]

    out = dict(
        win=np.full(int(keep.sum()), wi),
        track=pick(tid0 + rr * k_ + kk),
        rotor=pick(rr),
        order=pick(w.orders[None, :, None]),
        block=pick(bb),
        cls=pick(cls[..., None]),
        own_share=pick(w.own_share[..., None]),
        p0=pick(w.prom),
        p_max=pick(p_max),
        s_star=pick(s_star),
        d_own=pick(d_own),
        d_nb=pick(d_nb),
        d_nb_label=pick(nearest_other_rotor(w, x_own)),
        p_loo=pick(p_loo),
        dc_loo=pick(dc_loo),
        u=pick(u_c[None, None, :]),
        uj=pick(uj),
        local=pick(lf),
        carrier=pick(w.carrier[:, None, :]),
        label_dev_bins=pick(label_dev),
        s2_line=pick(w.s2_line),
        df=np.full(int(keep.sum()), w.df),
    )
    for li, name in enumerate(SMOOTH_LABELS):
        out[f"p_{name}"] = pick(w.prom_smooth[li])
    return out


def rig_rows(wins: Sequence[Win]) -> dict[str, np.ndarray]:
    parts = []
    tid = 0
    for wi, w in enumerate(wins):
        parts.append(window_rows(w, wi, tid))
        tid += w.prom.shape[0] * w.prom.shape[1]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def take(rows: dict[str, np.ndarray], m: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[m] for k, v in rows.items()}


def split_tracks(rows: dict[str, np.ndarray], key: str) -> list[np.ndarray]:
    """``rows[key]`` per track (rows are track-contiguous, blocks in order)."""
    t = rows["track"]
    if t.size == 0:
        return []
    cut = np.nonzero(np.diff(t))[0] + 1
    return np.split(rows[key], cut)


def _frac(m: np.ndarray, of: np.ndarray | None = None) -> float:
    base = m if of is None else m[of]
    return float(base.mean()) if base.size else NAN


def _qs(a: np.ndarray, qs: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95)) -> list[float]:
    v = np.asarray(a, dtype=np.float64)
    v = v[np.isfinite(v)]
    return [float(np.quantile(v, q)) for q in qs] if v.size else [NAN] * len(qs)


def regime_after(rows: dict[str, np.ndarray], p_new: np.ndarray) -> dict[str, float]:
    """The intermittent tracks under a counterfactual block prominence ``p_new``
    (aligned with ``rows``; NaN keeps the observed value): the share of tracks
    still intermittent / now resolvable / now always-under, the share of
    absent blocks that return and of present ones that go, and the
    present<->absent switches between consecutive blocks over the observed."""
    m = rows["cls"] == INTER
    sub = take(rows, m)
    sub["p_cf"] = np.where(np.isfinite(p_new[m]), p_new[m], sub["p0"])
    fr, sw_old, sw_new = [], 0, 0
    for old, new in zip(split_tracks(sub, "p0"), split_tracks(sub, "p_cf")):
        on_old, on_new = old >= PROM_DB, new >= PROM_DB
        fr.append(on_new.mean())
        sw_old += int(np.sum(on_old[1:] != on_old[:-1]))
        sw_new += int(np.sum(on_new[1:] != on_new[:-1]))
    f = np.array(fr)
    absent = sub["p0"] < PROM_DB
    return dict(
        absent_recovered=_frac(sub["p_cf"] >= PROM_DB, absent),
        present_lost=_frac(sub["p_cf"] < PROM_DB, ~absent),
        still_intermittent=_frac((f >= INTER_LO) & (f < INTER_HI)),
        now_resolvable=_frac(f >= INTER_HI),
        now_under=_frac(f < INTER_LO),
        switches_over_observed=sw_new / sw_old if sw_old else NAN,
    )


# ---------------------------------------------------------------------------
# counts
# ---------------------------------------------------------------------------


def counts(wins: Sequence[Win], rows: dict[str, np.ndarray]) -> dict[str, Any]:
    per = {c: 0 for c in CLASSES}
    for w in wins:
        cls = classify(w)
        for i, c in enumerate(CLASSES):
            per[c] += int((cls == i).sum())
    first = np.r_[True, np.diff(rows["track"]) != 0]
    t = take(rows, first)
    inter = t["cls"] == INTER
    groups = {
        name: int((inter & (t["order"] >= lo) & (t["order"] <= hi)).sum())
        for name, lo, hi in MW.ORDER_GROUPS
    }
    rotors = {str(r + 1): int((inter & (t["rotor"] == r)).sum()) for r in range(4)}
    lines = sorted({(int(r) + 1, int(k)) for r, k in zip(t["rotor"][inter], t["order"][inter])})
    by_line: dict[str, int] = {}
    for r, k in zip(t["rotor"][inter], t["order"][inter]):
        key = f"{int(r) + 1}:{int(k)}"
        by_line[key] = by_line.get(key, 0) + 1
    top = sorted(by_line.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
    inter_rows = rows["cls"] == INTER
    return dict(
        n_windows=len(wins),
        n_blocks=int(sum(w.prom.shape[-1] for w in wins)),
        tracks=per,
        intermittent_by_order_group=groups,
        intermittent_by_rotor=rotors,
        intermittent_distinct_lines=len(lines),
        intermittent_most_windows=[dict(line=k, windows=v) for k, v in top],
        intermittent_dominant=int((inter & (t["own_share"] >= W.OWN_SHARE_MIN)).sum()),
        intermittent_own_share_quantiles=_qs(t["own_share"][inter]),
        intermittent_blocks=int(inter_rows.sum()),
        intermittent_absent_blocks=int((inter_rows & (rows["p0"] < PROM_DB)).sum()),
        windows_with_intermittent=int(len(np.unique(rows["win"][inter_rows]))),
    )


# ---------------------------------------------------------------------------
# (a) label displacement
# ---------------------------------------------------------------------------


def label_test(rows: dict[str, np.ndarray]) -> dict[str, Any]:
    inter = rows["cls"] == INTER
    under = rows["cls"] == UNDER
    absent = rows["p0"] < PROM_DB
    present = ~absent
    nb = (rows["d_nb"] <= NEIGHBOUR_BINS) & (rows["d_nb"] < rows["d_own"])
    rec = rows["p_max"] >= PROM_DB
    dc_search = rows["s_star"] * rows["df"] / rows["order"]

    def search(sel: np.ndarray) -> dict[str, Any]:
        a = sel & absent
        own = a & rec & ~nb
        return dict(
            absent_blocks=int(a.sum()),
            recovered=_frac(rec, a),
            recovered_other_rotor=_frac(rec & nb, a),
            recovered_own=_frac(rec & ~nb, a),
            shift_bins_counts={
                str(int(s)): int((own & (rows["s_star"] == s)).sum())
                for s in range(-SHIFT_BINS, SHIFT_BINS + 1)
            },
            own_shift_abs_rev_s_quantiles=_qs(np.abs(dc_search[own])),
            own_shift_at_edge=_frac(np.abs(rows["s_star"]) == SHIFT_BINS, own),
        )

    a_i = inter & absent
    p_i = inter & present
    has_loo = np.isfinite(rows["p_loo"])
    smooth = {
        name: dict(
            regime_after(rows, rows[f"p_{name}"]),
            mean_gain_absent_db=float(np.nanmean(rows[f"p_{name}"][a_i] - rows["p0"][a_i])),
        )
        for name in SMOOTH_LABELS
    }
    jit = np.abs(rows["label_dev_bins"]) > 1.0
    return dict(
        search=dict(
            intermittent=search(inter),
            always_under_null=search(under),
            present_blocks_moved=_frac(
                (rows["s_star"] != 0) & (rows["p_max"] - rows["p0"] >= 1.0), p_i
            ),
            regime_any_peak=regime_after(rows, rows["p_max"]),
            regime_own_peak=regime_after(rows, own_peak(rows)),
        ),
        comb_loo=dict(
            regime_after(rows, rows["p_loo"]),
            absent_with_estimate=_frac(has_loo, a_i),
            dc_abs_rev_s_absent_quantiles=_qs(np.abs(rows["dc_loo"][a_i & has_loo])),
            dc_abs_rev_s_present_quantiles=_qs(np.abs(rows["dc_loo"][p_i & has_loo])),
            dc_at_zero_absent=_frac(rows["dc_loo"] == 0.0, a_i & has_loo),
        ),
        smoothed_label=smooth,
        label_jitter=dict(
            absent_off_by_gt_1bin=_frac(jit, a_i),
            present_off_by_gt_1bin=_frac(jit, p_i),
        ),
        neighbour_at_label=dict(
            absent=_frac(rows["d_nb_label"] <= NEIGHBOUR_BINS, a_i),
            present=_frac(rows["d_nb_label"] <= NEIGHBOUR_BINS, p_i),
        ),
    )


# ---------------------------------------------------------------------------
# (b) distribution shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gauss:
    """The Gaussian-in-dB hypothesis for one rig."""

    sigma_db: float  # line wander sd (the resolvable lines' sigma_total)
    tau_s: float
    s2_db2: float  # measured block noise of a line
    eta: np.ndarray  # line-free cells over their local floor (linear), the null pool
    sigma_d_db: float  # the rotor-common part of sigma_db


def rig_gauss(rig: str, rows: dict[str, np.ndarray]) -> Gauss:
    det = json.loads((WANDER_DIR / "wander_detail.json").read_text())["rigs"][rig]
    num = det["chosen"]["lag1"]["numbers"]
    s2 = det["noise"][f"{BLOCK_S:g}"]["line_s2_measured_median_db2"]
    per_track = split_tracks(take(rows, rows["cls"] == UNDER), "p0")
    quiet = np.concatenate([v for v in per_track if np.median(v) < NULL_MEDIAN_DB])
    return Gauss(
        sigma_db=float(num["sigma_total_db"]),
        tau_s=float(num["tau_total_s"]),
        s2_db2=float(s2),
        eta=10.0 ** (quiet / 10.0),
        sigma_d_db=float(num["sigma_d_db"]),
    )


def median_map(g: Gauss, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """``(mu, median P)``: the model's median block prominence at line level ``mu``
    (dB over the local floor), on a grid; monotone by construction."""
    n = 20_000
    z = rng.standard_normal(n) * math.sqrt(g.sigma_db**2 + g.s2_db2)
    e = rng.choice(g.eta, n)
    mu = np.arange(-30.0, 40.0001, 0.1)
    med = np.array([np.median(10.0 * np.log10(10.0 ** ((m + z) / 10.0) + e)) for m in mu])
    return mu, np.maximum.accumulate(med)


def line_levels(g: Gauss, tracks: Sequence[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Each track's line level ``mu`` whose model median prominence is the track's."""
    mu_grid, med_grid = median_map(g, rng)
    return np.interp(np.array([np.median(t) for t in tracks]), med_grid, mu_grid)


def simulate(
    g: Gauss,
    mus: np.ndarray,
    ns: np.ndarray,
    rng: np.random.Generator,
    floor: np.ndarray | None = None,
) -> list[np.ndarray]:
    """One forward-model draw of every track: OU(sigma, tau) + block noise in dB,
    minus the track's OBSERVED floor deviation ``floor`` ``(tracks, blocks)`` if
    given (the line held, the floor as measured), plus a line-free cell ratio,
    as block prominence."""
    n_max = int(ns.max())
    x = W.simulate_ou_blocks(rng, mus.size, n_max, g.sigma_db, g.tau_s, BLOCK_S)
    x = x + math.sqrt(g.s2_db2) * rng.standard_normal(x.shape)
    if floor is not None:
        x = x - floor
    p = 10.0 * np.log10(10.0 ** ((mus[:, None] + x) / 10.0) + rng.choice(g.eta, x.shape))
    return [p[i, : int(n)] for i, n in enumerate(ns)]


def floor_matrix(floors: Sequence[np.ndarray]) -> np.ndarray:
    """``(tracks, max blocks)`` the per-track floor deviations, 0 where missing."""
    out = np.zeros((len(floors), max(f.size for f in floors)))
    for i, f in enumerate(floors):
        out[i, : f.size] = np.nan_to_num(f)
    return out


def dbic(x: np.ndarray) -> float:
    """BIC(1 Gaussian) - BIC(2 Gaussians): > 0 favours two components."""
    from sklearn.mixture import GaussianMixture

    xx = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    b1 = GaussianMixture(1, random_state=0).fit(xx).bic(xx)
    b2 = GaussianMixture(2, n_init=3, random_state=0).fit(xx).bic(xx)
    return float(b1 - b2)


def gmm2(x: np.ndarray) -> dict[str, list[float]]:
    from sklearn.mixture import GaussianMixture

    xx = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    m = GaussianMixture(2, n_init=3, random_state=0).fit(xx)
    means = np.asarray(m.means_)[:, 0]
    var = np.asarray(m.covariances_).reshape(-1)
    weights = np.asarray(m.weights_)
    o = np.argsort(means)
    return dict(
        means=[float(v) for v in means[o]],
        sds=[float(math.sqrt(v)) for v in var[o]],
        weights=[float(v) for v in weights[o]],
    )


SHAPE_KEYS = (
    "sd_centred",
    "excess_kurtosis",
    "skew",
    "acf1",
    "switch_rate",
    "track_sd_cv",
    "present",
    "gone",
    "tail",
    "dbic_centred",
    "dbic_raw",
    "inter_share",
)


def _centred(tracks: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([t - t.mean() for t in tracks])


def shape_stats(tracks: Sequence[np.ndarray]) -> dict[str, float]:
    raw = np.concatenate(tracks)
    cen = _centred(tracks)
    mean = np.concatenate([np.full(t.size, t.mean()) for t in tracks])
    sd = float(cen.std())
    z = cen / sd if sd > 0 else cen
    lag1 = sum(float(np.dot(t[1:] - t.mean(), t[:-1] - t.mean())) for t in tracks)
    frac = np.array([(t >= PROM_DB).mean() for t in tracks])
    on = [t >= PROM_DB for t in tracks]
    switches = sum(int(np.sum(o[1:] != o[:-1])) for o in on)
    pairs = sum(max(o.size - 1, 0) for o in on)
    tsd = np.array([t.std() for t in tracks])
    return dict(
        sd_centred=sd,
        excess_kurtosis=float(np.mean(z**4) - 3.0),
        skew=float(np.mean(z**3)),
        acf1=lag1 / float(np.dot(cen, cen)) if sd > 0 else NAN,
        switch_rate=switches / pairs if pairs else NAN,
        track_sd_cv=float(tsd.std() / tsd.mean()) if tsd.mean() > 0 else NAN,
        present=float((raw >= PROM_DB).mean()),
        gone=float((raw < GONE_DB).mean()),
        tail=float((raw < mean - TAIL_DB).mean()),
        dbic_centred=dbic(cen),
        dbic_raw=dbic(raw),
        inter_share=float(((frac >= INTER_LO) & (frac < INTER_HI)).mean()),
    )


def fit_sigma(
    g: Gauss,
    tracks: Sequence[np.ndarray],
    rng: np.random.Generator,
    floor: np.ndarray | None = None,
) -> float:
    """The line wander sd at which the model's pooled centred prominence sd
    matches the observed one (bisection in log sigma, 4 draws per step)."""
    target = float(_centred(tracks).std())
    ns = np.array([t.size for t in tracks])
    lo, hi = 0.05, 20.0
    for _ in range(FIT_STEPS):
        mid = math.sqrt(lo * hi)
        gm = replace(g, sigma_db=mid)
        mus = line_levels(gm, tracks, rng)
        sd = float(np.median([_centred(simulate(gm, mus, ns, rng, floor)).std() for _ in range(4)]))
        lo, hi = (mid, hi) if sd < target else (lo, mid)
    return math.sqrt(lo * hi)


def against(
    g: Gauss,
    tracks: Sequence[np.ndarray],
    obs: dict[str, float],
    rng: np.random.Generator,
    n_sim: int,
    floor: np.ndarray | None = None,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    """``n_sim`` forward-model draws of ``tracks`` under ``g``: the statistics'
    [5, 50, 95] % and the one-sided p of the observed; plus ``HIST_DRAWS`` extra
    draws' tracks for the histograms."""
    mus = line_levels(g, tracks, rng)
    ns = np.array([t.size for t in tracks])
    sims = [shape_stats(simulate(g, mus, ns, rng, floor)) for _ in range(n_sim)]
    null = {k: np.array([s[k] for s in sims]) for k in SHAPE_KEYS}
    out = dict(
        sigma_db=g.sigma_db,
        with_floor=floor is not None,
        line_level_db_quantiles=_qs(mus),
        q05_50_95={k: _qs(v, (0.05, 0.5, 0.95)) for k, v in null.items()},
        p_upper={k: float((1 + np.sum(null[k] >= obs[k])) / (1 + n_sim)) for k in SHAPE_KEYS},
        p_lower={k: float((1 + np.sum(null[k] <= obs[k])) / (1 + n_sim)) for k in SHAPE_KEYS},
    )
    return out, [t for _ in range(HIST_DRAWS) for t in simulate(g, mus, ns, rng, floor)]


def comovement(rows: dict[str, np.ndarray], sel: np.ndarray) -> dict[str, Any]:
    """Mean correlation of the block prominence of two selected tracks of the
    same window (same rotor / different rotors), over their common blocks (>= 5)."""
    sub = take(rows, sel)
    same, diff = [], []
    for wi in np.unique(sub["win"]):
        s = take(sub, sub["win"] == wi)
        n_b = int(s["block"].max()) + 1
        tr = np.unique(s["track"])
        if tr.size < 2:
            continue
        mat = np.full((tr.size, n_b), np.nan)
        rot = np.zeros(tr.size, dtype=np.int64)
        for i, t in enumerate(tr):
            m = s["track"] == t
            mat[i, s["block"][m]] = s["p0"][m]
            rot[i] = s["rotor"][m][0]
        for i in range(tr.size):
            for j in range(i + 1, tr.size):
                r = (
                    _pearson(mat[i], mat[j])
                    if (np.isfinite(mat[i]) & np.isfinite(mat[j])).sum() >= 5
                    else NAN
                )
                if math.isfinite(r):
                    (same if rot[i] == rot[j] else diff).append(r)
    return dict(
        same_rotor_mean_r=float(np.mean(same)) if same else NAN,
        same_rotor_pairs=len(same),
        other_rotor_mean_r=float(np.mean(diff)) if diff else NAN,
        other_rotor_pairs=len(diff),
    )


def track_median(rows: dict[str, np.ndarray]) -> np.ndarray:
    """Each row's track median block prominence."""
    parts = split_tracks(rows, "p0")
    return np.concatenate([np.full(v.size, np.median(v)) for v in parts])


def shape_sets(rows: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Row masks of the track sets (b) is run on. ``near_threshold`` is chosen
    on the track median alone, blind to the scatter the 20-80 % rule selects on."""
    med = track_median(rows)
    lo, hi = NEAR_THRESHOLD_DB
    return dict(
        intermittent=rows["cls"] == INTER,
        near_threshold=(rows["cls"] <= UNDER) & (med >= lo) & (med <= hi),
        resolvable=rows["cls"] == RES,
    )


def shape_test(
    rows: dict[str, np.ndarray],
    sel: np.ndarray,
    g: Gauss,
    rng: np.random.Generator,
    n_sim: int,
) -> dict[str, Any] | None:
    """None when fewer than ``MIN_SHAPE_TRACKS`` tracks are selected."""
    sub = take(rows, sel)
    tracks = split_tracks(sub, "p0")
    if len(tracks) < MIN_SHAPE_TRACKS:
        return None
    fl = floor_matrix(split_tracks(sub, "local"))
    obs = shape_stats(tracks)
    orders = np.array([o[0] for o in split_tracks(sub, "order")])
    tsd = np.array([t.std() for t in tracks])
    by_group = {
        name: dict(
            tracks=int(m.sum()), median_track_sd_db=float(np.median(tsd[m])) if m.any() else NAN
        )
        for name, lo, hi in MW.ORDER_GROUPS
        for m in [(orders >= lo) & (orders <= hi)]
    }
    literal, one_lit = against(g, tracks, obs, rng, n_sim)
    literal_floor, one_litf = against(g, tracks, obs, rng, n_sim, fl)
    g_fit = replace(g, sigma_db=fit_sigma(g, tracks, rng, fl))
    fitted, one_fit = against(g_fit, tracks, obs, rng, n_sim, fl)
    cen = _centred(tracks)
    raw = np.concatenate(tracks)
    edges = np.arange(-16.0, 12.01, 0.5)
    raw_edges = np.arange(-6.0, 26.01, 0.5)

    def hist(x: np.ndarray, e: np.ndarray) -> np.ndarray:
        return np.histogram(x, e)[0]  # values outside the edges are left out, not piled at them

    return dict(
        n_tracks=len(tracks),
        n_blocks=int(raw.size),
        model=dict(
            tau_s=g.tau_s,
            block_noise_db2=g.s2_db2,
            null_pool_blocks=int(g.eta.size),
            null_pool_prominence_db_quantiles=_qs(10.0 * np.log10(g.eta)),
            same_rotor_share=g.sigma_d_db**2 / (g.sigma_db**2 + g.s2_db2),
        ),
        observed=obs,
        gaussian_literal=literal,
        gaussian_literal_floor=literal_floor,
        gaussian_fitted_floor=fitted,
        gmm2_centred=gmm2(cen),
        gmm2_raw=gmm2(raw),
        co_movement=comovement(rows, sel),
        track_sd_by_order_group=by_group,
        _hist=dict(
            edges=edges,
            raw_edges=raw_edges,
            obs=hist(cen, edges),
            obs_raw=hist(raw, raw_edges),
            literal=hist(_centred(one_lit), edges),
            literal_raw=hist(np.concatenate(one_lit), raw_edges),
            literal_floor=hist(_centred(one_litf), edges),
            literal_floor_raw=hist(np.concatenate(one_litf), raw_edges),
            fitted=hist(_centred(one_fit), edges),
            fitted_raw=hist(np.concatenate(one_fit), raw_edges),
        ),
    )


# ---------------------------------------------------------------------------
# (c) floor masking
# ---------------------------------------------------------------------------


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return NAN
    a, b = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / den) if den > 0 else NAN


def _within(
    rows: dict[str, np.ndarray], key: str, other: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Both columns centred per track over the blocks where both are finite,
    and ``rows[key]`` itself on those blocks."""
    xs, ys, raw = [], [], []
    for x, y in zip(split_tracks(rows, key), split_tracks(rows, other)):
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3:
            continue
        xs.append(x[ok] - x[ok].mean())
        ys.append(y[ok] - y[ok].mean())
        raw.append(x[ok])
    if not xs:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(raw)


def floor_held(rows: dict[str, np.ndarray], fkey: str) -> np.ndarray:
    """(c) counterfactual prominence: the line's absolute level held, the floor
    ``rows[fkey]`` (dB about its mean) moved to its mean; NaN where no line."""
    f_all = rows[fkey]
    lin = 10.0 ** (rows["p0"] / 10.0) - 1.0
    return np.where(
        (lin > 0) & np.isfinite(f_all),
        10.0 * np.log10(1.0 + np.maximum(lin, 0.0) * 10.0 ** (np.nan_to_num(f_all) / 10.0)),
        np.nan,
    )


def own_peak(rows: dict[str, np.ndarray]) -> np.ndarray:
    """(a2) the peak-search prominence where the peak is NOT another rotor's line."""
    nb = (rows["d_nb"] <= NEIGHBOUR_BINS) & (rows["d_nb"] < rows["d_own"])
    return np.where(nb, rows["p0"], rows["p_max"])


#: The counterfactuals broken down by order group (the unbiased ones).
GROUP_CF = (
    "peak +-3 bins, own",
    "carrier from other lines",
    "label: 1 s moving mean",
    "floor at mean: local",
    "floor at mean: band u+u_j",
)


def by_order_group(rows: dict[str, np.ndarray]) -> dict[str, Any]:
    """Per order group: the intermittent tracks, their own block sd, and the fate
    of the intermittency under each of :data:`GROUP_CF`."""
    p_new = {
        "peak +-3 bins, own": own_peak(rows),
        "carrier from other lines": rows["p_loo"],
        "label: 1 s moving mean": rows["p_moving_1s"],
        "floor at mean: local": floor_held(rows, "local"),
        "floor at mean: band u+u_j": floor_held(rows, "uj"),
    }
    out: dict[str, Any] = {}
    for name, lo, hi in MW.ORDER_GROUPS:
        m = (rows["order"] >= lo) & (rows["order"] <= hi)
        sub = take(rows, m)
        inter = take(sub, sub["cls"] == INTER)
        n = len(split_tracks(inter, "p0"))
        if n == 0:
            continue
        out[name] = dict(
            tracks=n,
            median_track_sd_db=float(np.median([t.std() for t in split_tracks(inter, "p0")])),
            **{cf: regime_after(sub, p_new[cf][m]) for cf in GROUP_CF},
        )
    return out


def floor_test(
    rows: dict[str, np.ndarray], sigma_u: float, rng: np.random.Generator, n_boot: int
) -> dict[str, Any]:
    inter = rows["cls"] == INTER
    ir = take(rows, inter)
    absent = ir["p0"] < PROM_DB
    present = ~absent
    wins = np.unique(ir["win"])
    out: dict[str, Any] = dict(sigma_u_db=sigma_u)
    for fkey in FLOOR_KEYS:
        per = np.array(
            [
                _pearson(p, f)
                for p, f in zip(split_tracks(ir, "p0"), split_tracks(ir, fkey))
                if np.isfinite(f).sum() >= 5
            ]
        )
        per = per[np.isfinite(per)]
        x, y, p_raw = _within(ir, "p0", fkey)
        boot = []
        for _ in range(n_boot):
            pick = rng.choice(wins, wins.size)
            m = np.concatenate([np.nonzero(ir["win"] == wi)[0] for wi in pick])
            boot.append(_pearson(*_within(take(ir, m), "p0", fkey)[:2]))
        cf = floor_held(rows, fkey)
        f = ir[fkey]
        high = f > sigma_u
        fin = np.isfinite(f)
        out[fkey] = dict(
            tracks=int(per.size),
            r_per_track_quantiles=_qs(per),
            r_per_track_negative=_frac(per < 0),
            r_per_track_below_minus_half=_frac(per < -0.5),
            r_pooled_within=_pearson(x, y),
            r_pooled_within_ci90=_qs(np.array(boot), (0.05, 0.95)),
            slope_db_per_db=float((x @ y) / (y @ y)) if y.size and (y @ y) > 0 else NAN,
            floor_sd_within_track_db=float(y.std()) if y.size else NAN,
            absent_floor_above_sigma_u=_frac(high, absent & fin),
            present_floor_above_sigma_u=_frac(high, present & fin),
            all_floor_above_sigma_u=_frac(high, fin),
            absent_floor_above_mean=_frac(f > 0, absent & fin),
            present_floor_above_mean=_frac(f > 0, present & fin),
            floor_at_mean=regime_after(rows, cf),
            _xy=(x, y, p_raw < PROM_DB),
        )
    x, y, _ = _within(ir, "p0", "carrier")
    out["speed"] = dict(r_pooled_within=_pearson(x, y))
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def pick_examples(
    wins: dict[str, list[Win]], rows: dict[str, dict[str, np.ndarray]]
) -> list[dict[str, Any]]:
    """Deterministic examples: per rig, intermittent tracks of order >= 3 (the
    fundamentals sit in the wind floor) of the longest windows, highest own
    share first, one per window."""
    out: list[dict[str, Any]] = []
    for rig, n in (("dregon", 4), ("michaels", 2)):
        r = rows[rig]
        first = np.r_[True, np.diff(r["track"]) != 0]
        t = take(r, first & (r["cls"] == INTER) & (r["order"] >= 3))
        n_b = np.array([wins[rig][wi].prom.shape[-1] for wi in t["win"]])
        seen: set[int] = set()
        for i in np.lexsort((-t["own_share"], -n_b)):
            wi = int(t["win"][i])
            if wi in seen:
                continue
            seen.add(wi)
            out.append(
                dict(
                    rig=rig,
                    window=wins[rig][wi].name,
                    win=wi,
                    track=int(t["track"][i]),
                    rotor=int(t["rotor"][i]) + 1,
                    order=int(t["order"][i]),
                    own_share=float(t["own_share"][i]),
                )
            )
            if len(seen) >= n:
                break
    return out


def examples_figure(
    picks: Sequence[dict[str, Any]], rows: dict[str, dict[str, np.ndarray]], path: Path
) -> None:
    plt = _plt()
    fig, axes = plt.subplots(len(picks), 1, figsize=(9.0, 2.5 * len(picks)), squeeze=False)
    for ax, p in zip(axes[:, 0], picks):
        r = take(rows[p["rig"]], rows[p["rig"]]["track"] == p["track"])
        t = (r["block"] + 0.5) * BLOCK_S
        ax.axhline(PROM_DB, color="k", lw=0.8, ls="-", alpha=0.6)
        ax.plot(t, r["p0"], "o-", color="C0", lw=1.8, ms=4, label="at the label")
        ax.plot(t, r["p_max"], ":", color="C1", lw=1.2, label="+-3-bin peak")
        ax.plot(
            t, r["p_loo"], "--", color="C3", lw=1.2, label="carrier from the rotor's other lines"
        )
        ax.plot(
            t, r["p_moving_1s"], "-", color="C2", lw=1.0, alpha=0.8, label="label, 1 s moving mean"
        )
        ax.set_ylabel("prominence dB")
        ax2 = ax.twinx()
        ax2.plot(t, r["local"], "-", color="0.35", lw=1.2, label="local floor")
        ax2.plot(t, r["uj"], "--", color="0.6", lw=1.2, label="band floor u + u_j")
        ax2.set_ylabel("floor - track mean, dB", color="0.35")
        rl = _pearson(r["p0"], r["local"])
        rj = _pearson(r["p0"], r["uj"])
        ax.set_title(
            f"{p['rig']}: rotor {p['rotor']}, order {p['order']} (own share {p['own_share']:.2f}) "
            f"-- {p['window']}\nr(prominence, local floor) = {rl:+.2f}, "
            f"r(prominence, band floor) = {rj:+.2f}",
            fontsize=8.5,
        )
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel(f"time in window (s), {BLOCK_S:g} s blocks; black line: {PROM_DB:g} dB")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color="0.35"), plt.Line2D([], [], color="0.6", ls="--")]
    labels += ["local floor (right axis)", "band floor u + u_j (right axis)"]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.55 / (2.5 * len(picks))))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def hist_figure(rigs: dict[str, Any], path: Path) -> None:
    plt = _plt()
    panels = (
        ("intermittent", False),
        ("intermittent", True),
        ("near_threshold", False),
        ("resolvable", False),
    )
    fig, axes = plt.subplots(
        len(rigs), len(panels), figsize=(4.4 * len(panels), 3.8 * len(rigs)), squeeze=False
    )
    for row, (rig, d) in zip(axes, rigs.items()):
        for ax, (set_name, raw) in zip(row, panels):
            sh = d["shape"][set_name]
            if sh is None:
                ax.set_axis_off()
                ax.set_title(f"{rig} {set_name}: too few tracks", fontsize=9)
                continue
            h = sh["_hist"]
            e = h["raw_edges"] if raw else h["edges"]
            x = 0.5 * (e[1:] + e[:-1])
            w = e[1] - e[0]
            sfx = "_raw" if raw else ""
            for key, style, lab in (
                (
                    "literal",
                    dict(color="C0", lw=1.4),
                    "(1) Gaussian at the resolvable lines' sigma",
                ),
                (
                    "literal_floor",
                    dict(color="C2", lw=1.4, ls="-."),
                    "(2) the same + measured local floor",
                ),
                (
                    "fitted",
                    dict(color="C1", lw=1.5, ls="--"),
                    "(3) + floor, sigma fitted to the scatter",
                ),
            ):
                c = h[key + sfx]
                ax.plot(x, c / max(c.sum(), 1) / w, **style, label=lab)
            c = h["obs" + sfx]
            ax.step(x, c / max(c.sum(), 1) / w, where="mid", color="k", lw=1.6, label="observed")
            if raw:
                ax.axvline(PROM_DB, color="0.5", lw=0.8)
                ax.set_xlabel("block prominence (dB)")
            else:
                ax.set_yscale("log")
                ax.set_ylim(1e-4, None)
                ax.set_xlabel("block prominence - track mean (dB)")
            ax.set_title(
                f"{rig} {set_name.replace('_', '-')}{' (raw)' if raw else ''}: "
                f"{sh['n_tracks']} tracks\nsigma (1),(2) {sh['gaussian_literal']['sigma_db']:.2f} dB, "
                f"(3) {sh['gaussian_fitted_floor']['sigma_db']:.2f} dB",
                fontsize=9,
            )
            ax.set_ylabel("density")
            ax.grid(alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


COUNTERFACTUALS: tuple[tuple[str, str], ...] = (
    ("peak +-3 bins, any", "label.search.regime_any_peak"),
    ("peak +-3 bins, own", "label.search.regime_own_peak"),
    ("carrier from other lines", "label.comb_loo"),
    ("label: window mean", "label.smoothed_label.window_mean"),
    ("label: 1 s moving mean", "label.smoothed_label.moving_1s"),
    ("floor at mean: local", "floor.local.floor_at_mean"),
    ("floor at mean: band u+u_j", "floor.uj.floor_at_mean"),
    ("floor at mean: level u", "floor.u.floor_at_mean"),
)


def _dig(d: dict[str, Any], path: str) -> dict[str, Any]:
    for k in path.split("."):
        d = d[k]
    return d


def counterfactual_figure(rigs: dict[str, Any], path: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, len(rigs), figsize=(6.5 * len(rigs), 4.6), squeeze=False)
    names = [n for n, _ in COUNTERFACTUALS]
    y = np.arange(len(names))
    for ax, (rig, d) in zip(axes[0], rigs.items()):
        vals = [_dig(d, p) for _, p in COUNTERFACTUALS]
        for off, key, col, lab in (
            (-0.27, "absent_recovered", "C2", "absent blocks that return"),
            (0.0, "present_lost", "C3", "present blocks that go"),
            (0.27, "still_intermittent", "C0", "tracks still intermittent"),
        ):
            ax.barh(y + off, [100 * v[key] for v in vals], height=0.26, color=col, label=lab)
        ax.set_yticks(y, names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("%")
        ax.set_title(f"{rig}: remove one mechanism, re-measure the intermittent tracks", fontsize=9)
        ax.grid(axis="x", alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def shift_figure(rows: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, len(rows), figsize=(6.0 * len(rows), 3.6), squeeze=False)
    s_all = np.arange(-SHIFT_BINS, SHIFT_BINS + 1)
    for ax, (rig, r) in zip(axes[0], rows.items()):
        absent = r["p0"] < PROM_DB
        rec = r["p_max"] >= PROM_DB
        nb = (r["d_nb"] <= NEIGHBOUR_BINS) & (r["d_nb"] < r["d_own"])
        for off, sel, col, lab in (
            (-0.2, r["cls"] == INTER, "C0", "intermittent tracks"),
            (0.2, r["cls"] == UNDER, "0.6", "always-under tracks (null)"),
        ):
            a = sel & absent
            own = [100 * np.mean((a & rec & ~nb & (r["s_star"] == s))[a]) for s in s_all]
            oth = [100 * np.mean((a & rec & nb & (r["s_star"] == s))[a]) for s in s_all]
            ax.bar(s_all + off, own, width=0.38, color=col, label=f"{lab}: own")
            ax.bar(
                s_all + off,
                oth,
                width=0.38,
                bottom=own,
                color=col,
                alpha=0.4,
                hatch="//",
                label=f"{lab}: another rotor's line",
            )
        ax.set_xlabel("peak offset from the label bin (bins)")
        ax.set_ylabel("% of absent blocks reaching 6 dB")
        ax.set_title(f"{rig}: where the +-3-bin search finds 6 dB", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def floor_figure(rigs: dict[str, Any], path: Path) -> None:
    plt = _plt()
    keys = ("local", "uj")
    fig, axes = plt.subplots(
        len(rigs), len(keys), figsize=(5.2 * len(keys), 4.0 * len(rigs)), squeeze=False
    )
    for row, (rig, d) in zip(axes, rigs.items()):
        for ax, key in zip(row, keys):
            fl = d["floor"][key]
            x, y, ab = fl["_xy"]
            ax.scatter(y[~ab], x[~ab], s=3, alpha=0.25, color="C0", label="present")
            ax.scatter(y[ab], x[ab], s=3, alpha=0.25, color="C3", label="absent (< 6 dB)")
            lim = np.nanquantile(np.abs(y), 0.995) if y.size else 1.0
            g = np.linspace(-lim, lim, 2)
            ax.plot(
                g,
                fl["slope_db_per_db"] * g,
                "k-",
                lw=1.2,
                label=f"slope {fl['slope_db_per_db']:.2f}",
            )
            ref = -(1.0 - 10.0 ** (-PROM_DB / 10.0))  # dP/dfloor of a held line at 6 dB
            ax.plot(
                g,
                ref * g,
                color="0.5",
                ls=":",
                lw=1,
                label=f"line held, floor moves (slope {ref:.2f} at {PROM_DB:g} dB)",
            )
            ax.set_xlim(-lim, lim)
            ax.set_xlabel(
                f"{'local floor' if key == 'local' else 'band floor u + u_j'} - track mean (dB)"
            )
            ax.set_ylabel("prominence - track mean (dB)")
            ax.set_title(
                f"{rig}: r = {fl['r_pooled_within']:+.2f} pooled within tracks ({fl['tracks']} tracks)",
                fontsize=9,
            )
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _f(v: Any, fmt: str = "{:.2f}") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return fmt.format(v)


def _pc(v: Any) -> str:
    return _f(None if v is None else 100.0 * v, "{:.1f} %")


def _p(v: Any) -> str:
    return _f(v, "{:.3f}")


SHAPE_ROWS: tuple[tuple[str, str, str], ...] = (
    ("sd_centred", "sd of block prominence around the track mean, dB", "{:.2f}"),
    ("excess_kurtosis", "excess kurtosis (centred)", "{:.2f}"),
    ("skew", "skewness (centred)", "{:.2f}"),
    ("acf1", "lag-1 autocorrelation (centred, pooled)", "{:.2f}"),
    ("switch_rate", "present/absent switches per block step", "{:.2f}"),
    ("track_sd_cv", "spread of the tracks' own sds (sd / mean over tracks)", "{:.2f}"),
    ("present", "blocks >= 6 dB", "pc"),
    ("gone", f"blocks < {GONE_DB:g} dB", "pc"),
    ("tail", f"blocks > {TAIL_DB:g} dB under the track mean", "pc"),
    ("dbic_centred", "BIC(1) - BIC(2 Gaussians), centred", "{:.1f}"),
    ("dbic_raw", "BIC(1) - BIC(2 Gaussians), raw", "{:.1f}"),
    ("inter_share", "tracks intermittent by the 20-80 % rule", "pc"),
)


def _fmt(v: Any, fmt: str) -> str:
    return _pc(v) if fmt == "pc" else _f(v, fmt)


def findings_md(payload: dict[str, Any], figures: Sequence[str]) -> str:
    rigs = payload["rigs"]
    out = [
        "# DREGON's intermittent rotor lines: label, amplitude or floor?",
        "",
        f"`scripts/_dregon_intermittency.py` at `{payload['git_head']}`; numbers in "
        "`intermittency.json`. A laptop diagnostic, no model is fitted. The wander measurement "
        "(`results/noise_v3/wander/findings.md`) read the line sigma (1.48 dB on DREGON) off "
        "RESOLVABLE tracks only; this looks at the tracks it left out, on the same windows "
        "(`_mic_gain_rank.rig_specs(extra=True)`), the same 2048/512 front end and the same "
        "block line measurement (`wander.measure_lines`, 0.5 s blocks).",
        "",
        "**Definitions.** A (window, rotor, k) track is classified when it is valid (in band, on "
        f"the grid) in >= {MIN_VALID_FRAC:.0%} of the window's blocks: RESOLVABLE when its block "
        f"prominence (mic-summed 3 label-tracked cells over the mic-summed local q25 floor) is "
        f">= {PROM_DB:g} dB in >= {INTER_HI:.0%} of its valid blocks, INTERMITTENT at "
        f"{INTER_LO:.0%}-{INTER_HI:.0%}, ALWAYS-UNDER below {INTER_LO:.0%}. An ABSENT block is one "
        f"under {PROM_DB:g} dB. DREGON's label is the commanded speed (`motors_command`); "
        "Michael's is the audio-refined `rps_refined` -- the control.",
        "",
        "## Counts",
        "",
        "| rig | windows | blocks | resolvable | intermittent | always-under | off grid | "
        "intermittent blocks | of them absent | distinct lines | rotor-dominant (own share >= 0.8) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rig, d in rigs.items():
        c = d["counts"]
        t = c["tracks"]
        out.append(
            f"| {rig} | {c['n_windows']} | {c['n_blocks']} | {t['resolvable']} | "
            f"{t['intermittent']} | {t['under']} | {t['off_grid']} | {c['intermittent_blocks']} | "
            f"{c['intermittent_absent_blocks']} | {c['intermittent_distinct_lines']} | "
            f"{c['intermittent_dominant']} |"
        )
    out += ["", "Intermittent tracks by order group and rotor:", ""]
    groups = [g for g, _, _ in MW.ORDER_GROUPS]
    out += [
        "| rig | "
        + " | ".join(f"k {g}" for g in groups)
        + " | rotor 1 | rotor 2 | rotor 3 | rotor 4 |",
        "| --- |" + " --- |" * (len(groups) + 4),
    ]
    for rig, d in rigs.items():
        c = d["counts"]
        out.append(
            f"| {rig} | "
            + " | ".join(str(c["intermittent_by_order_group"][g]) for g in groups)
            + " | "
            + " | ".join(str(c["intermittent_by_rotor"][str(r)]) for r in range(1, 5))
            + " |"
        )
    out.append("")
    for rig, d in rigs.items():
        top = ", ".join(
            f"{x['line']} ({x['windows']})" for x in d["counts"]["intermittent_most_windows"]
        )
        out.append(f"- {rig}, the lines intermittent in the most windows (rotor:order): {top}")

    out += [
        "",
        "## The three tests, as counterfactuals",
        "",
        "Each mechanism is removed in turn and the intermittent tracks re-measured: the "
        "share of their absent blocks that return (>= 6 dB), of their present blocks that "
        "go, of the tracks still intermittent by the same rule, and the present/absent "
        "switches between consecutive blocks against the observed. A mechanism that "
        "causes the intermittency leaves few tracks intermittent once removed: a carrier "
        "that fixes a wrong label returns absent blocks without losing present ones; a floor "
        "held still may leave a line steadily present or steadily under (seen only in floor "
        "dips). The +-3-bin peak search picks the best of seven positions per block, so its "
        "row is an upper bound, not an estimate.",
        "",
        "| rig | counterfactual | absent -> present | present -> absent | still intermittent "
        "| now resolvable | now always-under | switches / observed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rig, d in rigs.items():
        for name, path in COUNTERFACTUALS:
            v = _dig(d, path)
            out.append(
                f"| {rig} | {name} | {_pc(v['absent_recovered'])} | {_pc(v['present_lost'])} | "
                f"{_pc(v['still_intermittent'])} | {_pc(v['now_resolvable'])} | "
                f"{_pc(v['now_under'])} | {_f(v['switches_over_observed'])} |"
            )

    out += [
        "",
        "By order group, the tracks still intermittent after each counterfactual (and the "
        "median of the intermittent tracks' own block sd):",
        "",
        "| rig | orders | intermittent tracks | own block sd dB | " + " | ".join(GROUP_CF) + " |",
        "| --- | --- | --- | --- |" + " --- |" * len(GROUP_CF),
    ]
    for rig, d in rigs.items():
        for g, v in d["by_order_group"].items():
            out.append(
                f"| {rig} | {g} | {v['tracks']} | {_f(v['median_track_sd_db'])} | "
                + " | ".join(
                    f"{_pc(v[cf]['still_intermittent'])} ({_pc(v[cf]['now_under'])} under)"
                    for cf in GROUP_CF
                )
                + " |"
            )
    out += [
        "",
        "## (a) Label displacement",
        "",
        "- **Peak search** (the prominence of `measure_lines` on the periodogram shifted by "
        f"-{SHIFT_BINS}..+{SHIFT_BINS} bins, per block): a recovered peak belongs to ANOTHER rotor "
        f"when that rotor's nearest order (block-mean label) lies within {NEIGHBOUR_BINS:g} bin of "
        "the peak bin and nearer than the own label. The null is the same search on the "
        "always-under tracks.",
        f"- **Carrier from the rotor's other lines**: per block, the offset (grid "
        f"{COMB_STEP_REV_S:g} rev/s, up to {COMB_MAX_REV_S:g} rev/s or half the gap to the nearest "
        "rotor) that maximises the mean prominence of the rotor's OTHER resolvable or "
        f"intermittent orders >= {COMB_MIN_ORDER}, applied to the track (leave-one-out: no "
        "selection on the track itself).",
        "- **Smoothed label**: the rotor's window-mean carrier, and a centred 1 s moving mean "
        "of the frame carriers (a motor's inertia low-passes the command).",
        "",
        "| rig | tracks | absent blocks | any peak >= 6 dB | on another rotor's line | own | "
        "own at +-3 (search edge) | own shift rev/s, median [IQR] | present blocks whose peak moves (>= 1 dB) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rig, d in rigs.items():
        s = d["label"]["search"]
        for set_name, key in (
            ("intermittent", "intermittent"),
            ("always-under (null)", "always_under_null"),
        ):
            v = s[key]
            q = v["own_shift_abs_rev_s_quantiles"]
            out.append(
                f"| {rig} | {set_name} | {v['absent_blocks']} | {_pc(v['recovered'])} | "
                f"{_pc(v['recovered_other_rotor'])} | {_pc(v['recovered_own'])} | "
                f"{_pc(v['own_shift_at_edge'])} | {_f(q[2])} [{_f(q[1])}, {_f(q[3])}] | "
                + (_pc(s["present_blocks_moved"]) if key == "intermittent" else "")
                + " |"
            )
    out += ["", "Own-peak offsets of the recovered absent blocks (bins, -3..+3):", ""]
    for rig, d in rigs.items():
        for set_name, key in (
            ("intermittent", "intermittent"),
            ("always-under", "always_under_null"),
        ):
            cnt = d["label"]["search"][key]["shift_bins_counts"]
            out.append(f"- {rig} {set_name}: " + ", ".join(f"{k}: {v}" for k, v in cnt.items()))
    out += [""]
    for rig, d in rigs.items():
        lb = d["label"]
        cl = lb["comb_loo"]
        qa, qp = cl["dc_abs_rev_s_absent_quantiles"], cl["dc_abs_rev_s_present_quantiles"]
        out.append(
            f"- {rig}: the carrier re-estimate exists for {_pc(cl['absent_with_estimate'])} of the "
            f"absent blocks; its |offset| is {_f(qa[2])} rev/s median on absent blocks and "
            f"{_f(qp[2])} on present ones (zero on {_pc(cl['dc_at_zero_absent'])} of absent). "
            f"The label sits > 1 bin off its window mean in {_pc(lb['label_jitter']['absent_off_by_gt_1bin'])} "
            f"of absent and {_pc(lb['label_jitter']['present_off_by_gt_1bin'])} of present blocks; "
            f"another rotor's order lies within {NEIGHBOUR_BINS:g} bin of the label in "
            f"{_pc(lb['neighbour_at_label']['absent'])} of absent and "
            f"{_pc(lb['neighbour_at_label']['present'])} of present blocks."
        )

    out += [
        "",
        "## (b) Distribution shape",
        "",
        "The Gaussian hypothesis, forward-modelled to the observable: per track, the line "
        "level over the local floor is an OU in dB with the resolvable lines' `(sigma_total, "
        "tau_total)` (`results/noise_v3/wander/wander_detail.json`) plus the rig's measured "
        "block noise, the floor cells add the empirical cell-over-floor ratio of line-free "
        f"blocks (always-under tracks with window-median prominence < {NULL_MEDIAN_DB:g} dB), and "
        "the sum is read as block prominence; each track's level is set so the model's median "
        "prominence is the track's. Three versions: (1) that, at the resolvable lines' sigma -- "
        "the literal test; (2) the same with the track's MEASURED local floor deviation "
        "subtracted block by block (the line held in absolute level, the floor as it was: "
        "what v3's line wander plus its floor predicts); (3) as (2) with the line sigma at "
        "which the model's centred sd matches the observed one -- the line's OWN wander once "
        "the floor is accounted, and a shape test at matched width. "
        "Three track sets: the intermittent ones; NEAR-THRESHOLD -- every classified track "
        f"whose median block prominence is {NEAR_THRESHOLD_DB[0]:g}-{NEAR_THRESHOLD_DB[1]:g} dB, "
        "whatever its class, a set chosen blind to the scatter the 20-80 % rule selects on; and "
        "the resolvable ones, the calibration (the wander sigma was measured on them). "
        f"{payload['config']['n_sim']} draws each; `p` is the share of draws at least as "
        "extreme as the observed on the side it departs to (floor 1/(draws + 1)).",
        "",
    ]
    for rig, d in rigs.items():
        for set_name in SHAPE_SETS:
            sh = d["shape"][set_name]
            if sh is None:
                out += [f"**{rig}, {set_name}**: under {MIN_SHAPE_TRACKS} tracks, not tested.", ""]
                continue
            nulls = (
                sh["gaussian_literal"],
                sh["gaussian_literal_floor"],
                sh["gaussian_fitted_floor"],
            )
            out += [
                f"**{rig}, {set_name}** ({sh['n_tracks']} tracks, {sh['n_blocks']} blocks; "
                f"block noise {_f(sh['model']['block_noise_db2'])} dB^2, tau {_f(sh['model']['tau_s'])} s):",
                "",
                f"| statistic | observed | (1) sigma {_f(nulls[0]['sigma_db'])} dB [5, 50, 95 %] | p "
                f"| (2) sigma {_f(nulls[1]['sigma_db'])} dB + floor | p "
                f"| (3) sigma fitted {_f(nulls[2]['sigma_db'])} dB + floor | p |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
            for key, lab, fmt in SHAPE_ROWS:
                o = sh["observed"][key]
                cells = []
                for g in nulls:
                    q = g["q05_50_95"][key]
                    p = min(g["p_upper"][key], g["p_lower"][key])
                    cells.append(
                        f"{_fmt(q[1], fmt)} [{_fmt(q[0], fmt)}, {_fmt(q[2], fmt)}] | {_p(p)}"
                    )
                out.append(f"| {lab} | {_fmt(o, fmt)} | " + " | ".join(cells) + " |")
            gc, gr = sh["gmm2_centred"], sh["gmm2_raw"]
            cm = sh["co_movement"]
            out += [
                "",
                f"Two-Gaussian fits: centred means {_f(gc['means'][0])} / {_f(gc['means'][1])} dB, "
                f"sds {_f(gc['sds'][0])} / {_f(gc['sds'][1])}, weights {_f(gc['weights'][0])} / "
                f"{_f(gc['weights'][1])}; raw means {_f(gr['means'][0])} / {_f(gr['means'][1])} dB, "
                f"sds {_f(gr['sds'][0])} / {_f(gr['sds'][1])}, weights {_f(gr['weights'][0])} / "
                f"{_f(gr['weights'][1])}. Co-movement of two tracks of one window: mean r "
                f"{_f(cm['same_rotor_mean_r'])} for the same rotor ({cm['same_rotor_pairs']} pairs), "
                f"{_f(cm['other_rotor_mean_r'])} for different rotors ({cm['other_rotor_pairs']} pairs); "
                f"the v3 model's rotor-common share of a line's block variance is "
                f"{_f(sh['model']['same_rotor_share'])} (different rotors 0). A track's own "
                "block sd, median by order group: "
                + ", ".join(
                    f"k {g} {_f(v['median_track_sd_db'])} dB ({v['tracks']})"
                    for g, v in sh["track_sd_by_order_group"].items()
                )
                + ".",
                "",
            ]

    out += [
        "## (c) Floor masking",
        "",
        "Three floors, each centred on its own mean over the track's blocks (window for `u`): "
        "`local` -- the prominence's own denominator (q25 of the cells 3-12 bins either side, "
        "mic-summed; it shares the floor estimate's noise with the prominence, which biases "
        "its r slightly negative); `u+u_j` -- `measure_floor`'s comb-masked control band that "
        "holds the line, speed law removed (the v3 floor at the line); `u` -- the mean over "
        "bands of their deviations (the floor level latent). Pearson r of block prominence "
        "against the floor per track (>= 5 blocks) and pooled within tracks with a 90 % window "
        "bootstrap; the share of blocks whose floor sits > +1 sigma_u over its mean (sigma_u "
        "from `results/noise_v3/wander/<rig>.json`).",
        "",
        "| rig | floor | tracks | r per track, median [IQR] | r < 0 | r < -0.5 | r pooled within [90 %] "
        "| slope dB/dB | floor sd within track dB | floor > +1 sigma_u: absent / present / all | "
        "floor above its mean: absent / present |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rig, d in rigs.items():
        fl = d["floor"]
        for key in FLOOR_KEYS:
            v = fl[key]
            q = v["r_per_track_quantiles"]
            ci = v["r_pooled_within_ci90"]
            out.append(
                f"| {rig} (sigma_u {_f(fl['sigma_u_db'])}) | {key} | {v['tracks']} | {_f(q[2])} "
                f"[{_f(q[1])}, {_f(q[3])}] | {_pc(v['r_per_track_negative'])} | "
                f"{_pc(v['r_per_track_below_minus_half'])} | {_f(v['r_pooled_within'])} "
                f"[{_f(ci[0])}, {_f(ci[1])}] | {_f(v['slope_db_per_db'])} | "
                f"{_f(v['floor_sd_within_track_db'])} | {_pc(v['absent_floor_above_sigma_u'])} / "
                f"{_pc(v['present_floor_above_sigma_u'])} / {_pc(v['all_floor_above_sigma_u'])} | "
                f"{_pc(v['absent_floor_above_mean'])} / {_pc(v['present_floor_above_mean'])} |"
            )
    out += [""]
    for rig, d in rigs.items():
        out.append(
            f"- {rig}: the block prominence against the rotor's own block label speed, pooled "
            f"within tracks: r = {_f(d['floor']['speed']['r_pooled_within'])}."
        )
    out += ["", "## Verdict", "", payload["verdict"], "", "## Figures", ""]
    out += [f"- `{f}`" for f in figures]
    out += ["", "Detail: `intermittency.json`."]
    return "\n".join(out) + "\n"


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--limit", type=int, default=None, help="windows per rig (smoke runs)")
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--n-sim", type=int, default=N_SIM)
    ap.add_argument("--cache", type=Path, default=None, help="pickle of the measured windows")
    args = ap.parse_args(list(argv) if argv is not None else None)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cache is not None and args.cache.exists():
        wins = pickle.loads(args.cache.read_bytes())
    else:
        specs = MW.rig_rows()
        wins = {}
        for rig in RIGS:
            sel = specs[rig][: args.limit] if args.limit else specs[rig]
            print(f"[{rig}] measuring {len(sel)} windows ...", flush=True)
            wins[rig] = measure_rig(sel, args.workers)
        if args.cache is not None:
            args.cache.write_bytes(pickle.dumps(wins))

    rng = np.random.default_rng(SIM_SEED)
    rigs: dict[str, Any] = {}
    all_rows: dict[str, dict[str, np.ndarray]] = {}
    for rig in RIGS:
        rows = rig_rows(wins[rig])
        all_rows[rig] = rows
        g = rig_gauss(rig, rows)
        sigma_u = float(json.loads((WANDER_DIR / f"{rig}.json").read_text())["sigma_u_db"])
        print(f"[{rig}] counts, label, floor ...", flush=True)
        rigs[rig] = dict(
            windows=[
                dict(name=w.name, tag=w.tag, regime=w.regime, blocks=int(w.prom.shape[-1]))
                for w in wins[rig]
            ],
            counts=counts(wins[rig], rows),
            label=label_test(rows),
            floor=floor_test(rows, sigma_u, rng, N_BOOT),
            by_order_group=by_order_group(rows),
        )
        print(f"[{rig}] shape ({args.n_sim} draws) ...", flush=True)
        rigs[rig]["shape"] = {
            name: shape_test(rows, sel, g, rng, args.n_sim)
            for name, sel in shape_sets(rows).items()
        }
    payload: dict[str, Any] = dict(
        schema=SCHEMA,
        generated_by="scripts/_dregon_intermittency.py",
        git_head=git_head(),
        config=dict(
            block_s=BLOCK_S,
            prominence_db=PROM_DB,
            intermittent_frac=[INTER_LO, INTER_HI],
            min_valid_frac=MIN_VALID_FRAC,
            shift_bins=SHIFT_BINS,
            neighbour_bins=NEIGHBOUR_BINS,
            comb_step_rev_s=COMB_STEP_REV_S,
            comb_max_rev_s=COMB_MAX_REV_S,
            comb_min_order=COMB_MIN_ORDER,
            smooth_labels=list(SMOOTH_LABELS),
            n_sim=args.n_sim,
            null_median_db=NULL_MEDIAN_DB,
            tail_db=TAIL_DB,
            gone_db=GONE_DB,
            near_threshold_db=list(NEAR_THRESHOLD_DB),
            limit=args.limit,
        ),
        rigs=rigs,
    )
    picks = pick_examples(wins, all_rows)
    figures = [
        "example_tracks.png",
        "prominence_hist.png",
        "counterfactuals.png",
        "peak_search_offsets.png",
        "floor_scatter.png",
    ]
    examples_figure(picks, all_rows, out_dir / figures[0])
    hist_figure(rigs, out_dir / figures[1])
    counterfactual_figure(rigs, out_dir / figures[2])
    shift_figure(all_rows, out_dir / figures[3])
    floor_figure(rigs, out_dir / figures[4])
    payload["figures"] = figures
    payload["example_tracks"] = picks
    clean = _clean(payload)
    clean["verdict"] = verdict(clean["rigs"])
    (out_dir / "intermittency.json").write_text(json.dumps(clean, indent=1) + "\n")
    (out_dir / "findings.md").write_text(findings_md(clean, figures))
    print(f"wrote {out_dir}/intermittency.json, findings.md, {len(figures)} figures", flush=True)
    print(clean["verdict"], flush=True)
    return 0


#: A counterfactual EXPLAINS the intermittency when it leaves fewer than this
#: share of the tracks intermittent. A label re-estimate must also return more
#: absent blocks than it loses present ones (a worse carrier loses the line in
#: every block and "removes" the intermittency that way); holding the floor
#: still may leave a line steadily present or steadily under.
EXPLAINS_STILL = 0.5
#: Two raw-prominence Gaussians are two STATES when their means are this many
#: pooled sds apart (a 50/50 mixture turns bimodal at 2).
TWO_STATE_SEPARATION = 2.0
ALPHA = 0.05


def _label_explains(v: dict[str, Any]) -> bool:
    return v["still_intermittent"] < EXPLAINS_STILL and v["absent_recovered"] > v["present_lost"]


def _floor_explains(v: dict[str, Any]) -> bool:
    return v["still_intermittent"] < EXPLAINS_STILL


def _best(d: dict[str, Any], names: Sequence[str]) -> tuple[str, dict[str, Any]]:
    """The counterfactual among ``names`` leaving the fewest tracks intermittent."""
    rows = [(n, _dig(d, p)) for n, p in COUNTERFACTUALS if n in names]
    return min(rows, key=lambda nv: nv[1]["still_intermittent"])


LABEL_UNBIASED = ("carrier from other lines", "label: window mean", "label: 1 s moving mean")
FLOOR_CF = ("floor at mean: local", "floor at mean: band u+u_j", "floor at mean: level u")


def amplitude_kind(sh: dict[str, Any]) -> tuple[str, float]:
    """``two-state`` / ``heavy-tailed`` / ``wider Gaussian`` against the
    fitted-sigma Gaussian, and the raw two-Gaussian separation."""
    fit = sh["gaussian_fitted_floor"]
    gr = sh["gmm2_raw"]
    sep = abs(gr["means"][1] - gr["means"][0]) / math.sqrt(
        0.5 * (gr["sds"][0] ** 2 + gr["sds"][1] ** 2)
    )
    bic_sig = min(fit["p_upper"]["dbic_raw"], fit["p_upper"]["dbic_centred"]) < ALPHA
    if bic_sig and sep >= TWO_STATE_SEPARATION:
        return "two-state", sep
    tail_sig = max(fit["p_upper"]["tail"], fit["p_upper"]["excess_kurtosis"]) < ALPHA
    return ("heavy-tailed" if tail_sig else "wider Gaussian"), sep


def _amp_sentence(
    sh: dict[str, Any], near: dict[str, Any] | None, res: dict[str, Any] | None
) -> str:
    """The (b) numbers of one rig."""
    lit, litf, fit = (
        sh["gaussian_literal"],
        sh["gaussian_literal_floor"],
        sh["gaussian_fitted_floor"],
    )
    kind, sep = amplitude_kind(sh)
    obs = sh["observed"]

    def med(g: dict[str, Any], k: str) -> float:
        return g["q05_50_95"][k][1]

    out = (
        f"around its track mean the prominence scatters by {_f(obs['sd_centred'])} dB; a "
        f"Gaussian at the resolvable lines' sigma {_f(lit['sigma_db'])} dB predicts "
        f"{_f(med(lit, 'sd_centred'))} dB, {_f(med(litf, 'sd_centred'))} with the measured local "
        f"floor added, and would call only {_pc(med(litf, 'inter_share'))} of these tracks "
        "intermittent"
    )
    if near:
        out += (
            f"; the near-threshold tracks, chosen on their median alone "
            f"({near['n_tracks']} tracks), scatter by {_f(near['observed']['sd_centred'])} dB "
            f"against {_f(med(near['gaussian_literal_floor'], 'sd_centred'))}, so the excess "
            "is not the 20-80 % rule selecting noisy tracks; and their own sds spread by "
            f"{_f(near['observed']['track_sd_cv'])} (sd / mean) against "
            f"{_f(med(near['gaussian_fitted_floor'], 'track_sd_cv'))} for one width-matched "
            f"sigma ({_f(near['gaussian_fitted_floor']['sigma_db'])} dB)"
            + (
                ": the lines differ in how much they wander"
                if near["observed"]["track_sd_cv"]
                > near["gaussian_fitted_floor"]["q05_50_95"]["track_sd_cv"][2]
                else ", consistent with one sigma"
            )
        )
    out += (
        f". The line's own sigma, fitted with the floor in, is {_f(fit['sigma_db'])} dB"
        + (
            f" against {_f(res['gaussian_fitted_floor']['sigma_db'])} dB for the resolvable "
            "lines in the same observable"
            if res
            else ""
        )
        + f". At that width the shape reads '{kind}': excess kurtosis "
        f"{_f(obs['excess_kurtosis'])} vs {_f(med(fit, 'excess_kurtosis'))} (p "
        f"{_p(min(fit['p_upper']['excess_kurtosis'], fit['p_lower']['excess_kurtosis']))}); no "
        f"second mode (raw BIC gain of two Gaussians {_f(obs['dbic_raw'], '{:.0f}')} vs "
        f"{_f(med(fit, 'dbic_raw'), '{:.0f}')} for the Gaussian, the two components "
        f"{_f(sep)} sds apart); blocks > {TAIL_DB:g} dB under the track mean "
        f"{_pc(obs['tail'])} vs {_pc(med(fit, 'tail'))}, blocks under {GONE_DB:g} dB "
        f"{_pc(obs['gone'])} vs {_pc(med(fit, 'gone'))}; lag-1 autocorrelation "
        f"{_f(obs['acf1'])} vs {_f(med(fit, 'acf1'))}"
    )
    return out


def _floor_by_group(d: dict[str, Any]) -> str:
    """Where the floor matters: the lowest order group against the orders >= 9."""
    groups = {g: lo for g, lo, _ in MW.ORDER_GROUPS}
    bg = d["by_order_group"]
    if not bg:
        return ""
    first = next(iter(bg))
    v0 = bg[first]["floor at mean: local"]
    high = [
        v["floor at mean: local"]["still_intermittent"] for g, v in bg.items() if groups[g] >= 9
    ]
    if not high:
        return ""
    return (
        f"; by order group it matters most at the bottom -- holding the local floor leaves "
        f"{_pc(v0['still_intermittent'])} of the k {first} tracks intermittent "
        f"({_pc(v0['now_under'])} steadily under), against {_pc(min(high))}-{_pc(max(high))} at "
        "k >= 9"
    )


def verdict(rigs: dict[str, Any]) -> str:
    """One paragraph, every number read off ``rigs``; DREGON decides, Michael's is the control."""
    d = rigs["dregon"]
    m = rigs.get("michaels")
    lab_name, lab = _best(d, LABEL_UNBIASED)
    fl_name, fl = _best(d, FLOOR_CF)
    own = d["label"]["search"]["intermittent"]
    null = d["label"]["search"]["always_under_null"]
    sh = d["shape"]["intermittent"]
    kind, _ = amplitude_kind(sh)
    loc = d["floor"]["local"]
    label_ok, floor_ok = _label_explains(lab), _floor_explains(fl)
    if label_ok and (not floor_ok or lab["still_intermittent"] <= fl["still_intermittent"]):
        dominant = "(a) the label"
        implication = (
            "the amplitude process is not the problem: the fit needs the CARRIER DEVIATION "
            "track (a per-block offset of the label), not a new amplitude component"
        )
    elif floor_ok:
        dominant = "(c) floor masking"
        implication = (
            "a Gaussian OU is SUFFICIENT: the lines vanish under the floor's own wander, "
            "which the v3 floor latents (u, u_j) already carry"
        )
    else:
        dominant = f"(b) the line amplitude itself ({kind})"
        implication = {
            "two-state": "the amplitude process NEEDS A SWITCHING COMPONENT (an on/off state "
            "per line on top of the OU)",
            "heavy-tailed": "a single Gaussian OU at any sigma under-produces the deep drops: "
            "the amplitude process NEEDS A SWITCHING (jump) COMPONENT, or heavier-tailed "
            "innovations, for these lines",
            "wider Gaussian": "a Gaussian OU is SUFFICIENT in form -- neither a switching "
            "component nor the carrier deviation track is called for -- but ONE rig sigma_v is "
            "not: the lines near the 6 dB threshold wander by "
            f"{_f(sh['gaussian_fitted_floor']['sigma_db'])} dB where the resolvable ones need "
            + (
                _f(d["shape"]["resolvable"]["gaussian_fitted_floor"]["sigma_db"])
                if d["shape"]["resolvable"]
                else "n/a"
            )
            + " dB in the same observable, and the intermittent tracks' own block sd runs "
            + ", ".join(
                f"{_f(v['median_track_sd_db'])} dB at k {g}" for g, v in d["by_order_group"].items()
            )
            + ": sigma_v has to depend on the line (its order group, or its level over the "
            "floor), not be pinned at the resolvable lines' value",
        }[kind]
    parts = [
        f"**{dominant} dominates DREGON's intermittency.**",
        f"DREGON has {d['counts']['tracks']['intermittent']} intermittent tracks "
        f"({d['counts']['intermittent_absent_blocks']} absent blocks) against "
        f"{d['counts']['tracks']['resolvable']} resolvable and "
        f"{d['counts']['tracks']['under']} always-under.",
        f"(a) Label: no unbiased carrier re-estimate removes it; the best, '{lab_name}', "
        f"returns {_pc(lab['absent_recovered'])} of the absent blocks, takes "
        f"{_pc(lab['present_lost'])} of the present ones and leaves "
        f"{_pc(lab['still_intermittent'])} of the tracks intermittent. The +-3-bin search "
        f"reaches {PROM_DB:g} dB in {_pc(own['recovered'])} of absent blocks, "
        f"{_pc(own['recovered_other_rotor'])} of them on another rotor's line and "
        f"{_pc(own['recovered_own'])} elsewhere (always-under null {_pc(null['recovered_own'])})"
        + (
            f"; on Michael's audio-refined label that rate is "
            f"{_pc(m['label']['search']['intermittent']['recovered_own'])}"
            if m
            else ""
        )
        + ".",
        f"(c) Floor: the prominence against the floor under the line (median "
        f"per-track r {_f(loc['r_per_track_quantiles'][2])}, {_pc(loc['r_per_track_below_minus_half'])} "
        f"of tracks below -0.5; pooled within tracks {_f(loc['r_pooled_within'])}, so the floor "
        f"carries {_pc(loc['r_pooled_within'] ** 2)} of the pooled scatter), but holding the "
        f"floor at its mean ('{fl_name}') leaves {_pc(fl['still_intermittent'])} of the tracks "
        f"intermittent ({_pc(fl['now_under'])} become steadily under: lines seen only in floor "
        f"dips){_floor_by_group(d)}.",
        "(b) Amplitude: "
        + _amp_sentence(sh, d["shape"]["near_threshold"], d["shape"]["resolvable"])
        + ".",
    ]
    if m and m["shape"]["intermittent"]:
        mlab = _best(m, LABEL_UNBIASED)[1]
        mfl = _best(m, FLOOR_CF)[1]
        msh = m["shape"]["intermittent"]
        mres = m["shape"]["resolvable"]
        parts.append(
            f"Michael's control ({m['counts']['tracks']['intermittent']} intermittent tracks, "
            f"audio-refined label): the best label re-estimate leaves "
            f"{_pc(mlab['still_intermittent'])} intermittent, the floor held "
            f"{_pc(mfl['still_intermittent'])}; the prominence scatters by "
            f"{_f(msh['observed']['sd_centred'])} dB against "
            f"{_f(msh['gaussian_literal_floor']['q05_50_95']['sd_centred'][1])} for its "
            f"resolvable lines' sigma {_f(msh['gaussian_literal']['sigma_db'])} dB with the floor, "
            f"the line sigma fitted is {_f(msh['gaussian_fitted_floor']['sigma_db'])} dB"
            + (
                f" against {_f(mres['gaussian_fitted_floor']['sigma_db'])} for its resolvable lines"
                if mres
                else ""
            )
            + f", and the shape reads '{amplitude_kind(msh)[0]}'."
        )
    parts.append(f"Implication: {implication}.")
    return " ".join(parts)


def _clean(obj: Any) -> Any:
    """JSON-ready: private ``_`` keys dropped, arrays listed, non-finite -> None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


if __name__ == "__main__":
    raise SystemExit(main())
