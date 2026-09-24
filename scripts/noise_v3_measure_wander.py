"""Measure noise model v3's WANDER hyperparameters on the real windows of both rigs.

``docs/explainers/noise-model-v3-wander.qmd`` section 3.3a: the six OU
hyperparameters of the block latents -- ``(sigma_d, tau_d)`` rotor-common line
wander, ``(sigma_v, tau_v)`` per-line residual, ``(sigma_u, tau_u)`` floor
level -- plus ``(sigma_uj, tau_uj)`` for the floor's per-control-point colour
residual, are MEASURED here and held fixed by the v3 fit. No model is
evaluated and nothing is fitted to a likelihood.

Windows: the rank test's (``scripts/_mic_gain_rank.py::rig_specs``, which
reads ``rig_sampler._real_window_specs`` and adds DREGON's non-overlapping
4 s nosource grid): DREGON 50 windows over six nosource flights, Michael's 11
(FLY125 cruise + standby). Eight mics, 16 kHz, the 2048/512 flight front end.

Per window, per block of 0.25 / 0.5 / 1.0 s (:mod:`experiments.noise_model.wander`):
order-tracked line levels ``y[mic, rotor, k, block]`` and comb-masked floor
band levels ``yF[mic, band, block]``, each with its block noise. The rotor
speed law of the model (line power ``(f / f_ref)^2``, floor ``mean_i (f_i /
f_ref)^2``; ``amp_exp`` and ``floor_exp`` at their prior centres, the pin
every short-span pool gets) is removed per block before any statistic: a
speed change inside a window is deterministic in the model, not wander.

Statistics, on the mic mean of each track, each line (floor band) centred on
its mean over every window of the same regime -- the v3 fit shares ``p_ik``
across its pool, so its latents carry the offsets between windows as well as
the drift inside one -- with lag products inside windows only and the exact
expectation of the centred moments under an OU plus block noise
(:func:`wander.fit_ou`); within-window centring is the robustness row:

* ``sigma_total`` -- auto moments of every line track, minus the block noise;
* ``sigma_d`` / ``tau_d`` -- cross moments of pairs of distinct lines of the
  SAME rotor in the same window (no noise term: different cells);
* ``sigma_v`` / ``tau_v`` -- auto moments with the fitted rotor-common part
  taken out;
* ``sigma_u`` / ``tau_u`` -- cross moments of distinct floor bands;
  ``sigma_uj`` / ``tau_uj`` the auto moments minus that level part.

Usage
-----

    PYTHONPATH=src python scripts/noise_v3_measure_wander.py            # both rigs
    PYTHONPATH=src python scripts/noise_v3_measure_wander.py --limit 2  # smoke
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from experiments.noise_model import supports as SUP
from experiments.noise_model import wander as W

OUT_DIR = Path("results/noise_v3/wander")
SCHEMA = "noise-v3-wander/1"
RIGS = ("dregon", "michaels")
BLOCK_LENGTHS_S = (0.25, 0.5, 1.0)
#: The block length the JSON carries (explainer section 3.1: ~tau_d / 4, and
#: the carrier-motion limit of section 1.5 (3)); the sensitivity table checks it.
CHOSEN_BLOCK_S = 0.5
#: The model's speed laws at their prior centres (``Priors.amp_exp`` N(2, 1),
#: pinned at 2 on a pool under a 1.5x span; ``log_floor_exp`` median 2).
AMP_EXP = 2.0
FLOOR_EXP = 2.0
F_REF_REV_S = 80.0
#: Order groups of the breakdown: the blade-pass family, the aerodynamic
#: comb, and the motor's electromagnetic orders (42 = 6 x 7 pole pairs, ...).
ORDER_GROUPS: tuple[tuple[str, int, int], ...] = (
    ("1-2", 1, 2),
    ("3-8", 3, 8),
    ("9-24", 9, 24),
    ("25-60", 25, 60),
    ("61+", 61, 10_000),
)
#: Floor band groups (by band centre).
BAND_GROUPS: tuple[tuple[str, float, float], ...] = (
    ("<500 Hz", 0.0, 500.0),
    ("500-2k", 500.0, 2000.0),
    (">=2k", 2000.0, 1e9),
)
#: The rotor-common split needs same-rotor pairs of ROTOR-DOMINANT lines that
#: share this many blocks; below it the split uses every resolvable line.
MIN_SAME_ROTOR_BLOCKS = 100
#: Explainer section 1.5 (2): "report sigma_d only from orders whose
#: block-level sd is under, say, 2 dB" -- a track is used when the median
#: MEASURED block sd at the chosen block length is under this ...
MAX_TRACK_NOISE_DB = 2.0
#: ... and any block measured noisier than this is a missing value.
MAX_BLOCK_NOISE_DB = 4.0
#: Two lines of different rotors closer than this (window-mean frequencies)
#: share cells: their cross moment is the same measurement twice, not a
#: covariance, and the pair is skipped.
SHARED_CELL_HZ = (W.LINE_HALF_BINS + 2) * W.FLIGHT_SR / W.FLIGHT_N_FFT
LAGS_1 = (1,)
LAGS_4 = (1, 2, 3, 4)
#: The tau estimate the JSON carries: ``lag1`` (the explainer's) or
#: ``lags1_4``; chosen on the window-bootstrap spread (findings, "Estimates").
PRIMARY = "lag1"
PRIMARY_LAGS = LAGS_1 if PRIMARY == "lag1" else LAGS_4
N_BOOT = 200
BOOT_SEED = 20260924
#: The aerodynamic comb: orders up to here. Above it sit the motors'
#: electromagnetic orders (42 = 6 steps x 7 pole pairs, 84, ...), and there
#: a label error of 0.1 rev/s already moves the line by a bin.
AERO_MAX_ORDER = 24
LEGACY_YAML = "conf/online_mix/rig_fitted_5050.yaml"
NAN = float("nan")


def _module(name: str) -> Any:
    """A sibling script as a module (the rank test's window loader)."""
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


@dataclass
class WindowData:
    """One real window, measured at every block length."""

    name: str
    spec: str
    tag: str
    regime: str
    recording: str
    duration_s: float
    carrier_rev_s: np.ndarray  # (R,) window mean
    carrier_span_rev_s: tuple[float, float]
    n_mics: int
    lines: dict[float, W.LineBlocks]  # pruned to the orders any rotor resolves
    resolvable: np.ndarray  # (R, K') at CHOSEN_BLOCK_S
    dominant: np.ndarray  # (R, K')
    floor: dict[str, dict[float, W.FloorBlocks]]  # grid -> block_s -> levels
    n_orders: int  # before pruning
    eligible: np.ndarray  # (R, K') median measured block sd <= MAX_TRACK_NOISE_DB


def floor_edges() -> dict[str, np.ndarray]:
    return {"control": W.control_band_edges(), "third_octave": W.third_octave_edges()}


def measure_window(
    spec: SUP.SupportSpec, tag: str, regime: str, blocks_s: Sequence[float]
) -> WindowData:
    return measure_support(SUP.load_support(spec), spec.text, tag, regime, blocks_s)


def measure_support(
    sup: SUP.Support, spec_text: str, tag: str, regime: str, blocks_s: Sequence[float]
) -> WindowData:
    """:func:`measure_window` on an already-loaded support: a real window, or a
    RENDER through :func:`supports.synthetic_support` (the v3 held-out check)."""
    order = [CHOSEN_BLOCK_S] + [b for b in blocks_s if b != CHOSEN_BLOCK_S]
    lines: dict[float, W.LineBlocks] = {}
    floor: dict[str, dict[float, W.FloorBlocks]] = {g: {} for g in floor_edges()}
    keep = res = dom = strong = None
    n_orders = 0
    for bs in order:
        blocks = W.frame_blocks(sup.frame_centres_s, bs)
        lb = W.measure_lines(sup.power, sup.freqs_hz, sup.carrier_rev_s, blocks, block_s=bs)
        if keep is None:
            n_orders = int(lb.orders.size)
            res_all = lb.resolvable()
            keep = np.nonzero(res_all.any(axis=0))[0]
            res = res_all[:, keep]
            dom = lb.dominant()[:, keep]
            strong = lb.window_prominence_db >= W.STRONG_LINE_DB
        lines[bs] = lb.take_orders(keep)
        for grid, edges in floor_edges().items():
            floor[grid][bs] = W.measure_floor(
                sup.power, sup.freqs_hz, sup.carrier_rev_s, blocks, edges, strong=strong
            )
    assert res is not None and dom is not None
    lb0 = lines[CHOSEN_BLOCK_S]
    car = np.asarray(sup.carrier_rev_s, dtype=np.float64)
    return WindowData(
        name=sup.name,
        spec=spec_text,
        tag=tag,
        regime=regime,
        recording=str(sup.meta.get("recording_id", sup.name)),
        duration_s=float(sup.duration_s),
        carrier_rev_s=car.mean(axis=1),
        carrier_span_rev_s=(float(car.min()), float(car.max())),
        n_mics=int(sup.n_mics),
        lines=lines,
        resolvable=res,
        dominant=dom & res,
        floor=floor,
        n_orders=n_orders,
        eligible=median_block_sd(lb0) <= MAX_TRACK_NOISE_DB,
    )


def median_block_sd(lb: W.LineBlocks) -> np.ndarray:
    """``(R, K)`` median over the measured blocks of the measured mic-mean
    block sd, in dB (inf for a line with no measured block)."""
    ok = lb.measured() & np.isfinite(lb.s2_emp_micmean)
    out = np.full(ok.shape[:2], np.inf)
    for r, k in zip(*np.nonzero(ok.any(axis=-1))):
        out[r, k] = float(np.sqrt(np.median(lb.s2_emp_micmean[r, k][ok[r, k]])))
    return out


def rig_rows(extra: bool = True) -> dict[str, list[tuple[SUP.SupportSpec, str, str]]]:
    return _module("_mic_gain_rank").rig_specs(extra=extra)


# ---------------------------------------------------------------------------
# tracks and moments
# ---------------------------------------------------------------------------


def line_speed_db(lb: W.LineBlocks, amp_exp: float) -> np.ndarray:
    """``(R, B)`` the model's line speed law per block, in dB."""
    return 10.0 * amp_exp * np.log10(np.maximum(lb.carrier_rev_s, 1e-9) / F_REF_REV_S)


def floor_speed_db(lb: W.LineBlocks) -> np.ndarray:
    """``(B,)`` the model's floor speed law per block (static fraction 0)."""
    ratio = (np.maximum(lb.carrier_rev_s, 1e-9) / F_REF_REV_S) ** FLOOR_EXP
    return 10.0 * np.log10(ratio.mean(axis=0))


@dataclass
class Track:
    window: int
    rotor: int
    order: int
    y: np.ndarray  # (B,) dB, speed law removed, NaN = not measured
    s2: np.ndarray  # (B,) block-noise variance, dB^2
    freq_hz: float = NAN  # window-mean line frequency (lines only)


def line_tracks(
    wins: Sequence[WindowData],
    bs: float,
    *,
    which: str = "dominant",
    noise: str = "measured",
    mic: int | None = None,
    amp_exp: float = AMP_EXP,
) -> list[Track]:
    """Line tracks of the chosen set. A track is used only if it is
    :attr:`WindowData.eligible` (median measured block sd at the chosen block
    length within ``MAX_TRACK_NOISE_DB``: explainer section 1.5 (2), such an
    order's latent is prior-dominated), and a block noisier than
    ``MAX_BLOCK_NOISE_DB`` (measured) is missing. Both read the MEASURED noise,
    so the noise variants differ only in what they subtract."""
    out: list[Track] = []
    for wi, w in enumerate(wins):
        lb = w.lines[bs]
        sel = (w.dominant if which == "dominant" else w.resolvable) & w.eligible
        adj = line_speed_db(lb, amp_exp)
        if mic is None:
            y = lb.micmean_db()
            s_meas = lb.s2_emp_micmean
            s2 = s_meas if noise == "measured" else lb.s2_explainer_micmean()
        else:
            y = lb.mic_db()[mic]
            s_meas = lb.s2_emp[mic]
            s2 = s_meas if noise == "measured" else lb.s2_explainer[mic]
        for r, k in zip(*np.nonzero(sel)):
            yy = np.where(s_meas[r, k] <= MAX_BLOCK_NOISE_DB**2, y[r, k], np.nan)
            f_line = float(lb.orders[k]) * float(np.mean(lb.carrier_rev_s[r]))
            out.append(Track(wi, int(r), int(lb.orders[k]), yy - adj[r], s2[r, k], f_line))
    return out


def floor_tracks(
    wins: Sequence[WindowData],
    bs: float,
    *,
    grid: str = "control",
    noise: str = "measured",
    mic: int | None = None,
) -> list[Track]:
    out: list[Track] = []
    for wi, w in enumerate(wins):
        fb = w.floor[grid][bs]
        adj = floor_speed_db(w.lines[bs])
        if mic is None:
            y = fb.micmean_db()
            # the explainer count is the same for every mic (one cell pattern)
            s2 = fb.s2_micmean if noise == "measured" else fb.s2_explainer[0]
        else:
            y = fb.mic_db()[mic]
            s2 = fb.s2_mic[mic] if noise == "measured" else fb.s2_explainer[mic]
        for j in range(y.shape[0]):
            if not np.isfinite(y[j]).any():
                continue
            out.append(Track(wi, -1, j, y[j] - adj, s2[j]))
    return out


def _group_of(order: int) -> str:
    for name, lo, hi in ORDER_GROUPS:
        if lo <= order <= hi:
            return name
    return ORDER_GROUPS[-1][0]


def _band_group(centre_hz: float) -> str:
    for name, lo, hi in BAND_GROUPS:
        if lo <= centre_hz < hi:
            return name
    return BAND_GROUPS[-1][0]


def track_clip(t: Track) -> bool | None:
    """The explainer's per-track clip: ``Var_b(y) - mean s^2 <= 0`` (None if
    the track has under 3 measured blocks)."""
    ok = np.isfinite(t.y) & np.isfinite(t.s2)
    if ok.sum() < 3:
        return None
    return bool(np.var(t.y[ok], ddof=1) - np.mean(t.s2[ok]) <= 0)


@dataclass
class Moments:
    """Every pooled moment set one hierarchy needs."""

    auto: W.LagMoments = field(default_factory=W.LagMoments)
    same: W.LagMoments = field(default_factory=W.LagMoments)  # same rotor / distinct bands
    across: W.LagMoments = field(default_factory=W.LagMoments)  # different rotors
    rotor_mean: W.LagMoments = field(default_factory=W.LagMoments)
    by_rotor: dict[int, W.LagMoments] = field(default_factory=dict)
    by_rotor_same: dict[int, W.LagMoments] = field(default_factory=dict)
    by_group: dict[str, W.LagMoments] = field(default_factory=dict)
    by_line: dict[tuple[int, int], W.LagMoments] = field(default_factory=dict)
    n_tracks: int = 0  # (window, line) tracks with >= 3 measured blocks
    n_series: int = 0  # tracks as pooled: per window, or per rig line
    n_shared_skipped: int = 0  # pairs of different rotors sharing cells
    track_clips: int = 0
    track_checked: int = 0


@dataclass
class Series:
    """A track on a ``(W, B)`` layout: one window, or every window of one regime."""

    rotor: int
    order: int
    y: np.ndarray  # (W * B,) dB, NaN = not measured / padding
    s2: np.ndarray  # (W * B,)
    n_blocks: int
    freqs_hz: np.ndarray  # (W,) window-mean line frequency, NaN where absent


def assemble(
    tracks: Sequence[Track], wins: Sequence[WindowData], bs: float, mode: str
) -> list[list[Series]]:
    """Pairing units of series. ``mode="window"``: one unit per window, each
    track centred on its own window mean. ``mode="rig"``: one unit per regime
    (DREGON flight; Michael's cruise, standby), each line's windows laid side
    by side and centred on the line's mean over the regime -- the v3 fit's
    semantics, where ``p_ik`` is shared by the pool and every departure from
    it at the block's speed is a latent."""
    if mode == "window":
        per: dict[int, list[Series]] = {}
        for t in tracks:
            per.setdefault(t.window, []).append(
                Series(t.rotor, t.order, t.y, t.s2, int(t.y.size), np.array([t.freq_hz]))
            )
        return list(per.values())
    regimes: dict[str, list[int]] = {}
    for wi, w in enumerate(wins):
        regimes.setdefault(w.regime, []).append(wi)
    units = []
    for widx in regimes.values():
        bmax = max(wins[i].lines[bs].n_blocks for i in widx)
        pos = {wi: p for p, wi in enumerate(widx)}
        acc: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for t in tracks:
            if t.window not in pos:
                continue
            blank = np.full((len(widx), bmax), np.nan)
            y, s, f = acc.setdefault(
                (t.rotor, t.order), (blank, blank.copy(), np.full(len(widx), np.nan))
            )
            y[pos[t.window], : t.y.size] = t.y
            s[pos[t.window], : t.s2.size] = t.s2
            f[pos[t.window]] = t.freq_hz
        units.append(
            [
                Series(r, k, y.ravel(), s.ravel(), bmax, f)
                for (r, k), (y, s, f) in sorted(acc.items())
            ]
        )
    return units


def moments(
    tracks: Sequence[Track],
    wins: Sequence[WindowData],
    bs: float,
    *,
    mode: str,
    centres_hz: np.ndarray | None = None,
) -> Moments:
    """Every pooled moment of one track set. Lines (``centres_hz`` None):
    same-rotor pairs -> ``same``, other pairs -> ``across``. Floor bands
    (``centres_hz`` given): every pair of distinct bands -> ``same``."""
    floor = centres_hz is not None
    mom = Moments()
    for t in tracks:
        ok = np.isfinite(t.y) & np.isfinite(t.s2)
        mom.n_tracks += int(ok.sum() >= 3)
        c = track_clip(t)
        if c is not None:
            mom.track_checked += 1
            mom.track_clips += int(c)
    for unit in assemble(tracks, wins, bs, mode):
        used: list[Series] = []
        for s in unit:
            if not mom.auto.add(s.y, None, s.s2, n_blocks=s.n_blocks):
                continue
            used.append(s)
            mom.n_series += 1
            if floor:
                assert centres_hz is not None
                grp = _band_group(float(centres_hz[s.order]))
            else:
                grp = _group_of(s.order)
                mom.by_rotor.setdefault(s.rotor, W.LagMoments()).add(
                    s.y, None, s.s2, n_blocks=s.n_blocks
                )
            mom.by_group.setdefault(grp, W.LagMoments()).add(s.y, None, s.s2, n_blocks=s.n_blocks)
            mom.by_line.setdefault((s.rotor, s.order), W.LagMoments()).add(
                s.y, None, s.s2, n_blocks=s.n_blocks
            )
        for i, a in enumerate(used):
            for b in used[i + 1 :]:
                same = floor or a.rotor == b.rotor
                if not same and np.any(np.abs(a.freqs_hz - b.freqs_hz) <= SHARED_CELL_HZ):
                    mom.n_shared_skipped += 1
                    continue
                tgt = mom.same if same else mom.across
                if not tgt.add(a.y, b.y, n_blocks=a.n_blocks):
                    continue
                tgt.add(b.y, a.y, n_blocks=a.n_blocks)
                if same and not floor:
                    rs = mom.by_rotor_same.setdefault(a.rotor, W.LagMoments())
                    rs.add(a.y, b.y, n_blocks=a.n_blocks)
                    rs.add(b.y, a.y, n_blocks=a.n_blocks)
        if mode == "window" and not floor and used:
            for r in sorted({s.rotor for s in used}):
                mine = [s for s in used if s.rotor == r]
                ys = np.stack([s.y for s in mine])
                ss = np.stack([s.s2 for s in mine])
                ok = np.isfinite(ys) & np.isfinite(ss)
                full = ok.all(axis=0)
                n = len(mine)
                ybar = np.where(full, np.where(ok, ys, 0.0).mean(axis=0), np.nan)
                s2bar = np.where(full, np.where(ok, ss, 0.0).sum(axis=0) / n**2, np.nan)
                mom.rotor_mean.add(ybar, None, s2bar)
    return mom


def _common(mom: W.LagMoments, bs: float) -> W.OUFit:
    """A PAIR moment's shared OU: ``sigma^2`` from the lag-0 cross product
    (the explainer's "covariance between two orders of one rotor in the same
    block"), ``tau`` from lags 1-4 -- a pair's lag-1 product alone is too small
    against its scatter to pin ``tau``. No noise term: different cells."""
    return W.fit_ou(mom, bs, lags=LAGS_4, use_noise=False)


def _common_lags_only(mom: W.LagMoments, bs: float) -> W.OUFit:
    """The same without lag 0: block noise two tracks SHARE inside a block (a
    broadband event lifting every line of the block) would sit at lag 0 only;
    its size comes back as the nugget (positive = lag 0 inflated)."""
    return W.fit_ou(mom, bs, lags=LAGS_4, use_noise=False, skip_lag0=True)


def hierarchy(mom: Moments, bs: float, lags: Sequence[int]) -> dict[str, Any]:
    """``total`` (auto: sigma^2 at lag 0 minus the measured noise, tau at
    ``lags``), ``common`` (same-rotor / cross-band pairs, :func:`_common`),
    ``residual`` (auto minus the fitted common part, tau at ``lags``), and
    the lag-0-free diagnostics: ``common_lags_only`` (nugget = shared block
    noise) and ``total_lags_only`` (nugget = block noise the data show beyond
    the measured one); ``across`` (lines of different rotors)."""
    tot = W.fit_ou(mom.auto, bs, lags=lags)
    com = _common(mom.same, bs)
    off = None
    if com.sigma2 > 0 and np.isfinite(com.rho):
        off = com.sigma2 * mom.auto.expected_unit(np.array([com.rho]))[0]
    res = W.fit_ou(mom.auto, bs, lags=lags, offset=off)
    noise_mean = mom.auto.s2_sum / mom.auto.n_obs if mom.auto.n_obs else NAN
    out: dict[str, Any] = dict(
        total=tot.as_dict() | dict(n_pairs=mom.auto.n_pairs),
        common=com.as_dict() | dict(n_pairs=mom.same.n_pairs, n_common_blocks=float(mom.same.n[0])),
        residual=res.as_dict() | dict(n_pairs=mom.auto.n_pairs),
        common_lags_only=_common_lags_only(mom.same, bs).as_dict(),
        total_lags_only=W.fit_ou(mom.auto, bs, lags=LAGS_4, skip_lag0=True).as_dict()
        | dict(measured_noise_mean_db2=noise_mean),
    )
    if mom.across.n_pairs:
        out["across"] = _common(mom.across, bs).as_dict() | dict(n_pairs=mom.across.n_pairs)
    out["n_shared_skipped"] = mom.n_shared_skipped
    return out


def naive_block(mom: Moments, bs: float) -> dict[str, Any]:
    """The explainer's literal steps, no centring correction."""
    tot = mom.auto.naive(bs)
    com = mom.same.naive(bs)
    out: dict[str, Any] = dict(total=tot, common=com)
    if np.isfinite(tot.get("sigma2", NAN)) and np.isfinite(com.get("sigma2", NAN)):
        out["residual_sigma2"] = float(max(tot["sigma2"] - com["sigma2"], 0.0))
    if mom.rotor_mean.n_pairs:
        out["rotor_mean_track"] = mom.rotor_mean.naive(bs)
    return out


def six(line_h: dict[str, Any], floor_h: dict[str, Any]) -> dict[str, float]:
    """The contract's numbers (+ the floor colour pair) from two hierarchies."""

    def pair(d: dict[str, Any]) -> tuple[float, float]:
        return float(d["sigma_db"]), float(d["tau_s"])

    sd, td = pair(line_h["common"])
    sv, tv = pair(line_h["residual"])
    su, tu = pair(floor_h["common"])
    suj, tuj = pair(floor_h["residual"])
    return dict(
        sigma_d_db=sd,
        tau_d_s=td,
        sigma_v_db=sv,
        tau_v_s=tv,
        sigma_u_db=su,
        tau_u_s=tu,
        sigma_uj_db=suj,
        tau_uj_s=tuj,
        sigma_total_db=float(line_h["total"]["sigma_db"]),
        tau_total_s=float(line_h["total"]["tau_s"]),
        sigma_floor_total_db=float(floor_h["total"]["sigma_db"]),
    )


def within_speed_slope(
    tracks: Sequence[Track], wins: Sequence[WindowData], bs: float
) -> dict[str, float]:
    """Pooled within-window slope of the line level (speed law NOT removed)
    against ``10 log10`` of the block carrier: the speed exponent the data
    shows inside a window (the model's is ``AMP_EXP``)."""
    sxy = sxx = 0.0
    n = 0
    for t in tracks:
        lb = wins[t.window].lines[bs]
        x = 10.0 * np.log10(np.maximum(lb.carrier_rev_s[t.rotor], 1e-9))
        y = t.y + AMP_EXP * x  # put the removed law back (the constant cancels on centring)
        ok = np.isfinite(y)
        if ok.sum() < 3:
            continue
        xc, yc = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
        sxy += float(xc @ yc)
        sxx += float(xc @ xc)
        n += int(ok.sum())
    return dict(slope=sxy / sxx if sxx > 0 else NAN, n_blocks=n, x_var_db2=sxx / max(n, 1))


# ---------------------------------------------------------------------------
# one rig
# ---------------------------------------------------------------------------


def analyse_setting(
    wins: Sequence[WindowData],
    bs: float,
    *,
    which: str,
    mode: str = "rig",
    noise: str = "measured",
    mic: int | None = None,
    amp_exp: float = AMP_EXP,
    lags: Sequence[int] = PRIMARY_LAGS,
    grid: str = "control",
    max_order: int | None = None,
) -> dict[str, Any]:
    lt = line_tracks(wins, bs, which=which, noise=noise, mic=mic, amp_exp=amp_exp)
    if max_order is not None:
        lt = [t for t in lt if t.order <= max_order]
    ft = floor_tracks(wins, bs, grid=grid, noise=noise, mic=mic)
    centres = wins[0].floor[grid][bs].centres_hz
    lm = moments(lt, wins, bs, mode=mode)
    fm = moments(ft, wins, bs, mode=mode, centres_hz=centres)
    lh, fh = hierarchy(lm, bs, lags), hierarchy(fm, bs, lags)
    return dict(
        numbers=six(lh, fh),
        lines=lh,
        floor=fh,
        n_line_tracks=lm.n_tracks,
        n_floor_tracks=fm.n_tracks,
        _lm=lm,
        _fm=fm,
    )


def _run(
    wins: Sequence[WindowData], bs: float, **kw: Any
) -> tuple[dict[str, Any], Moments, Moments]:
    """:func:`analyse_setting` split into its JSON part and its moment sets."""
    s = analyse_setting(wins, bs, **kw)
    return s, s.pop("_lm"), s.pop("_fm")


def choose_line_set(wins: Sequence[WindowData]) -> str:
    """``dominant`` when its same-rotor pairs share enough blocks, else ``resolvable``."""
    tracks = line_tracks(wins, CHOSEN_BLOCK_S, which="dominant")
    lm = moments(tracks, wins, CHOSEN_BLOCK_S, mode="rig")
    return "dominant" if lm.same.n[0] >= 2 * MIN_SAME_ROTOR_BLOCKS else "resolvable"


def fits_by(moms: dict[Any, W.LagMoments], bs: float, *, pairs: bool = False) -> dict[str, Any]:
    """One fit per slice; ``pairs`` slices get the lag-0-free :func:`_common`."""
    out = {}
    for k, m in sorted(moms.items(), key=lambda kv: str(kv[0])):
        f = _common(m, bs) if pairs else W.fit_ou(m, bs, lags=LAGS_1)
        out[str(k)] = f.as_dict() | dict(n_pairs=m.n_pairs, n_obs=m.n_obs)
    return out


def line_clip_table(moms: dict[tuple[int, int], W.LagMoments], bs: float) -> dict[str, Any]:
    """Per rig line (rotor, order), pooled over its windows: the explainer's
    naive estimate and the centred fit, and how many clip at zero."""
    rows = {}
    naive_clips = fit_clips = 0
    for key, m in sorted(moms.items()):
        nv = m.naive(bs)
        ft = W.fit_ou(m, bs, lags=LAGS_1)
        naive_clips += int(bool(nv.get("clipped")))
        fit_clips += int(ft.clipped)
        rows[f"{key[0]}:{key[1]}"] = dict(
            n_windows=m.n_pairs,
            naive_sigma_db=math.sqrt(max(nv.get("sigma2", 0.0) or 0.0, 0.0)),
            naive_clipped=bool(nv.get("clipped")),
            sigma_db=ft.sigma_db,
            tau_s=ft.tau_s,
            clipped=ft.clipped,
        )
    return dict(n=len(rows), naive_clipped=naive_clips, fit_clipped=fit_clips, lines=rows)


def resolvable_counts(wins: Sequence[WindowData]) -> dict[str, Any]:
    """Track counts: ``[resolvable, dominant, resolvable & eligible, dominant &
    eligible]`` overall, by rotor, by order group and by line."""
    tot = [0, 0, 0, 0]
    by_rotor: dict[str, list[int]] = {}
    by_group: dict[str, list[int]] = {g: [0, 0, 0, 0] for g, _, _ in ORDER_GROUPS}
    distinct: dict[str, list[int]] = {}
    for w in wins:
        orders = w.lines[CHOSEN_BLOCK_S].orders
        for r, k in zip(*np.nonzero(w.resolvable)):
            d, e = bool(w.dominant[r, k]), bool(w.eligible[r, k])
            flags = (1, int(d), int(e), int(d and e))
            key = f"{r}:{int(orders[k])}"
            for acc in (
                tot,
                by_rotor.setdefault(str(r), [0, 0, 0, 0]),
                by_group[_group_of(int(orders[k]))],
                distinct.setdefault(key, [0, 0, 0, 0]),
            ):
                for i, f in enumerate(flags):
                    acc[i] += f
    return dict(
        tracks=tot,
        distinct_lines=[sum(1 for v in distinct.values() if v[i] > 0) for i in range(4)],
        orders_measured_per_window=float(np.mean([w.n_orders for w in wins])),
        by_rotor=by_rotor,
        by_group=by_group,
        by_line=dict(
            sorted(distinct.items(), key=lambda kv: tuple(int(v) for v in kv[0].split(":")))
        ),
    )


def noise_summary(wins: Sequence[WindowData], which: str) -> dict[str, Any]:
    """Block-noise variances of the used line tracks and floor bands at every
    block length: measured vs the explainer's count, and the overlap factor."""
    out: dict[str, Any] = {}
    for bs in BLOCK_LENGTHS_S:
        emp, expl = [], []
        for w in wins:
            lb = w.lines[bs]
            base = (w.dominant if which == "dominant" else w.resolvable) & w.eligible
            sel = base[..., None] & lb.measured()
            emp.append(lb.s2_emp_micmean[sel])
            expl.append(lb.s2_explainer_micmean()[sel])
        e, x = np.concatenate(emp), np.concatenate(expl)
        fl_mm, fl_ex, meff, neq_ratio = [], [], [], []
        for w in wins:
            fb = w.floor["control"][bs]
            fl_mm.append(fb.s2_micmean[fb.valid])
            fl_ex.append(fb.s2_explainer[0][fb.valid])
            meff.append(fb.m_eff[np.isfinite(fb.m_eff)])
            neq_ratio.append((fb.n_equiv / np.maximum(fb.n_cells, 1))[fb.valid])
        nfr = int(round(bs * W.FLIGHT_SR / W.FLIGHT_HOP))
        out[str(bs)] = dict(
            line_s2_measured_median_db2=_q(e, 0.5),
            line_s2_explainer_median_db2=_q(x, 0.5),
            line_ratio_measured_over_explainer_median=_q(e / np.where(x > 0, x, np.nan), 0.5),
            line_blocks=int(np.isfinite(e).sum()),
            floor_s2_measured_median_db2=_q(np.concatenate(fl_mm), 0.5),
            floor_s2_explainer_median_db2=_q(np.concatenate(fl_ex), 0.5),
            floor_m_eff_median=_q(np.concatenate(meff), 0.5),
            floor_equiv_over_cells_median=_q(np.concatenate(neq_ratio), 0.5),
            overlap_inflation_line_rectangle=W.overlap_inflation(nfr, 2 * W.LINE_HALF_BINS + 1),
        )
    return out


def _q(a: np.ndarray, q: float) -> float:
    v = np.asarray(a, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(np.quantile(v, q)) if v.size else NAN


def bootstrap(
    wins: Sequence[WindowData], which: str, bs: float, *, n_boot: int, seed: int
) -> dict[str, Any]:
    """Window bootstrap (resampled within each regime) of the chosen setting:
    ``[q05, q50, q95]`` of every number for both lag variants, and the share
    of draws in which a sigma clips to zero."""
    rng = np.random.default_rng(seed)
    regimes: dict[str, list[int]] = {}
    for i, w in enumerate(wins):
        regimes.setdefault(w.regime, []).append(i)
    centres = wins[0].floor["control"][bs].centres_hz
    draws: dict[str, list[dict[str, float]]] = {"lag1": [], "lags1_4": []}
    for _ in range(n_boot):
        idx = [int(j) for ids in regimes.values() for j in rng.choice(ids, size=len(ids))]
        sub = [wins[i] for i in idx]
        lm = moments(line_tracks(sub, bs, which=which), sub, bs, mode="rig")
        fm = moments(floor_tracks(sub, bs), sub, bs, mode="rig", centres_hz=centres)
        for name, lags in (("lag1", LAGS_1), ("lags1_4", LAGS_4)):
            draws[name].append(six(hierarchy(lm, bs, lags), hierarchy(fm, bs, lags)))
    out: dict[str, Any] = dict(n_boot=n_boot, seed=seed)
    for name, rows in draws.items():
        keys = rows[0].keys()
        out[name] = {
            k: [_q(np.array([r[k] for r in rows]), q) for q in (0.05, 0.5, 0.95)] for k in keys
        }
        out[name]["zero_share"] = {
            k: float(np.mean([r[k] == 0.0 for r in rows])) for k in keys if k.startswith("sigma")
        }
    return out


def analyse_rig(rig: str, wins: Sequence[WindowData]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(detail, figure data)`` of one rig."""
    which = choose_line_set(wins)
    bs0 = CHOSEN_BLOCK_S
    detail: dict[str, Any] = dict(line_set=which)
    detail["windows"] = [
        dict(
            name=w.name,
            spec=w.spec,
            tag=w.tag,
            regime=w.regime,
            recording=w.recording,
            duration_s=w.duration_s,
            carrier_mean_rev_s=[float(v) for v in w.carrier_rev_s],
            carrier_span_rev_s=list(w.carrier_span_rev_s),
            n_resolvable=int(w.resolvable.sum()),
            n_dominant=int(w.dominant.sum()),
        )
        for w in wins
    ]
    detail["resolvable"] = resolvable_counts(wins)
    detail["noise"] = noise_summary(wins, which)

    # the chosen setting (rig-centred), both lag variants
    main, lm, fm = _run(wins, bs0, which=which, lags=LAGS_1)
    main4, lm4, fm4 = _run(wins, bs0, which=which, lags=LAGS_4)
    detail["chosen"] = dict(block_s=bs0, lag1=main, lags1_4=main4, primary=PRIMARY)
    detail["bootstrap"] = bootstrap(wins, which, bs0, n_boot=N_BOOT, seed=BOOT_SEED)
    if PRIMARY == "lags1_4":
        main, lm, fm = main4, lm4, fm4

    # block-length sensitivity: rig-centred (primary), window-centred, and the
    # explainer's literal arithmetic on window-centred tracks
    sens = {}
    for bs in BLOCK_LENGTHS_S:
        s, slm, sfm = _run(wins, bs, which=which)
        sw, wlm, wfm = _run(wins, bs, which=which, mode="window")
        sens[str(bs)] = dict(
            numbers=s["numbers"],
            within_window=sw["numbers"],
            naive_lines=naive_block(wlm, bs),
            naive_floor=naive_block(wfm, bs),
            n_line_tracks=s["n_line_tracks"],
            n_floor_tracks=s["n_floor_tracks"],
            clips=dict(
                line_tracks=[slm.track_clips, slm.track_checked],
                floor_tracks=[sfm.track_clips, sfm.track_checked],
                lines_total_fit=bool(s["lines"]["total"]["clipped"]),
                lines_common_fit=bool(s["lines"]["common"]["clipped"]),
                lines_residual_fit=bool(s["lines"]["residual"]["clipped"]),
                floor_common_fit=bool(s["floor"]["common"]["clipped"]),
                floor_residual_fit=bool(s["floor"]["residual"]["clipped"]),
            ),
        )
    detail["block_length"] = sens

    # robustness rows at the chosen block length
    rob: dict[str, Any] = {}
    ex, exl, exf = _run(wins, bs0, which=which, noise="explainer")
    rob["noise_explainer"] = ex["numbers"] | dict(
        clips=dict(
            line_tracks=[exl.track_clips, exl.track_checked],
            floor_tracks=[exf.track_clips, exf.track_checked],
        )
    )
    other = "resolvable" if which == "dominant" else "dominant"
    alt, _, _ = _run(wins, bs0, which=other)
    rob[f"lines_{other}"] = alt["numbers"] | dict(n_line_tracks=alt["n_line_tracks"])
    low, _, _ = _run(wins, bs0, which=which, max_order=AERO_MAX_ORDER)
    rob["orders_aero"] = low["numbers"] | dict(n_line_tracks=low["n_line_tracks"])
    slope = within_speed_slope(line_tracks(wins, bs0, which=which), wins, bs0)
    rob["speed_slope_within_window"] = slope
    for a in (0.0, slope["slope"]) if np.isfinite(slope["slope"]) else (0.0,):
        s, _, _ = _run(wins, bs0, which=which, amp_exp=float(a))
        rob[f"amp_exp_{a:.2f}"] = s["numbers"]
    th, _, _ = _run(wins, bs0, which=which, grid="third_octave")
    rob["floor_third_octave"] = {
        k: v for k, v in th["numbers"].items() if "_u" in k or "floor" in k
    }
    per_mic = [_run(wins, bs0, which=which, mic=m)[0]["numbers"] for m in range(wins[0].n_mics)]
    rob["per_mic"] = per_mic
    rob["per_mic_quantiles"] = {
        k: [_q(np.array([p[k] for p in per_mic]), q) for q in (0.0, 0.5, 1.0)] for k in per_mic[0]
    }
    detail["robustness"] = rob

    # breakdowns (chosen setting)
    detail["by_rotor"] = dict(
        total=fits_by(lm.by_rotor, bs0), common=fits_by(lm.by_rotor_same, bs0, pairs=True)
    )
    detail["by_order_group"] = fits_by(lm.by_group, bs0)
    detail["floor_by_band_group"] = fits_by(fm.by_group, bs0)
    detail["line_clips"] = line_clip_table(lm.by_line, bs0)
    detail["floor_band_clips"] = line_clip_table(fm.by_line, bs0)

    figs = dict(lm=lm, fm=fm, main=main, which=which)
    return detail, figs


def contract(rig: str, detail: dict[str, Any], wins: Sequence[WindowData]) -> dict[str, Any]:
    """The ``<rig>.json`` the v3 fit reads (``--wander``)."""
    num = detail["chosen"][PRIMARY]["numbers"]
    lines = detail["chosen"][PRIMARY]["lines"]
    floor = detail["chosen"][PRIMARY]["floor"]
    out: dict[str, Any] = {}
    flags: dict[str, Any] = {}
    for part, blk in (
        ("d", lines["common"]),
        ("v", lines["residual"]),
        ("u", floor["common"]),
        ("uj", floor["residual"]),
    ):
        sig, tau = float(blk["sigma_db"]), float(blk["tau_s"])
        if blk["clipped"] or not np.isfinite(tau):
            # no wander above the block noise: sigma 0 makes tau irrelevant;
            # one block keeps it a finite positive number for the OU prior
            sig, tau = 0.0, CHOSEN_BLOCK_S
            flags[part] = "clipped: sigma set to 0, tau to block_s"
        elif blk["edge"]:
            flags[part] = f"tau on the grid edge ({blk['edge']})"
        out[f"sigma_{part}_db"] = sig
        out[f"tau_{part}_s"] = tau
    return dict(
        schema=SCHEMA,
        rig=rig,
        sigma_d_db=out["sigma_d_db"],
        tau_d_s=out["tau_d_s"],
        sigma_v_db=out["sigma_v_db"],
        tau_v_s=out["tau_v_s"],
        sigma_u_db=out["sigma_u_db"],
        tau_u_s=out["tau_u_s"],
        sigma_uj_db=out["sigma_uj_db"],
        tau_uj_s=out["tau_uj_s"],
        block_s=CHOSEN_BLOCK_S,
        n_lines_used=int(detail["chosen"][PRIMARY]["n_line_tracks"]),
        n_windows=len(wins),
        provenance=dict(
            generated_by="scripts/noise_v3_measure_wander.py",
            git_head=git_head(),
            explainer="docs/explainers/noise-model-v3-wander.qmd section 3.3a",
            windows="scripts/_mic_gain_rank.py::rig_specs(extra=True)",
            window_specs=[w.spec for w in wins],
            front_end="2048/512 periodic Hann, 16 kHz, frame-mean label carriers",
            units="sigma_*: stationary sd in dB; tau_*: OU correlation time in s; block_s in s",
            line_set=detail["line_set"],
            line_estimator=(
                "mic-mean order-tracked line level (3 bins at the label carrier minus the local "
                "q25 floor), speed law (f/f_ref)^2 removed, centred per line on its regime mean; "
                "d = lag-0 cross moment of two lines of one rotor with tau over lags 1-4, v = "
                "auto moments minus the fitted d part (lag 0 minus the measured noise); exact "
                "centred-moment OU fits; tau_v from "
                + ("lag 1" if PRIMARY == "lag1" else "lags 1-4")
            ),
            floor_estimator=(
                "mic-mean comb-masked level on bands around the 14 floor control points, speed "
                "law mean_i (f_i/f_ref)^2 removed, centred per band on its regime mean; u = "
                "lag-0 cross-band moment with tau over lags 1-4, uj = auto minus u"
            ),
            block_noise="measured: within-block frame scatter (lines); exact correlated-cell Gamma count x mic coherence (floor)",
            sigma_total_db=float(num["sigma_total_db"]),
            tau_total_s=float(num["tau_total_s"]),
            flags=flags,
            detail="results/noise_v3/wander/wander_detail.json",
        ),
    )


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def example_tracks(
    all_wins: dict[str, list[WindowData]], rig_which: dict[str, str], n: int = 3
) -> list[dict[str, Any]]:
    """The ``n`` most prominent used line tracks across both rigs, at most two
    per rig, each from a different line."""
    cands = []
    for rig, wins in all_wins.items():
        for wi, w in enumerate(wins):
            lb = w.lines[CHOSEN_BLOCK_S]
            sel = (w.dominant if rig_which[rig] == "dominant" else w.resolvable) & w.eligible
            for r, k in zip(*np.nonzero(sel)):
                cands.append((float(np.nanmean(lb.prominence_db[r, k])), rig, wi, int(r), int(k)))
    cands.sort(reverse=True)
    out, seen, per_rig = [], set(), {}
    for prom, rig, wi, r, k in cands:
        key = (rig, r, int(all_wins[rig][wi].lines[CHOSEN_BLOCK_S].orders[k]))
        if key in seen or per_rig.get(rig, 0) >= 2:
            continue
        seen.add(key)
        per_rig[rig] = per_rig.get(rig, 0) + 1
        out.append(dict(rig=rig, window=wi, rotor=r, k_idx=k, prominence_db=prom))
        if len(out) >= n:
            break
    return out


def tracks_figure(
    all_wins: dict[str, list[WindowData]], picks: Sequence[dict[str, Any]], path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(picks), 1, figsize=(8.5, 2.6 * len(picks)), squeeze=False)
    for ax, p in zip(axes[:, 0], picks):
        w = all_wins[p["rig"]][p["window"]]
        lb = w.lines[CHOSEN_BLOCK_S]
        r, k = p["rotor"], p["k_idx"]
        y = lb.micmean_db()[r, k] - line_speed_db(lb, AMP_EXP)[r]
        y = y - np.nanmean(y)
        t = (np.arange(y.size) + 0.5) * CHOSEN_BLOCK_S
        s_m = np.sqrt(lb.s2_emp_micmean[r, k])
        s_e = np.sqrt(lb.s2_explainer_micmean()[r, k])
        ax.fill_between(t, -s_m, s_m, color="0.8", label="+-s measured (within-block)")
        ax.plot(t, s_e, "--", color="0.45", lw=1, label="+-s explainer count")
        ax.plot(t, -s_e, "--", color="0.45", lw=1)
        ax.plot(t, y, "o-", color="C0", lw=1.5, ms=4, label="block level - window mean")
        for m in range(lb.line_db.shape[0]):
            ym = lb.mic_db()[m, r, k] - line_speed_db(lb, AMP_EXP)[r]
            ax.plot(t, ym - np.nanmean(ym), color="C1", lw=0.5, alpha=0.35)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("dB")
        ax.set_title(
            f"{p['rig']}: rotor {r + 1}, order {int(lb.orders[k])} -- {w.recording}, "
            f"window {p['window'] + 1} ({w.tag})\n"
            f"mean block prominence {p['prominence_db']:.1f} dB, "
            f"own share {float(lb.own_share[r, k]):.2f}",
            fontsize=9,
        )
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel(f"time in window (s), {CHOSEN_BLOCK_S:g} s blocks; thin: single mics")
    axes[0, 0].legend(fontsize=7, loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def acf_figure(figs: dict[str, dict[str, Any]], path: Path) -> None:
    """Mean centred lag products vs lag, with the fitted OU's expectation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rigs = list(figs)
    fig, axes = plt.subplots(1, len(rigs), figsize=(6.2 * len(rigs), 4.2), squeeze=False)
    for ax, rig in zip(axes[0], rigs):
        f = figs[rig]
        lm, fm, main = f["lm"], f["fm"], f["main"]
        lags = np.arange(lm.auto.max_lag + 1) * CHOSEN_BLOCK_S
        rows = [
            ("line auto (d + v + noise)", lm.auto, main["lines"]["total"], "C0", None),
            ("same-rotor line pairs (d)", lm.same, main["lines"]["common"], "C1", None),
            ("floor band auto", fm.auto, main["floor"]["total"], "C2", None),
            ("floor band pairs (u)", fm.same, main["floor"]["common"], "C3", None),
        ]
        for label, mom, fit, col, _ in rows:
            if mom.n_pairs == 0:
                continue
            ax.plot(
                lags, mom.mean_products(), "o", color=col, label=f"{label}  [{mom.n_pairs} pairs]"
            )
            ofit = W.OUFit(
                float(fit["sigma2_db2"]),
                float(fit["tau_s"]),
                float(fit["rho"]),
                bool(fit["clipped"]),
                str(fit["edge"]),
                tuple(fit["lags"]),
                NAN,
            )
            use_noise = mom is lm.auto or mom is fm.auto
            mm = mom if use_noise else _without_noise(mom)
            ax.plot(lags, W.ou_expected_products(mm, ofit), "-", color=col, lw=1.2)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("lag (s)")
        ax.set_ylabel("mean centred product (dB$^2$)")
        ax.set_title(
            f"{rig}: {f['which']} lines, {CHOSEN_BLOCK_S:g} s blocks\n"
            "markers: data; lines: fitted OU's expectation incl. the centring",
            fontsize=9,
        )
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _without_noise(mom: W.LagMoments) -> W.LagMoments:
    out = W.LagMoments(max_lag=mom.max_lag, max_len=mom.max_len).merge(mom)
    out.noise = np.zeros_like(out.noise)
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, np.generic):
        return _clean(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _f(v: Any, fmt: str = "{:.2f}") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return fmt.format(v)


def legacy_ranges() -> dict[str, Any]:
    """The legacy stochastic model's wander sliders in ``LEGACY_YAML`` (source
    0 = the DREGON refit, source 1 = the Michael's refit)."""
    import yaml

    with open(LEGACY_YAML) as fh:
        pol = yaml.safe_load(fh)
    out = {}
    names = ("dregon", "michaels")
    stoch = [s for s in pol["sources"]["noise"] if s.get("kind") == "stochastic"]
    for name, src in zip(names, stoch):
        rng = src.get("ranges", src)
        out[name] = {
            k: rng.get(k)
            for k in (
                "harm_gp_std_db",
                "harm_gp_tau_s",
                "harm_coherence",
                "floor_gp_std_db",
                "floor_gp_tau_s",
            )
        }
    return out


NUM_COLS = (
    ("sigma_d_db", "sigma_d"),
    ("tau_d_s", "tau_d"),
    ("sigma_v_db", "sigma_v"),
    ("tau_v_s", "tau_v"),
    ("sigma_u_db", "sigma_u"),
    ("tau_u_s", "tau_u"),
    ("sigma_uj_db", "sigma_uj"),
    ("tau_uj_s", "tau_uj"),
    ("sigma_total_db", "sigma_tot"),
    ("tau_total_s", "tau_tot"),
)


def _num_row(name: str, num: dict[str, Any]) -> str:
    return "| " + name + " | " + " | ".join(_f(num.get(k)) for k, _ in NUM_COLS) + " |"


def _num_head(first: str) -> list[str]:
    return [
        "| " + first + " | " + " | ".join(c for _, c in NUM_COLS) + " |",
        "|" + " --- |" * (len(NUM_COLS) + 1),
    ]


def findings_md(payload: dict[str, Any]) -> str:
    rigs = payload["rigs"]
    leg = payload["legacy"]
    out = [
        "# Noise model v3: the wander hyperparameters, measured on the real windows",
        "",
        f"`scripts/noise_v3_measure_wander.py` at `{payload['git_head']}`; core "
        "`src/experiments/noise_model/wander.py`. The measurement of "
        "`docs/explainers/noise-model-v3-wander.qmd` section 3.3a. No model is evaluated and "
        "nothing is fitted to a likelihood.",
        "",
        "**What is measured.** Per window and per block (0.25 / 0.5 / 1.0 s) on the 2048/512 "
        "flight front end (16 kHz, periodic Hann; R4's 8192-point frames are longer than a "
        "block): the ORDER-TRACKED line level of every rotor order -- the 3 bins nearest the "
        "frame's own label carrier `k f_r(n)` (98-100 % of a Hann line at any sub-bin offset), "
        "block-averaged, minus the local floor (q25 of the cells 3-12 bins either side) -- and "
        "the comb-masked floor level of every band (every order +-1 bin, lines >= 10 dB +-3 "
        "bins). The model's speed laws (line `(f/80)^2`, floor `mean_i (f_i/80)^2`, the prior "
        "centres every short-span pool is pinned at) are removed per block. Everything is "
        "the MIC MEAN of the dB levels.",
        "",
        "**Centring.** The v3 fit shares `p_ik` and the floor across its pool, so its block "
        "latents carry EVERY departure from the rig mean at the block's speed -- slow drift "
        "inside a flight and the offsets between windows alike. The primary estimate "
        "therefore centres each line (each floor band) on its mean over all windows of the "
        "same regime (DREGON: flight; Michael's: cruise, standby) and takes lag products "
        "inside windows only. The `within window` rows centre every window on its own mean: "
        "the wander inside a flight alone, blind to anything slower than the window.",
        "",
        "**Estimator.** The explainer's three lines, with the centring put back: under an OU "
        "of `(sigma^2, rho = e^{-T_b/tau})` plus independent block noise, every pooled "
        "centred lag product has an exact expectation linear in `sigma^2` "
        "(`wander.fit_ou`); `sigma^2` solves lag 0 and `tau` the lag-1 product, or the "
        "lags 1-4 products (both reported). `sigma_d` is the lag-0 CROSS moment of two "
        "distinct lines of the same rotor -- the explainer's 'covariance between two orders "
        "of one rotor in the same block', no noise term -- with `tau_d` over lags 1-4 (a "
        "pair's lag-1 product alone is too small against its scatter); `sigma_v` from the "
        "auto moments minus the fitted rotor-common part; `sigma_u` likewise from cross "
        "moments of two distinct floor bands, `sigma_uj` the rest. The explainer's raw "
        "arithmetic (`Var_b(y) - s^2`, `Cov_1 / sigma^2`, "
        "within-window, no centring correction) is reported as `naive`: on 4-8 s windows "
        "(8-16 blocks) it is biased low in both sigma and tau whenever tau is not short "
        "against the window (synthetic check in `tests/experiments/test_noise_model_wander.py`).",
        "",
        "**Block noise.** The explainer's `s^2 = (10/ln10)^2 psi_1(n_eff)` with `n_eff = "
        "frames x bins x lineFrac^2` assumes independent exponential cells. On this front "
        "end the frames overlap 75 % and adjacent Hann bins correlate, so a 15 x 3 rectangle "
        "of Gaussian-noise cells carries the information of ~45/3.1 independent ones; and a "
        "narrow line of steady amplitude -- the model's line -- has no exponential scatter of "
        "its own. The estimates therefore use the MEASURED block noise: for lines, the "
        "variance of the block mean from the scatter of the frames inside the block (lag "
        "0-3 autocovariance, the overlap's reach, centring-corrected); for the floor, whose "
        "cells are Gaussian noise, the exponential law at the exact correlated-cell count of "
        "the surviving cells, times the measured mic-coherence count for the mic mean. The "
        "explainer's count is a robustness row.",
        "",
    ]
    out += ["## Windows", ""]
    out += [
        "| rig | set | windows | duration s | carrier rev/s (min-max of window means) | recordings |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for rig, blk in rigs.items():
        tags: dict[str, list[dict[str, Any]]] = {}
        for w in blk["windows"]:
            tags.setdefault(w["tag"], []).append(w)
        for tag, ws in tags.items():
            cm = [float(np.mean(w["carrier_mean_rev_s"])) for w in ws]
            durs = sorted({round(w["duration_s"], 2) for w in ws})
            out.append(
                f"| {rig} | {tag} | {len(ws)} | {'/'.join(f'{d:g}' for d in durs)} | "
                f"{min(cm):.1f}-{max(cm):.1f} | {len({w['recording'] for w in ws})} |"
            )
    out += [""]

    out += ["## Resolvable lines", ""]
    out += [
        f"A (window, rotor, order) track is RESOLVABLE when its block-level prominence (mic-summed "
        f"line cells over mic-summed local floor) is >= {W.PROMINENCE_DB:g} dB in >= "
        f"{100 * W.PROMINENT_BLOCK_FRAC:g} % of the window's {CHOSEN_BLOCK_S:g} s blocks, and "
        f"ROTOR-DOMINANT when >= {100 * W.OWN_SHARE_MIN:g} % of its cells' line power is its own "
        "(window-mean line powers of every other rotor's orders times their Hann capture, "
        "per frame): with four rotors a few rev/s apart, another rotor's order sits within "
        "a bin or two of most lines. A track is USED when, in addition, its median measured "
        f"block sd is under {MAX_TRACK_NOISE_DB:g} dB (explainer section 1.5 (2): a noisier "
        "order's latent is prior-dominated); blocks under "
        f"{W.MIN_BLOCK_PROMINENCE_DB:g} dB prominence or over {MAX_BLOCK_NOISE_DB:g} dB "
        "measured sd are missing values. The line set is fixed at the chosen block length "
        "and used at every block length; the rotor-dominant set is used when its same-rotor "
        f"pairs share >= {MIN_SAME_ROTOR_BLOCKS} blocks, else every resolvable line.",
        "",
        "Counts are `resolvable / rotor-dominant / resolvable & sd <= 2 dB / dominant & sd <= 2 dB`.",
        "",
        "| rig | orders measured / window | tracks | distinct lines | line set used |",
        "| --- | --- | --- | --- | --- |",
    ]
    for rig, blk in rigs.items():
        rc = blk["resolvable"]
        out.append(
            f"| {rig} | {rc['orders_measured_per_window']:.0f} | {_slash(rc['tracks'])} | "
            f"{_slash(rc['distinct_lines'])} | {blk['line_set']} |"
        )
    out += ["", "By order group (tracks):", ""]
    out += [
        "| rig | " + " | ".join(g for g, _, _ in ORDER_GROUPS) + " |",
        "|" + " --- |" * (len(ORDER_GROUPS) + 1),
    ]
    for rig, blk in rigs.items():
        bg = blk["resolvable"]["by_group"]
        out.append(f"| {rig} | " + " | ".join(_slash(bg[g]) for g, _, _ in ORDER_GROUPS) + " |")
    out += ["", "By rotor (tracks):", ""]
    for rig, blk in rigs.items():
        br = blk["resolvable"]["by_rotor"]
        out.append(
            f"- {rig}: "
            + ", ".join(f"rotor {int(r) + 1}: {_slash(v)}" for r, v in sorted(br.items()))
        )
    out += ["", "Lines (rotor:order, windows):", ""]
    for rig, blk in rigs.items():
        bl = blk["resolvable"]["by_line"]
        out.append(
            f"- {rig}: "
            + ", ".join(
                f"{int(k.split(':')[0]) + 1}:{k.split(':')[1]} ({_slash(v)})" for k, v in bl.items()
            )
        )
    out += [""]

    out += ["## Block noise", ""]
    out += [
        "| rig | block s | line blocks | line s^2 measured (median) | line s^2 explainer (median) | measured / explainer (median) | Gaussian-cell overlap factor (3-bin line rectangle) | floor s^2 measured | floor s^2 explainer | floor equiv / cells | floor mic-coherence count |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rig, blk in rigs.items():
        for bs, nz in blk["noise"].items():
            out.append(
                f"| {rig} | {bs} | {nz['line_blocks']} | {_f(nz['line_s2_measured_median_db2'], '{:.3f}')} | "
                f"{_f(nz['line_s2_explainer_median_db2'], '{:.3f}')} | "
                f"{_f(nz['line_ratio_measured_over_explainer_median'])} | "
                f"{_f(nz['overlap_inflation_line_rectangle'])} | "
                f"{_f(nz['floor_s2_measured_median_db2'], '{:.3f}')} | "
                f"{_f(nz['floor_s2_explainer_median_db2'], '{:.3f}')} | "
                f"{_f(nz['floor_equiv_over_cells_median'])} | {_f(nz['floor_m_eff_median'])} |"
            )
    out += [
        "",
        "**What lag 0 hides** (chosen block length, rig-centred). A fit from lags 1-4 alone "
        "leaves a lag-0 excess, the nugget. For a PAIR of tracks it is block noise the two "
        "SHARE (a broadband event lifting every line and band of a block together), which "
        "would inflate the lag-0 covariance the common parts `d` and `u` are read from: a "
        "positive nugget is that inflation, a negative one says the OU shape over-predicts "
        "lag 0 from lags 1-4 (no sign of shared noise). For a track with itself it is the "
        "block noise the data show beyond the MEASURED one: near zero validates the "
        "within-block noise estimate.",
        "",
        "| rig | line pairs (same rotor): nugget dB^2 | floor band pairs: nugget dB^2 | line auto from lags 1-4: sd dB / tau s | its nugget over the measured noise dB^2 | measured line noise, mean dB^2 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for rig, blk in rigs.items():
        ch = blk["chosen"][PRIMARY]
        lo = ch["lines"]["total_lags_only"]
        out.append(
            f"| {rig} | {_f(ch['lines']['common_lags_only'].get('nugget_db2'), '{:.3f}')} | "
            f"{_f(ch['floor']['common_lags_only'].get('nugget_db2'), '{:.3f}')} | "
            f"{_f(lo['sigma_db'])} / {_f(lo['tau_s'])} | {_f(lo.get('nugget_db2'), '{:.3f}')} | "
            f"{_f(lo.get('measured_noise_mean_db2'), '{:.3f}')} |"
        )
    out += [""]

    out += ["## The estimates", ""]
    out += [
        "sd in dB, tau in s. `d`: rotor-common line wander; `v`: per-line residual; `u`: floor "
        "level; `uj`: floor colour residual per control band; `tot`: all of a line's wander "
        "(d + v under one OU). Rig-centred (per regime), exact centred-moment fits, "
        "measured block noise; sigma of `d`/`u` at lag 0 and their tau over lags 1-4, sigma "
        "of `v`/`uj`/`tot` at lag 0 minus the noise and their tau at "
        f"{'lag 1' if PRIMARY == 'lag1' else 'lags 1-4'}.",
        "",
        f"### Window bootstrap at {CHOSEN_BLOCK_S:g} s",
        "",
        "Windows resampled with replacement within each regime; median [5 %, 95 %] and the "
        "share of draws in which a sigma clips to zero. `lag 1` and `lags 1-4` are the two "
        "tau estimators of the explainer (the common parts use lags 1-4 in both).",
        "",
        "| rig / tau from | " + " | ".join(c for _, c in NUM_COLS) + " |",
        "|" + " --- |" * (len(NUM_COLS) + 1),
    ]
    for rig, blk in rigs.items():
        bt = blk["bootstrap"]
        for name, label in (("lag1", "lag 1"), ("lags1_4", "lags 1-4")):
            row = bt[name]
            cells = []
            for k, _ in NUM_COLS:
                q = row[k]
                z = row["zero_share"].get(k)
                zs = f" (0 in {100 * z:.0f} %)" if z else ""
                cells.append(f"{_f(q[1])} [{_f(q[0])}, {_f(q[2])}]{zs}")
            out.append(f"| {rig} / {label} | " + " | ".join(cells) + " |")
    out += [
        "",
        "### Three block lengths",
        "",
    ]
    out += _num_head("rig / block s")
    for rig, blk in rigs.items():
        for bs, s in blk["block_length"].items():
            out.append(_num_row(f"{rig} / {bs}", s["numbers"]))
    out += ["", "The same, WITHIN-WINDOW centring (the wander inside a flight only):", ""]
    out += _num_head("rig / block s")
    for rig, blk in rigs.items():
        for bs, s in blk["block_length"].items():
            out.append(_num_row(f"{rig} / {bs}", s["within_window"]))
    out += [
        "",
        "The explainer's literal arithmetic (within-window, no centring correction) on the "
        "same tracks:",
        "",
    ]
    out += [
        "| rig / block s | line Var_b - s^2 -> sd | line tau lag-1 | line tau log-lin 1-4 | same-rotor cov -> sd_d | rotor-mean track tau lag-1 | floor Var_b - s^2 -> sd | floor cross-band cov -> sd_u | floor tau lag-1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rig, blk in rigs.items():
        for bs, s in blk["block_length"].items():
            nl, nf = s["naive_lines"], s["naive_floor"]
            rm = nl.get("rotor_mean_track") or {}
            out.append(
                f"| {rig} / {bs} | {_f(_sd(nl['total'].get('sigma2')))} | {_f(nl['total'].get('tau_lag1_s'))} | "
                f"{_f(nl['total'].get('tau_loglin_s'))} | {_f(_sd(nl['common'].get('sigma2')))} | "
                f"{_f(rm.get('tau_lag1_s'))} | {_f(_sd(nf['total'].get('sigma2')))} | "
                f"{_f(_sd(nf['common'].get('sigma2')))} | {_f(nf['common'].get('tau_lag1_s'))} |"
            )
    out += ["", "### Clips at zero", ""]
    out += [
        "Per TRACK (one line or band in one window), the explainer's `Var_b(y) - s^2 <= 0`; per "
        "rig LINE (rotor, order pooled over its windows), the naive and the centred estimates; "
        "and whether any pooled fit clipped.",
        "",
        "| rig / block s | line tracks clipped (measured noise) | floor tracks clipped | pooled fits clipped |",
        "| --- | --- | --- | --- |",
    ]
    for rig, blk in rigs.items():
        for bs, s in blk["block_length"].items():
            c = s["clips"]
            fits = [
                k
                for k in (
                    "lines_total_fit",
                    "lines_common_fit",
                    "lines_residual_fit",
                    "floor_common_fit",
                    "floor_residual_fit",
                )
                if c[k]
            ]
            out.append(
                f"| {rig} / {bs} | {c['line_tracks'][0]} / {c['line_tracks'][1]} | "
                f"{c['floor_tracks'][0]} / {c['floor_tracks'][1]} | {', '.join(fits) or 'none'} |"
            )
    out += [""]
    for rig, blk in rigs.items():
        lc, ex = blk["line_clips"], blk["robustness"]["noise_explainer"]["clips"]
        out.append(
            f"- {rig}: rig lines pooled over windows: {lc['naive_clipped']} / {lc['n']} clip with the "
            f"naive arithmetic, {lc['fit_clipped']} / {lc['n']} with the centred fit; with the "
            f"EXPLAINER's noise count, {ex['line_tracks'][0]} / {ex['line_tracks'][1]} line tracks and "
            f"{ex['floor_tracks'][0]} / {ex['floor_tracks'][1]} floor tracks clip."
        )
    out += [""]

    out += ["## Robustness at the chosen block length", ""]
    out += _num_head("rig / variant")
    for rig, blk in rigs.items():
        rob = blk["robustness"]
        for name, label in (("lag1", "tau at lag 1"), ("lags1_4", "tau over lags 1-4")):
            tag = " (primary)" if name == PRIMARY else ""
            out.append(_num_row(f"{rig} / {label}{tag}", blk["chosen"][name]["numbers"]))
        out.append(_num_row(f"{rig} / explainer noise count", rob["noise_explainer"]))
        for k, v in rob.items():
            if k.startswith("lines_"):
                out.append(
                    _num_row(f"{rig} / {k.replace('_', ' ')} ({v['n_line_tracks']} tracks)", v)
                )
            if k == "orders_aero":
                out.append(
                    _num_row(f"{rig} / orders <= {AERO_MAX_ORDER} ({v['n_line_tracks']} tracks)", v)
                )
            if k.startswith("amp_exp_"):
                out.append(_num_row(f"{rig} / speed exponent {k.split('_')[-1]}", v))
        q = rob["per_mic_quantiles"]
        out.append(_num_row(f"{rig} / single mic, min", {k: v[0] for k, v in q.items()}))
        out.append(_num_row(f"{rig} / single mic, median", {k: v[1] for k, v in q.items()}))
        out.append(_num_row(f"{rig} / single mic, max", {k: v[2] for k, v in q.items()}))
        th = rob["floor_third_octave"]
        out.append(_num_row(f"{rig} / floor on 1/3-oct bands", th))
    out += [""]
    for rig, blk in rigs.items():
        sl = blk["robustness"]["speed_slope_within_window"]
        out.append(
            f"- {rig}: within-window speed exponent of the line levels (pooled slope of dB on "
            f"10 log10 f over {sl['n_blocks']} blocks, speed spread {math.sqrt(max(sl['x_var_db2'], 0.0)):.2f} dB): "
            f"{_f(sl['slope'])} (the model's is {AMP_EXP:g})."
        )
    out += [""]

    out += ["## Breakdown", ""]
    out += [
        "Per rotor (auto = all of a line's wander; common = same-rotor pairs), per order group, "
        "per floor band group; sd dB / tau s / pairs. Centred lag-1 fit, measured noise.",
        "",
        "| rig | slice | sd | tau | pairs |",
        "| --- | --- | --- | --- | --- |",
    ]
    for rig, blk in rigs.items():
        for r, f in blk["by_rotor"]["total"].items():
            out.append(
                f"| {rig} | rotor {int(r) + 1} auto | {_f(f['sigma_db'])} | {_f(f['tau_s'])} | {f['n_pairs']} |"
            )
        for r, f in blk["by_rotor"]["common"].items():
            out.append(
                f"| {rig} | rotor {int(r) + 1} common | {_f(f['sigma_db'])} | {_f(f['tau_s'])} | {f['n_pairs']} |"
            )
        for g, f in blk["by_order_group"].items():
            out.append(
                f"| {rig} | orders {g} auto | {_f(f['sigma_db'])} | {_f(f['tau_s'])} | {f['n_pairs']} |"
            )
        acr = blk["chosen"][PRIMARY]["lines"].get("across")
        if acr:
            out.append(
                f"| {rig} | lines of DIFFERENT rotors, pairs | {_f(acr['sigma_db'])} | {_f(acr['tau_s'])} | {acr['n_pairs']} |"
            )
        for g, f in blk["floor_by_band_group"].items():
            out.append(
                f"| {rig} | floor {g} auto | {_f(f['sigma_db'])} | {_f(f['tau_s'])} | {f['n_pairs']} |"
            )
    out += [""]

    out += ["## Verdict", ""]
    out += payload["verdict"]
    out += [
        "",
        f"Legacy sliders (`{LEGACY_YAML}`, `ranges` of the two stochastic sources): "
        + "; ".join(
            f"{rig}: harm_gp_std_db {v['harm_gp_std_db']}, harm_gp_tau_s {v['harm_gp_tau_s']}, "
            f"harm_coherence {v['harm_coherence']}, floor_gp_std_db {v['floor_gp_std_db']}, "
            f"floor_gp_tau_s {v['floor_gp_tau_s']}"
            for rig, v in leg.items()
        )
        + ".",
        "",
        "## Figures",
        "",
    ]
    out += [f"- `{f}`" for f in payload["figures"]]
    out += ["", "Detail: `wander_detail.json` (every number above).", ""]
    return "\n".join(out)


def _sd(v: Any) -> float:
    return math.sqrt(v) if isinstance(v, int | float) and math.isfinite(v) and v >= 0 else NAN


def _slash(v: Sequence[Any]) -> str:
    return " / ".join(str(x) for x in v)


def _ci(bt: dict[str, Any], key: str) -> str:
    q = bt.get(key) or [NAN, NAN, NAN]
    return f"[{_f(q[0])}, {_f(q[2])}]"


def _vs_range(value: float, rng: Any) -> str:
    """``below`` / ``inside`` / ``above`` a legacy ``[lo, hi]`` slider range."""
    if not isinstance(rng, list | tuple) or len(rng) != 2 or not math.isfinite(value):
        return "n/a"
    lo, hi = float(min(rng)), float(max(rng))
    return "below" if value < lo else ("above" if value > hi else "inside")


def _verdict_body(
    num: dict[str, Any],
    bt: dict[str, Any],
    lg: dict[str, Any],
    s_blk: float,
    share: float,
    loose: list[str],
    tot: float,
    u: float,
    sig_bs: str,
    u_bs: str,
) -> list[str]:
    out = [
        f"Line level around the rig mean: sd {_f(tot)} dB {_ci(bt, 'sigma_total_db')}, "
        f"tau {_f(num['tau_total_s'])} s, against a measured block noise of sd {_f(s_blk)} dB;",
        f"floor level: sd_u {_f(u)} dB {_ci(bt, 'sigma_u_db')}, tau_u {_f(num['tau_u_s'])} s "
        f"{_ci(bt, 'tau_u_s')}; floor colour: sd_uj {_f(num['sigma_uj_db'])} dB "
        f"{_ci(bt, 'sigma_uj_db')}, tau_uj {_f(num['tau_uj_s'])} s {_ci(bt, 'tau_uj_s')}.",
        f"Split of the line wander: rotor-common sd_d {_f(num['sigma_d_db'])} dB "
        f"{_ci(bt, 'sigma_d_db')} ({_f(100 * share, '{:.0f}')} % of the variance), tau_d "
        f"{_f(num['tau_d_s'])} s {_ci(bt, 'tau_d_s')}; per line sd_v {_f(num['sigma_v_db'])} dB "
        f"{_ci(bt, 'sigma_v_db')}, tau_v {_f(num['tau_v_s'])} s {_ci(bt, 'tau_v_s')}.",
    ]
    if loose:
        out.append(
            f"The correlation times of {', '.join(loose)} are NOT pinned by these windows "
            "(bootstrap 95 % over 5 % above 4x); the sigmas are."
        )
    out.append(
        f"Across block lengths 0.25 / 0.5 / 1.0 s the line sd reads {sig_bs} dB and the floor "
        f"level sd {u_bs} dB."
    )
    hs, ht = lg.get("harm_gp_std_db"), lg.get("harm_gp_tau_s")
    fs, ft = lg.get("floor_gp_std_db"), lg.get("floor_gp_tau_s")
    out.append(
        f"Against the legacy sliders: line sd {_vs_range(tot, hs)} harm_gp_std_db {hs}, line tau "
        f"{_vs_range(float(num['tau_total_s']), ht)} harm_gp_tau_s {ht}, common share "
        f"{_vs_range(share, lg.get('harm_coherence'))} harm_coherence {lg.get('harm_coherence')}; "
        f"floor sd {_vs_range(u, fs)} floor_gp_std_db {fs}, floor tau "
        f"{_vs_range(float(num['tau_u_s']), ft)} floor_gp_tau_s {ft}."
    )
    return out


def _verdict_text(
    rig: str,
    blk: dict[str, Any],
    num: dict[str, Any],
    bt: dict[str, Any],
    lg: dict[str, Any],
    s_blk: float,
    share: float,
    loose: list[str],
) -> str:
    rc = blk["resolvable"]
    sens = blk["block_length"]
    sig_bs = ", ".join(_f(sens[b]["numbers"]["sigma_total_db"]) for b in sens)
    u_bs = ", ".join(_f(sens[b]["numbers"]["sigma_u_db"]) for b in sens)
    tot, u = float(num["sigma_total_db"]), float(num["sigma_u_db"])
    parts = [
        f"**{rig}.** {blk['chosen'][PRIMARY]['n_line_tracks']} line tracks used "
        f"({rc['tracks'][0]} resolvable, {rc['tracks'][1]} rotor-dominant; set: {blk['line_set']}).",
    ]
    return " ".join(parts + _verdict_body(num, bt, lg, s_blk, share, loose, tot, u, sig_bs, u_bs))


def verdict(rigs: dict[str, Any], leg: dict[str, Any]) -> list[str]:
    """One paragraph per rig, every number read off ``rigs``."""
    out = []
    for rig, blk in rigs.items():
        num = blk["chosen"][PRIMARY]["numbers"]
        bt = blk["bootstrap"][PRIMARY]
        noise = blk["chosen"][PRIMARY]["lines"]["total_lags_only"].get("measured_noise_mean_db2")
        s_blk = math.sqrt(noise) if isinstance(noise, int | float) and noise > 0 else NAN
        tot = float(num["sigma_total_db"])
        share = num["sigma_d_db"] ** 2 / tot**2 if tot > 0 else NAN
        loose = [
            k.split("_")[1]
            for k in ("tau_d_s", "tau_v_s", "tau_u_s", "tau_uj_s")
            if not (bt[k][0] and bt[k][0] > 0 and bt[k][2] / bt[k][0] < 4.0)
        ]
        out.append(_verdict_text(rig, blk, num, bt, leg.get(rig, {}), s_blk, share, loose))
        out.append("")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--rigs", nargs="+", default=list(RIGS), choices=RIGS)
    ap.add_argument("--limit", type=int, default=None, help="windows per rig (smoke runs)")
    args = ap.parse_args(list(argv) if argv is not None else None)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = rig_rows()
    all_wins: dict[str, list[WindowData]] = {}
    details: dict[str, Any] = {}
    figs: dict[str, dict[str, Any]] = {}
    for rig in args.rigs:
        sel = rows[rig][: args.limit] if args.limit else rows[rig]
        print(f"[{rig}] measuring {len(sel)} windows ...", flush=True)
        wins = [measure_window(spec, tag, regime, BLOCK_LENGTHS_S) for spec, tag, regime in sel]
        all_wins[rig] = wins
        details[rig], figs[rig] = analyse_rig(rig, wins)
        num = details[rig]["chosen"][PRIMARY]["numbers"]
        print(f"[{rig}] " + ", ".join(f"{k}={v:.3f}" for k, v in num.items()), flush=True)

    figures = []
    picks = example_tracks(all_wins, {r: d["line_set"] for r, d in details.items()})
    if picks:
        tracks_figure(all_wins, picks, out_dir / "example_tracks.png")
        figures.append("example_tracks.png")
    acf_figure(figs, out_dir / "autocovariance_vs_lag.png")
    figures.append("autocovariance_vs_lag.png")

    leg = legacy_ranges()
    payload = dict(
        schema=SCHEMA + "/detail",
        generated_by="scripts/noise_v3_measure_wander.py",
        git_head=git_head(),
        config=dict(
            block_lengths_s=list(BLOCK_LENGTHS_S),
            chosen_block_s=CHOSEN_BLOCK_S,
            amp_exp=AMP_EXP,
            floor_exp=FLOOR_EXP,
            f_ref_rev_s=F_REF_REV_S,
            prominence_db=W.PROMINENCE_DB,
            prominent_block_frac=W.PROMINENT_BLOCK_FRAC,
            min_block_prominence_db=W.MIN_BLOCK_PROMINENCE_DB,
            own_share_min=W.OWN_SHARE_MIN,
            line_half_bins=W.LINE_HALF_BINS,
            local_floor=[W.LOCAL_FLOOR_GAP_BINS, W.LOCAL_FLOOR_HALF_BINS, W.LOCAL_FLOOR_QUANTILE],
            comb_mask_half_bins=W.COMB_MASK_HALF_BINS,
            strong_line_db=W.STRONG_LINE_DB,
            strong_mask_half_bins=W.STRONG_MASK_HALF_BINS,
            min_same_rotor_blocks=MIN_SAME_ROTOR_BLOCKS,
            limit=args.limit,
        ),
        legacy=leg,
        figures=figures,
        example_tracks=picks,
        rigs=details,
    )
    clean = _clean(payload)
    clean["verdict"] = verdict(clean["rigs"], leg)
    (out_dir / "wander_detail.json").write_text(json.dumps(clean, indent=1) + "\n")
    for rig in details:
        c = _clean(contract(rig, details[rig], all_wins[rig]))
        (out_dir / f"{rig}.json").write_text(json.dumps(c, indent=1) + "\n")
    (out_dir / "findings.md").write_text(findings_md(clean))
    print(
        f"wrote {out_dir}/{{{','.join(details)}}}.json, wander_detail.json, findings.md", flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
