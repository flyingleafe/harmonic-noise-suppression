"""Is each rig's per-microphone level deviation ONE gain per channel?

The noise-model-v2 fit carries a free per-microphone gain (and, in the richer
variants, a per-microphone floor). That freedom is only earned if the level
difference between the eight channels of a rig is genuinely a *per-channel*
quantity. If instead the deviation is the same number for every window and
every band, it is a fixed sensitivity difference: the data pipeline can divide
it out once and the fit needs no free per-mic gains at all.

This module measures that, and nothing else. No model is evaluated, nothing is
fitted to a likelihood; the only inputs are periodograms of REAL windows.

What is measured
----------------

For every real window of a rig (all eight microphones):

* ``L[window, mic, band]`` -- 1/3-octave band levels, 30-7900 Hz, in dB, on the
  same band grid :func:`experiments.stochastic_fit.rig_sampler.third_octave`
  gives every other level in this campaign. Three variants:

  ``full``  the whole periodogram;
  ``floor`` every bin within +-``COMB_MASK_HALF_HZ`` of a rotor order (at the
            window's own per-FRAME carriers) removed before banding -- on the
            flight front end (``df`` = 7.8125 Hz) that is exactly the +-1 bin
            the question asks for, and on the much finer bench grid it is the
            +-2 Hz that actually covers a line;
  ``comb``  the order-TRACKED line levels: the peak of each order inside its
            +-``COMB_MASK_HALF_HZ`` window, power-averaged over the orders that
            fall in a band and over frames.

* the window's rotor-mean carrier, its recording id, and its regime.

The rank test
-------------

``D[window, mic, band] = L - mean_over_mics(L)`` is the per-mic deviation (zero
mean over mics by construction). Three least-squares models are fitted to it:

    ``const``       D ~ g[mic]              a fixed sensitivity
    ``per_band``    D ~ g[mic, band]        frequency-dependent (a transfer)
    ``per_window``  D ~ g[mic, window]      time-varying (flow, orientation)

Each is the corresponding mean of ``D``, because the design is a set of
indicator columns; the useful output is what each model leaves behind. If
``const`` already explains ~all of the variance with a residual well under a
dB, the channel gains are a pipeline constant. If ``per_band`` is much better
than ``const``, the deviation is a frequency response and one scalar per
channel cannot carry it. If ``per_window`` is much better, it is not a property
of the channel at all.

The comb and the floor are tested separately, and their gains compared: a
sensitivity difference must move the rotor lines and the broadband floor by the
same dB, while anything flow-related moves only the floor.

The wind test
-------------

DREGON's per-mic floor excess is plotted against band (wind and self-noise sit
low) and against the window's rotor-mean carrier (rotor-driven noise scales
with speed, forward-flight wind need not). Michael's rig is the control. The
DREGON *flight* windows are all within a hair of 80 rev/s, so the speed axis
there is carried by the DREGON BENCH cells (one motor, throttle 50-90, the same
eight-microphone array), which are reported separately and clearly labelled.

Usage
-----

    PYTHONPATH=src python scripts/_mic_gain_rank.py
    PYTHONPATH=src python scripts/_mic_gain_rank.py --no-bench --no-extra
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.noise_model import supports as SUP
from experiments.noise_model.rig_sampler import REAL_WINDOW_SETS, _real_window_specs
from experiments.stochastic_fit import clips as C
from experiments.stochastic_fit.rig_sampler import third_octave

OUT_DIR = Path("results/noise_v2/mic_gains")

#: The band grid. 30 Hz is the model's own observation floor
#: (``supports.BAND_LEVEL_F_MIN``); 7900 Hz is where every 1/3-octave level in
#: this campaign stops.
BAND_LO_HZ = 30.0
BAND_HI_HZ = 7900.0

#: Band groups every residual table is broken down over.
BAND_GROUPS: tuple[tuple[str, float, float], ...] = (
    ("30-300", 30.0, 300.0),
    ("300-1k", 300.0, 1000.0),
    ("1k-3k", 1000.0, 3000.0),
    ("3k-8k", 3000.0, 8000.0),
)

#: The wind band: below this the excess is "low-frequency".
WIND_LO_HZ = 500.0

#: A speed slope needs a speed axis. Below this carrier span (in octaves) the
#: regression is reported but declared unidentified rather than believed: the
#: DREGON flight windows all sit within a few per cent of 80 rev/s.
MIN_SPEED_SPAN_LOG2 = 0.25

#: A slope this large (in dB per octave of rotor speed), and clearing two of
#: its own standard errors, counts as "the excess scales with speed". One dB
#: per octave is the smallest slope that moves a level by more than the 1 dB
#: window-to-window scatter over the ~1.3-octave span the data has.
SPEED_SLOPE_DB = 1.0

#: Half-width of the order mask, in Hz, floored at one bin. On the flight front
#: end (2048-point FFT at 16 kHz, df = 7.8125 Hz) this is one bin either side,
#: as asked; on a bench support (one FFT over the whole segment, df < 0.1 Hz) it
#: is +-2 Hz, which is what it takes to cover a real line plus its decoherence
#: pedestal (``supports.BENCH_LINE_BAND_HZ`` is 1 Hz for the peak alone).
COMB_MASK_HALF_HZ = 2.0

#: Extra DREGON flight windows: every nosource flight recording of both rooms,
#: on a non-overlapping 4 s grid, keeping only windows where every rotor stays
#: above this speed (i.e. motors running, no spin-down tail).
DREGON_EXTRA_MIN_RPS = 40.0
DREGON_EXTRA_DUR_S = 4.0
#: The six DREGON flight recordings with NO acoustic source playing. The other
#: room-1 flights carry speech or white noise and cannot measure a noise floor.
DREGON_FLIGHT_RECORDINGS = (
    "free-flight_nosource_room1",
    "free-flight_nosource_room2",
    "hovering_nosource_room2",
    "rectangle_nosource_room2",
    "spinning_nosource_room2",
    "updown_nosource_room2",
)

#: The auxiliary speed axis: the DREGON bench cells, one motor at a time.
BENCH_ROTORS = SUP.DREGON_BENCH_ROTORS
BENCH_THROTTLES = SUP.DREGON_BENCH_THROTTLES

MODELS = ("const", "per_band", "per_regime", "per_recording", "per_window")

#: The level variants. ``floor_q25`` is the robustness twin of ``floor``: the
#: same cells, read at a low quantile instead of a power mean.
VARIANTS = ("full", "floor", "comb", "floor_q25")
#: Variants the deviation heat map shows (the fourth is a check, not a picture).
HEATMAP_VARIANTS = ("full", "floor", "comb")

#: Quantile of the unmasked cells behind ``floor_q25``.
FLOOR_QUANTILE = 0.25
#: A band needs this many tracked orders per frame, on average, before a comb
#: level is reported for it.
COMB_MIN_ORDERS_PER_FRAME = 1.0


# ---------------------------------------------------------------------------
# bands
# ---------------------------------------------------------------------------


def band_edges(f_lo: float = BAND_LO_HZ, f_hi: float = BAND_HI_HZ) -> np.ndarray:
    """The 1/3-octave edges :func:`third_octave` uses, restated so the masked
    variants can average over the SAME bins the unmasked one does."""
    return 2.0 ** np.arange(np.log2(f_lo), np.log2(f_hi) + 1e-9, 1.0 / 3.0)


@dataclass(frozen=True)
class BandGrid:
    """The kept bands of one frequency axis: centre, edges and member bins."""

    centres: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    bins: tuple[np.ndarray, ...]

    @property
    def n(self) -> int:
        return int(self.centres.size)


def band_grid(freqs: np.ndarray, f_lo: float = BAND_LO_HZ, f_hi: float = BAND_HI_HZ) -> BandGrid:
    """:func:`third_octave`'s bands for ``freqs``, with their bin indices.

    Empty bands are dropped exactly as :func:`third_octave` drops them, so the
    centres returned here and the centres it returns are the same vector.
    """
    edges = band_edges(f_lo, f_hi)
    f = np.asarray(freqs, dtype=np.float64)
    cen, los, his, bins = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = np.nonzero((f >= lo) & (f < hi))[0]
        if idx.size == 0:
            continue
        cen.append(float(np.sqrt(lo * hi)))
        los.append(float(lo))
        his.append(float(hi))
        bins.append(idx)
    return BandGrid(np.asarray(cen), np.asarray(los), np.asarray(his), tuple(bins))


def group_mask(centres: np.ndarray, lo: float, hi: float) -> np.ndarray:
    c = np.asarray(centres, dtype=np.float64)
    return (c >= lo) & (c < hi)


# ---------------------------------------------------------------------------
# one window: full / floor / comb band levels
# ---------------------------------------------------------------------------


def comb_mask(freqs: np.ndarray, carriers: np.ndarray, half_bins: int) -> np.ndarray:
    """``(n_frames, n_bins)`` boolean: True where a rotor order sits.

    The carriers are the window's OWN per-frame rotor speeds, so a window whose
    speed drifts masks a drifting comb rather than a smeared average one.
    """
    df = float(freqs[1] - freqs[0])
    f_max = float(freqs[-1])
    n_f = int(freqs.size)
    n_t = int(carriers.shape[1])
    mask = np.zeros((n_t, n_f), dtype=bool)
    offs = np.arange(-half_bins, half_bins + 1)
    for t in range(n_t):
        for f0 in carriers[:, t]:
            if not np.isfinite(f0) or f0 <= 0.0:
                continue
            k = np.arange(1, int(np.floor(f_max / float(f0))) + 1)
            b = np.rint(k * float(f0) / df).astype(np.int64)
            cols = np.clip((b[:, None] + offs[None, :]).ravel(), 0, n_f - 1)
            mask[t, cols] = True
    return mask


def floor_levels(
    power: np.ndarray, mask: np.ndarray, grid: BandGrid
) -> tuple[np.ndarray, np.ndarray]:
    """``(mean, low-quantile)`` band levels over the bins no order occupies.

    Both are ``(n_mics, n_bands)`` dB over the unmasked ``(frame, bin)`` cells;
    a band whose every cell is masked comes out NaN rather than fabricated.

    The MEAN is the asked-for floor. The QUANTILE
    (:data:`FLOOR_QUANTILE` of the same cells) is the robustness check that
    matters here: on a four-rotor flight window at this resolution the order
    mask already covers most bins, so a single stray tonal -- or a line the
    COMMANDED rotor speed misplaces -- lands in the surviving cells and lifts
    the mean. A low quantile of the same cells cannot be moved by one loud
    cell, so agreement between the two says the floor is a floor.
    """
    keep_f = (~mask).astype(np.float64)  # (T, F)
    num = np.einsum("mtf,tf->mf", power, keep_f)  # (M, F)
    den = keep_f.sum(axis=0)  # (F,)
    out = np.full((power.shape[0], grid.n), np.nan)
    qout = np.full((power.shape[0], grid.n), np.nan)
    for b, idx in enumerate(grid.bins):
        d = float(den[idx].sum())
        if d <= 0.0:
            continue
        out[:, b] = 10.0 * np.log10(np.maximum(num[:, idx].sum(axis=1) / d, 1e-300))
        cells = power[:, :, idx][:, ~mask[:, idx]]  # (M, n_cells)
        qout[:, b] = 10.0 * np.log10(np.maximum(np.quantile(cells, FLOOR_QUANTILE, axis=1), 1e-300))
    return out, qout


def comb_levels(
    power: np.ndarray, freqs: np.ndarray, carriers: np.ndarray, half_bins: int, grid: BandGrid
) -> tuple[np.ndarray, np.ndarray]:
    """``((n_mics, n_bands) dB, (n_bands,) order count)`` of the tracked lines.

    Each order's level is the PEAK of its own ``+-half_bins`` window, which is
    the line itself rather than the line plus a band of floor; orders are then
    power-averaged inside each 1/3-octave band and over frames.
    """
    n_m, n_t, n_f = power.shape
    df = float(freqs[1] - freqs[0])
    f_max = float(grid.hi[-1])
    offs = np.arange(-half_bins, half_bins + 1)
    num = np.zeros((n_m, grid.n))
    cnt = np.zeros(grid.n)
    for t in range(n_t):
        lines = []
        for f0 in carriers[:, t]:
            if not np.isfinite(f0) or f0 <= 0.0:
                continue
            k = np.arange(1, int(np.floor(f_max / float(f0))) + 1)
            lines.append(k * float(f0))
        if not lines:
            continue
        f_lines = np.concatenate(lines)
        b = np.rint(f_lines / df).astype(np.int64)
        band_of = np.searchsorted(grid.lo, f_lines, side="right") - 1
        ok = (
            (band_of >= 0)
            & (band_of < grid.n)
            & (b - half_bins >= 0)
            & (b + half_bins < n_f)
            & (f_lines < grid.hi[np.clip(band_of, 0, grid.n - 1)])
        )
        if not ok.any():
            continue
        b, band_of = b[ok], band_of[ok]
        peak = power[:, t, :][:, b[:, None] + offs[None, :]].max(axis=2)  # (M, L)
        for m in range(n_m):
            num[m] += np.bincount(band_of, weights=peak[m], minlength=grid.n)
        cnt += np.bincount(band_of, minlength=grid.n)
    out = np.full((n_m, grid.n), np.nan)
    # A band that holds an order in only a handful of frames (the fundamental
    # dipping in and out of the band as the speed drifts) has no line level
    # worth reporting: require one order per frame on average.
    hit = cnt >= COMB_MIN_ORDERS_PER_FRAME * max(n_t, 1)
    out[:, hit] = 10.0 * np.log10(np.maximum(num[:, hit] / cnt[hit], 1e-300))
    return out, cnt


@dataclass(frozen=True)
class WindowInfo:
    """Provenance and speed of one measured window."""

    name: str
    tag: str
    recording: str
    regime: str
    carrier_rev_s: float
    carrier_min_rev_s: float
    carrier_max_rev_s: float
    duration_s: float
    n_frames: int
    n_mics: int
    comb_masked_frac: float


def measure_window(sup: SUP.Support, *, tag: str, regime: str, verify: bool = False) -> dict:
    """Band levels (full / floor / comb) plus provenance for one support."""
    power = np.asarray(sup.power, dtype=np.float64)
    freqs = np.asarray(sup.freqs_hz, dtype=np.float64)
    carriers = np.asarray(sup.carrier_rev_s, dtype=np.float64)
    grid = band_grid(freqs)
    df = float(freqs[1] - freqs[0])
    half = max(1, int(np.ceil(COMB_MASK_HALF_HZ / df)))

    mic_power = power.mean(axis=1)  # (M, F), frame mean
    full = np.stack(
        [
            third_octave(freqs, 10.0 * np.log10(np.maximum(p, 1e-300)), BAND_LO_HZ, BAND_HI_HZ)[1]
            for p in mic_power
        ]
    )
    if verify:
        cen = third_octave(
            freqs, 10.0 * np.log10(np.maximum(mic_power[0], 1e-300)), BAND_LO_HZ, BAND_HI_HZ
        )[0]
        assert np.allclose(cen, grid.centres), "band grid disagrees with third_octave"
        mine = np.array(
            [
                10.0 * np.log10(mic_power[0][idx].mean())
                for idx in grid.bins  # the same power mean, band by band
            ]
        )
        assert np.allclose(mine, full[0], atol=1e-9), "band averaging disagrees with third_octave"

    mask = comb_mask(freqs, carriers, half)
    floor, floor_q25 = floor_levels(power, mask, grid)
    comb, orders = comb_levels(power, freqs, carriers, half, grid)
    info = WindowInfo(
        name=sup.name,
        tag=tag,
        recording=str(sup.meta.get("recording_id", sup.name)),
        regime=regime,
        carrier_rev_s=float(np.mean(carriers)),
        carrier_min_rev_s=float(np.min(carriers)),
        carrier_max_rev_s=float(np.max(carriers)),
        duration_s=float(sup.duration_s),
        n_frames=int(sup.n_frames),
        n_mics=int(sup.n_mics),
        comb_masked_frac=float(mask.mean()),
    )
    return {
        "info": info,
        "centres": grid.centres,
        "full": full,
        "floor": floor,
        "floor_q25": floor_q25,
        "comb": comb,
        "orders_per_band": orders,
        "half_bins": half,
        "df_hz": df,
    }


# ---------------------------------------------------------------------------
# which windows each rig gets
# ---------------------------------------------------------------------------


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] - 1e-9 and b[0] < a[1] - 1e-9


def dregon_extra_specs(frozen: Sequence[SUP.SupportSpec]) -> list[SUP.SupportSpec]:
    """A non-overlapping 4 s grid over every nosource DREGON flight recording.

    Windows that overlap one of the frozen/R5 supports are dropped, so the
    pooled set counts each stretch of audio once.
    """
    taken: dict[str, list[tuple[float, float]]] = {}
    for spec in frozen:
        rec = str(spec.args["recording"])
        s, d = float(spec.args["start_s"]), float(spec.args["dur_s"])
        taken.setdefault(rec, []).append((s, s + d))
    out: list[SUP.SupportSpec] = []
    for rec in C.iter_recordings(
        "DREGON-frames", DREGON_FLIGHT_RECORDINGS, rps_key=SUP.DREGON_RPS_KEY
    ):
        spans = taken.get(rec.recording_id, [])
        for t0, dur in C.windows(
            rec, seconds=DREGON_EXTRA_DUR_S, max_clips=10_000, min_rps=DREGON_EXTRA_MIN_RPS
        ):
            if any(_overlaps((t0, t0 + dur), s) for s in spans):
                continue
            out.append(SUP.flight_dregon(rec.recording_id, t0, dur))
    return out


def rig_specs(*, extra: bool) -> dict[str, list[tuple[SUP.SupportSpec, str, str]]]:
    """``{rig: [(spec, set tag, regime)]}`` for the two real rigs.

    The frozen sets come from :func:`_real_window_specs` -- the campaign's own
    definition of "the real windows of a rig" -- and are never re-derived here.
    """
    sets = _real_window_specs()
    regimes = {
        "dregon_score": "flight",
        "dregon_fit": "flight",
        "michaels_cruise": "cruise",
        "michaels_standby": "standby",
    }
    out: dict[str, list[tuple[SUP.SupportSpec, str, str]]] = {}
    for rig, tags in (("dregon", REAL_WINDOW_SETS["dregon"]), ("michaels", ("michaels_cruise",))):
        rows = [(s, t, regimes[t]) for t in tags for s in sets[t]]
        out[rig] = rows
    # Michael's rig is ONE rig: its standby windows are the same eight
    # microphones and belong in the same rank test (and they carry the only
    # real speed span either rig has).
    out["michaels"] += [(s, "michaels_standby", "standby") for s in sets["michaels_standby"]]
    if extra:
        frozen = [s for s, _, _ in out["dregon"]]
        out["dregon"] += [(s, "dregon_extra", "flight") for s in dregon_extra_specs(frozen)]
    return out


def bench_specs() -> list[tuple[SUP.SupportSpec, str, str]]:
    return [
        (SUP.bench_dregon_motor(rotor, thr), f"bench_{rotor}", f"bench_{rotor}")
        for rotor in BENCH_ROTORS
        for thr in BENCH_THROTTLES
    ]


def collect(rows: Sequence[tuple[SUP.SupportSpec, str, str]], *, limit: int | None = None) -> dict:
    """Measure every window of one rig; assert the band grids agree."""
    rows = list(rows)[: limit or len(rows)]
    infos: list[WindowInfo] = []
    stacks: dict[str, list[np.ndarray]] = {v: [] for v in VARIANTS}
    centres: np.ndarray | None = None
    orders: list[np.ndarray] = []
    for i, (spec, tag, regime) in enumerate(rows):
        sup = SUP.load_support(spec)
        m = measure_window(sup, tag=tag, regime=regime, verify=(i == 0))
        if centres is None:
            centres = m["centres"]
        elif not np.allclose(centres, m["centres"]):
            raise ValueError(f"{sup.name}: band grid differs from the rig's first window")
        for v in VARIANTS:
            stacks[v].append(m[v])
        orders.append(m["orders_per_band"])
        infos.append(m["info"])
    assert centres is not None
    n_mics = {i.n_mics for i in infos}
    if len(n_mics) != 1:
        raise ValueError(f"mixed microphone counts {sorted(n_mics)}")
    return {
        "infos": infos,
        "centres": centres,
        "levels": {v: np.stack(stacks[v]) for v in VARIANTS},
        "orders_per_band": np.stack(orders),
        "n_mics": int(next(iter(n_mics))),
    }


# ---------------------------------------------------------------------------
# the rank test
# ---------------------------------------------------------------------------


def _nanmean(a: np.ndarray, axis: Any) -> np.ndarray:
    """``np.nanmean`` without the all-NaN warning (all-NaN -> NaN)."""
    n = np.sum(np.isfinite(a), axis=axis)
    s = np.nansum(np.where(np.isfinite(a), a, 0.0), axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def _rms(a: np.ndarray) -> float:
    v = np.asarray(a, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(np.sqrt((v**2).mean())) if v.size else float("nan")


def deviations(levels: np.ndarray) -> np.ndarray:
    """``D[w, m, b] = L - mean_over_mics(L)``, the per-mic deviation in dB."""
    return levels - _nanmean(levels, axis=1)[:, None, :]


def model_pids(
    shape: tuple[int, int, int], recording: np.ndarray, regime: np.ndarray
) -> dict[str, np.ndarray]:
    """Each model as a PARTITION of the cells: one parameter id per cell.

    Every design here gives each cell exactly one free parameter, so the
    least-squares fit is the group mean of ``D`` and needs no solver, and the
    leave-one-window-out fit is the same mean with that window's contribution
    subtracted.

    ``per_regime`` is the operating-point model: one gain per channel per
    regime. On DREGON's flight set there is one regime and it collapses onto
    ``const``; on Michael's it asks whether standby and cruise see the same
    channel gains; on the bench it is one gain per channel per MOTOR, i.e. per
    source position.
    """
    n_w, n_m, n_b = shape
    w = np.arange(n_w)[:, None, None]
    m = np.arange(n_m)[None, :, None]
    b = np.arange(n_b)[None, None, :]
    r = np.asarray(recording, dtype=np.int64)[:, None, None]
    q = np.asarray(regime, dtype=np.int64)[:, None, None]
    return {
        "const": np.broadcast_to(m, shape).copy(),
        "per_band": np.broadcast_to(m * n_b + b, shape).copy(),
        "per_regime": np.broadcast_to(q * n_m + m, shape).copy(),
        "per_recording": np.broadcast_to(r * n_m + m, shape).copy(),
        "per_window": np.broadcast_to(w * n_m + m, shape).copy(),
    }


def group_fit(dev: np.ndarray, pid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(parameter values, prediction)`` of one indicator partition."""
    ok = np.isfinite(dev)
    n_p = int(pid.max()) + 1
    s = np.bincount(pid[ok], weights=dev[ok], minlength=n_p)
    c = np.bincount(pid[ok], minlength=n_p)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return g, np.where(ok, g[pid], np.nan)


def loo_window_fit(dev: np.ndarray, pid: np.ndarray) -> np.ndarray:
    """Prediction of each cell by a fit that never saw that cell's WINDOW.

    ``per_band`` has as many parameters as there are (mic, band) pairs, so its
    in-sample fit is flattered by exactly the noise the comparison is about.
    Held out one window at a time, every model is asked the only question that
    matters: does a gain measured elsewhere predict THIS window? ``per_window``
    cannot answer (its held-out parameter IS what was held out) and comes back
    all-NaN, which is itself the answer for that model.
    """
    ok = np.isfinite(dev)
    n_w = dev.shape[0]
    n_p = int(pid.max()) + 1
    wid = np.broadcast_to(np.arange(n_w)[:, None, None], dev.shape)
    key = (pid * n_w + wid)[ok]
    s = np.bincount(key, weights=dev[ok], minlength=n_p * n_w).reshape(n_p, n_w)
    c = np.bincount(key, minlength=n_p * n_w).reshape(n_p, n_w)
    num = s.sum(axis=1)[:, None] - s
    den = c.sum(axis=1)[:, None] - c
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    return np.where(ok, g[pid, wid], np.nan)


def rank_report(
    dev: np.ndarray, centres: np.ndarray, infos: Sequence[WindowInfo]
) -> dict[str, Any]:
    """Gains, variance explained and residual RMS for every model."""
    recs = list(dict.fromkeys(i.recording for i in infos))
    rec_id = np.array([recs.index(i.recording) for i in infos], dtype=np.int64)
    regimes = list(dict.fromkeys(i.regime for i in infos))
    reg_id = np.array([regimes.index(i.regime) for i in infos], dtype=np.int64)
    pids = model_pids(dev.shape, rec_id, reg_id)
    finite = np.isfinite(dev)
    ss_tot = float(np.nansum(np.where(finite, dev, 0.0) ** 2))
    out: dict[str, Any] = {
        "n_cells": int(finite.sum()),
        "n_windows": int(dev.shape[0]),
        "n_mics": int(dev.shape[1]),
        "n_bands": int(dev.shape[2]),
        "n_recordings": len(recs),
        "n_regimes": len(regimes),
        "regimes": regimes,
        "rms_db": _rms(dev),
        "models": {},
    }
    params: dict[str, np.ndarray] = {}
    for model in MODELS:
        g, pred = group_fit(dev, pids[model])
        params[model] = g
        res = dev - pred
        ss_res = float(np.nansum(np.where(np.isfinite(res), res, 0.0) ** 2))
        n_free = int(np.isfinite(g).sum())
        loo = np.where(finite, dev - loo_window_fit(dev, pids[model]), np.nan)
        ss_loo = float(np.nansum(np.where(np.isfinite(loo), loo, 0.0) ** 2))
        out["models"][model] = {
            "n_params": n_free,
            "variance_explained": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "residual_rms_db": _rms(res),
            "residual_rms_db_adj": float(np.sqrt(ss_res / max(int(finite.sum()) - n_free, 1))),
            "residual_rms_db_loo_window": _rms(loo),
            "variance_explained_loo_window": (
                (1.0 - ss_loo / ss_tot) if (ss_tot > 0 and np.isfinite(loo).any()) else float("nan")
            ),
            "residual_rms_db_by_band_group": {
                name: _rms(res[:, :, group_mask(centres, lo, hi)]) for name, lo, hi in BAND_GROUPS
            },
            "residual_rms_db_by_window": [_rms(res[w]) for w in range(res.shape[0])],
            "residual_rms_db_by_mic": [_rms(res[:, m, :]) for m in range(res.shape[1])],
        }
    n_m, n_b = dev.shape[1], dev.shape[2]
    g_const = params["const"]
    g_band = params["per_band"].reshape(n_m, n_b)
    g_win = params["per_window"].reshape(dev.shape[0], n_m)
    g_rec = params["per_recording"].reshape(len(recs), n_m)
    out["gains_db"] = [float(v) for v in g_const]
    out["gain_span_db"] = float(np.nanmax(g_const) - np.nanmin(g_const))
    out["gain_std_over_windows_db"] = [float(np.nanstd(g_win[:, m], ddof=1)) for m in range(n_m)]
    out["gain_std_over_bands_db"] = [float(np.nanstd(g_band[m], ddof=1)) for m in range(n_m)]
    out["gain_std_over_recordings_db"] = [
        float(np.nanstd(g_rec[:, m], ddof=1)) if len(recs) > 1 else float("nan") for m in range(n_m)
    ]
    out["gains_db_by_band"] = g_band
    out["gains_db_by_window"] = g_win
    out["gains_db_by_recording"] = g_rec
    out["recordings"] = recs
    out["window_names"] = [i.name for i in infos]
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    x, y = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    den = float(np.sqrt((x**2).sum() * (y**2).sum()))
    return float((x * y).sum() / den) if den > 0 else float("nan")


def ols(x: np.ndarray, y: np.ndarray, *, extra_params: int = 0) -> dict[str, float]:
    """Slope, intercept and slope standard error of ``y ~ x``.

    ``extra_params`` charges the fit for parameters removed BEFORE it ran (the
    per-group intercepts of :func:`within_group_slopes`), so the standard error
    is not read off degrees of freedom that were already spent.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) <= 0:
        return {"slope": float("nan"), "intercept": float("nan"), "stderr": float("nan")}
    xv, yv = x[ok], y[ok]
    xm, ym = xv.mean(), yv.mean()
    sxx = float(((xv - xm) ** 2).sum())
    slope = float(((xv - xm) * (yv - ym)).sum() / sxx)
    inter = float(ym - slope * xm)
    resid = yv - (inter + slope * xv)
    dof = max(int(ok.sum()) - 2 - int(extra_params), 1)
    return {
        "slope": slope,
        "intercept": inter,
        "stderr": float(np.sqrt(float((resid**2).sum()) / dof / sxx)),
    }


def _slope_block(fits: Sequence[dict[str, float]]) -> dict[str, Any]:
    """The per-mic slope table plus how many slopes clear 2 standard errors."""
    slope = np.array([f["slope"] for f in fits])
    stderr = np.array([f["stderr"] for f in fits])
    sig = np.isfinite(slope) & np.isfinite(stderr) & (np.abs(slope) > 2.0 * stderr)
    return {
        "slope": [f["slope"] for f in fits],
        "stderr": [f["stderr"] for f in fits],
        "rms_slope": _rms(slope),
        "max_abs_slope": float(np.nanmax(np.abs(slope)))
        if np.isfinite(slope).any()
        else float("nan"),
        "max_stderr": float(np.nanmax(stderr)) if np.isfinite(stderr).any() else float("nan"),
        # What the data can still hide: the largest slope consistent with any
        # channel at two standard errors. A "no speed dependence" reading is
        # only worth as much as this bound.
        "bound_db_per_oct": float(np.nanmax(np.abs(slope) + 2.0 * stderr))
        if np.isfinite(slope + stderr).any()
        else float("nan"),
        "n_significant": int(sig.sum()),
        "n_mics": int(slope.size),
    }


def wind_report(
    dev_floor: np.ndarray, centres: np.ndarray, infos: Sequence[WindowInfo]
) -> dict[str, Any]:
    """Is the per-mic floor excess low-frequency, and does it track speed?"""
    speed = np.array([i.carrier_rev_s for i in infos], dtype=np.float64)
    x = np.log2(np.maximum(speed, 1e-9))
    low = group_mask(centres, 0.0, WIND_LO_HZ)
    high = group_mask(centres, WIND_LO_HZ, 1e9)
    g_band = _nanmean(dev_floor, axis=0)  # (M, B)
    per_mic_low = _nanmean(g_band[:, low], axis=1)
    per_mic_high = _nanmean(g_band[:, high], axis=1)
    out: dict[str, Any] = {
        "speed_rev_s": {
            "min": float(speed.min()),
            "max": float(speed.max()),
            "mean": float(speed.mean()),
            "span_log2": float(np.log2(speed.max() / speed.min())),
        },
        "excess_db_below_500hz": [float(v) for v in per_mic_low],
        "excess_db_above_500hz": [float(v) for v in per_mic_high],
        "spread_over_mics_db": {
            "below_500hz": float(np.nanstd(per_mic_low, ddof=1)),
            "above_500hz": float(np.nanstd(per_mic_high, ddof=1)),
        },
        "excess_db_by_band_group": {
            name: [float(v) for v in _nanmean(g_band[:, group_mask(centres, lo, hi)], axis=1)]
            for name, lo, hi in BAND_GROUPS
        },
        "slopes_db_per_log2_speed": {},
    }
    groups = np.array([i.regime for i in infos])
    for name, sel in (("below_500hz", low), ("above_500hz", high), ("full", np.ones_like(low))):
        y = _nanmean(dev_floor[:, :, sel], axis=2)  # (W, M)
        fits = [ols(x, y[:, m]) for m in range(y.shape[1])]
        row = _slope_block(fits)
        # WITHIN-group slope. The bench pools four motors whose geometry (and
        # so whose per-mic deviation) is completely different, and pooling them
        # would read that geometry as a speed effect; demeaning inside each
        # group leaves the speed axis each group carries on its own.
        row["within_group"] = within_group_slopes(x, y, groups)
        out["slopes_db_per_log2_speed"][name] = row
    out["groups"] = {g: int((groups == g).sum()) for g in dict.fromkeys(groups.tolist())}
    return out


def within_group_slopes(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    """Per-mic slope of ``y ~ x`` with one free intercept per group."""
    xc = np.array(x, dtype=np.float64)
    yc = np.array(y, dtype=np.float64)
    spans, n_groups = [], 0
    for g in dict.fromkeys(groups.tolist()):
        m = groups == g
        if m.sum() < 2:
            xc[m], yc[m] = np.nan, np.nan
            continue
        n_groups += 1
        spans.append(float(np.ptp(x[m])))
        xc[m] -= x[m].mean()
        yc[m] -= _nanmean(y[m], axis=0)[None, :]
    # The group means are already spent: charge the fit for them.
    fits = [ols(xc, yc[:, m], extra_params=max(n_groups - 1, 0)) for m in range(yc.shape[1])]
    out = _slope_block(fits)
    out["n_groups"] = n_groups
    out["max_span_log2"] = max(spans) if spans else float("nan")
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _band_ticks(centres: np.ndarray) -> tuple[list[int], list[str]]:
    want = [31.5, 63, 125, 250, 500, 1000, 2000, 4000]
    pos, lab = [], []
    for w in want:
        i = int(np.argmin(np.abs(centres - w)))
        if abs(centres[i] - w) / w < 0.2 and i not in pos:
            pos.append(i)
            lab.append(f"{w / 1000:g}k" if w >= 1000 else f"{w:g}")
    return pos, lab


def heatmap_figure(rig: str, devs: dict[str, np.ndarray], centres: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(HEATMAP_VARIANTS), figsize=(15.0, 4.2), constrained_layout=True)
    mats = {v: _nanmean(devs[v], axis=0) for v in HEATMAP_VARIANTS}
    lim = max(
        float(np.nanpercentile(np.abs(np.concatenate([m.ravel() for m in mats.values()])), 99.0)),
        0.5,
    )
    pos, lab = _band_ticks(centres)
    im = None
    for ax, v in zip(axes, HEATMAP_VARIANTS):
        m = mats[v]
        im = ax.imshow(
            m, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest"
        )
        ax.set_title(f"{v}  (window mean)")
        ax.set_xticks(pos)
        ax.set_xticklabels(lab)
        ax.set_xlabel("1/3-octave band centre [Hz]")
        ax.set_yticks(range(m.shape[0]))
        ax.set_yticklabels([f"mic {i + 1}" for i in range(m.shape[0])])
        for b in range(m.shape[1]):
            if not np.isfinite(m[:, b]).any():
                ax.axvspan(b - 0.5, b + 0.5, color="0.85", zorder=3)
    assert im is not None
    fig.colorbar(im, ax=axes, label="level minus mic mean [dB]", shrink=0.9)
    fig.suptitle(f"{rig}: per-microphone level deviation (grey = every bin masked)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


@dataclass(frozen=True)
class SpeedPanel:
    """One panel of :func:`speed_figure`.

    ``demean`` marks a set whose groups are different GEOMETRIES (the bench's
    four motors): there the between-group offsets are position, not speed, and
    must come out before a speed slope means anything. A set whose groups are
    different OPERATING POINTS (Michael's standby against cruise) must NOT be
    demeaned -- the between-group contrast IS the speed axis.
    """

    name: str
    speed: np.ndarray  # (W,)
    excess: np.ndarray  # (W, M) low-band floor excess
    groups: np.ndarray  # (W,) the group each window belongs to
    demean: bool


def speed_figure(title: str, panels: Sequence[SpeedPanel], path: Path) -> None:
    """Per-mic low-band floor excess against rotor speed, one panel per set.

    A panel whose usable carrier span is under :data:`MIN_SPEED_SPAN_LOG2` gets
    no fitted line at all: a regression over a few per cent of speed is a line
    through noise, and drawing it would invite exactly the wrong reading.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    fig, axes = plt.subplots(
        1, len(panels), figsize=(6.0 * len(panels), 4.6), constrained_layout=True, squeeze=False
    )
    cmap = plt.get_cmap("tab10")
    for ax, panel in zip(axes[0], panels):
        x = np.log2(panel.speed)
        y = np.array(panel.excess, dtype=np.float64)
        names = list(dict.fromkeys(panel.groups.tolist()))
        spans = []
        if panel.demean:
            for g in names:
                sel = panel.groups == g
                y[sel] -= _nanmean(y[sel], axis=0)[None, :]
                spans.append(float(np.ptp(x[sel])))
        span = max(spans) if spans else float(np.ptp(x))
        fit_ok = span >= MIN_SPEED_SPAN_LOG2
        for m in range(y.shape[1]):
            ax.plot(panel.speed, y[:, m], "o", ms=4, color=cmap(m % 10), alpha=0.75)
            if not fit_ok:
                continue
            f = ols(x, y[:, m], extra_params=max(len(names) - 1, 0) if panel.demean else 0)
            xx = np.linspace(x.min(), x.max(), 16)
            ax.plot(
                2.0**xx,
                f["intercept"] + f["slope"] * xx,
                "-",
                lw=1.5,
                color=cmap(m % 10),
                label=f"mic {m + 1}: {f['slope']:+.2f} +-{f['stderr']:.2f} dB/oct",
            )
        ax.set_xscale("log" if span > 0.25 else "linear")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(ScalarFormatter())
        ax.set_xlabel("window rotor-mean carrier [rev/s]")
        ax.set_ylabel(
            f"floor excess below {WIND_LO_HZ:g} Hz [dB]"
            + (f"\n(demeaned within each of {len(names)} geometries)" if panel.demean else "")
        )
        ax.set_title(f"{panel.name}  ({span:.2f} octaves of usable span)")
        ax.grid(alpha=0.3)
        if fit_ok:
            ax.legend(fontsize=6, ncol=2, loc="best")
        else:
            ax.text(
                0.5,
                0.97,
                f"speed axis unidentified ({span:.2f} < {MIN_SPEED_SPAN_LOG2:g} octaves):\n"
                "no slope is drawn",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.6"},
            )
    fig.suptitle(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def band_figure(panels: Sequence[tuple[str, np.ndarray, np.ndarray]], path: Path) -> None:
    """Per-mic floor excess against band centre, one panel per set."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(panels), figsize=(5.6 * len(panels), 4.4), constrained_layout=True, squeeze=False
    )
    cmap = plt.get_cmap("tab10")
    for ax, (name, centres, g) in zip(axes[0], panels):
        for m in range(g.shape[0]):
            ax.plot(centres, g[m], "-o", ms=3, lw=1.3, color=cmap(m % 10), label=f"mic {m + 1}")
        ax.axvline(WIND_LO_HZ, color="0.4", ls="--", lw=1.0)
        ax.set_xscale("log")
        ax.set_xlabel("1/3-octave band centre [Hz]")
        ax.set_ylabel("floor level minus mic mean [dB]")
        ax.set_title(name)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle(f"per-microphone FLOOR excess vs band (dashed: {WIND_LO_HZ:g} Hz)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _clean(obj: Any, sig: int = 6) -> Any:
    """JSON-safe: arrays to lists, non-finite floats to ``None``."""
    if isinstance(obj, dict):
        return {k: _clean(v, sig) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_clean(v, sig) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist(), sig)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return None if not np.isfinite(v) else float(f"{v:.{sig}g}")
    if isinstance(obj, (int, np.integer, bool, np.bool_)):
        return obj.item() if isinstance(obj, (np.integer, np.bool_)) else obj
    return obj


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _row(name: str, values: Sequence[float | None], fmt: str = "{:+.2f}") -> str:
    """One markdown row; ``None`` (a non-finite number in the JSON) reads `n/a`."""
    cells = " | ".join("n/a" if v is None or not np.isfinite(v) else fmt.format(v) for v in values)
    return f"| {name} | {cells} |"


def summary_section(rigs: dict[str, Any]) -> str:
    """The one table that answers all three questions across the sets."""
    out = ["## Summary", ""]
    out.append(
        "| set | windows | floor `const` var. expl. | floor `const` residual dB | "
        "floor `per_band` residual dB | comb-vs-floor r | excess <500 Hz spread dB | "
        "excess >500 Hz spread dB | speed span oct | max speed slope dB/oct |"
    )
    out.append("| --- |" + " --- |" * 9)
    for rig, blk in rigs.items():
        f = blk["rank"]["floor"]["models"]
        w = blk["wind"]
        sl = w["slopes_db_per_log2_speed"]["below_500hz"]
        if rig == "dregon_bench":
            sl, span = sl["within_group"], sl["within_group"]["max_span_log2"]
        else:
            span = w["speed_rev_s"]["span_log2"]
        out.append(
            f"| {rig} | {blk['n_windows']} | "
            f"{100 * f['const']['variance_explained']:.1f}% | "
            f"{f['const']['residual_rms_db']:.2f} | {f['per_band']['residual_rms_db']:.2f} | "
            f"{blk['comb_vs_floor']['gain_corr']:.3f} | "
            f"{w['spread_over_mics_db']['below_500hz']:.2f} | "
            f"{w['spread_over_mics_db']['above_500hz']:.2f} | {span:.2f} | "
            + (
                f"{sl['max_abs_slope']:.2f} ({sl['n_significant']}/{sl['n_mics']} sig.) |"
                if span >= MIN_SPEED_SPAN_LOG2
                else "unidentified |"
            )
        )
    out.append("")
    return "\n".join(out)


def findings_md(payload: dict[str, Any]) -> str:
    """The report. Every number is read back out of the JSON payload."""
    rigs = payload["rigs"]
    out: list[str] = []
    a = out.append
    a("# Per-microphone level deviations on the real windows: one gain, or not?")
    a("")
    a(f"`{payload['generated_by']}` at `{payload['git_head'][:12]}`.")
    a("")
    a(
        "Measurement only -- no model is evaluated and nothing is fitted to a "
        "likelihood. `L[window, mic, band]` is the 1/3-octave level "
        f"({payload['config']['band_lo_hz']:g}-{payload['config']['band_hi_hz']:g} Hz) of every "
        "real window of each rig, in four variants: `full`, `floor` (every bin within "
        f"+-{payload['config']['comb_mask_half_hz']:g} Hz -- one bin on the flight front end -- "
        "of a rotor order at the window's own per-frame carriers removed) and `comb` (the "
        "order-tracked line peaks, power-averaged over the orders inside a band) and "
        f"`floor_q25` (the {FLOOR_QUANTILE:g} quantile of the same unmasked cells, a floor a "
        "stray tonal cannot lift). "
        "`D = L - mean_over_mics(L)` is the per-mic deviation, and the models fitted to it are "
        "the indicator partitions `const` (`g[mic]`), `per_band` (`g[mic, band]`), `per_regime` "
        "(`g[mic, regime]`; on the bench a regime is one MOTOR), `per_recording` "
        "(`g[mic, recording]`) and `per_window` (`g[mic, window]`). Each is fitted by least "
        "squares, which for an indicator partition is the group mean."
    )
    a("")
    a("## Windows")
    a("")
    a("| set | windows | mics | duration | carrier rev/s (min-max of window means) | recordings |")
    a("| --- | --- | --- | --- | --- | --- |")
    for rig, blk in rigs.items():
        by_tag: dict[str, list[dict]] = {}
        for w in blk["windows"]:
            by_tag.setdefault(w["tag"], []).append(w)
        for tag, ws in by_tag.items():
            car = [w["carrier_rev_s"] for w in ws]
            dur = sorted({round(w["duration_s"], 2) for w in ws})
            recs = sorted({w["recording"] for w in ws})
            rec_txt = f"{len(recs)} ({', '.join(recs)})" if len(recs) <= 3 else f"{len(recs)}"
            a(
                f"| {rig}/{tag} | {len(ws)} | {blk['n_mics']} | "
                f"{'/'.join(f'{d:g}' for d in dur)} s | {min(car):.1f}-{max(car):.1f} | {rec_txt} |"
            )
    a("")
    a(
        f"Band grid: {rigs['dregon']['n_bands']} kept 1/3-octave bands on the flight front end "
        f"(df = {rigs['dregon']['df_hz']:.4g} Hz). Bands whose every bin is occupied by an order "
        "in every frame have no floor and are reported as `n/a`."
    )
    a("")
    a(
        "`dregon` and `michaels` are the two REAL RIGS -- the flight windows of a complete "
        "eight-microphone array on a flying drone. `dregon_bench` is AUXILIARY: one motor at a "
        "time on a bench, four different source positions pooled, present only because it is the "
        "only DREGON material with a real rotor-speed span. Its rank test is reported for "
        "completeness and answers a different question (source geometry), not `can the channel "
        "gains be normalised once`."
    )
    a("")
    a(summary_section(rigs))

    for rig, blk in rigs.items():
        a(f"## {rig}")
        a("")
        a("### Rank test")
        a("")
        a(
            "| variant | model | params | var. explained | residual RMS dB | held-out RMS dB | "
            "30-300 | 300-1k | 1k-3k | 3k-8k |"
        )
        a("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for v in VARIANTS:
            r = blk["rank"][v]
            for model in MODELS:
                m = r["models"][model]
                g = m["residual_rms_db_by_band_group"]
                ve = m["variance_explained"]
                loo = m["residual_rms_db_loo_window"]
                cells = " | ".join(
                    "n/a" if g[k] is None else f"{g[k]:.2f}" for k, _, _ in BAND_GROUPS
                )
                a(
                    f"| {v} | {model} | {m['n_params']} | "
                    f"{'n/a' if ve is None else f'{100 * ve:.1f}%'} | "
                    f"{m['residual_rms_db']:.2f} | "
                    f"{'n/a' if loo is None else f'{loo:.2f}'} | {cells} |"
                )
        a("")
        a(
            "`held-out RMS` is leave-one-window-out: the model is fitted on the other windows and "
            "asked to predict the held-out one. `per_window` has no out-of-sample prediction by "
            "construction, hence `n/a`."
        )
        a("")
        rms_txt = ", ".join(f"{v} {blk['rank'][v]['rms_db']:.2f} dB" for v in VARIANTS)
        a(f"Total deviation RMS (the thing being explained): {rms_txt}.")
        a("")
        a("### The eight gains (`const` model), dB relative to the mic mean")
        a("")
        n_m = blk["n_mics"]
        a("| quantity | " + " | ".join(f"mic {i + 1}" for i in range(n_m)) + " |")
        a("| --- |" + " --- |" * n_m)
        for v in VARIANTS:
            a(_row(f"{v} gain", blk["rank"][v]["gains_db"]))
        for v in VARIANTS:
            a(_row(f"{v} sd over bands", blk["rank"][v]["gain_std_over_bands_db"], "{:.2f}"))
        for v in VARIANTS:
            a(_row(f"{v} sd over windows", blk["rank"][v]["gain_std_over_windows_db"], "{:.2f}"))
        for v in VARIANTS:
            a(
                _row(
                    f"{v} sd over recordings",
                    blk["rank"][v]["gain_std_over_recordings_db"],
                    "{:.2f}",
                )
            )
        a(_row("comb minus floor", blk["comb_vs_floor"]["diff_db"]))
        a("")
        cf = blk["comb_vs_floor"]
        a(
            f"comb-vs-floor gain correlation over the {n_m} channels: "
            f"r = {cf['gain_corr']:.3f}; RMS difference {cf['rms_diff_db']:.2f} dB; "
            f"largest |difference| {cf['max_abs_diff_db']:.2f} dB."
        )
        a("")
        gap = blk["comb_minus_floor_db_by_band_group"]
        a(
            "Contrast the comb/floor split actually has (mic- and window-mean `comb - floor`): "
            + ", ".join(
                f"{k} {'n/a' if gap[k] is None else f'{gap[k]:+.2f}'} dB" for k, _, _ in BAND_GROUPS
            )
            + f". The order mask covers {100 * blk['comb_masked_frac_mean']:.0f}% of the "
            "(frame, bin) cells on average, so where the contrast is small the two variants are "
            "largely reading the same spectrum."
        )
        a("")
        fq = blk["floor_vs_q25"]
        a(
            f"Robustness of the floor: the `floor_q25` gains (a {FLOOR_QUANTILE:g} quantile of the "
            "SAME unmasked cells instead of their power mean) correlate with the `floor` gains at "
            f"r = {fq['gain_corr']:.3f}, RMS difference {fq['rms_diff_db']:.2f} dB, largest "
            f"{fq['max_abs_diff_db']:.2f} dB; its low-band spread over channels is "
            f"{blk['wind_q25']['spread_over_mics_db']['below_500hz']:.2f} dB against "
            f"{blk['wind_q25']['spread_over_mics_db']['above_500hz']:.2f} dB above 500 Hz "
            f"(power-mean floor: {blk['wind']['spread_over_mics_db']['below_500hz']:.2f} / "
            f"{blk['wind']['spread_over_mics_db']['above_500hz']:.2f} dB)."
        )
        a("")
        a("### Residual RMS of the `const` model per window (floor variant), dB")
        a("")
        rows = sorted(
            zip(
                blk["rank"]["floor"]["window_names"],
                blk["rank"]["floor"]["models"]["const"]["residual_rms_db_by_window"],
            ),
            key=lambda kv: -(kv[1] or 0.0),
        )
        a("| window | residual RMS dB |")
        a("| --- | --- |")
        for name, val in rows[:8]:
            a(f"| {name} | {val:.2f} |")
        if len(rows) > 8:
            worst = rows[8][1]
            a(f"| ... {len(rows) - 8} more | <= {worst:.2f} |")
        a("")
        a("### Wind test (floor excess)")
        a("")
        w = blk["wind"]
        s = w["speed_rev_s"]
        a(
            f"Window rotor-mean carriers span {s['min']:.1f}-{s['max']:.1f} rev/s "
            f"({s['span_log2']:.2f} octaves)."
        )
        if s["span_log2"] < MIN_SPEED_SPAN_LOG2:
            a(
                f"That is under the {MIN_SPEED_SPAN_LOG2:g}-octave minimum this report demands of "
                "a speed axis, so the `slope` rows below are UNIDENTIFIED -- they are tabled only "
                "to show that their standard errors are of the same size."
            )
        a("")
        a("| quantity | " + " | ".join(f"mic {i + 1}" for i in range(n_m)) + " |")
        a("| --- |" + " --- |" * n_m)
        for name, _, _ in BAND_GROUPS:
            a(_row(f"excess {name} Hz", w["excess_db_by_band_group"][name]))
        a(_row("excess < 500 Hz", w["excess_db_below_500hz"]))
        a(_row("excess > 500 Hz", w["excess_db_above_500hz"]))
        a(_row("slope <500 Hz dB/oct", w["slopes_db_per_log2_speed"]["below_500hz"]["slope"]))
        a(_row("slope stderr", w["slopes_db_per_log2_speed"]["below_500hz"]["stderr"], "{:.2f}"))
        a(_row("slope full band dB/oct", w["slopes_db_per_log2_speed"]["full"]["slope"]))
        wg = w["slopes_db_per_log2_speed"]["below_500hz"]["within_group"]
        if wg["max_span_log2"] is not None and wg["max_span_log2"] >= MIN_SPEED_SPAN_LOG2:
            a(_row("slope <500 Hz within-group", wg["slope"]))
            a(_row("within-group stderr", wg["stderr"], "{:.2f}"))
        a("")
        groups = ", ".join(f"{k} x{v}" for k, v in w["groups"].items())
        span_txt = "n/a" if wg["max_span_log2"] is None else f"{wg['max_span_log2']:.2f} octaves"
        a(
            f"Spread over channels: {w['spread_over_mics_db']['below_500hz']:.2f} dB below "
            f"500 Hz against {w['spread_over_mics_db']['above_500hz']:.2f} dB above it. "
            f"Groups (one free intercept each in the within-group fit): {groups}; largest "
            f"within-group speed span {span_txt}"
            + (
                "."
                if (wg["max_span_log2"] or 0.0) >= MIN_SPEED_SPAN_LOG2
                else f", below the {MIN_SPEED_SPAN_LOG2:g}-octave minimum, so the within-group "
                "slope is unidentified and not tabled."
            )
        )
        a("")
        a("### Verdict")
        a("")
        for line in blk["verdict"]:
            a(f"- {line}")
        a("")
    a("## Figures")
    a("")
    for name in payload["figures"]:
        a(f"- `{name}`")
    a("")
    return "\n".join(out)


def verdicts(rig: str, blk: dict[str, Any]) -> list[str]:
    """Three verdicts per rig, every number of them read off ``blk``."""
    out = []
    for v in ("full", "floor"):
        r = blk["rank"][v]
        c, pb, pq, pr, pw = (r["models"][m] for m in MODELS)
        ok = c["variance_explained"] >= 0.90 and c["residual_rms_db"] <= 1.0
        generalises = c["residual_rms_db_loo_window"] <= pb["residual_rms_db_loo_window"]
        out.append(
            f"(i) {v}: the rank-one `g[mic]` model explains "
            f"{100 * c['variance_explained']:.1f}% of the deviation with a "
            f"{c['residual_rms_db']:.2f} dB residual ({c['residual_rms_db_loo_window']:.2f} dB "
            f"held out), against {100 * pb['variance_explained']:.1f}% / "
            f"{pb['residual_rms_db']:.2f} dB ({pb['residual_rms_db_loo_window']:.2f} dB held out) "
            f"for `g[mic, band]`, {100 * pq['variance_explained']:.1f}% / "
            f"{pq['residual_rms_db']:.2f} dB for `g[mic, regime]`, "
            f"{100 * pr['variance_explained']:.1f}% / "
            f"{pr['residual_rms_db']:.2f} dB for `g[mic, recording]` and "
            f"{100 * pw['variance_explained']:.1f}% / "
            f"{pw['residual_rms_db']:.2f} dB for `g[mic, window]` -- "
            + (
                "one constant per channel IS enough, so the channel gains can be normalised once "
                "in the pipeline."
                if ok
                else "one constant per channel is NOT enough; the deviation is "
                + (
                    "frequency-dependent (a per-channel transfer function, not a gain)"
                    if pb["residual_rms_db"] < pw["residual_rms_db"]
                    else "time-varying (not a property of the channel)"
                )
                + (
                    ", and the richer per-band description also predicts a held-out window better."
                    if not generalises
                    else ", though the per-band description does not survive holding a window out."
                )
            )
        )
    cf = blk["comb_vs_floor"]
    gap = blk["comb_minus_floor_db_by_band_group"]
    out.append(
        f"(ii) comb and floor gains correlate at r = {cf['gain_corr']:.3f} with an RMS "
        f"difference of {cf['rms_diff_db']:.2f} dB (largest {cf['max_abs_diff_db']:.2f} dB) -- "
        + (
            "the same per-channel sensitivity moves the lines and the floor, as a gain must"
            if cf["gain_corr"] >= 0.9 and cf["rms_diff_db"] <= 1.0
            else "the lines and the floor do NOT see the same per-channel gain"
        )
        + f"; the split has {gap['30-300']:+.1f} dB of comb-over-floor contrast at 30-300 Hz and "
        f"{gap['3k-8k']:+.1f} dB at 3-8 kHz, with "
        f"{100 * blk['comb_masked_frac_mean']:.0f}% of the cells masked."
    )
    w = blk["wind"]
    lo, hi = w["spread_over_mics_db"]["below_500hz"], w["spread_over_mics_db"]["above_500hz"]
    sl = w["slopes_db_per_log2_speed"]["below_500hz"]
    # A set that pools several geometries (the bench: four motors) reads its
    # speed slope WITHIN each geometry; a single-geometry set has one group, and
    # then the two are the same number.
    if len(w["groups"]) > 1 and rig == "dregon_bench":
        sl = sl["within_group"]
        span = sl["max_span_log2"]
        kind = "within-motor"
    else:
        span = w["speed_rev_s"]["span_log2"]
        kind = "pooled"
    q = blk["wind_q25"]["spread_over_mics_db"]
    shape = (
        f"the floor excess spreads {lo:.2f} dB over channels below 500 Hz against "
        f"{hi:.2f} dB above it "
        f"({q['below_500hz']:.2f} / {q['above_500hz']:.2f} dB on the quantile floor), "
        + (
            "which IS the low-frequency concentration wind/self-noise has"
            if lo > 1.5 * hi
            else "i.e. it is NOT concentrated at low frequency"
        )
        + "; "
    )
    if span < MIN_SPEED_SPAN_LOG2:
        speed = (
            f"the speed axis is NOT identified on these windows ({span:.2f} octaves of carrier "
            f"span, {kind} slopes up to {sl['max_abs_slope']:.2f} dB/oct at standard errors up to "
            f"{sl['max_stderr']:.2f} dB/oct), so the speed question is left to the sets that do "
            "have a span."
        )
    else:
        strong = sl["max_abs_slope"] >= SPEED_SLOPE_DB and sl["n_significant"] > 0
        speed = (
            f"its {kind} speed slope reaches {sl['max_abs_slope']:.2f} dB per octave "
            f"(RMS {sl['rms_slope']:.2f}, largest standard error {sl['max_stderr']:.2f}) over a "
            f"{span:.2f}-octave span, with {sl['n_significant']}/{sl['n_mics']} channels beyond "
            "two standard errors, so the excess "
            + (
                f"DOES track rotor speed, though weakly (at most {sl['max_abs_slope']:.2f} dB/oct, "
                f"{sl['max_abs_slope'] * span:.1f} dB over the whole span)"
                if strong
                else "is consistent with NO speed dependence (the span bounds every channel's "
                f"slope at {sl['bound_db_per_oct']:.2f} dB/oct)"
            )
            + (
                " -- wind/self-noise-like."
                if (lo > 1.5 * hi and not strong)
                else " -- not the wind/self-noise signature."
            )
        )
    out.append("(iii) " + shape + speed)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def analyse(rig: str, data: dict) -> dict[str, Any]:
    centres = data["centres"]
    infos: list[WindowInfo] = data["infos"]
    devs = {v: deviations(data["levels"][v]) for v in VARIANTS}
    blk: dict[str, Any] = {
        "n_windows": len(infos),
        "n_mics": data["n_mics"],
        "n_bands": int(centres.size),
        "df_hz": float(16000.0 / SUP.OBS_N_FFT) if rig != "dregon_bench" else None,
        "band_centres_hz": centres,
        "windows": [asdict(i) for i in infos],
        "levels_db": data["levels"],
        "n_bands_without_floor": int(
            (~np.isfinite(data["levels"]["floor"])).all(axis=(0, 1)).sum()
        ),
        "orders_per_band": data["orders_per_band"].mean(axis=0),
        "rank": {v: rank_report(devs[v], centres, infos) for v in VARIANTS},
    }
    gf = np.asarray(blk["rank"]["floor"]["gains_db"])
    gc = np.asarray(blk["rank"]["comb"]["gains_db"])
    blk["comb_vs_floor"] = {
        "gain_corr": _pearson(gc, gf),
        "gains_floor_db": gf,
        "gains_comb_db": gc,
        "diff_db": gc - gf,
        "rms_diff_db": _rms(gc - gf),
        "max_abs_diff_db": float(np.nanmax(np.abs(gc - gf))),
    }
    gq = np.asarray(blk["rank"]["floor_q25"]["gains_db"])
    blk["floor_vs_q25"] = {
        "gain_corr": _pearson(gq, gf),
        "gains_q25_db": gq,
        "rms_diff_db": _rms(gq - gf),
        "max_abs_diff_db": float(np.nanmax(np.abs(gq - gf))),
    }
    # How much CONTRAST the comb/floor split actually has, per band group: the
    # mic- and window-mean of (comb - floor). Where this collapses towards zero
    # the two variants are reading the same cells and their gains cannot
    # disagree meaningfully.
    comb_gap = _nanmean(data["levels"]["comb"] - data["levels"]["floor"], axis=(0, 1))
    blk["comb_minus_floor_db_by_band_group"] = {
        name: float(_nanmean(comb_gap[group_mask(centres, lo, hi)], axis=0))
        for name, lo, hi in BAND_GROUPS
    }
    blk["comb_masked_frac_mean"] = float(np.mean([i.comb_masked_frac for i in infos]))
    blk["wind"] = wind_report(devs["floor"], centres, infos)
    blk["wind_q25"] = wind_report(devs["floor_q25"], centres, infos)
    blk["verdict"] = verdicts(rig, blk)
    return blk


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument(
        "--no-extra",
        action="store_true",
        help="skip the extra DREGON room-1/room-2 flight windows",
    )
    ap.add_argument("--no-bench", action="store_true", help="skip the DREGON bench speed axis")
    ap.add_argument("--limit", type=int, default=None, help="windows per set (smoke runs)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = rig_specs(extra=not args.no_extra)
    if not args.no_bench:
        specs["dregon_bench"] = bench_specs()

    rigs: dict[str, Any] = {}
    raw: dict[str, dict] = {}
    for rig, rows in specs.items():
        print(
            f"[{rig}] measuring {len(rows) if args.limit is None else min(len(rows), args.limit)} windows ...",
            flush=True,
        )
        raw[rig] = collect(rows, limit=args.limit)
        rigs[rig] = analyse(rig, raw[rig])
        r = rigs[rig]["rank"]["floor"]["models"]["const"]
        print(
            f"[{rig}] floor/const: {100 * r['variance_explained']:.1f}% explained, "
            f"residual {r['residual_rms_db']:.2f} dB",
            flush=True,
        )

    figures: list[str] = []
    for rig in rigs:
        name = f"deviation_heatmap_{rig}.png"
        heatmap_figure(
            rig,
            {v: deviations(raw[rig]["levels"][v]) for v in VARIANTS},
            raw[rig]["centres"],
            out_dir / name,
        )
        figures.append(name)

    def _panel(rig: str) -> SpeedPanel:
        dev = deviations(raw[rig]["levels"]["floor"])
        sel = group_mask(raw[rig]["centres"], 0.0, WIND_LO_HZ)
        infos: list[WindowInfo] = raw[rig]["infos"]
        return SpeedPanel(
            name=rig,
            speed=np.array([i.carrier_rev_s for i in infos]),
            excess=_nanmean(dev[:, :, sel], axis=2),
            groups=np.array([i.regime for i in infos]),
            demean=rig == "dregon_bench",
        )

    dregon_panels = [_panel("dregon")]
    if "dregon_bench" in rigs:
        dregon_panels.append(_panel("dregon_bench"))
    speed_figure(
        f"DREGON: per-mic FLOOR excess below {WIND_LO_HZ:g} Hz vs rotor speed",
        dregon_panels,
        out_dir / "floor_excess_vs_speed_dregon.png",
    )
    figures.append("floor_excess_vs_speed_dregon.png")
    speed_figure(
        f"Michael's (control): per-mic FLOOR excess below {WIND_LO_HZ:g} Hz vs rotor speed",
        [_panel("michaels")],
        out_dir / "floor_excess_vs_speed_michaels.png",
    )
    figures.append("floor_excess_vs_speed_michaels.png")
    band_figure(
        [
            (rig, raw[rig]["centres"], _nanmean(deviations(raw[rig]["levels"]["floor"]), axis=0))
            for rig in rigs
        ],
        out_dir / "floor_excess_vs_band.png",
    )
    figures.append("floor_excess_vs_band.png")

    payload = {
        "schema": "mic-gain-rank/1",
        "generated_by": "scripts/_mic_gain_rank.py",
        "git_head": git_head(),
        "config": {
            "band_lo_hz": BAND_LO_HZ,
            "band_hi_hz": BAND_HI_HZ,
            "band_groups": [list(g) for g in BAND_GROUPS],
            "wind_lo_hz": WIND_LO_HZ,
            "comb_mask_half_hz": COMB_MASK_HALF_HZ,
            "dregon_extra_min_rps": DREGON_EXTRA_MIN_RPS,
            "dregon_extra_dur_s": DREGON_EXTRA_DUR_S,
            "dregon_flight_recordings": list(DREGON_FLIGHT_RECORDINGS),
            "real_window_sets": {k: list(v) for k, v in REAL_WINDOW_SETS.items()},
            "variants": list(VARIANTS),
            "models": list(MODELS),
        },
        "figures": figures,
        "rigs": rigs,
    }
    clean = _clean(payload)
    (out_dir / "mic_gains.json").write_text(json.dumps(clean, indent=1) + "\n")
    (out_dir / "findings.md").write_text(findings_md(clean))
    print(f"wrote {out_dir}/mic_gains.json and findings.md", flush=True)
    for rig, blk in clean["rigs"].items():
        print(f"\n== {rig} ==")
        for line in blk["verdict"]:
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
